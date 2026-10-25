#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <planetmodel/json.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "Commands.hpp"
#include "Common.hpp"
#include "Input.hpp"
#include "Serialize.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/ModelViscosity.hpp"

namespace sl3dpp::app {

namespace {

/// One node of the accumulated history, SI, kept across restarts.
struct Node {
  double date_kyr_bp = 0.0;
  std::vector<double> seaLevelChange;
  std::vector<double> displacement;
  std::vector<double> potential;
  std::array<double, 3> angularVelocity{};
  std::vector<double> oceanFunction;  ///< migrating runs alone
};

/// A run stopped on purpose at a checkpoint: the wall clock passed the
/// configured limit after the checkpoint was written.
struct WalltimeReached {
  std::size_t node = 0;
};

// -- the checkpoint files: private scratch of a run, machine-local ----------

constexpr auto kMagic = std::uint64_t{0x534c3344434b5031};  // "SL3DCKP1"
constexpr auto kKind1D = std::uint64_t{1};
constexpr auto kKind3D = std::uint64_t{2};

constexpr std::uint64_t KindOf(const SeaLevelEvolution::Checkpoint&) {
  return kKind1D;
}
constexpr std::uint64_t KindOf(const SeaLevelEvolution3D::Checkpoint&) {
  return kKind3D;
}

/// FNV-1a, folded over whatever is added to it.
class Fnv1a {
 public:
  void Add(std::span<const std::byte> bytes) {
    for (const auto b : bytes) {
      hash_ ^= static_cast<std::uint64_t>(b);
      hash_ *= 1099511628211ull;
    }
  }
  template <class T>
  void AddValue(const T& value) {
    Add(std::as_bytes(std::span(&value, 1)));
  }
  /// A file's contents, read in blocks.
  void AddFile(const std::filesystem::path& path) {
    auto in = std::ifstream(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error(std::format("cannot read {}", path.string()));
    }
    auto block = std::vector<char>(std::size_t{1} << 20);
    while (in) {
      in.read(block.data(), static_cast<std::streamsize>(block.size()));
      Add(std::as_bytes(
          std::span(block.data(), static_cast<std::size_t>(in.gcount()))));
    }
  }
  [[nodiscard]] std::uint64_t Value() const { return hash_; }

 private:
  std::uint64_t hash_ = 14695981039346656037ull;
};

/// What a run's checkpoints belong to: the members that decide the
/// computation, and the contents of its input files, so that a restart
/// with another run's checkpoints -- another config, or the same config
/// over changed inputs -- is refused. The members that only steer the run
/// (the checkpoint stride and directory, the wall-time limit, restart, the
/// output) are left out, and so are the paths themselves, so that these can
/// change between a run and its resumption and a run directory can move.
std::uint64_t FingerprintOf(const EvolveInput& input) {
  auto hash = Fnv1a{};
  hash.AddFile(input.model);
  hash.AddValue(input.format);
  hash.AddValue(input.lmax);
  if (!input.viscosity.empty()) hash.AddFile(input.viscosity);
  hash.AddValue(input.viscosity_file_radial);
  hash.AddFile(input.state);
  hash.AddFile(input.ice_history);
  hash.AddValue(input.start_kyr_bp);
  hash.AddValue(input.end_kyr_bp);
  hash.AddValue(input.nodes);
  hash.AddValue(input.scheme);
  hash.AddValue(input.rotation);
  hash.AddValue(input.coupled_rtol);
  hash.AddValue(input.max_coupled_iterations);
  hash.AddValue(input.hydrostatic_moments);
  hash.AddValue(input.rtol);
  // added only when set, so a frozen, uncalibrated run's checkpoints
  // written before these members existed still resume
  if (input.shorelines != Shorelines::Frozen) hash.AddValue(input.shorelines);
  if (input.calibrate) {
    hash.AddValue(input.calibration_method);
    hash.AddValue(input.calibration_iterations);
    hash.AddValue(input.calibration_rtol);
  }
  return hash.Value();
}

/// The calibrated initial state a calibrating run keeps beside its
/// checkpoints, so that a resumption need not calibrate again.
constexpr auto kInitialStateName = std::string_view{"initial_state.bin"};
constexpr auto kInitialStateMagic =
    std::uint64_t{0x534c334449535431};  // "SL3DIST1"

void WriteInitialState(const std::filesystem::path& directory,
                       std::uint64_t hash, const SeaLevelSolver::State& state) {
  std::filesystem::create_directories(directory);
  const auto path = directory / kInitialStateName;
  const auto partial = directory / ".initial_state.partial";
  {
    auto out = std::ofstream(partial, std::ios::binary);
    PutOne(out, kInitialStateMagic);
    PutOne(out, hash);
    PutVector(out, state.iceThickness);
    PutVector(out, state.seaLevel);
    PutVector(out, state.oceanFunction);
    PutOne(out, state.oceanArea);
    if (!out) {
      throw std::runtime_error(
          std::format("cannot write {}", partial.string()));
    }
  }
  std::filesystem::rename(partial, path);
}

/// The calibrated initial state of this run, if the directory keeps one.
std::optional<SeaLevelSolver::State> ReadInitialState(
    const std::filesystem::path& directory, std::uint64_t hash,
    const SeaLevelSolver::State& background) {
  auto in = std::ifstream(directory / kInitialStateName, std::ios::binary);
  if (!in || GetOne<std::uint64_t>(in) != kInitialStateMagic ||
      GetOne<std::uint64_t>(in) != hash) {
    return std::nullopt;
  }
  auto state = background;
  state.iceThickness = GetVector(in);
  state.seaLevel = GetVector(in);
  state.oceanFunction = GetVector(in);
  state.oceanArea = GetOne<double>(in);
  if (!in || state.seaLevel.size() != background.seaLevel.size()) {
    return std::nullopt;
  }
  return state;
}

/// Whether a file name is one this program writes into a checkpoint
/// directory (node_NNNNNN.ckpt, or the .node_NNNNNN.partial it renames):
/// only those are read, pruned or cleared, whatever else is there.
bool IsCheckpointName(std::string_view name) {
  const auto digits = [](std::string_view text) {
    return text.size() == 6 &&
           text.find_first_not_of("0123456789") == std::string_view::npos;
  };
  for (const auto& [prefix, suffix] :
       {std::pair{std::string_view{"node_"}, std::string_view{".ckpt"}},
        std::pair{std::string_view{".node_"}, std::string_view{".partial"}}}) {
    if (name.starts_with(prefix) && name.ends_with(suffix) &&
        name.size() == prefix.size() + 6 + suffix.size() &&
        digits(name.substr(prefix.size(), 6))) {
      return true;
    }
  }
  return false;
}

/// The finished checkpoints in a directory, oldest (lowest node) first.
std::vector<std::filesystem::path> CheckpointsIn(
    const std::filesystem::path& directory) {
  auto found = std::vector<std::filesystem::path>{};
  if (!std::filesystem::is_directory(directory)) return found;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    const auto name = entry.path().filename().string();
    if (IsCheckpointName(name) && name.ends_with(".ckpt")) {
      found.push_back(entry.path());
    }
  }
  std::sort(found.begin(), found.end());
  return found;
}

/// Removes the checkpoint files an earlier run left in the directory, for
/// a run that starts over: its own checkpoints would otherwise be pruned
/// against the earlier run's by name, and a later resumption could pick
/// one of those. Returns how many were removed.
std::size_t ClearCheckpoints(const std::filesystem::path& directory) {
  auto removed = std::size_t{0};
  if (!std::filesystem::is_directory(directory)) return removed;
  auto doomed = std::vector<std::filesystem::path>{};
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (IsCheckpointName(entry.path().filename().string())) {
      doomed.push_back(entry.path());
    }
  }
  for (const auto& path : doomed) {
    if (std::filesystem::remove(path) && path.extension() == ".ckpt") {
      ++removed;
    }
  }
  std::filesystem::remove(directory / kInitialStateName);
  return removed;
}

/// Writes a checkpoint whole (a temporary renamed into place) and keeps
/// the newest two in the directory.
template <class CheckpointT>
void WriteCheckpoint(const std::filesystem::path& directory, std::uint64_t hash,
                     std::size_t node, const CheckpointT& cp,
                     const std::vector<Node>& history, bool migrating) {
  std::filesystem::create_directories(directory);
  const auto path = directory / std::format("node_{:06}.ckpt", node);
  const auto partial = directory / std::format(".node_{:06}.partial", node);
  {
    auto out = std::ofstream(partial, std::ios::binary);
    if (!out) {
      throw std::runtime_error(
          std::format("cannot write {}", partial.string()));
    }
    PutOne(out, kMagic);
    PutOne(out, hash);
    PutOne(out, KindOf(cp));
    PutOne(out, static_cast<std::uint64_t>(node));
    PutOne(out, cp.t);
    PutState(out, cp);
    PutComplex(out, cp.loadCoefficients);
    PutComplex(out, cp.tideCoefficients);
    PutOne(out, static_cast<std::uint64_t>(history.size()));
    for (const auto& kept : history) {
      PutOne(out, kept.date_kyr_bp);
      PutVector(out, kept.seaLevelChange);
      PutVector(out, kept.displacement);
      PutVector(out, kept.potential);
      Put(out, kept.angularVelocity.data(), 3 * sizeof(double));
      if (migrating) PutVector(out, kept.oceanFunction);
    }
    if (!out) {
      throw std::runtime_error(
          std::format("cannot write {}", partial.string()));
    }
  }
  std::filesystem::rename(partial, path);
  // keep the newest two
  auto kept = CheckpointsIn(directory);
  while (kept.size() > 2) {
    std::filesystem::remove(kept.front());
    kept.erase(kept.begin());
  }
}

template <class CheckpointT>
struct ReadBack {
  std::size_t node = 0;
  CheckpointT checkpoint;
  std::vector<Node> history;
};

/// The newest checkpoint of this run in the directory, or nothing.
template <class CheckpointT>
std::optional<ReadBack<CheckpointT>> ReadCheckpoint(
    const std::filesystem::path& directory, std::uint64_t hash,
    bool migrating) {
  auto candidates = CheckpointsIn(directory);
  std::reverse(candidates.begin(), candidates.end());
  for (const auto& path : candidates) {
    auto in = std::ifstream(path, std::ios::binary);
    if (!in || GetOne<std::uint64_t>(in) != kMagic) continue;
    if (GetOne<std::uint64_t>(in) != hash) {
      throw std::runtime_error(std::format(
          "{} was checkpointed by a different run (another config, or "
          "changed input files); set \"restart\": false to start over, "
          "or point checkpoint.directory elsewhere",
          path.string()));
    }
    auto out = ReadBack<CheckpointT>{};
    if (GetOne<std::uint64_t>(in) != KindOf(out.checkpoint)) {
      throw std::runtime_error(std::format(
          "{} holds the other command's recurrence; move it aside",
          path.string()));
    }
    out.node = GetOne<std::uint64_t>(in);
    out.checkpoint.t = GetOne<double>(in);
    GetState(in, out.checkpoint);
    out.checkpoint.loadCoefficients = GetComplex(in);
    out.checkpoint.tideCoefficients = GetComplex(in);
    const auto nodes = GetOne<std::uint64_t>(in);
    for (auto k = std::uint64_t{0}; k < nodes; ++k) {
      auto kept = Node{};
      kept.date_kyr_bp = GetOne<double>(in);
      kept.seaLevelChange = GetVector(in);
      kept.displacement = GetVector(in);
      kept.potential = GetVector(in);
      in.read(reinterpret_cast<char*>(kept.angularVelocity.data()),
              3 * sizeof(double));
      if (migrating) kept.oceanFunction = GetVector(in);
      out.history.push_back(std::move(kept));
    }
    if (in) return out;
  }
  return std::nullopt;
}

/// The histories as one surface file: SI node fields on a time axis in
/// seconds relative to the present (negative past).
void WriteHistoryFile(const EvolveInput& input,
                      const std::filesystem::path& config,
                      const SurfaceGrid& grid, const std::vector<Node>& history,
                      std::string_view command,
                      std::span<const double> initialSeaLevel,
                      std::string_view initialSeaLevelOrigin) {
  namespace pm = planetmodel;
  auto file = pm::SurfaceFile{};
  file.grid = pm::AngularGrid(
      std::vector<double>(grid.CoLatitudes().begin(), grid.CoLatitudes().end()),
      std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
      "gauss_legendre", input.lmax,
      std::vector<double>(grid.CoLatitudeWeights().begin(),
                          grid.CoLatitudeWeights().end()));
  file.source = std::format("sl3d {} (sl3dpp {})", command, Version());
  file.title = "viscoelastic GIA histories";
  auto axis = std::vector<double>{};
  auto dates = std::vector<double>{};
  auto omega = std::array<std::vector<double>, 3>{};
  for (const auto& kept : history) {
    axis.push_back(-kept.date_kyr_bp * KYR);
    dates.push_back(kept.date_kyr_bp);
    for (auto k = 0; k < 3; ++k) omega[k].push_back(kept.angularVelocity[k]);
  }
  file.time = axis;
  file.meta.emplace("config", config.string());
  file.meta.emplace("model", input.model.string());
  file.meta.emplace("ice_history", input.ice_history.string());
  file.meta.emplace("time",
                    std::string{"seconds relative to the present, negative "
                                "in the past; dates_kyr_bp lists the nodes"});
  file.meta.emplace("dates_kyr_bp", dates);
  file.meta.emplace("omega_x_rad_s", omega[0]);
  file.meta.emplace("omega_y_rad_s", omega[1]);
  file.meta.emplace("omega_z_rad_s", omega[2]);
  if (!input.viscosity.empty()) {
    file.meta.emplace("viscosity", input.viscosity.string());
  }
  file.meta.emplace(
      "shorelines",
      std::string{input.shorelines == Shorelines::Frozen ? "frozen"
                                                         : "migrating"});
  file.meta.emplace("initial_sea_level", std::string{initialSeaLevelOrigin});
  const auto fieldSize = grid.FieldSize();
  const auto potential = pm::Dimensions{.length = 2, .time = -2};
  auto gather = [&](auto member) {
    auto values = std::vector<double>{};
    values.reserve(history.size() * fieldSize);
    for (const auto& kept : history) {
      const auto& field = kept.*member;
      values.insert(values.end(), field.begin(), field.end());
    }
    return values;
  };
  auto put = [&file](std::string name, std::vector<double> values,
                     std::optional<pm::Dimensions> dims, std::string meaning) {
    auto options = pm::SurfaceFieldOptions{};
    options.support = "node";
    options.time_interpolation = "linear";
    options.dimensions = dims;
    options.meaning = std::move(meaning);
    file.variables.emplace(
        name, pm::SurfaceField(name, std::move(values), options));
  };
  put("sea_level_change", gather(&Node::seaLevelChange), pm::units::LENGTH,
      "sea-level change from the window's start");
  put("displacement", gather(&Node::displacement), pm::units::LENGTH,
      "radial surface displacement");
  put("potential", gather(&Node::potential), potential,
      "gravitational potential perturbation at the surface");
  {
    auto options = pm::SurfaceFieldOptions{};
    options.support = "static";
    options.dimensions = pm::units::LENGTH;
    options.meaning =
        "the initial sea level the run started from; the sea level at a node "
        "is this plus sea_level_change there";
    file.variables.emplace(
        "initial_sea_level",
        pm::SurfaceField(
            "initial_sea_level",
            std::vector<double>(initialSeaLevel.begin(), initialSeaLevel.end()),
            options));
  }
  if (input.shorelines == Shorelines::Migrating) {
    put("ocean_function", gather(&Node::oceanFunction), std::nullopt,
        "the ocean function at each node, following flotation");
  }
  if (const auto parent = input.output.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  pm::WriteSurfaceFile(input.output, file);
}

/// The run both commands share, over either evolution. `make` builds
/// the evolution from its options; `check` inspects each snapshot (the
/// 3D one refuses an unconverged node).
template <class Evolution>
std::string RunWith(
    const std::filesystem::path& config, const EvolveInput& input,
    const SeaLevelSolver& solver, Loaded& world, std::string_view command,
    const std::function<Evolution(typename Evolution::Options)>& make,
    const std::function<void(const typename Evolution::Snapshot&)>& check) {
  const auto started = std::chrono::steady_clock::now();
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto times = MakeTimes(input, scales);
  const auto iceLoad = IceLoadOf(input, solver, world);
  const auto migrating = input.shorelines == Shorelines::Migrating;
  const auto thickness = migrating
                             ? IceThicknessOf(input, solver, world)
                             : std::function<std::vector<double>(double)>{};

  const auto toPotential = scales.Factor(PotentialDimensions);
  const auto toOmega = 1.0 / scales.Time();
  auto history = std::vector<Node>{};
  auto keep = [&](const typename Evolution::Snapshot& snapshot) {
    check(snapshot);
    auto kept = Node{};
    kept.date_kyr_bp = input.start_kyr_bp - snapshot.t * scales.Time() / KYR;
    kept.seaLevelChange = snapshot.seaLevelChange;
    kept.displacement = snapshot.displacement;
    kept.potential = snapshot.potential;
    for (auto& v : kept.seaLevelChange) v *= length;
    for (auto& v : kept.displacement) v *= length;
    for (auto& v : kept.potential) v *= toPotential;
    for (auto k = 0; k < 3; ++k) {
      kept.angularVelocity[k] = snapshot.angularVelocity[k] * toOmega;
    }
    kept.oceanFunction = snapshot.oceanFunction;
    history.push_back(std::move(kept));
  };

  const auto checkpointing = !input.checkpoint_directory.empty();
  const auto hash = checkpointing ? FingerprintOf(input) : std::uint64_t{0};
  auto resumed = std::optional<ReadBack<typename Evolution::Checkpoint>>{};
  auto cleared = std::size_t{0};
  if (checkpointing && input.restart) {
    resumed = ReadCheckpoint<typename Evolution::Checkpoint>(
        input.checkpoint_directory, hash, migrating);
  } else if (checkpointing) {
    cleared = ClearCheckpoints(input.checkpoint_directory);
  }
  const auto replaced =
      cleared > 0 ? std::format(", replacing {} earlier checkpoint(s)", cleared)
                  : std::string{};
  auto options = typename Evolution::Options{};
  options.scheme = input.scheme;
  options.rotation = input.rotation;
  options.rtol = input.rtol;

  // The initial sea level: the state file's at the start, or calibrated
  // against its sea level at the end, by an evolution that streams nothing
  // (a resumption reads back the calibrated state its run kept).
  auto origin = std::string{"the state file at the window's start"};
  if (input.calibrate) {
    auto restored = std::optional<SeaLevelSolver::State>{};
    if (resumed) {
      restored =
          ReadInitialState(input.checkpoint_directory, hash, world.background);
    }
    if (restored) {
      StartFromIceHistory(input, solver, world, restored->seaLevel);
      world.background = std::move(*restored);
      origin = "calibrated (read back from the checkpoint directory)";
    } else {
      const auto top = planetmodel::ReadJsonFile(config);
      auto calibrating = make(options);
      const auto calibrated =
          CalibrateWorld(input, top, solver, world, calibrating, times);
      if (checkpointing) {
        WriteInitialState(input.checkpoint_directory, hash, world.background);
      }
      origin = std::format(
          "calibrated against the state file at the window's end, {} run(s), "
          "rms misfit {:.3g} m before, {:.3g} m after",
          calibrated.runs, calibrated.before_m, calibrated.after_m);
    }
  }

  options.on_snapshot = [&](std::size_t,
                            const typename Evolution::Snapshot& s) {
    keep(s);
  };
  const auto base_node = resumed ? resumed->node : std::size_t{0};
  if (checkpointing) {
    options.checkpoint_stride = input.checkpoint_every;
    options.on_checkpoint = [&](std::size_t node,
                                const typename Evolution::Checkpoint& cp) {
      WriteCheckpoint(input.checkpoint_directory, hash, base_node + node, cp,
                      history, migrating);
      const auto done =
          base_node + node + 1 >= static_cast<std::size_t>(input.nodes);
      if (input.max_walltime_min > 0.0 && !done) {
        const auto minutes = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count() /
                             60.0;
        if (minutes > input.max_walltime_min) {
          throw WalltimeReached{base_node + node};
        }
      }
    };
  }

  auto evolution = make(std::move(options));
  try {
    if (resumed) {
      history = std::move(resumed->history);
      const auto remaining = std::vector<double>(
          times.begin() + static_cast<std::ptrdiff_t>(resumed->node),
          times.end());
      if (remaining.size() >= 2) {  // a checkpoint at the last node is done
        if (migrating) {
          static_cast<void>(evolution.ResumeMigrating(resumed->checkpoint,
                                                      remaining, thickness));
        } else {
          static_cast<void>(
              evolution.Resume(resumed->checkpoint, remaining, iceLoad));
        }
      }
    } else if (migrating) {
      static_cast<void>(evolution.RunMigrating(times, thickness));
    } else {
      static_cast<void>(evolution.Run(times, iceLoad));
    }
  } catch (const WalltimeReached& at) {
    // with restart false, the same config would start over again
    return std::format(
        "checkpointed node {} of {} to {} at the wall-time limit{}; {}",
        at.node, input.nodes - 1, input.checkpoint_directory.string(), replaced,
        input.restart ? "rerun the same config to resume"
                      : "set \"restart\": true (the default) to resume");
  }

  auto initial = world.background.seaLevel;
  for (auto& v : initial) v *= length;
  WriteHistoryFile(input, config, grid, history, command, initial, origin);
  const auto calibrated = input.calibrate
                              ? std::string{", initial sea level calibrated"}
                              : std::string{};
  return std::format("wrote {} ({} nodes, {} to {} kyr BP{}{}{})",
                     input.output.string(), history.size(), input.start_kyr_bp,
                     input.end_kyr_bp, calibrated, resumed ? ", resumed" : "",
                     replaced);
}

}  // namespace

EvolveInput ReadEvolveInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  auto allowed = RunMemberNames();
  for (const auto name : {"checkpoint", "restart", "output"}) {
    allowed.push_back(name);
  }
  Allow(top, std::span(allowed));
  const auto directory = config.parent_path();
  auto out = EvolveInput{};
  ReadRunMembers(top, directory, out);
  if (Has(top, "checkpoint")) {
    const auto& checkpoint = top.At("checkpoint");
    Allow(checkpoint, {"directory", "every", "max_walltime_min"});
    out.checkpoint_directory = PathOf(checkpoint, "directory", directory);
    out.checkpoint_every = Integer(checkpoint, "every");
    if (out.checkpoint_every < 1) {
      Fail(checkpoint.At("every"), "is not a positive stride");
    }
    out.max_walltime_min = Number(checkpoint, "max_walltime_min", 0.0);
  }
  if (Has(top, "restart")) out.restart = top.At("restart").AsBool();
  out.output = PathOf(top, "output", directory);
  return out;
}

std::string RunEvolve(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  const auto input = ReadEvolveInput(config);
  if (!input.viscosity.empty()) {
    Fail(top.At("viscosity"),
         "names a 3D viscosity; evolve1d takes the model's own column — "
         "use `sl3d evolve`");
  }
  if (Has(top, "coupled")) {
    Fail(top.At("coupled"),
         "sets the 3D commands' within-step fixed point, which evolve1d does "
         "not have");
  }
  auto earth = BuildEarth(input, top);
  const auto rheology = Rheology::FromModel(earth.material, earth.model);
  if (!rheology) Fail(top.At("model"), rheology.error().message);
  ViscoelasticOperator op(*rheology, input.lmax);
  SeaLevelSolver solver(SurfaceGrid(input.lmax), op.ElasticLoveNumbers(),
                        ParametersOf(input, earth.material));
  auto world = LoadWorld(input, top, solver);
  return RunWith<SeaLevelEvolution>(
      config, input, solver, world, "evolve1d",
      [&](SeaLevelEvolution::Options options) {
        return SeaLevelEvolution(op, solver, world.background,
                                 std::move(options));
      },
      [](const SeaLevelEvolution::Snapshot& snapshot) {
        if (!snapshot.converged) {
          throw std::runtime_error(std::format(
              "the sea level solve did not converge at t = {} (shoreline "
              "cells flipping back and forth defeat the migrating solve's "
              "tolerance); loosen rtol",
              snapshot.t));
        }
      });
}

std::string RunEvolve3D(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  const auto input = ReadEvolveInput(config);
  if (input.viscosity.empty()) {
    Fail(top, "needs a viscosity section: a planetmodel model file whose "
              "viscosity drives the 3D run");
  }
  auto earth = BuildEarth(input, top);
  auto mv_options = io::ModelViscosityOptions{};
  mv_options.reference = input.viscosity_file_radial
                             ? io::ViscosityReference::FileRadial
                             : io::ViscosityReference::GeometricMean;
  const auto viscosity =
      io::ModelViscosity::Read(input.viscosity, earth.model, mv_options);
  if (!viscosity) Fail(top.At("viscosity").At("file"),
                       viscosity.error().message);
  const auto rheology =
      Rheology::Maxwell(earth.material, viscosity->Viscosity());
  if (!rheology) Fail(top.At("viscosity").At("file"),
                      rheology.error().message);
  Viscoelastic3D op(*rheology, input.lmax, viscosity->Delta());
  SeaLevelSolver solver(SurfaceGrid(input.lmax), op.ElasticLove(),
                        ParametersOf(input, earth.material));
  auto world = LoadWorld(input, top, solver);
  return RunWith<SeaLevelEvolution3D>(
      config, input, solver, world, "evolve",
      [&](SeaLevelEvolution3D::Options options) {
        options.coupled_rtol = input.coupled_rtol;
        options.max_coupled_iterations = input.max_coupled_iterations;
        return SeaLevelEvolution3D(op, solver, world.background,
                                   std::move(options));
      },
      [](const SeaLevelEvolution3D::Snapshot& snapshot) {
        if (!snapshot.converged) {
          throw std::runtime_error(std::format(
              "the coupled fixed point or its sea level solve did not "
              "converge at t = {} (residual {:.3g} after {} iterations; "
              "under migrating shorelines, cells flipping back and forth can "
              "defeat the tolerances); loosen coupled.rtol or rtol",
              snapshot.t, snapshot.coupledResidual,
              snapshot.coupledIterations));
        }
      });
}

}  // namespace sl3dpp::app
