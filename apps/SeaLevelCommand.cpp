/**
 * @file SeaLevelCommand.cpp
 * @brief `sl3d sea-level`: the sea level equation, the ocean coupled to
 * the solid Earth, under an ice load on a state read from planetmodel
 * surface files. Without a `times` block it is the elastic fingerprint of
 * an ice change on the state, one solve of the linear equation: with the
 * radial model's Love numbers (Fingerprint), or, under lateral moduli, the
 * instantaneous response of the 3-D problem (SeaLevelEvolution3D::
 * Instantaneous, distributed over radial segments on several ranks) with
 * the radial model's fingerprint beside it, written as a surface file of
 * static fields. With a `times` block it is GIA through time under an ice
 * history: the radial model's evolution (SeaLevelEvolution), or, under a
 * lateral viscosity or lateral moduli, the 3-D one (SeaLevelEvolution3D,
 * distributed over radial segments on several ranks), written as a
 * surface file with a slice per node and restartable from disk
 * checkpoints. The `model` block alone decides between 1-D and 3-D.
 */
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
#include "Ranks.hpp"
#include "Serialize.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

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
/// transforms' cache, the output) are left out, and so are the paths
/// themselves, so that these can change between a run and its resumption
/// and a run directory can move.
std::uint64_t RunHashOf(const SeaLevelInput& input) {
  const auto& run = input.run;
  const auto& window = *run.times;
  auto hash = Fnv1a{};
  hash.AddFile(run.model.deck);
  hash.AddValue(run.model.format);
  hash.AddValue(run.lmax);
  if (!run.model.viscosity.empty()) hash.AddFile(run.model.viscosity);
  hash.AddValue(run.model.viscosity_file_radial);
  hash.AddFile(run.forcing.state.file);
  hash.AddFile(run.forcing.ice.history);
  hash.AddValue(window.start_kyr_bp);
  hash.AddValue(window.end_kyr_bp);
  hash.AddValue(window.nodes);
  hash.AddValue(run.solver.scheme);
  hash.AddValue(run.rotation);
  hash.AddValue(run.solver.step_rtol);
  hash.AddValue(run.solver.step_max_iterations);
  hash.AddValue(run.hydrostatic_moments);
  hash.AddValue(run.solver.rtol);
  // the members below enter only when set away from their defaults: a run
  // that leaves them there hashes the members above alone
  if (run.ocean.shorelines != Shorelines::Frozen) {
    hash.AddValue(run.ocean.shorelines);
  }
  if (!run.solver.transform_degrees.empty()) {
    hash.AddValue(run.solver.transform_degrees.size());
    for (const auto degree : run.solver.transform_degrees) {
      hash.AddValue(degree);
    }
  }
  if (run.ocean.calibrate) {
    hash.AddValue(run.ocean.calibration_method);
    hash.AddValue(run.ocean.calibration_iterations);
    hash.AddValue(run.ocean.calibration_rtol);
  }
  if (!run.model.moduli.empty()) {
    hash.AddFile(run.model.moduli);
    hash.AddValue(run.model.moduli_file_radial);
  }
  if (run.solver.step != Viscoelastic3D::Options::StepSolver::FixedPoint) {
    hash.AddValue(run.solver.step);
  }
  // on several ranks each rank's checkpoint holds its own radial segment:
  // a restart over another number of ranks is another run
  if (TheRanks().Distributed()) hash.AddValue(TheRanks().size);
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

/// One checkpoint file of this run, or nothing when it is not a whole
/// checkpoint; refused when another run wrote it.
template <class CheckpointT>
std::optional<ReadBack<CheckpointT>> ReadCheckpointFile(
    const std::filesystem::path& path, std::uint64_t hash, bool migrating) {
  {
    auto in = std::ifstream(path, std::ios::binary);
    if (!in || GetOne<std::uint64_t>(in) != kMagic) return std::nullopt;
    if (GetOne<std::uint64_t>(in) != hash) {
      throw std::runtime_error(std::format(
          "{} was checkpointed by a different run (another config, changed "
          "input files, or another number of ranks); set \"restart\": false "
          "to start over, or point checkpoint.directory elsewhere",
          path.string()));
    }
    auto out = ReadBack<CheckpointT>{};
    if (GetOne<std::uint64_t>(in) != KindOf(out.checkpoint)) {
      throw std::runtime_error(std::format(
          "{} holds the recurrence of a run of the other dimension (1-D "
          "against 3-D); move it aside",
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

/// The newest checkpoint of this run in the directory, or nothing.
template <class CheckpointT>
std::optional<ReadBack<CheckpointT>> ReadCheckpoint(
    const std::filesystem::path& directory, std::uint64_t hash,
    bool migrating) {
  auto candidates = CheckpointsIn(directory);
  std::reverse(candidates.begin(), candidates.end());
  for (const auto& path : candidates) {
    if (auto read = ReadCheckpointFile<CheckpointT>(path, hash, migrating)) {
      return read;
    }
  }
  return std::nullopt;
}

/// On several ranks, each reading its own directory: the newest
/// checkpoint every rank has (a run stopped while the ranks wrote leaves
/// some a node ahead), or nothing on every rank. Each rank keeps its
/// newest two, so rank 0's newest two are the candidates.
template <class CheckpointT>
std::optional<ReadBack<CheckpointT>> ReadCommonCheckpoint(
    const std::filesystem::path& directory, std::uint64_t hash,
    bool migrating) {
  const auto& collectives = TheRanks().collectives;
  auto mine = std::vector<std::size_t>{};
  for (const auto& path : CheckpointsIn(directory)) {
    mine.push_back(static_cast<std::size_t>(
        std::stoul(path.filename().string().substr(5, 6))));
  }
  auto candidates = std::array{-1.0, -1.0};
  for (auto i = std::size_t{0}; i < std::min<std::size_t>(2, mine.size());
       ++i) {
    candidates[i] = static_cast<double>(mine[mine.size() - 1 - i]);
  }
  collectives.share(candidates);
  auto lacks = std::array{0.0, 0.0};
  for (auto i = 0; i < 2; ++i) {
    lacks[i] =
        candidates[i] < 0.0 ||
                std::ranges::find(
                    mine, static_cast<std::size_t>(candidates[i])) == mine.end()
            ? 1.0
            : 0.0;
  }
  collectives.max(lacks);
  for (auto i = 0; i < 2; ++i) {
    if (lacks[i] != 0.0) continue;
    const auto node = static_cast<std::size_t>(candidates[i]);
    auto read = ReadCheckpointFile<CheckpointT>(
        directory / std::format("node_{:06}.ckpt", node), hash, migrating);
    auto unreadable = read ? 0.0 : 1.0;
    collectives.max(std::span(&unreadable, 1));
    if (unreadable != 0.0) {
      throw std::runtime_error(std::format(
          "the checkpoint of node {} is unreadable on some rank", node));
    }
    return read;
  }
  return std::nullopt;
}

/// The histories as one surface file: SI node fields on a time axis in
/// seconds relative to the present (negative past).
void WriteHistoryFile(const SeaLevelInput& input,
                      const std::filesystem::path& config,
                      const SurfaceGrid& grid, const std::vector<Node>& history,
                      std::span<const double> initialSeaLevel,
                      std::string_view initialSeaLevelOrigin) {
  namespace pm = planetmodel;
  const auto& run = input.run;
  auto file = pm::SurfaceFile{};
  file.grid = pm::AngularGrid(
      std::vector<double>(grid.CoLatitudes().begin(), grid.CoLatitudes().end()),
      std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
      "gauss_legendre", run.lmax,
      std::vector<double>(grid.CoLatitudeWeights().begin(),
                          grid.CoLatitudeWeights().end()));
  file.source = std::format("sl3d sea-level (sl3dpp {})", Version());
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
  file.meta.emplace("model", run.model.deck.string());
  file.meta.emplace("ice_history", run.forcing.ice.history.string());
  file.meta.emplace("time",
                    std::string{"seconds relative to the present, negative "
                                "in the past; dates_kyr_bp lists the nodes"});
  file.meta.emplace("dates_kyr_bp", dates);
  file.meta.emplace("omega_x_rad_s", omega[0]);
  file.meta.emplace("omega_y_rad_s", omega[1]);
  file.meta.emplace("omega_z_rad_s", omega[2]);
  if (!run.model.viscosity.empty()) {
    file.meta.emplace("viscosity", run.model.viscosity.string());
  }
  if (!run.model.moduli.empty()) {
    file.meta.emplace("moduli", run.model.moduli.string());
  }
  file.meta.emplace("shorelines", std::string{NameOf(run.ocean.shorelines)});
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
    file.variables.emplace(name,
                           pm::SurfaceField(name, std::move(values), options));
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
  if (run.ocean.shorelines == Shorelines::Migrating) {
    put("ocean_function", gather(&Node::oceanFunction), std::nullopt,
        "the ocean function at each node, following flotation");
  }
  if (const auto parent = input.output.surface.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  pm::WriteSurfaceFile(input.output.surface, file);
}

/// What the config was taken to mean, for the summary: the model's
/// dimension, elastic or viscoelastic with the window, the shorelines and
/// the calibration, and the rotation and moments.
std::string Meaning(const SeaLevelRun& run, const BuiltEarth& earth) {
  auto out = earth.Dimension();
  if (!run.times) {
    out += ", elastic (fingerprint)";
  } else {
    const auto& window = *run.times;
    out += std::format(
        ", viscoelastic, {} nodes {} to {} kyr BP, shorelines {}", window.nodes,
        window.start_kyr_bp, window.end_kyr_bp, NameOf(run.ocean.shorelines));
    if (run.ocean.calibrate) out += ", initial sea level calibrated";
  }
  out += std::format(", rotation {}", NameOf(run.rotation));
  if (run.hydrostatic_moments) out += ", moments hydrostatic";
  return out;
}

/// A run through time over either evolution. `make` builds the evolution
/// from its options; `check` inspects each snapshot (refusing an
/// unconverged node). `meaning` heads the summary.
template <class Evolution>
std::string RunWith(
    const std::filesystem::path& config, const SeaLevelInput& input,
    const Json& top, std::string_view meaning, const SeaLevelSolver& solver,
    Loaded& world,
    const std::function<Evolution(typename Evolution::Options)>& make,
    const std::function<void(const typename Evolution::Snapshot&)>& check) {
  const auto started = std::chrono::steady_clock::now();
  const auto& run = input.run;
  const auto& window = *run.times;
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto times = MakeTimes(run, scales);
  const auto iceLoad = IceLoadOf(run, solver, world);
  const auto migrating = run.ocean.shorelines == Shorelines::Migrating;
  const auto thickness = migrating
                             ? IceThicknessOf(run, solver, world)
                             : std::function<std::vector<double>(double)>{};

  const auto toPotential = scales.Factor(PotentialDimensions);
  const auto toOmega = 1.0 / scales.Time();
  auto history = std::vector<Node>{};
  auto keep = [&](const typename Evolution::Snapshot& snapshot) {
    check(snapshot);
    auto kept = Node{};
    kept.date_kyr_bp = window.start_kyr_bp - snapshot.t * scales.Time() / KYR;
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

  // On several ranks each checkpoints its own radial segment into its own
  // subdirectory, and they resume from the newest node all of them have;
  // rank 0 alone keeps the history and writes the output.
  const auto& ranks = TheRanks();
  const auto& checkpoint = input.checkpoint;
  const auto checkpointing = !checkpoint.directory.empty();
  const auto directory = RankDirectory(checkpoint.directory);
  const auto hash = checkpointing ? RunHashOf(input) : std::uint64_t{0};
  auto resumed = std::optional<ReadBack<typename Evolution::Checkpoint>>{};
  auto cleared = std::size_t{0};
  if (checkpointing && input.restart) {
    using CheckpointT = typename Evolution::Checkpoint;
    resumed =
        ranks.Distributed()
            ? ReadCommonCheckpoint<CheckpointT>(directory, hash, migrating)
            : ReadCheckpoint<CheckpointT>(directory, hash, migrating);
    // a set the ranks do not share is stale: cleared, or its later nodes
    // would outrank the new run's in the pruning
    if (!resumed && ranks.Distributed()) cleared = ClearCheckpoints(directory);
  } else if (checkpointing) {
    cleared = ClearCheckpoints(directory);
  }
  const auto replaced =
      cleared > 0 ? std::format(", replacing {} earlier checkpoint(s)", cleared)
                  : std::string{};
  auto options = typename Evolution::Options{};
  options.scheme = run.solver.scheme;
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;

  // The initial sea level: the state file's at the start, or calibrated
  // against its sea level at the end, by an evolution that streams nothing
  // (a resumption reads back the calibrated state its run kept).
  auto origin = std::string{"the state file at the window's start"};
  if (run.ocean.calibrate) {
    auto restored = std::optional<SeaLevelSolver::State>{};
    if (resumed) {
      restored = ReadInitialState(directory, hash, world.background);
    }
    if (restored) {
      StartFromIceHistory(run, solver, world, restored->seaLevel);
      world.background = std::move(*restored);
      origin = "calibrated (read back from the checkpoint directory)";
    } else {
      auto calibrating = make(options);
      const auto calibrated =
          CalibrateWorld(run, top, solver, world, calibrating, times);
      if (checkpointing) WriteInitialState(directory, hash, world.background);
      origin = std::format(
          "calibrated against the state file at the window's end, {} run(s), "
          "rms misfit {:.3g} m before, {:.3g} m after",
          calibrated.runs, calibrated.before_m, calibrated.after_m);
    }
  }

  options.on_snapshot = [&](std::size_t,
                            const typename Evolution::Snapshot& s) { keep(s); };
  const auto base_node = resumed ? resumed->node : std::size_t{0};
  if (checkpointing) {
    options.checkpoint_stride = checkpoint.every;
    options.on_checkpoint = [&](std::size_t node,
                                const typename Evolution::Checkpoint& cp) {
      WriteCheckpoint(directory, hash, base_node + node, cp,
                      ranks.First() ? history : std::vector<Node>{}, migrating);
      const auto done =
          base_node + node + 1 >= static_cast<std::size_t>(window.nodes);
      if (checkpoint.max_walltime_min > 0.0 && !done) {
        const auto minutes = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count() /
                             60.0;
        // the ranks' clocks differ: they stop when any has passed it
        auto passed = minutes > checkpoint.max_walltime_min ? 1.0 : 0.0;
        if (ranks.Distributed()) ranks.collectives.max(std::span(&passed, 1));
        if (passed != 0.0) throw WalltimeReached{base_node + node};
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
        "{}; checkpointed node {} of {} to {} at the wall-time limit{}; {}",
        meaning, at.node, window.nodes - 1, checkpoint.directory.string(),
        replaced,
        input.restart ? "rerun the same config to resume"
                      : "set \"restart\": true (the default) to resume");
  }

  auto initial = world.background.seaLevel;
  for (auto& v : initial) v *= length;
  if (ranks.First()) {
    WriteHistoryFile(input, config, grid, history, initial, origin);
  }
  return std::format("{}; wrote {}{}{}", meaning, input.output.surface.string(),
                     resumed ? ", resumed" : "", replaced);
}

/// The fingerprint's forcing on a grid at the run's degree, SI: the
/// state's ice thickness and sea level at its date, and the ice change.
struct FingerprintForcing {
  std::vector<double> ice;
  std::vector<double> seaLevel;
  std::vector<double> thicknessChange;
};

/// Reads the fingerprint's state from `forcing.state.file` onto the grid
/// and builds the ice change there.
FingerprintForcing ReadFingerprintForcing(const SeaLevelRun& run,
                                          const Json& top,
                                          const SurfaceGrid& grid,
                                          const SeaLevelParameters& parameters) {
  const auto& state = run.forcing.state;
  const auto& change = *run.forcing.ice.change;
  const auto& stateMember = top.At("forcing").At("state").At("file");
  const auto history = io::FieldHistory::Read(state.file, grid);
  if (!history) Fail(stateMember, history.error().message);
  for (const auto name : {"ice_thickness", "topography"}) {
    if (!history->Has(name)) {
      Fail(stateMember, std::format("holds no field {}", name));
    }
  }
  // the state at its date, its sea level by the flotation construction
  // with the solver's own densities
  auto out = FingerprintForcing{};
  const auto date = -state.date_kyr_bp * KYR;
  out.ice = history->Values("ice_thickness", date);
  out.seaLevel = SeaLevelFromTopography(history->Values("topography", date),
                                        out.ice, parameters);

  // the change: the Gaussian cap's weight times the amount, of the local
  // ice thickness for a fraction
  const auto mask = CapField(grid, change.site, change.width, 1.0);
  out.thicknessChange.resize(out.ice.size());
  const auto fraction = change.kind == IceChange::Kind::CapFraction;
  for (auto i = std::size_t{0}; i < out.ice.size(); ++i) {
    out.thicknessChange[i] =
        change.amount * mask[i] * (fraction ? out.ice[i] : 1.0);
  }
  return out;
}

/// The radial model's fingerprint: one solve of the linear sea level
/// equation for the ice change on the state, with the Love numbers of the
/// deck's elastic structure (Fingerprint).
struct RadialFingerprint {
  Fingerprint fingerprint;
  FingerprintForcing forcing;
  Fingerprint::Result result;
};

RadialFingerprint SolveRadialFingerprint(const SeaLevelRun& run,
                                         const Json& top,
                                         const BuiltEarth& earth) {
  const auto love = ComputeLoveNumbers(earth.material, run.lmax);
  auto options = Fingerprint::Options{};
  options.parameters = ParametersOf(run, earth.material);
  options.rtol = run.solver.rtol;
  auto out = RadialFingerprint{Fingerprint(run.lmax, love, options), {}, {}};
  out.forcing = ReadFingerprintForcing(run, top, out.fingerprint.Grid(),
                                       options.parameters);
  out.fingerprint.SetState(out.forcing.ice, out.forcing.seaLevel);
  out.result = out.fingerprint.SolveForIceChange(out.forcing.thicknessChange,
                                                 run.rotation);
  if (!out.result.converged) {
    throw std::runtime_error(
        "the sea level equation did not converge; loosen solver.rtol");
  }
  return out;
}

/// A static SI field of a fingerprint's surface file.
void PutStatic(planetmodel::SurfaceFile& file, std::string name,
               const std::vector<double>& values,
               std::optional<planetmodel::Dimensions> dims,
               std::string meaning) {
  file.variables.emplace(
      name, planetmodel::SurfaceField(name, values,
                                      {.dimensions = dims, .meaning = meaning}));
}

/// A fingerprint as a surface file of static SI fields on the run's grid:
/// the ice change applied, the response (sea level, displacement and
/// potential, the angular velocity in the metadata) and the state's ocean
/// function, with the config, model, moduli, state, date and rotation in
/// the metadata; the runner adds its own.
planetmodel::SurfaceFile FingerprintFile(
    const std::filesystem::path& config, const SeaLevelRun& run,
    const SurfaceGrid& grid, std::string title,
    const std::vector<double>& thicknessChange,
    const std::vector<double>& seaLevelChange,
    const std::vector<double>& displacement,
    const std::vector<double>& potential,
    const std::array<double, 3>& angularVelocity,
    std::span<const double> ocean) {
  namespace pm = planetmodel;
  const auto& state = run.forcing.state;
  auto file = pm::SurfaceFile{};
  file.grid = pm::AngularGrid(
      std::vector<double>(grid.CoLatitudes().begin(), grid.CoLatitudes().end()),
      std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
      "gauss_legendre", run.lmax,
      std::vector<double>(grid.CoLatitudeWeights().begin(),
                          grid.CoLatitudeWeights().end()));
  file.source = std::format("sl3d sea-level (sl3dpp {})", Version());
  file.title = std::move(title);
  file.meta.emplace("config", config.string());
  file.meta.emplace("model", run.model.deck.string());
  if (!run.model.moduli.empty()) {
    file.meta.emplace("moduli", run.model.moduli.string());
  }
  file.meta.emplace("state", state.file.string());
  file.meta.emplace("date_kyr_bp", std::vector<double>{state.date_kyr_bp});
  file.meta.emplace("rotation", std::string{NameOf(run.rotation)});
  file.meta.emplace("omega_x_rad_s", std::vector<double>{angularVelocity[0]});
  file.meta.emplace("omega_y_rad_s", std::vector<double>{angularVelocity[1]});
  file.meta.emplace("omega_z_rad_s", std::vector<double>{angularVelocity[2]});
  const auto length = pm::units::LENGTH;
  PutStatic(file, "ice_change", thicknessChange, length,
            "the ice-thickness change applied");
  PutStatic(file, "sea_level_change", seaLevelChange, length,
            "sea-level change of the fingerprint");
  PutStatic(file, "displacement", displacement, length,
            "radial surface displacement");
  PutStatic(file, "potential", potential,
            pm::Dimensions{.length = 2, .time = -2},
            "gravitational potential perturbation at the surface");
  PutStatic(file, "ocean_function",
            std::vector<double>(ocean.begin(), ocean.end()), std::nullopt,
            "the ocean function of the state");
  return file;
}

/// Writes a surface file, creating its directory.
void WriteSurface(const planetmodel::SurfaceFile& file,
                  const std::filesystem::path& output) {
  if (const auto parent = output.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  planetmodel::WriteSurfaceFile(output, file);
}

/// The elastic fingerprint on a radial Earth: one solve of the linear sea
/// level equation for the ice change on the state, with the radial model's
/// Love numbers.
std::string RunFingerprint(const std::filesystem::path& config,
                           const SeaLevelInput& input, const Json& top,
                           const BuiltEarth& earth, std::string_view meaning) {
  const auto radial = SolveRadialFingerprint(input.run, top, earth);
  const auto& result = radial.result;
  auto file = FingerprintFile(
      config, input.run, radial.fingerprint.Grid(),
      "elastic sea-level fingerprint", radial.forcing.thicknessChange,
      result.seaLevelChange, result.displacement, result.potential,
      result.angularVelocity, radial.fingerprint.OceanFunction());
  file.meta.emplace("sle_iterations",
                    std::vector<long long>{result.iterations});
  const auto& output = input.output.surface;
  WriteSurface(file, output);
  return std::format("{}; wrote {} ({} SLE iterations)", meaning,
                     output.string(), result.iterations);
}

/// The elastic fingerprint on a laterally heterogeneous Earth: the
/// instantaneous response of the 3-D sea level problem to the ice change
/// on the state (SeaLevelEvolution3D::Instantaneous), the linear sea level
/// equation on the state's shorelines coupled to the lateral moduli's
/// unrelaxed solid by the fixed point of a zero-length step (or the sea
/// level closure nested over conjugate-gradient solves, solver.step). The
/// operator is the stepper on a branchless rheology of the deck's elastic
/// structure, as `response`'s elastic 3-D solve, distributed over radial
/// segments on several ranks, the cuts balancing the elements the moduli
/// make active; the radial model's fingerprint of the same change is
/// written beside it.
std::string RunFingerprint3D(const std::filesystem::path& config,
                             const SeaLevelInput& input, const Json& top,
                             const BuiltEarth& earth,
                             std::string_view meaning) {
  namespace pm = planetmodel;
  const auto& run = input.run;
  const auto elastic = Rheology::Create(earth.material, {});
  if (!elastic) Fail(top.At("model").At("deck"), elastic.error().message);
  const Viscoelastic3D op(*elastic, run.lmax,
                          [](double, double, double) { return 0.0; },
                          OperatorOptionsOf(run, earth, *elastic));
  // the sea level grid blocks its transforms for the operator's cache
  const auto parameters = ParametersOf(run, earth.material);
  SeaLevelSolver solver(
      SurfaceGrid(run.lmax, {.cache_bytes = op.GetOptions().cache_bytes}),
      op.ElasticLove(), parameters);
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();

  // the state and the grounded load of the change, solver units
  const auto forcing = ReadFingerprintForcing(run, top, grid, parameters);
  auto ice = forcing.ice;
  auto sea = forcing.seaLevel;
  for (auto& v : ice) v /= length;
  for (auto& v : sea) v /= length;
  const auto state = solver.MakeState(std::move(ice), std::move(sea));
  auto load = std::vector<double>(grid.FieldSize());
  for (auto i = std::size_t{0}; i < load.size(); ++i) {
    load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
              forcing.thicknessChange[i] / length;
  }

  auto options = SeaLevelEvolution3D::Options{};
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  options.coupled_rtol = run.solver.step_rtol;
  options.max_coupled_iterations = run.solver.step_max_iterations;
  SeaLevelEvolution3D evolution(op, solver, state, options);
  auto response = SeaLevelEvolution3D::Snapshot{};
  try {
    response = evolution.Instantaneous(load);
  } catch (const std::runtime_error& error) {
    throw std::runtime_error(std::format(
        "{}; loosen solver.step_rtol or raise solver.step_max_iterations",
        error.what()));
  }
  if (!response.converged) {
    throw std::runtime_error(
        "the sea level equation did not converge; loosen solver.rtol");
  }
  if (!TheRanks().First()) return "";

  // The radial model's fingerprint of the same change, the first rank's.
  const auto radial = SolveRadialFingerprint(run, top, earth);
  const auto toPotential = scales.Factor(PotentialDimensions);
  const auto toFrequency = scales.Factor(Dimensions{.time = -1});
  auto seaLevelChange = response.seaLevelChange;
  auto displacement = response.displacement;
  auto potential = response.potential;
  for (auto& v : seaLevelChange) v *= length;
  for (auto& v : displacement) v *= length;
  for (auto& v : potential) v *= toPotential;
  auto omega = response.angularVelocity;
  for (auto& v : omega) v *= toFrequency;

  auto file = FingerprintFile(
      config, run, grid,
      "elastic sea-level fingerprint of a laterally heterogeneous Earth",
      forcing.thicknessChange, seaLevelChange, displacement, potential, omega,
      state.oceanFunction);
  const auto conjugate = run.solver.step ==
                         Viscoelastic3D::Options::StepSolver::ConjugateGradients;
  file.meta.emplace("lateral", std::vector<long long>{1});
  file.meta.emplace("moduli_reference",
                    std::string{run.model.moduli_file_radial ? "file_radial"
                                                             : "lateral_mean"});
  file.meta.emplace("step_solver", std::string{conjugate ? "conjugate_gradients"
                                                         : "fixed_point"});
  file.meta.emplace("step_iterations",
                    std::vector<long long>{response.coupledIterations});
  file.meta.emplace("solve_iterations",
                    std::vector<long long>{response.solveIterations});
  file.meta.emplace("step_residual",
                    std::vector<double>{response.coupledResidual});
  const auto& radialOmega = radial.result.angularVelocity;
  file.meta.emplace("omega_x_radial_model_rad_s",
                    std::vector<double>{radialOmega[0]});
  file.meta.emplace("omega_y_radial_model_rad_s",
                    std::vector<double>{radialOmega[1]});
  file.meta.emplace("omega_z_radial_model_rad_s",
                    std::vector<double>{radialOmega[2]});
  const auto lengthDims = pm::units::LENGTH;
  PutStatic(file, "sea_level_change_radial_model",
            radial.result.seaLevelChange, lengthDims,
            "sea-level change of the radial model's fingerprint of the same "
            "ice change");
  PutStatic(file, "displacement_radial_model", radial.result.displacement,
            lengthDims,
            "radial surface displacement of the radial model's fingerprint");
  PutStatic(file, "potential_radial_model", radial.result.potential,
            pm::Dimensions{.length = 2, .time = -2},
            "the radial model's potential perturbation at the surface");
  const auto& output = input.output.surface;
  WriteSurface(file, output);
  return conjugate
             ? std::format("{}; wrote {} ({} conjugate-gradient iterations in "
                           "{} passes, residual {:.2e})",
                           meaning, output.string(), response.solveIterations,
                           response.coupledIterations, response.coupledResidual)
             : std::format("{}; wrote {} ({} fixed-point iterations, residual "
                           "{:.2e})",
                           meaning, output.string(), response.coupledIterations,
                           response.coupledResidual);
}

/// The radial model's run through time.
std::string Run1D(const std::filesystem::path& config,
                  const SeaLevelInput& input, const Json& top,
                  const BuiltEarth& earth, std::string_view meaning) {
  const auto& run = input.run;
  const auto rheology = RheologyOf(earth, top);
  ViscoelasticOperator op(rheology, run.lmax);
  SeaLevelSolver solver(SurfaceGrid(run.lmax), op.ElasticLoveNumbers(),
                        ParametersOf(run, earth.material));
  auto world = LoadWorld(run, top, solver);
  return RunWith<SeaLevelEvolution>(
      config, input, top, meaning, solver, world,
      [&](SeaLevelEvolution::Options options) {
        return SeaLevelEvolution(op, solver, world.background,
                                 std::move(options));
      },
      [](const SeaLevelEvolution::Snapshot& snapshot) {
        if (!snapshot.converged) {
          throw std::runtime_error(std::format(
              "the sea level solve did not converge at t = {} (shoreline "
              "cells flipping back and forth defeat the migrating solve's "
              "tolerance); loosen solver.rtol",
              snapshot.t));
        }
      });
}

/// The 3-D run through time: a lateral viscosity, lateral moduli or both.
std::string Run3D(const std::filesystem::path& config,
                  const SeaLevelInput& input, const Json& top,
                  const BuiltEarth& earth, std::string_view meaning) {
  const auto& run = input.run;
  const auto rheology = RheologyOf(earth, top);
  Viscoelastic3D op(rheology, run.lmax, DeltaOf(earth),
                    OperatorOptionsOf(run, earth, rheology));
  // the sea level grid blocks its transforms for the operator's cache
  SeaLevelSolver solver(
      SurfaceGrid(run.lmax, {.cache_bytes = op.GetOptions().cache_bytes}),
      op.ElasticLove(), ParametersOf(run, earth.material));
  auto world = LoadWorld(run, top, solver);
  return RunWith<SeaLevelEvolution3D>(
      config, input, top, meaning, solver, world,
      [&](SeaLevelEvolution3D::Options options) {
        options.coupled_rtol = run.solver.step_rtol;
        options.max_coupled_iterations = run.solver.step_max_iterations;
        return SeaLevelEvolution3D(op, solver, world.background,
                                   std::move(options));
      },
      [](const SeaLevelEvolution3D::Snapshot& snapshot) {
        if (!snapshot.converged) {
          throw std::runtime_error(std::format(
              "the coupled fixed point or its sea level solve did not "
              "converge at t = {} (residual {:.3g} after {} iterations; "
              "under migrating shorelines, cells flipping back and forth can "
              "defeat the tolerances); loosen solver.step_rtol or "
              "solver.rtol",
              snapshot.t, snapshot.coupledResidual,
              snapshot.coupledIterations));
        }
      });
}

}  // namespace

SeaLevelInput ReadSeaLevelInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  auto allowed = SeaLevelRunMemberNames();
  for (const auto name : {"checkpoint", "restart", "output"}) {
    allowed.push_back(name);
  }
  AllowTopLevel(top, allowed);
  const auto directory = config.parent_path();
  auto out = SeaLevelInput{};
  out.run = ReadSeaLevelRun(top, directory);
  if (!out.run.times) {
    for (const auto* name : {"checkpoint", "restart"}) {
      if (Has(top, name)) {
        Fail(top.At(name),
             "needs a `times` block (a run through time): the elastic "
             "fingerprint is one solve, with nothing to checkpoint");
      }
    }
  }
  if (Has(top, "checkpoint")) {
    out.checkpoint = ReadCheckpointBlock(top.At("checkpoint"), directory);
  }
  if (Has(top, "restart")) out.restart = top.At("restart").AsBool();
  const auto& output = top.At("output");
  if (output.IsObject()) Allow(output, {"surface"});
  out.output = ReadOutputBlock(output, directory);
  return out;
}

std::string RunSeaLevel(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  const auto input = ReadSeaLevelInput(config);
  const auto& run = input.run;
  const auto earth = BuildEarth(run, top);
  const auto meaning = Meaning(run, earth);
  const auto& ranks = TheRanks();
  if (!run.times) {
    // a viscosity file is refused without `times` when the config is read,
    // so a lateral Earth here is one of lateral moduli
    if (earth.LateralModuli()) {
      return RunFingerprint3D(config, input, top, earth, meaning);
    }
    RefuseThreeDSolver(top, "the elastic fingerprint on a radial Earth");
    // one process's run: on several ranks, the first rank's alone
    if (!ranks.First()) return "";
    const RanksOverride alone(Ranks{});
    return RunFingerprint(config, input, top, earth, meaning);
  }
  if (!earth.Lateral()) {
    RefuseThreeDSolver(top, "a 1-D run (a radial model)");
    if (!ranks.First()) return "";
    const RanksOverride alone(Ranks{});
    return Run1D(config, input, top, earth, meaning);
  }
  return Run3D(config, input, top, earth, meaning);
}

}  // namespace sl3dpp::app
