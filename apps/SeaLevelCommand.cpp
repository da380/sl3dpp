/**
 * @file SeaLevelCommand.cpp
 * @brief `sl3d sea-level`: the sea level equation, the ocean coupled to
 * the solid Earth, on a state (the sea level and the ice at one instant)
 * read from its file inputs, under a forcing of any mix of an ice change,
 * surface loads, a tide and seismic sources, with frozen or migrating
 * shorelines; without a state (no `ocean` block), the same runs on a body
 * without an ocean (GRDSolver::NoOceanState) under the loads, the tide and
 * the sources. The static problem is the solve at the state's date on the
 * elastic Earth, a run of one node of the library's evolutions: of the
 * radial model (Evolution, on a branchless rheology), or, under lateral
 * moduli, of the 3-D one (Evolution3D, distributed over radial segments
 * on several ranks) with the radial model's answer beside it, written as
 * a surface file of static fields. The quasi-static problem is GIA
 * through time from a state at an end of the window: the radial model's
 * evolution (Evolution), or, under a lateral viscosity or lateral moduli,
 * the 3-D one (Evolution3D, distributed over radial segments on several
 * ranks), written as a surface file with a slice per node (the response
 * and the potentials of the tangential displacement) and restartable from
 * disk checkpoints. The `model` block alone decides between 1-D and 3-D.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
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
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

namespace sl3dpp::app {

namespace {

/// One node of the accumulated history, SI, kept across restarts: the
/// response, the potentials of the tangential displacement (the toroidal
/// one in the frame of no net rotation of the surface, empty where the
/// node has none) and, under migrating shorelines, the ocean function.
struct Node {
  double date_kyr_bp = 0.0;
  std::vector<double> relativeSeaLevelChange;
  std::vector<double> displacement;
  std::vector<double> potential;
  std::vector<double> tangentialPotential;
  std::vector<double> toroidalPotential;
  std::array<double, 3> angularVelocity{};
  std::vector<double> oceanFunction;  ///< migrating runs alone
};

/// The outputs' `ocean` metadata of a run without an ocean.
constexpr auto kNoOcean = std::string_view{"none"};

/// A run stopped on purpose at a checkpoint: the wall clock passed the
/// configured limit after the checkpoint was written.
struct WalltimeReached {
  std::size_t node = 0;
};

// -- the checkpoint files: private scratch of a run, machine-local ----------

/// The checkpoints' magic, its last character the format's version: the
/// file of another version (a history of other fields) is refused.
constexpr auto kMagic = std::uint64_t{0x534c3344434b5032};  // "SL3DCKP2"
constexpr bool OtherFormat(std::uint64_t magic) {
  return magic != kMagic && (magic >> 8) == (kMagic >> 8);
}
constexpr auto kKind1D = std::uint64_t{1};
constexpr auto kKind3D = std::uint64_t{2};

constexpr std::uint64_t KindOf(const Evolution::Checkpoint&) {
  return kKind1D;
}
constexpr std::uint64_t KindOf(const Evolution3D::Checkpoint&) {
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
  // the state's files, or the body's having no ocean
  if (run.ocean) {
    hash.AddFile(run.ocean->seaLevel.input.file);
    hash.AddFile(run.ocean->ice.file);
  } else {
    hash.Add(std::as_bytes(std::span(kNoOcean)));
  }
  // the ice change, term by term: its kind, its file's contents and
  // variable, or a cap's description (its place, size, amount and
  // history)
  hash.AddValue(run.forcing.ice_change.size());
  for (const auto& term : run.forcing.ice_change) {
    hash.AddValue(term.kind);
    if (!term.file.file.empty()) {
      hash.AddFile(term.file.file);
      hash.Add(std::as_bytes(std::span(term.file.variable)));
    } else {
      const auto described = Describe(term);
      hash.Add(std::as_bytes(std::span(described)));
    }
  }
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
  if (run.Migrating()) hash.AddValue(Shorelines::Migrating);
  if (run.ocean && run.ocean->at != WindowEnd::Start) {
    hash.AddValue(run.ocean->at);
  }
  if (!run.solver.transform_degrees.empty()) {
    hash.AddValue(run.solver.transform_degrees.size());
    for (const auto degree : run.solver.transform_degrees) {
      hash.AddValue(degree);
    }
  }
  if (run.ocean) {
    const auto& ice = run.ocean->ice;
    const auto& sea = run.ocean->seaLevel;
    // the state's variables and quantity, where they are other than the
    // ice thickness and the topography under their own names
    if (ice.variable != "ice_thickness") {
      hash.Add(std::as_bytes(std::span(ice.variable)));
    }
    // the land held, where the run is told one
    if (run.ocean->land) {
      hash.AddFile(run.ocean->land->file);
      hash.Add(std::as_bytes(std::span(run.ocean->land->variable)));
    }
    if (sea.kind != SeaLevelSource::Kind::Topography ||
        sea.input.variable != "topography") {
      hash.AddValue(sea.kind);
      hash.Add(std::as_bytes(std::span(sea.input.variable)));
    }
  }
  // the forcing beside the ice, where there is one
  // (a term's description less its file's path, and the file's contents:
  // a run directory may move)
  for (const auto& term : run.forcing.load) {
    auto unplaced = term;
    unplaced.term.field.file.clear();
    unplaced.term.file.clear();
    const auto described = Describe(unplaced);
    hash.Add(std::as_bytes(std::span(described)));
    // and its numbers whole, which the description rounds
    for (const auto value :
         {term.term.site.latitude, term.term.site.longitude, term.term.width,
          term.term.radius, term.term.amplitude, term.term.time_origin_kyr,
          term.density}) {
      hash.AddValue(value);
    }
    for (const auto* values :
         {&term.term.history.times_kyr, &term.term.history.factors}) {
      for (const auto value : *values) hash.AddValue(value);
    }
    if (term.term.kind == LoadTerm::Kind::Field) {
      hash.AddFile(term.term.field.file);
    } else if (term.term.kind == LoadTerm::Kind::Coefficients) {
      hash.AddFile(term.term.file);
    }
  }
  // the tide, coefficient by coefficient, its numbers whole
  if (!run.forcing.tide.empty()) {
    hash.AddValue(run.forcing.tide.size());
    for (const auto& term : run.forcing.tide) {
      hash.AddValue(term.degree);
      hash.AddValue(term.order);
      hash.AddValue(term.amplitude);
      hash.AddValue(term.imaginary);
      hash.AddValue(term.history.times_kyr.size());
      for (auto k = std::size_t{0}; k < term.history.times_kyr.size(); ++k) {
        hash.AddValue(term.history.times_kyr[k]);
        hash.AddValue(term.history.factors[k]);
      }
    }
  }
  if (run.Calibrating()) {
    hash.AddValue(run.ocean->calibration_method);
    hash.AddValue(run.ocean->calibration_iterations);
    hash.AddValue(run.ocean->calibration_rtol);
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
                       std::uint64_t hash, const GRDSolver::State& state) {
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
std::optional<GRDSolver::State> ReadInitialState(
    const std::filesystem::path& directory, std::uint64_t hash,
    const GRDSolver::State& background) {
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
      PutVector(out, kept.relativeSeaLevelChange);
      PutVector(out, kept.displacement);
      PutVector(out, kept.potential);
      PutVector(out, kept.tangentialPotential);
      PutVector(out, kept.toroidalPotential);
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
    if (!in) return std::nullopt;
    const auto magic = GetOne<std::uint64_t>(in);
    if (OtherFormat(magic)) {
      throw std::runtime_error(std::format(
          "{} is a checkpoint of another format, which this program does "
          "not read; set \"restart\": false to start over, or point "
          "checkpoint.directory elsewhere",
          path.string()));
    }
    if (magic != kMagic) return std::nullopt;
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
      kept.relativeSeaLevelChange = GetVector(in);
      kept.displacement = GetVector(in);
      kept.potential = GetVector(in);
      kept.tangentialPotential = GetVector(in);
      kept.toroidalPotential = GetVector(in);
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
/// seconds relative to the present (negative past). Without an ocean the
/// fields and metadata of the state (the initial sea level, the state's
/// inputs, the shorelines) are left out, and `ocean` says there is none.
void WriteHistoryFile(const SeaLevelInput& input,
                      const std::filesystem::path& config,
                      const SurfaceGrid& grid, const std::vector<Node>& history,
                      std::span<const double> initialSeaLevel,
                      std::string_view initialSeaLevelOrigin,
                      const Loaded& world) {
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
  if (run.ocean) {
    file.meta.emplace("ice", world.iceRead);
    file.meta.emplace("sea_level", world.seaLevelRead);
  } else {
    file.meta.emplace("ocean", std::string{kNoOcean});
  }
  if (const auto change = DescribeIceChange(run, "; "); !change.empty()) {
    file.meta.emplace("ice_change", change);
  }
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
  if (run.ocean) {
    file.meta.emplace("shorelines",
                      std::string{NameOf(run.ocean->shorelines)});
    file.meta.emplace("initial_sea_level",
                      std::string{initialSeaLevelOrigin});
  }
  file.meta.emplace("rotation", std::string{NameOf(run.rotation)});
  auto listed = [](const auto& terms) {
    auto out = std::string{};
    for (const auto& term : terms) {
      if (!out.empty()) out += "; ";
      out += Describe(term);
    }
    return out;
  };
  if (!run.forcing.load.empty()) {
    file.meta.emplace("load", listed(run.forcing.load));
  }
  if (!run.forcing.tide.empty()) {
    file.meta.emplace("tide", listed(run.forcing.tide));
  }
  if (!run.forcing.seismic.empty()) {
    file.meta.emplace("seismic", listed(run.forcing.seismic));
  }
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
  put("relative_sea_level_change", gather(&Node::relativeSeaLevelChange), pm::units::LENGTH,
      "sea-level change from the window's start");
  put("displacement", gather(&Node::displacement), pm::units::LENGTH,
      "radial surface displacement");
  put("tangential_potential", gather(&Node::tangentialPotential),
      pm::units::LENGTH,
      "the tangential displacement's potential V (the displacement its "
      "surface gradient)");
  // the toroidal potential where the run has one (seismic sources, a 3-D
  // run), zero at a node without (a 3-D run's radial start)
  if (std::ranges::any_of(history, [](const Node& kept) {
        return !kept.toroidalPotential.empty();
      })) {
    auto values = std::vector<double>{};
    values.reserve(history.size() * fieldSize);
    for (const auto& kept : history) {
      if (kept.toroidalPotential.empty()) {
        values.insert(values.end(), fieldSize, 0.0);
      } else {
        values.insert(values.end(), kept.toroidalPotential.begin(),
                      kept.toroidalPotential.end());
      }
    }
    put("toroidal_potential", std::move(values), pm::units::LENGTH,
        "the toroidal displacement's potential W (the displacement minus the "
        "radial unit vector crossed with its surface gradient), in the frame "
        "of no net rotation of the surface");
  }
  put("potential", gather(&Node::potential), potential,
      "gravitational potential perturbation at the surface");
  if (run.ocean) {
    auto options = pm::SurfaceFieldOptions{};
    options.support = "static";
    options.dimensions = pm::units::LENGTH;
    options.meaning =
        "the initial sea level the run started from; the sea level at a node "
        "is this plus relative_sea_level_change there";
    file.variables.emplace(
        "initial_sea_level",
        pm::SurfaceField(
            "initial_sea_level",
            std::vector<double>(initialSeaLevel.begin(), initialSeaLevel.end()),
            options));
  }
  if (run.Migrating()) {
    put("ocean_function", gather(&Node::oceanFunction), std::nullopt,
        "the ocean function at each node, following flotation");
  }
  if (const auto parent = input.output.surface.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  pm::WriteSurfaceFile(input.output.surface, file);
}

/// What the config was taken to mean, for the summary: the model's
/// dimension, elastic (with migrating shorelines where they are) or
/// viscoelastic with the window, the shorelines and the calibration (or,
/// in their place, that the body has no ocean), and the rotation and
/// moments.
std::string Meaning(const SeaLevelRun& run, const BuiltEarth& earth) {
  auto out = earth.Dimension();
  if (!run.times) {
    if (!run.ocean) {
      out += ", elastic, no ocean";
    } else {
      out += ", elastic (fingerprint)";
      if (run.Migrating()) out += ", shorelines migrating";
    }
  } else {
    const auto& window = *run.times;
    out += std::format(", viscoelastic, {} nodes {} to {} kyr BP",
                       window.nodes, window.start_kyr_bp, window.end_kyr_bp);
    if (!run.ocean) {
      out += ", no ocean";
    } else {
      out += std::format(", shorelines {}", NameOf(run.ocean->shorelines));
      if (run.Calibrating()) out += ", initial sea level calibrated";
    }
  }
  out += std::format(", rotation {}", NameOf(run.rotation));
  if (run.hydrostatic_moments) out += ", moments hydrostatic";
  return out;
}

/// The forcing of a `sea-level` run in the form the library's evolutions
/// take, in the solver's units, each channel a function of the solver's
/// time read on the run's clock (kyr from the window's start through time;
/// the static problem's one node is at zero, where its terms are what they
/// are): the ice change's grounded load, `iceLoad` (given under frozen
/// shorelines alone: under migrating ones the ice is given as its
/// thickness, and a body without an ocean has none), joined by the surface
/// loads in the water budget; the loads outside the water budget, the
/// layer on the solid surface, the tide and the seismic sources. A channel
/// no term gives is absent. `loads` are the run's surface loads (null
/// without any), held by the forcing.
template <class EvolutionForcing>
EvolutionForcing ForcingOf(
    const SeaLevelRun& run, const GRDSolver& solver,
    const std::function<std::vector<double>(double)>& iceLoad,
    std::shared_ptr<const SurfaceLoads> loads) {
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto toKyr = scales.Time() / KYR;
  const auto toLoad = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto ice = static_cast<bool>(iceLoad);
  auto forcing = EvolutionForcing{};
  if (loads) {
    // a node's loads are built once, whichever is asked first
    struct Built {
      std::optional<double> t;
      SurfaceLoads::Node node;
    };
    const auto built = std::make_shared<Built>();
    const auto at = [built, loads, toKyr](double t) -> const SurfaceLoads::Node& {
      if (!built->t || *built->t != t) {
        built->node = loads->At(t * toKyr);
        built->t = t;
      }
      return built->node;
    };
    auto any = [&](auto member) {
      return std::ranges::any_of(run.forcing.load, member);
    };
    const auto inside =
        any([](const SurfaceLoadTerm& term) { return term.water_budget; });
    const auto outside =
        any([](const SurfaceLoadTerm& term) { return !term.water_budget; });
    const auto layered = any(
        [](const SurfaceLoadTerm& term) { return term.moves_solid_surface; });
    if (inside || ice) {
      forcing.directLoad = [at, iceLoad, ice, inside, toLoad](double t) {
        auto out = ice ? iceLoad(t) : std::vector<double>{};
        if (!inside) return out;
        const auto& given = at(t).inside;
        if (out.empty()) out.assign(given.size(), 0.0);
        for (auto i = std::size_t{0}; i < out.size(); ++i) {
          out[i] += given[i] / toLoad;
        }
        return out;
      };
    }
    if (outside) {
      forcing.directLoadOutsideWaterBudget = [at, toLoad](double t) {
        auto out = at(t).outside;
        for (auto& v : out) v /= toLoad;
        return out;
      };
    }
    if (layered) {
      forcing.solidSurfaceChange = [at, length](double t) {
        auto out = at(t).layer;
        for (auto& v : out) v /= length;
        return out;
      };
    }
  } else if (ice) {
    forcing.directLoad = iceLoad;
  }
  if (!run.forcing.tide.empty()) {
    const auto toPotential = scales.Factor(PotentialDimensions);
    forcing.tide = [tide = run.forcing.tide, grid = &solver.Grid(), toKyr,
                    toPotential](double t) {
      auto out = TideAt(tide, *grid, t * toKyr);
      for (auto& c : out) c /= toPotential;
      return out;
    };
  }
  if (!run.forcing.seismic.empty()) {
    auto seismic = Forcing{};
    seismic.seismic = run.forcing.seismic;
    const auto surfaceRadius = solver.Radius() * length;
    const auto energy =
        scales.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
    forcing.sources = [seismic = std::move(seismic), surfaceRadius, toKyr,
                       length, energy](double t) {
      auto sources = SeismicSourcesAt(seismic, t * toKyr, surfaceRadius);
      for (auto& source : sources) {
        source.at.radius /= length;
        for (auto* c : {&source.moment.rr, &source.moment.tt,
                        &source.moment.pp, &source.moment.rt,
                        &source.moment.rp, &source.moment.tp}) {
          *c /= energy;
        }
      }
      return sources;
    };
  }
  return forcing;
}

/// A run through time over either evolution. `make` builds the evolution
/// from its options; `check` inspects each snapshot (refusing an
/// unconverged node). `meaning` heads the summary.
template <class Evolution>
std::string RunWith(
    const std::filesystem::path& config, const SeaLevelInput& input,
    const Json& top, std::string_view meaning, const GRDSolver& solver,
    Loaded& world,
    const std::function<Evolution(typename Evolution::Options)>& make,
    const std::function<void(const typename Evolution::Snapshot&, double)>&
        check) {
  const auto started = std::chrono::steady_clock::now();
  const auto& run = input.run;
  const auto& window = *run.times;
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto times = MakeTimes(run, scales);
  const auto migrating = run.Migrating();
  // the ice's grounded load under frozen shorelines; a body without an
  // ocean has no ice
  const auto directLoad = run.ocean && !migrating
                              ? IceLoadOf(run, solver, world)
                              : std::function<std::vector<double>(double)>{};
  const auto thickness = migrating
                             ? IceThicknessOf(run, solver, world)
                             : std::function<std::vector<double>(double)>{};

  // The forcing beside the ice: the surface loads by their properties, the
  // tide and the seismic sources (ForcingOf). Under frozen shorelines the
  // ice's load joins the loads in the water budget; under migrating ones
  // the ice is given as its thickness, and a change present at the
  // window's start is taken through the forcing's entries (with nothing
  // beside the ice, an empty forcing), so that the start is solved. A run
  // without an ocean is forced beside the ice alone.
  const auto forced = run.forcing.BesideTheIce() ||
                      (migrating && world.ChangedAtStart());
  auto forcing = typename Evolution::Forcing{};
  if (forced) {
    forcing = ForcingOf<typename Evolution::Forcing>(
        run, solver, directLoad,
        run.forcing.load.empty()
            ? nullptr
            : std::make_shared<const SurfaceLoads>(run.forcing, grid,
                                                   top.At("forcing")));
  }

  const auto toPotential = scales.Factor(PotentialDimensions);
  const auto toOmega = 1.0 / scales.Time();
  auto history = std::vector<Node>{};
  auto keep = [&](const typename Evolution::Snapshot& snapshot) {
    auto kept = Node{};
    kept.date_kyr_bp = DateKyrBp(run, scales, snapshot.t);
    check(snapshot, kept.date_kyr_bp);
    kept.relativeSeaLevelChange = snapshot.relativeSeaLevelChange;
    kept.displacement = snapshot.displacement;
    kept.potential = snapshot.potential;
    kept.tangentialPotential = snapshot.tangentialPotential;
    if (!snapshot.toroidalPotential.empty()) {
      // in the frame of no net rotation of the surface: the rigid rotation
      // in the solver's W is its gauge
      kept.toroidalPotential =
          WithoutNetRotation(grid, snapshot.toroidalPotential);
    }
    for (auto& v : kept.relativeSeaLevelChange) v *= length;
    for (auto& v : kept.displacement) v *= length;
    for (auto& v : kept.potential) v *= toPotential;
    for (auto& v : kept.tangentialPotential) v *= length;
    for (auto& v : kept.toroidalPotential) v *= length;
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

  // The initial sea level: the state's at the start, or calibrated
  // against the state's at the end, by an evolution that streams nothing
  // (a resumption reads back the calibrated state its run kept).
  auto origin = std::string{"the state's sea level at the window's start"};
  if (run.Calibrating()) {
    auto restored = std::optional<GRDSolver::State>{};
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
          "calibrated against the state's sea level at the window's end, {} "
          "run(s), rms misfit {:.3g} m before, {:.3g} m after",
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
        if (migrating && forced) {
          static_cast<void>(evolution.ResumeMigrating(
              resumed->checkpoint, remaining, thickness, forcing));
        } else if (forced) {
          static_cast<void>(
              evolution.Resume(resumed->checkpoint, remaining, forcing));
        } else if (migrating) {
          static_cast<void>(evolution.ResumeMigrating(resumed->checkpoint,
                                                      remaining, thickness));
        } else {
          static_cast<void>(
              evolution.Resume(resumed->checkpoint, remaining, directLoad));
        }
      }
    } else if (migrating && forced) {
      static_cast<void>(evolution.RunMigrating(times, thickness, forcing));
    } else if (forced) {
      static_cast<void>(evolution.Run(times, forcing));
    } else if (migrating) {
      static_cast<void>(evolution.RunMigrating(times, thickness));
    } else {
      static_cast<void>(evolution.Run(times, directLoad));
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
    WriteHistoryFile(input, config, grid, history, initial, origin, world);
  }
  return std::format("{}; wrote {}{}{}", meaning, input.output.surface.string(),
                     resumed ? ", resumed" : "", replaced);
}

/// The static problem's world on a run's grid: the state at its date
/// (`ocean.ice` and `ocean.sea_level` or `ocean.topography`, each file
/// with a time axis read at `ocean.date_kyr_bp`, and `ocean.land`), the
/// background state built from it in the solver's units, its ocean
/// function flotation's, the ice change at the date (RunIce, held to what
/// the state holds), SI, and the state's inputs as read. Without an ocean
/// the background is GRDSolver::NoOceanState and the rest is empty.
struct StaticWorld {
  GRDSolver::State background;
  std::vector<double> change;
  std::string iceRead;
  std::string seaLevelRead;
};

StaticWorld LoadStaticWorld(const SeaLevelRun& run, const Json& top,
                            const GRDSolver& solver) {
  if (!run.ocean) {
    auto out = StaticWorld{};
    out.background = solver.NoOceanState();
    return out;
  }
  const auto& source = run.ocean->seaLevel;
  const auto length = solver.SolverScales().Length();
  auto files = InputFiles(solver.Grid());
  CheckStateDate(run, top, files);
  const auto sea = files.Open(source.input, SeaLevelMember(run, top));
  const auto iceFile = files.Open(run.ocean->ice, IceMember(top));
  // a state without a time axis is read at any date: the present's
  const auto date = run.ocean->date_kyr_bp;
  const auto time = -date.value_or(0.0) * KYR;
  const RunIce ice(run, top, files, time);
  ice.CheckHeld(std::span(&time, 1), false);
  // the state's ice, and its sea level: given, or by the flotation
  // construction with the solver's own densities from the topography and
  // the state's ice
  auto iceNd = ice.State();
  auto seaNd =
      SeaLevelFrom(run, top, *sea, iceNd, time, solver.Parameters());
  for (auto& v : iceNd) v /= length;
  for (auto& v : seaNd) v /= length;
  const auto land = ReadLand(run, top, files, time);
  auto out = StaticWorld{};
  out.background =
      solver.MakeState(std::move(iceNd), std::move(seaNd), land.held);
  out.change = ice.ChangeAt(time);
  const auto slice = [&date](const io::FieldHistory& file) {
    return file.Span() ? std::format("{:g} kyr BP", date.value_or(0.0))
                       : std::string{"static"};
  };
  out.iceRead = DescribeInput(run.ocean->ice, slice(*iceFile));
  out.seaLevelRead = std::format("{} {}", source.Quantity(),
                                 DescribeInput(source.input, slice(*sea)));
  if (!land.read.empty()) out.seaLevelRead += ", " + land.read;
  return out;
}

/// The static problem's surface loads on a grid (SurfaceLoads, each term
/// read once, a `field` file at its date), or null without any.
std::shared_ptr<const SurfaceLoads> StaticLoadsOf(const SeaLevelRun& run,
                                                  const Json& top,
                                                  const SurfaceGrid& grid) {
  if (run.forcing.load.empty()) return nullptr;
  return std::make_shared<const SurfaceLoads>(run.forcing, grid,
                                              top.At("forcing"), false);
}

/// The static problem solved as a run of one node of either evolution,
/// built by the caller on the world's background: the solve of the start
/// alone on the unrelaxed body. Under frozen shorelines the linear sea
/// level equation under the forcing, the ice change entering as its
/// grounded load on the state's ocean function, rho_ice (1 - C) dI, in the
/// water budget; under migrating shorelines the nonlinear one, the ice the
/// state's plus the change, whose flotation, with the sea level's, places
/// the shorelines. The surface loads, the tide and the seismic sources
/// join either (ForcingOf). Without an ocean, the linear equation on a
/// body without one under the forcing beside the ice alone: the solid
/// body's response.
template <class Evolution>
typename Evolution::Snapshot SolveStatic(
    const SeaLevelRun& run, const GRDSolver& solver, const StaticWorld& world,
    std::shared_ptr<const SurfaceLoads> loads, Evolution& evolution) {
  const auto length = solver.SolverScales().Length();
  const auto& background = world.background;
  const auto& change = world.change;
  const auto migrating = run.Migrating();
  auto iceLoad = std::function<std::vector<double>(double)>{};
  if (run.ocean && !migrating) {
    auto load = std::vector<double>(change.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - background.oceanFunction[i]) *
                change[i] / length;
    }
    iceLoad = [load = std::move(load)](double) { return load; };
  }
  const auto forcing = ForcingOf<typename Evolution::Forcing>(
      run, solver, iceLoad, std::move(loads));
  const auto one = std::vector<double>{0.0};
  auto snapshots = std::vector<typename Evolution::Snapshot>{};
  if (migrating) {
    auto thickness = background.iceThickness;
    for (auto i = std::size_t{0}; i < thickness.size(); ++i) {
      thickness[i] += change[i] / length;
    }
    snapshots = evolution.RunMigrating(
        one, [thickness = std::move(thickness)](double) { return thickness; },
        forcing);
  } else {
    snapshots = evolution.Run(one, forcing);
  }
  return std::move(snapshots.front());
}

/// A static problem's answer, SI: the sea level change, the radial
/// displacement and the potential perturbation at the surface, the
/// potentials of the tangential displacement (the toroidal one in the
/// frame of no net rotation of the surface, empty where the run has none),
/// the angular velocity perturbation and, under migrating shorelines, the
/// ocean function the solve reaches (empty otherwise).
struct StaticAnswer {
  std::vector<double> relativeSeaLevelChange;
  std::vector<double> displacement;
  std::vector<double> potential;
  std::vector<double> tangentialPotential;
  std::vector<double> toroidalPotential;
  std::array<double, 3> angularVelocity{};
  std::vector<double> oceanFunction;
};

template <class Snapshot>
StaticAnswer AnswerOf(const GRDSolver& solver, Snapshot snapshot) {
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto toPotential = scales.Factor(PotentialDimensions);
  auto out = StaticAnswer{
      .relativeSeaLevelChange = std::move(snapshot.relativeSeaLevelChange),
      .displacement = std::move(snapshot.displacement),
      .potential = std::move(snapshot.potential),
      .tangentialPotential = std::move(snapshot.tangentialPotential),
      .toroidalPotential = {},
      .angularVelocity = snapshot.angularVelocity,
      .oceanFunction = std::move(snapshot.oceanFunction)};
  for (auto& v : out.relativeSeaLevelChange) v *= length;
  for (auto& v : out.displacement) v *= length;
  for (auto& v : out.potential) v *= toPotential;
  for (auto& v : out.tangentialPotential) v *= length;
  if (!snapshot.toroidalPotential.empty()) {
    // in the frame of no net rotation of the surface: the rigid rotation
    // in the solver's W is its gauge
    out.toroidalPotential =
        WithoutNetRotation(solver.Grid(), snapshot.toroidalPotential);
    for (auto& v : out.toroidalPotential) v *= length;
  }
  for (auto& v : out.angularVelocity) v /= scales.Time();
  return out;
}

/// The elastic rheology of the static problem: the deck's elastic
/// structure with no relaxing branch.
Rheology ElasticRheologyOf(const BuiltEarth& earth, const Json& top) {
  auto elastic = Rheology::Create(earth.material, {});
  if (!elastic) Fail(top.At("model").At("deck"), elastic.error().message);
  return std::move(*elastic);
}

/// The static problem on the radial model: the one-node run of the radial
/// evolution (Evolution) on a branchless rheology of the deck's elastic
/// structure, its sea level solves to solver.rtol; the operator and solver
/// it ran on, its world, its answer and the surface loads applied (SI at
/// the grid's points; each kind empty where no term has it).
struct RadialStatic {
  std::unique_ptr<const ViscoelasticOperator> op;
  std::unique_ptr<const GRDSolver> solver;
  StaticWorld world;
  StaticAnswer answer;
  SurfaceLoads::Node applied;
};

RadialStatic SolveRadialStatic(const SeaLevelRun& run, const Json& top,
                               const BuiltEarth& earth) {
  auto out = RadialStatic{};
  out.op = std::make_unique<const ViscoelasticOperator>(
      ElasticRheologyOf(earth, top), run.lmax);
  out.solver = std::make_unique<const GRDSolver>(
      SurfaceGrid(run.lmax), out.op->ElasticLoveNumbers(),
      ParametersOf(run, earth.material));
  const auto& solver = *out.solver;
  out.world = LoadStaticWorld(run, top, solver);
  const auto loads = StaticLoadsOf(run, top, solver.Grid());
  if (loads) out.applied = loads->At(0.0);
  auto options = Evolution::Options{};
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  Evolution evolution(*out.op, solver, out.world.background, options);
  auto snapshot = SolveStatic(run, solver, out.world, loads, evolution);
  if (!snapshot.converged) {
    throw std::runtime_error(
        run.Migrating()
            ? "the sea level equation did not converge (shoreline cells "
              "flipping back and forth defeat the migrating solve's "
              "tolerance); loosen solver.rtol"
            : "the sea level equation did not converge; loosen solver.rtol");
  }
  out.answer = AnswerOf(solver, std::move(snapshot));
  return out;
}

/// A static SI field of a static problem's surface file.
void PutStatic(planetmodel::SurfaceFile& file, std::string name,
               const std::vector<double>& values,
               std::optional<planetmodel::Dimensions> dims,
               std::string meaning) {
  file.variables.emplace(
      name, planetmodel::SurfaceField(name, values,
                                      {.dimensions = dims, .meaning = meaning}));
}

/// Each term of a list as described (Describe), joined by `; `.
template <class Terms>
std::string Listed(const Terms& terms) {
  auto out = std::string{};
  for (const auto& term : terms) {
    if (!out.empty()) out += "; ";
    out += Describe(term);
  }
  return out;
}

/// A static problem's answer as a surface file of static SI fields on the
/// run's grid: the ice change applied, the surface loads applied where
/// there are any, the response (the sea level change, the displacement,
/// the potentials of the tangential displacement and the potential; the
/// angular velocity in the metadata), the state's ocean function and,
/// under migrating shorelines, the one the solve reaches; with the
/// config, model, moduli, the state's ice and sea level as read, the
/// forcing's terms, the shorelines and the rotation in the metadata.
/// Without an ocean the state's fields and metadata (the ice change, the
/// ocean function, the ice, the sea level, the shorelines) are left out,
/// and `ocean` says there is none. The runner adds its own.
planetmodel::SurfaceFile StaticFile(const std::filesystem::path& config,
                                    const SeaLevelRun& run,
                                    const StaticWorld& world,
                                    const SurfaceGrid& grid, std::string title,
                                    const StaticAnswer& answer,
                                    const SurfaceLoads::Node& applied) {
  namespace pm = planetmodel;
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
  if (run.ocean) {
    file.meta.emplace("ice", world.iceRead);
    file.meta.emplace("sea_level", world.seaLevelRead);
  } else {
    file.meta.emplace("ocean", std::string{kNoOcean});
  }
  if (!run.forcing.ice_change.empty()) {
    file.meta.emplace("ice_change", DescribeIceChange(run, "; "));
  }
  if (!run.forcing.load.empty()) {
    file.meta.emplace("load", Listed(run.forcing.load));
  }
  if (!run.forcing.tide.empty()) {
    file.meta.emplace("tide", Listed(run.forcing.tide));
  }
  if (!run.forcing.seismic.empty()) {
    file.meta.emplace("seismic", Listed(run.forcing.seismic));
  }
  if (run.ocean) {
    file.meta.emplace("shorelines",
                      std::string{NameOf(run.ocean->shorelines)});
  }
  file.meta.emplace("rotation", std::string{NameOf(run.rotation)});
  const auto& omega = answer.angularVelocity;
  file.meta.emplace("omega_x_rad_s", std::vector<double>{omega[0]});
  file.meta.emplace("omega_y_rad_s", std::vector<double>{omega[1]});
  file.meta.emplace("omega_z_rad_s", std::vector<double>{omega[2]});
  const auto length = pm::units::LENGTH;
  const auto load = pm::Dimensions{.mass = 1, .length = -2};
  if (run.ocean) {
    PutStatic(file, "ice_change", world.change, length,
              "the ice-thickness change applied");
  }
  if (!applied.inside.empty()) {
    PutStatic(file, "load_in_water_budget", applied.inside, load,
              "the surface loads applied whose mass is exchanged with the "
              "ocean");
  }
  if (!applied.outside.empty()) {
    PutStatic(file, "load_outside_water_budget", applied.outside, load,
              "the surface loads applied that exchange no water with the "
              "ocean");
  }
  if (!applied.layer.empty()) {
    PutStatic(file, "solid_surface_change", applied.layer, length,
              "the height of the layers laid on the solid surface");
  }
  PutStatic(file, "relative_sea_level_change", answer.relativeSeaLevelChange,
            length, "the sea-level change");
  PutStatic(file, "displacement", answer.displacement, length,
            "radial surface displacement");
  PutStatic(file, "tangential_potential", answer.tangentialPotential, length,
            "the tangential displacement's potential V (the displacement its "
            "surface gradient)");
  if (!answer.toroidalPotential.empty()) {
    PutStatic(file, "toroidal_potential", answer.toroidalPotential, length,
              "the toroidal displacement's potential W (the displacement "
              "minus the radial unit vector crossed with its surface "
              "gradient), in the frame of no net rotation of the surface");
  }
  PutStatic(file, "potential", answer.potential,
            pm::Dimensions{.length = 2, .time = -2},
            "gravitational potential perturbation at the surface");
  if (run.ocean) {
    PutStatic(file, "ocean_function",
              std::vector<double>(world.background.oceanFunction.begin(),
                                  world.background.oceanFunction.end()),
              std::nullopt, "the ocean function of the state");
  }
  if (!answer.oceanFunction.empty()) {
    PutStatic(file, "ocean_function_final", answer.oceanFunction,
              std::nullopt,
              "the ocean function the solve reaches, following flotation");
  }
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

/// What forces a run beside its ice, for a summary: `load (<n> term[s])`,
/// `tide (<n> coefficient[s])`, `seismic (<n> source[s])`, or several
/// joined by ` + `; empty without any.
std::string DescribeBesideTheIce(const SeaLevelRun& run) {
  auto forcing = Forcing{};
  for (const auto& term : run.forcing.load) forcing.load.push_back(term.term);
  forcing.tide = run.forcing.tide;
  forcing.seismic = run.forcing.seismic;
  return Summary(forcing);
}

/// What the config was taken to mean, followed by the state's inputs as
/// read (none without an ocean), the ice change's terms and the forcing
/// beside the ice, where there are any.
std::string WithInputs(std::string_view meaning, const SeaLevelRun& run,
                       std::string_view iceRead,
                       std::string_view seaLevelRead) {
  auto out = run.ocean ? std::format("{}; ice {}, {}", meaning, iceRead,
                                     seaLevelRead)
                       : std::string{meaning};
  if (const auto change = DescribeIceChange(run, " + "); !change.empty()) {
    out += "; ice change " + change;
  }
  if (const auto beside = DescribeBesideTheIce(run); !beside.empty()) {
    out += "; " + beside;
  }
  return out;
}

/// The static problem on a radial Earth: the one-node run of the radial
/// evolution (SolveRadialStatic).
std::string RunStatic1D(const std::filesystem::path& config,
                        const SeaLevelInput& input, const Json& top,
                        const BuiltEarth& earth, std::string_view meaning) {
  const auto& run = input.run;
  const auto radial = SolveRadialStatic(run, top, earth);
  const auto file =
      StaticFile(config, run, radial.world, radial.solver->Grid(),
                 "the static sea level problem on the elastic Earth",
                 radial.answer, radial.applied);
  const auto& output = input.output.surface;
  WriteSurface(file, output);
  return std::format(
      "{}; wrote {}",
      WithInputs(meaning, run, radial.world.iceRead, radial.world.seaLevelRead),
      output.string());
}

/// The static problem on a laterally heterogeneous Earth: the one-node
/// run of the 3-D evolution (Evolution3D) on the stepper of a branchless
/// rheology of the deck's elastic structure carrying the lateral moduli,
/// as `response`'s elastic 3-D solve, its start the combined fixed point
/// of a zero-length step over the strain, the ocean load and the rotation
/// (or the sea level closure nested over conjugate-gradient solves,
/// solver.step), distributed over radial segments on several ranks, the
/// cuts balancing the elements the moduli make active. The radial model's
/// answer under the same forcing is written beside it.
std::string RunStatic3D(const std::filesystem::path& config,
                        const SeaLevelInput& input, const Json& top,
                        const BuiltEarth& earth, std::string_view meaning) {
  namespace pm = planetmodel;
  const auto& run = input.run;
  const auto elastic = ElasticRheologyOf(earth, top);
  const Viscoelastic3D op(elastic, run.lmax,
                          [](double, double, double) { return 0.0; },
                          OperatorOptionsOf(run, earth, elastic));
  // the sea level grid blocks its transforms for the operator's cache
  GRDSolver solver(
      SurfaceGrid(run.lmax, {.cache_bytes = op.GetOptions().cache_bytes}),
      op.ElasticLove(), ParametersOf(run, earth.material));
  const auto& grid = solver.Grid();
  const auto world = LoadStaticWorld(run, top, solver);
  const auto loads = StaticLoadsOf(run, top, grid);

  auto options = Evolution3D::Options{};
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  options.coupled_rtol = run.solver.step_rtol;
  options.max_coupled_iterations = run.solver.step_max_iterations;
  Evolution3D evolution(op, solver, world.background, options);
  auto response = Evolution3D::Snapshot{};
  try {
    response = SolveStatic(run, solver, world, loads, evolution);
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
  const auto iterations = response.coupledIterations;
  const auto solves = response.solveIterations;
  const auto residual = response.coupledResidual;
  const auto answer = AnswerOf(solver, std::move(response));

  // The radial model's answer under the same forcing, the first rank's.
  const auto radial = SolveRadialStatic(run, top, earth);
  auto file = StaticFile(
      config, run, world, grid,
      "the static sea level problem on a laterally heterogeneous elastic "
      "Earth",
      answer, loads ? loads->At(0.0) : SurfaceLoads::Node{});
  const auto conjugate = run.solver.step ==
                         Viscoelastic3D::Options::StepSolver::ConjugateGradients;
  file.meta.emplace("lateral", std::vector<long long>{1});
  file.meta.emplace("moduli_reference",
                    std::string{run.model.moduli_file_radial ? "file_radial"
                                                             : "lateral_mean"});
  file.meta.emplace("step_solver", std::string{conjugate ? "conjugate_gradients"
                                                         : "fixed_point"});
  file.meta.emplace("step_iterations", std::vector<long long>{iterations});
  file.meta.emplace("solve_iterations", std::vector<long long>{solves});
  file.meta.emplace("step_residual", std::vector<double>{residual});
  const auto& radialOmega = radial.answer.angularVelocity;
  file.meta.emplace("omega_x_radial_model_rad_s",
                    std::vector<double>{radialOmega[0]});
  file.meta.emplace("omega_y_radial_model_rad_s",
                    std::vector<double>{radialOmega[1]});
  file.meta.emplace("omega_z_radial_model_rad_s",
                    std::vector<double>{radialOmega[2]});
  const auto lengthDims = pm::units::LENGTH;
  PutStatic(file, "relative_sea_level_change_radial_model",
            radial.answer.relativeSeaLevelChange, lengthDims,
            "sea-level change of the radial model under the same forcing");
  PutStatic(file, "displacement_radial_model", radial.answer.displacement,
            lengthDims,
            "radial surface displacement of the radial model under the same "
            "forcing");
  PutStatic(file, "potential_radial_model", radial.answer.potential,
            pm::Dimensions{.length = 2, .time = -2},
            "the radial model's potential perturbation at the surface under "
            "the same forcing");
  const auto& output = input.output.surface;
  WriteSurface(file, output);
  const auto read =
      WithInputs(meaning, run, world.iceRead, world.seaLevelRead);
  return conjugate
             ? std::format(
                   "{}; wrote {} ({} conjugate-gradient iterations in {} "
                   "passes, residual {:.2e})",
                   read, output.string(), solves, iterations, residual)
             : std::format(
                   "{}; wrote {} ({} fixed-point iterations, residual {:.2e})",
                   read, output.string(), iterations, residual);
}

/// The radial model's run through time.
std::string Run1D(const std::filesystem::path& config,
                  const SeaLevelInput& input, const Json& top,
                  const BuiltEarth& earth, std::string_view meaning) {
  const auto& run = input.run;
  const auto rheology = RheologyOf(earth, top);
  ViscoelasticOperator op(rheology, run.lmax);
  GRDSolver solver(SurfaceGrid(run.lmax), op.ElasticLoveNumbers(),
                        ParametersOf(run, earth.material));
  auto world = LoadWorld(run, top, solver);
  return RunWith<Evolution>(
      config, input, top,
      WithInputs(meaning, run, world.iceRead, world.seaLevelRead), solver,
      world,
      [&](Evolution::Options options) {
        return Evolution(op, solver, world.background,
                                 std::move(options));
      },
      [](const Evolution::Snapshot& snapshot, double date_kyr_bp) {
        if (!snapshot.converged) {
          throw std::runtime_error(std::format(
              "the sea level solve did not converge at {:.6g} kyr BP "
              "(shoreline cells flipping back and forth defeat the migrating "
              "solve's tolerance); loosen solver.rtol",
              date_kyr_bp));
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
  GRDSolver solver(
      SurfaceGrid(run.lmax, {.cache_bytes = op.GetOptions().cache_bytes}),
      op.ElasticLove(), ParametersOf(run, earth.material));
  auto world = LoadWorld(run, top, solver);
  return RunWith<Evolution3D>(
      config, input, top,
      WithInputs(meaning, run, world.iceRead, world.seaLevelRead), solver,
      world,
      [&](Evolution3D::Options options) {
        options.coupled_rtol = run.solver.step_rtol;
        options.max_coupled_iterations = run.solver.step_max_iterations;
        return Evolution3D(op, solver, world.background,
                                   std::move(options));
      },
      [](const Evolution3D::Snapshot& snapshot, double date_kyr_bp) {
        if (!snapshot.converged) {
          throw std::runtime_error(std::format(
              "the coupled fixed point or its sea level solve did not "
              "converge at {:.6g} kyr BP (residual {:.3g} after {} "
              "iterations; under migrating shorelines, cells flipping back "
              "and forth can defeat the tolerances); raise "
              "solver.step_max_iterations, or loosen solver.step_rtol or "
              "solver.rtol",
              date_kyr_bp, snapshot.coupledResidual,
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
  if (out.run.problem == Problem::Static) {
    for (const auto* name : {"checkpoint", "restart"}) {
      if (Has(top, name)) {
        Fail(top.At(name),
             std::format("{}: the static problem is one solve, with nothing "
                         "to checkpoint",
                         kNeedsQuasiStatic));
      }
    }
  }
  if (Has(top, "checkpoint")) {
    out.checkpoint = ReadCheckpointBlock(top.At("checkpoint"), directory);
  }
  // what a run forced beside its ice does not do
  if (out.run.forcing.BesideTheIce()) {
    const auto& forcing = top.At("forcing");
    const auto& member = forcing.At(
        !out.run.forcing.load.empty()
            ? "load"
            : (!out.run.forcing.tide.empty() ? "tide" : "seismic"));
    if (out.run.Calibrating()) {
      Fail(member,
           "is a forcing beside the ice, which a calibrated initial sea "
           "level does not take: the calibration's runs are under the ice "
           "alone. Give the state at the window's start (`ocean.at`: "
           "\"start\")");
    }
    if (!out.run.forcing.seismic.empty() && Has(top, "checkpoint")) {
      Fail(forcing.At("seismic"),
           "gives seismic sources, whose run is not checkpointed: the "
           "checkpoint files do not hold the state the sources' toroidal "
           "response is stepped from");
    }
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
  if (run.problem == Problem::Static) {
    // a viscosity file is refused in the static problem when the config is
    // read, so a lateral Earth here is one of lateral moduli
    if (earth.LateralModuli()) {
      return RunStatic3D(config, input, top, earth, meaning);
    }
    RefuseThreeDSolver(top, "the static problem on a radial Earth");
    // one process's run: on several ranks, the first rank's alone
    if (!ranks.First()) return "";
    const RanksOverride alone(Ranks{});
    return RunStatic1D(config, input, top, earth, meaning);
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
