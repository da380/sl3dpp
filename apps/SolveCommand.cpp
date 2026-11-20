/**
 * @file SolveCommand.cpp
 * @brief `sl3d solve`: the sea level equation, the ocean coupled to
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
 * a surface file of static fields. The quasi-static problem is the run
 * through time from a state at an end of the window, its nodes dates
 * before present or times on the run's own clock (Window): the radial
 * model's evolution (Evolution), or, under a lateral viscosity or lateral
 * moduli, the 3-D one (Evolution3D, distributed over radial segments on
 * several ranks) with the radial reference model's run beside it (on the
 * first rank, after the run), written as a surface file with a slice per
 * node (the response, the potentials of the tangential displacement and
 * the loads applied) and restartable from disk checkpoints. Either
 * problem's run is recorded, with a `trajectory` block naming a directory,
 * into the trajectory store there (Trajectory.hpp), which `kernel` sweeps
 * from without running the forward problem again; a run resumed from a
 * checkpoint continues the store its interrupted run left. The `model`
 * block alone decides between 1-D and 3-D. Beside the surface file either
 * problem writes exchange tables of surface coefficients (Outputs.hpp),
 * the displacement's and the potential's and, with an ocean, the sea level
 * change's, and the static problem a volume file, the response inside the
 * body to the total load and potential its solve's body answered. The run
 * is built from the pieces `kernel` shares (Common.hpp), and a config's
 * `kernels` block, which makes it a kernel config too, is checked by
 * kernel's reader and not used.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <planetmodel/json.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "Commands.hpp"
#include "Common.hpp"
#include "Input.hpp"
#include "Outputs.hpp"
#include "Ranks.hpp"
#include "RunHash.hpp"
#include "Serialize.hpp"
#include "SolveFiles.hpp"
#include "Trajectory.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/FieldHistory.hpp"
#include "sl3dpp/io/TrajectoryStore.hpp"

namespace sl3dpp::app {

namespace {

/// One node of the accumulated history, kept across restarts.
using Node = HistoryNode;

/// The outputs' `ocean` metadata of a run without an ocean, and their
/// `gravity` metadata of a run without gravity.
constexpr auto kNoOcean = std::string_view{"none"};
constexpr auto kNoGravity = std::string_view{"none"};

/// A run stopped on purpose at a checkpoint: the wall clock passed the
/// configured limit after the checkpoint was written.
struct WalltimeReached {
  std::size_t node = 0;
};

// -- the checkpoint files: private scratch of a run, machine-local ----------

/// The checkpoints' magic, its last character the format's version: the
/// file of another version (another layout of the state, or of the
/// history file it names) is refused.
constexpr auto kMagic = std::uint64_t{0x534c3344434b5034};  // "SL3DCKP4"
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

/// What a run's checkpoints belong to: the members that decide the
/// computation, and the contents of its input files (AddRunMembers, the
/// 3-D steps' transforms on a `lateral` run alone), so
/// that a restart with another run's checkpoints -- another config, or the
/// same config over changed inputs -- is refused. The members that only
/// steer the run (the checkpoint stride and directory, the wall-time limit,
/// restart, the transforms' cache, the output) are left out, and so are the
/// paths themselves, so that these can change between a run and its
/// resumption and a run directory can move.
std::uint64_t RunHashOf(const SolveInput& input, bool lateral) {
  auto hash = Fnv1a{};
  AddRunMembers(hash, input.run, lateral);
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

/// The history of the nodes kept so far, which a run's checkpoints share:
/// one file beside them, appended to at each checkpoint with the nodes kept
/// since the one before, so that each node's record is written once and a
/// checkpoint holds the run's state alone. A checkpoint names how many
/// records, and how many bytes of the file, belong to it; bytes past those
/// of the newest are a tail that a run stopped between its append and its
/// checkpoint left, which no checkpoint names. The file opens with its
/// magic, the run's hash and whether the run migrates (a record holds the
/// ocean function only then).
constexpr auto kHistoryName = std::string_view{"history.bin"};
constexpr auto kHistoryMagic = std::uint64_t{0x534c334448495331};  // "SL3DHIS1"

/// One node's record in the history file: the response's fields, the
/// angular velocity, a migrating run's ocean function and the step's
/// closure.
void PutRecord(std::ostream& out, const Node& kept, bool migrating) {
  PutOne(out, kept.node);
  PutVector(out, kept.relativeSeaLevelChange);
  PutVector(out, kept.displacement);
  PutVector(out, kept.potential);
  PutVector(out, kept.tangentialPotential);
  PutVector(out, kept.toroidalPotential);
  Put(out, kept.angularVelocity.data(), 3 * sizeof(double));
  if (migrating) PutVector(out, kept.oceanFunction);
  PutOne(out, kept.stepIterations);
  PutOne(out, kept.solveIterations);
  PutOne(out, kept.stepResidual);
}

Node GetRecord(std::istream& in, bool migrating) {
  auto kept = Node{};
  kept.node = GetOne<double>(in);
  kept.relativeSeaLevelChange = GetVector(in);
  kept.displacement = GetVector(in);
  kept.potential = GetVector(in);
  kept.tangentialPotential = GetVector(in);
  kept.toroidalPotential = GetVector(in);
  in.read(reinterpret_cast<char*>(kept.angularVelocity.data()),
          3 * sizeof(double));
  if (migrating) kept.oceanFunction = GetVector(in);
  kept.stepIterations = GetOne<long long>(in);
  kept.solveIterations = GetOne<long long>(in);
  kept.stepResidual = GetOne<double>(in);
  return kept;
}

/// Starts the history file of a run that does not resume: its header
/// alone, whatever file was there.
void StartHistory(const std::filesystem::path& directory, std::uint64_t hash,
                  bool migrating) {
  std::filesystem::create_directories(directory);
  const auto path = directory / kHistoryName;
  auto out = std::ofstream(path, std::ios::binary | std::ios::trunc);
  PutOne(out, kHistoryMagic);
  PutOne(out, hash);
  PutOne(out, static_cast<std::uint64_t>(migrating));
  out.close();
  if (!out) {
    throw std::runtime_error(std::format("cannot write {}", path.string()));
  }
}

/// Appends records to the history file, on disk (the file closed) before
/// a checkpoint names them, and returns the file's length.
std::uint64_t AppendHistory(const std::filesystem::path& directory,
                            std::span<const Node> records, bool migrating) {
  const auto path = directory / kHistoryName;
  auto out = std::ofstream(path, std::ios::binary | std::ios::app);
  for (const auto& kept : records) PutRecord(out, kept, migrating);
  out.close();
  if (!out) {
    throw std::runtime_error(std::format("cannot write {}", path.string()));
  }
  return std::filesystem::file_size(path);
}

/// The first `records` records of the history file in `directory`, which
/// a checkpoint says end `bytes` into it; nothing where the file is
/// another run's, shorter than that, or its records end elsewhere.
std::optional<std::vector<Node>> ReadHistory(
    const std::filesystem::path& directory, std::uint64_t hash, bool migrating,
    std::uint64_t records, std::uint64_t bytes) {
  const auto path = directory / kHistoryName;
  auto error = std::error_code{};
  const auto length = std::filesystem::file_size(path, error);
  if (error || length < bytes) return std::nullopt;
  auto in = std::ifstream(path, std::ios::binary);
  if (!in || GetOne<std::uint64_t>(in) != kHistoryMagic ||
      GetOne<std::uint64_t>(in) != hash ||
      GetOne<std::uint64_t>(in) != static_cast<std::uint64_t>(migrating)) {
    return std::nullopt;
  }
  auto history = std::vector<Node>{};
  for (auto k = std::uint64_t{0}; k < records && in; ++k) {
    history.push_back(GetRecord(in, migrating));
  }
  if (!in || static_cast<std::streamoff>(in.tellg()) !=
                 static_cast<std::streamoff>(bytes)) {
    return std::nullopt;
  }
  return history;
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

/// The node of a finished checkpoint, by its file's name.
std::size_t CheckpointNode(const std::filesystem::path& path) {
  return static_cast<std::size_t>(
      std::stoul(path.filename().string().substr(5, 6)));
}

/// Removes the checkpoint files an earlier run left in the directory, for
/// a run that starts over: its own checkpoints would otherwise be pruned
/// against the earlier run's by name, and a later resumption could pick
/// one of those. The history file and any kept initial state go with
/// them. Returns how many checkpoints were removed.
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
  std::filesystem::remove(directory / kHistoryName);
  std::filesystem::remove(directory / kInitialStateName);
  return removed;
}

/// Readies the directory for a run resuming from the checkpoint of node
/// `node`, whose history ends `bytes` into the history file: the file cut
/// there (dropping a tail no checkpoint names), and the checkpoints of
/// later nodes removed, which name lengths of a history the resumed run
/// writes again (on its own stride, which may be another).
void ResumeHistory(const std::filesystem::path& directory, std::size_t node,
                   std::uint64_t bytes) {
  std::filesystem::resize_file(directory / kHistoryName, bytes);
  for (const auto& path : CheckpointsIn(directory)) {
    if (CheckpointNode(path) > node) std::filesystem::remove(path);
  }
}

/// Writes a checkpoint whole (a temporary renamed into place) and keeps
/// the newest two in the directory. Its history is the first `records`
/// records of the history file, which end `bytes` into it, appended
/// before.
template <class CheckpointT>
void WriteCheckpoint(const std::filesystem::path& directory, std::uint64_t hash,
                     std::size_t node, const CheckpointT& cp,
                     std::uint64_t records, std::uint64_t bytes) {
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
    PutOne(out, records);
    PutOne(out, bytes);
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
  std::uint64_t historyBytes = 0;  ///< where its history ends in the file
};

/// One checkpoint file of this run with the history it names, or nothing
/// when it is not a whole checkpoint (the file short, or the history file
/// beside it short of what the checkpoint names, or another run's);
/// refused when another run wrote the checkpoint.
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
    const auto records = GetOne<std::uint64_t>(in);
    out.historyBytes = GetOne<std::uint64_t>(in);
    if (in) {
      auto history = ReadHistory(path.parent_path(), hash, migrating, records,
                                 out.historyBytes);
      if (history) {
        out.history = std::move(*history);
        return out;
      }
    }
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
    mine.push_back(CheckpointNode(path));
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

/// A run through time as far as it went: the history, SI, every node from
/// the window's start (a resumed run's restored from its checkpoint; on
/// ranks other than the first, each node's closure alone); the
/// initial sea level it started from, SI, and where that came from;
/// whether it resumed, and how many checkpoints a run starting over
/// removed; the node at which it stopped at the wall-time limit, where it
/// did; its surface loads on the solver's grid (null without any); and
/// the summary's account of its trajectory store (empty without one).
struct Reached {
  std::vector<Node> history;
  std::vector<double> initialSeaLevel;
  std::string origin;
  bool resumed = false;
  std::size_t replaced = 0;
  std::optional<std::size_t> stopped;
  /// The node stopped at as the summary states it, of the window's nodes:
  /// `node <n> of <last>`, a jump's left limit there said so, a pair of
  /// the run's own between nodes by its date.
  std::string stoppedAt;
  std::shared_ptr<const SurfaceLoads> loads;
  std::string trajectory;
};

/// The summary's account of a trajectory store a run recorded (a resumed
/// run continuing it from its checkpoint's node, `continued`), or found
/// holding the run complete and left as it is.
std::string RecordedNote(const io::TrajectoryStore& store,
                         std::optional<std::size_t> continued = {}) {
  return std::format(
      "recorded the trajectory to the store {} ({}){}",
      store.Directory().string(), SizeOf(store.Bytes()),
      continued ? std::format(", continuing it from node {}", *continued)
                : std::string{});
}
std::string KeptNote(const io::TrajectoryStore& store) {
  return std::format(
      "the trajectory store {} holds this run complete ({}), left as it is",
      store.Directory().string(), SizeOf(store.Bytes()));
}

/// A run's trajectory store at `trajectory.directory` for a run described
/// as `run` (null: none): the store this run records into, where the
/// directory is empty or holds an incomplete store of the run; nothing,
/// with `note` saying so, where it holds the run complete. A run that
/// resumes from the checkpoint of node `resumed` continues the incomplete
/// store of its interrupted run from that node (ContinueTrajectory),
/// which must hold the run to it: a store it would begin there, the
/// directory empty, is refused, its nodes before the checkpoint not run
/// again.
std::optional<io::TrajectoryStore> StoreToRecord(
    const Json& top, const TrajectoryBlock& block,
    const io::TrajectoryStore::Description* run,
    std::optional<std::size_t> resumed, std::string& note) {
  if (run == nullptr) return std::nullopt;
  const auto standing = SurveyTrajectory(top, block, *run);
  if (standing == io::TrajectoryStore::Standing::Complete) {
    note = KeptNote(OpenTrajectory(top, block, *run));
    return std::nullopt;
  }
  if (!resumed) return RecordTrajectory(top, block, *run);
  if (standing == io::TrajectoryStore::Standing::Empty) {
    Fail(top.At("trajectory").At("directory"),
         std::format("holds no trajectory store, and the run resumes from "
                     "the checkpoint of node {}: a store begun there would "
                     "lack the nodes before it, which are not run again. Set "
                     "\"restart\": false to start the run over and record "
                     "it, or drop `trajectory.directory` to resume without "
                     "recording",
                     *resumed));
  }
  return ContinueTrajectory(top, block, *run, *resumed);
}

}  // namespace

std::string_view StepSolverName(const Run& run, bool threeD) {
  if (!threeD) return "direct";
  return run.solver.step ==
                 Viscoelastic3D::Options::StepSolver::ConjugateGradients
             ? "conjugate_gradients"
             : "fixed_point";
}

namespace {

/// The radial operator of a run on a rheology: with gravity, or without
/// it (`gravity`: "none").
std::unique_ptr<const ViscoelasticOperator> RadialOperatorOf(
    const Run& run, const Rheology& rheology) {
  auto options = ViscoelasticOperator::Options{};
  options.gravitating = run.gravitating;
  return std::make_unique<const ViscoelasticOperator>(rheology, run.lmax,
                                                      options);
}

/// The sea level solver of a radial run, on the operator's elastic Love
/// numbers; without gravity, which has no sea level equation, the solver
/// serves for its grid and scales alone, built on the elastic structure's
/// Love numbers.
std::unique_ptr<const GRDSolver> RadialSolverOf(
    const Run& run, const BuiltEarth& earth, const ViscoelasticOperator& op) {
  if (!run.gravitating) {
    return std::make_unique<const GRDSolver>(
        SurfaceGrid(run.lmax), ComputeLoveNumbers(earth.material, run.lmax),
        ParametersOf(run, earth.material));
  }
  return std::make_unique<const GRDSolver>(SurfaceGrid(run.lmax),
                                           op.ElasticLoveNumbers(),
                                           ParametersOf(run, earth.material));
}

/// The time step's word, as a config gives it.
std::string_view SchemeName(Scheme scheme) {
  switch (scheme) {
    case Scheme::BackwardEuler:
      return "backward_euler";
    case Scheme::Sdirk2:
      return "sdirk2";
    case Scheme::ExponentialEuler:
      return "exponential_euler";
    case Scheme::ExponentialTrapezoid:
      break;
  }
  return "exponential_trapezoid";
}

}  // namespace

std::vector<SurfaceLoads::Node> LoadsAtNodes(const Run& run,
                                             const GRDSolver& solver,
                                             const SurfaceLoads* loads) {
  auto out = std::vector<SurfaceLoads::Node>{};
  if (loads == nullptr) return out;
  const auto& scales = solver.SolverScales();
  const auto clock = SolverClockOf(run, scales);
  for (const auto t : MakeTimes(run, scales)) {
    out.push_back(loads->At(clock.At(t)));
  }
  return out;
}

planetmodel::SurfaceFile HistoryFile(
    const Run& run, const std::filesystem::path& config,
    const SurfaceGrid& grid, std::span<const HistoryNode> history,
    const std::vector<double>& initialSeaLevel, std::string_view origin,
    const Loaded& world, const BuiltEarth& earth,
    const std::vector<SurfaceLoads::Node>& loads,
    const std::vector<HistoryNode>* reference,
    const StepsAgainstRelaxation& steps) {
  namespace pm = planetmodel;
  auto file = pm::SurfaceFile{};
  file.grid = AngularGridOf(grid);
  file.source = std::format("sl3d solve (sl3dpp {})", Version());
  file.title = "viscoelastic GIA histories";
  const auto& window = *run.times;
  auto axis = std::vector<double>{};
  auto nodes = std::vector<double>{};
  auto omega = std::array<std::vector<double>, 3>{};
  auto passes = std::vector<long long>{};
  auto solves = std::vector<long long>{};
  auto residuals = std::vector<double>{};
  for (const auto& kept : history) {
    // a dated run's axis is the calendar's, one counting forwards its own
    // clock's
    axis.push_back(window.Dated() ? -kept.node * KYR
                                  : kept.node * window.unit.seconds);
    nodes.push_back(kept.node);
    for (auto k = 0; k < 3; ++k) omega[k].push_back(kept.angularVelocity[k]);
    passes.push_back(kept.stepIterations);
    solves.push_back(kept.solveIterations);
    residuals.push_back(kept.stepResidual);
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
  if (!run.gravitating) file.meta.emplace("gravity", std::string{kNoGravity});
  if (const auto change = DescribeIceChange(run, "; "); !change.empty()) {
    file.meta.emplace("ice_change", change);
  }
  if (window.Dated()) {
    file.meta.emplace("time",
                      std::string{"seconds relative to the present, negative "
                                  "in the past; dates_kyr_bp lists the "
                                  "nodes"});
    file.meta.emplace("dates_kyr_bp", nodes);
  } else {
    file.meta.emplace(
        "time", std::format("seconds on the run's own clock, a time of t {} "
                            "in the times block at t * {:.17g} s; times lists "
                            "the nodes in time_unit",
                            window.unit.name, window.unit.seconds));
    file.meta.emplace("times", nodes);
    file.meta.emplace("time_unit", std::string{window.unit.name});
  }
  file.meta.emplace("omega_x_rad_s", omega[0]);
  file.meta.emplace("omega_y_rad_s", omega[1]);
  file.meta.emplace("omega_z_rad_s", omega[2]);
  if (!run.model.viscosity.empty()) {
    file.meta.emplace("viscosity", run.model.viscosity.string());
  }
  if (!run.model.burgers.empty()) {
    file.meta.emplace("burgers", DescribeBurgers(run.model.burgers));
  }
  if (!run.model.moduli.empty()) {
    file.meta.emplace("moduli", run.model.moduli.string());
  }
  file.meta.emplace("lateral", std::vector<long long>{earth.Lateral() ? 1 : 0});
  if (run.ocean) {
    file.meta.emplace("shorelines",
                      std::string{NameOf(run.ocean->shorelines)});
    file.meta.emplace("initial_sea_level", std::string{origin});
  }
  file.meta.emplace("rotation", std::string{NameOf(run.rotation)});
  if (!run.forcing.load.empty()) {
    file.meta.emplace("load", Listed(run.forcing.load));
  }
  if (!run.forcing.tide.empty()) {
    file.meta.emplace("tide", Listed(run.forcing.tide));
  }
  if (!run.forcing.seismic.empty()) {
    file.meta.emplace("seismic", Listed(run.forcing.seismic));
  }
  AddForcingCounts(file, run.forcing);
  file.meta.emplace("scheme", std::string{SchemeName(run.solver.scheme)});
  // the steps against the shortest relaxation time of the model as built
  // (nothing of it where nothing relaxes)
  if (steps.time_s > 0.0) {
    file.meta.emplace("longest_step_over_shortest_relaxation_time",
                      std::vector<double>{steps.ratio});
    file.meta.emplace("shortest_relaxation_time_s",
                      std::vector<double>{steps.time_s});
    file.meta.emplace("shortest_relaxation_radius_m",
                      std::vector<double>{steps.radius_m});
    file.meta.emplace("shortest_relaxation_at", steps.where);
  }
  file.meta.emplace("step_solver",
                    std::string{StepSolverName(run, earth.Lateral())});
  file.meta.emplace("step_iterations", passes);
  file.meta.emplace("solve_iterations", solves);
  file.meta.emplace("step_residual", residuals);
  const auto fieldSize = grid.FieldSize();
  const auto potential = pm::Dimensions{.length = 2, .time = -2};
  const auto load = pm::Dimensions{.mass = 1, .length = -2};
  auto gather = [&](std::span<const HistoryNode> from, auto member) {
    auto values = std::vector<double>{};
    values.reserve(from.size() * fieldSize);
    for (const auto& kept : from) {
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
  // the loads applied at each node, the kinds no term has left out
  auto putLoads = [&](std::string name, std::vector<double> SurfaceLoads::Node::*
                                            member,
                      std::optional<pm::Dimensions> dims, std::string meaning) {
    if (loads.empty() || (loads.front().*member).empty()) return;
    auto values = std::vector<double>{};
    values.reserve(loads.size() * fieldSize);
    for (const auto& node : loads) {
      const auto& field = node.*member;
      values.insert(values.end(), field.begin(), field.end());
    }
    put(std::move(name), std::move(values), dims, std::move(meaning));
  };
  putLoads("load", &SurfaceLoads::Node::given, load,
           "the surface loads applied at each node, every term summed as "
           "given (a cap's or a field's values as sampled, a disk's or a "
           "table's synthesised from its coefficients)");
  putLoads("load_in_water_budget", &SurfaceLoads::Node::inside, load,
           "the surface loads applied at each node whose mass is exchanged "
           "with the ocean");
  putLoads("load_outside_water_budget", &SurfaceLoads::Node::outside, load,
           "the surface loads applied at each node that exchange no water "
           "with the ocean");
  putLoads("solid_surface_change", &SurfaceLoads::Node::layer,
           pm::units::LENGTH,
           "the height of the layers laid on the solid surface at each node");
  // without gravity there is no equipotential and no potential
  if (run.gravitating) {
    put("relative_sea_level_change",
        gather(history, &HistoryNode::relativeSeaLevelChange), pm::units::LENGTH,
        "sea-level change from the window's start");
  }
  put("displacement", gather(history, &HistoryNode::displacement), pm::units::LENGTH,
      "radial surface displacement");
  put("tangential_potential", gather(history, &HistoryNode::tangentialPotential),
      pm::units::LENGTH,
      "the tangential displacement's potential V (the displacement its "
      "surface gradient)");
  // the toroidal potential where the run has one (seismic sources, a 3-D
  // run), zero at a node without (a 3-D run's radial start)
  if (std::ranges::any_of(history, [](const HistoryNode& kept) {
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
  if (run.gravitating) {
    put("potential", gather(history, &HistoryNode::potential), potential,
        "gravitational potential perturbation at the surface");
  }
  if (run.ocean) {
    auto options = pm::SurfaceFieldOptions{};
    options.support = "static";
    options.dimensions = pm::units::LENGTH;
    options.meaning =
        "the initial sea level the run started from; the sea level at a node "
        "is this plus relative_sea_level_change there";
    file.variables.emplace(
        "initial_sea_level",
        pm::SurfaceField("initial_sea_level", initialSeaLevel, options));
  }
  if (run.Migrating()) {
    put("ocean_function", gather(history, &HistoryNode::oceanFunction), std::nullopt,
        "the ocean function at each node, following flotation");
  } else if (run.ocean) {
    // frozen shorelines: the state's, which the run holds at every node
    auto options = pm::SurfaceFieldOptions{};
    options.support = "static";
    options.meaning =
        "the ocean function of the state the run started from, which frozen "
        "shorelines hold at every node";
    file.variables.emplace(
        "ocean_function",
        pm::SurfaceField("ocean_function",
                         std::vector<double>(
                             world.background.oceanFunction.begin(),
                             world.background.oceanFunction.end()),
                         options));
  }
  if (reference != nullptr) {
    auto radialOmega = std::array<std::vector<double>, 3>{};
    for (const auto& kept : *reference) {
      for (auto k = 0; k < 3; ++k) {
        radialOmega[k].push_back(kept.angularVelocity[k]);
      }
    }
    file.meta.emplace("omega_x_radial_model_rad_s", radialOmega[0]);
    file.meta.emplace("omega_y_radial_model_rad_s", radialOmega[1]);
    file.meta.emplace("omega_z_radial_model_rad_s", radialOmega[2]);
    put("relative_sea_level_change_radial_model",
        gather(*reference, &HistoryNode::relativeSeaLevelChange), pm::units::LENGTH,
        "sea-level change from the window's start of the radial reference "
        "model (the deck's moduli, the radial reference viscosity) under the "
        "same forcing");
    put("displacement_radial_model", gather(*reference, &HistoryNode::displacement),
        pm::units::LENGTH,
        "radial surface displacement of the radial reference model under the "
        "same forcing");
    put("potential_radial_model", gather(*reference, &HistoryNode::potential),
        potential,
        "the radial reference model's potential perturbation at the surface "
        "under the same forcing");
  }
  return file;
}

namespace {

/// Writes a run through time's surface file (HistoryFile), with the
/// fields `output` asks for added (AddAskedFields).
void WriteHistoryFile(const SolveInput& input,
                      const std::filesystem::path& config,
                      const SurfaceGrid& grid, const Reached& reached,
                      const Loaded& world, const BuiltEarth& earth,
                      const std::vector<SurfaceLoads::Node>& loads,
                      const std::vector<Node>* reference,
                      const StepsAgainstRelaxation& steps) {
  auto file = HistoryFile(input.run, config, grid, reached.history,
                          reached.initialSeaLevel, reached.origin, world,
                          earth, loads, reference, steps);
  AddAskedFields(file, input.output, grid);
  CreateParent(input.output.surface);
  planetmodel::WriteSurfaceFile(input.output.surface, file);
}


/// A run through time over either evolution, as far as it goes. `make`
/// builds the evolution from its options; `check` inspects each snapshot
/// (refusing an unconverged node), given the node as the window labels it
/// (Window::Label). The history's fields are the first rank's, the other
/// ranks keeping each node's closure alone; on several ranks each
/// checkpoints its own radial segment. With `trajectory`, the run's
/// description for the store, the run is recorded into the store of
/// `trajectory.directory` (StoreToRecord), each rank's states into its own
/// subdirectory.
template <class Evolution>
Reached RunThrough(
    const SolveInput& input, const Json& top, const GRDSolver& solver,
    Loaded& world,
    const std::function<Evolution(typename Evolution::Options)>& make,
    const std::function<void(const typename Evolution::Snapshot&,
                             std::string_view)>& check,
    const io::TrajectoryStore::Description* trajectory = nullptr) {
  const auto started = std::chrono::steady_clock::now();
  const auto& run = input.run;
  const auto& window = *run.times;
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  // the library's grid, a node where the forcing jumps repeated and a pair
  // inserted at an onset between nodes; the history keeps the window's
  // nodes, the right limit at a repeat
  const auto steps = StepGridOf(run, scales);
  const auto& times = steps.times;
  const auto migrating = run.Migrating();
  auto out = Reached{};
  // what drives the run: the ice's load or thickness and the forcing
  // beside it (DriveOf)
  const auto drive =
      DriveOf<typename Evolution::Forcing>(run, top, solver, world);
  out.loads = drive.loads;

  const auto& ranks = TheRanks();
  // The history: on the first rank, which writes the outputs, every node's
  // fields and closure; on the others each node's closure alone (the
  // summary's iterations), the evolution keeping no snapshot of its own.
  auto history = std::vector<Node>{};
  auto keep = [&](std::size_t n, const typename Evolution::Snapshot& snapshot) {
    auto kept = Node{};
    kept.node = window.nodes[n];
    check(snapshot, window.Label(window.Axis(n)));
    KeepClosure(kept, snapshot);
    if (!ranks.First()) {
      history.push_back(std::move(kept));
      return;
    }
    KeepFields(kept, snapshot, solver);
    history.push_back(std::move(kept));
  };

  // On several ranks each checkpoints its own radial segment into its own
  // subdirectory, and they resume from the newest node all of them have;
  // each keeps its own history file beside its checkpoints, the fields the
  // first rank's alone.
  // A checkpoint is of a node of the library's grid (its left limits and
  // inserted pairs among them), naming the window's records kept so far
  // in the history file.
  const auto& checkpoint = input.checkpoint;
  const auto checkpointing = !checkpoint.directory.empty();
  const auto directory = RankDirectory(checkpoint.directory);
  const auto hash =
      checkpointing ? RunHashOf(input, std::is_same_v<Evolution, Evolution3D>)
                    : std::uint64_t{0};
  auto resumed = std::optional<ReadBack<typename Evolution::Checkpoint>>{};
  if (checkpointing && input.restart) {
    using CheckpointT = typename Evolution::Checkpoint;
    resumed =
        ranks.Distributed()
            ? ReadCommonCheckpoint<CheckpointT>(directory, hash, migrating)
            : ReadCheckpoint<CheckpointT>(directory, hash, migrating);
  }
  // a run that does not resume writes its history anew: the checkpoints
  // in the directory (a set the ranks do not share among them) name
  // another history, and are cleared, or their later nodes would outrank
  // the new run's in the pruning
  if (checkpointing && !resumed) out.replaced = ClearCheckpoints(directory);
  // the trajectory store, recorded from the first node
  const auto store = StoreToRecord(
      top, input.trajectory, trajectory,
      resumed ? std::optional{resumed->node} : std::nullopt, out.trajectory);
  auto options = typename Evolution::Options{};
  options.scheme = run.solver.scheme;
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  // the land held at each node, where it changes in time
  HoldLand(options, world);

  // The initial sea level: the state's at the start, or calibrated
  // against the state's at the end, by an evolution that streams nothing
  // (a resumption reads back the calibrated state its run kept).
  out.origin = "the state's sea level at the window's start";
  if (run.Calibrating()) {
    auto restored = std::optional<GRDSolver::State>{};
    if (resumed) {
      restored = ReadInitialState(directory, hash, world.background);
    }
    if (restored) {
      StartFromIceHistory(run, solver, world, restored->seaLevel);
      world.background = std::move(*restored);
      out.origin = "calibrated (read back from the checkpoint directory)";
    } else {
      auto calibrating = make(options);
      const auto calibrated =
          CalibrateWorld(run, top, solver, world, calibrating, times);
      if (checkpointing) WriteInitialState(directory, hash, world.background);
      out.origin = std::format(
          "calibrated against the state's sea level at the window's end, {} "
          "run(s), rms misfit {:.3g} m before, {:.3g} m after",
          calibrated.runs, calibrated.before_m, calibrated.after_m);
    }
  }
  // a continued store keeps its interrupted run's start
  const auto continued =
      store && resumed ? std::optional{resumed->node} : std::nullopt;
  if (store && !continued) store->PutStart(world.background);

  const auto base_node = resumed ? resumed->node : std::size_t{0};
  // the evolution counts a resumed run's nodes from its checkpoint's; the
  // left limit of a repeat and an inserted pair are the library's alone,
  // checked as every node is and not kept
  options.on_snapshot = [&](std::size_t node,
                            const typename Evolution::Snapshot& s) {
    const auto k = base_node + node;
    if (steps.Left(k) || steps.Inserted(k)) {
      check(s, WhenOf(run, scales, s.t));
      return;
    }
    keep(steps.window[k], s);
  };
  options.keep_snapshots = false;
  // the history file, holding `logged` of the history's records: cut to
  // the checkpoint a run resumes from, or begun anew
  auto logged = std::size_t{0};
  if (resumed) {
    ResumeHistory(directory, resumed->node, resumed->historyBytes);
    logged = resumed->history.size();
  } else if (checkpointing) {
    StartHistory(directory, hash, migrating);
  }
  if (checkpointing) {
    options.checkpoint_stride = checkpoint.every;
    options.on_checkpoint = [&](std::size_t node,
                                const typename Evolution::Checkpoint& cp) {
      // the records kept since the last checkpoint, on disk before the
      // checkpoint that names them
      const auto bytes = AppendHistory(
          directory, std::span(history).subspan(logged), migrating);
      logged = history.size();
      WriteCheckpoint(directory, hash, base_node + node, cp, logged, bytes);
      const auto done = base_node + node + 1 >= times.size();
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

  // the store's hook follows the history's, so that a node refused there
  // is not recorded
  if (store) store->Attach(options);
  auto evolution = make(std::move(options));
  try {
    if (resumed) {
      history = std::move(resumed->history);
      const auto remaining = std::vector<double>(
          times.begin() + static_cast<std::ptrdiff_t>(resumed->node),
          times.end());
      if (remaining.size() >= 2) {  // a checkpoint at the last node is done
        static_cast<void>(ResumeDriven(evolution, resumed->checkpoint,
                                       remaining, drive, migrating));
      }
    } else {
      static_cast<void>(RunDriven(evolution, times, drive, migrating));
    }
  } catch (const WalltimeReached& at) {
    out.stopped = at.node;
    const auto last = window.Size() - 1;
    if (steps.Inserted(at.node)) {
      out.stoppedAt = std::format(
          "the run's own node at {} (a jump between the window's nodes) of "
          "nodes 0 to {}",
          WhenOf(run, scales, times[at.node]), last);
    } else {
      out.stoppedAt = std::format("node {} of {}{}", steps.window[at.node],
                                  last,
                                  steps.Left(at.node)
                                      ? " (the left limit of a jump there)"
                                      : "");
    }
    if (store) {
      out.trajectory = std::format(
          "the trajectory store {} left incomplete at node {}, the "
          "resumption recording on from there",
          store->Directory().string(), at.node);
    }
    return out;
  }
  if (store) {
    store->Close();
    out.trajectory = RecordedNote(*store, continued);
  }

  out.initialSeaLevel = world.background.seaLevel;
  for (auto& v : out.initialSeaLevel) v *= length;
  out.resumed = resumed.has_value();
  out.history = std::move(history);
  return out;
}

/// `, replacing <k> earlier checkpoint(s)` where a run starting over
/// removed some; empty otherwise.
std::string Replaced(const Reached& reached) {
  return reached.replaced > 0
             ? std::format(", replacing {} earlier checkpoint(s)",
                           reached.replaced)
             : std::string{};
}

/// The summary of a run stopped at the wall-time limit, `read` what the
/// config was taken to mean with the inputs as read.
std::string StoppedSummary(std::string_view read, const SolveInput& input,
                           const Reached& reached) {
  // with restart false, the same config would start over again
  return std::format(
      "{}; checkpointed {} to {} at the wall-time limit{}; {}{}",
      read, reached.stoppedAt,
      input.checkpoint.directory.string(), Replaced(reached),
      input.restart ? "rerun the same config to resume"
                    : "set \"restart\": true (the default) to resume",
      reached.trajectory.empty() ? std::string{} : "; " + reached.trajectory);
}

/// The named scalars and provenance of an exchange table, and its
/// heading: the shared scalars (SharedScalars, of the forcing beside the
/// ice), the sources (`config <path> model <deck> moduli <file or none>`,
/// through time with `viscosity <file or none>`), the state's inputs as
/// read where the run has an ocean (`state ice <ice>, <sea level>`) and a
/// `forcing` line per term, the ice change's first.
struct ExchangeHeader {
  std::string heading;
  std::vector<std::string> provenance;
  std::string scalars;
};

ExchangeHeader ExchangeHeaderOf(const std::filesystem::path& config,
                                const Run& run, const BuiltEarth& earth,
                                bool lateral, std::string_view iceRead,
                                std::string_view seaLevelRead) {
  const auto timed = run.times.has_value();
  auto out = ExchangeHeader{};
  out.heading = std::format(
      "{} surface response from sl3dpp solve ({} {} model, {}{}, rotation "
      "{})",
      timed ? "viscoelastic" : "elastic",
      lateral ? "laterally heterogeneous" : "radial",
      timed ? "viscoelastic" : "elastic",
      run.ocean ? std::format("shorelines {}", NameOf(run.ocean->shorelines))
                : std::string{"no ocean"},
      run.gravitating ? "" : ", no gravity", NameOf(run.rotation));
  auto sources = std::format(
      "config {} model {} moduli {}", config.string(), run.model.deck.string(),
      run.model.moduli.empty() ? "none" : run.model.moduli.string());
  if (timed) {
    sources += std::format(" viscosity {}", run.model.viscosity.empty()
                                                ? "none"
                                                : run.model.viscosity.string());
  }
  out.provenance.push_back(std::move(sources));
  if (run.ocean) {
    out.provenance.push_back(
        std::format("state ice {}, {}", iceRead, seaLevelRead));
  }
  const auto add = [&out](const auto& terms) {
    for (const auto& term : terms) {
      out.provenance.push_back("forcing " + Describe(term));
    }
  };
  add(run.forcing.ice_change);
  add(run.forcing.load);
  add(run.forcing.tide);
  add(run.forcing.seismic);
  const auto& scales = earth.model.ModelScales();
  out.scalars = SharedScalars(
      earth.material.Radius() * scales.Length(),
      earth.material.SurfaceGravity() *
          scales.Factor(Dimensions{.length = 1, .time = -2}),
      run.lmax, lateral, run.forcing);
  return out;
}

/// Writes an exchange table of fields on the grid, SI: the coefficients
/// of the displacement and the potential (U, Phi; U alone without
/// gravity) and, where the run has an ocean, of the relative sea level
/// change (SL, m), `node` holding the node's scalars of a run through
/// time.
void WriteExchange(const std::filesystem::path& path,
                   const ExchangeHeader& header, std::string_view node,
                   const Run& run, const SurfaceGrid& grid,
                   std::span<const double> displacement,
                   std::span<const double> potential,
                   std::span<const double> relativeSeaLevelChange) {
  const auto U = grid.Expand(displacement);
  const auto Phi = run.gravitating ? grid.Expand(potential)
                                   : std::vector<std::complex<double>>{};
  auto columns = DisplacementAndPotential(U, Phi);
  if (!run.gravitating) columns.pop_back();  // no potential without gravity
  auto SL = std::vector<std::complex<double>>{};
  if (run.ocean) {
    SL = grid.Expand(relativeSeaLevelChange);
    columns.push_back({.symbol = "SL",
                       .quantity = "the relative sea level change SL (m)",
                       .coefficients = SL});
  }
  WriteExchangeTable(path, header.heading, header.provenance,
                     header.scalars + std::string{node}, run.lmax, columns);
}

/// A node's scalars in its exchange table: `node` (its index), and its
/// time: in a dated run `time_kyr` (kyr from the window's start, the
/// terms' clock) and `date_kyr_bp` (its date); in one counting forwards
/// `time_<unit>` (its time on the run's clock in its unit).
std::string NodeScalars(const Window& window, std::size_t n) {
  const auto node = window.nodes[n];
  return window.Dated()
             ? std::format("node {}\ntime_kyr {:.17g}\ndate_kyr_bp {:.17g}\n",
                           n, window.Start() - node, node)
             : std::format("node {}\ntime_{} {:.17g}\n", n, window.unit.name,
                           node);
}

/// Writes a run through time's outputs on the first rank, the surface
/// file (WriteHistoryFile, `reference` the radial reference model's
/// history of a 3-D run) and the exchange tables of the nodes chosen, and
/// returns the summary on every rank: `read`, what the config was taken
/// to mean with the inputs as read, then the files, then in a 3-D run the
/// steps' iterations summed, then whether the run resumed and the
/// checkpoints it replaced, the trajectory store's account, and last the
/// run's steps against the shortest relaxation time (`steps`, which the
/// surface file's metadata holds too).
std::string WriteThrough(const std::filesystem::path& config,
                         const SolveInput& input, const BuiltEarth& earth,
                         const GRDSolver& solver, const Loaded& world,
                         const Reached& reached,
                         const std::vector<Node>* reference,
                         const StepsAgainstRelaxation& steps,
                         std::string_view read) {
  const auto& run = input.run;
  const auto& window = *run.times;
  const auto& grid = solver.Grid();
  const auto& history = reached.history;
  const auto first = TheRanks().First();
  if (first) {
    WriteHistoryFile(input, config, grid, reached, world, earth,
                     LoadsAtNodes(run, solver, reached.loads.get()),
                     reference, steps);
  }
  auto wrote = std::format("wrote {}", input.output.surface.string());
  if (!input.output.exchange.empty()) {
    const auto tables = ExchangePaths(input.output, window.Size());
    if (first) {
      const auto header = ExchangeHeaderOf(config, run, earth, earth.Lateral(),
                                           world.iceRead, world.seaLevelRead);
      for (const auto& [n, path] : tables) {
        const auto& kept = history[static_cast<std::size_t>(n)];
        WriteExchange(path, header,
                      NodeScalars(window, static_cast<std::size_t>(n)), run,
                      grid, kept.displacement, kept.potential,
                      kept.relativeSeaLevelChange);
      }
    }
    wrote += std::format(" and {} exchange table{}", tables.size(),
                         tables.size() == 1 ? "" : "s");
  }
  if (earth.Lateral()) {
    // the steps' closures summed, the start's left out
    const auto conjugate =
        run.solver.step ==
        Viscoelastic3D::Options::StepSolver::ConjugateGradients;
    auto total = 0LL;
    for (auto n = std::size_t{1}; n < history.size(); ++n) {
      total += conjugate ? history[n].solveIterations
                         : history[n].stepIterations;
    }
    wrote += std::format(" ({} {} iterations over {} steps)", total,
                         conjugate ? "conjugate-gradient" : "fixed-point",
                         history.size() - 1);
  }
  return std::format(
      "{}; {}{}{}{}; {}", read, wrote, reached.resumed ? ", resumed" : "",
      Replaced(reached),
      reached.trajectory.empty() ? std::string{} : "; " + reached.trajectory,
      steps.text);
}

/// The static problem on the radial model: the one-node run of the radial
/// evolution (Evolution) on a branchless rheology of the deck's elastic
/// structure, its sea level solves to solver.rtol, recording the total
/// load and the potential it answers where `recorded` (the volume file's
/// forcing), and recorded into the trajectory store of `trajectory`'s
/// directory where given (StoreToRecord); the operator and solver it ran
/// on, its world, its answer, the surface loads applied (SI at the grid's
/// points; each kind empty where no term has it), each `field` load's
/// input as read, and the summary's account of its trajectory store.
struct RadialStatic {
  std::unique_ptr<const ViscoelasticOperator> op;
  std::unique_ptr<const GRDSolver> solver;
  StaticWorld world;
  StaticAnswer answer;
  SurfaceLoads::Node applied;
  std::vector<std::string> inputs;
  std::string trajectory;
};

RadialStatic SolveRadialStatic(const Run& run, const Json& top,
                               const BuiltEarth& earth, bool recorded = false,
                               const TrajectoryBlock& trajectory = {}) {
  auto out = RadialStatic{};
  out.op = RadialOperatorOf(run, ElasticRheologyOf(earth, top));
  out.solver = RadialSolverOf(run, earth, *out.op);
  const auto& solver = *out.solver;
  out.world = LoadStaticWorld(run, top, solver);
  const auto loads = StaticLoadsOf(run, top, solver.Grid());
  if (loads) {
    out.applied = loads->At(0.0);
    out.inputs = loads->Inputs();
  }
  auto options = Evolution::Options{};
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  if (recorded) options.record_stride = 1;
  const auto described =
      trajectory.directory.empty()
          ? std::optional<io::TrajectoryStore::Description>{}
          : TrajectoryOf(run, 1, std::vector<double>{0.0}, solver);
  const auto store =
      StoreToRecord(top, trajectory, described ? &*described : nullptr,
                    std::nullopt, out.trajectory);
  if (store) {
    store->PutStart(out.world.background);
    store->Attach(options);
  }
  Evolution evolution(*out.op, solver, out.world.background, options);
  auto snapshot =
      std::move(SolveStatic(run, solver, out.world, loads, evolution).front());
  if (!snapshot.converged) {
    throw std::runtime_error(
        run.Migrating()
            ? "the sea level equation did not converge (shoreline cells "
              "flipping back and forth defeat the migrating solve's "
              "tolerance); loosen solver.rtol"
            : "the sea level equation did not converge; loosen solver.rtol");
  }
  if (store) {
    store->Close();
    out.trajectory = RecordedNote(*store);
  }
  out.answer = AnswerOf(solver, std::move(snapshot));
  return out;
}

/// A recorded static answer's total load and potential, non-dimensional
/// under the model's scales (what the radial reference and the 3-D
/// operator take).
std::pair<std::vector<std::complex<double>>, std::vector<std::complex<double>>>
NonDimensionalForcing(const BuiltEarth& earth, const StaticAnswer& answer) {
  const auto& scales = earth.model.ModelScales();
  auto load = answer.loadCoefficients;
  auto tide = answer.potentialCoefficients;
  const auto toLoad = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto toPotential = scales.Factor(PotentialDimensions);
  for (auto& c : load) c /= toLoad;
  for (auto& c : tide) c /= toPotential;
  return {std::move(load), std::move(tide)};
}

/// Adds per-degree forces into a sum.
void AddForces(ElasticPerturbation3D::Forces& to,
               ElasticPerturbation3D::Forces from) {
  for (auto& [l, f] : from) {
    auto found = to.find(l);
    if (found == to.end()) {
      to.emplace(l, std::move(f));
    } else {
      for (auto i = std::size_t{0}; i < f.size(); ++i) found->second[i] += f[i];
    }
  }
}

/// The static problem's response inside the body on a radial Earth: the
/// radial reference's systems solving the total load and the potential
/// the static solve's body answered (a recorded answer), the potential's
/// degree-0 part the spherical mean of a centrifugal change, as the
/// stepper takes it.
VolumeFields RadialVolume(const BuiltEarth& earth,
                          const ElasticPerturbation3D& reference,
                          const StaticAnswer& answer) {
  auto [load, tide] = NonDimensionalForcing(earth, answer);
  auto mean = std::vector<std::complex<double>>(tide.size());
  mean[SurfaceGrid::Index(0, 0)] =
      std::exchange(tide[SurfaceGrid::Index(0, 0)], 0.0);
  auto forces = reference.LoadForces(load);
  AddForces(forces, reference.TideForces(tide));
  AddForces(forces, reference.TideForces(mean, 2));
  return VolumeFieldsOf(reference, reference.Solve(forces, {}), nullptr);
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

}  // namespace

planetmodel::SurfaceFile StaticFile(const std::filesystem::path& config,
                                    const Run& run,
                                    const StaticWorld& world,
                                    const SurfaceGrid& grid, std::string title,
                                    const StaticAnswer& answer,
                                    const SurfaceLoads::Node& applied) {
  namespace pm = planetmodel;
  auto file = pm::SurfaceFile{};
  file.grid = AngularGridOf(grid);
  file.source = std::format("sl3d solve (sl3dpp {})", Version());
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
  if (!run.gravitating) file.meta.emplace("gravity", std::string{kNoGravity});
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
  AddForcingCounts(file, run.forcing);
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
  if (!applied.given.empty()) {
    PutStatic(file, "load", applied.given, load,
              "the surface loads applied, every term summed as given (a "
              "cap's or a field's values as sampled, a disk's or a table's "
              "synthesised from its coefficients)");
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
  // without gravity there is no equipotential and no potential
  if (run.gravitating) {
    PutStatic(file, "relative_sea_level_change",
              answer.relativeSeaLevelChange, length, "the sea-level change");
  }
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
  if (run.gravitating) {
    PutStatic(file, "potential", answer.potential,
              pm::Dimensions{.length = 2, .time = -2},
              "gravitational potential perturbation at the surface");
  }
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

namespace {

/// Writes a static problem's surface file, with the fields `output` asks
/// for added (AddAskedFields), and the members of `output` beside it, the
/// exchange table and the volume file (`volume`, its fields; empty without
/// one), and returns `wrote <surface>[ and <exchange>][ and <volume>]`.
std::string WriteStatic(const std::filesystem::path& config,
                        const SolveInput& input, const BuiltEarth& earth,
                        planetmodel::SurfaceFile file,
                        const SurfaceGrid& grid, const StaticAnswer& answer,
                        const StaticWorld& world, bool lateral,
                        const ElasticPerturbation3D* reference,
                        const VolumeFields& volume) {
  const auto& run = input.run;
  const auto& output = input.output;
  AddAskedFields(file, output, grid);
  CreateParent(output.surface);
  planetmodel::WriteSurfaceFile(output.surface, file);
  auto wrote = std::format("wrote {}", output.surface.string());
  if (!output.exchange.empty()) {
    WriteExchange(
        output.exchange,
        ExchangeHeaderOf(config, run, earth, lateral, world.iceRead,
                         world.seaLevelRead),
        "", run, grid, answer.displacement, answer.potential,
        answer.relativeSeaLevelChange);
    wrote += std::format(" and {}", output.exchange.string());
  }
  if (!output.volume.empty()) {
    WriteVolumeFile(output.volume, *reference, volume,
                    earth.material.Mesh(), earth.model, "sl3d solve");
    wrote += std::format(" and {}", output.volume.string());
  }
  return wrote;
}

/// The static problem on a radial Earth: the one-node run of the radial
/// evolution (SolveRadialStatic), its closure in the metadata as a 3-D
/// solve's (`direct`, zero iterations), and the response inside the body
/// where asked (RadialVolume).
std::string RunStatic1D(const std::filesystem::path& config,
                        const SolveInput& input, const Json& top,
                        const BuiltEarth& earth, std::string_view meaning) {
  const auto& run = input.run;
  const auto volumed = !input.output.volume.empty();
  const auto radial =
      SolveRadialStatic(run, top, earth, volumed, input.trajectory);
  auto file =
      StaticFile(config, run, radial.world, radial.solver->Grid(),
                 "the static sea level problem on the elastic Earth",
                 radial.answer, radial.applied);
  file.meta.emplace("lateral", std::vector<long long>{0});
  file.meta.emplace("step_solver", std::string{StepSolverName(run, false)});
  file.meta.emplace("step_iterations", std::vector<long long>{0});
  file.meta.emplace("solve_iterations", std::vector<long long>{0});
  file.meta.emplace("step_residual", std::vector<double>{0.0});
  auto reference = std::unique_ptr<const ElasticPerturbation3D>{};
  auto volume = VolumeFields{};
  if (volumed) {
    reference = std::make_unique<const ElasticPerturbation3D>(
        earth.material, run.lmax,
        ElasticPerturbation3D::Options{.cache_bytes = run.solver.cache_mb << 20});
    volume = RadialVolume(earth, *reference, radial.answer);
  }
  const auto wrote =
      WriteStatic(config, input, earth, std::move(file), radial.solver->Grid(),
                  radial.answer, radial.world, false, reference.get(), volume);
  return std::format("{}; {}{}",
                     WithInputs(meaning, run, radial.world.iceRead,
                                radial.world.seaLevelRead, radial.inputs),
                     wrote,
                     radial.trajectory.empty() ? std::string{}
                                               : "; " + radial.trajectory);
}

/// The static problem on a laterally heterogeneous Earth: the one-node
/// run of the 3-D evolution (Evolution3D) on the stepper of a branchless
/// rheology of the deck's elastic structure carrying the lateral moduli
/// (the 3-D static problem of docs/adjoints.md §13j), its start the
/// combined fixed point of a zero-length step over the strain, the ocean
/// load and the rotation (or the sea level closure nested over
/// conjugate-gradient solves, solver.step), distributed over radial
/// segments on several ranks, the cuts balancing the elements the moduli
/// make active. The radial model's answer under the same forcing is
/// written beside it. The response inside the body, where asked, is the
/// operator's conjugate-gradient elastic solve under the total load and
/// the potential the solve's body answered, on several ranks each rank's
/// fields over its own segment gathered to the first.
std::string RunStatic3D(const std::filesystem::path& config,
                        const SolveInput& input, const Json& top,
                        const BuiltEarth& earth, std::string_view meaning) {
  namespace pm = planetmodel;
  const auto& run = input.run;
  const auto volumed = !input.output.volume.empty();
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
  if (volumed) options.record_stride = 1;
  // the trajectory store, every rank's state in its own subdirectory
  auto note = std::string{};
  const auto described =
      input.trajectory.directory.empty()
          ? std::optional<io::TrajectoryStore::Description>{}
          : TrajectoryOf(run, 1, std::vector<double>{0.0}, solver, op);
  const auto store =
      StoreToRecord(top, input.trajectory, described ? &*described : nullptr,
                    std::nullopt, note);
  if (store) {
    store->PutStart(world.background);
    store->Attach(options);
  }
  Evolution3D evolution(op, solver, world.background, options);
  auto response = Evolution3D::Snapshot{};
  try {
    response =
        std::move(SolveStatic(run, solver, world, loads, evolution).front());
  } catch (const std::runtime_error& error) {
    throw std::runtime_error(std::format(
        "{}; loosen solver.step_rtol or raise solver.step_max_iterations",
        error.what()));
  }
  if (!response.converged) {
    throw std::runtime_error(
        "the sea level equation did not converge; loosen solver.rtol");
  }
  if (store) {
    store->Close();
    note = RecordedNote(*store);
  }
  const auto iterations = response.coupledIterations;
  const auto solves = response.solveIterations;
  const auto residual = response.coupledResidual;
  const auto answer = AnswerOf(solver, std::move(response));

  // The response inside the body: the operator's elastic solve under the
  // total load and potential (the first rank's, which every rank's
  // snapshot holds), at the radial reference's memory nodes, each rank's
  // own gathered to the first.
  auto reference = std::unique_ptr<const ElasticPerturbation3D>{};
  auto volume = VolumeFields{};
  if (volumed) {
    auto [load, tide] = NonDimensionalForcing(earth, answer);
    const auto x = op.SolveElastic(op.ZeroState(), ColumnsOf(std::move(load), 1.0),
                                   ColumnsOf(std::move(tide), 1.0));
    if (const auto info = op.LastElasticSolve(); !info.converged) {
      throw std::runtime_error(std::format(
          "the 3D elastic solve of the response inside the body did not "
          "converge: residual {} after {} iterations",
          info.residual, info.iterations));
    }
    reference = std::make_unique<const ElasticPerturbation3D>(
        earth.material, run.lmax,
        ElasticPerturbation3D::Options{.cache_bytes = op.GetOptions().cache_bytes});
    volume = GatheredVolumeFields(*reference, x, &*earth.moduli,
                                  op.Coupling().GetOptions().elements);
  }
  if (!TheRanks().First()) return "";

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
  file.meta.emplace("step_solver", std::string{StepSolverName(run, true)});
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
  const auto wrote =
      WriteStatic(config, input, earth, std::move(file), grid, answer, world,
                  true, reference.get(), volume);
  const auto read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                               InputsOf(loads.get()));
  const auto stored = note.empty() ? std::string{} : "; " + note;
  return conjugate
             ? std::format(
                   "{}; {} ({} conjugate-gradient iterations in {} "
                   "passes, residual {:.2e}){}",
                   read, wrote, solves, iterations, residual, stored)
             : std::format(
                   "{}; {} ({} fixed-point iterations, residual {:.2e}){}",
                   read, wrote, iterations, residual, stored);
}

/// The radial model's run through time.
std::string Run1D(const std::filesystem::path& config,
                  const SolveInput& input, const Json& top,
                  const BuiltEarth& earth, std::string_view meaning) {
  const auto& run = input.run;
  const auto rheology = RheologyOf(earth, top);
  const auto ownedOperator = RadialOperatorOf(run, rheology);
  const auto& op = *ownedOperator;
  const auto ownedSolver = RadialSolverOf(run, earth, op);
  const auto& solver = *ownedSolver;
  auto world = LoadWorld(run, top, solver);
  const auto trajectory =
      input.trajectory.directory.empty()
          ? std::optional<io::TrajectoryStore::Description>{}
          : TrajectoryOf(run, input.trajectory.record_stride,
                         StepGridOf(run, solver.SolverScales()).times, solver);
  const auto reached = RunThrough<Evolution>(
      input, top, solver, world,
      [&](Evolution::Options options) {
        return Evolution(op, solver, world.background, std::move(options));
      },
      [&run](const Evolution::Snapshot& snapshot, std::string_view when) {
        Check1D(snapshot, when, run.solver.rtol);
      },
      trajectory ? &*trajectory : nullptr);
  // the run's longest step over the shortest relaxation time of the model
  // as built: what its accuracy is read against
  const auto steps =
      StepsAgainst(run, solver, StepGridOf(run, solver.SolverScales()).times,
                   op.ShortestRelaxation(), earth, rheology.BranchCount());
  const auto read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                               InputsOf(reached.loads.get()));
  if (reached.stopped) {
    return StoppedSummary(read, input, reached) + "; " + steps.text;
  }
  return WriteThrough(config, input, earth, solver, world, reached, nullptr,
                      steps, read);
}

/// The radial reference model of a 3-D run through time under the same
/// forcing: the radial Maxwell rheology the 3-D run is about (RheologyOf:
/// the deck's moduli, the viscosity file's radial reference or the deck's
/// column), from its own world (its own calibration, where the run
/// calibrates), without checkpoints. Its history, SI.
std::vector<Node> RadialReference(const SolveInput& input, const Json& top,
                                  const BuiltEarth& earth) {
  auto alone = input;
  alone.checkpoint = CheckpointBlock{};
  alone.trajectory = TrajectoryBlock{};
  const auto& run = alone.run;
  const auto rheology = RheologyOf(earth, top);
  ViscoelasticOperator op(rheology, run.lmax);
  GRDSolver solver(SurfaceGrid(run.lmax), op.ElasticLoveNumbers(),
                   ParametersOf(run, earth.material));
  auto world = LoadWorld(run, top, solver);
  return RunThrough<Evolution>(
             alone, top, solver, world,
             [&](Evolution::Options options) {
               return Evolution(op, solver, world.background,
                                std::move(options));
             },
             [&run](const Evolution::Snapshot& snapshot,
                    std::string_view when) {
               Check1D(snapshot, when, run.solver.rtol);
             })
      .history;
}

/// The 3-D run through time: a lateral viscosity, lateral moduli or both,
/// distributed over radial segments on several ranks; the first rank then
/// steps the radial reference model beside it (RadialReference) and
/// writes the outputs, and every rank returns the summary.
std::string Run3D(const std::filesystem::path& config,
                  const SolveInput& input, const Json& top,
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
  const auto trajectory =
      input.trajectory.directory.empty()
          ? std::optional<io::TrajectoryStore::Description>{}
          : TrajectoryOf(run, input.trajectory.record_stride,
                         StepGridOf(run, solver.SolverScales()).times, solver,
                         op);
  const auto reached = RunThrough<Evolution3D>(
      input, top, solver, world,
      [&](Evolution3D::Options options) {
        options.coupled_rtol = run.solver.step_rtol;
        options.max_coupled_iterations = run.solver.step_max_iterations;
        return Evolution3D(op, solver, world.background, std::move(options));
      },
      [&](const Evolution3D::Snapshot& snapshot, std::string_view when) {
        Check3D(snapshot, when, run.solver.step_rtol);
      },
      trajectory ? &*trajectory : nullptr);
  // the run's longest step over the shortest relaxation time over the
  // volume (collective: every rank asks)
  const auto steps =
      StepsAgainst(run, solver, StepGridOf(run, solver.SolverScales()).times,
                   op.ShortestRelaxation(), earth, rheology.BranchCount());
  const auto read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                               InputsOf(reached.loads.get()));
  if (reached.stopped) {
    return StoppedSummary(read, input, reached) + "; " + steps.text;
  }
  // the radial reference is one process's run, the first rank's
  auto reference = std::vector<Node>{};
  if (TheRanks().First()) {
    const RanksOverride alone(Ranks{});
    reference = RadialReference(input, top, earth);
  }
  return WriteThrough(config, input, earth, solver, world, reached, &reference,
                      steps, read);
}

}  // namespace

SolveInput ReadSolveMembers(const Json& top, Run run,
                            const std::filesystem::path& directory,
                            bool outputRequired) {
  auto out = SolveInput{};
  out.run = std::move(run);
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
    if (!out.run.forcing.seismic.empty()) {
      Fail(top.At("forcing").At("seismic"),
           "gives seismic sources, whose run is not checkpointed: the "
           "checkpoint files do not hold the state the sources' toroidal "
           "response is stepped from");
    }
  }
  if (Has(top, "restart")) out.restart = top.At("restart").AsBool();
  if (!outputRequired && !Has(top, "output")) return out;
  const auto& output = top.At("output");
  out.output = ReadOutputBlock(output, directory);
  CheckOutputMembers(
      output, out.output,
      out.run.times ? std::optional{out.run.times->Size()} : std::nullopt,
      "a run through time",
      "is the static problem's alone: a run through time writes no volume "
      "file");
  if (!out.output.volume.empty() && !out.run.forcing.seismic.empty()) {
    Fail(output.At("volume"),
         "is not written under seismic sources (`forcing.seismic`): the "
         "response inside the body is solved from the static solve's loads "
         "and potentials alone; drop the member or the sources");
  }
  if (!out.output.volume.empty() && !out.run.gravitating) {
    Fail(output.At("volume"),
         "is not written without gravity (`gravity`: \"none\"): the response "
         "inside the body is solved by the systems with gravity, and their "
         "counterpart without it is not built; drop the member");
  }
  return out;
}

SolveInput ReadSolveInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  auto allowed = RunMemberNames();
  for (const auto name : {"checkpoint", "restart", "output", "kernels",
                          "trajectory", "perturbation", "tangent"}) {
    allowed.push_back(name);
  }
  AllowTopLevel(top, allowed);
  const auto directory = config.parent_path();
  auto out = ReadSolveMembers(top, ReadRun(top, directory), directory, true);
  out.trajectory = ReadTrajectoryBlock(top, out.run, directory);
  // the blocks of a config that serves `kernel` and `tangent` too, checked
  // here so that a fault in one is found before the run; solve does not
  // use them
  CheckOtherBlocks(top, out.run, directory, "solve");
  return out;
}

std::string RunSolve(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  auto input = ReadSolveInput(config);
  const auto earth = BuildEarth(input.run, top);
  PlaceSeismicSources(input.run, earth, top);
  FindFileJumps(input.run);
  const auto& run = input.run;
  if (!run.gravitating && earth.Lateral()) {
    const auto& model = top.At("model");
    Fail(earth.LateralModuli() ? model.At("moduli").At("file")
                               : model.At("viscosity").At("file"),
         "varies laterally, which makes the run 3-D, and a 3-D run without "
         "gravity (`gravity`: \"none\") is not built: give a radial "
         "model");
  }
  const auto meaning = Meaning(run, earth);
  const auto& ranks = TheRanks();
  if (run.problem == Problem::Static) {
    // a viscosity file is refused in the static problem when the config is
    // read, so a lateral Earth here is one of lateral moduli
    if (earth.LateralModuli()) {
      return RunStatic3D(config, input, top, earth, meaning);
    }
    // the grids' members pass where the kernels are lateral: the kernel
    // sweep's, unused here, as the `kernels` block is
    RefuseThreeDSolver(top, "the static problem on a radial Earth",
                       LateralKernelsAsked(top));
    // one process's run: on several ranks, the first rank's alone
    if (!ranks.First()) return "";
    const RanksOverride alone(Ranks{});
    return RunStatic1D(config, input, top, earth, meaning);
  }
  if (!earth.Lateral()) {
    RefuseThreeDSolver(top, "a 1-D run (a radial model)",
                       LateralKernelsAsked(top));
    if (!ranks.First()) return "";
    const RanksOverride alone(Ranks{});
    return Run1D(config, input, top, earth, meaning);
  }
  return Run3D(config, input, top, earth, meaning);
}

}  // namespace sl3dpp::app
