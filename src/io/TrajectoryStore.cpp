#include "sl3dpp/io/TrajectoryStore.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

#include "sl3dpp/Version.hpp"
#include "sl3dpp/io/Binary.hpp"

namespace sl3dpp::io {

namespace {

static_assert(std::numeric_limits<double>::is_iec559,
              "the store holds IEEE 754 doubles");

namespace fs = std::filesystem;
using Kind = TrajectoryStoreError::Kind;

// -- the on-disk form (docs/trajectory-store.md) -----------------------------

constexpr auto kManifestTag = std::string_view{"SL3DTRAJ"};
constexpr auto kRecordTag = std::string_view{"SL3DTREC"};
/// Written as the machine's bytes: a little-endian file reads it back as
/// itself, one of the other order as its byte swap.
constexpr auto kByteOrder = std::uint64_t{0x0102030405060708};
constexpr auto kManifestName = std::string_view{"trajectory.manifest"};
constexpr auto kStartName = std::string_view{"start.record"};
constexpr auto kSurfaceSuffix = std::string_view{".surface"};
constexpr auto kStateSuffix = std::string_view{".state"};
constexpr auto kPartialSuffix = std::string_view{".partial"};

/// The kinds of record, and the bytes of a record's header: its tag, kind,
/// node, rank and payload length.
constexpr auto kStartKind = std::uint64_t{1};
constexpr auto kSurfaceKind = std::uint64_t{2};
constexpr auto kStateKind = std::uint64_t{3};
constexpr auto kHeaderBytes = std::uint64_t{8 + 4 * 8};

[[noreturn]] void Throw(Kind kind, const std::string& message) {
  throw TrajectoryStoreError(kind, message);
}

/// The store needs a little-endian machine, its files' byte order.
void CheckMachine() {
  if constexpr (std::endian::native != std::endian::little) {
    Throw(Kind::ByteOrder,
          "the trajectory store is little-endian, and this machine is not");
  }
}

std::string NodeName(std::size_t node, std::string_view suffix) {
  return std::format("node_{:06}{}", node, suffix);
}

std::string Partial(std::string_view name) {
  return std::format(".{}{}", name, kPartialSuffix);
}

/// Whether a name is `node_` and six or more digits and `suffix`, or that
/// name's partial, `.<name>.partial`.
bool IsNodeName(std::string_view name, std::string_view suffix) {
  if (name.starts_with('.') && name.ends_with(kPartialSuffix)) {
    name = name.substr(1, name.size() - 1 - kPartialSuffix.size());
  }
  if (!name.starts_with("node_") || !name.ends_with(suffix)) return false;
  const auto digits = name.substr(5, name.size() - 5 - suffix.size());
  return digits.size() >= 6 &&
         digits.find_first_not_of("0123456789") == std::string_view::npos;
}

bool IsRankName(std::string_view name) {
  if (!name.starts_with("rank_")) return false;
  const auto digits = name.substr(5);
  return digits.size() >= 4 &&
         digits.find_first_not_of("0123456789") == std::string_view::npos;
}

/// What a directory holds: the files and subdirectories a store writes,
/// and anything else.
struct Contents {
  bool manifest = false;
  std::vector<fs::path> own;      ///< files, then the rank directories
  std::vector<fs::path> foreign;  ///< anything else
};

Contents Classify(const fs::path& directory) {
  auto out = Contents{};
  auto rankDirectories = std::vector<fs::path>{};
  for (const auto& entry : fs::directory_iterator(directory)) {
    const auto name = entry.path().filename().string();
    if (entry.is_regular_file()) {
      if (name == kManifestName) {
        out.manifest = true;
        out.own.push_back(entry.path());
      } else if (name == Partial(kManifestName) || name == kStartName ||
                 name == Partial(kStartName) ||
                 IsNodeName(name, kSurfaceSuffix)) {
        out.own.push_back(entry.path());
      } else {
        out.foreign.push_back(entry.path());
      }
    } else if (entry.is_directory() && IsRankName(name)) {
      for (const auto& inner : fs::directory_iterator(entry.path())) {
        if (inner.is_regular_file() &&
            IsNodeName(inner.path().filename().string(), kStateSuffix)) {
          out.own.push_back(inner.path());
        } else {
          out.foreign.push_back(inner.path());
        }
      }
      rankDirectories.push_back(entry.path());
    } else {
      out.foreign.push_back(entry.path());
    }
  }
  out.own.insert(out.own.end(), rankDirectories.begin(), rankDirectories.end());
  return out;
}

/// A bounded source: a stream with the bytes its record has left, every
/// read and every count held to them.
struct Bounded {
  std::istream* in = nullptr;
  std::uint64_t remaining = 0;
  fs::path path;
};

void ReadBytes(Bounded& from, void* data, std::size_t bytes) {
  if (bytes > from.remaining) {
    Throw(Kind::Truncated,
          std::format("{} is not whole: it ends inside a value",
                      from.path.string()));
  }
  from.in->read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
  if (!*from.in) {
    Throw(Kind::Truncated,
          std::format("{} could not be read whole", from.path.string()));
  }
  from.remaining -= bytes;
}

std::uint64_t Counted(Bounded& from, std::uint64_t count,
                      std::size_t valueBytes) {
  if (valueBytes > 0 && count > from.remaining / valueBytes) {
    Throw(Kind::Truncated,
          std::format("{} is not whole: a count of {} runs past its end",
                      from.path.string(), count));
  }
  return count;
}

/// Writes a file whole: to its partial name, then renamed into place.
template <class Write>
std::uint64_t WriteWhole(const fs::path& path, Write&& write) {
  const auto partial = path.parent_path() / Partial(path.filename().string());
  auto bytes = std::uint64_t{0};
  {
    auto out = std::ofstream(partial, std::ios::binary | std::ios::trunc);
    if (!out) {
      Throw(Kind::Io, std::format("cannot write {}", partial.string()));
    }
    write(out);
    bytes = static_cast<std::uint64_t>(out.tellp());
    if (!out) {
      Throw(Kind::Io, std::format("cannot write {}", partial.string()));
    }
  }
  auto error = std::error_code{};
  fs::rename(partial, path, error);
  if (error) {
    Throw(Kind::Io, std::format("cannot rename {} to {}: {}", partial.string(),
                                path.string(), error.message()));
  }
  return bytes;
}

/// Writes a record whole: its header, the payload `put` writes, and the
/// payload's length set into the header once known. Returns its bytes.
template <class PutPayload>
std::uint64_t WriteRecord(const fs::path& path, std::uint64_t kind,
                          std::uint64_t node, std::uint64_t rank,
                          PutPayload&& put) {
  return WriteWhole(path, [&](std::ofstream& out) {
    Put(out, kRecordTag.data(), kRecordTag.size());
    PutOne(out, kind);
    PutOne(out, node);
    PutOne(out, rank);
    PutOne(out, std::uint64_t{0});  // the payload's length, set below
    put(out);
    const auto end = static_cast<std::uint64_t>(out.tellp());
    out.seekp(static_cast<std::streamoff>(kHeaderBytes - 8));
    PutOne(out, end - kHeaderBytes);
    out.seekp(static_cast<std::streamoff>(end));
  });
}

/// An open record: its file, checked against the size the manifest gives
/// (when given) and its header against the kind, node and rank expected,
/// its payload left to read.
class RecordReader {
 public:
  RecordReader(const fs::path& path, std::uint64_t kind, std::uint64_t node,
               std::uint64_t rank, std::optional<std::uint64_t> bytes)
      : in_(path, std::ios::binary) {
    from_.in = &in_;
    from_.path = path;
    if (!in_) {
      Throw(fs::exists(path) ? Kind::Io : Kind::Missing,
            std::format("cannot read {}", path.string()));
    }
    auto error = std::error_code{};
    const auto size = fs::file_size(path, error);
    if (error) Throw(Kind::Io, std::format("cannot read {}", path.string()));
    if (bytes && size != *bytes) {
      Throw(Kind::Truncated,
            std::format("{} holds {} bytes, and the store's manifest gives {}",
                        path.string(), size, *bytes));
    }
    from_.remaining = size;
    auto tag = std::array<char, 8>{};
    if (size < kHeaderBytes) {
      Throw(Kind::Truncated,
            std::format("{} is shorter than a record's header", path.string()));
    }
    ReadBytes(from_, tag.data(), tag.size());
    const auto gotKind = GetOne<std::uint64_t>(from_);
    const auto gotNode = GetOne<std::uint64_t>(from_);
    const auto gotRank = GetOne<std::uint64_t>(from_);
    const auto length = GetOne<std::uint64_t>(from_);
    if (std::string_view(tag.data(), tag.size()) != kRecordTag ||
        gotKind != kind || gotNode != node || gotRank != rank) {
      Throw(Kind::Truncated,
            std::format("{} is not the record the store's layout puts there "
                        "(kind {}, node {}, rank {})",
                        path.string(), kind, node, rank));
    }
    if (length != from_.remaining) {
      Throw(Kind::Truncated,
            std::format("{} holds a payload of {} bytes, and its header gives "
                        "{}",
                        path.string(), from_.remaining, length));
    }
  }

  /// Its source points into it: neither copied nor moved.
  RecordReader(const RecordReader&) = delete;
  RecordReader& operator=(const RecordReader&) = delete;

  Bounded& From() { return from_; }

  /// Refuses a payload not read to its end.
  void Finish() const {
    if (from_.remaining != 0) {
      Throw(Kind::Truncated,
            std::format("{} holds {} bytes past what its layout reads",
                        from_.path.string(), from_.remaining));
    }
  }

 private:
  std::ifstream in_;
  Bounded from_;
};

// -- the payloads ---------------------------------------------------------------

void PutBackground(std::ostream& out, const GRDSolver::State& state) {
  PutVector(out, state.iceThickness);
  PutVector(out, state.seaLevel);
  PutVector(out, state.oceanFunction);
  PutOne(out, state.oceanArea);
  PutOne(out, static_cast<std::uint64_t>(state.forcedLand.size()));
  Put(out, state.forcedLand.data(), state.forcedLand.size());
}

GRDSolver::State GetBackground(Bounded& in) {
  auto state = GRDSolver::State{};
  state.iceThickness = GetVector(in);
  state.seaLevel = GetVector(in);
  state.oceanFunction = GetVector(in);
  state.oceanArea = GetOne<double>(in);
  state.forcedLand.resize(Counted(in, GetOne<std::uint64_t>(in), 1));
  ReadBytes(in, state.forcedLand.data(), state.forcedLand.size());
  return state;
}

/// A node's surface record: what either dimension's snapshot holds beside
/// its state, the 3D closure's diagnostics zero in 1D.
struct Surface {
  double t = 0.0;
  bool converged = true;
  std::int64_t coupledIterations = 0;
  double coupledResidual = 0.0;
  std::int64_t solveIterations = 0;
  std::array<double, 3> angularVelocity{};
  const std::vector<double>* fields[6] = {};
  const std::vector<std::complex<double>>* loads = nullptr;
  const std::vector<std::complex<double>>* tides = nullptr;
};

template <class Snapshot>
Surface SurfaceOf(const Snapshot& s) {
  auto out = Surface{};
  out.t = s.t;
  out.converged = s.converged;
  out.angularVelocity = s.angularVelocity;
  out.fields[0] = &s.relativeSeaLevelChange;
  out.fields[1] = &s.displacement;
  out.fields[2] = &s.potential;
  out.fields[3] = &s.tangentialPotential;
  out.fields[4] = &s.toroidalPotential;
  out.fields[5] = &s.oceanFunction;
  out.loads = &s.loadCoefficients;
  out.tides = &s.tideCoefficients;
  if constexpr (std::is_same_v<Snapshot, Evolution3D::Snapshot>) {
    out.coupledIterations = s.coupledIterations;
    out.coupledResidual = s.coupledResidual;
    out.solveIterations = s.solveIterations;
  }
  return out;
}

void PutSurface(std::ostream& out, const Surface& s) {
  PutOne(out, s.t);
  PutOne(out, static_cast<std::uint64_t>(s.converged ? 1 : 0));
  PutOne(out, s.coupledIterations);
  PutOne(out, s.coupledResidual);
  PutOne(out, s.solveIterations);
  Put(out, s.angularVelocity.data(), 3 * sizeof(double));
  for (const auto* field : s.fields) PutVector(out, *field);
  PutComplex(out, *s.loads);
  PutComplex(out, *s.tides);
}

template <class Snapshot>
Snapshot GetSurface(Bounded& in) {
  auto s = Snapshot{};
  s.t = GetOne<double>(in);
  s.converged = GetOne<std::uint64_t>(in) != 0;
  [[maybe_unused]] const auto coupledIterations = GetOne<std::int64_t>(in);
  [[maybe_unused]] const auto coupledResidual = GetOne<double>(in);
  [[maybe_unused]] const auto solveIterations = GetOne<std::int64_t>(in);
  if constexpr (std::is_same_v<Snapshot, Evolution3D::Snapshot>) {
    s.coupledIterations = static_cast<int>(coupledIterations);
    s.coupledResidual = coupledResidual;
    s.solveIterations = static_cast<int>(solveIterations);
  }
  ReadBytes(in, s.angularVelocity.data(), 3 * sizeof(double));
  for (auto* field : {&s.relativeSeaLevelChange, &s.displacement, &s.potential,
                      &s.tangentialPotential, &s.toroidalPotential,
                      &s.oceanFunction}) {
    *field = GetVector(in);
  }
  s.loadCoefficients = GetComplex(in);
  s.tideCoefficients = GetComplex(in);
  return s;
}

void PutState1D(std::ostream& out, const ViscoelasticState& state) {
  PutOne(out, state.t);
  PutDegrees(out, state.m);
  PutOne(out, static_cast<std::uint64_t>(state.columns.size()));
  for (const auto& [degree, columns] : state.columns) {
    PutOne(out, static_cast<std::int64_t>(degree));
    PutOne(out, static_cast<std::int64_t>(columns));
  }
  PutDegrees(out, state.strain);
}

ViscoelasticState GetState1D(Bounded& in) {
  auto state = ViscoelasticState{};
  state.t = GetOne<double>(in);
  state.m = GetDegrees(in);
  const auto count = Counted(in, GetOne<std::uint64_t>(in), 16);
  for (auto k = std::uint64_t{0}; k < count; ++k) {
    const auto degree = GetOne<std::int64_t>(in);
    state.columns[static_cast<int>(degree)] =
        static_cast<int>(GetOne<std::int64_t>(in));
  }
  state.strain = GetDegrees(in);
  return state;
}

/// The present flag of an optional block, then the block.
template <class T, class PutT>
void PutOptional(std::ostream& out, const std::optional<T>& value, PutT put) {
  PutOne(out, static_cast<std::uint64_t>(value ? 1 : 0));
  if (value) put(out, *value);
}

template <class GetT>
auto GetOptional(Bounded& in, GetT get) -> std::optional<decltype(get(in))> {
  if (GetOne<std::uint64_t>(in) == 0) return std::nullopt;
  return get(in);
}

void PutForces1D(std::ostream& out, const RadialSourceForces& forces) {
  PutDegrees(out, forces.spheroidal);
  PutDegrees(out, forces.toroidal);
}

RadialSourceForces GetForces1D(Bounded& in) {
  auto forces = RadialSourceForces{};
  forces.spheroidal = GetDegrees(in);
  forces.toroidal = GetDegrees(in);
  return forces;
}

void PutForces3D(std::ostream& out, const DegreeFields& forces) {
  PutComplexDegrees(out, forces.spheroidal);
  PutComplexDegrees(out, forces.toroidal);
}

DegreeFields GetForces3D(Bounded& in) {
  auto forces = DegreeFields{};
  forces.spheroidal = GetComplexDegrees(in);
  forces.toroidal = GetComplexDegrees(in);
  return forces;
}

void PutToroidal(std::ostream& out, const ToroidalViscoelastic::State& state) {
  PutOne(out, state.t);
  PutDegrees(out, state.m);
  PutDegrees(out, state.strain);
}

ToroidalViscoelastic::State GetToroidal(Bounded& in) {
  auto state = ToroidalViscoelastic::State{};
  state.t = GetOne<double>(in);
  state.m = GetDegrees(in);
  state.strain = GetDegrees(in);
  return state;
}

// -- the manifest ---------------------------------------------------------------

std::string ManifestBytes(const TrajectoryStore::Manifest& m) {
  auto out = std::ostringstream{};
  Put(out, kManifestTag.data(), kManifestTag.size());
  PutOne(out, kByteOrder);
  PutOne(out, m.version);
  PutOne(out, static_cast<std::uint64_t>(m.complete ? 1 : 0));
  const auto putText = [&out](const std::string& text) {
    PutOne(out, static_cast<std::uint64_t>(text.size()));
    Put(out, text.data(), text.size());
  };
  putText(m.hash);
  putText(m.writer);
  PutOne(out, static_cast<std::uint64_t>(m.dimension));
  PutVector(out, m.times);
  PutOne(out, static_cast<std::uint64_t>(m.record_stride));
  for (const auto kept : m.kept) {
    PutOne(out, static_cast<std::uint64_t>(kept ? 1 : 0));
  }
  for (const auto flag : {m.migrating, m.sources, m.tide}) {
    PutOne(out, static_cast<std::uint64_t>(flag ? 1 : 0));
  }
  PutOne(out, m.field_size);
  PutOne(out, static_cast<std::uint64_t>(m.cuts.size()));
  for (const auto cut : m.cuts) PutOne(out, static_cast<std::int64_t>(cut));
  PutOne(out, static_cast<std::uint64_t>(m.parts.size()));
  for (const auto& part : m.parts) {
    PutOne(out, part.first_segment);
    PutOne(out, part.segments);
    PutOne(out, part.grid_values);
  }
  PutOne(out, m.start_bytes);
  for (auto n = std::size_t{0}; n < m.times.size(); ++n) {
    PutOne(out, m.surface_bytes.empty() ? 0 : m.surface_bytes[n]);
  }
  for (auto r = std::size_t{0}; r < m.parts.size(); ++r) {
    for (auto n = std::size_t{0}; n < m.times.size(); ++n) {
      PutOne(out, m.state_bytes.empty() ? 0 : m.state_bytes[r][n]);
    }
  }
  return std::move(out).str();
}

TrajectoryStore::Manifest ReadManifest(const fs::path& path) {
  if (!fs::exists(path)) {
    Throw(Kind::Missing, std::format("{} is missing", path.string()));
  }
  auto in = std::ifstream(path, std::ios::binary);
  auto error = std::error_code{};
  const auto size = fs::file_size(path, error);
  if (!in || error) Throw(Kind::Io, std::format("cannot read {}", path.string()));
  auto from = Bounded{.in = &in, .remaining = size, .path = path};
  auto tag = std::array<char, 8>{};
  if (size < tag.size() + 2 * sizeof(std::uint64_t)) {
    Throw(Kind::Tag, std::format("{} is not a trajectory store's manifest",
                                 path.string()));
  }
  ReadBytes(from, tag.data(), tag.size());
  if (std::string_view(tag.data(), tag.size()) != kManifestTag) {
    Throw(Kind::Tag, std::format("{} is not a trajectory store's manifest (it "
                                 "does not begin with {})",
                                 path.string(), kManifestTag));
  }
  const auto order = GetOne<std::uint64_t>(from);
  if (order == std::byteswap(kByteOrder)) {
    Throw(Kind::ByteOrder,
          std::format("{} was written in the other byte order (big-endian), "
                      "and the store is little-endian",
                      path.string()));
  }
  if (order != kByteOrder) {
    Throw(Kind::Tag, std::format("{} is not a trajectory store's manifest (its "
                                 "byte-order mark is not the store's)",
                                 path.string()));
  }
  auto m = TrajectoryStore::Manifest{};
  m.version = GetOne<std::uint64_t>(from);
  if (m.version != TrajectoryStore::kVersion) {
    Throw(Kind::Version,
          std::format("{} is of trajectory store format version {}, and this "
                      "program reads version {}",
                      path.string(), m.version, TrajectoryStore::kVersion));
  }
  m.complete = GetOne<std::uint64_t>(from) != 0;
  const auto getText = [&from] {
    auto text = std::string(Counted(from, GetOne<std::uint64_t>(from), 1), '\0');
    ReadBytes(from, text.data(), text.size());
    return text;
  };
  m.hash = getText();
  m.writer = getText();
  m.dimension = static_cast<int>(GetOne<std::uint64_t>(from));
  m.times = GetVector(from);
  const auto nodes = m.times.size();
  m.record_stride = static_cast<int>(GetOne<std::uint64_t>(from));
  Counted(from, nodes, sizeof(std::uint64_t));
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    m.kept.push_back(GetOne<std::uint64_t>(from) != 0);
  }
  m.migrating = GetOne<std::uint64_t>(from) != 0;
  m.sources = GetOne<std::uint64_t>(from) != 0;
  m.tide = GetOne<std::uint64_t>(from) != 0;
  m.field_size = GetOne<std::uint64_t>(from);
  const auto cuts = Counted(from, GetOne<std::uint64_t>(from), 8);
  for (auto k = std::uint64_t{0}; k < cuts; ++k) {
    m.cuts.push_back(static_cast<int>(GetOne<std::int64_t>(from)));
  }
  const auto ranks = Counted(from, GetOne<std::uint64_t>(from), 24);
  for (auto r = std::uint64_t{0}; r < ranks; ++r) {
    auto part = TrajectoryStore::Part{};
    part.first_segment = GetOne<std::uint64_t>(from);
    part.segments = GetOne<std::uint64_t>(from);
    part.grid_values = GetOne<std::uint64_t>(from);
    m.parts.push_back(part);
  }
  m.start_bytes = GetOne<std::uint64_t>(from);
  Counted(from, nodes * (1 + ranks), sizeof(std::uint64_t));
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    m.surface_bytes.push_back(GetOne<std::uint64_t>(from));
  }
  m.state_bytes.assign(ranks, std::vector<std::uint64_t>(nodes, 0));
  for (auto r = std::size_t{0}; r < ranks; ++r) {
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      m.state_bytes[r][n] = GetOne<std::uint64_t>(from);
    }
  }
  if (from.remaining != 0) {
    Throw(Kind::Truncated,
          std::format("{} holds {} bytes past what a manifest reads",
                      path.string(), from.remaining));
  }
  return m;
}

// -- the processes ----------------------------------------------------------------

using Description = TrajectoryStore::Description;
using Part = TrajectoryStore::Part;

void Barrier(const Description& run) {
  if (!run.collectives.Distributed()) return;
  auto none = 0.0;
  run.collectives.sum(std::span(&none, 1));
}

/// Every process's part, in rank order.
std::vector<Part> PartsOf(const Description& run) {
  if (!run.collectives.Distributed()) return {run.part};
  const auto ranks = static_cast<std::size_t>(run.ranks);
  auto values = std::vector<double>(3 * ranks, 0.0);
  const auto r = static_cast<std::size_t>(run.rank);
  values[3 * r] = static_cast<double>(run.part.first_segment);
  values[3 * r + 1] = static_cast<double>(run.part.segments);
  values[3 * r + 2] = static_cast<double>(run.part.grid_values);
  run.collectives.sum(values);
  auto parts = std::vector<Part>(ranks);
  for (auto k = std::size_t{0}; k < ranks; ++k) {
    parts[k] = {.first_segment = static_cast<std::uint64_t>(values[3 * k]),
                .segments = static_cast<std::uint64_t>(values[3 * k + 1]),
                .grid_values = static_cast<std::uint64_t>(values[3 * k + 2])};
  }
  return parts;
}

/// Runs `check` on every process and agrees on its outcome: a fault on
/// any process is thrown on all, each with its own message where it found
/// one, the kind of the first process's fault where it did not.
template <class Check>
void Agreed(const Description& run, Check&& check) {
  auto fault = std::optional<TrajectoryStoreError>{};
  try {
    check();
  } catch (const TrajectoryStoreError& error) {
    fault = error;
  } catch (const fs::filesystem_error& error) {
    fault = TrajectoryStoreError(Kind::Io, error.what());
  }
  if (!run.collectives.Distributed()) {
    if (fault) throw *fault;
    return;
  }
  auto code = fault ? static_cast<double>(fault->GetKind()) + 1.0 : 0.0;
  run.collectives.max(std::span(&code, 1));
  if (fault) throw *fault;
  if (code != 0.0) {
    Throw(static_cast<Kind>(static_cast<int>(code) - 1),
          "the trajectory store was refused on another process");
  }
}

std::string Listed(const std::vector<fs::path>& paths) {
  auto out = std::string{};
  for (auto k = std::size_t{0}; k < std::min<std::size_t>(paths.size(), 3);
       ++k) {
    if (!out.empty()) out += ", ";
    out += paths[k].string();
  }
  if (paths.size() > 3) out += std::format(" and {} more", paths.size() - 3);
  return out;
}

/// What Survey finds, with the manifest it read.
struct Found {
  TrajectoryStore::Standing standing = TrajectoryStore::Standing::Empty;
  std::optional<TrajectoryStore::Manifest> manifest;
  std::vector<fs::path> own;
};

Found Inspect(const fs::path& directory, const Description& run,
              const std::vector<Part>& parts) {
  using Standing = TrajectoryStore::Standing;
  auto out = Found{};
  auto error = std::error_code{};
  if (!fs::exists(directory, error)) return out;
  if (!fs::is_directory(directory, error)) {
    Throw(Kind::Foreign, std::format("{} is not a directory: give a "
                                     "directory for the trajectory store",
                                     directory.string()));
  }
  const auto contents = Classify(directory);
  if (!contents.foreign.empty()) {
    Throw(Kind::Foreign,
          std::format("{} holds what a trajectory store does not write ({}): "
                      "give another directory, or remove it",
                      directory.string(), Listed(contents.foreign)));
  }
  out.own = contents.own;
  if (contents.own.empty()) return out;
  if (!contents.manifest) {
    Throw(Kind::Foreign,
          std::format("{} holds a trajectory store's records but no manifest, "
                      "which a recording writes first: give another "
                      "directory, or remove it",
                      directory.string()));
  }
  const auto m = ReadManifest(directory / kManifestName);
  if (m.hash != run.hash) {
    Throw(Kind::Hash,
          std::format("{} holds the trajectory store of another run ({}): "
                      "give another directory, or remove it",
                      directory.string(),
                      m.complete ? "complete" : "incomplete"));
  }
  out.manifest = m;
  if (!m.complete) {
    out.standing = Standing::Incomplete;
    return out;
  }
  if (m.dimension != run.dimension || m.times != run.times ||
      m.field_size != run.field_size || m.migrating != run.migrating ||
      m.sources != run.sources || m.tide != run.tide) {
    Throw(Kind::Mismatch,
          std::format("{} holds a store of this run's hash whose dimension, "
                      "nodes, grid or forcing are not this run's",
                      directory.string()));
  }
  if (m.record_stride != run.record_stride) {
    Throw(Kind::Stride,
          std::format("{} holds this run recorded at record stride {}, and "
                      "this run's is {}: record it into another directory, "
                      "or remove that one",
                      directory.string(), m.record_stride, run.record_stride));
  }
  if (m.parts.size() != parts.size()) {
    Throw(Kind::Layout,
          std::format("{} holds this run recorded over {} process(es), and "
                      "this run has {}: run over {}, or record into another "
                      "directory",
                      directory.string(), m.parts.size(), parts.size(),
                      m.parts.size()));
  }
  if (m.parts != parts || m.cuts != run.cuts) {
    Throw(Kind::Layout,
          std::format("{} holds this run recorded over {} process(es) whose "
                      "parts of the mesh are not this run's",
                      directory.string(), m.parts.size()));
  }
  out.standing = Standing::Complete;
  return out;
}

}  // namespace

// -- the store's state ---------------------------------------------------------------

struct TrajectoryStore::Impl {
  fs::path directory;
  Description run;
  Manifest manifest;
  bool recording = false;
  /// Recording: what has been written, and its bytes.
  bool startWritten = false;
  std::vector<bool> surfaceWritten;
  std::vector<bool> stateWritten;
  std::vector<std::uint64_t> stateBytes;
  /// A 3D node's sources' forces, held for its state record.
  std::map<std::size_t, DegreeFields> pendingForces;
  std::uint64_t manifestBytes = 0;
  std::size_t statesRead = 0;
  std::size_t recordsWritten = 0;

  [[nodiscard]] bool First() const { return run.rank == 0; }
  [[nodiscard]] fs::path RankDirectory(int rank) const {
    return directory / TrajectoryStore::RankDirectoryName(rank);
  }
  [[nodiscard]] fs::path SurfacePath(std::size_t node) const {
    return directory / NodeName(node, kSurfaceSuffix);
  }
  [[nodiscard]] fs::path StatePath(std::size_t node) const {
    return RankDirectory(run.rank) / NodeName(node, kStateSuffix);
  }
  [[nodiscard]] std::size_t Nodes() const { return manifest.times.size(); }

  void CheckNode(std::size_t node) const {
    if (node >= Nodes()) {
      throw std::logic_error(std::format(
          "node {} is off the recording's {} nodes", node, Nodes()));
    }
  }
  void CheckRecording() const {
    if (!recording) {
      throw std::logic_error("the trajectory store was opened for reading");
    }
  }
  void CheckKept(std::size_t node) const {
    if (!manifest.kept[node]) {
      throw std::logic_error(std::format(
          "node {} is not kept at record stride {}", node,
          manifest.record_stride));
    }
  }

  void WriteState(std::size_t node, const std::function<void(std::ostream&)>&
                                        put) {
    CheckKept(node);
    stateBytes[node] =
        WriteRecord(StatePath(node), kStateKind, node,
                    static_cast<std::uint64_t>(run.rank), put);
    stateWritten[node] = true;
    ++recordsWritten;
  }

  [[nodiscard]] RecordReader Surface(std::size_t node) const {
    return RecordReader(SurfacePath(node), kSurfaceKind, node, 0,
                        manifest.surface_bytes[node]);
  }
  [[nodiscard]] RecordReader State(std::size_t node) const {
    const auto rank = static_cast<std::size_t>(run.rank);
    return RecordReader(StatePath(node), kStateKind, node, rank,
                        manifest.state_bytes[rank][node]);
  }
};

TrajectoryStore::TrajectoryStore(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

bool TrajectoryStore::Description::Kept(std::size_t node) const {
  const auto stride = static_cast<std::size_t>(std::max(record_stride, 1));
  return node % stride == 0 || node + 1 == times.size();
}

TrajectoryStore::Manifest TrajectoryStore::ManifestOf(
    const std::filesystem::path& directory) {
  CheckMachine();
  return ReadManifest(directory / kManifestName);
}

std::string TrajectoryStore::RankDirectoryName(int rank) {
  return std::format("rank_{:04}", rank);
}

TrajectoryStore::Part TrajectoryStore::PartOf(const Viscoelastic3D& op) {
  const auto& options = op.GetOptions();
  auto part = Part{};
  part.grid_values = op.Coupling().GridValueCount();
  if (options.segments.empty()) {
    part.first_segment = 0;
    part.segments = options.radial_cuts.size() + 1;
  } else {
    part.first_segment = static_cast<std::uint64_t>(options.segments.front());
    part.segments = options.segments.size();
  }
  return part;
}

TrajectoryStore::Standing TrajectoryStore::Survey(
    const std::filesystem::path& directory, const Description& run) {
  const auto parts = PartsOf(run);
  auto found = Found{};
  Agreed(run, [&] { found = Inspect(directory, run, parts); });
  return found.standing;
}

TrajectoryStore TrajectoryStore::Record(const std::filesystem::path& directory,
                                        const Description& run) {
  CheckMachine();
  if (run.dimension != 1 && run.dimension != 3) {
    throw std::invalid_argument("a trajectory store's dimension is 1 or 3");
  }
  if (run.times.empty() || run.record_stride < 1) {
    throw std::invalid_argument(
        "a recording takes at least one node and a stride of at least one");
  }
  if (run.sources && run.record_stride != 1) {
    throw std::invalid_argument(
        "a run under interior sources is recorded at every node");
  }
  const auto parts = PartsOf(run);
  auto found = Found{};
  Agreed(run, [&] { found = Inspect(directory, run, parts); });
  if (found.standing == Standing::Complete) {
    throw std::logic_error(std::format(
        "{} holds this run's complete trajectory store: open it for reading",
        directory.string()));
  }
  auto impl = std::make_shared<Impl>();
  impl->directory = directory;
  impl->run = run;
  impl->recording = true;
  auto& m = impl->manifest;
  m.hash = run.hash;
  m.writer = std::format("sl3dpp {}", Version());
  m.dimension = run.dimension;
  m.times = run.times;
  m.record_stride = run.record_stride;
  for (auto n = std::size_t{0}; n < run.times.size(); ++n) {
    m.kept.push_back(run.Kept(n));
  }
  m.migrating = run.migrating;
  m.sources = run.sources;
  m.tide = run.tide;
  m.field_size = run.field_size;
  m.cuts = run.cuts;
  m.parts = parts;
  const auto nodes = run.times.size();
  m.surface_bytes.assign(nodes, 0);
  impl->surfaceWritten.assign(nodes, false);
  impl->stateWritten.assign(nodes, false);
  impl->stateBytes.assign(nodes, 0);
  // every process has surveyed before the first changes anything
  Barrier(run);
  Agreed(run, [&] {
    if (!impl->First()) return;
    auto error = std::error_code{};
    // an incomplete store of this run: its own files go, the files first
    for (const auto& path : found.own) {
      if (!fs::remove(path, error) && error) {
        Throw(Kind::Io, std::format("cannot remove {}: {}", path.string(),
                                    error.message()));
      }
    }
    fs::create_directories(directory, error);
    if (error) {
      Throw(Kind::Io, std::format("cannot create {}: {}", directory.string(),
                                  error.message()));
    }
    impl->manifestBytes = WriteWhole(
        directory / kManifestName,
        [&](std::ofstream& out) { out << ManifestBytes(m); });
  });
  Agreed(run, [&] {
    auto error = std::error_code{};
    fs::create_directories(impl->RankDirectory(run.rank), error);
    if (error) {
      Throw(Kind::Io,
            std::format("cannot create {}: {}",
                        impl->RankDirectory(run.rank).string(),
                        error.message()));
    }
  });
  return TrajectoryStore(std::move(impl));
}

TrajectoryStore TrajectoryStore::Open(const std::filesystem::path& directory,
                                      const Description& run) {
  CheckMachine();
  const auto parts = PartsOf(run);
  auto found = Found{};
  Agreed(run, [&] {
    found = Inspect(directory, run, parts);
    if (found.standing == Standing::Empty) {
      Throw(Kind::Missing, std::format("{} holds no trajectory store",
                                       directory.string()));
    }
    if (found.standing == Standing::Incomplete) {
      Throw(Kind::Incomplete,
            std::format("{} holds an incomplete trajectory store of this run "
                        "(its recording did not finish)",
                        directory.string()));
    }
  });
  auto impl = std::make_shared<Impl>();
  impl->directory = directory;
  impl->run = run;
  impl->manifest = std::move(*found.manifest);
  // every record this process reads, there at its size
  Agreed(run, [&] {
    const auto& m = impl->manifest;
    const auto rank = static_cast<std::size_t>(run.rank);
    const auto check = [](const fs::path& path, std::uint64_t bytes) {
      auto error = std::error_code{};
      const auto size = fs::file_size(path, error);
      if (error) {
        Throw(Kind::Missing,
              std::format("{} is missing from the trajectory store",
                          path.string()));
      }
      if (size != bytes) {
        Throw(Kind::Truncated,
              std::format("{} holds {} bytes, and the store's manifest gives "
                          "{}",
                          path.string(), size, bytes));
      }
    };
    check(directory / kStartName, m.start_bytes);
    for (auto n = std::size_t{0}; n < impl->Nodes(); ++n) {
      check(impl->SurfacePath(n), m.surface_bytes[n]);
      if (m.kept[n]) check(impl->StatePath(n), m.state_bytes[rank][n]);
    }
    auto error = std::error_code{};
    impl->manifestBytes = fs::file_size(directory / kManifestName, error);
  });
  return TrajectoryStore(std::move(impl));
}

// -- recording -------------------------------------------------------------------

void TrajectoryStore::PutStart(const GRDSolver::State& background) const {
  auto& s = *impl_;
  s.CheckRecording();
  s.startWritten = true;
  if (!s.First()) return;
  s.manifest.start_bytes =
      WriteRecord(s.directory / kStartName, kStartKind, 0, 0,
                  [&](std::ostream& out) { PutBackground(out, background); });
  ++s.recordsWritten;
}

void TrajectoryStore::Put(std::size_t node,
                          const Evolution::Snapshot& snapshot) const {
  auto& s = *impl_;
  s.CheckRecording();
  s.CheckNode(node);
  if (s.run.dimension != 1) {
    throw std::logic_error("a 1D snapshot put to a 3D trajectory store");
  }
  if (s.First()) {
    s.manifest.surface_bytes[node] = WriteRecord(
        s.SurfacePath(node), kSurfaceKind, node, 0,
        [&](std::ostream& out) { PutSurface(out, SurfaceOf(snapshot)); });
    ++s.recordsWritten;
  }
  s.surfaceWritten[node] = true;
  if (!s.manifest.kept[node]) return;
  if (!snapshot.internalState) {
    throw std::logic_error(std::format(
        "node {} is kept and its snapshot carries no state: run the "
        "evolution at the store's record stride",
        node));
  }
  s.WriteState(node, [&](std::ostream& out) {
    PutOptional(out, snapshot.sourceForces, PutForces1D);
    PutOptional(out, snapshot.toroidalState, PutToroidal);
    PutOptional(out, snapshot.internalState, PutState1D);
  });
}

void TrajectoryStore::Put(std::size_t node,
                          const Evolution3D::Snapshot& snapshot) const {
  auto& s = *impl_;
  s.CheckRecording();
  s.CheckNode(node);
  if (s.run.dimension != 3) {
    throw std::logic_error("a 3D snapshot put to a 1D trajectory store");
  }
  if (s.First()) {
    s.manifest.surface_bytes[node] = WriteRecord(
        s.SurfacePath(node), kSurfaceKind, node, 0,
        [&](std::ostream& out) { PutSurface(out, SurfaceOf(snapshot)); });
    ++s.recordsWritten;
  }
  s.surfaceWritten[node] = true;
  if (snapshot.sourceForces) s.pendingForces[node] = *snapshot.sourceForces;
  // a snapshot that keeps its state (no on_state): its record at once
  if (snapshot.internalState) {
    PutState(node, Viscoelastic3D::State(*snapshot.internalState));
  }
}

void TrajectoryStore::PutState(std::size_t node,
                               Viscoelastic3D::State&& state) const {
  auto& s = *impl_;
  s.CheckRecording();
  s.CheckNode(node);
  auto forces = std::optional<DegreeFields>{};
  if (const auto found = s.pendingForces.find(node);
      found != s.pendingForces.end()) {
    forces = std::move(found->second);
    s.pendingForces.erase(found);
  }
  s.WriteState(node, [&](std::ostream& out) {
    PutOptional(out, forces, PutForces3D);
    PutOne(out, std::uint64_t{1});
    PutState3D(out, state);
  });
}

void TrajectoryStore::Attach(Evolution::Options& options) const {
  options.record_stride = impl_->run.record_stride;
  options.on_snapshot = [store = *this, before = std::move(options.on_snapshot)](
                            std::size_t node, const Evolution::Snapshot& s) {
    if (before) before(node, s);
    store.Put(node, s);
  };
}

void TrajectoryStore::Attach(Evolution3D::Options& options) const {
  options.record_stride = impl_->run.record_stride;
  options.on_snapshot = [store = *this, before = std::move(options.on_snapshot)](
                            std::size_t node, const Evolution3D::Snapshot& s) {
    if (before) before(node, s);
    store.Put(node, s);
  };
  options.on_state = [store = *this](std::size_t node,
                                     Viscoelastic3D::State&& state) {
    store.PutState(node, std::move(state));
  };
}

void TrajectoryStore::Close() const {
  auto& s = *impl_;
  s.CheckRecording();
  const auto& run = s.run;
  Agreed(run, [&] {
    if (!s.startWritten) {
      Throw(Kind::Incomplete, std::format("the recording into {} holds no "
                                          "start (PutStart)",
                                          s.directory.string()));
    }
    for (auto n = std::size_t{0}; n < s.Nodes(); ++n) {
      if (!s.surfaceWritten[n] ||
          (s.manifest.kept[n] && !s.stateWritten[n])) {
        Throw(Kind::Incomplete,
              std::format("the recording into {} lacks node {}'s record",
                          s.directory.string(), n));
      }
    }
  });
  // every process's state records' sizes, to the first
  const auto nodes = s.Nodes();
  const auto ranks = static_cast<std::size_t>(run.ranks);
  auto& m = s.manifest;
  m.state_bytes.assign(ranks, std::vector<std::uint64_t>(nodes, 0));
  if (run.collectives.Distributed()) {
    auto sizes = std::vector<double>(ranks * nodes, 0.0);
    const auto r = static_cast<std::size_t>(run.rank);
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      sizes[r * nodes + n] = static_cast<double>(s.stateBytes[n]);
    }
    run.collectives.sum(sizes);
    for (auto k = std::size_t{0}; k < ranks; ++k) {
      for (auto n = std::size_t{0}; n < nodes; ++n) {
        m.state_bytes[k][n] = static_cast<std::uint64_t>(sizes[k * nodes + n]);
      }
    }
  } else {
    m.state_bytes[0] = s.stateBytes;
  }
  m.complete = true;
  Agreed(run, [&] {
    if (!s.First()) return;
    s.manifestBytes = WriteWhole(s.directory / kManifestName,
                                 [&](std::ofstream& out) {
                                   out << ManifestBytes(m);
                                 });
  });
  // the manifest's size, every process's to report
  if (run.collectives.Distributed()) {
    auto bytes = static_cast<double>(s.manifestBytes);
    run.collectives.share(std::span(&bytes, 1));
    s.manifestBytes = static_cast<std::uint64_t>(bytes);
    auto start = static_cast<double>(m.start_bytes);
    run.collectives.share(std::span(&start, 1));
    m.start_bytes = static_cast<std::uint64_t>(start);
    auto surfaces = std::vector<double>(nodes, 0.0);
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      if (s.First()) surfaces[n] = static_cast<double>(m.surface_bytes[n]);
    }
    run.collectives.share(surfaces);
    m.surface_bytes.resize(nodes);
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      m.surface_bytes[n] = static_cast<std::uint64_t>(surfaces[n]);
    }
  }
  s.recording = false;
}

// -- reading -------------------------------------------------------------------------

const TrajectoryStore::Manifest& TrajectoryStore::GetManifest() const {
  return impl_->manifest;
}

const std::filesystem::path& TrajectoryStore::Directory() const {
  return impl_->directory;
}

GRDSolver::State TrajectoryStore::Start() const {
  auto record = RecordReader(impl_->directory / kStartName, kStartKind, 0, 0,
                             impl_->manifest.start_bytes);
  auto state = GetBackground(record.From());
  record.Finish();
  return state;
}

std::vector<Evolution::Snapshot> TrajectoryStore::Snapshots1D() const {
  auto& s = *impl_;
  if (s.manifest.dimension != 1) {
    throw std::logic_error("the 1D snapshots of a 3D trajectory store");
  }
  auto out = std::vector<Evolution::Snapshot>{};
  for (auto n = std::size_t{0}; n < s.Nodes(); ++n) {
    auto surface = s.Surface(n);
    out.push_back(GetSurface<Evolution::Snapshot>(surface.From()));
    surface.Finish();
    if (!s.manifest.kept[n]) continue;
    auto record = s.State(n);
    auto& from = record.From();
    auto& snapshot = out.back();
    snapshot.sourceForces = GetOptional(from, GetForces1D);
    snapshot.toroidalState = GetOptional(from, GetToroidal);
    snapshot.internalState = GetOptional(from, GetState1D);
    record.Finish();
    ++s.statesRead;
  }
  return out;
}

std::vector<Evolution3D::Snapshot> TrajectoryStore::Snapshots3D() const {
  auto& s = *impl_;
  if (s.manifest.dimension != 3) {
    throw std::logic_error("the 3D snapshots of a 1D trajectory store");
  }
  auto out = std::vector<Evolution3D::Snapshot>{};
  for (auto n = std::size_t{0}; n < s.Nodes(); ++n) {
    auto surface = s.Surface(n);
    out.push_back(GetSurface<Evolution3D::Snapshot>(surface.From()));
    surface.Finish();
    // the sources' forces lead a state record, read without its state
    if (s.manifest.sources && s.manifest.kept[n]) {
      auto record = s.State(n);
      out.back().sourceForces = GetOptional(record.From(), GetForces3D);
    }
  }
  return out;
}

Viscoelastic3D::State TrajectoryStore::State3D(std::size_t node) const {
  auto& s = *impl_;
  if (s.manifest.dimension != 3) {
    throw std::logic_error("a 3D state of a 1D trajectory store");
  }
  s.CheckNode(node);
  s.CheckKept(node);
  auto record = s.State(node);
  auto& from = record.From();
  static_cast<void>(GetOptional(from, GetForces3D));
  auto state = GetOptional(from, [](Bounded& in) { return GetState3D(in); });
  record.Finish();
  if (!state) {
    Throw(Kind::Truncated,
          std::format("{} holds no state", s.StatePath(node).string()));
  }
  ++s.statesRead;
  return std::move(*state);
}

ForwardTrajectory3D TrajectoryStore::Trajectory3D(
    std::span<const Evolution3D::Snapshot> snapshots) const {
  auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
  trajectory.stored = impl_->manifest.kept;
  trajectory.fetch = [store = *this](std::size_t node) {
    return store.State3D(node);
  };
  return trajectory;
}

std::uint64_t TrajectoryStore::Bytes() const {
  const auto& s = *impl_;
  const auto& m = s.manifest;
  auto total = s.manifestBytes + m.start_bytes;
  for (const auto bytes : m.surface_bytes) total += bytes;
  for (const auto& rank : m.state_bytes) {
    for (const auto bytes : rank) total += bytes;
  }
  return total;
}

std::size_t TrajectoryStore::StatesRead() const { return impl_->statesRead; }

std::size_t TrajectoryStore::RecordsWritten() const {
  return impl_->recordsWritten;
}

}  // namespace sl3dpp::io
