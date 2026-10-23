#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <planetmodel/json.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Commands.hpp"
#include "Input.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

namespace sl3dpp::app {

namespace {

/// Seconds in a kiloyear of Julian years, as pyslfp counts them.
constexpr auto KYR = 1.0e3 * 365.25 * 86400.0;

/// One node of the accumulated history, SI, kept across restarts.
struct Node {
  double date_kyr_bp = 0.0;
  std::vector<double> seaLevelChange;
  std::vector<double> displacement;
  std::vector<double> potential;
  std::array<double, 3> angularVelocity{};
};

/// A run stopped on purpose at a checkpoint: the wall clock passed the
/// configured limit after the checkpoint was written.
struct WalltimeReached {
  std::size_t node = 0;
};

// -- the checkpoint files: private scratch of a run, machine-local ----------

constexpr auto kMagic = std::uint64_t{0x534c3344434b5031};  // "SL3DCKP1"

/// FNV-1a of the config's text: a restart with another config is refused.
std::uint64_t HashOf(const std::filesystem::path& config) {
  auto in = std::ifstream(config, std::ios::binary);
  auto hash = std::uint64_t{14695981039346656037ull};
  char c = 0;
  while (in.get(c)) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

void Put(std::ofstream& out, const void* data, std::size_t bytes) {
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

template <class T>
void PutOne(std::ofstream& out, T value) {
  Put(out, &value, sizeof(T));
}

void PutVector(std::ofstream& out, const std::vector<double>& values) {
  PutOne(out, static_cast<std::uint64_t>(values.size()));
  Put(out, values.data(), values.size() * sizeof(double));
}

void PutComplex(std::ofstream& out,
                const std::vector<std::complex<double>>& values) {
  PutOne(out, static_cast<std::uint64_t>(values.size()));
  Put(out, values.data(), values.size() * sizeof(std::complex<double>));
}

template <class T>
T GetOne(std::ifstream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

std::vector<double> GetVector(std::ifstream& in) {
  auto values = std::vector<double>(GetOne<std::uint64_t>(in));
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(double)));
  return values;
}

std::vector<std::complex<double>> GetComplex(std::ifstream& in) {
  auto values = std::vector<std::complex<double>>(GetOne<std::uint64_t>(in));
  in.read(
      reinterpret_cast<char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(std::complex<double>)));
  return values;
}

/// Writes a checkpoint whole (a temporary renamed into place) and keeps
/// the newest two in the directory.
void WriteCheckpoint(const std::filesystem::path& directory,
                     std::uint64_t hash, std::size_t node,
                     const SeaLevelEvolution::Checkpoint& cp,
                     const std::vector<Node>& history) {
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
    PutOne(out, static_cast<std::uint64_t>(node));
    PutOne(out, cp.t);
    for (const auto* held : {&cp.m, &cp.strain}) {
      PutOne(out, static_cast<std::uint64_t>(held->size()));
      for (const auto& [degree, values] : *held) {
        PutOne(out, static_cast<std::int64_t>(degree));
        PutVector(out, values);
      }
    }
    PutComplex(out, cp.loadCoefficients);
    PutComplex(out, cp.tideCoefficients);
    PutOne(out, static_cast<std::uint64_t>(history.size()));
    for (const auto& kept : history) {
      PutOne(out, kept.date_kyr_bp);
      PutVector(out, kept.seaLevelChange);
      PutVector(out, kept.displacement);
      PutVector(out, kept.potential);
      Put(out, kept.angularVelocity.data(), 3 * sizeof(double));
    }
    if (!out) {
      throw std::runtime_error(
          std::format("cannot write {}", partial.string()));
    }
  }
  std::filesystem::rename(partial, path);
  // keep the newest two
  auto kept = std::vector<std::filesystem::path>{};
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() == ".ckpt") kept.push_back(entry.path());
  }
  std::sort(kept.begin(), kept.end());
  while (kept.size() > 2) {
    std::filesystem::remove(kept.front());
    kept.erase(kept.begin());
  }
}

struct ReadBack {
  std::size_t node = 0;
  SeaLevelEvolution::Checkpoint checkpoint;
  std::vector<Node> history;
};

/// The newest checkpoint of this config in the directory, or nothing.
std::optional<ReadBack> ReadCheckpoint(const std::filesystem::path& directory,
                                       std::uint64_t hash) {
  if (!std::filesystem::is_directory(directory)) return std::nullopt;
  auto candidates = std::vector<std::filesystem::path>{};
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() == ".ckpt") candidates.push_back(entry.path());
  }
  std::sort(candidates.rbegin(), candidates.rend());
  for (const auto& path : candidates) {
    auto in = std::ifstream(path, std::ios::binary);
    if (!in || GetOne<std::uint64_t>(in) != kMagic) continue;
    if (GetOne<std::uint64_t>(in) != hash) {
      throw std::runtime_error(std::format(
          "{} was checkpointed by a different config; move it aside or "
          "point checkpoint.directory elsewhere",
          path.string()));
    }
    auto out = ReadBack{};
    out.node = GetOne<std::uint64_t>(in);
    out.checkpoint.t = GetOne<double>(in);
    for (auto* held : {&out.checkpoint.m, &out.checkpoint.strain}) {
      const auto degrees = GetOne<std::uint64_t>(in);
      for (auto k = std::uint64_t{0}; k < degrees; ++k) {
        const auto degree = GetOne<std::int64_t>(in);
        held->emplace(static_cast<int>(degree), GetVector(in));
      }
    }
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
      out.history.push_back(std::move(kept));
    }
    if (in) return out;
  }
  return std::nullopt;
}

}  // namespace

EvolveInput ReadEvolveInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  Allow(top, {"model", "model_format", "lmax", "state", "ice_history",
              "times", "scheme", "rotation", "moments", "rtol", "checkpoint",
              "restart", "output"});
  const auto directory = config.parent_path();
  auto out = EvolveInput{};
  out.model = PathOf(top, "model", directory);
  if (Has(top, "model_format")) {
    out.format = Choice<ModelFormat>(top.At("model_format"),
                                     {{"gia", ModelFormat::Gia},
                                      {"mineos", ModelFormat::Mineos}});
  }
  out.lmax = Integer(top, "lmax");
  if (out.lmax < 1) Fail(top.At("lmax"), "is not a degree");
  const auto& state = top.At("state");
  Allow(state, {"file"});
  out.state = PathOf(state, "file", directory);
  const auto& history = top.At("ice_history");
  Allow(history, {"file"});
  out.ice_history = PathOf(history, "file", directory);
  const auto& times = top.At("times");
  Allow(times, {"start_kyr_bp", "end_kyr_bp", "nodes"});
  out.start_kyr_bp = Number(times, "start_kyr_bp");
  out.end_kyr_bp = Number(times, "end_kyr_bp", 0.0);
  out.nodes = Integer(times, "nodes");
  if (!(out.start_kyr_bp > out.end_kyr_bp)) {
    Fail(times, "runs forward in time: start_kyr_bp exceeds end_kyr_bp");
  }
  if (out.nodes < 2) Fail(times.At("nodes"), "is fewer than two nodes");
  if (Has(top, "scheme")) {
    out.scheme = Choice<Scheme>(
        top.At("scheme"), {{"exponential_trapezoid", Scheme::ExponentialTrapezoid},
                           {"backward_euler", Scheme::BackwardEuler}});
  }
  if (Has(top, "rotation")) {
    out.rotation = Choice<RotationalFeedback>(
        top.At("rotation"), {{"none", RotationalFeedback::None},
                             {"axial", RotationalFeedback::Axial},
                             {"full", RotationalFeedback::Full}});
  }
  if (Has(top, "moments")) {
    out.hydrostatic_moments =
        Choice<bool>(top.At("moments"), {{"input", false},
                                         {"hydrostatic", true}});
  }
  out.rtol = Number(top, "rtol", 1e-9);
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
  namespace pm = planetmodel;
  const auto started = std::chrono::steady_clock::now();
  const auto top = pm::ReadJsonFile(config);
  const auto input = ReadEvolveInput(config);

  const auto deck = Deck::Read(input.model, FormatOf(input.format));
  if (!deck) Fail(top.At("model"), deck.error().message);
  const auto model = EarthModel::FromDeck(*deck);
  if (!model) Fail(top.At("model"), model.error().message);
  auto mesh_options = RadialMesh::Options{};
  mesh_options.lmax = input.lmax;
  const auto mesh = RadialMesh::Create(*model, mesh_options);
  if (!mesh) Fail(top.At("model"), mesh.error().message);
  const auto material = Material::Create(*model, *mesh);
  if (!material) Fail(top.At("model"), material.error().message);
  const auto rheology = Rheology::FromModel(*material, *model);
  if (!rheology) Fail(top.At("model"), rheology.error().message);

  ViscoelasticOperator op(*rheology, input.lmax);
  auto parameters = SeaLevelParameters{};
  if (input.hydrostatic_moments) {
    parameters =
        WithHydrostaticMoments(parameters, FluidTidalNumber(*material, 2));
  }
  SeaLevelSolver solver(SurfaceGrid(input.lmax), op.ElasticLoveNumbers(),
                        parameters);
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto fieldSize = grid.FieldSize();

  // the background state at the window's start, from the surface file
  const auto state_history = io::FieldHistory::Read(input.state, grid);
  if (!state_history) Fail(top.At("state").At("file"),
                           state_history.error().message);
  const auto ice_history = io::FieldHistory::Read(input.ice_history, grid);
  if (!ice_history) Fail(top.At("ice_history").At("file"),
                         ice_history.error().message);
  for (const auto* held : {&*state_history, &*ice_history}) {
    for (const auto name : {"ice_thickness", "topography"}) {
      if (held == &*state_history && !held->Has(name)) {
        Fail(top.At("state").At("file"),
             std::format("holds no field {}", name));
      }
    }
  }
  if (!ice_history->Has("ice_thickness")) {
    Fail(top.At("ice_history").At("file"), "holds no field ice_thickness");
  }
  const auto start_si = -input.start_kyr_bp * KYR;
  const auto ice0 = state_history->Values("ice_thickness", start_si);
  const auto topo0 = state_history->Values("topography", start_si);
  auto ice_nd = std::vector<double>(fieldSize);
  auto sl_nd = std::vector<double>(fieldSize);
  const auto flotation = parameters.ice_density / parameters.water_density;
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    auto sl = topo0[i] < 0.0 ? -topo0[i] : -topo0[i] + ice0[i];
    if (topo0[i] < 0.0 && ice0[i] > 0.0) sl += flotation * ice0[i];
    ice_nd[i] = ice0[i] / length;
    sl_nd[i] = sl / length;
  }
  auto state = solver.MakeState(std::move(ice_nd), std::move(sl_nd));

  // the time grid (solver units, zero at the window's start) and the
  // grounded ice load relative to the start
  auto times = std::vector<double>{};
  const auto span_s = (input.start_kyr_bp - input.end_kyr_bp) * KYR;
  for (auto n = 0; n < input.nodes; ++n) {
    times.push_back(span_s * n / (input.nodes - 1) / scales.Time());
  }
  const auto& C = state.oceanFunction;
  const auto rho_ice = solver.IceDensity();
  auto iceLoad = [&](double t) {
    const auto thickness =
        ice_history->Values("ice_thickness", start_si + t * scales.Time());
    auto load = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      load[i] =
          rho_ice * (1.0 - C[i]) * (thickness[i] - ice0[i]) / length;
    }
    return load;
  };

  // units out: SI fields, dates in kyr before present
  const auto toPotential = scales.Factor(PotentialDimensions);
  const auto toOmega = 1.0 / scales.Time();
  auto date_of = [&](double t) {
    return input.start_kyr_bp - t * scales.Time() / KYR;
  };
  auto history = std::vector<Node>{};
  auto keep = [&](const SeaLevelEvolution::Snapshot& snapshot) {
    auto kept = Node{};
    kept.date_kyr_bp = date_of(snapshot.t);
    kept.seaLevelChange = snapshot.seaLevelChange;
    kept.displacement = snapshot.displacement;
    kept.potential = snapshot.potential;
    for (auto& v : kept.seaLevelChange) v *= length;
    for (auto& v : kept.displacement) v *= length;
    for (auto& v : kept.potential) v *= toPotential;
    for (auto k = 0; k < 3; ++k) {
      kept.angularVelocity[k] = snapshot.angularVelocity[k] * toOmega;
    }
    history.push_back(std::move(kept));
  };

  // checkpointing, and an existing checkpoint to resume from
  const auto checkpointing = !input.checkpoint_directory.empty();
  const auto hash = HashOf(config);
  auto resumed = std::optional<ReadBack>{};
  if (checkpointing && input.restart) {
    resumed = ReadCheckpoint(input.checkpoint_directory, hash);
  }
  auto options = SeaLevelEvolution::Options{};
  options.scheme = input.scheme;
  options.rotation = input.rotation;
  options.rtol = input.rtol;
  options.on_snapshot = [&](std::size_t, const SeaLevelEvolution::Snapshot& s) {
    keep(s);
  };
  const auto base_node = resumed ? resumed->node : std::size_t{0};
  if (checkpointing) {
    options.checkpoint_stride = input.checkpoint_every;
    options.on_checkpoint = [&](std::size_t node,
                                const SeaLevelEvolution::Checkpoint& cp) {
      WriteCheckpoint(input.checkpoint_directory, hash, base_node + node, cp,
                      history);
      const auto done = base_node + node + 1 >=
                        static_cast<std::size_t>(input.nodes);
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

  SeaLevelEvolution evolution(op, solver, state, options);
  try {
    if (resumed) {
      history = std::move(resumed->history);
      const auto remaining = std::vector<double>(
          times.begin() + static_cast<std::ptrdiff_t>(resumed->node),
          times.end());
      if (remaining.size() >= 2) {  // a checkpoint at the last node is done
        static_cast<void>(
            evolution.Resume(resumed->checkpoint, remaining, iceLoad));
      }
    } else {
      static_cast<void>(evolution.Run(times, iceLoad));
    }
  } catch (const WalltimeReached& at) {
    return std::format(
        "checkpointed node {} of {} to {} at the wall-time limit; rerun "
        "the same config to resume",
        at.node, input.nodes - 1, input.checkpoint_directory.string());
  }

  // the histories as one surface file: SI node fields on a time axis in
  // seconds relative to the present (negative past)
  auto file = pm::SurfaceFile{};
  file.grid = pm::AngularGrid(
      std::vector<double>(grid.CoLatitudes().begin(), grid.CoLatitudes().end()),
      std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
      "gauss_legendre", input.lmax,
      std::vector<double>(grid.CoLatitudeWeights().begin(),
                          grid.CoLatitudeWeights().end()));
  file.source = std::format("sl3d evolve1d (sl3dpp {})", Version());
  file.title = "1D viscoelastic GIA histories";
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
  if (const auto parent = input.output.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  pm::WriteSurfaceFile(input.output, file);
  return std::format("wrote {} ({} nodes, {} to {} kyr BP{})",
                     input.output.string(), history.size(),
                     input.start_kyr_bp, input.end_kyr_bp,
                     resumed ? ", resumed" : "");
}

}  // namespace sl3dpp::app
