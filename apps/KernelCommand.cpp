#include <cmath>
#include <format>
#include <fstream>
#include <planetmodel/json.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Commands.hpp"
#include "Common.hpp"
#include "Input.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/KernelWriter.hpp"
#include "sl3dpp/io/ModelViscosity.hpp"

namespace sl3dpp::app {

namespace {

using Type = MeasurementSpec::Type;

MeasurementSpec ReadSpec(const Json& item, std::size_t index) {
  auto out = MeasurementSpec{};
  out.type = Choice<Type>(
      item.At("type"),
      {{"sea_level_cap", Type::SeaLevelCap},
       {"sea_level_point", Type::SeaLevelPoint},
       {"displacement_cap", Type::DisplacementCap},
       {"displacement_point", Type::DisplacementPoint},
       {"potential_cap", Type::PotentialCap},
       {"potential_point", Type::PotentialPoint},
       {"stokes_coefficient", Type::StokesCoefficient},
       {"rotation", Type::Rotation},
       {"decay_time", Type::DecayTime}});
  out.name = String(item, "name", std::format("measurement_{}", index));
  switch (out.type) {
    case Type::SeaLevelCap:
    case Type::DisplacementCap:
    case Type::PotentialCap:
      Allow(item, {"type", "name", "latitude", "longitude", "width_deg",
                   "date_kyr_bp"});
      out.site = {Number(item, "latitude"), Number(item, "longitude")};
      out.width = Number(item, "width_deg") * std::numbers::pi / 180.0;
      if (!(out.width > 0.0)) {
        Fail(item.At("width_deg"), "is not a positive angle");
      }
      out.date_kyr_bp = Number(item, "date_kyr_bp", 0.0);
      break;
    case Type::SeaLevelPoint:
    case Type::DisplacementPoint:
    case Type::PotentialPoint:
      Allow(item, {"type", "name", "latitude", "longitude", "date_kyr_bp"});
      out.site = {Number(item, "latitude"), Number(item, "longitude")};
      out.date_kyr_bp = Number(item, "date_kyr_bp", 0.0);
      break;
    case Type::StokesCoefficient:
      Allow(item, {"type", "name", "degree", "order", "date_kyr_bp"});
      out.degree = Integer(item, "degree");
      out.order = Integer(item, "order");
      out.date_kyr_bp = Number(item, "date_kyr_bp", 0.0);
      break;
    case Type::Rotation:
      Allow(item, {"type", "name", "component", "date_kyr_bp"});
      out.component = Choice<AngularVelocityComponent>(
          item.At("component"), {{"x", AngularVelocityComponent::X},
                                 {"y", AngularVelocityComponent::Y},
                                 {"z", AngularVelocityComponent::Z}});
      out.date_kyr_bp = Number(item, "date_kyr_bp", 0.0);
      break;
    case Type::DecayTime:
      Allow(item, {"type", "name", "latitude", "longitude", "width_deg",
                   "window_start_kyr_bp", "window_end_kyr_bp",
                   "time_uncertainty_kyr", "value_uncertainty_m"});
      out.site = {Number(item, "latitude"), Number(item, "longitude")};
      out.width = Number(item, "width_deg") * std::numbers::pi / 180.0;
      if (!(out.width > 0.0)) {
        Fail(item.At("width_deg"), "is not a positive angle");
      }
      out.window_start_kyr_bp = Number(item, "window_start_kyr_bp");
      out.window_end_kyr_bp = Number(item, "window_end_kyr_bp", 0.0);
      if (!(out.window_start_kyr_bp > out.window_end_kyr_bp)) {
        Fail(item, "the decay window runs forward in time");
      }
      out.time_uncertainty_kyr = Number(item, "time_uncertainty_kyr");
      out.value_uncertainty_m = Number(item, "value_uncertainty_m");
      break;
  }
  return out;
}

/// What one measurement becomes against a run: the node-keyed duals,
/// the forward value, the unit of J (J_SI = factor * J_nd) and its SI
/// dimensions, and, for a decay time, the fit.
struct Built {
  std::map<std::size_t, Measurement> measurements;
  double value_nd = 0.0;
  double factor = 1.0;
  Dimensions objective = {};
  std::optional<DecayTimeFit> fit;
  std::vector<std::size_t> nodes;  ///< the nodes the duals sit on
};

/// The node a date names, which must lie on the grid.
std::size_t NodeOf(const Json& where, double date_kyr_bp,
                   const EvolveInput& run) {
  const auto step =
      (run.start_kyr_bp - run.end_kyr_bp) / (run.nodes - 1);
  const auto exact = (run.start_kyr_bp - date_kyr_bp) / step;
  const auto node = std::llround(exact);
  if (node < 0 || node >= run.nodes ||
      std::abs(exact - static_cast<double>(node)) > 1e-9 * run.nodes) {
    Fail(where, std::format("{} kyr BP is not a node of the times grid",
                            date_kyr_bp));
  }
  return static_cast<std::size_t>(node);
}

template <class Snapshot>
Built BuildMeasurement(const MeasurementSpec& spec, const Json& where,
                       const EvolveInput& run, const SeaLevelSolver& solver,
                       std::span<const Snapshot> snapshots) {
  const auto& grid = solver.Grid();
  const auto a = solver.Radius();
  const auto& scales = solver.SolverScales();
  auto out = Built{};
  auto single = [&](FieldChannel channel, std::vector<double> dual,
                    auto field_of) {
    const auto node = NodeOf(where, spec.date_kyr_bp, run);
    out.nodes = {node};
    out.value_nd = PairedValue(grid, a, dual, field_of(snapshots[node]));
    out.measurements.emplace(node,
                             FieldMeasurement(channel, std::move(dual)));
  };
  const auto sea = [](const Snapshot& s) -> std::span<const double> {
    return s.seaLevelChange;
  };
  const auto displacement = [](const Snapshot& s) -> std::span<const double> {
    return s.displacement;
  };
  const auto potential = [](const Snapshot& s) -> std::span<const double> {
    return s.potential;
  };
  switch (spec.type) {
    case Type::SeaLevelCap:
    case Type::SeaLevelPoint:
      out.factor = scales.Length();
      out.objective = Dimensions{.length = 1};
      single(FieldChannel::SeaLevel,
             spec.type == Type::SeaLevelCap
                 ? CapMeanDual(grid, a, spec.site, spec.width)
                 : PointDual(grid, a, spec.site),
             sea);
      break;
    case Type::DisplacementCap:
    case Type::DisplacementPoint:
      out.factor = scales.Length();
      out.objective = Dimensions{.length = 1};
      single(FieldChannel::Displacement,
             spec.type == Type::DisplacementCap
                 ? CapMeanDual(grid, a, spec.site, spec.width)
                 : PointDual(grid, a, spec.site),
             displacement);
      break;
    case Type::PotentialCap:
    case Type::PotentialPoint:
      out.factor = scales.Factor(PotentialDimensions);
      out.objective = PotentialDimensions;
      single(FieldChannel::Potential,
             spec.type == Type::PotentialCap
                 ? CapMeanDual(grid, a, spec.site, spec.width)
                 : PointDual(grid, a, spec.site),
             potential);
      break;
    case Type::StokesCoefficient:
      out.factor = scales.Factor(PotentialDimensions);
      out.objective = PotentialDimensions;
      single(FieldChannel::Potential,
             CoefficientDual(grid, a, spec.degree, spec.order), potential);
      break;
    case Type::Rotation: {
      const auto node = NodeOf(where, spec.date_kyr_bp, run);
      out.nodes = {node};
      out.factor = 1.0 / scales.Time();
      out.objective = Dimensions{.time = -1};
      out.value_nd =
          snapshots[node]
              .angularVelocity[static_cast<std::size_t>(spec.component)];
      out.measurements.emplace(node, RotationMeasurement(spec.component));
      break;
    }
    case Type::DecayTime: {
      // the site series over the window, in SI metres against kyr, the
      // fit's weights becoming node-keyed duals: J in kyr, SI seconds
      // through the factor
      const auto first = NodeOf(where, spec.window_start_kyr_bp, run);
      const auto last = NodeOf(where, spec.window_end_kyr_bp, run);
      if (last <= first) Fail(where, "the decay window holds no nodes");
      const auto dual = CapMeanDual(grid, a, spec.site, spec.width);
      const auto length = scales.Length();
      auto elapsed = std::vector<double>{};
      auto series = std::vector<double>{};
      const auto step =
          (run.start_kyr_bp - run.end_kyr_bp) / (run.nodes - 1);
      for (auto node = first; node <= last; ++node) {
        elapsed.push_back(step * static_cast<double>(node - first));
        series.push_back(
            length * PairedValue(grid, a, dual, sea(snapshots[node])));
        out.nodes.push_back(node);
      }
      auto fit = DecayTimeFit{};
      out.measurements = DecayTimeMeasurements(
          out.nodes, elapsed, series, FieldChannel::SeaLevel, dual,
          {.time_uncertainty = spec.time_uncertainty_kyr,
           .value_uncertainty = spec.value_uncertainty_m},
          &fit);
      // the weights are per metre of series: scale the duals so the
      // pairing's J is in kyr
      for (auto& [node, measurement] : out.measurements) {
        for (auto& v : measurement.seaLevelDual) v *= length;
      }
      out.value_nd = fit.decay_time;
      out.factor = KYR;
      out.objective = Dimensions{.time = 1};
      out.fit = fit;
      break;
    }
  }
  return out;
}

/// The ice-history kernels of one measurement as a surface file: the
/// thickness duals at the nodes, the quelled rate kernel on the
/// intervals, the delta amplitudes at the measurement's instants, and
/// the static baseline and initial-sea-level duals — all SI densities
/// pairing by int K delta dS (per metre of ice, or of initial sea
/// level), the objective in its SI unit.
void WriteIceKernels(const std::filesystem::path& path,
                     const SurfaceGrid& grid, int lmax,
                     const EvolveInput& run, const IceKernels& kernels,
                     const Built& built, double solver_length,
                     std::string_view command) {
  namespace pm = planetmodel;
  const auto fieldSize = grid.FieldSize();
  auto file = pm::SurfaceFile{};
  file.grid = pm::AngularGrid(
      std::vector<double>(grid.CoLatitudes().begin(), grid.CoLatitudes().end()),
      std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
      "gauss_legendre", lmax,
      std::vector<double>(grid.CoLatitudeWeights().begin(),
                          grid.CoLatitudeWeights().end()));
  file.source = std::format("sl3d {} (sl3dpp {})", command, Version());
  file.title = "ice-history sensitivity kernels";
  const auto step = (run.start_kyr_bp - run.end_kyr_bp) / (run.nodes - 1);
  auto axis = std::vector<double>{};
  for (auto n = 0; n < run.nodes; ++n) {
    axis.push_back(-(run.start_kyr_bp - step * n) * KYR);
  }
  file.time = axis;
  auto instants = std::vector<double>{};
  for (const auto node : built.nodes) {
    instants.push_back(axis[node]);
  }
  file.instant = instants;
  file.meta.emplace("time", std::string{"seconds relative to the present, "
                                        "negative in the past"});
  // SI: dJ_nd = a_nd^2 int K_nd dI_nd dOmega with dI_nd = dI_SI / L and
  // dS_SI = L^2 a_nd^2 dOmega, so K_SI = f_J K_nd / L^3; the rate and
  // static duals convert by the same factor (the time units cancel).
  const auto toKernel =
      built.factor / (solver_length * solver_length * solver_length);
  const auto dims = pm::Dimensions{.mass = built.objective.mass,
                                   .length = built.objective.length - 3,
                                   .time = built.objective.time};
  auto convert = [&](const std::vector<std::vector<double>>& rows,
                     double factor) {
    auto values = std::vector<double>{};
    values.reserve(rows.size() * fieldSize);
    for (const auto& row : rows) {
      for (const auto v : row) values.push_back(v * factor);
    }
    return values;
  };
  auto put = [&file](std::string name, std::vector<double> values,
                     std::string support, pm::Dimensions dims_,
                     std::string meaning) {
    auto options = pm::SurfaceFieldOptions{};
    options.support = std::move(support);
    options.dimensions = dims_;
    options.meaning = std::move(meaning);
    file.variables.emplace(
        name, pm::SurfaceField(name, std::move(values), options));
  };
  put("thickness_dual", convert(kernels.thickness, toKernel),
      "node", dims,
      "dJ per metre of grounded ice thickness per m^2, at each node");
  put("rate_dual", convert(kernels.rate, toKernel), "interval", dims,
      "dJ per (metre of thickness per second) per m^2, the quelled view");
  auto lumps = std::vector<std::vector<double>>{};
  for (const auto node : built.nodes) {
    const auto found = kernels.instantaneous.find(node);
    lumps.push_back(found == kernels.instantaneous.end()
                        ? std::vector<double>(fieldSize, 0.0)
                        : found->second);
  }
  put("instantaneous_dual", convert(lumps, toKernel), "instant", dims,
      "the delta amplitude at each measurement instant (load-dual space)");
  put("thickness_baseline_dual", convert({kernels.baseline}, toKernel),
      "static", dims, "dJ per metre of the initial ice thickness per m^2");
  put("initial_sea_level_dual",
      convert({kernels.initialSeaLevel}, toKernel), "static", dims,
      "dJ per metre of the initial sea level per m^2");
  if (const auto parent = path.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  pm::WriteSurfaceFile(path, file);
}

}  // namespace

KernelInput ReadKernelInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  auto allowed = RunMemberNames();
  for (const auto name : {"record_stride", "adjoint", "measurements",
                          "output"}) {
    allowed.push_back(name);
  }
  Allow(top, std::span(allowed));
  const auto directory = config.parent_path();
  auto out = KernelInput{};
  ReadRunMembers(top, directory, out.run);
  out.record_stride = Integer(top, "record_stride", 1);
  if (out.record_stride < 1) {
    Fail(top.At("record_stride"), "is not a positive stride");
  }
  if (Has(top, "adjoint")) {
    out.method = Choice<AdjointMethod>(
        top.At("adjoint"), {{"continuous", AdjointMethod::Continuous},
                            {"discrete", AdjointMethod::Discrete}});
  }
  const auto& measurements = top.At("measurements");
  const auto& items = measurements.AsArray();
  if (items.empty()) Fail(measurements, "holds no measurements");
  for (auto k = std::size_t{0}; k < items.size(); ++k) {
    out.measurements.push_back(ReadSpec(items[k], k));
  }
  out.output = PathOf(top, "output", directory);
  return out;
}

namespace {

/// The run and sweeps both kernel commands share. `sweep` maps one
/// measurement's node-keyed duals to its IceKernels, writing any extra
/// (viscosity) outputs of its own under the measurement's stem.
template <class Snapshot>
std::string RunKernelsWith(
    const std::filesystem::path& config, const KernelInput& input,
    const SeaLevelSolver& solver, std::span<const Snapshot> snapshots,
    std::string_view command,
    const std::function<IceKernels(
        const std::map<std::size_t, Measurement>&,
        const std::filesystem::path& stem, const Built&)>& sweep) {
  std::filesystem::create_directories(input.output);
  const auto top = planetmodel::ReadJsonFile(config);
  const auto& grid = solver.Grid();
  auto values = std::ofstream(input.output / "measurements.txt");
  values << std::format("# sl3d {} (sl3dpp {})\n", command, Version())
         << std::format("# created {}\n", CreatedNow())
         << std::format("# config {}\n", config.string())
         << "# name type value_si [decay: tau_kyr amplitude_m offset_m]\n";
  const auto& items = top.At("measurements").AsArray();
  for (auto k = std::size_t{0}; k < input.measurements.size(); ++k) {
    const auto& spec = input.measurements[k];
    const auto built = BuildMeasurement<Snapshot>(spec, items[k], input.run,
                                                  solver, snapshots);
    const auto stem = input.output / spec.name;
    const auto kernels = sweep(built.measurements, stem, built);
    WriteIceKernels(
        std::filesystem::path(stem.string() + "_ice_kernels.nc"), grid,
        input.run.lmax, input.run, kernels, built,
        solver.SolverScales().Length(), command);
    values << spec.name << ' ' << items[k].At("type").AsString() << ' '
           << std::format("{:.17g}", built.factor * built.value_nd);
    if (built.fit) {
      values << std::format(" {:.17g} {:.17g} {:.17g}", built.fit->decay_time,
                            built.fit->amplitude * solver.SolverScales().Length(),
                            built.fit->offset * solver.SolverScales().Length());
    }
    values << '\n';
  }
  return std::format("wrote {} measurement(s) of kernels to {}",
                     input.measurements.size(), input.output.string());
}

}  // namespace

std::string RunKernel1D(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  const auto input = ReadKernelInput(config);
  if (!input.run.viscosity.empty()) {
    Fail(top.At("viscosity"),
         "names a 3D viscosity; kernel1d takes the model's own column — "
         "use `sl3d kernel`");
  }
  auto earth = BuildEarth(input.run, top);
  const auto rheology = Rheology::FromModel(earth.material, earth.model);
  if (!rheology) Fail(top.At("model"), rheology.error().message);
  ViscoelasticOperator op(*rheology, input.run.lmax);
  SeaLevelSolver solver(SurfaceGrid(input.run.lmax), op.ElasticLoveNumbers(),
                        ParametersOf(input.run, earth.material));
  auto world = LoadWorld(input.run, top, solver);
  const auto times = MakeTimes(input.run, solver.SolverScales());
  const auto iceLoad = IceLoadOf(input.run, solver, world);
  auto options = SeaLevelEvolution::Options{};
  options.scheme = input.run.scheme;
  options.rotation = input.run.rotation;
  options.rtol = input.run.rtol;
  options.record_stride = input.record_stride;
  SeaLevelEvolution evolution(op, solver, world.background, options);
  const auto snapshots = evolution.Run(times, iceLoad);
  const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);

  const auto& material = earth.material;
  const auto& mesh = material.Mesh();
  const auto& matScales = material.ModelScales();
  AdjointSolver adjoint(evolution,
                        {.method = input.method, .rtol = input.run.rtol});
  return RunKernelsWith<SeaLevelEvolution::Snapshot>(
      config, input, solver, snapshots, "kernel1d",
      [&](const std::map<std::size_t, Measurement>& measurements,
          const std::filesystem::path& stem, const Built& built) {
        const auto kernels =
            adjoint.IceHistoryKernels(times, measurements, &trajectory);
        // the radial viscosity kernels as densities in radius, SI:
        // dJ = int K(r) d eta(r) dr per branch
        const auto eta = kernels.Viscosity(*rheology);
        const auto etaSi = matScales.Factor(ViscosityDimensions);
        const auto rSi = matScales.Length();
        const auto w = mesh.Weights();
        auto out = std::ofstream(stem.string() + "_viscosity_kernel.txt");
        out << std::format("# sl3d kernel1d (sl3dpp {})\n", Version())
            << "# dJ = int K(r) d eta(r) dr, SI (J in its SI unit)\n"
            << "# r_m";
        for (auto b = std::size_t{0}; b < eta.size(); ++b) {
          out << std::format(" K_branch_{}", b);
        }
        out << '\n';
        for (auto e = 0; e < mesh.ElementCount(); ++e) {
          const auto jac = mesh.Jacobian(e);
          for (auto q = 0; q < mesh.NGll(); ++q) {
            const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
            out << std::format("{:.17g}", mesh.Radius(e, q) * rSi);
            for (auto b = std::size_t{0}; b < eta.size(); ++b) {
              out << std::format(
                  " {:.17g}",
                  built.factor * eta[b][i] / (w[q] * jac) / (etaSi * rSi));
            }
            out << '\n';
          }
        }
        return kernels;
      });
}

std::string RunKernel3D(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  const auto input = ReadKernelInput(config);
  if (input.run.viscosity.empty()) {
    Fail(top, "needs a viscosity section: a planetmodel model file whose "
              "viscosity drives the 3D run");
  }
  if (input.method == AdjointMethod::Discrete) {
    Fail(top.At("adjoint"), "the 3D sweep is the continuous adjoint alone");
  }
  auto earth = BuildEarth(input.run, top);
  auto mv_options = io::ModelViscosityOptions{};
  mv_options.reference = input.run.viscosity_file_radial
                             ? io::ViscosityReference::FileRadial
                             : io::ViscosityReference::GeometricMean;
  const auto viscosity =
      io::ModelViscosity::Read(input.run.viscosity, earth.model, mv_options);
  if (!viscosity) Fail(top.At("viscosity").At("file"),
                       viscosity.error().message);
  const auto rheology =
      Rheology::Maxwell(earth.material, viscosity->Viscosity());
  if (!rheology) Fail(top.At("viscosity").At("file"),
                      rheology.error().message);
  Viscoelastic3D op(*rheology, input.run.lmax, viscosity->Delta());
  SeaLevelSolver solver(SurfaceGrid(input.run.lmax), op.ElasticLove(),
                        ParametersOf(input.run, earth.material));
  auto world = LoadWorld(input.run, top, solver);
  const auto times = MakeTimes(input.run, solver.SolverScales());
  const auto iceLoad = IceLoadOf(input.run, solver, world);
  auto options = SeaLevelEvolution3D::Options{};
  options.scheme = input.run.scheme;
  options.rotation = input.run.rotation;
  options.rtol = input.run.rtol;
  options.coupled_rtol = input.run.coupled_rtol;
  options.max_coupled_iterations = input.run.max_coupled_iterations;
  options.record_stride = input.record_stride;
  SeaLevelEvolution3D evolution(op, solver, world.background, options);
  const auto snapshots = evolution.Run(times, iceLoad);
  for (const auto& snapshot : snapshots) {
    if (!snapshot.converged) {
      throw std::runtime_error(std::format(
          "the coupled fixed point did not converge at t = {}", snapshot.t));
    }
  }
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);

  const auto& matScales = earth.material.ModelScales();
  AdjointSolver3D adjoint(evolution, {.rtol = input.run.rtol});
  return RunKernelsWith<SeaLevelEvolution3D::Snapshot>(
      config, input, solver, snapshots, "kernel",
      [&](const std::map<std::size_t, Measurement>& measurements,
          const std::filesystem::path& stem, const Built& built) {
        const auto kernels =
            adjoint.IceHistoryKernels(times, measurements, &trajectory);
        // the 3D log-viscosity kernels as a planetmodel model file, the
        // values made non-dimensional under the model's own scales
        const auto toModel = built.factor / matScales.Factor(built.objective);
        auto scaled = std::vector<std::vector<double>>{};
        auto fields = std::vector<io::KernelField>{};
        for (auto b = std::size_t{0}; b < kernels.logViscosityKernel3D.size();
             ++b) {
          auto values = kernels.logViscosityKernel3D[b];
          for (auto& v : values) v *= toModel;
          scaled.push_back(std::move(values));
          fields.push_back(io::KernelField{
              .name = std::format("K_ln_eta_{}", b),
              .meaning = std::format(
                  "dJ per unit ln(viscosity) of branch {}, per unit volume",
                  b),
              .objective = built.objective,
              .values = scaled.back()});
        }
        const auto written = io::WriteKernels(
            std::filesystem::path(stem.string() + "_viscosity_kernels.nc"),
            op, earth.model, fields, {});
        if (!written) throw std::runtime_error(written.error().message);
        return kernels.ice;
      });
}

}  // namespace sl3dpp::app
