/**
 * @file StaticCommand.cpp
 * @brief `sl3d static`: the instantaneous (elastic) response of a
 * laterally heterogeneous elastic Earth to a surface load and a tidal
 * potential, with no sea level: the 3D static problem of
 * docs/adjoints.md §13j closed by the stepper's conjugate-gradient
 * solve on a branchless rheology (distributed over radial segments like
 * the evolutions), the radial model's response beside it, written as a
 * planetmodel surface file, an exchange table of the surface
 * coefficients for a cross-code comparison, and a model file of the
 * response inside the body for planetmodel's cutaways.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <format>
#include <fstream>
#include <numbers>
#include <optional>
#include <planetmodel/json.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Commands.hpp"
#include "Input.hpp"
#include "Ranks.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/KernelWriter.hpp"
#include "sl3dpp/io/ModelModuli.hpp"

namespace sl3dpp::app {

namespace {

using Complex = std::complex<double>;

ModelFormat ReadFormatWord(const Json& top) {
  if (!Has(top, "model_format")) return ModelFormat::Gia;
  return Choice<ModelFormat>(top.At("model_format"),
                             {{"maxwell", ModelFormat::Gia},
                              {"gia", ModelFormat::Gia},
                              {"mineos", ModelFormat::Mineos},
                              {"elastic", ModelFormat::Elastic}});
}

/// The per-degree m >= 0 columns of a real surface field's coefficients.
Viscoelastic3D::LoadCallable3D ColumnsOf(std::vector<Complex> coefficients,
                                         double factor) {
  return [coefficients = std::move(coefficients), factor](
             int l, double) -> std::optional<std::vector<Complex>> {
    auto columns = std::vector<Complex>(l + 1);
    for (auto m = 0; m <= l; ++m) {
      columns[m] = factor * coefficients[SurfaceGrid::Index(l, m)];
    }
    return columns;
  };
}

/// Per-degree columns back to a grid's coefficients, scaled.
std::vector<Complex> CoefficientsOf(
    const SurfaceGrid& grid, const std::map<int, std::vector<Complex>>& perDegree,
    double factor) {
  auto out = grid.ZeroCoefficients();
  for (const auto& [l, columns] : perDegree) {
    for (auto m = 0; m <= l; ++m) {
      out[SurfaceGrid::Index(l, m)] = factor * columns[m];
    }
  }
  return out;
}

std::string Today() {
  const auto now = std::chrono::floor<std::chrono::days>(
      std::chrono::system_clock::now());
  return std::format("{:%Y-%m-%d}", now);
}

}  // namespace

StaticInput ReadStaticInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  Allow(top, {"model", "model_format", "lmax", "moduli", "load", "tide",
              "solver", "transform_degrees", "output", "exchange_output",
              "volume_output"});
  const auto directory = config.parent_path();
  auto out = StaticInput{};
  out.model = PathOf(top, "model", directory);
  out.format = ReadFormatWord(top);
  out.lmax = Integer(top, "lmax");
  if (out.lmax < 1) Fail(top.At("lmax"), "is not a degree");
  if (Has(top, "moduli")) {
    const auto& moduli = top.At("moduli");
    Allow(moduli, {"file", "reference"});
    out.moduli = PathOf(moduli, "file", directory);
    if (Has(moduli, "reference")) {
      out.moduli_file_radial =
          Choice<bool>(moduli.At("reference"),
                       {{"lateral_mean", false}, {"file_radial", true}});
    }
  }
  if (Has(top, "load")) {
    const auto& load = top.At("load");
    Allow(load, {"latitude", "longitude", "width_deg", "amplitude_kg_m2"});
    out.load.set = true;
    out.load.site = {Number(load, "latitude"), Number(load, "longitude")};
    const auto width = Number(load, "width_deg");
    if (!(width > 0.0)) Fail(load.At("width_deg"), "is not a positive angle");
    out.load.width = width * std::numbers::pi / 180.0;
    out.load.amplitude = Number(load, "amplitude_kg_m2");
  }
  if (Has(top, "tide")) {
    const auto& tide = top.At("tide");
    Allow(tide, {"degree", "order", "amplitude_m2_s2", "imaginary_m2_s2"});
    out.tide.set = true;
    out.tide.degree = Integer(tide, "degree");
    if (out.tide.degree < 2 || out.tide.degree > out.lmax) {
      Fail(tide.At("degree"), "is not a tidal degree within [2, lmax]");
    }
    out.tide.order = Integer(tide, "order", 0);
    if (out.tide.order < 0 || out.tide.order > out.tide.degree) {
      Fail(tide.At("order"), "is not an order within [0, degree]");
    }
    out.tide.amplitude = Number(tide, "amplitude_m2_s2");
    out.tide.imaginary = Number(tide, "imaginary_m2_s2", 0.0);
    if (out.tide.order == 0 && out.tide.imaginary != 0.0) {
      Fail(tide.At("imaginary_m2_s2"), "is not zero at order 0 (a real field)");
    }
  }
  if (!out.load.set && !out.tide.set) {
    Fail(top, "needs a load, a tide or both");
  }
  if (Has(top, "solver")) {
    const auto& solver = top.At("solver");
    Allow(solver, {"rtol", "max_iterations"});
    out.rtol = Number(solver, "rtol", out.rtol);
    if (!(out.rtol > 0.0)) Fail(solver.At("rtol"), "is not a positive tolerance");
    out.max_iterations = Integer(solver, "max_iterations", out.max_iterations);
    if (out.max_iterations < 1) {
      Fail(solver.At("max_iterations"), "is not a positive count");
    }
  }
  if (Has(top, "transform_degrees")) {
    for (const auto& item : top.At("transform_degrees").AsArray()) {
      const auto value = item.AsNumber();
      if (value != std::floor(value) || value < 2 || value > out.lmax) {
        Fail(item, std::format("is not a degree in [2, lmax = {}]", out.lmax));
      }
      out.transform_degrees.push_back(static_cast<int>(value));
    }
  }
  out.output = PathOf(top, "output", directory);
  if (Has(top, "exchange_output")) {
    out.exchange_output = PathOf(top, "exchange_output", directory);
  }
  if (Has(top, "volume_output")) {
    out.volume_output = PathOf(top, "volume_output", directory);
  }
  return out;
}

std::string RunStatic3D(const std::filesystem::path& config) {
  namespace pm = planetmodel;
  const auto top = pm::ReadJsonFile(config);
  const auto input = ReadStaticInput(config);
  const auto& ranks = TheRanks();

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
  const auto elastic = Rheology::Create(*material, {});
  if (!elastic) Fail(top.At("model"), elastic.error().message);
  const auto& scales = model->ModelScales();

  // The lateral moduli, where given.
  auto moduli = std::optional<io::ModelModuli>{};
  if (!input.moduli.empty()) {
    auto read = io::ModelModuli::Read(
        input.moduli, *model,
        {.reference = input.moduli_file_radial
                          ? io::ModulusReference::FileRadial
                          : io::ModulusReference::LateralMean});
    if (!read) Fail(top.At("moduli").At("file"), read.error().message);
    moduli.emplace(std::move(*read));
  }
  const auto lateral = moduli && !moduli->IsRadial();

  // The forcing: a Gaussian cap of surface mass density (kg m^-2 at its
  // centre) and a tidal potential coefficient (m^2 s^-2), each made
  // non-dimensional under the model's scales.
  const SurfaceGrid grid(input.lmax);
  auto loadSi = grid.ZeroField();
  auto loadNd = grid.ZeroCoefficients();
  auto load = Viscoelastic3D::LoadCallable3D{};
  if (input.load.set) {
    const auto colat0 = input.load.site.CoLatitude();
    const auto lon0 = input.load.site.Longitude();
    loadSi = grid.FromFunction([&](double theta, double phi) {
      const auto c = std::cos(theta) * std::cos(colat0) +
                     std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
      const auto d = std::acos(std::clamp(c, -1.0, 1.0));
      return input.load.amplitude *
             std::exp(-0.5 * (d / input.load.width) * (d / input.load.width));
    });
    loadNd = grid.Expand(loadSi);
    for (auto& c : loadNd) c /= scales.Factor(Dimensions{.mass = 1, .length = -2});
    load = ColumnsOf(loadNd, 1.0);
  }
  auto tideNd = grid.ZeroCoefficients();
  auto tide = Viscoelastic3D::LoadCallable3D{};
  if (input.tide.set) {
    const auto amplitude =
        Complex{input.tide.amplitude, input.tide.imaginary} /
        scales.Factor(PotentialDimensions);
    tideNd[SurfaceGrid::Index(input.tide.degree, input.tide.order)] = amplitude;
    tide = [degree = input.tide.degree, order = input.tide.order, amplitude](
               int l, double) -> std::optional<std::vector<Complex>> {
      if (l != degree) return std::nullopt;
      auto columns = std::vector<Complex>(l + 1);
      columns[order] = amplitude;
      return columns;
    };
  }

  // The radial model's response, on every rank alike (its solves are the
  // 1D ones): the reference of the 3D problem, and the whole answer
  // without lateral moduli.
  const ElasticPerturbation3D reference(*material, input.lmax);
  auto sphForces = ElasticPerturbation3D::Forces{};
  if (input.load.set) sphForces = reference.LoadForces(loadNd);
  if (input.tide.set) {
    for (auto& [l, f] : reference.TideForces(tideNd)) {
      auto found = sphForces.find(l);
      if (found == sphForces.end()) {
        sphForces.emplace(l, std::move(f));
      } else {
        for (auto i = std::size_t{0}; i < f.size(); ++i) found->second[i] += f[i];
      }
    }
  }
  const auto x1 = reference.Solve(sphForces, {});

  // The 3D solve under lateral moduli: the stepper on a branchless
  // rheology, its conjugate gradients on the reference systems,
  // distributed over radial segments like the evolutions.
  auto op = std::optional<Viscoelastic3D>{};
  auto x = Viscoelastic3D::Solutions{};
  auto info = Viscoelastic3D::ElasticSolveInfo{};
  if (lateral) {
    // the rank cuts balance the work where the moduli are (a branchless
    // rheology has no viscous element to weight)
    auto probe = Viscoelastic3D::Options{};
    probe.transform_degrees = input.transform_degrees;
    probe.moduli = {.shear = moduli->Shear(), .bulk = moduli->Bulk()};
    const auto active = Viscoelastic3D::ActiveElementsFor(
        *elastic, input.lmax, reference.Coupling().Grid(), probe);
    auto options = OperatorOptions(input.lmax, input.transform_degrees, *elastic, active);
    options.solver = Viscoelastic3D::Options::StepSolver::ConjugateGradients;
    options.elastic_rtol = input.rtol;
    options.elastic_max_iterations = input.max_iterations;
    options.moduli = {.shear = moduli->Shear(), .bulk = moduli->Bulk()};
    op.emplace(*elastic, input.lmax, [](double, double, double) { return 0.0; },
               options);
    x = op->SolveElastic(op->ZeroState(), load, tide);
    info = op->LastElasticSolve();
    if (!info.converged) {
      throw std::runtime_error(std::format(
          "the 3D static solve did not converge: residual {} after {} "
          "iterations; loosen solver.rtol or raise solver.max_iterations",
          info.residual, info.iterations));
    }
  }
  const auto& x3 = lateral ? x : x1;
  auto surface = [&](Component component) {
    return lateral ? op->Surface(x, component) : reference.Surface(x1, component);
  };

  // The surface response: coefficients and SI fields on the grid.
  const auto length = scales.Length();
  const auto potential = scales.Factor(PotentialDimensions);
  const auto U = CoefficientsOf(grid, surface(Component::U), length);
  const auto Phi = CoefficientsOf(grid, surface(Component::Phi), potential);
  const auto V = CoefficientsOf(grid, surface(Component::V), length);
  const auto U1 = CoefficientsOf(grid, reference.Surface(x1, Component::U), length);
  const auto Phi1 =
      CoefficientsOf(grid, reference.Surface(x1, Component::Phi), potential);

  // The response inside the body, at the memory nodes of the radial
  // reference's coupling (every solid element but the centre's, whole on
  // every rank; the solutions' dof layout is the same): the radial
  // displacement, the deviatoric strain's magnitude sqrt(e:e), and the
  // lateral structure it answers to, sampled there.
  auto volume = std::vector<std::pair<std::string, std::vector<double>>>{};
  if (!input.volume_output.empty()) {
    const auto& coupling = reference.Coupling();
    auto fields = io::ResponseFieldsAtNodes(
        coupling, [&](int l) -> const DegreeOperator& { return reference.Spheroidal(l); },
        [&](int l) -> const ToroidalOperator& { return reference.Toroidal(l); }, x3);
    volume.emplace_back("radial_displacement", std::move(fields.radialDisplacement));
    volume.emplace_back("deviatoric_strain", std::move(fields.deviatoricStrain));
    if (lateral) {
      if (moduli->Shear()) {
        volume.emplace_back("mu_relative", io::SampleNodeField(coupling, moduli->Shear()));
      }
      if (moduli->Bulk()) {
        volume.emplace_back("kappa_relative", io::SampleNodeField(coupling, moduli->Bulk()));
      }
    }
  }
  if (!ranks.First()) return "";

  // The surface file.
  auto file = pm::SurfaceFile{};
  file.grid = pm::AngularGrid(
      std::vector<double>(grid.CoLatitudes().begin(), grid.CoLatitudes().end()),
      std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
      "gauss_legendre", input.lmax,
      std::vector<double>(grid.CoLatitudeWeights().begin(),
                          grid.CoLatitudeWeights().end()));
  file.source = std::format("sl3d static (sl3dpp {})", Version());
  file.title = lateral ? "static response of a laterally heterogeneous elastic Earth"
                       : "static response of a radial elastic Earth";
  file.meta.emplace("config", config.string());
  file.meta.emplace("model", input.model.string());
  if (!input.moduli.empty()) file.meta.emplace("moduli", input.moduli.string());
  file.meta.emplace("lateral", std::vector<long long>{lateral ? 1 : 0});
  if (input.load.set) {
    file.meta.emplace("load_latitude_deg",
                      std::vector<double>{input.load.site.latitude});
    file.meta.emplace("load_longitude_deg",
                      std::vector<double>{input.load.site.longitude});
    file.meta.emplace("load_width_deg",
                      std::vector<double>{input.load.width * 180.0 / std::numbers::pi});
    file.meta.emplace("load_amplitude_kg_m2",
                      std::vector<double>{input.load.amplitude});
  }
  if (input.tide.set) {
    file.meta.emplace("tide_degree", std::vector<long long>{input.tide.degree});
    file.meta.emplace("tide_order", std::vector<long long>{input.tide.order});
    file.meta.emplace("tide_amplitude_m2_s2",
                      std::vector<double>{input.tide.amplitude});
    file.meta.emplace("tide_imaginary_m2_s2",
                      std::vector<double>{input.tide.imaginary});
  }
  file.meta.emplace("solve_iterations",
                    std::vector<long long>{lateral ? info.iterations : 0});
  file.meta.emplace("solve_residual",
                    std::vector<double>{lateral ? info.residual : 0.0});
  const auto lengthDims = pm::units::LENGTH;
  const auto potentialDims = pm::Dimensions{.length = 2, .time = -2};
  auto put = [&file](std::string name, const std::vector<double>& values,
                     std::optional<pm::Dimensions> dims, std::string meaning) {
    file.variables.emplace(
        name, pm::SurfaceField(name, values,
                               {.dimensions = dims, .meaning = meaning}));
  };
  if (input.load.set) {
    put("load", loadSi, pm::Dimensions{.mass = 1, .length = -2},
        "the surface mass load applied");
  }
  put("displacement", grid.Synthesise(U), lengthDims,
      "radial surface displacement");
  put("tangential_potential", grid.Synthesise(V), lengthDims,
      "the tangential displacement's potential V (the displacement its "
      "surface gradient)");
  put("potential", grid.Synthesise(Phi), potentialDims,
      "gravitational potential perturbation at the surface");
  if (lateral) {
    put("displacement_radial_model", grid.Synthesise(U1), lengthDims,
        "radial surface displacement of the radial model under the same "
        "forcing");
    put("potential_radial_model", grid.Synthesise(Phi1), potentialDims,
        "the radial model's potential perturbation under the same forcing");
  }
  std::filesystem::create_directories(input.output.parent_path());
  pm::WriteSurfaceFile(input.output, file);
  auto wrote = std::format("wrote {}", input.output.string());

  // The exchange table: the surface coefficients, SI, with provenance.
  if (!input.exchange_output.empty()) {
    std::filesystem::create_directories(input.exchange_output.parent_path());
    auto out = std::ofstream(input.exchange_output);
    out << std::format(
        "# static surface response from sl3dpp ({}): the m >= 0 spherical "
        "harmonic coefficients of the radial displacement U (m) and the "
        "potential perturbation Phi (m^2 s^-2), real orthonormal harmonics "
        "with f_(l,-m) = (-1)^m conj(f_lm)\n",
        lateral ? "laterally heterogeneous elastic model" : "radial elastic model");
    out << std::format("# code sl3dpp version {} date {}\n", Version(), Today());
    out << std::format("# config {} model {} moduli {}\n", config.string(),
                       input.model.string(),
                       input.moduli.empty() ? "none" : input.moduli.string());
    out << std::format("radius {:.17g}\n", material->Radius() * length);
    out << std::format("surface_gravity {:.17g}\n",
                       material->SurfaceGravity() *
                           scales.Factor(Dimensions{.length = 1, .time = -2}));
    out << std::format("lmax {}\n", input.lmax);
    out << std::format("lateral {}\n", lateral ? 1 : 0);
    if (input.load.set) {
      out << std::format(
          "cap_latitude_deg {:.17g}\ncap_longitude_deg {:.17g}\ncap_width_deg "
          "{:.17g}\ncap_amplitude_kg_m2 {:.17g}\n",
          input.load.site.latitude, input.load.site.longitude,
          input.load.width * 180.0 / std::numbers::pi, input.load.amplitude);
    }
    if (input.tide.set) {
      out << std::format(
          "tide_degree {}\ntide_order {}\ntide_amplitude_m2_s2 {:.17g} {:.17g}\n",
          input.tide.degree, input.tide.order, input.tide.amplitude,
          input.tide.imaginary);
    }
    out << "columns l m U_re U_im Phi_re Phi_im\n";
    for (auto l = 0; l <= input.lmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto k = SurfaceGrid::Index(l, m);
        out << std::format("{} {} {:.17g} {:.17g} {:.17g} {:.17g}\n", l, m,
                           U[k].real(), U[k].imag(), Phi[k].real(),
                           Phi[k].imag());
      }
    }
    wrote += std::format(" and {}", input.exchange_output.string());
  }

  // The response inside the body as a model file.
  if (!input.volume_output.empty()) {
    auto fields = std::vector<io::NodeField>{};
    const auto strainDims = Dimensions{};
    for (const auto& [name, values] : volume) {
      auto meaning = std::string{};
      auto dims = Dimensions{};
      if (name == "radial_displacement") {
        meaning = "radial displacement of the static response";
        dims = Dimensions{.length = 1};
      } else if (name == "deviatoric_strain") {
        meaning = "the deviatoric strain's magnitude sqrt(e:e)";
        dims = strainDims;
      } else if (name == "mu_relative") {
        meaning = "the relative lateral perturbation of the shear modulus, "
                  "mu(x) / mu(r) - 1";
      } else {
        meaning = "the relative lateral perturbation of the bulk modulus, "
                  "kappa(x) / kappa(r) - 1";
      }
      fields.push_back({.name = name, .meaning = meaning, .dimensions = dims,
                        .values = values});
    }
    std::filesystem::create_directories(input.volume_output.parent_path());
    const auto written = io::WriteNodeFields(
        input.volume_output, reference.Coupling(), *mesh,
        *model, fields, {.name = "sl3d static response"});
    if (!written) throw std::runtime_error(written.error().message);
    wrote += std::format(" and {}", input.volume_output.string());
  }

  return lateral ? std::format("{} ({} conjugate-gradient iterations, residual {:.2e})",
                               wrote, info.iterations, info.residual)
                 : std::format("{} (radial model)", wrote);
}

}  // namespace sl3dpp::app
