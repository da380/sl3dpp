/**
 * @file ResponseCommand.cpp
 * @brief `sl3d response`: the response of the planet to a prescribed
 * surface load and external tidal potential, with no ocean. With no
 * `times` block it is the instantaneous elastic response: the radial
 * model's, or, under lateral moduli, the 3D static problem of
 * docs/adjoints.md §13j closed by the stepper's conjugate-gradient solve
 * on a branchless rheology (distributed over radial segments like the
 * evolutions) with the radial model's response beside it. It writes a
 * planetmodel surface file, an exchange table of the surface coefficients
 * for a cross-code comparison, and a model file of the response inside
 * the body for planetmodel's cutaways.
 *
 * With a `times` block it is the viscoelastic response, stepped through
 * the block's nodes under the forcing at each node: the radial model's
 * own stepper (ViscoelasticOperator) per degree, all orders as columns,
 * or, under a lateral viscosity or lateral moduli, the 3D stepper
 * (Viscoelastic3D) distributed like the evolutions, with the radial
 * reference model stepped beside it. The first node is the unrelaxed
 * solve with no memory before it; each step ends under the forcing at its
 * end node. It writes a surface file with a slice per node and an
 * exchange table per node.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <format>
#include <fstream>
#include <map>
#include <numbers>
#include <optional>
#include <planetmodel/json.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Commands.hpp"
#include "Input.hpp"
#include "Ranks.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/KernelWriter.hpp"
#include "sl3dpp/io/ModelModuli.hpp"
#include "sl3dpp/io/ModelViscosity.hpp"

namespace sl3dpp::app {

namespace {

using Complex = std::complex<double>;

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

/// The per-degree columns of the degrees in `degrees` alone.
Viscoelastic3D::LoadCallable3D ColumnsAt(std::vector<Complex> coefficients,
                                         std::set<int> degrees) {
  return [coefficients = std::move(coefficients), degrees = std::move(degrees)](
             int l, double) -> std::optional<std::vector<Complex>> {
    if (!degrees.contains(l)) return std::nullopt;
    auto columns = std::vector<Complex>(l + 1);
    for (auto m = 0; m <= l; ++m) {
      columns[m] = coefficients[SurfaceGrid::Index(l, m)];
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

/// The forcing's single cap, where its load is exactly one.
const LoadTerm* SingleCap(const Forcing& forcing) {
  return forcing.load.size() == 1 &&
                 forcing.load.front().kind == LoadTerm::Kind::Cap
             ? &forcing.load.front()
             : nullptr;
}

/// The forcing's single tide coefficient, where its tide is exactly one.
const TideTerm* SingleTide(const Forcing& forcing) {
  return forcing.tide.size() == 1 ? &forcing.tide.front() : nullptr;
}

/// A run's Gauss-Legendre grid as a surface file states it.
planetmodel::AngularGrid AngularGridOf(const SurfaceGrid& grid) {
  return planetmodel::AngularGrid(
      std::vector<double>(grid.CoLatitudes().begin(), grid.CoLatitudes().end()),
      std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
      "gauss_legendre", grid.LMax(),
      std::vector<double>(grid.CoLatitudeWeights().begin(),
                          grid.CoLatitudeWeights().end()));
}

/// The forcing in a surface file's metadata: the counts of load terms and
/// tide coefficients, each term described (`; ` between them) and, where
/// the load is exactly one cap or the tide exactly one coefficient, its
/// members.
void AddForcingMeta(planetmodel::SurfaceFile& file, const Forcing& forcing,
                    const std::vector<std::string>& described) {
  file.meta.emplace(
      "load_terms",
      std::vector<long long>{static_cast<long long>(forcing.load.size())});
  file.meta.emplace(
      "tide_coefficients",
      std::vector<long long>{static_cast<long long>(forcing.tide.size())});
  auto forcingText = std::string{};
  for (const auto& line : described) {
    forcingText += (forcingText.empty() ? "" : "; ") + line;
  }
  file.meta.emplace("forcing", forcingText);
  if (const auto* cap = SingleCap(forcing)) {
    file.meta.emplace("load_latitude_deg",
                      std::vector<double>{cap->site.latitude});
    file.meta.emplace("load_longitude_deg",
                      std::vector<double>{cap->site.longitude});
    file.meta.emplace("load_width_deg", std::vector<double>{cap->width * 180.0 /
                                                            std::numbers::pi});
    file.meta.emplace("load_amplitude_kg_m2",
                      std::vector<double>{cap->amplitude});
  }
  if (const auto* single = SingleTide(forcing)) {
    file.meta.emplace("tide_degree", std::vector<long long>{single->degree});
    file.meta.emplace("tide_order", std::vector<long long>{single->order});
    file.meta.emplace("tide_amplitude_m2_s2",
                      std::vector<double>{single->amplitude});
    file.meta.emplace("tide_imaginary_m2_s2",
                      std::vector<double>{single->imaginary});
  }
}

/// What a viscoelastic run's exchange path holds where a node's index goes.
constexpr auto kNodeField = std::string_view{"{node}"};

/// Creates the directory an output goes into, where its path names one.
void CreateParent(const std::filesystem::path& path) {
  if (const auto parent = path.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

/// The named scalars an exchange table starts with, a line each: the
/// body's radius (m) and surface gravity (m s^-2), lmax, whether the model
/// is lateral and the forcing's counts; then, where the load is exactly
/// one cap, its members, and where the tide is exactly one coefficient,
/// its own.
std::string SharedScalars(double radius, double gravity, int lmax, bool lateral,
                          const Forcing& forcing) {
  auto out = std::format("radius {:.17g}\n", radius);
  out += std::format("surface_gravity {:.17g}\n", gravity);
  out += std::format("lmax {}\n", lmax);
  out += std::format("lateral {}\n", lateral ? 1 : 0);
  out += std::format("load_terms {}\ntide_coefficients {}\n",
                     forcing.load.size(), forcing.tide.size());
  if (const auto* cap = SingleCap(forcing)) {
    out += std::format(
        "cap_latitude_deg {:.17g}\ncap_longitude_deg {:.17g}\ncap_width_deg "
        "{:.17g}\ncap_amplitude_kg_m2 {:.17g}\n",
        cap->site.latitude, cap->site.longitude,
        cap->width * 180.0 / std::numbers::pi, cap->amplitude);
  }
  if (const auto* single = SingleTide(forcing)) {
    out += std::format(
        "tide_degree {}\ntide_order {}\ntide_amplitude_m2_s2 {:.17g} {:.17g}\n",
        single->degree, single->order, single->amplitude, single->imaginary);
  }
  return out;
}

/// An exchange table: a provenance header (what the table holds,
/// `heading` naming the response; the code's version and the date;
/// `sources`, the config and the model's files; a line per forcing term),
/// the named scalars, and a row of SI coefficients of U and Phi per
/// (l, m >= 0), %.17g.
void WriteExchangeTable(const std::filesystem::path& path,
                        std::string_view heading, std::string_view sources,
                        const std::vector<std::string>& described,
                        std::string_view scalars, int lmax,
                        const std::vector<Complex>& U,
                        const std::vector<Complex>& Phi) {
  CreateParent(path);
  auto out = std::ofstream(path);
  out << std::format(
      "# {}: the m >= 0 spherical harmonic coefficients of the radial "
      "displacement U (m) and the potential perturbation Phi (m^2 s^-2), "
      "complex orthonormal harmonics with the Condon-Shortley phase of a "
      "real field, f_(l,-m) = (-1)^m conj(f_lm)\n",
      heading);
  out << std::format("# code sl3dpp version {} date {}\n", Version(), Today());
  out << std::format("# {}\n", sources);
  for (const auto& line : described) out << "# forcing " << line << '\n';
  out << scalars;
  out << "columns l m U_re U_im Phi_re Phi_im\n";
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto k = SurfaceGrid::Index(l, m);
      out << std::format("{} {} {:.17g} {:.17g} {:.17g} {:.17g}\n", l, m,
                         U[k].real(), U[k].imag(), Phi[k].real(),
                         Phi[k].imag());
    }
  }
  if (!out) {
    throw std::runtime_error(std::format("cannot write {}", path.string()));
  }
}

}  // namespace

ResponseInput ReadResponseInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  Allow(top, {"model", "lmax", "times", "forcing", "solver", "output"});
  const auto directory = config.parent_path();
  const auto timed = Has(top, "times");
  auto out = ResponseInput{};
  const auto& model = top.At("model");
  if (!timed && Has(model, "viscosity")) {
    Fail(model.At("viscosity"),
         "is a viscosity, which needs a `times` block (a viscoelastic "
         "response); an elastic response has no use for it");
  }
  out.model = ReadModelBlock(model, directory);
  if (timed && out.model.viscosity.empty() &&
      out.model.format != ModelFormat::Gia) {
    Fail(model.At("format"),
         "names a deck format with no viscosity column, which a "
         "viscoelastic response (a `times` block) needs: give a \"maxwell\" "
         "deck with a viscosity, or a `model.viscosity` file");
  }
  out.lmax = Integer(top, "lmax");
  if (out.lmax < 1) Fail(top.At("lmax"), "is not a degree");
  if (timed) {
    out.times = ReadTimesBlock(top.At("times"), TimesForm::Forwards);
  }
  out.forcing = ReadForcing(top.At("forcing"), directory, out.lmax, timed);
  if (Has(top, "solver")) {
    const auto& solver = top.At("solver");
    if (!timed) {
      for (const auto* name : {"scheme", "step"}) {
        if (Has(solver, name)) {
          Fail(solver.At(name),
               "sets the time step, which needs a `times` block (a "
               "viscoelastic response)");
        }
      }
    }
    out.solver = ReadSolverBlock(solver, out.lmax, SolverOf::Response);
  }
  const auto& output = top.At("output");
  out.output = ReadOutputBlock(output, directory);
  const auto perNode =
      out.output.exchange.string().find(kNodeField) != std::string::npos;
  if (!timed) {
    if (perNode) {
      Fail(output.At("exchange"),
           "holds `{node}`, which names a table per time node: it needs a "
           "`times` block");
    }
    if (Has(output, "exchange_nodes")) {
      Fail(output.At("exchange_nodes"),
           "chooses time nodes, which need a `times` block");
    }
    return out;
  }
  if (Has(output, "volume")) {
    Fail(output.At("volume"), "is not written for a viscoelastic run yet");
  }
  if (Has(output, "exchange") && !perNode) {
    Fail(output.At("exchange"),
         "lacks `{node}`: a viscoelastic run writes a table per time node, "
         "`{node}` replaced by the node's index");
  }
  if (Has(output, "exchange_nodes")) {
    const auto& nodes = output.At("exchange_nodes");
    if (!Has(output, "exchange")) {
      Fail(nodes, "chooses exchange tables, which need `output.exchange`");
    }
    const auto last = static_cast<int>(out.times->nodes_kyr.size()) - 1;
    for (auto k = std::size_t{0}; k < out.output.exchange_nodes.size(); ++k) {
      if (out.output.exchange_nodes[k] > last) {
        Fail(nodes.At(k), std::format("is not a node index in [0, {}]", last));
      }
    }
  }
  return out;
}

namespace {

/// The elastic response: the run without a `times` block.
std::string RunElastic(const std::filesystem::path& config,
                       const planetmodel::Json& top,
                       const ResponseInput& input) {
  namespace pm = planetmodel;
  const auto& ranks = TheRanks();
  const auto& deckMember = top.At("model").At("deck");
  const auto& forcing = input.forcing;
  const auto hasLoad = !forcing.load.empty();
  const auto hasTide = !forcing.tide.empty();
  const auto& solver = input.solver;

  const auto deck = Deck::Read(input.model.deck, FormatOf(input.model.format));
  if (!deck) Fail(deckMember, deck.error().message);
  const auto model = EarthModel::FromDeck(*deck);
  if (!model) Fail(deckMember, model.error().message);
  auto mesh_options = RadialMesh::Options{};
  mesh_options.lmax = input.lmax;
  const auto mesh = RadialMesh::Create(*model, mesh_options);
  if (!mesh) Fail(deckMember, mesh.error().message);
  const auto material = Material::Create(*model, *mesh);
  if (!material) Fail(deckMember, material.error().message);
  const auto elastic = Rheology::Create(*material, {});
  if (!elastic) Fail(deckMember, elastic.error().message);
  const auto& scales = model->ModelScales();

  // The lateral moduli, where given.
  auto moduli = std::optional<io::ModelModuli>{};
  if (!input.model.moduli.empty()) {
    auto read = io::ModelModuli::Read(
        input.model.moduli, *model,
        {.reference = input.model.moduli_file_radial
                          ? io::ModulusReference::FileRadial
                          : io::ModulusReference::LateralMean});
    if (!read) Fail(top.At("model").At("moduli").At("file"), read.error().message);
    moduli.emplace(std::move(*read));
  }
  const auto lateral = moduli && !moduli->IsRadial();

  // The forcing on the grid (SI), its load made non-dimensional under the
  // model's scales, kg m^-2 to the mass and length scales, and its tide
  // m^2 s^-2 to the potential's. Every grid of the run blocks its
  // transforms for the configured cache.
  const auto cacheBytes = solver.cache_mb << 20;
  const SurfaceGrid grid(input.lmax, {.cache_bytes = cacheBytes});
  const auto built = BuildForcing(forcing, grid, top.At("forcing"));
  auto loadNd = grid.ZeroCoefficients();
  auto load = Viscoelastic3D::LoadCallable3D{};
  if (hasLoad) {
    loadNd = built.loadCoefficients;
    for (auto& c : loadNd) c /= scales.Factor(Dimensions{.mass = 1, .length = -2});
    load = ColumnsOf(loadNd, 1.0);
  }
  auto tideNd = grid.ZeroCoefficients();
  auto tide = Viscoelastic3D::LoadCallable3D{};
  if (hasTide) {
    tideNd = built.tide;
    for (auto& c : tideNd) c /= scales.Factor(PotentialDimensions);
    auto degrees = std::set<int>{};
    for (const auto& term : forcing.tide) degrees.insert(term.degree);
    tide = ColumnsAt(tideNd, std::move(degrees));
  }

  // The radial model's response, on every rank alike (its solves are the
  // 1D ones): the reference of the 3D problem, and the whole answer
  // without lateral moduli.
  const ElasticPerturbation3D reference(*material, input.lmax,
                                        {.cache_bytes = cacheBytes});
  auto sphForces = ElasticPerturbation3D::Forces{};
  if (hasLoad) sphForces = reference.LoadForces(loadNd);
  if (hasTide) {
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
    probe.transform_degrees = solver.transform_degrees;
    probe.cache_bytes = cacheBytes;
    probe.moduli = {.shear = moduli->Shear(), .bulk = moduli->Bulk()};
    const auto active = Viscoelastic3D::ActiveElementsFor(
        *elastic, input.lmax, reference.Coupling().Grid(), probe);
    auto options = OperatorOptions(input.lmax, solver.transform_degrees,
                                   solver.cache_mb, *elastic, active);
    options.solver = Viscoelastic3D::Options::StepSolver::ConjugateGradients;
    options.elastic_rtol = solver.step_rtol;
    options.elastic_max_iterations = solver.step_max_iterations;
    options.moduli = {.shear = moduli->Shear(), .bulk = moduli->Bulk()};
    op.emplace(*elastic, input.lmax, [](double, double, double) { return 0.0; },
               options);
    x = op->SolveElastic(op->ZeroState(), load, tide);
    info = op->LastElasticSolve();
    if (!info.converged) {
      throw std::runtime_error(std::format(
          "the 3D elastic solve did not converge: residual {} after {} "
          "iterations; loosen solver.step_rtol or raise "
          "solver.step_max_iterations",
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
  if (!input.output.volume.empty()) {
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

  // What the config was taken to mean, for the summary and the outputs.
  const auto dimension =
      lateral ? std::string{"3-D (lateral moduli)"}
              : (moduli ? std::string{"1-D (the moduli file is radial)"}
                        : std::string{"1-D"});
  const auto meaning =
      std::format("{}, elastic, {}", dimension, Summary(forcing));
  auto described = std::vector<std::string>{};
  for (const auto& term : forcing.load) described.push_back(Describe(term));
  for (const auto& term : forcing.tide) described.push_back(Describe(term));

  // The surface file.
  auto file = pm::SurfaceFile{};
  file.grid = AngularGridOf(grid);
  file.source = std::format("sl3d response (sl3dpp {})", Version());
  file.title = lateral ? "elastic response of a laterally heterogeneous Earth"
                       : "elastic response of a radial Earth";
  file.meta.emplace("config", config.string());
  file.meta.emplace("model", input.model.deck.string());
  if (!input.model.moduli.empty()) {
    file.meta.emplace("moduli", input.model.moduli.string());
  }
  file.meta.emplace("lateral", std::vector<long long>{lateral ? 1 : 0});
  AddForcingMeta(file, forcing, described);
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
  if (hasLoad) {
    put("load", built.load, pm::Dimensions{.mass = 1, .length = -2},
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
  CreateParent(input.output.surface);
  pm::WriteSurfaceFile(input.output.surface, file);
  auto wrote = std::format("wrote {}", input.output.surface.string());

  // The exchange table: the surface coefficients, SI, with provenance.
  if (!input.output.exchange.empty()) {
    WriteExchangeTable(
        input.output.exchange,
        std::format("elastic surface response from sl3dpp ({})",
                    lateral ? "laterally heterogeneous elastic model"
                            : "radial elastic model"),
        std::format(
            "config {} model {} moduli {}", config.string(),
            input.model.deck.string(),
            input.model.moduli.empty() ? "none" : input.model.moduli.string()),
        described,
        SharedScalars(material->Radius() * length,
                      material->SurfaceGravity() *
                          scales.Factor(Dimensions{.length = 1, .time = -2}),
                      input.lmax, lateral, forcing),
        input.lmax, U, Phi);
    wrote += std::format(" and {}", input.output.exchange.string());
  }

  // The response inside the body as a model file.
  if (!input.output.volume.empty()) {
    auto fields = std::vector<io::NodeField>{};
    const auto strainDims = Dimensions{};
    for (const auto& [name, values] : volume) {
      auto meaning = std::string{};
      auto dims = Dimensions{};
      if (name == "radial_displacement") {
        meaning = "radial displacement of the elastic response";
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
    CreateParent(input.output.volume);
    const auto written = io::WriteNodeFields(
        input.output.volume, reference.Coupling(), *mesh,
        *model, fields, {.name = "sl3d response"});
    if (!written) throw std::runtime_error(written.error().message);
    wrote += std::format(" and {}", input.output.volume.string());
  }

  return lateral ? std::format("{}; {} ({} conjugate-gradient iterations, "
                               "residual {:.2e})",
                               meaning, wrote, info.iterations, info.residual)
                 : std::format("{}; {}", meaning, wrote);
}

/// The load and tide at one node, non-dimensional under the model's
/// scales.
struct NodeForcing {
  std::vector<Complex> load;
  std::vector<Complex> tide;
};

/// The surface response at one node: the coefficients of the radial
/// displacement U (m), the tangential displacement's potential V (m) and
/// the potential perturbation Phi (m^2 s^-2).
struct NodeSurface {
  std::vector<Complex> U, V, Phi;
};

/// The radial model of a rheology stepped through the nodes of `times`
/// under the forcing `forcingAt` gives at each node (non-dimensional
/// under the model's scales): the library's evolution on a body without
/// an ocean and without rotational feedback, which is the solid body
/// stepped under the forcing — the unrelaxed solve at the first node,
/// with no memory before it, then a step to each node ending under that
/// node's forcing. The surface response at every node, SI.
std::vector<NodeSurface> StepRadial(
    const Rheology& rheology, int lmax, const SurfaceGrid& grid,
    const Scales& scales, const TimesBlock& times, Scheme scheme,
    const std::function<NodeForcing(std::size_t)>& forcingAt, bool hasLoad,
    bool hasTide) {
  const ViscoelasticOperator op(rheology, lmax);
  const GRDSolver solver(grid, op.ElasticLoveNumbers(), SeaLevelParameters{});
  const auto& units = solver.SolverScales();
  const auto loadDims = Dimensions{.mass = 1, .length = -2};
  const auto toSolverLoad = scales.Factor(loadDims) / units.Factor(loadDims);
  const auto toSolverPotential =
      scales.Factor(PotentialDimensions) / units.Factor(PotentialDimensions);
  const auto toTime = KYR / units.Time();
  auto nodeTimes = std::vector<double>(times.nodes_kyr.size());
  for (auto n = std::size_t{0}; n < nodeTimes.size(); ++n) {
    nodeTimes[n] = times.nodes_kyr[n] * toTime;
  }

  // The evolution asks for the forcing by time, at the nodes in order;
  // each node's is built once.
  auto built = std::optional<std::pair<std::size_t, NodeForcing>>{};
  auto at = [&](double t) -> const NodeForcing& {
    const auto found = std::ranges::lower_bound(nodeTimes, t);
    const auto n = static_cast<std::size_t>(found - nodeTimes.begin());
    if (found == nodeTimes.end() || *found != t) {
      throw std::logic_error("the forcing was asked for away from a node");
    }
    if (!built || built->first != n) built.emplace(n, forcingAt(n));
    return built->second;
  };
  auto forcing = Evolution::Forcing{};
  // every node's forcing is built, whichever channels it holds
  forcing.directLoad = [&](double t) {
    const auto& node = at(t);
    if (!hasLoad) return std::vector<double>(grid.FieldSize(), 0.0);
    auto coefficients = node.load;
    for (auto& c : coefficients) c *= toSolverLoad;
    return grid.Synthesise(coefficients);
  };
  if (hasTide) {
    forcing.tide = [&](double t) {
      auto coefficients = at(t).tide;
      for (auto& c : coefficients) c *= toSolverPotential;
      return coefficients;
    };
  }

  Evolution evolution(op, solver, solver.NoOceanState(),
                      {.scheme = scheme,
                       .rotation = RotationalFeedback::None,
                       .record_surface = true});
  const auto length = units.Length();
  const auto potential = units.Factor(PotentialDimensions);
  auto out = std::vector<NodeSurface>{};
  for (const auto& snapshot : evolution.Run(nodeTimes, forcing)) {
    auto node = NodeSurface{.U = snapshot.surface->U,
                            .V = snapshot.surface->V,
                            .Phi = snapshot.surface->Phi};
    for (auto& c : node.U) c *= length;
    for (auto& c : node.V) c *= length;
    for (auto& c : node.Phi) c *= potential;
    out.push_back(std::move(node));
  }
  return out;
}

/// A step's channels from the channels at its start and at its end: the
/// start's before the step's middle, the end's from it on. A step reads
/// its start-time channels only to solve for the start strain, which a
/// state that carries its strain does not need.
Viscoelastic3D::LoadCallable3D Across(Viscoelastic3D::LoadCallable3D before,
                                      Viscoelastic3D::LoadCallable3D after,
                                      double middle) {
  if (!before && !after) return {};
  return [before = std::move(before), after = std::move(after), middle](
             int l, double t) -> std::optional<std::vector<Complex>> {
    const auto& channel = t < middle ? before : after;
    return channel ? channel(l, t) : std::nullopt;
  };
}

/// The viscoelastic response: the run with a `times` block.
std::string RunViscoelastic(const std::filesystem::path& config,
                            const planetmodel::Json& top,
                            const ResponseInput& input) {
  namespace pm = planetmodel;
  const auto& ranks = TheRanks();
  const auto& times = *input.times;
  const auto nodes = times.nodes_kyr.size();
  const auto& modelMember = top.At("model");
  const auto& deckMember = modelMember.At("deck");
  const auto& forcing = input.forcing;
  const auto hasLoad = !forcing.load.empty();
  const auto hasTide = !forcing.tide.empty();
  const auto& solver = input.solver;

  const auto deck = Deck::Read(input.model.deck, FormatOf(input.model.format));
  if (!deck) Fail(deckMember, deck.error().message);
  const auto model = EarthModel::FromDeck(*deck);
  if (!model) Fail(deckMember, model.error().message);
  auto mesh_options = RadialMesh::Options{};
  mesh_options.lmax = input.lmax;
  const auto mesh = RadialMesh::Create(*model, mesh_options);
  if (!mesh) Fail(deckMember, mesh.error().message);
  const auto material = Material::Create(*model, *mesh);
  if (!material) Fail(deckMember, material.error().message);
  const auto& scales = model->ModelScales();

  // The viscosity: a model file's, its radial reference the Maxwell body
  // and its log-perturbation the lateral part, or the deck's own column.
  auto viscosity = std::optional<io::ModelViscosity>{};
  if (!input.model.viscosity.empty()) {
    const auto& file = modelMember.At("viscosity").At("file");
    auto read = io::ModelViscosity::Read(
        input.model.viscosity, *model,
        {.reference = input.model.viscosity_file_radial
                          ? io::ViscosityReference::FileRadial
                          : io::ViscosityReference::GeometricMean});
    if (!read) Fail(file, read.error().message);
    viscosity.emplace(std::move(*read));
  }
  const auto rheology =
      viscosity ? Rheology::Maxwell(*material, viscosity->Viscosity())
                : Rheology::FromModel(*material, *model);
  if (!rheology) {
    if (viscosity) {
      Fail(modelMember.At("viscosity").At("file"), rheology.error().message);
    }
    Fail(deckMember, std::format("gives no viscosity, which a viscoelastic "
                                 "response (a `times` block) needs: {}",
                                 rheology.error().message));
  }

  // The lateral moduli, where given.
  auto moduli = std::optional<io::ModelModuli>{};
  if (!input.model.moduli.empty()) {
    auto read = io::ModelModuli::Read(
        input.model.moduli, *model,
        {.reference = input.model.moduli_file_radial
                          ? io::ModulusReference::FileRadial
                          : io::ModulusReference::LateralMean});
    if (!read) Fail(modelMember.At("moduli").At("file"), read.error().message);
    moduli.emplace(std::move(*read));
  }
  const auto lateralViscosity = viscosity && !viscosity->IsRadial();
  const auto lateralModuli = moduli && !moduli->IsRadial();
  const auto threeD = lateralViscosity || lateralModuli;
  // a radial run on several ranks is the first rank's alone
  if (!threeD && !ranks.First()) return "";

  // The forcing at each node on the grid (SI), its load made
  // non-dimensional under the model's scales, kg m^-2 to the mass and
  // length scales, and its tide m^2 s^-2 to the potential's. Every grid
  // of the run blocks its transforms for the configured cache.
  const auto cacheBytes = solver.cache_mb << 20;
  const SurfaceGrid grid(input.lmax, {.cache_bytes = cacheBytes});
  const TimedForcing timed(forcing, grid, top.At("forcing"));
  auto tideDegrees = std::set<int>{};
  for (const auto& term : forcing.tide) tideDegrees.insert(term.degree);
  const auto density = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto potential = scales.Factor(PotentialDimensions);
  auto forcingAt = [&](std::size_t n) {
    auto built = timed.At(times.nodes_kyr[n]);
    auto out = std::pair{NodeForcing{.load = std::move(built.loadCoefficients),
                                     .tide = std::move(built.tide)},
                         std::move(built.load)};
    for (auto& c : out.first.load) c /= density;
    for (auto& c : out.first.tide) c /= potential;
    return out;
  };
  // the load applied at each node, SI, for the surface file
  auto loads = std::vector<std::vector<double>>{};
  auto keep = [&](std::vector<double> load) {
    if (hasLoad && ranks.First()) loads.push_back(std::move(load));
  };

  // Each node's surface response, and the 3D step's iterations and
  // residual (the first node's: its unrelaxed solve's, under lateral
  // moduli).
  auto history = std::vector<NodeSurface>{};
  auto radial = std::vector<NodeSurface>{};
  auto iterations = std::vector<long long>(nodes, 0);
  auto residuals = std::vector<double>(nodes, 0.0);
  const auto toTime = KYR / scales.Time();
  if (!threeD) {
    history = StepRadial(
        *rheology, input.lmax, grid, scales, times, solver.scheme,
        [&](std::size_t n) {
          auto [nd, load] = forcingAt(n);
          keep(std::move(load));
          return std::move(nd);
        },
        hasLoad, hasTide);
  } else {
    // The 3D stepper, distributed over radial segments like the
    // evolutions: the rank cuts balance the transform work of the
    // elements the coupling carries, the viscous ones and, under lateral
    // moduli, those the moduli perturb.
    auto lateral = Viscoelastic3D::Options::LateralModuli{};
    if (lateralModuli) {
      lateral = {.shear = moduli->Shear(), .bulk = moduli->Bulk()};
    }
    auto probe = Viscoelastic3D::Options{};
    probe.transform_degrees = solver.transform_degrees;
    probe.cache_bytes = cacheBytes;
    probe.moduli = lateral;
    const SurfaceGrid coupled(input.lmax, {.nmax = 2,
                                           .oversampling = probe.oversampling,
                                           .parallel = probe.parallel,
                                           .matrix_kernel = probe.matrix_kernel,
                                           .cache_bytes = cacheBytes});
    const auto active = lateralModuli
                            ? Viscoelastic3D::ActiveElementsFor(
                                  *rheology, input.lmax, coupled, probe)
                            : std::vector<char>{};
    auto options = OperatorOptions(input.lmax, solver.transform_degrees,
                                   solver.cache_mb, *rheology, active);
    options.solver = solver.step;
    options.rtol = solver.step_rtol;
    options.max_iterations = solver.step_max_iterations;
    options.solve_rtol = solver.step_rtol;
    options.solve_max_iterations = solver.step_max_iterations;
    options.elastic_rtol = solver.step_rtol;
    options.elastic_max_iterations = solver.step_max_iterations;
    options.moduli = lateral;
    const auto delta = viscosity
                           ? viscosity->Delta()
                           : std::function<double(double, double, double)>{
                                 [](double, double, double) { return 0.0; }};
    const Viscoelastic3D op(*rheology, input.lmax, coupled, delta, options);

    // Every rank reads the forcing alike, but their transforms may round
    // differently: the first rank's coefficients serve them all.
    auto agree = [&](NodeForcing& nd) {
      if (!ranks.Distributed()) return;
      for (auto* coefficients : {&nd.load, &nd.tide}) {
        ranks.collectives.share(
            std::span(reinterpret_cast<double*>(coefficients->data()),
                      2 * coefficients->size()));
      }
    };
    auto channels = [&](NodeForcing nd) {
      agree(nd);
      return std::pair{hasLoad ? ColumnsOf(std::move(nd.load), 1.0)
                               : Viscoelastic3D::LoadCallable3D{},
                       hasTide ? ColumnsAt(std::move(nd.tide), tideDegrees)
                               : Viscoelastic3D::LoadCallable3D{}};
    };
    const auto length = scales.Length();
    auto record = [&](const Viscoelastic3D::Solutions& x) {
      // collective: every rank takes part, the first keeps them
      auto U = op.Surface(x, Component::U);
      auto V = op.Surface(x, Component::V);
      auto Phi = op.Surface(x, Component::Phi);
      if (!ranks.First()) return;
      history.push_back({.U = CoefficientsOf(grid, U, length),
                         .V = CoefficientsOf(grid, V, length),
                         .Phi = CoefficientsOf(grid, Phi, potential)});
    };

    // The first node: the unrelaxed solve under its forcing with no
    // memory (under lateral moduli the conjugate-gradient solve on the
    // reference systems), its strain kept for the first step.
    auto state = op.ZeroState();
    state.t = times.nodes_kyr.front() * toTime;
    auto [nd, load] = forcingAt(0);
    keep(std::move(load));
    auto now = channels(std::move(nd));
    {
      const auto x = op.SolveElastic(state, now.first, now.second);
      if (op.HasLateralModuli()) {
        const auto info = op.LastElasticSolve();
        if (!info.converged) {
          throw std::runtime_error(std::format(
              "the unrelaxed solve at the first node did not converge: "
              "residual {} after {} iterations; loosen solver.step_rtol or "
              "raise solver.step_max_iterations",
              info.residual, info.iterations));
        }
        iterations[0] = info.iterations;
        residuals[0] = info.residual;
      }
      state.strainCoefficients = op.StrainCoefficients(
          x,
          [&](int l) -> const DegreeOperator& {
            return op.UnrelaxedSpheroidal(l);
          },
          [&](int l) -> const ToroidalOperator& {
            return op.UnrelaxedToroidal(l);
          });
      op.PrimeStrain(state);
      record(x);
    }
    // Each step ends under the forcing at its end node.
    for (auto n = std::size_t{1}; n < nodes; ++n) {
      auto [ndNext, loadNext] = forcingAt(n);
      keep(std::move(loadNext));
      auto next = channels(std::move(ndNext));
      const auto dt = times.steps_kyr[n - 1] * toTime;
      const auto middle = state.t + 0.5 * dt;
      auto result = op.Step(state, dt, solver.scheme,
                            Across(now.first, next.first, middle),
                            Across(now.second, next.second, middle));
      const auto& info = result.info;
      if (!info.converged) {
        throw std::runtime_error(std::format(
            "the step to node {} ({:g} kyr) did not converge: residual {} "
            "after {} iterations; loosen solver.step_rtol or raise "
            "solver.step_max_iterations",
            n, times.nodes_kyr[n], info.residual, info.iterations));
      }
      iterations[n] = info.iterations;
      residuals[n] = info.residual;
      // the replaced state's tensors go to the next step
      op.Recycle(std::exchange(state, std::move(result.state)));
      record(result.solutions);
      now = std::move(next);
    }
    op.ReleaseScratch();
    if (!ranks.First()) return "";
    // The radial reference model under the same forcing: the deck's
    // radial moduli and the radial reference viscosity, by the radial
    // stepper.
    radial = StepRadial(
        *rheology, input.lmax, grid, scales, times, solver.scheme,
        [&](std::size_t n) { return forcingAt(n).first; }, hasLoad, hasTide);
  }

  // What the config was taken to mean, for the summary and the outputs.
  auto dimension = std::string{};
  if (threeD) {
    dimension = std::format(
        "3-D ({})",
        lateralViscosity && lateralModuli
            ? "lateral viscosity, lateral moduli"
            : (lateralViscosity ? "lateral viscosity" : "lateral moduli"));
  } else if (viscosity && moduli) {
    dimension = "1-D (the viscosity and moduli files are radial)";
  } else if (viscosity) {
    dimension = "1-D (the viscosity file is radial)";
  } else if (moduli) {
    dimension = "1-D (the moduli file is radial)";
  } else {
    dimension = "1-D";
  }
  const auto meaning = std::format(
      "{}, viscoelastic, {} nodes {:g} to {:g} kyr, {}", dimension, nodes,
      times.nodes_kyr.front(), times.nodes_kyr.back(), Summary(forcing));
  auto described = std::vector<std::string>{};
  for (const auto& term : forcing.load) described.push_back(Describe(term));
  for (const auto& term : forcing.tide) described.push_back(Describe(term));
  const auto schemeName = solver.scheme == Scheme::BackwardEuler
                              ? std::string{"backward_euler"}
                              : std::string{"exponential_trapezoid"};
  const auto conjugate =
      solver.step == Viscoelastic3D::Options::StepSolver::ConjugateGradients;
  const auto stepName = !threeD
                            ? std::string{"direct"}
                            : (conjugate ? std::string{"conjugate_gradients"}
                                         : std::string{"fixed_point"});

  // The surface file: a slice per node on the run's clock in seconds.
  auto file = pm::SurfaceFile{};
  file.grid = AngularGridOf(grid);
  file.source = std::format("sl3d response (sl3dpp {})", Version());
  file.title = threeD
                   ? "viscoelastic response of a laterally heterogeneous Earth"
                   : "viscoelastic response of a radial Earth";
  auto axis = std::vector<double>{};
  for (const auto t : times.nodes_kyr) axis.push_back(t * KYR);
  file.time = axis;
  file.meta.emplace("config", config.string());
  file.meta.emplace("model", input.model.deck.string());
  if (!input.model.moduli.empty()) {
    file.meta.emplace("moduli", input.model.moduli.string());
  }
  if (!input.model.viscosity.empty()) {
    file.meta.emplace("viscosity", input.model.viscosity.string());
  }
  file.meta.emplace("lateral", std::vector<long long>{threeD ? 1 : 0});
  AddForcingMeta(file, forcing, described);
  file.meta.emplace("time",
                    std::string{"seconds on the run's clock, a time of t kyr "
                                "in the times block at t * 3.15576e10 s; "
                                "dates_kyr lists the nodes"});
  file.meta.emplace("dates_kyr", times.nodes_kyr);
  file.meta.emplace("scheme", schemeName);
  file.meta.emplace("step_solver", stepName);
  file.meta.emplace("step_iterations", iterations);
  file.meta.emplace("step_residual", residuals);
  const auto lengthDims = pm::units::LENGTH;
  const auto potentialDims = pm::Dimensions{.length = 2, .time = -2};
  const auto fieldSize = grid.FieldSize();
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
  auto gather = [&](const std::vector<NodeSurface>& from,
                    std::vector<Complex> NodeSurface::*member) {
    auto values = std::vector<double>{};
    values.reserve(from.size() * fieldSize);
    for (const auto& node : from) {
      const auto field = grid.Synthesise(node.*member);
      values.insert(values.end(), field.begin(), field.end());
    }
    return values;
  };
  if (hasLoad) {
    auto values = std::vector<double>{};
    values.reserve(nodes * fieldSize);
    for (const auto& load : loads)
      values.insert(values.end(), load.begin(), load.end());
    put("load", std::move(values), pm::Dimensions{.mass = 1, .length = -2},
        "the surface mass load applied at each node");
  }
  put("displacement", gather(history, &NodeSurface::U), lengthDims,
      "radial surface displacement");
  put("tangential_potential", gather(history, &NodeSurface::V), lengthDims,
      "the tangential displacement's potential V (the displacement its "
      "surface gradient)");
  put("potential", gather(history, &NodeSurface::Phi), potentialDims,
      "gravitational potential perturbation at the surface");
  if (threeD) {
    put("displacement_radial_model", gather(radial, &NodeSurface::U),
        lengthDims,
        "radial surface displacement of the radial reference model (the "
        "deck's moduli, the radial reference viscosity) under the same "
        "forcing");
    put("potential_radial_model", gather(radial, &NodeSurface::Phi),
        potentialDims,
        "the radial reference model's potential perturbation under the same "
        "forcing");
  }
  CreateParent(input.output.surface);
  pm::WriteSurfaceFile(input.output.surface, file);
  auto wrote = std::format("wrote {}", input.output.surface.string());

  // The exchange tables, one per node chosen, `{node}` in the path the
  // node's index padded to the width of the last.
  if (!input.output.exchange.empty()) {
    auto chosen = std::set<int>(input.output.exchange_nodes.begin(),
                                input.output.exchange_nodes.end());
    if (input.output.exchange_nodes.empty()) {
      for (auto n = 0; n < static_cast<int>(nodes); ++n) chosen.insert(n);
    }
    const auto width = std::to_string(nodes - 1).size();
    const auto pattern = input.output.exchange.string();
    const auto heading =
        std::format("viscoelastic surface response from sl3dpp ({})",
                    threeD ? "laterally heterogeneous viscoelastic model"
                           : "radial viscoelastic model");
    const auto sources = std::format(
        "config {} model {} moduli {} viscosity {}", config.string(),
        input.model.deck.string(),
        input.model.moduli.empty() ? "none" : input.model.moduli.string(),
        input.model.viscosity.empty() ? "none"
                                      : input.model.viscosity.string());
    const auto scalars =
        SharedScalars(material->Radius() * scales.Length(),
                      material->SurfaceGravity() *
                          scales.Factor(Dimensions{.length = 1, .time = -2}),
                      input.lmax, threeD, forcing);
    for (const auto n : chosen) {
      auto path = pattern;
      const auto index = std::format("{:0{}}", n, width);
      for (auto at = path.find(kNodeField); at != std::string::npos;
           at = path.find(kNodeField, at + index.size())) {
        path.replace(at, kNodeField.size(), index);
      }
      const auto& node = history[static_cast<std::size_t>(n)];
      WriteExchangeTable(
          path, heading, sources, described,
          scalars + std::format("node {}\ntime_kyr {:.17g}\n", n,
                                times.nodes_kyr[static_cast<std::size_t>(n)]),
          input.lmax, node.U, node.Phi);
    }
    wrote += std::format(" and {} exchange table{}", chosen.size(),
                         chosen.size() == 1 ? "" : "s");
  }

  if (!threeD) return std::format("{}; {}", meaning, wrote);
  auto total = 0LL;
  for (auto n = std::size_t{1}; n < nodes; ++n) total += iterations[n];
  return std::format("{}; {} ({} {} iterations over {} steps)", meaning, wrote,
                     total, conjugate ? "conjugate-gradient" : "fixed-point",
                     nodes - 1);
}

}  // namespace

std::string RunResponse(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  const auto input = ReadResponseInput(config);
  return input.times ? RunViscoelastic(config, top, input)
                     : RunElastic(config, top, input);
}

}  // namespace sl3dpp::app
