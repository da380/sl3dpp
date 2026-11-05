/**
 * @file ExampleStatic3D.cpp
 * @brief The static response of a laterally heterogeneous elastic Earth:
 * a cap load over Hudson Bay on the synthetic 3D moduli of
 * synthetic_moduli.nc (MakeSyntheticModuli), solved exactly by the
 * stepper's conjugate gradients on a branchless rheology — the 3D static
 * problem of docs/adjoints.md §13j — against the radial model's
 * response to the same load.
 *
 * The lateral moduli come in through io::ModelModuli (the perturbations
 * of the file's mu and kappa about their lateral means), the solve runs
 * through Viscoelastic3D::SolveElastic with the conjugate-gradient
 * solver (distributed over radial segments under MPI, like the
 * evolutions), and the response inside the body goes out as a
 * planetmodel model file (io::WriteNodeFields) for planetmodel's
 * cutaways. Outputs: static_3d.txt (the surface fields and the
 * difference from the radial model, SI) and static_3d_volume.nc (the
 * radial displacement, the deviatoric strain's magnitude and the shear
 * perturbation at the memory nodes).
 *
 *   ExampleStatic3D [--lmax N] [--state FILE]
 *
 * Run from the build tree; plot with python/plot_static_3d.py.
 */
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>
#include <vector>

#include "Common.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/KernelWriter.hpp"
#include "sl3dpp/io/ModelModuli.hpp"

using namespace sl3dpp;

int main(int argc, char** argv) {
  const auto options = examples::ParseOptions(argc, argv, 32);
  const auto lmax = options.lmax;
  // Hudson Bay, a 2 km ice-equivalent cap 10 degrees wide
  constexpr auto siteLat = 60.0, siteLon = -85.0, widthDeg = 10.0;
  constexpr auto amplitude = 917.0 * 2000.0;  // kg m^-2 at the centre

  examples::Say(std::format("ExampleStatic3D: the 3D static response at lmax {}", lmax));
  const auto earth = examples::Earth::Make(lmax, 1e21);
  const auto elastic = Rheology::Create(earth.material, {}).value();
  const auto moduli = io::ModelModuli::Read("synthetic_moduli.nc", earth.model);
  if (!moduli) {
    std::cerr << "ExampleStatic3D: " << moduli.error().message << "\n";
    return 1;
  }

  // The load's coefficients, non-dimensional under the model's scales.
  const SurfaceGrid grid(lmax);
  const auto& scales = earth.model.ModelScales();
  constexpr auto deg = std::numbers::pi / 180.0;
  const auto loadSi = grid.FromFunction([&](double theta, double phi) {
    return amplitude * examples::Cap(theta, phi, (90.0 - siteLat) * deg,
                                     siteLon * deg, widthDeg * deg);
  });
  auto loadNd = grid.Expand(loadSi);
  for (auto& c : loadNd) c /= scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto load = [&loadNd](int l, double) -> std::optional<std::vector<std::complex<double>>> {
    auto columns = std::vector<std::complex<double>>(l + 1);
    for (auto m = 0; m <= l; ++m) columns[m] = loadNd[SurfaceGrid::Index(l, m)];
    return columns;
  };

  examples::Say("the radial model's response (the reference)");
  const ElasticPerturbation3D reference(earth.material, lmax);
  const auto x1 = reference.Solve(reference.LoadForces(loadNd), {});

  examples::Say("the 3D response by conjugate gradients on the reference systems");
  Viscoelastic3D::Options opOptions;
  opOptions.solver = Viscoelastic3D::Options::StepSolver::ConjugateGradients;
  opOptions.elastic_rtol = 1e-10;
  opOptions.moduli = {.shear = moduli->Shear(), .bulk = moduli->Bulk()};
  const Viscoelastic3D op(elastic, lmax, [](double, double, double) { return 0.0; },
                          opOptions);
  const auto x = op.SolveElastic(op.ZeroState(), load);
  const auto info = op.LastElasticSolve();
  examples::Say(std::format("  {} iterations, residual {:.2e}", info.iterations,
                            info.residual));

  // The surface fields in SI.
  const auto length = scales.Length();
  auto surface = [&](const std::map<int, std::vector<std::complex<double>>>& perDegree,
                     double factor) {
    auto coefficients = grid.ZeroCoefficients();
    for (const auto& [l, columns] : perDegree) {
      for (auto m = 0; m <= l; ++m) {
        coefficients[SurfaceGrid::Index(l, m)] = factor * columns[m];
      }
    }
    return grid.Synthesise(coefficients);
  };
  const auto U = surface(op.Surface(x, Component::U), length);
  const auto U1 = surface(reference.Surface(x1, Component::U), length);
  const auto Phi = surface(op.Surface(x, Component::Phi),
                           scales.Factor(PotentialDimensions));
  auto difference = std::vector<double>(U.size());
  auto worst = 0.0, scale = 0.0;
  for (auto i = std::size_t{0}; i < U.size(); ++i) {
    difference[i] = U[i] - U1[i];
    worst = std::max(worst, std::abs(difference[i]));
    scale = std::max(scale, std::abs(U1[i]));
  }
  examples::Say(std::format(
      "  the 3D displacement departs from the radial model's by at most {:.3f} m "
      "({:.1f}% of its largest value)",
      worst, 100.0 * worst / scale));

  examples::Writer out("static_3d.txt");
  out.Comment("the static response of the synthetic 3D elastic Earth to a cap "
              "load over Hudson Bay, against the radial model's (SI)");
  out.Scalar("site_lat", siteLat);
  out.Scalar("site_lon", siteLon);
  out.Scalar("cg_iterations", info.iterations);
  out.Scalar("cg_residual", info.residual);
  out.Grid(grid);
  out.Field("load_kg_m2", loadSi);
  out.Field("displacement_m", U);
  out.Field("displacement_radial_m", U1);
  out.Field("displacement_difference_m", difference);
  out.Field("potential_m2_s2", Phi);

  // The response inside the body, on the reference's whole coupling.
  const auto& coupling = reference.Coupling();
  auto fields = io::ResponseFieldsAtNodes(
      coupling, [&](int l) -> const DegreeOperator& { return reference.Spheroidal(l); },
      [&](int l) -> const ToroidalOperator& { return reference.Toroidal(l); }, x);
  const auto shear = io::SampleNodeField(coupling, moduli->Shear());
  const auto written = io::WriteNodeFields(
      "static_3d_volume.nc", coupling, earth.material.Mesh(), earth.model,
      {io::NodeField{.name = "radial_displacement",
                     .meaning = "radial displacement of the static response",
                     .dimensions = Dimensions{.length = 1},
                     .values = fields.radialDisplacement},
       io::NodeField{.name = "deviatoric_strain",
                     .meaning = "the deviatoric strain's magnitude sqrt(e:e)",
                     .values = fields.deviatoricStrain},
       io::NodeField{.name = "mu_relative",
                     .meaning = "the relative lateral perturbation of the shear "
                                "modulus, mu(x) / mu(r) - 1",
                     .values = shear}},
      {.name = "static response of the synthetic 3D elastic Earth"});
  if (!written) {
    std::cerr << "ExampleStatic3D: " << written.error().message << "\n";
    return 1;
  }
  std::cout << "wrote static_3d.txt and static_3d_volume.nc\n";
  return 0;
}
