// The elastic problem as the evolution asked for one solve: a time grid
// of one node. The tangent-linear runs, both adjoint sweeps and the
// second-order sweeps take it, in the forcing (a load and a point
// source) and in the moduli; and without an ocean their kernels in the
// moduli are the static problem's of ElasticPerturbation3D (its
// StaticKernels of the adjoint and forward static solutions), on a
// radial body and under lateral moduli.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "KernelScene.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using namespace kernel_scene;

namespace {

MaterialDirection Nodal(const Scene& setup,
                        const std::function<double(double)>& profile) {
  const auto& material = setup.material;
  const auto& mesh = material.Mesh();
  const auto [k0, m0] = material.Moduli().KappaMu();
  MaterialDirection d;
  d.kappa.assign(k0.size(), 0.0);
  d.mu.assign(m0.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e)) continue;
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      const auto p = 0.1 * profile(mesh.Radius(e, q));
      d.kappa[i] = 0.5 * p * k0[i];
      d.mu[i] = p * m0[i];
    }
  }
  return d;
}

}  // namespace

TEST(StaticDerivatives, OneNodeIsTheElasticProblemInEveryClass) {
  // One time node: the run is the instantaneous elastic response, the
  // tangent-linear run its derivative in the ice, the moduli and a
  // source's moment, both
  // sweeps' kernels pair with the direction to the tangent's objective
  // (the continuous sweep is its jump alone and exact), and the Hessian
  // is symmetric and equal to the second-order run; with an ocean and
  // each rotational feedback.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(103);
  const auto times = std::vector<double>{0.0};
  const auto load = setup.Smooth(rng, 100.0 * setup.Sigma());
  const auto nodalU = Nodal(
      setup, [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); });
  const auto nodalW = Nodal(setup, [](double r) { return 1.0 - 0.5 * r; });
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[0].displacementDual = setup.Smooth(rng, 0.5);
  measurements[0].potentialDual = setup.Smooth(rng, 0.7);
  measurements[0].tangentialPotentialDual = setup.Smooth(rng, 0.6);
  measurements[0].toroidalPotentialDual = setup.Smooth(rng, 0.8);
  measurements[0].angularVelocityDual = {0.2, -0.4, 0.8};
  // a point source beside the load, and its moment among the parameters
  const auto& sle = setup.solver.SolverScales();
  const auto surface = setup.material.Radius() *
                       setup.model.ModelScales().Length() / sle.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  const auto at = BodyPoint{.radius = surface - 400e3 / sle.Length(),
                            .colatitude = 1.1,
                            .longitude = 4.0};
  auto sourceOf = [&](double a, double b, double c) {
    return std::vector<PointSource>{
        {.at = at,
         .moment = {.rr = a * unit, .tt = b * unit, .pp = -(a + b) * unit,
                    .rt = c * unit, .rp = -a * unit, .tp = b * unit}}};
  };
  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Axial,
                        RotationalFeedback::Full}) {
    const auto label = "rotation " + std::to_string(static_cast<int>(rotation));
    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(op, setup.solver, setup.state,
                        {.rotation = rotation, .rtol = 1e-12, .record_stride = 1});
    Evolution::Forcing forcing;
    forcing.directLoad = [&](double) { return load; };
    forcing.sources = [&](double) { return sourceOf(0.9, -0.2, 0.3); };
    const auto run = evolution.Run(times, forcing);
    ASSERT_EQ(run.size(), 1u);
    const auto trajectory = ForwardTrajectory::FromSnapshots(run);
    TangentPerturbation u, w;
    u.iceThickness = {setup.Smooth(rng, 20.0 * setup.Metre())};
    u.material = AtMemoryPoints(op.Points(), nodalU);
    u.sources = {sourceOf(-0.4, 0.8, 0.5)};
    w.sources = {sourceOf(0.2, 0.3, -0.6)};
    w.iceThickness = {setup.Smooth(rng, 20.0 * setup.Metre())};
    w.material = AtMemoryPoints(op.Points(), nodalW);
    const auto& mesh = setup.material.Mesh();
    auto pair = [&](const auto& kernels, const TangentPerturbation& d) {
      return kernels.momentKernels[0][0].Contract(d.sources[0][0].moment) +
             setup.Surface(kernels.thickness[0], d.iceThickness[0]) +
             VolumePairing(op.Points(), mesh, kernels.kappaKernel, d.material.kappa) +
             VolumePairing(op.Points(), mesh, kernels.muKernel, d.material.mu);
    };
    auto exact = 0.0;
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver tangent(evolution,
                                  {.method = method, .rtol = 1e-12, .record = true});
      const auto runU = tangent.Run(times, u, &trajectory);
      const auto runW = tangent.Run(times, w, &trajectory);
      const auto along = Objective(setup, runU, measurements);
      ASSERT_NE(along, 0.0) << label;
      if (method == AdjointMethod::Discrete) exact = along;
      EXPECT_NEAR(along, exact, 1e-10 * std::abs(exact)) << label;
      AdjointSolver adjoint(evolution, {.method = method,
                                        .rtol = 1e-12,
                                        .material_kernels = true,
                                        .moment_points = {at}});
      const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
      EXPECT_NEAR(pair(first, u), along,
                  1e-8 * std::abs(along))
          << label << " method " << static_cast<int>(method);
      const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
          times, measurements, runU, {}, trajectory, {}, u.material);
      const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
          times, measurements, runW, {}, trajectory, {}, w.material);
      const auto uw = pair(alongU, w);
      const auto wu = pair(alongW, u);
      SecondOrderDirection direction;
      direction.materialU = u.material;
      direction.materialV = w.material;
      const auto forward = Objective(
          setup, tangent.RunSecondOrder(times, runU, runW, direction, &trajectory),
          measurements);
      ASSERT_NE(forward, 0.0) << label;
      EXPECT_NEAR(uw, wu, 1e-8 * std::abs(uw))
          << label << " method " << static_cast<int>(method);
      EXPECT_NEAR(uw, forward, 1e-8 * std::abs(forward))
          << label << " method " << static_cast<int>(method);
    }
  }
}

TEST(StaticDerivatives, AnElasticBodyWithNoRelaxingLayer) {
  // The same on a body with nothing to relax, two ways: a Maxwell
  // rheology with no viscous layer (one branch, elastic everywhere) and a
  // rheology with no branch at all (the static problem's): the classes
  // run, the kernels in the moduli pair with the tangent and the Hessian
  // is symmetric and the second-order run's.
  auto setup = Scene::Make(1e21);
  const auto noViscousLayer =
      Rheology::Maxwell(setup.material, std::map<std::size_t, RadialValue>{}).value();
  const auto noBranch = Rheology::Create(setup.material, {}).value();
  ASSERT_EQ(noBranch.BranchCount(), 0u);
  for (const auto* elastic : {&noViscousLayer, &noBranch}) {
    const auto label = std::string(elastic == &noBranch ? "no branch" : "no viscous layer");
    std::mt19937 rng(109);
    const auto times = std::vector<double>{0.0};
    const auto load = setup.Smooth(rng, 100.0 * setup.Sigma());
    const auto nodalU = Nodal(
        setup, [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); });
    const auto nodalW = Nodal(setup, [](double r) { return 1.0 - 0.5 * r; });
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[0].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
    measurements[0].displacementDual = setup.Smooth(rng, 0.5);
    ViscoelasticOperator op(*elastic, kLmax);
    Evolution evolution(op, setup.solver, setup.state,
                        {.rtol = 1e-12, .record_stride = 1});
    const auto run = evolution.Run(times, [&](double) { return load; });
    const auto trajectory = ForwardTrajectory::FromSnapshots(run);
    TangentPerturbation u, w;
    u.iceThickness = {setup.Smooth(rng, 20.0 * setup.Metre())};
    u.material = AtMemoryPoints(op.Points(), nodalU);
    w.iceThickness = {setup.Smooth(rng, 20.0 * setup.Metre())};
    w.material = AtMemoryPoints(op.Points(), nodalW);
    const auto& mesh = setup.material.Mesh();
    auto pair = [&](const auto& kernels, const TangentPerturbation& d) {
      return setup.Surface(kernels.thickness[0], d.iceThickness[0]) +
             VolumePairing(op.Points(), mesh, kernels.kappaKernel, d.material.kappa) +
             VolumePairing(op.Points(), mesh, kernels.muKernel, d.material.mu);
    };
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver tangent(evolution,
                                  {.method = method, .rtol = 1e-12, .record = true});
      const auto runU = tangent.Run(times, u, &trajectory);
      const auto runW = tangent.Run(times, w, &trajectory);
      const auto along = Objective(setup, runU, measurements);
      ASSERT_NE(along, 0.0) << label;
      AdjointSolver adjoint(evolution,
                            {.method = method, .rtol = 1e-12, .material_kernels = true});
      const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
      EXPECT_NEAR(pair(first, u), along, 1e-8 * std::abs(along)) << label;
      const auto alongU = adjoint
                              .SecondOrderKernels(times, measurements, runU, {},
                                                  trajectory, {}, u.material)
                              .second;
      const auto alongW = adjoint
                              .SecondOrderKernels(times, measurements, runW, {},
                                                  trajectory, {}, w.material)
                              .second;
      SecondOrderDirection direction;
      direction.materialU = u.material;
      direction.materialV = w.material;
      const auto forward = Objective(
          setup, tangent.RunSecondOrder(times, runU, runW, direction, &trajectory),
          measurements);
      EXPECT_NEAR(pair(alongU, w), pair(alongW, u), 1e-8 * std::abs(forward)) << label;
      EXPECT_NEAR(pair(alongU, w), forward, 1e-8 * std::abs(forward)) << label;
    }
  }
}

// -- the static kernels by one route --------------------------------------

namespace {

using Complex = std::complex<double>;

double Gap(double got, double want) { return std::abs(got - want) / std::abs(want); }

/// The unit ratios between the material's scales and the sea level
/// solver's (the evolution's toNd / toMat factors).
struct Ratios {
  double toNdLength = 1.0, toNdPotential = 1.0, toMatDensity = 1.0, toMatPotential = 1.0;

  static Ratios Of(const Scales& material, const Scales& sle) {
    auto ratio = [&](Dimensions d) { return material.Factor(d) / sle.Factor(d); };
    return {.toNdLength = ratio(Dimensions{.length = 1}),
            .toNdPotential = ratio(PotentialDimensions),
            .toMatDensity = 1.0 / ratio(Dimensions{.mass = 1, .length = -2}),
            .toMatPotential = 1.0 / ratio(PotentialDimensions)};
  }
};

/// The Hermitian pairing of m >= 0 coefficient sets (m > 0 twice), over
/// the degrees both carry.
double CoefficientPairing(const std::map<int, std::vector<Complex>>& x,
                          std::span<const Complex> dual) {
  auto out = 0.0;
  for (const auto& [l, values] : x) {
    for (auto m = 0; m <= l; ++m) {
      out += (m == 0 ? 1.0 : 2.0) *
             (values[m] * std::conj(dual[SurfaceGrid::Index(l, m)])).real();
    }
  }
  return out;
}

std::vector<Complex> Scaled(std::vector<Complex> c, double factor) {
  for (auto& v : c) v *= factor;
  return c;
}

/// The surface measurement a2 int (gU U + gV V + gPhi phi) dOmega of the
/// evolution, as the static problem's surface duals: each field's
/// coefficients scaled by the solver's radius squared, the coefficient
/// pairing's normalisation against the grid's integral, and the unit
/// ratio of its channel.
struct StaticMeasurement {
  std::vector<Complex> u, v, phi;

  static StaticMeasurement Of(const GRDSolver& solver, const Ratios& ratios,
                              const Measurement& m) {
    const auto& grid = solver.Grid();
    const auto a = solver.Radius();
    // the pairing's normalisation: the grid's integral of the square of a
    // band-limited field over its coefficients' Hermitian pairing
    auto coefficients = grid.ZeroCoefficients();
    for (auto l = 0; l <= grid.LMax(); ++l) {
      for (auto mm = 0; mm <= l; ++mm) {
        coefficients[SurfaceGrid::Index(l, mm)] = {1.0 / (1.0 + l + mm),
                                                   mm == 0 ? 0.0 : 0.5 / (1.0 + l)};
      }
    }
    const auto probe = grid.Synthesise(coefficients);
    auto square = probe;
    for (auto& x : square) x *= x;
    const auto c = grid.Expand(probe);
    auto pairing = 0.0;
    for (auto l = 0; l <= grid.LMax(); ++l) {
      for (auto mm = 0; mm <= l; ++mm) {
        pairing += (mm == 0 ? 1.0 : 2.0) * std::norm(c[SurfaceGrid::Index(l, mm)]);
      }
    }
    const auto norm = grid.Integrate(square) / pairing;
    const auto scale = a * a * norm;
    return {Scaled(grid.Expand(m.displacementDual), scale * ratios.toNdLength),
            Scaled(grid.Expand(m.tangentialPotentialDual), scale * ratios.toNdLength),
            Scaled(grid.Expand(m.potentialDual), scale * ratios.toNdPotential)};
  }

  [[nodiscard]] ElasticPerturbation3D::Forces Forces(const ElasticPerturbation3D& born) const {
    auto out = born.SurfaceDualForces(u, Component::U);
    for (const auto& [dual, component] :
         {std::pair{&v, Component::V}, std::pair{&phi, Component::Phi}}) {
      for (const auto& [l, f] : born.SurfaceDualForces(*dual, component)) {
        auto& to = out.at(l);
        for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += f[i];
      }
    }
    return out;
  }

  [[nodiscard]] double Read(const ElasticPerturbation3D& born,
                            const ElasticPerturbation3D::Solutions& x) const {
    return CoefficientPairing(born.Surface(x, Component::U), u) +
           CoefficientPairing(born.Surface(x, Component::V), v) +
           CoefficientPairing(born.Surface(x, Component::Phi), phi);
  }
};

/// The largest departure of a kernel from a reference over a set of
/// indices, relative to the reference's largest value there.
double Departure(std::span<const double> got, std::span<const double> want) {
  auto top = 0.0, worst = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    top = std::max(top, std::abs(want[i]));
    worst = std::max(worst, std::abs(got[i] - want[i]));
  }
  return worst / top;
}

}  // namespace

TEST(StaticDerivatives, OneNodeSweepsReproduceTheStaticKernels) {
  // The static problem without an ocean and without rotational feedback,
  // by two routes on the radial body: the one-node run of the evolution
  // on a branchless rheology (its memory points at the GLL nodes), its
  // kernels in kappa and mu by both sweeps; and ElasticPerturbation3D's
  // static adjoint and forward solutions with StaticKernels, the
  // measurement's kernels as minus those. Under a load, and under a body
  // tide, measured in U, V and phi at the surface: the two routes'
  // measurements agree, and at every node of the static coupling (the
  // solid above the element holding the centre) the static kernels'
  // means over the sphere — the kernels of a radial perturbation — are
  // the one-node sweeps' kernels, both methods.
  auto setup = Scene::Make(1e21);
  const auto elastic = Rheology::Create(setup.material, {}).value();
  ViscoelasticOperator op(elastic, kLmax, {.memory_points = MemoryPointSet::Gll});
  Evolution evolution(op, setup.solver, setup.solver.NoOceanState(),
                      {.rotation = RotationalFeedback::None, .rtol = 1e-12, .record_stride = 1});
  const ElasticPerturbation3D born(setup.material, kLmax);
  const auto& coupling = born.Coupling();
  const auto& grid = setup.solver.Grid();
  const auto ratios =
      Ratios::Of(setup.material.ModelScales(), setup.solver.SolverScales());
  const auto ngll = static_cast<std::size_t>(setup.material.Mesh().NGll());
  std::mt19937 rng(139);
  const auto load = setup.Smooth(rng, 100.0 * setup.Sigma());
  auto tide = grid.Expand(setup.Smooth(
      rng, 1.0 / setup.solver.SolverScales().Factor(PotentialDimensions)));
  tide[SurfaceGrid::Index(0, 0)] = 0.0;
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].displacementDual = setup.Smooth(rng, 0.5);
  measurements[0].tangentialPotentialDual = setup.Smooth(rng, 0.6);
  measurements[0].potentialDual = setup.Smooth(rng, 0.7);
  const auto measured = StaticMeasurement::Of(setup.solver, ratios, measurements[0]);
  const auto y = born.Solve(measured.Forces(born), {});
  const auto times = std::vector<double>{0.0};

  for (const bool tidal : {false, true}) {
    const auto label = std::string(tidal ? "tide" : "load");
    auto forcing = Evolution::Forcing{};
    if (tidal) {
      forcing.tide = [&](double) { return tide; };
    } else {
      forcing.directLoad = [&](double) { return load; };
    }
    const auto run = evolution.Run(times, forcing);
    const auto trajectory = ForwardTrajectory::FromSnapshots(run);
    const auto x =
        born.Solve(tidal ? born.TideForces(Scaled(tide, ratios.toMatPotential))
                         : born.LoadForces(Scaled(grid.Expand(load), ratios.toMatDensity)),
                   {});
    const auto viaEvolution = Objective(setup, run, measurements);
    const auto viaStatic = measured.Read(born, x);
    std::cout << std::format("[static, radial, {}] measurement gap {:.2e}\n", label,
                             Gap(viaStatic, viaEvolution));
    ASSERT_NE(viaEvolution, 0.0) << label;
    EXPECT_NEAR(viaStatic, viaEvolution, 1e-8 * std::abs(viaEvolution)) << label;

    const auto staticKernels = born.StaticKernels(y, x);
    auto kappaMean = SphericalMean(coupling, staticKernels.kappa);
    auto muMean = SphericalMean(coupling, staticKernels.mu);
    for (auto& v : kappaMean) v = -v;
    for (auto& v : muMean) v = -v;
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto kernels =
          AdjointSolver(evolution, {.method = method, .rtol = 1e-12, .material_kernels = true})
              .FirstOrderKernels(times, measurements, &trajectory);
      auto kappa = std::vector<double>{}, mu = std::vector<double>{};
      for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
        const auto full =
            static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll + coupling.NodeInElementOf(rn);
        kappa.push_back(kernels.kappaKernel[full]);
        mu.push_back(kernels.muKernel[full]);
      }
      std::cout << std::format("[static, radial, {}, method {}] kappa {:.2e}, mu {:.2e}\n",
                               label, static_cast<int>(method), Departure(kappa, kappaMean),
                               Departure(mu, muMean));
      EXPECT_LT(Departure(kappa, kappaMean), 1e-7) << label << ", method " << static_cast<int>(method);
      EXPECT_LT(Departure(mu, muMean), 1e-7) << label << ", method " << static_cast<int>(method);
    }
  }
}

TEST(StaticDerivatives, OneNodeSweepsReproduceTheStaticKernelsUnderLateralModuli) {
  // The same in 3D at degree 6: the one-node run of the 3D evolution on a
  // branchless rheology with the memory nodes at the GLL points, its
  // lateral moduli a 10% shear and bulk pattern varying with angle over
  // the whole solid, no ocean and no rotational feedback; against
  // ElasticPerturbation3D's full lateral solves (the forward solution and
  // the measurement's adjoint solution under the same moduli) and their
  // StaticKernels. Under a load and under a body tide, measured in U, V
  // and phi: the two routes' measurements agree, and the one-node sweeps'
  // kernels in kappa(x) and mu(x) are the static kernels at every grid
  // value of the coupling (the two couplings hold the same nodes and
  // grids), both methods.
  constexpr auto kLmax3 = 6;
  auto deck =
      Deck::Read(std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
                 DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  const auto model = EarthModel::FromDeck(deck).value();
  const auto mesh = RadialMesh::Create(model, {.lmax = kLmax3}).value();
  const auto material = Material::Create(model, mesh).value();
  const auto elastic = Rheology::Create(material, {}).value();
  const ElasticPerturbation3D born(material, kLmax3);
  const auto zero = [](double, double, double) { return 0.0; };
  {
    auto options = Viscoelastic3D::Options{.memory_points = MemoryPointSet::Gll};
    options.moduli = {
        .shear =
            [](double r, double theta, double phi) {
              return 0.1 * (0.5 + r * r) *
                     (std::sin(theta) * std::cos(theta) * std::cos(phi) +
                      0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
            },
        .bulk = [](double r, double theta, double) {
          return 0.1 * (1.0 - 0.5 * r) * std::cos(theta);
        }};
    const Viscoelastic3D op(elastic, kLmax3, zero, options);
    const auto& coupling = op.Coupling();
    ASSERT_EQ(coupling.NodeCount(), born.Coupling().NodeCount());
    ASSERT_EQ(coupling.GridValueCount(), born.Coupling().GridValueCount());
    const GRDSolver solver(SurfaceGrid(kLmax3), op.ElasticLove(), SeaLevelParameters{});
    const auto& grid = solver.Grid();
    const auto ratios = Ratios::Of(material.ModelScales(), solver.SolverScales());
    Evolution3D evolution(op, solver, solver.NoOceanState(),
                          {.rotation = RotationalFeedback::None,
                           .rtol = 1e-12,
                           .coupled_rtol = 1e-12,
                           .record_stride = 1});
    std::mt19937 rng(149);
    auto smooth = [&](double amplitude) {
      std::normal_distribution<double> normal;
      auto coeffs = grid.ZeroCoefficients();
      for (auto l = 0; l <= kLmax3; ++l) {
        for (auto m = 0; m <= l; ++m) {
          coeffs[SurfaceGrid::Index(l, m)] = {normal(rng), m == 0 ? 0.0 : normal(rng)};
        }
      }
      auto field = grid.Synthesise(coeffs);
      for (auto& v : field) v *= amplitude;
      return field;
    };
    const auto& sle = solver.SolverScales();
    const auto load = smooth(100.0 / sle.Factor(Dimensions{.mass = 1, .length = -2}));
    auto tide = grid.Expand(smooth(1.0 / sle.Factor(PotentialDimensions)));
    tide[SurfaceGrid::Index(0, 0)] = 0.0;
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[0].displacementDual = smooth(0.5);
    measurements[0].tangentialPotentialDual = smooth(0.6);
    measurements[0].potentialDual = smooth(0.7);
    const auto measured = StaticMeasurement::Of(solver, ratios, measurements[0]);
    const auto dMu = op.UnrelaxedShearPerturbation();
    const auto dKappa = op.BulkPerturbation();
    const auto solveOptions = ElasticPerturbation3D::SolveOptions{.rtol = 1e-13};
    const auto [y, infoY] = born.SolveLateral(measured.Forces(born), {}, dMu, dKappa, solveOptions);
    ASSERT_TRUE(infoY.converged);
    const auto times = std::vector<double>{0.0};
    for (const bool tidal : {false, true}) {
      const auto label = std::string(tidal ? "tide" : "load");
      auto forcing = Evolution3D::Forcing{};
      if (tidal) {
        forcing.tide = [&](double) { return tide; };
      } else {
        forcing.directLoad = [&](double) { return load; };
      }
      const auto run = evolution.Run(times, forcing);
      ASSERT_TRUE(run[0].converged) << label;
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(run);
      const auto [x, infoX] = born.SolveLateral(
          tidal ? born.TideForces(Scaled(tide, ratios.toMatPotential))
                : born.LoadForces(Scaled(grid.Expand(load), ratios.toMatDensity)),
          {}, dMu, dKappa, solveOptions);
      ASSERT_TRUE(infoX.converged) << label;
      // the measurement on the snapshot, as the radial test's Objective
      const auto& snap = run[0];
      const auto a = solver.Radius();
      auto surface = [&](std::span<const double> f, std::span<const double> g) {
        auto fg = std::vector<double>(f.size());
        for (auto i = std::size_t{0}; i < f.size(); ++i) fg[i] = f[i] * g[i];
        return a * a * grid.Integrate(fg);
      };
      const auto viaEvolution =
          surface(measurements[0].displacementDual, snap.displacement) +
          surface(measurements[0].tangentialPotentialDual, snap.tangentialPotential) +
          surface(measurements[0].potentialDual, snap.potential);
      const auto viaStatic = measured.Read(born, x);
      std::cout << std::format("[static, 3D, {}] measurement gap {:.2e}\n", label,
                               Gap(viaStatic, viaEvolution));
      ASSERT_NE(viaEvolution, 0.0) << label;
      EXPECT_NEAR(viaStatic, viaEvolution, 1e-8 * std::abs(viaEvolution)) << label;

      auto staticKernels = born.StaticKernels(y, x);
      for (auto& v : staticKernels.kappa) v = -v;
      for (auto& v : staticKernels.mu) v = -v;
      for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
        const auto kernels = AdjointSolver3D(evolution, {.method = method,
                                                         .rtol = 1e-12,
                                                         .coupled_rtol = 1e-12,
                                                         .material_kernels = true})
                                 .FirstOrderKernels(times, measurements, &trajectory);
        const auto kappa = Departure(kernels.kappaKernel3D, staticKernels.kappa);
        const auto mu = Departure(kernels.muKernel3D, staticKernels.mu);
        std::cout << std::format("[static, 3D, {}, method {}] kappa {:.2e}, mu {:.2e}\n",
                                 label, static_cast<int>(method), kappa, mu);
        EXPECT_LT(kappa, 1e-7) << label << ", method " << static_cast<int>(method);
        EXPECT_LT(mu, 1e-7) << label << ", method " << static_cast<int>(method);
      }
    }
  }
}
