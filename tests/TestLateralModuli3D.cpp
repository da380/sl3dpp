// Lateral elastic moduli in the 3D stepper: an angle-uniform shift of
// the unrelaxed shear and bulk moduli reproduces the 1D stepper on the
// shifted rheology (exactly at the GLL memory points; at Gauss points
// for a per-element constant shift of the Maxwell region, where every
// shear term is a point term); the unrelaxed start solve under lateral
// moduli is the static lateral solve; lateral moduli couple spheroidal
// and toroidal motion; the step self-converges at second order with
// lateral moduli and lateral viscosity on together.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <filesystem>
#include <iostream>
#include <map>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using Complex = std::complex<double>;

namespace {

constexpr auto kLmax = 6;

double Noise(std::size_t k) {
  const auto x = std::sin(12.9898 * static_cast<double>(k + 1)) * 43758.5453;
  return x - std::floor(x) - 0.5;
}

std::vector<Complex> LoadColumns(int l) {
  auto columns = std::vector<Complex>(l + 1);
  for (auto m = 0; m <= l; ++m) {
    columns[m] = m == 0 ? Complex(Noise(97 * l), 0.0)
                        : Complex(Noise(97 * l + m), Noise(55 * l + m));
  }
  return columns;
}

std::vector<double> Load1D(int l) {
  const auto columns = LoadColumns(l);
  auto real = std::vector<double>(2 * columns.size());
  for (auto c = std::size_t{0}; c < columns.size(); ++c) {
    real[c] = columns[c].real();
    real[columns.size() + c] = columns[c].imag();
  }
  return real;
}

/// PREM's Voigt average (isotropic, so that a shift of the isotropic
/// moduli is the whole of the shifted material) with a Maxwell mantle.
struct Body {
  EarthModel model;
  Material material;
  Rheology rheology;
  double inner = 0.0, outer = 0.0;  // the viscous range, dimensionless

  static Body Make() {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
    auto anisotropic = Material::Create(model, mesh).value();
    const auto [k0, m0] = anisotropic.Moduli().KappaMu();
    auto material = anisotropic.WithModuli(NodalModuli::Isotropic(k0, m0)).value();
    const auto& scales = model.ModelScales();
    auto viscosity = std::map<std::size_t, RadialValue>{};
    const auto b = model.DimensionlessView().Boundaries();
    Body body{std::move(model), std::move(material), Rheology::Maxwell(material, 1.0).value()};
    body.inner = 3.4799e6 / scales.Length();
    body.outer = 6.1511e6 / scales.Length();
    for (auto i = std::size_t{0}; i < body.model.LayerCount(); ++i) {
      if (!body.model.IsFluid(i) && b[i] >= body.inner && b[i + 1] <= body.outer) {
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
      }
    }
    body.rheology = Rheology::Maxwell(body.material, viscosity).value();
    return body;
  }

  /// The rheology with the moduli shifted nodally over the SOLID
  /// elements (the coupling holds no fluid nodes; the element holding
  /// the centre is left to the caller's functions): mu_U (1 + shear(r)),
  /// kappa (1 + bulk(r)), the branch moduli scaled with mu_U, the
  /// relaxation times held; evaluated at each node's radius, or at the
  /// element's midpoint (a per-element constant, which the Gauss points'
  /// interpolation reproduces exactly).
  [[nodiscard]] Rheology Shifted(const std::function<double(double)>& shear,
                                 const std::function<double(double)>& bulk,
                                 bool perElement = false) const {
    const auto& mesh = material.Mesh();
    const auto n = mesh.NGll();
    auto [kappa, mu] = material.Moduli().KappaMu();
    auto branches = std::vector<Rheology::Branch>{};
    for (auto k = std::size_t{0}; k < rheology.BranchCount(); ++k) {
      branches.push_back(rheology.BranchAt(k));
    }
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      if (material.IsFluid(e)) continue;
      for (auto q = 0; q < n; ++q) {
        const auto i = static_cast<std::size_t>(e) * n + q;
        const auto r = perElement ? 0.5 * (mesh.Left(e) + mesh.Right(e))
                                  : mesh.Radius(e, q);
        const auto s = 1.0 + shear(r), b = 1.0 + bulk(r);
        mu[i] *= s;
        kappa[i] *= b;
        for (auto& branch : branches) branch.mu[i] *= s;
      }
    }
    const auto shifted = material.WithModuli(NodalModuli::Isotropic(kappa, mu)).value();
    return Rheology::Create(shifted, std::move(branches)).value();
  }
};

/// The rheology with the density shifted nodally over the solid elements
/// above the centre, gravity, stratification and moduli held (the
/// lateral density's convention in 1D).
[[nodiscard]] Rheology ShiftedDensity(const Body& body,
                                      const std::function<double(double)>& f) {
  const auto& mesh = body.material.Mesh();
  const auto n = mesh.NGll();
  auto rho = std::vector<double>(body.material.Rho().begin(), body.material.Rho().end());
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (body.material.IsFluid(e) || (e == 0 && mesh.Left(0) == 0.0)) continue;
    for (auto q = 0; q < n; ++q) {
      const auto i = static_cast<std::size_t>(e) * n + q;
      rho[i] *= 1.0 + f(mesh.Radius(e, q));
    }
  }
  auto branches = std::vector<Rheology::Branch>{};
  for (auto k = std::size_t{0}; k < body.rheology.BranchCount(); ++k) {
    branches.push_back(body.rheology.BranchAt(k));
  }
  return Rheology::Create(body.material.WithDensityGravityHeld(std::move(rho)).value(),
                          std::move(branches))
      .value();
}

/// The steps from rest, both paths, and the worst relative difference of
/// the surface U, V and phi per (l, m) at the end.
double CompareSteps(const ViscoelasticOperator& op1, const Viscoelastic3D& op3,
                    Scheme scheme, double dt, int steps, int& iterations) {
  const auto load3 = [](int l, double) { return LoadColumns(l); };
  const auto load1 = [](int l, double) { return Load1D(l); };
  auto s1 = op1.ZeroState([](int l) { return 2 * (l + 1); });
  auto s3 = op3.ZeroState();
  auto x1 = std::map<int, std::vector<double>>{};
  Viscoelastic3D::Solutions x3;
  iterations = 0;
  for (auto step = 0; step < steps; ++step) {
    auto [n1, sols1] = op1.Step(s1, dt, scheme, load1);
    s1 = std::move(n1);
    x1 = std::move(sols1);
    auto result = op3.Step(s3, dt, scheme, load3);
    EXPECT_TRUE(result.info.converged);
    iterations = std::max(iterations, result.info.iterations);
    s3 = std::move(result.state);
    x3 = std::move(result.solutions);
  }
  auto worst = 0.0;
  for (const auto component : {Component::U, Component::V, Component::Phi}) {
    const auto u1 = op1.Surface(x1, component);
    const auto u3 = op3.Surface(x3, component);
    auto scale = 0.0, diff = 0.0;
    for (auto l = 0; l <= kLmax; ++l) {
      const auto columns = l + 1;
      auto dl = 0.0;
      for (auto c = 0; c < columns; ++c) {
        const Complex ref(u1.at(l)[c], u1.at(l)[columns + c]);
        scale = std::max(scale, std::abs(ref));
        dl = std::max(dl, std::abs(u3.at(l)[c] - ref));
      }
      diff = std::max(diff, dl);
      if (std::getenv("SL3DPP_VERBOSE")) {
        std::cout << "  component " << static_cast<int>(component) << " degree " << l
                  << " diff " << dl << '\n';
      }
    }
    EXPECT_GT(scale, 0.0);
    worst = std::max(worst, diff / scale);
  }
  auto toroidal = 0.0;
  for (const auto& [l, w] : x3.toroidal) {
    for (auto v : w) toroidal = std::max(toroidal, std::abs(v));
  }
  EXPECT_LT(toroidal, 1e-12);
  return worst;
}

}  // namespace

TEST(LateralModuli3D, AngleUniformShiftMatchesThe1DStepperAtGllPoints) {
  // Smooth 10% shifts of the unrelaxed shear and the bulk modulus over
  // the whole solid, at the GLL memory points: exactly the 1D stepper on
  // the shifted rheology, both schemes, two steps from rest.
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  // above the element holding the centre, which the coupling leaves out
  const auto r0 = body.material.Mesh().Right(0);
  const auto shear = [=](double r) {
    return r > r0 ? 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r)) : 0.0;
  };
  const auto bulk = [=](double r) { return r > r0 ? 0.1 * (1.0 - 0.5 * r) : 0.0; };
  ViscoelasticOperator op1(body.Shifted(shear, bulk), kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  Viscoelastic3D op3(
      body.rheology, kLmax, [](double, double, double) { return 0.0; },
      {.memory_points = MemoryPointSet::Gll,
       .moduli = {.shear = [&](double r, double, double) { return shear(r); },
                  .bulk = [&](double r, double, double) { return bulk(r); }}});
  EXPECT_TRUE(op3.HasLateralModuli());
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto iterations = 0;
    const auto worst = CompareSteps(op1, op3, scheme, 0.7 * tau, 2, iterations);
    EXPECT_LT(worst, 1e-10) << "scheme " << static_cast<int>(scheme);
    std::cout << "GLL scheme " << static_cast<int>(scheme) << ": worst "
              << worst << ", passes " << iterations << ", start solve "
              << op3.LastElasticSolve().iterations << " iterations\n";
  }
}

TEST(LateralModuli3D, ConstantShiftOfTheMaxwellRegionMatchesThe1DStepperAtGaussPoints) {
  // At Gauss memory points the shear of the Maxwell region is wholly a
  // point term (the long-term modulus vanishes there), so a shift
  // constant over each viscous element is exactly the 1D stepper's
  // point shear on the shifted rheology; the elastic regions and the
  // bulk modulus are left alone (their GLL quadrature differs).
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto inner = body.inner, outer = body.outer;
  const auto shear = [=](double r) { return r > inner && r < outer ? 0.1 : 0.0; };
  const auto none = [](double) { return 0.0; };
  ViscoelasticOperator op1(body.Shifted(shear, none, true), kLmax,
                           {.memory_points = MemoryPointSet::Gauss});
  Viscoelastic3D op3(
      body.rheology, kLmax, [](double, double, double) { return 0.0; },
      {.memory_points = MemoryPointSet::Gauss,
       .moduli = {.shear = [&](double r, double, double) { return shear(r); }}});
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto iterations = 0;
    const auto worst = CompareSteps(op1, op3, scheme, 0.7 * tau, 2, iterations);
    EXPECT_LT(worst, 1e-10) << "scheme " << static_cast<int>(scheme);
    std::cout << "Gauss scheme " << static_cast<int>(scheme) << ": worst "
              << worst << ", passes " << iterations << '\n';
  }
}

TEST(LateralModuli3D, StartSolveIsTheStaticLateralSolve) {
  // The unrelaxed solve of a zero state under a load, with lateral
  // (angle-varying) shear and bulk perturbations, equals the static
  // solver's lateral solve on the same fields (the two couplings hold
  // the same nodes: every solid element above the centre, GLL points).
  auto body = Body::Make();
  const auto top = [&] {
    auto s = 0.0;
    for (auto v : body.material.Moduli().L) s = std::max(s, v);
    return s;
  }();
  (void)top;
  Viscoelastic3D op3(
      body.rheology, kLmax, [](double, double, double) { return 0.0; },
      {.memory_points = MemoryPointSet::Gll,
       .moduli = {.shear =
                      [](double r, double theta, double phi) {
                        return 0.1 * (0.5 + r * r) *
                               (std::sin(theta) * std::cos(theta) * std::cos(phi) +
                                0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
                      },
                  .bulk = [](double r, double theta, double) {
                    return 0.1 * (1.0 - 0.5 * r) * std::cos(theta);
                  }}});
  ElasticPerturbation3D born(body.material, kLmax);
  ASSERT_EQ(born.Coupling().NodeCount(), op3.Coupling().NodeCount());
  ASSERT_EQ(born.Coupling().GridValueCount(), op3.Coupling().GridValueCount());
  const auto& grid = born.Coupling().Grid();
  auto load = grid.ZeroCoefficients();
  for (auto l = 0; l <= kLmax; ++l) {
    const auto columns = LoadColumns(l);
    for (auto m = 0; m <= l; ++m) load[SurfaceGrid::Index(l, m)] = columns[m];
  }
  const auto [y, info] = born.SolveLateral(
      load, op3.UnrelaxedShearPerturbation(), op3.BulkPerturbation(), {.rtol = 1e-13});
  ASSERT_TRUE(info.converged);
  const auto x = op3.SolveElastic(op3.ZeroState(), [](int l, double) { return LoadColumns(l); });
  EXPECT_TRUE(op3.LastElasticSolve().converged);
  for (const auto component : {Component::U, Component::V, Component::Phi}) {
    const auto a = op3.Surface(x, component);
    const auto b = born.Surface(y, component);
    auto scale = 0.0, diff = 0.0;
    for (const auto& [l, v] : b) {
      for (auto m = 0; m <= l; ++m) {
        scale = std::max(scale, std::abs(v[m]));
        diff = std::max(diff, std::abs(a.at(l)[m] - v[m]));
      }
    }
    EXPECT_LT(diff, 1e-9 * scale) << static_cast<int>(component);
  }
  // toroidal motion alive, and the same in both
  auto toroidal = 0.0, diff = 0.0;
  for (const auto& [l, w] : x.toroidal) {
    const auto& v = y.toroidal.at(l);
    for (auto i = std::size_t{0}; i < w.size(); ++i) {
      toroidal = std::max(toroidal, std::abs(v[i]));
      diff = std::max(diff, std::abs(w[i] - v[i]));
    }
  }
  EXPECT_GT(toroidal, 0.0);
  EXPECT_LT(diff, 1e-9 * toroidal);
  std::cout << "start solve: " << op3.LastElasticSolve().iterations
            << " iterations, static solve " << info.iterations << '\n';
}

TEST(LateralModuli3D, StepSelfConvergesAtSecondOrderWithModuliAndViscosity) {
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  Viscoelastic3D op(
      body.rheology, kLmax,
      [](double, double theta, double phi) { return 0.6 * std::sin(theta) * std::sin(phi); },
      {.moduli = {.shear =
                      [](double, double theta, double phi) {
                        return 0.1 * std::sin(theta) * std::cos(theta) * std::cos(phi);
                      },
                  .bulk = [](double, double theta, double) {
                    return 0.05 * std::cos(theta);
                  }}});
  const auto load = [](int l, double) { return LoadColumns(l); };
  const auto T = 2.0 * tau;
  auto passes = 0;
  auto advance = [&](int steps) {
    auto state = op.ZeroState();
    Viscoelastic3D::Solutions solutions;
    for (auto i = 0; i < steps; ++i) {
      auto result = op.Step(state, T / steps, Scheme::ExponentialTrapezoid, load);
      EXPECT_TRUE(result.info.converged);
      passes = std::max(passes, result.info.iterations);
      state = std::move(result.state);
      solutions = std::move(result.solutions);
    }
    return op.Surface(solutions, Component::U).at(2)[2].real();
  };
  const auto x1 = advance(4);
  const auto x2 = advance(8);
  const auto x4 = advance(16);
  const auto ratio = (x1 - x2) / (x2 - x4);
  EXPECT_NEAR(ratio, 4.0, 1.5);
  std::cout << "self-convergence ratio " << ratio << ", at most " << passes
            << " passes a step\n";
}

TEST(LateralModuli3D, CoupledEvolutionMatchesTheShiftedRheology) {
  // The sea-level-coupled evolution under angle-uniform lateral moduli
  // equals the evolution on the shifted rheology with no lateral moduli,
  // at every node INCLUDING the instantaneous start (the zero-length
  // step's fixed point closes the start's lateral response), with
  // rotation on: sea level, displacement, potential and angular velocity.
  auto body = Body::Make();
  const auto r0 = body.material.Mesh().Right(0);
  const auto shear = [=](double r) {
    return r > r0 ? 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r)) : 0.0;
  };
  const auto bulk = [=](double r) { return r > r0 ? 0.1 * (1.0 - 0.5 * r) : 0.0; };
  const auto shifted = body.Shifted(shear, bulk);
  ViscoelasticOperator reference(body.rheology, kLmax);
  SeaLevelSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
                        SeaLevelParameters{});
  Fingerprint fingerprint(kLmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  const auto length = solver.SolverScales().Length();
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto ice = toNd(fingerprint.IceThickness());
  const auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
  auto load = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < load.size(); ++i) {
    load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) * -0.1 * ice[i];
  }
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 3; ++n) times.push_back(2.5 * n * kyr);
  const auto iceLoad = [&](double t) {
    auto out = load;
    for (auto& v : out) v *= 0.5 + 0.5 * t / times.back();  // non-zero at the start
    return out;
  };
  auto run = [&](const Rheology& rheology, bool moduli) {
    Viscoelastic3D::Options options{.memory_points = MemoryPointSet::Gll};
    if (moduli) {
      options.moduli = {.shear = [&](double r, double, double) { return shear(r); },
                        .bulk = [&](double r, double, double) { return bulk(r); }};
    }
    Viscoelastic3D op(rheology, kLmax, [](double, double, double) { return 0.0; },
                      options);
    SeaLevelEvolution3D evolution(op, solver, state,
                                  {.rtol = 1e-12, .coupled_rtol = 1e-12});
    return evolution.Run(times, iceLoad);
  };
  const auto a = run(body.rheology, true);
  const auto b = run(shifted, false);
  ASSERT_EQ(a.size(), b.size());
  for (auto n = std::size_t{0}; n < a.size(); ++n) {
    EXPECT_TRUE(a[n].converged) << n;
    for (const auto& [name, fa, fb] :
         {std::tuple{"sea level", &a[n].seaLevelChange, &b[n].seaLevelChange},
          std::tuple{"displacement", &a[n].displacement, &b[n].displacement},
          std::tuple{"potential", &a[n].potential, &b[n].potential}}) {
      auto scale = 0.0, diff = 0.0;
      for (auto i = std::size_t{0}; i < fa->size(); ++i) {
        scale = std::max(scale, std::abs((*fb)[i]));
        diff = std::max(diff, std::abs((*fa)[i] - (*fb)[i]));
      }
      EXPECT_GT(scale, 0.0) << name << " node " << n;
      EXPECT_LT(diff, 1e-9 * scale) << name << " node " << n;
    }
    for (auto j = 0; j < 3; ++j) {
      EXPECT_NEAR(a[n].angularVelocity[j], b[n].angularVelocity[j],
                  1e-9 * std::abs(b[n].angularVelocity[j]) + 1e-16)
          << "node " << n;
    }
    std::cout << "node " << n << ": " << a[n].coupledIterations << " passes\n";
  }
}

TEST(LateralModuli3D, AngleUniformDensityMatchesThe1DStepperWithGravityHeld) {
  // A smooth 5% shift of the density over the solid above the centre, at
  // the GLL memory points, gravity and stratification held: exactly the
  // 1D stepper on the material with its density replaced and gravity
  // held (Material::WithDensityGravityHeld), both schemes, two steps
  // from rest — the stepped forward's density pipeline against the
  // radial operator.
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto density = [](double r) { return 0.05 * (0.3 + 0.7 * r * r); };
  ViscoelasticOperator op1(ShiftedDensity(body, density), kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  Viscoelastic3D op3(
      body.rheology, kLmax, [](double, double, double) { return 0.0; },
      {.memory_points = MemoryPointSet::Gll,
       .moduli = {.density = [&](double r, double, double) { return density(r); }}});
  EXPECT_EQ(op3.CompanionCount(), 4u);
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto iterations = 0;
    const auto worst = CompareSteps(op1, op3, scheme, 0.7 * tau, 2, iterations);
    EXPECT_LT(worst, 1e-10) << "scheme " << static_cast<int>(scheme);
    std::cout << "density, scheme " << static_cast<int>(scheme) << ": worst " << worst
              << ", passes " << iterations << '\n';
  }
}

TEST(LateralModuli3D, DISABLED_DensityDiagnostic) {
  // Not a gate: the stepped density path against the 1D stepper with
  // gravity held, piece by piece — the elastic start per degree, then
  // one step — printed.
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto density = [](double r) { return 0.05 * (0.3 + 0.7 * r * r); };
  const auto shiftedRheology = ShiftedDensity(body, density);
  const auto& shifted = shiftedRheology.BaseMaterial();
  ViscoelasticOperator op1(shiftedRheology, kLmax, {.memory_points = MemoryPointSet::Gll});
  Viscoelastic3D op3(
      body.rheology, kLmax, [](double, double, double) { return 0.0; },
      {.memory_points = MemoryPointSet::Gll,
       .moduli = {.density = [&](double r, double, double) { return density(r); }}});
  const auto load3 = [](int l, double) { return LoadColumns(l); };
  const auto x3 = op3.SolveElastic(op3.ZeroState(), load3);
  std::cout << "start solve: " << op3.LastElasticSolve().iterations << " iterations, residual "
            << op3.LastElasticSolve().residual << '\n';
  for (auto l = 0; l <= kLmax; ++l) {
    const DegreeOperator op(shifted, l, DegreeOperator::Options{});
    const auto ndof = static_cast<std::size_t>(op.DofCount());
    const auto unit = op.Solve(op.Load(), 1);
    const auto columns = LoadColumns(l);
    auto worst = 0.0, scale = 0.0;
    for (auto m = 0; m <= l; ++m) {
      for (auto d = std::size_t{0}; d < ndof; ++d) {
        const auto ref = columns[m] * unit[d];
        const auto ours = x3.spheroidal.at(l)[static_cast<std::size_t>(m) * ndof + d];
        scale = std::max(scale, std::abs(ref));
        worst = std::max(worst, std::abs(ours - ref));
      }
    }
    std::cout << "  start degree " << l << ": " << worst / scale << '\n';
  }
  for (auto steps : {1, 2}) {
    auto iterations = 0;
    const auto worst = CompareSteps(op1, op3, Scheme::BackwardEuler, 0.7 * tau, steps, iterations);
    std::cout << "after " << steps << " step(s): worst " << worst << '\n';
  }
}

TEST(LateralModuli3D, DISABLED_DensityStaticDiagnostic) {
  // Not a gate: the stepper's start solve against the static solver's
  // full solve on the same density field, and the static solver's full
  // and first-order solves against the 1D shifted operator, per degree.
  auto body = Body::Make();
  const auto density = [](double r) { return 0.05 * (0.3 + 0.7 * r * r); };
  const auto shiftedRheology = ShiftedDensity(body, density);
  const auto& shifted = shiftedRheology.BaseMaterial();
  Viscoelastic3D op3(
      body.rheology, kLmax, [](double, double, double) { return 0.0; },
      {.memory_points = MemoryPointSet::Gll,
       .moduli = {.density = [&](double r, double, double) { return density(r); }}});
  ElasticPerturbation3D born(body.material, kLmax);
  const auto& grid = born.Coupling().Grid();
  auto load = grid.ZeroCoefficients();
  for (auto l = 0; l <= kLmax; ++l) {
    const auto columns = LoadColumns(l);
    for (auto m = 0; m <= l; ++m) load[SurfaceGrid::Index(l, m)] = columns[m];
  }
  const auto x3 = op3.SolveElastic(op3.ZeroState(), [](int l, double) { return LoadColumns(l); });
  const auto [xb, info] = born.SolveLateral(load, {}, {}, op3.DensityPerturbation(), {.rtol = 1e-13});
  const auto x0 = born.SolveLoad(load);
  const auto dx = born.Respond(x0, {}, {}, op3.DensityPerturbation());
  std::cout << "static solve: " << info.iterations << " iterations\n";
  for (auto l = 0; l <= kLmax; ++l) {
    const DegreeOperator op(shifted, l, DegreeOperator::Options{});
    const auto ndof = static_cast<std::size_t>(op.DofCount());
    const auto unit = op.Solve(op.Load(), 1);
    const auto columns = LoadColumns(l);
    auto w3 = 0.0, wb = 0.0, w1 = 0.0, scale = 0.0, bornScale = 0.0;
    for (auto m = 0; m <= l; ++m) {
      for (auto d = std::size_t{0}; d < ndof; ++d) {
        const auto i = static_cast<std::size_t>(m) * ndof + d;
        const auto ref = columns[m] * unit[d];
        scale = std::max(scale, std::abs(ref));
        bornScale = std::max(bornScale, std::abs(dx.spheroidal.at(l)[i]));
        w3 = std::max(w3, std::abs(x3.spheroidal.at(l)[i] - ref));
        wb = std::max(wb, std::abs(xb.spheroidal.at(l)[i] - ref));
        w1 = std::max(w1, std::abs(x0.spheroidal.at(l)[i] + dx.spheroidal.at(l)[i] - ref));
      }
    }
    std::cout << "  degree " << l << ": stepper start " << w3 / scale << ", static full "
              << wb / scale << ", static first order " << w1 / scale
              << " (born/full " << bornScale / scale << ")\n";
  }
}

TEST(LateralModuli3D, CoupledEvolutionMatchesTheShiftedDensity) {
  // The sea-level-coupled evolution under an angle-uniform lateral
  // density with the axial rotational feedback on equals the evolution
  // on the rheology with its density replaced and gravity held, at every
  // node including the start: the tide's body force rho(x) grad psi and
  // the inertia moment int rho(x) U r^3 carry the lateral density too.
  auto body = Body::Make();
  const auto density = [](double r) { return 0.05 * (0.3 + 0.7 * r * r); };
  const auto shifted = ShiftedDensity(body, density);
  ViscoelasticOperator reference(body.rheology, kLmax);
  SeaLevelSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
                        SeaLevelParameters{});
  Fingerprint fingerprint(kLmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  const auto length = solver.SolverScales().Length();
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto ice = toNd(fingerprint.IceThickness());
  const auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
  auto load = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < load.size(); ++i) {
    load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) * -0.1 * ice[i];
  }
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 3; ++n) times.push_back(2.5 * n * kyr);
  const auto iceLoad = [&](double t) {
    auto out = load;
    for (auto& v : out) v *= 0.5 + 0.5 * t / times.back();
    return out;
  };
  auto run = [&](const Rheology& rheology, bool moduli) {
    Viscoelastic3D::Options options{.memory_points = MemoryPointSet::Gll};
    if (moduli) {
      options.moduli = {.density = [&](double r, double, double) { return density(r); }};
    }
    Viscoelastic3D op(rheology, kLmax, [](double, double, double) { return 0.0; },
                      options);
    SeaLevelEvolution3D evolution(
        op, solver, state,
        {.rotation = RotationalFeedback::Axial, .rtol = 1e-12, .coupled_rtol = 1e-12});
    return evolution.Run(times, iceLoad);
  };
  const auto a = run(body.rheology, true);
  const auto b = run(shifted, false);
  ASSERT_EQ(a.size(), b.size());
  for (auto n = std::size_t{0}; n < a.size(); ++n) {
    for (const auto& [name, fa, fb] :
         {std::tuple{"sea level", &a[n].seaLevelChange, &b[n].seaLevelChange},
          std::tuple{"displacement", &a[n].displacement, &b[n].displacement},
          std::tuple{"potential", &a[n].potential, &b[n].potential}}) {
      auto scale = 0.0, diff = 0.0;
      for (auto i = std::size_t{0}; i < fa->size(); ++i) {
        scale = std::max(scale, std::abs((*fb)[i]));
        diff = std::max(diff, std::abs((*fa)[i] - (*fb)[i]));
      }
      EXPECT_GT(scale, 0.0) << name << " node " << n;
      EXPECT_LT(diff, 1e-9 * scale) << name << " node " << n << " " << diff / scale;
    }
  }
}

TEST(LateralModuli3D, TideAndMomentTransposesUnderLateralDensity) {
  // The two rotational pieces of a lateral density and their transposes,
  // as identities on one operator: (i) the tide's lateral force paired
  // with solutions y equals the tide amplitudes paired with the
  // companions' dual of y read along the potential's radial profile;
  // (ii) the lateral inertia moment of solutions u equals u paired with
  // the force of the moment's grid weight W r^3 delta rho on the U
  // companion. What the discrete sweep's channel and moment duals rely on.
  auto body = Body::Make();
  Viscoelastic3D op(
      body.rheology, kLmax, [](double, double, double) { return 0.0; },
      {.memory_points = MemoryPointSet::Gll,
       .moduli = {.density = [](double r, double theta, double phi) {
         return 0.03 * (0.3 + 0.7 * r * r) *
                (std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi));
       }}});
  const auto& coupling = op.Coupling();
  const auto load3 = [](int l, double) { return LoadColumns(l); };
  const auto y = op.SolveElastic(op.ZeroState(), load3);
  const auto u = op.SolveElastic(op.ZeroState(), [](int l, double) {
    auto c = LoadColumns(l);
    for (auto& v : c) v *= 0.7 + 0.1 * l;
    return c;
  });
  const auto sph = [&](int l) -> const DegreeOperator& { return op.UnrelaxedSpheroidal(l); };
  const auto tor = [&](int l) -> const ToroidalOperator& { return op.UnrelaxedToroidal(l); };
  // (i) the tide
  auto tide = std::map<int, std::vector<Complex>>{};
  for (auto l = 0; l <= kLmax; ++l) {
    auto c = LoadColumns(l);
    for (auto& v : c) v *= 0.3 * (l + 1);
    tide.emplace(l, std::move(c));
  }
  const auto forces = op.LateralTideForces(tide, sph);
  auto lhs = 0.0;
  for (const auto& [l, f] : forces) {
    const auto& yl = y.spheroidal.at(l);
    const auto ndof = yl.size() / static_cast<std::size_t>(l + 1);
    for (auto m = 0; m <= l; ++m) {
      const auto w = m == 0 ? 1.0 : 2.0;
      for (auto d = std::size_t{0}; d < ndof; ++d) {
        const auto i = static_cast<std::size_t>(m) * ndof + d;
        lhs += w * (yl[i] * std::conj(f[i])).real();
      }
    }
  }
  // the companions' dual of y: A D S P y
  auto grid = coupling.Synthesise(op.StrainCoefficients(y, sph, tor));
  op.WeightCompanions(grid);
  const auto dual = coupling.Analyse(grid);
  const auto den = static_cast<std::size_t>(op.DensityBlock());
  const auto radius = body.material.Radius();
  auto rhs = 0.0;
  for (auto l = 0; l <= kLmax; ++l) {
    const auto p = l == 0 ? 2 : l;
    for (auto m = 0; m <= l; ++m) {
      Complex s = 0.0;
      for (auto node = 0; node < coupling.NodeCount(); ++node) {
        if (coupling.ElementOfNode(node) < op.UnrelaxedSpheroidal(l).FirstElement()) continue;
        const auto psi = std::pow(coupling.RadiusOfNode(node) / radius, p);
        const auto at = coupling.CoefficientOffset(node) + SurfaceGrid::Index(l, m);
        s += coupling.WeightOfNode(node) * coupling.RadiusOfNode(node) * psi *
             (static_cast<double>(p) * dual.scalars[den + 0][at] + dual.scalars[den + 1][at]);
      }
      const auto w = m == 0 ? 1.0 : 2.0;
      rhs += w * (tide.at(l)[m] * std::conj(s)).real();
    }
  }
  std::cout << "tide: lhs " << lhs << " rhs " << rhs << '\n';
  EXPECT_NEAR(lhs, rhs, 1e-10 * std::abs(lhs));
  EXPECT_NE(lhs, 0.0);
  // (ii) the moment
  auto gridU = coupling.Synthesise(op.StrainCoefficients(u, sph, tor));
  const auto moment = op.LateralMomentOf(gridU);
  auto field = std::vector<double>(coupling.GridValueCount(), 0.0);
  {
    const auto dRho = op.DensityPerturbation();
    for (auto node = 0; node < coupling.NodeCount(); ++node) {
      const auto r = coupling.RadiusOfNode(node);
      const auto factor = coupling.WeightOfNode(node) * r * r * r;
      const auto at = coupling.FieldOffset(node);
      for (auto x = std::size_t{0}; x < coupling.FieldSizeOf(node); ++x) {
        field[at + x] = factor * dRho[at + x];
      }
    }
  }
  const auto momentForces =
      coupling.PointwiseForce(coupling.AnalyseScalar(field), Component::U, sph);
  auto paired = 0.0;
  for (const auto& [l, f] : momentForces) {
    const auto& ul = u.spheroidal.at(l);
    const auto ndof = ul.size() / static_cast<std::size_t>(l + 1);
    for (auto m = 0; m <= l; ++m) {
      const auto w = m == 0 ? 1.0 : 2.0;
      for (auto d = std::size_t{0}; d < ndof; ++d) {
        const auto i = static_cast<std::size_t>(m) * ndof + d;
        paired += w * (ul[i] * std::conj(f[i])).real();
      }
    }
  }
  std::cout << "moment: direct " << moment << " paired " << paired << '\n';
  EXPECT_NEAR(moment, paired, 1e-10 * std::abs(moment));
  EXPECT_NE(moment, 0.0);
}

TEST(LateralModuli3D, DISABLED_StepTiming) {
  // Not a gate: the cost of a step with lateral viscosity alone, with
  // lateral moduli, and with density too, at SL3DPP_LMAX (default 32),
  // threads from OMP_NUM_THREADS; three steps each, the last timed.
  const auto* env = std::getenv("SL3DPP_LMAX");
  const auto lmax = env ? std::atoi(env) : 32;
  auto deck =
      Deck::Read(
          std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
          DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  auto model = EarthModel::FromDeck(deck).value();
  auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
  const auto material = Material::Create(model, mesh).value();
  const auto& scales = model.ModelScales();
  auto viscosity = std::map<std::size_t, RadialValue>{};
  const auto b = model.DimensionlessView().Boundaries();
  for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
    if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
        b[i + 1] <= 6.1511e6 / scales.Length()) {
      viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
    }
  }
  const auto rheology = Rheology::Maxwell(material, viscosity).value();
  const auto tau = rheology.MinRelaxationTime();
  const auto load = [](int l, double) {
    auto c = std::vector<Complex>(l + 1, 0.0);
    for (auto m = 0; m <= std::min(l, 4); ++m) c[m] = std::exp(-0.002 * l * l);
    return c;
  };
  auto delta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto shear = [](double r, double theta, double phi) {
    return 0.1 * (0.5 + r * r) * std::sin(theta) * std::sin(theta) * std::cos(2 * phi);
  };
  const auto density = [](double r, double theta, double) {
    return 0.03 * (0.3 + 0.7 * r * r) * std::cos(theta);
  };
  struct Case {
    const char* name;
    Viscoelastic3D::Options::LateralModuli moduli;
  };
  const auto inner = 3.4799e6 / scales.Length(), outer = 6.1511e6 / scales.Length();
  const auto mantleShear = [=](double r, double theta, double phi) {
    return r > inner && r < outer ? shear(r, theta, phi) : 0.0;
  };
  for (const auto& c : {Case{"viscosity only", {}},
                        Case{"+ mantle shear", {.shear = mantleShear}},
                        Case{"+ shear everywhere", {.shear = shear}},
                        Case{"+ shear + density", {.shear = shear, .density = density}}}) {
    Viscoelastic3D op(rheology, lmax, delta, {.moduli = c.moduli});
    auto state = op.ZeroState();
    using Seconds = std::chrono::duration<double>;
    auto last = 0.0;
    auto passes = 0;
    for (auto i = 0; i < 3; ++i) {
      const auto start = std::chrono::steady_clock::now();
      auto result = op.Step(state, tau, Scheme::ExponentialTrapezoid, load);
      last = Seconds(std::chrono::steady_clock::now() - start).count();
      passes = result.info.iterations;
      state = std::move(result.state);
    }
    std::cout << "lmax " << lmax << ", " << c.name << ": " << last << " s for "
              << passes << " passes (" << last / passes << " s a pass), nodes "
              << op.Coupling().NodeCount() << ", companions " << op.CompanionCount()
              << '\n';
  }
}
