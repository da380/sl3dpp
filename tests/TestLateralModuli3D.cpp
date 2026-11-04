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
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
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
