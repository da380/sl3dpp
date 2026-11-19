// Lateral elastic moduli in the 3D stepper: an angle-uniform shift of
// the unrelaxed shear and bulk moduli reproduces the 1D stepper on the
// shifted rheology at the GLL memory points, and at Gauss points the 1D
// stepper given the same perturbation at its memory points (and, for a
// per-element constant shift of the Maxwell region, where every shear
// term is a point term, the shifted rheology too); a perturbation at the
// GLL points is the rheology with its nodal moduli scaled; the unrelaxed
// start solve under lateral
// moduli is the static lateral solve; lateral moduli couple spheroidal
// and toroidal motion; the step self-converges at second order with
// lateral moduli and lateral viscosity on together.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <filesystem>
#include <format>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/MemoryPoints.hpp"
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

  /// With `viscousInnerCore`, the inner core takes the mantle's viscosity
  /// too: a body viscous down to its centre.
  static Body Make(bool viscousInnerCore = false) {
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
      if (body.model.IsFluid(i)) continue;
      if ((b[i] >= body.inner && b[i + 1] <= body.outer) ||
          (viscousInnerCore && b[i + 1] <= body.inner)) {
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
      }
    }
    body.rheology = Rheology::Maxwell(body.material, viscosity).value();
    return body;
  }

  /// The homogeneous test ball (AdGIA's E1 body) as a Maxwell body of
  /// 1e21 Pa s throughout: viscous down to its centre; the viscous range
  /// is the whole ball.
  static Body Ball() {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "e1_homogeneous.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
    auto material = Material::Create(model, mesh).value();
    const auto viscosity = 1e21 / model.ModelScales().Factor(ViscosityDimensions);
    auto rheology = Rheology::Maxwell(material, RadialValue(viscosity)).value();
    const auto outer = material.Radius();
    return Body{std::move(model), std::move(material), std::move(rheology), 0.0, outer};
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

TEST(LateralModuli3D, AngleUniformShiftMatchesThe1DStepperAtGaussPoints) {
  // Smooth 10% shifts of the unrelaxed shear and the bulk modulus over
  // the whole solid, at Gauss memory points: the 1D stepper with the
  // same perturbation sampled at its memory points (Options::moduli),
  // both schemes, two steps from rest. The two take the perturbation by
  // the one rule, so they agree as at the GLL points.
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto& mesh = body.material.Mesh();
  const auto r0 = mesh.Right(0);
  const auto shear = [=](double r) {
    return r > r0 ? 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r)) : 0.0;
  };
  const auto bulk = [=](double r) { return r > r0 ? 0.1 * (1.0 - 0.5 * r) : 0.0; };
  const auto points = MemoryPoints::Of(MemoryPointSet::Gauss, mesh);
  ViscoelasticOperator::Options::PointModuli moduli;
  moduli.shear.assign(static_cast<std::size_t>(mesh.ElementCount()) * points.count, 0.0);
  moduli.bulk = moduli.shear;
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (body.material.IsFluid(e)) continue;
    for (auto p = 0; p < points.count; ++p) {
      const auto i = static_cast<std::size_t>(e) * points.count + p;
      const auto r = points.RadiusAt(mesh, e, p);
      moduli.shear[i] = shear(r);
      moduli.bulk[i] = bulk(r);
    }
  }
  ViscoelasticOperator op1(body.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gauss,
                            .moduli = moduli});
  Viscoelastic3D op3(
      body.rheology, kLmax, [](double, double, double) { return 0.0; },
      {.memory_points = MemoryPointSet::Gauss,
       .moduli = {.shear = [&](double r, double, double) { return shear(r); },
                  .bulk = [&](double r, double, double) { return bulk(r); }}});
  // the reference's moduli at the points are the 3D stepper's at its nodes
  ViscoelasticOperator reference(body.rheology, kLmax,
                                 {.memory_points = MemoryPointSet::Gauss});
  const auto& coupling = op3.Coupling();
  // the same interpolation at the same points, no transforms: exact
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    const auto at = static_cast<std::size_t>(coupling.ElementOfNode(node)) *
                        points.count +
                    coupling.NodeInElementOf(node);
    ASSERT_EQ(reference.UnrelaxedShearAtPoints()[at],
              op3.UnrelaxedShearAtNodes()[node]);
    ASSERT_EQ(reference.BulkModulusAtPoints()[at],
              op3.BulkModulusAtNodes()[node]);
  }
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto iterations = 0;
    const auto worst = CompareSteps(op1, op3, scheme, 0.7 * tau, 2, iterations);
    EXPECT_LT(worst, 1e-10) << "scheme " << static_cast<int>(scheme);
    std::cout << "Gauss, smooth, scheme " << static_cast<int>(scheme)
              << ": worst " << worst << ", passes " << iterations << '\n';
    // and the perturbation is felt: the unperturbed stepper differs
    auto unused = 0;
    EXPECT_GT(CompareSteps(reference, op3, scheme, 0.7 * tau, 2, unused), 1e-4);
  }
}

TEST(LateralModuli3D, PointModuliAtGllPointsAreTheShiftedRheology) {
  // With the memory points at the GLL nodes a perturbation at the points
  // is a nodal one: the stepper given it is the stepper on the rheology
  // with its nodal moduli scaled, to rounding (the two form the scaled
  // moduli in different orders, which the solves carry to parts in 1e12).
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto& mesh = body.material.Mesh();
  const auto n = mesh.NGll();
  const auto r0 = mesh.Right(0);
  const auto shear = [=](double r) {
    return r > r0 ? 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r)) : 0.0;
  };
  const auto bulk = [=](double r) { return r > r0 ? 0.1 * (1.0 - 0.5 * r) : 0.0; };
  ViscoelasticOperator::Options::PointModuli moduli;
  moduli.shear.assign(static_cast<std::size_t>(mesh.ElementCount()) * n, 0.0);
  moduli.bulk = moduli.shear;
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (body.material.IsFluid(e)) continue;
    for (auto q = 0; q < n; ++q) {
      const auto i = static_cast<std::size_t>(e) * n + q;
      moduli.shear[i] = shear(mesh.Radius(e, q));
      moduli.bulk[i] = bulk(mesh.Radius(e, q));
    }
  }
  ViscoelasticOperator shifted(body.Shifted(shear, bulk), kLmax,
                               {.memory_points = MemoryPointSet::Gll});
  ViscoelasticOperator perturbed(body.rheology, kLmax,
                                 {.memory_points = MemoryPointSet::Gll,
                                  .moduli = moduli});
  const auto load = [](int l, double) { return Load1D(l); };
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto a = shifted.ZeroState([](int l) { return 2 * (l + 1); });
    auto b = perturbed.ZeroState([](int l) { return 2 * (l + 1); });
    auto xa = std::map<int, std::vector<double>>{};
    auto xb = xa;
    for (auto step = 0; step < 2; ++step) {
      auto [na, sa] = shifted.Step(a, 0.7 * tau, scheme, load);
      auto [nb, sb] = perturbed.Step(b, 0.7 * tau, scheme, load);
      a = std::move(na), b = std::move(nb);
      xa = std::move(sa), xb = std::move(sb);
    }
    for (const auto component : {Component::U, Component::V, Component::Phi}) {
      const auto ua = shifted.Surface(xa, component);
      const auto ub = perturbed.Surface(xb, component);
      auto scale = 0.0, diff = 0.0;
      for (auto l = 0; l <= kLmax; ++l) {
        for (auto c = std::size_t{0}; c < ua.at(l).size(); ++c) {
          scale = std::max(scale, std::abs(ua.at(l)[c]));
          diff = std::max(diff, std::abs(ub.at(l)[c] - ua.at(l)[c]));
        }
      }
      EXPECT_GT(scale, 0.0);
      EXPECT_LT(diff, 1e-10 * scale)
          << "component " << static_cast<int>(component);
    }
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

namespace {

void StepSelfConvergesAtSecondOrderWithModuliAndViscosity(Scheme scheme) {
  // The step under lateral viscosity, shear and bulk modulus, a constant
  // load switched on at the start: the degree-2, order-2 displacement
  // after 2 tau, over 4, 8 and 16 steps, converges at second order.
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
      auto result = op.Step(state, T / steps, scheme, load);
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

}  // namespace

TEST(LateralModuli3D, StepSelfConvergesAtSecondOrderWithModuliAndViscosity) {
  StepSelfConvergesAtSecondOrderWithModuliAndViscosity(
      Evolution3D::Options{}.scheme);
}
TEST(LateralModuli3D,
     StepSelfConvergesAtSecondOrderWithModuliAndViscosityUnderTheTrapezoid) {
  StepSelfConvergesAtSecondOrderWithModuliAndViscosity(
      Scheme::ExponentialTrapezoid);
}

TEST(LateralModuli3D, ConjugateGradientStepMatchesTheFixedPointUnderModuli) {
  // The conjugate-gradient closure of a step under lateral viscosity,
  // shear and bulk modulus (the bulk companion in the operator) lands
  // where the fixed point does, over two steps of 2 tau.
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto load = [](int l, double) { return LoadColumns(l); };
  auto run = [&](Viscoelastic3D::Options::StepSolver solver) {
    Viscoelastic3D::Options options;
    options.max_iterations = 200;
    // the fixed point's rounding floor here is about 1e-13
    options.rtol = 1e-11;
    options.solve_rtol = 1e-12;
    options.solver = solver;
    options.moduli = {.shear =
                          [](double, double theta, double phi) {
                            return 0.1 * std::sin(theta) * std::cos(theta) * std::cos(phi);
                          },
                      .bulk = [](double, double theta, double) {
                        return 0.05 * std::cos(theta);
                      }};
    const auto op = Viscoelastic3D(
        body.rheology, kLmax,
        [](double, double theta, double phi) { return 0.6 * std::sin(theta) * std::sin(phi); },
        options);
    auto first = op.Step(op.ZeroState(), 2.0 * tau, Scheme::ExponentialTrapezoid, load);
    EXPECT_TRUE(first.info.converged);
    auto second = op.Step(first.state, 2.0 * tau, Scheme::ExponentialTrapezoid, load);
    EXPECT_TRUE(second.info.converged);
    return std::pair{std::move(first), std::move(second)};
  };
  const auto [fixed1, fixed2] = run(Viscoelastic3D::Options::StepSolver::FixedPoint);
  const auto [cg1, cg2] = run(Viscoelastic3D::Options::StepSolver::ConjugateGradients);
  auto gap = [](const Viscoelastic3D::Solutions& a, const Viscoelastic3D::Solutions& b) {
    auto scale = 0.0, diff = 0.0;
    for (const auto& [l, x] : a.spheroidal) {
      const auto& y = b.spheroidal.at(l);
      for (auto i = std::size_t{0}; i < x.size(); ++i) {
        scale = std::max(scale, std::abs(y[i]));
        diff = std::max(diff, std::abs(x[i] - y[i]));
      }
    }
    return diff / scale;
  };
  EXPECT_LT(gap(cg1.solutions, fixed1.solutions), 1e-9);
  EXPECT_LT(gap(cg2.solutions, fixed2.solutions), 1e-9);
  std::cout << "step 2: " << fixed2.info.iterations << " fixed-point passes, "
            << cg2.info.iterations << " conjugate-gradient iterations\n";
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
  GRDSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
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
  const auto directLoad = [&](double t) {
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
    // the combined fixed point's rounding floor here is about 3e-13
    Evolution3D evolution(op, solver, state,
                                  {.rtol = 1e-12, .coupled_rtol = 1e-11});
    return evolution.Run(times, directLoad);
  };
  const auto a = run(body.rheology, true);
  const auto b = run(shifted, false);
  ASSERT_EQ(a.size(), b.size());
  for (auto n = std::size_t{0}; n < a.size(); ++n) {
    EXPECT_TRUE(a[n].converged) << n;
    for (const auto& [name, fa, fb] :
         {std::tuple{"sea level", &a[n].relativeSeaLevelChange, &b[n].relativeSeaLevelChange},
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

TEST(LateralModuli3D, NestedConjugateGradientEvolutionMatchesTheFixedPoint) {
  // The sea-level-coupled evolution closed by the nested conjugate-
  // gradient solves (the sea level closure outside, the step's solve
  // inside, warm-started and solved to the outer progress) equals the
  // evolution closed by the combined fixed point, under a 3x lateral
  // viscosity and lateral moduli with the rotational feedback on, at the
  // start (the zero-length step) and at every node: sea level,
  // displacement, potential and angular velocity.
  auto body = Body::Make();
  ViscoelasticOperator reference(body.rheology, kLmax);
  GRDSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
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
  const auto directLoad = [&](double t) {
    auto out = load;
    for (auto& v : out) v *= 0.5 + 0.5 * t / times.back();
    return out;
  };
  auto run = [&](Viscoelastic3D::Options::StepSolver stepSolver) {
    Viscoelastic3D::Options options;
    options.solver = stepSolver;
    options.rtol = 1e-12;
    options.solve_rtol = 1e-12;
    options.max_iterations = 200;
    options.moduli = {.shear =
                          [](double, double theta, double phi) {
                            return 0.1 * std::sin(theta) * std::cos(theta) * std::cos(phi);
                          },
                      .bulk = [](double, double theta, double) {
                        return 0.05 * std::cos(theta);
                      }};
    const auto amplitude = 0.5 * std::log(3.0);
    Viscoelastic3D op(
        body.rheology, kLmax,
        [=](double, double theta, double phi) { return amplitude * std::sin(theta) * std::sin(phi); },
        options);
    Evolution3D evolution(op, solver, state,
                                  {.rtol = 1e-12, .max_coupled_iterations = 200,
                                   .coupled_rtol = 1e-11});
    return evolution.Run(times, directLoad);
  };
  const auto a = run(Viscoelastic3D::Options::StepSolver::ConjugateGradients);
  const auto b = run(Viscoelastic3D::Options::StepSolver::FixedPoint);
  ASSERT_EQ(a.size(), b.size());
  for (auto n = std::size_t{0}; n < a.size(); ++n) {
    EXPECT_TRUE(a[n].converged) << n;
    EXPECT_TRUE(b[n].converged) << n;
    for (const auto& [name, fa, fb] :
         {std::tuple{"sea level", &a[n].relativeSeaLevelChange, &b[n].relativeSeaLevelChange},
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
    std::cout << "node " << n << ": nested " << a[n].coupledIterations
              << " passes, " << a[n].solveIterations
              << " conjugate-gradient iterations; fixed point "
              << b[n].coupledIterations << " passes\n";
  }
}

// -- the instantaneous response (Evolution3D::Instantaneous) ----------

namespace {

/// The sea level problem the instantaneous gates share: the radial
/// reference's elastic sea level solver, pyslfp's analytic present-day
/// state, and the grounded load of melting a tenth of its ice.
struct Ocean {
  GRDSolver solver;
  GRDSolver::State state;
  std::vector<double> load;

  static Ocean Make(const Body& body) {
    ViscoelasticOperator reference(body.rheology, kLmax);
    GRDSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
                          SeaLevelParameters{});
    Fingerprint fingerprint(kLmax, reference.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    const auto ice = toNd(fingerprint.IceThickness());
    auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
    auto load = std::vector<double>(ice.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) * -0.1 * ice[i];
    }
    return Ocean{std::move(solver), std::move(state), std::move(load)};
  }
};

/// Angle-varying moduli: 10% in shear, 5% in bulk.
Viscoelastic3D::Options::LateralModuli VaryingModuli() {
  return {.shear =
              [](double, double theta, double phi) {
                return 0.1 * std::sin(theta) * std::cos(theta) * std::cos(phi);
              },
          .bulk = [](double, double theta, double) { return 0.05 * std::cos(theta); }};
}

/// The largest difference of two fields over the largest magnitude of the
/// second.
template <class A, class B>
double Departure(const A& got, const B& want) {
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    scale = std::max(scale, std::abs(want[i]));
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  return scale > 0.0 ? diff / scale : diff;
}

/// The worst departure of a snapshot's surface fields and angular velocity
/// from a sea level response's.
double SnapshotDeparture(const Evolution3D::Snapshot& got,
                         const GRDSolver::Response& want) {
  return std::max({Departure(got.relativeSeaLevelChange, want.relativeSeaLevelChange),
                   Departure(got.displacement, want.displacement),
                   Departure(got.potential, want.potential),
                   Departure(got.angularVelocity, want.angularVelocity)});
}

}  // namespace

TEST(LateralModuli3D, InstantaneousIsTheRunsFirstNode) {
  // The instantaneous response to the load at a run's first node is the
  // run's first node, field by field (the same code; equal to the bit
  // here): with and without lateral moduli, rotation off and full, both
  // step solvers, the states recorded; at a first node away from zero.
  auto body = Body::Make();
  const auto ocean = Ocean::Make(body);
  const auto kyr = 1000.0 * 3.15576e7 / ocean.solver.SolverScales().Time();
  const auto times = std::vector<double>{2.0 * kyr, 3.0 * kyr};
  const auto directLoad = [&](double t) {
    auto out = ocean.load;
    for (auto& v : out) v *= 0.5 + 0.25 * t / kyr;
    return out;
  };
  using StepSolver = Viscoelastic3D::Options::StepSolver;
  auto worst = 0.0;
  for (const auto moduli : {false, true}) {
    for (const auto rotation : {RotationalFeedback::None, RotationalFeedback::Full}) {
      for (const auto stepSolver : {StepSolver::FixedPoint, StepSolver::ConjugateGradients}) {
        const auto label = std::format("moduli {}, rotation {}, solver {}", moduli,
                                       static_cast<int>(rotation),
                                       static_cast<int>(stepSolver));
        Viscoelastic3D::Options options;
        options.solver = stepSolver;
        options.rtol = 1e-12;
        options.solve_rtol = 1e-12;
        if (moduli) options.moduli = VaryingModuli();
        const Viscoelastic3D op(body.rheology, kLmax,
                                [](double, double, double) { return 0.0; }, options);
        const auto evolutionOptions = Evolution3D::Options{
            .rotation = rotation, .rtol = 1e-12, .max_coupled_iterations = 200,
            .coupled_rtol = 1e-11, .record_stride = 1};
        Evolution3D run(op, ocean.solver, ocean.state, evolutionOptions);
        const auto snapshots = run.Run(times, directLoad);
        const auto& a = snapshots.front();
        Evolution3D instant(op, ocean.solver, ocean.state, evolutionOptions);
        const auto b = instant.Instantaneous(directLoad(times[0]), times[0]);
        EXPECT_EQ(b.t, a.t) << label;
        EXPECT_TRUE(a.converged) << label;
        EXPECT_TRUE(b.converged) << label;
        EXPECT_EQ(b.coupledIterations, a.coupledIterations) << label;
        EXPECT_EQ(b.solveIterations, a.solveIterations) << label;
        EXPECT_EQ(b.coupledResidual, a.coupledResidual) << label;
        // the fixed point ran under lateral moduli alone
        EXPECT_EQ(b.coupledIterations > 0, moduli) << label;
        EXPECT_EQ(b.solveIterations > 0, moduli && stepSolver == StepSolver::ConjugateGradients)
            << label;
        auto departure = 0.0;
        for (const auto& [name, fa, fb] :
             {std::tuple{"sea level", &a.relativeSeaLevelChange, &b.relativeSeaLevelChange},
              std::tuple{"displacement", &a.displacement, &b.displacement},
              std::tuple{"potential", &a.potential, &b.potential}}) {
          auto scale = 0.0;
          for (const auto v : *fa) scale = std::max(scale, std::abs(v));
          EXPECT_GT(scale, 0.0) << name << ", " << label;
          departure = std::max(departure, Departure(*fb, *fa));
        }
        departure = std::max({departure, Departure(b.angularVelocity, a.angularVelocity),
                              Departure(b.loadCoefficients, a.loadCoefficients),
                              Departure(b.tideCoefficients, a.tideCoefficients)});
        EXPECT_LE(departure, 1e-13) << label;
        worst = std::max(worst, departure);
        // the recorded state and the state left behind
        EXPECT_TRUE(a.internalState.has_value()) << label;
        EXPECT_TRUE(b.internalState.has_value()) << label;
        EXPECT_EQ(b.internalState->t, a.internalState->t) << label;
        EXPECT_EQ(instant.FinalState().t, a.internalState->t) << label;
      }
    }
  }
  RecordProperty("worst_departure", std::format("{:.3e}", worst));
  std::cout << "Instantaneous against Run's first node: worst departure " << worst << '\n';
}

TEST(LateralModuli3D, InstantaneousWithoutLateralStructureIsTheRadialFingerprint) {
  // With no lateral moduli the instantaneous response is the sea level
  // equation closed by the elastic Love numbers: the radial fingerprint of
  // the load, to rounding. With lateral moduli set but zero everywhere it
  // goes through the zero-length step's fixed point, by either step
  // solver, and lands on the radial fingerprint to the solves' tolerance
  // (both equal to the bit here: the zero-length step's responses vanish,
  // and its sea level solve is the radial one).
  auto body = Body::Make();
  const auto ocean = Ocean::Make(body);
  using StepSolver = Viscoelastic3D::Options::StepSolver;
  for (const auto rotation : {RotationalFeedback::None, RotationalFeedback::Full}) {
    const auto radial = ocean.solver.Solve(ocean.state, ocean.load, rotation,
                                           {.rtol = 1e-12});
    ASSERT_TRUE(radial.converged);
    const auto evolutionOptions = Evolution3D::Options{
        .rotation = rotation, .rtol = 1e-12, .max_coupled_iterations = 200,
        .coupled_rtol = 1e-11};
    {
      const Viscoelastic3D op(body.rheology, kLmax,
                              [](double, double, double) { return 0.0; });
      ASSERT_FALSE(op.HasLateralModuli());
      Evolution3D evolution(op, ocean.solver, ocean.state, evolutionOptions);
      const auto response = evolution.Instantaneous(ocean.load);
      EXPECT_TRUE(response.converged);
      EXPECT_EQ(response.coupledIterations, 0);
      const auto departure = SnapshotDeparture(response, radial);
      EXPECT_LT(departure, 1e-12) << static_cast<int>(rotation);
      RecordProperty(std::format("radial_rotation{}", static_cast<int>(rotation)),
                     std::format("{:.3e}", departure));
    }
    for (const auto stepSolver : {StepSolver::FixedPoint, StepSolver::ConjugateGradients}) {
      Viscoelastic3D::Options options;
      options.solver = stepSolver;
      options.rtol = 1e-12;
      options.solve_rtol = 1e-12;
      options.moduli = {.shear = [](double, double, double) { return 0.0; },
                        .bulk = [](double, double, double) { return 0.0; }};
      const Viscoelastic3D op(body.rheology, kLmax,
                              [](double, double, double) { return 0.0; }, options);
      ASSERT_TRUE(op.HasLateralModuli());
      Evolution3D evolution(op, ocean.solver, ocean.state, evolutionOptions);
      const auto response = evolution.Instantaneous(ocean.load);
      EXPECT_TRUE(response.converged);
      EXPECT_GT(response.coupledIterations, 0);
      const auto departure = SnapshotDeparture(response, radial);
      EXPECT_LT(departure, 1e-9) << static_cast<int>(rotation) << ", "
                                 << static_cast<int>(stepSolver);
      RecordProperty(std::format("zero_moduli_rotation{}_solver{}", static_cast<int>(rotation),
                                 static_cast<int>(stepSolver)),
                     std::format("{:.3e}", departure));
    }
  }
}

TEST(LateralModuli3D, InstantaneousUnderLateralModuliDepartsFromTheRadialFingerprint) {
  // Under 10% shear and 5% bulk perturbations, angle-varying, the elastic
  // fingerprint departs from the radial one by a few per cent of its size
  // (measured 2.3e-2 in the sea level, 2.1e-2 in the displacement and
  // 2.6e-3 in the potential, relative to their largest magnitudes: below
  // the moduli's 10%, as a perturbation over part of the body should move
  // an elastic response, and far above the solves' tolerance), the two
  // step solvers agree, and the ocean's mass balances the load's to the
  // solve's tolerance.
  auto body = Body::Make();
  const auto ocean = Ocean::Make(body);
  const auto radial = ocean.solver.Solve(ocean.state, ocean.load, RotationalFeedback::Full,
                                         {.rtol = 1e-12});
  ASSERT_TRUE(radial.converged);
  using StepSolver = Viscoelastic3D::Options::StepSolver;
  auto responses = std::vector<Evolution3D::Snapshot>{};
  for (const auto stepSolver : {StepSolver::FixedPoint, StepSolver::ConjugateGradients}) {
    Viscoelastic3D::Options options;
    options.solver = stepSolver;
    options.rtol = 1e-12;
    options.solve_rtol = 1e-12;
    options.moduli = VaryingModuli();
    const Viscoelastic3D op(body.rheology, kLmax,
                            [](double, double, double) { return 0.0; }, options);
    Evolution3D evolution(op, ocean.solver, ocean.state,
                                  {.rtol = 1e-12, .max_coupled_iterations = 200,
                                   .coupled_rtol = 1e-11});
    responses.push_back(evolution.Instantaneous(ocean.load));
    EXPECT_TRUE(responses.back().converged);
  }
  const auto& response = responses.front();
  const auto seaLevel = Departure(response.relativeSeaLevelChange, radial.relativeSeaLevelChange);
  const auto displacement = Departure(response.displacement, radial.displacement);
  const auto potential = Departure(response.potential, radial.potential);
  RecordProperty("sea_level_departure", std::format("{:.3e}", seaLevel));
  RecordProperty("displacement_departure", std::format("{:.3e}", displacement));
  RecordProperty("potential_departure", std::format("{:.3e}", potential));
  std::cout << "lateral moduli against the radial fingerprint: sea level " << seaLevel
            << ", displacement " << displacement << ", potential " << potential << '\n';
  for (const auto departure : {seaLevel, displacement, potential}) {
    EXPECT_GT(departure, 1e-3);
    EXPECT_LT(departure, 0.1);
  }
  EXPECT_LT(Departure(responses[1].relativeSeaLevelChange, response.relativeSeaLevelChange), 1e-9);
  EXPECT_LT(Departure(responses[1].displacement, response.displacement), 1e-9);

  // the ocean gains what the ice loses
  const auto& grid = ocean.solver.Grid();
  auto gain = std::vector<double>(grid.FieldSize());
  for (auto i = std::size_t{0}; i < gain.size(); ++i) {
    gain[i] = ocean.solver.WaterDensity() * ocean.state.oceanFunction[i] *
              response.relativeSeaLevelChange[i];
  }
  const auto gained = grid.Integrate(gain);
  const auto removed = -grid.Integrate(ocean.load);
  EXPECT_NEAR(gained, removed, 1e-9 * std::abs(removed));
}

TEST(LateralModuli3D, DISABLED_StepTiming) {
  // Not a gate: the cost of a step with lateral viscosity alone, with a
  // lateral shear (confined, then everywhere), and with a bulk modulus
  // too, at SL3DPP_LMAX (default 32),
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
                        Case{"+ shear + bulk", {.shear = shear, .bulk = [](double r, double theta, double) {
                          return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
                        }}}}) {
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

// -- the element holding the centre ----------------------------------------

namespace {

/// The surface U, V and phi per degree (complex m >= 0 columns) after
/// `steps` steps of `dt` from rest under the noise load.
std::array<std::map<int, std::vector<Complex>>, 3> Steps3D(const Viscoelastic3D& op,
                                                           Scheme scheme, double dt,
                                                           int steps) {
  const auto load = [](int l, double) { return LoadColumns(l); };
  auto state = op.ZeroState();
  Viscoelastic3D::Solutions solutions;
  for (auto step = 0; step < steps; ++step) {
    auto result = op.Step(state, dt, scheme, load);
    EXPECT_TRUE(result.info.converged) << "step " << step;
    state = std::move(result.state);
    solutions = std::move(result.solutions);
  }
  return {op.Surface(solutions, Component::U), op.Surface(solutions, Component::V),
          op.Surface(solutions, Component::Phi)};
}

/// The unrelaxed solve of a zero state under the noise load, as Steps3D's.
std::array<std::map<int, std::vector<Complex>>, 3> Unrelaxed3D(const Viscoelastic3D& op) {
  const auto x = op.SolveElastic(op.ZeroState(), [](int l, double) { return LoadColumns(l); });
  return {op.Surface(x, Component::U), op.Surface(x, Component::V),
          op.Surface(x, Component::Phi)};
}

/// The worst difference of two such surfaces, per component over its
/// largest magnitude.
double SurfaceGap(const std::array<std::map<int, std::vector<Complex>>, 3>& got,
                  const std::array<std::map<int, std::vector<Complex>>, 3>& want) {
  auto worst = 0.0;
  for (auto c = 0; c < 3; ++c) {
    auto scale = 0.0, diff = 0.0;
    for (const auto& [l, w] : want[c]) {
      for (auto m = std::size_t{0}; m < w.size(); ++m) {
        scale = std::max(scale, std::abs(w[m]));
        diff = std::max(diff, std::abs(got[c].at(l)[m] - w[m]));
      }
    }
    EXPECT_GT(scale, 0.0);
    worst = std::max(worst, diff / scale);
  }
  return worst;
}

/// The schemes the centre gates run: the default two-stage step and the
/// trapezoid.
constexpr auto kCentreSchemes = std::array{Scheme::Sdirk2, Scheme::ExponentialTrapezoid};

const char* SchemeName(Scheme scheme) {
  return scheme == Scheme::Sdirk2 ? "sdirk2" : "trapezoid";
}

}  // namespace

TEST(LateralModuli3D, ZeroModuliOnBodiesViscousToTheirCentreAreTheRunWithout) {
  // On bodies viscous down to the centre (the homogeneous Maxwell ball,
  // and PREM's Voigt average with a Maxwell inner core beside its Maxwell
  // mantle), lateral moduli set but zero everywhere are the run without
  // them: the coupling carries the element holding the centre at its
  // interior Gauss points either way, so that element's relaxation is
  // stepped alike. Four steps from rest under both schemes, and the
  // unrelaxed solve (the conjugate-gradient lateral solve against the
  // reference solve), to rounding.
  const auto zero = [](double, double, double) { return 0.0; };
  auto check = [&](const std::string& name, const Body& body) {
    const auto tau = body.rheology.MinRelaxationTime();
    const Viscoelastic3D plain(body.rheology, kLmax, zero);
    const Viscoelastic3D zeroModuli(body.rheology, kLmax, zero,
                                    {.moduli = {.shear = zero, .bulk = zero}});
    ASSERT_TRUE(zeroModuli.HasLateralModuli());
    // the same memory nodes, the element holding the centre among them
    ASSERT_EQ(zeroModuli.Coupling().NodeCount(), plain.Coupling().NodeCount()) << name;
    EXPECT_EQ(zeroModuli.Coupling().ElementOfNode(0), 0) << name;
    const auto unrelaxed = SurfaceGap(Unrelaxed3D(zeroModuli), Unrelaxed3D(plain));
    std::cout << std::format("[{}] the unrelaxed solve: {:.3e}\n", name, unrelaxed);
    RecordProperty(name + "_unrelaxed", std::format("{:.3e}", unrelaxed));
    EXPECT_LT(unrelaxed, 1e-10) << name;
    for (const auto scheme : kCentreSchemes) {
      const auto gap = SurfaceGap(Steps3D(zeroModuli, scheme, 0.7 * tau, 4),
                                  Steps3D(plain, scheme, 0.7 * tau, 4));
      std::cout << std::format("[{}, {}] four steps: {:.3e}\n", name, SchemeName(scheme), gap);
      RecordProperty(name + "_" + SchemeName(scheme), std::format("{:.3e}", gap));
      EXPECT_LT(gap, 1e-11) << name << " " << SchemeName(scheme);
    }
  };
  check("ball", Body::Ball());
  check("prem_viscous_inner_core", Body::Make(true));
}

TEST(LateralModuli3D, ShiftsReachingTheCentreMatchThe1DStepperOnABallViscousToItsCentre) {
  // Lateral moduli over every element of the homogeneous Maxwell ball, the
  // element holding the centre included, angle-uniform:
  //  - a shift of the shear constant over each element, 25% in the element
  //    holding the centre and 10% to 15% above it (the long-term modulus of
  //    a Maxwell body vanishes, so at the Gauss points every shear term is
  //    a point term), is exactly the 1D stepper on the ball with the
  //    shifted modulus in its rheology, and the unrelaxed solve the shifted
  //    ball's;
  //  - smooth shifts of the shear and the bulk modulus, non-zero at the
  //    centre, are the 1D stepper given them at its memory points;
  // both schemes, four steps from rest, to rounding. The departure where
  // the element holding the centre is left out is printed for scale.
  const auto body = Body::Ball();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto& mesh = body.material.Mesh();
  const auto zero = [](double, double, double) { return 0.0; };
  // the element holding an interior radius (the Gauss points are interior)
  const auto elementOf = [&](double r) {
    auto e = 0;
    while (e + 1 < mesh.ElementCount() && r >= mesh.Right(e)) ++e;
    return e;
  };
  const auto stepped = [&](double r) { return r < mesh.Right(0) ? 0.25 : 0.1 + 0.05 * r; };
  const auto midpoint = [&](double r) {
    const auto e = elementOf(r);
    return stepped(0.5 * (mesh.Left(e) + mesh.Right(e)));
  };
  const auto none = [](double) { return 0.0; };
  const auto shifted = body.Shifted(stepped, none, true);
  const auto points = MemoryPoints::Of(MemoryPointSet::Gauss, mesh);
  {
    const Viscoelastic3D op3(body.rheology, kLmax, zero,
                             {.moduli = {.shear = [&](double r, double, double) {
                                return midpoint(r);
                              }}});
    EXPECT_EQ(op3.Coupling().ElementOfNode(0), 0);
    const ViscoelasticOperator op1(shifted, kLmax);
    const Viscoelastic3D shifted3(shifted, kLmax, zero);
    const auto unrelaxed = SurfaceGap(Unrelaxed3D(op3), Unrelaxed3D(shifted3));
    std::cout << std::format("[per-element shear] the unrelaxed solve: {:.3e}\n", unrelaxed);
    RecordProperty("per_element_unrelaxed", std::format("{:.3e}", unrelaxed));
    EXPECT_LT(unrelaxed, 1e-10);
    for (const auto scheme : kCentreSchemes) {
      auto iterations = 0;
      const auto gap = CompareSteps(op1, op3, scheme, 0.7 * tau, 4, iterations);
      std::cout << std::format("[per-element shear, {}] four steps: {:.3e} ({} passes)\n",
                               SchemeName(scheme), gap, iterations);
      RecordProperty(std::string{"per_element_"} + SchemeName(scheme),
                     std::format("{:.3e}", gap));
      EXPECT_LT(gap, 1e-10) << SchemeName(scheme);
    }
    // for scale: the same shift with the element holding the centre left
    // at its reference
    const Viscoelastic3D above(body.rheology, kLmax, zero,
                               {.moduli = {.shear = [&](double r, double, double) {
                                  return elementOf(r) == 0 ? 0.0 : midpoint(r);
                                }}});
    auto unused = 0;
    const auto centre =
        CompareSteps(op1, above, Scheme::Sdirk2, 0.7 * tau, 4, unused);
    std::cout << std::format("[per-element shear] without the centre's shift: {:.3e}\n",
                             centre);
    RecordProperty("centre_shift_share", std::format("{:.3e}", centre));
    EXPECT_GT(centre, 1e-8);
  }
  {
    const auto shear = [](double r) {
      return 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
    };
    const auto bulk = [](double r) { return 0.1 * (1.0 - 0.5 * r); };
    ViscoelasticOperator::Options::PointModuli moduli;
    moduli.shear.assign(static_cast<std::size_t>(mesh.ElementCount()) * points.count, 0.0);
    moduli.bulk = moduli.shear;
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      for (auto p = 0; p < points.count; ++p) {
        const auto i = static_cast<std::size_t>(e) * points.count + p;
        const auto r = points.RadiusAt(mesh, e, p);
        moduli.shear[i] = shear(r);
        moduli.bulk[i] = bulk(r);
      }
    }
    const ViscoelasticOperator op1(body.rheology, kLmax, {.moduli = moduli});
    const Viscoelastic3D op3(body.rheology, kLmax, zero,
                             {.moduli = {.shear = [&](double r, double, double) { return shear(r); },
                                         .bulk = [&](double r, double, double) { return bulk(r); }}});
    for (const auto scheme : kCentreSchemes) {
      auto iterations = 0;
      const auto gap = CompareSteps(op1, op3, scheme, 0.7 * tau, 4, iterations);
      std::cout << std::format("[smooth shear and bulk, {}] four steps: {:.3e} ({} passes)\n",
                               SchemeName(scheme), gap, iterations);
      RecordProperty(std::string{"smooth_"} + SchemeName(scheme), std::format("{:.3e}", gap));
      EXPECT_LT(gap, 1e-10) << SchemeName(scheme);
    }
  }
}
