// DegreeOperator and the Love-number driver: structural identities and
// closed forms (Kelvin's fluid number, degree-0 mass conservation),
// ahead of the pyslfp fixture gate in TestFixtures.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Scales.hpp"
#include "sl3dpp/ToroidalOperator.hpp"

using namespace sl3dpp;

namespace {

/// A deck of (lo, hi, rho, kappa, mu) layers, constant per layer.
Deck LayeredDeck(const std::vector<std::array<double, 5>>& layers,
                 std::size_t n = 5) {
  std::vector<double> radius, rho, kappa, mu;
  for (const auto& [lo, hi, density, bulk, shear] : layers) {
    for (auto k = std::size_t{0}; k < n; ++k) {
      radius.push_back(std::lerp(lo, hi, static_cast<double>(k) / (n - 1)));
      rho.push_back(density);
      kappa.push_back(bulk);
      mu.push_back(shear);
    }
  }
  return Deck::Create(std::move(radius), {{"rho", std::move(rho)},
                                          {"kappa", std::move(kappa)},
                                          {"mu", std::move(mu)}})
      .value();
}

Material HomogeneousMaterial(int lmax) {
  constexpr auto b = 6.371e6;
  static auto model =
      EarthModel::FromDeck(LayeredDeck({{0.0, 0.5 * b, 5515.0, 400e9, 150e9},
                                        {0.5 * b, b, 5515.0, 400e9, 150e9}}))
          .value();
  auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
  return Material::Create(model, mesh).value();
}

/// Elastic inner core, fluid outer core, solid mantle: the interface terms
/// and the fluid potential-only treatment all exercised.
Material CoreMantleMaterial(int lmax) {
  constexpr auto b = 6.371e6;
  static auto model =
      EarthModel::FromDeck(LayeredDeck({{0.0, 1.2e6, 13000.0, 1300e9, 170e9},
                                        {1.2e6, 3.5e6, 11000.0, 1100e9, 0.0},
                                        {3.5e6, b, 5000.0, 400e9, 150e9}}))
          .value();
  auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
  return Material::Create(model, mesh).value();
}

}  // namespace

TEST(DegreeOperator, DofBookkeeping) {
  auto material = CoreMantleMaterial(8);
  const auto& mesh = material.Mesh();

  // l = 1: the surface potential dof is removed (centre-of-mass frame).
  DegreeOperator l1(material, 1);
  EXPECT_EQ(l1.SurfaceDof(Component::Phi), -1);
  EXPECT_GE(l1.SurfaceDof(Component::U), 0);
  EXPECT_EQ(l1.FirstElement(), 0);  // graded radius ~ 1e-4 of rmax

  // l = 0: no tangential dofs anywhere, U removed at the centre.
  DegreeOperator l0(material, 0);
  EXPECT_EQ(l0.SurfaceDof(Component::V), -1);
  EXPECT_EQ(l0.Dof(0, Component::U), -1);
  EXPECT_GE(l0.Dof(0, Component::Phi), 0);

  // High degree: solved on an element subset of the one global mesh.
  DegreeOperator l64(material, 64);
  EXPECT_GT(l64.FirstElement(), 0);
  EXPECT_LT(l64.DofCount(), DegreeOperator(material, 2).DofCount());

  // Fluid elements carry the potential alone above degree zero.
  DegreeOperator l2(material, 2);
  auto fluidNode = -1;
  for (auto e = 0; e + 1 < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e) && material.IsFluid(e + 1)) {
      fluidNode = mesh.GlobalIndex(e, mesh.NGll() - 1);
      break;
    }
  }
  ASSERT_GE(fluidNode, 0);
  EXPECT_EQ(l2.Dof(fluidNode, Component::U), -1);
  EXPECT_EQ(l2.Dof(fluidNode, Component::V), -1);
  EXPECT_GE(l2.Dof(fluidNode, Component::Phi), 0);
  EXPECT_TRUE(l2.UsedCholesky());

  // Expand: NaN displacement through fluid elements, potential live.
  const auto x = l2.Solve(l2.Load());
  const auto nodal = l2.Expand(x);
  auto sawNaN = false;
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e)) {
      sawNaN = sawNaN || std::isnan(nodal.U[e * mesh.NGll()]);
      EXPECT_FALSE(std::isnan(nodal.phi[e * mesh.NGll()]));
    }
  }
  EXPECT_TRUE(sawNaN);
}

TEST(DegreeOperator, KelvinFluidTidalNumber) {
  // The hydrostatic fluid limit of a homogeneous body is Kelvin's
  // k = 3 / (2 (l - 1)), whatever the (finite) compressibility.
  auto material = HomogeneousMaterial(8);
  for (int l : {2, 3, 4}) {
    const auto kelvin = 3.0 / (2.0 * (l - 1.0));
    EXPECT_NEAR(FluidTidalNumber(material, l), kelvin, 1e-8 * kelvin)
        << "l = " << l;
  }
}

TEST(DegreeOperator, MassConservationAtDegreeZero) {
  // k_0 = -4 pi G a: the load monopole's exterior potential.
  auto material = CoreMantleMaterial(8);
  auto ln = ComputeLoveNumbers(material, 2);
  const double expected =
      -4.0 * std::numbers::pi * GravitationalConstantSI * ln.radius;
  EXPECT_NEAR(ln.k_u[0] + ln.k_phi[0], expected, 1e-10 * std::abs(expected));
  // The tidal numbers vanish at degree 0: a constant potential is a gauge.
  EXPECT_NEAR(ln.k_t[0], 0.0, 1e-14);
  EXPECT_NEAR(ln.h_t[0], 0.0, 1e-20);
}

TEST(DegreeOperator, ReciprocityAndSigns) {
  auto material = CoreMantleMaterial(16);
  auto ln = ComputeLoveNumbers(material, 16);
  // The bilinear form is symmetric; the identities should hold to
  // round-off (pyslfp sees ~1e-14).
  EXPECT_LT(ln.ReciprocityResidual(), 1e-11);
  // A surface load depresses the surface and lowers the potential there.
  for (int l : {2, 8, 16}) {
    EXPECT_LT(ln.h_u[l] + ln.h_phi[l], 0.0) << "l = " << l;
    EXPECT_LT(ln.k_u[l] + ln.k_phi[l], 0.0) << "l = " << l;
  }
  // The axial pair: m_phi vanishes with k_c (the shell theorem).
  EXPECT_NEAR(ln.k_c, 0.0, 1e-12);
}

TEST(ToroidalOperator, HomogeneousSphereClosedForm) {
  // In a uniform solid sphere the static toroidal solution regular at the
  // centre is W = c r^l, and a unit generalised surface force gives
  // W(a) = 1 / (k^2 mu (l - 1) a) through the natural boundary term.
  auto material = HomogeneousMaterial(8);
  const auto [kappa, mu] = material.Moduli().KappaMu();
  const auto& mesh = material.Mesh();
  const auto a = material.Radius();
  for (int l : {2, 4}) {
    ToroidalOperator toroidal(material, l);
    auto b = std::vector<double>(toroidal.DofCount(), 0.0);
    b[toroidal.SurfaceDof()] = 1.0;
    const auto x = toroidal.Solve(b);
    const auto k2 = l * (l + 1.0);
    const auto expected = 1.0 / (k2 * mu[0] * (l - 1.0) * a);
    EXPECT_NEAR(x[toroidal.SurfaceDof()], expected, 1e-6 * expected)
        << "l = " << l;
    // ... and the radial shape is r^l through the upper elements.
    const auto nodal = toroidal.Expand(x);
    const auto n = mesh.NGll();
    const auto e = mesh.ElementCount() - 2;
    for (auto i = 0; i < n; ++i) {
      const auto r = mesh.Radius(e, i);
      const auto ratio = nodal[e * n + i] / std::pow(r / a, l);
      EXPECT_NEAR(ratio, x[toroidal.SurfaceDof()],
                  1e-5 * std::abs(x[toroidal.SurfaceDof()]))
          << "l = " << l << ", node " << i;
    }
  }
}

TEST(ToroidalOperator, FluidDecouplingAndDegreeGuard) {
  auto material = CoreMantleMaterial(8);
  const auto& mesh = material.Mesh();
  ToroidalOperator toroidal(material, 2);

  // No toroidal dofs inside the fluid outer core.
  auto fluidInteriorNode = -1;
  for (auto e = 1; e + 1 < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e - 1) && material.IsFluid(e) &&
        material.IsFluid(e + 1)) {
      fluidInteriorNode = mesh.GlobalIndex(e, 1);
      break;
    }
  }
  ASSERT_GE(fluidInteriorNode, 0);
  EXPECT_EQ(toroidal.Dof(fluidInteriorNode), -1);

  // A surface force leaves the decoupled inner core at rest.
  auto b = std::vector<double>(toroidal.DofCount(), 0.0);
  b[toroidal.SurfaceDof()] = 1.0;
  const auto nodal = toroidal.Expand(toroidal.Solve(b));
  const auto n = mesh.NGll();
  double inner = 0.0, surface = std::abs(nodal.back());
  for (auto e = 0; e < mesh.ElementCount() && !material.IsFluid(e); ++e) {
    for (auto i = 0; i < n; ++i) {
      inner = std::max(inner, std::abs(nodal[e * n + i]));
    }
  }
  EXPECT_GT(surface, 0.0);
  EXPECT_LT(inner, 1e-12 * surface);

  // Degree 0 has no toroidal problem and is refused.
  EXPECT_THROW(ToroidalOperator(material, 0), std::invalid_argument);
}

TEST(ToroidalOperator, DegreeOneRigidRotationsArePinned) {
  // At l = 1 rigid rotation (W = r) carries no strain: one null vector per
  // solid region (inner core, mantle). The gauge pins one dof per region;
  // for a right-hand side in the operator's range (torque-free) the pinned
  // solve satisfies every equation, pinned rows included, and the pin
  // changes the solution only by rigid rotations (docs/design.md §8).
  auto material = CoreMantleMaterial(8);
  ToroidalOperator top(material, 1);
  ToroidalOperator deeper(material, 1, {.pin_fraction = 0.5});
  ASSERT_EQ(top.RegionCount(), 2);
  EXPECT_EQ(top.PinnedDofs().size(), 2u);
  const auto n = top.DofCount();
  auto maxAbs = [](const std::vector<double>& v) {
    auto m = 0.0;
    for (auto x : v) m = std::max(m, std::abs(x));
    return m;
  };

  // The rigid rotations span the null space of the full operator.
  auto y = std::vector<double>(n);
  for (auto d = 0; d < n; ++d) {
    y[d] = std::sin(1.3 * d + 0.4) + 0.3 * std::cos(2.9 * d);
  }
  const auto b = top.Apply(y);
  const auto bScale = maxAbs(b) / maxAbs(y);
  for (auto k = 0; k < top.RegionCount(); ++k) {
    const auto rigid = top.RigidRotation(k);
    EXPECT_LT(maxAbs(top.Apply(rigid)), 1e-11 * bScale * maxAbs(rigid))
        << "region " << k;
  }

  // b = A y is torque-free: the pinned solves solve A x = b in full ...
  for (const auto* op : {&top, &deeper}) {
    const auto x = op->Solve(b);
    const auto residual = op->Apply(x);
    auto worst = 0.0;
    for (auto d = 0; d < n; ++d) {
      worst = std::max(worst, std::abs(residual[d] - b[d]));
    }
    EXPECT_LT(worst, 1e-9 * maxAbs(b));
    // ... and differ from y only by a rigid rotation of each region.
    for (auto k = 0; k < op->RegionCount(); ++k) {
      const auto rigid = op->RigidRotation(k);
      auto ratioMin = 1e300, ratioMax = -1e300;
      for (auto d = 0; d < n; ++d) {
        if (rigid[d] <= 0.0) continue;
        const auto ratio = (x[d] - y[d]) / rigid[d];
        ratioMin = std::min(ratioMin, ratio);
        ratioMax = std::max(ratioMax, ratio);
      }
      EXPECT_LT(ratioMax - ratioMin, 1e-8 * maxAbs(y) / maxAbs(rigid))
          << "region " << k;
    }
  }

  // A pin fraction outside [0, 1) is refused (1 would be a region's
  // bottom node, the centre for the inner core, where the rigid mode
  // vanishes).
  EXPECT_THROW(ToroidalOperator(material, 1, {.pin_fraction = 1.0}),
               std::invalid_argument);
  EXPECT_THROW(ToroidalOperator(material, 1, {.pin_fraction = -0.1}),
               std::invalid_argument);
}
