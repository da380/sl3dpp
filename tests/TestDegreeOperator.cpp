// DegreeOperator and the Love-number driver: structural identities and
// closed forms (Kelvin's fluid number, degree-0 mass conservation),
// ahead of the pyslfp fixture gate in TestFixtures.

#include <gtest/gtest.h>

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
    for (std::size_t k = 0; k < n; ++k) {
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
  constexpr double b = 6.371e6;
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
  constexpr double b = 6.371e6;
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
  int fluidNode = -1;
  for (int e = 0; e + 1 < mesh.ElementCount(); ++e) {
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
  bool sawNaN = false;
  for (int e = 0; e < mesh.ElementCount(); ++e) {
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
    const double kelvin = 3.0 / (2.0 * (l - 1.0));
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
  const double a = material.Radius();
  for (int l : {2, 4}) {
    ToroidalOperator toroidal(material, l);
    std::vector<double> b(toroidal.DofCount(), 0.0);
    b[toroidal.SurfaceDof()] = 1.0;
    const auto x = toroidal.Solve(b);
    const double k2 = l * (l + 1.0);
    const double expected = 1.0 / (k2 * mu[0] * (l - 1.0) * a);
    EXPECT_NEAR(x[toroidal.SurfaceDof()], expected, 1e-6 * expected)
        << "l = " << l;
    // ... and the radial shape is r^l through the upper elements.
    const auto nodal = toroidal.Expand(x);
    const int n = mesh.NGll();
    const int e = mesh.ElementCount() - 2;
    for (int i = 0; i < n; ++i) {
      const double r = mesh.Radius(e, i);
      const double ratio = nodal[e * n + i] / std::pow(r / a, l);
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
  int fluidInteriorNode = -1;
  for (int e = 1; e + 1 < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e - 1) && material.IsFluid(e) &&
        material.IsFluid(e + 1)) {
      fluidInteriorNode = mesh.GlobalIndex(e, 1);
      break;
    }
  }
  ASSERT_GE(fluidInteriorNode, 0);
  EXPECT_EQ(toroidal.Dof(fluidInteriorNode), -1);

  // A surface force leaves the decoupled inner core at rest.
  std::vector<double> b(toroidal.DofCount(), 0.0);
  b[toroidal.SurfaceDof()] = 1.0;
  const auto nodal = toroidal.Expand(toroidal.Solve(b));
  const int n = mesh.NGll();
  double inner = 0.0, surface = std::abs(nodal.back());
  for (int e = 0; e < mesh.ElementCount() && !material.IsFluid(e); ++e) {
    for (int i = 0; i < n; ++i) {
      inner = std::max(inner, std::abs(nodal[e * n + i]));
    }
  }
  EXPECT_GT(surface, 0.0);
  EXPECT_LT(inner, 1e-12 * surface);

  // Degrees without a toroidal problem are refused.
  EXPECT_THROW(ToroidalOperator(material, 1), std::invalid_argument);
  EXPECT_THROW(ToroidalOperator(material, 0), std::invalid_argument);
}
