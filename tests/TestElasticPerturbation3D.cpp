// The lateral (Born) elastic perturbation: angle-uniform directions in mu
// and kappa reproduce the exact 1D Love-number derivatives through the
// grid pipeline; the perturbation force is symmetric (its pairing with two
// solutions commutes); a single-harmonic direction couples a degree-l
// order-m load only to the degrees and orders the selection rules allow,
// spheroidal and toroidal.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <numbers>
#include <random>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 6;
using Complex = std::complex<double>;

Material MakeMaterial() {
  auto deck =
      Deck::Read(
          std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
          DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  auto model = EarthModel::FromDeck(deck).value();
  auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
  return Material::Create(model, mesh).value();
}

/// A nodal field over the material's mesh carried to the coupling's grid
/// values, angle-uniform.
std::vector<double> Uniform(const GridCoupling& coupling, const Material& material,
                            std::span<const double> nodal) {
  const auto ngll = material.Mesh().NGll();
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full = static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
                      coupling.NodeInElementOf(rn);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < coupling.FieldSizeOf(rn); ++x) {
      out[at + x] = nodal[full];
    }
  }
  return out;
}

/// A radial profile times an angular function, over the grid values.
std::vector<double> Lateral(const GridCoupling& coupling,
                            const std::function<double(double)>& radial,
                            const std::function<double(double, double)>& angular) {
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto r = coupling.RadiusOfNode(rn);
    const auto values = coupling.GridOf(rn).FromFunction(
        [&](double theta, double phi) { return radial(r) * angular(theta, phi); });
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x) out[at + x] = values[x];
  }
  return out;
}

/// The smooth directions of the 1D derivative tests, as nodal fields.
MaterialDirection SmoothDirection(const Material& material, bool mu, bool kappa) {
  const auto& mesh = material.Mesh();
  const auto n = mesh.NGll();
  const auto size = material.Rho().size();
  const auto [k0, m0] = material.Moduli().KappaMu();
  MaterialDirection d;
  if (mu) d.mu.assign(size, 0.0);
  if (kappa) d.kappa.assign(size, 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < n; ++q) {
      const auto i = static_cast<std::size_t>(e) * n + q;
      const auto r = mesh.Radius(e, q);
      const auto profile = 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
      // not on the element holding the centre, which the coupling leaves
      // out (ElasticPerturbation3D): degree 0 would see the difference
      if (e == 0 && mesh.Left(0) == 0.0) continue;
      if (mu && !material.IsFluid(e)) d.mu[i] = profile * m0[i];
      if (kappa && !material.IsFluid(e)) d.kappa[i] = profile * k0[i];
    }
  }
  return d;
}

double MaxAbs(const std::map<int, std::vector<Complex>>& m) {
  auto out = 0.0;
  for (const auto& [l, v] : m) {
    for (auto c : v) out = std::max(out, std::abs(c));
  }
  return out;
}

}  // namespace

TEST(ElasticPerturbation3D, AngleUniformMatchesThe1DDerivative) {
  // Angle-uniform delta mu(r) and delta kappa(r) on the solid: the Born
  // response of a unit load at every degree and order zero, read at the
  // surface, equals the 1D Love-number derivatives (the exact
  // linearisation of the material) for U, V and phi, degree by degree.
  const auto material = MakeMaterial();
  ElasticPerturbation3D born(material, kLmax);
  const auto& coupling = born.Coupling();
  const auto& grid = coupling.Grid();
  auto load = grid.ZeroCoefficients();
  for (auto l = 0; l <= kLmax; ++l) load[SurfaceGrid::Index(l, 0)] = 1.0;
  const auto x = born.SolveLoad(load);
  const auto& scales = material.ModelScales();
  for (const bool mu : {true, false}) {
    const auto d = SmoothDirection(material, mu, !mu);
    const auto dlove =
        ComputeLoveNumberDerivatives(material, kLmax, d).ConvertedTo(scales);
    const auto field = Uniform(coupling, material, mu ? d.mu : d.kappa);
    const auto dx = born.Respond(x, mu ? field : std::span<const double>{},
                                 mu ? std::span<const double>{} : field);
    const auto U = born.Surface(dx, Component::U);
    const auto V = born.Surface(dx, Component::V);
    const auto P = born.Surface(dx, Component::Phi);
    auto scale = 0.0;
    for (auto l = 0; l <= kLmax; ++l) {
      scale = std::max({scale, std::abs(dlove.h_u[l] + dlove.h_phi[l]),
                        std::abs(dlove.l_u[l] + dlove.l_phi[l])});
    }
    for (auto l = 0; l <= kLmax; ++l) {
      EXPECT_NEAR(U.at(l)[0].real(), dlove.h_u[l] + dlove.h_phi[l], 1e-8 * scale)
          << (mu ? "mu" : "kappa") << " h at degree " << l;
      EXPECT_NEAR(V.at(l)[0].real(), dlove.l_u[l] + dlove.l_phi[l], 1e-8 * scale)
          << (mu ? "mu" : "kappa") << " l at degree " << l;
      EXPECT_NEAR(P.at(l)[0].real(), dlove.k_u[l] + dlove.k_phi[l],
                  1e-8 * std::abs(dlove.k_u[l] + dlove.k_phi[l]) + 1e-8 * scale)
          << (mu ? "mu" : "kappa") << " k at degree " << l;
      // angle-uniform: no coupling into the other orders, nor toroidal
      for (auto m = 1; m <= l; ++m) {
        EXPECT_LT(std::abs(U.at(l)[m]), 1e-12 * scale) << "degree " << l;
      }
    }
    EXPECT_LT(MaxAbs(born.SurfaceToroidal(dx)), 1e-12 * scale);
  }
}

TEST(ElasticPerturbation3D, ForceIsSymmetric) {
  // (dK) is symmetric: <x_b, (dK) x_a> = <x_a, (dK) x_b> under the
  // Hermitian pairing, for lateral directions in mu and kappa and random
  // real loads.
  const auto material = MakeMaterial();
  ElasticPerturbation3D born(material, kLmax);
  const auto& coupling = born.Coupling();
  const auto& grid = coupling.Grid();
  std::mt19937 rng(5);
  std::normal_distribution<double> normal;
  auto randomLoad = [&] {
    auto c = grid.ZeroCoefficients();
    for (auto l = 0; l <= kLmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        c[SurfaceGrid::Index(l, m)] = {normal(rng), m == 0 ? 0.0 : normal(rng)};
      }
    }
    return c;
  };
  const auto xa = born.SolveLoad(randomLoad());
  const auto xb = born.SolveLoad(randomLoad());
  const auto mean = [&] {
    auto s = 0.0;
    for (auto v : material.Moduli().L) s = std::max(s, v);
    return s;
  }();
  const auto dMu = Lateral(
      coupling, [&](double r) { return 0.1 * mean * (0.5 + r * r); },
      [](double theta, double phi) {
        return std::sin(theta) * std::cos(theta) * std::cos(phi) + 0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta);
      });
  const auto dKappa = Lateral(
      coupling, [&](double r) { return 0.2 * mean * (1.0 - 0.5 * r); },
      [](double theta, double phi) { return std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi); });
  for (const auto& [name, mu, kappa] :
       {std::tuple{"mu", std::span<const double>(dMu), std::span<const double>{}},
        std::tuple{"kappa", std::span<const double>{}, std::span<const double>(dKappa)},
        std::tuple{"both", std::span<const double>(dMu), std::span<const double>(dKappa)}}) {
    const auto [sa, ta] = born.Force(xa, mu, kappa);
    const auto [sb, tb] = born.Force(xb, mu, kappa);
    const auto lhs = ElasticPerturbation3D::Pair(xb, sa, ta);
    const auto rhs = ElasticPerturbation3D::Pair(xa, sb, tb);
    EXPECT_NEAR(lhs, rhs, 1e-10 * std::abs(lhs)) << name;
    EXPECT_NE(lhs, 0.0) << name;
  }
}

TEST(ElasticPerturbation3D, SelectionRules) {
  // A direction f(r) Y_{L,M} (real: P_L^M(cos theta) cos(M phi)) couples
  // a degree-l, order-m load only to degrees |l - L| .. l + L and orders
  // m +/- M; every other coefficient of the response vanishes, spheroidal
  // and toroidal alike, and the toroidal response is not zero.
  const auto material = MakeMaterial();
  ElasticPerturbation3D born(material, kLmax);
  const auto& coupling = born.Coupling();
  const auto& grid = coupling.Grid();
  constexpr auto L = 2, M = 1, l0 = 3, m0 = 0;
  auto load = grid.ZeroCoefficients();
  load[SurfaceGrid::Index(l0, m0)] = 1.0;
  const auto x = born.SolveLoad(load);
  const auto dMu = Lateral(
      coupling, [](double r) { return 0.3 * r; },
      [](double theta, double phi) {
        // P_2^1 ~ sin(theta) cos(theta)
        return std::sin(theta) * std::cos(theta) * std::cos(M * phi);
      });
  const auto dx = born.Respond(x, dMu, {});
  const auto U = born.Surface(dx, Component::U);
  const auto W = born.SurfaceToroidal(dx);
  const auto scale = std::max(MaxAbs(U), MaxAbs(W));
  ASSERT_GT(scale, 0.0);
  EXPECT_GT(MaxAbs(W), 1e-6 * scale) << "a lateral shear couples toroidal motion";
  for (const auto* field : {&U, &W}) {
    for (const auto& [l, v] : *field) {
      for (auto m = 0; m <= l; ++m) {
        const auto allowed = l >= std::abs(l0 - L) && l <= l0 + L &&
                             (m == std::abs(m0 + M) || m == std::abs(m0 - M));
        if (!allowed) {
          EXPECT_LT(std::abs(v[m]), 1e-11 * scale) << "degree " << l << " order " << m;
        }
      }
    }
  }
}

TEST(ElasticPerturbation3D, StaticKernelsAreTheTransposeOfTheForce) {
  // The static material kernels of a solution pair: their grid pairing
  // with lateral directions in kappa and mu equals the Hermitian pairing
  // of the dual solution with the force of those directions on the
  // other — the transpose of Force; with the dual the response to a
  // surface dual, the kernels of a static surface measurement.
  const auto material = MakeMaterial();
  ElasticPerturbation3D born(material, kLmax);
  const auto& coupling = born.Coupling();
  const auto& grid = coupling.Grid();
  std::mt19937 rng(7);
  std::normal_distribution<double> normal;
  auto randomCoefficients = [&] {
    auto c = grid.ZeroCoefficients();
    for (auto l = 0; l <= kLmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        c[SurfaceGrid::Index(l, m)] = {normal(rng), m == 0 ? 0.0 : normal(rng)};
      }
    }
    return c;
  };
  const auto x = born.SolveLoad(randomCoefficients());
  const auto y = born.SolveSurfaceDual(randomCoefficients(), Component::U);
  const auto top = [&] {
    auto s = 0.0;
    for (auto v : material.Moduli().L) s = std::max(s, v);
    return s;
  }();
  const auto dMu = Lateral(
      coupling, [&](double r) { return 0.1 * top * (0.5 + r * r); },
      [](double theta, double phi) {
        return std::sin(theta) * std::cos(theta) * std::cos(phi) +
               0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta);
      });
  const auto dKappa = Lateral(
      coupling, [&](double r) { return 0.2 * top * (1.0 - 0.5 * r); },
      [](double theta, double phi) {
        return std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
      });
  const auto kernels = born.StaticKernels(y, x);
  auto pairing = [&](std::span<const double> kernel, std::span<const double> d) {
    auto total = 0.0;
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto at = coupling.FieldOffset(rn);
      const auto size = coupling.FieldSizeOf(rn);
      auto product = std::vector<double>(size);
      for (auto i = std::size_t{0}; i < size; ++i) product[i] = kernel[at + i] * d[at + i];
      total += coupling.GridOf(rn).Integrate(product);
    }
    return total;
  };
  const auto [sMu, tMu] = born.Force(x, dMu, {});
  const auto [sKappa, tKappa] = born.Force(x, {}, dKappa);
  const auto direct = ElasticPerturbation3D::Pair(y, sMu, tMu) +
                      ElasticPerturbation3D::Pair(y, sKappa, tKappa);
  const auto paired = pairing(kernels.mu, dMu) + pairing(kernels.kappa, dKappa);
  EXPECT_NEAR(paired, direct, 1e-10 * std::abs(direct));
  EXPECT_NE(direct, 0.0);
  // the dual solution is the adjoint of the surface readout
  const auto dual = randomCoefficients();
  const auto z = born.SolveSurfaceDual(dual, Component::Phi);
  const auto surface = born.Surface(x, Component::Phi);
  auto readout = 0.0;
  for (auto l = 0; l <= kLmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      readout += (m == 0 ? 1.0 : 2.0) *
                 (surface.at(l)[m] * std::conj(dual[SurfaceGrid::Index(l, m)])).real();
    }
  }
  // <z, K x> = <dual, surface(x)>: pair z with the load force of x
  const auto [sx, tx] = born.Force(x, {}, {});  // empty: zero forces
  (void)sx; (void)tx;
  const auto& sph = [&](int l) -> const DegreeOperator& { return born.Spheroidal(l); };
  auto viaK = 0.0;
  for (const auto& [l, zl] : z.spheroidal) {
    const auto& op = sph(l);
    const auto ndof = op.DofCount();
    auto zr = std::vector<double>(static_cast<std::size_t>(ndof) * 2 * (l + 1));
    for (auto c = 0; c <= l; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        zr[static_cast<std::size_t>(c) * ndof + d] = zl[static_cast<std::size_t>(c) * ndof + d].real();
        zr[static_cast<std::size_t>(l + 1 + c) * ndof + d] = zl[static_cast<std::size_t>(c) * ndof + d].imag();
      }
    }
    const auto Kz = op.Apply(zr, 2 * (l + 1));
    const auto& xl = x.spheroidal.at(l);
    for (auto c = 0; c <= l; ++c) {
      const auto w = c == 0 ? 1.0 : 2.0;
      for (auto d = 0; d < ndof; ++d) {
        const auto i = static_cast<std::size_t>(c) * ndof + d;
        const Complex kz{Kz[i], Kz[static_cast<std::size_t>(l + 1 + c) * ndof + d]};
        viaK += w * (xl[i] * std::conj(kz)).real();
      }
    }
  }
  EXPECT_NEAR(viaK, readout, 1e-10 * std::abs(readout));
}

