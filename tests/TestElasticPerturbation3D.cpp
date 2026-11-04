// The lateral (Born) elastic perturbation: angle-uniform directions in mu
// and kappa reproduce the exact 1D Love-number derivatives through the
// grid pipeline; the perturbation force is symmetric (its pairing with two
// solutions commutes); a single-harmonic direction couples a degree-l
// order-m load only to the degrees and orders the selection rules allow,
// spheroidal and toroidal. The full lateral solve (preconditioned
// conjugate gradients on the reference systems): residual at tolerance,
// Born to first order, exact for angle-uniform shifts (loads and tides),
// its adjoint's kernels against finite differences, and the iteration
// count against the contrast.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
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


TEST(ElasticPerturbation3D, LateralSolveIsBornToFirstOrder) {
  // The full solve of (K_ref + dK) x = f with the reference solve as the
  // preconditioner: the converged solution's full (unpreconditioned)
  // residual is at the tolerance, and its departure from the reference
  // plus Born response is second order in the contrast (halving the
  // contrast quarters it), spheroidal and toroidal alike.
  const auto material = MakeMaterial();
  ElasticPerturbation3D born(material, kLmax);
  const auto& coupling = born.Coupling();
  const auto& grid = coupling.Grid();
  const auto mean = [&] {
    auto s = 0.0;
    for (auto v : material.Moduli().L) s = std::max(s, v);
    return s;
  }();
  auto load = grid.ZeroCoefficients();
  load[SurfaceGrid::Index(2, 0)] = 1.0;
  load[SurfaceGrid::Index(3, 1)] = {0.5, 0.25};
  const auto dMu1 = Lateral(
      coupling, [&](double r) { return 0.1 * mean * (0.5 + r * r); },
      [](double theta, double phi) {
        return std::sin(theta) * std::cos(theta) * std::cos(phi) +
               0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta);
      });
  const auto dKappa1 = Lateral(
      coupling, [&](double r) { return 0.1 * mean * (1.0 - 0.5 * r); },
      [](double theta, double phi) { return std::cos(theta); });
  const auto x0 = born.SolveLoad(load);
  auto scale = 0.0;
  for (const auto& [l, v] : x0.spheroidal) {
    for (auto c : v) scale = std::max(scale, std::abs(c));
  }
  auto departure = [&](double factor) {
    auto dMu = dMu1, dKappa = dKappa1;
    for (auto& v : dMu) v *= factor;
    for (auto& v : dKappa) v *= factor;
    const auto dx = born.Respond(x0, dMu, dKappa);
    const auto [x, info] = born.SolveLateral(load, dMu, dKappa, {.rtol = 1e-13});
    EXPECT_TRUE(info.converged) << factor;
    EXPECT_FALSE(info.indefinite) << factor;
    EXPECT_LT(info.iterations, 40) << factor;
    // the full residual, unpreconditioned
    const auto f = born.LoadForces(load);
    auto [as, at] = born.Apply(x, dMu, dKappa);
    auto residual = 0.0, force = 0.0;
    for (const auto& [l, v] : as) {
      const auto& fl = f.at(l);
      for (auto i = std::size_t{0}; i < v.size(); ++i) {
        residual = std::max(residual, std::abs(v[i] - fl[i]));
        force = std::max(force, std::abs(fl[i]));
      }
    }
    for (const auto& [l, v] : at) {
      for (auto c : v) residual = std::max(residual, std::abs(c));
    }
    EXPECT_LT(residual, 1e-9 * force) << factor;
    auto spheroidal = 0.0, toroidal = 0.0, first = 0.0;
    for (const auto& [l, v] : x.spheroidal) {
      const auto& a = x0.spheroidal.at(l);
      const auto& b = dx.spheroidal.at(l);
      for (auto i = std::size_t{0}; i < v.size(); ++i) {
        spheroidal = std::max(spheroidal, std::abs(v[i] - a[i] - b[i]));
        first = std::max(first, std::abs(b[i]));
      }
    }
    for (const auto& [l, v] : x.toroidal) {
      const auto& b = dx.toroidal.at(l);
      for (auto i = std::size_t{0}; i < v.size(); ++i) {
        toroidal = std::max(toroidal, std::abs(v[i] - b[i]));
        first = std::max(first, std::abs(b[i]));
      }
    }
    std::cout << "contrast factor " << factor << ": " << info.iterations
              << " iterations, first order " << first / scale
              << ", second-order departure " << spheroidal / scale << " (S) "
              << toroidal / scale << " (T)\n";
    EXPECT_LT(spheroidal, 0.2 * first) << factor;
    return std::pair{spheroidal, toroidal};
  };
  const auto [s1, t1] = departure(1.0);
  const auto [s2, t2] = departure(0.5);
  EXPECT_GT(s1, 1e-6 * scale);
  EXPECT_GT(t1, 1e-6 * scale);
  EXPECT_NEAR(s1 / s2, 4.0, 0.3);
  EXPECT_NEAR(t1 / t2, 4.0, 0.3);
}

TEST(ElasticPerturbation3D, AngleUniformSolveMatchesTheShiftedMaterial) {
  // Angle-uniform delta mu(r), delta kappa(r) of 10% on an isotropic
  // reference (the Voigt average of PREM; a shift of the isotropic moduli
  // is then the whole of the shifted material): the converged lateral
  // solve equals the exact 1D solve on the shifted material, degree by
  // degree, at every order of the load.
  const auto anisotropic = MakeMaterial();
  const auto [k0, m0] = anisotropic.Moduli().KappaMu();
  const auto material =
      anisotropic.WithModuli(NodalModuli::Isotropic(k0, m0)).value();
  ElasticPerturbation3D born(material, kLmax);
  const auto& coupling = born.Coupling();
  const auto& grid = coupling.Grid();
  const auto d = SmoothDirection(material, true, true);
  auto kappa = k0, mu = m0;
  for (auto i = std::size_t{0}; i < kappa.size(); ++i) {
    kappa[i] += d.kappa[i];
    mu[i] += d.mu[i];
  }
  const auto shifted =
      material.WithModuli(NodalModuli::Isotropic(kappa, mu)).value();
  ElasticPerturbation3D exact(shifted, kLmax);
  auto load = grid.ZeroCoefficients();
  for (auto l = 0; l <= kLmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      load[SurfaceGrid::Index(l, m)] = {1.0 / (1.0 + l), m == 0 ? 0.0 : 0.3 * m};
    }
  }
  const auto [x, info] = born.SolveLateral(
      load, Uniform(coupling, material, d.mu), Uniform(coupling, material, d.kappa),
      {.rtol = 1e-13});
  EXPECT_TRUE(info.converged);
  EXPECT_FALSE(info.indefinite);
  const auto y = exact.SolveLoad(load);
  for (const auto component : {Component::U, Component::V, Component::Phi}) {
    const auto a = born.Surface(x, component);
    const auto b = exact.Surface(y, component);
    auto scale = 0.0;
    for (const auto& [l, v] : b) {
      for (auto c : v) scale = std::max(scale, std::abs(c));
    }
    for (const auto& [l, v] : a) {
      const auto& w = b.at(l);
      for (auto m = 0; m <= l; ++m) {
        EXPECT_NEAR(std::abs(v[m] - w[m]), 0.0, 1e-9 * scale)
            << "component " << static_cast<int>(component) << " degree " << l
            << " order " << m;
      }
    }
  }
  EXPECT_LT(MaxAbs(born.SurfaceToroidal(x)), 1e-12);
  std::cout << "angle-uniform solve: " << info.iterations << " iterations\n";
}

TEST(ElasticPerturbation3D, ConvergenceAgainstContrast) {
  // Iterations to 1e-10 against the lateral contrast: a degree-2 shear
  // pattern of amplitude 5, 10, 20 and 40 percent of the shear modulus
  // (below the magnitude that would make the moduli locally negative).
  const auto material = MakeMaterial();
  ElasticPerturbation3D born(material, kLmax);
  const auto& coupling = born.Coupling();
  const auto& grid = coupling.Grid();
  const auto [k0, m0] = material.Moduli().KappaMu();
  // the local shear modulus on the grid, for a relative perturbation
  auto local = Uniform(coupling, material, m0);
  auto load = grid.ZeroCoefficients();
  for (auto l = 0; l <= kLmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      load[SurfaceGrid::Index(l, m)] = {1.0 / (1.0 + l), m == 0 ? 0.0 : 0.3 * m};
    }
  }
  auto pattern = Lateral(coupling, [](double) { return 1.0; },
                         [](double theta, double phi) {
                           return std::sin(theta) * std::sin(theta) * std::cos(2 * phi) +
                                  0.5 * (3 * std::cos(theta) * std::cos(theta) - 1);
                         });
  auto previous = 0;
  for (const auto contrast : {0.05, 0.1, 0.2, 0.4}) {
    auto dMu = pattern;
    for (auto i = std::size_t{0}; i < dMu.size(); ++i) dMu[i] *= contrast * local[i];
    const auto [x, info] = born.SolveLateral(load, dMu, {}, {.rtol = 1e-10});
    EXPECT_TRUE(info.converged) << contrast;
    EXPECT_FALSE(info.indefinite) << contrast;
    EXPECT_GE(info.iterations, previous) << contrast;
    previous = info.iterations;
    std::cout << "contrast " << contrast << ": " << info.iterations
              << " iterations, residuals";
    for (auto r : info.residuals) std::cout << ' ' << r;
    std::cout << '\n';
  }
}

TEST(ElasticPerturbation3D, TideUnderAngleUniformShiftMatchesTheShiftedMaterial) {
  // Body tides: a tidal potential of every degree and order, solved under
  // an angle-uniform 10% shift of an isotropic reference, equals the
  // shifted material's tidal response (the 1D tide solve) at the
  // surface, and departs from the reference's.
  const auto anisotropic = MakeMaterial();
  const auto [k0, m0] = anisotropic.Moduli().KappaMu();
  const auto material =
      anisotropic.WithModuli(NodalModuli::Isotropic(k0, m0)).value();
  ElasticPerturbation3D born(material, kLmax);
  const auto& coupling = born.Coupling();
  const auto& grid = coupling.Grid();
  const auto d = SmoothDirection(material, true, true);
  auto kappa = k0, mu = m0;
  for (auto i = std::size_t{0}; i < kappa.size(); ++i) {
    kappa[i] += d.kappa[i];
    mu[i] += d.mu[i];
  }
  ElasticPerturbation3D exact(
      material.WithModuli(NodalModuli::Isotropic(kappa, mu)).value(), kLmax);
  auto tide = grid.ZeroCoefficients();
  for (auto l = 2; l <= kLmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      tide[SurfaceGrid::Index(l, m)] = {1.0 / l, m == 0 ? 0.0 : 0.3 * m};
    }
  }
  const auto [x, info] = born.SolveLateral(
      born.TideForces(tide), {}, Uniform(coupling, material, d.mu),
      Uniform(coupling, material, d.kappa), {.rtol = 1e-13});
  EXPECT_TRUE(info.converged);
  const auto y = exact.Solve(exact.TideForces(tide), {}, false);
  const auto y0 = born.Solve(born.TideForces(tide), {}, false);
  for (const auto component : {Component::U, Component::V, Component::Phi}) {
    const auto a = born.Surface(x, component);
    const auto b = exact.Surface(y, component);
    const auto c = born.Surface(y0, component);
    auto scale = 0.0, departure = 0.0;
    for (const auto& [l, v] : b) {
      for (auto m = 0; m <= l; ++m) {
        scale = std::max(scale, std::abs(v[m]));
        departure = std::max(departure, std::abs(v[m] - c.at(l)[m]));
      }
    }
    EXPECT_GT(departure, 1e-3 * scale) << static_cast<int>(component);
    for (const auto& [l, v] : a) {
      const auto& w = b.at(l);
      for (auto m = 0; m <= l; ++m) {
        EXPECT_NEAR(std::abs(v[m] - w[m]), 0.0, 1e-9 * scale)
            << "component " << static_cast<int>(component) << " degree " << l
            << " order " << m;
      }
    }
  }
  EXPECT_LT(MaxAbs(born.SurfaceToroidal(x)), 1e-12);
}

TEST(ElasticPerturbation3D, KernelsUnderLateralStructureMatchFiniteDifferences) {
  // The adjoint of the laterally heterogeneous static problem: with y the
  // lateral solve of a surface measurement's dual forces and x that of
  // the load, both under the background delta mu_0(x), delta kappa_0(x),
  // -StaticKernels(y, x) paired with a direction (v_mu, v_kappa) equals
  // the central difference of the measurement along the direction,
  // through the full solve.
  const auto material = MakeMaterial();
  ElasticPerturbation3D born(material, kLmax);
  const auto& coupling = born.Coupling();
  const auto& grid = coupling.Grid();
  std::mt19937 rng(11);
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
  const auto load = randomCoefficients();
  const auto dual = randomCoefficients();
  const auto top = [&] {
    auto s = 0.0;
    for (auto v : material.Moduli().L) s = std::max(s, v);
    return s;
  }();
  const auto mu0 = Lateral(
      coupling, [&](double r) { return 0.1 * top * (0.5 + r * r); },
      [](double theta, double phi) {
        return std::sin(theta) * std::cos(theta) * std::cos(phi) +
               0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta);
      });
  const auto kappa0 = Lateral(
      coupling, [&](double r) { return 0.1 * top * (1.0 - 0.5 * r); },
      [](double theta, double phi) { return std::cos(theta); });
  const auto vMu = Lateral(
      coupling, [&](double r) { return top * (1.0 - r * r); },
      [](double theta, double phi) { return std::sin(theta) * std::sin(phi) + 0.2; });
  const auto vKappa = Lateral(
      coupling, [&](double r) { return top * r; },
      [](double theta, double phi) { return std::cos(2 * theta) * std::cos(phi); });
  const auto options = ElasticPerturbation3D::SolveOptions{.rtol = 1e-13};
  auto measure = [&](double eps) {
    auto dMu = mu0, dKappa = kappa0;
    for (auto i = std::size_t{0}; i < dMu.size(); ++i) {
      dMu[i] += eps * vMu[i];
      dKappa[i] += eps * vKappa[i];
    }
    const auto [x, info] = born.SolveLateral(load, dMu, dKappa, options);
    EXPECT_TRUE(info.converged);
    const auto surface = born.Surface(x, Component::Phi);
    auto value = 0.0;
    for (auto l = 0; l <= kLmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        value += (m == 0 ? 1.0 : 2.0) *
                 (surface.at(l)[m] * std::conj(dual[SurfaceGrid::Index(l, m)])).real();
      }
    }
    return value;
  };
  const auto [x, infoX] = born.SolveLateral(load, mu0, kappa0, options);
  const auto [y, infoY] = born.SolveLateral(
      born.SurfaceDualForces(dual, Component::Phi), {}, mu0, kappa0, options);
  EXPECT_TRUE(infoX.converged);
  EXPECT_TRUE(infoY.converged);
  const auto kernels = born.StaticKernels(y, x);
  auto pairing = [&](std::span<const double> kernel, std::span<const double> v) {
    auto total = 0.0;
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto at = coupling.FieldOffset(rn);
      const auto size = coupling.FieldSizeOf(rn);
      auto product = std::vector<double>(size);
      for (auto i = std::size_t{0}; i < size; ++i) product[i] = kernel[at + i] * v[at + i];
      total += coupling.GridOf(rn).Integrate(product);
    }
    return total;
  };
  const auto predicted = -(pairing(kernels.mu, vMu) + pairing(kernels.kappa, vKappa));
  const auto eps = 1e-3;
  const auto difference = (measure(eps) - measure(-eps)) / (2 * eps);
  std::cout << "kernel pairing " << predicted << ", finite difference " << difference << '\n';
  EXPECT_NEAR(predicted, difference, 1e-5 * std::abs(difference));
  EXPECT_NE(difference, 0.0);
}

TEST(ElasticPerturbation3D, DISABLED_TimingAtHigherDegree) {
  // Not a gate: the cost of the lateral solve at a benchmark-sized lmax
  // (SL3DPP_LMAX, default 32) for a 10% degree-2 shear pattern and a
  // cap-like load; run with --gtest_also_run_disabled_tests.
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
  const auto start = std::chrono::steady_clock::now();
  ElasticPerturbation3D born(material, lmax);
  const auto& coupling = born.Coupling();
  const auto& grid = coupling.Grid();
  const auto [k0, m0] = material.Moduli().KappaMu();
  auto local = Uniform(coupling, material, m0);
  auto dMu = Lateral(coupling, [](double) { return 0.1; },
                     [](double theta, double phi) {
                       return std::sin(theta) * std::sin(theta) * std::cos(2 * phi);
                     });
  for (auto i = std::size_t{0}; i < dMu.size(); ++i) dMu[i] *= local[i];
  auto load = grid.ZeroCoefficients();
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= std::min(l, 4); ++m) {
      load[SurfaceGrid::Index(l, m)] = std::exp(-0.002 * l * l);
    }
  }
  const auto setup = std::chrono::steady_clock::now();
  const auto x0 = born.SolveLoad(load);
  const auto reference = std::chrono::steady_clock::now();
  const auto [x, info] = born.SolveLateral(load, dMu, {}, {.rtol = 1e-10});
  const auto end = std::chrono::steady_clock::now();
  using Seconds = std::chrono::duration<double>;
  std::cout << "lmax " << lmax << ": setup "
            << Seconds(setup - start).count() << " s, reference solve "
            << Seconds(reference - setup).count() << " s, lateral solve "
            << Seconds(end - reference).count() << " s in " << info.iterations
            << " iterations (" << coupling.NodeCount() << " nodes, "
            << coupling.GridValueCount() << " grid values)\n";
  EXPECT_TRUE(info.converged);
}
