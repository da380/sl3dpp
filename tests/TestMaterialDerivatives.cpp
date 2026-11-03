// The material derivatives of the static gravitating elastic problem:
// the variation operator is the derivative of the assembled system along
// a direction (central differences of K(m +/- eps d) x at second order,
// Richardson to rounding), the Love-number derivatives match central
// differences of the Love numbers of the shifted material for directions
// in mu, kappa and rho separately and together (Richardson to 1e-7),
// the derivative respects the reciprocity identities differentiated, and
// the shifted material's gravity is the quadrature integral of the
// density direction.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <numbers>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 6;

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

/// A smooth direction: each field a fraction of the material's own, times
/// a smooth profile in radius, on the solid elements only for mu and
/// everywhere for kappa and rho.
MaterialDirection SmoothDirection(const Material& material, bool mu, bool kappa,
                                  bool rho) {
  const auto& mesh = material.Mesh();
  const auto n = mesh.NGll();
  const auto size = material.Rho().size();
  const auto [k0, m0] = material.Moduli().KappaMu();
  MaterialDirection d;
  if (mu) d.mu.assign(size, 0.0);
  if (kappa) d.kappa.assign(size, 0.0);
  if (rho) d.rho.assign(size, 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < n; ++q) {
      const auto i = static_cast<std::size_t>(e) * n + q;
      const auto r = mesh.Radius(e, q);
      const auto profile = 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
      if (mu && !material.IsFluid(e)) d.mu[i] = profile * m0[i];
      if (kappa) d.kappa[i] = profile * k0[i];
      if (rho) d.rho[i] = 0.5 * profile * material.Rho()[i];
    }
  }
  return d;
}

/// The twelve channels and the axial numbers as one vector, with names.
std::vector<std::pair<std::string, double>> Entries(const LoveNumbers& t) {
  auto out = std::vector<std::pair<std::string, double>>{};
  for (auto i = std::size_t{0}; i < t.degrees.size(); ++i) {
    const auto l = std::to_string(t.degrees[i]);
    for (const auto& [name, v] :
         {std::pair{"h_u", &t.h_u}, std::pair{"l_u", &t.l_u},
          std::pair{"k_u", &t.k_u}, std::pair{"h_phi", &t.h_phi},
          std::pair{"l_phi", &t.l_phi}, std::pair{"k_phi", &t.k_phi},
          std::pair{"h_v", &t.h_v}, std::pair{"l_v", &t.l_v},
          std::pair{"k_v", &t.k_v}, std::pair{"h_t", &t.h_t},
          std::pair{"l_t", &t.l_t}, std::pair{"k_t", &t.k_t}}) {
      out.emplace_back(std::string(name) + "[" + l + "]", (*v)[i]);
    }
  }
  out.emplace_back("h_c", t.h_c);
  out.emplace_back("k_c", t.k_c);
  out.emplace_back("m_u", t.m_u);
  out.emplace_back("m_phi", t.m_phi);
  out.emplace_back("m_c", t.m_c);
  out.emplace_back("g", t.surface_gravity);
  return out;
}

}  // namespace

TEST(MaterialDerivatives, GravityVariationIsTheIntegralOfTheDensity) {
  // The variation of gravity along a density direction is 4 pi G r^-2
  // times the integral of delta rho r^2 from the bottom of the mesh: for a
  // direction constant in radius on a mesh from the centre, 4 pi G
  // delta rho r / 3 exactly.
  const auto material = MakeMaterial();
  const auto& mesh = material.Mesh();
  MaterialDirection d;
  d.rho.assign(material.Rho().size(), 1.0);
  const auto v = material.Variation(d);
  ASSERT_EQ(mesh.Left(0), 0.0);
  for (auto i = std::size_t{0}; i < v.g.size(); ++i) {
    const auto r = mesh.Radii()[i];
    EXPECT_NEAR(v.g[i], 4.0 * std::numbers::pi * material.G() * r / 3.0,
                1e-12 * material.G() * mesh.Radii().back());
  }
  EXPECT_DOUBLE_EQ(v.surfaceGravity, v.g.back());
  for (auto i = std::size_t{0}; i < v.drho.size(); ++i) {
    EXPECT_NEAR(v.drho[i], 0.0, 1e-10);
  }
  // A quadratic direction: delta rho = r^2 gives 4 pi G r^3 / 5.
  for (auto i = std::size_t{0}; i < d.rho.size(); ++i) {
    d.rho[i] = mesh.Radii()[i] * mesh.Radii()[i];
  }
  const auto q = material.Variation(d);
  for (auto i = std::size_t{0}; i < q.g.size(); ++i) {
    const auto r = mesh.Radii()[i];
    EXPECT_NEAR(q.g[i], 4.0 * std::numbers::pi * material.G() * r * r * r / 5.0,
                1e-11 * material.G());
    EXPECT_NEAR(q.drho[i], 2.0 * r, 1e-9);
  }
  // Refusals and the shifted material's fields.
  MaterialDirection bad;
  bad.mu.assign(3, 1.0);
  EXPECT_THROW((void)material.Variation(bad), std::invalid_argument);
  const auto shifted = material.Shifted(d, 0.5);
  for (auto i = std::size_t{0}; i < q.g.size(); ++i) {
    EXPECT_DOUBLE_EQ(shifted.Rho()[i], material.Rho()[i] + 0.5 * d.rho[i]);
    EXPECT_DOUBLE_EQ(shifted.Gravity()[i], material.Gravity()[i] + 0.5 * q.g[i]);
  }
}

TEST(MaterialDerivatives, VariationOperatorIsTheDerivativeOfTheSystem) {
  // (dK) x against central differences of K(m +/- eps d) x on the shifted
  // materials, for every degree and both problems: the operator is affine
  // in the moduli and quadratic in the density, so the differences are
  // exact at second order and Richardson lands to rounding.
  const auto material = MakeMaterial();
  const auto d = SmoothDirection(material, true, true, true);
  const auto dm = material.Variation(d);
  // The gravitating problem alone: the model has a fluid core, which the
  // non-gravitating problem refuses (its path is a subset of this one).
  for (const bool gravitating : {true}) {
    for (auto l = 0; l <= kLmax; ++l) {
      DegreeOperator op(material, l, {.gravitating = gravitating});
      const auto n = op.DofCount();
      auto x = std::vector<double>(n);
      for (auto i = 0; i < n; ++i) x[i] = std::sin(0.37 * i + 0.1 * l) + 0.3;
      const auto predicted = op.ApplyVariation(dm, x);
      auto fd = [&](double eps) {
        DegreeOperator plus(material.Shifted(d, eps), l, {.gravitating = gravitating});
        DegreeOperator minus(material.Shifted(d, -eps), l,
                             {.gravitating = gravitating});
        const auto a = plus.Apply(x);
        const auto b = minus.Apply(x);
        auto out = std::vector<double>(n);
        for (auto i = 0; i < n; ++i) out[i] = (a[i] - b[i]) / (2.0 * eps);
        return out;
      };
      const auto coarse = fd(0.02), fine = fd(0.01);
      auto scale = 0.0, worst = 0.0;
      for (auto i = 0; i < n; ++i) {
        const auto richardson = (4.0 * fine[i] - coarse[i]) / 3.0;
        scale = std::max(scale, std::abs(predicted[i]));
        worst = std::max(worst, std::abs(richardson - predicted[i]));
      }
      EXPECT_GT(scale, 0.0) << "degree " << l;
      EXPECT_LT(worst, 1e-9 * scale) << "degree " << l << " gravitating "
                                     << gravitating;
    }
  }
}

TEST(MaterialDerivatives, LoveNumberDerivativesMatchDifferences) {
  // Along directions in mu, kappa and rho, alone and together, central
  // differences of the Love numbers of the shifted material converge at
  // second order to the derivatives and Richardson lands on them at 1e-7,
  // every channel and degree, the axial numbers and the surface gravity.
  const auto material = MakeMaterial();
  struct Case {
    const char* name;
    bool mu, kappa, rho;
  };
  for (const auto& c : {Case{"mu", true, false, false},
                        Case{"kappa", false, true, false},
                        Case{"rho", false, false, true},
                        Case{"all", true, true, true}}) {
    const auto d = SmoothDirection(material, c.mu, c.kappa, c.rho);
    const auto derivatives = Entries(ComputeLoveNumberDerivatives(material, kLmax, d));
    auto at = [&](double eps) {
      return Entries(ComputeLoveNumbers(material.Shifted(d, eps), kLmax));
    };
    const auto p2 = at(0.02), m2 = at(-0.02), p1 = at(0.01), m1 = at(-0.01);
    // Scales per kind of number (h, l, k, the axial m, g), since the kinds
    // differ in units by orders of magnitude.
    auto kind = [](const std::string& name) {
      return name.substr(0, name.find('_') == std::string::npos ? 1 : name.find('_'));
    };
    auto scales = std::map<std::string, double>{};
    for (const auto& [name, v] : derivatives) {
      scales[kind(name)] = std::max(scales[kind(name)], std::abs(v));
    }
    // Per entry, Richardson on the central differences lands on the
    // derivative (the identity); over all entries together, scaled per
    // kind, the residual falls at second order where it is resolved at
    // all (the moduli directions are nearly affine and sit at rounding).
    auto coarseNorm = 0.0, fineNorm = 0.0;
    for (auto i = std::size_t{0}; i < derivatives.size(); ++i) {
      const auto scale = scales[kind(derivatives[i].first)];
      if (scale == 0.0) continue;
      const auto coarse = (p2[i].second - m2[i].second) / 0.04;
      const auto fine = (p1[i].second - m1[i].second) / 0.02;
      const auto richardson = (4.0 * fine - coarse) / 3.0;
      const auto predicted = derivatives[i].second;
      EXPECT_NEAR(richardson, predicted, 1e-7 * scale)
          << c.name << " " << derivatives[i].first << " coarse " << coarse
          << " fine " << fine;
      coarseNorm += std::pow((coarse - predicted) / scale, 2);
      fineNorm += std::pow((fine - predicted) / scale, 2);
    }
    coarseNorm = std::sqrt(coarseNorm);
    fineNorm = std::sqrt(fineNorm);
    if (fineNorm > 1e-9) {
      EXPECT_GT(coarseNorm / fineNorm, 3.0) << c.name << " " << coarseNorm << " " << fineNorm;
      EXPECT_LT(coarseNorm / fineNorm, 5.0) << c.name << " " << coarseNorm << " " << fineNorm;
    }
    if (c.rho) EXPECT_GT(fineNorm, 1e-9) << c.name;
  }
}

TEST(MaterialDerivatives, ReciprocityDifferentiated) {
  // The identities g h_phi = k_u, h_v = l(l+1) l_u, k_v = g l(l+1) l_phi
  // hold at every material, so their derivatives along a direction hold
  // too: dg h_phi + g dh_phi = dk_u and so on.
  const auto material = MakeMaterial();
  const auto d = SmoothDirection(material, true, true, true);
  const auto love = ComputeLoveNumbers(material, kLmax);
  const auto dlove = ComputeLoveNumberDerivatives(material, kLmax, d);
  const auto g = love.surface_gravity;
  const auto dg = dlove.surface_gravity;
  // Relative to each channel's largest derivative over the degrees (an
  // entry that vanishes by construction, k_u at degree 0 say, is then held
  // at rounding).
  auto largest = [](const std::vector<double>& v) {
    auto out = 0.0;
    for (auto x : v) out = std::max(out, std::abs(x));
    return out;
  };
  const auto kScale = largest(dlove.k_u), hvScale = largest(dlove.h_v),
             kvScale = largest(dlove.k_v);
  for (auto i = std::size_t{0}; i < love.degrees.size(); ++i) {
    const auto l = love.degrees[i];
    const auto k2 = l * (l + 1.0);
    const auto lhs = dg * love.h_phi[i] + g * dlove.h_phi[i];
    EXPECT_NEAR(lhs, dlove.k_u[i], 1e-10 * kScale) << "degree " << l;
    EXPECT_NEAR(dlove.h_v[i], k2 * dlove.l_u[i], 1e-10 * hvScale)
        << "degree " << l;
    EXPECT_NEAR(dlove.k_v[i], k2 * (dg * love.l_phi[i] + g * dlove.l_phi[i]),
                1e-10 * kvScale)
        << "degree " << l;
  }
  // An empty direction gives zero derivatives.
  const auto none = ComputeLoveNumberDerivatives(material, kLmax, {});
  for (const auto& [name, v] : Entries(none)) EXPECT_EQ(v, 0.0) << name;
}

TEST(MaterialDerivatives, LoveNumberKernelsPairToTheDerivatives) {
  // The static kernels: for every channel and degree, the kappa and mu
  // kernels paired with a smooth direction in the moduli equal the
  // Love-number derivatives along it (step 1's exact linearisation) at
  // 1e-10 — the adjoint of the static problem against its tangent.
  const auto material = MakeMaterial();
  const auto& mesh = material.Mesh();
  const auto [k0, m0] = material.Moduli().KappaMu();
  MaterialDirection d;
  d.kappa.assign(k0.size(), 0.0);
  d.mu.assign(m0.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      const auto r = mesh.Radius(e, q);
      const auto profile = 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
      d.kappa[i] = 0.5 * profile * k0[i];
      if (!material.IsFluid(e)) d.mu[i] = profile * m0[i];
    }
  }
  const auto derivatives = ComputeLoveNumberDerivatives(material, kLmax, d);
  const auto kernels = ComputeLoveNumberKernels(material, kLmax);
  auto pair = [&](const std::vector<double>& kappa, const std::vector<double>& mu) {
    auto total = 0.0;
    for (auto i = std::size_t{0}; i < k0.size(); ++i) {
      total += kappa[i] * d.kappa[i] + mu[i] * d.mu[i];
    }
    return total;
  };
  struct Check {
    const char* name;
    const LoveNumberKernels::Channel* channel;
    const std::vector<double>* h;
    const std::vector<double>* l;
    const std::vector<double>* k;
  };
  const Check checks[4] = {
      {"load", &kernels.load, &derivatives.h_u, &derivatives.l_u, &derivatives.k_u},
      {"potential", &kernels.potential, &derivatives.h_phi, &derivatives.l_phi,
       &derivatives.k_phi},
      {"tangential", &kernels.tangential, &derivatives.h_v, &derivatives.l_v,
       &derivatives.k_v},
      {"tide", &kernels.tide, &derivatives.h_t, &derivatives.l_t, &derivatives.k_t}};
  for (const auto& c : checks) {
    auto scale = 0.0;
    for (auto l = 0; l <= kLmax; ++l) {
      scale = std::max({scale, std::abs((*c.h)[l]), std::abs((*c.l)[l]),
                        std::abs((*c.k)[l])});
    }
    ASSERT_GT(scale, 0.0) << c.name;
    for (auto l = 0; l <= kLmax; ++l) {
      EXPECT_NEAR(pair(c.channel->h_kappa[l], c.channel->h_mu[l]), (*c.h)[l],
                  1e-10 * scale)
          << c.name << " h degree " << l;
      EXPECT_NEAR(pair(c.channel->l_kappa[l], c.channel->l_mu[l]), (*c.l)[l],
                  1e-10 * scale)
          << c.name << " l degree " << l;
      EXPECT_NEAR(pair(c.channel->k_kappa[l], c.channel->k_mu[l]), (*c.k)[l],
                  1e-10 * scale)
          << c.name << " k degree " << l;
    }
  }
}

