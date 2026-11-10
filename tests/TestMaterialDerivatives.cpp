// The material derivatives of the static gravitating elastic problem:
// the variation operator is the derivative of the assembled system along
// a direction (central differences of K(m +/- eps d) x, Richardson to
// rounding), the Love-number derivatives match central differences of
// the Love numbers of the shifted material for directions in mu and kappa
// separately and together (Richardson to 1e-7), and the derivative
// respects the reciprocity identities differentiated.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <filesystem>
#include <map>
#include <string>
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
/// everywhere for kappa.
MaterialDirection SmoothDirection(const Material& material, bool mu,
                                  bool kappa) {
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
      if (mu && !material.IsFluid(e)) d.mu[i] = profile * m0[i];
      if (kappa) d.kappa[i] = profile * k0[i];
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

TEST(MaterialDerivatives, VariationOperatorIsTheDerivativeOfTheSystem) {
  // (dK) x against central differences of K(m +/- eps d) x on the shifted
  // materials, for every degree and both problems: the operator is affine
  // in the moduli, so the differences are exact and Richardson lands to
  // rounding.
  const auto material = MakeMaterial();
  const auto d = SmoothDirection(material, true, true);
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
  // A direction of another size is refused.
  MaterialDirection bad;
  bad.mu.assign(3, 1.0);
  EXPECT_THROW((void)material.Variation(bad), std::invalid_argument);
}

TEST(MaterialDerivatives, LoveNumberDerivativesMatchDifferences) {
  // Along directions in mu and kappa, alone and together, central
  // differences of the Love numbers of the shifted material converge at
  // second order to the derivatives and Richardson lands on them at 1e-7,
  // every channel and degree and the axial numbers.
  const auto material = MakeMaterial();
  struct Case {
    const char* name;
    bool mu, kappa;
  };
  for (const auto& c : {Case{"mu", true, false}, Case{"kappa", false, true},
                        Case{"both", true, true}}) {
    const auto d = SmoothDirection(material, c.mu, c.kappa);
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
  }
}

TEST(MaterialDerivatives, ReciprocityDifferentiated) {
  // The identities g h_phi = k_u, h_v = l(l+1) l_u, k_v = g l(l+1) l_phi
  // hold at every material, and the moduli do not move g, so their
  // derivatives along a direction in the moduli hold too: g dh_phi = dk_u
  // and so on.
  const auto material = MakeMaterial();
  const auto d = SmoothDirection(material, true, true);
  const auto love = ComputeLoveNumbers(material, kLmax);
  const auto dlove = ComputeLoveNumberDerivatives(material, kLmax, d);
  const auto g = love.surface_gravity;
  EXPECT_EQ(dlove.surface_gravity, 0.0);
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
    EXPECT_NEAR(g * dlove.h_phi[i], dlove.k_u[i], 1e-10 * kScale)
        << "degree " << l;
    EXPECT_NEAR(dlove.h_v[i], k2 * dlove.l_u[i], 1e-10 * hvScale)
        << "degree " << l;
    EXPECT_NEAR(dlove.k_v[i], k2 * (g * dlove.l_phi[i]), 1e-10 * kvScale)
        << "degree " << l;
  }
  // An empty direction gives zero derivatives.
  const auto none = ComputeLoveNumberDerivatives(material, kLmax, {});
  for (const auto& [name, v] : Entries(none)) EXPECT_EQ(v, 0.0) << name;
}

TEST(MaterialDerivatives, LoveNumberKernelsPairToTheDerivatives) {
  // The static kernels: for every channel and degree, the kappa and mu
  // kernels paired with a smooth direction in the moduli equal the
  // Love-number derivatives along it (the exact linearisation) at 1e-10 —
  // the adjoint of the static problem against its tangent.
  const auto material = MakeMaterial();
  const auto& mesh = material.Mesh();
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
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


namespace {

/// A smooth perturbation of the moduli at memory points, (element, point):
/// a fraction of the nodal modulus interpolated there, on the solid
/// elements, and for kappa in the fluid too when asked.
std::pair<std::vector<double>, std::vector<double>> PointDirection(
    const Material& material, const MemoryPoints& points, bool fluidKappa) {
  const auto& mesh = material.Mesh();
  const auto [k0, m0] = material.Moduli().KappaMu();
  const auto kAt = points.Interpolate(k0);
  const auto mAt = points.Interpolate(m0);
  auto dk = std::vector<double>(kAt.size(), 0.0);
  auto dm = std::vector<double>(mAt.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto p = 0; p < points.count; ++p) {
      const auto i = static_cast<std::size_t>(e) * points.count + p;
      const auto r = points.RadiusAt(mesh, e, p);
      const auto profile =
          0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
      if (!material.IsFluid(e)) dm[i] = profile * mAt[i];
      if (!material.IsFluid(e) || fluidKappa) dk[i] = 0.7 * profile * kAt[i];
    }
  }
  return {std::move(dk), std::move(dm)};
}

double RelativeDifference(const std::vector<double>& a,
                          const std::vector<double>& b) {
  auto diff = 0.0, scale = 0.0;
  for (auto i = std::size_t{0}; i < a.size(); ++i) {
    diff = std::max(diff, std::abs(a[i] - b[i]));
    scale = std::max(scale, std::abs(b[i]));
  }
  return diff / scale;
}

}  // namespace

TEST(MaterialDerivatives, PointVariationAtTheGllPointsIsTheNodalVariation) {
  // With the memory points at the GLL nodes a perturbation of the moduli
  // at the points is a nodal one, and the two variations of the system
  // are the same operator: on the solid for every degree, and with the
  // fluid core's bulk modulus too at degree zero, where it enters the
  // elastic block.
  const auto material = MakeMaterial();
  const auto points = MemoryPoints::Of(MemoryPointSet::Gll, material.Mesh());
  for (auto l = 0; l <= kLmax; ++l) {
    const auto [dk, dm] = PointDirection(material, points, l == 0);
    MaterialDirection d;
    d.kappa = dk;
    d.mu = dm;
    const auto variation = material.Variation(d);
    DegreeOperator op(material, l, {});
    const auto n = op.DofCount();
    constexpr auto nrhs = 2;
    auto x = std::vector<double>(static_cast<std::size_t>(n) * nrhs);
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      x[i] = std::sin(0.37 * i + 0.1 * l) + 0.3;
    }
    const auto nodal = op.ApplyVariation(variation, x, nrhs);
    const auto atPoints = op.ApplyPointVariation(points, dk, dm, x, nrhs);
    EXPECT_LT(RelativeDifference(atPoints, nodal), 1e-12) << "degree " << l;
  }
}

TEST(MaterialDerivatives, PointVariationIsThePointShearOfTheAssembly) {
  // At Gauss points: the system assembled with a shear modulus at the
  // points, less the system without, is the point variation along that
  // modulus, the form being linear in it.
  const auto material = MakeMaterial();
  const auto points = MemoryPoints::Of(MemoryPointSet::Gauss, material.Mesh());
  const auto [dk, dm] = PointDirection(material, points, false);
  const PointShear shear{&points, dm};
  for (auto l = 0; l <= kLmax; ++l) {
    DegreeOperator base(material, l, {});
    DegreeOperator raised(material, l, {.point_shear = &shear});
    const auto n = base.DofCount();
    ASSERT_EQ(raised.DofCount(), n);
    auto x = std::vector<double>(n);
    for (auto i = 0; i < n; ++i) x[i] = std::sin(0.37 * i + 0.1 * l) + 0.3;
    auto difference = raised.Apply(x);
    const auto without = base.Apply(x);
    for (auto i = 0; i < n; ++i) difference[i] -= without[i];
    const auto predicted = base.ApplyPointVariation(points, {}, dm, x);
    auto scale = 0.0, diff = 0.0;
    for (auto i = 0; i < n; ++i) {
      scale = std::max(scale, std::abs(without[i]));
      diff = std::max(diff, std::abs(difference[i] - predicted[i]));
    }
    EXPECT_LT(diff, 1e-12 * scale) << "degree " << l;
    if (l != 1) {
      EXPECT_GT(RelativeDifference(predicted, without), 1e-6)
          << "degree " << l;
    }
  }
}

TEST(MaterialDerivatives, PointGradientIsTheTransposeOfThePointVariation) {
  // sum_p kappa_p dkappa_p + mu_p dmu_p = y^T (dK) x at Gauss points, for
  // random solutions of two columns and every degree.
  const auto material = MakeMaterial();
  const auto points = MemoryPoints::Of(MemoryPointSet::Gauss, material.Mesh());
  std::mt19937 rng(11);
  std::normal_distribution<double> normal;
  for (auto l = 0; l <= kLmax; ++l) {
    const auto [dk, dm] = PointDirection(material, points, l == 0);
    DegreeOperator op(material, l, {});
    const auto n = op.DofCount();
    constexpr auto nrhs = 2;
    std::vector<double> x(static_cast<std::size_t>(n) * nrhs),
        y(static_cast<std::size_t>(n) * nrhs);
    for (auto& v : x) v = normal(rng);
    for (auto& v : y) v = normal(rng);
    const auto dKx = op.ApplyPointVariation(points, dk, dm, x, nrhs);
    auto direct = 0.0;
    for (auto i = std::size_t{0}; i < y.size(); ++i) direct += y[i] * dKx[i];
    const auto [gk, gm] = op.PointGradient(points, y, x, nrhs);
    auto transposed = 0.0;
    for (auto i = std::size_t{0}; i < gk.size(); ++i) {
      transposed += gk[i] * dk[i] + gm[i] * dm[i];
    }
    EXPECT_NEAR(transposed, direct, 1e-12 * std::abs(direct) + 1e-300)
        << "degree " << l;
    EXPECT_NE(direct, 0.0) << "degree " << l;
  }
  DegreeOperator op(material, 2, {});
  const auto x = std::vector<double>(op.DofCount(), 1.0);
  const auto bad = std::vector<double>(3, 1.0);
  EXPECT_THROW((void)op.ApplyPointVariation(points, bad, {}, x),
               std::invalid_argument);
}
