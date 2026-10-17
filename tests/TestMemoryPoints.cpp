// Memory points and the relaxation term (plan.md §5.5): the GLL set is the
// nodes themselves and the Gauss set interpolates and integrates exactly to
// its polynomial degrees; with GLL points the relaxation term reproduces the
// operators assembled with relaxed moduli (the step's exact elimination of
// the new internal variables), and with Gauss points it converges to them
// under mesh refinement.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/MemoryPoints.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/ToroidalOperator.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

namespace {

EarthModel Model() {
  auto deck =
      Deck::Read(
          std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
          DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  return EarthModel::FromDeck(deck).value();
}

Rheology Maxwell(const EarthModel& model, const Material& material) {
  const auto& scales = model.ModelScales();
  std::map<std::size_t, RadialValue> viscosity;
  const auto b = model.DimensionlessView().Boundaries();
  for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
    if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
        b[i + 1] <= 6.1511e6 / scales.Length()) {
      viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
    }
  }
  return Rheology::Maxwell(material, viscosity).value();
}

// The effective-modulus weight of a branch: 1/(1 + h) or phi1(h) = 1 - beta.
double EffectiveWeight(Scheme scheme, double h) {
  return scheme == Scheme::BackwardEuler
             ? BackwardEulerWeight(h)
             : 1.0 - ExponentialTrapezoidWeights(h).beta;
}

// The relaxation coefficient c = sum_k mu_k (1 - w_k) at the points, from
// the branch arrays interpolated there (the GLL set takes them as they are).
std::vector<double> Coefficient(const Rheology& rheology,
                                const MemoryPoints& points, Scheme scheme,
                                double dt) {
  auto c = std::vector<double>{};
  for (auto k = std::size_t{0}; k < rheology.BranchCount(); ++k) {
    const auto mu = points.Interpolate(rheology.BranchAt(k).mu);
    const auto itau = points.Interpolate(rheology.BranchAt(k).itau);
    c.resize(mu.size(), 0.0);
    for (auto i = std::size_t{0}; i < mu.size(); ++i) {
      c[i] += mu[i] * (1.0 - EffectiveWeight(scheme, dt * itau[i]));
    }
  }
  return c;
}

double MaxAbs(const std::vector<double>& v) {
  auto m = 0.0;
  for (auto x : v) m = std::max(m, std::abs(x));
  return m;
}

double MaxDiff(const std::vector<double>& a, const std::vector<double>& b) {
  auto m = 0.0;
  for (auto i = std::size_t{0}; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

}  // namespace

TEST(MemoryPoints, GllIsTheNodesAndGaussInterpolatesExactly) {
  const auto model = Model();
  const auto mesh = RadialMesh::Create(model, {.lmax = 8}).value();
  const auto n = mesh.NGll();

  const auto gll = MemoryPoints::Of(MemoryPointSet::Gll, mesh);
  ASSERT_EQ(gll.count, n);
  for (auto p = 0; p < n; ++p) {
    for (auto i = 0; i < n; ++i) {
      EXPECT_EQ(gll.Value(p, i), p == i ? 1.0 : 0.0);
      EXPECT_EQ(gll.Derivative(p, i), mesh.Derivative(p, i));
    }
  }

  const auto gauss = MemoryPoints::Of(MemoryPointSet::Gauss, mesh);
  ASSERT_EQ(gauss.count, n - 1);
  const auto nodes = mesh.Points();
  // The GLL basis reproduces every polynomial of degree <= ngll - 1 and
  // its derivative at the points; the N-point Gauss rule integrates
  // exactly to degree 2N - 1.
  for (auto k = 0; k <= n - 1; ++k) {
    for (auto p = 0; p < gauss.count; ++p) {
      auto value = 0.0, slope = 0.0;
      for (auto i = 0; i < n; ++i) {
        value += gauss.Value(p, i) * std::pow(nodes[i], k);
        slope += gauss.Derivative(p, i) * std::pow(nodes[i], k);
      }
      const auto x = gauss.xi[p];
      EXPECT_NEAR(value, std::pow(x, k), 1e-13) << "degree " << k;
      EXPECT_NEAR(slope, k == 0 ? 0.0 : k * std::pow(x, k - 1), 1e-12)
          << "degree " << k;
    }
  }
  for (auto k = 0; k <= 2 * gauss.count - 1; ++k) {
    auto sum = 0.0;
    for (auto p = 0; p < gauss.count; ++p) {
      sum += gauss.weight[p] * std::pow(gauss.xi[p], k);
    }
    EXPECT_NEAR(sum, k % 2 == 0 ? 2.0 / (k + 1) : 0.0, 1e-14) << "degree " << k;
  }

  // Interpolating nodal radii reproduces the affine radius at the points.
  auto radii =
      std::vector<double>(static_cast<std::size_t>(mesh.ElementCount()) * n);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto i = 0; i < n; ++i) radii[e * n + i] = mesh.Radius(e, i);
  }
  const auto atPoints = gauss.Interpolate(radii);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto p = 0; p < gauss.count; ++p) {
      const auto expected =
          mesh.Radius(e, 0) + (gauss.xi[p] + 1.0) * mesh.Jacobian(e);
      EXPECT_NEAR(atPoints[e * gauss.count + p], expected, 1e-14);
    }
  }
}

TEST(Relaxation, GllReproducesTheRelaxedModuli) {
  // The decisive gate: with internal variables at the GLL nodes the
  // relaxation term is the relaxed-moduli assembly, so the effective
  // operators agree to round-off, spheroidal (by their solves) and
  // toroidal (by their matrices).
  const auto model = Model();
  const auto mesh = RadialMesh::Create(model, {.lmax = 8}).value();
  const auto material = Material::Create(model, mesh).value();
  const auto rheology = Maxwell(model, material);
  const ViscoelasticOperator op(rheology, 8);
  const auto points = MemoryPoints::Of(MemoryPointSet::Gll, mesh);
  const auto dt = 2.0 * rheology.MinRelaxationTime();

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto relaxed = op.EffectiveMaterial(scheme, dt);
    const auto c = Coefficient(rheology, points, scheme, dt);
    const Relaxation relaxation{&points, c};

    for (auto l : {0, 1, 2, 5, 8}) {
      const DegreeOperator viaTerm(material, l, {.relaxation = &relaxation});
      const DegreeOperator viaModuli(relaxed, l);
      ASSERT_EQ(viaTerm.DofCount(), viaModuli.DofCount());
      for (const auto& rhs : {viaModuli.Load(), viaModuli.Tide()}) {
        const auto a = viaTerm.Solve(rhs);
        const auto b = viaModuli.Solve(rhs);
        EXPECT_LE(MaxDiff(a, b), 1e-10 * MaxAbs(b))
            << "scheme " << static_cast<int>(scheme) << ", l = " << l;
      }
    }
    for (auto l : {1, 2, 5}) {
      const ToroidalOperator viaTerm(material, l, {.relaxation = &relaxation});
      const ToroidalOperator viaModuli(relaxed, l);
      auto y = std::vector<double>(viaTerm.DofCount());
      for (auto d = std::size_t{0}; d < y.size(); ++d) {
        y[d] = std::sin(0.7 * static_cast<double>(d) + 0.2);
      }
      const auto a = viaTerm.Apply(y);
      const auto b = viaModuli.Apply(y);
      EXPECT_LT(MaxDiff(a, b), 1e-12 * MaxAbs(b))
          << "toroidal, scheme " << static_cast<int>(scheme) << ", l = " << l;
    }
  }
}

TEST(Relaxation, LoveNumbersFromTheRelaxedOperators) {
  // The step's effective Love numbers (what the sea level solver uses)
  // computed through the relaxation term equal the relaxed material's.
  const auto model = Model();
  const auto mesh = RadialMesh::Create(model, {.lmax = 8}).value();
  const auto material = Material::Create(model, mesh).value();
  const auto rheology = Maxwell(model, material);
  const ViscoelasticOperator op(rheology, 8);
  const auto points = MemoryPoints::Of(MemoryPointSet::Gll, mesh);
  const auto dt = 2.0 * rheology.MinRelaxationTime();
  const auto scheme = Scheme::ExponentialTrapezoid;

  const auto c = Coefficient(rheology, points, scheme, dt);
  const Relaxation relaxation{&points, c};
  const auto viaTerm =
      ComputeLoveNumbers(material, 8, {.relaxation = &relaxation});
  const auto viaModuli =
      ComputeLoveNumbers(op.EffectiveMaterial(scheme, dt), 8);
  for (const auto pair :
       {&LoveNumbers::h_u, &LoveNumbers::k_u, &LoveNumbers::l_u,
        &LoveNumbers::h_t, &LoveNumbers::k_t}) {
    EXPECT_LT(MaxDiff(viaTerm.*pair, viaModuli.*pair),
              1e-10 * MaxAbs(viaModuli.*pair));
  }
}

TEST(Relaxation, GaussConvergesToGllUnderRefinement) {
  // The two point sets integrate the relaxation term with different
  // quadrature errors, so their effective solids differ, but by an amount
  // that falls quickly under mesh refinement.
  const auto model = Model();
  const auto scheme = Scheme::ExponentialTrapezoid;
  auto gap = [&](double drmax) {
    const auto mesh = RadialMesh::Create(model, {.drmax = drmax}).value();
    const auto material = Material::Create(model, mesh).value();
    const auto rheology = Maxwell(model, material);
    const auto dt = 2.0 * rheology.MinRelaxationTime();
    auto worst = 0.0;
    std::vector<LoveNumbers> love;
    for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss}) {
      const auto points = MemoryPoints::Of(set, mesh);
      const auto c = Coefficient(rheology, points, scheme, dt);
      const Relaxation relaxation{&points, c};
      love.push_back(
          ComputeLoveNumbers(material, 8, {.relaxation = &relaxation}));
    }
    for (auto l = 2; l <= 8; ++l) {
      worst = std::max(worst, std::abs(love[1].h_u[l] - love[0].h_u[l]) /
                                  std::abs(love[0].h_u[l]));
    }
    return worst;
  };
  const auto coarse = gap(600e3);
  const auto fine = gap(300e3);
  EXPECT_GT(coarse, 0.0);
  EXPECT_LT(coarse, 1e-3);
  EXPECT_LT(fine, 0.25 * coarse);
}
