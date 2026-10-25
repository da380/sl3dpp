// Radial substructuring, step 1 of docs/dev/radial-substructuring.md: the
// degree systems solved segment by segment, the cut nodes' dofs together,
// against the direct banded solves. A synthetic spectral-element band
// system; every spheroidal and toroidal degree of the prem8 systems at
// lmax 64 (the degree-1 frame and pins, the fluid core, cuts at its
// boundaries too); and the 1D and 3D operators end to end.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <map>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/ToroidalOperator.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/detail/BandSolvers.hpp"
#include "sl3dpp/detail/BandSubstructure.hpp"

using namespace sl3dpp;

namespace {

/// The largest difference relative to the largest magnitude of `want`.
double RelativeDifference(std::span<const double> got,
                          std::span<const double> want) {
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    scale = std::max(scale, std::abs(want[i]));
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  return diff / scale;
}

/// A substructured solution `got` as good as the direct `want` of A x = b:
/// its residual within a few times the direct one's, and the two agreeing
/// to `tolerance`, as far as the system's conditioning determines them.
template <typename Operator>
void ExpectAsGoodAsDirect(const Operator& op, const std::vector<double>& b,
                          const std::vector<double>& want,
                          const std::vector<double>& got, double tolerance,
                          const std::string& label) {
  const auto residualWant = RelativeDifference(op.Apply(want), b);
  const auto residualGot = RelativeDifference(op.Apply(got), b);
  EXPECT_LT(residualGot, 3.0 * residualWant + 1e-15) << label;
  EXPECT_LT(RelativeDifference(got, want), tolerance) << label;
}

/// The prem8 deck's body on a mesh sized for lmax, and its Maxwell
/// rheology (1e21 Pa s between the CMB and 220 km depth).
struct Prem8 {
  EarthModel model;
  Material material;

  explicit Prem8(int lmax)
      : model(EarthModel::FromDeck(
                  Deck::Read(
                      std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                          "prem8.elastic.deck",
                      DeckFormat("elastic",
                                 {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
                      .value())
                  .value()),
        material(Material::Create(
                     model, RadialMesh::Create(model, {.lmax = lmax}).value())
                     .value()) {}

  [[nodiscard]] Rheology Maxwell() const {
    const auto& scales = model.ModelScales();
    auto viscosity = std::map<std::size_t, RadialValue>{};
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
      }
    }
    return Rheology::Maxwell(material, viscosity).value();
  }
};

/// The element boundaries where solid meets fluid.
std::vector<int> SolidFluidBoundaries(const Material& material) {
  auto out = std::vector<int>{};
  for (auto e = 1; e < material.Mesh().ElementCount(); ++e) {
    if (material.IsFluid(e - 1) != material.IsFluid(e)) out.push_back(e);
  }
  return out;
}

}  // namespace

TEST(BandSubstructure, MatchesTheDirectSolveOnASyntheticSystem) {
  // The degree systems' structure: elements of five GLL nodes sharing their
  // end nodes, three components per node in node-major order (band 14),
  // each element a random positive definite block, scaled over ten decades.
  constexpr auto elements = 40, nodes = 5, components = 3;
  constexpr auto kd = components * nodes - 1;
  constexpr auto n = components * (elements * (nodes - 1) + 1);
  constexpr auto ld = static_cast<std::size_t>(kd) + 1;
  auto engine = std::mt19937{7};
  auto uniform = std::uniform_real_distribution<double>{-1.0, 1.0};
  auto ab = std::vector<double>(ld * n, 0.0);
  constexpr auto local = components * nodes;
  for (auto e = 0; e < elements; ++e) {
    auto m = std::vector<double>(local * local);
    for (auto& v : m) v = uniform(engine);
    const auto scale = std::pow(10.0, 10.0 * e / (elements - 1));
    const auto first = components * e * (nodes - 1);
    for (auto a = 0; a < local; ++a) {
      for (auto b = a; b < local; ++b) {
        auto v = a == b ? 1.0 * local : 0.0;
        for (auto k = 0; k < local; ++k)
          v += m[a * local + k] * m[b * local + k];
        const auto i = first + a, j = first + b;
        ab[static_cast<std::size_t>(j) * ld + (kd + i - j)] += scale * v;
      }
    }
  }
  constexpr auto nrhs = 3;
  auto b = std::vector<double>(static_cast<std::size_t>(n) * nrhs);
  for (auto& v : b) v = uniform(engine);
  auto direct = b;
  auto factor = ab;
  ASSERT_EQ(detail::BandCholeskyFactor(factor, n, kd), 0);
  detail::BandCholeskySolve(factor, n, kd, direct, nrhs);

  // Cuts at every 2nd, 5th and 13th element boundary.
  for (const auto every : {2, 5, 13}) {
    auto separators = std::vector<std::pair<int, int>>{};
    for (auto e = every; e < elements; e += every) {
      const auto node = e * (nodes - 1);
      separators.emplace_back(components * node, components * (node + 1));
    }
    const auto solver = detail::BandSubstructure(ab, n, kd, separators);
    EXPECT_EQ(solver.SegmentCount(), static_cast<int>(separators.size()) + 1);
    EXPECT_EQ(solver.SeparatorDofCount(),
              components * static_cast<int>(separators.size()));
    EXPECT_TRUE(solver.UsedCholesky());
    auto x = b;
    solver.Solve(x, nrhs);
    EXPECT_LT(RelativeDifference(x, direct), 1e-13) << "every " << every;
  }

  // A cut through an element does not separate it.
  const auto inside = 3 * (nodes - 1) + 2;
  EXPECT_THROW(
      detail::BandSubstructure(
          ab, n, kd, {{components * inside, components * (inside + 1)}}),
      std::invalid_argument);
}

TEST(Substructure, EverySpheroidalDegreeMatchesTheDirectSolve) {
  // The loading problem at every degree to 64, cut into four segments of
  // about equal element counts within the layers: the load, tide and
  // tangential solutions have the direct solves' residuals, and agree with
  // them as far as the systems determine them (their condition numbers
  // reach 2.5e11 at degree 1: 1.4e-10 there, below 1e-12 from degree 10;
  // the direct solve's own error is as large).
  constexpr auto lmax = 64;
  const auto prem = Prem8(lmax);
  const auto& material = prem.material;
  const auto cuts = material.Mesh().BalancedCuts(4);
  ASSERT_EQ(cuts.size(), 3u);
  auto worst = 0.0;
  for (auto l = 0; l <= lmax; ++l) {
    const DegreeOperator direct(material, l);
    const DegreeOperator cut(material, l, {.cuts = cuts});
    const auto inside = std::ranges::count_if(
        cuts, [&](int c) { return c > direct.FirstElement(); });
    EXPECT_EQ(cut.SegmentCount(), 1 + inside) << "degree " << l;
    auto b = direct.Load(LoadPart::Both);
    const auto tide = direct.Tide(l == 0 ? 2 : -1);
    b.insert(b.end(), tide.begin(), tide.end());
    auto nrhs = 2;
    if (l > 0) {
      const auto tangential = direct.Tangential();
      b.insert(b.end(), tangential.begin(), tangential.end());
      ++nrhs;
    }
    const auto want = direct.Solve(b, nrhs);
    const auto got = cut.Solve(b, nrhs);
    const auto n = static_cast<std::size_t>(direct.DofCount());
    const auto appliedWant = direct.Apply(want, nrhs);
    const auto appliedGot = direct.Apply(got, nrhs);
    for (auto k = 0; k < nrhs; ++k) {
      auto column = [&](const std::vector<double>& v) {
        return std::span(v).subspan(k * n, n);
      };
      // Residuals relative to the right-hand side: the same for both.
      const auto residualWant =
          RelativeDifference(column(appliedWant), column(b));
      const auto residualGot =
          RelativeDifference(column(appliedGot), column(b));
      EXPECT_LT(residualGot, 3.0 * residualWant + 1e-15)
          << "degree " << l << " column " << k;
      const auto error = RelativeDifference(column(got), column(want));
      EXPECT_LT(error, l < 10 ? 1e-9 : 1e-11)
          << "degree " << l << " column " << k;
      worst = std::max(worst, error);
    }
  }
  RecordProperty("worst", std::to_string(worst));
}

TEST(Substructure, CutsAtTheSolidFluidBoundariesToo) {
  // Algebraically any element boundary separates; the fluid core's own
  // boundaries, where the dofs change, included. (The radial plan keeps
  // cuts inside layers, which BalancedCuts does.)
  constexpr auto lmax = 32;
  const auto prem = Prem8(lmax);
  const auto& material = prem.material;
  const auto cuts = SolidFluidBoundaries(material);
  ASSERT_EQ(cuts.size(), 2u);  // the inner core and the outer core
  for (const auto l : {0, 1, 2, 3}) {
    const DegreeOperator direct(material, l);
    const DegreeOperator cut(material, l, {.cuts = cuts});
    EXPECT_EQ(cut.SegmentCount(), 3) << "degree " << l;
    const auto b = direct.Load(LoadPart::Both);
    ExpectAsGoodAsDirect(direct, b, direct.Solve(b), cut.Solve(b), 1e-9,
                         std::format("degree {}", l));
  }
  for (const auto l : {1, 2}) {
    const ToroidalOperator direct(material, l);
    const ToroidalOperator cut(material, l, {.cuts = cuts});
    auto x = std::vector<double>(direct.DofCount());
    for (auto i = std::size_t{0}; i < x.size(); ++i) x[i] = std::sin(0.1 * i);
    const auto b = direct.Apply(x);
    ExpectAsGoodAsDirect(direct, b, direct.Solve(b), cut.Solve(b), 1e-9,
                         std::format("toroidal degree {}", l));
  }
}

TEST(Substructure, EveryToroidalDegreeMatchesTheDirectSolve) {
  // Degrees 1 to 64, degree 1 in the pinned gauge (one pin per solid
  // region), on right-hand sides in the operator's range: residuals as the
  // direct solve's, agreement as the spheroidal systems'.
  constexpr auto lmax = 64;
  const auto prem = Prem8(lmax);
  const auto& material = prem.material;
  const auto cuts = material.Mesh().BalancedCuts(4);
  for (auto l = 1; l <= lmax; ++l) {
    const ToroidalOperator direct(material, l);
    const ToroidalOperator cut(material, l, {.cuts = cuts});
    auto x = std::vector<double>(direct.DofCount());
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      x[i] = std::cos(0.37 * i + l);
    }
    const auto b = direct.Apply(x);
    ExpectAsGoodAsDirect(direct, b, direct.Solve(b), cut.Solve(b),
                         l < 10 ? 1e-9 : 1e-11, std::format("degree {}", l));
  }
}

TEST(Substructure, LoveNumbersAndEffectiveTablesAreTheDirectOnes) {
  // Through the 1D viscoelastic operator: its unrelaxed and per-step
  // effective tables, every system substructured.
  constexpr auto lmax = 32;
  const auto prem = Prem8(lmax);
  const auto rheology = prem.Maxwell();
  const auto cuts = prem.material.Mesh().BalancedCuts(3);
  ViscoelasticOperator direct(rheology, lmax);
  ViscoelasticOperator cut(rheology, lmax, {.radial_cuts = cuts});
  auto compare = [](const LoveNumbers& got, const LoveNumbers& want) {
    for (const auto member :
         {&LoveNumbers::h_u, &LoveNumbers::k_u, &LoveNumbers::h_phi,
          &LoveNumbers::k_phi, &LoveNumbers::h_t, &LoveNumbers::k_t}) {
      EXPECT_LT(RelativeDifference(got.*member, want.*member), 1e-12);
    }
  };
  compare(cut.ElasticLoveNumbers(), direct.ElasticLoveNumbers());
  const auto dt = 0.1;  // material units
  compare(cut.EffectiveLoveNumbers(Scheme::ExponentialTrapezoid, dt),
          direct.EffectiveLoveNumbers(Scheme::ExponentialTrapezoid, dt));
}

TEST(Substructure, ALateralEvolutionIsTheDirectOne) {
  // The coupled 3D evolution with a decade of lateral contrast, every
  // degree system (the reference's, the step's, spheroidal and toroidal)
  // substructured: the same history to the fixed point's tolerance.
  constexpr auto lmax = 8;
  const auto prem = Prem8(lmax);
  const auto rheology = prem.Maxwell();
  ViscoelasticOperator reference(rheology, lmax);
  SeaLevelSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
                        SeaLevelParameters{});
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
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
    load[i] =
        -0.1 * solver.IceDensity() * (1.0 - state.oceanFunction[i]) * ice[i];
  }
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, kyr, 2 * kyr, 3 * kyr};
  auto delta = [](double, double theta, double phi) {
    return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto run = [&](std::vector<int> cuts) {
    Viscoelastic3D op(rheology, lmax, delta, {.radial_cuts = std::move(cuts)});
    SeaLevelEvolution3D evolution(op, solver, state,
                                  {.rtol = 1e-12, .coupled_rtol = 1e-12});
    return evolution.Run(times, [&](double) { return load; });
  };
  const auto want = run({});
  const auto got = run(prem.material.Mesh().BalancedCuts(4));
  ASSERT_EQ(got.size(), want.size());
  for (auto n = std::size_t{1}; n < got.size(); ++n) {
    EXPECT_TRUE(got[n].converged);
    EXPECT_LT(RelativeDifference(got[n].seaLevelChange, want[n].seaLevelChange),
              1e-10)
        << "node " << n;
    EXPECT_LT(RelativeDifference(got[n].displacement, want[n].displacement),
              1e-10)
        << "node " << n;
  }
}

TEST(Substructure, BalancedCutsStayInsideLayers) {
  const auto prem = Prem8(32);
  const auto& mesh = prem.material.Mesh();
  for (const auto parts : {1, 2, 4, 8, 16}) {
    const auto cuts = mesh.BalancedCuts(parts);
    EXPECT_EQ(static_cast<int>(cuts.size()), parts - 1);
    for (auto k = std::size_t{0}; k < cuts.size(); ++k) {
      EXPECT_EQ(mesh.LayerOf(cuts[k] - 1), mesh.LayerOf(cuts[k]));
      if (k > 0) EXPECT_GT(cuts[k], cuts[k - 1]);
    }
  }
  // Weighted: all the weight in the top half of the elements puts every
  // cut there.
  auto weights = std::vector<double>(mesh.ElementCount(), 0.0);
  for (auto e = mesh.ElementCount() / 2; e < mesh.ElementCount(); ++e) {
    weights[e] = 1.0;
  }
  for (const auto cut : mesh.BalancedCuts(4, weights)) {
    EXPECT_GE(cut, mesh.ElementCount() / 2 - 1);
  }
  EXPECT_THROW(
      static_cast<void>(DegreeOperator(prem.material, 2, {.cuts = {5, 3}})),
      std::invalid_argument);
}
