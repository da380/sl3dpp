// The node blocks (GridCoupling::Options::transform_degrees): the
// transforms cut with depth to the degrees the solution carries there.
// The blocks follow the truncation rule; each block's transforms invert on
// its own grid, and OnSurfaceGrid carries its fields to the lmax grid
// exactly; a step on the blocks is the single grid's to round-off without
// lateral viscosity, and to the quadrature of the lateral product with it
// (the product's weights are resolved to each block's degree); and the 3D
// adjoint's kernels, laid out per block, pair with a smooth direction as
// the single grid's do.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <span>
#include <vector>

#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using Complex = std::complex<double>;

namespace {

constexpr auto kLmax = 16;
// A loose truncation, so that at this small lmax the deep mantle carries
// only the lower degrees and the blocks form (the default 1e-8 keeps
// every degree to 30 throughout the mantle).
constexpr auto kEps = 1e-2;

std::vector<int> Degrees() { return {8, 12}; }

double Noise(std::size_t k) {
  const auto x = std::sin(12.9898 * static_cast<double>(k + 1)) * 43758.5453;
  return x - std::floor(x) - 0.5;
}

struct Body {
  EarthModel model;
  Material material;
  Rheology rheology;

  static Body Make(double viscositySI = 1e21) {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
    auto material = Material::Create(model, mesh).value();
    const auto& scales = model.ModelScales();
    auto viscosity = std::map<std::size_t, RadialValue>{};
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i, viscositySI / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    return Body{std::move(model), std::move(material), std::move(rheology)};
  }
};

Viscoelastic3D::Options StepOptions(bool blocks) {
  return {.eps = kEps,
          .max_iterations = 100,
          .rtol = 1e-12,
          .transform_degrees = blocks ? Degrees() : std::vector<int>{}};
}

std::vector<Complex> LoadColumns(int l) {
  auto columns = std::vector<Complex>(l + 1);
  for (auto m = 0; m <= l; ++m) {
    columns[m] = m == 0 ? Complex(Noise(97 * l), 0.0)
                        : Complex(Noise(97 * l + m), Noise(55 * l + m));
  }
  return columns;
}

/// The largest relative difference between two sets of degree solutions.
double Difference(const Viscoelastic3D::Solutions& a,
                  const Viscoelastic3D::Solutions& b) {
  auto scale = 0.0, diff = 0.0;
  for (const auto* pair : {&a.spheroidal, &a.toroidal}) {
    const auto& other = pair == &a.spheroidal ? b.spheroidal : b.toroidal;
    for (const auto& [l, x] : *pair) {
      const auto& y = other.at(l);
      for (auto i = std::size_t{0}; i < x.size(); ++i) {
        scale = std::max(scale, std::abs(x[i]));
        diff = std::max(diff, std::abs(x[i] - y[i]));
      }
    }
  }
  return diff / scale;
}

/// Two steps from rest under a constant load, on the blocks or not.
Viscoelastic3D::Solutions TwoSteps(
    const Body& body,
    const std::function<double(double, double, double)>& delta, bool blocks,
    int* blockCount = nullptr) {
  Viscoelastic3D op(body.rheology, kLmax, delta, StepOptions(blocks));
  if (blockCount != nullptr) *blockCount = op.Coupling().BlockCount();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto load = [](int l, double) { return LoadColumns(l); };
  auto state = op.ZeroState();
  Viscoelastic3D::Solutions solutions;
  for (auto i = 0; i < 2; ++i) {
    auto result = op.Step(state, tau, Scheme::ExponentialTrapezoid, load);
    EXPECT_TRUE(result.info.converged);
    state = std::move(result.state);
    solutions = std::move(result.solutions);
  }
  return solutions;
}

}  // namespace

TEST(NodeBlocks, FollowTheTruncation) {
  const auto body = Body::Make();
  const auto& mesh = body.material.Mesh();
  const GridCoupling coupling(body.rheology, kLmax,
                              {.eps = kEps, .transform_degrees = Degrees()});
  ASSERT_GE(coupling.BlockCount(), 2);
  auto listed = Degrees();
  listed.push_back(kLmax);
  for (auto b = 1; b < coupling.BlockCount(); ++b) {
    EXPECT_EQ(
        coupling.FirstNodeOfBlock(b),
        coupling.FirstNodeOfBlock(b - 1) + coupling.NodeCountOfBlock(b - 1));
    EXPECT_GT(coupling.GridOfBlock(b).LMax(),
              coupling.GridOfBlock(b - 1).LMax());
  }
  auto offset = std::size_t{0};
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    // Every degree whose subset reaches the element is carried, by the
    // least listed degree that covers them all.
    const auto e = coupling.ElementOfNode(node);
    auto reach = 0;
    for (auto l = 1; l <= kLmax; ++l) {
      if (mesh.StartElement(l, TruncationRule::Graded, kEps) <= e) reach = l;
    }
    EXPECT_EQ(coupling.GridOf(node).LMax(),
              *std::ranges::lower_bound(listed, reach))
        << "node " << node;
    EXPECT_EQ(coupling.FieldOffset(node), offset);
    EXPECT_EQ(coupling.FieldSizeOf(node), coupling.GridOf(node).FieldSize());
    offset += coupling.FieldSizeOf(node);
  }
  EXPECT_EQ(coupling.GridValueCount(), offset);
  EXPECT_LT(coupling.GridValueCount(),
            coupling.NodeCount() * coupling.Grid().FieldSize());

  // The grid-free degrees and work weights are the blocks': per element,
  // its nodes' degree (0 without nodes) and its nodes times (L + 1)^3
  // over (lmax + 1)^3.
  const auto options =
      GridCoupling::Options{.eps = kEps, .transform_degrees = Degrees()};
  const auto degrees =
      GridCoupling::TransformDegrees(body.rheology, kLmax, options);
  const auto weights =
      GridCoupling::TransformWeights(body.rheology, kLmax, options);
  ASSERT_EQ(degrees.size(), static_cast<std::size_t>(mesh.ElementCount()));
  ASSERT_EQ(weights.size(), degrees.size());
  auto nodesIn = std::vector<int>(degrees.size(), 0);
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    const auto e = coupling.ElementOfNode(node);
    ++nodesIn[e];
    EXPECT_EQ(degrees[e], coupling.GridOf(node).LMax());
  }
  for (auto e = std::size_t{0}; e < degrees.size(); ++e) {
    if (nodesIn[e] == 0) {
      EXPECT_EQ(degrees[e], 0);
      EXPECT_EQ(weights[e], 0.0);
      continue;
    }
    EXPECT_DOUBLE_EQ(
        weights[e],
        nodesIn[e] * std::pow((degrees[e] + 1.0) / (kLmax + 1.0), 3));
  }

  // Under the default truncation every viscous element carries every
  // degree to 16: one block, the single grid.
  const GridCoupling whole(body.rheology, kLmax, {.transform_degrees = {4}});
  EXPECT_EQ(whole.BlockCount(), 1);
  EXPECT_EQ(whole.GridValueCount(),
            whole.NodeCount() * whole.Grid().FieldSize());
  EXPECT_THROW(GridCoupling(body.rheology, kLmax, {.transform_degrees = {1}}),
               std::invalid_argument);
  EXPECT_THROW(
      GridCoupling(body.rheology, kLmax, {.transform_degrees = {kLmax + 1}}),
      std::invalid_argument);
}

TEST(NodeBlocks, TransformsInvertAndCarryToTheSurfaceGrid) {
  const auto body = Body::Make();
  const GridCoupling blocked(body.rheology, kLmax,
                             {.eps = kEps, .transform_degrees = Degrees()});
  const GridCoupling single(body.rheology, kLmax, {.eps = kEps});
  ASSERT_GE(blocked.BlockCount(), 2);
  ASSERT_EQ(blocked.NodeCount(), single.NodeCount());

  // Coefficients to each node's own degree; the spin 0 ones of a real
  // field (m = 0 real), the spin -2 and -1 ones of complex fields.
  auto spectral = blocked.ZeroSpectral();
  for (auto c = 0; c < 2; ++c) {
    for (auto i = std::size_t{0}; i < spectral.c[c].size(); ++i) {
      spectral.c[c][i] = {Noise(i + 7919 * c), Noise(i + 31 + 7919 * c)};
    }
  }
  auto padded = single.ZeroSpectral();
  const auto full = single.Grid().CoefficientSize();
  auto at = std::size_t{0};
  for (auto node = 0; node < blocked.NodeCount(); ++node) {
    const auto degree = blocked.GridOf(node).LMax();
    for (auto l = 0; l <= degree; ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto k = SurfaceGrid::Index(l, m);
        const auto value =
            m == 0 ? Complex(Noise(at + k + 3), 0.0)
                   : Complex(Noise(at + k + 3), Noise(at + k + 101));
        spectral.r[at + k] = value;
        padded.r[static_cast<std::size_t>(node) * full + k] = value;
      }
    }
    at += blocked.GridOf(node).CoefficientSize();
  }
  ASSERT_EQ(at, spectral.r.size());

  const auto field = blocked.Synthesise(spectral);
  ASSERT_EQ(field.r.size(), blocked.GridValueCount());
  const auto back = blocked.Analyse(field);
  auto worst = 0.0;
  for (auto c = 0; c < 2; ++c) {
    for (auto i = std::size_t{0}; i < back.c[c].size(); ++i) {
      worst = std::max(worst, std::abs(back.c[c][i] - spectral.c[c][i]));
    }
  }
  for (auto i = std::size_t{0}; i < back.r.size(); ++i) {
    worst = std::max(worst, std::abs(back.r[i] - spectral.r[i]));
  }
  EXPECT_LT(worst, 1e-13);

  // A block's real field, carried to the lmax grid, is the single grid's
  // synthesis of the same coefficients zero padded.
  const auto carried = blocked.OnSurfaceGrid(field.r);
  const auto want = single.Synthesise(padded).r;
  ASSERT_EQ(carried.size(), want.size());
  auto carriedWorst = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    carriedWorst = std::max(carriedWorst, std::abs(carried[i] - want[i]));
  }
  EXPECT_LT(carriedWorst, 1e-13);
  EXPECT_THROW(static_cast<void>(blocked.OnSurfaceGrid(want)),
               std::invalid_argument);
}

TEST(NodeBlocks, ZeroLateralStepIsTheSingleGrids) {
  // Without lateral viscosity the pointwise weights are uniform, so every
  // product is exact on any grid that carries its factors: the blocks
  // change nothing beyond round-off.
  const auto body = Body::Make();
  const auto zero = [](double, double, double) { return 0.0; };
  auto blocks = 0;
  const auto blocked = TwoSteps(body, zero, true, &blocks);
  ASSERT_GE(blocks, 2);
  EXPECT_LT(Difference(blocked, TwoSteps(body, zero, false)), 1e-12);
}

TEST(NodeBlocks, SmoothLateralStepMatchesTheSingleGrid) {
  // With lateral viscosity a block's grid resolves the step weights to
  // about its own degree (the 3/2 rule), where the lmax grid resolves
  // them further: the difference is the weights' spectrum beyond the
  // block's degree, small for a smooth pattern and smaller the smoother.
  const auto body = Body::Make();
  auto differenceFor = [&](double amplitude) {
    const auto delta = [amplitude](double, double theta, double phi) {
      return amplitude * std::sin(theta) * std::cos(theta) * std::cos(phi);
    };
    return Difference(TwoSteps(body, delta, true),
                      TwoSteps(body, delta, false));
  };
  const auto strong = differenceFor(1.15);  // a decade peak to peak
  const auto mild = differenceFor(0.3);
  RecordProperty("difference_strong", std::format("{:.3e}", strong));
  RecordProperty("difference_mild", std::format("{:.3e}", mild));
  EXPECT_LT(strong, 1e-7);  // 3.8e-9 when written
  EXPECT_LT(mild, strong);  // 4.9e-12, the iteration's tolerance
}

TEST(NodeBlocks, AdjointKernelsPairAsTheSingleGrids) {
  // The 3D sweep runs on the blocks as the step does: its log-viscosity
  // kernel, laid out per block, pairs with a smooth direction (each node
  // sampling it on its own grid) as the single grid's does, and its
  // spherical kernel and ice kernels are the single grid's; to round-off
  // without lateral viscosity, to the forward's agreement with it.
  const auto body = Body::Make(1e22);
  ViscoelasticOperator reference(body.rheology, kLmax);
  SeaLevelSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
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
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  auto melt = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < melt.size(); ++i) {
    melt[i] =
        -0.1 * solver.IceDensity() * (1.0 - state.oceanFunction[i]) * ice[i];
  }
  const auto times =
      std::vector<double>{0.0, 2.5 * kyr, 5.0 * kyr, 7.5 * kyr, 10.0 * kyr};
  const auto iceLoad = [&](double t) {
    auto out = melt;
    for (auto& v : out) v *= t / times.back();
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual =
      solver.Grid().FromFunction([](double theta, double phi) {
        return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
      });
  const auto direction = [](double theta, double phi) {
    return std::sin(theta) * std::cos(theta) * std::sin(phi);
  };

  struct Result {
    double pairing = 0.0;
    std::vector<double> spherical;
    std::vector<double> loadDual;
  };
  auto run = [&](const std::function<double(double, double, double)>& delta,
                 bool blocks) {
    Viscoelastic3D op(
        body.rheology, kLmax, delta,
        {.eps = kEps,
         .transform_degrees = blocks ? Degrees() : std::vector<int>{}});
    EXPECT_EQ(op.Coupling().BlockCount() > 1, blocks);
    SeaLevelEvolution3D evolution(
        op, solver, state,
        {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    const auto snapshots = evolution.Run(times, iceLoad);
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
    AdjointSolver3D adjoint(evolution, {.rtol = 1e-12, .coupled_rtol = 1e-12});
    const auto kernels =
        adjoint.IceHistoryKernels(times, measurements, &trajectory);
    const auto& coupling = op.Coupling();
    auto out = Result{};
    for (auto node = 0; node < coupling.NodeCount(); ++node) {
      const auto& grid = coupling.GridOf(node);
      auto product = grid.FromFunction(direction);
      const auto row =
          std::span(kernels.logViscosityKernel3D[0])
              .subspan(coupling.FieldOffset(node), coupling.FieldSizeOf(node));
      for (auto x = std::size_t{0}; x < product.size(); ++x) {
        product[x] *= row[x];
      }
      out.pairing += grid.Integrate(product);
    }
    out.spherical = kernels.itauKernelSpherical[0];
    out.loadDual = kernels.ice.loadDuals.front();
    return out;
  };
  auto relative = [](std::span<const double> a, std::span<const double> b) {
    auto scale = 0.0, diff = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      scale = std::max(scale, std::abs(a[i]));
      diff = std::max(diff, std::abs(a[i] - b[i]));
    }
    return diff / scale;
  };
  auto compare = [&](const std::function<double(double, double, double)>& delta,
                     double tolerance, const char* label) {
    const auto blocked = run(delta, true);
    const auto single = run(delta, false);
    EXPECT_NE(single.pairing, 0.0) << label;
    const auto pairing =
        std::abs(blocked.pairing - single.pairing) / std::abs(single.pairing);
    const auto spherical = relative(blocked.spherical, single.spherical);
    const auto loadDual = relative(blocked.loadDual, single.loadDual);
    RecordProperty(std::format("{}_pairing", label),
                   std::format("{:.3e}", pairing));
    RecordProperty(std::format("{}_spherical", label),
                   std::format("{:.3e}", spherical));
    RecordProperty(std::format("{}_load_dual", label),
                   std::format("{:.3e}", loadDual));
    EXPECT_LT(pairing, tolerance) << label;
    EXPECT_LT(spherical, tolerance) << label;
    EXPECT_LT(loadDual, tolerance) << label;
  };
  // 4e-14 to 6e-14, and 3e-10 to 1.3e-9, when written
  compare([](double, double, double) { return 0.0; }, 1e-12, "zero_lateral");
  compare(
      [](double, double theta, double phi) {
        return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      },
      1e-7, "smooth_lateral");
}
