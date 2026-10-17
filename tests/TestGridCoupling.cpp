// The zero-lateral gate of stage 4 (docs/stage4.md §7a): a random
// degree-basis internal-variable state, pushed through the FULL grid
// pipeline — components, spin synthesis, a pointwise (here uniform)
// multiply, spin analysis, force assembly, degree solves — must reproduce
// the 1D stepper's slaved solutions at machine precision.

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <map>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;
using Complex = std::complex<double>;

namespace {

double Noise(std::size_t k) {
  const auto x = std::sin(12.9898 * static_cast<double>(k + 1)) * 43758.5453;
  return x - std::floor(x) - 0.5;
}

}  // namespace

TEST(Stage4, CouplingsShareOneGrid) {
  constexpr int lmax = 4;
  auto deck =
      Deck::Read(
          std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
          DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  auto model = EarthModel::FromDeck(deck).value();
  auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
  auto material = Material::Create(model, mesh).value();
  const auto& scales = model.ModelScales();
  std::map<std::size_t, RadialValue> viscosity;
  const auto boundaries = model.DimensionlessView().Boundaries();
  for (std::size_t i = 0; i < model.LayerCount(); ++i) {
    if (!model.IsFluid(i) && boundaries[i] >= 3.4799e6 / scales.Length() &&
        boundaries[i + 1] <= 6.1511e6 / scales.Length()) {
      viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
    }
  }
  auto rheology = Rheology::Maxwell(material, viscosity).value();

  // One grid, two couplings; the handle is cheap and the results equal
  // a coupling that built its own.
  SurfaceGrid grid(lmax, {.nmax = 2, .oversampling = 1.5});
  GridCoupling own(rheology, lmax);
  GridCoupling shared1(rheology, lmax, grid, {});
  GridCoupling shared2(rheology, lmax, grid, {});
  EXPECT_EQ(&shared1.Grid().Grid(), &shared2.Grid().Grid());

  auto spectral = own.ZeroSpectral();
  for (std::size_t i = 0; i < spectral.c[0].size(); ++i) {
    spectral.c[0][i] = Complex(Noise(i), Noise(i + 1));
  }
  for (std::size_t i = 0; i < spectral.r.size(); ++i) {
    spectral.r[i] = Noise(3 * i);
  }
  const auto a = own.Synthesise(spectral);
  const auto b = shared1.Synthesise(spectral);
  for (std::size_t i = 0; i < a.c[0].size(); ++i) {
    ASSERT_EQ(a.c[0][i], b.c[0][i]);
  }
  for (std::size_t i = 0; i < a.r.size(); ++i) ASSERT_EQ(a.r[i], b.r[i]);

  // A grid without spin coverage, or at another lmax, is refused.
  EXPECT_THROW(GridCoupling(rheology, lmax, SurfaceGrid(lmax), {}),
               std::invalid_argument);
  EXPECT_THROW(
      GridCoupling(rheology, lmax, SurfaceGrid(lmax + 1, {.nmax = 2}), {}),
      std::invalid_argument);
}

TEST(Stage4, TransformsIntoBuffersAndThreadedPointwise) {
  // The into-buffer transforms (what the step reuses its buffers through)
  // overwrite every value: on buffers full of junk they give bitwise the
  // allocating forms. The pointwise loops are element-wise, so threaded
  // they are bitwise the serial ones.
  constexpr int lmax = 4;
  auto deck =
      Deck::Read(
          std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
          DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  const auto model = EarthModel::FromDeck(deck).value();
  const auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
  const auto material = Material::Create(model, mesh).value();
  const auto& scales = model.ModelScales();
  auto viscosity = std::map<std::size_t, RadialValue>{};
  const auto boundaries = model.DimensionlessView().Boundaries();
  for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
    if (!model.IsFluid(i) && boundaries[i] >= 3.4799e6 / scales.Length() &&
        boundaries[i + 1] <= 6.1511e6 / scales.Length()) {
      viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
    }
  }
  const auto rheology = Rheology::Maxwell(material, viscosity).value();
  const auto serial = GridCoupling(rheology, lmax, {.parallel = false});
  const auto threaded = GridCoupling(rheology, lmax, {.parallel = true});

  auto spectral = serial.ZeroSpectral();
  for (auto c = 0; c < 2; ++c) {
    for (auto i = std::size_t{0}; i < spectral.c[c].size(); ++i) {
      spectral.c[c][i] = Complex(Noise(7 * i + c), Noise(7 * i + c + 3));
    }
  }
  for (auto i = std::size_t{0}; i < spectral.r.size(); ++i) {
    spectral.r[i] = Noise(5 * i + 1);
  }
  const auto field = serial.Synthesise(spectral);
  auto reused = serial.ZeroTensor();
  for (auto& component : reused.c) std::ranges::fill(component, 1e300);
  std::ranges::fill(reused.r, -1e300);
  serial.Synthesise(spectral, reused);
  for (auto c = 0; c < 2; ++c) ASSERT_EQ(reused.c[c], field.c[c]);
  ASSERT_EQ(reused.r, field.r);

  const auto analysed = serial.Analyse(field);
  auto reusedSpectral = serial.ZeroSpectral();
  for (auto& block : reusedSpectral.c) std::ranges::fill(block, 1e300);
  std::ranges::fill(reusedSpectral.r, -1e300);
  serial.Analyse(field, reusedSpectral);
  for (auto c = 0; c < 2; ++c) ASSERT_EQ(reusedSpectral.c[c], analysed.c[c]);
  ASSERT_EQ(reusedSpectral.r, analysed.r);

  const auto saved = omp_get_max_threads();
  omp_set_num_threads(4);
  auto weight = std::vector<double>(field.r.size());
  for (auto i = std::size_t{0}; i < weight.size(); ++i) weight[i] = Noise(i);
  auto factor = std::vector<double>(serial.NodeCount());
  for (auto i = std::size_t{0}; i < factor.size(); ++i)
    factor[i] = Noise(9 * i);
  auto a = field, b = field;
  serial.Multiply(a, weight);
  threaded.Multiply(b, weight);
  auto sa = analysed, sb = analysed;
  serial.ScaleByNode(sa, factor);
  threaded.ScaleByNode(sb, factor);
  omp_set_num_threads(saved);
  for (auto c = 0; c < 2; ++c) {
    ASSERT_EQ(a.c[c], b.c[c]);
    ASSERT_EQ(sa.c[c], sb.c[c]);
  }
  ASSERT_EQ(a.r, b.r);
  ASSERT_EQ(sa.r, sb.r);
}

TEST(Stage4Gate, ZeroLateralPipelineMatchesTheDegreeBasisStepper) {
  constexpr int lmax = 8;
  auto deck =
      Deck::Read(
          std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
          DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  auto model = EarthModel::FromDeck(deck).value();
  auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
  auto material = Material::Create(model, mesh).value();
  const auto& scales = model.ModelScales();
  std::map<std::size_t, RadialValue> viscosity;
  const auto boundaries = model.DimensionlessView().Boundaries();
  for (std::size_t i = 0; i < model.LayerCount(); ++i) {
    if (!model.IsFluid(i) && boundaries[i] >= 3.4799e6 / scales.Length() &&
        boundaries[i + 1] <= 6.1511e6 / scales.Length()) {
      viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
    }
  }
  auto rheology = Rheology::Maxwell(material, viscosity).value();

  // Both memory point sets: the memory nodes (element, point), the 1D
  // state's layout alike, and the unrelaxed solves the operator's own.
  for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss}) {
    SCOPED_TRACE(set == MemoryPointSet::Gll ? "GLL points" : "Gauss points");
    const auto op =
        ViscoelasticOperator(rheology, lmax, {.memory_points = set});
    const auto coupling = GridCoupling(rheology, lmax, {.memory_points = set});
    const auto np = coupling.Points().count;

    // The same deterministic random scalars feed both paths. The 1D state
    // carries 2(2l+1) real columns (Re then Im blocks of the m-columns);
    // the grid path takes the complex columns directly.
    auto vstate = op.ZeroState([](int l) { return 2 * (l + 1); });
    auto spectral = coupling.ZeroSpectral();
    for (int l = 0; l <= lmax; ++l) {
      const auto columns = static_cast<std::size_t>(l + 1);
      std::vector<Complex> M(coupling.NodeCount() * columns, 0.0);
      std::vector<Complex> R = M, S = M;
      auto& m1d = vstate.m.at(l);
      // The 1D state spans the degree's memory window.
      const auto window = op.MemoryElements(l);
      const auto e0 = window.first;
      const auto ne = window.second;
      const auto ncols1d = 2 * columns;
      const auto block = static_cast<std::size_t>(ne) * np * ncols1d;
      for (int node = 0; node < coupling.NodeCount(); ++node) {
        const auto e = coupling.ElementOfNode(node);
        if (e < e0 || e >= e0 + ne) continue;
        const auto q = coupling.NodeInElementOf(node);
        // Random m >= 0 columns of a real field (m = 0 real), fed
        // identically to both paths.
        for (int m = 0; m <= l; ++m) {
          const auto seed = 1000 * l + 31 * m + 7 * node;
          Complex values[3] = {
              l >= 2 ? Complex(Noise(seed), Noise(seed + 1)) : 0.0,
              Complex(Noise(seed + 2), Noise(seed + 3)),
              l >= 1 ? Complex(Noise(seed + 4), Noise(seed + 5)) : 0.0};
          if (m == 0) {
            for (auto& v : values) v = v.real();
          }
          const auto eL = e - e0;
          for (int s = 0; s < 3; ++s) {
            auto& scalars = s == 0 ? M : (s == 1 ? R : S);
            scalars[node * columns + m] = values[s];
            const auto at =
                ((static_cast<std::size_t>(s) * ne + eL) * np + q) * ncols1d;
            m1d[at + m] = values[s].real();
            m1d[at + columns + m] = values[s].imag();
          }
          (void)block;
        }
      }
      coupling.AddScalarState(spectral, l, M, R, S);
    }

    // The 1D reference: the slaved elastic solves of the state.
    const auto reference = op.SolveElastic(vstate);

    // The grid path: synthesis, a pointwise multiply by one (exercising the
    // product stage), analysis, the radial branch-modulus scaling, the
    // force assembly, and the same unrelaxed degree solves.
    auto field = coupling.Synthesise(spectral);
    const std::vector<double> ones(
        static_cast<std::size_t>(coupling.NodeCount()) *
            coupling.Grid().FieldSize(),
        1.0);
    coupling.Multiply(field, ones);
    auto analysed = coupling.Analyse(field);
    std::vector<double> mu(coupling.NodeCount());
    for (int node = 0; node < coupling.NodeCount(); ++node) {
      const auto e = coupling.ElementOfNode(node);
      const auto q = coupling.NodeInElementOf(node);
      mu[node] = op.PointMu(0)[static_cast<std::size_t>(e) * np + q];
    }
    coupling.ScaleByNode(analysed, mu);

    std::map<int, DegreeOperator> systems;
    for (int l = 0; l <= lmax; ++l) {
      systems.emplace(l,
                      DegreeOperator(op.NodalMaterial(), l,
                                     {.point_shear = op.ElasticPointShear()}));
    }
    const auto forces = coupling.SpheroidalForce(
        analysed,
        [&](int l) -> const DegreeOperator& { return systems.at(l); });

    double worst = 0.0, scale = 0.0;
    for (int l = 0; l <= lmax; ++l) {
      const auto& system = systems.at(l);
      const auto ndof = system.DofCount();
      const auto columns = l + 1;
      // Complex force columns -> interleaved real solves.
      std::vector<double> rhs(static_cast<std::size_t>(ndof) * 2 * columns);
      const auto& bc = forces.at(l);
      for (int c = 0; c < columns; ++c) {
        for (int d = 0; d < ndof; ++d) {
          rhs[static_cast<std::size_t>(c) * ndof + d] =
              bc[static_cast<std::size_t>(c) * ndof + d].real();
          rhs[static_cast<std::size_t>(columns + c) * ndof + d] =
              bc[static_cast<std::size_t>(c) * ndof + d].imag();
        }
      }
      const auto x = system.Solve(rhs, 2 * columns);
      const auto& ref = reference.at(l);
      for (std::size_t i = 0; i < x.size(); ++i) {
        scale = std::max(scale, std::abs(ref[i]));
        worst = std::max(worst, std::abs(x[i] - ref[i]));
      }
    }
    EXPECT_GT(scale, 0.0);
    EXPECT_LT(worst, 1e-11 * scale);

    // A purely spheroidal state exerts no toroidal force.
    std::map<int, ToroidalOperator> toroidal;
    for (int l = 1; l <= lmax; ++l) {
      toroidal.emplace(
          l, ToroidalOperator(op.NodalMaterial(), l,
                              {.point_shear = op.ElasticPointShear()}));
    }
    const auto torForces = coupling.ToroidalForce(
        analysed,
        [&](int l) -> const ToroidalOperator& { return toroidal.at(l); });
    double torWorst = 0.0;
    for (const auto& [l, b] : torForces) {
      for (auto v : b) torWorst = std::max(torWorst, std::abs(v));
    }
    double forceScale = 0.0;
    for (const auto& [l, b] : forces) {
      for (auto v : b) forceScale = std::max(forceScale, std::abs(v));
    }
    EXPECT_LT(torWorst, 1e-11 * forceScale);
  }
}

TEST(Stage4, DegreeOneToroidalForceIsTorqueFree) {
  // The memory force of any real tensor is orthogonal to the degree-1
  // rigid rotations (a symmetric stress does no work on a rigid motion):
  // the discrete consistency condition that lets the pinned degree-1
  // toroidal solve solve the full singular system.
  constexpr int lmax = 4;
  auto deck =
      Deck::Read(
          std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
          DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  auto model = EarthModel::FromDeck(deck).value();
  auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
  auto material = Material::Create(model, mesh).value();
  const auto& scales = model.ModelScales();
  std::map<std::size_t, RadialValue> viscosity;
  const auto boundaries = model.DimensionlessView().Boundaries();
  for (std::size_t i = 0; i < model.LayerCount(); ++i) {
    if (!model.IsFluid(i) && boundaries[i] >= 3.4799e6 / scales.Length() &&
        boundaries[i + 1] <= 6.1511e6 / scales.Length()) {
      viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
    }
  }
  auto rheology = Rheology::Maxwell(material, viscosity).value();
  GridCoupling coupling(rheology, lmax);

  // An arbitrary real tensor: any stored spin -2 and -1 values, real spin 0.
  auto field = coupling.ZeroTensor();
  for (auto c = 0; c < 2; ++c) {
    for (std::size_t i = 0; i < field.c[c].size(); ++i) {
      field.c[c][i] = Complex(Noise(7 * i + c), Noise(11 * i + 3 * c + 1));
    }
  }
  for (std::size_t i = 0; i < field.r.size(); ++i) field.r[i] = Noise(5 * i);
  const auto analysed = coupling.Analyse(field);

  std::map<int, ToroidalOperator> toroidal;
  for (int l = 1; l <= lmax; ++l) {
    toroidal.emplace(l, ToroidalOperator(material, l));
  }
  const auto forces = coupling.ToroidalForce(
      analysed,
      [&](int l) -> const ToroidalOperator& { return toroidal.at(l); });
  const auto& degreeOne = toroidal.at(1);
  const auto& b = forces.at(1);
  const auto ndof = static_cast<std::size_t>(degreeOne.DofCount());
  double scale = 0.0;
  for (auto v : b) scale = std::max(scale, std::abs(v));
  ASSERT_GT(scale, 0.0);
  for (int k = 0; k < degreeOne.RegionCount(); ++k) {
    const auto rigid = degreeOne.RigidRotation(k);
    for (std::size_t column = 0; column < 2; ++column) {
      Complex torque = 0.0;
      double size = 0.0;
      for (std::size_t d = 0; d < ndof; ++d) {
        torque += b[column * ndof + d] * rigid[d];
        size += std::abs(b[column * ndof + d]) * rigid[d];
      }
      EXPECT_LT(std::abs(torque), 1e-12 * std::max(size, scale))
          << "region " << k << ", column " << column;
    }
  }
}
