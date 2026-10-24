// Internal variables at the Gauss points in the 3D path (docs/design.md
// §5): the coupling's radial memory nodes are the
// elements' interior Gauss points, ~20% fewer than the GLL nodes; with no
// lateral perturbation a 3D step, a coupled run and its continuous kernels
// reproduce the 1D operator on the same points (the zero-lateral gate,
// now for either point set); with lateral structure on, the two point sets
// agree to discretisation level and the log-viscosity kernel still meets
// finite differences.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <map>
#include <span>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/MemoryPoints.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using Complex = std::complex<double>;

namespace {

constexpr auto kLmax = 8;

double Noise(std::size_t k) {
  const auto x = std::sin(12.9898 * static_cast<double>(k + 1)) * 43758.5453;
  return x - std::floor(x) - 0.5;
}

double Cap(double theta, double phi, double colat0, double lon0, double width) {
  const auto c = std::cos(theta) * std::cos(colat0) +
                 std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

double Lateral(double, double theta, double phi) {
  return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
}

double Uniform(double, double, double) { return 0.0; }

struct Scene {
  EarthModel model;
  Material material;
  Rheology rheology;
  SeaLevelSolver solver;
  SeaLevelSolver::State state;
  std::vector<double> iceNd;

  static Scene Make(double viscositySI) {
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
    const auto elastic = ViscoelasticOperator(rheology, kLmax);
    auto solver = SeaLevelSolver(
        SurfaceGrid(kLmax), elastic.ElasticLoveNumbers(), SeaLevelParameters{});
    auto fingerprint = Fingerprint(kLmax, elastic.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto ice = toNd(fingerprint.IceThickness());
    auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
    return Scene{std::move(model),  std::move(material), std::move(rheology),
                 std::move(solver), std::move(state),    std::move(ice)};
  }

  [[nodiscard]] std::vector<double> MeltLoad(double fraction) const {
    auto load = std::vector<double>(iceNd.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                fraction * iceNd[i];
    }
    return load;
  }

  [[nodiscard]] double Surface(std::span<const double> f,
                               std::span<const double> g) const {
    const auto a = solver.Radius();
    auto fg = std::vector<double>(f.size());
    for (auto i = std::size_t{0}; i < f.size(); ++i) fg[i] = f[i] * g[i];
    return a * a * solver.Grid().Integrate(fg);
  }

  [[nodiscard]] std::vector<double> TimesKyr(double span,
                                             std::size_t npts) const {
    const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(span * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * kyr);
    }
    return times;
  }
};

Viscoelastic3D Operator3D(const Rheology& rheology, MemoryPointSet set,
                          double (*delta)(double, double, double)) {
  return Viscoelastic3D(rheology, kLmax, delta, {.memory_points = set});
}

/// A deterministic constant load: m >= 0 columns of a real field.
std::vector<Complex> LoadColumns(int l) {
  auto columns = std::vector<Complex>(l + 1);
  for (auto m = 0; m <= l; ++m) {
    columns[m] = m == 0 ? Complex(Noise(97 * l), 0.0)
                        : Complex(Noise(97 * l + m), Noise(55 * l + m));
  }
  return columns;
}

double MaxRelative(const std::map<int, std::vector<Complex>>& a,
                   const std::map<int, std::vector<Complex>>& b) {
  auto scale = 0.0, diff = 0.0;
  for (const auto& [l, columns] : b) {
    for (auto c = std::size_t{0}; c < columns.size(); ++c) {
      scale = std::max(scale, std::abs(columns[c]));
      diff = std::max(diff, std::abs(a.at(l)[c] - columns[c]));
    }
  }
  return diff / scale;
}

}  // namespace

TEST(GaussMemory3D, FewerNodesAndTheZeroLateralStepGate) {
  // With delta = 0 the 3D step is the 1D two-phase step on the same
  // points, through the grid pipeline: surface responses and the grid
  // memory (against the synthesised degree-basis state) to round-off —
  // tighter than the 1-2e-12 by which the 1D GLL step differs here, so
  // the gate tells the point sets apart.
  const auto scene = Scene::Make(1e21);
  const auto& mesh = scene.material.Mesh();
  const auto tau = scene.rheology.MinRelaxationTime();

  const auto gll = Operator3D(scene.rheology, MemoryPointSet::Gll, Uniform);
  const auto op3 = Operator3D(scene.rheology, MemoryPointSet::Gauss, Uniform);
  const auto& coupling = op3.Coupling();
  EXPECT_EQ(coupling.NodeCount() * mesh.NGll(),
            gll.Coupling().NodeCount() * (mesh.NGll() - 1));

  const auto op1 = ViscoelasticOperator(
      scene.rheology, kLmax, {.memory_points = MemoryPointSet::Gauss});
  const auto np = op1.Points().count;
  const auto load3 = [](int l, double) { return LoadColumns(l); };
  const auto load1 = [](int l, double) {
    const auto columns = LoadColumns(l);
    auto real = std::vector<double>(2 * columns.size());
    for (auto c = std::size_t{0}; c < columns.size(); ++c) {
      real[c] = columns[c].real();
      real[columns.size() + c] = columns[c].imag();
    }
    return real;
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto s1 = op1.ZeroState([](int l) { return 2 * (l + 1); });
    auto s3 = op3.ZeroState();
    auto x1 = std::map<int, std::vector<double>>{};
    auto x3 = Viscoelastic3D::Solutions{};
    for (auto step = 0; step < 2; ++step) {
      auto [n1, sols1] = op1.Step(s1, 0.7 * tau, scheme, load1);
      s1 = std::move(n1);
      x1 = std::move(sols1);
      auto result = op3.Step(s3, 0.7 * tau, scheme, load3);
      EXPECT_TRUE(result.info.converged);
      EXPECT_LE(result.info.iterations, 3);
      s3 = std::move(result.state);
      x3 = std::move(result.solutions);
    }

    const auto u1 = op1.Surface(x1, Component::U);
    const auto u3 = op3.Surface(x3, Component::U);
    auto reference = std::map<int, std::vector<Complex>>{};
    for (auto l = 0; l <= kLmax; ++l) {
      const auto columns = static_cast<std::size_t>(l + 1);
      auto& out = reference[l];
      for (auto c = std::size_t{0}; c < columns; ++c) {
        out.emplace_back(u1.at(l)[c], u1.at(l)[columns + c]);
      }
    }
    EXPECT_LT(MaxRelative(u3, reference), 1e-13)
        << "scheme " << static_cast<int>(scheme);

    // The grid memory against the synthesised degree-basis state.
    auto spectral = coupling.ZeroSpectral();
    for (auto l = 0; l <= kLmax; ++l) {
      const auto columns = static_cast<std::size_t>(l + 1);
      auto M = std::vector<Complex>(coupling.NodeCount() * columns, 0.0);
      auto R = M, S = M;
      // The 1D state spans the degree's memory window.
      const auto window = op1.MemoryElements(l);
      const auto e0 = window.first;
      const auto ne = window.second;
      const auto& m1 = s1.m.at(l);
      const auto ncols = 2 * columns;
      for (auto node = 0; node < coupling.NodeCount(); ++node) {
        const auto e = coupling.ElementOfNode(node);
        if (e < e0 || e >= e0 + ne) continue;
        const auto p = coupling.NodeInElementOf(node);
        for (auto c = std::size_t{0}; c < columns; ++c) {
          auto scalar = [&](int s) {
            const auto at =
                ((static_cast<std::size_t>(s) * ne + (e - e0)) * np + p) *
                ncols;
            return Complex(m1[at + c], m1[at + columns + c]);
          };
          M[node * columns + c] = scalar(0);
          R[node * columns + c] = scalar(1);
          S[node * columns + c] = scalar(2);
        }
      }
      coupling.AddScalarState(spectral, l, M, R, S);
    }
    const auto expected = coupling.Synthesise(spectral);
    auto mScale = 0.0, mDiff = 0.0;
    for (auto c = 0; c < 2; ++c) {
      for (auto i = std::size_t{0}; i < expected.c[c].size(); ++i) {
        mScale = std::max(mScale, std::abs(expected.c[c][i]));
        mDiff = std::max(mDiff, std::abs(s3.m[0].c[c][i] - expected.c[c][i]));
      }
    }
    for (auto i = std::size_t{0}; i < expected.r.size(); ++i) {
      mScale = std::max(mScale, std::abs(expected.r[i]));
      mDiff = std::max(mDiff, std::abs(s3.m[0].r[i] - expected.r[i]));
    }
    EXPECT_GT(mScale, 0.0);
    EXPECT_LT(mDiff, 1e-12 * mScale) << "scheme " << static_cast<int>(scheme);
  }
}

TEST(GaussMemory3D, LateralStepsAgreeWithGll) {
  // With lateral structure on, the two point sets are two quadratures of
  // the same memory force: their steps agree to discretisation level, and
  // the within-step iteration converges alike (12 passes a step at this
  // contrast, for either).
  const auto scene = Scene::Make(1e21);
  const auto tau = scene.rheology.MinRelaxationTime();
  const auto load = [](int l, double) { return LoadColumns(l); };
  auto surface =
      std::map<MemoryPointSet, std::map<int, std::vector<Complex>>>{};
  for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss}) {
    const auto op = Operator3D(scene.rheology, set, Lateral);
    auto state = op.ZeroState();
    auto solutions = Viscoelastic3D::Solutions{};
    for (auto step = 0; step < 4; ++step) {
      auto result =
          op.Step(state, 2.0 * tau, Scheme::ExponentialTrapezoid, load);
      EXPECT_TRUE(result.info.converged);
      state = std::move(result.state);
      solutions = std::move(result.solutions);
    }
    surface[set] = op.Surface(solutions, Component::U);
  }
  const auto gap =
      MaxRelative(surface[MemoryPointSet::Gauss], surface[MemoryPointSet::Gll]);
  EXPECT_GT(gap, 0.0);  // two discretisations, not one
  EXPECT_LT(gap, 1e-9);
}

TEST(GaussMemory3D, ZeroLateralCoupledRunAndKernelsMatchThe1DSweep) {
  // The coupled 3D run with delta = 0 and its continuous adjoint against
  // the 1D ones on the same Gauss points: sea level, ice kernels, and the
  // spherically reduced itau kernel carried to the nodes as the 1D sweep
  // carries its own.
  const auto scene = Scene::Make(1e21);
  const auto times = scene.TimesKyr(10.0, 6);
  const auto load = scene.MeltLoad(-0.1);
  auto iceLoad = [&](double t) {
    auto out = std::vector<double>(load.size());
    const auto ramp = t / times.back();
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] = ramp * load[i];
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[5].seaLevelDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  measurements[3].displacementDual = scene.solver.Grid().FromFunction(
      [](double t, double p) { return -0.6 * Cap(t, p, 2.2, 3.9, 0.6); });

  const auto op1 = ViscoelasticOperator(
      scene.rheology, kLmax, {.memory_points = MemoryPointSet::Gauss});
  auto evolution1 = SeaLevelEvolution(op1, scene.solver, scene.state,
                                      {.rtol = 1e-12, .record_stride = 1});
  const auto snapshots1 = evolution1.Run(times, iceLoad);
  const auto trajectory1 = ForwardTrajectory::FromSnapshots(snapshots1);
  const auto adjoint1 = AdjointSolver(
      evolution1, {.method = AdjointMethod::Continuous, .rtol = 1e-12});
  const auto kernels1 =
      adjoint1.IceHistoryKernels(times, measurements, &trajectory1);

  const auto op3 = Operator3D(scene.rheology, MemoryPointSet::Gauss, Uniform);
  auto evolution3 = SeaLevelEvolution3D(
      op3, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto snapshots3 = evolution3.Run(times, iceLoad);
  const auto trajectory3 = ForwardTrajectory3D::FromSnapshots(snapshots3);
  const auto adjoint3 =
      AdjointSolver3D(evolution3, {.rtol = 1e-12, .coupled_rtol = 1e-12});
  const auto kernels3 =
      adjoint3.IceHistoryKernels(times, measurements, &trajectory3);

  auto scale = 0.0, worst = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    for (auto i = std::size_t{0}; i < snapshots1[n].seaLevelChange.size();
         ++i) {
      scale = std::max(scale, std::abs(snapshots1[n].seaLevelChange[i]));
      worst = std::max(worst, std::abs(snapshots3[n].seaLevelChange[i] -
                                       snapshots1[n].seaLevelChange[i]));
    }
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, 1e-13 * scale);

  scale = 0.0;
  worst = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    for (auto i = std::size_t{0}; i < kernels1.loadDuals[n].size(); ++i) {
      scale = std::max(scale, std::abs(kernels1.loadDuals[n][i]));
      worst = std::max(worst, std::abs(kernels3.ice.loadDuals[n][i] -
                                       kernels1.loadDuals[n][i]));
    }
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, 1e-12 * scale);

  // The spherical reduction lives at the memory nodes: laid out over the
  // whole mesh's points and carried to the nodes by the reference.
  const auto& coupling = op3.Coupling();
  const auto np = coupling.Points().count;
  const auto pointCount =
      static_cast<std::size_t>(scene.material.Mesh().ElementCount()) * np;
  auto atPoints = std::vector<std::vector<double>>{};
  for (const auto& spherical : kernels3.itauKernelSpherical) {
    auto full = std::vector<double>(pointCount, 0.0);
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      full[static_cast<std::size_t>(coupling.ElementOfNode(rn)) * np +
           coupling.NodeInElementOf(rn)] = spherical[rn];
    }
    atPoints.push_back(std::move(full));
  }
  const auto nodal = op3.Reference().ItauKernelsAtNodes(std::move(atPoints));
  for (auto k = std::size_t{0}; k < nodal.size(); ++k) {
    auto kScale = 0.0, kWorst = 0.0;
    for (auto i = std::size_t{0}; i < nodal[k].size(); ++i) {
      kScale = std::max(kScale, std::abs(kernels1.itauKernel[k][i]));
      kWorst =
          std::max(kWorst, std::abs(nodal[k][i] - kernels1.itauKernel[k][i]));
    }
    EXPECT_GT(kScale, 0.0);
    EXPECT_LT(kWorst, 1e-10 * kScale) << "branch " << k;
  }
}

TEST(GaussMemory3D, LateralLogViscosityKernelAgainstFiniteDifferences) {
  // The 3D log-viscosity kernel at the Gauss points with lateral structure
  // on: paired with an analytic direction it meets central differences of
  // the coupled forward to the continuous adjoint's commute error, as at
  // the GLL nodes, and the two point sets' predictions agree.
  const auto scene = Scene::Make(1e22);
  const auto times = scene.TimesKyr(10.0, 5);
  const auto load = scene.MeltLoad(-0.1);
  auto iceLoad = [&](double t) {
    auto out = std::vector<double>(load.size());
    const auto ramp = t / times.back();
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] = ramp * load[i];
    return out;
  };
  auto direction = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual =
      scene.solver.Grid().FromFunction(
          [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });

  auto run = [&](MemoryPointSet set, double eps, Kernels3D* kernelsOut) {
    const auto op = Viscoelastic3D(scene.rheology, kLmax,
                                   [&](double r, double theta, double phi) {
                                     return Lateral(r, theta, phi) +
                                            eps * direction(theta, phi);
                                   },
                                   {.memory_points = set});
    auto evolution =
        SeaLevelEvolution3D(op, scene.solver, scene.state,
                            {.rtol = 1e-12,
                             .coupled_rtol = 1e-12,
                             .record_stride = kernelsOut != nullptr ? 1 : 0});
    const auto snapshots = evolution.Run(times, iceLoad);
    if (kernelsOut != nullptr) {
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      const auto adjoint =
          AdjointSolver3D(evolution, {.rtol = 1e-12, .coupled_rtol = 1e-12});
      *kernelsOut = adjoint.IceHistoryKernels(times, measurements, &trajectory);
    }
    return scene.Surface(measurements[times.size() - 1].seaLevelDual,
                         snapshots.back().seaLevelChange);
  };

  auto predicted = std::map<MemoryPointSet, double>{};
  for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss}) {
    auto kernels = Kernels3D{};
    (void)run(set, 0.0, &kernels);
    const auto probe = Operator3D(scene.rheology, set, Uniform);
    const auto& coupling = probe.Coupling();
    const auto& grid = coupling.Grid();
    const auto p = grid.FromFunction(direction);
    const auto gridF = grid.FieldSize();
    auto sum = 0.0;
    auto product = std::vector<double>(gridF);
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto row =
          std::span(kernels.logViscosityKernel3D[0])
              .subspan(static_cast<std::size_t>(rn) * gridF, gridF);
      for (auto x = std::size_t{0}; x < gridF; ++x) product[x] = row[x] * p[x];
      sum += grid.Integrate(product);
    }
    predicted[set] = sum;
  }
  const auto eps = 0.1;
  const auto fd = (run(MemoryPointSet::Gauss, eps, nullptr) -
                   run(MemoryPointSet::Gauss, -eps, nullptr)) /
                  (2.0 * eps);
  const auto error =
      std::abs(fd - predicted[MemoryPointSet::Gauss]) / std::abs(fd);
  const auto setGap = std::abs(predicted[MemoryPointSet::Gauss] -
                               predicted[MemoryPointSet::Gll]) /
                      std::abs(predicted[MemoryPointSet::Gll]);
  EXPECT_LT(error, 0.1);
  EXPECT_LT(setGap, 1e-9);
}
