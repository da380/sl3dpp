// The 3D second-order adjoint (continuous): with no lateral perturbation
// its kernels and derivatives are the 1D continuous second-order sweep's
// through the grid pipeline; with lateral structure on, the kernels it
// carries equal the first-order sweep's, the Hessian it gives is
// symmetric and adjoint to the 3D second-order tangent-linear run at the
// commute level, improving under step refinement, and its derivatives
// match central differences of the continuous kernels at that level.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/TangentLinear3D.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 8;

double Cap(double theta, double phi, double colat0, double lon0,
           double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

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
        viscosity.emplace(i,
                          viscositySI / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    ViscoelasticOperator reference(rheology, kLmax);
    SeaLevelSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
                          SeaLevelParameters{});
    Fingerprint fingerprint(kLmax, reference.ElasticLoveNumbers());
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

  [[nodiscard]] double Kyr() const {
    return 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  }
  [[nodiscard]] double Metre() const {
    return 1.0 / solver.SolverScales().Length();
  }
  [[nodiscard]] std::vector<double> MeltLoad(double fraction) const {
    auto load = std::vector<double>(iceNd.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                fraction * iceNd[i];
    }
    return load;
  }
  [[nodiscard]] std::vector<double> Smooth(std::mt19937& rng,
                                           double amplitude) const {
    const auto& grid = solver.Grid();
    std::normal_distribution<double> normal;
    auto coeffs = grid.ZeroCoefficients();
    for (auto l = 0; l <= kLmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        coeffs[SurfaceGrid::Index(l, m)] = {normal(rng),
                                            m == 0 ? 0.0 : normal(rng)};
      }
    }
    auto field = grid.Synthesise(coeffs);
    for (auto& v : field) v *= amplitude;
    return field;
  }
  [[nodiscard]] double Surface(std::span<const double> f,
                               std::span<const double> g) const {
    const auto a = solver.Radius();
    auto fg = std::vector<double>(f.size());
    for (auto i = std::size_t{0}; i < f.size(); ++i) fg[i] = f[i] * g[i];
    return a * a * solver.Grid().Integrate(fg);
  }
};

std::vector<double> OnCoupling(const GridCoupling& coupling,
                               const std::function<double(double, double)>& f) {
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto values = coupling.GridOf(rn).FromFunction(f);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x) out[at + x] = values[x];
  }
  return out;
}

/// The grid pairing of a 3D kernel with an itau direction: the angular
/// integral per memory node, summed.
double GridPairing(const GridCoupling& coupling,
                   const std::vector<std::vector<double>>& kernel,
                   const std::vector<std::vector<double>>& direction) {
  auto total = 0.0;
  for (auto k = std::size_t{0}; k < kernel.size(); ++k) {
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto at = coupling.FieldOffset(rn);
      const auto size = coupling.FieldSizeOf(rn);
      auto product = std::vector<double>(size);
      for (auto x = std::size_t{0}; x < size; ++x) {
        product[x] = kernel[k][at + x] * direction[k][at + x];
      }
      total += coupling.GridOf(rn).Integrate(product);
    }
  }
  return total;
}

/// The frozen-shoreline load of a per-node thickness history relative to
/// its first node.
std::function<std::vector<double>(double)> FrozenLoad(
    const Scene& scene, const std::vector<double>& times,
    const std::vector<std::vector<double>>& thickness) {
  auto loads = std::vector<std::vector<double>>{};
  const auto& C = scene.state.oceanFunction;
  const auto rhoI = scene.solver.IceDensity();
  for (const auto& T : thickness) {
    auto load = std::vector<double>(T.size());
    for (auto i = std::size_t{0}; i < T.size(); ++i) {
      load[i] = rhoI * (1.0 - C[i]) * (T[i] - thickness[0][i]);
    }
    loads.push_back(std::move(load));
  }
  return [times, loads](double t) {
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) {
        return loads[n];
      }
    }
    throw std::runtime_error("the test history is read off its nodes");
  };
}

double MaxAbs(std::span<const double> f) {
  auto out = 0.0;
  for (auto v : f) out = std::max(out, std::abs(v));
  return out;
}

double MaxDiff(std::span<const double> a, std::span<const double> b) {
  auto out = 0.0;
  for (auto i = std::size_t{0}; i < a.size(); ++i) {
    out = std::max(out, std::abs(a[i] - b[i]));
  }
  return out;
}

/// A direction: thickness at every node and an itau perturbation.
struct Direction3D {
  TangentPerturbation3D tangent;
  std::vector<std::vector<double>> itau;  // the same as tangent.itau
};

/// Its pairing with a kernel object (kernels or derivatives).
template <class Ice, class Kernels>
double Pairing(const Scene& scene, const GridCoupling& coupling,
               const Ice& ice, const Kernels& itau3D,
               const Direction3D& d) {
  auto total = 0.0;
  for (auto n = std::size_t{0}; n < d.tangent.iceThickness.size(); ++n) {
    total += scene.Surface(ice.thickness[n], d.tangent.iceThickness[n]);
  }
  total += GridPairing(coupling, itau3D, d.itau);
  return total;
}

}  // namespace

TEST(SecondOrderAdjoint3D, ZeroLateralMatchesThe1DSweep) {
  // With delta = 0 at the GLL points the 3D second-order sweep's kernels
  // and derivatives are the 1D continuous second-order sweep's: the
  // thickness views and the spherical reductions of the itau kernel and
  // its derivative, along a direction angle-uniform in itau.
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(3);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * scene.Kyr());
  auto thickness = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    thickness.push_back(scene.Smooth(rng, 100.0 * scene.Metre()));
  }
  const auto iceLoad = FrozenLoad(scene, times, thickness);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[5].seaLevelDual = scene.Smooth(rng, 1.0);
  measurements[3].displacementDual = scene.Smooth(rng, -0.6);
  auto deltaIce = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    deltaIce.push_back(scene.Smooth(rng, 20.0 * scene.Metre()));
  }
  const auto& base = scene.rheology.BranchAt(0);
  const auto& mesh = scene.material.Mesh();
  auto top = 0.0;
  for (auto v : base.itau) top = std::max(top, v);
  auto ditau1 = std::vector<double>(base.itau.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      if (base.itau[i] > 0.0) {
        const auto r = mesh.Radius(e, q);
        ditau1[i] = 0.1 * (0.3 + 0.7 * r * r) * top;
      }
    }
  }

  // 1D.
  ViscoelasticOperator op1(scene.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution evolution1(op1, scene.solver, scene.state,
                               {.rtol = 1e-12, .record_stride = 1});
  const auto trajectory1 =
      ForwardTrajectory::FromSnapshots(evolution1.Run(times, iceLoad));
  TangentPerturbation v1;
  v1.iceThickness = deltaIce;
  v1.itau = {ditau1};
  const auto run1 = TangentLinearSolver(evolution1, {.rtol = 1e-12, .record = true})
                        .Run(times, v1, &trajectory1);
  const auto [kernels1, derivatives1] =
      AdjointSolver(evolution1, {.rtol = 1e-12})
          .SecondOrderKernels(times, measurements, run1, v1.itau, trajectory1);

  // 3D with delta = 0.
  Viscoelastic3D op3(scene.rheology, kLmax,
                     [](double, double, double) { return 0.0; },
                     {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution3D evolution3(
      op3, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto trajectory3 =
      ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, iceLoad));
  const auto& coupling = op3.Coupling();
  auto ditau3 = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full =
        static_cast<std::size_t>(coupling.ElementOfNode(rn)) * mesh.NGll() +
        coupling.NodeInElementOf(rn);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < coupling.FieldSizeOf(rn); ++x) {
      ditau3[at + x] = ditau1[full];
    }
  }
  TangentPerturbation3D v3;
  v3.iceThickness = deltaIce;
  v3.itau = {ditau3};
  const auto run3 =
      TangentLinearSolver3D(evolution3,
                            {.rtol = 1e-12, .coupled_rtol = 1e-12, .record = true})
          .Run(times, v3, &trajectory3);
  const auto [kernels3, derivatives3] =
      AdjointSolver3D(evolution3, {.rtol = 1e-12, .coupled_rtol = 1e-12})
          .SecondOrderKernels(times, measurements, run3, v3.itau, trajectory3);

  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_LT(MaxDiff(kernels3.ice.thickness[n], kernels1.thickness[n]),
              1e-6 * MaxAbs(kernels1.thickness[n]))
        << "kernel node " << n;
    EXPECT_LT(MaxDiff(derivatives3.ice.thickness[n], derivatives1.thickness[n]),
              1e-6 * MaxAbs(derivatives1.thickness[n]))
        << "derivative node " << n;
  }
  auto scale = 0.0, worst = 0.0, dScale = 0.0, dWorst = 0.0;
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full =
        static_cast<std::size_t>(coupling.ElementOfNode(rn)) * mesh.NGll() +
        coupling.NodeInElementOf(rn);
    scale = std::max(scale, std::abs(kernels1.itauKernel[0][full]));
    worst = std::max(worst, std::abs(kernels3.itauKernelSpherical[0][rn] -
                                     kernels1.itauKernel[0][full]));
    dScale = std::max(dScale, std::abs(derivatives1.itauKernel[0][full]));
    dWorst = std::max(dWorst, std::abs(derivatives3.itauKernelSpherical[0][rn] -
                                       derivatives1.itauKernel[0][full]));
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, 1e-6 * scale);
  EXPECT_GT(dScale, 0.0);
  EXPECT_LT(dWorst, 1e-6 * dScale);
}

TEST(SecondOrderAdjoint3D, LateralHessianIsSymmetricAndAdjointToTheSecondOrderRun) {
  // With lateral structure on: the carried kernels equal the first-order
  // sweep's to rounding; <H u, w> against <H w, u>, and against
  // <g, F''[u, w]> from the 3D second-order tangent-linear run, agree at
  // the continuous formulation's commute level and improve as the steps
  // are halved; and the derivatives match central differences of the
  // continuous kernels along v at that level.
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(11);
  const auto fieldT = scene.Smooth(rng, 100.0 * scene.Metre());
  const auto fieldU = scene.Smooth(rng, 20.0 * scene.Metre());
  const auto fieldW = scene.Smooth(rng, 20.0 * scene.Metre());
  const auto g = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  auto baseDelta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  auto pU = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto pW = [](double theta, double phi) {
    return 0.5 * Cap(theta, phi, 2.4, 2.0, 0.8) + 0.3 * std::cos(theta);
  };
  const auto span = 10.0 * scene.Kyr();
  auto symmetryGaps = std::vector<double>{}, adjointGaps = std::vector<double>{},
       fdGaps = std::vector<double>{};
  for (const auto nodes : {9, 17}) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
    auto scaled = [&](const std::vector<double>& field,
                      const std::function<double(double)>& profile) {
      auto out = std::vector<std::vector<double>>{};
      for (auto t : times) {
        auto f = field;
        for (auto& x : f) x *= profile(t / span);
        out.push_back(std::move(f));
      }
      return out;
    };
    const auto thickness = scaled(fieldT, [](double s) { return 1.0 - s; });
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[times.size() - 1].seaLevelDual = g;

    auto make = [&](double eps, const std::function<double(double, double)>& p,
                    Viscoelastic3D** opOut, SeaLevelEvolution3D** evOut) {
      *opOut = new Viscoelastic3D(scene.rheology, kLmax,
                                  [&, eps](double r, double theta, double phi) {
                                    return baseDelta(r, theta, phi) + eps * p(theta, phi);
                                  });
      *evOut = new SeaLevelEvolution3D(
          **opOut, scene.solver, scene.state,
          {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    };
    Viscoelastic3D* op = nullptr;
    SeaLevelEvolution3D* evolution = nullptr;
    make(0.0, pU, &op, &evolution);
    const auto& coupling = op->Coupling();
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(
        evolution->Run(times, FrozenLoad(scene, times, thickness)));
    auto direction = [&](const std::vector<double>& field,
                         const std::function<double(double)>& profile,
                         const std::function<double(double, double)>& p) {
      Direction3D d;
      d.tangent.iceThickness = scaled(field, profile);
      d.itau = TangentLinearSolver3D::ItauOfLogViscosity(*op, OnCoupling(coupling, p));
      d.tangent.itau = d.itau;
      return d;
    };
    const auto u = direction(fieldU, [](double s) { return s; }, pU);
    const auto w = direction(fieldW, [](double s) { return (1.0 - s) * (1.0 - s); }, pW);
    TangentLinearSolver3D tangent(
        *evolution, {.rtol = 1e-12, .coupled_rtol = 1e-12, .record = true});
    const auto runU = tangent.Run(times, u.tangent, &trajectory);
    const auto runW = tangent.Run(times, w.tangent, &trajectory);
    AdjointSolver3D adjoint(*evolution, {.rtol = 1e-12, .coupled_rtol = 1e-12});
    const auto first = adjoint.IceHistoryKernels(times, measurements, &trajectory);
    const auto [kernelsU, alongU] =
        adjoint.SecondOrderKernels(times, measurements, runU, u.itau, trajectory);
    const auto [kernelsW, alongW] =
        adjoint.SecondOrderKernels(times, measurements, runW, w.itau, trajectory);
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      EXPECT_LT(MaxDiff(kernelsU.ice.thickness[n], first.ice.thickness[n]),
                1e-10 * MaxAbs(first.ice.thickness[n]) + 1e-300);
    }
    EXPECT_LT(MaxDiff(kernelsU.itauKernel3D[0], first.itauKernel3D[0]),
              1e-10 * MaxAbs(first.itauKernel3D[0]));

    const auto huW = Pairing(scene, coupling, alongU.ice, alongU.itauKernel3D, w);
    const auto hwU = Pairing(scene, coupling, alongW.ice, alongW.itauKernel3D, u);
    symmetryGaps.push_back(std::abs(huW - hwU) / std::abs(huW));

    SecondOrderDirection3D uw;
    uw.itauU = u.itau;
    uw.itauV = w.itau;
    uw.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(
        *op, OnCoupling(coupling, pU), OnCoupling(coupling, pW));
    const auto second = tangent.RunSecondOrder(times, runU, runW, uw, &trajectory);
    const auto forward = scene.Surface(g, second.back().seaLevelChange);
    // F''[u, w] of the log parameterisation carries the uv term, whose
    // dual is the first-order kernel paired with itau_uv.
    const auto uvTerm = GridPairing(coupling, first.itauKernel3D, uw.itauUV);
    adjointGaps.push_back(std::abs(forward - uvTerm - huW) / std::abs(forward));

    // Central differences of the continuous kernels along u (delta moved
    // by eps p_u, the thickness by eps delta I_u) paired with w.
    auto gradientAt = [&](double eps) {
      Viscoelastic3D* movedOp = nullptr;
      SeaLevelEvolution3D* movedEv = nullptr;
      make(eps, pU, &movedOp, &movedEv);
      auto moved = thickness;
      for (auto n = std::size_t{0}; n < moved.size(); ++n) {
        for (auto i = std::size_t{0}; i < moved[n].size(); ++i) {
          moved[n][i] += eps * u.tangent.iceThickness[n][i];
        }
      }
      const auto movedTrajectory = ForwardTrajectory3D::FromSnapshots(
          movedEv->Run(times, FrozenLoad(scene, times, moved)));
      const auto kernels =
          AdjointSolver3D(*movedEv, {.rtol = 1e-12, .coupled_rtol = 1e-12})
              .IceHistoryKernels(times, measurements, &movedTrajectory);
      // w's itau direction at the moved operator: -itau(eps) p_w.
      Direction3D wAt = w;
      wAt.itau = TangentLinearSolver3D::ItauOfLogViscosity(
          *movedOp, OnCoupling(movedOp->Coupling(), pW));
      const auto value =
          Pairing(scene, movedOp->Coupling(), kernels.ice, kernels.itauKernel3D, wAt);
      delete movedEv;
      delete movedOp;
      return value;
    };
    const auto eps = 0.05;
    const auto fd = (gradientAt(eps) - gradientAt(-eps)) / (2.0 * eps);
    // The derivative along u of <K, w(eps)> carries the uv term too.
    fdGaps.push_back(std::abs(fd - uvTerm - huW) / std::abs(fd));
    delete evolution;
    delete op;
  }
  for (const auto* gaps : {&symmetryGaps, &adjointGaps, &fdGaps}) {
    EXPECT_LT((*gaps)[0], 0.1) << (*gaps)[0] << " " << (*gaps)[1];
    EXPECT_LT((*gaps)[1], (*gaps)[0]) << (*gaps)[0] << " " << (*gaps)[1];
  }
}
