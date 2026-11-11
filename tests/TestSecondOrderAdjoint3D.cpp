// The 3D second-order adjoint: with no lateral perturbation its kernels
// and derivatives are the 1D second-order sweep's (by the same method)
// through the grid pipeline; with lateral structure on, the kernels it
// carries equal the first-order sweep's, the Hessian it gives is
// symmetric and adjoint to the 3D second-order tangent-linear run — at
// the commute level for the continuous sweep, improving under step
// refinement, and to the solvers' tolerance for the discrete one — and
// its derivatives match central differences of the kernels at that level.

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <tuple>
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

/// With the memory points at the GLL nodes, as in these tests, the points
/// are the nodes: a nodal direction as one for the time-stepped problem,
/// and back.
[[maybe_unused]] PointMaterialDirection AtGll(const MaterialDirection& d) {
  return {.kappa = d.kappa, .mu = d.mu};
}
[[maybe_unused]] MaterialDirection Nodal(const PointMaterialDirection& d) {
  return {.kappa = d.kappa, .mu = d.mu};
}

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
  GRDSolver solver;
  GRDSolver::State state;
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
    GRDSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
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

/// The volume pairing of a 3D kernel with an itau direction, per branch,
/// summed.
double GridPairing(const GridCoupling& coupling,
                   const std::vector<std::vector<double>>& kernel,
                   const std::vector<std::vector<double>>& direction) {
  auto total = 0.0;
  for (auto k = std::size_t{0}; k < kernel.size(); ++k) {
    total += VolumePairing(coupling, kernel[k], direction[k]);
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

/// A radial profile times an angular function over the grid values, in
/// units of a nodal modulus.
std::vector<double> LateralModulus(const GridCoupling& coupling,
                                   const Material& material,
                                   std::span<const double> modulus,
                                   const std::function<double(double)>& radial,
                                   const std::function<double(double, double)>& angular) {
  const auto ngll = material.Mesh().NGll();
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full = static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
                      coupling.NodeInElementOf(rn);
    const auto r = coupling.RadiusOfNode(rn);
    const auto values = coupling.GridOf(rn).FromFunction(
        [&](double theta, double phi) {
          return modulus[full] * radial(r) * angular(theta, phi);
        });
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x) out[at + x] = values[x];
  }
  return out;
}

/// A nodal field over the material's mesh on the coupling's grid values,
/// angle-uniform.
std::vector<double> UniformOnCoupling(const GridCoupling& coupling,
                                      const Material& material,
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

/// A smooth nodal direction in kappa and mu on the relaxing elements, as
/// nodal fields over the mesh, times a radial profile.
MaterialDirection NodalMaterialDirection(
    const Scene& scene, const std::function<double(double)>& profile) {
  const auto& material = scene.material;
  const auto& mesh = material.Mesh();
  const auto [k0, m0] = material.Moduli().KappaMu();
  MaterialDirection d;
  d.kappa.assign(k0.size(), 0.0);
  d.mu.assign(m0.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      if (!scene.rheology.IsActive(i)) continue;
      const auto p = 0.1 * profile(mesh.Radius(e, q));
      d.kappa[i] = 0.5 * p * k0[i];
      d.mu[i] = p * m0[i];
    }
  }
  return d;
}

/// A direction: thickness at every node, an itau perturbation and a
/// material direction (may be empty).
struct Direction3D {
  TangentPerturbation3D tangent;
  std::vector<std::vector<double>> itau;  // the same as tangent.itau
  MaterialDirection3D material;           // the same as tangent.material
};

/// Its pairing with a kernel object (kernels or derivatives).
template <class Ice, class Kernels>
double Pairing(const Scene& scene, const GridCoupling& coupling,
               const Ice& ice, const Kernels& itau3D,
               const Direction3D& d, std::span<const double> kappa3D = {},
               std::span<const double> mu3D = {}) {
  auto total = 0.0;
  for (auto n = std::size_t{0}; n < d.tangent.iceThickness.size(); ++n) {
    total += scene.Surface(ice.thickness[n], d.tangent.iceThickness[n]);
  }
  total += GridPairing(coupling, itau3D, d.itau);
  if (!d.material.kappa.empty() && !kappa3D.empty()) {
    total += VolumePairing(coupling, kappa3D, d.material.kappa);
  }
  if (!d.material.mu.empty() && !mu3D.empty()) {
    total += VolumePairing(coupling, mu3D, d.material.mu);
  }
  return total;
}

}  // namespace

/// With delta = 0 at the GLL points the 3D second-order sweep's kernels
/// and derivatives are the 1D second-order sweep's by the same method: the
/// thickness views and the spherical reductions of the itau kernel and
/// its derivative, along a direction angle-uniform in itau.
void ZeroLateralMatchesThe1D(AdjointMethod method, double tol) {
  auto scene = Scene::Make(1e21);
  std::mt19937 rng(3);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * scene.Kyr());
  auto thickness = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    thickness.push_back(scene.Smooth(rng, 100.0 * scene.Metre()));
  }
  const auto directLoad = FrozenLoad(scene, times, thickness);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[5].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
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
  // The material: a direction in v (angle-uniform in 3D) and the kernels
  // on, both methods.
  const bool withMaterial = true;
  const auto d1 = NodalMaterialDirection(
      scene, [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); });

  // 1D.
  ViscoelasticOperator op1(scene.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  Evolution evolution1(op1, scene.solver, scene.state,
                               {.rotation = RotationalFeedback::None,
                                .rtol = 1e-12,
                                .record_stride = 1});
  const auto trajectory1 =
      ForwardTrajectory::FromSnapshots(evolution1.Run(times, directLoad));
  TangentPerturbation v1;
  v1.iceThickness = deltaIce;
  v1.itau = {ditau1};
  if (withMaterial) v1.material = AtGll(d1);
  const auto run1 = TangentLinearSolver(evolution1,
                                          {.method = method, .rtol = 1e-12, .record = true})
                        .Run(times, v1, &trajectory1);
  const auto [kernels1, derivatives1] =
      AdjointSolver(evolution1, {.method = method,
                                 .rtol = 1e-12,
                                 .material_kernels = withMaterial})
          .SecondOrderKernels(times, measurements, run1, v1.itau, trajectory1, {},
                              v1.material);

  // 3D with delta = 0.
  Viscoelastic3D op3(scene.rheology, kLmax,
                     [](double, double, double) { return 0.0; },
                     {.memory_points = MemoryPointSet::Gll});
  Evolution3D evolution3(
      op3, scene.solver, scene.state,
      {.rotation = RotationalFeedback::None,
       .rtol = 1e-12,
       .coupled_rtol = 1e-12,
       .record_stride = 1});
  const auto trajectory3 =
      ForwardTrajectory3D::FromSnapshots(evolution3.Run(times, directLoad));
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
  if (withMaterial) {
    v3.material.kappa = UniformOnCoupling(coupling, scene.material, d1.kappa);
    v3.material.mu = UniformOnCoupling(coupling, scene.material, d1.mu);
  }
  const auto run3 =
      TangentLinearSolver3D(evolution3, {.method = method,
                                         .rtol = 1e-12,
                                         .coupled_rtol = 1e-12,
                                         .record = true})
          .Run(times, v3, &trajectory3);
  const auto [kernels3, derivatives3] =
      AdjointSolver3D(evolution3, {.method = method,
                                   .rtol = 1e-12,
                                   .coupled_rtol = 1e-12,
                                   .material_kernels = withMaterial})
          .SecondOrderKernels(times, measurements, run3, v3.itau, trajectory3, {},
                              v3.material);

  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_LT(MaxDiff(kernels3.ice.thickness[n], kernels1.thickness[n]),
              tol * MaxAbs(kernels1.thickness[n]))
        << "kernel node " << n;
    EXPECT_LT(MaxDiff(derivatives3.ice.thickness[n], derivatives1.thickness[n]),
              tol * MaxAbs(derivatives1.thickness[n]))
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
  EXPECT_LT(worst, tol * scale);
  EXPECT_GT(dScale, 0.0);
  EXPECT_LT(dWorst, tol * dScale);
  if (withMaterial) {
    // The material kernels and their derivatives: the spherical
    // reductions against the 1D nodal kernels at the memory nodes.
    for (const auto& [name, k3, k1] :
         {std::tuple{"kappa", &kernels3.kappaKernelSpherical, &kernels1.kappaKernel},
          std::tuple{"mu", &kernels3.muKernelSpherical, &kernels1.muKernel},
          std::tuple{"dkappa", &derivatives3.kappaKernelSpherical,
                     &derivatives1.kappaKernel},
          std::tuple{"dmu", &derivatives3.muKernelSpherical, &derivatives1.muKernel}}) {
      auto s = 0.0, w = 0.0;
      for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
        const auto full =
            static_cast<std::size_t>(coupling.ElementOfNode(rn)) * mesh.NGll() +
            coupling.NodeInElementOf(rn);
        s = std::max(s, std::abs((*k1)[full]));
        w = std::max(w, std::abs((*k3)[rn] - (*k1)[full]));
      }
      EXPECT_GT(s, 0.0) << name;
      EXPECT_LT(w, tol * s) << name << " " << w << " " << s;
    }
  }
}

TEST(SecondOrderAdjoint3D, ZeroLateralMatchesThe1DSweep) {
  ZeroLateralMatchesThe1D(AdjointMethod::Continuous, 1e-6);
}

TEST(SecondOrderAdjoint3D, DiscreteZeroLateralMatchesThe1DDiscreteSweep) {
  ZeroLateralMatchesThe1D(AdjointMethod::Discrete, 1e-7);
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
    measurements[times.size() - 1].relativeSeaLevelDual = g;

    auto make = [&](double eps, const std::function<double(double, double)>& p,
                    Viscoelastic3D** opOut, Evolution3D** evOut) {
      *opOut = new Viscoelastic3D(scene.rheology, kLmax,
                                  [&, eps](double r, double theta, double phi) {
                                    return baseDelta(r, theta, phi) + eps * p(theta, phi);
                                  });
      *evOut = new Evolution3D(
          **opOut, scene.solver, scene.state,
          {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    };
    Viscoelastic3D* op = nullptr;
    Evolution3D* evolution = nullptr;
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
    const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
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
    const auto forward = scene.Surface(g, second.back().relativeSeaLevelChange);
    // F''[u, w] of the log parameterisation carries the uv term, whose
    // dual is the first-order kernel paired with itau_uv.
    const auto uvTerm = GridPairing(coupling, first.itauKernel3D, uw.itauUV);
    adjointGaps.push_back(std::abs(forward - uvTerm - huW) / std::abs(forward));

    // Central differences of the continuous kernels along u (delta moved
    // by eps p_u, the thickness by eps delta I_u) paired with w.
    auto gradientAt = [&](double eps) {
      Viscoelastic3D* movedOp = nullptr;
      Evolution3D* movedEv = nullptr;
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
              .FirstOrderKernels(times, measurements, &movedTrajectory);
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

TEST(SecondOrderAdjoint3D, DiscreteLateralHessianIsExact) {
  // With lateral structure on, the discrete second-order sweep is the
  // exact derivative of the exact gradient: the carried kernels equal the
  // first-order discrete sweep's to rounding; <H u, w> = <H w, u>, and
  // both equal <g, F''[u, w]> from the discrete 3D second-order
  // tangent-linear run, to the solvers' tolerance; and central
  // differences of the discrete kernels along u converge to <H u, w> at
  // second order with Richardson landing on it — the machine-grade gates
  // the continuous sweep passes only at its commute level.
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
  constexpr auto nodes = 9;
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
  measurements[times.size() - 1].relativeSeaLevelDual = g;
  const AdjointSolver3D::Options adjointOptions{
      .method = AdjointMethod::Discrete, .rtol = 1e-12, .coupled_rtol = 1e-12};

  // The operator with delta moved by eps p_u, and its evolution.
  auto make = [&](double eps) {
    auto op = std::make_unique<Viscoelastic3D>(
        scene.rheology, kLmax, [&, eps](double r, double theta, double phi) {
          return baseDelta(r, theta, phi) + eps * pU(theta, phi);
        });
    auto evolution = std::make_unique<Evolution3D>(
        *op, scene.solver, scene.state,
        Evolution3D::Options{
            .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
    return std::pair(std::move(op), std::move(evolution));
  };
  auto [op, evolution] = make(0.0);
  const auto& coupling = op->Coupling();
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(
      evolution->Run(times, FrozenLoad(scene, times, thickness)));
  auto direction = [&](const std::vector<double>& field,
                       const std::function<double(double)>& profile,
                       const std::function<double(double, double)>& p,
                       const Viscoelastic3D& at) {
    Direction3D d;
    d.tangent.iceThickness = scaled(field, profile);
    d.itau = TangentLinearSolver3D::ItauOfLogViscosity(
        at, OnCoupling(at.Coupling(), p));
    d.tangent.itau = d.itau;
    return d;
  };
  const auto u = direction(fieldU, [](double s) { return s; }, pU, *op);
  const auto w = direction(
      fieldW, [](double s) { return (1.0 - s) * (1.0 - s); }, pW, *op);
  TangentLinearSolver3D tangent(*evolution, {.method = AdjointMethod::Discrete,
                                             .rtol = 1e-12,
                                             .coupled_rtol = 1e-12,
                                             .record = true});
  const auto runU = tangent.Run(times, u.tangent, &trajectory);
  const auto runW = tangent.Run(times, w.tangent, &trajectory);
  AdjointSolver3D adjoint(*evolution, adjointOptions);
  const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
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

  // Symmetry, and the adjoint identity with the discrete second-order run
  // (F''[u, w] of the log parameterisation carries the uv term, whose
  // dual is the first-order kernel paired with itau_uv).
  const auto huW = Pairing(scene, coupling, alongU.ice, alongU.itauKernel3D, w);
  const auto hwU = Pairing(scene, coupling, alongW.ice, alongW.itauKernel3D, u);
  EXPECT_NEAR(huW, hwU, 1e-8 * std::abs(huW));
  SecondOrderDirection3D uw;
  uw.itauU = u.itau;
  uw.itauV = w.itau;
  uw.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(
      *op, OnCoupling(coupling, pU), OnCoupling(coupling, pW));
  const auto second = tangent.RunSecondOrder(times, runU, runW, uw, &trajectory);
  const auto forward = scene.Surface(g, second.back().relativeSeaLevelChange);
  const auto uvTerm = GridPairing(coupling, first.itauKernel3D, uw.itauUV);
  EXPECT_NEAR(forward, uvTerm + huW, 1e-8 * std::abs(forward));

  // Central differences of the discrete kernels along u (delta moved by
  // eps p_u, the thickness by eps delta I_u) paired with w at the moved
  // operator: second order, Richardson landing on <H u, w> + the uv term.
  auto gradientAt = [&](double eps) {
    auto [movedOp, movedEv] = make(eps);
    auto moved = thickness;
    for (auto n = std::size_t{0}; n < moved.size(); ++n) {
      for (auto i = std::size_t{0}; i < moved[n].size(); ++i) {
        moved[n][i] += eps * u.tangent.iceThickness[n][i];
      }
    }
    const auto movedTrajectory = ForwardTrajectory3D::FromSnapshots(
        movedEv->Run(times, FrozenLoad(scene, times, moved)));
    const auto kernels = AdjointSolver3D(*movedEv, adjointOptions)
                             .FirstOrderKernels(times, measurements, &movedTrajectory);
    Direction3D wAt = w;
    wAt.itau = TangentLinearSolver3D::ItauOfLogViscosity(
        *movedOp, OnCoupling(movedOp->Coupling(), pW));
    return Pairing(scene, movedOp->Coupling(), kernels.ice, kernels.itauKernel3D,
                   wAt);
  };
  const auto predicted = huW + uvTerm;
  auto fd = std::map<double, double>{};
  for (double eps : {0.02, 0.01}) {
    fd[eps] = (gradientAt(eps) - gradientAt(-eps)) / (2.0 * eps);
  }
  const auto r1 = std::abs(fd[0.02] - predicted);
  const auto r2 = std::abs(fd[0.01] - predicted);
  EXPECT_GT(r1 / r2, 3.0) << r1 << " " << r2;
  EXPECT_LT(r1 / r2, 5.0) << r1 << " " << r2;
  const auto richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0;
  EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted));
}

void DiscreteMaterialHessianIsExactAndSymmetric(RotationalFeedback rotation) {
  // With lateral structure on and LATERAL directions in the moduli as
  // well as the thickness and the log viscosity, the discrete second-order
  // sweep with the material kernels on: the carried kernels (material
  // included) equal the first-order discrete sweep's to rounding;
  // <H u, w> = <H w, u> and both equal <g, F''[u, w]> from the discrete
  // 3D second-order tangent-linear run in the material, to the solvers'
  // tolerance — the exact transposes, every mixed block exercised. At
  // the default memory nodes, the Gauss points (the same under lateral
  // moduli is held at the GLL points, below).
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(23);
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
  constexpr auto nodes = 9;
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
  measurements[times.size() - 1].relativeSeaLevelDual = g;
  measurements[4].displacementDual = scene.Smooth(rng, 0.5);
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = scene.material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;

  Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                    {.memory_points = MemoryPointSet::Gauss});
  Evolution3D evolution(
      op, scene.solver, scene.state,
      {.rotation = rotation,
       .rtol = 1e-12,
       .coupled_rtol = 1e-12,
       .record_stride = 1});
  const auto& coupling = op.Coupling();
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(
      evolution.Run(times, FrozenLoad(scene, times, thickness)));
  auto direction = [&](const std::vector<double>& field,
                       const std::function<double(double)>& profile,
                       const std::function<double(double, double)>& p,
                       const std::function<double(double, double)>& angular) {
    Direction3D d;
    d.tangent.iceThickness = scaled(field, profile);
    d.itau = TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, p));
    d.tangent.itau = d.itau;
    d.material.kappa = LateralModulus(
        coupling, scene.material, k0, [](double r) { return 0.05 * (1.0 - 0.5 * r); },
        angular);
    d.material.mu = LateralModulus(
        coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
        angular);
    d.tangent.material = d.material;
    return d;
  };
  const auto u = direction(
      fieldU, [](double s) { return s; }, pU,
      [](double theta, double phi) {
        return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
      });
  const auto w = direction(
      fieldW, [](double s) { return (1.0 - s) * (1.0 - s); }, pW,
      [](double theta, double phi) {
        return Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6) +
               0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                            .rtol = 1e-12,
                                            .coupled_rtol = 1e-12,
                                            .record = true});
  const auto runU = tangent.Run(times, u.tangent, &trajectory);
  const auto runW = tangent.Run(times, w.tangent, &trajectory);
  AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                      .rtol = 1e-12,
                                      .coupled_rtol = 1e-12,
                                      .material_kernels = true});
  const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
  const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
      times, measurements, runU, u.itau, trajectory, {}, u.material);
  const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
      times, measurements, runW, w.itau, trajectory, {}, w.material);
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_LT(MaxDiff(kernelsU.ice.thickness[n], first.ice.thickness[n]),
              1e-10 * MaxAbs(first.ice.thickness[n]) + 1e-300);
  }
  EXPECT_LT(MaxDiff(kernelsU.itauKernel3D[0], first.itauKernel3D[0]),
            1e-10 * MaxAbs(first.itauKernel3D[0]));
  EXPECT_LT(MaxDiff(kernelsU.kappaKernel3D, first.kappaKernel3D),
            1e-10 * MaxAbs(first.kappaKernel3D));
  EXPECT_LT(MaxDiff(kernelsU.muKernel3D, first.muKernel3D),
            1e-10 * MaxAbs(first.muKernel3D));
  EXPECT_GT(MaxAbs(alongU.kappaKernel3D), 0.0);
  EXPECT_GT(MaxAbs(alongU.muKernel3D), 0.0);

  const auto huW = Pairing(scene, coupling, alongU.ice, alongU.itauKernel3D, w,
                           alongU.kappaKernel3D, alongU.muKernel3D);
  const auto hwU = Pairing(scene, coupling, alongW.ice, alongW.itauKernel3D, u,
                           alongW.kappaKernel3D, alongW.muKernel3D);
  EXPECT_NEAR(huW, hwU, 1e-8 * std::abs(huW));
  SecondOrderDirection3D uw;
  uw.itauU = u.itau;
  uw.itauV = w.itau;
  uw.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(
      op, OnCoupling(coupling, pU), OnCoupling(coupling, pW));
  uw.materialU = u.material;
  uw.materialV = w.material;
  const auto second = tangent.RunSecondOrder(times, runU, runW, uw, &trajectory);
  auto forward = 0.0;
  for (const auto& [node, m] : measurements) {
    if (!m.relativeSeaLevelDual.empty()) {
      forward += scene.Surface(m.relativeSeaLevelDual, second[node].relativeSeaLevelChange);
    }
    if (!m.displacementDual.empty()) {
      forward += scene.Surface(m.displacementDual, second[node].displacement);
    }
  }
  const auto uvTerm = GridPairing(coupling, first.itauKernel3D, uw.itauUV);
  EXPECT_NEAR(forward, uvTerm + huW, 1e-8 * std::abs(forward));
}

TEST(SecondOrderAdjoint3D, DiscreteMaterialHessianIsExactAndSymmetric) {
  DiscreteMaterialHessianIsExactAndSymmetric(RotationalFeedback::None);
}

TEST(SecondOrderAdjoint3D, DiscreteMaterialHessianIsExactUnderRotation) {
  DiscreteMaterialHessianIsExactAndSymmetric(RotationalFeedback::Full);
}

TEST(SecondOrderAdjoint3D, ContinuousMaterialHessianConverges) {
  // With lateral structure on and lateral directions in the thickness,
  // the log viscosity and the moduli in both u and w, the continuous
  // second-order sweep's derivatives (the material kernels' included)
  // tend to the discrete ones block by block as the steps are halved,
  // and so does its mismatch with the continuous second-order run: the
  // commute level of the first-order 3D gates, on the eta = 1e22 mantle.
  // At the default memory nodes, the Gauss points, with the full
  // rotational feedback on (without it under lateral moduli, below).
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(29);
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
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = scene.material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  const auto span = 10.0 * scene.Kyr();
  Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                    {.memory_points = MemoryPointSet::Gauss});
  const auto& coupling = op.Coupling();
  struct Blocks {
    double ice, viscosity, material, forward;
  };
  auto iceGaps = std::vector<double>{}, viscosityGaps = std::vector<double>{},
       materialGaps = std::vector<double>{}, mismatches = std::vector<double>{};
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
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.rotation = RotationalFeedback::Full,
         .rtol = 1e-12,
         .coupled_rtol = 1e-12,
         .record_stride = 1});
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(
        evolution.Run(times, FrozenLoad(scene, times, thickness)));
    auto direction = [&](const std::vector<double>& field,
                         const std::function<double(double)>& profile,
                         const std::function<double(double, double)>& p,
                         const std::function<double(double, double)>& angular) {
      Direction3D d;
      d.tangent.iceThickness = scaled(field, profile);
      d.itau = TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, p));
      d.tangent.itau = d.itau;
      d.material.kappa = LateralModulus(
          coupling, scene.material, k0, [](double r) { return 0.05 * (1.0 - 0.5 * r); },
          angular);
      d.material.mu = LateralModulus(
          coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
          angular);
      d.tangent.material = d.material;
      return d;
    };
    const auto u = direction(
        fieldU, [](double s) { return s; }, pU,
        [](double theta, double phi) {
          return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
        });
    const auto w = direction(
        fieldW, [](double s) { return (1.0 - s) * (1.0 - s); }, pW,
        [](double theta, double phi) {
          return Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6) +
                 0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        });
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[times.size() - 1].relativeSeaLevelDual = g;
    SecondOrderDirection3D uw;
    uw.itauU = u.itau;
    uw.itauV = w.itau;
    uw.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(
        op, OnCoupling(coupling, pU), OnCoupling(coupling, pW));
    uw.materialU = u.material;
    uw.materialV = w.material;
    auto results = std::map<AdjointMethod, Blocks>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver3D tangent(evolution, {.method = method,
                                                .rtol = 1e-12,
                                                .coupled_rtol = 1e-12,
                                                .record = true});
      const auto runU = tangent.Run(times, u.tangent, &trajectory);
      const auto runW = tangent.Run(times, w.tangent, &trajectory);
      AdjointSolver3D adjoint(evolution, {.method = method,
                                          .rtol = 1e-12,
                                          .coupled_rtol = 1e-12,
                                          .material_kernels = true});
      const auto [kernels, alongU] = adjoint.SecondOrderKernels(
          times, measurements, runU, u.itau, trajectory, {}, u.material);
      const auto second = tangent.RunSecondOrder(times, runU, runW, uw, &trajectory);
      auto ice = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        ice += scene.Surface(alongU.ice.thickness[n], w.tangent.iceThickness[n]);
      }
      const auto viscosity = GridPairing(coupling, alongU.itauKernel3D, w.itau);
      const auto material = VolumePairing(coupling, alongU.kappaKernel3D, w.material.kappa) +
                            VolumePairing(coupling, alongU.muKernel3D, w.material.mu);
      const auto uvTerm = GridPairing(coupling, kernels.itauKernel3D, uw.itauUV);
      const auto forward =
          scene.Surface(g, second.back().relativeSeaLevelChange) - uvTerm;
      results[method] = {ice, viscosity, material, forward};
    }
    const auto& d = results[AdjointMethod::Discrete];
    const auto& c = results[AdjointMethod::Continuous];
    EXPECT_NEAR(d.ice + d.viscosity + d.material, d.forward, 1e-8 * std::abs(d.forward))
        << "nodes " << nodes;
    iceGaps.push_back(std::abs(d.ice - c.ice) / std::abs(d.ice));
    viscosityGaps.push_back(std::abs(d.viscosity - c.viscosity) / std::abs(d.viscosity));
    materialGaps.push_back(std::abs(d.material - c.material) / std::abs(d.material));
    // against the blocks' size: with the rotational feedback on they
    // nearly cancel in the sum
    mismatches.push_back(
        std::abs(c.ice + c.viscosity + c.material - c.forward) /
        (std::abs(c.ice) + std::abs(c.viscosity) + std::abs(c.material)));
  }
  for (const auto* gaps : {&iceGaps, &viscosityGaps, &materialGaps, &mismatches}) {
    EXPECT_LT((*gaps)[0], 0.1) << (*gaps)[0] << " " << (*gaps)[1];
    EXPECT_LT((*gaps)[1], (*gaps)[0]) << (*gaps)[0] << " " << (*gaps)[1];
    std::cout << "gap " << (*gaps)[0] << " -> " << (*gaps)[1] << '\n';
  }
}

void DiscreteMaterialHessianIsExactUnderLateralModuli(Viscoelastic3D::Options::StepSolver solver) {
  // The discrete second-order sweep with lateral elastic moduli in the
  // background as well as lateral viscosity: the carried kernels equal the
  // first-order sweep's, <H u, w> = <H w, u>, and both equal the second-
  // order tangent-linear run's objective less the itau cross term.
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(23);
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
  constexpr auto nodes = 9;
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
  measurements[times.size() - 1].relativeSeaLevelDual = g;
  measurements[4].displacementDual = scene.Smooth(rng, 0.5);
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = scene.material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;

  Viscoelastic3D op(
      scene.rheology, kLmax, baseDelta,
      {.solver = solver,
       .solve_rtol = 1e-12,
       .memory_points = MemoryPointSet::Gll,
       .moduli = {.shear =
                      [](double r, double theta, double phi) {
                        return 0.1 * (0.5 + r * r) *
                               (std::sin(theta) * std::cos(theta) * std::cos(phi) +
                                0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
                      },
                  .bulk = [](double r, double theta, double) {
                    return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
                  }}});
  Evolution3D evolution(
      op, scene.solver, scene.state,
      {.rotation = RotationalFeedback::None,
       .rtol = 1e-12,
       .coupled_rtol = 1e-12,
       .record_stride = 1});
  const auto& coupling = op.Coupling();
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(
      evolution.Run(times, FrozenLoad(scene, times, thickness)));
  auto direction = [&](const std::vector<double>& field,
                       const std::function<double(double)>& profile,
                       const std::function<double(double, double)>& p,
                       const std::function<double(double, double)>& angular) {
    Direction3D d;
    d.tangent.iceThickness = scaled(field, profile);
    d.itau = TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, p));
    d.tangent.itau = d.itau;
    d.material.kappa = LateralModulus(
        coupling, scene.material, k0, [](double r) { return 0.05 * (1.0 - 0.5 * r); },
        angular);
    d.material.mu = LateralModulus(
        coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
        angular);
    d.tangent.material = d.material;
    return d;
  };
  const auto u = direction(
      fieldU, [](double s) { return s; }, pU,
      [](double theta, double phi) {
        return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
      });
  const auto w = direction(
      fieldW, [](double s) { return (1.0 - s) * (1.0 - s); }, pW,
      [](double theta, double phi) {
        return Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6) +
               0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                            .rtol = 1e-12,
                                            .coupled_rtol = 1e-12,
                                            .record = true});
  const auto runU = tangent.Run(times, u.tangent, &trajectory);
  const auto runW = tangent.Run(times, w.tangent, &trajectory);
  AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                      .rtol = 1e-12,
                                      .coupled_rtol = 1e-12,
                                      .material_kernels = true});
  const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
  const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
      times, measurements, runU, u.itau, trajectory, {}, u.material);
  const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
      times, measurements, runW, w.itau, trajectory, {}, w.material);
  EXPECT_LT(MaxDiff(kernelsU.itauKernel3D[0], first.itauKernel3D[0]),
            1e-10 * MaxAbs(first.itauKernel3D[0]));
  EXPECT_LT(MaxDiff(kernelsU.kappaKernel3D, first.kappaKernel3D),
            1e-10 * MaxAbs(first.kappaKernel3D));
  EXPECT_LT(MaxDiff(kernelsU.muKernel3D, first.muKernel3D),
            1e-10 * MaxAbs(first.muKernel3D));
  const auto huW = Pairing(scene, coupling, alongU.ice, alongU.itauKernel3D, w,
                           alongU.kappaKernel3D, alongU.muKernel3D);
  const auto hwU = Pairing(scene, coupling, alongW.ice, alongW.itauKernel3D, u,
                           alongW.kappaKernel3D, alongW.muKernel3D);
  std::cout << "huW " << huW << " hwU " << hwU << " diff " << huW - hwU << '\n';
  EXPECT_NEAR(huW, hwU, 1e-8 * std::abs(huW));
  SecondOrderDirection3D uw;
  uw.itauU = u.itau;
  uw.itauV = w.itau;
  uw.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(
      op, OnCoupling(coupling, pU), OnCoupling(coupling, pW));
  uw.materialU = u.material;
  uw.materialV = w.material;
  const auto second = tangent.RunSecondOrder(times, runU, runW, uw, &trajectory);
  auto forward = 0.0;
  for (const auto& [node, m] : measurements) {
    if (!m.relativeSeaLevelDual.empty()) {
      forward += scene.Surface(m.relativeSeaLevelDual, second[node].relativeSeaLevelChange);
    }
    if (!m.displacementDual.empty()) {
      forward += scene.Surface(m.displacementDual, second[node].displacement);
    }
  }
  const auto uvTerm = GridPairing(coupling, first.itauKernel3D, uw.itauUV);
  std::cout << "forward " << forward << " uvTerm+huW " << uvTerm + huW << " diff "
            << forward - (uvTerm + huW) << '\n';
  EXPECT_NEAR(forward, uvTerm + huW, 1e-8 * std::abs(forward));
}

TEST(SecondOrderAdjoint3D, DiscreteMaterialHessianIsExactUnderLateralModuli) {
  DiscreteMaterialHessianIsExactUnderLateralModuli(Viscoelastic3D::Options::StepSolver::FixedPoint);
}

// The same with the steps closed by conjugate gradients, the discrete
// transpose's and its derivative state's included.
TEST(SecondOrderAdjoint3D, DiscreteMaterialHessianIsExactUnderConjugateGradientSteps) {
  DiscreteMaterialHessianIsExactUnderLateralModuli(Viscoelastic3D::Options::StepSolver::ConjugateGradients);
}

TEST(SecondOrderAdjoint3D, ContinuousMaterialHessianConvergesUnderLateralModuli) {
  // The continuous second-order sweep at a laterally heterogeneous
  // elastic background: its derivatives tend to the discrete ones block
  // by block as the steps are halved, and so does its mismatch with the
  // continuous second-order run. At the default memory nodes, the Gauss
  // points.
  auto scene = Scene::Make(1e22);
  std::mt19937 rng(29);
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
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = scene.material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  const auto span = 10.0 * scene.Kyr();
  Viscoelastic3D op(
      scene.rheology, kLmax, baseDelta,
      {.memory_points = MemoryPointSet::Gauss,
       .moduli = {.shear =
                      [](double r, double theta, double phi) {
                        return 0.1 * (0.5 + r * r) *
                               (std::sin(theta) * std::cos(theta) * std::cos(phi) +
                                0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
                      },
                  .bulk = [](double r, double theta, double) {
                    return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
                  }}});
  const auto& coupling = op.Coupling();
  struct Blocks {
    double ice, viscosity, material, forward;
  };
  auto iceGaps = std::vector<double>{}, viscosityGaps = std::vector<double>{},
       materialGaps = std::vector<double>{}, mismatches = std::vector<double>{};
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
    Evolution3D evolution(
        op, scene.solver, scene.state,
        {.rotation = RotationalFeedback::None,
         .rtol = 1e-12,
         .coupled_rtol = 1e-12,
         .record_stride = 1});
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(
        evolution.Run(times, FrozenLoad(scene, times, thickness)));
    auto direction = [&](const std::vector<double>& field,
                         const std::function<double(double)>& profile,
                         const std::function<double(double, double)>& p,
                         const std::function<double(double, double)>& angular) {
      Direction3D d;
      d.tangent.iceThickness = scaled(field, profile);
      d.itau = TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, p));
      d.tangent.itau = d.itau;
      d.material.kappa = LateralModulus(
          coupling, scene.material, k0, [](double r) { return 0.05 * (1.0 - 0.5 * r); },
          angular);
      d.material.mu = LateralModulus(
          coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
          angular);
      d.tangent.material = d.material;
      return d;
    };
    const auto u = direction(
        fieldU, [](double s) { return s; }, pU,
        [](double theta, double phi) {
          return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
        });
    const auto w = direction(
        fieldW, [](double s) { return (1.0 - s) * (1.0 - s); }, pW,
        [](double theta, double phi) {
          return Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6) +
                 0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        });
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[times.size() - 1].relativeSeaLevelDual = g;
    SecondOrderDirection3D uw;
    uw.itauU = u.itau;
    uw.itauV = w.itau;
    uw.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(
        op, OnCoupling(coupling, pU), OnCoupling(coupling, pW));
    uw.materialU = u.material;
    uw.materialV = w.material;
    auto results = std::map<AdjointMethod, Blocks>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver3D tangent(evolution, {.method = method,
                                                .rtol = 1e-12,
                                                .coupled_rtol = 1e-12,
                                                .record = true});
      const auto runU = tangent.Run(times, u.tangent, &trajectory);
      const auto runW = tangent.Run(times, w.tangent, &trajectory);
      AdjointSolver3D adjoint(evolution, {.method = method,
                                          .rtol = 1e-12,
                                          .coupled_rtol = 1e-12,
                                          .material_kernels = true});
      const auto [kernels, alongU] = adjoint.SecondOrderKernels(
          times, measurements, runU, u.itau, trajectory, {}, u.material);
      const auto second = tangent.RunSecondOrder(times, runU, runW, uw, &trajectory);
      auto ice = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        ice += scene.Surface(alongU.ice.thickness[n], w.tangent.iceThickness[n]);
      }
      const auto viscosity = GridPairing(coupling, alongU.itauKernel3D, w.itau);
      const auto material = VolumePairing(coupling, alongU.kappaKernel3D, w.material.kappa) +
                            VolumePairing(coupling, alongU.muKernel3D, w.material.mu);
      const auto uvTerm = GridPairing(coupling, kernels.itauKernel3D, uw.itauUV);
      const auto forward =
          scene.Surface(g, second.back().relativeSeaLevelChange) - uvTerm;
      results[method] = {ice, viscosity, material, forward};
    }
    const auto& d = results[AdjointMethod::Discrete];
    const auto& c = results[AdjointMethod::Continuous];
    EXPECT_NEAR(d.ice + d.viscosity + d.material, d.forward, 1e-8 * std::abs(d.forward))
        << "nodes " << nodes;
    iceGaps.push_back(std::abs(d.ice - c.ice) / std::abs(d.ice));
    viscosityGaps.push_back(std::abs(d.viscosity - c.viscosity) / std::abs(d.viscosity));
    materialGaps.push_back(std::abs(d.material - c.material) / std::abs(d.material));
    mismatches.push_back(std::abs(c.ice + c.viscosity + c.material - c.forward) /
                         std::abs(c.forward));
  }
  // the coarse gaps are a little larger than at a radial elastic
  // background (the viscosity block's 0.105 against 0.09); what is
  // gated is that each halves at the commute order
  for (const auto* gaps : {&iceGaps, &viscosityGaps, &materialGaps, &mismatches}) {
    EXPECT_LT((*gaps)[0], 0.15) << (*gaps)[0] << " " << (*gaps)[1];
    EXPECT_LT((*gaps)[1], 0.5 * (*gaps)[0]) << (*gaps)[0] << " " << (*gaps)[1];
  }
  std::cout << "continuous Hessian gaps under lateral moduli: ice " << iceGaps[0] << " -> "
            << iceGaps[1] << ", viscosity " << viscosityGaps[0] << " -> " << viscosityGaps[1]
            << ", material " << materialGaps[0] << " -> " << materialGaps[1]
            << ", mismatch " << mismatches[0] << " -> " << mismatches[1] << '\n';
}

TEST(SecondOrderAdjoint3D, HessianInTheSourcesAndTheViscosityAboutASourceDrivenRun) {
  // About a run under point sources with lateral viscosity, along
  // directions that perturb the sources' moment tensors and the inverse
  // relaxation times together: the discrete second-order sweep's
  // derivatives, the viscosity rows and the moment rows, give
  // <H u, w> = <H w, u>, and both equal <g, F''[u, w]> from the discrete
  // second-order tangent-linear run; the mixed block (sources against
  // viscosity) is non-zero and each pure-source Hessian vanishes, the
  // run being linear in the sources. The same with lateral directions in
  // the moduli, against the sources and the viscosity and against each
  // other. The displacement and the toroidal potential measured.
  auto scene = Scene::Make(1e22);
  const auto& sle = scene.solver.SolverScales();
  std::mt19937 rng(73);
  std::normal_distribution<double> normal;
  const auto span = 10.0 * scene.Kyr();
  constexpr auto nodes = 7;
  auto times = std::vector<double>{};
  for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
  const auto surface = scene.material.Radius() *
                       scene.model.ModelScales().Length() / sle.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  const auto points = std::vector<BodyPoint>{
      {.radius = surface - 400e3 / sle.Length(), .colatitude = 1.1, .longitude = 4.0},
      {.radius = surface - 800e3 / sle.Length(), .colatitude = 2.2, .longitude = 1.0}};
  auto tensors = [&](double scale) {
    auto out = std::vector<std::vector<PointSource>>{};
    for (auto n = 0; n < nodes; ++n) {
      auto at = std::vector<PointSource>{};
      for (const auto& point : points) {
        at.push_back({.at = point,
                      .moment = {.rr = scale * unit * normal(rng),
                                 .tt = scale * unit * normal(rng),
                                 .pp = scale * unit * normal(rng),
                                 .rt = scale * unit * normal(rng),
                                 .rp = scale * unit * normal(rng),
                                 .tp = scale * unit * normal(rng)}});
      }
      out.push_back(std::move(at));
    }
    return out;
  };
  const auto base = tensors(1.0);
  const auto gU = scene.Smooth(rng, 0.5);
  const auto gW = scene.Smooth(rng, 0.8);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].displacementDual = gU;
  measurements[times.size() - 1].toroidalPotentialDual = gW;
  auto objective = [&](const std::vector<Evolution3D::Snapshot>& run) {
    return scene.Surface(gU, run.back().displacement) +
           scene.Surface(gW, run.back().toroidalPotential);
  };
  auto nodeOf = [&](double t) {
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * times.back()) return n;
    }
    throw std::runtime_error("the test history is read off its nodes");
  };
  Viscoelastic3D op(scene.rheology, kLmax, [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  const auto& coupling = op.Coupling();
  Evolution3D evolution(
      op, scene.solver, scene.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(evolution.Run(
      times, Evolution3D::Forcing{.sources = [&](double t) {
        return base[nodeOf(t)];
      }}));
  struct Direction {
    std::vector<std::vector<PointSource>> sources;
    std::vector<std::vector<double>> itau;
    MaterialDirection3D material{};
  };
  const auto kappaMu = scene.material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  auto moduli = [&](const std::function<double(double, double)>& angular) {
    MaterialDirection3D d;
    d.kappa = LateralModulus(
        coupling, scene.material, k0,
        [](double r) { return 0.05 * (1.0 - 0.5 * r); }, angular);
    d.mu = LateralModulus(
        coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
        angular);
    return d;
  };
  auto pU = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto pW = [](double theta, double phi) {
    return 0.5 * Cap(theta, phi, 2.4, 2.0, 0.8) + 0.3 * std::cos(theta);
  };
  const Direction u{tensors(0.3), TangentLinearSolver3D::ItauOfLogViscosity(
                                      op, OnCoupling(coupling, pU))};
  const Direction w{tensors(0.3), TangentLinearSolver3D::ItauOfLogViscosity(
                                      op, OnCoupling(coupling, pW))};
  const Direction sourcesOnly{u.sources, {}};
  const Direction viscosityOnly{{}, w.itau};
  // the moduli, with the viscosity: a source perturbation and a moduli
  // perturbation go in separate directions
  const Direction withModuli{
      {}, w.itau, moduli([](double theta, double phi) {
        return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
      })};
  const Direction otherModuli{
      {}, u.itau, moduli([](double theta, double phi) {
        return Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6) +
               0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      })};
  TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                            .rtol = 1e-12,
                                            .coupled_rtol = 1e-12,
                                            .record = true});
  AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                      .rtol = 1e-12,
                                      .coupled_rtol = 1e-12,
                                      .material_kernels = true,
                                      .moment_points = points});
  auto runOf = [&](const Direction& d) {
    TangentPerturbation3D direction;
    direction.sources = d.sources;
    direction.itau = d.itau;
    direction.material = d.material;
    return tangent.Run(times, direction, &trajectory);
  };
  auto hessianOn = [&](const Direction& d,
                       const std::vector<Evolution3D::Snapshot>& run) {
    return adjoint
        .SecondOrderKernels(times, measurements, run, d.itau, trajectory, {},
                            d.material)
        .second;
  };
  auto pairing = [&](const KernelDerivatives3D& along, const Direction& d) {
    auto total = d.itau.empty() ? 0.0
                                : GridPairing(coupling, along.itauKernel3D, d.itau);
    if (!d.sources.empty()) {
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        for (auto p = std::size_t{0}; p < points.size(); ++p) {
          total += along.ice.momentKernels[n][p].Contract(d.sources[n][p].moment);
        }
      }
    }
    if (!d.material.Empty()) {
      total += VolumePairing(coupling, along.kappaKernel3D, d.material.kappa) +
               VolumePairing(coupling, along.muKernel3D, d.material.mu);
    }
    return total;
  };
  struct Case {
    const char* name;
    const Direction* a;
    const Direction* b;
    bool vanishes;
  };
  for (const auto& c : {Case{"both parts", &u, &w, false},
                        Case{"sources against viscosity", &sourcesOnly,
                             &viscosityOnly, false},
                        Case{"sources and viscosity against moduli", &u,
                             &withModuli, false},
                        Case{"moduli against moduli", &withModuli, &otherModuli,
                             false},
                        Case{"sources against sources", &sourcesOnly, &sourcesOnly,
                             true}}) {
    const auto runA = runOf(*c.a);
    const auto runB = runOf(*c.b);
    const auto alongA = hessianOn(*c.a, runA);
    const auto alongB = hessianOn(*c.b, runB);
    const auto haB = pairing(alongA, *c.b);
    const auto hbA = pairing(alongB, *c.a);
    SecondOrderDirection3D ab;
    ab.itauU = c.a->itau;
    ab.itauV = c.b->itau;
    ab.materialU = c.a->material;
    ab.materialV = c.b->material;
    const auto forward =
        objective(tangent.RunSecondOrder(times, runA, runB, ab, &trajectory));
    if (c.vanishes) {
      // linear in the sources: against the mixed block's size
      const auto scale = std::abs(pairing(hessianOn(u, runOf(u)), w));
      EXPECT_LT(std::abs(haB), 1e-9 * scale) << c.name;
      EXPECT_LT(std::abs(forward), 1e-9 * scale) << c.name;
      continue;
    }
    EXPECT_NE(haB, 0.0) << c.name;
    EXPECT_NEAR(haB, hbA, 1e-8 * std::abs(haB)) << c.name;
    EXPECT_NEAR(forward, haB, 1e-8 * std::abs(forward)) << c.name;
  }
}

TEST(SecondOrderAdjoint3D, ContinuousHessianRowsConvergeUnderSourcesAndRotation) {
  // The continuous second-order sweep with lateral viscosity and the
  // rotational feedback, the displacement and the toroidal potential
  // measured: its rows in the viscosity, the bulk modulus and the shear
  // modulus (each apart) tend to the discrete sweep's at second order as
  // the steps are halved.
  auto scene = Scene::Make(1e22);
  const auto& sle = scene.solver.SolverScales();
  std::mt19937 rng(79);
  const auto span = 10.0 * scene.Kyr();
  const auto surface = scene.material.Radius() *
                       scene.model.ModelScales().Length() / sle.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  const auto source = PointSource{
      .at = {.radius = surface - 400e3 / sle.Length(), .colatitude = 1.1, .longitude = 4.0},
      .moment = {.rr = 0.9 * unit, .tt = -0.2 * unit, .pp = -0.7 * unit,
                 .rt = 0.3 * unit, .rp = -0.7 * unit, .tp = 0.6 * unit}};
  const auto other = SphericalTensor{.rr = -0.4 * unit, .tt = 0.8 * unit,
                                     .pp = -0.4 * unit, .rt = 0.5 * unit,
                                     .rp = 0.1 * unit, .tp = -0.3 * unit};
  const auto gU = scene.Smooth(rng, 0.5);
  const auto gW = scene.Smooth(rng, 0.8);
  auto pU = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  auto pW = [](double theta, double phi) {
    return 0.5 * Cap(theta, phi, 2.4, 2.0, 0.8) + 0.3 * std::cos(theta);
  };
  const auto kappaMu = scene.material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  Viscoelastic3D op(scene.rheology, kLmax, [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  const auto& coupling = op.Coupling();
  auto moduli = [&](const std::function<double(double, double)>& angular) {
    MaterialDirection3D d;
    d.kappa = LateralModulus(
        coupling, scene.material, k0,
        [](double r) { return 0.05 * (1.0 - 0.5 * r); }, angular);
    d.mu = LateralModulus(
        coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
        angular);
    return d;
  };
  const auto itauU =
      TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, pU));
  const auto itauW =
      TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, pW));
  const auto materialU = moduli([](double theta, double phi) {
    return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
  });
  const auto materialW = moduli([](double theta, double phi) {
    return 1.0 + 0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  struct Rows {
    double viscosity, kappa, mu, moment;
  };
  const auto meltLoad = scene.MeltLoad(-0.1);
  auto loadDriven = false;
  auto part = 0;  // 0 both, 1 itau alone, 2 moduli alone
  const auto measured = 0;  // U and W
  const auto rotationOn = true;
  auto gapsOf = [&](bool sourceDirection) {
    auto gaps = std::vector<Rows>{};
    for (const auto nodes : {5, 9, 17}) {
      auto times = std::vector<double>{};
      for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
      auto forcing = Evolution3D::Forcing{};
      if (loadDriven) {
        forcing.directLoad = [&](double t) {
          auto out = meltLoad;
          for (auto& v : out) v *= 0.3 + t / span;
          return out;
        };
      } else {
        forcing.sources = [&](double t) {
          auto at = source;
          const auto shape = 0.3 + t / span;
          for (auto* c : {&at.moment.rr, &at.moment.tt, &at.moment.pp,
                          &at.moment.rt, &at.moment.rp, &at.moment.tp}) {
            *c *= shape;
          }
          return std::vector<PointSource>{at};
        };
      }
      Evolution3D evolution(
          op, scene.solver, scene.state,
          {.rotation = rotationOn ? RotationalFeedback::Full : RotationalFeedback::None, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
      const auto trajectory =
          ForwardTrajectory3D::FromSnapshots(evolution.Run(times, forcing));
      auto measurements = std::map<std::size_t, Measurement>{};
      if (measured == 0 || measured == 1) {
        measurements[times.size() - 1].displacementDual = gU;
      }
      if (measured == 0 || measured == 2) {
        measurements[times.size() - 1].toroidalPotentialDual = gW;
      }
      if (measured == 3) measurements[times.size() - 1].relativeSeaLevelDual = gU;
      TangentPerturbation3D u;
      if (sourceDirection) {
        for (auto t : times) {
          auto M = other;
          const auto shape = 1.0 - 0.5 * t / span;
          for (auto* c : {&M.rr, &M.tt, &M.pp, &M.rt, &M.rp, &M.tp}) *c *= shape;
          u.sources.push_back({PointSource{.at = source.at, .moment = M}});
        }
      } else {
        if (part != 2) u.itau = itauU;
        if (part != 1) u.material = materialU;
      }
      auto rows = std::map<AdjointMethod, Rows>{};
      for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
        const auto run =
            TangentLinearSolver3D(evolution, {.method = method,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12,
                                              .record = true})
                .Run(times, u, &trajectory);
        const auto along =
            AdjointSolver3D(evolution, {.method = method,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12,
                                        .material_kernels = true,
                                        .moment_points = {source.at}})
                .SecondOrderKernels(times, measurements, run, u.itau, trajectory,
                                    {}, u.material)
                .second;
        // the source row, against a moment that falls with time
        auto moment = 0.0;
        for (auto n = std::size_t{0}; n < times.size(); ++n) {
          auto M = other;
          const auto shape = 0.2 + times[n] / span;
          for (auto* c : {&M.rr, &M.tt, &M.pp, &M.rt, &M.rp, &M.tp}) *c *= shape;
          moment += along.ice.momentKernels[n][0].Contract(M);
        }
        rows[method] = {GridPairing(coupling, along.itauKernel3D, itauW),
                        VolumePairing(coupling, along.kappaKernel3D, materialW.kappa),
                        VolumePairing(coupling, along.muKernel3D, materialW.mu),
                        moment};
      }
      const auto& d = rows[AdjointMethod::Discrete];
      const auto& c = rows[AdjointMethod::Continuous];
      EXPECT_NE(d.viscosity, 0.0);
      EXPECT_NE(d.kappa, 0.0);
      EXPECT_NE(d.mu, 0.0);
      gaps.push_back({std::abs(c.viscosity - d.viscosity) / std::abs(d.viscosity),
                      std::abs(c.kappa - d.kappa) / std::abs(d.kappa),
                      std::abs(c.mu - d.mu) / std::abs(d.mu),
                      d.moment == 0.0
                          ? std::abs(c.moment)
                          : std::abs(c.moment - d.moment) / std::abs(d.moment)});
    }
    return gaps;
  };
  loadDriven = true;
  for (const auto& g : gapsOf(false)) {
    std::printf("load-driven, along the parameters: viscosity %.3e kappa %.3e mu %.3e\n",
                g.viscosity, g.kappa, g.mu);
  }
  // Three cases: about the source-driven run along a direction in the
  // viscosity and the moduli, and along one that perturbs the sources;
  // and about a load-driven run along the moduli alone with the
  // displacement and the toroidal potential measured, which the
  // derivative of the jump's rotation enters.
  struct Case {
    const char* label;
    bool load, sources;
    int part;
  };
  for (const auto& c :
       {Case{"source-driven, along the viscosity and the moduli", false, false, 0},
        Case{"source-driven, along the sources", false, true, 0},
        Case{"load-driven, along the moduli", true, false, 2}}) {
    loadDriven = c.load;
    part = c.part;
    const auto gaps = gapsOf(c.sources);
    for (auto row : {&Rows::viscosity, &Rows::kappa, &Rows::mu, &Rows::moment}) {
      const auto what =
          row == &Rows::viscosity
              ? "viscosity"
              : (row == &Rows::kappa ? "kappa" : (row == &Rows::mu ? "mu" : "moment"));
      if (row == &Rows::moment && c.sources) {
        // the run is linear in the sources: no source-source block
        EXPECT_LT(gaps[2].*row, 1e-10) << c.label;
        continue;
      }
      EXPECT_LT(gaps[2].*row, 0.05)
          << what << ", " << c.label << ": " << gaps[0].*row << " "
          << gaps[1].*row << " " << gaps[2].*row;
      EXPECT_GT((gaps[1].*row) / (gaps[2].*row), 3.0)
          << what << ", " << c.label << ": " << gaps[1].*row << " "
          << gaps[2].*row;
    }
  }
}

TEST(SecondOrderAdjoint3D, OneNodeIsTheElasticProblemInEveryClass) {
  // One time node, the elastic problem, with and without lateral moduli
  // in the background, an ocean and the rotational feedback: the
  // tangent-linear run along the ice, lateral moduli and a source's
  // moment pairs with both sweeps' kernels (the continuous sweep is its
  // jump alone), and the Hessian is symmetric and equal to the
  // second-order run.
  auto scene = Scene::Make(1e21);
  const auto& sle = scene.solver.SolverScales();
  std::mt19937 rng(107);
  const auto times = std::vector<double>{0.0};
  const auto surface = scene.material.Radius() *
                       scene.model.ModelScales().Length() / sle.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  const auto at = BodyPoint{.radius = surface - 400e3 / sle.Length(),
                            .colatitude = 1.1,
                            .longitude = 4.0};
  auto sourceOf = [&](double a, double b, double c) {
    return std::vector<PointSource>{
        {.at = at,
         .moment = {.rr = a * unit, .tt = b * unit, .pp = -(a + b) * unit,
                    .rt = c * unit, .rp = -a * unit, .tp = b * unit}}};
  };
  const auto load = scene.MeltLoad(-0.1);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  measurements[0].displacementDual = scene.Smooth(rng, 0.5);
  measurements[0].tangentialPotentialDual = scene.Smooth(rng, 0.6);
  measurements[0].toroidalPotentialDual = scene.Smooth(rng, 0.8);
  const auto kappaMu = scene.material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  const auto thicknessU = scene.Smooth(rng, 20.0 * scene.Metre());
  const auto thicknessW = scene.Smooth(rng, 20.0 * scene.Metre());
  for (const bool lateralModuli : {false, true}) {
    auto options = Viscoelastic3D::Options{.memory_points = MemoryPointSet::Gauss};
    if (lateralModuli) {
      options.solve_rtol = 1e-12;
      options.moduli = {.shear =
                            [](double r, double theta, double phi) {
                              return 0.1 * (0.5 + r * r) * std::sin(theta) *
                                     std::cos(theta) * std::cos(phi);
                            },
                        .bulk = [](double r, double theta, double) {
                          return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
                        }};
    }
    Viscoelastic3D op(
        scene.rheology, kLmax,
        [](double, double theta, double phi) {
          return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        },
        options);
    const auto& coupling = op.Coupling();
    Evolution3D evolution(op, scene.solver, scene.state,
                          {.rotation = RotationalFeedback::Full,
                           .rtol = 1e-12,
                           .coupled_rtol = 1e-12,
                           .record_stride = 1});
    Evolution3D::Forcing forcing;
    forcing.directLoad = [&](double) { return load; };
    forcing.sources = [&](double) { return sourceOf(0.9, -0.2, 0.3); };
    const auto run = evolution.Run(times, forcing);
    ASSERT_EQ(run.size(), 1u);
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(run);
    auto direction = [&](const std::vector<double>& thickness,
                         const std::function<double(double, double)>& angular,
                         std::vector<PointSource> sources) {
      TangentPerturbation3D d;
      d.iceThickness = {thickness};
      d.material.kappa = LateralModulus(
          coupling, scene.material, k0,
          [](double r) { return 0.05 * (1.0 - 0.5 * r); }, angular);
      d.material.mu = LateralModulus(
          coupling, scene.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
          angular);
      (void)sources;
      return d;
    };
    const auto u = direction(
        thicknessU,
        [](double theta, double phi) {
          return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
        },
        sourceOf(-0.4, 0.8, 0.5));
    const auto w = direction(
        thicknessW,
        [](double theta, double phi) {
          return Cap(theta, phi, 1.9, 4.4, 0.7) +
                 0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        },
        sourceOf(0.2, 0.3, -0.6));
    // the sources go in a direction of their own: one tangent run takes
    // the sources or the moduli
    TangentPerturbation3D s;
    s.sources = {sourceOf(-0.4, 0.8, 0.5)};
    auto pair = [&](const auto& kernels, const TangentPerturbation3D& d) {
      if (!d.sources.empty()) {
        return kernels.ice.momentKernels[0][0].Contract(d.sources[0][0].moment);
      }
      return scene.Surface(kernels.ice.thickness[0], d.iceThickness[0]) +
             VolumePairing(coupling, kernels.kappaKernel3D, d.material.kappa) +
             VolumePairing(coupling, kernels.muKernel3D, d.material.mu);
    };
    auto Objective = [&](const Scene&, const std::vector<Evolution3D::Snapshot>& snaps,
                         const std::map<std::size_t, Measurement>& measured) {
      const auto& g = measured.at(0);
      const auto& x = snaps[0];
      return scene.Surface(g.relativeSeaLevelDual, x.relativeSeaLevelChange) +
             scene.Surface(g.displacementDual, x.displacement) +
             scene.Surface(g.tangentialPotentialDual, x.tangentialPotential) +
             scene.Surface(g.toroidalPotentialDual, x.toroidalPotential);
    };
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto label = std::string(lateralModuli ? "lateral moduli" : "radial moduli") +
                         ", method " + std::to_string(static_cast<int>(method));
      TangentLinearSolver3D tangent(evolution, {.method = method,
                                                .rtol = 1e-12,
                                                .coupled_rtol = 1e-12,
                                                .record = true});
      struct Case {
        const char* name;
        const TangentPerturbation3D* a;
        const TangentPerturbation3D* b;
      };
      AdjointSolver3D adjoint(evolution, {.method = method,
                                          .rtol = 1e-12,
                                          .coupled_rtol = 1e-12,
                                          .material_kernels = true,
                                          .moment_points = {at}});
      const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
      for (const auto& c : {Case{"ice and moduli", &u, &w},
                            Case{"source against ice and moduli", &s, &w}}) {
        const auto runA = tangent.Run(times, *c.a, &trajectory);
        const auto runB = tangent.Run(times, *c.b, &trajectory);
        const auto along = Objective(scene, runA, measurements);
        ASSERT_NE(along, 0.0) << label << ", " << c.name;
        EXPECT_NEAR(pair(first, *c.a), along, 1e-7 * std::abs(along))
            << label << ", " << c.name;
        const auto alongA = adjoint
                                .SecondOrderKernels(times, measurements, runA, {},
                                                    trajectory, {}, c.a->material)
                                .second;
        const auto alongB = adjoint
                                .SecondOrderKernels(times, measurements, runB, {},
                                                    trajectory, {}, c.b->material)
                                .second;
        const auto ab = pair(alongA, *c.b);
        const auto ba = pair(alongB, *c.a);
        SecondOrderDirection3D second;
        second.materialU = c.a->material;
        second.materialV = c.b->material;
        const auto forward = Objective(
            scene, tangent.RunSecondOrder(times, runA, runB, second, &trajectory),
            measurements);
        ASSERT_NE(forward, 0.0) << label << ", " << c.name;
        EXPECT_NEAR(ab, ba, 1e-7 * std::abs(ab)) << label << ", " << c.name;
        EXPECT_NEAR(ab, forward, 1e-7 * std::abs(forward)) << label << ", " << c.name;
      }
    }
  }
}
