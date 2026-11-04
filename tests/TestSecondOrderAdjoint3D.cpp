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

double FieldPairing(const GridCoupling& coupling, std::span<const double> kernel,
                    std::span<const double> direction) {
  auto total = 0.0;
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto at = coupling.FieldOffset(rn);
    const auto size = coupling.FieldSizeOf(rn);
    auto product = std::vector<double>(size);
    for (auto x = std::size_t{0}; x < size; ++x) {
      product[x] = kernel[at + x] * direction[at + x];
    }
    total += coupling.GridOf(rn).Integrate(product);
  }
  return total;
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
    total += FieldPairing(coupling, kappa3D, d.material.kappa);
  }
  if (!d.material.mu.empty() && !mu3D.empty()) {
    total += FieldPairing(coupling, mu3D, d.material.mu);
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
  // The material: a direction in v (angle-uniform in 3D) and the kernels
  // on, both methods.
  const bool withMaterial = true;
  const auto d1 = NodalMaterialDirection(
      scene, [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); });

  // 1D.
  ViscoelasticOperator op1(scene.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution evolution1(op1, scene.solver, scene.state,
                               {.rotation = RotationalFeedback::None,
                                .rtol = 1e-12,
                                .record_stride = 1});
  const auto trajectory1 =
      ForwardTrajectory::FromSnapshots(evolution1.Run(times, iceLoad));
  TangentPerturbation v1;
  v1.iceThickness = deltaIce;
  v1.itau = {ditau1};
  if (withMaterial) v1.material = d1;
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
  SeaLevelEvolution3D evolution3(
      op3, scene.solver, scene.state,
      {.rotation = RotationalFeedback::None,
       .rtol = 1e-12,
       .coupled_rtol = 1e-12,
       .record_stride = 1});
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
  measurements[times.size() - 1].seaLevelDual = g;
  const AdjointSolver3D::Options adjointOptions{
      .method = AdjointMethod::Discrete, .rtol = 1e-12, .coupled_rtol = 1e-12};

  // The operator with delta moved by eps p_u, and its evolution.
  auto make = [&](double eps) {
    auto op = std::make_unique<Viscoelastic3D>(
        scene.rheology, kLmax, [&, eps](double r, double theta, double phi) {
          return baseDelta(r, theta, phi) + eps * pU(theta, phi);
        });
    auto evolution = std::make_unique<SeaLevelEvolution3D>(
        *op, scene.solver, scene.state,
        SeaLevelEvolution3D::Options{
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
  const auto forward = scene.Surface(g, second.back().seaLevelChange);
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
                             .IceHistoryKernels(times, measurements, &movedTrajectory);
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

TEST(SecondOrderAdjoint3D, DiscreteMaterialHessianIsExactAndSymmetric) {
  // With lateral structure on and LATERAL directions in the moduli as
  // well as the thickness and the log viscosity, the discrete second-order
  // sweep with the material kernels on: the carried kernels (material
  // included) equal the first-order discrete sweep's to rounding;
  // <H u, w> = <H w, u> and both equal <g, F''[u, w]> from the discrete
  // 3D second-order tangent-linear run in the material, to the solvers'
  // tolerance — the exact transposes, every mixed block exercised.
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
  measurements[times.size() - 1].seaLevelDual = g;
  measurements[4].displacementDual = scene.Smooth(rng, 0.5);
  const auto [k0, m0] = scene.material.Moduli().KappaMu();

  Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                    {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution3D evolution(
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
  const auto first = adjoint.IceHistoryKernels(times, measurements, &trajectory);
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
    if (!m.seaLevelDual.empty()) {
      forward += scene.Surface(m.seaLevelDual, second[node].seaLevelChange);
    }
    if (!m.displacementDual.empty()) {
      forward += scene.Surface(m.displacementDual, second[node].displacement);
    }
  }
  const auto uvTerm = GridPairing(coupling, first.itauKernel3D, uw.itauUV);
  EXPECT_NEAR(forward, uvTerm + huW, 1e-8 * std::abs(forward));
}

TEST(SecondOrderAdjoint3D, ContinuousMaterialHessianConverges) {
  // With lateral structure on and lateral directions in the thickness,
  // the log viscosity and the moduli in both u and w, the continuous
  // second-order sweep's derivatives (the material kernels' included)
  // tend to the discrete ones block by block as the steps are halved,
  // and so does its mismatch with the continuous second-order run: the
  // commute level of the first-order 3D gates, on the eta = 1e22 mantle.
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
  const auto [k0, m0] = scene.material.Moduli().KappaMu();
  const auto span = 10.0 * scene.Kyr();
  Viscoelastic3D op(scene.rheology, kLmax, baseDelta,
                    {.memory_points = MemoryPointSet::Gll});
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
    SeaLevelEvolution3D evolution(
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
    measurements[times.size() - 1].seaLevelDual = g;
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
      const auto material = FieldPairing(coupling, alongU.kappaKernel3D, w.material.kappa) +
                            FieldPairing(coupling, alongU.muKernel3D, w.material.mu);
      const auto uvTerm = GridPairing(coupling, kernels.itauKernel3D, uw.itauUV);
      const auto forward =
          scene.Surface(g, second.back().seaLevelChange) - uvTerm;
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
  for (const auto* gaps : {&iceGaps, &viscosityGaps, &materialGaps, &mismatches}) {
    EXPECT_LT((*gaps)[0], 0.1) << (*gaps)[0] << " " << (*gaps)[1];
    EXPECT_LT((*gaps)[1], (*gaps)[0]) << (*gaps)[0] << " " << (*gaps)[1];
  }
}

TEST(SecondOrderAdjoint3D, DiscreteMaterialHessianIsExactUnderLateralModuli) {
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
  measurements[times.size() - 1].seaLevelDual = g;
  measurements[4].displacementDual = scene.Smooth(rng, 0.5);
  const auto [k0, m0] = scene.material.Moduli().KappaMu();

  Viscoelastic3D op(
      scene.rheology, kLmax, baseDelta,
      {.memory_points = MemoryPointSet::Gll,
       .moduli = {.shear =
                      [](double r, double theta, double phi) {
                        return 0.1 * (0.5 + r * r) *
                               (std::sin(theta) * std::cos(theta) * std::cos(phi) +
                                0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
                      },
                  .bulk = [](double r, double theta, double) {
                    return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
                  }}});
  SeaLevelEvolution3D evolution(
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
  const auto first = adjoint.IceHistoryKernels(times, measurements, &trajectory);
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
    if (!m.seaLevelDual.empty()) {
      forward += scene.Surface(m.seaLevelDual, second[node].seaLevelChange);
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

TEST(SecondOrderAdjoint3D, ContinuousMaterialHessianConvergesUnderLateralModuli) {
  // The continuous second-order sweep at a laterally heterogeneous
  // elastic background: its derivatives tend to the discrete ones block
  // by block as the steps are halved, and so does its mismatch with the
  // continuous second-order run.
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
  const auto [k0, m0] = scene.material.Moduli().KappaMu();
  const auto span = 10.0 * scene.Kyr();
  Viscoelastic3D op(
      scene.rheology, kLmax, baseDelta,
      {.memory_points = MemoryPointSet::Gll,
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
    SeaLevelEvolution3D evolution(
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
    measurements[times.size() - 1].seaLevelDual = g;
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
      const auto material = FieldPairing(coupling, alongU.kappaKernel3D, w.material.kappa) +
                            FieldPairing(coupling, alongU.muKernel3D, w.material.mu);
      const auto uvTerm = GridPairing(coupling, kernels.itauKernel3D, uw.itauUV);
      const auto forward =
          scene.Surface(g, second.back().seaLevelChange) - uvTerm;
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
