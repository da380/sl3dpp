// The continuous derivatives under Sdirk2, the default scheme, whose step
// is two backward Euler stages, 1D and 3D. The continuous tangent-linear
// drive of a stage is backward Euler's, (1 - w) (delta itau / itau) (d - m)
// at the forward stage's end; for a backward Euler stage that is the
// derivative of the stage's weight, w^2 dh (d - m_start), so the
// continuous tangent-linear runs, first and second order, are the discrete
// ones to rounding along every direction. The continuous second-order
// sweep steps the adjoint state and its derivative together stage by
// stage, so its derivatives are the exact derivative of the continuous
// first-order kernels; against the discrete second-order sweep they
// converge at second order as the steps are halved (first order about a
// run whose shorelines move inside its steps), the continuous Hessian
// symmetric to that order.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "KernelScene.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/StepRecipe.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/TangentLinear3D.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using namespace kernel_scene;

namespace {

/// The scene's Maxwell mantle as two branches: 0.7 of its shear modulus
/// with its relaxation time and 0.3 with a fifth of it.
Rheology TwoBranches(const Scene& setup) {
  const auto& one = setup.rheology.BranchAt(0);
  auto slow = one, fast = one;
  for (auto i = std::size_t{0}; i < one.mu.size(); ++i) {
    slow.mu[i] = 0.7 * one.mu[i];
    fast.mu[i] = 0.3 * one.mu[i];
    fast.itau[i] = 5.0 * one.itau[i];
  }
  return Rheology::Create(setup.material, {slow, fast}).value();
}

/// A rheology's branches with their inverse relaxation times moved by eps
/// times a direction (one array per branch).
Rheology MovedTimes(const Material& material, const Rheology& rheology,
                    const std::vector<std::vector<double>>& ditau, double eps) {
  auto branches = std::vector<Rheology::Branch>{};
  for (auto k = std::size_t{0}; k < rheology.BranchCount(); ++k) {
    auto b = rheology.BranchAt(k);
    for (auto i = std::size_t{0}; i < b.itau.size(); ++i) {
      b.itau[i] += eps * ditau[k][i];
    }
    branches.push_back(std::move(b));
  }
  return Rheology::Create(material, branches).value();
}

/// A change of ln viscosity p(r) on every branch as the branches' inverse
/// relaxation times take it, -itau_k p (nodal).
std::vector<std::vector<double>> LogViscosity(
    const Scene& setup, const Rheology& rheology,
    const std::function<double(double)>& p) {
  const auto& mesh = setup.material.Mesh();
  auto out = std::vector<std::vector<double>>{};
  for (auto k = std::size_t{0}; k < rheology.BranchCount(); ++k) {
    const auto& itau = rheology.BranchAt(k).itau;
    auto d = std::vector<double>(itau.size(), 0.0);
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      for (auto q = 0; q < mesh.NGll(); ++q) {
        const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
        d[i] = -itau[i] * p(mesh.Radius(e, q));
      }
    }
    out.push_back(std::move(d));
  }
  return out;
}

/// A direction in kappa and mu at an operator's memory points, a profile
/// in radius times the moduli there (kappa at half the weight).
PointMaterialDirection Moduli(const Scene& setup,
                              const ViscoelasticOperator& op,
                              const std::function<double(double)>& profile) {
  const auto& mesh = setup.material.Mesh();
  const auto& points = op.Points();
  const auto muU = op.UnrelaxedShearAtPoints();
  const auto kappa0 = op.BulkModulusAtPoints();
  PointMaterialDirection d;
  d.kappa.assign(muU.size(), 0.0);
  d.mu.assign(muU.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (setup.material.IsFluid(e)) continue;
    for (auto p = 0; p < points.count; ++p) {
      const auto i = static_cast<std::size_t>(e) * points.count + p;
      const auto value = 0.1 * profile(points.RadiusAt(mesh, e, p));
      d.kappa[i] = 0.5 * value * kappa0[i];
      d.mu[i] = value * muU[i];
    }
  }
  return d;
}

/// An operator with its moduli moved by eps times a direction at its
/// memory points (ViscoelasticOperator::Options::moduli, relative).
ViscoelasticOperator MovedModuli(const Rheology& rheology,
                                 const PointMaterialDirection& d, double eps) {
  const ViscoelasticOperator base(rheology, kLmax);
  const auto muU = base.UnrelaxedShearAtPoints();
  const auto kappa0 = base.BulkModulusAtPoints();
  ViscoelasticOperator::Options::PointModuli moduli;
  moduli.shear.assign(muU.size(), 0.0);
  moduli.bulk.assign(muU.size(), 0.0);
  for (auto i = std::size_t{0}; i < muU.size(); ++i) {
    if (muU[i] > 0.0 && !d.mu.empty()) moduli.shear[i] = eps * d.mu[i] / muU[i];
    if (kappa0[i] > 0.0 && !d.kappa.empty()) {
      moduli.bulk[i] = eps * d.kappa[i] / kappa0[i];
    }
  }
  return ViscoelasticOperator(rheology, kLmax, {.moduli = moduli});
}

/// The scene's elastic solver with the body's radius and gravity held, so
/// that a run on moved moduli closes the same sea level equation.
GRDSolver HeldSolver(const Scene& setup) {
  const auto table =
      ViscoelasticOperator(setup.rheology, kLmax).ElasticLoveNumbers();
  SeaLevelParameters held;
  held.body_from_table = false;
  held.mean_sea_floor_radius = table.radius;
  held.gravitational_acceleration = table.surface_gravity;
  return GRDSolver(SurfaceGrid(kLmax), table, held);
}

/// The largest difference of two runs' fields at every node (the sea
/// level, the displacement and the internal variables) over the largest
/// value of the first.
double RunGap(const std::vector<Evolution::Snapshot>& a,
              const std::vector<Evolution::Snapshot>& b) {
  auto diff = 0.0, size = 0.0;
  for (auto n = std::size_t{0}; n < a.size(); ++n) {
    for (const auto& [x, y] :
         {std::pair{&a[n].relativeSeaLevelChange, &b[n].relativeSeaLevelChange},
          std::pair{&a[n].displacement, &b[n].displacement}}) {
      diff = std::max(diff, MaxDiff(*x, *y));
      size = std::max(size, MaxAbs(*x));
    }
    if (a[n].internalState && b[n].internalState) {
      for (const auto& [l, m] : a[n].internalState->m) {
        diff = std::max(diff, MaxDiff(m, b[n].internalState->m.at(l)));
        size = std::max(size, MaxAbs(m));
      }
    }
  }
  return diff / size;
}

/// The frozen-shoreline load of a per-node thickness history relative to
/// its first node, node by node.
std::vector<std::vector<double>> FrozenLoads(
    const Scene& setup, const std::vector<std::vector<double>>& thickness) {
  auto loads = std::vector<std::vector<double>>{};
  const auto& C = setup.state.oceanFunction;
  const auto rhoI = setup.solver.IceDensity();
  for (const auto& T : thickness) {
    auto load = std::vector<double>(T.size());
    for (auto i = std::size_t{0}; i < T.size(); ++i) {
      load[i] = rhoI * (1.0 - C[i]) * (T[i] - thickness[0][i]);
    }
    loads.push_back(std::move(load));
  }
  return loads;
}

/// A field scaled by a profile in time at each node of a grid over span.
std::vector<std::vector<double>> InTime(
    const std::vector<double>& times, double span,
    const std::vector<double>& field, const std::function<double(double)>& f) {
  auto out = std::vector<std::vector<double>>{};
  for (auto t : times) {
    auto x = field;
    for (auto& v : x) v *= f(t / span);
    out.push_back(std::move(x));
  }
  return out;
}

/// A kernel object paired with a direction: the thickness view with the
/// thickness at every node, the itau kernels with the itau direction and
/// the material kernels with the moduli's.
template <class Kernels>
double Pairing(const Scene& setup, const ViscoelasticOperator& op,
               const Kernels& kernels, const TangentPerturbation& d) {
  const auto& mesh = setup.material.Mesh();
  auto total = 0.0;
  for (auto n = std::size_t{0}; n < d.iceThickness.size(); ++n) {
    total += setup.Surface(kernels.thickness[n], d.iceThickness[n]);
  }
  if (!d.initialSeaLevel.empty()) {
    total += setup.Surface(kernels.initialSeaLevel, d.initialSeaLevel);
  }
  for (auto k = std::size_t{0}; k < d.itau.size(); ++k) {
    total += VolumePairing(mesh, kernels.itauKernel[k], d.itau[k]);
  }
  if (!kernels.kappaKernel.empty()) {
    if (!d.material.kappa.empty()) {
      total += VolumePairing(op.Points(), mesh, kernels.kappaKernel,
                             d.material.kappa);
    }
    if (!d.material.mu.empty()) {
      total +=
          VolumePairing(op.Points(), mesh, kernels.muKernel, d.material.mu);
    }
  }
  return total;
}

/// The marine cap's thickness at the nodes of a grid: a fraction `start`
/// of its cap at the first node, thinning linearly to `end` of it.
std::vector<std::vector<double>> Thinning(const MarineCap& cap,
                                          std::span<const double> times,
                                          double start, double end) {
  auto out = std::vector<std::vector<double>>{};
  for (const auto t : times) {
    const auto s = t / times.back();
    auto ice = cap.ice0;
    for (auto& v : ice) v *= start + (end - start) * s;
    out.push_back(std::move(ice));
  }
  return out;
}

/// A per-node field history as a callable read at the nodes alone.
std::function<std::vector<double>(double)> AtNodes(
    std::vector<double> times, std::vector<std::vector<double>> fields) {
  return [times = std::move(times), fields = std::move(fields)](double t) {
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) {
        return fields[n];
      }
    }
    throw std::runtime_error("the test history is read off its nodes");
  };
}

/// About a migrating run of the marine cap thinning from `start` to `end`
/// of its cap, on a grid of `nodes`: the cells whose ocean function at a
/// step's inner stage differs from the step's end node's (summed over the
/// steps), and the continuous second-order sweep's derivatives along v
/// paired with u against the discrete sweep's, block by block (the
/// thickness view, the relaxation times, the initial sea level).
struct MigratingGaps {
  std::size_t innerFlips = 0, nodeFlips = 0, startFlips = 0;
  double ice = 0.0, viscosity = 0.0, sea = 0.0;
};
MigratingGaps MigratingSecondOrder(const Scene& setup,
                                   const ViscoelasticOperator& op, int nodes,
                                   double start, double end) {
  std::mt19937 rng(97);
  const auto fieldU = setup.Smooth(rng, 0.1 * setup.Metre());
  const auto fieldV = setup.Smooth(rng, 0.1 * setup.Metre());
  const auto seaU = setup.Smooth(rng, 0.1 * setup.Metre());
  const auto g = setup.Smooth(rng, 1.0);
  const auto gMid = setup.Smooth(rng, 0.7);
  const auto cap = MarineCap::Make(setup, nodes);
  const auto& times = cap.times;
  const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
  Evolution evolution(
      op, setup.solver, state,
      {.rtol = 1e-12, .max_nonlinear_iterations = 200, .record_stride = 1});
  const auto run = evolution.RunMigrating(
      times, AtNodes(times, Thinning(cap, times, start, end)),
      Evolution::Forcing{});
  const auto oceans = OceanFunctions::Of(run);
  const auto trajectory = ForwardTrajectory::FromSnapshots(run);
  auto out = MigratingGaps{};
  for (auto i = std::size_t{0}; i < state.oceanFunction.size(); ++i) {
    if (state.oceanFunction[i] != oceans.nodes[0][i]) ++out.startFlips;
  }
  for (auto n = std::size_t{1}; n < times.size(); ++n) {
    for (auto i = std::size_t{0}; i < oceans.nodes[n].size(); ++i) {
      if (oceans.nodes[n - 1][i] != oceans.nodes[n][i]) ++out.nodeFlips;
    }
    for (const auto& stage : oceans.stages[n]) {
      for (auto i = std::size_t{0}; i < stage.size(); ++i) {
        if (stage[i] != oceans.nodes[n][i]) ++out.innerFlips;
      }
    }
  }
  // u and v: thickness histories rising from zero at the first node, the
  // relaxation times, and u the initial sea level
  TangentPerturbation u, v;
  u.iceThickness =
      InTime(times, times.back(), fieldU, [](double s) { return s; });
  v.iceThickness = InTime(times, times.back(), fieldV,
                          [](double s) { return s * (2.0 - s); });
  u.itau = setup.DeltaItau(0.1, [](double r) { return 0.3 + 0.7 * r * r; });
  v.itau = setup.DeltaItau(0.1, [](double r) { return 1.0 + r; });
  u.initialSeaLevel = seaU;
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() / 2].displacementDual = gMid;
  measurements[times.size() - 1].relativeSeaLevelDual = g;
  struct Blocks {
    double ice = 0.0, viscosity = 0.0, sea = 0.0;
  };
  auto blocks = std::map<AdjointMethod, Blocks>{};
  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    TangentLinearSolver tangent(
        evolution, {.method = method, .rtol = 1e-12, .record = true});
    const auto runV = tangent.Run(times, v, &trajectory, oceans);
    const auto derivatives =
        AdjointSolver(evolution, {.method = method, .rtol = 1e-12})
            .SecondOrderKernels(times, measurements, runV, v.itau, trajectory,
                                oceans)
            .second;
    auto& b = blocks[method];
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      b.ice += setup.Surface(derivatives.thickness[n], u.iceThickness[n]);
    }
    b.viscosity = VolumePairing(setup.material.Mesh(),
                                derivatives.itauKernel[0], u.itau[0]);
    b.sea = setup.Surface(derivatives.initialSeaLevel, u.initialSeaLevel);
  }
  const auto& d = blocks[AdjointMethod::Discrete];
  const auto& c = blocks[AdjointMethod::Continuous];
  out.ice = std::abs(c.ice - d.ice) / std::abs(d.ice);
  out.viscosity = std::abs(c.viscosity - d.viscosity) / std::abs(d.viscosity);
  out.sea = std::abs(c.sea - d.sea) / std::abs(d.sea);
  return out;
}

}  // namespace

TEST(Sdirk2Continuous, TheContinuousTangentIsTheDiscrete) {
  // On a mantle of two branches, along a change of ln viscosity, one
  // branch's relaxation times, the moduli, and a thickness history, frozen
  // and about the marine cap's migrating run (its shorelines moving inside
  // the steps): the continuous tangent-linear run, first and second
  // order, is the discrete one to rounding; and against central
  // differences of the forward runs it lands at second order, Richardson
  // at 1e-6. The rounding is the stage states' amplified by the stiffest
  // branch: the continuous drive forms d - m = w (d - m_start), a
  // difference of nearly equal numbers where w = 1 / (1 + h) is small
  // (measured 9.6e-12 to 9.7e-11 here, 0 along the moduli, the same at
  // solver tolerances from 1e-10 to 1e-14, and ten times smaller or larger
  // with the viscosity ten times larger or smaller).
  ASSERT_EQ(Evolution::Options{}.scheme, Scheme::Sdirk2);
  auto setup = Scene::Make(1e21);
  const auto rheology = TwoBranches(setup);
  const auto solver = HeldSolver(setup);
  ViscoelasticOperator op(rheology, kLmax);
  std::mt19937 rng(71);
  for (const bool migrating : {false, true}) {
    const auto cap = MarineCap::Make(setup, 6);
    const auto& times = cap.times;
    const auto nodes = times.size();
    const auto state =
        migrating ? setup.solver.MakeState(cap.ice0, cap.sea0) : setup.state;
    auto loads = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
    measurements[nodes - 1].displacementDual = setup.Smooth(rng, 0.5);
    measurements[2].potentialDual = setup.Smooth(rng, 0.7);
    const Evolution::Options options{
        .rtol = 1e-12, .max_nonlinear_iterations = 200, .record_stride = 1};
    auto runOf = [&](const ViscoelasticOperator& at) {
      Evolution ev(at, migrating ? setup.solver : solver, state, options);
      return migrating
                 ? ev.RunMigrating(times,
                                   [cap](double t) { return cap.Thickness(t); })
                 : ev.Run(times, History(times, loads));
    };
    Evolution evolution(op, migrating ? setup.solver : solver, state, options);
    const auto run =
        migrating ? evolution.RunMigrating(
                        times, [cap](double t) { return cap.Thickness(t); })
                  : evolution.Run(times, History(times, loads));
    const auto trajectory = ForwardTrajectory::FromSnapshots(run);
    const auto oceans = migrating ? OceanFunctions::Of(run) : OceanFunctions{};

    auto logViscosity = TangentPerturbation{};
    logViscosity.itau = LogViscosity(
        setup, rheology, [](double r) { return 0.3 + 0.7 * r * r; });
    auto oneBranch = TangentPerturbation{};
    oneBranch.itau = logViscosity.itau;
    std::ranges::fill(oneBranch.itau[0], 0.0);
    auto moduli = TangentPerturbation{};
    moduli.material = Moduli(setup, op, [](double r) {
      return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r);
    });
    auto thickness = TangentPerturbation{};
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      thickness.iceThickness.push_back(setup.Smooth(rng, 0.1 * setup.Metre()));
    }
    thickness.itau = logViscosity.itau;
    // the second-order run's first direction: the viscosity and the moduli
    // together
    auto both = TangentPerturbation{};
    both.itau = logViscosity.itau;
    both.material = moduli.material;
    const auto record = [](AdjointMethod method) {
      return TangentLinearSolver::Options{
          .method = method, .rtol = 1e-12, .record = true};
    };
    TangentLinearSolver discrete(evolution, record(AdjointMethod::Discrete));
    TangentLinearSolver continuous(evolution,
                                   record(AdjointMethod::Continuous));
    auto runs =
        std::map<std::string, std::pair<std::vector<Evolution::Snapshot>,
                                        std::vector<Evolution::Snapshot>>>{};
    for (const auto& [name, d] :
         {std::pair{"ln viscosity", &logViscosity},
          std::pair{"one branch", &oneBranch}, std::pair{"moduli", &moduli},
          std::pair{"viscosity and moduli", &both},
          std::pair{"thickness", &thickness}}) {
      auto a = discrete.Run(times, *d, &trajectory, oceans);
      auto b = continuous.Run(times, *d, &trajectory, oceans);
      const auto gap = RunGap(a, b);
      std::cout << std::format(
          "migrating {} {}: continuous - discrete {:.2e}\n", migrating, name,
          gap);
      EXPECT_LT(gap, 1e-9) << name << " migrating " << migrating;
      runs.emplace(name, std::pair{std::move(a), std::move(b)});
    }
    // the second-order run along (u, v), each method from its own runs
    SecondOrderDirection uv;
    uv.itauU = logViscosity.itau;
    uv.itauV = thickness.itau;
    uv.materialU = moduli.material;
    const auto& [uD, uC] = runs.at("viscosity and moduli");
    const auto& [vD, vC] = runs.at("thickness");
    const auto secondD =
        discrete.RunSecondOrder(times, uD, vD, uv, &trajectory, oceans);
    const auto secondC =
        continuous.RunSecondOrder(times, uC, vC, uv, &trajectory, oceans);
    const auto secondGap = RunGap(secondD, secondC);
    std::cout << std::format(
        "migrating {} second order: continuous - discrete {:.2e}\n", migrating,
        secondGap);
    EXPECT_LT(secondGap, 1e-9) << "migrating " << migrating;
    if (migrating) continue;  // differences of migrating runs flip cells

    // against central differences of the forward runs (frozen)
    for (const auto& [name, d] : {std::pair{"ln viscosity", &logViscosity},
                                  std::pair{"moduli", &moduli}}) {
      const auto predicted =
          Objective(setup, runs.at(name).second, measurements);
      auto at = [&](double eps) {
        if (!d->itau.empty()) {
          const auto moved = MovedTimes(setup.material, rheology, d->itau, eps);
          return Objective(setup, runOf(ViscoelasticOperator(moved, kLmax)),
                           measurements);
        }
        return Objective(setup, runOf(MovedModuli(rheology, d->material, eps)),
                         measurements);
      };
      auto fd = std::map<double, double>{};
      for (double eps : {0.02, 0.01})
        fd[eps] = (at(eps) - at(-eps)) / (2.0 * eps);
      const auto r1 = std::abs(fd[0.02] - predicted);
      const auto r2 = std::abs(fd[0.01] - predicted);
      const auto richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0;
      std::cout << std::format(
          "{}: differences off by {:.3e}, {:.3e} (ratio {:.3f}), Richardson "
          "{:.2e}\n",
          name, r1 / std::abs(predicted), r2 / std::abs(predicted), r1 / r2,
          std::abs(richardson - predicted) / std::abs(predicted));
      EXPECT_GT(r1 / r2, 3.0) << name;
      EXPECT_LT(r1 / r2, 5.0) << name;
      EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted)) << name;
    }
  }
}

TEST(Sdirk2Continuous, TheContinuousHessianIsTheDerivativeOfItsKernels) {
  // The continuous second-order sweep steps the adjoint state and its
  // derivative together, stage by stage, so its derivatives along v are
  // the exact derivative of the continuous first-order kernels: central
  // differences of those kernels along v (the relaxation times, the
  // moduli and the thickness moved), paired with u, converge to them at
  // second order and Richardson lands at 1e-6, on a mantle of two
  // branches, at a fixed step (measured: ratio 4.001, Richardson 1.8e-8).
  auto setup = Scene::Make(1e21);
  const auto rheology = TwoBranches(setup);
  const auto solver = HeldSolver(setup);
  ViscoelasticOperator op(rheology, kLmax);
  std::mt19937 rng(72);
  auto times = std::vector<double>{};
  for (auto n = 0; n < 6; ++n) times.push_back(2.0 * n * setup.Kyr());
  const auto span = times.back();
  const auto fieldT = setup.Smooth(rng, 100.0 * setup.Metre());
  const auto thickness =
      InTime(times, span, fieldT, [](double s) { return 1.0 - s; });
  TangentPerturbation u, v;
  u.iceThickness = InTime(times, span, setup.Smooth(rng, 20.0 * setup.Metre()),
                          [](double s) { return s; });
  u.itau =
      LogViscosity(setup, rheology, [](double r) { return 0.8 - 0.5 * r; });
  u.material = Moduli(setup, op, [](double r) { return 1.0 - 0.5 * r; });
  v.iceThickness = InTime(times, span, setup.Smooth(rng, 20.0 * setup.Metre()),
                          [](double s) { return (1.0 - s) * (1.0 - s); });
  v.itau =
      LogViscosity(setup, rheology, [](double r) { return 0.3 + 0.7 * r * r; });
  v.material = Moduli(setup, op, [](double r) {
    return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r);
  });
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[times.size() - 1].displacementDual = setup.Smooth(rng, 0.5);
  measurements[3].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  const AdjointSolver::Options adjointOptions{
      .method = AdjointMethod::Continuous,
      .rtol = 1e-12,
      .material_kernels = true};
  // u's ln-viscosity direction is a direction in itau linear in it: held
  // at the base's in the pairing.
  Evolution evolution(op, solver, setup.state,
                      {.rtol = 1e-12, .record_stride = 1});
  const auto trajectory = ForwardTrajectory::FromSnapshots(
      evolution.Run(times, History(times, FrozenLoads(setup, thickness))));
  TangentLinearSolver tangent(
      evolution,
      {.method = AdjointMethod::Continuous, .rtol = 1e-12, .record = true});
  const auto runV = tangent.Run(times, v, &trajectory);
  AdjointSolver adjoint(evolution, adjointOptions);
  const auto [kernels, derivatives] = adjoint.SecondOrderKernels(
      times, measurements, runV, v.itau, trajectory, {}, v.material);
  const auto predicted = Pairing(setup, op, derivatives, u);

  auto gradientAt = [&](double eps) {
    const auto movedRheology =
        MovedTimes(setup.material, rheology, v.itau, eps);
    const auto movedOp = MovedModuli(movedRheology, v.material, eps);
    Evolution moved(movedOp, solver, setup.state,
                    {.rtol = 1e-12, .record_stride = 1});
    auto movedThickness = thickness;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      for (auto i = std::size_t{0}; i < movedThickness[n].size(); ++i) {
        movedThickness[n][i] += eps * v.iceThickness[n][i];
      }
    }
    const auto loads = FrozenLoads(setup, movedThickness);
    const auto movedTrajectory = ForwardTrajectory::FromSnapshots(
        moved.Run(times, History(times, loads)));
    const auto k =
        AdjointSolver(moved, adjointOptions)
            .FirstOrderKernels(times, measurements, &movedTrajectory);
    return Pairing(setup, op, k, u);
  };
  auto fd = std::map<double, double>{};
  for (double eps : {0.02, 0.01}) {
    fd[eps] = (gradientAt(eps) - gradientAt(-eps)) / (2.0 * eps);
  }
  const auto r1 = std::abs(fd[0.02] - predicted);
  const auto r2 = std::abs(fd[0.01] - predicted);
  const auto richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0;
  std::cout << std::format(
      "continuous kernels' differences off by {:.3e}, {:.3e} (ratio {:.3f}), "
      "Richardson {:.2e}\n",
      r1 / std::abs(predicted), r2 / std::abs(predicted), r1 / r2,
      std::abs(richardson - predicted) / std::abs(predicted));
  EXPECT_GT(r1 / r2, 3.0);
  EXPECT_LT(r1 / r2, 5.0);
  EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted));
}

TEST(Sdirk2Continuous, TheContinuousSecondOrderConvergesToTheDiscrete) {
  // On a slowly relaxing mantle, along directions u and v in the
  // thickness, the relaxation times and the moduli: the continuous
  // derivatives along v paired with u tend to the discrete ones at second
  // order as the steps are halved, block by block (ice, viscosity,
  // material); the continuous Hessian's asymmetry, <H v, u> against
  // <H u, v>, falls at second order; and so does the continuous Hessian's
  // gap to the continuous second-order run (the discrete one's, the two
  // runs being one). Measured from 9 to 65 nodes over 10 kyr, every ratio
  // 3.96 to 4.01 (the gaps at 9 nodes 4.7e-4 to 6.0e-3); gated at 9, 17
  // and 33.
  auto setup = Scene::Make(1e22);
  ViscoelasticOperator op(setup.rheology, kLmax);
  std::mt19937 rng(11);
  const auto fieldT = setup.Smooth(rng, 100.0 * setup.Metre());
  const auto fieldU = setup.Smooth(rng, 20.0 * setup.Metre());
  const auto fieldV = setup.Smooth(rng, 20.0 * setup.Metre());
  const auto g = setup.Smooth(rng, 1.0);
  const auto gTangential = setup.Smooth(rng, 1.0);
  const auto gDisplacement = setup.Smooth(rng, 0.7);
  const auto span = 10.0 * setup.Kyr();
  auto gaps = std::map<std::string, std::vector<double>>{};
  for (const auto nodes : {9, 17, 33}) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
    const auto thickness =
        InTime(times, span, fieldT, [](double s) { return 1.0 - s; });
    Evolution evolution(op, setup.solver, setup.state,
                        {.rtol = 1e-12, .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, History(times, FrozenLoads(setup, thickness))));
    TangentPerturbation u, v;
    u.iceThickness = InTime(times, span, fieldU, [](double s) { return s; });
    u.itau = setup.DeltaItau(0.1, [](double r) { return 0.3 + 0.7 * r * r; });
    u.material = Moduli(setup, op, [](double r) {
      return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r);
    });
    v.iceThickness = InTime(times, span, fieldV,
                            [](double s) { return (1.0 - s) * (1.0 - s); });
    v.itau = setup.DeltaItau(0.1, [](double r) { return 1.0 + r; });
    v.material = Moduli(setup, op, [](double r) { return 1.0 - 0.5 * r; });
    auto measurement = std::map<std::size_t, Measurement>{};
    measurement[times.size() - 1].relativeSeaLevelDual = g;
    measurement[times.size() - 1].tangentialPotentialDual = gTangential;
    measurement[times.size() - 1].displacementDual = gDisplacement;
    struct Blocks {
      double ice, viscosity, material, other, forward;
    };
    auto results = std::map<AdjointMethod, Blocks>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver tangent(
          evolution, {.method = method, .rtol = 1e-12, .record = true});
      const auto runV = tangent.Run(times, v, &trajectory);
      const auto runU = tangent.Run(times, u, &trajectory);
      AdjointSolver adjoint(
          evolution,
          {.method = method, .rtol = 1e-12, .material_kernels = true});
      const auto alongV =
          adjoint
              .SecondOrderKernels(times, measurement, runV, v.itau, trajectory,
                                  {}, v.material)
              .second;
      const auto alongU =
          adjoint
              .SecondOrderKernels(times, measurement, runU, u.itau, trajectory,
                                  {}, u.material)
              .second;
      SecondOrderDirection vu;
      vu.itauU = v.itau;
      vu.itauV = u.itau;
      vu.materialU = v.material;
      vu.materialV = u.material;
      const auto second =
          tangent.RunSecondOrder(times, runV, runU, vu, &trajectory);
      auto ice = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        ice += setup.Surface(alongV.thickness[n], u.iceThickness[n]);
      }
      const auto& mesh = setup.material.Mesh();
      const auto viscosity =
          VolumePairing(mesh, alongV.itauKernel[0], u.itau[0]);
      const auto material =
          VolumePairing(op.Points(), mesh, alongV.kappaKernel,
                        u.material.kappa) +
          VolumePairing(op.Points(), mesh, alongV.muKernel, u.material.mu);
      results[method] = {ice, viscosity, material,
                         Pairing(setup, op, alongU, v),
                         Objective(setup, second, measurement)};
    }
    const auto& d = results[AdjointMethod::Discrete];
    const auto& c = results[AdjointMethod::Continuous];
    const auto hd = d.ice + d.viscosity + d.material;
    const auto hc = c.ice + c.viscosity + c.material;
    EXPECT_NEAR(hd, d.forward, 1e-9 * std::abs(d.forward)) << "nodes " << nodes;
    EXPECT_NEAR(hd, d.other, 1e-9 * std::abs(hd)) << "nodes " << nodes;
    EXPECT_NEAR(c.forward, d.forward, 1e-11 * std::abs(d.forward))
        << "nodes " << nodes;
    gaps["ice"].push_back(std::abs(d.ice - c.ice) / std::abs(d.ice));
    gaps["viscosity"].push_back(std::abs(d.viscosity - c.viscosity) /
                                std::abs(d.viscosity));
    gaps["material"].push_back(std::abs(d.material - c.material) /
                               std::abs(d.material));
    gaps["hessian"].push_back(std::abs(hd - hc) / std::abs(hd));
    gaps["asymmetry"].push_back(std::abs(hc - c.other) / std::abs(hd));
    gaps["run"].push_back(std::abs(hc - c.forward) / std::abs(c.forward));
  }
  for (const auto& [name, g] : gaps) {
    auto line = std::format("[1D, continuous against discrete] {:10}", name);
    for (auto i = std::size_t{0}; i < g.size(); ++i) {
      line += std::format(" {:.3e}", g[i]);
      if (i > 0) line += std::format(" ({:.2f})", g[i - 1] / g[i]);
    }
    std::cout << line << "\n";
    EXPECT_LT(g.front(), 1e-2) << name;
    for (auto i = std::size_t{1}; i < g.size(); ++i) {
      EXPECT_GT(g[i - 1] / g[i], 3.5) << name << " " << i;
      EXPECT_LT(g[i - 1] / g[i], 4.5) << name << " " << i;
    }
  }
}

TEST(Sdirk2Continuous, AboutShorelinesMovingAtTheNodesAlone) {
  // About a migrating run whose shorelines move at its solved start and
  // nowhere else (the marine cap at 85% of its cap thinning to 80%, on a
  // slowly relaxing mantle): every step's inner stage closes under its end
  // node's ocean function, the continuous sweep's inner stage under the
  // same, and the continuous derivatives along v paired with u tend to the
  // discrete ones at second order, block by block (the thickness view, the
  // relaxation times, the initial sea level). Measured from 5 to 65 nodes
  // the ratios 3.84 to 4.29, 3.96 to 4.01 from 17; gated at 9, 17 and 33.
  auto setup = Scene::Make(1e22);
  ViscoelasticOperator op(setup.rheology, kLmax);
  auto gaps = std::vector<MigratingGaps>{};
  for (const auto nodes : {9, 17, 33}) {
    gaps.push_back(MigratingSecondOrder(setup, op, nodes, 0.85, 0.80));
    const auto& g = gaps.back();
    std::cout << std::format(
        "[about shorelines moving at the start, {} nodes] cells flipped at the "
        "start {}, between nodes {}, inside steps {}; continuous against "
        "discrete: ice {:.3e}, viscosity {:.3e}, initial sea level {:.3e}\n",
        nodes, g.startFlips, g.nodeFlips, g.innerFlips, g.ice, g.viscosity,
        g.sea);
    EXPECT_GT(g.startFlips, 0u) << nodes;
    EXPECT_EQ(g.nodeFlips, 0u) << nodes;
    EXPECT_EQ(g.innerFlips, 0u) << nodes;
  }
  EXPECT_LT(gaps.front().ice, 2e-2);
  EXPECT_LT(gaps.front().viscosity, 2e-3);
  EXPECT_LT(gaps.front().sea, 4e-3);
  for (auto i = std::size_t{1}; i < gaps.size(); ++i) {
    for (const auto& [name, coarse, fine] :
         {std::tuple{"ice", gaps[i - 1].ice, gaps[i].ice},
          std::tuple{"viscosity", gaps[i - 1].viscosity, gaps[i].viscosity},
          std::tuple{"initial sea level", gaps[i - 1].sea, gaps[i].sea}}) {
      EXPECT_GT(coarse / fine, 3.5) << name << " " << i;
      EXPECT_LT(coarse / fine, 4.5) << name << " " << i;
    }
  }
}

TEST(Sdirk2Continuous, AboutShorelinesMovingInsideTheSteps) {
  // About the marine cap thinning from 85% of its cap to 35%, whose
  // shorelines move inside the steps (two cells whose ocean function at a
  // step's inner stage is not its end node's): the continuous sweep's
  // inner stage closes under
  // the run's shorelines interpolated in time where the discrete closes
  // under the run's own, and in the flipping cells the continuous
  // derivatives differ from the discrete at first order. Measured: erratic
  // up to 65 nodes (where in a step the flips fall moves with the grid),
  // then ratios 2.01 / 2.01 (ice), 2.46 / 2.04 (viscosity), 2.86 / 2.54
  // (initial sea level) from 129 to 257 to 513 nodes; gated at 129 and
  // 257.
  auto setup = Scene::Make(1e22);
  ViscoelasticOperator op(setup.rheology, kLmax);
  const auto coarse = MigratingSecondOrder(setup, op, 129, 0.85, 0.35);
  const auto fine = MigratingSecondOrder(setup, op, 257, 0.85, 0.35);
  for (const auto* g : {&coarse, &fine}) {
    std::cout << std::format(
        "[about shorelines moving inside the steps] cells flipped inside steps "
        "{}; continuous against discrete: ice {:.3e}, viscosity {:.3e}, "
        "initial "
        "sea level {:.3e}\n",
        g->innerFlips, g->ice, g->viscosity, g->sea);
    EXPECT_GT(g->innerFlips, 0u);
  }
  for (const auto& [name, c, f] :
       {std::tuple{"ice", coarse.ice, fine.ice},
        std::tuple{"viscosity", coarse.viscosity, fine.viscosity},
        std::tuple{"initial sea level", coarse.sea, fine.sea}}) {
    EXPECT_LT(c, 1e-4) << name;
    EXPECT_GT(c / f, 1.6) << name;
    EXPECT_LT(c / f, 3.5) << name;
  }
}

// -- 3D, under lateral viscosity ------------------------------------------

namespace {

constexpr auto kLmax3 = 8;

/// The prem8 body with its Maxwell mantle at degree 8, its elastic solver
/// and the analytic test state: the 3D second-order gates' scene.
struct Scene3D {
  Material material;
  Rheology rheology;
  GRDSolver solver;
  GRDSolver::State state;

  static Scene3D Make(double viscositySI) {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax3}).value();
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
    ViscoelasticOperator reference(rheology, kLmax3);
    GRDSolver solver(SurfaceGrid(kLmax3), reference.ElasticLoveNumbers(),
                     SeaLevelParameters{});
    Fingerprint fingerprint(kLmax3, reference.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto state = solver.MakeState(toNd(fingerprint.IceThickness()),
                                  toNd(fingerprint.SeaLevel()));
    return Scene3D{std::move(material), std::move(rheology), std::move(solver),
                   std::move(state)};
  }

  [[nodiscard]] double Kyr() const {
    return 1e3 * kYear / solver.SolverScales().Time();
  }
  [[nodiscard]] double Metre() const {
    return 1.0 / solver.SolverScales().Length();
  }
  [[nodiscard]] std::vector<double> Smooth(std::mt19937& rng,
                                           double amplitude) const {
    const auto& grid = solver.Grid();
    std::normal_distribution<double> normal;
    auto coeffs = grid.ZeroCoefficients();
    for (auto l = 0; l <= kLmax3; ++l) {
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

double Cap(double theta, double phi, double colat0, double lon0, double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

/// The lateral structure of the 3D gates: the background's ln viscosity
/// and the directions' changes of it.
double BaseDelta(double, double theta, double phi) {
  return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
}
double PU(double theta, double phi) {
  return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
}
double PW(double theta, double phi) {
  return 0.5 * Cap(theta, phi, 2.4, 2.0, 0.8) + 0.3 * std::cos(theta);
}

std::vector<double> OnCoupling(const GridCoupling& coupling,
                               const std::function<double(double, double)>& f) {
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto values = coupling.GridOf(rn).FromFunction(f);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x)
      out[at + x] = values[x];
  }
  return out;
}

/// A radial profile times an angular function over the grid values, in
/// units of a nodal modulus.
std::vector<double> LateralModulus(
    const GridCoupling& coupling, const Material& material,
    std::span<const double> modulus,
    const std::function<double(double)>& radial,
    const std::function<double(double, double)>& angular) {
  const auto ngll = material.Mesh().NGll();
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full =
        static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
        coupling.NodeInElementOf(rn);
    const auto r = coupling.RadiusOfNode(rn);
    const auto values =
        coupling.GridOf(rn).FromFunction([&](double theta, double phi) {
          return modulus[full] * radial(r) * angular(theta, phi);
        });
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x)
      out[at + x] = values[x];
  }
  return out;
}

double GridPairing(const GridCoupling& coupling,
                   const std::vector<std::vector<double>>& kernel,
                   const std::vector<std::vector<double>>& direction) {
  auto total = 0.0;
  for (auto k = std::size_t{0}; k < kernel.size(); ++k) {
    total += VolumePairing(coupling, kernel[k], direction[k]);
  }
  return total;
}

/// A 3D direction: the thickness at every node, a change of ln viscosity
/// (as the relaxation times take it) and, optionally, of the moduli.
struct Direction3D {
  TangentPerturbation3D tangent;
  std::function<double(double, double)> p;
};

/// The pairing of a kernel object (kernels or derivatives) with one.
template <class Kernels>
double Pairing3D(const Scene3D& scene, const GridCoupling& coupling,
                 const Kernels& k, const Direction3D& d) {
  auto total = 0.0;
  for (auto n = std::size_t{0}; n < d.tangent.iceThickness.size(); ++n) {
    total += scene.Surface(k.ice.thickness[n], d.tangent.iceThickness[n]);
  }
  total += GridPairing(coupling, k.itauKernel3D, d.tangent.itau);
  if (!d.tangent.material.kappa.empty() && !k.kappaKernel3D.empty()) {
    total += VolumePairing(coupling, k.kappaKernel3D, d.tangent.material.kappa);
    total += VolumePairing(coupling, k.muKernel3D, d.tangent.material.mu);
  }
  return total;
}

/// The run on an operator with ln viscosity BaseDelta + eps p, under the
/// default scheme, and the frozen-shoreline load of a thickness history
/// relative to its first node.
struct Run3D {
  std::unique_ptr<Viscoelastic3D> op;
  std::unique_ptr<Evolution3D> evolution;
  ForwardTrajectory3D trajectory;
};
Run3D RunOn(const Scene3D& scene, std::span<const double> times,
            const std::vector<std::vector<double>>& thickness, double eps,
            const std::function<double(double, double)>& p) {
  auto out = Run3D{};
  out.op = std::make_unique<Viscoelastic3D>(
      scene.rheology, kLmax3, [eps, p](double r, double theta, double phi) {
        return BaseDelta(r, theta, phi) + eps * p(theta, phi);
      });
  out.evolution = std::make_unique<Evolution3D>(
      *out.op, scene.solver, scene.state,
      Evolution3D::Options{
          .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto& C = scene.state.oceanFunction;
  const auto rhoI = scene.solver.IceDensity();
  auto loads = std::vector<std::vector<double>>{};
  for (const auto& T : thickness) {
    auto load = std::vector<double>(T.size());
    for (auto i = std::size_t{0}; i < T.size(); ++i) {
      load[i] = rhoI * (1.0 - C[i]) * (T[i] - thickness[0][i]);
    }
    loads.push_back(std::move(load));
  }
  out.trajectory = ForwardTrajectory3D::FromSnapshots(out.evolution->Run(
      times, AtNodes(std::vector<double>(times.begin(), times.end()), loads)));
  return out;
}

/// The largest difference of two 3D runs' surface fields at every node
/// over the largest value of the first's.
double RunGap3D(const std::vector<Evolution3D::Snapshot>& a,
                const std::vector<Evolution3D::Snapshot>& b) {
  auto diff = 0.0, size = 0.0;
  for (auto n = std::size_t{0}; n < a.size(); ++n) {
    for (const auto& [x, y] :
         {std::pair{&a[n].relativeSeaLevelChange, &b[n].relativeSeaLevelChange},
          std::pair{&a[n].displacement, &b[n].displacement},
          std::pair{&a[n].potential, &b[n].potential}}) {
      diff = std::max(diff, MaxDiff(*x, *y));
      size = std::max(size, MaxAbs(*x));
    }
  }
  return diff / size;
}

/// A grid of `nodes` over 10 kyr.
std::vector<double> Grid3D(const Scene3D& scene, int nodes) {
  auto times = std::vector<double>{};
  for (auto n = 0; n < nodes; ++n)
    times.push_back(10.0 * scene.Kyr() * n / (nodes - 1));
  return times;
}

}  // namespace

TEST(Sdirk2Continuous, TheContinuousTangentIsTheDiscrete3D) {
  // Under lateral viscosity, along a lateral change of ln viscosity, of
  // the moduli, and of the thickness with ln viscosity, and the
  // second-order run along two of them: the continuous run is the
  // discrete one to the coupled iteration's tolerance, 1e-12 here
  // (measured 1.1e-13 to 4.5e-13, 0 along the moduli).
  auto scene = Scene3D::Make(1e22);
  std::mt19937 rng(17);
  const auto times = Grid3D(scene, 6);
  const auto span = times.back();
  const auto thickness =
      InTime(times, span, scene.Smooth(rng, 100.0 * scene.Metre()),
             [](double s) { return 1.0 - s; });
  const auto run = RunOn(scene, times, thickness, 0.0, PU);
  const auto& op = *run.op;
  const auto& coupling = op.Coupling();
  const auto kappaMu = scene.material.Moduli().KappaMu();
  TangentPerturbation3D logViscosity, moduli, thicknessAndViscosity;
  logViscosity.itau =
      TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, PU));
  moduli.material.kappa = LateralModulus(
      coupling, scene.material, kappaMu.first,
      [](double r) { return 0.05 * (1.0 - 0.5 * r); },
      [](double theta, double phi) {
        return 1.0 + 0.7 * std::cos(theta) * std::sin(phi);
      });
  moduli.material.mu = LateralModulus(
      coupling, scene.material, kappaMu.second,
      [](double r) { return 0.1 * (0.5 + r * r); },
      [](double theta, double phi) {
        return 1.0 + 0.5 * std::sin(theta) * std::sin(phi);
      });
  thicknessAndViscosity.iceThickness =
      InTime(times, span, scene.Smooth(rng, 20.0 * scene.Metre()),
             [](double s) { return s; });
  thicknessAndViscosity.itau =
      TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, PW));
  const auto options = [](AdjointMethod method) {
    return TangentLinearSolver3D::Options{
        .method = method, .rtol = 1e-12, .coupled_rtol = 1e-12, .record = true};
  };
  TangentLinearSolver3D discrete(*run.evolution,
                                 options(AdjointMethod::Discrete));
  TangentLinearSolver3D continuous(*run.evolution,
                                   options(AdjointMethod::Continuous));
  auto runs =
      std::map<std::string, std::pair<std::vector<Evolution3D::Snapshot>,
                                      std::vector<Evolution3D::Snapshot>>>{};
  for (const auto& [name, d] :
       {std::pair{"ln viscosity", &logViscosity}, std::pair{"moduli", &moduli},
        std::pair{"thickness", &thicknessAndViscosity}}) {
    auto a = discrete.Run(times, *d, &run.trajectory);
    auto b = continuous.Run(times, *d, &run.trajectory);
    const auto gap = RunGap3D(a, b);
    std::cout << std::format("3D {}: continuous - discrete {:.2e}\n", name,
                             gap);
    EXPECT_LT(gap, 1e-10) << name;
    runs.emplace(name, std::pair{std::move(a), std::move(b)});
  }
  SecondOrderDirection3D uw;
  uw.itauU = logViscosity.itau;
  uw.itauV = thicknessAndViscosity.itau;
  uw.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(
      op, OnCoupling(coupling, PU), OnCoupling(coupling, PW));
  const auto& [uD, uC] = runs.at("ln viscosity");
  const auto& [wD, wC] = runs.at("thickness");
  const auto gap =
      RunGap3D(discrete.RunSecondOrder(times, uD, wD, uw, &run.trajectory),
               continuous.RunSecondOrder(times, uC, wC, uw, &run.trajectory));
  std::cout << std::format("3D second order: continuous - discrete {:.2e}\n",
                           gap);
  EXPECT_LT(gap, 1e-10);
}

TEST(Sdirk2Continuous, TheContinuousHessianIsTheDerivativeOfItsKernels3D) {
  // Under lateral viscosity, along u (a thickness history and a lateral
  // change of ln viscosity) paired with w (the same kinds): central
  // differences of the continuous 3D kernels paired with w (the operator's
  // ln viscosity moved by eps p_u, the thickness by eps dI_u, w's
  // relaxation times taken at the moved operator) converge to <H u, w>
  // plus the log parameterisation's uv term at second order in eps,
  // Richardson landing on it, at a fixed step (measured: ratio 4.000,
  // Richardson 6.5e-10).
  auto scene = Scene3D::Make(1e22);
  std::mt19937 rng(19);
  const auto times = Grid3D(scene, 6);
  const auto span = times.back();
  const auto thickness =
      InTime(times, span, scene.Smooth(rng, 100.0 * scene.Metre()),
             [](double s) { return 1.0 - s; });
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].relativeSeaLevelDual =
      scene.solver.Grid().FromFunction(
          [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  measurements[3].displacementDual = scene.Smooth(rng, 0.5);
  const AdjointSolver3D::Options adjointOptions{
      .method = AdjointMethod::Continuous,
      .rtol = 1e-12,
      .coupled_rtol = 1e-12};
  const auto run = RunOn(scene, times, thickness, 0.0, PU);
  const auto& coupling = run.op->Coupling();
  auto direction = [&](const std::vector<double>& field,
                       const std::function<double(double)>& profile,
                       const std::function<double(double, double)>& p,
                       const Viscoelastic3D& at) {
    Direction3D d;
    d.tangent.iceThickness = InTime(times, span, field, profile);
    d.tangent.itau = TangentLinearSolver3D::ItauOfLogViscosity(
        at, OnCoupling(at.Coupling(), p));
    d.p = p;
    return d;
  };
  const auto fieldU = scene.Smooth(rng, 20.0 * scene.Metre());
  const auto fieldW = scene.Smooth(rng, 20.0 * scene.Metre());
  const auto u = direction(fieldU, [](double s) { return s; }, PU, *run.op);
  const auto w = direction(
      fieldW, [](double s) { return (1.0 - s) * (1.0 - s); }, PW, *run.op);
  TangentLinearSolver3D tangent(*run.evolution,
                                {.method = AdjointMethod::Continuous,
                                 .rtol = 1e-12,
                                 .coupled_rtol = 1e-12,
                                 .record = true});
  const auto runU = tangent.Run(times, u.tangent, &run.trajectory);
  AdjointSolver3D adjoint(*run.evolution, adjointOptions);
  const auto [kernels, alongU] = adjoint.SecondOrderKernels(
      times, measurements, runU, u.tangent.itau, run.trajectory);
  const auto uvTerm = GridPairing(
      coupling, kernels.itauKernel3D,
      TangentLinearSolver3D::SecondVariationOfLogViscosity(
          *run.op, OnCoupling(coupling, PU), OnCoupling(coupling, PW)));
  const auto predicted = Pairing3D(scene, coupling, alongU, w) + uvTerm;
  auto gradientAt = [&](double eps) {
    auto moved = thickness;
    for (auto n = std::size_t{0}; n < moved.size(); ++n) {
      for (auto i = std::size_t{0}; i < moved[n].size(); ++i) {
        moved[n][i] += eps * u.tangent.iceThickness[n][i];
      }
    }
    const auto at = RunOn(scene, times, moved, eps, PU);
    const auto k = AdjointSolver3D(*at.evolution, adjointOptions)
                       .FirstOrderKernels(times, measurements, &at.trajectory);
    const auto wAt = direction(
        fieldW, [](double s) { return (1.0 - s) * (1.0 - s); }, PW, *at.op);
    return Pairing3D(scene, at.op->Coupling(), k, wAt);
  };
  auto fd = std::map<double, double>{};
  for (double eps : {0.04, 0.02})
    fd[eps] = (gradientAt(eps) - gradientAt(-eps)) / (2.0 * eps);
  const auto r1 = std::abs(fd[0.04] - predicted);
  const auto r2 = std::abs(fd[0.02] - predicted);
  const auto richardson = (4.0 * fd[0.02] - fd[0.04]) / 3.0;
  std::cout << std::format(
      "3D continuous kernels' differences off by {:.3e}, {:.3e} (ratio "
      "{:.3f}), "
      "Richardson {:.2e}\n",
      r1 / std::abs(predicted), r2 / std::abs(predicted), r1 / r2,
      std::abs(richardson - predicted) / std::abs(predicted));
  EXPECT_GT(r1 / r2, 3.0);
  EXPECT_LT(r1 / r2, 5.0);
  EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted));
}

TEST(Sdirk2Continuous, TheContinuousSecondOrderConvergesToTheDiscrete3D) {
  // Under lateral viscosity, along u and w in the thickness, ln viscosity
  // and the moduli (lateral changes of each), at the Gauss points: the
  // continuous derivatives along u paired with w tend to the discrete ones
  // block by block as the steps are halved, and so do the continuous
  // Hessian's asymmetry and its gap to the second-order run (against the
  // blocks' size: they nearly cancel in the sum, to 1.3% of it). The
  // continuous second-order run is the discrete one. Measured over 9, 17
  // and 33 nodes: the ratios 3.92 / 3.95 (ice), 3.99 / 3.99 (viscosity),
  // 4.01 / 4.00 (moduli), 3.87 / 3.93 (the Hessian), 3.94 / 3.96 (its
  // asymmetry) and 3.84 / 3.91 (its gap to the run); gated at 9 and 17.
  auto scene = Scene3D::Make(1e22);
  std::mt19937 rng(29);
  const auto fieldT = scene.Smooth(rng, 100.0 * scene.Metre());
  const auto fieldU = scene.Smooth(rng, 20.0 * scene.Metre());
  const auto fieldW = scene.Smooth(rng, 20.0 * scene.Metre());
  const auto g = scene.solver.Grid().FromFunction(
      [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
  const auto kappaMu = scene.material.Moduli().KappaMu();
  auto gaps = std::map<std::string, std::vector<double>>{};
  for (const auto nodes : {9, 17}) {
    const auto times = Grid3D(scene, nodes);
    const auto span = times.back();
    const auto run = RunOn(
        scene, times,
        InTime(times, span, fieldT, [](double s) { return 1.0 - s; }), 0.0, PU);
    const auto& op = *run.op;
    const auto& coupling = op.Coupling();
    auto direction = [&](const std::vector<double>& field,
                         const std::function<double(double)>& profile,
                         const std::function<double(double, double)>& p,
                         const std::function<double(double, double)>& angular) {
      Direction3D d;
      d.tangent.iceThickness = InTime(times, span, field, profile);
      d.tangent.itau = TangentLinearSolver3D::ItauOfLogViscosity(
          op, OnCoupling(coupling, p));
      d.tangent.material.kappa = LateralModulus(
          coupling, scene.material, kappaMu.first,
          [](double r) { return 0.05 * (1.0 - 0.5 * r); }, angular);
      d.tangent.material.mu = LateralModulus(
          coupling, scene.material, kappaMu.second,
          [](double r) { return 0.1 * (0.5 + r * r); }, angular);
      d.p = p;
      return d;
    };
    const auto u = direction(
        fieldU, [](double s) { return s; }, PU,
        [](double theta, double phi) {
          return 1.0 + 0.7 * std::cos(theta) +
                 0.5 * std::sin(theta) * std::sin(phi);
        });
    const auto w = direction(
        fieldW, [](double s) { return (1.0 - s) * (1.0 - s); }, PW,
        [](double theta, double phi) {
          return Cap(theta, phi, 1.9, 4.4, 0.7) -
                 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6) +
                 0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        });
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[times.size() - 1].relativeSeaLevelDual = g;
    SecondOrderDirection3D uw;
    uw.itauU = u.tangent.itau;
    uw.itauV = w.tangent.itau;
    uw.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(
        op, OnCoupling(coupling, PU), OnCoupling(coupling, PW));
    uw.materialU = u.tangent.material;
    uw.materialV = w.tangent.material;
    struct Blocks {
      double ice, viscosity, material, other, forward, run;
    };
    auto results = std::map<AdjointMethod, Blocks>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver3D tangent(*run.evolution, {.method = method,
                                                     .rtol = 1e-12,
                                                     .coupled_rtol = 1e-12,
                                                     .record = true});
      const auto runU = tangent.Run(times, u.tangent, &run.trajectory);
      const auto runW = tangent.Run(times, w.tangent, &run.trajectory);
      AdjointSolver3D adjoint(*run.evolution, {.method = method,
                                               .rtol = 1e-12,
                                               .coupled_rtol = 1e-12,
                                               .material_kernels = true});
      const auto [kernels, alongU] =
          adjoint.SecondOrderKernels(times, measurements, runU, u.tangent.itau,
                                     run.trajectory, {}, u.tangent.material);
      const auto alongW =
          adjoint
              .SecondOrderKernels(times, measurements, runW, w.tangent.itau,
                                  run.trajectory, {}, w.tangent.material)
              .second;
      const auto second =
          tangent.RunSecondOrder(times, runU, runW, uw, &run.trajectory);
      auto ice = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        ice +=
            scene.Surface(alongU.ice.thickness[n], w.tangent.iceThickness[n]);
      }
      const auto viscosity =
          GridPairing(coupling, alongU.itauKernel3D, w.tangent.itau);
      const auto material =
          VolumePairing(coupling, alongU.kappaKernel3D,
                        w.tangent.material.kappa) +
          VolumePairing(coupling, alongU.muKernel3D, w.tangent.material.mu);
      const auto uvTerm =
          GridPairing(coupling, kernels.itauKernel3D, uw.itauUV);
      const auto value = scene.Surface(g, second.back().relativeSeaLevelChange);
      results[method] = {ice,
                         viscosity,
                         material,
                         Pairing3D(scene, coupling, alongW, u),
                         value - uvTerm,
                         value};
    }
    const auto& d = results[AdjointMethod::Discrete];
    const auto& c = results[AdjointMethod::Continuous];
    const auto hd = d.ice + d.viscosity + d.material;
    const auto hc = c.ice + c.viscosity + c.material;
    const auto size =
        std::abs(d.ice) + std::abs(d.viscosity) + std::abs(d.material);
    EXPECT_NEAR(hd, d.other, 1e-8 * size) << "nodes " << nodes;
    EXPECT_NEAR(hd, d.forward, 1e-8 * size) << "nodes " << nodes;
    EXPECT_NEAR(c.run, d.run, 1e-9 * std::abs(d.run)) << "nodes " << nodes;
    gaps["ice"].push_back(std::abs(d.ice - c.ice) / std::abs(d.ice));
    gaps["viscosity"].push_back(std::abs(d.viscosity - c.viscosity) /
                                std::abs(d.viscosity));
    gaps["material"].push_back(std::abs(d.material - c.material) /
                               std::abs(d.material));
    gaps["hessian"].push_back(std::abs(hd - hc) / size);
    gaps["asymmetry"].push_back(std::abs(hc - c.other) / size);
    gaps["run"].push_back(std::abs(hc - c.forward) / size);
  }
  for (const auto& [name, g] : gaps) {
    auto line = std::format("[3D, continuous against discrete] {:10}", name);
    for (auto i = std::size_t{0}; i < g.size(); ++i) {
      line += std::format(" {:.3e}", g[i]);
      if (i > 0) line += std::format(" ({:.2f})", g[i - 1] / g[i]);
    }
    std::cout << line << "\n";
    EXPECT_LT(g.front(), 5e-2) << name;
    EXPECT_GT(g[0] / g[1], 3.5) << name;
    EXPECT_LT(g[0] / g[1], 4.5) << name;
  }
}
