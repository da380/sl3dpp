// The trajectory cursor (detail/TrajectoryCursor.hpp), which serves a
// recorded forward run's states to every 1D derivative run: a recording
// with gaps has them rebuilt by the run's own steps forward from the kept
// state below, so the derivative runs about it are the full recording's
// to rounding at any stride. Gated where a step at -dt fails: on a layer
// of 1e19 Pa s between 220 and 400 km depth, 254 of its relaxation times a
// 1 kyr step, over which a backward trapezoid step amplifies rounding by
// exp(dt / tau) and overflows within a gap. The
// first-order sweeps of both methods (every kernel they form), the
// second-order sweeps and the tangent-linear runs they read, about a
// frozen run under loads and a tide jumping at a repeated node, and about
// a migrating one whose ice steps at one; strides 2, 6 and 5 (an uneven
// last gap); what is refused.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <format>
#include <iostream>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "KernelScene.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/TangentLinear.hpp"

using namespace sl3dpp;
using namespace kernel_scene;

namespace {

using Complex = std::complex<double>;

/// The strides gated against the full recording on the 13-node grid:
/// gaps of one, of five, and of four with a last gap of one (kept nodes
/// 0, 5, 10, 12).
constexpr auto kStrides = std::array{2, 6, 5};

/// The departure of one array from another, against the other's largest
/// entry.
double Departure(std::span<const double> got, std::span<const double> want) {
  const auto top = MaxAbs(want);
  return top > 0.0 ? MaxDiff(got, want) / top : MaxAbs(got);
}

/// Likewise over per-node arrays, against the largest entry of all.
double Departure(const std::vector<std::vector<double>>& got,
                 const std::vector<std::vector<double>>& want) {
  auto top = 0.0, worst = 0.0;
  for (auto n = std::size_t{0}; n < want.size(); ++n) {
    top = std::max(top, MaxAbs(want[n]));
    worst = std::max(worst, MaxDiff(got[n], want[n]));
  }
  return top > 0.0 ? worst / top : worst;
}

double Departure(const std::vector<std::vector<Complex>>& got,
                 const std::vector<std::vector<Complex>>& want) {
  auto top = 0.0, worst = 0.0;
  for (auto n = std::size_t{0}; n < want.size(); ++n) {
    for (auto i = std::size_t{0}; i < want[n].size(); ++i) {
      top = std::max(top, std::abs(want[n][i]));
      worst = std::max(worst, std::abs(got[n][i] - want[n][i]));
    }
  }
  return top > 0.0 ? worst / top : worst;
}

double Departure(const std::vector<std::vector<SphericalTensor>>& got,
                 const std::vector<std::vector<SphericalTensor>>& want) {
  auto flat = [](const std::vector<std::vector<SphericalTensor>>& k) {
    auto out = std::vector<double>{};
    for (const auto& node : k) {
      for (const auto& e : node) {
        out.insert(out.end(), {e.rr, e.tt, e.pp, e.rt, e.rp, e.tp});
      }
    }
    return out;
  };
  return Departure(flat(got), flat(want));
}

/// The weak layer's scene, its operator, and the shortest relaxation time
/// in kyr.
struct WeakLayer {
  Scene setup = Scene::Make(1e21, 1e19);
  ViscoelasticOperator op{setup.rheology, kLmax};

  [[nodiscard]] double TauKyr() const {
    return op.ShortestRelaxation().time * setup.material.ModelScales().Time() /
           (1e3 * kYear);
  }
};

/// 11 steps of 1 kyr with node 4 repeated (nodes 4 and 5 at 4 kyr), the
/// loads and a degree-2 and -3 tide jumping there (their left limits,
/// Forcing::directLoadBefore and tideBefore, half the load and the tide
/// without its step): 13 nodes.
struct Forced {
  std::vector<double> times, distinct;
  std::vector<std::vector<double>> loads;
  std::vector<double> before;
  Evolution::Forcing forcing;

  Forced(const Scene& setup, std::mt19937& rng)
      : times(OnsetGrid(11, 11.0 * setup.Kyr(), 4)) {
    for (auto n = 0; n <= 11; ++n) distinct.push_back(n * setup.Kyr());
    for (auto n = std::size_t{0}; n < distinct.size(); ++n) {
      loads.push_back(setup.Smooth(rng, 300.0 * setup.Sigma()));
    }
    // the load before the jump half the load at it
    before = loads[4];
    for (auto& v : before) v *= 0.5;
    const auto a =
        30.0 / setup.solver.SolverScales().Factor(PotentialDimensions);
    const auto span = times.back();
    const auto jump = times[4];
    forcing.directLoad = History(distinct, loads);
    forcing.directLoadBefore = [plain = forcing.directLoad, jump,
                                this](double t) {
      return t == jump ? before : plain(t);
    };
    const auto tide = [&grid = setup.solver.Grid(), a, span, jump](double t,
                                                                 bool left) {
      auto psi = grid.ZeroCoefficients();
      const auto s = t / span;
      const auto stepped = t > jump || (t == jump && !left) ? 0.5 : 0.0;
      psi[SurfaceGrid::Index(2, 0)] = a * (0.6 + s + stepped);
      psi[SurfaceGrid::Index(2, 1)] = a * Complex(0.4, -0.3 * s);
      psi[SurfaceGrid::Index(3, 2)] = a * Complex(-0.2 * s, 0.5 - stepped);
      return psi;
    };
    forcing.tide = [tide](double t) { return tide(t, false); };
    forcing.tideBefore = [tide](double t) { return tide(t, true); };
  }
  // the forcing reads `before` through `this`
  Forced(const Forced&) = delete;
  Forced& operator=(const Forced&) = delete;
};

/// Measurements at the last node (every surface dual and the angular
/// velocity), at the repeated pair's right node, and at node 9, inside a
/// gap at every stride gated.
std::map<std::size_t, Measurement> Measurements(const Scene& setup,
                                                std::mt19937& rng) {
  auto out = std::map<std::size_t, Measurement>{};
  out[12].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  out[12].displacementDual = setup.Smooth(rng, 0.5);
  out[12].potentialDual = setup.Smooth(rng, 0.4);
  out[12].angularVelocityDual = {0.2, -0.4, 0.8};
  out[5].relativeSeaLevelDual = setup.Smooth(rng, 0.7);
  out[9].displacementDual = setup.Smooth(rng, -0.6);
  return out;
}

/// A material direction at the memory points: the moduli shifted by a
/// smooth relative profile, more in the weak layer's depths.
PointMaterialDirection MaterialAlong(const Scene& setup,
                                     const ViscoelasticOperator& op) {
  const auto& material = setup.material;
  const auto& mesh = material.Mesh();
  const auto kappaMu = material.Moduli().KappaMu();
  MaterialDirection d;
  d.kappa.assign(kappaMu.first.size(), 0.0);
  d.mu.assign(kappaMu.second.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e)) continue;
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      const auto p = 0.05 * (1.0 + mesh.Radius(e, q));
      d.kappa[i] = 0.5 * p * kappaMu.first[i];
      d.mu[i] = p * kappaMu.second[i];
    }
  }
  return AtMemoryPoints(op.Points(), d);
}

/// The kept nodes of a recording.
std::string KeptOf(const ForwardTrajectory& trajectory) {
  auto out = std::string{};
  for (auto n = std::size_t{0}; n < trajectory.states.size(); ++n) {
    if (!trajectory.states[n]) continue;
    out += (out.empty() ? "" : " ") + std::to_string(n);
  }
  return out;
}

const char* NameOf(AdjointMethod method) {
  return method == AdjointMethod::Discrete ? "discrete" : "continuous";
}

}  // namespace

TEST(TrajectoryCursor, FirstOrderKernelsAreTheFullRecordingsAtAnyStride) {
  // Both sweeps about the frozen run under loads and a tide, every kernel
  // they form (the load kernels in and outside the water budget, the
  // layer's, the thickness, the tide, the moments at a point in the weak
  // layer, the relaxation times, the moduli), at each stride against the
  // full recording. The load, tide and moment kernels do not read the
  // trajectory and come out the same. Those that do differ only through
  // the kept states' strain, regenerated by an elastic solve where the
  // run formed it by its step, a rounding the layer's relaxed d - m
  // amplifies (measured: the relaxation times and the moduli within
  // 1.9e-12 discrete, 4.3e-11 continuous, the jump's two sides each
  // regenerated under its own loads); with the strain kept, not at all.
  const WeakLayer weak;
  const auto& setup = weak.setup;
  std::mt19937 rng(71);
  const Forced forced(setup, rng);
  const auto& times = forced.times;
  const auto measurements = Measurements(setup, rng);
  std::cout << std::format(
      "[weak layer] shortest relaxation time {:.2e} kyr: {:.0f} of them a "
      "step\n",
      weak.TauKyr(), 1.0 / weak.TauKyr());
  ASSERT_GT(1.0 / weak.TauKyr(), 100.0);
  const auto point = BodyPoint{
      .radius = 6.06e6 / setup.solver.SolverScales().Length(),
      .colatitude = 0.9,
      .longitude = 1.3};

  auto run = [&](int stride, bool strain = false) {
    Evolution evolution(
        weak.op, setup.solver, setup.state,
        {.rtol = 1e-12, .record_stride = stride, .record_strain = strain});
    return ForwardTrajectory::FromSnapshots(evolution.Run(times, forced.forcing));
  };
  const auto full = run(1);
  const auto fullKept = run(1, true);
  Evolution evolution(weak.op, setup.solver, setup.state, {.rtol = 1e-12});
  for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const AdjointSolver adjoint(evolution, {.method = method,
                                            .rtol = 1e-12,
                                            .material_kernels = true,
                                            .moment_points = {point},
                                            .tide_kernels = true});
    const auto want = adjoint.FirstOrderKernels(times, measurements, &full);
    for (const auto stride : kStrides) {
      const auto sparse = run(stride);
      const auto got = adjoint.FirstOrderKernels(times, measurements, &sparse);
      const auto label = std::format("{}, stride {} (kept {})", NameOf(method),
                                     stride, KeptOf(sparse));
      const auto itau = Departure(got.itauKernel[0], want.itauKernel[0]);
      const auto kappa = Departure(got.kappaKernel, want.kappaKernel);
      const auto mu = Departure(got.muKernel, want.muKernel);
      std::cout << std::format("[{}] itau {:.1e}, kappa {:.1e}, mu {:.1e}\n",
                               label, itau, kappa, mu);
      const auto gate = method == AdjointMethod::Discrete ? 1e-10 : 1e-9;
      EXPECT_LT(itau, gate) << label;
      EXPECT_LT(kappa, gate) << label;
      EXPECT_LT(mu, gate) << label;
      // the kernels that do not read the trajectory
      for (const auto other :
           {Departure(got.directLoadKernels, want.directLoadKernels),
            Departure(got.directLoadOutsideWaterBudgetKernels,
                      want.directLoadOutsideWaterBudgetKernels),
            Departure(got.solidSurfaceChangeKernels,
                      want.solidSurfaceChangeKernels),
            Departure(got.thickness, want.thickness),
            Departure(got.tideKernels, want.tideKernels),
            Departure(got.momentKernels, want.momentKernels)}) {
        EXPECT_LT(other, 1e-14) << label;
      }
    }
    // With the strain kept on the recorded states, a kept state is the
    // run's to the bit and every state rebuilt from it is the run's own,
    // formed by the same steps on the same coefficients: the kernels are
    // the full recording's to the bit.
    const auto wantKept = adjoint.FirstOrderKernels(times, measurements, &fullKept);
    for (const auto stride : kStrides) {
      const auto sparse = run(stride, true);
      const auto got = adjoint.FirstOrderKernels(times, measurements, &sparse);
      EXPECT_EQ(got.itauKernel, wantKept.itauKernel) << stride;
      EXPECT_EQ(got.kappaKernel, wantKept.kappaKernel) << stride;
      EXPECT_EQ(got.muKernel, wantKept.muKernel) << stride;
    }
  }
}

TEST(TrajectoryCursor, SecondOrderSweepsAreTheFullRecordingsAtAnyStride) {
  // Along a direction in the thickness, the relaxation times and the
  // moduli: the tangent-linear run about each recording (the cursor
  // served forwards) and the second-order sweep about it (backwards), by
  // each method, the kernel derivatives at each stride against the full
  // recording's, and the second-order tangent run's objective along the
  // direction twice (measured: within 1.5e-11 discrete, 9.2e-11
  // continuous; the relaxation-time kernel's derivative the farthest).
  const WeakLayer weak;
  const auto& setup = weak.setup;
  std::mt19937 rng(73);
  const Forced forced(setup, rng);
  const auto& times = forced.times;
  const auto measurements = Measurements(setup, rng);
  TangentPerturbation direction;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    direction.iceThickness.push_back(setup.Smooth(rng, 20.0 * setup.Metre()));
  }
  direction.itau = setup.DeltaItau(0.1);
  direction.material = MaterialAlong(setup, weak.op);

  auto run = [&](int stride) {
    Evolution evolution(weak.op, setup.solver, setup.state,
                        {.rtol = 1e-12, .record_stride = stride});
    return ForwardTrajectory::FromSnapshots(evolution.Run(times, forced.forcing));
  };
  const auto full = run(1);
  Evolution evolution(weak.op, setup.solver, setup.state, {.rtol = 1e-12});
  for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const TangentLinearSolver tangent(
        evolution, {.method = method, .rtol = 1e-12, .record = true});
    const AdjointSolver adjoint(
        evolution, {.method = method, .rtol = 1e-12, .material_kernels = true});
    auto second = [&](const ForwardTrajectory& trajectory) {
      const auto along = tangent.Run(times, direction, &trajectory);
      auto twice = SecondOrderDirection{};
      twice.itauU = direction.itau;
      twice.itauV = direction.itau;
      twice.materialU = direction.material;
      twice.materialV = direction.material;
      const auto objective = Objective(
          setup, tangent.RunSecondOrder(times, along, along, twice, &trajectory),
          measurements);
      return std::pair(adjoint
                           .SecondOrderKernels(times, measurements, along,
                                               direction.itau, trajectory, {},
                                               direction.material)
                           .second,
                       objective);
    };
    const auto [want, wantObjective] = second(full);
    ASSERT_GT(MaxAbs(want.itauKernel[0]), 0.0);
    ASSERT_GT(MaxAbs(want.muKernel), 0.0);
    for (const auto stride : kStrides) {
      const auto [got, objective] = second(run(stride));
      const auto label = std::format("{}, stride {}", NameOf(method), stride);
      const auto itau = Departure(got.itauKernel[0], want.itauKernel[0]);
      const auto kappa = Departure(got.kappaKernel, want.kappaKernel);
      const auto mu = Departure(got.muKernel, want.muKernel);
      const auto loads = Departure(got.directLoadKernels, want.directLoadKernels);
      const auto thickness = Departure(got.thickness, want.thickness);
      const auto twice = std::abs(objective - wantObjective) / std::abs(wantObjective);
      std::cout << std::format(
          "[{}] d itau {:.1e}, d kappa {:.1e}, d mu {:.1e}, d loads {:.1e}, d "
          "thickness {:.1e}; F'' {:.1e}\n",
          label, itau, kappa, mu, loads, thickness, twice);
      const auto gate = method == AdjointMethod::Discrete ? 1e-10 : 1e-9;
      for (const auto value : {itau, kappa, mu, loads, thickness, twice}) {
        EXPECT_LT(value, gate) << label;
      }
    }
  }
}

TEST(TrajectoryCursor, AMigratingRunsKernelsAtAnyStride) {
  // About a migrating run (the marine cap melting over the weak layer, its
  // shorelines moving, a tenth of the cap gone at a stroke at a repeated
  // node, Forcing::iceThicknessBefore), whose radial steps read only the
  // recorded coefficients, the ocean functions entering the sea level
  // solves the sweeps take from the snapshots: the relaxation-time and
  // moduli kernels of both sweeps at each stride are the full recording's
  // (measured: within 4.1e-12 discrete, 1.1e-10 continuous).
  const WeakLayer weak;
  const auto& setup = weak.setup;
  const auto cap = MarineCap::Make(setup, 13);
  const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
  // the cap's grid with node 6 repeated, its step there
  auto times = cap.times;
  times.insert(times.begin() + 7, times[6]);
  const auto step = times[6];
  const auto melted = [&cap, step](double t, bool after) {
    auto out = cap.Thickness(t);
    if (!after) return out;
    for (auto i = std::size_t{0}; i < out.size(); ++i) {
      out[i] = std::max(0.0, out[i] - 0.1 * cap.ice0[i]);
    }
    return out;
  };
  const auto forcing = Evolution::Forcing{
      .iceThicknessBefore = [&melted, step](double t) {
        return melted(t, t > step);
      }};
  auto measurements = std::map<std::size_t, Measurement>{};
  std::mt19937 rng(79);
  measurements[13].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[7].displacementDual = setup.Smooth(rng, 0.5);
  auto run = [&](int stride) {
    Evolution evolution(weak.op, setup.solver, state,
                        {.rtol = 1e-12, .record_stride = stride});
    return evolution.RunMigrating(
        times, [&melted, step](double t) { return melted(t, t >= step); },
        forcing);
  };
  const auto fullRun = run(1);
  const auto oceans = OceanFunctionsOf(fullRun);
  auto moved = false;
  for (const auto& c : oceans) moved = moved || c != oceans.front();
  ASSERT_TRUE(moved);
  const auto full = ForwardTrajectory::FromSnapshots(fullRun);
  Evolution evolution(weak.op, setup.solver, state, {.rtol = 1e-12});
  for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const AdjointSolver adjoint(
        evolution, {.method = method, .rtol = 1e-12, .material_kernels = true});
    const auto want = adjoint.FirstOrderKernels(times, measurements, &full, oceans);
    for (const auto stride : kStrides) {
      const auto sparseRun = run(stride);
      EXPECT_EQ(OceanFunctionsOf(sparseRun), oceans);
      const auto sparse = ForwardTrajectory::FromSnapshots(sparseRun);
      const auto got = adjoint.FirstOrderKernels(times, measurements, &sparse,
                                                 oceans);
      const auto label = std::format("{}, stride {}", NameOf(method), stride);
      const auto itau = Departure(got.itauKernel[0], want.itauKernel[0]);
      const auto mu = Departure(got.muKernel, want.muKernel);
      std::cout << std::format("[migrating, {}] itau {:.1e}, mu {:.1e}\n", label,
                               itau, mu);
      const auto gate = method == AdjointMethod::Discrete ? 1e-10 : 1e-9;
      EXPECT_LT(itau, gate) << label;
      EXPECT_LT(mu, gate) << label;
    }
  }
}

TEST(TrajectoryCursor, WhatAGappedRecordingIsRefused) {
  // A recording whose first node's state is not kept (the rebuild starts
  // there) and one without the recorded coefficients are refused by the
  // sweeps and the tangent-linear run; a run under interior sources is
  // recorded at every node, a stride refused (its states are not rebuilt
  // from the loads alone); a gapped recording under Sdirk2 is refused by
  // the continuous sweep (TestAdjoint, Sdirk2's refusals).
  const WeakLayer weak;
  const auto& setup = weak.setup;
  std::mt19937 rng(83);
  const Forced forced(setup, rng);
  const auto& times = forced.times;
  const auto measurements = Measurements(setup, rng);
  Evolution evolution(weak.op, setup.solver, setup.state,
                      {.rtol = 1e-12, .record_stride = 3});
  const auto sparse =
      ForwardTrajectory::FromSnapshots(evolution.Run(times, forced.forcing));
  auto headless = sparse;
  headless.states.front().reset();
  auto bare = sparse;
  bare.loads.clear();
  bare.tides.clear();
  for (const auto* bad : {&headless, &bare}) {
    for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const AdjointSolver adjoint(evolution, {.method = method});
      EXPECT_THROW((void)adjoint.FirstOrderKernels(times, measurements, bad),
                   std::invalid_argument);
      const TangentLinearSolver tangent(evolution, {.method = method});
      EXPECT_THROW((void)tangent.Run(
                       times, TangentPerturbation{.itau = setup.DeltaItau(0.1)}, bad),
                   std::invalid_argument);
    }
  }
  const auto source = PointSource{
      .at = {.radius = 6.06e6 / setup.solver.SolverScales().Length(),
             .colatitude = 0.9,
             .longitude = 1.3},
      .moment = {.rr = 1e-9, .tt = -1e-9}};
  auto sourced = forced.forcing;
  sourced.sources = [source](double) { return std::vector{source}; };
  Evolution recorded(weak.op, setup.solver, setup.state, {.record_stride = 2});
  EXPECT_THROW((void)recorded.Run(times, sourced), std::invalid_argument);
}
