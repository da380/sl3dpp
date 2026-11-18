// The derivatives of a calibrated run (InitialSeaLevel.hpp): a run whose
// initial sea level is found so that it reaches a target at its last node,
// and is found again when a parameter changes. Under migrating shorelines
// the total kernels (ComputeCalibratedKernels) predict the differences of
// the objective between fully re-calibrated runs that flip no cell, in the
// ice thickness, the relaxation times, the moduli and the target, where
// the partial kernels (the initial sea level held) miss them; the
// multiplier is the fixed point of its relation; the continuous sweep's
// converges to the discrete's with the step; under frozen shorelines no
// sweep runs for the multiplier and the kernels are the partial ones; an
// iteration that runs out says so, and the second-order sweeps about a
// calibrated migrating state are refused by name. The linearised calibration
// (ComputeCalibratedTangent) is the same derivative in the forward mode: the
// change of the calibrated initial sea level against calibrations afresh,
// and its run's pairing with an objective's duals the total kernels' with
// the direction plus the multiplier's with the target's change. 1D and,
// under lateral viscosity, 3D.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <map>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "KernelScene.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using namespace kernel_scene;

namespace {

using Fields = std::vector<std::vector<double>>;

/// The node of a grid at time t.
std::size_t NodeOf(std::span<const double> times, double t) {
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) return n;
  }
  throw std::runtime_error("the test history is read off its nodes");
}

/// A per-node thickness history as the forward driver's callable, moved by
/// eps times a per-node direction (empty: none).
std::function<std::vector<double>(double)> Field(std::vector<double> times,
                                                 Fields base,
                                                 Fields direction = {},
                                                 double eps = 0.0) {
  return [times = std::move(times), base = std::move(base),
          direction = std::move(direction), eps](double t) {
    const auto n = NodeOf(times, t);
    auto out = base[n];
    if (!direction.empty()) {
      for (auto i = std::size_t{0}; i < out.size(); ++i) {
        out[i] += eps * direction[n][i];
      }
    }
    return out;
  };
}

std::vector<double> Moved(std::vector<double> x, std::span<const double> d,
                          double eps) {
  for (auto i = std::size_t{0}; i < x.size(); ++i) x[i] += eps * d[i];
  return x;
}

double Gap(double got, double want) {
  return std::abs(got - want) / std::abs(want);
}

/// Central differences at eps and eps / 2: the Richardson value.
double Richardson(const std::function<double(double)>& f, double eps) {
  const auto wide = (f(eps) - f(-eps)) / (2.0 * eps);
  const auto narrow = (f(0.5 * eps) - f(-0.5 * eps)) / eps;
  return (4.0 * narrow - wide) / 3.0;
}

/// The contraction of an iteration from its updates: the largest ratio of
/// two that follow each other, past the first (which starts from zero).
double Contraction(std::span<const double> updates) {
  auto out = 0.0;
  for (auto k = std::size_t{2}; k < updates.size(); ++k) {
    if (updates[k - 1] > 1e-11) out = std::max(out, updates[k] / updates[k - 1]);
  }
  return out;
}

std::string Listed(std::span<const double> updates) {
  auto out = std::string{};
  for (const auto u : updates) out += std::format(" {:.1e}", u);
  return out;
}

/// The operator with its inverse relaxation times moved by eps along ditau.
ViscoelasticOperator MovedItau(const Scene& setup, double eps,
                               const Fields& ditau) {
  auto branches = std::vector<Rheology::Branch>{};
  for (auto k = std::size_t{0}; k < setup.rheology.BranchCount(); ++k) {
    auto branch = setup.rheology.BranchAt(k);
    for (auto i = std::size_t{0}; i < branch.itau.size(); ++i) {
      branch.itau[i] += eps * ditau[k][i];
    }
    branches.push_back(std::move(branch));
  }
  return ViscoelasticOperator(
      Rheology::Create(setup.material, branches).value(), kLmax);
}

/// A direction in kappa and mu at the operator's memory points on the
/// solid, at a tenth of the moduli (kappa at half the shear's share).
PointMaterialDirection PointDirection(const ViscoelasticOperator& op,
                                      const Material& material) {
  const auto& mesh = material.Mesh();
  const auto& points = op.Points();
  const auto muU = op.UnrelaxedShearAtPoints();
  const auto kappa0 = op.BulkModulusAtPoints();
  PointMaterialDirection d;
  d.kappa.assign(muU.size(), 0.0);
  d.mu.assign(muU.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e)) continue;
    for (auto p = 0; p < points.count; ++p) {
      const auto i = static_cast<std::size_t>(e) * points.count + p;
      const auto r = points.RadiusAt(mesh, e, p);
      const auto value = 0.1 * (0.4 + 0.6 * std::pow(std::sin(3.0 * r), 2));
      d.kappa[i] = 0.5 * value * kappa0[i];
      d.mu[i] = value * muU[i];
    }
  }
  return d;
}

/// The operator with its moduli at the points moved by eps along d.
ViscoelasticOperator MovedModuli(const Scene& setup,
                                 const ViscoelasticOperator& op, double eps,
                                 const PointMaterialDirection& d) {
  const auto muU = op.UnrelaxedShearAtPoints();
  const auto kappa0 = op.BulkModulusAtPoints();
  ViscoelasticOperator::Options::PointModuli moduli;
  moduli.shear.assign(muU.size(), 0.0);
  moduli.bulk.assign(muU.size(), 0.0);
  for (auto i = std::size_t{0}; i < muU.size(); ++i) {
    if (muU[i] > 0.0) moduli.shear[i] = eps * d.mu[i] / muU[i];
    if (kappa0[i] > 0.0) moduli.bulk[i] = eps * d.kappa[i] / kappa0[i];
  }
  return ViscoelasticOperator(setup.rheology, kLmax, {.moduli = moduli});
}

/// The marine cap's history on a known, non-uniform initial sea level, and
/// the final sea level its migrating run reaches: the target of the
/// calibrations here. The cap melts to nothing over the window, flooding
/// its cells.
struct Marine1D {
  Scene setup;
  ViscoelasticOperator op;
  MarineCap cap;
  Fields ice;                  ///< the thickness at the nodes
  std::vector<double> target;  ///< the final sea level
  Evolution::Options options;

  Marine1D(double eta, int nodes, Scheme scheme)
      : setup(Scene::Make(eta)),
        op(setup.rheology, kLmax),
        cap(MarineCap::Make(setup, nodes)),
        options{.scheme = scheme,
                .rtol = 1e-12,
                .max_nonlinear_iterations = 200,
                .record_stride = 1} {
    for (const auto t : cap.times) ice.push_back(cap.Thickness(t));
    const auto bump =
        setup.solver.Grid().FromFunction([&](double theta, double phi) {
          return 200.0 * setup.Metre() * std::sin(theta) * std::cos(phi);
        });
    const auto truth =
        setup.solver.MakeState(cap.ice0, Moved(cap.sea0, bump, 1.0));
    Evolution evolution(op, setup.solver, truth, options);
    const auto run = evolution.RunMigrating(cap.times, Field(cap.times, ice));
    EXPECT_NE(run.back().oceanFunction, truth.oceanFunction);
    target = Moved(truth.seaLevel, run.back().relativeSeaLevelChange, 1.0);
  }

  /// The calibration of `evolution` (over any operator) under the ice
  /// moved by eps along a direction and towards a target, tightly: the
  /// state adopted, the run returned.
  [[nodiscard]] InitialSeaLevelCalibration Calibrate(
      Evolution& evolution, const Fields& direction, double eps,
      std::span<const double> towards, Shorelines shorelines,
      const std::vector<double>& weights = {},
      std::span<const double> guess = {}) const {
    const auto& times = cap.times;
    const auto first =
        direction.empty() ? ice[0] : Moved(ice[0], direction[0], eps);
    if (guess.empty()) guess = towards;
    evolution.SetBackgroundState(setup.solver.MakeState(
        first, std::vector<double>(guess.begin(), guess.end())));
    auto history = IceHistory::Migrating(Field(times, ice, direction, eps));
    if (shorelines == Shorelines::Frozen) {
      // the load of the history from its start, on the guess's shorelines
      const auto ocean = evolution.BackgroundState().oceanFunction;
      const auto rhoI = setup.solver.IceDensity();
      history = IceHistory::Frozen(
          [thickness = history.field, ocean, rhoI, t0 = times.front()](
              double t) {
            auto load = thickness(t);
            const auto start = thickness(t0);
            for (auto i = std::size_t{0}; i < load.size(); ++i) {
              load[i] = rhoI * (1.0 - ocean[i]) * (load[i] - start[i]);
            }
            return load;
          });
    }
    auto calibration = CalibrateInitialSeaLevel(
        evolution, times, history, towards,
        {.weights = weights, .iterations = 80, .rtol = 1e-13});
    EXPECT_TRUE(calibration.converged)
        << "in " << calibration.updates.size() << " runs, the last update "
        << calibration.updates.back();
    return calibration;
  }
};

double NodePairing(const Scene& setup, const Fields& kernel, const Fields& d) {
  auto out = 0.0;
  for (auto n = std::size_t{0}; n < d.size(); ++n) {
    out += setup.Surface(kernel[n], d[n]);
  }
  return out;
}

}  // namespace

// -- 1D ---------------------------------------------------------------------

TEST(CalibratedKernels, TotalKernelsPredictReCalibratedRuns) {
  // A calibrated migrating run of the marine cap, under Sdirk2 and under
  // the exponential trapezoid, the discrete sweeps. Every difference is
  // between runs calibrated afresh (to 1e-13 of the sea level) that flip no
  // cell of the base run, its start's included:
  //  - a thickness direction at every node (the first node's moving the
  //    ice the run starts from): the runs are linear in it while no cell
  //    flips, so the central difference is the derivative;
  //  - the relaxation times and the moduli, by Richardson's value of two
  //    central differences;
  //  - the target, as the thickness.
  // The total kernels pair with each to the differences' own accuracy, and
  // the multiplier with the target's; the partial kernels miss each by a
  // visible margin. The final sweep's initial sea level kernel is the
  // multiplier again.
  for (const auto scheme : {Scheme::Sdirk2, Scheme::ExponentialTrapezoid}) {
    const auto label =
        std::string{scheme == Scheme::Sdirk2 ? "sdirk2" : "trapezoid"};
    const auto scene = Marine1D(1e21, 6, scheme);
    const auto& setup = scene.setup;
    const auto& times = scene.cap.times;
    const auto nodes = times.size();
    std::mt19937 rng(211);
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[1].relativeSeaLevelDual = setup.Smooth(rng, 0.4);
    measurements[3].displacementDual = setup.Smooth(rng, 0.7);
    measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
    auto deltaIce = Fields{};
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      deltaIce.push_back(setup.Smooth(rng, 0.1 * setup.Metre()));
    }
    const auto deltaTarget = setup.Smooth(rng, 0.1 * setup.Metre());
    const auto ditau = setup.DeltaItau(0.1);
    const auto material = PointDirection(scene.op, setup.material);
    const auto none = Fields{};

    Evolution evolution(scene.op, setup.solver, setup.state, scene.options);
    const auto base = scene.Calibrate(evolution, none, 0.0, scene.target,
                                      Shorelines::Migrating);
    const auto& run = base.snapshots;
    const auto oceans = OceanFunctionsOf(run);
    const auto stageOceans = StageOceanFunctionsOf(run);
    ASSERT_NE(oceans.back(), oceans.front()) << "no cell flipped";
    const auto trajectory = ForwardTrajectory::FromSnapshots(run);

    // the objective of the run calibrated afresh
    auto objective = [&](const ViscoelasticOperator& at, const Fields& d,
                         double eps, std::span<const double> towards) {
      Evolution trial(at, setup.solver, setup.state, scene.options);
      const auto calibrated =
          scene.Calibrate(trial, d, eps, towards, Shorelines::Migrating);
      // flotation's 0/1: rounding changes it only at a cell on the threshold
      EXPECT_EQ(calibrated.state.oceanFunction, base.state.oceanFunction)
          << "the start flipped";
      for (auto n = std::size_t{0}; n < nodes; ++n) {
        EXPECT_TRUE(calibrated.snapshots[n].converged) << "node " << n;
        EXPECT_EQ(calibrated.snapshots[n].oceanFunction, oceans[n])
            << "a cell flipped at node " << n;
      }
      return Objective(setup, calibrated.snapshots, measurements);
    };
    const auto fdIce = 0.5 * (objective(scene.op, deltaIce, 1.0, scene.target) -
                              objective(scene.op, deltaIce, -1.0, scene.target));
    const auto fdTarget =
        0.5 * (objective(scene.op, none, 0.0,
                         Moved(scene.target, deltaTarget, 1.0)) -
               objective(scene.op, none, 0.0,
                         Moved(scene.target, deltaTarget, -1.0)));
    const auto fdItau = Richardson(
        [&](double eps) {
          return objective(MovedItau(setup, eps, ditau), none, 0.0,
                           scene.target);
        },
        0.004);
    const auto fdModuli = Richardson(
        [&](double eps) {
          return objective(MovedModuli(setup, scene.op, eps, material), none,
                           0.0, scene.target);
        },
        0.004);

    const auto adjoint =
        AdjointSolver::Options{.method = AdjointMethod::Discrete,
                               .rtol = 1e-12,
                               .material_kernels = true};
    const auto partial =
        AdjointSolver(evolution, adjoint)
            .FirstOrderKernels(times, measurements, &trajectory, oceans,
                               stageOceans);
    const auto total = ComputeCalibratedKernels(
        evolution, times, measurements, &trajectory, oceans, stageOceans,
        {.rtol = 1e-12}, adjoint);
    EXPECT_TRUE(total.converged);
    EXPECT_EQ(total.sweeps, static_cast<int>(total.updates.size()) + 1);

    const auto& mesh = setup.material.Mesh();
    const auto pairings = [&](const Kernels& k) {
      return std::array<double, 3>{
          NodePairing(setup, k.thickness, deltaIce),
          VolumePairing(mesh, k.itauKernel[0], ditau[0]),
          VolumePairing(scene.op.Points(), mesh, k.kappaKernel,
                        material.kappa) +
              VolumePairing(scene.op.Points(), mesh, k.muKernel, material.mu)};
    };
    const auto whole = pairings(total.kernels);
    const auto held = pairings(partial);
    const auto byTarget = setup.Surface(total.multiplier, deltaTarget);
    std::cout << std::format(
        "[calibrated, 1D, {}] multiplier in {} sweeps, updates{}, contraction "
        "{:.2e}\n"
        "  total from the differences: ice {:.2e}, itau {:.2e}, moduli "
        "{:.2e}, target {:.2e}\n"
        "  partial from them:          ice {:.2e}, itau {:.2e}, moduli "
        "{:.2e}\n",
        label, total.updates.size(), Listed(total.updates),
        Contraction(total.updates), Gap(whole[0], fdIce),
        Gap(whole[1], fdItau), Gap(whole[2], fdModuli),
        Gap(byTarget, fdTarget), Gap(held[0], fdIce), Gap(held[1], fdItau),
        Gap(held[2], fdModuli));
    // measured: ice and target a few 1e-10 (the calibrations' tolerance
    // against the differences' size), the relaxation times and the moduli
    // 1e-7 to 1e-6 (Richardson's remainder and the solves' noise in steps
    // small enough to flip no cell)
    EXPECT_NEAR(whole[0], fdIce, 1e-7 * std::abs(fdIce)) << label;
    EXPECT_NEAR(byTarget, fdTarget, 1e-7 * std::abs(fdTarget)) << label;
    EXPECT_NEAR(whole[1], fdItau, 1e-5 * std::abs(fdItau)) << label;
    EXPECT_NEAR(whole[2], fdModuli, 1e-5 * std::abs(fdModuli)) << label;
    // the initial sea level's re-calibration is no small part of each
    EXPECT_GT(Gap(held[0], fdIce), 1e-2) << label;
    EXPECT_GT(Gap(held[1], fdItau), 1e-2) << label;
    EXPECT_GT(Gap(held[2], fdModuli), 1e-2) << label;

    // the plain iteration reaches the same multiplier in more sweeps
    const auto plain = SolveCalibrationMultiplier(
        evolution, times, measurements, oceans, stageOceans,
        {.iterations = 60, .rtol = 1e-12, .anderson_memory = 0}, adjoint);
    EXPECT_TRUE(plain.converged);
    EXPECT_GT(plain.sweeps, total.sweeps + 5) << label;
    EXPECT_LT(MaxDiff(plain.multiplier, total.multiplier),
              1e-10 * MaxAbs(total.multiplier))
        << label;
    std::cout << std::format(
        "  the plain iteration: {} sweeps, updates{}, contraction {:.2e}\n",
        plain.sweeps, Listed(plain.updates), Contraction(plain.updates));
    // the fixed point: the final sweep's initial sea level kernel is the
    // multiplier (every point is calibrated here)
    EXPECT_GT(MaxAbs(total.multiplier), 0.0);
    EXPECT_LT(MaxDiff(total.kernels.initialSeaLevel, total.multiplier),
              1e-10 * MaxAbs(total.multiplier))
        << label;
    // and the first sweep's, from zero, the partial one's
    const auto first = SolveCalibrationMultiplier(
        evolution, times, measurements, oceans, stageOceans,
        {.iterations = 1, .rtol = 1e-12}, adjoint);
    EXPECT_FALSE(first.converged);
    EXPECT_EQ(first.sweeps, 1);
    EXPECT_LT(MaxDiff(first.multiplier, partial.initialSeaLevel),
              1e-12 * MaxAbs(partial.initialSeaLevel))
        << label;
  }
}

TEST(CalibratedKernels, FrozenShorelinesNeedNoMultiplier) {
  // Under frozen shorelines the run does not depend on its initial sea
  // level: the multiplier is zero, no sweep runs for it, the kernels are
  // the one sweep's, and they predict the difference of re-calibrated
  // frozen runs in the ice.
  const auto scene = Marine1D(1e21, 6, Scheme::Sdirk2);
  const auto& setup = scene.setup;
  const auto& times = scene.cap.times;
  const auto nodes = times.size();
  std::mt19937 rng(223);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[3].displacementDual = setup.Smooth(rng, 0.7);
  measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  auto deltaIce = Fields{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    deltaIce.push_back(setup.Smooth(rng, 0.1 * setup.Metre()));
  }
  Evolution evolution(scene.op, setup.solver, setup.state, scene.options);
  const auto base =
      scene.Calibrate(evolution, {}, 0.0, scene.target, Shorelines::Frozen);
  auto objective = [&](double eps) {
    Evolution trial(scene.op, setup.solver, setup.state, scene.options);
    const auto calibrated = scene.Calibrate(trial, deltaIce, eps, scene.target,
                                            Shorelines::Frozen);
    // flotation's 0/1: rounding changes it only at a cell on the threshold
    EXPECT_EQ(calibrated.state.oceanFunction, base.state.oceanFunction);
    return Objective(setup, calibrated.snapshots, measurements);
  };
  const auto fd = 0.5 * (objective(1.0) - objective(-1.0));
  const auto adjoint = AdjointSolver::Options{.rtol = 1e-12};
  auto calls = 0;
  const auto total = ComputeCalibratedKernels(
      evolution, times, measurements, nullptr, {}, {},
      {.on_iteration = [&](int, double) { ++calls; }}, adjoint);
  EXPECT_TRUE(total.converged);
  EXPECT_EQ(total.sweeps, 1);
  EXPECT_EQ(calls, 0);
  EXPECT_TRUE(total.updates.empty());
  ASSERT_EQ(total.multiplier.size(), setup.solver.Grid().FieldSize());
  EXPECT_EQ(MaxAbs(total.multiplier), 0.0);
  EXPECT_EQ(MaxAbs(total.kernels.initialSeaLevel), 0.0);
  const auto partial = AdjointSolver(evolution, adjoint)
                           .FirstOrderKernels(times, measurements);
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    EXPECT_LE(MaxDiff(total.kernels.thickness[n], partial.thickness[n]),
              1e-14 * MaxAbs(partial.thickness[n]))
        << "node " << n;
  }
  const auto predicted = NodePairing(setup, total.kernels.thickness, deltaIce);
  std::cout << std::format("[calibrated, 1D, frozen] ice {:.2e}\n",
                           Gap(predicted, fd));
  EXPECT_NEAR(predicted, fd, 1e-8 * std::abs(fd));
}

TEST(CalibratedKernels, WeightsHoldTheUncalibratedPoints) {
  // A calibration weighted on part of the sphere (here the meridian
  // through the cap's centre cuts it) sets the initial sea level there and
  // holds it elsewhere. The multiplier then lives on the
  // weighted points, where the final sweep's initial sea level kernel
  // returns it; elsewhere that kernel is the total derivative in the
  // initial sea level held there, against differences of runs calibrated
  // afresh from a guess moved off the weighted points. The total thickness
  // kernels predict the re-calibrated runs' difference in the ice.
  const auto scene = Marine1D(1e21, 6, Scheme::Sdirk2);
  const auto& setup = scene.setup;
  const auto& grid = setup.solver.Grid();
  const auto& times = scene.cap.times;
  const auto nodes = times.size();
  const auto weights = grid.FromFunction(
      [](double, double phi) { return phi > 4.8 ? 1.0 : 0.0; });
  std::mt19937 rng(227);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[2].relativeSeaLevelDual = setup.Smooth(rng, 0.4);
  measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  auto deltaIce = Fields{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    deltaIce.push_back(setup.Smooth(rng, 0.1 * setup.Metre()));
  }
  // a direction of the guess off the weighted points
  auto deltaGuess = setup.Smooth(rng, 0.1 * setup.Metre());
  for (auto i = std::size_t{0}; i < deltaGuess.size(); ++i) {
    if (weights[i] > 0.0) deltaGuess[i] = 0.0;
  }
  ASSERT_GT(MaxAbs(deltaGuess), 0.0);
  const auto none = Fields{};
  Evolution evolution(scene.op, setup.solver, setup.state, scene.options);
  const auto base = scene.Calibrate(evolution, none, 0.0, scene.target,
                                    Shorelines::Migrating, weights);
  const auto oceans = OceanFunctionsOf(base.snapshots);
  const auto stageOceans = StageOceanFunctionsOf(base.snapshots);
  ASSERT_NE(oceans.back(), oceans.front()) << "no cell flipped";
  auto objective = [&](const Fields& d, double eps,
                       std::span<const double> guess) {
    Evolution trial(scene.op, setup.solver, setup.state, scene.options);
    const auto calibrated =
        scene.Calibrate(trial, d, eps, scene.target, Shorelines::Migrating,
                        weights, guess);
    // flotation's 0/1: rounding changes it only at a cell on the threshold
    EXPECT_EQ(calibrated.state.oceanFunction, base.state.oceanFunction);
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      EXPECT_EQ(calibrated.snapshots[n].oceanFunction, oceans[n])
          << "a cell flipped at node " << n;
    }
    return Objective(setup, calibrated.snapshots, measurements);
  };
  const auto fdIce = 0.5 * (objective(deltaIce, 1.0, scene.target) -
                            objective(deltaIce, -1.0, scene.target));
  const auto fdGuess =
      0.5 * (objective(none, 0.0, Moved(scene.target, deltaGuess, 1.0)) -
             objective(none, 0.0, Moved(scene.target, deltaGuess, -1.0)));
  const auto adjoint = AdjointSolver::Options{.rtol = 1e-12};
  const auto total = ComputeCalibratedKernels(
      evolution, times, measurements, nullptr, oceans, stageOceans,
      {.weights = weights, .rtol = 1e-12}, adjoint);
  EXPECT_TRUE(total.converged);
  auto onWeighted = 0.0, offWeighted = 0.0;
  for (auto i = std::size_t{0}; i < weights.size(); ++i) {
    if (weights[i] > 0.0) {
      onWeighted = std::max(onWeighted,
                            std::abs(total.kernels.initialSeaLevel[i] -
                                     total.multiplier[i]));
    } else {
      offWeighted = std::max(offWeighted, std::abs(total.multiplier[i]));
    }
  }
  EXPECT_GT(MaxAbs(total.multiplier), 0.0);
  EXPECT_LT(onWeighted, 1e-10 * MaxAbs(total.multiplier));
  EXPECT_EQ(offWeighted, 0.0);
  const auto byIce = NodePairing(setup, total.kernels.thickness, deltaIce);
  const auto byGuess = setup.Surface(total.kernels.initialSeaLevel, deltaGuess);
  std::cout << std::format(
      "[calibrated, 1D, weighted] multiplier in {} sweeps; ice {:.2e}, the "
      "initial sea level off the weighted points {:.2e}\n",
      total.updates.size(), Gap(byIce, fdIce), Gap(byGuess, fdGuess));
  ASSERT_NE(fdGuess, 0.0) << "no cell off the weighted points flipped";
  EXPECT_NEAR(byIce, fdIce, 1e-7 * std::abs(fdIce));
  EXPECT_NEAR(byGuess, fdGuess, 1e-7 * std::abs(fdGuess));
}

TEST(CalibratedKernels, ARunOutIterationIsReportedAndSecondOrderRefused) {
  // An iteration cut short returns its last iterate and says so, and the
  // kernels formed with it carry the word; the second-order kernels of a
  // calibrated migrating run are refused by name.
  const auto scene = Marine1D(1e21, 6, Scheme::Sdirk2);
  const auto& setup = scene.setup;
  const auto& times = scene.cap.times;
  std::mt19937 rng(229);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  Evolution evolution(scene.op, setup.solver, setup.state, scene.options);
  const auto base = scene.Calibrate(evolution, {}, 0.0, scene.target,
                                    Shorelines::Migrating);
  const auto oceans = OceanFunctionsOf(base.snapshots);
  const auto stageOceans = StageOceanFunctionsOf(base.snapshots);
  auto seen = std::vector<double>{};
  const auto cut = ComputeCalibratedKernels(
      evolution, times, measurements, nullptr, oceans, stageOceans,
      {.iterations = 2,
       .rtol = 1e-12,
       .on_iteration = [&](int sweep, double update) {
         EXPECT_EQ(sweep, static_cast<int>(seen.size()) + 1);
         seen.push_back(update);
       }});
  EXPECT_FALSE(cut.converged);
  EXPECT_EQ(cut.sweeps, 3);
  EXPECT_EQ(cut.updates, seen);
  ASSERT_EQ(cut.updates.size(), 2u);
  EXPECT_GT(cut.updates.back(), 1e-12);
  EXPECT_THROW(
      static_cast<void>(SolveCalibrationMultiplier(
          evolution, times, measurements, oceans, stageOceans,
          {.iterations = 0})),
      std::invalid_argument);
  // The calibration marks the state it leaves, and a second-order sweep
  // about it, which would hold the initial sea level, is refused by name
  // (before it reads its arguments) unless it states that the derivative
  // of the partial kernels is what is wanted; a state set by hand is not
  // marked.
  EXPECT_TRUE(evolution.Calibrated());
  const auto trajectory = ForwardTrajectory::FromSnapshots(base.snapshots);
  const auto refused = [&](const AdjointSolver& sweep) {
    try {
      static_cast<void>(sweep.SecondOrderKernels(
          times, measurements, {}, {}, trajectory, oceans, {}, stageOceans));
      return std::string{};
    } catch (const std::logic_error& error) {
      return std::string{error.what()};
    }
  };
  const auto what = refused(AdjointSolver(evolution));
  EXPECT_NE(what.find("second-order kernels of a calibrated run under "
                      "migrating shorelines"),
            std::string::npos)
      << what;
  EXPECT_NE(what.find("not built"), std::string::npos) << what;
  EXPECT_NE(what.find("initial_sea_level_held"), std::string::npos) << what;
  TangentPerturbation direction;
  direction.initialSeaLevel = setup.Smooth(rng, 0.1 * setup.Metre());
  const auto tangent =
      TangentLinearSolver(evolution, {.rtol = 1e-12, .record = true})
          .Run(times, direction, &trajectory, oceans, stageOceans);
  const auto [kernels, along] =
      AdjointSolver(evolution, {.rtol = 1e-12, .initial_sea_level_held = true})
          .SecondOrderKernels(times, measurements, tangent, {}, trajectory,
                              oceans, {}, stageOceans);
  EXPECT_GT(MaxAbs(kernels.initialSeaLevel), 0.0);
  EXPECT_EQ(along.initialSeaLevel.size(), kernels.initialSeaLevel.size());
  evolution.SetBackgroundState(base.state);
  EXPECT_FALSE(evolution.Calibrated());
  // under frozen shorelines the sweep is the calibrated run's: not marked
  Evolution frozen(scene.op, setup.solver, setup.state, scene.options);
  static_cast<void>(
      scene.Calibrate(frozen, {}, 0.0, scene.target, Shorelines::Frozen));
  EXPECT_FALSE(frozen.Calibrated());
}

TEST(CalibratedKernels, TheContinuousSweepConvergesToTheDiscrete) {
  // On a slowly relaxing mantle (1e22 Pa s), one problem at 5, 9 and 17
  // nodes over the 10 kyr, the exponential trapezoid: the discrete total kernels
  // predict the re-calibrated runs' difference in the ice at the quarter
  // dates at either step, and the continuous ones, whose multiplier is the
  // fixed point of the continuous sweep's own approximation, come to it as
  // the step halves.
  auto check = [](int nodes) {
    const auto scene = Marine1D(1e22, nodes, Scheme::ExponentialTrapezoid);
    const auto& setup = scene.setup;
    const auto& times = scene.cap.times;
    std::mt19937 rng(233);
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[times.size() - 1].relativeSeaLevelDual =
        setup.Smooth(rng, 1.0);
    // directions at the quarter dates but the first, the same at either
    // step
    const auto per = (nodes - 1) / 4;
    auto deltaIce = Fields(times.size(),
                           std::vector<double>(setup.solver.Grid().FieldSize()));
    for (auto q = 1; q <= 4; ++q) {
      deltaIce[static_cast<std::size_t>(q * per)] =
          setup.Smooth(rng, 0.1 * setup.Metre());
    }
    Evolution evolution(scene.op, setup.solver, setup.state, scene.options);
    const auto base = scene.Calibrate(evolution, {}, 0.0, scene.target,
                                      Shorelines::Migrating);
    const auto oceans = OceanFunctionsOf(base.snapshots);
    EXPECT_NE(oceans.back(), oceans.front()) << "no cell flipped";
    auto objective = [&](double eps) {
      Evolution trial(scene.op, setup.solver, setup.state, scene.options);
      const auto calibrated = scene.Calibrate(
          trial, deltaIce, eps, scene.target, Shorelines::Migrating);
      // flotation's 0/1: rounding changes it only at a cell on the threshold
      EXPECT_EQ(calibrated.state.oceanFunction, base.state.oceanFunction);
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        EXPECT_EQ(calibrated.snapshots[n].oceanFunction, oceans[n])
            << "a cell flipped at node " << n;
      }
      return Objective(setup, calibrated.snapshots, measurements);
    };
    const auto fd = 0.5 * (objective(1.0) - objective(-1.0));
    auto errors = std::map<AdjointMethod, double>{};
    auto multipliers = std::map<AdjointMethod, std::vector<double>>{};
    for (const auto method :
         {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto total = ComputeCalibratedKernels(
          evolution, times, measurements, nullptr, oceans, {}, {.rtol = 1e-11},
          {.method = method, .rtol = 1e-12});
      EXPECT_TRUE(total.converged);
      errors[method] =
          Gap(NodePairing(setup, total.kernels.thickness, deltaIce), fd);
      multipliers[method] = total.multiplier;
    }
    const auto& discrete = multipliers[AdjointMethod::Discrete];
    return std::array<double, 3>{
        errors[AdjointMethod::Discrete], errors[AdjointMethod::Continuous],
        MaxDiff(multipliers[AdjointMethod::Continuous], discrete) /
            MaxAbs(discrete)};
  };
  const auto coarse = check(5);
  const auto fine = check(9);
  const auto finer = check(17);
  std::cout << std::format(
      "[calibrated, 1D, continuous] ice: discrete {:.2e}, {:.2e} and {:.2e}; "
      "continuous {:.2e}, {:.2e} and {:.2e} (ratios {:.2f} and {:.2f}); the "
      "multipliers apart by {:.2e}, {:.2e} and {:.2e} (ratios {:.2f} and "
      "{:.2f})\n",
      coarse[0], fine[0], finer[0], coarse[1], fine[1], finer[1],
      coarse[1] / fine[1], fine[1] / finer[1], coarse[2], fine[2], finer[2],
      coarse[2] / fine[2], fine[2] / finer[2]);
  // (the differences are small against the objective on this stiff mantle,
  // and keep fewer digits of the calibrations' tolerance)
  for (const auto& level : {coarse, fine, finer}) EXPECT_LT(level[0], 1e-6);
  // the total kernel is the partial one less a correction of its own
  // size, each at the commute level: their difference keeps less of it
  EXPECT_LT(coarse[1], 0.5);
  EXPECT_LT(finer[1], 0.05);
  EXPECT_GT(coarse[1] / fine[1], 3.0);
  EXPECT_GT(fine[1] / finer[1], 3.0);
  EXPECT_GT(coarse[2] / fine[2], 3.0);
  EXPECT_GT(fine[2] / finer[2], 3.0);
}

TEST(CalibratedKernels, TheLinearisedCalibrationIsTheMultipliersTranspose) {
  // The total tangent-linear run of the calibrated migrating run of the
  // marine cap, under Sdirk2 and the exponential trapezoid, the discrete
  // runs, along a direction in the thickness at every node, the relaxation
  // times and the moduli together with a change of the target:
  //  - its objective is the total kernels paired with the direction plus
  //    the multiplier paired with the target's change, the two modes of one
  //    derivative;
  //  - the change of the calibrated initial sea level it carries is the
  //    difference of the initial sea levels of calibrations afresh, along
  //    the thickness and along the target (linear in each while no cell
  //    flips);
  //  - the plain iteration reaches the same change in more runs;
  //  - a direction's own initial sea level is refused where the calibration
  //    sets it.
  for (const auto scheme : {Scheme::Sdirk2, Scheme::ExponentialTrapezoid}) {
    const auto label =
        std::string{scheme == Scheme::Sdirk2 ? "sdirk2" : "trapezoid"};
    const auto scene = Marine1D(1e21, 6, scheme);
    const auto& setup = scene.setup;
    const auto& times = scene.cap.times;
    const auto nodes = times.size();
    std::mt19937 rng(251);
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[1].relativeSeaLevelDual = setup.Smooth(rng, 0.4);
    measurements[3].displacementDual = setup.Smooth(rng, 0.7);
    measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
    auto deltaIce = Fields{};
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      deltaIce.push_back(setup.Smooth(rng, 0.1 * setup.Metre()));
    }
    const auto deltaTarget = setup.Smooth(rng, 0.1 * setup.Metre());
    auto direction = TangentPerturbation{};
    direction.iceThickness = deltaIce;
    direction.itau = setup.DeltaItau(0.1);
    direction.material = PointDirection(scene.op, setup.material);
    const auto none = Fields{};

    Evolution evolution(scene.op, setup.solver, setup.state, scene.options);
    const auto base = scene.Calibrate(evolution, none, 0.0, scene.target,
                                      Shorelines::Migrating);
    const auto oceans = OceanFunctionsOf(base.snapshots);
    const auto stageOceans = StageOceanFunctionsOf(base.snapshots);
    ASSERT_NE(oceans.back(), oceans.front()) << "no cell flipped";
    const auto trajectory = ForwardTrajectory::FromSnapshots(base.snapshots);

    const auto adjoint =
        AdjointSolver::Options{.method = AdjointMethod::Discrete,
                               .rtol = 1e-12,
                               .material_kernels = true};
    const auto total = ComputeCalibratedKernels(
        evolution, times, measurements, &trajectory, oceans, stageOceans,
        {.rtol = 1e-12}, adjoint);
    ASSERT_TRUE(total.converged);
    const auto& mesh = setup.material.Mesh();
    const auto predicted =
        NodePairing(setup, total.kernels.thickness, deltaIce) +
        VolumePairing(mesh, total.kernels.itauKernel[0], direction.itau[0]) +
        VolumePairing(scene.op.Points(), mesh, total.kernels.kappaKernel,
                      direction.material.kappa) +
        VolumePairing(scene.op.Points(), mesh, total.kernels.muKernel,
                      direction.material.mu) +
        setup.Surface(total.multiplier, deltaTarget);

    const auto runs = TangentLinearSolver::Options{
        .method = AdjointMethod::Discrete, .rtol = 1e-12};
    const auto tangent = ComputeCalibratedTangent(
        evolution, times, direction, deltaTarget, &trajectory, oceans,
        stageOceans, {.rtol = 1e-12}, runs);
    EXPECT_TRUE(tangent.converged);
    // the run along the direction, one per pass after the first, and the
    // run that formed the snapshots
    EXPECT_EQ(tangent.runs, static_cast<int>(tangent.updates.size()) + 1);
    const auto along = Objective(setup, tangent.snapshots, measurements);
    // the partial run, the initial sea level held, is another number
    const auto held = Objective(
        setup,
        TangentLinearSolver(evolution, runs)
            .Run(times, direction, &trajectory, oceans, stageOceans),
        measurements);

    // the change of the calibrated initial sea level, against calibrations
    // afresh
    auto calibrated = [&](const Fields& d, double eps,
                          std::span<const double> towards) {
      Evolution trial(scene.op, setup.solver, setup.state, scene.options);
      const auto found =
          scene.Calibrate(trial, d, eps, towards, Shorelines::Migrating);
      EXPECT_EQ(found.state.oceanFunction, base.state.oceanFunction)
          << "the start flipped";
      for (auto n = std::size_t{0}; n < nodes; ++n) {
        EXPECT_EQ(found.snapshots[n].oceanFunction, oceans[n])
            << "a cell flipped at node " << n;
      }
      return found.state.seaLevel;
    };
    const auto half = [](std::vector<double> up,
                         const std::vector<double>& down) {
      for (auto i = std::size_t{0}; i < up.size(); ++i) {
        up[i] = 0.5 * (up[i] - down[i]);
      }
      return up;
    };
    auto iceAlone = TangentPerturbation{};
    iceAlone.iceThickness = deltaIce;
    const auto byIce = SolveCalibrationTangent(
        evolution, times, iceAlone, {}, nullptr, oceans, stageOceans,
        {.rtol = 1e-12}, runs);
    EXPECT_TRUE(byIce.converged);
    const auto fdIce = half(calibrated(deltaIce, 1.0, scene.target),
                            calibrated(deltaIce, -1.0, scene.target));
    const auto byTarget = SolveCalibrationTangent(
        evolution, times, {}, deltaTarget, nullptr, oceans, stageOceans,
        {.rtol = 1e-12}, runs);
    EXPECT_TRUE(byTarget.converged);
    const auto fdTarget =
        half(calibrated(none, 0.0, Moved(scene.target, deltaTarget, 1.0)),
             calibrated(none, 0.0, Moved(scene.target, deltaTarget, -1.0)));
    std::cout << std::format(
        "[calibrated tangent, 1D, {}] the linearised calibration in {} runs, "
        "updates{}\n"
        "  its objective {:.2e} from the total kernels' pairing (the partial "
        "run {:.2e})\n"
        "  the initial sea level's change from calibrations afresh: ice "
        "{:.2e}, target {:.2e}\n",
        label, tangent.runs, Listed(tangent.updates), Gap(along, predicted),
        Gap(held, predicted), MaxDiff(byIce.initialSeaLevel, fdIce) / MaxAbs(fdIce),
        MaxDiff(byTarget.initialSeaLevel, fdTarget) / MaxAbs(fdTarget));
    EXPECT_NEAR(along, predicted, 1e-8 * std::abs(predicted)) << label;
    EXPECT_GT(Gap(held, predicted), 1e-2) << label;
    EXPECT_LT(MaxDiff(byIce.initialSeaLevel, fdIce), 1e-7 * MaxAbs(fdIce))
        << label;
    EXPECT_LT(MaxDiff(byTarget.initialSeaLevel, fdTarget),
              1e-7 * MaxAbs(fdTarget))
        << label;

    // the plain iteration: the same change in more runs
    const auto plain = SolveCalibrationTangent(
        evolution, times, iceAlone, {}, nullptr, oceans, stageOceans,
        {.iterations = 60, .rtol = 1e-12, .anderson_memory = 0}, runs);
    EXPECT_TRUE(plain.converged);
    EXPECT_GT(plain.runs, byIce.runs + 5) << label;
    EXPECT_LT(MaxDiff(plain.initialSeaLevel, byIce.initialSeaLevel),
              1e-10 * MaxAbs(byIce.initialSeaLevel))
        << label;
    // an iteration that runs out says so
    const auto short_ = SolveCalibrationTangent(
        evolution, times, iceAlone, {}, nullptr, oceans, stageOceans,
        {.iterations = 2, .rtol = 1e-12}, runs);
    EXPECT_FALSE(short_.converged);
    EXPECT_EQ(short_.updates.size(), 2u);
    // every point is calibrated here: an initial sea level of the
    // direction's own has no place
    auto withSeaLevel = iceAlone;
    withSeaLevel.initialSeaLevel = deltaTarget;
    EXPECT_THROW(static_cast<void>(SolveCalibrationTangent(
                     evolution, times, withSeaLevel, {}, nullptr, oceans,
                     stageOceans, {}, runs)),
                 std::invalid_argument);
  }
}

TEST(CalibratedKernels, TheLinearisedCalibrationUnderFrozenShorelines) {
  // Under frozen shorelines the run does not depend on its initial sea
  // level: the linearised calibration is one run, the initial sea level's
  // change the target's less the run's at the last node, as calibrations
  // afresh give it, and the run's objective the kernels' pairing with the
  // direction, whatever the target does.
  const auto scene = Marine1D(1e21, 6, Scheme::Sdirk2);
  const auto& setup = scene.setup;
  const auto& times = scene.cap.times;
  const auto nodes = times.size();
  std::mt19937 rng(257);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[3].displacementDual = setup.Smooth(rng, 0.7);
  measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  auto direction = TangentPerturbation{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    direction.iceThickness.push_back(setup.Smooth(rng, 0.1 * setup.Metre()));
  }
  const auto deltaTarget = setup.Smooth(rng, 0.1 * setup.Metre());
  Evolution evolution(scene.op, setup.solver, setup.state, scene.options);
  const auto base =
      scene.Calibrate(evolution, {}, 0.0, scene.target, Shorelines::Frozen);
  const auto runs = TangentLinearSolver::Options{.rtol = 1e-12};
  auto calls = 0;
  const auto tangent = ComputeCalibratedTangent(
      evolution, times, direction, deltaTarget, nullptr, {}, {},
      {.on_iteration = [&](int, double) { ++calls; }}, runs);
  EXPECT_TRUE(tangent.converged);
  EXPECT_EQ(tangent.runs, 1);
  EXPECT_EQ(calls, 0);
  EXPECT_TRUE(tangent.updates.empty());
  const auto& last = tangent.snapshots.back().relativeSeaLevelChange;
  auto expected = deltaTarget;
  for (auto i = std::size_t{0}; i < expected.size(); ++i) expected[i] -= last[i];
  EXPECT_LE(MaxDiff(tangent.initialSeaLevel, expected),
            1e-14 * MaxAbs(expected));
  auto calibrated = [&](double eps) {
    Evolution trial(scene.op, setup.solver, setup.state, scene.options);
    const auto found =
        scene.Calibrate(trial, direction.iceThickness, eps,
                        Moved(scene.target, deltaTarget, eps),
                        Shorelines::Frozen);
    EXPECT_EQ(found.state.oceanFunction, base.state.oceanFunction);
    return found.state.seaLevel;
  };
  auto fd = calibrated(1.0);
  {
    const auto down = calibrated(-1.0);
    for (auto i = std::size_t{0}; i < fd.size(); ++i) {
      fd[i] = 0.5 * (fd[i] - down[i]);
    }
  }
  const auto kernels = AdjointSolver(evolution, {.rtol = 1e-12})
                           .FirstOrderKernels(times, measurements);
  const auto predicted =
      NodePairing(setup, kernels.thickness, direction.iceThickness);
  const auto along = Objective(setup, tangent.snapshots, measurements);
  std::cout << std::format(
      "[calibrated tangent, 1D, frozen] the initial sea level's change {:.2e} "
      "from calibrations afresh; its objective {:.2e} from the kernels' "
      "pairing\n",
      MaxDiff(tangent.initialSeaLevel, fd) / MaxAbs(fd), Gap(along, predicted));
  EXPECT_LT(MaxDiff(tangent.initialSeaLevel, fd), 1e-8 * MaxAbs(fd));
  EXPECT_NEAR(along, predicted, 1e-9 * std::abs(predicted));
  // a change of the target alone moves the initial sea level by itself
  // and nothing of the run
  const auto alone = ComputeCalibratedTangent(evolution, times, {}, deltaTarget,
                                              nullptr, {}, {}, {}, runs);
  EXPECT_EQ(alone.initialSeaLevel, deltaTarget);
  ASSERT_EQ(alone.snapshots.size(), nodes);
  EXPECT_EQ(MaxAbs(alone.snapshots.back().relativeSeaLevelChange), 0.0);
}

// -- 3D ---------------------------------------------------------------------

namespace {

constexpr auto kLmax3 = 8;

double LateralDelta(double, double theta, double phi) {
  return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
}

double Bell(double theta, double phi, double colat0, double lon0,
            double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

/// A direction of the log viscosity over the sphere.
double Pattern(double theta, double phi) {
  return Bell(theta, phi, 1.9, 4.4, 0.7) - Bell(theta, phi, 0.9, 1.1, 0.6);
}

/// The 1D scene's problem at degree 8 under lateral viscosity: the prem8
/// body with a Maxwell mantle, a marine cap that melts to nothing over
/// four steps (9 km at its peak, so that its band-limited thickness
/// grounds on the 3 km ocean) on a non-uniform initial sea level, and the
/// final sea level its migrating run reaches.
struct Marine3D {
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  std::vector<double> times;
  Fields ice;
  std::vector<double> target;
  Evolution3D::Options options{.rtol = 1e-12,
                               .coupled_rtol = 1e-12,
                               .max_nonlinear_iterations = 200,
                               .record_stride = 1};

  static Marine3D Make() {
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
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    ViscoelasticOperator reference(rheology, kLmax3);
    GRDSolver solver(SurfaceGrid(kLmax3), reference.ElasticLoveNumbers(),
                     SeaLevelParameters{});
    return Marine3D{std::move(model), std::move(material), std::move(rheology),
                    std::move(solver), {}, {}, {}};
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
  [[nodiscard]] double Pairing(const Fields& kernel, const Fields& d) const {
    auto out = 0.0;
    for (auto n = std::size_t{0}; n < d.size(); ++n) {
      out += Surface(kernel[n], d[n]);
    }
    return out;
  }
  [[nodiscard]] double Objective(
      const std::vector<Evolution3D::Snapshot>& snapshots,
      const std::map<std::size_t, Measurement>& measurements) const {
    auto total = 0.0;
    for (const auto& [node, m] : measurements) {
      const auto& snap = snapshots[node];
      if (!m.relativeSeaLevelDual.empty()) {
        total += Surface(m.relativeSeaLevelDual, snap.relativeSeaLevelChange);
      }
      if (!m.displacementDual.empty()) {
        total += Surface(m.displacementDual, snap.displacement);
      }
    }
    return total;
  }

  /// The history and its target, from the run of `op` on the true start.
  void Set(const Viscoelastic3D& op, int nodes) {
    const auto& grid = solver.Grid();
    const auto metre = Metre();
    const auto kyr = 1e3 * kYear / solver.SolverScales().Time();
    const auto cap = grid.Synthesise(
        grid.Expand(grid.FromFunction([&](double theta, double phi) {
          const auto theta0 = 30.0 * std::numbers::pi / 180.0;
          const auto phi0 = 275.0 * std::numbers::pi / 180.0;
          const auto radius = 20.0 * std::numbers::pi / 180.0;
          const auto c = std::cos(theta) * std::cos(theta0) +
                         std::sin(theta) * std::sin(theta0) *
                             std::cos(phi - phi0);
          const auto d = std::acos(std::clamp(c, -1.0, 1.0));
          if (d >= radius) return 0.0;
          const auto taper = std::cos(0.5 * std::numbers::pi * d / radius);
          return 9000.0 * metre * taper * taper;
        })));
    for (auto n = 0; n < nodes; ++n) {
      const auto s = static_cast<double>(n) / (nodes - 1);
      times.push_back(10.0 * kyr * s);
      ice.push_back(cap);
      for (auto& v : ice.back()) v *= 1.0 - s;
    }
    const auto sea = grid.FromFunction([&](double theta, double phi) {
      return (3000.0 + 200.0 * std::sin(theta) * std::cos(phi)) * metre;
    });
    const auto truth = solver.MakeState(cap, sea);
    Evolution3D evolution(op, solver, truth, options);
    const auto run = evolution.RunMigrating(times, Field(times, ice));
    EXPECT_NE(run.back().oceanFunction, truth.oceanFunction);
    target = Moved(truth.seaLevel, run.back().relativeSeaLevelChange, 1.0);
  }

  [[nodiscard]] InitialSeaLevelCalibration3D Calibrate(
      Evolution3D& evolution, const Fields& direction, double eps) const {
    const auto first =
        direction.empty() ? ice[0] : Moved(ice[0], direction[0], eps);
    evolution.SetBackgroundState(solver.MakeState(first, target));
    auto calibration = CalibrateInitialSeaLevel(
        evolution, times,
        IceHistory::Migrating(Field(times, ice, direction, eps)), target,
        {.iterations = 80, .rtol = 1e-12});
    EXPECT_TRUE(calibration.converged)
        << "in " << calibration.updates.size() << " runs, the last update "
        << calibration.updates.back();
    return calibration;
  }
};

std::vector<double> OnCoupling(const GridCoupling& coupling,
                               const std::function<double(double, double)>& f) {
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto values = coupling.GridOf(rn).FromFunction(f);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x) {
      out[at + x] = values[x];
    }
  }
  return out;
}

}  // namespace

TEST(CalibratedKernels, TotalKernelsPredictReCalibratedRuns3D) {
  // The 1D gate under lateral viscosity, Sdirk2, the discrete 3D sweep: a
  // thickness direction at every node, between runs calibrated afresh that
  // flip no cell, and a lateral direction of the log viscosity by
  // Richardson's value. The total kernels pair with both, the partial ones
  // with neither, and the final sweep's initial sea level kernel is the
  // multiplier.
  auto scene = Marine3D::Make();
  const Viscoelastic3D op(scene.rheology, kLmax3, LateralDelta);
  scene.Set(op, 4);
  const auto& times = scene.times;
  const auto nodes = times.size();
  std::mt19937 rng(239);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[1].displacementDual = scene.Smooth(rng, 0.7);
  measurements[nodes - 1].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  auto deltaIce = Fields{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    deltaIce.push_back(scene.Smooth(rng, 0.1 * scene.Metre()));
  }
  const auto none = Fields{};
  const auto state = scene.solver.MakeState(scene.ice[0], scene.target);

  Evolution3D evolution(op, scene.solver, state, scene.options);
  const auto base = scene.Calibrate(evolution, none, 0.0);
  const auto oceans = OceanFunctionsOf(base.snapshots);
  const auto stageOceans = StageOceanFunctionsOf(base.snapshots);
  ASSERT_NE(oceans.back(), oceans.front()) << "no cell flipped";
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(base.snapshots);

  auto objective = [&](const Viscoelastic3D& at, const Fields& d, double eps) {
    Evolution3D trial(at, scene.solver, state, scene.options);
    const auto calibrated = scene.Calibrate(trial, d, eps);
    // flotation's 0/1: rounding changes it only at a cell on the threshold
    EXPECT_EQ(calibrated.state.oceanFunction, base.state.oceanFunction)
        << "the start flipped";
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      EXPECT_TRUE(calibrated.snapshots[n].converged) << "node " << n;
      EXPECT_EQ(calibrated.snapshots[n].oceanFunction, oceans[n])
          << "a cell flipped at node " << n;
    }
    return scene.Objective(calibrated.snapshots, measurements);
  };
  const auto fdIce =
      0.5 * (objective(op, deltaIce, 1.0) - objective(op, deltaIce, -1.0));
  const auto fdViscosity = Richardson(
      [&](double eps) {
        const Viscoelastic3D moved(
            scene.rheology, kLmax3, [&](double r, double theta, double phi) {
              return LateralDelta(r, theta, phi) + eps * Pattern(theta, phi);
            });
        return objective(moved, none, 0.0);
      },
      0.02);

  const auto adjoint =
      AdjointSolver3D::Options{.rtol = 1e-12, .coupled_rtol = 1e-12};
  const auto partial =
      AdjointSolver3D(evolution, adjoint)
          .FirstOrderKernels(times, measurements, &trajectory, oceans,
                             stageOceans);
  const auto total = ComputeCalibratedKernels(
      evolution, times, measurements, &trajectory, oceans, stageOceans,
      {.rtol = 1e-11}, adjoint);
  EXPECT_TRUE(total.converged);
  const auto pattern = OnCoupling(op.Coupling(), Pattern);
  const auto pairings = [&](const Kernels3D& k) {
    return std::array<double, 2>{
        scene.Pairing(k.ice.thickness, deltaIce),
        VolumePairing(op.Coupling(), k.logViscosityKernel3D[0], pattern)};
  };
  const auto whole = pairings(total.kernels);
  const auto held = pairings(partial);
  std::cout << std::format(
      "[calibrated, 3D] multiplier in {} sweeps, updates{}, contraction "
      "{:.2e}\n"
      "  total from the differences: ice {:.2e}, log viscosity {:.2e}\n"
      "  partial from them:          ice {:.2e}, log viscosity {:.2e}\n",
      total.updates.size(), Listed(total.updates), Contraction(total.updates),
      Gap(whole[0], fdIce),
      Gap(whole[1], fdViscosity), Gap(held[0], fdIce),
      Gap(held[1], fdViscosity));
  EXPECT_NEAR(whole[0], fdIce, 1e-7 * std::abs(fdIce));
  EXPECT_NEAR(whole[1], fdViscosity, 1e-5 * std::abs(fdViscosity));
  EXPECT_GT(Gap(held[0], fdIce), 1e-2);
  EXPECT_GT(Gap(held[1], fdViscosity), 1e-2);
  EXPECT_GT(MaxAbs(total.multiplier), 0.0);
  EXPECT_LT(MaxDiff(total.kernels.ice.initialSeaLevel, total.multiplier),
            1e-9 * MaxAbs(total.multiplier));
  // the second-order sweep about the calibrated state is refused by name,
  // as the radial one
  EXPECT_TRUE(evolution.Calibrated());
  try {
    static_cast<void>(
        AdjointSolver3D(evolution, adjoint)
            .SecondOrderKernels(times, measurements, {}, {}, trajectory,
                                oceans, {}, stageOceans));
    ADD_FAILURE() << "not refused";
  } catch (const std::logic_error& error) {
    EXPECT_NE(std::string{error.what()}.find("are not built"),
              std::string::npos)
        << error.what();
  }
}

TEST(CalibratedKernels, TheLinearisedCalibrationIsTheMultipliersTranspose3D) {
  // The 1D gate under lateral viscosity, Sdirk2, the discrete runs: the
  // total tangent-linear run along a thickness direction at every node and
  // a lateral direction of the log viscosity together with a change of the
  // target pairs with the objective's duals as the total kernels do with
  // the direction and the multiplier with the target's change; the partial
  // run does not.
  auto scene = Marine3D::Make();
  const Viscoelastic3D op(scene.rheology, kLmax3, LateralDelta);
  scene.Set(op, 4);
  const auto& times = scene.times;
  const auto nodes = times.size();
  std::mt19937 rng(263);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[1].displacementDual = scene.Smooth(rng, 0.7);
  measurements[nodes - 1].relativeSeaLevelDual = scene.Smooth(rng, 1.0);
  auto direction = TangentPerturbation3D{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    direction.iceThickness.push_back(scene.Smooth(rng, 0.1 * scene.Metre()));
  }
  direction.logViscosity = OnCoupling(op.Coupling(), Pattern);
  const auto deltaTarget = scene.Smooth(rng, 0.1 * scene.Metre());
  const auto state = scene.solver.MakeState(scene.ice[0], scene.target);

  Evolution3D evolution(op, scene.solver, state, scene.options);
  const auto base = scene.Calibrate(evolution, {}, 0.0);
  const auto oceans = OceanFunctionsOf(base.snapshots);
  const auto stageOceans = StageOceanFunctionsOf(base.snapshots);
  ASSERT_NE(oceans.back(), oceans.front()) << "no cell flipped";
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(base.snapshots);

  const auto total = ComputeCalibratedKernels(
      evolution, times, measurements, &trajectory, oceans, stageOceans,
      {.rtol = 1e-11},
      AdjointSolver3D::Options{.rtol = 1e-12, .coupled_rtol = 1e-12});
  ASSERT_TRUE(total.converged);
  const auto predicted =
      scene.Pairing(total.kernels.ice.thickness, direction.iceThickness) +
      VolumePairing(op.Coupling(), total.kernels.logViscosityKernel3D[0],
                    direction.logViscosity) +
      scene.Surface(total.multiplier, deltaTarget);

  const auto runs =
      TangentLinearSolver3D::Options{.rtol = 1e-12, .coupled_rtol = 1e-12};
  const auto tangent = ComputeCalibratedTangent(
      evolution, times, direction, deltaTarget, &trajectory, oceans,
      stageOceans, {.rtol = 1e-11}, runs);
  EXPECT_TRUE(tangent.converged);
  const auto along = scene.Objective(tangent.snapshots, measurements);
  const auto held = scene.Objective(
      TangentLinearSolver3D(evolution, runs)
          .Run(times, direction, &trajectory, oceans, stageOceans),
      measurements);
  std::cout << std::format(
      "[calibrated tangent, 3D] the linearised calibration in {} runs, "
      "updates{}; its objective {:.2e} from the total kernels' pairing (the "
      "partial run {:.2e})\n",
      tangent.runs, Listed(tangent.updates), Gap(along, predicted),
      Gap(held, predicted));
  EXPECT_NEAR(along, predicted, 1e-7 * std::abs(predicted));
  EXPECT_GT(Gap(held, predicted), 1e-2);
}
