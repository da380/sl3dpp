// The tangent-linear runs: linear in the loads (the run along a load
// perturbation is the difference of two forward runs), the exact
// transpose of the discrete adjoint (the pairing identity with the load
// duals and the itau kernel at solver tolerance), the continuous drive
// converging to the discrete one at the scheme's order, the derivative
// about a migrating trajectory against central differences while no
// flotation cell flips, the second derivatives of the scheme weights
// against differences of the first, and the second-order run as the
// derivative of the first-order one — pyslfp's test_tangent_linear.py.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <functional>
#include <map>
#include <numbers>
#include <random>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/Viscoelastic.hpp"

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

}  // namespace

#include "KernelScene.hpp"

using namespace kernel_scene;

TEST(StepWeights, SecondDerivativesMatchDifferencesOfTheFirst) {
  // Central differences of the first h-derivatives land on the second
  // ones, across the series switch and well above it, for both schemes
  // (the step wide enough that the quotients' rounding, ~1e-11 just above
  // the switch, stays below the truncation error).
  for (const double h : {0.0, 5e-4, 2e-3, 0.1, 1.0, 4.0}) {
    const auto eps = 1e-2 * std::max(h, 1e-2);
    const auto lo = h - eps >= 0.0 ? h - eps : 0.0;
    const auto hi = lo + 2.0 * eps;
    const auto a = ExponentialTrapezoidWeightDerivatives(lo);
    const auto b = ExponentialTrapezoidWeightDerivatives(hi);
    const auto second = ExponentialTrapezoidWeightSecondDerivatives(0.5 * (lo + hi));
    EXPECT_NEAR((b.e - a.e) / (hi - lo), second.e, 1e-5) << "h " << h;
    EXPECT_NEAR((b.alpha - a.alpha) / (hi - lo), second.alpha, 1e-5) << "h " << h;
    EXPECT_NEAR((b.beta - a.beta) / (hi - lo), second.beta, 1e-5) << "h " << h;
    // The backward Euler weight has no cancellation, so the narrow step.
    const auto fine = 1e-4 * std::max(h, 1e-2);
    EXPECT_NEAR((BackwardEulerWeightDerivative(h + fine) -
                 BackwardEulerWeightDerivative(h)) /
                    fine,
                BackwardEulerWeightSecondDerivative(h + 0.5 * fine), 1e-6)
        << "h " << h;
  }
  // The identities d2e = e and d2alpha + d2beta = -d2e (alpha + beta =
  // 1 - e at every order), and the series' value at zero, d2phi1(0) = 1/3.
  const auto at = ExponentialTrapezoidWeightSecondDerivatives(0.0);
  EXPECT_DOUBLE_EQ(at.e, 1.0);
  EXPECT_DOUBLE_EQ(at.beta, -1.0 / 3.0);
  EXPECT_DOUBLE_EQ(at.alpha + at.beta, -at.e);
  EXPECT_THROW((void)ExponentialTrapezoidWeightSecondDerivatives(-1.0),
               std::invalid_argument);
  EXPECT_THROW((void)BackwardEulerWeightSecondDerivative(-1.0),
               std::invalid_argument);
}

TEST(TangentLinear, LinearInTheLoads) {
  // The frozen problem is linear in its loads: the run along a load
  // perturbation is the difference of two forward runs, node by node and
  // channel by channel, to the solvers' tolerance, for both schemes and
  // both methods (which coincide in the loads).
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(7);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * setup.Kyr());
  auto loads = std::vector<std::vector<double>>{};
  auto deltas = std::vector<std::vector<double>>{};
  auto shifted = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    deltas.push_back(setup.Smooth(rng, 10.0 * setup.Sigma()));
    auto sum = loads.back();
    for (auto i = std::size_t{0}; i < sum.size(); ++i) sum[i] += deltas.back()[i];
    shifted.push_back(std::move(sum));
  }
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(op, setup.solver, setup.state,
                                {.scheme = scheme, .rtol = 1e-12});
    const auto base = evolution.Run(times, History(times, loads));
    const auto plus = evolution.Run(times, History(times, shifted));
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver tangent(evolution, {.method = method, .rtol = 1e-12});
      TangentPerturbation direction;
      direction.directLoad = deltas;
      const auto run = tangent.Run(times, direction);
      ASSERT_EQ(run.size(), times.size());
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        for (auto field : {&Evolution::Snapshot::relativeSeaLevelChange,
                           &Evolution::Snapshot::displacement,
                           &Evolution::Snapshot::potential}) {
          auto difference = plus[n].*field;
          for (auto i = std::size_t{0}; i < difference.size(); ++i) {
            difference[i] -= (base[n].*field)[i];
          }
          EXPECT_LT(MaxDiff(run[n].*field, difference),
                    1e-8 * MaxAbs(difference))
              << "node " << n << " scheme " << static_cast<int>(scheme);
        }
        for (auto j = 0; j < 3; ++j) {
          const auto difference =
              plus[n].angularVelocity[j] - base[n].angularVelocity[j];
          EXPECT_NEAR(run[n].angularVelocity[j], difference,
                      1e-8 * std::abs(difference) + 1e-20)
              << "node " << n;
        }
        EXPECT_TRUE(run[n].oceanFunction.empty());
        EXPECT_FALSE(run[n].internalState.has_value());
      }
    }
  }
}

TEST(TangentLinear, DiscretePairAreTransposes) {
  // The discrete tangent-linear run and the discrete adjoint sweep are
  // exact transposes: the objective of the run along (delta f, delta itau)
  // equals the pairing of the load duals with delta f, node 0 included,
  // plus the itau kernel's with delta itau, to the solvers' tolerance —
  // the forward mode of the scheme weights against their reverse mode.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(11);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * setup.Kyr());
  auto loads = std::vector<std::vector<double>>{};
  auto deltas = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    deltas.push_back(setup.Smooth(rng, 10.0 * setup.Sigma()));
  }
  const auto measurements = FullMeasurements(setup, rng, times.size() - 1);
  const auto ditau = setup.DeltaItau(0.1);

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(
        op, setup.solver, setup.state,
        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    const auto snapshots = evolution.Run(times, History(times, loads));
    const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);

    AdjointSolver adjoint(evolution,
                          {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto kernels =
        adjoint.FirstOrderKernels(times, measurements, &trajectory);
    auto rhs = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      rhs += setup.Surface(kernels.directLoadKernels[n], deltas[n]);
    }
    rhs += VolumePairing(setup.material.Mesh(), kernels.itauKernel[0], ditau[0]);

    TangentLinearSolver tangent(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    TangentPerturbation direction;
    direction.directLoad = deltas;
    direction.itau = ditau;
    const auto run = tangent.Run(times, direction, &trajectory);
    const auto lhs = Objective(setup, run, measurements);
    EXPECT_NEAR(lhs, rhs, 1e-8 * std::abs(lhs))
        << "scheme " << static_cast<int>(scheme);

    // Each argument alone pairs with its own kernel.
    TangentPerturbation loadsOnly;
    loadsOnly.directLoad = deltas;
    auto loadsRhs = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      loadsRhs += setup.Surface(kernels.directLoadKernels[n], deltas[n]);
    }
    EXPECT_NEAR(Objective(setup, tangent.Run(times, loadsOnly), measurements),
                loadsRhs, 1e-8 * std::abs(loadsRhs));
    TangentPerturbation itauOnly;
    itauOnly.itau = ditau;
    EXPECT_NEAR(
        Objective(setup, tangent.Run(times, itauOnly, &trajectory), measurements),
        rhs - loadsRhs, 1e-8 * std::abs(rhs - loadsRhs));

    // A checkpointed trajectory serves the same run: the gaps are refilled
    // by forward steps, which amplify nothing.
    Evolution sparse(op, setup.solver, setup.state,
                             {.scheme = scheme, .rtol = 1e-12, .record_stride = 2});
    const auto sparseTrajectory =
        ForwardTrajectory::FromSnapshots(sparse.Run(times, History(times, loads)));
    EXPECT_NEAR(Objective(setup, tangent.Run(times, direction, &sparseTrajectory),
                          measurements),
                lhs, 1e-10 * std::abs(lhs));

    // Refusals: itau without a trajectory, an empty direction, a direct
    // load about a migrating trajectory.
    EXPECT_THROW((void)tangent.Run(times, itauOnly), std::invalid_argument);
    EXPECT_THROW((void)tangent.Run(times, TangentPerturbation{}),
                 std::invalid_argument);
    const auto oceans = std::vector<std::vector<double>>(
        times.size(), setup.state.oceanFunction);
    EXPECT_THROW((void)tangent.Run(times, loadsOnly, nullptr, oceans),
                 std::invalid_argument);
  }
}

TEST(TangentLinear, WithoutAnOceanThePairAreTransposes) {
  // On a state with no ocean the evolution is the solid body under a
  // prescribed load, with the rotational feedback: its tangent-linear run
  // is the difference of two runs in the loads (the problem is linear in
  // them), and the discrete run and sweep are exact transposes in the
  // loads and the relaxation times, as with an ocean.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(23);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * setup.Kyr());
  auto loads = std::vector<std::vector<double>>{};
  auto deltas = std::vector<std::vector<double>>{};
  auto shifted = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    deltas.push_back(setup.Smooth(rng, 10.0 * setup.Sigma()));
    shifted.push_back(loads.back());
    for (auto i = std::size_t{0}; i < shifted.back().size(); ++i) {
      shifted.back()[i] += deltas.back()[i];
    }
  }
  const auto measurements = FullMeasurements(setup, rng, times.size() - 1);
  const auto ditau = setup.DeltaItau(0.1);
  const auto dry = setup.solver.NoOceanState();

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(
        op, setup.solver, dry,
        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    const auto snapshots = evolution.Run(times, History(times, loads));
    const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);
    const auto moved = evolution.Run(times, History(times, shifted));

    TangentLinearSolver tangent(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    TangentPerturbation loadsOnly;
    loadsOnly.directLoad = deltas;
    const auto alongLoads =
        Objective(setup, tangent.Run(times, loadsOnly), measurements);
    const auto difference = Objective(setup, moved, measurements) -
                            Objective(setup, snapshots, measurements);
    EXPECT_NEAR(alongLoads, difference, 1e-8 * std::abs(difference))
        << "scheme " << static_cast<int>(scheme);

    AdjointSolver adjoint(evolution,
                          {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto kernels =
        adjoint.FirstOrderKernels(times, measurements, &trajectory);
    auto loadsRhs = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      loadsRhs += setup.Surface(kernels.directLoadKernels[n], deltas[n]);
    }
    EXPECT_NEAR(alongLoads, loadsRhs, 1e-8 * std::abs(alongLoads))
        << "scheme " << static_cast<int>(scheme);

    TangentPerturbation itauOnly;
    itauOnly.itau = ditau;
    const auto alongItau = Objective(
        setup, tangent.Run(times, itauOnly, &trajectory), measurements);
    const auto itauRhs =
        VolumePairing(setup.material.Mesh(), kernels.itauKernel[0], ditau[0]);
    EXPECT_NE(alongItau, 0.0);
    EXPECT_NEAR(alongItau, itauRhs, 1e-8 * std::abs(alongItau))
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(TangentLinear, ATideAndTheTangentialDisplacementInThePair) {
  // Under a direct load and a prescribed tide, with the tangential
  // displacement's potential among the measurements: the tangent-linear
  // run along a perturbation of the load and the tide is the difference
  // of two runs, and the discrete run and sweep are exact transposes in
  // the load, the tide, the relaxation times and the moduli, with an
  // ocean and without one.
  auto setup = Scene::Make(1e21);
  const auto& grid = setup.solver.Grid();
  std::mt19937 rng(41);
  using Complex = std::complex<double>;
  const auto potential =
      1.0 / setup.solver.SolverScales().Factor(PotentialDimensions);
  auto smoothTide = [&](double amplitude) {
    auto out = grid.Expand(setup.Smooth(rng, amplitude * potential));
    out[SurfaceGrid::Index(0, 0)] = 0.0;
    return out;
  };
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.0 * n * setup.Kyr());
  auto loads = std::vector<std::vector<double>>{};
  auto deltas = std::vector<std::vector<double>>{};
  auto shifted = std::vector<std::vector<double>>{};
  auto tides = std::vector<std::vector<Complex>>{};
  auto tideDeltas = std::vector<std::vector<Complex>>{};
  auto tidesShifted = std::vector<std::vector<Complex>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    deltas.push_back(setup.Smooth(rng, 10.0 * setup.Sigma()));
    shifted.push_back(loads.back());
    for (auto i = std::size_t{0}; i < shifted.back().size(); ++i) {
      shifted.back()[i] += deltas.back()[i];
    }
    tides.push_back(smoothTide(1.0));
    tideDeltas.push_back(smoothTide(0.1));
    tidesShifted.push_back(tides.back());
    for (auto i = std::size_t{0}; i < tidesShifted.back().size(); ++i) {
      tidesShifted.back()[i] += tideDeltas.back()[i];
    }
  }
  auto tideHistory = [&](const std::vector<std::vector<Complex>>& of) {
    return [&times, &of](double t) {
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) {
          return of[n];
        }
      }
      throw std::runtime_error("the test history is read off its nodes");
    };
  };
  // the unit-sphere pairing of two coefficient vectors of real fields
  auto pair = [&](const std::vector<Complex>& a, const std::vector<Complex>& b) {
    const auto fa = grid.Synthesise(a);
    const auto fb = grid.Synthesise(b);
    auto product = std::vector<double>(fa.size());
    for (auto i = std::size_t{0}; i < fa.size(); ++i) product[i] = fa[i] * fb[i];
    return grid.Integrate(product);
  };
  auto measurements = FullMeasurements(setup, rng, times.size() - 1);
  for (auto& [node, m] : measurements) {
    m.tangentialPotentialDual = setup.Smooth(rng, 0.6);
  }
  auto tangentialAlone = std::map<std::size_t, Measurement>{};
  tangentialAlone[times.size() - 1].tangentialPotentialDual =
      setup.Smooth(rng, 1.0);
  tangentialAlone[1].tangentialPotentialDual = setup.Smooth(rng, 1.0);
  const auto ditau = setup.DeltaItau(0.1);
  const auto dry = setup.solver.NoOceanState();

  const auto& material = setup.material;
  const auto& mesh = material.Mesh();
  const auto [k0, m0] = material.Moduli().KappaMu();
  MaterialDirection d;
  d.kappa.assign(k0.size(), 0.0);
  d.mu.assign(m0.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e)) continue;
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      const auto r = mesh.Radius(e, q);
      const auto profile = 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
      d.kappa[i] = 0.5 * profile * k0[i];
      d.mu[i] = profile * m0[i];
    }
  }

  const auto& wet = setup.state;
  const auto& all = measurements;
  const auto& alone = tangentialAlone;
  for (const auto* background : {&wet, &dry})
  for (const auto* measured : {&all, &alone})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label =
        std::string(background == &dry ? "no ocean" : "ocean") +
        (measured == &alone ? ", the tangential dual alone" : "") +
        ", scheme " + std::to_string(static_cast<int>(scheme));
    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(op, setup.solver, *background,
                        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    const auto snapshots = evolution.Run(
        times, Evolution::Forcing{.directLoad = History(times, loads),
                                  .tide = tideHistory(tides)});
    const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);
    const auto moved = evolution.Run(
        times, Evolution::Forcing{.directLoad = History(times, shifted),
                                  .tide = tideHistory(tidesShifted)});

    TangentLinearSolver tangent(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    TangentPerturbation forcing;
    forcing.directLoad = deltas;
    forcing.tide = tideDeltas;
    const auto alongForcing =
        Objective(setup, tangent.Run(times, forcing), *measured);
    const auto difference = Objective(setup, moved, *measured) -
                            Objective(setup, snapshots, *measured);
    EXPECT_NEAR(alongForcing, difference, 1e-8 * std::abs(difference)) << label;

    AdjointSolver adjoint(evolution, {.method = AdjointMethod::Discrete,
                                      .rtol = 1e-12,
                                      .material_kernels = true});
    const auto kernels =
        adjoint.FirstOrderKernels(times, *measured, &trajectory);
    ASSERT_EQ(kernels.tideKernels.size(), times.size());
    TangentPerturbation loadOnly, tideOnly;
    loadOnly.directLoad = deltas;
    tideOnly.tide = tideDeltas;
    const auto alongLoad =
        Objective(setup, tangent.Run(times, loadOnly), *measured);
    const auto alongTide =
        Objective(setup, tangent.Run(times, tideOnly), *measured);
    auto loadRhs = 0.0, tideRhs = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      loadRhs += setup.Surface(kernels.directLoadKernels[n], deltas[n]);
      tideRhs += pair(kernels.tideKernels[n], tideDeltas[n]);
    }
    EXPECT_NE(alongTide, 0.0);
    EXPECT_NEAR(alongLoad, loadRhs, 1e-8 * std::abs(alongLoad)) << label;
    EXPECT_NEAR(alongTide, tideRhs, 1e-8 * std::abs(alongTide)) << label;

    TangentPerturbation itauOnly;
    itauOnly.itau = ditau;
    const auto alongItau =
        Objective(setup, tangent.Run(times, itauOnly, &trajectory), *measured);
    const auto itauRhs =
        VolumePairing(setup.material.Mesh(), kernels.itauKernel[0], ditau[0]);
    EXPECT_NE(alongItau, 0.0);
    EXPECT_NEAR(alongItau, itauRhs, 1e-8 * std::abs(alongItau)) << label;

    TangentPerturbation moduli;
    moduli.material = AtMemoryPoints(op.Points(), d);
    const auto alongModuli =
        Objective(setup, tangent.Run(times, moduli, &trajectory), *measured);
    const auto moduliRhs =
        VolumePairing(op.Points(), mesh, kernels.kappaKernel,
                      moduli.material.kappa) +
        VolumePairing(op.Points(), mesh, kernels.muKernel, moduli.material.mu);
    EXPECT_NE(alongModuli, 0.0);
    EXPECT_NEAR(alongModuli, moduliRhs, 1e-8 * std::abs(alongModuli)) << label;

    // the sweeps that do not carry the tangential dual say so
    AdjointSolver continuous(evolution, {.method = AdjointMethod::Continuous});
    EXPECT_THROW((void)continuous.FirstOrderKernels(times, *measured),
                 std::invalid_argument);
  }
}

TEST(TangentLinear, ContinuousConvergesToDiscreteInItau) {
  // In the loads the two methods coincide; in the relaxation times they
  // differ by the order of the scheme: on a slowly relaxing mantle under
  // a load ramp, the gap in a sea level functional falls by about four
  // when the time steps are halved.
  auto setup = Scene::Make(1e22);
  std::mt19937 rng(5);
  const auto field = setup.Smooth(rng, 100.0 * setup.Sigma());
  const auto g = setup.Smooth(rng, 1.0);
  const auto span = 10.0 * setup.Kyr();
  const auto ramp = [&](double t) {
    auto out = field;
    for (auto& v : out) v *= t / span;
    return out;
  };
  const auto ditau = setup.DeltaItau(0.1);
  ViscoelasticOperator op(setup.rheology, kLmax);
  auto gaps = std::vector<double>{};
  for (const auto nodes : {5, 9}) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
    Evolution evolution(op, setup.solver, setup.state,
                                {.rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory::FromSnapshots(evolution.Run(times, ramp));
    auto measurement = std::map<std::size_t, Measurement>{};
    measurement[times.size() - 1].seaLevelDual = g;
    TangentPerturbation direction;
    direction.itau = ditau;
    auto values = std::map<AdjointMethod, double>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver tangent(evolution, {.method = method, .rtol = 1e-12});
      values[method] = Objective(
          setup, tangent.Run(times, direction, &trajectory), measurement);
    }
    gaps.push_back(std::abs(values[AdjointMethod::Discrete] -
                            values[AdjointMethod::Continuous]) /
                   std::abs(values[AdjointMethod::Discrete]));
  }
  EXPECT_LT(gaps[0], 0.2);
  EXPECT_GT(gaps[0] / gaps[1], 3.0);
}


TEST(TangentLinear, AboutAMigratingTrajectory) {
  // About a migrating run the tangent-linear run along a thickness and
  // an initial sea level perturbation that flip no flotation cell matches
  // central differences of migrating runs — the map is piecewise affine,
  // so the difference quotient is exact on the piece — for both methods
  // (which coincide in these arguments), and the kernels' pairing agrees.
  auto setup = Scene::Make(1e21);
  ViscoelasticOperator op(setup.rheology, kLmax);
  const auto cap = MarineCap::Make(setup, 5);
  const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
  const auto thickness = [&](double t) { return cap.Thickness(t); };
  Evolution evolution(op, setup.solver, state, {.rtol = 1e-12});
  const auto run = evolution.RunMigrating(cap.times, thickness);
  const auto oceans = OceanFunctionsOf(run);
  std::mt19937 rng(3);
  const auto g = setup.Smooth(rng, 1.0);

  auto deltaIce = std::vector<std::vector<double>>(cap.times.size());
  for (auto n = std::size_t{1}; n < cap.times.size(); ++n) {
    deltaIce[n] = setup.Smooth(rng, 1.0 * setup.Metre());
  }
  deltaIce[0].assign(setup.solver.Grid().FieldSize(), 0.0);
  const auto deltaSea = setup.Smooth(rng, 1.0 * setup.Metre());

  auto functional = [&](double sign) {
    auto sea = state.seaLevel;
    for (auto i = std::size_t{0}; i < sea.size(); ++i) {
      sea[i] += sign * deltaSea[i];
    }
    const auto start = setup.solver.MakeState(state.iceThickness, sea);
    EXPECT_EQ(start.oceanFunction, state.oceanFunction) << "the start flipped";
    Evolution trial(op, setup.solver, start, {.rtol = 1e-12});
    const auto snaps = trial.RunMigrating(cap.times, [&, sign](double t) {
      auto out = cap.Thickness(t);
      for (auto n = std::size_t{0}; n < cap.times.size(); ++n) {
        if (t != cap.times[n]) continue;
        for (auto i = std::size_t{0}; i < out.size(); ++i) {
          out[i] += sign * deltaIce[n][i];
        }
      }
      return out;
    });
    for (auto n = std::size_t{0}; n < snaps.size(); ++n) {
      EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped";
    }
    return setup.Surface(g, snaps.back().relativeSeaLevelChange);
  };
  const auto fd = 0.5 * (functional(1.0) - functional(-1.0));

  auto measurement = std::map<std::size_t, Measurement>{};
  measurement[cap.times.size() - 1].seaLevelDual = g;
  TangentPerturbation direction;
  direction.iceThickness = deltaIce;
  direction.initialSeaLevel = deltaSea;
  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    TangentLinearSolver tangent(evolution, {.method = method, .rtol = 1e-12});
    const auto snaps = tangent.Run(cap.times, direction, nullptr, oceans);
    EXPECT_NEAR(Objective(setup, snaps, measurement), fd, 1e-7 * std::abs(fd))
        << "method " << static_cast<int>(method);
    for (auto n = std::size_t{0}; n < snaps.size(); ++n) {
      EXPECT_EQ(snaps[n].oceanFunction, oceans[n]);
    }
    EXPECT_EQ(MaxAbs(snaps[0].relativeSeaLevelChange), 0.0);
  }

  // The discrete kernels' pairing: the thickness view (grounded by each
  // node's shorelines) and the initial sea level kernel.
  const auto kernels =
      AdjointSolver(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
          .FirstOrderKernels(cap.times, measurement, nullptr, oceans);
  auto paired = setup.Surface(kernels.initialSeaLevel, deltaSea);
  for (auto n = std::size_t{1}; n < cap.times.size(); ++n) {
    paired += setup.Surface(kernels.thickness[n], deltaIce[n]);
  }
  EXPECT_NEAR(paired, fd, 1e-7 * std::abs(fd));

  // Under frozen shorelines the initial sea level contributes nothing.
  Evolution frozen(op, setup.solver, state, {.rtol = 1e-12});
  TangentPerturbation seaOnly;
  seaOnly.initialSeaLevel = deltaSea;
  const auto still =
      TangentLinearSolver(frozen, {.rtol = 1e-12}).Run(cap.times, seaOnly);
  for (const auto& snap : still) {
    EXPECT_EQ(MaxAbs(snap.relativeSeaLevelChange), 0.0);
  }
}

// -- second order -----------------------------------------------------------

TEST(TangentLinear, SecondOrderRunIsTheDerivativeOfTheFirst) {
  // The second-order run along (u, v) is the derivative, at the base
  // point along u, of the first-order run along v: central differences of
  // the latter — the loads moved by eps delta_u and the rheology by eps
  // delta itau_u — converge to it at second order and land on it after
  // extrapolation, for both schemes; and it is symmetric in u and v.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(17);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * setup.Kyr());
  auto loads = std::vector<std::vector<double>>{};
  auto deltaU = std::vector<std::vector<double>>{};
  auto deltaV = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    deltaU.push_back(setup.Smooth(rng, 20.0 * setup.Sigma()));
    deltaV.push_back(setup.Smooth(rng, 20.0 * setup.Sigma()));
  }
  const auto g = setup.Smooth(rng, 1.0);
  auto measurement = std::map<std::size_t, Measurement>{};
  measurement[times.size() - 1].seaLevelDual = g;
  const auto ditauU = setup.DeltaItau(0.1);
  const auto ditauV =
      setup.DeltaItau(0.1, [](double r) { return 0.8 - 0.5 * r; });
  const auto& base = setup.rheology.BranchAt(0);
  auto shifted = [&](const std::vector<std::vector<double>>& loads,
                     double eps) {
    auto out = loads;
    for (auto n = std::size_t{0}; n < out.size(); ++n) {
      for (auto i = std::size_t{0}; i < out[n].size(); ++i) {
        out[n][i] += eps * deltaU[n][i];
      }
    }
    return out;
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(
        op, setup.solver, setup.state,
        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, History(times, loads)));
    TangentLinearSolver tangent(
        evolution,
        {.method = AdjointMethod::Discrete, .rtol = 1e-12, .record = true});
    TangentPerturbation u, v;
    u.directLoad = deltaU;
    u.itau = ditauU;
    v.directLoad = deltaV;
    v.itau = ditauV;
    const auto tangentU = tangent.Run(times, u, &trajectory);
    const auto tangentV = tangent.Run(times, v, &trajectory);
    const auto second = tangent.RunSecondOrder(
        times, tangentU, tangentV, {.itauU = ditauU, .itauV = ditauV},
        &trajectory);
    const auto predicted = Objective(setup, second, measurement);

    auto firstOrderAt = [&](double eps) {
      auto branch = base;
      for (auto i = std::size_t{0}; i < branch.itau.size(); ++i) {
        branch.itau[i] += eps * ditauU[0][i];
      }
      auto rheology =
          Rheology::Create(setup.material, {std::move(branch)}).value();
      ViscoelasticOperator moved(rheology, kLmax);
      Evolution movedEvolution(
          moved, setup.solver, setup.state,
          {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
      const auto movedLoads = shifted(loads, eps);
      const auto movedTrajectory = ForwardTrajectory::FromSnapshots(
          movedEvolution.Run(times, History(times, movedLoads)));
      TangentLinearSolver along(
          movedEvolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
      return Objective(setup, along.Run(times, v, &movedTrajectory),
                       measurement);
    };
    auto fd = std::map<double, double>{};
    for (double eps : {0.02, 0.01}) {
      fd[eps] = (firstOrderAt(eps) - firstOrderAt(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.02] - predicted);
    const auto r2 = std::abs(fd[0.01] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme);
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0;
    EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme);

    const auto swapped = tangent.RunSecondOrder(
        times, tangentV, tangentU, {.itauU = ditauV, .itauV = ditauU},
        &trajectory);
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      EXPECT_LT(MaxDiff(swapped[n].relativeSeaLevelChange, second[n].relativeSeaLevelChange),
                1e-10 * MaxAbs(second[n].relativeSeaLevelChange) + 1e-300)
          << "node " << n;
      EXPECT_LT(MaxDiff(swapped[n].potential, second[n].potential),
                1e-10 * MaxAbs(second[n].potential) + 1e-300);
    }

    // Refusals: unrecorded tangent runs.
    TangentLinearSolver bare(evolution,
                             {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto light = bare.Run(times, u, &trajectory);
    EXPECT_THROW((void)tangent.RunSecondOrder(
                     times, light, tangentV,
                     {.itauU = ditauU, .itauV = ditauV}, &trajectory),
                 std::invalid_argument);
  }
}

TEST(TangentLinear, SecondOrderRunVanishesWithoutAViscosityPerturbation) {
  // With the ocean functions held the map is affine in the loads and the
  // initial sea level, about frozen and migrating runs alike.
  auto setup = Scene::Make(1e21);
  ViscoelasticOperator op(setup.rheology, kLmax);
  std::mt19937 rng(23);
  // frozen
  {
    auto times = std::vector<double>{};
    for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * setup.Kyr());
    auto loads = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    }
    Evolution evolution(op, setup.solver, setup.state,
                                {.rtol = 1e-12, .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, History(times, loads)));
    TangentLinearSolver tangent(evolution, {.rtol = 1e-12, .record = true});
    auto runs = std::vector<std::vector<Evolution::Snapshot>>{};
    for (auto seed = 0; seed < 2; ++seed) {
      TangentPerturbation d;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        d.directLoad.push_back(setup.Smooth(rng, 20.0 * setup.Sigma()));
      }
      runs.push_back(tangent.Run(times, d));
    }
    const auto second =
        tangent.RunSecondOrder(times, runs[0], runs[1], {}, &trajectory);
    for (const auto& snap : second) {
      EXPECT_EQ(MaxAbs(snap.relativeSeaLevelChange), 0.0);
      EXPECT_EQ(MaxAbs(snap.displacement), 0.0);
      EXPECT_EQ(MaxAbs(snap.potential), 0.0);
    }
  }
  // migrating
  {
    const auto cap = MarineCap::Make(setup, 5);
    const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
    Evolution evolution(op, setup.solver, state,
                                {.rtol = 1e-12, .record_stride = 1});
    const auto run = evolution.RunMigrating(
        cap.times, [&](double t) { return cap.Thickness(t); });
    const auto oceans = OceanFunctionsOf(run);
    const auto trajectory = ForwardTrajectory::FromSnapshots(run);
    TangentLinearSolver tangent(evolution, {.rtol = 1e-12, .record = true});
    auto runs = std::vector<std::vector<Evolution::Snapshot>>{};
    for (auto seed = 0; seed < 2; ++seed) {
      TangentPerturbation d;
      for (auto n = std::size_t{0}; n < cap.times.size(); ++n) {
        d.iceThickness.push_back(setup.Smooth(rng, 1.0 * setup.Metre()));
      }
      d.initialSeaLevel = setup.Smooth(rng, 1.0 * setup.Metre());
      runs.push_back(tangent.Run(cap.times, d, nullptr, oceans));
    }
    const auto second = tangent.RunSecondOrder(cap.times, runs[0], runs[1],
                                               {}, &trajectory, oceans);
    for (auto n = std::size_t{0}; n < second.size(); ++n) {
      EXPECT_EQ(MaxAbs(second[n].relativeSeaLevelChange), 0.0);
      EXPECT_EQ(second[n].oceanFunction, oceans[n]);
    }
  }
}

TEST(TangentLinear, ContinuousSecondOrderConvergesToDiscrete) {
  // By the continuous method the second-order run converges to the
  // discrete one at the scheme's order as the steps are refined. On this
  // mantle (tau about 2 kyr) the grids coarser than 17 nodes over 10 kyr
  // have h = dt / tau near one and sit before the asymptotic regime (the
  // gap is 2.9 at 5 nodes, 0.5 at 9); from 17 nodes it falls by four.
  auto setup = Scene::Make(1e22);
  std::mt19937 rng(29);
  const auto field = setup.Smooth(rng, 100.0 * setup.Sigma());
  const auto g = setup.Smooth(rng, 1.0);
  const auto span = 10.0 * setup.Kyr();
  const auto ramp = [&](double t) {
    auto out = field;
    for (auto& v : out) v *= t / span;
    return out;
  };
  const auto ditauU = setup.DeltaItau(0.1);
  const auto ditauV =
      setup.DeltaItau(0.1, [](double r) { return 0.8 - 0.5 * r; });
  ViscoelasticOperator op(setup.rheology, kLmax);
  auto gaps = std::vector<double>{};
  for (const auto nodes : {17, 33}) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
    Evolution evolution(op, setup.solver, setup.state,
                                {.rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory::FromSnapshots(evolution.Run(times, ramp));
    auto measurement = std::map<std::size_t, Measurement>{};
    measurement[times.size() - 1].seaLevelDual = g;
    auto values = std::map<AdjointMethod, double>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver tangent(
          evolution, {.method = method, .rtol = 1e-12, .record = true});
      TangentPerturbation u, v;
      u.itau = ditauU;
      v.itau = ditauV;
      const auto tangentU = tangent.Run(times, u, &trajectory);
      const auto tangentV = tangent.Run(times, v, &trajectory);
      values[method] = Objective(
          setup,
          tangent.RunSecondOrder(times, tangentU, tangentV,
                                 {.itauU = ditauU, .itauV = ditauV},
                                 &trajectory),
          measurement);
    }
    gaps.push_back(std::abs(values[AdjointMethod::Discrete] -
                            values[AdjointMethod::Continuous]) /
                   std::abs(values[AdjointMethod::Discrete]));
  }
  EXPECT_LT(gaps[0], 0.2);
  EXPECT_GT(gaps[0] / gaps[1], 3.0);
}

// -- the material ------------------------------------------------------------

TEST(TangentLinear, MaterialDirectionMatchesDifferences) {
  // Along a direction in the material's kappa and mu — the branch moduli
  // as fixed fractions of the unrelaxed one, the relaxation times held —
  // the tangent-linear run matches central differences of the coupled run
  // on the shifted material (its rheology rebuilt with the same fractions
  // and times, the sea level engine's gravity held), at second order with
  // Richardson landing at 1e-6, both schemes; the start's instantaneous
  // response moves with the material too (the trapezoid's first step
  // reads its strain).
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(31);
  // The engine's body held at the base table's.
  const auto baseTable = ViscoelasticOperator(setup.rheology, kLmax).ElasticLoveNumbers();
  SeaLevelParameters held;
  held.body_from_table = false;
  held.mean_sea_floor_radius = baseTable.radius;
  held.gravitational_acceleration = baseTable.surface_gravity;
  const GRDSolver solver(SurfaceGrid(kLmax), baseTable, held);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * setup.Kyr());
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = setup.Smooth(rng, 1.0);
  measurements[times.size() - 1].displacementDual = setup.Smooth(rng, 0.5);
  measurements[2].potentialDual = setup.Smooth(rng, 0.7);

  // A smooth direction in kappa and mu on the solid mantle and lithosphere
  // (not the fluid core; the centre element kept to keep the comparison
  // honest in 1D).
  const auto& material = setup.material;
  const auto& mesh = material.Mesh();
  const auto [k0, m0] = material.Moduli().KappaMu();
  MaterialDirection d;
  d.kappa.assign(k0.size(), 0.0);
  d.mu.assign(m0.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      const auto r = mesh.Radius(e, q);
      const auto profile = 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
      if (material.IsFluid(e)) continue;
      d.kappa[i] = 0.5 * profile * k0[i];
      d.mu[i] = profile * m0[i];
    }
  }
  // The rheology of a shifted material: the branch moduli as the same
  // fractions of the (shifted) unrelaxed modulus, the inverse times held.
  const auto& base = setup.rheology;
  auto rheologyAt = [&](double eps) {
    const auto shifted = material.Shifted(d, eps);
    const auto muU = base.MuUnrelaxed();
    const auto [kappa, mu] = shifted.Moduli().KappaMu();
    auto branches = std::vector<Rheology::Branch>{};
    for (auto k = std::size_t{0}; k < base.BranchCount(); ++k) {
      auto branch = base.BranchAt(k);
      for (auto i = std::size_t{0}; i < branch.mu.size(); ++i) {
        branch.mu[i] = muU[i] > 0.0 ? branch.mu[i] / muU[i] * mu[i] : 0.0;
      }
      branches.push_back(std::move(branch));
    }
    return std::pair(shifted, Rheology::Create(shifted, branches).value());
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax,
                            {.memory_points = MemoryPointSet::Gll});
    Evolution evolution(op, solver, setup.state,
                                {.scheme = scheme,
                                 .rotation = RotationalFeedback::None,
                                 .rtol = 1e-12,
                                 .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, History(times, loads)));
    TangentLinearSolver tangent(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    TangentPerturbation direction;
    direction.material = AtGll(d);
    const auto predicted =
        Objective(setup, tangent.Run(times, direction, &trajectory), measurements);

    auto functional = [&](double eps) {
      const auto [shifted, rheology] = rheologyAt(eps);
      ViscoelasticOperator moved(rheology, kLmax,
                                 {.memory_points = MemoryPointSet::Gll});
      Evolution movedEvolution(moved, solver, setup.state,
                                       {.scheme = scheme,
                                        .rotation = RotationalFeedback::None,
                                        .rtol = 1e-12});
      return Objective(setup, movedEvolution.Run(times, History(times, loads)),
                       measurements);
    };
    // The steps are wide (1% and 0.5% of the moduli) so the differences'
    // truncation error stands above the solves' 1e-12 noise: the
    // functional moves by parts in 1e-12 per unit step.
    auto fd = std::map<double, double>{};
    for (double eps : {0.1, 0.05}) {
      fd[eps] = (functional(eps) - functional(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.1] - predicted);
    const auto r2 = std::abs(fd[0.05] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme) << " "
                            << fd[0.1] << " " << fd[0.05] << " " << predicted;
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.05] - fd[0.1]) / 3.0;
    EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme);
  }
}


TEST(TangentLinear, MaterialDirectionMatchesDifferencesAtGaussPoints) {
  // The same at the default memory points: along a direction in kappa and
  // mu held at the Gauss points, the tangent-linear run matches central
  // differences of the coupled run with the moduli perturbed at the
  // points (ViscoelasticOperator::Options::moduli, the relative
  // perturbation eps * direction / modulus), at second order with
  // Richardson landing at 1e-6, both schemes, with and without the
  // rotational feedback.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(31);
  const auto baseTable = ViscoelasticOperator(setup.rheology, kLmax).ElasticLoveNumbers();
  SeaLevelParameters held;
  held.body_from_table = false;
  held.mean_sea_floor_radius = baseTable.radius;
  held.gravitational_acceleration = baseTable.surface_gravity;
  const GRDSolver solver(SurfaceGrid(kLmax), baseTable, held);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * setup.Kyr());
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = setup.Smooth(rng, 1.0);
  measurements[times.size() - 1].displacementDual = setup.Smooth(rng, 0.5);
  measurements[2].potentialDual = setup.Smooth(rng, 0.7);

  const auto& material = setup.material;
  const auto& mesh = material.Mesh();
  const ViscoelasticOperator reference(setup.rheology, kLmax);
  ASSERT_EQ(reference.Points().set, MemoryPointSet::Gauss);
  const auto& points = reference.Points();
  const auto muU = reference.UnrelaxedShearAtPoints();
  const auto kappa0 = reference.BulkModulusAtPoints();
  PointMaterialDirection d;
  d.kappa.assign(muU.size(), 0.0);
  d.mu.assign(muU.size(), 0.0);
  auto profile = [](double r) {
    return 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
  };
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e)) continue;
    for (auto p = 0; p < points.count; ++p) {
      const auto i = static_cast<std::size_t>(e) * points.count + p;
      const auto value = profile(points.RadiusAt(mesh, e, p));
      d.kappa[i] = 0.5 * value * kappa0[i];
      d.mu[i] = value * muU[i];
    }
  }
  // The stepper at eps along the direction: the moduli perturbed at the
  // points, relative, the branches kept.
  const auto& base = setup.rheology;
  auto operatorAt = [&](double eps) {
    auto branches = std::vector<Rheology::Branch>{};
    for (auto k = std::size_t{0}; k < base.BranchCount(); ++k) {
      branches.push_back(base.BranchAt(k));
    }
    const auto rheology = Rheology::Create(material, branches).value();
    ViscoelasticOperator::Options::PointModuli moduli;
    moduli.shear.assign(muU.size(), 0.0);
    moduli.bulk.assign(muU.size(), 0.0);
    for (auto i = std::size_t{0}; i < muU.size(); ++i) {
      if (muU[i] > 0.0) moduli.shear[i] = eps * d.mu[i] / muU[i];
      if (kappa0[i] > 0.0) moduli.bulk[i] = eps * d.kappa[i] / kappa0[i];
    }
    return ViscoelasticOperator(rheology, kLmax, {.moduli = moduli});
  };

  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Axial,
                        RotationalFeedback::Full})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(op, solver, setup.state,
                                {.scheme = scheme,
                                 .rotation = rotation,
                                 .rtol = 1e-12,
                                 .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, History(times, loads)));
    TangentLinearSolver tangent(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    TangentPerturbation direction;
    direction.material = d;
    const auto predicted =
        Objective(setup, tangent.Run(times, direction, &trajectory), measurements);

    auto functional = [&](double eps) {
      const auto moved = operatorAt(eps);
      Evolution movedEvolution(moved, solver, setup.state,
                                       {.scheme = scheme,
                                        .rotation = rotation,
                                        .rtol = 1e-12});
      return Objective(setup, movedEvolution.Run(times, History(times, loads)),
                       measurements);
    };
    auto fd = std::map<double, double>{};
    for (double eps : {0.1, 0.05}) {
      fd[eps] = (functional(eps) - functional(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.1] - predicted);
    const auto r2 = std::abs(fd[0.05] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme)
                            << " rotation " << static_cast<int>(rotation) << " "
                            << fd[0.1] << " " << fd[0.05] << " " << predicted;
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.05] - fd[0.1]) / 3.0;
    EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme) << " rotation "
        << static_cast<int>(rotation);
    if (rotation != RotationalFeedback::None) continue;

    // Each part of the direction alone, to the same standard: the two
    // enter by different routes.
    for (const auto* part : {"kappa", "mu"}) {
      TangentPerturbation only;
      if (part == std::string("kappa")) only.material.kappa = d.kappa;
      if (part == std::string("mu")) only.material.mu = d.mu;
      EXPECT_NE(Objective(setup, tangent.Run(times, only, &trajectory), measurements),
                0.0)
          << part;
    }
  }
  // A direction of the wrong size is refused.
  ViscoelasticOperator op(setup.rheology, kLmax);
  Evolution evolution(op, solver, setup.state,
                              {.rotation = RotationalFeedback::None,
                               .record_stride = 1});
  const auto trajectory = ForwardTrajectory::FromSnapshots(
      evolution.Run(times, History(times, loads)));
  TangentLinearSolver tangent(evolution, {.method = AdjointMethod::Discrete});
  TangentPerturbation bad;
  bad.material.mu.assign(3, 1.0);
  EXPECT_THROW((void)tangent.Run(times, bad, &trajectory), std::invalid_argument);
}

TEST(TangentLinear, SecondOrderRunInTheMaterial) {
  // With directions u and v in the loads, the inverse relaxation times
  // AND the moduli, the second-order run is the derivative at the base
  // point along u of the first-order run along v: central differences of
  // the latter — the loads moved by eps delta_u, the rheology by eps
  // delta itau_u and the material by eps delta m_u (the branch moduli as
  // the same fractions of the shifted unrelaxed modulus) — converge to it
  // at second order and Richardson lands on it, both schemes; and it is
  // symmetric in u and v. Every mixed block is exercised: moduli–moduli,
  // moduli–loads, moduli–relaxation times.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(41);
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * setup.Kyr());
  auto loads = std::vector<std::vector<double>>{};
  auto deltaU = std::vector<std::vector<double>>{};
  auto deltaV = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    deltaU.push_back(setup.Smooth(rng, 20.0 * setup.Sigma()));
    deltaV.push_back(setup.Smooth(rng, 20.0 * setup.Sigma()));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = setup.Smooth(rng, 1.0);
  measurements[times.size() - 1].displacementDual = setup.Smooth(rng, 0.5);
  measurements[2].potentialDual = setup.Smooth(rng, 0.7);
  const auto ditauU = setup.DeltaItau(0.1);
  const auto ditauV =
      setup.DeltaItau(0.1, [](double r) { return 0.8 - 0.5 * r; });

  // Smooth directions in kappa and mu on the solid, at 1% of the moduli.
  const auto& material = setup.material;
  const auto& mesh = material.Mesh();
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  auto materialDirection = [&](const std::function<double(double)>& profile) {
    MaterialDirection d;
    d.kappa.assign(k0.size(), 0.0);
    d.mu.assign(m0.size(), 0.0);
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      if (material.IsFluid(e)) continue;
      for (auto q = 0; q < mesh.NGll(); ++q) {
        const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
        const auto p = 0.1 * profile(mesh.Radius(e, q));
        d.kappa[i] = 0.5 * p * k0[i];
        d.mu[i] = p * m0[i];
      }
    }
    return d;
  };
  const auto dU = materialDirection(
      [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); });
  const auto dV = materialDirection([](double r) { return 1.0 - 0.5 * r; });

  // The base moved along u by eps: the material shifted, the branch
  // moduli as the same fractions of the shifted unrelaxed modulus, the
  // inverse times moved by eps delta itau_u.
  const auto& base = setup.rheology;
  auto rheologyAt = [&](double eps) {
    const auto shifted = material.Shifted(dU, eps);
    const auto muU = base.MuUnrelaxed();
    const auto [kappa, mu] = shifted.Moduli().KappaMu();
    auto branches = std::vector<Rheology::Branch>{};
    for (auto k = std::size_t{0}; k < base.BranchCount(); ++k) {
      auto branch = base.BranchAt(k);
      for (auto i = std::size_t{0}; i < branch.mu.size(); ++i) {
        branch.mu[i] = muU[i] > 0.0 ? branch.mu[i] / muU[i] * mu[i] : 0.0;
        branch.itau[i] += eps * ditauU[k][i];
      }
      branches.push_back(std::move(branch));
    }
    return Rheology::Create(shifted, branches).value();
  };
  auto shiftedLoads = [&](double eps) {
    auto out = loads;
    for (auto n = std::size_t{0}; n < out.size(); ++n) {
      for (auto i = std::size_t{0}; i < out[n].size(); ++i) {
        out[n][i] += eps * deltaU[n][i];
      }
    }
    return out;
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax,
                            {.memory_points = MemoryPointSet::Gll});
    Evolution evolution(op, setup.solver, setup.state,
                                {.scheme = scheme,
                                 .rotation = RotationalFeedback::None,
                                 .rtol = 1e-12,
                                 .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, History(times, loads)));
    TangentLinearSolver tangent(
        evolution,
        {.method = AdjointMethod::Discrete, .rtol = 1e-12, .record = true});
    TangentPerturbation u, v;
    u.directLoad = deltaU;
    u.itau = ditauU;
    u.material = AtGll(dU);
    v.directLoad = deltaV;
    v.itau = ditauV;
    v.material = AtGll(dV);
    const auto tangentU = tangent.Run(times, u, &trajectory);
    const auto tangentV = tangent.Run(times, v, &trajectory);
    SecondOrderDirection uv;
    uv.itauU = ditauU;
    uv.itauV = ditauV;
    uv.materialU = AtGll(dU);
    uv.materialV = AtGll(dV);
    const auto second =
        tangent.RunSecondOrder(times, tangentU, tangentV, uv, &trajectory);
    const auto predicted = Objective(setup, second, measurements);

    auto firstOrderAt = [&](double eps) {
      const auto rheology = rheologyAt(eps);
      ViscoelasticOperator moved(rheology, kLmax,
                                 {.memory_points = MemoryPointSet::Gll});
      Evolution movedEvolution(moved, setup.solver, setup.state,
                                       {.scheme = scheme,
                                        .rotation = RotationalFeedback::None,
                                        .rtol = 1e-12,
                                        .record_stride = 1});
      const auto movedTrajectory = ForwardTrajectory::FromSnapshots(
          movedEvolution.Run(times, History(times, shiftedLoads(eps))));
      TangentLinearSolver along(
          movedEvolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
      return Objective(setup, along.Run(times, v, &movedTrajectory),
                       measurements);
    };
    auto fd = std::map<double, double>{};
    for (double eps : {0.1, 0.05}) {
      fd[eps] = (firstOrderAt(eps) - firstOrderAt(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.1] - predicted);
    const auto r2 = std::abs(fd[0.05] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme) << " "
                            << fd[0.1] << " " << fd[0.05] << " " << predicted;
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.05] - fd[0.1]) / 3.0;
    EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme);

    SecondOrderDirection vu;
    vu.itauU = ditauV;
    vu.itauV = ditauU;
    vu.materialU = AtGll(dV);
    vu.materialV = AtGll(dU);
    const auto swapped =
        tangent.RunSecondOrder(times, tangentV, tangentU, vu, &trajectory);
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      EXPECT_LT(MaxDiff(swapped[n].relativeSeaLevelChange, second[n].relativeSeaLevelChange),
                1e-10 * MaxAbs(second[n].relativeSeaLevelChange) + 1e-300)
          << "node " << n;
    }

    // The material alone in one direction and the loads alone in the
    // other is the moduli–loads block, not zero.
    TangentPerturbation loadsOnly, materialOnly;
    loadsOnly.directLoad = deltaV;
    materialOnly.material = AtGll(dU);
    const auto runLoads = tangent.Run(times, loadsOnly, &trajectory);
    const auto runMaterial = tangent.Run(times, materialOnly, &trajectory);
    SecondOrderDirection cross;
    cross.materialU = AtGll(dU);
    const auto mixed =
        tangent.RunSecondOrder(times, runMaterial, runLoads, cross, &trajectory);
    EXPECT_GT(std::abs(Objective(setup, mixed, measurements)), 0.0);
  }
}
