// The coupled GIA evolution: the elastic-rheology limit against the
// fingerprint solver (machine-tight, fully self-contained), qualitative
// relaxation, and the coupled comparison with pyslfp; Sdirk2's second
// order and limit against the trapezoid's, and its damping after an
// abrupt unloading on a weak layer and in the fluid limit.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/ShConventions.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 16;

struct Setup {
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  GRDSolver::State state;
  std::vector<double> iceNd;  // the state's ice thickness, solver units

  static Setup Make(bool viscous) {
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

    Rheology rheology = [&] {
      if (!viscous) {
        // A branch that never relaxes: the elastic limit of the coupling.
        Rheology::Branch zero;
        zero.mu.assign(material.Rho().size(), 0.0);
        zero.itau.assign(material.Rho().size(), 0.0);
        return Rheology::Create(material, {std::move(zero)}).value();
      }
      auto viscosity = std::map<std::size_t, RadialValue>{};
      const auto b = model.DimensionlessView().Boundaries();
      for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
        if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
            b[i + 1] <= 6.1511e6 / scales.Length()) {
          viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
        }
      }
      return Rheology::Maxwell(material, viscosity).value();
    }();

    ViscoelasticOperator op(rheology, kLmax);
    GRDSolver solver(SurfaceGrid(kLmax), op.ElasticLoveNumbers(),
                          SeaLevelParameters{});

    // The analytic test state, built through the fingerprint helper and
    // rebuilt in the solver's units.
    Fingerprint fingerprint(kLmax, op.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto ice = toNd(fingerprint.IceThickness());
    auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
    return Setup{std::move(model),  std::move(material), std::move(rheology),
                 std::move(solver), std::move(state),    std::move(ice)};
  }

  /// The grounded load of a melt fraction, solver units.
  [[nodiscard]] std::vector<double> MeltLoad(double fraction) const {
    auto load = std::vector<double>(iceNd.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                fraction * iceNd[i];
    }
    return load;
  }
};

}  // namespace

TEST(Evolution, ElasticRheologyReducesToTheFingerprint) {
  auto setup = Setup::Make(false);
  ViscoelasticOperator op(setup.rheology, kLmax);
  Evolution evolution(op, setup.solver, setup.state, {.rtol = 1e-12});

  const auto tauScale = 1e10 / setup.solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, tauScale, 3.0 * tauScale};
  auto load = setup.MeltLoad(-0.1);
  auto snapshots = evolution.Run(times, [&](double) { return load; });
  ASSERT_EQ(snapshots.size(), 3u);

  // With nothing relaxing, every node is the same elastic solve.
  auto reference = setup.solver.Solve(
      setup.state, load, RotationalFeedback::Full, {.rtol = 1e-12});
  auto scale = 0.0;
  for (auto v : reference.relativeSeaLevelChange) scale = std::max(scale, std::abs(v));
  for (const auto& snapshot : snapshots) {
    auto diff = 0.0;
    for (auto i = std::size_t{0}; i < reference.relativeSeaLevelChange.size(); ++i) {
      diff = std::max(diff, std::abs(snapshot.relativeSeaLevelChange[i] -
                                     reference.relativeSeaLevelChange[i]));
    }
    EXPECT_LT(diff, 1e-9 * scale) << "t = " << snapshot.t;
    for (auto i = 0; i < 3; ++i) {
      EXPECT_NEAR(snapshot.angularVelocity[i], reference.angularVelocity[i],
                  1e-9 * std::abs(reference.angularVelocity[1]) + 1e-30);
    }
  }
}

TEST(Evolution, ABodyWithoutRelaxationBranchesIsElastic) {
  // A rheology with no relaxation branch is an elastic body, whose
  // internal variables are empty. Its run of one node under a load and a
  // point source, recorded, is the start of the viscoelastic body's (the
  // same unrelaxed moduli); and stepped under a forcing held, it does not
  // relax: every node is the first.
  auto setup = Setup::Make(true);
  const auto elastic = Rheology::Create(setup.material, {});
  ASSERT_TRUE(elastic);
  ViscoelasticOperator relaxing(setup.rheology, kLmax);
  ViscoelasticOperator op(*elastic, kLmax);
  const auto& scales = setup.solver.SolverScales();
  const auto unit =
      1e21 / scales.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto load = setup.MeltLoad(-0.1);
  const auto forcing = Evolution::Forcing{
      .directLoad = [&](double) { return load; },
      .sources = [&](double) {
        return std::vector<PointSource>{
            {.at = {.radius = setup.solver.Radius() - 400e3 / scales.Length(),
                    .colatitude = 1.1,
                    .longitude = 4.0},
             .moment = {.rr = 0.9 * unit, .tt = -0.2 * unit, .pp = -0.7 * unit,
                        .rt = 0.3 * unit, .rp = -0.7 * unit, .tp = 0.6 * unit}}};
      }};
  auto worst = [](std::span<const double> x, std::span<const double> y) {
    auto out = 0.0, top = 0.0;
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      out = std::max(out, std::abs(x[i] - y[i]));
      top = std::max(top, std::abs(y[i]));
    }
    return out / top;
  };
  const Evolution::Options options{.rtol = 1e-12, .record_stride = 1};
  Evolution evolution(op, setup.solver, setup.state, options);
  Evolution reference(relaxing, setup.solver, setup.state, options);
  const auto one = std::vector<double>{0.0};
  const auto alone = evolution.Run(one, forcing);
  const auto start = reference.Run(one, forcing);
  ASSERT_EQ(alone.size(), 1u);
  ASSERT_FALSE(alone[0].toroidalPotential.empty());
  for (auto field : {&Evolution::Snapshot::relativeSeaLevelChange,
                     &Evolution::Snapshot::displacement,
                     &Evolution::Snapshot::potential,
                     &Evolution::Snapshot::tangentialPotential,
                     &Evolution::Snapshot::toroidalPotential}) {
    EXPECT_LT(worst(alone[0].*field, start[0].*field), 1e-10);
  }
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  const auto held = evolution.Run(std::vector<double>{0.0, kyr, 5.0 * kyr}, forcing);
  ASSERT_EQ(held.size(), 3u);
  for (auto n = std::size_t{1}; n < held.size(); ++n) {
    EXPECT_LT(worst(held[n].displacement, held[0].displacement), 1e-9)
        << "node " << n;
    EXPECT_LT(worst(held[n].toroidalPotential, held[0].toroidalPotential), 1e-9)
        << "node " << n;
  }
}

TEST(Evolution, ARunOfOneNodeIsHeldToNoFluidLimit) {
  // The check of the transverse rotational feedback against the model's
  // hydrostatic fluid limit guards the steps, whose effective tidal
  // number long steps drive towards that limit. A run of one node takes
  // no step: moments of inertia that sustain less oblateness than the
  // fluid limit, but more than the elastic response, are taken there, and
  // the run is the elastic sea level solve under the full feedback, by
  // either entry and frozen or migrating; a run that steps is refused.
  auto setup = Setup::Make(true);
  ViscoelasticOperator op(setup.rheology, kLmax);
  const auto kHyd = FluidTidalNumber(setup.material, 2);
  const GRDSolver solver(
      SurfaceGrid(kLmax), op.ElasticLoveNumbers(),
      WithHydrostaticMoments(SeaLevelParameters{}, kHyd, -0.05));
  ASSERT_LE(solver.TransverseMarginFor(kHyd), 0.0);
  ASSERT_GT(solver.TransverseRotationDenominator(), 0.0);
  const auto state =
      solver.MakeState(setup.state.iceThickness, setup.state.seaLevel);
  ASSERT_EQ(state.oceanFunction, setup.state.oceanFunction);
  const auto load = setup.MeltLoad(-0.1);
  Evolution evolution(op, solver, state,
                      {.rtol = 1e-12, .max_nonlinear_iterations = 400});
  const auto reference =
      solver.Solve(state, load, RotationalFeedback::Full, {.rtol = 1e-12});
  ASSERT_TRUE(reference.converged);
  ASSERT_NE(reference.angularVelocity[0], 0.0);
  auto worst = [](std::span<const double> x, std::span<const double> y) {
    auto out = 0.0, top = 0.0;
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      out = std::max(out, std::abs(x[i] - y[i]));
      top = std::max(top, std::abs(y[i]));
    }
    return out / top;
  };
  const auto one = std::vector<double>{0.0};
  const auto byLoad = evolution.Run(one, [&](double) { return load; });
  const auto byForcing = evolution.Run(
      one, Evolution::Forcing{.directLoad = [&](double) { return load; }});
  for (const auto* run : {&byLoad, &byForcing}) {
    ASSERT_EQ(run->size(), 1u);
    const auto& node = run->front();
    EXPECT_LT(worst(node.relativeSeaLevelChange,
                    reference.relativeSeaLevelChange),
              1e-10);
    EXPECT_LT(worst(node.displacement, reference.displacement), 1e-10);
    EXPECT_LT(worst(node.angularVelocity, reference.angularVelocity), 1e-10);
  }
  // migrating, the ice thinned at the node: the nonlinear solve's own
  auto thinned = state.iceThickness;
  for (auto& v : thinned) v *= 0.9;
  const auto moving = evolution.RunMigrating(
      one, [&](double) { return thinned; }, Evolution::Forcing{});
  ASSERT_EQ(moving.size(), 1u);
  GRDSolver::NonlinearForcing melt;
  melt.iceThicknessChange = thinned;
  for (auto i = std::size_t{0}; i < thinned.size(); ++i) {
    melt.iceThicknessChange[i] -= state.iceThickness[i];
  }
  melt.rotation = RotationalFeedback::Full;
  const auto nonlinear = solver.SolveNonlinear(
      state, melt, {.rtol = 1e-12, .max_iterations = 400});
  ASSERT_TRUE(moving[0].converged);
  ASSERT_TRUE(nonlinear.response.converged);
  // flotation's 0/1: rounding changes it only at a cell on the threshold
  EXPECT_EQ(moving[0].oceanFunction, nonlinear.finalState.oceanFunction);
  EXPECT_LT(worst(moving[0].relativeSeaLevelChange,
                  nonlinear.response.relativeSeaLevelChange),
            1e-9);
  // a run that steps approaches the fluid limit: refused
  const auto two =
      std::vector<double>{0.0, 1e10 / solver.SolverScales().Time()};
  EXPECT_THROW((void)evolution.Run(two, [&](double) { return load; }),
               std::invalid_argument);
}

TEST(Evolution, ALandHeldThatChangesInTimeHoldsEachNodesOwn) {
  // Under migrating shorelines a land held that changes in time
  // (Options::land_held) is each node's own: a ring of the state's ocean
  // held as land from the second node on leaves the first node's ocean
  // function the run's without it and holds the ring at every later node,
  // the rest of the ocean still following flotation. The background's own
  // land at every node is the run without the option, to the bit, and a
  // land held at the first node other than the background's is refused.
  auto setup = Setup::Make(true);
  ViscoelasticOperator op(setup.rheology, kLmax);
  const auto& grid = setup.solver.Grid();
  const auto kyr = 1000.0 * 3.15576e7 / setup.solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, kyr, 2.0 * kyr, 3.0 * kyr};
  const auto thickness = [&](double t) {
    auto out = setup.state.iceThickness;
    for (auto& v : out) v *= 1.0 - 0.1 * t / times.back();
    return out;
  };
  // a ring of ocean
  const auto& ocean = setup.state.oceanFunction;
  auto ring = std::vector<char>(ocean.size(), 0);
  const auto nphi = grid.Longitudes().size();
  auto ringed = std::size_t{0};
  for (auto i = std::size_t{0}; i < ocean.size(); ++i) {
    const auto theta = grid.CoLatitudes()[i / nphi];
    if (ocean[i] == 1.0 && theta > 1.8 && theta < 2.2) {
      ring[i] = 1;
      ++ringed;
    }
  }
  ASSERT_GT(ringed, 0u);
  const auto options = Evolution::Options{.rtol = 1e-12};
  Evolution plain(op, setup.solver, setup.state, options);
  const auto free = plain.RunMigrating(times, thickness);
  auto held = options;
  held.land_held = [&](double t) {
    return t > 0.0 ? ring : std::vector<char>{};
  };
  Evolution holding(op, setup.solver, setup.state, held);
  const auto moved = holding.RunMigrating(times, thickness);
  ASSERT_EQ(moved.size(), times.size());
  // flotation's 0/1: rounding changes it only at a cell on the threshold
  EXPECT_EQ(moved[0].oceanFunction, free[0].oceanFunction);
  for (auto n = std::size_t{1}; n < times.size(); ++n) {
    auto stillOcean = 0.0;
    auto elsewhere = 0.0;
    for (auto i = std::size_t{0}; i < ocean.size(); ++i) {
      if (ring[i]) {
        stillOcean = std::max(stillOcean, moved[n].oceanFunction[i]);
      } else {
        elsewhere = std::max(elsewhere, moved[n].oceanFunction[i]);
      }
    }
    EXPECT_EQ(stillOcean, 0.0) << "node " << n;
    EXPECT_EQ(elsewhere, 1.0) << "node " << n;
    EXPECT_NE(moved[n].relativeSeaLevelChange, free[n].relativeSeaLevelChange)
        << "node " << n;
  }
  // the background's own land at every node: the run without the option
  auto own = options;
  own.land_held = [&](double) { return setup.state.forcedLand; };
  Evolution same(op, setup.solver, setup.state, own);
  const auto again = same.RunMigrating(times, thickness);
  // the same flotation and solves in one process: bitwise
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_EQ(again[n].relativeSeaLevelChange, free[n].relativeSeaLevelChange)
        << "node " << n;
    EXPECT_EQ(again[n].oceanFunction, free[n].oceanFunction) << "node " << n;
  }
  // a land at the first node that is not the background's
  auto wrong = options;
  wrong.land_held = [&](double) { return ring; };
  Evolution refused(op, setup.solver, setup.state, wrong);
  EXPECT_THROW(static_cast<void>(refused.RunMigrating(times, thickness)),
               std::invalid_argument);
}

TEST(Evolution, ViscousMantleKeepsRelaxing) {
  auto setup = Setup::Make(true);
  ViscoelasticOperator op(setup.rheology, kLmax);
  Evolution evolution(op, setup.solver, setup.state, {});
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto i = 0; i <= 10; ++i) times.push_back(i * kyr);

  // A load switched on at t = 0 and held.
  auto load = setup.MeltLoad(-0.1);
  auto snapshots = evolution.Run(times, [&](double) { return load; });

  // Viscous flow keeps amplifying the response after the elastic step:
  // the peak sea-level change grows monotonically over the history.
  auto peaks = std::vector<double>{};
  for (const auto& snapshot : snapshots) {
    auto peak = 0.0;
    for (auto v : snapshot.relativeSeaLevelChange) peak = std::max(peak, std::abs(v));
    peaks.push_back(peak);
  }
  for (auto i = std::size_t{1}; i < peaks.size(); ++i) {
    EXPECT_GT(peaks[i], peaks[i - 1] * (1.0 - 1e-9)) << "node " << i;
  }
  EXPECT_GT(peaks.back(), 1.2 * peaks.front());
}

TEST(Evolution, WithoutAnOceanIsTheDirectlySteppedBody) {
  // On a state with no ocean the evolution is the solid body stepped
  // under the prescribed load, and under the centrifugal potential of the
  // rotation it finds. The oracle steps the radial operator directly, each
  // degree under the load's coefficients and the centrifugal coefficients
  // of the evolution's own angular velocity, with no sea level solver.
  auto setup = Setup::Make(true);
  ViscoelasticOperator op(setup.rheology, kLmax);
  const auto& solver = setup.solver;
  const auto& grid = solver.Grid();
  const auto& sle = solver.SolverScales();
  const auto& material = setup.model.ModelScales();
  auto ratio = [&](Dimensions dims) {
    return material.Factor(dims) / sle.Factor(dims);
  };
  const auto toLength = ratio(Dimensions{.length = 1});
  const auto toPotential = ratio(PotentialDimensions);
  const auto toMatDensity = 1.0 / ratio(Dimensions{.mass = 1, .length = -2});
  const auto toMatTime = 1.0 / ratio(Dimensions{.time = 1});

  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  auto times = std::vector<double>{};
  for (auto i = 0; i <= 6; ++i) times.push_back((i + 0.5 * (i % 2)) * kyr);
  // A load switched on at the start and growing after it.
  const auto held = setup.MeltLoad(-0.1);
  auto loadAt = [&](double t) {
    auto load = held;
    for (auto& v : load) v *= 1.0 + 0.3 * (t - times.front()) / kyr;
    return load;
  };
  auto columns = [](const std::vector<std::complex<double>>& coefficients,
                    double factor) -> LoadCallable {
    return [coefficients, factor](int l, double)
               -> std::optional<std::vector<double>> {
      auto out = std::vector<double>(2 * (l + 1));
      for (auto m = 0; m <= l; ++m) {
        const auto c = coefficients[SurfaceGrid::Index(l, m)];
        out[m] = factor * c.real();
        out[l + 1 + m] = factor * c.imag();
      }
      return out;
    };
  };
  auto field = [&](const std::map<int, std::vector<double>>& perDegree,
                   double factor) {
    auto lm = grid.ZeroCoefficients();
    for (const auto& [l, c] : perDegree) {
      for (auto m = 0; m <= l; ++m) {
        lm[SurfaceGrid::Index(l, m)] =
            factor * std::complex<double>(c[m], c[l + 1 + m]);
      }
    }
    return grid.Synthesise(lm);
  };
  auto worst = [](std::span<const double> a, std::span<const double> b) {
    auto diff = 0.0, scale = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      diff = std::max(diff, std::abs(a[i] - b[i]));
      scale = std::max(scale, std::abs(b[i]));
    }
    return diff / scale;
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Full}) {
      Evolution evolution(
          op, solver, solver.NoOceanState(),
          {.scheme = scheme, .rotation = rotation, .rtol = 1e-12});
      const auto snapshots = evolution.Run(times, loadAt);
      ASSERT_EQ(snapshots.size(), times.size());

      auto channels = [&](std::size_t node) {
        const auto psi =
            solver.CentrifugalCoefficients(snapshots[node].angularVelocity);
        return std::pair{
            columns(grid.Expand(loadAt(times[node])), toMatDensity),
            columns(psi, 1.0 / toPotential)};
      };
      auto state = op.ZeroState([](int l) { return 2 * (l + 1); });
      state.t = times.front() * toMatTime;
      auto now = channels(0);
      auto solutions = op.SolveElastic(state, now.first, now.second);
      state.strain = op.Strain(solutions);
      for (auto node = std::size_t{0}; node < times.size(); ++node) {
        if (node > 0) {
          auto next = channels(node);
          const auto step =
              op.StartStep(state, (times[node] - times[node - 1]) * toMatTime,
                           scheme, now.first, now.second);
          auto finished = step.Finish(next.first, next.second);
          state = std::move(finished.first);
          solutions = std::move(finished.second);
          now = std::move(next);
        }
        const auto& snapshot = snapshots[node];
        EXPECT_TRUE(snapshot.converged);
        EXPECT_LT(worst(snapshot.displacement,
                        field(op.Surface(solutions, Component::U), toLength)),
                  1e-9)
            << "node " << node << ", rotation " << static_cast<int>(rotation);
        EXPECT_LT(worst(snapshot.potential,
                        field(op.Surface(solutions, Component::Phi),
                              toPotential)),
                  1e-9)
            << "node " << node << ", rotation " << static_cast<int>(rotation);
        const auto spin = std::hypot(snapshot.angularVelocity[0],
                                     snapshot.angularVelocity[1]);
        if (rotation == RotationalFeedback::None) {
          EXPECT_EQ(spin, 0.0);
        } else {
          EXPECT_GT(spin, 0.0);
        }
      }
    }
  }
}

TEST(Evolution, APrescribedTideDrivesTheBodyWithAndWithoutAnOcean) {
  // A prescribed tidal potential beside the direct load. The oracle steps
  // the radial operator directly under the load the evolution carried
  // (the direct load and, with an ocean, the induced one of its own
  // relative sea level change) and under the potential it should have
  // seen, the prescribed tide and the centrifugal potential of its own
  // rotation: the displacement and the potential perturbation must be
  // the evolution's. With an ocean the relative sea level change then
  // holds the tide's own potential, -(u + (phi + psi) / g) up to the
  // uniform shift.
  auto setup = Setup::Make(true);
  ViscoelasticOperator op(setup.rheology, kLmax);
  const auto& solver = setup.solver;
  const auto& grid = solver.Grid();
  const auto& sle = solver.SolverScales();
  const auto& material = setup.model.ModelScales();
  auto ratio = [&](Dimensions dims) {
    return material.Factor(dims) / sle.Factor(dims);
  };
  const auto toLength = ratio(Dimensions{.length = 1});
  const auto toPotential = ratio(PotentialDimensions);
  const auto toMatDensity = 1.0 / ratio(Dimensions{.mass = 1, .length = -2});
  const auto toMatTime = 1.0 / ratio(Dimensions{.time = 1});
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  auto times = std::vector<double>{};
  for (auto i = 0; i <= 5; ++i) times.push_back(1.5 * i * kyr);

  const auto held = setup.MeltLoad(-0.1);
  Evolution::Forcing forcing;
  forcing.directLoad = [&](double t) {
    auto load = held;
    for (auto& v : load) v *= 1.0 + 0.2 * t / kyr;
    return load;
  };
  // Degrees 2, 3 and 5, growing and turning in time.
  const auto amplitude = 5.0 / sle.Factor(PotentialDimensions);
  forcing.tide = [&](double t) {
    auto psi = grid.ZeroCoefficients();
    const auto s = t / (7.5 * kyr);
    psi[SurfaceGrid::Index(2, 0)] = amplitude * (1.0 + s);
    psi[SurfaceGrid::Index(2, 1)] = amplitude * std::complex<double>(0.4, -0.3 * s);
    psi[SurfaceGrid::Index(2, 2)] = amplitude * std::complex<double>(-0.2 * s, 0.5);
    psi[SurfaceGrid::Index(3, 1)] = amplitude * std::complex<double>(0.3, 0.2);
    psi[SurfaceGrid::Index(5, 4)] = amplitude * std::complex<double>(0.1 * s, -0.2);
    return psi;
  };
  auto columns = [](const std::vector<std::complex<double>>& coefficients,
                    double factor) -> LoadCallable {
    return [coefficients, factor](int l, double)
               -> std::optional<std::vector<double>> {
      auto out = std::vector<double>(2 * (l + 1));
      for (auto m = 0; m <= l; ++m) {
        const auto c = coefficients[SurfaceGrid::Index(l, m)];
        out[m] = factor * c.real();
        out[l + 1 + m] = factor * c.imag();
      }
      return out;
    };
  };
  auto field = [&](const std::map<int, std::vector<double>>& perDegree,
                   double factor) {
    auto lm = grid.ZeroCoefficients();
    for (const auto& [l, c] : perDegree) {
      for (auto m = 0; m <= l; ++m) {
        lm[SurfaceGrid::Index(l, m)] =
            factor * std::complex<double>(c[m], c[l + 1 + m]);
      }
    }
    return grid.Synthesise(lm);
  };
  auto worst = [](std::span<const double> a, std::span<const double> b) {
    auto diff = 0.0, scale = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      diff = std::max(diff, std::abs(a[i] - b[i]));
      scale = std::max(scale, std::abs(b[i]));
    }
    return diff / scale;
  };

  const auto dry = solver.NoOceanState();
  const auto& wet = setup.state;
  for (const auto* background : {&dry, &wet}) {
    for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Full}) {
      const auto label = std::string(background == &dry ? "no ocean" : "ocean") +
                         ", rotation " + std::to_string(static_cast<int>(rotation));
      Evolution evolution(op, solver, *background,
                          {.rotation = rotation, .rtol = 1e-13});
      const auto snapshots = evolution.Run(times, forcing);
      ASSERT_EQ(snapshots.size(), times.size());

      auto channels = [&](std::size_t node) {
        const auto& snapshot = snapshots[node];
        auto load = forcing.directLoad(times[node]);
        const auto induced = solver.LoadFromSeaLevelChange(
            *background, snapshot.relativeSeaLevelChange);
        for (auto i = std::size_t{0}; i < load.size(); ++i) load[i] += induced[i];
        auto psi = forcing.tide(times[node]);
        if (rotation != RotationalFeedback::None) {
          const auto spin =
              solver.CentrifugalCoefficients(snapshot.angularVelocity);
          for (auto i = std::size_t{0}; i < psi.size(); ++i) psi[i] += spin[i];
        }
        return std::pair{columns(grid.Expand(load), toMatDensity),
                         columns(psi, 1.0 / toPotential)};
      };
      auto state = op.ZeroState([](int l) { return 2 * (l + 1); });
      state.t = times.front() * toMatTime;
      auto now = channels(0);
      auto solutions = op.SolveElastic(state, now.first, now.second);
      state.strain = op.Strain(solutions);
      for (auto node = std::size_t{0}; node < times.size(); ++node) {
        if (node > 0) {
          auto next = channels(node);
          const auto step = op.StartStep(
              state, (times[node] - times[node - 1]) * toMatTime,
              Scheme::ExponentialTrapezoid, now.first, now.second);
          auto finished = step.Finish(next.first, next.second);
          state = std::move(finished.first);
          solutions = std::move(finished.second);
          now = std::move(next);
        }
        const auto& snapshot = snapshots[node];
        EXPECT_TRUE(snapshot.converged) << label;
        const auto u = field(op.Surface(solutions, Component::U), toLength);
        const auto phi = field(op.Surface(solutions, Component::Phi), toPotential);
        EXPECT_LT(worst(snapshot.displacement, u), 1e-9)
            << "node " << node << ", " << label;
        EXPECT_LT(worst(snapshot.potential, phi), 1e-9)
            << "node " << node << ", " << label;

        // The tangential displacement's potential, which the solver joins
        // from its tangential numbers and the step's memory.
        const auto v = field(op.Surface(solutions, Component::V), toLength);
        EXPECT_GT(*std::ranges::max_element(v), 0.0);
        EXPECT_LT(worst(snapshot.tangentialPotential, v), 1e-9)
            << "node " << node << ", " << label;

        // The sea level change holds the potentials that are not the
        // body's own: the tide's, and the centrifugal one's.
        auto psi = forcing.tide(times[node]);
        if (rotation != RotationalFeedback::None) {
          const auto spin =
              solver.CentrifugalCoefficients(snapshot.angularVelocity);
          for (auto i = std::size_t{0}; i < psi.size(); ++i) psi[i] += spin[i];
        }
        psi[SurfaceGrid::Index(0, 0)] = 0.0;
        const auto outer = grid.Synthesise(psi);
        auto low = 1e300, high = -1e300, scale = 0.0;
        for (auto i = std::size_t{0}; i < u.size(); ++i) {
          const auto expected =
              -(u[i] + (phi[i] + outer[i]) / solver.SurfaceGravity());
          const auto shift = snapshot.relativeSeaLevelChange[i] - expected;
          low = std::min(low, shift);
          high = std::max(high, shift);
          scale = std::max(scale, std::abs(expected));
        }
        EXPECT_LT(high - low, 1e-8 * scale) << "node " << node << ", " << label;
      }
    }
  }
}

TEST(EvolutionGate, CoupledEvolutionMatchesPyslfp) {
  // The reference is pyslfp on its GLQ grid, which coincides with ours at
  // this lmax, so the comparison isolates the implementation; the sharp
  // coastline makes the solution itself grid-determined at the ~19%
  // level (the fixture header records pyslfp's own DH-vs-GLQ spread).
  auto setup = Setup::Make(true);
  ASSERT_EQ(setup.solver.Grid().NumberOfLongitudes(), 2 * kLmax + 1)
      << "the GSHTrans grid no longer coincides with pyshtools GLQ here";
  ViscoelasticOperator op(setup.rheology, kLmax);
  Evolution evolution(op, setup.solver, setup.state, {.rtol = 1e-11});
  const auto& grid = setup.solver.Grid();
  const auto& scales = setup.solver.SolverScales();

  const auto tFinal = 5000.0 * 3.15576e7;  // s
  std::vector<double> times, timesSi;
  for (auto i = 0; i <= 10; ++i) {
    timesSi.push_back(i * tFinal / 10.0);
    times.push_back(timesSi.back() / scales.Time());
  }
  auto directLoad = [&](double t) {
    return setup.MeltLoad(-0.1 * (t / times.back()));
  };
  auto snapshots = evolution.Run(times, directLoad);

  // Parse the fixture and compare, SI, in coefficient space.
  std::ifstream file(std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                     "pyslfp_evolution.txt");
  ASSERT_TRUE(file.is_open());
  const auto length = scales.Length();
  const auto frequency = scales.Factor(Dimensions{.time = -1});
  auto node = -1;
  auto ours = std::vector<std::complex<double>>{};
  auto scale = 0.0, worst = 0.0;
  constexpr auto tol = 1e-5;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (line.starts_with("node")) {
      auto head = std::string{};
      auto t = 0.0;
      auto omega = std::array<double, 3>{};
      row >> head >> node >> t >> omega[0] >> omega[1] >> omega[2];
      ASSERT_LT(static_cast<std::size_t>(node), snapshots.size());
      EXPECT_NEAR(timesSi[node], t, 1e-6 * std::max(t, 1.0));
      ours = grid.Expand(snapshots[node].relativeSeaLevelChange);
      for (auto& v : ours) v *= length;
      if (node > 0) {
        const double omegaScale = std::max(
            {std::abs(omega[0]), std::abs(omega[1]), std::abs(omega[2])});
        for (auto i = 0; i < 3; ++i) {
          EXPECT_NEAR(snapshots[node].angularVelocity[i] * frequency, omega[i],
                      tol * omegaScale)
              << "node " << node << " omega " << i;
        }
      }
      continue;
    }
    auto l = 0, m = 0;
    auto a = 0.0, b = 0.0;
    row >> l >> m >> a >> b;
    const auto reference = ComplexFromRealPair({a, b}, m);
    scale = std::max(scale, std::abs(reference));
    if (node > 0) {
      worst =
          std::max(worst, std::abs(ours[SurfaceGrid::Index(l, m)] - reference));
    }
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LT(worst, tol * scale);
}

namespace {

/// prem8 at the given degree with a Maxwell mantle of `mantle` Pa s between
/// the CMB and 400 km depth, and between 400 and 220 km depth of `weak` Pa
/// s (the mantle's, unless given); its elastic solver and the analytic
/// test state, as Setup.
struct Layered {
  int lmax = 0;
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  GRDSolver::State state;
  std::vector<double> iceNd;

  static Layered Make(int lmax, double mantle, double weak = 0.0) {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
    auto material = Material::Create(model, mesh).value();
    const auto& scales = model.ModelScales();
    const auto b = model.DimensionlessView().Boundaries();
    auto viscosity = std::map<std::size_t, RadialValue>{};
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (model.IsFluid(i) || b[i] < 3.4799e6 / scales.Length() ||
          b[i + 1] > 6.1511e6 / scales.Length()) {
        continue;
      }
      const auto upper = b[i] >= 5.9709e6 / scales.Length();
      const auto eta = upper && weak > 0.0 ? weak : mantle;
      viscosity.emplace(i, eta / scales.Factor(ViscosityDimensions));
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    ViscoelasticOperator op(rheology, lmax);
    GRDSolver solver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
                     SeaLevelParameters{});
    Fingerprint fingerprint(lmax, op.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto ice = toNd(fingerprint.IceThickness());
    auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
    return Layered{lmax,
                   std::move(model),
                   std::move(material),
                   std::move(rheology),
                   std::move(solver),
                   std::move(state),
                   std::move(ice)};
  }

  [[nodiscard]] double Kyr() const {
    return 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  }
  [[nodiscard]] std::vector<double> MeltLoad(double fraction) const {
    auto load = std::vector<double>(iceNd.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                fraction * iceNd[i];
    }
    return load;
  }
};

/// The grid of n equal steps over [0, span].
std::vector<double> Even(double span, int n) {
  auto out = std::vector<double>{};
  for (auto k = 0; k <= n; ++k) out.push_back(span * k / n);
  return out;
}

/// The relative L2 gap of two grid fields.
double Gap(const GRDSolver& solver, std::span<const double> got,
           std::span<const double> want) {
  auto d = std::vector<double>(want.size()),
       w = std::vector<double>(want.size());
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    d[i] = (got[i] - want[i]) * (got[i] - want[i]);
    w[i] = want[i] * want[i];
  }
  return std::sqrt(solver.Grid().Integrate(d) / solver.Grid().Integrate(w));
}

}  // namespace

TEST(Evolution, Sdirk2ConvergesAtSecondOrderToTheTrapezoidsLimit) {
  // A cap's load ramped on over half the window and held, with an ocean
  // and the full rotational feedback, the steps from 9.3 times the
  // shortest relaxation time down: Sdirk2's sea level at the kink and at
  // the end converges at second order, to the limit the exponential
  // trapezoid converges to (each Richardson-extrapolated from fine steps),
  // and at the longest steps its error is many times the trapezoid's
  // smaller, the trapezoid acting as backward Euler on the slow modes the
  // stiff points carry.
  const auto scene = Layered::Make(8, 1e21);
  ViscoelasticOperator op(scene.rheology, scene.lmax);
  const auto kyr = scene.Kyr();
  const auto span = 8.0 * kyr;
  const auto ratio = op.ShortestRelaxation().StepRatio(
      Even(span, 8),
      scene.solver.SolverScales().Time() / scene.model.ModelScales().Time());
  const auto load = scene.MeltLoad(0.3);
  const auto direct = [&](double t) {
    auto out = load;
    const auto f = std::min(t / (0.5 * span), 1.0);
    for (auto& v : out) v *= f;
    return out;
  };
  // the sea level at the kink and at the end
  auto run = [&](Scheme scheme, int n) {
    Evolution evolution(op, scene.solver, scene.state,
                        {.scheme = scheme, .rtol = 1e-12});
    const auto snapshots = evolution.Run(Even(span, n), direct);
    return std::array{snapshots[n / 2].relativeSeaLevelChange,
                      snapshots[n].relativeSeaLevelChange};
  };
  auto richardson = [](const std::array<std::vector<double>, 2>& coarse,
                       const std::array<std::vector<double>, 2>& fine) {
    auto out = fine;
    for (auto k = 0; k < 2; ++k) {
      for (auto i = std::size_t{0}; i < out[k].size(); ++i) {
        out[k][i] += (fine[k][i] - coarse[k][i]) / 3.0;
      }
    }
    return out;
  };
  const auto limit =
      richardson(run(Scheme::Sdirk2, 64), run(Scheme::Sdirk2, 128));
  const auto trapezoid = richardson(run(Scheme::ExponentialTrapezoid, 256),
                                    run(Scheme::ExponentialTrapezoid, 512));
  auto errors = std::map<std::pair<Scheme, int>, std::array<double, 2>>{};
  for (const auto scheme : {Scheme::Sdirk2, Scheme::ExponentialTrapezoid}) {
    for (const auto n : {8, 16, 32}) {
      const auto x = run(scheme, n);
      for (auto k = 0; k < 2; ++k) {
        errors[{scheme, n}][k] = Gap(scene.solver, x[k], limit[k]);
      }
    }
  }
  std::cout << std::format("[sdirk2 convergence] step ratio {:.2f}\n", ratio);
  for (const auto& [key, e] : errors) {
    std::cout << std::format("  scheme {} n {}: {:.3e} {:.3e}\n",
                             static_cast<int>(key.first), key.second, e[0],
                             e[1]);
  }
  const auto limits = std::array{Gap(scene.solver, trapezoid[0], limit[0]),
                                 Gap(scene.solver, trapezoid[1], limit[1])};
  std::cout << std::format("  the limits' gap {:.3e} {:.3e}\n", limits[0],
                           limits[1]);
  // measured: steps 9.3 times the shortest relaxation time; Sdirk2's errors
  // 5.6e-4, 1.4e-4, 3.3e-5 at the kink (3.8e-4, 9.1e-5, 2.2e-5 at the
  // end), the trapezoid's 8.2e-3, 2.8e-3, 7.9e-4 (4.6e-3, 1.6e-3, 4.6e-4);
  // the two limits 7e-9 apart
  EXPECT_GT(ratio, 9.0);
  EXPECT_LT(ratio, 9.5);
  const auto error = [&](Scheme scheme, int n, int k) {
    return errors.at(std::pair{scheme, n})[static_cast<std::size_t>(k)];
  };
  for (auto k = 0; k < 2; ++k) {
    EXPECT_LT(limits[k], 1e-7) << k;
    for (const auto n : {8, 16}) {
      const auto order = std::log2(error(Scheme::Sdirk2, n, k) /
                                   error(Scheme::Sdirk2, 2 * n, k));
      EXPECT_GT(order, 1.9) << n << ", " << k;
      EXPECT_LT(order, 2.2) << n << ", " << k;
    }
    const auto damped = error(Scheme::Sdirk2, 8, k);
    EXPECT_LT(damped, error(Scheme::ExponentialTrapezoid, 8, k) / 10.0) << k;
    EXPECT_LT(damped, 1e-3) << k;
  }
}

TEST(Evolution, Sdirk2DampsAfterAnAbruptUnloading) {
  // A cap's load held for 2 kyr and taken off over 0.02 kyr, then 1 kyr
  // steps, on a mantle of 1e21 Pa s with a layer between 220 and 400 km
  // depth of 1e19 Pa s (its relaxation time 0.0039 kyr, steps 250 times
  // it), and of 1e17 Pa s (the fluid limit, 25 000 times). The sea level's
  // projection on its change since the unloading, against the run with
  // every step split in 64 (whose own split in 128 it meets to 6e-5): on
  // the 1e19 layer the steps' changes keep their sign, no sawtooth, and
  // after the first step Sdirk2's error is a fifth of the trapezoid's or
  // less; in the fluid limit its error after the first step is below a
  // tenth of the first's, and below 1e-3 of the change from the fourth
  // step on, its sign alternating at first (the channel modes the step
  // takes at h/T 3 to 20, where Sdirk2's amplification is -0.21 at the
  // least, damped at least 4.8 times a step), the trapezoid's still 0.15 %
  // at the end.
  for (const auto weak : {1e19, 1e17}) {
    const auto label = std::format("layer {:g} Pa s", weak);
    const auto scene = Layered::Make(8, 1e21, weak);
    ViscoelasticOperator op(scene.rheology, scene.lmax);
    const auto kyr = scene.Kyr();
    const auto load = scene.MeltLoad(0.3);
    const auto off = 2.0 * kyr, ramp = 0.02 * kyr;
    const auto direct = [&](double t) {
      auto out = load;
      const auto f = t <= off ? 1.0 : std::max(0.0, 1.0 - (t - off) / ramp);
      for (auto& v : out) v *= f;
      return out;
    };
    // 1 kyr to the unloading, the unloading, 6 kyr after; each step split
    auto grid = [&](int split) {
      auto out = std::vector<double>{0.0};
      auto add = [&](double from, double to) {
        for (auto k = 1; k <= split; ++k) {
          out.push_back(from + (to - from) * k / split);
        }
      };
      add(0.0, kyr);
      add(kyr, off);
      add(off, off + ramp);
      for (auto j = 0; j < 6; ++j) {
        add(off + ramp + j * kyr, off + ramp + (j + 1) * kyr);
      }
      return out;
    };
    auto run = [&](Scheme scheme, int split) {
      Evolution evolution(op, scene.solver, scene.state,
                          {.scheme = scheme, .rtol = 1e-12});
      const auto snapshots = evolution.Run(grid(split), direct);
      auto out = std::vector<std::vector<double>>{};
      for (auto k = std::size_t{0}; k < snapshots.size(); k += split) {
        out.push_back(snapshots[k].relativeSeaLevelChange);
      }
      return out;
    };
    const auto reference = run(Scheme::Sdirk2, 64);
    // nodes 3 on: the unloading's end and the six steps after
    auto change = reference.back();
    for (auto i = std::size_t{0}; i < change.size(); ++i) {
      change[i] -= reference[3][i];
    }
    auto project = [&](const std::vector<double>& field) {
      auto p = std::vector<double>(field.size());
      for (auto i = std::size_t{0}; i < field.size(); ++i) {
        p[i] = field[i] * change[i];
      }
      return scene.solver.Grid().Integrate(p);
    };
    const auto total =
        std::abs(project(reference.back()) - project(reference[3]));
    auto errorsOf = [&](const std::vector<std::vector<double>>& x) {
      auto out = std::vector<double>{};
      for (auto k = std::size_t{3}; k < x.size(); ++k) {
        out.push_back((project(x[k]) - project(reference[k])) / total);
      }
      return out;
    };
    const auto damped = run(Scheme::Sdirk2, 1);
    const auto trapezoid = run(Scheme::ExponentialTrapezoid, 1);
    const auto e = errorsOf(damped);
    const auto t = errorsOf(trapezoid);
    std::cout << std::format("[sdirk2 unloading] {}\n", label);
    for (auto k = std::size_t{0}; k < e.size(); ++k) {
      std::cout << std::format(
          "  node {} sdirk2 {:+.3e} trapezoid {:+.3e} step {:+.4e}\n", k + 3,
          e[k], t[k],
          k > 0 ? project(damped[k + 3]) - project(damped[k + 2]) : 0.0);
    }
    if (weak == 1e19) {
      // measured: errors -0.76, +2.8, +0.37, +0.38, +0.18, +0.12, +0.07 %
      // (the trapezoid's +8.1, -3.4, -3.5, -2.7, -2.0, -1.4, -1.0 %: 7 to
      // 15 times Sdirk2's after the first step), every step's change
      // positive
      const auto first = project(damped[4]) - project(damped[3]);
      for (auto k = std::size_t{5}; k < damped.size(); ++k) {
        const auto step = project(damped[k]) - project(damped[k - 1]);
        EXPECT_GT(step * first, 0.0) << label << ", node " << k;
      }
      for (auto k = std::size_t{2}; k < e.size(); ++k) {
        EXPECT_LT(std::abs(e[k]), std::abs(t[k]) / 5.0)
            << label << ", node " << k + 3;
      }
      EXPECT_LT(std::abs(e[1]), 0.04) << label;
    } else {
      // measured: +0.89, +12.7, -0.98, +0.27, -0.020, +0.016, +0.0046 %
      // (the trapezoid's -14, -4.2, -1.3, -0.50, -0.25, -0.15 % after the
      // first step)
      for (auto k = std::size_t{2}; k < e.size(); ++k) {
        EXPECT_LT(std::abs(e[k]), 0.1 * std::abs(e[1]))
            << label << ", node " << k + 3;
      }
      for (auto k = std::size_t{4}; k < e.size(); ++k) {
        EXPECT_LT(std::abs(e[k]), 1e-3) << label << ", node " << k + 3;
      }
      EXPECT_LT(std::abs(e.back()), std::abs(t.back()) / 10.0) << label;
    }
  }
}

TEST(Evolution, Sdirk2RefusesTheFactoredSources) {
  // factor_sources samples the sources' histories at the nodes, and
  // Sdirk2's first stage reads them between: refused, naming the scheme.
  const auto scene = Layered::Make(4, 1e21);
  ViscoelasticOperator op(scene.rheology, scene.lmax);
  Evolution evolution(op, scene.solver, scene.solver.NoOceanState(),
                      {.scheme = Scheme::Sdirk2,
                       .rotation = RotationalFeedback::None,
                       .factor_sources = true});
  const auto source = PointSource{.at = {.radius = 0.9 * scene.solver.Radius(),
                                         .colatitude = 1.0,
                                         .longitude = 2.0},
                                  .moment = {.rr = 1e-12, .tt = -1e-12}};
  const auto times = Even(scene.Kyr(), 2);
  try {
    static_cast<void>(
        evolution.Run(times, Evolution::Forcing{.sources = [&](double) {
                        return std::vector<PointSource>{source};
                      }}));
    ADD_FAILURE() << "the factored sources ran under Sdirk2";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("Sdirk2"), std::string::npos)
        << error.what();
  }
}
