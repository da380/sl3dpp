// The migrating-shoreline evolution and the initial sea level calibration,
// pyslfp 3.0.0's tests lifted: the frozen calibration is one linear step;
// migrating runs reduce to frozen ones when nothing can move, conserve mass
// when it does, and are the date-by-date nonlinear solve in the elastic
// limit; the kernels about a migrating trajectory match central
// differences; and both calibration methods recover a known initial sea
// level from the final one its migrating run reached. In 3D the
// zero-lateral migrating run and its kernels are the 1D ones, and with
// lateral viscosity the run conserves mass and resumes exactly, its kernels
// converge to central differences, and both methods calibrate.

#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <map>
#include <numbers>
#include <random>
#include <sstream>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/ShConventions.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 16;
constexpr auto kYear = 3.15576e7;  // s

/// The prem8 deck's body: a Maxwell mantle (eta 1e21 between the CMB and
/// 220 km depth) or one that never relaxes, with the solver for its
/// elastic table.
struct Body {
  EarthModel model;
  Material material;
  Rheology rheology;
  SeaLevelSolver solver;

  static Body Make(int lmax, bool viscous, double eta = 1e21) {
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
    auto rheology = [&] {
      if (!viscous) {
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
          viscosity.emplace(i, eta / scales.Factor(ViscosityDimensions));
        }
      }
      return Rheology::Maxwell(material, viscosity).value();
    }();
    ViscoelasticOperator op(rheology, lmax);
    SeaLevelSolver solver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
                          SeaLevelParameters{});
    return Body{std::move(model), std::move(material), std::move(rheology),
                std::move(solver)};
  }

  [[nodiscard]] double Metre() const {
    return 1.0 / solver.SolverScales().Length();
  }
  [[nodiscard]] double Kyr() const {
    return 1e3 * kYear / solver.SolverScales().Time();
  }
  /// int f dS, the radius factor included.
  [[nodiscard]] double Integrate(std::span<const double> f) const {
    return solver.Radius() * solver.Radius() * solver.Grid().Integrate(f);
  }
  /// A field band-limited to the grid's degree (expanded and synthesised).
  [[nodiscard]] std::vector<double> BandLimited(
      const std::vector<double>& field) const {
    return solver.Grid().Synthesise(solver.Grid().Expand(field));
  }
  /// A smooth random field: unit normal coefficients to the grid's degree,
  /// scaled so that its largest value is `amplitude`.
  [[nodiscard]] std::vector<double> Smooth(double amplitude,
                                           unsigned seed) const {
    const auto& grid = solver.Grid();
    auto engine = std::mt19937(seed);
    auto normal = std::normal_distribution<double>(0.0, 1.0);
    auto c = grid.ZeroCoefficients();
    for (auto l = 0; l <= grid.LMax(); ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto re = normal(engine);
        const auto im = m == 0 ? 0.0 : normal(engine);
        c[SurfaceGrid::Index(l, m)] = {re, im};
      }
    }
    auto f = grid.Synthesise(c);
    auto peak = 0.0;
    for (auto v : f) peak = std::max(peak, std::abs(v));
    for (auto& v : f) v *= amplitude / peak;
    return f;
  }
};

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

/// pyslfp's marine-grounded cap on a uniform ocean, non-smooth enough that
/// melting flips flotation cells: a 5 km cap centred at 60N 85W (radius 20
/// degrees, cos^2 profile, band-limited) on a uniform 3000 m ocean,
/// complete at 10 ka and melting linearly to nothing at present.
struct MarineCap {
  std::vector<double> ice0;  ///< solver units
  std::vector<double> sea0;
  std::vector<double> times;  ///< the dates 10 ka -> 0, SLE units
  double span = 0.0;

  static MarineCap Make(const Body& body, int nodes,
                        double thickness_m = 5000.0) {
    const auto& grid = body.solver.Grid();
    const auto theta0 = 30.0 * std::numbers::pi / 180.0;
    const auto phi0 = 275.0 * std::numbers::pi / 180.0;
    const auto radius = 20.0 * std::numbers::pi / 180.0;
    auto cap = grid.FromFunction([&](double theta, double phi) {
      const auto c = std::cos(theta) * std::cos(theta0) +
                     std::sin(theta) * std::sin(theta0) * std::cos(phi - phi0);
      const auto d = std::acos(std::clamp(c, -1.0, 1.0));
      if (d >= radius) return 0.0;
      const auto taper = std::cos(0.5 * std::numbers::pi * d / radius);
      return taper * taper;
    });
    MarineCap out;
    out.ice0 = body.BandLimited(cap);
    for (auto& v : out.ice0) v *= thickness_m * body.Metre();
    out.sea0.assign(grid.FieldSize(), 3000.0 * body.Metre());
    out.span = 10.0 * body.Kyr();
    for (auto n = 0; n < nodes; ++n) {
      out.times.push_back(out.span * n / (nodes - 1));
    }
    return out;
  }

  [[nodiscard]] std::vector<double> Thickness(double t) const {
    const auto fraction = std::clamp(1.0 - t / span, 0.0, 1.0);
    auto out = ice0;
    for (auto& v : out) v *= fraction;
    return out;
  }
};

/// The frozen-shoreline load of a thickness history relative to its start.
std::function<std::vector<double>(double)> FrozenLoad(
    const SeaLevelSolver& solver, const SeaLevelSolver::State& state,
    std::function<std::vector<double>(double)> thickness, double t0) {
  const auto first = thickness(t0);
  return [&solver, &state, thickness = std::move(thickness), first](double t) {
    auto load = thickness(t);
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                (load[i] - first[i]);
    }
    return load;
  };
}

/// The analytic test state on the body's grid, solver units.
SeaLevelSolver::State AnalyticState(const Body& body,
                                    const ViscoelasticOperator& op) {
  Fingerprint fingerprint(body.solver.Grid().LMax(), op.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v *= body.Metre();
    return si;
  };
  return body.solver.MakeState(toNd(fingerprint.IceThickness()),
                               toNd(fingerprint.SeaLevel()));
}

}  // namespace

// -- frozen shorelines -------------------------------------------------------

TEST(InitialSeaLevel, FrozenCalibrationIsOneLinearStep) {
  // With the shorelines fixed the problem is linear: one run gives the
  // initial sea level exactly, the calibrated state keeps the guess's
  // shorelines as a prescribed ocean function, and a fresh frozen run from
  // it reaches the target.
  const auto body = Body::Make(kLmax, true);
  ViscoelasticOperator op(body.rheology, kLmax);
  const auto analytic = AnalyticState(body, op);
  const auto& grid = body.solver.Grid();
  auto target = analytic.seaLevel;
  const auto bump = grid.FromFunction([&](double theta, double) {
    return 40.0 * body.Metre() * std::sin(theta);
  });
  for (auto i = std::size_t{0}; i < target.size(); ++i) target[i] += bump[i];
  const auto guess = body.solver.MakeState(analytic.iceThickness, target);

  auto times = std::vector<double>{};
  for (auto n = 0; n < 6; ++n) times.push_back(2.0 * n * body.Kyr());
  const auto melt = [&](double t) {
    auto out = guess.iceThickness;
    for (auto& v : out) v *= 1.0 - 0.1 * t / times.back();
    return out;
  };
  const auto load = FrozenLoad(body.solver, guess, melt, 0.0);

  SeaLevelEvolution evolution(op, body.solver, guess, {.rtol = 1e-12});
  const auto calibration = CalibrateInitialSeaLevel(
      evolution, times, IceHistory::Frozen(load), target);
  EXPECT_TRUE(calibration.converged);
  EXPECT_EQ(calibration.updates.size(), 1u);
  EXPECT_EQ(calibration.state.oceanFunction, guess.oceanFunction);
  EXPECT_EQ(evolution.BackgroundState().seaLevel, calibration.state.seaLevel);

  SeaLevelEvolution fresh(op, body.solver, calibration.state, {.rtol = 1e-12});
  const auto run = fresh.Run(times, load);
  auto reached = calibration.state.seaLevel;
  for (auto i = std::size_t{0}; i < reached.size(); ++i) {
    reached[i] += run.back().seaLevelChange[i];
  }
  EXPECT_LT(MaxDiff(reached, target), 1e-9 * MaxAbs(target));

  // A migrating run follows flotation from the first node, so it refuses
  // a prescribed ocean function that flotation would not give.
  const auto flotation = body.solver.MakeState(calibration.state.iceThickness,
                                               calibration.state.seaLevel);
  ASSERT_NE(flotation.oceanFunction, calibration.state.oceanFunction)
      << "the calibration moved no shoreline; the refusal is untested";
  EXPECT_THROW((void)fresh.RunMigrating(times, melt), std::invalid_argument);
}

TEST(InitialSeaLevel, WeightedCalibrationUpdatesOnlyWhereWeighted) {
  // With weights that vanish over a band, the update leaves the initial
  // sea level at its guess there and fits the final sea level elsewhere.
  const auto body = Body::Make(kLmax, true);
  ViscoelasticOperator op(body.rheology, kLmax);
  const auto truth = AnalyticState(body, op);
  const auto& grid = body.solver.Grid();
  auto times = std::vector<double>{};
  for (auto n = 0; n < 6; ++n) times.push_back(2.0 * n * body.Kyr());
  const auto melt = [&](double t) {
    auto out = truth.iceThickness;
    for (auto& v : out) v *= 1.0 - 0.1 * t / times.back();
    return out;
  };
  const auto load = FrozenLoad(body.solver, truth, melt, 0.0);
  SeaLevelEvolution forward(op, body.solver, truth, {.rtol = 1e-12});
  const auto run = forward.Run(times, load);
  auto final = truth.seaLevel;
  for (auto i = std::size_t{0}; i < final.size(); ++i) {
    final[i] += run.back().seaLevelChange[i];
  }
  const auto weights = grid.FromFunction([](double theta, double) {
    return std::abs(theta - 0.5 * std::numbers::pi) <
                   20.0 * std::numbers::pi / 180.0
               ? 0.0
               : 1.0;
  });
  auto guess = truth;  // the truth's shorelines, the final sea level
  guess.seaLevel = final;
  SeaLevelEvolution evolution(op, body.solver, guess, {.rtol = 1e-12});
  const auto calibration = CalibrateInitialSeaLevel(
      evolution, times, IceHistory::Frozen(load), final, {.weights = weights});
  ASSERT_TRUE(calibration.converged);
  auto worst = 0.0;
  for (auto i = std::size_t{0}; i < final.size(); ++i) {
    if (weights[i] == 0.0) {
      EXPECT_EQ(calibration.state.seaLevel[i], final[i]) << "point " << i;
    } else {
      const auto reached = calibration.state.seaLevel[i] +
                           calibration.snapshots.back().seaLevelChange[i];
      worst = std::max(worst, std::abs(reached - final[i]));
    }
  }
  EXPECT_LT(worst, 1e-9 * MaxAbs(final));
  // and, with the truth's own shorelines, it is the truth where weighted
  auto error = 0.0;
  for (auto i = std::size_t{0}; i < final.size(); ++i) {
    if (weights[i] > 0.0) {
      error = std::max(
          error, std::abs(calibration.state.seaLevel[i] - truth.seaLevel[i]));
    }
  }
  EXPECT_LT(error, 1e-9 * MaxAbs(truth.seaLevel));
}

TEST(InitialSeaLevel, FrozenCalibrationIn3D) {
  // The lateral structure changes xi(t_f), not the linearity: one run of
  // the 3D evolution calibrates, and a fresh run reaches the target.
  constexpr auto lmax = 8;
  const auto body = Body::Make(lmax, true);
  ViscoelasticOperator op1(body.rheology, lmax);
  const auto analytic = AnalyticState(body, op1);
  const auto& grid = body.solver.Grid();
  auto target = analytic.seaLevel;
  const auto bump = grid.FromFunction([&](double theta, double phi) {
    return 25.0 * body.Metre() * std::sin(theta) * std::cos(phi);
  });
  for (auto i = std::size_t{0}; i < target.size(); ++i) target[i] += bump[i];
  const auto guess = body.solver.MakeState(analytic.iceThickness, target);
  auto times = std::vector<double>{0.0, body.Kyr(), 2.0 * body.Kyr()};
  const auto melt = [&](double t) {
    auto out = guess.iceThickness;
    for (auto& v : out) v *= 1.0 - 0.1 * t / times.back();
    return out;
  };
  const auto load = FrozenLoad(body.solver, guess, melt, 0.0);
  Viscoelastic3D op(body.rheology, lmax, [](double, double theta, double phi) {
    return 0.8 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  SeaLevelEvolution3D evolution(op, body.solver, guess,
                                {.rtol = 1e-12, .coupled_rtol = 1e-11});
  const auto calibration = CalibrateInitialSeaLevel(
      evolution, times, IceHistory::Frozen(load), target);
  EXPECT_TRUE(calibration.converged);
  EXPECT_EQ(calibration.state.oceanFunction, guess.oceanFunction);
  SeaLevelEvolution3D fresh(op, body.solver, calibration.state,
                            {.rtol = 1e-12, .coupled_rtol = 1e-11});
  const auto run = fresh.Run(times, load);
  auto reached = calibration.state.seaLevel;
  for (auto i = std::size_t{0}; i < reached.size(); ++i) {
    reached[i] += run.back().seaLevelChange[i];
  }
  EXPECT_LT(MaxDiff(reached, target), 1e-8 * MaxAbs(target));
}

// -- migrating shorelines ----------------------------------------------------

TEST(Migrating, MatchesFrozenWhenShorelinesCannotMove) {
  // The 5 km cap losing one metre over the history: every cell stays
  // grounded, so the shorelines cannot move and the nonlinear step is the
  // linear one.
  const auto body = Body::Make(kLmax, true);
  ViscoelasticOperator op(body.rheology, kLmax);
  const auto cap = MarineCap::Make(body, 5);
  const auto thinning = [&](double t) {
    auto out = cap.ice0;
    const auto loss = std::clamp(t / cap.span, 0.0, 1.0) * body.Metre();
    for (auto& v : out) v = v > 0.0 ? v - loss * v / MaxAbs(cap.ice0) : v;
    return out;
  };
  const auto state = body.solver.MakeState(cap.ice0, cap.sea0);
  // The nonlinear solve forms its load from the absolute fields (3 km of
  // ocean, 5 km of ice) for a change of metres, so its error has a
  // round-off floor near 1e-11: pyslfp's tolerance for this gate.
  SeaLevelEvolution evolution(op, body.solver, state, {.rtol = 1e-9});
  const auto frozen =
      evolution.Run(cap.times, FrozenLoad(body.solver, state, thinning, 0.0));
  const auto moving = evolution.RunMigrating(cap.times, thinning);
  ASSERT_EQ(frozen.size(), moving.size());
  for (auto n = std::size_t{0}; n < frozen.size(); ++n) {
    EXPECT_TRUE(moving[n].converged);
    EXPECT_TRUE(frozen[n].oceanFunction.empty());
    EXPECT_EQ(moving[n].oceanFunction, state.oceanFunction) << "node " << n;
    const auto top = MaxAbs(frozen[n].seaLevelChange);
    EXPECT_LE(MaxDiff(frozen[n].seaLevelChange, moving[n].seaLevelChange),
              1e-6 * std::max(top, 1e-300))
        << "node " << n;
  }
}

TEST(Migrating, ShorelinesMoveAndConserveMass) {
  // The melting cap floods: its grounded region becomes ocean, the explicit
  // water budget balances the grounded ice lost, and the frozen run misses
  // the marine ingress.
  const auto body = Body::Make(kLmax, true);
  ViscoelasticOperator op(body.rheology, kLmax);
  const auto cap = MarineCap::Make(body, 5);
  const auto state = body.solver.MakeState(cap.ice0, cap.sea0);
  SeaLevelEvolution evolution(op, body.solver, state, {.rtol = 1e-12});
  const auto thickness = [&](double t) { return cap.Thickness(t); };
  const auto run = evolution.RunMigrating(cap.times, thickness);
  const auto& C0 = state.oceanFunction;
  const auto& Cf = run.back().oceanFunction;
  auto flooded = std::vector<double>(C0.size());
  for (auto i = std::size_t{0}; i < C0.size(); ++i) flooded[i] = Cf[i] - C0[i];
  EXPECT_GT(body.Integrate(flooded), 0.0);

  auto water = std::vector<double>(C0.size());
  auto grounded = std::vector<double>(C0.size());
  for (auto i = std::size_t{0}; i < C0.size(); ++i) {
    const auto finalSea = state.seaLevel[i] + run.back().seaLevelChange[i];
    water[i] = body.solver.WaterDensity() *
               (Cf[i] * finalSea - C0[i] * state.seaLevel[i]);
    grounded[i] = -body.solver.IceDensity() * (1.0 - C0[i]) *
                  state.iceThickness[i];  // all the ice is gone
  }
  const auto lost = body.Integrate(grounded);
  EXPECT_LT(std::abs(body.Integrate(water) + lost), 1e-8 * std::abs(lost));

  const auto frozen =
      evolution.Run(cap.times, FrozenLoad(body.solver, state, thickness, 0.0));
  EXPECT_GT(MaxDiff(frozen.back().seaLevelChange, run.back().seaLevelChange),
            1e-3 * MaxAbs(run.back().seaLevelChange));
}

TEST(Migrating, ElasticLimitIsTheNonlinearSolveDateByDate) {
  const auto body = Body::Make(kLmax, false);
  ViscoelasticOperator op(body.rheology, kLmax);
  const auto cap = MarineCap::Make(body, 4);
  const auto state = body.solver.MakeState(cap.ice0, cap.sea0);
  SeaLevelEvolution evolution(op, body.solver, state, {.rtol = 1e-12});
  const auto run = evolution.RunMigrating(
      cap.times, [&](double t) { return cap.Thickness(t); });
  for (auto n = std::size_t{1}; n < run.size(); ++n) {
    auto forcing = SeaLevelSolver::NonlinearForcing{};
    forcing.iceThicknessChange = cap.Thickness(cap.times[n]);
    for (auto i = std::size_t{0}; i < state.iceThickness.size(); ++i) {
      forcing.iceThicknessChange[i] -= state.iceThickness[i];
    }
    const auto want = body.solver.SolveNonlinear(
        state, forcing, {.rtol = 1e-12, .max_iterations = 50});
    const auto top = MaxAbs(want.response.seaLevelChange);
    EXPECT_LT(MaxDiff(run[n].seaLevelChange, want.response.seaLevelChange),
              1e-7 * top)
        << "node " << n;
    EXPECT_EQ(run[n].oceanFunction, want.finalState.oceanFunction);
  }
}

TEST(Migrating, AResumedRunIsTheUninterruptedOne) {
  const auto body = Body::Make(kLmax, true);
  ViscoelasticOperator op(body.rheology, kLmax);
  const auto cap = MarineCap::Make(body, 5);
  const auto state = body.solver.MakeState(cap.ice0, cap.sea0);
  const auto thickness = [&](double t) { return cap.Thickness(t); };
  auto checkpoints = std::map<std::size_t, SeaLevelEvolution::Checkpoint>{};
  auto options = SeaLevelEvolution::Options{.rtol = 1e-12};
  options.checkpoint_stride = 1;
  options.on_checkpoint = [&](std::size_t node,
                              const SeaLevelEvolution::Checkpoint& cp) {
    checkpoints.emplace(node, cp);
  };
  SeaLevelEvolution whole(op, body.solver, state, options);
  const auto full = whole.RunMigrating(cap.times, thickness);
  ASSERT_EQ(checkpoints.size(), cap.times.size());
  SeaLevelEvolution again(op, body.solver, state, {.rtol = 1e-12});
  const auto rest = std::vector<double>(cap.times.begin() + 2, cap.times.end());
  const auto resumed =
      again.ResumeMigrating(checkpoints.at(2), rest, thickness);
  ASSERT_EQ(resumed.size(), rest.size() - 1);
  for (auto k = std::size_t{0}; k < resumed.size(); ++k) {
    EXPECT_EQ(resumed[k].seaLevelChange, full[k + 3].seaLevelChange);
    EXPECT_EQ(resumed[k].oceanFunction, full[k + 3].oceanFunction);
  }
}

// -- kernels about a migrating trajectory -------------------------------------

TEST(Migrating, ConstantOceanFunctionHistoryIsTheFrozenCalculation) {
  const auto body = Body::Make(kLmax, true);
  ViscoelasticOperator op(body.rheology, kLmax);
  const auto cap = MarineCap::Make(body, 5);
  const auto state = body.solver.MakeState(cap.ice0, cap.sea0);
  SeaLevelEvolution evolution(op, body.solver, state, {.rtol = 1e-12});
  auto measurement = Measurement{};
  measurement.seaLevelDual = body.Smooth(1.0, 6);
  const auto measurements =
      std::map<std::size_t, Measurement>{{cap.times.size() - 1, measurement}};
  const auto constant =
      std::vector<std::vector<double>>(cap.times.size(), state.oceanFunction);
  for (const auto method :
       {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const AdjointSolver adjoint(evolution, {.method = method, .rtol = 1e-12});
    const auto frozen = adjoint.IceHistoryKernels(cap.times, measurements);
    const auto about =
        adjoint.IceHistoryKernels(cap.times, measurements, nullptr, constant);
    for (auto n = std::size_t{0}; n < cap.times.size(); ++n) {
      EXPECT_EQ(frozen.thickness[n], about.thickness[n]) << "node " << n;
    }
    EXPECT_EQ(MaxAbs(about.initialSeaLevel), 0.0);
  }
}

TEST(Migrating, InitialStateKernelsMatchCentralDifferences) {
  // The initial-sea-level kernel (assembled from the per-node load duals)
  // and the initial-thickness dual (node 0 plus the baseline, grounded by
  // the INITIAL shorelines) both match central differences with rebuilt
  // initial states, while no flotation cell flips.
  const auto body = Body::Make(kLmax, true);
  ViscoelasticOperator op(body.rheology, kLmax);
  const auto cap = MarineCap::Make(body, 5);
  const auto state = body.solver.MakeState(cap.ice0, cap.sea0);
  const auto thickness = [&](double t) { return cap.Thickness(t); };
  SeaLevelEvolution evolution(op, body.solver, state, {.rtol = 1e-12});
  const auto run = evolution.RunMigrating(cap.times, thickness);
  auto measurement = Measurement{};
  measurement.seaLevelDual = body.Smooth(1.0, 12);
  const auto& g = measurement.seaLevelDual;
  const auto kernels =
      AdjointSolver(evolution,
                    {.method = AdjointMethod::Discrete, .rtol = 1e-12})
          .IceHistoryKernels(cap.times, {{cap.times.size() - 1, measurement}},
                             nullptr, OceanFunctionsOf(run));
  EXPECT_GT(MaxAbs(kernels.initialSeaLevel), 0.0);

  auto functional = [&](const SeaLevelSolver::State& start,
                        const std::function<std::vector<double>(double)>& ice) {
    SeaLevelEvolution trial(op, body.solver, start, {.rtol = 1e-12});
    const auto snaps = trial.RunMigrating(cap.times, ice);
    auto product = g;
    for (auto i = std::size_t{0}; i < g.size(); ++i) {
      product[i] *= snaps.back().seaLevelChange[i];
    }
    return body.Integrate(product);
  };
  auto pairing = [&](std::span<const double> a, std::span<const double> b) {
    auto product = std::vector<double>(a.begin(), a.end());
    for (auto i = std::size_t{0}; i < product.size(); ++i) product[i] *= b[i];
    return body.Integrate(product);
  };

  // the initial sea level
  const auto deltaSea = body.Smooth(1.0 * body.Metre(), 13);
  auto shiftedSea = [&](double sign) {
    auto sea = state.seaLevel;
    for (auto i = std::size_t{0}; i < sea.size(); ++i) {
      sea[i] += sign * deltaSea[i];
    }
    const auto shifted = body.solver.MakeState(state.iceThickness, sea);
    EXPECT_EQ(shifted.oceanFunction, state.oceanFunction);
    return shifted;
  };
  auto fd = 0.5 * (functional(shiftedSea(1.0), thickness) -
                   functional(shiftedSea(-1.0), thickness));
  auto predicted = pairing(kernels.initialSeaLevel, deltaSea);
  EXPECT_LT(std::abs(predicted - fd), 1e-6 * std::abs(fd))
      << "fd " << fd << " predicted " << predicted;

  // the initial thickness: the history's I0 and the state's ice move
  // together
  const auto deltaIce = body.Smooth(1.0 * body.Metre(), 14);
  auto thicknessRun = [&](double sign) {
    auto ice = state.iceThickness;
    for (auto i = std::size_t{0}; i < ice.size(); ++i) {
      ice[i] += sign * deltaIce[i];
    }
    const auto shifted = body.solver.MakeState(ice, state.seaLevel);
    EXPECT_EQ(shifted.oceanFunction, state.oceanFunction);
    return functional(shifted, [&, sign](double t) {
      auto out = cap.Thickness(t);
      if (t == cap.times.front()) {
        for (auto i = std::size_t{0}; i < out.size(); ++i) {
          out[i] += sign * deltaIce[i];
        }
      }
      return out;
    });
  };
  fd = 0.5 * (thicknessRun(1.0) - thicknessRun(-1.0));
  predicted = pairing(kernels.thickness[0], deltaIce);
  EXPECT_LT(std::abs(predicted - fd), 1e-8 * std::abs(fd))
      << "fd " << fd << " predicted " << predicted;
}

TEST(Migrating, TrajectoryKernelsMatchCentralDifferences) {
  // About the moving trajectory the discrete transpose matches central
  // differences of the migrating run to solver precision (the forward is
  // the linearised map while no cell flips), and the continuous method
  // converges at second order once the rheology is resolved: their
  // disagreement on an under-resolved run is the resolution diagnostic.
  auto check = [](double eta, int nodes) {
    const auto body = Body::Make(kLmax, true, eta);
    ViscoelasticOperator op(body.rheology, kLmax);
    const auto cap = MarineCap::Make(body, nodes);
    const auto state = body.solver.MakeState(cap.ice0, cap.sea0);
    SeaLevelEvolution evolution(op, body.solver, state, {.rtol = 1e-12});
    const auto thickness = [&](double t) { return cap.Thickness(t); };
    const auto run = evolution.RunMigrating(cap.times, thickness);
    const auto oceans = OceanFunctionsOf(run);
    auto measurement = Measurement{};
    measurement.seaLevelDual = body.Smooth(1.0, 6);
    const auto& g = measurement.seaLevelDual;

    // Perturbations at the quarter dates but the first (the state's own
    // ice), the same on any refinement of them.
    auto deltaAt = std::map<int, std::vector<double>>{};
    for (auto q = 1; q <= 4; ++q) {
      deltaAt.emplace(q, body.Smooth(1.0 * body.Metre(), 100 + q));
    }
    const auto per = (nodes - 1) / 4;
    auto delta = [&](std::size_t n) -> const std::vector<double>* {
      if (n == 0 || n % per != 0) return nullptr;
      return &deltaAt.at(static_cast<int>(n / per));
    };
    auto functional = [&](double sign) {
      SeaLevelEvolution trial(op, body.solver, state, {.rtol = 1e-12});
      const auto snaps = trial.RunMigrating(cap.times, [&, sign](double t) {
        auto out = cap.Thickness(t);
        for (auto n = std::size_t{0}; n < cap.times.size(); ++n) {
          const auto* d = delta(n);
          if (t != cap.times[n] || d == nullptr) continue;
          for (auto i = std::size_t{0}; i < out.size(); ++i) {
            out[i] += sign * (*d)[i];
          }
        }
        return out;
      });
      for (auto n = std::size_t{0}; n < snaps.size(); ++n) {
        EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped";
      }
      auto product = g;
      for (auto i = std::size_t{0}; i < g.size(); ++i) {
        product[i] *= snaps.back().seaLevelChange[i];
      }
      return body.Integrate(product);
    };
    const auto fd = 0.5 * (functional(1.0) - functional(-1.0));
    auto errors = std::map<AdjointMethod, double>{};
    for (const auto method :
         {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto kernels =
          AdjointSolver(evolution, {.method = method, .rtol = 1e-12})
              .IceHistoryKernels(cap.times,
                                 {{cap.times.size() - 1, measurement}}, nullptr,
                                 oceans);
      auto predicted = 0.0;
      for (auto n = std::size_t{0}; n < cap.times.size(); ++n) {
        const auto* d = delta(n);
        if (d == nullptr) continue;
        auto product = kernels.thickness[n];
        for (auto i = std::size_t{0}; i < product.size(); ++i) {
          product[i] *= (*d)[i];
        }
        predicted += body.Integrate(product);
      }
      errors[method] = std::abs(predicted - fd) / std::abs(fd);
    }
    return errors;
  };
  const auto stiff = check(1e21, 5);
  EXPECT_LT(stiff.at(AdjointMethod::Discrete), 1e-8);
  // A ten times stiffer mantle (relaxation of order the 2.5 kyr step).
  const auto coarse = check(1e22, 5);
  const auto fine = check(1e22, 9);
  EXPECT_LT(coarse.at(AdjointMethod::Discrete), 1e-8);
  EXPECT_LT(fine.at(AdjointMethod::Discrete), 1e-8);
  EXPECT_LT(coarse.at(AdjointMethod::Continuous), 5e-2);
  EXPECT_GT(
      coarse.at(AdjointMethod::Continuous) / fine.at(AdjointMethod::Continuous),
      2.0)
      << "coarse " << coarse.at(AdjointMethod::Continuous) << " fine "
      << fine.at(AdjointMethod::Continuous);
}

// -- calibration under migrating shorelines -----------------------------------

namespace {

/// The marine cap's history over a known, non-uniform initial sea level,
/// and the final sea level its migrating run reaches: the round trip the
/// calibration must invert, with flotation cells flipping on the way.
struct MarineTruth {
  Body body = Body::Make(kLmax, true);
  ViscoelasticOperator op{body.rheology, kLmax};
  MarineCap cap = MarineCap::Make(body, 6);
  SeaLevelSolver::State truth;
  std::vector<double> final;

  MarineTruth() {
    const auto& grid = body.solver.Grid();
    const auto bump = grid.FromFunction([&](double theta, double phi) {
      return 200.0 * body.Metre() * std::sin(theta) * std::cos(phi);
    });
    auto sea = cap.sea0;
    for (auto i = std::size_t{0}; i < sea.size(); ++i) sea[i] += bump[i];
    truth = body.solver.MakeState(cap.ice0, sea);
    SeaLevelEvolution evolution(op, body.solver, truth, {.rtol = 1e-12});
    const auto run = evolution.RunMigrating(cap.times, Thickness());
    EXPECT_NE(run.back().oceanFunction, truth.oceanFunction);
    final = truth.seaLevel;
    for (auto i = std::size_t{0}; i < final.size(); ++i) {
      final[i] += run.back().seaLevelChange[i];
    }
  }

  [[nodiscard]] std::function<std::vector<double>(double)> Thickness() const {
    return [this](double t) { return cap.Thickness(t); };
  }
};

}  // namespace

TEST(InitialSeaLevel, FinalMisfitGradientMatchesCentralDifferences) {
  // About a migrating trajectory the adjoint gradient of the misfit
  // predicts central differences of it. The perturbation flips no initial
  // flotation cell, so the misfit is quadratic along it; the direct term
  // alone is far from the answer, so the agreement rests on the initial
  // sea level kernel.
  const auto setting = MarineTruth{};
  const auto& body = setting.body;
  const auto& grid = body.solver.Grid();
  auto target = setting.final;
  const auto tilt = grid.FromFunction([&](double theta, double) {
    return 50.0 * body.Metre() * std::cos(theta);
  });
  for (auto i = std::size_t{0}; i < target.size(); ++i) target[i] += tilt[i];
  const auto history = IceHistory::Migrating(setting.Thickness());
  SeaLevelEvolution evolution(setting.op, body.solver, setting.truth,
                              {.rtol = 1e-12});
  const auto base = ComputeFinalSeaLevelMisfit(
      evolution, setting.cap.times, history, target, {}, true,
      {.method = AdjointMethod::Discrete, .rtol = 1e-12});
  const auto direction = grid.FromFunction([&](double theta, double phi) {
    return 30.0 * body.Metre() * std::sin(theta) * std::sin(theta) *
           std::sin(phi);
  });
  auto pairing = [&](std::span<const double> f) {
    auto product = std::vector<double>(f.begin(), f.end());
    for (auto i = std::size_t{0}; i < product.size(); ++i) {
      product[i] *= direction[i];
    }
    return body.Integrate(product);
  };
  const auto predicted = pairing(base.gradient);
  const auto eps = 1e-2;
  auto values = std::vector<double>{};
  for (const auto sign : {1.0, -1.0}) {
    auto sea = setting.truth.seaLevel;
    for (auto i = std::size_t{0}; i < sea.size(); ++i) {
      sea[i] += sign * eps * direction[i];
    }
    const auto state = body.solver.MakeState(setting.truth.iceThickness, sea);
    ASSERT_EQ(state.oceanFunction, setting.truth.oceanFunction);
    evolution.SetBackgroundState(state);
    values.push_back(ComputeFinalSeaLevelMisfit(evolution, setting.cap.times,
                                                history, target, {}, false)
                         .value);
  }
  const auto fd = (values[0] - values[1]) / (2.0 * eps);
  EXPECT_LT(std::abs(predicted - fd), 1e-6 * std::abs(fd))
      << "fd " << fd << " predicted " << predicted;
  EXPECT_GT(std::abs(pairing(base.residual) - fd), 1e-3 * std::abs(fd))
      << "the residual alone already predicts the difference";
}

TEST(InitialSeaLevel, CalibrationRecoversTheTruthUnderMigratingShorelines) {
  // Both methods recover a known initial sea level from the final one its
  // migrating run reached. Unaccelerated, the classical update contracts
  // faster than unit steps down the gradient: it is Newton's method with
  // the identity for the Jacobian I + A, contracting at the rate of A,
  // where the gradient step applies I + A^T and contracts at about that of
  // A + A^T.
  const auto setting = MarineTruth{};
  const auto& body = setting.body;
  const auto history = IceHistory::Migrating(setting.Thickness());
  const auto scale = MaxAbs(setting.truth.seaLevel);
  const auto guess =
      body.solver.MakeState(setting.truth.iceThickness, setting.final);
  for (const auto method :
       {CalibrationMethod::FixedPoint, CalibrationMethod::Gradient}) {
    SeaLevelEvolution evolution(setting.op, body.solver, guess,
                                {.rtol = 1e-12});
    const auto calibration = CalibrateInitialSeaLevel(
        evolution, setting.cap.times, history, setting.final,
        {.method = method, .iterations = 12, .rtol = 1e-9});
    EXPECT_TRUE(calibration.converged)
        << static_cast<int>(method) << " in " << calibration.updates.size();
    EXPECT_LT(MaxDiff(calibration.state.seaLevel, setting.truth.seaLevel),
              1e-8 * scale)
        << static_cast<int>(method);
    EXPECT_EQ(evolution.BackgroundState().seaLevel, calibration.state.seaLevel);
  }
  auto rates = std::map<CalibrationMethod, double>{};
  for (const auto method :
       {CalibrationMethod::FixedPoint, CalibrationMethod::Gradient}) {
    SeaLevelEvolution evolution(setting.op, body.solver, guess,
                                {.rtol = 1e-12});
    const auto plain = CalibrateInitialSeaLevel(evolution, setting.cap.times,
                                                history, setting.final,
                                                {.method = method,
                                                 .iterations = 4,
                                                 .rtol = 1e-12,
                                                 .anderson_memory = 0});
    EXPECT_FALSE(plain.converged);
    ASSERT_EQ(plain.misfits.size(), 4u);
    rates[method] = plain.misfits[3] / plain.misfits[2];
  }
  // pyslfp's homogeneous body gives about 0.08 and 0.33; this one 0.6
  // for the gradient.
  EXPECT_LT(rates[CalibrationMethod::FixedPoint],
            0.5 * rates[CalibrationMethod::Gradient]);
  EXPECT_LT(rates[CalibrationMethod::Gradient], 0.8);
}

// -- the pyslfp gate ----------------------------------------------------------

namespace {

/// pyslfp_migrating.txt: the initial fields, the per-node run, the misfit
/// gradient and the calibration histories.
struct MigratingFixture {
  std::vector<std::complex<double>> ice0, sea0, gradient;
  struct Node {
    double t = 0.0, oceanArea = 0.0;
    std::array<double, 3> omega{};
    std::vector<std::complex<double>> slc, ocean;
  };
  std::vector<Node> nodes;
  std::map<std::string, std::vector<double>> misfits;

  static MigratingFixture Read(int lmax) {
    std::ifstream file(std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                       "pyslfp_migrating.txt");
    EXPECT_TRUE(file.is_open());
    MigratingFixture out;
    const auto size = SurfaceGrid::Index(lmax, lmax) + 1;
    std::vector<std::complex<double>>* block = nullptr;
    for (std::string line; std::getline(file, line);) {
      if (line.empty() || line.front() == '#') continue;
      std::istringstream row(line);
      auto head = std::string{};
      if (std::isalpha(static_cast<unsigned char>(line.front()))) {
        row >> head;
        if (head == "ice0") block = &out.ice0;
        if (head == "sea0") block = &out.sea0;
        if (head == "gradient") block = &out.gradient;
        if (head == "node") {
          auto& node = out.nodes.emplace_back();
          auto n = 0;
          row >> n >> node.t >> node.omega[0] >> node.omega[1] >>
              node.omega[2] >> node.oceanArea;
          block = &node.slc;
        }
        if (head == "ocean") block = &out.nodes.back().ocean;
        if (head == "misfits") {
          auto method = std::string{};
          row >> method;
          for (auto v = 0.0; row >> v;) out.misfits[method].push_back(v);
          block = nullptr;
        }
        if (block != nullptr) block->assign(size, 0.0);
        continue;
      }
      auto l = 0, m = 0;
      auto a = 0.0, b = 0.0;
      row >> l >> m >> a >> b;
      (*block)[SurfaceGrid::Index(l, m)] = ComplexFromRealPair({a, b}, m);
    }
    return out;
  }
};

double MaxAbsDiff(std::span<const std::complex<double>> a,
                  std::span<const std::complex<double>> b, double scale) {
  auto out = 0.0;
  for (auto i = std::size_t{0}; i < a.size(); ++i) {
    out = std::max(out, std::abs(a[i] * scale - b[i]));
  }
  return out;
}

double MaxAbs(std::span<const std::complex<double>> a) {
  auto out = 0.0;
  for (const auto& v : a) out = std::max(out, std::abs(v));
  return out;
}

}  // namespace

TEST(MigratingGate, RunMisfitAndCalibrationMatchPyslfp) {
  // pyslfp on its GLQ grid, which coincides with ours at this lmax; the
  // inputs are band-limited coefficients, so both codes flip the same
  // flotation cells.
  const auto body = Body::Make(kLmax, true);
  ViscoelasticOperator op(body.rheology, kLmax);
  const auto& grid = body.solver.Grid();
  ASSERT_EQ(grid.NumberOfLongitudes(), 2 * kLmax + 1);
  const auto fixture = MigratingFixture::Read(kLmax);
  ASSERT_EQ(fixture.nodes.size(), 6u);
  const auto length = body.solver.SolverScales().Length();
  auto field = [&](const std::vector<std::complex<double>>& si) {
    auto f = grid.Synthesise(si);
    for (auto& v : f) v /= length;
    return f;
  };
  const auto ice0 = field(fixture.ice0);
  const auto truth = body.solver.MakeState(ice0, field(fixture.sea0));
  const auto span = 10.0 * body.Kyr();
  auto times = std::vector<double>{};
  for (auto n = 0; n < 6; ++n) times.push_back(span * n / 5.0);
  const auto thickness = [&](double t) {
    auto out = ice0;
    for (auto& v : out) v *= std::clamp(1.0 - t / span, 0.0, 1.0);
    return out;
  };

  SeaLevelEvolution evolution(op, body.solver, truth, {.rtol = 1e-10});
  const auto run = evolution.RunMigrating(times, thickness);
  const auto frequency =
      body.solver.SolverScales().Factor(Dimensions{.time = -1});
  const auto a = body.solver.Radius() * length;
  // Measured 25 Oct 2026: the sea level change to 5e-12, the gradient to
  // 2e-14, the calibration histories to 1e-12 (1e-9 near convergence).
  constexpr auto tol = 1e-9;
  for (auto n = std::size_t{0}; n < run.size(); ++n) {
    const auto& want = fixture.nodes[n];
    EXPECT_NEAR(times[n] * body.solver.SolverScales().Time(), want.t,
                1e-6 * std::max(want.t, 1.0));
    EXPECT_LT(MaxAbsDiff(grid.Expand(run[n].oceanFunction), want.ocean, 1.0),
              1e-12)
        << "node " << n << ": the shorelines differ";
    EXPECT_NEAR(a * a * grid.Integrate(run[n].oceanFunction), want.oceanArea,
                1e-10 * want.oceanArea);
    if (n == 0) continue;
    const auto slc = grid.Expand(run[n].seaLevelChange);
    EXPECT_LT(MaxAbsDiff(slc, want.slc, length), tol * MaxAbs(want.slc))
        << "node " << n;
    const auto omegaScale =
        std::max({std::abs(want.omega[0]), std::abs(want.omega[1]),
                  std::abs(want.omega[2])});
    for (auto i = 0; i < 3; ++i) {
      EXPECT_NEAR(run[n].angularVelocity[i] * frequency, want.omega[i],
                  tol * omegaScale)
          << "node " << n << " omega " << i;
    }
  }

  // The misfit gradient against the reached sea level plus 50 m cos(theta).
  auto final = truth.seaLevel;
  for (auto i = std::size_t{0}; i < final.size(); ++i) {
    final[i] += run.back().seaLevelChange[i];
  }
  auto target = final;
  const auto tilt = grid.FromFunction([&](double theta, double) {
    return 50.0 * body.Metre() * std::cos(theta);
  });
  for (auto i = std::size_t{0}; i < target.size(); ++i) target[i] += tilt[i];
  const auto history = IceHistory::Migrating(thickness);
  const auto misfit = ComputeFinalSeaLevelMisfit(
      evolution, times, history, target, {}, true,
      {.method = AdjointMethod::Discrete, .rtol = 1e-10});
  EXPECT_LT(MaxAbsDiff(grid.Expand(misfit.gradient), fixture.gradient, length),
            1e-10 * MaxAbs(fixture.gradient));

  // The calibrations' convergence histories, run for run, while the misfit
  // is well above the solves' round-off.
  for (const auto& [name, method] :
       {std::pair{"fixed_point", CalibrationMethod::FixedPoint},
        std::pair{"gradient", CalibrationMethod::Gradient}}) {
    const auto guess = body.solver.MakeState(ice0, final);
    SeaLevelEvolution calibrating(op, body.solver, guess, {.rtol = 1e-10});
    const auto calibration = CalibrateInitialSeaLevel(
        calibrating, times, history, final,
        {.method = method,
         .iterations = 12,
         .rtol = 1e-9,
         .adjoint = {.method = AdjointMethod::Discrete, .rtol = 1e-10}});
    EXPECT_TRUE(calibration.converged) << name;
    const auto& want = fixture.misfits.at(name);
    EXPECT_EQ(calibration.misfits.size(), want.size()) << name;
    for (auto k = std::size_t{0};
         k < std::min(calibration.misfits.size(), want.size()); ++k) {
      if (want[k] < 1e-6 * want[0]) break;
      EXPECT_NEAR(calibration.misfits[k] * length, want[k], 1e-7 * want[k])
          << name << " run " << k + 1;
    }
    EXPECT_LT(MaxDiff(calibration.state.seaLevel, truth.seaLevel),
              1e-9 * MaxAbs(truth.seaLevel))
        << name;
  }
}

// -- migrating shorelines in 3D ----------------------------------------------

namespace {

constexpr auto kLmax3D = 8;

/// The marine cap at the 3D tests' degree, with the 3D operators over a
/// lateral pattern of the given peak (1.15 is a decade peak to peak; 0 is
/// the zero-lateral twin). Band-limited to degree 8, a 5 km cap peaks
/// below the 3.3 km that grounds on the 3 km ocean, so this one is 9 km.
struct Marine3D {
  Body body;
  MarineCap cap;
  SeaLevelSolver::State state;

  explicit Marine3D(int nodes = 5, double eta = 1e21)
      : body(Body::Make(kLmax3D, true, eta)),
        cap(MarineCap::Make(body, nodes, 9000.0)),
        state(body.solver.MakeState(cap.ice0, cap.sea0)) {}

  [[nodiscard]] Viscoelastic3D Operator(
      double peak, Viscoelastic3D::Options options = {}) const {
    return Viscoelastic3D(
        body.rheology, kLmax3D,
        [peak](double, double theta, double phi) {
          return peak * std::sin(theta) * std::cos(theta) * std::cos(phi);
        },
        options);
  }
  [[nodiscard]] std::function<std::vector<double>(double)> Thickness() const {
    return [this](double t) { return cap.Thickness(t); };
  }
};

}  // namespace

TEST(Migrating3D, ZeroLateralRunIsThe1DMigratingRun) {
  // With no lateral perturbation the combined fixed point converges on its
  // first pass and each pass's nonlinear solve is the 1D step's: the
  // migrating history, shorelines included, is the 1D one.
  const auto setting = Marine3D{};
  const auto& body = setting.body;
  ViscoelasticOperator op1(body.rheology, kLmax3D);
  SeaLevelEvolution evolution1(op1, body.solver, setting.state,
                               {.rtol = 1e-11});
  const auto reference =
      evolution1.RunMigrating(setting.cap.times, setting.Thickness());
  const auto op3 = setting.Operator(0.0);
  SeaLevelEvolution3D evolution3(op3, body.solver, setting.state,
                                 {.rtol = 1e-11, .coupled_rtol = 1e-10});
  const auto run =
      evolution3.RunMigrating(setting.cap.times, setting.Thickness());
  ASSERT_EQ(run.size(), reference.size());
  EXPECT_NE(reference.back().oceanFunction, setting.state.oceanFunction)
      << "no cell flipped";
  for (auto n = std::size_t{0}; n < run.size(); ++n) {
    EXPECT_TRUE(run[n].converged) << "node " << n;
    EXPECT_EQ(run[n].oceanFunction, reference[n].oceanFunction) << "node " << n;
    const auto top = MaxAbs(reference[n].seaLevelChange);
    EXPECT_LE(MaxDiff(run[n].seaLevelChange, reference[n].seaLevelChange),
              1e-8 * std::max(top, 1e-300))
        << "node " << n;
  }
}

TEST(Migrating3D, LateralRunFloodsTheCapAndConservesMass) {
  // Under lateral viscosity the melting cap floods, the explicit water
  // budget balances the grounded ice lost at every node, and the sea level
  // differs measurably from the zero-lateral twin's.
  const auto setting = Marine3D{};
  const auto& body = setting.body;
  const auto& state = setting.state;
  const auto op = setting.Operator(1.15);
  SeaLevelEvolution3D evolution(op, body.solver, state, {.rtol = 1e-11});
  const auto run =
      evolution.RunMigrating(setting.cap.times, setting.Thickness());
  const auto flat = setting.Operator(0.0);
  SeaLevelEvolution3D twin(flat, body.solver, state, {.rtol = 1e-11});
  const auto uniform =
      twin.RunMigrating(setting.cap.times, setting.Thickness());
  const auto& C0 = state.oceanFunction;
  for (auto n = std::size_t{1}; n < run.size(); ++n) {
    EXPECT_TRUE(run[n].converged) << "node " << n;
    const auto& Cn = run[n].oceanFunction;
    const auto ice = setting.cap.Thickness(setting.cap.times[n]);
    auto water = std::vector<double>(C0.size());
    auto grounded = std::vector<double>(C0.size());
    for (auto i = std::size_t{0}; i < C0.size(); ++i) {
      const auto sea = state.seaLevel[i] + run[n].seaLevelChange[i];
      water[i] = body.solver.WaterDensity() *
                 (Cn[i] * sea - C0[i] * state.seaLevel[i]);
      grounded[i] =
          body.solver.IceDensity() *
          ((1.0 - Cn[i]) * ice[i] - (1.0 - C0[i]) * state.iceThickness[i]);
    }
    const auto lost = body.Integrate(grounded);
    EXPECT_LT(std::abs(body.Integrate(water) + lost), 1e-8 * std::abs(lost))
        << "node " << n;
  }
  EXPECT_NE(run.back().oceanFunction, C0);
  const auto top = MaxAbs(uniform.back().seaLevelChange);
  const auto diff =
      MaxDiff(run.back().seaLevelChange, uniform.back().seaLevelChange);
  EXPECT_GT(diff, 1e-3 * top);
  EXPECT_LT(diff, 0.5 * top);
}

TEST(Migrating3D, AResumedRunIsTheUninterruptedOne) {
  const auto setting = Marine3D{};
  const auto& body = setting.body;
  const auto op = setting.Operator(1.15);
  auto checkpoints = std::map<std::size_t, SeaLevelEvolution3D::Checkpoint>{};
  auto options = SeaLevelEvolution3D::Options{.rtol = 1e-11};
  options.checkpoint_stride = 1;
  options.on_checkpoint = [&](std::size_t node,
                              const SeaLevelEvolution3D::Checkpoint& cp) {
    checkpoints.emplace(node, cp);
  };
  SeaLevelEvolution3D whole(op, body.solver, setting.state, options);
  const auto full = whole.RunMigrating(setting.cap.times, setting.Thickness());
  ASSERT_EQ(checkpoints.size(), setting.cap.times.size());
  SeaLevelEvolution3D again(op, body.solver, setting.state, {.rtol = 1e-11});
  const auto rest = std::vector<double>(setting.cap.times.begin() + 2,
                                        setting.cap.times.end());
  const auto resumed =
      again.ResumeMigrating(checkpoints.at(2), rest, setting.Thickness());
  ASSERT_EQ(resumed.size(), rest.size() - 1);
  for (auto k = std::size_t{0}; k < resumed.size(); ++k) {
    EXPECT_EQ(resumed[k].seaLevelChange, full[k + 3].seaLevelChange);
    EXPECT_EQ(resumed[k].oceanFunction, full[k + 3].oceanFunction);
    EXPECT_EQ(resumed[k].coupledIterations, full[k + 3].coupledIterations);
  }
}

TEST(Migrating3D, ZeroLateralKernelsMatchThe1DContinuousSweep) {
  // About the same migrating trajectory, the 3D sweep with no lateral
  // perturbation is the 1D continuous one: the ice kernels, the initial
  // sea level kernel and the spherical viscosity kernels agree.
  const auto setting = Marine3D{};
  const auto& body = setting.body;
  const auto& times = setting.cap.times;
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = body.Smooth(1.0, 6);
  measurements[2].displacementDual = body.Smooth(0.5, 7);

  ViscoelasticOperator op1(body.rheology, kLmax3D,
                           {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution evolution1(op1, body.solver, setting.state,
                               {.rtol = 1e-12, .record_stride = 1});
  const auto run1 = evolution1.RunMigrating(times, setting.Thickness());
  const auto trajectory1 = ForwardTrajectory::FromSnapshots(run1);
  const auto kernels1 =
      AdjointSolver(evolution1,
                    {.method = AdjointMethod::Continuous, .rtol = 1e-12})
          .IceHistoryKernels(times, measurements, &trajectory1,
                             OceanFunctionsOf(run1));

  const auto op3 =
      setting.Operator(0.0, {.memory_points = MemoryPointSet::Gll});
  SeaLevelEvolution3D evolution3(
      op3, body.solver, setting.state,
      {.rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto run3 = evolution3.RunMigrating(times, setting.Thickness());
  const auto trajectory3 = ForwardTrajectory3D::FromSnapshots(run3);
  const auto oceans3 = OceanFunctionsOf(run3);
  EXPECT_EQ(oceans3, OceanFunctionsOf(run1));
  const auto kernels3 =
      AdjointSolver3D(evolution3, {.rtol = 1e-12, .coupled_rtol = 1e-12})
          .IceHistoryKernels(times, measurements, &trajectory3, oceans3);

  // (the thickness kernels vanish once the cap is afloat: (1 - C_n) = 0)
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_GT(MaxAbs(kernels1.loadDuals[n]), 0.0) << "node " << n;
    EXPECT_LT(MaxDiff(kernels3.ice.loadDuals[n], kernels1.loadDuals[n]),
              1e-6 * MaxAbs(kernels1.loadDuals[n]))
        << "node " << n;
    EXPECT_LE(MaxDiff(kernels3.ice.thickness[n], kernels1.thickness[n]),
              1e-6 * MaxAbs(kernels1.thickness[n]))
        << "node " << n;
  }
  EXPECT_GT(MaxAbs(kernels1.initialSeaLevel), 0.0);
  EXPECT_LT(MaxDiff(kernels3.ice.initialSeaLevel, kernels1.initialSeaLevel),
            1e-6 * MaxAbs(kernels1.initialSeaLevel));
  const auto& coupling = op3.Coupling();
  const auto ngll = body.material.Mesh().NGll();
  for (auto k = std::size_t{0}; k < body.rheology.BranchCount(); ++k) {
    auto scale = 0.0, worst = 0.0;
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto full =
          static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
          coupling.NodeInElementOf(rn);
      scale = std::max(scale, std::abs(kernels1.itauKernel[k][full]));
      worst = std::max(worst, std::abs(kernels3.itauKernelSpherical[k][rn] -
                                       kernels1.itauKernel[k][full]));
    }
    EXPECT_GT(scale, 0.0);
    EXPECT_LT(worst, 1e-6 * scale) << "branch " << k;
  }
}

TEST(Migrating3D, LateralTrajectoryKernelsConvergeToCentralDifferences) {
  // With lateral viscosity the continuous sweep about the moving
  // trajectory predicts central differences of the 3D migrating run, and
  // its error falls with the step once the rheology is resolved (the 1D
  // test's stiff mantle and refinement).
  auto check = [](int nodes) {
    const auto setting = Marine3D(nodes, 1e22);
    const auto& body = setting.body;
    const auto& times = setting.cap.times;
    const auto op = setting.Operator(1.15);
    SeaLevelEvolution3D evolution(op, body.solver, setting.state,
                                  {.rtol = 1e-12, .coupled_rtol = 1e-12});
    const auto run = evolution.RunMigrating(times, setting.Thickness());
    const auto oceans = OceanFunctionsOf(run);
    EXPECT_NE(oceans.back(), oceans.front()) << "no cell flipped";
    auto measurement = Measurement{};
    measurement.seaLevelDual = body.Smooth(1.0, 6);
    const auto& g = measurement.seaLevelDual;
    auto deltaAt = std::map<int, std::vector<double>>{};
    for (auto q = 1; q <= 4; ++q) {
      deltaAt.emplace(q, body.Smooth(1.0 * body.Metre(), 100 + q));
    }
    const auto per = (nodes - 1) / 4;
    auto delta = [&](std::size_t n) -> const std::vector<double>* {
      if (n == 0 || n % per != 0) return nullptr;
      return &deltaAt.at(static_cast<int>(n / per));
    };
    auto functional = [&](double sign) {
      SeaLevelEvolution3D trial(op, body.solver, setting.state,
                                {.rtol = 1e-12, .coupled_rtol = 1e-12});
      const auto snaps = trial.RunMigrating(times, [&, sign](double t) {
        auto out = setting.cap.Thickness(t);
        for (auto n = std::size_t{0}; n < times.size(); ++n) {
          const auto* d = delta(n);
          if (t != times[n] || d == nullptr) continue;
          for (auto i = std::size_t{0}; i < out.size(); ++i) {
            out[i] += sign * (*d)[i];
          }
        }
        return out;
      });
      for (auto n = std::size_t{0}; n < snaps.size(); ++n) {
        EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped";
      }
      auto product = g;
      for (auto i = std::size_t{0}; i < g.size(); ++i) {
        product[i] *= snaps.back().seaLevelChange[i];
      }
      return body.Integrate(product);
    };
    const auto fd = 0.5 * (functional(1.0) - functional(-1.0));
    const auto kernels =
        AdjointSolver3D(evolution, {.rtol = 1e-12, .coupled_rtol = 1e-12})
            .IceHistoryKernels(times, {{times.size() - 1, measurement}},
                               nullptr, oceans);
    auto predicted = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      const auto* d = delta(n);
      if (d == nullptr) continue;
      auto product = kernels.ice.thickness[n];
      for (auto i = std::size_t{0}; i < product.size(); ++i) {
        product[i] *= (*d)[i];
      }
      predicted += body.Integrate(product);
    }
    return std::abs(predicted - fd) / std::abs(fd);
  };
  const auto coarse = check(5);
  const auto fine = check(9);
  EXPECT_LT(coarse, 5e-2);
  EXPECT_GT(coarse / fine, 2.0) << "coarse " << coarse << " fine " << fine;
}

TEST(Migrating3D, CalibrationRecoversTheTruth) {
  // Both methods recover a known initial sea level from the final one its
  // lateral migrating run reached. The gradient method's continuous kernel
  // is approximate, but it is linear in the residual, so the iteration's
  // fixed point is still the truth.
  const auto setting = Marine3D{};
  const auto& body = setting.body;
  const auto& grid = body.solver.Grid();
  const auto op = setting.Operator(1.15);
  const auto bump = grid.FromFunction([&](double theta, double phi) {
    return 200.0 * body.Metre() * std::sin(theta) * std::cos(phi);
  });
  auto sea = setting.cap.sea0;
  for (auto i = std::size_t{0}; i < sea.size(); ++i) sea[i] += bump[i];
  const auto truth = body.solver.MakeState(setting.cap.ice0, sea);
  SeaLevelEvolution3D forward(op, body.solver, truth, {.rtol = 1e-11});
  const auto run = forward.RunMigrating(setting.cap.times, setting.Thickness());
  EXPECT_NE(run.back().oceanFunction, truth.oceanFunction);
  auto final = truth.seaLevel;
  for (auto i = std::size_t{0}; i < final.size(); ++i) {
    final[i] += run.back().seaLevelChange[i];
  }
  const auto history = IceHistory::Migrating(setting.Thickness());
  const auto guess = body.solver.MakeState(truth.iceThickness, final);
  for (const auto method :
       {CalibrationMethod::FixedPoint, CalibrationMethod::Gradient}) {
    SeaLevelEvolution3D evolution(op, body.solver, guess, {.rtol = 1e-11});
    const auto calibration =
        CalibrateInitialSeaLevel(evolution, setting.cap.times, history, final,
                                 {.method = method,
                                  .iterations = 15,
                                  .rtol = 1e-9,
                                  .adjoint3D = {.rtol = 1e-11}});
    EXPECT_TRUE(calibration.converged)
        << static_cast<int>(method) << " in " << calibration.updates.size();
    EXPECT_LT(MaxDiff(calibration.state.seaLevel, truth.seaLevel),
              1e-7 * MaxAbs(truth.seaLevel))
        << static_cast<int>(method);
    EXPECT_EQ(calibration.state.oceanFunction, truth.oceanFunction);
    EXPECT_EQ(evolution.BackgroundState().seaLevel, calibration.state.seaLevel);
    EXPECT_FALSE(calibration.snapshots.back().oceanFunction.empty());
  }
}
