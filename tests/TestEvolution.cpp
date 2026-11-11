// The coupled GIA evolution: the elastic-rheology limit against the
// fingerprint solver (machine-tight, fully self-contained), qualitative
// relaxation, and the coupled comparison with pyslfp.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
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
                          {.rotation = rotation,
                           .rtol = 1e-13,
                           .record_surface = true});
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

        // The surface coefficients kept on the snapshot: the fields
        // above, and the tangential displacement's potential.
        ASSERT_TRUE(snapshot.surface.has_value());
        EXPECT_LT(worst(grid.Synthesise(snapshot.surface->U), u), 1e-9) << label;
        EXPECT_LT(worst(grid.Synthesise(snapshot.surface->Phi), phi), 1e-9)
            << label;
        const auto v = field(op.Surface(solutions, Component::V), toLength);
        EXPECT_GT(*std::ranges::max_element(v), 0.0);
        EXPECT_LT(worst(grid.Synthesise(snapshot.surface->V), v), 1e-9) << label;

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
