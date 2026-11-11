// The coupled 3D evolution: the combined (load + rotation + lateral)
// within-step fixed point. The zero-lateral limit: with delta = 0, the
// coupled history must equal the 1D coupled evolution's.

#include <gtest/gtest.h>

#include <numeric>

#include <cmath>
#include <filesystem>
#include <map>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kLmax = 8;

struct Setup {
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  GRDSolver::State state;
  std::vector<double> iceNd;

  static Setup Make() {
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
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
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
    return Setup{std::move(model),  std::move(material), std::move(rheology),
                 std::move(solver), std::move(state),    std::move(ice)};
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

}  // namespace

TEST(LateralGate, ZeroLateralCoupledEvolutionMatchesThe1DLoop) {
  auto setup = Setup::Make();
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{0.0, kyr, 2 * kyr, 4 * kyr};
  auto load = setup.MeltLoad(-0.1);
  auto directLoad = [&](double) { return load; };

  // The 1D coupled loop.
  ViscoelasticOperator op1(setup.rheology, kLmax);
  Evolution evolution1(op1, setup.solver, setup.state, {.rtol = 1e-11});
  const auto reference = evolution1.Run(times, directLoad);

  // The 3D coupled loop with no lateral perturbation.
  Viscoelastic3D op3(setup.rheology, kLmax,
                     [](double, double, double) { return 0.0; });
  Evolution3D evolution3(op3, setup.solver, setup.state,
                                 {.rtol = 1e-11, .coupled_rtol = 1e-10});
  const auto coupled = evolution3.Run(times, directLoad);

  ASSERT_EQ(coupled.size(), reference.size());
  for (auto node = std::size_t{0}; node < coupled.size(); ++node) {
    EXPECT_TRUE(coupled[node].converged);
    if (node > 0) EXPECT_LE(coupled[node].coupledIterations, 4);
    auto scale = 0.0, diff = 0.0;
    for (auto i = std::size_t{0}; i < reference[node].relativeSeaLevelChange.size();
         ++i) {
      scale = std::max(scale, std::abs(reference[node].relativeSeaLevelChange[i]));
      diff = std::max(diff, std::abs(coupled[node].relativeSeaLevelChange[i] -
                                     reference[node].relativeSeaLevelChange[i]));
    }
    EXPECT_LT(diff, 1e-8 * scale) << "node " << node;
    for (auto i = 0; i < 3; ++i) {
      EXPECT_NEAR(coupled[node].angularVelocity[i],
                  reference[node].angularVelocity[i],
                  1e-8 * std::abs(reference[node].angularVelocity[1]) + 1e-30)
          << "node " << node << " omega " << i;
    }
  }
}

namespace {

/// A tide of degrees 2 and 3, growing and turning in time (solver units).
std::function<std::vector<std::complex<double>>(double)> TestTide(
    const GRDSolver& solver, double kyr) {
  const auto& grid = solver.Grid();
  const auto amplitude = 5.0 / solver.SolverScales().Factor(PotentialDimensions);
  return [&grid, amplitude, kyr](double t) {
    auto psi = grid.ZeroCoefficients();
    const auto s = t / (4.0 * kyr);
    psi[SurfaceGrid::Index(2, 0)] = amplitude * (1.0 + s);
    psi[SurfaceGrid::Index(2, 1)] = amplitude * std::complex<double>(0.4, -0.3 * s);
    psi[SurfaceGrid::Index(2, 2)] = amplitude * std::complex<double>(-0.2 * s, 0.5);
    psi[SurfaceGrid::Index(3, 1)] = amplitude * std::complex<double>(0.3, 0.2);
    return psi;
  };
}

double WorstDeparture(std::span<const double> a, std::span<const double> b) {
  auto diff = 0.0, scale = 0.0;
  for (auto i = std::size_t{0}; i < a.size(); ++i) {
    diff = std::max(diff, std::abs(a[i] - b[i]));
    scale = std::max(scale, std::abs(b[i]));
  }
  return diff / scale;
}

}  // namespace

TEST(LateralGate, ZeroLateralForcedEvolutionMatchesThe1DLoop) {
  // Under a direct load and a prescribed tide, with an ocean and without
  // one, the 3D loop with no lateral perturbation is the 1D loop.
  auto setup = Setup::Make();
  const auto kyr = 1000.0 * 3.15576e7 / setup.solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, kyr, 2 * kyr, 4 * kyr};
  const auto held = setup.MeltLoad(-0.1);
  const auto tide = TestTide(setup.solver, kyr);
  auto load = [&](double t) {
    auto out = held;
    for (auto& v : out) v *= 1.0 + 0.2 * t / kyr;
    return out;
  };
  ViscoelasticOperator op1(setup.rheology, kLmax);
  Viscoelastic3D op3(setup.rheology, kLmax,
                     [](double, double, double) { return 0.0; });
  const auto dry = setup.solver.NoOceanState();
  const auto& wet = setup.state;
  for (const auto* background : {&dry, &wet}) {
    const auto label = background == &dry ? "no ocean" : "ocean";
    Evolution evolution1(op1, setup.solver, *background, {.rtol = 1e-11});
    const auto reference =
        evolution1.Run(times, Evolution::Forcing{.directLoad = load, .tide = tide});
    Evolution3D evolution3(op3, setup.solver, *background,
                           {.rtol = 1e-11, .coupled_rtol = 1e-10});
    const auto coupled = evolution3.Run(
        times, Evolution3D::Forcing{.directLoad = load, .tide = tide});
    ASSERT_EQ(coupled.size(), reference.size());
    for (auto node = std::size_t{0}; node < coupled.size(); ++node) {
      EXPECT_TRUE(coupled[node].converged) << label;
      EXPECT_LT(WorstDeparture(coupled[node].relativeSeaLevelChange,
                               reference[node].relativeSeaLevelChange),
                1e-8)
          << "node " << node << ", " << label;
      EXPECT_LT(WorstDeparture(coupled[node].displacement,
                               reference[node].displacement),
                1e-8)
          << "node " << node << ", " << label;
      EXPECT_LT(
          WorstDeparture(coupled[node].potential, reference[node].potential),
          1e-8)
          << "node " << node << ", " << label;
      EXPECT_LT(WorstDeparture(coupled[node].tangentialPotential,
                               reference[node].tangentialPotential),
                1e-8)
          << "node " << node << ", " << label;
      for (auto i = 0; i < 3; ++i) {
        EXPECT_NEAR(coupled[node].angularVelocity[i],
                    reference[node].angularVelocity[i],
                    1e-8 * std::abs(reference[node].angularVelocity[1]) + 1e-30)
            << "node " << node << ", " << label;
      }
    }
  }
}

TEST(Lateral, WithoutAnOceanIsTheDirectlySteppedBody) {
  // Under lateral viscosity, on a state with no ocean, the evolution is
  // the 3D stepper under the prescribed load and tide (and the
  // centrifugal potential of the rotation it finds): the oracle steps the
  // operator directly, with no sea level solver.
  auto setup = Setup::Make();
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
  const auto times = std::vector<double>{0.0, kyr, 2 * kyr, 3.5 * kyr};
  const auto held = setup.MeltLoad(-0.1);
  const auto tide = TestTide(solver, kyr);
  auto load = [&](double t) {
    auto out = held;
    for (auto& v : out) v *= 1.0 + 0.2 * t / kyr;
    return out;
  };
  Viscoelastic3D op(
      setup.rheology, kLmax, [](double, double theta, double phi) {
        return 0.8 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  using Complex = std::complex<double>;
  // The stepper's columns of a real field: its m >= 0 coefficients.
  auto columns = [](std::vector<Complex> coefficients,
                    double factor) -> Viscoelastic3D::LoadCallable3D {
    return [coefficients = std::move(coefficients), factor](
               int l, double) -> std::optional<std::vector<Complex>> {
      auto out = std::vector<Complex>(l + 1);
      for (auto m = 0; m <= l; ++m) {
        out[m] = factor * coefficients[SurfaceGrid::Index(l, m)];
      }
      return out;
    };
  };
  auto field = [&](const std::map<int, std::vector<Complex>>& perDegree,
                   double factor) {
    auto lm = grid.ZeroCoefficients();
    for (const auto& [l, c] : perDegree) {
      for (auto m = 0; m <= l; ++m) {
        lm[SurfaceGrid::Index(l, m)] = factor * c[m];
      }
    }
    return grid.Synthesise(lm);
  };

  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Full}) {
    const auto label = "rotation " + std::to_string(static_cast<int>(rotation));
    Evolution3D evolution(op, solver, solver.NoOceanState(),
                          {.rotation = rotation,
                           .rtol = 1e-12,
                           .coupled_rtol = 1e-11});
    const auto snapshots = evolution.Run(
        times, Evolution3D::Forcing{.directLoad = load, .tide = tide});
    ASSERT_EQ(snapshots.size(), times.size());

    auto channels = [&](std::size_t node) {
      auto psi = tide(times[node]);
      if (rotation != RotationalFeedback::None) {
        const auto spin =
            solver.CentrifugalCoefficients(snapshots[node].angularVelocity);
        for (auto i = std::size_t{0}; i < psi.size(); ++i) psi[i] += spin[i];
      }
      return std::pair{columns(grid.Expand(load(times[node])), toMatDensity),
                       columns(std::move(psi), 1.0 / toPotential)};
    };
    auto state = op.ZeroState();
    state.t = times.front() * toMatTime;
    auto now = channels(0);
    auto solutions = op.SolveElastic(state, now.first, now.second);
    op.PrimeStrain(state, now.first, now.second);
    for (auto node = std::size_t{0}; node < times.size(); ++node) {
      if (node > 0) {
        auto next = channels(node);
        const auto dt = (times[node] - times[node - 1]) * toMatTime;
        const auto middle = state.t + 0.5 * dt;
        auto across = [middle](Viscoelastic3D::LoadCallable3D before,
                               Viscoelastic3D::LoadCallable3D after)
            -> Viscoelastic3D::LoadCallable3D {
          return [before = std::move(before), after = std::move(after), middle](
                     int l, double t) { return (t < middle ? before : after)(l, t); };
        };
        auto result = op.Step(state, dt, Scheme::ExponentialTrapezoid,
                              across(now.first, next.first),
                              across(now.second, next.second));
        ASSERT_TRUE(result.info.converged);
        state = std::move(result.state);
        solutions = std::move(result.solutions);
        now = std::move(next);
      }
      const auto& snapshot = snapshots[node];
      EXPECT_TRUE(snapshot.converged) << label;
      EXPECT_LT(WorstDeparture(snapshot.displacement,
                               field(op.Surface(solutions, Component::U),
                                     toLength)),
                1e-7)
          << "node " << node << ", " << label;
      EXPECT_LT(WorstDeparture(snapshot.potential,
                               field(op.Surface(solutions, Component::Phi),
                                     toPotential)),
                1e-7)
          << "node " << node << ", " << label;
      EXPECT_LT(WorstDeparture(snapshot.tangentialPotential,
                               field(op.Surface(solutions, Component::V),
                                     toLength)),
                1e-7)
          << "node " << node << ", " << label;
    }
  }
}

TEST(Lateral, CoupledLateralEvolutionRunsAndConservesMass) {
  auto setup = Setup::Make();
  const auto& scales = setup.solver.SolverScales();
  const auto& grid = setup.solver.Grid();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{0.0, kyr, 2 * kyr, 3 * kyr};
  auto load = setup.MeltLoad(-0.1);
  auto directLoad = [&](double) { return load; };

  // A strong lateral pattern: one decade peak to peak.
  Viscoelastic3D op(
      setup.rheology, kLmax, [](double, double theta, double phi) {
        return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  Evolution3D evolution(op, setup.solver, setup.state, {});
  const auto run = evolution.Run(times, directLoad);

  // The zero-lateral twin, for comparison of the physics.
  Viscoelastic3D flat(setup.rheology, kLmax,
                      [](double, double, double) { return 0.0; });
  Evolution3D twin(flat, setup.solver, setup.state, {});
  const auto uniform = twin.Run(times, directLoad);

  auto maxRelDiff = 0.0;
  for (auto node = std::size_t{1}; node < run.size(); ++node) {
    EXPECT_TRUE(run[node].converged) << "node " << node;

    // Exact ocean mass conservation per node, lateral viscosity or not.
    auto gain = std::vector<double>(grid.FieldSize());
    for (auto i = std::size_t{0}; i < gain.size(); ++i) {
      gain[i] = setup.solver.WaterDensity() * setup.state.oceanFunction[i] *
                run[node].relativeSeaLevelChange[i];
    }
    const auto a = setup.solver.Radius();
    const auto gained = a * a * grid.Integrate(gain);
    const auto removed = -a * a * grid.Integrate(load);
    EXPECT_NEAR(gained, removed, 1e-7 * std::abs(removed)) << "node " << node;

    // The lateral viscosity changes the physics measurably.
    auto scale = 0.0, diff = 0.0;
    for (auto i = std::size_t{0}; i < run[node].relativeSeaLevelChange.size(); ++i) {
      scale = std::max(scale, std::abs(uniform[node].relativeSeaLevelChange[i]));
      diff = std::max(diff, std::abs(run[node].relativeSeaLevelChange[i] -
                                     uniform[node].relativeSeaLevelChange[i]));
    }
    maxRelDiff = std::max(maxRelDiff, diff / scale);
  }
  EXPECT_GT(maxRelDiff, 1e-3);
  EXPECT_LT(maxRelDiff, 0.5);
}

TEST(Lateral, AResumedCoupledRunIsTheUninterruptedOne) {
  auto setup = Setup::Make();
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{0.0, kyr, 2 * kyr, 3 * kyr};
  auto load = setup.MeltLoad(-0.1);
  auto directLoad = [&](double) { return load; };
  Viscoelastic3D op(
      setup.rheology, kLmax, [](double, double theta, double phi) {
        return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });

  auto checkpoints =
      std::map<std::size_t, Evolution3D::Checkpoint>{};
  auto options = Evolution3D::Options{};
  options.checkpoint_stride = 1;
  options.on_checkpoint = [&](std::size_t node,
                              const Evolution3D::Checkpoint& cp) {
    checkpoints.emplace(node, cp);
  };
  Evolution3D whole(op, setup.solver, setup.state, options);
  const auto full = whole.Run(times, directLoad);
  ASSERT_EQ(checkpoints.size(), times.size());

  Evolution3D resumed(op, setup.solver, setup.state, {});
  const auto tail = resumed.Resume(
      checkpoints.at(1),
      std::vector<double>(times.begin() + 1, times.end()), directLoad);
  ASSERT_EQ(tail.size(), times.size() - 2);
  for (auto k = std::size_t{0}; k < tail.size(); ++k) {
    const auto& a = full[2 + k];
    const auto& b = tail[k];
    EXPECT_EQ(a.relativeSeaLevelChange, b.relativeSeaLevelChange) << "node " << 2 + k;
    EXPECT_EQ(a.displacement, b.displacement) << "node " << 2 + k;
    EXPECT_EQ(a.potential, b.potential) << "node " << 2 + k;
    EXPECT_EQ(a.angularVelocity, b.angularVelocity) << "node " << 2 + k;
    EXPECT_EQ(a.coupledIterations, b.coupledIterations) << "node " << 2 + k;
  }
}

namespace {

/// A point source in the solver's units: a moment of `newtonMetres` times
/// the pattern, at `depthKm` below the surface.
PointSource SourceAt(const Setup& setup, double depthKm, double colatitude,
                     double longitude, SphericalTensor pattern,
                     double newtonMetres = 1e20) {
  const auto& sle = setup.solver.SolverScales();
  const auto& material = setup.model.ModelScales();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto surface =
      setup.material.Radius() * material.Length() / sle.Length();
  const auto moment = newtonMetres / sle.Factor(energy);
  for (auto* c : {&pattern.rr, &pattern.tt, &pattern.pp, &pattern.rt,
                  &pattern.rp, &pattern.tp}) {
    *c *= moment;
  }
  return {.at = {.radius = surface - depthKm * 1e3 / sle.Length(),
                 .colatitude = colatitude,
                 .longitude = longitude},
          .moment = pattern};
}

double MaxAbsOf(std::span<const double> f) {
  auto out = 0.0;
  for (auto v : f) out = std::max(out, std::abs(v));
  return out;
}

}  // namespace

TEST(PointSourceEvolution, TheElasticStartIsTheDirectDegreeSolves) {
  // A point source on a body without an ocean and without rotational
  // feedback: the response at the first node is the unrelaxed degree
  // systems solved under the source's forces, in U, V, phi and the
  // toroidal W. An isotropic source makes no toroidal motion.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto& grid = solver.Grid();
  const auto& sle = solver.SolverScales();
  const auto& material = setup.model.ModelScales();
  auto ratio = [&](Dimensions dims) {
    return material.Factor(dims) / sle.Factor(dims);
  };
  const auto toLength = ratio(Dimensions{.length = 1});
  const auto toPotential = ratio(PotentialDimensions);
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto times = std::vector<double>{0.0, kyr};
  Viscoelastic3D op(setup.rheology, kLmax,
                    [](double, double, double) { return 0.0; });
  using Complex = std::complex<double>;

  const auto thrust = SourceAt(
      setup, 400.0, 1.1, 4.0,
      {.rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6});
  const auto explosion =
      SourceAt(setup, 400.0, 1.1, 4.0, {.rr = 1.0, .tt = 1.0, .pp = 1.0});
  for (const auto* source : {&thrust, &explosion}) {
    Evolution3D evolution(op, solver, solver.NoOceanState(),
                          {.rotation = RotationalFeedback::None,
                           .rtol = 1e-12,
                           .coupled_rtol = 1e-12});
    const auto snapshots = evolution.Run(
        times, Evolution3D::Forcing{.sources = [&](double) {
          return std::vector<PointSource>{*source};
        }});
    ASSERT_EQ(snapshots.size(), times.size());
    EXPECT_TRUE(snapshots[0].converged);
    EXPECT_TRUE(snapshots[1].converged);
    // the elastic response on its own is the run's first node
    const auto alone = evolution.Instantaneous(
        Evolution3D::Forcing{.sources = [&](double) {
          return std::vector<PointSource>{*source};
        }});
    EXPECT_LT(WorstDeparture(alone.displacement, snapshots[0].displacement),
              1e-12);

    // the same source in the material's units, and the direct solves
    auto inMaterial = *source;
    const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
    inMaterial.at.radius /= toLength;
    for (auto* c : {&inMaterial.moment.rr, &inMaterial.moment.tt,
                    &inMaterial.moment.pp, &inMaterial.moment.rt,
                    &inMaterial.moment.rp, &inMaterial.moment.tp}) {
      *c /= ratio(energy);
    }
    const auto sph = [&](int l) -> const DegreeOperator& {
      return op.UnrelaxedSpheroidal(l);
    };
    const auto tor = [&](int l) -> const ToroidalOperator& {
      return op.UnrelaxedToroidal(l);
    };
    const auto forces =
        PointSourceForces(setup.material, kLmax, sph, tor, inMaterial);
    auto U = grid.ZeroCoefficients(), V = grid.ZeroCoefficients();
    auto Phi = grid.ZeroCoefficients(), W = grid.ZeroCoefficients();
    auto solve = [](const auto& system, const std::vector<Complex>& b, int l) {
      const auto ndof = static_cast<std::size_t>(system.DofCount());
      auto re = std::vector<double>(b.size()), im = re;
      for (auto i = std::size_t{0}; i < b.size(); ++i) {
        re[i] = b[i].real();
        im[i] = b[i].imag();
      }
      const auto xr = system.Solve(re, l + 1);
      const auto xi = system.Solve(im, l + 1);
      auto x = std::vector<Complex>(b.size());
      for (auto i = std::size_t{0}; i < b.size(); ++i) x[i] = {xr[i], xi[i]};
      (void)ndof;
      return x;
    };
    for (auto l = 0; l <= kLmax; ++l) {
      const auto& system = sph(l);
      const auto ndof = static_cast<std::size_t>(system.DofCount());
      const auto x = solve(system, forces.spheroidal.at(l), l);
      for (auto m = 0; m <= l; ++m) {
        const auto i = SurfaceGrid::Index(l, m);
        auto at = [&](Component c) -> Complex {
          const auto dof = system.SurfaceDof(c);
          return dof >= 0 ? x[static_cast<std::size_t>(m) * ndof + dof] : 0.0;
        };
        U[i] = toLength * at(Component::U);
        V[i] = toLength * at(Component::V);
        Phi[i] = toPotential * at(Component::Phi);
      }
      if (l >= 1) {
        const auto& twist = tor(l);
        const auto tdof = static_cast<std::size_t>(twist.DofCount());
        const auto w = solve(twist, forces.toroidal.at(l), l);
        for (auto m = 0; m <= l; ++m) {
          W[SurfaceGrid::Index(l, m)] =
              toLength * w[static_cast<std::size_t>(m) * tdof + twist.SurfaceDof()];
        }
      }
    }
    const auto& start = snapshots[0];
    const auto label = source == &explosion ? "explosion" : "thrust";
    EXPECT_GT(MaxAbsOf(start.displacement), 0.0) << label;
    EXPECT_LT(WorstDeparture(start.displacement, grid.Synthesise(U)), 1e-8)
        << label;
    EXPECT_LT(WorstDeparture(start.tangentialPotential, grid.Synthesise(V)), 1e-8)
        << label;
    EXPECT_LT(WorstDeparture(start.potential, grid.Synthesise(Phi)), 1e-8)
        << label;
    ASSERT_FALSE(start.toroidalPotential.empty());
    if (source == &explosion) {
      EXPECT_LT(MaxAbsOf(start.toroidalPotential),
                1e-10 * MaxAbsOf(start.tangentialPotential));
    } else {
      EXPECT_GT(MaxAbsOf(start.toroidalPotential), 0.0);
      EXPECT_LT(WorstDeparture(start.toroidalPotential, grid.Synthesise(W)),
                1e-8);
    }
  }
}

TEST(PointSourceEvolution, RelaxationIsLinearAndConvergesWithTheStep) {
  // A source held from the start on a viscoelastic body with an ocean
  // and the rotational feedback: the response is linear in the moment,
  // it relaxes (the late field differs from the elastic one), and under
  // halving of the steps the field at the end converges at the scheme's
  // second order.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  Viscoelastic3D op(setup.rheology, kLmax,
                    [](double, double, double) { return 0.0; });
  const SphericalTensor pattern{
      .rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6};
  auto runTo = [&](int steps, double scale) {
    auto times = std::vector<double>{};
    // steps below the mantle's relaxation time at the finer spacings
    for (auto n = 0; n <= steps; ++n) times.push_back(1.0 * kyr * n / steps);
    Evolution3D evolution(op, solver, setup.state,
                          {.rtol = 1e-12, .coupled_rtol = 1e-12});
    const auto source =
        SourceAt(setup, 400.0, 1.1, 4.0, pattern, scale * 1e22);
    auto snapshots = evolution.Run(
        times, Evolution3D::Forcing{.sources = [&](double) {
          return std::vector<PointSource>{source};
        }});
    for (const auto& s : snapshots) EXPECT_TRUE(s.converged);
    return snapshots;
  };
  const auto coarse = runTo(2, 1.0);
  const auto doubled = runTo(2, 2.0);
  auto twice = coarse.back().displacement;
  for (auto& v : twice) v *= 2.0;
  EXPECT_LT(WorstDeparture(doubled.back().displacement, twice), 1e-9);
  EXPECT_GT(MaxAbsOf(coarse.back().relativeSeaLevelChange), 0.0);
  // it relaxes
  EXPECT_GT(WorstDeparture(coarse.back().displacement, coarse.front().displacement),
            1e-2);

  const auto medium = runTo(4, 1.0);
  const auto fine = runTo(8, 1.0);
  const auto finest = runTo(16, 1.0);
  for (auto field : {&Evolution3D::Snapshot::displacement,
                     &Evolution3D::Snapshot::tangentialPotential,
                     &Evolution3D::Snapshot::toroidalPotential}) {
    const auto d1 = WorstDeparture(medium.back().*field, fine.back().*field);
    const auto d2 = WorstDeparture(fine.back().*field, finest.back().*field);
    EXPECT_GT(d1 / d2, 3.0) << d1 << " " << d2;
    EXPECT_LT(d1 / d2, 5.0) << d1 << " " << d2;
  }
}

TEST(PointSourceEvolution, AStressGlutDrivesTheEvolution) {
  // A stress glut on a shell, a smooth tensor field: the evolution's
  // elastic start is the unrelaxed toroidal and spheroidal systems solved
  // under the shell's forces, and a uniform isotropic shell (a spherical
  // pressure source) moves the surface at degree 0 alone.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto& grid = solver.Grid();
  const auto& sle = solver.SolverScales();
  const auto& material = setup.model.ModelScales();
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto times = std::vector<double>{0.0, kyr};
  Viscoelastic3D op(setup.rheology, kLmax,
                    [](double, double, double) { return 0.0; });
  const SurfaceGrid shellGrid(kLmax, {.nmax = 2});
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e20 / sle.Factor(energy);
  const auto radius = setup.material.Radius() * material.Length() / sle.Length() -
                      400e3 / sle.Length();
  auto run = [&](const StressGlut& shell) {
    Evolution3D evolution(op, solver, solver.NoOceanState(),
                          {.rotation = RotationalFeedback::None,
                           .rtol = 1e-12,
                           .coupled_rtol = 1e-12});
    return evolution.Run(times, Evolution3D::Forcing{.stressGluts = [&](double) {
      return std::vector<StressGlut>{shell};
    }});
  };

  // the uniform isotropic shell
  auto uniform = StressGlut{.radius = radius, .grid = &shellGrid, .moment = {}};
  const auto size = shellGrid.FieldSize();
  uniform.moment.rr.assign(size, unit);
  uniform.moment.tt.assign(size, unit);
  uniform.moment.pp.assign(size, unit);
  uniform.moment.rt.assign(size, 0.0);
  uniform.moment.rp.assign(size, 0.0);
  uniform.moment.tp.assign(size, 0.0);
  const auto radial = run(uniform);
  const auto& u = radial[0].displacement;
  const auto mean = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
  // an expansion source pushes the surface outwards
  EXPECT_GT(mean, 0.0);
  auto spread = 0.0;
  for (auto v : u) spread = std::max(spread, std::abs(v - mean));
  EXPECT_LT(spread, 1e-9 * std::abs(mean));
  EXPECT_LT(MaxAbsOf(radial[0].tangentialPotential), 1e-9 * std::abs(mean));

  // a smooth shear pattern: the start against the direct solves
  auto smooth = uniform;
  auto k = std::size_t{0};
  for (auto theta : shellGrid.CoLatitudes()) {
    for (auto phi : shellGrid.Longitudes()) {
      const auto a = std::sin(theta) * std::cos(phi);
      const auto b = std::cos(theta);
      // (a r + b theta) (x) (a r + b theta), a band-limited-ish field
      smooth.moment.rr[k] = unit * a * a;
      smooth.moment.tt[k] = unit * b * b * std::sin(theta) * std::sin(theta);
      smooth.moment.pp[k] = unit * 0.3 * std::sin(theta) * std::sin(theta);
      smooth.moment.rt[k] = unit * a * b * std::sin(theta);
      smooth.moment.rp[k] = unit * 0.2 * std::sin(theta) * std::sin(phi);
      smooth.moment.tp[k] = unit * 0.1 * std::sin(theta) * std::sin(theta) *
                            std::sin(2.0 * phi);
      ++k;
    }
  }
  const auto snapshots = run(smooth);
  auto inMaterial = smooth;
  inMaterial.radius *= sle.Length() / material.Length();
  const auto toMat = sle.Factor(energy) / material.Factor(energy);
  for (auto* c : {&inMaterial.moment.rr, &inMaterial.moment.tt,
                  &inMaterial.moment.pp, &inMaterial.moment.rt,
                  &inMaterial.moment.rp, &inMaterial.moment.tp}) {
    for (auto& v : *c) v *= toMat;
  }
  const auto forces = StressGlutForces(
      setup.material, kLmax,
      [&](int l) -> const DegreeOperator& { return op.UnrelaxedSpheroidal(l); },
      [&](int l) -> const ToroidalOperator& { return op.UnrelaxedToroidal(l); },
      inMaterial);
  using Complex = std::complex<double>;
  const auto toLength = material.Length() / sle.Length();
  auto U = grid.ZeroCoefficients(), W = grid.ZeroCoefficients();
  auto solve = [](const auto& system, const std::vector<Complex>& b, int l) {
    auto re = std::vector<double>(b.size()), im = re;
    for (auto i = std::size_t{0}; i < b.size(); ++i) {
      re[i] = b[i].real();
      im[i] = b[i].imag();
    }
    const auto xr = system.Solve(re, l + 1);
    const auto xi = system.Solve(im, l + 1);
    auto x = std::vector<Complex>(b.size());
    for (auto i = std::size_t{0}; i < b.size(); ++i) x[i] = {xr[i], xi[i]};
    return x;
  };
  for (auto l = 0; l <= kLmax; ++l) {
    const auto& system = op.UnrelaxedSpheroidal(l);
    const auto ndof = static_cast<std::size_t>(system.DofCount());
    const auto x = solve(system, forces.spheroidal.at(l), l);
    for (auto m = 0; m <= l; ++m) {
      U[SurfaceGrid::Index(l, m)] =
          toLength * x[static_cast<std::size_t>(m) * ndof +
                       system.SurfaceDof(Component::U)];
    }
    if (l >= 1) {
      const auto& twist = op.UnrelaxedToroidal(l);
      const auto tdof = static_cast<std::size_t>(twist.DofCount());
      const auto w = solve(twist, forces.toroidal.at(l), l);
      for (auto m = 0; m <= l; ++m) {
        W[SurfaceGrid::Index(l, m)] =
            toLength * w[static_cast<std::size_t>(m) * tdof + twist.SurfaceDof()];
      }
    }
  }
  EXPECT_TRUE(snapshots[0].converged);
  EXPECT_GT(MaxAbsOf(snapshots[0].toroidalPotential), 0.0);
  EXPECT_LT(WorstDeparture(snapshots[0].displacement, grid.Synthesise(U)), 1e-8);
  EXPECT_LT(WorstDeparture(snapshots[0].toroidalPotential, grid.Synthesise(W)),
            1e-8);
}

#include "sl3dpp/ToroidalViscoelastic.hpp"

TEST(PointSourceEvolution, TheRadialToroidalStepperMatchesThe3DClass) {
  // The radial model's toroidal systems stepped on their own under a
  // point source's toroidal forces: the surface W at every node is the
  // 3D class's with no lateral structure, for both schemes.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto& grid = solver.Grid();
  const auto& sle = solver.SolverScales();
  const auto& material = setup.model.ModelScales();
  const auto toLength = material.Length() / sle.Length();
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto times = std::vector<double>{0.0, 0.3 * kyr, 0.8 * kyr, 2.0 * kyr};
  const auto source = SourceAt(
      setup, 400.0, 1.1, 4.0,
      {.rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6});
  Viscoelastic3D op3(setup.rheology, kLmax,
                     [](double, double, double) { return 0.0; });
  ViscoelasticOperator op1(setup.rheology, kLmax);
  const ToroidalViscoelastic toroidal(op1);

  // the source's toroidal forces as real columns, material units
  auto inMaterial = source;
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  inMaterial.at.radius /= toLength;
  for (auto* c : {&inMaterial.moment.rr, &inMaterial.moment.tt,
                  &inMaterial.moment.pp, &inMaterial.moment.rt,
                  &inMaterial.moment.rp, &inMaterial.moment.tp}) {
    *c *= sle.Factor(energy) / material.Factor(energy);
  }
  const auto complexForces = PointSourceForces(
      setup.material, kLmax,
      [&](int l) -> const DegreeOperator& { return op3.UnrelaxedSpheroidal(l); },
      [&](int l) -> const ToroidalOperator& { return toroidal.Unrelaxed(l); },
      inMaterial);
  auto forces = ToroidalViscoelastic::Fields{};
  for (const auto& [l, b] : complexForces.toroidal) {
    const auto ndof = static_cast<std::size_t>(toroidal.Unrelaxed(l).DofCount());
    auto real = std::vector<double>(ndof * 2 * (l + 1));
    for (auto m = 0; m <= l; ++m) {
      for (auto d = std::size_t{0}; d < ndof; ++d) {
        real[m * ndof + d] = b[m * ndof + d].real();
        real[(l + 1 + m) * ndof + d] = b[m * ndof + d].imag();
      }
    }
    forces.emplace(l, std::move(real));
  }
  auto fieldOf = [&](const ToroidalViscoelastic::Fields& solutions) {
    auto lm = grid.ZeroCoefficients();
    for (const auto& [l, w] : toroidal.Surface(solutions)) {
      for (auto m = 0; m <= l; ++m) {
        lm[SurfaceGrid::Index(l, m)] =
            toLength * std::complex<double>(w[m], w[l + 1 + m]);
      }
    }
    return grid.Synthesise(lm);
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    Evolution3D evolution(op3, solver, solver.NoOceanState(),
                          {.scheme = scheme,
                           .rotation = RotationalFeedback::None,
                           .rtol = 1e-12,
                           .coupled_rtol = 1e-12});
    const auto snapshots = evolution.Run(
        times, Evolution3D::Forcing{.sources = [&](double) {
          return std::vector<PointSource>{source};
        }});
    auto state = toroidal.ZeroState();
    state.t = times.front() * sle.Time() / material.Time();
    auto solutions = toroidal.SolveElastic(state, forces);
    toroidal.SetStrain(state, solutions);
    for (auto node = std::size_t{0}; node < times.size(); ++node) {
      if (node > 0) {
        const auto dt =
            (times[node] - times[node - 1]) * sle.Time() / material.Time();
        auto stepped = toroidal.Step(state, dt, scheme, forces);
        state = std::move(stepped.first);
        solutions = std::move(stepped.second);
      }
      EXPECT_GT(MaxAbsOf(snapshots[node].toroidalPotential), 0.0);
      EXPECT_LT(WorstDeparture(fieldOf(solutions),
                               snapshots[node].toroidalPotential),
                1e-8)
          << "node " << node << ", scheme " << static_cast<int>(scheme);
    }
    // it relaxes
    EXPECT_GT(WorstDeparture(snapshots.back().toroidalPotential,
                             snapshots.front().toroidalPotential),
              1e-3);
  }
}

TEST(PointSourceEvolution, TheRadialEvolutionMatchesThe3DClass) {
  // Point sources and a stress glut on a shell in the radial evolution,
  // with an ocean and the rotational feedback and without either: every
  // field at every node is the 3D class's with no lateral structure, for
  // both schemes.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto& sle = solver.SolverScales();
  const auto& material = setup.model.ModelScales();
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto times = std::vector<double>{0.0, 0.3 * kyr, 0.8 * kyr, 2.0 * kyr};
  const auto thrust = SourceAt(
      setup, 400.0, 1.1, 4.0,
      {.rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6},
      1e22);
  const auto second = SourceAt(
      setup, 150.0, 2.2, 1.0,
      {.rr = -0.4, .tt = 0.8, .pp = -0.4, .rt = 0.5, .rp = 0.1, .tp = -0.3},
      5e21);
  const SurfaceGrid shellGrid(kLmax, {.nmax = 2});
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  auto glut = StressGlut{
      .radius = setup.material.Radius() * material.Length() / sle.Length() -
                600e3 / sle.Length(),
      .grid = &shellGrid,
      .moment = {}};
  for (auto theta : shellGrid.CoLatitudes()) {
    for (auto phi : shellGrid.Longitudes()) {
      const auto s = std::sin(theta);
      glut.moment.rr.push_back(unit * s * s * std::cos(phi) * std::cos(phi));
      glut.moment.tt.push_back(unit * 0.5 * s * s);
      glut.moment.pp.push_back(unit * 0.3 * s * s);
      glut.moment.rt.push_back(unit * s * std::cos(theta) * std::cos(phi));
      glut.moment.rp.push_back(unit * 0.2 * s * std::sin(phi));
      glut.moment.tp.push_back(unit * 0.1 * s * s * std::sin(2.0 * phi));
    }
  }
  // the second source acts from the second node on
  auto sourcesAt = [&](double t) {
    auto out = std::vector<PointSource>{thrust};
    if (t >= times[1]) out.push_back(second);
    return out;
  };
  auto glutsAt = [&](double) { return std::vector<StressGlut>{glut}; };

  ViscoelasticOperator op1(setup.rheology, kLmax);
  Viscoelastic3D op3(setup.rheology, kLmax,
                     [](double, double, double) { return 0.0; });
  const auto dry = solver.NoOceanState();
  const auto& wet = setup.state;
  for (const auto* background : {&wet, &dry})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto rotation = background == &wet ? RotationalFeedback::Full
                                             : RotationalFeedback::None;
    const auto label = std::string(background == &dry ? "no ocean" : "ocean") +
                       ", scheme " + std::to_string(static_cast<int>(scheme));
    Evolution evolution1(op1, solver, *background,
                         {.scheme = scheme, .rotation = rotation, .rtol = 1e-12});
    const auto radial = evolution1.Run(
        times,
        Evolution::Forcing{.sources = sourcesAt, .stressGluts = glutsAt});
    Evolution3D evolution3(op3, solver, *background,
                           {.scheme = scheme,
                            .rotation = rotation,
                            .rtol = 1e-12,
                            .coupled_rtol = 1e-12});
    const auto reference = evolution3.Run(
        times,
        Evolution3D::Forcing{.sources = sourcesAt, .stressGluts = glutsAt});
    ASSERT_EQ(radial.size(), reference.size());
    for (auto node = std::size_t{0}; node < radial.size(); ++node) {
      const auto where = "node " + std::to_string(node) + ", " + label;
      EXPECT_TRUE(radial[node].converged) << where;
      EXPECT_LT(WorstDeparture(radial[node].relativeSeaLevelChange,
                               reference[node].relativeSeaLevelChange),
                1e-8)
          << where;
      EXPECT_LT(WorstDeparture(radial[node].displacement,
                               reference[node].displacement),
                1e-8)
          << where;
      EXPECT_LT(WorstDeparture(radial[node].potential, reference[node].potential),
                1e-8)
          << where;
      EXPECT_LT(WorstDeparture(radial[node].tangentialPotential,
                               reference[node].tangentialPotential),
                1e-8)
          << where;
      ASSERT_FALSE(radial[node].toroidalPotential.empty()) << where;
      EXPECT_LT(WorstDeparture(radial[node].toroidalPotential,
                               reference[node].toroidalPotential),
                1e-8)
          << where;
      for (auto i = 0; i < 3; ++i) {
        EXPECT_NEAR(radial[node].angularVelocity[i],
                    reference[node].angularVelocity[i],
                    1e-8 * std::abs(reference[node].angularVelocity[1]) + 1e-30)
            << where;
      }
    }
  }
}

TEST(PointSourceEvolution, TheFactoredRouteIsTheGeneralOne) {
  // Options::factor_sources: on a body without an ocean and without
  // rotational feedback, the run by the sources' unit radial solutions
  // is the general run in every field at every node. Two point sources
  // at different depths, one switched on mid-run and one whose tensor
  // changes its pattern through time, and a stress glut on a shell
  // growing in time; both schemes.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto& sle = solver.SolverScales();
  const auto& material = setup.model.ModelScales();
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto times = std::vector<double>{0.0, 0.3 * kyr, 0.8 * kyr, 2.0 * kyr};
  const auto thrust = SourceAt(
      setup, 400.0, 1.1, 4.0,
      {.rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6},
      1e22);
  const auto second = SourceAt(
      setup, 150.0, 2.2, 1.0,
      {.rr = -0.4, .tt = 0.8, .pp = -0.4, .rt = 0.5, .rp = 0.1, .tp = -0.3},
      5e21);
  const SurfaceGrid shellGrid(kLmax, {.nmax = 2});
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  auto glut = StressGlut{
      .radius = setup.material.Radius() * material.Length() / sle.Length() -
                600e3 / sle.Length(),
      .grid = &shellGrid,
      .moment = {}};
  for (auto theta : shellGrid.CoLatitudes()) {
    for (auto phi : shellGrid.Longitudes()) {
      const auto s = std::sin(theta);
      glut.moment.rr.push_back(unit * s * s * std::cos(phi) * std::cos(phi));
      glut.moment.tt.push_back(unit * 0.5 * s * s);
      glut.moment.pp.push_back(unit * 0.3 * s * s);
      glut.moment.rt.push_back(unit * s * std::cos(theta) * std::cos(phi));
      glut.moment.rp.push_back(unit * 0.2 * s * std::sin(phi));
      glut.moment.tp.push_back(unit * 0.1 * s * s * std::sin(2.0 * phi));
    }
  }
  auto sourcesAt = [&](double t) {
    // the first changes its pattern: its shear parts grow, the rest hold
    auto first = thrust;
    const auto growth = 1.0 + t / times.back();
    first.moment.rt *= growth;
    first.moment.tp *= growth;
    auto out = std::vector<PointSource>{first};
    if (t >= times[1]) out.push_back(second);
    return out;
  };
  auto glutsAt = [&](double t) {
    auto at = glut;
    const auto factor = 0.5 + t / times.back();
    for (auto* c : {&at.moment.rr, &at.moment.tt, &at.moment.pp, &at.moment.rt,
                    &at.moment.rp, &at.moment.tp}) {
      for (auto& v : *c) v *= factor;
    }
    return std::vector<StressGlut>{at};
  };
  const auto forcing =
      Evolution::Forcing{.sources = sourcesAt, .stressGluts = glutsAt};

  ViscoelasticOperator op(setup.rheology, kLmax);
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = "scheme " + std::to_string(static_cast<int>(scheme));
    Evolution general(op, solver, solver.NoOceanState(),
                      {.scheme = scheme,
                       .rotation = RotationalFeedback::None,
                       .rtol = 1e-12});
    const auto reference = general.Run(times, forcing);
    Evolution factored(op, solver, solver.NoOceanState(),
                       {.scheme = scheme,
                        .rotation = RotationalFeedback::None,
                        .factor_sources = true});
    const auto quick = factored.Run(times, forcing);
    ASSERT_EQ(quick.size(), reference.size());
    for (auto node = std::size_t{0}; node < quick.size(); ++node) {
      const auto where = "node " + std::to_string(node) + ", " + label;
      for (auto field : {&Evolution::Snapshot::relativeSeaLevelChange,
                         &Evolution::Snapshot::displacement,
                         &Evolution::Snapshot::potential,
                         &Evolution::Snapshot::tangentialPotential,
                         &Evolution::Snapshot::toroidalPotential}) {
        ASSERT_FALSE((quick[node].*field).empty()) << where;
        EXPECT_LT(WorstDeparture(quick[node].*field, reference[node].*field),
                  1e-9)
            << where;
      }
    }
  }
  // what the route is not for
  Evolution wet(op, solver, setup.state, {.factor_sources = true});
  EXPECT_THROW((void)wet.Run(times, forcing), std::invalid_argument);
}

TEST(PointSourceEvolution, UnderLateralStructureTheStepSolversAgree) {
  // Point sources and a stress glut under lateral viscosity and lateral
  // moduli, with an ocean and the rotational feedback: the steps closed
  // by the fixed point and by conjugate gradients, two different
  // closures of the same lateral system, give the same fields at every
  // node, the toroidal potential among them, and the lateral structure
  // moves them from the radial model's.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto& sle = solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto times = std::vector<double>{0.0, 0.4 * kyr, 1.0 * kyr};
  const auto thrust = SourceAt(
      setup, 400.0, 1.1, 4.0,
      {.rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6},
      1e22);
  const auto second = SourceAt(
      setup, 150.0, 2.2, 1.0,
      {.rr = -0.4, .tt = 0.8, .pp = -0.4, .rt = 0.5, .rp = 0.1, .tp = -0.3},
      5e21);
  const auto forcing = Evolution3D::Forcing{.sources = [&](double t) {
    auto out = std::vector<PointSource>{thrust};
    if (t >= times[1]) out.push_back(second);
    return out;
  }};
  auto delta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto moduli = Viscoelastic3D::Options::LateralModuli{
      .shear =
          [](double r, double theta, double phi) {
            return 0.1 * (0.5 + r * r) *
                   (std::sin(theta) * std::cos(theta) * std::cos(phi) +
                    0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
          },
      .bulk = [](double r, double theta, double) {
        return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
      }};
  auto run = [&](Viscoelastic3D::Options::StepSolver step, bool lateral) {
    auto options = Viscoelastic3D::Options{};
    options.solver = step;
    options.solve_rtol = 1e-12;
    if (lateral) options.moduli = moduli;
    Viscoelastic3D op(
        setup.rheology, kLmax,
        lateral ? std::function<double(double, double, double)>(delta)
                : [](double, double, double) { return 0.0; },
        options);
    Evolution3D evolution(op, solver, setup.state,
                          {.rtol = 1e-12, .coupled_rtol = 1e-11});
    return evolution.Run(times, forcing);
  };
  using Solver = Viscoelastic3D::Options::StepSolver;
  const auto fixed = run(Solver::FixedPoint, true);
  const auto conjugate = run(Solver::ConjugateGradients, true);
  const auto radial = run(Solver::FixedPoint, false);
  for (auto node = std::size_t{0}; node < times.size(); ++node) {
    EXPECT_TRUE(fixed[node].converged);
    EXPECT_TRUE(conjugate[node].converged);
    for (auto field : {&Evolution3D::Snapshot::relativeSeaLevelChange,
                       &Evolution3D::Snapshot::displacement,
                       &Evolution3D::Snapshot::potential,
                       &Evolution3D::Snapshot::tangentialPotential,
                       &Evolution3D::Snapshot::toroidalPotential}) {
      EXPECT_LT(WorstDeparture(conjugate[node].*field, fixed[node].*field), 1e-7)
          << "node " << node;
    }
    EXPECT_GT(WorstDeparture(fixed[node].displacement, radial[node].displacement),
              1e-3)
        << "node " << node;
  }
}
