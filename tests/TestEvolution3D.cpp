// The coupled 3D evolution: the combined (load + rotation + lateral)
// within-step fixed point. The zero-lateral limit: with delta = 0, the
// coupled history must equal the 1D coupled evolution's.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/TangentLinear3D.hpp"
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

  static Setup Make(int lmax = kLmax) {
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
    auto viscosity = std::map<std::size_t, RadialValue>{};
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    ViscoelasticOperator reference(rheology, lmax);
    GRDSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
                     SeaLevelParameters{});
    Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
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

TEST(LateralGate, LoadsOutsideTheWaterBudgetAndLayersMatchThe1DLoop) {
  // A direct load outside the water budget and a layer on the solid
  // surface, beside one inside the budget: the 3D loop with no lateral
  // perturbation is the 1D loop, with an ocean and without; and under
  // lateral viscosity the tangent-linear run along the three, the run
  // being linear in them, is the run under them.
  auto setup = Setup::Make();
  const auto kyr = 1000.0 * 3.15576e7 / setup.solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, kyr, 2 * kyr, 4 * kyr};
  const auto held = setup.MeltLoad(-0.1);
  const auto& grid = setup.solver.Grid();
  const auto length = setup.solver.SolverScales().Length();
  const auto pattern = grid.FromFunction([&](double theta, double phi) {
    return 3.0 / length * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi - 1.0);
  });
  auto load = [&](double t) {
    auto out = held;
    for (auto& v : out) v *= 1.0 + 0.2 * t / kyr;
    return out;
  };
  // the same pattern as the ice's, turned about the axis by its sign
  auto outside = [&](double t) {
    auto out = held;
    for (auto i = std::size_t{0}; i < out.size(); ++i) {
      out[i] = -0.6 * held[out.size() - 1 - i] * (0.5 + 0.3 * t / kyr);
    }
    return out;
  };
  auto layer = [&](double t) {
    auto out = pattern;
    for (auto& v : out) v *= 0.3 + t / (4.0 * kyr);
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
    const auto reference = evolution1.Run(
        times, Evolution::Forcing{.directLoad = load,
                                  .directLoadOutsideWaterBudget = outside,
                                  .solidSurfaceChange = layer});
    Evolution3D evolution3(op3, setup.solver, *background,
                           {.rtol = 1e-11, .coupled_rtol = 1e-10});
    const auto coupled = evolution3.Run(
        times, Evolution3D::Forcing{.directLoad = load,
                                    .directLoadOutsideWaterBudget = outside,
                                    .solidSurfaceChange = layer});
    ASSERT_EQ(coupled.size(), reference.size());
    for (auto node = std::size_t{0}; node < coupled.size(); ++node) {
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
    }
    // and the two new forcings matter: without them, another run
    const auto bare =
        evolution3.Run(times, Evolution3D::Forcing{.directLoad = load});
    EXPECT_GT(WorstDeparture(bare.back().relativeSeaLevelChange,
                             coupled.back().relativeSeaLevelChange),
              1e-3)
        << label;
  }

  Viscoelastic3D lateral(
      setup.rheology, kLmax, [](double, double theta, double phi) {
        return 0.8 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  Evolution3D evolution(lateral, setup.solver, setup.state,
                        {.rtol = 1e-12, .coupled_rtol = 1e-12});
  const auto run = evolution.Run(
      times, Evolution3D::Forcing{.directLoad = load,
                                  .directLoadOutsideWaterBudget = outside,
                                  .solidSurfaceChange = layer});
  TangentPerturbation3D direction;
  for (auto t : times) {
    direction.directLoad.push_back(load(t));
    direction.directLoadOutsideWaterBudget.push_back(outside(t));
    direction.solidSurfaceChange.push_back(layer(t));
  }
  const auto along =
      TangentLinearSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12})
          .Run(times, direction);
  for (auto node = std::size_t{0}; node < run.size(); ++node) {
    EXPECT_LT(WorstDeparture(along[node].relativeSeaLevelChange,
                             run[node].relativeSeaLevelChange),
              1e-8)
        << "node " << node;
    EXPECT_LT(WorstDeparture(along[node].displacement, run[node].displacement),
              1e-8)
        << "node " << node;
  }
}

TEST(LateralGate, ForcedMigratingRunsMatchThe1DLoopAndTheFrozenRun) {
  // Migrating shorelines under the forcings beside the ice, present from
  // the first node so that the start is solved: a load inside the water
  // budget, one outside it, a layer on the solid surface and a point
  // source. With no lateral perturbation the 3D loop is the 1D loop,
  // through a melting that moves the shorelines. Under lateral viscosity,
  // where the forcing is too small to move a shoreline, the run is the
  // frozen run under the same forcing, a prescribed tide beside it.
  auto setup = Setup::Make();
  const auto& sle = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto times = std::vector<double>{0.0, kyr, 2 * kyr, 4 * kyr};
  const auto held = setup.MeltLoad(-0.1);
  const auto& grid = setup.solver.Grid();
  const auto length = sle.Length();
  const auto pattern = grid.FromFunction([&](double theta, double phi) {
    return std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi - 1.0);
  });
  const auto unit =
      1e21 / sle.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto source = PointSource{
      .at = {.radius = setup.solver.Radius() - 400e3 / length,
             .colatitude = 1.1,
             .longitude = 4.0},
      .moment = {.rr = 0.9 * unit, .tt = -0.2 * unit, .pp = -0.7 * unit,
                 .rt = 0.3 * unit, .rp = -0.7 * unit, .tp = 0.6 * unit}};
  // the forcing, scaled: the loads, the layer's height and the source
  auto forcingOf = [&](double loads, double height, double moment) {
    return Evolution3D::Forcing{
        .directLoad =
            [&, loads](double t) {
              auto out = held;
              for (auto& v : out) v *= loads * (1.0 + 0.2 * t / kyr);
              return out;
            },
        .directLoadOutsideWaterBudget =
            [&, loads](double t) {
              auto out = held;
              for (auto i = std::size_t{0}; i < out.size(); ++i) {
                out[i] = -0.6 * loads * held[out.size() - 1 - i] *
                         (0.5 + 0.3 * t / kyr);
              }
              return out;
            },
        .solidSurfaceChange =
            [&, height](double t) {
              auto out = pattern;
              for (auto& v : out) v *= height / length * (0.3 + t / (4.0 * kyr));
              return out;
            },
        .sources = [&, moment](double t) {
          auto at = source;
          for (auto* c : {&at.moment.rr, &at.moment.tt, &at.moment.pp,
                          &at.moment.rt, &at.moment.rp, &at.moment.tp}) {
            *c *= moment * (0.3 + t / (4.0 * kyr));
          }
          return std::vector<PointSource>{at};
        }};
  };
  auto as1D = [](const Evolution3D::Forcing& f) {
    return Evolution::Forcing{
        .directLoad = f.directLoad,
        .directLoadOutsideWaterBudget = f.directLoadOutsideWaterBudget,
        .solidSurfaceChange = f.solidSurfaceChange,
        .sources = f.sources};
  };
  const auto state =
      setup.solver.MakeState(setup.state.iceThickness, setup.state.seaLevel);
  auto compare = [&](const auto& got, const auto& want, double tolerance,
                     const char* label) {
    ASSERT_EQ(got.size(), want.size());
    for (auto node = std::size_t{0}; node < got.size(); ++node) {
      ASSERT_TRUE(got[node].converged) << label << ", node " << node;
      EXPECT_LT(WorstDeparture(got[node].relativeSeaLevelChange,
                               want[node].relativeSeaLevelChange),
                tolerance)
          << label << ", node " << node;
      EXPECT_LT(WorstDeparture(got[node].displacement, want[node].displacement),
                tolerance)
          << label << ", node " << node;
      EXPECT_LT(WorstDeparture(got[node].potential, want[node].potential),
                tolerance)
          << label << ", node " << node;
      EXPECT_LT(WorstDeparture(got[node].tangentialPotential,
                               want[node].tangentialPotential),
                tolerance)
          << label << ", node " << node;
      ASSERT_FALSE(want[node].toroidalPotential.empty());
      EXPECT_LT(WorstDeparture(got[node].toroidalPotential,
                               want[node].toroidalPotential),
                tolerance)
          << label << ", node " << node;
    }
  };

  // -- no lateral perturbation: the 1D loop, the shorelines moving
  {
    const auto melting = [&](double t) {
      auto out = state.iceThickness;
      for (auto& v : out) v *= 1.0 - 0.1 * t / kyr;
      return out;
    };
    const auto forcing = forcingOf(1.0, 3.0, 1.0);
    ViscoelasticOperator op1(setup.rheology, kLmax);
    Viscoelastic3D op3(setup.rheology, kLmax,
                       [](double, double, double) { return 0.0; });
    Evolution evolution1(op1, setup.solver, state,
                         {.rtol = 1e-11,
                          .max_nonlinear_iterations = 200,
                          .record_stride = 1});
    const auto reference =
        evolution1.RunMigrating(times, melting, as1D(forcing));
    Evolution3D evolution3(op3, setup.solver, state,
                           {.rtol = 1e-11,
                            .coupled_rtol = 1e-10,
                            .max_nonlinear_iterations = 200});
    const auto coupled = evolution3.RunMigrating(times, melting, forcing);
    compare(coupled, reference, 1e-7, "no lateral perturbation");
    auto moved = std::size_t{0};
    for (auto node = std::size_t{0}; node < coupled.size(); ++node) {
      // flotation's 0/1: rounding changes it only at a cell on the threshold
      EXPECT_EQ(coupled[node].oceanFunction, reference[node].oceanFunction)
          << "node " << node;
      for (auto i = std::size_t{0}; i < state.oceanFunction.size(); ++i) {
        if (coupled[node].oceanFunction[i] != state.oceanFunction[i]) ++moved;
      }
    }
    EXPECT_GT(moved, std::size_t{0});
    // the start is solved: the first node is not the undisturbed one
    auto top = 0.0;
    for (auto v : coupled.front().relativeSeaLevelChange) {
      top = std::max(top, std::abs(v));
    }
    EXPECT_GT(top, 0.0);
    // The run of one node is the start's solve alone, the static
    // problem, and the ice there may differ from the background's: the
    // 1D run's, shorelines included.
    const auto one = std::vector<double>{0.0};
    const auto thinned = [&](double) {
      auto out = state.iceThickness;
      for (auto& v : out) v *= 0.8;
      return out;
    };
    const auto alone1 = evolution1.RunMigrating(one, thinned, as1D(forcing));
    const auto alone3 = evolution3.RunMigrating(one, thinned, forcing);
    ASSERT_EQ(alone3.size(), 1u);
    compare(alone3, alone1, 1e-7, "one node");
    // flotation's 0/1: rounding changes it only at a cell on the threshold
    EXPECT_EQ(alone3[0].oceanFunction, alone1[0].oceanFunction);
    EXPECT_GT(WorstDeparture(alone3[0].relativeSeaLevelChange,
                             coupled[0].relativeSeaLevelChange),
              1e-3);
  }

  // -- lateral viscosity, no shoreline moving: the frozen run
  {
    const auto still = [&](double) { return state.iceThickness; };
    auto forcing = forcingOf(1e-3, 1e-3, 1e-2);
    forcing.tide = [&](double t) {
      auto psi = grid.ZeroCoefficients();
      const auto a = 1e-2 / sle.Factor(PotentialDimensions);
      psi[SurfaceGrid::Index(2, 1)] = a * std::complex<double>(0.4, -0.3 * t / kyr);
      psi[SurfaceGrid::Index(3, 2)] = a * std::complex<double>(0.2, 0.5);
      return psi;
    };
    Viscoelastic3D lateral(
        setup.rheology, kLmax, [](double, double theta, double phi) {
          return 0.8 * std::sin(theta) * std::cos(theta) * std::cos(phi);
        });
    Evolution3D evolution(lateral, setup.solver, state,
                          {.rtol = 1e-12,
                           .coupled_rtol = 1e-11,
                           .max_nonlinear_iterations = 400});
    const auto frozen = evolution.Run(times, forcing);
    const auto moving = evolution.RunMigrating(times, still, forcing);
    for (const auto& snapshot : moving) {
      // flotation's 0/1: rounding changes it only at a cell on the threshold
      ASSERT_EQ(snapshot.oceanFunction, state.oceanFunction);
    }
    compare(moving, frozen, 1e-7, "lateral viscosity");
  }
}

TEST(Lateral, ARunOfOneNodeIsHeldToNoFluidLimit) {
  // As Evolution's: a run of one node takes no step, so moments of
  // inertia that sustain less oblateness than the model's hydrostatic
  // fluid limit, but more than the elastic response, are taken there.
  // Under the full feedback the run is Instantaneous under the same
  // forcing, by the same code, and the elastic sea level solve; a run
  // that steps is refused.
  auto setup = Setup::Make();
  const Viscoelastic3D op(setup.rheology, kLmax,
                          [](double, double, double) { return 0.0; });
  const auto kHyd = FluidTidalNumber(setup.material, 2);
  const GRDSolver solver(
      SurfaceGrid(kLmax), op.ElasticLove(),
      WithHydrostaticMoments(SeaLevelParameters{}, kHyd, -0.05));
  ASSERT_LE(solver.TransverseMarginFor(kHyd), 0.0);
  ASSERT_GT(solver.TransverseRotationDenominator(), 0.0);
  const auto state =
      solver.MakeState(setup.state.iceThickness, setup.state.seaLevel);
  const auto load = setup.MeltLoad(-0.1);
  Evolution3D evolution(op, solver, state,
                        {.rtol = 1e-12, .coupled_rtol = 1e-11});
  const auto forcing =
      Evolution3D::Forcing{.directLoad = [&](double) { return load; }};
  const auto one = std::vector<double>{0.0};
  const auto alone = evolution.Run(one, forcing);
  ASSERT_EQ(alone.size(), 1u);
  ASSERT_TRUE(alone[0].converged);
  const auto instant = evolution.Instantaneous(forcing);
  // the same code in one process: bitwise
  EXPECT_EQ(alone[0].relativeSeaLevelChange, instant.relativeSeaLevelChange);
  EXPECT_EQ(alone[0].displacement, instant.displacement);
  EXPECT_EQ(alone[0].angularVelocity, instant.angularVelocity);
  const auto reference =
      solver.Solve(state, load, RotationalFeedback::Full, {.rtol = 1e-12});
  ASSERT_NE(reference.angularVelocity[0], 0.0);
  EXPECT_LT(WorstDeparture(alone[0].relativeSeaLevelChange,
                           reference.relativeSeaLevelChange),
            1e-10);
  EXPECT_LT(WorstDeparture(alone[0].angularVelocity, reference.angularVelocity),
            1e-10);
  const auto two =
      std::vector<double>{0.0, 1e10 / solver.SolverScales().Time()};
  EXPECT_THROW((void)evolution.Run(two, forcing), std::invalid_argument);
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
  // one process, the same transforms: bitwise, here and below
  for (auto k = std::size_t{0}; k < tail.size(); ++k) {
    const auto& a = full[2 + k];
    const auto& b = tail[k];
    EXPECT_EQ(a.relativeSeaLevelChange, b.relativeSeaLevelChange) << "node " << 2 + k;
    EXPECT_EQ(a.displacement, b.displacement) << "node " << 2 + k;
    EXPECT_EQ(a.potential, b.potential) << "node " << 2 + k;
    EXPECT_EQ(a.angularVelocity, b.angularVelocity) << "node " << 2 + k;
    EXPECT_EQ(a.coupledIterations, b.coupledIterations) << "node " << 2 + k;
  }

  // keeping no snapshots, the resumed run hands the same ones out, counted
  // from the checkpoint's node, and returns none
  auto handed = std::map<std::size_t, Evolution3D::Snapshot>{};
  auto streaming = Evolution3D::Options{};
  streaming.keep_snapshots = false;
  streaming.on_snapshot = [&](std::size_t node,
                              const Evolution3D::Snapshot& snapshot) {
    handed.emplace(node, snapshot);
  };
  Evolution3D streamed(op, setup.solver, setup.state, streaming);
  EXPECT_TRUE(streamed
                  .Resume(checkpoints.at(1),
                          std::vector<double>(times.begin() + 1, times.end()),
                          directLoad)
                  .empty());
  ASSERT_EQ(handed.size(), tail.size());
  for (auto k = std::size_t{0}; k < tail.size(); ++k) {
    ASSERT_TRUE(handed.contains(k + 1)) << "node " << k + 1;
    EXPECT_EQ(handed.at(k + 1).displacement, tail[k].displacement)
        << "node " << 2 + k;
  }
}

TEST(Lateral, AResumedRunUnderInteriorSourcesIsTheUninterruptedOne) {
  // Under a point source with lateral viscosity the checkpoint's
  // recurrence, with its strain coefficients, is all a resumed run
  // needs: the toroidal systems are part of the state.
  auto setup = Setup::Make();
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{0.0, kyr, 2 * kyr, 3 * kyr};
  const auto surface = setup.material.Radius() *
                       setup.model.ModelScales().Length() / scales.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / scales.Factor(energy);
  Evolution3D::Forcing forcing;
  forcing.sources = [&](double t) {
    const auto s = unit * (0.3 + t / times.back());
    return std::vector<PointSource>{
        {.at = {.radius = surface - 400e3 / scales.Length(),
                .colatitude = 1.1,
                .longitude = 4.0},
         .moment = {.rr = 0.9 * s, .tt = -0.2 * s, .pp = -0.7 * s,
                    .rt = 0.3 * s, .rp = -0.7 * s, .tp = 0.6 * s}}};
  };
  Viscoelastic3D op(
      setup.rheology, kLmax, [](double, double theta, double phi) {
        return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  auto checkpoints = std::map<std::size_t, Evolution3D::Checkpoint>{};
  auto options = Evolution3D::Options{};
  options.checkpoint_stride = 1;
  options.on_checkpoint = [&](std::size_t node, const Evolution3D::Checkpoint& cp) {
    checkpoints.emplace(node, cp);
  };
  Evolution3D whole(op, setup.solver, setup.state, options);
  const auto full = whole.Run(times, forcing);
  Evolution3D resumed(op, setup.solver, setup.state, {});
  const auto tail = resumed.Resume(
      checkpoints.at(1), std::vector<double>(times.begin() + 1, times.end()),
      forcing);
  ASSERT_EQ(tail.size(), times.size() - 2);
  // one process, the same transforms: bitwise
  for (auto k = std::size_t{0}; k < tail.size(); ++k) {
    const auto& a = full[2 + k];
    const auto& b = tail[k];
    EXPECT_EQ(a.relativeSeaLevelChange, b.relativeSeaLevelChange) << "node " << 2 + k;
    EXPECT_EQ(a.displacement, b.displacement) << "node " << 2 + k;
    EXPECT_EQ(a.toroidalPotential, b.toroidalPotential) << "node " << 2 + k;
    EXPECT_FALSE(a.toroidalPotential.empty());
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

namespace {

/// The fields of two snapshots' worst relative departure, the toroidal
/// potential among them, and their angular velocities' against the
/// later's size.
template <class SnapshotT>
double SnapshotDeparture(const SnapshotT& a, const SnapshotT& b) {
  auto worst = 0.0;
  for (auto field : {&SnapshotT::relativeSeaLevelChange, &SnapshotT::displacement,
                     &SnapshotT::potential, &SnapshotT::tangentialPotential,
                     &SnapshotT::toroidalPotential}) {
    if ((b.*field).empty() || MaxAbsOf(b.*field) == 0.0) continue;
    worst = std::max(worst, WorstDeparture(a.*field, b.*field));
  }
  auto omega = 0.0, size = 0.0;
  for (auto i = 0; i < 3; ++i) {
    omega = std::max(omega, std::abs(a.angularVelocity[i] - b.angularVelocity[i]));
    size = std::max(size, std::abs(b.angularVelocity[i]));
  }
  if (size > 0.0) worst = std::max(worst, omega / size);
  return worst;
}

}  // namespace

TEST(PointSourceEvolution, AnOnsetAtARepeatedNodeIsTheRunStartedThere) {
  // An earthquake at an interior node: the node repeated, its left limit
  // without the sources (Forcing::sourcesBefore) and its right with them,
  // the zero-length step between the elastic solve with the memory held.
  // From the onset on, every field at every node is the run started at
  // the onset, whose first node is the instantaneous elastic response:
  // the radial run without an ocean and with one and the rotational
  // feedback, both schemes; the 3D run under lateral viscosity, and under
  // lateral moduli beside it. Before the onset nothing moves.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto& sle = solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto onset = 0.3 * kyr;
  const auto jump =
      std::vector<double>{0.0, onset, onset, 0.8 * kyr, 2.0 * kyr};
  const auto started = std::vector<double>{onset, 0.8 * kyr, 2.0 * kyr};
  const auto thrust = SourceAt(
      setup, 400.0, 1.1, 4.0,
      {.rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6},
      1e22);
  const auto second = SourceAt(
      setup, 150.0, 2.2, 1.0,
      {.rr = -0.4, .tt = 0.8, .pp = -0.4, .rt = 0.5, .rp = 0.1, .tp = -0.3},
      5e21);
  auto sourcesAt = [&](double t) {
    return t >= onset ? std::vector<PointSource>{thrust, second}
                      : std::vector<PointSource>{};
  };
  auto before = [&](double t) {
    return t > onset ? std::vector<PointSource>{thrust, second}
                     : std::vector<PointSource>{};
  };
  const auto forcing =
      Evolution::Forcing{.sources = sourcesAt, .sourcesBefore = before};
  const auto forcing3 =
      Evolution3D::Forcing{.sources = sourcesAt, .sourcesBefore = before};

  auto compare = [&](const auto& got, const auto& want, double tolerance,
                     const std::string& label) {
    ASSERT_EQ(got.size(), jump.size()) << label;
    ASSERT_EQ(want.size(), started.size()) << label;
    EXPECT_EQ(got[1].t, got[2].t) << label;
    const auto size = MaxAbsOf(got[2].displacement);
    EXPECT_GT(size, 0.0) << label;
    for (auto node : {std::size_t{0}, std::size_t{1}}) {
      EXPECT_LE(MaxAbsOf(got[node].displacement), 1e-14 * size)
          << label << ", node " << node;
    }
    auto worst = 0.0;
    for (auto k = std::size_t{0}; k < started.size(); ++k) {
      EXPECT_TRUE(got[2 + k].converged) << label;
      worst = std::max(worst, SnapshotDeparture(got[2 + k], want[k]));
    }
    // measured: 1.4e-13 at worst, most cases bitwise
    EXPECT_LT(worst, tolerance) << label;
  };

  ViscoelasticOperator op1(setup.rheology, kLmax);
  const auto dry = solver.NoOceanState();
  const auto& wet = setup.state;
  for (const auto* background : {&wet, &dry})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto rotation = background == &wet ? RotationalFeedback::Full
                                             : RotationalFeedback::None;
    const auto label = std::string(background == &dry ? "no ocean" : "ocean") +
                       ", scheme " + std::to_string(static_cast<int>(scheme));
    const auto options = Evolution::Options{
        .scheme = scheme, .rotation = rotation, .rtol = 1e-12};
    Evolution evolution(op1, solver, *background, options);
    const auto reference = evolution.Run(started, forcing);
    compare(evolution.Run(jump, forcing), reference, 1e-11, label);
  }

  // the 3D run under lateral viscosity, with an ocean and the feedback
  Viscoelastic3D op3(setup.rheology, kLmax, [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  Evolution3D evolution3(op3, solver, wet,
                         {.rtol = 1e-12, .coupled_rtol = 1e-12});
  const auto reference3 = evolution3.Run(started, forcing3);
  compare(evolution3.Run(jump, forcing3), reference3, 1e-11,
          "3D, lateral viscosity");

  // and lateral moduli beside it, whose zero-length step closes the
  // instantaneous fixed point of the lateral operator
  auto options = Viscoelastic3D::Options{};
  options.solve_rtol = 1e-12;
  options.moduli = Viscoelastic3D::Options::LateralModuli{
      .shear =
          [](double r, double theta, double phi) {
            return 0.1 * (0.5 + r * r) *
                   (std::sin(theta) * std::cos(theta) * std::cos(phi) +
                    0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
          },
      .bulk = [](double r, double theta, double) {
        return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
      }};
  Viscoelastic3D moduli(
      setup.rheology, kLmax,
      [](double, double theta, double phi) {
        return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      },
      options);
  Evolution3D lateral(moduli, solver, wet,
                      {.rtol = 1e-12, .coupled_rtol = 1e-12});
  compare(lateral.Run(jump, forcing3), lateral.Run(started, forcing3), 1e-11,
          "3D, lateral viscosity and moduli");
}

namespace {

/// The forcings beside the sources that act through a jump: a load inside
/// the water budget and one outside it, a layer on the solid surface and
/// a prescribed tide, each growing in time, scaled by `scale`.
template <class ForcingT>
ForcingT LoadsThrough(const Setup& setup, double kyr, double scale) {
  const auto& grid = setup.solver.Grid();
  const auto length = setup.solver.SolverScales().Length();
  const auto held = setup.MeltLoad(-0.1);
  const auto pattern = grid.FromFunction([](double theta, double phi) {
    return std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi - 1.0);
  });
  const auto tide = TestTide(setup.solver, kyr);
  auto out = ForcingT{};
  out.directLoad = [held, scale, kyr](double t) {
    auto f = held;
    for (auto& v : f) v *= scale * (1.0 + 0.2 * t / kyr);
    return f;
  };
  out.directLoadOutsideWaterBudget = [held, scale, kyr](double t) {
    auto f = held;
    for (auto i = std::size_t{0}; i < f.size(); ++i) {
      f[i] = -0.6 * scale * held[f.size() - 1 - i] * (0.5 + 0.3 * t / kyr);
    }
    return f;
  };
  out.solidSurfaceChange = [pattern, scale, length, kyr](double t) {
    auto f = pattern;
    for (auto& v : f) v *= 3.0 * scale / length * (0.3 + t / (4.0 * kyr));
    return f;
  };
  out.tide = [tide, scale](double t) {
    auto psi = tide(t);
    for (auto& c : psi) c *= scale;
    return psi;
  };
  return out;
}

}  // namespace

TEST(PointSourceEvolution, ARepeatedNodeWithoutAJumpChangesNothing) {
  // A node repeated where nothing jumps, the loads, the layer, the tide,
  // the ice and a growing source all continuous there, each surface term
  // given a left limit equal to it (Forcing::directLoadBefore and its kin)
  // or none: the zero-length step holds the memory and solves under the
  // same forcing, so both nodes of the pair are the node of the grid
  // without the repeat, and so is every node after, with an ocean and the
  // rotational feedback: the radial run (both schemes, the shorelines
  // frozen and migrating) and the 3D run under lateral viscosity.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto& sle = solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto repeated =
      std::vector<double>{0.0, 0.3 * kyr, 0.3 * kyr, 0.8 * kyr, 2.0 * kyr};
  const auto plain = std::vector<double>{0.0, 0.3 * kyr, 0.8 * kyr, 2.0 * kyr};
  const auto into = std::vector<std::size_t>{0, 1, 1, 2, 3};
  const auto thrust = SourceAt(
      setup, 400.0, 1.1, 4.0,
      {.rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6},
      1e22);
  auto growing = [&](double t) {
    auto at = thrust;
    for (auto* c : {&at.moment.rr, &at.moment.tt, &at.moment.pp, &at.moment.rt,
                    &at.moment.rp, &at.moment.tp}) {
      *c *= 0.3 + t / kyr;
    }
    return std::vector<PointSource>{at};
  };
  const auto melting = [&](double t) {
    auto out = setup.state.iceThickness;
    for (auto& v : out) v *= 1.0 - 0.1 * t / kyr;
    return out;
  };
  // the left limits, the plain callbacks themselves
  const auto withLimits = [&](auto f) {
    f.directLoadBefore = f.directLoad;
    f.directLoadOutsideWaterBudgetBefore = f.directLoadOutsideWaterBudget;
    f.solidSurfaceChangeBefore = f.solidSurfaceChange;
    f.tideBefore = f.tide;
    return f;
  };
  auto forcing = LoadsThrough<Evolution::Forcing>(setup, kyr, 1.0);
  forcing.sources = growing;
  const auto limited = withLimits(forcing);
  auto migratingLimited = limited;
  migratingLimited.iceThicknessBefore = melting;
  auto forcing3 = LoadsThrough<Evolution3D::Forcing>(setup, kyr, 1.0);
  forcing3.sources = growing;
  const auto limited3 = withLimits(forcing3);
  auto compare = [&](const auto& got, const auto& want, const std::string& label) {
    ASSERT_EQ(got.size(), repeated.size()) << label;
    auto worst = 0.0;
    for (auto n = std::size_t{0}; n < repeated.size(); ++n) {
      EXPECT_TRUE(got[n].converged) << label;
      worst = std::max(worst, SnapshotDeparture(got[n], want[into[n]]));
      if (!want[into[n]].oceanFunction.empty()) {
        EXPECT_EQ(got[n].oceanFunction, want[into[n]].oceanFunction)
            << label << ", node " << n;
      }
    }
    // the sea level solves re-converged to their tolerance (rtol 1e-12):
    // measured 1.0e-11 at worst
    EXPECT_LT(worst, 1e-10) << label;
  };

  ViscoelasticOperator op1(setup.rheology, kLmax);
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = "scheme " + std::to_string(static_cast<int>(scheme));
    Evolution evolution(op1, solver, setup.state,
                        {.scheme = scheme,
                         .rtol = 1e-12,
                         .max_nonlinear_iterations = 200});
    compare(evolution.Run(repeated, forcing), evolution.Run(plain, forcing),
            "frozen, " + label);
    compare(evolution.Run(repeated, limited), evolution.Run(plain, forcing),
            "frozen, left limits, " + label);
    compare(evolution.RunMigrating(repeated, melting, forcing),
            evolution.RunMigrating(plain, melting, forcing), "migrating, " + label);
    compare(evolution.RunMigrating(repeated, melting, migratingLimited),
            evolution.RunMigrating(plain, melting, forcing),
            "migrating, left limits, " + label);
  }
  Viscoelastic3D op3(setup.rheology, kLmax, [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  Evolution3D evolution3(op3, solver, setup.state,
                         {.rtol = 1e-12, .coupled_rtol = 1e-12});
  compare(evolution3.Run(repeated, forcing3), evolution3.Run(plain, forcing3),
          "3D, lateral viscosity");
  compare(evolution3.Run(repeated, limited3), evolution3.Run(plain, forcing3),
          "3D, lateral viscosity, left limits");
}

TEST(PointSourceEvolution, AnOnsetUnderLoadsActingThroughIt) {
  // An earthquake at a repeated node while the loads, the layer and the
  // tide act through it, with an ocean and the rotational feedback. Under
  // frozen shorelines the problem is linear: the run under both is the
  // run under the loads (which the repeat leaves alone) plus the run under
  // the sources (the run started at the onset), the radial run for both
  // schemes and the 3D run under lateral viscosity. Under migrating
  // shorelines, through a melting, it is the limit of the runs with a node
  // eps before the onset, which the earthquake ramps over a step of eps:
  // their gap falls as eps.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto& sle = solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto onset = 0.3 * kyr;
  const auto jump =
      std::vector<double>{0.0, onset, onset, 0.8 * kyr, 2.0 * kyr};
  const auto thrust = SourceAt(
      setup, 400.0, 1.1, 4.0,
      {.rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6},
      1e24);
  auto sourcesAt = [&](double t) {
    return t >= onset ? std::vector<PointSource>{thrust}
                      : std::vector<PointSource>{};
  };
  auto before = [&](double t) {
    return t > onset ? std::vector<PointSource>{thrust}
                     : std::vector<PointSource>{};
  };
  auto sum = [](auto a, const auto& b) {
    for (auto field : {&std::decay_t<decltype(a)>::relativeSeaLevelChange,
                       &std::decay_t<decltype(a)>::displacement,
                       &std::decay_t<decltype(a)>::potential,
                       &std::decay_t<decltype(a)>::tangentialPotential}) {
      for (auto i = std::size_t{0}; i < (a.*field).size(); ++i) {
        (a.*field)[i] += (b.*field)[i];
      }
    }
    for (auto i = 0; i < 3; ++i) a.angularVelocity[i] += b.angularVelocity[i];
    // the loads on a radial model make no toroidal motion
    if (a.toroidalPotential.empty()) {
      a.toroidalPotential = b.toroidalPotential;
    } else {
      for (auto i = std::size_t{0}; i < a.toroidalPotential.size(); ++i) {
        a.toroidalPotential[i] += b.toroidalPotential[i];
      }
    }
    return a;
  };
  auto check = [&](const auto& both, const auto& loads, const auto& sources,
                   const std::string& label) {
    ASSERT_EQ(both.size(), jump.size()) << label;
    auto worst = 0.0;
    for (auto n = std::size_t{0}; n < jump.size(); ++n) {
      EXPECT_TRUE(both[n].converged) << label;
      worst = std::max(worst, SnapshotDeparture(both[n], sum(loads[n], sources[n])));
    }
    // measured: 1.9e-13 at worst
    EXPECT_LT(worst, 1e-11) << label;
    // the jump moves the field
    EXPECT_GT(WorstDeparture(both[2].displacement, both[1].displacement), 1e-2)
        << label;
  };

  ViscoelasticOperator op1(setup.rheology, kLmax);
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = "scheme " + std::to_string(static_cast<int>(scheme));
    Evolution evolution(op1, solver, setup.state,
                        {.scheme = scheme, .rtol = 1e-12});
    auto loads = LoadsThrough<Evolution::Forcing>(setup, kyr, 1.0);
    auto sources = Evolution::Forcing{.sources = sourcesAt, .sourcesBefore = before};
    auto both = loads;
    both.sources = sourcesAt;
    both.sourcesBefore = before;
    check(evolution.Run(jump, both), evolution.Run(jump, loads),
          evolution.Run(jump, sources), "frozen, " + label);
  }
  {
    Viscoelastic3D op3(setup.rheology, kLmax, [](double, double theta, double phi) {
      return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
    });
    Evolution3D evolution3(op3, solver, setup.state,
                           {.rtol = 1e-12, .coupled_rtol = 1e-12});
    auto loads = LoadsThrough<Evolution3D::Forcing>(setup, kyr, 1.0);
    auto sources =
        Evolution3D::Forcing{.sources = sourcesAt, .sourcesBefore = before};
    auto both = loads;
    both.sources = sourcesAt;
    both.sourcesBefore = before;
    check(evolution3.Run(jump, both), evolution3.Run(jump, loads),
          evolution3.Run(jump, sources), "3D, lateral viscosity");
  }

  // migrating shorelines: the limit of a node eps before the onset
  const auto melting = [&](double t) {
    auto out = setup.state.iceThickness;
    for (auto& v : out) v *= 1.0 - 0.1 * t / kyr;
    return out;
  };
  auto forcing = LoadsThrough<Evolution::Forcing>(setup, kyr, 1.0);
  forcing.sources = sourcesAt;
  forcing.sourcesBefore = before;
  Evolution evolution(op1, solver, setup.state,
                      {.rtol = 1e-12, .max_nonlinear_iterations = 200});
  const auto exact = evolution.RunMigrating(jump, melting, forcing);
  auto gaps = std::vector<double>{};
  for (const auto eps : {1e-2, 1e-3, 1e-4}) {
    auto near = jump;
    near[1] = onset - eps * kyr;
    const auto ramped = evolution.RunMigrating(near, melting, forcing);
    auto gap = 0.0;
    for (auto n = std::size_t{2}; n < jump.size(); ++n) {
      gap = std::max(gap, SnapshotDeparture(ramped[n], exact[n]));
    }
    gaps.push_back(gap);
  }
  // first order in eps: measured 1.69e-2, 1.68e-3, 1.68e-4
  for (auto k = std::size_t{0}; k + 1 < gaps.size(); ++k) {
    EXPECT_GT(gaps[k] / gaps[k + 1], 8.0);
    EXPECT_LT(gaps[k] / gaps[k + 1], 12.0);
  }
  EXPECT_LT(gaps.back(), 1e-3);
}

TEST(Lateral, AResumedRunAcrossAJumpIsTheUninterruptedOne) {
  // An earthquake and a step load at a repeated node under lateral
  // viscosity: resumed from the left node's checkpoint and from the right
  // node's, the run is the uninterrupted one.
  auto setup = Setup::Make();
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  const auto times = std::vector<double>{0.0, kyr, kyr, 2 * kyr, 3 * kyr};
  const auto source = SourceAt(
      setup, 400.0, 1.1, 4.0,
      {.rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6},
      1e24);
  Evolution3D::Forcing forcing;
  // a growing load with a step on it at the onset, its left limit without
  // the step
  const auto growing = [&](double t, bool stepped) {
    auto out = setup.MeltLoad(-0.1);
    for (auto& v : out) v *= t / times.back() + (stepped ? 0.5 : 0.0);
    return out;
  };
  forcing.directLoad = [&](double t) { return growing(t, t >= kyr); };
  forcing.directLoadBefore = [&](double t) { return growing(t, t > kyr); };
  forcing.sources = [&](double t) {
    return t >= kyr ? std::vector<PointSource>{source}
                    : std::vector<PointSource>{};
  };
  forcing.sourcesBefore = [&](double t) {
    return t > kyr ? std::vector<PointSource>{source}
                   : std::vector<PointSource>{};
  };
  Viscoelastic3D op(
      setup.rheology, kLmax, [](double, double theta, double phi) {
        return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  auto checkpoints = std::map<std::size_t, Evolution3D::Checkpoint>{};
  auto options = Evolution3D::Options{};
  options.checkpoint_stride = 1;
  options.on_checkpoint = [&](std::size_t node, const Evolution3D::Checkpoint& cp) {
    checkpoints.emplace(node, cp);
  };
  Evolution3D whole(op, setup.solver, setup.state, options);
  const auto full = whole.Run(times, forcing);
  for (const auto from : {std::size_t{1}, std::size_t{2}}) {
    Evolution3D resumed(op, setup.solver, setup.state, {});
    const auto tail = resumed.Resume(
        checkpoints.at(from),
        std::vector<double>(times.begin() + static_cast<long>(from), times.end()),
        forcing);
    ASSERT_EQ(tail.size(), times.size() - from - 1);
    // one process, the same transforms: bitwise
    for (auto k = std::size_t{0}; k < tail.size(); ++k) {
      const auto& a = full[from + 1 + k];
      const auto& b = tail[k];
      const auto where = "from node " + std::to_string(from) + ", node " +
                         std::to_string(from + 1 + k);
      EXPECT_EQ(a.relativeSeaLevelChange, b.relativeSeaLevelChange) << where;
      EXPECT_EQ(a.displacement, b.displacement) << where;
      EXPECT_EQ(a.toroidalPotential, b.toroidalPotential) << where;
    }
  }
  EXPECT_GT(WorstDeparture(full[2].displacement, full[1].displacement), 1e-2);
}

TEST(PointSourceEvolution,
     AnOnsetAtARepeatedNodeUnderSdirk2IsTheRunStartedThere) {
  // AnOnsetAtARepeatedNodeIsTheRunStartedThere under Sdirk2: the step to
  // the left limit takes its first stage under the sources there (none),
  // the zero-length step is backward Euler's elastic solve with the memory
  // held, and the stages after it the sources and the toroidal systems at
  // their times. From the onset on every field at every node is the run
  // started at the onset: radial without an ocean and with one and the
  // rotational feedback, and in 3D under lateral viscosity and under
  // lateral moduli beside it.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  const auto onset = 0.3 * kyr;
  const auto jump =
      std::vector<double>{0.0, onset, onset, 0.8 * kyr, 2.0 * kyr};
  const auto started = std::vector<double>{onset, 0.8 * kyr, 2.0 * kyr};
  const auto thrust = SourceAt(
      setup, 400.0, 1.1, 4.0,
      {.rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6},
      1e22);
  auto sourcesAt = [&](double t) {
    return t >= onset ? std::vector<PointSource>{thrust}
                      : std::vector<PointSource>{};
  };
  auto before = [&](double t) {
    return t > onset ? std::vector<PointSource>{thrust}
                     : std::vector<PointSource>{};
  };
  const auto forcing =
      Evolution::Forcing{.sources = sourcesAt, .sourcesBefore = before};
  const auto forcing3 =
      Evolution3D::Forcing{.sources = sourcesAt, .sourcesBefore = before};
  auto worst = 0.0;
  auto compare = [&](const auto& got, const auto& want,
                     const std::string& label) {
    ASSERT_EQ(got.size(), jump.size()) << label;
    ASSERT_EQ(want.size(), started.size()) << label;
    const auto size = MaxAbsOf(got[2].displacement);
    EXPECT_GT(size, 0.0) << label;
    for (auto node : {std::size_t{0}, std::size_t{1}}) {
      EXPECT_LE(MaxAbsOf(got[node].displacement), 1e-14 * size)
          << label << ", node " << node;
    }
    for (auto k = std::size_t{0}; k < started.size(); ++k) {
      EXPECT_TRUE(got[2 + k].converged) << label;
      const auto departure = SnapshotDeparture(got[2 + k], want[k]);
      worst = std::max(worst, departure);
      EXPECT_LT(departure, 1e-11) << label << ", node " << 2 + k;
    }
  };
  ViscoelasticOperator op1(setup.rheology, kLmax);
  const auto dry = solver.NoOceanState();
  const auto& wet = setup.state;
  for (const auto* background : {&wet, &dry}) {
    const auto label = std::string(background == &dry ? "no ocean" : "ocean");
    Evolution evolution(
        op1, solver, *background,
        {.scheme = Scheme::Sdirk2,
         .rotation = background == &wet ? RotationalFeedback::Full
                                        : RotationalFeedback::None,
         .rtol = 1e-12});
    compare(evolution.Run(jump, forcing), evolution.Run(started, forcing),
            label);
  }
  const auto delta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  Viscoelastic3D op3(setup.rheology, kLmax, delta);
  Evolution3D evolution3(
      op3, solver, wet,
      {.scheme = Scheme::Sdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12});
  compare(evolution3.Run(jump, forcing3), evolution3.Run(started, forcing3),
          "3D, lateral viscosity");
  auto options = Viscoelastic3D::Options{};
  options.solve_rtol = 1e-12;
  options.moduli = Viscoelastic3D::Options::LateralModuli{
      .shear =
          [](double r, double theta, double phi) {
            return 0.1 * (0.5 + r * r) * std::sin(theta) * std::cos(theta) *
                   std::cos(phi);
          },
      .bulk = [](double r, double theta,
                 double) { return 0.05 * (1.0 - 0.5 * r) * std::cos(theta); }};
  Viscoelastic3D moduli(setup.rheology, kLmax, delta, options);
  Evolution3D lateral(
      moduli, solver, wet,
      {.scheme = Scheme::Sdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12});
  compare(lateral.Run(jump, forcing3), lateral.Run(started, forcing3),
          "3D, lateral viscosity and moduli");
  // measured: bitwise in every case
  std::cout << std::format("[sdirk2 onset] worst departure {:.2e}\n", worst);
}

TEST(LateralGate, Sdirk2ZeroLateralEvolutionMatchesThe1DLoop) {
  // Under Sdirk2 the 3D loop with no lateral perturbation steps the same
  // stages as the 1D loop, each closed under the forcing at its time:
  // loads inside and outside the water budget, a layer and a tide growing
  // in time, with the full rotational feedback; every field to the
  // pipeline's round-off, as under the trapezoid
  // (ZeroLateralForcedEvolutionMatchesThe1DLoop).
  auto setup = Setup::Make();
  const auto kyr = 1000.0 * 3.15576e7 / setup.solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, 0.7 * kyr, 2.0 * kyr, 4.0 * kyr};
  const auto forcing = LoadsThrough<Evolution::Forcing>(setup, kyr, 1.0);
  const auto forcing3 = LoadsThrough<Evolution3D::Forcing>(setup, kyr, 1.0);
  ViscoelasticOperator op1(setup.rheology, kLmax);
  Evolution evolution1(op1, setup.solver, setup.state,
                       {.scheme = Scheme::Sdirk2, .rtol = 1e-12});
  const auto reference = evolution1.Run(times, forcing);
  Viscoelastic3D op3(setup.rheology, kLmax,
                     [](double, double, double) { return 0.0; });
  Evolution3D evolution3(
      op3, setup.solver, setup.state,
      {.scheme = Scheme::Sdirk2, .rtol = 1e-12, .coupled_rtol = 1e-11});
  const auto coupled = evolution3.Run(times, forcing3);
  ASSERT_EQ(coupled.size(), reference.size());
  auto worst = 0.0;
  for (auto node = std::size_t{0}; node < coupled.size(); ++node) {
    EXPECT_TRUE(coupled[node].converged) << node;
    const auto& a = coupled[node];
    const auto& b = reference[node];
    worst = std::max(
        {worst,
         WorstDeparture(a.relativeSeaLevelChange, b.relativeSeaLevelChange),
         WorstDeparture(a.displacement, b.displacement),
         WorstDeparture(a.potential, b.potential),
         WorstDeparture(a.tangentialPotential, b.tangentialPotential)});
    auto omega = 0.0, size = 0.0;
    for (auto i = 0; i < 3; ++i) {
      omega = std::max(omega,
                       std::abs(a.angularVelocity[i] - b.angularVelocity[i]));
      size = std::max(size, std::abs(b.angularVelocity[i]));
    }
    worst = std::max(worst, omega / size);
  }
  std::cout << std::format("[sdirk2 zero lateral] worst departure {:.2e}\n",
                           worst);
  // measured: 8e-15
  EXPECT_LT(worst, 1e-8);
}

TEST(Lateral, Sdirk2ConvergesAtSecondOrderUnderLateralViscosity) {
  // Under a lateral viscosity, at lmax 4, a cap's load ramped on over
  // half of 8 kyr and held: Sdirk2's sea level at the end converges at second
  // order, to the limit the trapezoid's runs extrapolate to, each stage's
  // combined fixed point converged; and a run resumed from a checkpoint (the
  // internal variables and their strain alone: a step of Sdirk2 starts from the
  // node's state) is the uninterrupted one, bitwise.
  constexpr auto lmax = 4;
  auto setup = Setup::Make(lmax);
  const auto kyr = 1000.0 * 3.15576e7 / setup.solver.SolverScales().Time();
  const auto span = 8.0 * kyr;
  // a cap of 1000 kg m^-2 at colatitude 1, longitude 2
  const auto sigma = 1e3 / setup.solver.SolverScales().Factor(
                               Dimensions{.mass = 1, .length = -2});
  const auto load =
      setup.solver.Grid().FromFunction([&](double theta, double phi) {
        const auto c = std::cos(theta) * std::cos(1.0) +
                       std::sin(theta) * std::sin(1.0) * std::cos(phi - 2.0);
        return sigma * std::exp(-6.0 * (1.0 - c));
      });
  const auto direct = [&](double t) {
    auto out = load;
    const auto f = std::min(t / (0.5 * span), 1.0);
    for (auto& v : out) v *= f;
    return out;
  };
  Viscoelastic3D op(setup.rheology, lmax, [](double, double theta, double phi) {
    return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  auto grid = [&](int n) {
    auto out = std::vector<double>{};
    for (auto k = 0; k <= n; ++k) out.push_back(span * k / n);
    return out;
  };
  auto passes = 0;
  auto run = [&](Scheme scheme, int n) {
    Evolution3D evolution(
        op, setup.solver, setup.state,
        {.scheme = scheme, .rtol = 1e-12, .coupled_rtol = 1e-12});
    const auto snapshots = evolution.Run(grid(n), direct);
    for (const auto& snapshot : snapshots) {
      EXPECT_TRUE(snapshot.converged) << static_cast<int>(scheme) << ", " << n;
      if (scheme == Scheme::Sdirk2 && n == 8)
        passes += snapshot.coupledIterations;
    }
    return snapshots.back().relativeSeaLevelChange;
  };
  auto extrapolate = [](std::vector<double> fine,
                        const std::vector<double>& coarse) {
    for (auto i = std::size_t{0}; i < fine.size(); ++i) {
      fine[i] += (fine[i] - coarse[i]) / 3.0;
    }
    return fine;
  };
  const auto limit =
      extrapolate(run(Scheme::Sdirk2, 64), run(Scheme::Sdirk2, 32));
  const auto trapezoid = extrapolate(run(Scheme::ExponentialTrapezoid, 128),
                                     run(Scheme::ExponentialTrapezoid, 64));
  auto errors = std::vector<double>{};
  for (const auto n : {4, 8, 16}) {
    errors.push_back(WorstDeparture(run(Scheme::Sdirk2, n), limit));
  }
  const auto apart = WorstDeparture(trapezoid, limit);
  std::cout << std::format(
      "[sdirk2 lateral] errors {:.3e} {:.3e} {:.3e}; the limits {:.2e} apart; "
      "{:.1f} passes a step at 8 steps\n",
      errors[0], errors[1], errors[2], apart, passes / 8.0);
  // measured: errors 1.24e-3, 3.40e-4, 8.24e-5 (orders 1.87, 2.04), the
  // limits 6.5e-7 apart, 43 passes a step (two stages) at 8 steps, the
  // trapezoid's 21
  EXPECT_GT(std::log2(errors[0] / errors[1]), 1.7);
  EXPECT_GT(std::log2(errors[1] / errors[2]), 1.9);
  EXPECT_LT(std::log2(errors[1] / errors[2]), 2.2);
  EXPECT_LT(apart, 1e-5);

  auto checkpoints = std::map<std::size_t, Evolution3D::Checkpoint>{};
  auto options = Evolution3D::Options{
      .scheme = Scheme::Sdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12};
  options.checkpoint_stride = 1;
  options.on_checkpoint = [&](std::size_t node,
                              const Evolution3D::Checkpoint& cp) {
    checkpoints.emplace(node, cp);
  };
  const auto times = grid(4);
  Evolution3D whole(op, setup.solver, setup.state, options);
  const auto full = whole.Run(times, direct);
  Evolution3D resumed(
      op, setup.solver, setup.state,
      {.scheme = Scheme::Sdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12});
  const auto tail = resumed.Resume(
      checkpoints.at(2), std::vector<double>(times.begin() + 2, times.end()),
      direct);
  ASSERT_EQ(tail.size(), times.size() - 3);
  for (auto k = std::size_t{0}; k < tail.size(); ++k) {
    EXPECT_EQ(full[3 + k].relativeSeaLevelChange,
              tail[k].relativeSeaLevelChange)
        << "node " << 3 + k;
    EXPECT_EQ(full[3 + k].coupledIterations, tail[k].coupledIterations)
        << "node " << 3 + k;
  }
}

namespace {

/// Step loads switched on at `onset`: a load inside the water budget, one
/// outside it, a layer and a tide, each a pattern held from the onset on,
/// added to `base`; their left limits (Forcing::directLoadBefore and its
/// kin) the base's alone at the onset. Without a base, the steps alone.
template <class ForcingT>
ForcingT StepsAt(const Setup& setup, double onset, double kyr,
                 const ForcingT* base) {
  const auto& grid = setup.solver.Grid();
  const auto length = setup.solver.SolverScales().Length();
  const auto pattern = grid.FromFunction([](double theta, double phi) {
    return std::cos(theta) * std::sin(theta) * std::sin(phi + 0.4);
  });
  const auto melt = setup.MeltLoad(-0.2);
  const auto tide = TestTide(setup.solver, kyr)(2.0 * kyr);
  const auto field = [base](auto member, double t) {
    return base != nullptr && (*base).*member ? ((*base).*member)(t)
                                              : decltype(((*base).*member)(t)){};
  };
  // a term's value with its step, on from `on(t)`
  const auto stepped = [](auto value, const auto& step, bool on) {
    if (!on) return value;
    if (value.empty()) value.assign(step.size(), {});
    for (auto i = std::size_t{0}; i < value.size(); ++i) value[i] += step[i];
    return value;
  };
  auto outside = melt;
  std::ranges::reverse(outside);
  for (auto& v : outside) v *= -0.7;
  auto layer = pattern;
  for (auto& v : layer) v *= 2.0 / length;
  auto out = ForcingT{};
  using F = ForcingT;
  out.directLoad = [=](double t) {
    return stepped(field(&F::directLoad, t), melt, t >= onset);
  };
  out.directLoadBefore = [=](double t) {
    return stepped(field(&F::directLoad, t), melt, t > onset);
  };
  out.directLoadOutsideWaterBudget = [=](double t) {
    return stepped(field(&F::directLoadOutsideWaterBudget, t), outside,
                   t >= onset);
  };
  out.directLoadOutsideWaterBudgetBefore = [=](double t) {
    return stepped(field(&F::directLoadOutsideWaterBudget, t), outside,
                   t > onset);
  };
  out.solidSurfaceChange = [=](double t) {
    return stepped(field(&F::solidSurfaceChange, t), layer, t >= onset);
  };
  out.solidSurfaceChangeBefore = [=](double t) {
    return stepped(field(&F::solidSurfaceChange, t), layer, t > onset);
  };
  out.tide = [=](double t) {
    return stepped(field(&F::tide, t), tide, t >= onset);
  };
  out.tideBefore = [=](double t) {
    return stepped(field(&F::tide, t), tide, t > onset);
  };
  return out;
}

/// A snapshot plus another, field by field.
template <class SnapshotT>
SnapshotT SumOf(SnapshotT a, const SnapshotT& b) {
  for (auto field : {&SnapshotT::relativeSeaLevelChange,
                     &SnapshotT::displacement, &SnapshotT::potential,
                     &SnapshotT::tangentialPotential}) {
    for (auto i = std::size_t{0}; i < (a.*field).size(); ++i) {
      (a.*field)[i] += (b.*field)[i];
    }
  }
  for (auto i = 0; i < 3; ++i) a.angularVelocity[i] += b.angularVelocity[i];
  // a radial run under loads makes no toroidal motion, a lateral one does
  if (a.toroidalPotential.empty()) {
    a.toroidalPotential = b.toroidalPotential;
  } else {
    for (auto i = std::size_t{0}; i < b.toroidalPotential.size(); ++i) {
      a.toroidalPotential[i] += b.toroidalPotential[i];
    }
  }
  return a;
}

}  // namespace

TEST(SurfaceJumps, AStepLoadAtARepeatedNodeIsTheRunStartedThere) {
  // Step loads at an interior node (a load in the water budget and one
  // outside it, a layer and a tide, switched on there), the node repeated,
  // its left limit without them (Forcing::directLoadBefore and its kin),
  // while growing loads act through it. Under frozen shorelines the run is
  // linear: from the right node on it is the run under the growing loads
  // alone plus the run started at the node under the steps alone (whose
  // first node is the instantaneous elastic response), and before it the
  // growing loads' alone: the radial run without an ocean and with one and
  // the rotational feedback, the trapezoid, backward Euler and Sdirk2
  // (whose stages inside the steps around the pair read the forcing's
  // plain callbacks, detail::AtStage); the 3D run under lateral viscosity
  // (the trapezoid and Sdirk2) and under lateral moduli beside it.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  const auto onset = 0.3 * kyr;
  const auto jump =
      std::vector<double>{0.0, onset, onset, 0.8 * kyr, 2.0 * kyr};
  const auto started = std::vector<double>{onset, 0.8 * kyr, 2.0 * kyr};
  auto worst = 0.0;
  auto check = [&](const auto& both, const auto& growing, const auto& steps,
                   const std::string& label) {
    ASSERT_EQ(both.size(), jump.size()) << label;
    ASSERT_EQ(steps.size(), started.size()) << label;
    auto gap = 0.0;
    for (auto n = std::size_t{0}; n < jump.size(); ++n) {
      EXPECT_TRUE(both[n].converged) << label << ", node " << n;
      const auto want = n < 2 ? growing[n] : SumOf(growing[n], steps[n - 2]);
      gap = std::max(gap, SnapshotDeparture(both[n], want));
    }
    worst = std::max(worst, gap);
    // the sea level solves to their tolerance (rtol 1e-12): measured
    // 2.5e-12 at worst
    EXPECT_LT(gap, 1e-10) << label;
    // the step moves the field
    EXPECT_GT(WorstDeparture(both[2].displacement, both[1].displacement), 1e-2)
        << label;
  };
  const auto dry = solver.NoOceanState();
  const auto& wet = setup.state;
  ViscoelasticOperator op1(setup.rheology, kLmax);
  for (const auto* background : {&wet, &dry})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler,
                      Scheme::Sdirk2}) {
    const auto label = std::string(background == &dry ? "no ocean" : "ocean") +
                       ", scheme " + std::to_string(static_cast<int>(scheme));
    const auto growing = LoadsThrough<Evolution::Forcing>(setup, kyr, 1.0);
    const auto both = StepsAt(setup, onset, kyr, &growing);
    const auto steps = StepsAt<Evolution::Forcing>(setup, onset, kyr, nullptr);
    Evolution evolution(op1, solver, *background,
                        {.scheme = scheme,
                         .rotation = background == &wet
                                         ? RotationalFeedback::Full
                                         : RotationalFeedback::None,
                         .rtol = 1e-12});
    check(evolution.Run(jump, both), evolution.Run(jump, growing),
          evolution.Run(started, steps), label);
  }
  const auto delta = [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto growing3 = LoadsThrough<Evolution3D::Forcing>(setup, kyr, 1.0);
  const auto both3 = StepsAt(setup, onset, kyr, &growing3);
  const auto steps3 = StepsAt<Evolution3D::Forcing>(setup, onset, kyr, nullptr);
  Viscoelastic3D op3(setup.rheology, kLmax, delta);
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::Sdirk2}) {
    Evolution3D evolution3(
        op3, solver, wet,
        {.scheme = scheme, .rtol = 1e-12, .coupled_rtol = 1e-12});
    check(evolution3.Run(jump, both3), evolution3.Run(jump, growing3),
          evolution3.Run(started, steps3),
          "3D, lateral viscosity, scheme " +
              std::to_string(static_cast<int>(scheme)));
  }
  auto options = Viscoelastic3D::Options{};
  options.solve_rtol = 1e-12;
  options.moduli = Viscoelastic3D::Options::LateralModuli{
      .shear =
          [](double r, double theta, double phi) {
            return 0.1 * (0.5 + r * r) * std::sin(theta) * std::cos(theta) *
                   std::cos(phi);
          },
      .bulk = [](double r, double theta,
                 double) { return 0.05 * (1.0 - 0.5 * r) * std::cos(theta); }};
  Viscoelastic3D moduli(setup.rheology, kLmax, delta, options);
  Evolution3D lateral(moduli, solver, wet,
                      {.rtol = 1e-12, .coupled_rtol = 1e-12});
  check(lateral.Run(jump, both3), lateral.Run(jump, growing3),
        lateral.Run(started, steps3), "3D, lateral viscosity and moduli");
  std::cout << std::format("[step load] worst departure {:.2e}\n", worst);
}

TEST(SurfaceJumps, AnIceStepUnderMigratingShorelinesIsTheLimitOfARamp) {
  // A step melt of the ice at a repeated node under migrating shorelines,
  // a step load beside it (Forcing::iceThicknessBefore, directLoadBefore):
  // the zero-length step is the nonlinear sea level equation on the
  // unrelaxed body with the memory held, as at any node. It is the limit
  // of the runs whose grid has a node eps before the jump instead of its
  // left limit, where the steps are read as at any time and taken over a
  // step of eps: the gap falls as eps, the radial run and the 3D run under
  // lateral viscosity.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  const auto onset = 0.3 * kyr;
  const auto jump =
      std::vector<double>{0.0, onset, onset, 0.8 * kyr, 2.0 * kyr};
  // the ice melting slowly, a fifth of it gone at the onset at a stroke
  const auto ice = [&](double t, bool stepped) {
    auto out = setup.state.iceThickness;
    for (auto& v : out) v *= (1.0 - 0.05 * t / kyr) * (stepped ? 0.8 : 1.0);
    return out;
  };
  const auto melting = [&](double t) { return ice(t, t >= onset); };
  auto growing = LoadsThrough<Evolution::Forcing>(setup, kyr, 1.0);
  auto forcing = StepsAt(setup, onset, kyr, &growing);
  forcing.iceThicknessBefore = [&](double t) { return ice(t, t > onset); };
  auto growing3 = LoadsThrough<Evolution3D::Forcing>(setup, kyr, 1.0);
  auto forcing3 = StepsAt(setup, onset, kyr, &growing3);
  forcing3.iceThicknessBefore = forcing.iceThicknessBefore;
  auto gapsOf = [&](auto& evolution, const auto& with, std::span<const double>
                                                           epsilons) {
    const auto exact = evolution.RunMigrating(jump, melting, with);
    for (const auto& snapshot : exact) EXPECT_TRUE(snapshot.converged);
    EXPECT_GT(WorstDeparture(exact[2].relativeSeaLevelChange,
                             exact[1].relativeSeaLevelChange),
              1e-2);
    auto gaps = std::vector<double>{};
    for (const auto eps : epsilons) {
      auto near = jump;
      near[1] = onset - eps * kyr;
      const auto ramped = evolution.RunMigrating(near, melting, with);
      auto gap = 0.0;
      for (auto n = std::size_t{2}; n < jump.size(); ++n) {
        gap = std::max(gap, SnapshotDeparture(ramped[n], exact[n]));
      }
      gaps.push_back(gap);
    }
    return gaps;
  };
  ViscoelasticOperator op1(setup.rheology, kLmax);
  Evolution evolution(op1, solver, setup.state,
                      {.rtol = 1e-12, .max_nonlinear_iterations = 200});
  const auto gaps =
      gapsOf(evolution, forcing, std::vector<double>{1e-2, 1e-3, 1e-4});
  // first order in eps: measured 5.96e-3, 5.94e-4, 5.94e-5 (3D 9.4e-2,
  // 9.3e-3)
  for (auto k = std::size_t{0}; k + 1 < gaps.size(); ++k) {
    EXPECT_GT(gaps[k] / gaps[k + 1], 8.0) << gaps[k] << " " << gaps[k + 1];
    EXPECT_LT(gaps[k] / gaps[k + 1], 12.0) << gaps[k] << " " << gaps[k + 1];
  }
  EXPECT_LT(gaps.back(), 1e-3);
  Viscoelastic3D op3(setup.rheology, kLmax,
                     [](double, double theta, double phi) {
                       return 0.6 * std::sin(theta) * std::cos(theta) *
                              std::cos(phi);
                     });
  Evolution3D evolution3(op3, solver, setup.state,
                         {.rtol = 1e-12, .coupled_rtol = 1e-12});
  const auto gaps3 = gapsOf(evolution3, forcing3, std::vector<double>{1e-2, 1e-3});
  EXPECT_GT(gaps3[0] / gaps3[1], 8.0) << gaps3[0] << " " << gaps3[1];
  EXPECT_LT(gaps3[0] / gaps3[1], 12.0) << gaps3[0] << " " << gaps3[1];
  std::cout << std::format(
      "[ice step] gaps {:.3e} {:.3e} {:.3e}; 3D {:.3e} {:.3e}\n", gaps[0],
      gaps[1], gaps[2], gaps3[0], gaps3[1]);
}

TEST(SurfaceJumps, ALeftLimitWithoutItsTermIsRefused) {
  // A left limit is of a term the forcing gives, and the ice thickness's
  // of a migrating run alone: refused otherwise, each entry alike.
  auto setup = Setup::Make();
  const auto& solver = setup.solver;
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, kyr, kyr, 2.0 * kyr};
  const auto load = [&](double) { return setup.MeltLoad(-0.1); };
  ViscoelasticOperator op1(setup.rheology, kLmax);
  Evolution evolution(op1, solver, setup.state);
  EXPECT_THROW(static_cast<void>(evolution.Run(
                   times, Evolution::Forcing{.directLoadBefore = load})),
               std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(evolution.Run(
          times, Evolution::Forcing{.directLoad = load,
                                    .iceThicknessBefore = load})),
      std::invalid_argument);
  Viscoelastic3D op3(setup.rheology, kLmax,
                     [](double, double, double) { return 0.0; });
  Evolution3D evolution3(op3, solver, setup.state);
  EXPECT_THROW(
      static_cast<void>(evolution3.Run(
          times, Evolution3D::Forcing{.directLoadOutsideWaterBudgetBefore =
                                          load})),
      std::invalid_argument);
}

