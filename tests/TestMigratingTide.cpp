// A prescribed tide under migrating shorelines, 1D and 3D: the tide
// enters each node's nonlinear sea level solve as it enters the linear
// one, its potential in the sea level change and so in the flotation that
// places the shorelines. On a toy world of steep coasts (deep ocean, high
// land) no shoreline moves, and the migrating run is the frozen run under
// the same tide and loads; on one of shallow shelves a large tide moves
// the shorelines, and each node's ocean function is flotation's for the
// sea surface the run reports, the ocean's water conserved and the run
// converging as the solve's tolerance tightens; in the small the response
// is linear in the tide while no cell flips; a zero tide is no tide; the
// runs resume from their checkpoints as the uninterrupted ones; a tide
// below degree 2 is refused wherever a tide enters; and Sdirk2's stages
// close the nonlinear equation at their own times.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
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

using Complex = std::complex<double>;
using Field = std::vector<double>;
using Tide = std::function<std::vector<Complex>(double)>;

/// prem8 with a Maxwell mantle of 1e21 Pa s, its elastic sea level solver
/// at lmax.
struct Body {
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  int lmax = 0;

  static Body Make(int lmax) {
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
    ViscoelasticOperator op(rheology, lmax);
    GRDSolver solver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
                     SeaLevelParameters{});
    return Body{std::move(model), std::move(material), std::move(rheology),
                std::move(solver), lmax};
  }

  [[nodiscard]] double Metre() const {
    return 1.0 / solver.SolverScales().Length();
  }
  [[nodiscard]] double Kyr() const {
    return 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  }
  [[nodiscard]] double Potential() const {  // 1 m^2 s^-2, solver units
    return 1.0 / solver.SolverScales().Factor(PotentialDimensions);
  }
  [[nodiscard]] double Sigma() const {  // 1 kg m^-2, solver units
    return 1.0 /
           solver.SolverScales().Factor(Dimensions{.mass = 1, .length = -2});
  }
};

/// The continents' shape: positive on land.
double Continents(double theta, double phi) {
  return 0.6 * std::cos(theta) + 0.5 * std::sin(theta) * std::cos(phi - 1.0) +
         0.3 * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi) - 0.2;
}

/// Steep coasts: an ocean 3000 m deep, land 1500 m high, and a 2000 m ice
/// sheet on the northern land, grounded far above the sea.
struct SteepWorld {
  Field ice0, sea0;
  GRDSolver::State state;

  static SteepWorld Make(const Body& body) {
    const auto& grid = body.solver.Grid();
    const auto metre = body.Metre();
    SteepWorld out;
    out.sea0 = grid.FromFunction([&](double theta, double phi) {
      return (Continents(theta, phi) > 0.0 ? -1500.0 : 3000.0) * metre;
    });
    out.ice0 = grid.FromFunction([&](double theta, double phi) {
      return Continents(theta, phi) > 0.0 && theta < 1.0 ? 2000.0 * metre : 0.0;
    });
    out.state = body.solver.MakeState(out.ice0, out.sea0);
    return out;
  }

  /// The ice sheet thinning by a tenth over `span`.
  [[nodiscard]] std::function<Field(double)> Thinning(double span) const {
    return [ice = ice0, span](double t) {
      auto out = ice;
      for (auto& v : out) v *= 1.0 - 0.1 * t / span;
      return out;
    };
  }
};

/// Shallow shelves: a sea floor rising gently to the coasts and a land
/// rising gently from them, sea level at most 40 m from zero, no ice.
struct ShelfWorld {
  Field ice0, sea0;
  GRDSolver::State state;

  static ShelfWorld Make(const Body& body) {
    const auto& grid = body.solver.Grid();
    ShelfWorld out;
    out.sea0 = grid.FromFunction([&](double theta, double phi) {
      return -40.0 * body.Metre() * Continents(theta, phi);
    });
    out.ice0.assign(grid.FieldSize(), 0.0);
    out.state = body.solver.MakeState(out.ice0, out.sea0);
    return out;
  }
};

/// A tide of degrees 2 and 3, `amplitude` m^2 s^-2, growing and turning
/// over `span`.
Tide TideOf(const Body& body, double amplitude, double span) {
  const auto a = amplitude * body.Potential();
  return [&grid = body.solver.Grid(), a, span](double t) {
    auto psi = grid.ZeroCoefficients();
    const auto s = t / span;
    psi[SurfaceGrid::Index(2, 0)] = a * (0.6 + s);
    psi[SurfaceGrid::Index(2, 1)] = a * Complex(0.4, -0.3 * s);
    psi[SurfaceGrid::Index(2, 2)] = a * Complex(-0.2 * s, 0.5);
    psi[SurfaceGrid::Index(3, 1)] = a * Complex(0.3, 0.2);
    return psi;
  };
}

/// A smooth load on the land, `amplitude` kg m^-2, growing over `span`.
std::function<Field(double)> LandLoad(const Body& body, double amplitude,
                                      double span) {
  const auto pattern = body.solver.Grid().FromFunction([](double theta,
                                                          double phi) {
    return std::max(0.0, Continents(theta, phi)) * std::cos(phi);
  });
  const auto a = amplitude * body.Sigma();
  return [pattern, a, span](double t) {
    auto out = pattern;
    for (auto& v : out) v *= a * (0.5 + t / span);
    return out;
  };
}

double Peak(std::span<const double> x) {
  auto out = 0.0;
  for (auto v : x) out = std::max(out, std::abs(v));
  return out;
}

double Departure(std::span<const double> got, std::span<const double> want) {
  auto out = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    out = std::max(out, std::abs(got[i] - want[i]));
  }
  return out;
}

double Departure(std::span<const Complex> got, std::span<const Complex> want) {
  auto out = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    out = std::max(out, std::abs(got[i] - want[i]));
  }
  return out;
}

double Peak(std::span<const Complex> x) {
  auto out = 0.0;
  for (auto v : x) out = std::max(out, std::abs(v));
  return out;
}

/// Every field of a snapshot against another's, relative to the
/// latter's peak; the angular velocity against its peak component.
template <class Snapshot>
double Worst(const Snapshot& got, const Snapshot& want) {
  auto worst = 0.0;
  for (auto field : {&Snapshot::relativeSeaLevelChange, &Snapshot::displacement,
                     &Snapshot::potential, &Snapshot::tangentialPotential}) {
    const auto& w = want.*field;
    if (Peak(w) == 0.0) continue;
    worst = std::max(worst, Departure(got.*field, w) / Peak(w));
  }
  const auto omega = Peak(want.angularVelocity);
  if (omega > 0.0) {
    worst = std::max(worst,
                     Departure(got.angularVelocity, want.angularVelocity) / omega);
  }
  if (!want.tideCoefficients.empty()) {
    worst = std::max(worst, Departure(got.tideCoefficients, want.tideCoefficients) /
                                Peak(want.tideCoefficients));
  }
  return worst;
}

/// The ocean function flotation gives on the sea level S0 + xi and the
/// ice I: rho_w S - rho_i I > 0.
Field Flotation(const GRDSolver& solver, const Field& sea0, const Field& xi,
                const Field& ice) {
  auto sea = sea0;
  for (auto i = std::size_t{0}; i < sea.size(); ++i) sea[i] += xi[i];
  return solver.MakeState(ice, sea).oceanFunction;
}

/// The change of the ocean's water from the background, int rho_w (C (S0 +
/// xi) - C0 S0) dS, against int rho_w C |xi| dS, its scale: the grounded
/// ice is unchanged.
std::pair<double, double> WaterChange(const GRDSolver& solver,
                                      const GRDSolver::State& state,
                                      const Field& ocean, const Field& xi) {
  const auto& grid = solver.Grid();
  auto change = Field(xi.size()), scale = Field(xi.size());
  for (auto i = std::size_t{0}; i < xi.size(); ++i) {
    change[i] = ocean[i] * (state.seaLevel[i] + xi[i]) -
                state.oceanFunction[i] * state.seaLevel[i];
    scale[i] = ocean[i] * std::abs(xi[i]);
  }
  return {grid.Integrate(change), grid.Integrate(scale)};
}

std::vector<double> Times(const Body& body) {
  const auto kyr = body.Kyr();
  return {0.0, 0.5 * kyr, 1.5 * kyr, 3.0 * kyr, 5.0 * kyr};
}

}  // namespace

TEST(MigratingTide, WhereNoShorelineMovesTheRunIsTheFrozenOne) {
  // Steep coasts under a tide of a few metres, a thinning ice sheet and a
  // load on the land: no cell of the ocean function changes, and the
  // migrating run is the frozen run under the same forcing, every field
  // and the tide coefficients the steps saw, with and without the
  // rotational feedback, through time and in the static problem (one
  // node), to the solves' tolerance (both at 1e-13).
  const auto body = Body::Make(16);
  const auto world = SteepWorld::Make(body);
  const auto times = Times(body);
  const auto span = times.back();
  const auto tide = TideOf(body, 30.0, span);
  const auto extra = LandLoad(body, 1e4, span);
  const auto thickness = world.Thinning(span);
  const auto& solver = body.solver;
  ViscoelasticOperator op(body.rheology, body.lmax);
  // the frozen run's direct load: the grounded ice's change and the load
  const auto frozenLoad = [&](double t) {
    auto change = thickness(t);
    for (auto i = std::size_t{0}; i < change.size(); ++i) {
      change[i] -= world.ice0[i];
    }
    auto out = solver.DirectLoadFromIceThicknessChange(world.state, change);
    const auto given = extra(t);
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += given[i];
    return out;
  };
  const auto one = std::vector<double>{0.0};
  for (const auto rotation :
       {RotationalFeedback::None, RotationalFeedback::Full}) {
    const auto label = std::format("rotation {}", static_cast<int>(rotation));
    Evolution evolution(op, solver, world.state,
                        {.rotation = rotation,
                         .rtol = 1e-13,
                         .max_nonlinear_iterations = 400,
                         .record_stride = 1});
    // through time, and the static problem
    for (const auto* grid : {&times, &one}) {
      const auto& at = *grid;
      const auto frozen = evolution.Run(
          at, Evolution::Forcing{.directLoad = frozenLoad, .tide = tide});
      const auto moving = evolution.RunMigrating(
          at, thickness, Evolution::Forcing{.directLoad = extra, .tide = tide});
      ASSERT_EQ(moving.size(), at.size());
      for (auto n = std::size_t{0}; n < at.size(); ++n) {
        ASSERT_TRUE(moving[n].converged) << label << ", node " << n;
        // flotation's 0/1: rounding changes it only at a cell on the threshold
        ASSERT_EQ(moving[n].oceanFunction, world.state.oceanFunction)
            << label << ", node " << n;
        EXPECT_GT(Peak(moving[n].relativeSeaLevelChange), 0.1 * body.Metre());
        // measured below 8e-14
        EXPECT_LT(Worst(moving[n], frozen[n]), 1e-11)
            << label << ", node " << n << " of " << at.size();
      }
    }
  }
}

TEST(MigratingTide, WhereNoShorelineMovesTheRunIsTheFrozenOneIn3D) {
  // The 3D gate of the steep coasts under lateral viscosity: no cell of
  // the ocean function changes, and the migrating run is the frozen run
  // under the same tide, ice and load, with and without the rotational
  // feedback, to the solves' tolerance.
  const auto body = Body::Make(8);
  const auto world = SteepWorld::Make(body);
  const auto times = Times(body);
  const auto span = times.back();
  const auto tide = TideOf(body, 30.0, span);
  const auto extra = LandLoad(body, 1e4, span);
  const auto thickness = world.Thinning(span);
  const auto& solver = body.solver;
  const auto frozenLoad = [&](double t) {
    auto change = thickness(t);
    for (auto i = std::size_t{0}; i < change.size(); ++i) {
      change[i] -= world.ice0[i];
    }
    auto out = solver.DirectLoadFromIceThicknessChange(world.state, change);
    const auto given = extra(t);
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += given[i];
    return out;
  };
  Viscoelastic3D lateral(body.rheology, body.lmax,
                         [](double, double theta, double phi) {
                           return 0.8 * std::sin(theta) * std::cos(theta) *
                                  std::cos(phi);
                         });
  for (const auto rotation :
       {RotationalFeedback::None, RotationalFeedback::Full}) {
    const auto label = std::format("rotation {}", static_cast<int>(rotation));
    Evolution3D evolution(lateral, solver, world.state,
                          {.rotation = rotation,
                           .rtol = 1e-13,
                           .coupled_rtol = 1e-12,
                           .max_nonlinear_iterations = 400,
                           .record_stride = 1});
    const auto frozen = evolution.Run(
        times, Evolution3D::Forcing{.directLoad = frozenLoad, .tide = tide});
    const auto moving = evolution.RunMigrating(
        times, thickness,
        Evolution3D::Forcing{.directLoad = extra, .tide = tide});
    ASSERT_EQ(moving.size(), times.size());
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      ASSERT_TRUE(moving[n].converged) << label << ", node " << n;
      // flotation's 0/1: rounding changes it only at a cell on the threshold
      ASSERT_EQ(moving[n].oceanFunction, world.state.oceanFunction)
          << label << ", node " << n;
      // measured below 1e-13
      EXPECT_LT(Worst(moving[n], frozen[n]), 1e-11) << label << ", node " << n;
    }
  }
}

TEST(MigratingTide, Sdirk2StagesSolveTheNonlinearEquationAtTheirTimes) {
  // Sdirk2 closes each stage by the nonlinear sea level equation at the
  // stage's time (the ice, the load, the tide and the land there). On the
  // steep coasts no cell flips, and the migrating run is the frozen run
  // under the same tide, ice and load, radial and under lateral viscosity,
  // with and without the rotational feedback, to the solves' tolerance; on
  // the shallow shelves a large tide moves the shorelines, and each node's
  // ocean function is flotation's on the background's sea level plus the
  // change the run reports, the ocean's water the background's.
  const auto body = Body::Make(8);
  const auto world = SteepWorld::Make(body);
  const auto times = Times(body);
  const auto span = times.back();
  const auto tide = TideOf(body, 30.0, span);
  const auto extra = LandLoad(body, 1e4, span);
  const auto thickness = world.Thinning(span);
  const auto& solver = body.solver;
  const auto frozenLoad = [&](double t) {
    auto change = thickness(t);
    for (auto i = std::size_t{0}; i < change.size(); ++i) {
      change[i] -= world.ice0[i];
    }
    auto out = solver.DirectLoadFromIceThicknessChange(world.state, change);
    const auto given = extra(t);
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += given[i];
    return out;
  };
  ViscoelasticOperator radial(body.rheology, body.lmax);
  Viscoelastic3D lateral(
      body.rheology, body.lmax, [](double, double theta, double phi) {
        return 0.8 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  for (const auto rotation :
       {RotationalFeedback::None, RotationalFeedback::Full}) {
    const auto label = std::format("rotation {}", static_cast<int>(rotation));
    Evolution evolution(radial, solver, world.state,
                        {.scheme = Scheme::Sdirk2,
                         .rotation = rotation,
                         .rtol = 1e-13,
                         .max_nonlinear_iterations = 400});
    const auto frozen = evolution.Run(
        times, Evolution::Forcing{.directLoad = frozenLoad, .tide = tide});
    const auto moving = evolution.RunMigrating(
        times, thickness,
        Evolution::Forcing{.directLoad = extra, .tide = tide});
    Evolution3D evolution3(lateral, solver, world.state,
                           {.scheme = Scheme::Sdirk2,
                            .rotation = rotation,
                            .rtol = 1e-13,
                            .coupled_rtol = 1e-12,
                            .max_nonlinear_iterations = 400});
    const auto frozen3 = evolution3.Run(
        times, Evolution3D::Forcing{.directLoad = frozenLoad, .tide = tide});
    const auto moving3 = evolution3.RunMigrating(
        times, thickness,
        Evolution3D::Forcing{.directLoad = extra, .tide = tide});
    ASSERT_EQ(moving.size(), times.size());
    ASSERT_EQ(moving3.size(), times.size());
    auto worst = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      worst = std::max(
          {worst, Worst(moving[n], frozen[n]), Worst(moving3[n], frozen3[n])});
    }
    std::cout << std::format("[sdirk2 migrating] {}: {:.2e}\n", label, worst);
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      ASSERT_TRUE(moving[n].converged && moving3[n].converged)
          << label << ", node " << n;
      ASSERT_EQ(moving[n].oceanFunction, world.state.oceanFunction)
          << label << ", node " << n;
      ASSERT_EQ(moving3[n].oceanFunction, world.state.oceanFunction)
          << label << ", node " << n;
      // measured below 1e-13
      EXPECT_LT(Worst(moving[n], frozen[n]), 1e-11) << label << ", node " << n;
      EXPECT_LT(Worst(moving3[n], frozen3[n]), 1e-11)
          << label << ", 3D, node " << n;
    }
  }

  const auto shelf = ShelfWorld::Make(body);
  const auto still = [&](double) { return shelf.ice0; };
  Evolution shelves(radial, solver, shelf.state,
                    {.scheme = Scheme::Sdirk2,
                     .rtol = 1e-12,
                     .max_nonlinear_iterations = 400});
  const auto run = shelves.RunMigrating(
      times, still, Evolution::Forcing{.tide = TideOf(body, 120.0, span)});
  auto moved = std::size_t{0};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    const auto& node = run[n];
    ASSERT_TRUE(node.converged) << "shelves, node " << n;
    EXPECT_EQ(
        node.oceanFunction,
        Flotation(solver, shelf.sea0, node.relativeSeaLevelChange, shelf.ice0))
        << "shelves, node " << n;
    for (auto i = std::size_t{0}; i < node.oceanFunction.size(); ++i) {
      if (node.oceanFunction[i] != shelf.state.oceanFunction[i]) ++moved;
    }
    const auto [water, scale] = WaterChange(
        solver, shelf.state, node.oceanFunction, node.relativeSeaLevelChange);
    EXPECT_LT(std::abs(water), 1e-13 * scale) << "shelves, node " << n;
  }
  std::cout << std::format("[sdirk2 migrating] shelves: {} cells moved\n",
                           moved);
  // measured: 24 cells moved, over the nodes
  EXPECT_GT(moved, std::size_t{10});
}

TEST(MigratingTide, ALargeTideMovesTheShorelinesByFlotation) {
  // Shallow shelves under a tide of some 10 m and nothing else: the
  // shorelines move, each node's ocean function is flotation's on the
  // background's sea level plus the change the run reports, the ocean's
  // water is the background's to rounding, and the run converges as the
  // solve's tolerance tightens.
  const auto body = Body::Make(16);
  const auto world = ShelfWorld::Make(body);
  const auto times = Times(body);
  const auto tide = TideOf(body, 120.0, times.back());
  const auto& solver = body.solver;
  ViscoelasticOperator op(body.rheology, body.lmax);
  const auto still = [&](double) { return world.ice0; };
  auto runAt = [&](double rtol, RotationalFeedback rotation) {
    Evolution evolution(op, solver, world.state,
                        {.rotation = rotation,
                         .rtol = rtol,
                         .max_nonlinear_iterations = 400});
    return evolution.RunMigrating(times, still,
                                  Evolution::Forcing{.tide = tide});
  };
  for (const auto rotation :
       {RotationalFeedback::None, RotationalFeedback::Full}) {
    const auto label = std::format("rotation {}", static_cast<int>(rotation));
    const auto run = runAt(1e-12, rotation);
    auto moved = std::size_t{0};
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      const auto& node = run[n];
      ASSERT_TRUE(node.converged) << label << ", node " << n;
      // flotation's 0/1: rounding changes it only at a cell on the threshold
      EXPECT_EQ(node.oceanFunction,
                Flotation(solver, world.sea0, node.relativeSeaLevelChange,
                          world.ice0))
          << label << ", node " << n;
      for (auto i = std::size_t{0}; i < node.oceanFunction.size(); ++i) {
        if (node.oceanFunction[i] != world.state.oceanFunction[i]) ++moved;
      }
      const auto [water, scale] = WaterChange(
          solver, world.state, node.oceanFunction, node.relativeSeaLevelChange);
      EXPECT_LT(std::abs(water), 1e-13 * scale) << label << ", node " << n;
    }
    EXPECT_GT(moved, std::size_t{20}) << label;
    // tightening the tolerance: the departures from the tightest run fall
    // with it (measured at 0.2 to 0.3 of the tolerance)
    auto previous = 1e300;
    for (const auto rtol : {1e-4, 1e-6, 1e-8, 1e-10}) {
      const auto looser = runAt(rtol, rotation);
      auto worst = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        worst = std::max(worst, Departure(looser[n].relativeSeaLevelChange,
                                          run[n].relativeSeaLevelChange) /
                                    Peak(run[n].relativeSeaLevelChange));
      }
      EXPECT_LT(worst, rtol) << label << ", rtol " << rtol;
      EXPECT_LE(worst, previous) << label << ", rtol " << rtol;
      previous = worst;
    }
  }
}

TEST(MigratingTide, ALargeTideMovesTheShorelinesByFlotationIn3D) {
  // The shelves under lateral viscosity: the 3D run's ocean function at
  // each node is flotation's for the sea level it reports, and its water
  // the background's.
  const auto body = Body::Make(8);
  const auto world = ShelfWorld::Make(body);
  const auto times = Times(body);
  const auto tide = TideOf(body, 120.0, times.back());
  const auto& solver = body.solver;
  Viscoelastic3D lateral(body.rheology, body.lmax,
                         [](double, double theta, double phi) {
                           return 0.8 * std::sin(theta) * std::cos(theta) *
                                  std::cos(phi);
                         });
  Evolution3D evolution(lateral, solver, world.state,
                        {.rtol = 1e-12,
                         .coupled_rtol = 1e-11,
                         .max_nonlinear_iterations = 400});
  const auto run = evolution.RunMigrating(
      times, [&](double) { return world.ice0; },
      Evolution3D::Forcing{.tide = tide});
  auto moved = std::size_t{0};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    const auto& node = run[n];
    ASSERT_TRUE(node.converged) << "node " << n;
    // flotation's 0/1: rounding changes it only at a cell on the threshold
    EXPECT_EQ(node.oceanFunction,
              Flotation(solver, world.sea0, node.relativeSeaLevelChange,
                        world.ice0))
        << "node " << n;
    for (auto i = std::size_t{0}; i < node.oceanFunction.size(); ++i) {
      if (node.oceanFunction[i] != world.state.oceanFunction[i]) ++moved;
    }
    const auto [water, scale] = WaterChange(
        solver, world.state, node.oceanFunction, node.relativeSeaLevelChange);
    EXPECT_LT(std::abs(water), 1e-13 * scale) << "node " << n;
  }
  EXPECT_GT(moved, std::size_t{5});
}

TEST(MigratingTide, InTheSmallTheResponseIsLinearInTheTide) {
  // About a run whose shorelines move under a large tide, a further tide
  // of amplitude epsilon flips no cell, and the response is linear in it:
  // the change at 2 epsilon is twice the change at epsilon, to parts in
  // 1e7 of the change (measured below 5e-11; the solves at 1e-13, the
  // change some 1e-4 of the run, so that the solves' tolerance alone
  // allows some 1e-9 of it).
  const auto body = Body::Make(16);
  const auto world = ShelfWorld::Make(body);
  const auto times = Times(body);
  const auto base = TideOf(body, 120.0, times.back());
  const auto& grid = body.solver.Grid();
  const auto perturbation = [&](double t) {
    auto psi = grid.ZeroCoefficients();
    psi[SurfaceGrid::Index(2, 1)] =
        body.Potential() * Complex(0.7, 0.2 + t / times.back());
    psi[SurfaceGrid::Index(4, 3)] = body.Potential() * Complex(-0.3, 0.5);
    return psi;
  };
  ViscoelasticOperator op(body.rheology, body.lmax);
  for (const auto rotation :
       {RotationalFeedback::None, RotationalFeedback::Full}) {
    const auto label = std::format("rotation {}", static_cast<int>(rotation));
    Evolution evolution(op, body.solver, world.state,
                        {.rotation = rotation,
                         .rtol = 1e-13,
                         .max_nonlinear_iterations = 400});
    auto runWith = [&](double epsilon) {
      const Tide tide = [&, epsilon](double t) {
        auto psi = base(t);
        const auto d = perturbation(t);
        for (auto i = std::size_t{0}; i < psi.size(); ++i) {
          psi[i] += epsilon * d[i];
        }
        return psi;
      };
      return evolution.RunMigrating(
          times, [&](double) { return world.ice0; },
          Evolution::Forcing{.tide = tide});
    };
    const auto epsilon = 0.01;
    const auto r0 = runWith(0.0);
    const auto r1 = runWith(epsilon);
    const auto r2 = runWith(2.0 * epsilon);
    auto moved = std::size_t{0};
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      // flotation's 0/1: rounding changes it only at a cell on the threshold
      ASSERT_EQ(r1[n].oceanFunction, r0[n].oceanFunction) << label << n;
      ASSERT_EQ(r2[n].oceanFunction, r0[n].oceanFunction) << label << n;
      for (auto i = std::size_t{0}; i < r0[n].oceanFunction.size(); ++i) {
        if (r0[n].oceanFunction[i] != world.state.oceanFunction[i]) ++moved;
      }
      for (auto field : {&Evolution::Snapshot::relativeSeaLevelChange,
                         &Evolution::Snapshot::displacement,
                         &Evolution::Snapshot::potential}) {
        const auto& a = r0[n].*field;
        const auto& b = r1[n].*field;
        const auto& c = r2[n].*field;
        auto once = Field(a.size()), twice = Field(a.size());
        for (auto i = std::size_t{0}; i < a.size(); ++i) {
          once[i] = b[i] - a[i];
          twice[i] = c[i] - a[i];
        }
        ASSERT_GT(Peak(once), 1e-6 * Peak(a)) << label << ", node " << n;
        auto residual = Field(a.size());
        for (auto i = std::size_t{0}; i < a.size(); ++i) {
          residual[i] = twice[i] - 2.0 * once[i];
        }
        EXPECT_LT(Peak(residual), 1e-7 * Peak(once)) << label << ", node " << n;
      }
    }
    EXPECT_GT(moved, std::size_t{20}) << label;
  }
}

TEST(MigratingTide, AZeroTideIsNoTide) {
  // A tide whose coefficients vanish leaves a migrating run as it is
  // without one, field for field, 1D and 3D.
  const auto body = Body::Make(8);
  const auto world = SteepWorld::Make(body);
  const auto times = Times(body);
  const auto thickness = world.Thinning(times.back());
  const auto extra = LandLoad(body, 1e4, times.back());
  const Tide zero = [&](double) { return body.solver.Grid().ZeroCoefficients(); };
  ViscoelasticOperator op(body.rheology, body.lmax);
  Evolution evolution(op, body.solver, world.state, {.record_stride = 1});
  const auto without =
      evolution.RunMigrating(times, thickness, Evolution::Forcing{.directLoad = extra});
  const auto with = evolution.RunMigrating(
      times, thickness, Evolution::Forcing{.directLoad = extra, .tide = zero});
  // a vanishing tide takes the path of no tide: bitwise
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_EQ(with[n].relativeSeaLevelChange, without[n].relativeSeaLevelChange);
    EXPECT_EQ(with[n].displacement, without[n].displacement);
    EXPECT_EQ(with[n].potential, without[n].potential);
    EXPECT_EQ(with[n].tangentialPotential, without[n].tangentialPotential);
    EXPECT_EQ(with[n].angularVelocity, without[n].angularVelocity);
    EXPECT_EQ(with[n].tideCoefficients, without[n].tideCoefficients);
    EXPECT_EQ(with[n].oceanFunction, without[n].oceanFunction);
  }
  Viscoelastic3D lateral(body.rheology, body.lmax,
                         [](double, double theta, double) {
                           return 0.5 * std::cos(theta);
                         });
  Evolution3D evolution3(lateral, body.solver, world.state, {});
  const auto without3 = evolution3.RunMigrating(
      times, thickness, Evolution3D::Forcing{.directLoad = extra});
  const auto with3 = evolution3.RunMigrating(
      times, thickness, Evolution3D::Forcing{.directLoad = extra, .tide = zero});
  // a vanishing tide takes the path of no tide: bitwise
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_EQ(with3[n].relativeSeaLevelChange, without3[n].relativeSeaLevelChange);
    EXPECT_EQ(with3[n].displacement, without3[n].displacement);
    EXPECT_EQ(with3[n].angularVelocity, without3[n].angularVelocity);
  }
}

TEST(MigratingTide, ATideBelowDegreeTwoIsRefused) {
  // A tide is taken relative to the centre of mass, degree 2 and up: a
  // degree-0 or degree-1 part is refused, the message naming the degree,
  // by the linear and the nonlinear sea level solves, by the evolutions at
  // the node that reads it (1D and 3D, frozen and migrating shorelines),
  // by the operators' tide channels at degree 1 (their degree 0 carries a
  // centrifugal change), by the static body tide's forces and by the
  // tangent-linear runs' directions, 1D and 3D; the same tide without that
  // part is taken. A vector too short for the grid is refused by its size.
  const auto body = Body::Make(8);
  const auto world = ShelfWorld::Make(body);
  const auto& grid = body.solver.Grid();
  const auto potential = body.Potential();
  // whether a call is refused for a part of the given degree
  const auto refused = [](auto&& call, int degree) {
    try {
      call();
    } catch (const std::invalid_argument& error) {
      return std::string(error.what())
                 .find(std::format("no degree-{} part", degree)) !=
             std::string::npos;
    }
    return false;
  };
  // a degree-2 tide with a part at (l, m)
  const auto with = [&](int l, int m) {
    auto psi = grid.ZeroCoefficients();
    psi[SurfaceGrid::Index(2, 1)] = Complex(0.5, -0.2) * potential;
    psi[SurfaceGrid::Index(l, m)] += Complex(1.0, m == 0 ? 0.0 : 0.3) * potential;
    return psi;
  };
  for (const auto& [l, m] : {std::pair{0, 0}, std::pair{1, 0}, std::pair{1, 1},
                             std::pair{2, 0}}) {
    const auto label = std::format("degree {} order {}", l, m);
    GRDSolver::GeneralisedForcing linear;
    linear.tide = with(l, m);
    GRDSolver::NonlinearForcing nonlinear;
    nonlinear.tide = with(l, m);
    if (l < 2) {
      EXPECT_TRUE(refused(
          [&] { (void)body.solver.SolveGeneralised(world.state, linear); }, l))
          << label;
      EXPECT_TRUE(refused(
          [&] { (void)body.solver.SolveNonlinear(world.state, nonlinear); }, l))
          << label;
    } else {
      EXPECT_TRUE(body.solver.SolveGeneralised(world.state, linear).converged);
      EXPECT_TRUE(
          body.solver.SolveNonlinear(world.state, nonlinear).response.converged);
    }
  }
  GRDSolver::NonlinearForcing shortened;
  shortened.tide = with(0, 0);
  shortened.tide.resize(3);
  EXPECT_THROW((void)body.solver.SolveNonlinear(world.state, shortened),
               std::invalid_argument);

  // the evolutions: a degree-1 part from the third node on
  const auto times = Times(body);
  const auto good = TideOf(body, 30.0, times.back());
  const Tide late = [&, good](double t) {
    auto psi = good(t);
    if (t > times[1]) psi[SurfaceGrid::Index(1, 1)] = Complex(0.0, potential);
    return psi;
  };
  const auto still = [&](double) { return world.ice0; };
  ViscoelasticOperator op(body.rheology, body.lmax);
  Evolution evolution(op, body.solver, world.state, {.record_stride = 1});
  EXPECT_TRUE(refused(
      [&] { (void)evolution.Run(times, Evolution::Forcing{.tide = late}); }, 1));
  EXPECT_TRUE(refused(
      [&] {
        (void)evolution.RunMigrating(times, still,
                                     Evolution::Forcing{.tide = late});
      },
      1));
  Viscoelastic3D lateral(body.rheology, body.lmax,
                         [](double, double theta, double) {
                           return 0.5 * std::cos(theta);
                         });
  Evolution3D evolution3(lateral, body.solver, world.state, {});
  EXPECT_TRUE(refused(
      [&] { (void)evolution3.Run(times, Evolution3D::Forcing{.tide = late}); },
      1));
  EXPECT_TRUE(refused(
      [&] {
        (void)evolution3.RunMigrating(times, still,
                                      Evolution3D::Forcing{.tide = late});
      },
      1));
  // and the same runs under the tide without it
  EXPECT_EQ(evolution.Run(times, Evolution::Forcing{.tide = good}).size(),
            times.size());
  EXPECT_EQ(evolution3.Run(times, Evolution3D::Forcing{.tide = good}).size(),
            times.size());

  // the operators' tide channels: degree 1 refused, degree 0 taken
  for (const auto degree : {0, 1}) {
    const LoadCallable radial =
        [degree](int l, double) -> std::optional<std::vector<double>> {
      if (l != degree) return std::nullopt;
      return std::vector<double>{1.0};
    };
    const Viscoelastic3D::LoadCallable3D spread =
        [degree](int l, double) -> std::optional<std::vector<Complex>> {
      if (l != degree) return std::nullopt;
      return std::vector<Complex>(static_cast<std::size_t>(l) + 1, 1.0);
    };
    if (degree == 1) {
      EXPECT_TRUE(refused(
          [&] { (void)op.SolveElastic(op.ZeroState(), {}, radial); }, 1));
      EXPECT_TRUE(refused(
          [&] { (void)lateral.SolveElastic(lateral.ZeroState(), {}, spread); },
          1));
    } else {
      EXPECT_NO_THROW((void)op.SolveElastic(op.ZeroState(), {}, radial));
      EXPECT_NO_THROW(
          (void)lateral.SolveElastic(lateral.ZeroState(), {}, spread));
    }
  }

  // the static body tide's forces
  ElasticPerturbation3D statics(body.material, body.lmax);
  EXPECT_TRUE(refused([&] { (void)statics.TideForces(with(1, 0)); }, 1));
  EXPECT_NO_THROW((void)statics.TideForces(with(2, 0)));

  // the tangent-linear runs' directions, refused before they march
  auto directions = std::vector<std::vector<Complex>>{};
  for (const auto t : times) directions.push_back(good(t));
  directions[2][SurfaceGrid::Index(1, 0)] = potential;
  TangentLinearSolver tangent(evolution, {.method = AdjointMethod::Discrete});
  EXPECT_TRUE(refused(
      [&] {
        (void)tangent.Run(times, TangentPerturbation{.tide = directions});
      },
      1));
  TangentLinearSolver3D tangent3(evolution3);
  EXPECT_TRUE(refused(
      [&] {
        (void)tangent3.Run(times, TangentPerturbation3D{.tide = directions});
      },
      1));
}

TEST(MigratingTide, AResumedRunIsTheUninterruptedOne) {
  // A migrating run under a tide that moves the shorelines checkpoints
  // its recurrence, the tide among the coefficients the steps saw, and a
  // fresh evolution resumed from a checkpoint marches the same history
  // node for node, bitwise on one thread, recording what the
  // uninterrupted run records; 1D and 3D.
  const auto body = Body::Make(8);
  const auto world = ShelfWorld::Make(body);
  auto times = std::vector<double>{};
  for (auto n = 0; n < 7; ++n) times.push_back(0.8 * n * body.Kyr());
  const auto tide = TideOf(body, 120.0, times.back());
  const auto still = [&](double) { return world.ice0; };
  const auto forcing = Evolution::Forcing{.tide = tide};
  ViscoelasticOperator op(body.rheology, body.lmax);
  {
    auto checkpoints = std::map<std::size_t, Evolution::Checkpoint>{};
    auto options = Evolution::Options{.record_stride = 2};
    options.checkpoint_stride = 3;
    options.on_checkpoint = [&](std::size_t node, const Evolution::Checkpoint& cp) {
      checkpoints.emplace(node, cp);
    };
    Evolution whole(op, body.solver, world.state, options);
    const auto full = whole.RunMigrating(times, still, forcing);
    ASSERT_TRUE(checkpoints.contains(3));
    auto moved = false;
    for (const auto& node : full) {
      moved = moved || node.oceanFunction != world.state.oceanFunction;
    }
    EXPECT_TRUE(moved);
    // the steps saw the tide: the checkpoint's coefficients are not the
    // centrifugal potential alone
    const auto spin = body.solver.CentrifugalCoefficients(full[3].angularVelocity);
    EXPECT_GT(Departure(checkpoints.at(3).tideCoefficients, spin),
              0.1 * body.Potential());
    auto resumedOptions = Evolution::Options{.record_stride = 2, .record_offset = 3};
    Evolution resumed(op, body.solver, world.state, resumedOptions);
    const auto tail = resumed.ResumeMigrating(
        checkpoints.at(3), std::vector<double>(times.begin() + 3, times.end()),
        still, forcing);
    ASSERT_EQ(tail.size(), times.size() - 4);
    // one process, the same transforms: bitwise
    for (auto k = std::size_t{0}; k < tail.size(); ++k) {
      const auto& a = full[4 + k];
      const auto& b = tail[k];
      EXPECT_EQ(a.relativeSeaLevelChange, b.relativeSeaLevelChange) << 4 + k;
      EXPECT_EQ(a.displacement, b.displacement) << 4 + k;
      EXPECT_EQ(a.potential, b.potential) << 4 + k;
      EXPECT_EQ(a.angularVelocity, b.angularVelocity) << 4 + k;
      EXPECT_EQ(a.oceanFunction, b.oceanFunction) << 4 + k;
      EXPECT_EQ(a.tideCoefficients, b.tideCoefficients) << 4 + k;
      EXPECT_EQ(a.loadCoefficients, b.loadCoefficients) << 4 + k;
      ASSERT_EQ(a.internalState.has_value(), b.internalState.has_value()) << 4 + k;
      if (a.internalState) {
        EXPECT_EQ(a.internalState->m, b.internalState->m) << 4 + k;
      }
    }
  }
  {
    Viscoelastic3D lateral(body.rheology, body.lmax,
                           [](double, double theta, double phi) {
                             return 0.8 * std::sin(theta) * std::cos(theta) *
                                    std::cos(phi);
                           });
    auto checkpoints = std::map<std::size_t, Evolution3D::Checkpoint>{};
    auto states = std::map<std::size_t, Viscoelastic3D::State>{};
    auto options = Evolution3D::Options{.record_stride = 2};
    options.checkpoint_stride = 3;
    options.on_checkpoint = [&](std::size_t node,
                                const Evolution3D::Checkpoint& cp) {
      checkpoints.emplace(node, cp);
    };
    options.on_state = [&](std::size_t node, Viscoelastic3D::State&& state) {
      states.emplace(node, std::move(state));
    };
    const auto forcing3 = Evolution3D::Forcing{.tide = tide};
    Evolution3D whole(lateral, body.solver, world.state, options);
    const auto full = whole.RunMigrating(times, still, forcing3);
    ASSERT_TRUE(checkpoints.contains(3));
    // the states handed out: every second node of the whole run, and
    // the last
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      EXPECT_EQ(states.contains(n), n % 2 == 0 || n + 1 == times.size()) << n;
    }
    Evolution3D resumed(lateral, body.solver, world.state, {});
    const auto tail = resumed.ResumeMigrating(
        checkpoints.at(3), std::vector<double>(times.begin() + 3, times.end()),
        still, forcing3);
    ASSERT_EQ(tail.size(), times.size() - 4);
    // one process, the same transforms: bitwise
    for (auto k = std::size_t{0}; k < tail.size(); ++k) {
      const auto& a = full[4 + k];
      const auto& b = tail[k];
      EXPECT_EQ(a.relativeSeaLevelChange, b.relativeSeaLevelChange) << 4 + k;
      EXPECT_EQ(a.displacement, b.displacement) << 4 + k;
      EXPECT_EQ(a.angularVelocity, b.angularVelocity) << 4 + k;
      EXPECT_EQ(a.oceanFunction, b.oceanFunction) << 4 + k;
    }
  }
}
