// The sea level engine: structural identities that need no external
// reference — conventions, conservation, the adjoint dot-product test, the
// nonlinear mass budget. The pyslfp comparisons live in TestFixtures.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/ShConventions.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kPi = std::numbers::pi;

LoveNumbers TestTable() {
  static auto table =
      LoveNumbers::FromFile(std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                            "pyslfp_love_numbers.txt")
          .value();
  return table;
}

/// A smooth cap, band-limited to spectral decay at modest lmax.
double Cap(double theta, double phi, double colat0, double lon0, double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

/// The L2 coefficient pairing matching the surface L2 product of real
/// fields in the reduced m >= 0 storage.
double Pair(std::span<const std::complex<double>> a,
            std::span<const std::complex<double>> b, int lmax) {
  auto sum = 0.0;
  for (auto l = 0; l <= lmax; ++l) {
    sum +=
        a[SurfaceGrid::Index(l, 0)].real() * b[SurfaceGrid::Index(l, 0)].real();
    for (auto m = 1; m <= l; ++m) {
      const auto i = SurfaceGrid::Index(l, m);
      sum += 2.0 * (a[i] * std::conj(b[i])).real();
    }
  }
  return sum;
}

}  // namespace

TEST(ShConventions, RoundTripAndParseval) {
  for (int m : {0, 1, 2, 5}) {
    const RealPair pair{0.7, m == 0 ? 0.0 : -1.3};
    const auto f = ComplexFromRealPair(pair, m);
    const auto back = RealPairFromComplex(f, m);
    EXPECT_NEAR(back.cos, pair.cos, 1e-15);
    EXPECT_NEAR(back.sin, pair.sin, 1e-15);
    // The pairing is preserved: a c + b d = 2 Re(f conj(g)) for m > 0.
    const RealPair other{-0.2, m == 0 ? 0.0 : 0.9};
    const auto g = ComplexFromRealPair(other, m);
    const auto realDot = pair.cos * other.cos + pair.sin * other.sin;
    const double complexDot =
        m == 0 ? (f * std::conj(g)).real() : 2.0 * (f * std::conj(g)).real();
    EXPECT_NEAR(realDot, complexDot, 1e-14);
  }
}

TEST(SurfaceGrid, TransformsAndIntegrals) {
  SurfaceGrid grid(16);
  // A unit constant integrates to 4 pi and expands to Y00 alone.
  auto one = grid.FromFunction([](double, double) { return 1.0; });
  EXPECT_NEAR(grid.Integrate(one), 4.0 * kPi, 1e-12);
  auto coeffs = grid.Expand(one);
  EXPECT_NEAR(coeffs[SurfaceGrid::Index(0, 0)].real(), std::sqrt(4.0 * kPi),
              1e-12);
  EXPECT_NEAR(std::abs(coeffs[SurfaceGrid::Index(2, 1)]), 0.0, 1e-13);
  // Round trip of a smooth field. The cap is not exactly band-limited
  // (its angular distance has an antipodal kink), so the comparison is at
  // its truncation tail, not machine precision.
  auto f = grid.FromFunction([](double theta, double phi) {
    return Cap(theta, phi, 0.4, 1.0, 0.7) - 0.5 * std::cos(theta);
  });
  auto g = grid.Synthesise(grid.Expand(f));
  for (auto i = std::size_t{0}; i < f.size(); i += 13) {
    EXPECT_NEAR(f[i], g[i], 2e-6);
  }
  // An exactly band-limited field round-trips to machine precision.
  auto zonal = grid.FromFunction([](double theta, double) {
    const auto c = std::cos(theta);
    return 1.0 + c + (1.5 * c * c - 0.5);
  });
  auto back = grid.Synthesise(grid.Expand(zonal));
  for (auto i = std::size_t{0}; i < zonal.size(); i += 13) {
    EXPECT_NEAR(zonal[i], back[i], 1e-13);
  }
  // Parseval: int f^2 dOmega = <f, f> in the reduced storage.
  auto f2 = std::vector<double>(f.size());
  for (auto i = std::size_t{0}; i < f.size(); ++i) f2[i] = f[i] * f[i];
  auto flm = grid.Expand(f);
  EXPECT_NEAR(grid.Integrate(f2), Pair(flm, flm, 16),
              1e-10 * Pair(flm, flm, 16));
}

// The sea level of a state from topography and ice, pyslfp's construction:
// ocean, land, grounded ice, and ice standing on the ocean floor.
TEST(SeaLevel, FromTopography) {
  const auto parameters = SeaLevelParameters{};
  const auto flotation = parameters.ice_density / parameters.water_density;
  const auto topography = std::vector<double>{-4000.0, 500.0, 2500.0, -300.0};
  const auto ice = std::vector<double>{0.0, 0.0, 2000.0, 400.0};
  const auto sl = SeaLevelFromTopography(topography, ice, parameters);
  EXPECT_DOUBLE_EQ(sl[0], 4000.0);
  EXPECT_DOUBLE_EQ(sl[1], -500.0);
  EXPECT_DOUBLE_EQ(sl[2], -500.0);
  EXPECT_DOUBLE_EQ(sl[3], 300.0 + flotation * 400.0);
  EXPECT_THROW(static_cast<void>(SeaLevelFromTopography(
                   topography, std::vector<double>{0.0}, parameters)),
               std::invalid_argument);
}

TEST(GRDSolver, RigidEarthEustaticLimit) {
  // With every Love number zero the sea level change is the uniform
  // eustatic value -int load dS / (rho_w A) over the whole sphere.
  auto table = TestTable();
  for (auto* v : {&table.h_u, &table.l_u, &table.k_u, &table.h_phi,
                  &table.l_phi, &table.k_phi, &table.h_v, &table.l_v,
                  &table.k_v, &table.h_t, &table.l_t, &table.k_t}) {
    std::ranges::fill(*v, 0.0);
  }
  table.h_c = table.k_c = table.m_u = table.m_phi = table.m_c = 0.0;

  Fingerprint fingerprint(16, table);
  fingerprint.SetAnalyticTestState();
  const auto& grid = fingerprint.Grid();
  auto load = grid.FromFunction([](double theta, double phi) {
    return 100.0 * Cap(theta, phi, 0.3, 0.3, 0.5);  // kg m^-2
  });
  auto result = fingerprint.Solve(load, RotationalFeedback::None);

  // By hand: the SI eustatic value on the state's ocean.
  const auto a = table.radius;
  std::vector<double> ocean(fingerprint.OceanFunction().begin(),
                            fingerprint.OceanFunction().end());
  const auto area = a * a * grid.Integrate(ocean);
  const auto mass = a * a * grid.Integrate(load);
  const auto eustatic = -mass / (1000.0 * area);
  for (auto i = std::size_t{0}; i < result.relativeSeaLevelChange.size(); i += 29) {
    EXPECT_NEAR(result.relativeSeaLevelChange[i], eustatic, 1e-9 * std::abs(eustatic));
  }
  EXPECT_NEAR(result.displacement[0], 0.0, 1e-15);
}

TEST(GRDSolver, ConservesMassAndConverges) {
  Fingerprint fingerprint(32, TestTable());
  fingerprint.SetAnalyticTestState();
  const auto& grid = fingerprint.Grid();

  // Melt a fraction of the polar cap: a physically shaped forcing.
  auto ice = fingerprint.IceThickness();
  auto iceChange = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < ice.size(); ++i)
    iceChange[i] = -0.1 * ice[i];
  auto load = fingerprint.DirectLoadFromIceThicknessChange(iceChange);
  auto result = fingerprint.SolveForIceChange(iceChange);
  EXPECT_TRUE(result.converged);
  EXPECT_LT(result.iterations, 40);

  // The ocean gains exactly the mass the load removed.
  auto oceanGain = std::vector<double>(load.size());
  for (auto i = std::size_t{0}; i < load.size(); ++i) {
    oceanGain[i] =
        1000.0 * fingerprint.OceanFunction()[i] * result.relativeSeaLevelChange[i];
  }
  const auto gained = grid.Integrate(oceanGain);
  const auto removed = -grid.Integrate(load);
  EXPECT_NEAR(gained, removed, 1e-8 * std::abs(removed));

  // Melting northern ice moves the pole measurably; melting at the pole
  // of a symmetric load would not.
  EXPECT_GT(std::hypot(result.angularVelocity[0], result.angularVelocity[1]),
            0.0);

  // Far-field sea level rises, near-field falls: the fingerprint shape.
  auto min = 1e300, max = -1e300;
  for (auto v : result.relativeSeaLevelChange) {
    min = std::min(min, v);
    max = std::max(max, v);
  }
  EXPECT_LT(min, 0.0);
  EXPECT_GT(max, 0.0);
}

TEST(GRDSolver, AdjointDotProductIdentity) {
  // The generalised solve against its adjoint: the defining identity
  //   <g, outputs(inputs)> = <adjoint(g), inputs>
  // holds to solver tolerance for every channel at once.
  const auto lmax = 24;
  auto table = TestTable();
  GRDSolver solver(SurfaceGrid(lmax), table, SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto a = solver.Radius();

  // A smooth state (non-dimensional metres via the solver's scales), and
  // the body without an ocean.
  const auto length = solver.SolverScales().Length();
  const auto withOcean = solver.MakeState(
      grid.FromFunction([&](double theta, double phi) {
        return 2000.0 * Cap(theta, phi, 0.1, 0.0, 0.3) / length;
      }),
      grid.FromFunction([&](double theta, double phi) {
        return (4000.0 - 7000.0 * Cap(theta, phi, 1.6, 2.0, 0.7)) / length;
      }));
  const auto withoutOcean = solver.NoOceanState();

  // Every dual at once, and the tangential potential's alone (so that its
  // terms are not lost beside the others).
  for (const auto tangentialAlone : {false, true})
  for (const auto* chosen : {&withOcean, &withoutOcean})
  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Axial,
                        RotationalFeedback::Full}) {
    const auto& state = *chosen;
    GRDSolver::GeneralisedForcing forcing;
    forcing.rotation = rotation;
    const double sigma =
        solver.SolverScales().Factor(Dimensions{.mass = 1, .length = -2});
    forcing.directLoad = grid.FromFunction([&](double theta, double phi) {
      return 150.0 * Cap(theta, phi, 0.4, 0.7, 0.5) / sigma;
    });
    forcing.displacementResponse =
        grid.Expand(grid.FromFunction([&](double theta, double phi) {
          return 0.5 * Cap(theta, phi, 2.0, 4.0, 0.6) / length;
        }));
    forcing.potentialResponse =
        grid.Expand(grid.FromFunction([&](double theta, double phi) {
          return -3.0 * Cap(theta, phi, 1.0, 5.5, 0.8) /
                 solver.SolverScales().Factor(PotentialDimensions);
        }));
    forcing.tangentialResponse =
        grid.Expand(grid.FromFunction([&](double theta, double phi) {
          return 0.2 * Cap(theta, phi, 0.8, 2.5, 0.7) / length;
        }));
    forcing.inertiaMomentChange = 1e-7;
    // a load outside the water budget, and a layer on the solid surface
    forcing.directLoadOutsideWaterBudget =
        grid.FromFunction([&](double theta, double phi) {
          return 90.0 * (Cap(theta, phi, 1.7, 3.0, 0.6) -
                         0.5 * Cap(theta, phi, 0.6, 5.0, 0.4)) / sigma;
        });
    forcing.solidSurfaceChange = grid.FromFunction([&](double theta, double phi) {
      return (0.3 * Cap(theta, phi, 1.8, 2.2, 0.8) + 0.05) / length;
    });
    forcing.tide = grid.Expand(grid.FromFunction([&](double theta, double phi) {
      return 2.0 * Cap(theta, phi, 2.6, 1.0, 0.9) /
             solver.SolverScales().Factor(PotentialDimensions);
    }));
    forcing.tide[SurfaceGrid::Index(0, 0)] = 0.0;

    GRDSolver::Options options{.rtol = 1e-13, .max_iterations = 400};
    const auto response = solver.SolveGeneralised(state, forcing, options);
    ASSERT_TRUE(response.converged);

    GRDSolver::AdjointForcing duals;
    duals.rotation = rotation;
    duals.relativeSeaLevelDual = grid.FromFunction([&](double theta, double phi) {
      return Cap(theta, phi, 0.9, 1.2, 0.5);
    });
    duals.displacementDual = grid.FromFunction([&](double theta, double phi) {
      return -0.7 * Cap(theta, phi, 2.4, 0.3, 0.6);
    });
    duals.potentialDual = grid.FromFunction([&](double theta, double phi) {
      return 0.4 * Cap(theta, phi, 1.4, 3.3, 0.7);
    });
    duals.angularVelocityDual = {0.3, -0.2, 0.5};
    duals.tangentialPotentialDual =
        grid.FromFunction([&](double theta, double phi) {
          return 0.6 * Cap(theta, phi, 1.9, 5.0, 0.6);
        });
    if (tangentialAlone) {
      duals.relativeSeaLevelDual.clear();
      duals.displacementDual.assign(grid.FieldSize(), 0.0);
      duals.potentialDual.assign(grid.FieldSize(), 0.0);
      duals.angularVelocityDual = {0.0, 0.0, 0.0};
    }
    const auto adjoint = solver.SolveAdjoint(state, duals, options);
    ASSERT_TRUE(adjoint.converged);

    auto surface = [&](std::span<const double> f, std::span<const double> g) {
      auto fg = std::vector<double>(f.size());
      for (auto i = std::size_t{0}; i < f.size(); ++i) fg[i] = f[i] * g[i];
      return a * a * grid.Integrate(fg);
    };
    ASSERT_GT(std::abs(surface(duals.tangentialPotentialDual,
                               response.tangentialPotential)),
              0.0);
    const double lhs =
        (tangentialAlone ? 0.0
                         : surface(duals.relativeSeaLevelDual,
                                   response.relativeSeaLevelChange)) +
        surface(duals.tangentialPotentialDual, response.tangentialPotential) +
        surface(duals.displacementDual, response.displacement) +
        surface(duals.potentialDual, response.potential) +
        duals.angularVelocityDual[0] * response.angularVelocity[0] +
        duals.angularVelocityDual[1] * response.angularVelocity[1] +
        duals.angularVelocityDual[2] * response.angularVelocity[2];
    const double rhs =
        surface(adjoint.directLoadDual, forcing.directLoad) +
        surface(adjoint.directLoadOutsideWaterBudgetDual,
                forcing.directLoadOutsideWaterBudget) +
        surface(adjoint.solidSurfaceChangeDual, forcing.solidSurfaceChange) +
        Pair(adjoint.displacementResponseDual, forcing.displacementResponse,
             lmax) +
        Pair(adjoint.potentialResponseDual, forcing.potentialResponse, lmax) +
        Pair(adjoint.tangentialResponseDual, forcing.tangentialResponse,
             lmax) +
        Pair(adjoint.tideDual, forcing.tide, lmax) +
        adjoint.inertiaMomentDual * forcing.inertiaMomentChange;
    EXPECT_NEAR(lhs, rhs, 1e-10 * std::abs(lhs))
        << "rotation mode " << static_cast<int>(rotation)
        << (chosen == &withOcean ? ", ocean" : ", no ocean")
        << (tangentialAlone ? ", the tangential dual alone" : "");
  }
}

TEST(GRDSolver, WithoutAnOceanIsTheSolidBodysResponse) {
  // A state with no ocean carries no water load: the solve is the solid
  // body's response to the forcing, rotation included. Its oracle is the
  // solve WITH an ocean, whose converged total load (direct plus the
  // ocean's) given to the body without one as a direct load must return
  // the same displacement, potential and rotation, and a sea level change
  // differing by the uniform shift that conserves the water.
  const auto lmax = 24;
  GRDSolver solver(SurfaceGrid(lmax), TestTable(), SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto length = solver.SolverScales().Length();
  const auto sigma =
      solver.SolverScales().Factor(Dimensions{.mass = 1, .length = -2});
  const auto withOcean = solver.MakeState(
      grid.FromFunction([&](double theta, double phi) {
        return 2000.0 * Cap(theta, phi, 0.1, 0.0, 0.3) / length;
      }),
      grid.FromFunction([&](double theta, double phi) {
        return (4000.0 - 7000.0 * Cap(theta, phi, 1.6, 2.0, 0.7)) / length;
      }));
  const auto dry = solver.NoOceanState();
  EXPECT_EQ(dry.oceanArea, 0.0);

  auto maxAbs = [](std::span<const double> v) {
    auto worst = 0.0;
    for (const auto x : v) worst = std::max(worst, std::abs(x));
    return worst;
  };
  GRDSolver::Options options{.rtol = 1e-13, .max_iterations = 400};
  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Axial,
                        RotationalFeedback::Full}) {
    GRDSolver::GeneralisedForcing forcing;
    forcing.rotation = rotation;
    forcing.directLoad = grid.FromFunction([&](double theta, double phi) {
      return 150.0 * Cap(theta, phi, 0.4, 0.7, 0.5) / sigma;
    });
    forcing.displacementResponse =
        grid.Expand(grid.FromFunction([&](double theta, double phi) {
          return 0.5 * Cap(theta, phi, 2.0, 4.0, 0.6) / length;
        }));
    forcing.inertiaMomentChange = 1e-7;
    const auto wet = solver.SolveGeneralised(withOcean, forcing, options);
    ASSERT_TRUE(wet.converged);

    auto total = forcing;
    for (auto i = std::size_t{0}; i < total.directLoad.size(); ++i) {
      total.directLoad[i] += solver.WaterDensity() *
                             withOcean.oceanFunction[i] * wet.relativeSeaLevelChange[i];
    }
    const auto solid = solver.SolveGeneralised(dry, total, options);
    EXPECT_TRUE(solid.converged);
    EXPECT_LE(solid.iterations, 1);

    const auto mode = static_cast<int>(rotation);
    auto worstU = 0.0, worstPhi = 0.0, lowest = 1e300, highest = -1e300;
    for (auto i = std::size_t{0}; i < wet.displacement.size(); ++i) {
      worstU = std::max(worstU,
                        std::abs(solid.displacement[i] - wet.displacement[i]));
      worstPhi =
          std::max(worstPhi, std::abs(solid.potential[i] - wet.potential[i]));
      const auto shift = wet.relativeSeaLevelChange[i] - solid.relativeSeaLevelChange[i];
      lowest = std::min(lowest, shift);
      highest = std::max(highest, shift);
    }
    EXPECT_LT(worstU, 1e-10 * maxAbs(wet.displacement)) << "rotation " << mode;
    EXPECT_LT(worstPhi, 1e-10 * maxAbs(wet.potential)) << "rotation " << mode;
    EXPECT_LT(highest - lowest, 1e-10 * maxAbs(wet.relativeSeaLevelChange))
        << "rotation " << mode;
    for (auto k = 0; k < 3; ++k) {
      EXPECT_NEAR(solid.angularVelocity[k], wet.angularVelocity[k],
                  1e-10 * maxAbs(wet.angularVelocity) + 1e-300)
          << "rotation " << mode;
    }
  }

  GRDSolver::NonlinearForcing moving;
  moving.iceThicknessChange.assign(grid.FieldSize(), 0.0);
  EXPECT_THROW((void)solver.SolveNonlinear(dry, moving), std::invalid_argument);
}

TEST(GRDSolver, LoadsOutsideTheWaterBudgetAndLayersOnTheSolidSurface) {
  // The three properties of a surface load. A load in the water budget
  // takes its mass from the ocean; one outside it moves no water, the
  // ocean's volume unchanged, and is otherwise the same load: the two
  // answers differ by the answer to the water moved alone. A layer on
  // the solid surface lowers the sea level by its thickness and
  // displaces its volume of water: a uniform layer changes nothing, a
  // layer on the ocean floor raises the sea elsewhere, and its sea
  // level is that of the same height given as a ready displacement,
  // which the displacement returned then holds and the layer's does not.
  const auto lmax = 24;
  auto table = TestTable();
  GRDSolver solver(SurfaceGrid(lmax), table, SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto length = solver.SolverScales().Length();
  const double sigma =
      solver.SolverScales().Factor(Dimensions{.mass = 1, .length = -2});
  const auto state = solver.MakeState(
      grid.FromFunction([&](double theta, double phi) {
        return 2000.0 * Cap(theta, phi, 0.1, 0.0, 0.3) / length;
      }),
      grid.FromFunction([&](double theta, double phi) {
        return (4000.0 - 7000.0 * Cap(theta, phi, 1.6, 2.0, 0.7)) / length;
      }));
  const auto& C = state.oceanFunction;
  auto oceanMean = [&](std::span<const double> f) {
    auto cf = std::vector<double>(f.size());
    for (auto i = std::size_t{0}; i < f.size(); ++i) cf[i] = C[i] * f[i];
    return grid.Integrate(cf);
  };
  auto maxAbs = [](std::span<const double> x) {
    auto out = 0.0;
    for (auto v : x) out = std::max(out, std::abs(v));
    return out;
  };
  auto worst = [](std::span<const double> x, std::span<const double> y) {
    auto out = 0.0;
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      out = std::max(out, std::abs(x[i] - y[i]));
    }
    return out;
  };
  const GRDSolver::Options options{.rtol = 1e-13, .max_iterations = 400};
  const auto load = grid.FromFunction([&](double theta, double phi) {
    return 150.0 * Cap(theta, phi, 0.4, 0.7, 0.5) / sigma;
  });
  const auto a2 = solver.Radius() * solver.Radius();

  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Full}) {
    GRDSolver::GeneralisedForcing inside, outside;
    inside.rotation = outside.rotation = rotation;
    inside.directLoad = load;
    outside.directLoadOutsideWaterBudget = load;
    const auto in = solver.SolveGeneralised(state, inside, options);
    const auto out = solver.SolveGeneralised(state, outside, options);
    ASSERT_TRUE(in.converged && out.converged);
    const auto scale = maxAbs(in.relativeSeaLevelChange);
    // the water moved: the load's mass out of the ocean, or none
    const auto water = solver.WaterDensity() * a2;
    EXPECT_NEAR(water * oceanMean(in.relativeSeaLevelChange),
                -a2 * grid.Integrate(load), 1e-10 * a2 * grid.Integrate(load));
    EXPECT_LT(std::abs(oceanMean(out.relativeSeaLevelChange)),
              1e-11 * scale * grid.Integrate(C));
    EXPECT_GT(worst(in.relativeSeaLevelChange, out.relativeSeaLevelChange),
              1e-3 * scale);
    // and the two together are the load counted twice, once each way
    GRDSolver::GeneralisedForcing both;
    both.rotation = rotation;
    both.directLoad = load;
    both.directLoadOutsideWaterBudget = load;
    const auto sum = solver.SolveGeneralised(state, both, options);
    auto added = in.relativeSeaLevelChange;
    for (auto i = std::size_t{0}; i < added.size(); ++i) {
      added[i] += out.relativeSeaLevelChange[i];
    }
    EXPECT_LT(worst(sum.relativeSeaLevelChange, added), 1e-9 * scale);

    // a uniform layer: nothing
    GRDSolver::GeneralisedForcing uniform;
    uniform.rotation = rotation;
    uniform.solidSurfaceChange.assign(grid.FieldSize(), 3.0 / length);
    const auto flat = solver.SolveGeneralised(state, uniform, options);
    EXPECT_LT(maxAbs(flat.relativeSeaLevelChange), 1e-10 * 3.0 / length);
    EXPECT_LT(maxAbs(flat.displacement), 1e-10 * 3.0 / length);

    // a layer on part of the ocean floor: the sea falls there by nearly
    // its thickness and rises elsewhere, no water lost
    const auto layer = grid.FromFunction([&](double theta, double phi) {
      return 2.0 * Cap(theta, phi, 2.6, 4.0, 0.4) / length;
    });
    GRDSolver::GeneralisedForcing layered, ready;
    layered.rotation = ready.rotation = rotation;
    layered.solidSurfaceChange = layer;
    ready.displacementResponse = grid.Expand(layer);
    const auto onFloor = solver.SolveGeneralised(state, layered, options);
    const auto asReady = solver.SolveGeneralised(state, ready, options);
    ASSERT_TRUE(onFloor.converged && asReady.converged);
    EXPECT_LT(std::abs(oceanMean(onFloor.relativeSeaLevelChange)),
              1e-11 * maxAbs(layer) * grid.Integrate(C));
    auto least = 0.0, greatest = 0.0;
    for (auto v : onFloor.relativeSeaLevelChange) {
      least = std::min(least, v);
      greatest = std::max(greatest, v);
    }
    EXPECT_LT(least, -0.9 * maxAbs(layer));
    EXPECT_GT(greatest, 0.0);
    if (rotation == RotationalFeedback::None) {
      // the same sea level as the height given as a ready displacement,
      // which is then in the displacement returned
      EXPECT_LT(worst(onFloor.relativeSeaLevelChange,
                      asReady.relativeSeaLevelChange),
                1e-10 * maxAbs(layer));
      auto moved = onFloor.displacement;
      const auto band = grid.Synthesise(grid.Expand(layer));
      for (auto i = std::size_t{0}; i < moved.size(); ++i) moved[i] += band[i];
      EXPECT_LT(worst(moved, asReady.displacement), 1e-10 * maxAbs(layer));
    }
  }
}

TEST(GRDSolver, LandHeldStaysLandThroughTheMigratingSolve) {
  // A state told to hold points as land has no ocean there whatever the
  // flotation says, its ocean area the less for it, and the solve with
  // migrating shorelines, which finds the ocean function again by
  // flotation, leaves them land and keeps the mark.
  auto table = TestTable();
  Fingerprint fingerprint(32, table);
  fingerprint.SetAnalyticTestState(5.0);
  const auto& grid = fingerprint.Grid();
  const auto& solver = fingerprint.Solver();
  const auto length = solver.SolverScales().Length();
  auto scaled = [&](std::vector<double> v) {
    for (auto& x : v) x /= length;
    return v;
  };
  const auto free = solver.MakeState(scaled(fingerprint.IceThickness()),
                                     scaled(fingerprint.SeaLevel()));
  // hold a patch of the open ocean
  auto land = std::vector<char>(grid.FieldSize(), 0);
  const auto nphi = grid.Longitudes().size();
  auto held = std::size_t{0};
  for (auto it = std::size_t{0}; it < grid.CoLatitudes().size(); ++it) {
    for (auto ip = std::size_t{0}; ip < nphi; ++ip) {
      const auto i = it * nphi + ip;
      const auto theta = grid.CoLatitudes()[it];
      if (free.oceanFunction[i] == 1.0 && theta > 1.2 && theta < 1.9 &&
          grid.Longitudes()[ip] < 1.0) {
        land[i] = 1;
        ++held;
      }
    }
  }
  ASSERT_GT(held, std::size_t{10});
  const auto state = solver.MakeState(scaled(fingerprint.IceThickness()),
                                      scaled(fingerprint.SeaLevel()), land);
  EXPECT_LT(state.oceanArea, free.oceanArea);
  for (auto i = std::size_t{0}; i < land.size(); ++i) {
    EXPECT_EQ(state.oceanFunction[i], land[i] ? 0.0 : free.oceanFunction[i]);
  }

  GRDSolver::NonlinearForcing forcing;
  forcing.iceThicknessChange = state.iceThickness;
  for (auto& v : forcing.iceThicknessChange) v *= -0.05;
  GRDSolver::Options options;
  options.rtol = 1e-10;
  options.max_iterations = 100;
  auto moved = state;
  const auto result = solver.SolveNonlinear(moved, forcing, options);
  ASSERT_TRUE(result.response.converged);
  ASSERT_EQ(moved.forcedLand.size(), land.size());
  for (auto i = std::size_t{0}; i < land.size(); ++i) {
    EXPECT_EQ(moved.forcedLand[i], land[i]);
    if (land[i]) {
      EXPECT_EQ(moved.oceanFunction[i], 0.0);
    }
  }
}

TEST(GRDSolver, NonlinearSolveTakesEveryLoad) {
  // The nonlinear solve under the loads of the linear one. Where no
  // shoreline moves the two are the same equation, and their answers
  // agree to the tolerance of the solves: ice, a load in the water
  // budget, one outside it and a layer on the solid surface, together.
  // A uniform layer changes nothing. A thick layer on the ocean floor
  // makes land of it: the shorelines are flotation's on the final state,
  // whose sea level is the first plus the change,
  // the water and grounded ice are conserved (less the mass of a load
  // inside the budget, where there is one), and the load returned is
  // the load the final state carries.
  const auto lmax = 32;
  auto table = TestTable();
  GRDSolver solver(SurfaceGrid(lmax), table, SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto length = solver.SolverScales().Length();
  const double sigma =
      solver.SolverScales().Factor(Dimensions{.mass = 1, .length = -2});
  const double density =
      solver.SolverScales().Factor(Dimensions{.mass = 1, .length = -3});
  // a continent with grounded ice at its centre, in a deep ocean
  const auto state = solver.MakeState(
      grid.FromFunction([&](double theta, double phi) {
        return 1500.0 * Cap(theta, phi, 1.6, 2.0, 0.3) / length;
      }),
      grid.FromFunction([&](double theta, double phi) {
        return (4000.0 - 7000.0 * Cap(theta, phi, 1.6, 2.0, 0.7)) / length;
      }));
  auto maxAbs = [](std::span<const double> x) {
    auto out = 0.0;
    for (auto v : x) out = std::max(out, std::abs(v));
    return out;
  };
  auto worst = [](std::span<const double> x, std::span<const double> y) {
    auto out = 0.0;
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      out = std::max(out, std::abs(x[i] - y[i]));
    }
    return out;
  };
  const auto a2 = solver.Radius() * solver.Radius();
  auto budget = [&](const GRDSolver::State& s) {
    auto total = std::vector<double>(s.seaLevel.size());
    for (auto i = std::size_t{0}; i < total.size(); ++i) {
      total[i] =
          solver.WaterDensity() * s.oceanFunction[i] * s.seaLevel[i] +
          solver.IceDensity() * (1.0 - s.oceanFunction[i]) * s.iceThickness[i];
    }
    return a2 * grid.Integrate(total);
  };
  const GRDSolver::Options options{.rtol = 1e-13, .max_iterations = 400};

  // -- no shoreline moves: the linear solve's answer
  auto iceChange = state.iceThickness;
  for (auto& v : iceChange) v *= -1e-3;
  const auto inside = grid.FromFunction([&](double theta, double phi) {
    return 150.0 * Cap(theta, phi, 0.4, 0.7, 0.5) / sigma;
  });
  const auto outside = grid.FromFunction([&](double theta, double phi) {
    return -90.0 * Cap(theta, phi, 2.3, 4.0, 0.4) / sigma;
  });
  const auto layer = grid.FromFunction([&](double theta, double phi) {
    return 0.2 * Cap(theta, phi, 0.9, 5.0, 0.4) / length;
  });
  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Full}) {
    GRDSolver::NonlinearForcing moving;
    moving.iceThicknessChange = iceChange;
    moving.directLoad = inside;
    moving.directLoadOutsideWaterBudget = outside;
    moving.solidSurfaceChange = layer;
    moving.rotation = rotation;
    const auto got = solver.SolveNonlinear(state, moving, options);
    ASSERT_TRUE(got.response.converged)
        << got.response.error << " after " << got.response.iterations;
    ASSERT_EQ(got.finalState.oceanFunction, state.oceanFunction);

    GRDSolver::GeneralisedForcing fixed;
    fixed.directLoad =
        solver.DirectLoadFromIceThicknessChange(state, iceChange);
    for (auto i = std::size_t{0}; i < inside.size(); ++i) {
      fixed.directLoad[i] += inside[i];
    }
    fixed.directLoadOutsideWaterBudget = outside;
    fixed.solidSurfaceChange = layer;
    fixed.rotation = rotation;
    const auto want = solver.SolveGeneralised(state, fixed, options);
    ASSERT_TRUE(want.converged);
    const auto mode = static_cast<int>(rotation);
    EXPECT_LT(worst(got.response.relativeSeaLevelChange,
                    want.relativeSeaLevelChange),
              1e-9 * maxAbs(want.relativeSeaLevelChange))
        << "rotation " << mode;
    EXPECT_LT(worst(got.response.displacement, want.displacement),
              1e-9 * maxAbs(want.displacement))
        << "rotation " << mode;
    EXPECT_LT(worst(got.response.potential, want.potential),
              1e-9 * maxAbs(want.potential))
        << "rotation " << mode;
    EXPECT_LT(worst(got.response.tangentialPotential, want.tangentialPotential),
              1e-9 * maxAbs(want.tangentialPotential))
        << "rotation " << mode;
    EXPECT_NEAR(got.response.uniformShift, want.uniformShift,
                1e-9 * maxAbs(want.relativeSeaLevelChange))
        << "rotation " << mode;
    for (auto k = 0; k < 3; ++k) {
      EXPECT_NEAR(got.response.angularVelocity[k], want.angularVelocity[k],
                  1e-9 * maxAbs(want.angularVelocity) + 1e-300)
          << "rotation " << mode;
    }
  }

  // -- a uniform layer displaces its own water and changes nothing
  {
    GRDSolver::NonlinearForcing moving;
    moving.solidSurfaceChange.assign(grid.FieldSize(), 3.0 / length);
    const auto got = solver.SolveNonlinear(state, moving, options);
    ASSERT_TRUE(got.response.converged);
    EXPECT_EQ(got.finalState.oceanFunction, state.oceanFunction);
    EXPECT_LT(maxAbs(got.response.relativeSeaLevelChange), 1e-12 * 3.0 / length);
    EXPECT_NEAR(got.response.uniformShift, 3.0 / length, 1e-12 * 3.0 / length);
  }

  // -- a layer that fills the sea where it lies
  const auto fill = grid.FromFunction([&](double theta, double phi) {
    return 6000.0 * Cap(theta, phi, 0.4, 0.7, 0.2) / length;
  });
  auto fillMass = fill;
  for (auto& v : fillMass) v *= 2300.0 / density;
  for (auto inBudget : {false, true}) {
    GRDSolver::NonlinearForcing moving;
    moving.solidSurfaceChange = fill;
    (inBudget ? moving.directLoad : moving.directLoadOutsideWaterBudget) =
        fillMass;
    const GRDSolver::Options loose{.rtol = 1e-11, .max_iterations = 400};
    const auto got = solver.SolveNonlinear(state, moving, loose);
    ASSERT_TRUE(got.response.converged) << got.response.error;
    const auto& final = got.finalState;
    EXPECT_LT(final.oceanArea, 0.999 * state.oceanArea);
    auto made = std::size_t{0};
    for (auto i = std::size_t{0}; i < fill.size(); ++i) {
      EXPECT_NEAR(final.seaLevel[i],
                  state.seaLevel[i] + got.response.relativeSeaLevelChange[i],
                  1e-12 * maxAbs(state.seaLevel));
      // land made by the layer rising above the sea floor (the floor
      // sinks under it, so less than the layer alone would make)
      if (state.oceanFunction[i] == 1.0 && final.oceanFunction[i] == 0.0 &&
          fill[i] > state.seaLevel[i]) {
        ++made;
      }
    }
    EXPECT_GT(made, std::size_t{10});
    // and the shorelines are flotation's on the final state
    EXPECT_EQ(
        solver.MakeState(final.iceThickness, final.seaLevel).oceanFunction,
        final.oceanFunction);
    const auto taken = inBudget ? a2 * grid.Integrate(fillMass) : 0.0;
    EXPECT_NEAR(budget(final), budget(state) - taken,
                1e-9 * std::abs(budget(state)))
        << "in budget " << inBudget;
    // the load returned is the final state's own
    auto carried = std::vector<double>(fill.size());
    for (auto i = std::size_t{0}; i < fill.size(); ++i) {
      carried[i] =
          fillMass[i] +
          solver.IceDensity() *
              ((1.0 - final.oceanFunction[i]) * final.iceThickness[i] -
               (1.0 - state.oceanFunction[i]) * state.iceThickness[i]) +
          solver.WaterDensity() * (final.oceanFunction[i] * final.seaLevel[i] -
                                   state.oceanFunction[i] * state.seaLevel[i]);
    }
    const auto want = grid.Expand(carried);
    auto off = 0.0, top = 0.0;
    for (auto i = std::size_t{0}; i < want.size(); ++i) {
      off = std::max(off, std::abs(want[i] - got.loadCoefficients[i]));
      top = std::max(top, std::abs(want[i]));
    }
    EXPECT_LT(off, 1e-8 * top) << "in budget " << inBudget;
  }
}

TEST(GRDSolver, NonlinearMassBudgetAndLinearLimit) {
  auto table = TestTable();
  Fingerprint fingerprint(32, table);
  fingerprint.SetAnalyticTestState(5.0);  // some mid-latitude ice to move
  const auto& grid = fingerprint.Grid();

  auto ice = fingerprint.IceThickness();
  auto iceChange = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < ice.size(); ++i)
    iceChange[i] = -0.02 * ice[i];

  const auto nonlinear =
      fingerprint.SolveNonlinear(iceChange, RotationalFeedback::Full);
  EXPECT_TRUE(nonlinear.converged);
  const auto linear =
      fingerprint.SolveForIceChange(iceChange, RotationalFeedback::Full);

  // A small perturbation: migration is second order, so the two solutions
  // agree to a fraction of a percent while not being identical.
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < linear.relativeSeaLevelChange.size(); ++i) {
    scale = std::max(scale, std::abs(linear.relativeSeaLevelChange[i]));
    diff = std::max(
        diff, std::abs(linear.relativeSeaLevelChange[i] - nonlinear.relativeSeaLevelChange[i]));
  }
  EXPECT_LT(diff, 5e-2 * scale);
  EXPECT_GT(diff, 0.0);

  // The explicit mass budget closes: water + grounded ice is conserved
  // through the migrating-shoreline solve. Checked at the solver level,
  // where the final state is visible.
  const auto& solver = fingerprint.Solver();
  const auto length = solver.SolverScales().Length();
  auto state = solver.MakeState(
      [&] {
        auto v = fingerprint.IceThickness();
        for (auto& x : v) x /= length;
        return v;
      }(),
      [&] {
        auto v = fingerprint.SeaLevel();
        for (auto& x : v) x /= length;
        return v;
      }());
  GRDSolver::NonlinearForcing forcing;
  forcing.iceThicknessChange.resize(iceChange.size());
  for (auto i = std::size_t{0}; i < iceChange.size(); ++i) {
    forcing.iceThicknessChange[i] = iceChange[i] / length;
  }
  GRDSolver::Options options;
  options.rtol = 1e-11;
  options.max_iterations = 100;
  const auto result = solver.SolveNonlinear(state, forcing, options);
  ASSERT_TRUE(result.response.converged);

  auto budget = [&](const GRDSolver::State& s) {
    auto total = std::vector<double>(s.seaLevel.size());
    for (auto i = std::size_t{0}; i < total.size(); ++i) {
      total[i] =
          solver.WaterDensity() * s.oceanFunction[i] * s.seaLevel[i] +
          solver.IceDensity() * (1.0 - s.oceanFunction[i]) * s.iceThickness[i];
    }
    const auto a = solver.Radius();
    return a * a * grid.Integrate(total);
  };
  // The forcing removed ice mass rho_i * (melted thickness over land at
  // the FINAL grounding); the budget identity the solver enforces is
  // target_water = initial_water - grounded_ice_change, i.e. the total of
  // water and grounded ice is unchanged by redistribution.
  EXPECT_NEAR(budget(result.finalState), budget(state),
              1e-9 * std::abs(budget(state)));
}
