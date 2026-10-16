/**
 * @file ExampleEvolution.cpp
 * @brief 1D viscoelastic GIA: the coupled sea-level evolution under a
 * melting ice cap (stage 3) — exponential-trapezoid stepping, the
 * generalised SLE closing each step, full rotational feedback.
 *
 * 20% of the West Antarctic ice (the real ICE-7G state, extracted from
 * pyslfp by the build) melts linearly over the first 5 kyr and the load
 * is then held for another 10 kyr: the elastic response appears
 * instantly, the viscous mantle keeps relaxing long after the melting
 * stops. A PREM-like deck with a 1e21 Pa s Maxwell mantle.
 *
 * Outputs (SI): a time series of the sea-level change at three sites and
 * the polar/rotational response, and maps at the end of melting and the
 * end of the run. Run from the build tree; plot with
 * plots/plot_evolution.py.
 */

#include <iostream>
#include <numbers>

#include "Common.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

int main() {
  constexpr int lmax = 128;
  constexpr double meltFraction = 0.2;
  constexpr double meltEnd = 5.0e3 * examples::kYear;  // s
  constexpr double tEnd = 15.0e3 * examples::kYear;
  constexpr int nodes = 31;

  const auto earth = examples::Earth::Make(lmax, 1e21);
  ViscoelasticOperator op(earth.rheology, lmax);
  SeaLevelSolver solver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
                        SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();

  // The present-day background state, rebuilt in the solver's units.
  const auto earthState = examples::LoadEarthState(grid);
  const double length = scales.Length();
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto& iceSi = earthState.iceThickness;
  auto state =
      solver.MakeState(toNd(iceSi), toNd(earthState.seaLevel));

  // The grounded melt load: a linear ramp to meltFraction over meltEnd,
  // then held, under a compact Amundsen-sector mask — tight enough that
  // the fingerprint's zero crossing sits offshore, so a nearby gauge
  // sees the classic near-field FALL.
  const GeographicPoint wais{-77.0, 245.0};
  const auto mask = grid.FromFunction([&](double theta, double phi) {
    return examples::Cap(theta, phi, wais.CoLatitude(), wais.Longitude(),
                         0.18);
  });
  const auto& C = state.oceanFunction;
  std::vector<double> unitLoad(grid.FieldSize());
  for (std::size_t i = 0; i < unitLoad.size(); ++i) {
    unitLoad[i] = -solver.IceDensity() * (1.0 - C[i]) * meltFraction *
                  mask[i] * iceSi[i] / length;
  }
  std::vector<double> times;
  for (int n = 0; n < nodes; ++n) {
    times.push_back(tEnd * n / (nodes - 1) / scales.Time());
  }
  const double rampEnd = meltEnd / scales.Time();
  auto iceLoad = [&](double t) {
    const double ramp = std::min(t / rampEnd, 1.0);
    std::vector<double> load(unitLoad.size());
    for (std::size_t i = 0; i < load.size(); ++i) load[i] = ramp * unitLoad[i];
    return load;
  };

  SeaLevelEvolution evolution(op, solver, state, {});
  const auto snapshots = evolution.Run(times, iceLoad);

  // Three sites, geographic degrees: near field (Amundsen Sea, inside
  // the fingerprint's fall region), mid field (Fremantle), far field
  // (North Atlantic).
  const GeographicPoint sites[] = {
      {-71.0, 240.0}, {-32.0, 115.7}, {58.0, 340.0}};
  auto sample = [&](std::span<const double> field,
                    const GeographicPoint& site) {
    // The nearest grid point serves a demo; no interpolation.
    const auto colats = grid.CoLatitudes();
    const auto lons = grid.Longitudes();
    const double theta = site.CoLatitude(), phi = site.Longitude();
    std::size_t bi = 0, bj = 0;
    for (std::size_t i = 1; i < colats.size(); ++i) {
      if (std::abs(colats[i] - theta) < std::abs(colats[bi] - theta)) bi = i;
    }
    for (std::size_t j = 1; j < lons.size(); ++j) {
      if (std::abs(lons[j] - phi) < std::abs(lons[bj] - phi)) bj = j;
    }
    return field[bi * lons.size() + bj];
  };

  examples::Writer out("evolution.txt");
  out.Comment(
      "1D viscoelastic GIA: 20% West Antarctic melt over 5 kyr, held to "
      "15 kyr; ICE-7G state (SI)");
  out.Grid(grid);
  const double omegaSi = 1.0 / scales.Time();
  std::vector<std::vector<double>> rows;
  for (std::size_t n = 0; n < snapshots.size(); ++n) {
    const auto& s = snapshots[n];
    rows.push_back({times[n] * scales.Time() / (1.0e3 * examples::kYear),
                    sample(s.seaLevelChange, sites[0]) * length,
                    sample(s.seaLevelChange, sites[1]) * length,
                    sample(s.seaLevelChange, sites[2]) * length,
                    s.angularVelocity[0] * omegaSi,
                    s.angularVelocity[1] * omegaSi,
                    s.angularVelocity[2] * omegaSi});
  }
  out.Series("history",
             {"t_kyr", "slc_near_m", "slc_mid_m", "slc_far_m", "omega_x",
              "omega_y", "omega_z"},
             rows);
  const std::size_t meltNode = 10;  // t = 5 kyr on this grid
  out.Series("sites", {"lat", "lon"},
             {{sites[0].latitude, sites[0].longitude},
              {sites[1].latitude, sites[1].longitude},
              {sites[2].latitude, sites[2].longitude}});
  out.Field("slc_end_of_melt_m", snapshots[meltNode].seaLevelChange, length);
  out.Field("slc_final_m", snapshots.back().seaLevelChange, length);
  out.Field("displacement_final_m", snapshots.back().displacement, length);
  out.Scalar("melt_end_kyr", 5.0);

  std::cout << "wrote evolution.txt (" << snapshots.size() << " nodes)\n";
  return 0;
}
