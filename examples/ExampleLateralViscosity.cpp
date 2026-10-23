/**
 * @file ExampleLateralViscosity.cpp
 * @brief 3D viscosity: the coupled evolution with a laterally
 * varying mantle, against its radially averaged twin.
 *
 * The viscosity is eta(x) = eta_bar(r) exp(delta(x)) with a degree-2-like
 * log perturbation of one decade peak to peak; the within-step lateral
 * iteration (design.md §6) converges each implicit
 * step under the same Anderson-mixed fixed point as the ocean load and
 * rotation. The zero-lateral twin isolates what the lateral structure
 * does to the predictions.
 *
 * The load is a tenth of the Antarctic ice of the present-day state,
 * melted at once and held.
 *
 * Outputs (SI): final sea-level maps for both runs and their difference,
 * and the per-step iteration counts of the lateral fixed point.
 *
 *   ExampleLateralViscosity [--lmax N] [--state FILE]
 *
 * Run from the build tree; plot with python/plot_lateral.py.
 */

#include <cmath>
#include <iostream>
#include <numbers>

#include "Common.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;

int main(int argc, char** argv) {
  const auto options = examples::ParseOptions(argc, argv, 16);
  const auto lmax = options.lmax;
  constexpr auto meltFraction = 0.1;
  constexpr auto tEnd = 10.0e3 * examples::kYear;
  constexpr auto nodes = 11;

  const auto earth = examples::Earth::Make(lmax, 1e21);
  Viscoelastic3D lateral(earth.rheology, lmax,
                         [](double, double theta, double phi) {
                           // one decade peak to peak
                           return 1.15 * std::sin(theta) * std::cos(theta) *
                                  std::cos(phi);
                         });
  Viscoelastic3D flat(earth.rheology, lmax,
                      [](double, double, double) { return 0.0; });

  SeaLevelSolver solver(SurfaceGrid(lmax), lateral.ElasticLove(),
                        SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();

  const auto earthState = examples::LoadEarthState(grid, options.state);
  const auto length = scales.Length();
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto& iceSi = earthState.iceThickness;
  auto state = solver.MakeState(toNd(iceSi), toNd(earthState.seaLevel));

  // A held melt load switched on at t = 0: a tenth of the Antarctic ice.
  const auto& C = state.oceanFunction;
  auto load = std::vector<double>(grid.FieldSize());
  for (auto i = std::size_t{0}; i < load.size(); ++i) {
    const auto south = grid.CoLatitudes()[i / grid.NumberOfLongitudes()] >
                       std::numbers::pi / 2;
    load[i] = south ? -solver.IceDensity() * (1.0 - C[i]) * meltFraction *
                          iceSi[i] / length
                    : 0.0;
  }
  auto iceLoad = [&](double) { return load; };
  auto times = std::vector<double>{};
  for (auto n = 0; n < nodes; ++n) {
    times.push_back(tEnd * n / (nodes - 1) / scales.Time());
  }

  SeaLevelEvolution3D run3(lateral, solver, state, {});
  const auto with = run3.Run(times, iceLoad);
  SeaLevelEvolution3D run1(flat, solver, state, {});
  const auto without = run1.Run(times, iceLoad);

  examples::Writer out("lateral.txt");
  out.Comment("state " + options.state.string());
  out.Comment(
      "3D lateral viscosity (one decade peak to peak) vs the radial "
      "reference, held northern melt, 10 kyr (SI)");
  out.Grid(grid);
  out.Field("slc_lateral_m", with.back().seaLevelChange, length);
  out.Field("slc_radial_m", without.back().seaLevelChange, length);
  auto difference = std::vector<double>(grid.FieldSize());
  for (auto i = std::size_t{0}; i < difference.size(); ++i) {
    difference[i] =
        with.back().seaLevelChange[i] - without.back().seaLevelChange[i];
  }
  out.Field("slc_difference_m", difference, length);
  // The lateral log-viscosity pattern at the surface, for the plot.
  out.Field("delta_log_viscosity",
            grid.FromFunction([](double theta, double phi) {
              return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
            }));
  auto rows = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < with.size(); ++n) {
    rows.push_back({times[n] * scales.Time() / (1.0e3 * examples::kYear),
                    static_cast<double>(with[n].coupledIterations),
                    with[n].coupledResidual});
  }
  out.Series("iterations", {"t_kyr", "coupled_iterations", "residual"}, rows);

  std::cout << "wrote lateral.txt\n";
  return 0;
}
