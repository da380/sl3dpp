/**
 * @file ExampleModelViscosity.cpp
 * @brief A 3D viscosity read from a planetmodel model file — the route a
 * converted tomography model takes — under a whole deglaciation read from
 * an ice history, against its radial twin.
 *
 * io::ModelViscosity splits the file's viscosity as
 * eta(x) = eta_bar(r) exp(delta(x)) about its lateral geometric mean:
 * eta_bar is the Maxwell rheology's radial profile and delta the lateral
 * perturbation Viscoelastic3D applies pseudospectrally. The file here is
 * the test suite's (viscosity_log.nc, copied beside the examples): on this
 * same PREM-like body, 1e21 Pa s with a degree-2 sectoral perturbation
 * delta = 0.4 (r/a)^2 sin^2(theta) cos(2 phi) through the mantle. A file
 * converted from a tomography model by planetmodel is read the same way.
 *
 * The load is the state file's own ice history from 21 ka to the present
 * (the synthetic Earth's glacial cycle, or ICE-7G with
 * `--state ice7g.nc`), relative to 21 ka, with the background state taken
 * at 21 ka. The radial twin uses the file's reference profile alone, so
 * the difference is what the lateral structure does.
 *
 * Outputs (SI): present-day sea-level change for both runs and their
 * difference, delta at 1000 km depth, the 21 ka ice, and the sea-level
 * histories at two sites near the former ice sheets.
 *
 *   ExampleModelViscosity [--lmax N] [--state FILE]
 *
 * Run from the build tree; plot with python/plot_model_viscosity.py.
 */

#include <cmath>
#include <iostream>

#include "Common.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/FieldHistory.hpp"
#include "sl3dpp/io/ModelViscosity.hpp"

using namespace sl3dpp;

int main(int argc, char** argv) {
  const auto options = examples::ParseOptions(argc, argv, 24);
  const auto lmax = options.lmax;
  constexpr auto startKyr = 21.0;  // the window, kyr before present
  constexpr auto nodes = 22;       // one per kyr
  const auto kyr = 1.0e3 * examples::kYear;

  // The model file's viscosity on the deck's body: its reference profile
  // as the Maxwell rheology, its lateral perturbation as delta.
  const auto earth = examples::Earth::Make(lmax, 1e21);
  const auto viscosity =
      io::ModelViscosity::Read("viscosity_log.nc", earth.model);
  if (!viscosity) {
    std::cerr << viscosity.error().message << "\n";
    return 1;
  }
  const auto rheology =
      Rheology::Maxwell(earth.material, viscosity->Viscosity()).value();
  Viscoelastic3D lateral(rheology, lmax, viscosity->Delta());
  Viscoelastic3D radial(rheology, lmax,
                        [](double, double, double) { return 0.0; });

  SeaLevelSolver solver(SurfaceGrid(lmax), lateral.ElasticLove(),
                        SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();

  // The background at the window's start, and the ice history over it.
  const auto start = examples::LoadEarthState(grid, options.state, startKyr);
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto state =
      solver.MakeState(toNd(start.iceThickness), toNd(start.seaLevel));
  const auto history = io::FieldHistory::Read(options.state, grid).value();

  // The grounded load relative to 21 ka, a callable of solver time.
  const auto& C = state.oceanFunction;
  const auto startSi = -startKyr * kyr;
  auto iceLoad = [&](double t) {
    const auto ice =
        history.Values("ice_thickness", startSi + t * scales.Time());
    auto load = std::vector<double>(grid.FieldSize());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - C[i]) *
                (ice[i] - start.iceThickness[i]) / length;
    }
    return load;
  };
  auto times = std::vector<double>{};
  for (auto n = 0; n < nodes; ++n) {
    times.push_back(startKyr * kyr * n / (nodes - 1) / scales.Time());
  }

  SeaLevelEvolution3D run3(lateral, solver, state, {});
  const auto with = run3.Run(times, iceLoad);
  SeaLevelEvolution3D run1(radial, solver, state, {});
  const auto without = run1.Run(times, iceLoad);
  for (const auto& snapshot : with) {
    if (!snapshot.converged) {
      std::cerr << "the coupled fixed point did not converge at t = "
                << snapshot.t << "\n";
      return 1;
    }
  }

  examples::Writer out("model_viscosity.txt");
  out.Comment("state " + options.state.string());
  out.Comment(
      "3D viscosity from viscosity_log.nc against its radial "
      "reference, ice history of " +
      options.state.string() + " from 21 ka (SI)");
  out.Grid(grid);
  out.Field("slc_lateral_m", with.back().seaLevelChange, length);
  out.Field("slc_radial_m", without.back().seaLevelChange, length);
  auto difference = std::vector<double>(grid.FieldSize());
  for (auto i = std::size_t{0}; i < difference.size(); ++i) {
    difference[i] =
        with.back().seaLevelChange[i] - without.back().seaLevelChange[i];
  }
  out.Field("slc_difference_m", difference, length);
  out.Field("ice_21ka_m", start.iceThickness);
  // delta at 1000 km depth: the functor takes the model's radius.
  const auto rDeep = 1.0 - 1.0e6 / earth.model.ModelScales().Length();
  const auto delta = viscosity->Delta();
  out.Field("delta_1000km", grid.FromFunction([&](double theta, double phi) {
    return delta(rDeep, theta, phi);
  }));

  // The cap-mean sea level at two sites near the former ice sheets.
  const auto a = solver.Radius();
  const GeographicPoint sites[] = {{60.0, 275.0},  // Hudson Bay
                                   {60.0, 10.0}};  // southern Norway
  auto rows = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < with.size(); ++n) {
    auto row = std::vector<double>{startKyr - startKyr * n / (nodes - 1)};
    for (const auto& site : sites) {
      const auto dual = CapMeanDual(grid, a, site, 0.05);
      row.push_back(length *
                    PairedValue(grid, a, dual, with[n].seaLevelChange));
      row.push_back(length *
                    PairedValue(grid, a, dual, without[n].seaLevelChange));
    }
    row.push_back(static_cast<double>(with[n].coupledIterations));
    rows.push_back(row);
  }
  out.Series("sites",
             {"t_kyr_bp", "hudson_lateral_m", "hudson_radial_m",
              "norway_lateral_m", "norway_radial_m", "coupled_iterations"},
             rows);
  for (auto k = 0; k < 2; ++k) {
    out.Scalar(std::format("site{}_lat", k), sites[k].latitude);
    out.Scalar(std::format("site{}_lon", k), sites[k].longitude);
  }

  const auto& last = rows.back();
  std::cout << "wrote model_viscosity.txt (present-day sea level at Hudson "
               "Bay: "
            << last[1] << " m lateral, " << last[2] << " m radial)\n";
  return 0;
}
