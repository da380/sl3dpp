/**
 * @file ExampleViscosityKernels3D.cpp
 * @brief 3D sensitivity kernels: what two observations at one site know
 * about the mantle's viscosity, point by point, by the continuous adjoint
 * of the coupled 3D evolution.
 *
 * The forward problem is ExampleModelViscosity's: the 3D viscosity of
 * viscosity_log.nc under the state file's deglaciation from 21 ka,
 * recorded at every node for the reverse sweeps. Two observations at a
 * site near the former Laurentide centre:
 *
 * - the cap-mean sea-level change from 21 ka to the present (m);
 * - the decay time of the site's sea-level series over the last 7 kyr
 *   (kyr): the relaxation time of a fitted a e^(-t / tau) + b, a datum
 *   whose kernel comes from the fit's exact sensitivity to the series
 *   (Measurements.hpp).
 *
 * Each sweep gives dJ / d delta(x), the sensitivity to the log-viscosity
 * perturbation at every memory node and grid point. Both kernels are
 * written as one planetmodel model file per unit volume in SI
 * (io::WriteKernels: viscosity_kernels.nc, for the Python to read, draw
 * and integrate), and as depth slices and radial profiles for the plot.
 *
 * Outputs: viscosity_kernels.nc and kernels_3d.txt (slices on the
 * coupling's grid, per km^3; the profiles per km of radius).
 *
 *   ExampleViscosityKernels3D [--lmax N] [--state FILE]
 *
 * Run from the build tree; plot with python/plot_kernels_3d.py.
 */

#include <array>
#include <cmath>
#include <format>
#include <iostream>
#include <map>
#include <span>
#include <string>

#include "Common.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/FieldHistory.hpp"
#include "sl3dpp/io/KernelWriter.hpp"
#include "sl3dpp/io/ModelViscosity.hpp"

using namespace sl3dpp;

int main(int argc, char** argv) {
  const auto options = examples::ParseOptions(argc, argv, 24);
  const auto lmax = options.lmax;
  constexpr auto startKyr = 21.0;  // the window, kyr before present
  constexpr auto nodes = 22;       // one per kyr
  constexpr auto decayFromKyr = 7.0;
  const auto kyr = 1.0e3 * examples::kYear;

  examples::Say(
      std::format("ExampleViscosityKernels3D: 3D kernels at lmax {}, state {}",
                  lmax, options.state.string()));
  // The forward problem, as ExampleModelViscosity sets it up.
  examples::Say("reading viscosity_log.nc; building the 3D operator");
  const auto earth = examples::Earth::Make(lmax, 1e21);
  const auto viscosity =
      io::ModelViscosity::Read("viscosity_log.nc", earth.model);
  if (!viscosity) {
    std::cerr << viscosity.error().message << "\n";
    return 1;
  }
  const auto rheology =
      Rheology::Maxwell(earth.material, viscosity->Viscosity()).value();
  Viscoelastic3D op(rheology, lmax, viscosity->Delta());
  SeaLevelSolver solver(SurfaceGrid(lmax), op.ElasticLove(),
                        SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();

  const auto start = examples::LoadEarthState(grid, options.state, startKyr);
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto state =
      solver.MakeState(toNd(start.iceThickness), toNd(start.seaLevel));
  const auto history = io::FieldHistory::Read(options.state, grid).value();
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

  // Checkpointed: the viscosity kernels pair the adjoint state with the
  // forward one, so every stride-th forward state is kept and each sweep
  // rebuilds the ones between by forward steps from the last one kept.
  const auto stride = examples::CheckpointStride(nodes);
  auto evolutionOptions = SeaLevelEvolution3D::Options{};
  evolutionOptions.record_stride = stride;
  evolutionOptions.on_snapshot =
      examples::NodeReporter<SeaLevelEvolution3D::Snapshot>(nodes - 1,
                                                            scales.Time());
  examples::Say(std::format(
      "the forward run, {} steps, one state in {} kept:", nodes - 1, stride));
  SeaLevelEvolution3D evolution(op, solver, state, evolutionOptions);
  const auto snapshots = evolution.Run(times, iceLoad);
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);

  // The two observations, as node-keyed duals: J is in the solver's length
  // unit for the sea level, and in kyr for the decay time (the series is
  // paired in solver units, so its weights pair with the fields as they
  // stand).
  const GeographicPoint site{60.0, 275.0};  // Hudson Bay
  const auto a = solver.Radius();
  const auto dual = CapMeanDual(grid, a, site, 0.05);
  auto seaLevel = std::map<std::size_t, Measurement>{};
  seaLevel[nodes - 1] = FieldMeasurement(FieldChannel::SeaLevel, dual);
  auto windowNodes = std::vector<std::size_t>{};
  auto elapsed = std::vector<double>{};
  auto series = std::vector<double>{};
  for (auto n = std::size_t{0}; n < snapshots.size(); ++n) {
    const auto dateKyr = startKyr - startKyr * n / (nodes - 1);
    if (dateKyr > decayFromKyr + 1e-9) continue;
    windowNodes.push_back(n);
    elapsed.push_back(decayFromKyr - dateKyr);
    series.push_back(PairedValue(grid, a, dual, snapshots[n].seaLevelChange));
  }
  // The fit's uncertainties in the series' and times' own units: 1 m (in
  // solver units) and exact times.
  auto fit = DecayTimeFit{};
  const auto decay = DecayTimeMeasurements(
      windowNodes, elapsed, series, FieldChannel::SeaLevel, dual,
      {.time_uncertainty = 0.0, .value_uncertainty = 1.0 / length}, &fit);

  // One reverse sweep per observation.
  AdjointSolver3D adjoint(evolution,
                          {.on_node = examples::SweepReporter(nodes - 1)});
  examples::Say("adjoint sweep 1 of 2, the sea-level change:");
  const auto seaLevelKernels =
      adjoint.IceHistoryKernels(times, seaLevel, &trajectory);
  examples::Say("adjoint sweep 2 of 2, the decay time:");
  const auto decayKernels =
      adjoint.IceHistoryKernels(times, decay, &trajectory);

  // The model file: values non-dimensional under the model's scales, each
  // with its objective's SI dimensions; J's SI unit per sweep unit is the
  // solver's length for the sea level, a kyr in seconds for the decay time.
  const auto& modelScales = earth.material.ModelScales();
  struct Objective {
    std::string name, meaning;
    Dimensions dimensions;
    double unitSi;  // the SI value of one unit of J as the sweep has it
    const std::vector<double>& kernel;
  };
  const auto objectives = std::array{
      Objective{"K_ln_eta_sea_level",
                "present-day sea-level change at the site per unit ln "
                "viscosity, per unit volume",
                Dimensions{.length = 1}, length,
                seaLevelKernels.logViscosityKernel3D[0]},
      Objective{"K_ln_eta_decay_time",
                "decay time of the site's sea level since 7 ka per unit ln "
                "viscosity, per unit volume",
                Dimensions{.time = 1}, kyr,
                decayKernels.logViscosityKernel3D[0]}};
  auto scaled = std::vector<std::vector<double>>{};
  auto fields = std::vector<io::KernelField>{};
  for (const auto& objective : objectives) {
    auto values = objective.kernel;
    const auto toModel =
        objective.unitSi / modelScales.Factor(objective.dimensions);
    for (auto& v : values) v *= toModel;
    scaled.push_back(std::move(values));
  }
  for (auto k = std::size_t{0}; k < objectives.size(); ++k) {
    fields.push_back({.name = objectives[k].name,
                      .meaning = objectives[k].meaning,
                      .objective = objectives[k].dimensions,
                      .values = scaled[k]});
  }
  examples::Say("writing viscosity_kernels.nc and kernels_3d.txt");
  const auto written =
      io::WriteKernels("viscosity_kernels.nc", op, earth.model, fields,
                       {.name = "sl3dpp example kernels"});
  if (!written) {
    std::cerr << written.error().message << "\n";
    return 1;
  }

  // Slices and profiles for the plot. Kernels3D holds an angular density
  // with the radial quadrature folded in: VolumeDensity divides out the
  // node's radial weight and r^2, RadialDensity integrates over angle per
  // unit radius; in SI, by the model length cubed and once.
  const auto& coupling = op.Coupling();
  const auto& kernelGrid = coupling.Grid();
  const auto fieldSize = kernelGrid.FieldSize();
  const auto modelLength = modelScales.Length();
  examples::Writer out("kernels_3d.txt");
  out.Comment("state " + options.state.string());
  out.Comment(
      "3D log-viscosity kernels of two observations at a Hudson Bay site; "
      "slices per km^3 (sea level in m, decay time in yr), profiles per km");
  out.Grid(kernelGrid);
  out.Scalar("site_lat", site.latitude);
  out.Scalar("site_lon", site.longitude);
  out.Scalar(
      "sea_level_change_m",
      length * PairedValue(grid, a, dual, snapshots.back().seaLevelChange));
  out.Scalar("decay_time_kyr", fit.decay_time);
  const auto displayUnit =
      std::array{1.0, 1.0 / examples::kYear};  // m; s to yr
  auto volume = std::vector<std::vector<double>>{};
  for (const auto& objective : objectives) {
    volume.push_back(VolumeDensity(coupling, objective.kernel));
  }
  for (const auto depthKm : {400.0, 1200.0, 2200.0}) {
    // the memory node nearest the depth
    const auto target = 1.0 - depthKm * 1.0e3 / modelLength;
    auto node = 0;
    for (auto p = 1; p < coupling.NodeCount(); ++p) {
      if (std::abs(coupling.RadiusOfNode(p) - target) <
          std::abs(coupling.RadiusOfNode(node) - target)) {
        node = p;
      }
    }
    const auto r = coupling.RadiusOfNode(node);
    const auto perKm3 = 1.0e9 / (modelLength * modelLength * modelLength);
    const auto depth =
        static_cast<int>(std::lround((1.0 - r) * modelLength / 1.0e3));
    for (auto k = std::size_t{0}; k < objectives.size(); ++k) {
      auto slice = std::vector<double>(fieldSize);
      for (auto x = std::size_t{0}; x < fieldSize; ++x) {
        slice[x] = volume[k][static_cast<std::size_t>(node) * fieldSize + x] *
                   objectives[k].unitSi * displayUnit[k] * perKm3;
      }
      out.Field(std::format("{}_{}km", objectives[k].name, depth), slice);
    }
  }
  auto radial = std::vector<std::vector<double>>{};
  for (const auto& objective : objectives) {
    radial.push_back(RadialDensity(coupling, objective.kernel));
  }
  auto rows = std::vector<std::vector<double>>{};
  for (auto p = 0; p < coupling.NodeCount(); ++p) {
    auto row =
        std::vector<double>{coupling.RadiusOfNode(p) * modelLength / 1.0e3};
    for (auto k = std::size_t{0}; k < objectives.size(); ++k) {
      // dJ/dr per km
      row.push_back(radial[k][p] * objectives[k].unitSi * displayUnit[k] *
                    1.0e3 / modelLength);
    }
    rows.push_back(row);
  }
  out.Series("radial", {"r_km", "sea_level_per_km", "decay_time_per_km"}, rows);

  std::cout << "wrote viscosity_kernels.nc and kernels_3d.txt (sea level "
               "change "
            << length *
                   PairedValue(grid, a, dual, snapshots.back().seaLevelChange)
            << " m, decay time " << fit.decay_time << " kyr)\n";
  return 0;
}
