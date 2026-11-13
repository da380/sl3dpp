/**
 * @file ExampleSeismic.cpp
 * @brief A seismic source: a great thrust off the west coast of the
 * synthetic South America as point sources on a planar fault, its
 * co-seismic response and three centuries of post-seismic relaxation
 * through the 1D evolution with an ocean, and the kernel of a tide gauge
 * in the fault's moment tensors predicting what another rake would do.
 *
 * The fault is the configs' (`sea_level_postseismic.json`): its centroid
 * at 22 S, 74.5 W and 25 km deep, striking north and dipping 18 degrees
 * east under the continent, 600 by 150 km, a scalar moment of 2e22 N m
 * (Mw 8.8) of pure thrust (rake 90), laid out as 12 by 3 point sources
 * (PlanarFaultSources). It acts from the start of a run on the run's own
 * clock, so the first node is the co-seismic (static) response and the
 * later ones, out to 300 years, the relaxation of a 1e21 Pa s mantle,
 * the ocean closing every node through the sea level equation with full
 * rotational feedback.
 *
 * The run is linear in the sources, so one reverse sweep gives the
 * derivative of a measurement with respect to every patch's moment tensor
 * at every node (AdjointSolver::Options::moment_points), with no forward
 * trajectory needed. Here the measurement is the relative sea level
 * about a tide gauge on the coast, averaged over 2 degrees, a century
 * after the event; contracted with the change of each patch's tensor when
 * the rake turns from 90 to 120 degrees (an oblique thrust), the kernels
 * predict the gauge's change, which a second forward run under the new
 * rake checks (to rounding: the solves are run to 1e-12).
 *
 * Outputs (SI), in output/library/seismic/seismic.txt: maps of the
 * co-seismic vertical, horizontal (east, north) and sea level response,
 * of the change from the event to the last node, on a grid three times
 * the solver's; the uplift and sea level through time at three places;
 * each patch's position and its kernel contracted with its own mechanism;
 * and the prediction against the second run.
 *
 *   ExampleSeismic [--lmax N] [--state FILE]
 *
 * Run from build/examples; plot with python/plot_seismic.py.
 */

#include <cmath>
#include <format>
#include <iostream>
#include <map>
#include <numbers>
#include <span>
#include <vector>

#include "Common.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fault.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

int main(int argc, char** argv) {
  const auto options = examples::ParseOptions(argc, argv, 64);
  const auto lmax = options.lmax;
  constexpr auto alongStrike = 12, downDip = 3;
  constexpr auto rake = 90.0, otherRake = 120.0;
  const auto years =
      std::vector<double>{0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0, 300.0};
  constexpr auto measured = std::size_t{5};  // the node at 100 years

  examples::Say(
      std::format("ExampleSeismic: a thrust fault at lmax {}, state {}", lmax,
                  options.state.string()));
  examples::Say("building the model, the operator and the sea level solver");
  const auto earth = examples::Earth::Make(lmax, 1e21);
  ViscoelasticOperator op(earth.rheology, lmax);
  GRDSolver solver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
                   SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto energy =  // a moment, N m, in the solver's units
      scales.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});

  // The present-day background state, in the solver's units.
  const auto earthState = examples::LoadEarthState(grid, options.state);
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto state = solver.MakeState(toNd(earthState.iceThickness),
                                      toNd(earthState.seaLevel));

  // The fault, in the solver's units, as point sources at its patches'
  // centres; each source acts from the start of the run.
  auto fault = PlanarFault{.centroid = {-22.0, 285.5},
                           .depth = 25.0e3 / length,
                           .strike = 0.0,
                           .dip = 18.0,
                           .rake = rake,
                           .length = 600.0e3 / length,
                           .width = 150.0e3 / length,
                           .moment = 2.0e22 / energy};
  const auto sources =
      PlanarFaultSources(fault, solver.Radius(), alongStrike, downDip);
  fault.rake = otherRake;
  const auto turned =
      PlanarFaultSources(fault, solver.Radius(), alongStrike, downDip);

  auto times = std::vector<double>{};
  for (const auto t : years)
    times.push_back(t * examples::kYear / scales.Time());
  // tight solves, so that the kernels' prediction below is checked to
  // rounding rather than to the sea level solves' tolerance
  Evolution evolution(op, solver, state, {.rtol = 1e-12});
  auto run = [&](const std::vector<PointSource>& under) {
    return evolution.Run(times, Evolution::Forcing{.sources = [&under](double) {
                           return under;
                         }});
  };
  examples::Say(std::format("the forward run, {} nodes to {} years, rake {}",
                            times.size(), years.back(), rake));
  const auto snapshots = run(sources);

  // The relative sea level a century on about a tide gauge on the coast,
  // averaged over a Gaussian cap of 2 degrees, J in metres: the cap mean
  // is the steady datum where the band-limited fields ring about a source
  // shallower than they resolve.
  const GeographicPoint gauge{-22.0, 288.0};
  const auto a = solver.Radius();
  const auto dual = CapMeanDual(grid, a, gauge, 2.0 * std::numbers::pi / 180.0);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[measured] = FieldMeasurement(FieldChannel::SeaLevel, dual);
  auto valueOf = [&](const std::vector<Evolution::Snapshot>& of) {
    return length *
           PairedValue(grid, a, dual, of[measured].relativeSeaLevelChange);
  };

  // The moment kernels at every patch and node, by the discrete adjoint.
  examples::Say("the adjoint sweep, the moment kernels at every patch");
  auto points = std::vector<BodyPoint>{};
  for (const auto& source : sources) points.push_back(source.at);
  const auto kernels =
      AdjointSolver(evolution, {.method = AdjointMethod::Discrete,
                                .rtol = 1e-12,
                                .moment_points = points})
          .FirstOrderKernels(times, measurements);

  // delta J = sum_n sum_p K_np : delta M_p, every source held from the
  // start; and each patch's kernel contracted with its own unit-moment
  // mechanism, the derivative with respect to its slip's moment.
  const auto share = 2.0e22 / energy / (alongStrike * downDip);
  auto predicted = 0.0;
  auto patches = std::vector<std::vector<double>>{};
  for (auto p = std::size_t{0}; p < sources.size(); ++p) {
    auto change = turned[p].moment;
    const auto& was = sources[p].moment;
    change.rr -= was.rr;
    change.tt -= was.tt;
    change.pp -= was.pp;
    change.rt -= was.rt;
    change.rp -= was.rp;
    change.tp -= was.tp;
    auto slip = 0.0;
    for (const auto& atNode : kernels.momentKernels) {
      predicted += atNode[p].Contract(change);
      slip += atNode[p].Contract(was) / share;
    }
    const auto site = GeographicPoint::FromSpherical(sources[p].at.colatitude,
                                                     sources[p].at.longitude);
    patches.push_back({site.latitude, site.longitude,
                       (a - sources[p].at.radius) * length / 1.0e3,
                       // mm of sea level per 1e21 N m
                       slip * length / energy * 1.0e21 * 1.0e3});
  }
  predicted *= length;
  examples::Say(std::format("the second forward run, rake {}", otherRake));
  const auto other = run(turned);
  const auto value = valueOf(snapshots);
  const auto actual = valueOf(other) - value;
  examples::Say(std::format(
      "the gauge's mean at {} years: {:.6f} m under rake {}, {:.6f} m under "
      "rake {}",
      years[measured], value, rake, value + actual, otherRake));
  examples::Say(std::format(
      "the change of rake: {:.9f} m by the second run, {:.9f} m by the "
      "kernels, relative difference {:.1e}",
      actual, predicted, std::abs(predicted - actual) / std::abs(actual)));

  // The maps on a grid three times the solver's in each direction, the
  // fields synthesised there from their coefficients.
  const SurfaceGrid fine(lmax, {.nmax = 1, .oversampling = 3.0});
  auto onFine = [&](const std::vector<double>& field) {
    if (field.empty()) return std::vector<double>{};
    return fine.Synthesise(grid.Expand(field));
  };
  // the horizontal motion in the frame of no net rotation of the surface,
  // a rigid rotation of the body being undetermined (Measurements.hpp)
  auto motion = [&](const Evolution::Snapshot& s) {
    return DisplacementField(
        fine, onFine(s.displacement), onFine(s.tangentialPotential),
        onFine(WithoutNetRotation(grid, s.toroidalPotential)));
  };
  const auto& event = snapshots.front();
  const auto& last = snapshots.back();
  const auto atEvent = motion(event);
  auto difference = [&](const std::vector<double>& to,
                        const std::vector<double>& from) {
    auto out = onFine(to);
    const auto base = onFine(from);
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] -= base[i];
    return out;
  };

  const auto table = examples::OutputDirectory("seismic") / "seismic.txt";
  examples::Writer out(table);
  out.Comment("state " + options.state.string());
  out.Comment(
      "a thrust off the synthetic South America (2e22 N m, 600 by 150 km, "
      "dipping 18 degrees east, 12 by 3 point sources) on prem8 with a "
      "1e21 Pa s mantle, with an ocean; SI");
  out.Grid(fine);
  out.Field("up_event_m", atEvent.up, length);
  out.Field("east_event_m", atEvent.east, length);
  out.Field("north_event_m", atEvent.north, length);
  out.Field("sea_level_event_m", onFine(event.relativeSeaLevelChange), length);
  out.Field("ocean", onFine(state.oceanFunction));
  out.Field("up_change_m", difference(last.displacement, event.displacement),
            length);
  out.Field(
      "sea_level_change_m",
      difference(last.relativeSeaLevelChange, event.relativeSeaLevelChange),
      length);
  out.Scalar("lmax", lmax);
  out.Scalar("last_yr", years.back());
  out.Scalar("centroid_lat", -22.0);
  out.Scalar("centroid_lon", 285.5);
  out.Scalar("strike_deg", 0.0);
  out.Scalar("dip_deg", 18.0);
  out.Scalar("length_km", 600.0);
  out.Scalar("width_km", 150.0);
  out.Scalar("patches_along_strike", alongStrike);
  out.Scalar("patches_down_dip", downDip);
  out.Scalar("gauge_lat", gauge.latitude);
  out.Scalar("gauge_lon", gauge.longitude);
  out.Scalar("measured_yr", years[measured]);
  out.Scalar("rake_deg", rake);
  out.Scalar("other_rake_deg", otherRake);
  out.Scalar("gauge_m", value);
  out.Scalar("change_by_run_m", actual);
  out.Scalar("change_by_kernels_m", predicted);
  out.Scalar("relative_difference",
             std::abs(predicted - actual) / std::abs(actual));
  out.Series("patches", {"lat", "lon", "depth_km", "dJ_dslip_mm_per_1e21_Nm"},
             patches);

  // The uplift and sea level through time over the fault and in the far
  // field to the west, read by the band-limited point evaluation, as the
  // maps show them, and about the gauge by its cap mean, as measured.
  const GeographicPoint places[] = {{-22.0, 285.5}, gauge, {-22.0, 276.0}};
  auto rows = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < snapshots.size(); ++n) {
    auto row = std::vector<double>{years[n]};
    for (auto k = 0; k < 3; ++k) {
      const auto at = k == 1 ? dual : PointDual(grid, a, places[k]);
      row.push_back(length *
                    PairedValue(grid, a, at, snapshots[n].displacement));
      row.push_back(length * PairedValue(grid, a, at,
                                         snapshots[n].relativeSeaLevelChange));
    }
    rows.push_back(std::move(row));
  }
  out.Series("history",
             {"t_yr", "up_fault_m", "sea_fault_m", "up_gauge_m", "sea_gauge_m",
              "up_far_m", "sea_far_m"},
             rows);
  out.Series("places", {"lat", "lon"},
             {{places[0].latitude, places[0].longitude},
              {places[1].latitude, places[1].longitude},
              {places[2].latitude, places[2].longitude}});

  std::cout << "wrote " << table.string() << "\n";
  return 0;
}
