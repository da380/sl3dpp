/**
 * @file ExampleSeismic.cpp
 * @brief A seismic source: a great thrust off the west coast of the
 * synthetic South America as point sources on a planar fault, its
 * co-seismic response and three centuries of post-seismic relaxation
 * through the 1D evolution with an ocean, on a Maxwell Earth with a weak
 * asthenosphere and on its Burgers twin, and the kernel of a tide gauge in
 * the fault's moment tensors predicting what another rake would do.
 *
 * The fault is the configs' (`sea_level_postseismic.json`): its centroid
 * at 22 S, 74.5 W and 25 km deep, striking north and dipping 18 degrees
 * east under the continent, 600 by 150 km (from 2 to 48 km deep), a
 * scalar moment of 2e22 N m (Mw 8.8) of pure thrust (rake 90), laid out as
 * 12 by 3 point sources (PlanarFaultSources). It acts from the start of a
 * run on the run's own clock, so the first node is the co-seismic (static)
 * response and the later ones, out to 300 years, the relaxation, the ocean
 * closing every node through the sea level equation with full rotational
 * feedback.
 *
 * The Earth is data/prem8.postseismic.deck (make_postseismic_deck.py): an
 * elastic lid 50 km thick over a Maxwell asthenosphere of 1e19 Pa s to
 * 217 km depth, an upper mantle of 1e20 Pa s to 667 km and a lower mantle
 * of 1e22 Pa s. Its Burgers twin (Rheology::Burgers) puts a Kelvin element
 * in series with the asthenosphere's Maxwell element, of the unrelaxed
 * shear modulus and a tenth of the steady viscosity: two Prony terms there
 * (BurgersPronyTerms), the steady viscosity and the unrelaxed modulus
 * unchanged, the rest of the mantle Maxwell as before. The Burgers body's
 * transient relaxes in months, where the Maxwell body has barely begun.
 *
 * The run is linear in the sources, so one reverse sweep gives the
 * derivative of a measurement with respect to every patch's moment tensor
 * at every node (AdjointSolver::Options::moment_points), with no forward
 * trajectory needed. Here the measurement, on the Maxwell Earth, is the
 * relative sea level about a tide gauge on the coast, averaged over 2
 * degrees, a century after the event; contracted with the change of each
 * patch's tensor when the rake turns from 90 to 120 degrees (an oblique
 * thrust), the kernels predict the gauge's change, which a second forward
 * run under the new rake checks (to rounding: the solves are run to
 * 1e-12).
 *
 * The maps and the time series are the fields' means over Gaussian caps of
 * 2 degrees about every point, not their point values: at these degrees
 * the fields of a source this shallow ring at the truncation wavelength,
 * and a point's value is not converged in lmax, while the 2 degree mean is
 * (to a few per cent at the default lmax 64).
 *
 * Outputs (SI), in output/library/seismic/seismic.txt: maps of the 2
 * degree means of the co-seismic vertical, horizontal (east, north) and
 * sea level response and of the change from the event to the last node
 * on the Maxwell Earth, on a grid three times the solver's; the uplift and sea
 * level through time at three places on both Earths; the Maxwell times of the
 * layers and the Burgers body's Prony terms; each patch's position and its
 * kernel contracted with its own mechanism; and the prediction against the
 * second run.
 *
 *   ExampleSeismic [--lmax N] [--state FILE]
 *
 * Run from build/examples; plot with python/plot_seismic.py.
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <span>
#include <vector>

#include "Common.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fault.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kKelvinModulusFraction = 1.0;    // mu_K / mu
constexpr auto kKelvinViscosityFraction = 0.1;  // eta_K / eta
constexpr auto kAsthenosphereBase = 6.151e6;    // m: PREM's 220 km
constexpr auto kDeck = "prem8.postseismic.deck";

/// The post-seismic Earth of data/: its model and material, its own
/// Maxwell rheology (each viscous layer of the deck a Maxwell body), and
/// the Burgers twin, a Kelvin element in series in the asthenosphere (the
/// viscous layers above PREM's 220 km discontinuity).
struct PostseismicEarth {
  EarthModel model;
  Material material;
  Rheology maxwell;
  Rheology burgers;
  std::vector<std::size_t> asthenosphere;

  static PostseismicEarth Make(int lmax) {
    const auto deck = Deck::Read(examples::DataFile(kDeck), GiaFormat());
    if (!deck) throw std::runtime_error(deck.error().message);
    auto model = EarthModel::FromDeck(*deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
    auto material = Material::Create(model, mesh).value();
    auto maxwell = Rheology::FromModel(material, model).value();
    const auto nd = model.DimensionlessView();
    const auto b = model.Boundaries();
    auto viscosity = std::map<std::size_t, RadialValue>{};
    auto kelvin = std::map<std::size_t, Rheology::Kelvin>{};
    auto asthenosphere = std::vector<std::size_t>{};
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.Has("viscosity", i)) continue;
      auto eta = [nd, i](double r) {
        return nd.EvaluateInLayer("viscosity", i, r);
      };
      viscosity[i] = eta;
      if (b[i] >= kAsthenosphereBase - 1.0) {
        kelvin[i] = Rheology::Kelvin{.modulus_fraction = kKelvinModulusFraction,
                                     .viscosity = [eta](double r) {
                                       return kKelvinViscosityFraction * eta(r);
                                     }};
        asthenosphere.push_back(i);
      }
    }
    auto burgers = Rheology::Burgers(material, viscosity, kelvin).value();
    return PostseismicEarth{std::move(model), std::move(material),
                            std::move(maxwell), std::move(burgers),
                            std::move(asthenosphere)};
  }
};

/// A branch's relaxation times over a set of layers' nodes, years:
/// {shortest, longest}, and the range of its modulus as a fraction of the
/// unrelaxed shear modulus.
struct BranchRange {
  double shortest = std::numeric_limits<double>::infinity();
  double longest = 0.0;
  double leastFraction = std::numeric_limits<double>::infinity();
  double mostFraction = 0.0;
};

/// The degree weights of a mean over a Gaussian cap of angular width w
/// (radians), exp(-d^2 / 2 w^2) in the distance d, degrees 0 to lmax: by
/// Funk and Hecke the cap mean of a field about every point is its
/// expansion with degree l weighted by the cap's mean of P_l(cos d), here
/// by Simpson's rule in d out to 12 w.
std::vector<double> CapWeights(int lmax, double width) {
  const auto reach = std::min(std::numbers::pi, 12.0 * width);
  constexpr auto intervals = 4000;
  const auto h = reach / intervals;
  auto sums = std::vector<double>(static_cast<std::size_t>(lmax) + 1, 0.0);
  for (auto i = 0; i <= intervals; ++i) {
    const auto d = i * h;
    const auto simpson = (i == 0 || i == intervals) ? 1.0 : (i % 2 ? 4.0 : 2.0);
    const auto weight =
        simpson * std::exp(-0.5 * (d / width) * (d / width)) * std::sin(d);
    const auto x = std::cos(d);
    auto before = 1.0, current = x;
    sums[0] += weight;
    for (auto l = 1; l <= lmax; ++l) {
      sums[static_cast<std::size_t>(l)] += weight * current;
      const auto next =
          ((2.0 * l + 1.0) * x * current - l * before) / (l + 1.0);
      before = current;
      current = next;
    }
  }
  for (auto l = lmax; l >= 0; --l) {
    sums[static_cast<std::size_t>(l)] /= sums[0];
  }
  return sums;
}

BranchRange RangeOf(const Rheology& rheology, std::size_t branch,
                    std::span<const std::size_t> layers, double timeScale) {
  const auto& mesh = rheology.BaseMaterial().Mesh();
  const auto& of = rheology.BranchAt(branch);
  const auto mu = rheology.MuUnrelaxed();
  auto out = BranchRange{};
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    const auto layer = static_cast<std::size_t>(mesh.LayerOf(e));
    if (std::ranges::find(layers, layer) == layers.end()) continue;
    for (auto i = 0; i < mesh.NGll(); ++i) {
      const auto node = static_cast<std::size_t>(e) * mesh.NGll() + i;
      if (!(of.mu[node] > 0.0)) continue;
      const auto tau = timeScale / of.itau[node] / examples::kYear;
      out.shortest = std::min(out.shortest, tau);
      out.longest = std::max(out.longest, tau);
      out.leastFraction = std::min(out.leastFraction, of.mu[node] / mu[node]);
      out.mostFraction = std::max(out.mostFraction, of.mu[node] / mu[node]);
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const auto options = examples::ParseOptions(argc, argv, 64);
  const auto lmax = options.lmax;
  constexpr auto alongStrike = 12, downDip = 3;
  constexpr auto rake = 90.0, otherRake = 120.0;
  // dense where the Burgers body's transient relaxes (months), then about
  // a third of the time elapsed a step, out past the upper mantle's
  // Maxwell times
  const auto years = std::vector<double>{
      0.0,  0.01, 0.02, 0.04, 0.07, 0.12,  0.2,   0.3,   0.45, 0.7,
      1.0,  1.5,  2.0,  3.0,  4.0,  5.5,   7.5,   10.0,  13.0, 17.0,
      22.0, 30.0, 40.0, 55.0, 75.0, 100.0, 140.0, 200.0, 300.0};
  const auto measured = static_cast<std::size_t>(
      std::ranges::find(years, 100.0) - years.begin());  // a century on

  examples::Say(
      std::format("ExampleSeismic: a thrust fault at lmax {}, state {}", lmax,
                  options.state.string()));
  examples::Say(std::format(
      "building data/{}: its Maxwell rheology and the Burgers twin", kDeck));
  const auto earth = PostseismicEarth::Make(lmax);
  ViscoelasticOperator op(earth.maxwell, lmax);
  ViscoelasticOperator burgersOp(earth.burgers, lmax);
  GRDSolver solver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
                   SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto energy =  // a moment, N m, in the solver's units
      scales.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});

  // The Maxwell times of the deck's viscous layers, those of one viscosity
  // together, and the Burgers body's two terms in the asthenosphere.
  const auto modelTime = earth.model.ModelScales().Time();
  auto layerRows = std::vector<std::vector<double>>{};
  {
    const auto b = earth.model.Boundaries();
    const auto top = b.back();
    auto group = std::vector<std::size_t>{};
    auto groupEta = 0.0;
    auto flush = [&] {
      if (group.empty()) return;
      const auto range = RangeOf(earth.maxwell, 0, group, modelTime);
      layerRows.push_back({(top - b[group.back() + 1]) / 1.0e3,
                           (top - b[group.front()]) / 1.0e3, groupEta,
                           range.shortest, range.longest});
      group.clear();
    };
    for (auto i = std::size_t{0}; i < earth.model.LayerCount(); ++i) {
      if (!earth.model.Has("viscosity", i)) {
        flush();
        continue;
      }
      const auto eta = earth.model.EvaluateInLayer("viscosity", i, b[i + 1]);
      if (!group.empty() && eta != groupEta) flush();
      group.push_back(i);
      groupEta = eta;
    }
    flush();
    std::ranges::reverse(layerRows);  // from the surface down
  }
  const auto slow = RangeOf(earth.burgers, 0, earth.asthenosphere, modelTime);
  const auto fast = RangeOf(earth.burgers, 1, earth.asthenosphere, modelTime);
  for (const auto& row : layerRows) {
    examples::Say(std::format(
        "  {:5.0f} to {:5.0f} km, {:.0e} Pa s: Maxwell times {:.3g} to {:.3g} "
        "years",
        row[0], row[1], row[2], row[3], row[4]));
  }
  examples::Say(std::format(
      "  the Burgers asthenosphere (mu_K = {:g} mu, eta_K = {:g} eta): "
      "{:.3f} mu at {:.3g} to {:.3g} years, {:.3f} mu at {:.3g} to {:.3g} "
      "years",
      kKelvinModulusFraction, kKelvinViscosityFraction, slow.leastFraction,
      slow.shortest, slow.longest, fast.leastFraction, fast.shortest,
      fast.longest));

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
  const auto toMeasured = std::span(times).first(measured + 1);
  // tight solves, so that the kernels' prediction below is checked to
  // rounding rather than to the sea level solves' tolerance
  Evolution evolution(op, solver, state, {.rtol = 1e-12});
  Evolution burgersEvolution(burgersOp, solver, state, {.rtol = 1e-12});
  auto run = [](Evolution& on, std::span<const double> at,
                const std::vector<PointSource>& under) {
    return on.Run(
        at, Evolution::Forcing{.sources = [&under](double) { return under; }});
  };
  examples::Say(std::format(
      "the forward runs, {} nodes to {} years, rake {}: the Maxwell Earth",
      times.size(), years.back(), rake));
  const auto snapshots = run(evolution, times, sources);
  examples::Say("  and the Burgers Earth");
  const auto burgersSnapshots = run(burgersEvolution, times, sources);

  // The relative sea level a century on about a tide gauge on the coast,
  // averaged over a Gaussian cap of 2 degrees, J in metres: the cap mean
  // is the steady datum where the band-limited fields ring about a source
  // shallower than they resolve.
  const GeographicPoint gauge{-22.0, 288.0};
  const auto a = solver.Radius();
  const auto capWidth = 2.0 * std::numbers::pi / 180.0;
  const auto dual = CapMeanDual(grid, a, gauge, capWidth);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[measured] = FieldMeasurement(FieldChannel::SeaLevel, dual);
  auto valueOf = [&](const std::vector<Evolution::Snapshot>& of) {
    return length *
           PairedValue(grid, a, dual, of[measured].relativeSeaLevelChange);
  };

  // The moment kernels at every patch and node to the measurement's, by
  // the discrete adjoint on the Maxwell Earth.
  examples::Say("the adjoint sweep, the moment kernels at every patch");
  auto points = std::vector<BodyPoint>{};
  for (const auto& source : sources) points.push_back(source.at);
  const auto kernels =
      AdjointSolver(evolution, {.method = AdjointMethod::Discrete,
                                .rtol = 1e-12,
                                .moment_points = points})
          .FirstOrderKernels(toMeasured, measurements);

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
  examples::Say(std::format("the second forward run to {} years, rake {}",
                            years[measured], otherRake));
  const auto other = run(evolution, toMeasured, turned);
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

  // The maps and the series show the fields' means over Gaussian caps of
  // 2 degrees about every point: the fields of a fault a few kilometres
  // deep, laid out as point sources 50 km apart, are not converged in
  // lmax at any degree a run like this reaches (the truncation rings about
  // the source, and finer degrees resolve the point sources one by one),
  // while their 2 degree means are (to a few per cent at lmax 64, a few
  // tenths of one at lmax 96).
  const auto weights = CapWeights(lmax, capWidth);
  auto meanOf = [&](const std::vector<double>& field) {
    auto coefficients = grid.Expand(field);
    for (auto l = 0; l <= lmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        coefficients[SurfaceGrid::Index(l, m)] *=
            weights[static_cast<std::size_t>(l)];
      }
    }
    return coefficients;
  };
  // The maps on a grid three times the solver's in each direction, the
  // means synthesised there from their coefficients (the ocean function
  // as it stands).
  const SurfaceGrid fine(lmax, {.nmax = 1, .oversampling = 3.0});
  auto onFine = [&](const std::vector<double>& field) {
    if (field.empty()) return std::vector<double>{};
    return fine.Synthesise(meanOf(field));
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
      "dipping 18 degrees east, 12 by 3 point sources) on "
      "data/prem8.postseismic.deck (Maxwell) and its Burgers twin, with an "
      "ocean; SI");
  out.Grid(fine);
  out.Field("up_event_m", atEvent.up, length);
  out.Field("east_event_m", atEvent.east, length);
  out.Field("north_event_m", atEvent.north, length);
  out.Field("sea_level_event_m", onFine(event.relativeSeaLevelChange), length);
  out.Field("ocean", fine.Synthesise(grid.Expand(state.oceanFunction)));
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
  out.Scalar("cap_width_deg", 2.0);
  out.Scalar("measured_yr", years[measured]);
  out.Scalar("rake_deg", rake);
  out.Scalar("other_rake_deg", otherRake);
  out.Scalar("gauge_m", value);
  out.Scalar("change_by_run_m", actual);
  out.Scalar("change_by_kernels_m", predicted);
  out.Scalar("relative_difference",
             std::abs(predicted - actual) / std::abs(actual));
  out.Scalar("kelvin_modulus_fraction", kKelvinModulusFraction);
  out.Scalar("kelvin_viscosity_fraction", kKelvinViscosityFraction);
  out.Series("layers",
             {"top_km", "bottom_km", "viscosity_pa_s", "maxwell_min_yr",
              "maxwell_max_yr"},
             layerRows);
  out.Series(
      "burgers", {"fraction_min", "fraction_max", "tau_min_yr", "tau_max_yr"},
      {{slow.leastFraction, slow.mostFraction, slow.shortest, slow.longest},
       {fast.leastFraction, fast.mostFraction, fast.shortest, fast.longest}});
  out.Series("patches", {"lat", "lon", "depth_km", "dJ_dslip_mm_per_1e21_Nm"},
             patches);

  // The uplift and sea level through time over the fault, about the gauge
  // and in the far field to the west, each the 2 degree mean there, on the
  // Maxwell Earth and on the Burgers one.
  const GeographicPoint places[] = {{-22.0, 285.5}, gauge, {-22.0, 276.0}};
  auto duals = std::vector<std::vector<double>>{};
  for (const auto& place : places) duals.push_back(PointDual(grid, a, place));
  auto rows = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < snapshots.size(); ++n) {
    auto row = std::vector<double>{years[n]};
    for (const auto* of : {&snapshots, &burgersSnapshots}) {
      const auto up = grid.Synthesise(meanOf((*of)[n].displacement));
      const auto sea = grid.Synthesise(meanOf((*of)[n].relativeSeaLevelChange));
      for (const auto& at : duals) {
        row.push_back(length * PairedValue(grid, a, at, up));
        row.push_back(length * PairedValue(grid, a, at, sea));
      }
    }
    rows.push_back(std::move(row));
  }
  out.Series("history",
             {"t_yr", "up_fault_m", "sea_fault_m", "up_gauge_m", "sea_gauge_m",
              "up_far_m", "sea_far_m", "burgers_up_fault_m",
              "burgers_sea_fault_m", "burgers_up_gauge_m",
              "burgers_sea_gauge_m", "burgers_up_far_m", "burgers_sea_far_m"},
             rows);
  out.Series("places", {"lat", "lon"},
             {{places[0].latitude, places[0].longitude},
              {places[1].latitude, places[1].longitude},
              {places[2].latitude, places[2].longitude}});

  std::cout << "wrote " << table.string() << "\n";
  return 0;
}
