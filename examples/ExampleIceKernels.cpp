/**
 * @file ExampleIceKernels.cpp
 * @brief Adjoint sensitivity kernels: what a single sea-level
 * observation knows about the ice history and the mantle viscosity.
 *
 * The objective is the cap-averaged sea-level change at a "tide gauge"
 * off south-east Greenland at the end of a 10 kyr melt of the Greenland
 * ice, so J is in metres. The gauge is in the near field, where the solid
 * Earth's response to the ice lost (the elastic fall, then the viscous
 * one) outweighs the eustatic share, so what the observation sees of the
 * ice history depends on when the ice was lost. One
 * reverse sweep per method gives, with no forward run needed for the ice
 * kernels (the frozen problem is linear in the load):
 *
 * - the quelled rate kernels at four intervals of the history: metres of
 *   site sea level per metre of ice lost during the interval, per unit
 *   area. Ice lost then stays lost until the observation, so each is the
 *   tail sum of the later node-wise thickness duals: bounded, and free of
 *   the delta those carry at the observation;
 * - the radial viscosity kernels dJ / d eta(r), from the recorded
 *   forward trajectory, by BOTH the continuous adjoint (the preferred
 *   path) and the exact discrete transpose: their agreement is the
 *   resolution diagnostic (docs/dev/adjoints.md §1).
 *
 *   ExampleIceKernels [--lmax N] [--state FILE]
 *
 * Run from the build tree; plot with python/plot_kernels.py.
 */

#include <format>
#include <iostream>
#include <numbers>

#include "Common.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

int main(int argc, char** argv) {
  const auto options = examples::ParseOptions(argc, argv, 128);
  const auto lmax = options.lmax;
  constexpr auto meltFraction = 0.2;
  constexpr auto tEnd = 10.0e3 * examples::kYear;
  constexpr auto nodes = 11;

  examples::Say(
      std::format("ExampleIceKernels: 1D kernels at lmax {}, state {}", lmax,
                  options.state.string()));
  examples::Say("building the model, the operator and the sea level solver");
  const auto earth = examples::Earth::Make(lmax, 1e21);
  ViscoelasticOperator op(earth.rheology, lmax);
  SeaLevelSolver solver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
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
  auto state =
      solver.MakeState(toNd(iceSi), toNd(earthState.seaLevel));

  // The forward history: a linear Greenland melt over the whole window,
  // recorded every node (the viscosity kernels pair with the
  // trajectory).
  const GeographicPoint greenland{72.0, 320.0};
  const auto mask = grid.FromFunction([&](double theta, double phi) {
    return examples::Cap(theta, phi, greenland.CoLatitude(),
                         greenland.Longitude(), 0.28);
  });
  const auto& C = state.oceanFunction;
  auto unitLoad = std::vector<double>(grid.FieldSize());
  for (auto i = std::size_t{0}; i < unitLoad.size(); ++i) {
    unitLoad[i] = -solver.IceDensity() * (1.0 - C[i]) * meltFraction *
                  mask[i] * iceSi[i] / length;
  }
  auto times = std::vector<double>{};
  for (auto n = 0; n < nodes; ++n) {
    times.push_back(tEnd * n / (nodes - 1) / scales.Time());
  }
  auto iceLoad = [&](double t) {
    const auto ramp = t / times.back();
    auto load = std::vector<double>(unitLoad.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i)
      load[i] = ramp * unitLoad[i];
    return load;
  };
  // Checkpointed: every other state is kept, and each sweep rebuilds the
  // one between by an exact backward step (ViscoelasticOperator::
  // StepBack). A backward step runs the dissipation the other way, so it
  // amplifies the forward solves' error by exp(h) per step, h = dt / tau;
  // here h is about 2, so one step per gap rather than the sqrt(N) of
  // examples::CheckpointStride (which suits the 3D sweep, whose rebuilt
  // steps go forward).
  constexpr auto stride = 2;
  examples::Say(std::format(
      "the forward run, {} steps, one state in {} kept:", nodes - 1, stride));
  SeaLevelEvolution evolution(
      op, solver, state,
      {.record_stride = stride,
       .on_snapshot = examples::NodeReporter<SeaLevelEvolution::Snapshot>(
           nodes - 1, scales.Time())});
  const auto snapshots = evolution.Run(times, iceLoad);
  const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);

  // The objective: the cap-weighted mean sea-level change at a site off
  // south-east Greenland at the final node — J in metres, since the dual
  // is normalised to unit area integral. The cap (about 3 degrees) keeps
  // the measurement local.
  const GeographicPoint gauge{65.0, 330.0};
  auto site = grid.FromFunction([&](double theta, double phi) {
    return examples::Cap(theta, phi, gauge.CoLatitude(), gauge.Longitude(),
                         0.05);
  });
  const auto a = solver.Radius();
  auto siteArea = std::vector<double>(site.size());
  for (auto i = std::size_t{0}; i < site.size(); ++i) siteArea[i] = site[i];
  const auto norm = a * a * grid.Integrate(siteArea);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[nodes - 1].seaLevelDual = site;
  for (auto& v : measurements[nodes - 1].seaLevelDual) v /= norm;

  examples::Writer out("kernels.txt");
  out.Comment("state " + options.state.string());
  out.Comment(
      "adjoint kernels of the site-mean sea level at 10 kyr under a "
      "linear Greenland melt; gauge off south-east Greenland");
  out.Comment(
      "thickness duals in (site metres) per metre of ice per m^2; "
      "viscosity kernels in (site metres) per (Pa s) per metre of radius");
  out.Grid(grid);
  out.Scalar("gauge_lat", gauge.latitude);
  out.Scalar("gauge_lon", gauge.longitude);
  out.Field("site_dual", measurements[nodes - 1].seaLevelDual);
  out.Field("ice_change_final_m", unitLoad);  // the load shape, context

  const auto& mesh = earth.material.Mesh();
  const auto& matScales = earth.material.ModelScales();
  const auto w = mesh.Weights();
  for (auto method : {AdjointMethod::Continuous, AdjointMethod::Discrete}) {
    const auto continuous = method == AdjointMethod::Continuous;
    const std::string tag = continuous ? "continuous" : "discrete";
    examples::Say(std::format("the {} adjoint sweep:", tag));
    AdjointSolver adjoint(
        evolution,
        {.method = method, .on_node = examples::SweepReporter(nodes - 1)});
    const auto kernels =
        adjoint.IceHistoryKernels(times, measurements, &trajectory);

    // The rate kernels of four intervals, PROJECTED onto the grounded
    // present-day ice: only
    // ice-covered ground can change thickness (the 100 m floor rejects the
    // band-limited field's ringing over bare land), and the unprojected duals
    // are dominated by the (irrelevant) loading sensitivity right at the
    // gauge. J is a length, ice a length: K_SI = K_nd / L^2.
    const auto thicknessSi = 1.0 / (length * length);
    auto grounded = std::vector<double>(iceSi.size());
    for (auto i = std::size_t{0}; i < grounded.size(); ++i) {
      grounded[i] = iceSi[i] > 100.0 && C[i] < 0.5 ? 1.0 : 0.0;
    }
    auto projected = [&](const std::vector<double>& dual) {
      auto masked = std::vector<double>(dual.size());
      for (auto i = std::size_t{0}; i < masked.size(); ++i) {
        masked[i] = grounded[i] * dual[i];
      }
      return masked;
    };
    if (continuous) {
      out.Field("ice_sheets", grounded);
      // rate[k - 1] is interval k, from node k - 1 to node k. It pairs
      // with the interval's thickness change, so it converts as a
      // thickness dual does.
      out.Scalar("interval_kyr",
                 tEnd / (nodes - 1) / (1.0e3 * examples::kYear));
      out.Scalar("history_kyr", tEnd / (1.0e3 * examples::kYear));
      for (const auto interval : {1, 4, 7, 10}) {
        out.Field(
            std::format("rate_kernel_{}", interval),
            projected(kernels.rate[static_cast<std::size_t>(interval - 1)]),
            thicknessSi);
      }
    }

    // The radial viscosity kernel as a density in radius: divide the
    // nodal values by their quadrature weight w jac, convert eta and
    // radius to SI. dJ = int K(r) d eta(r) dr.
    const auto eta = kernels.Viscosity(earth.rheology);
    const auto etaSi = matScales.Factor(ViscosityDimensions);
    const auto rSi = matScales.Length();
    auto rows = std::vector<std::vector<double>>{};
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      const auto jac = mesh.Jacobian(e);
      for (auto q = 0; q < mesh.NGll(); ++q) {
        const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
        if (earth.rheology.BranchAt(0).itau[i] <= 0.0) continue;
        rows.push_back({mesh.Radius(e, q) * rSi / 1.0e3,  // km
                        eta[0][i] / (w[q] * jac) / (etaSi * rSi)});
      }
    }
    out.Series("viscosity_kernel_" + tag, {"r_km", "dJ_deta"}, rows);
  }

  std::cout << "wrote kernels.txt\n";
  return 0;
}
