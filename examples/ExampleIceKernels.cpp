/**
 * @file ExampleIceKernels.cpp
 * @brief Adjoint sensitivity kernels (stage 5): what a single sea-level
 * observation knows about the ice history and the mantle viscosity.
 *
 * The objective is the cap-averaged sea-level change at a "tide gauge"
 * site at the end of a 10 kyr melt history, so J is in metres. One
 * reverse sweep per method gives, with no forward run needed for the ice
 * kernels (the frozen problem is linear in the load):
 *
 * - the ice-thickness duals: metres of site sea level per metre of ice
 *   change, per unit area — mapped at three epochs;
 * - the quelled rate kernel (pairs with dI/dt, bounded, no deltas);
 * - the radial viscosity kernels dJ / d eta(r), from the recorded
 *   forward trajectory, by BOTH the continuous adjoint (the preferred
 *   path) and the exact discrete transpose: their agreement is the
 *   resolution diagnostic (stage5.md §1).
 *
 * Run from the build tree; plot with plots/plot_kernels.py.
 */

#include <iostream>
#include <numbers>

#include "Common.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

int main() {
  constexpr int lmax = 128;
  constexpr double meltFraction = 0.2;
  constexpr double tEnd = 10.0e3 * examples::kYear;
  constexpr int nodes = 11;

  const auto earth = examples::Earth::Make(lmax, 1e21);
  ViscoelasticOperator op(earth.rheology, lmax);
  SeaLevelSolver solver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
                        SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();

  const auto earthState = examples::LoadEarthState(grid);
  const double length = scales.Length();
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
  std::vector<double> unitLoad(grid.FieldSize());
  for (std::size_t i = 0; i < unitLoad.size(); ++i) {
    unitLoad[i] = -solver.IceDensity() * (1.0 - C[i]) * meltFraction *
                  mask[i] * iceSi[i] / length;
  }
  std::vector<double> times;
  for (int n = 0; n < nodes; ++n) {
    times.push_back(tEnd * n / (nodes - 1) / scales.Time());
  }
  auto iceLoad = [&](double t) {
    const double ramp = t / times.back();
    std::vector<double> load(unitLoad.size());
    for (std::size_t i = 0; i < load.size(); ++i) load[i] = ramp * unitLoad[i];
    return load;
  };
  SeaLevelEvolution evolution(op, solver, state, {.record_stride = 1});
  const auto snapshots = evolution.Run(times, iceLoad);
  const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);

  // The objective: the cap-weighted mean sea-level change at a mid-
  // latitude site at the final node — J in metres, since the dual is
  // normalised to unit area integral.
  const GeographicPoint gauge{-32.05, 115.74};  // Fremantle
  auto site = grid.FromFunction([&](double theta, double phi) {
    return examples::Cap(theta, phi, gauge.CoLatitude(), gauge.Longitude(),
                         0.15);
  });
  const double a = solver.Radius();
  std::vector<double> siteArea(site.size());
  for (std::size_t i = 0; i < site.size(); ++i) siteArea[i] = site[i];
  const double norm = a * a * grid.Integrate(siteArea);
  std::map<std::size_t, Measurement> measurements;
  measurements[nodes - 1].seaLevelDual = site;
  for (auto& v : measurements[nodes - 1].seaLevelDual) v /= norm;

  examples::Writer out("kernels.txt");
  out.Comment(
      "adjoint kernels of the site-mean sea level at 10 kyr under a "
      "linear Greenland melt; Fremantle gauge");
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
    const bool continuous = method == AdjointMethod::Continuous;
    const std::string tag = continuous ? "continuous" : "discrete";
    AdjointSolver adjoint(evolution, {.method = method});
    const auto kernels =
        adjoint.IceHistoryKernels(times, measurements, &trajectory);

    // Thickness duals at three epochs and the first interval's rate
    // kernel, PROJECTED onto the grounded present-day ice: only
    // ice-covered ground can change thickness (the 100 m floor rejects the
    // band-limited field's ringing over bare land), and the unprojected duals
    // are dominated by the (irrelevant) loading sensitivity right at the
    // gauge. J is a length, ice a length: K_SI = K_nd / L^2.
    const double thicknessSi = 1.0 / (length * length);
    std::vector<double> grounded(iceSi.size());
    for (std::size_t i = 0; i < grounded.size(); ++i) {
      grounded[i] = iceSi[i] > 100.0 && C[i] < 0.5 ? 1.0 : 0.0;
    }
    auto projected = [&](const std::vector<double>& dual) {
      std::vector<double> masked(dual.size());
      for (std::size_t i = 0; i < masked.size(); ++i) {
        masked[i] = grounded[i] * dual[i];
      }
      return masked;
    };
    if (continuous) {
      out.Field("ice_sheets", grounded);
      out.Field("thickness_dual_2kyr", projected(kernels.thickness[2]),
                thicknessSi);
      out.Field("thickness_dual_6kyr", projected(kernels.thickness[6]),
                thicknessSi);
      out.Field("thickness_dual_10kyr", projected(kernels.thickness[10]),
                thicknessSi);
      // The quelled kernel pairs with h (dI/dt) — a thickness — so it
      // converts exactly as the thickness duals do.
      out.Field("rate_kernel_first_interval", projected(kernels.rate[0]),
                thicknessSi);
    }

    // The radial viscosity kernel as a density in radius: divide the
    // nodal values by their quadrature weight w jac, convert eta and
    // radius to SI. dJ = int K(r) d eta(r) dr.
    const auto eta = kernels.Viscosity(earth.rheology);
    const double etaSi = matScales.Factor(ViscosityDimensions);
    const double rSi = matScales.Length();
    std::vector<std::vector<double>> rows;
    for (int e = 0; e < mesh.ElementCount(); ++e) {
      const double jac = mesh.Jacobian(e);
      for (int q = 0; q < mesh.NGll(); ++q) {
        const std::size_t i = static_cast<std::size_t>(e) * mesh.NGll() + q;
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
