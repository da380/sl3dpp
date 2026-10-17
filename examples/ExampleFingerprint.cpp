/**
 * @file ExampleFingerprint.cpp
 * @brief Elastic sea-level fingerprint of a 10% Greenland melt — the
 * stage-2 public capability (plan §1), at full resolution (lmax 256) on
 * the real present-day Earth (ICE-7G ice and topography, extracted from
 * pyslfp by the build).
 *
 * One equilibrium solve of the generalised sea level equation with full
 * rotational feedback gives the fingerprint: the near-field sea-level
 * FALL from lost self-attraction and uplift, the far-field rise above
 * the eustatic mean, and the polar motion; all SI.
 *
 * Run from the build tree; plot with plots/plot_fingerprint.py.
 */

#include <iostream>
#include <numbers>

#include "Common.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/LoveNumbers.hpp"

using namespace sl3dpp;

int main() {
  constexpr auto lmax = 256;
  const auto earth = examples::Earth::Make(lmax, 1e21);  // viscosity unused
  const auto love = ComputeLoveNumbers(earth.material, lmax);

  Fingerprint fingerprint(lmax, love);
  const auto& grid = fingerprint.Grid();
  const auto state = examples::LoadEarthState(grid);
  fingerprint.SetState(state.iceThickness, state.seaLevel);

  // Melt 10% of the Greenland ice: the real cap under a smooth regional
  // mask centred on the island.
  const GeographicPoint greenland{72.0, 320.0};
  const auto mask = grid.FromFunction([&](double theta, double phi) {
    return examples::Cap(theta, phi, greenland.CoLatitude(),
                         greenland.Longitude(), 0.28);
  });
  const auto& ice = state.iceThickness;
  auto iceChange = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < ice.size(); ++i) {
    iceChange[i] = -0.1 * mask[i] * ice[i];
  }

  const auto result = fingerprint.SolveForIceChange(iceChange);
  if (!result.converged) {
    std::cerr << "the sea level equation did not converge\n";
    return 1;
  }

  examples::Writer out("fingerprint.txt");
  out.Comment("elastic fingerprint of a 10% Greenland melt, ICE-7G state (SI)");
  out.Grid(grid);
  out.Field("ice_thickness_m", ice);
  out.Field("ocean_function", fingerprint.OceanFunction());
  out.Field("ice_change_m", iceChange);
  out.Field("sea_level_change_m", result.seaLevelChange);
  out.Field("displacement_m", result.displacement);
  out.Field("potential_m2s2", result.potential);
  out.Scalar("omega_x_rad_s", result.angularVelocity[0]);
  out.Scalar("omega_y_rad_s", result.angularVelocity[1]);
  out.Scalar("omega_z_rad_s", result.angularVelocity[2]);

  std::cout << "wrote fingerprint.txt (" << result.iterations
            << " SLE iterations)\n";
  return 0;
}
