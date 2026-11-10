/**
 * @file MakeSyntheticModuli.cpp
 * @brief Writes the examples' synthetic 3D elastic Earth: a planetmodel
 * model file holding the PREM-like deck's density and its Voigt-averaged
 * isotropic moduli with a lateral perturbation of the upper mantle, in
 * the form planetmodel's conversion of a tomography model writes (the
 * `mu` and `kappa` fields of its vocabulary), so that the 3D static
 * example and the `sl3d response` config run with no data to fetch.
 *
 * The perturbation is a toy: a stiff "craton" under the Canadian shield
 * (+8% shear), a soft "ridge" under the eastern Pacific (-6%), and a
 * weak degree-2 background, all in a smooth radial envelope peaking at
 * 250 km depth and vanishing below 660 km; the bulk modulus follows at
 * four tenths of the amount. The lateral means are not zero, so the
 * reader's split against the lateral mean (io::ModelModuli) is the one
 * that gives these back less their means.
 *
 * The build runs it, writing synthetic_moduli.nc beside the examples:
 *
 *   MakeSyntheticModuli [output.nc]
 */
#include <cmath>
#include <iostream>
#include <numbers>
#include <string>

#include "Common.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/ModelModuli.hpp"

namespace {

constexpr auto kLmax = 32;
constexpr auto kSurface = 6371.0e3;

/// A Gaussian cap of unit height at (colat0, lon0) with width w, radians.
double Cap(double theta, double phi, double colat0, double lon0, double w) {
  const auto c = std::cos(theta) * std::cos(colat0) +
                 std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * (d / w) * (d / w));
}

/// The upper-mantle envelope of the depth d (metres): a bump at 250 km,
/// 150 km wide, cut off smoothly by 660 km.
double Envelope(double r) {
  const auto d = kSurface - r;
  if (d <= 0.0 || d >= 660.0e3) return 0.0;
  const auto bump = std::exp(-std::pow((d - 250.0e3) / 150.0e3, 2));
  const auto fade = d < 560.0e3 ? 1.0 : 0.5 * (1.0 + std::cos(std::numbers::pi * (d - 560.0e3) / 100.0e3));
  return bump * fade;
}

double Shear(double r, double theta, double phi) {
  constexpr auto deg = std::numbers::pi / 180.0;
  const auto craton = 0.08 * Cap(theta, phi, (90.0 - 58.0) * deg, -95.0 * deg, 14.0 * deg);
  const auto ridge = -0.06 * Cap(theta, phi, (90.0 + 15.0) * deg, -110.0 * deg, 18.0 * deg);
  const auto background = 0.015 * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi);
  return Envelope(r) * (craton + ridge + background);
}

double Bulk(double r, double theta, double phi) { return 0.4 * Shear(r, theta, phi); }

}  // namespace

int main(int argc, char** argv) {
  const auto output = std::string{argc > 1 ? argv[1] : "synthetic_moduli.nc"};
  const auto earth = examples::Earth::Make(kLmax, 1e21);
  const sl3dpp::SurfaceGrid grid(kLmax);
  const auto written = sl3dpp::io::WriteModuli(
      output, earth.model, grid, Shear, Bulk,
      {.name = "synthetic 3D elastic Earth", .elementMetres = 100.0e3});
  if (!written) {
    std::cerr << "MakeSyntheticModuli: " << written.error().message << "\n";
    return 1;
  }
  std::cout << "wrote " << output << "\n";
  return 0;
}
