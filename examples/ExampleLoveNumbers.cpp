/**
 * @file ExampleLoveNumbers.cpp
 * @brief Elastic Love numbers of a radial Earth model: the per-degree
 * spheroidal systems solved for a unit surface load and a unit tidal
 * potential, the starting point of every other calculation.
 *
 * The library returns the generalised numbers of Al-Attar et al. (2024),
 * SI per unit surface density, in four forcing channels (LoveNumbers.hpp).
 * This example converts them to the conventional dimensionless ones
 * (LoveNumbers::Conventional): Farrell's load numbers h', l', k' (referred
 * to the load's direct potential, geodetic sign convention) and the tidal
 * numbers h, l, k,
 *
 *   h' = (2l+1) g / (4 pi G a) h,  l' likewise,
 *   k' = -(2l+1) / (4 pi G a) k - 1,
 *   hT = -g h_t,  lT = -g l_t,  kT = k_t,
 *
 * on the PREM-like deck. At large degree h', l l' and l k' tend to
 * constants set by the elastic structure near the surface (Farrell 1972).
 *
 * Outputs: the table, one row per degree, and the reciprocity residual
 * (a structural check of the assembly's symmetry).
 *
 *   ExampleLoveNumbers [--lmax N] [--state FILE]  (the state is unused)
 *
 * Run from the build tree; plot with python/plot_love_numbers.py.
 */

#include <format>
#include <iostream>

#include "Common.hpp"
#include "sl3dpp/LoveNumbers.hpp"

using namespace sl3dpp;

int main(int argc, char** argv) {
  const auto options = examples::ParseOptions(argc, argv, 256);
  const auto lmax = options.lmax;
  examples::Say(std::format(
      "ExampleLoveNumbers: the per-degree systems of the deck to degree {}",
      lmax));
  const auto earth = examples::Earth::Make(lmax, 1e21);  // viscosity unused
  const auto love = ComputeLoveNumbers(earth.material, lmax);

  // The conventional numbers: the table carries g, a and G for them.
  const auto conventional = love.Conventional();
  const auto a = love.radius;
  const auto g = love.surface_gravity;
  auto rows = std::vector<std::vector<double>>{};
  for (auto i = std::size_t{0}; i < conventional.degrees.size(); ++i) {
    rows.push_back({static_cast<double>(conventional.degrees[i]),
                    conventional.h_load[i], conventional.l_load[i],
                    conventional.k_load[i], conventional.h_tide[i],
                    conventional.l_tide[i], conventional.k_tide[i]});
  }

  examples::Writer out("love_numbers.txt");
  out.Comment(
      "elastic Love numbers of the PREM-like deck: conventional load "
      "(h' l' k', Farrell) and tidal (h l k), dimensionless");
  out.Scalar("radius_m", a);
  out.Scalar("surface_gravity_m_s2", g);
  out.Scalar("reciprocity_residual", love.ReciprocityResidual());
  out.Series("love",
             {"l", "h_load", "l_load", "k_load", "h_tide", "l_tide", "k_tide"},
             rows);

  const auto& two = rows.at(2);
  std::cout << "wrote love_numbers.txt (degrees 0 to " << lmax
            << "; degree 2: h' = " << two[1] << ", k' = " << two[3]
            << ", k = " << two[6] << "; reciprocity "
            << love.ReciprocityResidual() << ")\n";
  return 0;
}
