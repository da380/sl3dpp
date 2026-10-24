// Uses the installed library through each of its dependencies: the deck is
// read by planetmodel, the Love numbers are solved with LAPACK, and the
// surface grid instantiates GSHTrans (and FFTW) in this translation unit.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <planetmodel/version.hpp>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Version.hpp"

int main(int argc, char** argv) {
  using namespace sl3dpp;
  if (argc != 2) {
    std::cerr << "usage: PackageTest <elastic deck>\n";
    return 2;
  }
  constexpr auto lmax = 8;
  const auto format =
      DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0);
  const auto deck = Deck::Read(argv[1], format);
  if (!deck) {
    std::cerr << deck.error().message << '\n';
    return 1;
  }
  const auto model = EarthModel::FromDeck(*deck).value();
  const auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
  const auto material = Material::Create(model, mesh).value();
  const auto love = ComputeLoveNumbers(material, lmax);

  const auto grid = SurfaceGrid(lmax);

  const auto two = std::ranges::find(love.degrees, 2) - love.degrees.begin();
  const auto k2 = love.k_u.at(static_cast<std::size_t>(two));
  std::cout << std::format(
      "sl3dpp {} (planetmodel {}, here {}): k_u at degree 2 = {:.6f}, grid of "
      "{} points\n",
      Version(), PlanetModelVersion(), planetmodel::Version(), k2,
      grid.FieldSize());
  return std::isfinite(k2) && k2 != 0.0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
