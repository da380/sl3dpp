// The E2 exchange writer: conventional Love numbers of a deck, as the
// cross-code exchange file the benchmark harness in AdGIA/benchmarks
// compares (elastic/grav there). One table of l, the conventional load
// numbers h' l' k' (Farrell's dimensionless numbers, referred to the
// load's direct potential in the geodetic sign convention) and the
// geodetic tidal numbers h l k (NaN below degree 2), from the generalised
// numbers by
//
//   h' = (2l+1) g / (4 pi G a) h,  l' likewise,
//   k' = -(2l+1) / (4 pi G a) k - 1,
//   hT = -g h_t,  lT = -g l_t,  kT = k_t,
//
// all in SI before the conversion; the results are dimensionless. Values
// are printed shortest-round-trip.
//
// Usage:
//   love_numbers <deck> <lmax> <out.txt>
//
// The deck is the elastic cross-code format (r rho vpv vsv vph vsh eta,
// SI, no header) and the mesh the uniform lmax rule shared with pyslfp.

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"

#ifndef SL3DPP_GIT_COMMIT
#define SL3DPP_GIT_COMMIT "unknown"
#endif

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: love_numbers <deck> <lmax> <out.txt>\n";
    return 1;
  }
  const auto deckPath = std::string(argv[1]);
  const auto lmax = std::stoi(argv[2]);
  const auto outPath = std::string(argv[3]);

  const auto deck =
      sl3dpp::Deck::Read(
          deckPath, sl3dpp::DeckFormat(
                        "elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                        0))
          .value();
  const auto model = sl3dpp::EarthModel::FromDeck(deck).value();
  const auto mesh = sl3dpp::RadialMesh::Create(model, {.lmax = lmax}).value();
  const auto material = sl3dpp::Material::Create(model, mesh).value();
  const auto ln = sl3dpp::ComputeLoveNumbers(material, lmax);

  // SI constants of the body: the table is SI, G through the scales.
  const auto& scales = material.ModelScales();
  const auto G =
      material.G() *
      scales.Factor(sl3dpp::Dimensions{.mass = -1, .length = 3, .time = -2});
  const auto a = ln.radius;
  const auto g = ln.surface_gravity;
  const auto fourPiGa = 4.0 * std::numbers::pi * G * a;

  constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();
  const auto today = std::chrono::floor<std::chrono::days>(
      std::chrono::system_clock::now());
  std::ofstream out(outPath);
  if (!out) {
    std::cerr << std::format("cannot write {}\n", outPath);
    return 1;
  }
  out << "# E2 conventional Love numbers from sl3dpp (gravitating)\n";
  out << std::format("# code sl3dpp commit {} date {:%F}\n", SL3DPP_GIT_COMMIT,
                     today);
  out << std::format(
      "# deck {} lmax {} uniform lmax mesh rule; dimensionless "
      "(load primed, geodetic tidal; tidal NaN below degree 2)\n",
      deckPath, lmax);
  out << std::format("radius {}\n", a);
  out << std::format("surface_gravity {}\n", g);
  out << "gravitating 1\n";
  out << "columns l hp lp kp hT lT kT\n";
  for (auto l = 0; l <= lmax; ++l) {
    const auto f = (2.0 * l + 1.0) / fourPiGa;
    const auto hp = f * g * ln.h_u[l] + f * g * ln.h_phi[l];
    const auto lp = f * g * ln.l_u[l] + f * g * ln.l_phi[l];
    const auto kp = -f * (ln.k_u[l] + ln.k_phi[l]) - 1.0;
    const auto hT = l >= 2 ? -g * ln.h_t[l] : kNaN;
    const auto lT = l >= 2 ? -g * ln.l_t[l] : kNaN;
    const auto kT = l >= 2 ? ln.k_t[l] : kNaN;
    out << std::format("{} {} {} {} {} {} {}\n", l, hp, lp, kp, hT, lT, kT);
  }
  std::cout << std::format("{}  reciprocity {}\n", outPath,
                           ln.ReciprocityResidual());
  return 0;
}
