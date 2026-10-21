// The E1 exchange writer: traction numbers of a deck, as the cross-code
// exchange file the benchmark harness in AdGIA/benchmarks compares
// (elastic/nongrav there; `common/exchange.py` states the format). One
// table of l, h_u, l_u, h_v, l_v in SI (m per Pa), NaN at an excluded
// degree; values are printed shortest-round-trip, so the file carries the
// computation exactly.
//
// Usage:
//   traction_numbers <deck> <lmax> <out.txt> [nongrav|grav]
//
// The deck is the elastic cross-code format (r rho vpv vsv vph vsh eta,
// SI, no header), the mesh the uniform lmax rule shared with pyslfp, and
// the problem non-gravitating unless `grav` asks for the full one under
// the same forcing.

#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
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
  if (argc < 4 || argc > 5) {
    std::cerr << "usage: traction_numbers <deck> <lmax> <out.txt> "
                 "[nongrav|grav]\n";
    return 1;
  }
  const auto deckPath = std::string(argv[1]);
  const auto lmax = std::stoi(argv[2]);
  const auto outPath = std::string(argv[3]);
  const auto mode = argc == 5 ? std::string(argv[4]) : std::string("nongrav");
  if (mode != "nongrav" && mode != "grav") {
    std::cerr << std::format("the mode is nongrav or grav, not {}\n", mode);
    return 1;
  }
  const auto gravitating = mode == "grav";

  const auto deck =
      sl3dpp::Deck::Read(
          deckPath, sl3dpp::DeckFormat(
                        "elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                        0))
          .value();
  const auto model = sl3dpp::EarthModel::FromDeck(deck).value();
  const auto mesh = sl3dpp::RadialMesh::Create(model, {.lmax = lmax}).value();
  const auto material = sl3dpp::Material::Create(model, mesh).value();
  const auto tn =
      sl3dpp::ComputeTractionNumbers(material, lmax, gravitating);

  const auto today = std::chrono::floor<std::chrono::days>(
      std::chrono::system_clock::now());
  std::ofstream out(outPath);
  if (!out) {
    std::cerr << std::format("cannot write {}\n", outPath);
    return 1;
  }
  out << std::format("# E1 traction numbers from sl3dpp ({})\n",
                     gravitating ? "gravitating" : "non-gravitating");
  out << std::format("# code sl3dpp commit {} date {:%F}\n", SL3DPP_GIT_COMMIT,
                     today);
  out << std::format(
      "# deck {} lmax {} uniform lmax mesh rule; SI (m per Pa)\n", deckPath,
      lmax);
  out << std::format("radius {}\n", tn.radius);
  out << std::format("gravitating {}\n", gravitating ? 1 : 0);
  out << "columns l h_u l_u h_v l_v\n";
  for (auto l = 0; l <= lmax; ++l) {
    out << std::format("{} {} {} {} {}\n", l, tn.h_u[l], tn.l_u[l], tn.h_v[l],
                       tn.l_v[l]);
  }
  std::cout << std::format("{}  reciprocity {}\n", outPath,
                           tn.ReciprocityResidual());
  return 0;
}
