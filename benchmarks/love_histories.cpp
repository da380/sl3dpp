// The V1 exchange writer: Maxwell load Love-number histories of a deck
// under a Heaviside unit surface load, as the cross-code exchange file the
// benchmark harness in AdGIA/benchmarks compares (viscoelastic/love
// there). One table: t (in units of the shortest Maxwell time) and, per
// degree, the conventional load numbers hp<l> lp<l> kp<l> (Farrell's,
// dimensionless) at that time. The load of every degree switches on at
// t = 0; the evolution is the exponential trapezoid with, between output
// times, equal steps of at least `steps` and no longer than `dtmax` tau_min
// -- the same rule as AdGIA's viscoelastic_love, and the same step-weight
// algebra as pyslfp's, so cross-code differences isolate the spatial
// discretisation.
//
// Usage:
//   love_histories <deck> <lmax> <taus> <out.txt> [steps] [dtmax]
//
// <taus> is a comma list of Maxwell times in seconds, one per layer from
// the centre out, `inf` keeping a layer elastic (a fluid layer must be
// `inf`). The deck is the elastic cross-code format; the mesh the uniform
// lmax rule. Output times are the love family's 13 geometric times
// 0.011 * 2^(5k/6) tau_min, k = 0..12.

#include <charconv>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Viscoelastic.hpp"

#ifndef SL3DPP_GIT_COMMIT
#define SL3DPP_GIT_COMMIT "unknown"
#endif

int main(int argc, char** argv) {
  if (argc < 5 || argc > 7) {
    std::cerr << "usage: love_histories <deck> <lmax> <taus> <out.txt> "
                 "[steps] [dtmax]\n";
    return 1;
  }
  const auto deckPath = std::string(argv[1]);
  const auto lmax = std::stoi(argv[2]);
  const auto tauList = std::string(argv[3]);
  const auto outPath = std::string(argv[4]);
  const auto minSteps = argc > 5 ? std::stoi(argv[5]) : 4;
  const auto dtMaxTau = argc > 6 ? std::stod(argv[6]) : 0.25;

  const auto deck =
      sl3dpp::Deck::Read(
          deckPath, sl3dpp::DeckFormat(
                        "elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                        0))
          .value();
  const auto model = sl3dpp::EarthModel::FromDeck(deck).value();
  const auto mesh = sl3dpp::RadialMesh::Create(model, {.lmax = lmax}).value();
  const auto material = sl3dpp::Material::Create(model, mesh).value();
  const auto& scales = model.ModelScales();

  // The Maxwell times, seconds, one per layer; `inf` elastic.
  auto taus = std::vector<double>{};
  {
    std::istringstream list(tauList);
    for (std::string item; std::getline(list, item, ',');) {
      taus.push_back(item == "inf"
                         ? std::numeric_limits<double>::infinity()
                         : std::stod(item));
    }
  }
  if (taus.size() != model.LayerCount()) {
    std::cerr << std::format("{} taus for {} layers\n", taus.size(),
                             model.LayerCount());
    return 1;
  }

  // eta = tau mu per layer, non-dimensional; mu from the deck's first
  // knot of the layer (the ladder's layers are homogeneous).
  const auto timeScale = scales.Factor(sl3dpp::Dimensions{.time = 1});
  const auto pressure =
      scales.Factor(sl3dpp::Dimensions{.mass = 1, .length = -1, .time = -2});
  auto viscosity = std::map<std::size_t, sl3dpp::RadialValue>{};
  auto tauMin = std::numeric_limits<double>::infinity();
  const auto layers = deck.Layers();
  const auto rho = deck.Column("rho");
  const auto vsv = deck.Column("vsv");
  for (auto i = std::size_t{0}; i < taus.size(); ++i) {
    if (!std::isfinite(taus[i])) continue;
    if (model.IsFluid(i)) {
      std::cerr << std::format("layer {} is fluid: its tau must be inf\n", i);
      return 1;
    }
    const auto k = layers[i].first;
    const auto muSi = rho[k] * vsv[k] * vsv[k];
    viscosity.emplace(i, (taus[i] / timeScale) * (muSi / pressure));
    tauMin = std::min(tauMin, taus[i] / timeScale);
  }
  if (!std::isfinite(tauMin)) {
    std::cerr << "every layer is elastic: nothing relaxes\n";
    return 1;
  }
  const auto rheology = sl3dpp::Rheology::Maxwell(material, viscosity).value();
  const sl3dpp::ViscoelasticOperator op(rheology, lmax);

  // The love family's 13 geometric output times, in units of tau_min,
  // rounded to six decimals exactly as laplace_reference.py writes them.
  auto times = std::vector<double>{};
  for (auto k = 0; k < 13; ++k) {
    times.push_back(std::round(0.011 * std::pow(2.0, 5.0 * k / 6.0) * 1e6) /
                    1e6);
  }

  const auto load = [](int, double) {
    return std::optional<std::vector<double>>(std::vector<double>{1.0});
  };
  const auto G = material.G();
  const auto a = material.Radius();
  const auto g = material.SurfaceGravity();
  const auto fourPiGa = 4.0 * std::numbers::pi * G * a;

  const auto today = std::chrono::floor<std::chrono::days>(
      std::chrono::system_clock::now());
  std::ofstream out(outPath);
  if (!out) {
    std::cerr << std::format("cannot write {}\n", outPath);
    return 1;
  }
  out << "# V1 Maxwell load Love-number histories from sl3dpp (Heaviside "
         "unit load)\n";
  out << std::format("# code sl3dpp commit {} date {:%F}\n", SL3DPP_GIT_COMMIT,
                     today);
  out << std::format(
      "# deck {} lmax {} taus {} s; exponential trapezoid, >= {} steps per "
      "interval, dt <= {} tau_min; t in units of tau_min; dimensionless "
      "conventional load numbers\n",
      deckPath, lmax, tauList, minSteps, dtMaxTau);
  out << std::format("radius {}\n", a * scales.Length());
  out << "columns t";
  for (auto l = 0; l <= lmax; ++l) {
    out << std::format(" hp{} lp{} kp{}", l, l, l);
  }
  out << "\n";

  auto state = op.ZeroState();
  auto previous = 0.0;
  for (const auto tTau : times) {
    const auto t = tTau * tauMin;
    const auto interval = t - previous;
    const auto n = std::max(
        minSteps, static_cast<int>(std::ceil(interval / (dtMaxTau * tauMin))));
    const auto dt = interval / n;
    auto solutions = std::map<int, std::vector<double>>{};
    for (auto s = 0; s < n; ++s) {
      std::tie(state, solutions) =
          op.Step(state, dt, sl3dpp::Scheme::ExponentialTrapezoid, load);
    }
    previous = t;
    const auto U = op.Surface(solutions, sl3dpp::Component::U);
    const auto V = op.Surface(solutions, sl3dpp::Component::V);
    const auto P = op.Surface(solutions, sl3dpp::Component::Phi);
    out << std::format("{}", tTau);
    for (auto l = 0; l <= lmax; ++l) {
      const auto f = (2.0 * l + 1.0) / fourPiGa;
      out << std::format(" {} {} {}", f * g * U.at(l)[0], f * g * V.at(l)[0],
                         -f * P.at(l)[0] - 1.0);
    }
    out << "\n";
  }
  std::cout << outPath << "\n";
  return 0;
}
