#include "sl3dpp/Deck.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <planetmodel/deck.hpp>
#include <utility>

namespace sl3dpp {

namespace {

/// sl3dpp's format handed to planetmodel's reader: the same columns and
/// header count, the header kept verbatim — sl3dpp formats state no
/// header contract, so planetmodel's own header checks are not wired in.
planetmodel::DeckFormat Translated(const DeckFormat& format) {
  return {.name = format.name,
          .columns = format.columns,
          .header_lines = format.header_lines};
}

}  // namespace

DeckFormat MineosFormat() {
  return DeckFormat::ByWidth("mineos",
                             {{8, planetmodel::MINEOS_COLUMNS},
                              {5, planetmodel::MINEOS_ISOTROPIC_COLUMNS}},
                             3);
}

DeckFormat GiaFormat() {
  return DeckFormat("gia", planetmodel::GIA_COLUMNS, 2);
}

std::expected<Deck, DeckError> Deck::Create(
    std::vector<double> radius,
    std::vector<std::pair<std::string, std::vector<double>>> columns,
    std::vector<std::string> header) {
  Deck deck;
  for (const auto& [name, values] : columns) {
    if (std::ranges::find(deck.names_, name) != deck.names_.end()) {
      return std::unexpected(
          DeckError{std::format("column '{}' appears twice", name)});
    }
    deck.names_.push_back(name);
  }
  try {
    // planetmodel's Deck is the one validator: the radii,
    // the column sizes, the layering and its refusals.
    const auto read = planetmodel::Deck(std::move(radius), std::move(columns),
                                        std::move(header));
    deck.radius_ = read.Radius();
    deck.header_ = read.Header();
    deck.layers_ = read.Layers();
    deck.boundaries_ = read.Boundaries();
    for (const auto& name : deck.names_) {
      deck.values_.push_back(read.Column(name));
    }
    return deck;
  } catch (const planetmodel::Error& error) {
    return std::unexpected(DeckError{error.what()});
  }
}

std::expected<Deck, DeckError> Deck::Read(const std::filesystem::path& path,
                                          const DeckFormat& format) {
  try {
    // planetmodel's reader is the one parser: deck.py's
    // own semantics — blank lines and `#` comments skipped, `nan` and a
    // leading `+` read as Python reads them.
    const auto read = planetmodel::ReadDeck(path, Translated(format));
    auto columns = std::vector<std::pair<std::string, std::vector<double>>>{};
    for (const auto& name : read.Names()) {
      columns.emplace_back(name, read.Column(name));
    }
    return Create(read.Radius(), std::move(columns), read.Header());
  } catch (const planetmodel::Error& error) {
    return std::unexpected(DeckError{error.what()});
  }
}

std::expected<Deck, DeckError> Deck::ReadLegacySl3d(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    return std::unexpected(
        DeckError{std::format("cannot open {}", path.string())});
  }
  auto title = std::string{};
  std::getline(file, title);
  auto nknot = std::size_t{0};
  auto ntau = 0;
  if (!(file >> nknot >> ntau)) {
    return std::unexpected(DeckError{std::format(
        "{}: the second line must hold `nknot ntau`", path.string())});
  }
  if (ntau != 1) {
    return std::unexpected(DeckError{
        std::format("{}: only ntau = 1 legacy files are supported, got {}",
                    path.string(), ntau)});
  }

  std::vector<double> radius(nknot), rho(nknot), kappa(nknot), mu(nknot),
      viscosity(nknot);
  for (auto i = std::size_t{0}; i < nknot; ++i) {
    if (!(file >> radius[i] >> rho[i] >> kappa[i] >> mu[i] >> viscosity[i])) {
      return std::unexpected(DeckError{std::format(
          "{}: expected {} knot rows `r rho kappa mu viscosity`, failed "
          "at row {}",
          path.string(), nknot, i + 1)});
    }
    kappa[i] *= 1e9;  // GPa -> Pa
    mu[i] *= 1e9;
    // log10(Pa s), 0 meaning no viscosity.
    viscosity[i] = viscosity[i] == 0.0
                       ? std::numeric_limits<double>::quiet_NaN()
                       : std::pow(10.0, viscosity[i]);
  }

  auto deck = Create(std::move(radius),
                     {{"rho", std::move(rho)},
                      {"kappa", std::move(kappa)},
                      {"mu", std::move(mu)},
                      {"viscosity", std::move(viscosity)}},
                     {std::move(title)});
  if (!deck) {
    return std::unexpected(
        DeckError{std::format("{}: {}", path.string(), deck.error().message)});
  }

  // NaN on part of a layer is meaningless downstream; refuse it here, where
  // the 0-means-none convention was translated.
  auto values = deck->Column("viscosity");
  for (auto [first, last] : deck->Layers()) {
    const auto absent = std::isnan(values[first]);
    for (std::size_t i = first; i < last; ++i) {
      if (std::isnan(values[i]) != absent) {
        return std::unexpected(DeckError{std::format(
            "{}: the layer starting at r = {:g} mixes zero and non-zero "
            "viscosity",
            path.string(), deck->Radius()[first])});
      }
    }
  }
  return deck;
}

bool Deck::Has(const std::string& name) const {
  return std::ranges::find(names_, name) != names_.end();
}

std::span<const double> Deck::Column(const std::string& name) const {
  auto found = std::ranges::find(names_, name);
  if (found == names_.end()) {
    throw std::out_of_range(std::format("the deck has no column '{}'", name));
  }
  return values_[static_cast<std::size_t>(found - names_.begin())];
}

}  // namespace sl3dpp
