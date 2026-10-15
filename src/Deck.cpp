#include "sl3dpp/Deck.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <sstream>

namespace sl3dpp {

namespace {

/// The whitespace-separated doubles of a line ("nan" included), or an error
/// naming the offending token.
std::expected<std::vector<double>, DeckError> ParseRow(
    const std::string& line) {
  std::vector<double> row;
  std::istringstream stream(line);
  std::string token;
  while (stream >> token) {
    try {
      std::size_t used = 0;
      double value = std::stod(token, &used);
      if (used != token.size()) throw std::invalid_argument(token);
      row.push_back(value);
    } catch (const std::exception&) {
      return std::unexpected(
          DeckError{std::format("not a number: {:.40}", token)});
    }
  }
  return row;
}

}  // namespace

DeckFormat MineosFormat() {
  return DeckFormat::ByWidth(
      "mineos",
      {{8, {"rho", "vpv", "vsv", "qkappa", "qmu", "vph", "vsh", "eta"}},
       {5, {"rho", "vp", "vs", "qkappa", "qmu"}}},
      3);
}

DeckFormat GiaFormat() {
  return DeckFormat("gia",
                    {"rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity"}, 2);
}

std::expected<Deck, DeckError> Deck::Create(
    std::vector<double> radius,
    std::vector<std::pair<std::string, std::vector<double>>> columns,
    std::vector<std::string> header) {
  if (radius.size() < 2) {
    return std::unexpected(DeckError{"a deck needs at least two knot radii"});
  }
  for (std::size_t i = 1; i < radius.size(); ++i) {
    if (radius[i] < radius[i - 1]) {
      return std::unexpected(DeckError{std::format(
          "knot radii must be non-decreasing; knot {} (r = {:g}) is below "
          "knot {} (r = {:g})",
          i, radius[i], i - 1, radius[i - 1])});
    }
  }

  Deck deck;
  for (auto& [name, values] : columns) {
    if (values.size() != radius.size()) {
      return std::unexpected(
          DeckError{std::format("column '{}' has {} values, the radii {}", name,
                                values.size(), radius.size())});
    }
    if (std::ranges::find(deck.names_, name) != deck.names_.end()) {
      return std::unexpected(
          DeckError{std::format("column '{}' appears twice", name)});
    }
    deck.names_.push_back(name);
    deck.values_.push_back(std::move(values));
  }

  // The layering: one layer per span between repeated radii.
  std::size_t start = 0;
  deck.boundaries_.push_back(radius.front());
  for (std::size_t i = 1; i < radius.size(); ++i) {
    if (radius[i] == radius[i - 1]) {
      if (i + 1 < radius.size() && radius[i + 1] == radius[i]) {
        return std::unexpected(DeckError{std::format(
            "the radius {:g} repeats more than twice: a layer of no "
            "thickness",
            radius[i])});
      }
      deck.layers_.emplace_back(start, i);
      deck.boundaries_.push_back(radius[i]);
      start = i;
    }
  }
  deck.layers_.emplace_back(start, radius.size());
  deck.boundaries_.push_back(radius.back());

  for (auto [first, last] : deck.layers_) {
    if (last - first < 2) {
      return std::unexpected(DeckError{std::format(
          "the layer starting at r = {:g} has one knot; every layer needs "
          "at least two",
          radius[first])});
    }
  }

  deck.radius_ = std::move(radius);
  deck.header_ = std::move(header);
  return deck;
}

std::expected<Deck, DeckError> Deck::Read(const std::filesystem::path& path,
                                          const DeckFormat& format) {
  std::ifstream file(path);
  if (!file) {
    return std::unexpected(
        DeckError{std::format("cannot open {}", path.string())});
  }
  std::vector<std::string> lines;
  for (std::string line; std::getline(file, line);) lines.push_back(line);

  if (lines.size() < format.header_lines) {
    return std::unexpected(DeckError{std::format(
        "{}: {} lines, fewer than the {} header lines of a {} deck",
        path.string(), lines.size(), format.header_lines, format.name)});
  }
  std::vector<std::string> header(lines.begin(),
                                  lines.begin() + format.header_lines);

  std::vector<double> radius;
  std::vector<std::vector<double>> table;  // one vector per column
  const std::vector<std::string>* names = nullptr;
  for (std::size_t i = format.header_lines; i < lines.size(); ++i) {
    auto row = ParseRow(lines[i]);
    if (!row) {
      return std::unexpected(DeckError{
          std::format("{}:{}: {}", path.string(), i + 1, row.error().message)});
    }
    if (row->empty()) continue;  // blank line
    const auto width = row->size() - 1;
    if (names == nullptr) {
      auto found = format.columns.find(width);
      if (found == format.columns.end()) {
        return std::unexpected(DeckError{std::format(
            "{}:{}: a {} deck does not admit {} columns after the radius",
            path.string(), i + 1, format.name, width)});
      }
      names = &found->second;
      table.resize(width);
    } else if (width != names->size()) {
      return std::unexpected(DeckError{std::format(
          "{}:{}: {} columns after the radius, previous rows had {}",
          path.string(), i + 1, width, names->size())});
    }
    radius.push_back((*row)[0]);
    for (std::size_t j = 0; j < width; ++j) table[j].push_back((*row)[j + 1]);
  }
  if (names == nullptr) {
    return std::unexpected(
        DeckError{std::format("{}: no knots after the header", path.string())});
  }

  std::vector<std::pair<std::string, std::vector<double>>> columns;
  for (std::size_t j = 0; j < names->size(); ++j) {
    columns.emplace_back((*names)[j], std::move(table[j]));
  }
  auto deck = Create(std::move(radius), std::move(columns), std::move(header));
  if (!deck) {
    return std::unexpected(
        DeckError{std::format("{}: {}", path.string(), deck.error().message)});
  }
  return deck;
}

std::expected<Deck, DeckError> Deck::ReadLegacySl3d(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    return std::unexpected(
        DeckError{std::format("cannot open {}", path.string())});
  }
  std::string title;
  std::getline(file, title);
  std::size_t nknot = 0;
  int ntau = 0;
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
  for (std::size_t i = 0; i < nknot; ++i) {
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
    const bool absent = std::isnan(values[first]);
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
