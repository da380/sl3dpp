#pragma once

/**
 * @file examples/Common.hpp
 * @brief Shared scaffolding for the example drivers: the command line, the
 * PREM-deck model and mantle Maxwell rheology, the Earth's state from a
 * surface file, simple text writers for the plotting scripts, and smooth
 * analytic shapes.
 *
 * Examples run FROM THE BUILD TREE: the deck is copied next to the
 * executables at configure time, and every output lands in the working
 * directory (never the source tree). File format, consumed by
 * python/plotutil.py: '# comment' lines, 'grid nlat nlon' followed by the
 * colatitude and longitude rows, then named blocks — 'field <name>' with
 * nlat rows of nlon values, 'series <name> <n>' with n rows, or
 * 'scalar <name> <value>'.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

namespace examples {

constexpr auto kYear = 3.15576e7;  // s

/// The PREM-like deck copied beside the executables, its mesh, material
/// and a mantle Maxwell rheology (viscous between 3480 and 6151 km).
struct Earth {
  sl3dpp::EarthModel model;
  sl3dpp::Material material;
  sl3dpp::Rheology rheology;

  static Earth Make(int lmax, double viscositySI) {
    auto deck = sl3dpp::Deck::Read(
                    "prem8.elastic.deck",
                    sl3dpp::DeckFormat(
                        "elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                        0))
                    .value();
    auto model = sl3dpp::EarthModel::FromDeck(deck).value();
    auto mesh = sl3dpp::RadialMesh::Create(model, {.lmax = lmax}).value();
    auto material = sl3dpp::Material::Create(model, mesh).value();
    const auto& scales = model.ModelScales();
    auto viscosity = std::map<std::size_t, sl3dpp::RadialValue>{};
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(
            i, viscositySI / scales.Factor(sl3dpp::ViscosityDimensions));
      }
    }
    auto rheology = sl3dpp::Rheology::Maxwell(material, viscosity).value();
    return Earth{std::move(model), std::move(material), std::move(rheology)};
  }
};

/// What every example takes on its command line: the degree (each has
/// its own default, sized for a laptop; CI passes small ones) and the
/// surface file of the Earth's state (the synthetic Earth the build
/// writes, or an ICE-7G file from python/get_ice7g.py).
struct Options {
  int lmax = 0;
  std::filesystem::path state = "synthetic_earth.nc";
};

/// Reads `[--lmax N] [--state FILE]`; prints the usage and exits on
/// --help or anything else.
inline Options ParseOptions(int argc, char** argv, int defaultLmax) {
  auto options = Options{.lmax = defaultLmax};
  const auto usage = [&](int status) {
    std::cerr << "usage: " << argv[0] << " [--lmax N] [--state FILE]\n"
              << "  --lmax N      the degree (default " << defaultLmax << ")\n"
              << "  --state FILE  the Earth's state: a planetmodel surface "
                 "file holding\n"
              << "                ice_thickness and topography (default "
                 "synthetic_earth.nc)\n";
    std::exit(status);
  };
  for (auto k = 1; k < argc; ++k) {
    const auto arg = std::string_view{argv[k]};
    if (arg == "--help" || arg == "-h") usage(0);
    if (k + 1 >= argc) usage(2);
    if (arg == "--lmax") {
      options.lmax = std::stoi(argv[++k]);
    } else if (arg == "--state") {
      options.state = argv[++k];
    } else {
      usage(2);
    }
  }
  if (options.lmax < 2) usage(2);
  return options;
}

/// The Earth's state at a date: ice thickness, topography and the sea
/// level built from them (SeaLevelFromTopography), SI metres on the grid.
struct EarthState {
  std::vector<double> iceThickness;
  std::vector<double> topography;
  std::vector<double> seaLevel;
};

/// Reads the state from a surface file holding `ice_thickness` and
/// `topography` (the synthetic Earth, or pyslfp's export of ICE-xG) at a
/// date in kyr before present, resampled onto the grid.
inline EarthState LoadEarthState(const sl3dpp::SurfaceGrid& grid,
                                 const std::filesystem::path& path,
                                 double dateKyrBp = 0.0) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error(std::format(
        "{} not found: run the examples from build/examples, where the "
        "build writes synthetic_earth.nc (python/get_ice7g.py writes "
        "ice7g.nc)",
        path.string()));
  }
  const auto history = sl3dpp::io::FieldHistory::Read(path, grid);
  if (!history) throw std::runtime_error(history.error().message);
  const auto time = -dateKyrBp * 1.0e3 * kYear;
  auto out = EarthState{};
  out.iceThickness = history->Values("ice_thickness", time);
  out.topography = history->Values("topography", time);
  out.seaLevel =
      sl3dpp::SeaLevelFromTopography(out.topography, out.iceThickness);
  return out;
}

/// A smooth Gaussian cap in great-circle distance, the examples' melt
/// masks and measurement sites.
inline double Cap(double theta, double phi, double colat0, double lon0,
                  double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * (d / width) * (d / width));
}

/// The text writer the plotting scripts read.
class Writer {
 public:
  explicit Writer(const std::string& path) : out_(path) {
    if (!out_.is_open()) {
      throw std::runtime_error(std::format("cannot open {}", path));
    }
    out_.precision(10);
  }

  void Comment(const std::string& text) { out_ << "# " << text << "\n"; }

  /// The grid header every map file opens with.
  void Grid(const sl3dpp::SurfaceGrid& grid) {
    nlat_ = grid.NumberOfCoLatitudes();
    nlon_ = grid.NumberOfLongitudes();
    out_ << "grid " << nlat_ << " " << nlon_ << "\n";
    for (auto theta : grid.CoLatitudes()) out_ << theta << " ";
    out_ << "\n";
    for (auto phi : grid.Longitudes()) out_ << phi << " ";
    out_ << "\n";
  }

  /// A field over the grid points (needs the Grid header), with an
  /// optional multiplicative unit conversion.
  void Field(const std::string& name, std::span<const double> values,
             double factor = 1.0) {
    out_ << "field " << name << "\n";
    for (auto i = 0; i < nlat_; ++i) {
      for (auto j = 0; j < nlon_; ++j) {
        out_ << factor * values[static_cast<std::size_t>(i) * nlon_ + j]
             << (j + 1 == nlon_ ? "" : " ");
      }
      out_ << "\n";
    }
  }

  /// Named columns, one row per entry of the first column.
  void Series(const std::string& name,
              const std::vector<std::string>& columns,
              const std::vector<std::vector<double>>& rows) {
    out_ << "series " << name << " " << rows.size() << "\n# ";
    for (const auto& c : columns) out_ << c << " ";
    out_ << "\n";
    for (const auto& row : rows) {
      for (auto j = std::size_t{0}; j < row.size(); ++j) {
        out_ << row[j] << (j + 1 == row.size() ? "" : " ");
      }
      out_ << "\n";
    }
  }

  void Scalar(const std::string& name, double value) {
    out_ << "scalar " << name << " " << value << "\n";
  }

 private:
  std::ofstream out_;
  int nlat_ = 0, nlon_ = 0;
};

}  // namespace examples
