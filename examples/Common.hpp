#pragma once

/**
 * @file examples/Common.hpp
 * @brief Shared scaffolding for the example drivers: the PREM-deck model
 * and mantle Maxwell rheology, simple text writers for the plotting
 * scripts, and smooth analytic shapes.
 *
 * Examples run FROM THE BUILD TREE: the deck is copied next to the
 * executables at configure time, and every output lands in the working
 * directory (never the source tree). File format, consumed by
 * plots/plotutil.py: '# comment' lines, 'grid nlat nlon' followed by the
 * colatitude and longitude rows, then named blocks — 'field <name>' with
 * nlat rows of nlon values, 'series <name> <n>' with n rows, or
 * 'scalar <name> <value>'.
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/ShConventions.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

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

/// The present-day Earth state — ICE-7G ice thickness and sea level
/// (minus the topography over land), SI metres — synthesised on the
/// grid from the coefficient file the build extracted from pyslfp
/// (earth_state_l256.txt beside the executables), truncated to the
/// grid's lmax.
struct EarthState {
  std::vector<double> iceThickness;
  std::vector<double> seaLevel;
};

inline EarthState LoadEarthState(const sl3dpp::SurfaceGrid& grid) {
  std::ifstream in("earth_state_l256.txt");
  if (!in.is_open()) {
    throw std::runtime_error(
        "earth_state_l256.txt not found: configure with a pyslfp checkout "
        "(SL3DPP_PYSLFP_DIR) so the build extracts it, and run from "
        "build/examples");
  }
  const auto lmax = grid.LMax();
  auto fileLmax = 0;
  auto iceLm = grid.ZeroCoefficients();
  auto slLm = grid.ZeroCoefficients();
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (line.starts_with("lmax")) {
      auto head = std::string{};
      row >> head >> fileLmax;
      if (fileLmax < lmax) {
        throw std::runtime_error(
            std::format("the Earth state reaches degree {}, the example "
                        "wants {}",
                        fileLmax, lmax));
      }
      continue;
    }
    auto l = 0, m = 0;
    auto ia = 0.0, ib = 0.0, sa = 0.0, sb = 0.0;
    row >> l >> m >> ia >> ib >> sa >> sb;
    if (l > lmax) break;  // rows ascend in degree
    const auto i = sl3dpp::SurfaceGrid::Index(l, m);
    iceLm[i] = sl3dpp::ComplexFromRealPair({ia, ib}, m);
    slLm[i] = sl3dpp::ComplexFromRealPair({sa, sb}, m);
  }
  EarthState out{grid.Synthesise(iceLm), grid.Synthesise(slLm)};
  // Band-limitation rings slightly negative over ice-free ground.
  for (auto& v : out.iceThickness) v = std::max(v, 0.0);
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
