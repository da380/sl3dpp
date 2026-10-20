#pragma once

/**
 * @file Deck.hpp
 * @brief Tabulated radial models: the deck file format and its readers.
 *
 * A deck is a table of knots, radius first, one column per named field, in
 * which a repeated radius marks a layer boundary: the layering comes from
 * that structure, so a boundary is recorded even where the tabulated values
 * happen to agree across it. A value of NaN means the field is absent at
 * that knot; a column that is NaN throughout a layer is absent from that
 * layer (the way a fluid layer holds no viscosity), while NaN on part of a
 * layer is refused downstream.
 *
 * The parsing and validation are planetmodel's own C++ deck reader
 * (stage 11 WP4), so the semantics are `deck.py`'s exactly: blank lines
 * and `#` comments in the table are skipped, `nan` and a leading `+`
 * read as Python reads them. Only the legacy sl3d format below is
 * parsed here. A deck is numbers and a header, nothing else: values are
 * kept exactly as the file gives them (SI for planetmodel decks), and
 * non-dimensionalisation belongs to EarthModel.
 */

#include <cstddef>
#include <expected>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sl3dpp {

/// Why a deck could not be read or constructed.
struct DeckError {
  std::string message;
};

/**
 * @brief What a family of deck files looks like: the column names after the
 * radius, keyed by row width where the format admits several, and the number
 * of verbatim header lines before the table.
 */
struct DeckFormat {
  std::string name = "custom";
  std::map<std::size_t, std::vector<std::string>> columns;
  std::size_t header_lines = 0;

  /// A format with a single admissible row width.
  DeckFormat(std::string format_name, std::vector<std::string> column_names,
             std::size_t header_line_count)
      : name(std::move(format_name)), header_lines(header_line_count) {
    columns.emplace(column_names.size(), std::move(column_names));
  }

  /// A format admitting several row widths, mineos-style.
  [[nodiscard]] static DeckFormat ByWidth(
      std::string format_name,
      std::map<std::size_t, std::vector<std::string>> columns_by_width,
      std::size_t header_line_count) {
    DeckFormat format(std::move(format_name), {}, header_line_count);
    format.columns = std::move(columns_by_width);
    return format;
  }
};

/// The mineos deck format: three header lines, then
/// `r rho vpv vsv qkappa qmu vph vsh eta` (transversely isotropic) or
/// `r rho vp vs qkappa qmu` (isotropic), in SI.
[[nodiscard]] DeckFormat MineosFormat();

/// planetmodel's GIA deck format (tutorial 11): two header lines, then
/// `r rho vpv vsv vph vsh eta viscosity` in SI, viscosity NaN where a layer
/// is elastic or fluid.
[[nodiscard]] DeckFormat GiaFormat();

/**
 * @brief A read deck: knot radii, one array per named column, and the
 * verbatim header lines. Radii are non-decreasing with every repeat a
 * boundary; no radius appears more than twice; every layer has at least two
 * knots. Construction validates all of that.
 */
class Deck {
 public:
  /// A validated deck from data. The header lines are kept verbatim.
  [[nodiscard]] static std::expected<Deck, DeckError> Create(
      std::vector<double> radius,
      std::vector<std::pair<std::string, std::vector<double>>> columns,
      std::vector<std::string> header = {});

  /// A deck from a file: the format's header lines, then
  /// whitespace-separated rows, radius first. Blank lines in the table are
  /// skipped; `nan` is a valid value.
  [[nodiscard]] static std::expected<Deck, DeckError> Read(
      const std::filesystem::path& path, const DeckFormat& format);

  /**
   * @brief A deck from a legacy sl3d/gia1d model file (`prem.simp.vm1`
   * and kin): a title line; `nknot ntau`; rows `r rho kappa mu viscosity`
   * with r in m, rho in kg m^-3, kappa and mu in GPa, and viscosity as
   * log10 of Pa s with 0 meaning none.
   *
   * The result is translated to the standard SI columns `rho`, `kappa`,
   * `mu` (Pa) and `viscosity` (Pa s, NaN where the file has 0). A layer
   * mixing zero and non-zero viscosity is refused, since NaN on part of a
   * layer is meaningless. Only ntau = 1 files are supported.
   */
  [[nodiscard]] static std::expected<Deck, DeckError> ReadLegacySl3d(
      const std::filesystem::path& path);

  /// The knot radii, non-decreasing, each repeat a boundary.
  [[nodiscard]] std::span<const double> Radius() const { return radius_; }

  /// The column names, in the deck's order.
  [[nodiscard]] std::span<const std::string> Names() const { return names_; }

  [[nodiscard]] bool Has(const std::string& name) const;

  /// The values of a column at the knots; throws std::out_of_range for a
  /// name the deck lacks.
  [[nodiscard]] std::span<const double> Column(const std::string& name) const;

  /// The verbatim header lines.
  [[nodiscard]] std::span<const std::string> Header() const { return header_; }

  [[nodiscard]] std::size_t KnotCount() const { return radius_.size(); }

  [[nodiscard]] std::size_t LayerCount() const { return layers_.size(); }

  /// The half-open knot ranges [first, last) of every layer, centre
  /// outward: one per span between repeated radii.
  [[nodiscard]] std::span<const std::pair<std::size_t, std::size_t>> Layers()
      const {
    return layers_;
  }

  /// The boundary radii: the ends and every repeated radius
  /// (LayerCount() + 1 values).
  [[nodiscard]] std::span<const double> Boundaries() const {
    return boundaries_;
  }

 private:
  Deck() = default;

  std::vector<double> radius_;
  std::vector<std::string> names_;
  std::vector<std::vector<double>> values_;  // parallel to names_
  std::vector<std::string> header_;
  std::vector<std::pair<std::size_t, std::size_t>> layers_;
  std::vector<double> boundaries_;
};

}  // namespace sl3dpp
