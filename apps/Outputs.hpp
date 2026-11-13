#pragma once

/**
 * @file Outputs.hpp
 * @brief The outputs of `solve` beside its surface file's fields: the
 * members of the `output` block held to the problem, the exchange tables
 * of surface coefficients (the cross-code benchmark files,
 * machine-independent text another code reads) with their provenance
 * header, named scalars and, through time, a table per node, the metadata
 * of a forcing's counts and of its single cap or tide coefficient, the
 * run's grid as a surface file states it, and the response inside the
 * body as a volume file (a planetmodel model file over the memory nodes of
 * the radial reference's coupling).
 */

#include <complex>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <planetmodel/grid.hpp>
#include <planetmodel/surfacefile.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Blocks.hpp"
#include "Forcing.hpp"
#include "Input.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/ElasticPerturbation3D.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/ModelModuli.hpp"

namespace sl3dpp::app {

/// The writing date, YYYY-MM-DD: the exchange tables' provenance stamp.
[[nodiscard]] std::string Today();

/// Creates the directory an output goes into, where its path names one.
void CreateParent(const std::filesystem::path& path);

/// A run's Gauss-Legendre grid as a surface file states it.
[[nodiscard]] planetmodel::AngularGrid AngularGridOf(const SurfaceGrid& grid);

/// The forcing's single cap, where its surface loads are exactly one cap;
/// and its single tide coefficient, where its tide is exactly one.
[[nodiscard]] const LoadTerm* SingleCap(const Forcing& forcing);
[[nodiscard]] const TideTerm* SingleTide(const Forcing& forcing);

/// The forcing's counts in a surface file's metadata, `load_terms` and
/// `tide_coefficients` (the surface loads and the tide's coefficients: an
/// ice change has no counts of its own), and, where the load is exactly
/// one cap, its members (`load_latitude_deg`, `load_longitude_deg`,
/// `load_width_deg`, `load_amplitude_kg_m2`), where the tide is exactly
/// one coefficient, its own (`tide_degree`, `tide_order`,
/// `tide_amplitude_m2_s2`, `tide_imaginary_m2_s2`).
void AddForcingCounts(planetmodel::SurfaceFile& file, const Forcing& forcing);

/// What a per-node exchange path holds where a node's index goes.
inline constexpr auto kNodeField = std::string_view{"{node}"};

/// Holds an `output` block's members beside the surface file to the
/// problem. In the static problem (`nodes` absent) an `exchange` path
/// holding `{node}` and `exchange_nodes` are refused. In a run through
/// time of `nodes` nodes, `volume` is refused with `volumeRefusal`, an
/// `exchange` path lacking `{node}` is refused (`run` naming the run, as
/// `a run through time`), and `exchange_nodes` needs `exchange` and
/// indices of the nodes.
void CheckOutputMembers(const Json& output, const OutputBlock& block,
                        std::optional<std::size_t> nodes, std::string_view run,
                        std::string_view volumeRefusal);

/// The named scalars an exchange table starts with, a line each: the
/// body's radius (m) and surface gravity (m s^-2), lmax, whether the model
/// is lateral and the forcing's counts; then, where the load is exactly
/// one cap, its members, and where the tide is exactly one coefficient,
/// its own.
[[nodiscard]] std::string SharedScalars(double radius, double gravity,
                                        int lmax, bool lateral,
                                        const Forcing& forcing);

/// One column pair of an exchange table: the quantity's symbol (the
/// columns are `<symbol>_re <symbol>_im`), what it is with its unit, as
/// the header names it (`the radial displacement U (m)`), and its m >= 0
/// coefficients, SI, in the grid's layout.
struct ExchangeColumn {
  std::string_view symbol;
  std::string_view quantity;
  std::span<const std::complex<double>> coefficients;
};

/// The columns of the radial displacement U (m) and the potential
/// perturbation Phi (m^2 s^-2): every exchange table's first two.
[[nodiscard]] std::vector<ExchangeColumn> DisplacementAndPotential(
    std::span<const std::complex<double>> U,
    std::span<const std::complex<double>> Phi);

/// An exchange table: a provenance header (what the table holds,
/// `heading` naming the response, then the columns' quantities; the
/// code's version and the date; a line per entry of `provenance`, each
/// `# ` and the entry), the named scalars, and a row of SI coefficients of
/// the columns per (l, m >= 0), %.17g.
void WriteExchangeTable(const std::filesystem::path& path,
                        std::string_view heading,
                        std::span<const std::string> provenance,
                        std::string_view scalars, int lmax,
                        std::span<const ExchangeColumn> columns);

/// The tables a run through time of `nodes` nodes writes: each node
/// chosen (`output.exchange_nodes`, every node without it) with its path,
/// `{node}` in `output.exchange` replaced by the node's index padded to
/// the width of the last, in increasing order.
[[nodiscard]] std::vector<std::pair<int, std::filesystem::path>>
ExchangePaths(const OutputBlock& output, std::size_t nodes);

/// The per-degree m >= 0 columns of a real surface field's coefficients,
/// each times `factor`: a load or a potential as the 3D operator's
/// channels take it.
[[nodiscard]] Viscoelastic3D::LoadCallable3D ColumnsOf(
    std::vector<std::complex<double>> coefficients, double factor);

/// The response inside the body for a volume file, at the memory nodes of
/// the radial reference's coupling (every solid element but the
/// centre's): the radial displacement and the deviatoric strain's
/// magnitude sqrt(e:e) of the degree solutions `x` (the reference's dof
/// layout), and, where `lateral` gives lateral moduli, the relative
/// perturbations they apply (`mu_relative`, `kappa_relative`), sampled
/// there. Non-dimensional under the model's scales.
using VolumeFields = std::vector<std::pair<std::string, std::vector<double>>>;
[[nodiscard]] VolumeFields VolumeFieldsOf(const ElasticPerturbation3D& reference,
                                          const Viscoelastic3D::Solutions& x,
                                          const io::ModelModuli* lateral);

/// The same over the ranks of a radially distributed solve, `x` holding
/// this rank's elements' solution, `elements` those elements, [first,
/// end) (the 3D operator's coupling's range, GridCoupling::Options::
/// elements): each rank's fields at the reference's nodes in its own
/// elements, gathered to the first rank in rank order, the bottom
/// segment's first, which is the reference's whole layout there (each
/// field empty on the other ranks). VolumeFieldsOf on one rank.
/// Collective.
[[nodiscard]] VolumeFields GatheredVolumeFields(
    const ElasticPerturbation3D& reference, const Viscoelastic3D::Solutions& x,
    const io::ModelModuli* lateral, std::pair<int, int> elements);

/// Writes the fields as a planetmodel model file over the reference's
/// coupling, in SI, with the model's density as the radial reference,
/// `name` the file's name attribute.
void WriteVolumeFile(const std::filesystem::path& path,
                     const ElasticPerturbation3D& reference,
                     const VolumeFields& fields, const RadialMesh& mesh,
                     const EarthModel& model, std::string_view name);

}  // namespace sl3dpp::app
