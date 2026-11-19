#pragma once

/**
 * @file Perturbation.hpp
 * @brief The `perturbation` block of the configs: a direction in a run's
 * inputs, given as the inputs themselves are. Its members mirror the
 * config's own, with the same spellings, file forms and units, and are
 * read by the readers of the things they perturb:
 *
 * - `forcing`: the lists of the `forcing` block (`ice_change`, `load`,
 *   `tide`, `seismic`), each term a change of what a term of the run's own
 *   list of that kind gives (ReadForcing, on the run's clock);
 * - `model.viscosity`: a planetmodel model file, as `model.viscosity`'s,
 *   of changes of the natural logarithm of the viscosity, pure numbers:
 *   its field `ln_viscosity` every relaxing branch's alike (every
 *   relaxation time moved by it, the branches' moduli held) and its fields
 *   `ln_viscosity_branch_<b>` one branch's alone, b as `kernel` numbers
 *   its `K_branch_<b>`; the fields add;
 * - `model.moduli`: a planetmodel model file, as `model.moduli`'s, whose
 *   fields `mu` and `kappa` (either alone) hold the changes of the moduli
 *   the run uses, Pa;
 * - `ocean`: the state's sea level, the member the run's own `ocean` block
 *   gives it by (`sea_level`, or `topography`), a file input read at the
 *   state's instant.
 *
 * The members together are ONE direction, their sum. A run with the
 * perturbation's members added to its inputs, epsilon times, differs from
 * the run by epsilon times the direction's tangent-linear run, to first
 * order: `tangent` computes that run, and the direction is what `kernel`'s
 * kernels pair with.
 *
 * The block is read in two steps. ReadPerturbationBlock reads its members
 * and holds them to the run when the config is read; the direction itself
 * (DirectionOf) is formed once the run's Earth is built and its forward
 * run recorded, at the nodes the library steps (StepGridOf): a term that
 * jumps does so where the run's own forcing jumps, or is refused.
 */

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Blocks.hpp"
#include "Common.hpp"
#include "Files.hpp"
#include "Forcing.hpp"
#include "Input.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/TangentLinear3D.hpp"

namespace sl3dpp::app {

/// A recorded forward run and what it ran on (Recorded.hpp).
struct RecordedRun;
struct Recorded1D;
struct Recorded3D;

/// The `perturbation` block as read: the direction's forcing (its lists
/// empty where the block gives none), the model files of its changes of
/// ln viscosity and of the moduli (empty paths where it gives none), and
/// the file input of its change of the state's sea level, by the member
/// the run's `ocean` block gives its own.
struct PerturbationBlock {
  Forcing forcing;
  std::filesystem::path viscosity;
  std::filesystem::path moduli;
  std::optional<FileInput> seaLevel;

  /// Whether the direction moves the Earth's parameters, which a run with
  /// lateral structure takes as fields.
  [[nodiscard]] bool OfTheEarth() const {
    return !viscosity.empty() || !moduli.empty();
  }
};

/// Reads a `perturbation` block of a run, relative paths against
/// `directory`, and holds it to the run: an object of `forcing`, `model`
/// and `ocean`, at least one, each holding at least one member.
///
/// - `forcing` is read as the run's own (ReadForcing, on the run's clock
///   and with its ocean), and held to the state as the run's is
///   (HoldForcingToState): an ice change needs an ocean; with the state at
///   the window's end the ice changes are `thickness` terms and nothing
///   acts beside the ice.
/// - `model` holds `viscosity` and `moduli`, each an object of `file`
///   alone (a `reference` is refused: the file's field is the change as it
///   stands). A change of the viscosity is refused in the static problem,
///   whose Earth is elastic; it is taken on every rheology (a Maxwell,
///   prony or burgers deck, the regions of `model.burgers`), as a change of
///   the branches' relaxation times.
/// - `ocean` holds exactly one of `sea_level` and `topography`, the member
///   the run's own block has, a file input of `file`, `format`, `variable`
///   and `outside_span`; `ice` and `land` are refused, naming what is not
///   built. Refused without an ocean.
///
/// The block is refused without gravity, whose tangent-linear run is not
/// built.
[[nodiscard]] PerturbationBlock ReadPerturbationBlock(
    const Json& block, const Run& run, const std::filesystem::path& directory);

/// The run with the perturbation's forcing in place of its own, readied
/// against the run's Earth as the run's own forcing is: its fault files
/// placed (PlaceSeismicSources) and its files' jumps found (FindFileJumps).
/// `block` is the `perturbation` block.
[[nodiscard]] Run PerturbedForcingOf(const Run& run,
                                     const PerturbationBlock& perturbation,
                                     const BuiltEarth& earth,
                                     const Json& block);

/// A direction against a recorded run, in the library's form, with what
/// the files of its tangent-linear run state beside the response.
template <class Perturbation>
struct Direction {
  /// The direction over the library's nodes (RecordedRun::steps): the
  /// forcing's terms at each node (their left limits at the first of a
  /// repeated pair), the ice's with the state held or the run's start
  /// moving as the kernels' ice views read it, the initial sea level's
  /// change, and the Earth's parameters'.
  Perturbation perturbation;
  /// A calibrated run's alone, where the block moves the state's sea
  /// level and the derivative is the total one: the change of the sea
  /// level the run is calibrated to, solver units. Empty otherwise.
  std::vector<double> targetSeaLevel;
  /// The direction's surface loads on the solver's grid (null without
  /// any), and its ice change at the static problem's instant, SI (empty
  /// through time and without one): what the files' `load` fields and
  /// `ice_change` hold.
  std::shared_ptr<const SurfaceLoads> loads;
  std::vector<double> iceChange;
};

/// The surface part of a direction against a recorded run (the forcing's
/// lists and the state's sea level), in either dimension's form.
/// `forcing` is PerturbedForcingOf's run and `block` the `perturbation`
/// block; `total` says whether a calibrated run's derivative is the total
/// one (the block's sea level then the target's change) or the partial one
/// (the calibrated initial sea level's).
///
/// The terms are read at the library's nodes on the terms' clock, by the
/// readers of the run's own forcing (ForcingOf, RunIce), in the solver's
/// units. A term that jumps at a time where the run's own forcing does
/// not (so that the library steps no pair there) is refused at its
/// member: the tangent-linear run along it is not built.
///
/// The ice at a node is the state's plus the change there. With the state
/// held (at the window's start, or the static problem's) a change at the
/// first node is the ice change there, the state's ice held: the
/// library's thickness direction and the direct load rho_i (1 - C) dI_0
/// at every node beside it (TangentPerturbation::iceThickness). With the
/// state at the window's end the run starts from the first node's ice,
/// which moves with it: the library's thickness direction as it stands.
/// Either is the view `kernel`'s `thickness_dual` pairs with.
template <class Perturbation>
[[nodiscard]] Direction<Perturbation> SurfaceDirectionOf(
    const RecordedRun& recorded, const PerturbationBlock& perturbation,
    const Run& forcing, const Json& block, bool total);

/// Holds the model files of a direction in the Earth's parameters to the
/// run's Earth, before the forward run: each is read (a file that cannot
/// be read, is of another body or lacks its fields refused at its member,
/// as is a viscosity file holding a field that is not a change of ln
/// viscosity: `viscosity`, a parameter a prony or burgers deck states, any
/// other but the moduli's),
/// and one whose field varies laterally is refused for a radial run
/// (`lateral` false), the tangent-linear run of which along lateral
/// structure is not built.
void CheckEarthDirection(const PerturbationBlock& perturbation,
                         const BuiltEarth& earth, bool lateral,
                         const Json& block);

/// Adds a radial run's direction in the Earth's parameters: the changes of
/// ln viscosity as changes of the branches' inverse relaxation times at
/// the mesh's nodes (delta itau_b = -itau_b delta ln eta_b, with
/// delta ln eta_b the file's `ln_viscosity` plus its
/// `ln_viscosity_branch_<b>`, the branch's modulus held), and the changes of
/// kappa and mu at the operator's memory points, the fluid's shear left
/// out, in the material's units. A file whose field varies laterally is
/// refused at its member: the tangent-linear run of a radial run along
/// lateral structure is not built. A field of one branch is refused for a
/// branch the rheology does not have, and where it is not zero at a point
/// that relaxes without that branch (a Maxwell layer beside Burgers
/// bodies, whose one branch is branch 0); where nothing relaxes a change
/// of ln viscosity moves nothing.
void AddEarthDirection(TangentPerturbation& direction,
                       const PerturbationBlock& perturbation,
                       const Recorded1D& recorded, const Json& block);

/// A radial run's direction in the unrelaxed shear modulus at the radial
/// mesh's nodes, (element, node) over the mesh, element by element, in the
/// material's units: the `mu` field of `perturbation.model.moduli` read at
/// each node within its element's layer, zero in the fluid; empty where
/// the block gives no change of mu. AddEarthDirection takes the direction
/// at the operator's memory points, where the run reads its moduli; this
/// is it at the nodes, where the radial viscosity kernels live, whose
/// branches' moduli move with it (`kernel`'s second-order viscosity
/// kernels).
[[nodiscard]] std::vector<double> ShearChangeAtNodes(
    const PerturbationBlock& perturbation, const Recorded1D& recorded,
    const Json& block);

/// Adds a 3-D run's direction in the Earth's parameters, over the grid
/// values of its coupling's memory nodes (each rank its own): the changes
/// of ln viscosity (every branch's alike, and each branch's own as a
/// change of its inverse relaxation time, the branches refused as the
/// radial run's are), and the changes of kappa and mu in the material's
/// units. A 3-D run takes lateral moduli in neither the fluid nor the
/// element holding the centre, and a direction's are left out there as the
/// run's own are. A change of the moduli that is not zero in another
/// element the coupling carries no memory nodes for (an elastic one
/// without lateral moduli of the run's own) is refused at its member: the
/// 3-D tangent-linear run takes the moduli's change where its lateral
/// structure lives alone.
void AddEarthDirection(TangentPerturbation3D& direction,
                       const PerturbationBlock& perturbation,
                       const Recorded3D& recorded, const Json& block);

/// The block's members as described, for a summary and the files'
/// metadata: by name (`ice_change`, `load`, `tide`, `seismic`,
/// `viscosity`, `moduli`, `sea_level` or `topography`), each term
/// described as the run's own are (Describe) and joined by `; `.
[[nodiscard]] std::vector<std::pair<std::string, std::string>>
DescribePerturbation(const Run& run, const PerturbationBlock& perturbation);

}  // namespace sl3dpp::app
