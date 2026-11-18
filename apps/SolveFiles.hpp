#pragma once

/**
 * @file SolveFiles.hpp
 * @brief The surface files of `solve`, built from a run's snapshots: what
 * `solve` writes of its own run, and what `tangent` writes of its
 * tangent-linear run of it, the same fields in the same layout. A run
 * through time is a history of its window's nodes (HistoryNode) laid out
 * as node fields on a time axis (HistoryFile); the static problem is one
 * answer (StaticAnswer) as static fields (StaticFile). The fields
 * `output.fields` and `output.smoothing` add are Outputs.hpp's
 * (AddAskedFields), added by the writer.
 */

#include <array>
#include <complex>
#include <filesystem>
#include <planetmodel/surfacefile.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Common.hpp"
#include "Forcing.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

namespace sl3dpp::app {

/// One node of a run through time's history, SI: the node as the config
/// gives it (Window::nodes: its date, kyr BP, in a dated run, its time on
/// the run's clock in its unit in one counting forwards), the response,
/// the potentials of the tangential displacement (the toroidal one in the
/// frame of no net rotation of the surface, empty where the node has
/// none), under migrating shorelines the ocean function, and, in a 3-D
/// run, the node's closure: the passes of its combined fixed point, their
/// conjugate-gradient iterations (under that step solver) and its last
/// residual (zero in 1-D).
struct HistoryNode {
  double node = 0.0;
  std::vector<double> relativeSeaLevelChange;
  std::vector<double> displacement;
  std::vector<double> potential;
  std::vector<double> tangentialPotential;
  std::vector<double> toroidalPotential;
  std::array<double, 3> angularVelocity{};
  std::vector<double> oceanFunction;  ///< migrating runs alone
  long long stepIterations = 0;
  long long solveIterations = 0;
  double stepResidual = 0.0;
};

/// A node's closure into its history: a 3-D snapshot's passes, their
/// conjugate-gradient iterations and its residual; a 1-D one has none.
inline void KeepClosure(HistoryNode&, const Evolution::Snapshot&) {}
inline void KeepClosure(HistoryNode& kept,
                        const Evolution3D::Snapshot& snapshot) {
  kept.stepIterations = snapshot.coupledIterations;
  kept.solveIterations = snapshot.solveIterations;
  kept.stepResidual = snapshot.coupledResidual;
}

/// A snapshot's fields as a history's node, SI under the solver's scales:
/// the response, the toroidal potential in the frame of no net rotation of
/// the surface (the rigid rotation in the solver's W is its gauge), the
/// angular velocity and the ocean function, at the node `node` of the
/// window (Window::nodes). The closure is KeepClosure's.
template <class Snapshot>
void KeepFields(HistoryNode& kept, const Snapshot& snapshot,
                const GRDSolver& solver) {
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto toPotential = scales.Factor(PotentialDimensions);
  kept.relativeSeaLevelChange = snapshot.relativeSeaLevelChange;
  kept.displacement = snapshot.displacement;
  kept.potential = snapshot.potential;
  kept.tangentialPotential = snapshot.tangentialPotential;
  if (!snapshot.toroidalPotential.empty()) {
    kept.toroidalPotential =
        WithoutNetRotation(solver.Grid(), snapshot.toroidalPotential);
  }
  for (auto& v : kept.relativeSeaLevelChange) v *= length;
  for (auto& v : kept.displacement) v *= length;
  for (auto& v : kept.potential) v *= toPotential;
  for (auto& v : kept.tangentialPotential) v *= length;
  for (auto& v : kept.toroidalPotential) v *= length;
  const auto toOmega = 1.0 / scales.Time();
  for (auto k = 0; k < 3; ++k) {
    kept.angularVelocity[k] = snapshot.angularVelocity[k] * toOmega;
  }
  kept.oceanFunction = snapshot.oceanFunction;
}

/// The closure of a run's steps as the outputs name it: `direct` in 1-D
/// (each step one solve per degree), the 3-D step solver's word
/// otherwise.
[[nodiscard]] std::string_view StepSolverName(const Run& run, bool threeD);

/// The surface loads a run through time applied at each node of its
/// window, SI: `loads` on the terms' clock at the nodes' solver times, as
/// the run took them (ForcingOf); empty without any.
[[nodiscard]] std::vector<SurfaceLoads::Node> LoadsAtNodes(
    const Run& run, const GRDSolver& solver, const SurfaceLoads* loads);

/// A run through time's histories as one surface file: SI node fields on a
/// time axis in seconds, relative to the present (negative past) in a
/// dated run, the nodes' dates listed in `dates_kyr_bp`, and on the run's
/// own clock in one counting forwards, its nodes listed in its unit in
/// `times` beside the unit's name, `time_unit`. Beside the response: the
/// surface loads applied at each node (`loads`, LoadsAtNodes: every term
/// summed as given, and by their properties), the forcing's counts and
/// single terms (AddForcingCounts), each node's closure (`step_iterations`,
/// `solve_iterations`, `step_residual`, zero in 1-D) and, in a 3-D run,
/// the radial reference model's history under the same forcing
/// (`reference`; null otherwise); the initial sea level the run started
/// from (`initialSeaLevel`, SI) with where it came from (`origin`); the
/// ocean function, under migrating shorelines each node's, under frozen
/// ones the state's, a static field. Without an ocean the fields and
/// metadata of the state (the initial sea level, the ocean function, the
/// state's inputs, the shorelines) are left out, and `ocean` says there is
/// none.
[[nodiscard]] planetmodel::SurfaceFile HistoryFile(
    const Run& run, const std::filesystem::path& config,
    const SurfaceGrid& grid, std::span<const HistoryNode> history,
    const std::vector<double>& initialSeaLevel, std::string_view origin,
    const Loaded& world, const BuiltEarth& earth,
    const std::vector<SurfaceLoads::Node>& loads,
    const std::vector<HistoryNode>* reference,
    const StepsAgainstRelaxation& steps);

/// A static problem's answer, SI: the sea level change, the radial
/// displacement and the potential perturbation at the surface, the
/// potentials of the tangential displacement (the toroidal one in the
/// frame of no net rotation of the surface, empty where the run has none),
/// the angular velocity perturbation and, under migrating shorelines, the
/// ocean function the solve reaches (empty otherwise); and, of a solve
/// that recorded them, the coefficients of the total load the body
/// answered (the forcing's loads and the ocean's, kg m^-2) and of the
/// potential (the tide's and the rotation's centrifugal change, m^2
/// s^-2), empty otherwise.
struct StaticAnswer {
  std::vector<double> relativeSeaLevelChange;
  std::vector<double> displacement;
  std::vector<double> potential;
  std::vector<double> tangentialPotential;
  std::vector<double> toroidalPotential;
  std::array<double, 3> angularVelocity{};
  std::vector<double> oceanFunction;
  std::vector<std::complex<double>> loadCoefficients;
  std::vector<std::complex<double>> potentialCoefficients;
};

/// A one-node run's snapshot as the static problem's answer.
template <class Snapshot>
[[nodiscard]] StaticAnswer AnswerOf(const GRDSolver& solver,
                                    Snapshot snapshot) {
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto toPotential = scales.Factor(PotentialDimensions);
  const auto toLoad = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto out = StaticAnswer{
      .relativeSeaLevelChange = std::move(snapshot.relativeSeaLevelChange),
      .displacement = std::move(snapshot.displacement),
      .potential = std::move(snapshot.potential),
      .tangentialPotential = std::move(snapshot.tangentialPotential),
      .toroidalPotential = {},
      .angularVelocity = snapshot.angularVelocity,
      .oceanFunction = std::move(snapshot.oceanFunction),
      .loadCoefficients = std::move(snapshot.loadCoefficients),
      .potentialCoefficients = std::move(snapshot.tideCoefficients)};
  for (auto& v : out.relativeSeaLevelChange) v *= length;
  for (auto& v : out.displacement) v *= length;
  for (auto& v : out.potential) v *= toPotential;
  for (auto& v : out.tangentialPotential) v *= length;
  if (!snapshot.toroidalPotential.empty()) {
    // in the frame of no net rotation of the surface: the rigid rotation
    // in the solver's W is its gauge
    out.toroidalPotential =
        WithoutNetRotation(solver.Grid(), snapshot.toroidalPotential);
    for (auto& v : out.toroidalPotential) v *= length;
  }
  for (auto& v : out.angularVelocity) v /= scales.Time();
  for (auto& c : out.loadCoefficients) c *= toLoad;
  for (auto& c : out.potentialCoefficients) c *= toPotential;
  return out;
}

/// A static problem's answer as a surface file of static SI fields on the
/// run's grid: the ice change applied, the surface loads applied where
/// there are any (every term summed as given, and by their properties),
/// the response (the sea level change, the displacement, the potentials of
/// the tangential displacement and the potential; the angular velocity in
/// the metadata), the state's ocean function and, under migrating
/// shorelines, the one the solve reaches; with the config, model, moduli,
/// the state's ice and sea level as read, the forcing's terms, its counts
/// and single terms (AddForcingCounts), the shorelines and the rotation in
/// the metadata. Without an ocean the state's fields and metadata (the
/// ice change, the ocean function, the ice, the sea level, the
/// shorelines) are left out, and `ocean` says there is none. The runner
/// adds its own.
[[nodiscard]] planetmodel::SurfaceFile StaticFile(
    const std::filesystem::path& config, const Run& run,
    const StaticWorld& world, const SurfaceGrid& grid, std::string title,
    const StaticAnswer& answer, const SurfaceLoads::Node& applied);

}  // namespace sl3dpp::app
