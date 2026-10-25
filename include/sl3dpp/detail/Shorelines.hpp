#pragma once

/**
 * @file detail/Shorelines.hpp
 * @brief What the 1D and 3D evolutions share under migrating shorelines
 * (pyslfp's `_run_migrating`). Internal.
 *
 * A migrating run starts from its background state, which must be the state
 * at the first node, and finishes each step under the load its converged
 * nonlinear state carries relative to that background.
 */

#include <complex>
#include <span>
#include <vector>

#include "sl3dpp/SeaLevel.hpp"

namespace sl3dpp::detail {

/// Refuses a migrating run whose background is not the state at its first
/// node: the background's ice thickness must be `startIce` (the history's
/// at times[0], on the solver's grid) to 1e-6 of its largest value, and its
/// ocean function the flotation one, since the shorelines follow flotation
/// from the first node.
void CheckMigratingStart(const SeaLevelSolver& solver,
                         const SeaLevelSolver::State& background,
                         std::span<const double> startIce);

/// The surface load of a converged nonlinear state relative to the
/// background, grounded ice and ocean water (pyslfp's `_total_load_coeffs`),
/// as m >= 0 coefficients in the solver's units: what the radial step
/// finishes under, so that it sees the load its sea level solution saw.
[[nodiscard]] std::vector<std::complex<double>> MigratedLoadCoefficients(
    const SeaLevelSolver& solver, const SeaLevelSolver::State& background,
    const SeaLevelSolver::State& final);

}  // namespace sl3dpp::detail
