#pragma once

/**
 * @file detail/Shorelines.hpp
 * @brief What the 1D and 3D evolutions share under migrating shorelines
 * (pyslfp's `_run_migrating`). Internal.
 *
 * A migrating run starts from its background state, which must be the state
 * at the first node.
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
void CheckMigratingStart(const GRDSolver& solver,
                         const GRDSolver::State& background,
                         std::span<const double> startIce);

}  // namespace sl3dpp::detail
