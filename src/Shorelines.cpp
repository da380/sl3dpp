#include "sl3dpp/detail/Shorelines.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sl3dpp::detail {

void CheckMigratingStart(const GRDSolver& solver,
                         const GRDSolver::State& background,
                         std::span<const double> startIce, bool startSolved) {
  const auto fieldSize = solver.Grid().FieldSize();
  if (startIce.size() != fieldSize) {
    throw std::invalid_argument(
        "the ice thickness must be a field on the solver's grid");
  }
  auto scale = 0.0, worst = 0.0;
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    scale = std::max(scale, std::abs(startIce[i]));
    worst = std::max(worst, std::abs(background.iceThickness[i] - startIce[i]));
  }
  if (!startSolved && worst > 1e-6 * scale) {
    throw std::invalid_argument(
        "the background state's ice thickness is not the history's at the "
        "first node; build the state at times[0]");
  }
  const auto flotation = solver.MakeState(
      background.iceThickness, background.seaLevel, background.forcedLand);
  if (flotation.oceanFunction != background.oceanFunction) {
    throw std::invalid_argument(
        "migrating shorelines follow the flotation condition from the first "
        "node, so the background state must not prescribe its ocean "
        "function");
  }
}

void CheckLandAtStart(const GRDSolver& solver,
                      const GRDSolver::State& background,
                      std::span<const char> startLand) {
  const auto fieldSize = solver.Grid().FieldSize();
  if (!startLand.empty() && startLand.size() != fieldSize) {
    throw std::invalid_argument(
        "the land held is not a mask on the solver's grid");
  }
  const auto& held = background.forcedLand;
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    const auto given = !startLand.empty() && startLand[i] != 0;
    const auto own = !held.empty() && held[i] != 0;
    if (given != own) {
      throw std::invalid_argument(
          "the land held at the first node is not the background state's: "
          "build the state with the land held at times[0]");
    }
  }
}

}  // namespace sl3dpp::detail
