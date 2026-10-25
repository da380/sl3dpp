#include "sl3dpp/detail/Shorelines.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sl3dpp::detail {

void CheckMigratingStart(const SeaLevelSolver& solver,
                         const SeaLevelSolver::State& background,
                         std::span<const double> startIce) {
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
  if (worst > 1e-6 * scale) {
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

std::vector<std::complex<double>> MigratedLoadCoefficients(
    const SeaLevelSolver& solver, const SeaLevelSolver::State& background,
    const SeaLevelSolver::State& final) {
  const auto fieldSize = solver.Grid().FieldSize();
  const auto& start = background;
  auto total = std::vector<double>(fieldSize);
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    total[i] =
        solver.IceDensity() *
            ((1.0 - final.oceanFunction[i]) * final.iceThickness[i] -
             (1.0 - start.oceanFunction[i]) * start.iceThickness[i]) +
        solver.WaterDensity() * (final.oceanFunction[i] * final.seaLevel[i] -
                                 start.oceanFunction[i] * start.seaLevel[i]);
  }
  return solver.Grid().Expand(total);
}

}  // namespace sl3dpp::detail
