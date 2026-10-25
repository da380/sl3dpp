#pragma once

/**
 * @file Common.hpp
 * @brief The run scaffold the time-dependent commands share: the earth
 * built from a config's deck, the background state and ice history read
 * from surface files onto the solver grid, the time grid of the window,
 * and the shared config members themselves.
 */

#include <filesystem>
#include <vector>

#include "Commands.hpp"
#include "Input.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

namespace sl3dpp::app {

/// The names of the run members `ReadRunMembers` reads, for a command's
/// Allow list.
[[nodiscard]] std::vector<std::string_view> RunMemberNames();

/// Reads the members every time-dependent command shares (the model,
/// the viscosity where given, the state and ice-history files, the
/// window, the scheme, rotation, moments and tolerances) into an
/// `EvolveInput`, relative paths against `directory`.
void ReadRunMembers(const Json& top, const std::filesystem::path& directory,
                    EvolveInput& out);

/// The model and material of a config's deck at its lmax.
struct BuiltEarth {
  EarthModel model;
  Material material;
};
[[nodiscard]] BuiltEarth BuildEarth(const EvolveInput& input, const Json& top);

/// The SLE parameters, the moments set hydrostatically where asked.
[[nodiscard]] SeaLevelParameters ParametersOf(const EvolveInput& input,
                                              const Material& material);

/// The world a run moves: the state and ice histories on the solver
/// grid, the window-start ice (SI), and the background state built from
/// the start's thickness and topography by the flotation construction.
struct Loaded {
  io::FieldHistory state;
  io::FieldHistory ice;
  std::vector<double> ice0;  ///< the start's ice thickness, SI
  SeaLevelSolver::State background;
};
[[nodiscard]] Loaded LoadWorld(const EvolveInput& input, const Json& top,
                               const SeaLevelSolver& solver);

/// The sea level the state file gives at the window's end, by the
/// flotation construction, solver units: the calibration's target. Refused
/// when the window's end lies outside the file's time span.
[[nodiscard]] std::vector<double> FinalSeaLevelOf(const EvolveInput& input,
                                                  const Json& top,
                                                  const SeaLevelSolver& solver,
                                                  const Loaded& world);

/// The window's uniform time grid, SLE units, zero at its start.
[[nodiscard]] std::vector<double> MakeTimes(const EvolveInput& input,
                                            const Scales& scales);

/// The grounded ice load relative to the window's start, as the drivers
/// take it: a callable of SLE time over the solver grid. The references
/// must outlive it.
[[nodiscard]] std::function<std::vector<double>(double)> IceLoadOf(
    const EvolveInput& input, const SeaLevelSolver& solver,
    const Loaded& world);

/// The ice thickness the migrating runs take, solver units: the
/// background's ice plus the history's change from the start's ice
/// (world.ice0, as IceLoadOf measures it). The references must outlive it.
[[nodiscard]] std::function<std::vector<double>(double)> IceThicknessOf(
    const EvolveInput& input, const SeaLevelSolver& solver,
    const Loaded& world);

/// The run's ice history under its shorelines: IceLoadOf when frozen,
/// IceThicknessOf when migrating.
[[nodiscard]] IceHistory HistoryOf(const EvolveInput& input,
                                   const SeaLevelSolver& solver,
                                   const Loaded& world);

/// Rebuilds the world's start from the ice history: its ice at the
/// window's start (world.ice0 too) under the given sea level, the ocean
/// function by flotation.
void StartFromIceHistory(const EvolveInput& input, const SeaLevelSolver& solver,
                         Loaded& world, std::span<const double> seaLevelGuess);

/// Calibrates the world's initial sea level against the state file's sea
/// level at the window's end, as the config asks, by an evolution the
/// caller built to stream nothing, and adopts it as the world's
/// background. The start is pyslfp's: the ice history's ice at the window's
/// start, the target sea level as the guess (StartFromIceHistory), so the
/// state file is read at the end alone. Refused when the calibration runs
/// out of iterations. Returns the misfits before and after (SI metres) and
/// the runs taken.
struct CalibrationSummary {
  double before_m = 0.0;
  double after_m = 0.0;
  std::size_t runs = 0;
};
CalibrationSummary CalibrateWorld(const EvolveInput& input, const Json& top,
                                  const SeaLevelSolver& solver, Loaded& world,
                                  SeaLevelEvolution& evolution,
                                  std::span<const double> times);
CalibrationSummary CalibrateWorld(const EvolveInput& input, const Json& top,
                                  const SeaLevelSolver& solver, Loaded& world,
                                  SeaLevelEvolution3D& evolution,
                                  std::span<const double> times);

}  // namespace sl3dpp::app
