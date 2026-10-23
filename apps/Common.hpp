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
#include "sl3dpp/Material.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/io/FieldHistory.hpp"

namespace sl3dpp::app {

/// Seconds in a kiloyear of Julian years, as pyslfp counts them.
inline constexpr auto KYR = 1.0e3 * 365.25 * 86400.0;

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

/// The window's uniform time grid, SLE units, zero at its start.
[[nodiscard]] std::vector<double> MakeTimes(const EvolveInput& input,
                                            const Scales& scales);

/// The grounded ice load relative to the window's start, as the drivers
/// take it: a callable of SLE time over the solver grid. The references
/// must outlive it.
[[nodiscard]] std::function<std::vector<double>(double)> IceLoadOf(
    const EvolveInput& input, const SeaLevelSolver& solver,
    const Loaded& world);

}  // namespace sl3dpp::app
