#include "sl3dpp/StepRecipe.hpp"

#include <stdexcept>

namespace sl3dpp {

StepRecipe RecipeOf(Scheme scheme, double h) {
  if (!(h >= 0.0)) {
    throw std::invalid_argument("a step's length must be non-negative");
  }
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "ExponentialEuler is explicit in the strain: it has no implicit "
        "stages (ExponentialTrapezoid, BackwardEuler or Sdirk2)");
  }
  if (!IsMultiStage(scheme) || h == 0.0) {
    return {{{.c = 1.0,
              .scale = 1.0,
              .family = ZeroLengthScheme(scheme),
              .start = {{0, 1.0}}}}};
  }
  // Sdirk2: two backward Euler stages of length gamma h, the second from
  // (1 - r) m_n + r Y_1 (the file comment of StepRecipe.hpp)
  constexpr auto r = (1.0 - kSdirk2Gamma) / kSdirk2Gamma;
  return {{{.c = kSdirk2Gamma,
            .scale = kSdirk2Gamma,
            .family = Scheme::BackwardEuler,
            .start = {{0, 1.0}}},
           {.c = 1.0,
            .scale = kSdirk2Gamma,
            .family = Scheme::BackwardEuler,
            .start = {{0, 1.0 - r}, {1, r}}}}};
}

}  // namespace sl3dpp
