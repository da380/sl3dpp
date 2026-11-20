#pragma once

/**
 * @file Serialize.hpp
 * @brief The binary plumbing of `solve`'s checkpoints: the library's raw
 * helpers (sl3dpp/io/Binary.hpp, which the trajectory store shares) and
 * the checkpoints' recurrences. Machine-local, private to a run, sized
 * and raw.
 */

#include <cstdint>
#include <fstream>

#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/io/Binary.hpp"

namespace sl3dpp::app {

using io::GetComplex;
using io::GetOne;
using io::GetSpectral;
using io::GetTensor;
using io::GetVector;
using io::Put;
using io::PutComplex;
using io::PutOne;
using io::PutSpectral;
using io::PutTensor;
using io::PutVector;

/// The recurrence of a 1D checkpoint: the per-degree internal variables
/// and their strain cache.
inline void PutState(std::ofstream& out,
                     const Evolution::Checkpoint& cp) {
  for (const auto* held : {&cp.m, &cp.strain}) {
    PutOne(out, static_cast<std::uint64_t>(held->size()));
    for (const auto& [degree, values] : *held) {
      PutOne(out, static_cast<std::int64_t>(degree));
      PutVector(out, values);
    }
  }
}

inline void GetState(std::ifstream& in, Evolution::Checkpoint& cp) {
  for (auto* held : {&cp.m, &cp.strain}) {
    const auto degrees = GetOne<std::uint64_t>(in);
    for (auto k = std::uint64_t{0}; k < degrees; ++k) {
      const auto degree = GetOne<std::int64_t>(in);
      held->emplace(static_cast<int>(degree), GetVector(in));
    }
  }
}

/// The recurrence of a 3D checkpoint: the memory tensors and the
/// strain's coefficients.
inline void PutState(std::ofstream& out,
                     const Evolution3D::Checkpoint& cp) {
  PutOne(out, static_cast<std::uint64_t>(cp.m.size()));
  for (const auto& tensor : cp.m) PutTensor(out, tensor);
  PutOne(out, static_cast<std::uint64_t>(
                  cp.strainCoefficients.has_value() ? 1 : 0));
  if (cp.strainCoefficients) PutSpectral(out, *cp.strainCoefficients);
}

inline void GetState(std::ifstream& in, Evolution3D::Checkpoint& cp) {
  cp.m.resize(GetOne<std::uint64_t>(in));
  for (auto& tensor : cp.m) tensor = GetTensor(in);
  if (GetOne<std::uint64_t>(in) != 0) cp.strainCoefficients = GetSpectral(in);
}

}  // namespace sl3dpp::app
