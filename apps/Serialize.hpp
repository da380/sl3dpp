#pragma once

/**
 * @file Serialize.hpp
 * @brief The binary plumbing of the programs' scratch files — the evolve
 * commands' checkpoints and the kernel command's spilled trajectories:
 * machine-local, private to a run, sized and raw.
 */

#include <complex>
#include <cstdint>
#include <fstream>
#include <vector>

#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"

namespace sl3dpp::app {

inline void Put(std::ofstream& out, const void* data, std::size_t bytes) {
  out.write(static_cast<const char*>(data),
            static_cast<std::streamsize>(bytes));
}

template <class T>
void PutOne(std::ofstream& out, T value) {
  Put(out, &value, sizeof(T));
}

inline void PutVector(std::ofstream& out, const std::vector<double>& values) {
  PutOne(out, static_cast<std::uint64_t>(values.size()));
  Put(out, values.data(), values.size() * sizeof(double));
}

inline void PutComplex(std::ofstream& out,
                       const std::vector<std::complex<double>>& values) {
  PutOne(out, static_cast<std::uint64_t>(values.size()));
  Put(out, values.data(), values.size() * sizeof(std::complex<double>));
}

template <class T>
T GetOne(std::ifstream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

inline std::vector<double> GetVector(std::ifstream& in) {
  auto values = std::vector<double>(GetOne<std::uint64_t>(in));
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(double)));
  return values;
}

inline std::vector<std::complex<double>> GetComplex(std::ifstream& in) {
  auto values = std::vector<std::complex<double>>(GetOne<std::uint64_t>(in));
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(values.size() *
                                       sizeof(std::complex<double>)));
  return values;
}

inline void PutTensor(std::ofstream& out,
                      const GridCoupling::TensorField& tensor) {
  PutComplex(out, tensor.c[0]);
  PutComplex(out, tensor.c[1]);
  PutVector(out, tensor.r);
}

inline GridCoupling::TensorField GetTensor(std::ifstream& in) {
  auto tensor = GridCoupling::TensorField{};
  tensor.c[0] = GetComplex(in);
  tensor.c[1] = GetComplex(in);
  tensor.r = GetVector(in);
  return tensor;
}

inline void PutSpectral(std::ofstream& out,
                        const GridCoupling::SpectralTensor& spectral) {
  PutComplex(out, spectral.c[0]);
  PutComplex(out, spectral.c[1]);
  PutComplex(out, spectral.r);
}

inline GridCoupling::SpectralTensor GetSpectral(std::ifstream& in) {
  auto spectral = GridCoupling::SpectralTensor{};
  spectral.c[0] = GetComplex(in);
  spectral.c[1] = GetComplex(in);
  spectral.r = GetComplex(in);
  return spectral;
}

/// A 3D internal state whole: t, the memory tensors and whichever strain
/// caches it carries.
inline void PutState3D(std::ofstream& out, const Viscoelastic3D::State& state) {
  PutOne(out, state.t);
  PutOne(out, static_cast<std::uint64_t>(state.m.size()));
  for (const auto& tensor : state.m) PutTensor(out, tensor);
  PutOne(out, static_cast<std::uint64_t>(state.strain.has_value() ? 1 : 0));
  if (state.strain) PutTensor(out, *state.strain);
  PutOne(out, static_cast<std::uint64_t>(
                  state.strainCoefficients.has_value() ? 1 : 0));
  if (state.strainCoefficients) PutSpectral(out, *state.strainCoefficients);
}

inline Viscoelastic3D::State GetState3D(std::ifstream& in) {
  auto state = Viscoelastic3D::State{};
  state.t = GetOne<double>(in);
  state.m.resize(GetOne<std::uint64_t>(in));
  for (auto& tensor : state.m) tensor = GetTensor(in);
  if (GetOne<std::uint64_t>(in) != 0) state.strain = GetTensor(in);
  if (GetOne<std::uint64_t>(in) != 0) state.strainCoefficients = GetSpectral(in);
  return state;
}

/// The recurrence of a 1D checkpoint: the per-degree internal variables
/// and their strain cache.
inline void PutState(std::ofstream& out,
                     const SeaLevelEvolution::Checkpoint& cp) {
  for (const auto* held : {&cp.m, &cp.strain}) {
    PutOne(out, static_cast<std::uint64_t>(held->size()));
    for (const auto& [degree, values] : *held) {
      PutOne(out, static_cast<std::int64_t>(degree));
      PutVector(out, values);
    }
  }
}

inline void GetState(std::ifstream& in, SeaLevelEvolution::Checkpoint& cp) {
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
                     const SeaLevelEvolution3D::Checkpoint& cp) {
  PutOne(out, static_cast<std::uint64_t>(cp.m.size()));
  for (const auto& tensor : cp.m) PutTensor(out, tensor);
  PutOne(out, static_cast<std::uint64_t>(
                  cp.strainCoefficients.has_value() ? 1 : 0));
  if (cp.strainCoefficients) PutSpectral(out, *cp.strainCoefficients);
}

inline void GetState(std::ifstream& in, SeaLevelEvolution3D::Checkpoint& cp) {
  cp.m.resize(GetOne<std::uint64_t>(in));
  for (auto& tensor : cp.m) tensor = GetTensor(in);
  if (GetOne<std::uint64_t>(in) != 0) cp.strainCoefficients = GetSpectral(in);
}

}  // namespace sl3dpp::app
