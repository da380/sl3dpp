#pragma once

/**
 * @file Binary.hpp
 * @brief The raw binary plumbing of the files a run keeps for itself:
 * sl3d's checkpoints and the trajectory store (TrajectoryStore.hpp,
 * docs/trajectory-store.md). Every value is written as its bytes in the
 * machine's order (the files that say so require little-endian IEEE
 * doubles): a vector as its count, an unsigned 64-bit integer, followed
 * by its values; a complex value as its real part and then its imaginary
 * part; a map of degrees as its count followed by each degree, a signed
 * 64-bit integer, with its values.
 *
 * Every reader takes its source by template, so that a reader of its own
 * (the store's, which holds each count to the bytes its record has left)
 * finds its overloads of `ReadBytes` and `Counted` by argument-dependent
 * lookup. A std::istream reads as it stands: a short read leaves the
 * stream failed, for the caller to test.
 */

#include <complex>
#include <cstdint>
#include <istream>
#include <map>
#include <ostream>
#include <span>
#include <vector>

#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp::io {

// -- writing ------------------------------------------------------------------

inline void Put(std::ostream& out, const void* data, std::size_t bytes) {
  out.write(static_cast<const char*>(data),
            static_cast<std::streamsize>(bytes));
}

template <class T>
void PutOne(std::ostream& out, T value) {
  Put(out, &value, sizeof(T));
}

inline void PutVector(std::ostream& out, std::span<const double> values) {
  PutOne(out, static_cast<std::uint64_t>(values.size()));
  Put(out, values.data(), values.size() * sizeof(double));
}

inline void PutComplex(std::ostream& out,
                       std::span<const std::complex<double>> values) {
  PutOne(out, static_cast<std::uint64_t>(values.size()));
  Put(out, values.data(), values.size() * sizeof(std::complex<double>));
}

/// A map of real values by degree.
inline void PutDegrees(std::ostream& out,
                       const std::map<int, std::vector<double>>& degrees) {
  PutOne(out, static_cast<std::uint64_t>(degrees.size()));
  for (const auto& [degree, values] : degrees) {
    PutOne(out, static_cast<std::int64_t>(degree));
    PutVector(out, values);
  }
}

/// A map of complex values by degree.
inline void PutComplexDegrees(
    std::ostream& out,
    const std::map<int, std::vector<std::complex<double>>>& degrees) {
  PutOne(out, static_cast<std::uint64_t>(degrees.size()));
  for (const auto& [degree, values] : degrees) {
    PutOne(out, static_cast<std::int64_t>(degree));
    PutComplex(out, values);
  }
}

inline void PutTensor(std::ostream& out,
                      const GridCoupling::TensorField& tensor) {
  PutComplex(out, tensor.c[0]);
  PutComplex(out, tensor.c[1]);
  PutVector(out, tensor.r);
  PutOne(out, static_cast<std::uint64_t>(tensor.scalars.size()));
  for (const auto& s : tensor.scalars) PutVector(out, s);
}

inline void PutSpectral(std::ostream& out,
                        const GridCoupling::SpectralTensor& spectral) {
  PutComplex(out, spectral.c[0]);
  PutComplex(out, spectral.c[1]);
  PutComplex(out, spectral.r);
  PutOne(out, static_cast<std::uint64_t>(spectral.scalars.size()));
  for (const auto& s : spectral.scalars) PutComplex(out, s);
}

/// A 3D internal state whole: t, the memory tensors and whichever strain
/// caches it carries, each behind a presence flag.
inline void PutState3D(std::ostream& out, const Viscoelastic3D::State& state) {
  PutOne(out, state.t);
  PutOne(out, static_cast<std::uint64_t>(state.m.size()));
  for (const auto& tensor : state.m) PutTensor(out, tensor);
  PutOne(out, static_cast<std::uint64_t>(state.strain.has_value() ? 1 : 0));
  if (state.strain) PutTensor(out, *state.strain);
  PutOne(out, static_cast<std::uint64_t>(
                  state.strainCoefficients.has_value() ? 1 : 0));
  if (state.strainCoefficients) PutSpectral(out, *state.strainCoefficients);
}

// -- reading ------------------------------------------------------------------

/// Reads raw bytes from a stream; a short read fails the stream.
inline void ReadBytes(std::istream& in, void* data, std::size_t bytes) {
  in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
}

/// The count a stream's vector holds, as read: a std::istream takes it as
/// it stands (the values' read fails the stream where they run short).
inline std::uint64_t Counted(std::istream&, std::uint64_t count,
                             std::size_t /*valueBytes*/) {
  return count;
}

template <class T, class In>
T GetOne(In& in) {
  T value{};
  ReadBytes(in, &value, sizeof(T));
  return value;
}

template <class In>
std::vector<double> GetVector(In& in) {
  auto values = std::vector<double>(
      Counted(in, GetOne<std::uint64_t>(in), sizeof(double)));
  ReadBytes(in, values.data(), values.size() * sizeof(double));
  return values;
}

template <class In>
std::vector<std::complex<double>> GetComplex(In& in) {
  auto values = std::vector<std::complex<double>>(
      Counted(in, GetOne<std::uint64_t>(in), sizeof(std::complex<double>)));
  ReadBytes(in, values.data(), values.size() * sizeof(std::complex<double>));
  return values;
}

template <class In>
std::map<int, std::vector<double>> GetDegrees(In& in) {
  auto out = std::map<int, std::vector<double>>{};
  const auto count = Counted(in, GetOne<std::uint64_t>(in),
                             sizeof(std::int64_t) + sizeof(std::uint64_t));
  for (auto k = std::uint64_t{0}; k < count; ++k) {
    const auto degree = GetOne<std::int64_t>(in);
    out[static_cast<int>(degree)] = GetVector(in);
  }
  return out;
}

template <class In>
std::map<int, std::vector<std::complex<double>>> GetComplexDegrees(In& in) {
  auto out = std::map<int, std::vector<std::complex<double>>>{};
  const auto count = Counted(in, GetOne<std::uint64_t>(in),
                             sizeof(std::int64_t) + sizeof(std::uint64_t));
  for (auto k = std::uint64_t{0}; k < count; ++k) {
    const auto degree = GetOne<std::int64_t>(in);
    out[static_cast<int>(degree)] = GetComplex(in);
  }
  return out;
}

template <class In>
GridCoupling::TensorField GetTensor(In& in) {
  auto tensor = GridCoupling::TensorField{};
  tensor.c[0] = GetComplex(in);
  tensor.c[1] = GetComplex(in);
  tensor.r = GetVector(in);
  tensor.scalars.resize(
      Counted(in, GetOne<std::uint64_t>(in), sizeof(std::uint64_t)));
  for (auto& s : tensor.scalars) s = GetVector(in);
  return tensor;
}

template <class In>
GridCoupling::SpectralTensor GetSpectral(In& in) {
  auto spectral = GridCoupling::SpectralTensor{};
  spectral.c[0] = GetComplex(in);
  spectral.c[1] = GetComplex(in);
  spectral.r = GetComplex(in);
  spectral.scalars.resize(
      Counted(in, GetOne<std::uint64_t>(in), sizeof(std::uint64_t)));
  for (auto& s : spectral.scalars) s = GetComplex(in);
  return spectral;
}

template <class In>
Viscoelastic3D::State GetState3D(In& in) {
  auto state = Viscoelastic3D::State{};
  state.t = GetOne<double>(in);
  // a tensor holds at least its three counts and its scalars' count
  state.m.resize(
      Counted(in, GetOne<std::uint64_t>(in), 4 * sizeof(std::uint64_t)));
  for (auto& tensor : state.m) tensor = GetTensor(in);
  if (GetOne<std::uint64_t>(in) != 0) state.strain = GetTensor(in);
  if (GetOne<std::uint64_t>(in) != 0) {
    state.strainCoefficients = GetSpectral(in);
  }
  return state;
}

}  // namespace sl3dpp::io
