#pragma once

/**
 * @file SurfaceGrid.hpp
 * @brief The spherical surface grid and the real scalar transforms the sea
 * level machinery runs on: a thin, value-semantic wrapper of GSHTrans's
 * Gauss-Legendre grid.
 *
 * Fields are plain vectors over the grid points in GSHTrans's storage order
 * (colatitude-major: lmax + 1 Gauss-Legendre colatitudes, uniform
 * longitudes from zero). Coefficients are complex vectors in GSHTrans's
 * reduced m >= 0 layout for real fields, index l (l + 1) / 2 + m, in the
 * library's native basis: complex, orthonormal, Condon-Shortley phase,
 * Y_lm = sqrt((2l+1)/4pi) d^l_0m e^{i m phi} — identical to the old
 * Fortran's. Conversions to the real orthonormal no-CS-phase (pyshtools)
 * convention live in ShConventions.hpp and are used only at the IO and
 * benchmark boundary.
 *
 * The grid is unit-sphere geometry: Integrate() carries no radius factor,
 * and the caller supplies r^2 where a physical surface integral is meant.
 */

#include <GSHTrans/All>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

namespace sl3dpp {

class SurfaceGrid {
 public:
  using Gsh = GSHTrans::GaussLegendreGrid<double, GSHTrans::All, GSHTrans::All>;

  struct Options {
    /// The largest upper index (spin) the grid's Wigner table covers;
    /// 0 suffices for scalar work, the 3D coupling wants 2.
    int nmax = 0;
    /// Grid resolution headroom for pointwise products (1.5 = the 3/2
    /// dealiasing rule); 1 resolves the band exactly.
    double oversampling = 1.0;
    /// Whether GSHTrans transforms run threaded.
    bool parallel = false;
  };

  SurfaceGrid(int lmax, Options options)
      : lmax_(lmax),
        parallel_(options.parallel),
        grid_(Gsh::ForBand(lmax, options.nmax, options.oversampling)) {
    for (auto theta : grid_.CoLatitudes()) coLatitudes_.push_back(theta);
    for (auto w : grid_.CoLatitudeWeights()) coLatitudeWeights_.push_back(w);
    for (auto phi : grid_.Longitudes()) longitudes_.push_back(phi);
  }
  explicit SurfaceGrid(int lmax) : SurfaceGrid(lmax, Options{}) {}

  [[nodiscard]] int LMax() const { return lmax_; }
  [[nodiscard]] int NTheta() const {
    return static_cast<int>(coLatitudes_.size());
  }
  [[nodiscard]] int NPhi() const {
    return static_cast<int>(longitudes_.size());
  }

  [[nodiscard]] std::span<const double> CoLatitudes() const {
    return coLatitudes_;
  }
  [[nodiscard]] std::span<const double> Longitudes() const {
    return longitudes_;
  }

  /// The number of grid values of a field (colatitude-major).
  [[nodiscard]] std::size_t FieldSize() const {
    return coLatitudes_.size() * longitudes_.size();
  }

  /// The number of coefficients of a real scalar field expanded to LMax().
  [[nodiscard]] std::size_t CoefficientSize() const {
    return static_cast<std::size_t>((lmax_ + 1) * (lmax_ + 2) / 2);
  }

  /// The coefficient index of (l, m), m >= 0.
  [[nodiscard]] static std::size_t Index(int l, int m) {
    return static_cast<std::size_t>(l) * (l + 1) / 2 + m;
  }

  /// A field sampled from f(theta, phi) over the points, storage order.
  template <typename Function>
  [[nodiscard]] std::vector<double> FromFunction(Function&& f) const {
    std::vector<double> out;
    out.reserve(FieldSize());
    for (auto theta : coLatitudes_) {
      for (auto phi : longitudes_) out.push_back(f(theta, phi));
    }
    return out;
  }

  [[nodiscard]] std::vector<double> ZeroField() const {
    return std::vector<double>(FieldSize(), 0.0);
  }
  [[nodiscard]] std::vector<std::complex<double>> ZeroCoefficients() const {
    return std::vector<std::complex<double>>(CoefficientSize(), 0.0);
  }

  /// Forward transform of a real scalar field into the m >= 0 coefficients.
  void Expand(std::span<const double> field,
              std::span<std::complex<double>> coefficients) const {
    grid_.ForwardTransformation(lmax_, 0, field, coefficients, Policy());
  }
  [[nodiscard]] std::vector<std::complex<double>> Expand(
      std::span<const double> field) const {
    auto out = ZeroCoefficients();
    Expand(field, out);
    return out;
  }

  /// Inverse transform back to grid values.
  void Synthesise(std::span<const std::complex<double>> coefficients,
                  std::span<double> field) const {
    grid_.InverseTransformation(lmax_, 0, coefficients, field, Policy());
  }
  [[nodiscard]] std::vector<double> Synthesise(
      std::span<const std::complex<double>> coefficients) const {
    std::vector<double> out(FieldSize());
    Synthesise(coefficients, out);
    return out;
  }

  // -- spin-weighted (complex) fields, |N| <= the grid's nmax -----------------

  /// The number of coefficients of a spin-N field: degrees |N|..lmax,
  /// all orders.
  [[nodiscard]] std::size_t SpinCoefficientSize(int N) const {
    return static_cast<std::size_t>((lmax_ + 1) * (lmax_ + 1) - N * N);
  }

  /// The coefficient index of (l, m) at upper index N (GSHTrans's
  /// GSHIndices<All> layout): l^2 - N^2 + (m + l).
  [[nodiscard]] static std::size_t SpinIndex(int N, int l, int m) {
    return static_cast<std::size_t>(l) * l - static_cast<std::size_t>(N * N) +
           (m + l);
  }

  /// Forward transform of a complex spin-N field (values in storage
  /// order) into its coefficients.
  void ExpandSpin(int N, std::span<const std::complex<double>> field,
                  std::span<std::complex<double>> coefficients) const {
    grid_.ForwardTransformation(lmax_, N, field, coefficients, Policy());
  }
  [[nodiscard]] std::vector<std::complex<double>> ExpandSpin(
      int N, std::span<const std::complex<double>> field) const {
    std::vector<std::complex<double>> out(SpinCoefficientSize(N));
    ExpandSpin(N, field, out);
    return out;
  }

  /// Inverse transform of spin-N coefficients back to grid values.
  void SynthesiseSpin(int N, std::span<const std::complex<double>> coefficients,
                      std::span<std::complex<double>> field) const {
    grid_.InverseTransformation(lmax_, N, coefficients, field, Policy());
  }
  [[nodiscard]] std::vector<std::complex<double>> SynthesiseSpin(
      int N, std::span<const std::complex<double>> coefficients) const {
    std::vector<std::complex<double>> out(FieldSize());
    SynthesiseSpin(N, coefficients, out);
    return out;
  }

  /// int f dOmega over the unit sphere, by the grid's quadrature (exact
  /// for fields band-limited at the grid's resolution).
  [[nodiscard]] double Integrate(std::span<const double> field) const {
    const auto nphi = longitudes_.size();
    const double dphi = 2.0 * std::numbers::pi / static_cast<double>(nphi);
    double sum = 0.0;
    for (std::size_t i = 0; i < coLatitudes_.size(); ++i) {
      double ring = 0.0;
      for (std::size_t j = 0; j < nphi; ++j) ring += field[i * nphi + j];
      sum += coLatitudeWeights_[i] * ring;
    }
    return sum * dphi;
  }

  /// The underlying GSHTrans grid, for machinery beyond scalars.
  [[nodiscard]] const Gsh& Grid() const { return grid_; }

 private:
  [[nodiscard]] GSHTrans::Execution Policy() const {
    return parallel_ ? GSHTrans::Execution::Parallel()
                     : GSHTrans::Execution::Sequential();
  }

  int lmax_;
  bool parallel_;
  Gsh grid_;
  std::vector<double> coLatitudes_;
  std::vector<double> coLatitudeWeights_;
  std::vector<double> longitudes_;
};

}  // namespace sl3dpp
