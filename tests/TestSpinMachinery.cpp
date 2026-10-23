// The spin machinery: spin-N transforms on the surface grid, the
// degree <-> component maps, their reduction to the 1D memory-force
// algebra (the zero-lateral identity, checked constant by constant), and
// the reality symmetry of the component fields.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include "sl3dpp/SpinComponents.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

using namespace sl3dpp;
using Complex = std::complex<double>;

namespace {

/// A deterministic pseudo-random stream in [-1, 1].
double Noise(std::size_t k) {
  return std::sin(12.9898 * static_cast<double>(k + 1)) * 43758.5453 -
         std::floor(std::sin(12.9898 * static_cast<double>(k + 1)) *
                    43758.5453) -
         0.5;
}

}  // namespace

TEST(SpinMachinery, SpinTransformRoundTrips) {
  constexpr auto lmax = 12;
  SurfaceGrid grid(lmax, {.nmax = 2});
  for (int N : {-2, -1, 0, 1, 2}) {
    auto coefficients = std::vector<Complex>(grid.SpinCoefficientSize(N));
    for (int l = std::abs(N); l <= lmax; ++l) {
      for (int m = -l; m <= l; ++m) {
        const auto k = SurfaceGrid::SpinIndex(N, l, m);
        coefficients[k] = Complex(Noise(7 * k + N + 2), Noise(11 * k));
      }
    }
    const auto field = grid.SynthesiseSpin(N, coefficients);
    const auto back = grid.ExpandSpin(N, field);
    auto scale = 0.0, diff = 0.0;
    for (auto k = std::size_t{0}; k < coefficients.size(); ++k) {
      scale = std::max(scale, std::abs(coefficients[k]));
      diff = std::max(diff, std::abs(back[k] - coefficients[k]));
    }
    EXPECT_LT(diff, 1e-12 * scale) << "N = " << N;
  }
}

TEST(SpinMachinery, ReductionToTheDegreeBasisMemoryForce) {
  // With W = 0 and uniform weights, the composition components -> force
  // rows must equal the 1D stepper's dual-from-scalars algebra:
  //   bU(i) = sum_q B(q,i) cR(q) - cR(i) + cS(i),
  //   bV(i) = (1/2) k^2 cR(i) + sum_q B(q,i) cS(q) - cS(i) + cM(i),
  // with cM = k^2 (k^2 - 2) mu Wq e_M, cR = (4/3) mu Wq e_R,
  // cS = k^2 mu Wq e_S. Checked per node: the pointwise row against
  // (-cR + cS | ...) and the derivative row against the B pairing.
  for (int l : {0, 1, 2, 5, 9}) {
    const auto k2 = l * (l + 1.0);
    const auto r = 0.83;
    const auto mu = 1.7, Wq = 0.31, jac = 0.21;
    DegreeNodeValues v;
    v.U = Complex(0.37, -0.81);
    v.rdU = Complex(-1.21, 0.44);
    if (l > 0) {
      v.V = Complex(0.95, 0.12);
      v.rdV = Complex(0.05, -0.66);
    }

    const auto eM = l > 0 ? v.V : Complex(0.0);
    const auto eR = v.rdU - v.U + 0.5 * k2 * v.V;
    const auto eS = l > 0 ? v.rdV - v.V + v.U : Complex(0.0);
    const auto cM = k2 * (k2 - 2.0) * mu * Wq * jac * eM;
    const auto cR = (4.0 / 3.0) * mu * Wq * jac * eR;
    const auto cS = k2 * mu * Wq * jac * eS;

    // The grid path: strain components, weighted by mu (uniform lateral
    // weight), back through the force rows with the documented
    // quadrature factors.
    auto A = DeviatoricComponents(l, v, r);
    const auto rows = ForceRowsFromComponents(l, A, r);
    const auto pointU = rows.u * jac * Wq * r * mu;
    const auto pointV = rows.v * jac * Wq * r * mu;
    const auto pointW = rows.w * jac * Wq * r * mu;
    // The 1D dual's pointwise parts (the B terms are compared below).
    EXPECT_NEAR(std::abs(pointU - (-cR + cS)), 0.0, 1e-13) << "l = " << l;
    EXPECT_NEAR(std::abs(pointV - (0.5 * k2 * cR - cS + cM)), 0.0, 1e-12)
        << "l = " << l;
    EXPECT_NEAR(std::abs(pointW), 0.0, 1e-13) << "l = " << l;  // W = 0

    // Derivative rows: du pairs as B cR, dv as B cS. B(q,i) = r D / jac,
    // so rows.du * (Wq r mu D) == D * (r/jac) * cR requires
    // rows.du * Wq * r * mu == (r / jac) * cR, i.e. du * jac * Wq * mu = cR.
    EXPECT_NEAR(std::abs(rows.du * jac * Wq * mu - cR), 0.0, 1e-13)
        << "l = " << l;
    EXPECT_NEAR(std::abs(rows.dv * jac * Wq * mu - cS), 0.0, 1e-12)
        << "l = " << l;
    EXPECT_NEAR(std::abs(rows.dw), 0.0, 1e-13) << "l = " << l;
  }
}

TEST(SpinMachinery, RealTensorsHaveConjugateComponentFields) {
  // Real scalar fields U, V, W (Hermitian coefficient symmetry) give a
  // real tensor. In this basis conj(Y^N_lm) = (-1)^(N-m) Y^(-N)_(l,-m),
  // so the grid components satisfy A(++) = conj(A(--)),
  // A(0+) = -conj(A(-0)) (the spin-1 pair ANTI-conjugates) and real
  // A(-+), pointwise.
  constexpr auto lmax = 8;
  SurfaceGrid grid(lmax, {.nmax = 2});
  const auto r = 1.0;

  // Hermitian per-(l, m) columns: f_{l,-m} = (-1)^m conj(f_{lm}).
  auto hermitian = [&](std::size_t seed, int l, int m) {
    const auto base = Complex(
        Noise(seed + 13 * SurfaceGrid::SpinIndex(0, l, std::abs(m))),
        m == 0 ? 0.0
               : Noise(seed + 7 * SurfaceGrid::SpinIndex(0, l, std::abs(m))));
    if (m >= 0) return base;
    return (std::abs(m) % 2 == 0 ? 1.0 : -1.0) * std::conj(base);
  };

  auto componentCoefficients = std::array<std::vector<Complex>, 5>{};
  const int spins[5] = {-2, -1, 0, 1, 2};
  for (auto c = 0; c < 5; ++c) {
    componentCoefficients[c].assign(grid.SpinCoefficientSize(spins[c]), 0.0);
  }
  for (auto l = 0; l <= lmax; ++l) {
    for (int m = -l; m <= l; ++m) {
      DegreeNodeValues v;
      v.U = hermitian(1, l, m);
      v.rdU = hermitian(2, l, m);
      if (l > 0) {
        v.V = hermitian(3, l, m);
        v.rdV = hermitian(4, l, m);
        v.W = hermitian(5, l, m);
        v.rdW = hermitian(6, l, m);
      }
      const auto A = DeviatoricComponents(l, v, r);
      for (auto c = 0; c < 5; ++c) {
        if (l < std::abs(spins[c])) continue;
        componentCoefficients[c][SurfaceGrid::SpinIndex(spins[c], l, m)] = A[c];
      }
    }
  }
  auto fields = std::array<std::vector<Complex>, 5>{};
  for (auto c = 0; c < 5; ++c) {
    fields[c] = grid.SynthesiseSpin(spins[c], componentCoefficients[c]);
  }
  auto scale = 0.0;
  for (const auto& f : fields) {
    for (auto value : f) scale = std::max(scale, std::abs(value));
  }
  for (auto k = std::size_t{0}; k < fields[0].size(); k += 5) {
    EXPECT_LT(std::abs(fields[4][k] - std::conj(fields[0][k])), 1e-11 * scale);
    EXPECT_LT(std::abs(fields[3][k] + std::conj(fields[1][k])), 1e-11 * scale);
    EXPECT_LT(std::abs(fields[2][k].imag()), 1e-11 * scale);
  }
}
