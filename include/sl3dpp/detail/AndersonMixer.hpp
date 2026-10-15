#pragma once

/**
 * @file detail/AndersonMixer.hpp
 * @brief Anderson acceleration for fixed-point iterations, shared by the
 * sea level solvers and the within-step lateral iteration. Internal.
 */

#include <algorithm>
#include <span>
#include <vector>

// LAPACK: the small least-squares solve of the mixer.
extern "C" {
void dgelss_(const int* m, const int* n, const int* nrhs, double* a,
             const int* lda, double* b, const int* ldb, double* s,
             const double* rcond, int* rank, double* work, const int* lwork,
             int* info);
}

namespace sl3dpp::detail {

/// Anderson acceleration for a fixed point x = G(x) (Walker and Ni 2011),
/// pyslfp's `AndersonMixer`: ring buffers of iterate and residual
/// differences, combined in the least-squares sense through the small Gram
/// system. Operates on flat real views; complex coefficient vectors pass
/// through as twice-as-long real ones.
class AndersonMixer {
 public:
  explicit AndersonMixer(int memory) : memory_(memory) {}

  std::vector<double> Update(std::span<const double> x,
                             std::span<const double> gx) {
    std::vector<double> next(gx.begin(), gx.end());
    if (memory_ <= 0) return next;
    const auto n = x.size();
    std::vector<double> f(n);
    for (std::size_t i = 0; i < n; ++i) f[i] = gx[i] - x[i];

    if (xPrev_.empty()) {
      dX_.assign(memory_, std::vector<double>(n));
      dF_.assign(memory_, std::vector<double>(n));
      gram_.assign(static_cast<std::size_t>(memory_) * memory_, 0.0);
      xPrev_.assign(x.begin(), x.end());
      fPrev_ = std::move(f);
      return next;
    }

    const int slot = slot_;
    for (std::size_t i = 0; i < n; ++i) {
      dX_[slot][i] = x[i] - xPrev_[i];
      dF_[slot][i] = f[i] - fPrev_[i];
    }
    xPrev_.assign(x.begin(), x.end());
    fPrev_ = f;
    count_ = std::min(count_ + 1, memory_);
    slot_ = (slot + 1) % memory_;
    const int m = count_;

    auto dot = [n](std::span<const double> a, std::span<const double> b) {
      double sum = 0.0;
      for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
      return sum;
    };
    for (int i = 0; i < m; ++i) {
      const double value = dot(dF_[i], dF_[slot]);
      gram_[static_cast<std::size_t>(slot) * memory_ + i] = value;
      gram_[static_cast<std::size_t>(i) * memory_ + slot] = value;
    }

    // gamma = argmin |F gamma - f|, through the Gram system by LAPACK's
    // SVD least squares (singular directions dropped at machine
    // precision, as numpy's lstsq with rcond=None does).
    std::vector<double> a(static_cast<std::size_t>(m) * m);
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < m; ++i) {
        a[static_cast<std::size_t>(j) * m + i] =
            gram_[static_cast<std::size_t>(i) * memory_ + j];
      }
    }
    std::vector<double> gamma(m);
    for (int i = 0; i < m; ++i) gamma[i] = dot(dF_[i], f);
    std::vector<double> singular(m);
    const double rcond = -1.0;
    int rank = 0, info = 0, nrhs = 1;
    int lwork = 12 * m + 64;
    std::vector<double> work(lwork);
    dgelss_(&m, &m, &nrhs, a.data(), &m, gamma.data(), &m, singular.data(),
            &rcond, &rank, work.data(), &lwork, &info);
    if (info != 0) return next;  // fall back to the Picard step

    for (int i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        next[j] -= gamma[i] * (dX_[i][j] + dF_[i][j]);
      }
    }
    return next;
  }

 private:
  int memory_;
  int count_ = 0;
  int slot_ = 0;
  std::vector<double> xPrev_, fPrev_, gram_;
  std::vector<std::vector<double>> dX_, dF_;
};

}  // namespace sl3dpp::detail
