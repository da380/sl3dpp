#pragma once

/**
 * @file detail/AndersonMixer.hpp
 * @brief Anderson acceleration for fixed-point iterations, shared by the
 * sea level solvers and the within-step lateral iteration. Internal.
 */

#include <algorithm>
#include <array>
#include <format>
#include <span>
#include <stdexcept>
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
    std::vector<double> next(x.begin(), x.end());
    const std::array<std::span<double>, 1> out{next};
    const std::array<std::span<const double>, 1> g{gx};
    UpdateInPlace(out, g);
    return next;
  }

  /// The same update for a vector held as several contiguous segments,
  /// taken in order (a tensor's component arrays, say), overwriting x with
  /// the mixed iterate. Nothing is copied or allocated per call, and the
  /// result is bitwise Update's on the concatenation: every sum runs over
  /// the same terms in the same order.
  void UpdateInPlace(std::span<const std::span<double>> x,
                     std::span<const std::span<const double>> gx) {
    if (x.size() != gx.size()) {
      throw std::invalid_argument("the iterate and its image differ in shape");
    }
    std::size_t n = 0;
    for (std::size_t s = 0; s < x.size(); ++s) {
      if (x[s].size() != gx[s].size()) {
        throw std::invalid_argument(
            std::format("segment {} holds {} values, its image {}", s,
                        x[s].size(), gx[s].size()));
      }
      n += x[s].size();
    }
    auto takeImage = [&] {
      for (std::size_t s = 0; s < x.size(); ++s) {
        std::ranges::copy(gx[s], x[s].begin());
      }
    };
    if (memory_ <= 0) {
      takeImage();
      return;
    }

    if (xPrev_.empty()) {
      dX_.assign(memory_, std::vector<double>(n));
      dF_.assign(memory_, std::vector<double>(n));
      gram_.assign(static_cast<std::size_t>(memory_) * memory_, 0.0);
      xPrev_.resize(n);
      fPrev_.resize(n);
      std::size_t at = 0;
      for (std::size_t s = 0; s < x.size(); ++s) {
        for (std::size_t j = 0; j < x[s].size(); ++j, ++at) {
          xPrev_[at] = x[s][j];
          fPrev_[at] = gx[s][j] - x[s][j];
        }
      }
      takeImage();
      return;
    }

    // One sweep forms the new differences, stores the iterate and its
    // residual, and accumulates the new Gram column and the least-squares
    // right-hand side; each sum takes its terms in index order.
    const int slot = slot_;
    const int m = std::min(count_ + 1, memory_);
    std::vector<double> column(m, 0.0), gamma(m, 0.0);
    std::vector<const double*> dF(m);
    for (int i = 0; i < m; ++i) dF[i] = dF_[i].data();
    auto* dXs = dX_[slot].data();
    auto* dFs = dF_[slot].data();
    std::size_t at = 0;
    for (std::size_t s = 0; s < x.size(); ++s) {
      for (std::size_t j = 0; j < x[s].size(); ++j, ++at) {
        const double f = gx[s][j] - x[s][j];
        dXs[at] = x[s][j] - xPrev_[at];
        dFs[at] = f - fPrev_[at];
        xPrev_[at] = x[s][j];
        fPrev_[at] = f;
        for (int i = 0; i < m; ++i) {
          column[i] += dF[i][at] * dFs[at];
          gamma[i] += dF[i][at] * f;
        }
      }
    }
    count_ = m;
    slot_ = (slot + 1) % memory_;
    for (int i = 0; i < m; ++i) {
      gram_[static_cast<std::size_t>(slot) * memory_ + i] = column[i];
      gram_[static_cast<std::size_t>(i) * memory_ + slot] = column[i];
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
    std::vector<double> singular(m);
    const double rcond = -1.0;
    int rank = 0, info = 0, nrhs = 1;
    int lwork = 12 * m + 64;
    std::vector<double> work(lwork);
    dgelss_(&m, &m, &nrhs, a.data(), &m, gamma.data(), &m, singular.data(),
            &rcond, &rank, work.data(), &lwork, &info);
    if (info != 0) {  // fall back to the Picard step
      takeImage();
      return;
    }

    // The mixed iterate; x is no longer read, so it takes the result.
    std::vector<const double*> dX(m);
    for (int i = 0; i < m; ++i) dX[i] = dX_[i].data();
    at = 0;
    for (std::size_t s = 0; s < x.size(); ++s) {
      for (std::size_t j = 0; j < x[s].size(); ++j, ++at) {
        double value = gx[s][j];
        for (int i = 0; i < m; ++i) {
          value -= gamma[i] * (dX[i][at] + dF[i][at]);
        }
        x[s][j] = value;
      }
    }
  }

 private:
  int memory_;
  int count_ = 0;
  int slot_ = 0;
  std::vector<double> xPrev_, fPrev_, gram_;
  std::vector<std::vector<double>> dX_, dF_;
};

}  // namespace sl3dpp::detail
