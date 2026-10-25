#pragma once

/**
 * @file detail/AndersonMixer.hpp
 * @brief Anderson acceleration for fixed-point iterations, shared by the
 * sea level solvers and the within-step lateral iteration. Internal.
 */

#include <omp.h>

#include <algorithm>
#include <array>
#include <format>
#include <functional>
#include <span>
#include <stdexcept>
#include <utility>
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
/// through as twice-as-long real ones. The iterate differences are kept
/// only as their sums with the residual differences, the one form the
/// mixed iterate reads: the vectors are large and the update is bound by
/// memory traffic.
///
/// With `parallel`, the sweeps over the vector run threaded on OpenMP's
/// thread count, the sums as per-thread partials added in thread order:
/// reproducible at a given thread count, and at one thread bitwise the
/// serial update. With `sum`, the vector is distributed (each process
/// holding its own part) and the Gram system's entries are summed over
/// the processes, which then mix alike: each update is collective.
class AndersonMixer {
 public:
  explicit AndersonMixer(int memory, bool parallel = false,
                         std::function<void(std::span<double>)> sum = {})
      : memory_(memory), parallel_(parallel), sum_(std::move(sum)) {}

  std::vector<double> Update(std::span<const double> x,
                             std::span<const double> gx) {
    auto next = std::vector<double>(x.begin(), x.end());
    const auto out = std::array<std::span<double>, 1>{next};
    const auto g = std::array<std::span<const double>, 1>{gx};
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
    // Where each segment starts in the concatenation.
    auto offset = std::vector<std::size_t>(x.size() + 1, 0);
    for (auto s = std::size_t{0}; s < x.size(); ++s) {
      if (x[s].size() != gx[s].size()) {
        throw std::invalid_argument(
            std::format("segment {} holds {} values, its image {}", s,
                        x[s].size(), gx[s].size()));
      }
      offset[s + 1] = offset[s] + x[s].size();
    }
    const auto n = offset.back();
    // body(s, j, at) over every value, segment by segment, each thread
    // taking the same static block of every segment.
    auto sweep = [&](auto&& body) {
#pragma omp parallel if (parallel_)
      {
        for (auto s = std::size_t{0}; s < x.size(); ++s) {
          const auto size = x[s].size();
#pragma omp for schedule(static) nowait
          for (auto j = std::size_t{0}; j < size; ++j) {
            body(s, j, offset[s] + j);
          }
        }
      }
    };
    auto takeImage = [&] {
      sweep([&](std::size_t s, std::size_t j, std::size_t) {
        x[s][j] = gx[s][j];
      });
    };
    if (memory_ <= 0) {
      takeImage();
      return;
    }

    if (xPrev_.empty()) {
      dG_.assign(memory_, std::vector<double>(n));
      dF_.assign(memory_, std::vector<double>(n));
      gram_.assign(static_cast<std::size_t>(memory_) * memory_, 0.0);
      xPrev_.resize(n);
      fPrev_.resize(n);
      sweep([&](std::size_t s, std::size_t j, std::size_t at) {
        xPrev_[at] = x[s][j];
        fPrev_[at] = gx[s][j] - x[s][j];
        x[s][j] = gx[s][j];
      });
      return;
    }

    // One sweep forms the new differences, stores the iterate and its
    // residual, and accumulates the new Gram column and the least-squares
    // right-hand side: per thread, then the partials in thread order.
    const int slot = slot_;
    const int m = std::min(count_ + 1, memory_);
    auto dF = std::vector<const double*>(m);
    for (auto i = 0; i < m; ++i) dF[i] = dF_[i].data();
    auto* dGs = dG_[slot].data();
    auto* dFs = dF_[slot].data();
    const auto threads = parallel_ ? omp_get_max_threads() : 1;
    auto partial =
        std::vector<double>(static_cast<std::size_t>(threads) * 2 * m, 0.0);
#pragma omp parallel if (parallel_) num_threads(threads)
    {
      auto* column = partial.data() +
                     static_cast<std::size_t>(omp_get_thread_num()) * 2 * m;
      auto* gamma = column + m;
      for (auto s = std::size_t{0}; s < x.size(); ++s) {
        const auto size = x[s].size();
        const auto xs = x[s];
        const auto gs = gx[s];
#pragma omp for schedule(static) nowait
        for (auto j = std::size_t{0}; j < size; ++j) {
          const auto at = offset[s] + j;
          const auto f = gs[j] - xs[j];
          const auto dx = xs[j] - xPrev_[at];
          dFs[at] = f - fPrev_[at];
          dGs[at] = dx + dFs[at];
          xPrev_[at] = xs[j];
          fPrev_[at] = f;
          for (auto i = 0; i < m; ++i) {
            column[i] += dF[i][at] * dFs[at];
            gamma[i] += dF[i][at] * f;
          }
        }
      }
    }
    for (auto t = 1; t < threads; ++t) {
      const auto* p = partial.data() + static_cast<std::size_t>(t) * 2 * m;
      for (auto i = 0; i < 2 * m; ++i) partial[i] += p[i];
    }
    if (sum_) sum_(std::span(partial).first(static_cast<std::size_t>(2 * m)));
    std::vector<double> column(partial.begin(), partial.begin() + m),
        gamma(partial.begin() + m, partial.begin() + 2 * m);
    count_ = m;
    slot_ = (slot + 1) % memory_;
    for (auto i = 0; i < m; ++i) {
      gram_[static_cast<std::size_t>(slot) * memory_ + i] = column[i];
      gram_[static_cast<std::size_t>(i) * memory_ + slot] = column[i];
    }

    // gamma = argmin |F gamma - f|, through the Gram system by LAPACK's
    // SVD least squares (singular directions dropped at machine
    // precision, as numpy's lstsq with rcond=None does).
    auto a = std::vector<double>(static_cast<std::size_t>(m) * m);
    for (auto j = 0; j < m; ++j) {
      for (auto i = 0; i < m; ++i) {
        a[static_cast<std::size_t>(j) * m + i] =
            gram_[static_cast<std::size_t>(i) * memory_ + j];
      }
    }
    auto singular = std::vector<double>(m);
    const auto rcond = -1.0;
    auto rank = 0, info = 0, nrhs = 1;
    int lwork = 12 * m + 64;
    auto work = std::vector<double>(lwork);
    dgelss_(&m, &m, &nrhs, a.data(), &m, gamma.data(), &m, singular.data(),
            &rcond, &rank, work.data(), &lwork, &info);
    if (info != 0) {  // fall back to the Picard step
      takeImage();
      return;
    }

    // The mixed iterate; x is no longer read, so it takes the result.
    auto dG = std::vector<const double*>(m);
    for (auto i = 0; i < m; ++i) dG[i] = dG_[i].data();
    sweep([&](std::size_t s, std::size_t j, std::size_t at) {
      auto value = gx[s][j];
      for (auto i = 0; i < m; ++i) value -= gamma[i] * dG[i][at];
      x[s][j] = value;
    });
  }

 private:
  int memory_;
  bool parallel_;
  std::function<void(std::span<double>)> sum_;
  int count_ = 0;
  int slot_ = 0;
  std::vector<double> xPrev_, fPrev_, gram_;
  std::vector<std::vector<double>> dG_, dF_;  // dX + dF, and dF
};

}  // namespace sl3dpp::detail
