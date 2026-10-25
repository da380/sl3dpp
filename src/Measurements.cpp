#include "sl3dpp/Measurements.hpp"

// LAPACK: one dense LU solve serves the fit's small systems.
extern "C" {
void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
            int* ipiv, double* b, const int* ldb, int* info);
}

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>
#include <utility>

#include "sl3dpp/ShConventions.hpp"

namespace sl3dpp {

namespace {

/// cos of the great-circle distance from (colat0, lon0).
double CosDistance(double theta, double phi, double colat0, double lon0) {
  return std::cos(theta) * std::cos(colat0) +
         std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
}

/// Solve A x = b for a small dense system, in place; A is destroyed.
void Solve(std::vector<double>& a, std::vector<double>& b, int n, int nrhs,
           std::string_view what) {
  auto pivots = std::vector<int>(static_cast<std::size_t>(n));
  auto info = 0;
  dgesv_(&n, &nrhs, a.data(), &n, pivots.data(), b.data(), &n, &info);
  if (info != 0) {
    throw std::runtime_error(
        std::format("{}: the fit's linear system is singular", what));
  }
}

/// The decay fit's residual and Jacobian at q = (a, b, s[, tm...]),
/// whitened; the penalty rows follow the data rows when the times are
/// free. Column-major Jacobian, M rows by nq columns.
struct FitProblem {
  std::span<const double> t;
  std::span<const double> y;
  std::vector<double> wd;
  std::vector<double> wt;  // empty when the times are pinned
  bool free_times = false;

  [[nodiscard]] std::size_t Count() const { return t.size(); }
  [[nodiscard]] std::size_t Parameters() const {
    return 3 + (free_times ? Count() : 0);
  }
  [[nodiscard]] std::size_t Rows() const {
    return Count() + (free_times ? Count() : 0);
  }

  [[nodiscard]] std::vector<double> Residual(
      const std::vector<double>& q) const {
    const auto n = Count();
    auto out = std::vector<double>(Rows());
    for (auto i = std::size_t{0}; i < n; ++i) {
      const auto tm = free_times ? q[3 + i] : t[i];
      out[i] = wd[i] * (q[0] * std::exp(-q[2] * tm) + q[1] - y[i]);
      if (free_times) out[n + i] = wt[i] * (tm - t[i]);
    }
    return out;
  }

  [[nodiscard]] std::vector<double> Jacobian(
      const std::vector<double>& q) const {
    const auto n = Count();
    const auto rows = Rows();
    const auto nq = Parameters();
    auto out = std::vector<double>(rows * nq, 0.0);
    auto at = [&out, rows](std::size_t i, std::size_t j) -> double& {
      return out[j * rows + i];
    };
    for (auto i = std::size_t{0}; i < n; ++i) {
      const auto tm = free_times ? q[3 + i] : t[i];
      const auto e = std::exp(-q[2] * tm);
      at(i, 0) = wd[i] * e;
      at(i, 1) = wd[i];
      at(i, 2) = -wd[i] * q[0] * tm * e;
      if (free_times) {
        at(i, 3 + i) = -wd[i] * q[0] * q[2] * e;
        at(n + i, 3 + i) = wt[i];
      }
    }
    return out;
  }
};

/// 0.5 r.r, the cost scipy's least_squares reports.
double Cost(const std::vector<double>& r) {
  auto out = 0.0;
  for (const auto v : r) out += v * v;
  return 0.5 * out;
}

/// A variable-projection start: for each trial rate the amplitude and
/// offset are the linear least-squares pair, and the best trial seeds
/// the full fit.
std::array<double, 3> InitialGuess(std::span<const double> t,
                                   std::span<const double> y) {
  const auto n = t.size();
  const auto span = t.back() - t.front();
  auto best = std::array<double, 4>{};  // rss, a, b, s
  auto first = true;
  const auto s0 = 0.1 / span;
  const auto s1 = 30.0 / span;
  for (auto k = 0; k < 60; ++k) {
    const auto s = s0 * std::pow(s1 / s0, k / 59.0);
    auto ee = 0.0, e1 = 0.0, ey = 0.0, sy = 0.0;
    for (auto i = std::size_t{0}; i < n; ++i) {
      const auto e = std::exp(-s * (t[i] - t[0]));
      ee += e * e;
      e1 += e;
      ey += e * y[i];
      sy += y[i];
    }
    const auto det = ee * static_cast<double>(n) - e1 * e1;
    if (det == 0.0) continue;
    const auto c0 = (ey * static_cast<double>(n) - e1 * sy) / det;
    const auto c1 = (ee * sy - e1 * ey) / det;
    auto rss = 0.0;
    for (auto i = std::size_t{0}; i < n; ++i) {
      const auto r = c0 * std::exp(-s * (t[i] - t[0])) + c1 - y[i];
      rss += r * r;
    }
    if (first || rss < best[0]) {
      best = {rss, c0 * std::exp(s * t[0]), c1, s};
      first = false;
    }
  }
  return {best[1], best[2], best[3]};
}

/// Levenberg-Marquardt on the whitened residual, to stationarity; the
/// caller's exactness check on the Newton step is the gate behind it.
std::vector<double> Minimise(const FitProblem& problem,
                             std::vector<double> q) {
  const auto rows = problem.Rows();
  const auto nq = problem.Parameters();
  const auto size = static_cast<int>(nq);
  auto r = problem.Residual(q);
  auto cost = Cost(r);
  auto lambda = 1e-6;
  for (auto iteration = 0; iteration < 400; ++iteration) {
    const auto jac = problem.Jacobian(q);
    // the normal matrix and gradient
    auto normal = std::vector<double>(nq * nq);
    auto gradient = std::vector<double>(nq);
    for (auto j = std::size_t{0}; j < nq; ++j) {
      for (auto k = j; k < nq; ++k) {
        auto sum = 0.0;
        for (auto i = std::size_t{0}; i < rows; ++i) {
          sum += jac[j * rows + i] * jac[k * rows + i];
        }
        normal[j * nq + k] = sum;
        normal[k * nq + j] = sum;
      }
      auto sum = 0.0;
      for (auto i = std::size_t{0}; i < rows; ++i) {
        sum += jac[j * rows + i] * r[i];
      }
      gradient[j] = sum;
    }
    auto largest = 0.0;
    for (const auto g : gradient) largest = std::max(largest, std::abs(g));
    if (largest < 1e-14 * std::max(1.0, cost)) break;
    auto stepped = false;
    for (auto attempt = 0; attempt < 60 && !stepped; ++attempt) {
      auto damped = normal;
      for (auto j = std::size_t{0}; j < nq; ++j) {
        damped[j * nq + j] += lambda * std::max(normal[j * nq + j], 1e-300);
      }
      auto step = gradient;
      try {
        Solve(damped, step, size, 1, "decay fit");
      } catch (const std::runtime_error&) {
        lambda *= 10.0;
        continue;
      }
      auto trial = q;
      for (auto j = std::size_t{0}; j < nq; ++j) trial[j] -= step[j];
      const auto trial_r = problem.Residual(trial);
      const auto trial_cost = Cost(trial_r);
      if (trial_cost < cost) {
        q = std::move(trial);
        r = trial_r;
        cost = trial_cost;
        lambda = std::max(lambda * 0.3, 1e-14);
        stepped = true;
      } else {
        lambda *= 10.0;
      }
    }
    if (!stepped) break;
  }
  return q;
}

}  // namespace

std::vector<double> CapMeanDual(const SurfaceGrid& grid, double a,
                                GeographicPoint site, double width) {
  if (!(width > 0.0)) {
    throw std::invalid_argument("a cap's width is a positive angle in radians");
  }
  const auto colat0 = site.CoLatitude();
  const auto lon0 = site.Longitude();
  auto dual = grid.FromFunction([&](double theta, double phi) {
    const auto d =
        std::acos(std::clamp(CosDistance(theta, phi, colat0, lon0), -1.0, 1.0));
    return std::exp(-0.5 * (d / width) * (d / width));
  });
  const auto norm = a * a * grid.Integrate(dual);
  for (auto& v : dual) v /= norm;
  return dual;
}

std::vector<double> PointDual(const SurfaceGrid& grid, double a,
                              GeographicPoint site) {
  const auto colat0 = site.CoLatitude();
  const auto lon0 = site.Longitude();
  const auto lmax = grid.LMax();
  return grid.FromFunction([&](double theta, double phi) {
    const auto x = std::clamp(CosDistance(theta, phi, colat0, lon0), -1.0, 1.0);
    // sum_l (2l + 1) P_l(x), by the upward recurrence
    auto before = 1.0, current = x;
    auto sum = 1.0 + (lmax >= 1 ? 3.0 * x : 0.0);
    for (auto l = 2; l <= lmax; ++l) {
      const auto next =
          ((2.0 * l - 1.0) * x * current - (l - 1.0) * before) / l;
      sum += (2.0 * l + 1.0) * next;
      before = std::exchange(current, next);
    }
    return sum / (4.0 * std::numbers::pi * a * a);
  });
}

std::vector<double> CoefficientDual(const SurfaceGrid& grid, double a, int l,
                                    int m) {
  if (l < 0 || l > grid.LMax() || std::abs(m) > l) {
    throw std::invalid_argument(
        std::format("({}, {}) is not a degree and order of a grid of band {}",
                    l, m, grid.LMax()));
  }
  auto coefficients = grid.ZeroCoefficients();
  const auto pair = m >= 0 ? RealPair{1.0, 0.0} : RealPair{0.0, 1.0};
  coefficients[SurfaceGrid::Index(l, std::abs(m))] =
      ComplexFromRealPair(pair, std::abs(m));
  auto dual = grid.Synthesise(coefficients);
  for (auto& v : dual) v /= a * a;
  return dual;
}

double PairedValue(const SurfaceGrid& grid, double a,
                   std::span<const double> dual,
                   std::span<const double> field) {
  if (dual.size() != grid.FieldSize() || field.size() != grid.FieldSize()) {
    throw std::invalid_argument(
        "the dual and the field live on the grid's points");
  }
  auto product = std::vector<double>(dual.size());
  for (auto i = std::size_t{0}; i < dual.size(); ++i) {
    product[i] = dual[i] * field[i];
  }
  return a * a * grid.Integrate(product);
}

Measurement FieldMeasurement(FieldChannel channel, std::vector<double> dual) {
  auto out = Measurement{};
  switch (channel) {
    case FieldChannel::SeaLevel:
      out.seaLevelDual = std::move(dual);
      break;
    case FieldChannel::Displacement:
      out.displacementDual = std::move(dual);
      break;
    case FieldChannel::Potential:
      out.potentialDual = std::move(dual);
      break;
  }
  return out;
}

Measurement RotationMeasurement(AngularVelocityComponent component) {
  auto out = Measurement{};
  out.angularVelocityDual[static_cast<std::size_t>(component)] = 1.0;
  return out;
}

DecayTimeFit FitDecayTime(std::span<const double> times,
                          std::span<const double> values,
                          const DecayTimeOptions& options) {
  const auto n = times.size();
  if (n < 4 || values.size() != n) {
    throw std::invalid_argument(
        "times and values must match, with at least four entries");
  }
  for (auto i = std::size_t{1}; i < n; ++i) {
    if (!(times[i] > times[i - 1])) {
      throw std::invalid_argument("times must be increasing");
    }
  }
  if (!(options.value_uncertainty > 0.0) || options.time_uncertainty < 0.0) {
    throw std::invalid_argument(
        "the value uncertainty must be positive and the time uncertainty "
        "non-negative");
  }
  // Only the ratio of the uncertainties shapes the fit, so one given in
  // the wrong unit changes the answer silently; these bounds catch the
  // slips that leave the fit meaningless.
  const auto span = times.back() - times.front();
  if (options.time_uncertainty >= span) {
    throw std::invalid_argument(std::format(
        "the time uncertainty ({:.3g}) is not smaller than the window's span "
        "({:.3g}): the fitted times would be free; is it in the times' unit?",
        options.time_uncertainty, span));
  }
  const auto [low, high] = std::ranges::minmax(values);
  if (options.value_uncertainty >= high - low) {
    throw std::invalid_argument(std::format(
        "the value uncertainty ({:.3g}) is not smaller than the series' range "
        "({:.3g}): the series constrains no decay at that uncertainty; is it "
        "in the values' unit?",
        options.value_uncertainty, high - low));
  }
  const auto free_times = options.time_uncertainty > 0.0;
  auto problem = FitProblem{
      .t = times,
      .y = values,
      .wd = std::vector<double>(n, 1.0 / options.value_uncertainty),
      .wt = free_times
                ? std::vector<double>(n, 1.0 / options.time_uncertainty)
                : std::vector<double>{},
      .free_times = free_times};

  const auto [a0, b0, s0] = InitialGuess(times, values);
  auto q = std::vector<double>{a0, b0, s0};
  if (free_times) q.insert(q.end(), times.begin(), times.end());
  q = Minimise(problem, std::move(q));

  const auto a = q[0];
  const auto b = q[1];
  const auto s = q[2];
  auto tm = std::vector<double>(times.begin(), times.end());
  if (free_times) tm.assign(q.begin() + 3, q.end());
  if (!(s > 0.0) || s * (times.back() - times.front()) < 1e-2) {
    throw std::invalid_argument(
        "the window does not constrain a decay: the fitted rate is not "
        "positive and well above the window's inverse span");
  }

  // the exact sensitivity: stationarity gives
  // delta q = -H^{-1} (grad_q grad_d chi) delta d, with H the FULL
  // Hessian of the misfit, its second-order residual terms included
  const auto nq = problem.Parameters();
  const auto& wd = problem.wd;
  auto e = std::vector<double>(n);
  auto r = std::vector<double>(n);
  for (auto i = std::size_t{0}; i < n; ++i) {
    e[i] = std::exp(-s * tm[i]);
    r[i] = wd[i] * (a * e[i] + b - values[i]);
  }
  auto df = std::vector<double>(n * nq, 0.0);  // row-major n by nq
  for (auto i = std::size_t{0}; i < n; ++i) {
    df[i * nq + 0] = wd[i] * e[i];
    df[i * nq + 1] = wd[i];
    df[i * nq + 2] = -wd[i] * a * tm[i] * e[i];
    if (free_times) df[i * nq + 3 + i] = -wd[i] * a * s * e[i];
  }
  auto h = std::vector<double>(nq * nq, 0.0);  // symmetric; order moot
  for (auto j = std::size_t{0}; j < nq; ++j) {
    for (auto k = j; k < nq; ++k) {
      auto sum = 0.0;
      for (auto i = std::size_t{0}; i < n; ++i) {
        sum += df[i * nq + j] * df[i * nq + k];
      }
      h[j * nq + k] = sum;
      h[k * nq + j] = sum;
    }
  }
  auto add = [&h, nq](std::size_t j, std::size_t k, double value) {
    h[j * nq + k] += value;
    if (j != k) h[k * nq + j] += value;
  };
  for (auto i = std::size_t{0}; i < n; ++i) {
    add(0, 2, -r[i] * wd[i] * tm[i] * e[i]);
    add(2, 2, r[i] * wd[i] * a * tm[i] * tm[i] * e[i]);
    if (free_times) {
      add(0, 3 + i, -r[i] * wd[i] * s * e[i]);
      add(2, 3 + i, r[i] * wd[i] * a * e[i] * (s * tm[i] - 1.0));
      add(3 + i, 3 + i,
          r[i] * wd[i] * a * s * s * e[i] +
              1.0 / (options.time_uncertainty * options.time_uncertainty));
    }
  }
  auto gradient = std::vector<double>(nq, 0.0);
  for (auto j = std::size_t{0}; j < nq; ++j) {
    for (auto i = std::size_t{0}; i < n; ++i) {
      gradient[j] += df[i * nq + j] * r[i];
    }
  }
  if (free_times) {
    for (auto i = std::size_t{0}; i < n; ++i) {
      gradient[3 + i] +=
          (tm[i] - times[i]) /
          (options.time_uncertainty * options.time_uncertainty);
    }
  }
  const auto size = static_cast<int>(nq);
  auto newton = gradient;
  auto h_copy = h;
  Solve(h_copy, newton, size, 1, "decay fit");
  if (std::abs(newton[2]) > 1e-8 * s) {
    throw std::runtime_error(
        "the fit did not reach stationarity; check that the two "
        "uncertainties sit near the scales of the series and times");
  }
  // one solve against e_2 of the symmetric H gives the rate's row of
  // H^{-1}, and with it d s / d d_j = wd_j (z . df_j)
  auto z = std::vector<double>(nq, 0.0);
  z[2] = 1.0;
  h_copy = h;
  Solve(h_copy, z, size, 1, "decay fit");
  auto weights = std::vector<double>(n);
  for (auto i = std::size_t{0}; i < n; ++i) {
    auto dot = 0.0;
    for (auto j = std::size_t{0}; j < nq; ++j) dot += z[j] * df[i * nq + j];
    weights[i] = -wd[i] * dot / (s * s);
  }

  auto misfit = r;
  if (free_times) {
    for (auto i = std::size_t{0}; i < n; ++i) {
      misfit.push_back((tm[i] - times[i]) / options.time_uncertainty);
    }
  }
  return DecayTimeFit{.decay_time = 1.0 / s,
                      .amplitude = a,
                      .offset = b,
                      .misfit = Cost(misfit),
                      .model_times = std::move(tm),
                      .weights = std::move(weights)};
}

std::map<std::size_t, Measurement> DecayTimeMeasurements(
    std::span<const std::size_t> nodes, std::span<const double> times,
    std::span<const double> series, FieldChannel channel,
    std::span<const double> dual, const DecayTimeOptions& options,
    DecayTimeFit* fit) {
  if (nodes.size() != times.size() || series.size() != times.size()) {
    throw std::invalid_argument(
        "nodes, times and series must hold one entry per sample");
  }
  auto fitted = FitDecayTime(times, series, options);
  auto out = std::map<std::size_t, Measurement>{};
  for (auto j = std::size_t{0}; j < nodes.size(); ++j) {
    auto weighted = std::vector<double>(dual.begin(), dual.end());
    for (auto& v : weighted) v *= fitted.weights[j];
    if (!out.emplace(nodes[j], FieldMeasurement(channel, std::move(weighted)))
             .second) {
      throw std::invalid_argument("a node appears twice among the samples");
    }
  }
  if (fit != nullptr) *fit = std::move(fitted);
  return out;
}

}  // namespace sl3dpp
