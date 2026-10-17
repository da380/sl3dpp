#include "sl3dpp/ToroidalOperator.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

// LAPACK banded Cholesky (the toroidal blocks are SPD at l >= 2, and at
// l = 1 once one dof per solid region is pinned).
extern "C" {
void dpbtrf_(const char* uplo, const int* n, const int* kd, double* ab,
             const int* ldab, int* info);
void dpbtrs_(const char* uplo, const int* n, const int* kd, const int* nrhs,
             const double* ab, const int* ldab, double* b, const int* ldb,
             int* info);
}

namespace sl3dpp {

ToroidalOperator::ToroidalOperator(const Material& material, int l,
                                   Options options)
    : material_(&material), l_(l), pinFraction_(options.pin_fraction) {
  if (l < 1) {
    throw std::invalid_argument(
        "the toroidal system exists for degrees one and above (l = 0 has no "
        "toroidal motion)");
  }
  const auto& mesh = material.Mesh();
  firstElement_ = mesh.StartElement(l, options.truncation, options.eps);
  kd_ = mesh.NGll() - 1;
  BuildDofMap();
  Assemble(options.point_shear);
  Factorise();
}

void ToroidalOperator::BuildDofMap() {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  dof_.assign(mesh.NodeCount(), -1);
  ndof_ = 0;
  // The dofs run up through each solid region in turn, so a region (a run
  // of solid elements between fluid ones) is a contiguous dof range.
  auto inRegion = false;
  for (auto e = firstElement_; e < mesh.ElementCount(); ++e) {
    if (material_->IsFluid(e)) {
      if (inRegion) regions_.back().second = ndof_;
      inRegion = false;
      continue;
    }
    if (!inRegion) regions_.push_back({ndof_, ndof_});
    inRegion = true;
    for (auto i = 0; i < n; ++i) {
      const auto gn = mesh.GlobalIndex(e, i);
      if (dof_[gn] < 0) dof_[gn] = ndof_++;
    }
  }
  if (inRegion) regions_.back().second = ndof_;
  if (ndof_ == 0) {
    throw std::invalid_argument(
        "the degree's element subset holds no solid element");
  }
  // Degree 1: the gauge, one pinned dof per region, a fraction of the way
  // down from the region's top (never its bottom node: the rigid mode
  // vanishes at the centre).
  if (l_ == 1) {
    if (!(pinFraction_ >= 0.0 && pinFraction_ < 1.0)) {
      throw std::invalid_argument(std::format(
          "the pin fraction must lie in [0, 1), not {}", pinFraction_));
    }
    for (const auto& [first, end] : regions_) {
      const auto below = static_cast<int>(
          std::floor(pinFraction_ * static_cast<double>(end - first - 1)));
      pinned_.push_back(end - 1 - below);
    }
  }
}

void ToroidalOperator::Assemble(const PointShear* pointShear) {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto k2 = l_ * (l_ + 1.0);
  const auto& moduli = material_->Moduli();
  const auto w = mesh.Weights();

  band_.assign(static_cast<std::size_t>(kd_ + 1) * ndof_, 0.0);
  for (auto e = firstElement_; e < mesh.ElementCount(); ++e) {
    if (material_->IsFluid(e)) continue;
    const auto base = static_cast<std::size_t>(e) * n;
    const auto jac = mesh.Jacobian(e);
    for (auto i = 0; i < n; ++i) {
      const auto gi = dof_[mesh.GlobalIndex(e, i)];
      for (auto j = i; j < n; ++j) {
        const auto gj = dof_[mesh.GlobalIndex(e, j)];
        // k^2 L (r d_r W - W)^2 by GLL quadrature: the operator rows are
        // s(q, i) = r_q D(q, i) / jac - delta_qi.
        auto value = 0.0;
        for (auto q = 0; q < n; ++q) {
          const auto rq = mesh.Radius(e, q);
          const auto si =
              rq * mesh.Derivative(q, i) / jac - (q == i ? 1.0 : 0.0);
          const auto sj =
              rq * mesh.Derivative(q, j) / jac - (q == j ? 1.0 : 0.0);
          value += w[q] * jac * k2 * moduli.L[base + q] * si * sj;
        }
        if (i == j) {
          value += w[i] * jac * k2 * (k2 - 2.0) * moduli.N[base + i];
        }
        const auto [lo, hi] = std::minmax(gi, gj);
        Band(lo, hi) += value;
      }
    }
  }

  // The point shear (MemoryPoints.hpp): at each memory point,
  // jac w_p s_p (k^2 t^2 + k^2 (k^2 - 2) w^2) with t = r d_r W - W and
  // w = W there, which with GLL points is exactly the assembly above with
  // L and N raised by s.
  if (pointShear != nullptr) {
    const auto& points = *pointShear->points;
    for (auto e = firstElement_; e < mesh.ElementCount(); ++e) {
      if (material_->IsFluid(e)) continue;
      const auto jac = mesh.Jacobian(e);
      for (auto p = 0; p < points.count; ++p) {
        const auto s =
            pointShear->modulus[static_cast<std::size_t>(e) * points.count + p];
        if (s == 0.0) continue;
        auto rp = 0.0;
        for (auto i = 0; i < n; ++i)
          rp += points.Value(p, i) * mesh.Radius(e, i);
        const auto weight = points.weight[p] * jac * s;
        for (auto i = 0; i < n; ++i) {
          const auto gi = dof_[mesh.GlobalIndex(e, i)];
          const auto ti =
              rp * points.Derivative(p, i) / jac - points.Value(p, i);
          const auto wi = points.Value(p, i);
          for (auto j = i; j < n; ++j) {
            const auto gj = dof_[mesh.GlobalIndex(e, j)];
            const auto tj =
                rp * points.Derivative(p, j) / jac - points.Value(p, j);
            const auto wj = points.Value(p, j);
            const auto value =
                weight * (k2 * ti * tj + k2 * (k2 - 2.0) * wi * wj);
            if (value == 0.0) continue;
            const auto [lo, hi] = std::minmax(gi, gj);
            Band(lo, hi) += value;
          }
        }
      }
    }
  }
}

void ToroidalOperator::Factorise() {
  const auto n = ndof_;
  const auto ldab = kd_ + 1;
  scale_.resize(n);
  for (auto j = 0; j < n; ++j) {
    const auto d = band_[static_cast<std::size_t>(j) * ldab + kd_];
    if (!(d > 0.0)) {
      throw std::runtime_error(std::format(
          "toroidal degree {}: a non-positive diagonal at dof {}", l_, j));
    }
    scale_[j] = 1.0 / std::sqrt(d);
  }
  factor_.assign(static_cast<std::size_t>(ldab) * n, 0.0);
  for (auto j = 0; j < n; ++j) {
    for (auto i = std::max(0, j - kd_); i <= j; ++i) {
      factor_[static_cast<std::size_t>(j) * ldab + (kd_ + i - j)] =
          band_[static_cast<std::size_t>(j) * ldab + (kd_ + i - j)] *
          scale_[i] * scale_[j];
    }
  }
  // The gauge: a pinned dof's row and column become the identity.
  for (const auto p : pinned_) {
    for (auto j = std::max(0, p - kd_); j <= std::min(n - 1, p + kd_); ++j) {
      const auto [i, k] = std::minmax(p, j);
      factor_[static_cast<std::size_t>(k) * ldab + (kd_ + i - k)] =
          i == k ? 1.0 : 0.0;
    }
  }
  auto info = 0;
  dpbtrf_("U", &n, &kd_, factor_.data(), &ldab, &info);
  if (info != 0) {
    throw std::runtime_error(std::format(
        "toroidal degree {}: Cholesky failed (info = {})", l_, info));
  }
}

std::vector<double> ToroidalOperator::Solve(std::span<const double> b,
                                            int nrhs) const {
  const auto n = ndof_;
  if (static_cast<int>(b.size()) != n * nrhs) {
    throw std::invalid_argument(std::format(
        "expected {} x {} right-hand-side values, got {}", n, nrhs, b.size()));
  }
  auto x = std::vector<double>(b.begin(), b.end());
  for (auto k = 0; k < nrhs; ++k) {
    for (auto j = 0; j < n; ++j) {
      x[static_cast<std::size_t>(k) * n + j] *= scale_[j];
    }
    for (const auto p : pinned_) x[static_cast<std::size_t>(k) * n + p] = 0.0;
  }
  const auto ldab = kd_ + 1;
  auto info = 0;
  dpbtrs_("U", &n, &kd_, &nrhs, factor_.data(), &ldab, x.data(), &n, &info);
  if (info != 0) {
    throw std::runtime_error(
        std::format("toroidal degree {}: solve failed (info = {})", l_, info));
  }
  for (auto k = 0; k < nrhs; ++k) {
    for (auto j = 0; j < n; ++j) {
      x[static_cast<std::size_t>(k) * n + j] *= scale_[j];
    }
  }
  return x;
}

std::vector<double> ToroidalOperator::Apply(std::span<const double> x,
                                            int nrhs) const {
  const auto n = ndof_;
  if (static_cast<int>(x.size()) != n * nrhs) {
    throw std::invalid_argument(
        std::format("expected {} x {} values, got {}", n, nrhs, x.size()));
  }
  const auto ldab = kd_ + 1;
  auto y = std::vector<double>(x.size(), 0.0);
  for (auto k = 0; k < nrhs; ++k) {
    const auto* xk = x.data() + static_cast<std::size_t>(k) * n;
    auto* yk = y.data() + static_cast<std::size_t>(k) * n;
    for (auto j = 0; j < n; ++j) {
      for (auto i = std::max(0, j - kd_); i <= j; ++i) {
        const auto a =
            band_[static_cast<std::size_t>(j) * ldab + (kd_ + i - j)];
        yk[i] += a * xk[j];
        if (i != j) yk[j] += a * xk[i];
      }
    }
  }
  return y;
}

std::vector<double> ToroidalOperator::RigidRotation(int region) const {
  if (region < 0 || region >= RegionCount()) {
    throw std::invalid_argument(std::format("no solid region {}", region));
  }
  const auto [first, end] = regions_[static_cast<std::size_t>(region)];
  const auto& mesh = material_->Mesh();
  auto out = std::vector<double>(ndof_, 0.0);
  for (auto e = firstElement_; e < mesh.ElementCount(); ++e) {
    for (auto i = 0; i < mesh.NGll(); ++i) {
      const auto d = dof_[mesh.GlobalIndex(e, i)];
      if (d >= first && d < end) out[d] = mesh.Radius(e, i);
    }
  }
  return out;
}

int ToroidalOperator::SurfaceDof() const {
  const auto& mesh = material_->Mesh();
  return dof_[mesh.GlobalIndex(mesh.ElementCount() - 1, mesh.NGll() - 1)];
}

std::vector<double> ToroidalOperator::Expand(std::span<const double> x) const {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  std::vector<double> out(static_cast<std::size_t>(mesh.ElementCount()) * n,
                          0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto i = 0; i < n; ++i) {
      const auto d = dof_[mesh.GlobalIndex(e, i)];
      if (d >= 0) out[static_cast<std::size_t>(e) * n + i] = x[d];
    }
  }
  return out;
}

}  // namespace sl3dpp
