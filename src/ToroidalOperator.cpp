#include "sl3dpp/ToroidalOperator.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

// LAPACK banded Cholesky (the toroidal blocks are SPD at l >= 2).
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
    : material_(&material), l_(l) {
  if (l < 2) {
    throw std::invalid_argument(
        "the toroidal system exists for degrees two and above (l = 0 has no "
        "toroidal motion; l = 1 is the rigid-rotation null space)");
  }
  const auto& mesh = material.Mesh();
  firstElement_ = mesh.StartElement(l, options.truncation, options.eps);
  kd_ = mesh.NGll() - 1;
  BuildDofMap();
  Assemble();
  Factorise();
}

void ToroidalOperator::BuildDofMap() {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  dof_.assign(mesh.NodeCount(), -1);
  ndof_ = 0;
  for (auto e = firstElement_; e < mesh.ElementCount(); ++e) {
    if (material_->IsFluid(e)) continue;
    for (auto i = 0; i < n; ++i) {
      const auto gn = mesh.GlobalIndex(e, i);
      if (dof_[gn] < 0) dof_[gn] = ndof_++;
    }
  }
  if (ndof_ == 0) {
    throw std::invalid_argument(
        "the degree's element subset holds no solid element");
  }
}

void ToroidalOperator::Assemble() {
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
  std::vector<double> x(b.begin(), b.end());
  for (auto k = 0; k < nrhs; ++k) {
    for (auto j = 0; j < n; ++j) {
      x[static_cast<std::size_t>(k) * n + j] *= scale_[j];
    }
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
