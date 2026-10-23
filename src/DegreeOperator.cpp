#include "sl3dpp/DegreeOperator.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>

// LAPACK: banded Cholesky and banded LU, double precision.
extern "C" {
void dpbtrf_(const char* uplo, const int* n, const int* kd, double* ab,
             const int* ldab, int* info);
void dpbtrs_(const char* uplo, const int* n, const int* kd, const int* nrhs,
             const double* ab, const int* ldab, double* b, const int* ldb,
             int* info);
void dgbtrf_(const int* m, const int* n, const int* kl, const int* ku,
             double* ab, const int* ldab, int* ipiv, int* info);
void dgbtrs_(const char* trans, const int* n, const int* kl, const int* ku,
             const int* nrhs, const double* ab, const int* ldab,
             const int* ipiv, double* b, const int* ldb, int* info);
}

namespace sl3dpp {

namespace {
constexpr auto kPi = std::numbers::pi;

/// r^2 (d rho / dr) / g, zero where g vanishes (the centre).
double Stratification(double r, double drho, double g) {
  return g > 0.0 ? r * r * drho / g : 0.0;
}
}  // namespace

DegreeOperator::DegreeOperator(const Material& material, int l, Options options)
    : material_(&material), l_(l), gravitating_(options.gravitating) {
  if (l < 0) {
    throw std::invalid_argument("degree must be non-negative");
  }
  const auto& mesh = material.Mesh();
  if (!gravitating_) {
    if (l == 1) {
      throw std::invalid_argument(
          "the non-gravitating problem excludes degree 1: a surface "
          "traction there has a net force against the rigid translations");
    }
    for (int e = 0; e < mesh.ElementCount(); ++e) {
      if (material.IsFluid(e)) {
        throw std::invalid_argument(
            "the non-gravitating problem takes a solid model: a fluid "
            "region has no restoring force without gravity");
      }
    }
  }
  firstElement_ =
      l > 0 ? mesh.StartElement(l, options.truncation, options.eps) : 0;
  const auto ncomp = gravitating_ ? 3 : (l == 0 ? 1 : 2);
  kd_ = ncomp * (mesh.NGll() - 1) + ncomp - 1;
  BuildDofMap();
  Assemble(options.point_shear);
  Factorise();
}

// -- numbering ----------------------------------------------------------------

void DegreeOperator::BuildDofMap() {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto nglob = mesh.NodeCount();
  const int g0 = mesh.GlobalIndex(firstElement_, 0);

  auto solid = std::vector<char>(nglob, 0);
  for (int e = firstElement_; e < mesh.ElementCount(); ++e) {
    if (l_ == 0 || !material_->IsFluid(e)) {
      for (auto i = 0; i < n; ++i) solid[mesh.GlobalIndex(e, i)] = 1;
    }
  }

  dof_.assign(static_cast<std::size_t>(nglob) * 3, -1);
  ndof_ = 0;
  static constexpr int kAll[] = {0, 1, 2};
  static constexpr int kNoTangential[] = {0, 2};  // l = 0
  static constexpr int kDisplacement[] = {0, 1};  // non-gravitating
  static constexpr int kRadial[] = {0};           // non-gravitating l = 0
  const std::span<const int> components =
      gravitating_ ? (l_ == 0 ? std::span<const int>(kNoTangential)
                              : std::span<const int>(kAll))
                   : (l_ == 0 ? std::span<const int>(kRadial)
                              : std::span<const int>(kDisplacement));
  for (int gn = g0; gn < nglob; ++gn) {
    if (solid[gn]) {
      for (int c : components) {
        dof_[static_cast<std::size_t>(gn) * 3 + c] = ndof_++;
      }
    } else {
      dof_[static_cast<std::size_t>(gn) * 3 + 2] = ndof_++;
    }
  }

  auto drop = [&](std::size_t position) {
    const int removed = dof_[position];
    dof_[position] = -1;
    for (auto& d : dof_) {
      if (d > removed) --d;
    }
    --ndof_;
  };

  if (l_ == 1) {
    // The centre-of-mass frame: no surface potential dof.
    drop(static_cast<std::size_t>(nglob - 1) * 3 + 2);
  }
  if (l_ == 0 && mesh.Left(firstElement_) == 0.0) {
    // U(0) = 0 at a centre.
    drop(static_cast<std::size_t>(g0) * 3 + 0);
  }
}

// -- assembly
// -------------------------------------------------------------------

void DegreeOperator::Point(int i, int j, double value) {
  if (i < 0 || j < 0) return;
  if (i > j) std::swap(i, j);
  Band(i, j) += value;
}

std::vector<DegreeOperator::Interface> DegreeOperator::Interfaces() const {
  auto out = std::vector<Interface>{};
  if (l_ == 0) return out;
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto rho = material_->Rho();
  const auto g = material_->Gravity();
  for (int e = firstElement_; e + 1 < mesh.ElementCount(); ++e) {
    if (material_->IsFluid(e) == material_->IsFluid(e + 1)) continue;
    const auto gn = mesh.GlobalIndex(e, n - 1);
    const auto rt = mesh.Radius(e, n - 1);
    const auto gt = g[static_cast<std::size_t>(e) * n + (n - 1)];
    if (material_->IsFluid(e)) {
      out.push_back(
          {gn, rt, gt, rho[static_cast<std::size_t>(e) * n + (n - 1)], +1.0});
    } else {
      out.push_back(
          {gn, rt, gt, rho[static_cast<std::size_t>(e + 1) * n], -1.0});
    }
  }
  return out;
}

void DegreeOperator::Assemble(const PointShear* pointShear) {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto n3 = 3 * n;
  const auto k2 = l_ * (l_ + 1.0);
  const auto G = material_->G();
  const auto ifpg = 1.0 / (4.0 * kPi * G);
  const auto& moduli = material_->Moduli();
  const auto rho = material_->Rho();
  const auto drho = material_->DRho();
  const auto g = material_->Gravity();
  const auto w = mesh.Weights();

  band_.assign(static_cast<std::size_t>(kd_ + 1) * ndof_, 0.0);
  auto local = std::vector<double>(static_cast<std::size_t>(n3) * n3);
  auto B = std::vector<double>(static_cast<std::size_t>(n) * n);
  std::vector<double> X(n3), Y(n3), S(n3);
  std::vector<double> dM(n3), dR(n3), dS(n3);

  auto at = [n3](std::vector<double>& m, int a, int b) -> double& {
    return m[static_cast<std::size_t>(a) * n3 + b];
  };

  for (int e = firstElement_; e < mesh.ElementCount(); ++e) {
    std::ranges::fill(local, 0.0);
    const auto base = static_cast<std::size_t>(e) * n;
    const auto jac = mesh.Jacobian(e);
    const auto fluid = material_->IsFluid(e) && l_ > 0;

    // B(q, i) = r_q D(q, i) / jac: r d_r of nodal values.
    for (auto q = 0; q < n; ++q) {
      for (auto i = 0; i < n; ++i) {
        B[static_cast<std::size_t>(q) * n + i] =
            mesh.Radius(e, q) * mesh.Derivative(q, i) / jac;
      }
    }

    // phi'phi' stiffness, no jac: int r^2 phi' psi' dr.
    if (gravitating_) {
      for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < n; ++j) {
          auto kp = 0.0;
          for (auto q = 0; q < n; ++q) {
            const auto rq = mesh.Radius(e, q);
            kp +=
                mesh.Derivative(q, i) * w[q] * rq * rq * mesh.Derivative(q, j);
          }
          at(local, 3 * i + 2, 3 * j + 2) += ifpg * kp / jac;
        }
      }
    }

    for (auto q = 0; q < n; ++q) {
      const auto Wq = w[q] * jac;
      const auto rq = mesh.Radius(e, q);
      if (gravitating_) {
        at(local, 3 * q + 2, 3 * q + 2) += ifpg * k2 * Wq;
      }
      if (fluid) {
        at(local, 3 * q + 2, 3 * q + 2) +=
            Stratification(rq, drho[base + q], g[base + q]) * Wq;
        continue;
      }

      // The elastic block, from the quadrature-point vectors X, Y, S.
      std::ranges::fill(X, 0.0);
      std::ranges::fill(Y, 0.0);
      std::ranges::fill(S, 0.0);
      for (auto i = 0; i < n; ++i) {
        const auto b = B[static_cast<std::size_t>(q) * n + i];
        X[3 * i] = b;
        S[3 * i + 1] = b;
      }
      Y[3 * q] = 2.0;
      S[3 * q] = 1.0;
      Y[3 * q + 1] = -k2;
      S[3 * q + 1] -= 1.0;

      const double cC = moduli.C[base + q];
      const double cAN = moduli.A[base + q] - moduli.N[base + q];
      const double cF = moduli.F[base + q];
      const double cL = moduli.L[base + q];
      for (auto a = 0; a < n3; ++a) {
        const double xa = X[a], ya = Y[a], sa = S[a];
        if (xa == 0.0 && ya == 0.0 && sa == 0.0) continue;
        for (auto b = 0; b < n3; ++b) {
          at(local, a, b) +=
              Wq * (cC * xa * X[b] + cAN * ya * Y[b] +
                    cF * (xa * Y[b] + ya * X[b]) + k2 * cL * sa * S[b]);
        }
      }

      // Pointwise couplings at the node q.
      at(local, 3 * q + 1, 3 * q + 1) +=
          k2 * (k2 - 2.0) * moduli.N[base + q] * Wq;
      if (!gravitating_) continue;
      at(local, 3 * q, 3 * q) += 4.0 * rho[base + q] *
                                 (kPi * G * rho[base + q] * rq - g[base + q]) *
                                 rq * Wq;
      const auto cUV = k2 * rho[base + q] * g[base + q] * rq * Wq;
      at(local, 3 * q, 3 * q + 1) += cUV;
      at(local, 3 * q + 1, 3 * q) += cUV;
      const auto cVP = k2 * rho[base + q] * rq * Wq;
      at(local, 3 * q + 1, 3 * q + 2) += cVP;
      at(local, 3 * q + 2, 3 * q + 1) += cVP;
    }

    if (!fluid && gravitating_) {
      // The U-phi gradient coupling int rho r^2 (U phi' + phi U') dr,
      // whose jac cancels against the derivative's.
      for (auto i = 0; i < n; ++i) {
        const double mi =
            rho[base + i] * mesh.Radius(e, i) * mesh.Radius(e, i) * w[i];
        for (auto j = 0; j < n; ++j) {
          const auto m = mi * mesh.Derivative(i, j);
          at(local, 3 * i, 3 * j + 2) += m;
          at(local, 3 * j + 2, 3 * i) += m;
        }
      }
    }

    // The point shear (MemoryPoints.hpp): at each memory point,
    // jac w_p s_p (k^2 (k^2 - 2) dM^2 + 4/3 dR^2 + k^2 dS^2) in the
    // deviatoric scalars dM = V, dR = r d_r U - U + k^2 V / 2 and
    // dS = U + r d_r V - V, which with GLL points is exactly the elastic
    // block with the shear modulus raised by s.
    if (pointShear != nullptr && !fluid) {
      const auto& points = *pointShear->points;
      for (auto p = 0; p < points.count; ++p) {
        const auto s =
            pointShear->modulus[static_cast<std::size_t>(e) * points.count + p];
        if (s == 0.0) continue;
        auto rp = 0.0;
        for (auto i = 0; i < n; ++i)
          rp += points.Value(p, i) * mesh.Radius(e, i);
        std::ranges::fill(dM, 0.0);
        std::ranges::fill(dR, 0.0);
        std::ranges::fill(dS, 0.0);
        for (auto i = 0; i < n; ++i) {
          const auto v = points.Value(p, i);
          const auto b = rp * points.Derivative(p, i) / jac;
          dR[3 * i] = b - v;
          dR[3 * i + 1] = 0.5 * k2 * v;
          dS[3 * i] = v;
          dS[3 * i + 1] = b - v;
          dM[3 * i + 1] = v;
        }
        const auto weight = points.weight[p] * jac * s;
        for (auto a = 0; a < n3; ++a) {
          if (dM[a] == 0.0 && dR[a] == 0.0 && dS[a] == 0.0) continue;
          for (auto b = 0; b < n3; ++b) {
            at(local, a, b) +=
                weight * (k2 * (k2 - 2.0) * dM[a] * dM[b] +
                          (4.0 / 3.0) * dR[a] * dR[b] + k2 * dS[a] * dS[b]);
          }
        }
      }
    }

    // Scatter the upper triangle in global dofs.
    for (auto a = 0; a < n3; ++a) {
      const auto ga = Dof(mesh.GlobalIndex(e, a / 3), Component(a % 3));
      if (ga < 0) continue;
      for (auto b = 0; b < n3; ++b) {
        const auto gb = Dof(mesh.GlobalIndex(e, b / 3), Component(b % 3));
        if (gb < ga) continue;
        const auto value = at(local, a, b);
        if (value != 0.0) Band(ga, gb) += value;
      }
    }
  }

  // Fluid-solid interfaces: +/- rho_f (g U U' + phi U' + U phi') r^2.
  for (const auto& f : Interfaces()) {
    const auto iU = Dof(f.node, Component::U);
    const auto iP = Dof(f.node, Component::Phi);
    Point(iU, iU, f.sign * f.rhoF * f.g * f.r * f.r);
    Point(iU, iP, f.sign * f.rhoF * f.r * f.r);
  }

  // Dirichlet-to-Neumann closures.
  if (!gravitating_) return;
  const auto& mesh2 = material_->Mesh();
  const auto top = mesh2.GlobalIndex(mesh2.ElementCount() - 1, n - 1);
  if (l_ != 1) {
    Point(Dof(top, Component::Phi), Dof(top, Component::Phi),
          (l_ + 1) * material_->Radius() * ifpg);
  }
  if (mesh2.Left(firstElement_) > 0.0) {
    const int bottom = mesh2.GlobalIndex(firstElement_, 0);
    Point(Dof(bottom, Component::Phi), Dof(bottom, Component::Phi),
          l_ * mesh2.Left(firstElement_) * ifpg);
  }
}

// -- factorisation
// ---------------------------------------------------------------

void DegreeOperator::Factorise() {
  const int n = ndof_;
  const int kd = kd_;
  const int ldab = kd + 1;

  // Symmetric diagonal scaling, equalising the units of the dofs.
  scale_.resize(n);
  auto positive = true;
  for (auto j = 0; j < n; ++j) {
    const double d = band_[static_cast<std::size_t>(j) * ldab + kd];
    positive = positive && d > 0.0;
    scale_[j] = 1.0 / std::sqrt(std::abs(d) > 0.0 ? std::abs(d) : 1.0);
  }

  auto scaled = [&](int i, int j) {
    return band_[static_cast<std::size_t>(j) * ldab + (kd + i - j)] *
           scale_[i] * scale_[j];
  };

  if (positive) {
    factor_.assign(static_cast<std::size_t>(ldab) * n, 0.0);
    for (auto j = 0; j < n; ++j) {
      for (int i = std::max(0, j - kd); i <= j; ++i) {
        factor_[static_cast<std::size_t>(j) * ldab + (kd + i - j)] =
            scaled(i, j);
      }
    }
    auto info = 0;
    dpbtrf_("U", &n, &kd, factor_.data(), &ldab, &info);
    if (info == 0) return;
    choleskyFailed_ = true;
  } else {
    choleskyFailed_ = true;
  }

  // Banded LU on the symmetrically scaled matrix.
  const int ldlu = 3 * kd + 1;
  factor_.assign(static_cast<std::size_t>(ldlu) * n, 0.0);
  for (auto j = 0; j < n; ++j) {
    for (int i = std::max(0, j - kd); i <= std::min(n - 1, j + kd); ++i) {
      factor_[static_cast<std::size_t>(j) * ldlu + (2 * kd + i - j)] =
          i <= j ? scaled(i, j) : scaled(j, i);
    }
  }
  pivots_.resize(n);
  auto info = 0;
  dgbtrf_(&n, &n, &kd, &kd, factor_.data(), &ldlu, pivots_.data(), &info);
  if (info != 0) {
    throw std::runtime_error(std::format(
        "degree {}: the banded LU factorisation failed (info = {})", l_, info));
  }
}

std::vector<double> DegreeOperator::Solve(std::span<const double> b,
                                          int nrhs) const {
  const int n = ndof_;
  if (static_cast<int>(b.size()) != n * nrhs) {
    throw std::invalid_argument(std::format(
        "expected {} x {} right-hand-side values, got {}", n, nrhs, b.size()));
  }
  auto x = std::vector<double>(b.begin(), b.end());
  for (auto k = 0; k < nrhs; ++k) {
    for (auto j = 0; j < n; ++j)
      x[static_cast<std::size_t>(k) * n + j] *= scale_[j];
  }
  auto info = 0;
  if (!choleskyFailed_) {
    const int ldab = kd_ + 1;
    dpbtrs_("U", &n, &kd_, &nrhs, factor_.data(), &ldab, x.data(), &n, &info);
  } else {
    const int ldlu = 3 * kd_ + 1;
    dgbtrs_("N", &n, &kd_, &kd_, &nrhs, factor_.data(), &ldlu, pivots_.data(),
            x.data(), &n, &info);
  }
  if (info != 0) {
    throw std::runtime_error(std::format(
        "degree {}: the banded solve failed (info = {})", l_, info));
  }
  for (auto k = 0; k < nrhs; ++k) {
    for (auto j = 0; j < n; ++j)
      x[static_cast<std::size_t>(k) * n + j] *= scale_[j];
  }
  return x;
}

// -- right-hand sides
// -------------------------------------------------------------

int DegreeOperator::SurfaceDof(Component component) const {
  const auto& mesh = material_->Mesh();
  const auto top = mesh.GlobalIndex(mesh.ElementCount() - 1, mesh.NGll() - 1);
  return Dof(top, component);
}

std::vector<double> DegreeOperator::Load(LoadPart part) const {
  auto b = std::vector<double>(ndof_, 0.0);
  const auto a = material_->Radius();
  if (part != LoadPart::Potential) {
    b[SurfaceDof(Component::U)] = -material_->SurfaceGravity() * a * a;
  }
  if (part != LoadPart::Force && SurfaceDof(Component::Phi) >= 0) {
    b[SurfaceDof(Component::Phi)] = -a * a;
  }
  return b;
}

std::vector<double> DegreeOperator::Tangential() const {
  auto b = std::vector<double>(ndof_, 0.0);
  const auto i = SurfaceDof(Component::V);
  if (i >= 0) {
    const auto a = material_->Radius();
    b[i] = -material_->SurfaceGravity() * a * a * l_ * (l_ + 1.0);
  }
  return b;
}

std::vector<double> DegreeOperator::Traction() const {
  auto b = std::vector<double>(ndof_, 0.0);
  const auto a = material_->Radius();
  b[SurfaceDof(Component::U)] = -a * a;
  return b;
}

std::vector<double> DegreeOperator::TractionTangential() const {
  auto b = std::vector<double>(ndof_, 0.0);
  const auto i = SurfaceDof(Component::V);
  if (i >= 0) {
    const auto a = material_->Radius();
    b[i] = -a * a * l_ * (l_ + 1.0);
  }
  return b;
}

std::vector<double> DegreeOperator::Tide(int power) const {
  const auto p = power < 0 ? l_ : power;
  auto b = std::vector<double>(ndof_, 0.0);
  if (p == 0) return b;
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto a = material_->Radius();
  const auto k2 = l_ * (l_ + 1.0);
  const auto rho = material_->Rho();
  const auto drho = material_->DRho();
  const auto g = material_->Gravity();
  const auto w = mesh.Weights();
  for (int e = firstElement_; e < mesh.ElementCount(); ++e) {
    const auto base = static_cast<std::size_t>(e) * n;
    const auto jac = mesh.Jacobian(e);
    const auto fluid = material_->IsFluid(e) && l_ > 0;
    for (auto q = 0; q < n; ++q) {
      const auto Wq = w[q] * jac;
      const auto rq = mesh.Radius(e, q);
      const auto psi = std::pow(rq / a, p);
      const auto gn = mesh.GlobalIndex(e, q);
      if (fluid) {
        b[Dof(gn, Component::Phi)] -=
            Wq * Stratification(rq, drho[base + q], g[base + q]) * psi;
        continue;
      }
      const auto dpsi = (p / a) * std::pow(rq / a, p - 1);
      const auto iU = Dof(gn, Component::U);
      if (iU >= 0) b[iU] -= Wq * rho[base + q] * rq * rq * dpsi;
      if (l_ > 0) {
        b[Dof(gn, Component::V)] -= k2 * Wq * rho[base + q] * rq * psi;
      }
    }
  }
  for (const auto& f : Interfaces()) {
    b[Dof(f.node, Component::U)] -=
        f.sign * f.rhoF * f.r * f.r * std::pow(f.r / a, p);
  }
  return b;
}

// -- views
// --------------------------------------------------------------------------

DegreeOperator::NodalSolution DegreeOperator::Expand(
    std::span<const double> x) const {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto size = static_cast<std::size_t>(mesh.ElementCount()) * n;
  NodalSolution out{.U = std::vector<double>(size, 0.0),
                    .V = std::vector<double>(size, 0.0),
                    .phi = std::vector<double>(size, 0.0)};
  std::vector<double>* fields[3] = {&out.U, &out.V, &out.phi};
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto i = 0; i < n; ++i) {
      const auto gn = mesh.GlobalIndex(e, i);
      for (auto c = 0; c < 3; ++c) {
        const auto d = Dof(gn, Component(c));
        if (d >= 0) (*fields[c])[static_cast<std::size_t>(e) * n + i] = x[d];
      }
    }
    if (l_ > 0 && material_->IsFluid(e)) {
      for (auto i = 0; i < n; ++i) {
        out.U[static_cast<std::size_t>(e) * n + i] =
            std::numeric_limits<double>::quiet_NaN();
        out.V[static_cast<std::size_t>(e) * n + i] =
            std::numeric_limits<double>::quiet_NaN();
      }
    }
  }
  return out;
}

}  // namespace sl3dpp
