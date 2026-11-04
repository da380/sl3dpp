#include "sl3dpp/DegreeOperator.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "sl3dpp/detail/BandSolvers.hpp"
#include "sl3dpp/detail/BandSubstructure.hpp"

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
  Factorise(options);
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

struct DegreeOperator::Coefficients {
  // Per node (element, gll): the elastic block's moduli, the pointwise
  // V-V modulus, the gravitational couplings, the fluid stratification and,
  // per fluid-solid interface, the two point couplings.
  std::vector<double> C, AN, F, L, N;
  std::vector<double> UU, UV, VP, UPhi, strat;
  std::vector<double> interfaceUU, interfaceUP;
};

DegreeOperator::Coefficients DegreeOperator::BaseCoefficients() const {
  const auto& mesh = material_->Mesh();
  const auto k2 = l_ * (l_ + 1.0);
  const auto G = material_->G();
  const auto& moduli = material_->Moduli();
  const auto rho = material_->Rho();
  const auto drho = material_->DRho();
  const auto g = material_->Gravity();
  const auto size = rho.size();
  Coefficients c;
  c.C.assign(moduli.C.begin(), moduli.C.end());
  c.F.assign(moduli.F.begin(), moduli.F.end());
  c.L.assign(moduli.L.begin(), moduli.L.end());
  c.N.assign(moduli.N.begin(), moduli.N.end());
  c.AN.resize(size);
  c.UU.resize(size);
  c.UV.resize(size);
  c.VP.resize(size);
  c.UPhi.resize(size);
  c.strat.resize(size);
  for (auto i = std::size_t{0}; i < size; ++i) {
    const auto r = mesh.Radii()[i];
    c.AN[i] = moduli.A[i] - moduli.N[i];
    c.UU[i] = 4.0 * rho[i] * (kPi * G * rho[i] * r - g[i]) * r;
    c.UV[i] = k2 * rho[i] * g[i] * r;
    c.VP[i] = k2 * rho[i] * r;
    c.UPhi[i] = rho[i] * r * r;
    c.strat[i] = Stratification(r, drho[i], g[i]);
  }
  for (const auto& f : Interfaces()) {
    c.interfaceUU.push_back(f.sign * f.rhoF * f.g * f.r * f.r);
    c.interfaceUP.push_back(f.sign * f.rhoF * f.r * f.r);
  }
  return c;
}

DegreeOperator::Coefficients DegreeOperator::VariationCoefficients(
    const MaterialVariation& dm) const {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto k2 = l_ * (l_ + 1.0);
  const auto G = material_->G();
  const auto rho = material_->Rho();
  const auto drho = material_->DRho();
  const auto g = material_->Gravity();
  const auto size = rho.size();
  if (dm.rho.size() != size || dm.moduli.A.size() != size) {
    throw std::invalid_argument(
        "the material variation must have the mesh's nodal size");
  }
  Coefficients c;
  c.C.assign(dm.moduli.C.begin(), dm.moduli.C.end());
  c.F.assign(dm.moduli.F.begin(), dm.moduli.F.end());
  c.L.assign(dm.moduli.L.begin(), dm.moduli.L.end());
  c.N.assign(dm.moduli.N.begin(), dm.moduli.N.end());
  c.AN.resize(size);
  c.UU.resize(size);
  c.UV.resize(size);
  c.VP.resize(size);
  c.UPhi.resize(size);
  c.strat.resize(size);
  for (auto i = std::size_t{0}; i < size; ++i) {
    const auto r = mesh.Radii()[i];
    c.AN[i] = dm.moduli.A[i] - dm.moduli.N[i];
    // d/d(rho, g) of 4 rho (pi G rho r - g) r
    c.UU[i] = 4.0 * dm.rho[i] * (kPi * G * rho[i] * r - g[i]) * r +
              4.0 * rho[i] * (kPi * G * dm.rho[i] * r - dm.g[i]) * r;
    c.UV[i] = k2 * (dm.rho[i] * g[i] + rho[i] * dm.g[i]) * r;
    c.VP[i] = k2 * dm.rho[i] * r;
    c.UPhi[i] = dm.rho[i] * r * r;
    // d of r^2 drho / g
    c.strat[i] = g[i] > 0.0 ? r * r * (dm.drho[i] / g[i] -
                                       drho[i] * dm.g[i] / (g[i] * g[i]))
                            : 0.0;
  }
  // The interfaces: the fluid-side density and gravity at the interface
  // node, varied.
  for (int e = firstElement_; e + 1 < mesh.ElementCount(); ++e) {
    if (material_->IsFluid(e) == material_->IsFluid(e + 1)) continue;
    if (l_ == 0) break;
    const auto rt = mesh.Radius(e, n - 1);
    const auto top = static_cast<std::size_t>(e) * n + (n - 1);
    const auto fluidNode =
        material_->IsFluid(e) ? top : static_cast<std::size_t>(e + 1) * n;
    const auto sign = material_->IsFluid(e) ? 1.0 : -1.0;
    const auto rhoF = rho[fluidNode], dRhoF = dm.rho[fluidNode];
    const auto gt = g[top], dgt = dm.g[top];
    c.interfaceUU.push_back(sign * (dRhoF * gt + rhoF * dgt) * rt * rt);
    c.interfaceUP.push_back(sign * dRhoF * rt * rt);
  }
  return c;
}

void DegreeOperator::AssembleBand(std::vector<double>& band,
                                  const Coefficients& c,
                                  const PointShear* pointShear,
                                  bool closures) const {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto n3 = 3 * n;
  const auto k2 = l_ * (l_ + 1.0);
  const auto G = material_->G();
  const auto ifpg = 1.0 / (4.0 * kPi * G);
  const auto w = mesh.Weights();

  band.assign(static_cast<std::size_t>(kd_ + 1) * ndof_, 0.0);
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
    if (gravitating_ && closures) {
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
      if (gravitating_ && closures) {
        at(local, 3 * q + 2, 3 * q + 2) += ifpg * k2 * Wq;
      }
      if (fluid) {
        at(local, 3 * q + 2, 3 * q + 2) += c.strat[base + q] * Wq;
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

      const double cC = c.C[base + q];
      const double cAN = c.AN[base + q];
      const double cF = c.F[base + q];
      const double cL = c.L[base + q];
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
      at(local, 3 * q + 1, 3 * q + 1) += k2 * (k2 - 2.0) * c.N[base + q] * Wq;
      if (!gravitating_) continue;
      at(local, 3 * q, 3 * q) += c.UU[base + q] * Wq;
      const auto cUV = c.UV[base + q] * Wq;
      at(local, 3 * q, 3 * q + 1) += cUV;
      at(local, 3 * q + 1, 3 * q) += cUV;
      const auto cVP = c.VP[base + q] * Wq;
      at(local, 3 * q + 1, 3 * q + 2) += cVP;
      at(local, 3 * q + 2, 3 * q + 1) += cVP;
    }

    if (!fluid && gravitating_) {
      // The U-phi gradient coupling int rho r^2 (U phi' + phi U') dr,
      // whose jac cancels against the derivative's.
      for (auto i = 0; i < n; ++i) {
        const double mi = c.UPhi[base + i] * w[i];
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
        if (value != 0.0) BandOf(band, kd_, ga, gb) += value;
      }
    }
  }

  // Fluid-solid interfaces: +/- rho_f (g U U' + phi U' + U phi') r^2.
  auto point = [&](int i, int j, double value) {
    if (i < 0 || j < 0) return;
    if (i > j) std::swap(i, j);
    BandOf(band, kd_, i, j) += value;
  };
  const auto interfaces = Interfaces();
  for (auto f = std::size_t{0}; f < interfaces.size(); ++f) {
    const auto iU = Dof(interfaces[f].node, Component::U);
    const auto iP = Dof(interfaces[f].node, Component::Phi);
    point(iU, iU, c.interfaceUU[f]);
    point(iU, iP, c.interfaceUP[f]);
  }

  // Dirichlet-to-Neumann closures.
  if (!gravitating_ || !closures) return;
  const auto top = mesh.GlobalIndex(mesh.ElementCount() - 1, n - 1);
  if (l_ != 1) {
    point(Dof(top, Component::Phi), Dof(top, Component::Phi),
          (l_ + 1) * material_->Radius() * ifpg);
  }
  if (mesh.Left(firstElement_) > 0.0) {
    const int bottom = mesh.GlobalIndex(firstElement_, 0);
    point(Dof(bottom, Component::Phi), Dof(bottom, Component::Phi),
          l_ * mesh.Left(firstElement_) * ifpg);
  }
}

void DegreeOperator::Assemble(const PointShear* pointShear) {
  AssembleBand(band_, BaseCoefficients(), pointShear, true);
}

std::vector<double> DegreeOperator::ApplyBand(const std::vector<double>& band,
                                              std::span<const double> x,
                                              int nrhs) const {
  const auto n = ndof_;
  if (static_cast<int>(x.size()) != n * nrhs) {
    throw std::invalid_argument(
        std::format("expected {} x {} values, got {}", n, nrhs, x.size()));
  }
  // Hand-written (not BLAS dsbmv): a BLAS call inside the threaded
  // degree loops starts the BLAS library's own threads on the server
  // and the two thread pools fight.
  const auto ldab = kd_ + 1;
  auto y = std::vector<double>(x.size(), 0.0);
  for (auto k = 0; k < nrhs; ++k) {
    const auto* xk = x.data() + static_cast<std::size_t>(k) * n;
    auto* yk = y.data() + static_cast<std::size_t>(k) * n;
    for (auto j = 0; j < n; ++j) {
      for (auto i = std::max(0, j - kd_); i <= j; ++i) {
        const auto a = band[static_cast<std::size_t>(j) * ldab + (kd_ + i - j)];
        yk[i] += a * xk[j];
        if (i != j) yk[j] += a * xk[i];
      }
    }
  }
  return y;
}

std::pair<std::vector<double>, std::vector<double>>
DegreeOperator::MaterialGradient(std::span<const double> y,
                                 std::span<const double> x, int nrhs) const {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto k2 = l_ * (l_ + 1.0);
  const auto w = mesh.Weights();
  const auto expected = static_cast<std::size_t>(ndof_) * nrhs;
  if (y.size() != expected || x.size() != expected) {
    throw std::invalid_argument("the solutions have ndof x nrhs entries");
  }
  const auto size = material_->Rho().size();
  auto kappa = std::vector<double>(size, 0.0);
  auto mu = std::vector<double>(size, 0.0);
  auto B = std::vector<double>(static_cast<std::size_t>(n) * n);
  std::vector<double> Uy(n), Vy(n), Ux(n), Vx(n);
  for (int e = firstElement_; e < mesh.ElementCount(); ++e) {
    if (material_->IsFluid(e) && l_ > 0) continue;
    const auto base = static_cast<std::size_t>(e) * n;
    const auto jac = mesh.Jacobian(e);
    for (auto q = 0; q < n; ++q) {
      for (auto i = 0; i < n; ++i) {
        B[static_cast<std::size_t>(q) * n + i] =
            mesh.Radius(e, q) * mesh.Derivative(q, i) / jac;
      }
    }
    for (auto c = 0; c < nrhs; ++c) {
      const auto column = static_cast<std::size_t>(c) * ndof_;
      for (auto i = 0; i < n; ++i) {
        const auto g = mesh.GlobalIndex(e, i);
        const auto iU = Dof(g, Component::U);
        const auto iV = Dof(g, Component::V);
        Uy[i] = iU >= 0 ? y[column + iU] : 0.0;
        Ux[i] = iU >= 0 ? x[column + iU] : 0.0;
        Vy[i] = iV >= 0 ? y[column + iV] : 0.0;
        Vx[i] = iV >= 0 ? x[column + iV] : 0.0;
      }
      for (auto q = 0; q < n; ++q) {
        const auto Wq = w[q] * jac;
        auto Xy = 0.0, Xx = 0.0, dVy = 0.0, dVx = 0.0;
        for (auto i = 0; i < n; ++i) {
          const auto b = B[static_cast<std::size_t>(q) * n + i];
          Xy += b * Uy[i];
          Xx += b * Ux[i];
          dVy += b * Vy[i];
          dVx += b * Vx[i];
        }
        const auto Yy = 2.0 * Uy[q] - k2 * Vy[q];
        const auto Yx = 2.0 * Ux[q] - k2 * Vx[q];
        const auto Sy = dVy + Uy[q] - Vy[q];
        const auto Sx = dVx + Ux[q] - Vx[q];
        const auto dC = Xy * Xx;
        const auto dAN = Yy * Yx;
        const auto dF = Xy * Yx + Yy * Xx;
        const auto dL = k2 * Sy * Sx;
        const auto dN = k2 * (k2 - 2.0) * Vy[q] * Vx[q];
        // dC = dA = dkappa + 4/3 dmu, d(A - N) = dkappa + 1/3 dmu,
        // dF = dkappa - 2/3 dmu, dL = dN = dmu
        kappa[base + q] += Wq * (dC + dAN + dF);
        mu[base + q] += Wq * ((4.0 / 3.0) * dC + dAN / 3.0 - (2.0 / 3.0) * dF +
                              dL + dN);
      }
    }
  }
  return {std::move(kappa), std::move(mu)};
}

std::vector<double> DegreeOperator::ApplyVariation(const MaterialVariation& dm,
                                                   std::span<const double> x,
                                                   int nrhs) const {
  std::vector<double> band;
  AssembleBand(band, VariationCoefficients(dm), nullptr, false);
  return ApplyBand(band, x, nrhs);
}

std::vector<double> DegreeOperator::LoadVariation(const MaterialVariation& dm,
                                                  LoadPart part) const {
  auto b = std::vector<double>(ndof_, 0.0);
  const auto a = material_->Radius();
  if (part != LoadPart::Potential && gravitating_) {
    b[SurfaceDof(Component::U)] = -dm.surfaceGravity * a * a;
  }
  return b;
}

std::vector<double> DegreeOperator::TangentialVariation(
    const MaterialVariation& dm) const {
  auto b = std::vector<double>(ndof_, 0.0);
  const auto i = SurfaceDof(Component::V);
  if (i >= 0 && gravitating_) {
    const auto a = material_->Radius();
    b[i] = -dm.surfaceGravity * a * a * l_ * (l_ + 1.0);
  }
  return b;
}

std::vector<double> DegreeOperator::TideVariation(const MaterialVariation& dm,
                                                  int power) const {
  const auto p = power < 0 ? l_ : power;
  auto b = std::vector<double>(ndof_, 0.0);
  if (p == 0 || !gravitating_) return b;
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
      const auto i = base + q;
      if (fluid) {
        const auto dstrat =
            g[i] > 0.0
                ? rq * rq * (dm.drho[i] / g[i] - drho[i] * dm.g[i] / (g[i] * g[i]))
                : 0.0;
        b[Dof(gn, Component::Phi)] -= Wq * dstrat * psi;
        continue;
      }
      const auto dpsi = (p / a) * std::pow(rq / a, p - 1);
      const auto iU = Dof(gn, Component::U);
      if (iU >= 0) b[iU] -= Wq * dm.rho[i] * rq * rq * dpsi;
      if (l_ > 0) b[Dof(gn, Component::V)] -= k2 * Wq * dm.rho[i] * rq * psi;
    }
  }
  if (l_ > 0) {
    for (int e = firstElement_; e + 1 < mesh.ElementCount(); ++e) {
      if (material_->IsFluid(e) == material_->IsFluid(e + 1)) continue;
      const auto gn = mesh.GlobalIndex(e, n - 1);
      const auto rt = mesh.Radius(e, n - 1);
      const auto fluidNode = material_->IsFluid(e)
                                 ? static_cast<std::size_t>(e) * n + (n - 1)
                                 : static_cast<std::size_t>(e + 1) * n;
      const auto sign = material_->IsFluid(e) ? 1.0 : -1.0;
      b[Dof(gn, Component::U)] -=
          sign * dm.rho[fluidNode] * rt * rt * std::pow(rt / a, p);
    }
  }
  return b;
}

// -- the density transposes
// -----------------------------------------------------

void DegreeOperator::AddDensityDuals(std::span<const double> y,
                                     std::span<const double> x, int nrhs,
                                     double scale, DensityDuals& duals) const {
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto k2 = l_ * (l_ + 1.0);
  const auto G = material_->G();
  const auto rho = material_->Rho();
  const auto drho = material_->DRho();
  const auto g = material_->Gravity();
  const auto w = mesh.Weights();
  const auto size = rho.size();
  const auto expected = static_cast<std::size_t>(ndof_) * nrhs;
  if (y.size() != expected || x.size() != expected) {
    throw std::invalid_argument("the solutions have ndof x nrhs entries");
  }
  if (duals.rho.size() != size) duals.Resize(size);
  std::vector<double> Uy(n), Vy(n), Py(n), Ux(n), Vx(n), Px(n);
  for (int e = firstElement_; e < mesh.ElementCount(); ++e) {
    const auto base = static_cast<std::size_t>(e) * n;
    const auto jac = mesh.Jacobian(e);
    const auto fluid = material_->IsFluid(e) && l_ > 0;
    for (auto c = 0; c < nrhs; ++c) {
      const auto column = static_cast<std::size_t>(c) * ndof_;
      for (auto i = 0; i < n; ++i) {
        const auto gn = mesh.GlobalIndex(e, i);
        const auto iU = Dof(gn, Component::U);
        const auto iV = Dof(gn, Component::V);
        const auto iP = Dof(gn, Component::Phi);
        Uy[i] = iU >= 0 ? y[column + iU] : 0.0;
        Ux[i] = iU >= 0 ? x[column + iU] : 0.0;
        Vy[i] = iV >= 0 ? y[column + iV] : 0.0;
        Vx[i] = iV >= 0 ? x[column + iV] : 0.0;
        Py[i] = iP >= 0 ? y[column + iP] : 0.0;
        Px[i] = iP >= 0 ? x[column + iP] : 0.0;
      }
      for (auto q = 0; q < n; ++q) {
        const auto Wq = w[q] * jac;
        const auto r = mesh.Radius(e, q);
        const auto i = base + q;
        if (fluid) {
          // the stratification r^2 drho / g on phi phi
          if (g[i] > 0.0) {
            const auto pp = Py[q] * Px[q] * Wq * r * r;
            duals.drho[i] += scale * pp / g[i];
            duals.g[i] -= scale * pp * drho[i] / (g[i] * g[i]);
          }
          continue;
        }
        if (!gravitating_) continue;
        const auto uu = Uy[q] * Ux[q];
        const auto uv = Uy[q] * Vx[q] + Vy[q] * Ux[q];
        const auto vp = Vy[q] * Px[q] + Py[q] * Vx[q];
        // 4 rho (pi G rho r - g) r on U U, k^2 rho g r on U V, k^2 rho r
        // on V phi, differentiated in rho and g
        duals.rho[i] += scale * Wq *
                        (4.0 * r * (2.0 * kPi * G * rho[i] * r - g[i]) * uu +
                         k2 * g[i] * r * uv + k2 * r * vp);
        duals.g[i] += scale * Wq * (-4.0 * rho[i] * r * uu + k2 * rho[i] * r * uv);
        // the U-phi gradient coupling rho r^2 (U phi' + phi U'), no jac
        auto s = 0.0;
        for (auto j = 0; j < n; ++j) {
          s += mesh.Derivative(q, j) * (Uy[q] * Px[j] + Ux[q] * Py[j]);
        }
        duals.rho[i] += scale * w[q] * r * r * s;
      }
    }
  }
  // The fluid-solid interfaces: +/- rho_f (g U U' + phi U' + U phi') r^2.
  if (l_ > 0) {
    for (int e = firstElement_; e + 1 < mesh.ElementCount(); ++e) {
      if (material_->IsFluid(e) == material_->IsFluid(e + 1)) continue;
      const auto gn = mesh.GlobalIndex(e, n - 1);
      const auto rt = mesh.Radius(e, n - 1);
      const auto top = static_cast<std::size_t>(e) * n + (n - 1);
      const auto fluidNode =
          material_->IsFluid(e) ? top : static_cast<std::size_t>(e + 1) * n;
      const auto sign = material_->IsFluid(e) ? 1.0 : -1.0;
      const auto iU = Dof(gn, Component::U);
      const auto iP = Dof(gn, Component::Phi);
      if (iU < 0) continue;
      for (auto c = 0; c < nrhs; ++c) {
        const auto column = static_cast<std::size_t>(c) * ndof_;
        const auto yu = y[column + iU], xu = x[column + iU];
        const auto yp = iP >= 0 ? y[column + iP] : 0.0;
        const auto xp = iP >= 0 ? x[column + iP] : 0.0;
        duals.rho[fluidNode] +=
            scale * sign * rt * rt * (g[top] * yu * xu + yu * xp + yp * xu);
        duals.g[top] += scale * sign * rho[fluidNode] * rt * rt * yu * xu;
      }
    }
  }
}

void DegreeOperator::AddLoadDensityDuals(std::span<const double> y,
                                         std::span<const double> amplitudes,
                                         LoadPart part, double scale,
                                         DensityDuals& duals) const {
  if (duals.rho.size() != material_->Rho().size()) {
    duals.Resize(material_->Rho().size());
  }
  if (part == LoadPart::Potential || !gravitating_) return;
  const auto a = material_->Radius();
  const auto iU = SurfaceDof(Component::U);
  const auto nrhs = amplitudes.size();
  for (auto c = std::size_t{0}; c < nrhs; ++c) {
    duals.surfaceGravity -=
        scale * a * a * y[c * static_cast<std::size_t>(ndof_) + iU] * amplitudes[c];
  }
}

void DegreeOperator::AddTangentialDensityDuals(std::span<const double> y,
                                               std::span<const double> amplitudes,
                                               double scale,
                                               DensityDuals& duals) const {
  if (duals.rho.size() != material_->Rho().size()) {
    duals.Resize(material_->Rho().size());
  }
  const auto iV = SurfaceDof(Component::V);
  if (iV < 0 || !gravitating_) return;
  const auto a = material_->Radius();
  const auto nrhs = amplitudes.size();
  for (auto c = std::size_t{0}; c < nrhs; ++c) {
    duals.surfaceGravity -= scale * a * a * l_ * (l_ + 1.0) *
                            y[c * static_cast<std::size_t>(ndof_) + iV] *
                            amplitudes[c];
  }
}

void DegreeOperator::AddTideDensityDuals(std::span<const double> y,
                                         std::span<const double> amplitudes,
                                         int power, double scale,
                                         DensityDuals& duals) const {
  const auto size = material_->Rho().size();
  if (duals.rho.size() != size) duals.Resize(size);
  const auto p = power < 0 ? l_ : power;
  if (p == 0 || !gravitating_) return;
  const auto& mesh = material_->Mesh();
  const auto n = mesh.NGll();
  const auto a = material_->Radius();
  const auto k2 = l_ * (l_ + 1.0);
  const auto drho = material_->DRho();
  const auto g = material_->Gravity();
  const auto w = mesh.Weights();
  const auto nrhs = amplitudes.size();
  auto yAt = [&](int dof, std::size_t c) {
    return dof >= 0 ? y[c * static_cast<std::size_t>(ndof_) + dof] : 0.0;
  };
  for (int e = firstElement_; e < mesh.ElementCount(); ++e) {
    const auto base = static_cast<std::size_t>(e) * n;
    const auto jac = mesh.Jacobian(e);
    const auto fluid = material_->IsFluid(e) && l_ > 0;
    for (auto q = 0; q < n; ++q) {
      const auto Wq = w[q] * jac;
      const auto rq = mesh.Radius(e, q);
      const auto psi = std::pow(rq / a, p);
      const auto gn = mesh.GlobalIndex(e, q);
      const auto i = base + q;
      if (fluid) {
        if (g[i] <= 0.0) continue;
        auto yp = 0.0;
        for (auto c = std::size_t{0}; c < nrhs; ++c) {
          yp += yAt(Dof(gn, Component::Phi), c) * amplitudes[c];
        }
        duals.drho[i] -= scale * Wq * psi * rq * rq * yp / g[i];
        duals.g[i] += scale * Wq * psi * rq * rq * drho[i] * yp / (g[i] * g[i]);
        continue;
      }
      const auto dpsi = (p / a) * std::pow(rq / a, p - 1);
      auto yu = 0.0, yv = 0.0;
      for (auto c = std::size_t{0}; c < nrhs; ++c) {
        yu += yAt(Dof(gn, Component::U), c) * amplitudes[c];
        if (l_ > 0) yv += yAt(Dof(gn, Component::V), c) * amplitudes[c];
      }
      duals.rho[i] -= scale * Wq * rq * rq * dpsi * yu;
      if (l_ > 0) duals.rho[i] -= scale * k2 * Wq * rq * psi * yv;
    }
  }
  if (l_ > 0) {
    for (int e = firstElement_; e + 1 < mesh.ElementCount(); ++e) {
      if (material_->IsFluid(e) == material_->IsFluid(e + 1)) continue;
      const auto gn = mesh.GlobalIndex(e, n - 1);
      const auto rt = mesh.Radius(e, n - 1);
      const auto fluidNode = material_->IsFluid(e)
                                 ? static_cast<std::size_t>(e) * n + (n - 1)
                                 : static_cast<std::size_t>(e + 1) * n;
      const auto sign = material_->IsFluid(e) ? 1.0 : -1.0;
      auto yu = 0.0;
      for (auto c = std::size_t{0}; c < nrhs; ++c) {
        yu += yAt(Dof(gn, Component::U), c) * amplitudes[c];
      }
      duals.rho[fluidNode] -= scale * sign * rt * rt * std::pow(rt / a, p) * yu;
    }
  }
}

// -- factorisation
// ---------------------------------------------------------------

std::pair<std::vector<std::pair<int, int>>, int> DegreeOperator::Separators(
    std::span<const int> cuts) const {
  const auto& mesh = material_->Mesh();
  auto out = std::vector<std::pair<int, int>>{};
  auto dropped = 0;
  auto previous = 0;
  for (const auto cut : cuts) {
    if (cut <= previous || cut >= mesh.ElementCount()) {
      throw std::invalid_argument(std::format(
          "the radial cuts must be increasing elements inside the mesh "
          "(1 to {}); got {}",
          mesh.ElementCount() - 1, cut));
    }
    previous = cut;
    if (cut <= firstElement_) {
      ++dropped;
      continue;
    }
    // The cut node's dofs, consecutive in the node-major numbering.
    const auto node = mesh.GlobalIndex(cut, 0);
    auto first = ndof_, end = 0;
    for (const auto c : {Component::U, Component::V, Component::Phi}) {
      if (const auto d = Dof(node, c); d >= 0) {
        first = std::min(first, d);
        end = std::max(end, d + 1);
      }
    }
    if (end > first) out.emplace_back(first, end);
  }
  return {out, dropped};
}

int DegreeOperator::SegmentCount() const {
  return substructure_ ? substructure_->SegmentCount() : 1;
}

void DegreeOperator::Factorise(const Options& options) {
  auto [separators, dropped] = Separators(options.cuts);
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

  if (!separators.empty()) {
    // The scaled band, substructured over the cuts.
    auto band = std::vector<double>(static_cast<std::size_t>(ldab) * n, 0.0);
    for (auto j = 0; j < n; ++j) {
      for (int i = std::max(0, j - kd); i <= j; ++i) {
        band[static_cast<std::size_t>(j) * ldab + (kd + i - j)] = scaled(i, j);
      }
    }
    // This process's segments, renumbered past those the degree drops.
    auto owned = std::vector<int>{};
    for (const auto p : options.segments) {
      if (p >= dropped) owned.push_back(p - dropped);
    }
    try {
      substructure_ = std::make_shared<const detail::BandSubstructure>(
          band, n, kd, std::move(separators), owned, options.combine);
    } catch (const std::runtime_error& error) {
      throw std::runtime_error(std::format("degree {}: {}", l_, error.what()));
    }
    choleskyFailed_ = !substructure_->UsedCholesky();
    return;
  }

  if (positive) {
    factor_.assign(static_cast<std::size_t>(ldab) * n, 0.0);
    for (auto j = 0; j < n; ++j) {
      for (int i = std::max(0, j - kd); i <= j; ++i) {
        factor_[static_cast<std::size_t>(j) * ldab + (kd + i - j)] =
            scaled(i, j);
      }
    }
    if (detail::BandCholeskyFactor(factor_, n, kd) == 0) return;
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
  const auto info = detail::BandLuFactor(factor_, n, kd, kd, pivots_);
  if (info != 0) {
    throw std::runtime_error(std::format(
        "degree {}: the banded LU factorisation failed (info = {})", l_, info));
  }
}

std::vector<double> DegreeOperator::Solve(std::span<const double> b, int nrhs,
                                          RightHandSide rhs) const {
  auto pending = BeginSolve(b, nrhs, rhs);
  if (substructure_) substructure_->Sum(pending.separators);
  return FinishSolve(std::move(pending));
}

PendingSolve DegreeOperator::BeginSolve(std::span<const double> b, int nrhs,
                                        RightHandSide rhs) const {
  const int n = ndof_;
  if (static_cast<int>(b.size()) != n * nrhs) {
    throw std::invalid_argument(std::format(
        "expected {} x {} right-hand-side values, got {}", n, nrhs, b.size()));
  }
  auto pending = PendingSolve{};
  pending.nrhs = nrhs;
  auto& x = pending.values;
  x.assign(b.begin(), b.end());
  for (auto k = 0; k < nrhs; ++k) {
    for (auto j = 0; j < n; ++j)
      x[static_cast<std::size_t>(k) * n + j] *= scale_[j];
  }
  if (substructure_) {
    substructure_->Condense(x, nrhs, rhs, pending.separators,
                            pending.interiors);
  } else if (!choleskyFailed_) {
    detail::BandCholeskySolve(factor_, n, kd_, x, nrhs);
  } else {
    detail::BandLuSolve(factor_, n, kd_, kd_, pivots_, x, nrhs);
  }
  return pending;
}

std::vector<double> DegreeOperator::FinishSolve(PendingSolve&& pending) const {
  const int n = ndof_;
  auto x = std::move(pending.values);
  if (substructure_) {
    substructure_->Complete(x, pending.nrhs, pending.separators,
                            pending.interiors);
  }
  for (auto k = 0; k < pending.nrhs; ++k) {
    for (auto j = 0; j < n; ++j)
      x[static_cast<std::size_t>(k) * n + j] *= scale_[j];
  }
  return x;
}

bool DegreeOperator::Holds(int dof) const {
  return !substructure_ || substructure_->Holds(dof);
}

void DegreeOperator::MaskToShare(std::span<double> x, int nrhs) const {
  if (substructure_) substructure_->MaskToShare(x, nrhs);
}

std::vector<double> DegreeOperator::Apply(std::span<const double> x,
                                          int nrhs) const {
  return ApplyBand(band_, x, nrhs);
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
