#include "sl3dpp/Material.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>

namespace sl3dpp {

NodalModuli NodalModuli::Isotropic(std::span<const double> kappa,
                                   std::span<const double> mu) {
  NodalModuli m;
  const auto n = kappa.size();
  m.A.resize(n);
  m.C.resize(n);
  m.F.resize(n);
  m.L.assign(mu.begin(), mu.end());
  m.N.assign(mu.begin(), mu.end());
  for (auto i = std::size_t{0}; i < n; ++i) {
    m.A[i] = kappa[i] + 4.0 * mu[i] / 3.0;
    m.C[i] = m.A[i];
    m.F[i] = kappa[i] - 2.0 * mu[i] / 3.0;
  }
  return m;
}

std::pair<std::vector<double>, std::vector<double>> NodalModuli::KappaMu()
    const {
  const auto n = A.size();
  std::vector<double> kappa(n), mu(n);
  for (auto i = std::size_t{0}; i < n; ++i) {
    kappa[i] = (4.0 * A[i] + C[i] + 4.0 * F[i] - 4.0 * N[i]) / 9.0;
    mu[i] = (A[i] + C[i] - 2.0 * F[i] + 5.0 * N[i] + 6.0 * L[i]) / 15.0;
  }
  return {std::move(kappa), std::move(mu)};
}

std::expected<Material, MaterialError> Material::Create(
    const EarthModel& model, const RadialMesh& mesh) {
  const auto nd = model.DimensionlessView();
  const auto b = nd.Boundaries();
  const auto last = mesh.ElementCount() - 1;
  if (mesh.Left(0) != b.front() || mesh.Right(last) != b.back()) {
    return std::unexpected(MaterialError{
        std::format("the mesh spans [{:g}, {:g}] but the model [{:g}, {:g}] "
                    "(non-dimensional); truncate the model, not the mesh",
                    mesh.Left(0), mesh.Right(last), b.front(), b.back())});
  }
  if (model.IsFluid(static_cast<std::size_t>(mesh.LayerOf(last)))) {
    return std::unexpected(MaterialError{
        "the surface of the model is fluid, so it cannot carry a load; "
        "build the model without its ocean"});
  }

  Material material(mesh);
  material.scales_ = model.ModelScales();
  material.G_ = nd.G();
  material.radius_ = mesh.Right(last);
  material.rho_ = mesh.Nodal(model, "rho");
  material.rhoPotential_ = material.rho_;
  material.drho_ = mesh.Nodal(model, "rho", 1);
  material.g_ = mesh.NodalGravity(model);
  material.surfaceGravity_ = material.g_.back();

  material.fluid_.resize(mesh.ElementCount());
  for (auto e = 0; e <= last; ++e) {
    material.fluid_[e] =
        model.IsFluid(static_cast<std::size_t>(mesh.LayerOf(e))) ? 1 : 0;
  }

  // The moduli, from whichever fields the model holds.
  if (model.Has("kappa") && model.Has("mu")) {
    material.moduli_ = NodalModuli::Isotropic(mesh.Nodal(model, "kappa"),
                                              mesh.Nodal(model, "mu"));
  } else if (model.Has("vpv") && model.Has("vsv")) {
    const auto vpv = mesh.Nodal(model, "vpv");
    const auto vsv = mesh.Nodal(model, "vsv");
    const auto vph = model.Has("vph") ? mesh.Nodal(model, "vph") : vpv;
    const auto vsh = model.Has("vsh") ? mesh.Nodal(model, "vsh") : vsv;
    const auto eta = model.Has("eta") ? mesh.Nodal(model, "eta")
                                      : std::vector<double>(vpv.size(), 1.0);
    const auto n = vpv.size();
    auto& m = material.moduli_;
    m.A.resize(n);
    m.C.resize(n);
    m.F.resize(n);
    m.L.resize(n);
    m.N.resize(n);
    for (auto i = std::size_t{0}; i < n; ++i) {
      const double rho = material.rho_[i];
      m.A[i] = rho * vph[i] * vph[i];
      m.C[i] = rho * vpv[i] * vpv[i];
      m.L[i] = rho * vsv[i] * vsv[i];
      m.N[i] = rho * vsh[i] * vsh[i];
      m.F[i] = eta[i] * (m.A[i] - 2.0 * m.L[i]);
    }
  } else if (model.Has("vp") && model.Has("vs")) {
    const auto vp = mesh.Nodal(model, "vp");
    const auto vs = mesh.Nodal(model, "vs");
    const auto n = vp.size();
    std::vector<double> kappa(n), mu(n);
    for (auto i = std::size_t{0}; i < n; ++i) {
      const double rho = material.rho_[i];
      mu[i] = rho * vs[i] * vs[i];
      kappa[i] = rho * vp[i] * vp[i] - 4.0 * mu[i] / 3.0;
    }
    material.moduli_ = NodalModuli::Isotropic(kappa, mu);
  } else {
    return std::unexpected(MaterialError{
        "the model holds no moduli: it needs kappa and mu, vpv and vsv "
        "(with optional vph, vsh, eta), or vp and vs"});
  }

  return material;
}

namespace {

/// Gauss-Legendre nodes and weights of order n on [-1, 1] (Newton on the
/// Legendre polynomial from the standard cosine guesses).
std::pair<std::vector<double>, std::vector<double>> GaussLegendre(int n) {
  std::vector<double> x(n), w(n);
  for (auto i = 0; i < n; ++i) {
    auto z = std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5));
    double dp = 0.0;
    for (auto it = 0; it < 100; ++it) {
      // P_n(z) and P_n'(z) by the recurrence.
      double p0 = 1.0, p1 = z;
      for (auto k = 2; k <= n; ++k) {
        const auto p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      dp = n * (z * p1 - p0) / (z * z - 1.0);
      const auto dz = p1 / dp;
      z -= dz;
      if (std::abs(dz) < 1e-15) break;
    }
    x[i] = z;
    w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
  }
  return {x, w};
}

/// The integral of a nodal field f over the mesh from its bottom to each
/// node: within an element, the exact integral of the Lagrange
/// interpolant of f from the left end to the node (the integrals of the
/// GLL basis functions, P(i, j) = int_{-1}^{xi_i} l_j, by Gauss-Legendre
/// quadrature on each part), the elements' totals accumulated below (the
/// last row of P is the GLL weights, so the totals are the quadrature's).
/// P(i, j) = int_{-1}^{xi_i} l_j dt on the reference element: the partial
/// integrals of the Lagrange basis up to each GLL point, by Gauss-Legendre.
std::vector<double> PartialIntegrals(const RadialMesh& mesh) {
  const auto n = mesh.NGll();
  const auto xi = mesh.Points();
  auto lagrange = [&](int j, double t) {
    auto v = 1.0;
    for (auto k = 0; k < n; ++k) {
      if (k != j) v *= (t - xi[k]) / (xi[j] - xi[k]);
    }
    return v;
  };
  const auto [gx, gw] = GaussLegendre(n);
  auto P = std::vector<double>(static_cast<std::size_t>(n) * n, 0.0);
  for (auto i = 0; i < n; ++i) {
    const auto half = 0.5 * (xi[i] + 1.0);
    const auto mid = 0.5 * (xi[i] - 1.0);
    for (auto j = 0; j < n; ++j) {
      auto sum = 0.0;
      for (auto q = 0; q < n; ++q) sum += gw[q] * lagrange(j, mid + half * gx[q]);
      P[static_cast<std::size_t>(i) * n + j] = half * sum;
    }
  }
  return P;
}

/// The transpose of CumulativeIntegral: for duals v of the cumulative
/// integral at every node, the nodal array t with sum_i v_i out_i =
/// sum_j t_j f_j — each node's partial-integral weights within its
/// element, plus the element's full-integral weight times the duals of
/// every node above the element.
std::vector<double> CumulativeIntegralTranspose(const RadialMesh& mesh,
                                                std::span<const double> v) {
  const auto n = mesh.NGll();
  const auto P = PartialIntegrals(mesh);
  auto out = std::vector<double>(v.size(), 0.0);
  auto tail = 0.0;  // the duals of every node above the element
  for (auto e = mesh.ElementCount() - 1; e >= 0; --e) {
    const auto base = static_cast<std::size_t>(e) * n;
    const auto jac = mesh.Jacobian(e);
    for (auto j = 0; j < n; ++j) {
      auto sum = tail * P[static_cast<std::size_t>(n - 1) * n + j];
      for (auto i = 0; i < n; ++i) {
        sum += v[base + i] * P[static_cast<std::size_t>(i) * n + j];
      }
      out[base + j] = jac * sum;
    }
    for (auto i = 0; i < n; ++i) tail += v[base + i];
  }
  return out;
}

std::vector<double> CumulativeIntegral(const RadialMesh& mesh,
                                       std::span<const double> f) {
  const auto n = mesh.NGll();
  const auto P = PartialIntegrals(mesh);
  auto out = std::vector<double>(f.size(), 0.0);
  auto below = 0.0;
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    const auto base = static_cast<std::size_t>(e) * n;
    const auto jac = mesh.Jacobian(e);
    for (auto i = 0; i < n; ++i) {
      auto sum = 0.0;
      for (auto j = 0; j < n; ++j) {
        sum += P[static_cast<std::size_t>(i) * n + j] * f[base + j];
      }
      out[base + i] = below + jac * sum;
    }
    below = out[base + n - 1];
  }
  return out;
}

}  // namespace

MaterialVariation Material::Variation(const MaterialDirection& d) const {
  const auto& mesh = mesh_;
  const auto size = rho_.size();
  for (const auto* a : {&d.kappa, &d.mu, &d.rho}) {
    if (!a->empty() && a->size() != size) {
      throw std::invalid_argument(std::format(
          "a material direction's arrays have the mesh's {} nodal values",
          size));
    }
  }
  const auto zeros = std::vector<double>(size, 0.0);
  MaterialVariation out;
  out.moduli = NodalModuli::Isotropic(d.kappa.empty() ? zeros : d.kappa,
                                      d.mu.empty() ? zeros : d.mu);
  out.rho = d.rho.empty() ? zeros : d.rho;
  out.drho.assign(size, 0.0);
  out.g.assign(size, 0.0);
  if (!d.rho.empty()) {
    const auto n = mesh.NGll();
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      const auto base = static_cast<std::size_t>(e) * n;
      const auto jac = mesh.Jacobian(e);
      for (auto i = 0; i < n; ++i) {
        auto v = 0.0;
        for (auto j = 0; j < n; ++j) v += mesh.Derivative(i, j) * d.rho[base + j];
        out.drho[base + i] = v / jac;
      }
    }
    auto f = std::vector<double>(size);
    for (auto i = std::size_t{0}; i < size; ++i) {
      const auto r = mesh.Radii()[i];
      f[i] = d.rho[i] * r * r;
    }
    const auto mass = CumulativeIntegral(mesh, f);
    for (auto i = std::size_t{0}; i < size; ++i) {
      const auto r = mesh.Radii()[i];
      out.g[i] = r > 0.0 ? 4.0 * std::numbers::pi * G_ * mass[i] / (r * r) : 0.0;
    }
  }
  out.surfaceGravity = out.g.back();
  return out;
}

std::vector<double> Material::DensityGradient(const DensityDuals& duals) const {
  const auto& mesh = mesh_;
  const auto size = rho_.size();
  for (const auto* a : {&duals.rho, &duals.drho, &duals.g}) {
    if (a->size() != size) {
      throw std::invalid_argument(std::format(
          "the density duals have the mesh's {} nodal values", size));
    }
  }
  auto grad = duals.rho;
  // The element differentiation's transpose.
  const auto n = mesh.NGll();
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    const auto base = static_cast<std::size_t>(e) * n;
    const auto jac = mesh.Jacobian(e);
    for (auto i = 0; i < n; ++i) {
      const auto w = duals.drho[base + i] / jac;
      if (w == 0.0) continue;
      for (auto j = 0; j < n; ++j) grad[base + j] += w * mesh.Derivative(i, j);
    }
  }
  // Gravity: g_i = 4 pi G mass_i / r_i^2 with mass the cumulative integral
  // of rho r^2; the surface gravity's dual joins g's at the top node.
  auto v = std::vector<double>(size, 0.0);
  for (auto i = std::size_t{0}; i < size; ++i) {
    const auto r = mesh.Radii()[i];
    v[i] = r > 0.0 ? 4.0 * std::numbers::pi * G_ * duals.g[i] / (r * r) : 0.0;
  }
  {
    const auto r = mesh.Radii()[size - 1];
    v[size - 1] += 4.0 * std::numbers::pi * G_ * duals.surfaceGravity / (r * r);
  }
  const auto t = CumulativeIntegralTranspose(mesh, v);
  for (auto i = std::size_t{0}; i < size; ++i) {
    const auto r = mesh.Radii()[i];
    grad[i] += t[i] * r * r;
  }
  return grad;
}

Material Material::Shifted(const MaterialDirection& d, double eps) const {
  const auto v = Variation(d);
  auto other = *this;
  const auto size = rho_.size();
  for (auto i = std::size_t{0}; i < size; ++i) {
    other.moduli_.A[i] += eps * v.moduli.A[i];
    other.moduli_.C[i] += eps * v.moduli.C[i];
    other.moduli_.F[i] += eps * v.moduli.F[i];
    other.moduli_.L[i] += eps * v.moduli.L[i];
    other.moduli_.N[i] += eps * v.moduli.N[i];
    other.rho_[i] += eps * v.rho[i];
    other.rhoPotential_[i] += eps * v.rho[i];  // a self-consistent shift
    other.drho_[i] += eps * v.drho[i];
    other.g_[i] += eps * v.g[i];
  }
  other.surfaceGravity_ += eps * v.surfaceGravity;
  return other;
}

std::expected<Material, MaterialError> Material::WithModuli(
    NodalModuli moduli) const {
  if (moduli.A.size() != rho_.size()) {
    return std::unexpected(MaterialError{
        std::format("the moduli have {} nodal values, the mesh {}",
                    moduli.A.size(), rho_.size())});
  }
  auto other = *this;
  other.moduli_ = std::move(moduli);
  return other;
}

std::expected<Material, MaterialError> Material::WithDensityGravityHeld(
    std::vector<double> rho) const {
  if (rho.size() != rho_.size()) {
    return std::unexpected(MaterialError{
        std::format("the density has {} nodal values, the mesh {}", rho.size(),
                    rho_.size())});
  }
  auto other = *this;
  other.rho_ = std::move(rho);  // rhoPotential_, g_, drho_ stay
  return other;
}

}  // namespace sl3dpp
