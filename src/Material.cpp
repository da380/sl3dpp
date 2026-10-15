#include "sl3dpp/Material.hpp"

#include <algorithm>
#include <cmath>
#include <format>

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
  for (std::size_t i = 0; i < n; ++i) {
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
  for (std::size_t i = 0; i < n; ++i) {
    kappa[i] = (4.0 * A[i] + C[i] + 4.0 * F[i] - 4.0 * N[i]) / 9.0;
    mu[i] = (A[i] + C[i] - 2.0 * F[i] + 5.0 * N[i] + 6.0 * L[i]) / 15.0;
  }
  return {std::move(kappa), std::move(mu)};
}

std::expected<Material, MaterialError> Material::Create(
    const EarthModel& model, const RadialMesh& mesh) {
  const auto nd = model.DimensionlessView();
  const auto b = nd.Boundaries();
  const int last = mesh.ElementCount() - 1;
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
  material.drho_ = mesh.Nodal(model, "rho", 1);
  material.g_ = mesh.NodalGravity(model);
  material.surfaceGravity_ = material.g_.back();

  material.fluid_.resize(mesh.ElementCount());
  for (int e = 0; e <= last; ++e) {
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
    for (std::size_t i = 0; i < n; ++i) {
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
    for (std::size_t i = 0; i < n; ++i) {
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

std::expected<Material, MaterialError> Material::WithModuli(
    NodalModuli moduli) const {
  if (moduli.A.size() != rho_.size()) {
    return std::unexpected(MaterialError{
        std::format("the moduli have {} nodal values, the mesh {}",
                    moduli.A.size(), rho_.size())});
  }
  Material other = *this;
  other.moduli_ = std::move(moduli);
  return other;
}

}  // namespace sl3dpp
