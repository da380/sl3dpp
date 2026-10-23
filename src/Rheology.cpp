#include "sl3dpp/Rheology.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>

namespace sl3dpp {

namespace {

double Evaluate(const RadialValue& value, double r) {
  if (std::holds_alternative<double>(value)) return std::get<double>(value);
  return std::get<std::function<double(double)>>(value)(r);
}

}  // namespace

std::expected<Rheology, RheologyError> Rheology::Create(
    const Material& material, std::vector<Branch> branches) {
  const auto& mesh = material.Mesh();
  const auto size = material.Rho().size();
  const auto n = mesh.NGll();

  Rheology rheology;
  rheology.material_ = std::make_shared<const Material>(material);
  for (auto k = std::size_t{0}; k < branches.size(); ++k) {
    for (const auto* a : {&branches[k].mu, &branches[k].itau}) {
      if (a->size() != size) {
        return std::unexpected(RheologyError{std::format(
            "branch {}: arrays must have the mesh's {} nodal values", k,
            size)});
      }
      for (double v : *a) {
        if (!std::isfinite(v) || v < 0.0) {
          return std::unexpected(RheologyError{std::format(
              "branch {}: mu and itau must be finite and >= 0", k)});
        }
      }
    }
  }
  rheology.branches_ = std::move(branches);

  auto [kappa, mu] = material.Moduli().KappaMu();
  rheology.kappa_ = std::move(kappa);
  rheology.muUnrelaxed_ = std::move(mu);
  rheology.muLongTerm_ = rheology.muUnrelaxed_;
  rheology.active_.assign(size, 0);
  auto scale = 0.0;
  for (double v : rheology.muUnrelaxed_) scale = std::max(scale, v);
  for (const auto& branch : rheology.branches_) {
    for (auto i = std::size_t{0}; i < size; ++i) {
      rheology.muLongTerm_[i] -= branch.mu[i];
      if (branch.mu[i] > 0.0) rheology.active_[i] = 1;
    }
  }
  constexpr auto rtol = 1e-10;
  for (auto& v : rheology.muLongTerm_) {
    if (v < -rtol * scale) {
      return std::unexpected(RheologyError{
          "the branch moduli exceed the material's shear modulus: "
          "mu_inf = mu - sum_k mu_k would be negative"});
    }
    v = std::max(v, 0.0);
  }

  // No relaxing fluid, and isotropy where a branch is active.
  const auto iso =
      NodalModuli::Isotropic(rheology.kappa_, rheology.muUnrelaxed_);
  const auto& m = material.Moduli();
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto i = 0; i < n; ++i) {
      const auto node = static_cast<std::size_t>(e) * n + i;
      if (!rheology.active_[node]) continue;
      if (material.IsFluid(e)) {
        return std::unexpected(
            RheologyError{"a fluid element cannot carry a relaxing branch"});
      }
      for (const auto& [actual, isotropic] :
           {std::pair{&m.A, &iso.A}, std::pair{&m.C, &iso.C},
            std::pair{&m.F, &iso.F}, std::pair{&m.L, &iso.L},
            std::pair{&m.N, &iso.N}}) {
        if (std::abs((*actual)[node] - (*isotropic)[node]) > rtol * scale) {
          return std::unexpected(RheologyError{
              "the material is anisotropic where a branch relaxes; the "
              "degree-basis internal variables assume isotropic shear "
              "relaxation (collapse to the Voigt average first)"});
        }
      }
    }
  }
  return rheology;
}

std::expected<Rheology, RheologyError> Rheology::Maxwell(
    const Material& material,
    const std::map<std::size_t, RadialValue>& viscosity) {
  const auto& mesh = material.Mesh();
  const auto n = mesh.NGll();
  auto [kappa, mu] = material.Moduli().KappaMu();
  Branch branch;
  branch.mu.assign(mu.size(), 0.0);
  branch.itau.assign(mu.size(), 0.0);
  for (const auto& [layer, value] : viscosity) {
    auto found = false;
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      if (static_cast<std::size_t>(mesh.LayerOf(e)) != layer) continue;
      found = true;
      if (material.IsFluid(e)) {
        return std::unexpected(RheologyError{
            std::format("layer {} is fluid and cannot relax", layer)});
      }
      for (auto i = 0; i < n; ++i) {
        const auto node = static_cast<std::size_t>(e) * n + i;
        const auto eta = Evaluate(value, mesh.Radius(e, i));
        if (!(eta > 0.0)) {
          return std::unexpected(RheologyError{"viscosity must be positive"});
        }
        branch.mu[node] = mu[node];
        branch.itau[node] = mu[node] / eta;
      }
    }
    if (!found) {
      return std::unexpected(RheologyError{
          std::format("the mesh has no elements in layer {}", layer)});
    }
  }
  return Create(material, {std::move(branch)});
}

std::expected<Rheology, RheologyError> Rheology::Maxwell(
    const Material& material, RadialValue viscosity) {
  const auto& mesh = material.Mesh();
  auto perLayer = std::map<std::size_t, RadialValue>{};
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (!material.IsFluid(e)) {
      // Assigned rather than emplaced (the same value for every layer):
      // GCC 14 warns spuriously when a variant holding a std::function is
      // emplaced into a map.
      perLayer[static_cast<std::size_t>(mesh.LayerOf(e))] = viscosity;
    }
  }
  return Maxwell(material, perLayer);
}

std::expected<Rheology, RheologyError> Rheology::FromModel(
    const Material& material, const EarthModel& model) {
  const auto& mesh = material.Mesh();
  const auto nd = model.DimensionlessView();
  auto viscosity = std::map<std::size_t, RadialValue>{};
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    const auto layer = static_cast<std::size_t>(mesh.LayerOf(e));
    if (model.Has("viscosity", layer) && !viscosity.contains(layer)) {
      viscosity.emplace(layer, [nd, layer](double r) {
        return nd.EvaluateInLayer("viscosity", layer, r);
      });
    }
  }
  if (viscosity.empty()) {
    return std::unexpected(RheologyError{
        "no layer of the model holds a viscosity field; give the rheology "
        "explicitly (Rheology::Maxwell, say)"});
  }
  return Maxwell(material, viscosity);
}

double Rheology::MinRelaxationTime() const {
  auto worst = 0.0;
  for (const auto& branch : branches_) {
    for (auto i = std::size_t{0}; i < branch.mu.size(); ++i) {
      if (branch.mu[i] > 0.0) worst = std::max(worst, branch.itau[i]);
    }
  }
  return worst > 0.0 ? 1.0 / worst : std::numeric_limits<double>::infinity();
}

NodalModuli Rheology::EffectiveModuli(
    std::span<const std::vector<double>> weights) const {
  if (weights.size() != branches_.size()) {
    throw std::invalid_argument(std::format("expected {} weight arrays, got {}",
                                            branches_.size(), weights.size()));
  }
  auto out = material_->Moduli();
  for (auto k = std::size_t{0}; k < branches_.size(); ++k) {
    const auto& branch = branches_[k];
    for (auto i = std::size_t{0}; i < branch.mu.size(); ++i) {
      const double delta = (weights[k][i] - 1.0) * branch.mu[i];
      out.A[i] += (4.0 / 3.0) * delta;
      out.C[i] += (4.0 / 3.0) * delta;
      out.F[i] -= (2.0 / 3.0) * delta;
      out.L[i] += delta;
      out.N[i] += delta;
    }
  }
  return out;
}

Material Rheology::LongTermMaterial() const {
  std::vector<std::vector<double>> zeros(
      branches_.size(), std::vector<double>(kappa_.size(), 0.0));
  return material_->WithModuli(EffectiveModuli(zeros)).value();
}

}  // namespace sl3dpp
