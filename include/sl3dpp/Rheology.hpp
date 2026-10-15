#pragma once

/**
 * @file Rheology.hpp
 * @brief Generalised Maxwell (Prony) branch data on the mesh nodes: a port
 * of pyslfp's `viscoelastic.rheology`, non-dimensional like every internal
 * component.
 *
 * A Rheology pairs an elastic Material (whose moduli are the UNRELAXED
 * ones) with relaxing shear branches: per branch k, nodal arrays mu_k and
 * itau_k = 1 / tau_k. The long-term shear modulus is
 * mu_inf = mu_U - sum_k mu_k with mu_U the Voigt shear modulus; the stress
 * is T = kappa tr(eps) I + 2 mu_inf d + sum_k 2 mu_k (d - m_k) with
 * dm_k/dt = (d - m_k) / tau_k. The classical Maxwell body is one branch
 * with mu_1 = mu_U and tau = eta / mu_U — planetmodel's `viscosity`
 * convention, so a model frozen at a frequency and the time stepping
 * describe the same body. mu_k = 0 deactivates a node in branch k;
 * itau_k = 0 is an infinite relaxation time. Bulk relaxation is not
 * modelled, and the material must be isotropic where any branch is active
 * (the degree-basis internal variables of A&T 2014 App. D assume isotropic
 * shear relaxation; the TI generalisation is plan open question 7).
 */

#include <expected>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "sl3dpp/Material.hpp"

namespace sl3dpp {

/// Why a Rheology could not be built.
struct RheologyError {
  std::string message;
};

/// A radial prescription: a constant, or a function of (non-dimensional)
/// radius.
using RadialValue = std::variant<double, std::function<double(double)>>;

class Rheology {
 public:
  /// One relaxing shear branch: nodal arrays over the mesh, (element,
  /// node) row-major as RadialMesh::Nodal lays them out.
  struct Branch {
    std::vector<double> mu;
    std::vector<double> itau;  ///< 1 / tau, zero meaning infinite
  };

  /// From explicit branches; refused when a branch array has the wrong
  /// size or a negative entry, when sum_k mu_k exceeds the material's
  /// shear modulus, when a fluid node relaxes, or when the material is
  /// anisotropic where a branch is active. The material is copied into
  /// immutable shared storage, so a Rheology is freely copyable and
  /// movable and never dangles.
  [[nodiscard]] static std::expected<Rheology, RheologyError> Create(
      const Material& material, std::vector<Branch> branches);

  /// A classical Maxwell body, tau = eta / mu: `viscosity` per layer
  /// index (unnamed layers stay elastic), each a constant or a function
  /// of radius, non-dimensional.
  [[nodiscard]] static std::expected<Rheology, RheologyError> Maxwell(
      const Material& material,
      const std::map<std::size_t, RadialValue>& viscosity);

  /// A Maxwell body over every solid element under one viscosity.
  [[nodiscard]] static std::expected<Rheology, RheologyError> Maxwell(
      const Material& material, RadialValue viscosity);

  /// The rheology the model's own fields declare: each layer holding a
  /// `viscosity` field is a Maxwell body with tau = viscosity / mu.
  [[nodiscard]] static std::expected<Rheology, RheologyError> FromModel(
      const Material& material, const EarthModel& model);

  /// Stable for the lifetime of any copy of this Rheology.
  [[nodiscard]] const Material& BaseMaterial() const { return *material_; }
  [[nodiscard]] std::size_t BranchCount() const { return branches_.size(); }
  [[nodiscard]] const Branch& BranchAt(std::size_t k) const {
    return branches_[k];
  }

  /// The Voigt pair of the unrelaxed material and the long-term shear.
  [[nodiscard]] std::span<const double> Kappa() const { return kappa_; }
  [[nodiscard]] std::span<const double> MuUnrelaxed() const {
    return muUnrelaxed_;
  }
  [[nodiscard]] std::span<const double> MuLongTerm() const {
    return muLongTerm_;
  }

  /// Whether any branch relaxes at a node.
  [[nodiscard]] bool IsActive(std::size_t node) const {
    return active_[node] != 0;
  }

  /// The smallest relaxation time of any active node: the explicit
  /// stability scale; infinity when nothing relaxes.
  [[nodiscard]] double MinRelaxationTime() const;

  /// The moduli with each branch's shear scaled by its nodal weight,
  /// through the isotropic perturbation rule (A, C += 4/3 dmu,
  /// F -= 2/3 dmu, L, N += dmu), leaving inactive regions untouched.
  [[nodiscard]] NodalModuli EffectiveModuli(
      std::span<const std::vector<double>> weights) const;

  /// The fully relaxed elastic solid (every branch weight zero): the
  /// t -> infinity limit a time-domain run must approach.
  [[nodiscard]] Material LongTermMaterial() const;

 private:
  Rheology() = default;

  std::shared_ptr<const Material> material_;  ///< owned, immutable
  std::vector<Branch> branches_;
  std::vector<double> kappa_, muUnrelaxed_, muLongTerm_;
  std::vector<char> active_;
};

}  // namespace sl3dpp
