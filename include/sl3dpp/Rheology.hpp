#pragma once

/**
 * @file Rheology.hpp
 * @brief Generalised Maxwell (Prony) branch data on the mesh nodes, as
 * pyslfp's `viscoelastic.rheology`; non-dimensional like every internal
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
 * describe the same body. A Burgers body (a Maxwell element in series
 * with a Kelvin element) is exactly two branches with no long-term
 * modulus (BurgersPronyTerms). mu_k = 0 deactivates a node in branch k;
 * itau_k = 0 is an infinite relaxation time. Bulk relaxation is not
 * modelled, and the material must be isotropic where any branch is active
 * (the degree-basis internal variables of A&T 2014 App. D assume isotropic
 * shear relaxation; docs/design.md).
 */

#include <array>
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

/// One relaxing shear branch of a homogeneous body: its modulus and its
/// inverse relaxation time.
struct PronyTerm {
  double mu = 0.0;
  double itau = 0.0;
};

/**
 * @brief A Burgers body's shear relaxation as Prony terms, exactly.
 *
 * The Burgers body is a Maxwell element (shear modulus mu, viscosity eta)
 * in series with a Kelvin element (modulus mu_K, viscosity eta_K). Its
 * creep function is J(t) = 1 / mu + t / eta + (1 - exp(-t mu_K / eta_K))
 * / mu_K, and its modulus in the Laplace variable s,
 * 1 / M(s) = 1 / mu + 1 / (eta s) + 1 / (mu_K + eta_K s), is that of two
 * Maxwell branches with no long-term modulus,
 * M(s) = sum_k mu_k s / (s + itau_k): the relaxation function is
 * G(t) = mu_1 exp(-itau_1 t) + mu_2 exp(-itau_2 t). With a = mu / eta,
 * b = mu / eta_K and c = mu_K / eta_K, the rates itau_k are the roots of
 * x^2 - (a + b + c) x + a c, and mu_1 = mu (c - itau_1) / (itau_2 -
 * itau_1), mu_2 = mu (itau_2 - c) / (itau_2 - itau_1); c lies between the
 * roots, so both moduli are positive. mu_1 + mu_2 = mu (the unrelaxed
 * modulus) and mu_1 / itau_1 + mu_2 / itau_2 = eta (the steady
 * viscosity). Any consistent units; every argument positive and finite
 * (std::invalid_argument otherwise). The slower term first.
 */
[[nodiscard]] std::array<PronyTerm, 2> BurgersPronyTerms(
    double mu, double eta, double kelvinModulus, double kelvinViscosity);

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

  /// The transient of a Burgers layer, a Kelvin element in series with
  /// the layer's Maxwell element: its shear modulus as a fraction of the
  /// material's unrelaxed one, and its viscosity, each a constant or a
  /// function of radius, non-dimensional; both positive (the default
  /// viscosity, zero, is refused).
  struct Kelvin {
    RadialValue modulus_fraction = 1.0;
    RadialValue viscosity = 0.0;
  };

  /// Maxwell and Burgers layers: `viscosity` gives each relaxing layer's
  /// steady viscosity, as Maxwell's (unnamed layers stay elastic), and
  /// each layer in `kelvin`, which must be in `viscosity` too, is a
  /// Burgers body, its Kelvin element the entry's. Two branches
  /// throughout: the first the Maxwell layers' branch and the Burgers
  /// layers' slower term, the second their faster term and zero elsewhere
  /// (BurgersPronyTerms at every node).
  [[nodiscard]] static std::expected<Rheology, RheologyError> Burgers(
      const Material& material,
      const std::map<std::size_t, RadialValue>& viscosity,
      const std::map<std::size_t, Kelvin>& kelvin);

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
