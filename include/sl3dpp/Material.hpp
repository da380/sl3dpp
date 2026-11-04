#pragma once

/**
 * @file Material.hpp
 * @brief What the per-degree radial solver needs of a model on a mesh, as
 * nodal arrays: pyslfp's `love_numbers.material`, non-dimensional like
 * every internal component.
 *
 * `NodalModuli` holds the transversely isotropic moduli A, C, F, L, N at
 * the mesh nodes, per element, so a shared node carries both one-sided
 * values. They are read from the model's fields: `kappa` and `mu` give the
 * isotropic A = C = kappa + 4 mu / 3, F = kappa - 2 mu / 3, L = N = mu;
 * velocities give A = rho vph^2, C = rho vpv^2, L = rho vsv^2,
 * N = rho vsh^2, F = eta (A - 2 L), with vph = vpv, vsh = vsv, eta = 1
 * where the transverse columns are absent. A fluid layer gets L = N = 0
 * and A = C = F = kappa either way.
 *
 * `Material` bundles, once, everything a `DegreeOperator` assembles from:
 * density and its radial derivative, gravity, the fluid flag of every
 * element and the moduli. It owns a copy of the mesh, refuses a mesh that
 * does not span the whole model, and refuses a fluid surface (the loaded
 * surface must be solid).
 */

#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/RadialMesh.hpp"

namespace sl3dpp {

/// Why a Material could not be built.
struct MaterialError {
  std::string message;
};

/// The moduli A, C, F, L, N at the mesh nodes, per element, row-major
/// (element, node), non-dimensional.
struct NodalModuli {
  std::vector<double> A;
  std::vector<double> C;
  std::vector<double> F;
  std::vector<double> L;
  std::vector<double> N;

  /// A = C = kappa + 4 mu / 3, F = kappa - 2 mu / 3, L = N = mu.
  [[nodiscard]] static NodalModuli Isotropic(std::span<const double> kappa,
                                             std::span<const double> mu);

  /// The Voigt-averaged isotropic (kappa, mu) of the five.
  [[nodiscard]] std::pair<std::vector<double>, std::vector<double>> KappaMu()
      const;
};

/// A direction in the material's nodal fields — the isotropic moduli
/// kappa and mu and the density rho — each (element, node) over the mesh
/// in the material's non-dimensional units, or empty for no perturbation.
/// The parameters of the material derivatives (docs/dev/adjoints.md).
struct MaterialDirection {
  std::vector<double> kappa, mu, rho;
};

/// The linearised nodal fields along a direction: the moduli's variation
/// (A, C, F, L, N from delta kappa and delta mu), the density's, its radial
/// derivative's (the element differentiation of delta rho, one-sided at
/// interfaces), gravity's (4 pi G r^-2 of the integral of delta rho r^2
/// from the bottom of the mesh, elementwise spectral integration — the
/// natural integral of a nodal field, exact for the polynomial it
/// interpolates) and the surface gravity's.
struct MaterialVariation {
  NodalModuli moduli;
  std::vector<double> rho, drho, g;
  double surfaceGravity = 0.0;
};

/// The duals of a density variation's fields — what pairs with a
/// MaterialVariation's rho, drho and g (nodal) and its surfaceGravity —
/// which Material::DensityGradient maps to a nodal density gradient: the
/// transpose of Variation in the density. DegreeOperator::AddDensityDuals
/// and the forcings' accumulators fill them.
struct DensityDuals {
  std::vector<double> rho, drho, g;
  double surfaceGravity = 0.0;

  void Resize(std::size_t size) {
    rho.assign(size, 0.0);
    drho.assign(size, 0.0);
    g.assign(size, 0.0);
    surfaceGravity = 0.0;
  }
  void Add(const DensityDuals& other, double scale = 1.0) {
    for (auto i = std::size_t{0}; i < rho.size(); ++i) {
      rho[i] += scale * other.rho[i];
      drho[i] += scale * other.drho[i];
      g[i] += scale * other.g[i];
    }
    surfaceGravity += scale * other.surfaceGravity;
  }
};

class Material {
 public:
  /// The nodal arrays of a model on a mesh; the mesh is copied in.
  [[nodiscard]] static std::expected<Material, MaterialError> Create(
      const EarthModel& model, const RadialMesh& mesh);

  [[nodiscard]] const RadialMesh& Mesh() const { return mesh_; }

  // Nodal arrays, row-major (element, node), as RadialMesh::Nodal lays
  // them out.
  [[nodiscard]] std::span<const double> Rho() const { return rho_; }
  /// The density of the REFERENCE POTENTIAL: what Poisson's equation
  /// puts into the gravity gradient's 4 pi G rho piece of the UU
  /// coupling. The density itself, except for WithDensityGravityHeld,
  /// which holds it with the gravity (the lateral density's convention).
  [[nodiscard]] std::span<const double> PotentialDensity() const {
    return rhoPotential_;
  }
  [[nodiscard]] std::span<const double> DRho() const { return drho_; }
  [[nodiscard]] std::span<const double> Gravity() const { return g_; }
  [[nodiscard]] const NodalModuli& Moduli() const { return moduli_; }

  [[nodiscard]] bool IsFluid(int element) const { return fluid_[element] != 0; }

  [[nodiscard]] double G() const { return G_; }

  /// The outer boundary radius and the gravity there, non-dimensional.
  [[nodiscard]] double Radius() const { return radius_; }
  [[nodiscard]] double SurfaceGravity() const { return surfaceGravity_; }

  /// The same material with the moduli replaced (the fluid limit, a frozen
  /// rheology); the replacement must have the mesh's nodal size.
  [[nodiscard]] std::expected<Material, MaterialError> WithModuli(
      NodalModuli moduli) const;
  /// The same material with the nodal density replaced and EVERYTHING
  /// ELSE HELD: gravity, its surface value, the density gradient (the
  /// stratification) and the moduli stay as they were. Not a
  /// self-consistent body — it is the 1D image of the lateral density's
  /// convention (docs/dev/adjoints.md §13l), for gating the 3D density
  /// pipeline against the radial operator.
  [[nodiscard]] std::expected<Material, MaterialError> WithDensityGravityHeld(
      std::vector<double> rho) const;

  /// The scales carried from the model, for dimensionalising outputs.
  [[nodiscard]] const Scales& ModelScales() const { return scales_; }

  /// The variation of the nodal fields along a direction (above); a
  /// direction's arrays must have the mesh's nodal size.
  [[nodiscard]] MaterialVariation Variation(const MaterialDirection& d) const;

  /// The material with its nodal fields moved by eps times the variation
  /// along a direction (gravity included): the map whose derivative at
  /// eps = 0 the material derivatives are, exactly. The fluid flags, the
  /// mesh and the scales stay.
  [[nodiscard]] Material Shifted(const MaterialDirection& d, double eps) const;

  /// The transpose of Variation in the density: the nodal density
  /// gradient of a pairing whose duals with the density, its radial
  /// derivative, gravity and the surface gravity are given — the density's
  /// own dual, the element differentiation's transpose, and the
  /// transposed cumulative integral of 4 pi G (dual of g) / r^2 times
  /// r^2 (the surface gravity's dual joining g's at the top node).
  [[nodiscard]] std::vector<double> DensityGradient(
      const DensityDuals& duals) const;

 private:
  explicit Material(RadialMesh mesh) : mesh_(std::move(mesh)) {}

  RadialMesh mesh_;
  std::vector<double> rho_;
  std::vector<double> rhoPotential_;
  std::vector<double> drho_;
  std::vector<double> g_;
  std::vector<char> fluid_;  // per element
  NodalModuli moduli_;
  double G_ = 1.0;
  double radius_ = 1.0;
  double surfaceGravity_ = 1.0;
  Scales scales_;
};

}  // namespace sl3dpp
