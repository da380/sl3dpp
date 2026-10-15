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

class Material {
 public:
  /// The nodal arrays of a model on a mesh; the mesh is copied in.
  [[nodiscard]] static std::expected<Material, MaterialError> Create(
      const EarthModel& model, const RadialMesh& mesh);

  [[nodiscard]] const RadialMesh& Mesh() const { return mesh_; }

  // Nodal arrays, row-major (element, node), as RadialMesh::Nodal lays
  // them out.
  [[nodiscard]] std::span<const double> Rho() const { return rho_; }
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

  /// The scales carried from the model, for dimensionalising outputs.
  [[nodiscard]] const Scales& ModelScales() const { return scales_; }

 private:
  explicit Material(RadialMesh mesh) : mesh_(std::move(mesh)) {}

  RadialMesh mesh_;
  std::vector<double> rho_;
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
