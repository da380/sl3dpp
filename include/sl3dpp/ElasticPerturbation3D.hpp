#pragma once

/**
 * @file ElasticPerturbation3D.hpp
 * @brief The first-order (Born) response of the static gravitating elastic
 * problem to LATERAL perturbations of the shear and bulk moduli, through
 * the grid pipeline of the 3D memory coupling: the second step of the
 * material derivatives (docs/dev/adjoints.md §13), and the first step
 * towards a 3D elastic reference for the cross-code benchmarks.
 *
 * The elastic structure of the reference problem is radial, and a lateral
 * delta mu(x) enters its bilinear form exactly as the lateral viscosity
 * term enters a step: as a pointwise weight on the deviatoric strain, fed
 * back through the force assembly (GridCoupling), which in the
 * angle-uniform limit is the 1D shear block exactly. A lateral delta
 * kappa(x) weights the dilatation, carried by the coupling's scalar
 * companion. The first-order change of a solution x is then
 *
 *   delta x = -K^-1 (dK) x,   (dK) x = Entry(delta mu . Exit(x)) + bulk term,
 *
 * the reference (radial, unrelaxed) systems solving the negated force,
 * spheroidal and toroidal degrees alike (a lateral perturbation couples
 * them, and couples the degrees by the selection rules of the products).
 * No lateral density: the hydrostatic background does not admit it.
 *
 * Conventions: directions over the coupling's grid values
 * ([FieldOffset(node) + x]), material units; the nodes are the GLL nodes
 * of every solid element (the coupling of a throwaway Maxwell rheology
 * over the solid, so the perturbation can live anywhere solid, and the
 * reference systems are the Love-number operators of the material).
 */

#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <vector>

#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/ToroidalOperator.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp {

class ElasticPerturbation3D {
 public:
  using Complex = std::complex<double>;
  using Solutions = Viscoelastic3D::Solutions;
  using Forces = std::map<int, std::vector<Complex>>;

  struct Options {
    double eps = 1e-8;
    TruncationRule truncation = TruncationRule::Graded;
    double oversampling = 1.5;
    bool parallel = true;
  };

  /// The material is copied into the rheology's shared storage, so the
  /// object never dangles.
  ElasticPerturbation3D(const Material& material, int lmax, Options options);
  ElasticPerturbation3D(const Material& material, int lmax)
      : ElasticPerturbation3D(material, lmax, Options{}) {}

  [[nodiscard]] const GridCoupling& Coupling() const { return coupling_; }
  [[nodiscard]] int LMax() const { return lmax_; }
  [[nodiscard]] const Material& GetMaterial() const { return material_; }

  /// The reference response to a surface mass load given by its m >= 0
  /// coefficients on the lmax grid (SurfaceGrid::Index; material units
  /// of surface density): per degree the unit load response (both pieces
  /// of DegreeOperator::Load) times the coefficient. Spheroidal only.
  [[nodiscard]] Solutions SolveLoad(std::span<const Complex> load) const;

  /// (dK) x: the force of a moduli perturbation on a solution, spheroidal
  /// and toroidal per degree; either direction may be empty.
  [[nodiscard]] std::pair<Forces, Forces> Force(
      const Solutions& x, std::span<const double> dMu,
      std::span<const double> dKappa) const;

  /// The first-order response -K^-1 (dK) x.
  [[nodiscard]] Solutions Respond(const Solutions& x, std::span<const double> dMu,
                                  std::span<const double> dKappa) const;

  /// The reference systems solving given forces (negated when asked).
  [[nodiscard]] Solutions Solve(const Forces& spheroidal, const Forces& toroidal,
                                bool negate = false) const;

  /// The surface values per degree, complex m >= 0 columns: U, V or phi
  /// of the spheroidal solutions, and W of the toroidal ones.
  [[nodiscard]] std::map<int, std::vector<Complex>> Surface(
      const Solutions& x, Component component) const;
  [[nodiscard]] std::map<int, std::vector<Complex>> SurfaceToroidal(
      const Solutions& x) const;

  /// The Hermitian pairing of solutions with forces over the degrees and
  /// the m >= 0 columns (m > 0 weighted twice): what a symmetric force
  /// makes symmetric.
  [[nodiscard]] static double Pair(const Solutions& x, const Forces& spheroidal,
                                   const Forces& toroidal);

  [[nodiscard]] const DegreeOperator& Spheroidal(int l) const;
  [[nodiscard]] const ToroidalOperator& Toroidal(int l) const;

 private:
  /// The material itself (the reference systems' and the perturbation's
  /// meaning: an isotropic increment of whatever anisotropy it has), and
  /// the throwaway rheology over its isotropic Voigt average, whose only
  /// roles are the coupling's node set (every solid element) and mesh — a
  /// branch may not be active where the material is anisotropic.
  Material material_;
  Rheology rheology_;
  int lmax_;
  Options options_;
  GridCoupling coupling_;
  mutable std::map<int, std::unique_ptr<DegreeOperator>> spheroidal_;
  mutable std::map<int, std::unique_ptr<ToroidalOperator>> toroidal_;
  mutable std::map<int, std::vector<double>> unitLoad_;
};

}  // namespace sl3dpp
