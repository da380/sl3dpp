#pragma once

/**
 * @file ElasticPerturbation3D.hpp
 * @brief The static gravitating elastic problem under LATERAL
 * perturbations of the shear and bulk moduli, through the grid pipeline
 * of the 3D memory coupling (docs/dev/adjoints.md §13b, §13j): its
 * first-order (Born) response, and the full solve of the laterally
 * heterogeneous problem by preconditioned conjugate gradients with the
 * reference systems as the preconditioner (SolveLateral), for loads,
 * body tides and measurement duals (the adjoint under the structure).
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
 * No lateral density: the hydrostatic background does not admit it
 * (docs/dev/adjoints.md §13l).
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

  /// The load's forces per degree (what SolveLoad solves): the unit load
  /// vector times the coefficients, spheroidal.
  [[nodiscard]] Forces LoadForces(std::span<const Complex> load) const;
  /// The forces of an external (tidal) potential given by its m >= 0
  /// coefficients on the lmax grid (material units of potential at the
  /// surface): per degree DegreeOperator::Tide(power) times the
  /// coefficient, the power defaulting to the degree (the harmonic body
  /// tide); a power of 2 at degree 0 is the spherical mean of a
  /// centrifugal change. Spheroidal.
  [[nodiscard]] Forces TideForces(std::span<const Complex> amplitudes,
                                  int power = -1) const;
  /// The forces of a surface dual (what SolveSurfaceDual solves): the
  /// dual's coefficients at the component's surface dof.
  [[nodiscard]] Forces SurfaceDualForces(std::span<const Complex> dual,
                                         Component component) const;
  /// The full 3D operator's action (K_ref + dK) x: the reference systems'
  /// (the degree-1 toroidal unpinned) plus the lateral force, per degree.
  [[nodiscard]] std::pair<Forces, Forces> Apply(const Solutions& x,
                                                std::span<const double> dMu,
                                                std::span<const double> dKappa) const;

  struct SolveOptions {
    /// The relative residual (in the preconditioned, energy norm,
    /// relative to the load's) to stop at, and the iteration budget.
    double rtol = 1e-10;
    int max_iterations = 200;
  };
  struct SolveInfo {
    int iterations = 0;
    double residual = 0.0;
    bool converged = false;
    /// The relative residual after each iteration (the first entry the
    /// reference response's).
    std::vector<double> residuals;
    /// Whether a curvature <p, A p> or <r, M r> came out non-positive:
    /// the iteration continued on magnitudes, but the operator pair is
    /// then not definite and conjugate gradients are not guaranteed.
    bool indefinite = false;
  };
  /// The full response of the laterally heterogeneous static problem,
  /// (K_ref + dK) x = f, by preconditioned conjugate gradients with the
  /// reference solve as the preconditioner (Maitra & Al-Attar 2019; the
  /// lateral action pseudospectral through the grid pipeline), started
  /// from the reference response so that the first iterate is the Born
  /// response; the correction is orthogonal to the degree-1 toroidal
  /// rigid rotations, which the pinned reference solve removes.
  [[nodiscard]] std::pair<Solutions, SolveInfo> SolveLateral(
      std::span<const Complex> load, std::span<const double> dMu,
      std::span<const double> dKappa, SolveOptions options) const;
  [[nodiscard]] std::pair<Solutions, SolveInfo> SolveLateral(
      std::span<const Complex> load, std::span<const double> dMu,
      std::span<const double> dKappa) const {
    return SolveLateral(load, dMu, dKappa, SolveOptions{});
  }
  /// The same for any forces (a load's, a tide's, a surface dual's or
  /// their sum; toroidal forces allowed): the operator is symmetric, so
  /// the solve of a measurement's dual forces under the lateral
  /// structure is the adjoint solution of that structure, and
  /// -StaticKernels(y, x) are then the measurement's kernels in the
  /// moduli at the laterally heterogeneous model.
  [[nodiscard]] std::pair<Solutions, SolveInfo> SolveLateral(
      const Forces& spheroidal, const Forces& toroidal,
      std::span<const double> dMu, std::span<const double> dKappa,
      SolveOptions options) const;

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

  /// The static material kernels of a pair of solutions: dJ/d kappa(x),
  /// dJ/d mu(x) over the grid values as angular densities per node (the
  /// radial weight folded in, the itauKernel3D layout) for J = <y, K x>
  /// — the transpose of Force: sum_node int (kappa dKappa + mu dMu)
  /// dOmega = Pair(y, Force(x, dMu, dKappa)). With y the response to a
  /// measurement's dual (SolveSurfaceDual), the kernels of that
  /// measurement of the static problem.
  struct Kernels {
    std::vector<double> kappa;
    std::vector<double> mu;
  };
  [[nodiscard]] Kernels StaticKernels(const Solutions& y,
                                      const Solutions& x) const;

  /// The reference response to a dual on a surface component: per degree
  /// the m >= 0 coefficients of a surface field's dual (the pairing
  /// weights of the measurement), placed at the component's surface dof
  /// and solved by the spheroidal systems — the adjoint solution of the
  /// static problem for a surface measurement.
  [[nodiscard]] Solutions SolveSurfaceDual(std::span<const Complex> dual,
                                           Component component) const;

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
