#pragma once

// The lateral kernels about a radial background, formed inside the 1D
// sweeps (docs/adjoints.md §21): the sweeps hand this accumulator the
// degree-domain fields the 3D sweeps' kernel integrands read — the forward
// strain and internal variables, the adjoint internal variables or the
// duals of the decayed memory, the adjoint and forward solutions — and it
// expands them on the angular grids of a GridCoupling over the radial
// model's memory points (GridCoupling::AddScalarState,
// AddToroidalScalarState, AddStrainOfSolution, AddDilatationOfSolution),
// multiplies them pointwise with the 3D sweeps' energy contraction and
// accumulates them with the 3D sweeps' weights. At zero lateral structure
// the 3D sweeps' grid fields ARE these expansions (the step is laterally
// uniform), so the kernels are the 3D sweeps' kernels of the radial model.
//
// Conventions. The 1D arrays are the radial sweeps' own: per degree, the
// 2 (l + 1) real columns of a real field (the real block of orders 0..l,
// then the imaginary). A forward state, the continuous adjoint state and
// the solutions are natural coefficients; a discrete sweep's duals carry
// the doubling of the m > 0 orders (the real-column pairing), and the
// duals of the decayed memory also the memory pairing's weights
// k-weight x Wq per scalar, which the expansion divides out to give the 3D
// sweep's energy-pairing duals.

#include <map>
#include <span>
#include <vector>

#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/ToroidalViscoelastic.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace sl3dpp::detail {

class LateralAccumulator {
 public:
  using Fields = std::map<int, std::vector<double>>;
  using ToroidalState = ToroidalViscoelastic::State;

  /// The coupling holds the radial operator's memory points and the
  /// elements the kernels live on (both borrowed); `viscosity` and
  /// `moduli` say which kernels are formed.
  LateralAccumulator(const ViscoelasticOperator& op,
                     const GridCoupling& coupling, bool viscosity, bool moduli);

  [[nodiscard]] bool Viscosity() const { return viscosity_; }
  [[nodiscard]] bool Moduli() const { return moduli_; }
  [[nodiscard]] const GridCoupling& Coupling() const { return *coupling_; }

  // -- the continuous sweep -------------------------------------------------

  /// The multiplier integrand at a node: scale mu_k C(m-dagger_k, d - m_k)
  /// / itau_k per branch (the 3D continuous sweep's accumulateItau).
  void ContinuousItau(double scale, const ViscoelasticState& adjoint,
                      const ToroidalState* adjointToroidal,
                      const ViscoelasticState& forward,
                      const ToroidalState* forwardToroidal);
  /// The material pairing at a node: the adjoint displacement lambda (and
  /// its toroidal part) against the forward full solution x, strain and
  /// memory (the 3D continuous sweep's pairMaterial).
  void ContinuousMaterial(double scale, const Fields& lambda,
                          const Fields* lambdaToroidal, const Fields& x,
                          const ViscoelasticState& forward,
                          const ToroidalState* forwardToroidal);

  // -- the discrete sweep ---------------------------------------------------

  /// A reversed step's itau pairing: the duals of the decayed memory
  /// against dt (dE m + dA d0 + dB d1) of the forward step from `prev` to
  /// `next` (the 3D discrete sweep's).
  void DiscreteItau(double dt, Scheme scheme, const Fields& decayedDual,
                    const Fields* decayedDualToroidal,
                    const ViscoelasticState& prev,
                    const ViscoelasticState& next,
                    const ToroidalState* prevToroidal,
                    const ToroidalState* nextToroidal);
  /// A reversed step's material pairing: the step's dual solution y
  /// against the forward full solution x at the step's end (its strain
  /// under the effective factor), and the decayed memory of the step from
  /// `prev` (the toroidal end strain `nextToroidal`'s).
  void DiscreteMaterial(double dt, Scheme scheme, const Fields& y,
                        const Fields* yToroidal, const Fields& x,
                        const ViscoelasticState& prev,
                        const ToroidalState* prevToroidal,
                        const ToroidalState* nextToroidal);
  /// The start's: the transposed unrelaxed solve's dual solution against
  /// the forward start solution, every weight one.
  void DiscreteStart(const Fields& y, const Fields* yToroidal, const Fields& x,
                     const Fields* xToroidal);

  /// The kernels as the 3D sweeps return them (Kernels3D's layout and
  /// convention on the coupling), `ice` the radial sweep's own.
  [[nodiscard]] Kernels3D Finish(Kernels ice);

 private:
  enum class Columns { Natural, Dual, DecayedDual };
  void AddSpheroidal(GridCoupling::SpectralTensor& to, int l,
                     const double* scalars, Columns columns) const;
  void AddToroidal(GridCoupling::SpectralTensor& to, int l,
                   const double* scalars, Columns columns) const;
  [[nodiscard]] GridCoupling::TensorField StrainOf(const Fields& x,
                                                   const Fields* toroidal,
                                                   Columns columns) const;
  [[nodiscard]] GridCoupling::SpectralTensor SpectralStrainOf(
      const Fields& x, const Fields* toroidal, Columns columns) const;
  /// Adds the strain of a forward solution (natural columns, spheroidal),
  /// each node's coefficients times its factor.
  void AddScaledStrain(GridCoupling::SpectralTensor& to, const Fields& x,
                       std::span<const double> factor) const;
  [[nodiscard]] std::vector<double> DilatationOf(const Fields& x,
                                                 Columns columns) const;
  void AddMaterial(double scale, const GridCoupling::TensorField& adjointStrain,
                   std::span<const double> adjointDilatation,
                   const GridCoupling::TensorField& forward,
                   std::span<const double> forwardDilatation);

  const ViscoelasticOperator* op_;
  /// The toroidal systems' dof maps and windows (its own instance: the
  /// operators only).
  ToroidalViscoelastic toroidal_;
  const GridCoupling* coupling_;
  bool viscosity_, moduli_;
  std::size_t branches_;
  int lmax_;
  /// Per coupling node: its point index over the whole mesh, (element,
  /// point), and its element.
  std::vector<std::size_t> pointOf_;
  std::vector<int> elementOf_;
  std::vector<std::vector<double>> itau_, mu_;
  std::vector<std::vector<double>> fractions_;
  /// The kernels as accumulated: angular densities with each node's
  /// radial weight folded in, as the 3D sweeps accumulate theirs.
  std::vector<std::vector<double>> itauKernel_;
  std::vector<double> kappaKernel_, muKernel_;
};

}  // namespace sl3dpp::detail
