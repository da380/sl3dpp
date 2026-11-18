#pragma once

/**
 * @file LateralKernels.hpp
 * @brief The kernels of measurements in LATERAL structure — each branch's
 * log viscosity, and the bulk and unrelaxed shear moduli, as fields over
 * the memory nodes' angular grids — about any run, by one entry point that
 * chooses the route from the run's background (docs/adjoints.md §21).
 *
 * About a RADIAL background the forward and adjoint problems separate by
 * degree, so the lateral kernels need no 3D solve (Al-Attar & Tromp 2014;
 * Crawford et al. 2018): the radial sweeps (AdjointSolver) run as they do
 * for the radial kernels, and at each place where they pair the adjoint
 * fields with the forward ones they also expand both on the angular grids
 * of a GridCoupling over the radial model and multiply them pointwise, with
 * the 3D sweeps' energy contraction and time quadrature. The kernels are
 * then exactly those of the 3D sweeps (AdjointSolver3D) on the same model
 * with zero lateral structure — the continuous sweep's commute-error form,
 * the discrete sweep's exact transpose — and their means over the sphere
 * are the radial kernels. The static problem (one time node) is the same
 * route at one node.
 *
 * About a LATERAL background (an Evolution3D: lateral viscosity, lateral
 * moduli or both) the kernels are the 3D sweeps' as they stand, the
 * static problem's on lateral moduli at one node likewise.
 *
 * This is the route to kernels in the moduli, the static problem's
 * included; ElasticPerturbation3D::ForceTranspose, the transpose of its
 * lateral force, is the independent check of its static case.
 *
 * Either way the result is a Kernels3D on Coupling(): the layout, pairing
 * (VolumePairing) and units of the 3D sweeps, its `ice` the run's surface
 * kernels and, about a radial background, the radial kernels too.
 */

#include <map>
#include <memory>
#include <span>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/GridCoupling.hpp"

namespace sl3dpp {

/// Which lateral kernels to form, and the grids a radial run's are
/// expanded on.
struct LateralKernelOptions {
  /// The kernels in each branch's inverse relaxation time and log
  /// viscosity (Kernels3D::itauKernel3D, logViscosityKernel3D).
  bool viscosity = true;
  /// The kernels in the bulk and the unrelaxed shear modulus
  /// (Kernels3D::kappaKernel3D, muKernel3D).
  bool moduli = false;
  /// A radial background's coupling (LateralCoupling): the transform
  /// degrees by depth (GridCoupling::Options::transform_degrees), the
  /// grids' oversampling, threading and transform blocking. A lateral
  /// background's coupling is its operator's.
  std::vector<int> transform_degrees{};
  double oversampling = 1.5;
  bool parallel = true;
  bool matrix_kernel = false;
  std::size_t cache_bytes = 0;
};

/// The coupling a radial operator's lateral kernels are laid out on: the
/// operator's memory points at its lmax, over its viscous elements when
/// the viscosity's kernels alone are asked for, and over every solid
/// element when the moduli's are (an elastic element's viscosity kernel
/// is then zero), the element holding the centre at the Gauss points but
/// not at the GLL points, whose first node there is the centre itself
/// (GridCoupling::Options). These are the
/// elements the 3D operator carries for a lateral viscosity and for
/// lateral moduli over the whole solid.
[[nodiscard]] GridCoupling LateralCoupling(const ViscoelasticOperator& op,
                                           const LateralKernelOptions& options);

/**
 * @brief The lateral kernels of measurements about a run: the one entry
 * point, routed by the run's background.
 *
 * Built on a radial run (an Evolution) it takes that run's adjoint options
 * and builds the run's coupling; built on a 3D run (an Evolution3D) it
 * takes the 3D sweeps' options, its coupling the operator's. The
 * evolution is borrowed for the solver's lifetime. FirstOrderKernels takes
 * the trajectory of the run's own kind (the other is refused).
 *
 * Refused about a radial background: a run without a recorded trajectory
 * (the forward fields are the trajectory's). Distributed runs: a radial
 * run is one process's; the 3D route is distributed as the 3D sweeps are.
 */
class LateralKernelSolver {
 public:
  enum class Route {
    /// The radial sweeps with the grid expansion (a radial background,
    /// through time or at one node).
    Radial,
    /// The 3D sweeps (a lateral background).
    Lateral,
  };

  LateralKernelSolver(const Evolution& evolution, AdjointSolver::Options options,
                      LateralKernelOptions lateral = {});
  LateralKernelSolver(const Evolution3D& evolution,
                      AdjointSolver3D::Options options,
                      LateralKernelOptions lateral = {});
  ~LateralKernelSolver();
  LateralKernelSolver(LateralKernelSolver&&) noexcept;
  LateralKernelSolver& operator=(LateralKernelSolver&&) noexcept;

  [[nodiscard]] Route GetRoute() const { return route_; }
  /// The layout of the kernels: the radial run's coupling, or the 3D
  /// operator's (distributed, this rank's nodes).
  [[nodiscard]] const GridCoupling& Coupling() const { return *coupling_; }

  /// The kernels over the radial run's recorded trajectory, its time grid
  /// and measurements as AdjointSolver::FirstOrderKernels takes them (one
  /// node is the static problem); `ice` carries the radial sweep's kernels,
  /// the radial viscosity and moduli kernels among them.
  [[nodiscard]] Kernels3D FirstOrderKernels(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      const ForwardTrajectory& trajectory,
      std::span<const std::vector<double>> oceanFunctions = {},
      std::span<const std::vector<std::vector<double>>> stageOceanFunctions =
          {}) const;
  /// The kernels over the 3D run's recorded trajectory, as
  /// AdjointSolver3D::FirstOrderKernels gives them.
  [[nodiscard]] Kernels3D FirstOrderKernels(
      std::span<const double> times,
      const std::map<std::size_t, Measurement>& measurements,
      const ForwardTrajectory3D& trajectory,
      std::span<const std::vector<double>> oceanFunctions = {},
      std::span<const std::vector<std::vector<double>>> stageOceanFunctions =
          {}) const;

 private:
  Route route_;
  const Evolution* radial_ = nullptr;
  const Evolution3D* lateral_ = nullptr;
  AdjointSolver::Options radialOptions_{};
  AdjointSolver3D::Options lateralOptions_{};
  LateralKernelOptions options_;
  /// The radial run's coupling (owned), or the 3D operator's.
  std::unique_ptr<GridCoupling> owned_;
  const GridCoupling* coupling_ = nullptr;
};

}  // namespace sl3dpp
