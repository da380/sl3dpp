#pragma once

/**
 * @file Fault.hpp
 * @brief Point sources laid over a fault: a planar fault given by its
 * moment, and a fault given by its slip (planetmodel's fault file).
 *
 * A planar fault is a flat rectangle in space, given as seismologists give
 * it (Aki & Richards 2002, §4.2): the centroid's position and depth, the
 * strike (the azimuth of the fault's trace, clockwise from north, with the
 * fault dipping to its right), the dip (from the horizontal) and the rake
 * (the slip of the hanging wall in the plane, anticlockwise from the
 * strike direction), all at the centroid, with its length along strike
 * and its width down dip. With unit normal n and unit slip u the moment
 * tensor of a slip of moment M0 is M0 (n u + u n), the same tensor in
 * space at every point of a flat fault with uniform slip direction.
 *
 * `PlanarFaultSources` divides the rectangle into equal patches and puts
 * a point source at each patch's centre: the tensor's components are
 * taken in the spherical polar basis at that point, which turns with
 * position though the tensor does not. Every patch carries the same
 * share of the moment, M0 over the number of patches. A patch whose
 * centre lies above the surface is left out, its share of the moment
 * with it.
 *
 * A fault given by its slip is a `planetmodel::Fault` (the fault file,
 * `planetmodel.fault/1`, read by `planetmodel::ReadFaultFile`): patches of
 * surface, each with the displacement s of its hanging wall relative to
 * its footwall. It holds no rigidity. `FaultSourcesOf` lays planetmodel's
 * quadrature over every patch (n_u by n_v equal cells of its parameter
 * square, Gauss-Legendre points in each) and puts a point source at each
 * point, its moment tensor the point's weight w (the area it stands for)
 * times the elastic tensor C there contracted with the slip and the unit
 * normal n (from the footwall into the hanging wall), SlipMoment:
 *
 *   M = w C : (s n).
 *
 * C is transversely isotropic about the radial direction, Love's A, C, F,
 * L, N, with 1 and 2 the horizontal directions (theta, phi) and 3 the
 * radial one, and e = (s n + n s) / 2:
 *
 *   M_tt = w (A e_tt + (A - 2N) e_pp + F e_rr)
 *   M_pp = w ((A - 2N) e_tt + A e_pp + F e_rr)
 *   M_rr = w (F (e_tt + e_pp) + C e_rr)
 *   M_rt = 2 w L e_rt,   M_rp = 2 w L e_rp,   M_tp = 2 w N e_tp;
 *
 * isotropic (A = C = lambda + 2 mu, F = lambda, L = N = mu) this is
 * w (lambda (s.n) I + mu (s n + n s)), and a slip in the surface (s.n = 0)
 * gives mu w |s| (n u + u n), the planar fault's tensor of moment
 * mu w |s|. A plane of planetmodel's is the rectangle `PlanarFaultSources`
 * lays out, by the same strike, dip and rake, and its midpoint quadrature
 * on N by M cells puts its points at the patches' centres. The fault's
 * time axis is its caller's business: FaultSources holds the tensors of
 * each slice of the slip. The parts that read planetmodel's fault are
 * compiled where planetmodel's C++ library holds it (SL3DPP_HAVE_FAULT_FILE
 * defined).
 *
 * Lengths and the moment are in the caller's units (the evolution's
 * solver units, for Evolution3D::Forcing::sources); angles are degrees.
 * FaultSourcesOf works in SI: metres, Pa and N m.
 */

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Geography.hpp"
#include "sl3dpp/PointSource.hpp"

#if __has_include(<planetmodel/faultfile.hpp>)
#include <planetmodel/fault.hpp>
/// Defined where planetmodel's C++ library reads and evaluates the fault
/// file (planetmodel.fault/1): FaultSourcesOf, and the sl3d program's
/// `fault_file` term, are compiled then.
#define SL3DPP_HAVE_FAULT_FILE 1
#endif

namespace sl3dpp {

struct PlanarFault {
  GeographicPoint centroid;  ///< the centroid's latitude and longitude
  double depth = 0.0;        ///< the centroid's depth below the surface
  double strike = 0.0;       ///< degrees clockwise from north
  double dip = 90.0;         ///< degrees from the horizontal, 0 to 90
  double rake = 0.0;         ///< degrees, in the plane from the strike
  double length = 0.0;       ///< along strike
  double width = 0.0;        ///< down dip
  double moment = 0.0;       ///< the total scalar moment M0
};

/// The moment tensor of a slip of unit moment on a plane of the given
/// strike, dip and rake (degrees), in the spherical polar basis at the
/// point where they are given (r up, theta south, phi east).
[[nodiscard]] SphericalTensor DoubleCouple(double strike, double dip,
                                           double rake);

/// The fault as `alongStrike` x `downDip` point sources at its patches'
/// centres, `surfaceRadius` the radius of the surface the depth is
/// measured from. Patches above the surface are left out.
[[nodiscard]] std::vector<PointSource> PlanarFaultSources(
    const PlanarFault& fault, double surfaceRadius, int alongStrike,
    int downDip);

/// The elastic moduli of a medium transversely isotropic about the radial
/// direction (Love's): A and C the P moduli horizontally and radially, L
/// and N the shear moduli in the vertical and horizontal planes, F the
/// fifth. Isotropic: A = C = lambda + 2 mu, F = lambda, L = N = mu.
struct LoveModuli {
  double A = 0.0, C = 0.0, F = 0.0, L = 0.0, N = 0.0;
};

/// The model's moduli at radius r (metres), SI, from the layer above at a
/// boundary, its fields read as Material reads them at its nodes: `kappa`
/// and `mu` (isotropic), the velocities `vpv` and `vsv` with `vph`, `vsh`
/// and `eta` where present (A = rho vph^2, C = rho vpv^2, L = rho vsv^2,
/// N = rho vsh^2, F = eta (A - 2L)), or `vp` and `vs` (isotropic). A model
/// holding none of these is refused (std::invalid_argument).
[[nodiscard]] LoveModuli LoveModuliAt(const EarthModel& model, double r);

/// The moment tensor of a slip `slip` across an element of area `area` and
/// unit normal `normal` (from the footwall into the hanging wall) at the
/// point `at`: area C : (slip normal) (file comment), the vectors in
/// Cartesian components (z the polar axis, x through longitude zero), the
/// tensor in the spherical polar basis at the point (r up, theta south,
/// phi east).
[[nodiscard]] SphericalTensor SlipMoment(const LoveModuli& moduli,
                                         const BodyPoint& at,
                                         const std::array<double, 3>& slip,
                                         const std::array<double, 3>& normal,
                                         double area);

/// How a fault's positions are placed in a model whose surface radius is
/// not the fault's reference radius: each point at its depth below the
/// fault's reference sphere, that depth below the model's surface (moved
/// along its radius, its normal, slip and weight kept), or at its radius
/// as stored.
enum class FaultPlacement { Depth, Radius };

/// The relative tolerance within which a fault's reference radius is the
/// model's surface radius, its positions then taken as stored.
inline constexpr double kSameSurface = 1e-9;

#ifdef SL3DPP_HAVE_FAULT_FILE

/// What FaultSourcesOf lays over a fault, and on what model.
struct FaultSourceOptions {
  int cellsAlongStrike = 1;  ///< the quadrature's cells along u, every patch
  int cellsDownDip = 1;      ///< and down v
  int points = 1;            ///< Gauss-Legendre points each way in a cell
  double surfaceRadius = 0.0;  ///< the model's surface radius, m
  /// Required where the fault's reference radius is not the model's
  /// surface radius (kSameSurface); taken as given otherwise.
  std::optional<FaultPlacement> placement;
  /// The elastic moduli at a point (its radius in metres), SI.
  std::function<LoveModuli(const BodyPoint&)> moduli;
};

/// A fault's point sources: the quadrature's points, patch by patch (u
/// slowest, as planetmodel orders them), and the moment tensors there of
/// each slice of the slip.
struct FaultSources {
  std::vector<BodyPoint> points;   ///< radius in metres, as placed
  std::vector<double> depths;      ///< below the model's surface, m
  std::vector<std::size_t> patch;  ///< the patch of each point
  /// slices[k][p]: slice k's moment tensor at point p, N m.
  std::vector<std::vector<SphericalTensor>> slices;
};

/// The point sources of a fault (file comment), every slice's. Refused
/// (std::invalid_argument, naming what): a fault in a Cartesian frame (the
/// body is a sphere); a reference radius not the model's surface radius
/// without a placement; a point above the model's surface, by its depth
/// and place; a point where the moduli have no shear modulus (a fluid);
/// counts below one.
[[nodiscard]] FaultSources FaultSourcesOf(const planetmodel::Fault& fault,
                                          const FaultSourceOptions& options);

#endif

}  // namespace sl3dpp
