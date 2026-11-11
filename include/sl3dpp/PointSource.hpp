#pragma once

/**
 * @file PointSource.hpp
 * @brief A point stress glut (a moment tensor) as a force on the degree
 * systems, and the strain of degree solutions at a point.
 *
 * A stress glut S enters the weak form as the work int S : e(w) dV on a
 * test displacement w (the equivalent body force is -div S). For the
 * point source S = M delta(x - x_s) the work is M : e(w)(x_s), so the
 * force on the coefficient (l, m) of each radial unknown is the
 * conjugate of M contracted with the strain, at the source, of that
 * unknown's unit mode. `PointSourceForces` builds exactly that, and
 * `PointStrain` is the evaluation it transposes. Being a stress glut and
 * not its rate, it suits steppers of the displacement: a source that
 * acts from a time on is a force held from then.
 *
 * The strain of a mode goes through the canonical components of the
 * generalised spherical harmonics (Phinney & Burridge 1973; Dahlen &
 * Tromp 1998, appendix C; Crawford et al. 2017, appendix B): with the
 * displacement u = U r Y + V grad_1 Y - W r x grad_1 Y, the deviatoric
 * strain's components at upper indices -2..2 are SpinComponents.hpp's
 * DeviatoricComponents times Y^N_lm, and the dilatation is
 * (r dU/dr + 2 U - l (l + 1) V) / r times Y_lm. The radial delta is
 * exact in the weak form: the unknowns and their radial derivatives are
 * read at the source radius through the Lagrange basis of the element
 * holding it. The source may lie anywhere in a solid element; a degree
 * whose truncated system does not reach the source's element takes no
 * force from it.
 *
 * Everything is in the units of the material the systems were built on
 * (non-dimensional under its scales): a moment has the dimensions of
 * stress times volume.
 */

#include <array>
#include <complex>
#include <functional>
#include <map>
#include <vector>

#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/ToroidalOperator.hpp"

namespace sl3dpp {

/// A symmetric tensor at a point by its components in the spherical
/// polar basis there: r up, theta south, phi east.
struct SphericalTensor {
  double rr = 0.0, tt = 0.0, pp = 0.0;
  double rt = 0.0, rp = 0.0, tp = 0.0;

  /// The full contraction A : B.
  [[nodiscard]] double Contract(const SphericalTensor& other) const {
    return rr * other.rr + tt * other.tt + pp * other.pp +
           2.0 * (rt * other.rt + rp * other.rp + tp * other.tp);
  }
};

/// A point of the body: radius, colatitude and longitude (radians).
struct BodyPoint {
  double radius = 0.0;
  double colatitude = 0.0;
  double longitude = 0.0;
};

/// A point stress glut: the moment tensor M at a point.
struct PointSource {
  BodyPoint at;
  SphericalTensor moment;
};

/// A symmetric tensor field on a sphere: its six components in the
/// spherical polar basis, each a field on a grid.
struct SphericalTensorField {
  std::vector<double> rr, tt, pp, rt, rp, tp;
};

/// A stress glut on a spherical shell, S = moment(Omega) delta(r - radius)
/// / radius^2: `moment` is the moment per unit solid angle, a tensor field
/// on `grid`, which must carry upper index 2 (SurfaceGrid::Options::nmax)
/// and outlive the source. A point source is the shell whose moment is
/// concentrated at one direction; a fault is a shell's patch, or several
/// shells'; a stress glut through a volume is a radial quadrature of
/// shells, each one's moment the glut there times its radial weight and
/// its radius squared.
struct StressGlut {
  double radius = 0.0;
  const SurfaceGrid* grid = nullptr;
  SphericalTensorField moment;
};

/// Per-degree forces or solutions in the 3D stepper's layout: complex
/// m >= 0 columns (l + 1 of them, column-major) over each system's dofs,
/// the coefficients of a real field.
struct DegreeFields {
  std::map<int, std::vector<std::complex<double>>> spheroidal;  ///< l >= 0
  std::map<int, std::vector<std::complex<double>>> toroidal;    ///< l >= 1
};

/// The strain at `point` of the real displacement field the degree
/// solutions `x` are the coefficients of, degrees 0..lmax (a degree
/// absent from `x` contributes nothing). `spheroidal(l)` and
/// `toroidal(l)` give the systems, for their dof maps.
[[nodiscard]] SphericalTensor PointStrain(
    const Material& material, int lmax,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    const DegreeFields& x, BodyPoint point);

/// The forces of a point source on the degree systems, degrees 0..lmax:
/// for any solutions x, the sum over degrees and orders (m = 0 once,
/// m > 0 twice the real part) of conj(force) . x is
/// source.moment : PointStrain(x, source.at).
[[nodiscard]] DegreeFields PointSourceForces(
    const Material& material, int lmax,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    const PointSource& source);

// -- the factored form --------------------------------------------------------
//
// A source at one radius forces the coefficient (l, m) of each unknown by
// a few radial vectors that depend on the degree alone, with amplitudes
// that carry the order: the source's moments against the generalised
// spherical harmonics of upper index -2..2 and its trace's against Y_lm.
// The spheroidal unknowns see the upper indices +N and -N alike and the
// toroidal ones with opposite signs, so there are four spheroidal vectors
// (the deviatoric N = 0 part, the trace, |N| = 1, |N| = 2) and two
// toroidal (|N| = 1, |N| = 2). The radial systems do not depend on the
// order, so the unit vectors' solutions serve every order, every
// position on the sphere and every moment tensor at that radius.

/// The moments a source presents to the modes: per (l, m >= 0), at
/// SurfaceGrid::Index(l, m), the integrals over directions of the
/// tensor's five canonical parts against Y^N_lm, N = -2..2, and of its
/// trace against Y_lm. Linear in the source: those of several sources at
/// one radius add.
struct SourceMoments {
  int lmax = 0;
  std::vector<std::array<std::complex<double>, 6>> at;
};
[[nodiscard]] SourceMoments MomentsOf(int lmax, const PointSource& source);
[[nodiscard]] SourceMoments MomentsOf(int lmax, const StressGlut& source);

/// The amplitudes of the unit force vectors at (l, m) for a set of
/// moments: spheroidal (N = 0 deviatoric, trace, |N| = 1, |N| = 2) and
/// toroidal (|N| = 1, |N| = 2).
struct UnitAmplitudes {
  std::array<std::complex<double>, 4> spheroidal{};
  std::array<std::complex<double>, 2> toroidal{};
};
[[nodiscard]] UnitAmplitudes AmplitudesOf(const SourceMoments& moments, int l,
                                          int m);

/// The unit force vectors at a radius: per degree, real (dof, column)
/// column-major arrays of four spheroidal columns and (from degree 1) two
/// toroidal ones, in AmplitudesOf's order. The force on coefficient
/// (l, m) is the columns' sum under the amplitudes.
struct UnitSourceForces {
  std::map<int, std::vector<double>> spheroidal;
  std::map<int, std::vector<double>> toroidal;
};
[[nodiscard]] UnitSourceForces UnitForces(
    const Material& material, int lmax,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal, double radius);

/// The forces of a shell source, as PointSourceForces: for any solutions
/// x, conj(force) . x summed is the integral over the shell's directions
/// of moment : PointStrain(x, there). The moment's components are
/// expanded in generalised spherical harmonics of upper index -2..2, so
/// it is taken to the grid's band limit.
[[nodiscard]] DegreeFields StressGlutForces(
    const Material& material, int lmax,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    const StressGlut& source);

}  // namespace sl3dpp
