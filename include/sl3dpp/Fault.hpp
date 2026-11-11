#pragma once

/**
 * @file Fault.hpp
 * @brief Point sources laid over a planar fault.
 *
 * A fault is a flat rectangle in space, given as seismologists give it
 * (Aki & Richards 2002, §4.2): the centroid's position and depth, the
 * strike (the azimuth of the fault's trace, clockwise from north, with
 * the fault dipping to its right), the dip (from the horizontal) and the
 * rake (the slip of the hanging wall in the plane, anticlockwise from
 * the strike direction), all at the centroid, with its length along
 * strike and its width down dip. With unit normal n and unit slip u the
 * moment tensor of a slip of moment M0 is M0 (n u + u n), the same
 * tensor in space at every point of a flat fault with uniform slip
 * direction.
 *
 * `PlanarFaultSources` divides the rectangle into equal patches and puts
 * a point source at each patch's centre: the tensor's components are
 * taken in the spherical polar basis at that point, which turns with
 * position though the tensor does not. Every patch carries the same
 * share of the moment, M0 over the number of patches. A patch whose
 * centre lies above the surface is left out, its share of the moment
 * with it.
 *
 * Lengths and the moment are in the caller's units (the evolution's
 * solver units, for Evolution3D::Forcing::sources); angles are degrees.
 */

#include <vector>

#include "sl3dpp/Geography.hpp"
#include "sl3dpp/PointSource.hpp"

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

}  // namespace sl3dpp
