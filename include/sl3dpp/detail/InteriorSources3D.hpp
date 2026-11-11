#pragma once

/**
 * @file detail/InteriorSources3D.hpp
 * @brief The forces of interior sources on the 3D stepper's degree
 * systems: shared by the evolution and the tangent-linear run.
 */

#include <vector>

#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/Scales.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp::detail {

/// The forces of point sources and stress gluts given in the solver's
/// units (`toNdLength` the material-to-solver length ratio), summed, in
/// the material's and the stepper's layout; empty maps for no sources.
[[nodiscard]] inline DegreeFields ForcesOfSources3D(
    const Viscoelastic3D& op, const Scales& sle, double toNdLength,
    std::vector<PointSource> sources, std::vector<StressGlut> gluts) {
  auto out = DegreeFields{};
  if (sources.empty() && gluts.empty()) return out;
  const auto& material = op.GetRheology().BaseMaterial();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto toMatMoment =
      sle.Factor(energy) / material.ModelScales().Factor(energy);
  const std::function<const DegreeOperator&(int)> sph =
      [&](int l) -> const DegreeOperator& { return op.UnrelaxedSpheroidal(l); };
  const std::function<const ToroidalOperator&(int)> tor =
      [&](int l) -> const ToroidalOperator& { return op.UnrelaxedToroidal(l); };
  auto add = [&](DegreeFields forces) {
    auto join = [](auto& to, auto& from) {
      for (auto& [l, f] : from) {
        auto& t = to[l];
        if (t.empty()) {
          t = std::move(f);
        } else {
          for (auto i = std::size_t{0}; i < t.size(); ++i) t[i] += f[i];
        }
      }
    };
    join(out.spheroidal, forces.spheroidal);
    join(out.toroidal, forces.toroidal);
  };
  for (auto& source : sources) {
    source.at.radius /= toNdLength;
    for (auto* c : {&source.moment.rr, &source.moment.tt, &source.moment.pp,
                    &source.moment.rt, &source.moment.rp, &source.moment.tp}) {
      *c *= toMatMoment;
    }
    add(PointSourceForces(material, op.LMax(), sph, tor, source));
  }
  for (auto& glut : gluts) {
    glut.radius /= toNdLength;
    for (auto* c : {&glut.moment.rr, &glut.moment.tt, &glut.moment.pp,
                    &glut.moment.rt, &glut.moment.rp, &glut.moment.tp}) {
      for (auto& v : *c) v *= toMatMoment;
    }
    add(StressGlutForces(material, op.LMax(), sph, tor, glut));
  }
  return out;
}

}  // namespace sl3dpp::detail
