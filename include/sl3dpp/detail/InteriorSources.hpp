#pragma once

/**
 * @file detail/InteriorSources.hpp
 * @brief The forces of interior sources on a radial model's degree
 * systems, in the layouts its steppers take: shared by the evolution and
 * the tangent-linear run.
 */

#include <complex>
#include <functional>
#include <map>
#include <vector>

#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/Scales.hpp"
#include "sl3dpp/ToroidalViscoelastic.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace sl3dpp::detail {

/// The moment's factor from the solver's units to the material's.
[[nodiscard]] inline double MomentToMaterial(const ViscoelasticOperator& op,
                                             const Scales& sle) {
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  return sle.Factor(energy) / op.BaseMaterial().ModelScales().Factor(energy);
}

/// The forces of point sources and stress gluts given in the solver's
/// units (`toNdLength` the material-to-solver length ratio), summed, in
/// the material's.
[[nodiscard]] inline RadialSourceForces ForcesOfSources(
    const ViscoelasticOperator& op, const ToroidalViscoelastic& toroidal,
    const Scales& sle, double toNdLength, std::vector<PointSource> sources,
    std::vector<StressGlut> gluts) {
  auto out = RadialSourceForces{};
  const auto& material = op.BaseMaterial();
  const auto toMatMoment = MomentToMaterial(op, sle);
  const std::function<const DegreeOperator&(int)> sph =
      [&](int l) -> const DegreeOperator& { return op.UnrelaxedSystem(l); };
  const std::function<const ToroidalOperator&(int)> tor =
      [&](int l) -> const ToroidalOperator& { return toroidal.Unrelaxed(l); };
  // complex m >= 0 columns to the real block then the imaginary block
  auto add = [](std::map<int, std::vector<double>>& to,
                const std::map<int, std::vector<std::complex<double>>>& from) {
    for (const auto& [l, b] : from) {
      const auto ndof = b.size() / (static_cast<std::size_t>(l) + 1);
      auto& real = to[l];
      if (real.empty()) real.assign(2 * b.size(), 0.0);
      for (auto m = 0; m <= l; ++m) {
        for (auto d = std::size_t{0}; d < ndof; ++d) {
          const auto v = b[static_cast<std::size_t>(m) * ndof + d];
          real[static_cast<std::size_t>(m) * ndof + d] += v.real();
          real[static_cast<std::size_t>(l + 1 + m) * ndof + d] += v.imag();
        }
      }
    }
  };
  for (auto& source : sources) {
    source.at.radius /= toNdLength;
    for (auto* c : {&source.moment.rr, &source.moment.tt, &source.moment.pp,
                    &source.moment.rt, &source.moment.rp, &source.moment.tp}) {
      *c *= toMatMoment;
    }
    const auto forces = PointSourceForces(material, op.LMax(), sph, tor, source);
    add(out.spheroidal, forces.spheroidal);
    add(out.toroidal, forces.toroidal);
  }
  for (auto& glut : gluts) {
    glut.radius /= toNdLength;
    for (auto* c : {&glut.moment.rr, &glut.moment.tt, &glut.moment.pp,
                    &glut.moment.rt, &glut.moment.rp, &glut.moment.tp}) {
      for (auto& v : *c) v *= toMatMoment;
    }
    const auto forces = StressGlutForces(material, op.LMax(), sph, tor, glut);
    add(out.spheroidal, forces.spheroidal);
    add(out.toroidal, forces.toroidal);
  }
  return out;
}

}  // namespace sl3dpp::detail
