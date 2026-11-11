#include "sl3dpp/PointSource.hpp"

#include <GSHTrans/All>
#include <array>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>

#include "sl3dpp/SpinComponents.hpp"

namespace sl3dpp {

namespace {

using Complex = std::complex<double>;
constexpr Complex kI(0.0, 1.0);

/// The element holding a radius and the Lagrange basis of its nodes
/// there: values and radial derivatives.
struct RadialPoint {
  int element = 0;
  double radius = 0.0;
  std::vector<double> value, derivative;
};

RadialPoint LocateRadius(const Material& material, double r) {
  const auto& mesh = material.Mesh();
  const auto n = mesh.NGll();
  const auto count = mesh.ElementCount();
  const auto bottom = mesh.Radius(0, 0);
  const auto top = mesh.Radius(count - 1, n - 1);
  if (!(r >= bottom && r <= top)) {
    throw std::invalid_argument(std::format(
        "the point's radius {} is outside the body, {} to {}", r, bottom, top));
  }
  // the element above, where the radius is a shared node
  auto element = count - 1;
  for (auto e = 0; e < count; ++e) {
    if (r >= mesh.Radius(e, 0) && r < mesh.Radius(e, n - 1)) {
      element = e;
      break;
    }
  }
  if (material.IsFluid(element)) {
    throw std::invalid_argument(std::format(
        "the point's radius {} is in a fluid region, which holds no stress "
        "glut and has no strain here",
        r));
  }
  auto out = RadialPoint{.element = element,
                         .radius = r,
                         .value = std::vector<double>(n),
                         .derivative = std::vector<double>(n)};
  for (auto i = 0; i < n; ++i) {
    const auto ri = mesh.Radius(element, i);
    auto value = 1.0, derivative = 0.0;
    for (auto j = 0; j < n; ++j) {
      if (j == i) continue;
      const auto rj = mesh.Radius(element, j);
      // the product rule, the running derivative first
      derivative = derivative * (r - rj) / (ri - rj) + value / (ri - rj);
      value *= (r - rj) / (ri - rj);
    }
    out.value[i] = value;
    out.derivative[i] = derivative;
  }
  return out;
}

/// The generalised spherical harmonics Y^N_lm at a point, N = -2..2,
/// degrees to lmax and orders 0..l.
class HarmonicsAtPoint {
 public:
  HarmonicsAtPoint(int lmax, double theta, double phi) : phi_(phi) {
    for (auto n = -2; n <= 2; ++n) {
      if (std::abs(n) > lmax) continue;
      tables_[n + 2] = Table(lmax, lmax, n, theta);
      has_[n + 2] = true;
    }
  }
  [[nodiscard]] Complex operator()(int n, int l, int m) const {
    if (l < std::abs(n) || !has_[n + 2]) return 0.0;
    return tables_[n + 2][l][m] * std::exp(kI * (m * phi_));
  }

 private:
  using Table = GSHTrans::Wigner<double, GSHTrans::All, GSHTrans::Single,
                                 GSHTrans::Single>;
  std::array<Table, 5> tables_;
  std::array<bool, 5> has_{};
  double phi_;
};

/// The strain at the point of one complex mode (l, m), from the radial
/// scalars and their r d_r derivatives at the point: the canonical
/// components of the deviatoric part times Y^N_lm, turned to the
/// spherical polar basis, and a third of the dilatation on the diagonal.
/// The components are complex, those of the complex mode.
struct ComplexTensor {
  Complex rr, tt, pp, rt, rp, tp;
};

ComplexTensor ModeStrain(int l, int m, const DegreeNodeValues& values, double r,
                         const HarmonicsAtPoint& Y) {
  const auto A = DeviatoricComponents(l, values, r);
  // d^{--}, d^{-0}, d^{-+} = d^{00} / 2, d^{0+}, d^{++}
  const auto dmm = A[0] * Y(-2, l, m);
  const auto dm0 = A[1] * Y(-1, l, m);
  const auto d00 = 2.0 * A[2] * Y(0, l, m);
  const auto d0p = A[3] * Y(1, l, m);
  const auto dpp = A[4] * Y(2, l, m);
  const auto third = (values.rdU + 2.0 * values.U - l * (l + 1.0) * values.V) /
                     (3.0 * r) * Y(0, l, m);
  // With u^(+-) = (-+ u_theta + i u_phi) / sqrt 2:
  //   d^{0 +-} = (-+ d_rt + i d_rp) / sqrt 2,
  //   d^{+- +-} = (d_tt - d_pp) / 2 -+ i d_tp,
  //   d^{+-} = -(d_tt + d_pp) / 2 = d^{00} / 2 for a trace-free tensor.
  auto out = ComplexTensor{};
  out.rr = d00 + third;
  out.rt = (dm0 - d0p) / std::numbers::sqrt2;
  out.rp = (dm0 + d0p) / (kI * std::numbers::sqrt2);
  const auto difference = 0.5 * (dpp + dmm);  // (d_tt - d_pp) / 2
  out.tp = (dmm - dpp) / (2.0 * kI);
  out.tt = -0.5 * d00 + difference + third;
  out.pp = -0.5 * d00 - difference + third;
  return out;
}

Complex Contract(const SphericalTensor& M, const ComplexTensor& e) {
  return M.rr * e.rr + M.tt * e.tt + M.pp * e.pp +
         2.0 * (M.rt * e.rt + M.rp * e.rp + M.tp * e.tp);
}

/// The six unit modes of a degree at a radial node j of the element: the
/// unknown U, V or W one at that node, whose value and r d_r derivative
/// at the point are the node's Lagrange value and r times its derivative.
DegreeNodeValues UnitMode(int which, double value, double rDerivative) {
  auto v = DegreeNodeValues{};
  if (which == 0) {
    v.U = value;
    v.rdU = rDerivative;
  } else if (which == 1) {
    v.V = value;
    v.rdV = rDerivative;
  } else {
    v.W = value;
    v.rdW = rDerivative;
  }
  return v;
}

}  // namespace

SphericalTensor PointStrain(
    const Material& material, int lmax,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    const DegreeFields& x, BodyPoint point) {
  const auto& mesh = material.Mesh();
  const auto n = mesh.NGll();
  const auto at = LocateRadius(material, point.radius);
  const HarmonicsAtPoint Y(lmax, point.colatitude, point.longitude);
  auto total = ComplexTensor{};
  for (auto l = 0; l <= lmax; ++l) {
    const auto sph = x.spheroidal.find(l);
    const auto tor = x.toroidal.find(l);
    if (sph == x.spheroidal.end() && tor == x.toroidal.end()) continue;
    for (auto m = 0; m <= l; ++m) {
      auto values = DegreeNodeValues{};
      auto gather = [&](Complex& value, Complex& rd, const auto& dofOf,
                        const std::vector<Complex>& column, std::size_t ndof) {
        for (auto j = 0; j < n; ++j) {
          const auto dof = dofOf(mesh.GlobalIndex(at.element, j));
          if (dof < 0) continue;
          const auto c = column[static_cast<std::size_t>(m) * ndof + dof];
          value += at.value[j] * c;
          rd += at.radius * at.derivative[j] * c;
        }
      };
      if (sph != x.spheroidal.end()) {
        const auto& op = spheroidal(l);
        if (at.element >= op.FirstElement()) {
          const auto ndof = static_cast<std::size_t>(op.DofCount());
          gather(values.U, values.rdU,
                 [&](int g) { return op.Dof(g, Component::U); }, sph->second,
                 ndof);
          gather(values.V, values.rdV,
                 [&](int g) { return op.Dof(g, Component::V); }, sph->second,
                 ndof);
        }
      }
      if (l >= 1 && tor != x.toroidal.end()) {
        const auto& op = toroidal(l);
        if (at.element >= op.FirstElement()) {
          gather(values.W, values.rdW, [&](int g) { return op.Dof(g); },
                 tor->second, static_cast<std::size_t>(op.DofCount()));
        }
      }
      const auto e = ModeStrain(l, m, values, at.radius, Y);
      // a real field: m = 0 once, m > 0 with its conjugate partner
      const auto weight = m == 0 ? 1.0 : 2.0;
      total.rr += weight * e.rr;
      total.tt += weight * e.tt;
      total.pp += weight * e.pp;
      total.rt += weight * e.rt;
      total.rp += weight * e.rp;
      total.tp += weight * e.tp;
    }
  }
  return {.rr = total.rr.real(),
          .tt = total.tt.real(),
          .pp = total.pp.real(),
          .rt = total.rt.real(),
          .rp = total.rp.real(),
          .tp = total.tp.real()};
}

namespace {

/// The moments a source presents to the modes of (l, m): the integrals
/// over directions of its tensor's canonical parts against Y^N_lm,
/// N = -2..2, and of its trace against Y_lm. With them the work of the
/// source on a mode whose deviatoric strain components are A (SpinComponents'
/// order) and whose dilatation is theta is
///   sum_N moment[N] a_N + trace theta / 3,   a = (A0, A1, 2 A2, A3, A4).
struct ModeMoments {
  std::array<Complex, 5> canonical{};
  Complex trace{};
};

/// The canonical parts of a tensor, the coefficients of the mode strain's
/// components in the contraction M : e (ModeStrain's turn to the
/// spherical basis, transposed).
std::array<Complex, 5> CanonicalParts(double rr, double tt, double pp, double rt,
                                      double rp, double tp) {
  const auto half = 0.5 * (tt - pp);
  return {Complex(half, -tp), std::numbers::sqrt2 * Complex(rt, -rp),
          Complex(rr - 0.5 * (tt + pp), 0.0),
          -std::numbers::sqrt2 * Complex(rt, rp), Complex(half, tp)};
}

/// The forces of a source at a radius from its moments.
DegreeFields ForcesFromMoments(
    const Material& material, int lmax,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal, double radius,
    const std::function<ModeMoments(int l, int m)>& momentsOf) {
  const auto& mesh = material.Mesh();
  const auto n = mesh.NGll();
  const auto at = LocateRadius(material, radius);
  auto out = DegreeFields{};
  for (auto l = 0; l <= lmax; ++l) {
    const auto& sph = spheroidal(l);
    const auto sphDofs = static_cast<std::size_t>(sph.DofCount());
    auto bs = std::vector<Complex>(sphDofs * (static_cast<std::size_t>(l) + 1),
                                   0.0);
    const ToroidalOperator* tor = l >= 1 ? &toroidal(l) : nullptr;
    const auto torDofs =
        tor != nullptr ? static_cast<std::size_t>(tor->DofCount()) : 0;
    auto bt = std::vector<Complex>(torDofs * (static_cast<std::size_t>(l) + 1),
                                   0.0);
    const bool reachesS = at.element >= sph.FirstElement();
    const bool reachesT = tor != nullptr && at.element >= tor->FirstElement();
    for (auto m = 0; m <= l; ++m) {
      const auto moments = momentsOf(l, m);
      for (auto j = 0; j < n; ++j) {
        const auto global = mesh.GlobalIndex(at.element, j);
        const auto value = at.value[j];
        const auto rd = at.radius * at.derivative[j];
        auto forceOf = [&](int which) {
          const auto mode = UnitMode(which, value, rd);
          const auto A = DeviatoricComponents(l, mode, at.radius);
          const auto theta =
              (mode.rdU + 2.0 * mode.U - l * (l + 1.0) * mode.V) / at.radius;
          const auto work =
              moments.canonical[0] * A[0] + moments.canonical[1] * A[1] +
              moments.canonical[2] * 2.0 * A[2] + moments.canonical[3] * A[3] +
              moments.canonical[4] * A[4] + moments.trace * theta / 3.0;
          return std::conj(work);
        };
        if (reachesS) {
          if (const auto dof = sph.Dof(global, Component::U); dof >= 0) {
            bs[static_cast<std::size_t>(m) * sphDofs + dof] += forceOf(0);
          }
          if (const auto dof = sph.Dof(global, Component::V); dof >= 0) {
            bs[static_cast<std::size_t>(m) * sphDofs + dof] += forceOf(1);
          }
        }
        if (reachesT) {
          if (const auto dof = tor->Dof(global); dof >= 0) {
            bt[static_cast<std::size_t>(m) * torDofs + dof] += forceOf(2);
          }
        }
      }
    }
    out.spheroidal.emplace(l, std::move(bs));
    if (tor != nullptr) out.toroidal.emplace(l, std::move(bt));
  }
  return out;
}

}  // namespace

SourceMoments MomentsOf(int lmax, const PointSource& source) {
  const HarmonicsAtPoint Y(lmax, source.at.colatitude, source.at.longitude);
  const auto& M = source.moment;
  const auto parts = CanonicalParts(M.rr, M.tt, M.pp, M.rt, M.rp, M.tp);
  const auto trace = M.rr + M.tt + M.pp;
  auto out = SourceMoments{
      .lmax = lmax,
      .at = std::vector<std::array<Complex, 6>>(
          SurfaceGrid::Index(lmax, lmax) + 1)};
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      auto& to = out.at[SurfaceGrid::Index(l, m)];
      for (auto N = -2; N <= 2; ++N) to[N + 2] = parts[N + 2] * Y(N, l, m);
      to[5] = trace * Y(0, l, m);
    }
  }
  return out;
}

SourceMoments MomentsOf(int lmax, const StressGlut& source) {
  if (source.grid == nullptr) {
    throw std::invalid_argument("a stress glut needs the grid of its moment");
  }
  const auto& grid = *source.grid;
  if (grid.NMax() < 2 || grid.LMax() < lmax) {
    throw std::invalid_argument(std::format(
        "a stress glut's grid must carry upper index 2 and degree {}: it "
        "carries {} and {}",
        lmax, grid.NMax(), grid.LMax()));
  }
  const auto& S = source.moment;
  const auto size = grid.FieldSize();
  for (const auto* f : {&S.rr, &S.tt, &S.pp, &S.rt, &S.rp, &S.tp}) {
    if (f->size() != size) {
      throw std::invalid_argument(
          "a stress glut's moment components live on its grid");
    }
  }
  // int C_N(S) Y^N_lm dOmega is the conjugate of the spin-N coefficient
  // of conj(C_N(S)); the trace likewise at N = 0.
  auto out = SourceMoments{
      .lmax = lmax,
      .at = std::vector<std::array<Complex, 6>>(
          SurfaceGrid::Index(lmax, lmax) + 1)};
  auto field = std::vector<Complex>(size);
  for (auto N = -2; N <= 2; ++N) {
    for (auto k = std::size_t{0}; k < size; ++k) {
      field[k] = std::conj(
          CanonicalParts(S.rr[k], S.tt[k], S.pp[k], S.rt[k], S.rp[k], S.tp[k])
              [N + 2]);
    }
    const auto coefficients = grid.ExpandSpin(N, field);
    for (auto l = std::abs(N); l <= lmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        out.at[SurfaceGrid::Index(l, m)][N + 2] =
            std::conj(coefficients[SurfaceGrid::SpinIndex(N, l, m)]);
      }
    }
  }
  for (auto k = std::size_t{0}; k < size; ++k) {
    field[k] = S.rr[k] + S.tt[k] + S.pp[k];
  }
  const auto trace = grid.ExpandSpin(0, field);
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      out.at[SurfaceGrid::Index(l, m)][5] =
          std::conj(trace[SurfaceGrid::SpinIndex(0, l, m)]);
    }
  }
  return out;
}

namespace {

DegreeFields ForcesOfMoments(
    const Material& material, int lmax,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal, double radius,
    const SourceMoments& moments) {
  return ForcesFromMoments(material, lmax, spheroidal, toroidal, radius,
                           [&](int l, int m) {
                             const auto& from =
                                 moments.at[SurfaceGrid::Index(l, m)];
                             auto out = ModeMoments{};
                             for (auto k = 0; k < 5; ++k) {
                               out.canonical[k] = from[k];
                             }
                             out.trace = from[5];
                             return out;
                           });
}

}  // namespace

DegreeFields PointSourceForces(
    const Material& material, int lmax,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    const PointSource& source) {
  return ForcesOfMoments(material, lmax, spheroidal, toroidal, source.at.radius,
                         MomentsOf(lmax, source));
}

DegreeFields StressGlutForces(
    const Material& material, int lmax,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    const StressGlut& source) {
  return ForcesOfMoments(material, lmax, spheroidal, toroidal, source.radius,
                         MomentsOf(lmax, source));
}

UnitAmplitudes AmplitudesOf(const SourceMoments& moments, int l, int m) {
  // The work on a mode is sum_N mu_N a_N + mu_tr theta / 3 and the force
  // its conjugate. The spheroidal unknowns have a_{-N} = a_N, real; the
  // toroidal one a_{-N} = -a_N, imaginary, so with g = Im a its force is
  // -i conj(mu_N - mu_{-N}) g.
  const auto& mu = moments.at[SurfaceGrid::Index(l, m)];
  auto out = UnitAmplitudes{};
  out.spheroidal = {std::conj(mu[2]), std::conj(mu[5]),
                    std::conj(mu[3] + mu[1]), std::conj(mu[4] + mu[0])};
  out.toroidal = {-kI * std::conj(mu[3] - mu[1]),
                  -kI * std::conj(mu[4] - mu[0])};
  return out;
}

UnitSourceForces UnitForces(
    const Material& material, int lmax,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    double radius) {
  const auto& mesh = material.Mesh();
  const auto n = mesh.NGll();
  const auto at = LocateRadius(material, radius);
  auto out = UnitSourceForces{};
  for (auto l = 0; l <= lmax; ++l) {
    const auto& sph = spheroidal(l);
    const auto sphDofs = static_cast<std::size_t>(sph.DofCount());
    auto bs = std::vector<double>(sphDofs * 4, 0.0);
    const ToroidalOperator* tor = l >= 1 ? &toroidal(l) : nullptr;
    const auto torDofs =
        tor != nullptr ? static_cast<std::size_t>(tor->DofCount()) : 0;
    auto bt = std::vector<double>(torDofs * 2, 0.0);
    const bool reachesS = at.element >= sph.FirstElement();
    const bool reachesT = tor != nullptr && at.element >= tor->FirstElement();
    for (auto j = 0; j < n; ++j) {
      const auto global = mesh.GlobalIndex(at.element, j);
      const auto value = at.value[j];
      const auto rd = at.radius * at.derivative[j];
      // the strain components of the unknown's unit mode at the point
      auto partsOf = [&](int which) {
        const auto mode = UnitMode(which, value, rd);
        const auto A = DeviatoricComponents(l, mode, at.radius);
        const auto theta =
            (mode.rdU + 2.0 * mode.U - l * (l + 1.0) * mode.V) / at.radius;
        return std::pair{A, theta};
      };
      if (reachesS) {
        for (const auto [which, component] :
             {std::pair{0, Component::U}, std::pair{1, Component::V}}) {
          const auto dof = sph.Dof(global, component);
          if (dof < 0) continue;
          const auto [A, theta] = partsOf(which);
          bs[0 * sphDofs + dof] += (2.0 * A[2]).real();
          bs[1 * sphDofs + dof] += (theta / 3.0).real();
          bs[2 * sphDofs + dof] += A[3].real();
          bs[3 * sphDofs + dof] += A[4].real();
        }
      }
      if (reachesT) {
        if (const auto dof = tor->Dof(global); dof >= 0) {
          const auto [A, theta] = partsOf(2);
          bt[0 * torDofs + dof] += A[3].imag();
          bt[1 * torDofs + dof] += A[4].imag();
        }
      }
    }
    out.spheroidal.emplace(l, std::move(bs));
    if (tor != nullptr) out.toroidal.emplace(l, std::move(bt));
  }
  return out;
}

}  // namespace sl3dpp
