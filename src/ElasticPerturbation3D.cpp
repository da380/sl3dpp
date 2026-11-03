#include "sl3dpp/ElasticPerturbation3D.hpp"

#include <format>
#include <stdexcept>

#include "sl3dpp/SurfaceGrid.hpp"

namespace sl3dpp {

namespace {

/// Complex columns solved as real pairs through a real symmetric system.
template <class Op>
std::vector<std::complex<double>> SolveColumns(
    const Op& op, const std::vector<std::complex<double>>& b, int columns) {
  const auto ndof = op.DofCount();
  auto rhs = std::vector<double>(static_cast<std::size_t>(ndof) * 2 * columns);
  for (auto c = 0; c < columns; ++c) {
    for (auto d = 0; d < ndof; ++d) {
      rhs[static_cast<std::size_t>(c) * ndof + d] =
          b[static_cast<std::size_t>(c) * ndof + d].real();
      rhs[static_cast<std::size_t>(columns + c) * ndof + d] =
          b[static_cast<std::size_t>(c) * ndof + d].imag();
    }
  }
  const auto x = op.Solve(rhs, 2 * columns);
  auto out = std::vector<std::complex<double>>(
      static_cast<std::size_t>(ndof) * columns);
  for (auto c = 0; c < columns; ++c) {
    for (auto d = 0; d < ndof; ++d) {
      out[static_cast<std::size_t>(c) * ndof + d] = {
          x[static_cast<std::size_t>(c) * ndof + d],
          x[static_cast<std::size_t>(columns + c) * ndof + d]};
    }
  }
  return out;
}

}  // namespace

ElasticPerturbation3D::ElasticPerturbation3D(const Material& material, int lmax,
                                             Options options)
    : material_(material),
      rheology_(Rheology::Maxwell(
                    material
                        .WithModuli([&] {
                          const auto [kappa, mu] = material.Moduli().KappaMu();
                          return NodalModuli::Isotropic(kappa, mu);
                        }())
                        .value(),
                    1.0)
                    .value()),
      lmax_(lmax),
      options_(options),
      coupling_(rheology_, lmax,
                {.eps = options.eps,
                 .truncation = options.truncation,
                 .oversampling = options.oversampling,
                 .parallel = options.parallel,
                 .memory_points = MemoryPointSet::Gll,
                 // The strain components of the spherical decomposition
                 // divide by r, so the element holding the centre carries
                 // no nodes: a perturbation starts above it.
                 .elements = {material.Mesh().Left(0) == 0.0 ? 1 : 0, -1}}) {
  if (lmax < 0) throw std::invalid_argument("lmax must be non-negative");
}

const DegreeOperator& ElasticPerturbation3D::Spheroidal(int l) const {
  auto found = spheroidal_.find(l);
  if (found == spheroidal_.end()) {
    found = spheroidal_
                .emplace(l, std::make_unique<DegreeOperator>(
                                GetMaterial(), l,
                                DegreeOperator::Options{
                                    .eps = options_.eps,
                                    .truncation = options_.truncation}))
                .first;
  }
  return *found->second;
}

const ToroidalOperator& ElasticPerturbation3D::Toroidal(int l) const {
  auto found = toroidal_.find(l);
  if (found == toroidal_.end()) {
    found = toroidal_
                .emplace(l, std::make_unique<ToroidalOperator>(
                                GetMaterial(), l,
                                ToroidalOperator::Options{
                                    .eps = options_.eps,
                                    .truncation = options_.truncation}))
                .first;
  }
  return *found->second;
}

ElasticPerturbation3D::Solutions ElasticPerturbation3D::SolveLoad(
    std::span<const Complex> load) const {
  const auto& grid = coupling_.Grid();
  if (load.size() != grid.CoefficientSize()) {
    throw std::invalid_argument(std::format(
        "a load on the lmax {} grid has {} coefficients, got {}", lmax_,
        grid.CoefficientSize(), load.size()));
  }
  Solutions out;
  for (auto l = 0; l <= lmax_; ++l) {
    const auto& op = Spheroidal(l);
    const auto ndof = op.DofCount();
    auto found = unitLoad_.find(l);
    if (found == unitLoad_.end()) {
      found = unitLoad_.emplace(l, op.Solve(op.Load(), 1)).first;
    }
    const auto& unit = found->second;
    auto x = std::vector<Complex>(static_cast<std::size_t>(ndof) * (l + 1));
    for (auto m = 0; m <= l; ++m) {
      const auto amplitude = load[SurfaceGrid::Index(l, m)];
      for (auto d = 0; d < ndof; ++d) {
        x[static_cast<std::size_t>(m) * ndof + d] = amplitude * unit[d];
      }
    }
    out.spheroidal.emplace(l, std::move(x));
  }
  return out;
}

std::pair<ElasticPerturbation3D::Forces, ElasticPerturbation3D::Forces>
ElasticPerturbation3D::Force(const Solutions& x, std::span<const double> dMu,
                             std::span<const double> dKappa) const {
  const auto values = coupling_.GridValueCount();
  for (const auto d : {dMu, dKappa}) {
    if (!d.empty() && d.size() != values) {
      throw std::invalid_argument(
          "a moduli direction lives on the coupling's grid values");
    }
  }
  const auto sph = [&](int l) -> const DegreeOperator& { return Spheroidal(l); };
  const auto tor = [&](int l) -> const ToroidalOperator& { return Toroidal(l); };
  Forces spheroidal, toroidal;
  for (auto l = 0; l <= lmax_; ++l) {
    spheroidal[l].assign(
        static_cast<std::size_t>(Spheroidal(l).DofCount()) * (l + 1), 0.0);
    if (l >= 1) {
      toroidal[l].assign(
          static_cast<std::size_t>(Toroidal(l).DofCount()) * (l + 1), 0.0);
    }
  }
  auto add = [](Forces& to, const Forces& from) {
    for (const auto& [l, f] : from) {
      auto& t = to.at(l);
      for (auto i = std::size_t{0}; i < t.size(); ++i) t[i] += f[i];
    }
  };
  if (!dMu.empty()) {
    // The deviatoric pipeline: strain, weighted pointwise, back as force.
    auto spectral = coupling_.ZeroSpectral();
    for (const auto& [l, xl] : x.spheroidal) {
      const auto torIt = x.toroidal.find(l);
      coupling_.AddStrainOfSolution(
          spectral, l, Spheroidal(l), xl,
          torIt != x.toroidal.end() ? &Toroidal(l) : nullptr,
          torIt != x.toroidal.end() ? std::span<const Complex>(torIt->second)
                                    : std::span<const Complex>{});
    }
    auto grid = coupling_.Synthesise(spectral);
    coupling_.Multiply(grid, dMu);
    const auto weighted = coupling_.Analyse(grid);
    add(spheroidal, coupling_.SpheroidalForce(weighted, sph));
    add(toroidal, coupling_.ToroidalForce(weighted, tor));
  }
  if (!dKappa.empty()) {
    // The scalar companion: the dilatation, weighted, back as force.
    auto scalar = std::vector<Complex>(coupling_.ScalarCoefficientCount(), 0.0);
    for (const auto& [l, xl] : x.spheroidal) {
      coupling_.AddDilatationOfSolution(scalar, l, Spheroidal(l), xl);
    }
    auto field = coupling_.SynthesiseScalar(scalar);
    for (auto i = std::size_t{0}; i < values; ++i) field[i] *= dKappa[i];
    const auto weighted = coupling_.AnalyseScalar(field);
    add(spheroidal, coupling_.DilatationForce(weighted, sph));
  }
  return {std::move(spheroidal), std::move(toroidal)};
}

ElasticPerturbation3D::Solutions ElasticPerturbation3D::Solve(
    const Forces& spheroidal, const Forces& toroidal, bool negate) const {
  Solutions out;
  for (const auto& [l, b] : spheroidal) {
    auto rhs = b;
    if (negate) {
      for (auto& v : rhs) v = -v;
    }
    out.spheroidal.emplace(l, SolveColumns(Spheroidal(l), rhs, l + 1));
  }
  for (const auto& [l, b] : toroidal) {
    if (l < 1) continue;
    auto rhs = b;
    if (negate) {
      for (auto& v : rhs) v = -v;
    }
    out.toroidal.emplace(l, SolveColumns(Toroidal(l), rhs, l + 1));
  }
  return out;
}

ElasticPerturbation3D::Solutions ElasticPerturbation3D::Respond(
    const Solutions& x, std::span<const double> dMu,
    std::span<const double> dKappa) const {
  const auto [spheroidal, toroidal] = Force(x, dMu, dKappa);
  return Solve(spheroidal, toroidal, true);
}

std::map<int, std::vector<ElasticPerturbation3D::Complex>>
ElasticPerturbation3D::Surface(const Solutions& x, Component component) const {
  auto out = std::map<int, std::vector<Complex>>{};
  for (const auto& [l, xl] : x.spheroidal) {
    const auto& op = Spheroidal(l);
    const auto dof = op.SurfaceDof(component);
    auto values = std::vector<Complex>(l + 1, 0.0);
    if (dof >= 0) {
      for (auto m = 0; m <= l; ++m) {
        values[m] = xl[static_cast<std::size_t>(m) * op.DofCount() + dof];
      }
    }
    out.emplace(l, std::move(values));
  }
  return out;
}

std::map<int, std::vector<ElasticPerturbation3D::Complex>>
ElasticPerturbation3D::SurfaceToroidal(const Solutions& x) const {
  auto out = std::map<int, std::vector<Complex>>{};
  for (const auto& [l, w] : x.toroidal) {
    const auto& op = Toroidal(l);
    const auto dof = op.SurfaceDof();
    auto values = std::vector<Complex>(l + 1, 0.0);
    if (dof >= 0) {
      for (auto m = 0; m <= l; ++m) {
        values[m] = w[static_cast<std::size_t>(m) * op.DofCount() + dof];
      }
    }
    out.emplace(l, std::move(values));
  }
  return out;
}

ElasticPerturbation3D::Kernels ElasticPerturbation3D::StaticKernels(
    const Solutions& y, const Solutions& x) const {
  const auto& coupling = coupling_;
  const auto values = coupling.GridValueCount();
  const auto nodes = coupling.NodeCount();
  const auto sph = [&](int l) -> const DegreeOperator& { return Spheroidal(l); };
  const auto tor = [&](int l) -> const ToroidalOperator& { return Toroidal(l); };
  auto strainOf = [&](const Solutions& s) {
    auto spectral = coupling.ZeroSpectral();
    for (const auto& [l, sl] : s.spheroidal) {
      const auto torIt = s.toroidal.find(l);
      coupling.AddStrainOfSolution(
          spectral, l, sph(l), sl, torIt != s.toroidal.end() ? &tor(l) : nullptr,
          torIt != s.toroidal.end() ? std::span<const Complex>(torIt->second)
                                    : std::span<const Complex>{});
    }
    return coupling.Synthesise(spectral);
  };
  auto dilatationOf = [&](const Solutions& s) {
    auto scalar = std::vector<Complex>(coupling.ScalarCoefficientCount(), 0.0);
    for (const auto& [l, sl] : s.spheroidal) {
      coupling.AddDilatationOfSolution(scalar, l, sph(l), sl);
    }
    return coupling.SynthesiseScalar(scalar);
  };
  const auto ey = strainOf(y);
  const auto ex = strainOf(x);
  const auto Dy = dilatationOf(y);
  const auto Dx = dilatationOf(x);
  Kernels out;
  out.kappa.assign(values, 0.0);
  out.mu.assign(values, 0.0);
  for (auto rn = 0; rn < nodes; ++rn) {
    const auto base = coupling.FieldOffset(rn);
    const auto size = coupling.FieldSizeOf(rn);
    const auto rr = coupling.RadiusOfNode(rn);
    const auto wq = coupling.WeightOfNode(rn);
    for (auto xg = std::size_t{0}; xg < size; ++xg) {
      const auto i = base + xg;
      out.kappa[i] = wq * Dy[i] * Dx[i];
      out.mu[i] = wq * rr * rr *
                  (12.0 * ey.r[i] * ex.r[i] +
                   8.0 * (ey.c[1][i] * std::conj(ex.c[1][i])).real() +
                   4.0 * (ey.c[0][i] * std::conj(ex.c[0][i])).real());
    }
  }
  return out;
}

ElasticPerturbation3D::Solutions ElasticPerturbation3D::SolveSurfaceDual(
    std::span<const Complex> dual, Component component) const {
  const auto& grid = coupling_.Grid();
  if (dual.size() != grid.CoefficientSize()) {
    throw std::invalid_argument("a surface dual lives on the lmax grid");
  }
  Forces forces;
  for (auto l = 0; l <= lmax_; ++l) {
    const auto& op = Spheroidal(l);
    const auto ndof = op.DofCount();
    const auto dof = op.SurfaceDof(component);
    auto b = std::vector<Complex>(static_cast<std::size_t>(ndof) * (l + 1), 0.0);
    if (dof >= 0) {
      for (auto m = 0; m <= l; ++m) {
        b[static_cast<std::size_t>(m) * ndof + dof] = dual[SurfaceGrid::Index(l, m)];
      }
    }
    forces.emplace(l, std::move(b));
  }
  return Solve(forces, {}, false);
}

double ElasticPerturbation3D::Pair(const Solutions& x, const Forces& spheroidal,
                                   const Forces& toroidal) {
  auto total = 0.0;
  auto pair = [&](const std::map<int, std::vector<Complex>>& a,
                  const Forces& b) {
    for (const auto& [l, al] : a) {
      const auto found = b.find(l);
      if (found == b.end()) continue;
      const auto& bl = found->second;
      const auto ndof = al.size() / static_cast<std::size_t>(l + 1);
      for (auto m = 0; m <= l; ++m) {
        const auto weight = m == 0 ? 1.0 : 2.0;
        for (auto d = std::size_t{0}; d < ndof; ++d) {
          const auto i = static_cast<std::size_t>(m) * ndof + d;
          total += weight * (al[i] * std::conj(bl[i])).real();
        }
      }
    }
  };
  pair(x.spheroidal, spheroidal);
  pair(x.toroidal, toroidal);
  return total;
}

}  // namespace sl3dpp
