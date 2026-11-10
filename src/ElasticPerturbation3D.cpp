#include "sl3dpp/ElasticPerturbation3D.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/LateralSolve.hpp"
#include "sl3dpp/detail/Parallel.hpp"

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
                 .cache_bytes = options.cache_bytes,
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
  auto workspace = GridCoupling::TensorField{};
  return Force(x, dMu, dKappa, workspace);
}

std::pair<ElasticPerturbation3D::Forces, ElasticPerturbation3D::Forces>
ElasticPerturbation3D::Force(const Solutions& x, std::span<const double> dMu,
                             std::span<const double> dKappa,
                             GridCoupling::TensorField& grid) const {
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
  // The degrees' gathers, threaded as the stepper's (each degree writes
  // only its own coefficients), the largest degrees first; the operators
  // resolved serially before.
  struct Gather {
    int l;
    const std::vector<Complex>* x;
    const DegreeOperator* sph;
    const ToroidalOperator* tor;
    const std::vector<Complex>* w;
  };
  auto gathers = std::vector<Gather>{};
  for (const auto& [l, xl] : x.spheroidal) {
    const auto torIt = x.toroidal.find(l);
    const auto hasTor = torIt != x.toroidal.end();
    gathers.push_back({l, &xl, &Spheroidal(l), hasTor ? &Toroidal(l) : nullptr,
                       hasTor ? &torIt->second : nullptr});
  }
  const auto count = static_cast<int>(gathers.size());
  if (!dMu.empty()) {
    // The deviatoric pipeline: strain, weighted pointwise, back as force.
    auto spectral = coupling_.ZeroSpectral();
    detail::ForEachTask(options_.parallel, count, [&](int i) {
      const auto& g = gathers[count - 1 - i];
      coupling_.AddStrainOfSolution(
          spectral, g.l, *g.sph, *g.x, g.tor,
          g.w == nullptr ? std::span<const Complex>{}
                         : std::span<const Complex>(*g.w));
    });
    // Into the caller's workspace: every value is overwritten, so a
    // workspace kept across calls (the conjugate gradients') is neither
    // reallocated nor zeroed.
    coupling_.Synthesise(spectral, grid);
    coupling_.Multiply(grid, dMu);
    const auto weighted = coupling_.Analyse(grid);
    add(spheroidal, coupling_.SpheroidalForce(weighted, sph));
    add(toroidal, coupling_.ToroidalForce(weighted, tor));
  }
  if (!dKappa.empty()) {
    // The scalar companion: the dilatation, weighted, back as force. The
    // field goes in the workspace's real component (the tensor pipeline
    // above is done with it), and its analysis over the dilatation's
    // coefficients: no grid-sized allocation per call.
    auto scalar = std::vector<Complex>(coupling_.ScalarCoefficientCount(), 0.0);
    detail::ForEachTask(options_.parallel, count, [&](int i) {
      const auto& g = gathers[count - 1 - i];
      coupling_.AddDilatationOfSolution(scalar, g.l, *g.sph, *g.x);
    });
    // The whole tensor is allocated, not its real component alone, so
    // that a later call with a shear direction finds a workspace it can
    // synthesise into.
    if (grid.r.empty()) grid = coupling_.ZeroTensor();
    auto& field = grid.r;
    coupling_.SynthesiseScalar(scalar, field);
    detail::ForEach(options_.parallel, values,
                    [&](std::size_t i) { field[i] *= dKappa[i]; });
    coupling_.AnalyseScalar(field, scalar);
    add(spheroidal, coupling_.DilatationForce(scalar, sph));
  }
  return {std::move(spheroidal), std::move(toroidal)};
}

namespace {

/// One degree's columns through a reference system: the operator (built
/// before the threaded loop, the caches not being thread-safe), the
/// columns in, and the slot they go to (made before the loop, so the
/// threads only write existing map entries).
struct DegreeTask {
  int l = 0;
  const std::vector<std::complex<double>>* in = nullptr;
  const DegreeOperator* sph = nullptr;
  const ToroidalOperator* tor = nullptr;
  std::vector<std::complex<double>>* out = nullptr;
};

}  // namespace

ElasticPerturbation3D::Solutions ElasticPerturbation3D::Solve(
    const Forces& spheroidal, const Forces& toroidal, bool negate) const {
  Solutions out;
  auto tasks = std::vector<DegreeTask>{};
  for (const auto& [l, b] : spheroidal) {
    tasks.push_back({.l = l, .in = &b, .sph = &Spheroidal(l),
                     .out = &out.spheroidal[l]});
  }
  for (const auto& [l, b] : toroidal) {
    if (l < 1) continue;
    tasks.push_back({.l = l, .in = &b, .tor = &Toroidal(l),
                     .out = &out.toroidal[l]});
  }
  detail::ForEachTask(options_.parallel, static_cast<int>(tasks.size()),
                      [&](int i) {
                        const auto& task = tasks[i];
                        auto rhs = *task.in;
                        if (negate) {
                          for (auto& v : rhs) v = -v;
                        }
                        *task.out = task.tor != nullptr
                                        ? SolveColumns(*task.tor, rhs, task.l + 1)
                                        : SolveColumns(*task.sph, rhs, task.l + 1);
                      });
  return out;
}

ElasticPerturbation3D::Forces ElasticPerturbation3D::LoadForces(
    std::span<const Complex> load) const {
  const auto& grid = coupling_.Grid();
  if (load.size() != grid.CoefficientSize()) {
    throw std::invalid_argument("a load lives on the lmax grid");
  }
  Forces out;
  for (auto l = 0; l <= lmax_; ++l) {
    const auto& op = Spheroidal(l);
    const auto ndof = op.DofCount();
    const auto unit = op.Load();
    auto b = std::vector<Complex>(static_cast<std::size_t>(ndof) * (l + 1));
    for (auto m = 0; m <= l; ++m) {
      const auto amplitude = load[SurfaceGrid::Index(l, m)];
      for (auto d = 0; d < ndof; ++d) {
        b[static_cast<std::size_t>(m) * ndof + d] = amplitude * unit[d];
      }
    }
    out.emplace(l, std::move(b));
  }
  return out;
}

namespace {

/// A real symmetric operator applied to complex columns, as real pairs.
template <class Op>
std::vector<std::complex<double>> ApplyColumns(
    const Op& op, const std::vector<std::complex<double>>& x, int columns) {
  const auto ndof = op.DofCount();
  auto real = std::vector<double>(static_cast<std::size_t>(ndof) * 2 * columns);
  for (auto c = 0; c < columns; ++c) {
    for (auto d = 0; d < ndof; ++d) {
      real[static_cast<std::size_t>(c) * ndof + d] =
          x[static_cast<std::size_t>(c) * ndof + d].real();
      real[static_cast<std::size_t>(columns + c) * ndof + d] =
          x[static_cast<std::size_t>(c) * ndof + d].imag();
    }
  }
  const auto y = op.Apply(real, 2 * columns);
  auto out = std::vector<std::complex<double>>(
      static_cast<std::size_t>(ndof) * columns);
  for (auto c = 0; c < columns; ++c) {
    for (auto d = 0; d < ndof; ++d) {
      out[static_cast<std::size_t>(c) * ndof + d] = {
          y[static_cast<std::size_t>(c) * ndof + d],
          y[static_cast<std::size_t>(columns + c) * ndof + d]};
    }
  }
  return out;
}

}  // namespace

std::pair<ElasticPerturbation3D::Forces, ElasticPerturbation3D::Forces>
ElasticPerturbation3D::Apply(const Solutions& x, std::span<const double> dMu,
                             std::span<const double> dKappa) const {
  auto workspace = GridCoupling::TensorField{};
  return Apply(x, dMu, dKappa, workspace);
}

std::pair<ElasticPerturbation3D::Forces, ElasticPerturbation3D::Forces>
ElasticPerturbation3D::Apply(const Solutions& x, std::span<const double> dMu,
                             std::span<const double> dKappa,
                             GridCoupling::TensorField& grid) const {
  auto [spheroidal, toroidal] = Force(x, dMu, dKappa, grid);
  // the reference bands, threaded over the degrees, each added to its
  // own degree's force
  auto tasks = std::vector<DegreeTask>{};
  for (const auto& [l, xl] : x.spheroidal) {
    tasks.push_back({.l = l, .in = &xl, .sph = &Spheroidal(l),
                     .out = &spheroidal.at(l)});
  }
  for (const auto& [l, wl] : x.toroidal) {
    if (l < 1) continue;
    tasks.push_back({.l = l, .in = &wl, .tor = &Toroidal(l),
                     .out = &toroidal.at(l)});
  }
  detail::ForEachTask(options_.parallel, static_cast<int>(tasks.size()),
                      [&](int i) {
                        const auto& task = tasks[i];
                        const auto reference =
                            task.tor != nullptr
                                ? ApplyColumns(*task.tor, *task.in, task.l + 1)
                                : ApplyColumns(*task.sph, *task.in, task.l + 1);
                        auto& b = *task.out;
                        for (auto k = std::size_t{0}; k < b.size(); ++k) {
                          b[k] += reference[k];
                        }
                      });
  return {std::move(spheroidal), std::move(toroidal)};
}

ElasticPerturbation3D::Forces ElasticPerturbation3D::TideForces(
    std::span<const Complex> amplitudes, int power) const {
  const auto& grid = coupling_.Grid();
  if (amplitudes.size() != grid.CoefficientSize()) {
    throw std::invalid_argument("a tidal potential lives on the lmax grid");
  }
  Forces out;
  for (auto l = 0; l <= lmax_; ++l) {
    const auto& op = Spheroidal(l);
    const auto ndof = op.DofCount();
    const auto unit = op.Tide(power);
    auto b = std::vector<Complex>(static_cast<std::size_t>(ndof) * (l + 1));
    for (auto m = 0; m <= l; ++m) {
      const auto amplitude = amplitudes[SurfaceGrid::Index(l, m)];
      for (auto d = 0; d < ndof; ++d) {
        b[static_cast<std::size_t>(m) * ndof + d] = amplitude * unit[d];
      }
    }
    out.emplace(l, std::move(b));
  }
  return out;
}

ElasticPerturbation3D::Forces ElasticPerturbation3D::SurfaceDualForces(
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
  return forces;
}

std::pair<ElasticPerturbation3D::Solutions, ElasticPerturbation3D::SolveInfo>
ElasticPerturbation3D::SolveLateral(std::span<const Complex> load,
                                    std::span<const double> dMu,
                                    std::span<const double> dKappa,
                                    SolveOptions options) const {
  return SolveLateral(LoadForces(load), {}, dMu, dKappa, options);
}

std::pair<ElasticPerturbation3D::Solutions, ElasticPerturbation3D::SolveInfo>
ElasticPerturbation3D::SolveLateral(const Forces& spheroidalForces,
                                    const Forces& toroidalForces,
                                    std::span<const double> dMu,
                                    std::span<const double> dKappa,
                                    SolveOptions options) const {
  // Every degree gets a slot, spheroidal and toroidal, so that the
  // lateral coupling has somewhere to go.
  Forces fs, ft;
  for (auto l = 0; l <= lmax_; ++l) {
    const auto found = spheroidalForces.find(l);
    if (found != spheroidalForces.end()) {
      fs.emplace(l, found->second);
    } else {
      fs.emplace(l, std::vector<Complex>(
                        static_cast<std::size_t>(Spheroidal(l).DofCount()) * (l + 1),
                        0.0));
    }
    if (l < 1) continue;
    const auto foundT = toroidalForces.find(l);
    if (foundT != toroidalForces.end()) {
      ft.emplace(l, foundT->second);
    } else {
      ft.emplace(l, std::vector<Complex>(
                        static_cast<std::size_t>(Toroidal(l).DofCount()) * (l + 1),
                        0.0));
    }
  }
  // one grid workspace for every iteration's lateral force
  auto workspace = GridCoupling::TensorField{};
  const auto system = detail::LateralSystem{
      .apply = [&](const Solutions& x) { return Apply(x, dMu, dKappa, workspace); },
      .solve = [&](const Forces& s, const Forces& t) { return Solve(s, t, false); }};
  auto [x, info] = detail::LateralConjugateGradients(
      fs, ft, system,
      {.rtol = options.rtol, .max_iterations = options.max_iterations},
      !(dMu.empty() && dKappa.empty()));
  SolveInfo out;
  out.iterations = info.iterations;
  out.residual = info.residual;
  out.converged = info.converged;
  out.residuals = std::move(info.residuals);
  out.indefinite = info.indefinite;
  return {std::move(x), std::move(out)};
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
  detail::ForEach(options_.parallel, nodes, [&](std::size_t n) {
    const auto rn = static_cast<int>(n);
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
  });
  return out;
}

ElasticPerturbation3D::Solutions ElasticPerturbation3D::SolveSurfaceDual(
    std::span<const Complex> dual, Component component) const {
  return Solve(SurfaceDualForces(dual, component), {}, false);
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
