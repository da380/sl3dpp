#include "sl3dpp/Viscoelastic3D.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sl3dpp/detail/AndersonMixer.hpp"
#include "sl3dpp/detail/LateralSolve.hpp"
#include "sl3dpp/detail/ParallelFor.hpp"

namespace sl3dpp {

namespace {

// A spectral tensor's coefficients as real segments for the mixer: the
// spin -2, spin -1 and spin 0 blocks in turn, then the scalar companion
// where carried, each complex value as its real-imaginary pair (the
// array layout std::complex guarantees).
std::vector<std::span<double>> RealSegments(GridCoupling::SpectralTensor& t) {
  auto pairs = [](std::vector<std::complex<double>>& v) {
    return std::span<double>(reinterpret_cast<double*>(v.data()), 2 * v.size());
  };
  auto out = std::vector{pairs(t.c[0]), pairs(t.c[1]), pairs(t.r)};
  if (!t.scalar.empty()) out.push_back(pairs(t.scalar));
  return out;
}

std::vector<std::span<const double>> RealSegments(
    const GridCoupling::SpectralTensor& t) {
  auto pairs = [](const std::vector<std::complex<double>>& v) {
    return std::span<const double>(reinterpret_cast<const double*>(v.data()),
                                   2 * v.size());
  };
  auto out = std::vector{pairs(t.c[0]), pairs(t.c[1]), pairs(t.r)};
  if (!t.scalar.empty()) out.push_back(pairs(t.scalar));
  return out;
}

// The coefficient blocks of a spectral tensor (three, and the scalar
// companion where carried), for loops that treat them alike.
std::vector<std::vector<std::complex<double>>*> Blocks(
    GridCoupling::SpectralTensor& t) {
  auto out = std::vector{&t.c[0], &t.c[1], &t.r};
  if (!t.scalar.empty()) out.push_back(&t.scalar);
  return out;
}

std::vector<const std::vector<std::complex<double>>*> Blocks(
    const GridCoupling::SpectralTensor& t) {
  auto out = std::vector{&t.c[0], &t.c[1], &t.r};
  if (!t.scalar.empty()) out.push_back(&t.scalar);
  return out;
}

// out += in over the three tensor blocks, and the scalar companion where
// both carry it.
void AddSpectral(GridCoupling::SpectralTensor& out,
                 const GridCoupling::SpectralTensor& in, bool parallel) {
  const auto a = std::array{&out.c[0], &out.c[1], &out.r};
  const auto b = std::array{&in.c[0], &in.c[1], &in.r};
  for (auto k = std::size_t{0}; k < a.size(); ++k) {
    auto& o = *a[k];
    const auto& f = *b[k];
    detail::ParallelElements(o.size(), parallel,
                             [&](std::size_t i) { o[i] += f[i]; });
  }
  if (!out.scalar.empty() && !in.scalar.empty()) {
    detail::ParallelElements(out.scalar.size(), parallel, [&](std::size_t i) {
      out.scalar[i] += in.scalar[i];
    });
  }
}

// Whether two spectral tensors hold the same values.
bool SameValues(const GridCoupling::SpectralTensor& a,
                const GridCoupling::SpectralTensor& b, bool parallel) {
  const auto x = Blocks(a);
  const auto y = Blocks(b);
  auto same = true;
  for (auto k = std::size_t{0}; k < x.size() && same; ++k) {
    const auto& u = *x[k];
    const auto& v = *y[k];
    if (u.size() != v.size()) return false;
    const auto size = u.size();
#pragma omp parallel for schedule(static) reduction(&& : same) if (parallel)
    for (auto i = std::size_t{0}; i < size; ++i) same = same && u[i] == v[i];
  }
  return same;
}

// The elements of this process's segments of the radial cuts: the whole
// mesh (end -1) undistributed.
std::pair<int, int> OwnedElements(const RadialMesh& mesh,
                                  const Viscoelastic3D::Options& options) {
  if (!options.collectives.Distributed()) return {0, -1};
  const auto& cuts = options.radial_cuts;
  const auto& segments = options.segments;
  const auto count = static_cast<int>(cuts.size()) + 1;
  auto contiguous = !segments.empty();
  for (auto i = std::size_t{0}; i < segments.size() && contiguous; ++i) {
    contiguous = segments[i] >= 0 && segments[i] < count &&
                 (i == 0 || segments[i] == segments[i - 1] + 1);
  }
  if (!contiguous) {
    throw std::invalid_argument(std::format(
        "a distributed operator owns contiguous segments among the {} of its "
        "radial cuts",
        count));
  }
  const auto first = segments.front() == 0 ? 0 : cuts[segments.front() - 1];
  const auto end = segments.back() + 1 == count ? mesh.ElementCount()
                                                : cuts[segments.back()];
  return {first, end};
}

// A complex vector's values as real ones, for the collectives.
std::span<double> Reals(std::vector<std::complex<double>>& v) {
  return {reinterpret_cast<double*>(v.data()), 2 * v.size()};
}

// m += beta d over every stored component, point by point.
void AdvanceMemory(GridCoupling::TensorField& m,
                   const std::vector<double>& beta,
                   const GridCoupling::TensorField& d, bool parallel) {
  detail::ParallelElements(m.r.size(), parallel, [&](std::size_t i) {
    m.c[0][i] += beta[i] * d.c[0][i];
    m.c[1][i] += beta[i] * d.c[1][i];
    m.r[i] += beta[i] * d.r[i];
  });
}

}  // namespace

struct Viscoelastic3D::Flavour {
  std::unique_ptr<Material> material;  // the reference effective solid
  // At Gauss points the material is the long-term one and the operators
  // add the branches' unrelaxed shear at the points (MemoryPoints.hpp).
  std::vector<double> pointShearModulus;
  PointShear pointShear;
  std::map<int, std::unique_ptr<DegreeOperator>> spheroidal;
  std::map<int, std::unique_ptr<ToroidalOperator>> toroidal;
  std::map<int, std::vector<double>> units;  // (ndof x 2): load, tide
};

Viscoelastic3D::Viscoelastic3D(
    const Rheology& rheology, int lmax,
    const std::function<double(double, double, double)>& delta, Options options)
    : Viscoelastic3D(
          rheology, lmax,
          SurfaceGrid(lmax, {.nmax = 2,
                             .oversampling = options.oversampling,
                             .parallel = options.parallel,
                             .matrix_kernel = options.matrix_kernel}),
          delta, options) {}

Viscoelastic3D::Viscoelastic3D(
    const Rheology& rheology, int lmax, const SurfaceGrid& grid,
    const std::function<double(double, double, double)>& delta, Options options)
    : rheology_(std::make_shared<const Rheology>(rheology)),
      lmax_(lmax),
      options_(options),
      coupling_(
          *rheology_, lmax, grid,
          {.eps = options.eps,
           .truncation = options.truncation,
           .oversampling = options.oversampling,
           .parallel = options.parallel,
           .matrix_kernel = options.matrix_kernel,
           .memory_points = options.memory_points,
           .transform_degrees = options.transform_degrees,
           .elements = OwnedElements(rheology.BaseMaterial().Mesh(), options),
           .all_solid = static_cast<bool>(options.moduli.shear) ||
                        static_cast<bool>(options.moduli.bulk)}),
      reference_(*rheology_, lmax,
                 {.eps = options.eps,
                  .truncation = options.truncation,
                  .memory_points = options.memory_points,
                  .radial_cuts = options.radial_cuts}) {
  // Sample exp(-delta) once, on each node's grid, and scale every
  // branch's inverse times; the reference profile is the unperturbed
  // radial rheology itself, at the memory points as the 1D reference
  // holds it.
  const auto np = coupling_.Points().count;
  const auto nodes = coupling_.NodeCount();
  auto factor = std::vector<double>(coupling_.GridValueCount());
  for (auto node = 0; node < nodes; ++node) {
    const auto& sampling = coupling_.GridOf(node);
    const auto r = coupling_.RadiusOfNode(node);
    auto x = coupling_.FieldOffset(node);
    for (auto theta : sampling.CoLatitudes()) {
      for (auto phi : sampling.Longitudes()) {
        factor[x++] = std::exp(-delta(r, theta, phi));
      }
    }
  }
  for (auto k = std::size_t{0}; k < rheology_->BranchCount(); ++k) {
    const auto pointMu = reference_.PointMu(k);
    const auto pointItau = reference_.PointItau(k);
    auto itau = std::vector<double>(coupling_.GridValueCount());
    auto mu = std::vector<double>(nodes);
    auto itauRef = std::vector<double>(nodes);
    for (auto node = 0; node < nodes; ++node) {
      const auto e = coupling_.ElementOfNode(node);
      const auto p = coupling_.NodeInElementOf(node);
      const auto at = static_cast<std::size_t>(e) * np + p;
      mu[node] = pointMu[at];
      itauRef[node] = pointItau[at];
      const auto first = coupling_.FieldOffset(node);
      for (auto x = first; x < first + coupling_.FieldSizeOf(node); ++x) {
        itau[x] = pointItau[at] * factor[x];
      }
    }
    itau_.push_back(std::move(itau));
    muBranch_.push_back(std::move(mu));
    itauRef_.push_back(std::move(itauRef));
  }

  // The lateral moduli: the reference's unrelaxed shear and bulk modulus
  // at each memory node (nodal at the GLL points; at Gauss points the
  // long-term shear and the bulk modulus interpolated, plus the branches'
  // shear at the points as the operators carry it), and the perturbation
  // functions sampled on each node's grid.
  const auto& moduli = options.moduli;
  if (moduli.shear || moduli.bulk) {
    if (options.collectives.Distributed()) {
      throw std::invalid_argument(
          "lateral moduli are not yet distributed over radial segments");
    }
    const auto& material = rheology_->BaseMaterial();
    const auto& mesh = material.Mesh();
    const auto n = mesh.NGll();
    const auto& points = coupling_.Points();
    const auto gll = points.set == MemoryPointSet::Gll;
    const auto longTerm = rheology_->MuLongTerm();
    const auto kappaNodal = rheology_->Kappa();
    muUnrelaxed_.assign(nodes, 0.0);
    auto kappaAtNodes = std::vector<double>(nodes, 0.0);
    for (auto node = 0; node < nodes; ++node) {
      const auto e = coupling_.ElementOfNode(node);
      const auto q = coupling_.NodeInElementOf(node);
      const auto at = static_cast<std::size_t>(e) * np + q;
      auto muInf = 0.0, kappa = 0.0;
      if (gll) {
        muInf = longTerm[static_cast<std::size_t>(e) * n + q];
        kappa = kappaNodal[static_cast<std::size_t>(e) * n + q];
      } else {
        for (auto i = 0; i < n; ++i) {
          const auto v = points.Value(q, i);
          muInf += v * longTerm[static_cast<std::size_t>(e) * n + i];
          kappa += v * kappaNodal[static_cast<std::size_t>(e) * n + i];
        }
      }
      auto mu = muInf;
      for (auto k = std::size_t{0}; k < rheology_->BranchCount(); ++k) {
        mu += reference_.PointMu(k)[at];
      }
      muUnrelaxed_[node] = mu;
      kappaAtNodes[node] = kappa;
    }
    auto sample = [&](const std::function<double(double, double, double)>& f,
                      std::span<const double> scale, std::vector<double>& out,
                      std::vector<double>* relative) {
      out.assign(coupling_.GridValueCount(), 0.0);
      if (relative != nullptr) relative->assign(coupling_.GridValueCount(), 0.0);
      for (auto node = 0; node < nodes; ++node) {
        const auto& sampling = coupling_.GridOf(node);
        const auto r = coupling_.RadiusOfNode(node);
        auto x = coupling_.FieldOffset(node);
        for (auto theta : sampling.CoLatitudes()) {
          for (auto phi : sampling.Longitudes()) {
            const auto value = f(r, theta, phi);
            if (relative != nullptr) (*relative)[x] = value;
            out[x++] = value * scale[node];
          }
        }
      }
    };
    if (moduli.shear) sample(moduli.shear, muUnrelaxed_, shearUnrelaxed_, &shear_);
    if (moduli.bulk) {
      sample(moduli.bulk, kappaAtNodes, bulk_, nullptr);
      negativeBulk_.resize(bulk_.size());
      for (auto i = std::size_t{0}; i < bulk_.size(); ++i) negativeBulk_[i] = -bulk_[i];
    }
  }
}

Viscoelastic3D::~Viscoelastic3D() = default;
Viscoelastic3D::Viscoelastic3D(Viscoelastic3D&&) noexcept = default;
Viscoelastic3D& Viscoelastic3D::operator=(Viscoelastic3D&&) noexcept = default;

Viscoelastic3D::State Viscoelastic3D::ZeroState() const {
  State state;
  for (auto k = std::size_t{0}; k < rheology_->BranchCount(); ++k) {
    state.m.push_back(coupling_.ZeroTensor());
  }
  return state;
}

const DegreeOperator& Viscoelastic3D::Spheroidal(int l) const {
  const auto lock = LockCaches();
  auto found = spheroidal_.find(l);
  if (found == spheroidal_.end()) {
    found =
        spheroidal_
            .emplace(l, std::make_unique<DegreeOperator>(
                            reference_.NodalMaterial(), l,
                            DegreeOperator::Options{
                                .eps = options_.eps,
                                .truncation = options_.truncation,
                                .point_shear = reference_.ElasticPointShear(),
                                .cuts = options_.radial_cuts,
                                .segments = options_.segments,
                                .combine = options_.collectives.sum}))
            .first;
  }
  return *found->second;
}

const ToroidalOperator& Viscoelastic3D::Toroidal(int l) const {
  const auto lock = LockCaches();
  auto found = toroidal_.find(l);
  if (found == toroidal_.end()) {
    found =
        toroidal_
            .emplace(l, std::make_unique<ToroidalOperator>(
                            reference_.NodalMaterial(), l,
                            ToroidalOperator::Options{
                                .eps = options_.eps,
                                .truncation = options_.truncation,
                                .pin_fraction = options_.toroidal_pin_fraction,
                                .point_shear = reference_.ElasticPointShear(),
                                .cuts = options_.radial_cuts,
                                .segments = options_.segments,
                                .combine = options_.collectives.sum}))
            .first;
  }
  return *found->second;
}

std::shared_ptr<Viscoelastic3D::Flavour> Viscoelastic3D::FlavourOf(
    Scheme scheme, double dt) const {
  const auto lock = LockCaches();
  const auto key = std::pair(static_cast<int>(scheme), dt);
  for (auto it = flavours_.begin(); it != flavours_.end(); ++it) {
    if (it->first == key) {
      flavours_.splice(flavours_.begin(), flavours_, it);
      return flavours_.front().second;
    }
  }
  constexpr auto kFlavourCache = std::size_t{4};
  while (flavours_.size() >= kFlavourCache) flavours_.pop_back();
  auto flavour = std::make_shared<Flavour>();
  if (options_.memory_points == MemoryPointSet::Gll) {
    flavour->material =
        std::make_unique<Material>(reference_.EffectiveMaterial(scheme, dt));
  } else {
    flavour->material = std::make_unique<Material>(reference_.NodalMaterial());
    flavour->pointShearModulus = reference_.PointShearModulus(scheme, dt);
    flavour->pointShear = {&reference_.Points(), flavour->pointShearModulus};
  }
  flavours_.emplace_front(key, flavour);
  return flavour;
}

GridCoupling::SpectralTensor Viscoelastic3D::ZeroSpectralFor() const {
  auto spectral = coupling_.ZeroSpectral();
  if (!bulk_.empty()) {
    spectral.scalar.assign(coupling_.ScalarCoefficientCount(), 0.0);
  }
  return spectral;
}

GridCoupling::SpectralTensor Viscoelastic3D::StrainCoefficients(
    const Solutions& solutions,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal) const {
  auto spectral = ZeroSpectralFor();
  AddStrainCoefficients(spectral, solutions, spheroidal, toroidal);
  return spectral;
}

std::map<int, std::vector<Viscoelastic3D::Complex>>
Viscoelastic3D::SpheroidalForces(
    const GridCoupling::SpectralTensor& weighted,
    const std::function<const DegreeOperator&(int)>& spheroidal) const {
  auto forces = coupling_.SpheroidalForce(weighted, spheroidal);
  if (weighted.scalar.empty()) return forces;
  const auto bulk = coupling_.DilatationForce(weighted.scalar, spheroidal);
  for (auto& [l, b] : forces) {
    const auto& extra = bulk.at(l);
    for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += extra[i];
  }
  return forces;
}

std::pair<std::map<int, std::vector<Viscoelastic3D::Complex>>,
          std::map<int, std::vector<Viscoelastic3D::Complex>>>
Viscoelastic3D::LateralElasticForce(const Solutions& x) const {
  const auto spheroidal = [&](int l) -> const DegreeOperator& {
    return Spheroidal(l);
  };
  const auto toroidal = [&](int l) -> const ToroidalOperator& {
    return Toroidal(l);
  };
  const auto strain = StrainCoefficients(x, spheroidal, toroidal);
  auto grid = coupling_.Synthesise(strain);
  if (!shearUnrelaxed_.empty()) {
    coupling_.Multiply(grid, shearUnrelaxed_);
  } else {
    coupling_.Multiply(grid, std::vector<double>(coupling_.GridValueCount(), 0.0));
  }
  if (!grid.scalar.empty()) coupling_.MultiplyScalar(grid, bulk_);
  const auto weighted = coupling_.Analyse(grid);
  return {SpheroidalForces(weighted, spheroidal),
          coupling_.ToroidalForce(weighted, toroidal)};
}

Viscoelastic3D::Solutions Viscoelastic3D::LateralElasticSolve(
    const std::map<int, std::vector<Complex>>& sphForces,
    const std::map<int, std::vector<Complex>>& torForces) const {
  const auto spheroidal = [&](int l) -> const DegreeOperator& {
    return Spheroidal(l);
  };
  const auto toroidal = [&](int l) -> const ToroidalOperator& {
    return Toroidal(l);
  };
  // The full unrelaxed operator: the reference systems' action on the
  // complex columns as real pairs, plus the lateral force.
  auto applyColumns = [](const auto& op, const std::vector<Complex>& x,
                         int columns) {
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
    auto out = std::vector<Complex>(static_cast<std::size_t>(ndof) * columns);
    for (auto c = 0; c < columns; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        out[static_cast<std::size_t>(c) * ndof + d] = {
            y[static_cast<std::size_t>(c) * ndof + d],
            y[static_cast<std::size_t>(columns + c) * ndof + d]};
      }
    }
    return out;
  };
  const auto system = detail::LateralSystem{
      .apply =
          [&](const Solutions& x) {
            auto [sph, tor] = LateralElasticForce(x);
            for (const auto& [l, xl] : x.spheroidal) {
              const auto reference = applyColumns(Spheroidal(l), xl, l + 1);
              auto& b = sph.at(l);
              for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += reference[i];
            }
            for (const auto& [l, wl] : x.toroidal) {
              if (l < 1) continue;
              const auto reference = applyColumns(Toroidal(l), wl, l + 1);
              auto& b = tor.at(l);
              for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += reference[i];
            }
            return std::pair{std::move(sph), std::move(tor)};
          },
      .solve =
          [&](const std::map<int, std::vector<Complex>>& s,
              const std::map<int, std::vector<Complex>>& t) {
            return SolveForces(s, t, spheroidal, toroidal);
          }};
  auto [x, info] = detail::LateralConjugateGradients(
      sphForces, torForces, system,
      {.rtol = options_.elastic_rtol,
       .max_iterations = options_.elastic_max_iterations});
  lastElasticSolve_ = {info.iterations, info.residual, info.converged};
  if (!info.converged) {
    throw std::runtime_error(std::format(
        "the unrelaxed solve under lateral moduli did not converge: residual "
        "{} after {} iterations",
        info.residual, info.iterations));
  }
  return std::move(x);
}

void Viscoelastic3D::AddChannelForces(
    std::map<int, std::vector<Complex>>& forces, double t,
    const LoadCallable3D& load, const LoadCallable3D& tide,
    const std::function<const DegreeOperator&(int)>& spheroidal) const {
  if (!load && !tide) return;
  for (auto& [l, b] : forces) {
    const auto& system = spheroidal(l);
    const auto ndof = system.DofCount();
    auto add = [&](const std::optional<std::vector<Complex>>& columns,
                   const std::vector<double>& unit) {
      if (!columns) return;
      for (auto c = 0; c <= l; ++c) {
        const auto amplitude = (*columns)[c];
        if (amplitude == 0.0) continue;
        for (auto dof = 0; dof < ndof; ++dof) {
          b[static_cast<std::size_t>(c) * ndof + dof] += amplitude * unit[dof];
        }
      }
    };
    if (load) add(load(l, t), system.Load());
    if (tide) add(tide(l, t), system.Tide(l == 0 ? 2 : -1));
  }
}

void Viscoelastic3D::AddStrainCoefficients(
    GridCoupling::SpectralTensor& spectral, const Solutions& solutions,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal) const {
  // The operators are resolved first (any built lazily, serially); one
  // task per degree then writes only that degree's coefficient slots.
  struct Task {
    int l;
    const std::vector<Complex>* x;
    const DegreeOperator* sph;
    const ToroidalOperator* tor;
    const std::vector<Complex>* w;
  };
  auto tasks = std::vector<Task>{};
  for (const auto& [l, x] : solutions.spheroidal) {
    const auto torIt = solutions.toroidal.find(l);
    const auto hasTor = torIt != solutions.toroidal.end();
    tasks.push_back({l, &x, &spheroidal(l), hasTor ? &toroidal(l) : nullptr,
                     hasTor ? &torIt->second : nullptr});
  }
  const auto count = static_cast<int>(tasks.size());
  detail::ParallelFor(count, options_.parallel, [&](int i) {
    const auto& task = tasks[count - 1 - i];  // the largest degrees first
    coupling_.AddStrainOfSolution(
        spectral, task.l, *task.sph, *task.x, task.tor,
        task.w == nullptr ? std::span<const Complex>{}
                          : std::span<const Complex>(*task.w));
    if (!spectral.scalar.empty()) {
      coupling_.AddDilatationOfSolution(spectral.scalar, task.l, *task.sph,
                                        *task.x);
    }
  });
}

Viscoelastic3D::Solutions Viscoelastic3D::SolveAll(
    const GridCoupling::SpectralTensor& weighted,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal) const {
  return SolveForces(coupling_.SpheroidalForce(weighted, spheroidal),
                     coupling_.ToroidalForce(weighted, toroidal), spheroidal,
                     toroidal);
}

Viscoelastic3D::Solutions Viscoelastic3D::SolveForces(
    const std::map<int, std::vector<Complex>>& sphForces,
    const std::map<int, std::vector<Complex>>& torForces,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal) const {

  // The degree tasks, spheroidal and toroidal, with their operators
  // resolved here, serially; each task then writes only its own solution.
  struct Task {
    int l = 0;
    const std::vector<Complex>* b = nullptr;
    const DegreeOperator* sph = nullptr;    // a spheroidal task
    const ToroidalOperator* tor = nullptr;  // a toroidal task
    std::vector<Complex> solution;
  };
  auto tasks = std::vector<Task>{};
  for (const auto& [l, b] : sphForces) {
    Task task;
    task.l = l;
    task.b = &b;
    task.sph = &spheroidal(l);
    tasks.push_back(std::move(task));
  }
  for (const auto& [l, b] : torForces) {
    Task task;
    task.l = l;
    task.b = &b;
    task.tor = &toroidal(l);
    tasks.push_back(std::move(task));
  }
  std::ranges::stable_sort(tasks, std::greater{}, &Task::l);  // largest first

  // The complex columns solved as real pairs, the force held in shares
  // (each process's own elements' rows): begun task by task, threaded and
  // local to the process; distributed, the cut nodes' parts of every
  // system summed in one collective; then finished, threaded again.
  auto begin = [](const auto& op, const std::vector<Complex>& b, int columns) {
    const auto ndof = op.DofCount();
    auto rhs =
        std::vector<double>(static_cast<std::size_t>(ndof) * 2 * columns);
    for (auto c = 0; c < columns; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        rhs[static_cast<std::size_t>(c) * ndof + d] =
            b[static_cast<std::size_t>(c) * ndof + d].real();
        rhs[static_cast<std::size_t>(columns + c) * ndof + d] =
            b[static_cast<std::size_t>(c) * ndof + d].imag();
      }
    }
    return op.BeginSolve(rhs, 2 * columns, RightHandSide::Shares);
  };
  auto finish = [](const auto& op, PendingSolve&& pending, int columns) {
    const auto ndof = op.DofCount();
    const auto x = op.FinishSolve(std::move(pending));
    auto solution =
        std::vector<Complex>(static_cast<std::size_t>(ndof) * columns);
    for (auto c = 0; c < columns; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        solution[static_cast<std::size_t>(c) * ndof + d] = {
            x[static_cast<std::size_t>(c) * ndof + d],
            x[static_cast<std::size_t>(columns + c) * ndof + d]};
      }
    }
    return solution;
  };

  auto pending = std::vector<PendingSolve>(tasks.size());
  detail::ParallelFor(
      static_cast<int>(tasks.size()), options_.parallel, [&](int i) {
        const auto& task = tasks[i];
        const auto columns = task.l + 1;
        pending[i] = task.tor != nullptr ? begin(*task.tor, *task.b, columns)
                                         : begin(*task.sph, *task.b, columns);
      });
  if (options_.collectives.Distributed()) {
    auto size = std::size_t{0};
    for (const auto& p : pending) size += p.separators.size();
    auto all = std::vector<double>(size);
    auto at = all.begin();
    for (const auto& p : pending) at = std::ranges::copy(p.separators, at).out;
    options_.collectives.sum(all);
    at = all.begin();
    for (auto& p : pending) {
      std::copy(at, at + static_cast<std::ptrdiff_t>(p.separators.size()),
                p.separators.begin());
      at += static_cast<std::ptrdiff_t>(p.separators.size());
    }
  }
  detail::ParallelFor(
      static_cast<int>(tasks.size()), options_.parallel, [&](int i) {
        auto& task = tasks[i];
        const auto columns = task.l + 1;
        task.solution = task.tor != nullptr
                            ? finish(*task.tor, std::move(pending[i]), columns)
                            : finish(*task.sph, std::move(pending[i]), columns);
      });

  Solutions out;
  for (auto& task : tasks) {
    (task.tor != nullptr ? out.toroidal : out.spheroidal)
        .emplace(task.l, std::move(task.solution));
  }
  return out;
}

Viscoelastic3D::Solutions Viscoelastic3D::SolveElastic(
    const State& state, const LoadCallable3D& load,
    const LoadCallable3D& tide) const {
  // The combined weighted memory tensor sum_k mu_k(r) m_k(x).
  auto combined = coupling_.ZeroTensor();
  const auto nodes = static_cast<std::size_t>(coupling_.NodeCount());
  for (auto k = std::size_t{0}; k < state.m.size(); ++k) {
    const auto& m = state.m[k];
    detail::ParallelElements(nodes, options_.parallel, [&](std::size_t node) {
      const auto mu = muBranch_[k][node];
      if (mu == 0.0) return;
      const auto first = coupling_.FieldOffset(static_cast<int>(node));
      const auto last = first + coupling_.FieldSizeOf(static_cast<int>(node));
      for (auto i = first; i < last; ++i) {
        combined.c[0][i] += mu * m.c[0][i];
        combined.c[1][i] += mu * m.c[1][i];
        combined.r[i] += mu * m.r[i];
      }
    });
  }
  // Under lateral moduli the branch moduli carry the shear factor.
  if (!shear_.empty() && !state.m.empty()) {
    detail::ParallelElements(combined.r.size(), options_.parallel, [&](std::size_t i) {
      const auto factor = 1.0 + shear_[i];
      combined.c[0][i] *= factor;
      combined.c[1][i] *= factor;
      combined.r[i] *= factor;
    });
  }
  const auto weighted = coupling_.Analyse(combined);
  const auto spheroidal = [&](int l) -> const DegreeOperator& {
    return Spheroidal(l);
  };
  const auto toroidal = [&](int l) -> const ToroidalOperator& {
    return Toroidal(l);
  };
  if (!HasLateralModuli()) {
    auto solutions = SolveAll(weighted, spheroidal, toroidal);
    AddUnitChannels(solutions, state.t, load, tide, spheroidal, unrelaxedUnits_);
    return solutions;
  }
  // The laterally heterogeneous unrelaxed problem: the memory and channel
  // forces through the preconditioned conjugate-gradient solve.
  auto sph = coupling_.SpheroidalForce(weighted, spheroidal);
  auto tor = coupling_.ToroidalForce(weighted, toroidal);
  AddChannelForces(sph, state.t, load, tide, spheroidal);
  return LateralElasticSolve(sph, tor);
}

void Viscoelastic3D::AddUnitChannels(
    Solutions& solutions, double t, const LoadCallable3D& load,
    const LoadCallable3D& tide,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    std::map<int, std::vector<double>>& units) const {
  if (!load && !tide) return;
  for (auto& [degree, columns_] : solutions.spheroidal) {
    // Plain names: clang does not allow capturing a structured binding
    // in a lambda under OpenMP.
    const auto l = degree;
    auto& x = columns_;
    const auto& system = spheroidal(l);
    const auto ndof = system.DofCount();
    const auto& responses = [&]() -> const std::vector<double>& {
      const auto lock = LockCaches();
      auto found = units.find(l);
      if (found == units.end()) {
        auto unit =
            std::vector<double>(static_cast<std::size_t>(ndof) * 2, 0.0);
        auto loadVec = system.Load();
        auto tideVec = system.Tide(l == 0 ? 2 : -1);
        std::ranges::copy(loadVec, unit.begin());
        std::ranges::copy(tideVec, unit.begin() + ndof);
        found = units.emplace(l, system.Solve(unit, 2)).first;
      }
      return found->second;
    }();
    const auto columns = l + 1;
    auto add = [&](const std::optional<std::vector<Complex>>& columnsIn,
                   int which) {
      if (!columnsIn) return;
      for (auto c = 0; c < columns; ++c) {
        const auto amplitude = (*columnsIn)[c];
        if (amplitude == 0.0) continue;
        for (auto dof = 0; dof < ndof; ++dof) {
          x[static_cast<std::size_t>(c) * ndof + dof] +=
              amplitude *
              responses[static_cast<std::size_t>(which) * ndof + dof];
        }
      }
    };
    if (load) add(load(l, t), 0);
    if (tide) add(tide(l, t), 1);
  }
}

const DegreeOperator& Viscoelastic3D::FlavourSpheroidal(Flavour& flavour,
                                                        int l) const {
  const auto lock = LockCaches();
  auto found = flavour.spheroidal.find(l);
  if (found == flavour.spheroidal.end()) {
    found =
        flavour.spheroidal
            .emplace(l,
                     std::make_unique<DegreeOperator>(
                         *flavour.material, l,
                         DegreeOperator::Options{
                             .eps = options_.eps,
                             .truncation = options_.truncation,
                             .point_shear = flavour.pointShear.points != nullptr
                                                ? &flavour.pointShear
                                                : nullptr,
                             .cuts = options_.radial_cuts,
                             .segments = options_.segments,
                             .combine = options_.collectives.sum}))
            .first;
  }
  return *found->second;
}

const ToroidalOperator& Viscoelastic3D::FlavourToroidal(Flavour& flavour,
                                                        int l) const {
  const auto lock = LockCaches();
  auto found = flavour.toroidal.find(l);
  if (found == flavour.toroidal.end()) {
    found =
        flavour.toroidal
            .emplace(l,
                     std::make_unique<ToroidalOperator>(
                         *flavour.material, l,
                         ToroidalOperator::Options{
                             .eps = options_.eps,
                             .truncation = options_.truncation,
                             .pin_fraction = options_.toroidal_pin_fraction,
                             .point_shear = flavour.pointShear.points != nullptr
                                                ? &flavour.pointShear
                                                : nullptr,
                             .cuts = options_.radial_cuts,
                             .segments = options_.segments,
                             .combine = options_.collectives.sum}))
            .first;
  }
  return *found->second;
}

GridCoupling::SpectralTensor Viscoelastic3D::ElasticStrainCoefficients(
    const State& state, const LoadCallable3D& load,
    const LoadCallable3D& tide) const {
  return StrainCoefficients(
      SolveElastic(state, load, tide),
      [&](int l) -> const DegreeOperator& { return Spheroidal(l); },
      [&](int l) -> const ToroidalOperator& { return Toroidal(l); });
}

GridCoupling::TensorField Viscoelastic3D::ElasticStrain(
    const State& state, const LoadCallable3D& load,
    const LoadCallable3D& tide) const {
  return coupling_.Synthesise(ElasticStrainCoefficients(state, load, tide));
}

void Viscoelastic3D::PrimeStrain(State& state, const LoadCallable3D& load,
                                 const LoadCallable3D& tide) const {
  if (state.strain.has_value()) return;
  if (!state.strainCoefficients.has_value()) {
    state.strainCoefficients = ElasticStrainCoefficients(state, load, tide);
  }
  state.strain = coupling_.Synthesise(*state.strainCoefficients);
}

GridCoupling::TensorField Viscoelastic3D::TakeTensor() const {
  {
    auto lock = std::scoped_lock(*scratchMutex_);
    if (!spareTensors_.empty()) {
      auto tensor = std::move(spareTensors_.back());
      spareTensors_.pop_back();
      return tensor;
    }
  }
  return coupling_.ZeroTensor();
}

std::vector<double> Viscoelastic3D::TakeField() const {
  {
    auto lock = std::scoped_lock(*scratchMutex_);
    if (!spareFields_.empty()) {
      auto field = std::move(spareFields_.back());
      spareFields_.pop_back();
      return field;
    }
  }
  return std::vector<double>(coupling_.GridValueCount());
}

void Viscoelastic3D::Give(GridCoupling::TensorField&& tensor) const {
  const auto size = coupling_.GridValueCount();
  if (tensor.c[0].size() != size || tensor.c[1].size() != size ||
      tensor.r.size() != size) {
    return;
  }
  auto lock = std::scoped_lock(*scratchMutex_);
  // one step's worth: a memory tensor per branch and the workspace
  if (spareTensors_.size() < rheology_->BranchCount() + 1) {
    spareTensors_.push_back(std::move(tensor));
  }
}

void Viscoelastic3D::Give(std::vector<double>&& field) const {
  const auto size = coupling_.GridValueCount();
  if (field.size() != size) return;
  auto lock = std::scoped_lock(*scratchMutex_);
  // one step's worth: a weight per branch and the lateral weight
  if (spareFields_.size() < rheology_->BranchCount() + 1) {
    spareFields_.push_back(std::move(field));
  }
}

void Viscoelastic3D::Recycle(State&& state) const {
  for (auto& tensor : state.m) Give(std::move(tensor));
  if (state.strain.has_value()) Give(std::move(*state.strain));
}

void Viscoelastic3D::ReleaseScratch() const {
  auto tensors = std::vector<GridCoupling::TensorField>{};
  auto fields = std::vector<std::vector<double>>{};
  {
    auto lock = std::scoped_lock(*scratchMutex_);
    tensors.swap(spareTensors_);
    fields.swap(spareFields_);
  }
  // freed here, outside the lock
}

Viscoelastic3D::OpenStep::~OpenStep() {
  if (op_ == nullptr) return;
  try {
    for (auto& tensor : memory_) op_->Give(std::move(tensor));
    op_->Give(std::move(grid_));
    op_->Give(std::move(weight_));
    for (auto& field : beta_) op_->Give(std::move(field));
  } catch (...) {
    // the arrays are freed instead
  }
}

Viscoelastic3D::OpenStep Viscoelastic3D::StartStep(
    const State& state, double dt, Scheme scheme, const LoadCallable3D& load,
    const LoadCallable3D& tide) const {
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "the 3D step is implicit: ExponentialTrapezoid or BackwardEuler");
  }
  const auto values = coupling_.GridValueCount();
  const auto nodes = static_cast<std::size_t>(coupling_.NodeCount());
  const auto K = state.m.size();

  OpenStep step;
  step.op_ = this;
  step.flavour_ = FlavourOf(scheme, dt);
  step.scheme_ = scheme;
  step.dt_ = dt;
  step.t_ = state.t + dt;
  // The start strain, as the coefficients the iteration starts from and
  // on the grid for the decayed memory: from the state's cache (either
  // half made from the other), or a consistent solve under the start-time
  // channels.
  if (state.strainCoefficients.has_value() &&
      (bulk_.empty() || !state.strainCoefficients->scalar.empty())) {
    step.s0_ = *state.strainCoefficients;
  } else if (state.strain.has_value() &&
             (bulk_.empty() || !state.strain->scalar.empty())) {
    step.s0_ = coupling_.Analyse(*state.strain);
  } else {
    step.s0_ = ElasticStrainCoefficients(state, load, tide);
  }
  GridCoupling::TensorField made;
  if (!state.strain.has_value()) made = coupling_.Synthesise(step.s0_);
  const auto& d0 = state.strain.has_value() ? *state.strain : made;

  // Per branch and grid point the step weights, the decayed memory m~_k
  // and its beta_k for Finish; summed over branches, the lateral weight
  // and the fixed memory force sum_k mu_k m~_k, analysed once here. The
  // fixed force is formed in what becomes Respond's grid workspace.
  // Every array is written before it is read (the first branch assigns
  // what the others add to), so reused ones need no clearing.
  step.beta_.resize(K);
  step.weight_ = TakeField();
  step.grid_ = TakeTensor();
  auto& fixed = step.grid_;
  if (K == 0) {
    detail::ParallelElements(values, options_.parallel, [&](std::size_t i) {
      step.weight_[i] = 0.0;
      fixed.c[0][i] = 0.0;
      fixed.c[1][i] = 0.0;
      fixed.r[i] = 0.0;
    });
  }
  for (auto k = std::size_t{0}; k < K; ++k) {
    step.memory_.push_back(TakeTensor());
    step.beta_[k] = TakeField();
    detail::ParallelElements(nodes, options_.parallel, [&](std::size_t node) {
      const auto itauRef = itauRef_[k][node];
      const auto refBeta = scheme == Scheme::BackwardEuler
                               ? 1.0 - BackwardEulerWeight(dt * itauRef)
                               : ExponentialTrapezoidWeights(dt * itauRef).beta;
      const auto mu = muBranch_[k][node];
      const auto base = coupling_.FieldOffset(static_cast<int>(node));
      const auto size = coupling_.FieldSizeOf(static_cast<int>(node));
      for (auto x = std::size_t{0}; x < size; ++x) {
        const auto h = dt * itau_[k][base + x];
        double e_w, a_w, b_w;
        if (scheme == Scheme::BackwardEuler) {
          const auto w = BackwardEulerWeight(h);
          e_w = w;
          a_w = 0.0;
          b_w = 1.0 - w;
        } else {
          const auto w = ExponentialTrapezoidWeights(h);
          e_w = w.e;
          a_w = w.alpha;
          b_w = w.beta;
        }
        const auto at = base + x;
        step.beta_[k][at] = b_w;
        auto& memory = step.memory_[k];
        for (auto c = 0; c < 2; ++c) {
          memory.c[c][at] = e_w * state.m[k].c[c][at] + a_w * d0.c[c][at];
        }
        memory.r[at] = e_w * state.m[k].r[at] + a_w * d0.r[at];
        if (k == 0) {
          // 0.0 + v as an accumulation onto zeros gives (a -0.0 sum
          // becomes +0.0), so the step is bitwise the cleared form
          step.weight_[at] = 0.0 + mu * (b_w - refBeta);
          for (auto c = 0; c < 2; ++c) {
            // complex + complex, so a -0.0 imaginary part becomes +0.0 too
            fixed.c[c][at] = Complex{} + mu * memory.c[c][at];
          }
          fixed.r[at] = 0.0 + mu * memory.r[at];
          continue;
        }
        if (mu == 0.0) continue;
        step.weight_[at] += mu * (b_w - refBeta);
        for (auto c = 0; c < 2; ++c) fixed.c[c][at] += mu * memory.c[c][at];
        fixed.r[at] += mu * memory.r[at];
      }
    });
  }
  // The lateral shear modulus: the weight gains -rho(x) mu_eff(x) with
  // mu_eff(x) = mu_U - sum_k mu_k beta_k(x) the reference moduli under
  // the local weights, and the memory force the factor (1 + rho(x)) on
  // every branch modulus.
  if (!shear_.empty()) {
    detail::ParallelElements(nodes, options_.parallel, [&](std::size_t node) {
      const auto base = coupling_.FieldOffset(static_cast<int>(node));
      const auto size = coupling_.FieldSizeOf(static_cast<int>(node));
      for (auto x = std::size_t{0}; x < size; ++x) {
        const auto at = base + x;
        auto effective = muUnrelaxed_[node];
        for (auto k = std::size_t{0}; k < K; ++k) {
          effective -= muBranch_[k][node] * step.beta_[k][at];
        }
        step.weight_[at] -= shear_[at] * effective;
        const auto factor = 1.0 + shear_[at];
        fixed.c[0][at] *= factor;
        fixed.c[1][at] *= factor;
        fixed.r[at] *= factor;
      }
    });
  }
  fixed.scalar.clear();
  step.fixed_ = coupling_.Analyse(fixed);

  // The first pass's force, from the grid strain already in hand, so that
  // the first Respond (of s0) skips its synthesis; under a lateral bulk
  // modulus its dilatation too, which the start strain must carry.
  detail::ParallelElements(values, options_.parallel, [&](std::size_t i) {
    fixed.c[0][i] = step.weight_[i] * d0.c[0][i];
    fixed.c[1][i] = step.weight_[i] * d0.c[1][i];
    fixed.r[i] = step.weight_[i] * d0.r[i];
  });
  if (!bulk_.empty()) {
    if (d0.scalar.empty()) {
      throw std::invalid_argument(
          "under a lateral bulk modulus a strain cache carries the dilatation");
    }
    fixed.scalar.resize(values);
    detail::ParallelElements(values, options_.parallel, [&](std::size_t i) {
      fixed.scalar[i] = negativeBulk_[i] * d0.scalar[i];
    });
  }
  coupling_.Analyse(fixed, step.analysed_);
  AddSpectral(step.analysed_, step.fixed_, options_.parallel);
  step.startForce_ = true;
  return step;
}

Viscoelastic3D::Solutions Viscoelastic3D::OpenStep::Respond(
    const GridCoupling::SpectralTensor& strain) const {
  const auto& op = *op_;
  const auto& coupling = op.coupling_;
  auto& flavour = *flavour_;
  // The lateral force of the strain (synthesised, weighted pointwise,
  // analysed, in the step's buffers) plus the step's fixed memory force;
  // for the start strain, the first time, formed already by StartStep.
  auto formed = std::exchange(startForce_, false) &&
                SameValues(strain, s0_, op.options_.parallel);
  if (op.options_.collectives.Distributed()) {
    // every process alike: the start force only where all may use it
    auto differs = formed ? 0.0 : 1.0;
    op.options_.collectives.max(std::span(&differs, 1));
    formed = differs == 0.0;
  }
  if (!formed) {
    coupling.Synthesise(strain, grid_);
    coupling.Multiply(grid_, weight_);
    if (!grid_.scalar.empty()) coupling.MultiplyScalar(grid_, op.negativeBulk_);
    coupling.Analyse(grid_, analysed_);
    AddSpectral(analysed_, fixed_, op.options_.parallel);
  }
  const auto spheroidal = [&](int l) -> const DegreeOperator& {
    return op.FlavourSpheroidal(flavour, l);
  };
  const auto toroidal = [&](int l) -> const ToroidalOperator& {
    return op.FlavourToroidal(flavour, l);
  };
  if (extraSpheroidal_.empty() && analysed_.scalar.empty()) {
    return op.SolveAll(analysed_, spheroidal, toroidal);
  }
  auto sph = op.SpheroidalForces(analysed_, spheroidal);
  for (const auto& [l, f] : extraSpheroidal_) {
    auto& b = sph.at(l);
    for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
  }
  return op.SolveForces(sph, coupling.ToroidalForce(analysed_, toroidal),
                        spheroidal, toroidal);
}

void Viscoelastic3D::OpenStep::AddForce(
    const GridCoupling::SpectralTensor& analysed,
    const std::map<int, std::vector<Complex>>& spheroidal) {
  const auto& op = *op_;
  const auto in = Blocks(analysed);
  const auto out = Blocks(fixed_);
  if (in.size() != out.size()) {
    throw std::invalid_argument("a force's tensor has the coupling's layout");
  }
  for (auto b = std::size_t{0}; b < out.size(); ++b) {
    auto& o = *out[b];
    const auto& f = *in[b];
    if (f.size() != o.size()) {
      throw std::invalid_argument("a force's tensor has the coupling's layout");
    }
    detail::ParallelElements(o.size(), op.options_.parallel,
                             [&](std::size_t i) { o[i] += f[i]; });
  }
  for (const auto& [l, f] : spheroidal) {
    const auto expected =
        static_cast<std::size_t>(Spheroidal(l).DofCount()) * (l + 1);
    if (f.size() != expected) {
      throw std::invalid_argument(std::format(
          "a degree-{} force has {} entries, the solutions {}", l, f.size(),
          expected));
    }
    auto& extra = extraSpheroidal_[l];
    if (extra.empty()) {
      extra = f;
    } else {
      for (auto i = std::size_t{0}; i < extra.size(); ++i) extra[i] += f[i];
    }
  }
  startForce_ = false;
}

void Viscoelastic3D::OpenStep::AddChannels(Solutions& solutions,
                                           const LoadCallable3D& load,
                                           const LoadCallable3D& tide) const {
  if (!load && !tide) return;
  const auto& op = *op_;
  auto& flavour = *flavour_;
  op.AddUnitChannels(
      solutions, t_, load, tide,
      [&](int l) -> const DegreeOperator& {
        return op.FlavourSpheroidal(flavour, l);
      },
      flavour.units);
}

GridCoupling::SpectralTensor Viscoelastic3D::OpenStep::StrainOf(
    const Solutions& solutions) const {
  const auto& op = *op_;
  auto& flavour = *flavour_;
  // Into the iterate Converge last spent, zeroed, when there is one.
  auto spectral = std::move(spare_);
  if (spectral.r.empty()) {
    spectral = op.ZeroSpectralFor();
  } else {
    for (auto* block : Blocks(spectral)) {
      auto& b = *block;
      detail::ParallelElements(b.size(), op.options_.parallel,
                               [&](std::size_t i) { b[i] = 0.0; });
    }
  }
  op.AddStrainCoefficients(
      spectral, solutions,
      [&](int l) -> const DegreeOperator& {
        return op.FlavourSpheroidal(flavour, l);
      },
      [&](int l) -> const ToroidalOperator& {
        return op.FlavourToroidal(flavour, l);
      });
  return spectral;
}

double Viscoelastic3D::OpenStep::InertiaMoment(
    const Solutions& solutions) const {
  const auto& x = solutions.spheroidal.at(0);
  auto column = std::vector<double>(x.size());
  for (auto i = std::size_t{0}; i < x.size(); ++i) column[i] = x[i].real();
  // Distributed, each process's share, summed.
  op_->FlavourSpheroidal(*flavour_, 0).MaskToShare(column);
  auto moment = 0.0;
  {
    const auto lock = op_->LockCaches();
    moment = op_->reference_.InertiaMoment(column, 1)[0];
  }
  if (op_->options_.collectives.Distributed()) {
    op_->options_.collectives.sum(std::span(&moment, 1));
  }
  return moment;
}

const DegreeOperator& Viscoelastic3D::OpenStep::Spheroidal(int l) const {
  return op_->FlavourSpheroidal(*flavour_, l);
}

const ToroidalOperator& Viscoelastic3D::OpenStep::Toroidal(int l) const {
  return op_->FlavourToroidal(*flavour_, l);
}

const std::vector<double>& Viscoelastic3D::OpenStep::Units(int l) const {
  const auto& system = Spheroidal(l);
  const auto lock = op_->LockCaches();
  auto& units = flavour_->units;
  auto found = units.find(l);
  if (found == units.end()) {
    const auto ndof = system.DofCount();
    auto unit = std::vector<double>(static_cast<std::size_t>(ndof) * 2, 0.0);
    const auto loadVec = system.Load();
    const auto tideVec = system.Tide(l == 0 ? 2 : -1);
    std::ranges::copy(loadVec, unit.begin());
    std::ranges::copy(tideVec, unit.begin() + ndof);
    found = units.emplace(l, system.Solve(unit, 2)).first;
  }
  return found->second;
}

const LoveNumbers& Viscoelastic3D::OpenStep::EffectiveLove() const {
  const auto lock = op_->LockCaches();
  return op_->reference_.EffectiveLoveNumbers(scheme_, dt_);
}

void Viscoelastic3D::OpenStep::AddMemory(
    const std::vector<GridCoupling::TensorField>& increment) {
  const auto& op = *op_;
  const auto& coupling = op.coupling_;
  const auto K = memory_.size();
  if (increment.size() != K) {
    throw std::invalid_argument(std::format(
        "a memory increment gives one tensor per branch: {} branches", K));
  }
  const auto values = coupling.GridValueCount();
  const auto nodes = static_cast<std::size_t>(coupling.NodeCount());
  // The decayed memory gains the increment, and the fixed force its
  // branch-weighted sum, analysed once.
  auto force = coupling.ZeroTensor();
  for (auto k = std::size_t{0}; k < K; ++k) {
    const auto& add = increment[k];
    if (add.r.size() != values) {
      throw std::invalid_argument(
          "a memory increment's tensors live on the coupling's grid");
    }
    const auto& mu = op.muBranch_[k];
    auto& memory = memory_[k];
    detail::ParallelElements(nodes, op.options_.parallel, [&](std::size_t node) {
      const auto base = coupling.FieldOffset(static_cast<int>(node));
      const auto size = coupling.FieldSizeOf(static_cast<int>(node));
      for (auto x = std::size_t{0}; x < size; ++x) {
        const auto at = base + x;
        for (auto c = 0; c < 2; ++c) {
          memory.c[c][at] += add.c[c][at];
          force.c[c][at] += mu[node] * add.c[c][at];
        }
        memory.r[at] += add.r[at];
        force.r[at] += mu[node] * add.r[at];
      }
    });
  }
  const auto analysed = coupling.Analyse(force);
  const auto in = Blocks(analysed);
  const auto out = Blocks(fixed_);
  for (auto b = std::size_t{0}; b < out.size(); ++b) {
    auto& o = *out[b];
    const auto& f = *in[b];
    detail::ParallelElements(o.size(), op.options_.parallel,
                             [&](std::size_t i) { o[i] += f[i]; });
  }
  startForce_ = false;
}

Viscoelastic3D::State Viscoelastic3D::OpenStep::Finish(
    const GridCoupling::SpectralTensor& strain) const& {
  auto d = op_->coupling_.Synthesise(strain);
  State next;
  next.t = t_;
  for (auto k = std::size_t{0}; k < memory_.size(); ++k) {
    auto tensor = memory_[k];
    AdvanceMemory(tensor, beta_[k], d, op_->options_.parallel);
    next.m.push_back(std::move(tensor));
  }
  next.strain = std::move(d);
  next.strainCoefficients = strain;
  return next;
}

Viscoelastic3D::State Viscoelastic3D::OpenStep::Finish(
    GridCoupling::SpectralTensor strain) && {
  // The grid workspace takes d and the decayed memory advances in place:
  // both become the new state.
  op_->coupling_.Synthesise(strain, grid_);
  State next;
  next.t = t_;
  for (auto k = std::size_t{0}; k < memory_.size(); ++k) {
    AdvanceMemory(memory_[k], beta_[k], grid_, op_->options_.parallel);
    next.m.push_back(std::move(memory_[k]));
  }
  next.strain = std::move(grid_);
  next.strainCoefficients = std::move(strain);
  return next;
}

std::pair<GridCoupling::SpectralTensor, Viscoelastic3D::StepInfo>
Viscoelastic3D::OpenStep::Converge(
    const Pass& pass, const IterationControl& control,
    const GridCoupling::SpectralTensor* start) const {
  const auto parallel = op_->options_.parallel;
  auto s = start != nullptr ? *start : s0_;
  StepInfo info;
  info.converged = false;
  // Picard until a pass contracts too slowly, then Anderson for the rest
  // of the step (IterationControl). Not adaptive (anderson_rate <= 0):
  // Anderson from the first pass.
  const auto adaptive = control.anderson_rate > 0.0;
  auto mixer = std::optional<detail::AndersonMixer>{};
  if (control.anderson_memory > 0 && !adaptive) {
    mixer.emplace(control.anderson_memory, parallel,
                  op_->options_.collectives.sum);
  }
  auto previous = 0.0;  // the last pass's residual
  for (auto iteration = 0; iteration < control.max_iterations; ++iteration) {
    auto sNext = pass(s);
    const auto next = Blocks(sNext);
    const auto now = Blocks(std::as_const(s));

    // The relative sup-norm change of the coefficients, compared on
    // squared magnitudes (maxima: the same whatever the thread count).
    // The maxima pass over NaNs, so a diverged pass is caught apart: it
    // ends the iteration unconverged, on the last finite iterate.
    auto scale = 0.0, diff = 0.0;
    auto finite = true;
    for (auto b = std::size_t{0}; b < next.size(); ++b) {
      const auto& n = *next[b];
      const auto& o = *now[b];
      const auto size = n.size();
#pragma omp parallel for schedule(static) reduction(max : scale, diff) \
    reduction(&& : finite) if (parallel)
      for (auto i = std::size_t{0}; i < size; ++i) {
        finite =
            finite && std::isfinite(n[i].real()) && std::isfinite(n[i].imag());
        scale = std::max(scale, std::norm(n[i]));
        diff = std::max(diff, std::norm(n[i] - o[i]));
      }
    }
    const auto& collectives = op_->options_.collectives;
    if (collectives.Distributed()) {
      auto global = std::array{scale, diff, finite ? 0.0 : 1.0};
      collectives.max(global);
      scale = global[0];
      diff = global[1];
      finite = global[2] == 0.0;
    }
    info.iterations = iteration + 1;
    if (!finite) {
      info.residual = std::numeric_limits<double>::quiet_NaN();
      break;
    }
    info.residual = scale > 0.0 ? std::sqrt(diff) / std::sqrt(scale) : 0.0;
    if (info.residual <= control.rtol) {
      info.converged = true;
      s = std::move(sNext);
      break;
    }
    if (control.damping != 1.0) {
      for (auto b = std::size_t{0}; b < next.size(); ++b) {
        auto& n = *next[b];
        const auto& o = *now[b];
        detail::ParallelElements(n.size(), parallel, [&](std::size_t i) {
          n[i] = o[i] + control.damping * (n[i] - o[i]);
        });
      }
    }
    const auto residual = info.residual;
    if (adaptive && !mixer && control.anderson_memory > 0 && iteration > 0 &&
        residual > control.anderson_rate * previous) {
      mixer.emplace(control.anderson_memory, parallel,
                    op_->options_.collectives.sum);
    }
    previous = residual;
    if (mixer) {
      mixer->UpdateInPlace(RealSegments(s), RealSegments(std::as_const(sNext)));
      ++info.mixedPasses;
    } else {
      std::swap(s, sNext);
    }
    spare_ = std::move(sNext);  // the spent iterate, for StrainOf to fill
  }
  return {std::move(s), info};
}

Viscoelastic3D::StepResult Viscoelastic3D::Step(
    const State& state, double dt, Scheme scheme, const LoadCallable3D& load,
    const LoadCallable3D& tide) const {
  auto step = StartStep(state, dt, scheme, load, tide);
  Solutions solutions;
  auto [strain, info] = step.Converge(
      [&](const GridCoupling::SpectralTensor& s) {
        solutions = step.Respond(s);
        step.AddChannels(solutions, load, tide);
        return step.StrainOf(solutions);
      },
      {.max_iterations = options_.max_iterations,
       .rtol = options_.rtol,
       .damping = options_.damping,
       .anderson_memory = options_.anderson_memory,
       .anderson_rate = options_.anderson_rate});
  return {std::move(step).Finish(std::move(strain)), std::move(solutions),
          info};
}

std::map<int, std::vector<Viscoelastic3D::Complex>> Viscoelastic3D::Surface(
    const Solutions& solutions, Component component) const {
  auto out = std::map<int, std::vector<Complex>>{};
  for (const auto& [l, x] : solutions.spheroidal) {
    const auto& op = Spheroidal(l);
    const auto columns = l + 1;
    const auto dof = op.SurfaceDof(component);
    auto values = std::vector<Complex>(columns, 0.0);
    if (dof >= 0 && op.Holds(dof)) {
      for (auto c = 0; c < columns; ++c) {
        values[c] = x[static_cast<std::size_t>(c) * op.DofCount() + dof];
      }
    }
    out.emplace(l, std::move(values));
  }
  if (options_.collectives.Distributed()) {
    // The surface segment's process holds them: summed, every process has
    // them.
    auto all = std::vector<Complex>{};
    for (const auto& [l, values] : out) {
      all.insert(all.end(), values.begin(), values.end());
    }
    options_.collectives.sum(Reals(all));
    auto at = all.begin();
    for (auto& [l, values] : out) {
      std::copy(at, at + static_cast<std::ptrdiff_t>(values.size()),
                values.begin());
      at += static_cast<std::ptrdiff_t>(values.size());
    }
  }
  return out;
}

}  // namespace sl3dpp
