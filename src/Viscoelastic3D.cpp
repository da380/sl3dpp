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
#include "sl3dpp/detail/Parallel.hpp"

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
  for (auto& s : t.scalars) out.push_back(pairs(s));
  return out;
}

std::vector<std::span<const double>> RealSegments(
    const GridCoupling::SpectralTensor& t) {
  auto pairs = [](const std::vector<std::complex<double>>& v) {
    return std::span<const double>(reinterpret_cast<const double*>(v.data()),
                                   2 * v.size());
  };
  auto out = std::vector{pairs(t.c[0]), pairs(t.c[1]), pairs(t.r)};
  for (const auto& s : t.scalars) out.push_back(pairs(s));
  return out;
}

// The coefficient blocks of a spectral tensor (three, and the scalar
// companions where carried), for loops that treat them alike.
std::vector<std::vector<std::complex<double>>*> Blocks(
    GridCoupling::SpectralTensor& t) {
  auto out = std::vector{&t.c[0], &t.c[1], &t.r};
  for (auto& s : t.scalars) out.push_back(&s);
  return out;
}

std::vector<const std::vector<std::complex<double>>*> Blocks(
    const GridCoupling::SpectralTensor& t) {
  auto out = std::vector{&t.c[0], &t.c[1], &t.r};
  for (const auto& s : t.scalars) out.push_back(&s);
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
    detail::ForEach(parallel, o.size(), [&](std::size_t i) { o[i] += f[i]; });
  }
  for (auto k = std::size_t{0}; k < out.scalars.size() && k < in.scalars.size(); ++k) {
    auto& o = out.scalars[k];
    const auto& f = in.scalars[k];
    detail::ForEach(parallel, o.size(), [&](std::size_t i) { o[i] += f[i]; });
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
  detail::ForEach(parallel, m.r.size(), [&](std::size_t i) {
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
    : Viscoelastic3D(rheology, lmax,
                     SurfaceGrid(lmax, {.nmax = 2,
                                        .oversampling = options.oversampling,
                                        .parallel = options.parallel,
                                        .matrix_kernel = options.matrix_kernel,
                                        .cache_bytes = options.cache_bytes}),
                     delta, options) {}

std::vector<char> Viscoelastic3D::ActiveElementsFor(const Rheology& rheology,
                                                     int lmax,
                                                     const SurfaceGrid& grid,
                                                     const Options& options) {
  const auto& moduli = options.moduli;
  if (!moduli.shear && !moduli.bulk) return {};
  // A probe over the whole solid: the functions sampled at every memory
  // point's grid; an element is carried if viscous or perturbed there.
  const GridCoupling probe(rheology, lmax, grid,
                           {.eps = options.eps,
                            .truncation = options.truncation,
                            .oversampling = options.oversampling,
                            .parallel = options.parallel,
                            .matrix_kernel = options.matrix_kernel,
                            .cache_bytes = options.cache_bytes,
                            .memory_points = options.memory_points,
                            .transform_degrees = options.transform_degrees,
                            .all_solid = true});
  const auto& mesh = rheology.BaseMaterial().Mesh();
  const auto n = mesh.NGll();
  auto active = std::vector<char>(static_cast<std::size_t>(mesh.ElementCount()), 0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto i = 0; i < n; ++i) {
      if (rheology.IsActive(static_cast<std::size_t>(e) * n + i)) active[e] = 1;
    }
  }
  for (auto node = 0; node < probe.NodeCount(); ++node) {
    const auto e = probe.ElementOfNode(node);
    if (active[e] != 0) continue;
    const auto r = probe.RadiusOfNode(node);
    const auto& sampling = probe.GridOf(node);
    auto perturbed = false;
    for (auto theta : sampling.CoLatitudes()) {
      for (auto phi : sampling.Longitudes()) {
        for (const auto* f : {&moduli.shear, &moduli.bulk}) {
          if (*f && (*f)(r, theta, phi) != 0.0) perturbed = true;
        }
        if (perturbed) break;
      }
      if (perturbed) break;
    }
    if (perturbed) active[e] = 1;
  }
  return active;
}

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
           .cache_bytes = options.cache_bytes,
           .memory_points = options.memory_points,
           .transform_degrees = options.transform_degrees,
           .elements = OwnedElements(rheology.BaseMaterial().Mesh(), options),
           .active_elements =
               ActiveElementsFor(rheology, lmax, grid, options)}),
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
  lateralModuli_ = static_cast<bool>(moduli.shear) || static_cast<bool>(moduli.bulk);
  if (lateralModuli_) {
    const auto& material = rheology_->BaseMaterial();
    const auto& mesh = material.Mesh();
    const auto n = mesh.NGll();
    const auto& points = coupling_.Points();
    const auto gll = points.set == MemoryPointSet::Gll;
    const auto longTerm = rheology_->MuLongTerm();
    const auto kappaNodal = rheology_->Kappa();
    muUnrelaxed_.assign(nodes, 0.0);
    auto& kappaAtNodes = kappaAtNodes_;
    kappaAtNodes.assign(nodes, 0.0);
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
      dilatationBlock_ = static_cast<int>(companions_);
      companions_ += 1;
    }
  }
}

void Viscoelastic3D::WeightCompanions(GridCoupling::TensorField& field) const {
  if (field.scalars.size() != companions_) {
    throw std::invalid_argument(
        "a grid tensor's scalar companions match the operator's layout");
  }
  if (dilatationBlock_ >= 0) {
    coupling_.MultiplyScalar(field, static_cast<std::size_t>(dilatationBlock_),
                             negativeBulk_);
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
  spectral.scalars.assign(companions_,
                          std::vector<Complex>(coupling_.ScalarCoefficientCount(), 0.0));
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
  if (weighted.scalars.empty()) return forces;
  if (weighted.scalars.size() < companions_) {
    throw std::invalid_argument(
        "a weighted tensor's scalar companions match the operator's layout");
  }
  auto add = [&](const std::map<int, std::vector<Complex>>& extra) {
    for (auto& [l, b] : forces) {
      const auto& f = extra.at(l);
      for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
    }
  };
  if (dilatationBlock_ >= 0) {
    add(coupling_.DilatationForce(
        weighted.scalars[static_cast<std::size_t>(dilatationBlock_)], spheroidal));
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
  // the companions weighted by the positive perturbations: +(dK) x
  if (dilatationBlock_ >= 0) {
    coupling_.MultiplyScalar(grid, static_cast<std::size_t>(dilatationBlock_), bulk_);
  }
  const auto weighted = coupling_.Analyse(grid);
  return {SpheroidalForces(weighted, spheroidal),
          coupling_.ToroidalForce(weighted, toroidal)};
}

std::pair<std::map<int, std::vector<Viscoelastic3D::Complex>>,
          std::map<int, std::vector<Viscoelastic3D::Complex>>>
Viscoelastic3D::ApplyLateralOperator(
    const Solutions& x,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    const std::function<std::pair<std::map<int, std::vector<Complex>>,
                                  std::map<int, std::vector<Complex>>>(
        const Solutions&)>& lateralForce) const {
  const auto& collectives = options_.collectives;
  // One degree's columns through its reference band: the complex columns
  // as real pairs, assembled from the shares when distributed.
  struct Task {
    int l = 0;
    const std::vector<Complex>* x = nullptr;
    const DegreeOperator* sph = nullptr;
    const ToroidalOperator* tor = nullptr;
    std::vector<double> real;
    std::vector<Complex>* out = nullptr;
  };
  auto toReal = [](const std::vector<Complex>& x, int ndof, int columns) {
    auto real = std::vector<double>(static_cast<std::size_t>(ndof) * 2 * columns);
    for (auto c = 0; c < columns; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        real[static_cast<std::size_t>(c) * ndof + d] =
            x[static_cast<std::size_t>(c) * ndof + d].real();
        real[static_cast<std::size_t>(columns + c) * ndof + d] =
            x[static_cast<std::size_t>(c) * ndof + d].imag();
      }
    }
    return real;
  };
  auto [sph, tor] = lateralForce(x);
  auto tasks = std::vector<Task>{};
  for (const auto& [l, xl] : x.spheroidal) {
    tasks.push_back({.l = l, .x = &xl, .sph = &spheroidal(l), .out = &sph.at(l)});
  }
  for (const auto& [l, wl] : x.toroidal) {
    if (l < 1) continue;
    tasks.push_back({.l = l, .x = &wl, .tor = &toroidal(l), .out = &tor.at(l)});
  }
  // the real pairs, as shares when distributed, summed in one
  // collective for every degree (the collective serial, in the
  // same order on every process)
  detail::ForEachTask(
      options_.parallel, static_cast<int>(tasks.size()), [&](int i) {
        auto& task = tasks[i];
        const auto columns = task.l + 1;
        const auto ndof =
            task.tor != nullptr ? task.tor->DofCount() : task.sph->DofCount();
        task.real = toReal(*task.x, ndof, columns);
        if (collectives.Distributed()) {
          if (task.tor != nullptr) {
            task.tor->MaskToShare(task.real, 2 * columns);
          } else {
            task.sph->MaskToShare(task.real, 2 * columns);
          }
        }
      });
  if (collectives.Distributed()) {
    auto size = std::size_t{0};
    for (const auto& task : tasks) size += task.real.size();
    auto all = std::vector<double>(size);
    auto at = all.begin();
    for (const auto& task : tasks) at = std::ranges::copy(task.real, at).out;
    collectives.sum(all);
    at = all.begin();
    for (auto& task : tasks) {
      std::copy(at, at + static_cast<std::ptrdiff_t>(task.real.size()),
                task.real.begin());
      at += static_cast<std::ptrdiff_t>(task.real.size());
    }
  }
  // the bands, threaded over the degrees; the products masked
  // back to this process's share, the layout the forces have
  detail::ForEachTask(
      options_.parallel, static_cast<int>(tasks.size()), [&](int i) {
        auto& task = tasks[i];
        const auto columns = task.l + 1;
        const auto ndof =
            task.tor != nullptr ? task.tor->DofCount() : task.sph->DofCount();
        auto y = task.tor != nullptr ? task.tor->Apply(task.real, 2 * columns)
                                     : task.sph->Apply(task.real, 2 * columns);
        if (collectives.Distributed()) {
          if (task.tor != nullptr) {
            task.tor->MaskToShare(y, 2 * columns);
          } else {
            task.sph->MaskToShare(y, 2 * columns);
          }
        }
        auto& b = *task.out;
        for (auto c = 0; c < columns; ++c) {
          for (auto d = 0; d < ndof; ++d) {
            const auto k = static_cast<std::size_t>(c) * ndof + d;
            b[k] =
                Complex{y[k],
                        y[static_cast<std::size_t>(columns + c) * ndof + d]} -
                b[k];
          }
        }
      });
  return std::pair{std::move(sph), std::move(tor)};
}

std::pair<Viscoelastic3D::Solutions, Viscoelastic3D::ElasticSolveInfo>
Viscoelastic3D::ConjugateGradients(
    const std::map<int, std::vector<Complex>>& sphForces,
    const std::map<int, std::vector<Complex>>& torForces,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    const std::function<std::pair<std::map<int, std::vector<Complex>>,
                                  std::map<int, std::vector<Complex>>>(
        const Solutions&)>& lateralForce,
    double rtol, int maxIterations, const Solutions* start,
    double reduction) const {
  // The full operator: the reference systems' action on the complex
  // columns as real pairs, minus the lateral force. Distributed, each
  // process's solution holds its segments and the cut nodes: the whole
  // vector is assembled from the held shares (one collective per
  // degree), the band applied, and the product masked back to this
  // process's share, the layout the forces' shares have.
  const auto& collectives = options_.collectives;
  const auto system = detail::LateralSystem{
      .apply =
          [&](const Solutions& x) {
            return ApplyLateralOperator(x, spheroidal, toroidal, lateralForce);
          },
      .solve =
          [&](const std::map<int, std::vector<Complex>>& s,
              const std::map<int, std::vector<Complex>>& t) {
            return SolveForces(s, t, spheroidal, toroidal);
          },
      .sum = collectives.Distributed() ? collectives.sum : nullptr};
  auto [x, info] = detail::LateralConjugateGradients(
      sphForces, torForces, system,
      {.rtol = rtol, .max_iterations = maxIterations, .reduction = reduction}, true,
      start);
  return {std::move(x), ElasticSolveInfo{info.iterations, info.residual, info.converged,
                                         info.residuals.front()}};
}

Viscoelastic3D::Solutions Viscoelastic3D::LateralElasticSolve(
    const std::map<int, std::vector<Complex>>& sphForces,
    const std::map<int, std::vector<Complex>>& torForces) const {
  // -L x = +(dK) x: the lateral force of the positive perturbations
  auto [x, info] = ConjugateGradients(
      sphForces, torForces,
      [&](int l) -> const DegreeOperator& { return Spheroidal(l); },
      [&](int l) -> const ToroidalOperator& { return Toroidal(l); },
      [&](const Solutions& y) {
        auto [sph, tor] = LateralElasticForce(y);
        for (auto& [l, b] : sph) for (auto& c : b) c = -c;
        for (auto& [l, b] : tor) for (auto& c : b) c = -c;
        return std::pair{std::move(sph), std::move(tor)};
      },
      options_.elastic_rtol, options_.elastic_max_iterations, nullptr);
  lastElasticSolve_ = info;
  if (!info.converged) {
    throw std::runtime_error(std::format(
        "the unrelaxed solve under lateral moduli did not converge: residual "
        "{} after {} iterations",
        info.residual, info.iterations));
  }
  return std::move(x);
}

Viscoelastic3D::Solutions Viscoelastic3D::SolveUnrelaxed(
    const std::map<int, std::vector<Complex>>& sphForces,
    const std::map<int, std::vector<Complex>>& torForces) const {
  if (HasLateralModuli()) return LateralElasticSolve(sphForces, torForces);
  return SolveForces(
      sphForces, torForces,
      [&](int l) -> const DegreeOperator& { return Spheroidal(l); },
      [&](int l) -> const ToroidalOperator& { return Toroidal(l); });
}

void Viscoelastic3D::AddChannelForces(
    std::map<int, std::vector<Complex>>& forces, double t,
    const LoadCallable3D& load, const LoadCallable3D& tide,
    const std::function<const DegreeOperator&(int)>& spheroidal) const {
  if (!load && !tide) return;
  // Named, not a structured binding: Clang 18 does not capture those in a
  // lambda when OpenMP is enabled.
  for (auto& entry : forces) {
    const auto l = entry.first;
    auto& b = entry.second;
    const auto& system = spheroidal(l);
    const auto ndof = system.DofCount();
    auto add = [&](const std::optional<std::vector<Complex>>& columns,
                   std::vector<double> unit) {
      if (!columns) return;
      // distributed, the unit vector as this process's share
      if (options_.collectives.Distributed()) system.MaskToShare(unit, 1);
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
  detail::ForEachTask(options_.parallel, count, [&](int i) {
    const auto& task = tasks[count - 1 - i];  // the largest degrees first
    coupling_.AddStrainOfSolution(
        spectral, task.l, *task.sph, *task.x, task.tor,
        task.w == nullptr ? std::span<const Complex>{}
                          : std::span<const Complex>(*task.w));
    if (dilatationBlock_ >= 0 && !spectral.scalars.empty()) {
      coupling_.AddDilatationOfSolution(
          spectral.scalars[static_cast<std::size_t>(dilatationBlock_)], task.l,
          *task.sph, *task.x);
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
  detail::ForEachTask(
      options_.parallel, static_cast<int>(tasks.size()), [&](int i) {
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
  detail::ForEachTask(
      options_.parallel, static_cast<int>(tasks.size()), [&](int i) {
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
    detail::ForEach(options_.parallel, nodes, [&](std::size_t node) {
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
    detail::ForEach(options_.parallel, combined.r.size(), [&](std::size_t i) {
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
  if (dt < 0.0) throw std::invalid_argument("dt must be non-negative");
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
      state.strainCoefficients->scalars.size() == companions_) {
    step.s0_ = *state.strainCoefficients;
  } else if (state.strain.has_value() &&
             state.strain->scalars.size() == companions_) {
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
    detail::ForEach(options_.parallel, values, [&](std::size_t i) {
      step.weight_[i] = 0.0;
      fixed.c[0][i] = 0.0;
      fixed.c[1][i] = 0.0;
      fixed.r[i] = 0.0;
    });
  }
  for (auto k = std::size_t{0}; k < K; ++k) {
    step.memory_.push_back(TakeTensor());
    step.beta_[k] = TakeField();
    detail::ForEach(options_.parallel, nodes, [&](std::size_t node) {
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
    detail::ForEach(options_.parallel, nodes, [&](std::size_t node) {
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
  fixed.scalars.clear();
  step.fixed_ = coupling_.Analyse(fixed);

  // The first pass's force, from the grid strain already in hand, so that
  // the first Respond (of s0) skips its synthesis; under a lateral bulk
  // modulus its dilatation too, which the start strain must carry.
  detail::ForEach(options_.parallel, values, [&](std::size_t i) {
    fixed.c[0][i] = step.weight_[i] * d0.c[0][i];
    fixed.c[1][i] = step.weight_[i] * d0.c[1][i];
    fixed.r[i] = step.weight_[i] * d0.r[i];
  });
  if (companions_ > 0) {
    if (d0.scalars.size() != companions_) {
      throw std::invalid_argument(
          "under lateral moduli a strain cache carries the scalar companions");
    }
    fixed.scalars = d0.scalars;
    WeightCompanions(fixed);
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
    if (!grid_.scalars.empty()) op.WeightCompanions(grid_);
    coupling.Analyse(grid_, analysed_);
    AddSpectral(analysed_, fixed_, op.options_.parallel);
  }
  const auto spheroidal = [&](int l) -> const DegreeOperator& {
    return op.FlavourSpheroidal(flavour, l);
  };
  const auto toroidal = [&](int l) -> const ToroidalOperator& {
    return op.FlavourToroidal(flavour, l);
  };
  if (extraSpheroidal_.empty() && analysed_.scalars.empty()) {
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

std::pair<Viscoelastic3D::OpenStep::Fields, Viscoelastic3D::OpenStep::Fields>
Viscoelastic3D::OpenStep::LateralForce(const Solutions& x) const {
  const auto& op = *op_;
  const auto& coupling = op.coupling_;
  auto& flavour = *flavour_;
  const auto spheroidal = [&](int l) -> const DegreeOperator& {
    return op.FlavourSpheroidal(flavour, l);
  };
  const auto toroidal = [&](int l) -> const ToroidalOperator& {
    return op.FlavourToroidal(flavour, l);
  };
  auto strain = StrainOf(x);
  coupling.Synthesise(strain, grid_);
  spare_ = std::move(strain);
  coupling.Multiply(grid_, weight_);
  if (!grid_.scalars.empty()) op.WeightCompanions(grid_);
  coupling.Analyse(grid_, analysed_);
  startForce_ = false;  // the buffers no longer hold the start force
  return {op.SpheroidalForces(analysed_, spheroidal),
          coupling.ToroidalForce(analysed_, toroidal)};
}

std::pair<Viscoelastic3D::OpenStep::Fields, Viscoelastic3D::OpenStep::Fields>
Viscoelastic3D::OpenStep::Forces(const LoadCallable3D& load,
                                 const LoadCallable3D& tide) const {
  const auto& op = *op_;
  auto& flavour = *flavour_;
  const auto spheroidal = [&](int l) -> const DegreeOperator& {
    return op.FlavourSpheroidal(flavour, l);
  };
  auto sph = op.SpheroidalForces(fixed_, spheroidal);
  for (const auto& [l, f] : extraSpheroidal_) {
    auto& b = sph.at(l);
    for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
  }
  op.AddChannelForces(sph, t_, load, tide, spheroidal);
  return {std::move(sph),
          op.coupling_.ToroidalForce(fixed_, [&](int l) -> const ToroidalOperator& {
            return op.FlavourToroidal(flavour, l);
          })};
}

std::pair<Viscoelastic3D::OpenStep::Fields, Viscoelastic3D::OpenStep::Fields>
Viscoelastic3D::OpenStep::Apply(const Solutions& x) const {
  const auto& op = *op_;
  auto& flavour = *flavour_;
  return op.ApplyLateralOperator(
      x, [&](int l) -> const DegreeOperator& { return op.FlavourSpheroidal(flavour, l); },
      [&](int l) -> const ToroidalOperator& { return op.FlavourToroidal(flavour, l); },
      [&](const Solutions& y) { return LateralForce(y); });
}

std::pair<Viscoelastic3D::Solutions, Viscoelastic3D::ElasticSolveInfo>
Viscoelastic3D::OpenStep::Solve(const Fields& spheroidal, const Fields& toroidal,
                                const Solutions* start, double rtol,
                                double reduction) const {
  const auto& op = *op_;
  auto& flavour = *flavour_;
  return op.ConjugateGradients(
      spheroidal, toroidal,
      [&](int l) -> const DegreeOperator& { return op.FlavourSpheroidal(flavour, l); },
      [&](int l) -> const ToroidalOperator& { return op.FlavourToroidal(flavour, l); },
      [&](const Solutions& x) { return LateralForce(x); },
      rtol > 0.0 ? rtol : op.options_.solve_rtol, op.options_.solve_max_iterations,
      start, reduction);
}

std::pair<GridCoupling::SpectralTensor, Viscoelastic3D::StepInfo>
Viscoelastic3D::OpenStep::Nest(const Closure& closure,
                               const IterationControl& control,
                               Solutions* solutions) const {
  const auto& op = *op_;
  StepInfo info;
  info.converged = false;
  auto negated = [](const LoadCallable3D& channel) -> LoadCallable3D {
    if (!channel) return {};
    return [channel](int l, double t) {
      auto columns = channel(l, t);
      if (columns) for (auto& c : *columns) c = -c;
      return columns;
    };
  };
  // The outer iterate is the CHANNEL-FREE solutions r (the start strain's
  // response first): the closure maps r to the channels, the solve under
  // them to x, and x less its channels is the image g(r). The outer map
  // is Picard, Anderson-mixed on r as Converge mixes the strain (the
  // control's memory and rate), and measured by the relative sup-norm
  // change of the solutions from the iterate to its image.
  const auto parallel = op.options_.parallel;
  const auto& collectives = op.options_.collectives;
  auto segments = [](Solutions& s) {
    auto out = std::vector<std::span<double>>{};
    for (auto* block : {&s.spheroidal, &s.toroidal}) {
      for (auto& [l, v] : *block) {
        out.emplace_back(reinterpret_cast<double*>(v.data()), 2 * v.size());
      }
    }
    return out;
  };
  auto constSegments = [](const Solutions& s) {
    auto out = std::vector<std::span<const double>>{};
    for (const auto* block : {&s.spheroidal, &s.toroidal}) {
      for (const auto& [l, v] : *block) {
        out.emplace_back(reinterpret_cast<const double*>(v.data()), 2 * v.size());
      }
    }
    return out;
  };
  auto changeOf = [&](const Solutions& image, const Solutions& iterate) {
    auto scale = 0.0, diff = 0.0;
    auto finite = true;
    for (const auto* block : {&image.spheroidal, &image.toroidal}) {
      const auto& other = block == &image.spheroidal ? iterate.spheroidal : iterate.toroidal;
      // Named, not a structured binding: Clang 18 does not take those into
      // an OpenMP region.
      for (const auto& entry : *block) {
        const auto& n = entry.second;
        const auto& o = other.at(entry.first);
        const auto size = n.size();
#pragma omp parallel for schedule(static) reduction(max : scale, diff) \
    reduction(&& : finite) if (parallel)
        for (auto i = std::size_t{0}; i < size; ++i) {
          finite = finite && std::isfinite(n[i].real()) && std::isfinite(n[i].imag());
          scale = std::max(scale, std::norm(n[i]));
          diff = std::max(diff, std::norm(n[i] - o[i]));
        }
      }
    }
    if (collectives.Distributed()) {
      auto global = std::array{scale, diff, finite ? 0.0 : 1.0};
      collectives.max(global);
      scale = global[0];
      diff = global[1];
      finite = global[2] == 0.0;
    }
    if (!finite) return std::numeric_limits<double>::quiet_NaN();
    return scale > 0.0 ? std::sqrt(diff) / std::sqrt(scale) : 0.0;
  };
  const auto adaptive = control.anderson_rate > 0.0;
  auto mixer = std::optional<detail::AndersonMixer>{};
  if (control.anderson_memory > 0 && !adaptive) {
    mixer.emplace(control.anderson_memory, parallel, collectives.sum);
  }
  auto iterate = Respond(s0_);  // the start strain's, channel-free
  LoadCallable3D load, tide;
  Solutions x;
  auto previous = 0.0;
  for (auto pass = 0; pass < control.max_iterations; ++pass) {
    std::tie(load, tide) = closure(iterate);
    const auto [fs, ft] = Forces(load, tide);
    auto start = iterate;
    AddChannels(start, load, tide);  // the Picard iterate: the warm start
    auto [solved, solveInfo] =
        Solve(fs, ft, &start, 0.0, op.options_.solve_outer_factor);
    info.solveIterations += solveInfo.iterations;
    x = std::move(solved);
    auto image = x;
    AddChannels(image, negated(load), negated(tide));
    // the outer residual: the warm start's, before the solve
    const auto change = solveInfo.initial;
    info.iterations = pass + 1;
    info.residual = change;
    // a non-finite image ends the iteration unconverged
    if (!std::isfinite(change) || !std::isfinite(changeOf(image, iterate))) break;
    if (change <= control.rtol && solveInfo.converged) {
      info.converged = true;
      break;
    }
    if (adaptive && !mixer && control.anderson_memory > 0 && pass > 0 &&
        change > control.anderson_rate * previous) {
      mixer.emplace(control.anderson_memory, parallel, collectives.sum);
    }
    previous = change;
    if (mixer) {
      mixer->UpdateInPlace(segments(iterate), constSegments(image));
      ++info.mixedPasses;
    } else {
      iterate = std::move(image);
    }
  }
  auto strain = StrainOf(x);
  if (solutions != nullptr) *solutions = std::move(x);
  return {std::move(strain), info};
}

std::pair<GridCoupling::SpectralTensor, Viscoelastic3D::StepInfo>
Viscoelastic3D::OpenStep::Close(const LoadCallable3D& load,
                                const LoadCallable3D& tide,
                                const IterationControl& control) const {
  if (op_->options_.solver == Options::StepSolver::ConjugateGradients) {
    const auto [sph, tor] = Forces(load, tide);
    auto [solved, solveInfo] = Solve(sph, tor);
    return {StrainOf(solved),
            StepInfo{solveInfo.iterations, solveInfo.residual, solveInfo.converged,
                     0, solveInfo.iterations}};
  }
  return Converge(
      [&](const GridCoupling::SpectralTensor& s) {
        auto solutions = Respond(s);
        AddChannels(solutions, load, tide);
        return StrainOf(solutions);
      },
      control);
}

double Viscoelastic3D::OpenStep::RelativeChange(
    const GridCoupling::SpectralTensor& next,
    const GridCoupling::SpectralTensor& now) const {
  const auto parallel = op_->options_.parallel;
  const auto n_ = Blocks(next);
  const auto o_ = Blocks(now);
  // The relative sup-norm change of the coefficients, compared on
  // squared magnitudes (maxima: the same whatever the thread count).
  // The maxima pass over NaNs, so a diverged pass is caught apart.
  auto scale = 0.0, diff = 0.0;
  auto finite = true;
  for (auto b = std::size_t{0}; b < n_.size(); ++b) {
    const auto& n = *n_[b];
    const auto& o = *o_[b];
    const auto size = n.size();
#pragma omp parallel for schedule(static) reduction(max : scale, diff) \
    reduction(&& : finite) if (parallel)
    for (auto i = std::size_t{0}; i < size; ++i) {
      finite = finite && std::isfinite(n[i].real()) && std::isfinite(n[i].imag());
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
  if (!finite) return std::numeric_limits<double>::quiet_NaN();
  return scale > 0.0 ? std::sqrt(diff) / std::sqrt(scale) : 0.0;
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
    detail::ForEach(op.options_.parallel, o.size(),
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
  // Into the iterate Converge last spent, when there is one.
  auto spectral = std::move(spare_);
  StrainOf(solutions, spectral);
  return spectral;
}

void Viscoelastic3D::OpenStep::StrainOf(
    const Solutions& solutions, GridCoupling::SpectralTensor& spectral) const {
  const auto& op = *op_;
  auto& flavour = *flavour_;
  if (spectral.r.empty()) {
    spectral = op.ZeroSpectralFor();
  } else {
    for (auto* block : Blocks(spectral)) {
      auto& b = *block;
      detail::ForEach(op.options_.parallel, b.size(),
                      [&](std::size_t i) { b[i] = 0.0; });
    }
  }
  const auto spheroidal = [&](int l) -> const DegreeOperator& {
    return op.FlavourSpheroidal(flavour, l);
  };
  op.AddStrainCoefficients(spectral, solutions, spheroidal,
                           [&](int l) -> const ToroidalOperator& {
                             return op.FlavourToroidal(flavour, l);
                           });
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
    detail::ForEach(op.options_.parallel, nodes, [&](std::size_t node) {
      const auto base = coupling.FieldOffset(static_cast<int>(node));
      const auto size = coupling.FieldSizeOf(static_cast<int>(node));
      for (auto x = std::size_t{0}; x < size; ++x) {
        const auto at = base + x;
        const auto muAt = mu[node] * op.ShearFactor(at);
        for (auto c = 0; c < 2; ++c) {
          memory.c[c][at] += add.c[c][at];
          force.c[c][at] += muAt * add.c[c][at];
        }
        memory.r[at] += add.r[at];
        force.r[at] += muAt * add.r[at];
      }
    });
  }
  const auto analysed = coupling.Analyse(force);
  const auto in = Blocks(analysed);
  const auto out = Blocks(fixed_);
  for (auto b = std::size_t{0}; b < out.size(); ++b) {
    auto& o = *out[b];
    const auto& f = *in[b];
    detail::ForEach(op.options_.parallel, o.size(),
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

    // A diverged (non-finite) pass ends the iteration unconverged, on the
    // last finite iterate.
    info.iterations = iteration + 1;
    info.residual = RelativeChange(sNext, s);
    if (!std::isfinite(info.residual)) break;
    if (info.residual <= control.rtol) {
      info.converged = true;
      s = std::move(sNext);
      break;
    }
    if (control.damping != 1.0) {
      for (auto b = std::size_t{0}; b < next.size(); ++b) {
        auto& n = *next[b];
        const auto& o = *now[b];
        detail::ForEach(parallel, n.size(), [&](std::size_t i) {
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
  if (options_.solver == Options::StepSolver::ConjugateGradients) {
    const auto [sph, tor] = step.Forces(load, tide);
    auto [solved, solveInfo] = step.Solve(sph, tor);
    auto strain = step.StrainOf(solved);
    const auto info = StepInfo{solveInfo.iterations, solveInfo.residual,
                               solveInfo.converged, 0, solveInfo.iterations};
    return {std::move(step).Finish(std::move(strain)), std::move(solved), info};
  }
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

double Viscoelastic3D::InertiaMoment(const Solutions& solutions) const {
  const auto found = solutions.spheroidal.find(0);
  if (found == solutions.spheroidal.end()) return 0.0;
  const auto& x = found->second;
  auto column = std::vector<double>(x.size());
  for (auto i = std::size_t{0}; i < x.size(); ++i) column[i] = x[i].real();
  Spheroidal(0).MaskToShare(column);
  auto moment = 0.0;
  {
    const auto lock = LockCaches();
    moment = reference_.InertiaMoment(column, 1)[0];
  }
  if (options_.collectives.Distributed()) {
    options_.collectives.sum(std::span(&moment, 1));
  }
  return moment;
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
