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
#include "sl3dpp/detail/ParallelFor.hpp"

namespace sl3dpp {

namespace {

// A spectral tensor's coefficients as real segments for the mixer: the
// spin -2, spin -1 and spin 0 blocks in turn, each complex value as its
// real-imaginary pair (the array layout std::complex guarantees).
std::array<std::span<double>, 3> RealSegments(GridCoupling::SpectralTensor& t) {
  auto pairs = [](std::vector<std::complex<double>>& v) {
    return std::span<double>(reinterpret_cast<double*>(v.data()), 2 * v.size());
  };
  return {pairs(t.c[0]), pairs(t.c[1]), pairs(t.r)};
}

std::array<std::span<const double>, 3> RealSegments(
    const GridCoupling::SpectralTensor& t) {
  auto pairs = [](const std::vector<std::complex<double>>& v) {
    return std::span<const double>(reinterpret_cast<const double*>(v.data()),
                                   2 * v.size());
  };
  return {pairs(t.c[0]), pairs(t.c[1]), pairs(t.r)};
}

// The three coefficient blocks of a spectral tensor, for loops that treat
// them alike.
std::array<std::vector<std::complex<double>>*, 3> Blocks(
    GridCoupling::SpectralTensor& t) {
  return {&t.c[0], &t.c[1], &t.r};
}

std::array<const std::vector<std::complex<double>>*, 3> Blocks(
    const GridCoupling::SpectralTensor& t) {
  return {&t.c[0], &t.c[1], &t.r};
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
    for (std::size_t i = 0; i < size; ++i) same = same && u[i] == v[i];
  }
  return same;
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
      coupling_(*rheology_, lmax, grid,
                {.eps = options.eps,
                 .truncation = options.truncation,
                 .oversampling = options.oversampling,
                 .parallel = options.parallel,
                 .matrix_kernel = options.matrix_kernel,
                 .memory_points = options.memory_points}),
      reference_(*rheology_, lmax,
                 {.eps = options.eps,
                  .truncation = options.truncation,
                  .memory_points = options.memory_points}) {
  // Sample exp(-delta) once and scale every branch's inverse times; the
  // reference profile is the unperturbed radial rheology itself, at the
  // memory points as the 1D reference holds it.
  const auto& sampling = coupling_.Grid();
  const auto fieldSize = sampling.FieldSize();
  const auto np = coupling_.Points().count;
  const auto nodes = static_cast<std::size_t>(coupling_.NodeCount());
  std::vector<double> factor(nodes * fieldSize);
  const auto thetas = sampling.CoLatitudes();
  const auto phis = sampling.Longitudes();
  for (std::size_t node = 0; node < nodes; ++node) {
    const auto r = coupling_.RadiusOfNode(static_cast<int>(node));
    std::size_t x = 0;
    for (auto theta : thetas) {
      for (auto phi : phis) {
        factor[node * fieldSize + x++] = std::exp(-delta(r, theta, phi));
      }
    }
  }
  for (auto k = std::size_t{0}; k < rheology_->BranchCount(); ++k) {
    const auto pointMu = reference_.PointMu(k);
    const auto pointItau = reference_.PointItau(k);
    auto itau = std::vector<double>(nodes * fieldSize);
    auto mu = std::vector<double>(nodes);
    auto itauRef = std::vector<double>(nodes);
    for (auto node = std::size_t{0}; node < nodes; ++node) {
      const auto e = coupling_.ElementOfNode(static_cast<int>(node));
      const auto p = coupling_.NodeInElementOf(static_cast<int>(node));
      const auto at = static_cast<std::size_t>(e) * np + p;
      mu[node] = pointMu[at];
      itauRef[node] = pointItau[at];
      for (auto x = std::size_t{0}; x < fieldSize; ++x) {
        itau[node * fieldSize + x] =
            pointItau[at] * factor[node * fieldSize + x];
      }
    }
    itau_.push_back(std::move(itau));
    muBranch_.push_back(std::move(mu));
    itauRef_.push_back(std::move(itauRef));
  }
}

Viscoelastic3D::~Viscoelastic3D() = default;
Viscoelastic3D::Viscoelastic3D(Viscoelastic3D&&) noexcept = default;
Viscoelastic3D& Viscoelastic3D::operator=(Viscoelastic3D&&) noexcept = default;

Viscoelastic3D::State Viscoelastic3D::ZeroState() const {
  State state;
  for (std::size_t k = 0; k < rheology_->BranchCount(); ++k) {
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
                                .point_shear = reference_.ElasticPointShear()}))
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
                                .point_shear = reference_.ElasticPointShear()}))
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

GridCoupling::SpectralTensor Viscoelastic3D::StrainCoefficients(
    const Solutions& solutions,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal) const {
  auto spectral = coupling_.ZeroSpectral();
  AddStrainCoefficients(spectral, solutions, spheroidal, toroidal);
  return spectral;
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
  std::vector<Task> tasks;
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
  });
}

Viscoelastic3D::Solutions Viscoelastic3D::SolveAll(
    const GridCoupling::SpectralTensor& weighted,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal) const {
  auto sphForces = coupling_.SpheroidalForce(weighted, spheroidal);
  auto torForces = coupling_.ToroidalForce(weighted, toroidal);

  // The degree tasks, spheroidal and toroidal, with their operators
  // resolved here, serially; each task then writes only its own solution.
  struct Task {
    int l = 0;
    const std::vector<Complex>* b = nullptr;
    const DegreeOperator* sph = nullptr;    // a spheroidal task
    const ToroidalOperator* tor = nullptr;  // a toroidal task
    std::vector<Complex> solution;
  };
  std::vector<Task> tasks;
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

  // The complex columns solved as real pairs.
  auto solve = [](const auto& op, const std::vector<Complex>& b, int columns) {
    const auto ndof = op.DofCount();
    std::vector<double> rhs(static_cast<std::size_t>(ndof) * 2 * columns);
    for (auto c = 0; c < columns; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        rhs[static_cast<std::size_t>(c) * ndof + d] =
            b[static_cast<std::size_t>(c) * ndof + d].real();
        rhs[static_cast<std::size_t>(columns + c) * ndof + d] =
            b[static_cast<std::size_t>(c) * ndof + d].imag();
      }
    }
    auto x = op.Solve(rhs, 2 * columns);
    std::vector<Complex> solution(static_cast<std::size_t>(ndof) * columns);
    for (auto c = 0; c < columns; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        solution[static_cast<std::size_t>(c) * ndof + d] = {
            x[static_cast<std::size_t>(c) * ndof + d],
            x[static_cast<std::size_t>(columns + c) * ndof + d]};
      }
    }
    return solution;
  };

  detail::ParallelFor(
      static_cast<int>(tasks.size()), options_.parallel, [&](int i) {
        auto& task = tasks[i];
        const auto l = task.l;
        const auto columns = l + 1;
        task.solution = task.tor != nullptr
                            ? solve(*task.tor, *task.b, columns)
                            : solve(*task.sph, *task.b, columns);
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
  const auto fieldSize = coupling_.Grid().FieldSize();
  const auto nodes = static_cast<std::size_t>(coupling_.NodeCount());
  for (auto k = std::size_t{0}; k < state.m.size(); ++k) {
    const auto& m = state.m[k];
    detail::ParallelElements(nodes, options_.parallel, [&](std::size_t node) {
      const auto mu = muBranch_[k][node];
      if (mu == 0.0) return;
      for (auto i = node * fieldSize; i < (node + 1) * fieldSize; ++i) {
        combined.c[0][i] += mu * m.c[0][i];
        combined.c[1][i] += mu * m.c[1][i];
        combined.r[i] += mu * m.r[i];
      }
    });
  }
  const auto weighted = coupling_.Analyse(combined);
  const auto spheroidal = [&](int l) -> const DegreeOperator& {
    return Spheroidal(l);
  };
  auto solutions =
      SolveAll(weighted, spheroidal,
               [&](int l) -> const ToroidalOperator& { return Toroidal(l); });
  AddUnitChannels(solutions, state.t, load, tide, spheroidal, unrelaxedUnits_);
  return solutions;
}

void Viscoelastic3D::AddUnitChannels(
    Solutions& solutions, double t, const LoadCallable3D& load,
    const LoadCallable3D& tide,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    std::map<int, std::vector<double>>& units) const {
  if (!load && !tide) return;
  for (auto& [l, columns_] : solutions.spheroidal) {
    // A plain reference: clang does not yet allow capturing a structured
    // binding in a lambda under OpenMP.
    auto& x = columns_;
    const auto& system = spheroidal(l);
    const auto ndof = system.DofCount();
    const auto& responses = [&]() -> const std::vector<double>& {
      const auto lock = LockCaches();
      auto found = units.find(l);
      if (found == units.end()) {
        std::vector<double> unit(static_cast<std::size_t>(ndof) * 2, 0.0);
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
                                                : nullptr}))
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
                                                : nullptr}))
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

Viscoelastic3D::OpenStep Viscoelastic3D::StartStep(
    const State& state, double dt, Scheme scheme, const LoadCallable3D& load,
    const LoadCallable3D& tide) const {
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "the 3D step is implicit: ExponentialTrapezoid or BackwardEuler");
  }
  const auto fieldSize = coupling_.Grid().FieldSize();
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
  if (state.strainCoefficients.has_value()) {
    step.s0_ = *state.strainCoefficients;
  } else if (state.strain.has_value()) {
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
  step.beta_.resize(K);
  step.weight_.assign(nodes * fieldSize, 0.0);
  step.grid_ = coupling_.ZeroTensor();
  auto& fixed = step.grid_;
  for (std::size_t k = 0; k < K; ++k) {
    step.memory_.push_back(coupling_.ZeroTensor());
    step.beta_[k].assign(nodes * fieldSize, 0.0);
    detail::ParallelElements(nodes, options_.parallel, [&](std::size_t node) {
      const auto itauRef = itauRef_[k][node];
      const auto refBeta = scheme == Scheme::BackwardEuler
                               ? 1.0 - BackwardEulerWeight(dt * itauRef)
                               : ExponentialTrapezoidWeights(dt * itauRef).beta;
      const auto mu = muBranch_[k][node];
      for (std::size_t x = 0; x < fieldSize; ++x) {
        const auto h = dt * itau_[k][node * fieldSize + x];
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
        const auto at = node * fieldSize + x;
        step.beta_[k][at] = b_w;
        auto& memory = step.memory_[k];
        for (auto c = 0; c < 2; ++c) {
          memory.c[c][at] = e_w * state.m[k].c[c][at] + a_w * d0.c[c][at];
        }
        memory.r[at] = e_w * state.m[k].r[at] + a_w * d0.r[at];
        if (mu == 0.0) continue;
        step.weight_[at] += mu * (b_w - refBeta);
        for (auto c = 0; c < 2; ++c) fixed.c[c][at] += mu * memory.c[c][at];
        fixed.r[at] += mu * memory.r[at];
      }
    });
  }
  step.fixed_ = coupling_.Analyse(fixed);

  // The first pass's force, from the grid strain already in hand, so that
  // the first Respond (of s0) skips its synthesis.
  detail::ParallelElements(nodes * fieldSize, options_.parallel,
                           [&](std::size_t i) {
                             fixed.c[0][i] = step.weight_[i] * d0.c[0][i];
                             fixed.c[1][i] = step.weight_[i] * d0.c[1][i];
                             fixed.r[i] = step.weight_[i] * d0.r[i];
                           });
  coupling_.Analyse(fixed, step.analysed_);
  const auto in = Blocks(std::as_const(step.fixed_));
  const auto out = Blocks(step.analysed_);
  for (auto b = std::size_t{0}; b < out.size(); ++b) {
    auto& o = *out[b];
    const auto& f = *in[b];
    detail::ParallelElements(o.size(), options_.parallel,
                             [&](std::size_t i) { o[i] += f[i]; });
  }
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
  const auto formed = std::exchange(startForce_, false) &&
                      SameValues(strain, s0_, op.options_.parallel);
  if (!formed) {
    coupling.Synthesise(strain, grid_);
    coupling.Multiply(grid_, weight_);
    coupling.Analyse(grid_, analysed_);
    const auto fixed = Blocks(fixed_);
    const auto out = Blocks(analysed_);
    for (std::size_t b = 0; b < out.size(); ++b) {
      auto& o = *out[b];
      const auto& f = *fixed[b];
      detail::ParallelElements(o.size(), op.options_.parallel,
                               [&](std::size_t i) { o[i] += f[i]; });
    }
  }
  return op.SolveAll(
      analysed_,
      [&](int l) -> const DegreeOperator& {
        return op.FlavourSpheroidal(flavour, l);
      },
      [&](int l) -> const ToroidalOperator& {
        return op.FlavourToroidal(flavour, l);
      });
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
    spectral = op.coupling_.ZeroSpectral();
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
  std::vector<double> column(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) column[i] = x[i].real();
  const auto lock = op_->LockCaches();
  return op_->reference_.InertiaMoment(column, 1)[0];
}

const LoveNumbers& Viscoelastic3D::OpenStep::EffectiveLove() const {
  const auto lock = op_->LockCaches();
  return op_->reference_.EffectiveLoveNumbers(scheme_, dt_);
}

Viscoelastic3D::State Viscoelastic3D::OpenStep::Finish(
    const GridCoupling::SpectralTensor& strain) const& {
  auto d = op_->coupling_.Synthesise(strain);
  State next;
  next.t = t_;
  for (std::size_t k = 0; k < memory_.size(); ++k) {
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
  for (std::size_t k = 0; k < memory_.size(); ++k) {
    AdvanceMemory(memory_[k], beta_[k], grid_, op_->options_.parallel);
    next.m.push_back(std::move(memory_[k]));
  }
  next.strain = std::move(grid_);
  next.strainCoefficients = std::move(strain);
  return next;
}

std::pair<GridCoupling::SpectralTensor, Viscoelastic3D::StepInfo>
Viscoelastic3D::OpenStep::Converge(const Pass& pass,
                                   const IterationControl& control) const {
  const auto parallel = op_->options_.parallel;
  auto s = s0_;
  StepInfo info;
  info.converged = false;
  // Picard until a pass contracts too slowly, then Anderson for the rest
  // of the step (IterationControl). Not adaptive: Anderson from the first
  // pass (the iteration before 17 Oct, chunk 9).
  const auto adaptive = control.anderson_rate > 0.0;
  auto mixer = std::optional<detail::AndersonMixer>{};
  if (control.anderson_memory > 0 && !adaptive) {
    mixer.emplace(control.anderson_memory, parallel);
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
    double scale = 0.0, diff = 0.0;
    auto finite = true;
    for (std::size_t b = 0; b < next.size(); ++b) {
      const auto& n = *next[b];
      const auto& o = *now[b];
      const auto size = n.size();
#pragma omp parallel for schedule(static) reduction(max : scale, diff) \
    reduction(&& : finite) if (parallel)
      for (std::size_t i = 0; i < size; ++i) {
        finite =
            finite && std::isfinite(n[i].real()) && std::isfinite(n[i].imag());
        scale = std::max(scale, std::norm(n[i]));
        diff = std::max(diff, std::norm(n[i] - o[i]));
      }
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
      for (std::size_t b = 0; b < next.size(); ++b) {
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
      mixer.emplace(control.anderson_memory, parallel);
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
  std::map<int, std::vector<Complex>> out;
  for (const auto& [l, x] : solutions.spheroidal) {
    const auto& op = Spheroidal(l);
    const auto columns = l + 1;
    const auto dof = op.SurfaceDof(component);
    std::vector<Complex> values(columns, 0.0);
    if (dof >= 0) {
      for (auto c = 0; c < columns; ++c) {
        values[c] = x[static_cast<std::size_t>(c) * op.DofCount() + dof];
      }
    }
    out.emplace(l, std::move(values));
  }
  return out;
}

}  // namespace sl3dpp
