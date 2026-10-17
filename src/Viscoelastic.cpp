#include "sl3dpp/Viscoelastic.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>

#include "sl3dpp/detail/ParallelFor.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

namespace {

// A channel's columns at a degree and time, from the caller's callable
// (called serially: it need not be thread-safe), checked for shape.
std::optional<std::vector<double>> FetchColumns(const LoadCallable& channel,
                                                int l, double t, int ncols) {
  if (!channel) return std::nullopt;
  auto columns = channel(l, t);
  if (columns && static_cast<int>(columns->size()) != ncols) {
    throw std::invalid_argument(
        std::format("a degree-{} channel must supply {} columns, got {}", l,
                    ncols, columns->size()));
  }
  return columns;
}

// Adds a channel's columns through its unit response, column by column.
void AddColumns(std::vector<double>& x, const std::vector<double>& units,
                int unit, int ndof, const std::vector<double>& columns,
                int ncols) {
  for (int c = 0; c < ncols; ++c) {
    const double amplitude = columns[c];
    if (amplitude == 0.0) continue;
    for (int d = 0; d < ndof; ++d) {
      x[static_cast<std::size_t>(c) * ndof + d] +=
          units[static_cast<std::size_t>(unit) * ndof + d] * amplitude;
    }
  }
}

}  // namespace

// The per-degree workspace and the flavour struct live in
// detail/ViscoelasticDegree.hpp, shared with the adjoint sweeps.


ViscoelasticOperator::ViscoelasticOperator(const Rheology& rheology, int lmax,
                                           Options options)
    : rheology_(std::make_shared<const Rheology>(rheology)),
      lmax_(lmax),
      options_(options) {
  if (lmax < 0) throw std::invalid_argument("lmax must be non-negative");
}

ViscoelasticOperator::~ViscoelasticOperator() = default;
ViscoelasticOperator::ViscoelasticOperator(ViscoelasticOperator&&) noexcept =
    default;
ViscoelasticOperator& ViscoelasticOperator::operator=(
    ViscoelasticOperator&&) noexcept = default;

ViscoelasticOperator::Degree& ViscoelasticOperator::Deg(int l) const {
  auto found = degrees_.find(l);
  if (found == degrees_.end()) {
    found =
        degrees_.emplace(l, std::make_unique<Degree>(*rheology_, l, options_))
            .first;
  }
  return *found->second;
}

const std::vector<ViscoelasticOperator::BranchWeights>&
ViscoelasticOperator::SchemeWeightsOf(Scheme scheme, double dt) const {
  const auto key = std::pair(static_cast<int>(scheme), dt);
  auto found = weights_.find(key);
  if (found == weights_.end()) {
    std::vector<BranchWeights> out;
    for (std::size_t k = 0; k < rheology_->BranchCount(); ++k) {
      const auto& itau = rheology_->BranchAt(k).itau;
      BranchWeights w;
      w.e.resize(itau.size());
      if (scheme == Scheme::BackwardEuler) {
        for (std::size_t i = 0; i < itau.size(); ++i) {
          w.e[i] = BackwardEulerWeight(dt * itau[i]);
        }
      } else {
        w.alpha.resize(itau.size());
        w.beta.resize(itau.size());
        for (std::size_t i = 0; i < itau.size(); ++i) {
          const auto tw = ExponentialTrapezoidWeights(dt * itau[i]);
          w.e[i] = tw.e;
          w.alpha[i] = tw.alpha;
          w.beta[i] = tw.beta;
        }
      }
      out.push_back(std::move(w));
    }
    found = weights_.emplace(key, std::move(out)).first;
  }
  return found->second;
}

const std::vector<ViscoelasticOperator::BranchWeights>&
ViscoelasticOperator::SchemeWeightDerivativesOf(Scheme scheme,
                                                double dt) const {
  const auto key = std::pair(static_cast<int>(scheme), dt);
  auto found = weightDerivatives_.find(key);
  if (found == weightDerivatives_.end()) {
    std::vector<BranchWeights> out;
    for (std::size_t k = 0; k < rheology_->BranchCount(); ++k) {
      const auto& itau = rheology_->BranchAt(k).itau;
      BranchWeights w;
      w.e.resize(itau.size());
      if (scheme == Scheme::BackwardEuler) {
        for (std::size_t i = 0; i < itau.size(); ++i) {
          w.e[i] = BackwardEulerWeightDerivative(dt * itau[i]);
        }
      } else {
        w.alpha.resize(itau.size());
        w.beta.resize(itau.size());
        for (std::size_t i = 0; i < itau.size(); ++i) {
          const auto tw = ExponentialTrapezoidWeightDerivatives(dt * itau[i]);
          w.e[i] = tw.e;
          w.alpha[i] = tw.alpha;
          w.beta[i] = tw.beta;
        }
      }
      out.push_back(std::move(w));
    }
    found = weightDerivatives_.emplace(key, std::move(out)).first;
  }
  return found->second;
}

Material ViscoelasticOperator::EffectiveMaterial(Scheme scheme,
                                                 double dt) const {
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument("no effective operator for ExponentialEuler");
  }
  const auto& weights = SchemeWeightsOf(scheme, dt);
  std::vector<std::vector<double>> w;
  for (const auto& branch : weights) {
    if (scheme == Scheme::BackwardEuler) {
      w.push_back(branch.e);
    } else {
      std::vector<double> phi1(branch.beta.size());
      for (std::size_t i = 0; i < phi1.size(); ++i) {
        phi1[i] = 1.0 - branch.beta[i];
      }
      w.push_back(std::move(phi1));
    }
  }
  return BaseMaterial().WithModuli(rheology_->EffectiveModuli(w)).value();
}

ViscoelasticOperator::Flavour& ViscoelasticOperator::FlavourOf(
    Scheme scheme, double dt) const {
  const auto key = std::pair(static_cast<int>(scheme), dt);
  for (auto it = flavours_.begin(); it != flavours_.end(); ++it) {
    if (it->first == key) {
      flavours_.splice(flavours_.begin(), flavours_, it);
      return *flavours_.front().second;
    }
  }
  constexpr std::size_t kFlavourCache = 4;
  while (flavours_.size() >= kFlavourCache) flavours_.pop_back();
  auto flavour = std::make_unique<Flavour>();
  flavour->material = std::make_unique<Material>(EffectiveMaterial(scheme, dt));
  flavours_.emplace_front(key, std::move(flavour));
  return *flavours_.front().second;
}

const DegreeOperator& ViscoelasticOperator::EffectiveSystem(Flavour& flavour,
                                                            int l) const {
  auto found = flavour.systems.find(l);
  if (found == flavour.systems.end()) {
    found = flavour.systems
                .emplace(l, DegreeOperator(*flavour.material, l,
                                           {.eps = options_.eps,
                                            .truncation = options_.truncation}))
                .first;
  }
  return found->second;
}

const std::vector<double>& ViscoelasticOperator::UnitResponses(Flavour* flavour,
                                                               int l) const {
  auto& deg = Deg(l);
  auto& cache = flavour == nullptr ? unrelaxedUnits_ : flavour->units;
  auto found = cache.find(l);
  if (found == cache.end()) {
    const auto& system =
        flavour == nullptr ? deg.system : EffectiveSystem(*flavour, l);
    std::vector<double> rhs(static_cast<std::size_t>(deg.ndof) * 2, 0.0);
    std::ranges::copy(deg.loadVec, rhs.begin());
    std::ranges::copy(deg.tideVec, rhs.begin() + deg.ndof);
    found = cache.emplace(l, system.Solve(rhs, 2)).first;
  }
  return found->second;
}

const std::vector<double>& ViscoelasticOperator::SplitUnitResponses(
    Flavour* flavour, int l) const {
  auto& deg = Deg(l);
  auto& cache =
      flavour == nullptr ? unrelaxedSplitUnits_ : flavour->splitUnits;
  auto found = cache.find(l);
  if (found == cache.end()) {
    const auto& system =
        flavour == nullptr ? deg.system : EffectiveSystem(*flavour, l);
    const auto force = deg.system.Load(LoadPart::Force);
    const auto potential = deg.system.Load(LoadPart::Potential);
    std::vector<double> rhs(static_cast<std::size_t>(deg.ndof) * 2, 0.0);
    std::ranges::copy(force, rhs.begin());
    std::ranges::copy(potential, rhs.begin() + deg.ndof);
    found = cache.emplace(l, system.Solve(rhs, 2)).first;
  }
  return found->second;
}

void ViscoelasticOperator::AddChannels(std::vector<double>& x, int l,
                                       Flavour* flavour, double t,
                                       const LoadCallable& load,
                                       const LoadCallable& tide,
                                       int ncols) const {
  if (!load && !tide) return;
  const auto& deg = Deg(l);
  const auto& units = UnitResponses(flavour, l);
  if (auto columns = FetchColumns(load, l, t, ncols)) {
    AddColumns(x, units, 0, deg.ndof, *columns, ncols);
  }
  if (auto columns = FetchColumns(tide, l, t, ncols)) {
    AddColumns(x, units, 1, deg.ndof, *columns, ncols);
  }
}

ViscoelasticState ViscoelasticOperator::ZeroState(
    const std::function<int(int)>& ncols) const {
  ViscoelasticState state;
  const auto K = rheology_->BranchCount();
  for (int l = 0; l <= lmax_; ++l) {
    const auto& deg = Deg(l);
    state.m[l].assign(K * 3 * deg.NodalSize(ncols(l)), 0.0);
  }
  return state;
}

std::map<int, std::vector<double>> ViscoelasticOperator::SolveElastic(
    const ViscoelasticState& state, const LoadCallable& load,
    const LoadCallable& tide) const {
  // Degree tasks: the workspaces, unit responses and channel columns are
  // resolved here, serially (the caches fill lazily, and the callables
  // need not be thread-safe); each task writes only its own solution.
  struct Task {
    int l = 0;
    int ncols = 0;
    const std::vector<double>* m = nullptr;
    const Degree* deg = nullptr;
    const std::vector<double>* units = nullptr;
    std::optional<std::vector<double>> load, tide;
    std::vector<double> x;
  };
  const auto K = rheology_->BranchCount();
  std::vector<Task> tasks;
  for (const auto& [l, m] : state.m) {
    Task task;
    task.l = l;
    task.m = &m;
    task.deg = &Deg(l);
    task.ncols =
        static_cast<int>(m.size() / (K * 3 * task.deg->ne * task.deg->ngll));
    if (load || tide) {
      task.units = &UnitResponses(nullptr, l);
      task.load = FetchColumns(load, l, state.t, task.ncols);
      task.tide = FetchColumns(tide, l, state.t, task.ncols);
    }
    tasks.push_back(std::move(task));
  }
  const auto count = static_cast<int>(tasks.size());
  detail::ParallelFor(count, options_.parallel, [&](int i) {
    auto& task = tasks[count - 1 - i];  // the largest degrees first
    const auto& deg = *task.deg;
    task.x = deg.system.Solve(deg.MemoryForce(*task.m, task.ncols), task.ncols);
    if (task.load) {
      AddColumns(task.x, *task.units, 0, deg.ndof, *task.load, task.ncols);
    }
    if (task.tide) {
      AddColumns(task.x, *task.units, 1, deg.ndof, *task.tide, task.ncols);
    }
  });
  std::map<int, std::vector<double>> solutions;
  for (auto& task : tasks) solutions.emplace(task.l, std::move(task.x));
  return solutions;
}

std::map<int, std::vector<double>> ViscoelasticOperator::Strain(
    const std::map<int, std::vector<double>>& solutions) const {
  // One task per degree, the workspaces resolved first.
  std::vector<std::pair<int, const std::vector<double>*>> inputs;
  std::vector<const Degree*> degs;
  for (const auto& [l, x] : solutions) {
    inputs.emplace_back(l, &x);
    degs.push_back(&Deg(l));
  }
  std::vector<std::vector<double>> strains(inputs.size());
  const auto count = static_cast<int>(inputs.size());
  detail::ParallelFor(count, options_.parallel, [&](int i) {
    const auto j = static_cast<std::size_t>(count - 1 - i);
    const auto& x = *inputs[j].second;
    strains[j] = degs[j]->Strain(x, static_cast<int>(x.size() / degs[j]->ndof));
  });
  std::map<int, std::vector<double>> out;
  for (std::size_t j = 0; j < inputs.size(); ++j) {
    out.emplace(inputs[j].first, std::move(strains[j]));
  }
  return out;
}

std::map<int, std::vector<double>> ViscoelasticOperator::Surface(
    const std::map<int, std::vector<double>>& solutions,
    Component component) const {
  std::map<int, std::vector<double>> out;
  for (const auto& [l, x] : solutions) {
    const auto& deg = Deg(l);
    const int ncols = static_cast<int>(x.size() / deg.ndof);
    const int dof = deg.system.SurfaceDof(component);
    std::vector<double> values(ncols, 0.0);
    if (dof >= 0) {
      for (int c = 0; c < ncols; ++c) {
        values[c] = x[static_cast<std::size_t>(c) * deg.ndof + dof];
      }
    }
    out.emplace(l, std::move(values));
  }
  return out;
}

std::map<int, std::vector<double>> ViscoelasticOperator::StartStrain(
    const ViscoelasticState& state, const LoadCallable& load,
    const LoadCallable& tide) const {
  if (state.HasStrain()) return state.strain;
  return Strain(SolveElastic(state, load, tide));
}

const std::map<int, std::vector<double>>& ViscoelasticOperator::PrimeStrain(
    ViscoelasticState& state, const LoadCallable& load,
    const LoadCallable& tide) const {
  if (!state.HasStrain()) {
    state.strain = Strain(SolveElastic(state, load, tide));
  }
  return state.strain;
}

std::vector<double> ViscoelasticOperator::InertiaMoment(
    std::span<const double> solution0, int ncols) const {
  const auto& deg = Deg(0);
  std::vector<double> out(ncols, 0.0);
  for (int c = 0; c < ncols; ++c) {
    double sum = 0.0;
    for (int d = 0; d < deg.ndof; ++d) {
      sum += deg.momentRow[d] *
             solution0[static_cast<std::size_t>(c) * deg.ndof + d];
    }
    out[c] = sum;
  }
  return out;
}

// -- the steps
// -----------------------------------------------------------------------

ViscoelasticStep::ViscoelasticStep(const ViscoelasticOperator& op,
                                   const ViscoelasticState& state, double dt,
                                   Scheme scheme, const LoadCallable& load,
                                   const LoadCallable& tide)
    : op_(&op), scheme_(scheme), dt_(dt), t_(state.t + dt) {
  const auto& weights = op.SchemeWeightsOf(scheme, dt);
  std::map<int, std::vector<double>> dStart;
  if (scheme == Scheme::ExponentialTrapezoid) {
    dStart = op.StartStrain(state, load, tide);
  }
  const auto K = op.GetRheology().BranchCount();
  // One task per degree, the workspaces resolved first: the decayed
  // memory, pointwise.
  std::vector<std::pair<int, const std::vector<double>*>> inputs;
  std::vector<const ViscoelasticOperator::Degree*> degs;
  for (const auto& [l, m] : state.m) {
    const auto& deg = op.Deg(l);
    ncols_[l] = static_cast<int>(m.size() / (K * 3 * deg.ne * deg.ngll));
    inputs.emplace_back(l, &m);
    degs.push_back(&deg);
  }
  std::vector<std::vector<double>> decayed(inputs.size());
  const auto count = static_cast<int>(inputs.size());
  detail::ParallelFor(count, op.options_.parallel, [&](int i) {
    const auto j = static_cast<std::size_t>(count - 1 - i);
    const auto l = inputs[j].first;
    const auto& m = *inputs[j].second;
    const auto& deg = *degs[j];
    const int ncols = ncols_.at(l);
    const std::size_t block = deg.NodalSize(ncols);
    std::vector<double> mem(m.size());
    for (std::size_t k = 0; k < K; ++k) {
      const auto& w = weights[k];
      for (int s = 0; s < 3; ++s) {
        for (int e = 0; e < deg.ne; ++e) {
          for (int q = 0; q < deg.ngll; ++q) {
            const std::size_t full =
                static_cast<std::size_t>(deg.e0 + e) * deg.ngll + q;
            const std::size_t node =
                (static_cast<std::size_t>(e) * deg.ngll + q) * ncols;
            const std::size_t at = (k * 3 + s) * block + node;
            for (int c = 0; c < ncols; ++c) {
              if (scheme == Scheme::BackwardEuler) {
                mem[at + c] = w.e[full] * m[at + c];
              } else {
                mem[at + c] =
                    w.e[full] * m[at + c] +
                    w.alpha[full] * dStart.at(l)[s * block + node + c];
              }
            }
          }
        }
      }
    }
    decayed[j] = std::move(mem);
  });
  for (std::size_t j = 0; j < inputs.size(); ++j) {
    mem_.emplace(inputs[j].first, std::move(decayed[j]));
  }
}

const std::map<int, std::vector<double>>& ViscoelasticStep::MemorySolutions()
    const {
  if (memorySolutions_.empty()) {
    auto& flavour = op_->FlavourOf(scheme_, dt_);
    // One task per degree, the workspaces and effective systems resolved
    // first: the memory force through the effective system.
    std::vector<std::pair<int, const std::vector<double>*>> inputs;
    std::vector<const ViscoelasticOperator::Degree*> degs;
    std::vector<const DegreeOperator*> systems;
    for (const auto& [l, mem] : mem_) {
      inputs.emplace_back(l, &mem);
      degs.push_back(&op_->Deg(l));
      systems.push_back(&op_->EffectiveSystem(flavour, l));
    }
    std::vector<std::vector<double>> solved(inputs.size());
    const auto count = static_cast<int>(inputs.size());
    detail::ParallelFor(count, op_->options_.parallel, [&](int i) {
      const auto j = static_cast<std::size_t>(count - 1 - i);
      const int ncols = ncols_.at(inputs[j].first);
      solved[j] = systems[j]->Solve(
          degs[j]->MemoryForce(*inputs[j].second, ncols), ncols);
    });
    for (std::size_t j = 0; j < inputs.size(); ++j) {
      memorySolutions_.emplace(inputs[j].first, std::move(solved[j]));
    }
  }
  return memorySolutions_;
}

std::map<int, std::vector<double>> ViscoelasticStep::MemorySurface(
    Component component) const {
  return op_->Surface(MemorySolutions(), component);
}

std::vector<double> ViscoelasticStep::InertiaMoment() const {
  const auto& solutions = MemorySolutions();
  auto found = solutions.find(0);
  if (found == solutions.end()) {
    throw std::invalid_argument("the state does not carry degree 0");
  }
  return op_->InertiaMoment(found->second, ncols_.at(0));
}

const LoveNumbers& ViscoelasticStep::EffectiveLove() const {
  return op_->EffectiveLoveNumbers(scheme_, dt_);
}

std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
ViscoelasticStep::Finish(const LoadCallable& load,
                         const LoadCallable& tide) const {
  const auto& op = *op_;
  const auto& weights = op.SchemeWeightsOf(scheme_, dt_);
  auto& flavour = op.FlavourOf(scheme_, dt_);
  const auto& memory = MemorySolutions();
  const auto K = op.GetRheology().BranchCount();

  // Degree tasks: the workspaces, unit responses and channel columns are
  // resolved here, serially (the caches fill lazily, and the callables
  // need not be thread-safe); each task writes only its own results.
  struct Task {
    int l = 0;
    int ncols = 0;
    const std::vector<double>* mem = nullptr;
    const ViscoelasticOperator::Degree* deg = nullptr;
    const std::vector<double>* units = nullptr;
    std::optional<std::vector<double>> load, tide;
    std::vector<double> x, d1, mNew;
  };
  std::vector<Task> tasks;
  for (const auto& [l, mem] : mem_) {
    Task task;
    task.l = l;
    task.mem = &mem;
    task.deg = &op.Deg(l);
    task.ncols = ncols_.at(l);
    if (load || tide) {
      task.units = &op.UnitResponses(&flavour, l);
      task.load = FetchColumns(load, l, t_, task.ncols);
      task.tide = FetchColumns(tide, l, t_, task.ncols);
    }
    tasks.push_back(std::move(task));
  }
  const auto count = static_cast<int>(tasks.size());
  detail::ParallelFor(count, op.options_.parallel, [&](int i) {
    auto& task = tasks[count - 1 - i];  // the largest degrees first
    const auto& deg = *task.deg;
    const int ncols = task.ncols;
    auto x = memory.at(task.l);
    if (task.load) AddColumns(x, *task.units, 0, deg.ndof, *task.load, ncols);
    if (task.tide) AddColumns(x, *task.units, 1, deg.ndof, *task.tide, ncols);
    auto d1 = deg.Strain(x, ncols);
    const std::size_t block = deg.NodalSize(ncols);
    std::vector<double> mNew = *task.mem;
    for (std::size_t k = 0; k < K; ++k) {
      const auto& w = weights[k];
      for (int s = 0; s < 3; ++s) {
        for (int e = 0; e < deg.ne; ++e) {
          for (int q = 0; q < deg.ngll; ++q) {
            const std::size_t full =
                static_cast<std::size_t>(deg.e0 + e) * deg.ngll + q;
            const double factor = scheme_ == Scheme::BackwardEuler
                                      ? 1.0 - w.e[full]
                                      : w.beta[full];
            const std::size_t node =
                (static_cast<std::size_t>(e) * deg.ngll + q) * ncols;
            const std::size_t at = (k * 3 + s) * block + node;
            for (int c = 0; c < ncols; ++c) {
              mNew[at + c] += factor * d1[s * block + node + c];
            }
          }
        }
      }
    }
    task.x = std::move(x);
    task.d1 = std::move(d1);
    task.mNew = std::move(mNew);
  });

  ViscoelasticState out;
  out.t = t_;
  std::map<int, std::vector<double>> solutions;
  for (auto& task : tasks) {
    out.m.emplace(task.l, std::move(task.mNew));
    out.strain.emplace(task.l, std::move(task.d1));
    solutions.emplace(task.l, std::move(task.x));
  }
  return {std::move(out), std::move(solutions)};
}

ViscoelasticStep ViscoelasticOperator::StartStep(
    const ViscoelasticState& state, double dt, Scheme scheme,
    const LoadCallable& load, const LoadCallable& tide) const {
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "a two-phase step is implicit: ExponentialTrapezoid or "
        "BackwardEuler");
  }
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  return ViscoelasticStep(*this, state, dt, scheme, load, tide);
}

std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
ViscoelasticOperator::Step(const ViscoelasticState& state, double dt,
                           Scheme scheme, const LoadCallable& load,
                           const LoadCallable& tide) const {
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  if (scheme == Scheme::ExponentialEuler) {
    auto dStart = StartStrain(state, load, tide);
    const auto& weights = SchemeWeightsOf(Scheme::ExponentialTrapezoid, dt);
    const auto K = rheology_->BranchCount();
    ViscoelasticState out;
    out.t = state.t + dt;
    for (const auto& [l, m] : state.m) {
      const auto& deg = Deg(l);
      const int ncols =
          static_cast<int>(m.size() / (K * 3 * deg.ne * deg.ngll));
      const std::size_t block = deg.NodalSize(ncols);
      std::vector<double> mNew(m.size());
      for (std::size_t k = 0; k < K; ++k) {
        const auto& w = weights[k];
        for (int s = 0; s < 3; ++s) {
          for (int e = 0; e < deg.ne; ++e) {
            for (int q = 0; q < deg.ngll; ++q) {
              const std::size_t full =
                  static_cast<std::size_t>(deg.e0 + e) * deg.ngll + q;
              const std::size_t node =
                  (static_cast<std::size_t>(e) * deg.ngll + q) * ncols;
              const std::size_t at = (k * 3 + s) * block + node;
              for (int c = 0; c < ncols; ++c) {
                mNew[at + c] =
                    w.e[full] * m[at + c] +
                    (1.0 - w.e[full]) * dStart[l][s * block + node + c];
              }
            }
          }
        }
      }
      out.m.emplace(l, std::move(mNew));
    }
    // the displacement is not consistent with the new state
    return {std::move(out), SolveElastic(state, load, tide)};
  }
  return ViscoelasticStep(*this, state, dt, scheme, load, tide)
      .Finish(load, tide);
}

std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
ViscoelasticOperator::StepBack(const ViscoelasticState& state, double dt,
                               Scheme scheme, const LoadCallable& load,
                               const LoadCallable& tide) const {
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "a step inverts for ExponentialTrapezoid or BackwardEuler");
  }
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  if (!state.HasStrain()) {
    throw std::invalid_argument(
        "the state must carry its strain cache; reconstruct from a "
        "recorded state or a completed step");
  }
  if (scheme == Scheme::ExponentialTrapezoid) {
    // the same update at -dt with the start and end strains swapped
    return ViscoelasticStep(*this, state, -dt, scheme, {}, {})
        .Finish(load, tide);
  }
  // backward Euler inverts explicitly: m^n = (1 + h) m^{n+1} - h d^{n+1}
  const auto K = rheology_->BranchCount();
  ViscoelasticState prev;
  prev.t = state.t - dt;
  for (const auto& [l, m] : state.m) {
    const auto& deg = Deg(l);
    const int ncols = static_cast<int>(m.size() / (K * 3 * deg.ne * deg.ngll));
    const std::size_t block = deg.NodalSize(ncols);
    const auto& d1 = state.strain.at(l);
    std::vector<double> mPrev(m.size());
    for (std::size_t k = 0; k < K; ++k) {
      for (int s = 0; s < 3; ++s) {
        for (int e = 0; e < deg.ne; ++e) {
          for (int q = 0; q < deg.ngll; ++q) {
            const double h =
                dt * deg.itau[k][static_cast<std::size_t>(e) * deg.ngll + q];
            const std::size_t node =
                (static_cast<std::size_t>(e) * deg.ngll + q) * ncols;
            const std::size_t at = (k * 3 + s) * block + node;
            for (int c = 0; c < ncols; ++c) {
              mPrev[at + c] =
                  (1.0 + h) * m[at + c] - h * d1[s * block + node + c];
            }
          }
        }
      }
    }
    prev.m.emplace(l, std::move(mPrev));
  }
  auto solutions = SolveElastic(prev, load, tide);
  prev.strain = Strain(solutions);
  return {std::move(prev), std::move(solutions)};
}

ViscoelasticOperator::AdaptiveResult ViscoelasticOperator::AdaptiveStep(
    ViscoelasticState& state, double dt, double rtol, const LoadCallable& load,
    const LoadCallable& tide) const {
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  const auto& dStart = PrimeStrain(state, load, tide);
  const auto& weights = SchemeWeightsOf(Scheme::ExponentialTrapezoid, dt);
  auto [newState, solutions] =
      Step(state, dt, Scheme::ExponentialTrapezoid, load, tide);

  const auto K = rheology_->BranchCount();
  double top = 0.0;
  for (const auto& [l, m] : newState.m) {
    for (double v : m) top = std::max(top, std::abs(v));
  }
  const double atol = top > 0.0 ? 1e-6 * rtol * top : 1e-300;

  double total = 0.0;
  std::size_t count = 0;
  for (const auto& [l, m] : state.m) {
    const auto& deg = Deg(l);
    const int ncols = static_cast<int>(m.size() / (K * 3 * deg.ne * deg.ngll));
    const std::size_t block = deg.NodalSize(ncols);
    const auto& mNew = newState.m.at(l);
    for (std::size_t k = 0; k < K; ++k) {
      const auto& w = weights[k];
      for (int s = 0; s < 3; ++s) {
        for (int e = 0; e < deg.ne; ++e) {
          for (int q = 0; q < deg.ngll; ++q) {
            const std::size_t full =
                static_cast<std::size_t>(deg.e0 + e) * deg.ngll + q;
            const std::size_t node =
                (static_cast<std::size_t>(e) * deg.ngll + q) * ncols;
            const std::size_t at = (k * 3 + s) * block + node;
            for (int c = 0; c < ncols; ++c) {
              const double companion =
                  w.e[full] * m[at + c] +
                  (1.0 - w.e[full]) * dStart.at(l)[s * block + node + c];
              const double scale =
                  atol + rtol * std::max(std::abs(m[at + c]),
                                         std::abs(mNew.at(at + c)));
              const double ratio = (mNew[at + c] - companion) / scale;
              total += ratio * ratio;
              ++count;
            }
          }
        }
      }
    }
  }
  const double err = count ? std::sqrt(total / count) : 0.0;
  constexpr double kSafety = 0.9, kMin = 0.2, kMax = 4.0;
  const double factor =
      err == 0.0 ? kMax : std::clamp(kSafety / std::sqrt(err), kMin, kMax);
  AdaptiveResult result;
  result.error = err;
  result.dtNext = dt * factor;
  if (err <= 1.0) {
    result.state = std::move(newState);
    result.solutions = std::move(solutions);
  }
  return result;
}

const LoveNumbers& ViscoelasticOperator::EffectiveLoveNumbers(Scheme scheme,
                                                              double dt) const {
  const auto key = std::pair(static_cast<int>(scheme), dt);
  auto found = effectiveLove_.find(key);
  if (found == effectiveLove_.end()) {
    const auto material = EffectiveMaterial(scheme, dt);
    found = effectiveLove_
                .emplace(key, ComputeLoveNumbers(
                                  material, lmax_,
                                  {.eps = options_.eps,
                                   .truncation = options_.truncation}))
                .first;
  }
  return found->second;
}

const LoveNumbers& ViscoelasticOperator::ElasticLoveNumbers() const {
  const auto key = std::pair(-1, 0.0);
  auto found = effectiveLove_.find(key);
  if (found == effectiveLove_.end()) {
    found = effectiveLove_
                .emplace(key, ComputeLoveNumbers(
                                  BaseMaterial(), lmax_,
                                  {.eps = options_.eps,
                                   .truncation = options_.truncation}))
                .first;
  }
  return found->second;
}

}  // namespace sl3dpp
