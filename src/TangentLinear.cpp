#include "sl3dpp/TangentLinear.hpp"

#include <format>
#include <optional>
#include <stdexcept>

#include "sl3dpp/StepRecipe.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/InteriorSources.hpp"
#include "sl3dpp/detail/StepStages.hpp"
#include "sl3dpp/detail/TimeGrid.hpp"
#include "sl3dpp/detail/TrajectoryCursor.hpp"
#include "sl3dpp/detail/ToroidalSecondOrder.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

namespace {

using Complex = std::complex<double>;

using detail::StageRef;
using detail::TrajectoryCursor;

/// a F1 + b F2 over the forces of two sets of sources (a degree absent
/// from one has none there).
RadialSourceForces Blend(double a, const RadialSourceForces& f1, double b,
                         const RadialSourceForces& f2) {
  auto out = RadialSourceForces{};
  auto add = [](std::map<int, std::vector<double>>& to, double factor,
                const std::map<int, std::vector<double>>& from) {
    for (const auto& [l, f] : from) {
      auto& sum = to[l];
      if (sum.empty()) sum.assign(f.size(), 0.0);
      for (auto i = std::size_t{0}; i < f.size(); ++i) sum[i] += factor * f[i];
    }
  };
  add(out.spheroidal, a, f1.spheroidal);
  add(out.spheroidal, b, f2.spheroidal);
  add(out.toroidal, a, f1.toroidal);
  add(out.toroidal, b, f2.toroidal);
  return out;
}

/// The stages of the step ending at a node of a grid of material steps.
std::pair<StepRecipe, std::vector<StageRef>> StagesAt(
    Scheme scheme, const std::vector<double>& dts, std::size_t node) {
  auto recipe = RecipeOf(scheme, dts[node - 1]);
  auto refs = detail::StagesOf(recipe, dts[node - 1], node);
  return {std::move(recipe), std::move(refs)};
}

void CheckTimes(std::span<const double> times) { detail::CheckTimeGrid(times); }

void CheckPerNode(const std::vector<std::vector<double>>& fields,
                  std::size_t nodes, std::size_t fieldSize, const char* what) {
  if (fields.empty()) return;
  if (fields.size() != nodes) {
    throw std::invalid_argument(
        std::format("{} must give one grid per time node", what));
  }
  for (const auto& f : fields) {
    if (f.size() != fieldSize) {
      throw std::invalid_argument(
          std::format("{} must be fields on the solver's grid", what));
    }
  }
}

void CheckTrajectory(const ForwardTrajectory* trajectory, std::size_t nodes) {
  if (trajectory == nullptr) {
    throw std::invalid_argument(
        "an itau perturbation needs the recorded forward trajectory");
  }
  if (trajectory->states.size() != nodes ||
      trajectory->loads.size() != nodes || trajectory->tides.size() != nodes) {
    throw std::invalid_argument(
        "the trajectory must give one entry per time node, with the "
        "recorded load and centrifugal coefficients (build it from a "
        "recording run's snapshots)");
  }
  if (!trajectory->states.front().has_value()) {
    throw std::invalid_argument(
        "the trajectory must hold the first node's recorded state");
  }
}

}  // namespace

/// The material-unit step sizes of the time grid.
std::vector<double> TangentLinearSolver::StepSizes(const Evolution& ev,
                              std::span<const double> times) {
  auto dts = std::vector<double>{};
  for (auto n = std::size_t{0}; n + 1 < times.size(); ++n) {
    dts.push_back((times[n + 1] - times[n]) * ev.toMatTime_);
  }
  return dts;
}

std::function<LoadCallable(const std::vector<Complex>&)>
TangentLinearSolver::LoadColumnsOf(
    const Evolution& ev) {
  return [&ev](const std::vector<Complex>& c) {
    return ev.Columns(c, ev.toMatDensity_);
  };
}

std::function<LoadCallable(const std::vector<Complex>&)>
TangentLinearSolver::TideColumnsOf(
    const Evolution& ev) {
  return [&ev](const std::vector<Complex>& c) {
    return ev.Columns(c, ev.toMatPotential_);
  };
}

std::map<int, std::vector<double>> TangentLinearSolver::FirstOrderDrive(
    const ViscoelasticOperator& op, AdjointMethod method, Scheme scheme,
    double dt, const ViscoelasticState& prev, const ViscoelasticState& next,
    const std::vector<std::vector<double>>& ditauPoints) {
  const bool implicit = scheme == Scheme::BackwardEuler;
  const bool discrete = method == AdjointMethod::Discrete;
  const auto K = op.GetRheology().BranchCount();
  const auto& weights = op.SchemeWeightsOf(scheme, dt);
  const auto* dweights =
      discrete ? &op.SchemeWeightDerivativesOf(scheme, dt) : nullptr;
  auto out = std::map<int, std::vector<double>>{};
  for (const auto& [l, mPrev] : prev.m) {
    const auto& deg = op.Deg(l);
    const auto ncols = static_cast<int>(
        mPrev.size() / (K * 3 * static_cast<std::size_t>(deg.ne) * deg.np));
    const auto block = deg.NodalSize(ncols);
    const auto& d0 = prev.strain.at(l);
    const auto& d1 = next.strain.at(l);
    const auto& mNext = next.m.at(l);
    auto extra = std::vector<double>(mPrev.size(), 0.0);
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& w = weights[k];
      for (auto s = 0; s < 3; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto delta = ditauPoints[k][full];
            if (delta == 0.0) continue;
            const auto node =
                (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const auto at = (k * 3 + s) * block + node;
            const auto sn = s * block + node;
            const auto dh = dt * delta;
            if (discrete) {
              const auto& dw = (*dweights)[k];
              for (auto c = 0; c < ncols; ++c) {
                extra[at + c] =
                    implicit
                        ? dw.e[full] * dh * (mPrev[at + c] - d1[sn + c])
                        : dh * (dw.e[full] * mPrev[at + c] +
                                dw.alpha[full] * d0[sn + c] +
                                dw.beta[full] * d1[sn + c]);
              }
            } else {
              const auto itau =
                  deg.itau[k][static_cast<std::size_t>(e) * deg.np + q];
              if (itau <= 0.0) continue;
              const auto ratio = delta / itau;
              for (auto c = 0; c < ncols; ++c) {
                const auto q0 = (d0[sn + c] - mPrev[at + c]) * ratio;
                const auto q1 = (d1[sn + c] - mNext[at + c]) * ratio;
                extra[at + c] = implicit ? (1.0 - w.e[full]) * q1
                                         : w.alpha[full] * q0 + w.beta[full] * q1;
              }
            }
          }
        }
      }
    }
    out.emplace(l, std::move(extra));
  }
  return out;
}

std::map<int, std::vector<double>> TangentLinearSolver::SecondWeightDrive(
    const ViscoelasticOperator& op, Scheme scheme, double dt,
    const ViscoelasticState& prev, const ViscoelasticState& next,
    const std::vector<std::vector<double>>& ditauU,
    const std::vector<std::vector<double>>& ditauV) {
  const bool implicit = scheme == Scheme::BackwardEuler;
  const auto K = op.GetRheology().BranchCount();
  const auto& d2weights = op.SchemeWeightSecondDerivativesOf(scheme, dt);
  auto out = std::map<int, std::vector<double>>{};
  for (const auto& [l, mPrev] : prev.m) {
    const auto& deg = op.Deg(l);
    const auto ncols = static_cast<int>(
        mPrev.size() / (K * 3 * static_cast<std::size_t>(deg.ne) * deg.np));
    const auto block = deg.NodalSize(ncols);
    const auto& d0 = prev.strain.at(l);
    const auto& d1 = next.strain.at(l);
    auto extra = std::vector<double>(mPrev.size(), 0.0);
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& d2w = d2weights[k];
      for (auto s = 0; s < 3; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto dh2 = dt * dt * ditauU[k][full] * ditauV[k][full];
            if (dh2 == 0.0) continue;
            const auto node =
                (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const auto at = (k * 3 + s) * block + node;
            const auto sn = s * block + node;
            for (auto c = 0; c < ncols; ++c) {
              extra[at + c] =
                  implicit ? d2w.e[full] * dh2 * (mPrev[at + c] - d1[sn + c])
                           : dh2 * (d2w.e[full] * mPrev[at + c] +
                                    d2w.alpha[full] * d0[sn + c] +
                                    d2w.beta[full] * d1[sn + c]);
            }
          }
        }
      }
    }
    out.emplace(l, std::move(extra));
  }
  return out;
}

TangentLinearSolver::TangentLinearSolver(const Evolution& evolution,
                                         Options options)
    : evolution_(&evolution), options_(std::move(options)) {
  if (!evolution.op_->Gravitating()) {
    throw std::invalid_argument(
        "the tangent-linear runs are not built for the problem without "
        "gravity");
  }
}

/// What March takes for the material: a start response and a force per
/// step, from a direction (MaterialDrive) or from a pair of directions
/// and tangent runs (MaterialSecondDrive).
struct TangentLinearSolver::MaterialForcing {
  virtual ~MaterialForcing() = default;
  [[nodiscard]] virtual std::map<int, std::vector<double>> StartResponse() = 0;
  /// The force on a stage of a step (the step's one stage under a
  /// one-stage scheme).
  [[nodiscard]] virtual std::map<int, std::vector<double>> Force(
      const StageRef& stage) = 0;
};

struct TangentLinearSolver::MaterialDrive : TangentLinearSolver::MaterialForcing {
  const Evolution* ev = nullptr;
  PointMaterialDirection direction;  // kappa and mu at the points
  const ForwardTrajectory* trajectory = nullptr;
  std::vector<double> dts;
  // The branch fractions f_k = mu_k / mu_U at the memory points.
  std::vector<std::span<const double>> fractions;
  // The effective shear's variation per stage (its one-stage scheme and
  // length) at the points, delta mu scaled by (1 - sum_k f_k (1 - w_k))
  // there.
  std::map<std::pair<Scheme, double>, std::vector<double>> effectiveMu;
  std::optional<TrajectoryCursor> cursor;
  // the base states of the stages of the step in hand
  TrajectoryCursor::StepStates states;

  const std::vector<double>& EffectiveMuOf(Scheme scheme, double dt) {
    const auto key = std::pair(scheme, dt);
    auto found = effectiveMu.find(key);
    if (found != effectiveMu.end()) return found->second;
    const auto& op = *ev->op_;
    const auto& weights = op.SchemeWeightsOf(scheme, dt);
    auto effective = direction.mu;
    for (auto i = std::size_t{0}; i < effective.size(); ++i) {
      auto factor = 1.0;
      for (auto k = std::size_t{0}; k < fractions.size(); ++k) {
        const auto w = scheme == Scheme::BackwardEuler ? weights[k].e[i]
                                                       : 1.0 - weights[k].beta[i];
        factor -= fractions[k][i] * (1.0 - w);
      }
      effective[i] *= factor;
    }
    return effectiveMu.emplace(key, std::move(effective)).first->second;
  }

  /// The moduli's variation of a system along (delta kappa, delta mu) at
  /// the points, on x.
  std::vector<double> ApplyVariation(const DegreeOperator& system,
                                     std::span<const double> dmu,
                                     std::span<const double> x,
                                     int ncols) const {
    return system.ApplyPointVariation(ev->op_->Points(), direction.kappa, dmu,
                                      x, ncols);
  }

  /// The memory force of a memory state with the branch moduli's
  /// variation, delta mu_k = f_k delta mu at the points, per degree.
  std::vector<double> MemoryVariationForce(
      const ViscoelasticOperator::Degree& deg, int ncols,
      const std::vector<double>& mem) const {
    const auto K = fractions.size();
    const auto block = deg.NodalSize(ncols);
    const auto wc = detail::ScalarWeights(deg.k2);
    auto weighted = std::vector<double>(3 * block, 0.0);
    for (auto k = std::size_t{0}; k < K; ++k) {
      for (auto s = 0; s < 3; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full_ = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto dmu = fractions[k][full_] * direction.mu[full_];
            if (dmu == 0.0) continue;
            const auto factor =
                wc[s] * dmu * deg.Wq[static_cast<std::size_t>(e) * deg.np + q];
            const auto node_ = (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const auto src = (k * 3 + s) * block + node_;
            for (auto c = 0; c < ncols; ++c) {
              weighted[s * block + node_ + c] += factor * mem[src + c];
            }
          }
        }
      }
    }
    return deg.DualFromScalars(weighted, ncols);
  }

  /// The direction's force on a full solution and a decayed memory under
  /// the (scheme, dt) flavour of a stage: -(dK_eff) x plus the memory
  /// force of the memory with the branch moduli's variation.
  std::map<int, std::vector<double>> ForceOn(
      Scheme scheme, double dt, const std::map<int, std::vector<double>>& full,
      const std::map<int, std::vector<double>>& mem) {
    const auto& op = *ev->op_;
    const auto none = std::vector<double>{};
    const auto& effective =
        direction.mu.empty() ? none : EffectiveMuOf(scheme, dt);
    const auto flavour = op.FlavourOf(scheme, dt);
    auto out = std::map<int, std::vector<double>>{};
    for (const auto& [l, x] : full) {
      const auto& deg = op.Deg(l);
      const auto ncols = static_cast<int>(x.size() / deg.ndof);
      const auto& system = op.EffectiveSystem(*flavour, l);
      auto force = ApplyVariation(system, effective, x, ncols);
      for (auto& v : force) v = -v;
      if (!direction.mu.empty()) {
        const auto memoryForce = MemoryVariationForce(deg, ncols, mem.at(l));
        for (auto i = std::size_t{0}; i < force.size(); ++i) {
          force[i] += memoryForce[i];
        }
      }
      out.emplace(l, std::move(force));
    }
    return out;
  }

  /// The same on the unrelaxed systems (every weight one): -(dK_U) x plus,
  /// with a memory state, its force with the branch moduli's variation —
  /// the form a step's END solution satisfies with its new memory.
  std::map<int, std::vector<double>> UnrelaxedForceOn(
      const std::map<int, std::vector<double>>& full,
      const std::map<int, std::vector<double>>* mem) {
    const auto& op = *ev->op_;
    auto out = std::map<int, std::vector<double>>{};
    for (const auto& [l, x] : full) {
      const auto& deg = op.Deg(l);
      const auto ncols = static_cast<int>(x.size() / deg.ndof);
      auto force = ApplyVariation(deg.system, direction.mu, x, ncols);
      for (auto& v : force) v = -v;
      if (mem != nullptr && !direction.mu.empty()) {
        const auto memoryForce = MemoryVariationForce(deg, ncols, mem->at(l));
        for (auto i = std::size_t{0}; i < force.size(); ++i) {
          force[i] += memoryForce[i];
        }
      }
      out.emplace(l, std::move(force));
    }
    return out;
  }

  /// The unrelaxed systems' response to a force per degree.
  std::map<int, std::vector<double>> UnrelaxedResponse(
      const std::map<int, std::vector<double>>& force) const {
    const auto& op = *ev->op_;
    auto out = std::map<int, std::vector<double>>{};
    for (const auto& [l, f] : force) {
      const auto& deg = op.Deg(l);
      const auto ncols = static_cast<int>(f.size() / deg.ndof);
      out.emplace(l, deg.system.Solve(f, ncols));
    }
    return out;
  }

  /// The start's response: -K_U^-1 (dK_U) x_0 per degree, the unrelaxed
  /// systems about the base start solution under the recorded loads. Its
  /// surface values close the start's sea level solve and its strain
  /// seeds the first step (the trapezoid's alpha d_0).
  std::map<int, std::vector<double>> StartResponse() override {
    const auto& op = *ev->op_;
    auto x0 = op.SolveElastic(
        cursor->State(0), ev->Columns(trajectory->loads[0], ev->toMatDensity_),
        ev->Columns(trajectory->tides[0], ev->toMatPotential_));
    if (!trajectory->sourceForces.empty()) {
      // the base start under its sources too (none where none act there)
      for (const auto& [l, add] :
           UnrelaxedResponse(trajectory->sourceForces[0].spheroidal)) {
        auto& x = x0.at(l);
        for (auto i = std::size_t{0}; i < x.size(); ++i) x[i] += add[i];
      }
    }
    return UnrelaxedResponse(UnrelaxedForceOn(x0, nullptr));
  }

  /// The force on a stage of the step from node - 1 to node (the base
  /// stage again, from its memory state under its recorded closure):
  /// -(dK_eff) x_full plus the base decayed memory's force with the branch
  /// moduli's variation. Acting on the FULL base solution, it carries the
  /// material derivative of the load response too, so the sea level
  /// closure needs nothing beyond the memory surface it already reads.
  std::map<int, std::vector<double>> Force(const StageRef& ref) override {
    const auto& op = *ev->op_;
    const auto node = ref.node;
    if (ref.index == 0) states = cursor->Step(node);
    const auto closure = detail::ClosureOf(*trajectory, ref);
    auto step = op.StartStep(
        states.starts[ref.index], ref.length, ref.family,
        ev->Columns(trajectory->loads[node - 1], ev->toMatDensity_),
        ev->Columns(trajectory->tides[node - 1], ev->toMatPotential_));
    if (closure.forces != nullptr) step.AddForce(closure.forces->spheroidal);
    auto [next, full] =
        step.Finish(ev->Columns(*closure.load, ev->toMatDensity_),
                    ev->Columns(*closure.tide, ev->toMatPotential_));
    return ForceOn(ref.family, ref.length, full, step.mem_);
  }
};

/// The material part of a second-order run: for each ordered pair (a, b)
/// of the two directions with a material part in a, the forces of a's
/// direction on b's tangent trajectory — b's full tangent solution at the
/// node (recovered by one unrelaxed solve under b's recorded loads and
/// memory, plus b's own unrelaxed material force on the base solution
/// when b has one) and b's decayed tangent memory with b's itau drive
/// (the mixed moduli–relaxation-time term) — and at the start a's
/// unrelaxed variation on b's start solution. Symmetric in u and v.
struct TangentLinearSolver::MaterialSecondDrive
    : TangentLinearSolver::MaterialForcing {
  struct Part {
    std::shared_ptr<MaterialDrive> a;
    std::span<const Evolution::Snapshot> b;
    std::shared_ptr<MaterialDrive> bMaterial;  // may be null
    std::vector<std::vector<double>> bDitauPoints;  // empty: no itau in b
  };
  const Evolution* ev = nullptr;
  AdjointMethod method = AdjointMethod::Continuous;
  const ForwardTrajectory* trajectory = nullptr;
  std::vector<Part> parts;
  std::optional<TrajectoryCursor> cursor;
  std::vector<double> dts;
  // the states of the stages of the step in hand: the base's, and each
  // part's tangent run's
  TrajectoryCursor::StepStates baseStates;
  std::vector<TrajectoryCursor::StepStates> tangentStates;

  LoadCallable Loads(const std::vector<Complex>& c) const {
    return ev->Columns(c, ev->toMatDensity_);
  }
  LoadCallable Tides(const std::vector<Complex>& c) const {
    return ev->Columns(c, ev->toMatPotential_);
  }

  /// b's full tangent solution at a stage's end (a node, or an inner
  /// stage): the unrelaxed solve under its recorded loads and memory,
  /// plus its own material force's response on the base solution there.
  std::map<int, std::vector<double>> TangentSolution(
      const Part& part, const StageRef& ref,
      const std::map<int, std::vector<double>>& baseFull,
      const std::map<int, std::vector<double>>* baseMemory) const {
    const auto& op = *ev->op_;
    const auto snap = detail::TangentRecordOf(part.b, ref);
    auto x = op.SolveElastic(*snap.state, Loads(*snap.load), Tides(*snap.tide));
    if (snap.forces != nullptr) {
      // b perturbs the sources: its solution is under their forces too
      AddUnrelaxed(x, snap.forces->spheroidal);
    }
    if (part.bMaterial) {
      const auto extra = part.bMaterial->UnrelaxedResponse(
          part.bMaterial->UnrelaxedForceOn(baseFull, baseMemory));
      for (auto& [l, xl] : x) {
        const auto& add = extra.at(l);
        for (auto i = std::size_t{0}; i < xl.size(); ++i) xl[i] += add[i];
      }
    }
    return x;
  }

  /// Adds the unrelaxed response to degree forces.
  void AddUnrelaxed(std::map<int, std::vector<double>>& x,
                    const std::map<int, std::vector<double>>& forces) const {
    const auto& op = *ev->op_;
    for (const auto& [l, f] : forces) {
      const auto add = op.UnrelaxedSystem(l).Solve(f, 2 * (l + 1));
      auto& to = x.at(l);
      for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += add[i];
    }
  }

  /// The base run's full solution at a stage's end: the unrelaxed solve
  /// under the recorded loads and, about a source-driven run, the sources'
  /// forces.
  std::map<int, std::vector<double>> BaseSolution(const ViscoelasticState& state,
                                                  const StageRef& ref) const {
    const auto closure = detail::ClosureOf(*trajectory, ref);
    auto x = ev->op_->SolveElastic(state, Loads(*closure.load),
                                   Tides(*closure.tide));
    if (closure.forces != nullptr) AddUnrelaxed(x, closure.forces->spheroidal);
    return x;
  }

  std::map<int, std::vector<double>> StartResponse() override {
    const auto start = StageRef{};  // the first node
    const auto& base0 = cursor->State(0);
    const auto x0 = BaseSolution(base0, start);
    auto total = std::map<int, std::vector<double>>{};
    for (const auto& part : parts) {
      const auto xb = TangentSolution(part, start, x0, nullptr);
      Accumulate(total, part.a->UnrelaxedResponse(
                            part.a->UnrelaxedForceOn(xb, nullptr)));
    }
    return total;
  }

  std::map<int, std::vector<double>> Force(const StageRef& ref) override {
    const auto& op = *ev->op_;
    const auto node = ref.node;
    if (ref.index == 0) {
      const auto recipe = RecipeOf(ev->options_.scheme, dts[node - 1]);
      baseStates = cursor->Step(node);
      tangentStates.clear();
      for (const auto& part : parts) {
        tangentStates.push_back(
            detail::TangentStepStates(part.b, recipe, dts[node - 1], node));
      }
    }
    const auto& basePrev = baseStates.starts[ref.index];
    const auto& baseN = baseStates.ends[ref.index];
    const auto xN = BaseSolution(baseN, ref);
    auto total = std::map<int, std::vector<double>>{};
    for (auto p = std::size_t{0}; p < parts.size(); ++p) {
      const auto& part = parts[p];
      const auto xb = TangentSolution(part, ref, xN, &baseN.m);
      // (a stage from the step's start node opens under that node's
      // coefficients, which the trapezoid's start strain was formed
      // under; a backward Euler stage reads none)
      const auto& prev = part.b[node - 1];
      auto step = op.StartStep(tangentStates[p].starts[ref.index], ref.length,
                               ref.family, Loads(prev.loadCoefficients),
                               Tides(prev.tideCoefficients));
      if (!part.bDitauPoints.empty()) {
        step.AddMemory(FirstOrderDrive(op, method, ref.family, ref.length,
                                       basePrev, baseN, part.bDitauPoints));
      }
      Accumulate(total, part.a->ForceOn(ref.family, ref.length, xb, step.mem_));
    }
    return total;
  }

  static void Accumulate(std::map<int, std::vector<double>>& total,
                         std::map<int, std::vector<double>> term) {
    for (auto& [l, value] : term) {
      auto found = total.find(l);
      if (found == total.end()) {
        total.emplace(l, std::move(value));
      } else {
        for (auto i = std::size_t{0}; i < value.size(); ++i) {
          found->second[i] += value[i];
        }
      }
    }
  }
};

std::shared_ptr<TangentLinearSolver::MaterialDrive>
TangentLinearSolver::MakeMaterialDrive(const PointMaterialDirection& d,
                                       const ForwardTrajectory& trajectory,
                                       std::span<const double> times) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& rheology = op.GetRheology();
  const auto pointCount = op.UnrelaxedShearAtPoints().size();
  if ((!d.kappa.empty() && d.kappa.size() != pointCount) ||
      (!d.mu.empty() && d.mu.size() != pointCount)) {
    throw std::invalid_argument(
        "a material direction's moduli live at the memory points, "
        "(element, point)");
  }
  auto drive = std::make_shared<MaterialDrive>();
  drive->ev = &ev;
  drive->direction = d;
  drive->trajectory = &trajectory;
  drive->dts = StepSizes(ev, times);
  for (auto k = std::size_t{0}; k < rheology.BranchCount(); ++k) {
    drive->fractions.push_back(op.BranchFractionAtPoints(k));
  }
  drive->cursor.emplace(op, trajectory, drive->dts, ev.options_.scheme,
                        LoadColumnsOf(ev), TideColumnsOf(ev));
  return drive;
}

std::vector<Evolution::Snapshot> TangentLinearSolver::Run(
    std::span<const double> times, const TangentPerturbation& direction,
    const ForwardTrajectory* trajectory,
    std::span<const std::vector<double>> oceanFunctions,
    std::span<const std::vector<std::vector<double>>> stageOceanFunctions)
    const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto fieldSize = solver.Grid().FieldSize();
  const auto scheme = ev.options_.scheme;
  const auto nodes = times.size();
  const bool migrating = !oceanFunctions.empty();

  CheckTimes(times);
  if (direction.Empty()) {
    throw std::invalid_argument("the tangent-linear run needs a direction");
  }
  if (!direction.tide.empty()) {
    const auto coefSize = solver.Grid().CoefficientSize();
    if (direction.tide.size() != nodes) {
      throw std::invalid_argument("tide must give one entry per time node");
    }
    for (const auto& c : direction.tide) {
      if (c.size() != coefSize) {
        throw std::invalid_argument(
            "tide must be coefficient vectors of the solver's grid");
      }
      RequireTideDegrees(c, "a tide direction");
    }
  }
  CheckPerNode(direction.directLoad, nodes, fieldSize, "directLoad");
  CheckPerNode(direction.directLoadOutsideWaterBudget, nodes, fieldSize,
               "directLoadOutsideWaterBudget");
  CheckPerNode(direction.solidSurfaceChange, nodes, fieldSize,
               "solidSurfaceChange");
  CheckPerNode(direction.iceThickness, nodes, fieldSize, "iceThickness");
  if (!direction.initialSeaLevel.empty() &&
      direction.initialSeaLevel.size() != fieldSize) {
    throw std::invalid_argument(
        "initialSeaLevel must be a field on the solver's grid");
  }
  if (!direction.itau.empty()) CheckTrajectory(trajectory, nodes);
  const bool hasMaterial = !direction.material.Empty();
  if (hasMaterial) CheckTrajectory(trajectory, nodes);
  if (IsMultiStage(scheme) && options_.method == AdjointMethod::Continuous &&
      !direction.itau.empty()) {
    throw std::invalid_argument(
        "the continuous tangent-linear drive of a direction in the "
        "relaxation times (Options::method AdjointMethod::Continuous) is "
        "not built for Sdirk2: it serves the continuous second-order sweep, "
        "which is not built for it either. The discrete run (AdjointMethod::"
        "Discrete, the default method) steps by the scheme's stages; or "
        "state a one-stage scheme, Scheme::ExponentialTrapezoid, in the "
        "evolution's Options::scheme, whose default is Sdirk2");
  }
  detail::CheckStageOceanFunctions(scheme, times, migrating,
                                   stageOceanFunctions, fieldSize);

  const auto rhoI = solver.IceDensity();
  const auto rhoW = solver.WaterDensity();
  const auto& C0 = ev.state_.oceanFunction;
  // the shorelines a stage is linearised about: the node's at the step's
  // last stage, an inner stage's own before
  auto oceanAt = [&](const StageRef& ref) -> const std::vector<double>& {
    if (!migrating) return C0;
    return ref.Last() ? oceanFunctions[ref.node]
                      : stageOceanFunctions[ref.node][ref.index];
  };
  const auto dts = StepSizes(ev, times);

  // The direct load of a stage: the load perturbation, the grounded
  // thickness perturbation relative to the first node's, and the initial
  // sea level through the shorelines that moved; empty when nothing
  // drives it. The direction's node values are combined as the forward
  // run combines its forcing's at the stage (detail::AtStage), the
  // grounding is the stage's own.
  const Direct directAt = [&, migrating](const StageRef& ref) {
    auto f = std::vector<double>{};
    auto ensure = [&] {
      if (f.empty()) f.assign(fieldSize, 0.0);
    };
    if (!direction.directLoad.empty()) {
      ensure();
      const auto df = detail::AtStage(
          ref, [&](std::size_t node) { return direction.directLoad[node]; });
      for (auto i = std::size_t{0}; i < fieldSize; ++i) f[i] += df[i];
    }
    if (!direction.iceThickness.empty()) {
      ensure();
      const auto& Cn = oceanAt(ref);
      const auto dIn = detail::AtStage(
          ref, [&](std::size_t node) { return direction.iceThickness[node]; });
      const auto& dI0 = direction.iceThickness[0];
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        f[i] += rhoI * (1.0 - Cn[i]) * dIn[i] - rhoI * (1.0 - C0[i]) * dI0[i];
      }
    }
    if (migrating && !direction.initialSeaLevel.empty()) {
      ensure();
      const auto& Cn = oceanAt(ref);
      const auto& dS = direction.initialSeaLevel;
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        f[i] += rhoW * (Cn[i] - C0[i]) * dS[i];
      }
    }
    return f;
  };

  // The relaxation-time drive: the forward trajectory served forwards,
  // the perturbation carried to the memory points; each stage driven
  // about its own forward states.
  auto cursor = std::optional<TrajectoryCursor>{};
  auto forward = TrajectoryCursor::StepStates{};
  auto ditauPoints = std::vector<std::vector<double>>{};
  Drive drive{};
  if (!direction.itau.empty()) {
    ditauPoints = op.ItauPerturbationAtPoints(direction.itau);
    cursor.emplace(op, *trajectory, dts, scheme, LoadColumnsOf(ev),
                   TideColumnsOf(ev));
    drive = [&](const StageRef& ref) {
      if (ref.index == 0) forward = cursor->Step(ref.node);
      return FirstOrderDrive(op, options_.method, ref.family, ref.length,
                             forward.starts[ref.index],
                             forward.ends[ref.index], ditauPoints);
    };
  }
  std::shared_ptr<MaterialDrive> material;
  if (hasMaterial) {
    material = MakeMaterialDrive(direction.material, *trajectory, times);
  }
  Tide tideAt{};
  if (!direction.tide.empty()) {
    tideAt = [&](const StageRef& ref) {
      return detail::AtStage(
          ref, [&](std::size_t node) { return direction.tide[node]; });
    };
  }
  Sources sourcesAt{};
  if (!direction.sources.empty() || !direction.stressGluts.empty()) {
    if ((!direction.sources.empty() && direction.sources.size() != nodes) ||
        (!direction.stressGluts.empty() &&
         direction.stressGluts.size() != nodes)) {
      throw std::invalid_argument(
          "sources and stressGluts must give one entry per time node");
    }
    sourcesAt = [&](std::size_t node) {
      return std::pair{direction.sources.empty() ? std::vector<PointSource>{}
                                                 : direction.sources[node],
                       direction.stressGluts.empty()
                           ? std::vector<StressGlut>{}
                           : direction.stressGluts[node]};
    };
  }
  // About a source-driven trajectory the toroidal systems carry memory
  // too: their drive under the relaxation-time perturbation.
  const bool sourcedBase = trajectory != nullptr && !trajectory->toroidal.empty();
  ToroidalDrive toroidalDrive{};
  const bool toroidalItau = sourcedBase && !direction.itau.empty();
  const bool toroidalMaterial = sourcedBase && !direction.material.mu.empty();
  // the base run's toroidal states of the stages of the step in hand
  using Twist = detail::ToroidalTrajectories;
  auto twistOf = std::make_shared<std::optional<Twist>>();
  auto twistStages = std::make_shared<Twist::Stages>();
  if (toroidalItau || toroidalMaterial) {
    toroidalDrive.step = [&, trajectory, toroidalItau, toroidalMaterial,
                          twistOf, twistStages](
                             const ToroidalViscoelastic& toroidal,
                             const StageRef& ref) {
      if (!twistOf->has_value()) {
        twistOf->emplace(Twist::Make(toroidal, scheme, true, trajectory, {},
                                     nullptr, {}));
      }
      const auto& base = **twistOf;
      if (ref.index == 0) {
        const auto [recipe, refs] = StagesAt(scheme, dts, ref.node);
        *twistStages =
            base.BaseStages(recipe, refs, dts[ref.node - 1], ref.node);
      }
      const auto& from = twistStages->starts[ref.index];
      auto out = std::pair<ToroidalViscoelastic::Fields,
                           ToroidalViscoelastic::Fields>{};
      if (toroidalItau) {
        out.first = toroidal.FirstOrderDrive(
            options_.method == AdjointMethod::Discrete, ref.family, ref.length,
            from, twistStages->ends[ref.index], ditauPoints);
      }
      if (toroidalMaterial) {
        out.second = toroidal.MaterialForce(ref.length, ref.family,
                                            direction.material.mu,
                                            base.BaseStageOf(ref, from));
      }
      return out;
    };
    if (toroidalMaterial) {
      toroidalDrive.start = [&, trajectory](const ToroidalViscoelastic& toroidal) {
        return toroidal.StartMaterialResponse(
            direction.material.mu,
            toroidal.SolveElastic(toroidal.ZeroState(),
                                  trajectory->sourceForces[0].toroidal));
      };
    }
  }
  Direct outsideAt{}, layerAt{};
  if (!direction.directLoadOutsideWaterBudget.empty()) {
    outsideAt = [&](const StageRef& ref) {
      return detail::AtStage(ref, [&](std::size_t node) {
        return direction.directLoadOutsideWaterBudget[node];
      });
    };
  }
  if (!direction.solidSurfaceChange.empty()) {
    layerAt = [&](const StageRef& ref) {
      return detail::AtStage(ref, [&](std::size_t node) {
        return direction.solidSurfaceChange[node];
      });
    };
  }
  return March(times, oceanFunctions, stageOceanFunctions, directAt, drive,
               material.get(), tideAt, sourcesAt, toroidalDrive, outsideAt,
               layerAt);
}

std::vector<Evolution::Snapshot> TangentLinearSolver::RunSecondOrder(
    std::span<const double> times,
    std::span<const Evolution::Snapshot> tangentU,
    std::span<const Evolution::Snapshot> tangentV,
    const SecondOrderDirection& direction, const ForwardTrajectory* trajectory,
    std::span<const std::vector<double>> oceanFunctions,
    std::span<const std::vector<std::vector<double>>> stageOceanFunctions)
    const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto scheme = ev.options_.scheme;
  const auto nodes = times.size();
  const bool discrete = options_.method == AdjointMethod::Discrete;

  CheckTimes(times);
  if (IsMultiStage(scheme) && !discrete) {
    throw std::invalid_argument(
        "the continuous second-order tangent-linear run (Options::method "
        "AdjointMethod::Continuous) is not built for Sdirk2: its drive is a "
        "one-stage step's. The discrete run (AdjointMethod::Discrete, the "
        "default method) steps by the scheme's stages; or state a one-stage "
        "scheme, Scheme::ExponentialTrapezoid, in the evolution's "
        "Options::scheme, whose default is Sdirk2");
  }
  const auto dts = StepSizes(ev, times);
  const bool both = !direction.itauU.empty() && !direction.itauV.empty();
  // Which first-order drives the run carries: v's trajectory with u's
  // perturbation, u's with v's, the forward one with itau_uv.
  struct Part {
    std::span<const Evolution::Snapshot> along;  // empty: the base
    std::vector<std::vector<double>> ditau;
  };
  auto parts = std::vector<Part>{};
  if (!direction.itauU.empty()) {
    parts.push_back({tangentV, op.ItauPerturbationAtPoints(direction.itauU)});
  }
  if (!direction.itauV.empty()) {
    parts.push_back({tangentU, op.ItauPerturbationAtPoints(direction.itauV)});
  }
  if (!direction.itauUV.empty()) {
    parts.push_back({{}, op.ItauPerturbationAtPoints(direction.itauUV)});
  }
  auto cursor = std::optional<TrajectoryCursor>{};
  Drive drive{};
  const bool materialU =
      !direction.materialU.kappa.empty() || !direction.materialU.mu.empty();
  const bool materialV =
      !direction.materialV.kappa.empty() || !direction.materialV.mu.empty();
  auto requireRecorded = [&] {
    for (const auto* run : {&tangentU, &tangentV}) {
      detail::CheckRecordedTangent(*run, scheme, times);
    }
    CheckTrajectory(trajectory, nodes);
  };
  // the states of the stages of the step in hand: the base's and the two
  // tangent runs'
  auto baseStates = TrajectoryCursor::StepStates{};
  auto alongStates = std::map<const Evolution::Snapshot*,
                              TrajectoryCursor::StepStates>{};
  if (!parts.empty()) {
    requireRecorded();
    cursor.emplace(op, *trajectory, dts, scheme, LoadColumnsOf(ev),
                   TideColumnsOf(ev));
    auto uPoints = std::vector<std::vector<double>>{};
    auto vPoints = std::vector<std::vector<double>>{};
    if (both && discrete) {
      uPoints = op.ItauPerturbationAtPoints(direction.itauU);
      vPoints = op.ItauPerturbationAtPoints(direction.itauV);
    }
    drive = [&, uPoints, vPoints](const StageRef& ref) {
      const auto node = ref.node;
      if (ref.index == 0) {
        const auto recipe = RecipeOf(scheme, dts[node - 1]);
        baseStates = cursor->Step(node);
        alongStates.clear();
        for (const auto& part : parts) {
          if (part.along.empty()) continue;
          alongStates.insert_or_assign(
              part.along.data(), detail::TangentStepStates(
                                     part.along, recipe, dts[node - 1], node));
        }
      }
      const auto& basePrev = baseStates.starts[ref.index];
      const auto& baseNext = baseStates.ends[ref.index];
      auto total = std::map<int, std::vector<double>>{};
      auto add = [&](std::map<int, std::vector<double>> term) {
        for (auto& [l, value] : term) {
          auto found = total.find(l);
          if (found == total.end()) {
            total.emplace(l, std::move(value));
          } else {
            for (auto i = std::size_t{0}; i < value.size(); ++i) {
              found->second[i] += value[i];
            }
          }
        }
      };
      for (const auto& part : parts) {
        if (part.along.empty()) {
          add(FirstOrderDrive(op, options_.method, ref.family, ref.length,
                              basePrev, baseNext, part.ditau));
        } else {
          const auto& along = alongStates.at(part.along.data());
          add(FirstOrderDrive(op, options_.method, ref.family, ref.length,
                              along.starts[ref.index], along.ends[ref.index],
                              part.ditau));
        }
      }
      if (both && discrete) {
        add(SecondWeightDrive(op, ref.family, ref.length, basePrev, baseNext,
                              uPoints, vPoints));
      }
      return total;
    };
  }
  // The moduli: a's direction on b's tangent trajectory, for both orders.
  std::shared_ptr<MaterialSecondDrive> material;
  if (materialU || materialV) {
    requireRecorded();
    material = std::make_shared<MaterialSecondDrive>();
    material->ev = &ev;
    material->method = options_.method;
    material->trajectory = trajectory;
    material->dts = dts;
    material->cursor.emplace(op, *trajectory, dts, scheme, LoadColumnsOf(ev),
                             TideColumnsOf(ev));
    const auto driveU =
        materialU ? MakeMaterialDrive(direction.materialU, *trajectory, times)
                  : nullptr;
    const auto driveV =
        materialV ? MakeMaterialDrive(direction.materialV, *trajectory, times)
                  : nullptr;
    auto pointsOf = [&](const std::vector<std::vector<double>>& itau) {
      return itau.empty() ? std::vector<std::vector<double>>{}
                          : op.ItauPerturbationAtPoints(itau);
    };
    if (materialU) {
      material->parts.push_back(
          {driveU, tangentV, driveV, pointsOf(direction.itauV)});
    }
    if (materialV) {
      material->parts.push_back(
          {driveV, tangentU, driveU, pointsOf(direction.itauU)});
    }
  }
  // The toroidal systems, where the base or either run twists: each
  // direction's relaxation times on the other's toroidal trajectory (and
  // the two on the base's, through the weights' second derivatives), and
  // each direction's shear modulus on the other's toroidal steps.
  using Twist = detail::ToroidalTrajectories;
  ToroidalDrive toroidalDrive{};
  const bool twists = (trajectory != nullptr && !trajectory->toroidal.empty()) ||
                      Twist::Twists(tangentU) || Twist::Twists(tangentV);
  auto atPoints = [&](const std::vector<std::vector<double>>& itau) {
    return itau.empty() ? std::vector<std::vector<double>>{}
                        : op.ItauPerturbationAtPoints(itau);
  };
  const auto uTimes = atPoints(direction.itauU);
  const auto vTimes = atPoints(direction.itauV);
  const auto uvTimes = atPoints(direction.itauUV);
  const auto& muU = direction.materialU.mu;
  const auto& muV = direction.materialV.mu;
  auto twistOf = std::make_shared<std::optional<std::pair<Twist, Twist>>>();
  if (twists && (!uTimes.empty() || !vTimes.empty() || !uvTimes.empty() ||
                 !muU.empty() || !muV.empty())) {
    requireRecorded();
    auto along = [&, twistOf](const ToroidalViscoelastic& toroidal)
        -> const std::pair<Twist, Twist>& {
      if (!twistOf->has_value()) {
        twistOf->emplace(
            Twist::Make(toroidal, scheme, discrete, trajectory, tangentU,
                        uTimes.empty() ? nullptr : &uTimes, muU),
            Twist::Make(toroidal, scheme, discrete, trajectory, tangentV,
                        vTimes.empty() ? nullptr : &vTimes, muV));
      }
      return **twistOf;
    };
    auto add = [](ToroidalViscoelastic::Fields& total,
                  ToroidalViscoelastic::Fields term) {
      for (auto& [l, value] : term) {
        auto& to = total[l];
        if (to.empty()) {
          to = std::move(value);
        } else {
          for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += value[i];
        }
      }
    };
    // the toroidal states of the stages of the step in hand: the base's
    // and the two tangent runs'
    struct TwistStages {
      Twist::Stages base, u, v;
    };
    auto held = std::make_shared<TwistStages>();
    toroidalDrive.step = [&, along, add, held](
                             const ToroidalViscoelastic& toroidal,
                             const StageRef& ref) {
      const auto& [u, v] = along(toroidal);
      if (ref.index == 0) {
        const auto [recipe, refs] = StagesAt(scheme, dts, ref.node);
        const auto dt = dts[ref.node - 1];
        held->base = u.BaseStages(recipe, refs, dt, ref.node);
        held->u = u.TangentStages(recipe, refs, dt, ref.node);
        held->v = v.TangentStages(recipe, refs, dt, ref.node);
      }
      const auto i = ref.index;
      const auto& basePrev = held->base.starts[i];
      const auto& baseNext = held->base.ends[i];
      auto out = std::pair<ToroidalViscoelastic::Fields,
                           ToroidalViscoelastic::Fields>{};
      if (!uTimes.empty()) {
        add(out.first,
            toroidal.FirstOrderDrive(discrete, ref.family, ref.length,
                                     held->v.starts[i], held->v.ends[i], uTimes));
      }
      if (!vTimes.empty()) {
        add(out.first,
            toroidal.FirstOrderDrive(discrete, ref.family, ref.length,
                                     held->u.starts[i], held->u.ends[i], vTimes));
      }
      if (!uvTimes.empty()) {
        add(out.first, toroidal.FirstOrderDrive(discrete, ref.family, ref.length,
                                                basePrev, baseNext, uvTimes));
      }
      if (both && discrete) {
        add(out.first,
            toroidal.SecondWeightDrive(ref.family, ref.length, basePrev,
                                       baseNext, uTimes, vTimes));
      }
      if (!muU.empty() || !muV.empty()) {
        const auto baseStage = u.BaseStageOf(ref, basePrev);
        if (!muU.empty()) {
          add(out.second,
              toroidal.MaterialForce(
                  ref.length, ref.family, muU,
                  v.TangentStageOf(ref, held->v.starts[i], basePrev, baseNext,
                                   baseStage)));
        }
        if (!muV.empty()) {
          add(out.second,
              toroidal.MaterialForce(
                  ref.length, ref.family, muV,
                  u.TangentStageOf(ref, held->u.starts[i], basePrev, baseNext,
                                   baseStage)));
        }
      }
      return out;
    };
    if (!muU.empty() || !muV.empty()) {
      toroidalDrive.start = [&, along, add](const ToroidalViscoelastic& toroidal) {
        const auto& [u, v] = along(toroidal);
        const auto baseStart = u.BaseStart();
        auto out = ToroidalViscoelastic::Fields{};
        if (!muU.empty()) {
          add(out, toroidal.StartMaterialResponse(muU, v.TangentStart(baseStart)));
        }
        if (!muV.empty()) {
          add(out, toroidal.StartMaterialResponse(muV, u.TangentStart(baseStart)));
        }
        return out;
      };
    }
  }
  // No direct load under the sharp flotation rule.
  const Direct none = [](const StageRef&) { return std::vector<double>{}; };
  return March(times, oceanFunctions, stageOceanFunctions, none, drive,
               material.get(), {}, {}, toroidalDrive);
}

std::vector<Evolution::Snapshot> TangentLinearSolver::March(
    std::span<const double> times,
    std::span<const std::vector<double>> oceanFunctions,
    std::span<const std::vector<std::vector<double>>> stageOceanFunctions,
    const Direct& directAt, const Drive& drive, MaterialForcing* material,
    const Tide& tideAt, const Sources& sourcesAt,
    const ToroidalDrive& toroidalDrive, const Direct& outsideAt,
    const Direct& layerAt) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto scheme = ev.options_.scheme;
  const auto rotation = ev.options_.rotation;
  const auto nodes = times.size();
  const bool migrating = !oceanFunctions.empty();
  if (migrating) {
    if (oceanFunctions.size() != nodes) {
      throw std::invalid_argument(
          "the ocean functions must give one grid per time node");
    }
    for (const auto& Cn : oceanFunctions) {
      if (Cn.size() != fieldSize) {
        throw std::invalid_argument(
            "the ocean functions must be fields on the solver's grid");
      }
    }
  }
  detail::CheckStageOceanFunctions(scheme, times, migrating,
                                   stageOceanFunctions, fieldSize);
  if (rotation == RotationalFeedback::Full) {
    const auto kHyd = FluidTidalNumber(op.BaseMaterial(), 2);
    if (solver.TransverseMarginFor(kHyd) <= 0.0) {
      throw std::invalid_argument(
          "the transverse rotational feedback is unstable; see "
          "Evolution::Run");
    }
  }
  const auto rhoW = solver.WaterDensity();
  const auto& C0 = ev.state_.oceanFunction;
  // the shorelines a stage is linearised about: the node's at the step's
  // last stage (and at the first node), an inner stage's own before
  auto oceanAt = [&](const StageRef& ref) -> const std::vector<double>& {
    if (!migrating) return C0;
    return ref.Last() ? oceanFunctions[ref.node]
                      : stageOceanFunctions[ref.node][ref.index];
  };
  auto overrideAt = [&](const StageRef& ref) {
    return migrating ? oceanAt(ref) : std::vector<double>{};
  };
  auto totalLoadCoefficients = [&](const std::vector<double>& f,
                                   const std::vector<double>& slc,
                                   const StageRef& ref) {
    const auto& Cn = oceanAt(ref);
    auto total = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      total[i] = (f.empty() ? 0.0 : f[i]) + rhoW * Cn[i] * slc[i];
    }
    return grid.Expand(total);
  };
  // every direct load of a node, for the radial step's own load
  auto withOutside = [&](std::vector<double> f,
                         const std::vector<double>& outside) {
    if (outside.empty()) return f;
    if (f.empty()) return outside;
    for (auto i = std::size_t{0}; i < fieldSize; ++i) f[i] += outside[i];
    return f;
  };
  // The potential the radial steps see at a node: the centrifugal one of
  // the response's rotation, and the tide's perturbation.
  auto centrifugal = [&](const GRDSolver::Response& r, const StageRef& ref) {
    auto out = rotation == RotationalFeedback::None
                   ? grid.ZeroCoefficients()
                   : solver.CentrifugalCoefficients(r.angularVelocity);
    if (tideAt) {
      const auto given = tideAt(ref);
      for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += given[i];
    }
    return out;
  };
  GRDSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;
  // (keyed by a stage's one-stage scheme and length, as the forward run's)
  auto flavours = std::map<std::pair<Scheme, double>, GRDSolver>{};
  auto solverFor = [&](Scheme family, double dt) -> const GRDSolver& {
    const auto key = std::pair(family, dt);
    auto found = flavours.find(key);
    if (found == flavours.end()) {
      found = flavours
                  .emplace(key, GRDSolver(
                                    grid,
                                    dt == 0.0 ? op.ElasticLoveNumbers()
                                              : op.EffectiveLoveNumbers(
                                                    family, dt),
                                    solver.Parameters()))
                  .first;
    }
    return found->second;
  };
  auto checkConverged = [](const GRDSolver::Response& r,
                           std::size_t node) {
    if (!r.converged) {
      throw std::runtime_error(std::format(
          "the tangent-linear sea level solve at node {} did not converge "
          "(error {})",
          node, r.error));
    }
  };

  auto vstate = op.ZeroState([](int l) { return 2 * (l + 1); });
  vstate.t = times[0] * ev.toMatTime_;
  auto sigma = grid.ZeroCoefficients();
  auto psi = grid.ZeroCoefficients();
  auto response = GRDSolver::Response{};
  response.relativeSeaLevelChange.assign(fieldSize, 0.0);
  response.displacement.assign(fieldSize, 0.0);
  response.potential.assign(fieldSize, 0.0);
  response.tangentialPotential.assign(fieldSize, 0.0);

  // Interior sources (the evolution's own treatment of them): their
  // forces on the spheroidal steps, and the toroidal systems stepped
  // beside the loop.
  const bool sourced = static_cast<bool>(sourcesAt);
  // the toroidal systems are stepped under sources, or under a drive alone
  const bool twisting = sourced || static_cast<bool>(toroidalDrive.step);
  auto toroidal = std::optional<ToroidalViscoelastic>{};
  auto tstate = ToroidalViscoelastic::State{};
  auto twist = ToroidalViscoelastic::Fields{};
  auto nodeForces = RadialSourceForces{};  // the node's perturbed sources'
  if (twisting) {
    toroidal.emplace(op);
    tstate = toroidal->ZeroState();
    tstate.t = times[0] * ev.toMatTime_;
  }
  auto nodeSourceForces = [&](std::size_t node) {
    auto [points, gluts] = sourcesAt(node);
    return detail::ForcesOfSources(op, *toroidal, solver.SolverScales(),
                                   ev.toNdLength_, std::move(points),
                                   std::move(gluts));
  };
  // the perturbed sources' forces at a stage: the node's at the step's
  // last stage; inside the step the two nodes' forces, linear in time
  // between them (the forces are linear in the sources)
  auto sourceForcesAt = [&](const StageRef& ref) {
    if (ref.Last()) return nodeSourceForces(ref.node);
    return Blend(1.0 - ref.c, nodeSourceForces(ref.node - 1), ref.c,
                 nodeSourceForces(ref.node));
  };
  auto toroidalCoefficients = [&] {
    return ev.Coefficients(toroidal->Surface(twist), ev.toNdLength_);
  };

  // The start: the instantaneous elastic response to the first node's
  // direct load, where there is one (a thickness or initial sea level
  // perturbation gives none there).
  auto startResponse = std::map<int, std::vector<double>>{};
  const auto first = StageRef{};  // the first node, a stage of its own
  if (const auto f0 = directAt(first);
      !f0.empty() || material != nullptr || tideAt || sourced || outsideAt ||
      layerAt) {
    GRDSolver::GeneralisedForcing forcing;
    forcing.directLoad = f0;
    if (outsideAt) forcing.directLoadOutsideWaterBudget = outsideAt(first);
    if (layerAt) forcing.solidSurfaceChange = layerAt(first);
    if (tideAt) forcing.tide = tideAt(first);
    forcing.rotation = rotation;
    forcing.oceanFunctionOverride = overrideAt(first);
    if (material != nullptr) startResponse = material->StartResponse();
    if (sourced) {
      // the sources' unrelaxed response joins the start's
      nodeForces = sourceForcesAt(first);
      const auto& forces = nodeForces;
      for (const auto& [l, f] : forces.spheroidal) {
        auto x = op.UnrelaxedSystem(l).Solve(f, 2 * (l + 1));
        auto& to = startResponse[l];
        if (to.empty()) {
          to = std::move(x);
        } else {
          for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += x[i];
        }
      }
      twist = toroidal->SolveElastic(tstate, forces.toroidal);
    }
    if (twisting) {
      // the toroidal start: the perturbed sources' response (above), the
      // moduli's own on the base start, or nothing
      if (!sourced) twist = toroidal->SolveElastic(tstate, {});
      if (toroidalDrive.start) {
        for (const auto& [l, w] : toroidalDrive.start(*toroidal)) {
          auto& to = twist.at(l);
          for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += w[i];
        }
      }
      toroidal->SetStrain(tstate, twist);
      forcing.toroidalResponse = toroidalCoefficients();
    }
    if (material != nullptr || sourced) {
      // The start solve closes about the start response.
      forcing.displacementResponse =
          ev.Coefficients(op.Surface(startResponse, Component::U), ev.toNdLength_);
      forcing.potentialResponse = ev.Coefficients(
          op.Surface(startResponse, Component::Phi), ev.toNdPotential_);
      forcing.tangentialResponse =
          ev.Coefficients(op.Surface(startResponse, Component::V), ev.toNdLength_);
      // and, for the axial rotational feedback, its inertia moment
      if (const auto found = startResponse.find(0);
          found != startResponse.end()) {
        forcing.inertiaMomentChange =
            op.InertiaMoment(found->second, 2)[0] * ev.toNdMoment_;
      }
    }
    response =
        solverFor(scheme, 0.0).SolveGeneralised(ev.state_, forcing, sleOptions);
    checkConverged(response, 0);
    sigma = totalLoadCoefficients(
        withOutside(f0, forcing.directLoadOutsideWaterBudget),
        response.relativeSeaLevelChange, first);
    psi = centrifugal(response, first);
  }
  if (options_.record || material != nullptr || sourced) {
    op.PrimeStrain(vstate, ev.Columns(sigma, ev.toMatDensity_),
                   ev.Columns(psi, ev.toMatPotential_));
  }
  if (material != nullptr || sourced) {
    // The start strain carries the start response too.
    for (auto& [l, d] : op.Strain(startResponse)) {
      auto& to = vstate.strain.at(l);
      for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += d[i];
    }
  }

  if (twisting && tstate.strain.empty()) {
    // no start solve was made: the zero solution's strain
    twist = toroidal->SolveElastic(tstate, {});
    toroidal->SetStrain(tstate, twist);
  }

  auto snapshots = std::vector<Evolution::Snapshot>{};
  // the records of the inner stages of the step in hand
  auto innerStages = std::vector<Evolution::Snapshot::Stage>{};
  auto record = [&](std::size_t node, GRDSolver::Response& r) {
    Evolution::Snapshot snapshot;
    snapshot.t = times[node];
    snapshot.relativeSeaLevelChange = std::move(r.relativeSeaLevelChange);
    snapshot.displacement = std::move(r.displacement);
    snapshot.potential = std::move(r.potential);
    snapshot.tangentialPotential = std::move(r.tangentialPotential);
    snapshot.toroidalPotential = std::move(r.toroidalPotential);
    snapshot.angularVelocity = r.angularVelocity;
    snapshot.converged = r.converged;
    snapshot.oceanFunction =
        migrating ? oceanFunctions[node] : std::vector<double>{};
    snapshot.stages = std::move(innerStages);
    innerStages.clear();
    if (options_.record) {
      snapshot.internalState = vstate;
      snapshot.loadCoefficients = sigma;
      snapshot.tideCoefficients = psi;
      // a second-order run forms this run's solutions again from them
      if (twisting) snapshot.toroidalState = tstate;
      if (sourced) snapshot.sourceForces = nodeForces;
    }
    snapshots.push_back(std::move(snapshot));
  };
  record(0, response);

  // the toroidal systems' stage states in the step in hand
  auto toroidalStages = std::vector<ToroidalViscoelastic::State>{};
  auto toroidalCombined = ToroidalViscoelastic::State{};
  for (auto node = std::size_t{1}; node < nodes; ++node) {
    if (options_.on_node) options_.on_node(node);
    const auto dt = (times[node] - times[node - 1]) * ev.toMatTime_;
    const auto recipe = RecipeOf(scheme, dt);
    const auto refs = detail::StagesOf(recipe, dt, node);
    // One stage of the step (StepRecipe.hpp; the step's one stage under a
    // one-stage scheme), as the forward run takes it: opened from its
    // memory state, driven, closed by the sea level equation under the
    // direction as the stage takes it and the stage's own shorelines,
    // finished.
    auto stageOf = [&](const ViscoelasticState& from, const RecipeStage& stage,
                       double length, std::size_t index) {
      const auto& ref = refs[index];
      auto step = op.StartStep(from, length, stage.family,
                               ev.Columns(sigma, ev.toMatDensity_),
                               ev.Columns(psi, ev.toMatPotential_));
      if (drive) step.AddMemory(drive(ref));
      if (material != nullptr) step.AddForce(material->Force(ref));
      GRDSolver::GeneralisedForcing coupled;
      if (twisting) {
        auto forces = RadialSourceForces{};
        if (sourced) {
          forces = sourceForcesAt(ref);
          nodeForces = forces;
          step.AddForce(forces.spheroidal);
        }
        auto increment = ToroidalViscoelastic::Fields{};
        if (toroidalDrive.step) {
          auto [memory, force] = toroidalDrive.step(*toroidal, ref);
          increment = std::move(memory);
          for (auto& [l, f] : force) {
            auto& to = forces.toroidal[l];
            if (to.empty()) {
              to = std::move(f);
            } else {
              for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += f[i];
            }
          }
        }
        const auto& twistFrom = StageStart(
            stage, dt, tstate,
            std::span<const ToroidalViscoelastic::State>(toroidalStages),
            ToroidalViscoelastic::CombinedState, toroidalCombined);
        auto stepped =
            toroidal->Step(twistFrom, length, stage.family, forces.toroidal,
                           increment.empty() ? nullptr : &increment);
        toroidalStages.push_back(std::move(stepped.first));
        twist = std::move(stepped.second);
        coupled.toroidalResponse = toroidalCoefficients();
      }
      coupled.directLoad = directAt(ref);
      if (outsideAt) coupled.directLoadOutsideWaterBudget = outsideAt(ref);
      if (layerAt) coupled.solidSurfaceChange = layerAt(ref);
      if (tideAt) coupled.tide = tideAt(ref);
      coupled.displacementResponse =
          ev.Coefficients(step.MemorySurface(Component::U), ev.toNdLength_);
      coupled.potentialResponse =
          ev.Coefficients(step.MemorySurface(Component::Phi), ev.toNdPotential_);
      coupled.tangentialResponse =
          ev.Coefficients(step.MemorySurface(Component::V), ev.toNdLength_);

      coupled.inertiaMomentChange = step.InertiaMoment()[0] * ev.toNdMoment_;
      coupled.rotation = rotation;
      coupled.oceanFunctionOverride = overrideAt(ref);
      response = solverFor(stage.family, length)
                     .SolveGeneralised(ev.state_, coupled, sleOptions);
      checkConverged(response, node);
      sigma = totalLoadCoefficients(
          withOutside(coupled.directLoad, coupled.directLoadOutsideWaterBudget),
          response.relativeSeaLevelChange, ref);
      psi = centrifugal(response, ref);
      auto [next, solutions] = step.Finish(ev.Columns(sigma, ev.toMatDensity_),
                                           ev.Columns(psi, ev.toMatPotential_));
      if (!ref.Last()) {
        // an inner stage's record: a second-order run about this one reads
        // the stage's state as it reads a node's
        auto inner = Evolution::Snapshot::Stage{};
        inner.oceanFunction = overrideAt(ref);
        if (options_.record) {
          inner.internalState = next;
          inner.loadCoefficients = sigma;
          inner.tideCoefficients = psi;
          if (twisting) inner.toroidalState = toroidalStages.back();
          if (sourced) inner.sourceForces = nodeForces;
        }
        innerStages.push_back(std::move(inner));
      }
      return std::move(next);
    };
    const auto twistStart = tstate.t;
    vstate = StepByRecipe(
        recipe, dt, vstate, stageOf,
        [](std::span<const std::pair<double, const ViscoelasticState*>> terms,
           const RecipeStage&) { return CombinedState(terms); });
    if (twisting) {
      tstate = std::move(toroidalStages.back());
      tstate.t = twistStart + dt;
      toroidalStages.clear();
    }
    record(node, response);
  }
  return snapshots;
}

}  // namespace sl3dpp
