#include "sl3dpp/TangentLinear.hpp"

#include <format>
#include <optional>
#include <stdexcept>

#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

namespace {

using Complex = std::complex<double>;

/// Serves the forward internal states to a forward sweep, in order:
/// stored states where the recording kept them (their strain regenerated
/// by one elastic solve under the recorded loads), exact forward steps
/// under the recorded loads between — which amplify nothing, unlike the
/// reverse sweeps' backward reconstruction.
class ForwardCursor {
 public:
  ForwardCursor(
      const ViscoelasticOperator& op, const ForwardTrajectory& trajectory,
      std::vector<double> dts, Scheme scheme,
      std::function<LoadCallable(const std::vector<Complex>&)> loadOf,
      std::function<LoadCallable(const std::vector<Complex>&)> tideOf)
      : op_(&op),
        trajectory_(&trajectory),
        dts_(std::move(dts)),
        scheme_(scheme),
        loadOf_(std::move(loadOf)),
        tideOf_(std::move(tideOf)),
        node_(0),
        state_(*trajectory.states.front()) {
    Prime();
  }

  /// The forward state at a node, reconstructing as needed; the reference
  /// stays valid until the next call.
  const ViscoelasticState& State(std::size_t node) {
    if (node < node_) {
      throw std::logic_error("the forward cursor moves forwards");
    }
    while (node_ < node) {
      const auto target = node_ + 1;
      if (trajectory_->states[target].has_value()) {
        state_ = *trajectory_->states[target];
        node_ = target;
        Prime();
      } else {
        auto [next, solutions] =
            op_->Step(state_, dts_[node_], scheme_,
                      loadOf_(trajectory_->loads[target]),
                      tideOf_(trajectory_->tides[target]));
        state_ = std::move(next);
        node_ = target;
      }
    }
    return state_;
  }

 private:
  void Prime() {
    op_->PrimeStrain(state_, loadOf_(trajectory_->loads[node_]),
                     tideOf_(trajectory_->tides[node_]));
  }

  const ViscoelasticOperator* op_;
  const ForwardTrajectory* trajectory_;
  std::vector<double> dts_;
  Scheme scheme_;
  std::function<LoadCallable(const std::vector<Complex>&)> loadOf_, tideOf_;
  std::size_t node_;
  ViscoelasticState state_;
};

void CheckTimes(std::span<const double> times) {
  if (times.size() < 2) {
    throw std::invalid_argument("times must hold at least two entries");
  }
  for (auto i = std::size_t{1}; i < times.size(); ++i) {
    if (times[i] <= times[i - 1]) {
      throw std::invalid_argument("times must be increasing");
    }
  }
}

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
std::vector<double> TangentLinearSolver::StepSizes(const SeaLevelEvolution& ev,
                              std::span<const double> times) {
  auto dts = std::vector<double>{};
  for (auto n = std::size_t{0}; n + 1 < times.size(); ++n) {
    dts.push_back((times[n + 1] - times[n]) * ev.toMatTime_);
  }
  return dts;
}

std::function<LoadCallable(const std::vector<Complex>&)>
TangentLinearSolver::LoadColumnsOf(
    const SeaLevelEvolution& ev) {
  return [&ev](const std::vector<Complex>& c) {
    return ev.Columns(c, ev.toMatDensity_);
  };
}

std::function<LoadCallable(const std::vector<Complex>&)>
TangentLinearSolver::TideColumnsOf(
    const SeaLevelEvolution& ev) {
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

TangentLinearSolver::TangentLinearSolver(const SeaLevelEvolution& evolution,
                                         Options options)
    : evolution_(&evolution), options_(std::move(options)) {}

/// What March takes for the material: a start response and a force per
/// step, from a direction (MaterialDrive) or from a pair of directions
/// and tangent runs (MaterialSecondDrive).
struct TangentLinearSolver::MaterialForcing {
  virtual ~MaterialForcing() = default;
  [[nodiscard]] virtual std::map<int, std::vector<double>> StartResponse() = 0;
  [[nodiscard]] virtual std::map<int, std::vector<double>> Force(
      std::size_t node, double dt) = 0;
};

struct TangentLinearSolver::MaterialDrive : TangentLinearSolver::MaterialForcing {
  const SeaLevelEvolution* ev = nullptr;
  PointMaterialDirection direction;  // kappa and mu at the points
  const ForwardTrajectory* trajectory = nullptr;
  std::vector<double> dts;
  // The branch fractions f_k = mu_k / mu_U at the memory points.
  std::vector<std::span<const double>> fractions;
  // The effective shear's variation per dt at the points, delta mu scaled
  // by (1 - sum_k f_k (1 - w_k)) there.
  std::map<double, std::vector<double>> effectiveMu;
  std::optional<ForwardCursor> cursor;

  const std::vector<double>& EffectiveMuOf(double dt) {
    auto found = effectiveMu.find(dt);
    if (found != effectiveMu.end()) return found->second;
    const auto& op = *ev->op_;
    const auto scheme = ev->options_.scheme;
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
    return effectiveMu.emplace(dt, std::move(effective)).first->second;
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
  /// the dt flavour: -(dK_eff) x plus the memory force of the memory with
  /// the branch moduli's variation.
  std::map<int, std::vector<double>> ForceOn(
      double dt, const std::map<int, std::vector<double>>& full,
      const std::map<int, std::vector<double>>& mem) {
    const auto& op = *ev->op_;
    const auto scheme = ev->options_.scheme;
    const auto none = std::vector<double>{};
    const auto& effective = direction.mu.empty() ? none : EffectiveMuOf(dt);
    auto& opFlavour = op.FlavourOf(scheme, dt);
    auto out = std::map<int, std::vector<double>>{};
    for (const auto& [l, x] : full) {
      const auto& deg = op.Deg(l);
      const auto ncols = static_cast<int>(x.size() / deg.ndof);
      const auto& system = op.EffectiveSystem(opFlavour, l);
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
    const auto x0 = op.SolveElastic(
        cursor->State(0), ev->Columns(trajectory->loads[0], ev->toMatDensity_),
        ev->Columns(trajectory->tides[0], ev->toMatPotential_));
    return UnrelaxedResponse(UnrelaxedForceOn(x0, nullptr));
  }

  /// The force of the step from node - 1 to node: -(dK_eff) x_full plus
  /// the base decayed memory's force with the branch moduli's variation.
  /// Acting on the FULL base solution, it carries the material derivative
  /// of the load response too, so the sea level closure needs nothing
  /// beyond the memory surface it already reads.
  std::map<int, std::vector<double>> Force(std::size_t node,
                                           double dt) override {
    const auto& op = *ev->op_;
    const auto scheme = ev->options_.scheme;
    const auto& prev = cursor->State(node - 1);
    auto step = op.StartStep(prev, dt, scheme,
                             ev->Columns(trajectory->loads[node - 1], ev->toMatDensity_),
                             ev->Columns(trajectory->tides[node - 1], ev->toMatPotential_));
    auto [next, full] =
        step.Finish(ev->Columns(trajectory->loads[node], ev->toMatDensity_),
                    ev->Columns(trajectory->tides[node], ev->toMatPotential_));
    return ForceOn(dt, full, step.mem_);
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
    std::span<const SeaLevelEvolution::Snapshot> b;
    std::shared_ptr<MaterialDrive> bMaterial;  // may be null
    std::vector<std::vector<double>> bDitauPoints;  // empty: no itau in b
  };
  const SeaLevelEvolution* ev = nullptr;
  AdjointMethod method = AdjointMethod::Continuous;
  const ForwardTrajectory* trajectory = nullptr;
  std::vector<Part> parts;
  std::optional<ForwardCursor> cursor;

  LoadCallable Loads(const std::vector<Complex>& c) const {
    return ev->Columns(c, ev->toMatDensity_);
  }
  LoadCallable Tides(const std::vector<Complex>& c) const {
    return ev->Columns(c, ev->toMatPotential_);
  }

  /// b's full tangent solution at a node: the unrelaxed solve under its
  /// recorded loads and memory, plus its own material force's response
  /// on the base solution there.
  std::map<int, std::vector<double>> TangentSolution(
      const Part& part, std::size_t node,
      const std::map<int, std::vector<double>>& baseFull,
      const std::map<int, std::vector<double>>* baseMemory) const {
    const auto& op = *ev->op_;
    const auto& snap = part.b[node];
    auto x = op.SolveElastic(*snap.internalState, Loads(snap.loadCoefficients),
                             Tides(snap.tideCoefficients));
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

  std::map<int, std::vector<double>> StartResponse() override {
    const auto& op = *ev->op_;
    const auto& base0 = cursor->State(0);
    const auto x0 = op.SolveElastic(base0, Loads(trajectory->loads[0]),
                                    Tides(trajectory->tides[0]));
    auto total = std::map<int, std::vector<double>>{};
    for (const auto& part : parts) {
      const auto xb = TangentSolution(part, 0, x0, nullptr);
      Accumulate(total, part.a->UnrelaxedResponse(
                            part.a->UnrelaxedForceOn(xb, nullptr)));
    }
    return total;
  }

  std::map<int, std::vector<double>> Force(std::size_t node,
                                           double dt) override {
    const auto& op = *ev->op_;
    const auto scheme = ev->options_.scheme;
    const ViscoelasticState basePrev = cursor->State(node - 1);
    const auto& baseN = cursor->State(node);
    const auto xN = op.SolveElastic(baseN, Loads(trajectory->loads[node]),
                                    Tides(trajectory->tides[node]));
    auto total = std::map<int, std::vector<double>>{};
    for (const auto& part : parts) {
      const auto xb = TangentSolution(part, node, xN, &baseN.m);
      const auto& prev = part.b[node - 1];
      auto step = op.StartStep(*prev.internalState, dt, scheme,
                               Loads(prev.loadCoefficients),
                               Tides(prev.tideCoefficients));
      if (!part.bDitauPoints.empty()) {
        step.AddMemory(FirstOrderDrive(op, method, scheme, dt, basePrev, baseN,
                                       part.bDitauPoints));
      }
      Accumulate(total, part.a->ForceOn(dt, xb, step.mem_));
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

std::vector<SeaLevelEvolution::Snapshot> TangentLinearSolver::Run(
    std::span<const double> times, const TangentPerturbation& direction,
    const ForwardTrajectory* trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
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
  if (migrating && !direction.directLoad.empty()) {
    throw std::invalid_argument(
        "a migrating run is driven by its ice thickness, not a direct "
        "load: perturb iceThickness");
  }
  CheckPerNode(direction.directLoad, nodes, fieldSize, "directLoad");
  CheckPerNode(direction.iceThickness, nodes, fieldSize, "iceThickness");
  if (!direction.initialSeaLevel.empty() &&
      direction.initialSeaLevel.size() != fieldSize) {
    throw std::invalid_argument(
        "initialSeaLevel must be a field on the solver's grid");
  }
  if (!direction.itau.empty()) CheckTrajectory(trajectory, nodes);
  const bool hasMaterial = !direction.material.Empty();
  if (hasMaterial) CheckTrajectory(trajectory, nodes);

  const auto rhoI = solver.IceDensity();
  const auto rhoW = solver.WaterDensity();
  const auto& C0 = ev.state_.oceanFunction;
  auto oceanAt = [&](std::size_t node) -> const std::vector<double>& {
    return migrating ? oceanFunctions[node] : C0;
  };

  // The direct load of a node: the load perturbation, the grounded
  // thickness perturbation relative to the first node's, and the initial
  // sea level through the shorelines that moved; empty when nothing
  // drives the node.
  const Direct directAt = [&, migrating](std::size_t node) {
    auto f = std::vector<double>{};
    auto ensure = [&] {
      if (f.empty()) f.assign(fieldSize, 0.0);
    };
    if (!direction.directLoad.empty()) {
      ensure();
      const auto& df = direction.directLoad[node];
      for (auto i = std::size_t{0}; i < fieldSize; ++i) f[i] += df[i];
    }
    if (!direction.iceThickness.empty()) {
      ensure();
      const auto& Cn = oceanAt(node);
      const auto& dIn = direction.iceThickness[node];
      const auto& dI0 = direction.iceThickness[0];
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        f[i] += rhoI * (1.0 - Cn[i]) * dIn[i] - rhoI * (1.0 - C0[i]) * dI0[i];
      }
    }
    if (migrating && !direction.initialSeaLevel.empty()) {
      ensure();
      const auto& Cn = oceanAt(node);
      const auto& dS = direction.initialSeaLevel;
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        f[i] += rhoW * (Cn[i] - C0[i]) * dS[i];
      }
    }
    return f;
  };

  // The relaxation-time drive: the forward trajectory served forwards,
  // the perturbation carried to the memory points.
  auto cursor = std::optional<ForwardCursor>{};
  auto ditauPoints = std::vector<std::vector<double>>{};
  Drive drive{};
  if (!direction.itau.empty()) {
    ditauPoints = op.ItauPerturbationAtPoints(direction.itau);
    cursor.emplace(op, *trajectory, StepSizes(ev, times), scheme,
                   LoadColumnsOf(ev), TideColumnsOf(ev));
    drive = [&](std::size_t node, double dt) {
      const ViscoelasticState prev = cursor->State(node - 1);
      const auto& next = cursor->State(node);
      return FirstOrderDrive(op, options_.method, scheme, dt, prev, next,
                             ditauPoints);
    };
  }
  std::shared_ptr<MaterialDrive> material;
  if (hasMaterial) {
    material = MakeMaterialDrive(direction.material, *trajectory, times);
  }
  return March(times, oceanFunctions, directAt, drive, material.get());
}

std::vector<SeaLevelEvolution::Snapshot> TangentLinearSolver::RunSecondOrder(
    std::span<const double> times,
    std::span<const SeaLevelEvolution::Snapshot> tangentU,
    std::span<const SeaLevelEvolution::Snapshot> tangentV,
    const SecondOrderDirection& direction, const ForwardTrajectory* trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto scheme = ev.options_.scheme;
  const auto nodes = times.size();
  const bool discrete = options_.method == AdjointMethod::Discrete;

  CheckTimes(times);
  const bool both = !direction.itauU.empty() && !direction.itauV.empty();
  // Which first-order drives the run carries: v's trajectory with u's
  // perturbation, u's with v's, the forward one with itau_uv.
  struct Part {
    std::span<const SeaLevelEvolution::Snapshot> along;  // empty: the base
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
  auto cursor = std::optional<ForwardCursor>{};
  Drive drive{};
  const bool materialU =
      !direction.materialU.kappa.empty() || !direction.materialU.mu.empty();
  const bool materialV =
      !direction.materialV.kappa.empty() || !direction.materialV.mu.empty();
  auto requireRecorded = [&] {
    for (const auto* run : {&tangentU, &tangentV}) {
      if (run->size() != nodes) {
        throw std::invalid_argument(
            "the tangent-linear runs must give one snapshot per time node");
      }
      for (const auto& snap : *run) {
        if (!snap.internalState.has_value() || !snap.internalState->HasStrain()) {
          throw std::invalid_argument(
              "the tangent-linear runs must be recorded at every node "
              "(TangentLinearSolver::Options::record)");
        }
      }
    }
    CheckTrajectory(trajectory, nodes);
  };
  if (!parts.empty()) {
    requireRecorded();
    cursor.emplace(op, *trajectory, StepSizes(ev, times), scheme,
                   LoadColumnsOf(ev), TideColumnsOf(ev));
    auto uPoints = std::vector<std::vector<double>>{};
    auto vPoints = std::vector<std::vector<double>>{};
    if (both && discrete) {
      uPoints = op.ItauPerturbationAtPoints(direction.itauU);
      vPoints = op.ItauPerturbationAtPoints(direction.itauV);
    }
    drive = [&, uPoints, vPoints](std::size_t node, double dt) {
      const ViscoelasticState basePrev = cursor->State(node - 1);
      const auto& baseNext = cursor->State(node);
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
          add(FirstOrderDrive(op, options_.method, scheme, dt, basePrev,
                              baseNext, part.ditau));
        } else {
          add(FirstOrderDrive(op, options_.method, scheme, dt,
                              *part.along[node - 1].internalState,
                              *part.along[node].internalState, part.ditau));
        }
      }
      if (both && discrete) {
        add(SecondWeightDrive(op, scheme, dt, basePrev, baseNext, uPoints,
                              vPoints));
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
    material->cursor.emplace(op, *trajectory, StepSizes(ev, times), scheme,
                             LoadColumnsOf(ev), TideColumnsOf(ev));
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
  // No direct load under the sharp flotation rule.
  const Direct none = [](std::size_t) { return std::vector<double>{}; };
  return March(times, oceanFunctions, none, drive, material.get());
}

std::vector<SeaLevelEvolution::Snapshot> TangentLinearSolver::March(
    std::span<const double> times,
    std::span<const std::vector<double>> oceanFunctions,
    const Direct& directAt, const Drive& drive,
    MaterialForcing* material) const {
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
  if (rotation == RotationalFeedback::Full) {
    const auto kHyd = FluidTidalNumber(op.BaseMaterial(), 2);
    if (solver.TransverseMarginFor(kHyd) <= 0.0) {
      throw std::invalid_argument(
          "the transverse rotational feedback is unstable; see "
          "SeaLevelEvolution::Run");
    }
  }
  const auto rhoW = solver.WaterDensity();
  const auto& C0 = ev.state_.oceanFunction;
  auto oceanAt = [&](std::size_t node) -> const std::vector<double>& {
    return migrating ? oceanFunctions[node] : C0;
  };
  auto overrideAt = [&](std::size_t node) {
    return migrating ? oceanFunctions[node] : std::vector<double>{};
  };
  auto totalLoadCoefficients = [&](const std::vector<double>& f,
                                   const std::vector<double>& slc,
                                   std::size_t node) {
    const auto& Cn = oceanAt(node);
    auto total = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      total[i] = (f.empty() ? 0.0 : f[i]) + rhoW * Cn[i] * slc[i];
    }
    return grid.Expand(total);
  };
  auto centrifugal = [&](const SeaLevelSolver::Response& r) {
    return rotation == RotationalFeedback::None
               ? grid.ZeroCoefficients()
               : solver.CentrifugalCoefficients(r.angularVelocity);
  };
  SeaLevelSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;
  auto flavours = std::map<double, SeaLevelSolver>{};
  auto solverFor = [&](double dt) -> const SeaLevelSolver& {
    auto found = flavours.find(dt);
    if (found == flavours.end()) {
      found = flavours
                  .emplace(dt, SeaLevelSolver(
                                   grid,
                                   dt == 0.0 ? op.ElasticLoveNumbers()
                                             : op.EffectiveLoveNumbers(
                                                   scheme, dt),
                                   solver.Parameters()))
                  .first;
    }
    return found->second;
  };
  auto checkConverged = [](const SeaLevelSolver::Response& r,
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
  auto response = SeaLevelSolver::Response{};
  response.seaLevelChange.assign(fieldSize, 0.0);
  response.displacement.assign(fieldSize, 0.0);
  response.potential.assign(fieldSize, 0.0);

  // The start: the instantaneous elastic response to the first node's
  // direct load, where there is one (a thickness or initial sea level
  // perturbation gives none there).
  auto startResponse = std::map<int, std::vector<double>>{};
  if (const auto f0 = directAt(0); !f0.empty() || material != nullptr) {
    SeaLevelSolver::GeneralisedForcing forcing;
    forcing.directLoad = f0;
    forcing.rotation = rotation;
    forcing.oceanFunctionOverride = overrideAt(0);
    if (material != nullptr) {
      // The start solve closes about the material's own start response.
      startResponse = material->StartResponse();
      forcing.displacementResponse =
          ev.Coefficients(op.Surface(startResponse, Component::U), ev.toNdLength_);
      forcing.potentialResponse = ev.Coefficients(
          op.Surface(startResponse, Component::Phi), ev.toNdPotential_);
      // and, for the axial rotational feedback, its inertia moment
      if (const auto found = startResponse.find(0);
          found != startResponse.end()) {
        forcing.inertiaMomentChange =
            op.InertiaMoment(found->second, 2)[0] * ev.toNdMoment_;
      }
    }
    response = solverFor(0.0).SolveGeneralised(ev.state_, forcing, sleOptions);
    checkConverged(response, 0);
    sigma = totalLoadCoefficients(f0, response.seaLevelChange, 0);
    psi = centrifugal(response);
  }
  if (options_.record || material != nullptr) {
    op.PrimeStrain(vstate, ev.Columns(sigma, ev.toMatDensity_),
                   ev.Columns(psi, ev.toMatPotential_));
  }
  if (material != nullptr) {
    // The start strain carries the start response too.
    for (auto& [l, d] : op.Strain(startResponse)) {
      auto& to = vstate.strain.at(l);
      for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += d[i];
    }
  }

  auto snapshots = std::vector<SeaLevelEvolution::Snapshot>{};
  auto record = [&](std::size_t node, SeaLevelSolver::Response& r) {
    SeaLevelEvolution::Snapshot snapshot;
    snapshot.t = times[node];
    snapshot.seaLevelChange = std::move(r.seaLevelChange);
    snapshot.displacement = std::move(r.displacement);
    snapshot.potential = std::move(r.potential);
    snapshot.angularVelocity = r.angularVelocity;
    snapshot.converged = r.converged;
    snapshot.oceanFunction = overrideAt(node);
    if (options_.record) {
      snapshot.internalState = vstate;
      snapshot.loadCoefficients = sigma;
      snapshot.tideCoefficients = psi;
    }
    snapshots.push_back(std::move(snapshot));
  };
  record(0, response);

  for (auto node = std::size_t{1}; node < nodes; ++node) {
    if (options_.on_node) options_.on_node(node);
    const auto dt = (times[node] - times[node - 1]) * ev.toMatTime_;
    auto step = op.StartStep(vstate, dt, scheme,
                             ev.Columns(sigma, ev.toMatDensity_),
                             ev.Columns(psi, ev.toMatPotential_));
    if (drive) step.AddMemory(drive(node, dt));
    if (material != nullptr) step.AddForce(material->Force(node, dt));
    SeaLevelSolver::GeneralisedForcing coupled;
    coupled.directLoad = directAt(node);
    coupled.displacementResponse =
        ev.Coefficients(step.MemorySurface(Component::U), ev.toNdLength_);
    coupled.potentialResponse =
        ev.Coefficients(step.MemorySurface(Component::Phi), ev.toNdPotential_);

    coupled.inertiaMomentChange = step.InertiaMoment()[0] * ev.toNdMoment_;
    coupled.rotation = rotation;
    coupled.oceanFunctionOverride = overrideAt(node);
    response = solverFor(dt).SolveGeneralised(ev.state_, coupled, sleOptions);
    checkConverged(response, node);
    sigma = totalLoadCoefficients(coupled.directLoad, response.seaLevelChange,
                                  node);
    psi = centrifugal(response);
    auto [next, solutions] = step.Finish(ev.Columns(sigma, ev.toMatDensity_),
                                         ev.Columns(psi, ev.toMatPotential_));
    vstate = std::move(next);
    record(node, response);
  }
  return snapshots;
}

}  // namespace sl3dpp
