#include "sl3dpp/TangentLinear3D.hpp"

#include <complex>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sl3dpp/InitialSeaLevel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/StepRecipe.hpp"
#include "sl3dpp/detail/StepStages3D.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/InteriorSources3D.hpp"
#include "sl3dpp/detail/Parallel.hpp"
#include "sl3dpp/detail/TimeGrid.hpp"

namespace sl3dpp {

namespace {

using Complex = std::complex<double>;
using Snapshot = Evolution3D::Snapshot;
using State = Viscoelastic3D::State;
using detail::StageRef;

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

void CheckTrajectory(const ForwardTrajectory3D* trajectory,
                     std::size_t nodes) {
  if (trajectory == nullptr) {
    throw std::invalid_argument(
        "a viscosity perturbation needs the recorded forward trajectory");
  }
  if (trajectory->states.size() != nodes || !trajectory->Kept(0) ||
      trajectory->loads.size() != nodes || trajectory->tides.size() != nodes) {
    throw std::invalid_argument(
        "the trajectory must give one entry per time node, keep the first "
        "node's state, and carry the recorded load and centrifugal "
        "coefficients (build it from a recording run's snapshots)");
  }
}

void CheckGridLayout(const std::vector<std::vector<double>>& ditau,
                     std::size_t branches, std::size_t values) {
  if (ditau.size() != branches) {
    throw std::invalid_argument(std::format(
        "an itau perturbation gives one array per branch: {} branches",
        branches));
  }
  for (const auto& a : ditau) {
    if (a.size() != values) {
      throw std::invalid_argument(
          "an itau perturbation's arrays live on the coupling's grid values");
    }
  }
}

/// Serves the forward states to a forward run, in order: kept states as
/// recorded (strain regenerated from their coefficients or by one elastic
/// solve under the recorded loads), the states between rebuilt by forward
/// coupled steps under the recorded loads with the run's own iteration.
}  // namespace

class TangentLinearSolver3D::ForwardCursor3D {
 public:
  using LoadOf = std::function<Viscoelastic3D::LoadCallable3D(
      const std::vector<Complex>&)>;
  ForwardCursor3D(const Viscoelastic3D& op,
                  const ForwardTrajectory3D& trajectory,
                  std::vector<double> dts, Scheme scheme,
                  Viscoelastic3D::IterationControl control, LoadOf loadOf,
                  LoadOf tideOf)
      : op_(&op),
        trajectory_(&trajectory),
        dts_(std::move(dts)),
        scheme_(scheme),
        control_(control),
        loadOf_(std::move(loadOf)),
        tideOf_(std::move(tideOf)),
        redo_{&op, &trajectory, scheme, control, loadOf_, tideOf_} {
    redo_.Check(dts_);
    state_ = Kept(0);
    Prime(0);
  }

  /// The forward states of the stages of the step ending at a node >= 1
  /// (detail::StagePointers3D): under a one-stage scheme the two nodes'
  /// states; under a multi-stage one an inner stage's end state is the
  /// stage repeated under its recorded closure. Formed once per node; the
  /// reference stays valid until the cursor advances past the node.
  const detail::StagePointers3D& Step(std::size_t node) {
    if (stagesNode_ != node) {
      const auto& end = At(node);
      const auto& start = At(node - 1);
      const auto dt = dts_[node - 1];
      const auto recipe = RecipeOf(scheme_, dt);
      const auto refs = detail::StagesOf(recipe, dt, node);
      stages_ = redo_.Stages(recipe, refs, dt, start, end);
      stagesNode_ = node;
    }
    return stages_;
  }
  /// What closed a stage on the recorded run.
  [[nodiscard]] detail::StageClosure3D ClosureOf(const StageRef& ref) const {
    return detail::ClosureOf(*trajectory_, ref);
  }
  [[nodiscard]] const std::vector<double>& StepSizes() const { return dts_; }
  [[nodiscard]] Scheme GetScheme() const { return scheme_; }

  /// The forward state at a node, reconstructing as needed; the node
  /// before the current one is served from the kept previous state, so
  /// the drives of one step may ask for (node - 1, node) in any order.
  /// The references stay valid until the cursor advances.
  const Viscoelastic3D::State& At(std::size_t node) {
    if (node + 1 == node_ && previous_.has_value()) return *previous_;
    if (node < node_) {
      throw std::logic_error("the forward cursor moves forwards");
    }
    while (node_ < node) {
      const auto target = node_ + 1;
      if (trajectory_->Kept(target)) {
        auto next = Kept(target);
        previous_ = std::move(state_);
        state_ = std::move(next);
        node_ = target;
        Prime(node_);
        continue;
      }
      // the run's step again, stage by stage under a multi-stage scheme
      auto next = redo_.Step(target, dts_[node_], state_);
      previous_ = std::move(state_);
      state_ = std::move(next);
      node_ = target;
    }
    return state_;
  }

 private:
  [[nodiscard]] Viscoelastic3D::State Kept(std::size_t node) const {
    if (trajectory_->states[node].has_value()) {
      return *trajectory_->states[node];
    }
    return trajectory_->fetch(node);
  }
  void Prime(std::size_t node) {
    if (state_.strain.has_value()) return;
    if (state_.strainCoefficients.has_value()) {
      state_.strain = op_->Coupling().Synthesise(*state_.strainCoefficients);
    } else {
      op_->PrimeStrain(state_, LoadsAt(node), TidesAt(node));
    }
  }
  [[nodiscard]] Viscoelastic3D::LoadCallable3D LoadsAt(std::size_t n) const {
    return loadOf_(trajectory_->loads[n]);
  }
  [[nodiscard]] Viscoelastic3D::LoadCallable3D TidesAt(std::size_t n) const {
    return tideOf_(trajectory_->tides[n]);
  }

  const Viscoelastic3D* op_;
  const ForwardTrajectory3D* trajectory_;
  std::vector<double> dts_;
  Scheme scheme_;
  Viscoelastic3D::IterationControl control_;
  LoadOf loadOf_, tideOf_;
  detail::ForwardStages3D redo_;
  std::size_t node_ = 0;
  Viscoelastic3D::State state_;
  std::optional<Viscoelastic3D::State> previous_;
  /// The stage states of the step ending at stagesNode_.
  detail::StagePointers3D stages_;
  std::size_t stagesNode_ = 0;
};

namespace {

/// The strain of a recorded tangent-linear state, which must be present.
const GridCoupling::TensorField& StrainOf(const State& state,
                                          const char* what) {
  if (!state.strain.has_value()) {
    throw std::invalid_argument(std::format(
        "{} must be recorded at every node with its strain", what));
  }
  return *state.strain;
}

void Accumulate(std::vector<GridCoupling::TensorField>& total,
                std::vector<GridCoupling::TensorField>&& term) {
  if (total.empty()) {
    total = std::move(term);
    return;
  }
  for (auto k = std::size_t{0}; k < total.size(); ++k) {
    for (auto c = 0; c < 2; ++c) {
      for (auto i = std::size_t{0}; i < total[k].c[c].size(); ++i) {
        total[k].c[c][i] += term[k].c[c][i];
      }
    }
    for (auto i = std::size_t{0}; i < total[k].r.size(); ++i) {
      total[k].r[i] += term[k].r[i];
    }
  }
}

}  // namespace

TangentLinearSolver3D::TangentLinearSolver3D(const Evolution3D& evolution,
                                             Options options)
    : evolution_(&evolution), options_(std::move(options)) {}

std::vector<std::vector<double>> TangentLinearSolver3D::ItauOfLogViscosity(
    const Viscoelastic3D& op, std::span<const double> field) {
  const auto values = op.Coupling().GridValueCount();
  if (field.size() != values) {
    throw std::invalid_argument(
        "a log-viscosity perturbation lives on the coupling's grid values");
  }
  auto out = std::vector<std::vector<double>>{};
  for (const auto& itau : op.itau_) {
    auto d = std::vector<double>(values);
    for (auto i = std::size_t{0}; i < values; ++i) d[i] = -itau[i] * field[i];
    out.push_back(std::move(d));
  }
  return out;
}

std::vector<std::vector<double>>
TangentLinearSolver3D::SecondVariationOfLogViscosity(
    const Viscoelastic3D& op, std::span<const double> u,
    std::span<const double> v) {
  const auto values = op.Coupling().GridValueCount();
  if (u.size() != values || v.size() != values) {
    throw std::invalid_argument(
        "a log-viscosity perturbation lives on the coupling's grid values");
  }
  auto out = std::vector<std::vector<double>>{};
  for (const auto& itau : op.itau_) {
    auto d = std::vector<double>(values);
    for (auto i = std::size_t{0}; i < values; ++i) {
      d[i] = itau[i] * u[i] * v[i];
    }
    out.push_back(std::move(d));
  }
  return out;
}

std::vector<GridCoupling::TensorField> TangentLinearSolver3D::FirstOrderDrive(
    const Viscoelastic3D& op, AdjointMethod method, Scheme scheme, double dt,
    const State& prev, const State& next,
    const std::vector<std::vector<double>>& ditau) {
  const bool implicit = scheme == Scheme::BackwardEuler;
  const bool discrete = method == AdjointMethod::Discrete;
  const auto& coupling = op.Coupling();
  const auto values = coupling.GridValueCount();
  const auto K = prev.m.size();
  // (a backward Euler step reads no start strain: a combined stage start
  // or a state just jumped need carry none)
  const auto* d0Of =
      implicit ? nullptr : &StrainOf(prev, "the trajectory's states");
  const auto& d1 = StrainOf(next, "the trajectory's states");
  auto out = std::vector<GridCoupling::TensorField>{};
  for (auto k = std::size_t{0}; k < K; ++k) {
    auto extra = coupling.ZeroTensor();
    const auto& itauK = op.itau_[k];
    const auto& mPrev = prev.m[k];
    const auto& mNext = next.m[k];
    const auto& delta = ditau[k];
    detail::ForEach(op.options_.parallel, values, [&](std::size_t i) {
      if (delta[i] == 0.0) return;
      const auto h = dt * itauK[i];
      if (discrete) {
        const auto dh = dt * delta[i];
        if (implicit) {
          const auto f = BackwardEulerWeightDerivative(h) * dh;
          for (auto c = 0; c < 2; ++c) {
            extra.c[c][i] = f * (mPrev.c[c][i] - d1.c[c][i]);
          }
          extra.r[i] = f * (mPrev.r[i] - d1.r[i]);
        } else {
          const auto& d0 = *d0Of;
          const auto dw = ExponentialTrapezoidWeightDerivatives(h);
          for (auto c = 0; c < 2; ++c) {
            extra.c[c][i] = dh * (dw.e * mPrev.c[c][i] + dw.alpha * d0.c[c][i] +
                                  dw.beta * d1.c[c][i]);
          }
          extra.r[i] = dh * (dw.e * mPrev.r[i] + dw.alpha * d0.r[i] +
                             dw.beta * d1.r[i]);
        }
      } else {
        if (itauK[i] <= 0.0) return;
        const auto ratio = delta[i] / itauK[i];
        if (implicit) {
          const auto f = (1.0 - BackwardEulerWeight(h)) * ratio;
          for (auto c = 0; c < 2; ++c) {
            extra.c[c][i] = f * (d1.c[c][i] - mNext.c[c][i]);
          }
          extra.r[i] = f * (d1.r[i] - mNext.r[i]);
        } else {
          const auto& d0 = *d0Of;
          const auto w = ExponentialTrapezoidWeights(h);
          for (auto c = 0; c < 2; ++c) {
            extra.c[c][i] = ratio * (w.alpha * (d0.c[c][i] - mPrev.c[c][i]) +
                                     w.beta * (d1.c[c][i] - mNext.c[c][i]));
          }
          extra.r[i] = ratio * (w.alpha * (d0.r[i] - mPrev.r[i]) +
                                w.beta * (d1.r[i] - mNext.r[i]));
        }
      }
    });
    out.push_back(std::move(extra));
  }
  return out;
}

std::vector<GridCoupling::TensorField> TangentLinearSolver3D::SecondWeightDrive(
    const Viscoelastic3D& op, Scheme scheme, double dt, const State& prev,
    const State& next, const std::vector<std::vector<double>>& ditauU,
    const std::vector<std::vector<double>>& ditauV) {
  const bool implicit = scheme == Scheme::BackwardEuler;
  const auto& coupling = op.Coupling();
  const auto values = coupling.GridValueCount();
  const auto K = prev.m.size();
  const auto& d0 = StrainOf(prev, "the trajectory's states");
  const auto& d1 = StrainOf(next, "the trajectory's states");
  auto out = std::vector<GridCoupling::TensorField>{};
  for (auto k = std::size_t{0}; k < K; ++k) {
    auto extra = coupling.ZeroTensor();
    const auto& itauK = op.itau_[k];
    const auto& mPrev = prev.m[k];
    detail::ForEach(op.options_.parallel, values, [&](std::size_t i) {
      const auto dh2 = dt * dt * ditauU[k][i] * ditauV[k][i];
      if (dh2 == 0.0) return;
      const auto h = dt * itauK[i];
      if (implicit) {
        const auto f = BackwardEulerWeightSecondDerivative(h) * dh2;
        for (auto c = 0; c < 2; ++c) {
          extra.c[c][i] = f * (mPrev.c[c][i] - d1.c[c][i]);
        }
        extra.r[i] = f * (mPrev.r[i] - d1.r[i]);
      } else {
        const auto d2w = ExponentialTrapezoidWeightSecondDerivatives(h);
        for (auto c = 0; c < 2; ++c) {
          extra.c[c][i] = dh2 * (d2w.e * mPrev.c[c][i] + d2w.alpha * d0.c[c][i] +
                                 d2w.beta * d1.c[c][i]);
        }
        extra.r[i] = dh2 * (d2w.e * mPrev.r[i] + d2w.alpha * d0.r[i] +
                            d2w.beta * d1.r[i]);
      }
    });
    out.push_back(std::move(extra));
  }
  return out;
}

std::vector<double> TangentLinearSolver3D::StepSizes(
    std::span<const double> times) const {
  auto dts = std::vector<double>{};
  for (auto n = std::size_t{0}; n + 1 < times.size(); ++n) {
    dts.push_back((times[n + 1] - times[n]) * evolution_->toMatTime_);
  }
  return dts;
}

Viscoelastic3D::IterationControl TangentLinearSolver3D::Control() const {
  const auto& o = evolution_->options_;
  return {.max_iterations = o.max_coupled_iterations,
          .rtol = o.coupled_rtol,
          .anderson_memory = o.anderson_memory,
          .anderson_rate = o.anderson_rate};
}

std::function<Viscoelastic3D::LoadCallable3D(const std::vector<std::complex<double>>&)>
TangentLinearSolver3D::LoadColumnsOf() const {
  const auto* ev = evolution_;
  return [ev](const std::vector<std::complex<double>>& c) {
    return ev->Columns(c, ev->toMatDensity_);
  };
}

std::function<Viscoelastic3D::LoadCallable3D(const std::vector<std::complex<double>>&)>
TangentLinearSolver3D::TideColumnsOf() const {
  const auto* ev = evolution_;
  return [ev](const std::vector<std::complex<double>>& c) {
    return ev->Columns(c, ev->toMatPotential_);
  };
}

// -- the material drive
// -----------------------------------------------------------------------

/// What March takes for the material: a start response and a force per
/// step, from a direction (MaterialDrive3D) or from a pair of directions
/// and tangent runs (MaterialSecondDrive3D).
struct TangentLinearSolver3D::MaterialForcing3D {
  virtual ~MaterialForcing3D() = default;
  [[nodiscard]] virtual Viscoelastic3D::Solutions StartResponse() const = 0;
  /// The start's forces on a zero-length step instead (the start under
  /// lateral moduli, where the instantaneous response is a fixed point).
  virtual void ApplyStart(Viscoelastic3D::OpenStep& step) const = 0;
  /// The forces on a stage of a step (the step's one stage under a
  /// one-stage scheme), on the stage as opened.
  virtual void Apply(const StageRef& stage,
                     Viscoelastic3D::OpenStep& step) const = 0;
};

struct TangentLinearSolver3D::MaterialDrive3D
    : TangentLinearSolver3D::MaterialForcing3D {
  using Solutions = Viscoelastic3D::Solutions;
  using Complex = std::complex<double>;
  using Tensor = GridCoupling::TensorField;
  const Evolution3D* ev = nullptr;
  MaterialDirection3D direction;
  const ForwardTrajectory3D* trajectory = nullptr;
  ForwardCursor3D* cursor = nullptr;
  /// The branch fractions f_k = mu_k / mu_U per node, zero where the
  /// unrelaxed modulus vanishes.
  std::vector<std::vector<double>> fractions;

  /// The forces of a direction on solutions x with the grid strain of x
  /// (its deviatoric part through the tensor pipeline, its dilatation
  /// through the scalar one): -(dK) x for the shear weight `shear` per
  /// grid value (delta mu_eff of a step, delta mu of the start) plus the
  /// tensor `memoryForce` (the memory force's variation, empty for none),
  /// as an analysed tensor and per-degree spheroidal forces.
  std::pair<GridCoupling::SpectralTensor, std::map<int, std::vector<Complex>>>
  Forces(const GridCoupling::TensorField& strain, const Solutions* x,
         std::span<const double> shear,
         const GridCoupling::TensorField* memoryForce) const {
    const auto& op = *ev->op_;
    const auto& coupling = op.coupling_;
    const auto values = coupling.GridValueCount();
    auto grid = coupling.ZeroTensor();
    detail::ForEach(op.options_.parallel, values, [&](std::size_t i) {
      const auto w = shear.empty() ? 0.0 : -shear[i];
      for (auto c = 0; c < 2; ++c) {
        grid.c[c][i] = w * strain.c[c][i] +
                       (memoryForce ? memoryForce->c[c][i] : Complex{});
      }
      grid.r[i] = w * strain.r[i] + (memoryForce ? memoryForce->r[i] : 0.0);
    });
    auto analysed = coupling.Analyse(grid);
    auto spheroidal = std::map<int, std::vector<Complex>>{};
    if (!direction.kappa.empty()) {
      const auto sph = [&](int l) -> const DegreeOperator& {
        return op.Spheroidal(l);
      };
      auto scalar = std::vector<Complex>(coupling.ScalarCoefficientCount(), 0.0);
      for (const auto& [l, xl] : x->spheroidal) {
        coupling.AddDilatationOfSolution(scalar, l, op.Spheroidal(l), xl);
      }
      // Through grid's real component, analysed (done with it), and back
      // over the dilatation's coefficients.
      auto& field = grid.r;
      coupling.SynthesiseScalar(scalar, field);
      for (auto i = std::size_t{0}; i < values; ++i) {
        field[i] *= -direction.kappa[i];
      }
      coupling.AnalyseScalar(field, scalar);
      spheroidal = coupling.DilatationForce(scalar, sph);
    }
    return {std::move(analysed), std::move(spheroidal)};
  }

  /// The memory force's variation of a memory state: sum_k f_k delta
  /// mu(x) memory_k(x), pointwise.
  Tensor MemoryVariation(const std::vector<Tensor>& memory) const {
    const auto& op = *ev->op_;
    const auto& coupling = op.coupling_;
    const auto nodes = static_cast<std::size_t>(coupling.NodeCount());
    auto out = coupling.ZeroTensor();
    detail::ForEach(op.options_.parallel, nodes, [&](std::size_t n) {
      const auto first = coupling.FieldOffset(static_cast<int>(n));
      const auto last = first + coupling.FieldSizeOf(static_cast<int>(n));
      for (auto i = first; i < last; ++i) {
        const auto dmu = direction.mu[i];
        for (auto k = std::size_t{0}; k < fractions.size(); ++k) {
          const auto fk = fractions[k][n];
          if (fk == 0.0) continue;
          for (auto c = 0; c < 2; ++c) {
            out.c[c][i] += fk * dmu * memory[k].c[c][i];
          }
          out.r[i] += fk * dmu * memory[k].r[i];
        }
      }
    });
    return out;
  }

  /// delta mu_eff(x) = delta mu(x) (1 - sum_k f_k beta_k(x)) with a
  /// step's own lateral weights; empty without a mu direction.
  std::vector<double> EffectiveShear(const Viscoelastic3D::OpenStep& step) const {
    if (direction.mu.empty()) return {};
    const auto& op = *ev->op_;
    const auto& coupling = op.coupling_;
    const auto nodes = static_cast<std::size_t>(coupling.NodeCount());
    auto shear = std::vector<double>(coupling.GridValueCount(), 0.0);
    detail::ForEach(op.options_.parallel, nodes, [&](std::size_t n) {
      const auto first = coupling.FieldOffset(static_cast<int>(n));
      const auto last = first + coupling.FieldSizeOf(static_cast<int>(n));
      for (auto i = first; i < last; ++i) {
        auto factor = 1.0;
        for (auto k = std::size_t{0}; k < fractions.size(); ++k) {
          factor -= fractions[k][n] * step.beta_[k][i];
        }
        shear[i] = direction.mu[i] * factor;
      }
    });
    return shear;
  }

  /// The unrelaxed systems' response to the direction's forces on a
  /// solution with its grid strain and, optionally, a memory state
  /// (every weight one): -(dK_U) x plus the memory force's variation —
  /// the form a step's END solution satisfies with its new memory.
  Solutions UnrelaxedResponse(const Tensor& strain, const Solutions* x,
                              const std::vector<Tensor>* memory) const {
    const auto& op = *ev->op_;
    const auto& coupling = op.coupling_;
    const auto sph = [&](int l) -> const DegreeOperator& {
      return op.Spheroidal(l);
    };
    const auto tor = [&](int l) -> const ToroidalOperator& {
      return op.Toroidal(l);
    };
    auto memoryForce = std::optional<Tensor>{};
    if (memory != nullptr && !direction.mu.empty()) {
      memoryForce = MemoryVariation(*memory);
    }
    auto [analysed, extra] =
        Forces(strain, x, direction.mu, memoryForce ? &*memoryForce : nullptr);
    auto forces = coupling.SpheroidalForce(analysed, sph);
    for (const auto& [l, f] : extra) {
      auto& b = forces.at(l);
      for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f[i];
    }
    // under lateral moduli the unrelaxed systems are the lateral ones
    return op.SolveUnrelaxed(forces, coupling.ToroidalForce(analysed, tor));
  }

  /// The full base solutions at a node (the unrelaxed solve under its
  /// memory and recorded loads: a step's converged solution satisfies it)
  /// when the bulk direction needs them, and the node's grid strain.
  struct Base {
    std::optional<Solutions> x;
    std::optional<GridCoupling::TensorField> own;
    const GridCoupling::TensorField* strain = nullptr;
  };
  /// (at a stage's end: a node, the first among them, or an inner stage)
  Base BaseAt(const StageRef& ref) const {
    const auto& op = *ev->op_;
    const auto& state = ref.node == 0 ? cursor->At(0)
                                      : *cursor->Step(ref.node).ends[ref.index];
    const auto closure = cursor->ClosureOf(ref);
    Base base;
    const auto load = ev->Columns(*closure.load, ev->toMatDensity_);
    const auto tide = ev->Columns(*closure.tide, ev->toMatPotential_);
    if (!direction.kappa.empty() || !state.strain) {
      base.x = op.SolveElastic(state, load, tide);
      if (closure.forces != nullptr) {
        // a source-driven base: under the sources' forces too
        const auto& forces = *closure.forces;
        const auto add = op.SolveUnrelaxed(forces.spheroidal, forces.toroidal);
        // (none where no source acts at the node)
        for (auto* pair : {&base.x->spheroidal, &base.x->toroidal}) {
          const auto& from =
              pair == &base.x->spheroidal ? add.spheroidal : add.toroidal;
          for (const auto& [l, a] : from) {
            auto& to = pair->at(l);
            for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += a[i];
          }
        }
      }
    }
    if (state.strain) {
      base.strain = &*state.strain;
    } else {
      base.own = op.coupling_.Synthesise(op.StrainCoefficients(
          *base.x, [&](int l) -> const DegreeOperator& { return op.Spheroidal(l); },
          [&](int l) -> const ToroidalOperator& { return op.Toroidal(l); }));
      base.strain = &*base.own;
    }
    return base;
  }

  /// The start's response: -K_U^-1 (dK_U) x_0 by the unrelaxed systems,
  /// whose surface values close the start's sea level solve and whose
  /// strain seeds the first step.
  Solutions StartResponse() const override {
    const auto base = BaseAt(StageRef{});
    return UnrelaxedResponse(*base.strain, base.x ? &*base.x : nullptr, nullptr);
  }

  void ApplyStart(Viscoelastic3D::OpenStep& step) const override {
    const auto base = BaseAt(StageRef{});
    auto [analysed, bulk] =
        Forces(*base.strain, base.x ? &*base.x : nullptr, direction.mu, nullptr);
    step.AddForce(analysed, bulk);
  }

  /// The forces of the step ending at `node`, on the opened step: the
  /// effective system's variation on the full base solution — delta
  /// mu_eff(x) = delta mu(x) (1 - sum_k f_k beta_k(x)) with the step's
  /// own lateral weights — and the memory force of the decayed memory
  /// with delta mu_k = f_k delta mu.
  void Apply(const StageRef& ref,
             Viscoelastic3D::OpenStep& step) const override {
    const auto& op = *ev->op_;
    const auto node = ref.node;
    auto shear = std::vector<double>{};
    auto memoryForce = std::optional<Tensor>{};
    if (!direction.mu.empty()) {
      // The BASE stage's decayed memory m~ = e m + alpha d_0 (the step
      // given is the tangent's, whose memory is the tangent's own),
      // opened on the base stage's memory state — the weights are the
      // stage's, lateral alike.
      const auto& from = *cursor->Step(node).starts[ref.index];
      const auto baseStep = op.StartStep(
          from, ref.length, ref.family,
          ev->Columns(trajectory->loads[node - 1], ev->toMatDensity_),
          ev->Columns(trajectory->tides[node - 1], ev->toMatPotential_));
      shear = EffectiveShear(baseStep);
      memoryForce = MemoryVariation(baseStep.memory_);
    }
    const auto base = BaseAt(ref);
    auto [analysed, extra] = Forces(*base.strain, base.x ? &*base.x : nullptr,
                                    shear, memoryForce ? &*memoryForce : nullptr);
    step.AddForce(analysed, extra);
  }
};

/// The material part of a second-order run (docs/adjoints.md §15): for
/// each ordered pair (a, b) of the two directions with a material part in
/// a, the forces of a's direction on b's tangent trajectory — b's full
/// tangent solution at the node (recovered by one unrelaxed solve under
/// b's recorded loads and memory, plus b's own unrelaxed material force
/// on the base solution when b has one; its grid strain is the recorded
/// one) and b's decayed tangent memory with b's itau drive (the mixed
/// moduli–relaxation-time term), with the base step's lateral weights —
/// and at the start a's unrelaxed variation on b's start solution.
/// Symmetric in u and v.
struct TangentLinearSolver3D::MaterialSecondDrive3D
    : TangentLinearSolver3D::MaterialForcing3D {
  using Solutions = Viscoelastic3D::Solutions;
  using Tensor = GridCoupling::TensorField;
  struct Part {
    std::shared_ptr<MaterialDrive3D> a;
    std::span<const Snapshot> b;
    std::shared_ptr<MaterialDrive3D> bMaterial;  // may be null
    std::vector<std::vector<double>> bDitau;    // empty: no itau in b
  };
  const Evolution3D* ev = nullptr;
  AdjointMethod method = AdjointMethod::Continuous;
  const ForwardTrajectory3D* trajectory = nullptr;
  ForwardCursor3D* cursor = nullptr;
  std::vector<Part> parts;
  /// Each part's tangent run's states of the stages of the step in hand.
  mutable std::vector<detail::StagePointers3D> tangentStages;
  mutable std::size_t tangentNode = 0;

  /// A recorded tangent run's record at a stage's end.
  struct Record {
    const Viscoelastic3D::State* state = nullptr;
    const std::vector<Complex>* load = nullptr;
    const std::vector<Complex>* tide = nullptr;
    const DegreeFields* forces = nullptr;
  };
  static Record RecordOf(std::span<const Snapshot> run, const StageRef& ref) {
    const auto& snap = run[ref.node];
    if (ref.Last()) {
      return {&*snap.internalState, &snap.loadCoefficients,
              &snap.tideCoefficients,
              snap.sourceForces ? &*snap.sourceForces : nullptr};
    }
    const auto& stage = snap.stages[ref.index];
    return {&*stage.internalState, &stage.loadCoefficients,
            &stage.tideCoefficients,
            stage.sourceForces ? &*stage.sourceForces : nullptr};
  }

  Viscoelastic3D::LoadCallable3D Loads(const std::vector<Complex>& c) const {
    return ev->Columns(c, ev->toMatDensity_);
  }
  Viscoelastic3D::LoadCallable3D Tides(const std::vector<Complex>& c) const {
    return ev->Columns(c, ev->toMatPotential_);
  }
  static void Add(Solutions& to, const Solutions& add) {
    for (auto* pair : {&to.spheroidal, &to.toroidal}) {
      const auto& from = pair == &to.spheroidal ? add.spheroidal : add.toroidal;
      for (const auto& [l, f] : from) {
        auto found = pair->find(l);
        if (found == pair->end()) {
          pair->emplace(l, f);
        } else {
          for (auto i = std::size_t{0}; i < f.size(); ++i) found->second[i] += f[i];
        }
      }
    }
  }

  /// b's full tangent solutions at a node: the unrelaxed solve under its
  /// recorded loads and memory, plus its own material force's response
  /// on the base solution (with its strain and memory) there.
  Solutions TangentSolution(const Part& part, const StageRef& ref,
                            const Tensor& baseStrain, const Solutions* baseX,
                            const std::vector<Tensor>* baseMemory) const {
    const auto& op = *ev->op_;
    const auto snap = RecordOf(part.b, ref);
    auto x = op.SolveElastic(*snap.state, Loads(*snap.load), Tides(*snap.tide));
    if (snap.forces != nullptr) {
      // b perturbs the sources: its solution is under their forces too
      Add(x, op.SolveUnrelaxed(snap.forces->spheroidal, snap.forces->toroidal));
    }
    if (part.bMaterial) {
      Add(x, part.bMaterial->UnrelaxedResponse(baseStrain, baseX, baseMemory));
    }
    return x;
  }

  /// The base run's full solution at a node: the unrelaxed solve under
  /// the recorded loads and, about a source-driven run, the sources'
  /// forces.
  Solutions BaseSolution(const Viscoelastic3D::State& state,
                         const StageRef& ref) const {
    const auto& op = *ev->op_;
    const auto closure = cursor->ClosureOf(ref);
    auto x = op.SolveElastic(state, Loads(*closure.load), Tides(*closure.tide));
    if (closure.forces != nullptr) {
      Add(x, op.SolveUnrelaxed(closure.forces->spheroidal,
                               closure.forces->toroidal));
    }
    return x;
  }

  /// The base state's grid strain at a node, synthesised from the full
  /// solution when the cursor's state lacks it.
  const Tensor& BaseStrain(const Viscoelastic3D::State& state,
                           const Solutions& x, std::optional<Tensor>& own) const {
    if (state.strain) return *state.strain;
    const auto& op = *ev->op_;
    own = op.coupling_.Synthesise(op.StrainCoefficients(
        x, [&](int l) -> const DegreeOperator& { return op.Spheroidal(l); },
        [&](int l) -> const ToroidalOperator& { return op.Toroidal(l); }));
    return *own;
  }

  Solutions StartResponse() const override {
    const auto first = StageRef{};
    const auto& base0 = cursor->At(0);
    const auto x0 = BaseSolution(base0, first);
    auto own = std::optional<Tensor>{};
    const auto& strain0 = BaseStrain(base0, x0, own);
    auto total = Solutions{};
    for (const auto& part : parts) {
      const auto xb = TangentSolution(part, first, strain0, &x0, nullptr);
      Add(total, part.a->UnrelaxedResponse(*part.b[0].internalState->strain,
                                           &xb, nullptr));
    }
    return total;
  }

  void ApplyStart(Viscoelastic3D::OpenStep& step) const override {
    const auto first = StageRef{};
    const auto& base0 = cursor->At(0);
    const auto x0 = BaseSolution(base0, first);
    auto own = std::optional<Tensor>{};
    const auto& strain0 = BaseStrain(base0, x0, own);
    for (const auto& part : parts) {
      const auto xb = TangentSolution(part, first, strain0, &x0, nullptr);
      auto [analysed, bulk] = part.a->Forces(*part.b[0].internalState->strain,
                                             &xb, part.a->direction.mu, nullptr);
      step.AddForce(analysed, bulk);
    }
  }

  void Apply(const StageRef& ref,
             Viscoelastic3D::OpenStep& step) const override {
    const auto& op = *ev->op_;
    const auto node = ref.node;
    const auto& base = cursor->Step(node);
    if (tangentNode != node) {
      // each part's tangent run's stage states, from its record
      const auto dt = cursor->StepSizes()[node - 1];
      const auto recipe = RecipeOf(cursor->GetScheme(), dt);
      tangentStages.clear();
      for (const auto& part : parts) {
        tangentStages.push_back(detail::ForwardStages3D::PointersOfStages(
            op, recipe, dt, *part.b[node - 1].internalState,
            *part.b[node].internalState,
            [&](std::size_t index, const Viscoelastic3D::State&,
                detail::StagePointers3D::Owned&) {
              return &*part.b[node].stages[index].internalState;
            }));
      }
      tangentNode = node;
    }
    const auto& basePrev = *base.starts[ref.index];
    const auto& baseN = *base.ends[ref.index];
    const auto xN = BaseSolution(baseN, ref);
    auto own = std::optional<Tensor>{};
    const auto& strainN = BaseStrain(baseN, xN, own);
    // The base stage, for its lateral weights.
    const auto baseStep = op.StartStep(basePrev, ref.length, ref.family,
                                       Loads(trajectory->loads[node - 1]),
                                       Tides(trajectory->tides[node - 1]));
    for (auto p = std::size_t{0}; p < parts.size(); ++p) {
      const auto& part = parts[p];
      const auto xb = TangentSolution(part, ref, strainN, &xN, &baseN.m);
      const auto& prev = part.b[node - 1];
      auto tstep = op.StartStep(*tangentStages[p].starts[ref.index], ref.length,
                                ref.family, Loads(prev.loadCoefficients),
                                Tides(prev.tideCoefficients));
      if (!part.bDitau.empty()) {
        tstep.AddMemory(FirstOrderDrive(op, method, ref.family, ref.length,
                                        basePrev, baseN, part.bDitau));
      }
      const auto shear = part.a->EffectiveShear(baseStep);
      auto memoryForce = std::optional<Tensor>{};
      if (!part.a->direction.mu.empty()) {
        memoryForce = part.a->MemoryVariation(tstep.memory_);
      }
      auto [analysed, extra] =
          part.a->Forces(*RecordOf(part.b, ref).state->strain, &xb, shear,
                         memoryForce ? &*memoryForce : nullptr);
      step.AddForce(analysed, extra);
    }
  }
};

std::shared_ptr<TangentLinearSolver3D::MaterialDrive3D>
TangentLinearSolver3D::MakeMaterialDrive3D(const MaterialDirection3D& direction,
                                           const ForwardTrajectory3D* trajectory,
                                           ForwardCursor3D* cursor) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& coupling = op.Coupling();
  const auto values = coupling.GridValueCount();
  const auto K = op.GetRheology().BranchCount();
  for (const auto* field : {&direction.kappa, &direction.mu}) {
    if (!field->empty() && field->size() != values) {
      throw std::invalid_argument(
          "a material direction lives on the coupling's grid values");
    }
  }
  auto material = std::make_shared<MaterialDrive3D>();
  material->ev = &ev;
  material->direction = direction;
  material->trajectory = trajectory;
  material->cursor = cursor;
  // the branch fractions at each memory node, the radial reference's at
  // its memory points
  const auto np = coupling.Points().count;
  for (auto k = std::size_t{0}; k < K; ++k) {
    const auto atPoints = op.Reference().BranchFractionAtPoints(k);
    auto f = std::vector<double>(coupling.NodeCount(), 0.0);
    for (auto n = 0; n < coupling.NodeCount(); ++n) {
      f[n] = atPoints[static_cast<std::size_t>(coupling.ElementOfNode(n)) * np +
                      coupling.NodeInElementOf(n)];
    }
    material->fractions.push_back(std::move(f));
  }
  return material;
}

std::vector<Snapshot> TangentLinearSolver3D::Run(
    std::span<const double> times, const TangentPerturbation3D& direction,
    const ForwardTrajectory3D* trajectory,
    const OceanFunctions& oceanFunctions) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto fieldSize = solver.Grid().FieldSize();
  const auto scheme = ev.options_.scheme;
  const auto nodes = times.size();
  const bool migrating = !oceanFunctions.nodes.empty();
  const auto K = op.GetRheology().BranchCount();
  const auto values = op.Coupling().GridValueCount();

  CheckTimes(times);
  detail::CheckStageOceanFunctions(scheme, times, migrating,
                                   oceanFunctions.stages, fieldSize);
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
  // The itau perturbation over the grid values: the given one plus the
  // log-viscosity one.
  auto ditau = std::vector<std::vector<double>>{};
  if (!direction.itau.empty()) {
    CheckGridLayout(direction.itau, K, values);
    ditau = direction.itau;
  }
  if (!direction.logViscosity.empty()) {
    auto fromLog = ItauOfLogViscosity(op, direction.logViscosity);
    if (ditau.empty()) {
      ditau = std::move(fromLog);
    } else {
      for (auto k = std::size_t{0}; k < K; ++k) {
        for (auto i = std::size_t{0}; i < values; ++i) {
          ditau[k][i] += fromLog[k][i];
        }
      }
    }
  }
  if (!ditau.empty()) CheckTrajectory(trajectory, nodes);
  const bool hasMaterial = !direction.material.Empty();
  if (hasMaterial) CheckTrajectory(trajectory, nodes);

  const auto rhoI = solver.IceDensity();
  const auto rhoW = solver.WaterDensity();
  const auto& C0 = ev.state_.oceanFunction;
  // the shorelines a stage is linearised about: the node's at the step's
  // last stage, an inner stage's own before
  auto oceanAt = [&](const StageRef& ref) -> const std::vector<double>& {
    if (!migrating) return C0;
    return ref.Last() ? oceanFunctions.nodes[ref.node]
                      : oceanFunctions.stages[ref.node][ref.index];
  };
  // (the direction's node values combined as the forward run combines its
  // forcing's at the stage, detail::AtStage; the grounding the stage's own)
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

  auto cursor = std::optional<ForwardCursor3D>{};
  Drive drive{};
  if (!ditau.empty() || hasMaterial) {
    cursor.emplace(op, *trajectory, StepSizes(times), scheme, Control(),
                   LoadColumnsOf(), TideColumnsOf());
  }
  if (!ditau.empty()) {
    // each stage driven about its own forward states
    drive = [&](const StageRef& ref) {
      const auto& forward = cursor->Step(ref.node);
      return FirstOrderDrive(op, options_.method, ref.family, ref.length,
                             *forward.starts[ref.index],
                             *forward.ends[ref.index], ditau);
    };
  }
  std::shared_ptr<MaterialDrive3D> material;
  if (hasMaterial) {
    material = MakeMaterialDrive3D(direction.material, trajectory, &*cursor);
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
  return March(times, oceanFunctions, directAt, drive, material.get(), tideAt,
               sourcesAt, outsideAt, layerAt);
}

std::vector<Snapshot> TangentLinearSolver3D::RunSecondOrder(
    std::span<const double> times, std::span<const Snapshot> tangentU,
    std::span<const Snapshot> tangentV,
    const SecondOrderDirection3D& direction,
    const ForwardTrajectory3D* trajectory,
    const OceanFunctions& oceanFunctions) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto scheme = ev.options_.scheme;
  const auto nodes = times.size();
  const bool discrete = options_.method == AdjointMethod::Discrete;
  const auto K = op.GetRheology().BranchCount();
  const auto values = op.Coupling().GridValueCount();

  // about a state calibrated under migrating shorelines the run holds the
  // initial sea level, which the calibrated run's own does not
  if (ev.Calibrated() && !oceanFunctions.nodes.empty() &&
      !options_.initial_sea_level_held) {
    RefuseCalibratedSecondOrderRun();
  }
  CheckTimes(times);
  for (const auto run : {tangentU, tangentV}) {
    detail::CheckTangentMethod(
        run, options_.method,
        "the second-order run (TangentLinearSolver3D::RunSecondOrder)");
  }
  // the tangent runs recorded at every node with their strains and, under
  // a multi-stage scheme, at every inner stage
  auto requireRecorded = [&] {
    for (const auto* run : {&tangentU, &tangentV}) {
      detail::CheckRecordedTangent3D(*run, scheme, times);
    }
    CheckTrajectory(trajectory, nodes);
  };
  struct Part {
    std::span<const Snapshot> along;  // empty: the forward trajectory
    const std::vector<std::vector<double>>* ditau;
  };
  auto parts = std::vector<Part>{};
  if (!direction.itauU.empty()) {
    CheckGridLayout(direction.itauU, K, values);
    parts.push_back({tangentV, &direction.itauU});
  }
  if (!direction.itauV.empty()) {
    CheckGridLayout(direction.itauV, K, values);
    parts.push_back({tangentU, &direction.itauV});
  }
  if (!direction.itauUV.empty()) {
    CheckGridLayout(direction.itauUV, K, values);
    parts.push_back({{}, &direction.itauUV});
  }
  const bool both = !direction.itauU.empty() && !direction.itauV.empty();
  auto cursor = std::optional<ForwardCursor3D>{};
  Drive drive{};
  // each tangent run's states of the stages of the step in hand
  auto alongStages =
      std::map<const Snapshot*, detail::StagePointers3D>{};
  auto alongNode = std::size_t{0};
  if (!parts.empty()) {
    requireRecorded();
    cursor.emplace(op, *trajectory, StepSizes(times), scheme, Control(),
                   LoadColumnsOf(), TideColumnsOf());
    drive = [&, both](const StageRef& ref) {
      const auto node = ref.node;
      const auto& base = cursor->Step(node);
      if (alongNode != node) {
        const auto dt = cursor->StepSizes()[node - 1];
        const auto recipe = RecipeOf(scheme, dt);
        alongStages.clear();
        for (const auto& part : parts) {
          if (part.along.empty()) continue;
          const auto run = part.along;
          alongStages.insert_or_assign(
              run.data(),
              detail::ForwardStages3D::PointersOfStages(
                  op, recipe, dt, *run[node - 1].internalState,
                  *run[node].internalState,
                  [run, node](std::size_t index, const State&,
                              detail::StagePointers3D::Owned&) {
                    return &*run[node].stages[index].internalState;
                  }));
        }
        alongNode = node;
      }
      const auto& basePrev = *base.starts[ref.index];
      const auto& baseNext = *base.ends[ref.index];
      auto total = std::vector<GridCoupling::TensorField>{};
      for (const auto& part : parts) {
        if (part.along.empty()) {
          Accumulate(total,
                     FirstOrderDrive(op, options_.method, ref.family, ref.length,
                                     basePrev, baseNext, *part.ditau));
        } else {
          const auto& along = alongStages.at(part.along.data());
          Accumulate(total,
                     FirstOrderDrive(op, options_.method, ref.family, ref.length,
                                     *along.starts[ref.index],
                                     *along.ends[ref.index], *part.ditau));
        }
      }
      if (both && discrete) {
        Accumulate(total,
                   SecondWeightDrive(op, ref.family, ref.length, basePrev,
                                     baseNext, direction.itauU, direction.itauV));
      }
      return total;
    };
  }
  // The moduli: a's direction on b's tangent trajectory, for both orders.
  const bool materialU = !direction.materialU.Empty();
  const bool materialV = !direction.materialV.Empty();
  std::shared_ptr<MaterialSecondDrive3D> material;
  if (materialU || materialV) {
    requireRecorded();
    if (!cursor) {
      cursor.emplace(op, *trajectory, StepSizes(times), scheme, Control(),
                     LoadColumnsOf(), TideColumnsOf());
    }
    material = std::make_shared<MaterialSecondDrive3D>();
    material->ev = &ev;
    material->method = options_.method;
    material->trajectory = trajectory;
    material->cursor = &*cursor;
    const auto driveU =
        materialU ? MakeMaterialDrive3D(direction.materialU, trajectory, &*cursor)
                  : nullptr;
    const auto driveV =
        materialV ? MakeMaterialDrive3D(direction.materialV, trajectory, &*cursor)
                  : nullptr;
    if (materialU) {
      material->parts.push_back({driveU, tangentV, driveV, direction.itauV});
    }
    if (materialV) {
      material->parts.push_back({driveV, tangentU, driveU, direction.itauU});
    }
  }
  const Direct none = [](const StageRef&) { return std::vector<double>{}; };
  return March(times, oceanFunctions, none, drive, material.get());
}

std::vector<Snapshot> TangentLinearSolver3D::March(
    std::span<const double> times,
    const OceanFunctions& oceanFunctions,
    const Direct& directAt, const Drive& drive,
    const MaterialForcing3D* material, const Tide& tideAt,
    const Sources& sourcesAt, const Direct& outsideAt,
    const Direct& layerAt) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto scheme = ev.options_.scheme;
  const auto rotation = ev.options_.rotation;
  const auto nodes = times.size();
  const bool migrating = !oceanFunctions.nodes.empty();
  if (migrating) {
    if (oceanFunctions.nodes.size() != nodes) {
      throw std::invalid_argument(
          "the ocean functions must give one grid per time node");
    }
    for (const auto& Cn : oceanFunctions.nodes) {
      if (Cn.size() != fieldSize) {
        throw std::invalid_argument(
            "the ocean functions must be fields on the solver's grid");
      }
    }
  }
  if (rotation == RotationalFeedback::Full) {
    const auto kHyd = FluidTidalNumber(op.GetRheology().BaseMaterial(), 2);
    if (solver.TransverseMarginFor(kHyd) <= 0.0) {
      throw std::invalid_argument(
          "the transverse rotational feedback is unstable; see "
          "Evolution3D::Run");
    }
  }
  detail::CheckStageOceanFunctions(scheme, times, migrating,
                                   oceanFunctions.stages, fieldSize);
  const auto rhoW = solver.WaterDensity();
  const auto& C0 = ev.state_.oceanFunction;
  // the shorelines a stage is linearised about: the node's at the step's
  // last stage (and at the first node), an inner stage's own before
  auto oceanAt = [&](const StageRef& ref) -> const std::vector<double>& {
    if (!migrating) return C0;
    return ref.Last() ? oceanFunctions.nodes[ref.node]
                      : oceanFunctions.stages[ref.node][ref.index];
  };
  auto overrideAt = [&](const StageRef& ref) {
    return migrating ? oceanAt(ref) : std::vector<double>{};
  };
  // the stage's load outside the water budget, which joins the total
  // load, and its layer on the solid surface
  auto outsideNow = std::vector<double>{};
  auto layerNow = std::vector<double>{};
  auto totalLoadCoefficients = [&](const std::vector<double>& f,
                                   const std::vector<double>& slc,
                                   const StageRef& ref) {
    const auto& Cn = oceanAt(ref);
    auto total = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      total[i] = (f.empty() ? 0.0 : f[i]) +
                 (outsideNow.empty() ? 0.0 : outsideNow[i]) +
                 rhoW * Cn[i] * slc[i];
    }
    return grid.Expand(total);
  };
  // The potential the steps see: the centrifugal one of the response's
  // rotation, and the node's tide perturbation.
  auto tideNow = std::vector<std::complex<double>>{};
  auto centrifugal = [&](const GRDSolver::Response& r) {
    auto out = rotation == RotationalFeedback::None
                   ? grid.ZeroCoefficients()
                   : solver.CentrifugalCoefficients(r.angularVelocity);
    for (auto i = std::size_t{0}; i < tideNow.size(); ++i) out[i] += tideNow[i];
    return out;
  };

  GRDSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;
  auto flavours = std::map<const LoveNumbers*, GRDSolver>{};
  auto solverFor = [&](const LoveNumbers& table) -> const GRDSolver& {
    auto found = flavours.find(&table);
    if (found == flavours.end()) {
      found =
          flavours
              .emplace(&table, GRDSolver(grid, table, solver.Parameters()))
              .first;
    }
    return found->second;
  };

  // Distributed, every rank solves the sea level equation alike and the
  // first rank's answers serve all (Evolution3D::March).
  const auto& collectives = op.GetOptions().collectives;
  auto sigma = grid.ZeroCoefficients();
  auto psi = grid.ZeroCoefficients();
  auto response = GRDSolver::Response{};
  auto agree = [&] {
    if (!collectives.Distributed()) return;
    for (auto* coefficients : {&sigma, &psi}) {
      collectives.share(
          std::span(reinterpret_cast<double*>(coefficients->data()),
                    2 * coefficients->size()));
    }
    for (auto* field : {&response.relativeSeaLevelChange, &response.displacement,
                        &response.potential, &response.tangentialPotential,
                        &response.toroidalPotential}) {
      collectives.share(*field);
    }
    collectives.share(response.angularVelocity);
    auto converged = response.converged ? 1.0 : 0.0;
    collectives.share(std::span(&converged, 1));
    response.converged = converged != 0.0;
  };

  auto tstate = op.ZeroState();
  tstate.t = times[0] * ev.toMatTime_;
  response.relativeSeaLevelChange.assign(fieldSize, 0.0);
  response.displacement.assign(fieldSize, 0.0);
  response.potential.assign(fieldSize, 0.0);
  response.tangentialPotential.assign(fieldSize, 0.0);
  // The coupled pass of a step, shared by the steps and, under lateral
  // moduli, the start as a zero-length step.
  auto step = std::optional<Viscoelastic3D::OpenStep>{};
  const GRDSolver* effectiveSolver = nullptr;
  auto f = std::vector<double>{};
  // the stage in hand (the first node a stage of its own)
  auto stageNow = StageRef{};
  auto closure = [&](const Viscoelastic3D::Solutions& responses) {
    GRDSolver::GeneralisedForcing coupled;
    coupled.directLoad = f;
    coupled.directLoadOutsideWaterBudget = outsideNow;
    coupled.solidSurfaceChange = layerNow;
    coupled.tide = tideNow;
    coupled.displacementResponse =
        ev.Coefficients(op.Surface(responses, Component::U), ev.toNdLength_);
    coupled.potentialResponse =
        ev.Coefficients(op.Surface(responses, Component::Phi), ev.toNdPotential_);
    coupled.tangentialResponse =
        ev.Coefficients(op.Surface(responses, Component::V), ev.toNdLength_);
    coupled.toroidalResponse =
        ev.Coefficients(op.SurfaceToroidal(responses), ev.toNdLength_);
    coupled.inertiaMomentChange = step->InertiaMoment(responses) * ev.toNdMoment_;
    coupled.rotation = rotation;
    coupled.oceanFunctionOverride = overrideAt(stageNow);
    response = effectiveSolver->SolveGeneralised(ev.state_, coupled, sleOptions);
    sigma = totalLoadCoefficients(f, response.relativeSeaLevelChange, stageNow);
    psi = centrifugal(response);
    agree();
    return std::pair{ev.Columns(sigma, ev.toMatDensity_),
                     ev.Columns(psi, ev.toMatPotential_)};
  };
  auto coupledPass = [&](const GridCoupling::SpectralTensor& s) {
    auto responses = step->Respond(s);
    const auto [load, tide] = closure(responses);
    step->AddChannels(responses, load, tide);
    return step->StrainOf(responses);
  };
  const Viscoelastic3D::IterationControl control{
      .max_iterations = options_.max_coupled_iterations,
      .rtol = options_.coupled_rtol,
      .anderson_memory = options_.anderson_memory,
      .anderson_rate = options_.anderson_rate};
  const bool nested = op.GetOptions().solver ==
                      Viscoelastic3D::Options::StepSolver::ConjugateGradients;
  auto close = [&] {
    return nested ? step->Nest(closure, control)
                  : step->Converge(coupledPass, control);
  };
  auto startInfo = Viscoelastic3D::StepInfo{};
  const bool lateral = op.HasLateralModuli();
  auto startResponse = std::optional<Viscoelastic3D::Solutions>{};
  const auto first = StageRef{};
  if (tideAt) tideNow = tideAt(first);
  if (outsideAt) outsideNow = outsideAt(first);
  if (layerAt) layerNow = layerAt(first);
  // The perturbed sources' forces at a stage, on the open step: the
  // node's at the step's last stage; inside the step the two nodes'
  // forces, linear in time between them (the forces are linear in the
  // sources).
  const bool sourced = static_cast<bool>(sourcesAt);
  auto nodeForces = DegreeFields{};  // the stage's perturbed sources' forces
  auto forcesAtNode = [&](std::size_t at) {
    auto [points, gluts] = sourcesAt(at);
    return detail::ForcesOfSources3D(op, solver.SolverScales(), ev.toNdLength_,
                                     std::move(points), std::move(gluts));
  };
  auto applySources = [&](const StageRef& ref) {
    if (!sourced) return;
    nodeForces = ref.Last() ? forcesAtNode(ref.node)
                            : detail::Blend(1.0 - ref.c,
                                            forcesAtNode(ref.node - 1), ref.c,
                                            forcesAtNode(ref.node));
    step->AddDegreeForces(nodeForces.spheroidal, nodeForces.toroidal);
  };
  // the start as a zero-length step: under lateral moduli, and under
  // perturbed sources, whose forces act on the degree systems directly
  const bool stepStart = lateral || sourced;
  if (stepStart) {
    // The start under lateral moduli: the instantaneous coupled fixed
    // point of a zero-length step, the material's start forces on it.
    f = directAt(first);
    step.emplace(op.StartStep(tstate, 0.0, scheme));
    if (material != nullptr) material->ApplyStart(*step);
    applySources(first);
    effectiveSolver = &solverFor(step->EffectiveLove());
    auto [strain, info] = close();
    if (!info.converged || !response.converged) {
      throw std::runtime_error(std::format(
          "the tangent-linear coupled fixed point at the start did not "
          "converge: {}{}",
          info.Describe(options_.coupled_rtol),
          response.converged ? "" : "; its sea level solve did not"));
    }
    startInfo = info;
    op.Recycle(std::exchange(tstate, std::move(*step).Finish(std::move(strain))));
    step.reset();
  } else if (const auto f0 = directAt(first);
             !f0.empty() || material != nullptr || tideAt || outsideAt ||
             layerAt) {
    GRDSolver::GeneralisedForcing forcing;
    forcing.directLoad = f0;
    forcing.directLoadOutsideWaterBudget = outsideNow;
    forcing.solidSurfaceChange = layerNow;
    forcing.tide = tideNow;
    forcing.rotation = rotation;
    forcing.oceanFunctionOverride = overrideAt(first);
    if (material != nullptr) {
      // The start solve closes about the material's own start response.
      startResponse = material->StartResponse();
      forcing.displacementResponse = ev.Coefficients(
          op.Surface(*startResponse, Component::U), ev.toNdLength_);
      forcing.potentialResponse = ev.Coefficients(
          op.Surface(*startResponse, Component::Phi), ev.toNdPotential_);
      forcing.tangentialResponse = ev.Coefficients(
          op.Surface(*startResponse, Component::V), ev.toNdLength_);
      forcing.toroidalResponse = ev.Coefficients(
          op.SurfaceToroidal(*startResponse), ev.toNdLength_);
      // and, for the axial rotational feedback, its inertia moment
      forcing.inertiaMomentChange =
          op.InertiaMoment(*startResponse) * ev.toNdMoment_;
    }
    response =
        solverFor(op.ElasticLove()).SolveGeneralised(ev.state_, forcing, sleOptions);
    if (!response.converged) {
      throw std::runtime_error(
          "the tangent-linear elastic sea level solve did not converge");
    }
    sigma = totalLoadCoefficients(f0, response.relativeSeaLevelChange, first);
    psi = centrifugal(response);
    agree();
  }
  if (!stepStart && (options_.record || material != nullptr)) {
    op.PrimeStrain(tstate, ev.Columns(sigma, ev.toMatDensity_),
                   ev.Columns(psi, ev.toMatPotential_));
  }
  if (!stepStart && material != nullptr) {
    // The start strain carries the start response too.
    op.AddStrainCoefficients(
        *tstate.strainCoefficients, *startResponse,
        [&](int l) -> const DegreeOperator& { return op.Spheroidal(l); },
        [&](int l) -> const ToroidalOperator& { return op.Toroidal(l); });
    tstate.strain = op.coupling_.Synthesise(*tstate.strainCoefficients);
  }

  auto snapshots = std::vector<Snapshot>{};
  // the records of the inner stages of the step in hand
  auto innerStages = std::vector<Snapshot::Stage>{};
  auto record = [&](std::size_t node, GRDSolver::Response& r,
                    const Viscoelastic3D::StepInfo& info) {
    Snapshot snapshot;
    snapshot.t = times[node];
    snapshot.relativeSeaLevelChange = std::move(r.relativeSeaLevelChange);
    snapshot.displacement = std::move(r.displacement);
    snapshot.potential = std::move(r.potential);
    snapshot.tangentialPotential = std::move(r.tangentialPotential);
    snapshot.toroidalPotential = std::move(r.toroidalPotential);
    snapshot.angularVelocity = r.angularVelocity;
    snapshot.coupledIterations = info.iterations;
    snapshot.coupledResidual = info.residual;
    snapshot.solveIterations = info.solveIterations;
    snapshot.converged = info.converged && r.converged;
    snapshot.oceanFunction =
        migrating ? oceanFunctions.nodes[node] : std::vector<double>{};
    snapshot.stages = std::move(innerStages);
    innerStages.clear();
    if (options_.record) {
      snapshot.internalState = tstate;
      snapshot.loadCoefficients = sigma;
      snapshot.tideCoefficients = psi;
      // a second-order run rebuilds this run's solutions under them
      if (sourced) snapshot.sourceForces = nodeForces;
    }
    snapshots.push_back(std::move(snapshot));
  };
  record(0, response, startInfo);

  for (auto node = std::size_t{1}; node < nodes; ++node) {
    if (options_.on_node) options_.on_node(node);
    const auto dt = (times[node] - times[node - 1]) * ev.toMatTime_;
    const auto recipe = RecipeOf(scheme, dt);
    const auto refs = detail::StagesOf(recipe, dt, node);
    // the node's diagnostics, over the step's stages
    auto info = Viscoelastic3D::StepInfo{};
    // One stage of the step (StepRecipe.hpp; the step's one stage under a
    // one-stage scheme), as the forward run takes it: opened from its
    // memory state, driven, its combined fixed point closed under the
    // direction as the stage takes it and the stage's own shorelines,
    // finished.
    auto stageOf = [&](const State& from, const RecipeStage& stage,
                       double length, std::size_t index) {
      const auto& ref = refs[index];
      stageNow = ref;
      step.emplace(op.StartStep(from, length, stage.family,
                                ev.Columns(sigma, ev.toMatDensity_),
                                ev.Columns(psi, ev.toMatPotential_)));
      if (drive) step->AddMemory(drive(ref));
      if (material != nullptr) material->Apply(ref, *step);
      applySources(ref);
      effectiveSolver = &solverFor(step->EffectiveLove());
      f = directAt(ref);
      if (tideAt) tideNow = tideAt(ref);
      if (outsideAt) outsideNow = outsideAt(ref);
      if (layerAt) layerNow = layerAt(ref);
      auto [strain, stageInfo] = close();
      if (!stageInfo.converged || !response.converged) {
        throw std::runtime_error(std::format(
            "the tangent-linear coupled fixed point at node {} did not "
            "converge: {}{}",
            node, stageInfo.Describe(options_.coupled_rtol),
            response.converged ? "" : "; its sea level solve did not"));
      }
      info.iterations += stageInfo.iterations;
      info.mixedPasses += stageInfo.mixedPasses;
      info.solveIterations += stageInfo.solveIterations;
      info.residual = stageInfo.residual;
      auto next = std::move(*step).Finish(std::move(strain));
      step.reset();
      if (!ref.Last()) {
        // an inner stage's record: a second-order run about this one reads
        // the stage's state as it reads a node's
        auto inner = Snapshot::Stage{};
        inner.oceanFunction = overrideAt(ref);
        if (options_.record) {
          inner.internalState = next;
          inner.loadCoefficients = sigma;
          inner.tideCoefficients = psi;
          if (sourced) inner.sourceForces = nodeForces;
        }
        innerStages.push_back(std::move(inner));
      }
      return next;
    };
    op.Recycle(std::exchange(
        tstate,
        StepByRecipe(
            recipe, dt, tstate, stageOf,
            [&](std::span<const std::pair<double, const State*>> terms,
                const RecipeStage&) { return op.StageState(terms); },
            [&](State&& spent) { op.Recycle(std::move(spent)); })));
    record(node, response, info);
  }
  op.ReleaseScratch();
  // the method the run is by, which a second-order sweep or run reading it
  // must share (detail::CheckTangentMethod)
  for (auto& snapshot : snapshots) snapshot.tangentMethod = options_.method;
  return snapshots;
}

}  // namespace sl3dpp
