#include "sl3dpp/TangentLinear3D.hpp"

#include <complex>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/Parallel.hpp"

namespace sl3dpp {

namespace {

using Complex = std::complex<double>;
using Snapshot = SeaLevelEvolution3D::Snapshot;
using State = Viscoelastic3D::State;

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
        tideOf_(std::move(tideOf)) {
    state_ = Kept(0);
    Prime(0);
  }

  /// The forward state at a node, reconstructing as needed; the node
  /// before the current one is served from the kept previous state, so
  /// the drives of one step may ask for (node - 1, node) in any order.
  /// The references stay valid until the cursor advances.
  const Viscoelastic3D::State& At(std::size_t node) {
    if (node + 1 == node_ && previous_.has_value()) return *previous_;
    if (node < node_) {
      throw std::logic_error("the forward cursor moves forwards");
    }
    const auto& op = *op_;
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
      auto step = op.StartStep(state_, dts_[node_], scheme_, LoadsAt(node_),
                               TidesAt(node_));
      auto [strain, info] = step.Close(LoadsAt(target), TidesAt(target), control_);
      if (!info.converged) {
        throw std::runtime_error(std::format(
            "rebuilding the forward state at node {} did not converge",
            target));
      }
      auto next = std::move(step).Finish(std::move(strain));
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
  std::size_t node_ = 0;
  Viscoelastic3D::State state_;
  std::optional<Viscoelastic3D::State> previous_;
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

TangentLinearSolver3D::TangentLinearSolver3D(
    const SeaLevelEvolution3D& evolution, Options options)
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
  const auto& d0 = StrainOf(prev, "the trajectory's states");
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
  virtual void Apply(std::size_t node, double dt, Scheme scheme,
                     Viscoelastic3D::OpenStep& step) const = 0;
};

struct TangentLinearSolver3D::MaterialDrive3D
    : TangentLinearSolver3D::MaterialForcing3D {
  using Solutions = Viscoelastic3D::Solutions;
  using Complex = std::complex<double>;
  using Tensor = GridCoupling::TensorField;
  const SeaLevelEvolution3D* ev = nullptr;
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
  Base BaseAt(std::size_t node) const {
    const auto& op = *ev->op_;
    const auto& state = cursor->At(node);
    Base base;
    const auto load = ev->Columns(trajectory->loads[node], ev->toMatDensity_);
    const auto tide = ev->Columns(trajectory->tides[node], ev->toMatPotential_);
    if (!direction.kappa.empty() || !state.strain) {
      base.x = op.SolveElastic(state, load, tide);
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
    const auto base = BaseAt(0);
    return UnrelaxedResponse(*base.strain, base.x ? &*base.x : nullptr, nullptr);
  }

  void ApplyStart(Viscoelastic3D::OpenStep& step) const override {
    const auto base = BaseAt(0);
    auto [analysed, bulk] =
        Forces(*base.strain, base.x ? &*base.x : nullptr, direction.mu, nullptr);
    step.AddForce(analysed, bulk);
  }

  /// The forces of the step ending at `node`, on the opened step: the
  /// effective system's variation on the full base solution — delta
  /// mu_eff(x) = delta mu(x) (1 - sum_k f_k beta_k(x)) with the step's
  /// own lateral weights — and the memory force of the decayed memory
  /// with delta mu_k = f_k delta mu.
  void Apply(std::size_t node, double dt, Scheme scheme,
             Viscoelastic3D::OpenStep& step) const override {
    const auto& op = *ev->op_;
    auto shear = std::vector<double>{};
    auto memoryForce = std::optional<Tensor>{};
    if (!direction.mu.empty()) {
      // The BASE step's decayed memory m~ = e m + alpha d_0 (the step
      // given is the tangent's, whose memory is the tangent's own),
      // opened on the base state before the node — the weights are the
      // step's, lateral alike.
      const auto prev = cursor->At(node - 1);
      const auto baseStep = op.StartStep(
          prev, dt, scheme,
          ev->Columns(trajectory->loads[node - 1], ev->toMatDensity_),
          ev->Columns(trajectory->tides[node - 1], ev->toMatPotential_));
      shear = EffectiveShear(baseStep);
      memoryForce = MemoryVariation(baseStep.memory_);
    }
    const auto base = BaseAt(node);
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
  const SeaLevelEvolution3D* ev = nullptr;
  AdjointMethod method = AdjointMethod::Continuous;
  const ForwardTrajectory3D* trajectory = nullptr;
  ForwardCursor3D* cursor = nullptr;
  std::vector<Part> parts;

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
  Solutions TangentSolution(const Part& part, std::size_t node,
                            const Tensor& baseStrain, const Solutions* baseX,
                            const std::vector<Tensor>* baseMemory) const {
    const auto& op = *ev->op_;
    const auto& snap = part.b[node];
    auto x = op.SolveElastic(*snap.internalState, Loads(snap.loadCoefficients),
                             Tides(snap.tideCoefficients));
    if (part.bMaterial) {
      Add(x, part.bMaterial->UnrelaxedResponse(baseStrain, baseX, baseMemory));
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
    const auto& op = *ev->op_;
    const auto& base0 = cursor->At(0);
    const auto x0 = op.SolveElastic(base0, Loads(trajectory->loads[0]),
                                    Tides(trajectory->tides[0]));
    auto own = std::optional<Tensor>{};
    const auto& strain0 = BaseStrain(base0, x0, own);
    auto total = Solutions{};
    for (const auto& part : parts) {
      const auto xb = TangentSolution(part, 0, strain0, &x0, nullptr);
      Add(total, part.a->UnrelaxedResponse(*part.b[0].internalState->strain,
                                           &xb, nullptr));
    }
    return total;
  }

  void ApplyStart(Viscoelastic3D::OpenStep& step) const override {
    const auto& op = *ev->op_;
    const auto& base0 = cursor->At(0);
    const auto x0 = op.SolveElastic(base0, Loads(trajectory->loads[0]),
                                    Tides(trajectory->tides[0]));
    auto own = std::optional<Tensor>{};
    const auto& strain0 = BaseStrain(base0, x0, own);
    for (const auto& part : parts) {
      const auto xb = TangentSolution(part, 0, strain0, &x0, nullptr);
      auto [analysed, bulk] = part.a->Forces(*part.b[0].internalState->strain,
                                             &xb, part.a->direction.mu, nullptr);
      step.AddForce(analysed, bulk);
    }
  }

  void Apply(std::size_t node, double dt, Scheme scheme,
             Viscoelastic3D::OpenStep& step) const override {
    const auto& op = *ev->op_;
    const Viscoelastic3D::State basePrev = cursor->At(node - 1);
    const auto& baseN = cursor->At(node);
    const auto xN = op.SolveElastic(baseN, Loads(trajectory->loads[node]),
                                    Tides(trajectory->tides[node]));
    auto own = std::optional<Tensor>{};
    const auto& strainN = BaseStrain(baseN, xN, own);
    // The base step before the node, for its lateral weights.
    const auto baseStep =
        op.StartStep(basePrev, dt, scheme, Loads(trajectory->loads[node - 1]),
                     Tides(trajectory->tides[node - 1]));
    for (const auto& part : parts) {
      const auto xb = TangentSolution(part, node, strainN, &xN, &baseN.m);
      const auto& prev = part.b[node - 1];
      auto tstep = op.StartStep(*prev.internalState, dt, scheme,
                                Loads(prev.loadCoefficients),
                                Tides(prev.tideCoefficients));
      if (!part.bDitau.empty()) {
        tstep.AddMemory(FirstOrderDrive(op, method, scheme, dt, basePrev, baseN,
                                        part.bDitau));
      }
      const auto shear = part.a->EffectiveShear(baseStep);
      auto memoryForce = std::optional<Tensor>{};
      if (!part.a->direction.mu.empty()) {
        memoryForce = part.a->MemoryVariation(tstep.memory_);
      }
      auto [analysed, extra] =
          part.a->Forces(*part.b[node].internalState->strain, &xb, shear,
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
  if (op.GetOptions().memory_points != MemoryPointSet::Gll) {
    throw std::invalid_argument(
        "the material tangent needs the memory nodes at the GLL points "
        "(Viscoelastic3D::Options::memory_points)");
  }
  if (ev.options_.rotation != RotationalFeedback::None) {
    throw std::invalid_argument(
        "the material tangent takes no rotational feedback");
  }
  auto material = std::make_shared<MaterialDrive3D>();
  material->ev = &ev;
  material->direction = direction;
  material->trajectory = trajectory;
  material->cursor = cursor;
  const auto& base = op.GetRheology().BaseMaterial();
  const auto ngll = base.Mesh().NGll();
  const auto [kappaU, muU] = base.Moduli().KappaMu();
  for (auto k = std::size_t{0}; k < K; ++k) {
    auto f = std::vector<double>(coupling.NodeCount(), 0.0);
    for (auto n = 0; n < coupling.NodeCount(); ++n) {
      const auto full =
          static_cast<std::size_t>(coupling.ElementOfNode(n)) * ngll +
          coupling.NodeInElementOf(n);
      f[n] = muU[full] > 0.0 ? op.muBranch_[k][n] / muU[full] : 0.0;
    }
    material->fractions.push_back(std::move(f));
  }
  return material;
}

std::vector<Snapshot> TangentLinearSolver3D::Run(
    std::span<const double> times, const TangentPerturbation3D& direction,
    const ForwardTrajectory3D* trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto& solver = *ev.solver_;
  const auto fieldSize = solver.Grid().FieldSize();
  const auto scheme = ev.options_.scheme;
  const auto nodes = times.size();
  const bool migrating = !oceanFunctions.empty();
  const auto K = op.GetRheology().BranchCount();
  const auto values = op.Coupling().GridValueCount();

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
  auto oceanAt = [&](std::size_t node) -> const std::vector<double>& {
    return migrating ? oceanFunctions[node] : C0;
  };
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

  auto cursor = std::optional<ForwardCursor3D>{};
  Drive drive{};
  if (!ditau.empty() || hasMaterial) {
    cursor.emplace(op, *trajectory, StepSizes(times), scheme, Control(),
                   LoadColumnsOf(), TideColumnsOf());
  }
  if (!ditau.empty()) {
    drive = [&](std::size_t node, double dt) {
      const State prev = cursor->At(node - 1);
      const auto& next = cursor->At(node);
      return FirstOrderDrive(op, options_.method, scheme, dt, prev, next, ditau);
    };
  }
  std::shared_ptr<MaterialDrive3D> material;
  if (hasMaterial) {
    material = MakeMaterialDrive3D(direction.material, trajectory, &*cursor);
  }
  return March(times, oceanFunctions, directAt, drive, material.get());
}

std::vector<Snapshot> TangentLinearSolver3D::RunSecondOrder(
    std::span<const double> times, std::span<const Snapshot> tangentU,
    std::span<const Snapshot> tangentV,
    const SecondOrderDirection3D& direction,
    const ForwardTrajectory3D* trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  const auto& ev = *evolution_;
  const auto& op = *ev.op_;
  const auto scheme = ev.options_.scheme;
  const auto nodes = times.size();
  const bool discrete = options_.method == AdjointMethod::Discrete;
  const auto K = op.GetRheology().BranchCount();
  const auto values = op.Coupling().GridValueCount();

  CheckTimes(times);
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
  if (!parts.empty()) {
    for (const auto* run : {&tangentU, &tangentV}) {
      if (run->size() != nodes) {
        throw std::invalid_argument(
            "the tangent-linear runs must give one snapshot per time node");
      }
      for (const auto& snap : *run) {
        if (!snap.internalState.has_value() ||
            !snap.internalState->strain.has_value()) {
          throw std::invalid_argument(
              "the tangent-linear runs must be recorded at every node "
              "(TangentLinearSolver3D::Options::record)");
        }
      }
    }
    CheckTrajectory(trajectory, nodes);
    cursor.emplace(op, *trajectory, StepSizes(times), scheme, Control(),
                   LoadColumnsOf(), TideColumnsOf());
    drive = [&, both](std::size_t node, double dt) {
      const State basePrev = cursor->At(node - 1);
      const auto& baseNext = cursor->At(node);
      auto total = std::vector<GridCoupling::TensorField>{};
      for (const auto& part : parts) {
        if (part.along.empty()) {
          Accumulate(total, FirstOrderDrive(op, options_.method, scheme, dt,
                                            basePrev, baseNext, *part.ditau));
        } else {
          Accumulate(total,
                     FirstOrderDrive(op, options_.method, scheme, dt,
                                     *part.along[node - 1].internalState,
                                     *part.along[node].internalState,
                                     *part.ditau));
        }
      }
      if (both && discrete) {
        Accumulate(total, SecondWeightDrive(op, scheme, dt, basePrev, baseNext,
                                            direction.itauU, direction.itauV));
      }
      return total;
    };
  }
  // The moduli: a's direction on b's tangent trajectory, for both orders.
  const bool materialU = !direction.materialU.Empty();
  const bool materialV = !direction.materialV.Empty();
  std::shared_ptr<MaterialSecondDrive3D> material;
  if (materialU || materialV) {
    for (const auto* run : {&tangentU, &tangentV}) {
      if (run->size() != nodes) {
        throw std::invalid_argument(
            "the tangent-linear runs must give one snapshot per time node");
      }
      for (const auto& snap : *run) {
        if (!snap.internalState.has_value() ||
            !snap.internalState->strain.has_value()) {
          throw std::invalid_argument(
              "the tangent-linear runs must be recorded at every node "
              "(TangentLinearSolver3D::Options::record)");
        }
      }
    }
    CheckTrajectory(trajectory, nodes);
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
  const Direct none = [](std::size_t) { return std::vector<double>{}; };
  return March(times, oceanFunctions, none, drive, material.get());
}

std::vector<Snapshot> TangentLinearSolver3D::March(
    std::span<const double> times,
    std::span<const std::vector<double>> oceanFunctions,
    const Direct& directAt, const Drive& drive,
    const MaterialForcing3D* material) const {
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
    const auto kHyd = FluidTidalNumber(op.GetRheology().BaseMaterial(), 2);
    if (solver.TransverseMarginFor(kHyd) <= 0.0) {
      throw std::invalid_argument(
          "the transverse rotational feedback is unstable; see "
          "SeaLevelEvolution3D::Run");
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
  auto flavours = std::map<const LoveNumbers*, SeaLevelSolver>{};
  auto solverFor = [&](const LoveNumbers& table) -> const SeaLevelSolver& {
    auto found = flavours.find(&table);
    if (found == flavours.end()) {
      found =
          flavours
              .emplace(&table, SeaLevelSolver(grid, table, solver.Parameters()))
              .first;
    }
    return found->second;
  };

  // Distributed, every rank solves the sea level equation alike and the
  // first rank's answers serve all (SeaLevelEvolution3D::March).
  const auto& collectives = op.GetOptions().collectives;
  auto sigma = grid.ZeroCoefficients();
  auto psi = grid.ZeroCoefficients();
  auto response = SeaLevelSolver::Response{};
  auto agree = [&] {
    if (!collectives.Distributed()) return;
    for (auto* coefficients : {&sigma, &psi}) {
      collectives.share(
          std::span(reinterpret_cast<double*>(coefficients->data()),
                    2 * coefficients->size()));
    }
    for (auto* field : {&response.seaLevelChange, &response.displacement,
                        &response.potential}) {
      collectives.share(*field);
    }
    collectives.share(response.angularVelocity);
    auto converged = response.converged ? 1.0 : 0.0;
    collectives.share(std::span(&converged, 1));
    response.converged = converged != 0.0;
  };

  auto tstate = op.ZeroState();
  tstate.t = times[0] * ev.toMatTime_;
  response.seaLevelChange.assign(fieldSize, 0.0);
  response.displacement.assign(fieldSize, 0.0);
  response.potential.assign(fieldSize, 0.0);
  // The coupled pass of a step, shared by the steps and, under lateral
  // moduli, the start as a zero-length step.
  auto step = std::optional<Viscoelastic3D::OpenStep>{};
  const SeaLevelSolver* effectiveSolver = nullptr;
  auto f = std::vector<double>{};
  auto node = std::size_t{0};
  auto closure = [&](const Viscoelastic3D::Solutions& responses) {
    SeaLevelSolver::GeneralisedForcing coupled;
    coupled.directLoad = f;
    coupled.displacementResponse =
        ev.Coefficients(op.Surface(responses, Component::U), ev.toNdLength_);
    coupled.potentialResponse =
        ev.Coefficients(op.Surface(responses, Component::Phi), ev.toNdPotential_);
    coupled.inertiaMomentChange = step->InertiaMoment(responses) * ev.toNdMoment_;
    coupled.rotation = rotation;
    coupled.oceanFunctionOverride = overrideAt(node);
    response = effectiveSolver->SolveGeneralised(ev.state_, coupled, sleOptions);
    sigma = totalLoadCoefficients(f, response.seaLevelChange, node);
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
  if (lateral) {
    // The start under lateral moduli: the instantaneous coupled fixed
    // point of a zero-length step, the material's start forces on it.
    f = directAt(0);
    step.emplace(op.StartStep(tstate, 0.0, scheme));
    if (material != nullptr) material->ApplyStart(*step);
    effectiveSolver = &solverFor(step->EffectiveLove());
    auto [strain, info] = close();
    if (!info.converged || !response.converged) {
      throw std::runtime_error(std::format(
          "the tangent-linear coupled fixed point at the start did not "
          "converge (residual {})",
          info.residual));
    }
    startInfo = info;
    op.Recycle(std::exchange(tstate, std::move(*step).Finish(std::move(strain))));
    step.reset();
  } else if (const auto f0 = directAt(0); !f0.empty() || material != nullptr) {
    SeaLevelSolver::GeneralisedForcing forcing;
    forcing.directLoad = f0;
    forcing.rotation = rotation;
    forcing.oceanFunctionOverride = overrideAt(0);
    if (material != nullptr) {
      // The start solve closes about the material's own start response.
      startResponse = material->StartResponse();
      forcing.displacementResponse = ev.Coefficients(
          op.Surface(*startResponse, Component::U), ev.toNdLength_);
      forcing.potentialResponse = ev.Coefficients(
          op.Surface(*startResponse, Component::Phi), ev.toNdPotential_);
    }
    response =
        solverFor(op.ElasticLove()).SolveGeneralised(ev.state_, forcing, sleOptions);
    if (!response.converged) {
      throw std::runtime_error(
          "the tangent-linear elastic sea level solve did not converge");
    }
    sigma = totalLoadCoefficients(f0, response.seaLevelChange, 0);
    psi = centrifugal(response);
    agree();
  }
  if (!lateral && (options_.record || material != nullptr)) {
    op.PrimeStrain(tstate, ev.Columns(sigma, ev.toMatDensity_),
                   ev.Columns(psi, ev.toMatPotential_));
  }
  if (!lateral && material != nullptr) {
    // The start strain carries the start response too.
    op.AddStrainCoefficients(
        *tstate.strainCoefficients, *startResponse,
        [&](int l) -> const DegreeOperator& { return op.Spheroidal(l); },
        [&](int l) -> const ToroidalOperator& { return op.Toroidal(l); });
    tstate.strain = op.coupling_.Synthesise(*tstate.strainCoefficients);
  }

  auto snapshots = std::vector<Snapshot>{};
  auto record = [&](std::size_t node, SeaLevelSolver::Response& r,
                    const Viscoelastic3D::StepInfo& info) {
    Snapshot snapshot;
    snapshot.t = times[node];
    snapshot.seaLevelChange = std::move(r.seaLevelChange);
    snapshot.displacement = std::move(r.displacement);
    snapshot.potential = std::move(r.potential);
    snapshot.angularVelocity = r.angularVelocity;
    snapshot.coupledIterations = info.iterations;
    snapshot.coupledResidual = info.residual;
    snapshot.solveIterations = info.solveIterations;
    snapshot.converged = info.converged && r.converged;
    snapshot.oceanFunction = overrideAt(node);
    if (options_.record) {
      snapshot.internalState = tstate;
      snapshot.loadCoefficients = sigma;
      snapshot.tideCoefficients = psi;
    }
    snapshots.push_back(std::move(snapshot));
  };
  record(0, response, startInfo);

  for (node = 1; node < nodes; ++node) {
    if (options_.on_node) options_.on_node(node);
    const auto dt = (times[node] - times[node - 1]) * ev.toMatTime_;
    step.emplace(op.StartStep(tstate, dt, scheme,
                              ev.Columns(sigma, ev.toMatDensity_),
                              ev.Columns(psi, ev.toMatPotential_)));
    if (drive) step->AddMemory(drive(node, dt));
    if (material != nullptr) material->Apply(node, dt, scheme, *step);
    effectiveSolver = &solverFor(step->EffectiveLove());
    f = directAt(node);
    auto [strain, info] = close();
    if (!info.converged || !response.converged) {
      throw std::runtime_error(std::format(
          "the tangent-linear coupled fixed point at node {} did not "
          "converge (residual {})",
          node, info.residual));
    }
    op.Recycle(std::exchange(tstate, std::move(*step).Finish(std::move(strain))));
    step.reset();
    record(node, response, info);
  }
  op.ReleaseScratch();
  return snapshots;
}

}  // namespace sl3dpp
