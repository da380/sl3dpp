#include "sl3dpp/TangentLinear3D.hpp"

#include <format>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/ParallelFor.hpp"

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
class ForwardCursor3D {
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

  const Viscoelastic3D::State& At(std::size_t node) {
    if (node < node_) {
      throw std::logic_error("the forward cursor moves forwards");
    }
    const auto& op = *op_;
    while (node_ < node) {
      const auto target = node_ + 1;
      if (trajectory_->Kept(target)) {
        state_ = Kept(target);
        node_ = target;
        Prime(node_);
        continue;
      }
      auto step = op.StartStep(state_, dts_[node_], scheme_, LoadsAt(node_),
                               TidesAt(node_));
      const auto load = LoadsAt(target);
      const auto tide = TidesAt(target);
      auto [strain, info] = step.Converge(
          [&](const GridCoupling::SpectralTensor& s) {
            auto solutions = step.Respond(s);
            step.AddChannels(solutions, load, tide);
            return step.StrainOf(solutions);
          },
          control_);
      if (!info.converged) {
        throw std::runtime_error(std::format(
            "rebuilding the forward state at node {} did not converge",
            target));
      }
      state_ = std::move(step).Finish(std::move(strain));
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
};

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
    detail::ParallelElements(values, op.options_.parallel, [&](std::size_t i) {
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
    detail::ParallelElements(values, op.options_.parallel, [&](std::size_t i) {
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
  if (!ditau.empty()) {
    cursor.emplace(op, *trajectory, StepSizes(times), scheme, Control(),
                   LoadColumnsOf(), TideColumnsOf());
    drive = [&](std::size_t node, double dt) {
      const State prev = cursor->At(node - 1);
      const auto& next = cursor->At(node);
      return FirstOrderDrive(op, options_.method, scheme, dt, prev, next, ditau);
    };
  }
  return March(times, oceanFunctions, directAt, drive);
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
  const Direct none = [](std::size_t) { return std::vector<double>{}; };
  return March(times, oceanFunctions, none, drive);
}

std::vector<Snapshot> TangentLinearSolver3D::March(
    std::span<const double> times,
    std::span<const std::vector<double>> oceanFunctions,
    const Direct& directAt, const Drive& drive) const {
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
  if (const auto f0 = directAt(0); !f0.empty()) {
    SeaLevelSolver::GeneralisedForcing forcing;
    forcing.directLoad = f0;
    forcing.rotation = rotation;
    forcing.oceanFunctionOverride = overrideAt(0);
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
  if (options_.record) {
    op.PrimeStrain(tstate, ev.Columns(sigma, ev.toMatDensity_),
                   ev.Columns(psi, ev.toMatPotential_));
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
    snapshot.converged = info.converged && r.converged;
    snapshot.oceanFunction = overrideAt(node);
    if (options_.record) {
      snapshot.internalState = tstate;
      snapshot.loadCoefficients = sigma;
      snapshot.tideCoefficients = psi;
    }
    snapshots.push_back(std::move(snapshot));
  };
  record(0, response, {});

  for (auto node = std::size_t{1}; node < nodes; ++node) {
    if (options_.on_node) options_.on_node(node);
    const auto dt = (times[node] - times[node - 1]) * ev.toMatTime_;
    auto step = op.StartStep(tstate, dt, scheme,
                             ev.Columns(sigma, ev.toMatDensity_),
                             ev.Columns(psi, ev.toMatPotential_));
    if (drive) step.AddMemory(drive(node, dt));
    const auto& effectiveSolver = solverFor(step.EffectiveLove());
    const auto f = directAt(node);
    auto [strain, info] = step.Converge(
        [&](const GridCoupling::SpectralTensor& s) {
          auto responses = step.Respond(s);
          SeaLevelSolver::GeneralisedForcing coupled;
          coupled.directLoad = f;
          coupled.displacementResponse = ev.Coefficients(
              op.Surface(responses, Component::U), ev.toNdLength_);
          coupled.potentialResponse = ev.Coefficients(
              op.Surface(responses, Component::Phi), ev.toNdPotential_);
          coupled.inertiaMomentChange =
              step.InertiaMoment(responses) * ev.toNdMoment_;
          coupled.rotation = rotation;
          coupled.oceanFunctionOverride = overrideAt(node);
          response =
              effectiveSolver.SolveGeneralised(ev.state_, coupled, sleOptions);
          sigma = totalLoadCoefficients(f, response.seaLevelChange, node);
          psi = centrifugal(response);
          agree();
          step.AddChannels(responses, ev.Columns(sigma, ev.toMatDensity_),
                           ev.Columns(psi, ev.toMatPotential_));
          return step.StrainOf(responses);
        },
        {.max_iterations = options_.max_coupled_iterations,
         .rtol = options_.coupled_rtol,
         .anderson_memory = options_.anderson_memory,
         .anderson_rate = options_.anderson_rate});
    if (!info.converged || !response.converged) {
      throw std::runtime_error(std::format(
          "the tangent-linear coupled fixed point at node {} did not "
          "converge (residual {})",
          node, info.residual));
    }
    op.Recycle(std::exchange(tstate, std::move(step).Finish(std::move(strain))));
    record(node, response, info);
  }
  op.ReleaseScratch();
  return snapshots;
}

}  // namespace sl3dpp
