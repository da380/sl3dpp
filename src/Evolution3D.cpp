#include "sl3dpp/Evolution3D.hpp"

#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/Shorelines.hpp"

namespace sl3dpp {

SeaLevelEvolution3D::SeaLevelEvolution3D(const Viscoelastic3D& op,
                                         const SeaLevelSolver& solver,
                                         SeaLevelSolver::State state,
                                         Options options)
    : op_(&op), solver_(&solver), state_(std::move(state)), options_(options) {
  if (op.LMax() != solver.Grid().LMax()) {
    throw std::invalid_argument(std::format(
        "the operator reaches degree {}, the solver's grid degree {}",
        op.LMax(), solver.Grid().LMax()));
  }
  if (options.scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "the coupled loop steps implicitly: ExponentialTrapezoid or "
        "BackwardEuler");
  }
  const auto& material = op.GetRheology().BaseMaterial().ModelScales();
  const auto& sle = solver.SolverScales();
  auto ratio = [&](Dimensions dims) {
    return material.Factor(dims) / sle.Factor(dims);
  };
  toNdLength_ = ratio(Dimensions{.length = 1});
  toNdPotential_ = ratio(PotentialDimensions);
  toNdMoment_ = ratio(Dimensions{.mass = 1, .length = 2});
  toMatDensity_ = 1.0 / ratio(Dimensions{.mass = 1, .length = -2});
  toMatPotential_ = 1.0 / ratio(PotentialDimensions);
  toMatTime_ = 1.0 / ratio(Dimensions{.time = 1});
}

Viscoelastic3D::LoadCallable3D SeaLevelEvolution3D::Columns(
    const std::vector<Complex>& coefficients, double factor) const {
  return [coefficients, factor](int l,
                                double) -> std::optional<std::vector<Complex>> {
    auto columns = std::vector<Complex>(l + 1);
    for (auto m = 0; m <= l; ++m) {
      columns[m] = factor * coefficients[SurfaceGrid::Index(l, m)];
    }
    return columns;
  };
}

std::vector<SeaLevelEvolution3D::Complex> SeaLevelEvolution3D::Coefficients(
    const std::map<int, std::vector<Complex>>& perDegree, double factor) const {
  auto out = solver_->Grid().ZeroCoefficients();
  for (const auto& [l, columns] : perDegree) {
    for (auto m = 0; m <= l; ++m) {
      out[SurfaceGrid::Index(l, m)] = factor * columns[m];
    }
  }
  return out;
}

namespace {

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

}  // namespace

void SeaLevelEvolution3D::SetBackgroundState(SeaLevelSolver::State state) {
  const auto fieldSize = solver_->Grid().FieldSize();
  if (state.iceThickness.size() != fieldSize ||
      state.seaLevel.size() != fieldSize ||
      state.oceanFunction.size() != fieldSize ||
      (!state.forcedLand.empty() && state.forcedLand.size() != fieldSize)) {
    throw std::invalid_argument("state fields must live on the solver's grid");
  }
  state_ = std::move(state);
}

std::vector<SeaLevelEvolution3D::Snapshot> SeaLevelEvolution3D::Run(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceLoad) {
  CheckTimes(times);
  return March(times, &iceLoad, nullptr, false);
}

std::vector<SeaLevelEvolution3D::Snapshot> SeaLevelEvolution3D::RunMigrating(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness) {
  CheckTimes(times);
  return March(times, nullptr, &iceThickness, false);
}

void SeaLevelEvolution3D::Restore(const Checkpoint& from,
                                  std::span<const double> times) {
  CheckTimes(times);
  if (times[0] != from.t) {
    throw std::invalid_argument(
        "the first time of a resumed run is the checkpoint's own node");
  }
  vstate_ = op_->ZeroState();
  vstate_.t = from.t * toMatTime_;
  vstate_.m = from.m;
  vstate_.strain.reset();
  vstate_.strainCoefficients = from.strainCoefficients;
  sigma_ = from.loadCoefficients;
  psi_ = from.tideCoefficients;
}

std::vector<SeaLevelEvolution3D::Snapshot> SeaLevelEvolution3D::Resume(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceLoad) {
  Restore(from, times);
  return March(times, &iceLoad, nullptr, true);
}

std::vector<SeaLevelEvolution3D::Snapshot> SeaLevelEvolution3D::ResumeMigrating(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness) {
  Restore(from, times);
  return March(times, nullptr, &iceThickness, true);
}

std::vector<SeaLevelEvolution3D::Snapshot> SeaLevelEvolution3D::March(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>* iceLoad,
    const std::function<std::vector<double>(double)>* iceThickness,
    bool resumed) {
  const auto& op = *op_;
  const auto& solver = *solver_;
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const bool migrating = iceThickness != nullptr;

  if (migrating && !resumed) {
    // The shorelines follow flotation from the first node, so the
    // background must be the state there, flotation's own C included.
    detail::CheckMigratingStart(solver, state_, (*iceThickness)(times[0]));
  }

  if (options_.rotation == RotationalFeedback::Full) {
    const auto kHyd = FluidTidalNumber(op.GetRheology().BaseMaterial(), 2);
    if (solver.TransverseMarginFor(kHyd) <= 0.0) {
      throw std::invalid_argument(
          "the transverse rotational feedback is unstable against the "
          "model's hydrostatic fluid limit; use WithHydrostaticMoments, or "
          "rotation = Axial or None");
    }
  }

  SeaLevelSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;
  auto nonlinearOptions = sleOptions;
  nonlinearOptions.max_iterations = options_.max_nonlinear_iterations;

  // Flavour SLE solvers under the caller's parameters, keyed by the
  // (operator-cached, address-stable) Love table.
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

  const auto stride = options_.record_stride;
  const auto lastNode = times.size() - 1;
  if (!resumed) {
    vstate_ = op.ZeroState();
    vstate_.t = times[0] * toMatTime_;
  }

  auto checkpoint = [&](std::size_t node) {
    const auto every = options_.checkpoint_stride;
    if (every <= 0 || !options_.on_checkpoint) return;
    if (node % static_cast<std::size_t>(every) != 0 && node != lastNode) {
      return;
    }
    options_.on_checkpoint(
        node, Checkpoint{.t = times[node],
                         .m = vstate_.m,
                         .strainCoefficients = vstate_.strainCoefficients,
                         .loadCoefficients = sigma_,
                         .tideCoefficients = psi_});
  };

  auto totalLoadCoefficients = [&](const std::vector<double>& loadGrid,
                                   const std::vector<double>& slc) {
    auto total = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      total[i] = loadGrid[i] +
                 solver.WaterDensity() * state_.oceanFunction[i] * slc[i];
    }
    return grid.Expand(total);
  };

  auto loadGrid = std::vector<double>{};
  auto response = SeaLevelSolver::Response{};
  auto ocean = std::vector<double>{};  // the node's C, migrating only
  if (!resumed && migrating) {
    // The background is the state at times[0]: the run starts undisturbed.
    response.seaLevelChange.assign(fieldSize, 0.0);
    response.displacement.assign(fieldSize, 0.0);
    response.potential.assign(fieldSize, 0.0);
    sigma_ = grid.ZeroCoefficients();
    psi_ = grid.ZeroCoefficients();
    ocean = state_.oceanFunction;
  } else if (!resumed) {
    // The instantaneous elastic response at the start.
    loadGrid = (*iceLoad)(times[0]);
    const auto& elastic = solverFor(op.ElasticLove());
    SeaLevelSolver::GeneralisedForcing forcing;
    forcing.directLoad = loadGrid;
    forcing.rotation = options_.rotation;
    response = elastic.SolveGeneralised(state_, forcing, sleOptions);
    sigma_ = totalLoadCoefficients(loadGrid, response.seaLevelChange);
    psi_ = options_.rotation == RotationalFeedback::None
               ? grid.ZeroCoefficients()
               : solver.CentrifugalCoefficients(response.angularVelocity);
  }
  if (stride > 0) op.PrimeStrain(vstate_, Columns(sigma_, toMatDensity_),
                                 Columns(psi_, toMatPotential_));

  auto snapshots = std::vector<Snapshot>{};
  auto record = [&](double t, SeaLevelSolver::Response& r, int iterations,
                    double residual, bool converged, bool keepState) {
    Snapshot snapshot;
    snapshot.t = t;
    snapshot.seaLevelChange = std::move(r.seaLevelChange);
    snapshot.displacement = std::move(r.displacement);
    snapshot.potential = std::move(r.potential);
    snapshot.angularVelocity = r.angularVelocity;
    snapshot.coupledIterations = iterations;
    snapshot.coupledResidual = residual;
    snapshot.converged = converged && r.converged;
    snapshot.oceanFunction = std::move(ocean);
    // The internal variables, and the strain only on request: an adjoint
    // otherwise regenerates it by one elastic solve under the recorded
    // loads. With on_state, the kept state is handed out instead.
    auto kept = std::optional<Viscoelastic3D::State>{};
    if (keepState && options_.record_strain) {
      kept = vstate_;
    } else if (keepState) {
      auto trimmed = Viscoelastic3D::State{};
      trimmed.t = vstate_.t;
      trimmed.m = vstate_.m;
      kept = std::move(trimmed);
    }
    if (kept && !options_.on_state) snapshot.internalState = std::move(kept);
    if (stride > 0) {
      snapshot.loadCoefficients = sigma_;
      snapshot.tideCoefficients = psi_;
    }
    snapshots.push_back(std::move(snapshot));
    const auto node = snapshots.size() - (resumed ? 0 : 1);
    if (options_.on_snapshot) options_.on_snapshot(node, snapshots.back());
    if (kept && options_.on_state) options_.on_state(node, std::move(*kept));
  };
  if (!resumed) {
    record(times[0], response, 0, 0.0, true, stride > 0);
    checkpoint(0);
  }

  for (auto node = std::size_t{1}; node < times.size(); ++node) {
    const auto dt = (times[node] - times[node - 1]) * toMatTime_;
    auto step = op.StartStep(vstate_, dt, options_.scheme,
                             Columns(sigma_, toMatDensity_),
                             Columns(psi_, toMatPotential_));
    const auto& effectiveSolver = solverFor(step.EffectiveLove());
    // Under migrating shorelines each pass solves the nonlinear equation
    // from the background, the memory and lateral responses entering
    // through the response channels, so the ocean function follows the
    // flotation condition inside the fixed point.
    auto moving = SeaLevelSolver::NonlinearForcing{};
    if (migrating) {
      moving.iceThicknessChange = (*iceThickness)(times[node]);
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        moving.iceThicknessChange[i] -= state_.iceThickness[i];
      }
      moving.rotation = options_.rotation;
    } else {
      loadGrid = (*iceLoad)(times[node]);
    }

    // The combined fixed point: strain, ocean load and rotation together.
    auto [strain, info] = step.Converge(
        [&](const GridCoupling::SpectralTensor& s) {
          auto responses = step.Respond(s);
          auto displacement =
              Coefficients(op.Surface(responses, Component::U), toNdLength_);
          auto potential = Coefficients(op.Surface(responses, Component::Phi),
                                        toNdPotential_);
          const auto moment = step.InertiaMoment(responses) * toNdMoment_;
          if (migrating) {
            moving.displacementResponse = std::move(displacement);
            moving.potentialResponse = std::move(potential);
            moving.inertiaMomentChange = moment;
            auto solved = effectiveSolver.SolveNonlinear(state_, moving,
                                                         nonlinearOptions);
            response = std::move(solved.response);
            sigma_ = detail::MigratedLoadCoefficients(solver, state_,
                                                      solved.finalState);
            ocean = std::move(solved.finalState.oceanFunction);
          } else {
            SeaLevelSolver::GeneralisedForcing coupled;
            coupled.directLoad = loadGrid;
            coupled.displacementResponse = std::move(displacement);
            coupled.potentialResponse = std::move(potential);
            coupled.inertiaMomentChange = moment;
            coupled.rotation = options_.rotation;
            response =
                effectiveSolver.SolveGeneralised(state_, coupled, sleOptions);
            sigma_ = totalLoadCoefficients(loadGrid, response.seaLevelChange);
          }
          psi_ = options_.rotation == RotationalFeedback::None
                     ? grid.ZeroCoefficients()
                     : solver.CentrifugalCoefficients(response.angularVelocity);
          step.AddChannels(responses, Columns(sigma_, toMatDensity_),
                           Columns(psi_, toMatPotential_));
          return step.StrainOf(responses);
        },
        {.max_iterations = options_.max_coupled_iterations,
         .rtol = options_.coupled_rtol,
         .anderson_memory = options_.anderson_memory,
         .anderson_rate = options_.anderson_rate});
    // the replaced state's tensors go to the next StartStep
    op.Recycle(
        std::exchange(vstate_, std::move(step).Finish(std::move(strain))));
    const bool keep = stride > 0 && (node % stride == 0 || node == lastNode);
    record(times[node], response, info.iterations, info.residual,
           info.converged, keep);
    checkpoint(node);
  }
  op.ReleaseScratch();
  return snapshots;
}

}  // namespace sl3dpp
