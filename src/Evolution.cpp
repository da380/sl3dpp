#include "sl3dpp/Evolution.hpp"

#include <complex>
#include <format>
#include <optional>
#include <stdexcept>

#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/ToroidalViscoelastic.hpp"
#include "sl3dpp/detail/InteriorSources.hpp"
#include "sl3dpp/detail/Shorelines.hpp"

namespace sl3dpp {

Evolution::Evolution(const ViscoelasticOperator& op,
                                     const GRDSolver& solver,
                                     GRDSolver::State state,
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
  const auto& material = op.BaseMaterial().ModelScales();
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

LoadCallable Evolution::Columns(
    const std::vector<std::complex<double>>& coefficients,
    double factor) const {
  return [coefficients, factor](int l,
                                double) -> std::optional<std::vector<double>> {
    auto columns = std::vector<double>(2 * (l + 1));
    for (auto m = 0; m <= l; ++m) {
      const auto c = coefficients[SurfaceGrid::Index(l, m)];
      columns[m] = factor * c.real();
      columns[l + 1 + m] = factor * c.imag();
    }
    return columns;
  };
}

std::vector<std::complex<double>> Evolution::Coefficients(
    const std::map<int, std::vector<double>>& perDegree, double factor) const {
  auto out = solver_->Grid().ZeroCoefficients();
  for (const auto& [l, columns] : perDegree) {
    for (auto m = 0; m <= l; ++m) {
      out[SurfaceGrid::Index(l, m)] =
          factor * std::complex<double>(columns[m], columns[l + 1 + m]);
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

void Evolution::SetBackgroundState(GRDSolver::State state) {
  const auto fieldSize = solver_->Grid().FieldSize();
  if (state.iceThickness.size() != fieldSize ||
      state.seaLevel.size() != fieldSize ||
      state.oceanFunction.size() != fieldSize ||
      (!state.forcedLand.empty() && state.forcedLand.size() != fieldSize)) {
    throw std::invalid_argument("state fields must live on the solver's grid");
  }
  state_ = std::move(state);
}

std::vector<Evolution::Snapshot> Evolution::Run(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& directLoad) {
  CheckTimes(times);
  return March(times, &directLoad, nullptr, nullptr, false);
}

std::vector<Evolution::Snapshot> Evolution::Run(std::span<const double> times,
                                                const Forcing& forcing) {
  CheckTimes(times);
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               nullptr, forcing.tide ? &forcing.tide : nullptr, false,
               &forcing);
}

std::vector<Evolution::Snapshot> Evolution::Resume(
    const Checkpoint& from, std::span<const double> times,
    const Forcing& forcing) {
  Restore(from, times);
  if (forcing.sources || forcing.stressGluts) {
    throw std::invalid_argument(
        "a run under interior sources is not resumed from a checkpoint, "
        "which does not hold the toroidal state");
  }
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               nullptr, forcing.tide ? &forcing.tide : nullptr, true);
}

std::vector<Evolution::Snapshot> Evolution::RunMigrating(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness) {
  CheckTimes(times);
  return March(times, nullptr, &iceThickness, nullptr, false);
}

void Evolution::Restore(const Checkpoint& from,
                                std::span<const double> times) {
  CheckTimes(times);
  if (times[0] != from.t) {
    throw std::invalid_argument(
        "the first time of a resumed run is the checkpoint's own node");
  }
  vstate_ = op_->ZeroState([](int l) { return 2 * (l + 1); });
  vstate_.t = from.t * toMatTime_;
  vstate_.m = from.m;
  vstate_.strain = from.strain;
  sigma_ = from.loadCoefficients;
  psi_ = from.tideCoefficients;
}

std::vector<Evolution::Snapshot> Evolution::Resume(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& directLoad) {
  Restore(from, times);
  return March(times, &directLoad, nullptr, nullptr, true);
}

std::vector<Evolution::Snapshot> Evolution::ResumeMigrating(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness) {
  Restore(from, times);
  return March(times, nullptr, &iceThickness, nullptr, true);
}

std::vector<Evolution::Snapshot> Evolution::March(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>* directLoad,
    const std::function<std::vector<double>(double)>* iceThickness,
    const std::function<std::vector<std::complex<double>>(double)>* tide,
    bool resumed, const Forcing* interior) {
  const auto& op = *op_;
  const auto& solver = *solver_;
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto stride = options_.record_stride;
  const auto lastNode = times.size() - 1;
  const bool migrating = iceThickness != nullptr;

  if (migrating && !resumed) {
    // The shorelines follow flotation from the first node, so the
    // background must be the state there, flotation's own C included.
    detail::CheckMigratingStart(solver, state_, (*iceThickness)(times[0]));
  }

  if (options_.rotation == RotationalFeedback::Full) {
    // Long steps drive the effective tidal number towards the fluid
    // limit, so the specified moments must sustain more oblateness than
    // the model's hydrostatic one (Mitrovica et al. 2005; Yu et al.
    // 2025); the margin with the elastic number alone is not enough.
    const auto kHyd = FluidTidalNumber(op.BaseMaterial(), 2);
    if (solver.TransverseMarginFor(kHyd) <= 0.0) {
      throw std::invalid_argument(
          "the transverse rotational feedback is unstable: the specified "
          "moments of inertia sustain less oblateness than the model's "
          "hydrostatic fluid limit. Use WithHydrostaticMoments, or rotation "
          "= Axial or None.");
    }
  }

  GRDSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;
  auto nonlinearOptions = sleOptions;
  nonlinearOptions.max_iterations = options_.max_nonlinear_iterations;

  if (!resumed) {
    vstate_ = op.ZeroState([](int l) { return 2 * (l + 1); });
    vstate_.t = times[0] * toMatTime_;
  }

  auto totalLoadCoefficients = [&](const std::vector<double>& loadGrid,
                                   const std::vector<double>& slc) {
    auto total = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      total[i] = loadGrid[i] +
                 solver.WaterDensity() * state_.oceanFunction[i] * slc[i];
    }
    return grid.Expand(total);
  };
  // Solvers for the unrelaxed table and the per-dt effective tables, each
  // under the caller's parameters; the Love tables themselves are cached
  // by the operator, so a repeated dt costs a table conversion only.
  auto flavours = std::map<double, GRDSolver>{};
  auto solverFor = [&](double dt) -> const GRDSolver& {
    auto found = flavours.find(dt);
    if (found == flavours.end()) {
      found =
          flavours
              .emplace(dt, GRDSolver(grid,
                                          dt == 0.0 ? op.ElasticLoveNumbers()
                                                    : op.EffectiveLoveNumbers(
                                                          options_.scheme, dt),
                                          solver.Parameters()))
              .first;
    }
    return found->second;
  };

  auto checkpoint = [&](std::size_t node) {
    const auto every = options_.checkpoint_stride;
    if (every <= 0 || !options_.on_checkpoint) return;
    if (node % static_cast<std::size_t>(every) != 0 && node != lastNode) {
      return;
    }
    options_.on_checkpoint(
        node, Checkpoint{.t = times[node],
                         .m = vstate_.m,
                         .strain = vstate_.strain,
                         .loadCoefficients = sigma_,
                         .tideCoefficients = psi_});
  };

  // The direct load at a node (none given: none), and the potential the
  // radial steps see there: the centrifugal potential of the rotation the
  // node's closure found, and the prescribed tide.
  auto directLoadAt = [&](double t) {
    return directLoad != nullptr ? (*directLoad)(t)
                                 : std::vector<double>(fieldSize, 0.0);
  };
  auto potentialAt = [&](double t, const std::array<double, 3>& omega) {
    auto psi = options_.rotation == RotationalFeedback::None
                   ? grid.ZeroCoefficients()
                   : solver.CentrifugalCoefficients(omega);
    if (tide != nullptr) {
      const auto given = (*tide)(t);
      if (given.size() != psi.size()) {
        throw std::invalid_argument(
            "the tide is not a coefficient vector of the solver's grid");
      }
      for (auto i = std::size_t{0}; i < psi.size(); ++i) psi[i] += given[i];
    }
    return psi;
  };

  // Interior sources: their forces on the spheroidal systems (the real
  // columns of each degree) and on the toroidal ones, which are stepped
  // on their own beside the loop.
  const bool sourced =
      interior != nullptr && (interior->sources || interior->stressGluts);
  auto toroidal = std::optional<ToroidalViscoelastic>{};
  auto tstate = ToroidalViscoelastic::State{};
  auto twist = ToroidalViscoelastic::Fields{};  // the node's toroidal solutions
  if (sourced && stride > 0) {
    throw std::invalid_argument(
        "a run under interior sources is not recorded for the kernels in "
        "the Earth's parameters: its trajectory would not hold the "
        "sources' forces or the toroidal state");
  }
  if (sourced) {
    toroidal.emplace(op);
    tstate = toroidal->ZeroState();
    tstate.t = times[0] * toMatTime_;
  }
  auto sourceForcesAt = [&](double t) {
    return detail::ForcesOfSources(
        op, *toroidal, solver.SolverScales(), toNdLength_,
        interior->sources ? interior->sources(t) : std::vector<PointSource>{},
        interior->stressGluts ? interior->stressGluts(t)
                              : std::vector<StressGlut>{});
  };
  // The toroidal potential's coefficients of the node's toroidal solutions.
  auto toroidalCoefficients = [&] {
    return Coefficients(toroidal->Surface(twist), toNdLength_);
  };

  auto loadGrid = std::vector<double>{};
  auto response = GRDSolver::Response{};
  auto ocean = std::vector<double>{};  // the node's C, migrating only
  if (!resumed && migrating) {
    // The background is the state at times[0]: the run starts undisturbed.
    response.relativeSeaLevelChange.assign(fieldSize, 0.0);
    response.displacement.assign(fieldSize, 0.0);
    response.potential.assign(fieldSize, 0.0);
    response.tangentialPotential.assign(fieldSize, 0.0);
    sigma_ = grid.ZeroCoefficients();
    psi_ = grid.ZeroCoefficients();
    ocean = state_.oceanFunction;
  } else if (!resumed) {
    // The instantaneous elastic response at the start: the right limit,
    // so a load switched on at times[0] is stepped at full order.
    loadGrid = directLoadAt(times[0]);
    const auto& elasticSolver = solverFor(0.0);
    GRDSolver::GeneralisedForcing forcing;
    forcing.directLoad = loadGrid;
    if (tide != nullptr) forcing.tide = (*tide)(times[0]);
    forcing.rotation = options_.rotation;
    // The sources' own unrelaxed response enters as ready channels, and
    // its strain joins the start's.
    auto startResponse = std::map<int, std::vector<double>>{};
    if (sourced) {
      const auto forces = sourceForcesAt(times[0]);
      for (const auto& [l, f] : forces.spheroidal) {
        startResponse.emplace(l, op.UnrelaxedSystem(l).Solve(f, 2 * (l + 1)));
      }
      forcing.displacementResponse =
          Coefficients(op.Surface(startResponse, Component::U), toNdLength_);
      forcing.potentialResponse =
          Coefficients(op.Surface(startResponse, Component::Phi), toNdPotential_);
      forcing.tangentialResponse =
          Coefficients(op.Surface(startResponse, Component::V), toNdLength_);
      if (const auto found = startResponse.find(0);
          found != startResponse.end()) {
        forcing.inertiaMomentChange =
            op.InertiaMoment(found->second, 2)[0] * toNdMoment_;
      }
      twist = toroidal->SolveElastic(tstate, forces.toroidal);
      toroidal->SetStrain(tstate, twist);
      forcing.toroidalResponse = toroidalCoefficients();
    }
    response = elasticSolver.SolveGeneralised(state_, forcing, sleOptions);
    sigma_ = totalLoadCoefficients(loadGrid, response.relativeSeaLevelChange);
    psi_ = potentialAt(times[0], response.angularVelocity);
    if (sourced) {
      op.PrimeStrain(vstate_, Columns(sigma_, toMatDensity_),
                     Columns(psi_, toMatPotential_));
      for (auto& [l, d] : op.Strain(startResponse)) {
        auto& to = vstate_.strain.at(l);
        for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += d[i];
      }
    }
  }
  if (stride > 0 && !sourced) {
    op.PrimeStrain(vstate_, Columns(sigma_, toMatDensity_),
                   Columns(psi_, toMatPotential_));
  }

  auto snapshots = std::vector<Snapshot>{};
  auto record = [&](double t, GRDSolver::Response& r, bool keepState) {
    Snapshot snapshot;
    snapshot.t = t;
    snapshot.relativeSeaLevelChange = std::move(r.relativeSeaLevelChange);
    snapshot.displacement = std::move(r.displacement);
    snapshot.potential = std::move(r.potential);
    snapshot.tangentialPotential = std::move(r.tangentialPotential);
    snapshot.toroidalPotential = std::move(r.toroidalPotential);
    snapshot.angularVelocity = r.angularVelocity;
    snapshot.converged = r.converged;
    snapshot.oceanFunction = std::move(ocean);
    // The internal variables, and the strain only on request: an adjoint
    // otherwise regenerates it by one elastic solve under the recorded
    // loads.
    if (keepState && options_.record_strain) {
      snapshot.internalState = vstate_;
    } else if (keepState) {
      ViscoelasticState kept;
      kept.t = vstate_.t;
      kept.m = vstate_.m;
      snapshot.internalState = std::move(kept);
    }
    if (stride > 0) {
      snapshot.loadCoefficients = sigma_;
      snapshot.tideCoefficients = psi_;
    }
    snapshots.push_back(std::move(snapshot));
    if (options_.on_snapshot) {
      options_.on_snapshot(snapshots.size() - (resumed ? 0 : 1),
                           snapshots.back());
    }
  };
  if (!resumed) {
    record(times[0], response, stride > 0);
    checkpoint(0);
  }

  for (auto node = std::size_t{1}; node < times.size(); ++node) {
    const auto dt = (times[node] - times[node - 1]) * toMatTime_;
    auto step = op.StartStep(vstate_, dt, options_.scheme,
                             Columns(sigma_, toMatDensity_),
                             Columns(psi_, toMatPotential_));
    const auto& effectiveSolver = solverFor(dt);
    if (migrating) {
      // The step's equilibrium is the nonlinear equation from the
      // background, the memory field entering through the response
      // channels (pyslfp's `_run_migrating`).
      GRDSolver::NonlinearForcing moving;
      moving.iceThicknessChange = (*iceThickness)(times[node]);
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        moving.iceThicknessChange[i] -= state_.iceThickness[i];
      }
      moving.displacementResponse =
          Coefficients(step.MemorySurface(Component::U), toNdLength_);
      moving.potentialResponse =
          Coefficients(step.MemorySurface(Component::Phi), toNdPotential_);
      moving.tangentialResponse =
          Coefficients(step.MemorySurface(Component::V), toNdLength_);
      moving.inertiaMomentChange = step.InertiaMoment()[0] * toNdMoment_;
      moving.rotation = options_.rotation;
      auto solved =
          effectiveSolver.SolveNonlinear(state_, moving, nonlinearOptions);
      response = std::move(solved.response);
      // the load the converged state carries relative to the background,
      // so the radial step finishes under the load its solution saw
      sigma_ =
          detail::MigratedLoadCoefficients(solver, state_, solved.finalState);
      ocean = std::move(solved.finalState.oceanFunction);
    } else {
      GRDSolver::GeneralisedForcing coupled;
      loadGrid = directLoadAt(times[node]);
      coupled.directLoad = loadGrid;
      if (tide != nullptr) coupled.tide = (*tide)(times[node]);
      if (sourced) {
        // the node's sources force the step; the toroidal systems are
        // stepped under theirs
        const auto forces = sourceForcesAt(times[node]);
        step.AddForce(forces.spheroidal);
        auto stepped =
            toroidal->Step(tstate, dt, options_.scheme, forces.toroidal);
        tstate = std::move(stepped.first);
        twist = std::move(stepped.second);
        coupled.toroidalResponse = toroidalCoefficients();
      }
      coupled.displacementResponse =
          Coefficients(step.MemorySurface(Component::U), toNdLength_);
      coupled.potentialResponse =
          Coefficients(step.MemorySurface(Component::Phi), toNdPotential_);
      coupled.tangentialResponse =
          Coefficients(step.MemorySurface(Component::V), toNdLength_);
      coupled.inertiaMomentChange = step.InertiaMoment()[0] * toNdMoment_;
      coupled.rotation = options_.rotation;
      response = effectiveSolver.SolveGeneralised(state_, coupled, sleOptions);
      sigma_ = totalLoadCoefficients(loadGrid, response.relativeSeaLevelChange);
    }
    psi_ = potentialAt(times[node], response.angularVelocity);
    vstate_ = step.Finish(Columns(sigma_, toMatDensity_),
                          Columns(psi_, toMatPotential_))
                  .first;
    const bool keep = stride > 0 && (node % stride == 0 || node == lastNode);
    record(times[node], response, keep);
    checkpoint(node);
  }
  return snapshots;
}

}  // namespace sl3dpp
