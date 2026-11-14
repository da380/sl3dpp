#include "sl3dpp/Evolution.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <format>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>

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
  if (times.empty()) {
    throw std::invalid_argument("times must hold at least one entry");
  }
  for (auto i = std::size_t{1}; i < times.size(); ++i) {
    if (times[i] <= times[i - 1]) {
      throw std::invalid_argument("times must be increasing");
    }
  }
}

/// The refusal of what a run without gravity cannot take.
void RefuseWithoutGravity(const ViscoelasticOperator& op,
                          std::string_view what) {
  if (!op.Gravitating()) {
    throw std::invalid_argument(std::format(
        "{} is refused without gravity: the run is the solid body under "
        "surface loads (their weight) and interior sources",
        what));
  }
}

}  // namespace

double DegreeOneShare(std::span<const std::complex<double>> coefficients,
                      int lmax) {
  auto all = 0.0, one = 0.0;
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      // an order m > 0 stands for its conjugate at -m too
      const auto power = (m == 0 ? 1.0 : 2.0) *
                         std::norm(coefficients[SurfaceGrid::Index(l, m)]);
      all += power;
      if (l == 1) one += power;
    }
  }
  return all > 0.0 ? std::sqrt(one / all) : 0.0;
}

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
  if (!op_->Gravitating()) {
    return MarchWithoutGravity(times, Forcing{.directLoad = directLoad},
                               false);
  }
  return March(times, &directLoad, nullptr, nullptr, false);
}

std::vector<Evolution::Snapshot> Evolution::Run(std::span<const double> times,
                                                const Forcing& forcing) {
  CheckTimes(times);
  if (options_.factor_sources) return MarchFactored(times, forcing);
  if (!op_->Gravitating()) return MarchWithoutGravity(times, forcing, false);
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               nullptr, forcing.tide ? &forcing.tide : nullptr, false,
               &forcing);
}

std::vector<Evolution::Snapshot> Evolution::Resume(
    const Checkpoint& from, std::span<const double> times,
    const Forcing& forcing) {
  Restore(from, times);
  if ((forcing.sources || forcing.stressGluts) &&
      (!from.toroidal.has_value() || from.strain.empty())) {
    throw std::invalid_argument(
        "a run under interior sources is resumed from a checkpoint of "
        "such a run, which holds the strain and the toroidal state");
  }
  if (!op_->Gravitating()) return MarchWithoutGravity(times, forcing, true);
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               nullptr, forcing.tide ? &forcing.tide : nullptr, true,
               &forcing);
}

std::vector<Evolution::Snapshot> Evolution::RunMigrating(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness) {
  CheckTimes(times);
  RefuseWithoutGravity(*op_, "a run under migrating shorelines (an ocean)");
  return March(times, nullptr, &iceThickness, nullptr, false);
}

std::vector<Evolution::Snapshot> Evolution::RunMigrating(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness,
    const Forcing& forcing) {
  CheckTimes(times);
  RefuseWithoutGravity(*op_, "a run under migrating shorelines (an ocean)");
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               &iceThickness, forcing.tide ? &forcing.tide : nullptr, false,
               &forcing);
}

std::vector<Evolution::Snapshot> Evolution::ResumeMigrating(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness,
    const Forcing& forcing) {
  RefuseWithoutGravity(*op_, "a run under migrating shorelines (an ocean)");
  Restore(from, times);
  if ((forcing.sources || forcing.stressGluts) &&
      (!from.toroidal.has_value() || from.strain.empty())) {
    throw std::invalid_argument(
        "a run under interior sources is resumed from a checkpoint of "
        "such a run, which holds the strain and the toroidal state");
  }
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               &iceThickness, forcing.tide ? &forcing.tide : nullptr, true,
               &forcing);
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
  restoredToroidal_ = from.toroidal;
}

std::vector<Evolution::Snapshot> Evolution::Resume(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& directLoad) {
  Restore(from, times);
  if (!op_->Gravitating()) {
    return MarchWithoutGravity(times, Forcing{.directLoad = directLoad}, true);
  }
  return March(times, &directLoad, nullptr, nullptr, true);
}

std::vector<Evolution::Snapshot> Evolution::ResumeMigrating(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness) {
  RefuseWithoutGravity(*op_, "a run under migrating shorelines (an ocean)");
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
  // the prescribed tide, refused at any node where it has a part below
  // degree 2 (RequireTideDegrees)
  auto checkedTide = std::function<std::vector<std::complex<double>>(double)>{};
  if (tide != nullptr) {
    checkedTide = [tide](double t) {
      auto given = (*tide)(t);
      RequireTideDegrees(given);
      return given;
    };
    tide = &checkedTide;
  }

  if (migrating && !resumed) {
    // The shorelines follow flotation from the first node, so the
    // background must be the state there, flotation's own C included.
    detail::CheckMigratingStart(solver, state_, (*iceThickness)(times[0]),
                                interior != nullptr);
    if (options_.land_held) {
      detail::CheckLandAtStart(solver, state_, options_.land_held(times[0]));
    }
  }

  if (options_.rotation == RotationalFeedback::Full && times.size() > 1) {
    // Long steps drive the effective tidal number towards the fluid
    // limit, so the specified moments must sustain more oblateness than
    // the model's hydrostatic one (Mitrovica et al. 2005; Yu et al.
    // 2025); the margin with the elastic number alone is not enough. A
    // run of one node takes no step: its elastic problem is closed on the
    // elastic numbers alone.
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

  // the toroidal state a checkpoint carries, under interior sources
  const ToroidalViscoelastic::State* twisted = nullptr;
  auto checkpoint = [&](std::size_t node) {
    const auto every = options_.checkpoint_stride;
    if (every <= 0 || !options_.on_checkpoint) return;
    if (node % static_cast<std::size_t>(every) != 0 && node != lastNode) {
      return;
    }
    auto out = Checkpoint{.t = times[node],
                          .m = vstate_.m,
                          .strain = vstate_.strain,
                          .loadCoefficients = sigma_,
                          .tideCoefficients = psi_,
                          .toroidal = std::nullopt};
    if (twisted != nullptr) out.toroidal = *twisted;
    options_.on_checkpoint(node, out);
  };

  // The direct load at a node (none given: none), and the potential the
  // radial steps see there: the centrifugal potential of the rotation the
  // node's closure found, and the prescribed tide.
  auto directLoadAt = [&](double t) {
    return directLoad != nullptr ? (*directLoad)(t)
                                 : std::vector<double>(fieldSize, 0.0);
  };
  // The load outside the water budget and the layer on the solid surface
  // at a node, where the forcing gives them (empty: none).
  auto outsideAt = [&](double t) {
    return interior != nullptr && interior->directLoadOutsideWaterBudget
               ? interior->directLoadOutsideWaterBudget(t)
               : std::vector<double>{};
  };
  auto layerAt = [&](double t) {
    return interior != nullptr && interior->solidSurfaceChange
               ? interior->solidSurfaceChange(t)
               : std::vector<double>{};
  };
  // every direct load of a node, for the radial step's own load
  auto withOutside = [&](std::vector<double> load,
                         const std::vector<double>& outside) {
    if (outside.empty()) return load;
    if (outside.size() != load.size()) {
      throw std::invalid_argument(
          "a direct load outside the water budget lives on the solver's grid");
    }
    for (auto i = std::size_t{0}; i < load.size(); ++i) load[i] += outside[i];
    return load;
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
  auto nodeForces = RadialSourceForces{};  // the node's source forces
  if (sourced && stride > 1) {
    throw std::invalid_argument(
        "a run under interior sources is recorded at every node "
        "(record_stride 1): the states between could not be rebuilt "
        "without the sources' forces");
  }
  if (sourced) {
    toroidal.emplace(op);
    if (resumed) {
      tstate = restoredToroidal_.value();
    } else {
      tstate = toroidal->ZeroState();
      tstate.t = times[0] * toMatTime_;
    }
    twisted = &tstate;
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

  // The sources' own unrelaxed response at the start, as ready channels
  // of the start's sea level solve (linear or nonlinear); its strain
  // joins the start's.
  auto startResponse = std::map<int, std::vector<double>>{};
  auto startSources = [&](auto& forcing) {
    nodeForces = sourceForcesAt(times[0]);
    const auto& forces = nodeForces;
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
  };
  auto primeUnderSources = [&] {
    op.PrimeStrain(vstate_, Columns(sigma_, toMatDensity_),
                   Columns(psi_, toMatPotential_));
    for (auto& [l, d] : op.Strain(startResponse)) {
      auto& to = vstate_.strain.at(l);
      for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += d[i];
    }
  };
  // The forcing beside the ice of a migrating node: the loads, the layer
  // and the tide, changes from the background.
  auto movingLoadsAt = [&](double t, GRDSolver::NonlinearForcing& moving) {
    if (directLoad != nullptr) moving.directLoad = (*directLoad)(t);
    moving.directLoadOutsideWaterBudget = outsideAt(t);
    moving.solidSurfaceChange = layerAt(t);
    if (tide != nullptr) moving.tide = (*tide)(t);
    // a land mask that changes in time: the node's own
    if (options_.land_held) moving.forcedLand = options_.land_held(t);
  };
  // every stride-th node of the whole run is kept, and the last
  auto kept = [&](std::size_t node) {
    if (stride <= 0) return false;
    const auto every = static_cast<std::size_t>(stride);
    return (options_.record_offset + node) % every == 0 || node == lastNode;
  };

  auto loadGrid = std::vector<double>{};
  auto response = GRDSolver::Response{};
  auto ocean = std::vector<double>{};  // the node's C, migrating only
  if (!resumed && migrating && interior != nullptr) {
    // The start under a forcing beside the ice: the nonlinear equation
    // on the unrelaxed body, from the background.
    GRDSolver::NonlinearForcing moving;
    // the ice at the first node may differ from the background's
    moving.iceThicknessChange = (*iceThickness)(times[0]);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      moving.iceThicknessChange[i] -= state_.iceThickness[i];
    }
    movingLoadsAt(times[0], moving);
    moving.rotation = options_.rotation;
    if (sourced) startSources(moving);
    auto solved = solverFor(0.0).SolveNonlinear(state_, moving, nonlinearOptions);
    response = std::move(solved.response);
    sigma_ = std::move(solved.loadCoefficients);
    ocean = std::move(solved.finalState.oceanFunction);
    psi_ = potentialAt(times[0], response.angularVelocity);
    if (sourced) primeUnderSources();
  } else if (!resumed && migrating) {
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
    forcing.directLoadOutsideWaterBudget = outsideAt(times[0]);
    forcing.solidSurfaceChange = layerAt(times[0]);
    if (tide != nullptr) forcing.tide = (*tide)(times[0]);
    forcing.rotation = options_.rotation;
    if (sourced) startSources(forcing);
    response = elasticSolver.SolveGeneralised(state_, forcing, sleOptions);
    sigma_ = totalLoadCoefficients(
        withOutside(loadGrid, forcing.directLoadOutsideWaterBudget),
        response.relativeSeaLevelChange);
    psi_ = potentialAt(times[0], response.angularVelocity);
    if (sourced) primeUnderSources();
  }
  if (stride > 0 && !sourced) {
    op.PrimeStrain(vstate_, Columns(sigma_, toMatDensity_),
                   Columns(psi_, toMatPotential_));
  }

  auto snapshots = std::vector<Snapshot>{};
  auto recorded = std::size_t{0};
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
    if (keepState && (options_.record_strain || sourced)) {
      snapshot.internalState = vstate_;
      if (sourced) {
        snapshot.toroidalState = tstate;
        snapshot.sourceForces = nodeForces;
      }
    } else if (keepState) {
      ViscoelasticState kept;
      kept.t = vstate_.t;
      kept.m = vstate_.m;
      kept.columns = vstate_.columns;
      snapshot.internalState = std::move(kept);
    }
    if (stride > 0) {
      snapshot.loadCoefficients = sigma_;
      snapshot.tideCoefficients = psi_;
    }
    snapshots.push_back(std::move(snapshot));
    const auto node = ++recorded - (resumed ? 0 : 1);
    if (options_.on_snapshot) options_.on_snapshot(node, snapshots.back());
    if (!options_.keep_snapshots) snapshots.pop_back();
  };
  if (!resumed) {
    record(times[0], response, kept(0));
    checkpoint(0);
  }

  for (auto node = std::size_t{1}; node < times.size(); ++node) {
    const auto dt = (times[node] - times[node - 1]) * toMatTime_;
    auto step = op.StartStep(vstate_, dt, options_.scheme,
                             Columns(sigma_, toMatDensity_),
                             Columns(psi_, toMatPotential_));
    const auto& effectiveSolver = solverFor(dt);
    // the node's sources force the step; the toroidal systems are
    // stepped under theirs
    auto stepSources = [&](auto& forcing) {
      nodeForces = sourceForcesAt(times[node]);
      const auto& forces = nodeForces;
      step.AddForce(forces.spheroidal);
      auto stepped =
          toroidal->Step(tstate, dt, options_.scheme, forces.toroidal);
      tstate = std::move(stepped.first);
      twist = std::move(stepped.second);
      forcing.toroidalResponse = toroidalCoefficients();
    };
    if (migrating) {
      // The step's equilibrium is the nonlinear equation from the
      // background, the memory field entering through the response
      // channels (pyslfp's `_run_migrating`).
      GRDSolver::NonlinearForcing moving;
      moving.iceThicknessChange = (*iceThickness)(times[node]);
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        moving.iceThicknessChange[i] -= state_.iceThickness[i];
      }
      movingLoadsAt(times[node], moving);
      if (sourced) stepSources(moving);
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
      sigma_ = std::move(solved.loadCoefficients);
      ocean = std::move(solved.finalState.oceanFunction);
    } else {
      GRDSolver::GeneralisedForcing coupled;
      loadGrid = directLoadAt(times[node]);
      coupled.directLoad = loadGrid;
      coupled.directLoadOutsideWaterBudget = outsideAt(times[node]);
      coupled.solidSurfaceChange = layerAt(times[node]);
      if (tide != nullptr) coupled.tide = (*tide)(times[node]);
      if (sourced) stepSources(coupled);
      coupled.displacementResponse =
          Coefficients(step.MemorySurface(Component::U), toNdLength_);
      coupled.potentialResponse =
          Coefficients(step.MemorySurface(Component::Phi), toNdPotential_);
      coupled.tangentialResponse =
          Coefficients(step.MemorySurface(Component::V), toNdLength_);
      coupled.inertiaMomentChange = step.InertiaMoment()[0] * toNdMoment_;
      coupled.rotation = options_.rotation;
      response = effectiveSolver.SolveGeneralised(state_, coupled, sleOptions);
      sigma_ = totalLoadCoefficients(
          withOutside(loadGrid, coupled.directLoadOutsideWaterBudget),
          response.relativeSeaLevelChange);
    }
    psi_ = potentialAt(times[node], response.angularVelocity);
    vstate_ = step.Finish(Columns(sigma_, toMatDensity_),
                          Columns(psi_, toMatPotential_))
                  .first;
    record(times[node], response, kept(node));
    checkpoint(node);
  }
  return snapshots;
}

std::vector<Evolution::Snapshot> Evolution::MarchFactored(
    std::span<const double> times, const Forcing& forcing) {
  const auto& op = *op_;
  const auto& solver = *solver_;
  const auto& grid = solver.Grid();
  const auto lmax = op.LMax();
  const auto nodes = times.size();
  if (state_.oceanArea > 0.0 || options_.rotation != RotationalFeedback::None ||
      forcing.directLoad || forcing.directLoadOutsideWaterBudget ||
      forcing.solidSurfaceChange || forcing.tide || options_.record_stride > 0 ||
      (!forcing.sources && !forcing.stressGluts)) {
    throw std::invalid_argument(
        "factor_sources is for interior sources alone on a body without an "
        "ocean and without rotational feedback, unrecorded: with an ocean, a "
        "load, a tide or the feedback the loop carries every order anyway");
  }
  const auto& material = op.BaseMaterial();
  const auto toMatMoment = detail::MomentToMaterial(op, solver.SolverScales());
  using Complex = std::complex<double>;

  // A class of sources: one radius and one history, s(t_n) times a
  // pattern, whose moments add.
  struct Class {
    double radius = 0.0;          // material units
    std::vector<double> history;  // per node, its largest entry one
    SourceMoments moments;
  };
  auto classes = std::vector<Class>{};
  auto classOf = [&](double radius, const std::vector<double>& history)
      -> Class& {
    for (auto& c : classes) {
      if (c.radius != radius) continue;
      auto same = true;
      for (auto n = std::size_t{0}; n < nodes && same; ++n) {
        same = std::abs(c.history[n] - history[n]) <= 1e-12;
      }
      if (same) return c;
    }
    classes.push_back(
        {.radius = radius,
         .history = history,
         .moments = {.lmax = lmax,
                     .at = std::vector<std::array<Complex, 6>>(
                         SurfaceGrid::Index(lmax, lmax) + 1)}});
    return classes.back();
  };
  auto addMoments = [](SourceMoments& to, const SourceMoments& from,
                       double factor) {
    for (auto i = std::size_t{0}; i < to.at.size(); ++i) {
      for (auto k = 0; k < 6; ++k) to.at[i][k] += factor * from.at[i][k];
    }
  };
  // a history scaled to its largest entry: the scale and the shape
  auto normalised = [&](const std::vector<double>& h) {
    auto top = 0.0;
    for (auto v : h) {
      if (std::abs(v) > std::abs(top)) top = v;
    }
    auto shape = h;
    if (top != 0.0) {
      for (auto& v : shape) v /= top;
    }
    return std::pair{top, std::move(shape)};
  };

  // The point sources: each position's tensor history, its components
  // grouped by their histories (one group when the tensor keeps its
  // pattern through time).
  if (forcing.sources) {
    struct Position {
      BodyPoint at;
      std::vector<std::array<double, 6>> tensor;  // per node
    };
    auto positions = std::vector<Position>{};
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      for (const auto& source : forcing.sources(times[n])) {
        auto found = std::ranges::find_if(positions, [&](const Position& p) {
          return p.at.radius == source.at.radius &&
                 p.at.colatitude == source.at.colatitude &&
                 p.at.longitude == source.at.longitude;
        });
        if (found == positions.end()) {
          positions.push_back({.at = source.at,
                               .tensor = std::vector<std::array<double, 6>>(
                                   nodes, std::array<double, 6>{})});
          found = positions.end() - 1;
        }
        const auto& M = source.moment;
        const std::array<double, 6> components{M.rr, M.tt, M.pp,
                                               M.rt, M.rp, M.tp};
        for (auto k = 0; k < 6; ++k) found->tensor[n][k] += components[k];
      }
    }
    for (const auto& position : positions) {
      auto at = position.at;
      at.radius /= toNdLength_;
      // the components' scales and shapes, then one pattern per shape
      auto scales = std::array<double, 6>{};
      auto shapes = std::array<std::vector<double>, 6>{};
      for (auto k = 0; k < 6; ++k) {
        auto h = std::vector<double>(nodes);
        for (auto n = std::size_t{0}; n < nodes; ++n) h[n] = position.tensor[n][k];
        std::tie(scales[k], shapes[k]) = normalised(h);
      }
      auto done = std::array<bool, 6>{};
      for (auto k = 0; k < 6; ++k) {
        if (done[k] || scales[k] == 0.0) continue;
        auto pattern = std::array<double, 6>{};
        for (auto j = k; j < 6; ++j) {
          if (done[j] || scales[j] == 0.0) continue;
          auto same = true;
          for (auto n = std::size_t{0}; n < nodes && same; ++n) {
            same = std::abs(shapes[j][n] - shapes[k][n]) <= 1e-12;
          }
          if (!same) continue;
          pattern[j] = scales[j] * toMatMoment;
          done[j] = true;
        }
        const auto source = PointSource{
            .at = at,
            .moment = {.rr = pattern[0], .tt = pattern[1], .pp = pattern[2],
                       .rt = pattern[3], .rp = pattern[4], .tp = pattern[5]}};
        addMoments(classOf(at.radius, shapes[k]).moments,
                   MomentsOf(lmax, source), 1.0);
      }
    }
  }
  // The stress gluts: each one factor of time times a pattern.
  if (forcing.stressGluts) {
    auto histories = std::vector<std::vector<StressGlut>>{};
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      histories.push_back(forcing.stressGluts(times[n]));
      if (histories.back().size() != histories.front().size()) {
        throw std::invalid_argument(
            "factor_sources takes the same stress gluts at every node, each "
            "scaled by a factor of time");
      }
    }
    for (auto i = std::size_t{0}; i < histories.front().size(); ++i) {
      auto norm = [](const StressGlut& g) {
        auto sum = 0.0;
        for (const auto* c : {&g.moment.rr, &g.moment.tt, &g.moment.pp,
                              &g.moment.rt, &g.moment.rp, &g.moment.tp}) {
          for (auto v : *c) sum += v * v;
        }
        return sum;
      };
      auto dot = [](const StressGlut& a, const StressGlut& b) {
        auto sum = 0.0;
        const std::vector<double>* x[6] = {&a.moment.rr, &a.moment.tt,
                                           &a.moment.pp, &a.moment.rt,
                                           &a.moment.rp, &a.moment.tp};
        const std::vector<double>* y[6] = {&b.moment.rr, &b.moment.tt,
                                           &b.moment.pp, &b.moment.rt,
                                           &b.moment.rp, &b.moment.tp};
        for (auto k = 0; k < 6; ++k) {
          for (auto j = std::size_t{0}; j < x[k]->size(); ++j) {
            sum += (*x[k])[j] * (*y[k])[j];
          }
        }
        return sum;
      };
      auto reference = std::size_t{0};
      for (auto n = std::size_t{1}; n < nodes; ++n) {
        if (norm(histories[n][i]) > norm(histories[reference][i])) reference = n;
      }
      const auto& pattern = histories[reference][i];
      const auto size2 = norm(pattern);
      if (size2 == 0.0) continue;
      auto history = std::vector<double>(nodes);
      for (auto n = std::size_t{0}; n < nodes; ++n) {
        const auto& at = histories[n][i];
        if (at.radius != pattern.radius || at.grid != pattern.grid) {
          throw std::invalid_argument(
              "factor_sources takes each stress glut on one shell and grid "
              "through the run");
        }
        history[n] = dot(at, pattern) / size2;
        // the residual of the factored form
        auto residual = 0.0;
        {
          const std::vector<double>* x[6] = {&at.moment.rr, &at.moment.tt,
                                             &at.moment.pp, &at.moment.rt,
                                             &at.moment.rp, &at.moment.tp};
          const std::vector<double>* y[6] = {
              &pattern.moment.rr, &pattern.moment.tt, &pattern.moment.pp,
              &pattern.moment.rt, &pattern.moment.rp, &pattern.moment.tp};
          for (auto k = 0; k < 6; ++k) {
            for (auto j = std::size_t{0}; j < x[k]->size(); ++j) {
              const auto r = (*x[k])[j] - history[n] * (*y[k])[j];
              residual += r * r;
            }
          }
        }
        if (residual > 1e-24 * size2) {
          throw std::invalid_argument(
              "factor_sources takes a stress glut whose history is one "
              "factor of time times a pattern; use the general route");
        }
      }
      auto glut = pattern;
      glut.radius /= toNdLength_;
      addMoments(classOf(glut.radius, history).moments, MomentsOf(lmax, glut),
                 toMatMoment);
    }
  }

  // The unit solutions of each class through the nodes: the surface
  // values of the four spheroidal columns (U, V, phi) and the two
  // toroidal ones (W), per degree.
  struct UnitSurface {
    std::map<int, std::vector<double>> U, V, Phi, W;
  };
  auto surfaces =
      std::vector<std::vector<UnitSurface>>(classes.size());
  const ToroidalViscoelastic toroidal(op, [](int) { return 2; });
  const std::function<const DegreeOperator&(int)> sph =
      [&](int l) -> const DegreeOperator& { return op.UnrelaxedSystem(l); };
  const std::function<const ToroidalOperator&(int)> tor =
      [&](int l) -> const ToroidalOperator& { return toroidal.Unrelaxed(l); };
  for (auto c = std::size_t{0}; c < classes.size(); ++c) {
    const auto& which = classes[c];
    const auto units = UnitForces(material, lmax, sph, tor, which.radius);
    auto scaled = [](const std::map<int, std::vector<double>>& from,
                     double factor) {
      auto out = from;
      for (auto& [l, f] : out) {
        for (auto& v : f) v *= factor;
      }
      return out;
    };
    auto surfaceOf = [&](const std::map<int, std::vector<double>>& x,
                         const ToroidalViscoelastic::Fields& w) {
      return UnitSurface{.U = op.Surface(x, Component::U),
                         .V = op.Surface(x, Component::V),
                         .Phi = op.Surface(x, Component::Phi),
                         .W = toroidal.Surface(w)};
    };
    auto state = op.ZeroState([](int) { return 4; });
    state.t = times[0] * toMatTime_;
    auto tstate = toroidal.ZeroState();
    tstate.t = state.t;
    // the start: the unrelaxed response to the sources there
    auto x = std::map<int, std::vector<double>>{};
    for (const auto& [l, f] : scaled(units.spheroidal, which.history[0])) {
      x.emplace(l, op.UnrelaxedSystem(l).Solve(f, 4));
    }
    state.strain = op.Strain(x);
    auto w = toroidal.SolveElastic(tstate,
                                   scaled(units.toroidal, which.history[0]));
    toroidal.SetStrain(tstate, w);
    surfaces[c].push_back(surfaceOf(x, w));
    for (auto n = std::size_t{1}; n < nodes; ++n) {
      const auto dt = (times[n] - times[n - 1]) * toMatTime_;
      auto step = op.StartStep(state, dt, options_.scheme);
      step.AddForce(scaled(units.spheroidal, which.history[n]));
      auto finished = step.Finish({}, {});
      state = std::move(finished.first);
      auto stepped = toroidal.Step(tstate, dt, options_.scheme,
                                   scaled(units.toroidal, which.history[n]));
      tstate = std::move(stepped.first);
      surfaces[c].push_back(surfaceOf(finished.second, stepped.second));
    }
  }

  // The fields: each coefficient the unit surface values under the
  // class's amplitudes at its order.
  const auto g = solver.SurfaceGravity();
  auto snapshots = std::vector<Snapshot>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    auto U = grid.ZeroCoefficients(), V = grid.ZeroCoefficients();
    auto Phi = grid.ZeroCoefficients(), W = grid.ZeroCoefficients();
    for (auto c = std::size_t{0}; c < classes.size(); ++c) {
      const auto& at = surfaces[c][n];
      for (auto l = 0; l <= lmax; ++l) {
        for (auto m = 0; m <= l; ++m) {
          const auto amplitudes = AmplitudesOf(classes[c].moments, l, m);
          const auto i = SurfaceGrid::Index(l, m);
          for (auto k = 0; k < 4; ++k) {
            U[i] += amplitudes.spheroidal[k] * at.U.at(l)[k];
            V[i] += amplitudes.spheroidal[k] * at.V.at(l)[k];
            Phi[i] += amplitudes.spheroidal[k] * at.Phi.at(l)[k];
          }
          if (l >= 1) {
            for (auto k = 0; k < 2; ++k) {
              W[i] += amplitudes.toroidal[k] * at.W.at(l)[k];
            }
          }
        }
      }
    }
    auto slc = grid.ZeroCoefficients();
    for (auto i = std::size_t{0}; i < U.size(); ++i) {
      U[i] *= toNdLength_;
      V[i] *= toNdLength_;
      W[i] *= toNdLength_;
      Phi[i] *= toNdPotential_;
      // a body without an ocean: the equipotential's height above the
      // surface
      slc[i] = -(U[i] + Phi[i] / g);
    }
    Snapshot snapshot;
    snapshot.t = times[n];
    if (op.Gravitating()) {
      // without gravity there is no potential and no equipotential
      snapshot.relativeSeaLevelChange = grid.Synthesise(slc);
      snapshot.potential = grid.Synthesise(Phi);
    }
    snapshot.displacement = grid.Synthesise(U);
    snapshot.tangentialPotential = grid.Synthesise(V);
    snapshot.toroidalPotential = grid.Synthesise(W);
    snapshots.push_back(std::move(snapshot));
    if (options_.on_snapshot) options_.on_snapshot(n, snapshots.back());
    if (!options_.keep_snapshots) snapshots.pop_back();
  }
  return snapshots;
}

std::vector<Evolution::Snapshot> Evolution::MarchWithoutGravity(
    std::span<const double> times, const Forcing& forcing, bool resumed) {
  const auto& op = *op_;
  const auto& solver = *solver_;
  const auto& grid = solver.Grid();
  if (state_.oceanArea > 0.0) {
    throw std::invalid_argument(
        "a body with an ocean is refused without gravity: there is no sea "
        "level equation (GRDSolver::NoOceanState)");
  }
  if (options_.rotation != RotationalFeedback::None) {
    throw std::invalid_argument(
        "the rotational feedback is refused without gravity (it reads the "
        "inertia's change from the degree-2 potential): RotationalFeedback::"
        "None");
  }
  if (forcing.tide) RefuseWithoutGravity(op, "a tide");
  const auto fieldSize = grid.FieldSize();
  // The surface loads act by their weight, a normal traction (the load
  // channel of the operator without gravity): the loads inside and outside
  // the water budget alike, as on a body without an ocean with gravity. A
  // layer's height enters the sea level change alone, which is not formed
  // without gravity; its weight is the direct load beside it.
  const bool loaded =
      forcing.directLoad || forcing.directLoadOutsideWaterBudget;
  // The loads' coefficients at a node, solver units, their degree-1 part
  // removed: a net force, which nothing balances without gravity (the load
  // projected on the loads that exert none; a balanced load's part is grid
  // error).
  auto loadAt = [&](double t) {
    auto total = std::vector<double>(fieldSize, 0.0);
    for (const auto* given :
         {&forcing.directLoad, &forcing.directLoadOutsideWaterBudget}) {
      if (!*given) continue;
      const auto field = (*given)(t);
      if (field.empty()) continue;
      if (field.size() != fieldSize) {
        throw std::invalid_argument("a direct load lives on the solver's grid");
      }
      for (auto i = std::size_t{0}; i < fieldSize; ++i) total[i] += field[i];
    }
    if (forcing.solidSurfaceChange) {
      const auto layer = forcing.solidSurfaceChange(t);
      if (!layer.empty() && layer.size() != fieldSize) {
        throw std::invalid_argument(
            "a layer's height lives on the solver's grid");
      }
    }
    auto coefficients = grid.Expand(total);
    for (auto m = 0; m <= 1; ++m) coefficients[SurfaceGrid::Index(1, m)] = 0.0;
    return coefficients;
  };
  // the operator's load channel under coefficients (none without loads)
  auto channel = [&](const std::vector<std::complex<double>>& sigma) {
    return loaded ? Columns(sigma, toMatDensity_) : LoadCallable{};
  };
  const auto stride = options_.record_stride;
  const auto lastNode = times.size() - 1;
  // every stride-th node of the whole run is kept, and the last
  const auto kept = [&](std::size_t node) {
    if (stride <= 0) return false;
    const auto every = static_cast<std::size_t>(stride);
    return (options_.record_offset + node) % every == 0 || node == lastNode;
  };
  const bool sourced = forcing.sources || forcing.stressGluts;
  if (sourced && stride > 1) {
    throw std::invalid_argument(
        "a run under interior sources is recorded at every node "
        "(record_stride 1): the states between could not be rebuilt "
        "without the sources' forces");
  }

  const ToroidalViscoelastic toroidal(op);
  auto tstate = ToroidalViscoelastic::State{};
  if (!resumed) {
    vstate_ = op.ZeroState([](int l) { return 2 * (l + 1); });
    vstate_.t = times[0] * toMatTime_;
    sigma_ = grid.ZeroCoefficients();
    psi_ = grid.ZeroCoefficients();
    tstate = toroidal.ZeroState();
    tstate.t = vstate_.t;
  } else if (restoredToroidal_) {
    tstate = *restoredToroidal_;
  } else {
    tstate = toroidal.ZeroState();
    tstate.t = vstate_.t;
  }
  if (resumed && loaded && sigma_.size() != grid.ZeroCoefficients().size()) {
    throw std::invalid_argument(
        "a run under surface loads is resumed from a checkpoint holding the "
        "loads' coefficients at its node");
  }
  auto nodeForces = RadialSourceForces{};
  auto forcesAt = [&](double t) {
    if (!sourced) return RadialSourceForces{};
    return detail::ForcesOfSources(
        op, toroidal, solver.SolverScales(), toNdLength_,
        forcing.sources ? forcing.sources(t) : std::vector<PointSource>{},
        forcing.stressGluts ? forcing.stressGluts(t)
                            : std::vector<StressGlut>{});
  };

  auto spheroidal = std::map<int, std::vector<double>>{};
  auto twist = ToroidalViscoelastic::Fields{};
  auto snapshots = std::vector<Snapshot>{};
  auto recorded = std::size_t{0};
  auto record = [&](double t, bool keepState) {
    Snapshot snapshot;
    snapshot.t = t;
    snapshot.displacement = grid.Synthesise(
        Coefficients(op.Surface(spheroidal, Component::U), toNdLength_));
    snapshot.tangentialPotential = grid.Synthesise(
        Coefficients(op.Surface(spheroidal, Component::V), toNdLength_));
    if (sourced) {
      snapshot.toroidalPotential =
          grid.Synthesise(Coefficients(toroidal.Surface(twist), toNdLength_));
    }
    if (keepState && (options_.record_strain || sourced)) {
      snapshot.internalState = vstate_;
      if (sourced) {
        snapshot.toroidalState = tstate;
        snapshot.sourceForces = nodeForces;
      }
    } else if (keepState) {
      ViscoelasticState kept;
      kept.t = vstate_.t;
      kept.m = vstate_.m;
      kept.columns = vstate_.columns;
      snapshot.internalState = std::move(kept);
    }
    if (stride > 0) {
      snapshot.loadCoefficients = sigma_;
      snapshot.tideCoefficients = psi_;
    }
    snapshots.push_back(std::move(snapshot));
    const auto node = ++recorded - (resumed ? 0 : 1);
    if (options_.on_snapshot) options_.on_snapshot(node, snapshots.back());
    if (!options_.keep_snapshots) snapshots.pop_back();
  };
  auto checkpoint = [&](std::size_t node) {
    const auto every = options_.checkpoint_stride;
    if (every <= 0 || !options_.on_checkpoint) return;
    if (node % static_cast<std::size_t>(every) != 0 && node != lastNode) {
      return;
    }
    auto out = Checkpoint{.t = times[node],
                          .m = vstate_.m,
                          .strain = vstate_.strain,
                          .loadCoefficients = sigma_,
                          .tideCoefficients = psi_,
                          .toroidal = tstate};
    options_.on_checkpoint(node, out);
  };

  if (!resumed) {
    // The start: the unrelaxed response to the loads and the sources
    // there, with no memory before it.
    nodeForces = forcesAt(times[0]);
    if (loaded) sigma_ = loadAt(times[0]);
    spheroidal = op.SolveElastic(vstate_, channel(sigma_));
    for (const auto& [l, f] : nodeForces.spheroidal) {
      const auto x = op.UnrelaxedSystem(l).Solve(f, 2 * (l + 1));
      auto& to = spheroidal.at(l);
      for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += x[i];
    }
    vstate_.strain = op.Strain(spheroidal);
    if (sourced) {
      twist = toroidal.SolveElastic(tstate, nodeForces.toroidal);
      toroidal.SetStrain(tstate, twist);
    }
    record(times[0], kept(0));
    checkpoint(0);
  } else {
    op.PrimeStrain(vstate_, channel(sigma_));
  }

  for (auto node = std::size_t{1}; node < times.size(); ++node) {
    const auto dt = (times[node] - times[node - 1]) * toMatTime_;
    auto step = op.StartStep(vstate_, dt, options_.scheme, channel(sigma_));
    nodeForces = forcesAt(times[node]);
    step.AddForce(nodeForces.spheroidal);
    if (loaded) sigma_ = loadAt(times[node]);
    auto finished = step.Finish(channel(sigma_));
    vstate_ = std::move(finished.first);
    spheroidal = std::move(finished.second);
    if (sourced) {
      auto stepped =
          toroidal.Step(tstate, dt, options_.scheme, nodeForces.toroidal);
      tstate = std::move(stepped.first);
      twist = std::move(stepped.second);
    }
    record(times[node], kept(node));
    checkpoint(node);
  }
  return snapshots;
}

}  // namespace sl3dpp
