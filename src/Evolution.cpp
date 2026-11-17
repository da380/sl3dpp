#include "sl3dpp/Evolution.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <format>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "sl3dpp/StepRecipe.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/ToroidalViscoelastic.hpp"
#include "sl3dpp/detail/InteriorSources.hpp"
#include "sl3dpp/detail/Shorelines.hpp"
#include "sl3dpp/detail/TimeGrid.hpp"

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
        "the coupled loop steps implicitly: ExponentialTrapezoid, "
        "BackwardEuler or Sdirk2");
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

void CheckTimes(std::span<const double> times, bool resumed = false) {
  detail::CheckTimeGrid(times, resumed);
}

/// The interior sources at a node: their left limits at the first node of
/// a repeated pair (Forcing::sourcesBefore), the plain callbacks elsewhere;
/// at c < 1 the sources of the two nodes of the step ending at the node,
/// their moments times 1 - c and c (detail::AtStage).
std::vector<PointSource> PointSourcesAt(const Evolution::Forcing& forcing,
                                        std::span<const double> times,
                                        std::size_t node, double c = 1.0) {
  return detail::AtStage(times, node, c, forcing.sources,
                         forcing.sourcesBefore);
}
std::vector<StressGlut> StressGlutsAt(const Evolution::Forcing& forcing,
                                      std::span<const double> times,
                                      std::size_t node, double c = 1.0) {
  return detail::AtStage(times, node, c, forcing.stressGluts,
                         forcing.stressGlutsBefore);
}

/// A stage's memory state of a 1D run (StepRecipe.hpp).
ViscoelasticState Combined(
    std::span<const std::pair<double, const ViscoelasticState*>> terms,
    const RecipeStage&) {
  return CombinedState(terms);
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
  calibrated_ = false;
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
  detail::CheckLeftLimits(forcing, false);
  if (!op_->Gravitating()) return MarchWithoutGravity(times, forcing, false);
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               nullptr, forcing.tide ? &forcing.tide : nullptr, false,
               &forcing);
}

std::vector<Evolution::Snapshot> Evolution::Resume(
    const Checkpoint& from, std::span<const double> times,
    const Forcing& forcing) {
  Restore(from, times);
  detail::CheckLeftLimits(forcing, false);
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
  detail::CheckLeftLimits(forcing, true);
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
  detail::CheckLeftLimits(forcing, true);
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
  CheckTimes(times, true);
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
  using Field = std::function<std::vector<double>(double)>;
  using TideField = std::function<std::vector<std::complex<double>>(double)>;
  // the surface terms' left limits (Forcing::directLoadBefore and its
  // kin), absent where the forcing gives none
  const auto noField = Field{};
  const auto& loadBefore =
      interior != nullptr ? interior->directLoadBefore : noField;
  const auto& outsideBefore =
      interior != nullptr ? interior->directLoadOutsideWaterBudgetBefore
                          : noField;
  const auto& layerBefore =
      interior != nullptr ? interior->solidSurfaceChangeBefore : noField;
  const auto& iceBefore =
      interior != nullptr ? interior->iceThicknessBefore : noField;
  // the prescribed tide and its left limit, refused at any node where
  // either has a part below degree 2 (RequireTideDegrees)
  auto checked = [](TideField given) {
    return given ? TideField{[given = std::move(given)](double t) {
      auto value = given(t);
      RequireTideDegrees(value);
      return value;
    }}
                 : TideField{};
  };
  auto checkedTide = TideField{};
  const auto tideBefore =
      checked(interior != nullptr ? interior->tideBefore : TideField{});
  if (tide != nullptr) {
    checkedTide = checked(*tide);
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
  // Solvers for the unrelaxed table and the effective tables of each
  // (one-stage scheme, dt) a stage takes, each under the caller's
  // parameters; the Love tables themselves are cached by the operator, so
  // a repeated dt costs a table conversion only.
  auto flavours = std::map<std::pair<Scheme, double>, GRDSolver>{};
  auto solverFor = [&](Scheme family, double dt) -> const GRDSolver& {
    const auto key = std::pair(family, dt);
    auto found = flavours.find(key);
    if (found == flavours.end()) {
      found = flavours
                  .emplace(key, GRDSolver(grid,
                                          dt == 0.0 ? op.ElasticLoveNumbers()
                                                    : op.EffectiveLoveNumbers(
                                                          family, dt),
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

  // Every term of the forcing is read at a node, or at a stage of the step
  // ending there (c < 1; StepRecipe.hpp), through detail::AtStage, the one
  // place that decides which side of a repeated pair the node reads: a
  // surface term's left limit (Forcing::directLoadBefore and its kin) at
  // the first node of a pair, where the forcing gives one. A stage inside
  // the step takes the step's two nodes' values combined, no callback
  // being called between nodes.
  auto surfaceAt = [&](std::size_t node, const auto& term, const auto& before,
                       double c = 1.0) {
    return detail::AtStage(times, node, c, term, before);
  };
  // The direct load at a node (none given: none), and the potential the
  // radial steps see there: the centrifugal potential of the rotation the
  // node's closure found, and the prescribed tide.
  auto directLoadAt = [&](std::size_t node, double c = 1.0) {
    return directLoad != nullptr ? surfaceAt(node, *directLoad, loadBefore, c)
                                 : std::vector<double>(fieldSize, 0.0);
  };
  // The load outside the water budget and the layer on the solid surface
  // at a node, where the forcing gives them (empty: none).
  auto outsideAt = [&](std::size_t node, double c = 1.0) {
    return interior != nullptr
               ? surfaceAt(node, interior->directLoadOutsideWaterBudget,
                           outsideBefore, c)
               : std::vector<double>{};
  };
  auto layerAt = [&](std::size_t node, double c = 1.0) {
    return interior != nullptr
               ? surfaceAt(node, interior->solidSurfaceChange, layerBefore, c)
               : std::vector<double>{};
  };
  auto tideAt = [&](std::size_t node, double c = 1.0) {
    return surfaceAt(node, *tide, tideBefore, c);
  };
  auto iceAt = [&](std::size_t node, double c = 1.0) {
    return surfaceAt(node, *iceThickness, iceBefore, c);
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
  auto potentialAt = [&](std::size_t node, const std::array<double, 3>& omega,
                         double c = 1.0) {
    auto psi = options_.rotation == RotationalFeedback::None
                   ? grid.ZeroCoefficients()
                   : solver.CentrifugalCoefficients(omega);
    if (tide != nullptr) {
      const auto given = tideAt(node, c);
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
  auto sourceForcesAt = [&](std::size_t node, double c = 1.0) {
    return detail::ForcesOfSources(op, *toroidal, solver.SolverScales(),
                                   toNdLength_,
                                   PointSourcesAt(*interior, times, node, c),
                                   StressGlutsAt(*interior, times, node, c));
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
    nodeForces = sourceForcesAt(0);
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
  auto movingLoadsAt = [&](std::size_t node,
                           GRDSolver::NonlinearForcing& moving,
                           double c = 1.0) {
    if (directLoad != nullptr) moving.directLoad = directLoadAt(node, c);
    moving.directLoadOutsideWaterBudget = outsideAt(node, c);
    moving.solidSurfaceChange = layerAt(node, c);
    if (tide != nullptr) moving.tide = tideAt(node, c);
    // a land mask that changes in time: the node's own, which a stage
    // inside the step ending there holds too (a mask is not combined)
    if (options_.land_held) {
      moving.forcedLand = options_.land_held(times[node]);
    }
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
    moving.iceThicknessChange = iceAt(0);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      moving.iceThicknessChange[i] -= state_.iceThickness[i];
    }
    movingLoadsAt(0, moving);
    moving.rotation = options_.rotation;
    if (sourced) startSources(moving);
    auto solved = solverFor(options_.scheme, 0.0)
                      .SolveNonlinear(state_, moving, nonlinearOptions);
    response = std::move(solved.response);
    sigma_ = std::move(solved.loadCoefficients);
    ocean = std::move(solved.finalState.oceanFunction);
    psi_ = potentialAt(0, response.angularVelocity);
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
    loadGrid = directLoadAt(0);
    const auto& elasticSolver = solverFor(options_.scheme, 0.0);
    GRDSolver::GeneralisedForcing forcing;
    forcing.directLoad = loadGrid;
    forcing.directLoadOutsideWaterBudget = outsideAt(0);
    forcing.solidSurfaceChange = layerAt(0);
    if (tide != nullptr) forcing.tide = tideAt(0);
    forcing.rotation = options_.rotation;
    if (sourced) startSources(forcing);
    response = elasticSolver.SolveGeneralised(state_, forcing, sleOptions);
    sigma_ = totalLoadCoefficients(
        withOutside(loadGrid, forcing.directLoadOutsideWaterBudget),
        response.relativeSeaLevelChange);
    psi_ = potentialAt(0, response.angularVelocity);
    if (sourced) primeUnderSources();
  }
  if (stride > 0 && !sourced) {
    op.PrimeStrain(vstate_, Columns(sigma_, toMatDensity_),
                   Columns(psi_, toMatPotential_));
  }

  auto snapshots = std::vector<Snapshot>{};
  auto recorded = std::size_t{0};
  // the closures of the inner stages of the step in hand (Snapshot::Stage)
  auto innerStages = std::vector<Snapshot::Stage>{};
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
    snapshot.seaLevelIterations = r.iterations;
    snapshot.seaLevelResidual = r.error;
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
    snapshot.stages = std::move(innerStages);
    innerStages.clear();
    snapshots.push_back(std::move(snapshot));
    const auto node = ++recorded - (resumed ? 0 : 1);
    if (options_.on_snapshot) options_.on_snapshot(node, snapshots.back());
    if (!options_.keep_snapshots) snapshots.pop_back();
  };
  if (!resumed) {
    record(times[0], response, kept(0));
    checkpoint(0);
  }

  // the toroidal systems' stage states in the step in hand (sources)
  auto toroidalStages = std::vector<ToroidalViscoelastic::State>{};
  auto toroidalCombined = ToroidalViscoelastic::State{};
  for (auto node = std::size_t{1}; node < times.size(); ++node) {
    const auto dt = (times[node] - times[node - 1]) * toMatTime_;
    const auto recipe = RecipeOf(options_.scheme, dt);
    // One stage of the step (StepRecipe.hpp; the step's one stage under a
    // one-stage scheme): opened from its memory state, closed by the sea
    // level equation under its forcing (detail::AtStage), finished.
    auto stageOf = [&](const ViscoelasticState& from, const RecipeStage& stage,
                       double length, std::size_t index) {
      const auto c = stage.c;
      auto step = op.StartStep(from, length, stage.family,
                               Columns(sigma_, toMatDensity_),
                               Columns(psi_, toMatPotential_));
      const auto& effectiveSolver = solverFor(stage.family, length);
      // the stage's sources force it; the toroidal systems are stepped
      // through the same stage under theirs
      auto stepSources = [&](auto& forcing) {
        nodeForces = sourceForcesAt(node, c);
        const auto& forces = nodeForces;
        step.AddForce(forces.spheroidal);
        const auto& twistFrom = StageStart(
            stage, dt, tstate,
            std::span<const ToroidalViscoelastic::State>(toroidalStages),
            ToroidalViscoelastic::CombinedState, toroidalCombined);
        auto stepped =
            toroidal->Step(twistFrom, length, stage.family, forces.toroidal);
        toroidalStages.push_back(std::move(stepped.first));
        twist = std::move(stepped.second);
        forcing.toroidalResponse = toroidalCoefficients();
      };
      if (migrating) {
        // The stage's equilibrium is the nonlinear equation from the
        // background, the memory field entering through the response
        // channels (pyslfp's `_run_migrating`).
        GRDSolver::NonlinearForcing moving;
        moving.iceThicknessChange = iceAt(node, c);
        for (auto i = std::size_t{0}; i < fieldSize; ++i) {
          moving.iceThicknessChange[i] -= state_.iceThickness[i];
        }
        movingLoadsAt(node, moving, c);
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
        loadGrid = directLoadAt(node, c);
        coupled.directLoad = loadGrid;
        coupled.directLoadOutsideWaterBudget = outsideAt(node, c);
        coupled.solidSurfaceChange = layerAt(node, c);
        if (tide != nullptr) coupled.tide = tideAt(node, c);
        if (sourced) stepSources(coupled);
        coupled.displacementResponse =
            Coefficients(step.MemorySurface(Component::U), toNdLength_);
        coupled.potentialResponse =
            Coefficients(step.MemorySurface(Component::Phi), toNdPotential_);
        coupled.tangentialResponse =
            Coefficients(step.MemorySurface(Component::V), toNdLength_);
        coupled.inertiaMomentChange = step.InertiaMoment()[0] * toNdMoment_;
        coupled.rotation = options_.rotation;
        response =
            effectiveSolver.SolveGeneralised(state_, coupled, sleOptions);
        sigma_ = totalLoadCoefficients(
            withOutside(loadGrid, coupled.directLoadOutsideWaterBudget),
            response.relativeSeaLevelChange);
      }
      psi_ = potentialAt(node, response.angularVelocity, c);
      if (index + 1 < recipe.stages.size()) {
        // an inner stage's closure, for the derivative runs about the step
        // and the rebuild of a state between kept ones (Snapshot::Stage)
        auto inner = Snapshot::Stage{};
        if (stride > 0) {
          inner.loadCoefficients = sigma_;
          inner.tideCoefficients = psi_;
        }
        if (migrating) inner.oceanFunction = ocean;
        if (sourced && kept(node)) inner.sourceForces = nodeForces;
        innerStages.push_back(std::move(inner));
      }
      return step
          .Finish(Columns(sigma_, toMatDensity_),
                  Columns(psi_, toMatPotential_))
          .first;
    };
    const auto twistStart = tstate.t;
    vstate_ = StepByRecipe(recipe, dt, vstate_, stageOf, Combined);
    if (sourced) {
      tstate = std::move(toroidalStages.back());
      tstate.t = twistStart + dt;
      toroidalStages.clear();
    }
    record(times[node], response, kept(node));
    checkpoint(node);
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
  // (every term read at a node, or at a stage of the step ending there,
  // through detail::AtStage, as March's, a surface term's left limit at
  // the first node of a pair where the forcing gives one)
  auto surfaceAt = [&](std::size_t node, const auto& term, const auto& before,
                       double c) {
    return detail::AtStage(times, node, c, term, before);
  };
  auto loadAt = [&](std::size_t node, double c = 1.0) {
    auto total = std::vector<double>(fieldSize, 0.0);
    for (const auto& [given, before] :
         {std::pair{&forcing.directLoad, &forcing.directLoadBefore},
          std::pair{&forcing.directLoadOutsideWaterBudget,
                    &forcing.directLoadOutsideWaterBudgetBefore}}) {
      if (!*given) continue;
      const auto field = surfaceAt(node, *given, *before, c);
      if (field.empty()) continue;
      if (field.size() != fieldSize) {
        throw std::invalid_argument("a direct load lives on the solver's grid");
      }
      for (auto i = std::size_t{0}; i < fieldSize; ++i) total[i] += field[i];
    }
    if (forcing.solidSurfaceChange) {
      const auto layer = surfaceAt(node, forcing.solidSurfaceChange,
                                   forcing.solidSurfaceChangeBefore, c);
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
  auto forcesAt = [&](std::size_t node, double c = 1.0) {
    if (!sourced) return RadialSourceForces{};
    return detail::ForcesOfSources(op, toroidal, solver.SolverScales(),
                                   toNdLength_,
                                   PointSourcesAt(forcing, times, node, c),
                                   StressGlutsAt(forcing, times, node, c));
  };

  auto spheroidal = std::map<int, std::vector<double>>{};
  auto twist = ToroidalViscoelastic::Fields{};
  auto snapshots = std::vector<Snapshot>{};
  auto recorded = std::size_t{0};
  // the closures of the inner stages of the step in hand (Snapshot::Stage)
  auto innerStages = std::vector<Snapshot::Stage>{};
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
    snapshot.stages = std::move(innerStages);
    innerStages.clear();
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
    nodeForces = forcesAt(0);
    if (loaded) sigma_ = loadAt(0);
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

  // the toroidal systems' stage states in the step in hand (sources)
  auto toroidalStages = std::vector<ToroidalViscoelastic::State>{};
  auto toroidalCombined = ToroidalViscoelastic::State{};
  for (auto node = std::size_t{1}; node < times.size(); ++node) {
    const auto dt = (times[node] - times[node - 1]) * toMatTime_;
    const auto recipe = RecipeOf(options_.scheme, dt);
    // one stage of the step (StepRecipe.hpp) under the loads and the
    // sources it takes, the toroidal systems through the same stage
    auto stageOf = [&](const ViscoelasticState& from, const RecipeStage& stage,
                       double length, std::size_t index) {
      auto step = op.StartStep(from, length, stage.family, channel(sigma_));
      nodeForces = forcesAt(node, stage.c);
      step.AddForce(nodeForces.spheroidal);
      if (loaded) sigma_ = loadAt(node, stage.c);
      if (index + 1 < recipe.stages.size()) {
        // an inner stage's closure (Snapshot::Stage)
        auto inner = Snapshot::Stage{};
        if (stride > 0) {
          inner.loadCoefficients = sigma_;
          inner.tideCoefficients = psi_;
        }
        if (sourced && kept(node)) inner.sourceForces = nodeForces;
        innerStages.push_back(std::move(inner));
      }
      auto finished = step.Finish(channel(sigma_));
      spheroidal = std::move(finished.second);
      if (sourced) {
        const auto& twistFrom = StageStart(
            stage, dt, tstate,
            std::span<const ToroidalViscoelastic::State>(toroidalStages),
            ToroidalViscoelastic::CombinedState, toroidalCombined);
        auto stepped =
            toroidal.Step(twistFrom, length, stage.family, nodeForces.toroidal);
        toroidalStages.push_back(std::move(stepped.first));
        twist = std::move(stepped.second);
      }
      return std::move(finished.first);
    };
    const auto twistStart = tstate.t;
    vstate_ = StepByRecipe(recipe, dt, vstate_, stageOf, Combined);
    if (sourced) {
      tstate = std::move(toroidalStages.back());
      tstate.t = twistStart + dt;
      toroidalStages.clear();
    }
    record(times[node], kept(node));
    checkpoint(node);
  }
  return snapshots;
}

}  // namespace sl3dpp
