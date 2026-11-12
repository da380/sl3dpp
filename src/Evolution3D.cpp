#include "sl3dpp/Evolution3D.hpp"

#include <cmath>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/detail/InteriorSources3D.hpp"
#include "sl3dpp/detail/Shorelines.hpp"

namespace sl3dpp {

Evolution3D::Evolution3D(const Viscoelastic3D& op,
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

Viscoelastic3D::LoadCallable3D Evolution3D::Columns(
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

std::vector<Evolution3D::Complex> Evolution3D::Coefficients(
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
  if (times.empty()) {
    throw std::invalid_argument("times must hold at least one entry");
  }
  for (auto i = std::size_t{1}; i < times.size(); ++i) {
    if (times[i] <= times[i - 1]) {
      throw std::invalid_argument("times must be increasing");
    }
  }
}

}  // namespace

void Evolution3D::SetBackgroundState(GRDSolver::State state) {
  const auto fieldSize = solver_->Grid().FieldSize();
  if (state.iceThickness.size() != fieldSize ||
      state.seaLevel.size() != fieldSize ||
      state.oceanFunction.size() != fieldSize ||
      (!state.forcedLand.empty() && state.forcedLand.size() != fieldSize)) {
    throw std::invalid_argument("state fields must live on the solver's grid");
  }
  state_ = std::move(state);
}

std::vector<Evolution3D::Snapshot> Evolution3D::Run(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& directLoad) {
  CheckTimes(times);
  return March(times, &directLoad, nullptr, nullptr, false);
}

std::vector<Evolution3D::Snapshot> Evolution3D::Run(
    std::span<const double> times, const Forcing& forcing) {
  CheckTimes(times);
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               nullptr, forcing.tide ? &forcing.tide : nullptr, false,
               &forcing);
}

std::vector<Evolution3D::Snapshot> Evolution3D::Resume(
    const Checkpoint& from, std::span<const double> times,
    const Forcing& forcing) {
  Restore(from, times);
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               nullptr, forcing.tide ? &forcing.tide : nullptr, true,
               &forcing);
}

std::vector<Evolution3D::Snapshot> Evolution3D::RunMigrating(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness) {
  CheckTimes(times);
  return March(times, nullptr, &iceThickness, nullptr, false);
}

namespace {
void CheckMigratingForcing(const Evolution3D::Forcing& forcing) {
  if (forcing.tide) {
    throw std::invalid_argument(
        "a prescribed tide is not taken under migrating shorelines");
  }
}
}  // namespace

std::vector<Evolution3D::Snapshot> Evolution3D::RunMigrating(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness,
    const Forcing& forcing) {
  CheckTimes(times);
  CheckMigratingForcing(forcing);
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               &iceThickness, nullptr, false, &forcing);
}

std::vector<Evolution3D::Snapshot> Evolution3D::ResumeMigrating(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness,
    const Forcing& forcing) {
  CheckMigratingForcing(forcing);
  Restore(from, times);
  return March(times, forcing.directLoad ? &forcing.directLoad : nullptr,
               &iceThickness, nullptr, true, &forcing);
}

void Evolution3D::Restore(const Checkpoint& from,
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

std::vector<Evolution3D::Snapshot> Evolution3D::Resume(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& directLoad) {
  Restore(from, times);
  return March(times, &directLoad, nullptr, nullptr, true);
}

std::vector<Evolution3D::Snapshot> Evolution3D::ResumeMigrating(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceThickness) {
  Restore(from, times);
  return March(times, nullptr, &iceThickness, nullptr, true);
}

/// The machinery of one call's combined fixed point, which the start
/// (Start) and the steps (March) share: the flavour sea level solvers
/// under the caller's parameters, the closure of a pass through the sea
/// level equation, and the step's closer, the fixed point on the strain
/// or (Viscoelastic3D::Options::solver) the closure nested over
/// conjugate-gradient solves. A pass's load, response and ocean function
/// live here between passes, the recurrence's coefficients on the
/// evolution.
class Evolution3D::Coupled {
 public:
  Coupled(Evolution3D& owner, bool migrating)
      : owner_(owner),
        migrating_(migrating),
        control_{.max_iterations = owner.options_.max_coupled_iterations,
                 .rtol = owner.options_.coupled_rtol,
                 .anderson_memory = owner.options_.anderson_memory,
                 .anderson_rate = owner.options_.anderson_rate},
        nested_(owner.op_->GetOptions().solver ==
                Viscoelastic3D::Options::StepSolver::ConjugateGradients) {
    sleOptions.rtol = owner.options_.rtol;
    sleOptions.anderson_memory = owner.options_.anderson_memory;
    nonlinearOptions = sleOptions;
    nonlinearOptions.max_iterations = owner.options_.max_nonlinear_iterations;
  }

  /// The node's point sources (solver units) as forces on the degree
  /// systems, the sum of theirs, in the material's units.
  void SetSources(const std::vector<PointSource>& sources,
                  const std::vector<StressGlut>& gluts) {
    sourceForces = detail::ForcesOfSources3D(
        *owner_.op_, owner_.solver_->SolverScales(), owner_.toNdLength_, sources,
        gluts);
  }
  [[nodiscard]] bool HasSources() const {
    return !sourceForces.spheroidal.empty();
  }
  [[nodiscard]] bool Migrating() const { return migrating_; }
  /// The node's sources on the open step.
  void ApplySources() {
    if (HasSources()) {
      step->AddDegreeForces(sourceForces.spheroidal, sourceForces.toroidal);
    }
  }

  /// The flavour SLE solver of a Love table, keyed by the (operator-cached,
  /// address-stable) table.
  const GRDSolver& SolverFor(const LoveNumbers& table) {
    auto found = flavours_.find(&table);
    if (found == flavours_.end()) {
      const auto& solver = *owner_.solver_;
      found = flavours_
                  .emplace(&table, GRDSolver(solver.Grid(), table,
                                                  solver.Parameters()))
                  .first;
    }
    return found->second;
  }

  /// The total load's coefficients: the non-ocean load and the ocean's.
  [[nodiscard]] std::vector<Complex> TotalLoadCoefficients(
      const std::vector<double>& load, const std::vector<double>& slc) const {
    const auto& solver = *owner_.solver_;
    const auto fieldSize = solver.Grid().FieldSize();
    const auto& ocean = owner_.state_.oceanFunction;
    auto total = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      total[i] = load[i] + (outsideGrid.empty() ? 0.0 : outsideGrid[i]) +
                 solver.WaterDensity() * ocean[i] * slc[i];
    }
    return solver.Grid().Expand(total);
  }

  /// The potential the radial steps see: the centrifugal coefficients of
  /// the response's angular velocity, and the node's prescribed tide.
  [[nodiscard]] std::vector<Complex> CentrifugalCoefficients() const {
    const auto& solver = *owner_.solver_;
    auto psi = owner_.options_.rotation == RotationalFeedback::None
                   ? solver.Grid().ZeroCoefficients()
                   : solver.CentrifugalCoefficients(response.angularVelocity);
    if (!tide.empty()) {
      if (tide.size() != psi.size()) {
        throw std::invalid_argument(
            "the tide is not a coefficient vector of the solver's grid");
      }
      for (auto i = std::size_t{0}; i < psi.size(); ++i) psi[i] += tide[i];
    }
    return psi;
  }

  /// Distributed (Viscoelastic3D::Options::collectives), every process
  /// solves the sea level equation alike, but their transforms may round
  /// differently (each process plans its own FFTs): the first process's
  /// loads, ocean function and response then serve them all, so that
  /// every segment is forced alike and every process's snapshots are the
  /// same (whatever is decided from them, a calibration's next run say,
  /// is decided alike).
  void Agree() {
    const auto& collectives = owner_.op_->GetOptions().collectives;
    if (!collectives.Distributed()) return;
    for (auto* coefficients : {&owner_.sigma_, &owner_.psi_}) {
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
    if (migrating_) collectives.share(ocean);
  }

  /// A pass's closure: the sea level equation on the step's memory and
  /// lateral responses, then the load and centrifugal channels of its
  /// answer.
  std::pair<Viscoelastic3D::LoadCallable3D, Viscoelastic3D::LoadCallable3D>
  Closure(const Viscoelastic3D::Solutions& responses) {
    auto& owner = owner_;
    const auto& op = *owner.op_;
    auto displacement = owner.Coefficients(op.Surface(responses, Component::U),
                                           owner.toNdLength_);
    auto potential = owner.Coefficients(op.Surface(responses, Component::Phi),
                                        owner.toNdPotential_);
    auto tangential = owner.Coefficients(op.Surface(responses, Component::V),
                                         owner.toNdLength_);
    auto toroidal =
        owner.Coefficients(op.SurfaceToroidal(responses), owner.toNdLength_);
    const auto moment = step->InertiaMoment(responses) * owner.toNdMoment_;
    if (migrating_) {
      moving.toroidalResponse = std::move(toroidal);
      moving.displacementResponse = std::move(displacement);
      moving.potentialResponse = std::move(potential);
      moving.tangentialResponse = std::move(tangential);
      moving.inertiaMomentChange = moment;
      auto solved = effectiveSolver->SolveNonlinear(owner.state_, moving,
                                                    nonlinearOptions);
      response = std::move(solved.response);
      owner.sigma_ = std::move(solved.loadCoefficients);
      ocean = std::move(solved.finalState.oceanFunction);
    } else {
      GRDSolver::GeneralisedForcing coupled;
      coupled.directLoad = loadGrid;
      coupled.directLoadOutsideWaterBudget = outsideGrid;
      coupled.solidSurfaceChange = layerGrid;
      coupled.tide = tide;
      coupled.displacementResponse = std::move(displacement);
      coupled.potentialResponse = std::move(potential);
      coupled.tangentialResponse = std::move(tangential);
      coupled.toroidalResponse = std::move(toroidal);
      coupled.inertiaMomentChange = moment;
      coupled.rotation = owner.options_.rotation;
      response =
          effectiveSolver->SolveGeneralised(owner.state_, coupled, sleOptions);
      owner.sigma_ = TotalLoadCoefficients(loadGrid, response.relativeSeaLevelChange);
    }
    owner.psi_ = CentrifugalCoefficients();
    Agree();
    return std::pair{owner.Columns(owner.sigma_, owner.toMatDensity_),
                     owner.Columns(owner.psi_, owner.toMatPotential_)};
  }

  /// One pass of the fixed point on the strain.
  GridCoupling::SpectralTensor CoupledPass(const GridCoupling::SpectralTensor& s) {
    auto responses = step->Respond(s);
    const auto [load, tide] = Closure(responses);
    step->AddChannels(responses, load, tide);
    return step->StrainOf(responses);
  }

  /// The open step's combined fixed point: strain, ocean load and rotation
  /// together, closed as the operator's step solver says.
  std::pair<GridCoupling::SpectralTensor, Viscoelastic3D::StepInfo> Close() {
    return nested_
               ? step->Nest(
                     [this](const Viscoelastic3D::Solutions& responses) {
                       return Closure(responses);
                     },
                     control_)
               : step->Converge(
                     [this](const GridCoupling::SpectralTensor& s) {
                       return CoupledPass(s);
                     },
                     control_);
  }

  GRDSolver::Options sleOptions;
  GRDSolver::Options nonlinearOptions;
  /// The non-ocean load of the node (frozen shorelines), and its
  /// prescribed tidal potential (empty: none).
  std::vector<double> loadGrid;
  std::vector<Complex> tide;
  /// The node's direct load outside the water budget, which joins the
  /// total load, and the height of a layer on the solid surface (empty:
  /// none).
  std::vector<double> outsideGrid;
  std::vector<double> layerGrid;
  /// The node's point sources as forces (empty: none).
  DegreeFields sourceForces;
  /// The last pass's sea level response, and the node's ocean function
  /// (migrating shorelines alone).
  GRDSolver::Response response;
  std::vector<double> ocean;
  /// The open step and the sea level solver of its effective Love numbers.
  std::optional<Viscoelastic3D::OpenStep> step;
  const GRDSolver* effectiveSolver = nullptr;
  /// The nonlinear forcing of the node (migrating shorelines).
  GRDSolver::NonlinearForcing moving;

 private:
  Evolution3D& owner_;
  bool migrating_;
  std::map<const LoveNumbers*, GRDSolver> flavours_;
  Viscoelastic3D::IterationControl control_;
  bool nested_;
};

Viscoelastic3D::StepInfo Evolution3D::Start(Coupled& coupled, double t,
                                                   std::vector<double> load) {
  const auto& op = *op_;
  vstate_ = op.ZeroState();
  vstate_.t = t * toMatTime_;
  coupled.loadGrid = std::move(load);
  if (op.HasLateralModuli() || coupled.HasSources()) {
    // (a migrating start comes here with its nonlinear forcing set, and
    // the closure solves the nonlinear equation in each pass)
    // Under lateral moduli the unrelaxed operator varies laterally: the
    // instantaneous response is the combined fixed point of a zero-length
    // step. Point sources force the degree systems directly, and take
    // the same route.
    coupled.step.emplace(op.StartStep(vstate_, 0.0, options_.scheme));
    coupled.ApplySources();
    coupled.effectiveSolver = &coupled.SolverFor(coupled.step->EffectiveLove());
    auto [strain, info] = coupled.Close();
    if (!info.converged) {
      throw std::runtime_error(std::format(
          "the instantaneous coupled fixed point at the start did not "
          "converge (residual {})",
          info.residual));
    }
    op.Recycle(
        std::exchange(vstate_, std::move(*coupled.step).Finish(std::move(strain))));
    coupled.step.reset();
    return info;
  }
  // The sea level equation closed by the unrelaxed Love numbers.
  const auto& elastic = coupled.SolverFor(op.ElasticLove());
  if (coupled.Migrating()) {
    // the nonlinear one, under the node's forcing beside the ice
    auto solved = elastic.SolveNonlinear(state_, coupled.moving,
                                         coupled.nonlinearOptions);
    coupled.response = std::move(solved.response);
    sigma_ = std::move(solved.loadCoefficients);
    coupled.ocean = std::move(solved.finalState.oceanFunction);
    psi_ = coupled.CentrifugalCoefficients();
    coupled.Agree();
    return {};
  }
  GRDSolver::GeneralisedForcing forcing;
  forcing.directLoad = coupled.loadGrid;
  forcing.directLoadOutsideWaterBudget = coupled.outsideGrid;
  forcing.solidSurfaceChange = coupled.layerGrid;
  forcing.tide = coupled.tide;
  forcing.rotation = options_.rotation;
  coupled.response = elastic.SolveGeneralised(state_, forcing, coupled.sleOptions);
  sigma_ = coupled.TotalLoadCoefficients(coupled.loadGrid,
                                         coupled.response.relativeSeaLevelChange);
  psi_ = coupled.CentrifugalCoefficients();
  coupled.Agree();
  return {};
}

void Evolution3D::PrimeForRecording() {
  if (options_.record_stride > 0) {
    op_->PrimeStrain(vstate_, Columns(sigma_, toMatDensity_),
                     Columns(psi_, toMatPotential_));
  }
}

Evolution3D::Snapshot Evolution3D::MakeSnapshot(
    double t, GRDSolver::Response& response,
    const Viscoelastic3D::StepInfo& info, bool converged,
    std::vector<double> oceanFunction, bool keepState) const {
  Snapshot snapshot;
  snapshot.t = t;
  snapshot.relativeSeaLevelChange = std::move(response.relativeSeaLevelChange);
  snapshot.displacement = std::move(response.displacement);
  snapshot.potential = std::move(response.potential);
  snapshot.tangentialPotential = std::move(response.tangentialPotential);
  snapshot.toroidalPotential = std::move(response.toroidalPotential);
  snapshot.angularVelocity = response.angularVelocity;
  snapshot.coupledIterations = info.iterations;
  snapshot.coupledResidual = info.residual;
  snapshot.solveIterations = info.solveIterations;
  snapshot.converged = converged && response.converged;
  snapshot.oceanFunction = std::move(oceanFunction);
  // The internal variables, and the strain only on request: an adjoint
  // otherwise regenerates it by one elastic solve under the recorded
  // loads.
  if (keepState && (options_.record_strain || sourced_)) {
    snapshot.internalState = vstate_;
  } else if (keepState) {
    auto trimmed = Viscoelastic3D::State{};
    trimmed.t = vstate_.t;
    trimmed.m = vstate_.m;
    snapshot.internalState = std::move(trimmed);
  }
  if (options_.record_stride > 0) {
    snapshot.loadCoefficients = sigma_;
    snapshot.tideCoefficients = psi_;
  }
  return snapshot;
}

Evolution3D::Snapshot Evolution3D::Instantaneous(
    const std::vector<double>& load, double t) {
  if (load.size() != solver_->Grid().FieldSize()) {
    throw std::invalid_argument("the load must live on the solver's grid");
  }
  Coupled coupled(*this, false);
  const auto info = Start(coupled, t, load);
  PrimeForRecording();
  auto snapshot = MakeSnapshot(t, coupled.response, info, true,
                               std::move(coupled.ocean),
                               options_.record_stride > 0);
  op_->ReleaseScratch();
  return snapshot;
}

Evolution3D::Snapshot Evolution3D::Instantaneous(const Forcing& forcing,
                                                 double t) {
  sourced_ = static_cast<bool>(forcing.sources) ||
             static_cast<bool>(forcing.stressGluts);
  Coupled coupled(*this, false);
  if (forcing.tide) coupled.tide = forcing.tide(t);
  if (forcing.directLoadOutsideWaterBudget) {
    coupled.outsideGrid = forcing.directLoadOutsideWaterBudget(t);
  }
  if (forcing.solidSurfaceChange) coupled.layerGrid = forcing.solidSurfaceChange(t);
  if (forcing.sources || forcing.stressGluts) {
    coupled.SetSources(
        forcing.sources ? forcing.sources(t) : std::vector<PointSource>{},
        forcing.stressGluts ? forcing.stressGluts(t) : std::vector<StressGlut>{});
  }
  auto load = forcing.directLoad
                  ? forcing.directLoad(t)
                  : std::vector<double>(solver_->Grid().FieldSize(), 0.0);
  if (load.size() != solver_->Grid().FieldSize()) {
    throw std::invalid_argument("the load must live on the solver's grid");
  }
  const auto info = Start(coupled, t, std::move(load));
  PrimeForRecording();
  auto snapshot = MakeSnapshot(t, coupled.response, info, true,
                               std::move(coupled.ocean),
                               options_.record_stride > 0);
  op_->ReleaseScratch();
  return snapshot;
}

std::vector<Evolution3D::Snapshot> Evolution3D::March(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>* directLoad,
    const std::function<std::vector<double>(double)>* iceThickness,
    const std::function<std::vector<std::complex<double>>(double)>* tide,
    bool resumed, const Forcing* interior) {
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

  sourced_ = interior != nullptr && (interior->sources || interior->stressGluts);
  if (sourced_ && options_.record_stride > 1) {
    throw std::invalid_argument(
        "a run under interior sources is recorded at every node "
        "(record_stride 1): the states between could not be rebuilt "
        "without the sources' forces");
  }
  if (sourced_ && options_.record_stride > 0 && options_.on_state) {
    throw std::invalid_argument(
        "a recorded run under interior sources keeps its states on the "
        "snapshots, with their strains");
  }
  Coupled coupled(*this, migrating);
  auto& response = coupled.response;
  auto& ocean = coupled.ocean;
  // The forcing at a node: the direct load (none given: none) and the
  // prescribed tide, which the coupling carries through the node's passes.
  auto directLoadAt = [&](double t) {
    return directLoad != nullptr ? (*directLoad)(t)
                                 : std::vector<double>(fieldSize, 0.0);
  };
  auto setTide = [&](double t) {
    if (tide != nullptr) coupled.tide = (*tide)(t);
    if (interior != nullptr && interior->directLoadOutsideWaterBudget) {
      coupled.outsideGrid = interior->directLoadOutsideWaterBudget(t);
    }
    if (interior != nullptr && interior->solidSurfaceChange) {
      coupled.layerGrid = interior->solidSurfaceChange(t);
    }
    if (interior != nullptr && (interior->sources || interior->stressGluts)) {
      coupled.SetSources(
          interior->sources ? interior->sources(t) : std::vector<PointSource>{},
          interior->stressGluts ? interior->stressGluts(t) : std::vector<StressGlut>{});
    }
  };

  const auto stride = options_.record_stride;
  const auto lastNode = times.size() - 1;

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

  // The forcing beside the ice of a migrating node, changes from the
  // background: the loads, the layer and the sources.
  auto setMoving = [&](double t) {
    auto& moving = coupled.moving;
    moving = GRDSolver::NonlinearForcing{};
    moving.rotation = options_.rotation;
    if (interior == nullptr) return;
    setTide(t);
    if (directLoad != nullptr) moving.directLoad = (*directLoad)(t);
    moving.directLoadOutsideWaterBudget = coupled.outsideGrid;
    moving.solidSurfaceChange = coupled.layerGrid;
  };

  auto startInfo = Viscoelastic3D::StepInfo{};
  if (!resumed && migrating && interior != nullptr) {
    // The start under a forcing beside the ice: solved, from the
    // background.
    setMoving(times[0]);
    startInfo = Start(coupled, times[0], std::vector<double>(fieldSize, 0.0));
  } else if (!resumed && migrating) {
    // The background is the state at times[0]: the run starts undisturbed.
    vstate_ = op.ZeroState();
    vstate_.t = times[0] * toMatTime_;
    response.relativeSeaLevelChange.assign(fieldSize, 0.0);
    response.displacement.assign(fieldSize, 0.0);
    response.potential.assign(fieldSize, 0.0);
    response.tangentialPotential.assign(fieldSize, 0.0);
    sigma_ = grid.ZeroCoefficients();
    psi_ = grid.ZeroCoefficients();
    ocean = state_.oceanFunction;
  } else if (!resumed) {
    // The instantaneous elastic response at the start (Instantaneous).
    setTide(times[0]);
    startInfo = Start(coupled, times[0], directLoadAt(times[0]));
  }
  PrimeForRecording();

  auto snapshots = std::vector<Snapshot>{};
  auto record = [&](double t, GRDSolver::Response& r,
                    const Viscoelastic3D::StepInfo& info, bool converged,
                    bool keepState) {
    auto snapshot =
        MakeSnapshot(t, r, info, converged, std::move(ocean), keepState);
    if (sourced_ && keepState) snapshot.sourceForces = coupled.sourceForces;
    // With on_state, the kept state is handed out instead.
    auto kept = std::optional<Viscoelastic3D::State>{};
    if (options_.on_state) std::swap(kept, snapshot.internalState);
    snapshots.push_back(std::move(snapshot));
    const auto node = snapshots.size() - (resumed ? 0 : 1);
    if (options_.on_snapshot) options_.on_snapshot(node, snapshots.back());
    if (kept && options_.on_state) options_.on_state(node, std::move(*kept));
  };
  if (!resumed) {
    record(times[0], response, startInfo, true, stride > 0);
    checkpoint(0);
  }

  for (auto node = std::size_t{1}; node < times.size(); ++node) {
    const auto dt = (times[node] - times[node - 1]) * toMatTime_;
    auto& step = coupled.step;
    step.emplace(op.StartStep(vstate_, dt, options_.scheme,
                              Columns(sigma_, toMatDensity_),
                              Columns(psi_, toMatPotential_)));
    coupled.effectiveSolver = &coupled.SolverFor(step->EffectiveLove());
    // Under migrating shorelines each pass solves the nonlinear equation
    // from the background, the memory and lateral responses entering
    // through the response channels, so the ocean function follows the
    // flotation condition inside the fixed point.
    auto& moving = coupled.moving;
    if (migrating) {
      setMoving(times[node]);
      moving.iceThicknessChange = (*iceThickness)(times[node]);
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        moving.iceThicknessChange[i] -= state_.iceThickness[i];
      }
    } else {
      coupled.loadGrid = directLoadAt(times[node]);
      setTide(times[node]);
    }
    coupled.ApplySources();

    // The combined fixed point: strain, ocean load and rotation together.
    auto [strain, info] = coupled.Close();
    // the replaced state's tensors go to the next StartStep
    op.Recycle(
        std::exchange(vstate_, std::move(*step).Finish(std::move(strain))));
    step.reset();
    const bool keep = stride > 0 && (node % stride == 0 || node == lastNode);
    record(times[node], response, info, info.converged, keep);
    checkpoint(node);
  }
  op.ReleaseScratch();
  return snapshots;
}

}  // namespace sl3dpp
