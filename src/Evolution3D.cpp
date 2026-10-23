#include "sl3dpp/Evolution3D.hpp"

#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

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

std::vector<SeaLevelEvolution3D::Snapshot> SeaLevelEvolution3D::Run(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceLoad) {
  CheckTimes(times);
  return March(times, iceLoad, false);
}

std::vector<SeaLevelEvolution3D::Snapshot> SeaLevelEvolution3D::Resume(
    const Checkpoint& from, std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceLoad) {
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
  return March(times, iceLoad, true);
}

std::vector<SeaLevelEvolution3D::Snapshot> SeaLevelEvolution3D::March(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceLoad, bool resumed) {
  const auto& op = *op_;
  const auto& solver = *solver_;
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();

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
  if (!resumed) {
    // The instantaneous elastic response at the start.
    loadGrid = iceLoad(times[0]);
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
    snapshot.converged = converged;
    // The internal variables, and the strain only on request: an adjoint
    // otherwise regenerates it by one elastic solve under the recorded
    // loads.
    if (keepState && options_.record_strain) {
      snapshot.internalState = vstate_;
    } else if (keepState) {
      Viscoelastic3D::State kept;
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
    record(times[0], response, 0, 0.0, true, stride > 0);
    checkpoint(0);
  }

  for (auto node = std::size_t{1}; node < times.size(); ++node) {
    const auto dt = (times[node] - times[node - 1]) * toMatTime_;
    auto step = op.StartStep(vstate_, dt, options_.scheme,
                             Columns(sigma_, toMatDensity_),
                             Columns(psi_, toMatPotential_));
    const auto& effectiveSolver = solverFor(step.EffectiveLove());
    loadGrid = iceLoad(times[node]);

    // The combined fixed point: strain, ocean load and rotation together.
    auto [strain, info] = step.Converge(
        [&](const GridCoupling::SpectralTensor& s) {
          auto responses = step.Respond(s);
          SeaLevelSolver::GeneralisedForcing coupled;
          coupled.directLoad = loadGrid;
          coupled.displacementResponse =
              Coefficients(op.Surface(responses, Component::U), toNdLength_);
          coupled.potentialResponse = Coefficients(
              op.Surface(responses, Component::Phi), toNdPotential_);
          coupled.inertiaMomentChange =
              step.InertiaMoment(responses) * toNdMoment_;
          coupled.rotation = options_.rotation;
          response =
              effectiveSolver.SolveGeneralised(state_, coupled, sleOptions);
          sigma_ = totalLoadCoefficients(loadGrid, response.seaLevelChange);
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
    vstate_ = std::move(step).Finish(std::move(strain));
    const bool keep = stride > 0 && (node % stride == 0 || node == lastNode);
    record(times[node], response, info.iterations, info.residual,
           info.converged, keep);
    checkpoint(node);
  }
  return snapshots;
}

}  // namespace sl3dpp
