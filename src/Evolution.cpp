#include "sl3dpp/Evolution.hpp"

#include <cmath>
#include <format>
#include <stdexcept>

#include "sl3dpp/SurfaceGrid.hpp"

namespace sl3dpp {

SeaLevelEvolution::SeaLevelEvolution(const ViscoelasticOperator& op,
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

LoadCallable SeaLevelEvolution::Columns(
    const std::vector<std::complex<double>>& coefficients,
    double factor) const {
  return [coefficients, factor](int l,
                                double) -> std::optional<std::vector<double>> {
    std::vector<double> columns(2 * (l + 1));
    for (int m = 0; m <= l; ++m) {
      const auto c = coefficients[SurfaceGrid::Index(l, m)];
      columns[m] = factor * c.real();
      columns[l + 1 + m] = factor * c.imag();
    }
    return columns;
  };
}

std::vector<std::complex<double>> SeaLevelEvolution::Coefficients(
    const std::map<int, std::vector<double>>& perDegree, double factor) const {
  auto out = solver_->Grid().ZeroCoefficients();
  for (const auto& [l, columns] : perDegree) {
    for (int m = 0; m <= l; ++m) {
      out[SurfaceGrid::Index(l, m)] =
          factor * std::complex<double>(columns[m], columns[l + 1 + m]);
    }
  }
  return out;
}

std::vector<SeaLevelEvolution::Snapshot> SeaLevelEvolution::Run(
    std::span<const double> times,
    const std::function<std::vector<double>(double)>& iceLoad) {
  if (times.size() < 2) {
    throw std::invalid_argument("times must hold at least two entries");
  }
  for (std::size_t i = 1; i < times.size(); ++i) {
    if (times[i] <= times[i - 1]) {
      throw std::invalid_argument("times must be increasing");
    }
  }
  const auto& op = *op_;
  const auto& solver = *solver_;
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto stride = options_.record_stride;
  const auto lastNode = times.size() - 1;

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

  SeaLevelSolver::Options sleOptions;
  sleOptions.rtol = options_.rtol;
  sleOptions.anderson_memory = options_.anderson_memory;

  vstate_ = op.ZeroState([](int l) { return 2 * (l + 1); });
  vstate_.t = times[0] * toMatTime_;

  auto totalLoadCoefficients = [&](const std::vector<double>& loadGrid,
                                   const std::vector<double>& slc) {
    std::vector<double> total(fieldSize);
    for (std::size_t i = 0; i < fieldSize; ++i) {
      total[i] = loadGrid[i] +
                 solver.WaterDensity() * state_.oceanFunction[i] * slc[i];
    }
    return grid.Expand(total);
  };

  // Solvers for the unrelaxed table and the per-dt effective tables, each
  // under the caller's parameters; the Love tables themselves are cached
  // by the operator, so a repeated dt costs a table conversion only.
  std::map<double, SeaLevelSolver> flavours;
  auto solverFor = [&](double dt) -> const SeaLevelSolver& {
    auto found = flavours.find(dt);
    if (found == flavours.end()) {
      found =
          flavours
              .emplace(dt, SeaLevelSolver(grid,
                                          dt == 0.0 ? op.ElasticLoveNumbers()
                                                    : op.EffectiveLoveNumbers(
                                                          options_.scheme, dt),
                                          solver.Parameters()))
              .first;
    }
    return found->second;
  };

  // The instantaneous elastic response at the start: the right limit, so
  // a load switched on at times[0] is stepped at full order.
  auto loadGrid = iceLoad(times[0]);
  const auto& elasticSolver = solverFor(0.0);
  SeaLevelSolver::GeneralisedForcing forcing;
  forcing.directLoad = loadGrid;
  forcing.rotation = options_.rotation;
  auto response = elasticSolver.SolveGeneralised(state_, forcing, sleOptions);
  auto sigma = totalLoadCoefficients(loadGrid, response.seaLevelChange);
  auto psi = options_.rotation == RotationalFeedback::None
                 ? grid.ZeroCoefficients()
                 : solver.CentrifugalCoefficients(response.angularVelocity);
  if (stride > 0) {
    op.PrimeStrain(vstate_, Columns(sigma, toMatDensity_),
                   Columns(psi, toMatPotential_));
  }

  std::vector<Snapshot> snapshots;
  auto record = [&](double t, SeaLevelSolver::Response& r, bool keepState) {
    Snapshot snapshot;
    snapshot.t = t;
    snapshot.seaLevelChange = std::move(r.seaLevelChange);
    snapshot.displacement = std::move(r.displacement);
    snapshot.potential = std::move(r.potential);
    snapshot.angularVelocity = r.angularVelocity;
    // The internal variables only: an adjoint regenerates the strain by
    // one elastic solve under the recorded loads.
    if (keepState) {
      ViscoelasticState kept;
      kept.t = vstate_.t;
      kept.m = vstate_.m;
      snapshot.internalState = std::move(kept);
    }
    if (stride > 0) {
      snapshot.loadCoefficients = sigma;
      snapshot.tideCoefficients = psi;
    }
    snapshots.push_back(std::move(snapshot));
  };
  record(times[0], response, stride > 0);

  for (std::size_t node = 1; node < times.size(); ++node) {
    const auto dt = (times[node] - times[node - 1]) * toMatTime_;
    auto step = op.StartStep(vstate_, dt, options_.scheme,
                             Columns(sigma, toMatDensity_),
                             Columns(psi, toMatPotential_));
    const auto& effectiveSolver = solverFor(dt);
    SeaLevelSolver::GeneralisedForcing coupled;
    loadGrid = iceLoad(times[node]);
    coupled.directLoad = loadGrid;
    coupled.displacementResponse =
        Coefficients(step.MemorySurface(Component::U), toNdLength_);
    coupled.potentialResponse =
        Coefficients(step.MemorySurface(Component::Phi), toNdPotential_);
    coupled.inertiaMomentChange = step.InertiaMoment()[0] * toNdMoment_;
    coupled.rotation = options_.rotation;
    response = effectiveSolver.SolveGeneralised(state_, coupled, sleOptions);
    sigma = totalLoadCoefficients(loadGrid, response.seaLevelChange);
    psi = options_.rotation == RotationalFeedback::None
              ? grid.ZeroCoefficients()
              : solver.CentrifugalCoefficients(response.angularVelocity);
    auto [next, solutions] = step.Finish(Columns(sigma, toMatDensity_),
                                         Columns(psi, toMatPotential_));
    vstate_ = std::move(next);
    const bool keep = stride > 0 && (node % stride == 0 || node == lastNode);
    record(times[node], response, keep);
  }
  return snapshots;
}

}  // namespace sl3dpp
