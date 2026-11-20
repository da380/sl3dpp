#include "sl3dpp/Viscoelastic.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>
#include <string_view>

#include "sl3dpp/StepRecipe.hpp"
#include "sl3dpp/detail/Parallel.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

namespace {

// A channel's columns at a degree and time, from the caller's callable
// (called serially: it need not be thread-safe), checked for shape.
std::optional<std::vector<double>> FetchColumns(const LoadCallable& channel,
                                                int l, double t, int ncols) {
  if (!channel) return std::nullopt;
  auto columns = channel(l, t);
  if (columns && static_cast<int>(columns->size()) != ncols) {
    throw std::invalid_argument(
        std::format("a degree-{} channel must supply {} columns, got {}", l,
                    ncols, columns->size()));
  }
  return columns;
}

// Adds a channel's columns through its unit response, column by column.
void AddColumns(std::vector<double>& x, const std::vector<double>& units,
                int unit, int ndof, const std::vector<double>& columns,
                int ncols) {
  for (auto c = 0; c < ncols; ++c) {
    const auto amplitude = columns[c];
    if (amplitude == 0.0) continue;
    for (auto d = 0; d < ndof; ++d) {
      x[static_cast<std::size_t>(c) * ndof + d] +=
          units[static_cast<std::size_t>(unit) * ndof + d] * amplitude;
    }
  }
}

}  // namespace

// The per-degree workspace and the flavour struct live in
// detail/ViscoelasticDegree.hpp, shared with the adjoint sweeps.

namespace {

/// The rheology an operator steps: the one given, or with the internal
/// variables at the GLL nodes and a perturbation of the moduli there, the
/// rheology with its nodal moduli scaled (the points are the nodes).
std::shared_ptr<const Rheology> SteppedRheology(
    const Rheology& rheology, const ViscoelasticOperator::Options& options) {
  const auto& moduli = options.moduli;
  if (moduli.Empty() || options.memory_points != MemoryPointSet::Gll) {
    return std::make_shared<const Rheology>(rheology);
  }
  const auto size = rheology.BaseMaterial().Rho().size();
  if ((!moduli.shear.empty() && moduli.shear.size() != size) ||
      (!moduli.bulk.empty() && moduli.bulk.size() != size)) {
    throw std::invalid_argument(
        "a perturbation of the moduli has (element, point) entries");
  }
  MaterialDirection d;
  const auto kappa = rheology.Kappa();
  const auto muU = rheology.MuUnrelaxed();
  d.kappa.assign(size, 0.0);
  d.mu.assign(size, 0.0);
  for (auto i = std::size_t{0}; i < size; ++i) {
    if (!moduli.bulk.empty()) d.kappa[i] = moduli.bulk[i] * kappa[i];
    if (!moduli.shear.empty()) d.mu[i] = moduli.shear[i] * muU[i];
  }
  auto branches = std::vector<Rheology::Branch>{};
  for (auto k = std::size_t{0}; k < rheology.BranchCount(); ++k) {
    branches.push_back(rheology.BranchAt(k));
    if (moduli.shear.empty()) continue;
    for (auto i = std::size_t{0}; i < size; ++i) {
      branches.back().mu[i] *= 1.0 + moduli.shear[i];
    }
  }
  auto shifted =
      Rheology::Create(rheology.BaseMaterial().Shifted(d, 1.0), std::move(branches));
  if (!shifted) {
    throw std::invalid_argument(
        "the perturbed moduli do not make a rheology");
  }
  return std::make_shared<const Rheology>(std::move(*shifted));
}

}  // namespace

ViscoelasticOperator::ViscoelasticOperator(const Rheology& rheology, int lmax,
                                           Options options)
    : rheology_(SteppedRheology(rheology, options)),
      lmax_(lmax),
      options_(options),
      points_(std::make_unique<const MemoryPoints>(MemoryPoints::Of(
          options.memory_points, rheology.BaseMaterial().Mesh()))) {
  if (lmax < 0) throw std::invalid_argument("lmax must be non-negative");
  const auto gll = options.memory_points == MemoryPointSet::Gll;
  const auto& moduli = options_.moduli;
  const auto pointCount =
      static_cast<std::size_t>(rheology.BaseMaterial().Mesh().ElementCount()) *
      points_->count;
  if ((!moduli.shear.empty() && moduli.shear.size() != pointCount) ||
      (!moduli.bulk.empty() && moduli.bulk.size() != pointCount)) {
    throw std::invalid_argument(
        "a perturbation of the moduli has (element, point) entries");
  }
  auto sample = [&](const std::vector<double>& nodal) {
    if (gll) return nodal;
    auto out = points_->Interpolate(nodal);
    for (auto& v : out) v = std::max(v, 0.0);
    return out;
  };
  auto atPoints = [&](std::span<const double> nodal) {
    return gll ? std::vector<double>(nodal.begin(), nodal.end())
               : points_->Interpolate(nodal);
  };
  // The reference's moduli at the points, before any perturbation.
  muUnrelaxedAtPoints_ = atPoints(rheology.MuLongTerm());
  kappaAtPoints_ = atPoints(rheology.Kappa());
  const auto longTermAtPoints = muUnrelaxedAtPoints_;
  for (auto k = std::size_t{0}; k < rheology.BranchCount(); ++k) {
    branchFraction_.push_back(sample(rheology.BranchAt(k).mu));
    for (auto i = std::size_t{0}; i < pointCount; ++i) {
      muUnrelaxedAtPoints_[i] += branchFraction_.back()[i];
    }
  }
  for (auto& fraction : branchFraction_) {
    for (auto i = std::size_t{0}; i < pointCount; ++i) {
      fraction[i] = muUnrelaxedAtPoints_[i] > 0.0
                        ? fraction[i] / muUnrelaxedAtPoints_[i]
                        : 0.0;
    }
  }
  for (auto k = std::size_t{0}; k < rheology_->BranchCount(); ++k) {
    pointMu_.push_back(sample(rheology_->BranchAt(k).mu));
    pointItau_.push_back(sample(rheology_->BranchAt(k).itau));
  }
  if (!gll) {
    // The shear energy splits into the long-term part at the nodes and
    // the branches' at the points (MemoryPoints.hpp).
    nodalMaterial_ =
        std::make_unique<const Material>(rheology_->LongTermMaterial());
    elasticShear_.assign(pointCount, 0.0);
    if (!moduli.shear.empty()) {
      // The branches scaled at the points; the long-term shear's
      // perturbation a point term that every operator carries whole.
      longTermShearShift_.resize(pointCount);
      for (auto i = std::size_t{0}; i < pointCount; ++i) {
        longTermShearShift_[i] = moduli.shear[i] * longTermAtPoints[i];
        elasticShear_[i] = longTermShearShift_[i];
      }
      for (auto& mu : pointMu_) {
        for (auto i = std::size_t{0}; i < pointCount; ++i) {
          mu[i] *= 1.0 + moduli.shear[i];
        }
      }
    }
    if (!moduli.bulk.empty()) {
      pointBulk_.resize(pointCount);
      for (auto i = std::size_t{0}; i < pointCount; ++i) {
        pointBulk_[i] = moduli.bulk[i] * kappaAtPoints_[i];
      }
    }
    for (const auto& mu : pointMu_) {
      for (auto i = std::size_t{0}; i < mu.size(); ++i)
        elasticShear_[i] += mu[i];
    }
    elasticPointShear_ = {points_.get(), elasticShear_, pointBulk_};
  }
}

PointMaterialDirection AtMemoryPoints(const MemoryPoints& points,
                                      const MaterialDirection& nodal) {
  const auto gll = points.set == MemoryPointSet::Gll;
  auto at = [&](const std::vector<double>& a) {
    return a.empty() || gll ? a : points.Interpolate(a);
  };
  return {.kappa = at(nodal.kappa), .mu = at(nodal.mu)};
}

double RelaxationSite::StepRatio(std::span<const double> times,
                                 double unit) const {
  if (!std::isfinite(time)) return 0.0;
  auto longest = 0.0;
  for (auto n = std::size_t{1}; n < times.size(); ++n) {
    longest = std::max(longest, times[n] - times[n - 1]);
  }
  return longest * unit / time;
}

RelaxationSite ViscoelasticOperator::ShortestRelaxation() const {
  const auto& mesh = rheology_->BaseMaterial().Mesh();
  const auto np = static_cast<std::size_t>(points_->count);
  const auto radii = points_->Radii(mesh);
  auto site = RelaxationSite{};
  auto largest = 0.0;
  for (auto k = std::size_t{0}; k < pointItau_.size(); ++k) {
    for (auto i = std::size_t{0}; i < pointItau_[k].size(); ++i) {
      const auto itau = pointItau_[k][i];
      // ties to the innermost point, as Viscoelastic3D's
      const auto shorter =
          itau > largest || (itau == largest && radii[i] < site.radius);
      if (!(pointMu_[k][i] > 0.0) || !(itau > 0.0) || !shorter) continue;
      largest = itau;
      const auto e = static_cast<int>(i / np);
      site.branch = k;
      site.element = e;
      site.layer = mesh.LayerOf(e);
      site.radius = radii[i];
    }
  }
  if (largest > 0.0) site.time = 1.0 / largest;
  return site;
}

ViscoelasticOperator::~ViscoelasticOperator() = default;
ViscoelasticOperator::ViscoelasticOperator(ViscoelasticOperator&&) noexcept =
    default;
ViscoelasticOperator& ViscoelasticOperator::operator=(
    ViscoelasticOperator&&) noexcept = default;

const DegreeOperator& ViscoelasticOperator::UnrelaxedSystem(int l) const {
  return Deg(l).system;
}

std::pair<int, int> ViscoelasticOperator::MemoryElements(int l) const {
  const auto& deg = Deg(l);
  return {deg.e0, deg.ne};
}

ViscoelasticOperator::Degree& ViscoelasticOperator::Deg(int l) const {
  auto found = degrees_.find(l);
  if (found == degrees_.end()) {
    found = degrees_
                .emplace(
                    l, std::make_unique<Degree>(
                           *rheology_, l, options_, NodalMaterial(),
                           ElasticPointShear(), *points_, pointMu_, pointItau_))
                .first;
  }
  return *found->second;
}

namespace {
/// The refusal of a multi-stage scheme where one set of step weights or
/// one effective operator is asked for at a step of nonzero length.
void RefuseMultiStage(Scheme scheme, double dt, std::string_view what) {
  if (IsMultiStage(scheme) && dt != 0.0) {
    throw std::invalid_argument(std::format(
        "Sdirk2 has no {} at a step of nonzero length: it steps in two "
        "backward Euler stages of length gamma dt (StepRecipe.hpp), each "
        "BackwardEuler's at that length",
        what));
  }
}
}  // namespace

const std::vector<ViscoelasticOperator::BranchWeights>&
ViscoelasticOperator::WeightsOfOrder(Scheme scheme, double dt,
                                     int order) const {
  RefuseMultiStage(scheme, dt, "the step weights");
  auto& cache = order == 0   ? weights_
                : order == 1 ? weightDerivatives_
                             : weightSecondDerivatives_;
  const auto key = std::pair(static_cast<int>(scheme), dt);
  auto found = cache.find(key);
  if (found == cache.end()) {
    auto out = std::vector<BranchWeights>{};
    for (auto k = std::size_t{0}; k < rheology_->BranchCount(); ++k) {
      const auto& itau = pointItau_[k];
      BranchWeights w;
      w.e.resize(itau.size());
      if (scheme == Scheme::BackwardEuler) {
        for (auto i = std::size_t{0}; i < itau.size(); ++i) {
          const auto h = dt * itau[i];
          w.e[i] = order == 0   ? BackwardEulerWeight(h)
                   : order == 1 ? BackwardEulerWeightDerivative(h)
                                : BackwardEulerWeightSecondDerivative(h);
        }
      } else {
        w.alpha.resize(itau.size());
        w.beta.resize(itau.size());
        for (auto i = std::size_t{0}; i < itau.size(); ++i) {
          const auto h = dt * itau[i];
          const auto tw = order == 0   ? ExponentialTrapezoidWeights(h)
                          : order == 1 ? ExponentialTrapezoidWeightDerivatives(h)
                                       : ExponentialTrapezoidWeightSecondDerivatives(h);
          w.e[i] = tw.e;
          w.alpha[i] = tw.alpha;
          w.beta[i] = tw.beta;
        }
      }
      out.push_back(std::move(w));
    }
    found = cache.emplace(key, std::move(out)).first;
  }
  return found->second;
}

const std::vector<ViscoelasticOperator::BranchWeights>&
ViscoelasticOperator::SchemeWeightsOf(Scheme scheme, double dt) const {
  return WeightsOfOrder(scheme, dt, 0);
}

const std::vector<ViscoelasticOperator::BranchWeights>&
ViscoelasticOperator::SchemeWeightDerivativesOf(Scheme scheme,
                                                double dt) const {
  return WeightsOfOrder(scheme, dt, 1);
}

const std::vector<ViscoelasticOperator::BranchWeights>&
ViscoelasticOperator::SchemeWeightSecondDerivativesOf(Scheme scheme,
                                                      double dt) const {
  return WeightsOfOrder(scheme, dt, 2);
}

std::vector<double> ViscoelasticOperator::PointShearModulus(Scheme scheme,
                                                            double dt) const {
  // a zero-length step's weights are one: the elastic operators' own
  if (dt == 0.0 && options_.memory_points != MemoryPointSet::Gll) {
    return elasticShear_;
  }
  const auto& weights = SchemeWeightsOf(scheme, dt);
  auto s = std::vector<double>(
      static_cast<std::size_t>(BaseMaterial().Mesh().ElementCount()) *
          points_->count,
      0.0);
  for (auto k = std::size_t{0}; k < weights.size(); ++k) {
    for (auto i = std::size_t{0}; i < s.size(); ++i) {
      const auto unrelaxed = scheme == Scheme::BackwardEuler
                                 ? weights[k].e[i]
                                 : 1.0 - weights[k].beta[i];
      s[i] += pointMu_[k][i] * unrelaxed;
    }
  }
  for (auto i = std::size_t{0}; i < longTermShearShift_.size(); ++i) {
    s[i] += longTermShearShift_[i];
  }
  return s;
}

std::vector<std::vector<double>> ViscoelasticOperator::ItauKernelsAtNodes(
    std::vector<std::vector<double>> atPoints) const {
  if (options_.memory_points == MemoryPointSet::Gll) return atPoints;
  auto out = std::vector<std::vector<double>>{};
  for (auto k = std::size_t{0}; k < atPoints.size(); ++k) {
    // itau at a point is max(interpolant, 0): the chain rule is the
    // interpolation's transpose where the interpolant is positive.
    const auto raw = points_->Interpolate(rheology_->BranchAt(k).itau);
    auto masked = std::move(atPoints[k]);
    for (auto i = std::size_t{0}; i < masked.size(); ++i) {
      if (raw[i] < 0.0) masked[i] = 0.0;
    }
    out.push_back(points_->InterpolateTranspose(masked));
  }
  return out;
}

std::vector<std::vector<double>> ViscoelasticOperator::ItauPerturbationAtPoints(
    const std::vector<std::vector<double>>& nodal) const {
  const auto K = rheology_->BranchCount();
  const auto meshSize = rheology_->BaseMaterial().Rho().size();
  if (nodal.size() != K) {
    throw std::invalid_argument(std::format(
        "an itau perturbation gives one nodal array per branch: {} branches",
        K));
  }
  for (const auto& a : nodal) {
    if (a.size() != meshSize) {
      throw std::invalid_argument(
          "an itau perturbation's arrays live on the material's mesh "
          "(element, gll)");
    }
  }
  if (options_.memory_points == MemoryPointSet::Gll) return nodal;
  auto out = std::vector<std::vector<double>>{};
  for (auto k = std::size_t{0}; k < K; ++k) {
    // itau at a point is max(interpolant, 0): the derivative is the
    // interpolant where it is positive, zero where the clamp acted.
    const auto raw = points_->Interpolate(rheology_->BranchAt(k).itau);
    auto at = points_->Interpolate(nodal[k]);
    for (auto i = std::size_t{0}; i < at.size(); ++i) {
      if (raw[i] < 0.0) at[i] = 0.0;
    }
    out.push_back(std::move(at));
  }
  return out;
}

Material ViscoelasticOperator::EffectiveMaterial(Scheme scheme,
                                                 double dt) const {
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument("no effective operator for ExponentialEuler");
  }
  RefuseMultiStage(scheme, dt, "one effective material");
  if (options_.memory_points != MemoryPointSet::Gll) {
    throw std::logic_error(
        "with internal variables away from the GLL nodes a step's "
        "effective solid is the long-term material plus a shear at the "
        "memory points (MemoryPoints.hpp), not a material");
  }
  // a zero-length step's weights are one: the unrelaxed material itself
  if (dt == 0.0) return BaseMaterial();
  const auto& weights = SchemeWeightsOf(scheme, dt);
  auto w = std::vector<std::vector<double>>{};
  for (const auto& branch : weights) {
    if (scheme == Scheme::BackwardEuler) {
      w.push_back(branch.e);
    } else {
      auto phi1 = std::vector<double>(branch.beta.size());
      for (auto i = std::size_t{0}; i < phi1.size(); ++i) {
        phi1[i] = 1.0 - branch.beta[i];
      }
      w.push_back(std::move(phi1));
    }
  }
  return BaseMaterial().WithModuli(rheology_->EffectiveModuli(w)).value();
}

std::shared_ptr<ViscoelasticOperator::Flavour> ViscoelasticOperator::FlavourOf(
    Scheme scheme, double dt) const {
  const auto key = std::pair(static_cast<int>(scheme), dt);
  for (auto it = flavours_.begin(); it != flavours_.end(); ++it) {
    if (it->first == key) {
      flavours_.splice(flavours_.begin(), flavours_, it);
      return flavours_.front().second;
    }
  }
  constexpr auto kFlavourCache = std::size_t{4};
  while (flavours_.size() >= kFlavourCache) flavours_.pop_back();
  auto flavour = std::make_shared<Flavour>();
  if (options_.memory_points == MemoryPointSet::Gll) {
    flavour->material =
        std::make_unique<Material>(EffectiveMaterial(scheme, dt));
  } else {
    // The long-term material plus the branches' unrelaxed share at the
    // points.
    flavour->material = std::make_unique<Material>(NodalMaterial());
    flavour->pointShearModulus = PointShearModulus(scheme, dt);
    flavour->pointShear = {points_.get(), flavour->pointShearModulus,
                           pointBulk_};
  }
  flavours_.emplace_front(key, flavour);
  return flavour;
}

const DegreeOperator& ViscoelasticOperator::EffectiveSystem(Flavour& flavour,
                                                            int l) const {
  auto found = flavour.systems.find(l);
  if (found == flavour.systems.end()) {
    found =
        flavour.systems
            .emplace(l, DegreeOperator(
                            *flavour.material, l,
                            {.eps = options_.eps,
                             .truncation = options_.truncation,
                             .point_shear = flavour.pointShear.points != nullptr
                                                ? &flavour.pointShear
                                                : nullptr,
                             .gravitating = options_.gravitating,
                             .cuts = options_.radial_cuts}))
            .first;
  }
  return found->second;
}

namespace {
/// The refusal of what only the problem with gravity has.
void RequireGravity(const ViscoelasticOperator& op, std::string_view what) {
  if (!op.Gravitating()) {
    throw std::invalid_argument(std::format(
        "{} needs gravity: the operator solves the problem without it, "
        "forced by the load channel (a surface mass's weight) and interior "
        "sources (ViscoelasticStep::AddForce)",
        what));
  }
}

/// Without gravity the tide channel is refused: an external potential
/// acts through the potential, which the problem has not.
void RefuseTideWithoutGravity(const ViscoelasticOperator& op,
                              const LoadCallable& tide) {
  if (tide && !op.Gravitating()) {
    throw std::invalid_argument(
        "a tide channel (an external potential) needs gravity: the operator "
        "solves the problem without it");
  }
}

/// The load channel's columns at a degree and time (FetchColumns).
/// Without gravity its degree-1 columns are removed: a degree-1 traction
/// exerts a net force on the body, which nothing balances, and the load is
/// projected on the loads that exert none (Evolution.hpp).
std::optional<std::vector<double>> LoadColumns(const ViscoelasticOperator& op,
                                               const LoadCallable& load, int l,
                                               double t, int ncols) {
  auto columns = FetchColumns(load, l, t, ncols);
  if (columns && l == 1 && !op.Gravitating()) {
    std::ranges::fill(*columns, 0.0);
  }
  return columns;
}

/// The tide channel's columns at a degree and time (FetchColumns). Its
/// degree-1 columns must vanish: a tide is taken relative to the centre
/// of mass (kLowestTideDegree), the channel's degree 0 the spherical
/// mean of a centrifugal change.
std::optional<std::vector<double>> TideColumns(const LoadCallable& tide, int l,
                                               double t, int ncols) {
  auto columns = FetchColumns(tide, l, t, ncols);
  if (columns && l == 1 &&
      std::ranges::any_of(*columns, [](double v) { return v != 0.0; })) {
    throw std::invalid_argument(
        "a tide channel has no degree-1 part: tides are taken relative to "
        "the centre of mass, its degree-1 columns must vanish");
  }
  return columns;
}
}  // namespace

const std::vector<double>& ViscoelasticOperator::UnitResponses(Flavour* flavour,
                                                               int l) const {
  auto& deg = Deg(l);
  auto& cache = flavour == nullptr ? unrelaxedUnits_ : flavour->units;
  auto found = cache.find(l);
  if (found == cache.end()) {
    const auto& system =
        flavour == nullptr ? deg.system : EffectiveSystem(*flavour, l);
    auto rhs = std::vector<double>(static_cast<std::size_t>(deg.ndof) * 2, 0.0);
    if (options_.gravitating) {
      std::ranges::copy(deg.loadVec, rhs.begin());
      std::ranges::copy(deg.tideVec, rhs.begin() + deg.ndof);
      found = cache.emplace(l, system.Solve(rhs, 2)).first;
    } else if (l == 1) {
      // without gravity the degree-1 load is refused (LoadColumns) and
      // there is no tide: no response to solve
      found = cache.emplace(l, std::move(rhs)).first;
    } else {
      // without gravity the load's weight alone, the normal traction g
      // pressing on the surface per unit surface density, -g a^2 on U(a)
      // (DegreeOperator::Load on the system without the potential); no
      // tide
      std::ranges::copy(deg.loadVec, rhs.begin());
      found = cache.emplace(l, system.Solve(rhs, 2)).first;
    }
  }
  return found->second;
}

const std::vector<double>& ViscoelasticOperator::SplitUnitResponses(
    Flavour* flavour, int l) const {
  RequireGravity(*this, "a surface load's split responses");
  auto& deg = Deg(l);
  auto& cache =
      flavour == nullptr ? unrelaxedSplitUnits_ : flavour->splitUnits;
  auto found = cache.find(l);
  if (found == cache.end()) {
    const auto& system =
        flavour == nullptr ? deg.system : EffectiveSystem(*flavour, l);
    const auto force = deg.system.Load(LoadPart::Force);
    const auto potential = deg.system.Load(LoadPart::Potential);
    const auto tangential = deg.system.Tangential();
    auto rhs = std::vector<double>(static_cast<std::size_t>(deg.ndof) * 3, 0.0);
    std::ranges::copy(force, rhs.begin());
    std::ranges::copy(potential, rhs.begin() + deg.ndof);
    std::ranges::copy(tangential, rhs.begin() + 2 * deg.ndof);
    found = cache.emplace(l, system.Solve(rhs, 3)).first;
  }
  return found->second;
}

namespace {
/// A degree's column count in a state: the count it carries, or, of a
/// state without one, the size of its memory over `perColumn`, the
/// memory's size per column.
int ColumnsOf(const ViscoelasticState& state, int l, std::size_t memory,
              std::size_t perColumn) {
  if (const auto found = state.columns.find(l); found != state.columns.end()) {
    return found->second;
  }
  if (perColumn == 0) {
    throw std::logic_error(
        "a state of a rheology without relaxation branches carries its "
        "column counts (ViscoelasticOperator::ZeroState)");
  }
  return static_cast<int>(memory / perColumn);
}
}  // namespace

ViscoelasticState CombinedState(
    std::span<const std::pair<double, const ViscoelasticState*>> terms) {
  if (terms.empty()) {
    throw std::invalid_argument("a combined state takes at least one state");
  }
  using Fields = std::map<int, std::vector<double>>;
  auto fields = std::vector<std::pair<double, const Fields*>>{};
  for (const auto& [a, state] : terms) fields.emplace_back(a, &state->m);
  ViscoelasticState out;
  out.m = CombineFields<Fields>(fields);
  out.columns = terms.front().second->columns;
  out.t = terms.front().second->t;
  return out;
}

ViscoelasticState ViscoelasticOperator::ZeroState(
    const std::function<int(int)>& ncols) const {
  ViscoelasticState state;
  const auto K = rheology_->BranchCount();
  for (auto l = 0; l <= lmax_; ++l) {
    const auto& deg = Deg(l);
    state.columns[l] = ncols(l);
    state.m[l].assign(K * 3 * deg.NodalSize(ncols(l)), 0.0);
  }
  return state;
}

std::map<int, std::vector<double>> ViscoelasticOperator::SolveElastic(
    const ViscoelasticState& state, const LoadCallable& load,
    const LoadCallable& tide) const {
  // Degree tasks: the workspaces, unit responses and channel columns are
  // resolved here, serially (the caches fill lazily, and the callables
  // need not be thread-safe); each task writes only its own solution.
  struct Task {
    int l = 0;
    int ncols = 0;
    const std::vector<double>* m = nullptr;
    const Degree* deg = nullptr;
    const std::vector<double>* units = nullptr;
    std::optional<std::vector<double>> load, tide;
    std::vector<double> x;
  };
  RefuseTideWithoutGravity(*this, tide);
  const auto K = rheology_->BranchCount();
  auto tasks = std::vector<Task>{};
  for (const auto& [l, m] : state.m) {
    Task task;
    task.l = l;
    task.m = &m;
    task.deg = &Deg(l);
    task.ncols =
        ColumnsOf(state, l, m.size(), K * 3 * task.deg->ne * task.deg->np);
    if (load || tide) {
      task.units = &UnitResponses(nullptr, l);
      task.load = LoadColumns(*this, load, l, state.t, task.ncols);
      task.tide = TideColumns(tide, l, state.t, task.ncols);
    }
    tasks.push_back(std::move(task));
  }
  const auto count = static_cast<int>(tasks.size());
  detail::ForEachTask(options_.parallel, count, [&](int i) {
    auto& task = tasks[count - 1 - i];  // the largest degrees first
    const auto& deg = *task.deg;
    task.x = deg.system.Solve(deg.MemoryForce(*task.m, task.ncols), task.ncols);
    if (task.load) {
      AddColumns(task.x, *task.units, 0, deg.ndof, *task.load, task.ncols);
    }
    if (task.tide) {
      AddColumns(task.x, *task.units, 1, deg.ndof, *task.tide, task.ncols);
    }
  });
  auto solutions = std::map<int, std::vector<double>>{};
  for (auto& task : tasks) solutions.emplace(task.l, std::move(task.x));
  return solutions;
}

std::map<int, std::vector<double>> ViscoelasticOperator::Strain(
    const std::map<int, std::vector<double>>& solutions) const {
  // One task per degree, the workspaces resolved first.
  auto inputs = std::vector<std::pair<int, const std::vector<double>*>>{};
  auto degs = std::vector<const Degree*>{};
  for (const auto& [l, x] : solutions) {
    inputs.emplace_back(l, &x);
    degs.push_back(&Deg(l));
  }
  auto strains = std::vector<std::vector<double>>(inputs.size());
  const auto count = static_cast<int>(inputs.size());
  detail::ForEachTask(options_.parallel, count, [&](int i) {
    const auto j = static_cast<std::size_t>(count - 1 - i);
    const auto& x = *inputs[j].second;
    strains[j] = degs[j]->Strain(x, static_cast<int>(x.size() / degs[j]->ndof));
  });
  auto out = std::map<int, std::vector<double>>{};
  for (auto j = std::size_t{0}; j < inputs.size(); ++j) {
    out.emplace(inputs[j].first, std::move(strains[j]));
  }
  return out;
}

std::map<int, std::vector<double>> ViscoelasticOperator::Surface(
    const std::map<int, std::vector<double>>& solutions,
    Component component) const {
  auto out = std::map<int, std::vector<double>>{};
  for (const auto& [l, x] : solutions) {
    const auto& deg = Deg(l);
    const auto ncols = static_cast<int>(x.size() / deg.ndof);
    const auto dof = deg.system.SurfaceDof(component);
    auto values = std::vector<double>(ncols, 0.0);
    if (dof >= 0) {
      for (auto c = 0; c < ncols; ++c) {
        values[c] = x[static_cast<std::size_t>(c) * deg.ndof + dof];
      }
    }
    out.emplace(l, std::move(values));
  }
  return out;
}

std::map<int, std::vector<double>> ViscoelasticOperator::StartStrain(
    const ViscoelasticState& state, const LoadCallable& load,
    const LoadCallable& tide) const {
  if (state.HasStrain()) return state.strain;
  return Strain(SolveElastic(state, load, tide));
}

const std::map<int, std::vector<double>>& ViscoelasticOperator::PrimeStrain(
    ViscoelasticState& state, const LoadCallable& load,
    const LoadCallable& tide) const {
  if (!state.HasStrain()) {
    state.strain = Strain(SolveElastic(state, load, tide));
  }
  return state.strain;
}

std::pair<std::vector<double>, std::vector<double>>
ViscoelasticOperator::MaterialGradient(int l, std::span<const double> y,
                                       std::span<const double> x,
                                       int ncols) const {
  if (options_.memory_points != MemoryPointSet::Gll) {
    throw std::invalid_argument(
        "the material gradient needs the memory points at the GLL nodes");
  }
  return Deg(l).system.MaterialGradient(y, x, ncols);
}

std::vector<double> ViscoelasticOperator::InertiaMoment(
    std::span<const double> solution0, int ncols) const {
  const auto& deg = Deg(0);
  auto out = std::vector<double>(ncols, 0.0);
  for (auto c = 0; c < ncols; ++c) {
    auto sum = 0.0;
    for (auto d = 0; d < deg.ndof; ++d) {
      sum += deg.momentRow[d] *
             solution0[static_cast<std::size_t>(c) * deg.ndof + d];
    }
    out[c] = sum;
  }
  return out;
}

// -- the steps
// -----------------------------------------------------------------------

ViscoelasticStep::ViscoelasticStep(const ViscoelasticOperator& op,
                                   const ViscoelasticState& state, double dt,
                                   Scheme scheme, const LoadCallable& load,
                                   const LoadCallable& tide)
    : op_(&op),
      flavour_(op.FlavourOf(scheme, dt)),
      scheme_(scheme),
      dt_(dt),
      t_(state.t + dt) {
  const auto& weights = op.SchemeWeightsOf(scheme, dt);
  auto dStart = std::map<int, std::vector<double>>{};
  if (scheme == Scheme::ExponentialTrapezoid) {
    dStart = op.StartStrain(state, load, tide);
  }
  const auto K = op.GetRheology().BranchCount();
  // One task per degree, the workspaces resolved first: the decayed
  // memory, pointwise.
  auto inputs = std::vector<std::pair<int, const std::vector<double>*>>{};
  auto degs = std::vector<const ViscoelasticOperator::Degree*>{};
  for (const auto& [l, m] : state.m) {
    const auto& deg = op.Deg(l);
    ncols_[l] = ColumnsOf(state, l, m.size(), K * 3 * deg.ne * deg.np);
    inputs.emplace_back(l, &m);
    degs.push_back(&deg);
  }
  auto decayed = std::vector<std::vector<double>>(inputs.size());
  const auto count = static_cast<int>(inputs.size());
  detail::ForEachTask(op.options_.parallel, count, [&](int i) {
    const auto j = static_cast<std::size_t>(count - 1 - i);
    const auto l = inputs[j].first;
    const auto& m = *inputs[j].second;
    const auto& deg = *degs[j];
    const auto ncols = ncols_.at(l);
    const auto block = deg.NodalSize(ncols);
    auto mem = std::vector<double>(m.size());
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& w = weights[k];
      for (auto s = 0; s < 3; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto node =
                (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const auto at = (k * 3 + s) * block + node;
            for (auto c = 0; c < ncols; ++c) {
              if (scheme == Scheme::BackwardEuler) {
                mem[at + c] = w.e[full] * m[at + c];
              } else {
                mem[at + c] =
                    w.e[full] * m[at + c] +
                    w.alpha[full] * dStart.at(l)[s * block + node + c];
              }
            }
          }
        }
      }
    }
    decayed[j] = std::move(mem);
  });
  for (auto j = std::size_t{0}; j < inputs.size(); ++j) {
    mem_.emplace(inputs[j].first, std::move(decayed[j]));
  }
}

const std::map<int, std::vector<double>>& ViscoelasticStep::MemorySolutions()
    const {
  if (memorySolutions_.empty()) {
    auto& flavour = *flavour_;
    // One task per degree, the workspaces and effective systems resolved
    // first: the memory force through the effective system.
    auto inputs = std::vector<std::pair<int, const std::vector<double>*>>{};
    auto degs = std::vector<const ViscoelasticOperator::Degree*>{};
    auto systems = std::vector<const DegreeOperator*>{};
    for (const auto& [l, mem] : mem_) {
      inputs.emplace_back(l, &mem);
      degs.push_back(&op_->Deg(l));
      systems.push_back(&op_->EffectiveSystem(flavour, l));
    }
    auto solved = std::vector<std::vector<double>>(inputs.size());
    const auto count = static_cast<int>(inputs.size());
    detail::ForEachTask(op_->options_.parallel, count, [&](int i) {
      const auto j = static_cast<std::size_t>(count - 1 - i);
      const auto l = inputs[j].first;
      const auto ncols = ncols_.at(l);
      auto force = degs[j]->MemoryForce(*inputs[j].second, ncols);
      if (const auto extra = extraForce_.find(l); extra != extraForce_.end()) {
        for (auto k = std::size_t{0}; k < force.size(); ++k) {
          force[k] += extra->second[k];
        }
      }
      solved[j] = systems[j]->Solve(force, ncols);
    });
    for (auto j = std::size_t{0}; j < inputs.size(); ++j) {
      memorySolutions_.emplace(inputs[j].first, std::move(solved[j]));
    }
  }
  return memorySolutions_;
}

std::map<int, std::vector<double>> ViscoelasticStep::MemorySurface(
    Component component) const {
  return op_->Surface(MemorySolutions(), component);
}

std::vector<double> ViscoelasticStep::InertiaMoment() const {
  const auto& solutions = MemorySolutions();
  auto found = solutions.find(0);
  if (found == solutions.end()) {
    throw std::invalid_argument("the state does not carry degree 0");
  }
  return op_->InertiaMoment(found->second, ncols_.at(0));
}

const LoveNumbers& ViscoelasticStep::EffectiveLove() const {
  return op_->EffectiveLoveNumbers(scheme_, dt_);
}

void ViscoelasticStep::AddForce(
    const std::map<int, std::vector<double>>& force) {
  for (const auto& [l, f] : force) {
    if (!mem_.contains(l)) continue;
    const auto expected =
        static_cast<std::size_t>(op_->Deg(l).ndof) * ncols_.at(l);
    if (f.size() != expected) {
      throw std::invalid_argument(std::format(
          "a degree-{} force has {} entries, the step's solutions {}", l,
          f.size(), expected));
    }
    auto& extra = extraForce_[l];
    if (extra.empty()) {
      extra = f;
    } else {
      for (auto i = std::size_t{0}; i < extra.size(); ++i) extra[i] += f[i];
    }
  }
  memorySolutions_.clear();
}

void ViscoelasticStep::AddMemory(
    const std::map<int, std::vector<double>>& increment) {
  for (auto& [l, mem] : mem_) {
    const auto found = increment.find(l);
    if (found == increment.end()) continue;
    const auto& add = found->second;
    if (add.size() != mem.size()) {
      throw std::invalid_argument(std::format(
          "a degree-{} memory increment has {} entries, the step's memory {}",
          l, add.size(), mem.size()));
    }
    for (auto i = std::size_t{0}; i < mem.size(); ++i) mem[i] += add[i];
  }
  memorySolutions_.clear();
}

std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
ViscoelasticStep::Finish(const LoadCallable& load,
                         const LoadCallable& tide) const {
  const auto& op = *op_;
  const auto& weights = op.SchemeWeightsOf(scheme_, dt_);
  const auto& memory = MemorySolutions();
  const auto K = op.GetRheology().BranchCount();

  // Degree tasks: the workspaces, unit responses and channel columns are
  // resolved here, serially (the caches fill lazily, and the callables
  // need not be thread-safe); each task writes only its own results.
  struct Task {
    int l = 0;
    int ncols = 0;
    const std::vector<double>* mem = nullptr;
    const ViscoelasticOperator::Degree* deg = nullptr;
    const std::vector<double>* units = nullptr;
    std::optional<std::vector<double>> load, tide;
    std::vector<double> x, d1, mNew;
  };
  RefuseTideWithoutGravity(op, tide);
  auto tasks = std::vector<Task>{};
  for (const auto& [l, mem] : mem_) {
    Task task;
    task.l = l;
    task.mem = &mem;
    task.deg = &op.Deg(l);
    task.ncols = ncols_.at(l);
    if (load || tide) {
      task.units = &op.UnitResponses(flavour_.get(), l);
      task.load = LoadColumns(op, load, l, t_, task.ncols);
      task.tide = TideColumns(tide, l, t_, task.ncols);
    }
    tasks.push_back(std::move(task));
  }
  const auto count = static_cast<int>(tasks.size());
  detail::ForEachTask(op.options_.parallel, count, [&](int i) {
    auto& task = tasks[count - 1 - i];  // the largest degrees first
    const auto& deg = *task.deg;
    const auto ncols = task.ncols;
    auto x = memory.at(task.l);
    if (task.load) AddColumns(x, *task.units, 0, deg.ndof, *task.load, ncols);
    if (task.tide) AddColumns(x, *task.units, 1, deg.ndof, *task.tide, ncols);
    auto d1 = deg.Strain(x, ncols);
    const auto block = deg.NodalSize(ncols);
    std::vector<double> mNew = *task.mem;
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& w = weights[k];
      for (auto s = 0; s < 3; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto factor = scheme_ == Scheme::BackwardEuler
                                    ? 1.0 - w.e[full]
                                    : w.beta[full];
            const auto node =
                (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const auto at = (k * 3 + s) * block + node;
            for (auto c = 0; c < ncols; ++c) {
              mNew[at + c] += factor * d1[s * block + node + c];
            }
          }
        }
      }
    }
    task.x = std::move(x);
    task.d1 = std::move(d1);
    task.mNew = std::move(mNew);
  });

  ViscoelasticState out;
  out.t = t_;
  out.columns = ncols_;
  auto solutions = std::map<int, std::vector<double>>{};
  for (auto& task : tasks) {
    out.m.emplace(task.l, std::move(task.mNew));
    out.strain.emplace(task.l, std::move(task.d1));
    solutions.emplace(task.l, std::move(task.x));
  }
  return {std::move(out), std::move(solutions)};
}

ViscoelasticStep ViscoelasticOperator::StartStep(
    const ViscoelasticState& state, double dt, Scheme scheme,
    const LoadCallable& load, const LoadCallable& tide) const {
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "a two-phase step is implicit: ExponentialTrapezoid or "
        "BackwardEuler");
  }
  if (IsMultiStage(scheme) && dt != 0.0) {
    throw std::invalid_argument(
        "an opened step is one stage, ExponentialTrapezoid or BackwardEuler: "
        "Sdirk2 is opened stage by stage, each a BackwardEuler step "
        "(StepRecipe.hpp)");
  }
  // a zero-length step is the elastic solve with the memory held
  if (dt < 0.0) throw std::invalid_argument("dt must be non-negative");
  return ViscoelasticStep(*this, state, dt, ZeroLengthScheme(scheme), load,
                          tide);
}

std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
ViscoelasticOperator::Step(const ViscoelasticState& state, double dt,
                           Scheme scheme, const LoadCallable& load,
                           const LoadCallable& tide) const {
  if (dt < 0.0) throw std::invalid_argument("dt must be non-negative");
  if (scheme == Scheme::ExponentialEuler) {
    auto dStart = StartStrain(state, load, tide);
    const auto& weights = SchemeWeightsOf(Scheme::ExponentialTrapezoid, dt);
    const auto K = rheology_->BranchCount();
    ViscoelasticState out;
    out.t = state.t + dt;
    out.columns = state.columns;
    for (const auto& [l, m] : state.m) {
      const auto& deg = Deg(l);
      const int ncols = ColumnsOf(state, l, m.size(), K * 3 * deg.ne * deg.np);
      const std::size_t block = deg.NodalSize(ncols);
      auto mNew = std::vector<double>(m.size());
      for (auto k = std::size_t{0}; k < K; ++k) {
        const auto& w = weights[k];
        for (auto s = 0; s < 3; ++s) {
          for (auto e = 0; e < deg.ne; ++e) {
            for (auto q = 0; q < deg.np; ++q) {
              const std::size_t full =
                  static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
              const std::size_t node =
                  (static_cast<std::size_t>(e) * deg.np + q) * ncols;
              const std::size_t at = (k * 3 + s) * block + node;
              for (auto c = 0; c < ncols; ++c) {
                mNew[at + c] =
                    w.e[full] * m[at + c] +
                    (1.0 - w.e[full]) * dStart[l][s * block + node + c];
              }
            }
          }
        }
      }
      out.m.emplace(l, std::move(mNew));
    }
    // the displacement is not consistent with the new state
    return {std::move(out), SolveElastic(state, load, tide)};
  }
  if (IsMultiStage(scheme)) {
    // stage by stage, each closed under the channels at its own end time
    auto solutions = std::map<int, std::vector<double>>{};
    auto out = StepByRecipe(
        RecipeOf(scheme, dt), dt, state,
        [&](const ViscoelasticState& from, const RecipeStage& stage,
            double length, std::size_t) {
          auto [next, x] =
              ViscoelasticStep(*this, from, length, stage.family, load, tide)
                  .Finish(load, tide);
          solutions = std::move(x);
          return std::move(next);
        },
        [](std::span<const std::pair<double, const ViscoelasticState*>> terms,
           const RecipeStage&) { return CombinedState(terms); });
    return {std::move(out), std::move(solutions)};
  }
  return ViscoelasticStep(*this, state, dt, scheme, load, tide)
      .Finish(load, tide);
}

ViscoelasticOperator::AdaptiveResult ViscoelasticOperator::AdaptiveStep(
    ViscoelasticState& state, double dt, double rtol, const LoadCallable& load,
    const LoadCallable& tide) const {
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  const auto& dStart = PrimeStrain(state, load, tide);
  const auto& weights = SchemeWeightsOf(Scheme::ExponentialTrapezoid, dt);
  auto [newState, solutions] =
      Step(state, dt, Scheme::ExponentialTrapezoid, load, tide);

  const auto K = rheology_->BranchCount();
  auto top = 0.0;
  for (const auto& [l, m] : newState.m) {
    for (double v : m) top = std::max(top, std::abs(v));
  }
  const auto atol = top > 0.0 ? 1e-6 * rtol * top : 1e-300;

  auto total = 0.0;
  auto count = std::size_t{0};
  for (const auto& [l, m] : state.m) {
    const auto& deg = Deg(l);
    const int ncols = ColumnsOf(state, l, m.size(), K * 3 * deg.ne * deg.np);
    const std::size_t block = deg.NodalSize(ncols);
    const auto& mNew = newState.m.at(l);
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& w = weights[k];
      for (auto s = 0; s < 3; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const std::size_t full =
                static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const std::size_t node =
                (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const std::size_t at = (k * 3 + s) * block + node;
            for (auto c = 0; c < ncols; ++c) {
              const double companion =
                  w.e[full] * m[at + c] +
                  (1.0 - w.e[full]) * dStart.at(l)[s * block + node + c];
              const double scale =
                  atol + rtol * std::max(std::abs(m[at + c]),
                                         std::abs(mNew.at(at + c)));
              const double ratio = (mNew[at + c] - companion) / scale;
              total += ratio * ratio;
              ++count;
            }
          }
        }
      }
    }
  }
  const auto err = count ? std::sqrt(total / count) : 0.0;
  constexpr auto kSafety = 0.9, kMin = 0.2, kMax = 4.0;
  const double factor =
      err == 0.0 ? kMax : std::clamp(kSafety / std::sqrt(err), kMin, kMax);
  AdaptiveResult result;
  result.error = err;
  result.dtNext = dt * factor;
  if (err <= 1.0) {
    result.state = std::move(newState);
    result.solutions = std::move(solutions);
  }
  return result;
}

const LoveNumbers& ViscoelasticOperator::EffectiveLoveNumbers(Scheme scheme,
                                                              double dt) const {
  RequireGravity(*this, "the Love numbers of the sea level equation");
  // a zero-length step's solid is the unrelaxed one
  if (dt == 0.0) return ElasticLoveNumbers();
  RefuseMultiStage(scheme, dt, "one effective solid");
  const auto key = std::pair(static_cast<int>(scheme), dt);
  auto found = effectiveLove_.find(key);
  if (found == effectiveLove_.end()) {
    if (options_.memory_points == MemoryPointSet::Gll) {
      const auto material = EffectiveMaterial(scheme, dt);
      found = effectiveLove_
                  .emplace(key, ComputeLoveNumbers(
                                    material, lmax_,
                                    {.eps = options_.eps,
                                     .truncation = options_.truncation,
                                     .parallel = options_.parallel}))
                  .first;
    } else {
      const auto flavour = FlavourOf(scheme, dt);
      found = effectiveLove_
                  .emplace(key, ComputeLoveNumbers(
                                    NodalMaterial(), lmax_,
                                    {.eps = options_.eps,
                                     .truncation = options_.truncation,
                                     .point_shear = &flavour->pointShear,
                                     .parallel = options_.parallel}))
                  .first;
    }
  }
  return found->second;
}

const LoveNumbers& ViscoelasticOperator::ElasticLoveNumbers() const {
  RequireGravity(*this, "the Love numbers of the sea level equation");
  const auto key = std::pair(-1, 0.0);
  auto found = effectiveLove_.find(key);
  if (found == effectiveLove_.end()) {
    found = effectiveLove_
                .emplace(key, ComputeLoveNumbers(
                                  NodalMaterial(), lmax_,
                                  {.eps = options_.eps,
                                   .truncation = options_.truncation,
                                   .point_shear = ElasticPointShear(),
                                   .parallel = options_.parallel}))
                .first;
  }
  return found->second;
}

}  // namespace sl3dpp
