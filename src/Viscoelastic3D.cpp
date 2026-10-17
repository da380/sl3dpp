#include "sl3dpp/Viscoelastic3D.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

#include "sl3dpp/detail/AndersonMixer.hpp"

namespace sl3dpp {

namespace {

// A spectral tensor's coefficients as real segments for the mixer: the
// spin -2, spin -1 and spin 0 blocks in turn, each complex value as its
// real-imaginary pair (the array layout std::complex guarantees).
std::array<std::span<double>, 3> RealSegments(GridCoupling::SpectralTensor& t) {
  auto pairs = [](std::vector<std::complex<double>>& v) {
    return std::span<double>(reinterpret_cast<double*>(v.data()), 2 * v.size());
  };
  return {pairs(t.c[0]), pairs(t.c[1]), pairs(t.r)};
}

std::array<std::span<const double>, 3> RealSegments(
    const GridCoupling::SpectralTensor& t) {
  auto pairs = [](const std::vector<std::complex<double>>& v) {
    return std::span<const double>(reinterpret_cast<const double*>(v.data()),
                                   2 * v.size());
  };
  return {pairs(t.c[0]), pairs(t.c[1]), pairs(t.r)};
}

// The three coefficient blocks of a spectral tensor, for loops that treat
// them alike.
std::array<std::vector<std::complex<double>>*, 3> Blocks(
    GridCoupling::SpectralTensor& t) {
  return {&t.c[0], &t.c[1], &t.r};
}

std::array<const std::vector<std::complex<double>>*, 3> Blocks(
    const GridCoupling::SpectralTensor& t) {
  return {&t.c[0], &t.c[1], &t.r};
}

}  // namespace

struct Viscoelastic3D::Flavour {
  std::unique_ptr<Material> material;  // the reference effective solid
  std::map<int, std::unique_ptr<DegreeOperator>> spheroidal;
  std::map<int, std::unique_ptr<ToroidalOperator>> toroidal;
  std::map<int, std::vector<double>> units;  // (ndof x 2): load, tide
};

Viscoelastic3D::Viscoelastic3D(
    const Rheology& rheology, int lmax,
    const std::function<double(double, double, double)>& delta, Options options)
    : Viscoelastic3D(
          rheology, lmax,
          SurfaceGrid(lmax, {.nmax = 2,
                             .oversampling = options.oversampling,
                             .parallel = options.parallel,
                             .matrix_kernel = options.matrix_kernel}),
          delta, options) {}

Viscoelastic3D::Viscoelastic3D(
    const Rheology& rheology, int lmax, const SurfaceGrid& grid,
    const std::function<double(double, double, double)>& delta, Options options)
    : rheology_(std::make_shared<const Rheology>(rheology)),
      lmax_(lmax),
      options_(options),
      coupling_(*rheology_, lmax, grid,
                {.eps = options.eps,
                 .truncation = options.truncation,
                 .oversampling = options.oversampling,
                 .parallel = options.parallel,
                 .matrix_kernel = options.matrix_kernel}),
      reference_(*rheology_, lmax,
                 {.eps = options.eps, .truncation = options.truncation}) {
  // Sample exp(-delta) once and scale every branch's inverse times; the
  // reference profile is the unperturbed radial rheology itself.
  const auto& sampling = coupling_.Grid();
  const auto& mesh = rheology_->BaseMaterial().Mesh();
  const auto fieldSize = sampling.FieldSize();
  const auto n = mesh.NGll();
  const auto nodes = static_cast<std::size_t>(coupling_.NodeCount());
  std::vector<double> factor(nodes * fieldSize);
  const auto thetas = sampling.CoLatitudes();
  const auto phis = sampling.Longitudes();
  for (std::size_t node = 0; node < nodes; ++node) {
    const auto r = coupling_.RadiusOfNode(static_cast<int>(node));
    std::size_t x = 0;
    for (auto theta : thetas) {
      for (auto phi : phis) {
        factor[node * fieldSize + x++] = std::exp(-delta(r, theta, phi));
      }
    }
  }
  for (std::size_t k = 0; k < rheology_->BranchCount(); ++k) {
    const auto& branch = rheology_->BranchAt(k);
    std::vector<double> itau(nodes * fieldSize);
    std::vector<double> mu(nodes);
    for (std::size_t node = 0; node < nodes; ++node) {
      const auto e = coupling_.ElementOfNode(static_cast<int>(node));
      const auto q = static_cast<int>(node) % n;
      const auto at = static_cast<std::size_t>(e) * n + q;
      mu[node] = branch.mu[at];
      for (std::size_t x = 0; x < fieldSize; ++x) {
        itau[node * fieldSize + x] =
            branch.itau[at] * factor[node * fieldSize + x];
      }
    }
    itau_.push_back(std::move(itau));
    muBranch_.push_back(std::move(mu));
  }
}

Viscoelastic3D::~Viscoelastic3D() = default;
Viscoelastic3D::Viscoelastic3D(Viscoelastic3D&&) noexcept = default;
Viscoelastic3D& Viscoelastic3D::operator=(Viscoelastic3D&&) noexcept = default;

Viscoelastic3D::State Viscoelastic3D::ZeroState() const {
  State state;
  for (std::size_t k = 0; k < rheology_->BranchCount(); ++k) {
    state.m.push_back(coupling_.ZeroTensor());
  }
  return state;
}

const DegreeOperator& Viscoelastic3D::Spheroidal(int l) const {
  auto found = spheroidal_.find(l);
  if (found == spheroidal_.end()) {
    found = spheroidal_
                .emplace(l, std::make_unique<DegreeOperator>(
                                rheology_->BaseMaterial(), l,
                                DegreeOperator::Options{
                                    .eps = options_.eps,
                                    .truncation = options_.truncation}))
                .first;
  }
  return *found->second;
}

const ToroidalOperator& Viscoelastic3D::Toroidal(int l) const {
  auto found = toroidal_.find(l);
  if (found == toroidal_.end()) {
    found = toroidal_
                .emplace(l, std::make_unique<ToroidalOperator>(
                                rheology_->BaseMaterial(), l,
                                ToroidalOperator::Options{
                                    .eps = options_.eps,
                                    .truncation = options_.truncation}))
                .first;
  }
  return *found->second;
}

Viscoelastic3D::Flavour& Viscoelastic3D::FlavourOf(Scheme scheme,
                                                   double dt) const {
  const auto key = std::pair(static_cast<int>(scheme), dt);
  auto found = flavours_.find(key);
  if (found == flavours_.end()) {
    auto flavour = std::make_unique<Flavour>();
    flavour->material =
        std::make_unique<Material>(reference_.EffectiveMaterial(scheme, dt));
    found = flavours_.emplace(key, std::move(flavour)).first;
  }
  return *found->second;
}

GridCoupling::SpectralTensor Viscoelastic3D::StrainCoefficients(
    const Solutions& solutions,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal) const {
  auto spectral = coupling_.ZeroSpectral();
  for (const auto& [l, x] : solutions.spheroidal) {
    const auto torIt = solutions.toroidal.find(l);
    coupling_.AddStrainOfSolution(
        spectral, l, spheroidal(l), x,
        torIt == solutions.toroidal.end() ? nullptr : &toroidal(l),
        torIt == solutions.toroidal.end()
            ? std::span<const Complex>{}
            : std::span<const Complex>(torIt->second));
  }
  return spectral;
}

GridCoupling::TensorField Viscoelastic3D::StrainField(
    const Solutions& solutions,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal) const {
  return coupling_.Synthesise(
      StrainCoefficients(solutions, spheroidal, toroidal));
}

Viscoelastic3D::Solutions Viscoelastic3D::SolveAll(
    const GridCoupling::SpectralTensor& weighted, double t,
    const LoadCallable3D& load, const LoadCallable3D& tide,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal) const {
  Solutions out;
  auto sphForces = coupling_.SpheroidalForce(weighted, spheroidal);
  auto torForces = coupling_.ToroidalForce(weighted, toroidal);
  for (auto& [l, b] : sphForces) {
    const auto& op = spheroidal(l);
    const auto columns = l + 1;
    const auto ndof = op.DofCount();
    // The load channels enter through the cached unit responses of the
    // same operator: solve the complex columns as real pairs.
    std::vector<double> rhs(static_cast<std::size_t>(ndof) * 2 * columns);
    for (auto c = 0; c < columns; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        rhs[static_cast<std::size_t>(c) * ndof + d] =
            b[static_cast<std::size_t>(c) * ndof + d].real();
        rhs[static_cast<std::size_t>(columns + c) * ndof + d] =
            b[static_cast<std::size_t>(c) * ndof + d].imag();
      }
    }
    auto x = op.Solve(rhs, 2 * columns);
    std::vector<Complex> solution(static_cast<std::size_t>(ndof) * columns);
    for (auto c = 0; c < columns; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        solution[static_cast<std::size_t>(c) * ndof + d] = {
            x[static_cast<std::size_t>(c) * ndof + d],
            x[static_cast<std::size_t>(columns + c) * ndof + d]};
      }
    }
    // Channels: linear in the forcing, through unit load/tide solutions.
    if (load || tide) {
      std::vector<double> unit(static_cast<std::size_t>(ndof) * 2, 0.0);
      auto loadVec = op.Load();
      auto tideVec = op.Tide(l == 0 ? 2 : -1);
      std::ranges::copy(loadVec, unit.begin());
      std::ranges::copy(tideVec, unit.begin() + ndof);
      const auto responses = op.Solve(unit, 2);
      auto add = [&](const std::optional<std::vector<Complex>>& columnsIn,
                     int which) {
        if (!columnsIn) return;
        for (auto c = 0; c < columns; ++c) {
          const auto amplitude = (*columnsIn)[c];
          if (amplitude == 0.0) continue;
          for (auto d = 0; d < ndof; ++d) {
            solution[static_cast<std::size_t>(c) * ndof + d] +=
                amplitude *
                responses[static_cast<std::size_t>(which) * ndof + d];
          }
        }
      };
      if (load) add(load(l, t), 0);
      if (tide) add(tide(l, t), 1);
    }
    out.spheroidal.emplace(l, std::move(solution));
  }
  for (auto& [l, b] : torForces) {
    const auto& op = toroidal(l);
    const auto columns = l + 1;
    const auto ndof = op.DofCount();
    std::vector<double> rhs(static_cast<std::size_t>(ndof) * 2 * columns);
    for (auto c = 0; c < columns; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        rhs[static_cast<std::size_t>(c) * ndof + d] =
            b[static_cast<std::size_t>(c) * ndof + d].real();
        rhs[static_cast<std::size_t>(columns + c) * ndof + d] =
            b[static_cast<std::size_t>(c) * ndof + d].imag();
      }
    }
    auto x = op.Solve(rhs, 2 * columns);
    std::vector<Complex> solution(static_cast<std::size_t>(ndof) * columns);
    for (auto c = 0; c < columns; ++c) {
      for (auto d = 0; d < ndof; ++d) {
        solution[static_cast<std::size_t>(c) * ndof + d] = {
            x[static_cast<std::size_t>(c) * ndof + d],
            x[static_cast<std::size_t>(columns + c) * ndof + d]};
      }
    }
    out.toroidal.emplace(l, std::move(solution));
  }
  return out;
}

Viscoelastic3D::Solutions Viscoelastic3D::SolveElastic(
    const State& state, const LoadCallable3D& load,
    const LoadCallable3D& tide) const {
  // The combined weighted memory tensor sum_k mu_k(r) m_k(x).
  auto combined = coupling_.ZeroTensor();
  const auto fieldSize = coupling_.Grid().FieldSize();
  auto accumulate = [&](auto& out, const auto& in, std::size_t k) {
    for (std::size_t node = 0;
         node < static_cast<std::size_t>(coupling_.NodeCount()); ++node) {
      const auto mu = muBranch_[k][node];
      if (mu == 0.0) continue;
      for (std::size_t x = 0; x < fieldSize; ++x) {
        out[node * fieldSize + x] += mu * in[node * fieldSize + x];
      }
    }
  };
  for (std::size_t k = 0; k < state.m.size(); ++k) {
    for (auto c = 0; c < 2; ++c) {
      accumulate(combined.c[c], state.m[k].c[c], k);
    }
    accumulate(combined.r, state.m[k].r, k);
  }
  const auto weighted = coupling_.Analyse(combined);
  return SolveAll(
      weighted, state.t, load, tide,
      [&](int l) -> const DegreeOperator& { return Spheroidal(l); },
      [&](int l) -> const ToroidalOperator& { return Toroidal(l); });
}

const DegreeOperator& Viscoelastic3D::FlavourSpheroidal(Flavour& flavour,
                                                        int l) const {
  auto found = flavour.spheroidal.find(l);
  if (found == flavour.spheroidal.end()) {
    found = flavour.spheroidal
                .emplace(l, std::make_unique<DegreeOperator>(
                                *flavour.material, l,
                                DegreeOperator::Options{
                                    .eps = options_.eps,
                                    .truncation = options_.truncation}))
                .first;
  }
  return *found->second;
}

const ToroidalOperator& Viscoelastic3D::FlavourToroidal(Flavour& flavour,
                                                        int l) const {
  auto found = flavour.toroidal.find(l);
  if (found == flavour.toroidal.end()) {
    found = flavour.toroidal
                .emplace(l, std::make_unique<ToroidalOperator>(
                                *flavour.material, l,
                                ToroidalOperator::Options{
                                    .eps = options_.eps,
                                    .truncation = options_.truncation}))
                .first;
  }
  return *found->second;
}

void Viscoelastic3D::PrimeStrain(State& state, const LoadCallable3D& load,
                                 const LoadCallable3D& tide) const {
  if (state.strain.has_value()) return;
  state.strain = StrainField(
      SolveElastic(state, load, tide),
      [&](int l) -> const DegreeOperator& { return Spheroidal(l); },
      [&](int l) -> const ToroidalOperator& { return Toroidal(l); });
}

Viscoelastic3D::OpenStep Viscoelastic3D::StartStep(
    const State& state, double dt, Scheme scheme, const LoadCallable3D& load,
    const LoadCallable3D& tide) const {
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "the 3D step is implicit: ExponentialTrapezoid or BackwardEuler");
  }
  const auto fieldSize = coupling_.Grid().FieldSize();
  const auto nodes = static_cast<std::size_t>(coupling_.NodeCount());
  const auto K = state.m.size();

  OpenStep step;
  step.op_ = this;
  step.scheme_ = scheme;
  step.dt_ = dt;
  step.t_ = state.t + dt;
  // The start strain, on the grid for the decayed memory and as the
  // coefficients the iteration starts from: from the state's cache, or
  // from a consistent solve under the start-time channels.
  GridCoupling::TensorField solved;
  if (state.strain.has_value()) {
    step.s0_ = coupling_.Analyse(*state.strain);
  } else {
    step.s0_ = StrainCoefficients(
        SolveElastic(state, load, tide),
        [&](int l) -> const DegreeOperator& { return Spheroidal(l); },
        [&](int l) -> const ToroidalOperator& { return Toroidal(l); });
    solved = coupling_.Synthesise(step.s0_);
  }
  const auto& d0 = state.strain.has_value() ? *state.strain : solved;

  // Per branch and grid point the step weights, the decayed memory m~_k
  // and its beta_k for Finish; summed over branches, the lateral weight
  // and the fixed memory force sum_k mu_k m~_k, analysed once here.
  const auto& branches = *rheology_;
  const auto& mesh = branches.BaseMaterial().Mesh();
  const auto n = mesh.NGll();
  step.beta_.resize(K);
  step.weight_.assign(nodes * fieldSize, 0.0);
  auto fixed = coupling_.ZeroTensor();
  for (std::size_t k = 0; k < K; ++k) {
    step.memory_.push_back(coupling_.ZeroTensor());
    step.beta_[k].assign(nodes * fieldSize, 0.0);
    const auto& branch = branches.BranchAt(k);
    for (std::size_t node = 0; node < nodes; ++node) {
      const auto e = coupling_.ElementOfNode(static_cast<int>(node));
      const auto q = static_cast<int>(node) % n;
      const auto itauRef = branch.itau[static_cast<std::size_t>(e) * n + q];
      const auto refBeta = scheme == Scheme::BackwardEuler
                               ? 1.0 - BackwardEulerWeight(dt * itauRef)
                               : ExponentialTrapezoidWeights(dt * itauRef).beta;
      const auto mu = muBranch_[k][node];
      for (std::size_t x = 0; x < fieldSize; ++x) {
        const auto h = dt * itau_[k][node * fieldSize + x];
        double e_w, a_w, b_w;
        if (scheme == Scheme::BackwardEuler) {
          const auto w = BackwardEulerWeight(h);
          e_w = w;
          a_w = 0.0;
          b_w = 1.0 - w;
        } else {
          const auto w = ExponentialTrapezoidWeights(h);
          e_w = w.e;
          a_w = w.alpha;
          b_w = w.beta;
        }
        const auto at = node * fieldSize + x;
        step.beta_[k][at] = b_w;
        auto& memory = step.memory_[k];
        for (auto c = 0; c < 2; ++c) {
          memory.c[c][at] = e_w * state.m[k].c[c][at] + a_w * d0.c[c][at];
        }
        memory.r[at] = e_w * state.m[k].r[at] + a_w * d0.r[at];
        if (mu == 0.0) continue;
        step.weight_[at] += mu * (b_w - refBeta);
        for (auto c = 0; c < 2; ++c) fixed.c[c][at] += mu * memory.c[c][at];
        fixed.r[at] += mu * memory.r[at];
      }
    }
  }
  step.fixed_ = coupling_.Analyse(fixed);
  return step;
}

Viscoelastic3D::Solutions Viscoelastic3D::OpenStep::Respond(
    const GridCoupling::SpectralTensor& strain) const {
  const auto& op = *op_;
  const auto& coupling = op.coupling_;
  auto& flavour = op.FlavourOf(scheme_, dt_);
  // The lateral force of the strain (synthesised, weighted pointwise,
  // analysed) plus the step's fixed memory force.
  auto d = coupling.Synthesise(strain);
  coupling.Multiply(d, weight_);
  auto weighted = coupling.Analyse(d);
  const auto fixed = Blocks(fixed_);
  const auto out = Blocks(weighted);
  for (std::size_t b = 0; b < out.size(); ++b) {
    for (std::size_t i = 0; i < out[b]->size(); ++i) {
      (*out[b])[i] += (*fixed[b])[i];
    }
  }
  return op.SolveAll(
      weighted, t_, {}, {},
      [&](int l) -> const DegreeOperator& {
        return op.FlavourSpheroidal(flavour, l);
      },
      [&](int l) -> const ToroidalOperator& {
        return op.FlavourToroidal(flavour, l);
      });
}

void Viscoelastic3D::OpenStep::AddChannels(Solutions& solutions,
                                           const LoadCallable3D& load,
                                           const LoadCallable3D& tide) const {
  if (!load && !tide) return;
  const auto& op = *op_;
  auto& flavour = op.FlavourOf(scheme_, dt_);
  for (auto& [l, columns_] : solutions.spheroidal) {
    // A plain reference: clang does not yet allow capturing a structured
    // binding in a lambda under OpenMP.
    auto& x = columns_;
    const auto& system = op.FlavourSpheroidal(flavour, l);
    const auto ndof = system.DofCount();
    auto found = flavour.units.find(l);
    if (found == flavour.units.end()) {
      std::vector<double> unit(static_cast<std::size_t>(ndof) * 2, 0.0);
      auto loadVec = system.Load();
      auto tideVec = system.Tide(l == 0 ? 2 : -1);
      std::ranges::copy(loadVec, unit.begin());
      std::ranges::copy(tideVec, unit.begin() + ndof);
      found = flavour.units.emplace(l, system.Solve(unit, 2)).first;
    }
    const auto& responses = found->second;
    const auto columns = l + 1;
    auto add = [&](const std::optional<std::vector<Complex>>& columnsIn,
                   int which) {
      if (!columnsIn) return;
      for (auto c = 0; c < columns; ++c) {
        const auto amplitude = (*columnsIn)[c];
        if (amplitude == 0.0) continue;
        for (auto dof = 0; dof < ndof; ++dof) {
          x[static_cast<std::size_t>(c) * ndof + dof] +=
              amplitude *
              responses[static_cast<std::size_t>(which) * ndof + dof];
        }
      }
    };
    if (load) add(load(l, t_), 0);
    if (tide) add(tide(l, t_), 1);
  }
}

GridCoupling::SpectralTensor Viscoelastic3D::OpenStep::StrainOf(
    const Solutions& solutions) const {
  const auto& op = *op_;
  auto& flavour = op.FlavourOf(scheme_, dt_);
  return op.StrainCoefficients(
      solutions,
      [&](int l) -> const DegreeOperator& {
        return op.FlavourSpheroidal(flavour, l);
      },
      [&](int l) -> const ToroidalOperator& {
        return op.FlavourToroidal(flavour, l);
      });
}

double Viscoelastic3D::OpenStep::InertiaMoment(
    const Solutions& solutions) const {
  const auto& x = solutions.spheroidal.at(0);
  std::vector<double> column(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) column[i] = x[i].real();
  return op_->reference_.InertiaMoment(column, 1)[0];
}

const LoveNumbers& Viscoelastic3D::OpenStep::EffectiveLove() const {
  return op_->reference_.EffectiveLoveNumbers(scheme_, dt_);
}

Viscoelastic3D::State Viscoelastic3D::OpenStep::Finish(
    const GridCoupling::SpectralTensor& strain) const {
  auto d = op_->coupling_.Synthesise(strain);
  State next;
  next.t = t_;
  for (std::size_t k = 0; k < memory_.size(); ++k) {
    auto tensor = memory_[k];
    for (auto c = 0; c < 2; ++c) {
      for (std::size_t i = 0; i < tensor.c[c].size(); ++i) {
        tensor.c[c][i] += beta_[k][i] * d.c[c][i];
      }
    }
    for (std::size_t i = 0; i < tensor.r.size(); ++i) {
      tensor.r[i] += beta_[k][i] * d.r[i];
    }
    next.m.push_back(std::move(tensor));
  }
  next.strain = std::move(d);
  return next;
}

std::pair<GridCoupling::SpectralTensor, Viscoelastic3D::StepInfo>
Viscoelastic3D::OpenStep::Converge(const Pass& pass,
                                   const IterationControl& control) const {
  auto s = s0_;
  StepInfo info;
  info.converged = false;
  detail::AndersonMixer mixer(control.anderson_memory);
  for (auto iteration = 0; iteration < control.max_iterations; ++iteration) {
    auto sNext = pass(s);
    const auto next = Blocks(sNext);
    const auto now = Blocks(std::as_const(s));

    // The relative sup-norm change of the coefficients, compared on
    // squared magnitudes.
    double scale = 0.0, diff = 0.0;
    for (std::size_t b = 0; b < next.size(); ++b) {
      for (std::size_t i = 0; i < next[b]->size(); ++i) {
        scale = std::max(scale, std::norm((*next[b])[i]));
        diff = std::max(diff, std::norm((*next[b])[i] - (*now[b])[i]));
      }
    }
    info.iterations = iteration + 1;
    info.residual = scale > 0.0 ? std::sqrt(diff) / std::sqrt(scale) : 0.0;
    if (info.residual <= control.rtol) {
      info.converged = true;
      s = std::move(sNext);
      break;
    }
    if (control.damping != 1.0) {
      for (std::size_t b = 0; b < next.size(); ++b) {
        for (std::size_t i = 0; i < next[b]->size(); ++i) {
          (*next[b])[i] =
              (*now[b])[i] + control.damping * ((*next[b])[i] - (*now[b])[i]);
        }
      }
    }
    if (control.anderson_memory > 0) {
      mixer.UpdateInPlace(RealSegments(s), RealSegments(std::as_const(sNext)));
    } else {
      s = std::move(sNext);
    }
  }
  return {std::move(s), info};
}

Viscoelastic3D::StepResult Viscoelastic3D::Step(
    const State& state, double dt, Scheme scheme, const LoadCallable3D& load,
    const LoadCallable3D& tide) const {
  auto step = StartStep(state, dt, scheme, load, tide);
  Solutions solutions;
  auto [strain, info] = step.Converge(
      [&](const GridCoupling::SpectralTensor& s) {
        solutions = step.Respond(s);
        step.AddChannels(solutions, load, tide);
        return step.StrainOf(solutions);
      },
      {.max_iterations = options_.max_iterations,
       .rtol = options_.rtol,
       .damping = options_.damping,
       .anderson_memory = options_.anderson_memory});
  return {step.Finish(strain), std::move(solutions), info};
}

std::map<int, std::vector<Viscoelastic3D::Complex>> Viscoelastic3D::Surface(
    const Solutions& solutions, Component component) const {
  std::map<int, std::vector<Complex>> out;
  for (const auto& [l, x] : solutions.spheroidal) {
    const auto& op = Spheroidal(l);
    const auto columns = l + 1;
    const auto dof = op.SurfaceDof(component);
    std::vector<Complex> values(columns, 0.0);
    if (dof >= 0) {
      for (auto c = 0; c < columns; ++c) {
        values[c] = x[static_cast<std::size_t>(c) * op.DofCount() + dof];
      }
    }
    out.emplace(l, std::move(values));
  }
  return out;
}

}  // namespace sl3dpp
