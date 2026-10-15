#include "sl3dpp/Viscoelastic.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>

namespace sl3dpp {

namespace {
/// The force weights of the scalar order (M, R, S).
std::array<double, 3> ScalarWeights(double k2) {
  return {k2 * (k2 - 2.0), 4.0 / 3.0, k2};
}
}  // namespace

// -- the per-degree workspace
// ----------------------------------------------------

struct ViscoelasticOperator::Degree {
  int l;
  double k2;
  int e0, ne, ngll, ndof;
  DegreeOperator system;
  std::vector<double> B;                      // (ne, q, i): r_q D(q, i) / jac_e
  std::vector<double> Wq;                     // (ne, q)
  std::vector<int> Udof, Vdof;                // (ne, i), -1 absent
  std::vector<std::vector<double>> mu, itau;  // per branch, (ne, q)
  std::vector<double> loadVec, tideVec;
  std::vector<double> momentRow;  // degree 0 only

  Degree(const Rheology& rheology, int degree,
         const ViscoelasticOperator::Options& options)
      : l(degree),
        k2(degree * (degree + 1.0)),
        system(rheology.BaseMaterial(), degree,
               {.eps = options.eps, .truncation = options.truncation}) {
    const auto& mesh = rheology.BaseMaterial().Mesh();
    e0 = system.FirstElement();
    ne = mesh.ElementCount() - e0;
    ngll = mesh.NGll();
    ndof = system.DofCount();
    B.resize(static_cast<std::size_t>(ne) * ngll * ngll);
    Wq.resize(static_cast<std::size_t>(ne) * ngll);
    Udof.resize(static_cast<std::size_t>(ne) * ngll);
    Vdof.resize(static_cast<std::size_t>(ne) * ngll);
    const auto w = mesh.Weights();
    for (int e = 0; e < ne; ++e) {
      const double jac = mesh.Jacobian(e0 + e);
      for (int q = 0; q < ngll; ++q) {
        Wq[static_cast<std::size_t>(e) * ngll + q] = w[q] * jac;
        for (int i = 0; i < ngll; ++i) {
          B[(static_cast<std::size_t>(e) * ngll + q) * ngll + i] =
              mesh.Radius(e0 + e, q) * mesh.Derivative(q, i) / jac;
        }
        const int gn = mesh.GlobalIndex(e0 + e, q);
        Udof[static_cast<std::size_t>(e) * ngll + q] =
            system.Dof(gn, Component::U);
        Vdof[static_cast<std::size_t>(e) * ngll + q] =
            system.Dof(gn, Component::V);
      }
    }
    for (std::size_t k = 0; k < rheology.BranchCount(); ++k) {
      const auto& branch = rheology.BranchAt(k);
      std::vector<double> muk(static_cast<std::size_t>(ne) * ngll);
      std::vector<double> itauk(static_cast<std::size_t>(ne) * ngll);
      for (int e = 0; e < ne; ++e) {
        for (int q = 0; q < ngll; ++q) {
          const std::size_t full = static_cast<std::size_t>(e0 + e) * ngll + q;
          muk[static_cast<std::size_t>(e) * ngll + q] = branch.mu[full];
          itauk[static_cast<std::size_t>(e) * ngll + q] = branch.itau[full];
        }
      }
      mu.push_back(std::move(muk));
      itau.push_back(std::move(itauk));
    }
    loadVec = system.Load();
    tideVec = system.Tide(l == 0 ? 2 : -1);
    if (l == 0) {
      const auto& material = rheology.BaseMaterial();
      const auto rho = material.Rho();
      momentRow.assign(ndof, 0.0);
      const double root = std::sqrt(4.0 * std::numbers::pi);
      for (int e = 0; e < ne; ++e) {
        for (int q = 0; q < ngll; ++q) {
          const int dof = Udof[static_cast<std::size_t>(e) * ngll + q];
          if (dof < 0) continue;
          const double r = mesh.Radius(e0 + e, q);
          momentRow[dof] += root * Wq[static_cast<std::size_t>(e) * ngll + q] *
                            rho[static_cast<std::size_t>(e0 + e) * ngll + q] *
                            r * r * r;
        }
      }
    }
  }

  [[nodiscard]] std::size_t NodalSize(int ncols) const {
    return static_cast<std::size_t>(ne) * ngll * ncols;
  }

  /// One component of a solution gathered to the nodes, zero where
  /// absent: out(e, q, c).
  void Gather(std::span<const double> x, const std::vector<int>& dofs,
              int ncols, std::span<double> out) const {
    for (int e = 0; e < ne; ++e) {
      for (int q = 0; q < ngll; ++q) {
        const int dof = dofs[static_cast<std::size_t>(e) * ngll + q];
        for (int c = 0; c < ncols; ++c) {
          out[(static_cast<std::size_t>(e) * ngll + q) * ncols + c] =
              dof >= 0 ? x[static_cast<std::size_t>(c) * ndof + dof] : 0.0;
        }
      }
    }
  }

  /// The driving scalars (3, ne, ngll, ncols) of a solution array.
  [[nodiscard]] std::vector<double> Strain(std::span<const double> x,
                                           int ncols) const {
    const std::size_t block = NodalSize(ncols);
    std::vector<double> d(3 * block, 0.0);
    std::vector<double> U(block), V(block);
    Gather(x, Udof, ncols, U);
    Gather(x, Vdof, ncols, V);
    auto at = [&](std::span<const double> f, int e, int i, int c) {
      return f[(static_cast<std::size_t>(e) * ngll + i) * ncols + c];
    };
    for (int e = 0; e < ne; ++e) {
      for (int q = 0; q < ngll; ++q) {
        for (int c = 0; c < ncols; ++c) {
          const std::size_t node =
              (static_cast<std::size_t>(e) * ngll + q) * ncols + c;
          double dU = 0.0, dV = 0.0;
          for (int i = 0; i < ngll; ++i) {
            const double b =
                B[(static_cast<std::size_t>(e) * ngll + q) * ngll + i];
            dU += b * at(U, e, i, c);
            dV += b * at(V, e, i, c);
          }
          d[block + node] = dU - at(U, e, q, c) + 0.5 * k2 * at(V, e, q, c);
          if (l > 0) {
            d[node] = at(V, e, q, c);
            d[2 * block + node] = dV - at(V, e, q, c) + at(U, e, q, c);
          }
        }
      }
    }
    return d;
  }

  /// The dual vector pairing nodal scalar coefficients (cM, cR, cS) with
  /// the test functions' strain operators: the memory force when the
  /// coefficients carry the branch moduli and quadrature.
  [[nodiscard]] std::vector<double> DualFromScalars(
      std::span<const double> scalars, int ncols) const {
    const std::size_t block = NodalSize(ncols);
    const auto cM = scalars.subspan(0, block);
    const auto cR = scalars.subspan(block, block);
    const auto cS = scalars.subspan(2 * block, block);
    std::vector<double> b(static_cast<std::size_t>(ndof) * ncols, 0.0);
    auto at = [&](std::span<const double> f, int e, int i, int c) {
      return f[(static_cast<std::size_t>(e) * ngll + i) * ncols + c];
    };
    for (int e = 0; e < ne; ++e) {
      for (int i = 0; i < ngll; ++i) {
        const int iU = Udof[static_cast<std::size_t>(e) * ngll + i];
        const int iV = Vdof[static_cast<std::size_t>(e) * ngll + i];
        for (int c = 0; c < ncols; ++c) {
          double bU = -at(cR, e, i, c) + at(cS, e, i, c);
          double bV =
              0.5 * k2 * at(cR, e, i, c) - at(cS, e, i, c) + at(cM, e, i, c);
          for (int q = 0; q < ngll; ++q) {
            const double bqi =
                B[(static_cast<std::size_t>(e) * ngll + q) * ngll + i];
            bU += bqi * at(cR, e, q, c);
            bV += bqi * at(cS, e, q, c);
          }
          if (iU >= 0) b[static_cast<std::size_t>(c) * ndof + iU] += bU;
          if (iV >= 0) b[static_cast<std::size_t>(c) * ndof + iV] += bV;
        }
      }
    }
    return b;
  }

  /// The force of branch states mem (K, 3, ne, ngll, ncols).
  [[nodiscard]] std::vector<double> MemoryForce(std::span<const double> mem,
                                                int ncols) const {
    const std::size_t block = NodalSize(ncols);
    const auto wc = ScalarWeights(k2);
    std::vector<double> weighted(3 * block, 0.0);
    for (std::size_t k = 0; k < mu.size(); ++k) {
      for (int s = 0; s < 3; ++s) {
        for (int e = 0; e < ne; ++e) {
          for (int q = 0; q < ngll; ++q) {
            const double factor =
                wc[s] * mu[k][static_cast<std::size_t>(e) * ngll + q] *
                Wq[static_cast<std::size_t>(e) * ngll + q];
            if (factor == 0.0) continue;
            const std::size_t node =
                (static_cast<std::size_t>(e) * ngll + q) * ncols;
            const std::size_t src = (k * 3 + s) * block + node;
            for (int c = 0; c < ncols; ++c) {
              weighted[s * block + node + c] += factor * mem[src + c];
            }
          }
        }
      }
    }
    return DualFromScalars(weighted, ncols);
  }
};

// -- flavours
// ----------------------------------------------------------------------

struct ViscoelasticOperator::Flavour {
  std::unique_ptr<Material> material;
  std::map<int, DegreeOperator> systems;
  std::map<int, std::vector<double>> units;  // (ndof x 2): load, tide
};

ViscoelasticOperator::ViscoelasticOperator(const Rheology& rheology, int lmax,
                                           Options options)
    : rheology_(std::make_shared<const Rheology>(rheology)),
      lmax_(lmax),
      options_(options) {
  if (lmax < 0) throw std::invalid_argument("lmax must be non-negative");
}

ViscoelasticOperator::~ViscoelasticOperator() = default;
ViscoelasticOperator::ViscoelasticOperator(ViscoelasticOperator&&) noexcept =
    default;
ViscoelasticOperator& ViscoelasticOperator::operator=(
    ViscoelasticOperator&&) noexcept = default;

ViscoelasticOperator::Degree& ViscoelasticOperator::Deg(int l) const {
  auto found = degrees_.find(l);
  if (found == degrees_.end()) {
    found =
        degrees_.emplace(l, std::make_unique<Degree>(*rheology_, l, options_))
            .first;
  }
  return *found->second;
}

const std::vector<ViscoelasticOperator::BranchWeights>&
ViscoelasticOperator::SchemeWeightsOf(Scheme scheme, double dt) const {
  const auto key = std::pair(static_cast<int>(scheme), dt);
  auto found = weights_.find(key);
  if (found == weights_.end()) {
    std::vector<BranchWeights> out;
    for (std::size_t k = 0; k < rheology_->BranchCount(); ++k) {
      const auto& itau = rheology_->BranchAt(k).itau;
      BranchWeights w;
      w.e.resize(itau.size());
      if (scheme == Scheme::BackwardEuler) {
        for (std::size_t i = 0; i < itau.size(); ++i) {
          w.e[i] = BackwardEulerWeight(dt * itau[i]);
        }
      } else {
        w.alpha.resize(itau.size());
        w.beta.resize(itau.size());
        for (std::size_t i = 0; i < itau.size(); ++i) {
          const auto tw = ExponentialTrapezoidWeights(dt * itau[i]);
          w.e[i] = tw.e;
          w.alpha[i] = tw.alpha;
          w.beta[i] = tw.beta;
        }
      }
      out.push_back(std::move(w));
    }
    found = weights_.emplace(key, std::move(out)).first;
  }
  return found->second;
}

Material ViscoelasticOperator::EffectiveMaterial(Scheme scheme,
                                                 double dt) const {
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument("no effective operator for ExponentialEuler");
  }
  const auto& weights = SchemeWeightsOf(scheme, dt);
  std::vector<std::vector<double>> w;
  for (const auto& branch : weights) {
    if (scheme == Scheme::BackwardEuler) {
      w.push_back(branch.e);
    } else {
      std::vector<double> phi1(branch.beta.size());
      for (std::size_t i = 0; i < phi1.size(); ++i) {
        phi1[i] = 1.0 - branch.beta[i];
      }
      w.push_back(std::move(phi1));
    }
  }
  return BaseMaterial().WithModuli(rheology_->EffectiveModuli(w)).value();
}

ViscoelasticOperator::Flavour& ViscoelasticOperator::FlavourOf(
    Scheme scheme, double dt) const {
  const auto key = std::pair(static_cast<int>(scheme), dt);
  for (auto it = flavours_.begin(); it != flavours_.end(); ++it) {
    if (it->first == key) {
      flavours_.splice(flavours_.begin(), flavours_, it);
      return *flavours_.front().second;
    }
  }
  constexpr std::size_t kFlavourCache = 4;
  while (flavours_.size() >= kFlavourCache) flavours_.pop_back();
  auto flavour = std::make_unique<Flavour>();
  flavour->material = std::make_unique<Material>(EffectiveMaterial(scheme, dt));
  flavours_.emplace_front(key, std::move(flavour));
  return *flavours_.front().second;
}

const DegreeOperator& ViscoelasticOperator::EffectiveSystem(Flavour& flavour,
                                                            int l) const {
  auto found = flavour.systems.find(l);
  if (found == flavour.systems.end()) {
    found = flavour.systems
                .emplace(l, DegreeOperator(*flavour.material, l,
                                           {.eps = options_.eps,
                                            .truncation = options_.truncation}))
                .first;
  }
  return found->second;
}

const std::vector<double>& ViscoelasticOperator::UnitResponses(Flavour* flavour,
                                                               int l) const {
  auto& deg = Deg(l);
  auto& cache = flavour == nullptr ? unrelaxedUnits_ : flavour->units;
  auto found = cache.find(l);
  if (found == cache.end()) {
    const auto& system =
        flavour == nullptr ? deg.system : EffectiveSystem(*flavour, l);
    std::vector<double> rhs(static_cast<std::size_t>(deg.ndof) * 2, 0.0);
    std::ranges::copy(deg.loadVec, rhs.begin());
    std::ranges::copy(deg.tideVec, rhs.begin() + deg.ndof);
    found = cache.emplace(l, system.Solve(rhs, 2)).first;
  }
  return found->second;
}

void ViscoelasticOperator::AddChannels(std::vector<double>& x, int l,
                                       Flavour* flavour, double t,
                                       const LoadCallable& load,
                                       const LoadCallable& tide,
                                       int ncols) const {
  if (!load && !tide) return;
  const auto& deg = Deg(l);
  const auto& units = UnitResponses(flavour, l);
  auto add = [&](const std::optional<std::vector<double>>& columns, int unit) {
    if (!columns) return;
    if (static_cast<int>(columns->size()) != ncols) {
      throw std::invalid_argument(
          std::format("a degree-{} channel must supply {} columns, got {}", l,
                      ncols, columns->size()));
    }
    for (int c = 0; c < ncols; ++c) {
      const double amplitude = (*columns)[c];
      if (amplitude == 0.0) continue;
      for (int d = 0; d < deg.ndof; ++d) {
        x[static_cast<std::size_t>(c) * deg.ndof + d] +=
            units[static_cast<std::size_t>(unit) * deg.ndof + d] * amplitude;
      }
    }
  };
  if (load) add(load(l, t), 0);
  if (tide) add(tide(l, t), 1);
}

ViscoelasticState ViscoelasticOperator::ZeroState(
    const std::function<int(int)>& ncols) const {
  ViscoelasticState state;
  const auto K = rheology_->BranchCount();
  for (int l = 0; l <= lmax_; ++l) {
    const auto& deg = Deg(l);
    state.m[l].assign(K * 3 * deg.NodalSize(ncols(l)), 0.0);
  }
  return state;
}

std::map<int, std::vector<double>> ViscoelasticOperator::SolveElastic(
    const ViscoelasticState& state, const LoadCallable& load,
    const LoadCallable& tide) const {
  std::map<int, std::vector<double>> solutions;
  const auto K = rheology_->BranchCount();
  for (const auto& [l, m] : state.m) {
    const auto& deg = Deg(l);
    const int ncols = static_cast<int>(m.size() / (K * 3 * deg.ne * deg.ngll));
    auto x = deg.system.Solve(deg.MemoryForce(m, ncols), ncols);
    AddChannels(x, l, nullptr, state.t, load, tide, ncols);
    solutions.emplace(l, std::move(x));
  }
  return solutions;
}

std::map<int, std::vector<double>> ViscoelasticOperator::Strain(
    const std::map<int, std::vector<double>>& solutions) const {
  std::map<int, std::vector<double>> out;
  for (const auto& [l, x] : solutions) {
    const auto& deg = Deg(l);
    const int ncols = static_cast<int>(x.size() / deg.ndof);
    out.emplace(l, deg.Strain(x, ncols));
  }
  return out;
}

std::map<int, std::vector<double>> ViscoelasticOperator::Surface(
    const std::map<int, std::vector<double>>& solutions,
    Component component) const {
  std::map<int, std::vector<double>> out;
  for (const auto& [l, x] : solutions) {
    const auto& deg = Deg(l);
    const int ncols = static_cast<int>(x.size() / deg.ndof);
    const int dof = deg.system.SurfaceDof(component);
    std::vector<double> values(ncols, 0.0);
    if (dof >= 0) {
      for (int c = 0; c < ncols; ++c) {
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

std::vector<double> ViscoelasticOperator::InertiaMoment(
    std::span<const double> solution0, int ncols) const {
  const auto& deg = Deg(0);
  std::vector<double> out(ncols, 0.0);
  for (int c = 0; c < ncols; ++c) {
    double sum = 0.0;
    for (int d = 0; d < deg.ndof; ++d) {
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
    : op_(&op), scheme_(scheme), dt_(dt), t_(state.t + dt) {
  const auto& weights = op.SchemeWeightsOf(scheme, dt);
  std::map<int, std::vector<double>> dStart;
  if (scheme == Scheme::ExponentialTrapezoid) {
    dStart = op.StartStrain(state, load, tide);
  }
  const auto K = op.GetRheology().BranchCount();
  for (const auto& [l, m] : state.m) {
    const auto& deg = op.Deg(l);
    const int ncols = static_cast<int>(m.size() / (K * 3 * deg.ne * deg.ngll));
    ncols_[l] = ncols;
    const std::size_t block = deg.NodalSize(ncols);
    std::vector<double> mem(m.size());
    for (std::size_t k = 0; k < K; ++k) {
      const auto& w = weights[k];
      for (int s = 0; s < 3; ++s) {
        for (int e = 0; e < deg.ne; ++e) {
          for (int q = 0; q < deg.ngll; ++q) {
            const std::size_t full =
                static_cast<std::size_t>(deg.e0 + e) * deg.ngll + q;
            const std::size_t node =
                (static_cast<std::size_t>(e) * deg.ngll + q) * ncols;
            const std::size_t at = (k * 3 + s) * block + node;
            for (int c = 0; c < ncols; ++c) {
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
    mem_.emplace(l, std::move(mem));
  }
}

const std::map<int, std::vector<double>>& ViscoelasticStep::MemorySolutions()
    const {
  if (memorySolutions_.empty()) {
    auto& flavour = op_->FlavourOf(scheme_, dt_);
    for (const auto& [l, mem] : mem_) {
      const auto& deg = op_->Deg(l);
      const int ncols = ncols_.at(l);
      memorySolutions_.emplace(l,
                               op_->EffectiveSystem(flavour, l)
                                   .Solve(deg.MemoryForce(mem, ncols), ncols));
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

std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
ViscoelasticStep::Finish(const LoadCallable& load,
                         const LoadCallable& tide) const {
  const auto& op = *op_;
  const auto& weights = op.SchemeWeightsOf(scheme_, dt_);
  auto& flavour = op.FlavourOf(scheme_, dt_);
  const auto& memory = MemorySolutions();
  const auto K = op.GetRheology().BranchCount();

  ViscoelasticState out;
  out.t = t_;
  std::map<int, std::vector<double>> solutions;
  for (const auto& [l, mem] : mem_) {
    const auto& deg = op.Deg(l);
    const int ncols = ncols_.at(l);
    auto x = memory.at(l);
    op.AddChannels(x, l, &flavour, t_, load, tide, ncols);
    auto d1 = deg.Strain(x, ncols);
    const std::size_t block = deg.NodalSize(ncols);
    std::vector<double> mNew = mem;
    for (std::size_t k = 0; k < K; ++k) {
      const auto& w = weights[k];
      for (int s = 0; s < 3; ++s) {
        for (int e = 0; e < deg.ne; ++e) {
          for (int q = 0; q < deg.ngll; ++q) {
            const std::size_t full =
                static_cast<std::size_t>(deg.e0 + e) * deg.ngll + q;
            const double factor = scheme_ == Scheme::BackwardEuler
                                      ? 1.0 - w.e[full]
                                      : w.beta[full];
            const std::size_t node =
                (static_cast<std::size_t>(e) * deg.ngll + q) * ncols;
            const std::size_t at = (k * 3 + s) * block + node;
            for (int c = 0; c < ncols; ++c) {
              mNew[at + c] += factor * d1[s * block + node + c];
            }
          }
        }
      }
    }
    out.m.emplace(l, std::move(mNew));
    out.strain.emplace(l, std::move(d1));
    solutions.emplace(l, std::move(x));
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
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  return ViscoelasticStep(*this, state, dt, scheme, load, tide);
}

std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
ViscoelasticOperator::Step(const ViscoelasticState& state, double dt,
                           Scheme scheme, const LoadCallable& load,
                           const LoadCallable& tide) const {
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  if (scheme == Scheme::ExponentialEuler) {
    auto dStart = StartStrain(state, load, tide);
    const auto& weights = SchemeWeightsOf(Scheme::ExponentialTrapezoid, dt);
    const auto K = rheology_->BranchCount();
    ViscoelasticState out;
    out.t = state.t + dt;
    for (const auto& [l, m] : state.m) {
      const auto& deg = Deg(l);
      const int ncols =
          static_cast<int>(m.size() / (K * 3 * deg.ne * deg.ngll));
      const std::size_t block = deg.NodalSize(ncols);
      std::vector<double> mNew(m.size());
      for (std::size_t k = 0; k < K; ++k) {
        const auto& w = weights[k];
        for (int s = 0; s < 3; ++s) {
          for (int e = 0; e < deg.ne; ++e) {
            for (int q = 0; q < deg.ngll; ++q) {
              const std::size_t full =
                  static_cast<std::size_t>(deg.e0 + e) * deg.ngll + q;
              const std::size_t node =
                  (static_cast<std::size_t>(e) * deg.ngll + q) * ncols;
              const std::size_t at = (k * 3 + s) * block + node;
              for (int c = 0; c < ncols; ++c) {
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
  return ViscoelasticStep(*this, state, dt, scheme, load, tide)
      .Finish(load, tide);
}

std::pair<ViscoelasticState, std::map<int, std::vector<double>>>
ViscoelasticOperator::StepBack(const ViscoelasticState& state, double dt,
                               Scheme scheme, const LoadCallable& load,
                               const LoadCallable& tide) const {
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "a step inverts for ExponentialTrapezoid or BackwardEuler");
  }
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  if (!state.HasStrain()) {
    throw std::invalid_argument(
        "the state must carry its strain cache; reconstruct from a "
        "recorded state or a completed step");
  }
  if (scheme == Scheme::ExponentialTrapezoid) {
    // the same update at -dt with the start and end strains swapped
    return ViscoelasticStep(*this, state, -dt, scheme, {}, {})
        .Finish(load, tide);
  }
  // backward Euler inverts explicitly: m^n = (1 + h) m^{n+1} - h d^{n+1}
  const auto K = rheology_->BranchCount();
  ViscoelasticState prev;
  prev.t = state.t - dt;
  for (const auto& [l, m] : state.m) {
    const auto& deg = Deg(l);
    const int ncols = static_cast<int>(m.size() / (K * 3 * deg.ne * deg.ngll));
    const std::size_t block = deg.NodalSize(ncols);
    const auto& d1 = state.strain.at(l);
    std::vector<double> mPrev(m.size());
    for (std::size_t k = 0; k < K; ++k) {
      for (int s = 0; s < 3; ++s) {
        for (int e = 0; e < deg.ne; ++e) {
          for (int q = 0; q < deg.ngll; ++q) {
            const double h =
                dt * deg.itau[k][static_cast<std::size_t>(e) * deg.ngll + q];
            const std::size_t node =
                (static_cast<std::size_t>(e) * deg.ngll + q) * ncols;
            const std::size_t at = (k * 3 + s) * block + node;
            for (int c = 0; c < ncols; ++c) {
              mPrev[at + c] =
                  (1.0 + h) * m[at + c] - h * d1[s * block + node + c];
            }
          }
        }
      }
    }
    prev.m.emplace(l, std::move(mPrev));
  }
  auto solutions = SolveElastic(prev, load, tide);
  prev.strain = Strain(solutions);
  return {std::move(prev), std::move(solutions)};
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
  double top = 0.0;
  for (const auto& [l, m] : newState.m) {
    for (double v : m) top = std::max(top, std::abs(v));
  }
  const double atol = top > 0.0 ? 1e-6 * rtol * top : 1e-300;

  double total = 0.0;
  std::size_t count = 0;
  for (const auto& [l, m] : state.m) {
    const auto& deg = Deg(l);
    const int ncols = static_cast<int>(m.size() / (K * 3 * deg.ne * deg.ngll));
    const std::size_t block = deg.NodalSize(ncols);
    const auto& mNew = newState.m.at(l);
    for (std::size_t k = 0; k < K; ++k) {
      const auto& w = weights[k];
      for (int s = 0; s < 3; ++s) {
        for (int e = 0; e < deg.ne; ++e) {
          for (int q = 0; q < deg.ngll; ++q) {
            const std::size_t full =
                static_cast<std::size_t>(deg.e0 + e) * deg.ngll + q;
            const std::size_t node =
                (static_cast<std::size_t>(e) * deg.ngll + q) * ncols;
            const std::size_t at = (k * 3 + s) * block + node;
            for (int c = 0; c < ncols; ++c) {
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
  const double err = count ? std::sqrt(total / count) : 0.0;
  constexpr double kSafety = 0.9, kMin = 0.2, kMax = 4.0;
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
  const auto key = std::pair(static_cast<int>(scheme), dt);
  auto found = effectiveLove_.find(key);
  if (found == effectiveLove_.end()) {
    const auto material = EffectiveMaterial(scheme, dt);
    found = effectiveLove_
                .emplace(key, ComputeLoveNumbers(
                                  material, lmax_,
                                  {.eps = options_.eps,
                                   .truncation = options_.truncation}))
                .first;
  }
  return found->second;
}

const LoveNumbers& ViscoelasticOperator::ElasticLoveNumbers() const {
  const auto key = std::pair(-1, 0.0);
  auto found = effectiveLove_.find(key);
  if (found == effectiveLove_.end()) {
    found = effectiveLove_
                .emplace(key, ComputeLoveNumbers(
                                  BaseMaterial(), lmax_,
                                  {.eps = options_.eps,
                                   .truncation = options_.truncation}))
                .first;
  }
  return found->second;
}

}  // namespace sl3dpp
