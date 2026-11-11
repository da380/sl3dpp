#include "sl3dpp/ToroidalViscoelastic.hpp"

#include <algorithm>
#include <format>
#include <span>
#include <stdexcept>

#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

/// The per-degree workspace: the unrelaxed system, the window of its
/// elements and the strain operators of the two toroidal scalars.
struct ToroidalViscoelastic::Degree {
  int l;
  double k2;
  int e0 = 0, ne = 0, ngll = 0, np = 0, ndof = 0;
  bool gll = false;
  ToroidalOperator system;
  const MemoryPoints* points;
  std::vector<double> B;   // (ne, p, i): r_p l_i'(xi_p) / jac_e
  std::vector<double> Wq;  // (ne, p): w_p jac_e
  std::vector<int> Wdof;   // (ne, i), -1 absent
  std::vector<std::vector<double>> mu;  // per branch, (ne, p)

  Degree(const Rheology& rheology, int degree, const Material& nodalMaterial,
         const ToroidalOperator::Options& options,
         const MemoryPoints& memoryPoints,
         const std::vector<std::vector<double>>& pointMu)
      : l(degree),
        k2(degree * (degree + 1.0)),
        system(nodalMaterial, degree, options),
        points(&memoryPoints) {
    const auto& mesh = rheology.BaseMaterial().Mesh();
    const auto subset = system.FirstElement();
    ngll = mesh.NGll();
    np = memoryPoints.count;
    gll = memoryPoints.set == MemoryPointSet::Gll;
    ndof = system.DofCount();
    // The window is the system's whole subset, the elastic regions
    // included (their branch moduli vanish, so their memory is inert):
    // the strain operators then serve a perturbation of the moduli
    // anywhere the toroidal unknowns live.
    static_cast<void>(rheology);
    e0 = subset;
    ne = mesh.ElementCount() - subset;
    B.resize(static_cast<std::size_t>(ne) * np * ngll);
    Wq.resize(static_cast<std::size_t>(ne) * np);
    Wdof.resize(static_cast<std::size_t>(ne) * ngll);
    for (auto e = 0; e < ne; ++e) {
      const auto jac = mesh.Jacobian(e0 + e);
      for (auto p = 0; p < np; ++p) {
        Wq[static_cast<std::size_t>(e) * np + p] = memoryPoints.weight[p] * jac;
        const auto rp = memoryPoints.RadiusAt(mesh, e0 + e, p);
        for (auto i = 0; i < ngll; ++i) {
          B[(static_cast<std::size_t>(e) * np + p) * ngll + i] =
              rp * memoryPoints.Derivative(p, i) / jac;
        }
      }
      for (auto q = 0; q < ngll; ++q) {
        Wdof[static_cast<std::size_t>(e) * ngll + q] =
            system.Dof(mesh.GlobalIndex(e0 + e, q));
      }
    }
    for (const auto& all : pointMu) {
      auto muk = std::vector<double>(static_cast<std::size_t>(ne) * np);
      for (auto e = 0; e < ne; ++e) {
        for (auto p = 0; p < np; ++p) {
          muk[static_cast<std::size_t>(e) * np + p] =
              all[static_cast<std::size_t>(e0 + e) * np + p];
        }
      }
      mu.push_back(std::move(muk));
    }
  }

  [[nodiscard]] std::size_t Block(int ncols) const {
    return static_cast<std::size_t>(ne) * np * ncols;
  }

  /// The scalars (2, ne, np, ncols) of a solution: M = W and
  /// S = r dW/dr - W at the memory points.
  [[nodiscard]] std::vector<double> Strain(std::span<const double> x,
                                           int ncols) const {
    const auto block = Block(ncols);
    auto d = std::vector<double>(2 * block, 0.0);
    for (auto e = 0; e < ne; ++e) {
      for (auto q = 0; q < np; ++q) {
        for (auto c = 0; c < ncols; ++c) {
          auto value = 0.0, rd = 0.0;
          for (auto i = 0; i < ngll; ++i) {
            const auto dof = Wdof[static_cast<std::size_t>(e) * ngll + i];
            if (dof < 0) continue;
            const auto w = x[static_cast<std::size_t>(c) * ndof + dof];
            rd += B[(static_cast<std::size_t>(e) * np + q) * ngll + i] * w;
            if (gll ? i == q : true) {
              value += (gll ? 1.0 : points->Value(q, i)) * w;
            }
          }
          const auto node = (static_cast<std::size_t>(e) * np + q) * ncols + c;
          d[node] = value;
          d[block + node] = rd - value;
        }
      }
    }
    return d;
  }

  /// The transpose of Strain on point coefficients (cM, cS).
  [[nodiscard]] std::vector<double> DualFromScalars(
      std::span<const double> scalars, int ncols) const {
    const auto block = Block(ncols);
    auto b = std::vector<double>(static_cast<std::size_t>(ndof) * ncols, 0.0);
    for (auto e = 0; e < ne; ++e) {
      for (auto i = 0; i < ngll; ++i) {
        const auto dof = Wdof[static_cast<std::size_t>(e) * ngll + i];
        if (dof < 0) continue;
        for (auto c = 0; c < ncols; ++c) {
          auto sum = 0.0;
          for (auto q = 0; q < np; ++q) {
            const auto node = (static_cast<std::size_t>(e) * np + q) * ncols + c;
            const auto value = gll ? (i == q ? 1.0 : 0.0) : points->Value(q, i);
            const auto bqi = B[(static_cast<std::size_t>(e) * np + q) * ngll + i];
            sum += value * (scalars[node] - scalars[block + node]) +
                   bqi * scalars[block + node];
          }
          b[static_cast<std::size_t>(c) * ndof + dof] += sum;
        }
      }
    }
    return b;
  }

  /// S^T (wc w Wq d): the force of scalars d (2, ne, np, ncols) under a
  /// point weight w over the whole mesh, (element, point).
  [[nodiscard]] std::vector<double> WeightedDual(std::span<const double> d,
                                                 std::span<const double> weight,
                                                 int ncols) const {
    const auto block = Block(ncols);
    const double wc[2] = {k2 * (k2 - 2.0), k2};
    auto weighted = std::vector<double>(2 * block, 0.0);
    for (auto s = 0; s < 2; ++s) {
      for (auto e = 0; e < ne; ++e) {
        for (auto q = 0; q < np; ++q) {
          const auto point = static_cast<std::size_t>(e) * np + q;
          const auto factor =
              wc[s] * weight[static_cast<std::size_t>(e0 + e) * np + q] * Wq[point];
          if (factor == 0.0) continue;
          for (auto c = 0; c < ncols; ++c) {
            weighted[s * block + point * ncols + c] =
                factor * d[s * block + point * ncols + c];
          }
        }
      }
    }
    return DualFromScalars(weighted, ncols);
  }

  /// The force of branch states mem (K, 2, ne, np, ncols).
  [[nodiscard]] std::vector<double> MemoryForce(std::span<const double> mem,
                                                int ncols) const {
    const auto block = Block(ncols);
    const double wc[2] = {k2 * (k2 - 2.0), k2};
    auto weighted = std::vector<double>(2 * block, 0.0);
    for (auto k = std::size_t{0}; k < mu.size(); ++k) {
      for (auto s = 0; s < 2; ++s) {
        for (auto e = 0; e < ne; ++e) {
          for (auto q = 0; q < np; ++q) {
            const auto factor = wc[s] *
                                mu[k][static_cast<std::size_t>(e) * np + q] *
                                Wq[static_cast<std::size_t>(e) * np + q];
            if (factor == 0.0) continue;
            const auto node = (static_cast<std::size_t>(e) * np + q) * ncols;
            for (auto c = 0; c < ncols; ++c) {
              weighted[s * block + node + c] +=
                  factor * mem[(k * 2 + s) * block + node + c];
            }
          }
        }
      }
    }
    return DualFromScalars(weighted, ncols);
  }
};

ToroidalViscoelastic::ToroidalViscoelastic(const ViscoelasticOperator& op)
    : op_(&op) {}
ToroidalViscoelastic::ToroidalViscoelastic(const ViscoelasticOperator& op,
                                           std::function<int(int)> columns)
    : op_(&op), columns_(std::move(columns)) {}

int ToroidalViscoelastic::Columns(int l) const {
  return columns_ ? columns_(l) : 2 * (l + 1);
}
ToroidalViscoelastic::~ToroidalViscoelastic() = default;

int ToroidalViscoelastic::LMax() const { return op_->LMax(); }

const ToroidalViscoelastic::Degree& ToroidalViscoelastic::Deg(int l) const {
  auto found = degrees_.find(l);
  if (found == degrees_.end()) {
    const auto& op = *op_;
    found = degrees_
                .emplace(l, std::make_unique<Degree>(
                                op.GetRheology(), l, op.NodalMaterial(),
                                ToroidalOperator::Options{
                                    .eps = op.options_.eps,
                                    .truncation = op.options_.truncation,
                                    .point_shear = op.ElasticPointShear()},
                                *op.points_, op.pointMu_))
                .first;
  }
  return *found->second;
}

const ToroidalOperator& ToroidalViscoelastic::Unrelaxed(int l) const {
  return Deg(l).system;
}

ToroidalViscoelastic::State ToroidalViscoelastic::ZeroState() const {
  State state;
  const auto K = op_->GetRheology().BranchCount();
  for (auto l = 1; l <= LMax(); ++l) {
    state.m[l].assign(K * 2 * Deg(l).Block(Columns(l)), 0.0);
  }
  return state;
}

ToroidalViscoelastic::Fields ToroidalViscoelastic::SolveElastic(
    const State& state, const Fields& force) const {
  auto out = Fields{};
  for (const auto& [l, m] : state.m) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    auto b = deg.MemoryForce(m, ncols);
    if (const auto f = force.find(l); f != force.end()) {
      if (f->second.size() != b.size()) {
        throw std::invalid_argument(std::format(
            "a degree-{} toroidal force has {} entries, the solutions {}", l,
            f->second.size(), b.size()));
      }
      for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f->second[i];
    }
    out.emplace(l, deg.system.Solve(b, ncols));
  }
  return out;
}

void ToroidalViscoelastic::SetStrain(State& state,
                                     const Fields& solutions) const {
  state.strain.clear();
  for (const auto& [l, x] : solutions) {
    state.strain.emplace(l, Deg(l).Strain(x, Columns(l)));
  }
}

std::pair<ToroidalViscoelastic::State, ToroidalViscoelastic::Fields>
ToroidalViscoelastic::Step(const State& state, double dt, Scheme scheme,
                           const Fields& force, const Fields* memory) const {
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "the toroidal step is implicit: ExponentialTrapezoid or "
        "BackwardEuler");
  }
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  const auto& op = *op_;
  const bool implicit = scheme == Scheme::BackwardEuler;
  const auto& weights = op.SchemeWeightsOf(scheme, dt);
  const auto K = op.GetRheology().BranchCount();
  State next;
  next.t = state.t + dt;
  auto solutions = Fields{};
  for (const auto& [l, m] : state.m) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    const auto block = deg.Block(ncols);
    const std::vector<double>* d0 = nullptr;
    if (!implicit) {
      const auto found = state.strain.find(l);
      if (found == state.strain.end()) {
        throw std::invalid_argument(
            "the trapezoid step needs the state's strain (SetStrain)");
      }
      d0 = &found->second;
    }
    // the decayed memory
    auto mem = std::vector<double>(m.size());
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& w = weights[k];
      for (auto s = 0; s < 2; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto node = (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const auto at = (k * 2 + s) * block + node;
            for (auto c = 0; c < ncols; ++c) {
              mem[at + c] = w.e[full] * m[at + c] +
                            (implicit ? 0.0
                                      : w.alpha[full] *
                                            (*d0)[s * block + node + c]);
            }
          }
        }
      }
    }
    if (memory != nullptr) {
      if (const auto add = memory->find(l); add != memory->end()) {
        for (auto i = std::size_t{0}; i < mem.size(); ++i) mem[i] += add->second[i];
      }
    }
    auto b = deg.MemoryForce(mem, ncols);
    if (const auto f = force.find(l); f != force.end()) {
      if (f->second.size() != b.size()) {
        throw std::invalid_argument(std::format(
            "a degree-{} toroidal force has {} entries, the solutions {}", l,
            f->second.size(), b.size()));
      }
      for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f->second[i];
    }
    auto x = Effective(scheme, dt, l).Solve(b, ncols);
    auto d1 = deg.Strain(x, ncols);
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& w = weights[k];
      for (auto s = 0; s < 2; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto factor = implicit ? 1.0 - w.e[full] : w.beta[full];
            const auto node = (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const auto at = (k * 2 + s) * block + node;
            for (auto c = 0; c < ncols; ++c) {
              mem[at + c] += factor * d1[s * block + node + c];
            }
          }
        }
      }
    }
    next.m.emplace(l, std::move(mem));
    next.strain.emplace(l, std::move(d1));
    solutions.emplace(l, std::move(x));
  }
  return {std::move(next), std::move(solutions)};
}

namespace {

/// The weight of a natural column in the sum over orders of a real
/// field's pairing: m = 0 real once, its imaginary part not at all,
/// m > 0 twice.
double ColumnWeight(int l, int c) {
  const auto m = c <= l ? c : c - (l + 1);
  return m == 0 ? (c == 0 ? 1.0 : 0.0) : 2.0;
}

}  // namespace

void ToroidalViscoelastic::AccumulateItau(
    const State& adjoint, const State& forward, double scale,
    std::vector<std::vector<double>>& kernel) const {
  const auto& op = *op_;
  const auto K = op.GetRheology().BranchCount();
  for (const auto& [l, md] : adjoint.m) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    const auto block = deg.Block(ncols);
    const auto& dFwd = forward.strain.at(l);
    const auto& mFwd = forward.m.at(l);
    const double wc[2] = {deg.k2 * (deg.k2 - 2.0), deg.k2};
    for (auto k = std::size_t{0}; k < K; ++k) {
      for (auto e = 0; e < deg.ne; ++e) {
        for (auto q = 0; q < deg.np; ++q) {
          const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
          const auto itau = op.pointItau_[k][full];
          if (itau <= 0.0) continue;
          const auto point = static_cast<std::size_t>(e) * deg.np + q;
          auto sum = 0.0;
          for (auto s = 0; s < 2; ++s) {
            const auto factor = wc[s] * deg.mu[k][point] * deg.Wq[point];
            for (auto c = 0; c < ncols; ++c) {
              const auto at = (k * 2 + s) * block + point * ncols + c;
              const auto sn = s * block + point * ncols + c;
              sum += ColumnWeight(l, c) * factor * md[at] * (dFwd[sn] - mFwd[at]);
            }
          }
          kernel[k][full] += scale * sum / itau;
        }
      }
    }
  }
}

void ToroidalViscoelastic::AccumulateMaterial(
    const Fields& adjointSolution, const State& forward,
    const Fields& forwardSolution, double weight,
    std::vector<double>& mu) const {
  const auto& op = *op_;
  const auto K = op.GetRheology().BranchCount();
  auto fractions = std::vector<std::span<const double>>{};
  for (auto k = std::size_t{0}; k < K; ++k) {
    fractions.push_back(op.BranchFractionAtPoints(k));
  }
  for (const auto& [l, y] : adjointSolution) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    const auto block = deg.Block(ncols);
    const auto sy = deg.Strain(y, ncols);
    const auto sx = deg.Strain(forwardSolution.at(l), ncols);
    const auto& mFwd = forward.m.at(l);
    const double wc[2] = {deg.k2 * (deg.k2 - 2.0), deg.k2};
    for (auto e = 0; e < deg.ne; ++e) {
      for (auto q = 0; q < deg.np; ++q) {
        const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
        const auto point = static_cast<std::size_t>(e) * deg.np + q;
        auto sum = 0.0;
        for (auto s = 0; s < 2; ++s) {
          for (auto c = 0; c < ncols; ++c) {
            const auto sn = s * block + point * ncols + c;
            auto term = -sx[sn];
            for (auto k = std::size_t{0}; k < K; ++k) {
              term += fractions[k][full] *
                      mFwd[(k * 2 + s) * block + point * ncols + c];
            }
            sum += ColumnWeight(l, c) * wc[s] * sy[sn] * term;
          }
        }
        mu[full] += weight * deg.Wq[point] * sum;
      }
    }
  }
}

void ToroidalViscoelastic::Jump(State& state, const Fields& solutions) const {
  const auto& op = *op_;
  const auto K = op.GetRheology().BranchCount();
  for (const auto& [l, x] : solutions) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    const auto block = deg.Block(ncols);
    const auto d = deg.Strain(x, ncols);
    auto& m = state.m.at(l);
    for (auto k = std::size_t{0}; k < K; ++k) {
      for (auto s = 0; s < 2; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto itau =
                op.pointItau_[k][static_cast<std::size_t>(deg.e0 + e) * deg.np + q];
            if (itau == 0.0) continue;
            const auto node = (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            for (auto c = 0; c < ncols; ++c) {
              m[(k * 2 + s) * block + node + c] +=
                  itau * d[s * block + node + c];
            }
          }
        }
      }
    }
  }
  state.strain.clear();
}

ToroidalViscoelastic::Fields ToroidalViscoelastic::Surface(
    const Fields& solutions) const {
  auto out = Fields{};
  for (const auto& [l, x] : solutions) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    const auto dof = deg.system.SurfaceDof();
    auto values = std::vector<double>(ncols, 0.0);
    for (auto c = 0; c < ncols; ++c) {
      values[c] = x[static_cast<std::size_t>(c) * deg.ndof + dof];
    }
    out.emplace(l, std::move(values));
  }
  return out;
}

const ToroidalOperator& ToroidalViscoelastic::Effective(Scheme scheme,
                                                        double dt,
                                                        int l) const {
  const auto& op = *op_;
  auto& flavour = op.FlavourOf(scheme, dt);
  auto system = flavour.toroidal.find(l);
  if (system == flavour.toroidal.end()) {
    system =
        flavour.toroidal
            .emplace(l, ToroidalOperator(
                            *flavour.material, l,
                            {.eps = op.options_.eps,
                             .truncation = op.options_.truncation,
                             .point_shear = flavour.pointShear.points != nullptr
                                                ? &flavour.pointShear
                                                : nullptr}))
            .first;
  }
  return system->second;
}

ToroidalViscoelastic::Dual ToroidalViscoelastic::ZeroDual() const {
  Dual dual;
  const auto K = op_->GetRheology().BranchCount();
  for (auto l = 1; l <= LMax(); ++l) {
    const auto block = Deg(l).Block(Columns(l));
    dual.m[l].assign(K * 2 * block, 0.0);
    dual.strain[l].assign(2 * block, 0.0);
  }
  return dual;
}

ToroidalViscoelastic::BaseStep ToroidalViscoelastic::Redo(
    const State& prev, double dt, Scheme scheme, const Fields& force) const {
  const auto& op = *op_;
  const bool implicit = scheme == Scheme::BackwardEuler;
  const auto& weights = op.SchemeWeightsOf(scheme, dt);
  const auto K = op.GetRheology().BranchCount();
  auto out = BaseStep{};
  for (const auto& [l, m] : prev.m) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    const auto block = deg.Block(ncols);
    const auto* d0 = implicit ? nullptr : &prev.strain.at(l);
    auto mem = std::vector<double>(m.size());
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& w = weights[k];
      for (auto s = 0; s < 2; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto node = (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const auto at = (k * 2 + s) * block + node;
            for (auto c = 0; c < ncols; ++c) {
              mem[at + c] = w.e[full] * m[at + c] +
                            (implicit ? 0.0
                                      : w.alpha[full] *
                                            (*d0)[s * block + node + c]);
            }
          }
        }
      }
    }
    auto b = deg.MemoryForce(mem, ncols);
    if (const auto f = force.find(l); f != force.end()) {
      for (auto i = std::size_t{0}; i < b.size(); ++i) b[i] += f->second[i];
    }
    out.solutions.emplace(l, Effective(scheme, dt, l).Solve(b, ncols));
    out.decayed.emplace(l, std::move(mem));
  }
  return out;
}

ToroidalViscoelastic::Fields ToroidalViscoelastic::MaterialForce(
    double dt, Scheme scheme, std::span<const double> dmu,
    const BaseStep& base) const {
  const auto& op = *op_;
  const bool implicit = scheme == Scheme::BackwardEuler;
  const auto& weights = op.SchemeWeightsOf(scheme, dt);
  const auto K = op.GetRheology().BranchCount();
  auto fractions = std::vector<std::span<const double>>{};
  for (auto k = std::size_t{0}; k < K; ++k) {
    fractions.push_back(op.BranchFractionAtPoints(k));
  }
  // minus the effective modulus's variation: -(1 - sum_k f_k b_k) dmu
  auto effective = std::vector<double>(dmu.size());
  for (auto i = std::size_t{0}; i < dmu.size(); ++i) {
    auto factor = 1.0;
    for (auto k = std::size_t{0}; k < K; ++k) {
      factor -= fractions[k][i] *
                (implicit ? 1.0 - weights[k].e[i] : weights[k].beta[i]);
    }
    effective[i] = -factor * dmu[i];
  }
  auto out = Fields{};
  for (const auto& [l, w] : base.solutions) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    const auto block = deg.Block(ncols);
    auto force = deg.WeightedDual(deg.Strain(w, ncols), effective, ncols);
    // the decayed memory's force with the branch moduli's variation
    const auto& mem = base.decayed.at(l);
    auto branch = std::vector<double>(dmu.size());
    for (auto k = std::size_t{0}; k < K; ++k) {
      for (auto i = std::size_t{0}; i < dmu.size(); ++i) {
        branch[i] = fractions[k][i] * dmu[i];
      }
      const auto part = deg.WeightedDual(
          std::span(mem).subspan(k * 2 * block, 2 * block), branch, ncols);
      for (auto i = std::size_t{0}; i < force.size(); ++i) force[i] += part[i];
    }
    out.emplace(l, std::move(force));
  }
  return out;
}

ToroidalViscoelastic::Fields ToroidalViscoelastic::StartMaterialResponse(
    std::span<const double> dmu, const Fields& baseStart) const {
  auto minus = std::vector<double>(dmu.begin(), dmu.end());
  for (auto& v : minus) v = -v;
  auto out = Fields{};
  for (const auto& [l, w] : baseStart) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    out.emplace(l, deg.system.Solve(
                       deg.WeightedDual(deg.Strain(w, ncols), minus, ncols),
                       ncols));
  }
  return out;
}

ToroidalViscoelastic::Fields ToroidalViscoelastic::FirstOrderDrive(
    bool discrete, Scheme scheme, double dt, const State& prev,
    const State& next,
    const std::vector<std::vector<double>>& ditauPoints) const {
  const auto& op = *op_;
  const bool implicit = scheme == Scheme::BackwardEuler;
  const auto K = op.GetRheology().BranchCount();
  const auto& weights = op.SchemeWeightsOf(scheme, dt);
  const auto* dweights =
      discrete ? &op.SchemeWeightDerivativesOf(scheme, dt) : nullptr;
  auto out = Fields{};
  for (const auto& [l, mPrev] : prev.m) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    const auto block = deg.Block(ncols);
    const auto& d0 = prev.strain.at(l);
    const auto& d1 = next.strain.at(l);
    const auto& mNext = next.m.at(l);
    auto extra = std::vector<double>(mPrev.size(), 0.0);
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& w = weights[k];
      for (auto s = 0; s < 2; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto delta = ditauPoints[k][full];
            if (delta == 0.0) continue;
            const auto node = (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            const auto at = (k * 2 + s) * block + node;
            const auto sn = s * block + node;
            const auto dh = dt * delta;
            if (discrete) {
              const auto& dw = (*dweights)[k];
              for (auto c = 0; c < ncols; ++c) {
                extra[at + c] =
                    implicit ? dw.e[full] * dh * (mPrev[at + c] - d1[sn + c])
                             : dh * (dw.e[full] * mPrev[at + c] +
                                     dw.alpha[full] * d0[sn + c] +
                                     dw.beta[full] * d1[sn + c]);
              }
            } else {
              const auto itau = op.pointItau_[k][full];
              if (itau <= 0.0) continue;
              const auto ratio = delta / itau;
              for (auto c = 0; c < ncols; ++c) {
                const auto q0 = (d0[sn + c] - mPrev[at + c]) * ratio;
                const auto q1 = (d1[sn + c] - mNext[at + c]) * ratio;
                extra[at + c] = implicit ? (1.0 - w.e[full]) * q1
                                         : w.alpha[full] * q0 + w.beta[full] * q1;
              }
            }
          }
        }
      }
    }
    out.emplace(l, std::move(extra));
  }
  return out;
}

ToroidalViscoelastic::Fields ToroidalViscoelastic::StepTransposed(
    Dual& dual, double dt, Scheme scheme, const Fields& surfaceDual,
    const ItauPairing* pairing, const MaterialPairing* material) const {
  const auto& op = *op_;
  auto base = BaseStep{};
  auto fractions = std::vector<std::span<const double>>{};
  if (material != nullptr) {
    base = Redo(*material->prev, dt, scheme, *material->force);
    for (auto k = std::size_t{0}; k < op_->GetRheology().BranchCount(); ++k) {
      fractions.push_back(op_->BranchFractionAtPoints(k));
    }
  }
  const bool implicit = scheme == Scheme::BackwardEuler;
  const auto& weights = op.SchemeWeightsOf(scheme, dt);
  const auto* dweights =
      pairing != nullptr ? &op.SchemeWeightDerivativesOf(scheme, dt) : nullptr;
  const auto K = op.GetRheology().BranchCount();
  auto out = Fields{};
  for (auto& [l, md] : dual.m) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    const auto block = deg.Block(ncols);
    auto& dd = dual.strain.at(l);
    // the duals of the end strain: its own, and the memory's through the
    // end weights
    auto summed = dd;
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& w = weights[k];
      for (auto s = 0; s < 2; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto b = implicit ? 1.0 - w.e[full] : w.beta[full];
            const auto node = (static_cast<std::size_t>(e) * deg.np + q) * ncols;
            for (auto c = 0; c < ncols; ++c) {
              summed[s * block + node + c] +=
                  b * md[(k * 2 + s) * block + node + c];
            }
          }
        }
      }
    }
    auto rhs = deg.DualFromScalars(summed, ncols);
    if (const auto g = surfaceDual.find(l); g != surfaceDual.end()) {
      const auto dof = deg.system.SurfaceDof();
      for (auto c = 0; c < ncols; ++c) {
        rhs[static_cast<std::size_t>(c) * deg.ndof + dof] += g->second[c];
      }
    }
    auto y = Effective(scheme, dt, l).Solve(rhs, ncols);
    const auto sy = deg.Strain(y, ncols);
    const double wc[2] = {deg.k2 * (deg.k2 - 2.0), deg.k2};
    if (material != nullptr) {
      // d/d mu of y . MaterialForce: the dual solution's strain with the
      // base solution's under minus the effective share, and with the
      // base decayed memory under the branch fractions
      const auto sx = deg.Strain(base.solutions.at(l), ncols);
      const auto& mem = base.decayed.at(l);
      for (auto e = 0; e < deg.ne; ++e) {
        for (auto q = 0; q < deg.np; ++q) {
          const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
          const auto point = static_cast<std::size_t>(e) * deg.np + q;
          auto share = 1.0;
          for (auto k = std::size_t{0}; k < K; ++k) {
            share -= fractions[k][full] *
                     (implicit ? 1.0 - weights[k].e[full] : weights[k].beta[full]);
          }
          auto sum = 0.0;
          for (auto s = 0; s < 2; ++s) {
            for (auto c = 0; c < ncols; ++c) {
              const auto sn = s * block + point * ncols + c;
              auto term = -share * sx[sn];
              for (auto k = std::size_t{0}; k < K; ++k) {
                term += fractions[k][full] *
                        mem[(k * 2 + s) * block + point * ncols + c];
              }
              sum += wc[s] * sy[sn] * term;
            }
          }
          (*material->mu)[full] += deg.Wq[point] * sum;
        }
      }
    }
    std::ranges::fill(dd, 0.0);
    for (auto k = std::size_t{0}; k < K; ++k) {
      const auto& w = weights[k];
      for (auto s = 0; s < 2; ++s) {
        for (auto e = 0; e < deg.ne; ++e) {
          for (auto q = 0; q < deg.np; ++q) {
            const auto full = static_cast<std::size_t>(deg.e0 + e) * deg.np + q;
            const auto point = static_cast<std::size_t>(e) * deg.np + q;
            const auto factor = wc[s] * deg.mu[k][point] * deg.Wq[point];
            const auto node = point * ncols;
            const auto at = (k * 2 + s) * block + node;
            for (auto c = 0; c < ncols; ++c) {
              const auto mdOld = md[at + c];
              const auto mtilde =
                  md[at + c] + factor * sy[s * block + node + c];
              if (pairing != nullptr) {
                // the step's derivative in h = dt itau pairs the duals
                // with the forward trajectory (the spheroidal sweep's
                // expression, on the toroidal scalars)
                const auto& dw = (*dweights)[k];
                const auto sn = s * block + node + c;
                const auto mPrev = pairing->prev->m.at(l)[at + c];
                const auto d1 = pairing->next->strain.at(l)[sn];
                const auto wd1 = factor * d1;
                const double g =
                    implicit
                        ? (mtilde * mPrev - mdOld * d1 - wd1 * sy[sn]) *
                              dw.e[full]
                        : mtilde * (dw.e[full] * mPrev +
                                    dw.alpha[full] *
                                        pairing->prev->strain.at(l)[sn]) +
                              (mdOld * d1 + wd1 * sy[sn]) * dw.beta[full];
                (*pairing->kernel)[k][full] += dt * g;
              }
              md[at + c] = w.e[full] * mtilde;
              if (!implicit) {
                dd[s * block + node + c] += w.alpha[full] * mtilde;
              }
            }
          }
        }
      }
    }
    out.emplace(l, std::move(y));
  }
  return out;
}

ToroidalViscoelastic::Fields ToroidalViscoelastic::StartTransposed(
    const Dual& dual, const Fields& surfaceDual, const Fields* baseStart,
    std::vector<double>* mu) const {
  auto out = Fields{};
  for (const auto& [l, dd] : dual.strain) {
    const auto& deg = Deg(l);
    const auto ncols = Columns(l);
    auto rhs = deg.DualFromScalars(dd, ncols);
    if (const auto g = surfaceDual.find(l); g != surfaceDual.end()) {
      const auto dof = deg.system.SurfaceDof();
      for (auto c = 0; c < ncols; ++c) {
        rhs[static_cast<std::size_t>(c) * deg.ndof + dof] += g->second[c];
      }
    }
    auto y = deg.system.Solve(rhs, ncols);
    if (baseStart != nullptr && mu != nullptr) {
      const auto sy = deg.Strain(y, ncols);
      const auto sx = deg.Strain(baseStart->at(l), ncols);
      const auto block = deg.Block(ncols);
      const double wc[2] = {deg.k2 * (deg.k2 - 2.0), deg.k2};
      for (auto e = 0; e < deg.ne; ++e) {
        for (auto q = 0; q < deg.np; ++q) {
          const auto point = static_cast<std::size_t>(e) * deg.np + q;
          auto sum = 0.0;
          for (auto s = 0; s < 2; ++s) {
            for (auto c = 0; c < ncols; ++c) {
              const auto sn = s * block + point * ncols + c;
              sum += wc[s] * sy[sn] * sx[sn];
            }
          }
          (*mu)[static_cast<std::size_t>(deg.e0 + e) * deg.np + q] -=
              deg.Wq[point] * sum;
        }
      }
    }
    out.emplace(l, std::move(y));
  }
  return out;
}

}  // namespace sl3dpp
