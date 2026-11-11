#include "sl3dpp/ToroidalViscoelastic.hpp"

#include <algorithm>
#include <format>
#include <span>
#include <stdexcept>

#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp {

/// The per-degree workspace: the unrelaxed system, the memory window and
/// the strain operators of the two toroidal scalars.
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
    auto first = mesh.ElementCount(), last = subset;
    for (auto e = subset; e < mesh.ElementCount(); ++e) {
      for (auto i = 0; i < ngll; ++i) {
        if (rheology.IsActive(static_cast<std::size_t>(e) * ngll + i)) {
          first = std::min(first, e);
          last = e + 1;
          break;
        }
      }
    }
    if (first >= last) {
      first = subset;
      last = subset + 1;
    }
    e0 = first;
    ne = last - first;
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
    state.m[l].assign(K * 2 * Deg(l).Block(2 * (l + 1)), 0.0);
  }
  return state;
}

ToroidalViscoelastic::Fields ToroidalViscoelastic::SolveElastic(
    const State& state, const Fields& force) const {
  auto out = Fields{};
  for (const auto& [l, m] : state.m) {
    const auto& deg = Deg(l);
    const auto ncols = 2 * (l + 1);
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
    state.strain.emplace(l, Deg(l).Strain(x, 2 * (l + 1)));
  }
}

std::pair<ToroidalViscoelastic::State, ToroidalViscoelastic::Fields>
ToroidalViscoelastic::Step(const State& state, double dt, Scheme scheme,
                           const Fields& force) const {
  if (scheme == Scheme::ExponentialEuler) {
    throw std::invalid_argument(
        "the toroidal step is implicit: ExponentialTrapezoid or "
        "BackwardEuler");
  }
  if (dt <= 0.0) throw std::invalid_argument("dt must be positive");
  const auto& op = *op_;
  const bool implicit = scheme == Scheme::BackwardEuler;
  const auto& weights = op.SchemeWeightsOf(scheme, dt);
  auto& flavour = op.FlavourOf(scheme, dt);
  const auto K = op.GetRheology().BranchCount();
  State next;
  next.t = state.t + dt;
  auto solutions = Fields{};
  for (const auto& [l, m] : state.m) {
    const auto& deg = Deg(l);
    const auto ncols = 2 * (l + 1);
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
    // the effective system, kept with the operator's flavour
    auto system = flavour.toroidal.find(l);
    if (system == flavour.toroidal.end()) {
      system = flavour.toroidal
                   .emplace(l, ToroidalOperator(
                                   *flavour.material, l,
                                   {.eps = op.options_.eps,
                                    .truncation = op.options_.truncation,
                                    .point_shear =
                                        flavour.pointShear.points != nullptr
                                            ? &flavour.pointShear
                                            : nullptr}))
                   .first;
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
    auto x = system->second.Solve(b, ncols);
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

ToroidalViscoelastic::Fields ToroidalViscoelastic::Surface(
    const Fields& solutions) const {
  auto out = Fields{};
  for (const auto& [l, x] : solutions) {
    const auto& deg = Deg(l);
    const auto ncols = 2 * (l + 1);
    const auto dof = deg.system.SurfaceDof();
    auto values = std::vector<double>(ncols, 0.0);
    for (auto c = 0; c < ncols; ++c) {
      values[c] = x[static_cast<std::size_t>(c) * deg.ndof + dof];
    }
    out.emplace(l, std::move(values));
  }
  return out;
}

}  // namespace sl3dpp
