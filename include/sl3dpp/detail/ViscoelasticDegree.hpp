#pragma once

/**
 * @file detail/ViscoelasticDegree.hpp
 * @brief The per-degree workspace and the (scheme, dt) flavour of the
 * viscoelastic operator — implementation detail shared by Viscoelastic.cpp
 * and Adjoint.cpp (the adjoint sweeps act on the same strain operators and
 * their transposes). Array layouts are Viscoelastic.hpp's.
 */

#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <numbers>
#include <span>
#include <vector>

#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/MemoryPoints.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace sl3dpp {

namespace detail {
/// The force weights of the scalar order (M, R, S).
[[nodiscard]] inline std::array<double, 3> ScalarWeights(double k2) {
  return {k2 * (k2 - 2.0), 4.0 / 3.0, k2};
}
}  // namespace detail

// -- the per-degree workspace
// ----------------------------------------------------

struct ViscoelasticOperator::Degree {
  int l;
  double k2;
  int e0, ne, ngll, ndof;
  /// The memory points per element (MemoryPoints.hpp): ngll at the GLL
  /// nodes (the default), ngll - 1 at the Gauss points. Every memory and
  /// strain array is (ne, np) per scalar and column; the dof maps and
  /// gathered displacements stay on the nodes, (ne, ngll).
  int np;
  bool gll;
  DegreeOperator system;
  const MemoryPoints* points;
  std::vector<double> B;        // (ne, p, i): r_p l_i'(xi_p) / jac_e
  std::vector<double> Wq;       // (ne, p): w_p jac_e, the memory quadrature
  std::vector<int> Udof, Vdof;  // (ne, i), -1 absent
  std::vector<std::vector<double>> mu, itau;  // per branch, (ne, p)
  std::vector<double> loadVec, tideVec;
  std::vector<double> momentRow;  // degree 0 only

  /// pointMu and pointItau are the branch arrays at the memory points
  /// over the whole mesh, (element, p), as the operator samples them.
  Degree(const Rheology& rheology, int degree,
         const ViscoelasticOperator::Options& options,
         const MemoryPoints& memoryPoints,
         const std::vector<std::vector<double>>& pointMu,
         const std::vector<std::vector<double>>& pointItau)
      : l(degree),
        k2(degree * (degree + 1.0)),
        system(rheology.BaseMaterial(), degree,
               {.eps = options.eps, .truncation = options.truncation}),
        points(&memoryPoints) {
    const auto& mesh = rheology.BaseMaterial().Mesh();
    e0 = system.FirstElement();
    ne = mesh.ElementCount() - e0;
    ngll = mesh.NGll();
    np = memoryPoints.count;
    gll = memoryPoints.set == MemoryPointSet::Gll;
    ndof = system.DofCount();
    B.resize(static_cast<std::size_t>(ne) * np * ngll);
    Wq.resize(static_cast<std::size_t>(ne) * np);
    Udof.resize(static_cast<std::size_t>(ne) * ngll);
    Vdof.resize(static_cast<std::size_t>(ne) * ngll);
    for (auto e = 0; e < ne; ++e) {
      const auto jac = mesh.Jacobian(e0 + e);
      for (auto p = 0; p < np; ++p) {
        Wq[static_cast<std::size_t>(e) * np + p] = memoryPoints.weight[p] * jac;
        auto rp = 0.0;
        if (gll) {
          rp = mesh.Radius(e0 + e, p);
        } else {
          for (auto i = 0; i < ngll; ++i) {
            rp += memoryPoints.Value(p, i) * mesh.Radius(e0 + e, i);
          }
        }
        for (auto i = 0; i < ngll; ++i) {
          B[(static_cast<std::size_t>(e) * np + p) * ngll + i] =
              rp * memoryPoints.Derivative(p, i) / jac;
        }
      }
      for (auto q = 0; q < ngll; ++q) {
        const auto gn = mesh.GlobalIndex(e0 + e, q);
        Udof[static_cast<std::size_t>(e) * ngll + q] =
            system.Dof(gn, Component::U);
        Vdof[static_cast<std::size_t>(e) * ngll + q] =
            system.Dof(gn, Component::V);
      }
    }
    for (auto k = std::size_t{0}; k < rheology.BranchCount(); ++k) {
      auto muk = std::vector<double>(static_cast<std::size_t>(ne) * np);
      auto itauk = std::vector<double>(static_cast<std::size_t>(ne) * np);
      for (auto e = 0; e < ne; ++e) {
        for (auto p = 0; p < np; ++p) {
          const auto full = static_cast<std::size_t>(e0 + e) * np + p;
          muk[static_cast<std::size_t>(e) * np + p] = pointMu[k][full];
          itauk[static_cast<std::size_t>(e) * np + p] = pointItau[k][full];
        }
      }
      mu.push_back(std::move(muk));
      itau.push_back(std::move(itauk));
    }
    loadVec = system.Load();
    tideVec = system.Tide(l == 0 ? 2 : -1);
    if (l == 0) {
      // The inertia moment is a displacement integral: the nodes' own
      // quadrature, whatever the memory points.
      const auto& material = rheology.BaseMaterial();
      const auto rho = material.Rho();
      const auto w = mesh.Weights();
      momentRow.assign(ndof, 0.0);
      const auto root = std::sqrt(4.0 * std::numbers::pi);
      for (auto e = 0; e < ne; ++e) {
        for (auto q = 0; q < ngll; ++q) {
          const auto dof = Udof[static_cast<std::size_t>(e) * ngll + q];
          if (dof < 0) continue;
          const auto r = mesh.Radius(e0 + e, q);
          const auto wq = w[q] * mesh.Jacobian(e0 + e);
          momentRow[dof] += root * wq *
                            rho[static_cast<std::size_t>(e0 + e) * ngll + q] *
                            r * r * r;
        }
      }
    }
  }

  /// The size of one scalar's memory-point array, (ne, np, ncols).
  [[nodiscard]] std::size_t NodalSize(int ncols) const {
    return static_cast<std::size_t>(ne) * np * ncols;
  }

  /// The size of one gathered nodal component, (ne, ngll, ncols).
  [[nodiscard]] std::size_t NodeSize(int ncols) const {
    return static_cast<std::size_t>(ne) * ngll * ncols;
  }

  /// One component of a solution gathered to the nodes, zero where
  /// absent: out(e, q, c).
  void Gather(std::span<const double> x, const std::vector<int>& dofs,
              int ncols, std::span<double> out) const {
    for (auto e = 0; e < ne; ++e) {
      for (auto q = 0; q < ngll; ++q) {
        const auto dof = dofs[static_cast<std::size_t>(e) * ngll + q];
        for (auto c = 0; c < ncols; ++c) {
          out[(static_cast<std::size_t>(e) * ngll + q) * ncols + c] =
              dof >= 0 ? x[static_cast<std::size_t>(c) * ndof + dof] : 0.0;
        }
      }
    }
  }

  /// The driving scalars (3, ne, np, ncols) of a solution array, at the
  /// memory points.
  [[nodiscard]] std::vector<double> Strain(std::span<const double> x,
                                           int ncols) const {
    const auto block = NodalSize(ncols);
    auto d = std::vector<double>(3 * block, 0.0);
    auto U = std::vector<double>(NodeSize(ncols));
    auto V = std::vector<double>(NodeSize(ncols));
    Gather(x, Udof, ncols, U);
    Gather(x, Vdof, ncols, V);
    auto at = [&](std::span<const double> f, int e, int i, int c) {
      return f[(static_cast<std::size_t>(e) * ngll + i) * ncols + c];
    };
    for (auto e = 0; e < ne; ++e) {
      for (auto q = 0; q < np; ++q) {
        for (auto c = 0; c < ncols; ++c) {
          const auto node = (static_cast<std::size_t>(e) * np + q) * ncols + c;
          auto dU = 0.0, dV = 0.0;
          for (auto i = 0; i < ngll; ++i) {
            const auto b = B[(static_cast<std::size_t>(e) * np + q) * ngll + i];
            dU += b * at(U, e, i, c);
            dV += b * at(V, e, i, c);
          }
          // The displacement at the point: the node itself at the GLL
          // points, the element's interpolant at the Gauss points.
          auto Uq = 0.0, Vq = 0.0;
          if (gll) {
            Uq = at(U, e, q, c);
            Vq = at(V, e, q, c);
          } else {
            for (auto i = 0; i < ngll; ++i) {
              Uq += points->Value(q, i) * at(U, e, i, c);
              Vq += points->Value(q, i) * at(V, e, i, c);
            }
          }
          d[block + node] = dU - Uq + 0.5 * k2 * Vq;
          if (l > 0) {
            d[node] = Vq;
            d[2 * block + node] = dV - Vq + Uq;
          }
        }
      }
    }
    return d;
  }

  /// The dual vector pairing memory-point scalar coefficients (cM, cR,
  /// cS) with the test functions' strain operators: the plain transpose
  /// of Strain under the point pairing when the coefficients carry no
  /// weights, and the memory force when they carry the branch moduli and
  /// quadrature.
  [[nodiscard]] std::vector<double> DualFromScalars(
      std::span<const double> scalars, int ncols) const {
    const auto block = NodalSize(ncols);
    const auto cM = scalars.subspan(0, block);
    const auto cR = scalars.subspan(block, block);
    const auto cS = scalars.subspan(2 * block, block);
    auto b = std::vector<double>(static_cast<std::size_t>(ndof) * ncols, 0.0);
    auto at = [&](std::span<const double> f, int e, int p, int c) {
      return f[(static_cast<std::size_t>(e) * np + p) * ncols + c];
    };
    for (auto e = 0; e < ne; ++e) {
      for (auto i = 0; i < ngll; ++i) {
        const auto iU = Udof[static_cast<std::size_t>(e) * ngll + i];
        const auto iV = Vdof[static_cast<std::size_t>(e) * ngll + i];
        for (auto c = 0; c < ncols; ++c) {
          auto bU = 0.0, bV = 0.0;
          if (gll) {
            bU = -at(cR, e, i, c) + at(cS, e, i, c);
            bV = 0.5 * k2 * at(cR, e, i, c) - at(cS, e, i, c) + at(cM, e, i, c);
          } else {
            for (auto p = 0; p < np; ++p) {
              const auto v = points->Value(p, i);
              bU += v * (-at(cR, e, p, c) + at(cS, e, p, c));
              bV += v * (0.5 * k2 * at(cR, e, p, c) - at(cS, e, p, c) +
                         at(cM, e, p, c));
            }
          }
          for (auto q = 0; q < np; ++q) {
            const auto bqi =
                B[(static_cast<std::size_t>(e) * np + q) * ngll + i];
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

  /// The point weight W of the memory pairing, per branch: the branch
  /// modulus, the quadrature and the scalar weights (k^2 (k^2 - 2), 4/3,
  /// k^2). `field` has shape (3, ne, np, ncols), one set shared by every
  /// branch, or (K, 3, ne, np, ncols); the result always carries the
  /// branch axis. (MemoryForce keeps its fused loop for the hot forward
  /// path; the algebra is identical.)
  [[nodiscard]] std::vector<double> ApplyWeights(std::span<const double> field,
                                                 int ncols) const {
    const auto block = NodalSize(ncols);
    const auto shared = field.size() == 3 * block;
    const auto wc = detail::ScalarWeights(k2);
    auto out = std::vector<double>(mu.size() * 3 * block);
    for (auto k = std::size_t{0}; k < mu.size(); ++k) {
      for (auto s = 0; s < 3; ++s) {
        for (auto e = 0; e < ne; ++e) {
          for (auto q = 0; q < np; ++q) {
            const auto factor = wc[s] *
                                mu[k][static_cast<std::size_t>(e) * np + q] *
                                Wq[static_cast<std::size_t>(e) * np + q];
            const auto node = (static_cast<std::size_t>(e) * np + q) * ncols;
            const auto src = (shared ? s : k * 3 + s) * block + node;
            const auto dst = (k * 3 + s) * block + node;
            for (auto c = 0; c < ncols; ++c) {
              out[dst + c] = factor * field[src + c];
            }
          }
        }
      }
    }
    return out;
  }

  /// The force of branch states mem (K, 3, ne, np, ncols).
  [[nodiscard]] std::vector<double> MemoryForce(std::span<const double> mem,
                                                int ncols) const {
    const auto block = NodalSize(ncols);
    const auto wc = detail::ScalarWeights(k2);
    auto weighted = std::vector<double>(3 * block, 0.0);
    for (auto k = std::size_t{0}; k < mu.size(); ++k) {
      for (auto s = 0; s < 3; ++s) {
        for (auto e = 0; e < ne; ++e) {
          for (auto q = 0; q < np; ++q) {
            const auto factor = wc[s] *
                                mu[k][static_cast<std::size_t>(e) * np + q] *
                                Wq[static_cast<std::size_t>(e) * np + q];
            if (factor == 0.0) continue;
            const auto node = (static_cast<std::size_t>(e) * np + q) * ncols;
            const auto src = (k * 3 + s) * block + node;
            for (auto c = 0; c < ncols; ++c) {
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
  /// Away from the GLL points the effective systems are the unrelaxed
  /// material less the relaxation term at the memory points
  /// (MemoryPoints.hpp), whose coefficient sum_k mu_k (1 - w_k) lives
  /// here; at the GLL points `material` is relaxed and this stays empty.
  std::vector<double> relaxationCoefficient;
  Relaxation relaxation;
  std::map<int, DegreeOperator> systems;
  std::map<int, std::vector<double>> units;  // (ndof x 2): load, tide
  /// (ndof x 2): the traction-only and attraction-only pieces of the
  /// unit surface load — the continuous adjoint's jump channels.
  std::map<int, std::vector<double>> splitUnits;
};

}  // namespace sl3dpp
