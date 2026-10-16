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
  /// the test functions' strain operators: the plain transpose of Strain
  /// under the nodal pairing when the coefficients carry no weights, and
  /// the memory force when they carry the branch moduli and quadrature.
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

  /// The nodal weight W of the memory pairing, per branch: the branch
  /// modulus, the quadrature and the scalar weights (k^2 (k^2 - 2), 4/3,
  /// k^2). `field` has shape (3, ne, ngll, ncols), one set shared by
  /// every branch, or (K, 3, ne, ngll, ncols); the result always carries
  /// the branch axis. (MemoryForce keeps its fused loop for the hot
  /// forward path; the algebra is identical.)
  [[nodiscard]] std::vector<double> ApplyWeights(std::span<const double> field,
                                                 int ncols) const {
    const std::size_t block = NodalSize(ncols);
    const bool shared = field.size() == 3 * block;
    const auto wc = detail::ScalarWeights(k2);
    std::vector<double> out(mu.size() * 3 * block);
    for (std::size_t k = 0; k < mu.size(); ++k) {
      for (int s = 0; s < 3; ++s) {
        for (int e = 0; e < ne; ++e) {
          for (int q = 0; q < ngll; ++q) {
            const double factor =
                wc[s] * mu[k][static_cast<std::size_t>(e) * ngll + q] *
                Wq[static_cast<std::size_t>(e) * ngll + q];
            const std::size_t node =
                (static_cast<std::size_t>(e) * ngll + q) * ncols;
            const std::size_t src =
                (shared ? s : k * 3 + s) * block + node;
            const std::size_t dst = (k * 3 + s) * block + node;
            for (int c = 0; c < ncols; ++c) {
              out[dst + c] = factor * field[src + c];
            }
          }
        }
      }
    }
    return out;
  }

  /// The force of branch states mem (K, 3, ne, ngll, ncols).
  [[nodiscard]] std::vector<double> MemoryForce(std::span<const double> mem,
                                                int ncols) const {
    const std::size_t block = NodalSize(ncols);
    const auto wc = detail::ScalarWeights(k2);
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
  /// (ndof x 2): the traction-only and attraction-only pieces of the
  /// unit surface load — the continuous adjoint's jump channels.
  std::map<int, std::vector<double>> splitUnits;
};

}  // namespace sl3dpp
