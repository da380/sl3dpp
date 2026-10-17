#include "sl3dpp/GridCoupling.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

#include "sl3dpp/detail/ParallelFor.hpp"

namespace sl3dpp {

GridCoupling::GridCoupling(const Rheology& rheology, int lmax, Options options)
    : GridCoupling(rheology, lmax,
                   SurfaceGrid(lmax, {.nmax = 2,
                                      .oversampling = options.oversampling,
                                      .parallel = options.parallel,
                                      .matrix_kernel = options.matrix_kernel}),
                   options) {}

GridCoupling::GridCoupling(const Rheology& rheology, int lmax,
                           const SurfaceGrid& grid, Options options)
    : rheology_(&rheology),
      lmax_(lmax),
      options_(options),
      grid_(grid),
      points_(MemoryPoints::Of(options.memory_points,
                               rheology.BaseMaterial().Mesh())) {
  if (grid.LMax() != lmax || grid.NMax() < 2) {
    throw std::invalid_argument(
        std::format("the shared grid is lmax {} with spin coverage {}; the "
                    "coupling needs lmax {} and spins to 2",
                    grid.LMax(), grid.NMax(), lmax));
  }
  const auto& material = rheology.BaseMaterial();
  const auto& mesh = material.Mesh();
  const auto n = mesh.NGll();
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    auto active = false;
    for (auto i = 0; i < n && !active; ++i) {
      active = rheology.IsActive(static_cast<std::size_t>(e) * n + i);
    }
    if (!active) continue;
    for (auto p = 0; p < points_.count; ++p) {
      nodes_.push_back({e, p, points_.RadiusAt(mesh, e, p)});
    }
  }
  if (nodes_.empty()) {
    throw std::invalid_argument(
        "the rheology has no active branch anywhere: nothing for the grid "
        "coupling to carry");
  }
  for (auto l = 0; l <= lmax_; ++l) {
    firstElementOf_[l] =
        l > 0 ? mesh.StartElement(l, options.truncation, options.eps) : 0;
  }
}

double GridCoupling::WeightOfNode(int node) const {
  const auto& mesh = rheology_->BaseMaterial().Mesh();
  return points_.weight[nodes_[node].nodeInElement] *
         mesh.Jacobian(nodes_[node].element);
}

GridCoupling::TensorField GridCoupling::ZeroTensor() const {
  TensorField field;
  for (auto& component : field.c) {
    component.assign(nodes_.size() * grid_.FieldSize(), 0.0);
  }
  field.r.assign(nodes_.size() * grid_.FieldSize(), 0.0);
  return field;
}

GridCoupling::SpectralTensor GridCoupling::ZeroSpectral() const {
  SpectralTensor spectral;
  for (auto c = 0; c < 2; ++c) {
    spectral.c[c].assign(nodes_.size() * grid_.SpinCoefficientSize(kSpins[c]),
                         0.0);
  }
  spectral.r.assign(nodes_.size() * grid_.CoefficientSize(), 0.0);
  return spectral;
}

GridCoupling::TensorField GridCoupling::Synthesise(
    const SpectralTensor& spectral) const {
  auto field = ZeroTensor();
  const auto count = static_cast<int>(nodes_.size());
  for (auto c = 0; c < 2; ++c) {
    grid_.SynthesiseSpinBatch(kSpins[c], spectral.c[c], count, field.c[c]);
  }
  grid_.SynthesiseBatch(spectral.r, count, field.r);
  return field;
}

GridCoupling::SpectralTensor GridCoupling::Analyse(
    const TensorField& field) const {
  auto spectral = ZeroSpectral();
  const auto count = static_cast<int>(nodes_.size());
  for (auto c = 0; c < 2; ++c) {
    grid_.ExpandSpinBatch(kSpins[c], field.c[c], count, spectral.c[c]);
  }
  grid_.ExpandBatch(field.r, count, spectral.r);
  return spectral;
}

void GridCoupling::Multiply(TensorField& field,
                            std::span<const double> weight) const {
  const auto size = nodes_.size() * grid_.FieldSize();
  if (weight.size() != size) {
    throw std::invalid_argument(std::format(
        "the weight field has {} values, the tensor {}", weight.size(), size));
  }
  for (auto& component : field.c) {
    for (std::size_t i = 0; i < size; ++i) component[i] *= weight[i];
  }
  for (std::size_t i = 0; i < size; ++i) field.r[i] *= weight[i];
}

void GridCoupling::ScaleByNode(SpectralTensor& spectral,
                               std::span<const double> factor) const {
  if (factor.size() != nodes_.size()) {
    throw std::invalid_argument("one factor per radial node, please");
  }
  for (auto c = 0; c < 2; ++c) {
    const auto size = grid_.SpinCoefficientSize(kSpins[c]);
    for (std::size_t node = 0; node < nodes_.size(); ++node) {
      for (std::size_t k = 0; k < size; ++k) {
        spectral.c[c][node * size + k] *= factor[node];
      }
    }
  }
  const auto size = grid_.CoefficientSize();
  for (std::size_t node = 0; node < nodes_.size(); ++node) {
    for (std::size_t k = 0; k < size; ++k) {
      spectral.r[node * size + k] *= factor[node];
    }
  }
}

int GridCoupling::FirstNodeFor(int firstElement) const {
  for (std::size_t node = 0; node < nodes_.size(); ++node) {
    if (nodes_[node].element >= firstElement) return static_cast<int>(node);
  }
  return static_cast<int>(nodes_.size());
}

void GridCoupling::AddScalarState(SpectralTensor& spectral, int l,
                                  std::span<const Complex> M,
                                  std::span<const Complex> R,
                                  std::span<const Complex> S) const {
  const auto columns = static_cast<std::size_t>(l + 1);
  if (M.size() != nodes_.size() * columns || R.size() != M.size() ||
      S.size() != M.size()) {
    throw std::invalid_argument(
        std::format("a degree-{} scalar state needs {} nodes x {} columns", l,
                    nodes_.size(), columns));
  }
  const auto first = FirstNodeFor(firstElementOf_.at(l));
  for (auto node = first; node < NodeCount(); ++node) {
    const auto r = nodes_[node].radius;
    for (auto m = 0; m <= l; ++m) {
      const auto at = node * columns + m;
      const auto A = ComponentsFromScalars(l, M[at], R[at], S[at], r);
      Scatter(spectral, node, l, m, A);
    }
  }
}

void GridCoupling::Scatter(SpectralTensor& spectral, int node, int l, int m,
                           std::span<const Complex, 5> A) const {
  // One m >= 0 column of a real field carries its m < 0 twin implicitly:
  // the stored spin blocks are full-m, so the -m entries are written here
  // from the derived components, A^{(-N)}_{l,-m} = (-1)^m conj(A^{(N)}_{lm}).
  const auto sign = (m % 2 == 0) ? 1.0 : -1.0;
  for (auto c = 0; c < 2; ++c) {
    if (l < std::abs(kSpins[c])) continue;
    const auto size = grid_.SpinCoefficientSize(kSpins[c]);
    auto* block = &spectral.c[c][node * size];
    block[SurfaceGrid::SpinIndex(kSpins[c], l, m)] += A[c];
    if (m > 0) {
      block[SurfaceGrid::SpinIndex(kSpins[c], l, -m)] +=
          sign * std::conj(A[4 - c]);
    }
  }
  spectral.r[node * grid_.CoefficientSize() + SurfaceGrid::Index(l, m)] += A[2];
}

std::array<GridCoupling::Complex, 5> GridCoupling::Gather(
    const SpectralTensor& spectral, int node, int l, int m) const {
  // The derived components by A^{(N)}_{lm} = (-1)^m conj(A^{(-N)}_{l,-m}).
  const auto sign = (m % 2 == 0) ? 1.0 : -1.0;
  std::array<Complex, 5> A{};
  const auto real = [&](int mm) {
    return spectral
        .r[node * grid_.CoefficientSize() + SurfaceGrid::Index(l, mm)];
  };
  A[2] = m >= 0 ? real(m) : sign * std::conj(real(-m));
  if (l >= 1) {
    const auto size = grid_.SpinCoefficientSize(-1);
    const auto* block = &spectral.c[1][node * size];
    A[1] = block[SurfaceGrid::SpinIndex(-1, l, m)];
    A[3] = sign * std::conj(block[SurfaceGrid::SpinIndex(-1, l, -m)]);
  }
  if (l >= 2) {
    const auto size = grid_.SpinCoefficientSize(-2);
    const auto* block = &spectral.c[0][node * size];
    A[0] = block[SurfaceGrid::SpinIndex(-2, l, m)];
    A[4] = sign * std::conj(block[SurfaceGrid::SpinIndex(-2, l, -m)]);
  }
  return A;
}

void GridCoupling::AddStrainOfSolution(SpectralTensor& spectral, int l,
                                       const DegreeOperator& spheroidal,
                                       std::span<const Complex> x,
                                       const ToroidalOperator* toroidal,
                                       std::span<const Complex> w) const {
  const auto& mesh = rheology_->BaseMaterial().Mesh();
  const auto n = mesh.NGll();
  const auto columns = static_cast<std::size_t>(l + 1);
  const auto ndof = spheroidal.DofCount();
  const auto first = FirstNodeFor(spheroidal.FirstElement());
  const auto gll = points_.set == MemoryPointSet::Gll;

  auto gatherS = [&](int e, int i, Component component,
                     std::size_t column) -> Complex {
    const auto d = spheroidal.Dof(mesh.GlobalIndex(e, i), component);
    return d >= 0 ? x[column * ndof + d] : Complex{};
  };
  auto gatherT = [&](int e, int i, std::size_t column) -> Complex {
    if (toroidal == nullptr) return {};
    const auto d = toroidal->Dof(mesh.GlobalIndex(e, i));
    return d >= 0 ? w[column * toroidal->DofCount() + d] : Complex{};
  };

  for (auto node = first; node < NodeCount(); ++node) {
    const auto e = nodes_[node].element;
    const auto q = nodes_[node].nodeInElement;
    const auto r = nodes_[node].radius;
    const auto jac = mesh.Jacobian(e);
    for (std::size_t column = 0; column < columns; ++column) {
      DegreeNodeValues values;
      if (gll) {
        values.U = gatherS(e, q, Component::U, column);
        values.V = gatherS(e, q, Component::V, column);
        values.W = gatherT(e, q, column);
      }
      for (auto i = 0; i < n; ++i) {
        const auto U = gatherS(e, i, Component::U, column);
        const auto V = gatherS(e, i, Component::V, column);
        const auto W = gatherT(e, i, column);
        if (!gll) {
          const auto v = points_.Value(q, i);
          values.U += v * U;
          values.V += v * V;
          values.W += v * W;
        }
        const auto b = r * points_.Derivative(q, i) / jac;
        values.rdU += b * U;
        values.rdV += b * V;
        values.rdW += b * W;
      }
      const auto A = DeviatoricComponents(l, values, r);
      Scatter(spectral, node, l, static_cast<int>(column), A);
    }
  }
}

std::map<int, std::vector<GridCoupling::Complex>> GridCoupling::SpheroidalForce(
    const SpectralTensor& weighted,
    const std::function<const DegreeOperator&(int)>& system) const {
  const auto& mesh = rheology_->BaseMaterial().Mesh();
  const auto n = mesh.NGll();
  const auto& w = points_.weight;
  const auto gll = points_.set == MemoryPointSet::Gll;
  // The operators are resolved first (any built lazily, serially), so the
  // degree tasks below only read; each task writes only its own force.
  std::vector<const DegreeOperator*> ops(lmax_ + 1);
  for (auto l = 0; l <= lmax_; ++l) ops[l] = &system(l);
  std::vector<std::vector<Complex>> forces(lmax_ + 1);
  detail::ParallelFor(lmax_ + 1, options_.parallel, [&](int task) {
    const auto l = lmax_ - task;  // the largest degrees first
    const auto& op = *ops[l];
    const auto columns = static_cast<std::size_t>(l + 1);
    std::vector<Complex> b(static_cast<std::size_t>(op.DofCount()) * columns,
                           0.0);
    const auto first = FirstNodeFor(op.FirstElement());
    for (auto node = first; node < NodeCount(); ++node) {
      const auto e = nodes_[node].element;
      const auto q = nodes_[node].nodeInElement;
      const auto r = nodes_[node].radius;
      const auto jac = mesh.Jacobian(e);
      for (std::size_t column = 0; column < columns; ++column) {
        const auto m = static_cast<int>(column);
        const auto A = Gather(weighted, node, l, m);
        const auto rows = ForceRowsFromComponents(l, A, r);
        // The pointwise rows: at the quadrature node itself, or spread
        // over the element's nodes by the Lagrange values at the point.
        const auto point = jac * w[q] * r;
        if (gll) {
          const auto iU = op.Dof(mesh.GlobalIndex(e, q), Component::U);
          const auto iV = op.Dof(mesh.GlobalIndex(e, q), Component::V);
          if (iU >= 0) b[column * op.DofCount() + iU] += point * rows.u;
          if (iV >= 0) b[column * op.DofCount() + iV] += point * rows.v;
        } else {
          for (auto i = 0; i < n; ++i) {
            const auto v = point * points_.Value(q, i);
            const auto iU = op.Dof(mesh.GlobalIndex(e, i), Component::U);
            const auto iV = op.Dof(mesh.GlobalIndex(e, i), Component::V);
            if (iU >= 0) b[column * op.DofCount() + iU] += v * rows.u;
            if (iV >= 0) b[column * op.DofCount() + iV] += v * rows.v;
          }
        }
        // The derivative rows against every node of the element.
        for (auto j = 0; j < n; ++j) {
          const auto deriv = w[q] * points_.Derivative(q, j) * r;
          const auto jU = op.Dof(mesh.GlobalIndex(e, j), Component::U);
          const auto jV = op.Dof(mesh.GlobalIndex(e, j), Component::V);
          if (jU >= 0) b[column * op.DofCount() + jU] += deriv * rows.du;
          if (jV >= 0) b[column * op.DofCount() + jV] += deriv * rows.dv;
        }
      }
    }
    forces[l] = std::move(b);
  });
  std::map<int, std::vector<Complex>> out;
  for (auto l = 0; l <= lmax_; ++l) out.emplace(l, std::move(forces[l]));
  return out;
}

std::map<int, std::vector<GridCoupling::Complex>> GridCoupling::ToroidalForce(
    const SpectralTensor& weighted,
    const std::function<const ToroidalOperator&(int)>& system) const {
  const auto& mesh = rheology_->BaseMaterial().Mesh();
  const auto n = mesh.NGll();
  const auto& w = points_.weight;
  const auto gll = points_.set == MemoryPointSet::Gll;
  // As SpheroidalForce: operators resolved first, one task per degree,
  // from degree 1 (whose rigid rotations the operator's gauge fixes).
  std::vector<const ToroidalOperator*> ops(lmax_ + 1, nullptr);
  for (auto l = 1; l <= lmax_; ++l) ops[l] = &system(l);
  std::vector<std::vector<Complex>> forces(lmax_ + 1);
  detail::ParallelFor(lmax_, options_.parallel, [&](int task) {
    const auto l = lmax_ - task;  // the largest degrees first
    const auto& op = *ops[l];
    const auto columns = static_cast<std::size_t>(l + 1);
    std::vector<Complex> b(static_cast<std::size_t>(op.DofCount()) * columns,
                           0.0);
    const auto first = FirstNodeFor(op.FirstElement());
    for (auto node = first; node < NodeCount(); ++node) {
      const auto e = nodes_[node].element;
      const auto q = nodes_[node].nodeInElement;
      const auto r = nodes_[node].radius;
      const auto jac = mesh.Jacobian(e);
      for (std::size_t column = 0; column < columns; ++column) {
        const auto m = static_cast<int>(column);
        const auto A = Gather(weighted, node, l, m);
        const auto rows = ForceRowsFromComponents(l, A, r);
        if (gll) {
          const auto iW = op.Dof(mesh.GlobalIndex(e, q));
          if (iW >= 0) {
            b[column * op.DofCount() + iW] += jac * w[q] * r * rows.w;
          }
        } else {
          for (auto i = 0; i < n; ++i) {
            const auto iW = op.Dof(mesh.GlobalIndex(e, i));
            if (iW >= 0) {
              b[column * op.DofCount() + iW] +=
                  jac * w[q] * r * points_.Value(q, i) * rows.w;
            }
          }
        }
        for (auto j = 0; j < n; ++j) {
          const auto jW = op.Dof(mesh.GlobalIndex(e, j));
          if (jW >= 0) {
            b[column * op.DofCount() + jW] +=
                w[q] * points_.Derivative(q, j) * r * rows.dw;
          }
        }
      }
    }
    forces[l] = std::move(b);
  });
  std::map<int, std::vector<Complex>> out;
  for (auto l = 1; l <= lmax_; ++l) out.emplace(l, std::move(forces[l]));
  return out;
}

}  // namespace sl3dpp
