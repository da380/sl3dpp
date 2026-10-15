#include "sl3dpp/GridCoupling.hpp"

#include <cmath>
#include <format>
#include <stdexcept>

namespace sl3dpp {

GridCoupling::GridCoupling(const Rheology& rheology, int lmax, Options options)
    : rheology_(&rheology),
      lmax_(lmax),
      options_(options),
      grid_(lmax, {.nmax = 2,
                   .oversampling = options.oversampling,
                   .parallel = options.parallel,
                   .matrix_kernel = options.matrix_kernel}) {
  const auto& material = rheology.BaseMaterial();
  const auto& mesh = material.Mesh();
  const auto n = mesh.NGll();
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    auto active = false;
    for (auto i = 0; i < n && !active; ++i) {
      active = rheology.IsActive(static_cast<std::size_t>(e) * n + i);
    }
    if (!active) continue;
    for (auto i = 0; i < n; ++i) {
      nodes_.push_back({e, i, mesh.Radius(e, i)});
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

GridCoupling::TensorField GridCoupling::ZeroTensor() const {
  TensorField field;
  for (auto& component : field.c) {
    component.assign(nodes_.size() * grid_.FieldSize(), 0.0);
  }
  return field;
}

GridCoupling::SpectralTensor GridCoupling::ZeroSpectral() const {
  SpectralTensor spectral;
  for (auto c = 0; c < 5; ++c) {
    spectral.c[c].assign(nodes_.size() * grid_.SpinCoefficientSize(kSpins[c]),
                         0.0);
  }
  return spectral;
}

GridCoupling::TensorField GridCoupling::Synthesise(
    const SpectralTensor& spectral) const {
  auto field = ZeroTensor();
  const auto fieldSize = grid_.FieldSize();
  for (auto c = 0; c < 5; ++c) {
    const auto size = grid_.SpinCoefficientSize(kSpins[c]);
    for (std::size_t node = 0; node < nodes_.size(); ++node) {
      grid_.SynthesiseSpin(
          kSpins[c], std::span(spectral.c[c]).subspan(node * size, size),
          std::span(field.c[c]).subspan(node * fieldSize, fieldSize));
    }
  }
  return field;
}

GridCoupling::SpectralTensor GridCoupling::Analyse(
    const TensorField& field) const {
  auto spectral = ZeroSpectral();
  const auto fieldSize = grid_.FieldSize();
  for (auto c = 0; c < 5; ++c) {
    const auto size = grid_.SpinCoefficientSize(kSpins[c]);
    for (std::size_t node = 0; node < nodes_.size(); ++node) {
      grid_.ExpandSpin(
          kSpins[c], std::span(field.c[c]).subspan(node * fieldSize, fieldSize),
          std::span(spectral.c[c]).subspan(node * size, size));
    }
  }
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
}

void GridCoupling::ScaleByNode(SpectralTensor& spectral,
                               std::span<const double> factor) const {
  if (factor.size() != nodes_.size()) {
    throw std::invalid_argument("one factor per radial node, please");
  }
  for (auto c = 0; c < 5; ++c) {
    const auto size = grid_.SpinCoefficientSize(kSpins[c]);
    for (std::size_t node = 0; node < nodes_.size(); ++node) {
      for (std::size_t k = 0; k < size; ++k) {
        spectral.c[c][node * size + k] *= factor[node];
      }
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
  const auto columns = static_cast<std::size_t>(2 * l + 1);
  if (M.size() != nodes_.size() * columns || R.size() != M.size() ||
      S.size() != M.size()) {
    throw std::invalid_argument(
        std::format("a degree-{} scalar state needs {} nodes x {} columns", l,
                    nodes_.size(), columns));
  }
  const auto first = FirstNodeFor(firstElementOf_.at(l));
  for (auto node = first; node < NodeCount(); ++node) {
    const auto r = nodes_[node].radius;
    for (std::size_t column = 0; column < columns; ++column) {
      const auto at = node * columns + column;
      const auto A = ComponentsFromScalars(l, M[at], R[at], S[at], r);
      const auto m = static_cast<int>(column) - l;
      for (auto c = 0; c < 5; ++c) {
        if (l < std::abs(kSpins[c])) continue;
        const auto size = grid_.SpinCoefficientSize(kSpins[c]);
        spectral.c[c][node * size + SurfaceGrid::SpinIndex(kSpins[c], l, m)] +=
            A[c];
      }
    }
  }
}

void GridCoupling::AddStrainOfSolution(SpectralTensor& spectral, int l,
                                       const DegreeOperator& spheroidal,
                                       std::span<const Complex> x,
                                       const ToroidalOperator* toroidal,
                                       std::span<const Complex> w) const {
  const auto& mesh = rheology_->BaseMaterial().Mesh();
  const auto n = mesh.NGll();
  const auto columns = static_cast<std::size_t>(2 * l + 1);
  const auto ndof = spheroidal.DofCount();
  const auto first = FirstNodeFor(spheroidal.FirstElement());

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
      values.U = gatherS(e, q, Component::U, column);
      values.V = gatherS(e, q, Component::V, column);
      values.W = gatherT(e, q, column);
      for (auto i = 0; i < n; ++i) {
        const auto b = r * mesh.Derivative(q, i) / jac;
        values.rdU += b * gatherS(e, i, Component::U, column);
        values.rdV += b * gatherS(e, i, Component::V, column);
        values.rdW += b * gatherT(e, i, column);
      }
      const auto A = DeviatoricComponents(l, values, r);
      const auto m = static_cast<int>(column) - l;
      for (auto c = 0; c < 5; ++c) {
        if (l < std::abs(kSpins[c])) continue;
        const auto size = grid_.SpinCoefficientSize(kSpins[c]);
        spectral.c[c][node * size + SurfaceGrid::SpinIndex(kSpins[c], l, m)] +=
            A[c];
      }
    }
  }
}

std::map<int, std::vector<GridCoupling::Complex>> GridCoupling::SpheroidalForce(
    const SpectralTensor& weighted,
    const std::function<const DegreeOperator&(int)>& system) const {
  const auto& mesh = rheology_->BaseMaterial().Mesh();
  const auto n = mesh.NGll();
  const auto w = mesh.Weights();
  std::map<int, std::vector<Complex>> out;
  for (auto l = 0; l <= lmax_; ++l) {
    const auto& op = system(l);
    const auto columns = static_cast<std::size_t>(2 * l + 1);
    std::vector<Complex> b(static_cast<std::size_t>(op.DofCount()) * columns,
                           0.0);
    const auto first = FirstNodeFor(op.FirstElement());
    for (auto node = first; node < NodeCount(); ++node) {
      const auto e = nodes_[node].element;
      const auto q = nodes_[node].nodeInElement;
      const auto r = nodes_[node].radius;
      const auto jac = mesh.Jacobian(e);
      for (std::size_t column = 0; column < columns; ++column) {
        const auto m = static_cast<int>(column) - l;
        std::array<Complex, 5> A{};
        for (auto c = 0; c < 5; ++c) {
          if (l < std::abs(kSpins[c])) continue;
          const auto size = grid_.SpinCoefficientSize(kSpins[c]);
          A[c] =
              weighted
                  .c[c][node * size + SurfaceGrid::SpinIndex(kSpins[c], l, m)];
        }
        const auto rows = ForceRowsFromComponents(l, A, r);
        // The pointwise rows at the quadrature node itself.
        const auto iU = op.Dof(mesh.GlobalIndex(e, q), Component::U);
        const auto iV = op.Dof(mesh.GlobalIndex(e, q), Component::V);
        const auto point = jac * w[q] * r;
        if (iU >= 0) b[column * op.DofCount() + iU] += point * rows.u;
        if (iV >= 0) b[column * op.DofCount() + iV] += point * rows.v;
        // The derivative rows against every node of the element.
        for (auto j = 0; j < n; ++j) {
          const auto deriv = w[q] * mesh.Derivative(q, j) * r;
          const auto jU = op.Dof(mesh.GlobalIndex(e, j), Component::U);
          const auto jV = op.Dof(mesh.GlobalIndex(e, j), Component::V);
          if (jU >= 0) b[column * op.DofCount() + jU] += deriv * rows.du;
          if (jV >= 0) b[column * op.DofCount() + jV] += deriv * rows.dv;
        }
      }
    }
    out.emplace(l, std::move(b));
  }
  return out;
}

std::map<int, std::vector<GridCoupling::Complex>> GridCoupling::ToroidalForce(
    const SpectralTensor& weighted,
    const std::function<const ToroidalOperator&(int)>& system) const {
  const auto& mesh = rheology_->BaseMaterial().Mesh();
  const auto n = mesh.NGll();
  const auto w = mesh.Weights();
  std::map<int, std::vector<Complex>> out;
  for (auto l = 2; l <= lmax_; ++l) {
    const auto& op = system(l);
    const auto columns = static_cast<std::size_t>(2 * l + 1);
    std::vector<Complex> b(static_cast<std::size_t>(op.DofCount()) * columns,
                           0.0);
    const auto first = FirstNodeFor(op.FirstElement());
    for (auto node = first; node < NodeCount(); ++node) {
      const auto e = nodes_[node].element;
      const auto q = nodes_[node].nodeInElement;
      const auto r = nodes_[node].radius;
      const auto jac = mesh.Jacobian(e);
      for (std::size_t column = 0; column < columns; ++column) {
        const auto m = static_cast<int>(column) - l;
        std::array<Complex, 5> A{};
        for (auto c = 0; c < 5; ++c) {
          if (l < std::abs(kSpins[c])) continue;
          const auto size = grid_.SpinCoefficientSize(kSpins[c]);
          A[c] =
              weighted
                  .c[c][node * size + SurfaceGrid::SpinIndex(kSpins[c], l, m)];
        }
        const auto rows = ForceRowsFromComponents(l, A, r);
        const auto iW = op.Dof(mesh.GlobalIndex(e, q));
        if (iW >= 0) {
          b[column * op.DofCount() + iW] += jac * w[q] * r * rows.w;
        }
        for (auto j = 0; j < n; ++j) {
          const auto jW = op.Dof(mesh.GlobalIndex(e, j));
          if (jW >= 0) {
            b[column * op.DofCount() + jW] +=
                w[q] * mesh.Derivative(q, j) * r * rows.dw;
          }
        }
      }
    }
    out.emplace(l, std::move(b));
  }
  return out;
}

}  // namespace sl3dpp
