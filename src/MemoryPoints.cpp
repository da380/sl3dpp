#include "sl3dpp/MemoryPoints.hpp"

#include <GaussQuad/All>
#include <format>
#include <stdexcept>

namespace sl3dpp {

MemoryPoints MemoryPoints::Of(MemoryPointSet set, const RadialMesh& mesh) {
  MemoryPoints out;
  out.set = set;
  out.ngll = mesh.NGll();
  const auto n = out.ngll;
  const auto nodes = mesh.Points();
  if (set == MemoryPointSet::Gll) {
    out.count = n;
    out.xi.assign(nodes.begin(), nodes.end());
    out.weight.assign(mesh.Weights().begin(), mesh.Weights().end());
    out.value.assign(static_cast<std::size_t>(n) * n, 0.0);
    out.derivative.resize(static_cast<std::size_t>(n) * n);
    for (auto p = 0; p < n; ++p) {
      out.value[static_cast<std::size_t>(p) * n + p] = 1.0;
      for (auto i = 0; i < n; ++i) {
        out.derivative[static_cast<std::size_t>(p) * n + i] =
            mesh.Derivative(p, i);
      }
    }
    return out;
  }

  // The N interior Gauss-Legendre points, and the GLL Lagrange basis there:
  // l_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j), and its derivative by
  // the product rule.
  out.count = n - 1;
  const auto rule = GaussQuad::GaussLegendreQuadrature1D<double>(out.count);
  out.xi.assign(rule.Points().begin(), rule.Points().end());
  out.weight.assign(rule.Weights().begin(), rule.Weights().end());
  out.value.resize(static_cast<std::size_t>(out.count) * n);
  out.derivative.resize(static_cast<std::size_t>(out.count) * n);
  for (auto p = 0; p < out.count; ++p) {
    const auto x = out.xi[p];
    for (auto i = 0; i < n; ++i) {
      auto value = 1.0, slope = 0.0;
      for (auto j = 0; j < n; ++j) {
        if (j == i) continue;
        const auto scale = 1.0 / (nodes[i] - nodes[j]);
        slope = slope * (x - nodes[j]) * scale + value * scale;
        value *= (x - nodes[j]) * scale;
      }
      out.value[static_cast<std::size_t>(p) * n + i] = value;
      out.derivative[static_cast<std::size_t>(p) * n + i] = slope;
    }
  }
  return out;
}

std::vector<double> MemoryPoints::Interpolate(
    std::span<const double> nodal) const {
  if (nodal.size() % ngll != 0) {
    throw std::invalid_argument(std::format(
        "{} nodal values are not whole elements of {}", nodal.size(), ngll));
  }
  const auto elements = nodal.size() / ngll;
  auto out = std::vector<double>(elements * count, 0.0);
  for (auto e = std::size_t{0}; e < elements; ++e) {
    for (auto p = 0; p < count; ++p) {
      auto sum = 0.0;
      for (auto i = 0; i < ngll; ++i) sum += Value(p, i) * nodal[e * ngll + i];
      out[e * count + p] = sum;
    }
  }
  return out;
}

std::vector<double> MemoryPoints::InterpolateTranspose(
    std::span<const double> atPoints) const {
  if (atPoints.size() % count != 0) {
    throw std::invalid_argument(
        std::format("{} point values are not whole elements of {}",
                    atPoints.size(), count));
  }
  const auto elements = atPoints.size() / count;
  auto out = std::vector<double>(elements * ngll, 0.0);
  for (auto e = std::size_t{0}; e < elements; ++e) {
    for (auto i = 0; i < ngll; ++i) {
      auto sum = 0.0;
      for (auto p = 0; p < count; ++p)
        sum += Value(p, i) * atPoints[e * count + p];
      out[e * ngll + i] = sum;
    }
  }
  return out;
}

}  // namespace sl3dpp
