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

double MemoryPoints::RadiusAt(const RadialMesh& mesh, int e, int p) const {
  if (set == MemoryPointSet::Gll) return mesh.Radius(e, p);
  auto r = 0.0;
  for (auto i = 0; i < ngll; ++i) r += Value(p, i) * mesh.Radius(e, i);
  return r;
}

std::vector<double> MemoryPoints::Radii(const RadialMesh& mesh) const {
  auto out = std::vector<double>{};
  out.reserve(static_cast<std::size_t>(mesh.ElementCount()) * count);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto p = 0; p < count; ++p) out.push_back(RadiusAt(mesh, e, p));
  }
  return out;
}

std::vector<double> MemoryPoints::QuadratureWeights(
    const RadialMesh& mesh) const {
  auto out = std::vector<double>{};
  out.reserve(static_cast<std::size_t>(mesh.ElementCount()) * count);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    const auto jac = mesh.Jacobian(e);
    for (auto p = 0; p < count; ++p) out.push_back(weight[p] * jac);
  }
  return out;
}

std::vector<double> MemoryPoints::VolumeWeights(const RadialMesh& mesh) const {
  constexpr auto kFourPi = 4.0 * 3.14159265358979323846;
  auto out = QuadratureWeights(mesh);
  auto i = std::size_t{0};
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto p = 0; p < count; ++p, ++i) {
      const auto r = RadiusAt(mesh, e, p);
      out[i] *= kFourPi * r * r;
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

namespace {

void CheckOnPoints(std::size_t size, std::size_t points) {
  if (size != points) {
    throw std::invalid_argument(std::format(
        "an array of {} entries does not live on the {} points of the mesh",
        size, points));
  }
}

}  // namespace

double VolumePairing(const MemoryPoints& points, const RadialMesh& mesh,
                     std::span<const double> kernel,
                     std::span<const double> direction) {
  const auto weights = points.VolumeWeights(mesh);
  CheckOnPoints(kernel.size(), weights.size());
  CheckOnPoints(direction.size(), weights.size());
  auto total = 0.0;
  for (auto i = std::size_t{0}; i < weights.size(); ++i) {
    total += weights[i] * kernel[i] * direction[i];
  }
  return total;
}

double VolumePairing(const RadialMesh& mesh, std::span<const double> nodalKernel,
                     std::span<const double> nodalDirection) {
  return VolumePairing(MemoryPoints::Of(MemoryPointSet::Gll, mesh), mesh,
                       nodalKernel, nodalDirection);
}

namespace detail {

std::vector<double> KernelSamples(const MemoryPoints& points,
                                  const RadialMesh& mesh,
                                  std::vector<double> gradient) {
  const auto weights = points.VolumeWeights(mesh);
  CheckOnPoints(gradient.size(), weights.size());
  const auto np = static_cast<std::size_t>(points.count);
  for (auto e = std::size_t{0}; e * np < gradient.size(); ++e) {
    auto centre = np;  // the point without volume, if the element has one
    for (auto p = std::size_t{0}; p < np; ++p) {
      const auto i = e * np + p;
      if (weights[i] > 0.0) {
        gradient[i] /= weights[i];
      } else {
        centre = p;
      }
    }
    if (centre == np) continue;
    // the Lagrange polynomial through the element's other points
    auto value = 0.0;
    for (auto p = std::size_t{0}; p < np; ++p) {
      if (p == centre) continue;
      auto basis = 1.0;
      for (auto q = std::size_t{0}; q < np; ++q) {
        if (q == p || q == centre) continue;
        basis *= (points.xi[centre] - points.xi[q]) / (points.xi[p] - points.xi[q]);
      }
      value += basis * gradient[e * np + p];
    }
    gradient[e * np + centre] = value;
  }
  return gradient;
}

std::vector<double> KernelSamples(const RadialMesh& mesh,
                                  std::vector<double> gradient) {
  return KernelSamples(MemoryPoints::Of(MemoryPointSet::Gll, mesh), mesh,
                       std::move(gradient));
}

}  // namespace detail

}  // namespace sl3dpp
