#include "sl3dpp/RadialMesh.hpp"

#include <GaussQuad/All>
#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sl3dpp {

std::expected<RadialMesh, MeshError> RadialMesh::Create(const EarthModel& model,
                                                        Options options) {
  // The SI options cross into the model's non-dimensional numbers here,
  // once; everything below is non-dimensional.
  const auto length = model.ModelScales().Length();
  const auto nd = model.DimensionlessView();
  const auto b = nd.Boundaries();
  const double r0 =
      options.rmin.has_value() ? *options.rmin / length : b.front();
  const double r1 =
      options.rmax.has_value() ? *options.rmax / length : nd.SurfaceRadius();
  if (!(b.front() <= r0 && r0 < r1 && r1 <= b.back())) {
    return std::unexpected(MeshError{std::format(
        "mesh range [{:g}, {:g}] m must lie within the model [{:g}, {:g}] m",
        r0 * length, r1 * length, b.front() * length, b.back() * length)});
  }
  auto breakpoints = std::vector<double>{r0};
  auto spanLayer = std::vector<int>{};
  for (auto i = std::size_t{0}; i < b.size(); ++i) {
    if (b[i] > r0 && b[i] < r1) {
      spanLayer.push_back(static_cast<int>(nd.LayerOf(b[i], Side::Below)));
      breakpoints.push_back(b[i]);
    }
  }
  spanLayer.push_back(static_cast<int>(nd.LayerOf(r1, Side::Below)));
  breakpoints.push_back(r1);
  auto inner = options;
  inner.rmin.reset();
  inner.rmax.reset();
  if (inner.drmax) *inner.drmax /= length;
  return Build(breakpoints, spanLayer, inner);
}

std::expected<RadialMesh, MeshError> RadialMesh::Create(
    std::span<const double> breakpoints, Options options) {
  if (options.rmin || options.rmax) {
    return std::unexpected(MeshError{
        "explicit breakpoints already fix the range: rmin/rmax are for the "
        "model overload"});
  }
  auto spanLayer =
      std::vector<int>(breakpoints.empty() ? 0 : breakpoints.size() - 1);
  std::iota(spanLayer.begin(), spanLayer.end(), 0);
  return Build(breakpoints, spanLayer, options);
}

std::expected<RadialMesh, MeshError> RadialMesh::CreateGraded(
    const EarthModel& model, int lmax, int ngll, double eps) {
  if (lmax < 0) {
    return std::unexpected(MeshError{"lmax must be non-negative"});
  }
  if (!(eps > 0.0 && eps < 1.0)) {
    return std::unexpected(MeshError{"eps must lie in (0, 1)"});
  }
  const auto nd = model.DimensionlessView();
  const auto b = nd.Boundaries();
  const auto a = b.back();
  const auto floor = 0.1 * a / (lmax + 1);
  const auto logEps = std::log(eps);

  auto width = [&](double r) {
    const auto depth = a - r;
    if (depth <= 0.0) return floor;
    return std::max(floor, 0.1 * a * std::log1p(-depth / a) / logEps);
  };

  // Edges from the surface down, every model boundary pinned; the last
  // elements above a boundary shortened or split so none is wider than
  // the rule at its top.
  auto edges = std::vector<double>{a};
  for (std::size_t s = b.size() - 1; s-- > 0;) {
    const auto lo = b[s];
    auto r = edges.back();
    while (true) {
      const auto w = width(r);
      if (r - lo <= 2.0 * w) {
        const auto n = r - lo <= w ? 1 : 2;
        for (auto k = 1; k <= n; ++k) {
          edges.push_back(std::lerp(r, lo, static_cast<double>(k) / n));
        }
        edges.back() = lo;  // exact
        break;
      }
      r -= w;
      edges.push_back(r);
    }
  }
  std::ranges::reverse(edges);

  auto spanLayer = std::vector<int>(edges.size() - 1);
  for (auto e = std::size_t{0}; e + 1 < edges.size(); ++e) {
    spanLayer[e] = static_cast<int>(
        nd.LayerOf(0.5 * (edges[e] + edges[e + 1]), Side::Above));
  }
  Options options;
  options.ngll = ngll;
  options.drmax = std::numeric_limits<double>::infinity();
  return Build(edges, spanLayer, options);
}

std::expected<RadialMesh, MeshError> RadialMesh::Build(
    std::span<const double> breakpoints, std::span<const int> spanLayer,
    const Options& options) {
  if (breakpoints.size() < 2) {
    return std::unexpected(MeshError{"need at least two breakpoints"});
  }
  for (auto i = std::size_t{1}; i < breakpoints.size(); ++i) {
    if (!(breakpoints[i] > breakpoints[i - 1])) {
      return std::unexpected(MeshError{std::format(
          "breakpoints must be strictly increasing; {:g} follows {:g}",
          breakpoints[i], breakpoints[i - 1])});
    }
  }
  if (options.ngll < 2) {
    return std::unexpected(MeshError{
        std::format("ngll must be at least 2, got {}", options.ngll)});
  }
  if (options.drmax.has_value() == options.lmax.has_value()) {
    return std::unexpected(MeshError{"give exactly one of drmax and lmax"});
  }
  double step;
  if (options.drmax) {
    step = *options.drmax;
  } else {
    if (*options.lmax < 0) {
      return std::unexpected(MeshError{"lmax must be non-negative"});
    }
    // planetmodel's sizing rule.
    step = 0.1 * breakpoints.back() / (*options.lmax + 1);
  }
  if (step <= 0.0) {
    return std::unexpected(MeshError{"drmax must be positive"});
  }

  RadialMesh mesh;
  mesh.ngll_ = options.ngll;

  // Each span cut uniformly into elements no wider than step.
  for (auto s = std::size_t{0}; s + 1 < breakpoints.size(); ++s) {
    const auto lo = breakpoints[s];
    const auto hi = breakpoints[s + 1];
    const auto nel = std::max(1, static_cast<int>(std::ceil((hi - lo) / step)));
    for (auto e = 0; e < nel; ++e) {
      // The linspace form, so shared edges agree exactly between elements.
      const auto a = std::lerp(lo, hi, static_cast<double>(e) / nel);
      const double b =
          e + 1 == nel ? hi
                       : std::lerp(lo, hi, static_cast<double>(e + 1) / nel);
      mesh.left_.push_back(a);
      mesh.right_.push_back(b);
      mesh.jac_.push_back(0.5 * (b - a));
      mesh.layer_.push_back(spanLayer[s]);
    }
  }

  // Reference element: GLL rule and the Lagrange derivative matrix
  // D(k, i) = l_i'(xi_k) in the closed form for GLL nodes (Canuto et al.
  // 1988): off-diagonal P_N(xi_k) / (P_N(xi_i) (xi_k - xi_i)), diagonal
  // zero but for D(0,0) = -N(N+1)/4 and D(N,N) = +N(N+1)/4.
  const auto rule =
      GaussQuad::GaussLobattoLegendreQuadrature1D<double>(options.ngll);
  mesh.xi_.assign(rule.Points().begin(), rule.Points().end());
  mesh.weights_.assign(rule.Weights().begin(), rule.Weights().end());
  const int n = options.ngll;
  const int N = n - 1;
  const auto legendre = GaussQuad::LegendrePolynomial<double>();
  auto P = std::vector<double>(n);
  for (auto i = 0; i < n; ++i) P[i] = legendre(N, mesh.xi_[i]);
  mesh.deriv_.assign(static_cast<std::size_t>(n) * n, 0.0);
  for (auto k = 0; k < n; ++k) {
    for (auto i = 0; i < n; ++i) {
      if (k != i) {
        mesh.deriv_[static_cast<std::size_t>(k) * n + i] =
            P[k] / (P[i] * (mesh.xi_[k] - mesh.xi_[i]));
      }
    }
  }
  mesh.deriv_[0] = -0.25 * N * (N + 1);
  mesh.deriv_[static_cast<std::size_t>(n) * n - 1] = 0.25 * N * (N + 1);

  // Node radii per element, endpoints exact so shared nodes agree.
  mesh.radius_.resize(mesh.left_.size() * static_cast<std::size_t>(n));
  for (auto e = std::size_t{0}; e < mesh.left_.size(); ++e) {
    for (auto i = 0; i < n; ++i) {
      mesh.radius_[e * n + i] =
          mesh.left_[e] + (mesh.xi_[i] + 1.0) * mesh.jac_[e];
    }
    mesh.radius_[e * n] = mesh.left_[e];
    mesh.radius_[e * n + N] = mesh.right_[e];
  }

  return mesh;
}

int RadialMesh::ElementOf(double r) const {
  auto it = std::ranges::upper_bound(left_, r);
  const auto i = static_cast<std::ptrdiff_t>(it - left_.begin()) - 1;
  return static_cast<int>(std::clamp<std::ptrdiff_t>(
      i, 0, static_cast<std::ptrdiff_t>(left_.size()) - 1));
}

double RadialMesh::TruncationRadius(int l, TruncationRule rule,
                                    double eps) const {
  const double rmax = right_.back();
  double r = rmax;
  switch (rule) {
    case TruncationRule::Graded:
      r = rmax * std::pow(eps, 1.0 / (l + 1));
      break;
    case TruncationRule::Legacy:
      r = rmax * (1.0 - 20.0 / (l + 0.5));
      break;
  }
  return std::clamp(r, left_.front(), rmax);
}

int RadialMesh::StartElement(int l, TruncationRule rule, double eps) const {
  return ElementOf(TruncationRadius(l, rule, eps));
}

std::vector<int> RadialMesh::BalancedCuts(
    int parts, std::span<const double> elementWeights) const {
  const auto count = ElementCount();
  if (parts < 1) {
    throw std::invalid_argument("a mesh is cut into one part or more");
  }
  if (!elementWeights.empty() &&
      elementWeights.size() != static_cast<std::size_t>(count)) {
    throw std::invalid_argument("the weights must give one per element");
  }
  // The summed weight below each element boundary.
  auto below = std::vector<double>(static_cast<std::size_t>(count) + 1, 0.0);
  for (auto e = 0; e < count; ++e) {
    below[e + 1] =
        below[e] + (elementWeights.empty() ? 1.0 : elementWeights[e]);
  }
  auto out = std::vector<int>{};
  for (auto k = 1; k < parts; ++k) {
    const auto target = below.back() * k / parts;
    auto best = -1;
    for (auto e = out.empty() ? 1 : out.back() + 1; e < count; ++e) {
      if (layer_[e - 1] != layer_[e]) continue;
      if (best < 0 ||
          std::abs(below[e] - target) < std::abs(below[best] - target)) {
        best = e;
      }
    }
    if (best >= 0) out.push_back(best);
  }
  return out;
}

std::vector<double> RadialMesh::Nodal(const EarthModel& model,
                                      const std::string& name, int nu) const {
  const auto nd = model.DimensionlessView();
  auto out = std::vector<double>(radius_.size());
  for (auto e = std::size_t{0}; e < left_.size(); ++e) {
    const auto layer = static_cast<std::size_t>(layer_[e]);
    for (auto i = 0; i < ngll_; ++i) {
      out[e * ngll_ + i] =
          nd.EvaluateInLayer(name, layer, radius_[e * ngll_ + i], nu);
    }
  }
  return out;
}

std::vector<double> RadialMesh::NodalGravity(const EarthModel& model) const {
  const auto nd = model.DimensionlessView();
  auto out = std::vector<double>(radius_.size());
  for (auto i = std::size_t{0}; i < radius_.size(); ++i) {
    out[i] = nd.Gravity(radius_[i]);
  }
  return out;
}

}  // namespace sl3dpp
