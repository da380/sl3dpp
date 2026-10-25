#include "sl3dpp/LoveNumbers.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>

#include "sl3dpp/DegreeOperator.hpp"

namespace sl3dpp {

namespace {

/// sqrt(4 pi) int rho U r^3 dr over the mesh for a nodal degree-0 radial
/// displacement: int rho u . x dV for u = U Y_00 r_hat.
double InertiaMoment(const Material& material, std::span<const double> U) {
  const auto& mesh = material.Mesh();
  const auto n = mesh.NGll();
  const auto rho = material.Rho();
  const auto w = mesh.Weights();
  auto sum = 0.0;
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < n; ++q) {
      const auto i = static_cast<std::size_t>(e) * n + q;
      const auto r = mesh.Radius(e, q);
      sum += w[q] * mesh.Jacobian(e) * rho[i] * U[i] * r * r * r;
    }
  }
  return std::sqrt(4.0 * std::numbers::pi) * sum;
}

// SI dimensions of the outputs: displacement or potential per unit surface
// density (channels u, phi, v) or per unit potential (channel t and the
// axial pair), and the inertia moments likewise.
constexpr Dimensions kHPerLoad{.mass = -1, .length = 3};
constexpr Dimensions kKPerLoad{.mass = -1, .length = 4, .time = -2};
constexpr Dimensions kHPerPotential{.length = -1, .time = 2};
constexpr Dimensions kKPerPotential{};
constexpr Dimensions kMomentPerLoad{.length = 4};
constexpr Dimensions kMomentPerPotential{.mass = 1, .time = 2};

}  // namespace

double LoveNumbers::ReciprocityResidual() const {
  auto worst = 0.0;
  const double g = surface_gravity;
  for (auto i = std::size_t{0}; i < degrees.size(); ++i) {
    const double k2 = degrees[i] * (degrees[i] + 1.0);
    const double pairs[3][2] = {
        {g * h_phi[i], k_u[i]},
        {h_v[i], k2 * l_u[i]},
        {k_v[i], g * k2 * l_phi[i]},
    };
    auto scale = 0.0;
    for (const auto& p : pairs) {
      scale = std::max({scale, std::abs(p[0]), std::abs(p[1])});
    }
    if (scale == 0.0) continue;
    for (const auto& p : pairs) {
      worst = std::max(worst, std::abs(p[0] - p[1]) / scale);
    }
  }
  const auto root = std::sqrt(4.0 * std::numbers::pi);
  const double a4 = std::pow(radius, 4);
  const double axial[2][2] = {
      {m_u, root * g * a4 * h_c / 2.0},
      {m_phi, root * a4 * k_c / 2.0},
  };
  auto scale = 0.0;
  for (const auto& p : axial) {
    scale = std::max({scale, std::abs(p[0]), std::abs(p[1])});
  }
  if (scale > 0.0) {
    for (const auto& p : axial) {
      worst = std::max(worst, std::abs(p[0] - p[1]) / scale);
    }
  }
  return worst;
}

LoveNumbers ComputeLoveNumbers(const Material& material, int lmax,
                               LoveNumberOptions options) {
  if (lmax < 0) {
    throw std::invalid_argument("lmax must be non-negative");
  }
  const auto& scales = material.ModelScales();
  LoveNumbers out;
  out.degrees.resize(lmax + 1);
  for (auto l = 0; l <= lmax; ++l) out.degrees[l] = l;
  for (auto* v :
       {&out.h_u, &out.l_u, &out.k_u, &out.h_phi, &out.l_phi, &out.k_phi,
        &out.h_v, &out.l_v, &out.k_v, &out.h_t, &out.l_t, &out.k_t}) {
    v->assign(lmax + 1, 0.0);
  }
  out.radius = material.Radius() * scales.Length();
  out.surface_gravity =
      material.SurfaceGravity() * scales.Factor(AccelerationDimensions);
  out.gravitational_constant =
      material.G() * scales.Factor(GravitationalConstantDimensions);

  const auto hLoad = scales.Factor(kHPerLoad);
  const auto kLoad = scales.Factor(kKPerLoad);
  const auto hPot = scales.Factor(kHPerPotential);
  const auto kPot = scales.Factor(kKPerPotential);

#pragma omp parallel for schedule(dynamic)
  for (auto l = 0; l <= lmax; ++l) {
    DegreeOperator op(material, l,
                      {.eps = options.eps,
                       .truncation = options.truncation,
                       .point_shear = options.point_shear});
    const auto n = op.DofCount();
    const auto nrhs = l == 0 ? 5 : 4;
    auto rhs = std::vector<double>(static_cast<std::size_t>(n) * nrhs);
    const auto columns = {op.Load(LoadPart::Force),
                          op.Load(LoadPart::Potential), op.Tangential(),
                          op.Tide()};
    auto k = 0;
    for (const auto& column : columns) {
      std::ranges::copy(column, rhs.begin() + (k++) * n);
    }
    if (l == 0) {
      std::ranges::copy(op.Tide(2), rhs.begin() + (k++) * n);
    }
    const auto x = op.Solve(rhs, nrhs);

    const auto iU = op.SurfaceDof(Component::U);
    const auto iV = op.SurfaceDof(Component::V);
    const auto iP = op.SurfaceDof(Component::Phi);
    auto value = [&](int dof, int column) {
      return dof >= 0 ? x[static_cast<std::size_t>(column) * n + dof] : 0.0;
    };
    struct Channel {
      int column;
      std::vector<double>* h;
      std::vector<double>* l;
      std::vector<double>* k;
      double hFactor;
      double kFactor;
    };
    const Channel channels[4] = {
        {0, &out.h_u, &out.l_u, &out.k_u, hLoad, kLoad},
        {1, &out.h_phi, &out.l_phi, &out.k_phi, hLoad, kLoad},
        {2, &out.h_v, &out.l_v, &out.k_v, hLoad, kLoad},
        {3, &out.h_t, &out.l_t, &out.k_t, hPot, kPot},
    };
    for (const auto& c : channels) {
      (*c.h)[l] = value(iU, c.column) * c.hFactor;
      (*c.l)[l] = value(iV, c.column) * c.hFactor;
      (*c.k)[l] = value(iP, c.column) * c.kFactor;
    }

    if (l == 0) {
      out.h_c = value(iU, 4) * hPot;
      out.k_c = value(iP, 4) * kPot;
      const auto mLoad = scales.Factor(kMomentPerLoad);
      const auto mPot = scales.Factor(kMomentPerPotential);
      for (const auto& [name, column, factor] :
           {std::tuple{&out.m_u, 0, mLoad}, std::tuple{&out.m_phi, 1, mLoad},
            std::tuple{&out.m_c, 4, mPot}}) {
        const auto nodal =
            op.Expand({x.data() + static_cast<std::size_t>(column) * n,
                       static_cast<std::size_t>(n)});
        *name = InertiaMoment(material, nodal.U) * factor;
      }
    }
  }
  return out;
}

LoveNumbers LoveNumbers::ConvertedTo(const Scales& scales) const {
  auto out = *this;
  const auto hLoad = scales.Factor(kHPerLoad);
  const auto kLoad = scales.Factor(kKPerLoad);
  const auto hPot = scales.Factor(kHPerPotential);
  const auto kPot = scales.Factor(kKPerPotential);
  auto divide = [](std::vector<double>& v, double f) {
    for (auto& x : v) x /= f;
  };
  for (auto* v :
       {&out.h_u, &out.l_u, &out.h_phi, &out.l_phi, &out.h_v, &out.l_v}) {
    divide(*v, hLoad);
  }
  for (auto* v : {&out.k_u, &out.k_phi, &out.k_v}) divide(*v, kLoad);
  divide(out.h_t, hPot);
  divide(out.l_t, hPot);
  divide(out.k_t, kPot);
  out.h_c /= hPot;
  out.k_c /= kPot;
  out.m_u /= scales.Factor(kMomentPerLoad);
  out.m_phi /= scales.Factor(kMomentPerLoad);
  out.m_c /= scales.Factor(kMomentPerPotential);
  out.radius /= scales.Length();
  out.surface_gravity /= scales.Factor(AccelerationDimensions);
  out.gravitational_constant /= scales.Factor(GravitationalConstantDimensions);
  return out;
}

ConventionalLoveNumbers LoveNumbers::Conventional() const {
  constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();
  const auto fourPiGa =
      4.0 * std::numbers::pi * gravitational_constant * radius;
  const auto g = surface_gravity;
  auto out = ConventionalLoveNumbers{};
  out.degrees = degrees;
  for (auto i = std::size_t{0}; i < degrees.size(); ++i) {
    const auto l = degrees[i];
    const auto f = (2.0 * l + 1.0) / fourPiGa;
    out.h_load.push_back(f * g * (h_u[i] + h_phi[i]));
    out.l_load.push_back(f * g * (l_u[i] + l_phi[i]));
    out.k_load.push_back(-f * (k_u[i] + k_phi[i]) - 1.0);
    out.h_tide.push_back(l >= 2 ? -g * h_t[i] : kNaN);
    out.l_tide.push_back(l >= 2 ? -g * l_t[i] : kNaN);
    out.k_tide.push_back(l >= 2 ? k_t[i] : kNaN);
  }
  return out;
}

std::expected<LoveNumbers, ModelError> LoveNumbers::FromFile(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    return std::unexpected(
        ModelError{std::format("cannot open {}", path.string())});
  }
  LoveNumbers out;
  std::vector<double>* columns[12] = {
      &out.h_u, &out.l_u, &out.k_u, &out.h_phi, &out.l_phi, &out.k_phi,
      &out.h_v, &out.l_v, &out.k_v, &out.h_t,   &out.l_t,   &out.k_t};
  auto inTable = false;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (!inTable) {
      auto head = std::string{};
      row >> head;
      if (head == "radius") {
        row >> out.radius;
      } else if (head == "surface_gravity") {
        row >> out.surface_gravity;
      } else if (head == "axial") {
        row >> out.h_c >> out.k_c >> out.m_u >> out.m_phi >> out.m_c;
      } else if (head == "table") {
        inTable = true;
      }
      continue;
    }
    auto l = 0;
    row >> l;
    if (!row || l != static_cast<int>(out.degrees.size())) {
      return std::unexpected(ModelError{
          std::format("{}: table degrees must run 0, 1, 2, ... without gaps",
                      path.string())});
    }
    out.degrees.push_back(l);
    for (auto* column : columns) {
      auto value = 0.0;
      row >> value;
      column->push_back(value);
    }
    if (!row) {
      return std::unexpected(ModelError{
          std::format("{}: a table row needs l and 12 values", path.string())});
    }
  }
  if (out.degrees.empty() || out.radius <= 0 || out.surface_gravity <= 0) {
    return std::unexpected(ModelError{std::format(
        "{}: not a Love-number table (needs radius, surface_gravity and a "
        "table)",
        path.string())});
  }
  return out;
}

double TractionNumbers::ReciprocityResidual() const {
  auto worst = 0.0;
  for (auto i = std::size_t{0}; i < degrees.size(); ++i) {
    if (std::isnan(h_v[i]) || std::isnan(l_u[i])) continue;
    const double k2 = degrees[i] * (degrees[i] + 1.0);
    const auto scale = std::max(std::abs(h_v[i]), std::abs(k2 * l_u[i]));
    if (scale == 0.0) continue;
    worst = std::max(worst, std::abs(h_v[i] - k2 * l_u[i]) / scale);
  }
  return worst;
}

TractionNumbers ComputeTractionNumbers(const Material& material, int lmax,
                                       bool gravitating,
                                       LoveNumberOptions options) {
  if (lmax < 0) {
    throw std::invalid_argument("lmax must be non-negative");
  }
  // Displacement per unit traction, SI.
  constexpr Dimensions kHPerTraction{.mass = -1, .length = 2, .time = 2};
  const auto& scales = material.ModelScales();
  const auto hFactor = scales.Factor(kHPerTraction);

  TractionNumbers out;
  out.gravitating = gravitating;
  out.radius = material.Radius() * scales.Length();
  out.degrees.resize(lmax + 1);
  for (auto l = 0; l <= lmax; ++l) out.degrees[l] = l;
  constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();
  for (auto* v : {&out.h_u, &out.l_u, &out.h_v, &out.l_v}) {
    v->assign(lmax + 1, kNaN);
  }

#pragma omp parallel for schedule(dynamic)
  for (auto l = 0; l <= lmax; ++l) {
    if (l == 1 && !gravitating) continue;
    DegreeOperator op(material, l,
                      {.eps = options.eps,
                       .truncation = options.truncation,
                       .point_shear = options.point_shear,
                       .gravitating = gravitating});
    const auto n = op.DofCount();
    auto rhs = std::vector<double>(static_cast<std::size_t>(n) * 2);
    std::ranges::copy(op.Traction(), rhs.begin());
    std::ranges::copy(op.TractionTangential(), rhs.begin() + n);
    const auto x = op.Solve(rhs, 2);
    const auto iU = op.SurfaceDof(Component::U);
    const auto iV = op.SurfaceDof(Component::V);
    auto value = [&](int dof, int column) {
      return dof >= 0 ? x[static_cast<std::size_t>(column) * n + dof] : 0.0;
    };
    out.h_u[l] = value(iU, 0) * hFactor;
    out.l_u[l] = value(iV, 0) * hFactor;
    out.h_v[l] = value(iU, 1) * hFactor;
    out.l_v[l] = value(iV, 1) * hFactor;
  }
  return out;
}

double FluidTidalNumber(const Material& material, int l,
                        LoveNumberOptions options) {
  if (l < 2) {
    throw std::invalid_argument(
        "the fluid tidal number is defined for degree >= 2");
  }
  auto [kappa, mu] = material.Moduli().KappaMu();
  std::ranges::fill(mu, 0.0);
  const auto fluid =
      material.WithModuli(NodalModuli::Isotropic(kappa, mu)).value();
  DegreeOperator op(fluid, l,
                    {.eps = options.eps, .truncation = options.truncation});
  const auto x = op.Solve(op.Tide());
  return x[op.SurfaceDof(Component::Phi)];  // k_t is dimensionless
}

}  // namespace sl3dpp
