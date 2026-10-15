#include "sl3dpp/LoveNumbers.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
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
  const int n = mesh.NGll();
  const auto rho = material.Rho();
  const auto w = mesh.Weights();
  double sum = 0.0;
  for (int e = 0; e < mesh.ElementCount(); ++e) {
    for (int q = 0; q < n; ++q) {
      const std::size_t i = static_cast<std::size_t>(e) * n + q;
      const double r = mesh.Radius(e, q);
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
  double worst = 0.0;
  const double g = surface_gravity;
  for (std::size_t i = 0; i < degrees.size(); ++i) {
    const double k2 = degrees[i] * (degrees[i] + 1.0);
    const double pairs[3][2] = {
        {g * h_phi[i], k_u[i]},
        {h_v[i], k2 * l_u[i]},
        {k_v[i], g * k2 * l_phi[i]},
    };
    double scale = 0.0;
    for (const auto& p : pairs) {
      scale = std::max({scale, std::abs(p[0]), std::abs(p[1])});
    }
    if (scale == 0.0) continue;
    for (const auto& p : pairs) {
      worst = std::max(worst, std::abs(p[0] - p[1]) / scale);
    }
  }
  const double root = std::sqrt(4.0 * std::numbers::pi);
  const double a4 = std::pow(radius, 4);
  const double axial[2][2] = {
      {m_u, root * g * a4 * h_c / 2.0},
      {m_phi, root * a4 * k_c / 2.0},
  };
  double scale = 0.0;
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
  for (int l = 0; l <= lmax; ++l) out.degrees[l] = l;
  for (auto* v :
       {&out.h_u, &out.l_u, &out.k_u, &out.h_phi, &out.l_phi, &out.k_phi,
        &out.h_v, &out.l_v, &out.k_v, &out.h_t, &out.l_t, &out.k_t}) {
    v->assign(lmax + 1, 0.0);
  }
  out.radius = material.Radius() * scales.Length();
  out.surface_gravity =
      material.SurfaceGravity() * scales.Factor(AccelerationDimensions);

  const double hLoad = scales.Factor(kHPerLoad);
  const double kLoad = scales.Factor(kKPerLoad);
  const double hPot = scales.Factor(kHPerPotential);
  const double kPot = scales.Factor(kKPerPotential);

#pragma omp parallel for schedule(dynamic)
  for (int l = 0; l <= lmax; ++l) {
    DegreeOperator op(material, l,
                      {.eps = options.eps, .truncation = options.truncation});
    const int n = op.DofCount();
    const int nrhs = l == 0 ? 5 : 4;
    std::vector<double> rhs(static_cast<std::size_t>(n) * nrhs);
    const auto columns = {op.Load(LoadPart::Force),
                          op.Load(LoadPart::Potential), op.Tangential(),
                          op.Tide()};
    int k = 0;
    for (const auto& column : columns) {
      std::ranges::copy(column, rhs.begin() + (k++) * n);
    }
    if (l == 0) {
      std::ranges::copy(op.Tide(2), rhs.begin() + (k++) * n);
    }
    const auto x = op.Solve(rhs, nrhs);

    const int iU = op.SurfaceDof(Component::U);
    const int iV = op.SurfaceDof(Component::V);
    const int iP = op.SurfaceDof(Component::Phi);
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
      const double mLoad = scales.Factor(kMomentPerLoad);
      const double mPot = scales.Factor(kMomentPerPotential);
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
  LoveNumbers out = *this;
  const double hLoad = scales.Factor(kHPerLoad);
  const double kLoad = scales.Factor(kKPerLoad);
  const double hPot = scales.Factor(kHPerPotential);
  const double kPot = scales.Factor(kKPerPotential);
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
  bool inTable = false;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (!inTable) {
      std::string head;
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
    int l = 0;
    row >> l;
    if (!row || l != static_cast<int>(out.degrees.size())) {
      return std::unexpected(ModelError{
          std::format("{}: table degrees must run 0, 1, 2, ... without gaps",
                      path.string())});
    }
    out.degrees.push_back(l);
    for (auto* column : columns) {
      double value = 0;
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
