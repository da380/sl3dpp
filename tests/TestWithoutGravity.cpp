// The problem without gravity through time: the radial evolution on a
// non-gravitating operator against the closed form of the homogeneous
// Maxwell sphere under a point source switched on at t = 0 and under a
// surface load (its weight, a normal traction) held from t = 0 or ramped
// up; a static load against the non-gravitating traction numbers times its
// weight; a body with a fluid core against its shell above the core; a
// resumed run and the factored sources against the general run; loads and
// sources together against the sum of their runs; and what a run without
// gravity refuses, a load's degree-1 part among it.
//
// The oracle. By the correspondence principle the Laplace transform of the
// response to a step source is the elastic closed form
// (PointSource.AHomogeneousSphereWithoutGravityInClosedForm: Kelvin's
// full-space solution and the regular solutions of the sphere cancelling
// its surface traction, degree 1 in the frame of the centre of mass) with
// the shear modulus replaced by the Maxwell modulus mu s / (s + 1 / tau),
// the bulk modulus elastic, divided by s. Kelvin's field is
// p F1 + q F2 with p = 1 / mu and q = 1 / (lambda + 2 mu) and F1, F2 fixed
// fields, so each coefficient of the surface displacement is a rational
// function of s evaluated from a handful of fixed coefficients; it is
// inverted numerically on two contours, Talbot's (Abate and Valko's fixed
// Talbot) and Weideman and Trefethen's hyperbola, in long double. The
// oracle is checked against itself: the two contours and two node counts
// on each agree, the limit t -> 0+ is the elastic closed form, and the
// toroidal response is the elastic one times mu / mu(s), W_el (1 + t /
// tau). Under a held deviatoric source a Maxwell body without gravity
// flows without bound, so there is no relaxed limit to check against.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <numbers>
#include <string>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

namespace {

using Complex = std::complex<double>;
using Wide = std::complex<long double>;
using Vector = std::array<double, 3>;

constexpr auto kLmax = 16;  // compared to here
constexpr auto kGrid = 48;  // the closed form's surface transforms

struct Body {
  EarthModel model;
  Material material;

  static Body Make() {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "e1_homogeneous.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
    auto material = Material::Create(model, mesh).value();
    return Body{std::move(model), std::move(material)};
  }
};

struct Basis {
  Vector r, t, p;
  Basis(double theta, double phi)
      : r{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
          std::cos(theta)},
        t{std::cos(theta) * std::cos(phi), std::cos(theta) * std::sin(phi),
          -std::sin(theta)},
        p{-std::sin(phi), std::cos(phi), 0.0} {}
};

double Dot(const Vector& a, const Vector& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// The fixed coefficients of one (l, m) from which the closed form is
/// evaluated at any moduli: the surface U, V, W of F1 and F2, the radial
/// coefficient of (div F) r_hat and the U, V, W of (grad F + grad F^T)
/// r_hat for each, and at degree 1 the radial moments int (U + 2 V) r^2 dr
/// of F1 and F2 over the ball. Real (or imaginary) parts alone: the
/// closed form is linear in them.
struct Data {
  std::array<double, 3> u1{}, u2{}, s1{}, s2{};
  double d1 = 0.0, d2 = 0.0, k1 = 0.0, k2 = 0.0;
};

/// The regular solutions of the homogeneous sphere without gravity at a
/// degree l >= 2, Lame parameters (lam, mu), that carry the radial and
/// shear tractions -tU and -tV at the surface: their surface U and V. The
/// two are grad(r^l Y) (divergence-free) and the solution of Navier's
/// equation growing as r^(l+1).
std::array<Wide, 2> Regular(int l, long double a, Wide lam, Wide mu, Wide tU,
                            Wide tV) {
  const auto L = static_cast<long double>(l);
  const auto alpha = -2.0L * (L * lam + (3.0L * L + 1.0L) * mu) /
                     ((L + 3.0L) * lam + (L + 5.0L) * mu);
  const auto UA = Wide(L / a), VA = Wide(1.0L / a);
  const auto dUA = Wide(L * (L - 1.0L) / (a * a)),
             dVA = Wide((L - 1.0L) / (a * a));
  const auto UB = (L + alpha) * a, VB = Wide(a);
  const auto dUB = (L + alpha) * (L + 1.0L), dVB = Wide(L + 1.0L);
  auto radial = [&](Wide Us, Wide Vs, Wide dU) {
    return (lam + 2.0L * mu) * dU + lam * (2.0L * Us - L * (L + 1.0L) * Vs) / a;
  };
  auto shear = [&](Wide Us, Wide Vs, Wide dV) {
    return mu * (dV - Vs / a + Us / a);
  };
  const auto a11 = radial(UA, VA, dUA), a12 = radial(UB, VB, dUB);
  const auto a21 = shear(UA, VA, dVA), a22 = shear(UB, VB, dVB);
  const auto det = a11 * a22 - a12 * a21;
  const auto cA = (-tU * a22 + tV * a12) / det;
  const auto cB = (-tV * a11 + tU * a21) / det;
  return {cA * UA + cB * UB, cA * VA + cB * VB};
}

/// The closed form of the sphere at moduli (kappa, mu), its surface U, V
/// and W (W at degree 1 a rigid rotation's gauge, left at Kelvin's).
std::array<Wide, 3> ClosedForm(int l, const Data& d, long double a,
                               long double kappa, Wide mu) {
  const auto lam = kappa - 2.0L * mu / 3.0L;
  const auto p = 1.0L / mu;
  const auto q = 1.0L / (kappa + 4.0L * mu / 3.0L);
  auto K = std::array<Wide, 3>{};
  for (auto c = 0; c < 3; ++c) {
    K[c] = p * static_cast<long double>(d.u1[c]) +
           q * static_cast<long double>(d.u2[c]);
  }
  const auto tU = lam * (p * static_cast<long double>(d.d1) +
                         q * static_cast<long double>(d.d2)) +
                  mu * (p * static_cast<long double>(d.s1[0]) +
                        q * static_cast<long double>(d.s2[0]));
  const auto tV = mu * (p * static_cast<long double>(d.s1[1]) +
                        q * static_cast<long double>(d.s2[1]));
  const auto tW = mu * (p * static_cast<long double>(d.s1[2]) +
                        q * static_cast<long double>(d.s2[2]));
  const auto L = static_cast<long double>(l);
  if (l == 0) {
    return {K[0] - tU / (3.0L * lam + 2.0L * mu) * a, Wide{}, Wide{}};
  }
  if (l == 1) {
    const auto alpha = -2.0L * (lam + 4.0L * mu) / (4.0L * lam + 6.0L * mu);
    const auto UB = (1.0L + alpha) * a, VB = Wide(a);
    const auto dUB = 2.0L * (1.0L + alpha), dVB = Wide(2.0L);
    const auto radialB =
        (lam + 2.0L * mu) * dUB + lam * (2.0L * UB - 2.0L * VB) / a;
    const auto cB = -tU / radialB;
    const auto moment =
        p * static_cast<long double>(d.k1) + q * static_cast<long double>(d.k2);
    const auto cA =
        -(moment + cB * (3.0L + alpha) * a * a * a * a / 5.0L) / (a * a);
    return {K[0] + cA / a + cB * UB, K[1] + cA / a + cB * VB, K[2]};
  }
  const auto regular = Regular(l, a, lam, mu, tU, tV);
  return {K[0] + regular[0], K[1] + regular[1],
          K[2] - tW / (mu * (L - 1.0L) / a)};
}

/// The homogeneous sphere without gravity under a unit pressure Y_lm
/// pressing on its surface (the radial traction -1, no shear traction) at
/// moduli (kappa, mu): its surface U and V, the regular solutions carrying
/// that traction, and at degree 0 the uniform compression by the bulk
/// modulus alone, U = -a / (3 kappa). Degree 1 has none: a unit pressure
/// there exerts a net force.
std::array<Wide, 2> PressureClosedForm(int l, long double a, long double kappa,
                                       Wide mu) {
  if (l == 0) return {Wide(-a / (3.0L * kappa)), Wide{}};
  const auto lam = kappa - 2.0L * mu / 3.0L;
  return Regular(l, a, lam, mu, Wide(1.0L), Wide{});
}

/// The inverse Laplace transform at t > 0 of F, the transform of a real
/// function, by the fixed Talbot contour with M nodes (Abate and Valko
/// 2004).
long double Talbot(const std::function<Wide(Wide)>& F, long double t, int M) {
  const auto r = 2.0L * M / (5.0L * t);
  auto sum = 0.5L * (F(Wide(r)) * std::exp(r * t)).real();
  for (auto k = 1; k < M; ++k) {
    const auto theta = k * std::numbers::pi_v<long double> / M;
    const auto cot = std::cos(theta) / std::sin(theta);
    const auto s = Wide(r * theta * cot, r * theta);
    const auto sigma = theta + (theta * cot - 1.0L) * cot;
    sum += (std::exp(s * t) * F(s) * Wide(1.0L, sigma)).real();
  }
  return r / M * sum;
}

/// The same on Weideman and Trefethen's (2007) hyperbola
/// z = mu (1 + sin(i theta - alpha)), 2 N + 1 nodes, their parameters.
long double Hyperbola(const std::function<Wide(Wide)>& F, long double t,
                      int N) {
  const auto alpha = 1.1721L, h = 1.0818L / N, scale = 4.4920L * N / t;
  auto sum = 0.0L;
  for (auto k = 0; k <= N; ++k) {
    const auto theta = k * h;
    const auto arg = Wide(-alpha, theta);  // i theta - alpha
    const auto z = scale * (1.0L + std::sin(arg));
    const auto dz = Wide(0.0L, scale) * std::cos(arg);
    const auto term = (std::exp(z * t) * F(z) * dz).imag();
    sum += k == 0 ? 0.5L * term : term;
  }
  return h / std::numbers::pi_v<long double> * sum;
}

/// The closed form through time: per (l, m) the real and imaginary parts'
/// Data, and the material.
struct Oracle {
  long double a = 0.0, kappa = 0.0, mu = 0.0, itau = 0.0;
  std::map<std::pair<int, int>, std::array<Data, 2>> data;

  /// The elastic closed form, coefficient (U, V, W) of (l, m).
  [[nodiscard]] std::array<Complex, 3> Elastic(int l, int m) const {
    const auto& parts = data.at({l, m});
    const auto re = ClosedForm(l, parts[0], a, kappa, Wide(mu));
    const auto im = ClosedForm(l, parts[1], a, kappa, Wide(mu));
    auto out = std::array<Complex, 3>{};
    for (auto c = 0; c < 3; ++c) {
      out[c] = Complex(static_cast<double>(re[c].real()),
                       static_cast<double>(im[c].real()));
    }
    return out;
  }

  /// The response at t > 0 to the source held from t = 0, by one contour
  /// (`hyperbola` or Talbot's) with `nodes` nodes.
  [[nodiscard]] std::array<Complex, 3> At(int l, int m, double t,
                                          bool hyperbola, int nodes) const {
    const auto& parts = data.at({l, m});
    auto out = std::array<Complex, 3>{};
    for (auto c = 0; c < 3; ++c) {
      long double value[2];
      for (auto part = 0; part < 2; ++part) {
        const auto F = [&](Wide s) {
          const auto maxwell = mu * s / (s + itau);
          return ClosedForm(l, parts[part], a, kappa, maxwell)[c] / s;
        };
        value[part] = hyperbola ? Hyperbola(F, t, nodes) : Talbot(F, t, nodes);
      }
      out[c] =
          Complex(static_cast<double>(value[0]), static_cast<double>(value[1]));
    }
    return out;
  }
};

/// The oracle of a point source in the body (material units).
Oracle MakeOracle(const Material& material, const PointSource& source,
                  double itau) {
  const auto a = material.Radius();
  const auto [kappas, mus] = material.Moduli().KappaMu();
  const auto mu = mus.front();
  const auto kappa = kappas.front();
  const Basis at(source.at.colatitude, source.at.longitude);
  const Vector xi{source.at.radius * at.r[0], source.at.radius * at.r[1],
                  source.at.radius * at.r[2]};
  double M[3][3] = {};
  {
    const Vector* e[3] = {&at.r, &at.t, &at.p};
    const auto& S = source.moment;
    const double c[3][3] = {
        {S.rr, S.rt, S.rp}, {S.rt, S.tt, S.tp}, {S.rp, S.tp, S.pp}};
    for (auto p = 0; p < 3; ++p) {
      for (auto q = 0; q < 3; ++q) {
        for (auto i = 0; i < 3; ++i) {
          for (auto j = 0; j < 3; ++j) {
            M[i][j] += c[p][q] * (*e[p])[i] * (*e[q])[j];
          }
        }
      }
    }
  }
  const auto trace = M[0][0] + M[1][1] + M[2][2];
  const auto eightPi = 8.0 * std::numbers::pi;
  // Phi1_i = M_ik R_k / R^3 / 8 pi,
  // Phi2_i = -(M_ij R_j + R_i tr M - 3 R_i (R.M.R) / R^2) / R^3 / 8 pi;
  // F1 = Phi1 + Phi2 and F2 = Phi1 - Phi2, Kelvin's u = F1 / mu +
  // F2 / (lambda + 2 mu)
  auto phis = [&](const Vector& x) {
    const Vector R{x[0] - xi[0], x[1] - xi[1], x[2] - xi[2]};
    const auto r2 = Dot(R, R);
    const auto r3 = r2 * std::sqrt(r2);
    auto MR = Vector{};
    for (auto i = 0; i < 3; ++i) {
      for (auto k = 0; k < 3; ++k) MR[i] += M[i][k] * R[k];
    }
    const auto RMR = Dot(R, MR);
    auto out = std::array<Vector, 2>{};
    for (auto i = 0; i < 3; ++i) {
      const auto phi1 = MR[i] / r3 / eightPi;
      const auto phi2 =
          -(MR[i] + R[i] * trace - 3.0 * R[i] * RMR / r2) / r3 / eightPi;
      out[0][i] = phi1 + phi2;
      out[1][i] = phi1 - phi2;
    }
    return out;
  };
  // the decomposition is Kelvin's tensor's
  {
    const auto lam = kappa - 2.0 * mu / 3.0;
    const auto nu = lam / (2.0 * (lam + mu));
    const auto c = 1.0 / (16.0 * std::numbers::pi * mu * (1.0 - nu));
    const Vector x{0.3 * a, -0.7 * a, 0.6 * a};
    const Vector R{x[0] - xi[0], x[1] - xi[1], x[2] - xi[2]};
    const auto r2 = Dot(R, R);
    const auto r1 = std::sqrt(r2);
    const auto F = phis(x);
    for (auto i = 0; i < 3; ++i) {
      auto u = 0.0;
      for (auto j = 0; j < 3; ++j) {
        for (auto k = 0; k < 3; ++k) {
          const auto dG = c * (-(3.0 - 4.0 * nu) * (i == j) * R[k] / (r2 * r1) +
                               ((i == k) * R[j] + (j == k) * R[i]) / (r2 * r1) -
                               3.0 * R[i] * R[j] * R[k] / (r2 * r2 * r1));
          u -= M[j][k] * dG;
        }
      }
      const auto mine = F[0][i] / mu + F[1][i] / (lam + 2.0 * mu);
      EXPECT_NEAR(mine, u, 1e-13 * std::abs(u));
    }
  }

  const SurfaceGrid grid(kGrid, {.nmax = 1});
  // per field: F1, F2 (displacement), their (div F) n and their
  // (grad F + grad F^T) n
  std::array<SurfaceVectorField, 2> uField, sField;
  std::array<std::vector<double>, 2> dField;
  std::array<std::array<std::vector<double>, 3>, 2> ballIntegrand;
  std::array<std::vector<double>, 3> unit;
  const auto h = 1e-6 * a;
  for (auto theta : grid.CoLatitudes()) {
    for (auto phi : grid.Longitudes()) {
      const Basis here(theta, phi);
      const Vector x{a * here.r[0], a * here.r[1], a * here.r[2]};
      const auto F = phis(x);
      std::array<std::array<Vector, 3>, 2> grad;  // grad[f][i][k]
      for (auto k = 0; k < 3; ++k) {
        auto plus = x, minus = x;
        plus[k] += h;
        minus[k] -= h;
        const auto Fp = phis(plus), Fm = phis(minus);
        for (auto f = 0; f < 2; ++f) {
          for (auto i = 0; i < 3; ++i) {
            grad[f][i][k] = (Fp[f][i] - Fm[f][i]) / (2.0 * h);
          }
        }
      }
      for (auto f = 0; f < 2; ++f) {
        const auto div = grad[f][0][0] + grad[f][1][1] + grad[f][2][2];
        auto s = Vector{};
        for (auto i = 0; i < 3; ++i) {
          for (auto j = 0; j < 3; ++j) {
            s[i] += (grad[f][i][j] + grad[f][j][i]) * here.r[j];
          }
        }
        uField[f].up.push_back(Dot(F[f], here.r));
        uField[f].north.push_back(-Dot(F[f], here.t));
        uField[f].east.push_back(Dot(F[f], here.p));
        sField[f].up.push_back(Dot(s, here.r));
        sField[f].north.push_back(-Dot(s, here.t));
        sField[f].east.push_back(Dot(s, here.p));
        dField[f].push_back(div);
      }
      // the ball integrals by the divergence theorem: int Phi1_i dV =
      // -M_ik oint n_k / R dS / 8 pi, int Phi2_i dV =
      // -M_jk oint R_i R_j n_k / R^3 dS / 8 pi
      const Vector R{x[0] - xi[0], x[1] - xi[1], x[2] - xi[2]};
      const auto r1 = std::sqrt(Dot(R, R));
      for (auto i = 0; i < 3; ++i) {
        auto phi1 = 0.0, phi2 = 0.0;
        for (auto k = 0; k < 3; ++k) phi1 -= M[i][k] * here.r[k] / r1;
        for (auto j = 0; j < 3; ++j) {
          for (auto k = 0; k < 3; ++k) {
            phi2 -= M[j][k] * R[i] * R[j] * here.r[k] / (r1 * r1 * r1);
          }
        }
        phi1 *= a * a / eightPi;
        phi2 *= a * a / eightPi;
        ballIntegrand[0][i].push_back(phi1 + phi2);
        ballIntegrand[1][i].push_back(phi1 - phi2);
        unit[i].push_back(here.r[i]);
      }
    }
  }
  struct Coefficients {
    std::vector<Complex> U, V, W;
  };
  auto coefficientsOf = [&](const SurfaceVectorField& f) {
    const auto duals = DisplacementMeasurement(grid, f);
    auto out = Coefficients{grid.Expand(f.up),
                            grid.Expand(duals.tangentialPotentialDual),
                            grid.Expand(duals.toroidalPotentialDual)};
    for (auto l = 1; l <= kGrid; ++l) {
      for (auto m = 0; m <= l; ++m) {
        out.V[SurfaceGrid::Index(l, m)] /= l * (l + 1.0);
        out.W[SurfaceGrid::Index(l, m)] /= l * (l + 1.0);
      }
    }
    return out;
  };
  std::array<Coefficients, 2> uK, sK;
  std::array<std::vector<Complex>, 2> dK;
  std::array<Vector, 2> J{};
  for (auto f = 0; f < 2; ++f) {
    uK[f] = coefficientsOf(uField[f]);
    sK[f] = coefficientsOf(sField[f]);
    dK[f] = grid.Expand(dField[f]);
    for (auto i = 0; i < 3; ++i) J[f][i] = grid.Integrate(ballIntegrand[f][i]);
  }
  std::array<std::vector<Complex>, 3> E;
  for (auto i = 0; i < 3; ++i) E[i] = grid.Expand(unit[i]);

  Oracle oracle{.a = a, .kappa = kappa, .mu = mu, .itau = itau, .data = {}};
  for (auto l = 0; l <= kLmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto i = SurfaceGrid::Index(l, m);
      auto moments = std::array<Complex, 2>{};
      if (l == 1) {
        for (auto f = 0; f < 2; ++f) {
          for (auto c = 0; c < 3; ++c) moments[f] += E[c][i] * J[f][c];
          moments[f] /= 4.0 * std::numbers::pi / 3.0;
        }
      }
      auto parts = std::array<Data, 2>{};
      for (auto part = 0; part < 2; ++part) {
        auto pick = [part](Complex z) {
          return part == 0 ? z.real() : z.imag();
        };
        auto& d = parts[part];
        d.u1 = {pick(uK[0].U[i]), pick(uK[0].V[i]), pick(uK[0].W[i])};
        d.u2 = {pick(uK[1].U[i]), pick(uK[1].V[i]), pick(uK[1].W[i])};
        d.s1 = {pick(sK[0].U[i]), pick(sK[0].V[i]), pick(sK[0].W[i])};
        d.s2 = {pick(sK[1].U[i]), pick(sK[1].V[i]), pick(sK[1].W[i])};
        d.d1 = pick(dK[0][i]);
        d.d2 = pick(dK[1][i]);
        d.k1 = pick(moments[0]);
        d.k2 = pick(moments[1]);
      }
      oracle.data.emplace(std::pair{l, m}, parts);
    }
  }
  return oracle;
}

/// The SLE side of a run: a solver for the grid and the scales.
GRDSolver SolverOf(const Material& material) {
  return GRDSolver(SurfaceGrid(kLmax), ComputeLoveNumbers(material, kLmax),
                   SeaLevelParameters{});
}

}  // namespace

TEST(WithoutGravity, TheMaxwellSphereThroughTimeInClosedForm) {
  const auto body = Body::Make();
  const auto& material = body.material;
  const auto a = material.Radius();
  // tau = eta / mu = 1 in the material's units
  const auto [kappas, mus] = material.Moduli().KappaMu();
  const auto tau = 1.0;
  const auto rheology = Rheology::Maxwell(material, mus.front() * tau).value();
  // the degrees' truncation with depth at 1e-12, below the default's
  // 1e-8, which is the floor of the default's departure (2.4e-8), so that
  // the steps' error shows above it
  const ViscoelasticOperator op(rheology, kLmax,
                                {.eps = 1e-12, .gravitating = false});
  const auto solver = SolverOf(material);
  const auto& sle = solver.SolverScales();
  const auto& scales = material.ModelScales();
  const auto length = scales.Length() / sle.Length();  // material to SLE
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto moment = sle.Factor(energy) / scales.Factor(energy);  // SLE to mat
  const auto time = sle.Time() / scales.Time();                    // SLE to mat

  // the static test's source, at half the radius, in the material's units
  const PointSource source{
      .at = {.radius = 0.5 * a, .colatitude = 1.1, .longitude = 4.0},
      .moment = {
          .rr = 0.9, .tt = -0.2, .pp = 0.5, .rt = 0.3, .rp = -0.7, .tp = 0.6}};
  auto given = source;  // the same in the solver's units
  given.at.radius *= length;
  for (auto* c : {&given.moment.rr, &given.moment.tt, &given.moment.pp,
                  &given.moment.rt, &given.moment.rp, &given.moment.tp}) {
    *c /= moment;
  }
  const auto oracle = MakeOracle(material, source, 1.0 / tau);

  // -- the oracle against itself
  const auto checks = std::vector<double>{0.5 * tau, tau, 2.0 * tau, 3.0 * tau};
  auto exact =
      std::map<std::pair<int, int>, std::vector<std::array<Complex, 3>>>{};
  auto scaleS = 0.0, scaleT = 0.0, contours = 0.0, nodes = 0.0, start = 0.0;
  auto linear = 0.0;
  for (auto l = 0; l <= kLmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto elastic = oracle.Elastic(l, m);
      const auto early = oracle.At(l, m, 1e-10 * tau, false, 24);
      auto& series = exact[{l, m}];
      series.push_back(elastic);
      for (auto c = 0; c < 3; ++c) {
        start = std::max(start, std::abs(early[c] - elastic[c]));
      }
      for (const auto t : checks) {
        const auto talbot = oracle.At(l, m, t, false, 24);
        const auto fewer = oracle.At(l, m, t, false, 18);
        const auto hyperbola = oracle.At(l, m, t, true, 32);
        const auto shorter = oracle.At(l, m, t, true, 24);
        for (auto c = 0; c < 3; ++c) {
          if (c < 2) {
            scaleS = std::max(scaleS, std::abs(talbot[c]));
          } else if (l >= 2) {
            scaleT = std::max(scaleT, std::abs(talbot[c]));
          }
          contours = std::max(contours, std::abs(talbot[c] - hyperbola[c]));
          nodes = std::max({nodes, std::abs(talbot[c] - fewer[c]),
                            std::abs(hyperbola[c] - shorter[c])});
        }
        // the toroidal response of the homogeneous body is the elastic
        // one times mu / mu(s): W_el (1 + t / tau)
        if (l >= 2) {
          linear = std::max(linear,
                            std::abs(talbot[2] - elastic[2] * (1.0 + t / tau)));
        }
        series.push_back(talbot);
      }
    }
  }
  const auto scale = std::max(scaleS, scaleT);
  RecordProperty("oracle_contours", std::format("{:.3e}", contours / scale));
  RecordProperty("oracle_node_counts", std::format("{:.3e}", nodes / scale));
  RecordProperty("oracle_start", std::format("{:.3e}", start / scale));
  RecordProperty("oracle_toroidal_linear",
                 std::format("{:.3e}", linear / scaleT));
  EXPECT_LT(contours, 1e-13 * scale);
  EXPECT_LT(nodes, 1e-11 * scale);
  EXPECT_LT(start, 1e-9 * scale);
  EXPECT_LT(linear, 1e-11 * scaleT);

  // -- the evolution without gravity against it, at three steps a scheme
  const SurfaceGrid& grid = solver.Grid();
  const auto sources = [&](double) { return std::vector<PointSource>{given}; };
  for (const auto scheme :
       {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    auto gaps = std::vector<std::array<double, 2>>{};
    for (const auto steps : {4, 8, 16}) {
      const auto dt = tau / steps;
      auto times = std::vector<double>{};
      for (auto n = 0; n <= 3 * steps; ++n) times.push_back(n * dt / time);
      Evolution evolution(
          op, solver, solver.NoOceanState(),
          {.scheme = scheme, .rotation = RotationalFeedback::None});
      const auto snapshots =
          evolution.Run(times, Evolution::Forcing{.sources = sources});
      ASSERT_EQ(snapshots.size(), times.size());
      EXPECT_TRUE(snapshots.front().potential.empty());
      EXPECT_TRUE(snapshots.front().relativeSeaLevelChange.empty());
      auto worstS = 0.0, worstT = 0.0, worst1 = 0.0;
      const auto compared =
          std::vector<int>{0, steps / 2, steps, 2 * steps, 3 * steps};
      for (auto k = std::size_t{0}; k < compared.size(); ++k) {
        const auto& snapshot = snapshots[compared[k]];
        const auto U = grid.Expand(snapshot.displacement);
        const auto V = grid.Expand(snapshot.tangentialPotential);
        const auto W = grid.Expand(snapshot.toroidalPotential);
        for (auto l = 0; l <= kLmax; ++l) {
          for (auto m = 0; m <= l; ++m) {
            const auto i = SurfaceGrid::Index(l, m);
            const auto& want = exact.at({l, m})[k];
            const auto gap = std::max(std::abs(U[i] / length - want[0]),
                                      std::abs(V[i] / length - want[1]));
            worstS = std::max(worstS, gap);
            if (l == 1) worst1 = std::max(worst1, gap);
            if (l >= 2) {
              worstT = std::max(worstT, std::abs(W[i] / length - want[2]));
            }
          }
        }
      }
      gaps.push_back({worstS / scaleS, worstT / scaleT});
      const auto name = std::format(
          "{}_dt_tau_over_{}",
          scheme == Scheme::ExponentialTrapezoid ? "trapezoid" : "euler",
          steps);
      RecordProperty(name + "_spheroidal",
                     std::format("{:.3e}", gaps.back()[0]));
      RecordProperty(name + "_toroidal", std::format("{:.3e}", gaps.back()[1]));
      RecordProperty(name + "_degree1", std::format("{:.3e}", worst1 / scaleS));
    }
    // the scheme's order: the spheroidal gap falls by 2^order a halving
    // of the step. The toroidal response grows linearly in time, which
    // both schemes step exactly: its gap is the discretisation's in space
    // at every step.
    const auto order = scheme == Scheme::ExponentialTrapezoid ? 2.0 : 1.0;
    for (auto k = std::size_t{1}; k < gaps.size(); ++k) {
      const auto rate = std::log2(gaps[k - 1][0] / gaps[k][0]);
      RecordProperty(
          std::format("{}_rate_{}", order == 2.0 ? "trapezoid" : "euler", k),
          std::format("{:.3f}", rate));
      EXPECT_GT(rate, order - 0.25) << "step " << k;
      EXPECT_LT(rate, order + 0.25) << "step " << k;
    }
    EXPECT_LT(gaps.back()[0], order == 2.0 ? 1e-8 : 2e-5);
    for (const auto& gap : gaps) EXPECT_LT(gap[1], 1e-9);
  }
}

namespace {

/// The closed form through time of the homogeneous Maxwell sphere under a
/// surface pressure (PressureClosedForm), per unit pressure and by degree:
/// the transform of the response to a pressure p(t) is the elastic closed
/// form at the Maxwell modulus mu s / (s + 1 / tau) times the transform of
/// p, inverted on the contours of Oracle.
struct PressureOracle {
  long double a = 0.0, kappa = 0.0, mu = 0.0, itau = 0.0;

  /// The elastic response (U, V) of degree l.
  [[nodiscard]] std::array<double, 2> Elastic(int l) const {
    const auto x = PressureClosedForm(l, a, kappa, Wide(mu));
    return {static_cast<double>(x[0].real()), static_cast<double>(x[1].real())};
  }

  /// The response at t > 0 of degree l to the pressure t^(power - 1) /
  /// (power - 1)! from t = 0 (power 1: held from t = 0; power 2: the unit
  /// ramp t), by one contour with `nodes` nodes.
  [[nodiscard]] std::array<long double, 2> At(int l, long double t, int power,
                                               bool hyperbola,
                                               int nodes) const {
    auto out = std::array<long double, 2>{};
    for (auto c = 0; c < 2; ++c) {
      const auto F = [&](Wide s) {
        const auto maxwell = mu * s / (s + itau);
        auto value = PressureClosedForm(l, a, kappa, maxwell)[c];
        for (auto k = 0; k < power; ++k) value /= s;
        return value;
      };
      out[c] = hyperbola ? Hyperbola(F, t, nodes) : Talbot(F, t, nodes);
    }
    return out;
  }

  /// The response at t >= 0 to the pressure ramped from zero at t = 0 to
  /// one at t = T and held after: (R(t) - R(t - T)) / T with R the unit
  /// ramp's response (zero before its start).
  [[nodiscard]] std::array<double, 2> Ramped(int l, double t, double T) const {
    if (t <= 0.0) return {0.0, 0.0};
    const auto up = At(l, t, 2, false, 24);
    auto out = std::array<double, 2>{};
    for (auto c = 0; c < 2; ++c) {
      auto value = up[c];
      if (t > T) value -= At(l, t - T, 2, false, 24)[c];
      out[c] = static_cast<double>(value / T);
    }
    return out;
  }

  /// The response to the pressure held from t = 0 (at t = 0, the elastic).
  [[nodiscard]] std::array<double, 2> Held(int l, double t) const {
    if (t <= 0.0) return Elastic(l);
    const auto x = At(l, t, 1, false, 24);
    return {static_cast<double>(x[0]), static_cast<double>(x[1])};
  }
};

/// Gauss-Legendre nodes and weights on [0, 1], n of them.
std::vector<std::pair<long double, long double>> GaussLegendre(int n) {
  auto out = std::vector<std::pair<long double, long double>>{};
  for (auto i = 1; i <= n; ++i) {
    auto x = std::cos(std::numbers::pi_v<long double> * (i - 0.25L) /
                      (n + 0.5L));
    auto p0 = 1.0L, p1 = x;
    for (auto iteration = 0; iteration < 100; ++iteration) {
      p0 = 1.0L;
      p1 = x;
      for (auto k = 2; k <= n; ++k) {
        const auto p2 = ((2.0L * k - 1.0L) * x * p1 - (k - 1.0L) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      const auto dp = n * (x * p1 - p0) / (x * x - 1.0L);
      const auto dx = p1 / dp;
      x -= dx;
      if (std::abs(dx) < 1e-19L) break;
    }
    p0 = 1.0L;
    p1 = x;
    for (auto k = 2; k <= n; ++k) {
      const auto p2 = ((2.0L * k - 1.0L) * x * p1 - (k - 1.0L) * p0) / k;
      p0 = p1;
      p1 = p2;
    }
    const auto dp = n * (x * p1 - p0) / (x * x - 1.0L);
    const auto w = 2.0L / ((1.0L - x * x) * dp * dp);
    out.emplace_back((1.0L - x) / 2.0L, w / 2.0L);
  }
  return out;
}

/// A load's coefficients without a degree-1 part, every other degree to
/// lmax at every order, of size about one (a fixed pattern).
std::vector<Complex> LoadWithoutDegreeOne(const SurfaceGrid& grid) {
  auto out = grid.ZeroCoefficients();
  for (auto l = 0; l <= grid.LMax(); ++l) {
    if (l == 1) continue;
    for (auto m = 0; m <= l; ++m) {
      const auto re = std::cos(1.7 * l + 0.9 * m + 0.3);
      const auto im = m == 0 ? 0.0 : std::sin(2.3 * l - 1.1 * m);
      out[SurfaceGrid::Index(l, m)] = Complex(re, im) / (1.0 + 0.2 * l);
    }
  }
  return out;
}

}  // namespace

TEST(WithoutGravity, TheMaxwellSphereUnderASurfaceLoadInClosedForm) {
  // A load is a mass per unit area sigma acting by its weight, the normal
  // traction g sigma pressing on the surface (g the reference surface
  // gravity). On the homogeneous sphere without gravity the response is
  // the closed form under that pressure (PressureClosedForm), statically
  // and, on the Maxwell sphere, through time by the correspondence
  // principle, for a load held from t = 0 and for one ramped up over tau
  // and held. The oracle is checked against itself: two contours and two
  // node counts agree, t -> 0+ is the elastic closed form, degree 0 (the
  // bulk modulus alone) does not relax, and the ramp's response is the
  // held load's integrated in time. Under a held load with a deviatoric
  // part the Maxwell body without gravity flows without bound.
  const auto body = Body::Make();
  const auto& material = body.material;
  const auto [kappas, mus] = material.Moduli().KappaMu();
  const auto tau = 1.0;
  const auto rheology = Rheology::Maxwell(material, mus.front() * tau).value();
  const ViscoelasticOperator op(rheology, kLmax,
                                {.eps = 1e-12, .gravitating = false});
  const auto solver = SolverOf(material);
  const auto& grid = solver.Grid();
  const auto& sle = solver.SolverScales();
  const auto& scales = material.ModelScales();
  const auto length = scales.Length() / sle.Length();  // material to SLE
  const auto time = sle.Time() / scales.Time();        // SLE to material
  const auto density = Dimensions{.mass = 1, .length = -2};
  const auto toSolver = scales.Factor(density) / sle.Factor(density);
  const auto g = material.SurfaceGravity();  // material units

  const PressureOracle oracle{.a = material.Radius(),
                             .kappa = kappas.front(),
                             .mu = mus.front(),
                             .itau = 1.0 / tau};

  // -- the oracle against itself
  auto scale = 0.0, contours = 0.0, nodes = 0.0, start = 0.0, still = 0.0;
  auto integrated = 0.0;
  const auto rule = GaussLegendre(40);
  for (auto l = 0; l <= kLmax; ++l) {
    if (l == 1) continue;
    const auto elastic = oracle.Elastic(l);
    const auto early = oracle.At(l, 1e-10L, 1, false, 24);
    for (auto c = 0; c < 2; ++c) {
      start = std::max(start, std::abs(static_cast<double>(early[c]) -
                                       elastic[c]));
    }
    for (const auto t : {0.5L, 1.0L, 2.0L, 3.0L}) {
      const auto talbot = oracle.At(l, t, 1, false, 24);
      const auto fewer = oracle.At(l, t, 1, false, 18);
      const auto hyperbola = oracle.At(l, t, 1, true, 32);
      const auto shorter = oracle.At(l, t, 1, true, 24);
      const auto ramp = oracle.At(l, t, 2, false, 24);
      const auto rampHyperbola = oracle.At(l, t, 2, true, 32);
      // the ramp's response, the held load's integrated over (0, t)
      auto sum = std::array<long double, 2>{};
      for (const auto& [x, w] : rule) {
        const auto held = oracle.At(l, x * t, 1, false, 24);
        for (auto c = 0; c < 2; ++c) sum[c] += w * t * held[c];
      }
      for (auto c = 0; c < 2; ++c) {
        scale = std::max(scale, static_cast<double>(std::abs(talbot[c])));
        contours = std::max(
            {contours, static_cast<double>(std::abs(talbot[c] - hyperbola[c])),
             static_cast<double>(std::abs(ramp[c] - rampHyperbola[c]) / t)});
        nodes = std::max(
            {nodes, static_cast<double>(std::abs(talbot[c] - fewer[c])),
             static_cast<double>(std::abs(hyperbola[c] - shorter[c]))});
        integrated = std::max(
            integrated, static_cast<double>(std::abs(ramp[c] - sum[c]) / t));
      }
      if (l == 0) {
        still = std::max(still, static_cast<double>(
                                    std::abs(talbot[0] - elastic[0])));
      }
    }
  }
  RecordProperty("oracle_contours", std::format("{:.3e}", contours / scale));
  RecordProperty("oracle_node_counts", std::format("{:.3e}", nodes / scale));
  RecordProperty("oracle_start", std::format("{:.3e}", start / scale));
  RecordProperty("oracle_degree0_still", std::format("{:.3e}", still / scale));
  RecordProperty("oracle_ramp_integrated",
                 std::format("{:.3e}", integrated / scale));
  EXPECT_LT(contours, 1e-13 * scale);
  EXPECT_LT(nodes, 1e-11 * scale);
  EXPECT_LT(start, 1e-9 * scale);
  EXPECT_LT(still, 1e-13 * scale);
  EXPECT_LT(integrated, 1e-13 * scale);

  // the load: every degree but 1, at every order, in the material's units
  const auto sigma = LoadWithoutDegreeOne(grid);
  auto sigmaSolver = sigma;
  for (auto& c : sigmaSolver) c *= toSolver;
  const auto field = grid.Synthesise(sigmaSolver);

  // -- the static problem: the run's start under the held load is the
  // elastic closed form
  {
    Evolution evolution(op, solver, solver.NoOceanState(),
                        {.rotation = RotationalFeedback::None});
    const auto one = std::vector<double>{0.0};
    const auto snapshot =
        evolution.Run(one, [&](double) { return field; }).front();
    EXPECT_TRUE(snapshot.potential.empty());
    EXPECT_TRUE(snapshot.relativeSeaLevelChange.empty());
    const auto U = grid.Expand(snapshot.displacement);
    const auto V = grid.Expand(snapshot.tangentialPotential);
    auto worst = 0.0, size = 0.0;
    for (auto l = 0; l <= kLmax; ++l) {
      const auto want = l == 1 ? std::array<double, 2>{} : oracle.Elastic(l);
      for (auto m = 0; m <= l; ++m) {
        const auto i = SurfaceGrid::Index(l, m);
        const auto p = g * sigma[i];
        worst = std::max({worst, std::abs(U[i] / length - want[0] * p),
                          std::abs(V[i] / length - want[1] * p)});
        size = std::max({size, std::abs(want[0] * p), std::abs(want[1] * p)});
      }
    }
    RecordProperty("static_closed_form", std::format("{:.3e}", worst / size));
    EXPECT_LT(worst, 2e-11 * size);
  }

  // -- through time, a load held from t = 0 and one ramped over tau, at
  // three steps a scheme
  const auto compare = std::vector<double>{0.0, 0.5, 1.0, 2.0, 3.0};  // tau
  for (const auto ramped : {false, true}) {
    // the oracle's response per unit pressure at the compared times
    auto exact = std::vector<std::vector<std::array<double, 2>>>{};
    auto size = 0.0;
    for (const auto t : compare) {
      auto row = std::vector<std::array<double, 2>>(kLmax + 1);
      for (auto l = 0; l <= kLmax; ++l) {
        if (l == 1) continue;
        row[l] = ramped ? oracle.Ramped(l, t * tau, tau)
                        : oracle.Held(l, t * tau);
        for (auto m = 0; m <= l; ++m) {
          const auto p = std::abs(g * sigma[SurfaceGrid::Index(l, m)]);
          size = std::max({size, std::abs(row[l][0]) * p,
                           std::abs(row[l][1]) * p});
        }
      }
      exact.push_back(std::move(row));
    }
    const auto factor = [&](double t) {
      return ramped ? std::min(t * time / tau, 1.0) : 1.0;
    };
    const auto load = [&](double t) {
      auto out = field;
      for (auto& v : out) v *= factor(t);
      return out;
    };
    for (const auto scheme :
         {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
      auto gaps = std::vector<double>{};
      for (const auto steps : {4, 8, 16}) {
        const auto dt = tau / steps;
        auto times = std::vector<double>{};
        for (auto n = 0; n <= 3 * steps; ++n) times.push_back(n * dt / time);
        Evolution evolution(
            op, solver, solver.NoOceanState(),
            {.scheme = scheme, .rotation = RotationalFeedback::None});
        const auto snapshots =
            evolution.Run(times, Evolution::Forcing{.directLoad = load});
        ASSERT_EQ(snapshots.size(), times.size());
        auto worst = 0.0, worst1 = 0.0;
        for (auto k = std::size_t{0}; k < compare.size(); ++k) {
          const auto& snapshot =
              snapshots[static_cast<std::size_t>(compare[k] * steps)];
          const auto U = grid.Expand(snapshot.displacement);
          const auto V = grid.Expand(snapshot.tangentialPotential);
          for (auto l = 0; l <= kLmax; ++l) {
            for (auto m = 0; m <= l; ++m) {
              const auto i = SurfaceGrid::Index(l, m);
              const auto p = g * sigma[i];
              const auto gap =
                  std::max(std::abs(U[i] / length - exact[k][l][0] * p),
                           std::abs(V[i] / length - exact[k][l][1] * p));
              worst = std::max(worst, gap);
              if (l == 1) worst1 = std::max(worst1, gap);
            }
          }
        }
        gaps.push_back(worst / size);
        const auto name = std::format(
            "{}_{}_dt_tau_over_{}", ramped ? "ramp" : "held",
            scheme == Scheme::ExponentialTrapezoid ? "trapezoid" : "euler",
            steps);
        RecordProperty(name, std::format("{:.3e}", gaps.back()));
        // degree 1 has no load, and no response
        EXPECT_LT(worst1, 1e-14 * size) << name;
      }
      // the scheme's order: the gap falls by 2^order a halving of the step.
      // Under the ramp the flow without bound grows quadratically, which
      // the steps do not follow exactly, where under the held load it
      // grows linearly, which they do: the ramp's constant is the larger.
      const auto order = scheme == Scheme::ExponentialTrapezoid ? 2.0 : 1.0;
      for (auto k = std::size_t{1}; k < gaps.size(); ++k) {
        const auto rate = std::log2(gaps[k - 1] / gaps[k]);
        RecordProperty(std::format("{}_{}_rate_{}", ramped ? "ramp" : "held",
                                   order == 2.0 ? "trapezoid" : "euler", k),
                       std::format("{:.3f}", rate));
        EXPECT_GT(rate, order - 0.25) << "step " << k;
        EXPECT_LT(rate, order + 0.25) << "step " << k;
      }
      if (ramped) {
        EXPECT_LT(gaps.back(), order == 2.0 ? 3e-4 : 3e-2);
      } else {
        EXPECT_LT(gaps.back(), order == 2.0 ? 1e-7 : 1e-4);
      }
    }
  }
}

namespace {

/// The examples' PREM-like deck (inner core, fluid outer core, mantle),
/// whole or its shell above the fluid core alone, on the whole deck's
/// scales.
struct Earth {
  EarthModel model;
  Material material;
  Rheology rheology;

  static Earth Make(bool shell, int lmax) {
    const auto format =
        DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0);
    const auto whole = Deck::Read(std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                                      "prem8.elastic.deck",
                                  format)
                           .value();
    const auto scales = EarthModel::FromDeck(whole).value().ModelScales();
    auto deck = whole;
    if (shell) {
      // the knots from the mantle's base, the outermost fluid layer's top
      const auto vs = whole.Column("vsv");
      auto first = std::size_t{0};
      for (auto k = std::size_t{0}; k < whole.KnotCount(); ++k) {
        if (vs[k] == 0.0) first = k + 1;
      }
      auto radius = std::vector<double>(whole.Radius().begin() + first,
                                        whole.Radius().end());
      auto columns = std::vector<std::pair<std::string, std::vector<double>>>{};
      for (const auto& name : whole.Names()) {
        const auto c = whole.Column(name);
        columns.emplace_back(name,
                             std::vector<double>(c.begin() + first, c.end()));
      }
      deck = Deck::Create(std::move(radius), std::move(columns)).value();
    }
    auto model = EarthModel::FromDeck(deck, {.scales = scales}).value();
    auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
    auto material = Material::Create(model, mesh).value();
    // a Maxwell mantle of 1e19 Pa s below the transversely isotropic
    // upper mantle, its relaxation times of a few years
    auto viscosity = std::map<std::size_t, RadialValue>{};
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i, 1e19 / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    return Earth{std::move(model), std::move(material), std::move(rheology)};
  }
};

constexpr auto kEarthLmax = 8;

/// A thrust 600 km down, in the solver's units.
PointSource ThrustIn(const GRDSolver& solver, const Material& material) {
  const auto& sle = solver.SolverScales();
  const auto radius =
      material.Radius() * material.ModelScales().Length() / sle.Length();
  return PointSource{
      .at = {.radius = radius - 600e3 / sle.Length(),
             .colatitude = 1.1,
             .longitude = 4.0},
      .moment = {
          .rr = 0.9, .tt = -0.2, .pp = -0.7, .rt = 0.3, .rp = -0.7, .tp = 0.6}};
}

/// The largest coefficient departure of two fields above degree `from`,
/// over the largest coefficient of the second.
double Departure(const SurfaceGrid& grid, const std::vector<double>& x,
                 const std::vector<double>& y, int from) {
  const auto cx = grid.Expand(x), cy = grid.Expand(y);
  auto worst = 0.0, scale = 0.0;
  for (auto l = from; l <= grid.LMax(); ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto i = SurfaceGrid::Index(l, m);
      worst = std::max(worst, std::abs(cx[i] - cy[i]));
      scale = std::max(scale, std::abs(cy[i]));
    }
  }
  return scale > 0.0 ? worst / scale : worst;
}

}  // namespace

TEST(WithoutGravity, AFluidCoreStopsTheSystemsAtItsTop) {
  // Without gravity the fluid core has no degree-l pressure above degree
  // zero: the body's run is its shell's above the core, traction-free at
  // its base, at every degree from 1; at degree zero the core keeps its
  // bulk modulus and the two differ.
  const auto whole = Earth::Make(false, kEarthLmax);
  const auto shell = Earth::Make(true, kEarthLmax);
  const auto solver = GRDSolver(SurfaceGrid(kEarthLmax),
                                ComputeLoveNumbers(whole.material, kEarthLmax),
                                SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto source = ThrustIn(solver, whole.material);
  const auto sources = [&](double) { return std::vector<PointSource>{source}; };
  const auto year = 3.15576e7 / solver.SolverScales().Time();
  const auto times =
      std::vector<double>{0.0, 50.0 * year, 200.0 * year, 1000.0 * year};
  auto runOf = [&](const Earth& earth) {
    const ViscoelasticOperator op(earth.rheology, kEarthLmax,
                                  {.gravitating = false});
    Evolution evolution(op, solver, solver.NoOceanState(),
                        {.rotation = RotationalFeedback::None});
    return evolution.Run(times, Evolution::Forcing{.sources = sources});
  };
  const auto a = runOf(whole);
  const auto b = runOf(shell);
  ASSERT_EQ(a.size(), times.size());
  auto worst = 0.0, degreeZero = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    worst = std::max(
        {worst, Departure(grid, a[n].displacement, b[n].displacement, 1),
         Departure(grid, a[n].tangentialPotential, b[n].tangentialPotential, 1),
         Departure(grid, a[n].toroidalPotential, b[n].toroidalPotential, 1)});
    const auto ca = grid.Expand(a[n].displacement);
    const auto cb = grid.Expand(b[n].displacement);
    degreeZero =
        std::max(degreeZero, std::abs(ca[0] - cb[0]) / std::abs(cb[0]));
  }
  RecordProperty("shell_departure", std::format("{:.3e}", worst));
  RecordProperty("degree0_core_effect", std::format("{:.3e}", degreeZero));
  EXPECT_LT(worst, 1e-11);
  EXPECT_GT(degreeZero, 1e-3);
}

TEST(WithoutGravity, AResumedRunAndTheFactoredSourcesAreTheGeneralRun) {
  const auto earth = Earth::Make(false, kEarthLmax);
  const auto solver = GRDSolver(SurfaceGrid(kEarthLmax),
                                ComputeLoveNumbers(earth.material, kEarthLmax),
                                SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const ViscoelasticOperator op(earth.rheology, kEarthLmax,
                                {.gravitating = false});
  const auto source = ThrustIn(solver, earth.material);
  const auto year = 3.15576e7 / solver.SolverScales().Time();
  // a second thrust from the third node on
  auto second = source;
  second.at.colatitude = 2.0;
  second.moment.rr = -0.5;
  const auto times = std::vector<double>{
      0.0, 20.0 * year, 50.0 * year, 100.0 * year, 200.0 * year, 400.0 * year};
  const auto sources = [&](double t) {
    auto out = std::vector<PointSource>{source};
    if (t >= times[2]) out.push_back(second);
    return out;
  };
  const auto forcing = Evolution::Forcing{.sources = sources};
  auto checkpoints = std::map<std::size_t, Evolution::Checkpoint>{};
  Evolution evolution(
      op, solver, solver.NoOceanState(),
      {.rotation = RotationalFeedback::None,
       .checkpoint_stride = 2,
       .on_checkpoint = [&](std::size_t node, const Evolution::Checkpoint& c) {
         checkpoints.emplace(node, c);
       }});
  const auto run = evolution.Run(times, forcing);
  ASSERT_TRUE(checkpoints.contains(2));
  Evolution again(op, solver, solver.NoOceanState(),
                  {.rotation = RotationalFeedback::None});
  const auto resumed =
      again.Resume(checkpoints.at(2), std::span(times).subspan(2), forcing);
  ASSERT_EQ(resumed.size(), times.size() - 3);
  for (auto k = std::size_t{0}; k < resumed.size(); ++k) {
    EXPECT_EQ(resumed[k].displacement, run[k + 3].displacement) << k;
    EXPECT_EQ(resumed[k].toroidalPotential, run[k + 3].toroidalPotential) << k;
  }

  Evolution factored(
      op, solver, solver.NoOceanState(),
      {.rotation = RotationalFeedback::None, .factor_sources = true});
  const auto units = factored.Run(times, forcing);
  ASSERT_EQ(units.size(), run.size());
  auto worst = 0.0;
  for (auto n = std::size_t{0}; n < run.size(); ++n) {
    EXPECT_TRUE(units[n].potential.empty());
    EXPECT_TRUE(units[n].relativeSeaLevelChange.empty());
    worst = std::max(
        {worst, Departure(grid, units[n].displacement, run[n].displacement, 0),
         Departure(grid, units[n].tangentialPotential,
                   run[n].tangentialPotential, 0),
         Departure(grid, units[n].toroidalPotential, run[n].toroidalPotential,
                   0)});
  }
  RecordProperty("factored_departure", std::format("{:.3e}", worst));
  EXPECT_LT(worst, 1e-11);
}

TEST(WithoutGravity, AStaticLoadIsTheTractionNumbersTimesItsWeight) {
  // The one-node run without gravity under a load of one degree l is the
  // non-gravitating traction numbers times the load's weight, g sigma,
  // with g the reference surface gravity the Love numbers take: U = g h_u
  // sigma and V = g l_u sigma, coefficient by coefficient, on the
  // homogeneous deck and on the PREM-like deck with its fluid core, on the
  // elastic structure (the static problem's branchless rheology).
  constexpr auto lmax = 16;
  struct Case {
    std::string name;
    Material material;
  };
  auto cases = std::vector<Case>{};
  cases.push_back({"homogeneous", Body::Make().material});
  cases.push_back({"prem", Earth::Make(false, lmax).material});
  for (const auto& [name, material] : cases) {
    const auto elastic = Rheology::Create(material, {}).value();
    const ViscoelasticOperator op(elastic, lmax, {.gravitating = false});
    const auto love = ComputeLoveNumbers(material, lmax);
    const auto numbers = ComputeTractionNumbers(material, lmax, false);
    const auto g = love.surface_gravity;  // SI
    const GRDSolver solver(SurfaceGrid(lmax), love, SeaLevelParameters{});
    const auto& grid = solver.Grid();
    const auto& sle = solver.SolverScales();
    const auto toLoad = sle.Factor(Dimensions{.mass = 1, .length = -2});
    const auto one = std::vector<double>{0.0};
    auto worst = 0.0;
    for (const auto l : {0, 2, 3, 7, lmax}) {
      // the load of degree l, kg m^-2: every order, of tonnes per square
      // metre
      auto sigma = grid.ZeroCoefficients();
      for (auto m = 0; m <= l; ++m) {
        sigma[SurfaceGrid::Index(l, m)] =
            1e3 * Complex(1.0 + 0.3 * m, m == 0 ? 0.0 : 0.5 - 0.2 * m);
      }
      auto given = sigma;
      for (auto& c : given) c /= toLoad;
      const auto field = grid.Synthesise(given);
      Evolution evolution(op, solver, solver.NoOceanState(),
                          {.rotation = RotationalFeedback::None});
      const auto snapshot =
          evolution
              .Run(one, Evolution::Forcing{.directLoadOutsideWaterBudget =
                                               [&](double) { return field; }})
              .front();
      const auto U = grid.Expand(snapshot.displacement);
      const auto V = grid.Expand(snapshot.tangentialPotential);
      auto gap = 0.0, size = 0.0, elsewhere = 0.0;
      for (auto k = 0; k <= lmax; ++k) {
        for (auto m = 0; m <= k; ++m) {
          const auto i = SurfaceGrid::Index(k, m);
          if (k != l) {
            elsewhere = std::max(
                {elsewhere, std::abs(U[i]), std::abs(V[i])});
            continue;
          }
          const auto wantU = g * numbers.h_u[l] * sigma[i];
          const auto wantV = g * numbers.l_u[l] * sigma[i];
          gap = std::max({gap, std::abs(U[i] * sle.Length() - wantU),
                          std::abs(V[i] * sle.Length() - wantV)});
          size = std::max({size, std::abs(wantU), std::abs(wantV)});
        }
      }
      RecordProperty(std::format("{}_degree_{}", name, l),
                     std::format("{:.3e}", gap / size));
      EXPECT_LT(elsewhere * sle.Length(), 1e-14 * size) << name << " " << l;
      worst = std::max(worst, gap / size);
    }
    EXPECT_LT(worst, 1e-12) << name;
  }
}

TEST(WithoutGravity, LoadsAndSourcesTogetherAreTheSumOfTheirRuns) {
  // The problem is linear: a run under a load and a source is the sum of
  // the run under the load and the run under the source, to rounding, on
  // the PREM-like Maxwell Earth with its fluid core, the load present at
  // the start and ramped up after it.
  const auto earth = Earth::Make(false, kEarthLmax);
  const auto solver = GRDSolver(SurfaceGrid(kEarthLmax),
                                ComputeLoveNumbers(earth.material, kEarthLmax),
                                SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const ViscoelasticOperator op(earth.rheology, kEarthLmax,
                                {.gravitating = false});
  const auto source = ThrustIn(solver, earth.material);
  const auto year = 3.15576e7 / solver.SolverScales().Time();
  const auto times = std::vector<double>{
      0.0, 20.0 * year, 50.0 * year, 100.0 * year, 200.0 * year, 400.0 * year};
  const auto field = grid.Synthesise(LoadWithoutDegreeOne(grid));
  const auto load = [&](double t) {
    auto out = field;
    const auto factor = 0.5 + 0.5 * std::min(t / times[2], 1.0);
    for (auto& v : out) v *= factor;
    return out;
  };
  const auto sources = [&](double) { return std::vector<PointSource>{source}; };
  auto runOf = [&](const Evolution::Forcing& forcing) {
    Evolution evolution(op, solver, solver.NoOceanState(),
                        {.rotation = RotationalFeedback::None});
    return evolution.Run(times, forcing);
  };
  const auto both = runOf({.directLoad = load, .sources = sources});
  const auto loaded = runOf({.directLoad = load});
  const auto sourced = runOf({.sources = sources});
  ASSERT_EQ(both.size(), times.size());
  auto worst = 0.0, scale = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_TRUE(loaded[n].toroidalPotential.empty());
    for (const auto member : {&Evolution::Snapshot::displacement,
                              &Evolution::Snapshot::tangentialPotential}) {
      const auto& a = both[n].*member;
      const auto& b = loaded[n].*member;
      const auto& c = sourced[n].*member;
      for (auto i = std::size_t{0}; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(a[i] - b[i] - c[i]));
        scale = std::max({scale, std::abs(b[i]), std::abs(c[i])});
      }
    }
    EXPECT_EQ(both[n].toroidalPotential, sourced[n].toroidalPotential) << n;
  }
  RecordProperty("superposition", std::format("{:.3e}", worst / scale));
  EXPECT_LT(worst, 1e-13 * scale);
}

TEST(WithoutGravity, ALoadedRunResumesRecordsAndStreamsAsTheGeneralRun) {
  // Under a load, alone and beside a source: a resumed run is the
  // uninterrupted one bitwise from a checkpoint with its strain, and to
  // rounding from one without (the strain regenerated under the
  // checkpoint's loads); the snapshots handed to on_snapshot without being
  // kept are the kept ones; a resumed call given its checkpoint's node as
  // the record offset keeps the uninterrupted run's states; and a layer's
  // height changes nothing. The factored sources take no load, as with
  // gravity.
  const auto earth = Earth::Make(false, kEarthLmax);
  const auto solver = GRDSolver(SurfaceGrid(kEarthLmax),
                                ComputeLoveNumbers(earth.material, kEarthLmax),
                                SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const ViscoelasticOperator op(earth.rheology, kEarthLmax,
                                {.gravitating = false});
  const auto source = ThrustIn(solver, earth.material);
  const auto year = 3.15576e7 / solver.SolverScales().Time();
  const auto times = std::vector<double>{
      0.0, 20.0 * year, 50.0 * year, 100.0 * year, 200.0 * year, 400.0 * year};
  const auto field = grid.Synthesise(LoadWithoutDegreeOne(grid));
  const auto load = [&](double t) {
    auto out = field;
    const auto factor = std::min(t / times[2], 1.0);
    for (auto& v : out) v *= factor;
    return out;
  };
  const auto sources = [&](double) { return std::vector<PointSource>{source}; };
  for (const auto sourced : {false, true}) {
    const auto label = sourced ? "load and source" : "load";
    const auto forcing =
        sourced ? Evolution::Forcing{.directLoadOutsideWaterBudget = load,
                                     .sources = sources}
                : Evolution::Forcing{.directLoadOutsideWaterBudget = load};
    auto checkpoints = std::map<std::size_t, Evolution::Checkpoint>{};
    Evolution evolution(
        op, solver, solver.NoOceanState(),
        {.rotation = RotationalFeedback::None,
         .record_stride = 1,
         .checkpoint_stride = 2,
         .on_checkpoint = [&](std::size_t node, const Evolution::Checkpoint& c) {
           checkpoints.emplace(node, c);
         }});
    const auto run = evolution.Run(times, forcing);
    ASSERT_TRUE(checkpoints.contains(2)) << label;
    const auto rest = std::span(times).subspan(2);
    {
      Evolution again(op, solver, solver.NoOceanState(),
                      {.rotation = RotationalFeedback::None});
      const auto resumed = again.Resume(checkpoints.at(2), rest, forcing);
      ASSERT_EQ(resumed.size(), times.size() - 3) << label;
      for (auto k = std::size_t{0}; k < resumed.size(); ++k) {
        EXPECT_EQ(resumed[k].displacement, run[k + 3].displacement)
            << label << " " << k;
        EXPECT_EQ(resumed[k].tangentialPotential,
                  run[k + 3].tangentialPotential)
            << label << " " << k;
        EXPECT_EQ(resumed[k].toroidalPotential, run[k + 3].toroidalPotential)
            << label << " " << k;
      }
    }
    if (!sourced) {
      // from a checkpoint without its strain, regenerated under its loads
      auto bare = checkpoints.at(2);
      bare.strain.clear();
      Evolution again(op, solver, solver.NoOceanState(),
                      {.rotation = RotationalFeedback::None});
      const auto resumed = again.Resume(bare, rest, forcing);
      ASSERT_EQ(resumed.size(), times.size() - 3);
      auto worst = 0.0;
      for (auto k = std::size_t{0}; k < resumed.size(); ++k) {
        worst = std::max(worst, Departure(grid, resumed[k].displacement,
                                          run[k + 3].displacement, 0));
      }
      // the regenerated strain is the unrelaxed system's, the kept one the
      // step's effective system's: the two agree to the solves' rounding
      RecordProperty("resumed_regenerated", std::format("{:.3e}", worst));
      EXPECT_LT(worst, 1e-11);
      // and the direct load's own entries, Run and Resume
      Evolution direct(op, solver, solver.NoOceanState(),
                       {.rotation = RotationalFeedback::None});
      const auto plain = direct.Run(times, load);
      ASSERT_EQ(plain.size(), times.size());
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        EXPECT_EQ(plain[n].displacement, run[n].displacement) << n;
      }
      const auto plainResumed = direct.Resume(checkpoints.at(2), rest, load);
      ASSERT_EQ(plainResumed.size(), times.size() - 3);
      EXPECT_EQ(plainResumed.back().displacement, run.back().displacement);
    }
    {
      // streamed: every node handed over, none kept
      auto streamed = std::vector<Evolution::Snapshot>{};
      Evolution evolution(op, solver, solver.NoOceanState(),
                          {.rotation = RotationalFeedback::None,
                           .on_snapshot =
                               [&](std::size_t, const Evolution::Snapshot& s) {
                                 streamed.push_back(s);
                               },
                           .keep_snapshots = false});
      EXPECT_TRUE(evolution.Run(times, forcing).empty()) << label;
      ASSERT_EQ(streamed.size(), times.size()) << label;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        EXPECT_EQ(streamed[n].displacement, run[n].displacement)
            << label << " " << n;
      }
    }
    if (!sourced) {
      // the record offset: every second state of the whole run kept by a
      // call resumed at node 2
      Evolution again(op, solver, solver.NoOceanState(),
                      {.rotation = RotationalFeedback::None,
                       .record_stride = 2,
                       .record_offset = 2});
      const auto resumed = again.Resume(checkpoints.at(2), rest, forcing);
      ASSERT_EQ(resumed.size(), times.size() - 3);
      for (auto k = std::size_t{0}; k < resumed.size(); ++k) {
        const auto whole = k + 3;
        const auto keeps = whole % 2 == 0 || whole + 1 == times.size();
        EXPECT_EQ(resumed[k].internalState.has_value(), keeps) << whole;
        if (keeps) {
          EXPECT_EQ(resumed[k].internalState->m, run[whole].internalState->m)
              << whole;
        }
        EXPECT_EQ(resumed[k].loadCoefficients, run[whole].loadCoefficients)
            << whole;
      }
    }
  }

  // a layer's height enters the sea level change alone, which a run
  // without gravity does not form: beside the load it changes nothing
  {
    Evolution evolution(op, solver, solver.NoOceanState(),
                        {.rotation = RotationalFeedback::None});
    const auto layered = evolution.Run(
        times, Evolution::Forcing{.directLoad = load,
                                  .solidSurfaceChange = [&](double t) {
                                    auto out = load(t);
                                    for (auto& v : out) v *= 1e-3;
                                    return out;
                                  }});
    Evolution plain(op, solver, solver.NoOceanState(),
                    {.rotation = RotationalFeedback::None});
    const auto alone = plain.Run(times, Evolution::Forcing{.directLoad = load});
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      EXPECT_EQ(layered[n].displacement, alone[n].displacement) << n;
    }
    EXPECT_THROW((void)evolution.Run(
                     times, Evolution::Forcing{.directLoad = load,
                                               .solidSurfaceChange =
                                                   [](double) {
                                                     return std::vector<double>(
                                                         3, 0.0);
                                                   }}),
                 std::invalid_argument);
  }
  // the factored sources take no load
  Evolution factored(
      op, solver, solver.NoOceanState(),
      {.rotation = RotationalFeedback::None, .factor_sources = true});
  EXPECT_THROW(
      (void)factored.Run(times, Evolution::Forcing{.directLoad = load,
                                                   .sources = sources}),
      std::invalid_argument);
}

TEST(WithoutGravity, ALoadWithADegreeOnePartIsRefused) {
  // A degree-1 load exerts a net force on the body, which nothing balances
  // without gravity: a load whose degree-1 part exceeds 1e-10 of its size
  // is refused, by the evolution and by the operator's channel; a pair of
  // antipodal caps, and a load whose degree-1 part is at rounding, pass.
  const auto earth = Earth::Make(false, kEarthLmax);
  const auto solver = GRDSolver(SurfaceGrid(kEarthLmax),
                                ComputeLoveNumbers(earth.material, kEarthLmax),
                                SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const ViscoelasticOperator op(earth.rheology, kEarthLmax,
                                {.gravitating = false});
  const auto times = std::vector<double>{0.0, 1.0};
  auto runUnder = [&](const std::vector<double>& field) {
    Evolution evolution(op, solver, solver.NoOceanState(),
                        {.rotation = RotationalFeedback::None});
    return evolution.Run(
        times, Evolution::Forcing{.directLoad = [&](double) { return field; }});
  };
  auto expectRefused = [&](const std::vector<double>& field,
                           const char* what) {
    try {
      static_cast<void>(runUnder(field));
      ADD_FAILURE() << what << " was not refused";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string{error.what()}.find("degree-1 part"),
                std::string::npos)
          << error.what();
    }
  };
  // a cap at the pole, and one degree-1 coefficient
  auto cap = std::vector<double>{};
  auto pair = std::vector<double>{};
  for (const auto theta : grid.CoLatitudes()) {
    for (auto k = 0; k < grid.NumberOfLongitudes(); ++k) {
      const auto north = std::exp(-theta * theta / (2.0 * 0.04));
      const auto south = std::exp(-(std::numbers::pi - theta) *
                                  (std::numbers::pi - theta) / (2.0 * 0.04));
      cap.push_back(north);
      pair.push_back(north + south);
    }
  }
  EXPECT_GT(DegreeOneShare(grid.Expand(cap), kEarthLmax), 0.1);
  expectRefused(cap, "a cap");
  auto coefficients = LoadWithoutDegreeOne(grid);
  auto tilted = coefficients;
  tilted[SurfaceGrid::Index(1, 1)] = Complex(1e-6, 0.0);
  expectRefused(grid.Synthesise(tilted), "a degree-1 coefficient");
  // the antipodal pair and a degree-1 part at rounding pass
  const auto paired = DegreeOneShare(grid.Expand(pair), kEarthLmax);
  RecordProperty("antipodal_pair_share", std::format("{:.3e}", paired));
  EXPECT_LT(paired, kDegreeOneTolerance);
  EXPECT_EQ(runUnder(pair).size(), times.size());
  auto rounding = coefficients;
  rounding[SurfaceGrid::Index(1, 0)] = Complex(1e-15, 0.0);
  EXPECT_EQ(runUnder(grid.Synthesise(rounding)).size(), times.size());
  // and taken as none: the run is the run without it
  {
    const auto with = runUnder(grid.Synthesise(rounding));
    const auto without = runUnder(grid.Synthesise(coefficients));
    EXPECT_LT(Departure(grid, with.back().displacement,
                        without.back().displacement, 0),
              1e-13);
  }

  // the operator's load channel: degree-1 columns refused, zero ones taken
  const auto state = op.ZeroState([](int l) { return 2 * (l + 1); });
  const LoadCallable zero = [](int l, double) {
    return std::optional<std::vector<double>>(
        std::vector<double>(2 * (l + 1), 0.0));
  };
  const LoadCallable tilt = [](int l, double) {
    auto columns = std::vector<double>(2 * (l + 1), 0.0);
    if (l == 1) columns[0] = 1.0;
    return std::optional<std::vector<double>>(std::move(columns));
  };
  EXPECT_NO_THROW((void)op.SolveElastic(state, zero));
  EXPECT_THROW((void)op.SolveElastic(state, tilt), std::invalid_argument);
}


TEST(WithoutGravity, WhatNeedsGravityIsRefused) {
  const auto earth = Earth::Make(false, kEarthLmax);
  const auto solver = GRDSolver(SurfaceGrid(kEarthLmax),
                                ComputeLoveNumbers(earth.material, kEarthLmax),
                                SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const ViscoelasticOperator op(earth.rheology, kEarthLmax,
                                {.gravitating = false});
  const auto source = ThrustIn(solver, earth.material);
  const auto sources = [&](double) { return std::vector<PointSource>{source}; };
  const auto times = std::vector<double>{0.0, 1.0};
  const auto dry = solver.NoOceanState();
  auto zero = [&](double) {
    return std::vector<double>(grid.FieldSize(), 0.0);
  };

  // the operator: the Love numbers and the tide channel (the load channel
  // is a load's weight, ALoadWithADegreeOnePartIsRefused)
  EXPECT_THROW((void)op.ElasticLoveNumbers(), std::invalid_argument);
  EXPECT_THROW((void)op.EffectiveLoveNumbers(Scheme::ExponentialTrapezoid, 0.1),
               std::invalid_argument);
  const LoadCallable channel = [](int l, double) {
    return std::optional<std::vector<double>>(
        std::vector<double>(2 * (l + 1), 0.0));
  };
  const auto state = op.ZeroState([](int l) { return 2 * (l + 1); });
  EXPECT_THROW((void)op.SolveElastic(state, {}, channel),
               std::invalid_argument);
  EXPECT_THROW(
      (void)op.StartStep(state, 0.1, Scheme::ExponentialTrapezoid, channel)
          .Finish(channel, channel),
      std::invalid_argument);

  // the evolution: the rotational feedback, an ocean, a tide and the
  // migrating runs
  {
    Evolution evolution(op, solver, dry, {});  // RotationalFeedback::Full
    EXPECT_THROW(
        (void)evolution.Run(times, Evolution::Forcing{.sources = sources}),
        std::invalid_argument);
    EXPECT_THROW(
        (void)evolution.Run(times, Evolution::Forcing{.directLoad = zero}),
        std::invalid_argument);
  }
  Evolution evolution(op, solver, dry, {.rotation = RotationalFeedback::None});
  EXPECT_THROW(
      (void)evolution.Run(
          times,
          Evolution::Forcing{
              .tide = [&](double) { return grid.ZeroCoefficients(); }}),
      std::invalid_argument);
  EXPECT_THROW((void)evolution.RunMigrating(times, zero),
               std::invalid_argument);
  {
    // a body with an ocean everywhere
    const auto wet =
        solver.MakeState(std::vector<double>(grid.FieldSize(), 0.0),
                         std::vector<double>(grid.FieldSize(), 1.0));
    ASSERT_GT(wet.oceanArea, 0.0);
    Evolution oceanic(op, solver, wet, {.rotation = RotationalFeedback::None});
    EXPECT_THROW(
        (void)oceanic.Run(times, Evolution::Forcing{.sources = sources}),
        std::invalid_argument);
    EXPECT_THROW((void)oceanic.Run(times, zero), std::invalid_argument);
  }
  // the derivative solvers
  EXPECT_THROW((void)AdjointSolver(evolution), std::invalid_argument);
  EXPECT_THROW((void)TangentLinearSolver(evolution), std::invalid_argument);
  // and runs under the sources, the loads (inside and outside the water
  // budget alike, with a layer) and the direct load go
  const auto run = evolution.Run(times, Evolution::Forcing{.sources = sources});
  EXPECT_EQ(run.size(), times.size());
  EXPECT_EQ(evolution.Run(times, zero).size(), times.size());
  EXPECT_EQ(evolution
                .Run(times, Evolution::Forcing{.directLoad = zero,
                                               .directLoadOutsideWaterBudget =
                                                   zero,
                                               .solidSurfaceChange = zero,
                                               .sources = sources})
                .size(),
            times.size());
}
