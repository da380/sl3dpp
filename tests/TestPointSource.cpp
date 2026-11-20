// The point source: the strain of degree solutions at a point against
// displacement fields whose strain is known (a uniform strain, a rigid
// rotation), and the source's forces as the transpose of that
// evaluation.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/ToroidalOperator.hpp"

using namespace sl3dpp;

namespace {

using Complex = std::complex<double>;
constexpr auto kLmax = 6;

struct Body {
  EarthModel model;
  Material material;
  std::map<int, std::unique_ptr<DegreeOperator>> spheroidal;
  std::map<int, std::unique_ptr<ToroidalOperator>> toroidal;

  static Body Make() {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
    auto material = Material::Create(model, mesh).value();
    return Body{std::move(model), std::move(material), {}, {}};
  }
  void Build() {
    for (auto l = 0; l <= kLmax; ++l) {
      spheroidal.emplace(l, std::make_unique<DegreeOperator>(
                                material, l, DegreeOperator::Options{}));
      if (l >= 1) {
        toroidal.emplace(l, std::make_unique<ToroidalOperator>(
                                material, l, ToroidalOperator::Options{}));
      }
    }
  }
  [[nodiscard]] std::function<const DegreeOperator&(int)> Sph() const {
    return [this](int l) -> const DegreeOperator& { return *spheroidal.at(l); };
  }
  [[nodiscard]] std::function<const ToroidalOperator&(int)> Tor() const {
    return [this](int l) -> const ToroidalOperator& { return *toroidal.at(l); };
  }
  /// A radius in the solid mantle, inside an element.
  [[nodiscard]] double MantleRadius() const {
    const auto& mesh = material.Mesh();
    for (auto e = mesh.ElementCount() - 2; e >= 0; --e) {
      if (!material.IsFluid(e)) {
        return 0.3 * mesh.Radius(e, 0) + 0.7 * mesh.Radius(e, mesh.NGll() - 1);
      }
    }
    return 0.0;
  }
};

/// Degree solutions whose unknown `which` (0 U, 1 V, 2 W) at degree l is
/// r times the given coefficients (m >= 0), added to `x`.
void AddLinearInRadius(const Body& body, DegreeFields& x, int which, int l,
                       const std::vector<Complex>& coefficients) {
  const auto& mesh = body.material.Mesh();
  const auto toroidal = which == 2;
  const auto ndof = static_cast<std::size_t>(
      toroidal ? body.toroidal.at(l)->DofCount()
               : body.spheroidal.at(l)->DofCount());
  auto& column = toroidal ? x.toroidal[l] : x.spheroidal[l];
  if (column.empty()) column.assign(ndof * (l + 1), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto g = mesh.GlobalIndex(e, q);
      const auto dof = toroidal
                           ? body.toroidal.at(l)->Dof(g)
                           : body.spheroidal.at(l)->Dof(
                                 g, which == 0 ? Component::U : Component::V);
      if (dof < 0) continue;
      for (auto m = 0; m <= l; ++m) {
        column[static_cast<std::size_t>(m) * ndof + dof] =
            mesh.Radius(e, q) * coefficients[SurfaceGrid::Index(l, m)];
      }
    }
  }
}

using Vector = std::array<double, 3>;
using Matrix = std::array<Vector, 3>;

double Quadratic(const Vector& a, const Matrix& E, const Vector& b) {
  auto out = 0.0;
  for (auto i = 0; i < 3; ++i) {
    for (auto j = 0; j < 3; ++j) out += a[i] * E[i][j] * b[j];
  }
  return out;
}

}  // namespace

TEST(PointSource, TheStrainOfAUniformStrainAndOfARotation) {
  // u = E x for a constant symmetric E is the gradient of r^2 e / 2 with
  // e = r . E . r on the unit sphere: U = r e and V = r e / 2 (degrees 0
  // and 2), and its strain is E everywhere. A rigid rotation, W linear
  // in r at degree 1, has none.
  auto body = Body::Make();
  body.Build();
  const SurfaceGrid grid(kLmax);
  const Matrix E{{{0.7, -0.3, 0.5}, {-0.3, -1.1, 0.2}, {0.5, 0.2, 0.4}}};
  const auto e = grid.Expand(grid.FromFunction([&](double theta, double phi) {
    const Vector r{std::sin(theta) * std::cos(phi),
                   std::sin(theta) * std::sin(phi), std::cos(theta)};
    return Quadratic(r, E, r);
  }));
  auto half = e;
  for (auto& c : half) c *= 0.5;
  auto x = DegreeFields{};
  AddLinearInRadius(body, x, 0, 0, e);
  AddLinearInRadius(body, x, 0, 2, e);
  AddLinearInRadius(body, x, 1, 2, half);
  // and a rigid rotation on top
  auto spin = grid.ZeroCoefficients();
  spin[SurfaceGrid::Index(1, 0)] = 0.8;
  spin[SurfaceGrid::Index(1, 1)] = Complex(-0.4, 0.6);
  AddLinearInRadius(body, x, 2, 1, spin);

  for (const auto& [theta, phi] :
       {std::pair{0.9, 2.3}, std::pair{2.4, 5.1}, std::pair{1.57, 0.2}}) {
    const auto strain =
        PointStrain(body.material, kLmax, body.Sph(), body.Tor(), x,
                    {.radius = body.MantleRadius(),
                     .colatitude = theta,
                     .longitude = phi});
    const Vector r{std::sin(theta) * std::cos(phi),
                   std::sin(theta) * std::sin(phi), std::cos(theta)};
    const Vector t{std::cos(theta) * std::cos(phi),
                   std::cos(theta) * std::sin(phi), -std::sin(theta)};
    const Vector p{-std::sin(phi), std::cos(phi), 0.0};
    EXPECT_NEAR(strain.rr, Quadratic(r, E, r), 1e-10);
    EXPECT_NEAR(strain.tt, Quadratic(t, E, t), 1e-10);
    EXPECT_NEAR(strain.pp, Quadratic(p, E, p), 1e-10);
    EXPECT_NEAR(strain.rt, Quadratic(r, E, t), 1e-10);
    EXPECT_NEAR(strain.rp, Quadratic(r, E, p), 1e-10);
    EXPECT_NEAR(strain.tp, Quadratic(t, E, p), 1e-10);
  }
}

TEST(PointSource, TheForcesAreTheTransposeOfTheStrain) {
  auto body = Body::Make();
  body.Build();
  std::mt19937 rng(3);
  std::normal_distribution<double> normal;
  auto x = DegreeFields{};
  for (auto l = 0; l <= kLmax; ++l) {
    auto fill = [&](std::vector<Complex>& column, std::size_t ndof) {
      column.resize(ndof * (l + 1));
      for (auto m = 0; m <= l; ++m) {
        for (auto d = std::size_t{0}; d < ndof; ++d) {
          column[m * ndof + d] = {normal(rng), m == 0 ? 0.0 : normal(rng)};
        }
      }
    };
    fill(x.spheroidal[l], body.spheroidal.at(l)->DofCount());
    if (l >= 1) fill(x.toroidal[l], body.toroidal.at(l)->DofCount());
  }
  const PointSource source{
      .at = {.radius = body.MantleRadius(), .colatitude = 1.1, .longitude = 4.0},
      .moment = {.rr = 0.9, .tt = -0.2, .pp = 0.5, .rt = 0.3, .rp = -0.7, .tp = 0.6}};
  const auto forces = PointSourceForces(body.material, kLmax, body.Sph(),
                                        body.Tor(), source);
  auto paired = 0.0;
  auto pair = [&](const std::vector<Complex>& b, const std::vector<Complex>& v,
                  int l, std::size_t ndof) {
    for (auto m = 0; m <= l; ++m) {
      for (auto d = std::size_t{0}; d < ndof; ++d) {
        paired += (m == 0 ? 1.0 : 2.0) *
                  (std::conj(b[m * ndof + d]) * v[m * ndof + d]).real();
      }
    }
  };
  for (auto l = 0; l <= kLmax; ++l) {
    pair(forces.spheroidal.at(l), x.spheroidal.at(l), l,
         body.spheroidal.at(l)->DofCount());
    if (l >= 1) {
      pair(forces.toroidal.at(l), x.toroidal.at(l), l,
           body.toroidal.at(l)->DofCount());
    }
  }
  const auto strain = PointStrain(body.material, kLmax, body.Sph(), body.Tor(),
                                  x, source.at);
  const auto direct = source.moment.Contract(strain);
  EXPECT_GT(std::abs(direct), 1e-3);
  EXPECT_NEAR(paired, direct, 1e-11 * std::abs(direct));
}

TEST(PointSource, AStressGlutIsTheIntegralOfPointSources) {
  // A tensor field on a shell as a source: its forces paired with any
  // solutions are the integral over the shell's directions of the moment
  // contracted with the solutions' strain there. The moment is itself a
  // band-limited tensor field, the strain of other solutions, so the
  // grid's quadrature of the scalar integrand is exact.
  auto body = Body::Make();
  body.Build();
  std::mt19937 rng(11);
  std::normal_distribution<double> normal;
  constexpr auto kBand = 3;
  auto random = [&] {
    auto x = DegreeFields{};
    for (auto l = 0; l <= kBand; ++l) {
      auto fill = [&](std::vector<Complex>& column, std::size_t ndof) {
        column.resize(ndof * (l + 1));
        for (auto m = 0; m <= l; ++m) {
          for (auto d = std::size_t{0}; d < ndof; ++d) {
            column[m * ndof + d] = {normal(rng), m == 0 ? 0.0 : normal(rng)};
          }
        }
      };
      fill(x.spheroidal[l], body.spheroidal.at(l)->DofCount());
      if (l >= 1) fill(x.toroidal[l], body.toroidal.at(l)->DofCount());
    }
    return x;
  };
  const auto x = random();
  const auto y = random();
  const SurfaceGrid grid(8, {.nmax = 2});
  const auto radius = body.MantleRadius();
  const auto ntheta = grid.CoLatitudes().size();
  const auto nphi = grid.Longitudes().size();
  auto shell = StressGlut{.radius = radius, .grid = &grid, .moment = {}};
  auto integrand = std::vector<double>(grid.FieldSize());
  for (auto it = std::size_t{0}; it < ntheta; ++it) {
    for (auto ip = std::size_t{0}; ip < nphi; ++ip) {
      const BodyPoint at{.radius = radius,
                         .colatitude = grid.CoLatitudes()[it],
                         .longitude = grid.Longitudes()[ip]};
      const auto S =
          PointStrain(body.material, kLmax, body.Sph(), body.Tor(), y, at);
      const auto e =
          PointStrain(body.material, kLmax, body.Sph(), body.Tor(), x, at);
      shell.moment.rr.push_back(S.rr);
      shell.moment.tt.push_back(S.tt);
      shell.moment.pp.push_back(S.pp);
      shell.moment.rt.push_back(S.rt);
      shell.moment.rp.push_back(S.rp);
      shell.moment.tp.push_back(S.tp);
      integrand[it * nphi + ip] = S.Contract(e);
    }
  }
  const auto direct = grid.Integrate(integrand);
  const auto forces = StressGlutForces(body.material, kLmax, body.Sph(),
                                        body.Tor(), shell);
  auto paired = 0.0;
  auto pair = [&](const std::vector<Complex>& b, const std::vector<Complex>& v,
                  int l, std::size_t ndof) {
    for (auto m = 0; m <= l; ++m) {
      for (auto d = std::size_t{0}; d < ndof; ++d) {
        paired += (m == 0 ? 1.0 : 2.0) *
                  (std::conj(b[m * ndof + d]) * v[m * ndof + d]).real();
      }
    }
  };
  for (auto l = 0; l <= kBand; ++l) {
    pair(forces.spheroidal.at(l), x.spheroidal.at(l), l,
         body.spheroidal.at(l)->DofCount());
    if (l >= 1) {
      pair(forces.toroidal.at(l), x.toroidal.at(l), l,
           body.toroidal.at(l)->DofCount());
    }
  }
  EXPECT_GT(std::abs(direct), 1e-3);
  EXPECT_NEAR(paired, direct, 1e-10 * std::abs(direct));
}

#include "sl3dpp/Fault.hpp"

TEST(Fault, TheDoubleCoupleIsAkiAndRichards) {
  // The moment tensor of a strike, dip and rake in (r, theta, phi)
  // against the standard expressions (Aki & Richards 2002, box 4.4, in
  // the up-south-east components of the CMT convention).
  for (const auto& [strikeDeg, dipDeg, rakeDeg] :
       {std::array{30.0, 60.0, 90.0}, std::array{210.0, 25.0, -40.0},
        std::array{0.0, 90.0, 0.0}, std::array{135.0, 45.0, 170.0}}) {
    const auto s = Radians(strikeDeg), d = Radians(dipDeg), l = Radians(rakeDeg);
    const auto M = DoubleCouple(strikeDeg, dipDeg, rakeDeg);
    EXPECT_NEAR(M.rr, std::sin(2 * d) * std::sin(l), 1e-14);
    EXPECT_NEAR(M.tt,
                -(std::sin(d) * std::cos(l) * std::sin(2 * s) +
                  std::sin(2 * d) * std::sin(l) * std::sin(s) * std::sin(s)),
                1e-14);
    EXPECT_NEAR(M.pp,
                std::sin(d) * std::cos(l) * std::sin(2 * s) -
                    std::sin(2 * d) * std::sin(l) * std::cos(s) * std::cos(s),
                1e-14);
    EXPECT_NEAR(M.rt,
                -(std::cos(d) * std::cos(l) * std::cos(s) +
                  std::cos(2 * d) * std::sin(l) * std::sin(s)),
                1e-14);
    EXPECT_NEAR(M.rp,
                std::cos(d) * std::cos(l) * std::sin(s) -
                    std::cos(2 * d) * std::sin(l) * std::cos(s),
                1e-14);
    EXPECT_NEAR(M.tp,
                -(std::sin(d) * std::cos(l) * std::cos(2 * s) +
                  0.5 * std::sin(2 * d) * std::sin(l) * std::sin(2 * s)),
                1e-14);
  }
}

TEST(Fault, APlanarFaultIsItsPatches) {
  // The patches lie in one plane about the centroid, carry equal shares
  // of the moment, and each has the fault's tensor in its own basis; one
  // patch is the double couple at the centroid; patches above the
  // surface are left out.
  const auto surface = 6371.0;
  const PlanarFault fault{.centroid = {35.0, 140.0},
                          .depth = 30.0,
                          .strike = 200.0,
                          .dip = 35.0,
                          .rake = 80.0,
                          .length = 300.0,
                          .width = 80.0,
                          .moment = 4.0};
  const auto one = PlanarFaultSources(fault, surface, 1, 1);
  ASSERT_EQ(one.size(), 1u);
  EXPECT_NEAR(one[0].at.radius, surface - fault.depth, 1e-9);
  EXPECT_NEAR(one[0].at.colatitude, fault.centroid.CoLatitude(), 1e-12);
  EXPECT_NEAR(one[0].at.longitude, fault.centroid.Longitude(), 1e-12);
  const auto unit = DoubleCouple(fault.strike, fault.dip, fault.rake);
  EXPECT_NEAR(one[0].moment.rr, fault.moment * unit.rr, 1e-13);
  EXPECT_NEAR(one[0].moment.tp, fault.moment * unit.tp, 1e-13);

  const auto patches = PlanarFaultSources(fault, surface, 12, 4);
  ASSERT_EQ(patches.size(), 48u);
  using Vector = std::array<double, 3>;
  auto position = [](const BodyPoint& p) {
    return Vector{p.radius * std::sin(p.colatitude) * std::cos(p.longitude),
                  p.radius * std::sin(p.colatitude) * std::sin(p.longitude),
                  p.radius * std::cos(p.colatitude)};
  };
  // the tensor in space from a patch's components
  auto inSpace = [](const PointSource& s) {
    const auto t = s.at.colatitude, p = s.at.longitude;
    const std::array<Vector, 3> e{
        Vector{std::sin(t) * std::cos(p), std::sin(t) * std::sin(p), std::cos(t)},
        Vector{std::cos(t) * std::cos(p), std::cos(t) * std::sin(p), -std::sin(t)},
        Vector{-std::sin(p), std::cos(p), 0.0}};
    const double c[3][3] = {{s.moment.rr, s.moment.rt, s.moment.rp},
                            {s.moment.rt, s.moment.tt, s.moment.tp},
                            {s.moment.rp, s.moment.tp, s.moment.pp}};
    auto M = std::array<Vector, 3>{};
    for (auto a = 0; a < 3; ++a) {
      for (auto b = 0; b < 3; ++b) {
        for (auto i = 0; i < 3; ++i) {
          for (auto j = 0; j < 3; ++j) M[i][j] += c[a][b] * e[a][i] * e[b][j];
        }
      }
    }
    return M;
  };
  const auto reference = inSpace(one[0]);
  const auto centre = position(one[0].at);
  auto mean = Vector{};
  // the plane's normal from two in-plane offsets
  const auto p0 = position(patches.front().at);
  const auto p1 = position(patches[3].at);
  const auto p2 = position(patches.back().at);
  const Vector u{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
  const Vector v{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
  const Vector n{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                 u[0] * v[1] - u[1] * v[0]};
  const auto nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  for (const auto& patch : patches) {
    const auto x = position(patch.at);
    for (auto i = 0; i < 3; ++i) mean[i] += x[i] / patches.size();
    const auto off = ((x[0] - centre[0]) * n[0] + (x[1] - centre[1]) * n[1] +
                      (x[2] - centre[2]) * n[2]) /
                     nn;
    EXPECT_NEAR(off, 0.0, 1e-8);
    const auto M = inSpace(patch);
    for (auto i = 0; i < 3; ++i) {
      for (auto j = 0; j < 3; ++j) {
        EXPECT_NEAR(M[i][j], reference[i][j] / 48.0, 1e-12);
      }
    }
  }
  for (auto i = 0; i < 3; ++i) EXPECT_NEAR(mean[i], centre[i], 1e-8);

  // a steep fault whose top would stand above the surface
  auto shallow = fault;
  shallow.depth = 10.0;
  shallow.dip = 90.0;
  shallow.width = 60.0;
  const auto kept = PlanarFaultSources(shallow, surface, 3, 6);
  EXPECT_LT(kept.size(), 18u);
  EXPECT_GT(kept.size(), 0u);
  for (const auto& patch : kept) EXPECT_LE(patch.at.radius, surface);
}

#include "sl3dpp/Measurements.hpp"

TEST(PointSource, AHomogeneousSphereWithoutGravityInClosedForm) {
  // The benchmark of the source: a point moment tensor in a homogeneous
  // elastic sphere without gravity, whose surface displacement is known
  // in closed form. The full-space solution (Kelvin's tensor,
  // u_i = M_jk dG_ij / d xi_k) is corrected by the regular solutions of
  // the sphere that cancel its traction on the surface, degree by
  // degree: the spheroidal pair grad(r^l Y) and r^2 grad(r^l Y) +
  // alpha_l x r^l Y, alpha_l = -2 (l lam + (3l + 1) mu) / ((l + 3) lam +
  // (l + 5) mu), and the toroidal r^l. The code's side is the source's
  // forces on the non-gravitating degree systems, solved. At degree 1
  // the regular pair is the translation grad(r Y), which bears no
  // traction, and r^2 grad(r Y) + alpha_1 x r Y, which cancels the
  // full-space traction alone (its net force is zero); the translation is
  // then set by the frame of the centre of mass, int rho u dV = 0, the
  // full-space part's integral over the ball being
  // -M_jk oint G_ij(x - xi) n_k dS by the divergence theorem. The code's
  // degree 1 is checked besides for its frame to rounding, for not
  // depending on the pinned dof, and for solving the full (unpinned)
  // system. The toroidal degree 1 is a rigid rotation's gauge and is not
  // compared. The source is at half the radius, so the surface fields are
  // band-limited to rounding at the grid's degree.
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  auto deck = Deck::Read(data / "e1_homogeneous.deck",
                         DeckFormat("elastic",
                                    {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                                    0))
                  .value();
  auto model = EarthModel::FromDeck(deck).value();
  constexpr auto kDegree = 16;  // compared to here
  constexpr auto kGrid = 48;    // the surface fields' transforms
  auto mesh = RadialMesh::Create(model, {.lmax = kDegree}).value();
  const auto material = Material::Create(model, mesh).value();
  const auto a = material.Radius();
  const auto [kappas, mus] = material.Moduli().KappaMu();
  const auto mu = mus.front();
  const auto lam = kappas.front() - 2.0 * mu / 3.0;
  const auto nu = lam / (2.0 * (lam + mu));

  const PointSource source{
      .at = {.radius = 0.5 * a, .colatitude = 1.1, .longitude = 4.0},
      .moment = {.rr = 0.9, .tt = -0.2, .pp = 0.5, .rt = 0.3, .rp = -0.7, .tp = 0.6}};

  // -- the code: forces on the non-gravitating systems, solved
  auto spheroidal = std::map<int, std::unique_ptr<DegreeOperator>>{};
  auto toroidal = std::map<int, std::unique_ptr<ToroidalOperator>>{};
  for (auto l = 0; l <= kDegree; ++l) {
    spheroidal.emplace(
        l, std::make_unique<DegreeOperator>(
               material, l, DegreeOperator::Options{.gravitating = false}));
    if (l >= 1) {
      toroidal.emplace(l, std::make_unique<ToroidalOperator>(
                              material, l, ToroidalOperator::Options{}));
    }
  }
  const auto forces = PointSourceForces(
      material, kDegree,
      [&](int l) -> const DegreeOperator& { return *spheroidal.at(l); },
      [&](int l) -> const ToroidalOperator& { return *toroidal.at(l); },
      source);
  const auto size = SurfaceGrid::Index(kDegree, kDegree) + 1;
  auto U = std::vector<Complex>(size, 0.0), V = U, W = U;
  auto solve = [](const auto& system, const std::vector<Complex>& b, int l) {
    auto re = std::vector<double>(b.size()), im = re;
    for (auto i = std::size_t{0}; i < b.size(); ++i) {
      re[i] = b[i].real();
      im[i] = b[i].imag();
    }
    const auto xr = system.Solve(re, l + 1);
    const auto xi = system.Solve(im, l + 1);
    auto x = std::vector<Complex>(b.size());
    for (auto i = std::size_t{0}; i < b.size(); ++i) x[i] = {xr[i], xi[i]};
    return x;
  };
  for (auto l = 0; l <= kDegree; ++l) {
    const auto& system = *spheroidal.at(l);
    const auto ndof = static_cast<std::size_t>(system.DofCount());
    const auto x = solve(system, forces.spheroidal.at(l), l);
    for (auto m = 0; m <= l; ++m) {
      const auto i = SurfaceGrid::Index(l, m);
      U[i] = x[m * ndof + system.SurfaceDof(Component::U)];
      if (l > 0) V[i] = x[m * ndof + system.SurfaceDof(Component::V)];
    }
    if (l >= 2) {
      const auto& twist = *toroidal.at(l);
      const auto tdof = static_cast<std::size_t>(twist.DofCount());
      const auto w = solve(twist, forces.toroidal.at(l), l);
      for (auto m = 0; m <= l; ++m) {
        W[SurfaceGrid::Index(l, m)] = w[m * tdof + twist.SurfaceDof()];
      }
    }
  }

  // -- degree 1 on its own: the frame, the gauge and the residual
  {
    const auto& b = forces.spheroidal.at(1);
    auto realPart = [](const std::vector<Complex>& v) {
      auto out = std::vector<double>(v.size());
      for (auto i = std::size_t{0}; i < v.size(); ++i) out[i] = v[i].real();
      return out;
    };
    auto imagPart = [](const std::vector<Complex>& v) {
      auto out = std::vector<double>(v.size());
      for (auto i = std::size_t{0}; i < v.size(); ++i) out[i] = v[i].imag();
      return out;
    };
    // the real and imaginary columns of orders 0 and 1
    auto rhs = realPart(b);
    const auto im = imagPart(b);
    rhs.insert(rhs.end(), im.begin(), im.end());
    const auto& base = *spheroidal.at(1);
    const auto n = static_cast<std::size_t>(base.DofCount());
    const auto translation = base.RigidTranslation();
    const auto row = base.CentreOfMassRow();
    auto worstForce = 0.0, worstFrame = 0.0, worstResidual = 0.0,
         worstGauge = 0.0;
    const auto x0 = base.Solve(rhs, 4);
    for (const auto fraction : {0.0, 0.37, 0.8}) {
      const DegreeOperator pinned(
          material, 1, {.gravitating = false, .pin_fraction = fraction});
      const auto x = pinned.Solve(rhs, 4);
      const auto Ax = pinned.Apply(x, 4);
      for (auto c = std::size_t{0}; c < 4; ++c) {
        auto force = 0.0, forceScale = 0.0, moment = 0.0, momentScale = 0.0;
        auto residual = 0.0, rhsScale = 0.0, gauge = 0.0, size = 0.0;
        for (auto d = std::size_t{0}; d < n; ++d) {
          const auto k = c * n + d;
          force += translation[d] * rhs[k];
          forceScale += std::abs(translation[d] * rhs[k]);
          moment += row[d] * x[k];
          momentScale += std::abs(row[d] * x[k]);
          residual = std::max(residual, std::abs(Ax[k] - rhs[k]));
          rhsScale = std::max(rhsScale, std::abs(rhs[k]));
          gauge = std::max(gauge, std::abs(x[k] - x0[k]));
          size = std::max(size, std::abs(x0[k]));
        }
        if (forceScale > 0.0) {
          worstForce = std::max(worstForce, std::abs(force) / forceScale);
        }
        if (momentScale > 0.0) {
          worstFrame = std::max(worstFrame, std::abs(moment) / momentScale);
        }
        if (rhsScale > 0.0) {
          worstResidual = std::max(worstResidual, residual / rhsScale);
        }
        if (size > 0.0) worstGauge = std::max(worstGauge, gauge / size);
      }
    }
    RecordProperty("degree1_net_force", std::format("{:.3e}", worstForce));
    RecordProperty("degree1_frame", std::format("{:.3e}", worstFrame));
    RecordProperty("degree1_residual", std::format("{:.3e}", worstResidual));
    RecordProperty("degree1_gauge", std::format("{:.3e}", worstGauge));
    // (rounding: the frame 5e-15 of its terms here, the gauge 1.4e-11)
    EXPECT_LT(worstForce, 1e-12);
    EXPECT_LT(worstFrame, 1e-12);
    EXPECT_LT(worstResidual, 1e-10);
    EXPECT_LT(worstGauge, 1e-9);
  }

  // -- the closed form
  using Vector = std::array<double, 3>;
  struct Basis {
    Vector r, t, p;
    Basis(double theta, double phi)
        : r{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
            std::cos(theta)},
          t{std::cos(theta) * std::cos(phi), std::cos(theta) * std::sin(phi),
            -std::sin(theta)},
          p{-std::sin(phi), std::cos(phi), 0.0} {}
  };
  const Basis at(source.at.colatitude, source.at.longitude);
  const Vector xi{source.at.radius * at.r[0], source.at.radius * at.r[1],
                  source.at.radius * at.r[2]};
  // the moment tensor in space
  double M[3][3] = {};
  {
    const Vector* e[3] = {&at.r, &at.t, &at.p};
    const auto& S = source.moment;
    const double c[3][3] = {
        {S.rr, S.rt, S.rp}, {S.rt, S.tt, S.tp}, {S.rp, S.tp, S.pp}};
    for (auto p = 0; p < 3; ++p) {
      for (auto q = 0; q < 3; ++q) {
        for (auto i = 0; i < 3; ++i) {
          for (auto j = 0; j < 3; ++j) M[i][j] += c[p][q] * (*e[p])[i] * (*e[q])[j];
        }
      }
    }
  }
  // Kelvin: G_ij = ((3 - 4 nu) delta_ij / R + R_i R_j / R^3) / (16 pi mu (1 - nu)),
  // and u_i = M_jk dG_ij / d xi_k = -M_jk d_k G_ij.
  auto fullSpace = [&](const Vector& x) {
    const Vector R{x[0] - xi[0], x[1] - xi[1], x[2] - xi[2]};
    const auto r2 = R[0] * R[0] + R[1] * R[1] + R[2] * R[2];
    const auto r1 = std::sqrt(r2);
    const auto c = 1.0 / (16.0 * std::numbers::pi * mu * (1.0 - nu));
    auto u = Vector{};
    for (auto i = 0; i < 3; ++i) {
      for (auto j = 0; j < 3; ++j) {
        for (auto k = 0; k < 3; ++k) {
          const auto dij = i == j ? 1.0 : 0.0;
          const auto dik = i == k ? 1.0 : 0.0;
          const auto djk = j == k ? 1.0 : 0.0;
          const auto dG =
              c * (-(3.0 - 4.0 * nu) * dij * R[k] / (r2 * r1) +
                   (dik * R[j] + djk * R[i]) / (r2 * r1) -
                   3.0 * R[i] * R[j] * R[k] / (r2 * r2 * r1));
          u[i] -= M[j][k] * dG;
        }
      }
    }
    return u;
  };
  const SurfaceGrid grid(kGrid, {.nmax = 1});
  auto uField = SurfaceVectorField{}, tField = SurfaceVectorField{};
  const auto h = 1e-6 * a;
  for (auto theta : grid.CoLatitudes()) {
    for (auto phi : grid.Longitudes()) {
      const Basis here(theta, phi);
      const Vector x{a * here.r[0], a * here.r[1], a * here.r[2]};
      const auto u = fullSpace(x);
      // the strain by central differences, then the traction on r
      double grad[3][3];
      for (auto k = 0; k < 3; ++k) {
        auto plus = x, minus = x;
        plus[k] += h;
        minus[k] -= h;
        const auto up = fullSpace(plus), um = fullSpace(minus);
        for (auto i = 0; i < 3; ++i) grad[i][k] = (up[i] - um[i]) / (2.0 * h);
      }
      const auto trace = grad[0][0] + grad[1][1] + grad[2][2];
      auto t = Vector{};
      for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
          const auto stress = lam * trace * (i == j ? 1.0 : 0.0) +
                              mu * (grad[i][j] + grad[j][i]);
          t[i] += stress * here.r[j];
        }
      }
      auto dot = [](const Vector& p, const Vector& q) {
        return p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
      };
      uField.up.push_back(dot(u, here.r));
      uField.north.push_back(-dot(u, here.t));
      uField.east.push_back(dot(u, here.p));
      tField.up.push_back(dot(t, here.r));
      tField.north.push_back(-dot(t, here.t));
      tField.east.push_back(dot(t, here.p));
    }
  }
  // a vector field's U, V and W coefficients: the duals of V and W over
  // l (l + 1)
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
  const auto uK = coefficientsOf(uField);
  const auto tK = coefficientsOf(tField);

  auto kelvinMoment = std::array<Complex, 2>{};
  // The full-space part's integral over the ball, J_i =
  // -M_jk oint G_ij(x - xi) n_k dS, and its degree-1 radial moments
  // I_m = int (U_1m + 2 V_1m) r^2 dr = E(m) . J / (4 pi / 3), E_c(m) the
  // degree-1 coefficients of the unit vector's Cartesian components.
  auto J = Vector{};
  {
    const auto c = 1.0 / (16.0 * std::numbers::pi * mu * (1.0 - nu));
    std::array<std::vector<double>, 3> integrand;
    std::array<std::vector<double>, 3> unit;
    for (auto theta : grid.CoLatitudes()) {
      for (auto phi : grid.Longitudes()) {
        const Basis here(theta, phi);
        const Vector R{a * here.r[0] - xi[0], a * here.r[1] - xi[1],
                       a * here.r[2] - xi[2]};
        const auto r1 = std::sqrt(R[0] * R[0] + R[1] * R[1] + R[2] * R[2]);
        for (auto i = 0; i < 3; ++i) {
          auto sum = 0.0;
          for (auto j = 0; j < 3; ++j) {
            const auto G = c * ((3.0 - 4.0 * nu) * (i == j ? 1.0 : 0.0) / r1 +
                                R[i] * R[j] / (r1 * r1 * r1));
            for (auto k = 0; k < 3; ++k) sum -= M[j][k] * G * here.r[k];
          }
          integrand[i].push_back(a * a * sum);
          unit[i].push_back(here.r[i]);
        }
      }
    }
    for (auto i = 0; i < 3; ++i) J[i] = grid.Integrate(integrand[i]);
    std::array<std::vector<Complex>, 3> E;
    for (auto i = 0; i < 3; ++i) E[i] = grid.Expand(unit[i]);
    for (auto m = 0; m <= 1; ++m) {
      auto moment = Complex{};
      for (auto i = 0; i < 3; ++i) moment += E[i][SurfaceGrid::Index(1, m)] * J[i];
      kelvinMoment[m] = moment / (4.0 * std::numbers::pi / 3.0);
    }
  }

  auto scale = 0.0, worst = 0.0, worstW = 0.0, scaleW = 0.0;
  auto scale1 = 0.0, worst1 = 0.0, consistency = 0.0;
  for (auto l = 0; l <= kDegree; ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto i = SurfaceGrid::Index(l, m);
      auto exactU = uK.U[i], exactV = uK.V[i], exactW = uK.W[i];
      if (l == 0) {
        // U = r: the traction (3 lam + 2 mu)
        exactU += -tK.U[i] / (3.0 * lam + 2.0 * mu) * a;
      } else if (l == 1) {
        // B = r^2 grad(r Y) + alpha_1 x r Y cancels the traction; the
        // translation A = grad(r Y) (U = V = 1/a) sets the frame
        const auto alpha = -2.0 * (lam + 4.0 * mu) / (4.0 * lam + 6.0 * mu);
        const auto UB = (1.0 + alpha) * a, VB = a, dUB = 2.0 * (1.0 + alpha),
                   dVB = 2.0;
        const auto radialB =
            (lam + 2.0 * mu) * dUB + lam * (2.0 * UB - 2.0 * VB) / a;
        const auto shearB = mu * (dVB - VB / a + UB / a);
        const auto cB = -tK.U[i] / radialB;
        consistency = std::max(consistency,
                               std::abs(cB + tK.V[i] / shearB) / std::abs(cB));
        // int (U + 2 V) r^2 dr: a^2 for A, (3 + alpha) a^4 / 5 for B
        const auto cA = -(kelvinMoment[m] +
                          cB * (3.0 + alpha) * a * a * a * a / 5.0) /
                        (a * a);
        exactU += cA / a + cB * UB;
        exactV += cA / a + cB * VB;
        scale1 = std::max({scale1, std::abs(exactU), std::abs(exactV)});
        worst1 = std::max({worst1, std::abs(U[i] - exactU),
                           std::abs(V[i] - exactV)});
      } else {
        const auto alpha = -2.0 * (l * lam + (3.0 * l + 1.0) * mu) /
                           ((l + 3.0) * lam + (l + 5.0) * mu);
        // with r in units of a: A = grad(r^l Y), B = r^2 grad(r^l Y) + alpha x r^l Y
        const auto UA = l / a, VA = 1.0 / a, dUA = l * (l - 1.0) / (a * a),
                   dVA = (l - 1.0) / (a * a);
        const auto UB = (l + alpha) * a, VB = a, dUB = (l + alpha) * (l + 1.0),
                   dVB = l + 1.0;
        auto radial = [&](double Us, double Vs, double dU) {
          return (lam + 2.0 * mu) * dU + lam * (2.0 * Us - l * (l + 1.0) * Vs) / a;
        };
        auto shear = [&](double Us, double Vs, double dV) {
          return mu * (dV - Vs / a + Us / a);
        };
        const auto a11 = radial(UA, VA, dUA), a12 = radial(UB, VB, dUB);
        const auto a21 = shear(UA, VA, dVA), a22 = shear(UB, VB, dVB);
        const auto det = a11 * a22 - a12 * a21;
        const auto cA = (-tK.U[i] * a22 + tK.V[i] * a12) / det;
        const auto cB = (-tK.V[i] * a11 + tK.U[i] * a21) / det;
        exactU += cA * UA + cB * UB;
        exactV += cA * VA + cB * VB;
        // W = r^l: the traction mu (l - 1) / a
        exactW += -tK.W[i] / (mu * (l - 1.0) / a);
      }
      scale = std::max({scale, std::abs(exactU), std::abs(exactV)});
      worst = std::max({worst, std::abs(U[i] - exactU), std::abs(V[i] - exactV)});
      if (l == 1) continue;  // W: a rigid rotation's gauge
      scaleW = std::max(scaleW, std::abs(exactW));
      worstW = std::max(worstW, std::abs(W[i] - exactW));
    }
  }
  RecordProperty("spheroidal_departure", std::format("{:.3e}", worst / scale));
  RecordProperty("degree1_departure", std::format("{:.3e}", worst1 / scale1));
  RecordProperty("degree1_traction_consistency", std::format("{:.3e}", consistency));
  RecordProperty("toroidal_departure", std::format("{:.3e}", worstW / scaleW));
  EXPECT_GT(scale, 0.0);
  EXPECT_GT(scale1, 0.0);
  EXPECT_GT(scaleW, 0.0);
  EXPECT_LT(worst, 1e-6 * scale);
  EXPECT_LT(worst1, 1e-6 * scale1);
  EXPECT_LT(consistency, 1e-8);
  EXPECT_LT(worstW, 1e-6 * scaleW);
}
