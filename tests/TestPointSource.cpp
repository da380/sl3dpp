// The point source: the strain of degree solutions at a point against
// displacement fields whose strain is known (a uniform strain, a rigid
// rotation), and the source's forces as the transpose of that
// evaluation.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
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
