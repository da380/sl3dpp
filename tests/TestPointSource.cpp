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
