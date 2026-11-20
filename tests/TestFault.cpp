// A fault given by its slip: the moment of a slip across an element
// (isotropic and transversely isotropic, its basis against the double
// couple's), planetmodel's plane as the planar fault point for point, the
// slices' moments at a date as the moments of planetmodel's slip there, the
// placement and the refusals of FaultSourcesOf, and the forces of sources
// at one radius formed together.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Fault.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/ToroidalOperator.hpp"

using namespace sl3dpp;

namespace {

using Vector = std::array<double, 3>;

/// The polar basis at a point, Cartesian: r, theta (south), phi (east).
std::array<Vector, 3> PolarAt(double colatitude, double longitude) {
  const auto st = std::sin(colatitude), ct = std::cos(colatitude);
  const auto sp = std::sin(longitude), cp = std::cos(longitude);
  return {Vector{st * cp, st * sp, ct}, Vector{ct * cp, ct * sp, -st},
          Vector{-sp, cp, 0.0}};
}

/// A vector of components (r, theta, phi) at a point, Cartesian.
Vector FromPolar(const std::array<Vector, 3>& e, const Vector& c) {
  auto out = Vector{};
  for (auto i = 0; i < 3; ++i) {
    for (auto k = 0; k < 3; ++k) out[i] += c[k] * e[k][i];
  }
  return out;
}

double Largest(const SphericalTensor& M) {
  return std::max({std::abs(M.rr), std::abs(M.tt), std::abs(M.pp),
                   std::abs(M.rt), std::abs(M.rp), std::abs(M.tp)});
}

double Difference(const SphericalTensor& a, const SphericalTensor& b) {
  return std::max({std::abs(a.rr - b.rr), std::abs(a.tt - b.tt),
                   std::abs(a.pp - b.pp), std::abs(a.rt - b.rt),
                   std::abs(a.rp - b.rp), std::abs(a.tp - b.tp)});
}

LoveModuli Isotropic(double lambda, double mu) {
  return {.A = lambda + 2.0 * mu, .C = lambda + 2.0 * mu, .F = lambda,
          .L = mu, .N = mu};
}

}  // namespace

TEST(SlipMoment, IsTheIsotropicFormAndTheDoubleCouple) {
  // On an isotropic medium the moment of a slip s across an element of
  // normal n and area w is w (lambda (s.n) I + mu (s n + n s)), in the
  // polar basis at the point; a slip in the plane of a strike, dip and
  // rake is mu w |s| times that plane's double couple, so that the
  // Cartesian vectors are turned into the point's basis as the planar
  // fault's tensor is.
  const auto moduli = Isotropic(2.1e11, 1.7e11);
  auto random = std::mt19937_64(7);
  auto uniform = std::uniform_real_distribution<double>(-1.0, 1.0);
  for (auto trial = 0; trial < 5; ++trial) {
    const auto at = BodyPoint{.radius = 6.3e6,
                              .colatitude = 1.5 + 1.4 * uniform(random),
                              .longitude = 3.0 * uniform(random)};
    const auto e = PolarAt(at.colatitude, at.longitude);
    const auto s = Vector{uniform(random), uniform(random), uniform(random)};
    const auto n = Vector{uniform(random), uniform(random), uniform(random)};
    const auto M = SlipMoment(moduli, at, FromPolar(e, s), FromPolar(e, n), 2.5);
    const auto dot = s[0] * n[0] + s[1] * n[1] + s[2] * n[2];
    auto want = [&](int i, int j) {
      return 2.5 * (moduli.F * dot * (i == j ? 1.0 : 0.0) +
                    moduli.L * (s[i] * n[j] + n[i] * s[j]));
    };
    const auto expected = SphericalTensor{.rr = want(0, 0), .tt = want(1, 1),
                                          .pp = want(2, 2), .rt = want(0, 1),
                                          .rp = want(0, 2), .tp = want(1, 2)};
    EXPECT_LT(Difference(M, expected), 1e-14 * Largest(expected));
  }
  // the double couple: at the point, north = -theta, east = phi, up = r
  for (const auto& [strike, dip, rake] :
       {std::array{30.0, 60.0, 90.0}, std::array{210.0, 25.0, -40.0},
        std::array{135.0, 45.0, 170.0}}) {
    const auto at = BodyPoint{.radius = 6.3e6, .colatitude = 0.9, .longitude = 2.2};
    const auto e = PolarAt(at.colatitude, at.longitude);
    const auto phi = Radians(strike), delta = Radians(dip), lambda = Radians(rake);
    // Aki & Richards (2002) box 4.4, (north, east, down): the normal from
    // the footwall into the hanging wall and the hanging wall's slip
    const auto nu = Vector{-std::sin(delta) * std::sin(phi),
                           std::sin(delta) * std::cos(phi), -std::cos(delta)};
    const auto u = Vector{
        std::cos(lambda) * std::cos(phi) +
            std::cos(delta) * std::sin(lambda) * std::sin(phi),
        std::cos(lambda) * std::sin(phi) -
            std::cos(delta) * std::sin(lambda) * std::cos(phi),
        -std::sin(lambda) * std::sin(delta)};
    // (north, east, down) to (r, theta, phi)
    auto polar = [](const Vector& ned) { return Vector{-ned[2], -ned[0], ned[1]}; };
    const auto M = SlipMoment(moduli, at, FromPolar(e, polar(u)),
                              FromPolar(e, polar(nu)), 3.0);
    const auto unit = DoubleCouple(strike, dip, rake);
    const auto factor = 3.0 * moduli.L;
    const auto expected =
        SphericalTensor{.rr = factor * unit.rr, .tt = factor * unit.tt,
                        .pp = factor * unit.pp, .rt = factor * unit.rt,
                        .rp = factor * unit.rp, .tp = factor * unit.tp};
    EXPECT_LT(Difference(M, expected), 1e-14 * Largest(expected));
  }
}

TEST(SlipMoment, IsTheTransverselyIsotropicTensor) {
  // C : (s n) for a medium transversely isotropic about r: a radial slip
  // across a horizontal element strains it along r alone, M_rr = C w |s|
  // and M_tt = M_pp = F w |s|; a horizontal one along theta across a
  // horizontal element shears it in the vertical plane, M_rt = L w |s|;
  // across a vertical element of normal phi, in the horizontal plane,
  // M_tp = N w |s|; and C has major symmetry, so that a slip's moment
  // contracted with another's strain is the other's contracted with the
  // first's.
  const auto moduli = LoveModuli{.A = 3.1, .C = 2.9, .F = 1.3, .L = 0.8, .N = 0.9};
  const auto at = BodyPoint{.radius = 1.0, .colatitude = 1.1, .longitude = -0.4};
  const auto e = PolarAt(at.colatitude, at.longitude);
  const auto r = FromPolar(e, {1.0, 0.0, 0.0});
  const auto theta = FromPolar(e, {0.0, 1.0, 0.0});
  const auto phi = FromPolar(e, {0.0, 0.0, 1.0});
  const auto radial = SlipMoment(moduli, at, {2 * r[0], 2 * r[1], 2 * r[2]}, r, 0.5);
  EXPECT_NEAR(radial.rr, 2.9, 1e-14);
  EXPECT_NEAR(radial.tt, 1.3, 1e-14);
  EXPECT_NEAR(radial.pp, 1.3, 1e-14);
  EXPECT_NEAR(radial.rt, 0.0, 1e-14);
  EXPECT_NEAR(SlipMoment(moduli, at, theta, r, 1.0).rt, 0.8, 1e-14);
  EXPECT_NEAR(SlipMoment(moduli, at, theta, phi, 1.0).tp, 0.9, 1e-14);
  EXPECT_NEAR(SlipMoment(moduli, at, theta, phi, 1.0).rt, 0.0, 1e-14);
  auto random = std::mt19937_64(11);
  auto uniform = std::uniform_real_distribution<double>(-1.0, 1.0);
  auto vector = [&] { return Vector{uniform(random), uniform(random), uniform(random)}; };
  for (auto trial = 0; trial < 5; ++trial) {
    const auto s1 = vector(), n1 = vector(), s2 = vector(), n2 = vector();
    auto strain = [&](const Vector& s, const Vector& n) {
      auto c = [&](const Vector& a, const Vector& b) {
        auto out = 0.0;
        for (auto i = 0; i < 3; ++i) out += a[i] * b[i];
        return out;
      };
      auto part = [&](int i, int j) {
        return 0.5 * (c(s, e[i]) * c(n, e[j]) + c(s, e[j]) * c(n, e[i]));
      };
      return SphericalTensor{.rr = part(0, 0), .tt = part(1, 1), .pp = part(2, 2),
                             .rt = part(0, 1), .rp = part(0, 2), .tp = part(1, 2)};
    };
    const auto a = SlipMoment(moduli, at, s1, n1, 1.0).Contract(strain(s2, n2));
    const auto b = SlipMoment(moduli, at, s2, n2, 1.0).Contract(strain(s1, n1));
    EXPECT_NEAR(a, b, 1e-13 * std::abs(a));
  }
}

TEST(LoveModuli, AreTheDecksAtARadius) {
  // The prem8 deck's moduli at a radius of its transversely isotropic
  // upper mantle, from its velocities, as the material forms them at its
  // nodes.
  const auto deck =
      Deck::Read(std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
                 DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  const auto model = EarthModel::FromDeck(deck).value();
  const auto r = 6.25e6;
  const auto m = LoveModuliAt(model, r);
  const auto rho = model.Density(r);
  // the same evaluations and products, no transform: to the last bits
  EXPECT_DOUBLE_EQ(m.C, rho * std::pow(model.Evaluate("vpv", r), 2));
  EXPECT_DOUBLE_EQ(m.A, rho * std::pow(model.Evaluate("vph", r), 2));
  EXPECT_DOUBLE_EQ(m.L, rho * std::pow(model.Evaluate("vsv", r), 2));
  EXPECT_DOUBLE_EQ(m.N, rho * std::pow(model.Evaluate("vsh", r), 2));
  EXPECT_DOUBLE_EQ(m.F, model.Evaluate("eta", r) * (m.A - 2.0 * m.L));
  EXPECT_NE(m.L, m.N);  // transversely isotropic there
}

TEST(PointSourcesForces, AreTheSumOfEachSourcesForces) {
  // Sources formed together, those at one radius from their moments
  // summed and those at one point from their tensors summed, give the sum
  // of each one's forces.
  const auto deck =
      Deck::Read(std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
                 DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  const auto model = EarthModel::FromDeck(deck).value();
  constexpr auto kLmax = 6;
  const auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
  const auto material = Material::Create(model, mesh).value();
  auto spheroidal = std::map<int, std::unique_ptr<DegreeOperator>>{};
  auto toroidal = std::map<int, std::unique_ptr<ToroidalOperator>>{};
  for (auto l = 0; l <= kLmax; ++l) {
    spheroidal.emplace(l, std::make_unique<DegreeOperator>(material, l,
                                                           DegreeOperator::Options{}));
    if (l >= 1) {
      toroidal.emplace(l, std::make_unique<ToroidalOperator>(
                              material, l, ToroidalOperator::Options{}));
    }
  }
  const std::function<const DegreeOperator&(int)> sph =
      [&](int l) -> const DegreeOperator& { return *spheroidal.at(l); };
  const std::function<const ToroidalOperator&(int)> tor =
      [&](int l) -> const ToroidalOperator& { return *toroidal.at(l); };
  const auto r1 = 0.995 * material.Radius(), r2 = 0.99 * material.Radius();
  auto sources = std::vector<PointSource>{
      {.at = {r1, 1.0, 0.3}, .moment = {1.0, -0.4, -0.6, 0.2, 0.3, -0.1}},
      {.at = {r1, 1.2, 0.5}, .moment = {0.3, 0.5, -0.8, -0.2, 0.1, 0.4}},
      {.at = {r2, 1.1, 0.4}, .moment = {-0.7, 0.2, 0.5, 0.6, -0.3, 0.2}},
      {.at = {r1, 1.0, 0.3}, .moment = {0.2, 0.1, -0.3, 0.5, 0.4, 0.3}},
      // within rounding of r1
      {.at = {r1 * (1.0 + 2e-16), 0.9, 0.2},
       .moment = {0.4, -0.1, -0.3, 0.1, -0.2, 0.6}}};
  const auto together = PointSourcesForces(material, kLmax, sph, tor, sources);
  auto apart = DegreeFields{};
  for (const auto& source : sources) {
    const auto forces = PointSourceForces(material, kLmax, sph, tor, source);
    for (const auto& [l, f] : forces.spheroidal) {
      auto& to = apart.spheroidal[l];
      if (to.empty()) to.assign(f.size(), 0.0);
      for (auto i = std::size_t{0}; i < f.size(); ++i) to[i] += f[i];
    }
    for (const auto& [l, f] : forces.toroidal) {
      auto& to = apart.toroidal[l];
      if (to.empty()) to.assign(f.size(), 0.0);
      for (auto i = std::size_t{0}; i < f.size(); ++i) to[i] += f[i];
    }
  }
  for (const auto which : {&DegreeFields::spheroidal, &DegreeFields::toroidal}) {
    ASSERT_EQ((together.*which).size(), (apart.*which).size());
    for (const auto& [l, f] : apart.*which) {
      const auto& g = (together.*which).at(l);
      ASSERT_EQ(g.size(), f.size());
      auto worst = 0.0, size = 0.0;
      for (auto i = std::size_t{0}; i < f.size(); ++i) {
        worst = std::max(worst, std::abs(g[i] - f[i]));
        size = std::max(size, std::abs(f[i]));
      }
      // to rounding: the source formed at r1 rather than at its own radius
      // 2e-16 above moves the forces by 7e-14 of their size
      EXPECT_LE(worst, 1e-11 * size) << "degree " << l;
    }
  }
  EXPECT_TRUE(PointSourcesForces(material, kLmax, sph, tor, {}).spheroidal.empty());
}


namespace {

namespace pm = planetmodel;

/// A plane of uniform slip `magnitude` at `rake` (degrees) on a sphere of
/// radius `reference`, its centroid `depth` below it.
pm::Fault PlaneFault(double reference, double latitude, double longitude,
                     double depth, double strike, double dip, double rake,
                     double length, double width, double magnitude,
                     const pm::FaultFrame& frame) {
  const auto lambda = Radians(rake);
  auto slip = pm::FaultSlip({magnitude * std::cos(lambda),
                             magnitude * std::sin(lambda), 0.0},
                            1, 1, 1, "cell");
  auto patch = pm::FaultPatch::Plane(
      frame,
      pm::PlaneParameters{.centroid = {latitude, longitude, reference - depth},
                          .strike = strike,
                          .dip = dip,
                          .length = length,
                          .width = width},
      std::move(slip));
  return pm::Fault({std::move(patch)});
}

}  // namespace

TEST(FaultSources, APlaneIsThePlanarFault) {
  // planetmodel's plane of a uniform slip D is sl3dpp's planar fault of
  // moment mu D L W: at the midpoints of N by M cells the quadrature's
  // points are the patches' centres, every one at the same radius,
  // colatitude and longitude to rounding, and its normal and the slip's
  // direction are Aki & Richards' (box 4.4) for the strike, dip and rake,
  // the planar fault's own; the moment tensors, mu w D (n u + u n) in the
  // point's basis, are the patches' shares.
  const auto R = 6371e3, mu = 3.0e10, lambda = 4.0e10;
  const auto strike = 200.0, dip = 35.0, rake = 80.0;
  const auto length = 300e3, width = 80e3, depth = 30e3, moment = 4e21;
  const auto D = moment / (mu * length * width);
  const auto frame = pm::FaultFrame::Geographic(R);
  const auto fault = PlaneFault(R, 35.0, 140.0, depth, strike, dip, rake, length,
                                width, D, frame);
  const auto sources = FaultSourcesOf(
      fault, {.cellsAlongStrike = 12,
              .cellsDownDip = 4,
              .points = 1,
              .surfaceRadius = R,
              .placement = std::nullopt,
              .moduli = [&](const BodyPoint&) { return Isotropic(lambda, mu); }});
  const auto planar = PlanarFaultSources(
      PlanarFault{.centroid = {35.0, 140.0},
                  .depth = depth,
                  .strike = strike,
                  .dip = dip,
                  .rake = rake,
                  .length = length,
                  .width = width,
                  .moment = moment},
      R, 12, 4);
  ASSERT_EQ(sources.points.size(), planar.size());
  ASSERT_EQ(sources.slices.size(), 1u);
  auto positions = 0.0, tensors = 0.0;
  for (auto p = std::size_t{0}; p < planar.size(); ++p) {
    const auto& a = sources.points[p];
    const auto& b = planar[p].at;
    positions = std::max({positions, std::abs(a.radius - b.radius) / R,
                          std::abs(a.colatitude - b.colatitude),
                          std::abs(a.longitude - b.longitude)});
    tensors = std::max(tensors, Difference(sources.slices[0][p], planar[p].moment) /
                                    Largest(planar[p].moment));
    EXPECT_DOUBLE_EQ(sources.depths[p], R - a.radius);  // one subtraction
  }
  RecordProperty("positions", std::format("{:.2e}", positions));
  RecordProperty("tensors", std::format("{:.2e}", tensors));
  // (a few ulps here: 4e-16 and 1.5e-15)
  EXPECT_LT(positions, 1e-13);
  EXPECT_LT(tensors, 1e-13);
  // the normal and the slip's direction at the centroid: Aki & Richards'
  const auto& patch = fault.Patches().front();
  const auto e = PolarAt(Radians(90.0 - 35.0), Radians(140.0));
  const auto phi = Radians(strike), delta = Radians(dip), l = Radians(rake);
  const auto nu = Vector{-std::sin(delta) * std::sin(phi),
                         std::sin(delta) * std::cos(phi), -std::cos(delta)};
  const auto u = Vector{
      std::cos(l) * std::cos(phi) + std::cos(delta) * std::sin(l) * std::sin(phi),
      std::cos(l) * std::sin(phi) - std::cos(delta) * std::sin(l) * std::cos(phi),
      -std::sin(l) * std::sin(delta)};
  auto polar = [](const Vector& ned) { return Vector{-ned[2], -ned[0], ned[1]}; };
  const auto normal = patch.Normal(0.0, 0.0);
  const auto slip = patch.SlipVector(0.0, 0.0, 0, pm::SlipBasis::Cartesian);
  const auto wantNormal = FromPolar(e, polar(nu));
  const auto wantSlip = FromPolar(e, polar(u));
  for (auto i = 0; i < 3; ++i) {
    EXPECT_NEAR(normal[i], wantNormal[i], 1e-15);
    EXPECT_NEAR(slip[i] / D, wantSlip[i], 1e-15);
  }
}

TEST(FaultSources, AtADateAreTheMomentsOfItsSlipThere) {
  // A plane whose slip turns between two slices (a thrust at t = 0, a
  // strike-slip of twice it at t = 10, linear between): at each date the
  // slices' tensors under the rows planetmodel reads there (MomentsAt) are
  // the moments of the slip planetmodel gives at that date, both by its
  // slice quadrature's combination (SlipOf) and by its quadrature at the
  // date (Fault::Quadrature), to rounding; rows naming a slice the sources
  // lack are refused.
  const auto R = 6371e3, mu = 3.0e10, lambda = 4.0e10;
  const auto frame = pm::FaultFrame::Geographic(R);
  auto slip = pm::FaultSlip({0.0, 2.0, 0.0, 4.0, 0.0, 0.0}, 2, 1, 1, "cell");
  auto patch = pm::FaultPatch::Plane(
      frame,
      pm::PlaneParameters{.centroid = {10.0, -30.0, R - 25e3},
                          .strike = 45.0,
                          .dip = 30.0,
                          .length = 200e3,
                          .width = 60e3},
      std::move(slip));
  const auto fault =
      pm::Fault({std::move(patch)}, pm::TimeSupport::Node, pm::TimeRule::Linear,
                std::vector<double>{0.0, 10.0});
  const auto moduli = [&](const BodyPoint&) { return Isotropic(lambda, mu); };
  const auto sources = FaultSourcesOf(fault, {.cellsAlongStrike = 5,
                                              .cellsDownDip = 3,
                                              .points = 2,
                                              .surfaceRadius = R,
                                              .placement = std::nullopt,
                                              .moduli = moduli});
  const auto slices = fault.SliceQuadrature(5, 3, 2);
  ASSERT_EQ(sources.slices.size(), 2u);
  ASSERT_EQ(sources.points.size(), slices.Size());
  auto worst = 0.0;
  for (const auto t : {0.0, 2.5, 7.5, 10.0}) {
    const auto rows = fault.Rows(t);
    const auto moments = sources.MomentsAt(rows);
    const auto combined = slices.SlipOf(rows);
    const auto atDate = fault.Quadrature(5, 3, 2, t);
    ASSERT_EQ(moments.size(), slices.Size());
    for (auto p = std::size_t{0}; p < moments.size(); ++p) {
      const auto& at = sources.points[p];
      for (const auto* s : {&combined[p], &atDate.slip[p]}) {
        const auto want = SlipMoment(moduli(at), at, *s, slices.normals[p],
                                     slices.weights[p]);
        worst = std::max(worst, Difference(moments[p], want) / Largest(want));
      }
    }
  }
  RecordProperty("worst", std::format("{:.2e}", worst));
  EXPECT_LT(worst, 1e-13);  // (8e-16 here)
  EXPECT_THROW(static_cast<void>(sources.MomentsAt({{2, 1.0}, {2, 0.0}})),
               std::invalid_argument);
}

TEST(FaultSources, ArePlacedAndRefusedByName) {
  // A fault whose reference radius is the model's surface radius is taken
  // as stored; one whose is not is placed by depth (each point moved along
  // its radius, its depth kept) or by radius (as stored), and refused
  // without a placement. A point above the surface, a fault in a
  // Cartesian frame and a point in a fluid are refused, naming what.
  const auto R = 6371e3;
  const auto fault = PlaneFault(R, -20.0, 70.0, 20e3, 10.0, 30.0, 90.0, 100e3,
                                40e3, 2.0, pm::FaultFrame::Geographic(R));
  const auto moduli = [](const BodyPoint&) { return Isotropic(3e10, 3e10); };
  auto options = FaultSourceOptions{.cellsAlongStrike = 3,
                                    .cellsDownDip = 2,
                                    .points = 2,
                                    .surfaceRadius = R,
                                    .placement = std::nullopt,
                                    .moduli = moduli};
  const auto stored = FaultSourcesOf(fault, options);
  EXPECT_EQ(stored.points.size(), 24u);
  options.surfaceRadius = 6368e3;
  try {
    static_cast<void>(FaultSourcesOf(fault, options));
    ADD_FAILURE() << "a mismatched reference radius was not refused";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string{error.what()}.find("is not the model's surface radius"),
              std::string::npos)
        << error.what();
  }
  options.placement = FaultPlacement::Depth;
  const auto byDepth = FaultSourcesOf(fault, options);
  options.placement = FaultPlacement::Radius;
  const auto byRadius = FaultSourcesOf(fault, options);
  for (auto p = std::size_t{0}; p < stored.points.size(); ++p) {
    EXPECT_NEAR(byDepth.depths[p], stored.depths[p], 1e-9);
    EXPECT_NEAR(byDepth.points[p].radius, stored.points[p].radius - 3e3, 1e-9);
    // the same quadrature, no transform: what placement leaves is bitwise
    EXPECT_EQ(byDepth.points[p].colatitude, stored.points[p].colatitude);
    EXPECT_EQ(byRadius.points[p].radius, stored.points[p].radius);
    EXPECT_NEAR(byRadius.depths[p], stored.depths[p] - 3e3, 1e-9);
    // the tensor kept: the slip, the normal and the weight are the file's
    EXPECT_EQ(byDepth.slices[0][p].rt, stored.slices[0][p].rt);
  }
  // the shallowest points lie about 15 km deep: above a surface 20 km lower
  options.surfaceRadius = R - 20e3;
  try {
    static_cast<void>(FaultSourcesOf(fault, options));
    ADD_FAILURE() << "a point above the surface was not refused";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string{error.what()}.find("above the model's surface"),
              std::string::npos)
        << error.what();
  }
  const auto flat = PlaneFault(0.0, 10.0, 20.0, 5.0, 0.0, 30.0, 90.0, 4.0, 2.0,
                               1.0, pm::FaultFrame::Cartesian(0.0));
  options.surfaceRadius = R;
  options.placement = std::nullopt;
  try {
    static_cast<void>(FaultSourcesOf(flat, options));
    ADD_FAILURE() << "a Cartesian frame was not refused";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string{error.what()}.find("Cartesian frame"), std::string::npos)
        << error.what();
  }
  options.moduli = [](const BodyPoint&) { return LoveModuli{.A = 1, .C = 1, .F = 1}; };
  try {
    static_cast<void>(FaultSourcesOf(fault, options));
    ADD_FAILURE() << "a point in a fluid was not refused";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string{error.what()}.find("no shear modulus"), std::string::npos)
        << error.what();
  }
}

