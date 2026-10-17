// The stage-0 verification gate (plan §6): reference gravity against
// planetmodel on the identical deck file, read by both codes with the same
// (not-a-knot) interpolation, so agreement is expected near machine
// precision.

#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <complex>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/ShConventions.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace {

struct GravityFixture {
  double mass = 0;
  double moment = 0;
  std::vector<std::pair<double, double>> gravity;  // (r, g), SI
};

GravityFixture ReadFixture(const std::filesystem::path& path) {
  std::ifstream file(path);
  EXPECT_TRUE(file.is_open()) << path;
  GravityFixture fixture;
  auto inTable = false;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    auto head = std::string{};
    if (!inTable) {
      row >> head;
      if (head == "mass") {
        row >> fixture.mass;
      } else if (head == "moment") {
        row >> fixture.moment;
      } else if (head == "gravity") {
        inTable = true;
      }
      continue;
    }
    auto r = 0.0, g = 0.0;
    row >> r >> g;
    fixture.gravity.emplace_back(r, g);
  }
  return fixture;
}

}  // namespace

namespace {

/// A stage-2 fixture: named coefficient blocks (converted into GSHTrans's
/// basis on read) and named scalar rows.
struct Stage2Fixture {
  std::map<std::string, std::vector<std::complex<double>>> fields;
  std::map<std::string, std::vector<double>> scalars;
};

Stage2Fixture ReadStage2(const std::filesystem::path& path, int lmax) {
  Stage2Fixture fixture;
  std::ifstream file(path);
  EXPECT_TRUE(file.is_open()) << path;
  std::vector<std::complex<double>>* current = nullptr;
  const auto size = static_cast<std::size_t>((lmax + 1) * (lmax + 2) / 2);
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (!std::isdigit(line.front())) {
      auto head = std::string{};
      row >> head;
      if (head == "field") {
        auto name = std::string{};
        row >> name;
        current = &fixture.fields[name];
        current->assign(size, 0.0);
      } else {
        auto& values = fixture.scalars[head];
        for (double v; row >> v;) values.push_back(v);
        current = nullptr;
      }
      continue;
    }
    if (current == nullptr) {
      ADD_FAILURE() << "coefficient row outside a field block";
      continue;
    }
    auto l = 0, m = 0;
    auto a = 0.0, b = 0.0;
    row >> l >> m >> a >> b;
    (*current)[sl3dpp::SurfaceGrid::Index(l, m)] =
        sl3dpp::ComplexFromRealPair({a, b}, m);
  }
  return fixture;
}

/// Worst coefficient difference relative to the field's largest
/// coefficient.
double CoefficientError(std::span<const std::complex<double>> ours,
                        std::span<const std::complex<double>> theirs) {
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < ours.size(); ++i) {
    scale = std::max(scale, std::abs(theirs[i]));
    diff = std::max(diff, std::abs(ours[i] - theirs[i]));
  }
  return scale > 0.0 ? diff / scale : 0.0;
}

}  // namespace

TEST(Stage2Gate, GeneralisedSolveMatchesPyslfp) {
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  constexpr auto lmax = 32;
  auto table =
      sl3dpp::LoveNumbers::FromFile(data / "stage1_love_numbers.txt").value();
  sl3dpp::SeaLevelSolver solver(sl3dpp::SurfaceGrid(lmax), table,
                                sl3dpp::SeaLevelParameters{});
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto fixture = ReadStage2(data / "stage2_generalised.txt", lmax);

  constexpr sl3dpp::Dimensions kSigma{.mass = 1, .length = -2};
  constexpr sl3dpp::Dimensions kMoment{.mass = 1, .length = 2};
  auto gridInput = [&](const std::string& name, sl3dpp::Dimensions dims) {
    auto si = grid.Synthesise(fixture.fields.at(name));
    for (auto& v : si) v /= scales.Factor(dims);
    return si;
  };
  auto coefInput = [&](const std::string& name, sl3dpp::Dimensions dims) {
    auto c = fixture.fields.at(name);
    for (auto& v : c) v /= scales.Factor(dims);
    return c;
  };

  sl3dpp::SeaLevelSolver::GeneralisedForcing forcing;
  forcing.oceanFunctionOverride = gridInput("ocean_function", {});
  forcing.directLoad = gridInput("direct_load", kSigma);
  forcing.displacementLoad = gridInput("displacement_load", kSigma);
  forcing.potentialLoad = gridInput("potential_load", kSigma);
  forcing.displacementResponse =
      coefInput("displacement_response", {.length = 1});
  forcing.potentialResponse =
      coefInput("potential_response", sl3dpp::PotentialDimensions);
  forcing.inertiaMomentChange =
      fixture.scalars.at("inertia_moment")[0] / scales.Factor(kMoment);
  for (auto i = 0; i < 3; ++i) {
    forcing.angularMomentumChange[i] =
        fixture.scalars.at("angular_momentum")[i] /
        scales.Factor({.mass = 1, .length = 2, .time = -1});
  }
  forcing.rotation = sl3dpp::RotationalFeedback::Full;

  // Any state: the ocean function is overridden.
  auto state = solver.MakeState(
      grid.ZeroField(), grid.FromFunction([](double, double) { return 1.0; }));
  const auto response = solver.SolveGeneralised(
      state, forcing, {.rtol = 1e-13, .max_iterations = 500});
  ASSERT_TRUE(response.converged);

  auto toSi = [&](std::span<const double> field, sl3dpp::Dimensions dims) {
    auto c = grid.Expand(field);
    for (auto& v : c) v *= scales.Factor(dims);
    return c;
  };
  // The floor is the aliasing tail of near-lmax products, which differs
  // between grids; observed ~2e-9.
  constexpr auto tol = 5e-9;
  EXPECT_LT(CoefficientError(toSi(response.seaLevelChange, {.length = 1}),
                             fixture.fields.at("sea_level_change")),
            tol);
  EXPECT_LT(CoefficientError(toSi(response.displacement, {.length = 1}),
                             fixture.fields.at("displacement")),
            tol);
  EXPECT_LT(
      CoefficientError(toSi(response.potential, sl3dpp::PotentialDimensions),
                       fixture.fields.at("potential")),
      tol);
  const auto& omega = fixture.scalars.at("omega");
  auto omegaScale = 0.0;
  for (auto v : omega) omegaScale = std::max(omegaScale, std::abs(v));
  for (auto i = 0; i < 3; ++i) {
    EXPECT_NEAR(response.angularVelocity[i] * scales.Factor({.time = -1}),
                omega[i], tol * omegaScale)
        << "omega component " << i;
  }
}

TEST(Stage2Gate, SharpOceanFingerprintNearPyslfp) {
  // The true flotation ocean function is a step function sampled on each
  // code's own grid, so agreement is limited by the coastline quadrature,
  // not by either implementation; the comparison is correspondingly
  // coarse (plan §6).
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  constexpr auto lmax = 32;
  auto table =
      sl3dpp::LoveNumbers::FromFile(data / "stage1_love_numbers.txt").value();
  sl3dpp::Fingerprint fingerprint(lmax, table);
  fingerprint.SetAnalyticTestState(0.0);
  const auto& grid = fingerprint.Grid();
  const auto fixture = ReadStage2(data / "stage2_fingerprint.txt", lmax);

  auto ice = fingerprint.IceThickness();
  auto iceChange = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < ice.size(); ++i)
    iceChange[i] = -0.1 * ice[i];
  const auto result = fingerprint.SolveForIceChange(
      iceChange, sl3dpp::RotationalFeedback::Full);
  ASSERT_TRUE(result.converged);

  // The tolerances below are pyslfp's own measured grid sensitivity for
  // this problem (DH vs GLQ at lmax 32): ocean area 0.44%, sea-level
  // coefficients 0.049 of the largest, the axial rotation 4%. The
  // transverse rotation is a small residual of a nearly zonal load and is
  // not determined at this resolution (pyslfp's own grids disagree on it
  // by a factor of two), so it is not compared.
  std::vector<double> ocean(fingerprint.OceanFunction().begin(),
                            fingerprint.OceanFunction().end());
  const auto area = table.radius * table.radius * grid.Integrate(ocean);
  EXPECT_NEAR(area, fixture.scalars.at("ocean_area")[0], 1e-2 * area);

  auto toSi = [&](std::span<const double> field) { return grid.Expand(field); };
  constexpr auto tol = 8e-2;
  EXPECT_LT(CoefficientError(toSi(result.seaLevelChange),
                             fixture.fields.at("sea_level_change")),
            tol);
  EXPECT_LT(CoefficientError(toSi(result.displacement),
                             fixture.fields.at("displacement")),
            tol);
  const auto& omega = fixture.scalars.at("omega");
  EXPECT_NEAR(result.angularVelocity[2], omega[2], 0.1 * std::abs(omega[2]));
}

TEST(Stage3Gate, StepperMatchesPyslfp) {
  // The 1D Love-number history: grid-free, so the stepper is isolated
  // from the SLE and the agreement is near machine precision.
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  constexpr auto lmax = 10;
  constexpr int kDegrees[3] = {2, 5, 10};
  auto deck = sl3dpp::Deck::Read(
                  data / "prem8.elastic.deck",
                  sl3dpp::DeckFormat(
                      "elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
                  .value();
  auto model = sl3dpp::EarthModel::FromDeck(deck).value();
  auto mesh = sl3dpp::RadialMesh::Create(model, {.lmax = lmax}).value();
  auto material = sl3dpp::Material::Create(model, mesh).value();
  const auto& scales = model.ModelScales();

  // The fixture's rheology rule: 1e21 Pa s on solid layers fully inside
  // [3480, 6151] km.
  auto viscosity = std::map<std::size_t, sl3dpp::RadialValue>{};
  const auto b = model.DimensionlessView().Boundaries();
  const auto etaNd = 1e21 / scales.Factor(sl3dpp::ViscosityDimensions);
  for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
    if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
        b[i + 1] <= 6.1511e6 / scales.Length()) {
      viscosity.emplace(i, etaNd);
    }
  }
  auto rheology = sl3dpp::Rheology::Maxwell(material, viscosity).value();
  sl3dpp::ViscoelasticOperator op(rheology, lmax);

  const double sigma =
      scales.Factor(sl3dpp::Dimensions{.mass = 1, .length = -2});
  const auto load = [&](int, double) {
    return std::vector<double>{1.0 / sigma};  // 1 kg m^-2
  };
  const auto length = scales.Length();
  const auto potential = scales.Factor(sl3dpp::PotentialDimensions);
  const auto dt = 3.15576e9 / scales.Time();

  // Parse and replay the fixture.
  std::ifstream file(data / "stage3_stepper.txt");
  ASSERT_TRUE(file.is_open());
  constexpr auto tol = 1e-9;
  auto scheme = sl3dpp::Scheme::ExponentialTrapezoid;
  auto state = std::optional<sl3dpp::ViscoelasticState>{};
  auto solutions = std::map<int, std::vector<double>>{};
  auto inEffective = false;
  auto tauMin = 0.0;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (!std::isdigit(line.front()) && line.front() != '-') {
      auto head = std::string{};
      row >> head;
      if (head == "min_relaxation_time") {
        row >> tauMin;
        EXPECT_NEAR(rheology.MinRelaxationTime() * scales.Time(), tauMin,
                    1e-10 * tauMin);
      } else if (head == "scheme") {
        auto name = std::string{};
        row >> name;
        scheme = name == "backward_euler"
                     ? sl3dpp::Scheme::BackwardEuler
                     : sl3dpp::Scheme::ExponentialTrapezoid;
        state = op.ZeroState();
        solutions = op.SolveElastic(*state, load);
      } else if (head == "effective_love") {
        inEffective = true;
      }
      continue;
    }
    if (inEffective) {
      auto l = 0;
      auto hU = 0.0, kU = 0.0, hT = 0.0, kT = 0.0;
      row >> l >> hU >> kU >> hT >> kT;
      const auto& table =
          op.EffectiveLoveNumbers(sl3dpp::Scheme::ExponentialTrapezoid, dt);
      // Each value against its column's natural scale: a number that is
      // zero up to roundoff (k_u at l = 0, say) is compared absolutely.
      const auto hScale = std::max(std::abs(hU), std::abs(table.h_u[2]));
      const auto kScale = std::max(std::abs(kU), std::abs(table.k_u[2]));
      EXPECT_NEAR(table.h_u[l], hU, tol * hScale) << "l = " << l;
      EXPECT_NEAR(table.k_u[l], kU, tol * kScale) << "l = " << l;
      EXPECT_NEAR(table.h_t[l], hT, tol * std::max(std::abs(hT), hScale))
          << "l = " << l;
      EXPECT_NEAR(table.k_t[l], kT, tol * std::max(std::abs(kT), kScale))
          << "l = " << l;
      continue;
    }
    // A history row: t then (U, phi) per compared degree, all SI.
    ASSERT_TRUE(state.has_value());
    auto t = 0.0;
    row >> t;
    if (t > 0.0) {
      auto [next, sols] = op.Step(*state, dt, scheme, load);
      state = std::move(next);
      solutions = std::move(sols);
      EXPECT_NEAR(state->t * scales.Time(), t, 1e-9 * t);
    }
    const auto U = op.Surface(solutions, sl3dpp::Component::U);
    const auto phi = op.Surface(solutions, sl3dpp::Component::Phi);
    for (int l : kDegrees) {
      auto uRef = 0.0, phiRef = 0.0;
      row >> uRef >> phiRef;
      EXPECT_NEAR(U.at(l)[0] * length, uRef, tol * std::abs(uRef))
          << "t = " << t << ", l = " << l;
      EXPECT_NEAR(phi.at(l)[0] * potential, phiRef, tol * std::abs(phiRef))
          << "t = " << t << ", l = " << l;
    }
  }
  EXPECT_GT(tauMin, 0.0);
}

TEST(Stage1Gate, LoveNumbersMatchPyslfp) {
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  auto deck = sl3dpp::Deck::Read(
                  data / "prem8.elastic.deck",
                  sl3dpp::DeckFormat(
                      "elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
                  .value();
  auto model = sl3dpp::EarthModel::FromDeck(deck).value();
  constexpr auto lmax = 32;
  auto mesh = sl3dpp::RadialMesh::Create(model, {.lmax = lmax}).value();
  auto material = sl3dpp::Material::Create(model, mesh).value();
  auto ln = sl3dpp::ComputeLoveNumbers(material, lmax);

  // Parse the pyslfp reference.
  std::ifstream file(data / "stage1_love_numbers.txt");
  ASSERT_TRUE(file.is_open());
  auto radius = 0.0, surfaceGravity = 0.0, fluidTidal = 0.0;
  double axial[5] = {};
  auto table = std::vector<std::array<double, 13>>{};
  auto inTable = false;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (!inTable) {
      auto head = std::string{};
      row >> head;
      if (head == "radius") row >> radius;
      if (head == "surface_gravity") row >> surfaceGravity;
      if (head == "axial") {
        for (auto& v : axial) row >> v;
      }
      if (head == "fluid_tidal_2") row >> fluidTidal;
      if (head == "table") inTable = true;
      continue;
    }
    auto& entries = table.emplace_back();
    for (auto& v : entries) row >> v;
  }
  ASSERT_EQ(table.size(), static_cast<std::size_t>(lmax + 1));

  constexpr auto tol = 1e-10;
  EXPECT_NEAR(ln.radius, radius, tol * radius);
  EXPECT_NEAR(ln.surface_gravity, surfaceGravity, tol * surfaceGravity);
  const std::vector<double>* columns[12] = {
      &ln.h_u, &ln.l_u, &ln.k_u, &ln.h_phi, &ln.l_phi, &ln.k_phi,
      &ln.h_v, &ln.l_v, &ln.k_v, &ln.h_t,   &ln.l_t,   &ln.k_t};
  for (const auto& entries : table) {
    const auto l = static_cast<int>(entries[0]);
    // One scale per degree: tiny channels (a removed dof's zeros) are
    // compared absolutely against the degree's magnitude.
    auto scale = 0.0;
    for (auto c = 0; c < 12; ++c)
      scale = std::max(scale, std::abs(entries[c + 1]));
    for (auto c = 0; c < 12; ++c) {
      EXPECT_NEAR((*columns[c])[l], entries[c + 1], tol * scale)
          << "l = " << l << ", column " << c;
    }
  }
  // k_c and m_phi vanish by the shell theorem, so each code computes pure
  // round-off there (~1e-11 of m_u, uncorrelated): the floor is absolute,
  // on the natural scale m_u.
  const double values[5] = {ln.h_c, ln.k_c, ln.m_u, ln.m_phi, ln.m_c};
  for (auto i = 0; i < 5; ++i) {
    const double scale =
        std::max({std::abs(axial[i]), std::abs(values[i]), 1e-30});
    EXPECT_NEAR(values[i], axial[i],
                std::max(tol * scale, 1e-10 * std::abs(axial[2])))
        << "axial " << i;
  }
  EXPECT_NEAR(sl3dpp::FluidTidalNumber(material, 2), fluidTidal,
              tol * fluidTidal);
}

TEST(Stage0Gate, GravityMatchesPlanetmodel) {
  const auto data = std::filesystem::path(SL3DPP_TEST_DATA_DIR);
  auto deck = sl3dpp::Deck::Read(data / "prem8.rho.deck",
                                 sl3dpp::DeckFormat("rho-only", {"rho"}, 0))
                  .value();
  auto model = sl3dpp::EarthModel::FromDeck(deck).value();
  const auto fixture = ReadFixture(data / "stage0_gravity.txt");
  ASSERT_GT(fixture.gravity.size(), 10u);

  EXPECT_NEAR(model.Mass(), fixture.mass, 1e-13 * fixture.mass);
  EXPECT_NEAR(model.MomentOfInertia(), fixture.moment, 1e-13 * fixture.moment);
  for (const auto& [r, g] : fixture.gravity) {
    EXPECT_NEAR(model.Gravity(r), g, 1e-13 * std::max(g, 1.0))
        << "at r = " << r;
  }
}
