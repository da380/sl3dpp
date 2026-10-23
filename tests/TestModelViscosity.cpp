/**
 * @file TestModelViscosity.cpp
 * @brief The io/ boundary: planetmodel model files read
 * back as a radial reference and a delta.
 *
 * The fixtures (scripts/make_model_viscosity_fixtures.py) write the
 * TestViscoelastic3D body — prem8 deck, 1e21 Pa s on the mantle — as
 * model files: a zero-mean log perturbation delta = A g(r) sin^2(theta)
 * cos(2 phi) with g = (r / R)^2 (the happy path), the same with a
 * nonzero lateral mean C0 g(r), the viscosity stored fully 3D, and the
 * radial model alone. Each laterally varying file comes with the
 * Python's own reading at off-grid probe points: both ends interpolate
 * the same stored numbers by the same stated rules, so the C++ reading
 * must match it to rounding.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/ModelViscosity.hpp"

namespace {

using namespace sl3dpp;
using io::ModelViscosity;
using io::ViscosityReference;

constexpr auto kLmax = 8;
constexpr auto kEtaSI = 1e21;
constexpr auto kA = 0.4;
constexpr auto kC0 = 0.3;

std::filesystem::path DataPath(const std::string& name) {
  return std::filesystem::path(SL3DPP_TEST_DATA_DIR) / name;
}

/// The fixtures' radial profile, of the non-dimensional radius: their R
/// is the deck's surface radius, the model's length scale.
double G(double r) { return r * r; }

struct Probe {
  double r = 0.0;      // SI
  double theta = 0.0;
  double phi = 0.0;
  double eta = 0.0;    // SI, the Python's reading
};

std::vector<Probe> ReadProbes(const std::filesystem::path& path) {
  auto in = std::ifstream(path);
  EXPECT_TRUE(in.is_open()) << path;
  auto probes = std::vector<Probe>{};
  auto line = std::string{};
  while (std::getline(in, line)) {
    if (line.empty() || line.front() == '#') continue;
    auto probe = Probe{};
    std::istringstream(line) >> probe.r >> probe.theta >> probe.phi >>
        probe.eta;
    probes.push_back(probe);
  }
  EXPECT_FALSE(probes.empty()) << path;
  return probes;
}

/// The TestViscoelastic3D body without its viscosity: the deck model,
/// its mesh and material, and the deck route's viscous layer set.
struct Body {
  EarthModel model;
  Material material;
  std::vector<std::size_t> viscous;

  static Body Make() {
    auto deck =
        Deck::Read(
            DataPath("prem8.elastic.deck"),
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
    auto material = Material::Create(model, mesh).value();
    const auto& scales = model.ModelScales();
    auto viscous = std::vector<std::size_t>{};
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscous.push_back(i);
      }
    }
    return Body{std::move(model), std::move(material), std::move(viscous)};
  }
};

/// eta_bar of the adapter's Maxwell map at a non-dimensional radius.
double MapValue(const std::map<std::size_t, RadialValue>& viscosity,
                std::size_t layer, double r) {
  return std::get<std::function<double(double)>>(viscosity.at(layer))(r);
}

/// A deterministic load on degree 2 alone.
std::optional<std::vector<Viscoelastic3D::Complex>> Load(int l, double) {
  if (l != 2) return std::nullopt;
  auto columns = std::vector<Viscoelastic3D::Complex>(l + 1);
  columns[0] = {0.7, 0.0};
  columns[1] = {0.3, -0.2};
  columns[2] = {-0.4, 0.5};
  return columns;
}

}  // namespace

// The happy path: the C++ reading of the log-perturbed file matches the
// Python's at off-grid points to rounding, under either reference, and
// the reference is the deck route's 1e21.
TEST(ModelViscosity, LogFileMatchesThePythonReading) {
  const auto body = Body::Make();
  const auto& scales = body.model.ModelScales();
  const auto length = scales.Length();
  const auto etaNd = kEtaSI / scales.Factor(ViscosityDimensions);
  for (const auto reference :
       {ViscosityReference::GeometricMean, ViscosityReference::FileRadial}) {
    const auto read = ModelViscosity::Read(DataPath("viscosity_log.nc"),
                                           body.model, {reference});
    ASSERT_TRUE(read) << read.error().message;
    EXPECT_FALSE(read->IsRadial());

    // The viscous layers are the deck route's, each at 1e21.
    ASSERT_EQ(read->Viscosity().size(), body.viscous.size());
    for (const auto layer : body.viscous) {
      ASSERT_TRUE(read->Viscosity().contains(layer));
    }

    const auto delta = read->Delta();
    for (const auto& probe : ReadProbes(DataPath("viscosity_probes_log.txt"))) {
      const auto r = probe.r / length;
      const auto expected = std::log(probe.eta / kEtaSI);
      EXPECT_NEAR(delta(r, probe.theta, probe.phi), expected,
                  1e-11 * (1.0 + std::abs(expected)));
      const auto layer = body.model.LayerOf(probe.r);
      EXPECT_NEAR(MapValue(read->Viscosity(), layer, r), etaNd,
                  1e-12 * etaNd);
    }
  }
}

// On a zero-mean perturbation the geometric mean recovers the file's
// own radial reference to rounding.
TEST(ModelViscosity, ReferencesCoincideOnAZeroMeanPerturbation) {
  const auto body = Body::Make();
  const auto length = body.model.ModelScales().Length();
  const auto mean =
      ModelViscosity::Read(DataPath("viscosity_log.nc"), body.model,
                           {ViscosityReference::GeometricMean});
  const auto file =
      ModelViscosity::Read(DataPath("viscosity_log.nc"), body.model,
                           {ViscosityReference::FileRadial});
  ASSERT_TRUE(mean && file);
  const auto dMean = mean->Delta();
  const auto dFile = file->Delta();
  for (const auto& probe : ReadProbes(DataPath("viscosity_probes_log.txt"))) {
    const auto r = probe.r / length;
    EXPECT_NEAR(dMean(r, probe.theta, probe.phi),
                dFile(r, probe.theta, probe.phi), 1e-12);
    const auto layer = body.model.LayerOf(probe.r);
    const auto vMean = MapValue(mean->Viscosity(), layer, r);
    const auto vFile = MapValue(file->Viscosity(), layer, r);
    EXPECT_NEAR(vMean, vFile, 1e-12 * vFile);
  }
}

// A nonzero lateral mean C0 g(r) splits the references: the file's
// radial one keeps 1e21 while the geometric mean absorbs exp(C0 g),
// and the two deltas differ by exactly the references' log ratio.
TEST(ModelViscosity, ANonZeroMeanSplitsTheReferences) {
  const auto body = Body::Make();
  const auto& scales = body.model.ModelScales();
  const auto length = scales.Length();
  const auto etaNd = kEtaSI / scales.Factor(ViscosityDimensions);
  const auto mean =
      ModelViscosity::Read(DataPath("viscosity_mean.nc"), body.model,
                           {ViscosityReference::GeometricMean});
  const auto file =
      ModelViscosity::Read(DataPath("viscosity_mean.nc"), body.model,
                           {ViscosityReference::FileRadial});
  ASSERT_TRUE(mean && file);
  const auto dMean = mean->Delta();
  const auto dFile = file->Delta();
  for (const auto& probe : ReadProbes(DataPath("viscosity_probes_mean.txt"))) {
    const auto r = probe.r / length;
    const auto layer = body.model.LayerOf(probe.r);
    const auto vFile = MapValue(file->Viscosity(), layer, r);
    const auto vMean = MapValue(mean->Viscosity(), layer, r);
    EXPECT_NEAR(vFile, etaNd, 1e-12 * etaNd);
    // The nodal means are exact (the perturbation is band-limited within
    // the grid); between nodes the radial rule interpolates exp(C0 g).
    EXPECT_NEAR(vMean, etaNd * std::exp(kC0 * G(r)), 1e-6 * vMean);
    // The split moves between the reference and delta, never the field.
    EXPECT_NEAR(dFile(r, probe.theta, probe.phi) + std::log(vFile),
                dMean(r, probe.theta, probe.phi) + std::log(vMean), 1e-12);
    EXPECT_NEAR(dFile(r, probe.theta, probe.phi),
                std::log(probe.eta / kEtaSI),
                1e-11 * (1.0 + std::abs(std::log(probe.eta / kEtaSI))));
  }
}

// A viscosity stored fully 3D: no radial reference to take, so
// FileRadial is refused, and the geometric mean recovers 1e21 exactly
// (the stored field has zero lateral log-mean at every node).
TEST(ModelViscosity, AFullyThreeDimensionalViscosity) {
  const auto body = Body::Make();
  const auto& scales = body.model.ModelScales();
  const auto length = scales.Length();
  const auto etaNd = kEtaSI / scales.Factor(ViscosityDimensions);

  const auto refused =
      ModelViscosity::Read(DataPath("viscosity_full3d.nc"), body.model,
                           {ViscosityReference::FileRadial});
  ASSERT_FALSE(refused);
  EXPECT_NE(refused.error().message.find("GeometricMean"), std::string::npos)
      << refused.error().message;

  const auto read =
      ModelViscosity::Read(DataPath("viscosity_full3d.nc"), body.model,
                           {ViscosityReference::GeometricMean});
  ASSERT_TRUE(read) << read.error().message;
  const auto delta = read->Delta();
  for (const auto& probe :
       ReadProbes(DataPath("viscosity_probes_full3d.txt"))) {
    const auto r = probe.r / length;
    const auto expected = std::log(probe.eta / kEtaSI);
    EXPECT_NEAR(delta(r, probe.theta, probe.phi), expected,
                1e-11 * (1.0 + std::abs(expected)));
    const auto layer = body.model.LayerOf(probe.r);
    EXPECT_NEAR(MapValue(read->Viscosity(), layer, r), etaNd, 1e-11 * etaNd);
  }
}

// A radial file: delta is identically zero and the map is the profile.
TEST(ModelViscosity, ARadialFileNeedsNoDelta) {
  const auto body = Body::Make();
  const auto& scales = body.model.ModelScales();
  const auto etaNd = kEtaSI / scales.Factor(ViscosityDimensions);
  const auto read =
      ModelViscosity::Read(DataPath("viscosity_radial.nc"), body.model);
  ASSERT_TRUE(read) << read.error().message;
  EXPECT_TRUE(read->IsRadial());
  const auto delta = read->Delta();
  const auto b = body.model.DimensionlessView().Boundaries();
  for (const auto layer : body.viscous) {
    const auto r = 0.5 * (b[layer] + b[layer + 1]);
    EXPECT_EQ(delta(r, 1.0, 2.0), 0.0);
    EXPECT_NEAR(MapValue(read->Viscosity(), layer, r), etaNd, 1e-12 * etaNd);
  }
}

TEST(ModelViscosity, AMissingFileIsRefused) {
  const auto body = Body::Make();
  const auto read =
      ModelViscosity::Read(DataPath("no_such_model.nc"), body.model);
  EXPECT_FALSE(read);
}

// The operator gate: a run under the file-backed delta reproduces the run
// under the analytic lambda the file was written from. The deltas agree
// to the file's lateral sampling (band 32, bicubic, a degree-2 field:
// measured 1.7e-6), and the surface responses follow (measured 1.8e-8
// of the largest column).
TEST(ModelViscosity, AStepMatchesTheAnalyticLambda) {
  const auto body = Body::Make();
  const auto read =
      ModelViscosity::Read(DataPath("viscosity_log.nc"), body.model);
  ASSERT_TRUE(read) << read.error().message;
  const auto rheology =
      Rheology::Maxwell(body.material, read->Viscosity()).value();
  const auto analytic = [](double r, double theta, double phi) {
    const auto s = std::sin(theta);
    return kA * G(r) * s * s * std::cos(2.0 * phi);
  };

  const auto fromFile = read->Delta();
  Viscoelastic3D opAnalytic(rheology, kLmax, analytic);
  Viscoelastic3D opFile(rheology, kLmax, fromFile);

  // The sampled deltas agree to the file's lateral resolution.
  const auto& coupling = opFile.Coupling();
  const auto& grid = coupling.Grid();
  auto worst = 0.0;
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    const auto r = coupling.RadiusOfNode(node);
    for (const auto theta : grid.CoLatitudes()) {
      for (const auto phi : grid.Longitudes()) {
        worst = std::max(
            worst, std::abs(fromFile(r, theta, phi) - analytic(r, theta, phi)));
      }
    }
  }
  EXPECT_LT(worst, 1e-5);

  const auto tau = rheology.MinRelaxationTime();
  auto sAnalytic = opAnalytic.ZeroState();
  auto sFile = opFile.ZeroState();
  auto xAnalytic = Viscoelastic3D::Solutions{};
  auto xFile = Viscoelastic3D::Solutions{};
  for (auto step = 0; step < 2; ++step) {
    auto a = opAnalytic.Step(sAnalytic, 0.7 * tau,
                             Scheme::ExponentialTrapezoid, Load);
    auto f =
        opFile.Step(sFile, 0.7 * tau, Scheme::ExponentialTrapezoid, Load);
    EXPECT_TRUE(a.info.converged);
    EXPECT_TRUE(f.info.converged);
    sAnalytic = std::move(a.state);
    sFile = std::move(f.state);
    xAnalytic = std::move(a.solutions);
    xFile = std::move(f.solutions);
  }
  const auto uAnalytic = opAnalytic.Surface(xAnalytic, Component::U);
  const auto uFile = opFile.Surface(xFile, Component::U);
  auto scale = 0.0;
  for (const auto& [l, columns] : uAnalytic) {
    for (const auto& u : columns) scale = std::max(scale, std::abs(u));
  }
  for (const auto& [l, columns] : uAnalytic) {
    const auto& others = uFile.at(l);
    for (auto c = std::size_t{0}; c < columns.size(); ++c) {
      EXPECT_NEAR(std::abs(columns[c] - others[c]), 0.0, 1e-7 * scale)
          << "l = " << l << ", c = " << c;
    }
  }
}

// A realistic file: the viscosity3d mini-app's pipeline (S40RTS,
// the sl3d conversion, band 96, bicubic, 50 km elements) run on this
// suite's prem8 body — scripts/make_model_viscosity_fixtures.py --miniapp
// <out>. Too large to commit (~90 MB), so the gate runs only when
// SL3DPP_VISCOSITY3D_FILE names it, and skips otherwise.
TEST(ModelViscosity, TheMiniAppFileWhenPresent) {
  const auto* given = std::getenv("SL3DPP_VISCOSITY3D_FILE");
  if (!given) GTEST_SKIP() << "SL3DPP_VISCOSITY3D_FILE is not set";
  const auto path = std::filesystem::path(given);
  auto probesPath = path;
  probesPath.replace_extension(".probes.txt");

  const auto body = Body::Make();
  const auto& scales = body.model.ModelScales();
  const auto length = scales.Length();
  const auto etaNd = kEtaSI / scales.Factor(ViscosityDimensions);
  const auto probes = ReadProbes(probesPath);
  for (const auto reference :
       {ViscosityReference::GeometricMean, ViscosityReference::FileRadial}) {
    const auto read = ModelViscosity::Read(path, body.model, {reference});
    ASSERT_TRUE(read) << read.error().message;
    const auto delta = read->Delta();
    for (const auto& probe : probes) {
      const auto r = probe.r / length;
      const auto expected = std::log(probe.eta / kEtaSI);
      // The written reference is the body's radial 1e21, and S40RTS is
      // transplanted without its degree 0, so the geometric mean
      // recovers the same reference to the grid's quadrature.
      EXPECT_NEAR(delta(r, probe.theta, probe.phi), expected,
                  1e-10 * (1.0 + std::abs(expected)));
      const auto layer = body.model.LayerOf(probe.r);
      EXPECT_NEAR(MapValue(read->Viscosity(), layer, r), etaNd,
                  1e-10 * etaNd);
    }
  }
}
