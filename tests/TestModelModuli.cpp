/**
 * @file TestModelModuli.cpp
 * @brief The io/ boundary for lateral elastic moduli: a model file
 * written by io::WriteModuli from a radial model with given relative
 * perturbations reads back through io::ModelModuli as those
 * perturbations less their lateral means, and runs the 3D stepper's
 * static solve as the same functors would.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <numbers>
#include <string>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/ModelModuli.hpp"

namespace {

using namespace sl3dpp;

constexpr auto kLmax = 8;

struct Scene {
  EarthModel model;
  Material material;
  std::filesystem::path file;

  static Scene Make(const std::string& name) {
    auto deck = Deck::Read(std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                               "prem8.elastic.deck",
                           DeckFormat("elastic",
                                      {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
                    .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
    auto material = Material::Create(model, mesh).value();
    const auto dir = std::filesystem::temp_directory_path() / "sl3dpp_test_model_moduli";
    std::filesystem::create_directories(dir);
    return {std::move(model), std::move(material), dir / name};
  }
};

/// The relative perturbation: zonal and sectoral terms with a radial
/// envelope over the mantle, plus a lateral mean the reader removes.
double Shear(double r, double theta, double phi) {
  const auto d = 6371.0e3 - r;
  const auto envelope = d > 0.0 && d < 2.0e6 ? std::sin(std::numbers::pi * d / 2.0e6) : 0.0;
  return envelope * (0.02 + 0.1 * std::cos(theta) +
                     0.05 * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi));
}
double Bulk(double r, double theta, double phi) { return 0.5 * Shear(r, theta, phi); }

/// The lateral mean of a function at a radius under the grid's weights.
double LateralMean(const SurfaceGrid& grid, double r,
                   const std::function<double(double, double, double)>& f) {
  return grid.Integrate(grid.FromFunction([&](double theta, double phi) {
           return f(r, theta, phi);
         })) /
         (4.0 * std::numbers::pi);
}

TEST(ModelModuli, WrittenPerturbationsReadBackLessTheirLateralMeans) {
  auto scene = Scene::Make("moduli.nc");
  const SurfaceGrid grid(kLmax);
  const auto written = io::WriteModuli(scene.file, scene.model, grid, Shear, Bulk,
                                       {.elementMetres = 100.0e3});
  ASSERT_TRUE(written) << written.error().message;
  const auto read = io::ModelModuli::Read(scene.file, scene.model);
  ASSERT_TRUE(read) << read.error().message;
  EXPECT_FALSE(read->IsRadial());
  const auto shear = read->Shear();
  const auto bulk = read->Bulk();
  ASSERT_TRUE(shear);
  ASSERT_TRUE(bulk);
  const auto length = scene.model.ModelScales().Length();
  // at grid points of a few radii in the mantle: the written pattern
  // less its lateral mean, to the file's linear lateral interpolation
  // (exact at the grid's own points) and its radial rule
  auto worst = 0.0;
  for (const auto depthKm : {150.0, 700.0, 1500.0}) {
    const auto r = 6371.0e3 - depthKm * 1.0e3;
    const auto meanShear = LateralMean(grid, r, Shear);
    const auto meanBulk = LateralMean(grid, r, Bulk);
    for (const auto theta : grid.CoLatitudes()) {
      for (const auto phi : grid.Longitudes()) {
        const auto wantShear = (1.0 + Shear(r, theta, phi)) / (1.0 + meanShear) - 1.0;
        const auto wantBulk = (1.0 + Bulk(r, theta, phi)) / (1.0 + meanBulk) - 1.0;
        worst = std::max(worst, std::abs(shear(r / length, theta, phi) - wantShear));
        worst = std::max(worst, std::abs(bulk(r / length, theta, phi) - wantBulk));
      }
    }
  }
  EXPECT_LT(worst, 2e-4) << worst;
  // the stepper takes them where its functors would go
  const auto elastic = Rheology::Create(scene.material, {}).value();
  Viscoelastic3D::Options options;
  options.solver = Viscoelastic3D::Options::StepSolver::ConjugateGradients;
  options.moduli = {.shear = shear, .bulk = bulk};
  const Viscoelastic3D op(elastic, kLmax, [](double, double, double) { return 0.0; },
                          options);
  EXPECT_TRUE(op.HasLateralModuli());
  std::filesystem::remove(scene.file);
}

TEST(ModelModuli, ARadialFileIsRadial) {
  auto scene = Scene::Make("radial.nc");
  const SurfaceGrid grid(kLmax);
  const auto written = io::WriteModuli(scene.file, scene.model, grid, {}, {});
  ASSERT_TRUE(written) << written.error().message;
  const auto read = io::ModelModuli::Read(scene.file, scene.model);
  ASSERT_TRUE(read) << read.error().message;
  EXPECT_TRUE(read->IsRadial());
  EXPECT_EQ(read->Shear()(0.9, 1.0, 1.0), 0.0);
  EXPECT_EQ(read->Bulk()(0.9, 1.0, 1.0), 0.0);
  std::filesystem::remove(scene.file);
}

}  // namespace
