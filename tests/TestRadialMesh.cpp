// RadialMesh: geometry, reference element, conformity to model boundaries,
// truncation rules, and nodal sampling.

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/RadialMesh.hpp"

using sl3dpp::Deck;
using sl3dpp::EarthModel;
using sl3dpp::RadialMesh;
using sl3dpp::TruncationRule;

namespace {

EarthModel TwoLayerModel(double b = 6.0e6, double rhoLower = 5000.0,
                         double rhoUpper = 3000.0) {
  std::vector<double> radius, rho;
  for (int k = 0; k < 5; ++k) {
    radius.push_back(std::lerp(0.0, 0.5 * b, k / 4.0));
    rho.push_back(rhoLower);
  }
  for (int k = 0; k < 5; ++k) {
    radius.push_back(std::lerp(0.5 * b, b, k / 4.0));
    rho.push_back(rhoUpper);
  }
  auto deck = Deck::Create(std::move(radius), {{"rho", std::move(rho)}});
  return EarthModel::FromDeck(deck.value()).value();
}

}  // namespace

TEST(RadialMesh, SubdividesBreakpoints) {
  const std::vector<double> breakpoints{0.0, 1.0};
  auto mesh = RadialMesh::Create(breakpoints, {.drmax = 0.26}).value();
  EXPECT_EQ(mesh.ElementCount(), 4);  // ceil(1 / 0.26)
  EXPECT_EQ(mesh.NGll(), 5);
  EXPECT_EQ(mesh.NodeCount(), 4 * 4 + 1);
  EXPECT_DOUBLE_EQ(mesh.Left(0), 0.0);
  EXPECT_DOUBLE_EQ(mesh.Right(3), 1.0);
  // Shared edges agree exactly and the node radii are increasing.
  for (int e = 0; e + 1 < mesh.ElementCount(); ++e) {
    EXPECT_DOUBLE_EQ(mesh.Right(e), mesh.Left(e + 1));
    EXPECT_DOUBLE_EQ(mesh.Radius(e, 4), mesh.Radius(e + 1, 0));
  }
  EXPECT_EQ(mesh.ElementOf(0.3), 1);
  EXPECT_EQ(mesh.ElementOf(0.25), 1);  // a boundary resolves upward
  EXPECT_EQ(mesh.ElementOf(-1.0), 0);
  EXPECT_EQ(mesh.ElementOf(2.0), 3);
}

TEST(RadialMesh, ReferenceElementIsConsistent) {
  auto mesh =
      RadialMesh::Create(std::vector<double>{0.0, 1.0}, {.drmax = 1.0}).value();
  const int n = mesh.NGll();
  // GLL weights integrate constants on [-1, 1].
  double sum = 0.0;
  for (auto w : mesh.Weights()) sum += w;
  EXPECT_NEAR(sum, 2.0, 1e-14);
  // The derivative matrix kills constants and differentiates xi exactly.
  for (int k = 0; k < n; ++k) {
    double dConst = 0.0, dXi = 0.0;
    for (int i = 0; i < n; ++i) {
      dConst += mesh.Derivative(k, i);
      dXi += mesh.Derivative(k, i) * mesh.Points()[i];
    }
    EXPECT_NEAR(dConst, 0.0, 1e-12);
    EXPECT_NEAR(dXi, 1.0, 1e-12);
  }
}

TEST(RadialMesh, ConformsToModelBoundaries) {
  auto model = TwoLayerModel();
  auto mesh = RadialMesh::Create(model, {.lmax = 32}).value();
  // planetmodel's sizing rule, in the model's non-dimensional numbers.
  const double drmax = 0.1 * 1.0 / (32 + 1);
  // The internal boundary (0.5 in non-dimensional radius) is an element
  // edge, and every element is within drmax and inside one layer.
  bool boundaryIsEdge = false;
  for (int e = 0; e < mesh.ElementCount(); ++e) {
    EXPECT_LE(mesh.Right(e) - mesh.Left(e), drmax * (1 + 1e-12));
    boundaryIsEdge = boundaryIsEdge || mesh.Right(e) == 0.5;
    EXPECT_EQ(mesh.LayerOf(e), mesh.Left(e) < 0.5 ? 0 : 1);
  }
  EXPECT_TRUE(boundaryIsEdge);
}

TEST(RadialMesh, RangeAndOptionValidation) {
  auto model = TwoLayerModel();
  EXPECT_FALSE(RadialMesh::Create(model, {}).has_value());  // neither sizing
  EXPECT_FALSE(
      RadialMesh::Create(model, {.drmax = 1.0, .lmax = 4}).has_value());
  EXPECT_FALSE(
      RadialMesh::Create(model, {.drmax = 1e5, .rmax = 7.0e6}).has_value());
  // A mesh over the mantle only (SI range), still layer-tagged.
  auto mantle =
      RadialMesh::Create(model, {.drmax = 1e5, .rmin = 3.0e6}).value();
  EXPECT_NEAR(mantle.Left(0), 0.5, 1e-12);
  EXPECT_EQ(mantle.LayerOf(0), 1);
}

TEST(RadialMesh, TruncationRules) {
  auto mesh = RadialMesh::Create(std::vector<double>{0.0, 1.0}, {.drmax = 0.01})
                  .value();
  // Graded: rmax eps^(1/(l+1)).
  EXPECT_NEAR(mesh.TruncationRadius(1, TruncationRule::Graded, 1e-8), 1e-4,
              1e-12);
  EXPECT_NEAR(mesh.TruncationRadius(255, TruncationRule::Graded, 1e-8),
              std::pow(1e-8, 1.0 / 256.0), 1e-12);
  // Legacy: rmax (1 - 20/(l + 0.5)), clipped below.
  EXPECT_DOUBLE_EQ(mesh.TruncationRadius(2, TruncationRule::Legacy), 0.0);
  EXPECT_NEAR(mesh.TruncationRadius(100, TruncationRule::Legacy),
              1.0 - 20.0 / 100.5, 1e-12);
  // The start element contains the truncation radius.
  const int start = mesh.StartElement(100, TruncationRule::Legacy);
  EXPECT_LE(mesh.Left(start),
            mesh.TruncationRadius(100, TruncationRule::Legacy));
  EXPECT_GT(mesh.Right(start),
            mesh.TruncationRadius(100, TruncationRule::Legacy));
}

TEST(RadialMesh, NodalFieldsCarryOneSidedValues) {
  auto model = TwoLayerModel();
  auto nd = model.DimensionlessView();
  auto mesh = RadialMesh::Create(model, {.drmax = 5e5}).value();
  auto rho = mesh.Nodal(model, "rho");
  const int n = mesh.NGll();

  // Find the elements meeting at the internal boundary.
  int below = -1;
  for (int e = 0; e < mesh.ElementCount(); ++e) {
    if (mesh.Right(e) == 0.5) below = e;
  }
  ASSERT_GE(below, 0);
  const double lower = 5000.0 / sl3dpp::EarthMeanDensitySI;
  const double upper = 3000.0 / sl3dpp::EarthMeanDensitySI;
  EXPECT_NEAR(rho[below * n + (n - 1)], lower, 1e-12);
  EXPECT_NEAR(rho[(below + 1) * n + 0], upper, 1e-12);

  // Nodal gravity matches the model's at the nodes.
  auto gravity = mesh.NodalGravity(model);
  const auto radii = mesh.Radii();
  for (std::size_t i = 0; i < radii.size(); i += 7) {
    EXPECT_DOUBLE_EQ(gravity[i], nd.Gravity(radii[i]));
  }
}
