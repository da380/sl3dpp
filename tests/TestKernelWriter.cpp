/**
 * @file TestKernelWriter.cpp
 * @brief Kernels out: Kernels3D-layout values written as
 * a planetmodel model file and read back.
 *
 * The gates read the file back at the memory nodes and grid points,
 * where the writing is exact to rounding for ARBITRARY values — the
 * file stores a shared element-edge radius once per side, so piecewise
 * fields pass as they are; deterministic noise is the hardest case for
 * any smoothing. A plain Value() read resolves a shared radius to the
 * element above, which the round trip accounts for. The pairing
 * dJ = sum_node int K p dOmega computed from the file's content then
 * matches the in-memory pairing to rounding; that in-memory pairing is
 * the directional derivative by TestAdjoint3D's Taylor gate, which
 * closes the loop. Both memory point sets are written, the Gauss
 * default and GLL.
 *
 * With SL3DPP_KERNEL_FILE_OUT set, a smooth kernel is also written
 * to that path with its pairing in the meta, for the Python's own
 * reading (scripts/check_kernel_file.py) and pictures.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <planetmodel/modelfile.hpp>
#include <span>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/KernelWriter.hpp"

namespace {

using namespace sl3dpp;
using io::KernelField;
using io::WriteKernels;

constexpr auto kLmax = 8;

std::filesystem::path DataPath(const std::string& name) {
  return std::filesystem::path(SL3DPP_TEST_DATA_DIR) / name;
}

/// Deterministic pseudo-noise in [-0.5, 0.5].
double Noise(std::size_t k) {
  const auto x = std::sin(12.9898 * static_cast<double>(k + 1)) * 43758.5453;
  return x - std::floor(x) - 0.5;
}

struct Body {
  EarthModel model;
  Material material;
  Rheology rheology;
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
    auto viscosity = std::map<std::size_t, RadialValue>{};
    auto viscous = std::vector<std::size_t>{};
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
        viscous.push_back(i);
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    return Body{std::move(model), std::move(material), std::move(rheology),
                std::move(viscous)};
  }
};

/// A band-limited test perturbation of the log viscosity.
double Direction(double theta, double phi) {
  const auto s = std::sin(theta);
  return 0.3 + s * s * std::cos(2.0 * phi) - 0.5 * std::cos(theta);
}

/// The file-side pairing sum_node w r^2 int K p dOmega from the read
/// field alone, non-dimensionalised back under the model's scales.
double FilePairing(const planetmodel::ModelFile& file,
                   const std::string& name, const GridCoupling& coupling,
                   const RadialMesh& mesh, double length,
                   std::span<const double> p) {
  const auto& grid = coupling.Grid();
  const auto fieldSize = grid.FieldSize();
  auto fields = std::map<int, planetmodel::ScalarField>{};
  auto product = std::vector<double>(fieldSize);
  auto total = 0.0;
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    const auto layer = mesh.LayerOf(coupling.ElementOfNode(node));
    if (!fields.contains(layer)) {
      fields.emplace(layer, file.FieldOn(layer, name).AsScalar());
    }
    const auto& field = fields.at(layer);
    const auto r = coupling.RadiusOfNode(node);
    const auto rSI = r * length;
    auto x = std::size_t{0};
    for (const auto theta : grid.CoLatitudes()) {
      for (const auto phi : grid.Longitudes()) {
        product[x] = field.Value(rSI, theta, phi) * p[x];
        ++x;
      }
    }
    total += coupling.WeightOfNode(node) * r * r * grid.Integrate(product);
  }
  return total * length * length * length;
}

}  // namespace

// Writing and reading back are exact at the memory nodes and grid
// points, for arbitrary values and either memory point set, and the
// pairing against a perturbation survives the file unchanged.
TEST(KernelWriter, RoundTripIsExactAtTheNodes) {
  const auto body = Body::Make();
  const auto& mesh = body.material.Mesh();
  const auto length = body.model.ModelScales().Length();
  for (const auto set : {MemoryPointSet::Gauss, MemoryPointSet::Gll}) {
    Viscoelastic3D op(body.rheology, kLmax,
                      [](double, double, double) { return 0.0; },
                      {.memory_points = set});
    const auto& coupling = op.Coupling();
    const auto& grid = coupling.Grid();
    const auto fieldSize = grid.FieldSize();
    const auto count = static_cast<std::size_t>(coupling.NodeCount());

    // Arbitrary values: the file stores one value per node with a
    // shared element-edge radius held once per side, so a piecewise
    // field passes as it is; noise is the hardest case for any
    // smoothing, radial or lateral.
    auto values = std::vector<double>(count * fieldSize);
    for (auto i = std::size_t{0}; i < values.size(); ++i) {
      values[i] = Noise(i);
    }
    const auto path =
        std::filesystem::temp_directory_path() /
        (set == MemoryPointSet::Gauss ? "sl3dpp_kernel_writer_gauss.nc"
                                      : "sl3dpp_kernel_writer_gll.nc");
    const auto written = WriteKernels(
        path, op, body.model,
        {{.name = "log_viscosity_kernel",
          .meaning = "derivative of the test objective with respect to the "
                     "log viscosity, per unit volume",
          .values = values}});
    ASSERT_TRUE(written) << written.error().message;

    const auto file = planetmodel::ReadModelFile(path);
    std::filesystem::remove(path);

    // The written values come back at the memory nodes and grid points.
    auto fields = std::map<int, planetmodel::ScalarField>{};
    auto worst = 0.0;
    auto scale = 0.0;
    for (auto node = 0; node < coupling.NodeCount(); ++node) {
      const auto layer = mesh.LayerOf(coupling.ElementOfNode(node));
      if (!fields.contains(layer)) {
        fields.emplace(
            layer, file.FieldOn(layer, "log_viscosity_kernel").AsScalar());
      }
      const auto& field = fields.at(layer);
      const auto r = coupling.RadiusOfNode(node);
      // Value() reads a shared element-edge radius right-continuously
      // within a layer (both one-sided values are in the file; the
      // element above answers), so a GLL node at its element's top
      // expects the next element's bottom copy — unless that edge is a
      // layer boundary, where each layer's own field answers with its
      // own side.
      auto expected = node;
      if (node + 1 < coupling.NodeCount() &&
          coupling.RadiusOfNode(node + 1) == r &&
          mesh.LayerOf(coupling.ElementOfNode(node + 1)) == layer) {
        expected = node + 1;
      }
      auto x = std::size_t{0};
      for (const auto theta : grid.CoLatitudes()) {
        for (const auto phi : grid.Longitudes()) {
          const auto wrote =
              values[static_cast<std::size_t>(expected) * fieldSize + x];
          const auto scaled =
              field.Value(r * length, theta, phi) *
              coupling.WeightOfNode(expected) *
              coupling.RadiusOfNode(expected) *
              coupling.RadiusOfNode(expected) * length * length * length;
          worst = std::max(worst, std::abs(scaled - wrote));
          scale = std::max(scale, std::abs(wrote));
          ++x;
        }
      }
    }
    EXPECT_LT(worst, 1e-11 * scale);

    // The pairing against a perturbation survives the file. The nodal
    // quadrature reads each node's own side, so this gate runs on the
    // Gauss set, whose memory radii are strictly interior; the GLL
    // data's integrity is the round trip above (a side-aware quadrature
    // pairs it the same way).
    const auto p = grid.FromFunction(Direction);
    auto inMemory = 0.0;
    auto product = std::vector<double>(fieldSize);
    for (auto node = 0; node < coupling.NodeCount(); ++node) {
      const auto row =
          std::span(values).subspan(static_cast<std::size_t>(node) * fieldSize,
                                    fieldSize);
      for (auto x = std::size_t{0}; x < fieldSize; ++x) {
        product[x] = row[x] * p[x];
      }
      inMemory += grid.Integrate(product);
    }
    if (set == MemoryPointSet::Gauss) {
      const auto fromFile = FilePairing(file, "log_viscosity_kernel",
                                        coupling, mesh, length, p);
      EXPECT_NEAR(fromFile, inMemory, 1e-11 * std::abs(inMemory));
    }

    // The file's shape: the model's skeleton, the viscous layers held.
    const auto& skeleton = file.GetSkeleton().Boundaries();
    ASSERT_EQ(skeleton.size(), body.model.Boundaries().size());
    for (auto i = std::size_t{0}; i < skeleton.size(); ++i) {
      EXPECT_EQ(skeleton[i], body.model.Boundaries()[i]);
    }
    for (auto i = std::size_t{0}; i < body.model.LayerCount(); ++i) {
      EXPECT_EQ(file.HasField(static_cast<int>(i), "log_viscosity_kernel"),
                std::ranges::contains(body.viscous, i));
    }
  }
}

// Mismatched sizes and nameless kernels are refused before any writing.
TEST(KernelWriter, MalformedKernelsAreRefused) {
  const auto body = Body::Make();
  Viscoelastic3D op(body.rheology, kLmax,
                    [](double, double, double) { return 0.0; });
  const auto path = std::filesystem::temp_directory_path() /
                    "sl3dpp_kernel_writer_refused.nc";
  auto values = std::vector<double>(7);
  const auto wrong = WriteKernels(
      path, op, body.model, {{.name = "k", .meaning = "", .values = values}});
  ASSERT_FALSE(wrong);
  EXPECT_NE(wrong.error().message.find("memory nodes"), std::string::npos);
  const auto nameless = WriteKernels(
      path, op, body.model, {{.name = "", .meaning = "", .values = values}});
  ASSERT_FALSE(nameless);
  EXPECT_FALSE(std::filesystem::exists(path));
}

// With SL3DPP_KERNEL_FILE_OUT set, a smooth kernel goes to that path
// with its pairing and the perturbation's parameters in the meta, for
// scripts/check_kernel_file.py and the pictures; otherwise skipped.
TEST(KernelWriter, ASmoothKernelForThePythonWhenAsked) {
  const auto* given = std::getenv("SL3DPP_KERNEL_FILE_OUT");
  if (!given) GTEST_SKIP() << "SL3DPP_KERNEL_FILE_OUT is not set";
  const auto body = Body::Make();
  Viscoelastic3D op(body.rheology, kLmax,
                    [](double, double, double) { return 0.0; });
  const auto& coupling = op.Coupling();
  const auto& grid = coupling.Grid();
  const auto fieldSize = grid.FieldSize();

  // A two-signed pattern with a peak beside a "receiver",
  // write_kernel.cpp's shape, as the angular density K = w r^2 F of a
  // volume density F whose radial profile is quadratic, so the pairing
  // survives the file to rounding and the Python's own quadrature
  // recovers it exactly (scripts/check_kernel_file.py).
  const auto smooth = [](double r, double theta, double phi) {
    const auto delta = std::acos(
        std::clamp(std::cos(theta) * 0.5 +
                       std::sin(theta) * std::sqrt(3.0) / 2.0 *
                           std::cos(phi - 0.5),
                   -1.0, 1.0));
    const auto pattern = std::cos(2.0 * delta) +
                         5.0 * std::exp(-delta * delta / (2.0 * 0.2 * 0.2));
    return (0.5 + r + 2.0 * r * r) * pattern;
  };
  auto values = std::vector<double>(
      static_cast<std::size_t>(coupling.NodeCount()) * fieldSize);
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    const auto r = coupling.RadiusOfNode(node);
    const auto w = coupling.WeightOfNode(node) * r * r;
    auto x = std::size_t{0};
    for (const auto theta : grid.CoLatitudes()) {
      for (const auto phi : grid.Longitudes()) {
        values[static_cast<std::size_t>(node) * fieldSize + x++] =
            w * smooth(r, theta, phi);
      }
    }
  }
  const auto p = grid.FromFunction(Direction);
  auto pairing = 0.0;
  auto product = std::vector<double>(fieldSize);
  for (auto node = 0; node < coupling.NodeCount(); ++node) {
    for (auto x = std::size_t{0}; x < fieldSize; ++x) {
      product[x] =
          values[static_cast<std::size_t>(node) * fieldSize + x] * p[x];
    }
    pairing += grid.Integrate(product);
  }
  const auto written = WriteKernels(
      std::filesystem::path(given), op, body.model,
      {{.name = "log_viscosity_kernel",
        .meaning = "derivative of the test objective with respect to the "
                   "log viscosity, per unit volume",
        .values = values}},
      {.name = "sl3dpp smooth test kernel",
       .meta = {{"pairing", pairing},
                {"length_scale", body.model.ModelScales().Length()}}});
  ASSERT_TRUE(written) << written.error().message;
}
