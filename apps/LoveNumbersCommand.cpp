#include <format>
#include <fstream>
#include <planetmodel/json.hpp>
#include <stdexcept>

#include "Commands.hpp"
#include "Input.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Version.hpp"

namespace sl3dpp::app {

DeckFormat FormatOf(ModelFormat format) {
  switch (format) {
    case ModelFormat::Gia:
      return GiaFormat();
    case ModelFormat::Mineos:
      return MineosFormat();
    case ModelFormat::Elastic:
      break;
  }
  return DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0);
}

namespace {

/// The members every model-bearing config shares.
ModelFormat ReadFormat(const Json& top) {
  if (!Has(top, "model_format")) return ModelFormat::Gia;
  return Choice<ModelFormat>(top.At("model_format"),
                             {{"gia", ModelFormat::Gia},
                              {"mineos", ModelFormat::Mineos},
                              {"elastic", ModelFormat::Elastic}});
}

int ReadLmax(const Json& top) {
  const auto lmax = Integer(top, "lmax");
  if (lmax < 0) Fail(top.At("lmax"), "is not a degree");
  return lmax;
}

/// The model, mesh and material of a config's deck.
Material MaterialOf(const std::filesystem::path& deck_path, ModelFormat format,
                    int lmax, const Json& top) {
  const auto deck = Deck::Read(deck_path, FormatOf(format));
  if (!deck) Fail(top.At("model"), deck.error().message);
  const auto model = EarthModel::FromDeck(*deck);
  if (!model) Fail(top.At("model"), model.error().message);
  auto mesh_options = RadialMesh::Options{};
  mesh_options.lmax = lmax;
  const auto mesh = RadialMesh::Create(*model, mesh_options);
  if (!mesh) Fail(top.At("model"), mesh.error().message);
  const auto material = Material::Create(*model, *mesh);
  if (!material) Fail(top.At("model"), material.error().message);
  return *material;
}

}  // namespace

LoveNumbersInput ReadLoveNumbersInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  Allow(top, {"model", "model_format", "lmax", "output"});
  const auto directory = config.parent_path();
  return LoveNumbersInput{.model = PathOf(top, "model", directory),
                          .format = ReadFormat(top),
                          .lmax = ReadLmax(top),
                          .output = PathOf(top, "output", directory)};
}

std::string RunLoveNumbers(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  const auto input = ReadLoveNumbersInput(config);
  const auto material = MaterialOf(input.model, input.format, input.lmax, top);
  const auto love = ComputeLoveNumbers(material, input.lmax);

  if (const auto parent = input.output.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  auto out = std::ofstream(input.output);
  if (!out) {
    throw std::runtime_error(
        std::format("cannot write {}", input.output.string()));
  }
  out << std::format("# sl3d love-numbers (sl3dpp {})\n", Version())
      << std::format("# created {}\n", CreatedNow())
      << std::format("# config {}\n", config.string())
      << std::format("# model {}\n", input.model.string())
      << std::format("# lmax {}\n", input.lmax)
      << std::format("# radius_m {:.17g}\n", love.radius)
      << std::format("# surface_gravity_m_s2 {:.17g}\n", love.surface_gravity)
      << std::format("# degree_zero_axial h_c {:.17g} k_c {:.17g} m_u {:.17g} "
                     "m_phi {:.17g} m_c {:.17g}\n",
                     love.h_c, love.k_c, love.m_u, love.m_phi, love.m_c)
      << std::format("# reciprocity_residual {:.3g}\n",
                     love.ReciprocityResidual())
      << "# l h_u l_u k_u h_phi l_phi k_phi h_v l_v k_v h_t l_t k_t\n";
  for (auto i = std::size_t{0}; i < love.degrees.size(); ++i) {
    out << std::format(
        "{} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} "
        "{:.17g} {:.17g} {:.17g} {:.17g}\n",
        love.degrees[i], love.h_u[i], love.l_u[i], love.k_u[i], love.h_phi[i],
        love.l_phi[i], love.k_phi[i], love.h_v[i], love.l_v[i], love.k_v[i],
        love.h_t[i], love.l_t[i], love.k_t[i]);
  }
  return std::format("wrote {} (degrees 0..{})", input.output.string(),
                     input.lmax);
}

}  // namespace sl3dpp::app
