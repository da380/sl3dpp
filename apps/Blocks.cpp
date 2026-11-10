#include "Blocks.hpp"

#include <cmath>
#include <format>

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

ModelBlock ReadModelBlock(const Json& block,
                          const std::filesystem::path& directory) {
  Allow(block, {"deck", "format", "moduli"});
  auto out = ModelBlock{};
  out.deck = PathOf(block, "deck", directory);
  if (Has(block, "format")) {
    out.format = Choice<ModelFormat>(block.At("format"),
                                     {{"maxwell", ModelFormat::Gia},
                                      {"gia", ModelFormat::Gia},
                                      {"mineos", ModelFormat::Mineos},
                                      {"elastic", ModelFormat::Elastic}});
  }
  if (Has(block, "moduli")) {
    const auto& moduli = block.At("moduli");
    Allow(moduli, {"file", "reference"});
    out.moduli = PathOf(moduli, "file", directory);
    if (Has(moduli, "reference")) {
      out.moduli_file_radial =
          Choice<bool>(moduli.At("reference"),
                       {{"lateral_mean", false}, {"file_radial", true}});
    }
  }
  return out;
}

SolverBlock ReadSolverBlock(const Json& block, int lmax) {
  Allow(block, {"rtol", "max_iterations", "transform_degrees", "cache_mb"});
  auto out = SolverBlock{};
  out.rtol = Number(block, "rtol", out.rtol);
  if (!(out.rtol > 0.0)) Fail(block.At("rtol"), "is not a positive tolerance");
  out.max_iterations = Integer(block, "max_iterations", out.max_iterations);
  if (out.max_iterations < 1) {
    Fail(block.At("max_iterations"), "is not a positive count");
  }
  if (Has(block, "transform_degrees")) {
    for (const auto& item : block.At("transform_degrees").AsArray()) {
      const auto value = item.AsNumber();
      if (value != std::floor(value) || value < 2 || value > lmax) {
        Fail(item, std::format("is not a degree in [2, lmax = {}]", lmax));
      }
      out.transform_degrees.push_back(static_cast<int>(value));
    }
  }
  const auto cache_mb = Integer(block, "cache_mb", 0);
  if (cache_mb < 0) {
    Fail(block.At("cache_mb"), "is not a cache size in MiB (0 or more)");
  }
  out.cache_mb = static_cast<std::size_t>(cache_mb);
  return out;
}

OutputBlock ReadOutputBlock(const Json& block,
                            const std::filesystem::path& directory) {
  Allow(block, {"surface", "exchange", "volume"});
  auto out = OutputBlock{};
  out.surface = PathOf(block, "surface", directory);
  if (Has(block, "exchange")) {
    out.exchange = PathOf(block, "exchange", directory);
  }
  if (Has(block, "volume")) out.volume = PathOf(block, "volume", directory);
  return out;
}

}  // namespace sl3dpp::app
