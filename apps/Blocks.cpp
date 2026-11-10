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
  Allow(block, {"deck", "format", "moduli", "viscosity"});
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
  if (Has(block, "viscosity")) {
    const auto& viscosity = block.At("viscosity");
    Allow(viscosity, {"file", "reference"});
    out.viscosity = PathOf(viscosity, "file", directory);
    if (Has(viscosity, "reference")) {
      out.viscosity_file_radial =
          Choice<bool>(viscosity.At("reference"),
                       {{"geometric_mean", false}, {"file_radial", true}});
    }
  }
  return out;
}

namespace {

/// A finite number.
double Finite(const Json& item) {
  const auto value = item.AsNumber();
  if (!std::isfinite(value)) Fail(item, "is not a finite number");
  return value;
}

}  // namespace

TimesBlock ReadTimesBlock(const Json& block) {
  Allow(block, {"start_kyr", "end_kyr", "nodes", "at_kyr"});
  const auto even = Has(block, "end_kyr") || Has(block, "nodes");
  const auto listed = Has(block, "at_kyr");
  if (even && listed) {
    Fail(block,
         "gives the nodes twice: either `end_kyr` and `nodes` (evenly "
         "spaced from `start_kyr`) or `at_kyr` (the nodes listed)");
  }
  if (!even && !listed) {
    Fail(block,
         "gives no nodes: either `end_kyr` and `nodes` (evenly spaced from "
         "`start_kyr`) or `at_kyr` (the nodes listed)");
  }
  auto out = TimesBlock{};
  if (listed) {
    if (Has(block, "start_kyr")) {
      Fail(block.At("start_kyr"),
           "is given by `at_kyr`, whose first entry is the start");
    }
    const auto& at = block.At("at_kyr");
    for (const auto& item : at.AsArray()) {
      const auto value = Finite(item);
      if (!out.nodes_kyr.empty() && !(value > out.nodes_kyr.back())) {
        Fail(item,
             "is not later than the node before it: the nodes are "
             "strictly increasing");
      }
      out.nodes_kyr.push_back(value);
    }
    if (out.nodes_kyr.size() < 2) Fail(at, "is fewer than two nodes");
    for (auto n = std::size_t{1}; n < out.nodes_kyr.size(); ++n) {
      out.steps_kyr.push_back(out.nodes_kyr[n] - out.nodes_kyr[n - 1]);
    }
    return out;
  }
  for (const auto& [name, other] :
       {std::pair{"end_kyr", "nodes"}, std::pair{"nodes", "end_kyr"}}) {
    if (!Has(block, name)) {
      Fail(block, std::format("gives `{}` without `{}`: evenly spaced nodes "
                              "need both",
                              other, name));
    }
  }
  const auto start =
      Has(block, "start_kyr") ? Finite(block.At("start_kyr")) : 0.0;
  const auto end = Finite(block.At("end_kyr"));
  const auto nodes = Integer(block, "nodes");
  if (!(end > start)) {
    Fail(block.At("end_kyr"), "is not later than start_kyr");
  }
  if (nodes < 2) Fail(block.At("nodes"), "is fewer than two nodes");
  const auto spacing = (end - start) / (nodes - 1);
  for (auto n = 0; n < nodes; ++n) {
    out.nodes_kyr.push_back(n + 1 == nodes ? end : start + n * spacing);
  }
  out.steps_kyr.assign(static_cast<std::size_t>(nodes - 1), spacing);
  return out;
}

SolverBlock ReadSolverBlock(const Json& block, int lmax) {
  Allow(block, {"rtol", "max_iterations", "transform_degrees", "cache_mb",
                "scheme", "step"});
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
  if (Has(block, "scheme")) {
    out.scheme =
        Choice<Scheme>(block.At("scheme"),
                       {{"exponential_trapezoid", Scheme::ExponentialTrapezoid},
                        {"backward_euler", Scheme::BackwardEuler}});
  }
  if (Has(block, "step")) {
    using StepSolver = Viscoelastic3D::Options::StepSolver;
    out.step = Choice<StepSolver>(
        block.At("step"),
        {{"fixed_point", StepSolver::FixedPoint},
         {"conjugate_gradients", StepSolver::ConjugateGradients}});
  }
  return out;
}

OutputBlock ReadOutputBlock(const Json& block,
                            const std::filesystem::path& directory) {
  Allow(block, {"surface", "exchange", "volume", "exchange_nodes"});
  auto out = OutputBlock{};
  out.surface = PathOf(block, "surface", directory);
  if (Has(block, "exchange")) {
    out.exchange = PathOf(block, "exchange", directory);
  }
  if (Has(block, "volume")) out.volume = PathOf(block, "volume", directory);
  if (Has(block, "exchange_nodes")) {
    for (const auto& item : block.At("exchange_nodes").AsArray()) {
      const auto value = item.AsNumber();
      if (value != std::floor(value) || value < 0) {
        Fail(item, "is not a node index (an integer, 0 or more)");
      }
      out.exchange_nodes.push_back(static_cast<int>(value));
    }
  }
  return out;
}

}  // namespace sl3dpp::app
