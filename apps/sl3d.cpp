/**
 * @file sl3d.cpp
 * @brief The sl3d program: the library's common capabilities from JSON
 * configs, one subcommand per capability.
 *
 *   sl3d <command> <config.json>
 *   sl3d --help | --version
 *
 * Each command reads one strict-JSON config (unknown members refused,
 * relative paths against the config's directory), runs, writes its
 * outputs with provenance, and prints a one-line summary to stderr.
 */

#include <array>
#include <chrono>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

#include "Commands.hpp"
#include "sl3dpp/Version.hpp"

namespace {

constexpr auto kUsage =
    "usage: sl3d <command> <config.json>\n"
    "\n"
    "commands:\n"
    "  love-numbers   the SI Love-number table of a radial model\n"
    "  fingerprint    an elastic sea-level fingerprint on a present-day\n"
    "                 state read from a planetmodel surface file\n"
    "  evolve         a 3D (lateral-viscosity) GIA run under an ice\n"
    "                 history, restartable from disk checkpoints\n"
    "  evolve1d       its 1D counterpart, on the deck's own viscosity\n"
    "  kernel         adjoint sensitivity kernels of configured\n"
    "                 measurements, 3D lateral viscosity\n"
    "  kernel1d       its 1D counterpart\n"
    "\n"
    "sl3d --version prints the library versions. Every command's config\n"
    "is documented in docs/sl3d-configs.md, also at\n"
    "https://da380.github.io/sl3dpp/\n";

struct Command {
  std::string_view name;
  std::string (*run)(const std::filesystem::path&);
};

constexpr auto kCommands = std::array{
    Command{"love-numbers", sl3dpp::app::RunLoveNumbers},
    Command{"fingerprint", sl3dpp::app::RunFingerprint},
    Command{"evolve", sl3dpp::app::RunEvolve3D},
    Command{"evolve1d", sl3dpp::app::RunEvolve},
    Command{"kernel", sl3dpp::app::RunKernel3D},
    Command{"kernel1d", sl3dpp::app::RunKernel1D},
};

}  // namespace

int main(int argc, char** argv) {
  const auto args = std::span(argv, static_cast<std::size_t>(argc));
  if (argc == 2 && std::string_view{args[1]} == "--version") {
    std::cout << std::format("sl3d (sl3dpp {}, planetmodel {})\n",
                             sl3dpp::Version(), sl3dpp::PlanetModelVersion());
    return 0;
  }
  if (argc == 2 && (std::string_view{args[1]} == "--help" ||
                    std::string_view{args[1]} == "-h")) {
    std::cout << kUsage;
    return 0;
  }
  if (argc != 3) {
    std::cerr << kUsage;
    return 2;
  }
  const auto name = std::string_view{args[1]};
  for (const auto& command : kCommands) {
    if (command.name != name) continue;
    try {
      const auto start = std::chrono::steady_clock::now();
      const auto summary = command.run(std::filesystem::path{args[2]});
      const auto seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start)
              .count();
      std::cerr << std::format("sl3d {}: {} in {:.2f} s\n", name, summary,
                               seconds);
      return 0;
    } catch (const std::exception& error) {
      std::cerr << std::format("sl3d {}: {}\n", name, error.what());
      return 1;
    }
  }
  std::cerr << std::format("sl3d: \"{}\" is not a command\n\n{}", name,
                           kUsage);
  return 2;
}
