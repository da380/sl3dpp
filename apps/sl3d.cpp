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
 *
 * Built with SL3DPP_WITH_MPI, the commands with a 3D run (response,
 * sea-level, kernel) run under mpiexec over several ranks, one radial
 * segment each (Ranks.hpp); rank 0 writes and prints, and a failure on any
 * rank ends the job.
 */

#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <ostream>
#include <span>
#include <string>
#include <string_view>

#include "Commands.hpp"
#include "Ranks.hpp"
#include "sl3dpp/Version.hpp"

#ifdef SL3DPP_HAVE_MPI
#include <mpi.h>
#endif

namespace {

constexpr auto kUsage =
    "usage: sl3d <command> <config.json>\n"
    "\n"
    "commands:\n"
    "  love-numbers   the SI Love-number table of a radial model\n"
    "  response       the response to a prescribed surface load and tide,\n"
    "                 no ocean: the instantaneous elastic response or, with\n"
    "                 a times block, the viscoelastic response through\n"
    "                 time; 1D or 3D as the model block's moduli (and\n"
    "                 viscosity) make it\n"
    "  sea-level      the sea level equation under an ice load on a state\n"
    "                 read from planetmodel surface files: the elastic\n"
    "                 fingerprint of an ice change or, with a times block,\n"
    "                 GIA through time under an ice history, restartable\n"
    "                 from disk checkpoints; 1D or 3D as the model block's\n"
    "                 viscosity (and moduli) make it\n"
    "  kernel1d       adjoint sensitivity kernels of configured\n"
    "                 measurements of a sea-level run through time, on a\n"
    "                 radial model\n"
    "  kernel         their 3D counterpart, on a lateral viscosity\n"
    "\n"
    "sl3d --version prints the library versions. Every command's config\n"
    "is documented in docs/sl3d-configs.md, also at\n"
    "https://da380.github.io/sl3dpp/\n";

struct Command {
  std::string_view name;
  std::string (*run)(const std::filesystem::path&);
  bool distributes;  ///< runs over several ranks
};

constexpr auto kCommands = std::array{
    Command{"love-numbers", sl3dpp::app::RunLoveNumbers, false},
    Command{"response", sl3dpp::app::RunResponse, true},
    Command{"sea-level", sl3dpp::app::RunSeaLevel, true},
    Command{"kernel1d", sl3dpp::app::RunKernel1D, false},
    Command{"kernel", sl3dpp::app::RunKernel3D, true},
};

/// MPI from the start of main to its end (nothing without MPI).
class MpiSession {
 public:
  MpiSession([[maybe_unused]] int* argc, [[maybe_unused]] char*** argv) {
#ifdef SL3DPP_HAVE_MPI
    auto provided = 0;
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    sl3dpp::app::StartRanks();
#endif
  }
  ~MpiSession() {
#ifdef SL3DPP_HAVE_MPI
    MPI_Finalize();
#endif
  }
  MpiSession(const MpiSession&) = delete;
  MpiSession& operator=(const MpiSession&) = delete;

  /// Ends every rank's run: a rank that failed alone would leave the
  /// others waiting at their next collective.
  [[noreturn]] static void Abort(int code) {
#ifdef SL3DPP_HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, code);
#endif
    std::exit(code);
  }
};

int Run(int argc, char** argv) {
  const auto args = std::span(argv, static_cast<std::size_t>(argc));
  const auto& ranks = sl3dpp::app::TheRanks();
  // the other ranks are silent; rank 0 speaks for the run
  static auto silent = std::ostream(nullptr);
  auto& err = ranks.First() ? std::cerr : silent;
  auto& out = ranks.First() ? std::cout : silent;
  if (argc == 2 && std::string_view{args[1]} == "--version") {
    out << std::format("sl3d (sl3dpp {}, planetmodel {})\n", sl3dpp::Version(),
                       sl3dpp::PlanetModelVersion());
    return 0;
  }
  if (argc == 2 && (std::string_view{args[1]} == "--help" ||
                    std::string_view{args[1]} == "-h")) {
    out << kUsage;
    return 0;
  }
  if (argc != 3) {
    err << kUsage;
    return 2;
  }
  const auto name = std::string_view{args[1]};
  for (const auto& command : kCommands) {
    if (command.name != name) continue;
    if (ranks.Distributed() && !command.distributes) {
      err << std::format(
          "sl3d {}: runs on one process; only response, sea-level and "
          "kernel run over several ranks (this run has {})\n",
          name, ranks.size);
      return 2;
    }
    try {
      const auto start = std::chrono::steady_clock::now();
      const auto summary = command.run(std::filesystem::path{args[2]});
      const auto seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start)
              .count();
      const auto over = ranks.Distributed()
                            ? std::format(" on {} ranks", ranks.size)
                            : std::string{};
      err << std::format("sl3d {}: {} in {:.2f} s{}\n", name, summary, seconds,
                         over);
      return 0;
    } catch (const std::exception& error) {
      if (!ranks.Distributed()) {
        std::cerr << std::format("sl3d {}: {}\n", name, error.what());
        return 1;
      }
      std::cerr << std::format("sl3d {}: rank {}: {}\n", name, ranks.rank,
                               error.what());
      MpiSession::Abort(1);
    }
  }
  err << std::format("sl3d: \"{}\" is not a command\n\n{}", name, kUsage);
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  const MpiSession session(&argc, &argv);
  return Run(argc, argv);
}
