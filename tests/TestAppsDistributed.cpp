// sl3d's 3D runs over MPI ranks (apps/Ranks.hpp), built with
// SL3DPP_WITH_MPI and run under mpiexec: solve's 3-D run (a migrating one
// under a tide among them), its fingerprint
// on lateral moduli, its static problem there without an ocean and the
// volume file it writes, its viscoelastic run without an ocean, a run
// under a seismic onset between two nodes, and
// kernel's 3-D runs (through time
// with an ocean, the static problem on lateral moduli without one, a run
// on its own clock without one, a change at the start over listed dates),
// over the ranks against the same configs on one process (rank 0 alone,
// as a run of one rank), a run resumed from per-rank checkpoints against
// the uninterrupted one, the trajectory store in per-rank subdirectories
// against the in-memory recording (recorded by kernel itself, or by solve
// and swept by kernel without the forward run, a store over another count
// of ranks refused), a recording run resumed from per-rank checkpoints
// continuing its store to the uninterrupted one's, a 1-D run of solve,
// which is the first rank's alone,
// what kernel refuses on several ranks, and tangent's 3-D runs (along the
// ice, the viscosity and the moduli; the total derivative of a calibrated
// run, its linearised calibration collective) over the ranks against one
// process, a 1-D one being the first rank's alone, and kernel's
// second-order kernels of a 3-D run against one process. Rank 0 writes
// the outputs and compares them; every rank's verdict is reduced to rank
// 0, which alone prints.

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <planetmodel/error.hpp>
#include <planetmodel/modelfile.hpp>
#include <planetmodel/surfacefile.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include "AppFixtures.hpp"
#include "Commands.hpp"
#include "Ranks.hpp"
#include "RankListener.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/ModelModuli.hpp"
#include "sl3dpp/io/TrajectoryStore.hpp"

namespace {

namespace fs = std::filesystem;
namespace pm = planetmodel;
using namespace sl3dpp;

int Rank() { return app::TheRanks().rank; }
int Size() { return app::TheRanks().size; }
void Barrier() { MPI_Barrier(MPI_COMM_WORLD); }

double Relative(const std::vector<double>& got,
                const std::vector<double>& want) {
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    scale = std::max(scale, std::abs(want[i]));
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  return scale > 0.0 ? diff / scale : diff;
}

/// The departure allowed a run over the ranks from the one-process run of
/// the same config: the two are the same algebra summed in other orders
/// (measured up to 8e-11 of the fields), but either may close a solve a
/// pass later than the other at the solvers' tolerance (by default 1e-9).
constexpr auto kOverRanks = 1e-8;

class AppsOverRanks : public ::testing::Test {
 protected:
  void SetUp() override {
    // one directory for every rank of this run (named by the rank count,
    // so that runs over different counts may go side by side)
    dir_ = fs::temp_directory_path() /
           std::format(
               "sl3dpp_test_apps_mpi{}_{}", Size(),
               ::testing::UnitTest::GetInstance()->current_test_info()->name());
    if (Rank() == 0) {
      fs::remove_all(dir_);
      fs::create_directories(dir_);
      test::WriteToyHistory(dir_ / "history.nc");
    }
    Barrier();
  }
  void TearDown() override {
    Barrier();
    if (Rank() == 0) fs::remove_all(dir_);
  }

  /// A config under the test's directory, written by rank 0.
  fs::path Write(const std::string& name, const std::string& text) const {
    const auto path = dir_ / name;
    if (Rank() == 0) std::ofstream(path) << text;
    Barrier();
    return path;
  }

  /// f on rank 0 alone, as a run of one rank; the others wait.
  template <class F>
  void OneProcess(F&& f) const {
    if (Rank() == 0) {
      const app::RanksOverride alone(app::Ranks{});
      f();
    }
    Barrier();
  }

  /// A 3-D run through time on the toy history, its state at the window's
  /// start, or with `ocean` given at its end with these members beside it,
  /// under the history as a `thickness` term.
  [[nodiscard]] static std::string Body(const std::string& ocean = "") {
    const auto data = std::string{SL3DPP_TEST_DATA_DIR};
    const auto at = ocean.empty() ? std::string{"start"} : std::string{"end"};
    return std::string{R"(
      "problem": "quasi_static",
      "model": {"deck": ")"} +
           data + R"(/prem8.elastic.deck", "format": "elastic",
                "viscosity": {"file": ")" +
           data + R"(/viscosity_log.nc"}},
      "discretisation": {"lmax": 8},
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4},
      "forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]},
      "ocean": {)" +
           (ocean.empty() ? std::string{} : ocean + ", ") +
           R"("at": ")" + at + R"(", "topography": {"file": "history.nc"},
                "ice": {"file": "history.nc"}},
      "rotation": "full")";
  }

  fs::path dir_;
};

TEST_F(AppsOverRanks, SolveMatchesOneProcessAndResumesFromRankCheckpoints) {
  const auto one = Write("one.json", "{" + Body() + R"(,
      "output": {"surface": "out/one.nc"}})");
  const auto over = Write("over.json", "{" + Body() + R"(,
      "output": {"surface": "out/over.nc"}})");
  const auto pieces = Write("pieces.json", "{" + Body() + R"(,
      "checkpoint": {"directory": "ckpt", "every": 1,
                     "max_walltime_min": 1e-9},
      "output": {"surface": "out/pieces.nc"}})");
  OneProcess([&] { static_cast<void>(app::RunSolve(one)); });
  const auto summary = app::RunSolve(over);
  EXPECT_NE(summary.find("wrote"), std::string::npos) << summary;
  // stopped at every checkpoint by the wall-time limit, and resumed from
  // the node every rank's own subdirectory holds
  auto attempts = 0;
  for (; attempts < 8; ++attempts) {
    const auto part = app::RunSolve(pieces);
    if (part.find("wrote") != std::string::npos) break;
    EXPECT_NE(part.find("checkpointed"), std::string::npos) << part;
  }
  EXPECT_GE(attempts, 1);
  EXPECT_TRUE(
      fs::is_directory(dir_ / "ckpt" / std::format("rank_{:04}", Rank())));
  Barrier();
  if (Rank() == 0) {
    const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
    const auto distributed = pm::ReadSurfaceFile(dir_ / "out" / "over.nc");
    const auto resumed = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
    for (const auto* name : {"relative_sea_level_change", "displacement", "potential"}) {
      EXPECT_LT(Relative(distributed.variables.at(name).values,
                         reference.variables.at(name).values),
                kOverRanks)
          << name;
      // resumed on the same ranks, each rank's process the same transforms:
      // bitwise
      EXPECT_EQ(resumed.variables.at(name).values,
                distributed.variables.at(name).values)
          << name;
    }
  }
}

TEST_F(AppsOverRanks, CalibratedMigratingSolveMatchesOneProcess) {
  // Calibration iterates whole runs until the initial sea level settles:
  // every rank must decide alike when to stop (they see the same, rank 0's,
  // sea level), and the calibrated run, migrating shorelines and all,
  // matches the one-process one; stopped at every checkpoint, it resumes
  // without calibrating again, as uninterrupted.
  if (Rank() == 0) test::WriteToyHistory(dir_ / "marine.nc", 0.0, true);
  Barrier();
  auto body =
      Body(R"("shorelines": "migrating", "calibration": {"rtol": 1e-8})");
  for (const auto* from : {"\"file\": \"history.nc\""}) {
    for (auto at = body.find(from); at != std::string::npos;
         at = body.find(from, at + 1)) {
      body.replace(at, std::string_view{from}.size(),
                   "\"file\": \"marine.nc\"");
    }
  }
  const auto one = Write("one.json", "{" + body + R"(,
      "output": {"surface": "out/one.nc"}})");
  const auto over = Write("over.json", "{" + body + R"(,
      "output": {"surface": "out/over.nc"}})");
  const auto pieces = Write("pieces.json", "{" + body + R"(,
      "checkpoint": {"directory": "ckpt", "every": 1,
                     "max_walltime_min": 1e-9},
      "output": {"surface": "out/pieces.nc"}})");
  OneProcess([&] { static_cast<void>(app::RunSolve(one)); });
  const auto summary = app::RunSolve(over);
  EXPECT_NE(summary.find("initial sea level calibrated"), std::string::npos)
      << summary;
  for (auto attempts = 0; attempts < 8; ++attempts) {
    if (app::RunSolve(pieces).find("wrote") != std::string::npos) break;
  }
  Barrier();
  if (Rank() != 0) return;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
  const auto distributed = pm::ReadSurfaceFile(dir_ / "out" / "over.nc");
  const auto resumed = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  for (const auto* name :
       {"relative_sea_level_change", "initial_sea_level", "ocean_function"}) {
    EXPECT_LT(Relative(distributed.variables.at(name).values,
                       reference.variables.at(name).values),
              1e-8)
        << name;
    // resumed on the same ranks: bitwise
    EXPECT_EQ(resumed.variables.at(name).values,
              distributed.variables.at(name).values)
        << name;
  }
}

TEST_F(AppsOverRanks, MigratingSolveUnderATideMatchesOneProcess) {
  // A tide beside the marine history's ice under migrating shorelines:
  // the run over the ranks (each solving the nonlinear sea level equation,
  // rank 0's loads, potentials and ocean function shared) matches the
  // one-process run, its ocean functions to the bit; stopped at every
  // checkpoint, it resumes as uninterrupted.
  if (Rank() == 0) test::WriteToyHistory(dir_ / "marine.nc", 0.0, true);
  Barrier();
  auto body = Body();
  for (auto at = body.find("history.nc"); at != std::string::npos;
       at = body.find("history.nc", at + 1)) {
    body.replace(at, std::string_view{"history.nc"}.size(), "marine.nc");
  }
  body.replace(body.find(R"("at": "start")"), std::string_view{R"("at": "start")"}.size(),
               R"("shorelines": "migrating", "at": "start")");
  const auto term = std::string{R"([{"thickness": {"file": "marine.nc"}}])"};
  body.replace(body.find(term), term.size(),
               term + R"(, "tide": [{"degree": 2, "order": 1,
                   "amplitude_m2_s2": 60, "imaginary_m2_s2": -20,
                   "history_kyr_bp": [[2, 0], [0, 1]]}])");
  const auto one = Write("one.json", "{" + body + R"(,
      "output": {"surface": "out/one.nc"}})");
  const auto over = Write("over.json", "{" + body + R"(,
      "output": {"surface": "out/over.nc"}})");
  const auto pieces = Write("pieces.json", "{" + body + R"(,
      "checkpoint": {"directory": "ckpt", "every": 1,
                     "max_walltime_min": 1e-9},
      "output": {"surface": "out/pieces.nc"}})");
  OneProcess([&] { static_cast<void>(app::RunSolve(one)); });
  const auto summary = app::RunSolve(over);
  EXPECT_NE(summary.find("shorelines migrating"), std::string::npos) << summary;
  EXPECT_NE(summary.find("tide (1 coefficient)"), std::string::npos) << summary;
  for (auto attempts = 0; attempts < 8; ++attempts) {
    if (app::RunSolve(pieces).find("wrote") != std::string::npos) break;
  }
  Barrier();
  if (Rank() != 0) return;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
  const auto distributed = pm::ReadSurfaceFile(dir_ / "out" / "over.nc");
  const auto resumed = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  for (const auto* name :
       {"relative_sea_level_change", "displacement", "potential"}) {
    EXPECT_LT(Relative(distributed.variables.at(name).values,
                       reference.variables.at(name).values),
              kOverRanks)
        << name;
    // resumed on the same ranks: bitwise
    EXPECT_EQ(resumed.variables.at(name).values,
              distributed.variables.at(name).values)
        << name;
  }
  // flotation's 0/1: rounding changes it only at a cell on the threshold
  EXPECT_EQ(distributed.variables.at("ocean_function").values,
            reference.variables.at("ocean_function").values);
  EXPECT_EQ(resumed.variables.at("ocean_function").values,
            distributed.variables.at("ocean_function").values);
}

TEST_F(AppsOverRanks, KernelMatchesOneProcessAndSpillsPerRank) {
  // The continuous 3-D sweep over the ranks: its kernels the one process's
  // to the bound of another summation order, and from a store spilled per
  // rank bitwise those of the run in memory on the same ranks (the
  // discrete sweep's are KernelsOnTheRunsClockMatchOneProcess's and the
  // tests after it).
  const auto measurement = std::string{R"(
      "method": "continuous", "parameters": ["ice", "viscosity"],
      "measurements": [
        {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
         "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0}])"};
  const auto kernels = [&](const std::string& name,
                           const std::string& trajectory) {
    return Write(name + ".json", "{" + Body() + R"(, "trajectory": {)" +
                                     trajectory + R"(}, "kernels": {)" +
                                     measurement + R"(, "output": "out_)" +
                                     name + R"("}})");
  };
  const auto one = kernels("one", "");
  const auto over = kernels("over", "");
  const auto memory = kernels("memory", R"("record_stride": 2)");
  const auto spilled = kernels(
      "spilled", R"("record_stride": 2, "directory": "spill")");
  OneProcess([&] { static_cast<void>(app::RunKernel(one)); });
  static_cast<void>(app::RunKernel(over));
  static_cast<void>(app::RunKernel(memory));
  const auto summary = app::RunKernel(spilled);
  EXPECT_NE(summary.find("; the forward run recorded to the trajectory store "),
            std::string::npos)
      << summary;
  // each rank recorded into its own subdirectory, and the store stays
  const auto mine = dir_ / "spill" / std::format("rank_{:04}", Rank());
  EXPECT_TRUE(fs::exists(mine / "node_000000.state"));
  EXPECT_TRUE(fs::exists(mine / "node_000002.state"));
  EXPECT_FALSE(fs::exists(mine / "node_000001.state"));
  Barrier();
  if (Rank() != 0) return;

  auto ice = [&](const std::string& out) {
    return pm::ReadSurfaceFile(dir_ / out / "site_ice_kernels.nc");
  };
  auto viscosity = [&](const std::string& out) {
    return pm::ReadModelFile(dir_ / out / "site_viscosity_kernels.nc");
  };
  for (const auto* name : {"thickness_dual", "rate_dual"}) {
    EXPECT_LT(Relative(ice("out_over").variables.at(name).values,
                       ice("out_one").variables.at(name).values),
              kOverRanks)
        << name;
    // the store read back against the run in memory, on the same ranks:
    // bitwise
    EXPECT_EQ(ice("out_spilled").variables.at(name).values,
              ice("out_memory").variables.at(name).values)
        << name;
  }
  const auto reference = viscosity("out_one");
  const auto distributed = viscosity("out_over");
  const auto fromMemory = viscosity("out_memory");
  const auto fromDisk = viscosity("out_spilled");
  ASSERT_EQ(distributed.Parts().size(), reference.Parts().size());
  auto compared = 0;
  for (auto p = std::size_t{0}; p < reference.Parts().size(); ++p) {
    for (const auto& [name, var] : reference.Parts()[p].variables) {
      if (!name.starts_with("K_ln_eta")) continue;
      ++compared;
      EXPECT_LT(Relative(distributed.Parts()[p].variables.at(name).values,
                         var.values),
                kOverRanks)
          << name << " in part " << p;
      EXPECT_EQ(fromDisk.Parts()[p].variables.at(name).values,
                fromMemory.Parts()[p].variables.at(name).values)
          << name << " in part " << p;
    }
  }
  EXPECT_GT(compared, 0);
}

TEST_F(AppsOverRanks, KernelSweepsWhatSolveRecordedOverTheRanks) {
  // solve records the 3-D run over the ranks into the trajectory store,
  // each rank's states in its own subdirectory and rank 0's manifest
  // listing every rank's part; kernel, over the same ranks, sweeps from
  // it without the forward run, its kernels bitwise those of the run in
  // memory over the ranks, under frozen and migrating shorelines, by the
  // continuous and the discrete sweeps. On one process the store, another
  // count of ranks', is refused.
  const auto measurement = std::string{R"(
      "parameters": ["ice", "viscosity"],
      "measurements": [
        {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
         "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0}])"};
  if (Rank() == 0) test::WriteToyHistory(dir_ / "marine.nc", 0.0, true);
  Barrier();
  auto migrating = Body(R"("shorelines": "migrating")");
  for (auto at = migrating.find("history.nc"); at != std::string::npos;
       at = migrating.find("history.nc", at + 1)) {
    migrating.replace(at, std::string_view{"history.nc"}.size(), "marine.nc");
  }
  migrating.replace(migrating.find(R"("at": "end")"),
                    std::string_view{R"("at": "end")"}.size(),
                    R"("at": "start")");
  for (const auto& [label, body, method] :
       {std::tuple{std::string{"frozen"}, Body(), std::string{"continuous"}},
        std::tuple{std::string{"migrating"}, migrating,
                   std::string{"discrete"}}}) {
    const auto config = [&](const std::string& name,
                            const std::string& trajectory) {
      return Write(name + ".json",
                   "{" + body + R"(, "trajectory": {)" + trajectory +
                       R"(}, "output": {"surface": "out/)" + name +
                       R"(.nc"}, "kernels": {"method": ")" + method + R"(", )" +
                       measurement + R"(, "output": "out_)" + name + R"("}})");
    };
    const auto memory = config(label + "_memory", R"("record_stride": 2)");
    const auto stored = config(
        label + "_stored",
        R"("record_stride": 2, "directory": "store_)" + label + R"(")");
    static_cast<void>(app::RunKernel(memory));
    const auto solved = app::RunSolve(stored);
    EXPECT_NE(solved.find("; recorded the trajectory to the store "),
              std::string::npos)
        << solved;
    const auto store = dir_ / ("store_" + label);
    EXPECT_TRUE(fs::exists(store / std::format("rank_{:04}", Rank()) /
                           "node_000002.state"));
    const auto manifest = io::TrajectoryStore::ManifestOf(store);
    EXPECT_TRUE(manifest.complete);
    EXPECT_EQ(manifest.parts.size(), static_cast<std::size_t>(Size()));
    const auto swept = app::RunKernel(stored);
    EXPECT_NE(swept.find("; the forward run read back from the trajectory "
                         "store "),
              std::string::npos)
        << swept;
    // over another count of ranks: refused
    const auto ranks = Size();
    OneProcess([&] {
      EXPECT_THROW(
          {
            try {
              static_cast<void>(app::RunKernel(stored));
            } catch (const planetmodel::Error& error) {
              const auto message = std::string{error.what()};
              EXPECT_NE(message.find("trajectory.directory"), std::string::npos)
                  << message;
              EXPECT_NE(message.find(std::format(
                            "holds this run recorded over {} process(es), and "
                            "this run has 1",
                            ranks)),
                        std::string::npos)
                  << message;
              throw;
            }
          },
          planetmodel::Error);
    });
    if (Rank() == 0) {
      // solve's recorded forward run is the one kernel makes, over the
      // same ranks: bitwise
      const auto a = pm::ReadSurfaceFile(dir_ / ("out_" + label + "_stored") /
                                         "site_ice_kernels.nc");
      const auto b = pm::ReadSurfaceFile(dir_ / ("out_" + label + "_memory") /
                                         "site_ice_kernels.nc");
      for (const auto& [name, variable] : b.variables) {
        EXPECT_EQ(a.variables.at(name).values, variable.values)
            << label << " " << name;
      }
      const auto k = pm::ReadModelFile(dir_ / ("out_" + label + "_stored") /
                                       "site_viscosity_kernels.nc");
      const auto m = pm::ReadModelFile(dir_ / ("out_" + label + "_memory") /
                                       "site_viscosity_kernels.nc");
      ASSERT_EQ(k.Parts().size(), m.Parts().size());
      for (auto p = std::size_t{0}; p < m.Parts().size(); ++p) {
        for (const auto& [name, variable] : m.Parts()[p].variables) {
          EXPECT_EQ(k.Parts()[p].variables.at(name).values, variable.values)
              << label << " " << name << " in part " << p;
        }
      }
      auto values = [](const fs::path& path) {
        auto in = std::ifstream(path);
        auto out = std::vector<std::string>{};
        for (std::string line; std::getline(in, line);) {
          if (!line.starts_with('#')) out.push_back(line);
        }
        return out;
      };
      EXPECT_EQ(values(dir_ / ("out_" + label + "_stored") / "measurements.txt"),
                values(dir_ / ("out_" + label + "_memory") / "measurements.txt"))
          << label;
    }
    Barrier();
  }
}

TEST_F(AppsOverRanks, ResumedSolveContinuesItsStoreOverTheRanks) {
  // A 3-D run over the ranks recording into the trajectory store, stopped
  // by the wall-time limit after every node's checkpoint, continues its
  // store from each checkpoint every rank holds: the store it finishes,
  // every rank's states in its own subdirectory, is file by file and byte
  // by byte the uninterrupted run's over the same ranks, at a record
  // stride the resumptions' nodes do not all divide, under frozen and
  // migrating shorelines, and kernel sweeps from it the uninterrupted
  // store's kernels to the bit.
  const auto measurement = std::string{R"(
      "parameters": ["ice", "viscosity"],
      "measurements": [
        {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
         "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0}])"};
  auto migrating = Body(R"("shorelines": "migrating")");
  migrating.replace(migrating.find(R"("at": "end")"),
                    std::string_view{R"("at": "end")"}.size(),
                    R"("at": "start")");
  const auto bytes = [](const fs::path& root) {
    auto out = std::map<fs::path, std::string>{};
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
      if (!entry.is_regular_file()) continue;
      auto in = std::ifstream(entry.path(), std::ios::binary);
      out.emplace(fs::relative(entry.path(), root),
                  std::string(std::istreambuf_iterator<char>(in), {}));
    }
    return out;
  };
  for (const auto& [label, body] :
       {std::pair{std::string{"frozen"}, Body()},
        std::pair{std::string{"migrating"}, migrating}}) {
    const auto config = [&](const std::string& name,
                            const std::string& checkpoint) {
      return Write(name + ".json",
                   "{" + body + R"(, "trajectory": {"record_stride": 2,
                     "directory": "store_)" +
                       name + R"("})" + checkpoint +
                       R"(, "output": {"surface": "out/)" + name +
                       R"(.nc"}, "kernels": {)" + measurement +
                       R"(, "output": "out_)" + name + R"("}})");
    };
    const auto whole = config(label + "_whole", "");
    const auto parts = config(label + "_parts", R"(, "checkpoint": {
        "directory": "ckpt_)" + label + R"(", "every": 1,
        "max_walltime_min": 1e-9})");
    static_cast<void>(app::RunSolve(whole));
    auto attempts = 0;
    auto last = std::string{};
    for (; attempts < 8; ++attempts) {
      last = app::RunSolve(parts);
      if (last.find("wrote") != std::string::npos) break;
      EXPECT_NE(last.find(std::format("left incomplete at node {}", attempts)),
                std::string::npos)
          << last;
    }
    EXPECT_EQ(attempts, 3) << label;
    EXPECT_NE(last.find(", continuing it from node 2"), std::string::npos)
        << last;
    static_cast<void>(app::RunKernel(parts));
    static_cast<void>(app::RunKernel(whole));
    Barrier();
    if (Rank() == 0) {
      // resumed on the same ranks, each rank's process the same transforms:
      // byte for byte
      const auto got = bytes(dir_ / ("store_" + label + "_parts"));
      const auto want = bytes(dir_ / ("store_" + label + "_whole"));
      EXPECT_EQ(got.size(), want.size()) << label;
      for (const auto& [name, contents] : want) {
        const auto found = got.find(name);
        ASSERT_NE(found, got.end()) << label << " " << name;
        EXPECT_TRUE(found->second == contents) << label << " " << name;
      }
      const auto a = pm::ReadSurfaceFile(dir_ / ("out_" + label + "_parts") /
                                         "site_ice_kernels.nc");
      const auto b = pm::ReadSurfaceFile(dir_ / ("out_" + label + "_whole") /
                                         "site_ice_kernels.nc");
      for (const auto& [name, variable] : b.variables) {
        EXPECT_EQ(a.variables.at(name).values, variable.values)
            << label << " " << name;
      }
      const auto k = pm::ReadModelFile(dir_ / ("out_" + label + "_parts") /
                                       "site_viscosity_kernels.nc");
      const auto m = pm::ReadModelFile(dir_ / ("out_" + label + "_whole") /
                                       "site_viscosity_kernels.nc");
      ASSERT_EQ(k.Parts().size(), m.Parts().size());
      for (auto p = std::size_t{0}; p < m.Parts().size(); ++p) {
        for (const auto& [name, variable] : m.Parts()[p].variables) {
          EXPECT_EQ(k.Parts()[p].variables.at(name).values, variable.values)
              << label << " " << name << " in part " << p;
        }
      }
    }
    Barrier();
  }
}

TEST_F(AppsOverRanks, SolveUnderLateralModuliMatchesOneProcess) {
  // Under lateral moduli the rank cuts balance the transform work of the
  // elements the moduli perturb as well as the viscous ones: the run over
  // the ranks is the one-process run's to the substructured solves'
  // round-off, and so is the radial reference model the first rank steps
  // beside it.
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  if (Rank() == 0) {
    const auto deck = Deck::Read(data + "/prem8.elastic.deck",
                                 app::FormatOf(app::ModelFormat::Elastic));
    ASSERT_TRUE(deck);
    const auto model = EarthModel::FromDeck(*deck);
    ASSERT_TRUE(model);
    ASSERT_TRUE(
        io::WriteModuli(dir_ / "moduli.nc", *model, SurfaceGrid(8),
                        [](double, double theta, double phi) {
                          return 0.05 * std::sin(theta) * std::cos(phi);
                        },
                        {}, {.name = "toy moduli", .elementMetres = 500e3}));
  }
  Barrier();
  auto body = Body();
  const auto viscosity = std::string{"/viscosity_log.nc\"}"};
  body.replace(body.find(viscosity), viscosity.size(),
               viscosity + R"(, "moduli": {"file": "moduli.nc"})");
  const auto one = Write("one.json", "{" + body + R"(,
      "output": {"surface": "out/one.nc"}})");
  const auto over = Write("over.json", "{" + body + R"(,
      "output": {"surface": "out/over.nc"}})");
  OneProcess([&] { static_cast<void>(app::RunSolve(one)); });
  const auto summary = app::RunSolve(over);
  if (Rank() == 0) {
    EXPECT_TRUE(summary.starts_with(
        "3-D (lateral viscosity, lateral moduli), viscoelastic"))
        << summary;
  }
  Barrier();
  if (Rank() != 0) return;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
  const auto distributed = pm::ReadSurfaceFile(dir_ / "out" / "over.nc");
  for (const auto* name : {"relative_sea_level_change", "displacement",
                           "potential", "displacement_radial_model"}) {
    EXPECT_LT(Relative(distributed.variables.at(name).values,
                       reference.variables.at(name).values),
              kOverRanks)
        << name;
  }
}

TEST_F(AppsOverRanks, SolveFingerprintUnderLateralModuliMatchesOneProcess) {
  // The elastic fingerprint on lateral moduli (the static problem) over the
  // ranks, its zero-length step closed by the fixed point and by the
  // nested conjugate gradients, the cuts balancing the elements the moduli
  // make active: the one-process fingerprint to the substructured solves'
  // round-off, the radial model's fields beside it the same. A fingerprint
  // on a radial Earth is the first rank's alone, the one process's to the
  // bit.
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  if (Rank() == 0) {
    const auto deck = Deck::Read(data + "/prem8.elastic.deck",
                                 app::FormatOf(app::ModelFormat::Elastic));
    ASSERT_TRUE(deck);
    const auto model = EarthModel::FromDeck(*deck);
    ASSERT_TRUE(model);
    ASSERT_TRUE(io::WriteModuli(
        dir_ / "moduli.nc", *model, SurfaceGrid(8),
        [](double, double theta, double phi) {
          return 0.1 * std::sin(theta) * std::cos(phi);
        },
        [](double, double theta, double) { return 0.05 * std::cos(theta); },
        {.name = "toy moduli", .elementMetres = 500e3}));
  }
  Barrier();
  const auto config = [&](const std::string& name, const std::string& moduli,
                          const std::string& solver) {
    return Write(name + ".json", std::format(R"({{
      "problem": "static",
      "model": {{"deck": "{}/prem8.elastic.deck", "format": "elastic"{}}},
      "discretisation": {{"lmax": 8}},
      "forcing": {{"ice_change": [{{"cap_thickness": {{"latitude": 70,
                                   "longitude": 30, "width_deg": 10,
                                   "thickness_m": 50}}}}]}},
      "ocean": {{"date_kyr_bp": 0, "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full",{}
      "output": {{"surface": "out/{}.nc"}}}})",
                                             data, moduli, solver, name));
  };
  const auto lateral = std::string{R"(, "moduli": {"file": "moduli.nc"})"};
  for (const auto* step : {"fixed_point", "conjugate_gradients"}) {
    const auto solver = std::format(R"( "solver": {{"step": "{}"}},)", step);
    const auto one = config(std::format("one_{}", step), lateral, solver);
    const auto over = config(std::format("over_{}", step), lateral, solver);
    OneProcess([&] { static_cast<void>(app::RunSolve(one)); });
    const auto summary = app::RunSolve(over);
    if (Rank() == 0) {
      EXPECT_TRUE(
          summary.starts_with("3-D (lateral moduli), elastic, static, "
                              "shorelines frozen, rotation full; ice "))
          << summary;
    } else {
      EXPECT_EQ(summary, "");
    }
    Barrier();
    if (Rank() != 0) continue;
    const auto reference =
        pm::ReadSurfaceFile(dir_ / "out" / std::format("one_{}.nc", step));
    const auto distributed =
        pm::ReadSurfaceFile(dir_ / "out" / std::format("over_{}.nc", step));
    for (const auto* name :
         {"relative_sea_level_change", "displacement", "potential",
          "relative_sea_level_change_radial_model", "displacement_radial_model",
          "potential_radial_model"}) {
      EXPECT_LT(Relative(distributed.variables.at(name).values,
                         reference.variables.at(name).values),
                kOverRanks)
          << step << " " << name;
    }
  }

  const auto radialOne = config("radial_one", "", "");
  const auto radialOver = config("radial_over", "", "");
  OneProcess([&] { static_cast<void>(app::RunSolve(radialOne)); });
  const auto summary = app::RunSolve(radialOver);
  if (Rank() == 0) {
    EXPECT_TRUE(summary.starts_with("1-D, elastic, static, shorelines frozen"))
        << summary;
  } else {
    EXPECT_EQ(summary, "");
  }
  Barrier();
  if (Rank() != 0) return;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "radial_one.nc");
  const auto first = pm::ReadSurfaceFile(dir_ / "out" / "radial_over.nc");
  // both the first rank's, one process: bitwise
  for (const auto* name : {"relative_sea_level_change", "displacement", "potential"}) {
    EXPECT_EQ(first.variables.at(name).values,
              reference.variables.at(name).values)
        << name;
  }
}

TEST_F(AppsOverRanks, SolveWithoutAnOceanUnderLateralModuliMatchesOneProcess) {
  // The static problem without an ocean on lateral moduli, under a load
  // that is both in the water budget and a layer on the solid surface and
  // a point source, its zero-length step distributed over the ranks: the
  // one-process answer to the substructured solves' round-off, the
  // toroidal displacement and the radial model's fields beside it
  // included.
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  if (Rank() == 0) {
    const auto deck = Deck::Read(data + "/prem8.elastic.deck",
                                 app::FormatOf(app::ModelFormat::Elastic));
    ASSERT_TRUE(deck);
    const auto model = EarthModel::FromDeck(*deck);
    ASSERT_TRUE(model);
    ASSERT_TRUE(io::WriteModuli(
        dir_ / "moduli.nc", *model, SurfaceGrid(8),
        [](double, double theta, double phi) {
          return 0.1 * std::sin(theta) * std::cos(phi);
        },
        [](double, double theta, double) { return 0.05 * std::cos(theta); },
        {.name = "toy moduli", .elementMetres = 500e3}));
  }
  Barrier();
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", std::format(R"({{
      "problem": "static",
      "model": {{"deck": "{}/prem8.elastic.deck", "format": "elastic",
                 "moduli": {{"file": "moduli.nc"}}}},
      "discretisation": {{"lmax": 8}},
      "forcing": {{
        "load": [{{"cap": {{"latitude": 60, "longitude": -85, "width_deg": 20,
                            "amplitude_kg_m2": 2000}},
                   "water_budget": true, "moves_solid_surface": true,
                   "density_kg_m3": 2300}}],
        "seismic": [{{"point": {{"latitude": 30, "longitude": 40,
                      "depth_km": 300, "strike_deg": 20, "dip_deg": 60,
                      "rake_deg": 90, "moment_nm": 1e21}}}}]}},
      "rotation": "full",
      "output": {{"surface": "out/{}.nc"}}}})",
                                             data, name));
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunSolve(one)); });
  const auto summary = app::RunSolve(over);
  if (Rank() == 0) {
    EXPECT_TRUE(
        summary.starts_with("3-D (lateral moduli), elastic, static, no ocean, "
                            "rotation full; load (1 "
                            "term) + seismic (1 source); wrote "))
        << summary;
  } else {
    EXPECT_EQ(summary, "");
  }
  Barrier();
  if (Rank() != 0) return;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
  const auto distributed = pm::ReadSurfaceFile(dir_ / "out" / "over.nc");
  for (const auto* name :
       {"relative_sea_level_change", "displacement", "tangential_potential",
        "toroidal_potential", "potential",
        "relative_sea_level_change_radial_model", "displacement_radial_model",
        "potential_radial_model"}) {
    EXPECT_LT(Relative(distributed.variables.at(name).values,
                       reference.variables.at(name).values),
              kOverRanks)
        << name;
  }
}

TEST_F(AppsOverRanks, SolveUnderAnOnsetBetweenNodesMatchesOneProcess) {
  // A 3-D run through time on the toy history under a point source whose
  // onset (1 kyr BP) falls between two nodes of the window (4/3 and 2/3 kyr
  // BP): the pair of nodes the program steps there is stepped over the
  // ranks, the window's nodes alone are written, and the fields, the
  // radial model's beside them, are the one-process run's to the solves'
  // round-off.
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", std::format(R"({{
      "problem": "quasi_static",
      "model": {{"deck": "{0}/prem8.elastic.deck", "format": "elastic",
                 "viscosity": {{"file": "{0}/viscosity_log.nc"}}}},
      "discretisation": {{"lmax": 8}},
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4}},
      "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
        "seismic": [{{"point": {{"latitude": 30, "longitude": 40,
                      "depth_km": 300, "strike_deg": 20, "dip_deg": 60,
                      "rake_deg": 90, "moment_nm": 1e21}},
                     "onset_kyr_bp": 1}}]}},
      "ocean": {{"at": "start", "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full",
      "output": {{"surface": "out/{1}.nc"}}}})",
                                             data, name));
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunSolve(one)); });
  const auto summary = app::RunSolve(over);
  if (Rank() == 0) {
    EXPECT_TRUE(summary.starts_with(
        "3-D (lateral viscosity), viscoelastic, 4 nodes 2 to 0 kyr BP"))
        << summary;
  }
  Barrier();
  if (Rank() != 0) return;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
  const auto distributed = pm::ReadSurfaceFile(dir_ / "out" / "over.nc");
  ASSERT_EQ(distributed.time->size(), 4u);
  for (const auto* name :
       {"relative_sea_level_change", "displacement", "tangential_potential",
        "toroidal_potential", "potential",
        "relative_sea_level_change_radial_model", "displacement_radial_model",
        "potential_radial_model"}) {
    EXPECT_LT(Relative(distributed.variables.at(name).values,
                       reference.variables.at(name).values),
              kOverRanks)
        << name;
  }
}

TEST_F(AppsOverRanks, StaticVolumeFileOverTheRanksIsTheOneProcessFile) {
  // The static problem on lateral moduli writes its volume file over the
  // ranks: each rank's response inside the body at the reference's nodes
  // in its own segment (the centre's element left out on the first),
  // gathered to the first rank. The one-process file to the distributed
  // elastic solves' tolerance, the lateral moduli sampled beside the
  // response to the bit (the nodes laid out as the one process lays
  // them).
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  if (Rank() == 0) {
    const auto deck = Deck::Read(data + "/prem8.elastic.deck",
                                 app::FormatOf(app::ModelFormat::Elastic));
    ASSERT_TRUE(deck);
    const auto model = EarthModel::FromDeck(*deck);
    ASSERT_TRUE(model);
    ASSERT_TRUE(io::WriteModuli(
        dir_ / "moduli.nc", *model, SurfaceGrid(8),
        [](double, double theta, double phi) {
          return 0.1 * std::sin(theta) * std::cos(phi);
        },
        [](double, double theta, double) { return 0.05 * std::cos(theta); },
        {.name = "toy moduli", .elementMetres = 500e3}));
  }
  Barrier();
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", std::format(R"({{
      "problem": "static",
      "model": {{"deck": "{}/prem8.elastic.deck", "format": "elastic",
                 "moduli": {{"file": "moduli.nc"}}}},
      "discretisation": {{"lmax": 8}},
      "forcing": {{
        "load": [{{"cap": {{"latitude": 60, "longitude": -85, "width_deg": 20,
                            "amplitude_kg_m2": 2000}}}}],
        "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0}}]}},
      "rotation": "full",
      "solver": {{"step_rtol": 1e-12, "step_max_iterations": 200}},
      "output": {{"surface": "out/{}.nc", "volume": "out/{}_volume.nc"}}}})",
                                             data, name, name));
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunSolve(one)); });
  const auto summary = app::RunSolve(over);
  if (Rank() == 0) {
    EXPECT_NE(summary.find("out/over_volume.nc"), std::string::npos)
        << summary;
  } else {
    EXPECT_EQ(summary, "");
  }
  Barrier();
  if (Rank() != 0) return;
  const auto reference = pm::ReadModelFile(dir_ / "out" / "one_volume.nc");
  const auto distributed = pm::ReadModelFile(dir_ / "out" / "over_volume.nc");
  ASSERT_EQ(distributed.Parts().size(), reference.Parts().size());
  auto compared = std::vector<std::string>{};
  auto worst = 0.0;
  for (auto p = std::size_t{0}; p < reference.Parts().size(); ++p) {
    for (const auto& [name, variable] : reference.Parts()[p].variables) {
      const auto& got = distributed.Parts()[p].variables.at(name).values;
      ASSERT_EQ(got.size(), variable.values.size()) << name << " in part " << p;
      if (name == "mu_relative" || name == "kappa_relative") {
        // sampled by planetmodel's interpolant, no transform: bitwise on
        // any rank
        EXPECT_EQ(got, variable.values) << name << " in part " << p;
      } else if (name == "radial_displacement" ||
                 name == "deviatoric_strain") {
        const auto departure = Relative(got, variable.values);
        worst = std::max(worst, departure);
        EXPECT_LT(departure, kOverRanks) << name << " in part " << p;
      } else {
        continue;
      }
      compared.push_back(name);
    }
  }
  RecordProperty("volume_departure", std::format("{:.3e}", worst));
  std::printf("[volume file over %d ranks] response %.3e from one process's\n",
              Size(), worst);
  for (const auto* name : {"radial_displacement", "deviatoric_strain",
                           "mu_relative", "kappa_relative"}) {
    EXPECT_TRUE(std::ranges::find(compared, name) != compared.end()) << name;
  }
}

TEST_F(AppsOverRanks, RadialSolveIsTheFirstRanksAlone) {
  // A 1-D run of solve is one process's: over the ranks the first runs it
  // alone, as a run of one rank (its checkpoints in the directory itself,
  // not a rank's subdirectory), and the others return at once. The output
  // is the one-process run's to the bit.
  if (Rank() == 0) {
    std::ofstream(dir_ / "ball.deck")
        << "homogeneous test ball\n"
           "gia deck for the sl3d tests\n"
           "0 5500 10000 5500 10000 5500 1 1e21\n"
           "6371000 5500 10000 5500 10000 5500 1 1e21\n";
  }
  Barrier();
  const auto body = std::string{R"(
      "problem": "quasi_static",
      "model": {"deck": "ball.deck"}, "discretisation": {"lmax": 8},
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5},
      "forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]},
      "ocean": {"at": "start", "topography": {"file": "history.nc"},
                "ice": {"file": "history.nc"}},
      "rotation": "full", "moments": "hydrostatic")"};
  const auto one = Write("one.json", "{" + body + R"(,
      "output": {"surface": "out/one.nc"}})");
  const auto over = Write("over.json", "{" + body + R"(,
      "checkpoint": {"directory": "ckpt", "every": 2},
      "output": {"surface": "out/over.nc"}})");
  OneProcess([&] { static_cast<void>(app::RunSolve(one)); });
  const auto summary = app::RunSolve(over);
  if (Rank() == 0) {
    EXPECT_TRUE(summary.starts_with("1-D, viscoelastic")) << summary;
  } else {
    EXPECT_EQ(summary, "");
  }
  Barrier();
  if (Rank() != 0) return;
  EXPECT_TRUE(fs::exists(dir_ / "ckpt" / "node_000004.ckpt"));
  EXPECT_FALSE(fs::exists(dir_ / "ckpt" / "rank_0000"));
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
  const auto first = pm::ReadSurfaceFile(dir_ / "out" / "over.nc");
  // both the first rank's, one process: bitwise
  for (const auto* name : {"relative_sea_level_change", "displacement", "potential"}) {
    EXPECT_EQ(first.variables.at(name).values,
              reference.variables.at(name).values)
        << name;
  }
}

TEST_F(AppsOverRanks, ViscoelasticResponseMatchesOneProcess) {
  // The viscoelastic response without an ocean under a lateral viscosity
  // and lateral moduli, each step closed by the fixed point and by the
  // closure nested over conjugate-gradient solves: over the ranks (the
  // first node's zero-length step and every step distributed over radial
  // segments) as on one process, to the radially substructured solves'
  // round-off and the solvers' tolerance (solver.step_rtol 1e-10); the
  // radial reference model beside it is the first rank's alone and the
  // same.
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  if (Rank() == 0) {
    const auto deck = Deck::Read(data + "/prem8.elastic.deck",
                                 app::FormatOf(app::ModelFormat::Elastic));
    ASSERT_TRUE(deck);
    const auto model = EarthModel::FromDeck(*deck);
    ASSERT_TRUE(model);
    ASSERT_TRUE(
        io::WriteModuli(dir_ / "moduli.nc", *model, SurfaceGrid(8),
                        [](double, double theta, double phi) {
                          return 0.05 * std::sin(theta) * std::cos(phi);
                        },
                        {}, {.name = "toy moduli", .elementMetres = 500e3}));
  }
  Barrier();
  const auto body = std::string{R"(
      "problem": "quasi_static",
      "model": {"deck": ")"} +
                    data +
                    R"(/prem8.elastic.deck", "format": "elastic",
                "viscosity": {"file": ")" +
                    data + R"(/viscosity_log.nc"},
                "moduli": {"file": "moduli.nc"}},
      "discretisation": {"lmax": 8},
      "times": {"unit": "kyr", "end": 1, "nodes": 3},
      "forcing": {
        "load": [{"cap": {"latitude": 60, "longitude": -85, "width_deg": 20,
                          "amplitude_kg_m2": 1.0e6},
                  "history": [[0, 0.5], [1, 1]]}],
        "tide": [{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0}]},
      "rotation": "none")";
  for (const auto* step : {"fixed_point", "conjugate_gradients"}) {
    const auto config = [&](const std::string& name) {
      return Write(name + ".json",
                   std::format(R"({{{}, "solver": {{"step": "{}",
                                     "step_rtol": 1e-10,
                                     "step_max_iterations": 200}},
                                   "output": {{"surface": "out/{}.nc"}}}})",
                               body, step, name));
    };
    const auto one = config(std::format("one_{}", step));
    const auto over = config(std::format("over_{}", step));
    OneProcess([&] { static_cast<void>(app::RunSolve(one)); });
    // every rank returns a run through time's summary
    const auto summary = app::RunSolve(over);
    EXPECT_TRUE(summary.starts_with(
        "3-D (lateral viscosity, lateral moduli), viscoelastic"))
        << summary;
    Barrier();
    if (Rank() != 0) continue;
    const auto reference =
        pm::ReadSurfaceFile(dir_ / "out" / std::format("one_{}.nc", step));
    const auto distributed =
        pm::ReadSurfaceFile(dir_ / "out" / std::format("over_{}.nc", step));
    for (const auto* name :
         {"load", "displacement", "tangential_potential", "potential",
          "displacement_radial_model", "potential_radial_model"}) {
      EXPECT_LT(Relative(distributed.variables.at(name).values,
                         reference.variables.at(name).values),
                kOverRanks)
          << step << " " << name;
    }
  }
}

// -- the kernels of the problems lifted to 3-D, over the ranks ----------------

/// The test's prem8 model with a lateral shear and bulk modulus, written by
/// rank 0 to `moduli.nc` in the test's directory.
void WriteToyModuli(const fs::path& dir) {
  if (Rank() == 0) {
    const auto data = std::string{SL3DPP_TEST_DATA_DIR};
    const auto deck = Deck::Read(data + "/prem8.elastic.deck",
                                 app::FormatOf(app::ModelFormat::Elastic));
    ASSERT_TRUE(deck);
    const auto model = EarthModel::FromDeck(*deck);
    ASSERT_TRUE(model);
    ASSERT_TRUE(io::WriteModuli(
        dir / "moduli.nc", *model, SurfaceGrid(8),
        [](double, double theta, double phi) {
          return 0.1 * std::sin(theta) * std::cos(phi);
        },
        [](double, double theta, double) { return 0.05 * std::cos(theta); },
        {.name = "toy moduli", .elementMetres = 500e3}));
  }
  Barrier();
}

/// The largest relative departure of two kernel model files' kernel
/// fields (each `K_` field over every part), and their pairings in the
/// metadata.
double KernelFileDeparture(const fs::path& mine, const fs::path& theirs) {
  const auto a = pm::ReadModelFile(mine);
  const auto b = pm::ReadModelFile(theirs);
  EXPECT_EQ(a.Parts().size(), b.Parts().size());
  auto worst = 0.0;
  auto compared = 0;
  for (auto p = std::size_t{0}; p < b.Parts().size(); ++p) {
    for (const auto& [name, var] : b.Parts()[p].variables) {
      if (!name.starts_with("K_")) continue;
      ++compared;
      worst = std::max(worst, Relative(a.Parts()[p].variables.at(name).values,
                                       var.values));
    }
  }
  EXPECT_GT(compared, 0) << theirs;
  auto pairings = 0;
  for (const auto& [key, value] : b.Meta()) {
    const auto* want = std::get_if<std::vector<double>>(&value);
    if (want == nullptr || !key.ends_with("_pairing")) continue;
    ++pairings;
    worst = std::max(worst, Relative(std::get<std::vector<double>>(
                                         a.Meta().at(key)),
                                     *want));
  }
  EXPECT_GE(pairings, 1) << theirs;
  return worst;
}

/// The largest relative departure of the numbers of two kernel tables.
double TableDeparture(const fs::path& mine, const fs::path& theirs) {
  const auto numbers = [](const fs::path& path) {
    auto in = std::ifstream(path);
    auto out = std::vector<double>{};
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || line.starts_with("#")) continue;
      auto words = std::istringstream(line);
      for (double v; words >> v;) out.push_back(v);
    }
    return out;
  };
  const auto a = numbers(mine);
  const auto b = numbers(theirs);
  EXPECT_EQ(a.size(), b.size()) << theirs;
  EXPECT_FALSE(b.empty()) << theirs;
  return a.size() == b.size() ? Relative(a, b) : 1.0;
}

/// The largest relative departure of the fields two kernel surface files
/// share.
double SurfaceKernelDeparture(const fs::path& mine, const fs::path& theirs) {
  const auto a = pm::ReadSurfaceFile(mine);
  const auto b = pm::ReadSurfaceFile(theirs);
  auto worst = 0.0;
  for (const auto& [name, var] : b.variables) {
    if (std::ranges::all_of(var.values, [](double v) { return v == 0.0; })) {
      continue;
    }
    worst = std::max(worst,
                     Relative(a.variables.at(name).values, var.values));
  }
  return worst;
}

/// The measured values of two kernel runs (each row's third word on), the
/// largest relative departure.
double ValuesDeparture(const fs::path& mine, const fs::path& theirs) {
  const auto values = [](const fs::path& directory) {
    auto in = std::ifstream(directory / "measurements.txt");
    auto out = std::vector<double>{};
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || line.starts_with("#")) continue;
      auto words = std::istringstream(line);
      auto name = std::string{}, type = std::string{};
      words >> name >> type;
      for (double v; words >> v;) out.push_back(v);
    }
    return out;
  };
  const auto a = values(mine);
  const auto b = values(theirs);
  EXPECT_EQ(a.size(), b.size()) << theirs;
  EXPECT_FALSE(b.empty()) << theirs;
  return a.size() == b.size() ? Relative(a, b) : 1.0;
}

TEST_F(AppsOverRanks, TotalKernelsOfACalibratedRunMatchOneProcess) {
  // The kernels of a calibrated migrating 3-D run are by default its total
  // derivatives: beside each measurement's sweep the multiplier of the
  // calibration's constraint is found by sweeps of its own, collective
  // over the ranks, every rank deciding alike when they have converged.
  // The ice kernels, the multiplier (`target_sea_level_dual`, not zero:
  // the toy's marine ice floods) and the log-viscosity kernels over the
  // ranks are the one process's.
  if (Rank() == 0) test::WriteToyHistory(dir_ / "marine.nc", 0.0, true);
  Barrier();
  auto body =
      Body(R"("shorelines": "migrating", "calibration": {"rtol": 1e-10})");
  for (auto at = body.find("history.nc"); at != std::string::npos;
       at = body.find("history.nc", at + 1)) {
    body.replace(at, std::string_view{"history.nc"}.size(), "marine.nc");
  }
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", "{" + body + R"(,
      "solver": {"step_rtol": 1e-10, "step_max_iterations": 200},
      "kernels": {"parameters": ["ice", "viscosity"],
        "measurements": [
          {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
           "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0}],
        "output": "out_)" + name + R"("}})");
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunKernel(one)); });
  const auto summary = app::RunKernel(over);
  Barrier();
  if (Rank() != 0) return;
  EXPECT_NE(summary.find("total derivatives, the initial sea level calibrated "
                         "again, the multiplier in at most "),
            std::string::npos)
      << summary;
  const auto ice = pm::ReadSurfaceFile(dir_ / "out_over" / "site_ice_kernels.nc");
  const auto reference =
      pm::ReadSurfaceFile(dir_ / "out_one" / "site_ice_kernels.nc");
  EXPECT_EQ(std::get<std::string>(ice.meta.at("calibration_derivative")),
            "total");
  EXPECT_EQ(std::get<std::vector<double>>(
                ice.meta.at("calibration_multiplier_sweeps")),
            std::get<std::vector<double>>(
                reference.meta.at("calibration_multiplier_sweeps")));
  const auto& multiplier = ice.variables.at("target_sea_level_dual").values;
  EXPECT_TRUE(
      std::ranges::any_of(multiplier, [](double v) { return v != 0.0; }));
  const auto surface =
      SurfaceKernelDeparture(dir_ / "out_over" / "site_ice_kernels.nc",
                             dir_ / "out_one" / "site_ice_kernels.nc");
  const auto viscosity =
      KernelFileDeparture(dir_ / "out_over" / "site_viscosity_kernels.nc",
                          dir_ / "out_one" / "site_viscosity_kernels.nc");
  std::cout << std::format(
      "[calibrated, total, over the ranks] ice kernels and multiplier {:.1e}, "
      "viscosity {:.1e}\n",
      surface, viscosity);
  EXPECT_LT(ValuesDeparture(dir_ / "out_over", dir_ / "out_one"), 1e-8);
  EXPECT_LT(surface, 1e-7);
  EXPECT_LT(viscosity, 1e-7);
}

TEST_F(AppsOverRanks, StaticKernelsOnLateralModuliMatchOneProcess) {
  // The kernels of the static problem without an ocean on lateral moduli,
  // in the load and in the moduli (the 3-D kernels gathered to the first
  // rank and written over the whole mesh, with their pairing against the
  // run's lateral moduli): over the ranks as on one process, to the
  // substructured solves' round-off and the closures' tolerances.
  WriteToyModuli(dir_);
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", std::format(R"({{
      "problem": "static",
      "model": {{"deck": "{}/prem8.elastic.deck", "format": "elastic",
                 "moduli": {{"file": "moduli.nc"}}}},
      "discretisation": {{"lmax": 8}},
      "forcing": {{"load": [{{"cap": {{"latitude": 60, "longitude": -85,
          "width_deg": 20, "amplitude_kg_m2": 2000}}}}]}},
      "rotation": "full",
      "solver": {{"rtol": 1e-12, "step_rtol": 1e-12,
                 "step_max_iterations": 200}},
      "kernels": {{"method": "discrete", "parameters": ["load", "moduli"],
        "measurements": [
          {{"type": "displacement_cap", "name": "uplift", "latitude": 55,
            "longitude": -80, "width_deg": 10}},
          {{"type": "displacement_vector_point", "name": "drift",
            "latitude": 40, "longitude": -60,
            "direction": {{"east": 1, "north": 1}}}}],
        "output": "out_{}"}}}})",
                                             data, name));
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunKernel(one)); });
  const auto summary = app::RunKernel(over);
  if (Rank() == 0) {
    EXPECT_TRUE(
        summary.starts_with("3-D (lateral moduli), elastic, static, no ocean"))
        << summary;
  }
  Barrier();
  if (Rank() != 0) return;
  EXPECT_LT(ValuesDeparture(dir_ / "out_over", dir_ / "out_one"), 1e-10);
  for (const auto* name : {"uplift", "drift"}) {
    const auto stem = std::string{name};
    EXPECT_LT(SurfaceKernelDeparture(dir_ / "out_over" / (stem + "_load_kernels.nc"),
                                     dir_ / "out_one" / (stem + "_load_kernels.nc")),
              1e-9)
        << name;
    // measured 1.8e-10
    EXPECT_LT(KernelFileDeparture(dir_ / "out_over" / (stem + "_moduli_kernels.nc"),
                                  dir_ / "out_one" / (stem + "_moduli_kernels.nc")),
              1e-7)
        << name;
  }
}

TEST_F(AppsOverRanks, KernelsOnTheRunsClockMatchOneProcess) {
  // The kernels of a 3-D run without an ocean on its own clock at uneven
  // nodes, under a lateral viscosity and lateral moduli, a load ramped up
  // and a tide: in the load, the tide (by the discrete sweep), the
  // viscosity and the moduli, of a vector displacement at a time of the
  // clock and of a potential at the last node; over the ranks as on one
  // process.
  WriteToyModuli(dir_);
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", std::format(R"({{
      "problem": "quasi_static",
      "model": {{"deck": "{}/prem8.elastic.deck", "format": "elastic",
                 "viscosity": {{"file": "{}/viscosity_log.nc"}},
                 "moduli": {{"file": "moduli.nc"}}}},
      "discretisation": {{"lmax": 8}},
      "times": {{"unit": "kyr", "at": [0, 0.4, 1, 1.25]}},
      "forcing": {{
        "load": [{{"cap": {{"latitude": 60, "longitude": -85, "width_deg": 20,
                            "amplitude_kg_m2": 1.0e6}},
                  "history": [[0, 0.5], [1, 1]]}}],
        "tide": [{{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0}}]}},
      "rotation": "none",
      "solver": {{"step_rtol": 1e-10, "step_max_iterations": 200}},
      "kernels": {{"method": "discrete",
        "parameters": ["load", "tide", "viscosity", "moduli"],
        "measurements": [
          {{"type": "displacement_vector_point", "name": "drift",
            "latitude": 50, "longitude": -70,
            "direction": {{"east": 1, "up": 0.5}}, "time": 1}},
          {{"type": "potential_point", "name": "geoid", "latitude": 60,
            "longitude": -80}}],
        "output": "out_{}"}}}})",
                                             data, data, name));
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunKernel(one)); });
  static_cast<void>(app::RunKernel(over));
  Barrier();
  if (Rank() != 0) return;
  EXPECT_LT(ValuesDeparture(dir_ / "out_over", dir_ / "out_one"), 1e-9);
  for (const auto* name : {"drift", "geoid"}) {
    const auto stem = std::string{name};
    const auto at = [&](const std::string& run, const std::string& suffix) {
      return dir_ / ("out_" + run) / (stem + suffix);
    };
    EXPECT_LT(SurfaceKernelDeparture(at("over", "_load_kernels.nc"),
                                     at("one", "_load_kernels.nc")),
              1e-8)
        << name;
    EXPECT_LT(TableDeparture(at("over", "_tide_kernels.txt"),
                             at("one", "_tide_kernels.txt")),
              1e-8)
        << name;
    for (const auto* suffix :
         {"_viscosity_kernels.nc", "_moduli_kernels.nc"}) {
      EXPECT_LT(KernelFileDeparture(at("over", suffix), at("one", suffix)),
                1e-7)
          << name << suffix;
    }
  }
}

TEST_F(AppsOverRanks, KernelsOfAChangeAtTheStartOverListedDatesMatchOneProcess) {
  // The kernels of a 3-D run with an ocean over listed dates, frozen
  // shorelines, under the ice history and a cap of ice present at the
  // window's start, and a layer of sediment outside the water budget: in
  // the ice (the state held) and the loads; over the ranks as on one
  // process.
  const auto config = [&](const std::string& name) {
    auto body = Body();
    for (const auto& [from, to] :
         {std::pair{std::string{R"("times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4})"},
                    std::string{R"("times": {"at_kyr_bp": [2, 1.4, 0.5, 0]})"}},
          std::pair{std::string{R"("forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]})"},
                    std::string{R"("forcing": {
        "ice_change": [{"thickness": {"file": "history.nc"}},
                       {"cap_thickness": {"latitude": 50, "longitude": 0,
                         "width_deg": 8, "thickness_m": 200}}],
        "load": [{"cap": {"latitude": 60, "longitude": -85, "width_deg": 20,
                          "amplitude_kg_m2": 5e4},
                  "water_budget": false, "moves_solid_surface": true,
                  "density_kg_m3": 2000}]})"}}}) {
      const auto at = body.find(from);
      EXPECT_NE(at, std::string::npos) << from;
      if (at != std::string::npos) body.replace(at, from.size(), to);
    }
    return Write(name + ".json", "{" + body + std::format(R"(,
      "kernels": {{"method": "discrete", "parameters": ["ice", "load"],
        "measurements": [
          {{"type": "relative_sea_level_cap", "name": "site",
            "latitude": -30, "longitude": 10, "width_deg": 25}}],
        "output": "out_{}"}}}})",
                                                          name));
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunKernel(one)); });
  static_cast<void>(app::RunKernel(over));
  Barrier();
  if (Rank() != 0) return;
  EXPECT_LT(ValuesDeparture(dir_ / "out_over", dir_ / "out_one"), 1e-9);
  for (const auto* suffix : {"_ice_kernels.nc", "_load_kernels.nc"}) {
    EXPECT_LT(SurfaceKernelDeparture(dir_ / "out_over" / ("site" + std::string{suffix}),
                                     dir_ / "out_one" / ("site" + std::string{suffix})),
              1e-8)
        << suffix;
  }
}

TEST_F(AppsOverRanks, KernelRefusesOnRanksWhatOneProcessForms) {
  // On several ranks a 3-D kernel run refuses what the sweeps form on one
  // process: the kernels about a run under seismic sources (by either
  // sweep), and the tide kernels of the continuous sweep, the refusal
  // naming the discrete sweep's, which are distributed.
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  // the kernels block's members before its parameters: the method, or
  // nothing for the default
  const auto config = [&](const std::string& forcing,
                          const std::string& parameters,
                          const std::string& method) {
    return Write("bad.json", std::format(R"({{
      "problem": "quasi_static",
      "model": {{"deck": "{}/prem8.elastic.deck", "format": "elastic",
                 "viscosity": {{"file": "{}/viscosity_log.nc"}}}},
      "discretisation": {{"lmax": 8}},
      "times": {{"unit": "kyr", "end": 1, "nodes": 3}},
      "forcing": {}, "rotation": "none",
      "kernels": {{{}"parameters": {},
        "measurements": [{{"type": "potential_point", "latitude": 60,
                           "longitude": -80}}],
        "output": "k"}}}})",
                                         data, data, forcing, method,
                                         parameters));
  };
  constexpr auto kContinuous = R"("method": "continuous", )";
  const auto refused = [&](const fs::path& path, const std::string& member,
                           const std::string& what) {
    try {
      static_cast<void>(app::RunKernel(path));
      ADD_FAILURE() << "accepted on " << Size() << " ranks";
    } catch (const pm::Error& error) {
      const auto message = std::string{error.what()};
      EXPECT_NE(message.find(member), std::string::npos) << message;
      EXPECT_NE(message.find(what), std::string::npos) << message;
    }
  };
  for (const auto* method : {"", kContinuous}) {
    refused(config(R"({"seismic": [{"point": {"latitude": 30, "longitude": 40,
                        "depth_km": 300, "strike_deg": 20, "dip_deg": 60,
                        "rake_deg": 90, "moment_nm": 1e21}}]})",
                   R"(["viscosity"])", method),
            "forcing.seismic",
            "the 3-D kernels about a run under them are formed on one "
            "process");
  }
  const auto tide = config(R"({"tide": [{"degree": 2, "order": 1,
                                          "amplitude_m2_s2": 1.0}]})",
                           R"(["tide"])", kContinuous);
  refused(tide, "kernels.parameters[0]",
          "asks the tide kernels of the continuous 3-D sweep, which are "
          "formed on one process");
  refused(tide, "kernels.parameters[0]",
          "take the discrete sweep's, which are distributed (`kernels.method` "
          "\"discrete\", the default)");
}

/// A planetmodel model file over the prem8 deck's solid layers (or those
/// of `deck`, a deck in sl3d's own format) holding the fields given, each
/// a function of (r, theta, phi) in SI metres and radians: a perturbation
/// of the Earth's parameters, a field of a direction in the viscosity with
/// its spec (test::LogViscositySpec). Written by rank 0.
void WriteModelFields(
    const fs::path& path,
    const std::map<std::string, std::function<double(double, double, double)>>&
        fields,
    const fs::path& own = {}) {
  if (Rank() == 0) {
    const auto data = std::string{SL3DPP_TEST_DATA_DIR};
    const auto deck =
        own.empty() ? Deck::Read(data + "/prem8.elastic.deck",
                                 app::FormatOf(app::ModelFormat::Elastic))
                    : Deck::Read(own, app::FormatOf(app::ModelFormat::Gia));
    ASSERT_TRUE(deck);
    const auto model = EarthModel::FromDeck(*deck);
    ASSERT_TRUE(model);
    const auto& boundaries = model->Boundaries();
    auto file = pm::ModelFile(pm::Skeleton(
        std::vector<double>(boundaries.begin(), boundaries.end())));
    file.SetName("sl3dpp tangent tests");
    const auto& skeleton = file.GetSkeleton();
    const auto layers = static_cast<int>(skeleton.NLayers());
    file.AddPart(0, layers - 1,
                 pm::ElementEdges(skeleton, 0, layers - 1, 500e3), 5,
                 pm::GaussLegendre(8));
    for (const auto& [name, field] : fields) {
      auto sources = std::map<int, pm::LayerSource>{};
      for (auto layer = 0; layer < layers; ++layer) {
        if (model->IsFluid(static_cast<std::size_t>(layer))) continue;
        sources.emplace(layer, pm::LayerSource::Lateral(field));
      }
      if (const auto spec = test::LogViscositySpec(name)) {
        file.AddField(name, *spec, sources);
      } else {
        file.AddField(name, sources);
      }
    }
    pm::WriteModelFile(path, file);
  }
  Barrier();
}

/// The largest relative departure of two tangent surface files' response
/// fields, each over the largest magnitude of the reference's.
double TangentDeparture(const fs::path& mine, const fs::path& theirs) {
  const auto got = pm::ReadSurfaceFile(mine);
  const auto want = pm::ReadSurfaceFile(theirs);
  auto worst = 0.0;
  for (const auto* name :
       {"relative_sea_level_change", "displacement", "tangential_potential",
        "toroidal_potential", "potential", "initial_sea_level"}) {
    EXPECT_EQ(got.variables.contains(name), want.variables.contains(name))
        << name;
    if (!got.variables.contains(name) || !want.variables.contains(name)) {
      continue;
    }
    worst = std::max(worst, Relative(got.variables.at(name).values,
                                     want.variables.at(name).values));
  }
  return worst;
}

/// The measurements' first-order changes in a tangent run's table.
std::vector<double> ChangesIn(const fs::path& table) {
  auto in = std::ifstream(table);
  EXPECT_TRUE(in) << table;
  auto out = std::vector<double>{};
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line.starts_with("#")) continue;
    auto words = std::istringstream(line);
    auto name = std::string{}, type = std::string{};
    auto value = 0.0, change = 0.0;
    words >> name >> type >> value >> change;
    out.push_back(change);
  }
  return out;
}

TEST_F(AppsOverRanks, TangentMatchesOneProcess) {
  // tangent's 3-D run over the ranks: the tangent-linear run of a run
  // under a lateral viscosity and lateral moduli, along a direction in the
  // ice, a load, ln viscosity and the moduli at once, each rank holding
  // the Earth's part of the direction over its own radial segment. Its
  // file and its measurements' changes are the one-process run's to the
  // substructured solves' round-off.
  WriteToyModuli(dir_);
  if (Rank() == 0) test::WriteToyChange(dir_ / "d_ice.nc", true, 0.02);
  Barrier();
  WriteModelFields(
      dir_ / "d_earth.nc",
      {{"ln_viscosity",
        [](double r, double theta, double phi) {
          return 0.5 + (r / 6368000.0) * std::sin(theta) * std::cos(phi);
        }},
       {"mu",
        [](double, double theta, double phi) {
          return 2e9 * std::sin(theta) * std::sin(phi);
        }},
       {"kappa",
        [](double, double theta, double) { return 3e9 * std::cos(theta); }}});
  auto body = Body();
  const auto viscosity = std::string{"/viscosity_log.nc\"}"};
  body.replace(body.find(viscosity), viscosity.size(),
               viscosity + R"(, "moduli": {"file": "moduli.nc"})");
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", "{" + body + R"(,
      "output": {"surface": "out/)" + name + R"(.nc"},
      "kernels": {"parameters": ["ice"],
        "measurements": [
          {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
           "longitude": 10.0, "width_deg": 25.0},
          {"type": "displacement_point", "name": "uplift", "latitude": 55.0,
           "longitude": 10.0, "date_kyr_bp": 0.0}],
        "output": "out/)" + name + R"(_kernels"},
      "perturbation": {
        "forcing": {"ice_change": [{"change": {"file": "d_ice.nc"}}],
                    "load": [{"cap": {"latitude": 20, "longitude": 120,
                                      "width_deg": 20, "amplitude_kg_m2": 3.0e4},
                              "water_budget": true,
                              "moves_solid_surface": false}]},
        "model": {"viscosity": {"file": "d_earth.nc"},
                  "moduli": {"file": "d_earth.nc"}}},
      "tangent": {"output": {"surface": "out/)" + name + R"(_tangent.nc"}}})");
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunTangent(one)); });
  const auto summary = app::RunTangent(over);
  Barrier();
  if (Rank() != 0) return;
  EXPECT_TRUE(summary.starts_with(
      "3-D (lateral viscosity, lateral moduli), viscoelastic"))
      << summary;
  EXPECT_NE(summary.find("(the tangent-linear run along the perturbation's "
                         "ice_change, load, viscosity, moduli)"),
            std::string::npos)
      << summary;
  const auto fields = TangentDeparture(dir_ / "out" / "over_tangent.nc",
                                       dir_ / "out" / "one_tangent.nc");
  const auto changes =
      Relative(ChangesIn(dir_ / "out" / "over_tangent_measurements.txt"),
               ChangesIn(dir_ / "out" / "one_tangent_measurements.txt"));
  std::cout << std::format(
      "[tangent, over the ranks] fields {:.1e}, measurements' changes "
      "{:.1e} from the one-process run\n",
      fields, changes);
  EXPECT_LT(fields, kOverRanks);
  EXPECT_LT(changes, kOverRanks);
}

TEST_F(AppsOverRanks, OnABallViscousToItsCentreSolveKernelAndTangentMatchOneProcess) {
  // The homogeneous test ball, viscous down to its centre, on lateral
  // moduli (non-zero throughout): the 3-D run carries the element holding
  // the centre, on the first rank's segment. Over the ranks, solve's
  // fields, kernel's kernels in the ice, the viscosity and the moduli, and
  // tangent's run along the ice, ln viscosity and the moduli at once (its
  // file and its measurements' changes) are the one process's to the
  // substructured solves' round-off.
  if (Rank() == 0) {
    auto deck = std::string{"homogeneous test ball\ngia deck for the sl3d tests\n"};
    for (const auto r : {0.0, 6371000.0}) {
      deck += std::format("{} 5500 10000 5500 10000 5500 1 1e21\n", r);
    }
    std::ofstream(dir_ / "ball.deck") << deck;
    const auto read = Deck::Read(dir_ / "ball.deck", app::FormatOf(app::ModelFormat::Gia));
    ASSERT_TRUE(read);
    const auto model = EarthModel::FromDeck(*read);
    ASSERT_TRUE(model);
    ASSERT_TRUE(io::WriteModuli(
        dir_ / "moduli.nc", *model, SurfaceGrid(8),
        [](double, double theta, double phi) {
          return 0.1 * std::sin(theta) * std::cos(phi);
        },
        [](double, double theta, double) { return 0.05 * std::cos(theta); },
        {.name = "toy moduli", .elementMetres = 500e3}));
    test::WriteToyChange(dir_ / "d_ice.nc", true, 0.02);
  }
  Barrier();
  WriteModelFields(
      dir_ / "d_earth.nc",
      {{"ln_viscosity",
        [](double r, double theta, double phi) {
          return 0.5 + (1.0 - r / 6371000.0) * std::sin(theta) * std::cos(phi);
        }},
       {"mu",
        [](double r, double theta, double phi) {
          return 2e9 * (1.0 - 0.5 * r / 6371000.0) * std::sin(theta) * std::sin(phi);
        }},
       {"kappa",
        [](double, double theta, double) { return 3e9 * std::cos(theta); }}},
      dir_ / "ball.deck");
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", R"({
      "problem": "quasi_static",
      "model": {"deck": "ball.deck", "moduli": {"file": "moduli.nc"}},
      "discretisation": {"lmax": 8},
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4},
      "forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]},
      "ocean": {"at": "start", "topography": {"file": "history.nc"},
                "ice": {"file": "history.nc"}},
      "rotation": "full", "moments": "hydrostatic",
      "output": {"surface": "out/)" + name + R"(.nc"},
      "kernels": {"parameters": ["ice", "viscosity", "moduli"],
        "measurements": [
          {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
           "longitude": 10.0, "width_deg": 25.0},
          {"type": "displacement_point", "name": "uplift", "latitude": 55.0,
           "longitude": 10.0, "date_kyr_bp": 0.0}],
        "output": "out/)" + name + R"(_kernels"},
      "perturbation": {
        "forcing": {"ice_change": [{"change": {"file": "d_ice.nc"}}]},
        "model": {"viscosity": {"file": "d_earth.nc"},
                  "moduli": {"file": "d_earth.nc"}}},
      "tangent": {"output": {"surface": "out/)" + name + R"(_tangent.nc"}}})");
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] {
    static_cast<void>(app::RunSolve(one));
    static_cast<void>(app::RunKernel(one));
    static_cast<void>(app::RunTangent(one));
  });
  const auto solved = app::RunSolve(over);
  static_cast<void>(app::RunKernel(over));
  static_cast<void>(app::RunTangent(over));
  Barrier();
  if (Rank() != 0) return;
  EXPECT_TRUE(solved.starts_with("3-D (lateral moduli), viscoelastic")) << solved;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
  const auto distributed = pm::ReadSurfaceFile(dir_ / "out" / "over.nc");
  auto fields = 0.0;
  for (const auto* name : {"relative_sea_level_change", "displacement", "potential"}) {
    fields = std::max(fields, Relative(distributed.variables.at(name).values,
                                       reference.variables.at(name).values));
  }
  const auto kernels = std::max(
      {SurfaceKernelDeparture(dir_ / "out" / "over_kernels" / "site_ice_kernels.nc",
                              dir_ / "out" / "one_kernels" / "site_ice_kernels.nc"),
       KernelFileDeparture(dir_ / "out" / "over_kernels" / "site_viscosity_kernels.nc",
                           dir_ / "out" / "one_kernels" / "site_viscosity_kernels.nc"),
       KernelFileDeparture(dir_ / "out" / "over_kernels" / "site_moduli_kernels.nc",
                           dir_ / "out" / "one_kernels" / "site_moduli_kernels.nc"),
       KernelFileDeparture(dir_ / "out" / "over_kernels" / "uplift_moduli_kernels.nc",
                           dir_ / "out" / "one_kernels" / "uplift_moduli_kernels.nc")});
  const auto tangent = TangentDeparture(dir_ / "out" / "over_tangent.nc",
                                        dir_ / "out" / "one_tangent.nc");
  const auto changes =
      Relative(ChangesIn(dir_ / "out" / "over_tangent_measurements.txt"),
               ChangesIn(dir_ / "out" / "one_tangent_measurements.txt"));
  std::cout << std::format(
      "[ball viscous to its centre, over the ranks] solve {:.1e}, kernels "
      "{:.1e}, tangent {:.1e}, measurements' changes {:.1e} from the one "
      "process\n",
      fields, kernels, tangent, changes);
  EXPECT_LT(fields, kOverRanks);
  EXPECT_LT(kernels, 1e-7);
  EXPECT_LT(tangent, kOverRanks);
  EXPECT_LT(changes, kOverRanks);
}

TEST_F(AppsOverRanks, TangentUnderAnEarthquakeMatchesOneProcess) {
  // A 3-D run through time under a point source whose onset falls between
  // two nodes of its window, where the program steps a pair of its own: the
  // tangent-linear run along a seismic source of that onset and the ice,
  // over the ranks, is the one-process run's.
  const auto data = std::string{SL3DPP_TEST_DATA_DIR};
  if (Rank() == 0) test::WriteToyChange(dir_ / "d_ice.nc", true, 0.02);
  Barrier();
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", std::format(R"({{
      "problem": "quasi_static",
      "model": {{"deck": "{0}/prem8.elastic.deck", "format": "elastic",
                 "viscosity": {{"file": "{0}/viscosity_log.nc"}}}},
      "discretisation": {{"lmax": 8}},
      "times": {{"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4}},
      "forcing": {{"ice_change": [{{"thickness": {{"file": "history.nc"}}}}],
        "seismic": [{{"point": {{"latitude": 30, "longitude": 40,
                      "depth_km": 300, "strike_deg": 20, "dip_deg": 60,
                      "rake_deg": 90, "moment_nm": 1e21}},
                     "onset_kyr_bp": 1}}]}},
      "ocean": {{"at": "start", "topography": {{"file": "history.nc"}},
                "ice": {{"file": "history.nc"}}}},
      "rotation": "full",
      "perturbation": {{"forcing": {{
        "ice_change": [{{"change": {{"file": "d_ice.nc"}}}}],
        "seismic": [{{"point": {{"latitude": 28, "longitude": 43,
                      "depth_km": 250, "moment_tensor_nm": {{"rr": 1e20,
                        "tt": -2e20, "pp": 1e20, "rt": 3e20, "rp": -1e20,
                        "tp": 2e20}}}},
                     "onset_kyr_bp": 1}}]}}}},
      "tangent": {{"output": {{"surface": "out/{1}_tangent.nc"}}}}}})",
                                             data, name));
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunTangent(one)); });
  const auto summary = app::RunTangent(over);
  Barrier();
  if (Rank() != 0) return;
  EXPECT_NE(summary.find("(the tangent-linear run along the perturbation's "
                         "ice_change, seismic)"),
            std::string::npos)
      << summary;
  const auto fields = TangentDeparture(dir_ / "out" / "over_tangent.nc",
                                       dir_ / "out" / "one_tangent.nc");
  std::cout << std::format(
      "[tangent, an earthquake between nodes, over the ranks] fields {:.1e} "
      "from the one-process run\n",
      fields);
  EXPECT_LT(fields, kOverRanks);
}

TEST_F(AppsOverRanks, TotalTangentOfACalibratedRunMatchesOneProcess) {
  // The tangent-linear run of a calibrated migrating 3-D run is by default
  // the total derivative: the change of the calibrated initial sea level is
  // found by tangent-linear runs of the linearised calibration, collective
  // over the ranks, every rank deciding alike when they have converged.
  // Along a thickness history, ln viscosity and the target, the file (its
  // `initial_sea_level` the calibrated one's change) and the runs taken
  // are the one process's.
  if (Rank() == 0) {
    test::WriteToyHistory(dir_ / "marine.nc", 0.0, true);
    // a thickness history over the marine cap and a change of the target's
    // topography: the toy fields themselves, scaled
    test::WriteToyHistory(dir_ / "d_marine.nc", 0.0, true);
  }
  Barrier();
  WriteModelFields(dir_ / "d_earth.nc",
                   {{"ln_viscosity", [](double, double theta, double phi) {
                       return 0.5 + std::sin(theta) * std::cos(phi);
                     }}});
  auto body =
      Body(R"("shorelines": "migrating", "calibration": {"rtol": 1e-10})");
  for (auto at = body.find("history.nc"); at != std::string::npos;
       at = body.find("history.nc", at + 1)) {
    body.replace(at, std::string_view{"history.nc"}.size(), "marine.nc");
  }
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", "{" + body + R"(,
      "solver": {"step_rtol": 1e-10, "step_max_iterations": 200},
      "perturbation": {
        "forcing": {"ice_change": [{"thickness": {"file": "d_marine.nc"}}]},
        "model": {"viscosity": {"file": "d_earth.nc"}},
        "ocean": {"topography": {"file": "d_marine.nc"}}},
      "tangent": {"output": {"surface": "out/)" + name + R"(_tangent.nc"}}})");
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunTangent(one)); });
  const auto summary = app::RunTangent(over);
  Barrier();
  if (Rank() != 0) return;
  EXPECT_NE(summary.find("; total derivative, the initial sea level "
                         "calibrated again, its change in "),
            std::string::npos)
      << summary;
  const auto mine = pm::ReadSurfaceFile(dir_ / "out" / "over_tangent.nc");
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one_tangent.nc");
  EXPECT_EQ(std::get<std::string>(mine.meta.at("calibration_derivative")),
            "total");
  EXPECT_EQ(
      std::get<std::vector<double>>(mine.meta.at("calibration_tangent_runs")),
      std::get<std::vector<double>>(
          reference.meta.at("calibration_tangent_runs")));
  const auto& moved = mine.variables.at("initial_sea_level").values;
  EXPECT_TRUE(std::ranges::any_of(moved, [](double v) { return v != 0.0; }));
  const auto fields = TangentDeparture(dir_ / "out" / "over_tangent.nc",
                                       dir_ / "out" / "one_tangent.nc");
  std::cout << std::format(
      "[tangent, calibrated, total, over the ranks] the linearised "
      "calibration in {} runs; fields {:.1e} from the one-process run\n",
      std::get<std::vector<double>>(mine.meta.at("calibration_tangent_runs"))
          .front(),
      fields);
  EXPECT_LT(fields, 1e-7);
}

TEST_F(AppsOverRanks, RadialTangentIsTheFirstRanksAlone) {
  // A 1-D run of tangent is one process's, as solve's and kernel's: over
  // the ranks the first runs it alone and the others return at once.
  if (Rank() == 0) {
    std::ofstream(dir_ / "ball.deck")
        << "homogeneous test ball\n"
           "gia deck for the sl3d tests\n"
           "0 5500 10000 5500 10000 5500 1 1e21\n"
           "6371000 5500 10000 5500 10000 5500 1 1e21\n";
  }
  Barrier();
  const auto config = [&](const std::string& name) {
    return Write(name + ".json", std::string{R"({
      "problem": "quasi_static",
      "model": {"deck": "ball.deck"}, "discretisation": {"lmax": 8},
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5},
      "forcing": {"ice_change": [{"thickness": {"file": "history.nc"}}]},
      "ocean": {"at": "start", "topography": {"file": "history.nc"},
                "ice": {"file": "history.nc"}},
      "rotation": "full", "moments": "hydrostatic",
      "perturbation": {"forcing": {"tide": [{"degree": 2, "order": 1,
                                             "amplitude_m2_s2": 1.0}]}},
      "tangent": {"output": {"surface": "out/)"} +
                                     name + R"(_tangent.nc"}}})");
  };
  const auto one = config("one");
  const auto over = config("over");
  OneProcess([&] { static_cast<void>(app::RunTangent(one)); });
  const auto summary = app::RunTangent(over);
  if (Rank() == 0) {
    EXPECT_TRUE(summary.starts_with("1-D, viscoelastic")) << summary;
  } else {
    EXPECT_EQ(summary, "");
  }
  Barrier();
  if (Rank() != 0) return;
  // both the first rank's, one process: bitwise
  EXPECT_EQ(pm::ReadSurfaceFile(dir_ / "out" / "over_tangent.nc")
                .variables.at("displacement")
                .values,
            pm::ReadSurfaceFile(dir_ / "out" / "one_tangent.nc")
                .variables.at("displacement")
                .values);
}

TEST_F(AppsOverRanks, SecondOrderKernelsMatchOneProcess) {
  // kernel's second-order kernels of a 3-D run over the ranks
  // (`kernels.second_order`): about a run under a lateral viscosity and
  // lateral moduli, along a direction in the ice, ln viscosity and the
  // moduli at once (each rank holding the Earth's part of the direction over
  // its own radial segment), the tangent-linear run and the second-order
  // sweep collective. The second-order files (the ice's, the fields of the
  // viscosity's and the moduli's) and the first-order files beside them are
  // the one-process run's to the substructured solves' round-off, by
  // either `kernels.method` under Sdirk2, the default scheme.
  WriteToyModuli(dir_);
  if (Rank() == 0) test::WriteToyChange(dir_ / "d_ice.nc", true, 0.02);
  Barrier();
  WriteModelFields(
      dir_ / "d_earth.nc",
      {{"ln_viscosity",
        [](double r, double theta, double phi) {
          return 0.5 + (r / 6368000.0) * std::sin(theta) * std::cos(phi);
        }},
       {"mu",
        [](double, double theta, double phi) {
          return 2e9 * std::sin(theta) * std::sin(phi);
        }},
       {"kappa",
        [](double, double theta, double) { return 3e9 * std::cos(theta); }}});
  auto body = Body();
  const auto viscosity = std::string{"/viscosity_log.nc\"}"};
  body.replace(body.find(viscosity), viscosity.size(),
               viscosity + R"(, "moduli": {"file": "moduli.nc"})");
  const auto config = [&](const std::string& name, const std::string& method) {
    return Write(name + ".json", "{" + body + R"(,
      "kernels": {"parameters": ["ice", "viscosity", "moduli"],
        "measurements": [
          {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
           "longitude": 10.0, "width_deg": 25.0},
          {"type": "displacement_point", "name": "uplift", "latitude": 55.0,
           "longitude": 10.0, "date_kyr_bp": 0.0}],
        "output": "out/)" + name + R"(_kernels", "second_order": true,
        "method": ")" + method + R"("},
      "perturbation": {
        "forcing": {"ice_change": [{"change": {"file": "d_ice.nc"}}]},
        "model": {"viscosity": {"file": "d_earth.nc"},
                  "moduli": {"file": "d_earth.nc"}}}})");
  };
  for (const std::string method : {"discrete", "continuous"}) {
    const auto one = config("one_" + method, method);
    const auto over = config("over_" + method, method);
    OneProcess([&] { static_cast<void>(app::RunKernel(one)); });
    const auto summary = app::RunKernel(over);
    Barrier();
    if (Rank() != 0) continue;
    EXPECT_NE(summary.find("; and their second-order kernels along the "
                           "perturbation's ice_change, viscosity, moduli)"),
              std::string::npos)
        << summary;
    const auto mine = dir_ / "out" / ("over_" + method + "_kernels");
    const auto theirs = dir_ / "out" / ("one_" + method + "_kernels");
    auto second = 0.0, first = 0.0;
    for (const auto* m : {"site", "uplift"}) {
      const auto name = std::string{m};
      for (const auto* suffix : {"", "_second_order"}) {
        const auto ice = name + "_ice_kernels" + suffix + ".nc";
        const auto eta = name + "_viscosity_kernels" + suffix + ".nc";
        const auto moduli = name + "_moduli_kernels" + suffix + ".nc";
        auto& worst = std::string_view{suffix}.empty() ? first : second;
        worst = std::max({worst, SurfaceKernelDeparture(mine / ice, theirs / ice),
                          KernelFileDeparture(mine / eta, theirs / eta),
                          KernelFileDeparture(mine / moduli, theirs / moduli)});
      }
    }
    std::cout << std::format(
        "[second-order kernels, over the ranks, {}] second order {:.1e}, first "
        "order {:.1e} from the one-process run\n",
        method, second, first);
    // measured on two and three ranks, discrete: 8.8e-11 and 3.3e-10
    // (second order), 7.5e-10 and 3.8e-10 (first order); continuous: 4.2e-10
    // and 2.5e-10, 2.1e-9 and 9.6e-10
    EXPECT_LT(second, kOverRanks) << method;
    EXPECT_LT(first, kOverRanks) << method;
  }
}

TEST_F(AppsOverRanks, SecondOrderKernelsOfAMigratingRunMatchOneProcess) {
  // The second-order kernels about a run under migrating shorelines (the
  // marine history, whose melting ice floods its cells; the state at the
  // window's start, so no calibration), over the ranks: along a direction
  // in the ice and ln viscosity at once, the tangent-linear run and the
  // second-order sweep reading the run's shorelines on every rank. The
  // second-order files and the first-order files beside them are the
  // one-process run's to the substructured solves' round-off, by either
  // `kernels.method`.
  if (Rank() == 0) {
    test::WriteToyHistory(dir_ / "marine.nc", 0.0, true);
    test::WriteToyChange(dir_ / "d_ice.nc", true, 0.02);
  }
  Barrier();
  WriteModelFields(
      dir_ / "d_earth.nc",
      {{"ln_viscosity", [](double r, double theta, double phi) {
          return 0.5 + (r / 6368000.0) * std::sin(theta) * std::cos(phi);
        }}});
  auto body = Body();
  for (auto at = body.find("history.nc"); at != std::string::npos;
       at = body.find("history.nc", at + 1)) {
    body.replace(at, std::string_view{"history.nc"}.size(), "marine.nc");
  }
  body.replace(body.find(R"("at": "start")"),
               std::string_view{R"("at": "start")"}.size(),
               R"("shorelines": "migrating", "at": "start")");
  const auto config = [&](const std::string& name, const std::string& method) {
    return Write(name + ".json", "{" + body + R"(,
      "solver": {"step_rtol": 1e-10, "step_max_iterations": 200},
      "kernels": {"parameters": ["ice", "viscosity"],
        "measurements": [
          {"type": "relative_sea_level_cap", "name": "site", "latitude": -30.0,
           "longitude": 10.0, "width_deg": 25.0},
          {"type": "displacement_point", "name": "uplift", "latitude": 55.0,
           "longitude": 10.0, "date_kyr_bp": 0.0}],
        "output": "out/)" + name + R"(_kernels", "second_order": true,
        "method": ")" + method + R"("},
      "perturbation": {
        "forcing": {"ice_change": [{"change": {"file": "d_ice.nc"}}]},
        "model": {"viscosity": {"file": "d_earth.nc"}}}})");
  };
  for (const std::string method : {"discrete", "continuous"}) {
    const auto one = config("one_" + method, method);
    const auto over = config("over_" + method, method);
    OneProcess([&] { static_cast<void>(app::RunKernel(one)); });
    const auto summary = app::RunKernel(over);
    Barrier();
    if (Rank() != 0) continue;
    EXPECT_NE(summary.find("shorelines migrating"), std::string::npos) << summary;
    EXPECT_NE(summary.find("; and their second-order kernels along the "
                           "perturbation's ice_change, viscosity)"),
              std::string::npos)
        << summary;
    const auto mine = dir_ / "out" / ("over_" + method + "_kernels");
    const auto theirs = dir_ / "out" / ("one_" + method + "_kernels");
    auto second = 0.0, first = 0.0;
    for (const auto* m : {"site", "uplift"}) {
      const auto name = std::string{m};
      for (const auto* suffix : {"", "_second_order"}) {
        const auto ice = name + "_ice_kernels" + suffix + ".nc";
        const auto eta = name + "_viscosity_kernels" + suffix + ".nc";
        auto& worst = std::string_view{suffix}.empty() ? first : second;
        worst = std::max({worst, SurfaceKernelDeparture(mine / ice, theirs / ice),
                          KernelFileDeparture(mine / eta, theirs / eta)});
      }
    }
    std::cout << std::format(
        "[second-order kernels, migrating, over the ranks, {}] second order "
        "{:.1e}, first order {:.1e} from the one-process run\n",
        method, second, first);
    // measured on two and three ranks, discrete: 2.6e-11 and 2.0e-11
    // (second order), 5.9e-11 and 7.8e-11 (first order); continuous:
    // 3.0e-11 and 2.2e-11, 1.1e-10 and 2.4e-11
    EXPECT_LT(second, kOverRanks) << method;
    EXPECT_LT(first, kOverRanks) << method;
  }
}

}  // namespace

int main(int argc, char** argv) {
  auto provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  app::StartRanks();
  ::testing::InitGoogleTest(&argc, argv);
  if (Rank() != 0) {
    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
  }
  sl3dpp::test::EndEveryRankOnAFatalFailure();
  auto failed = RUN_ALL_TESTS() != 0 ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return failed;
}
