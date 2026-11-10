// sl3d's 3D runs over MPI ranks (apps/Ranks.hpp), built with
// SL3DPP_WITH_MPI and run under mpiexec: sea-level's 3-D run, kernel and
// the viscoelastic response over the ranks against the same configs on one
// process (rank 0 alone, as a run of one rank), a run resumed from
// per-rank checkpoints against the uninterrupted one, the spilled
// trajectory in per-rank subdirectories against the in-memory one, and a
// 1-D sea-level run, which is the first rank's alone. Rank 0 writes the
// outputs and compares them; every rank's verdict is reduced to rank 0,
// which alone prints.

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <planetmodel/modelfile.hpp>
#include <planetmodel/surfacefile.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "AppFixtures.hpp"
#include "Commands.hpp"
#include "Ranks.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/SurfaceGrid.hpp"
#include "sl3dpp/io/ModelModuli.hpp"

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

  [[nodiscard]] static std::string Body() {
    const auto data = std::string{SL3DPP_TEST_DATA_DIR};
    return std::string{R"(
      "model": {"deck": ")"} +
           data + R"(/prem8.elastic.deck", "format": "elastic",
                "viscosity": {"file": ")" +
           data + R"(/viscosity_log.nc"}},
      "lmax": 8,
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 4},
      "forcing": {"state": {"file": "history.nc"},
                  "ice": {"file": "history.nc"}},
      "rotation": "full")";
  }

  fs::path dir_;
};

TEST_F(AppsOverRanks, SeaLevelMatchesOneProcessAndResumesFromRankCheckpoints) {
  const auto one = Write("one.json", "{" + Body() + R"(,
      "output": {"surface": "out/one.nc"}})");
  const auto over = Write("over.json", "{" + Body() + R"(,
      "output": {"surface": "out/over.nc"}})");
  const auto pieces = Write("pieces.json", "{" + Body() + R"(,
      "checkpoint": {"directory": "ckpt", "every": 1,
                     "max_walltime_min": 1e-9},
      "output": {"surface": "out/pieces.nc"}})");
  OneProcess([&] { static_cast<void>(app::RunSeaLevel(one)); });
  const auto summary = app::RunSeaLevel(over);
  EXPECT_NE(summary.find("wrote"), std::string::npos) << summary;
  // stopped at every checkpoint by the wall-time limit, and resumed from
  // the node every rank's own subdirectory holds
  auto attempts = 0;
  for (; attempts < 8; ++attempts) {
    const auto part = app::RunSeaLevel(pieces);
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
    for (const auto* name : {"sea_level_change", "displacement", "potential"}) {
      EXPECT_LT(Relative(distributed.variables.at(name).values,
                         reference.variables.at(name).values),
                1e-10)
          << name;
      EXPECT_EQ(resumed.variables.at(name).values,
                distributed.variables.at(name).values)
          << name;
    }
  }
}

TEST_F(AppsOverRanks, CalibratedMigratingSeaLevelMatchesOneProcess) {
  // Calibration iterates whole runs until the initial sea level settles:
  // every rank must decide alike when to stop (they see the same, rank 0's,
  // sea level), and the calibrated run, migrating shorelines and all,
  // matches the one-process one; stopped at every checkpoint, it resumes
  // without calibrating again, as uninterrupted.
  if (Rank() == 0) test::WriteToyHistory(dir_ / "marine.nc", 0.0, true);
  Barrier();
  auto body = Body();
  for (const auto* from : {"\"file\": \"history.nc\""}) {
    for (auto at = body.find(from); at != std::string::npos;
         at = body.find(from, at + 1)) {
      body.replace(at, std::string_view{from}.size(),
                   "\"file\": \"marine.nc\"");
    }
  }
  body += R"(,
      "ocean": {"shorelines": "migrating",
                "calibrate_initial_sea_level": {"rtol": 1e-8}})";
  const auto one = Write("one.json", "{" + body + R"(,
      "output": {"surface": "out/one.nc"}})");
  const auto over = Write("over.json", "{" + body + R"(,
      "output": {"surface": "out/over.nc"}})");
  const auto pieces = Write("pieces.json", "{" + body + R"(,
      "checkpoint": {"directory": "ckpt", "every": 1,
                     "max_walltime_min": 1e-9},
      "output": {"surface": "out/pieces.nc"}})");
  OneProcess([&] { static_cast<void>(app::RunSeaLevel(one)); });
  const auto summary = app::RunSeaLevel(over);
  EXPECT_NE(summary.find("initial sea level calibrated"), std::string::npos)
      << summary;
  for (auto attempts = 0; attempts < 8; ++attempts) {
    if (app::RunSeaLevel(pieces).find("wrote") != std::string::npos) break;
  }
  Barrier();
  if (Rank() != 0) return;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
  const auto distributed = pm::ReadSurfaceFile(dir_ / "out" / "over.nc");
  const auto resumed = pm::ReadSurfaceFile(dir_ / "out" / "pieces.nc");
  for (const auto* name :
       {"sea_level_change", "initial_sea_level", "ocean_function"}) {
    EXPECT_LT(Relative(distributed.variables.at(name).values,
                       reference.variables.at(name).values),
              1e-8)
        << name;
    EXPECT_EQ(resumed.variables.at(name).values,
              distributed.variables.at(name).values)
        << name;
  }
}

TEST_F(AppsOverRanks, KernelMatchesOneProcessAndSpillsPerRank) {
  const auto measurement = std::string{R"(
      "measurements": [
        {"type": "sea_level_cap", "name": "site", "latitude": -30.0,
         "longitude": 10.0, "width_deg": 25.0, "date_kyr_bp": 0.0}])"};
  const auto one = Write("one.json", "{" + Body() + "," + measurement +
                                         R"(, "output": "out_one"})");
  const auto over = Write("over.json", "{" + Body() + "," + measurement +
                                           R"(, "output": "out_over"})");
  const auto memory = Write("memory.json", "{" + Body() + "," + measurement +
                                               R"(, "record_stride": 2,
      "output": "out_memory"})");
  const auto spilled = Write("spilled.json", "{" + Body() + "," + measurement +
                                                 R"(, "record_stride": 2,
      "trajectory": {"directory": "spill"}, "output": "out_spilled"})");
  OneProcess([&] { static_cast<void>(app::RunKernel3D(one)); });
  static_cast<void>(app::RunKernel3D(over));
  static_cast<void>(app::RunKernel3D(memory));
  const auto summary = app::RunKernel3D(spilled);
  EXPECT_NE(summary.find("spilled state(s) removed"), std::string::npos)
      << summary;
  // each rank spilled into its own subdirectory, and cleared it
  const auto mine = dir_ / "spill" / std::format("rank_{:04}", Rank());
  EXPECT_TRUE(fs::is_directory(mine));
  EXPECT_FALSE(fs::exists(mine / "node_000000.traj"));
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
              1e-10)
        << name;
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
                1e-10)
          << name << " in part " << p;
      EXPECT_EQ(fromDisk.Parts()[p].variables.at(name).values,
                fromMemory.Parts()[p].variables.at(name).values)
          << name << " in part " << p;
    }
  }
  EXPECT_GT(compared, 0);
}

TEST_F(AppsOverRanks, SeaLevelUnderLateralModuliMatchesOneProcess) {
  // Under lateral moduli the rank cuts balance the transform work of the
  // elements the moduli perturb as well as the viscous ones: the run over
  // the ranks is the one-process run's to the substructured solves'
  // round-off.
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
  OneProcess([&] { static_cast<void>(app::RunSeaLevel(one)); });
  const auto summary = app::RunSeaLevel(over);
  if (Rank() == 0) {
    EXPECT_TRUE(summary.starts_with(
        "3-D (lateral viscosity, lateral moduli), viscoelastic"))
        << summary;
  }
  Barrier();
  if (Rank() != 0) return;
  const auto reference = pm::ReadSurfaceFile(dir_ / "out" / "one.nc");
  const auto distributed = pm::ReadSurfaceFile(dir_ / "out" / "over.nc");
  for (const auto* name : {"sea_level_change", "displacement", "potential"}) {
    EXPECT_LT(Relative(distributed.variables.at(name).values,
                       reference.variables.at(name).values),
              1e-10)
        << name;
  }
}

TEST_F(AppsOverRanks, RadialSeaLevelIsTheFirstRanksAlone) {
  // A 1-D sea-level run is one process's: over the ranks the first runs it
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
      "model": {"deck": "ball.deck"}, "lmax": 8,
      "times": {"start_kyr_bp": 2.0, "end_kyr_bp": 0.0, "nodes": 5},
      "forcing": {"state": {"file": "history.nc"},
                  "ice": {"file": "history.nc"}},
      "rotation": "full", "moments": "hydrostatic")"};
  const auto one = Write("one.json", "{" + body + R"(,
      "output": {"surface": "out/one.nc"}})");
  const auto over = Write("over.json", "{" + body + R"(,
      "checkpoint": {"directory": "ckpt", "every": 2},
      "output": {"surface": "out/over.nc"}})");
  OneProcess([&] { static_cast<void>(app::RunSeaLevel(one)); });
  const auto summary = app::RunSeaLevel(over);
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
  for (const auto* name : {"sea_level_change", "displacement", "potential"}) {
    EXPECT_EQ(first.variables.at(name).values,
              reference.variables.at(name).values)
        << name;
  }
}

TEST_F(AppsOverRanks, ViscoelasticResponseMatchesOneProcess) {
  // The viscoelastic response under a lateral viscosity and lateral
  // moduli, each step closed by the fixed point and by conjugate
  // gradients: over the ranks (the first node's conjugate-gradient solve
  // and every step distributed over radial segments) as on one process,
  // to the radially substructured solves' round-off and the solvers'
  // tolerance; the radial reference model beside it is the first rank's
  // alone and the same.
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
      "model": {"deck": ")"} +
                    data +
                    R"(/prem8.elastic.deck", "format": "elastic",
                "viscosity": {"file": ")" +
                    data + R"(/viscosity_log.nc"},
                "moduli": {"file": "moduli.nc"}},
      "lmax": 8,
      "times": {"end_kyr": 1, "nodes": 3},
      "forcing": {
        "load": [{"cap": {"latitude": 60, "longitude": -85, "width_deg": 20,
                          "amplitude_kg_m2": 1.0e6},
                  "history_kyr": [[0, 0.5], [1, 1]]}],
        "tide": [{"degree": 2, "order": 1, "amplitude_m2_s2": 1.0}]})";
  for (const auto* step : {"fixed_point", "conjugate_gradients"}) {
    const auto config = [&](const std::string& name) {
      return Write(name + ".json",
                   std::format(R"({{{}, "solver": {{"step": "{}"}},
                                   "output": {{"surface": "out/{}.nc"}}}})",
                               body, step, name));
    };
    const auto one = config(std::format("one_{}", step));
    const auto over = config(std::format("over_{}", step));
    OneProcess([&] { static_cast<void>(app::RunResponse(one)); });
    const auto summary = app::RunResponse(over);
    if (Rank() == 0) {
      EXPECT_TRUE(summary.starts_with(
          "3-D (lateral viscosity, lateral moduli), viscoelastic"))
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
         {"load", "displacement", "tangential_potential", "potential",
          "displacement_radial_model", "potential_radial_model"}) {
      EXPECT_LT(Relative(distributed.variables.at(name).values,
                         reference.variables.at(name).values),
                1e-10)
          << step << " " << name;
    }
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
  auto failed = RUN_ALL_TESTS() != 0 ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return failed;
}
