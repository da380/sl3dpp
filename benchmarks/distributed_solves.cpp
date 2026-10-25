// The distributed degree solves over MPI ranks (Distributed.hpp), built
// with SL3DPP_WITH_MPI: every spheroidal and toroidal degree system of a
// model at lmax, one radial segment per rank, 2(l + 1) columns per degree
// as a 3D pass solves them, one thread per rank. Reports the slowest
// rank's construction and solve times summed over degrees (the
// collectives included: one all-reduce per construction and per solve),
// and the one-process direct solves' for comparison (every rank times
// them; rank 0's are reported). TestDistributed holds the answers to the
// direct ones.
//
// Usage:
//   mpiexec -n <ranks> distributed_solves <deck> <lmax> [repeats]
//
// The deck is read as step_timing reads it (e.g. tests/data/prem.simp.vm1).

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/Distributed.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/ToroidalOperator.hpp"

using namespace sl3dpp;

namespace {

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

double MaxOverRanks(double value) {
  auto out = 0.0;
  MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  auto ranks = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc < 3) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "usage: mpiexec -n <ranks> distributed_solves <deck> "
                   "<lmax> [repeats]\n");
    }
    MPI_Finalize();
    return 2;
  }
  const auto lmax = std::atoi(argv[2]);
  const auto repeats = argc > 3 ? std::atoi(argv[3]) : 3;
  auto deck = Deck::ReadLegacySl3d(argv[1]).value();
  auto model =
      EarthModel::FromDeck(deck, {.spline = SplineKind::Natural}).value();
  auto mesh = RadialMesh::CreateGraded(model, lmax).value();
  auto material = Material::Create(model, mesh).value();
  const auto share = ShareMesh(mesh, MPI_COMM_WORLD);

  auto engine = std::mt19937{11};
  auto uniform = std::uniform_real_distribution<double>{-1.0, 1.0};
  auto directBuild = 0.0, directSolve = 0.0;
  auto build = 0.0, solve = 0.0;
  auto exchanged = 0.0;  // doubles summed per pass, over all degrees
  for (auto l = 0; l <= lmax; ++l) {
    const auto columns = 2 * (l + 1);
    // The same right-hand sides on every rank (the same seed).
    auto start = std::chrono::steady_clock::now();
    const DegreeOperator direct(material, l);
    directBuild += Seconds(start);
    auto b = std::vector<double>(static_cast<std::size_t>(direct.DofCount()) *
                                 columns);
    for (auto& v : b) v = uniform(engine);
    auto best = 1e300;
    for (auto r = 0; r < repeats; ++r) {
      start = std::chrono::steady_clock::now();
      static_cast<void>(direct.Solve(b, columns));
      best = std::min(best, Seconds(start));
    }
    directSolve += best;

    MPI_Barrier(MPI_COMM_WORLD);
    start = std::chrono::steady_clock::now();
    const DegreeOperator system(material, l,
                                {.cuts = share.cuts,
                                 .segments = share.segments,
                                 .combine = share.collectives.sum});
    build += MaxOverRanks(Seconds(start));
    best = 1e300;
    for (auto r = 0; r < repeats; ++r) {
      MPI_Barrier(MPI_COMM_WORLD);
      start = std::chrono::steady_clock::now();
      static_cast<void>(system.Solve(b, columns));
      best = std::min(best, MaxOverRanks(Seconds(start)));
    }
    solve += best;
    exchanged += 3.0 * (system.SegmentCount() - 1) * columns;
    if (l == 0) continue;
    start = std::chrono::steady_clock::now();
    const ToroidalOperator directToroidal(material, l);
    directBuild += Seconds(start);
    auto bt = std::vector<double>(
        static_cast<std::size_t>(directToroidal.DofCount()) * columns);
    for (auto& v : bt) v = uniform(engine);
    best = 1e300;
    for (auto r = 0; r < repeats; ++r) {
      start = std::chrono::steady_clock::now();
      static_cast<void>(directToroidal.Solve(bt, columns));
      best = std::min(best, Seconds(start));
    }
    directSolve += best;
    MPI_Barrier(MPI_COMM_WORLD);
    start = std::chrono::steady_clock::now();
    const ToroidalOperator toroidal(material, l,
                                    {.cuts = share.cuts,
                                     .segments = share.segments,
                                     .combine = share.collectives.sum});
    build += MaxOverRanks(Seconds(start));
    best = 1e300;
    for (auto r = 0; r < repeats; ++r) {
      MPI_Barrier(MPI_COMM_WORLD);
      start = std::chrono::steady_clock::now();
      static_cast<void>(toroidal.Solve(bt, columns));
      best = std::min(best, MaxOverRanks(Seconds(start)));
    }
    solve += best;
  }
  if (rank == 0) {
    std::printf(
        "# %s, lmax %d, %d rank(s) of one thread; every spheroidal and "
        "toroidal degree\n"
        "direct       build %.3f s  solves %.3f s\n"
        "distributed  build %.3f s  solves %.3f s  (slowest rank; %.0f "
        "separator values summed per pass)\n",
        argv[1], lmax, ranks, directBuild, directSolve, build, solve,
        exchanged);
  }
  MPI_Finalize();
  return 0;
}
