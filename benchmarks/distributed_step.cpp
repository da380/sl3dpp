// The phases of one pass of the 3D step distributed over MPI ranks
// (Viscoelastic3D::Options::segments and collectives), built with
// SL3DPP_WITH_MPI: one radial segment per rank, the cuts balanced on the
// transform work (GridCoupling::TransformWeights), OpenMP threads inside
// each rank. step_timing's setup (PREM, the Y_{2,+-2} log-viscosity
// pattern, a constant zonal load rate, 50 yr trapezoid steps) and its
// phases, timed alone (no whole steps), each the slowest rank's best of
// `repeats`:
//
//   analyse     GridCoupling::Analyse of the rank's own nodes
//   synthesise  GridCoupling::Synthesise likewise
//   respond     OpenStep::Respond: the transforms, the force, the solves
//               and their one collective (the cut nodes' sums)
//   strainOf    OpenStep::StrainOf (strain recovery, own nodes)
//   startStep   once per step
//
// with each rank's memory nodes, its TensorField's size and the peak RSS.
// Run it as ranks x threads over a node, e.g.
//
//   OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close \
//     mpiexec -n 2 -bind-to core:4 distributed_step <deck> <lmax> 2 32,48
//
// Usage: distributed_step <deck> <lmax> [repeats] [degrees]

#include <mpi.h>
#include <omp.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/Distributed.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using Complex = std::complex<double>;

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

// The slowest rank's best of `repeats` timings of f, each started together.
double Best(int repeats, const std::function<void()>& f) {
  auto best = 0.0;
  for (auto i = 0; i < repeats; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    f();
    const auto t = MaxOverRanks(Seconds(start));
    best = i == 0 ? t : std::min(best, t);
  }
  return best;
}

double PeakRssGb() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
  auto provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  auto ranks = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc < 3) {
    if (rank == 0) {
      std::fprintf(stderr,
                   "usage: mpiexec -n <ranks> distributed_step <deck> <lmax> "
                   "[repeats] [degrees]\n");
    }
    MPI_Finalize();
    return 2;
  }
  const auto lmax = std::atoi(argv[2]);
  const auto repeats = argc > 3 ? std::atoi(argv[3]) : 2;
  auto degrees = std::vector<int>{};
  if (argc > 4) {
    const auto list = std::string(argv[4]);
    for (auto at = std::size_t{0}; at < list.size();) {
      auto end = list.find(',', at);
      if (end == std::string::npos) end = list.size();
      degrees.push_back(std::stoi(list.substr(at, end - at)));
      at = end + 1;
    }
  }
  const auto threads = omp_get_max_threads();
  constexpr auto year = 365.25 * 24.0 * 3600.0;

  auto deck = Deck::ReadLegacySl3d(argv[1]).value();
  auto model =
      EarthModel::FromDeck(deck, {.spline = SplineKind::Natural}).value();
  auto mesh = RadialMesh::CreateGraded(model, lmax).value();
  auto material = Material::Create(model, mesh).value();
  auto rheology = Rheology::FromModel(material, model).value();
  const auto& scales = model.ModelScales();
  const auto c22 = 0.25 * std::sqrt(15.0 / (2.0 * std::numbers::pi));
  auto delta = [c22](double, double theta, double phi) {
    return -0.6 * c22 * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi);
  };

  const auto share =
      ShareMesh(material.Mesh(), MPI_COMM_WORLD,
                GridCoupling::TransformWeights(rheology, lmax,
                                               {.transform_degrees = degrees}));
  auto options = Viscoelastic3D::Options{};
  options.parallel = threads > 1;
  options.rtol = 1e-10;
  options.radial_cuts = share.cuts;
  options.transform_degrees = degrees;
  options.segments = share.segments;
  options.collectives = share.collectives;
  const Viscoelastic3D op(rheology, lmax, delta, options);
  const auto& coupling = op.Coupling();

  const auto sigmaFactor = scales.Factor({.mass = 1, .length = -2});
  const auto timeFactor = scales.Time();
  auto rate = std::vector<double>(lmax + 1, 0.0);
  for (auto l = 1; l <= lmax; ++l) {
    rate[l] = std::exp(-2.0 * std::numbers::pi * (l + 1) / 10.5) *
              std::sqrt((2.0 * l + 1) / (4.0 * std::numbers::pi)) * 5515.0 /
              (30.0 * year);
  }
  auto load = [&](int l, double t) -> std::optional<std::vector<Complex>> {
    if (l < 1) return std::nullopt;
    auto columns = std::vector<Complex>(l + 1, 0.0);
    columns[0] = rate[l] * (t * timeFactor) / sigmaFactor;
    return columns;
  };
  const auto dt = 50.0 * year / timeFactor;
  const auto scheme = Scheme::ExponentialTrapezoid;

  // Per rank: memory nodes and TensorField size, gathered for rank 0.
  auto nodes = std::vector<double>(ranks, 0.0);
  auto gigabytes = std::vector<double>(ranks, 0.0);
  nodes[rank] = coupling.NodeCount();
  gigabytes[rank] = static_cast<double>(coupling.GridValueCount()) * 40.0 / 1e9;
  share.collectives.sum(nodes);
  share.collectives.sum(gigabytes);

  // One warm-up step builds and factorises every operator.
  MPI_Barrier(MPI_COMM_WORLD);
  auto start = std::chrono::steady_clock::now();
  auto state = op.Step(op.ZeroState(), dt, scheme, load).state;
  const auto warmUp = MaxOverRanks(Seconds(start));

  auto analyse = 0.0, synthesise = 0.0, respond = 0.0, strainOf = 0.0;
  {
    auto step = op.StartStep(state, dt, scheme, load);
    const auto& s0 = step.StartStrain();
    const auto d = coupling.Synthesise(s0);
    auto spectral = coupling.Analyse(d);
    analyse = Best(repeats, [&] { coupling.Analyse(d, spectral); });
    auto field = coupling.Synthesise(spectral);
    synthesise = Best(repeats, [&] { coupling.Synthesise(spectral, field); });
    // The first Respond of s0 finds its force formed by StartStep.
    auto solutions = step.Respond(s0);
    respond = Best(repeats, [&] { solutions = step.Respond(s0); });
    strainOf = Best(repeats, [&] { auto out = step.StrainOf(solutions); });
  }
  const auto startStep = Best(
      repeats, [&] { auto opened = op.StartStep(state, dt, scheme, load); });
  const auto rss = MaxOverRanks(PeakRssGb());

  if (rank == 0) {
    std::printf("distributed_step: lmax %d, %d rank%s x %d thread%s", lmax,
                ranks, ranks == 1 ? "" : "s", threads, threads == 1 ? "" : "s");
    if (!degrees.empty()) {
      std::printf(", transform degrees");
      for (const auto d : degrees) std::printf(" %d", d);
    }
    std::printf("\n  per rank: memory nodes");
    for (const auto n : nodes) std::printf(" %.0f", n);
    std::printf("; TensorField GB");
    for (const auto g : gigabytes) std::printf(" %.2f", g);
    std::printf("\n  warm-up step %.3f s\n", warmUp);
    std::printf("  one pass, slowest rank's best of %d [s]\n", repeats);
    std::printf("    %-11s %9.4f\n", "analyse", analyse);
    std::printf("    %-11s %9.4f\n", "synthesise", synthesise);
    std::printf("    %-11s %9.4f  (transforms, force, solves, one sum)\n",
                "respond", respond);
    std::printf("    %-11s %9.4f\n", "strainOf", strainOf);
    std::printf("    %-11s %9.4f  (once per step)\n", "startStep", startStep);
    std::printf("  peak RSS, largest rank %.2f GB\n", rss);
  }
  MPI_Finalize();
  return 0;
}
