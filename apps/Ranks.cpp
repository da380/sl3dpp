#include "Ranks.hpp"

#include <format>
#include <execinfo.h>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

#include "sl3dpp/GridCoupling.hpp"

#ifdef SL3DPP_HAVE_MPI
#include <mpi.h>

#include "sl3dpp/Distributed.hpp"
#endif

namespace sl3dpp::app {

namespace {

Ranks& Current() {
  static auto ranks = Ranks{};
  return ranks;
}

}  // namespace

const Ranks& TheRanks() { return Current(); }

void StartRanks() {
#ifdef SL3DPP_HAVE_MPI
  auto& ranks = Current();
  MPI_Comm_rank(MPI_COMM_WORLD, &ranks.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks.size);
  ranks.collectives =
      ranks.size > 1 ? CollectivesOver(MPI_COMM_WORLD) : Collectives{};
#endif
}

RanksOverride::RanksOverride(Ranks ranks)
    : saved_(std::exchange(Current(), std::move(ranks))) {}

RanksOverride::~RanksOverride() { Current() = std::move(saved_); }

Viscoelastic3D::Options OperatorOptions(const EvolveInput& input,
                                        const Rheology& rheology) {
  return OperatorOptions(input.lmax, input.transform_degrees, input.cache_mb,
                         rheology);
}

Viscoelastic3D::Options OperatorOptions(int lmax,
                                        const std::vector<int>& transform_degrees,
                                        std::size_t cache_mb,
                                        const Rheology& rheology,
                                        const std::vector<char>& active_elements) {
  auto options = Viscoelastic3D::Options{};
  options.transform_degrees = transform_degrees;
  options.cache_bytes = cache_mb << 20;
  const auto& ranks = TheRanks();
  if (!ranks.Distributed()) return options;
#ifdef SL3DPP_HAVE_MPI
  // The coupling the operator builds from these options, and so the work
  // its cuts balance.
  const auto weights = GridCoupling::TransformWeights(
      rheology, lmax,
      {.eps = options.eps,
       .truncation = options.truncation,
       .oversampling = options.oversampling,
       .cache_bytes = options.cache_bytes,
       .memory_points = options.memory_points,
       .transform_degrees = options.transform_degrees,
       .active_elements = active_elements});
  const auto share =
      ShareMesh(rheology.BaseMaterial().Mesh(), MPI_COMM_WORLD, weights);
  options.radial_cuts = share.cuts;
  options.segments = share.segments;
  options.collectives = ranks.collectives;
  return options;
#else
  static_cast<void>(rheology);
  throw std::logic_error("several ranks without MPI");
#endif
}

std::filesystem::path RankDirectory(const std::filesystem::path& directory) {
  const auto& ranks = TheRanks();
  return ranks.Distributed() ? directory / std::format("rank_{:04}", ranks.rank)
                             : directory;
}

std::vector<double> GatherToFirst(std::span<const double> part) {
  const auto& ranks = TheRanks();
  if (!ranks.Distributed()) return {part.begin(), part.end()};
#ifdef SL3DPP_HAVE_MPI
  if (part.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("too many values for one MPI_Gatherv");
  }
  auto count = static_cast<int>(part.size());
  auto counts = std::vector<int>(ranks.First() ? ranks.size : 0);
  MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  auto offsets = std::vector<int>(counts.size(), 0);
  auto total = std::size_t{0};
  for (auto r = std::size_t{0}; r < counts.size(); ++r) {
    if (total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::length_error("too many values for one MPI_Gatherv");
    }
    offsets[r] = static_cast<int>(total);
    total += static_cast<std::size_t>(counts[r]);
  }
  auto whole = std::vector<double>(total);
  MPI_Gatherv(part.data(), count, MPI_DOUBLE, whole.data(), counts.data(),
              offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return whole;
#else
  throw std::logic_error("several ranks without MPI");
#endif
}

}  // namespace sl3dpp::app
