#include "sl3dpp/Distributed.hpp"

#ifdef SL3DPP_HAVE_MPI

#include <format>
#include <limits>
#include <stdexcept>

namespace sl3dpp {

std::function<void(std::span<double>)> SumOver(MPI_Comm comm) {
  return [comm](std::span<double> values) {
    if (values.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::length_error("too many values for one MPI_Allreduce");
    }
    MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()),
                  MPI_DOUBLE, MPI_SUM, comm);
  };
}

RadialShare ShareMesh(const RadialMesh& mesh, MPI_Comm comm,
                      std::span<const double> weights) {
  auto ranks = 1, rank = 0;
  MPI_Comm_size(comm, &ranks);
  MPI_Comm_rank(comm, &rank);
  auto share = RadialShare{};
  share.cuts = mesh.BalancedCuts(ranks, weights);
  if (static_cast<int>(share.cuts.size()) != ranks - 1) {
    throw std::invalid_argument(std::format(
        "the mesh's layers leave {} element boundaries for {} ranks",
        share.cuts.size(), ranks));
  }
  share.segments = {rank};
  share.combine = SumOver(comm);
  return share;
}

}  // namespace sl3dpp

#endif  // SL3DPP_HAVE_MPI
