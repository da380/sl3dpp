// The distributed radial solves over MPI ranks (Distributed.hpp), built
// with SL3DPP_WITH_MPI and run under mpiexec. One segment of the prem8 mesh
// per rank: every spheroidal and toroidal degree to lmax 32 is solved
// collectively, the ranks' parts (each its own interiors, rank 0 the cut
// nodes) are summed into the whole solution, and that is held to the
// serial substructured solve's gate: residuals within a few times the
// direct solve's, and agreement as far as the systems determine their
// solutions. Each rank's errors are reduced to rank 0, which alone prints.

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
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

double RelativeDifference(std::span<const double> got,
                          std::span<const double> want) {
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    scale = std::max(scale, std::abs(want[i]));
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  return diff / scale;
}

double MaxOverRanks(double value) {
  auto out = 0.0;
  MPI_Allreduce(&value, &out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return out;
}

Material Prem8(int lmax) {
  const auto deck =
      Deck::Read(
          std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
          DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
          .value();
  static const auto model = EarthModel::FromDeck(deck).value();
  return Material::Create(model,
                          RadialMesh::Create(model, {.lmax = lmax}).value())
      .value();
}

/// The global nodes whose dofs this rank's part of a solution holds: the
/// nodes of its segment's elements but the cut nodes, and on rank 0 every
/// cut node above the degree's first element (a separator). A cut at or
/// below it is no separator, and its node, when it has dofs, lies in the
/// segment above.
std::vector<int> NodesOf(const RadialMesh& mesh, const RadialShare& share,
                         int firstElement) {
  auto ranks = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto lo = rank == 0 ? 0 : share.cuts[rank - 1];
  const auto hi = rank + 1 == ranks ? mesh.ElementCount() : share.cuts[rank];
  auto separator = [&](int cut) { return cut > firstElement; };
  auto out = std::vector<int>{};
  for (auto e = lo; e < hi; ++e) {
    for (auto i = 0; i < mesh.NGll(); ++i) {
      const auto node = mesh.GlobalIndex(e, i);
      if (!out.empty() && out.back() == node) continue;
      if (rank > 0 && node == mesh.GlobalIndex(lo, 0) && separator(lo)) {
        continue;
      }
      if (rank + 1 < ranks && node == mesh.GlobalIndex(hi, 0)) continue;
      out.push_back(node);
    }
  }
  if (rank == 0) {
    for (const auto cut : share.cuts) {
      if (separator(cut)) out.push_back(mesh.GlobalIndex(cut, 0));
    }
  }
  return out;
}

/// The whole solution from the ranks' parts: each keeps the dofs it
/// holds, and the sum over the ranks fills the rest.
std::vector<double> Gather(std::span<const double> x, std::span<const int> dofs,
                           int n, int nrhs) {
  auto out = std::vector<double>(x.size(), 0.0);
  for (auto k = 0; k < nrhs; ++k) {
    for (const auto d : dofs) {
      const auto at = static_cast<std::size_t>(k) * n + d;
      out[at] = x[at];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, out.data(), static_cast<int>(out.size()),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return out;
}

}  // namespace

TEST(Distributed, EverySpheroidalDegreeMatchesTheSerialSolve) {
  constexpr auto lmax = 32;
  const auto material = Prem8(lmax);
  const auto& mesh = material.Mesh();
  const auto share = ShareMesh(mesh, MPI_COMM_WORLD);
  auto worstResidual = 0.0, worstLow = 0.0, worstHigh = 0.0;
  for (auto l = 0; l <= lmax; ++l) {
    const DegreeOperator direct(material, l);
    const DegreeOperator distributed(material, l,
                                     {.cuts = share.cuts,
                                      .segments = share.segments,
                                      .combine = share.combine});
    auto b = direct.Load(LoadPart::Both);
    const auto tide = direct.Tide(l == 0 ? 2 : -1);
    b.insert(b.end(), tide.begin(), tide.end());
    constexpr auto nrhs = 2;
    const auto want = direct.Solve(b, nrhs);
    auto dofs = std::vector<int>{};
    for (const auto node : NodesOf(mesh, share, direct.FirstElement())) {
      for (const auto c : {Component::U, Component::V, Component::Phi}) {
        if (const auto d = direct.Dof(node, c); d >= 0) dofs.push_back(d);
      }
    }
    const auto n = direct.DofCount();
    const auto got = Gather(distributed.Solve(b, nrhs), dofs, n, nrhs);
    const auto residualWant = RelativeDifference(direct.Apply(want, nrhs), b);
    const auto residualGot = RelativeDifference(direct.Apply(got, nrhs), b);
    worstResidual = std::max(worstResidual, residualGot / residualWant);
    auto& worst = l < 10 ? worstLow : worstHigh;
    worst = std::max(worst, RelativeDifference(got, want));
  }
  EXPECT_LT(MaxOverRanks(worstResidual), 3.0);
  EXPECT_LT(MaxOverRanks(worstLow), 1e-9);
  EXPECT_LT(MaxOverRanks(worstHigh), 1e-11);
}

TEST(Distributed, EveryToroidalDegreeMatchesTheSerialSolve) {
  constexpr auto lmax = 32;
  const auto material = Prem8(lmax);
  const auto& mesh = material.Mesh();
  const auto share = ShareMesh(mesh, MPI_COMM_WORLD);
  auto worstResidual = 0.0, worst = 0.0;
  for (auto l = 1; l <= lmax; ++l) {
    const ToroidalOperator direct(material, l);
    const ToroidalOperator distributed(material, l,
                                       {.cuts = share.cuts,
                                        .segments = share.segments,
                                        .combine = share.combine});
    auto x = std::vector<double>(direct.DofCount());
    for (auto i = std::size_t{0}; i < x.size(); ++i) {
      x[i] = std::cos(0.37 * i + l);
    }
    const auto b = direct.Apply(x);
    const auto want = direct.Solve(b);
    auto dofs = std::vector<int>{};
    for (const auto node : NodesOf(mesh, share, direct.FirstElement())) {
      if (const auto d = direct.Dof(node); d >= 0) dofs.push_back(d);
    }
    const auto got = Gather(distributed.Solve(b), dofs, direct.DofCount(), 1);
    const auto residualWant = RelativeDifference(direct.Apply(want), b);
    const auto residualGot = RelativeDifference(direct.Apply(got), b);
    worstResidual =
        std::max(worstResidual, residualGot / (residualWant + 1e-300));
    worst = std::max(worst, RelativeDifference(got, want));
  }
  EXPECT_LT(MaxOverRanks(worstResidual), 3.0);
  EXPECT_LT(MaxOverRanks(worst), 1e-9);
}

TEST(Distributed, TheShareIsTheSameOnEveryRank) {
  const auto material = Prem8(16);
  const auto share = ShareMesh(material.Mesh(), MPI_COMM_WORLD);
  auto ranks = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  EXPECT_EQ(static_cast<int>(share.cuts.size()), ranks - 1);
  EXPECT_EQ(share.segments, std::vector<int>{rank});
  auto cuts = share.cuts;
  MPI_Bcast(cuts.data(), static_cast<int>(cuts.size()), MPI_INT, 0,
            MPI_COMM_WORLD);
  EXPECT_EQ(cuts, share.cuts);
  auto values = std::vector<double>{1.0, static_cast<double>(rank)};
  share.combine(values);
  EXPECT_EQ(values[0], static_cast<double>(ranks));
  EXPECT_EQ(values[1], ranks * (ranks - 1) / 2.0);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
  }
  auto failed = RUN_ALL_TESTS() != 0 ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return failed;
}
