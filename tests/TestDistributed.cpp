// The distributed radial solves over MPI ranks (Distributed.hpp), built
// with SL3DPP_WITH_MPI and run under mpiexec. One segment of the prem8 mesh
// per rank: every spheroidal and toroidal degree to lmax 32 is solved
// collectively, the ranks' parts (each its own interiors, rank 0 the cut
// nodes) are summed into the whole solution, and that is held to the
// serial substructured solve's gate: residuals within a few times the
// direct solve's, and agreement as far as the systems determine their
// solutions. Then the 3D step and the coupled evolution, distributed
// (each rank its segment's memory nodes, the cuts balanced on the
// transform work), and the 3D adjoint's kernels, against the one-process
// run on the same cuts that every rank also makes (TestRadialRanks makes the
// same comparisons with threads for ranks). Each rank's errors are reduced to
// rank 0, which alone prints.

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <map>
#include <span>
#include <vector>

#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/Distributed.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/ToroidalOperator.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

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

/// prem8 at lmax with a Maxwell mantle of uniform viscosity.
struct Body {
  EarthModel model;
  Material material;
  Rheology rheology;

  static Body Make(int lmax) {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
    auto material = Material::Create(model, mesh).value();
    const auto& scales = model.ModelScales();
    auto viscosity = std::map<std::size_t, RadialValue>{};
    const auto b = model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    return Body{std::move(model), std::move(material), std::move(rheology)};
  }
};

double Peaked(double, double theta, double phi) {
  return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
}

template <class A, class B>
double Relative(const A& got, const B& want) {
  auto scale = 0.0, diff = 0.0;
  for (auto i = std::size_t{0}; i < want.size(); ++i) {
    scale = std::max(scale, std::abs(want[i]));
    diff = std::max(diff, std::abs(got[i] - want[i]));
  }
  return scale > 0.0 ? diff / scale : diff;
}

/// The largest relative difference of this rank's memory tensors from the
/// one process's at the same nodes.
double SliceDifference(const Viscoelastic3D& rank,
                       const Viscoelastic3D::State& got,
                       const Viscoelastic3D& whole,
                       const Viscoelastic3D::State& want) {
  const auto& part = rank.Coupling();
  const auto& all = whole.Coupling();
  if (part.NodeCount() == 0) return 0.0;
  auto first = 0;
  while (all.ElementOfNode(first) != part.ElementOfNode(0)) ++first;
  const auto offset = all.FieldOffset(first);
  auto worst = 0.0;
  for (auto k = std::size_t{0}; k < want.m.size(); ++k) {
    auto compare = [&](const auto& g, const auto& w) {
      auto scale = 0.0, diff = 0.0;
      for (const auto& v : w) scale = std::max(scale, std::abs(v));
      for (auto i = std::size_t{0}; i < g.size(); ++i) {
        diff = std::max(diff, std::abs(g[i] - w[offset + i]));
      }
      worst = std::max(worst, scale > 0.0 ? diff / scale : diff);
    };
    compare(got.m[k].c[0], want.m[k].c[0]);
    compare(got.m[k].c[1], want.m[k].c[1]);
    compare(got.m[k].r, want.m[k].r);
  }
  return worst;
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
                                      .combine = share.collectives.sum});
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
                                        .combine = share.collectives.sum});
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
  share.collectives.sum(values);
  EXPECT_EQ(values[0], static_cast<double>(ranks));
  EXPECT_EQ(values[1], ranks * (ranks - 1) / 2.0);
}

TEST(Distributed, TheLateralStepMatchesTheOneProcessStep) {
  // The transforms cut with depth (lmax 16, a loose truncation) and the
  // cuts balanced on their work: two lateral steps, each rank's surface
  // response and memory against the one process's.
  constexpr auto lmax = 16;
  constexpr auto eps = 1e-2;
  const auto degrees = std::vector<int>{8, 12};
  const auto body = Body::Make(lmax);
  const auto share = ShareMesh(
      body.material.Mesh(), MPI_COMM_WORLD,
      GridCoupling::TransformWeights(
          body.rheology, lmax, {.eps = eps, .transform_degrees = degrees}));
  const auto tau = body.rheology.MinRelaxationTime();
  auto noise = [](std::size_t k) {
    const auto x = std::sin(12.9898 * static_cast<double>(k + 1)) * 43758.5453;
    return x - std::floor(x) - 0.5;
  };
  const auto load = [&](int l, double) {
    auto columns = std::vector<std::complex<double>>(l + 1);
    for (auto m = 0; m <= l; ++m) {
      columns[m] = m == 0 ? std::complex(noise(97 * l), 0.0)
                          : std::complex(noise(97 * l + m), noise(55 * l + m));
    }
    return columns;
  };
  const auto options = Viscoelastic3D::Options{.eps = eps,
                                               .parallel = false,
                                               .max_iterations = 100,
                                               .rtol = 1e-12,
                                               .radial_cuts = share.cuts,
                                               .transform_degrees = degrees};
  struct Result {
    Viscoelastic3D::State state;
    std::map<int, std::vector<std::complex<double>>> U;
    std::vector<int> iterations;
  };
  auto twoSteps = [&](const Viscoelastic3D& op) {
    auto result = Result{op.ZeroState()};
    Viscoelastic3D::Solutions solutions;
    for (auto i = 0; i < 2; ++i) {
      auto step =
          op.Step(result.state, tau, Scheme::ExponentialTrapezoid, load);
      EXPECT_TRUE(step.info.converged);
      result.iterations.push_back(step.info.iterations);
      result.state = std::move(step.state);
      solutions = std::move(step.solutions);
    }
    result.U = op.Surface(solutions, Component::U);
    return result;
  };
  const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
  const auto want = twoSteps(whole);
  auto mine = options;
  mine.segments = share.segments;
  mine.collectives = share.collectives;
  const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
  const auto got = twoSteps(op);
  auto surface = 0.0;
  for (const auto& [l, values] : want.U) {
    surface = std::max(surface, Relative(got.U.at(l), values));
  }
  EXPECT_EQ(got.iterations, want.iterations);
  const auto worstSurface = MaxOverRanks(surface);
  const auto worstMemory =
      MaxOverRanks(SliceDifference(op, got.state, whole, want.state));
  // Round-off of another summation order (MPI's): 5e-16 and 2.4e-14 on
  // two ranks, 1.8e-13 and 1.1e-13 on four, when written.
  EXPECT_LT(worstSurface, 1e-11);
  EXPECT_LT(worstMemory, 1e-11);
}

TEST(Distributed, TheEvolutionMatchesTheOneProcessRun) {
  // The coupled evolution, every rank solving the sea level equation
  // (rank 0's loads shared): frozen shorelines under a melting load and
  // migrating ones under a thinning ice sheet.
  constexpr auto lmax = 8;
  const auto body = Body::Make(lmax);
  const auto share =
      ShareMesh(body.material.Mesh(), MPI_COMM_WORLD,
                GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  SeaLevelSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
                        SeaLevelParameters{});
  const auto length = solver.SolverScales().Length();
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto ice = toNd(fingerprint.IceThickness());
  const auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, 3.0 * kyr, 6.0 * kyr, 10.0 * kyr};
  auto melt = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < melt.size(); ++i) {
    melt[i] =
        -0.1 * solver.IceDensity() * (1.0 - state.oceanFunction[i]) * ice[i];
  }
  const auto iceLoad = [&](double t) {
    auto out = melt;
    for (auto& v : out) v *= t / times.back();
    return out;
  };
  const auto thickness = [&](double t) {
    auto out = ice;
    for (auto& v : out) v *= 1.0 - 0.3 * t / times.back();
    return out;
  };
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .radial_cuts = share.cuts};
  const auto evolutionOptions = SeaLevelEvolution3D::Options{
      .rtol = 1e-12, .max_coupled_iterations = 100, .coupled_rtol = 1e-11};
  for (const auto migrating : {false, true}) {
    auto run = [&](SeaLevelEvolution3D& evolution) {
      return migrating ? evolution.RunMigrating(times, thickness)
                       : evolution.Run(times, iceLoad);
    };
    const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
    SeaLevelEvolution3D one(whole, solver, state, evolutionOptions);
    const auto want = run(one);
    auto mine = options;
    mine.segments = share.segments;
    mine.collectives = share.collectives;
    const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
    SeaLevelEvolution3D evolution(op, solver, state, evolutionOptions);
    const auto got = run(evolution);
    ASSERT_EQ(got.size(), want.size());
    auto worst = 0.0;
    for (auto n = std::size_t{0}; n < want.size(); ++n) {
      EXPECT_TRUE(got[n].converged) << "node " << n;
      worst = std::max(worst,
                       Relative(got[n].seaLevelChange, want[n].seaLevelChange));
      if (migrating) {
        EXPECT_EQ(got[n].oceanFunction, want[n].oceanFunction) << n;
      }
    }
    // 6e-16 to 5e-15 on two to four ranks when written
    EXPECT_LT(MaxOverRanks(worst), 1e-12)
        << (migrating ? "migrating" : "frozen");
  }
}

TEST(Distributed, TheAdjointMatchesTheOneProcessSweep) {
  // The 3D continuous adjoint, each rank sweeping its own memory nodes
  // back over its own forward trajectory: the ice kernels on every rank,
  // and each rank's slice of the log-viscosity kernel, frozen shorelines
  // and about a migrating run.
  constexpr auto lmax = 8;
  const auto body = Body::Make(lmax);
  const auto share =
      ShareMesh(body.material.Mesh(), MPI_COMM_WORLD,
                GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  SeaLevelSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
                        SeaLevelParameters{});
  const auto length = solver.SolverScales().Length();
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto ice = toNd(fingerprint.IceThickness());
  const auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  const auto times =
      std::vector<double>{0.0, 2.5 * kyr, 5.0 * kyr, 7.5 * kyr, 10.0 * kyr};
  auto melt = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < melt.size(); ++i) {
    melt[i] =
        -0.1 * solver.IceDensity() * (1.0 - state.oceanFunction[i]) * ice[i];
  }
  const auto iceLoad = [&](double t) {
    auto out = melt;
    for (auto& v : out) v *= t / times.back();
    return out;
  };
  const auto thickness = [&](double t) {
    auto out = ice;
    for (auto& v : out) v *= 1.0 - 0.3 * t / times.back();
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual =
      solver.Grid().FromFunction([](double theta, double phi) {
        return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
      });
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .radial_cuts = share.cuts};
  const auto evolutionOptions =
      SeaLevelEvolution3D::Options{.rtol = 1e-12,
                                   .max_coupled_iterations = 100,
                                   .coupled_rtol = 1e-11,
                                   .record_stride = 1};
  const auto adjointOptions = AdjointSolver3D::Options{
      .rtol = 1e-12, .max_coupled_iterations = 100, .coupled_rtol = 1e-11};
  for (const auto migrating : {false, true}) {
    auto sweep = [&](const Viscoelastic3D& op) {
      SeaLevelEvolution3D evolution(op, solver, state, evolutionOptions);
      const auto snapshots = migrating
                                 ? evolution.RunMigrating(times, thickness)
                                 : evolution.Run(times, iceLoad);
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      AdjointSolver3D adjoint(evolution, adjointOptions);
      return adjoint.IceHistoryKernels(
          times, measurements, &trajectory,
          migrating ? OceanFunctionsOf(snapshots)
                    : std::vector<std::vector<double>>{});
    };
    const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
    const auto want = sweep(whole);
    auto mine = options;
    mine.segments = share.segments;
    mine.collectives = share.collectives;
    const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
    const auto got = sweep(op);
    auto worstIce = 0.0;
    for (auto n = std::size_t{0}; n < want.ice.loadDuals.size(); ++n) {
      worstIce = std::max(
          worstIce, Relative(got.ice.loadDuals[n], want.ice.loadDuals[n]));
    }
    const auto& part = op.Coupling();
    const auto& all = whole.Coupling();
    auto worstViscosity = 0.0;
    if (part.NodeCount() > 0) {
      auto first = 0;
      while (all.ElementOfNode(first) != part.ElementOfNode(0)) ++first;
      const auto& kernel = got.logViscosityKernel3D[0];
      worstViscosity =
          Relative(kernel, std::span(want.logViscosityKernel3D[0])
                               .subspan(all.FieldOffset(first), kernel.size()));
    }
    // 3e-15 and 4e-14 with threads for ranks when written
    const auto label = migrating ? "migrating" : "frozen";
    EXPECT_LT(MaxOverRanks(worstIce), 1e-11) << label;
    EXPECT_LT(MaxOverRanks(worstViscosity), 1e-11) << label;
  }
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
