// The distributed radial solves over MPI ranks (Distributed.hpp), built
// with SL3DPP_WITH_MPI and run under mpiexec. One segment of the prem8 mesh
// per rank: every spheroidal and toroidal degree to lmax 32 is solved
// collectively, the ranks' parts (each its own interiors, rank 0 the cut
// nodes) are summed into the whole solution, and that is held to the
// serial substructured solve's criteria: residuals within a few times the
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
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/TangentLinear3D.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/Distributed.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/InitialSeaLevel.hpp"
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
  // the within-step passes may differ by one under MPI's summation order
  // when a residual lands at the tolerance
  ASSERT_EQ(got.iterations.size(), want.iterations.size());
  for (auto i = std::size_t{0}; i < got.iterations.size(); ++i) {
    EXPECT_LE(std::abs(got.iterations[i] - want.iterations[i]), 1) << i;
  }
  const auto worstSurface = MaxOverRanks(surface);
  const auto worstMemory =
      MaxOverRanks(SliceDifference(op, got.state, whole, want.state));
  // Round-off of another summation order (MPI's): measured 5e-16 and
  // 2.4e-14 on two ranks, 1.8e-13 and 1.1e-13 on four; a pass more at the
  // tolerance of 1e-12 stays below the bound.
  EXPECT_LT(worstSurface, 1e-10);
  EXPECT_LT(worstMemory, 1e-10);
}

TEST(Distributed, TheShortestRelaxationMatchesTheOneProcess) {
  // Viscoelastic3D::ShortestRelaxation over the ranks under a lateral
  // viscosity: the one process's, on every rank. Each rank samples its
  // own nodes as the one process samples them, and the collectives' max
  // is exact: bitwise.
  constexpr auto lmax = 8;
  const auto body = Body::Make(lmax);
  const auto share = ShareMesh(body.material.Mesh(), MPI_COMM_WORLD);
  const auto options =
      Viscoelastic3D::Options{.parallel = false, .radial_cuts = share.cuts};
  const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
  auto mine = options;
  mine.segments = share.segments;
  mine.collectives = share.collectives;
  const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
  const auto want = whole.ShortestRelaxation();
  const auto got = op.ShortestRelaxation();
  EXPECT_TRUE(want.lateral);
  EXPECT_EQ(got.time, want.time);
  EXPECT_EQ(got.branch, want.branch);
  EXPECT_EQ(got.element, want.element);
  EXPECT_EQ(got.layer, want.layer);
  EXPECT_EQ(got.radius, want.radius);
  EXPECT_EQ(got.lateral, want.lateral);
  EXPECT_EQ(got.colatitude, want.colatitude);
  EXPECT_EQ(got.longitude, want.longitude);
}

TEST(Distributed, TheEvolutionMatchesTheOneProcessRun) {
  // The coupled evolution, every rank solving the sea level equation
  // (rank 0's loads shared): frozen shorelines under a melting load and
  // migrating ones under a thinning ice sheet, alone and with a tide
  // beside it, whose coefficients the steps saw are rank 0's on every
  // rank.
  constexpr auto lmax = 8;
  const auto body = Body::Make(lmax);
  const auto share =
      ShareMesh(body.material.Mesh(), MPI_COMM_WORLD,
                GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  GRDSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
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
  const auto directLoad = [&](double t) {
    auto out = melt;
    for (auto& v : out) v *= t / times.back();
    return out;
  };
  const auto thickness = [&](double t) {
    auto out = ice;
    for (auto& v : out) v *= 1.0 - 0.3 * t / times.back();
    return out;
  };
  const auto tide = [&](double t) {
    auto psi = solver.Grid().ZeroCoefficients();
    const auto a = 40.0 / solver.SolverScales().Factor(PotentialDimensions);
    psi[SurfaceGrid::Index(2, 1)] = a * std::complex<double>(0.5, -0.2) *
                                    (t / times.back());
    psi[SurfaceGrid::Index(3, 3)] = a * std::complex<double>(0.1, 0.3);
    return psi;
  };
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .radial_cuts = share.cuts};
  const auto evolutionOptions =
      Evolution3D::Options{.rtol = 1e-12,
                           .max_coupled_iterations = 100,
                           .coupled_rtol = 1e-11,
                           .record_stride = 1};
  for (const auto [migrating, tidal] :
       {std::pair{false, false}, std::pair{true, false}, std::pair{true, true}}) {
    auto run = [&, migrating, tidal](Evolution3D& evolution) {
      if (tidal) {
        return evolution.RunMigrating(times, thickness,
                                      Evolution3D::Forcing{.tide = tide});
      }
      return migrating ? evolution.RunMigrating(times, thickness)
                       : evolution.Run(times, directLoad);
    };
    const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
    Evolution3D one(whole, solver, state, evolutionOptions);
    const auto want = run(one);
    auto mine = options;
    mine.segments = share.segments;
    mine.collectives = share.collectives;
    const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
    Evolution3D evolution(op, solver, state, evolutionOptions);
    const auto got = run(evolution);
    ASSERT_EQ(got.size(), want.size());
    auto worst = 0.0;
    for (auto n = std::size_t{0}; n < want.size(); ++n) {
      EXPECT_TRUE(got[n].converged) << "node " << n;
      worst = std::max(worst,
                       Relative(got[n].relativeSeaLevelChange, want[n].relativeSeaLevelChange));
      if (migrating) {
        // flotation's 0/1: rounding changes it only at a cell on the threshold
        EXPECT_EQ(got[n].oceanFunction, want[n].oceanFunction) << n;
      }
      // the coefficients the steps saw: rank 0's on every rank, copied
      // from it, bitwise
      auto first = got[n].tideCoefficients;
      MPI_Bcast(first.data(), static_cast<int>(2 * first.size()), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
      EXPECT_EQ(got[n].tideCoefficients, first) << n;
    }
    // measured 6e-16 to 5e-15 on two to four ranks
    EXPECT_LT(MaxOverRanks(worst), 1e-12)
        << (migrating ? "migrating" : "frozen") << (tidal ? ", tide" : "");
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
  GRDSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
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
  const auto directLoad = [&](double t) {
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
  measurements[times.size() - 1].relativeSeaLevelDual =
      solver.Grid().FromFunction([](double theta, double phi) {
        return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
      });
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .radial_cuts = share.cuts};
  const auto evolutionOptions =
      Evolution3D::Options{.rtol = 1e-12,
                                   .max_coupled_iterations = 100,
                                   .coupled_rtol = 1e-11,
                                   .record_stride = 1};
  const auto adjointOptions =
      AdjointSolver3D::Options{.method = AdjointMethod::Continuous,
                               .rtol = 1e-12,
                               .max_coupled_iterations = 100,
                               .coupled_rtol = 1e-11};
  for (const auto migrating : {false, true}) {
    auto sweep = [&](const Viscoelastic3D& op) {
      Evolution3D evolution(op, solver, state, evolutionOptions);
      const auto snapshots = migrating
                                 ? evolution.RunMigrating(times, thickness)
                                 : evolution.Run(times, directLoad);
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      AdjointSolver3D adjoint(evolution, adjointOptions);
      return adjoint.FirstOrderKernels(
          times, measurements, &trajectory,
          migrating ? OceanFunctions::Of(snapshots) : OceanFunctions{});
    };
    const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
    const auto want = sweep(whole);
    auto mine = options;
    mine.segments = share.segments;
    mine.collectives = share.collectives;
    const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
    const auto got = sweep(op);
    auto worstIce = 0.0;
    for (auto n = std::size_t{0}; n < want.ice.directLoadKernels.size(); ++n) {
      worstIce = std::max(
          worstIce, Relative(got.ice.directLoadKernels[n], want.ice.directLoadKernels[n]));
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
    // measured 6e-15 and 5.5e-14 with threads for ranks, 1.4e-13 and
    // 1.8e-12 on four ranks
    const auto label = migrating ? "migrating" : "frozen";
    EXPECT_LT(MaxOverRanks(worstIce), 1e-9) << label;
    EXPECT_LT(MaxOverRanks(worstViscosity), 1e-9) << label;
  }
}

namespace {

/// The largest relative difference of a rank's field over its memory
/// nodes (its coupling's layout) from the one process's at the same nodes.
double FieldSliceDifference(const GridCoupling& part,
                            std::span<const double> got,
                            const GridCoupling& all,
                            std::span<const double> want) {
  if (part.NodeCount() == 0) return 0.0;
  auto first = 0;
  while (all.ElementOfNode(first) != part.ElementOfNode(0)) ++first;
  return Relative(got, want.subspan(all.FieldOffset(first), got.size()));
}

/// A rank's slice of a field over the one process's grid values.
std::vector<double> Slice(const GridCoupling& part, const GridCoupling& all,
                          std::span<const double> field) {
  if (part.NodeCount() == 0) return {};
  auto first = 0;
  while (all.ElementOfNode(first) != part.ElementOfNode(0)) ++first;
  const auto at = all.FieldOffset(first);
  return std::vector<double>(field.begin() + static_cast<std::ptrdiff_t>(at),
                             field.begin() + static_cast<std::ptrdiff_t>(
                                                 at + part.GridValueCount()));
}

/// A radial profile times an angular function over a coupling's grid
/// values, in units of a nodal modulus.
std::vector<double> LateralModulus(const GridCoupling& coupling,
                                   const Material& material,
                                   std::span<const double> modulus,
                                   const std::function<double(double)>& radial,
                                   const std::function<double(double, double)>& angular) {
  const auto ngll = material.Mesh().NGll();
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto full = static_cast<std::size_t>(coupling.ElementOfNode(rn)) * ngll +
                      coupling.NodeInElementOf(rn);
    const auto r = coupling.RadiusOfNode(rn);
    const auto values = coupling.GridOf(rn).FromFunction(
        [&](double theta, double phi) {
          return modulus[full] * radial(r) * angular(theta, phi);
        });
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x) out[at + x] = values[x];
  }
  return out;
}

double Cap(double theta, double phi, double colat0, double lon0, double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

}  // namespace

TEST(Distributed, DerivativesMatchTheOneProcess) {
  // The derivative machinery under mpiexec against the one process on
  // the same cuts (TestRadialRanks makes the same comparison with threads
  // for ranks): the 3D tangent-linear runs (both methods) along loads,
  // a lateral log viscosity and lateral moduli; the first-order sweeps
  // with the material kernels, discrete (the transposed fixed point's
  // partial contractions summed, the surface duals on their holding
  // rank) and continuous; and the second-order sweeps of both methods
  // along a direction in all of those, the kernels' derivatives included
  // — the ice views on every rank, each rank's slices of the 3D kernels.
  constexpr auto lmax = 8;
  auto body = Body::Make(lmax);
  {
    // the 1e22 mantle of the threads-for-ranks gate
    const auto& scales = body.model.ModelScales();
    auto viscosity = std::map<std::size_t, RadialValue>{};
    const auto b = body.model.DimensionlessView().Boundaries();
    for (auto i = std::size_t{0}; i < body.model.LayerCount(); ++i) {
      if (!body.model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i, 1e22 / scales.Factor(ViscosityDimensions));
      }
    }
    body.rheology = Rheology::Maxwell(body.material, viscosity).value();
  }
  const auto share =
      ShareMesh(body.material.Mesh(), MPI_COMM_WORLD,
                GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  auto solvers = std::vector<GRDSolver>{};
  for (auto rank = 0; rank < 2; ++rank) {
    solvers.emplace_back(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
                         SeaLevelParameters{});
  }
  const auto length = solvers[0].SolverScales().Length();
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto ice = toNd(fingerprint.IceThickness());
  const auto state = solvers[0].MakeState(ice, toNd(fingerprint.SeaLevel()));
  const auto kyr = 1000.0 * 3.15576e7 / solvers[0].SolverScales().Time();
  const auto times =
      std::vector<double>{0.0, 2.5 * kyr, 5.0 * kyr, 7.5 * kyr, 10.0 * kyr};
  auto melt = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < melt.size(); ++i) {
    melt[i] = -0.1 * solvers[0].IceDensity() * (1.0 - state.oceanFunction[i]) *
              ice[i];
  }
  const auto directLoad = [&](double t) {
    auto out = melt;
    for (auto& v : out) v *= t / times.back();
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].relativeSeaLevelDual =
      solvers[0].Grid().FromFunction([](double theta, double phi) {
        return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
      });
  measurements[2].displacementDual = measurements[times.size() - 1].relativeSeaLevelDual;
  // the direction: a thickness history, a lateral log viscosity, lateral
  // moduli — over the one process's grid values, sliced per rank
  auto deltaIce = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    auto field = solvers[0].Grid().FromFunction([n](double theta, double phi) {
      return 20.0 * (0.3 + 0.1 * n) * Cap(theta, phi, 1.0 + 0.2 * n, 0.5, 0.6);
    });
    for (auto& v : field) v /= length;
    deltaIce.push_back(std::move(field));
  }
  auto pU = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  const auto [k0, m0] = body.material.Moduli().KappaMu();
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .memory_points = MemoryPointSet::Gauss,
      .radial_cuts = share.cuts};
  const auto evolutionOptions =
      // the exponential trapezoid stated: the continuous second order is built
      // for the one-stage schemes
      Evolution3D::Options{.scheme = Scheme::ExponentialTrapezoid,
                           .rotation = RotationalFeedback::Full,
                                   .rtol = 1e-12,
                                   .max_coupled_iterations = 100,
                                   .coupled_rtol = 1e-11,
                                   .record_stride = 1};
  const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
  const auto& all = whole.Coupling();
  auto wholeDirection = TangentPerturbation3D{};
  wholeDirection.iceThickness = deltaIce;
  auto logField = std::vector<double>(all.GridValueCount(), 0.0);
  for (auto rn = 0; rn < all.NodeCount(); ++rn) {
    const auto values = all.GridOf(rn).FromFunction(pU);
    const auto at = all.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x) logField[at + x] = values[x];
  }
  wholeDirection.logViscosity = logField;
  wholeDirection.material.kappa = LateralModulus(
      all, body.material, k0, [](double r) { return 0.05 * (1.0 - 0.5 * r); },
      [](double theta, double phi) {
        return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
      });
  wholeDirection.material.mu = LateralModulus(
      all, body.material, m0, [](double r) { return 0.1 * (0.5 + r * r); },
      [](double theta, double phi) {
        return Cap(theta, phi, 1.9, 4.4, 0.7) - 0.5 * Cap(theta, phi, 0.9, 1.1, 0.6);
      });
  auto directionFor = [&](const GridCoupling& part) {
    auto d = wholeDirection;
    d.logViscosity = Slice(part, all, wholeDirection.logViscosity);
    d.material.kappa = Slice(part, all, wholeDirection.material.kappa);
    d.material.mu = Slice(part, all, wholeDirection.material.mu);
    return d;
  };

  struct Result {
    std::map<AdjointMethod, std::vector<Evolution3D::Snapshot>> tangent;
    std::map<AdjointMethod, Kernels3D> kernels;
    std::map<AdjointMethod, std::pair<Kernels3D, KernelDerivatives3D>> second;
  };
  auto run = [&](const Viscoelastic3D& op, const GRDSolver& solver) {
    Evolution3D evolution(op, solver, state, evolutionOptions);
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, directLoad));
    const auto direction = directionFor(op.Coupling());
    const auto itau = TangentLinearSolver3D::ItauOfLogViscosity(op, direction.logViscosity);
    Result out;
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver3D tangent(evolution, {.method = method,
                                                .rtol = 1e-12,
                                                .max_coupled_iterations = 100,
                                                .coupled_rtol = 1e-11,
                                                .record = true});
      out.tangent[method] = tangent.Run(times, direction, &trajectory);
      AdjointSolver3D adjoint(evolution, {.method = method,
                                          .rtol = 1e-12,
                                          .max_coupled_iterations = 100,
                                          .coupled_rtol = 1e-11,
                                          .material_kernels = true});
      out.kernels[method] = adjoint.FirstOrderKernels(times, measurements, &trajectory);
      out.second[method] = adjoint.SecondOrderKernels(
          times, measurements, out.tangent[method], itau, trajectory, {},
          direction.material);
    }
    return out;
  };
  const auto want = run(whole, solvers[1]);

  auto mine = options;
  mine.segments = share.segments;
  mine.collectives = share.collectives;
  const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
  const auto got = run(op, solvers[0]);

  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const auto label = method == AdjointMethod::Discrete ? "discrete" : "continuous";
    auto worstTangent = 0.0, worstIce = 0.0, worstField = 0.0, worstSecond = 0.0;
    {
      const auto& part = op.Coupling();
      const auto& r = got;
      const auto& w = want;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        worstTangent = std::max(
            worstTangent, Relative(r.tangent.at(method)[n].relativeSeaLevelChange,
                                   w.tangent.at(method)[n].relativeSeaLevelChange));
        worstIce = std::max(worstIce, Relative(r.kernels.at(method).ice.directLoadKernels[n],
                                               w.kernels.at(method).ice.directLoadKernels[n]));
        worstSecond = std::max(
            worstSecond, Relative(r.second.at(method).second.ice.thickness[n],
                                  w.second.at(method).second.ice.thickness[n]));
      }
      const auto& k = r.kernels.at(method);
      const auto& wk = w.kernels.at(method);
      worstField = std::max(
          {worstField,
           FieldSliceDifference(part, k.logViscosityKernel3D[0], all, wk.logViscosityKernel3D[0]),
           FieldSliceDifference(part, k.kappaKernel3D, all, wk.kappaKernel3D),
           FieldSliceDifference(part, k.muKernel3D, all, wk.muKernel3D)});
      const auto& d = r.second.at(method).second;
      const auto& wd = w.second.at(method).second;
      worstSecond = std::max(
          {worstSecond,
           FieldSliceDifference(part, d.itauKernel3D[0], all, wd.itauKernel3D[0]),
           FieldSliceDifference(part, d.kappaKernel3D, all, wd.kappaKernel3D),
           FieldSliceDifference(part, d.muKernel3D, all, wd.muKernel3D)});
    }
    // measured 7e-14 at most in the tangent and the ice views, 1.5e-12 in
    // the 3D kernels, whose sweeps close at the coupled tolerance of 1e-11
    EXPECT_LT(MaxOverRanks(worstTangent), 1e-11) << label;
    EXPECT_LT(MaxOverRanks(worstIce), 1e-11) << label;
    EXPECT_LT(MaxOverRanks(worstField), 1e-9) << label;
    EXPECT_LT(MaxOverRanks(worstSecond), 1e-9) << label;
  }
}

TEST(Distributed, AcrossAJumpTheRunAndItsSweepsMatchTheOneProcess) {
  // An earthquake at a repeated node under lateral viscosity beside a
  // melting load that steps there, its left limit without the source and
  // the step (Forcing::sourcesBefore, directLoadBefore): the zero-length
  // step under mpiexec as on one process, the run and the tangent along a
  // lateral log viscosity on every rank, and the first-order sweeps of
  // both methods (the ice views on every rank, each rank's slice of the
  // log-viscosity kernel), measured at the pair's right node and after.
  constexpr auto lmax = 8;
  const auto body = Body::Make(lmax);
  const auto share =
      ShareMesh(body.material.Mesh(), MPI_COMM_WORLD,
                GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  GRDSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
                   SeaLevelParameters{});
  const auto& sle = solver.SolverScales();
  const auto length = sle.Length();
  auto toNd = [&](std::vector<double> si) {
    for (auto& v : si) v /= length;
    return si;
  };
  const auto ice = toNd(fingerprint.IceThickness());
  const auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
  const auto kyr = 1000.0 * 3.15576e7 / sle.Time();
  const auto onset = 5.0 * kyr;
  const auto times = std::vector<double>{0.0,   2.5 * kyr, onset,
                                         onset, 7.5 * kyr, 10.0 * kyr};
  auto melt = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < melt.size(); ++i) {
    melt[i] =
        -0.1 * solver.IceDensity() * (1.0 - state.oceanFunction[i]) * ice[i];
  }
  const auto unit = 1e24 / sle.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto source = PointSource{
      .at = {.radius = solver.Radius() - 400e3 / length,
             .colatitude = 1.1,
             .longitude = 4.0},
      .moment = {.rr = 0.9 * unit, .tt = -0.2 * unit, .pp = -0.7 * unit,
                 .rt = 0.3 * unit, .rp = -0.7 * unit, .tp = 0.6 * unit}};
  // the melt ramped, a fifth more of it at a stroke at the onset
  const auto meltAt = [&](double t, bool stepped) {
    auto out = melt;
    for (auto& v : out) v *= t / times.back() + (stepped ? 0.2 : 0.0);
    return out;
  };
  const auto forcing = Evolution3D::Forcing{
      .directLoad = [&](double t) { return meltAt(t, t >= onset); },
      .sources =
          [&](double t) {
            return t >= onset ? std::vector<PointSource>{source}
                              : std::vector<PointSource>{};
          },
      .sourcesBefore =
          [&](double t) {
            return t > onset ? std::vector<PointSource>{source}
                             : std::vector<PointSource>{};
          },
      .directLoadBefore = [&](double t) { return meltAt(t, t > onset); }};
  auto measurements = std::map<std::size_t, Measurement>{};
  const auto cap = solver.Grid().FromFunction([](double theta, double phi) {
    return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
  });
  measurements[times.size() - 1].relativeSeaLevelDual = cap;
  measurements[3].displacementDual = cap;
  measurements[3].toroidalPotentialDual = cap;
  auto pU = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
  };
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .radial_cuts = share.cuts};
  const auto evolutionOptions =
      Evolution3D::Options{.rtol = 1e-12,
                           .max_coupled_iterations = 100,
                           .coupled_rtol = 1e-11,
                           .record_stride = 1};
  const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
  const auto& all = whole.Coupling();
  auto logField = std::vector<double>(all.GridValueCount(), 0.0);
  for (auto rn = 0; rn < all.NodeCount(); ++rn) {
    const auto values = all.GridOf(rn).FromFunction(pU);
    const auto at = all.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x) logField[at + x] = values[x];
  }
  struct Result {
    std::vector<Evolution3D::Snapshot> run;
    std::vector<Evolution3D::Snapshot> tangent;
    std::map<AdjointMethod, Kernels3D> kernels;
  };
  auto runOn = [&](const Viscoelastic3D& op) {
    Evolution3D evolution(op, solver, state, evolutionOptions);
    Result out;
    out.run = evolution.Run(times, forcing);
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(out.run);
    auto direction = TangentPerturbation3D{};
    direction.logViscosity = Slice(op.Coupling(), all, logField);
    out.tangent = TangentLinearSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                    .rtol = 1e-12,
                                                    .max_coupled_iterations = 100,
                                                    .coupled_rtol = 1e-11})
                      .Run(times, direction, &trajectory);
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      out.kernels[method] =
          AdjointSolver3D(evolution, {.method = method,
                                      .rtol = 1e-12,
                                      .max_coupled_iterations = 100,
                                      .coupled_rtol = 1e-11})
              .FirstOrderKernels(times, measurements, &trajectory);
    }
    return out;
  };
  const auto want = runOn(whole);
  auto mine = options;
  mine.segments = share.segments;
  mine.collectives = share.collectives;
  const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
  const auto got = runOn(op);
  ASSERT_EQ(got.run.size(), times.size());
  auto worstRun = 0.0, worstTangent = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    EXPECT_TRUE(got.run[n].converged) << n;
    for (auto field : {&Evolution3D::Snapshot::relativeSeaLevelChange,
                       &Evolution3D::Snapshot::displacement,
                       &Evolution3D::Snapshot::toroidalPotential}) {
      worstRun = std::max(worstRun, Relative(got.run[n].*field, want.run[n].*field));
    }
    worstTangent = std::max(worstTangent,
                            Relative(got.tangent[n].relativeSeaLevelChange,
                                     want.tangent[n].relativeSeaLevelChange));
  }
  EXPECT_GT(Relative(got.run[3].displacement, got.run[2].displacement), 1e-3);
  // measured on two and three ranks: 3e-15 in the run, 1.4e-15 in the
  // tangent, 1.8e-14 in the ice views and 9e-13 in the 3D kernel, whose
  // sweeps close at the coupled tolerance of 1e-11
  EXPECT_LT(MaxOverRanks(worstRun), 1e-11);
  EXPECT_LT(MaxOverRanks(worstTangent), 1e-11);
  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const auto label = method == AdjointMethod::Discrete ? "discrete" : "continuous";
    auto worstIce = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      worstIce = std::max(worstIce,
                          Relative(got.kernels.at(method).ice.directLoadKernels[n],
                                   want.kernels.at(method).ice.directLoadKernels[n]));
    }
    const auto worstField =
        FieldSliceDifference(op.Coupling(), got.kernels.at(method).logViscosityKernel3D[0],
                             all, want.kernels.at(method).logViscosityKernel3D[0]);
    EXPECT_LT(MaxOverRanks(worstIce), 1e-11) << label;
    EXPECT_LT(MaxOverRanks(worstField), 1e-9) << label;
  }
}

TEST(Distributed, Sdirk2RunAndSweepMatchTheOneProcess) {
  // Sdirk2 over radial segments: each stage's combined fixed point
  // collective as a step's, the stage's memory state formed from each
  // rank's own nodes. The evolution, frozen and migrating, and the
  // continuous sweep stepped by the same stages, against one process: the
  // sea level at every node, the ice kernels on every rank and each rank's
  // slice of the log-viscosity kernel, frozen and about the migrating run.
  constexpr auto lmax = 8;
  const auto body = Body::Make(lmax);
  const auto share =
      ShareMesh(body.material.Mesh(), MPI_COMM_WORLD,
                GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  GRDSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
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
  const auto directLoad = [&](double t) {
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
  measurements[times.size() - 1].relativeSeaLevelDual =
      solver.Grid().FromFunction([](double theta, double phi) {
        return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
      });
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .radial_cuts = share.cuts};
  const auto evolutionOptions =
      Evolution3D::Options{.scheme = Scheme::Sdirk2,
                           .rtol = 1e-12,
                           .max_coupled_iterations = 100,
                           .coupled_rtol = 1e-11,
                           .record_stride = 1};
  const auto adjointOptions =
      AdjointSolver3D::Options{.method = AdjointMethod::Continuous,
                               .rtol = 1e-12,
                               .max_coupled_iterations = 100,
                               .coupled_rtol = 1e-11};
  for (const auto migrating : {false, true}) {
    const auto label = migrating ? "migrating" : "frozen";
    auto run = [&](const Viscoelastic3D& op) {
      Evolution3D evolution(op, solver, state, evolutionOptions);
      const auto snapshots = migrating
                                 ? evolution.RunMigrating(times, thickness)
                                 : evolution.Run(times, directLoad);
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      const AdjointSolver3D adjoint(evolution, adjointOptions);
      auto kernels = adjoint.FirstOrderKernels(
          times, measurements, &trajectory,
          migrating ? OceanFunctions::Of(snapshots) : OceanFunctions{});
      return std::pair{snapshots, kernels};
    };
    const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
    const auto [want, wantKernels] = run(whole);
    auto mine = options;
    mine.segments = share.segments;
    mine.collectives = share.collectives;
    const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
    const auto [got, gotKernels] = run(op);
    ASSERT_EQ(got.size(), want.size());
    auto worst = 0.0;
    for (auto n = std::size_t{0}; n < want.size(); ++n) {
      EXPECT_TRUE(got[n].converged) << label << ", node " << n;
      worst = std::max(worst, Relative(got[n].relativeSeaLevelChange,
                                       want[n].relativeSeaLevelChange));
      if (migrating)
        EXPECT_EQ(got[n].oceanFunction, want[n].oceanFunction) << n;
    }
    const auto evolutionWorst = MaxOverRanks(worst);
    auto rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cout << std::format("[sdirk2 distributed] {}: run {:.1e}\n", label,
                               evolutionWorst);
    }
    // measured 4e-16 to 3e-14 on two to four ranks
    EXPECT_LT(evolutionWorst, 1e-12) << label;
    auto worstIce = 0.0;
    for (auto n = std::size_t{0}; n < wantKernels.ice.directLoadKernels.size();
         ++n) {
      worstIce =
          std::max(worstIce, Relative(gotKernels.ice.directLoadKernels[n],
                                      wantKernels.ice.directLoadKernels[n]));
    }
    const auto& part = op.Coupling();
    const auto& all = whole.Coupling();
    auto worstViscosity = 0.0;
    if (part.NodeCount() > 0) {
      auto first = 0;
      while (all.ElementOfNode(first) != part.ElementOfNode(0)) ++first;
      const auto& kernel = gotKernels.logViscosityKernel3D[0];
      worstViscosity =
          Relative(kernel, std::span(wantKernels.logViscosityKernel3D[0])
                               .subspan(all.FieldOffset(first), kernel.size()));
    }
    const auto ice = MaxOverRanks(worstIce);
    const auto viscosity = MaxOverRanks(worstViscosity);
    if (rank == 0) {
      std::cout << std::format(
          "[sdirk2 distributed] {}: ice kernels {:.1e}, viscosity {:.1e}\n",
          label, ice, viscosity);
    }
    // measured 1.1e-14 to 2.5e-13 (ice) and 6.7e-13 to 2.2e-12 (viscosity)
    // on two to four ranks
    EXPECT_LT(ice, 1e-9) << label;
    EXPECT_LT(viscosity, 1e-9) << label;
  }
}

TEST(Distributed, Sdirk2DerivativesMatchTheOneProcess) {
  // The derivatives of the Sdirk2 stepped map over radial segments, about
  // a frozen and a migrating run recorded at stride 2 (its gaps, and each
  // step's first stage, the run's stages again under their recorded
  // closures, collective as a step's): the discrete sweep's ice kernels on
  // every rank and each rank's slice of its log-viscosity kernel, the
  // discrete tangent-linear run along a uniform log viscosity, and the
  // discrete second-order sweep's derivatives along it, against one
  // process; and the sweep's kernel paired with the direction over the
  // ranks is the tangent run's objective.
  constexpr auto lmax = 8;
  const auto body = Body::Make(lmax);
  const auto share =
      ShareMesh(body.material.Mesh(), MPI_COMM_WORLD,
                GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  GRDSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
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
      std::vector<double>{0.0, 3.0 * kyr, 6.0 * kyr, 8.0 * kyr, 10.0 * kyr};
  auto melt = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < melt.size(); ++i) {
    melt[i] =
        -0.1 * solver.IceDensity() * (1.0 - state.oceanFunction[i]) * ice[i];
  }
  const auto directLoad = [&](double t) {
    auto out = melt;
    for (auto& v : out) v *= t / times.back();
    return out;
  };
  const auto thickness = [&](double t) {
    auto out = ice;
    for (auto& v : out) v *= 1.0 - 0.3 * t / times.back();
    return out;
  };
  const auto dual = solver.Grid().FromFunction([](double theta, double phi) {
    return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
  });
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].relativeSeaLevelDual = dual;
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .radial_cuts = share.cuts};
  const auto evolutionOptions =
      Evolution3D::Options{.scheme = Scheme::Sdirk2,
                           .rtol = 1e-12,
                           .max_coupled_iterations = 100,
                           .coupled_rtol = 1e-11,
                           .record_stride = 2};
  const auto adjointOptions =
      AdjointSolver3D::Options{.method = AdjointMethod::Discrete,
                               .rtol = 1e-12,
                               .max_coupled_iterations = 100,
                               .coupled_rtol = 1e-11};
  struct Results {
    Kernels3D kernels;
    KernelDerivatives3D along;
    std::vector<double> tangentSeaLevel;
    double paired = 0.0;  // the kernel with the direction, this rank's part
  };
  auto rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (const auto migrating : {false, true}) {
    const auto label = migrating ? "migrating" : "frozen";
    auto run = [&](const Viscoelastic3D& op) {
      Evolution3D evolution(op, solver, state, evolutionOptions);
      const auto snapshots = migrating
                                 ? evolution.RunMigrating(times, thickness)
                                 : evolution.Run(times, directLoad);
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      const auto oceans =
          migrating ? OceanFunctions::Of(snapshots) : OceanFunctions{};
      TangentPerturbation3D direction;
      direction.logViscosity.assign(op.Coupling().GridValueCount(), 0.1);
      const TangentLinearSolver3D tangent(
          evolution, {.method = AdjointMethod::Discrete,
                      .rtol = 1e-12,
                      .max_coupled_iterations = 100,
                      .coupled_rtol = 1e-11,
                      .record = true});
      const auto tangentRun =
          tangent.Run(times, direction, &trajectory, oceans);
      const AdjointSolver3D adjoint(evolution, adjointOptions);
      auto [kernels, along] = adjoint.SecondOrderKernels(
          times, measurements, tangentRun,
          TangentLinearSolver3D::ItauOfLogViscosity(op, direction.logViscosity),
          trajectory, oceans);
      auto out = Results{std::move(kernels), std::move(along),
                         tangentRun.back().relativeSeaLevelChange, 0.0};
      out.paired = VolumePairing(op.Coupling(), out.kernels.logViscosityKernel3D[0],
                                 direction.logViscosity);
      return out;
    };
    const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
    const auto want = run(whole);
    auto mine = options;
    mine.segments = share.segments;
    mine.collectives = share.collectives;
    const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
    const auto got = run(op);
    const auto& part = op.Coupling();
    const auto& all = whole.Coupling();
    // each rank's slice of a field over the whole mesh's memory nodes
    auto slice = [&](const std::vector<double>& field, std::size_t size) {
      auto first = 0;
      while (all.ElementOfNode(first) != part.ElementOfNode(0)) ++first;
      return std::span(field).subspan(all.FieldOffset(first), size);
    };
    auto worstIce = 0.0, worstAlong = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      worstIce = std::max(worstIce, Relative(got.kernels.ice.thickness[n],
                                             want.kernels.ice.thickness[n]));
      worstAlong = std::max(worstAlong, Relative(got.along.ice.thickness[n],
                                                 want.along.ice.thickness[n]));
    }
    auto worstViscosity = 0.0, worstAlongViscosity = 0.0;
    if (part.NodeCount() > 0) {
      const auto& kernel = got.kernels.logViscosityKernel3D[0];
      worstViscosity = Relative(
          kernel, slice(want.kernels.logViscosityKernel3D[0], kernel.size()));
      const auto& derivative = got.along.logViscosityKernel3D[0];
      worstAlongViscosity = Relative(
          derivative,
          slice(want.along.logViscosityKernel3D[0], derivative.size()));
    }
    const auto tangentGap =
        MaxOverRanks(Relative(got.tangentSeaLevel, want.tangentSeaLevel));
    const auto iceGap = MaxOverRanks(worstIce);
    const auto viscosityGap = MaxOverRanks(worstViscosity);
    const auto alongGap = MaxOverRanks(std::max(worstAlong, worstAlongViscosity));
    // the transpose identity over the ranks
    auto paired = got.paired;
    MPI_Allreduce(MPI_IN_PLACE, &paired, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const auto radius = solver.Radius();
    auto product = got.tangentSeaLevel;
    for (auto i = std::size_t{0}; i < product.size(); ++i) product[i] *= dual[i];
    const auto objective = radius * radius * solver.Grid().Integrate(product);
    const auto transposeGap = std::abs(paired - objective) / std::abs(objective);
    if (rank == 0) {
      std::cout << std::format(
          "[sdirk2 distributed derivatives] {}: tangent {:.1e}, ice kernels "
          "{:.1e}, viscosity {:.1e}, second order {:.1e}; transpose over the "
          "ranks {:.1e}\n",
          label, tangentGap, iceGap, viscosityGap, alongGap, transposeGap);
    }
    EXPECT_LT(tangentGap, 1e-9) << label;
    EXPECT_LT(iceGap, 1e-9) << label;
    EXPECT_LT(viscosityGap, 1e-9) << label;
    EXPECT_LT(alongGap, 1e-8) << label;
    EXPECT_LT(transposeGap, 1e-8) << label;
  }
}

namespace {

/// A marine cap on a 3 km ocean (9 km at its peak, so that its band-limited
/// thickness grounds at degree 8), solver units: its melting floods its
/// cells, and the run's initial sea level kernel is not zero.
std::vector<double> MarineCapOn(const GRDSolver& solver) {
  const auto& grid = solver.Grid();
  const auto metre = 1.0 / solver.SolverScales().Length();
  return grid.Synthesise(
      grid.Expand(grid.FromFunction([&](double theta, double phi) {
        const auto c = std::cos(theta) * std::cos(0.5236) +
                       std::sin(theta) * std::sin(0.5236) * std::cos(phi - 4.8);
        const auto d = std::acos(std::clamp(c, -1.0, 1.0));
        if (d >= 0.349) return 0.0;
        const auto taper = std::cos(0.5 * 3.14159265358979323846 * d / 0.349);
        return 9000.0 * metre * taper * taper;
      })));
}

}  // namespace

TEST(Distributed, CalibratedKernelsMatchTheOneProcess) {
  // The total kernels of a calibrated run (ComputeCalibratedKernels) over
  // radial segments: the multiplier's sweeps and the final one collective
  // as every 3D sweep, about a migrating run of a marine cap that floods.
  // Against one process: the multiplier and the ice kernels on every rank,
  // each rank's slice of the log-viscosity kernel, and the sweeps taken.
  constexpr auto lmax = 8;
  const auto body = Body::Make(lmax);
  const auto share =
      ShareMesh(body.material.Mesh(), MPI_COMM_WORLD,
                GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ViscoelasticOperator reference(body.rheology, lmax);
  GRDSolver solver(SurfaceGrid(lmax), reference.ElasticLoveNumbers(),
                   SeaLevelParameters{});
  const auto ice = MarineCapOn(solver);
  const auto state = solver.MakeState(
      ice, std::vector<double>(ice.size(),
                               3000.0 / solver.SolverScales().Length()));
  const auto kyr = 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  const auto times = std::vector<double>{0.0, 3.0 * kyr, 6.0 * kyr, 10.0 * kyr};
  const auto thickness = [&](double t) {
    auto out = ice;
    for (auto& v : out) v *= 1.0 - t / times.back();
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].relativeSeaLevelDual =
      solver.Grid().FromFunction([](double theta, double phi) {
        return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
      });
  measurements[1].displacementDual =
      measurements[times.size() - 1].relativeSeaLevelDual;
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .radial_cuts = share.cuts};
  const auto evolutionOptions =
      Evolution3D::Options{.rtol = 1e-12,
                           .max_coupled_iterations = 100,
                           .coupled_rtol = 1e-11,
                           .record_stride = 1};
  const auto adjointOptions =
      AdjointSolver3D::Options{.rtol = 1e-12,
                               .max_coupled_iterations = 100,
                               .coupled_rtol = 1e-11};
  auto run = [&](const Viscoelastic3D& op) {
    Evolution3D evolution(op, solver, state, evolutionOptions);
    const auto snapshots = evolution.RunMigrating(times, thickness);
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
    return ComputeCalibratedKernels(evolution, times, measurements,
                                    &trajectory, OceanFunctions::Of(snapshots),
                                    {.rtol = 1e-10}, adjointOptions);
  };
  const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
  const auto want = run(whole);
  auto mine = options;
  mine.segments = share.segments;
  mine.collectives = share.collectives;
  const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
  const auto got = run(op);
  EXPECT_TRUE(want.converged);
  EXPECT_TRUE(got.converged);
  EXPECT_GT(want.sweeps, 2);
  EXPECT_EQ(got.sweeps, want.sweeps);
  auto scale = 0.0;
  for (const auto v : want.multiplier) scale = std::max(scale, std::abs(v));
  EXPECT_GT(scale, 0.0);
  auto worstIce = 0.0;
  for (auto n = std::size_t{0}; n < want.kernels.ice.thickness.size(); ++n) {
    worstIce = std::max(worstIce, Relative(got.kernels.ice.thickness[n],
                                           want.kernels.ice.thickness[n]));
  }
  const auto& part = op.Coupling();
  const auto& all = whole.Coupling();
  auto worstViscosity = 0.0;
  if (part.NodeCount() > 0) {
    auto first = 0;
    while (all.ElementOfNode(first) != part.ElementOfNode(0)) ++first;
    const auto& kernel = got.kernels.logViscosityKernel3D[0];
    worstViscosity =
        Relative(kernel, std::span(want.kernels.logViscosityKernel3D[0])
                             .subspan(all.FieldOffset(first), kernel.size()));
  }
  const auto multiplier =
      MaxOverRanks(Relative(got.multiplier, want.multiplier));
  const auto iceGap = MaxOverRanks(worstIce);
  const auto viscosity = MaxOverRanks(worstViscosity);
  auto rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << std::format(
        "[calibrated distributed] {} sweeps: multiplier {:.1e}, ice kernels "
        "{:.1e}, viscosity {:.1e}\n",
        got.sweeps, multiplier, iceGap, viscosity);
  }
  EXPECT_LT(multiplier, 1e-9);
  EXPECT_LT(iceGap, 1e-9);
  EXPECT_LT(viscosity, 1e-9);
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

TEST(Distributed, LateralModuliStepsMatchTheOneProcessSteps) {
  // Lateral moduli under mpiexec: the conjugate-gradient start
  // solve pairing each rank's share, two lateral steps, and the static
  // problem on a branchless rheology, against the one process.
  constexpr auto lmax = 16;
  constexpr auto eps = 1e-2;
  const auto degrees = std::vector<int>{8, 12};
  const auto body = Body::Make(lmax);
  const auto elastic = Rheology::Create(body.material, {}).value();
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
  const auto moduli = Viscoelastic3D::Options::LateralModuli{
      .shear =
          [](double r, double theta, double phi) {
            return 0.1 * (0.5 + r * r) *
                   (std::sin(theta) * std::cos(theta) * std::cos(phi) +
                    0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
          },
      .bulk = [](double r, double theta, double) {
        return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
      }};
  const auto options = Viscoelastic3D::Options{
      .eps = eps,
      .parallel = false,
      .max_iterations = 100,
      .rtol = 1e-12,
      .solver = Viscoelastic3D::Options::StepSolver::ConjugateGradients,
      .solve_rtol = 1e-12,
      .moduli = moduli,
      .radial_cuts = share.cuts,
      .transform_degrees = degrees};
  struct Result {
    Viscoelastic3D::State state;
    std::map<int, std::vector<std::complex<double>>> U, staticU;
    std::vector<int> iterations;
    int startIterations = 0;
  };
  auto run = [&](const Viscoelastic3D& op, const Viscoelastic3D& stat) {
    auto result = Result{op.ZeroState()};
    Viscoelastic3D::Solutions solutions;
    for (auto i = 0; i < 2; ++i) {
      auto step = op.Step(result.state, tau, Scheme::ExponentialTrapezoid, load);
      EXPECT_TRUE(step.info.converged);
      result.iterations.push_back(step.info.iterations);
      result.state = std::move(step.state);
      solutions = std::move(step.solutions);
    }
    result.U = op.Surface(solutions, Component::U);
    const auto x = stat.SolveElastic(stat.ZeroState(), load);
    result.startIterations = stat.LastElasticSolve().iterations;
    result.staticU = stat.Surface(x, Component::U);
    return result;
  };
  const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
  const Viscoelastic3D wholeStatic(elastic, lmax, Peaked, options);
  const auto want = run(whole, wholeStatic);
  auto mine = options;
  mine.segments = share.segments;
  mine.collectives = share.collectives;
  const Viscoelastic3D op(body.rheology, lmax, Peaked, mine);
  const Viscoelastic3D stat(elastic, lmax, Peaked, mine);
  const auto got = run(op, stat);
  auto surface = 0.0, staticSurface = 0.0;
  for (const auto& [l, values] : want.U) {
    surface = std::max(surface, Relative(got.U.at(l), values));
    staticSurface = std::max(staticSurface, Relative(got.staticU.at(l), want.staticU.at(l)));
  }
  // the within-step passes may differ by one under MPI's summation order
  // when a residual lands at the tolerance (seen: 17 against 16)
  ASSERT_EQ(got.iterations.size(), want.iterations.size());
  for (auto i = std::size_t{0}; i < got.iterations.size(); ++i) {
    EXPECT_LE(std::abs(got.iterations[i] - want.iterations[i]), 1) << i;
  }
  EXPECT_LE(std::abs(got.startIterations - want.startIterations), 1);
  const auto worstSurface = MaxOverRanks(surface);
  const auto worstStatic = MaxOverRanks(staticSurface);
  const auto worstMemory = MaxOverRanks(SliceDifference(op, got.state, whole, want.state));
  // the conjugate-gradient solves at 1e-12 in MPI's summation order:
  // measured 5.5e-12 at most on two to four ranks
  EXPECT_LT(worstSurface, 1e-9);
  EXPECT_LT(worstStatic, 1e-9);
  EXPECT_LT(worstMemory, 1e-9);
}
