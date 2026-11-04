// The 3D step distributed over radial segments (Viscoelastic3D::Options::
// segments and collectives), its ranks emulated by threads of this one
// process: ThreadRanks makes each collective a rendezvous at which every
// rank reduces all the ranks' values in rank order, so that all get the
// same. The distributed paths run without MPI and are held to the
// one-process run on the same cuts: the steps (with the transforms cut
// with depth), the coupled evolutions, frozen and migrating, and the 3D
// adjoint's kernels. Each rank's memory nodes are a slice of the one
// process's, compared value by value. TestDistributed runs the same under
// mpiexec.

#include <gtest/gtest.h>

#include <iostream>

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <complex>
#include <exception>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/TangentLinear3D.hpp"
#include "sl3dpp/Collectives.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using Complex = std::complex<double>;

namespace {

/// `count` ranks as threads of this process. Each collective is a
/// rendezvous: every rank posts its values, then reads all of them and
/// reduces in rank order (the same result on every rank), then waits
/// until all have read before overwriting its own. A rank that throws
/// drops out and the others fail at their next collective rather than
/// wait for it; Run rethrows the first failure.
class ThreadRanks {
 public:
  explicit ThreadRanks(int count)
      : count_(count), slots_(count), errors_(count), barrier_(count) {}

  [[nodiscard]] Collectives For(int rank) {
    return {
        .sum = [this, rank](std::span<double> v) { Reduce(rank, v, Op::Sum); },
        .max = [this, rank](std::span<double> v) { Reduce(rank, v, Op::Max); },
        .share = [this,
                  rank](std::span<double> v) { Reduce(rank, v, Op::Share); }};
  }

  /// body(rank) on every rank's thread, then joined.
  void Run(const std::function<void(int)>& body) {
    {
      auto threads = std::vector<std::jthread>{};
      for (auto rank = 0; rank < count_; ++rank) {
        threads.emplace_back([&, rank] {
          try {
            body(rank);
          } catch (...) {
            errors_[rank] = std::current_exception();
            failed_ = true;
            barrier_.arrive_and_drop();
          }
        });
      }
    }
    for (const auto& error : errors_) {
      if (error) std::rethrow_exception(error);
    }
  }

 private:
  enum class Op { Sum, Max, Share };

  void Wait() {
    barrier_.arrive_and_wait();
    if (failed_) throw std::runtime_error("another rank failed");
  }

  void Reduce(int rank, std::span<double> v, Op op) {
    slots_[rank] = v;
    Wait();
    auto out = std::vector<double>(v.size());
    for (auto r = 0; r < count_; ++r) {
      if (slots_[r].size() != v.size()) {
        throw std::length_error(
            std::format("rank {} posted {} values, rank {} {}", r,
                        slots_[r].size(), rank, v.size()));
      }
    }
    for (auto i = std::size_t{0}; i < v.size(); ++i) {
      auto value = slots_[0][i];
      for (auto r = 1; r < count_ && op != Op::Share; ++r) {
        value = op == Op::Sum ? value + slots_[r][i]
                              : std::max(value, slots_[r][i]);
      }
      out[i] = value;
    }
    Wait();
    std::ranges::copy(out, v.begin());
  }

  int count_;
  std::vector<std::span<double>> slots_;
  std::vector<std::exception_ptr> errors_;
  std::atomic<bool> failed_ = false;
  std::barrier<> barrier_;
};

double Noise(std::size_t k) {
  const auto x = std::sin(12.9898 * static_cast<double>(k + 1)) * 43758.5453;
  return x - std::floor(x) - 0.5;
}

struct Body {
  EarthModel model;
  Material material;
  Rheology rheology;

  static Body Make(int lmax, double viscositySI = 1e21) {
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
        viscosity.emplace(i, viscositySI / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    return Body{std::move(model), std::move(material), std::move(rheology)};
  }
};

std::vector<Complex> LoadColumns(int l) {
  auto columns = std::vector<Complex>(l + 1);
  for (auto m = 0; m <= l; ++m) {
    columns[m] = m == 0 ? Complex(Noise(97 * l), 0.0)
                        : Complex(Noise(97 * l + m), Noise(55 * l + m));
  }
  return columns;
}

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

/// The largest relative difference of a rank's memory tensors from the
/// one process's at the same nodes (the rank's are a contiguous slice).
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

/// The same for one value per memory node.
double NodeSliceDifference(const GridCoupling& part,
                           std::span<const double> got, const GridCoupling& all,
                           std::span<const double> want) {
  if (part.NodeCount() == 0) return 0.0;
  auto first = 0;
  while (all.ElementOfNode(first) != part.ElementOfNode(0)) ++first;
  return Relative(got,
                  want.subspan(static_cast<std::size_t>(first), got.size()));
}

}  // namespace

TEST(RadialRanks, ThreadRanksReduceAlike) {
  ThreadRanks ranks(3);
  auto out = std::vector<std::vector<double>>(3);
  ranks.Run([&](int rank) {
    auto collectives = ranks.For(rank);
    auto v = std::vector<double>{1.0, static_cast<double>(rank),
                                 rank == 1 ? 7.0 : -1.0};
    auto w = v;
    auto u = v;
    collectives.sum(v);
    collectives.max(w);
    collectives.share(u);
    v.insert(v.end(), w.begin(), w.end());
    v.insert(v.end(), u.begin(), u.end());
    out[rank] = v;
  });
  for (const auto& v : out) {
    EXPECT_EQ(
        v, (std::vector<double>{3.0, 3.0, 5.0, 1.0, 2.0, 7.0, 1.0, 0.0, -1.0}));
  }
  EXPECT_THROW(ranks.Run([&](int rank) {
    auto v = std::vector<double>(rank == 2 ? 1 : 2);
    ranks.For(rank).sum(v);
  }),
               std::exception);
}

TEST(RadialRanks, StepsMatchTheOneProcessSteps) {
  // Two and three ranks, the transforms cut with depth (lmax 16 with a
  // loose truncation, as TestNodeBlocks), the cuts balanced on the
  // transform work: two lateral steps, each rank's surface response and
  // memory against the one process's on the same cuts.
  constexpr auto lmax = 16;
  constexpr auto eps = 1e-2;
  const auto degrees = std::vector<int>{8, 12};
  const auto body = Body::Make(lmax);
  const auto& mesh = body.material.Mesh();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto load = [](int l, double) { return LoadColumns(l); };

  struct Result {
    Viscoelastic3D::State state;
    std::map<int, std::vector<Complex>> U, phi;
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
    result.phi = op.Surface(solutions, Component::Phi);
    return result;
  };

  for (const auto count : {2, 3}) {
    const auto cuts = mesh.BalancedCuts(
        count,
        GridCoupling::TransformWeights(
            body.rheology, lmax, {.eps = eps, .transform_degrees = degrees}));
    ASSERT_EQ(cuts.size(), static_cast<std::size_t>(count - 1));
    const auto options = Viscoelastic3D::Options{.eps = eps,
                                                 .parallel = false,
                                                 .max_iterations = 100,
                                                 .rtol = 1e-12,
                                                 .radial_cuts = cuts,
                                                 .transform_degrees = degrees};
    const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
    const auto want = twoSteps(whole);

    ThreadRanks ranks(count);
    auto ops = std::vector<std::unique_ptr<Viscoelastic3D>>{};
    for (auto rank = 0; rank < count; ++rank) {
      auto mine = options;
      mine.segments = {rank};
      mine.collectives = ranks.For(rank);
      ops.push_back(
          std::make_unique<Viscoelastic3D>(body.rheology, lmax, Peaked, mine));
    }
    auto nodes = 0;
    for (const auto& op : ops) nodes += op->Coupling().NodeCount();
    EXPECT_EQ(nodes, whole.Coupling().NodeCount());
    auto got = std::vector<Result>(count);
    ranks.Run([&](int rank) { got[rank] = twoSteps(*ops[rank]); });

    auto worstSurface = 0.0, worstMemory = 0.0;
    for (auto rank = 0; rank < count; ++rank) {
      EXPECT_EQ(got[rank].iterations, got[0].iterations);
      for (const auto& [l, values] : want.U) {
        worstSurface =
            std::max({worstSurface, Relative(got[rank].U.at(l), values),
                      Relative(got[rank].phi.at(l), want.phi.at(l))});
      }
      worstMemory = std::max(
          worstMemory,
          SliceDifference(*ops[rank], got[rank].state, whole, want.state));
    }
    RecordProperty(std::format("ranks{}_surface", count),
                   std::format("{:.3e}", worstSurface));
    RecordProperty(std::format("ranks{}_memory", count),
                   std::format("{:.3e}", worstMemory));
    // measured 5e-16 and 2.5e-14: the same algebra, summed in
    // another order
    EXPECT_LT(worstSurface, 1e-13) << count << " ranks";
    EXPECT_LT(worstMemory, 1e-12) << count << " ranks";
  }
}

TEST(RadialRanks, EvolutionsMatchTheOneProcessRuns) {
  // The coupled evolution on three ranks, each solving the sea level
  // equation itself (the first rank's loads shared): frozen shorelines
  // under a melting load, and migrating ones under a thinning ice sheet.
  constexpr auto lmax = 8;
  constexpr auto count = 3;
  const auto body = Body::Make(lmax, 1e21);
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  auto solvers = std::vector<SeaLevelSolver>{};
  for (auto rank = 0; rank <= count; ++rank) {
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
  const auto times = std::vector<double>{0.0, 3.0 * kyr, 6.0 * kyr, 10.0 * kyr};
  auto melt = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < melt.size(); ++i) {
    melt[i] = -0.1 * solvers[0].IceDensity() * (1.0 - state.oceanFunction[i]) *
              ice[i];
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
  const auto cuts = body.material.Mesh().BalancedCuts(
      count, GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ASSERT_EQ(cuts.size(), static_cast<std::size_t>(count - 1));
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .radial_cuts = cuts};
  const auto evolutionOptions = SeaLevelEvolution3D::Options{
      .rtol = 1e-12, .max_coupled_iterations = 100, .coupled_rtol = 1e-11};

  for (const auto migrating : {false, true}) {
    auto run = [&](SeaLevelEvolution3D& evolution) {
      return migrating ? evolution.RunMigrating(times, thickness)
                       : evolution.Run(times, iceLoad);
    };
    const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
    SeaLevelEvolution3D one(whole, solvers[count], state, evolutionOptions);
    const auto want = run(one);

    ThreadRanks ranks(count);
    auto ops = std::vector<std::unique_ptr<Viscoelastic3D>>{};
    for (auto rank = 0; rank < count; ++rank) {
      auto mine = options;
      mine.segments = {rank};
      mine.collectives = ranks.For(rank);
      ops.push_back(
          std::make_unique<Viscoelastic3D>(body.rheology, lmax, Peaked, mine));
    }
    auto got = std::vector<std::vector<SeaLevelEvolution3D::Snapshot>>(count);
    ranks.Run([&](int rank) {
      SeaLevelEvolution3D evolution(*ops[rank], solvers[rank], state,
                                    evolutionOptions);
      got[rank] = run(evolution);
    });

    auto worst = 0.0;
    for (auto rank = 0; rank < count; ++rank) {
      ASSERT_EQ(got[rank].size(), want.size());
      for (auto n = std::size_t{0}; n < want.size(); ++n) {
        EXPECT_TRUE(got[rank][n].converged) << rank << ", node " << n;
        EXPECT_EQ(got[rank][n].coupledIterations, got[0][n].coupledIterations);
        worst = std::max(worst, Relative(got[rank][n].seaLevelChange,
                                         want[n].seaLevelChange));
        if (migrating) {
          EXPECT_EQ(got[rank][n].oceanFunction, want[n].oceanFunction)
              << rank << ", node " << n;
        }
      }
    }
    const auto label = migrating ? "migrating" : "frozen";
    RecordProperty(std::format("{}_sea_level", label),
                   std::format("{:.3e}", worst));
    EXPECT_LT(worst, 1e-13) << label;  // measured 4e-16 and 1.3e-15
    if (migrating) {
      auto flips = 0;
      for (auto i = std::size_t{0}; i < state.oceanFunction.size(); ++i) {
        flips += want.back().oceanFunction[i] != state.oceanFunction[i];
      }
      RecordProperty("migrating_flips", flips);
      EXPECT_GT(flips, 0) << "no cell flipped";
    }
  }
}

TEST(RadialRanks, AdjointKernelsMatchTheOneProcessSweep) {
  // The 3D continuous adjoint on three ranks, each sweeping its own
  // memory nodes back over its own forward trajectory: the ice kernels on
  // every rank, and each rank's slice of the viscosity kernels, against
  // the one process's sweep; under frozen shorelines and linearised about
  // a migrating run.
  constexpr auto lmax = 8;
  constexpr auto count = 3;
  const auto body = Body::Make(lmax, 1e22);
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  auto solvers = std::vector<SeaLevelSolver>{};
  for (auto rank = 0; rank <= count; ++rank) {
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
      solvers[0].Grid().FromFunction([](double theta, double phi) {
        return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
      });
  measurements[2].seaLevelDual = measurements[times.size() - 1].seaLevelDual;
  const auto cuts = body.material.Mesh().BalancedCuts(
      count, GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ASSERT_EQ(cuts.size(), static_cast<std::size_t>(count - 1));
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .radial_cuts = cuts};
  const auto evolutionOptions =
      SeaLevelEvolution3D::Options{.rtol = 1e-12,
                                   .max_coupled_iterations = 100,
                                   .coupled_rtol = 1e-11,
                                   .record_stride = 1};
  const auto adjointOptions = AdjointSolver3D::Options{
      .rtol = 1e-12, .max_coupled_iterations = 100, .coupled_rtol = 1e-11};

  for (const auto migrating : {false, true}) {
    auto sweep = [&](const Viscoelastic3D& op, const SeaLevelSolver& solver) {
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
    const auto want = sweep(whole, solvers[count]);

    ThreadRanks ranks(count);
    auto ops = std::vector<std::unique_ptr<Viscoelastic3D>>{};
    for (auto rank = 0; rank < count; ++rank) {
      auto mine = options;
      mine.segments = {rank};
      mine.collectives = ranks.For(rank);
      ops.push_back(
          std::make_unique<Viscoelastic3D>(body.rheology, lmax, Peaked, mine));
    }
    auto got = std::vector<Kernels3D>(count);
    ranks.Run([&](int rank) { got[rank] = sweep(*ops[rank], solvers[rank]); });

    auto worstIce = 0.0, worstViscosity = 0.0, worstSpherical = 0.0;
    for (auto rank = 0; rank < count; ++rank) {
      const auto& kernels = got[rank];
      ASSERT_EQ(kernels.ice.loadDuals.size(), want.ice.loadDuals.size());
      for (auto n = std::size_t{0}; n < want.ice.loadDuals.size(); ++n) {
        worstIce = std::max(worstIce, Relative(kernels.ice.loadDuals[n],
                                               want.ice.loadDuals[n]));
        // rank 0's sea level results serve every rank
        EXPECT_EQ(kernels.ice.loadDuals[n], got[0].ice.loadDuals[n]);
      }
      const auto& part = ops[rank]->Coupling();
      ASSERT_EQ(kernels.logViscosityKernel3D[0].size(), part.GridValueCount());
      worstViscosity = std::max(
          worstViscosity,
          FieldSliceDifference(part, kernels.logViscosityKernel3D[0],
                               whole.Coupling(), want.logViscosityKernel3D[0]));
      worstSpherical = std::max(
          worstSpherical,
          NodeSliceDifference(part, kernels.itauKernelSpherical[0],
                              whole.Coupling(), want.itauKernelSpherical[0]));
    }
    const auto label = migrating ? "migrating" : "frozen";
    RecordProperty(std::format("{}_ice", label),
                   std::format("{:.3e}", worstIce));
    RecordProperty(std::format("{}_viscosity", label),
                   std::format("{:.3e}", worstViscosity));
    RecordProperty(std::format("{}_spherical", label),
                   std::format("{:.3e}", worstSpherical));
    // measured 5e-15, 4e-14 and 2.6e-14 at most
    EXPECT_LT(worstIce, 1e-12) << label;
    EXPECT_LT(worstViscosity, 1e-12) << label;
    EXPECT_LT(worstSpherical, 1e-12) << label;
  }
}

namespace {

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

TEST(RadialRanks, DerivativesMatchTheOneProcess) {
  // The derivative machinery on three ranks against the one process on
  // the same cuts: the 3D tangent-linear runs (both methods) along loads,
  // a lateral log viscosity and lateral moduli; the first-order sweeps
  // with the material kernels, discrete (the transposed fixed point's
  // partial contractions summed, the surface duals on their holding
  // rank) and continuous; and the second-order sweeps of both methods
  // along a direction in all of those, the kernels' derivatives included
  // — the ice views on every rank, each rank's slices of the 3D kernels.
  constexpr auto lmax = 8;
  constexpr auto count = 3;
  const auto body = Body::Make(lmax, 1e22);
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  auto solvers = std::vector<SeaLevelSolver>{};
  for (auto rank = 0; rank <= count; ++rank) {
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
  const auto iceLoad = [&](double t) {
    auto out = melt;
    for (auto& v : out) v *= t / times.back();
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual =
      solvers[0].Grid().FromFunction([](double theta, double phi) {
        return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
      });
  measurements[2].displacementDual = measurements[times.size() - 1].seaLevelDual;
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
  const auto cuts = body.material.Mesh().BalancedCuts(
      count, GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ASSERT_EQ(cuts.size(), static_cast<std::size_t>(count - 1));
  const auto options = Viscoelastic3D::Options{
      .parallel = false, .rtol = 1e-12, .memory_points = MemoryPointSet::Gll,
      .radial_cuts = cuts};
  const auto evolutionOptions =
      SeaLevelEvolution3D::Options{.rotation = RotationalFeedback::None,
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
    std::map<AdjointMethod, std::vector<SeaLevelEvolution3D::Snapshot>> tangent;
    std::map<AdjointMethod, Kernels3D> kernels;
    std::map<AdjointMethod, std::pair<Kernels3D, KernelDerivatives3D>> second;
  };
  auto run = [&](const Viscoelastic3D& op, const SeaLevelSolver& solver) {
    SeaLevelEvolution3D evolution(op, solver, state, evolutionOptions);
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, iceLoad));
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
      out.kernels[method] = adjoint.IceHistoryKernels(times, measurements, &trajectory);
      out.second[method] = adjoint.SecondOrderKernels(
          times, measurements, out.tangent[method], itau, trajectory, {},
          direction.material);
    }
    return out;
  };
  const auto want = run(whole, solvers[count]);

  ThreadRanks ranks(count);
  auto ops = std::vector<std::unique_ptr<Viscoelastic3D>>{};
  for (auto rank = 0; rank < count; ++rank) {
    auto mine = options;
    mine.segments = {rank};
    mine.collectives = ranks.For(rank);
    ops.push_back(std::make_unique<Viscoelastic3D>(body.rheology, lmax, Peaked, mine));
  }
  auto got = std::vector<Result>(count);
  ranks.Run([&](int rank) { got[rank] = run(*ops[rank], solvers[rank]); });

  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    const auto label = method == AdjointMethod::Discrete ? "discrete" : "continuous";
    auto worstTangent = 0.0, worstIce = 0.0, worstField = 0.0, worstSecond = 0.0;
    for (auto rank = 0; rank < count; ++rank) {
      const auto& part = ops[rank]->Coupling();
      const auto& r = got[rank];
      const auto& w = want;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        worstTangent = std::max(
            worstTangent, Relative(r.tangent.at(method)[n].seaLevelChange,
                                   w.tangent.at(method)[n].seaLevelChange));
        worstIce = std::max(worstIce, Relative(r.kernels.at(method).ice.loadDuals[n],
                                               w.kernels.at(method).ice.loadDuals[n]));
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
    EXPECT_LT(worstTangent, 1e-11) << label;
    EXPECT_LT(worstIce, 1e-11) << label;
    EXPECT_LT(worstField, 1e-11) << label;
    EXPECT_LT(worstSecond, 1e-11) << label;
  }
}

// -- lateral moduli over the ranks --------------------------------------------

namespace {

Viscoelastic3D::Options::LateralModuli RankModuli() {
  return {.shear =
              [](double r, double theta, double phi) {
                return 0.1 * (0.5 + r * r) *
                       (std::sin(theta) * std::cos(theta) * std::cos(phi) +
                        0.3 * std::cos(2 * phi) * std::sin(theta) * std::sin(theta));
              },
          .bulk = [](double r, double theta, double) {
            return 0.05 * (1.0 - 0.5 * r) * std::cos(theta);
          }};
}

}  // namespace

TEST(RadialRanks, LateralModuliStepsMatchTheOneProcessSteps) {
  // Lateral moduli on two and three ranks: the start solve
  // (the conjugate-gradient solve pairing each rank's share) and two
  // lateral steps, each rank's surface response and memory against the
  // one process's; and the static problem on a branchless rheology.
  constexpr auto lmax = 16;
  constexpr auto eps = 1e-2;
  const auto degrees = std::vector<int>{8, 12};
  const auto body = Body::Make(lmax);
  const auto& mesh = body.material.Mesh();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto load = [](int l, double) { return LoadColumns(l); };
  const auto elastic = Rheology::Create(body.material, {}).value();

  struct Result {
    Viscoelastic3D::State state;
    std::map<int, std::vector<Complex>> U, phi, staticU;
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
    result.phi = op.Surface(solutions, Component::Phi);
    // the static problem: a branchless rheology's elastic solve under a
    // load and a tidal potential (the 3D body tide over the ranks)
    const auto tide = [](int l, double) {
      auto c = LoadColumns(l);
      for (auto& v : c) v *= 0.3 * (l + 1);
      return c;
    };
    const auto x = stat.SolveElastic(stat.ZeroState(), load, tide);
    result.startIterations = stat.LastElasticSolve().iterations;
    result.staticU = stat.Surface(x, Component::U);
    return result;
  };

  for (const auto count : {2, 3}) {
    const auto cuts = mesh.BalancedCuts(
        count,
        GridCoupling::TransformWeights(
            body.rheology, lmax, {.eps = eps, .transform_degrees = degrees}));
    ASSERT_EQ(cuts.size(), static_cast<std::size_t>(count - 1));
    const auto options = Viscoelastic3D::Options{
        .eps = eps,
        .parallel = false,
        .max_iterations = 100,
        .rtol = 1e-12,
        .solver = Viscoelastic3D::Options::StepSolver::ConjugateGradients,
        .solve_rtol = 1e-12,
        .moduli = RankModuli(),
        .radial_cuts = cuts,
        .transform_degrees = degrees};
    const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
    const Viscoelastic3D wholeStatic(elastic, lmax, Peaked, options);
    const auto want = run(whole, wholeStatic);

    ThreadRanks ranks(count);
    auto ops = std::vector<std::unique_ptr<Viscoelastic3D>>{};
    auto statics = std::vector<std::unique_ptr<Viscoelastic3D>>{};
    for (auto rank = 0; rank < count; ++rank) {
      auto mine = options;
      mine.segments = {rank};
      mine.collectives = ranks.For(rank);
      ops.push_back(std::make_unique<Viscoelastic3D>(body.rheology, lmax, Peaked, mine));
      statics.push_back(std::make_unique<Viscoelastic3D>(elastic, lmax, Peaked, mine));
    }
    auto nodes = 0;
    for (const auto& op : ops) nodes += op->Coupling().NodeCount();
    EXPECT_EQ(nodes, whole.Coupling().NodeCount());
    auto got = std::vector<Result>(count);
    ranks.Run([&](int rank) { got[rank] = run(*ops[rank], *statics[rank]); });

    auto worstSurface = 0.0, worstMemory = 0.0, worstStatic = 0.0;
    for (auto rank = 0; rank < count; ++rank) {
      EXPECT_EQ(got[rank].iterations, got[0].iterations);
      EXPECT_EQ(got[rank].startIterations, want.startIterations) << rank;
      for (const auto& [l, values] : want.U) {
        worstSurface =
            std::max({worstSurface, Relative(got[rank].U.at(l), values),
                      Relative(got[rank].phi.at(l), want.phi.at(l))});
        worstStatic = std::max(worstStatic,
                               Relative(got[rank].staticU.at(l), want.staticU.at(l)));
      }
      worstMemory = std::max(
          worstMemory, SliceDifference(*ops[rank], got[rank].state, whole, want.state));
    }
    RecordProperty(std::format("ranks{}_surface", count), std::format("{:.3e}", worstSurface));
    RecordProperty(std::format("ranks{}_memory", count), std::format("{:.3e}", worstMemory));
    RecordProperty(std::format("ranks{}_static", count), std::format("{:.3e}", worstStatic));
    EXPECT_LT(worstSurface, 1e-11) << count << " ranks";
    EXPECT_LT(worstMemory, 1e-11) << count << " ranks";
    EXPECT_LT(worstStatic, 1e-11) << count << " ranks";
  }
}

TEST(RadialRanks, LateralModuliAdjointsMatchTheOneProcessSweeps) {
  // The 3D adjoints on three ranks with lateral moduli in the background
  // and the rotational feedback on, continuous
  // and discrete: the ice kernels on every rank, and each rank's slice of
  // the viscosity kernels, against the one process's sweep.
  constexpr auto lmax = 8;
  constexpr auto count = 3;
  const auto body = Body::Make(lmax, 1e22);
  ViscoelasticOperator reference(body.rheology, lmax);
  Fingerprint fingerprint(lmax, reference.ElasticLoveNumbers());
  fingerprint.SetAnalyticTestState(0.0);
  auto solvers = std::vector<SeaLevelSolver>{};
  for (auto rank = 0; rank <= count; ++rank) {
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
  const auto times = std::vector<double>{0.0, 2.5 * kyr, 5.0 * kyr, 7.5 * kyr, 10.0 * kyr};
  auto melt = std::vector<double>(ice.size());
  for (auto i = std::size_t{0}; i < melt.size(); ++i) {
    melt[i] = -0.1 * solvers[0].IceDensity() * (1.0 - state.oceanFunction[i]) * ice[i];
  }
  const auto iceLoad = [&](double t) {
    auto out = melt;
    for (auto& v : out) v *= t / times.back();
    return out;
  };
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual =
      solvers[0].Grid().FromFunction([](double theta, double phi) {
        return std::exp(-2.0 * (1.0 - std::sin(theta) * std::cos(phi - 1.1)));
      });
  measurements[2].seaLevelDual = measurements[times.size() - 1].seaLevelDual;
  const auto cuts = body.material.Mesh().BalancedCuts(
      count, GridCoupling::TransformWeights(body.rheology, lmax, {}));
  ASSERT_EQ(cuts.size(), static_cast<std::size_t>(count - 1));
  const auto options = Viscoelastic3D::Options{
      .parallel = false,
      .rtol = 1e-12,
      .solver = Viscoelastic3D::Options::StepSolver::ConjugateGradients,
      .solve_rtol = 1e-12,
      .moduli = RankModuli(),
      .radial_cuts = cuts};
  const auto evolutionOptions =
      SeaLevelEvolution3D::Options{.rotation = RotationalFeedback::Axial,
                                   .rtol = 1e-12,
                                   .max_coupled_iterations = 100,
                                   .coupled_rtol = 1e-11,
                                   .record_stride = 1};
  for (auto method : {AdjointMethod::Continuous, AdjointMethod::Discrete}) {
    const auto adjointOptions = AdjointSolver3D::Options{
        .method = method, .rtol = 1e-12, .max_coupled_iterations = 100,
        .coupled_rtol = 1e-11};
    auto sweep = [&](const Viscoelastic3D& op, const SeaLevelSolver& solver) {
      SeaLevelEvolution3D evolution(op, solver, state, evolutionOptions);
      const auto snapshots = evolution.Run(times, iceLoad);
      const auto trajectory = ForwardTrajectory3D::FromSnapshots(snapshots);
      AdjointSolver3D adjoint(evolution, adjointOptions);
      return adjoint.IceHistoryKernels(times, measurements, &trajectory);
    };
    const Viscoelastic3D whole(body.rheology, lmax, Peaked, options);
    const auto want = sweep(whole, solvers[count]);

    ThreadRanks ranks(count);
    auto ops = std::vector<std::unique_ptr<Viscoelastic3D>>{};
    for (auto rank = 0; rank < count; ++rank) {
      auto mine = options;
      mine.segments = {rank};
      mine.collectives = ranks.For(rank);
      ops.push_back(std::make_unique<Viscoelastic3D>(body.rheology, lmax, Peaked, mine));
    }
    auto got = std::vector<Kernels3D>(count);
    ranks.Run([&](int rank) { got[rank] = sweep(*ops[rank], solvers[rank]); });

    auto worstIce = 0.0, worstViscosity = 0.0;
    for (auto rank = 0; rank < count; ++rank) {
      const auto& kernels = got[rank];
      ASSERT_EQ(kernels.ice.loadDuals.size(), want.ice.loadDuals.size());
      for (auto n = std::size_t{0}; n < want.ice.loadDuals.size(); ++n) {
        worstIce = std::max(worstIce, Relative(kernels.ice.loadDuals[n],
                                               want.ice.loadDuals[n]));
      }
      const auto& part = ops[rank]->Coupling();
      ASSERT_EQ(kernels.logViscosityKernel3D[0].size(), part.GridValueCount());
      worstViscosity = std::max(
          worstViscosity,
          FieldSliceDifference(part, kernels.logViscosityKernel3D[0],
                               whole.Coupling(), want.logViscosityKernel3D[0]));
    }
    const auto label = method == AdjointMethod::Discrete ? "discrete" : "continuous";
    RecordProperty(std::format("{}_ice", label), std::format("{:.3e}", worstIce));
    RecordProperty(std::format("{}_viscosity", label), std::format("{:.3e}", worstViscosity));
    EXPECT_LT(worstIce, 1e-10) << label;
    EXPECT_LT(worstViscosity, 1e-10) << label;
    std::cout << label << ": ice " << worstIce << ", viscosity " << worstViscosity << '\n';
  }
}
