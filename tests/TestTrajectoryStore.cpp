// The trajectory store (io/TrajectoryStore.hpp, docs/trajectory-store.md):
// every record a 1D and a 3D run hands it (the start, the surface fields,
// the loads and tides, a migrating run's ocean functions, the states, and
// under interior sources the forces, the strains and the toroidal states)
// read back bitwise; the adjoint sweeps from a re-opened store bitwise the
// sweeps over the run in memory, by both methods; and each fault of a
// store refused as its own kind: another tag, version or byte order, an
// incomplete recording, another run's hash, another stride, another
// layout of the processes, another shape, a truncated record, a missing
// one, and files no store writes. An interrupted recording is known as
// one, and a new recording replaces it, or the resumed run continues it
// to the uninterrupted run's store.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/TrajectoryStore.hpp"

namespace {

namespace fs = std::filesystem;
using namespace sl3dpp;
using Store = io::TrajectoryStore;
using Kind = io::TrajectoryStoreError::Kind;
using Complex = std::complex<double>;

constexpr auto kLmax = 6;

double Cap(double theta, double phi, double colat0, double lon0,
           double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

/// The prem8 deck's body with a 1e21 Pa s Maxwell mantle, its elastic
/// sea level solver at lmax 6 and the analytic test state.
struct Scene {
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  GRDSolver::State state;
  std::vector<double> ice;

  static Scene Make() {
    auto deck =
        Deck::Read(
            fs::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"},
                       0))
            .value();
    auto model = EarthModel::FromDeck(deck).value();
    auto mesh = RadialMesh::Create(model, {.lmax = kLmax}).value();
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
    ViscoelasticOperator reference(rheology, kLmax);
    GRDSolver solver(SurfaceGrid(kLmax), reference.ElasticLoveNumbers(),
                     SeaLevelParameters{});
    Fingerprint fingerprint(kLmax, reference.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto ice = toNd(fingerprint.IceThickness());
    auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
    return Scene{std::move(model),  std::move(material), std::move(rheology),
                 std::move(solver), std::move(state),    std::move(ice)};
  }

  [[nodiscard]] double Kyr() const {
    return 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  }
  [[nodiscard]] std::vector<double> Times(int nodes) const {
    auto out = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) out.push_back(0.5 * n * Kyr());
    return out;
  }
  /// A melt of the grounded ice ramped in over the run.
  [[nodiscard]] std::function<std::vector<double>(double)> Melt(
      double span) const {
    auto load = std::vector<double>(ice.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = -0.1 * solver.IceDensity() * (1.0 - state.oceanFunction[i]) *
                ice[i];
    }
    return [load, span](double t) {
      auto out = load;
      for (auto& v : out) v *= t / span;
      return out;
    };
  }
  /// A tide of degree 2 and order 1 growing over the run.
  [[nodiscard]] std::function<std::vector<Complex>(double)> Tide(
      double span) const {
    auto coefficients = solver.Grid().ZeroCoefficients();
    coefficients[SurfaceGrid::Index(2, 1)] = {1e-9, 4e-10};
    return [coefficients, span](double t) {
      auto out = coefficients;
      for (auto& c : out) c *= 0.5 + t / span;
      return out;
    };
  }
  /// A point source under the surface acting from the start.
  [[nodiscard]] std::vector<PointSource> Sources() const {
    const auto& scales = solver.SolverScales();
    const auto unit =
        1e21 / scales.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
    return {{.at = {.radius = solver.Radius() - 300e3 / scales.Length(),
                    .colatitude = 1.1,
                    .longitude = 4.0},
             .moment = {.rr = 0.9 * unit,
                        .tt = -0.2 * unit,
                        .pp = -0.7 * unit,
                        .rt = 0.3 * unit,
                        .rp = -0.7 * unit,
                        .tp = 0.6 * unit}}};
  }
  /// The measurements of the sweeps: a cap of sea level at the last node
  /// and a displacement at a middle one.
  [[nodiscard]] std::map<std::size_t, Measurement> Measurements(
      std::size_t last) const {
    auto out = std::map<std::size_t, Measurement>{};
    out[last].relativeSeaLevelDual = solver.Grid().FromFunction(
        [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
    out[last / 2].displacementDual = solver.Grid().FromFunction(
        [](double t, double p) { return -0.6 * Cap(t, p, 2.2, 3.9, 0.6); });
    return out;
  }
};

/// pyslfp's marine-grounded cap on a uniform 3000 m ocean, melting over
/// the run: a background whose ocean function is flotation's.
struct Marine {
  std::vector<double> ice0;
  GRDSolver::State state;
  double span = 0.0;

  static Marine Make(const Scene& scene, double span) {
    const auto& grid = scene.solver.Grid();
    const auto metre = 1.0 / scene.solver.SolverScales().Length();
    auto out = Marine{};
    out.ice0 = grid.FromFunction([&](double theta, double phi) {
      return 5000.0 * metre * Cap(theta, phi, 0.52, 4.8, 0.2);
    });
    out.state = scene.solver.MakeState(
        out.ice0, std::vector<double>(grid.FieldSize(), 3000.0 * metre));
    out.span = span;
    return out;
  }
  [[nodiscard]] std::vector<double> Thickness(double t) const {
    auto out = ice0;
    for (auto& v : out) v *= std::clamp(1.0 - 0.5 * t / span, 0.0, 1.0);
    return out;
  }
};

/// A run's description for the store, on one process.
Store::Description Describe(const Scene& scene, std::vector<double> times,
                            int stride, int dimension,
                            std::string hash = "the test run") {
  auto out = Store::Description{};
  out.hash = std::move(hash);
  out.dimension = dimension;
  out.times = std::move(times);
  out.record_stride = stride;
  out.field_size = scene.solver.Grid().FieldSize();
  return out;
}

class TrajectoryStore : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("sl3dpp_test_trajectory_" +
            std::string{::testing::UnitTest::GetInstance()
                            ->current_test_info()
                            ->name()});
    fs::remove_all(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
};

// -- the comparisons -------------------------------------------------------------

// Records written and read back, or runs repeated, in one process: bitwise.
void ExpectSame(const GRDSolver::State& got, const GRDSolver::State& want) {
  EXPECT_EQ(got.iceThickness, want.iceThickness);
  EXPECT_EQ(got.seaLevel, want.seaLevel);
  EXPECT_EQ(got.oceanFunction, want.oceanFunction);
  EXPECT_EQ(got.oceanArea, want.oceanArea);
  EXPECT_EQ(got.forcedLand, want.forcedLand);
}

template <class Snapshot>
void ExpectSameSurface(const Snapshot& got, const Snapshot& want,
                       std::size_t node) {
  EXPECT_EQ(got.t, want.t) << node;
  EXPECT_EQ(got.converged, want.converged) << node;
  EXPECT_EQ(got.relativeSeaLevelChange, want.relativeSeaLevelChange) << node;
  EXPECT_EQ(got.displacement, want.displacement) << node;
  EXPECT_EQ(got.potential, want.potential) << node;
  EXPECT_EQ(got.tangentialPotential, want.tangentialPotential) << node;
  EXPECT_EQ(got.toroidalPotential, want.toroidalPotential) << node;
  EXPECT_EQ(got.angularVelocity, want.angularVelocity) << node;
  EXPECT_EQ(got.oceanFunction, want.oceanFunction) << node;
  EXPECT_EQ(got.loadCoefficients, want.loadCoefficients) << node;
  EXPECT_EQ(got.tideCoefficients, want.tideCoefficients) << node;
}

void ExpectSame1D(std::span<const Evolution::Snapshot> got,
                  std::span<const Evolution::Snapshot> want) {
  ASSERT_EQ(got.size(), want.size());
  for (auto n = std::size_t{0}; n < got.size(); ++n) {
    ExpectSameSurface(got[n], want[n], n);
    ASSERT_EQ(got[n].internalState.has_value(),
              want[n].internalState.has_value())
        << n;
    if (want[n].internalState) {
      EXPECT_EQ(got[n].internalState->t, want[n].internalState->t) << n;
      EXPECT_EQ(got[n].internalState->m, want[n].internalState->m) << n;
      EXPECT_EQ(got[n].internalState->columns, want[n].internalState->columns)
          << n;
      EXPECT_EQ(got[n].internalState->strain, want[n].internalState->strain)
          << n;
    }
    ASSERT_EQ(got[n].toroidalState.has_value(),
              want[n].toroidalState.has_value())
        << n;
    if (want[n].toroidalState) {
      EXPECT_EQ(got[n].toroidalState->t, want[n].toroidalState->t) << n;
      EXPECT_EQ(got[n].toroidalState->m, want[n].toroidalState->m) << n;
      EXPECT_EQ(got[n].toroidalState->strain, want[n].toroidalState->strain)
          << n;
    }
    ASSERT_EQ(got[n].sourceForces.has_value(), want[n].sourceForces.has_value())
        << n;
    if (want[n].sourceForces) {
      EXPECT_EQ(got[n].sourceForces->spheroidal,
                want[n].sourceForces->spheroidal)
          << n;
      EXPECT_EQ(got[n].sourceForces->toroidal, want[n].sourceForces->toroidal)
          << n;
    }
  }
}

void ExpectSame(const GridCoupling::TensorField& got,
                const GridCoupling::TensorField& want) {
  EXPECT_EQ(got.c[0], want.c[0]);
  EXPECT_EQ(got.c[1], want.c[1]);
  EXPECT_EQ(got.r, want.r);
  EXPECT_EQ(got.scalars, want.scalars);
}

void ExpectSame(const Viscoelastic3D::State& got,
                const Viscoelastic3D::State& want) {
  EXPECT_EQ(got.t, want.t);
  ASSERT_EQ(got.m.size(), want.m.size());
  for (auto k = std::size_t{0}; k < got.m.size(); ++k) {
    ExpectSame(got.m[k], want.m[k]);
  }
  ASSERT_EQ(got.strain.has_value(), want.strain.has_value());
  if (want.strain) ExpectSame(*got.strain, *want.strain);
  ASSERT_EQ(got.strainCoefficients.has_value(),
            want.strainCoefficients.has_value());
  if (want.strainCoefficients) {
    EXPECT_EQ(got.strainCoefficients->c[0], want.strainCoefficients->c[0]);
    EXPECT_EQ(got.strainCoefficients->c[1], want.strainCoefficients->c[1]);
    EXPECT_EQ(got.strainCoefficients->r, want.strainCoefficients->r);
    EXPECT_EQ(got.strainCoefficients->scalars,
              want.strainCoefficients->scalars);
  }
}

// -- the runs ---------------------------------------------------------------------

/// A 1D run recorded into a store at `directory` and closed: its snapshots.
template <class RunIt>
std::vector<Evolution::Snapshot> Record1D(const fs::path& directory,
                                          const Store::Description& run,
                                          const ViscoelasticOperator& op,
                                          const GRDSolver& solver,
                                          const GRDSolver::State& state,
                                          Evolution::Options options,
                                          RunIt&& runIt) {
  const auto store = Store::Record(directory, run);
  store.PutStart(state);
  store.Attach(options);
  Evolution evolution(op, solver, state, options);
  auto snapshots = runIt(evolution);
  store.Close();
  return snapshots;
}

// -- the round trips --------------------------------------------------------------

TEST_F(TrajectoryStore, RoundTripsEveryRecordOf1DRuns) {
  const auto scene = Scene::Make();
  const ViscoelasticOperator op(scene.rheology, kLmax);
  // frozen, under a load and a tide, every other node kept
  {
    const auto times = scene.Times(5);
    auto run = Describe(scene, times, 2, 1);
    run.tide = true;
    const auto forcing = Evolution::Forcing{
        .directLoad = scene.Melt(times.back()),
        .tide = scene.Tide(times.back())};
    const auto snapshots = Record1D(
        dir_ / "frozen", run, op, scene.solver, scene.state, {.rtol = 1e-11},
        [&](Evolution& e) { return e.Run(times, forcing); });
    ASSERT_TRUE(snapshots[2].internalState.has_value());
    ASSERT_FALSE(snapshots[1].internalState.has_value());
    const auto store = Store::Open(dir_ / "frozen", run);
    EXPECT_TRUE(store.GetManifest().complete);
    EXPECT_EQ(store.GetManifest().kept,
              (std::vector<bool>{true, false, true, false, true}));
    ExpectSame(store.Start(), scene.state);
    ExpectSame1D(store.Snapshots1D(), snapshots);
    EXPECT_EQ(store.StatesRead(), 3u);
  }
  // migrating shorelines: every node's ocean function
  {
    const auto times = scene.Times(4);
    const auto marine = Marine::Make(scene, times.back());
    auto run = Describe(scene, times, 1, 1);
    run.migrating = true;
    const auto snapshots = Record1D(
        dir_ / "migrating", run, op, scene.solver, marine.state,
        {.rtol = 1e-11}, [&](Evolution& e) {
          return e.RunMigrating(
              times, [&](double t) { return marine.Thickness(t); });
        });
    ASSERT_FALSE(snapshots.back().oceanFunction.empty());
    const auto store = Store::Open(dir_ / "migrating", run);
    ExpectSame(store.Start(), marine.state);
    ExpectSame1D(store.Snapshots1D(), snapshots);
  }
  // under a point source, without an ocean: every node kept with its
  // strain, its toroidal state and the source's forces
  {
    const auto times = scene.Times(3);
    auto run = Describe(scene, times, 1, 1);
    run.sources = true;
    const auto sources = scene.Sources();
    const auto forcing = Evolution::Forcing{
        .sources = [&](double) { return sources; }};
    const auto dry = scene.solver.NoOceanState();
    const auto snapshots = Record1D(
        dir_ / "sourced", run, op, scene.solver, dry,
        {.rotation = RotationalFeedback::None, .rtol = 1e-11},
        [&](Evolution& e) { return e.Run(times, forcing); });
    ASSERT_TRUE(snapshots[1].toroidalState.has_value());
    ASSERT_TRUE(snapshots[1].sourceForces.has_value());
    ASSERT_TRUE(snapshots[1].internalState->HasStrain());
    const auto store = Store::Open(dir_ / "sourced", run);
    ExpectSame(store.Start(), dry);
    ExpectSame1D(store.Snapshots1D(), snapshots);
  }
}

TEST_F(TrajectoryStore, RoundTripsEveryRecordOf3DRuns) {
  const auto scene = Scene::Make();
  const Viscoelastic3D op(
      scene.rheology, kLmax,
      [](double, double theta, double) { return 0.3 * std::cos(theta); });
  // run through the store's hooks, each state handed out copied aside
  const auto record = [&](const fs::path& directory,
                          const Store::Description& run,
                          const GRDSolver::State& state,
                          Evolution3D::Options options, auto&& runIt,
                          std::map<std::size_t, Viscoelastic3D::State>& kept) {
    const auto store = Store::Record(directory, run);
    store.PutStart(state);
    store.Attach(options);
    options.on_state = [&kept, put = options.on_state](
                           std::size_t node, Viscoelastic3D::State&& s) {
      kept.emplace(node, s);
      put(node, std::move(s));
    };
    Evolution3D evolution(op, scene.solver, state, options);
    auto snapshots = runIt(evolution);
    store.Close();
    return snapshots;
  };
  // frozen, under a load and a tide, every other node kept
  {
    const auto times = scene.Times(5);
    auto run = Describe(scene, times, 2, 3);
    run.tide = true;
    run.part = Store::PartOf(op);
    const auto forcing = Evolution3D::Forcing{
        .directLoad = scene.Melt(times.back()),
        .tide = scene.Tide(times.back())};
    auto kept = std::map<std::size_t, Viscoelastic3D::State>{};
    const auto snapshots = record(
        dir_ / "frozen", run, scene.state, {.rtol = 1e-11},
        [&](Evolution3D& e) { return e.Run(times, forcing); }, kept);
    ASSERT_EQ(kept.size(), 3u);
    for (const auto& snapshot : snapshots) {
      EXPECT_FALSE(snapshot.internalState.has_value());
    }
    const auto store = Store::Open(dir_ / "frozen", run);
    ExpectSame(store.Start(), scene.state);
    const auto read = store.Snapshots3D();
    ASSERT_EQ(read.size(), snapshots.size());
    for (auto n = std::size_t{0}; n < read.size(); ++n) {
      ExpectSameSurface(read[n], snapshots[n], n);
      EXPECT_EQ(read[n].coupledIterations, snapshots[n].coupledIterations);
      EXPECT_EQ(read[n].coupledResidual, snapshots[n].coupledResidual);
      EXPECT_EQ(read[n].solveIterations, snapshots[n].solveIterations);
      EXPECT_FALSE(read[n].sourceForces.has_value());
    }
    for (const auto& [node, state] : kept) ExpectSame(store.State3D(node), state);
    EXPECT_EQ(store.StatesRead(), 3u);
    const auto trajectory = store.Trajectory3D(read);
    EXPECT_TRUE(trajectory.Kept(2));
    EXPECT_FALSE(trajectory.Kept(3));
    ExpectSame(trajectory.fetch(4), kept.at(4));
  }
  // under a point source, without an ocean: every node's state with its
  // strain, and the source's forces
  {
    const auto times = scene.Times(3);
    auto run = Describe(scene, times, 1, 3);
    run.sources = true;
    run.part = Store::PartOf(op);
    const auto sources = scene.Sources();
    const auto forcing = Evolution3D::Forcing{
        .sources = [&](double) { return sources; }};
    const auto dry = scene.solver.NoOceanState();
    auto kept = std::map<std::size_t, Viscoelastic3D::State>{};
    const auto snapshots = record(
        dir_ / "sourced", run, dry,
        {.rotation = RotationalFeedback::None, .rtol = 1e-11},
        [&](Evolution3D& e) { return e.Run(times, forcing); }, kept);
    ASSERT_EQ(kept.size(), 3u);
    ASSERT_TRUE(kept.at(1).strain.has_value() ||
                kept.at(1).strainCoefficients.has_value());
    ASSERT_TRUE(snapshots[1].sourceForces.has_value());
    const auto store = Store::Open(dir_ / "sourced", run);
    const auto read = store.Snapshots3D();
    for (auto n = std::size_t{0}; n < read.size(); ++n) {
      ExpectSameSurface(read[n], snapshots[n], n);
      ASSERT_TRUE(read[n].sourceForces.has_value()) << n;
      EXPECT_EQ(read[n].sourceForces->spheroidal,
                snapshots[n].sourceForces->spheroidal);
      EXPECT_EQ(read[n].sourceForces->toroidal,
                snapshots[n].sourceForces->toroidal);
      ExpectSame(store.State3D(n), kept.at(n));
    }
  }
}

// -- the sweeps -------------------------------------------------------------------

TEST_F(TrajectoryStore, SweepsFromAReopenedStoreAreTheSweepsInMemory) {
  const auto scene = Scene::Make();
  const auto times = scene.Times(7);
  const auto measurements = scene.Measurements(times.size() - 1);
  // 1D, under a load and a tide, every third node kept
  {
    const ViscoelasticOperator op(scene.rheology, kLmax);
    const auto forcing = Evolution::Forcing{
        .directLoad = scene.Melt(times.back()),
        .tide = scene.Tide(times.back())};
    const auto options = Evolution::Options{.rtol = 1e-11, .record_stride = 3};
    Evolution memory(op, scene.solver, scene.state, options);
    const auto snapshots = memory.Run(times, forcing);
    const auto inMemory = ForwardTrajectory::FromSnapshots(snapshots);
    auto run = Describe(scene, times, 3, 1);
    run.tide = true;
    static_cast<void>(Record1D(dir_ / "one", run, op, scene.solver,
                               scene.state, options,
                               [&](Evolution& e) { return e.Run(times, forcing); }));
    const auto store = Store::Open(dir_ / "one", run);
    const auto read = store.Snapshots1D();
    const auto fromDisk = ForwardTrajectory::FromSnapshots(read);
    Evolution reader(op, scene.solver, store.Start(), options);
    for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto adjointOptions = AdjointSolver::Options{
          .method = method, .rtol = 1e-11, .material_kernels = true,
          .tide_kernels = true};
      const auto a = AdjointSolver(memory, adjointOptions)
                         .FirstOrderKernels(times, measurements, &inMemory);
      const auto b = AdjointSolver(reader, adjointOptions)
                         .FirstOrderKernels(times, measurements, &fromDisk);
      const auto label =
          method == AdjointMethod::Discrete ? "discrete" : "continuous";
      // one process, the same transforms over the same states: bitwise
      EXPECT_EQ(a.thickness, b.thickness) << label;
      EXPECT_EQ(a.directLoadKernels, b.directLoadKernels) << label;
      EXPECT_EQ(a.tideKernels, b.tideKernels) << label;
      EXPECT_EQ(a.itauKernel, b.itauKernel) << label;
      EXPECT_EQ(a.kappaKernel, b.kappaKernel) << label;
      EXPECT_EQ(a.muKernel, b.muKernel) << label;
      EXPECT_FALSE(a.itauKernel.empty());
    }
  }
  // 3D, under a load, every third node kept, the states fetched from disk
  {
    const Viscoelastic3D op(
        scene.rheology, kLmax,
        [](double, double theta, double) { return 0.3 * std::cos(theta); });
    const auto forcing =
        Evolution3D::Forcing{.directLoad = scene.Melt(times.back())};
    auto options = Evolution3D::Options{.rtol = 1e-11, .record_stride = 3};
    options.coupled_rtol = 1e-11;
    Evolution3D memory(op, scene.solver, scene.state, options);
    const auto snapshots = memory.Run(times, forcing);
    const auto inMemory = ForwardTrajectory3D::FromSnapshots(snapshots);
    auto run = Describe(scene, times, 3, 3);
    run.part = Store::PartOf(op);
    {
      const auto store = Store::Record(dir_ / "three", run);
      store.PutStart(scene.state);
      auto recording = options;
      store.Attach(recording);
      Evolution3D evolution(op, scene.solver, scene.state, recording);
      static_cast<void>(evolution.Run(times, forcing));
      store.Close();
    }
    const auto store = Store::Open(dir_ / "three", run);
    const auto read = store.Snapshots3D();
    const auto fromDisk = store.Trajectory3D(read);
    Evolution3D reader(op, scene.solver, store.Start(), options);
    for (const auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      auto adjointOptions = AdjointSolver3D::Options{};
      adjointOptions.method = method;
      adjointOptions.rtol = 1e-11;
      adjointOptions.coupled_rtol = 1e-11;
      const auto a = AdjointSolver3D(memory, adjointOptions)
                         .FirstOrderKernels(times, measurements, &inMemory);
      const auto b = AdjointSolver3D(reader, adjointOptions)
                         .FirstOrderKernels(times, measurements, &fromDisk);
      const auto label =
          method == AdjointMethod::Discrete ? "discrete" : "continuous";
      // one process, the same transforms over the same states: bitwise
      EXPECT_EQ(a.ice.thickness, b.ice.thickness) << label;
      EXPECT_EQ(a.logViscosityKernel3D, b.logViscosityKernel3D) << label;
      EXPECT_EQ(a.itauKernelSpherical, b.itauKernelSpherical) << label;
      EXPECT_FALSE(a.logViscosityKernel3D.empty());
    }
    EXPECT_GT(store.StatesRead(), 0u);
  }
}

// -- the refusals ------------------------------------------------------------------

/// A small 1D run recorded into `directory` (three nodes, every one kept).
void RecordSmall(const Scene& scene, const fs::path& directory,
                 const Store::Description& run) {
  const ViscoelasticOperator op(scene.rheology, kLmax);
  static_cast<void>(Record1D(
      directory, run, op, scene.solver, scene.state, {.rtol = 1e-10},
      [&](Evolution& e) { return e.Run(run.times, scene.Melt(run.times.back())); }));
}

/// The kind of the store error `f` throws, or nothing.
template <class F>
std::optional<Kind> KindOf(F&& f) {
  try {
    f();
  } catch (const io::TrajectoryStoreError& error) {
    return error.GetKind();
  }
  return std::nullopt;
}

/// Overwrites bytes of a file in place.
void Patch(const fs::path& path, std::size_t offset, const void* bytes,
           std::size_t count) {
  auto file = std::fstream(path, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(static_cast<std::streamoff>(offset));
  file.write(static_cast<const char*>(bytes), static_cast<std::streamsize>(count));
}

TEST_F(TrajectoryStore, RefusesEachFaultAsItsOwnKind) {
  const auto scene = Scene::Make();
  const auto run = Describe(scene, scene.Times(3), 1, 1);
  const auto fresh = [&](const std::string& name) {
    const auto directory = dir_ / name;
    RecordSmall(scene, directory, run);
    EXPECT_EQ(Store::Survey(directory, run), Store::Standing::Complete);
    return directory;
  };
  const auto opened = [&](const fs::path& directory,
                          const Store::Description& as) {
    return KindOf([&] { static_cast<void>(Store::Open(directory, as)); });
  };
  const auto manifest = [](const fs::path& directory) {
    return directory / "trajectory.manifest";
  };
  // absent or empty: nothing to read
  EXPECT_EQ(Store::Survey(dir_ / "absent", run), Store::Standing::Empty);
  fs::create_directories(dir_ / "empty");
  EXPECT_EQ(Store::Survey(dir_ / "empty", run), Store::Standing::Empty);
  EXPECT_EQ(opened(dir_ / "empty", run), Kind::Missing);
  // the manifest's tag, byte order and version
  {
    const auto directory = fresh("tag");
    Patch(manifest(directory), 0, "NOTASTOR", 8);
    EXPECT_EQ(opened(directory, run), Kind::Tag);
  }
  {
    const auto directory = fresh("order");
    const auto swapped = std::byteswap(std::uint64_t{0x0102030405060708});
    Patch(manifest(directory), 8, &swapped, 8);
    EXPECT_EQ(opened(directory, run), Kind::ByteOrder);
  }
  {
    const auto directory = fresh("version");
    const auto version = std::uint64_t{2};
    Patch(manifest(directory), 16, &version, 8);
    EXPECT_EQ(opened(directory, run), Kind::Version);
  }
  // another run's hash, another stride, other processes, another shape
  {
    const auto directory = fresh("other");
    auto other = run;
    other.hash = "another run";
    EXPECT_EQ(opened(directory, other), Kind::Hash);
    EXPECT_EQ(KindOf([&] { static_cast<void>(Store::Survey(directory, other)); }),
              Kind::Hash);
    // a recording of another run is refused too, and its files kept
    EXPECT_EQ(KindOf([&] { static_cast<void>(Store::Record(directory, other)); }),
              Kind::Hash);
    EXPECT_TRUE(fs::exists(manifest(directory)));
    auto strided = run;
    strided.record_stride = 2;
    EXPECT_EQ(opened(directory, strided), Kind::Stride);
    auto cut = run;
    cut.cuts = {3};
    cut.part = {.first_segment = 0, .segments = 2, .grid_values = 0};
    EXPECT_EQ(opened(directory, cut), Kind::Layout);
    auto later = run;
    later.times.back() *= 2.0;
    EXPECT_EQ(opened(directory, later), Kind::Mismatch);
    auto threeD = run;
    threeD.dimension = 3;
    EXPECT_EQ(opened(directory, threeD), Kind::Mismatch);
  }
  // recorded over two processes (collectives that stand in for them),
  // read over one
  {
    auto two = run;
    two.ranks = 2;
    two.collectives = Collectives{.sum = [](std::span<double>) {},
                                  .max = [](std::span<double>) {},
                                  .share = [](std::span<double>) {}};
    RecordSmall(scene, dir_ / "ranks", two);
    EXPECT_EQ(Store::ManifestOf(dir_ / "ranks").parts.size(), 2u);
    EXPECT_EQ(opened(dir_ / "ranks", run), Kind::Layout);
  }
  // a truncated record, a record whose header is another's, a missing one
  {
    const auto directory = fresh("truncated");
    const auto surface = directory / "node_000001.surface";
    fs::resize_file(surface, fs::file_size(surface) - 8);
    EXPECT_EQ(opened(directory, run), Kind::Truncated);
  }
  {
    const auto directory = fresh("header");
    const auto node = std::uint64_t{2};
    Patch(directory / "node_000001.surface", 16, &node, 8);
    const auto store = Store::Open(directory, run);
    EXPECT_EQ(KindOf([&] { static_cast<void>(store.Snapshots1D()); }),
              Kind::Truncated);
  }
  {
    const auto directory = fresh("missing");
    fs::remove(directory / "rank_0000" / "node_000002.state");
    EXPECT_EQ(opened(directory, run), Kind::Missing);
    fs::remove(directory / "node_000000.surface");
    EXPECT_EQ(opened(directory, run), Kind::Missing);
  }
  // files no store writes, and records without a manifest
  {
    const auto directory = fresh("foreign");
    std::ofstream(directory / "notes.txt") << "not a record";
    EXPECT_EQ(KindOf([&] { static_cast<void>(Store::Survey(directory, run)); }),
              Kind::Foreign);
    EXPECT_EQ(KindOf([&] { static_cast<void>(Store::Record(directory, run)); }),
              Kind::Foreign);
    EXPECT_TRUE(fs::exists(directory / "notes.txt"));
    fs::remove(directory / "notes.txt");
    std::ofstream(directory / "rank_0000" / "node_000000.traj") << "spilled";
    EXPECT_EQ(opened(directory, run), Kind::Foreign);
    fs::remove(directory / "rank_0000" / "node_000000.traj");
    fs::remove(manifest(directory));
    EXPECT_EQ(opened(directory, run), Kind::Foreign);
  }
}

TEST_F(TrajectoryStore, AnInterruptedRecordingIsKnownAndReplaced) {
  const auto scene = Scene::Make();
  const auto run = Describe(scene, scene.Times(3), 1, 1);
  const auto directory = dir_ / "store";
  // a recording that stops after its first node: no Close
  {
    const ViscoelasticOperator op(scene.rheology, kLmax);
    const auto store = Store::Record(directory, run);
    store.PutStart(scene.state);
    auto options = Evolution::Options{.rtol = 1e-10};
    store.Attach(options);
    Evolution evolution(op, scene.solver, scene.state, options);
    static_cast<void>(evolution.Run(std::vector<double>{run.times[0]},
                                    scene.Melt(run.times.back())));
  }
  EXPECT_TRUE(fs::exists(directory / "node_000000.surface"));
  EXPECT_FALSE(Store::ManifestOf(directory).complete);
  EXPECT_EQ(Store::Survey(directory, run), Store::Standing::Incomplete);
  EXPECT_EQ(KindOf([&] { static_cast<void>(Store::Open(directory, run)); }),
            Kind::Incomplete);
  // another run's incomplete store is another run's still
  auto other = run;
  other.hash = "another run";
  EXPECT_EQ(KindOf([&] { static_cast<void>(Store::Survey(directory, other)); }),
            Kind::Hash);
  // a recording of the run replaces it, and a complete one is not recorded
  // over
  RecordSmall(scene, directory, run);
  EXPECT_EQ(Store::Survey(directory, run), Store::Standing::Complete);
  EXPECT_THROW(static_cast<void>(Store::Record(directory, run)),
               std::logic_error);
  const auto store = Store::Open(directory, run);
  EXPECT_EQ(store.Snapshots1D().size(), 3u);
  EXPECT_GT(store.Bytes(), 0u);
}

TEST_F(TrajectoryStore, AnInterruptedRecordingIsContinuedFromItsCheckpoint) {
  // A run stopped after the checkpoint of node 2 (of five, every second
  // kept) leaves an incomplete store with its records to that node, and a
  // record past it; the resumed run, its evolution attached to the store
  // continued from node 2, records the rest, the records past the node
  // replaced: the store is, file by file, the uninterrupted run's. A store
  // short of a record to the node is refused as incomplete, nothing
  // removed; an empty directory, or a store recorded at another stride,
  // is refused too.
  const auto scene = Scene::Make();
  const ViscoelasticOperator op(scene.rheology, kLmax);
  const auto run = Describe(scene, scene.Times(5), 2, 1);
  const auto melt = scene.Melt(run.times.back());
  RecordSmall(scene, dir_ / "whole", run);
  // the interrupted run: its checkpoint at node 2, its records to node 3
  const auto stopped = [&](const fs::path& directory) {
    auto checkpoint = std::optional<Evolution::Checkpoint>{};
    const auto store = Store::Record(directory, run);
    store.PutStart(scene.state);
    auto options = Evolution::Options{.rtol = 1e-10};
    options.checkpoint_stride = 1;
    options.on_checkpoint = [&](std::size_t node,
                                const Evolution::Checkpoint& at) {
      if (node == 2) checkpoint = at;
    };
    store.Attach(options);
    Evolution evolution(op, scene.solver, scene.state, options);
    static_cast<void>(evolution.Run(
        std::vector<double>(run.times.begin(), run.times.begin() + 4), melt));
    return *checkpoint;
  };
  const auto directory = dir_ / "parts";
  const auto checkpoint = stopped(directory);
  EXPECT_TRUE(fs::exists(directory / "node_000003.surface"));
  std::ofstream(directory / ".node_000004.surface.partial") << "half";
  {
    const auto store = Store::Continue(directory, run, 2);
    EXPECT_FALSE(fs::exists(directory / "node_000003.surface"));
    EXPECT_FALSE(fs::exists(directory / ".node_000004.surface.partial"));
    EXPECT_TRUE(fs::exists(directory / "node_000002.surface"));
    auto options = Evolution::Options{.rtol = 1e-10};
    store.Attach(options);
    EXPECT_EQ(options.record_offset, 2u);
    Evolution evolution(op, scene.solver, scene.state, options);
    static_cast<void>(evolution.Resume(
        checkpoint, std::vector<double>(run.times.begin() + 2, run.times.end()),
        melt));
    store.Close();
  }
  EXPECT_EQ(Store::Survey(directory, run), Store::Standing::Complete);
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
  // the resumed run is the uninterrupted one in one process: byte for byte
  EXPECT_TRUE(bytes(directory) == bytes(dir_ / "whole"));
  // short of a record to the node: refused, nothing removed
  const auto shorter = dir_ / "short";
  static_cast<void>(stopped(shorter));
  fs::remove(shorter / "rank_0000" / "node_000002.state");
  EXPECT_EQ(KindOf([&] {
              static_cast<void>(Store::Continue(shorter, run, 2));
            }),
            Kind::Incomplete);
  EXPECT_TRUE(fs::exists(shorter / "node_000003.surface"));
  // nothing to continue, or a store of another stride
  EXPECT_EQ(KindOf([&] {
              static_cast<void>(Store::Continue(dir_ / "none", run, 2));
            }),
            Kind::Missing);
  auto strided = run;
  strided.record_stride = 1;
  EXPECT_EQ(KindOf([&] {
              static_cast<void>(Store::Continue(shorter, strided, 2));
            }),
            Kind::Stride);
}

}  // namespace
