// The adjoint sweeps: the discrete gradient identity at solver
// tolerance for a full measurement set, the thickness-view identity with
// the baseline, the continuous adjoint's agreement with the discrete and
// its convergence under step refinement, and the cross-validation of the
// delta amplitudes — the structure of pyslfp's
// test_continuous_adjoint_against_the_discrete; the continuous sweep
// stepped by Sdirk2's stages against finite differences of its forward
// runs, and the derivatives not built for it refused.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numbers>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/PointSource.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/ShConventions.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

namespace {

/// With the memory points at the GLL nodes, as in these tests, the points
/// are the nodes: a nodal direction as one for the time-stepped problem,
/// and back.
[[maybe_unused]] PointMaterialDirection AtGll(const MaterialDirection& d) {
  return {.kappa = d.kappa, .mu = d.mu};
}
[[maybe_unused]] MaterialDirection Nodal(const PointMaterialDirection& d) {
  return {.kappa = d.kappa, .mu = d.mu};
}

constexpr auto kLmax = 16;

/// The smooth cap of the solver tests (TestSeaLevel.cpp), the kernel
/// fixture's measurement formula.
double Cap(double theta, double phi, double colat0, double lon0,
           double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

struct Scene {
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  GRDSolver::State state;

  /// A viscous mantle with the given viscosity (Pa s).
  static Scene Make(double viscositySI) {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
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
        viscosity.emplace(i,
                          viscositySI / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();

    ViscoelasticOperator op(rheology, kLmax);
    GRDSolver solver(SurfaceGrid(kLmax), op.ElasticLoveNumbers(),
                          SeaLevelParameters{});

    Fingerprint fingerprint(kLmax, op.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto state = solver.MakeState(toNd(fingerprint.IceThickness()),
                                  toNd(fingerprint.SeaLevel()));
    return Scene{std::move(model), std::move(material), std::move(rheology),
                 std::move(solver), std::move(state)};
  }

  /// A smooth random band-limited field of the given amplitude.
  [[nodiscard]] std::vector<double> Smooth(std::mt19937& rng,
                                           double amplitude) const {
    const auto& grid = solver.Grid();
    std::normal_distribution<double> normal;
    auto coeffs = grid.ZeroCoefficients();
    for (auto l = 0; l <= kLmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        coeffs[SurfaceGrid::Index(l, m)] = {normal(rng),
                                            m == 0 ? 0.0 : normal(rng)};
      }
    }
    auto field = grid.Synthesise(coeffs);
    for (auto& v : field) v *= amplitude;
    return field;
  }

  [[nodiscard]] double Surface(std::span<const double> f,
                               std::span<const double> g) const {
    const auto a = solver.Radius();
    auto fg = std::vector<double>(f.size());
    for (auto i = std::size_t{0}; i < f.size(); ++i) fg[i] = f[i] * g[i];
    return a * a * solver.Grid().Integrate(fg);
  }
};

/// The objective the measurements define, evaluated on a forward run.
double Objective(const Scene& setup,
                 const std::vector<Evolution::Snapshot>& snapshots,
                 const std::map<std::size_t, Measurement>& measurements) {
  auto total = 0.0;
  for (const auto& [node, m] : measurements) {
    const auto& snap = snapshots[node];
    if (!m.relativeSeaLevelDual.empty()) {
      total += setup.Surface(m.relativeSeaLevelDual, snap.relativeSeaLevelChange);
    }
    if (!m.displacementDual.empty()) {
      total += setup.Surface(m.displacementDual, snap.displacement);
    }
    if (!m.potentialDual.empty()) {
      total += setup.Surface(m.potentialDual, snap.potential);
    }
    for (auto j = 0; j < 3; ++j) {
      total += m.angularVelocityDual[j] * snap.angularVelocity[j];
    }
  }
  return total;
}

/// A per-node load history as the forward driver's callable.
std::function<std::vector<double>(double)> History(
    const std::vector<double>& times,
    const std::vector<std::vector<double>>& loads) {
  return [&times, &loads](double t) {
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) {
        return loads[n];
      }
    }
    throw std::runtime_error("the test history is read off its nodes");
  };
}

}  // namespace

TEST(Adjoint, DiscreteGradientIdentity) {
  // The exact transpose: delta J = sum_n <directLoadKernels[n], delta f_n> to the
  // solvers' tolerance, for a full measurement set (all four duals), both
  // schemes. The ice history enters linearly, so J(delta) is the exact
  // directional derivative.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(7);

  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);

  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 / sigma));
  }

  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].relativeSeaLevelDual = setup.Smooth(rng, 0.3);
  Measurement middle;
  middle.relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  middle.displacementDual = setup.Smooth(rng, -0.5);
  measurements[3] = std::move(middle);
  Measurement last;
  last.displacementDual = setup.Smooth(rng, 0.7);
  last.potentialDual = setup.Smooth(rng, 0.4);
  last.angularVelocityDual = {0.2, -0.4, 0.8};
  measurements[5] = std::move(last);

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(op, setup.solver, setup.state,
                                {.scheme = scheme, .rtol = 1e-12});
    auto snapshots = evolution.Run(times, History(times, loads));
    const auto exact = Objective(setup, snapshots, measurements);

    AdjointSolver adjoint(evolution,
                          {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto kernels = adjoint.FirstOrderKernels(times, measurements);
    auto predicted = 0.0;
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      predicted += setup.Surface(kernels.directLoadKernels[n], loads[n]);
    }
    EXPECT_NEAR(predicted, exact, 1e-8 * std::abs(exact))
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(Adjoint, ThicknessViewCarriesTheBaseline) {
  // The thickness duals: with the forward load f_n = rho_i (1 - C)
  // (I_n - I_0), delta J = sum_n <thickness[n], delta I_n> — the
  // initial-thickness dependence folded into node 0. A uniform-in-time
  // thickness change loads nothing and pairs to zero by construction.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(11);

  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 4; ++n) times.push_back(2.5 * n * kyr);

  const auto length = scales.Length();
  auto thicknesses = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    thicknesses.push_back(setup.Smooth(rng, 10.0 / length));
  }
  const auto& C = setup.state.oceanFunction;
  const auto rhoI = setup.solver.IceDensity();
  auto groundedLoads = [&](const std::vector<std::vector<double>>& I) {
    auto loads = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      auto load = std::vector<double>(I[n].size());
      for (auto i = std::size_t{0}; i < load.size(); ++i) {
        load[i] = rhoI * (1.0 - C[i]) * (I[n][i] - I[0][i]);
      }
      loads.push_back(std::move(load));
    }
    return loads;
  };
  const auto loads = groundedLoads(thicknesses);

  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[2].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[4].relativeSeaLevelDual = setup.Smooth(rng, -0.6);

  ViscoelasticOperator op(setup.rheology, kLmax);
  Evolution evolution(op, setup.solver, setup.state, {.rtol = 1e-12});
  auto snapshots = evolution.Run(times, History(times, loads));
  const auto exact = Objective(setup, snapshots, measurements);

  AdjointSolver adjoint(evolution,
                        {.method = AdjointMethod::Discrete, .rtol = 1e-12});
  const auto kernels = adjoint.FirstOrderKernels(times, measurements);
  auto predicted = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    predicted += setup.Surface(kernels.thickness[n], thicknesses[n]);
  }
  EXPECT_NEAR(predicted, exact, 1e-8 * std::abs(exact));

  // A uniform shift of the whole history pairs to zero: the node
  // thickness duals and the baseline cancel identically.
  auto ones = std::vector<double>(setup.solver.Grid().FieldSize(), 1.0);
  auto shift = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    shift += setup.Surface(kernels.thickness[n], ones);
  }
  auto scale = 0.0;
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    scale = std::max(scale, std::abs(setup.Surface(kernels.thickness[n],
                                                   thicknesses[n])));
  }
  EXPECT_LT(std::abs(shift) * 10.0 / length, 1e-10 * scale);
}

TEST(Adjoint, ContinuousAgreesWithDiscreteAndConverges) {
  // The comparison (docs/adjoints.md): the discrete
  // transpose is exact at any step; the continuous adjoint carries the
  // discretise/transpose commute error — at the per-cent level or better
  // once the steps resolve the relaxation, falling under refinement.
  // Their delta amplitudes agree although computed by different
  // realisations (the engine transpose against the forward channel
  // mapping), cross-validating both.
  std::mt19937 rng(4);

  auto gradientErrors = [&](Scene& setup, std::size_t npts, double T,
                            std::map<AdjointMethod, Kernels>* kernelsOut =
                                nullptr) {
    const auto& scales = setup.solver.SolverScales();
    const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(T * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * kyr);
    }
    const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
    std::mt19937 local(17);  // the same loads at every resolution
    auto loads = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      loads.push_back(setup.Smooth(local, 100.0 / sigma));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts / 2].relativeSeaLevelDual = setup.Smooth(local, 1.0);
    measurements[npts - 1].relativeSeaLevelDual = setup.Smooth(local, 0.5);

    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(op, setup.solver, setup.state,
                                {.rtol = 1e-12});
    auto snapshots = evolution.Run(times, History(times, loads));
    const auto exact = Objective(setup, snapshots, measurements);

    auto errors = std::map<AdjointMethod, double>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      AdjointSolver adjoint(evolution, {.method = method, .rtol = 1e-12});
      auto kernels = adjoint.FirstOrderKernels(times, measurements);
      auto predicted = 0.0;
      for (auto n = std::size_t{0}; n < npts; ++n) {
        predicted += setup.Surface(kernels.directLoadKernels[n], loads[n]);
      }
      errors[method] = std::abs(predicted - exact) / std::abs(exact);
      if (kernelsOut) kernelsOut->emplace(method, std::move(kernels));
    }
    return errors;
  };

  // The stiff regime: 2.5 kyr steps over several 450-yr relaxation times.
  auto stiff = Scene::Make(1e21);
  auto kernels = std::map<AdjointMethod, Kernels>{};
  auto coarse = gradientErrors(stiff, 5, 10.0, &kernels);
  auto fine = gradientErrors(stiff, 9, 10.0);
  EXPECT_LT(coarse[AdjointMethod::Discrete], 1e-8);
  EXPECT_LT(fine[AdjointMethod::Discrete], 1e-8);
  EXPECT_LT(coarse[AdjointMethod::Continuous], 0.5);
  EXPECT_GT(coarse[AdjointMethod::Continuous] /
                fine[AdjointMethod::Continuous],
            1.8);  // converging

  // The delta amplitudes cross-validate the two realisations.
  for (const auto& [node, a] : kernels[AdjointMethod::Discrete].instantaneous) {
    const auto& b = kernels[AdjointMethod::Continuous].instantaneous.at(node);
    auto top = 0.0, diff = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      top = std::max(top, std::abs(a[i]));
      diff = std::max(diff, std::abs(a[i] - b[i]));
    }
    EXPECT_LT(diff, 1e-7 * top) << "node " << node;
  }

  // The resolved regime: relaxation ten times slower; the commute error
  // is small and falls under refinement.
  auto resolved = Scene::Make(1e22);
  auto coarseR = gradientErrors(resolved, 5, 10.0);
  auto fineR = gradientErrors(resolved, 9, 10.0);
  EXPECT_LT(coarseR[AdjointMethod::Discrete], 1e-8);
  EXPECT_LT(coarseR[AdjointMethod::Continuous], 5e-2);
  EXPECT_GT(coarseR[AdjointMethod::Continuous] /
                fineR[AdjointMethod::Continuous],
            2.0);
}

TEST(Adjoint, ViscosityKernelsAgainstFiniteDifferences) {
  // The discrete inverse-relaxation-time kernels: the reverse sweep given
  // the recorded trajectory pairs the scheme-weight derivatives with the
  // duals, and the FD residual against a smooth nodal itau perturbation
  // falls at second order in eps — the kernel itself is exact (Richardson
  // extrapolation lands on it to solver precision). Both implicit
  // schemes, and the viscosity mapping -itau^2 / mu.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(7);

  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 / sigma));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);

  // A smooth nodal perturbation of the branch's inverse relaxation time,
  // in material units, supported where the branch lives.
  const auto& base = setup.rheology.BranchAt(0);
  const auto& mesh = setup.material.Mesh();
  auto itauTop = 0.0;
  for (auto v : base.itau) itauTop = std::max(itauTop, v);
  auto ditau = std::vector<double>(base.itau.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      if (base.itau[i] > 0.0) {
        const auto r = mesh.Radius(e, q);
        ditau[i] = (0.3 + 0.7 * r * r) * itauTop;
      }
    }
  }

  auto functional = [&](double eps, Scheme scheme) {
    auto branch = base;
    for (auto i = std::size_t{0}; i < branch.itau.size(); ++i) {
      branch.itau[i] += eps * ditau[i];
    }
    auto rheology = Rheology::Create(setup.material, {std::move(branch)}).value();
    ViscoelasticOperator op(rheology, kLmax);
    Evolution evolution(op, setup.solver, setup.state,
                                {.scheme = scheme, .rtol = 1e-12});
    auto snapshots = evolution.Run(times, History(times, loads));
    return Objective(setup, snapshots, measurements);
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(
        op, setup.solver, setup.state,
        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    auto snapshots = evolution.Run(times, History(times, loads));
    const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);

    AdjointSolver adjoint(evolution,
                          {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto kernels =
        adjoint.FirstOrderKernels(times, measurements, &trajectory);
    ASSERT_FALSE(kernels.itauKernel.empty());
    const auto predicted = VolumePairing(mesh, kernels.itauKernel[0], ditau);

    auto fd = std::map<double, double>{};
    for (double eps : {0.02, 0.01}) {
      fd[eps] =
          (functional(eps, scheme) - functional(-eps, scheme)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.02] - predicted);
    const auto r2 = std::abs(fd[0.01] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme);
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0;
    // The identity to FD/solver noise: the directional derivative is
    // five orders below J, so the difference quotient keeps ~6 digits.
    EXPECT_NEAR(richardson, predicted, 1e-5 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme);

    // The viscosity mapping, and the refusal without a trajectory.
    const auto eta = kernels.Viscosity(setup.rheology);
    for (auto i = std::size_t{0}; i < ditau.size(); ++i) {
      if (base.mu[i] > 0.0) {
        EXPECT_NEAR(eta[0][i],
                    -(base.itau[i] * base.itau[i]) / base.mu[i] *
                        kernels.itauKernel[0][i],
                    1e-14 * std::abs(kernels.itauKernel[0][i]) + 1e-300);
      }
    }
    const auto bare = adjoint.FirstOrderKernels(times, measurements);
    EXPECT_TRUE(bare.itauKernel.empty());
    EXPECT_THROW((void)bare.Viscosity(setup.rheology), std::invalid_argument);
  }
}

TEST(Adjoint, ContinuousViscosityKernelsConverge) {
  // The continuous itau kernels differ from the discrete by the commute
  // error, which falls under step refinement — the resolution diagnostic
  // on the viscosity kernels themselves.
  auto setup = Scene::Make(1e22);
  std::mt19937 rng(3);
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();

  auto kernelDiff = [&](std::size_t npts) {
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * kyr);
    }
    const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
    std::mt19937 local(23);
    auto loads = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      loads.push_back(setup.Smooth(local, 100.0 / sigma));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts - 1].relativeSeaLevelDual = setup.Smooth(local, 1.0);

    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(op, setup.solver, setup.state,
                                {.rtol = 1e-12, .record_stride = 1});
    auto snapshots = evolution.Run(times, History(times, loads));
    const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);

    auto kernels = std::map<AdjointMethod, std::vector<double>>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      AdjointSolver adjoint(evolution, {.method = method, .rtol = 1e-12});
      kernels[method] =
          adjoint.FirstOrderKernels(times, measurements, &trajectory)
              .itauKernel[0];
    }
    auto diff = 0.0, norm = 0.0;
    const auto& a = kernels[AdjointMethod::Discrete];
    const auto& b = kernels[AdjointMethod::Continuous];
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      diff += (a[i] - b[i]) * (a[i] - b[i]);
      norm += a[i] * a[i];
    }
    return std::sqrt(diff / norm);
  };

  const auto coarse = kernelDiff(6);
  const auto fine = kernelDiff(11);
  EXPECT_LT(coarse, 0.3);
  EXPECT_LT(fine, coarse / 1.5);
}

TEST(Adjoint, SavedAndRegeneratedStrainsAgree) {
  // Recorded states keep only the internal variables unless
  // record_strain is set; the sweeps then regenerate each node's strain
  // by one elastic solve. Either way the viscosity kernels agree, the
  // regenerated strain differing from the saved one only at the forward
  // tolerance.
  auto setup = Scene::Make(1e22);
  std::mt19937 rng(13);
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(n * kyr);
  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 / sigma));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[5].relativeSeaLevelDual = setup.Smooth(rng, 1.0);

  ViscoelasticOperator op(setup.rheology, kLmax);
  Evolution saving(
      op, setup.solver, setup.state,
      {.rtol = 1e-12, .record_stride = 1, .record_strain = true});
  Evolution lean(op, setup.solver, setup.state,
                         {.rtol = 1e-12, .record_stride = 1});
  const auto saved = saving.Run(times, History(times, loads));
  const auto plain = lean.Run(times, History(times, loads));
  for (const auto& s : saved) EXPECT_TRUE(s.internalState->HasStrain());
  for (const auto& s : plain) EXPECT_FALSE(s.internalState->HasStrain());
  const auto withStrain = ForwardTrajectory::FromSnapshots(saved);
  const auto withoutStrain = ForwardTrajectory::FromSnapshots(plain);

  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    AdjointSolver adjoint(saving, {.method = method, .rtol = 1e-12});
    const auto a = adjoint.FirstOrderKernels(times, measurements, &withStrain)
                       .itauKernel[0];
    const auto b =
        adjoint.FirstOrderKernels(times, measurements, &withoutStrain)
            .itauKernel[0];
    auto scale = 0.0, worst = 0.0;
    for (auto i = std::size_t{0}; i < a.size(); ++i) {
      scale = std::max(scale, std::abs(a[i]));
      worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    EXPECT_GT(scale, 0.0);
    EXPECT_LT(worst, 1e-8 * scale) << "method " << static_cast<int>(method);
  }
}

TEST(AdjointGate, KernelsMatchPyslfp) {
  // The reference is pyslfp's adjoint kernels on its GLQ grid (which
  // coincides with ours at this lmax, isolating implementation error):
  // load duals per node and the branch-0 itau kernel, each method against
  // its own counterpart — the algebra is identical. The resolved
  // regime: the continuous kernels amplify exponentially when stiff
  // (see the generating script), so only here is parity meaningful.
  auto setup = Scene::Make(1e22);
  const auto& grid = setup.solver.Grid();
  ASSERT_EQ(grid.NumberOfLongitudes(), 2 * kLmax + 1)
      << "the GSHTrans grid no longer coincides with pyshtools GLQ here";
  const auto& scales = setup.solver.SolverScales();
  const auto& matScales = setup.material.ModelScales();

  // The fixture's setting: 6 nodes over 10 kyr, linear melt, duals at
  // nodes 3 and 5 (the Cap formulas of the generating script).
  std::ifstream file(std::filesystem::path(SL3DPP_TEST_DATA_DIR) /
                     "pyslfp_kernels.txt");
  ASSERT_TRUE(file.is_open());

  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  // The potential and angular-velocity duals are SI-anchored (the
  // generating script's PHI_ANCHOR / OMEGA_ANCHOR): pyslfp's EarthModel
  // scales differ from planetmodel/sl3dpp geophysical (phi_nd ~ T^2, a
  // factor 4 pi / 3 — docs/design.md), so each code
  // multiplies its dual by ITS OWN factor against the shared anchor and
  // the objective is code-invariant. Lengths need no anchor.
  const double potConv =
      scales.Factor(PotentialDimensions) / 1.0e7;  // PHI_ANCHOR
  const double omConv =
      scales.Factor(Dimensions{.time = -1}) / 1.0e-4;  // OMEGA_ANCHOR
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[3].relativeSeaLevelDual = grid.FromFunction([](double t, double p) {
    return Cap(t, p, 0.9, 1.2, 0.5);
  });
  Measurement last;
  last.displacementDual = grid.FromFunction([](double t, double p) {
    return 0.7 * Cap(t, p, 2.4, 0.3, 0.6);
  });
  last.potentialDual = grid.FromFunction([&](double t, double p) {
    return 0.4 * potConv * Cap(t, p, 1.4, 3.3, 0.7);
  });
  last.angularVelocityDual = {0.2 * omConv, -0.4 * omConv, 0.8 * omConv};
  measurements[5] = std::move(last);

  // The recorded linear-melt trajectory for the itau kernels.
  const auto& C = setup.state.oceanFunction;
  auto load = [&](double t) {
    auto out = std::vector<double>(setup.state.iceThickness.size());
    const auto fraction = -0.1 * (t / times.back());
    for (auto i = std::size_t{0}; i < out.size(); ++i) {
      out[i] = setup.solver.IceDensity() * (1.0 - C[i]) * fraction *
               setup.state.iceThickness[i];
    }
    return out;
  };
  ViscoelasticOperator op(setup.rheology, kLmax);
  Evolution evolution(op, setup.solver, setup.state,
                              {.rtol = 1e-12, .record_stride = 1});
  auto snapshots = evolution.Run(times, load);
  const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);

  const double loadSi =
      scales.Length() / scales.Factor(Dimensions{.mass = 1, .length = -2});
  const double itauSi =
      std::pow(scales.Length(), 3) * matScales.Time();
  const auto& mesh = setup.material.Mesh();

  constexpr auto tol = 1e-5;
  auto line = std::string{};
  auto method = std::optional<AdjointMethod>{};
  Kernels kernels;
  auto ours = std::vector<std::complex<double>>{};
  auto coefScale = 0.0, coefWorst = 0.0;
  auto checkedMethods = 0;
  auto closeBlock = [&] {
    if (!method.has_value()) return;
    EXPECT_GT(coefScale, 0.0);
    EXPECT_LT(coefWorst, tol * coefScale)
        << "load duals, method " << static_cast<int>(*method);
    coefScale = coefWorst = 0.0;
  };
  while (std::getline(file, line)) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream row(line);
    if (line.starts_with("method")) {
      closeBlock();
      std::string head, name;
      row >> head >> name;
      method = name == "discrete" ? AdjointMethod::Discrete
                                  : AdjointMethod::Continuous;
      AdjointSolver adjoint(evolution, {.method = *method, .rtol = 1e-12});
      kernels = adjoint.FirstOrderKernels(times, measurements, &trajectory);
      ++checkedMethods;
      continue;
    }
    if (line.starts_with("node")) {
      auto head = std::string{};
      auto node = 0;
      auto tSi = 0.0;
      row >> head >> node >> tSi;
      EXPECT_NEAR(times[node] * scales.Time(), tSi,
                  1e-6 * std::max(tSi, 1.0));
      ours = grid.Expand(kernels.directLoadKernels[node]);
      for (auto& v : ours) v *= loadSi;
      continue;
    }
    if (line.starts_with("itau")) {
      auto head = std::string{};
      auto count = std::size_t{0};
      row >> head >> count;
      // The continuous block carries no itau reference (the generating
      // script records why). sl3dpp's own continuous kernel is checked
      // against its exact discrete in ContinuousViscosityKernelsConverge.
      if (count == 0) continue;
      ASSERT_EQ(count, kernels.itauKernel[0].size());
      // the reference pairs by the plain nodal sum: our density per unit
      // volume times the nodes' volume weights
      const auto nodalWeights =
          MemoryPoints::Of(MemoryPointSet::Gll, mesh).VolumeWeights(mesh);
      ASSERT_EQ(nodalWeights.size(), count);
      auto kScale = 0.0, kWorst = 0.0;
      for (auto i = std::size_t{0}; i < count; ++i) {
        std::getline(file, line);
        std::istringstream krow(line);
        auto rSi = 0.0, value = 0.0;
        krow >> rSi >> value;
        const auto e = static_cast<int>(i) / mesh.NGll();
        const auto q = static_cast<int>(i) % mesh.NGll();
        ASSERT_NEAR(mesh.Radius(e, q) * matScales.Length(), rSi,
                    1e-6 * std::max(rSi, 1.0))
            << "radial meshes diverge at node " << i;
        kScale = std::max(kScale, std::abs(value));
        kWorst = std::max(
            kWorst, std::abs(kernels.itauKernel[0][i] * nodalWeights[i] * itauSi -
                             value));
      }
      EXPECT_GT(kScale, 0.0);
      EXPECT_LT(kWorst, tol * kScale)
          << "itau kernel, method " << static_cast<int>(*method);
      continue;
    }
    auto l = 0, m = 0;
    auto a = 0.0, b = 0.0;
    row >> l >> m >> a >> b;
    const auto reference = ComplexFromRealPair({a, b}, m);
    coefScale = std::max(coefScale, std::abs(reference));
    coefWorst = std::max(
        coefWorst, std::abs(ours[SurfaceGrid::Index(l, m)] - reference));
  }
  closeBlock();
  EXPECT_EQ(checkedMethods, 2);
}

TEST(Adjoint, MaterialKernelsAreTheTangentTranspose) {
  // The discrete material kernels: their pairing with a direction in
  // kappa and mu at the memory points equals the material tangent's
  // objective change (the exact transpose, at solver precision), both
  // schemes, with the memory points at the GLL nodes and at the Gauss
  // points; and at the GLL points, where the direction is a nodal one,
  // central differences of the objective on the shifted material (the
  // rheology rebuilt at the same fractions and times, the sea level
  // engine's gravity held) land on them at second order with Richardson
  // at 1e-6. With rotational feedback on, axial and full, the kernels are
  // the tangent's transpose alike. Refused without a trajectory.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(31);
  const auto baseTable = ViscoelasticOperator(setup.rheology, kLmax).ElasticLoveNumbers();
  SeaLevelParameters held;
  held.body_from_table = false;
  held.mean_sea_floor_radius = baseTable.radius;
  held.gravitational_acceleration = baseTable.surface_gravity;
  const GRDSolver solver(SurfaceGrid(kLmax), baseTable, held);
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 / sigma));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[times.size() - 1].displacementDual = setup.Smooth(rng, 0.5);
  measurements[2].potentialDual = setup.Smooth(rng, 0.7);

  const auto& material = setup.material;
  const auto& mesh = material.Mesh();
  const auto [k0, m0] = material.Moduli().KappaMu();
  MaterialDirection d;
  d.kappa.assign(k0.size(), 0.0);
  d.mu.assign(m0.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      const auto r = mesh.Radius(e, q);
      const auto profile = 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
      if (material.IsFluid(e)) continue;
      d.kappa[i] = 0.5 * profile * k0[i];
      d.mu[i] = profile * m0[i];
    }
  }
  const auto& base = setup.rheology;
  auto rheologyAt = [&](double eps) {
    const auto shifted = material.Shifted(d, eps);
    const auto muU = base.MuUnrelaxed();
    const auto [kappa, mu] = shifted.Moduli().KappaMu();
    auto branches = std::vector<Rheology::Branch>{};
    for (auto k = std::size_t{0}; k < base.BranchCount(); ++k) {
      auto branch = base.BranchAt(k);
      for (auto i = std::size_t{0}; i < branch.mu.size(); ++i) {
        branch.mu[i] = muU[i] > 0.0 ? branch.mu[i] / muU[i] * mu[i] : 0.0;
      }
      branches.push_back(std::move(branch));
    }
    return std::pair(shifted, Rheology::Create(shifted, branches).value());
  };

  for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss})
  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Axial,
                        RotationalFeedback::Full})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const bool rotating = rotation != RotationalFeedback::None;
    // rotational feedback at the default points only
    if (rotating && set != MemoryPointSet::Gauss) continue;
    ViscoelasticOperator op(setup.rheology, kLmax, {.memory_points = set});
    const auto atPoints = AtMemoryPoints(op.Points(), d);
    Evolution evolution(op, solver, setup.state,
                                {.scheme = scheme,
                                 .rotation = rotation,
                                 .rtol = 1e-12,
                                 .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, History(times, loads)));
    AdjointSolver adjoint(evolution, {.method = AdjointMethod::Discrete,
                                      .rtol = 1e-12,
                                      .material_kernels = true});
    const auto kernels = adjoint.FirstOrderKernels(times, measurements, &trajectory);
    ASSERT_EQ(kernels.kappaKernel.size(), atPoints.kappa.size());
    ASSERT_EQ(kernels.muKernel.size(), atPoints.mu.size());
    const auto paired =
        VolumePairing(op.Points(), mesh, kernels.kappaKernel, atPoints.kappa) +
        VolumePairing(op.Points(), mesh, kernels.muKernel, atPoints.mu);

    TangentPerturbation direction;
    direction.material = atPoints;
    const auto tangent = Objective(
        setup,
        TangentLinearSolver(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
            .Run(times, direction, &trajectory),
        measurements);
    EXPECT_NEAR(paired, tangent, 1e-10 * std::abs(tangent))
        << "scheme " << static_cast<int>(scheme) << ", points "
        << static_cast<int>(set) << ", rotation " << static_cast<int>(rotation);
    // The refusal without a trajectory.
    EXPECT_THROW((void)adjoint.FirstOrderKernels(times, measurements),
                 std::invalid_argument);
    if (set != MemoryPointSet::Gll) continue;

    auto functional = [&](double eps) {
      const auto [shifted, rheology] = rheologyAt(eps);
      ViscoelasticOperator moved(rheology, kLmax,
                                 {.memory_points = MemoryPointSet::Gll});
      Evolution movedEvolution(moved, solver, setup.state,
                                       {.scheme = scheme,
                                        .rotation = RotationalFeedback::None,
                                        .rtol = 1e-12});
      return Objective(setup, movedEvolution.Run(times, History(times, loads)),
                       measurements);
    };
    auto fd = std::map<double, double>{};
    for (double eps : {0.1, 0.05}) {
      fd[eps] = (functional(eps) - functional(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.1] - paired);
    const auto r2 = std::abs(fd[0.05] - paired);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme);
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.05] - fd[0.1]) / 3.0;
    EXPECT_NEAR(richardson, paired, 1e-6 * std::abs(paired))
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(Adjoint, MaterialKernelsAreFunctionsOfRadius) {
  // The kernels as functions. The point-held kernels in kappa and mu and
  // the nodal itau kernel are densities per unit volume sampled at their
  // points, which pair by the points' volume weights (an array of
  // another layout is refused); and the two memory point sets, two
  // discretisations of one problem, give samples of the same functions:
  // the GLL set's, interpolated within each element to the Gauss points,
  // against the Gauss set's, in the L2 norm weighted by volume.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(31);
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  auto times = std::vector<double>{};
  for (auto n = 0; n <= 5; ++n) times.push_back(2.0 * n * kyr);
  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  auto loads = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 / sigma));
  }
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[2].displacementDual = setup.Smooth(rng, 0.5);
  const auto& mesh = setup.material.Mesh();

  struct Functions {
    std::vector<double> kappa, mu, itau;
  };
  auto functionsAt = [&](MemoryPointSet set) {
    ViscoelasticOperator op(setup.rheology, kLmax, {.memory_points = set});
    Evolution evolution(op, setup.solver, setup.state,
                                {.rotation = RotationalFeedback::None,
                                 .rtol = 1e-12,
                                 .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, History(times, loads)));
    const auto kernels =
        AdjointSolver(evolution, {.method = AdjointMethod::Discrete,
                                  .rtol = 1e-12,
                                  .material_kernels = true})
            .FirstOrderKernels(times, measurements, &trajectory);
    const auto& points = op.Points();
    const auto weights = points.VolumeWeights(mesh);
    const auto radii = points.Radii(mesh);
    EXPECT_EQ(weights.size(), kernels.muKernel.size());
    EXPECT_EQ(radii.size(), weights.size());
    Functions f;
    f.kappa = kernels.kappaKernel;
    f.mu = kernels.muKernel;
    f.itau = kernels.itauKernel[0];
    // a kernel of another layout is refused
    EXPECT_THROW(
        (void)VolumePairing(points, mesh, std::vector<double>(3), kernels.muKernel),
        std::invalid_argument);
    return f;
  };
  const auto gll = functionsAt(MemoryPointSet::Gll);
  const auto gauss = functionsAt(MemoryPointSet::Gauss);
  const auto points = MemoryPoints::Of(MemoryPointSet::Gauss, mesh);
  const auto weights = points.VolumeWeights(mesh);
  auto distance = [&](const std::vector<double>& nodal,
                      const std::vector<double>& atPoints) {
    const auto carried = points.Interpolate(nodal);
    auto diff = 0.0, size = 0.0;
    for (auto i = std::size_t{0}; i < weights.size(); ++i) {
      diff += weights[i] * (carried[i] - atPoints[i]) * (carried[i] - atPoints[i]);
      size += weights[i] * atPoints[i] * atPoints[i];
    }
    return std::sqrt(diff / size);
  };
  const auto kappa = distance(gll.kappa, gauss.kappa);
  const auto mu = distance(gll.mu, gauss.mu);
  // the nodal itau kernels of both sets, as functions at the GLL nodes
  // (the node at the centre, without volume, has no weight)
  const auto nodalWeights =
      MemoryPoints::Of(MemoryPointSet::Gll, mesh).VolumeWeights(mesh);
  auto diff = 0.0, size = 0.0;
  for (auto i = std::size_t{0}; i < nodalWeights.size(); ++i) {
    diff += nodalWeights[i] * (gll.itau[i] - gauss.itau[i]) * (gll.itau[i] - gauss.itau[i]);
    size += nodalWeights[i] * gauss.itau[i] * gauss.itau[i];
  }
  const auto itau = std::sqrt(diff / size);
  std::cout << "functions, GLL against Gauss: kappa " << kappa << ", mu " << mu
            << ", itau " << itau << '\n';
  // (they land at 2.5e-9, 3.6e-8 and 7.1e-7)
  EXPECT_LT(kappa, 1e-6);
  EXPECT_LT(mu, 1e-6);
  EXPECT_LT(itau, 1e-5);
}

TEST(Adjoint, ContinuousMaterialKernelsAgreeAndConverge) {
  // The continuous material kernels carry the commute error of the
  // viscosity kernels: against the exact discrete ones (paired with a
  // direction in kappa and mu) they agree at the per-cent level in the
  // resolved regime and the gap falls under step refinement, both
  // schemes, with the memory points at the GLL nodes and at the Gauss
  // points; the lumped observation pairings are included (a measurement
  // mid-run and at the end).
  auto setup = Scene::Make(1e22);
  const auto& material = setup.material;
  const auto& mesh = material.Mesh();
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  MaterialDirection d;
  d.kappa.assign(k0.size(), 0.0);
  d.mu.assign(m0.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      const auto r = mesh.Radius(e, q);
      const auto profile = 0.1 * (0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r));
      if (material.IsFluid(e)) continue;
      d.kappa[i] = 0.5 * profile * k0[i];
      d.mu[i] = profile * m0[i];
    }
  }
  // One problem at every resolution: the load a ramp of a fixed field
  // plus a fixed field rising and falling, the measurements at the
  // same times (mid-run and the end) with fixed duals.
  const auto& scales = setup.solver.SolverScales();
  const auto kyr = 1000.0 * 3.15576e7 / scales.Time();
  const auto sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  std::mt19937 rng(23);
  const auto ramp = setup.Smooth(rng, 100.0 / sigma);
  const auto bump = setup.Smooth(rng, 60.0 / sigma);
  const auto gMid = setup.Smooth(rng, 1.0);
  const auto gEnd = setup.Smooth(rng, 0.5);
  const auto gDisp = setup.Smooth(rng, 0.4);
  auto gap = [&](std::size_t npts, Scheme scheme, MemoryPointSet set,
                 RotationalFeedback rotation = RotationalFeedback::None) {
    auto times = std::vector<double>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      times.push_back(10.0 * static_cast<double>(n) /
                      static_cast<double>(npts - 1) * kyr);
    }
    auto loads = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < npts; ++n) {
      const auto s = static_cast<double>(n) / static_cast<double>(npts - 1);
      auto load = std::vector<double>(ramp.size());
      for (auto i = std::size_t{0}; i < load.size(); ++i) {
        load[i] = s * ramp[i] + std::sin(std::numbers::pi * s) * bump[i];
      }
      loads.push_back(std::move(load));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[npts / 2].relativeSeaLevelDual = gMid;
    measurements[npts - 1].relativeSeaLevelDual = gEnd;
    measurements[npts - 1].displacementDual = gDisp;
    ViscoelasticOperator op(setup.rheology, kLmax, {.memory_points = set});
    const auto atPoints = AtMemoryPoints(op.Points(), d);
    Evolution evolution(op, setup.solver, setup.state,
                                {.scheme = scheme,
                                 .rotation = rotation,
                                 .rtol = 1e-12,
                                 .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, History(times, loads)));
    auto paired = std::map<AdjointMethod, double>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      const auto kernels =
          AdjointSolver(evolution, {.method = method,
                                    .rtol = 1e-12,
                                    .material_kernels = true})
              .FirstOrderKernels(times, measurements, &trajectory);
      paired[method] =
          VolumePairing(op.Points(), mesh, kernels.kappaKernel, atPoints.kappa) +
          VolumePairing(op.Points(), mesh, kernels.muKernel, atPoints.mu);
    }
    return std::abs(paired[AdjointMethod::Continuous] - paired[AdjointMethod::Discrete]) /
           std::abs(paired[AdjointMethod::Discrete]);
  };
  for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto coarse = gap(5, scheme, set);
    const auto fine = gap(9, scheme, set);
    // a few per cent at 2.5 kyr steps (the first-order scheme the larger),
    // falling under refinement
    EXPECT_LT(coarse, 0.1) << "scheme " << static_cast<int>(scheme);
    EXPECT_LT(fine, coarse / 1.5) << "scheme " << static_cast<int>(scheme)
                                  << " " << coarse << " " << fine;
    // a third level at the default points
    if (set != MemoryPointSet::Gauss) continue;
    const auto finer = gap(17, scheme, set);
    EXPECT_LT(finer, fine / 1.5) << "scheme " << static_cast<int>(scheme)
                                 << " " << fine << " " << finer;
  }
  // the same with the rotational feedback on, in kappa and mu
  for (auto rotation : {RotationalFeedback::Axial, RotationalFeedback::Full}) {
    const auto coarse =
        gap(5, Scheme::ExponentialTrapezoid, MemoryPointSet::Gauss, rotation);
    const auto fine =
        gap(9, Scheme::ExponentialTrapezoid, MemoryPointSet::Gauss, rotation);
    EXPECT_LT(coarse, 0.1) << "rotation " << static_cast<int>(rotation);
    EXPECT_LT(fine, coarse / 1.5) << "rotation " << static_cast<int>(rotation)
                                  << " " << coarse << " " << fine;
  }
}

TEST(Adjoint, MaterialGradientIsTheTransposeOfTheVariation) {
  // For random solution pairs of every degree, the operator's nodal
  // material gradients paired with a direction in kappa and mu equal the
  // pair's product with the variation operator (the linearisation of the
  // assembled system): the band assembly's integrands, pointwise.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(11);
  std::normal_distribution<double> normal;
  const auto& material = setup.material;
  const auto [k0, m0] = material.Moduli().KappaMu();
  MaterialDirection d;
  d.kappa.assign(k0.size(), 0.0);
  d.mu.assign(m0.size(), 0.0);
  for (auto i = std::size_t{0}; i < k0.size(); ++i) {
    d.kappa[i] = normal(rng) * k0[i];
    d.mu[i] = normal(rng) * m0[i];
  }
  const auto variation = material.Variation(d);
  ViscoelasticOperator op(setup.rheology, kLmax,
                          {.memory_points = MemoryPointSet::Gll});
  for (auto l = 0; l <= kLmax; ++l) {
    DegreeOperator system(material, l);
    const auto ndof = system.DofCount();
    const auto ncols = 3;
    auto y = std::vector<double>(static_cast<std::size_t>(ndof) * ncols);
    auto x = std::vector<double>(y.size());
    for (auto& v : y) v = normal(rng);
    for (auto& v : x) v = normal(rng);
    const auto dKx = system.ApplyVariation(variation, x, ncols);
    auto direct = 0.0;
    for (auto i = std::size_t{0}; i < y.size(); ++i) direct += y[i] * dKx[i];
    const auto [gk, gm] = op.MaterialGradient(l, y, x, ncols);
    auto paired = 0.0;
    for (auto i = std::size_t{0}; i < k0.size(); ++i) {
      paired += gk[i] * d.kappa[i] + gm[i] * d.mu[i];
    }
    EXPECT_NEAR(paired, direct, 1e-12 * std::abs(direct)) << "degree " << l;
  }
}

namespace {

/// The kernels' scene at a lower degree: prem8 at lmax 8, a Maxwell mantle
/// of 1e21 Pa s, the analytic test state, a cap's load ramped on over the
/// first half of 8 kyr and held, measured by the sea level under a cap at
/// the kink and, at half the weight, at the end.
struct RampScene {
  static constexpr int lmax = 8;
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  GRDSolver::State state;
  std::vector<double> load;
  std::vector<double> cap;

  static RampScene Make() {
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
    ViscoelasticOperator op(rheology, lmax);
    GRDSolver solver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
                     SeaLevelParameters{});
    Fingerprint fingerprint(lmax, op.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto ice = toNd(fingerprint.IceThickness());
    auto state = solver.MakeState(ice, toNd(fingerprint.SeaLevel()));
    auto load = std::vector<double>(ice.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] =
          solver.IceDensity() * (1.0 - state.oceanFunction[i]) * 0.3 * ice[i];
    }
    auto cap = solver.Grid().FromFunction(
        [](double t, double p) { return Cap(t, p, 1.2, 0.8, 0.5); });
    return RampScene{std::move(model),    std::move(material),
                     std::move(rheology), std::move(solver),
                     std::move(state),    std::move(load),
                     std::move(cap)};
  }

  [[nodiscard]] double Span() const {
    return 8.0 * 1000.0 * 3.15576e7 / solver.SolverScales().Time();
  }
  [[nodiscard]] std::vector<double> Times(int n) const {
    auto out = std::vector<double>{};
    for (auto k = 0; k <= n; ++k) out.push_back(Span() * k / n);
    return out;
  }
  [[nodiscard]] std::function<std::vector<double>(double)> Direct() const {
    return [this](double t) {
      auto out = load;
      const auto f = std::min(t / (0.5 * Span()), 1.0);
      for (auto& v : out) v *= f;
      return out;
    };
  }
  /// The measurements on an n-step grid, and the objective they define.
  [[nodiscard]] std::map<std::size_t, Measurement> Measurements(int n) const {
    const auto a2 = 1.0 / (solver.Radius() * solver.Radius());
    auto out = std::map<std::size_t, Measurement>{};
    for (const auto [node, weight] :
         {std::pair{static_cast<std::size_t>(n / 2), 1.0},
          std::pair{static_cast<std::size_t>(n), 0.5}}) {
      out[node].relativeSeaLevelDual = cap;
      for (auto& v : out[node].relativeSeaLevelDual) v *= weight * a2;
    }
    return out;
  }
  [[nodiscard]] double Objective(
      const std::vector<Evolution::Snapshot>& snapshots, int n) const {
    auto f = std::vector<double>(cap.size());
    for (auto i = std::size_t{0}; i < f.size(); ++i) {
      f[i] = cap[i] * (snapshots[n / 2].relativeSeaLevelChange[i] +
                       0.5 * snapshots[n].relativeSeaLevelChange[i]);
    }
    return solver.Grid().Integrate(f);
  }
};

}  // namespace

TEST(Adjoint, Sdirk2ContinuousKernelAgainstFiniteDifferences) {
  // The continuous viscosity kernel stepped by Sdirk2's stages (the
  // reversed sweep's), paired with a smooth itau direction, against the
  // derivative of the objective: central differences of Sdirk2 forward
  // runs at 128 and 256 steps, Richardson-extrapolated, which the
  // trapezoid's runs extrapolate to as well. From steps 9.3 times the
  // shortest relaxation time down, Sdirk2's continuous derivative
  // converges to it at second order, and at the longest steps its error
  // is under half the trapezoid's continuous kernel's.
  const auto scene = RampScene::Make();
  const auto& mesh = scene.material.Mesh();
  const auto& base = scene.rheology.BranchAt(0);
  auto top = 0.0;
  for (auto v : base.itau) top = std::max(top, v);
  auto ditau = std::vector<double>(base.itau.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      if (base.itau[i] > 0.0) {
        const auto r = mesh.Radius(e, q);
        ditau[i] = (0.3 + 0.7 * r * r) * top;
      }
    }
  }
  auto objective = [&](Scheme scheme, int n, double eps) {
    auto branch = base;
    for (auto i = std::size_t{0}; i < branch.itau.size(); ++i) {
      branch.itau[i] += eps * ditau[i];
    }
    const auto rheology =
        Rheology::Create(scene.material, {std::move(branch)}).value();
    ViscoelasticOperator op(rheology, RampScene::lmax);
    Evolution evolution(op, scene.solver, scene.state,
                        {.scheme = scheme, .rtol = 1e-13});
    return scene.Objective(evolution.Run(scene.Times(n), scene.Direct()), n);
  };
  auto differences = [&](Scheme scheme) {
    constexpr auto eps = 1e-3;
    auto fd = std::array<double, 2>{};
    for (auto k = 0; k < 2; ++k) {
      const auto n = 128 << k;
      fd[k] = (objective(scheme, n, eps) - objective(scheme, n, -eps)) /
              (2.0 * eps);
    }
    return fd[1] + (fd[1] - fd[0]) / 3.0;
  };
  const ViscoelasticOperator op(scene.rheology, RampScene::lmax);
  auto kernel = [&](Scheme scheme, int n) {
    Evolution evolution(op, scene.solver, scene.state,
                        {.scheme = scheme, .rtol = 1e-13, .record_stride = 1});
    const auto times = scene.Times(n);
    const auto trajectory =
        ForwardTrajectory::FromSnapshots(evolution.Run(times, scene.Direct()));
    const AdjointSolver adjoint(
        evolution, {.method = AdjointMethod::Continuous, .rtol = 1e-13});
    const auto kernels =
        adjoint.FirstOrderKernels(times, scene.Measurements(n), &trajectory);
    return VolumePairing(mesh, kernels.itauKernel[0], ditau);
  };
  const auto truth = differences(Scheme::Sdirk2);
  const auto trapezoidTruth = differences(Scheme::ExponentialTrapezoid);
  auto damped = std::map<int, double>{};
  for (const auto n : {8, 16, 32}) {
    damped[n] = (kernel(Scheme::Sdirk2, n) - truth) / truth;
  }
  const auto trapezoid =
      (kernel(Scheme::ExponentialTrapezoid, 8) - truth) / truth;
  const auto extrapolated = (damped[32] + (damped[32] - damped[16]) / 3.0);
  std::cout << std::format(
      "[sdirk2 continuous kernel] truths' gap {:.3e}; errors {:+.3e} {:+.3e} "
      "{:+.3e}, Richardson {:+.3e}; the trapezoid's at 8 steps {:+.3e}\n",
      (trapezoidTruth - truth) / truth, damped[8], damped[16], damped[32],
      extrapolated, trapezoid);
  // measured: the truths 5e-9 apart; errors +1.37, +0.32, +0.077 % and the
  // Richardson value -4e-5; the trapezoid's continuous kernel +3.4 %
  EXPECT_LT(std::abs(trapezoidTruth - truth), 1e-6 * std::abs(truth));
  for (const auto n : {8, 16}) {
    const auto order = std::log2(damped[n] / damped[2 * n]);
    EXPECT_GT(order, 1.9) << n;
    EXPECT_LT(order, 2.3) << n;
  }
  EXPECT_LT(std::abs(extrapolated), 1e-4);
  EXPECT_LT(std::abs(damped[8]), 0.5 * std::abs(trapezoid));
}

TEST(Adjoint, Sdirk2MomentKernelsAgainstTheForwardRun) {
  // The continuous sweep's kernels in point sources' moment tensors under
  // Sdirk2: the adjoint displacement's strain at the sources, the toroidal
  // adjoint stepped by the same stages beside the spheroidal. The run is
  // linear in its sources, so the objective of a run under them alone is
  // the derivative along them: Sdirk2's run at 128 steps, against which
  // the kernels' pairing converges as the steps are halved, many times
  // closer than the trapezoid's continuous kernels at the same steps; with
  // an ocean and the rotational feedback and without either, measured in
  // the sea level, the displacement and the tangential and toroidal
  // potentials.
  const auto scene = RampScene::Make();
  const auto& grid = scene.solver.Grid();
  const auto& sle = scene.solver.SolverScales();
  const auto span = scene.Span();
  const auto surface = scene.material.Radius() *
                       scene.model.ModelScales().Length() / sle.Length();
  const auto unit =
      1e21 / sle.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto points =
      std::vector<BodyPoint>{{.radius = surface - 400e3 / sle.Length(),
                              .colatitude = 1.1,
                              .longitude = 4.0},
                             {.radius = surface - 800e3 / sle.Length(),
                              .colatitude = 2.2,
                              .longitude = 1.0}};
  const auto patterns = std::vector<SphericalTensor>{{.rr = 0.9 * unit,
                                                      .tt = -0.2 * unit,
                                                      .pp = -0.7 * unit,
                                                      .rt = 0.3 * unit,
                                                      .rp = -0.7 * unit,
                                                      .tp = 0.6 * unit},
                                                     {.rr = -0.4 * unit,
                                                      .tt = 0.8 * unit,
                                                      .pp = -0.4 * unit,
                                                      .rt = 0.5 * unit,
                                                      .rp = 0.1 * unit,
                                                      .tp = -0.3 * unit}};
  const auto shape = [&](double t) {
    return std::sin(0.5 * std::numbers::pi * t / span) + 0.3;
  };
  const auto sources = [&](double t) {
    auto at = std::vector<PointSource>{};
    for (auto p = std::size_t{0}; p < points.size(); ++p) {
      auto M = patterns[p];
      for (auto* c : {&M.rr, &M.tt, &M.pp, &M.rt, &M.rp, &M.tp}) *c *= shape(t);
      at.push_back({.at = points[p], .moment = M});
    }
    return at;
  };
  const auto a2 = 1.0 / (scene.solver.Radius() * scene.solver.Radius());
  auto field = [&](double colatitude, double longitude, double amplitude) {
    return grid.FromFunction([&](double t, double p) {
      return amplitude * a2 * Cap(t, p, colatitude, longitude, 0.6);
    });
  };
  const auto gSea = field(1.2, 0.8, 1.0), gU = field(2.0, 3.0, 0.5),
             gV = field(0.7, 5.0, 0.6), gW = field(1.6, 2.2, 0.8);
  auto measurementsOf = [&](int n) {
    auto out = std::map<std::size_t, Measurement>{};
    out[n / 2].relativeSeaLevelDual = gSea;
    out[n / 2].toroidalPotentialDual = gW;
    out[n].displacementDual = gU;
    out[n].tangentialPotentialDual = gV;
    out[n].toroidalPotentialDual = gW;
    return out;
  };
  auto objective = [&](const std::vector<Evolution::Snapshot>& snapshots,
                       const std::map<std::size_t, Measurement>& measurements) {
    auto total = 0.0;
    auto add = [&](const std::vector<double>& dual,
                   const std::vector<double>& x) {
      if (dual.empty() || x.empty()) return;
      auto f = std::vector<double>(x.size());
      for (auto i = std::size_t{0}; i < f.size(); ++i) f[i] = dual[i] * x[i];
      total += grid.Integrate(f) / a2;
    };
    for (const auto& [node, m] : measurements) {
      const auto& snap = snapshots[node];
      add(m.relativeSeaLevelDual, snap.relativeSeaLevelChange);
      add(m.displacementDual, snap.displacement);
      add(m.tangentialPotentialDual, snap.tangentialPotential);
      add(m.toroidalPotentialDual, snap.toroidalPotential);
    }
    return total;
  };
  const ViscoelasticOperator op(scene.rheology, RampScene::lmax);
  const auto dry = scene.solver.NoOceanState();
  for (const auto* background : {&scene.state, &dry}) {
    const auto label = std::string(background == &dry ? "no ocean" : "ocean");
    Evolution evolution(
        op, scene.solver, *background,
        {.scheme = Scheme::Sdirk2,
         .rotation = background == &dry ? RotationalFeedback::None
                                        : RotationalFeedback::Full,
         .rtol = 1e-12});
    const auto truth = objective(
        evolution.Run(scene.Times(128), Evolution::Forcing{.sources = sources}),
        measurementsOf(128));
    auto gapOf = [&](Scheme scheme, int n) {
      Evolution stepped(
          op, scene.solver, *background,
          {.scheme = scheme,
           .rotation = background == &dry ? RotationalFeedback::None
                                          : RotationalFeedback::Full,
           .rtol = 1e-12});
      const auto times = scene.Times(n);
      const auto kernels =
          AdjointSolver(stepped, {.method = AdjointMethod::Continuous,
                                  .rtol = 1e-12,
                                  .moment_points = points})
              .FirstOrderKernels(times, measurementsOf(n));
      auto paired = 0.0;
      for (auto k = std::size_t{0}; k < times.size(); ++k) {
        const auto at = sources(times[k]);
        for (auto p = std::size_t{0}; p < points.size(); ++p) {
          paired += kernels.momentKernels[k][p].Contract(at[p].moment);
        }
      }
      return std::abs(paired - truth) / std::abs(truth);
    };
    auto gaps = std::vector<double>{};
    for (const auto n : {8, 16, 32}) gaps.push_back(gapOf(Scheme::Sdirk2, n));
    const auto trapezoid = gapOf(Scheme::ExponentialTrapezoid, 16);
    std::cout << std::format(
        "[sdirk2 moment kernels] {}: {:.3e} {:.3e} {:.3e}; the trapezoid's at "
        "16 steps {:.3e}\n",
        label, gaps[0], gaps[1], gaps[2], trapezoid);
    // measured: with an ocean 1.06e-2, 2.47e-3, 5.96e-4 (the trapezoid's
    // continuous kernels 3.8e-2 at 16 steps); without one 1.05e-2, 3.62e-3,
    // 1.04e-3 (1.1e-1), Sdirk2's ratio rising to 3.7 at 64 steps
    EXPECT_LT(gaps[2], 2e-3) << label;
    EXPECT_GT(gaps[0] / gaps[1], 2.5) << label;
    EXPECT_GT(gaps[1] / gaps[2], 2.5) << label;
    EXPECT_LT(gaps[1], trapezoid / 8.0) << label;
  }
}

TEST(Adjoint, Sdirk2RefusesTheDerivativesNotBuilt) {
  // About a run under Sdirk2 the discrete sweep, the second-order sweep and
  // the tangent-linear run are not built (their exact derivatives of the
  // stepped map need each step's stage states), and the continuous sweep
  // reads every node's state: each refused by name.
  const auto scene = RampScene::Make();
  const ViscoelasticOperator op(scene.rheology, RampScene::lmax);
  const auto times = scene.Times(4);
  const auto measurements = scene.Measurements(4);
  auto named = [](const std::function<void()>& call, std::string_view what) {
    try {
      call();
      ADD_FAILURE() << what << " ran under Sdirk2";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find("Sdirk2"), std::string::npos)
          << what << ": " << error.what();
    }
  };
  Evolution evolution(
      op, scene.solver, scene.state,
      {.scheme = Scheme::Sdirk2, .rtol = 1e-12, .record_stride = 1});
  const auto snapshots = evolution.Run(times, scene.Direct());
  const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);
  named(
      [&] {
        const AdjointSolver adjoint(evolution,
                                    {.method = AdjointMethod::Discrete});
        static_cast<void>(
            adjoint.FirstOrderKernels(times, measurements, &trajectory));
      },
      "the discrete sweep");
  named(
      [&] {
        const AdjointSolver adjoint(evolution,
                                    {.method = AdjointMethod::Continuous});
        static_cast<void>(adjoint.SecondOrderKernels(
            times, measurements, snapshots, {}, trajectory));
      },
      "the second-order sweep");
  named([&] { const TangentLinearSolver tangent(evolution); },
        "the tangent-linear run");
  Evolution strided(
      op, scene.solver, scene.state,
      {.scheme = Scheme::Sdirk2, .rtol = 1e-12, .record_stride = 2});
  const auto gapped =
      ForwardTrajectory::FromSnapshots(strided.Run(times, scene.Direct()));
  named(
      [&] {
        const AdjointSolver adjoint(strided,
                                    {.method = AdjointMethod::Continuous});
        static_cast<void>(
            adjoint.FirstOrderKernels(times, measurements, &gapped));
      },
      "the continuous sweep about a gapped trajectory");
  // the continuous sweep about the whole trajectory, and without one
  const AdjointSolver adjoint(evolution, {.method = AdjointMethod::Continuous});
  EXPECT_FALSE(adjoint.FirstOrderKernels(times, measurements, &trajectory)
                   .itauKernel.empty());
  EXPECT_TRUE(
      adjoint.FirstOrderKernels(times, measurements).itauKernel.empty());
}
