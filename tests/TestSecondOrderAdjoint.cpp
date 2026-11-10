// The second-order adjoint: the kernels it carries agree with the
// first-order sweep's, its derivatives match central differences of the
// gradient along the direction, the Hessian it gives is symmetric and is
// the adjoint of the second-order tangent-linear run, and the continuous
// method converges to the discrete one — pyslfp's test_second_order.py.

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <random>
#include <vector>

#include "KernelScene.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/TangentLinear.hpp"

using namespace sl3dpp;
using namespace kernel_scene;

namespace {

/// A kernel object (kernels or their derivatives) paired with a direction
/// in every argument: the thickness view against the thickness
/// perturbation at every node, the initial sea level kernel against its
/// perturbation, the itau kernel against its and the material kernels
/// against theirs (at the operator's memory points) by the volume pairing.
template <class Kernels>
double Pairing(const Scene& setup, const MemoryPoints& points,
               const Kernels& kernels, const TangentPerturbation& d) {
  const auto& mesh = setup.material.Mesh();
  auto total = 0.0;
  for (auto n = std::size_t{0}; n < d.iceThickness.size(); ++n) {
    total += setup.Surface(kernels.thickness[n], d.iceThickness[n]);
  }
  if (!d.initialSeaLevel.empty()) {
    total += setup.Surface(kernels.initialSeaLevel, d.initialSeaLevel);
  }
  if (!d.itau.empty()) {
    total += VolumePairing(mesh, kernels.itauKernel[0], d.itau[0]);
  }
  if (!kernels.kappaKernel.empty()) {
    if (!d.material.kappa.empty()) {
      total += VolumePairing(points, mesh, kernels.kappaKernel, d.material.kappa);
    }
    if (!d.material.mu.empty()) {
      total += VolumePairing(points, mesh, kernels.muKernel, d.material.mu);
    }
  }
  return total;
}

/// A direction in the ice thickness at every node, the initial sea level
/// (when asked) and the inverse relaxation times.
TangentPerturbation Direction(const Scene& setup, std::mt19937& rng,
                              std::size_t nodes, bool seaLevel,
                              const std::function<double(double)>& shape) {
  TangentPerturbation d;
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    d.iceThickness.push_back(setup.Smooth(rng, 20.0 * setup.Metre()));
  }
  if (seaLevel) d.initialSeaLevel = setup.Smooth(rng, 20.0 * setup.Metre());
  d.itau = setup.DeltaItau(0.1, shape);
  return d;
}

/// The frozen-shoreline load of a per-node thickness history relative to
/// its first node, as the forward driver's callable.
std::function<std::vector<double>(double)> FrozenLoad(
    const Scene& setup, const std::vector<double>& times,
    const std::vector<std::vector<double>>& thickness) {
  auto loads = std::vector<std::vector<double>>{};
  const auto& C = setup.state.oceanFunction;
  const auto rhoI = setup.solver.IceDensity();
  for (const auto& T : thickness) {
    auto load = std::vector<double>(T.size());
    for (auto i = std::size_t{0}; i < T.size(); ++i) {
      load[i] = rhoI * (1.0 - C[i]) * (T[i] - thickness[0][i]);
    }
    loads.push_back(std::move(load));
  }
  return [times, loads](double t) {
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) {
        return loads[n];
      }
    }
    throw std::runtime_error("the test history is read off its nodes");
  };
}

/// A random per-node thickness history (metres of ice).
std::vector<std::vector<double>> RandomThickness(const Scene& setup,
                                                 std::mt19937& rng,
                                                 std::size_t nodes) {
  auto out = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    out.push_back(setup.Smooth(rng, 100.0 * setup.Metre()));
  }
  return out;
}

std::map<std::size_t, Measurement> Measurements(const Scene& setup,
                                                std::mt19937& rng,
                                                std::size_t last) {
  auto out = std::map<std::size_t, Measurement>{};
  out[last].seaLevelDual = setup.Smooth(rng, 1.0);
  out[last].displacementDual = setup.Smooth(rng, 0.5);
  out[last / 2].seaLevelDual = setup.Smooth(rng, 1.0);
  return out;
}

/// A recorded base run: frozen (a random thickness history on the
/// analytic state) or migrating (the marine cap), with its trajectory and
/// ocean functions.
struct Base {
  SeaLevelSolver::State state;
  std::vector<double> times;
  std::vector<std::vector<double>> oceans;
  ForwardTrajectory trajectory;
};

Base MakeBase(const Scene& setup, const ViscoelasticOperator& op,
              std::mt19937& rng, bool migrating, Scheme scheme,
              SeaLevelEvolution** evolution, std::vector<double>* thicknessOut) {
  Base base;
  if (migrating) {
    const auto cap = MarineCap::Make(setup, 6);
    base.state = setup.solver.MakeState(cap.ice0, cap.sea0);
    base.times = cap.times;
    *evolution = new SeaLevelEvolution(
        op, setup.solver, base.state,
        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    const auto run = (*evolution)->RunMigrating(
        cap.times, [cap](double t) { return cap.Thickness(t); });
    base.oceans = OceanFunctionsOf(run);
    base.trajectory = ForwardTrajectory::FromSnapshots(run);
  } else {
    base.state = setup.state;
    for (auto n = 0; n < 6; ++n) base.times.push_back(2.0 * n * setup.Kyr());
    const auto thickness = RandomThickness(setup, rng, base.times.size());
    *evolution = new SeaLevelEvolution(
        op, setup.solver, base.state,
        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    const auto run = (*evolution)->Run(
        base.times, FrozenLoad(setup, base.times, thickness));
    base.trajectory = ForwardTrajectory::FromSnapshots(run);
  }
  (void)thicknessOut;
  return base;
}

}  // namespace

TEST(SecondOrderAdjoint, KernelsAgreeWithTheFirstOrderSweep) {
  // Both methods, about frozen and migrating runs: the kernels the
  // second-order sweep carries are the first-order sweep's to rounding,
  // and the derivatives it adds are not zero.
  auto setup = Scene::Make(1e21);
  ViscoelasticOperator op(setup.rheology, kLmax);
  for (const bool migrating : {false, true}) {
    std::mt19937 rng(1);
    SeaLevelEvolution* evolution = nullptr;
    const auto base = MakeBase(setup, op, rng, migrating,
                               Scheme::ExponentialTrapezoid, &evolution, nullptr);
    const auto measurements = Measurements(setup, rng, base.times.size() - 1);
    const auto v = Direction(setup, rng, base.times.size(), migrating,
                             [](double r) { return 0.3 + 0.7 * r * r; });
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver tangent(*evolution,
                                  {.method = method, .rtol = 1e-12, .record = true});
      const auto run = tangent.Run(base.times, v, &base.trajectory, base.oceans);
      AdjointSolver adjoint(*evolution, {.method = method, .rtol = 1e-12});
      const auto first = adjoint.IceHistoryKernels(base.times, measurements,
                                                   &base.trajectory, base.oceans);
      const auto [kernels, derivatives] = adjoint.SecondOrderKernels(
          base.times, measurements, run, v.itau, base.trajectory, base.oceans);
      for (auto n = std::size_t{0}; n < base.times.size(); ++n) {
        EXPECT_LT(MaxDiff(kernels.thickness[n], first.thickness[n]),
                  1e-12 * MaxAbs(first.thickness[n]) + 1e-300)
            << "node " << n << " migrating " << migrating;
      }
      EXPECT_LT(MaxDiff(kernels.initialSeaLevel, first.initialSeaLevel),
                1e-12 * MaxAbs(first.initialSeaLevel) + 1e-300);
      EXPECT_LT(MaxDiff(kernels.itauKernel[0], first.itauKernel[0]),
                1e-12 * MaxAbs(first.itauKernel[0]));
      auto any = 0.0;
      for (const auto& th : derivatives.thickness) any += MaxAbs(th);
      EXPECT_GT(any, 0.0);
      EXPECT_GT(MaxAbs(derivatives.itauKernel[0]), 0.0);
      EXPECT_EQ(derivatives.times, base.times);
    }
    delete evolution;
  }
}

TEST(SecondOrderAdjoint, DerivativesMatchDifferencesOfTheGradient) {
  // Along v, central differences of the gradient (the kernels paired
  // with u) converge at second order to the second-order adjoint's
  // derivatives paired with u, and Richardson lands on them, for both
  // schemes.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(2);
  auto times = std::vector<double>{};
  for (auto n = 0; n < 6; ++n) times.push_back(2.0 * n * setup.Kyr());
  const auto thickness = RandomThickness(setup, rng, times.size());
  const auto v = Direction(setup, rng, times.size(), false,
                           [](double r) { return 0.3 + 0.7 * r * r; });
  const auto u = Direction(setup, rng, times.size(), false,
                           [](double r) { return 0.8 - 0.5 * r; });
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = setup.Smooth(rng, 1.0);
  measurements[3].seaLevelDual = setup.Smooth(rng, 1.0);
  const auto& branch = setup.rheology.BranchAt(0);
  auto moved = [&](double eps) {
    auto out = thickness;
    for (auto n = std::size_t{0}; n < out.size(); ++n) {
      for (auto i = std::size_t{0}; i < out[n].size(); ++i) {
        out[n][i] += eps * v.iceThickness[n][i];
      }
    }
    return out;
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    ViscoelasticOperator op(setup.rheology, kLmax);
    SeaLevelEvolution evolution(
        op, setup.solver, setup.state,
        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, FrozenLoad(setup, times, thickness)));
    TangentLinearSolver tangent(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .record = true});
    const auto run = tangent.Run(times, v, &trajectory);
    AdjointSolver adjoint(evolution,
                          {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto [kernels, derivatives] =
        adjoint.SecondOrderKernels(times, measurements, run, v.itau, trajectory);
    const auto predicted = Pairing(setup, op.Points(), derivatives, u);

    auto gradientAt = [&](double eps) {
      auto b = branch;
      for (auto i = std::size_t{0}; i < b.itau.size(); ++i) {
        b.itau[i] += eps * v.itau[0][i];
      }
      auto rheology = Rheology::Create(setup.material, {std::move(b)}).value();
      ViscoelasticOperator movedOp(rheology, kLmax);
      SeaLevelEvolution movedEvolution(
          movedOp, setup.solver, setup.state,
          {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
      const auto movedTrajectory = ForwardTrajectory::FromSnapshots(
          movedEvolution.Run(times, FrozenLoad(setup, times, moved(eps))));
      AdjointSolver movedAdjoint(
          movedEvolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
      return Pairing(setup, op.Points(),
                     movedAdjoint.IceHistoryKernels(times, measurements,
                                                    &movedTrajectory),
                     u);
    };
    auto fd = std::map<double, double>{};
    for (double eps : {0.02, 0.01}) {
      fd[eps] = (gradientAt(eps) - gradientAt(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.02] - predicted);
    const auto r2 = std::abs(fd[0.01] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme);
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0;
    EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme);
  }
}

TEST(SecondOrderAdjoint, HessianIsSymmetricAndAdjointToTheSecondOrderRun) {
  // <H u, w> = <H w, u> to rounding, and <g, F''[u, w]> from the
  // second-order tangent-linear run equals the second-order adjoint's
  // derivatives along u paired with w, about frozen and migrating runs.
  auto setup = Scene::Make(1e21);
  ViscoelasticOperator op(setup.rheology, kLmax);
  for (const bool migrating : {false, true}) {
    std::mt19937 rng(3);
    SeaLevelEvolution* evolution = nullptr;
    const auto base = MakeBase(setup, op, rng, migrating,
                               Scheme::ExponentialTrapezoid, &evolution, nullptr);
    const auto measurements = Measurements(setup, rng, base.times.size() - 1);
    const auto u = Direction(setup, rng, base.times.size(), migrating,
                             [](double r) { return 0.3 + 0.7 * r * r; });
    const auto w = Direction(setup, rng, base.times.size(), migrating,
                             [](double r) { return 0.8 - 0.5 * r; });
    TangentLinearSolver tangent(
        *evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .record = true});
    const auto runU = tangent.Run(base.times, u, &base.trajectory, base.oceans);
    const auto runW = tangent.Run(base.times, w, &base.trajectory, base.oceans);
    AdjointSolver adjoint(*evolution,
                          {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
        base.times, measurements, runU, u.itau, base.trajectory, base.oceans);
    const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
        base.times, measurements, runW, w.itau, base.trajectory, base.oceans);
    const auto huW = Pairing(setup, op.Points(), alongU, w);
    const auto hwU = Pairing(setup, op.Points(), alongW, u);
    EXPECT_NEAR(huW, hwU, 1e-9 * std::abs(huW)) << "migrating " << migrating;

    const auto second = tangent.RunSecondOrder(
        base.times, runU, runW, {.itauU = u.itau, .itauV = w.itau},
        &base.trajectory, base.oceans);
    const auto forward = Objective(setup, second, measurements);
    EXPECT_NEAR(forward, huW, 1e-9 * std::abs(forward))
        << "migrating " << migrating;

    // Refusals: an unrecorded tangent run, a trajectory of the wrong length.
    TangentLinearSolver light(*evolution,
                              {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    EXPECT_THROW((void)adjoint.SecondOrderKernels(
                     base.times, measurements,
                     light.Run(base.times, u, &base.trajectory, base.oceans),
                     u.itau, base.trajectory, base.oceans),
                 std::invalid_argument);
    delete evolution;
  }
}

TEST(SecondOrderAdjoint, ContinuousConvergesToDiscrete) {
  // On a slowly relaxing mantle, along directions smooth in time in the
  // thickness, the relaxation times AND the moduli, the continuous
  // second-order adjoint's derivatives tend to the discrete ones at the
  // scheme's order as the steps are halved, block by block (ice,
  // viscosity, material), and so does their mismatch with the continuous
  // second-order tangent-linear run; the discrete pair match to rounding
  // at every spacing. As for the second-order run, the grids coarser
  // than 17 nodes over 10 kyr sit before the asymptotic regime on this
  // mantle. At the default memory points, the Gauss points, the moduli's
  // directions held there.
  auto setup = Scene::Make(1e22);
  std::mt19937 rng(11);
  const auto fieldT = setup.Smooth(rng, 100.0 * setup.Metre());
  const auto fieldU = setup.Smooth(rng, 20.0 * setup.Metre());
  const auto fieldV = setup.Smooth(rng, 20.0 * setup.Metre());
  const auto g = setup.Smooth(rng, 1.0);
  const auto span = 10.0 * setup.Kyr();
  const auto& mesh = setup.material.Mesh();
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = setup.material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  auto materialDirection = [&](const std::function<double(double)>& profile) {
    MaterialDirection d;
    d.kappa.assign(k0.size(), 0.0);
    d.mu.assign(m0.size(), 0.0);
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      if (setup.material.IsFluid(e)) continue;
      for (auto q = 0; q < mesh.NGll(); ++q) {
        const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
        const auto p = 0.1 * profile(mesh.Radius(e, q));
        d.kappa[i] = 0.5 * p * k0[i];
        d.mu[i] = p * m0[i];
      }
    }
    return d;
  };
  ViscoelasticOperator op(setup.rheology, kLmax);
  ASSERT_EQ(op.Points().set, MemoryPointSet::Gauss);
  auto iceGaps = std::vector<double>{}, viscosityGaps = std::vector<double>{},
       materialGaps = std::vector<double>{}, mismatches = std::vector<double>{};
  for (const auto nodes : {17, 33}) {
    auto times = std::vector<double>{};
    for (auto n = 0; n < nodes; ++n) times.push_back(span * n / (nodes - 1));
    auto scaled = [&](const std::vector<double>& field,
                      const std::function<double(double)>& profile) {
      auto out = std::vector<std::vector<double>>{};
      for (auto t : times) {
        auto f = field;
        for (auto& x : f) x *= profile(t / span);
        out.push_back(std::move(f));
      }
      return out;
    };
    const auto thickness = scaled(fieldT, [](double s) { return 1.0 - s; });
    SeaLevelEvolution evolution(op, setup.solver, setup.state,
                                {.rotation = RotationalFeedback::None,
                                 .rtol = 1e-12,
                                 .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, FrozenLoad(setup, times, thickness)));
    TangentPerturbation u, v;
    u.iceThickness = scaled(fieldU, [](double s) { return s; });
    u.itau = setup.DeltaItau(0.1, [](double r) { return 0.3 + 0.7 * r * r; });
    u.material = AtMemoryPoints(op.Points(), materialDirection([](double r) {
      return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r);
    }));
    v.iceThickness = scaled(fieldV, [](double s) { return (1.0 - s) * (1.0 - s); });
    v.itau = setup.DeltaItau(0.1, [](double r) { return 1.0 + r; });
    v.material = AtMemoryPoints(
        op.Points(), materialDirection([](double r) { return 1.0 - 0.5 * r; }));
    auto measurement = std::map<std::size_t, Measurement>{};
    measurement[times.size() - 1].seaLevelDual = g;
    struct Blocks {
      double ice, viscosity, material, forward;
    };
    auto results = std::map<AdjointMethod, Blocks>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver tangent(evolution,
                                  {.method = method, .rtol = 1e-12, .record = true});
      const auto runV = tangent.Run(times, v, &trajectory);
      const auto runU = tangent.Run(times, u, &trajectory);
      AdjointSolver adjoint(evolution, {.method = method,
                                        .rtol = 1e-12,
                                        .material_kernels = true});
      const auto [kernels, derivatives] = adjoint.SecondOrderKernels(
          times, measurement, runV, v.itau, trajectory, {}, v.material);
      SecondOrderDirection vu;
      vu.itauU = v.itau;
      vu.itauV = u.itau;
      vu.materialU = v.material;
      vu.materialV = u.material;
      const auto second = tangent.RunSecondOrder(times, runV, runU, vu, &trajectory);
      auto ice = 0.0;
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        ice += setup.Surface(derivatives.thickness[n], u.iceThickness[n]);
      }
      const auto viscosity =
          VolumePairing(mesh, derivatives.itauKernel[0], u.itau[0]);
      const auto materialBlock =
          VolumePairing(op.Points(), mesh, derivatives.kappaKernel,
                        u.material.kappa) +
          VolumePairing(op.Points(), mesh, derivatives.muKernel, u.material.mu);
      results[method] = {ice, viscosity, materialBlock,
                         Objective(setup, second, measurement)};
    }
    const auto& d = results[AdjointMethod::Discrete];
    const auto& c = results[AdjointMethod::Continuous];
    EXPECT_NEAR(d.ice + d.viscosity + d.material, d.forward,
                1e-9 * std::abs(d.forward))
        << "nodes " << nodes;
    iceGaps.push_back(std::abs(d.ice - c.ice) / std::abs(d.ice));
    viscosityGaps.push_back(std::abs(d.viscosity - c.viscosity) /
                            std::abs(d.viscosity));
    materialGaps.push_back(std::abs(d.material - c.material) /
                           std::abs(d.material));
    mismatches.push_back(
        std::abs(c.ice + c.viscosity + c.material - c.forward) /
        std::abs(c.forward));
  }
  for (const auto* gaps : {&iceGaps, &viscosityGaps, &materialGaps, &mismatches}) {
    EXPECT_LT((*gaps)[0], 0.1) << (*gaps)[0] << " " << (*gaps)[1];
    EXPECT_GT((*gaps)[0] / (*gaps)[1], 3.0) << (*gaps)[0] << " " << (*gaps)[1];
  }
}

TEST(SecondOrderAdjoint, MaterialBlocksAreExactAndSymmetric) {
  // With directions u and w in the ice thickness, the inverse relaxation
  // times AND the moduli, the discrete second-order sweep along a
  // material direction with the material kernels on: the carried kernels
  // (material included) equal the first-order sweep's to rounding;
  // <H u, w> = <H w, u> to rounding and equals <g, F''[u, w]> from the
  // second-order tangent-linear run in the material; and central
  // differences of the gradient along w (the material shifted, the
  // relaxation times and the thickness moved) paired with u converge at
  // second order to the derivatives along w paired with u, Richardson
  // landing on them — both schemes, with the memory points at the GLL
  // nodes and at the Gauss points, the moduli's directions held at the
  // points and the base moved through ViscoelasticOperator::Options::
  // moduli. At the Gauss points with the rotational feedback on too,
  // axial and full.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(43);
  auto times = std::vector<double>{};
  for (auto n = 0; n < 6; ++n) times.push_back(2.0 * n * setup.Kyr());
  const auto thickness = RandomThickness(setup, rng, times.size());
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[times.size() - 1].seaLevelDual = setup.Smooth(rng, 1.0);
  measurements[times.size() - 1].displacementDual = setup.Smooth(rng, 0.5);
  measurements[3].potentialDual = setup.Smooth(rng, 0.7);

  const auto& material = setup.material;
  const auto& mesh = material.Mesh();
  // named, not a structured binding: Clang 18 does not capture those
  // in a lambda when OpenMP is enabled
  const auto kappaMu = material.Moduli().KappaMu();
  const auto& k0 = kappaMu.first;
  const auto& m0 = kappaMu.second;
  auto materialDirection = [&](const std::function<double(double)>& profile) {
    MaterialDirection d;
    d.kappa.assign(k0.size(), 0.0);
    d.mu.assign(m0.size(), 0.0);
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      if (material.IsFluid(e)) continue;
      for (auto q = 0; q < mesh.NGll(); ++q) {
        const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
        const auto p = 0.1 * profile(mesh.Radius(e, q));
        d.kappa[i] = 0.5 * p * k0[i];
        d.mu[i] = p * m0[i];
      }
    }
    return d;
  };
  auto u = Direction(setup, rng, times.size(), false,
                     [](double r) { return 0.3 + 0.7 * r * r; });
  const auto nodalU = materialDirection(
      [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); });
  auto w = Direction(setup, rng, times.size(), false,
                     [](double r) { return 0.8 - 0.5 * r; });
  const auto nodalW = materialDirection([](double r) { return 1.0 - 0.5 * r; });

  // The base moved along w by eps: the moduli perturbed at the memory
  // points (relative, eps delta_w / modulus; the branch moduli follow as
  // the same fractions), the inverse times by eps delta itau_w, the
  // thickness by eps delta I_w.
  const auto& base = setup.rheology;
  auto operatorAt = [&](double eps, MemoryPointSet set,
                        const PointMaterialDirection& along) {
    auto branches = std::vector<Rheology::Branch>{};
    for (auto k = std::size_t{0}; k < base.BranchCount(); ++k) {
      auto branch = base.BranchAt(k);
      for (auto i = std::size_t{0}; i < branch.itau.size(); ++i) {
        branch.itau[i] += eps * w.itau[k][i];
      }
      branches.push_back(std::move(branch));
    }
    const ViscoelasticOperator reference(base, kLmax, {.memory_points = set});
    const auto muU = reference.UnrelaxedShearAtPoints();
    const auto kappa = reference.BulkModulusAtPoints();
    ViscoelasticOperator::Options::PointModuli moduli;
    moduli.shear.assign(muU.size(), 0.0);
    moduli.bulk.assign(muU.size(), 0.0);
    for (auto i = std::size_t{0}; i < muU.size(); ++i) {
      if (muU[i] > 0.0) moduli.shear[i] = eps * along.mu[i] / muU[i];
      if (kappa[i] > 0.0) moduli.bulk[i] = eps * along.kappa[i] / kappa[i];
    }
    return ViscoelasticOperator(Rheology::Create(material, branches).value(),
                                kLmax, {.memory_points = set, .moduli = moduli});
  };
  auto movedThickness = [&](double eps) {
    auto out = thickness;
    for (auto n = std::size_t{0}; n < out.size(); ++n) {
      for (auto i = std::size_t{0}; i < out[n].size(); ++i) {
        out[n][i] += eps * w.iceThickness[n][i];
      }
    }
    return out;
  };

  for (auto set : {MemoryPointSet::Gll, MemoryPointSet::Gauss})
  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Axial,
                        RotationalFeedback::Full})
  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const bool rotating = rotation != RotationalFeedback::None;
    // rotational feedback at the default points, one scheme each
    if (rotating && (set != MemoryPointSet::Gauss ||
                     (rotation == RotationalFeedback::Axial) !=
                         (scheme == Scheme::BackwardEuler))) {
      continue;
    }
    ViscoelasticOperator op(setup.rheology, kLmax, {.memory_points = set});
    u.material = AtMemoryPoints(op.Points(), nodalU);
    w.material = AtMemoryPoints(op.Points(), nodalW);
    SeaLevelEvolution evolution(op, setup.solver, setup.state,
                                {.scheme = scheme,
                                 .rotation = rotation,
                                 .rtol = 1e-12,
                                 .record_stride = 1});
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, FrozenLoad(setup, times, thickness)));
    TangentLinearSolver tangent(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .record = true});
    const auto runU = tangent.Run(times, u, &trajectory);
    const auto runW = tangent.Run(times, w, &trajectory);
    AdjointSolver adjoint(evolution, {.method = AdjointMethod::Discrete,
                                      .rtol = 1e-12,
                                      .material_kernels = true});
    const auto first = adjoint.IceHistoryKernels(times, measurements, &trajectory);
    const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
        times, measurements, runU, u.itau, trajectory, {}, u.material);
    const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
        times, measurements, runW, w.itau, trajectory, {}, w.material);
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      EXPECT_LT(MaxDiff(kernelsU.thickness[n], first.thickness[n]),
                1e-12 * MaxAbs(first.thickness[n]) + 1e-300);
    }
    EXPECT_LT(MaxDiff(kernelsU.itauKernel[0], first.itauKernel[0]),
              1e-12 * MaxAbs(first.itauKernel[0]));
    EXPECT_LT(MaxDiff(kernelsU.kappaKernel, first.kappaKernel),
              1e-12 * MaxAbs(first.kappaKernel));
    EXPECT_LT(MaxDiff(kernelsU.muKernel, first.muKernel),
              1e-12 * MaxAbs(first.muKernel));
    EXPECT_GT(MaxAbs(alongU.kappaKernel), 0.0);
    EXPECT_GT(MaxAbs(alongU.muKernel), 0.0);

    const auto huW = Pairing(setup, op.Points(), alongU, w);
    const auto hwU = Pairing(setup, op.Points(), alongW, u);
    EXPECT_NEAR(huW, hwU, 1e-9 * std::abs(huW))
        << "scheme " << static_cast<int>(scheme) << " rotation "
        << static_cast<int>(rotation);
    SecondOrderDirection uw;
    uw.itauU = u.itau;
    uw.itauV = w.itau;
    uw.materialU = u.material;
    uw.materialV = w.material;
    const auto second = tangent.RunSecondOrder(times, runU, runW, uw, &trajectory);
    const auto forward = Objective(setup, second, measurements);
    EXPECT_NEAR(forward, huW, 1e-9 * std::abs(forward))
        << "scheme " << static_cast<int>(scheme) << " rotation "
        << static_cast<int>(rotation);

    auto gradientAt = [&](double eps) {
      const auto movedOp = operatorAt(eps, set, w.material);
      SeaLevelEvolution movedEvolution(movedOp, setup.solver, setup.state,
                                       {.scheme = scheme,
                                        .rotation = rotation,
                                        .rtol = 1e-12,
                                        .record_stride = 1});
      const auto movedTrajectory = ForwardTrajectory::FromSnapshots(
          movedEvolution.Run(times, FrozenLoad(setup, times, movedThickness(eps))));
      AdjointSolver movedAdjoint(movedEvolution, {.method = AdjointMethod::Discrete,
                                                  .rtol = 1e-12,
                                                  .material_kernels = true});
      return Pairing(setup, op.Points(),
                     movedAdjoint.IceHistoryKernels(times, measurements,
                                                    &movedTrajectory),
                     u);
    };
    // The derivatives along w paired with u.
    const auto predicted = Pairing(setup, op.Points(), alongW, u);
    auto fd = std::map<double, double>{};
    for (double eps : {0.1, 0.05}) {
      fd[eps] = (gradientAt(eps) - gradientAt(-eps)) / (2.0 * eps);
    }
    const auto r1 = std::abs(fd[0.1] - predicted);
    const auto r2 = std::abs(fd[0.05] - predicted);
    EXPECT_GT(r1 / r2, 3.0) << "scheme " << static_cast<int>(scheme) << " "
                            << fd[0.1] << " " << fd[0.05] << " " << predicted;
    EXPECT_LT(r1 / r2, 5.0) << "scheme " << static_cast<int>(scheme);
    const auto richardson = (4.0 * fd[0.05] - fd[0.1]) / 3.0;
    EXPECT_NEAR(richardson, predicted, 1e-6 * std::abs(predicted))
        << "scheme " << static_cast<int>(scheme) << " rotation "
        << static_cast<int>(rotation);
  }
}
