// The elastic problem as the evolution asked for one solve: a time grid
// of one node. The tangent-linear runs, both adjoint sweeps and the
// second-order sweeps take it, in the forcing (a load and a point
// source) and in the moduli.

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

MaterialDirection Nodal(const Scene& setup,
                        const std::function<double(double)>& profile) {
  const auto& material = setup.material;
  const auto& mesh = material.Mesh();
  const auto [k0, m0] = material.Moduli().KappaMu();
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
}

}  // namespace

TEST(StaticDerivatives, OneNodeIsTheElasticProblemInEveryClass) {
  // One time node: the run is the instantaneous elastic response, the
  // tangent-linear run its derivative in the ice, the moduli and a
  // source's moment, both
  // sweeps' kernels pair with the direction to the tangent's objective
  // (the continuous sweep is its jump alone and exact), and the Hessian
  // is symmetric and equal to the second-order run; with an ocean and
  // each rotational feedback.
  auto setup = Scene::Make(1e21);
  std::mt19937 rng(103);
  const auto times = std::vector<double>{0.0};
  const auto load = setup.Smooth(rng, 100.0 * setup.Sigma());
  const auto nodalU = Nodal(
      setup, [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); });
  const auto nodalW = Nodal(setup, [](double r) { return 1.0 - 0.5 * r; });
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[0].displacementDual = setup.Smooth(rng, 0.5);
  measurements[0].potentialDual = setup.Smooth(rng, 0.7);
  measurements[0].tangentialPotentialDual = setup.Smooth(rng, 0.6);
  measurements[0].toroidalPotentialDual = setup.Smooth(rng, 0.8);
  measurements[0].angularVelocityDual = {0.2, -0.4, 0.8};
  // a point source beside the load, and its moment among the parameters
  const auto& sle = setup.solver.SolverScales();
  const auto surface = setup.material.Radius() *
                       setup.model.ModelScales().Length() / sle.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  const auto at = BodyPoint{.radius = surface - 400e3 / sle.Length(),
                            .colatitude = 1.1,
                            .longitude = 4.0};
  auto sourceOf = [&](double a, double b, double c) {
    return std::vector<PointSource>{
        {.at = at,
         .moment = {.rr = a * unit, .tt = b * unit, .pp = -(a + b) * unit,
                    .rt = c * unit, .rp = -a * unit, .tp = b * unit}}};
  };
  for (auto rotation : {RotationalFeedback::None, RotationalFeedback::Axial,
                        RotationalFeedback::Full}) {
    const auto label = "rotation " + std::to_string(static_cast<int>(rotation));
    ViscoelasticOperator op(setup.rheology, kLmax);
    Evolution evolution(op, setup.solver, setup.state,
                        {.rotation = rotation, .rtol = 1e-12, .record_stride = 1});
    Evolution::Forcing forcing;
    forcing.directLoad = [&](double) { return load; };
    forcing.sources = [&](double) { return sourceOf(0.9, -0.2, 0.3); };
    const auto run = evolution.Run(times, forcing);
    ASSERT_EQ(run.size(), 1u);
    const auto trajectory = ForwardTrajectory::FromSnapshots(run);
    TangentPerturbation u, w;
    u.iceThickness = {setup.Smooth(rng, 20.0 * setup.Metre())};
    u.material = AtMemoryPoints(op.Points(), nodalU);
    u.sources = {sourceOf(-0.4, 0.8, 0.5)};
    w.sources = {sourceOf(0.2, 0.3, -0.6)};
    w.iceThickness = {setup.Smooth(rng, 20.0 * setup.Metre())};
    w.material = AtMemoryPoints(op.Points(), nodalW);
    const auto& mesh = setup.material.Mesh();
    auto pair = [&](const auto& kernels, const TangentPerturbation& d) {
      return kernels.momentKernels[0][0].Contract(d.sources[0][0].moment) +
             setup.Surface(kernels.thickness[0], d.iceThickness[0]) +
             VolumePairing(op.Points(), mesh, kernels.kappaKernel, d.material.kappa) +
             VolumePairing(op.Points(), mesh, kernels.muKernel, d.material.mu);
    };
    auto exact = 0.0;
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver tangent(evolution,
                                  {.method = method, .rtol = 1e-12, .record = true});
      const auto runU = tangent.Run(times, u, &trajectory);
      const auto runW = tangent.Run(times, w, &trajectory);
      const auto along = Objective(setup, runU, measurements);
      ASSERT_NE(along, 0.0) << label;
      if (method == AdjointMethod::Discrete) exact = along;
      EXPECT_NEAR(along, exact, 1e-10 * std::abs(exact)) << label;
      AdjointSolver adjoint(evolution, {.method = method,
                                        .rtol = 1e-12,
                                        .material_kernels = true,
                                        .moment_points = {at}});
      const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
      EXPECT_NEAR(pair(first, u), along,
                  1e-8 * std::abs(along))
          << label << " method " << static_cast<int>(method);
      const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
          times, measurements, runU, {}, trajectory, {}, u.material);
      const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
          times, measurements, runW, {}, trajectory, {}, w.material);
      const auto uw = pair(alongU, w);
      const auto wu = pair(alongW, u);
      SecondOrderDirection direction;
      direction.materialU = u.material;
      direction.materialV = w.material;
      const auto forward = Objective(
          setup, tangent.RunSecondOrder(times, runU, runW, direction, &trajectory),
          measurements);
      ASSERT_NE(forward, 0.0) << label;
      EXPECT_NEAR(uw, wu, 1e-8 * std::abs(uw))
          << label << " method " << static_cast<int>(method);
      EXPECT_NEAR(uw, forward, 1e-8 * std::abs(forward))
          << label << " method " << static_cast<int>(method);
    }
  }
}

TEST(StaticDerivatives, AnElasticBodyWithNoRelaxingLayer) {
  // The same on a rheology with no branch at all (the purely elastic
  // body): the classes run with nothing to relax, the kernels in the
  // moduli pair with the tangent and the Hessian is symmetric.
  auto setup = Scene::Make(1e21);
  const auto elastic = Rheology::Maxwell(setup.material, std::map<std::size_t, RadialValue>{}).value();
  std::mt19937 rng(109);
  const auto times = std::vector<double>{0.0};
  const auto load = setup.Smooth(rng, 100.0 * setup.Sigma());
  const auto nodalU = Nodal(
      setup, [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); });
  const auto nodalW = Nodal(setup, [](double r) { return 1.0 - 0.5 * r; });
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[0].displacementDual = setup.Smooth(rng, 0.5);
  ViscoelasticOperator op(elastic, kLmax);
  Evolution evolution(op, setup.solver, setup.state,
                      {.rtol = 1e-12, .record_stride = 1});
  const auto run = evolution.Run(times, [&](double) { return load; });
  const auto trajectory = ForwardTrajectory::FromSnapshots(run);
  TangentPerturbation u, w;
  u.iceThickness = {setup.Smooth(rng, 20.0 * setup.Metre())};
  u.material = AtMemoryPoints(op.Points(), nodalU);
  w.iceThickness = {setup.Smooth(rng, 20.0 * setup.Metre())};
  w.material = AtMemoryPoints(op.Points(), nodalW);
  const auto& mesh = setup.material.Mesh();
  auto pair = [&](const auto& kernels, const TangentPerturbation& d) {
    return setup.Surface(kernels.thickness[0], d.iceThickness[0]) +
           VolumePairing(op.Points(), mesh, kernels.kappaKernel, d.material.kappa) +
           VolumePairing(op.Points(), mesh, kernels.muKernel, d.material.mu);
  };
  for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
    TangentLinearSolver tangent(evolution,
                                {.method = method, .rtol = 1e-12, .record = true});
    const auto runU = tangent.Run(times, u, &trajectory);
    const auto runW = tangent.Run(times, w, &trajectory);
    const auto along = Objective(setup, runU, measurements);
    ASSERT_NE(along, 0.0);
    AdjointSolver adjoint(evolution,
                          {.method = method, .rtol = 1e-12, .material_kernels = true});
    const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
    EXPECT_NEAR(pair(first, u), along, 1e-8 * std::abs(along));
    const auto alongU = adjoint
                            .SecondOrderKernels(times, measurements, runU, {},
                                                trajectory, {}, u.material)
                            .second;
    const auto alongW = adjoint
                            .SecondOrderKernels(times, measurements, runW, {},
                                                trajectory, {}, w.material)
                            .second;
    SecondOrderDirection direction;
    direction.materialU = u.material;
    direction.materialV = w.material;
    const auto forward = Objective(
        setup, tangent.RunSecondOrder(times, runU, runW, direction, &trajectory),
        measurements);
    EXPECT_NEAR(pair(alongU, w), pair(alongW, u), 1e-8 * std::abs(forward));
    EXPECT_NEAR(pair(alongU, w), forward, 1e-8 * std::abs(forward));
  }
}
