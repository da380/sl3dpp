// The derivative classes on time grids of unequal steps (1 : 3 : 0.5 : 2 :
// 1.5): the tangent-linear runs against differences of the forward run
// (loads, ice thickness, relaxation times, moduli), the discrete sweeps
// as the tangents' exact transposes, the continuous forms converging to
// the discrete ones at the schemes' orders as every step is halved, the
// discrete Hessian symmetric and equal to the second-order run, and the
// decay-time datum over such a window — 1D and, under lateral viscosity,
// 3D.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <map>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/Measurements.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/TangentLinear.hpp"
#include "sl3dpp/TangentLinear3D.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;

namespace {

constexpr auto kYear = 3.15576e7;  // s

/// The prem8 deck's body with a Maxwell mantle of the given viscosity
/// (Pa s) between the CMB and 220 km depth, at the given degree, with its
/// elastic solver and the analytic test state (the kernel tests' scene).
struct Scene {
  int lmax = 0;
  EarthModel model;
  Material material;
  Rheology rheology;
  GRDSolver solver;
  GRDSolver::State state;

  static Scene Make(int lmax, double viscositySI) {
    auto deck =
        Deck::Read(
            std::filesystem::path(SL3DPP_TEST_DATA_DIR) / "prem8.elastic.deck",
            DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0))
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
    ViscoelasticOperator op(rheology, lmax);
    GRDSolver solver(SurfaceGrid(lmax), op.ElasticLoveNumbers(), SeaLevelParameters{});
    Fingerprint fingerprint(lmax, op.ElasticLoveNumbers());
    fingerprint.SetAnalyticTestState(0.0);
    const auto length = solver.SolverScales().Length();
    auto toNd = [&](std::vector<double> si) {
      for (auto& v : si) v /= length;
      return si;
    };
    auto state = solver.MakeState(toNd(fingerprint.IceThickness()),
                                  toNd(fingerprint.SeaLevel()));
    return Scene{lmax,
                 std::move(model),
                 std::move(material),
                 std::move(rheology),
                 std::move(solver),
                 std::move(state)};
  }

  [[nodiscard]] double Metre() const { return 1.0 / solver.SolverScales().Length(); }
  [[nodiscard]] double Kyr() const { return 1e3 * kYear / solver.SolverScales().Time(); }
  [[nodiscard]] double Sigma() const {
    return 1.0 / solver.SolverScales().Factor(Dimensions{.mass = 1, .length = -2});
  }

  [[nodiscard]] std::vector<double> Smooth(std::mt19937& rng, double amplitude) const {
    const auto& grid = solver.Grid();
    std::normal_distribution<double> normal;
    auto coeffs = grid.ZeroCoefficients();
    for (auto l = 0; l <= lmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        coeffs[SurfaceGrid::Index(l, m)] = {normal(rng), m == 0 ? 0.0 : normal(rng)};
      }
    }
    auto field = grid.Synthesise(coeffs);
    for (auto& v : field) v *= amplitude;
    return field;
  }

  [[nodiscard]] double Surface(std::span<const double> f, std::span<const double> g) const {
    const auto a = solver.Radius();
    auto fg = std::vector<double>(f.size());
    for (auto i = std::size_t{0}; i < f.size(); ++i) fg[i] = f[i] * g[i];
    return a * a * solver.Grid().Integrate(fg);
  }

  /// A smooth nodal perturbation of branch 0's inverse relaxation time,
  /// material units, supported where the branch relaxes.
  [[nodiscard]] std::vector<std::vector<double>> DeltaItau(
      double fraction, const std::function<double(double)>& shape =
                           [](double r) { return 0.3 + 0.7 * r * r; }) const {
    const auto& base = rheology.BranchAt(0);
    const auto& mesh = material.Mesh();
    auto top = 0.0;
    for (auto v : base.itau) top = std::max(top, v);
    auto ditau = std::vector<double>(base.itau.size(), 0.0);
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      for (auto q = 0; q < mesh.NGll(); ++q) {
        const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
        if (base.itau[i] > 0.0) ditau[i] = fraction * shape(mesh.Radius(e, q)) * top;
      }
    }
    return {ditau};
  }
};

/// The objective the measurements define on a run's snapshots.
template <class Snapshot>
double Objective(const Scene& setup, const std::vector<Snapshot>& snapshots,
                 const std::map<std::size_t, Measurement>& measurements) {
  auto total = 0.0;
  for (const auto& [node, m] : measurements) {
    const auto& snap = snapshots[node];
    for (const auto& [dual, field] :
         {std::pair{&m.relativeSeaLevelDual, &snap.relativeSeaLevelChange},
          std::pair{&m.displacementDual, &snap.displacement},
          std::pair{&m.potentialDual, &snap.potential},
          std::pair{&m.tangentialPotentialDual, &snap.tangentialPotential},
          std::pair{&m.toroidalPotentialDual, &snap.toroidalPotential}}) {
      if (!dual->empty() && !field->empty()) total += setup.Surface(*dual, *field);
    }
    for (auto j = 0; j < 3; ++j) total += m.angularVelocityDual[j] * snap.angularVelocity[j];
  }
  return total;
}

/// A full measurement set: every dual, at the first node, at `mid` and at
/// the last.
std::map<std::size_t, Measurement> FullMeasurements(const Scene& setup, std::mt19937& rng,
                                                    std::size_t mid, std::size_t last) {
  auto out = std::map<std::size_t, Measurement>{};
  out[0].relativeSeaLevelDual = setup.Smooth(rng, 0.3);
  out[mid].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  out[mid].displacementDual = setup.Smooth(rng, -0.5);
  out[last].displacementDual = setup.Smooth(rng, 0.7);
  out[last].potentialDual = setup.Smooth(rng, 0.4);
  out[last].tangentialPotentialDual = setup.Smooth(rng, 0.6);
  out[last].angularVelocityDual = {0.2, -0.4, 0.8};
  return out;
}

double MaxAbs(std::span<const double> f) {
  auto out = 0.0;
  for (auto v : f) out = std::max(out, std::abs(v));
  return out;
}

double MaxDiff(std::span<const double> a, std::span<const double> b) {
  auto out = 0.0;
  for (auto i = std::size_t{0}; i < a.size(); ++i) out = std::max(out, std::abs(a[i] - b[i]));
  return out;
}

double Cap(double theta, double phi, double colat0, double lon0, double width) {
  const double c = std::cos(theta) * std::cos(colat0) +
                   std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
  const auto d = std::acos(std::clamp(c, -1.0, 1.0));
  return std::exp(-0.5 * std::pow(d / width, 2));
}

/// The steps of the uneven grids, in kyr: no two neighbours alike, the
/// largest six times the smallest.
const auto kSteps = std::vector<double>{1.0, 3.0, 0.5, 2.0, 1.5};

/// The time grid of the given steps (in `unit`), each step split into
/// `split` equal parts: the coarse grid's node k is the split grid's node
/// k * split.
std::vector<double> UnevenTimes(double unit, std::span<const double> steps,
                                int split = 1) {
  auto times = std::vector<double>{0.0};
  auto at = 0.0;
  for (const auto step : steps) {
    for (auto part = 1; part <= split; ++part) {
      times.push_back(at + step * unit * part / split);
    }
    at += step * unit;
  }
  return times;
}

/// The node of a grid at time t.
std::size_t NodeOf(std::span<const double> times, double t) {
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) return n;
  }
  throw std::runtime_error("the test history is read off its nodes");
}

/// A per-node load history as the forward driver's callable.
std::function<std::vector<double>(double)> Held(std::vector<double> times,
                                                std::vector<std::vector<double>> loads) {
  return [times = std::move(times), loads = std::move(loads)](double t) {
    return loads[NodeOf(times, t)];
  };
}

/// The frozen-shoreline load of a per-node thickness history relative to
/// its first node, per node.
std::vector<std::vector<double>> FrozenLoads(
    const Scene& setup, const std::vector<std::vector<double>>& thickness) {
  const auto& C = setup.state.oceanFunction;
  const auto rhoI = setup.solver.IceDensity();
  auto out = std::vector<std::vector<double>>{};
  for (const auto& T : thickness) {
    auto load = std::vector<double>(T.size());
    for (auto i = std::size_t{0}; i < T.size(); ++i) {
      load[i] = rhoI * (1.0 - C[i]) * (T[i] - thickness[0][i]);
    }
    out.push_back(std::move(load));
  }
  return out;
}

std::vector<std::vector<double>> Plus(const std::vector<std::vector<double>>& a,
                                      const std::vector<std::vector<double>>& b,
                                      double eps = 1.0) {
  auto out = a;
  for (auto n = std::size_t{0}; n < out.size(); ++n) {
    for (auto i = std::size_t{0}; i < out[n].size(); ++i) out[n][i] += eps * b[n][i];
  }
  return out;
}

/// A per-node history of a field times a profile of t / span.
std::vector<std::vector<double>> Scaled(std::span<const double> times, double span,
                                        const std::vector<double>& field,
                                        const std::function<double(double)>& profile) {
  auto out = std::vector<std::vector<double>>{};
  for (const auto t : times) {
    auto f = field;
    for (auto& x : f) x *= profile(t / span);
    out.push_back(std::move(f));
  }
  return out;
}

/// A direction in kappa and mu at the operator's memory points on the
/// solid: a radial profile times the point moduli (kappa at half the
/// shear's share), at a tenth of the moduli.
PointMaterialDirection PointDirection(const ViscoelasticOperator& op,
                                      const Material& material,
                                      const std::function<double(double)>& profile) {
  const auto& mesh = material.Mesh();
  const auto& points = op.Points();
  const auto muU = op.UnrelaxedShearAtPoints();
  const auto kappa0 = op.BulkModulusAtPoints();
  PointMaterialDirection d;
  d.kappa.assign(muU.size(), 0.0);
  d.mu.assign(muU.size(), 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e)) continue;
    for (auto p = 0; p < points.count; ++p) {
      const auto i = static_cast<std::size_t>(e) * points.count + p;
      const auto value = 0.1 * profile(points.RadiusAt(mesh, e, p));
      d.kappa[i] = 0.5 * value * kappa0[i];
      d.mu[i] = value * muU[i];
    }
  }
  return d;
}

/// The operator moved by eps along a direction: its moduli at the points
/// along d (relative perturbations; the branch fractions and the
/// relaxation times held) and its inverse relaxation times along ditau.
ViscoelasticOperator MovedOperator(const Scene& setup, const ViscoelasticOperator& op,
                                   double eps, const PointMaterialDirection& d,
                                   const std::vector<std::vector<double>>& ditau = {}) {
  const auto muU = op.UnrelaxedShearAtPoints();
  const auto kappa0 = op.BulkModulusAtPoints();
  ViscoelasticOperator::Options::PointModuli moduli;
  if (!d.Empty()) {
    moduli.shear.assign(muU.size(), 0.0);
    moduli.bulk.assign(muU.size(), 0.0);
    for (auto i = std::size_t{0}; i < muU.size(); ++i) {
      if (!d.mu.empty() && muU[i] > 0.0) moduli.shear[i] = eps * d.mu[i] / muU[i];
      if (!d.kappa.empty() && kappa0[i] > 0.0) moduli.bulk[i] = eps * d.kappa[i] / kappa0[i];
    }
  }
  auto branches = std::vector<Rheology::Branch>{};
  for (auto k = std::size_t{0}; k < setup.rheology.BranchCount(); ++k) {
    auto branch = setup.rheology.BranchAt(k);
    if (!ditau.empty()) {
      for (auto i = std::size_t{0}; i < branch.itau.size(); ++i) {
        branch.itau[i] += eps * ditau[k][i];
      }
    }
    branches.push_back(std::move(branch));
  }
  const auto rheology = Rheology::Create(setup.material, branches).value();
  return ViscoelasticOperator(rheology, setup.lmax, {.moduli = moduli});
}

/// Central differences at eps and eps / 2 against a prediction: the
/// ratio of their residuals (four at second order) and the Richardson
/// value.
struct Differences {
  double ratio = 0.0;
  double richardson = 0.0;
};
Differences Difference(const std::function<double(double)>& f, double eps,
                       double predicted) {
  const auto wide = (f(eps) - f(-eps)) / (2.0 * eps);
  const auto narrow = (f(0.5 * eps) - f(-0.5 * eps)) / eps;
  return {.ratio = std::abs(wide - predicted) / std::abs(narrow - predicted),
          .richardson = (4.0 * narrow - wide) / 3.0};
}

/// The relative gap between two numbers.
double Gap(double got, double want) { return std::abs(got - want) / std::abs(want); }

std::string SchemeName(Scheme scheme) {
  return scheme == Scheme::BackwardEuler ? "backward Euler" : "trapezoid";
}

/// The surface pairing summed over the nodes.
double NodePairing(const Scene& setup, const std::vector<std::vector<double>>& kernel,
                   const std::vector<std::vector<double>>& d) {
  auto out = 0.0;
  for (auto n = std::size_t{0}; n < d.size(); ++n) out += setup.Surface(kernel[n], d[n]);
  return out;
}

/// A 1D kernel object (kernels or their derivatives) paired with a
/// direction in the thickness, the relaxation times and the moduli.
template <class K>
double Pairing(const Scene& setup, const ViscoelasticOperator& op, const K& kernels,
               const TangentPerturbation& d) {
  const auto& mesh = setup.material.Mesh();
  auto total = NodePairing(setup, kernels.thickness, d.iceThickness);
  if (!d.itau.empty()) total += VolumePairing(mesh, kernels.itauKernel[0], d.itau[0]);
  if (!d.material.kappa.empty()) {
    total += VolumePairing(op.Points(), mesh, kernels.kappaKernel, d.material.kappa);
  }
  if (!d.material.mu.empty()) {
    total += VolumePairing(op.Points(), mesh, kernels.muKernel, d.material.mu);
  }
  return total;
}

}  // namespace

// -- 1D -------------------------------------------------------------------

TEST(UnevenSteps, TheTangentIsTheDerivativeOfTheRun) {
  // Both schemes on the uneven grid: along the loads and the ice
  // thickness (non-zero at the first node) the run is the difference of
  // two forward runs, node by node, both methods (the frozen problem is
  // linear in them); along the relaxation times and the moduli the
  // discrete run is the derivative of the forward run — central
  // differences at second order, Richardson landing on it.
  auto setup = Scene::Make(16, 1e21);
  std::mt19937 rng(41);
  const auto times = UnevenTimes(setup.Kyr(), kSteps);
  const auto nodes = times.size();
  auto loads = std::vector<std::vector<double>>{};
  auto deltas = std::vector<std::vector<double>>{};
  auto thickness = std::vector<std::vector<double>>{};
  auto deltaIce = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    deltas.push_back(setup.Smooth(rng, 10.0 * setup.Sigma()));
    thickness.push_back(setup.Smooth(rng, 100.0 * setup.Metre()));
    deltaIce.push_back(setup.Smooth(rng, 10.0 * setup.Metre()));
  }
  const auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);
  const auto ditau = setup.DeltaItau(0.1);

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = SchemeName(scheme);
    ViscoelasticOperator op(setup.rheology, setup.lmax);
    const Evolution::Options options{.scheme = scheme, .rtol = 1e-12, .record_stride = 1};
    Evolution evolution(op, setup.solver, setup.state, options);
    const auto base = evolution.Run(times, Held(times, loads));
    const auto plus = evolution.Run(times, Held(times, Plus(loads, deltas)));
    const auto iceBase = evolution.Run(times, Held(times, FrozenLoads(setup, thickness)));
    const auto icePlus =
        evolution.Run(times, Held(times, FrozenLoads(setup, Plus(thickness, deltaIce))));
    const auto trajectory = ForwardTrajectory::FromSnapshots(base);

    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver tangent(evolution, {.method = method, .rtol = 1e-12});
      TangentPerturbation alongLoads, alongIce;
      alongLoads.directLoad = deltas;
      alongIce.iceThickness = deltaIce;
      const auto runLoads = tangent.Run(times, alongLoads);
      const auto runIce = tangent.Run(times, alongIce);
      for (auto n = std::size_t{0}; n < nodes; ++n) {
        for (auto field : {&Evolution::Snapshot::relativeSeaLevelChange,
                           &Evolution::Snapshot::displacement,
                           &Evolution::Snapshot::potential,
                           &Evolution::Snapshot::tangentialPotential}) {
          for (const auto& [run, p, b] : {std::tuple{&runLoads, &plus, &base},
                                          std::tuple{&runIce, &icePlus, &iceBase}}) {
            auto difference = (*p)[n].*field;
            for (auto i = std::size_t{0}; i < difference.size(); ++i) {
              difference[i] -= ((*b)[n].*field)[i];
            }
            // (the ice's first node carries no load: both sides zero there)
            EXPECT_LE(MaxDiff((*run)[n].*field, difference), 1e-8 * MaxAbs(difference))
                << label << ", node " << n << ", method " << static_cast<int>(method);
          }
        }
      }
    }

    // The relaxation times: the forward run on the moved rheology.
    TangentLinearSolver tangent(evolution,
                                {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    TangentPerturbation alongItau;
    alongItau.itau = ditau;
    const auto predictedItau =
        Objective(setup, tangent.Run(times, alongItau, &trajectory), measurements);
    const auto itau = Difference(
        [&](double eps) {
          const auto moved = MovedOperator(setup, op, eps, {}, ditau);
          Evolution trial(moved, setup.solver, setup.state, options);
          return Objective(setup, trial.Run(times, Held(times, loads)), measurements);
        },
        0.02, predictedItau);
    std::cout << std::format(
        "[uneven 1D tangent, {}] itau: ratio {:.3f}, Richardson gap {:.2e}\n", label,
        itau.ratio, Gap(itau.richardson, predictedItau));
    EXPECT_GT(itau.ratio, 3.0) << label;
    EXPECT_LT(itau.ratio, 5.0) << label;
    EXPECT_NEAR(itau.richardson, predictedItau, 1e-6 * std::abs(predictedItau)) << label;

    // The moduli at the memory points: the forward run with them moved.
    TangentPerturbation alongMaterial;
    alongMaterial.material = PointDirection(op, setup.material, [](double r) {
      return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r);
    });
    const auto predictedMaterial =
        Objective(setup, tangent.Run(times, alongMaterial, &trajectory), measurements);
    const auto material = Difference(
        [&](double eps) {
          const auto moved = MovedOperator(setup, op, eps, alongMaterial.material);
          Evolution trial(moved, setup.solver, setup.state, options);
          return Objective(setup, trial.Run(times, Held(times, loads)), measurements);
        },
        0.1, predictedMaterial);
    std::cout << std::format(
        "[uneven 1D tangent, {}] moduli: ratio {:.3f}, Richardson gap {:.2e}\n", label,
        material.ratio, Gap(material.richardson, predictedMaterial));
    EXPECT_GT(material.ratio, 3.0) << label;
    EXPECT_LT(material.ratio, 5.0) << label;
    EXPECT_NEAR(material.richardson, predictedMaterial, 1e-6 * std::abs(predictedMaterial))
        << label;
  }
}

TEST(UnevenSteps, TheDiscreteSweepIsTheTangentsTranspose) {
  // Both schemes on the uneven grid, a full measurement set: the discrete
  // sweep's load kernels (node 0 included), thickness view, relaxation-time
  // kernel and moduli kernels each pair with their direction to the
  // discrete tangent's objective change; a stride-2 recording serves the
  // same kernels, its gaps rebuilt by steps of their own sizes; and the
  // views agree on the grid — the distributional view (the density
  // weighted by the trapezoid's node weights, plus the observations'
  // lumps) and the quelled view (the rates against the thickness change
  // over each interval) give the thickness view's numbers.
  auto setup = Scene::Make(16, 1e21);
  std::mt19937 rng(43);
  const auto times = UnevenTimes(setup.Kyr(), kSteps);
  const auto nodes = times.size();
  auto loads = std::vector<std::vector<double>>{};
  auto deltas = std::vector<std::vector<double>>{};
  auto deltaIce = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    deltas.push_back(setup.Smooth(rng, 10.0 * setup.Sigma()));
    deltaIce.push_back(setup.Smooth(rng, 10.0 * setup.Metre()));
  }
  const auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);
  const auto ditau = setup.DeltaItau(0.1);
  const auto& mesh = setup.material.Mesh();

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = SchemeName(scheme);
    ViscoelasticOperator op(setup.rheology, setup.lmax);
    Evolution evolution(op, setup.solver, setup.state,
                        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory::FromSnapshots(evolution.Run(times, Held(times, loads)));
    const auto d = PointDirection(op, setup.material, [](double r) { return 1.0 - 0.5 * r; });
    const auto kernels = AdjointSolver(evolution, {.method = AdjointMethod::Discrete,
                                                   .rtol = 1e-12,
                                                   .material_kernels = true})
                             .FirstOrderKernels(times, measurements, &trajectory);
    TangentLinearSolver tangent(evolution,
                                {.method = AdjointMethod::Discrete, .rtol = 1e-12});
    auto along = [&](const TangentPerturbation& p) {
      return Objective(setup, tangent.Run(times, p, &trajectory), measurements);
    };
    TangentPerturbation pLoads, pIce, pItau, pMaterial;
    pLoads.directLoad = deltas;
    pIce.iceThickness = deltaIce;
    pItau.itau = ditau;
    pMaterial.material = d;
    const auto itauFull = VolumePairing(mesh, kernels.itauKernel[0], ditau[0]);
    for (const auto& [name, paired, p] :
         {std::tuple{"loads", NodePairing(setup, kernels.directLoadKernels, deltas), &pLoads},
          std::tuple{"ice", NodePairing(setup, kernels.thickness, deltaIce), &pIce},
          std::tuple{"itau", itauFull, &pItau},
          std::tuple{"moduli",
                     VolumePairing(op.Points(), mesh, kernels.kappaKernel, d.kappa) +
                         VolumePairing(op.Points(), mesh, kernels.muKernel, d.mu),
                     &pMaterial}}) {
      const auto want = along(*p);
      ASSERT_NE(want, 0.0) << label << ", " << name;
      std::cout << std::format("[uneven 1D transpose, {}] {}: gap {:.2e}\n", label, name,
                               Gap(paired, want));
      EXPECT_NEAR(paired, want, 1e-8 * std::abs(want)) << label << ", " << name;
    }

    // A stride-2 recording: the gaps rebuilt by backward steps of their
    // own sizes (the reverse cursor) and forward ones (the tangent's). A
    // backward trapezoid step amplifies rounding by exp(h), here about 800
    // on the rebuilt 3 kyr step against the 450-year relaxation time, so
    // the reverse cursor keeps fewer digits than the forward one; a step
    // of the wrong size would miss by order one.
    Evolution sparse(op, setup.solver, setup.state,
                     {.scheme = scheme, .rtol = 1e-12, .record_stride = 2});
    const auto sparseTrajectory =
        ForwardTrajectory::FromSnapshots(sparse.Run(times, Held(times, loads)));
    const auto again = AdjointSolver(evolution, {.method = AdjointMethod::Discrete,
                                                 .rtol = 1e-12,
                                                 .material_kernels = true})
                           .FirstOrderKernels(times, measurements, &sparseTrajectory);
    const auto itauSparse = VolumePairing(mesh, again.itauKernel[0], ditau[0]);
    std::cout << std::format("[uneven 1D transpose, {}] stride 2: itau gap {:.2e}\n", label,
                             Gap(itauSparse, itauFull));
    EXPECT_NEAR(itauSparse, itauFull, 1e-5 * std::abs(itauFull)) << label;
    const auto viaSparse =
        Objective(setup, tangent.Run(times, pItau, &sparseTrajectory), measurements);
    EXPECT_NEAR(viaSparse, along(pItau), 1e-10 * std::abs(viaSparse)) << label;

    // The views on the uneven grid, against a thickness direction zero at
    // the first node (the quelled view carries no baseline).
    auto rising = deltaIce;
    rising[0].assign(rising[0].size(), 0.0);
    const auto thicknessView = NodePairing(setup, kernels.thickness, rising);
    auto quelled = 0.0;
    for (auto k = std::size_t{0}; k + 1 < nodes; ++k) {
      const auto h = times[k + 1] - times[k];
      auto rate = rising[k + 1];
      for (auto i = std::size_t{0}; i < rate.size(); ++i) {
        rate[i] = (rising[k + 1][i] - rising[k][i]) / h;
      }
      quelled += h * setup.Surface(kernels.rate[k], rate);
    }
    EXPECT_NEAR(quelled, thicknessView, 1e-12 * std::abs(thicknessView)) << label;
    auto distributional = 0.0;
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      const auto w = n == 0           ? 0.5 * (times[1] - times[0])
                     : n + 1 == nodes ? 0.5 * (times[n] - times[n - 1])
                                      : 0.5 * (times[n + 1] - times[n - 1]);
      distributional += w * setup.Surface(kernels.density[n], rising[n]);
      if (const auto lump = kernels.instantaneous.find(n);
          lump != kernels.instantaneous.end()) {
        distributional += setup.Surface(lump->second, rising[n]);
      }
    }
    EXPECT_NEAR(distributional, thicknessView, 1e-10 * std::abs(thicknessView)) << label;
  }
}

TEST(UnevenSteps, TheContinuousFormsConvergeAtTheSchemesOrder) {
  // One problem at every resolution, on a slowly relaxing mantle (1e22 Pa
  // s): a load ramping and a load rising and falling, observed at 4 kyr
  // (between a 3 kyr and a 0.5 kyr step) and at the end. As every step of
  // the uneven grid is halved, the continuous forms tend to the discrete
  // ones at the scheme's order — the gaps fall by about four for the
  // trapezoid and two for backward Euler, as on even grids: the
  // objective through the continuous load kernels (the discrete give it
  // exactly), the kernels in the relaxation times and the moduli (in their
  // volume norms), the continuous tangent-linear run along the relaxation
  // times (its fields at every node), and the continuous second-order
  // sweep's derivatives of those kernels along a direction u in the
  // thickness, the relaxation times and the moduli. Under backward Euler
  // the two tangent-linear runs coincide in the relaxation times, and
  // under either scheme in the moduli.
  auto setup = Scene::Make(8, 1e22);
  std::mt19937 rng(47);
  const auto ramp = setup.Smooth(rng, 100.0 * setup.Sigma());
  const auto bump = setup.Smooth(rng, 60.0 * setup.Sigma());
  const auto iceU = setup.Smooth(rng, 20.0 * setup.Metre());
  const auto gMid = setup.Smooth(rng, 1.0);
  const auto gEnd = setup.Smooth(rng, 0.5);
  const auto gDisp = setup.Smooth(rng, 0.4);
  const auto gTangential = setup.Smooth(rng, 0.6);
  const auto span = 8.0 * setup.Kyr();
  const auto mid = 4.0 * setup.Kyr();
  ViscoelasticOperator op(setup.rheology, setup.lmax);
  const auto& mesh = setup.material.Mesh();
  const auto ditauU = setup.DeltaItau(0.1);
  const auto materialU = PointDirection(op, setup.material, [](double r) {
    return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r);
  });

  enum Quantity { kLoads, kItau, kModuli, kTangentItau, kTangentModuli, kHessian, kCount };
  const auto names = std::array<const char*, kCount>{
      "load kernels", "itau kernels", "moduli kernels", "itau tangent", "moduli tangent",
      "Hessian"};
  auto gapsOf = [&](Scheme scheme, int split) {
    const auto times = UnevenTimes(setup.Kyr(), kSteps, split);
    const auto nodes = times.size();
    auto loads = std::vector<std::vector<double>>{};
    for (const auto t : times) {
      const auto s = t / span;
      auto load = ramp;
      for (auto i = std::size_t{0}; i < load.size(); ++i) {
        load[i] = s * ramp[i] + std::sin(std::numbers::pi * s) * bump[i];
      }
      loads.push_back(std::move(load));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[NodeOf(times, mid)].relativeSeaLevelDual = gMid;
    measurements[nodes - 1].relativeSeaLevelDual = gEnd;
    measurements[nodes - 1].displacementDual = gDisp;
    measurements[nodes - 1].tangentialPotentialDual = gTangential;
    Evolution evolution(op, setup.solver, setup.state,
                        {.scheme = scheme, .rtol = 1e-12, .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory::FromSnapshots(evolution.Run(times, Held(times, loads)));
    TangentPerturbation u, alongItau, alongMaterial;
    u.iceThickness = Scaled(times, span, iceU, [](double s) { return s; });
    u.itau = ditauU;
    u.material = materialU;
    alongItau.itau = ditauU;
    alongMaterial.material = materialU;
    // What each method gives: the objective through the load kernels, the
    // kernels in the relaxation times and the moduli and their derivatives
    // along u, and the tangent runs' fields at every node.
    struct Results {
      double loads = 0.0;
      std::vector<double> itau, kappa, mu, dItau, dKappa, dMu;
      std::vector<std::vector<double>> tangentItau, tangentModuli;
    };
    auto results = std::map<AdjointMethod, Results>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      AdjointSolver adjoint(evolution,
                            {.method = method, .rtol = 1e-12, .material_kernels = true});
      const auto kernels = adjoint.FirstOrderKernels(times, measurements, &trajectory);
      TangentLinearSolver tangent(evolution,
                                  {.method = method, .rtol = 1e-12, .record = true});
      const auto runU = tangent.Run(times, u, &trajectory);
      const auto derivatives = adjoint
                                   .SecondOrderKernels(times, measurements, runU, u.itau,
                                                       trajectory, {}, u.material)
                                   .second;
      auto fields = [&](const TangentPerturbation& p) {
        auto out = std::vector<std::vector<double>>{};
        for (const auto& snap : tangent.Run(times, p, &trajectory)) {
          out.push_back(snap.relativeSeaLevelChange);
          out.push_back(snap.displacement);
        }
        return out;
      };
      auto& r = results[method];
      // the load kernels paired with the run's own loads: the objective
      // itself, exactly, for the discrete sweep (the run is linear in them)
      r.loads = NodePairing(setup, kernels.directLoadKernels, loads);
      r.itau = kernels.itauKernel[0];
      r.kappa = kernels.kappaKernel;
      r.mu = kernels.muKernel;
      r.dItau = derivatives.itauKernel[0];
      r.dKappa = derivatives.kappaKernel;
      r.dMu = derivatives.muKernel;
      r.tangentItau = fields(alongItau);
      r.tangentModuli = fields(alongMaterial);
    }
    const auto& d = results[AdjointMethod::Discrete];
    const auto& c = results[AdjointMethod::Continuous];
    // relative distances in the kernels' own norms: the volume L2 norm of
    // a kernel (in the relaxation times nodal, in the moduli at the
    // points) and the surface L2 norm over the nodes of a run's fields
    auto difference = [](const std::vector<double>& x, const std::vector<double>& y) {
      auto out = x;
      for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] -= y[i];
      return out;
    };
    auto nodal = [&](const std::vector<double>& x, const std::vector<double>& y) {
      const auto e = difference(x, y);
      return std::sqrt(VolumePairing(mesh, e, e) / VolumePairing(mesh, y, y));
    };
    auto atPoints = [&](const std::vector<double>& xk, const std::vector<double>& yk,
                        const std::vector<double>& xm, const std::vector<double>& ym) {
      const auto ek = difference(xk, yk);
      const auto em = difference(xm, ym);
      return std::sqrt((VolumePairing(op.Points(), mesh, ek, ek) +
                        VolumePairing(op.Points(), mesh, em, em)) /
                       (VolumePairing(op.Points(), mesh, yk, yk) +
                        VolumePairing(op.Points(), mesh, ym, ym)));
    };
    auto onSurface = [&](const std::vector<std::vector<double>>& x,
                         const std::vector<std::vector<double>>& y) {
      auto num = 0.0, den = 0.0;
      for (auto n = std::size_t{0}; n < x.size(); ++n) {
        const auto e = difference(x[n], y[n]);
        num += setup.Surface(e, e);
        den += setup.Surface(y[n], y[n]);
      }
      return std::sqrt(num / den);
    };
    auto gaps = std::array<double, kCount>{};
    gaps[kLoads] = Gap(c.loads, d.loads);
    gaps[kItau] = nodal(c.itau, d.itau);
    gaps[kModuli] = atPoints(c.kappa, d.kappa, c.mu, d.mu);
    gaps[kTangentItau] = onSurface(c.tangentItau, d.tangentItau);
    gaps[kTangentModuli] = onSurface(c.tangentModuli, d.tangentModuli);
    gaps[kHessian] = std::hypot(nodal(c.dItau, d.dItau), atPoints(c.dKappa, d.dKappa,
                                                                  c.dMu, d.dMu));
    return gaps;
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = SchemeName(scheme);
    const bool trapezoid = scheme == Scheme::ExponentialTrapezoid;
    // the trapezoid from the coarse grid, backward Euler from its first
    // halving (its rates settle later)
    const auto splits = trapezoid ? std::vector<int>{1, 2, 4} : std::vector<int>{2, 4};
    auto gaps = std::vector<std::array<double, kCount>>{};
    for (const auto split : splits) gaps.push_back(gapsOf(scheme, split));
    const auto& coarse = gaps[gaps.size() - 2];
    const auto& fine = gaps.back();
    for (auto q = 0; q < kCount; ++q) {
      auto line = std::format("[uneven 1D continuous, {}] {}:", label, names[q]);
      for (const auto& g : gaps) line += std::format(" {:.2e}", g[q]);
      std::cout << line << '\n';
    }
    // the two tangents coincide in the moduli
    EXPECT_LT(fine[kTangentModuli], 1e-10) << label;
    EXPECT_LT(coarse[kTangentModuli], 1e-10) << label;
    if (trapezoid) {
      for (auto q : {kLoads, kItau, kModuli, kTangentItau, kHessian}) {
        EXPECT_LT(gaps.front()[q], 0.3) << label << ", " << names[q];
        EXPECT_GT(coarse[q] / fine[q], 3.3) << label << ", " << names[q];
      }
    } else {
      for (auto q : {kLoads, kItau, kModuli, kHessian}) {
        EXPECT_LT(coarse[q], 0.4) << label << ", " << names[q];
        EXPECT_GT(coarse[q] / fine[q], 1.7) << label << ", " << names[q];
      }
      // and in the relaxation times under backward Euler
      EXPECT_LT(fine[kTangentItau], 1e-10) << label;
    }
  }
}

TEST(UnevenSteps, TheDiscreteHessianIsExactAndSymmetric) {
  // Both schemes on the uneven grid, with directions u and w in the ice
  // thickness (non-zero at the first node), the relaxation times and the
  // moduli: the discrete second-order sweep carries the first-order
  // kernels; <H u, w> = <H w, u>, and both equal <g, F''[u, w]> from the
  // discrete second-order run, to the solvers' tolerance; and that run is
  // the derivative along u of the first-order run along w (central
  // differences with the thickness, the relaxation times and the moduli
  // moved by eps u: second order, Richardson landing on it).
  auto setup = Scene::Make(16, 1e21);
  std::mt19937 rng(53);
  const auto times = UnevenTimes(setup.Kyr(), kSteps);
  const auto nodes = times.size();
  auto thickness = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    thickness.push_back(setup.Smooth(rng, 100.0 * setup.Metre()));
  }
  const auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = SchemeName(scheme);
    ViscoelasticOperator op(setup.rheology, setup.lmax);
    const Evolution::Options options{.scheme = scheme, .rtol = 1e-12, .record_stride = 1};
    Evolution evolution(op, setup.solver, setup.state, options);
    const auto trajectory = ForwardTrajectory::FromSnapshots(
        evolution.Run(times, Held(times, FrozenLoads(setup, thickness))));
    std::mt19937 local(59);
    TangentPerturbation u, w;
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      u.iceThickness.push_back(setup.Smooth(local, 20.0 * setup.Metre()));
      w.iceThickness.push_back(setup.Smooth(local, 20.0 * setup.Metre()));
    }
    u.itau = setup.DeltaItau(0.1);
    w.itau = setup.DeltaItau(0.1, [](double r) { return 0.8 - 0.5 * r; });
    u.material = PointDirection(op, setup.material, [](double r) {
      return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r);
    });
    w.material = PointDirection(op, setup.material, [](double r) { return 1.0 - 0.5 * r; });
    TangentLinearSolver tangent(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .record = true});
    const auto runU = tangent.Run(times, u, &trajectory);
    const auto runW = tangent.Run(times, w, &trajectory);
    AdjointSolver adjoint(evolution, {.method = AdjointMethod::Discrete,
                                      .rtol = 1e-12,
                                      .material_kernels = true});
    const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
    const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
        times, measurements, runU, u.itau, trajectory, {}, u.material);
    const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
        times, measurements, runW, w.itau, trajectory, {}, w.material);
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      EXPECT_LE(MaxDiff(kernelsU.thickness[n], first.thickness[n]),
                1e-12 * MaxAbs(first.thickness[n]))
          << label << ", node " << n;
    }
    EXPECT_LE(MaxDiff(kernelsU.itauKernel[0], first.itauKernel[0]),
              1e-12 * MaxAbs(first.itauKernel[0]))
        << label;
    EXPECT_LE(MaxDiff(kernelsU.muKernel, first.muKernel), 1e-12 * MaxAbs(first.muKernel))
        << label;

    const auto uw = Pairing(setup, op, alongU, w);
    const auto wu = Pairing(setup, op, alongW, u);
    SecondOrderDirection direction;
    direction.itauU = u.itau;
    direction.itauV = w.itau;
    direction.materialU = u.material;
    direction.materialV = w.material;
    const auto forward = Objective(
        setup, tangent.RunSecondOrder(times, runU, runW, direction, &trajectory),
        measurements);
    std::cout << std::format(
        "[uneven 1D Hessian, {}] symmetry gap {:.2e}, against the second-order run "
        "{:.2e}\n",
        label, Gap(wu, uw), Gap(uw, forward));
    ASSERT_NE(forward, 0.0) << label;
    EXPECT_NEAR(uw, wu, 1e-9 * std::abs(uw)) << label;
    EXPECT_NEAR(uw, forward, 1e-9 * std::abs(forward)) << label;

    // The second-order run as the derivative of the first-order one.
    const auto fd = Difference(
        [&](double eps) {
          const auto moved = MovedOperator(setup, op, eps, u.material, u.itau);
          Evolution trial(moved, setup.solver, setup.state, options);
          const auto movedTrajectory = ForwardTrajectory::FromSnapshots(trial.Run(
              times, Held(times, FrozenLoads(setup, Plus(thickness, u.iceThickness, eps)))));
          TangentLinearSolver along(trial,
                                    {.method = AdjointMethod::Discrete, .rtol = 1e-12});
          return Objective(setup, along.Run(times, w, &movedTrajectory), measurements);
        },
        0.1, forward);
    std::cout << std::format(
        "[uneven 1D Hessian, {}] second-order run: ratio {:.3f}, Richardson gap {:.2e}\n",
        label, fd.ratio, Gap(fd.richardson, forward));
    EXPECT_GT(fd.ratio, 3.0) << label;
    EXPECT_LT(fd.ratio, 5.0) << label;
    EXPECT_NEAR(fd.richardson, forward, 1e-6 * std::abs(forward)) << label;
  }
}

TEST(UnevenSteps, TheDecayTimeOverAnUnevenWindow) {
  // A datum over a window of uneven nodes: the decay time of a site's
  // subsidence under a load held from the first node. The fit's weights
  // are the decay time's derivative with respect to the series (central
  // differences of the fit), and the decay-time measurement through the
  // discrete sweep gives the decay time's derivative along the relaxation
  // times (central differences of the forward runs' fitted decay time,
  // Richardson landing on the kernel's pairing).
  auto setup = Scene::Make(16, 1e21);
  const auto times = UnevenTimes(setup.Kyr(), kSteps);
  const auto nodes = times.size();
  const auto& grid = setup.solver.Grid();
  const auto a = setup.solver.Radius();
  const auto site = GeographicPoint{.latitude = 60.0, .longitude = 275.0};
  const auto load = grid.FromFunction([&](double theta, double phi) {
    return 3e4 * setup.Sigma() *
           Cap(theta, phi, std::numbers::pi / 2.0 - site.latitude * std::numbers::pi / 180.0,
               site.longitude * std::numbers::pi / 180.0, 0.15);
  });
  const auto dual = CapMeanDual(grid, a, site, 0.1);
  auto window = std::vector<std::size_t>{};
  auto windowTimes = std::vector<double>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    window.push_back(n);
    windowTimes.push_back(times[n] / setup.Kyr());
  }
  const auto ditau = setup.DeltaItau(0.1);
  ViscoelasticOperator op(setup.rheology, setup.lmax);
  auto seriesOf = [&](const std::vector<Evolution::Snapshot>& run) {
    auto out = std::vector<double>{};
    for (const auto& snap : run) out.push_back(PairedValue(grid, a, dual, snap.displacement));
    return out;
  };
  Evolution evolution(op, setup.solver, setup.state, {.rtol = 1e-12, .record_stride = 1});
  const auto run = evolution.Run(times, [&](double) { return load; });
  const auto series = seriesOf(run);
  auto range = 0.0;
  for (const auto v : series) range = std::max(range, std::abs(v - series.front()));
  const auto options =
      DecayTimeOptions{.time_uncertainty = 0.0, .value_uncertainty = 0.01 * range};
  auto fit = DecayTimeFit{};
  const auto measurements = DecayTimeMeasurements(
      window, windowTimes, series, FieldChannel::Displacement, dual, options, &fit);
  ASSERT_GT(fit.decay_time, 0.0);

  // the weights: the fit's derivative in each sample
  for (auto j = std::size_t{0}; j < nodes; ++j) {
    // (wide enough that the fit's own stopping tolerance stays below the
    // quotient's truncation error)
    const auto step = 1e-4 * range;
    auto plus = series, minus = series;
    plus[j] += step;
    minus[j] -= step;
    const auto difference = (FitDecayTime(windowTimes, plus, options).decay_time -
                             FitDecayTime(windowTimes, minus, options).decay_time) /
                            (2.0 * step);
    EXPECT_NEAR(fit.weights[j], difference,
                1e-5 * std::max(std::abs(difference), MaxAbs(fit.weights)))
        << "sample " << j;
  }

  // the datum through the sweep, against the forward runs' decay times
  const auto trajectory = ForwardTrajectory::FromSnapshots(run);
  const auto kernels =
      AdjointSolver(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
          .FirstOrderKernels(times, measurements, &trajectory);
  const auto predicted =
      VolumePairing(setup.material.Mesh(), kernels.itauKernel[0], ditau[0]);
  const auto fd = Difference(
      [&](double eps) {
        const auto moved = MovedOperator(setup, op, eps, {}, ditau);
        Evolution trial(moved, setup.solver, setup.state, {.rtol = 1e-12});
        return FitDecayTime(windowTimes,
                            seriesOf(trial.Run(times, [&](double) { return load; })),
                            options)
            .decay_time;
      },
      0.02, predicted);
  std::cout << std::format(
      "[uneven decay time] {:.4f} kyr; along itau: ratio {:.3f}, Richardson gap {:.2e}\n",
      fit.decay_time, fd.ratio, Gap(fd.richardson, predicted));
  EXPECT_GT(fd.ratio, 3.0);
  EXPECT_LT(fd.ratio, 5.0);
  EXPECT_NEAR(fd.richardson, predicted, 1e-5 * std::abs(predicted));
}

// -- 3D, under lateral viscosity -----------------------------------------

namespace {

/// The 3D gates' steps (kyr): the 1D pattern's first four.
const auto kSteps3D = std::vector<double>{1.0, 3.0, 0.5, 2.0};

/// The lateral log-viscosity of the 3D gates.
double LateralDelta(double, double theta, double phi) {
  return 0.6 * std::sin(theta) * std::cos(theta) * std::cos(phi);
}

std::vector<double> OnCoupling(const GridCoupling& coupling,
                               const std::function<double(double, double)>& f) {
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto values = coupling.GridOf(rn).FromFunction(f);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < values.size(); ++x) out[at + x] = values[x];
  }
  return out;
}

/// A lateral direction in kappa and mu over the coupling's grid values: a
/// radial profile times an angular pattern times the node's moduli
/// (kappa at half the shear's share), at a tenth of the moduli.
MaterialDirection3D LateralDirection(const Viscoelastic3D& op,
                                     const std::function<double(double)>& radial,
                                     const std::function<double(double, double)>& angular) {
  const auto& coupling = op.Coupling();
  const auto np = static_cast<std::size_t>(coupling.Points().count);
  const auto muU = op.Reference().UnrelaxedShearAtPoints();
  const auto kappa0 = op.Reference().BulkModulusAtPoints();
  MaterialDirection3D d;
  d.kappa.assign(coupling.GridValueCount(), 0.0);
  d.mu.assign(coupling.GridValueCount(), 0.0);
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto point = static_cast<std::size_t>(coupling.ElementOfNode(rn)) * np +
                       coupling.NodeInElementOf(rn);
    const auto value = 0.1 * radial(coupling.RadiusOfNode(rn));
    const auto pattern = coupling.GridOf(rn).FromFunction(angular);
    const auto at = coupling.FieldOffset(rn);
    for (auto x = std::size_t{0}; x < pattern.size(); ++x) {
      d.kappa[at + x] = 0.5 * value * pattern[x] * kappa0[point];
      d.mu[at + x] = value * pattern[x] * muU[point];
    }
  }
  return d;
}

/// The 3D operator moved by eps along a direction: its log-viscosity by
/// eps p and its moduli by eps times the relative pattern of
/// LateralDirection (confined to the coupling's elements, so that the
/// moved operator carries the same nodes).
std::unique_ptr<Viscoelastic3D> Moved3D(const Scene& setup, const Viscoelastic3D& base,
                                        double eps,
                                        const std::function<double(double, double)>& p,
                                        const std::function<double(double)>& radial,
                                        const std::function<double(double, double)>& angular) {
  const auto& coupling = base.Coupling();
  const auto& mesh = setup.material.Mesh();
  auto inner = mesh.Right(mesh.ElementCount() - 1), outer = 0.0;
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto e = coupling.ElementOfNode(rn);
    inner = std::min(inner, mesh.Left(e));
    outer = std::max(outer, mesh.Right(e));
  }
  auto options = Viscoelastic3D::Options{};
  if (radial) {
    auto relative = [=](double r, double theta, double phi) {
      return r > inner && r < outer ? 0.1 * eps * radial(r) * angular(theta, phi) : 0.0;
    };
    options.moduli = {.shear = relative,
                      .bulk = [=](double r, double theta, double phi) {
                        return 0.5 * relative(r, theta, phi);
                      }};
  }
  auto moved = std::make_unique<Viscoelastic3D>(
      setup.rheology, setup.lmax,
      [=](double r, double theta, double phi) {
        return LateralDelta(r, theta, phi) + (p ? eps * p(theta, phi) : 0.0);
      },
      options);
  EXPECT_EQ(moved->Coupling().NodeCount(), coupling.NodeCount());
  return moved;
}

/// A 3D kernel object (kernels or their derivatives) paired with a
/// direction in the thickness, the relaxation times and the moduli.
template <class Ice, class K>
double Pairing3D(const Scene& setup, const GridCoupling& coupling, const Ice& ice,
                 const K& kernels, const TangentPerturbation3D& d) {
  auto total = NodePairing(setup, ice.thickness, d.iceThickness);
  for (auto k = std::size_t{0}; k < d.itau.size(); ++k) {
    total += VolumePairing(coupling, kernels.itauKernel3D[k], d.itau[k]);
  }
  if (!d.material.kappa.empty()) {
    total += VolumePairing(coupling, kernels.kappaKernel3D, d.material.kappa);
  }
  if (!d.material.mu.empty()) {
    total += VolumePairing(coupling, kernels.muKernel3D, d.material.mu);
  }
  return total;
}

const auto kPattern = [](double theta, double phi) {
  return Cap(theta, phi, 1.9, 4.4, 0.7) - Cap(theta, phi, 0.9, 1.1, 0.6);
};
const auto kAngular = [](double theta, double phi) {
  return 1.0 + 0.7 * std::cos(theta) + 0.5 * std::sin(theta) * std::sin(phi);
};
const auto kRadial = [](double r) { return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r); };

}  // namespace

TEST(UnevenSteps, TheTangentIsTheDerivativeOfTheRun3D) {
  // Under lateral viscosity on the uneven grid, both schemes: along the
  // loads and the ice thickness (non-zero at the first node) the coupled
  // run is the difference of two coupled runs, both methods; along a
  // lateral log-viscosity direction and a lateral direction in the moduli
  // the discrete run is the derivative of the coupled run (central
  // differences at second order, Richardson landing on it).
  auto setup = Scene::Make(8, 1e21);
  std::mt19937 rng(67);
  const auto times = UnevenTimes(setup.Kyr(), kSteps3D);
  const auto nodes = times.size();
  auto loads = std::vector<std::vector<double>>{};
  auto deltas = std::vector<std::vector<double>>{};
  auto thickness = std::vector<std::vector<double>>{};
  auto deltaIce = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    deltas.push_back(setup.Smooth(rng, 10.0 * setup.Sigma()));
    thickness.push_back(setup.Smooth(rng, 100.0 * setup.Metre()));
    deltaIce.push_back(setup.Smooth(rng, 10.0 * setup.Metre()));
  }
  auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);
  measurements[nodes - 1].toroidalPotentialDual = setup.Smooth(rng, 0.5);

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = SchemeName(scheme);
    Viscoelastic3D op(setup.rheology, setup.lmax, LateralDelta);
    const Evolution3D::Options options{
        .scheme = scheme, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1};
    Evolution3D evolution(op, setup.solver, setup.state, options);
    const auto base = evolution.Run(times, Held(times, loads));
    const auto plus = evolution.Run(times, Held(times, Plus(loads, deltas)));
    const auto iceBase = evolution.Run(times, Held(times, FrozenLoads(setup, thickness)));
    const auto icePlus =
        evolution.Run(times, Held(times, FrozenLoads(setup, Plus(thickness, deltaIce))));
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(base);

    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      TangentLinearSolver3D tangent(
          evolution, {.method = method, .rtol = 1e-12, .coupled_rtol = 1e-12});
      TangentPerturbation3D alongLoads, alongIce;
      alongLoads.directLoad = deltas;
      alongIce.iceThickness = deltaIce;
      const auto runLoads = tangent.Run(times, alongLoads);
      const auto runIce = tangent.Run(times, alongIce);
      for (auto n = std::size_t{0}; n < nodes; ++n) {
        for (auto field : {&Evolution3D::Snapshot::relativeSeaLevelChange,
                           &Evolution3D::Snapshot::displacement,
                           &Evolution3D::Snapshot::toroidalPotential}) {
          for (const auto& [run, p, b] : {std::tuple{&runLoads, &plus, &base},
                                          std::tuple{&runIce, &icePlus, &iceBase}}) {
            auto difference = (*p)[n].*field;
            for (auto i = std::size_t{0}; i < difference.size(); ++i) {
              difference[i] -= ((*b)[n].*field)[i];
            }
            EXPECT_LE(MaxDiff((*run)[n].*field, difference), 1e-8 * MaxAbs(difference))
                << label << ", node " << n << ", method " << static_cast<int>(method);
          }
        }
      }
    }

    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
    auto runAt = [&](const Viscoelastic3D& moved) {
      Evolution3D trial(moved, setup.solver, setup.state, options);
      return Objective(setup, trial.Run(times, Held(times, loads)), measurements);
    };
    // the log viscosity
    TangentPerturbation3D alongViscosity;
    alongViscosity.logViscosity = OnCoupling(op.Coupling(), kPattern);
    const auto predictedViscosity =
        Objective(setup, tangent.Run(times, alongViscosity, &trajectory), measurements);
    const auto viscosity = Difference(
        [&](double eps) { return runAt(*Moved3D(setup, op, eps, kPattern, {}, {})); }, 0.02,
        predictedViscosity);
    std::cout << std::format(
        "[uneven 3D tangent, {}] log viscosity: ratio {:.3f}, Richardson gap {:.2e}\n",
        label, viscosity.ratio, Gap(viscosity.richardson, predictedViscosity));
    EXPECT_GT(viscosity.ratio, 3.0) << label;
    EXPECT_LT(viscosity.ratio, 5.0) << label;
    EXPECT_NEAR(viscosity.richardson, predictedViscosity, 1e-6 * std::abs(predictedViscosity))
        << label;
    // the moduli, laterally
    TangentPerturbation3D alongMaterial;
    alongMaterial.material = LateralDirection(op, kRadial, kAngular);
    const auto predictedMaterial =
        Objective(setup, tangent.Run(times, alongMaterial, &trajectory), measurements);
    const auto material = Difference(
        [&](double eps) { return runAt(*Moved3D(setup, op, eps, {}, kRadial, kAngular)); },
        0.1, predictedMaterial);
    std::cout << std::format(
        "[uneven 3D tangent, {}] moduli: ratio {:.3f}, Richardson gap {:.2e}\n", label,
        material.ratio, Gap(material.richardson, predictedMaterial));
    EXPECT_GT(material.ratio, 3.0) << label;
    EXPECT_LT(material.ratio, 5.0) << label;
    EXPECT_NEAR(material.richardson, predictedMaterial, 1e-6 * std::abs(predictedMaterial))
        << label;
  }
}

TEST(UnevenSteps, TheDiscreteSweepIsTheTangentsTranspose3D) {
  // Under lateral viscosity on the uneven grid, both schemes, a full
  // measurement set with the toroidal potential: the discrete 3D sweep's
  // load kernels (node 0 included), thickness view, log-viscosity kernel
  // and lateral moduli kernels each pair with their direction to the
  // discrete 3D tangent's objective change; a stride-2 recording, its
  // gaps rebuilt by forward steps of their own sizes, serves the same
  // kernels.
  auto setup = Scene::Make(8, 1e21);
  std::mt19937 rng(71);
  const auto times = UnevenTimes(setup.Kyr(), kSteps3D);
  const auto nodes = times.size();
  auto loads = std::vector<std::vector<double>>{};
  auto deltas = std::vector<std::vector<double>>{};
  auto deltaIce = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    loads.push_back(setup.Smooth(rng, 100.0 * setup.Sigma()));
    deltas.push_back(setup.Smooth(rng, 10.0 * setup.Sigma()));
    deltaIce.push_back(setup.Smooth(rng, 10.0 * setup.Metre()));
  }
  auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);
  measurements[nodes - 1].toroidalPotentialDual = setup.Smooth(rng, 0.5);

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = SchemeName(scheme);
    Viscoelastic3D op(setup.rheology, setup.lmax, LateralDelta);
    const auto& coupling = op.Coupling();
    Evolution3D evolution(op, setup.solver, setup.state,
                          {.scheme = scheme,
                           .rtol = 1e-12,
                           .coupled_rtol = 1e-12,
                           .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, Held(times, loads)));
    const AdjointSolver3D::Options adjointOptions{.method = AdjointMethod::Discrete,
                                                  .rtol = 1e-12,
                                                  .coupled_rtol = 1e-12,
                                                  .material_kernels = true};
    const auto kernels =
        AdjointSolver3D(evolution, adjointOptions).FirstOrderKernels(times, measurements, &trajectory);
    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
    auto along = [&](const TangentPerturbation3D& p) {
      return Objective(setup, tangent.Run(times, p, &trajectory), measurements);
    };
    TangentPerturbation3D pLoads, pIce, pViscosity, pMaterial;
    pLoads.directLoad = deltas;
    pIce.iceThickness = deltaIce;
    pViscosity.logViscosity = OnCoupling(coupling, kPattern);
    pMaterial.material = LateralDirection(op, kRadial, kAngular);
    const auto viscosity =
        VolumePairing(coupling, kernels.logViscosityKernel3D[0], pViscosity.logViscosity);
    for (const auto& [name, paired, p] :
         {std::tuple{"loads", NodePairing(setup, kernels.ice.directLoadKernels, deltas),
                     &pLoads},
          std::tuple{"ice", NodePairing(setup, kernels.ice.thickness, deltaIce), &pIce},
          std::tuple{"log viscosity", viscosity, &pViscosity},
          std::tuple{"moduli",
                     VolumePairing(coupling, kernels.kappaKernel3D, pMaterial.material.kappa) +
                         VolumePairing(coupling, kernels.muKernel3D, pMaterial.material.mu),
                     &pMaterial}}) {
      const auto want = along(*p);
      ASSERT_NE(want, 0.0) << label << ", " << name;
      std::cout << std::format("[uneven 3D transpose, {}] {}: gap {:.2e}\n", label, name,
                               Gap(paired, want));
      EXPECT_NEAR(paired, want, 1e-8 * std::abs(want)) << label << ", " << name;
    }

    Evolution3D sparse(op, setup.solver, setup.state,
                       {.scheme = scheme,
                        .rtol = 1e-12,
                        .coupled_rtol = 1e-12,
                        .record_stride = 2});
    const auto sparseTrajectory =
        ForwardTrajectory3D::FromSnapshots(sparse.Run(times, Held(times, loads)));
    const auto again = AdjointSolver3D(evolution, adjointOptions)
                           .FirstOrderKernels(times, measurements, &sparseTrajectory);
    const auto viaSparse =
        VolumePairing(coupling, again.logViscosityKernel3D[0], pViscosity.logViscosity);
    std::cout << std::format("[uneven 3D transpose, {}] stride 2: log viscosity gap {:.2e}\n",
                             label, Gap(viaSparse, viscosity));
    EXPECT_NEAR(viaSparse, viscosity, 1e-9 * std::abs(viscosity)) << label;
    EXPECT_NEAR(Objective(setup, tangent.Run(times, pViscosity, &sparseTrajectory),
                          measurements),
                along(pViscosity), 1e-9 * std::abs(viscosity))
        << label;
  }
}

TEST(UnevenSteps, TheContinuousFormsConvergeAtTheSchemesOrder3D) {
  // The 1D convergence gate's problem under lateral viscosity (1e22 Pa s
  // mantle): as every step of the uneven grid is halved, the continuous
  // 3D sweep's objective through the load kernels, its log-viscosity and
  // lateral moduli kernels (in their volume norms), the continuous 3D
  // tangent-linear run along the log viscosity (its fields at every
  // node) and the continuous 3D second-order sweep's derivatives of those
  // kernels along a direction in the thickness, the log viscosity and the
  // moduli tend to the discrete ones at the scheme's order. From the
  // coarse grid's first halving: rates of 3.2 to 3.8 for the trapezoid
  // (the load kernels' 2.8 still settling; the next halving gives 3.7 to
  // 3.9) and of 1.7 to 1.9 for backward Euler (its load kernels' 1.3
  // likewise, then 1.7). Under backward Euler the two tangent-linear runs
  // coincide in the log viscosity, and under either scheme in the moduli.
  auto setup = Scene::Make(8, 1e22);
  std::mt19937 rng(73);
  const auto ramp = setup.Smooth(rng, 100.0 * setup.Sigma());
  const auto bump = setup.Smooth(rng, 60.0 * setup.Sigma());
  const auto iceU = setup.Smooth(rng, 20.0 * setup.Metre());
  const auto gMid = setup.Smooth(rng, 1.0);
  const auto gEnd = setup.Smooth(rng, 0.5);
  const auto gDisp = setup.Smooth(rng, 0.4);
  const auto gTangential = setup.Smooth(rng, 0.6);
  const auto span = 6.5 * setup.Kyr();
  const auto mid = 4.0 * setup.Kyr();
  Viscoelastic3D op(setup.rheology, setup.lmax, LateralDelta);
  const auto& coupling = op.Coupling();
  const auto pattern = OnCoupling(coupling, kPattern);
  const auto materialU = LateralDirection(op, kRadial, kAngular);

  enum Quantity { kLoads, kViscosity, kModuli, kTangent, kTangentModuli, kHessian, kCount };
  const auto names = std::array<const char*, kCount>{
      "load kernels", "log-viscosity kernels", "moduli kernels", "log-viscosity tangent",
      "moduli tangent", "Hessian"};
  auto gapsOf = [&](Scheme scheme, int split) {
    const auto times = UnevenTimes(setup.Kyr(), kSteps3D, split);
    const auto nodes = times.size();
    auto loads = std::vector<std::vector<double>>{};
    for (const auto t : times) {
      const auto s = t / span;
      auto load = ramp;
      for (auto i = std::size_t{0}; i < load.size(); ++i) {
        load[i] = s * ramp[i] + std::sin(std::numbers::pi * s) * bump[i];
      }
      loads.push_back(std::move(load));
    }
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[NodeOf(times, mid)].relativeSeaLevelDual = gMid;
    measurements[nodes - 1].relativeSeaLevelDual = gEnd;
    measurements[nodes - 1].displacementDual = gDisp;
    measurements[nodes - 1].tangentialPotentialDual = gTangential;
    Evolution3D evolution(op, setup.solver, setup.state,
                          {.scheme = scheme,
                           .rtol = 1e-12,
                           .coupled_rtol = 1e-12,
                           .record_stride = 1});
    const auto trajectory =
        ForwardTrajectory3D::FromSnapshots(evolution.Run(times, Held(times, loads)));
    TangentPerturbation3D u, alongViscosity, alongMaterial;
    u.iceThickness = Scaled(times, span, iceU, [](double s) { return s; });
    u.itau = TangentLinearSolver3D::ItauOfLogViscosity(op, pattern);
    u.material = materialU;
    alongViscosity.logViscosity = pattern;
    alongMaterial.material = materialU;
    struct Results {
      double loads = 0.0;
      std::vector<double> viscosity, kappa, mu, dViscosity, dKappa, dMu;
      std::vector<std::vector<double>> tangent, tangentModuli;
    };
    auto results = std::map<AdjointMethod, Results>{};
    for (auto method : {AdjointMethod::Discrete, AdjointMethod::Continuous}) {
      AdjointSolver3D adjoint(evolution, {.method = method,
                                          .rtol = 1e-12,
                                          .coupled_rtol = 1e-12,
                                          .material_kernels = true});
      const auto kernels = adjoint.FirstOrderKernels(times, measurements, &trajectory);
      TangentLinearSolver3D tangent(evolution, {.method = method,
                                                .rtol = 1e-12,
                                                .coupled_rtol = 1e-12,
                                                .record = true});
      const auto runU = tangent.Run(times, u, &trajectory);
      const auto derivatives = adjoint
                                   .SecondOrderKernels(times, measurements, runU, u.itau,
                                                       trajectory, {}, u.material)
                                   .second;
      auto fields = [&](const TangentPerturbation3D& p) {
        auto out = std::vector<std::vector<double>>{};
        for (const auto& snap : tangent.Run(times, p, &trajectory)) {
          out.push_back(snap.relativeSeaLevelChange);
          out.push_back(snap.displacement);
        }
        return out;
      };
      auto& r = results[method];
      r.loads = NodePairing(setup, kernels.ice.directLoadKernels, loads);
      r.viscosity = kernels.logViscosityKernel3D[0];
      r.kappa = kernels.kappaKernel3D;
      r.mu = kernels.muKernel3D;
      r.dViscosity = derivatives.logViscosityKernel3D[0];
      r.dKappa = derivatives.kappaKernel3D;
      r.dMu = derivatives.muKernel3D;
      r.tangent = fields(alongViscosity);
      r.tangentModuli = fields(alongMaterial);
    }
    const auto& d = results[AdjointMethod::Discrete];
    const auto& c = results[AdjointMethod::Continuous];
    auto difference = [](const std::vector<double>& x, const std::vector<double>& y) {
      auto out = x;
      for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] -= y[i];
      return out;
    };
    auto inVolume = [&](const std::vector<double>& x, const std::vector<double>& y) {
      const auto e = difference(x, y);
      return std::sqrt(VolumePairing(coupling, e, e) / VolumePairing(coupling, y, y));
    };
    auto moduli = [&](const std::vector<double>& xk, const std::vector<double>& yk,
                      const std::vector<double>& xm, const std::vector<double>& ym) {
      const auto ek = difference(xk, yk);
      const auto em = difference(xm, ym);
      return std::sqrt((VolumePairing(coupling, ek, ek) + VolumePairing(coupling, em, em)) /
                       (VolumePairing(coupling, yk, yk) + VolumePairing(coupling, ym, ym)));
    };
    auto onSurface = [&](const std::vector<std::vector<double>>& x,
                         const std::vector<std::vector<double>>& y) {
      auto num = 0.0, den = 0.0;
      for (auto n = std::size_t{0}; n < x.size(); ++n) {
        const auto e = difference(x[n], y[n]);
        num += setup.Surface(e, e);
        den += setup.Surface(y[n], y[n]);
      }
      return std::sqrt(num / den);
    };
    auto gaps = std::array<double, kCount>{};
    gaps[kLoads] = Gap(c.loads, d.loads);
    gaps[kViscosity] = inVolume(c.viscosity, d.viscosity);
    gaps[kModuli] = moduli(c.kappa, d.kappa, c.mu, d.mu);
    gaps[kTangent] = onSurface(c.tangent, d.tangent);
    gaps[kTangentModuli] = onSurface(c.tangentModuli, d.tangentModuli);
    gaps[kHessian] = std::hypot(inVolume(c.dViscosity, d.dViscosity),
                                moduli(c.dKappa, d.dKappa, c.dMu, d.dMu));
    return gaps;
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = SchemeName(scheme);
    auto gaps = std::vector<std::array<double, kCount>>{};
    for (const auto split : {1, 2}) gaps.push_back(gapsOf(scheme, split));
    const auto& coarse = gaps[0];
    const auto& fine = gaps[1];
    for (auto q = 0; q < kCount; ++q) {
      auto line = std::format("[uneven 3D continuous, {}] {}:", label, names[q]);
      for (const auto& g : gaps) line += std::format(" {:.2e}", g[q]);
      std::cout << line << '\n';
    }
    EXPECT_LT(coarse[kTangentModuli], 1e-10) << label;
    EXPECT_LT(fine[kTangentModuli], 1e-10) << label;
    if (scheme == Scheme::ExponentialTrapezoid) {
      for (auto q : {kViscosity, kModuli, kTangent, kHessian}) {
        EXPECT_LT(coarse[q], 0.2) << label << ", " << names[q];
        EXPECT_GT(coarse[q] / fine[q], 3.0) << label << ", " << names[q];
      }
      EXPECT_LT(coarse[kLoads], 0.2) << label;
      EXPECT_GT(coarse[kLoads] / fine[kLoads], 2.6) << label;
    } else {
      for (auto q : {kViscosity, kModuli, kHessian}) {
        EXPECT_LT(coarse[q], 0.3) << label << ", " << names[q];
        EXPECT_GT(coarse[q] / fine[q], 1.6) << label << ", " << names[q];
      }
      EXPECT_LT(coarse[kLoads], 0.2) << label;
      EXPECT_GT(coarse[kLoads] / fine[kLoads], 1.2) << label;
      EXPECT_LT(fine[kTangent], 1e-10) << label;
    }
  }
}

TEST(UnevenSteps, TheDiscreteHessianIsExactAndSymmetric3D) {
  // Under lateral viscosity on the uneven grid, both schemes, directions u
  // and w in the ice thickness (non-zero at the first node), the log
  // viscosity and the lateral moduli: the discrete 3D second-order sweep
  // carries the first-order kernels; <H u, w> = <H w, u>, and both equal
  // <g, F''[u, w]> from the discrete 3D second-order run (whose log
  // parameterisation adds the first-order kernel paired with itau_uv), to
  // the solvers' tolerance; and that run is the derivative along u of the
  // first-order run along w (central differences with the thickness, the
  // log viscosity and the moduli moved by eps u, second order, Richardson
  // landing on it).
  auto setup = Scene::Make(8, 1e21);
  std::mt19937 rng(79);
  const auto times = UnevenTimes(setup.Kyr(), kSteps3D);
  const auto nodes = times.size();
  auto thickness = std::vector<std::vector<double>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    thickness.push_back(setup.Smooth(rng, 100.0 * setup.Metre()));
  }
  auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);
  measurements[nodes - 1].toroidalPotentialDual = setup.Smooth(rng, 0.5);
  const auto patternW = [](double theta, double phi) {
    return 0.5 * Cap(theta, phi, 2.4, 2.0, 0.8) + 0.3 * std::cos(theta);
  };
  const auto angularW = [](double theta, double phi) {
    return Cap(theta, phi, 1.9, 4.4, 0.7) + 0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto radialW = [](double r) { return 1.0 - 0.5 * r; };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    const auto label = SchemeName(scheme);
    Viscoelastic3D op(setup.rheology, setup.lmax, LateralDelta);
    const auto& coupling = op.Coupling();
    const Evolution3D::Options options{
        .scheme = scheme, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1};
    Evolution3D evolution(op, setup.solver, setup.state, options);
    const auto trajectory = ForwardTrajectory3D::FromSnapshots(
        evolution.Run(times, Held(times, FrozenLoads(setup, thickness))));
    std::mt19937 local(83);
    TangentPerturbation3D u, w;
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      u.iceThickness.push_back(setup.Smooth(local, 20.0 * setup.Metre()));
      w.iceThickness.push_back(setup.Smooth(local, 20.0 * setup.Metre()));
    }
    u.itau = TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, kPattern));
    w.itau = TangentLinearSolver3D::ItauOfLogViscosity(op, OnCoupling(coupling, patternW));
    u.material = LateralDirection(op, kRadial, kAngular);
    w.material = LateralDirection(op, radialW, angularW);
    TangentLinearSolver3D tangent(evolution, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12,
                                              .record = true});
    const auto runU = tangent.Run(times, u, &trajectory);
    const auto runW = tangent.Run(times, w, &trajectory);
    AdjointSolver3D adjoint(evolution, {.method = AdjointMethod::Discrete,
                                        .rtol = 1e-12,
                                        .coupled_rtol = 1e-12,
                                        .material_kernels = true});
    const auto first = adjoint.FirstOrderKernels(times, measurements, &trajectory);
    const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
        times, measurements, runU, u.itau, trajectory, {}, u.material);
    const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
        times, measurements, runW, w.itau, trajectory, {}, w.material);
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      EXPECT_LE(MaxDiff(kernelsU.ice.thickness[n], first.ice.thickness[n]),
                1e-10 * MaxAbs(first.ice.thickness[n]))
          << label << ", node " << n;
    }
    EXPECT_LE(MaxDiff(kernelsU.itauKernel3D[0], first.itauKernel3D[0]),
              1e-10 * MaxAbs(first.itauKernel3D[0]))
        << label;
    EXPECT_LE(MaxDiff(kernelsU.muKernel3D, first.muKernel3D),
              1e-10 * MaxAbs(first.muKernel3D))
        << label;

    const auto uw = Pairing3D(setup, coupling, alongU.ice, alongU, w);
    const auto wu = Pairing3D(setup, coupling, alongW.ice, alongW, u);
    SecondOrderDirection3D direction;
    direction.itauU = u.itau;
    direction.itauV = w.itau;
    direction.itauUV = TangentLinearSolver3D::SecondVariationOfLogViscosity(
        op, OnCoupling(coupling, kPattern), OnCoupling(coupling, patternW));
    direction.materialU = u.material;
    direction.materialV = w.material;
    const auto forward = Objective(
        setup, tangent.RunSecondOrder(times, runU, runW, direction, &trajectory),
        measurements);
    auto uvTerm = 0.0;
    for (auto k = std::size_t{0}; k < direction.itauUV.size(); ++k) {
      uvTerm += VolumePairing(coupling, first.itauKernel3D[k], direction.itauUV[k]);
    }
    std::cout << std::format(
        "[uneven 3D Hessian, {}] symmetry gap {:.2e}, against the second-order run "
        "{:.2e}\n",
        label, Gap(wu, uw), Gap(uw + uvTerm, forward));
    ASSERT_NE(forward, 0.0) << label;
    EXPECT_NEAR(uw, wu, 1e-8 * std::abs(uw)) << label;
    EXPECT_NEAR(uw + uvTerm, forward, 1e-8 * std::abs(forward)) << label;

    // The second-order run as the derivative of the first-order one, w's
    // relaxation-time direction taken at the moved operator (the log
    // parameterisation's).
    const auto fd = Difference(
        [&](double eps) {
          const auto moved = Moved3D(setup, op, eps, kPattern, kRadial, kAngular);
          Evolution3D trial(*moved, setup.solver, setup.state, options);
          const auto movedTrajectory = ForwardTrajectory3D::FromSnapshots(trial.Run(
              times, Held(times, FrozenLoads(setup, Plus(thickness, u.iceThickness, eps)))));
          auto wAt = w;
          wAt.itau = TangentLinearSolver3D::ItauOfLogViscosity(
              *moved, OnCoupling(moved->Coupling(), patternW));
          TangentLinearSolver3D along(trial, {.method = AdjointMethod::Discrete,
                                              .rtol = 1e-12,
                                              .coupled_rtol = 1e-12});
          return Objective(setup, along.Run(times, wAt, &movedTrajectory), measurements);
        },
        0.1, forward);
    std::cout << std::format(
        "[uneven 3D Hessian, {}] second-order run: ratio {:.3f}, Richardson gap {:.2e}\n",
        label, fd.ratio, Gap(fd.richardson, forward));
    EXPECT_GT(fd.ratio, 3.0) << label;
    EXPECT_LT(fd.ratio, 5.0) << label;
    EXPECT_NEAR(fd.richardson, forward, 1e-6 * std::abs(forward)) << label;
  }
}
