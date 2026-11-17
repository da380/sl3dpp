// The derivatives of the stepped map under Sdirk2 (StepRecipe.hpp), whose
// step is two backward Euler stages, the first closed under the forcing at
// its own time inside the step: the tangent-linear runs against
// differences of the forward run (the forcing linear in time across each
// step, as the derivative runs take it between its node values), the
// discrete sweeps as the tangents' exact transposes, the discrete Hessian
// symmetric and the derivative of the gradient — about frozen and migrating
// shorelines, with rotation, across a jump of the forcing, under interior
// sources, from a recording with gaps, 1D and, under lateral viscosity, 3D;
// and what stays refused there by name.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
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

namespace {

/// A per-node history read at any time of its span: linear in time between
/// the two nodes about it, a node's own value at the node. What a stage
/// inside a step reads, and what the derivative runs take a direction
/// given at the nodes to be there.
template <class T>
std::function<std::vector<T>(double)> Linear(std::vector<double> times,
                                             std::vector<std::vector<T>> values) {
  return [times = std::move(times), values = std::move(values)](double t) {
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) return values[n];
    }
    for (auto n = std::size_t{1}; n < times.size(); ++n) {
      if (t > times[n - 1] && t < times[n]) {
        const auto c = (t - times[n - 1]) / (times[n] - times[n - 1]);
        auto out = values[n];
        for (auto i = std::size_t{0}; i < out.size(); ++i) {
          out[i] = c * values[n][i] + (1.0 - c) * values[n - 1][i];
        }
        return out;
      }
    }
    throw std::runtime_error("the test history is read inside its span");
  };
}

/// The same on a grid with a repeated node (a jump): `plain` reads the
/// pair's second node at its time and `before` its first, and between
/// nodes both are linear between the step's start (a right limit) and its
/// end (a left limit).
template <class T>
struct LinearByNode {
  std::function<std::vector<T>(double)> plain, before;

  LinearByNode(std::vector<double> times, std::vector<std::vector<T>> values) {
    auto read = [times, values](double t, bool first) {
      auto found = times.size();
      for (auto n = std::size_t{0}; n < times.size(); ++n) {
        if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) {
          found = n;
          if (first) break;
        }
      }
      if (found != times.size()) return values[found];
      for (auto n = std::size_t{1}; n < times.size(); ++n) {
        if (t > times[n - 1] && t < times[n]) {
          const auto c = (t - times[n - 1]) / (times[n] - times[n - 1]);
          auto out = values[n];
          for (auto i = std::size_t{0}; i < out.size(); ++i) {
            out[i] = c * values[n][i] + (1.0 - c) * values[n - 1][i];
          }
          return out;
        }
      }
      throw std::runtime_error("the test history is read inside its span");
    };
    plain = [read](double t) { return read(t, false); };
    before = [read](double t) { return read(t, true); };
  }
};

const auto kSdirk2 = Scheme::Sdirk2;

/// A marine-grounded cap on a uniform ocean (the migrating gates' scene):
/// a 5 km cap at 60N 85W, of the given angular radius, on a 3000 m ocean,
/// complete at the start and melting linearly to nothing over the span.
struct MarineCap {
  std::vector<double> ice0, sea0;
  double span = 0.0;

  static MarineCap Make(const Scene& setup, double span, double degrees = 20.0) {
    const auto& grid = setup.solver.Grid();
    const auto theta0 = 30.0 * std::numbers::pi / 180.0;
    const auto phi0 = 275.0 * std::numbers::pi / 180.0;
    const auto radius = degrees * std::numbers::pi / 180.0;
    auto cap = grid.FromFunction([&](double theta, double phi) {
      const auto c = std::cos(theta) * std::cos(theta0) +
                     std::sin(theta) * std::sin(theta0) * std::cos(phi - phi0);
      const auto d = std::acos(std::clamp(c, -1.0, 1.0));
      if (d >= radius) return 0.0;
      const auto taper = std::cos(0.5 * std::numbers::pi * d / radius);
      return taper * taper;
    });
    MarineCap out;
    out.ice0 = grid.Synthesise(grid.Expand(cap));
    for (auto& v : out.ice0) v *= 5000.0 * setup.Metre();
    out.sea0.assign(grid.FieldSize(), 3000.0 * setup.Metre());
    out.span = span;
    return out;
  }

  [[nodiscard]] std::vector<double> Thickness(double t) const {
    const auto fraction = std::clamp(1.0 - t / span, 0.0, 1.0);
    auto out = ice0;
    for (auto& v : out) v *= fraction;
    return out;
  }
};

/// Interior sources given node by node at fixed points, read at any time:
/// each moment linear in time between the two nodes about it; on a grid
/// with a repeated node the plain callback reads the pair's second node at
/// its time and the left limit its first.
template <class Forcing = Evolution::Forcing>
Forcing LinearSources(std::vector<double> times,
                      std::vector<std::vector<PointSource>> perNode) {
  auto read = [times, perNode](double t, bool first) {
    auto found = times.size();
    for (auto n = std::size_t{0}; n < times.size(); ++n) {
      if (std::abs(t - times[n]) <= 1e-12 * std::abs(times.back())) {
        found = n;
        if (first) break;
      }
    }
    if (found != times.size()) return perNode[found];
    for (auto n = std::size_t{1}; n < times.size(); ++n) {
      if (t > times[n - 1] && t < times[n]) {
        const auto c = (t - times[n - 1]) / (times[n] - times[n - 1]);
        // the points of either node (a node without sources has none: its
        // moments are zero)
        const auto& a = perNode[n - 1];
        const auto& b = perNode[n];
        auto out = b.empty() ? a : b;
        for (auto p = std::size_t{0}; p < out.size(); ++p) {
          auto& m = out[p].moment;
          const auto ma = a.empty() ? SphericalTensor{} : a[p].moment;
          const auto mb = b.empty() ? SphericalTensor{} : b[p].moment;
          m.rr = (1.0 - c) * ma.rr + c * mb.rr;
          m.tt = (1.0 - c) * ma.tt + c * mb.tt;
          m.pp = (1.0 - c) * ma.pp + c * mb.pp;
          m.rt = (1.0 - c) * ma.rt + c * mb.rt;
          m.rp = (1.0 - c) * ma.rp + c * mb.rp;
          m.tp = (1.0 - c) * ma.tp + c * mb.tp;
        }
        return out;
      }
    }
    throw std::runtime_error("the test sources are read inside their span");
  };
  auto out = Forcing{};
  out.sources = [read](double t) { return read(t, false); };
  out.sourcesBefore = [read](double t) { return read(t, true); };
  return out;
}

/// A time grid of `steps` equal steps over [0, span] with node `at`
/// repeated: a jump of the forcing there.
std::vector<double> JumpGrid(int steps, double span, int at) {
  auto times = std::vector<double>{};
  for (auto n = 0; n <= steps; ++n) {
    times.push_back(span * n / steps);
    if (n == at) times.push_back(times.back());
  }
  return times;
}

}  // namespace

// -- 1D -------------------------------------------------------------------

TEST(Sdirk2Derivatives, TheTangentIsTheDerivativeOfTheRun) {
  // On the uneven grid: along the loads and the ice thickness (non-zero at
  // the first node), each linear in time across a step, the run is the
  // difference of two forward runs (the frozen problem is linear in
  // them); along the relaxation times and the moduli the discrete run is
  // the derivative of the forward run (central differences at second
  // order, Richardson landing on it).
  auto setup = Scene::Make(16, 1e21);
  std::mt19937 rng(101);
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
  const auto measurements = FullMeasurements(setup, rng, 3, nodes - 1);

  ViscoelasticOperator op(setup.rheology, setup.lmax);
  const Evolution::Options options{.scheme = kSdirk2, .rtol = 1e-12, .record_stride = 1};
  Evolution evolution(op, setup.solver, setup.state, options);
  const auto base = evolution.Run(times, Linear(times, loads));
  const auto plus = evolution.Run(times, Linear(times, Plus(loads, deltas)));
  const auto iceBase = evolution.Run(times, Linear(times, FrozenLoads(setup, thickness)));
  const auto icePlus =
      evolution.Run(times, Linear(times, FrozenLoads(setup, Plus(thickness, deltaIce))));
  const auto trajectory = ForwardTrajectory::FromSnapshots(base);
  // the recording keeps each step's first-stage closure beside the nodes'
  ASSERT_EQ(trajectory.stages.size(), nodes);
  EXPECT_TRUE(trajectory.stages[0].empty());
  for (auto n = std::size_t{1}; n < nodes; ++n) EXPECT_EQ(trajectory.stages[n].size(), 1u);

  TangentLinearSolver tangent(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
  TangentPerturbation alongLoads, alongIce;
  alongLoads.directLoad = deltas;
  alongIce.iceThickness = deltaIce;
  const auto runLoads = tangent.Run(times, alongLoads);
  const auto runIce = tangent.Run(times, alongIce);
  auto worst = 0.0;
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    for (auto field : {&Evolution::Snapshot::relativeSeaLevelChange,
                       &Evolution::Snapshot::displacement, &Evolution::Snapshot::potential,
                       &Evolution::Snapshot::tangentialPotential}) {
      for (const auto& [run, p, b] : {std::tuple{&runLoads, &plus, &base},
                                      std::tuple{&runIce, &icePlus, &iceBase}}) {
        auto difference = (*p)[n].*field;
        for (auto i = std::size_t{0}; i < difference.size(); ++i) {
          difference[i] -= ((*b)[n].*field)[i];
        }
        worst = std::max(worst, MaxDiff((*run)[n].*field, difference) / MaxAbs(difference));
        EXPECT_LE(MaxDiff((*run)[n].*field, difference), 1e-8 * MaxAbs(difference))
            << "node " << n;
      }
    }
  }
  std::cout << std::format("[sdirk2 tangent] loads and ice against differences: {:.2e}\n",
                           worst);

  // the relaxation times
  TangentPerturbation alongItau;
  alongItau.itau = setup.DeltaItau(0.1);
  const auto predictedItau =
      Objective(setup, tangent.Run(times, alongItau, &trajectory), measurements);
  const auto itau = Difference(
      [&](double eps) {
        const auto moved = MovedOperator(setup, op, eps, {}, alongItau.itau);
        Evolution trial(moved, setup.solver, setup.state, options);
        return Objective(setup, trial.Run(times, Linear(times, loads)), measurements);
      },
      0.1, predictedItau);
  std::cout << std::format("[sdirk2 tangent] itau: ratio {:.3f}, Richardson gap {:.2e}\n",
                           itau.ratio, Gap(itau.richardson, predictedItau));
  EXPECT_GT(itau.ratio, 3.0);
  EXPECT_LT(itau.ratio, 5.0);
  EXPECT_NEAR(itau.richardson, predictedItau, 1e-6 * std::abs(predictedItau));

  // the moduli
  TangentPerturbation alongMaterial;
  alongMaterial.material =
      PointDirection(op, setup.material, [](double r) { return 0.5 + 0.5 * r; });
  const auto predictedMaterial =
      Objective(setup, tangent.Run(times, alongMaterial, &trajectory), measurements);
  const auto material = Difference(
      [&](double eps) {
        const auto moved = MovedOperator(setup, op, eps, alongMaterial.material);
        Evolution trial(moved, setup.solver, setup.state, options);
        return Objective(setup, trial.Run(times, Linear(times, loads)), measurements);
      },
      0.1, predictedMaterial);
  std::cout << std::format("[sdirk2 tangent] moduli: ratio {:.3f}, Richardson gap {:.2e}\n",
                           material.ratio, Gap(material.richardson, predictedMaterial));
  EXPECT_GT(material.ratio, 3.0);
  EXPECT_LT(material.ratio, 5.0);
  EXPECT_NEAR(material.richardson, predictedMaterial, 1e-6 * std::abs(predictedMaterial));
}

TEST(Sdirk2Derivatives, TheDiscreteSweepIsTheTangentsTranspose) {
  // The discrete sweep is the exact transpose of the two-stage step: its
  // load kernels (node 0 included), thickness view, itau kernel and moduli
  // kernels each pair with their direction to the discrete tangent's
  // objective change, to rounding, under the full rotational feedback; a
  // stride-2 recording, its gaps rebuilt by the run's own stages under
  // their recorded closures, serves the same kernels.
  auto setup = Scene::Make(16, 1e21);
  std::mt19937 rng(103);
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
  const auto measurements = FullMeasurements(setup, rng, 3, nodes - 1);

  ViscoelasticOperator op(setup.rheology, setup.lmax);
  Evolution evolution(op, setup.solver, setup.state,
                      {.scheme = kSdirk2, .rtol = 1e-12, .record_stride = 1});
  const auto trajectory =
      ForwardTrajectory::FromSnapshots(evolution.Run(times, Linear(times, loads)));
  const AdjointSolver::Options adjointOptions{
      .method = AdjointMethod::Discrete, .rtol = 1e-12, .material_kernels = true};
  const auto kernels =
      AdjointSolver(evolution, adjointOptions).FirstOrderKernels(times, measurements, &trajectory);
  TangentLinearSolver tangent(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
  auto along = [&](const TangentPerturbation& p) {
    return Objective(setup, tangent.Run(times, p, &trajectory), measurements);
  };
  TangentPerturbation pLoads, pIce, pItau, pMaterial;
  pLoads.directLoad = deltas;
  pIce.iceThickness = deltaIce;
  pItau.itau = setup.DeltaItau(0.1);
  pMaterial.material =
      PointDirection(op, setup.material, [](double r) { return 0.5 + 0.5 * r; });
  const auto& mesh = setup.material.Mesh();
  const auto itau = VolumePairing(mesh, kernels.itauKernel[0], pItau.itau[0]);
  for (const auto& [name, paired, p] :
       {std::tuple{"loads", NodePairing(setup, kernels.directLoadKernels, deltas), &pLoads},
        std::tuple{"ice", NodePairing(setup, kernels.thickness, deltaIce), &pIce},
        std::tuple{"itau", itau, &pItau},
        std::tuple{"moduli",
                   VolumePairing(op.Points(), mesh, kernels.kappaKernel,
                                 pMaterial.material.kappa) +
                       VolumePairing(op.Points(), mesh, kernels.muKernel,
                                     pMaterial.material.mu),
                   &pMaterial}}) {
    const auto want = along(*p);
    ASSERT_NE(want, 0.0) << name;
    std::cout << std::format("[sdirk2 transpose] {}: gap {:.2e}\n", name, Gap(paired, want));
    EXPECT_NEAR(paired, want, 1e-9 * std::abs(want)) << name;
  }

  Evolution sparse(op, setup.solver, setup.state,
                   {.scheme = kSdirk2, .rtol = 1e-12, .record_stride = 2});
  const auto sparseTrajectory =
      ForwardTrajectory::FromSnapshots(sparse.Run(times, Linear(times, loads)));
  const auto again = AdjointSolver(evolution, adjointOptions)
                         .FirstOrderKernels(times, measurements, &sparseTrajectory);
  const auto viaSparse = VolumePairing(mesh, again.itauKernel[0], pItau.itau[0]);
  std::cout << std::format("[sdirk2 transpose] stride 2: itau gap {:.2e}\n",
                           Gap(viaSparse, itau));
  EXPECT_NEAR(viaSparse, itau, 1e-10 * std::abs(itau));
  EXPECT_NEAR(Objective(setup, tangent.Run(times, pItau, &sparseTrajectory), measurements),
              along(pItau), 1e-10 * std::abs(itau));
}

TEST(Sdirk2Derivatives, AboutAMigratingTrajectory) {
  // About a migrating run each stage is linearised about its own
  // shorelines, the first stage's recorded beside the nodes'
  // (StageOceanFunctionsOf): the tangent run along a thickness and an
  // initial sea level perturbation that flip no flotation cell matches
  // central differences of migrating runs (the map is piecewise affine),
  // the relaxation times' against differences at second order, and the
  // discrete sweep's thickness view, initial sea level kernel and itau
  // kernel pair to the tangent's numbers to rounding.
  auto setup = Scene::Make(16, 1e21);
  ViscoelasticOperator op(setup.rheology, setup.lmax);
  const auto times = UnevenTimes(setup.Kyr(), std::vector<double>{2.0, 3.0, 1.0, 4.0});
  const auto nodes = times.size();
  const auto cap = MarineCap::Make(setup, times.back());
  const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
  const auto thickness = [&](double t) { return cap.Thickness(t); };
  const Evolution::Options options{.scheme = kSdirk2, .rtol = 1e-12, .record_stride = 1};
  Evolution evolution(op, setup.solver, state, options);
  const auto run = evolution.RunMigrating(times, thickness);
  const auto oceans = OceanFunctionsOf(run);
  const auto stageOceans = StageOceanFunctionsOf(run);
  const auto trajectory = ForwardTrajectory::FromSnapshots(run);
  ASSERT_EQ(stageOceans.size(), nodes);
  // the shorelines move through the run, and inside its steps
  EXPECT_NE(oceans.front(), oceans.back());
  auto between = false;
  for (auto n = std::size_t{1}; n < nodes; ++n) {
    ASSERT_EQ(stageOceans[n].size(), 1u);
    between = between || (stageOceans[n][0] != oceans[n] && stageOceans[n][0] != oceans[n - 1]);
  }
  EXPECT_TRUE(between) << "no stage has shorelines of its own";

  std::mt19937 rng(107);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[2].displacementDual = setup.Smooth(rng, 0.5);
  auto deltaIce = std::vector<std::vector<double>>(nodes);
  for (auto n = std::size_t{1}; n < nodes; ++n) {
    deltaIce[n] = setup.Smooth(rng, 1.0 * setup.Metre());
  }
  deltaIce[0].assign(setup.solver.Grid().FieldSize(), 0.0);
  const auto deltaSea = setup.Smooth(rng, 1.0 * setup.Metre());
  const auto iceAt = Linear(times, deltaIce);

  auto functional = [&](double sign) {
    auto sea = state.seaLevel;
    for (auto i = std::size_t{0}; i < sea.size(); ++i) sea[i] += sign * deltaSea[i];
    const auto start = setup.solver.MakeState(state.iceThickness, sea);
    EXPECT_EQ(start.oceanFunction, state.oceanFunction) << "the start flipped";
    Evolution trial(op, setup.solver, start, options);
    const auto snaps = trial.RunMigrating(times, [&, sign](double t) {
      auto out = cap.Thickness(t);
      const auto d = iceAt(t);
      for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += sign * d[i];
      return out;
    });
    for (auto n = std::size_t{0}; n < snaps.size(); ++n) {
      EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped";
      for (auto i = std::size_t{0}; i < snaps[n].stages.size(); ++i) {
        EXPECT_EQ(snaps[n].stages[i].oceanFunction, stageOceans[n][i])
            << "a stage's cell flipped";
      }
    }
    return Objective(setup, snaps, measurements);
  };
  const auto fd = 0.5 * (functional(1.0) - functional(-1.0));

  TangentLinearSolver tangent(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
  TangentPerturbation direction;
  direction.iceThickness = deltaIce;
  direction.initialSeaLevel = deltaSea;
  const auto along =
      Objective(setup, tangent.Run(times, direction, nullptr, oceans, stageOceans),
                measurements);
  std::cout << std::format("[sdirk2 migrating] tangent against differences: {:.2e}\n",
                           Gap(along, fd));
  EXPECT_NEAR(along, fd, 1e-7 * std::abs(fd));

  const auto kernels =
      AdjointSolver(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
          .FirstOrderKernels(times, measurements, &trajectory, oceans, stageOceans);
  const auto paired = setup.Surface(kernels.initialSeaLevel, deltaSea) +
                      NodePairing(setup, kernels.thickness, deltaIce);
  std::cout << std::format("[sdirk2 migrating] ice and sea level transpose: {:.2e}\n",
                           Gap(paired, along));
  EXPECT_NEAR(paired, along, 1e-9 * std::abs(along));
  // the quelled view: the rates of a thickness linear across each step
  auto rated = 0.0;
  for (auto k = std::size_t{0}; k + 1 < nodes; ++k) {
    auto rate = deltaIce[k + 1];
    for (auto i = std::size_t{0}; i < rate.size(); ++i) rate[i] -= deltaIce[k][i];
    rated += setup.Surface(kernels.rate[k], rate);
  }
  EXPECT_NEAR(rated, NodePairing(setup, kernels.thickness, deltaIce),
              1e-10 * std::abs(rated));

  // the relaxation times about the migrating trajectory
  TangentPerturbation alongItau;
  alongItau.itau = setup.DeltaItau(0.1);
  const auto predicted =
      Objective(setup, tangent.Run(times, alongItau, &trajectory, oceans, stageOceans),
                measurements);
  const auto itau = Difference(
      [&](double eps) {
        const auto moved = MovedOperator(setup, op, eps, {}, alongItau.itau);
        Evolution trial(moved, setup.solver, state, options);
        const auto snaps = trial.RunMigrating(times, thickness);
        for (auto n = std::size_t{0}; n < snaps.size(); ++n) {
          EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped";
        }
        return Objective(setup, snaps, measurements);
      },
      0.1, predicted);
  std::cout << std::format(
      "[sdirk2 migrating] itau: ratio {:.3f}, Richardson gap {:.2e}; transpose {:.2e}\n",
      itau.ratio, Gap(itau.richardson, predicted),
      Gap(VolumePairing(setup.material.Mesh(), kernels.itauKernel[0], alongItau.itau[0]),
          predicted));
  EXPECT_GT(itau.ratio, 3.0);
  EXPECT_LT(itau.ratio, 5.0);
  EXPECT_NEAR(itau.richardson, predicted, 1e-6 * std::abs(predicted));
  EXPECT_NEAR(VolumePairing(setup.material.Mesh(), kernels.itauKernel[0], alongItau.itau[0]),
              predicted, 1e-9 * std::abs(predicted));

  // refused by name: the stages' ocean functions left out
  for (const auto& what : {"tangent", "sweep"}) {
    try {
      if (std::string(what) == "tangent") {
        (void)tangent.Run(times, direction, nullptr, oceans);
      } else {
        (void)AdjointSolver(evolution, {.rtol = 1e-12})
            .FirstOrderKernels(times, measurements, &trajectory, oceans);
      }
      ADD_FAILURE() << what << " ran without the stages' ocean functions";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find("StageOceanFunctionsOf"), std::string::npos)
          << error.what();
    }
  }
}

TEST(Sdirk2Derivatives, AcrossAnOnsetUnderInteriorSources) {
  // An earthquake at an interior node, the node repeated (a zero-length
  // step, one stage, under any scheme) on a grid stepped by Sdirk2 on
  // either side. The sources as parameters, each moment linear in time
  // across a step: the tangent run is the forward run under them, and the
  // discrete sweep's moment kernels pair with them to the same number.
  // The relaxation times and the moduli about the source-driven run (its
  // toroidal systems stepped beside the spheroidal, stage by stage): the
  // tangent against differences, the kernels against the tangent.
  auto setup = Scene::Make(16, 1e21);
  const auto& sle = setup.solver.SolverScales();
  std::mt19937 rng(109);
  std::normal_distribution<double> normal;
  const auto times = JumpGrid(4, 4.0 * setup.Kyr(), 2);  // nodes 2 and 3 at 2 kyr
  const auto nodes = times.size();
  const auto surface =
      setup.material.Radius() * setup.model.ModelScales().Length() / sle.Length();
  const auto energy = Dimensions{.mass = 1, .length = 2, .time = -2};
  const auto unit = 1e21 / sle.Factor(energy);
  const auto points = std::vector<BodyPoint>{
      {.radius = surface - 400e3 / sle.Length(), .colatitude = 1.1, .longitude = 4.0},
      {.radius = surface - 150e3 / sle.Length(), .colatitude = 2.2, .longitude = 1.0}};
  auto randomSources = [&] {
    auto at = std::vector<PointSource>{};
    for (const auto& point : points) {
      at.push_back({.at = point,
                    .moment = {.rr = unit * normal(rng),
                               .tt = unit * normal(rng),
                               .pp = unit * normal(rng),
                               .rt = unit * normal(rng),
                               .rp = unit * normal(rng),
                               .tp = unit * normal(rng)}});
    }
    return at;
  };
  const auto acting = randomSources();
  auto base = std::vector<std::vector<PointSource>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    base.push_back(n >= 3 ? acting : std::vector<PointSource>{});
  }
  const auto forcing = LinearSources(times, base);
  auto perturbation = std::vector<std::vector<PointSource>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) perturbation.push_back(randomSources());
  auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);
  measurements[3].displacementDual = setup.Smooth(rng, 0.9);
  for (auto& [node, m] : measurements) {
    m.tangentialPotentialDual = setup.Smooth(rng, 0.6);
    m.toroidalPotentialDual = setup.Smooth(rng, 0.8);
  }

  ViscoelasticOperator op(setup.rheology, setup.lmax);
  const Evolution::Options options{.scheme = kSdirk2, .rtol = 1e-12, .record_stride = 1};
  Evolution evolution(op, setup.solver, setup.state, options);
  // -- the sources as parameters
  {
    const auto forward =
        Objective(setup, evolution.Run(times, LinearSources(times, perturbation)),
                  measurements);
    ASSERT_NE(forward, 0.0);
    TangentPerturbation direction;
    direction.sources = perturbation;
    const auto along = Objective(
        setup,
        TangentLinearSolver(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
            .Run(times, direction),
        measurements);
    const auto kernels = AdjointSolver(evolution, {.method = AdjointMethod::Discrete,
                                                   .rtol = 1e-12,
                                                   .moment_points = points})
                             .FirstOrderKernels(times, measurements);
    ASSERT_EQ(kernels.momentKernels.size(), nodes);
    auto paired = 0.0;
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      for (auto p = std::size_t{0}; p < points.size(); ++p) {
        paired += kernels.momentKernels[n][p].Contract(perturbation[n][p].moment);
      }
    }
    std::cout << std::format(
        "[sdirk2 onset] sources: tangent against the run {:.2e}, moment kernels {:.2e}\n",
        Gap(along, forward), Gap(paired, along));
    EXPECT_NEAR(along, forward, 1e-9 * std::abs(forward));
    EXPECT_NEAR(paired, along, 1e-9 * std::abs(along));
  }

  // -- the relaxation times and the moduli about the source-driven run
  const auto snapshots = evolution.Run(times, forcing);
  const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);
  ASSERT_EQ(trajectory.toroidal.size(), nodes);
  // the pair's step is one stage; the steps beside it two, their first
  // stage's forces recorded
  EXPECT_TRUE(trajectory.stages[3].empty());
  ASSERT_EQ(trajectory.stages[4].size(), 1u);
  EXPECT_TRUE(trajectory.stages[4][0].sourceForces.has_value());
  TangentLinearSolver tangent(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
  const auto kernels = AdjointSolver(evolution, {.method = AdjointMethod::Discrete,
                                                 .rtol = 1e-12,
                                                 .material_kernels = true})
                           .FirstOrderKernels(times, measurements, &trajectory);
  const auto& mesh = setup.material.Mesh();
  TangentPerturbation alongItau, alongMaterial;
  alongItau.itau = setup.DeltaItau(0.1);
  alongMaterial.material =
      PointDirection(op, setup.material, [](double r) { return 0.5 + 0.5 * r; });
  for (const auto& [name, p, eps, paired] :
       {std::tuple{"itau", &alongItau, 0.1,
                   VolumePairing(mesh, kernels.itauKernel[0], alongItau.itau[0])},
        std::tuple{"moduli", &alongMaterial, 0.1,
                   VolumePairing(op.Points(), mesh, kernels.kappaKernel,
                                 alongMaterial.material.kappa) +
                       VolumePairing(op.Points(), mesh, kernels.muKernel,
                                     alongMaterial.material.mu)}}) {
    const auto predicted =
        Objective(setup, tangent.Run(times, *p, &trajectory), measurements);
    ASSERT_NE(predicted, 0.0) << name;
    const auto fd = Difference(
        [&](double e) {
          const auto moved = MovedOperator(setup, op, e, p->material, p->itau);
          Evolution trial(moved, setup.solver, setup.state, options);
          return Objective(setup, trial.Run(times, forcing), measurements);
        },
        eps, predicted);
    std::cout << std::format(
        "[sdirk2 onset] {}: ratio {:.3f}, Richardson gap {:.2e}; transpose {:.2e}\n", name,
        fd.ratio, Gap(fd.richardson, predicted), Gap(paired, predicted));
    EXPECT_GT(fd.ratio, 3.0) << name;
    EXPECT_LT(fd.ratio, 5.0) << name;
    EXPECT_NEAR(fd.richardson, predicted, 1e-6 * std::abs(predicted)) << name;
    EXPECT_NEAR(paired, predicted, 1e-9 * std::abs(predicted)) << name;
  }
}

TEST(Sdirk2Derivatives, AcrossALoadJumpWithATide) {
  // A load put on at an interior node (the node repeated, its left limit
  // without the load) with a prescribed tide, a load outside the water
  // budget and a layer on the solid surface, each linear in time across a
  // step: the tangent run along every surface term is the difference of
  // two forward runs, and each kernel of the discrete sweep (the pair's
  // two sides kernels of their own) pairs with its direction to the
  // tangent's objective change.
  auto setup = Scene::Make(16, 1e21);
  std::mt19937 rng(113);
  const auto times = JumpGrid(4, 6.0 * setup.Kyr(), 2);
  const auto nodes = times.size();
  const auto& grid = setup.solver.Grid();
  auto tideOf = [&](double amplitude) {
    auto c = grid.Expand(setup.Smooth(rng, amplitude));
    ZeroBelowTideDegrees(c);
    return c;
  };
  const auto potential =
      1.0 / setup.solver.SolverScales().Factor(PotentialDimensions);
  struct Terms {
    std::vector<std::vector<double>> load, outside, layer;
    std::vector<std::vector<std::complex<double>>> tide;
  };
  auto terms = [&](double scale, bool jump) {
    Terms out;
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      // the base load is put on at the pair's right node
      out.load.push_back(setup.Smooth(rng, (jump && n < 3 ? 0.0 : 100.0) * scale * setup.Sigma()));
      out.outside.push_back(setup.Smooth(rng, 30.0 * scale * setup.Sigma()));
      out.layer.push_back(setup.Smooth(rng, 10.0 * scale * setup.Metre()));
      out.tide.push_back(tideOf(5.0 * scale * potential));
    }
    return out;
  };
  const auto base = terms(1.0, true);
  const auto delta = terms(0.1, false);
  auto forcingOf = [&](const Terms& t) {
    const auto load = LinearByNode<double>(times, t.load);
    const auto outside = LinearByNode<double>(times, t.outside);
    const auto layer = LinearByNode<double>(times, t.layer);
    const auto tide = LinearByNode<std::complex<double>>(times, t.tide);
    return Evolution::Forcing{.directLoad = load.plain,
                              .directLoadOutsideWaterBudget = outside.plain,
                              .solidSurfaceChange = layer.plain,
                              .tide = tide.plain,
                              .directLoadBefore = load.before,
                              .directLoadOutsideWaterBudgetBefore = outside.before,
                              .solidSurfaceChangeBefore = layer.before,
                              .tideBefore = tide.before};
  };
  auto plus = base;
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    for (auto i = std::size_t{0}; i < plus.load[n].size(); ++i) {
      plus.load[n][i] += delta.load[n][i];
      plus.outside[n][i] += delta.outside[n][i];
      plus.layer[n][i] += delta.layer[n][i];
    }
    for (auto i = std::size_t{0}; i < plus.tide[n].size(); ++i) {
      plus.tide[n][i] += delta.tide[n][i];
    }
  }
  auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);
  measurements[3].relativeSeaLevelDual = setup.Smooth(rng, 0.9);

  ViscoelasticOperator op(setup.rheology, setup.lmax);
  Evolution evolution(op, setup.solver, setup.state, {.scheme = kSdirk2, .rtol = 1e-12});
  const auto difference =
      Objective(setup, evolution.Run(times, forcingOf(plus)), measurements) -
      Objective(setup, evolution.Run(times, forcingOf(base)), measurements);
  TangentLinearSolver tangent(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12});
  TangentPerturbation all;
  all.directLoad = delta.load;
  all.directLoadOutsideWaterBudget = delta.outside;
  all.solidSurfaceChange = delta.layer;
  all.tide = delta.tide;
  const auto along = Objective(setup, tangent.Run(times, all), measurements);
  std::cout << std::format("[sdirk2 load jump] tangent against the runs' difference: {:.2e}\n",
                           Gap(along, difference));
  EXPECT_NEAR(along, difference, 1e-8 * std::abs(difference));

  const auto kernels =
      AdjointSolver(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
          .FirstOrderKernels(times, measurements);
  auto tidePaired = 0.0;
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    const auto a = grid.Synthesise(kernels.tideKernels[n]);
    const auto b = grid.Synthesise(delta.tide[n]);
    auto product = std::vector<double>(a.size());
    for (auto i = std::size_t{0}; i < a.size(); ++i) product[i] = a[i] * b[i];
    tidePaired += grid.Integrate(product);
  }
  TangentPerturbation pLoad, pOutside, pLayer, pTide;
  pLoad.directLoad = delta.load;
  pOutside.directLoadOutsideWaterBudget = delta.outside;
  pLayer.solidSurfaceChange = delta.layer;
  pTide.tide = delta.tide;
  for (const auto& [name, paired, p] :
       {std::tuple{"load", NodePairing(setup, kernels.directLoadKernels, delta.load), &pLoad},
        std::tuple{"outside the budget",
                   NodePairing(setup, kernels.directLoadOutsideWaterBudgetKernels,
                               delta.outside),
                   &pOutside},
        std::tuple{"layer", NodePairing(setup, kernels.solidSurfaceChangeKernels, delta.layer),
                   &pLayer},
        std::tuple{"tide", tidePaired, &pTide}}) {
    const auto want = Objective(setup, tangent.Run(times, *p), measurements);
    ASSERT_NE(want, 0.0) << name;
    std::cout << std::format("[sdirk2 load jump] {}: gap {:.2e}\n", name, Gap(paired, want));
    EXPECT_NEAR(paired, want, 1e-9 * std::abs(want)) << name;
  }
}

TEST(Sdirk2Derivatives, TheDiscreteHessianIsExactAndSymmetric) {
  // The discrete second-order sweep under Sdirk2, about frozen and
  // migrating shorelines, with directions u and w in the ice thickness, the
  // relaxation times and the moduli: it carries the first-order kernels;
  // <H u, w> = <H w, u>, and both equal <g, F''[u, w]> from the discrete
  // second-order run, to rounding; and its derivatives along u are the
  // derivative of the first-order kernels (central differences of the
  // gradient paired with w, the thickness, the relaxation times and the
  // moduli moved by eps u: second order, Richardson landing on it).
  auto setup = Scene::Make(16, 1e21);
  const auto times = UnevenTimes(setup.Kyr(), std::vector<double>{2.0, 3.0, 1.0, 4.0});
  const auto nodes = times.size();
  const auto fieldSize = setup.solver.Grid().FieldSize();
  const auto cap = MarineCap::Make(setup, times.back());
  const Evolution::Options options{.scheme = kSdirk2, .rtol = 1e-12, .record_stride = 1};
  for (const bool migrating : {false, true}) {
    const auto label = migrating ? "migrating" : "frozen";
    std::mt19937 rng(127);
    ViscoelasticOperator op(setup.rheology, setup.lmax);
    const auto state =
        migrating ? setup.solver.MakeState(cap.ice0, cap.sea0) : setup.state;
    auto thickness = std::vector<std::vector<double>>{};
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      thickness.push_back(migrating ? cap.Thickness(times[n])
                                    : setup.Smooth(rng, 100.0 * setup.Metre()));
    }
    const auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);
    TangentPerturbation u, w;
    const auto amplitude = (migrating ? 1.0 : 20.0) * setup.Metre();
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      // (about a migrating run the background's ice is the first node's)
      const auto scale = migrating && n == 0 ? 0.0 : amplitude;
      u.iceThickness.push_back(setup.Smooth(rng, scale));
      w.iceThickness.push_back(setup.Smooth(rng, scale));
      if (scale == 0.0) {
        u.iceThickness.back().assign(fieldSize, 0.0);
        w.iceThickness.back().assign(fieldSize, 0.0);
      }
    }
    u.itau = setup.DeltaItau(0.1);
    w.itau = setup.DeltaItau(0.1, [](double r) { return 0.8 - 0.5 * r; });
    u.material = PointDirection(op, setup.material, [](double r) {
      return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r);
    });
    w.material = PointDirection(op, setup.material, [](double r) { return 1.0 - 0.5 * r; });

    // the base run, and the run with the thickness, the relaxation times
    // and the moduli moved by eps u
    struct Base {
      std::vector<Evolution::Snapshot> run;
      ForwardTrajectory trajectory;
      std::vector<std::vector<double>> oceans;
      StageOceanFunctions stageOceans;
    };
    auto baseOf = [&](Evolution& evolution, double eps) {
      const auto moved = Plus(thickness, u.iceThickness, eps);
      Base out;
      if (migrating) {
        out.run = evolution.RunMigrating(times, Linear(times, moved));
        out.oceans = OceanFunctionsOf(out.run);
        out.stageOceans = StageOceanFunctionsOf(out.run);
      } else {
        out.run = evolution.Run(times, Linear(times, FrozenLoads(setup, moved)));
      }
      out.trajectory = ForwardTrajectory::FromSnapshots(out.run);
      return out;
    };
    Evolution evolution(op, setup.solver, state, options);
    const auto base = baseOf(evolution, 0.0);
    TangentLinearSolver tangent(
        evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .record = true});
    const auto runU =
        tangent.Run(times, u, &base.trajectory, base.oceans, base.stageOceans);
    const auto runW =
        tangent.Run(times, w, &base.trajectory, base.oceans, base.stageOceans);
    // a recorded tangent run keeps each step's first stage
    ASSERT_EQ(runU[1].stages.size(), 1u);
    EXPECT_TRUE(runU[1].stages[0].internalState.has_value());
    const AdjointSolver::Options adjointOptions{
        .method = AdjointMethod::Discrete, .rtol = 1e-12, .material_kernels = true};
    AdjointSolver adjoint(evolution, adjointOptions);
    const auto first = adjoint.FirstOrderKernels(times, measurements, &base.trajectory,
                                                 base.oceans, base.stageOceans);
    const auto [kernelsU, alongU] =
        adjoint.SecondOrderKernels(times, measurements, runU, u.itau, base.trajectory,
                                   base.oceans, u.material, base.stageOceans);
    const auto [kernelsW, alongW] =
        adjoint.SecondOrderKernels(times, measurements, runW, w.itau, base.trajectory,
                                   base.oceans, w.material, base.stageOceans);
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
        setup,
        tangent.RunSecondOrder(times, runU, runW, direction, &base.trajectory, base.oceans,
                               base.stageOceans),
        measurements);
    std::cout << std::format(
        "[sdirk2 Hessian, {}] symmetry gap {:.2e}, against the second-order run {:.2e}\n",
        label, Gap(wu, uw), Gap(uw, forward));
    ASSERT_NE(forward, 0.0) << label;
    EXPECT_NEAR(uw, wu, 1e-9 * std::abs(uw)) << label;
    EXPECT_NEAR(uw, forward, 1e-9 * std::abs(forward)) << label;

    // The derivatives along u against differences of the first-order
    // kernels, paired with w.
    const auto fd = Difference(
        [&](double eps) {
          const auto moved = MovedOperator(setup, op, eps, u.material, u.itau);
          Evolution trial(moved, setup.solver, state, options);
          const auto movedBase = baseOf(trial, eps);
          if (migrating) {
            EXPECT_EQ(movedBase.oceans, base.oceans) << "a cell flipped";
            EXPECT_EQ(movedBase.stageOceans, base.stageOceans) << "a stage's cell flipped";
          }
          return Pairing(setup, moved,
                         AdjointSolver(trial, adjointOptions)
                             .FirstOrderKernels(times, measurements, &movedBase.trajectory,
                                                movedBase.oceans, movedBase.stageOceans),
                         w);
        },
        0.1, uw);
    std::cout << std::format(
        "[sdirk2 Hessian, {}] differences of the gradient: ratio {:.3f}, Richardson gap "
        "{:.2e}\n",
        label, fd.ratio, Gap(fd.richardson, uw));
    EXPECT_GT(fd.ratio, 3.0) << label;
    EXPECT_LT(fd.ratio, 5.0) << label;
    EXPECT_NEAR(fd.richardson, uw, 1e-6 * std::abs(uw)) << label;
  }
}

TEST(Sdirk2Derivatives, TheHessianAboutASourceDrivenRun) {
  // About a run under interior sources that come on at a repeated node,
  // its toroidal systems stepped stage by stage beside the spheroidal:
  // the discrete Hessian in the relaxation times and the moduli is
  // symmetric and equals the second-order run, and its source rows (the
  // moment kernels' derivatives along u) pair with a perturbation of the
  // sources to the mixed second-order run's objective.
  auto setup = Scene::Make(16, 1e21);
  const auto& sle = setup.solver.SolverScales();
  std::mt19937 rng(131);
  std::normal_distribution<double> normal;
  const auto times = JumpGrid(4, 4.0 * setup.Kyr(), 2);
  const auto nodes = times.size();
  const auto surface =
      setup.material.Radius() * setup.model.ModelScales().Length() / sle.Length();
  const auto unit = 1e21 / sle.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto points = std::vector<BodyPoint>{
      {.radius = surface - 400e3 / sle.Length(), .colatitude = 1.1, .longitude = 4.0},
      {.radius = surface - 150e3 / sle.Length(), .colatitude = 2.2, .longitude = 1.0}};
  auto randomSources = [&] {
    auto at = std::vector<PointSource>{};
    for (const auto& point : points) {
      at.push_back({.at = point,
                    .moment = {.rr = unit * normal(rng),
                               .tt = unit * normal(rng),
                               .pp = unit * normal(rng),
                               .rt = unit * normal(rng),
                               .rp = unit * normal(rng),
                               .tp = unit * normal(rng)}});
    }
    return at;
  };
  const auto acting = randomSources();
  auto base = std::vector<std::vector<PointSource>>{};
  auto perturbation = std::vector<std::vector<PointSource>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    base.push_back(n >= 3 ? acting : std::vector<PointSource>{});
    perturbation.push_back(randomSources());
  }
  auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);
  for (auto& [node, m] : measurements) m.toroidalPotentialDual = setup.Smooth(rng, 0.8);

  ViscoelasticOperator op(setup.rheology, setup.lmax);
  Evolution evolution(op, setup.solver, setup.state,
                      {.scheme = kSdirk2, .rtol = 1e-12, .record_stride = 1});
  const auto trajectory =
      ForwardTrajectory::FromSnapshots(evolution.Run(times, LinearSources(times, base)));
  TangentPerturbation u, w, s;
  u.itau = setup.DeltaItau(0.1);
  w.itau = setup.DeltaItau(0.1, [](double r) { return 0.8 - 0.5 * r; });
  u.material = PointDirection(op, setup.material, [](double r) {
    return 0.4 + 0.6 * std::sin(3.0 * r) * std::sin(3.0 * r);
  });
  w.material = PointDirection(op, setup.material, [](double r) { return 1.0 - 0.5 * r; });
  s.sources = perturbation;
  TangentLinearSolver tangent(
      evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .record = true});
  const auto runU = tangent.Run(times, u, &trajectory);
  const auto runW = tangent.Run(times, w, &trajectory);
  const auto runS = tangent.Run(times, s);
  AdjointSolver adjoint(evolution, {.method = AdjointMethod::Discrete,
                                    .rtol = 1e-12,
                                    .material_kernels = true,
                                    .moment_points = points});
  const auto [kernelsU, alongU] = adjoint.SecondOrderKernels(
      times, measurements, runU, u.itau, trajectory, {}, u.material);
  const auto [kernelsW, alongW] = adjoint.SecondOrderKernels(
      times, measurements, runW, w.itau, trajectory, {}, w.material);
  const auto uw = Pairing(setup, op, alongU, w);
  const auto wu = Pairing(setup, op, alongW, u);
  SecondOrderDirection both;
  both.itauU = u.itau;
  both.itauV = w.itau;
  both.materialU = u.material;
  both.materialV = w.material;
  const auto forward = Objective(
      setup, tangent.RunSecondOrder(times, runU, runW, both, &trajectory), measurements);
  ASSERT_NE(forward, 0.0);
  // the source rows against the run mixed in u and the sources
  SecondOrderDirection mixedDirection;
  mixedDirection.itauU = u.itau;
  mixedDirection.materialU = u.material;
  const auto mixed = Objective(
      setup, tangent.RunSecondOrder(times, runU, runS, mixedDirection, &trajectory),
      measurements);
  auto rows = 0.0;
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    for (auto p = std::size_t{0}; p < points.size(); ++p) {
      rows += alongU.momentKernels[n][p].Contract(perturbation[n][p].moment);
    }
  }
  std::cout << std::format(
      "[sdirk2 source-driven Hessian] symmetry gap {:.2e}, against the second-order run "
      "{:.2e}, source rows {:.2e}\n",
      Gap(wu, uw), Gap(uw, forward), Gap(rows, mixed));
  EXPECT_NEAR(uw, wu, 1e-9 * std::abs(uw));
  EXPECT_NEAR(uw, forward, 1e-9 * std::abs(forward));
  ASSERT_NE(mixed, 0.0);
  EXPECT_NEAR(rows, mixed, 1e-9 * std::abs(mixed));
}

namespace {

/// A field history curved in time, quadratic over [0, span]: a + (t / span)
/// b + (t / span)^2 c, which a callback counts its reads of, by time.
struct Curved {
  std::vector<double> a, b, c;
  double span = 1.0;
  std::shared_ptr<std::vector<double>> read = std::make_shared<std::vector<double>>();

  [[nodiscard]] std::vector<double> At(double t) const {
    const auto x = t / span;
    auto out = a;
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += x * b[i] + x * x * c[i];
    return out;
  }
  [[nodiscard]] std::function<std::vector<double>(double)> Callback() const {
    return [self = *this](double t) {
      self.read->push_back(t);
      return self.At(t);
    };
  }
  [[nodiscard]] std::vector<std::vector<double>> AtNodes(std::span<const double> times) const {
    auto out = std::vector<std::vector<double>>{};
    for (const auto t : times) out.push_back(At(t));
    return out;
  }
  /// Whether every read so far was at a node of the grid.
  [[nodiscard]] bool ReadAtNodesAlone(std::span<const double> times) const {
    return std::ranges::all_of(*read, [&](double t) {
      return std::ranges::any_of(times, [&](double node) { return node == t; });
    });
  }
};

/// The largest relative difference of two runs' fields over every node.
template <class Snapshot>
double RunsApart(const std::vector<Snapshot>& got, const std::vector<Snapshot>& want) {
  auto worst = 0.0;
  for (auto n = std::size_t{0}; n < want.size(); ++n) {
    for (auto field : {&Snapshot::relativeSeaLevelChange, &Snapshot::displacement,
                       &Snapshot::potential, &Snapshot::tangentialPotential}) {
      const auto scale = MaxAbs(want[n].*field);
      if (scale > 0.0) worst = std::max(worst, MaxDiff(got[n].*field, want[n].*field) / scale);
    }
  }
  return worst;
}

}  // namespace

TEST(Sdirk2Derivatives, AStageReadsItsStepsNodesAlone) {
  // A run reads its forcing at its nodes alone: Sdirk2's first stage takes
  // 1 - gamma of the forcing at the step's start node and gamma of it at
  // the end node, and calls no callback between nodes. Under a load curved
  // in time across every step, and under a curved ice history with
  // migrating shorelines, the run is the run under the forcing's
  // piecewise-linear interpolant through its nodes, to rounding, and every
  // read of the callback is at a node; the discrete tangent-linear run
  // along a curved direction, given by its node values, is then the
  // derivative of the run (the loads' the difference of two runs, the
  // problem linear in them; the ice's against central differences while
  // no flotation cell flips), as it is for a forcing linear across steps.
  auto setup = Scene::Make(16, 1e21);
  std::mt19937 rng(163);
  const auto times = UnevenTimes(setup.Kyr(), kSteps);
  const auto nodes = times.size();
  ViscoelasticOperator op(setup.rheology, setup.lmax);
  const Evolution::Options options{.scheme = kSdirk2, .rtol = 1e-12};
  auto curved = [&](double amplitude) {
    return Curved{setup.Smooth(rng, amplitude), setup.Smooth(rng, amplitude),
                  setup.Smooth(rng, 2.0 * amplitude), times.back()};
  };
  // -- a curved load, frozen shorelines
  {
    const auto load = curved(100.0 * setup.Sigma());
    const auto delta = curved(10.0 * setup.Sigma());
    Evolution evolution(op, setup.solver, setup.state, options);
    const auto run = evolution.Run(times, load.Callback());
    EXPECT_TRUE(load.ReadAtNodesAlone(times));
    const auto viaNodes = evolution.Run(times, Linear(times, load.AtNodes(times)));
    const auto apart = RunsApart(run, viaNodes);
    // the curve is far from its interpolant inside the steps
    const auto mid = 0.5 * (times[1] + times[2]);
    const auto chord = Linear(times, load.AtNodes(times))(mid);
    EXPECT_GT(MaxDiff(load.At(mid), chord), 0.05 * MaxAbs(chord));
    const auto plus = evolution.Run(times, [&](double t) {
      auto out = load.At(t);
      const auto d = delta.At(t);
      for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += d[i];
      return out;
    });
    TangentPerturbation direction;
    direction.directLoad = delta.AtNodes(times);
    const auto tangent =
        TangentLinearSolver(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
            .Run(times, direction);
    auto worst = 0.0;
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      auto difference = plus[n].relativeSeaLevelChange;
      for (auto i = std::size_t{0}; i < difference.size(); ++i) {
        difference[i] -= run[n].relativeSeaLevelChange[i];
      }
      worst = std::max(worst, MaxDiff(tangent[n].relativeSeaLevelChange, difference) /
                                  MaxAbs(difference));
    }
    std::cout << std::format(
        "[sdirk2 curved load] the run against its nodes' interpolant: {:.2e}; the tangent "
        "against the runs' difference: {:.2e}\n",
        apart, worst);
    EXPECT_LT(apart, 1e-12);
    EXPECT_LT(worst, 1e-8);
  }
  // -- a curved ice history, migrating shorelines
  {
    const auto cap = MarineCap::Make(setup, times.back());
    const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
    // the cap melting along a curve: 1 - (t / span)^2 of it left at t
    auto ice = Curved{cap.ice0, std::vector<double>(cap.ice0.size(), 0.0), cap.ice0,
                      times.back()};
    for (auto& v : ice.c) v = -v;
    Evolution evolution(op, setup.solver, state,
                        {.scheme = kSdirk2, .rtol = 1e-12, .record_stride = 1});
    const auto run = evolution.RunMigrating(times, ice.Callback());
    EXPECT_TRUE(ice.ReadAtNodesAlone(times));
    const auto viaNodes = evolution.RunMigrating(times, Linear(times, ice.AtNodes(times)));
    const auto apart = RunsApart(run, viaNodes);
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      EXPECT_EQ(run[n].oceanFunction, viaNodes[n].oceanFunction) << n;
    }
    const auto oceans = OceanFunctionsOf(run);
    const auto stageOceans = StageOceanFunctionsOf(run);
    EXPECT_NE(oceans.front(), oceans.back());
    auto measurements = std::map<std::size_t, Measurement>{};
    measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
    measurements[2].displacementDual = setup.Smooth(rng, 0.5);
    // a direction curved in time, zero at the first node
    auto delta = curved(0.0);
    delta.b = setup.Smooth(rng, 0.5 * setup.Metre());
    delta.c = setup.Smooth(rng, 1.0 * setup.Metre());
    auto functional = [&](double sign) {
      const auto snaps = evolution.RunMigrating(times, [&, sign](double t) {
        auto out = ice.At(t);
        const auto d = delta.At(t);
        for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += sign * d[i];
        return out;
      });
      for (auto n = std::size_t{0}; n < snaps.size(); ++n) {
        EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped";
        for (auto i = std::size_t{0}; i < snaps[n].stages.size(); ++i) {
          EXPECT_EQ(snaps[n].stages[i].oceanFunction, stageOceans[n][i])
              << "a stage's cell flipped";
        }
      }
      return Objective(setup, snaps, measurements);
    };
    const auto fd = 0.5 * (functional(1.0) - functional(-1.0));
    TangentPerturbation direction;
    direction.iceThickness = delta.AtNodes(times);
    const auto along = Objective(
        setup,
        TangentLinearSolver(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
            .Run(times, direction, nullptr, oceans, stageOceans),
        measurements);
    const auto trajectory = ForwardTrajectory::FromSnapshots(run);
    const auto kernels =
        AdjointSolver(evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12})
            .FirstOrderKernels(times, measurements, &trajectory, oceans, stageOceans);
    const auto paired = NodePairing(setup, kernels.thickness, direction.iceThickness);
    std::cout << std::format(
        "[sdirk2 curved ice, migrating] the run against its nodes' interpolant: {:.2e}; "
        "the tangent against differences: {:.2e}; the thickness kernel: {:.2e}\n",
        apart, Gap(along, fd), Gap(paired, fd));
    EXPECT_LT(apart, 1e-12);
    EXPECT_NEAR(along, fd, 1e-7 * std::abs(fd));
    EXPECT_NEAR(paired, fd, 1e-7 * std::abs(fd));
  }
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

TEST(Sdirk2Derivatives, TheTangentIsTheDerivativeOfTheRun3D) {
  // Under lateral viscosity on the uneven grid: along the loads and the ice
  // thickness (each linear in time across a step) the coupled run is the
  // difference of two coupled runs; along a lateral log-viscosity direction
  // and a lateral direction in the moduli the discrete run is the
  // derivative of the coupled run (central differences at second order,
  // Richardson landing on it).
  auto setup = Scene::Make(8, 1e21);
  std::mt19937 rng(137);
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

  Viscoelastic3D op(setup.rheology, setup.lmax, LateralDelta);
  const Evolution3D::Options options{
      .scheme = kSdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1};
  Evolution3D evolution(op, setup.solver, setup.state, options);
  const auto base = evolution.Run(times, Linear(times, loads));
  const auto plus = evolution.Run(times, Linear(times, Plus(loads, deltas)));
  const auto iceBase = evolution.Run(times, Linear(times, FrozenLoads(setup, thickness)));
  const auto icePlus =
      evolution.Run(times, Linear(times, FrozenLoads(setup, Plus(thickness, deltaIce))));
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(base);
  ASSERT_EQ(trajectory.stages.size(), nodes);
  for (auto n = std::size_t{1}; n < nodes; ++n) EXPECT_EQ(trajectory.stages[n].size(), 1u);

  TangentLinearSolver3D tangent(
      evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .coupled_rtol = 1e-12});
  TangentPerturbation3D alongLoads, alongIce;
  alongLoads.directLoad = deltas;
  alongIce.iceThickness = deltaIce;
  const auto runLoads = tangent.Run(times, alongLoads);
  const auto runIce = tangent.Run(times, alongIce);
  auto worst = 0.0;
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
        worst = std::max(worst, MaxDiff((*run)[n].*field, difference) / MaxAbs(difference));
        EXPECT_LE(MaxDiff((*run)[n].*field, difference), 1e-8 * MaxAbs(difference))
            << "node " << n;
      }
    }
  }
  std::cout << std::format("[sdirk2 3D tangent] loads and ice against differences: {:.2e}\n",
                           worst);

  auto runAt = [&](const Viscoelastic3D& moved) {
    Evolution3D trial(moved, setup.solver, setup.state, options);
    return Objective(setup, trial.Run(times, Linear(times, loads)), measurements);
  };
  TangentPerturbation3D alongViscosity;
  alongViscosity.logViscosity = OnCoupling(op.Coupling(), kPattern);
  const auto predictedViscosity =
      Objective(setup, tangent.Run(times, alongViscosity, &trajectory), measurements);
  const auto viscosity = Difference(
      [&](double eps) { return runAt(*Moved3D(setup, op, eps, kPattern, {}, {})); }, 0.02,
      predictedViscosity);
  std::cout << std::format(
      "[sdirk2 3D tangent] log viscosity: ratio {:.3f}, Richardson gap {:.2e}\n",
      viscosity.ratio, Gap(viscosity.richardson, predictedViscosity));
  EXPECT_GT(viscosity.ratio, 3.0);
  EXPECT_LT(viscosity.ratio, 5.0);
  EXPECT_NEAR(viscosity.richardson, predictedViscosity, 1e-6 * std::abs(predictedViscosity));
  TangentPerturbation3D alongMaterial;
  alongMaterial.material = LateralDirection(op, kRadial, kAngular);
  const auto predictedMaterial =
      Objective(setup, tangent.Run(times, alongMaterial, &trajectory), measurements);
  const auto material = Difference(
      [&](double eps) { return runAt(*Moved3D(setup, op, eps, {}, kRadial, kAngular)); }, 0.1,
      predictedMaterial);
  std::cout << std::format("[sdirk2 3D tangent] moduli: ratio {:.3f}, Richardson gap {:.2e}\n",
                           material.ratio, Gap(material.richardson, predictedMaterial));
  EXPECT_GT(material.ratio, 3.0);
  EXPECT_LT(material.ratio, 5.0);
  EXPECT_NEAR(material.richardson, predictedMaterial, 1e-6 * std::abs(predictedMaterial));
}

TEST(Sdirk2Derivatives, TheDiscreteSweepIsTheTangentsTranspose3D) {
  // Under lateral viscosity, a full measurement set with the toroidal
  // potential: the discrete 3D sweep's load kernels (node 0 included),
  // thickness view, log-viscosity kernel and lateral moduli kernels each
  // pair with their direction to the discrete 3D tangent's objective
  // change; a stride-2 recording, its gaps rebuilt by the run's own stages
  // under their recorded closures, serves the same kernels.
  auto setup = Scene::Make(8, 1e21);
  std::mt19937 rng(139);
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

  Viscoelastic3D op(setup.rheology, setup.lmax, LateralDelta);
  const auto& coupling = op.Coupling();
  Evolution3D evolution(
      op, setup.solver, setup.state,
      {.scheme = kSdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  const auto trajectory =
      ForwardTrajectory3D::FromSnapshots(evolution.Run(times, Linear(times, loads)));
  const AdjointSolver3D::Options adjointOptions{.method = AdjointMethod::Discrete,
                                                .rtol = 1e-12,
                                                .coupled_rtol = 1e-12,
                                                .material_kernels = true};
  const auto kernels =
      AdjointSolver3D(evolution, adjointOptions).FirstOrderKernels(times, measurements, &trajectory);
  TangentLinearSolver3D tangent(
      evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .coupled_rtol = 1e-12});
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
       {std::tuple{"loads", NodePairing(setup, kernels.ice.directLoadKernels, deltas), &pLoads},
        std::tuple{"ice", NodePairing(setup, kernels.ice.thickness, deltaIce), &pIce},
        std::tuple{"log viscosity", viscosity, &pViscosity},
        std::tuple{"moduli",
                   VolumePairing(coupling, kernels.kappaKernel3D, pMaterial.material.kappa) +
                       VolumePairing(coupling, kernels.muKernel3D, pMaterial.material.mu),
                   &pMaterial}}) {
    const auto want = along(*p);
    ASSERT_NE(want, 0.0) << name;
    std::cout << std::format("[sdirk2 3D transpose] {}: gap {:.2e}\n", name,
                             Gap(paired, want));
    EXPECT_NEAR(paired, want, 1e-8 * std::abs(want)) << name;
  }

  Evolution3D sparse(
      op, setup.solver, setup.state,
      {.scheme = kSdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 2});
  const auto sparseTrajectory =
      ForwardTrajectory3D::FromSnapshots(sparse.Run(times, Linear(times, loads)));
  const auto again = AdjointSolver3D(evolution, adjointOptions)
                         .FirstOrderKernels(times, measurements, &sparseTrajectory);
  const auto viaSparse =
      VolumePairing(coupling, again.logViscosityKernel3D[0], pViscosity.logViscosity);
  std::cout << std::format("[sdirk2 3D transpose] stride 2: log viscosity gap {:.2e}\n",
                           Gap(viaSparse, viscosity));
  EXPECT_NEAR(viaSparse, viscosity, 1e-9 * std::abs(viscosity));
  EXPECT_NEAR(Objective(setup, tangent.Run(times, pViscosity, &sparseTrajectory), measurements),
              along(pViscosity), 1e-9 * std::abs(viscosity));
  // the continuous sweep reads a recording with gaps through the same
  // stages
  const auto continuous = AdjointSolver3D(evolution, {.method = AdjointMethod::Continuous,
                                                      .rtol = 1e-12,
                                                      .coupled_rtol = 1e-12});
  const auto dense = continuous.FirstOrderKernels(times, measurements, &trajectory);
  const auto gapped = continuous.FirstOrderKernels(times, measurements, &sparseTrajectory);
  EXPECT_LE(MaxDiff(gapped.logViscosityKernel3D[0], dense.logViscosityKernel3D[0]),
            1e-9 * MaxAbs(dense.logViscosityKernel3D[0]));
}

TEST(Sdirk2Derivatives, TheDiscreteHessianIsExactAndSymmetric3D) {
  // Under lateral viscosity, directions u and w in the ice thickness, the
  // log viscosity and the lateral moduli: the discrete 3D second-order
  // sweep carries the first-order kernels; <H u, w> = <H w, u>, and both
  // equal <g, F''[u, w]> from the discrete 3D second-order run (whose log
  // parameterisation adds the first-order kernel paired with itau_uv); and
  // that run is the derivative along u of the first-order run along w
  // (central differences, second order, Richardson landing on it).
  auto setup = Scene::Make(8, 1e21);
  std::mt19937 rng(149);
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
    return Cap(theta, phi, 1.9, 4.4, 0.7) +
           0.3 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto radialW = [](double r) { return 1.0 - 0.5 * r; };

  Viscoelastic3D op(setup.rheology, setup.lmax, LateralDelta);
  const auto& coupling = op.Coupling();
  const Evolution3D::Options options{
      .scheme = kSdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1};
  Evolution3D evolution(op, setup.solver, setup.state, options);
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(
      evolution.Run(times, Linear(times, FrozenLoads(setup, thickness))));
  TangentPerturbation3D u, w;
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    u.iceThickness.push_back(setup.Smooth(rng, 20.0 * setup.Metre()));
    w.iceThickness.push_back(setup.Smooth(rng, 20.0 * setup.Metre()));
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
  ASSERT_EQ(runU[1].stages.size(), 1u);
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
        << "node " << n;
  }
  EXPECT_LE(MaxDiff(kernelsU.itauKernel3D[0], first.itauKernel3D[0]),
            1e-10 * MaxAbs(first.itauKernel3D[0]));
  EXPECT_LE(MaxDiff(kernelsU.muKernel3D, first.muKernel3D), 1e-10 * MaxAbs(first.muKernel3D));

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
      setup, tangent.RunSecondOrder(times, runU, runW, direction, &trajectory), measurements);
  auto uvTerm = 0.0;
  for (auto k = std::size_t{0}; k < direction.itauUV.size(); ++k) {
    uvTerm += VolumePairing(coupling, first.itauKernel3D[k], direction.itauUV[k]);
  }
  std::cout << std::format(
      "[sdirk2 3D Hessian] symmetry gap {:.2e}, against the second-order run {:.2e}\n",
      Gap(wu, uw), Gap(uw + uvTerm, forward));
  ASSERT_NE(forward, 0.0);
  EXPECT_NEAR(uw, wu, 1e-8 * std::abs(uw));
  EXPECT_NEAR(uw + uvTerm, forward, 1e-8 * std::abs(forward));

  const auto fd = Difference(
      [&](double eps) {
        const auto moved = Moved3D(setup, op, eps, kPattern, kRadial, kAngular);
        Evolution3D trial(*moved, setup.solver, setup.state, options);
        const auto movedTrajectory = ForwardTrajectory3D::FromSnapshots(trial.Run(
            times, Linear(times, FrozenLoads(setup, Plus(thickness, u.iceThickness, eps)))));
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
      "[sdirk2 3D Hessian] second-order run: ratio {:.3f}, Richardson gap {:.2e}\n", fd.ratio,
      Gap(fd.richardson, forward));
  EXPECT_GT(fd.ratio, 3.0);
  EXPECT_LT(fd.ratio, 5.0);
  EXPECT_NEAR(fd.richardson, forward, 1e-6 * std::abs(forward));
}

TEST(Sdirk2Derivatives, AMigratingRunUnderLateralModuli3D) {
  // A marine cap melting on a uniform ocean, its shorelines migrating
  // through the steps and inside them, on a body with lateral viscosity
  // and lateral moduli (whose start is a zero-length step's fixed point):
  // the tangent run along the thickness and the initial sea level against
  // central differences of the migrating runs while no cell flips, the
  // discrete sweep's thickness view, initial sea level kernel and
  // log-viscosity kernel against the tangent to rounding.
  auto setup = Scene::Make(8, 1e21);
  const auto times = UnevenTimes(setup.Kyr(), std::vector<double>{2.0, 3.0, 1.0, 4.0});
  const auto nodes = times.size();
  // (a wide cap: the grid of degree 8 is coarse)
  const auto cap = MarineCap::Make(setup, times.back(), 50.0);
  const auto state = setup.solver.MakeState(cap.ice0, cap.sea0);
  const auto thickness = [&](double t) { return cap.Thickness(t); };
  Viscoelastic3D plain(setup.rheology, setup.lmax, LateralDelta);
  const auto op = Moved3D(setup, plain, 1.0, {}, kRadial, kAngular);
  ASSERT_TRUE(op->HasLateralModuli());
  const auto& coupling = op->Coupling();
  const Evolution3D::Options options{
      .scheme = kSdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1};
  Evolution3D evolution(*op, setup.solver, state, options);
  const auto run = evolution.RunMigrating(times, thickness);
  const auto oceans = OceanFunctionsOf(run);
  const auto stageOceans = StageOceanFunctionsOf(run);
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(run);
  ASSERT_EQ(stageOceans.size(), nodes);
  // the shorelines move through the run, and inside its steps
  EXPECT_NE(oceans.front(), oceans.back());
  auto between = false;
  for (auto n = std::size_t{1}; n < nodes; ++n) {
    between = between || (stageOceans[n][0] != oceans[n] && stageOceans[n][0] != oceans[n - 1]);
  }
  EXPECT_TRUE(between) << "no stage has shorelines of its own";

  std::mt19937 rng(151);
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[nodes - 1].relativeSeaLevelDual = setup.Smooth(rng, 1.0);
  measurements[2].displacementDual = setup.Smooth(rng, 0.5);
  auto deltaIce = std::vector<std::vector<double>>(nodes);
  for (auto n = std::size_t{1}; n < nodes; ++n) {
    deltaIce[n] = setup.Smooth(rng, 0.02 * setup.Metre());
  }
  deltaIce[0].assign(setup.solver.Grid().FieldSize(), 0.0);
  const auto deltaSea = setup.Smooth(rng, 0.02 * setup.Metre());
  const auto iceAt = Linear(times, deltaIce);
  auto functional = [&](double sign) {
    auto sea = state.seaLevel;
    for (auto i = std::size_t{0}; i < sea.size(); ++i) sea[i] += sign * deltaSea[i];
    const auto start = setup.solver.MakeState(state.iceThickness, sea);
    EXPECT_EQ(start.oceanFunction, state.oceanFunction) << "the start flipped";
    Evolution3D trial(*op, setup.solver, start, options);
    const auto snaps = trial.RunMigrating(times, [&, sign](double t) {
      auto out = cap.Thickness(t);
      const auto d = iceAt(t);
      for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += sign * d[i];
      return out;
    });
    for (auto n = std::size_t{0}; n < snaps.size(); ++n) {
      EXPECT_EQ(snaps[n].oceanFunction, oceans[n]) << "a cell flipped";
      for (auto i = std::size_t{0}; i < snaps[n].stages.size(); ++i) {
        EXPECT_EQ(snaps[n].stages[i].oceanFunction, stageOceans[n][i])
            << "a stage's cell flipped";
      }
    }
    return Objective(setup, snaps, measurements);
  };
  const auto fd = 0.5 * (functional(1.0) - functional(-1.0));
  TangentLinearSolver3D tangent(
      evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .coupled_rtol = 1e-12});
  TangentPerturbation3D direction;
  direction.iceThickness = deltaIce;
  direction.initialSeaLevel = deltaSea;
  const auto along = Objective(
      setup, tangent.Run(times, direction, nullptr, oceans, stageOceans), measurements);
  const auto kernels = AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                   .rtol = 1e-12,
                                                   .coupled_rtol = 1e-12})
                           .FirstOrderKernels(times, measurements, &trajectory, oceans,
                                              stageOceans);
  const auto paired = setup.Surface(kernels.ice.initialSeaLevel, deltaSea) +
                      NodePairing(setup, kernels.ice.thickness, deltaIce);
  TangentPerturbation3D alongViscosity;
  alongViscosity.logViscosity = OnCoupling(coupling, kPattern);
  const auto viscosity = Objective(
      setup, tangent.Run(times, alongViscosity, &trajectory, oceans, stageOceans),
      measurements);
  const auto viscosityPaired = VolumePairing(coupling, kernels.logViscosityKernel3D[0],
                                             alongViscosity.logViscosity);
  std::cout << std::format(
      "[sdirk2 3D migrating, lateral moduli] tangent against differences {:.2e}; ice and "
      "sea level transpose {:.2e}; log viscosity transpose {:.2e}\n",
      Gap(along, fd), Gap(paired, along), Gap(viscosityPaired, viscosity));
  EXPECT_NEAR(along, fd, 1e-7 * std::abs(fd));
  EXPECT_NEAR(paired, along, 1e-8 * std::abs(along));
  ASSERT_NE(viscosity, 0.0);
  EXPECT_NEAR(viscosityPaired, viscosity, 1e-8 * std::abs(viscosity));
  // refused by name without the stages' ocean functions
  EXPECT_THROW((void)tangent.Run(times, direction, nullptr, oceans), std::invalid_argument);
}

TEST(Sdirk2Derivatives, AcrossAnOnsetUnderInteriorSources3D) {
  // An earthquake at a repeated node under lateral viscosity. The sources
  // as parameters, each moment linear in time across a step: the tangent
  // run is the forward run under them and the discrete sweep's moment
  // kernels pair with them to the same number. The log viscosity about
  // the source-driven run (recorded at every node with the first stages'
  // forces): the kernel against the tangent.
  auto setup = Scene::Make(8, 1e21);
  const auto& sle = setup.solver.SolverScales();
  std::mt19937 rng(157);
  std::normal_distribution<double> normal;
  const auto times = JumpGrid(3, 3.0 * setup.Kyr(), 1);  // nodes 1 and 2 at 1 kyr
  const auto nodes = times.size();
  const auto surface =
      setup.material.Radius() * setup.model.ModelScales().Length() / sle.Length();
  const auto unit = 1e21 / sle.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto points = std::vector<BodyPoint>{
      {.radius = surface - 400e3 / sle.Length(), .colatitude = 1.1, .longitude = 4.0}};
  auto randomSources = [&] {
    auto at = std::vector<PointSource>{};
    for (const auto& point : points) {
      at.push_back({.at = point,
                    .moment = {.rr = unit * normal(rng),
                               .tt = unit * normal(rng),
                               .pp = unit * normal(rng),
                               .rt = unit * normal(rng),
                               .rp = unit * normal(rng),
                               .tp = unit * normal(rng)}});
    }
    return at;
  };
  const auto acting = randomSources();
  auto base = std::vector<std::vector<PointSource>>{};
  auto perturbation = std::vector<std::vector<PointSource>>{};
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    base.push_back(n >= 2 ? acting : std::vector<PointSource>{});
    perturbation.push_back(randomSources());
  }
  auto measurements = FullMeasurements(setup, rng, 2, nodes - 1);
  measurements[nodes - 1].toroidalPotentialDual = setup.Smooth(rng, 0.8);

  Viscoelastic3D op(setup.rheology, setup.lmax, LateralDelta);
  const auto& coupling = op.Coupling();
  Evolution3D evolution(
      op, setup.solver, setup.state,
      {.scheme = kSdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12, .record_stride = 1});
  TangentLinearSolver3D tangent(
      evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .coupled_rtol = 1e-12});
  {
    const auto forward = Objective(
        setup,
        evolution.Run(times, LinearSources<Evolution3D::Forcing>(times, perturbation)),
        measurements);
    TangentPerturbation3D direction;
    direction.sources = perturbation;
    const auto along = Objective(setup, tangent.Run(times, direction), measurements);
    const auto kernels = AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                     .rtol = 1e-12,
                                                     .coupled_rtol = 1e-12,
                                                     .moment_points = points})
                             .FirstOrderKernels(times, measurements);
    auto paired = 0.0;
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      for (auto p = std::size_t{0}; p < points.size(); ++p) {
        paired += kernels.ice.momentKernels[n][p].Contract(perturbation[n][p].moment);
      }
    }
    std::cout << std::format(
        "[sdirk2 3D onset] sources: tangent against the run {:.2e}, moment kernels {:.2e}\n",
        Gap(along, forward), Gap(paired, along));
    ASSERT_NE(forward, 0.0);
    EXPECT_NEAR(along, forward, 1e-8 * std::abs(forward));
    EXPECT_NEAR(paired, along, 1e-8 * std::abs(along));
  }
  const auto trajectory = ForwardTrajectory3D::FromSnapshots(
      evolution.Run(times, LinearSources<Evolution3D::Forcing>(times, base)));
  ASSERT_EQ(trajectory.stages[3].size(), 1u);
  EXPECT_TRUE(trajectory.stages[3][0].sourceForces.has_value());
  const auto kernels = AdjointSolver3D(evolution, {.method = AdjointMethod::Discrete,
                                                   .rtol = 1e-12,
                                                   .coupled_rtol = 1e-12})
                           .FirstOrderKernels(times, measurements, &trajectory);
  TangentPerturbation3D alongViscosity;
  alongViscosity.logViscosity = OnCoupling(coupling, kPattern);
  const auto predicted =
      Objective(setup, tangent.Run(times, alongViscosity, &trajectory), measurements);
  const auto paired = VolumePairing(coupling, kernels.logViscosityKernel3D[0],
                                    alongViscosity.logViscosity);
  std::cout << std::format("[sdirk2 3D onset] log viscosity transpose: {:.2e}\n",
                           Gap(paired, predicted));
  ASSERT_NE(predicted, 0.0);
  EXPECT_NEAR(paired, predicted, 1e-8 * std::abs(predicted));
}

TEST(Sdirk2Derivatives, AStageReadsItsStepsNodesAlone3D) {
  // The 3D run under lateral viscosity reads its forcing at its nodes
  // alone too: under a load curved in time across every step it is the
  // run under the load's piecewise-linear interpolant through its nodes,
  // to the closure's tolerance, every read of the callback at a node, and
  // the discrete tangent-linear run along a curved direction's node values
  // is the difference of two coupled runs.
  auto setup = Scene::Make(8, 1e21);
  std::mt19937 rng(167);
  const auto times = UnevenTimes(setup.Kyr(), kSteps3D);
  const auto nodes = times.size();
  auto curved = [&](double amplitude) {
    return Curved{setup.Smooth(rng, amplitude), setup.Smooth(rng, amplitude),
                  setup.Smooth(rng, 2.0 * amplitude), times.back()};
  };
  const auto load = curved(100.0 * setup.Sigma());
  const auto delta = curved(10.0 * setup.Sigma());
  Viscoelastic3D op(setup.rheology, setup.lmax, LateralDelta);
  Evolution3D evolution(op, setup.solver, setup.state,
                        {.scheme = kSdirk2, .rtol = 1e-12, .coupled_rtol = 1e-12});
  const auto run = evolution.Run(times, load.Callback());
  EXPECT_TRUE(load.ReadAtNodesAlone(times));
  const auto viaNodes = evolution.Run(times, Linear(times, load.AtNodes(times)));
  const auto apart = RunsApart(run, viaNodes);
  const auto plus = evolution.Run(times, [&](double t) {
    auto out = load.At(t);
    const auto d = delta.At(t);
    for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += d[i];
    return out;
  });
  TangentPerturbation3D direction;
  direction.directLoad = delta.AtNodes(times);
  const auto tangent =
      TangentLinearSolver3D(
          evolution, {.method = AdjointMethod::Discrete, .rtol = 1e-12, .coupled_rtol = 1e-12})
          .Run(times, direction);
  auto worst = 0.0;
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    auto difference = plus[n].relativeSeaLevelChange;
    for (auto i = std::size_t{0}; i < difference.size(); ++i) {
      difference[i] -= run[n].relativeSeaLevelChange[i];
    }
    worst = std::max(worst, MaxDiff(tangent[n].relativeSeaLevelChange, difference) /
                                MaxAbs(difference));
  }
  std::cout << std::format(
      "[sdirk2 3D curved load] the run against its nodes' interpolant: {:.2e}; the tangent "
      "against the runs' difference: {:.2e}\n",
      apart, worst);
  EXPECT_LT(apart, 1e-10);
  EXPECT_LT(worst, 1e-8);
}
