#pragma once

// The scene the kernel tests share (TestTangentLinear, TestSecondOrderAdjoint):
// the prem8 deck's body with a Maxwell mantle, its elastic solver and the
// analytic test state; smooth random fields and itau perturbations; the
// objective of a measurement set on snapshots; a per-node load history;
// and pyslfp's marine-grounded cap for migrating runs.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <map>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Fingerprint.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

namespace kernel_scene {

using namespace sl3dpp;

constexpr auto kLmax = 16;
constexpr auto kYear = 3.15576e7;  // s

/// The prem8 deck's body with a Maxwell mantle of the given viscosity
/// (Pa s) between the CMB and 220 km depth, its elastic solver and the
/// analytic test state (the adjoint tests' scene).
struct Scene {
  EarthModel model;
  Material material;
  Rheology rheology;
  SeaLevelSolver solver;
  SeaLevelSolver::State state;

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
    SeaLevelSolver solver(SurfaceGrid(kLmax), op.ElasticLoveNumbers(),
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

  [[nodiscard]] double Metre() const {
    return 1.0 / solver.SolverScales().Length();
  }
  [[nodiscard]] double Kyr() const {
    return 1e3 * kYear / solver.SolverScales().Time();
  }
  /// A unit surface density, solver units.
  [[nodiscard]] double Sigma() const {
    return 1.0 / solver.SolverScales().Factor(Dimensions{.mass = 1, .length = -2});
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

  /// A smooth nodal perturbation of branch 0's inverse relaxation time,
  /// material units, supported where the branch relaxes.
  [[nodiscard]] std::vector<std::vector<double>> DeltaItau(
      double fraction = 1.0,
      const std::function<double(double)>& shape =
          [](double r) { return 0.3 + 0.7 * r * r; }) const {
    const auto& base = rheology.BranchAt(0);
    const auto& mesh = material.Mesh();
    auto top = 0.0;
    for (auto v : base.itau) top = std::max(top, v);
    auto ditau = std::vector<double>(base.itau.size(), 0.0);
    for (auto e = 0; e < mesh.ElementCount(); ++e) {
      for (auto q = 0; q < mesh.NGll(); ++q) {
        const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
        if (base.itau[i] > 0.0) {
          const auto r = mesh.Radius(e, q);
          ditau[i] = fraction * shape(r) * top;
        }
      }
    }
    return {ditau};
  }
};

/// The objective the measurements define, evaluated on snapshots.
inline double Objective(const Scene& setup,
                 const std::vector<SeaLevelEvolution::Snapshot>& snapshots,
                 const std::map<std::size_t, Measurement>& measurements) {
  auto total = 0.0;
  for (const auto& [node, m] : measurements) {
    const auto& snap = snapshots[node];
    if (!m.seaLevelDual.empty()) {
      total += setup.Surface(m.seaLevelDual, snap.seaLevelChange);
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
inline std::function<std::vector<double>(double)> History(
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

inline double MaxAbs(std::span<const double> f) {
  auto out = 0.0;
  for (auto v : f) out = std::max(out, std::abs(v));
  return out;
}

inline double MaxDiff(std::span<const double> a, std::span<const double> b) {
  auto out = 0.0;
  for (auto i = std::size_t{0}; i < a.size(); ++i) {
    out = std::max(out, std::abs(a[i] - b[i]));
  }
  return out;
}

/// The full measurement set of the adjoint tests: all four duals, at the
/// first, a middle and the last node.
inline std::map<std::size_t, Measurement> FullMeasurements(const Scene& setup,
                                                    std::mt19937& rng,
                                                    std::size_t last) {
  auto measurements = std::map<std::size_t, Measurement>{};
  measurements[0].seaLevelDual = setup.Smooth(rng, 0.3);
  Measurement middle;
  middle.seaLevelDual = setup.Smooth(rng, 1.0);
  middle.displacementDual = setup.Smooth(rng, -0.5);
  measurements[last / 2 + 1] = std::move(middle);
  Measurement end;
  end.displacementDual = setup.Smooth(rng, 0.7);
  end.potentialDual = setup.Smooth(rng, 0.4);
  end.angularVelocityDual = {0.2, -0.4, 0.8};
  measurements[last] = std::move(end);
  return measurements;
}

/// pyslfp's marine-grounded cap on a uniform ocean (the initial-sea-level
/// tests' scene): a 5 km cap at 60N 85W on a 3000 m ocean, complete at
/// 10 ka and melting linearly to nothing.
struct MarineCap {
  std::vector<double> ice0, sea0, times;
  double span = 0.0;

  static MarineCap Make(const Scene& setup, int nodes) {
    const auto& grid = setup.solver.Grid();
    const auto theta0 = 30.0 * std::numbers::pi / 180.0;
    const auto phi0 = 275.0 * std::numbers::pi / 180.0;
    const auto radius = 20.0 * std::numbers::pi / 180.0;
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
    out.span = 10.0 * setup.Kyr();
    for (auto n = 0; n < nodes; ++n) {
      out.times.push_back(out.span * n / (nodes - 1));
    }
    return out;
  }

  [[nodiscard]] std::vector<double> Thickness(double t) const {
    const auto fraction = std::clamp(1.0 - t / span, 0.0, 1.0);
    auto out = ice0;
    for (auto& v : out) v *= fraction;
    return out;
  }
};

}  // namespace kernel_scene
