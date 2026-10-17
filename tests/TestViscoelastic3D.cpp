// The 3D stepper: the zero-lateral step gate (a full implicit step through
// the grid machinery equals the 1D two-phase stepper at machine
// precision), the onset of spheroidal-toroidal coupling under lateral
// viscosity, and the dt self-convergence of the laterally varying step —
// the first research-grade measurements of stage 4 — and the threaded step
// against the serial one.

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <map>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using Complex = std::complex<double>;

namespace {

constexpr int kLmax = 8;

double Noise(std::size_t k) {
  const auto x = std::sin(12.9898 * static_cast<double>(k + 1)) * 43758.5453;
  return x - std::floor(x) - 0.5;
}

struct Body {
  EarthModel model;
  Material material;
  Rheology rheology;

  static Body Make() {
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
    std::map<std::size_t, RadialValue> viscosity;
    const auto b = model.DimensionlessView().Boundaries();
    for (std::size_t i = 0; i < model.LayerCount(); ++i) {
      if (!model.IsFluid(i) && b[i] >= 3.4799e6 / scales.Length() &&
          b[i + 1] <= 6.1511e6 / scales.Length()) {
        viscosity.emplace(i, 1e21 / scales.Factor(ViscosityDimensions));
      }
    }
    auto rheology = Rheology::Maxwell(material, viscosity).value();
    return Body{std::move(model), std::move(material), std::move(rheology)};
  }
};

/// A deterministic constant load: the m >= 0 columns of a real field
/// (m = 0 real) for the 3D channels and the matching [Re | Im] real
/// blocks for the 1D ones.
std::vector<Complex> LoadColumns(int l) {
  std::vector<Complex> columns(l + 1);
  for (int m = 0; m <= l; ++m) {
    columns[m] = m == 0 ? Complex(Noise(97 * l), 0.0)
                        : Complex(Noise(97 * l + m), Noise(55 * l + m));
  }
  return columns;
}

}  // namespace

TEST(Stage4Gate, ZeroLateralStepMatchesTheDegreeBasisStepper) {
  auto body = Body::Make();
  const double tau = body.rheology.MinRelaxationTime();
  const auto& mesh = body.material.Mesh();
  const auto n = mesh.NGll();

  ViscoelasticOperator op1(body.rheology, kLmax);
  Viscoelastic3D op3(body.rheology, kLmax,
                     [](double, double, double) { return 0.0; });
  const auto& coupling = op3.Coupling();

  const auto load3 = [](int l, double) { return LoadColumns(l); };
  const auto load1 = [](int l, double) {
    const auto columns = LoadColumns(l);
    std::vector<double> real(2 * columns.size());
    for (std::size_t c = 0; c < columns.size(); ++c) {
      real[c] = columns[c].real();
      real[columns.size() + c] = columns[c].imag();
    }
    return real;
  };

  for (auto scheme : {Scheme::ExponentialTrapezoid, Scheme::BackwardEuler}) {
    // Two steps from rest under the held load, both paths.
    auto s1 = op1.ZeroState([](int l) { return 2 * (l + 1); });
    auto s3 = op3.ZeroState();
    std::map<int, std::vector<double>> x1;
    Viscoelastic3D::Solutions x3;
    for (int step = 0; step < 2; ++step) {
      auto [n1, sols1] = op1.Step(s1, 0.7 * tau, scheme, load1);
      s1 = std::move(n1);
      x1 = std::move(sols1);
      auto result = op3.Step(s3, 0.7 * tau, scheme, load3);
      EXPECT_TRUE(result.info.converged);
      EXPECT_LE(result.info.iterations, 3);  // no lateral term: immediate
      s3 = std::move(result.state);
      x3 = std::move(result.solutions);
    }

    // The end-time surface responses agree per (l, m).
    const auto u1 = op1.Surface(x1, Component::U);
    const auto u3 = op3.Surface(x3, Component::U);
    double scale = 0.0, diff = 0.0;
    for (int l = 0; l <= kLmax; ++l) {
      const auto columns = l + 1;
      for (int c = 0; c < columns; ++c) {
        const Complex ours = u3.at(l)[c];
        const Complex ref(u1.at(l)[c], u1.at(l)[columns + c]);
        scale = std::max(scale, std::abs(ref));
        diff = std::max(diff, std::abs(ours - ref));
      }
    }
    EXPECT_GT(scale, 0.0);
    EXPECT_LT(diff, 1e-10 * scale) << "scheme " << static_cast<int>(scheme);

    // The grid state equals the synthesised degree-basis state.
    auto spectral = coupling.ZeroSpectral();
    const auto K = body.rheology.BranchCount();
    for (int l = 0; l <= kLmax; ++l) {
      const auto columns = static_cast<std::size_t>(l + 1);
      std::vector<Complex> M(coupling.NodeCount() * columns, 0.0);
      std::vector<Complex> R = M, S = M;
      const auto e0 =
          l > 0 ? mesh.StartElement(l, TruncationRule::Graded, 1e-8) : 0;
      const auto ne = mesh.ElementCount() - e0;
      const auto& m1 = s1.m.at(l);
      const auto ncols = 2 * columns;
      for (int node = 0; node < coupling.NodeCount(); ++node) {
        const auto e = coupling.ElementOfNode(node);
        if (e < e0) continue;
        const auto q = node % n;
        const auto eL = e - e0;
        for (std::size_t c = 0; c < columns; ++c) {
          auto scalar = [&](int s) {
            const auto at =
                ((static_cast<std::size_t>(s) * ne + eL) * n + q) * ncols;
            return Complex(m1[at + c], m1[at + columns + c]);
          };
          M[node * columns + c] = scalar(0);
          R[node * columns + c] = scalar(1);
          S[node * columns + c] = scalar(2);
        }
      }
      coupling.AddScalarState(spectral, l, M, R, S);
      (void)K;
    }
    const auto reference = coupling.Synthesise(spectral);
    double mScale = 0.0, mDiff = 0.0;
    for (int c = 0; c < 2; ++c) {
      for (std::size_t i = 0; i < reference.c[c].size(); ++i) {
        mScale = std::max(mScale, std::abs(reference.c[c][i]));
        mDiff = std::max(mDiff, std::abs(s3.m[0].c[c][i] - reference.c[c][i]));
      }
    }
    for (std::size_t i = 0; i < reference.r.size(); ++i) {
      mScale = std::max(mScale, std::abs(reference.r[i]));
      mDiff = std::max(mDiff, std::abs(s3.m[0].r[i] - reference.r[i]));
    }
    EXPECT_GT(mScale, 0.0);
    EXPECT_LT(mDiff, 1e-10 * mScale) << "scheme " << static_cast<int>(scheme);
  }
}

TEST(Stage4, LateralViscosityCouplesAndConverges) {
  auto body = Body::Make();
  const double tau = body.rheology.MinRelaxationTime();
  // A strong smooth lateral pattern: one decade peak to peak.
  Viscoelastic3D op(body.rheology, kLmax, [](double, double theta, double phi) {
    return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  });
  const auto load = [](int l, double) { return LoadColumns(l); };

  auto state = op.ZeroState();
  auto result = op.Step(state, tau, Scheme::ExponentialTrapezoid, load);
  ASSERT_TRUE(result.info.converged);
  // The research observable: the within-step contraction. Recorded, not
  // asserted tightly; a decade of lateral contrast should converge well
  // inside the iteration budget.
  EXPECT_LE(result.info.iterations, 25);

  // Lateral viscosity couples spheroidal and toroidal motion: after a
  // step, the toroidal response is alive.
  double toroidal = 0.0, spheroidal = 0.0;
  for (const auto& [l, x] : result.solutions.toroidal) {
    for (auto v : x) toroidal = std::max(toroidal, std::abs(v));
  }
  for (const auto& [l, x] : result.solutions.spheroidal) {
    for (auto v : x) spheroidal = std::max(spheroidal, std::abs(v));
  }
  EXPECT_GT(spheroidal, 0.0);
  EXPECT_GT(toroidal, 1e-8 * spheroidal);
  EXPECT_LT(toroidal, spheroidal);
}

TEST(Stage4, LateralStepSelfConvergesAtSecondOrder) {
  auto body = Body::Make();
  const double tau = body.rheology.MinRelaxationTime();
  Viscoelastic3D op(body.rheology, kLmax, [](double, double theta, double phi) {
    return 0.6 * std::sin(theta) * std::sin(phi);
  });
  const auto load = [](int l, double) { return LoadColumns(l); };
  const double T = 2.0 * tau;

  auto advance = [&](int steps) {
    auto state = op.ZeroState();
    Viscoelastic3D::Solutions solutions;
    for (int i = 0; i < steps; ++i) {
      auto result =
          op.Step(state, T / steps, Scheme::ExponentialTrapezoid, load);
      EXPECT_TRUE(result.info.converged);
      state = std::move(result.state);
      solutions = std::move(result.solutions);
    }
    // The observable: surface U at (2, 0).
    return op.Surface(solutions, Component::U).at(2)[2].real();
  };
  const auto x1 = advance(4);
  const auto x2 = advance(8);
  const auto x4 = advance(16);
  const auto ratio = (x1 - x2) / (x2 - x4);
  EXPECT_NEAR(ratio, 4.0, 1.5);  // the trapezoid's order two
}

TEST(Stage4, ThreadedStepMatchesSerial) {
  // Options::parallel threads the transforms and the per-degree loops
  // (force assembly, solves, strain recovery). The degree loops are
  // bitwise independent of the thread count; the threaded transforms are
  // not (each thread plans its own FFTs), so the comparison is at the
  // iteration's tolerance, not bitwise.
  auto body = Body::Make();
  const double tau = body.rheology.MinRelaxationTime();
  const auto delta = [](double, double theta, double phi) {
    return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto load = [](int l, double) { return LoadColumns(l); };
  auto run = [&](bool parallel) {
    Viscoelastic3D::Options options;
    options.parallel = parallel;
    Viscoelastic3D op(body.rheology, kLmax, delta, options);
    auto state = op.ZeroState();
    Viscoelastic3D::Solutions solutions;
    for (int i = 0; i < 2; ++i) {
      auto result = op.Step(state, tau, Scheme::ExponentialTrapezoid, load);
      EXPECT_TRUE(result.info.converged);
      state = std::move(result.state);
      solutions = std::move(result.solutions);
    }
    return solutions;
  };
  const auto serial = run(false);
  const int saved = omp_get_max_threads();
  omp_set_num_threads(4);
  const auto threaded = run(true);
  omp_set_num_threads(saved);

  double scale = 0.0, diff = 0.0;
  auto compare = [&](const std::map<int, std::vector<Complex>>& a,
                     const std::map<int, std::vector<Complex>>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (const auto& [l, x] : a) {
      const auto& y = b.at(l);
      ASSERT_EQ(x.size(), y.size());
      for (std::size_t i = 0; i < x.size(); ++i) {
        scale = std::max(scale, std::abs(x[i]));
        diff = std::max(diff, std::abs(x[i] - y[i]));
      }
    }
  };
  compare(serial.spheroidal, threaded.spheroidal);
  compare(serial.toroidal, threaded.toroidal);
  EXPECT_GT(scale, 0.0);
  EXPECT_LE(diff, 1e-10 * scale);
}
