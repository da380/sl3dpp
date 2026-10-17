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
#include <limits>
#include <map>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/ToroidalOperator.hpp"
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

  // At the GLL nodes, whose (element, node) layout the grid-state check
  // below reads directly; TestGaussMemory3D is the Gauss-point gate.
  ViscoelasticOperator op1(body.rheology, kLmax,
                           {.memory_points = MemoryPointSet::Gll});
  Viscoelastic3D op3(body.rheology, kLmax,
                     [](double, double, double) { return 0.0; },
                     {.memory_points = MemoryPointSet::Gll});
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
  // Options::parallel threads the transforms, the per-degree loops (force
  // assembly, solves, strain recovery) and the step's own loops (weights
  // and decay, the residual, the mixing, the memory update). The degree
  // and element-wise loops are bitwise independent of the thread count;
  // the threaded transforms are not (each thread plans its own FFTs), nor
  // are the mixer's partial sums, so the comparison is at the iteration's
  // tolerance, not bitwise.
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

TEST(Stage4, FinishingAMovedStepTakesItsBuffers) {
  // A step that is done with finishes into its own buffers (the grid
  // workspace becomes the strain, the decayed memory advances in place);
  // the state is bitwise the copying Finish's.
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto op = Viscoelastic3D(
      body.rheology, kLmax, [](double, double theta, double phi) {
        return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
      });
  const auto load = [](int l, double) { return LoadColumns(l); };
  auto state =
      op.Step(op.ZeroState(), tau, Scheme::ExponentialTrapezoid, load).state;
  auto step = op.StartStep(state, tau, Scheme::ExponentialTrapezoid, load);
  const auto [strain, info] = step.Converge(
      [&](const GridCoupling::SpectralTensor& s) {
        auto solutions = step.Respond(s);
        step.AddChannels(solutions, load, {});
        return step.StrainOf(solutions);
      },
      {});
  EXPECT_TRUE(info.converged);
  const auto copied = step.Finish(strain);
  const auto moved = std::move(step).Finish(strain);
  ASSERT_EQ(copied.m.size(), moved.m.size());
  auto same = [](const GridCoupling::TensorField& a,
                 const GridCoupling::TensorField& b) {
    return a.c[0] == b.c[0] && a.c[1] == b.c[1] && a.r == b.r;
  };
  for (auto k = std::size_t{0}; k < copied.m.size(); ++k) {
    EXPECT_TRUE(same(copied.m[k], moved.m[k])) << "branch " << k;
  }
  ASSERT_TRUE(copied.strain.has_value() && moved.strain.has_value());
  EXPECT_TRUE(same(*copied.strain, *moved.strain));
  EXPECT_EQ(copied.t, moved.t);
}

TEST(Stage4, AdaptiveMixingEngagesOnlyWhenPicardIsSlow) {
  // The within-step iteration is Picard until a pass contracts by less
  // than anderson_rate. A weak contrast at a short step never gets there,
  // so its step is bitwise plain Picard; a long step does, and Anderson
  // then takes fewer passes than Picard and lands where Anderson from the
  // first pass does.
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto load = [](int l, double) { return LoadColumns(l); };
  const auto amplitude = 0.5 * std::log(3.0);  // a 3x contrast
  const auto delta = [amplitude](double, double theta, double phi) {
    return amplitude * std::sin(theta) * std::cos(phi);
  };
  auto second = [&](double h, int memory, double rate) {
    Viscoelastic3D::Options options;
    options.max_iterations = 200;
    options.anderson_memory = memory;
    options.anderson_rate = rate;
    const auto op = Viscoelastic3D(body.rheology, kLmax, delta, options);
    const auto first =
        op.Step(op.ZeroState(), h * tau, Scheme::ExponentialTrapezoid, load);
    EXPECT_TRUE(first.info.converged);
    return op.Step(first.state, h * tau, Scheme::ExponentialTrapezoid, load);
  };
  auto gap = [](const Viscoelastic3D::Solutions& a,
                const Viscoelastic3D::Solutions& b) {
    auto scale = 0.0, diff = 0.0;
    for (const auto& [l, x] : a.spheroidal) {
      const auto& y = b.spheroidal.at(l);
      for (auto i = std::size_t{0}; i < x.size(); ++i) {
        scale = std::max(scale, std::abs(y[i]));
        diff = std::max(diff, std::abs(x[i] - y[i]));
      }
    }
    return diff / scale;
  };

  const auto weak = second(0.5, 5, 0.5);
  const auto weakPicard = second(0.5, 0, 0.5);
  EXPECT_TRUE(weak.info.converged);
  EXPECT_EQ(weak.info.mixedPasses, 0);
  EXPECT_EQ(weak.info.iterations, weakPicard.info.iterations);
  EXPECT_EQ(gap(weak.solutions, weakPicard.solutions), 0.0);

  const auto slow = second(10.0, 5, 0.5);
  const auto slowPicard = second(10.0, 0, 0.5);
  const auto always = second(10.0, 5, 0.0);
  EXPECT_TRUE(slow.info.converged);
  EXPECT_GT(slow.info.mixedPasses, 0);
  EXPECT_LT(slow.info.iterations, slowPicard.info.iterations);
  EXPECT_LT(gap(slow.solutions, always.solutions), 1e-9);
}

TEST(Stage4, NonFinitePassEndsTheIteration) {
  // The residual's maxima pass over NaNs, so a pass gone non-finite is
  // caught apart: the iteration stops unconverged, with a NaN residual,
  // on the last finite iterate (here the start), instead of reading an
  // all-NaN iterate as a zero change.
  auto body = Body::Make();
  const auto tau = body.rheology.MinRelaxationTime();
  const auto load = [](int l, double) { return LoadColumns(l); };
  const auto op = Viscoelastic3D(body.rheology, kLmax,
                                 [](double, double, double) { return 0.0; });
  const auto step =
      op.StartStep(op.ZeroState(), tau, Scheme::ExponentialTrapezoid, load);
  const auto& s0 = step.StartStrain();
  const auto poisoned = [&](const GridCoupling::SpectralTensor& s) {
    auto out = s;
    out.r[0] = std::numeric_limits<double>::quiet_NaN();
    return out;
  };
  const auto [last, stopped] = step.Converge(poisoned, {});
  EXPECT_FALSE(stopped.converged);
  EXPECT_EQ(stopped.iterations, 1);
  EXPECT_TRUE(std::isnan(stopped.residual));
  EXPECT_EQ(last.r, s0.r);
}

TEST(Stage4, DegreeOneToroidalIsDrivenAndGaugeFree) {
  // Lateral viscosity drives degree-1 toroidal motion, whose rigid
  // rotations are gauge (fixed by pinning one dof per solid region).
  // Moving the pin must leave everything well defined unchanged (the
  // strain, the memory, the spheroidal and l >= 2 toroidal solutions) and
  // change the degree-1 toroidal solution only by rigid rotations
  // (plan.md §4: don't measure things that are gauge).
  auto body = Body::Make();
  const double tau = body.rheology.MinRelaxationTime();
  const auto delta = [](double, double theta, double phi) {
    return 1.15 * std::sin(theta) * std::cos(theta) * std::cos(phi);
  };
  const auto load = [](int l, double) { return LoadColumns(l); };
  auto run = [&](double pinFraction) {
    Viscoelastic3D::Options options;
    options.toroidal_pin_fraction = pinFraction;
    Viscoelastic3D op(body.rheology, kLmax, delta, options);
    auto state = op.ZeroState();
    Viscoelastic3D::StepResult result;
    for (int i = 0; i < 2; ++i) {
      result = op.Step(state, tau, Scheme::ExponentialTrapezoid, load);
      EXPECT_TRUE(result.info.converged);
      state = result.state;
    }
    return result;
  };
  // Pinned at each region's top, and halfway down: in the mantle the
  // latter sits in the viscous, forced layer, so the gauges differ.
  const auto a = run(0.0);
  const auto b = run(0.5);

  double scale = 0.0, diff = 0.0;
  auto compare = [&](const std::vector<Complex>& x,
                     const std::vector<Complex>& y) {
    ASSERT_EQ(x.size(), y.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      scale = std::max(scale, std::abs(x[i]));
      diff = std::max(diff, std::abs(x[i] - y[i]));
    }
  };
  for (const auto& [l, x] : a.solutions.spheroidal) {
    compare(x, b.solutions.spheroidal.at(l));
  }
  for (const auto& [l, x] : a.solutions.toroidal) {
    if (l >= 2) compare(x, b.solutions.toroidal.at(l));
  }
  for (auto c = 0; c < 2; ++c) {
    std::vector<Complex> x(a.state.strain->c[c]), y(b.state.strain->c[c]);
    compare(x, y);
  }
  EXPECT_GT(scale, 0.0);
  EXPECT_LE(diff, 1e-9 * scale);

  // The degree-1 toroidal response is alive, and the two gauges differ by
  // a rigid rotation of each solid region, per column.
  ToroidalOperator probe(body.material, 1);
  const auto& wa = a.solutions.toroidal.at(1);
  const auto& wb = b.solutions.toroidal.at(1);
  const auto ndof = static_cast<std::size_t>(probe.DofCount());
  ASSERT_EQ(wa.size(), 2 * ndof);
  double size = 0.0, gaugeShift = 0.0;
  for (auto v : wa) size = std::max(size, std::abs(v));
  EXPECT_GT(size, 1e-8 * scale);
  for (int k = 0; k < probe.RegionCount(); ++k) {
    const auto rigid = probe.RigidRotation(k);
    for (std::size_t column = 0; column < 2; ++column) {
      Complex dr = 0.0;
      double rr = 0.0;
      for (std::size_t d = 0; d < ndof; ++d) {
        dr += (wa[column * ndof + d] - wb[column * ndof + d]) * rigid[d];
        rr += rigid[d] * rigid[d];
      }
      const auto alpha = dr / rr;
      for (std::size_t d = 0; d < ndof; ++d) {
        gaugeShift = std::max(gaugeShift, std::abs(alpha) * rigid[d]);
      }
      double off = 0.0;
      for (std::size_t d = 0; d < ndof; ++d) {
        if (rigid[d] == 0.0) continue;
        off = std::max(off, std::abs(wa[column * ndof + d] -
                                     wb[column * ndof + d] - alpha * rigid[d]));
      }
      EXPECT_LE(off, 1e-8 * size) << "region " << k << ", column " << column;
    }
  }
  // Not vacuous: the two gauges really do differ by a rigid rotation.
  EXPECT_GT(gaugeShift, 1e-3 * size);
}
