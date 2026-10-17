// Phase timings of the 3D lateral-viscosity step: the baseline for the
// performance work (docs/walkthrough-3d-step.md §4 and §7), to be run
// before and after each change and on the target machine.
//
// The physical setup is fortran_gate's (PREM, the Y_{2,+-2} log-viscosity
// pattern, the constant zonal load rate, 50 yr trapezoid steps). Warm-up
// steps build the lazy operator caches; then each phase of one pass is
// timed on its own, through the public pieces, best of `repeats`:
//
//   analyse     GridCoupling::Analyse of a full tensor (3 batched SHTs)
//   synthesise  GridCoupling::Synthesise of a full spectral tensor
//   force       SpheroidalForce + ToroidalForce, every degree
//   solves      the banded solves, 2(l + 1) real columns per degree
//   strain      AddStrainOfSolution, every degree
//   respond     OpenStep::Respond (pointwise + analyse + force + solves)
//   channels    OpenStep::AddChannels
//   strainOf    OpenStep::StrainOf (strain + synthesise)
//
// and then whole Viscoelastic3D::Step calls, whose per-pass time less
// respond + channels + strainOf is the iteration's own overhead (the
// residual, the copies and the mixing). Peak RSS closes the report.
//
// Usage:
//   step_timing <deck> <lmax> [steps] [repeats] [loop|matrix]
//
// Threading follows fortran_gate: the transforms run threaded when
// OMP_NUM_THREADS > 1; everything else in the step is serial today.

#include <omp.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/DegreeOperator.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/ToroidalOperator.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using Complex = std::complex<double>;

namespace {

double Seconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

// The best of `repeats` timings of f, in seconds.
double Best(int repeats, const std::function<void()>& f) {
  double best = 0.0;
  for (int i = 0; i < repeats; ++i) {
    const auto start = std::chrono::steady_clock::now();
    f();
    const auto t = Seconds(start);
    best = i == 0 ? t : std::min(best, t);
  }
  return best;
}

double PeakRssGb() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <deck> <lmax> [steps] [repeats] [loop|matrix]\n",
                 argv[0]);
    return 1;
  }
  const std::string deckPath = argv[1];
  const int lmax = std::atoi(argv[2]);
  const int steps = argc > 3 ? std::atoi(argv[3]) : 3;
  const int repeats = argc > 4 ? std::atoi(argv[4]) : 3;
  const bool matrix = argc > 5 && std::string(argv[5]) == "matrix";
  const int threads = omp_get_max_threads();
  constexpr double year = 365.25 * 24.0 * 3600.0;

  auto deck = Deck::ReadLegacySl3d(deckPath).value();
  auto model =
      EarthModel::FromDeck(deck, {.spline = SplineKind::Natural}).value();
  auto mesh = RadialMesh::CreateGraded(model, lmax).value();
  auto material = Material::Create(model, mesh).value();
  auto rheology = Rheology::FromModel(material, model).value();
  const auto& scales = model.ModelScales();

  const double c22 = 0.25 * std::sqrt(15.0 / (2.0 * std::numbers::pi));
  auto delta = [c22](double, double theta, double phi) {
    return -0.6 * c22 * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi);
  };
  Viscoelastic3D::Options options;
  options.parallel = threads > 1;
  options.matrix_kernel = matrix;
  options.rtol = 1e-10;

  auto start = std::chrono::steady_clock::now();
  Viscoelastic3D op(rheology, lmax, delta, options);
  const double construction = Seconds(start);

  const double sigmaFactor = scales.Factor({.mass = 1, .length = -2});
  const double timeFactor = scales.Time();
  std::vector<double> rate(lmax + 1, 0.0);
  for (int l = 1; l <= lmax; ++l) {
    rate[l] = std::exp(-2.0 * std::numbers::pi * (l + 1) / 10.5) *
              std::sqrt((2.0 * l + 1) / (4.0 * std::numbers::pi)) * 5515.0 /
              (30.0 * year);
  }
  auto load = [&](int l, double t) -> std::optional<std::vector<Complex>> {
    if (l < 1) return std::nullopt;
    std::vector<Complex> columns(l + 1, 0.0);
    columns[0] = rate[l] * (t * timeFactor) / sigmaFactor;
    return columns;
  };
  const double dt = 50.0 * year / timeFactor;
  const auto scheme = Scheme::ExponentialTrapezoid;

  const auto& coupling = op.Coupling();
  const auto& grid = coupling.Grid();
  const auto nodes = coupling.NodeCount();
  const double tensorGb = static_cast<double>(nodes) *
                          static_cast<double>(grid.FieldSize()) * 40.0 / 1e9;
  std::printf("step_timing: lmax %d, %d thread%s, %s kernel\n", lmax, threads,
              threads == 1 ? "" : "s", matrix ? "matrix" : "loop");
  std::printf("  memory nodes %d, grid %d x %d, one TensorField %.2f GB\n",
              nodes, grid.NumberOfCoLatitudes(), grid.NumberOfLongitudes(),
              tensorGb);
  std::printf("  construction %.3f s\n", construction);
  std::fflush(stdout);

  // Warm-up: the first steps build and factorise every operator.
  auto state = op.ZeroState();
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < 2; ++i) {
    state = op.Step(state, dt, scheme, load).state;
  }
  std::printf("  warm-up (2 steps) %.3f s\n\n", Seconds(start));
  std::fflush(stdout);

  // The phases in isolation, on the warmed state's opened step.
  auto step = op.StartStep(state, dt, scheme, load);
  const auto& d = step.StartStrain();

  // Unrelaxed operators of the same sizes and dof maps as the step's own,
  // for the force/solve/strain phases the operator keeps private.
  const auto& base = rheology.BaseMaterial();
  std::map<int, std::unique_ptr<DegreeOperator>> sph;
  std::map<int, std::unique_ptr<ToroidalOperator>> tor;
  for (int l = 0; l <= lmax; ++l) {
    sph.emplace(l, std::make_unique<DegreeOperator>(base, l));
    if (l >= 2) tor.emplace(l, std::make_unique<ToroidalOperator>(base, l));
  }
  auto sphOf = [&](int l) -> const DegreeOperator& { return *sph.at(l); };
  auto torOf = [&](int l) -> const ToroidalOperator& { return *tor.at(l); };

  GridCoupling::SpectralTensor spectral;
  const double analyse = Best(repeats, [&] { spectral = coupling.Analyse(d); });
  const double synthesise =
      Best(repeats, [&] { auto field = coupling.Synthesise(spectral); });

  std::map<int, std::vector<Complex>> sphForce, torForce;
  const double force = Best(repeats, [&] {
    sphForce = coupling.SpheroidalForce(spectral, sphOf);
    torForce = coupling.ToroidalForce(spectral, torOf);
  });

  // Real-pair right-hand sides, laid out as Viscoelastic3D::SolveAll does.
  auto realPairs = [](const std::vector<Complex>& b, int ndof, int columns) {
    std::vector<double> rhs(static_cast<std::size_t>(ndof) * 2 * columns);
    for (int c = 0; c < columns; ++c) {
      for (int i = 0; i < ndof; ++i) {
        const auto at = static_cast<std::size_t>(c) * ndof + i;
        rhs[at] = b[at].real();
        rhs[static_cast<std::size_t>(columns + c) * ndof + i] = b[at].imag();
      }
    }
    return rhs;
  };
  std::map<int, std::vector<double>> sphRhs, torRhs;
  for (const auto& [l, b] : sphForce) {
    sphRhs.emplace(l, realPairs(b, sphOf(l).DofCount(), l + 1));
  }
  for (const auto& [l, b] : torForce) {
    torRhs.emplace(l, realPairs(b, torOf(l).DofCount(), l + 1));
  }
  const double solves = Best(repeats, [&] {
    for (const auto& [l, rhs] : sphRhs) {
      auto x = sphOf(l).Solve(rhs, 2 * (l + 1));
    }
    for (const auto& [l, rhs] : torRhs) {
      auto x = torOf(l).Solve(rhs, 2 * (l + 1));
    }
  });

  // The strain phase reads solution columns; their values do not matter
  // to the cost, so zeros of the right sizes stand in.
  std::map<int, std::vector<Complex>> sphX, torX;
  for (int l = 0; l <= lmax; ++l) {
    sphX.emplace(
        l, std::vector<Complex>(static_cast<std::size_t>(sphOf(l).DofCount()) *
                                (l + 1)));
    if (l >= 2) {
      torX.emplace(
          l, std::vector<Complex>(
                 static_cast<std::size_t>(torOf(l).DofCount()) * (l + 1)));
    }
  }
  const double strain = Best(repeats, [&] {
    auto out = coupling.ZeroSpectral();
    for (int l = 0; l <= lmax; ++l) {
      const auto found = torX.find(l);
      coupling.AddStrainOfSolution(
          out, l, sphOf(l), sphX.at(l),
          found == torX.end() ? nullptr : &torOf(l),
          found == torX.end() ? std::span<const Complex>{}
                              : std::span<const Complex>(found->second));
    }
  });

  Viscoelastic3D::Solutions solutions;
  const double respond = Best(repeats, [&] { solutions = step.Respond(d); });
  const double channels = Best(repeats, [&] {
    auto copy = solutions;
    step.AddChannels(copy, load, {});
  });
  const double strainOf =
      Best(repeats, [&] { auto field = step.StrainOf(solutions); });

  std::printf("  one pass, phase by phase (best of %d) [s]\n", repeats);
  std::printf("    %-11s %9.4f\n", "analyse", analyse);
  std::printf("    %-11s %9.4f\n", "synthesise", synthesise);
  std::printf("    %-11s %9.4f\n", "force", force);
  std::printf("    %-11s %9.4f\n", "solves", solves);
  std::printf("    %-11s %9.4f\n", "strain", strain);
  std::printf("    %-11s %9.4f  (pointwise + analyse + force + solves)\n",
              "respond", respond);
  std::printf("    %-11s %9.4f\n", "channels", channels);
  std::printf("    %-11s %9.4f  (strain + synthesise)\n", "strainOf", strainOf);
  std::printf("    %-11s %9.4f  (respond less its parts)\n", "pointwise",
              respond - analyse - force - solves);
  std::fflush(stdout);

  // Whole steps: per-pass time and the iteration's own overhead.
  int passes = 0;
  start = std::chrono::steady_clock::now();
  std::printf("\n  %d whole steps:", steps);
  for (int i = 0; i < steps; ++i) {
    auto result = op.Step(state, dt, scheme, load);
    passes += result.info.iterations;
    std::printf(" %d%s", result.info.iterations,
                result.info.converged ? "" : "(NOT converged)");
    state = std::move(result.state);
  }
  const double total = Seconds(start);
  const double perPass = passes > 0 ? total / passes : 0.0;
  std::printf(" passes\n");
  std::printf("    %-11s %9.4f\n", "total", total);
  std::printf("    %-11s %9.4f\n", "per pass", perPass);
  std::printf("    %-11s %9.4f  (per pass less respond, channels, strainOf)\n",
              "overhead", perPass - respond - channels - strainOf);
  std::printf("\n  peak RSS %.2f GB\n", PeakRssGb());
  return 0;
}
