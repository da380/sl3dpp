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
//   pointwise   GridCoupling::Multiply of a full tensor by a weight field
//   force       SpheroidalForce + ToroidalForce, every degree
//   respond     OpenStep::Respond (synthesise + pointwise + analyse +
//               force + solves)
//   solves      the banded solves, 2(l + 1) real columns per degree:
//               respond less its other parts (SolveAll is private)
//   channels    OpenStep::AddChannels
//   strainOf    OpenStep::StrainOf (strain recovery, every degree)
//
// and then whole Viscoelastic3D::Step calls, whose per-pass time less
// respond + channels + strainOf is the iteration's own overhead (the
// residual, the copies and the mixing). The last step's surface sums (to
// full precision), a fingerprint of the final state (equal before and
// after a pure refactor) and the peak RSS close the report.
//
// Usage:
//   step_timing <deck> <lmax> [steps] [repeats] [loop|matrix]
//
// Options::parallel is set when OMP_NUM_THREADS > 1, as in fortran_gate:
// the transforms and the per-degree loops (force, solves, strain) then run
// threaded; the pointwise loops and the mixing stay serial. The
// fingerprint is reproducible run to run on one thread; on several,
// GSHTrans's per-thread FFTW plans (FFTW_MEASURE) can differ at round-off.

#include <omp.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
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

// FNV-1a over the bytes of a state's tensors: equal fingerprints mean a
// bitwise-identical state, which is what a pure refactor must leave.
std::uint64_t Fingerprint(const Viscoelastic3D::State& state) {
  std::uint64_t hash = 1469598103934665603ULL;
  auto add = [&](const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
      hash = (hash ^ p[i]) * 1099511628211ULL;
    }
  };
  auto addTensor = [&](const GridCoupling::TensorField& tensor) {
    for (const auto& component : tensor.c) {
      add(component.data(), component.size() * sizeof(Complex));
    }
    add(tensor.r.data(), tensor.r.size() * sizeof(double));
  };
  for (const auto& m : state.m) addTensor(m);
  if (state.strain) addTensor(*state.strain);
  return hash;
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
  const auto& s0 = step.StartStrain();
  const auto d = coupling.Synthesise(s0);

  // Unrelaxed operators of the same sizes and dof maps as the step's own,
  // for the force phase (the step keeps its own operators private).
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

  auto field = d;
  const std::vector<double> weight(field.r.size(), 1.0);
  const double pointwise =
      Best(repeats, [&] { coupling.Multiply(field, weight); });

  Viscoelastic3D::Solutions solutions;
  const double respond = Best(repeats, [&] { solutions = step.Respond(s0); });
  const double channels = Best(repeats, [&] {
    auto copy = solutions;
    step.AddChannels(copy, load, {});
  });
  const double strainOf =
      Best(repeats, [&] { auto out = step.StrainOf(solutions); });

  const double solves = respond - synthesise - pointwise - analyse - force;
  std::printf("  one pass, phase by phase (best of %d) [s]\n", repeats);
  std::printf("    %-11s %9.4f\n", "analyse", analyse);
  std::printf("    %-11s %9.4f\n", "synthesise", synthesise);
  std::printf("    %-11s %9.4f\n", "pointwise", pointwise);
  std::printf("    %-11s %9.4f\n", "force", force);
  std::printf("    %-11s %9.4f  (respond less its other parts)\n", "solves",
              solves);
  std::printf(
      "    %-11s %9.4f  (synthesise + pointwise + analyse + force + "
      "solves)\n",
      "respond", respond);
  std::printf("    %-11s %9.4f\n", "channels", channels);
  std::printf("    %-11s %9.4f  (strain recovery)\n", "strainOf", strainOf);
  std::fflush(stdout);

  // Whole steps: per-pass time and the iteration's own overhead.
  int passes = 0;
  start = std::chrono::steady_clock::now();
  std::printf("\n  %d whole steps:", steps);
  Viscoelastic3D::Solutions last;
  for (int i = 0; i < steps; ++i) {
    auto result = op.Step(state, dt, scheme, load);
    passes += result.info.iterations;
    std::printf(" %d%s", result.info.iterations,
                result.info.converged ? "" : "(NOT converged)");
    state = std::move(result.state);
    last = std::move(result.solutions);
  }
  const double total = Seconds(start);
  const double perPass = passes > 0 ? total / passes : 0.0;
  std::printf(" passes\n");
  std::printf("    %-11s %9.4f\n", "total", total);
  std::printf("    %-11s %9.4f\n", "per pass", perPass);
  std::printf("    %-11s %9.4f  (per pass less respond, channels, strainOf)\n",
              "overhead", perPass - respond - channels - strainOf);
  // The last step's surface response, to full precision: what a change
  // that is exact only to the iteration tolerance must leave agreeing.
  double sumU = 0.0, sumPhi = 0.0;
  for (const auto& [l, columns] : op.Surface(last, Component::U)) {
    for (const auto& v : columns) sumU += std::abs(v);
  }
  for (const auto& [l, columns] : op.Surface(last, Component::Phi)) {
    for (const auto& v : columns) sumPhi += std::abs(v);
  }
  std::printf("\n  surface sums |U| %.15e  |phi| %.15e\n", sumU, sumPhi);
  std::printf("  state fingerprint %016llx (bitwise check; one thread)\n",
              static_cast<unsigned long long>(Fingerprint(state)));
  std::printf("  peak RSS %.2f GB\n", PeakRssGb());
  return 0;
}
