// The stage-4 cross-code benchmark against the original Fortran
// (docs/stage4.md §5b): replays sl3d's `visco_test` setup through
// Viscoelastic3D and compares surface coefficients, SI, per output time.
//
// The physical setup is visco_test's, fixed: PREM (prem.simp.vm1, natural
// splines), s = s(r) exp(0.3 (Y_{2,-2} + Y_{2,2})), a constant zonal load
// rate sigma_dot_l = exp(-2 pi (l+1)/10.5) sqrt((2l+1)/4 pi) 5515/(30 yr),
// 100 steps of 50 yr with outputs every 20 (t = 1..5 kyr). Only lmax is a
// knob, and it must match the Fortran run that produced the fixture.
//
// Usage:
//   fortran_gate <deck> <fortran-coefficients> <lmax> [threads]
//
// Run via scripts/run_fortran_gate.sh, which builds and drives both codes.

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

using namespace sl3dpp;
using Complex = std::complex<double>;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <deck> <fortran-coefficients> <lmax> [threads]\n",
                 argv[0]);
    return 1;
  }
  const std::string deckPath = argv[1];
  const std::string fixturePath = argv[2];
  const int lmax = std::atoi(argv[3]);
  const bool parallel = argc > 4 && std::atoi(argv[4]) > 1;
  constexpr double year = 365.25 * 24.0 * 3600.0;  // the Fortran's

  auto deck = Deck::ReadLegacySl3d(deckPath).value();
  auto model =
      EarthModel::FromDeck(deck, {.spline = SplineKind::Natural}).value();
  auto mesh = RadialMesh::CreateGraded(model, lmax).value();
  auto material = Material::Create(model, mesh).value();
  auto rheology = Rheology::FromModel(material, model).value();
  const auto& scales = model.ModelScales();

  // The Fortran's lateral pattern; our delta is the log-viscosity
  // perturbation, hence the sign.
  const double c22 = 0.25 * std::sqrt(15.0 / (2.0 * std::numbers::pi));
  auto delta = [c22](double, double theta, double phi) {
    return -0.6 * c22 * std::sin(theta) * std::sin(theta) * std::cos(2.0 * phi);
  };
  Viscoelastic3D::Options options;
  options.parallel = parallel;
  // The matrix kernel stays off until GSHTrans's open issue 8 is fixed:
  // under a threaded BLAS its GEMMs livelock inside the transform's
  // parallel region at production sizes.
  options.rtol = 1e-10;
  Viscoelastic3D op(rheology, lmax, delta, options);

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
    std::vector<Complex> columns(2 * l + 1, 0.0);
    columns[l] = rate[l] * (t * timeFactor) / sigmaFactor;  // m = 0
    return columns;
  };

  const double dt = 50.0 * year / timeFactor;
  const double lengthF = scales.Length();
  const double potF = scales.Factor(PotentialDimensions);
  auto state = op.ZeroState();
  std::map<int, std::map<std::pair<int, int>, Complex>> oursU, oursP;
  int iout = 0;
  for (int it = 2; it <= 101; ++it) {
    auto result = op.Step(state, dt, Scheme::ExponentialTrapezoid, load);
    if (!result.info.converged) {
      std::printf("step %d: NOT converged (residual %.2e)\n", it,
                  result.info.residual);
    }
    state = std::move(result.state);
    if ((it - 1) % 20 == 0) {
      ++iout;
      std::printf("t = %d kyr done (%d iterations last step)\n", iout,
                  result.info.iterations);
      std::fflush(stdout);
      auto U = op.Surface(result.solutions, Component::U);
      auto P = op.Surface(result.solutions, Component::Phi);
      for (int l = 0; l <= lmax; ++l) {
        for (int m = -l; m <= l; ++m) {
          oursU[iout][{l, m}] = U.at(l)[l + m] * lengthF;
          oursP[iout][{l, m}] = P.at(l)[l + m] * potF;
        }
      }
    }
  }

  std::ifstream file(fixturePath);
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", fixturePath.c_str());
    return 1;
  }
  // Degree 1 is reference-frame convention (sl3dpp works in the
  // centre-of-mass frame, the Fortran in its own l = 1 gauge), so the
  // headline comparison is the frame-invariant content l >= 2; the l = 1
  // row is reported separately.
  std::map<int, double> worstU, worstP, scaleU, scaleP, u1ours, u1theirs;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream row(line);
    int io = 0, l = 0, m = 0;
    double ur, ui, pr, pi;
    row >> io >> l >> m >> ur >> ui >> pr >> pi;
    if (!row || io == 0 || l > lmax) continue;
    if (l == 1) {
      if (m == 0) {
        u1ours[io] = oursU[io][{1, 0}].real();
        u1theirs[io] = ur;
      }
      continue;
    }
    const Complex fu(ur, ui), fp(pr, pi);
    scaleU[io] = std::max(scaleU[io], std::abs(fu));
    scaleP[io] = std::max(scaleP[io], std::abs(fp));
    worstU[io] = std::max(worstU[io], std::abs(oursU[io][{l, m}] - fu));
    worstP[io] = std::max(worstP[io], std::abs(oursP[io][{l, m}] - fp));
  }
  std::printf(
      "\nsl3dpp (trapezoid, graded mesh) vs sl3d Fortran "
      "(RK2 rate form), lmax %d, degrees >= 2:\n",
      lmax);
  std::printf("%8s %14s %14s %12s %28s\n", "t [kyr]", "U rel diff",
              "phi rel diff", "U scale [m]", "U(1,0) ours/theirs [frame]");
  for (const auto& [io, w] : worstU) {
    std::printf("%8d %14.4e %14.4e %12.4e %13.4e %13.4e\n", io, w / scaleU[io],
                worstP[io] / scaleP[io], scaleU[io], u1ours[io], u1theirs[io]);
  }
  return 0;
}
