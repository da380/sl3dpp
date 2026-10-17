#include "sl3dpp/Fingerprint.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace sl3dpp {

Fingerprint::Fingerprint(int lmax, const LoveNumbers& table, Options options)
    : solver_(SurfaceGrid(lmax, {.parallel = options.parallel_transforms}),
              table, options.parameters),
      rtol_(options.rtol),
      andersonMemory_(options.anderson_memory) {}

void Fingerprint::SetState(std::span<const double> iceThickness,
                           std::span<const double> seaLevel,
                           std::span<const char> forcedLand) {
  state_ = solver_.MakeState(
      ToDimensionless(iceThickness, Dimensions{.length = 1}),
      ToDimensionless(seaLevel, Dimensions{.length = 1}), forcedLand);
}

void Fingerprint::SetAnalyticTestState(double date) {
  // pyslfp's AnalyticalIceModel, verbatim: smooth super-Gaussian features
  // in SI metres, with the ice masked onto land by a sigmoid of the
  // topography and thresholded at a millimetre.
  auto feature = [](double theta, double phi, double peak, double colat0,
                    double lon0, double width, int power) {
    const double cosDistance =
        std::cos(theta) * std::cos(colat0) +
        std::sin(theta) * std::sin(colat0) * std::cos(phi - lon0);
    const auto distance = std::acos(std::clamp(cosDistance, -1.0, 1.0));
    return peak * std::exp(-0.5 * std::pow(distance / width, power));
  };
  constexpr auto kPi = std::numbers::pi;
  const auto& grid = solver_.Grid();
  auto seaLevel = grid.FromFunction([&](double theta, double phi) {
    auto sl = 4000.0;
    sl -= feature(theta, phi, 6000.0, 0.0, 0.0, 0.35, 6);
    sl -= feature(theta, phi, 5000.0, kPi / 2, kPi / 2, 0.45, 4);
    sl -= feature(theta, phi, 4050.0, 3 * kPi / 4, 3 * kPi / 2, 1.5, 4);
    return sl;
  });
  auto ice = grid.FromFunction([&](double theta, double phi) {
    auto thickness = feature(theta, phi, 3000.0, 0.0, 0.0, 0.2, 2);
    if (date > 0.0) {
      const auto factor = std::min(date / 21.0, 1.0);
      thickness +=
          feature(theta, phi, 2500.0 * factor, kPi / 4, kPi / 2, 0.15, 2);
    }
    return thickness;
  });
  for (auto i = std::size_t{0}; i < ice.size(); ++i) {
    const auto landMask = 1.0 / (1.0 + std::exp(seaLevel[i] / 100.0));
    ice[i] *= landMask;
    if (ice[i] < 1e-3) ice[i] = 0.0;
  }
  SetState(ice, seaLevel);
}

std::vector<double> Fingerprint::ToDimensionless(std::span<const double> field,
                                                 Dimensions dimensions) const {
  if (field.size() != solver_.Grid().FieldSize()) {
    throw std::invalid_argument("the field does not live on the grid");
  }
  const auto factor = solver_.SolverScales().Factor(dimensions);
  auto out = std::vector<double>(field.begin(), field.end());
  for (auto& v : out) v /= factor;
  return out;
}

std::vector<double> Fingerprint::IceThickness() const {
  std::vector<double> out = state_.iceThickness;
  const auto length = solver_.SolverScales().Length();
  for (auto& v : out) v *= length;
  return out;
}

std::vector<double> Fingerprint::SeaLevel() const {
  std::vector<double> out = state_.seaLevel;
  const auto length = solver_.SolverScales().Length();
  for (auto& v : out) v *= length;
  return out;
}

std::vector<double> Fingerprint::LoadFromIceThicknessChange(
    std::span<const double> iceThicknessChange) const {
  if (iceThicknessChange.size() != solver_.Grid().FieldSize()) {
    throw std::invalid_argument("the field does not live on the grid");
  }
  const auto& scales = solver_.SolverScales();
  // Non-dimensional rho_ice (1 - C) Delta I, redimensionalised to kg m^-2.
  const double sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto length = scales.Length();
  auto out = std::vector<double>(iceThicknessChange.size());
  for (auto i = std::size_t{0}; i < out.size(); ++i) {
    out[i] = solver_.IceDensity() * (1.0 - state_.oceanFunction[i]) *
             (iceThicknessChange[i] / length) * sigma;
  }
  return out;
}

std::vector<double> Fingerprint::LoadFromSeaLevelChange(
    std::span<const double> seaLevelChange) const {
  if (seaLevelChange.size() != solver_.Grid().FieldSize()) {
    throw std::invalid_argument("the field does not live on the grid");
  }
  const auto& scales = solver_.SolverScales();
  const double sigma = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto length = scales.Length();
  auto out = std::vector<double>(seaLevelChange.size());
  for (auto i = std::size_t{0}; i < out.size(); ++i) {
    out[i] = solver_.WaterDensity() * state_.oceanFunction[i] *
             (seaLevelChange[i] / length) * sigma;
  }
  return out;
}

Fingerprint::Result Fingerprint::FromResponse(
    SeaLevelSolver::Response response) const {
  const auto& scales = solver_.SolverScales();
  const auto length = scales.Length();
  const auto potential = scales.Factor(PotentialDimensions);
  const double frequency = scales.Factor(Dimensions{.time = -1});
  Result out;
  out.seaLevelChange = std::move(response.seaLevelChange);
  out.displacement = std::move(response.displacement);
  out.potential = std::move(response.potential);
  for (auto& v : out.seaLevelChange) v *= length;
  for (auto& v : out.displacement) v *= length;
  for (auto& v : out.potential) v *= potential;
  for (auto i = 0; i < 3; ++i) {
    out.angularVelocity[i] = response.angularVelocity[i] * frequency;
  }
  out.iterations = response.iterations;
  out.error = response.error;
  out.converged = response.converged;
  return out;
}

Fingerprint::Result Fingerprint::Solve(std::span<const double> directLoad,
                                       RotationalFeedback rotation) const {
  if (state_.oceanFunction.empty()) {
    throw std::logic_error("set a background state before solving");
  }
  SeaLevelSolver::Options options;
  options.rtol = rtol_;
  options.anderson_memory = andersonMemory_;
  return FromResponse(solver_.Solve(
      state_, ToDimensionless(directLoad, Dimensions{.mass = 1, .length = -2}),
      rotation, options));
}

Fingerprint::Result Fingerprint::SolveForIceChange(
    std::span<const double> iceThicknessChange,
    RotationalFeedback rotation) const {
  return Solve(LoadFromIceThicknessChange(iceThicknessChange), rotation);
}

Fingerprint::Result Fingerprint::SolveNonlinear(
    std::span<const double> iceThicknessChange,
    RotationalFeedback rotation) const {
  if (state_.oceanFunction.empty()) {
    throw std::logic_error("set a background state before solving");
  }
  SeaLevelSolver::Options options;
  options.rtol = rtol_;
  options.max_iterations = 50;
  options.anderson_memory = andersonMemory_;
  SeaLevelSolver::NonlinearForcing forcing;
  forcing.iceThicknessChange =
      ToDimensionless(iceThicknessChange, Dimensions{.length = 1});
  forcing.rotation = rotation;
  return FromResponse(
      solver_.SolveNonlinear(state_, forcing, options).response);
}

}  // namespace sl3dpp
