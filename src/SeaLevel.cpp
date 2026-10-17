#include "sl3dpp/SeaLevel.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <optional>
#include <stdexcept>

#include "sl3dpp/ShConventions.hpp"
#include "sl3dpp/detail/AndersonMixer.hpp"

namespace sl3dpp {

namespace {

constexpr auto kPi = std::numbers::pi;

using Block2 = std::array<std::array<double, 2>, 2>;  // (cos/sin, m = 0/1)

Block2 operator*(double s, const Block2& b) {
  return {{{s * b[0][0], s * b[0][1]}, {s * b[1][0], s * b[1][1]}}};
}
Block2 operator+(const Block2& x, const Block2& y) {
  return {{{x[0][0] + y[0][0], x[0][1] + y[0][1]},
           {x[1][0] + y[1][0], x[1][1] + y[1][1]}}};
}

/// The degree-2, orders 0..1 block of a complex coefficient vector, as the
/// real (cos, sin) pairs of the pyshtools basis the rotation algebra is
/// written in.
Block2 BlockOf(std::span<const std::complex<double>> c) {
  const auto pair = RealPairFromComplex(c[SurfaceGrid::Index(2, 1)], 1);
  return {{{c[SurfaceGrid::Index(2, 0)].real(), pair.cos}, {0.0, pair.sin}}};
}

void SetBlock(std::span<std::complex<double>> c, const Block2& block) {
  c[SurfaceGrid::Index(2, 0)] = block[0][0];
  c[SurfaceGrid::Index(2, 1)] =
      ComplexFromRealPair({block[0][1], block[1][1]}, 1);
}

void AddBlock(std::span<std::complex<double>> c, const Block2& block) {
  c[SurfaceGrid::Index(2, 0)] += block[0][0];
  c[SurfaceGrid::Index(2, 1)] +=
      ComplexFromRealPair({block[0][1], block[1][1]}, 1);
}

double MaxAbs(std::span<const double> v) {
  auto out = 0.0;
  for (auto x : v) out = std::max(out, std::abs(x));
  return out;
}

double MaxAbs(std::span<const std::complex<double>> v) {
  auto out = 0.0;
  for (auto x : v) out = std::max(out, std::abs(x));
  return out;
}

std::span<const double> AsReal(std::span<const std::complex<double>> v) {
  return {reinterpret_cast<const double*>(v.data()), 2 * v.size()};
}

std::vector<std::complex<double>> FromReal(std::span<const double> v) {
  auto out = std::vector<std::complex<double>>(v.size() / 2);
  for (auto i = std::size_t{0}; i < out.size(); ++i) {
    out[i] = {v[2 * i], v[2 * i + 1]};
  }
  return out;
}

}  // namespace

SeaLevelParameters WithHydrostaticMoments(SeaLevelParameters parameters,
                                          double fluid_tidal_number,
                                          double non_hydrostatic_fraction) {
  if (fluid_tidal_number <= 0.0) {
    throw std::invalid_argument("the fluid tidal number must be positive");
  }
  const double oblateness = (1.0 + non_hydrostatic_fraction) *
                            fluid_tidal_number * parameters.rotation_frequency *
                            parameters.rotation_frequency *
                            std::pow(parameters.mean_sea_floor_radius, 5) /
                            (3.0 * parameters.gravitational_constant);
  parameters.equatorial_moment_of_inertia =
      parameters.polar_moment_of_inertia - oblateness;
  return parameters;
}

SeaLevelSolver::SeaLevelSolver(SurfaceGrid grid, const LoveNumbers& table,
                               SeaLevelParameters parameters)
    : grid_(std::move(grid)), parameters_(parameters) {
  const auto lmax = grid_.LMax();
  if (static_cast<int>(table.degrees.size()) < lmax + 1) {
    throw std::invalid_argument(
        std::format("the Love table reaches degree {}, the grid needs {}",
                    table.degrees.empty() ? -1 : table.degrees.back(), lmax));
  }

  // The body of the parameters is the Love table's (pyslfp's with_body).
  parameters.mean_sea_floor_radius = table.radius;
  parameters.gravitational_acceleration = table.surface_gravity;

  // The one conversion to the internal non-dimensional scheme.
  scales_ = Scales::Geophysical(parameters.mean_radius);
  const auto nd = table.ConvertedTo(scales_);
  auto truncate = [lmax](const std::vector<double>& v) {
    return std::vector<double>(v.begin(), v.begin() + lmax + 1);
  };
  hU_ = truncate(nd.h_u);
  kU_ = truncate(nd.k_u);
  hPhi_ = truncate(nd.h_phi);
  kPhi_ = truncate(nd.k_phi);
  h_.resize(lmax + 1);
  k_.resize(lmax + 1);
  for (auto l = 0; l <= lmax; ++l) {
    h_[l] = hU_[l] + hPhi_[l];
    k_[l] = kU_[l] + kPhi_[l];
  }
  h2_ = h_[2];
  k2_ = k_[2];
  ht2_ = nd.h_t[2];
  kt2_ = nd.k_t[2];
  hc_ = nd.h_c;
  kc_ = nd.k_c;
  mu_ = nd.m_u;
  mphi_ = nd.m_phi;
  mc_ = nd.m_c;

  g_ = parameters.gravitational_acceleration /
       scales_.Factor(AccelerationDimensions);
  waterDensity_ = parameters.water_density / scales_.Factor(DensityDimensions);
  iceDensity_ = parameters.ice_density / scales_.Factor(DensityDimensions);
  radius_ = parameters.mean_sea_floor_radius / scales_.Length();
  const double G = parameters.gravitational_constant /
                   scales_.Factor(GravitationalConstantDimensions);
  const Dimensions inertiaDims{.mass = 1, .length = 2};
  const double polar =
      parameters.polar_moment_of_inertia / scales_.Factor(inertiaDims);
  const double equatorial =
      parameters.equatorial_moment_of_inertia / scales_.Factor(inertiaDims);
  const double omega =
      parameters.rotation_frequency / scales_.Factor(Dimensions{.time = -1});

  slcFactor_.resize(lmax + 1);
  for (auto l = 0; l <= lmax; ++l) slcFactor_[l] = -(h_[l] + k_[l] / g_);

  // The rotational feedback factors, pyslfp's `EarthModelParameters`
  // verbatim (there in the pyshtools real basis, which is where our
  // rotation block runs too).
  const double a = radius_;
  rotationFactor_ = std::sqrt(4.0 * kPi / 15.0) * omega * a * a;
  inertiaFactor_ = std::sqrt(5.0 / (12.0 * kPi)) * omega * a * a * a /
                   (G * (polar - equatorial));
  axialRotationFactor_ = std::sqrt(16.0 * kPi / 45.0) * omega * a * a;
  uniformRotationFactor_ = -std::sqrt(16.0 * kPi / 9.0) * omega * a * a;
  axialInertiaFactor_ =
      std::sqrt(5.0 / (9.0 * kPi)) * omega * a * a * a / (G * polar);
  inverseInertiaDifference_ = 1.0 / (polar - equatorial);
  inversePolarInertia_ = 1.0 / polar;
  traceFactor_ = (4.0 / 3.0) * omega / polar;
  shellFactor_ =
      (2.0 / 3.0) * omega * std::pow(a, 4) * std::sqrt(4.0 * kPi) / polar;
  rotationDenominator_ = 1.0 - inertiaFactor_ * kt2_ * rotationFactor_;
  axialRotationDenominator_ =
      1.0 + axialInertiaFactor_ * kt2_ * axialRotationFactor_ +
      traceFactor_ * mc_ * uniformRotationFactor_;
}

std::vector<std::complex<double>> SeaLevelSolver::CentrifugalCoefficients(
    const std::array<double, 3>& omega) const {
  auto psi = grid_.ZeroCoefficients();
  psi[SurfaceGrid::Index(2, 1)] = ComplexFromRealPair(
      {rotationFactor_ * omega[0], rotationFactor_ * omega[1]}, 1);
  psi[SurfaceGrid::Index(2, 0)] = axialRotationFactor_ * omega[2];
  psi[SurfaceGrid::Index(0, 0)] = uniformRotationFactor_ * omega[2];
  return psi;
}

std::array<double, 3> SeaLevelSolver::CentrifugalDual(
    std::span<const std::complex<double>> tideDual) const {
  // Pair(a, b) = sum_l a_l0 b_l0 + 2 sum_{m>0} Re(a conj b) equals the
  // plain real-pair pairing under RealPairFromComplex, so the transpose
  // reads the (cos, sin) parts of the degree-2 dual directly.
  const auto pair = RealPairFromComplex(tideDual[SurfaceGrid::Index(2, 1)], 1);
  return {rotationFactor_ * pair.cos, rotationFactor_ * pair.sin,
          axialRotationFactor_ * tideDual[SurfaceGrid::Index(2, 0)].real() +
              uniformRotationFactor_ * tideDual[SurfaceGrid::Index(0, 0)].real()};
}

double SeaLevelSolver::IntegrateSurface(std::span<const double> field) const {
  return radius_ * radius_ * grid_.Integrate(field);
}

SeaLevelSolver::State SeaLevelSolver::MakeState(
    std::vector<double> iceThickness, std::vector<double> seaLevel,
    std::span<const char> forcedLand) const {
  if (iceThickness.size() != grid_.FieldSize() ||
      seaLevel.size() != grid_.FieldSize()) {
    throw std::invalid_argument("state fields must live on the grid");
  }
  State state;
  state.oceanFunction.resize(grid_.FieldSize());
  for (auto i = std::size_t{0}; i < state.oceanFunction.size(); ++i) {
    const bool floats =
        waterDensity_ * seaLevel[i] - iceDensity_ * iceThickness[i] > 0.0;
    const auto land = !forcedLand.empty() && forcedLand[i] != 0;
    state.oceanFunction[i] = (floats && !land) ? 1.0 : 0.0;
  }
  state.iceThickness = std::move(iceThickness);
  state.seaLevel = std::move(seaLevel);
  state.oceanArea = IntegrateSurface(state.oceanFunction);
  return state;
}

// -- the rotation block
// ----------------------------------------------------------

/**
 * The degree-2 / degree-0 rotational feedback, solved exactly within an
 * iteration: pyslfp's `SeaLevelEquation._rotation`, verbatim, on the real
 * (cos, sin) pairs. The transverse components are set by the (2, 1)
 * potential and the axial by (2, 0) (MacCullagh), each feeding back
 * through the degree-2 tidal numbers; the axial component also carries the
 * trace of the inertia perturbation through the axial Love numbers and the
 * uniform shell term.
 */
struct SeaLevelSolver::Rotation {
  std::array<double, 3> omega{0.0, 0.0, 0.0};
  Block2 disp2{};             // full degree-2 displacement (forcing + tidal)
  Block2 grav2{};             // likewise the potential, without the centrifugal
  Block2 centrifugal{};       // the degree-2 centrifugal potential
  double centrifugal0 = 0.0;  // its degree-0 part

  Rotation(const SeaLevelSolver& s, const Block2& load2, double load0,
           const std::optional<Block2>& staticDisp2,
           const std::optional<Block2>& staticGrav2, double displacementLoad0,
           double potentialLoad0, double inertiaMoment,
           const std::array<double, 3>& angularMomentum, bool transverse) {
    Block2 gravForcing = s.k2_ * load2;
    Block2 dispForcing = s.h2_ * load2;
    if (staticGrav2) gravForcing = gravForcing + *staticGrav2;
    if (staticDisp2) dispForcing = dispForcing + *staticDisp2;

    std::array<double, 2> transverseForcing = {
        s.inertiaFactor_ * gravForcing[0][1],
        s.inertiaFactor_ * gravForcing[1][1]};
    double axial = -s.axialInertiaFactor_ * gravForcing[0][0] -
                   s.traceFactor_ *
                       ((s.mu_ + s.mphi_) * load0 + s.mu_ * displacementLoad0 +
                        s.mphi_ * potentialLoad0 + inertiaMoment) -
                   s.shellFactor_ * (load0 + potentialLoad0);
    for (auto i = 0; i < 2; ++i) {
      transverseForcing[i] -= s.inverseInertiaDifference_ * angularMomentum[i];
    }
    axial += s.inversePolarInertia_ * angularMomentum[2];

    if (transverse) {
      omega[0] = transverseForcing[0] / s.rotationDenominator_;
      omega[1] = transverseForcing[1] / s.rotationDenominator_;
    }
    omega[2] = axial / s.axialRotationDenominator_;

    centrifugal[0][1] = s.rotationFactor_ * omega[0];
    centrifugal[1][1] = s.rotationFactor_ * omega[1];
    centrifugal[0][0] = s.axialRotationFactor_ * omega[2];
    centrifugal0 = s.uniformRotationFactor_ * omega[2];
    disp2 = dispForcing + s.ht2_ * centrifugal;
    grav2 = gravForcing + s.kt2_ * centrifugal;
  }
};

// -- the generalised linear solve
// ---------------------------------------------------

SeaLevelSolver::Response SeaLevelSolver::SolveGeneralised(
    const State& state, const GeneralisedForcing& forcing,
    Options options) const {
  const auto fieldSize = grid_.FieldSize();
  const auto coefSize = grid_.CoefficientSize();
  const bool transverse = forcing.rotation == RotationalFeedback::Full;
  const bool rotational = forcing.rotation != RotationalFeedback::None;

  auto checkField = [&](const std::vector<double>& f, const char* name) {
    if (!f.empty() && f.size() != fieldSize) {
      throw std::invalid_argument(
          std::format("{} does not live on the grid", name));
    }
  };
  auto checkCoefficients = [&](const std::vector<std::complex<double>>& c,
                               const char* name) {
    if (!c.empty() && c.size() != coefSize) {
      throw std::invalid_argument(
          std::format("{} is not a coefficient vector of the grid", name));
    }
  };
  checkField(forcing.directLoad, "direct load");
  checkField(forcing.displacementLoad, "displacement load");
  checkField(forcing.potentialLoad, "potential load");
  checkField(forcing.oceanFunctionOverride, "ocean function override");
  checkCoefficients(forcing.displacementResponse, "displacement response");
  checkCoefficients(forcing.potentialResponse, "potential response");

  // The static (load-independent) coefficient terms.
  std::vector<std::complex<double>> staticDisp, staticGrav;
  auto displacementLoad0 = 0.0, potentialLoad0 = 0.0;
  auto anyForcing = false;
  auto ensureStatic = [&] {
    if (staticDisp.empty()) {
      staticDisp.assign(coefSize, 0.0);
      staticGrav.assign(coefSize, 0.0);
    }
  };
  if (!forcing.displacementLoad.empty() &&
      MaxAbs(forcing.displacementLoad) > 0.0) {
    anyForcing = true;
    ensureStatic();
    const auto lm = grid_.Expand(forcing.displacementLoad);
    displacementLoad0 = lm[SurfaceGrid::Index(0, 0)].real();
    for (auto l = 0; l <= grid_.LMax(); ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto i = SurfaceGrid::Index(l, m);
        staticDisp[i] += hU_[l] * lm[i];
        staticGrav[i] += kU_[l] * lm[i];
      }
    }
  }
  if (!forcing.potentialLoad.empty() && MaxAbs(forcing.potentialLoad) > 0.0) {
    anyForcing = true;
    ensureStatic();
    const auto lm = grid_.Expand(forcing.potentialLoad);
    potentialLoad0 = lm[SurfaceGrid::Index(0, 0)].real();
    for (auto l = 0; l <= grid_.LMax(); ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto i = SurfaceGrid::Index(l, m);
        staticDisp[i] += hPhi_[l] * lm[i];
        staticGrav[i] += kPhi_[l] * lm[i];
      }
    }
  }
  if (!forcing.displacementResponse.empty() &&
      MaxAbs(forcing.displacementResponse) > 0.0) {
    anyForcing = true;
    ensureStatic();
    for (auto i = std::size_t{0}; i < coefSize; ++i) {
      staticDisp[i] += forcing.displacementResponse[i];
    }
  }
  if (!forcing.potentialResponse.empty() &&
      MaxAbs(forcing.potentialResponse) > 0.0) {
    anyForcing = true;
    ensureStatic();
    for (auto i = std::size_t{0}; i < coefSize; ++i) {
      staticGrav[i] += forcing.potentialResponse[i];
    }
  }
  anyForcing =
      anyForcing || forcing.inertiaMomentChange != 0.0 ||
      MaxAbs(std::span<const double>(forcing.angularMomentumChange.data(), 3)) >
          0.0 ||
      (!forcing.directLoad.empty() && MaxAbs(forcing.directLoad) > 0.0);

  Response response;
  response.seaLevelChange.assign(fieldSize, 0.0);
  response.displacement.assign(fieldSize, 0.0);
  response.potential.assign(fieldSize, 0.0);
  if (!anyForcing) return response;

  const auto& oceanFunction = forcing.oceanFunctionOverride.empty()
                                  ? state.oceanFunction
                                  : forcing.oceanFunctionOverride;
  const double oceanArea = forcing.oceanFunctionOverride.empty()
                               ? state.oceanArea
                               : IntegrateSurface(oceanFunction);

  auto directData = std::vector<double>(fieldSize, 0.0);
  auto meanSlc = 0.0;
  if (!forcing.directLoad.empty()) {
    directData = forcing.directLoad;
    meanSlc = -IntegrateSurface(directData) / (waterDensity_ * oceanArea);
  }

  auto staticSlc = std::vector<std::complex<double>>{};
  std::optional<Block2> staticDisp2, staticGrav2;
  if (!staticDisp.empty()) {
    staticSlc.resize(coefSize);
    for (auto i = std::size_t{0}; i < coefSize; ++i) {
      staticSlc[i] = -(staticDisp[i] + staticGrav[i] / g_);
    }
    staticDisp2 = BlockOf(staticDisp);
    staticGrav2 = BlockOf(staticGrav);
  }

  // One iteration: the mass-conserving sea level change of a load.
  auto seaLevelFromLoad = [&](std::span<const std::complex<double>> loadLm,
                              std::array<double, 3>& omega) {
    auto slcLm = std::vector<std::complex<double>>(coefSize);
    for (auto l = 0; l <= grid_.LMax(); ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto i = SurfaceGrid::Index(l, m);
        slcLm[i] = slcFactor_[l] * loadLm[i];
      }
    }
    if (!staticSlc.empty()) {
      for (auto i = std::size_t{0}; i < coefSize; ++i) slcLm[i] += staticSlc[i];
    }
    omega = {0.0, 0.0, 0.0};
    if (rotational) {
      const Rotation rotation(*this, BlockOf(loadLm),
                              loadLm[SurfaceGrid::Index(0, 0)].real(),
                              staticDisp2, staticGrav2, displacementLoad0,
                              potentialLoad0, forcing.inertiaMomentChange,
                              forcing.angularMomentumChange, transverse);
      omega = rotation.omega;
      SetBlock(slcLm,
               -1.0 * (rotation.disp2 +
                       (1.0 / g_) * (rotation.grav2 + rotation.centrifugal)));
      // uniform, and so absorbed by mass conservation below
      slcLm[SurfaceGrid::Index(0, 0)] -=
          (hc_ + (kc_ + 1.0) / g_) * rotation.centrifugal0;
    }
    auto slc = grid_.Synthesise(slcLm);
    auto cslc = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      cslc[i] = oceanFunction[i] * slc[i];
    }
    const auto shift = meanSlc - IntegrateSurface(cslc) / oceanArea;
    for (auto& v : slc) v += shift;
    return slc;
  };

  // The load fixed point under Anderson mixing.
  auto seeded = std::vector<double>(fieldSize);
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    seeded[i] = directData[i] + waterDensity_ * oceanFunction[i] * meanSlc;
  }
  auto loadLm = grid_.Expand(seeded);
  detail::AndersonMixer mixer(options.anderson_memory);

  auto oceanSlcPrev = std::vector<double>{};
  auto slc = std::vector<double>{};
  auto omega = std::array<double, 3>{0.0, 0.0, 0.0};
  auto err = std::numeric_limits<double>::infinity();
  auto converged = false;
  auto count = 0;
  while (count < options.max_iterations) {
    slc = seaLevelFromLoad(loadLm, omega);
    auto oceanSlc = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      oceanSlc[i] = oceanFunction[i] * slc[i];
    }
    if (!oceanSlcPrev.empty()) {
      const auto scale = MaxAbs(oceanSlc);
      auto diff = 0.0;
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        diff = std::max(diff, std::abs(oceanSlc[i] - oceanSlcPrev[i]));
      }
      err = scale > 0.0 ? diff / scale : 0.0;
      if (err <= options.rtol) {
        converged = true;
        break;
      }
    }
    oceanSlcPrev = std::move(oceanSlc);
    auto newLoad = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      newLoad[i] = directData[i] + waterDensity_ * oceanSlcPrev[i];
    }
    const auto newLoadLm = grid_.Expand(newLoad);
    loadLm = FromReal(mixer.Update(AsReal(loadLm), AsReal(newLoadLm)));
    ++count;
  }

  // The remaining fields, from the same load as the sea level.
  std::vector<std::complex<double>> dispLm(coefSize), potLm(coefSize);
  for (auto l = 0; l <= grid_.LMax(); ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto i = SurfaceGrid::Index(l, m);
      dispLm[i] = h_[l] * loadLm[i];
      potLm[i] = k_[l] * loadLm[i];
    }
  }
  if (!staticDisp.empty()) {
    for (auto i = std::size_t{0}; i < coefSize; ++i) {
      dispLm[i] += staticDisp[i];
      potLm[i] += staticGrav[i];
    }
  }
  if (rotational) {
    const Rotation rotation(
        *this, BlockOf(loadLm), loadLm[SurfaceGrid::Index(0, 0)].real(),
        staticDisp2, staticGrav2, displacementLoad0, potentialLoad0,
        forcing.inertiaMomentChange, forcing.angularMomentumChange, transverse);
    SetBlock(dispLm, rotation.disp2);
    SetBlock(potLm, rotation.grav2);
    dispLm[SurfaceGrid::Index(0, 0)] += hc_ * rotation.centrifugal0;
    potLm[SurfaceGrid::Index(0, 0)] += kc_ * rotation.centrifugal0;
  }

  response.seaLevelChange = std::move(slc);
  response.displacement = grid_.Synthesise(dispLm);
  response.potential = grid_.Synthesise(potLm);
  response.angularVelocity = omega;
  response.iterations = count;
  response.error = err;
  response.converged = converged;
  return response;
}

SeaLevelSolver::Response SeaLevelSolver::Solve(const State& state,
                                               std::vector<double> directLoad,
                                               RotationalFeedback rotation,
                                               Options options) const {
  GeneralisedForcing forcing;
  forcing.directLoad = std::move(directLoad);
  forcing.rotation = rotation;
  return SolveGeneralised(state, forcing, options);
}

// -- the adjoint solve
// -----------------------------------------------------------

SeaLevelSolver::AdjointResponse SeaLevelSolver::SolveAdjoint(
    const State& state, const AdjointForcing& forcing, Options options) const {
  const auto fieldSize = grid_.FieldSize();
  const auto coefSize = grid_.CoefficientSize();
  const bool transverse = forcing.rotation == RotationalFeedback::Full;
  const bool rotational = forcing.rotation != RotationalFeedback::None;
  const double r2 = radius_ * radius_;

  const auto& oceanFunction = forcing.oceanFunctionOverride.empty()
                                  ? state.oceanFunction
                                  : forcing.oceanFunctionOverride;
  const double oceanArea = forcing.oceanFunctionOverride.empty()
                               ? state.oceanArea
                               : IntegrateSurface(oceanFunction);

  auto gXi = std::vector<double>(fieldSize, 0.0);
  if (!forcing.seaLevelDual.empty()) gXi = forcing.seaLevelDual;
  std::vector<std::complex<double>> lamU(coefSize, 0.0), lamPhi(coefSize, 0.0);
  if (!forcing.displacementDual.empty()) {
    lamU = grid_.Expand(forcing.displacementDual);
    for (auto& v : lamU) v *= r2;
  }
  if (!forcing.potentialDual.empty()) {
    lamPhi = grid_.Expand(forcing.potentialDual);
    for (auto& v : lamPhi) v *= r2;
  }
  const auto& gOmega = forcing.angularVelocityDual;

  AdjointResponse out;
  out.directLoadDual.assign(fieldSize, 0.0);
  out.displacementResponseDual.assign(coefSize, 0.0);
  out.potentialResponseDual.assign(coefSize, 0.0);
  if (MaxAbs(gXi) == 0.0 && MaxAbs(lamU) == 0.0 && MaxAbs(lamPhi) == 0.0 &&
      MaxAbs(std::span<const double>(gOmega.data(), 3)) == 0.0) {
    return out;
  }

  // Duals the synthesised outputs contribute outside the fixed point.
  std::vector<std::complex<double>> lamUBase = lamU, lamPhiBase = lamPhi;
  Block2 du2Out{}, dphi2Out{};
  auto dpsi0Out = 0.0;
  if (rotational) {
    du2Out = BlockOf(lamU);
    dphi2Out = BlockOf(lamPhi);
    SetBlock(lamUBase, Block2{});
    SetBlock(lamPhiBase, Block2{});
    dpsi0Out = hc_ * lamU[SurfaceGrid::Index(0, 0)].real() +
               kc_ * lamPhi[SurfaceGrid::Index(0, 0)].real();
  }
  auto rOut = std::vector<std::complex<double>>(coefSize);
  for (auto l = 0; l <= grid_.LMax(); ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto i = SurfaceGrid::Index(l, m);
      rOut[i] = h_[l] * lamUBase[i] + k_[l] * lamPhiBase[i];
    }
  }
  const double hkc = hc_ + (kc_ + 1.0) / g_;

  struct Stash {
    double meanDual = 0.0;
    double j = 0.0;
    std::vector<std::complex<double>> zBase;
    Block2 dsU2{}, dsPhi2{};
  };

  // One application of the transposed per-iteration map.
  auto reverseOnce = [&](std::span<const std::complex<double>> lam) {
    Stash stash;
    const auto lamGrid = grid_.Synthesise(lam);
    auto xiDual = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      xiDual[i] = gXi[i] + waterDensity_ * oceanFunction[i] * (lamGrid[i] / r2);
    }
    stash.meanDual = IntegrateSurface(xiDual);
    auto xgDual = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      xgDual[i] = xiDual[i] - oceanFunction[i] * (stash.meanDual / oceanArea);
    }
    auto zDual = grid_.Expand(xgDual);
    for (auto& v : zDual) v *= r2;

    auto next = rOut;
    if (rotational) {
      // Duals of the full degree-2 objects, then of the centrifugal
      // potential, the angular velocity, and the forcings.
      const auto z2 = BlockOf(zDual);
      stash.zBase = zDual;
      SetBlock(stash.zBase, Block2{});
      const auto dDisp2 = du2Out + (-1.0 * z2);
      const Block2 dGrav2 = dphi2Out + (-1.0 / g_) * z2;
      const Block2 dPsi2 = (-1.0 / g_) * z2 + ht2_ * dDisp2 + kt2_ * dGrav2;
      const double dPsi0 =
          dpsi0Out - hkc * zDual[SurfaceGrid::Index(0, 0)].real();
      Block2 dGravForcing = dGrav2;
      std::array<double, 3> domega = gOmega;
      domega[0] += rotationFactor_ * dPsi2[0][1];
      domega[1] += rotationFactor_ * dPsi2[1][1];
      domega[2] +=
          axialRotationFactor_ * dPsi2[0][0] + uniformRotationFactor_ * dPsi0;
      if (transverse) {
        dGravForcing[0][1] += inertiaFactor_ * domega[0] / rotationDenominator_;
        dGravForcing[1][1] += inertiaFactor_ * domega[1] / rotationDenominator_;
      }
      const double axial = domega[2] / axialRotationDenominator_;
      dGravForcing[0][0] += -axialInertiaFactor_ * axial;
      const double dl0 = -(traceFactor_ * (mu_ + mphi_) + shellFactor_) * axial;
      stash.j = -traceFactor_ * axial;
      for (auto l = 0; l <= grid_.LMax(); ++l) {
        for (auto m = 0; m <= l; ++m) {
          const auto i = SurfaceGrid::Index(l, m);
          next[i] += slcFactor_[l] * stash.zBase[i];
        }
      }
      AddBlock(next, k2_ * dGravForcing + h2_ * dDisp2);
      next[SurfaceGrid::Index(0, 0)] += dl0;
      stash.dsU2 = dDisp2;
      stash.dsPhi2 = dGravForcing;
    } else {
      stash.zBase = std::move(zDual);
      for (auto l = 0; l <= grid_.LMax(); ++l) {
        for (auto m = 0; m <= l; ++m) {
          const auto i = SurfaceGrid::Index(l, m);
          next[i] += slcFactor_[l] * stash.zBase[i];
        }
      }
    }
    return std::pair(std::move(next), std::move(stash));
  };

  detail::AndersonMixer mixer(options.anderson_memory);
  auto lam = std::vector<std::complex<double>>(coefSize, 0.0);
  auto err = std::numeric_limits<double>::infinity();
  auto converged = false;
  auto count = 0;
  while (count < options.max_iterations) {
    auto [next, stash] = reverseOnce(lam);
    const auto scale = MaxAbs(std::span<const std::complex<double>>(next));
    auto diff = 0.0;
    for (auto i = std::size_t{0}; i < coefSize; ++i) {
      diff = std::max(diff, std::abs(next[i] - lam[i]));
    }
    err = scale > 0.0 ? diff / scale : 0.0;
    if (err <= options.rtol) {
      converged = true;
      lam = std::move(next);
      break;
    }
    lam = FromReal(mixer.Update(AsReal(lam), AsReal(next)));
    ++count;
  }

  // The input duals, read at the converged dual load.
  auto [unused, stash] = reverseOnce(lam);
  const auto lamGrid = grid_.Synthesise(lam);
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    out.directLoadDual[i] =
        lamGrid[i] / r2 - stash.meanDual / (waterDensity_ * oceanArea);
  }
  for (auto i = std::size_t{0}; i < coefSize; ++i) {
    out.displacementResponseDual[i] = -stash.zBase[i] + lamUBase[i];
    out.potentialResponseDual[i] = -stash.zBase[i] / g_ + lamPhiBase[i];
  }
  if (rotational) {
    AddBlock(out.displacementResponseDual, stash.dsU2);
    AddBlock(out.potentialResponseDual, stash.dsPhi2);
  }
  out.inertiaMomentDual = stash.j;
  out.iterations = count;
  out.error = err;
  out.converged = converged;
  return out;
}

// -- the nonlinear (migrating shoreline) solve
// ---------------------------------------

SeaLevelSolver::NonlinearResponse SeaLevelSolver::SolveNonlinear(
    const State& state, const NonlinearForcing& forcing,
    Options options) const {
  const auto fieldSize = grid_.FieldSize();
  const auto coefSize = grid_.CoefficientSize();
  const bool transverse = forcing.rotation == RotationalFeedback::Full;
  const bool rotational = forcing.rotation != RotationalFeedback::None;
  const double sedimentDensity =
      forcing.sedimentDensity >= 0.0
          ? forcing.sedimentDensity
          : 2300.0 / scales_.Factor(DensityDimensions);

  std::vector<std::complex<double>> staticDisp, staticGrav, staticSlc;
  std::optional<Block2> staticDisp2, staticGrav2;
  if (!forcing.displacementResponse.empty() ||
      !forcing.potentialResponse.empty()) {
    staticDisp.assign(coefSize, 0.0);
    staticGrav.assign(coefSize, 0.0);
    if (!forcing.displacementResponse.empty()) {
      staticDisp = forcing.displacementResponse;
    }
    if (!forcing.potentialResponse.empty()) {
      staticGrav = forcing.potentialResponse;
    }
    staticSlc.resize(coefSize);
    for (auto i = std::size_t{0}; i < coefSize; ++i) {
      staticSlc[i] = -(staticDisp[i] + staticGrav[i] / g_);
    }
    staticDisp2 = BlockOf(staticDisp);
    staticGrav2 = BlockOf(staticGrav);
  }

  const auto& initialBathy = state.seaLevel;
  const auto& initialIce = state.iceThickness;
  const auto& initialOcean = state.oceanFunction;

  std::vector<double> newIce = initialIce;
  if (!forcing.iceThicknessChange.empty()) {
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      newIce[i] += forcing.iceThicknessChange[i];
    }
  }
  auto sedimentLoad = std::vector<double>{};
  auto sedimentMassChange = 0.0;
  if (!forcing.sedimentThicknessChange.empty()) {
    sedimentLoad.resize(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      sedimentLoad[i] = sedimentDensity * forcing.sedimentThicknessChange[i];
    }
    sedimentMassChange = IntegrateSurface(sedimentLoad);
  }

  auto initialWater = std::vector<double>(fieldSize);
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    initialWater[i] = initialOcean[i] * initialBathy[i];
  }
  const double initialWaterMass =
      waterDensity_ * IntegrateSurface(initialWater);

  std::vector<double> currentOcean = initialOcean;
  std::vector<double> currentBathy = initialBathy;
  auto slcData = std::vector<double>(fieldSize, 0.0);
  auto omega = std::array<double, 3>{0.0, 0.0, 0.0};

  auto loadLm = std::vector<std::complex<double>>(coefSize, 0.0);
  auto lastRotation = std::optional<Rotation>{};

  auto err = std::numeric_limits<double>::infinity();
  auto converged = false;
  auto count = 0;
  while (count < options.max_iterations) {
    auto totalLoad = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      const double groundedIceChange =
          iceDensity_ * ((1.0 - currentOcean[i]) * newIce[i] -
                         (1.0 - initialOcean[i]) * initialIce[i]);
      const double oceanMassChange =
          waterDensity_ * (currentOcean[i] * currentBathy[i] -
                           initialOcean[i] * initialBathy[i]);
      totalLoad[i] = groundedIceChange + oceanMassChange;
      if (!sedimentLoad.empty()) totalLoad[i] += sedimentLoad[i];
    }
    loadLm = grid_.Expand(totalLoad);

    auto slcLm = std::vector<std::complex<double>>(coefSize);
    for (auto l = 0; l <= grid_.LMax(); ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto i = SurfaceGrid::Index(l, m);
        slcLm[i] = slcFactor_[l] * loadLm[i];
      }
    }
    if (!staticSlc.empty()) {
      for (auto i = std::size_t{0}; i < coefSize; ++i) slcLm[i] += staticSlc[i];
    }
    if (rotational) {
      lastRotation.emplace(*this, BlockOf(loadLm),
                           loadLm[SurfaceGrid::Index(0, 0)].real(), staticDisp2,
                           staticGrav2, 0.0, 0.0, forcing.inertiaMomentChange,
                           std::array<double, 3>{0.0, 0.0, 0.0}, transverse);
      omega = lastRotation->omega;
      SetBlock(slcLm, -1.0 * (lastRotation->disp2 +
                              (1.0 / g_) * (lastRotation->grav2 +
                                            lastRotation->centrifugal)));
      slcLm[SurfaceGrid::Index(0, 0)] -=
          (hc_ + (kc_ + 1.0) / g_) * lastRotation->centrifugal0;
    }
    const auto slcLocal = grid_.Synthesise(slcLm);

    // The explicit mass budget: grounded ice and sediment taken from the
    // ocean, the eustatic shift closing it exactly.
    auto rawBathy = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      rawBathy[i] = initialBathy[i] + slcLocal[i];
      if (!forcing.sedimentThicknessChange.empty()) {
        rawBathy[i] -= forcing.sedimentThicknessChange[i];
      }
      if (!forcing.dynamicSeaLevelChange.empty()) {
        rawBathy[i] += forcing.dynamicSeaLevelChange[i];
      }
    }
    auto groundedIceChange = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      groundedIceChange[i] =
          iceDensity_ * ((1.0 - currentOcean[i]) * newIce[i] -
                         (1.0 - initialOcean[i]) * initialIce[i]);
    }
    const auto iceMassChange = IntegrateSurface(groundedIceChange);
    const double targetWaterMass =
        initialWaterMass - iceMassChange - sedimentMassChange;
    auto currentRaw = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      currentRaw[i] = currentOcean[i] * rawBathy[i];
    }
    const double currentRawWaterMass =
        waterDensity_ * IntegrateSurface(currentRaw);
    const double oceanDensityArea =
        waterDensity_ * IntegrateSurface(currentOcean);
    const double eustaticShift =
        (targetWaterMass - currentRawWaterMass) / oceanDensityArea;

    std::vector<double> newSlc(fieldSize), newBathy(fieldSize),
        newOcean(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      newSlc[i] = slcLocal[i] + eustaticShift;
      newBathy[i] = rawBathy[i] + eustaticShift;
      newOcean[i] = waterDensity_ * newBathy[i] - iceDensity_ * newIce[i] > 0.0
                        ? 1.0
                        : 0.0;
    }

    const auto maxSlc = MaxAbs(slcData);
    if (maxSlc > 0.0) {
      auto diff = 0.0;
      for (auto i = std::size_t{0}; i < fieldSize; ++i) {
        diff = std::max(diff, std::abs(newSlc[i] - slcData[i]));
      }
      err = diff / maxSlc;
    } else {
      err = MaxAbs(newSlc) == 0.0 ? 0.0 : 1.0;
    }

    slcData = std::move(newSlc);
    currentBathy = std::move(newBathy);
    currentOcean = std::move(newOcean);
    ++count;
    if (err <= options.rtol) {
      converged = true;
      break;
    }
  }

  std::vector<std::complex<double>> dispLm(coefSize), potLm(coefSize);
  for (auto l = 0; l <= grid_.LMax(); ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto i = SurfaceGrid::Index(l, m);
      dispLm[i] = h_[l] * loadLm[i];
      potLm[i] = k_[l] * loadLm[i];
    }
  }
  if (!staticDisp.empty()) {
    for (auto i = std::size_t{0}; i < coefSize; ++i) {
      dispLm[i] += staticDisp[i];
      potLm[i] += staticGrav[i];
    }
  }
  if (rotational && lastRotation) {
    SetBlock(dispLm, lastRotation->disp2);
    SetBlock(potLm, lastRotation->grav2);
    dispLm[SurfaceGrid::Index(0, 0)] += hc_ * lastRotation->centrifugal0;
    potLm[SurfaceGrid::Index(0, 0)] += kc_ * lastRotation->centrifugal0;
  }

  NonlinearResponse out;
  out.finalState.iceThickness = std::move(newIce);
  out.finalState.seaLevel = std::move(currentBathy);
  out.finalState.oceanFunction = std::move(currentOcean);
  out.finalState.oceanArea = IntegrateSurface(out.finalState.oceanFunction);
  out.response.seaLevelChange = std::move(slcData);
  out.response.displacement = grid_.Synthesise(dispLm);
  out.response.potential = grid_.Synthesise(potLm);
  out.response.angularVelocity = omega;
  out.response.iterations = count;
  out.response.error = err;
  out.response.converged = converged;
  return out;
}

}  // namespace sl3dpp
