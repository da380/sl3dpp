// The lateral kernels about a radial background (docs/adjoints.md §21):
// the accumulator the 1D sweeps feed (detail/LateralKernels.hpp).

#include "sl3dpp/LateralKernels.hpp"
#include "sl3dpp/detail/LateralKernels.hpp"

#include <array>
#include <complex>
#include <format>
#include <span>
#include <stdexcept>
#include <utility>

#include "sl3dpp/StepWeights.hpp"
#include "sl3dpp/detail/Parallel.hpp"
#include "sl3dpp/detail/ViscoelasticDegree.hpp"

namespace sl3dpp::detail {

namespace {

using Complex = std::complex<double>;
using Tensor = GridCoupling::TensorField;
using Spectral = GridCoupling::SpectralTensor;

/// The null-basis energy contraction of two real tensors at a grid value
/// (the derived spins by reality): the metric the 3D sweeps pair with,
/// whose angle-uniform reduction is the 1D force weights.
double Contract(const Tensor& a, const Tensor& b, std::size_t i) {
  return 12.0 * a.r[i] * b.r[i] + 8.0 * (a.c[1][i] * std::conj(b.c[1][i])).real() +
         4.0 * (a.c[0][i] * std::conj(b.c[0][i])).real();
}

/// The weight of order m in the real-column pairing: m > 0 twice.
double OrderWeight(int m) { return m == 0 ? 1.0 : 2.0; }

}  // namespace

LateralAccumulator::LateralAccumulator(const ViscoelasticOperator& op,
                                       const GridCoupling& coupling,
                                       bool viscosity, bool moduli)
    : op_(&op),
      toroidal_(op),
      coupling_(&coupling),
      viscosity_(viscosity),
      moduli_(moduli),
      branches_(op.GetRheology().BranchCount()),
      lmax_(op.LMax()) {
  if (coupling.LMax() != op.LMax()) {
    throw std::invalid_argument(std::format(
        "the lateral kernels' coupling is at lmax {}, the run at {}",
        coupling.LMax(), op.LMax()));
  }
  if (coupling.Points().set != op.Points().set ||
      coupling.Points().count != op.Points().count) {
    throw std::invalid_argument(
        "the lateral kernels' coupling holds other memory points than the "
        "run's operator");
  }
  // the degrees' workspaces filled here, one thread (the caches fill
  // lazily; the expansions run over the degrees in parallel)
  for (auto l = 0; l <= lmax_; ++l) {
    static_cast<void>(op.MemoryElements(l));
    static_cast<void>(op.UnrelaxedSystem(l));
    if (l >= 1) static_cast<void>(toroidal_.Unrelaxed(l));
  }
  const auto np = static_cast<std::size_t>(op.Points().count);
  const auto nodes = static_cast<std::size_t>(coupling.NodeCount());
  pointOf_.resize(nodes);
  elementOf_.resize(nodes);
  for (auto rn = std::size_t{0}; rn < nodes; ++rn) {
    const auto e = coupling.ElementOfNode(static_cast<int>(rn));
    elementOf_[rn] = e;
    pointOf_[rn] = static_cast<std::size_t>(e) * np +
                   coupling.NodeInElementOf(static_cast<int>(rn));
  }
  for (auto k = std::size_t{0}; k < branches_; ++k) {
    const auto itau = op.PointItau(k);
    const auto mu = op.PointMu(k);
    const auto fraction = op.BranchFractionAtPoints(k);
    auto a = std::vector<double>(nodes), b = a, f = a;
    for (auto rn = std::size_t{0}; rn < nodes; ++rn) {
      a[rn] = itau[pointOf_[rn]];
      b[rn] = mu[pointOf_[rn]];
      f[rn] = fraction[pointOf_[rn]];
    }
    itau_.push_back(std::move(a));
    mu_.push_back(std::move(b));
    fractions_.push_back(std::move(f));
  }
  const auto values = coupling.GridValueCount();
  if (viscosity_) itauKernel_.assign(branches_, std::vector<double>(values, 0.0));
  if (moduli_) {
    kappaKernel_.assign(values, 0.0);
    muKernel_.assign(values, 0.0);
  }
}

// -- the expansions -----------------------------------------------------------

void LateralAccumulator::AddSpheroidal(Spectral& to, int l,
                                       const double* scalars,
                                       Columns columns) const {
  const auto& coupling = *coupling_;
  const auto [e0, ne] = op_->MemoryElements(l);
  const auto np = op_->Points().count;
  const auto ncols = 2 * (l + 1);
  const auto block = static_cast<std::size_t>(ne) * np * ncols;
  const auto cols = static_cast<std::size_t>(l) + 1;
  const auto nodes = static_cast<std::size_t>(coupling.NodeCount());
  const auto wc = ScalarWeights(l * (l + 1.0));
  auto M = std::vector<Complex>(nodes * cols), R = M, S = M;
  const auto out = std::array{&M, &R, &S};
  for (auto rn = std::size_t{0}; rn < nodes; ++rn) {
    const auto e = elementOf_[rn];
    if (e < e0 || e >= e0 + ne) continue;
    const auto p = static_cast<std::size_t>(e - e0) * np +
                   coupling.NodeInElementOf(static_cast<int>(rn));
    const auto wq = coupling.WeightOfNode(static_cast<int>(rn));
    for (auto s = 0; s < 3; ++s) {
      for (auto m = 0; m <= l; ++m) {
        auto factor = 1.0;
        if (columns == Columns::DecayedDual) {
          factor = wc[s] == 0.0 ? 0.0 : 1.0 / (wc[s] * wq * OrderWeight(m));
        } else if (columns == Columns::Dual) {
          factor = 1.0 / OrderWeight(m);
        }
        const auto at = s * block + p * ncols;
        const auto re = scalars[at + m];
        const auto im = m == 0 ? 0.0 : scalars[at + l + 1 + m];
        (*out[s])[rn * cols + m] = factor * Complex(re, im);
      }
    }
  }
  coupling.AddScalarState(to, l, M, R, S);
}

void LateralAccumulator::AddToroidal(Spectral& to, int l, const double* scalars,
                                     Columns columns) const {
  if (l < 1) return;
  const auto& coupling = *coupling_;
  const auto e0 = toroidal_.Unrelaxed(l).FirstElement();
  const auto ne = op_->BaseMaterial().Mesh().ElementCount() - e0;
  const auto np = op_->Points().count;
  const auto ncols = 2 * (l + 1);
  const auto block = static_cast<std::size_t>(ne) * np * ncols;
  const auto cols = static_cast<std::size_t>(l) + 1;
  const auto nodes = static_cast<std::size_t>(coupling.NodeCount());
  const auto k2 = l * (l + 1.0);
  const double wc[2] = {k2 * (k2 - 2.0), k2};
  auto M = std::vector<Complex>(nodes * cols), S = M;
  const auto out = std::array{&M, &S};
  for (auto rn = std::size_t{0}; rn < nodes; ++rn) {
    const auto e = elementOf_[rn];
    if (e < e0 || e >= e0 + ne) continue;
    const auto p = static_cast<std::size_t>(e - e0) * np +
                   coupling.NodeInElementOf(static_cast<int>(rn));
    const auto wq = coupling.WeightOfNode(static_cast<int>(rn));
    for (auto s = 0; s < 2; ++s) {
      for (auto m = 0; m <= l; ++m) {
        auto factor = 1.0;
        if (columns == Columns::DecayedDual) {
          factor = wc[s] == 0.0 ? 0.0 : 1.0 / (wc[s] * wq * OrderWeight(m));
        } else if (columns == Columns::Dual) {
          factor = 1.0 / OrderWeight(m);
        }
        const auto at = s * block + p * ncols;
        const auto re = scalars[at + m];
        const auto im = m == 0 ? 0.0 : scalars[at + l + 1 + m];
        (*out[s])[rn * cols + m] = factor * Complex(re, im);
      }
    }
  }
  coupling.AddToroidalScalarState(to, l, M, S);
}

namespace {

/// A degree's real columns (dof, 2 (l + 1)) as the complex m >= 0 columns
/// (dof, l + 1) of the same real field, each order divided by `weight(m)`.
std::vector<Complex> ComplexColumns(std::span<const double> x, int l,
                                    bool dual) {
  const auto ndof = x.size() / (2 * (static_cast<std::size_t>(l) + 1));
  auto out = std::vector<Complex>(ndof * (static_cast<std::size_t>(l) + 1));
  for (auto m = 0; m <= l; ++m) {
    const auto factor = dual ? 1.0 / OrderWeight(m) : 1.0;
    for (auto d = std::size_t{0}; d < ndof; ++d) {
      const auto re = x[static_cast<std::size_t>(m) * ndof + d];
      const auto im = m == 0 ? 0.0 : x[static_cast<std::size_t>(l + 1 + m) * ndof + d];
      out[static_cast<std::size_t>(m) * ndof + d] = factor * Complex(re, im);
    }
  }
  return out;
}

}  // namespace

Tensor LateralAccumulator::StrainOf(const Fields& x, const Fields* toroidal,
                                    Columns columns) const {
  return coupling_->Synthesise(SpectralStrainOf(x, toroidal, columns));
}

Spectral LateralAccumulator::SpectralStrainOf(const Fields& x,
                                              const Fields* toroidal,
                                              Columns columns) const {
  const auto& coupling = *coupling_;
  const auto dual = columns != Columns::Natural;
  auto spectral = coupling.ZeroSpectral();
  // one task per degree: each writes only its own degree's coefficients
  auto degrees = std::vector<int>{};
  for (auto l = 0; l <= lmax_; ++l) {
    const auto hasS = x.contains(l);
    const auto hasT = toroidal != nullptr && toroidal->contains(l);
    if (hasS || hasT) degrees.push_back(l);
  }
  const auto count = static_cast<int>(degrees.size());
  ForEachTask(coupling.GetOptions().parallel, count, [&](int task) {
    const auto l = degrees[static_cast<std::size_t>(count - 1 - task)];
    const auto& sph = op_->UnrelaxedSystem(l);
    auto xs = std::vector<Complex>{};
    if (const auto found = x.find(l); found != x.end()) {
      xs = ComplexColumns(found->second, l, dual);
    } else {
      xs.assign(static_cast<std::size_t>(sph.DofCount()) * (l + 1), Complex{});
    }
    const ToroidalOperator* tor = nullptr;
    auto ws = std::vector<Complex>{};
    if (toroidal != nullptr && l >= 1) {
      if (const auto found = toroidal->find(l); found != toroidal->end()) {
        tor = &toroidal_.Unrelaxed(l);
        ws = ComplexColumns(found->second, l, dual);
      }
    }
    coupling.AddStrainOfSolution(spectral, l, sph, xs, tor, ws);
  });
  return spectral;
}

void LateralAccumulator::AddScaledStrain(Spectral& to, const Fields& x,
                                         std::span<const double> factor) const {
  auto strain = SpectralStrainOf(x, nullptr, Columns::Natural);
  coupling_->ScaleByNode(strain, factor);
  for (auto c = 0; c < 2; ++c) {
    for (auto i = std::size_t{0}; i < strain.c[c].size(); ++i) {
      to.c[c][i] += strain.c[c][i];
    }
  }
  for (auto i = std::size_t{0}; i < strain.r.size(); ++i) to.r[i] += strain.r[i];
}

std::vector<double> LateralAccumulator::DilatationOf(const Fields& x,
                                                     Columns columns) const {
  const auto& coupling = *coupling_;
  auto scalar = std::vector<Complex>(coupling.ScalarCoefficientCount(), 0.0);
  auto degrees = std::vector<int>{};
  for (const auto& [l, xl] : x) degrees.push_back(l);
  const auto count = static_cast<int>(degrees.size());
  ForEachTask(coupling.GetOptions().parallel, count, [&](int task) {
    const auto l = degrees[static_cast<std::size_t>(count - 1 - task)];
    coupling.AddDilatationOfSolution(
        scalar, l, op_->UnrelaxedSystem(l),
        ComplexColumns(x.at(l), l, columns != Columns::Natural));
  });
  return coupling.SynthesiseScalar(scalar);
}

void LateralAccumulator::AddMaterial(double scale, const Tensor& adjointStrain,
                                     std::span<const double> adjointDilatation,
                                     const Tensor& forward,
                                     std::span<const double> forwardDilatation) {
  const auto& coupling = *coupling_;
  ForEach(coupling.GetOptions().parallel,
          static_cast<std::size_t>(coupling.NodeCount()), [&](std::size_t rn) {
            const auto node = static_cast<int>(rn);
            const auto base = coupling.FieldOffset(node);
            const auto size = coupling.FieldSizeOf(node);
            const auto r = coupling.RadiusOfNode(node);
            const auto wq = coupling.WeightOfNode(node);
            for (auto x = std::size_t{0}; x < size; ++x) {
              const auto i = base + x;
              kappaKernel_[i] -=
                  scale * wq * adjointDilatation[i] * forwardDilatation[i];
              muKernel_[i] += scale * wq * r * r * Contract(adjointStrain, forward, i);
            }
          });
}

// -- the continuous sweep -----------------------------------------------------

void LateralAccumulator::ContinuousItau(double scale,
                                        const ViscoelasticState& adjoint,
                                        const ToroidalState* adjointToroidal,
                                        const ViscoelasticState& forward,
                                        const ToroidalState* forwardToroidal) {
  if (!viscosity_ || scale == 0.0) return;
  const auto& coupling = *coupling_;
  const auto np = op_->Points().count;
  for (auto k = std::size_t{0}; k < branches_; ++k) {
    auto a = coupling.ZeroSpectral();
    auto f = coupling.ZeroSpectral();
    ForEachTask(coupling.GetOptions().parallel, lmax_ + 1, [&](int task) {
      const auto l = lmax_ - task;
      const auto ne = op_->MemoryElements(l).second;
      const auto block = static_cast<std::size_t>(ne) * np * 2 * (l + 1);
      if (const auto found = adjoint.m.find(l); found != adjoint.m.end()) {
        AddSpheroidal(a, l, found->second.data() + k * 3 * block, Columns::Natural);
      }
      if (const auto found = forward.m.find(l); found != forward.m.end()) {
        const auto& d = forward.strain.at(l);
        auto difference = std::vector<double>(3 * block);
        for (auto j = std::size_t{0}; j < difference.size(); ++j) {
          difference[j] = d[j] - found->second[k * 3 * block + j];
        }
        AddSpheroidal(f, l, difference.data(), Columns::Natural);
      }
      if (l < 1) return;
      const auto neT = op_->BaseMaterial().Mesh().ElementCount() -
                       toroidal_.Unrelaxed(l).FirstElement();
      const auto blockT = static_cast<std::size_t>(neT) * np * 2 * (l + 1);
      if (adjointToroidal != nullptr) {
        if (const auto found = adjointToroidal->m.find(l);
            found != adjointToroidal->m.end()) {
          AddToroidal(a, l, found->second.data() + k * 2 * blockT, Columns::Natural);
        }
      }
      if (forwardToroidal != nullptr) {
        if (const auto found = forwardToroidal->m.find(l);
            found != forwardToroidal->m.end()) {
          const auto& d = forwardToroidal->strain.at(l);
          auto difference = std::vector<double>(2 * blockT);
          for (auto j = std::size_t{0}; j < difference.size(); ++j) {
            difference[j] = d[j] - found->second[k * 2 * blockT + j];
          }
          AddToroidal(f, l, difference.data(), Columns::Natural);
        }
      }
    });
    const auto A = coupling.Synthesise(a);
    const auto F = coupling.Synthesise(f);
    auto& kernel = itauKernel_[k];
    ForEach(coupling.GetOptions().parallel,
            static_cast<std::size_t>(coupling.NodeCount()), [&](std::size_t rn) {
              const auto itau = itau_[k][rn];
              const auto mu = mu_[k][rn];
              if (itau <= 0.0 || mu <= 0.0) return;
              const auto node = static_cast<int>(rn);
              const auto r = coupling.RadiusOfNode(node);
              const auto factor =
                  scale * mu * coupling.WeightOfNode(node) * r * r / itau;
              const auto base = coupling.FieldOffset(node);
              const auto size = coupling.FieldSizeOf(node);
              for (auto x = std::size_t{0}; x < size; ++x) {
                kernel[base + x] += factor * Contract(A, F, base + x);
              }
            });
  }
}

void LateralAccumulator::ContinuousMaterial(
    double scale, const Fields& lambda, const Fields* lambdaToroidal,
    const Fields& x, const ViscoelasticState& forward,
    const ToroidalState* forwardToroidal) {
  if (!moduli_ || scale == 0.0) return;
  const auto& coupling = *coupling_;
  const auto np = op_->Points().count;
  // the forward sum_k f_k m_k - d, the fractions radial: the memory over
  // its window, the strain the full solution's over every node (the
  // elastic elements' too, where the state carries none)
  auto f = coupling.ZeroSpectral();
  ForEachTask(coupling.GetOptions().parallel, lmax_ + 1, [&](int task) {
    const auto l = lmax_ - task;
    const auto [e0, ne] = op_->MemoryElements(l);
    const auto ncols = 2 * (l + 1);
    const auto block = static_cast<std::size_t>(ne) * np * ncols;
    if (const auto found = forward.m.find(l); found != forward.m.end()) {
      auto term = std::vector<double>(3 * block, 0.0);
      const auto& m = found->second;
      for (auto s = std::size_t{0}; s < 3; ++s) {
        for (auto p = std::size_t{0}; p < static_cast<std::size_t>(ne) * np; ++p) {
          const auto full = static_cast<std::size_t>(e0) * np + p;
          for (auto c = 0; c < ncols; ++c) {
            const auto at = s * block + p * ncols + c;
            auto v = 0.0;
            for (auto k = std::size_t{0}; k < branches_; ++k) {
              const auto fk = op_->BranchFractionAtPoints(k)[full];
              if (fk != 0.0) v += fk * m[k * 3 * block + at];
            }
            term[at] = v;
          }
        }
      }
      AddSpheroidal(f, l, term.data(), Columns::Natural);
    }
    if (l < 1 || forwardToroidal == nullptr) return;
    if (const auto found = forwardToroidal->strain.find(l);
        found != forwardToroidal->strain.end()) {
      const auto e0T = toroidal_.Unrelaxed(l).FirstElement();
      const auto neT = op_->BaseMaterial().Mesh().ElementCount() - e0T;
      const auto blockT = static_cast<std::size_t>(neT) * np * ncols;
      auto term = std::vector<double>(2 * blockT);
      const auto& m = forwardToroidal->m.at(l);
      for (auto s = std::size_t{0}; s < 2; ++s) {
        for (auto p = std::size_t{0}; p < static_cast<std::size_t>(neT) * np; ++p) {
          const auto full = static_cast<std::size_t>(e0T) * np + p;
          for (auto c = 0; c < ncols; ++c) {
            const auto at = s * blockT + p * ncols + c;
            auto v = -found->second[at];
            for (auto k = std::size_t{0}; k < branches_; ++k) {
              const auto fk = op_->BranchFractionAtPoints(k)[full];
              if (fk != 0.0) v += fk * m[k * 2 * blockT + at];
            }
            term[at] = v;
          }
        }
      }
      AddToroidal(f, l, term.data(), Columns::Natural);
    }
  });
  AddScaledStrain(
      f, x, std::vector<double>(static_cast<std::size_t>(coupling.NodeCount()), -1.0));
  const auto forwardTensor = coupling.Synthesise(f);
  const auto adjointStrain = StrainOf(lambda, lambdaToroidal, Columns::Natural);
  AddMaterial(scale, adjointStrain, DilatationOf(lambda, Columns::Natural),
              forwardTensor, DilatationOf(x, Columns::Natural));
}

// -- the discrete sweep -------------------------------------------------------

namespace {

/// A branch's scheme weights at a point: the end-strain weight b (1 - e
/// for backward Euler, beta for the trapezoid), the decay e and the start
/// strain's alpha (zero for backward Euler); and their h-derivatives.
struct PointWeights {
  double e, alpha, b, de, dalpha, db;
};

PointWeights WeightsAt(Scheme scheme, double h) {
  if (scheme == Scheme::BackwardEuler) {
    const auto e = BackwardEulerWeight(h);
    const auto de = BackwardEulerWeightDerivative(h);
    return {e, 0.0, 1.0 - e, de, 0.0, -de};
  }
  const auto w = ExponentialTrapezoidWeights(h);
  const auto dw = ExponentialTrapezoidWeightDerivatives(h);
  return {w.e, w.alpha, w.beta, dw.e, dw.alpha, dw.beta};
}

}  // namespace

void LateralAccumulator::DiscreteItau(double dt, Scheme scheme,
                                      const Fields& decayedDual,
                                      const Fields* decayedDualToroidal,
                                      const ViscoelasticState& prev,
                                      const ViscoelasticState& next,
                                      const ToroidalState* prevToroidal,
                                      const ToroidalState* nextToroidal) {
  if (!viscosity_) return;
  const auto& coupling = *coupling_;
  const auto np = op_->Points().count;
  for (auto k = std::size_t{0}; k < branches_; ++k) {
    const auto itauK = op_->PointItau(k);
    auto z = coupling.ZeroSpectral();
    auto t = coupling.ZeroSpectral();
    ForEachTask(coupling.GetOptions().parallel, lmax_ + 1, [&](int task) {
      const auto l = lmax_ - task;
      const auto ncols = 2 * (l + 1);
      // dt-free drive dE m + dA d0 + dB d1 over a window starting at e0
      auto drive = [&](const std::vector<double>& m, const std::vector<double>& d0,
                       const std::vector<double>& d1, int e0, int ne, int scalars) {
        const auto block = static_cast<std::size_t>(ne) * np * ncols;
        auto out = std::vector<double>(static_cast<std::size_t>(scalars) * block);
        for (auto p = std::size_t{0}; p < static_cast<std::size_t>(ne) * np; ++p) {
          const auto full = static_cast<std::size_t>(e0) * np + p;
          const auto w = WeightsAt(scheme, dt * itauK[full]);
          for (auto s = 0; s < scalars; ++s) {
            for (auto c = 0; c < ncols; ++c) {
              const auto at = s * block + p * ncols + c;
              out[at] = w.de * m[k * scalars * block + at] + w.dalpha * d0[at] +
                        w.db * d1[at];
            }
          }
        }
        return out;
      };
      {
        const auto [e0, ne] = op_->MemoryElements(l);
        const auto block = static_cast<std::size_t>(ne) * np * ncols;
        if (const auto found = decayedDual.find(l); found != decayedDual.end()) {
          AddSpheroidal(z, l, found->second.data() + k * 3 * block,
                        Columns::DecayedDual);
        }
        if (prev.m.contains(l)) {
          const auto out = drive(prev.m.at(l), prev.strain.at(l), next.strain.at(l),
                                 e0, ne, 3);
          AddSpheroidal(t, l, out.data(), Columns::Natural);
        }
      }
      if (l < 1) return;
      const auto e0T = toroidal_.Unrelaxed(l).FirstElement();
      const auto neT = op_->BaseMaterial().Mesh().ElementCount() - e0T;
      const auto blockT = static_cast<std::size_t>(neT) * np * ncols;
      if (decayedDualToroidal != nullptr) {
        if (const auto found = decayedDualToroidal->find(l);
            found != decayedDualToroidal->end()) {
          AddToroidal(z, l, found->second.data() + k * 2 * blockT,
                      Columns::DecayedDual);
        }
      }
      if (prevToroidal != nullptr && nextToroidal != nullptr &&
          prevToroidal->m.contains(l)) {
        const auto out = drive(prevToroidal->m.at(l), prevToroidal->strain.at(l),
                               nextToroidal->strain.at(l), e0T, neT, 2);
        AddToroidal(t, l, out.data(), Columns::Natural);
      }
    });
    const auto Z = coupling.Synthesise(z);
    const auto T = coupling.Synthesise(t);
    auto& kernel = itauKernel_[k];
    ForEach(coupling.GetOptions().parallel,
            static_cast<std::size_t>(coupling.NodeCount()), [&](std::size_t rn) {
              if (itau_[k][rn] <= 0.0) return;
              const auto node = static_cast<int>(rn);
              const auto r = coupling.RadiusOfNode(node);
              const auto factor = dt * coupling.WeightOfNode(node) * r * r;
              const auto base = coupling.FieldOffset(node);
              const auto size = coupling.FieldSizeOf(node);
              for (auto x = std::size_t{0}; x < size; ++x) {
                kernel[base + x] += factor * Contract(Z, T, base + x);
              }
            });
  }
}

void LateralAccumulator::DiscreteMaterial(double dt, Scheme scheme,
                                          const Fields& y, const Fields* yToroidal,
                                          const Fields& x,
                                          const ViscoelasticState& prev,
                                          const ToroidalState* prevToroidal,
                                          const ToroidalState* nextToroidal) {
  if (!moduli_) return;
  const auto& coupling = *coupling_;
  const auto np = op_->Points().count;
  // the forward sum_k f_k m~_k - (1 - sum_k f_k b_k) d1, the decayed memory
  // m~_k = e_k m_k + alpha_k d0 and the weights radial
  auto f = coupling.ZeroSpectral();
  ForEachTask(coupling.GetOptions().parallel, lmax_ + 1, [&](int task) {
    const auto l = lmax_ - task;
    const auto ncols = 2 * (l + 1);
    auto term = [&](const std::vector<double>& m, const std::vector<double>& d0,
                    const std::vector<double>* d1, int e0, int ne, int scalars) {
      const auto block = static_cast<std::size_t>(ne) * np * ncols;
      auto out = std::vector<double>(static_cast<std::size_t>(scalars) * block, 0.0);
      for (auto p = std::size_t{0}; p < static_cast<std::size_t>(ne) * np; ++p) {
        const auto full = static_cast<std::size_t>(e0) * np + p;
        auto factor = 1.0;
        for (auto k = std::size_t{0}; k < branches_; ++k) {
          const auto fk = op_->BranchFractionAtPoints(k)[full];
          if (fk == 0.0) continue;
          const auto w = WeightsAt(scheme, dt * op_->PointItau(k)[full]);
          factor -= fk * w.b;
          for (auto s = 0; s < scalars; ++s) {
            for (auto c = 0; c < ncols; ++c) {
              const auto at = s * block + p * ncols + c;
              out[at] += fk * (w.e * m[k * scalars * block + at] + w.alpha * d0[at]);
            }
          }
        }
        if (d1 == nullptr) continue;
        for (auto s = 0; s < scalars; ++s) {
          for (auto c = 0; c < ncols; ++c) {
            const auto at = s * block + p * ncols + c;
            out[at] -= factor * (*d1)[at];
          }
        }
      }
      return out;
    };
    if (prev.m.contains(l)) {
      const auto [e0, ne] = op_->MemoryElements(l);
      const auto out = term(prev.m.at(l), prev.strain.at(l), nullptr, e0, ne, 3);
      AddSpheroidal(f, l, out.data(), Columns::Natural);
    }
    if (l < 1 || prevToroidal == nullptr ||
        nextToroidal == nullptr || !prevToroidal->m.contains(l)) {
      return;
    }
    const auto e0T = toroidal_.Unrelaxed(l).FirstElement();
    const auto neT = op_->BaseMaterial().Mesh().ElementCount() - e0T;
    const auto out = term(prevToroidal->m.at(l), prevToroidal->strain.at(l),
                          &nextToroidal->strain.at(l), e0T, neT, 2);
    AddToroidal(f, l, out.data(), Columns::Natural);
  });
  // the end strain the full solution's over every node, under minus the
  // effective factor 1 - sum_k f_k b_k
  auto factor = std::vector<double>(static_cast<std::size_t>(coupling.NodeCount()), -1.0);
  for (auto rn = std::size_t{0}; rn < factor.size(); ++rn) {
    for (auto k = std::size_t{0}; k < branches_; ++k) {
      if (fractions_[k][rn] == 0.0) continue;
      factor[rn] += fractions_[k][rn] * WeightsAt(scheme, dt * itau_[k][rn]).b;
    }
  }
  AddScaledStrain(f, x, factor);
  const auto forwardTensor = coupling.Synthesise(f);
  const auto adjointStrain = StrainOf(y, yToroidal, Columns::Dual);
  AddMaterial(1.0, adjointStrain, DilatationOf(y, Columns::Dual), forwardTensor,
              DilatationOf(x, Columns::Natural));
}

void LateralAccumulator::DiscreteStart(const Fields& y, const Fields* yToroidal,
                                       const Fields& x, const Fields* xToroidal) {
  if (!moduli_) return;
  const auto& coupling = *coupling_;
  auto forward = StrainOf(x, xToroidal, Columns::Natural);
  ForEach(coupling.GetOptions().parallel, coupling.GridValueCount(),
          [&](std::size_t i) {
            forward.c[0][i] = -forward.c[0][i];
            forward.c[1][i] = -forward.c[1][i];
            forward.r[i] = -forward.r[i];
          });
  AddMaterial(1.0, StrainOf(y, yToroidal, Columns::Dual),
              DilatationOf(y, Columns::Dual), forward,
              DilatationOf(x, Columns::Natural));
}

Kernels3D LateralAccumulator::Finish(Kernels ice) {
  const auto& coupling = *coupling_;
  const auto nodes = static_cast<std::size_t>(coupling.NodeCount());
  auto integrated = [&](const std::vector<double>& kernel) {
    auto out = std::vector<double>(nodes, 0.0);
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      out[static_cast<std::size_t>(rn)] = coupling.GridOf(rn).Integrate(
          std::span(kernel).subspan(coupling.FieldOffset(rn), coupling.FieldSizeOf(rn)));
    }
    return out;
  };
  Kernels3D out;
  out.ice = std::move(ice);
  if (viscosity_) {
    for (auto k = std::size_t{0}; k < branches_; ++k) {
      out.itauKernelSpherical.push_back(integrated(itauKernel_[k]));
      auto log = std::vector<double>(itauKernel_[k].size(), 0.0);
      for (auto rn = std::size_t{0}; rn < nodes; ++rn) {
        const auto base = coupling.FieldOffset(static_cast<int>(rn));
        const auto size = coupling.FieldSizeOf(static_cast<int>(rn));
        for (auto x = base; x < base + size; ++x) {
          log[x] = -itau_[k][rn] * itauKernel_[k][x];
        }
      }
      out.logViscosityKernel3D.push_back(std::move(log));
    }
    out.itauKernel3D = std::move(itauKernel_);
  }
  if (moduli_) {
    out.kappaKernelSpherical = integrated(kappaKernel_);
    out.muKernelSpherical = integrated(muKernel_);
    out.kappaKernel3D = std::move(kappaKernel_);
    out.muKernel3D = std::move(muKernel_);
  }
  KernelsToFields(coupling, out);
  return out;
}

}  // namespace sl3dpp::detail

// -- the entry point ------------------------------------------------------------

namespace sl3dpp {

GridCoupling LateralCoupling(const ViscoelasticOperator& op,
                             const LateralKernelOptions& options) {
  const auto& o = op.GetOptions();
  if (!options.viscosity && !options.moduli) {
    throw std::invalid_argument(
        "lateral kernels in neither the viscosity nor the moduli: ask for one");
  }
  if (!options.moduli) {
    const auto& rheology = op.GetRheology();
    const auto& mesh = rheology.BaseMaterial().Mesh();
    auto viscous = false;
    for (auto i = std::size_t{0};
         i < static_cast<std::size_t>(mesh.ElementCount()) * mesh.NGll() && !viscous;
         ++i) {
      viscous = rheology.IsActive(i);
    }
    if (!viscous) {
      throw std::invalid_argument(
          "the run has no viscous element, and so no kernel in the "
          "viscosity: ask for the moduli's");
    }
  }
  return GridCoupling(op.GetRheology(), op.LMax(),
                      {.eps = o.eps,
                       .truncation = o.truncation,
                       .oversampling = options.oversampling,
                       .parallel = options.parallel,
                       .matrix_kernel = options.matrix_kernel,
                       .cache_bytes = options.cache_bytes,
                       .memory_points = o.memory_points,
                       .transform_degrees = options.transform_degrees,
                       .all_solid = options.moduli});
}

LateralKernelSolver::LateralKernelSolver(const Evolution& evolution,
                                         AdjointSolver::Options options,
                                         LateralKernelOptions lateral)
    : route_(Route::Radial),
      radial_(&evolution),
      radialOptions_(std::move(options)),
      options_(std::move(lateral)),
      owned_(std::make_unique<GridCoupling>(
          LateralCoupling(evolution.Operator(), options_))),
      coupling_(owned_.get()) {
  if (!evolution.Operator().Gravitating()) {
    throw std::invalid_argument(
        "the lateral kernels are not built for the problem without gravity");
  }
}

LateralKernelSolver::LateralKernelSolver(const Evolution3D& evolution,
                                         AdjointSolver3D::Options options,
                                         LateralKernelOptions lateral)
    : route_(Route::Lateral),
      lateral_(&evolution),
      lateralOptions_(std::move(options)),
      options_(std::move(lateral)),
      coupling_(&evolution.Operator().Coupling()) {
  if (!options_.viscosity && !options_.moduli) {
    throw std::invalid_argument(
        "lateral kernels in neither the viscosity nor the moduli: ask for one");
  }
  lateralOptions_.material_kernels = lateralOptions_.material_kernels || options_.moduli;
}

LateralKernelSolver::~LateralKernelSolver() = default;
LateralKernelSolver::LateralKernelSolver(LateralKernelSolver&&) noexcept = default;
LateralKernelSolver& LateralKernelSolver::operator=(LateralKernelSolver&&) noexcept =
    default;

Kernels3D LateralKernelSolver::FirstOrderKernels(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory& trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  if (route_ != Route::Radial) {
    throw std::invalid_argument(
        "a radial trajectory for a 3D run: the 3D run's lateral kernels are "
        "over its own (ForwardTrajectory3D)");
  }
  const AdjointSolver sweeps(*radial_, radialOptions_);
  return sweeps.LateralSweep(*coupling_, options_.viscosity, options_.moduli, times,
                             measurements, trajectory, oceanFunctions);
}

Kernels3D LateralKernelSolver::FirstOrderKernels(
    std::span<const double> times,
    const std::map<std::size_t, Measurement>& measurements,
    const ForwardTrajectory3D& trajectory,
    std::span<const std::vector<double>> oceanFunctions) const {
  if (route_ != Route::Lateral) {
    throw std::invalid_argument(
        "a 3D trajectory for a radial run: the radial run's lateral kernels "
        "are over its own (ForwardTrajectory)");
  }
  auto out = AdjointSolver3D(*lateral_, lateralOptions_)
                 .FirstOrderKernels(times, measurements, &trajectory, oceanFunctions);
  if (!options_.viscosity) {
    out.itauKernel3D.clear();
    out.itauKernelSpherical.clear();
    out.logViscosityKernel3D.clear();
  }
  return out;
}

}  // namespace sl3dpp
