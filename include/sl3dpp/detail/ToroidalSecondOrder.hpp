#pragma once

/**
 * @file detail/ToroidalSecondOrder.hpp
 * @brief The toroidal steps of a base trajectory and of a tangent-linear
 * run about it, formed again from what the two record: what the
 * second-order run and sweeps of a radial model pair and drive with.
 */

#include <span>
#include <vector>

#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/ToroidalViscoelastic.hpp"

namespace sl3dpp::detail {

/// A base trajectory's toroidal part (none: at rest) and the toroidal
/// part of a recorded tangent-linear run about it along a direction in
/// the inverse relaxation times and the shear modulus (each at the
/// memory points; either absent) and in the sources (the run's recorded
/// forces).
struct ToroidalTrajectories {
  using Fields = ToroidalViscoelastic::Fields;
  using State = ToroidalViscoelastic::State;
  using BaseStep = ToroidalViscoelastic::BaseStep;

  const ToroidalViscoelastic* toroidal = nullptr;
  Scheme scheme = Scheme::ExponentialTrapezoid;
  bool discrete = true;
  const ForwardTrajectory* base = nullptr;  // may be null
  std::span<const Evolution::Snapshot> tangent{};
  const std::vector<std::vector<double>>* ditau = nullptr;
  std::span<const double> dmu{};
  State rest{};  // the state at rest, with its strain

  static ToroidalTrajectories Make(
      const ToroidalViscoelastic& toroidal, Scheme scheme, bool discrete,
      const ForwardTrajectory* base, std::span<const Evolution::Snapshot> tangent,
      const std::vector<std::vector<double>>* ditau, std::span<const double> dmu) {
    ToroidalTrajectories out;
    out.toroidal = &toroidal;
    out.scheme = scheme;
    out.discrete = discrete;
    out.base = base;
    out.tangent = tangent;
    out.ditau = ditau;
    out.dmu = dmu;
    out.rest = toroidal.ZeroState();
    toroidal.SetStrain(out.rest, toroidal.SolveElastic(out.rest, {}));
    return out;
  }

  /// Whether a run recorded a toroidal part.
  [[nodiscard]] static bool Twists(std::span<const Evolution::Snapshot> run) {
    return !run.empty() && run.front().toroidalState.has_value();
  }
  [[nodiscard]] bool Sourced() const {
    return base != nullptr && !base->toroidal.empty();
  }

  [[nodiscard]] const State& BaseState(std::size_t node) const {
    return Sourced() ? base->toroidal[node] : rest;
  }
  [[nodiscard]] const State& TangentState(std::size_t node) const {
    return Twists(tangent) ? *tangent[node].toroidalState : rest;
  }
  [[nodiscard]] Fields BaseForce(std::size_t node) const {
    return Sourced() ? base->sourceForces[node].toroidal : Fields{};
  }
  /// The forces of the tangent run's own sources.
  [[nodiscard]] Fields TangentSourceForce(std::size_t node) const {
    return !tangent.empty() && tangent[node].sourceForces
               ? tangent[node].sourceForces->toroidal
               : Fields{};
  }

  [[nodiscard]] Fields BaseStart() const {
    return toroidal->SolveElastic(rest, BaseForce(0));
  }
  [[nodiscard]] Fields TangentStart(const Fields& baseStart) const {
    auto x = toroidal->SolveElastic(rest, TangentSourceForce(0));
    if (!dmu.empty()) {
      for (const auto& [l, w] : toroidal->StartMaterialResponse(dmu, baseStart)) {
        auto& to = x.at(l);
        for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += w[i];
      }
    }
    return x;
  }

  /// The base run's unrelaxed solution at a node, and the tangent run's:
  /// under its state and its sources' forces, with the direction's
  /// unrelaxed material response on the base solution and memory.
  [[nodiscard]] Fields BaseSolution(std::size_t node) const {
    return toroidal->SolveElastic(BaseState(node), BaseForce(node));
  }
  [[nodiscard]] Fields TangentSolution(std::size_t node,
                                       const Fields& baseSolution) const {
    auto x = toroidal->SolveElastic(TangentState(node), TangentSourceForce(node));
    if (!dmu.empty()) {
      for (const auto& [l, w] : toroidal->UnrelaxedMaterialResponse(
               dmu, baseSolution, &BaseState(node))) {
        auto& to = x.at(l);
        for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += w[i];
      }
    }
    return x;
  }

  /// The base step ending at a node.
  [[nodiscard]] BaseStep BaseStepTo(std::size_t node, double dt) const {
    return toroidal->Redo(BaseState(node - 1), dt, scheme, BaseForce(node));
  }
  /// The tangent run's step ending at a node: its decayed memory with
  /// the direction's drive, its solutions under its sources' forces and
  /// the direction's material force on the base step.
  [[nodiscard]] BaseStep TangentStepTo(std::size_t node, double dt,
                                       const BaseStep& baseStep) const {
    auto force = TangentSourceForce(node);
    if (!dmu.empty()) {
      for (auto& [l, f] : toroidal->MaterialForce(dt, scheme, dmu, baseStep)) {
        auto& to = force[l];
        if (to.empty()) {
          to = std::move(f);
        } else {
          for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += f[i];
        }
      }
    }
    auto memory = Fields{};
    if (ditau != nullptr) {
      memory = toroidal->FirstOrderDrive(discrete, scheme, dt,
                                         BaseState(node - 1), BaseState(node),
                                         *ditau);
    }
    return toroidal->Redo(TangentState(node - 1), dt, scheme, force,
                          memory.empty() ? nullptr : &memory);
  }
};

}  // namespace sl3dpp::detail
