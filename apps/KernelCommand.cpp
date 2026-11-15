/**
 * @file KernelCommand.cpp
 * @brief `sl3d kernel`: the adjoint sensitivity kernels of measurements of
 * the problems `solve` solves, read from the same config. The forward run
 * is `solve`'s, built by the same pieces (Common.hpp): the static problem
 * as a run of one node, or the quasi-static problem through time, with an
 * ocean or without one, under any mix of an ice change, surface loads, a
 * tide and seismic sources, radial or, as the `model` block makes it, 3-D
 * (a lateral viscosity, lateral moduli or both). Its snapshots are
 * recorded, in memory or into the trajectory store of the `trajectory`
 * block (Trajectory.hpp), from which a run of the same forward problem
 * sweeps again without running it, and one reverse sweep per measurement
 * of the library's derivative classes (AdjointSolver, AdjointSolver3D)
 * gives the kernels
 * the `kernels` block names: in the ice history, in a surface load, in
 * the tidal potential, in the moment tensors of the run's seismic sources,
 * in the viscosity and in the elastic moduli, each written in SI with the
 * pairing it takes.
 */
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <complex>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <planetmodel/json.hpp>
#include <planetmodel/surfacefile.hpp>
#include <planetmodel/units.hpp>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "Commands.hpp"
#include "Common.hpp"
#include "Input.hpp"
#include "Outputs.hpp"
#include "Ranks.hpp"
#include "Trajectory.hpp"
#include "sl3dpp/Adjoint.hpp"
#include "sl3dpp/Adjoint3D.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Evolution3D.hpp"
#include "sl3dpp/LateralKernels.hpp"
#include "sl3dpp/MemoryPoints.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/KernelWriter.hpp"

namespace sl3dpp::app {

std::string_view NameOf(KernelParameter parameter) {
  switch (parameter) {
    case KernelParameter::Ice:
      return "ice";
    case KernelParameter::Load:
      return "load";
    case KernelParameter::Tide:
      return "tide";
    case KernelParameter::Moment:
      return "moment";
    case KernelParameter::Viscosity:
      return "viscosity";
    case KernelParameter::Moduli:
      return "moduli";
  }
  return "";
}

bool KernelsBlock::Writes(KernelParameter parameter) const {
  return std::ranges::find(parameters, parameter) != parameters.end();
}

namespace {

using Type = MeasurementSpec::Type;

// -- reading -----------------------------------------------------------------

/// The node of the run's times a value names, located against the node
/// list within 1e-9 of the window's span: a date, kyr BP, in a dated run,
/// a time on the run's own clock in its unit in one counting forwards.
/// Refused at `where` when it is not a node.
std::size_t NodeOf(const Json& where, double value, const Run& run) {
  const auto& window = *run.times;
  const auto tolerance = 1e-9 * std::abs(window.Start() - window.End());
  for (auto n = std::size_t{0}; n < window.Size(); ++n) {
    if (std::abs(window.nodes[n] - value) <= tolerance) return n;
  }
  Fail(where, window.Dated()
                  ? std::format("{} kyr BP is not a node of the times grid",
                                value)
                  : std::format("{} {} is not a node of the times grid",
                                value, window.unit.name));
}

/// The member a value came from, or the object where it defaulted.
const Json& MemberOr(const Json& object, std::string_view member) {
  return Has(object, member) ? object.At(member) : object;
}

/// The node a single-instant measurement reads: in a dated run its
/// `date_kyr_bp` (default 0, the present), in a run counting forwards its
/// `time` in the run's unit (default the window's last node), each a node
/// of the run; in the static problem its one node, the instant taking no
/// member. The member of the other form is refused, naming this form's.
std::size_t InstantOf(const Json& item, const Run& run) {
  if (!run.times) {
    for (const auto* name : {"date_kyr_bp", "time"}) {
      if (Has(item, name)) {
        Fail(item.At(name),
             "dates a measurement of the static problem, whose one instant "
             "is the state's: drop it");
      }
    }
    return 0;
  }
  const auto& window = *run.times;
  if (window.Dated()) {
    if (Has(item, "time")) {
      Fail(item.At("time"),
           "is a time on a run's own clock, and the run's nodes are dates "
           "before present: give the measurement's date, `date_kyr_bp`");
    }
    return NodeOf(MemberOr(item, "date_kyr_bp"),
                  Number(item, "date_kyr_bp", 0.0), run);
  }
  if (Has(item, "date_kyr_bp")) {
    Fail(item.At("date_kyr_bp"),
         std::format("is a date before present, and the run counts forwards "
                     "on its own clock: give the measurement's time on it, "
                     "`time`, in {}",
                     window.unit.name));
  }
  if (!Has(item, "time")) return window.Size() - 1;
  return NodeOf(item.At("time"), Number(item, "time"), run);
}

/// A vector displacement's `direction`: its components `east`, `north`
/// and `up` (each optional, zero by default; not all zero).
SurfaceDirection ReadDirection(const Json& member) {
  Allow(member, {"east", "north", "up"});
  const auto out = SurfaceDirection{.east = Number(member, "east", 0.0),
                                    .north = Number(member, "north", 0.0),
                                    .up = Number(member, "up", 0.0)};
  if (out.east == 0.0 && out.north == 0.0 && out.up == 0.0) {
    Fail(member,
         "is the zero vector: give at least one of `east`, `north` and `up`");
  }
  return out;
}

/// A cap's width, `width_deg`, in radians; refused unless positive.
double WidthOf(const Json& item) {
  const auto width = Number(item, "width_deg") * std::numbers::pi / 180.0;
  if (!(width > 0.0)) Fail(item.At("width_deg"), "is not a positive angle");
  return width;
}

/// Whether a measurement reads the sea level, which a run needs an ocean
/// for.
bool OfTheSeaLevel(Type type) {
  return type == Type::RelativeSeaLevelCap ||
         type == Type::RelativeSeaLevelPoint || type == Type::DecayTime;
}

/// The decay time's window and uncertainties, in the form of the run's
/// times (`window_start_kyr_bp`, `window_end_kyr_bp` default 0,
/// `time_uncertainty_kyr` in a dated run; `window_start`, `window_end`
/// default the window's last node, `time_uncertainty` in the run's unit
/// in one counting forwards), held to the nodes: both ends nodes, the
/// start the earlier, and at least four nodes for the fit.
void ReadDecayWindow(const Json& item, const Run& run, MeasurementSpec& out) {
  const auto& window = *run.times;
  constexpr auto kDated = std::array{"window_start_kyr_bp",
                                     "window_end_kyr_bp",
                                     "time_uncertainty_kyr"};
  constexpr auto kForwards =
      std::array{"window_start", "window_end", "time_uncertainty"};
  const auto& mine = window.Dated() ? kDated : kForwards;
  const auto& other = window.Dated() ? kForwards : kDated;
  for (auto k = std::size_t{0}; k < other.size(); ++k) {
    if (Has(item, other[k])) {
      Fail(item.At(other[k]),
           window.Dated()
               ? std::format("is a time on a run's own clock, and the run's "
                             "nodes are dates before present: give `{}`",
                             mine[k])
               : std::format("is a date before present, and the run counts "
                             "forwards on its own clock: give `{}`, in {}",
                             mine[k], window.unit.name));
    }
  }
  const auto start = Number(item, mine[0]);
  const auto end = window.Dated() ? Number(item, mine[1], 0.0)
                                  : Number(item, mine[1], window.End());
  if (window.Dated() ? !(start > end) : !(end > start)) {
    Fail(item, std::format("the decay window runs {} in time",
                           window.Dated() ? "forward" : "backward"));
  }
  out.first = NodeOf(item.At(mine[0]), start, run);
  out.last = NodeOf(MemberOr(item, mine[1]), end, run);
  // the fit (DecayTimeMeasurements) needs four samples
  if (const auto held = out.last - out.first + 1; held < 4) {
    Fail(item, std::format("the decay window holds {} nodes of the times "
                           "grid; the fit needs at least four",
                           held));
  }
  out.time_uncertainty = Number(item, mine[2]);
  out.value_uncertainty_m = Number(item, "value_uncertainty_m");
}

MeasurementSpec ReadSpec(const Json& item, std::size_t index, const Run& run) {
  auto out = MeasurementSpec{};
  out.type = Choice<Type>(
      item.At("type"),
      {{"relative_sea_level_cap", Type::RelativeSeaLevelCap},
       {"relative_sea_level_point", Type::RelativeSeaLevelPoint},
       {"displacement_cap", Type::DisplacementCap},
       {"displacement_point", Type::DisplacementPoint},
       {"displacement_vector_cap", Type::DisplacementVectorCap},
       {"displacement_vector_point", Type::DisplacementVectorPoint},
       {"potential_cap", Type::PotentialCap},
       {"potential_point", Type::PotentialPoint},
       {"stokes_coefficient", Type::StokesCoefficient},
       {"rotation", Type::Rotation},
       {"decay_time", Type::DecayTime}});
  out.word = item.At("type").AsString();
  out.name = String(item, "name", std::format("measurement_{}", index));
  if (out.type == Type::DecayTime && !run.times) {
    Fail(item.At("type"),
         std::format("is the decay time of a series through time, which {}",
                     kNeedsQuasiStatic));
  }
  if (OfTheSeaLevel(out.type) && !run.ocean) {
    Fail(item.At("type"),
         "is a measurement of the sea level, which a run without an ocean "
         "(no `ocean` block) does not have");
  }
  switch (out.type) {
    case Type::RelativeSeaLevelCap:
    case Type::DisplacementCap:
    case Type::PotentialCap:
      Allow(item, {"type", "name", "latitude", "longitude", "width_deg",
                   "date_kyr_bp", "time"});
      out.site = ReadSite(item);
      out.width = WidthOf(item);
      out.node = InstantOf(item, run);
      break;
    case Type::RelativeSeaLevelPoint:
    case Type::DisplacementPoint:
    case Type::PotentialPoint:
      Allow(item, {"type", "name", "latitude", "longitude", "date_kyr_bp",
                   "time"});
      out.site = ReadSite(item);
      out.node = InstantOf(item, run);
      break;
    case Type::DisplacementVectorCap:
      Allow(item, {"type", "name", "latitude", "longitude", "width_deg",
                   "direction", "date_kyr_bp", "time"});
      out.site = ReadSite(item);
      out.width = WidthOf(item);
      out.direction = ReadDirection(item.At("direction"));
      out.node = InstantOf(item, run);
      break;
    case Type::DisplacementVectorPoint:
      Allow(item, {"type", "name", "latitude", "longitude", "direction",
                   "date_kyr_bp", "time"});
      out.site = ReadSite(item);
      out.direction = ReadDirection(item.At("direction"));
      out.node = InstantOf(item, run);
      break;
    case Type::StokesCoefficient:
      Allow(item, {"type", "name", "degree", "order", "date_kyr_bp", "time"});
      out.degree = Integer(item, "degree");
      out.order = Integer(item, "order");
      if (out.degree < 0 || out.degree > run.lmax ||
          std::abs(out.order) > out.degree) {
        Fail(item, std::format("({}, {}) is not a degree and order at "
                               "lmax {}",
                               out.degree, out.order, run.lmax));
      }
      out.node = InstantOf(item, run);
      break;
    case Type::Rotation:
      Allow(item, {"type", "name", "component", "date_kyr_bp", "time"});
      out.component = Choice<AngularVelocityComponent>(
          item.At("component"), {{"x", AngularVelocityComponent::X},
                                 {"y", AngularVelocityComponent::Y},
                                 {"z", AngularVelocityComponent::Z}});
      out.node = InstantOf(item, run);
      break;
    case Type::DecayTime:
      Allow(item, {"type", "name", "latitude", "longitude", "width_deg",
                   "window_start_kyr_bp", "window_end_kyr_bp",
                   "time_uncertainty_kyr", "window_start", "window_end",
                   "time_uncertainty", "value_uncertainty_m"});
      out.site = ReadSite(item);
      out.width = WidthOf(item);
      ReadDecayWindow(item, run, out);
      break;
  }
  return out;
}

/// The words of the kernels a run gives, in the order of KernelParameter,
/// as a refusal lists them: the ice's with an ocean, the load's, the
/// tide's, the moment tensors' under seismic sources, the viscosity's
/// through time and the moduli's.
std::string ParametersOf(const Run& run) {
  using enum KernelParameter;
  auto taken = std::vector<KernelParameter>{};
  if (run.ocean) taken.push_back(Ice);
  taken.push_back(Load);
  taken.push_back(Tide);
  if (!run.forcing.seismic.empty()) taken.push_back(Moment);
  if (run.times) taken.push_back(Viscosity);
  taken.push_back(Moduli);
  auto out = std::string{};
  for (auto k = std::size_t{0}; k < taken.size(); ++k) {
    if (k > 0) out += k + 1 == taken.size() ? " and " : ", ";
    out += std::format("\"{}\"", NameOf(taken[k]));
  }
  return out;
}

/// The kernels a block names, `parameters` (required), held to the run.
std::vector<KernelParameter> ReadParameters(const Json& block, const Run& run) {
  using enum KernelParameter;
  const auto timed = run.times.has_value();
  auto out = std::vector<KernelParameter>{};
  if (!Has(block, "parameters")) {
    Fail(block, std::format("names no `parameters`, the kernels to write: "
                            "list them, of {} (the words this run takes)",
                            ParametersOf(run)));
  }
  const auto& list = block.At("parameters");
  const auto& items = list.AsArray();
  if (items.empty()) Fail(list, "names no kernels");
  for (const auto& item : items) {
    if (item.IsString() && item.AsString() == "seismic") {
      Fail(item,
           "is the forcing's word for the sources: their kernels, in the "
           "point moment tensors, are \"moment\"");
    }
    const auto parameter = Choice<KernelParameter>(
        item, {{"ice", Ice},
               {"load", Load},
               {"tide", Tide},
               {"moment", Moment},
               {"viscosity", Viscosity},
               {"moduli", Moduli}});
    if (std::ranges::find(out, parameter) != out.end()) {
      Fail(item, "names a kernel named before");
    }
    switch (parameter) {
      case Ice:
        if (!run.ocean) {
          Fail(item,
               "is the kernels in the ice, which a run without an ocean (no "
               "`ocean` block) has none of");
        }
        break;
      case Moment:
        if (run.forcing.seismic.empty()) {
          Fail(item,
               "is the kernels in the moment tensors of the run's seismic "
               "sources, and the run has none (`forcing.seismic`)");
        }
        break;
      case Viscosity:
        if (!timed) {
          Fail(item, std::format("is the kernels in the viscosity, which {}: "
                                 "the static problem's Earth is elastic",
                                 kNeedsQuasiStatic));
        }
        break;
      case Load:
      case Tide:
      case Moduli:
        break;
    }
    out.push_back(parameter);
  }
  std::ranges::sort(out);
  return out;
}

}  // namespace

KernelsBlock ReadKernelsBlock(const Json& block, const Run& run,
                              const std::filesystem::path& directory) {
  if (!block.IsObject()) {
    Fail(block,
         "is not a block: `kernels` is an object of `method`, `parameters`, "
         "`measurements` and `output`");
  }
  Allow(block, {"method", "parameters", "lateral", "measurements",
                "record_stride", "trajectory", "output"});
  if (!run.gravitating) {
    Fail(block,
         "asks for kernels, which a run without gravity (`gravity`: "
         "\"none\") does not have");
  }
  // the forward run's recording is the top-level block's, which solve
  // shares
  for (const auto& [name, moved] :
       {std::pair{"record_stride", "trajectory.record_stride"},
        std::pair{"trajectory", "trajectory.directory"}}) {
    if (Has(block, name)) {
      Fail(block.At(name),
           std::format("records the forward run, which is the top-level "
                       "`trajectory` block's, shared with solve: give `{}`",
                       moved));
    }
  }
  auto out = KernelsBlock{};
  if (Has(block, "method")) {
    out.method = Choice<AdjointMethod>(
        block.At("method"), {{"continuous", AdjointMethod::Continuous},
                             {"discrete", AdjointMethod::Discrete}});
  }
  out.parameters = ReadParameters(block, run);
  if (Has(block, "lateral")) {
    out.lateral = block.At("lateral").AsBool();
    if (*out.lateral && !out.Writes(KernelParameter::Viscosity) &&
        !out.Writes(KernelParameter::Moduli)) {
      Fail(block.At("lateral"),
           "asks for the kernels in the viscosity and the moduli as fields over "
           "the Earth, and `parameters` names neither: add \"viscosity\" or "
           "\"moduli\"");
    }
  }
  const auto& measurements = block.At("measurements");
  const auto& items = measurements.AsArray();
  if (items.empty()) Fail(measurements, "holds no measurements");
  for (auto k = std::size_t{0}; k < items.size(); ++k) {
    out.measurements.push_back(ReadSpec(items[k], k, run));
  }
  // A name is the stem of its output files in the output directory, so it
  // must be a plain file name, and one only one measurement uses.
  auto names = std::set<std::string>{};
  for (auto k = std::size_t{0}; k < items.size(); ++k) {
    const auto& name = out.measurements[k].name;
    const auto& where = MemberOr(items[k], "name");
    const auto plain = !name.empty() && name.front() != '.' &&
                       std::ranges::all_of(name, [](char c) {
                         return std::isalnum(static_cast<unsigned char>(c)) ||
                                c == '_' || c == '-' || c == '.';
                       });
    if (!plain) {
      Fail(where, std::format("\"{}\" is not a plain file name (letters, "
                              "digits, '_', '-' and '.', not leading '.'): "
                              "it names the measurement's output files",
                              name));
    }
    if (!names.insert(name).second) {
      Fail(where, std::format("\"{}\" names another measurement too; their "
                              "output files would collide",
                              name));
    }
  }
  out.output = PathOf(block, "output", directory);
  return out;
}

KernelInput ReadKernelInput(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  if (Has(top, "output") && top.At("output").IsString()) {
    Fail(top.At("output"),
         "is a path: the kernel files' directory is `kernels.output`; a "
         "top-level `output` is the block of solve's files (`surface`, "
         "`exchange`, `volume`, `exchange_nodes`)");
  }
  auto allowed = RunMemberNames();
  for (const auto name :
       {"kernels", "output", "checkpoint", "restart", "trajectory"}) {
    allowed.push_back(name);
  }
  AllowTopLevel(top, allowed);
  const auto directory = config.parent_path();
  auto out = KernelInput{};
  out.run = ReadRun(top, directory);
  if (!out.run.gravitating) {
    Fail(top.At("gravity"),
         "is \"none\", and the kernels of a run without gravity are not "
         "built");
  }
  // The kernels are over the window's nodes, each the derivative with
  // respect to a term's value at its node. An onset between two nodes is
  // a pair of the library's own (StepGridOf), where the run reads every
  // term by the term's own rule (a history's knots, a file's time slices,
  // the onset's co-seismic moment), which no fixed combination of its
  // values at the window's nodes gives: the kernels there are not carried
  // onto the window's nodes, and such a run's are not built.
  if (const auto between = OnsetsBetweenNodes(out.run); !between.empty()) {
    const auto& first = between.front();
    const auto& term = out.run.forcing.seismic[first.term];
    const auto& window = *out.run.times;
    Fail(top.At("forcing")
             .At("seismic")
             .At(first.term)
             .At(term.start_kyr_bp ? "onset_kyr_bp" : "onset"),
         std::format("falls between the window's nodes {} and {}, where "
                     "solve takes the earthquake at a node of its own, and "
                     "the kernels about a run with an onset between nodes "
                     "are not built (they are over the window's nodes): give "
                     "the onset as a node of `times`",
                     window.Label(window.Axis(first.node)),
                     window.Label(window.Axis(first.node + 1))));
  }
  out.trajectory = ReadTrajectoryBlock(top, out.run, directory);
  if (!Has(top, "kernels")) {
    Fail(top,
         "has no `kernels` block, which names the measurements, the kernels "
         "written and the directory they go into");
  }
  // solve's members, checked so that one config serves both commands,
  // and left unused
  if (Has(top, "output") || Has(top, "checkpoint") || Has(top, "restart")) {
    static_cast<void>(ReadSolveMembers(top, out.run, directory, false));
  }
  out.kernels = ReadKernelsBlock(top.At("kernels"), out.run, directory);
  return out;
}

namespace {

// -- the measurements against a run ------------------------------------------

/// What one measurement becomes against a run: the node-keyed duals, the
/// forward value, the unit of J (J_SI = factor * J_nd) with its SI
/// dimensions and its unit's name, and, for a decay time, the fit.
struct Built {
  std::map<std::size_t, Measurement> measurements;
  double value_nd = 0.0;
  double factor = 1.0;
  Dimensions objective = {};
  std::string unit;
  std::optional<DecayTimeFit> fit;
  /// The window's nodes the duals sit on; `measurements` is keyed by their
  /// library nodes (StepGrid), the right limit at a repeat.
  std::vector<std::size_t> nodes;
};

/// The surface pairing a^2 int dual field dOmega, nothing where either is
/// absent (a toroidal potential a run does not have, say).
double PairedOrNothing(const SurfaceGrid& grid, double a,
                       std::span<const double> dual,
                       std::span<const double> field) {
  if (dual.empty() || field.empty()) return 0.0;
  return PairedValue(grid, a, dual, field);
}

/// `snapshots` are the library's nodes (`steps`): a measurement dated at a
/// repeated node reads its right limit, the co-seismic field.
template <class Snapshot>
Built BuildMeasurement(const MeasurementSpec& spec, const Run& run,
                       const GRDSolver& solver,
                       std::span<const Snapshot> snapshots,
                       const StepGrid& steps) {
  const auto& grid = solver.Grid();
  const auto a = solver.Radius();
  const auto& scales = solver.SolverScales();
  const auto at = [&steps](std::size_t node) { return steps.library.at(node); };
  auto out = Built{};
  auto single = [&](FieldChannel channel, std::vector<double> dual,
                    auto field_of) {
    out.nodes = {spec.node};
    out.value_nd =
        PairedValue(grid, a, dual, field_of(snapshots[at(spec.node)]));
    out.measurements.emplace(at(spec.node),
                             FieldMeasurement(channel, std::move(dual)));
  };
  const auto sea = [](const Snapshot& s) -> std::span<const double> {
    return s.relativeSeaLevelChange;
  };
  const auto displacement = [](const Snapshot& s) -> std::span<const double> {
    return s.displacement;
  };
  const auto potential = [](const Snapshot& s) -> std::span<const double> {
    return s.potential;
  };
  const auto length = [&] {
    out.factor = scales.Length();
    out.objective = Dimensions{.length = 1};
    out.unit = "m";
  };
  const auto potentialUnit = [&] {
    out.factor = scales.Factor(PotentialDimensions);
    out.objective = PotentialDimensions;
    out.unit = "m^2 s^-2";
  };
  switch (spec.type) {
    case Type::RelativeSeaLevelCap:
    case Type::RelativeSeaLevelPoint:
      length();
      single(FieldChannel::SeaLevel,
             spec.type == Type::RelativeSeaLevelCap
                 ? CapMeanDual(grid, a, spec.site, spec.width)
                 : PointDual(grid, a, spec.site),
             sea);
      break;
    case Type::DisplacementCap:
    case Type::DisplacementPoint:
      length();
      single(FieldChannel::Displacement,
             spec.type == Type::DisplacementCap
                 ? CapMeanDual(grid, a, spec.site, spec.width)
                 : PointDual(grid, a, spec.site),
             displacement);
      break;
    case Type::DisplacementVectorCap:
    case Type::DisplacementVectorPoint: {
      // the radial displacement, and the potentials V and W of the
      // tangential one (W's dual has no degree 1: the displacement in the
      // frame of no net rotation of the surface)
      length();
      auto m = spec.type == Type::DisplacementVectorCap
                   ? CapDisplacementMeasurement(grid, a, spec.site, spec.width,
                                                spec.direction)
                   : PointDisplacementMeasurement(grid, a, spec.site,
                                                  spec.direction);
      const auto& s = snapshots[at(spec.node)];
      out.nodes = {spec.node};
      out.value_nd =
          PairedOrNothing(grid, a, m.displacementDual, s.displacement) +
          PairedOrNothing(grid, a, m.tangentialPotentialDual,
                          s.tangentialPotential) +
          PairedOrNothing(grid, a, m.toroidalPotentialDual,
                          s.toroidalPotential);
      out.measurements.emplace(at(spec.node), std::move(m));
      break;
    }
    case Type::PotentialCap:
    case Type::PotentialPoint:
      potentialUnit();
      single(FieldChannel::Potential,
             spec.type == Type::PotentialCap
                 ? CapMeanDual(grid, a, spec.site, spec.width)
                 : PointDual(grid, a, spec.site),
             potential);
      break;
    case Type::StokesCoefficient:
      potentialUnit();
      single(FieldChannel::Potential,
             CoefficientDual(grid, a, spec.degree, spec.order), potential);
      break;
    case Type::Rotation: {
      out.nodes = {spec.node};
      out.factor = 1.0 / scales.Time();
      out.objective = Dimensions{.time = -1};
      out.unit = "rad s^-1";
      const auto component = static_cast<std::size_t>(spec.component);
      out.value_nd = snapshots[at(spec.node)].angularVelocity[component];
      out.measurements.emplace(at(spec.node),
                               RotationMeasurement(spec.component));
      break;
    }
    case Type::DecayTime: {
      // the site series over the window, in SI metres against the run's
      // unit (kyr in a dated run), the fit's weights becoming node-keyed
      // duals: J in that unit, SI seconds through the factor
      const auto dual = CapMeanDual(grid, a, spec.site, spec.width);
      const auto metres = scales.Length();
      const auto& window = *run.times;
      const auto unit = window.Dated() ? KYR : window.unit.seconds;
      const auto even = window.Dated() && !window.listed;
      const auto step = (window.Start() - window.End()) /
                        (static_cast<int>(window.Size()) - 1);
      auto elapsed = std::vector<double>{};
      auto series = std::vector<double>{};
      auto library = std::vector<std::size_t>{};
      for (auto node = spec.first; node <= spec.last; ++node) {
        elapsed.push_back(even ? step * static_cast<double>(node - spec.first)
                               : (window.Axis(node) - window.Axis(spec.first)) /
                                     unit);
        series.push_back(metres *
                         PairedValue(grid, a, dual, sea(snapshots[at(node)])));
        out.nodes.push_back(node);
        library.push_back(at(node));
      }
      auto fit = DecayTimeFit{};
      out.measurements = DecayTimeMeasurements(
          library, elapsed, series, FieldChannel::SeaLevel, dual,
          {.time_uncertainty = spec.time_uncertainty,
           .value_uncertainty = spec.value_uncertainty_m},
          &fit);
      // the weights are per metre of series: scale the duals so the
      // pairing's J is in the run's unit
      for (auto& [node, measurement] : out.measurements) {
        for (auto& v : measurement.relativeSeaLevelDual) v *= metres;
      }
      out.value_nd = fit.decay_time;
      out.factor = unit;
      out.objective = Dimensions{.time = 1};
      out.unit = "s";
      out.fit = fit;
      break;
    }
  }
  return out;
}

// -- the kernel files ---------------------------------------------------------

/// A point of the run's seismic sources, where its moment kernels are
/// formed: the term it belongs to, its patch (0 for a point source; a
/// fault file's quadrature point), its place (degrees and the depth below
/// the surface in metres) and the point in the solver's units.
struct SourcePoint {
  std::size_t term = 0;
  std::size_t patch = 0;
  double latitude_deg = 0.0;
  double longitude_deg = 0.0;
  double depth_m = 0.0;
  BodyPoint at{};
};

/// The run's seismic sources' points, in the order the forcing gives them
/// (each term's, a fault's patches along strike then down dip, as
/// PlanarFaultSources lays them out, a fault file's quadrature points patch
/// by patch: SeismicPointsOf).
std::vector<SourcePoint> SourcePointsOf(const Run& run,
                                        const GRDSolver& solver) {
  const auto length = solver.SolverScales().Length();
  const auto surface = solver.Radius() * length;
  auto out = std::vector<SourcePoint>{};
  const auto degrees = 180.0 / std::numbers::pi;
  for (auto t = std::size_t{0}; t < run.forcing.seismic.size(); ++t) {
    const auto points = SeismicPointsOf(run.forcing.seismic[t], surface);
    for (auto p = std::size_t{0}; p < points.size(); ++p) {
      const auto& at = points[p];
      out.push_back({.term = t,
                     .patch = p,
                     .latitude_deg = 90.0 - at.colatitude * degrees,
                     .longitude_deg = at.longitude * degrees,
                     .depth_m = surface - at.radius,
                     .at = {.radius = at.radius / length,
                            .colatitude = at.colatitude,
                            .longitude = at.longitude}});
    }
  }
  return out;
}

/// What every kernel file of a run is written with: the config, the run,
/// the solver's grid, scales and radius, each node's place on the files'
/// time axis (SI seconds: relative to the present, negative in the past,
/// in a dated run; on the run's own clock in one counting forwards; none
/// in the static problem), whether the ice views read with the state held
/// (a state at the window's start, or the static problem's), and the
/// seismic sources' points.
struct Writing {
  std::filesystem::path config;
  const Run* run = nullptr;
  const GRDSolver* solver = nullptr;
  std::vector<double> axis;
  bool held = false;
  std::vector<SourcePoint> points;
  /// The library's grid the sweeps ran over (StepGridOf; the static
  /// problem's one node).
  StepGrid steps;
};

/// The first lines of a kernel table: the program, the config, and the
/// measurement with its type and unit.
std::string TableHeader(const Writing& w, const MeasurementSpec& spec,
                        const Built& built) {
  return std::format(
      "# sl3d kernel (sl3dpp {})\n# config {}\n# measurement {} ({}), J in "
      "{}\n",
      Version(), w.config.string(), spec.name, spec.word, built.unit);
}

/// A node's label in a kernel table: its date (kyr BP) in a dated run, its
/// time on the run's clock in its unit in one counting forwards, 0 in the
/// static problem; and the column's name.
double NodeLabel(const Run& run, std::size_t node) {
  return run.times ? run.times->nodes[node] : 0.0;
}
std::string NodeColumn(const Run& run) {
  if (!run.times) return "instant";
  return run.times->Dated() ? "date_kyr_bp"
                            : std::format("time_{}", run.times->unit.name);
}

/// A surface file of kernels on the solver's grid with its provenance,
/// the measurement and the pairing's radius in the metadata; through time
/// with its time axis, the nodes listed as solve's surface file lists
/// them.
planetmodel::SurfaceFile KernelSurfaceFile(const Writing& w,
                                           const MeasurementSpec& spec,
                                           const Built& built,
                                           std::string title) {
  namespace pm = planetmodel;
  const auto& run = *w.run;
  auto file = pm::SurfaceFile{};
  file.grid = AngularGridOf(w.solver->Grid());
  file.source = std::format("sl3d kernel (sl3dpp {})", Version());
  file.title = std::move(title);
  file.meta.emplace("config", w.config.string());
  file.meta.emplace("measurement", spec.name);
  file.meta.emplace("measurement_type", spec.word);
  file.meta.emplace("measurement_unit", built.unit);
  file.meta.emplace("radius_m", std::vector<double>{
                                    w.solver->Radius() *
                                    w.solver->SolverScales().Length()});
  if (!run.times) return file;
  const auto& window = *run.times;
  file.time = w.axis;
  if (window.Dated()) {
    file.meta.emplace("time",
                      std::string{"seconds relative to the present, negative "
                                  "in the past; dates_kyr_bp lists the "
                                  "nodes"});
    file.meta.emplace("dates_kyr_bp", window.nodes);
  } else {
    file.meta.emplace(
        "time", std::format("seconds on the run's own clock, a time of t {} "
                            "in the times block at t * {:.17g} s; times lists "
                            "the nodes in time_unit",
                            window.unit.name, window.unit.seconds));
    file.meta.emplace("times", window.nodes);
    file.meta.emplace("time_unit", std::string{window.unit.name});
  }
  return file;
}

/// Rows of fields laid end to end, each value times `factor`.
std::vector<double> Laid(std::span<const std::vector<double>> rows,
                         double factor) {
  auto values = std::vector<double>{};
  for (const auto& row : rows) {
    for (const auto v : row) values.push_back(v * factor);
  }
  return values;
}

/// Adds a kernel field to a surface file: its support (`node`, `interval`,
/// `instant` or `static`), dimensions and meaning.
void PutKernel(planetmodel::SurfaceFile& file, const std::string& name,
               std::vector<double> values, std::string support,
               const planetmodel::Dimensions& dims, std::string meaning) {
  auto options = planetmodel::SurfaceFieldOptions{};
  options.support = std::move(support);
  options.dimensions = dims;
  options.meaning = std::move(meaning);
  file.variables.emplace(
      name, planetmodel::SurfaceField(name, std::move(values), options));
}

/// The kernels' SI dimensions: the objective's over the perturbation's
/// unit and a square metre of surface.
planetmodel::Dimensions PerSurface(const Dimensions& objective,
                                   const Dimensions& perturbation) {
  return {.mass = objective.mass - perturbation.mass,
          .length = objective.length - perturbation.length - 2,
          .time = objective.time - perturbation.time};
}

void WriteFile(const std::filesystem::path& path,
               const planetmodel::SurfaceFile& file) {
  CreateParent(path);
  planetmodel::WriteSurfaceFile(path, file);
}

/// The ice-history kernels of one measurement as a surface file (see
/// docs/sl3d-configs.md, "kernel"): the thickness duals at the nodes, the
/// quelled rate view on the intervals and the delta amplitudes at the
/// measurement's instants through time, the baseline and initial-sea-level
/// duals as static fields; in the static problem the thickness dual of
/// its one node, static. All SI densities pairing by int K delta dS. With
/// the state held (a state at the window's start, the static problem's)
/// the first node's thickness dual is the library's less its baseline, so
/// that every node's is the derivative with respect to the ice change
/// there.
void WriteIceKernels(const std::filesystem::path& path, const Writing& w,
                     const MeasurementSpec& spec, const Kernels& kernels,
                     const Built& built) {
  const auto& run = *w.run;
  const auto L = w.solver->SolverScales().Length();
  // dJ_nd = a_nd^2 int K_nd dI_nd dOmega with dI_nd = dI_SI / L and
  // dS_SI = L^2 a_nd^2 dOmega, so K_SI = f_J K_nd / L^3; the rate and
  // static duals convert by the same factor
  const auto toKernel = built.factor / (L * L * L);
  const auto dims = PerSurface(built.objective, Dimensions{.length = 1});
  auto file = KernelSurfaceFile(w, spec, built,
                                "ice-history sensitivity kernels");
  auto thickness = kernels.thickness;
  if (w.held) {
    for (auto i = std::size_t{0}; i < thickness[0].size(); ++i) {
      thickness[0][i] -= kernels.baseline[i];
    }
  }
  file.meta.emplace(
      "pairing",
      std::string{
          "dJ = sum_n int K_n dI_n dS over the nodes (thickness_dual), "
          "dS in m^2 on the sphere of radius radius_m; the rates: dJ = "
          "sum_k int K_k (I_{k+1} - I_k) dS over the intervals (rate_dual), "
          "the first node's ice held"});
  file.meta.emplace("ice_views", std::string{w.held ? "state_held"
                                                    : "run_start_moves"});
  const auto first = w.held
                         ? std::string{"the state's ice held, so that it is "
                                       "the derivative with respect to the "
                                       "ice change at each node"}
                         : std::string{"the ice at the other nodes held, the "
                                       "ice the run starts from moving with "
                                       "the first node's, the initial sea "
                                       "level held"};
  if (!run.times) {
    PutKernel(file, "thickness_dual", Laid(thickness, toKernel), "static",
              dims,
              "dJ per metre of grounded ice thickness per m^2, " + first);
  } else {
    PutKernel(file, "thickness_dual", Laid(thickness, toKernel), "node", dims,
              "dJ per metre of grounded ice thickness at each node per m^2, " +
                  first);
    PutKernel(file, "rate_dual", Laid(kernels.rate, toKernel), "interval",
              dims,
              "dJ per metre of ice-thickness change across each interval per "
              "m^2, the first node's ice held (the quelled view: dJ = sum_k "
              "h_k int K_k (dI/dt)_k dS)");
    auto lumps = std::vector<std::vector<double>>{};
    for (const auto node : built.nodes) {
      const auto found = kernels.instantaneous.find(node);
      lumps.push_back(found == kernels.instantaneous.end()
                          ? std::vector<double>(thickness[0].size(), 0.0)
                          : found->second);
    }
    auto instants = std::vector<double>{};
    for (const auto node : built.nodes) instants.push_back(w.axis[node]);
    file.instant = instants;
    PutKernel(file, "instantaneous_dual", Laid(lumps, toKernel), "instant",
              dims,
              "the delta amplitude at each measurement instant of the "
              "thickness kernel's distributional view, per metre of ice "
              "thickness per m^2");
  }
  PutKernel(file, "thickness_baseline_dual",
            Laid(std::span(&kernels.baseline, 1), toKernel), "static", dims,
            w.held ? "dJ per metre of the state's ice thickness per m^2, the "
                     "ice at every node held"
                   : "dJ per metre of the ice the run starts from per m^2, "
                     "the ice at every node held");
  PutKernel(file, "initial_sea_level_dual",
            Laid(std::span(&kernels.initialSeaLevel, 1), toKernel), "static",
            dims,
            "dJ per metre of the initial sea level per m^2, the ice held "
            "(zero under frozen shorelines); the other fields hold the "
            "initial sea level, which a calibrated run re-calibrates: a "
            "change of its ice changes J by them and by this paired with "
            "the calibrated initial sea level's change");
  WriteFile(path, file);
}

/// The kernels in a surface load at each node as a surface file: in the
/// water budget, outside it and in the height of a layer on the solid
/// surface, SI densities pairing by int K delta dS summed over the nodes
/// (static fields in the static problem). Without an ocean there is no
/// water budget, and one load kernel.
void WriteLoadKernels(const std::filesystem::path& path, const Writing& w,
                      const MeasurementSpec& spec, const Kernels& kernels,
                      const Built& built) {
  const auto& run = *w.run;
  const auto& scales = w.solver->SolverScales();
  const auto L = scales.Length();
  // the load: K_SI = f_J K_nd / (L^2 rho_unit L), the load's unit being
  // toLoad; the layer's height as the ice's thickness
  const auto toLoad = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto perLoad = built.factor / (L * L * toLoad);
  const auto perHeight = built.factor / (L * L * L);
  const auto loadDims =
      PerSurface(built.objective, Dimensions{.mass = 1, .length = -2});
  const auto heightDims = PerSurface(built.objective, Dimensions{.length = 1});
  const auto support = std::string{run.times ? "node" : "static"};
  const auto each = std::string{run.times ? " at each node" : ""};
  auto file =
      KernelSurfaceFile(w, spec, built, "surface-load sensitivity kernels");
  file.meta.emplace(
      "pairing",
      std::string{"dJ = sum_n int K_n delta_n dS over the nodes, delta_n the "
                  "change of the load (kg m^-2) or of the layer's height (m) "
                  "at node n, dS in m^2 on the sphere of radius radius_m; a "
                  "layer of density rho whose mass is outside the water "
                  "budget has the kernel rho load_outside_water_budget_dual + "
                  "solid_surface_change_dual in its height"});
  if (!run.ocean) {
    PutKernel(file, "load_dual", Laid(kernels.directLoadKernels, perLoad),
              support, loadDims,
              "dJ per kg m^-2 of surface load" + each + " per m^2");
    WriteFile(path, file);
    return;
  }
  if (kernels.directLoadOutsideWaterBudgetKernels.empty() ||
      kernels.solidSurfaceChangeKernels.empty()) {
    throw std::logic_error(
        "the sweep formed no kernels of a load outside the water budget or "
        "of a layer");
  }
  PutKernel(file, "load_dual", Laid(kernels.directLoadKernels, perLoad),
            support, loadDims,
            "dJ per kg m^-2 of surface load in the water budget" + each +
                " per m^2");
  PutKernel(file, "load_outside_water_budget_dual",
            Laid(kernels.directLoadOutsideWaterBudgetKernels, perLoad), support,
            loadDims,
            "dJ per kg m^-2 of surface load outside the water budget" + each +
                " per m^2");
  PutKernel(file, "solid_surface_change_dual",
            Laid(kernels.solidSurfaceChangeKernels, perHeight), support,
            heightDims,
            "dJ per metre of the height of a layer laid on the solid "
            "surface" +
                each + " per m^2 (its mass a load beside it)");
  WriteFile(path, file);
}

/// The kernels in the tidal potential's coefficients at each node as a
/// text table: the derivatives of J with respect to the real and the
/// imaginary part of each coefficient of degrees 2 to lmax, the members
/// `amplitude_m2_s2` and `imaginary_m2_s2` of a `forcing.tide` term.
void WriteTideKernels(const std::filesystem::path& path, const Writing& w,
                      const MeasurementSpec& spec, const Kernels& kernels,
                      const Built& built) {
  const auto& run = *w.run;
  const auto lmax = w.solver->Grid().LMax();
  // K_SI = f_J K_nd / toPotential: the pairing is over the unit sphere
  const auto toPotential = w.solver->SolverScales().Factor(PotentialDimensions);
  const auto factor = built.factor / toPotential;
  CreateParent(path);
  auto out = std::ofstream(path);
  out << TableHeader(w, spec, built)
      << "# the tide kernels: the derivatives of J with respect to the real "
         "and the imaginary part of the\n"
         "# tidal potential's coefficient of degree l and order m >= 0 at "
         "each node, the members amplitude_m2_s2\n"
         "# and imaginary_m2_s2 of a forcing.tide term (the orthonormal "
         "complex coefficient of a real potential,\n"
         "# Condon-Shortley phase), in J's unit per m^2 s^-2: dJ = sum_n "
         "sum_lm (dJ_damplitude d Re psi_lm(t_n)\n"
         "# + dJ_dimaginary d Im psi_lm(t_n)); a term's coefficient at a "
         "node is its value times its history's factor\n"
         "# there; at m = 0 the imaginary part is zero, and so is its "
         "column\n"
      << std::format("# node {} l m dJ_damplitude dJ_dimaginary\n",
                     NodeColumn(run));
  for (auto n = std::size_t{0}; n < kernels.tideKernels.size(); ++n) {
    const auto& K = kernels.tideKernels[n];
    for (auto l = 2; l <= lmax; ++l) {
      for (auto m = 0; m <= l; ++m) {
        const auto k = K[SurfaceGrid::Index(l, m)] * factor;
        // the pairing of real fields counts the orders m > 0 twice
        const auto weight = m == 0 ? 1.0 : 2.0;
        out << std::format("{} {:.17g} {} {} {:.17g} {:.17g}\n", n,
                           NodeLabel(run, n), l, m, weight * k.real(),
                           m == 0 ? 0.0 : weight * k.imag());
      }
    }
  }
}

/// The kernels in the moment tensors of the run's seismic sources at each
/// node as a text table: the derivatives of J with respect to the members
/// of each point's moment tensor (`rr`, ..., `tp`, the off-diagonal ones
/// setting the symmetric pair) in the basis of `moment_tensor_nm`, SI.
void WriteMomentKernels(const std::filesystem::path& path, const Writing& w,
                        const MeasurementSpec& spec, const Kernels& kernels,
                        const Built& built) {
  const auto& run = *w.run;
  // K_SI = f_J K_nd / (the solver's unit of energy, N m)
  const auto energy = w.solver->SolverScales().Factor(
      Dimensions{.mass = 1, .length = 2, .time = -2});
  const auto factor = built.factor / energy;
  CreateParent(path);
  auto out = std::ofstream(path);
  out << TableHeader(w, spec, built)
      << "# the moment kernels: the derivatives of J with respect to the "
         "members rr, tt, pp, rt, rp, tp of the\n"
         "# moment tensor of each source point at each node (an "
         "off-diagonal member setting the symmetric pair),\n"
         "# in the spherical polar basis at the point (r up, theta south, "
         "phi east), in J's unit per N m:\n"
         "# dJ = sum_n sum_p sum_c dJ_dM_c(n, p) dM_c(p, t_n); a term's "
         "moment at a node is its tensor times its\n"
         "# factor there (zero before its onset, its co-seismic moment at "
         "it; a fault file's on a time axis, its slices as its rule reads "
         "them),\n"
         "# a fault's patches and a fault file's quadrature points each a "
         "point of its own\n"
         "# the points: point term patch latitude_deg longitude_deg depth_m\n";
  for (auto p = std::size_t{0}; p < w.points.size(); ++p) {
    const auto& point = w.points[p];
    out << std::format("# point {} {} {} {:.17g} {:.17g} {:.17g}\n", p,
                       point.term, point.patch, point.latitude_deg,
                       point.longitude_deg, point.depth_m);
  }
  out << std::format(
      "# node {} point dJ_dMrr dJ_dMtt dJ_dMpp dJ_dMrt dJ_dMrp dJ_dMtp\n",
      NodeColumn(run));
  for (auto n = std::size_t{0}; n < kernels.momentKernels.size(); ++n) {
    for (auto p = std::size_t{0}; p < kernels.momentKernels[n].size(); ++p) {
      const auto& K = kernels.momentKernels[n][p];
      // K : dM counts the off-diagonal components twice
      out << std::format(
          "{} {:.17g} {} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g}\n", n,
          NodeLabel(run, n), p, factor * K.rr, factor * K.tt, factor * K.pp,
          2.0 * factor * K.rt, 2.0 * factor * K.rp, 2.0 * factor * K.tp);
    }
  }
}

/// Kernels over the library's nodes (Writing::steps) as kernels over the
/// window's. At a repeated node both sides read the ice, the loads, the
/// tide and every seismic term without its onset there at the node's one
/// time, so a change of any of them there pairs with the sum of the two
/// sides' kernels; a term whose onset falls there has, at the node, its
/// co-seismic moment, the right side's (the left limit its absence), and
/// its points take the right side's kernel alone. The quelled view's
/// zero-length interval goes (no ice changes across it), the delta
/// amplitudes are keyed by the window's node, and the density, per unit
/// time, is the two sides' weighted by the half-steps each pairs with. A
/// pair inserted between nodes is refused before the run
/// (ReadKernelInput).
Kernels OnTheWindow(const Kernels& kernels, const Writing& w) {
  const auto& steps = w.steps;
  if (!steps.Repeats()) return kernels;
  if (std::ranges::contains(steps.window, StepGrid::kInserted)) {
    throw std::logic_error(
        "the kernels about a run with an onset between nodes are not built");
  }
  const auto& run = *w.run;
  const auto onClock = NodesOnTheClock(*run.times);
  const auto nodes = steps.library.size();
  auto out = kernels;
  auto summed = [&](const auto& perNode) {
    using Row = std::decay_t<decltype(perNode.front())>;
    auto rows = std::vector<Row>{};
    if (perNode.empty()) return rows;
    for (auto n = std::size_t{0}; n < nodes; ++n) {
      const auto k = steps.library[n];
      auto row = perNode[k];
      if (k > 0 && steps.Left(k - 1)) {
        for (auto i = std::size_t{0}; i < row.size(); ++i) {
          row[i] += perNode[k - 1][i];
        }
      }
      rows.push_back(std::move(row));
    }
    return rows;
  };
  out.directLoadKernels = summed(kernels.directLoadKernels);
  out.directLoadOutsideWaterBudgetKernels =
      summed(kernels.directLoadOutsideWaterBudgetKernels);
  out.solidSurfaceChangeKernels = summed(kernels.solidSurfaceChangeKernels);
  out.tideKernels = summed(kernels.tideKernels);
  out.thickness = summed(kernels.thickness);
  out.times.clear();
  out.momentKernels.clear();
  out.density.clear();
  out.rate.clear();
  for (auto n = std::size_t{0}; n < nodes; ++n) {
    const auto k = steps.library[n];
    const auto repeated = k > 0 && steps.Left(k - 1);
    out.times.push_back(steps.times[k]);
    if (!kernels.momentKernels.empty()) {
      auto row = kernels.momentKernels[k];
      if (repeated) {
        // the node on the terms' clock, where StepGridOf found the onset
        const auto t = onClock[n];
        for (auto p = std::size_t{0}; p < row.size(); ++p) {
          if (run.forcing.seismic[w.points[p].term].OnsetAt(t)) continue;
          const auto& left = kernels.momentKernels[k - 1][p];
          row[p].rr += left.rr;
          row[p].tt += left.tt;
          row[p].pp += left.pp;
          row[p].rt += left.rt;
          row[p].rp += left.rp;
          row[p].tp += left.tp;
        }
      }
      out.momentKernels.push_back(std::move(row));
    }
    if (!kernels.density.empty()) {
      auto row = kernels.density[k];
      if (repeated) {
        // the left side pairs with the half-step before, the right with
        // the half-step after
        const auto before = steps.times[k - 1] - steps.times[k - 2];
        const auto after = k + 1 < steps.times.size()
                               ? steps.times[k + 1] - steps.times[k]
                               : 0.0;
        for (auto i = std::size_t{0}; i < row.size(); ++i) {
          row[i] = (before * kernels.density[k - 1][i] + after * row[i]) /
                   (before + after);
        }
      }
      out.density.push_back(std::move(row));
    }
    if (n > 0 && !kernels.rate.empty()) {
      out.rate.push_back(kernels.rate[k - (repeated ? 2 : 1)]);
    }
  }
  out.instantaneous.clear();
  for (const auto& [k, lump] : kernels.instantaneous) {
    auto& to = out.instantaneous[steps.window[k]];
    if (to.empty()) {
      to = lump;
    } else {
      for (auto i = std::size_t{0}; i < to.size(); ++i) to[i] += lump[i];
    }
  }
  return out;
}

/// The kernel files every dimension shares, as the block names them: the
/// ice's, the load's, the tide's and the moments', over the window's nodes
/// (OnTheWindow).
void WriteSurfaceKernels(const std::filesystem::path& stem, const Writing& w,
                         const KernelsBlock& block,
                         const MeasurementSpec& spec, const Kernels& library,
                         const Built& built) {
  const auto kernels = OnTheWindow(library, w);
  const auto named = [&stem](std::string_view suffix) {
    return std::filesystem::path(stem.string() + std::string{suffix});
  };
  if (block.Writes(KernelParameter::Ice)) {
    WriteIceKernels(named("_ice_kernels.nc"), w, spec, kernels, built);
  }
  if (block.Writes(KernelParameter::Load)) {
    WriteLoadKernels(named("_load_kernels.nc"), w, spec, kernels, built);
  }
  if (block.Writes(KernelParameter::Tide)) {
    if (kernels.tideKernels.empty()) {
      throw std::logic_error("the sweep formed no tide kernels");
    }
    WriteTideKernels(named("_tide_kernels.txt"), w, spec, kernels, built);
  }
  if (block.Writes(KernelParameter::Moment)) {
    if (kernels.momentKernels.empty()) {
      throw std::logic_error("the sweep formed no moment kernels");
    }
    WriteMomentKernels(named("_moment_kernels.txt"), w, spec, kernels, built);
  }
}

/// The radial viscosity kernels of each Maxwell branch as a text table,
/// densities per unit volume in SI at the mesh's nodes, element by
/// element, with each row's volume weight.
void WriteViscosityKernel1D(const std::filesystem::path& path,
                            const Writing& w, const MeasurementSpec& spec,
                            const Built& built, const Kernels& kernels,
                            const Rheology& rheology,
                            const Material& material) {
  const auto& mesh = material.Mesh();
  const auto& matScales = material.ModelScales();
  // dJ = int K d eta dV = 4 pi int K(r) d eta(r) r^2 dr per branch
  const auto eta = kernels.Viscosity(rheology);
  const auto etaSi = matScales.Factor(ViscosityDimensions);
  const auto rSi = matScales.Length();
  const auto weights =
      MemoryPoints::Of(MemoryPointSet::Gll, mesh).VolumeWeights(mesh);
  auto out = std::ofstream(path);
  out << TableHeader(w, spec, built)
      << "# K per unit volume: dJ = int K d eta dV = 4 pi int K(r) d eta(r) "
         "r^2 dr = sum_rows W_m3 K d eta, SI (J in its SI unit, eta in Pa "
         "s)\n"
         "# the rows are the radial mesh's nodes, element by element (a "
         "shared edge twice, once per element),\n"
         "# W_m3 the node's volume weight (4 pi r^2 times its radial "
         "quadrature weight)\n"
      << "# r_m";
  for (auto b = std::size_t{0}; b < eta.size(); ++b) {
    out << std::format(" K_branch_{}", b);
  }
  out << " W_m3\n";
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    for (auto q = 0; q < mesh.NGll(); ++q) {
      const auto i = static_cast<std::size_t>(e) * mesh.NGll() + q;
      out << std::format("{:.17g}", mesh.Radius(e, q) * rSi);
      for (auto b = std::size_t{0}; b < eta.size(); ++b) {
        out << std::format(" {:.17g}",
                           built.factor * eta[b][i] /
                               (etaSi * rSi * rSi * rSi));
      }
      out << std::format(" {:.17g}\n", weights[i] * rSi * rSi * rSi);
    }
  }
}

/// The radial kernels in the bulk and the unrelaxed shear modulus as a
/// text table at the memory points, element by element, densities per
/// unit volume in SI with each point's radius, volume weight and moduli.
void WriteModuliKernel1D(const std::filesystem::path& path, const Writing& w,
                         const MeasurementSpec& spec, const Built& built,
                         const Kernels& kernels,
                         const ViscoelasticOperator& op,
                         const Material& material) {
  const auto& mesh = material.Mesh();
  const auto& matScales = material.ModelScales();
  const auto& points = op.Points();
  const auto radii = points.Radii(mesh);
  const auto weights = points.VolumeWeights(mesh);
  const auto shear = op.UnrelaxedShearAtPoints();
  const auto bulk = op.BulkModulusAtPoints();
  const auto pa = matScales.Factor(ModulusDimensions);
  const auto rSi = matScales.Length();
  const auto toKernel = built.factor / (pa * rSi * rSi * rSi);
  if (kernels.kappaKernel.size() != radii.size() ||
      kernels.muKernel.size() != radii.size()) {
    throw std::logic_error("the moduli kernels are not at the memory points");
  }
  auto out = std::ofstream(path);
  out << TableHeader(w, spec, built)
      << "# the kernels in the bulk modulus kappa and the unrelaxed shear "
         "modulus mu, densities per unit volume, SI:\n"
         "# dJ = int (K_kappa d kappa + K_mu d mu) dV = sum_rows W_m3 "
         "(K_kappa d kappa + K_mu d mu), d kappa and d mu the\n"
         "# absolute perturbations at the point, Pa (the branch moduli as "
         "fixed fractions of the unrelaxed one,\n"
         "# the relaxation times held); a relative perturbation d ln mu "
         "pairs as K_mu mu_Pa\n"
         "# the rows are the memory points, element by element, W_m3 the "
         "point's volume weight\n"
         "# r_m W_m3 kappa_Pa mu_Pa K_kappa K_mu\n";
  for (auto i = std::size_t{0}; i < radii.size(); ++i) {
    out << std::format("{:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g}\n",
                       radii[i] * rSi, weights[i] * rSi * rSi * rSi,
                       bulk[i] * pa, shear[i] * pa,
                       toKernel * kernels.kappaKernel[i],
                       toKernel * kernels.muKernel[i]);
  }
}

// -- the runs -----------------------------------------------------------------

/// The sweeps of every measurement over a run's snapshots, and the
/// measurements' values: `sweep` forms one measurement's kernels and
/// writes their files under its stem (every rank sweeps, the first alone
/// writes). Returns the summary, `read` saying what the config was taken
/// to mean with the inputs as read.
template <class Snapshot>
std::string SweepEach(
    const KernelInput& input, const Writing& w,
    std::span<const Snapshot> snapshots, std::string_view iceRead,
    std::string_view seaLevelRead, std::string_view read,
    const std::function<void(const MeasurementSpec&, const Built&,
                             const std::filesystem::path& stem)>& sweep) {
  const auto& run = input.run;
  const auto& block = input.kernels;
  const auto writes = TheRanks().First();
  if (writes) std::filesystem::create_directories(block.output);
  auto values = writes ? std::ofstream(block.output / "measurements.txt")
                       : std::ofstream{};
  values << std::format("# sl3d kernel (sl3dpp {})\n", Version())
         << std::format("# created {}\n", CreatedNow())
         << std::format("# config {}\n", w.config.string());
  if (run.ocean) {
    values << std::format("# ice {}\n", iceRead)
           << std::format("# sea_level {}\n", seaLevelRead);
  } else {
    values << "# ocean none\n";
  }
  if (const auto change = DescribeIceChange(run, "; "); !change.empty()) {
    values << std::format("# ice_change {}\n", change);
  }
  const auto add = [&values](std::string_view kind, const auto& terms) {
    for (const auto& term : terms) {
      values << std::format("# {} {}\n", kind, Describe(term));
    }
  };
  add("load", run.forcing.load);
  add("tide", run.forcing.tide);
  add("seismic", run.forcing.seismic);
  const auto unit =
      run.times && !run.times->Dated() ? run.times->unit.name : "kyr";
  values << std::format(
      "# name type value_si [decay: tau_{} amplitude_m offset_m]\n", unit);
  for (const auto& spec : block.measurements) {
    const auto built = BuildMeasurement<Snapshot>(spec, run, *w.solver,
                                                  snapshots, w.steps);
    sweep(spec, built, block.output / spec.name);
    values << spec.name << ' ' << spec.word << ' '
           << std::format("{:.17g}", built.factor * built.value_nd);
    if (built.fit) {
      // the fit ran on the series in SI metres (BuildMeasurement), so its
      // amplitude and offset are metres already
      values << std::format(" {:.17g} {:.17g} {:.17g}", built.fit->decay_time,
                            built.fit->amplitude, built.fit->offset);
    }
    values << '\n';
  }
  auto names = std::string{};
  for (const auto parameter : block.parameters) {
    if (!names.empty()) names += ", ";
    names += NameOf(parameter);
  }
  return std::format(
      "{}; wrote {} measurement(s) of kernels to {} ({}; {} adjoint{})", read,
      block.measurements.size(), block.output.string(), names,
      block.method == AdjointMethod::Discrete ? "discrete" : "continuous",
      block.lateral.value_or(false) ? "; the kernels in the Earth as fields"
                                    : "");
}

/// The adjoint's options of the kernels asked for: the method and the
/// tolerance, the moduli's kernels, the moment kernels at the sources'
/// points and the tide kernels.
template <class Options>
Options AdjointOptionsOf(const KernelInput& input, const Writing& w) {
  const auto& block = input.kernels;
  auto options = Options{};
  options.method = block.method;
  options.rtol = input.run.solver.rtol;
  options.material_kernels = block.Writes(KernelParameter::Moduli);
  options.tide_kernels = block.Writes(KernelParameter::Tide);
  if (block.Writes(KernelParameter::Moment)) {
    for (const auto& point : w.points) {
      options.moment_points.push_back(point.at);
    }
  }
  return options;
}

/// What a kernel run writes beside the measurement: its config, run,
/// solver, time axis, ice views and seismic points.
Writing WritingOf(const std::filesystem::path& config, const Run& run,
                  const GRDSolver& solver) {
  auto w = Writing{.config = config,
                   .run = &run,
                   .solver = &solver,
                   .axis = {},
                   .held = false,
                   .points = {},
                   .steps = {.times = {0.0}, .window = {0}, .library = {0}}};
  if (run.times) {
    for (auto n = std::size_t{0}; n < run.times->Size(); ++n) {
      w.axis.push_back(run.times->Axis(n));
    }
    w.steps = StepGridOf(run, solver.SolverScales());
  }
  w.held = run.ocean && (!run.times || run.ocean->at == WindowEnd::Start);
  w.points = SourcePointsOf(run, solver);
  return w;
}

/// The whole mesh's coupling of a 3-D operator distributed over the ranks,
/// on the first rank (null elsewhere and on one rank): the layout of the
/// ranks' parts laid end to end, which the kernel files are written over.
std::optional<GridCoupling> WholeCoupling(const Viscoelastic3D& op, int lmax) {
  const auto& ranks = TheRanks();
  auto whole = std::optional<GridCoupling>{};
  if (!ranks.Distributed() || !ranks.First()) return whole;
  const auto& o = op.GetOptions();
  whole.emplace(op.GetRheology(), lmax, op.Coupling().Grid(),
                GridCoupling::Options{
                    .eps = o.eps,
                    .truncation = o.truncation,
                    .oversampling = o.oversampling,
                    .parallel = o.parallel,
                    .matrix_kernel = o.matrix_kernel,
                    .cache_bytes = o.cache_bytes,
                    .memory_points = o.memory_points,
                    .transform_degrees = o.transform_degrees,
                    .active_elements = Viscoelastic3D::ActiveElementsFor(
                        op.GetRheology(), lmax, op.Coupling().Grid(), o)});
  return whole;
}

/// The volume pairing of a 3-D kernel with a field over this rank's memory
/// nodes, summed over the ranks: nothing where either is empty (a field
/// the rank's segment does not carry).
double PairedOverRanks(const GridCoupling& coupling,
                       std::span<const double> kernel,
                       std::span<const double> field) {
  auto total = kernel.empty() || field.empty()
                   ? 0.0
                   : VolumePairing(coupling, kernel, field);
  if (const auto& ranks = TheRanks(); ranks.Distributed()) {
    ranks.collectives.sum(std::span(&total, 1));
  }
  return total;
}

/// The field kernels of one measurement as planetmodel model files over a
/// coupling's memory nodes (gathered to the first rank on several, over the
/// whole mesh's coupling `whole` there): the log-viscosity kernels of each
/// branch, with their pairings against the run's own lateral log
/// viscosity `delta` and against a uniform one (ln eta raised by one
/// wherever the coupling carries nodes), and the moduli kernels, with
/// their pairing against the run's own lateral moduli (empty where it has
/// none), in the metadata. The values in SI under the model's scales.
void WriteFieldKernels(const std::filesystem::path& stem,
                       const KernelsBlock& block, const MeasurementSpec& spec,
                       const Built& built, const Kernels3D& kernels,
                       const GridCoupling& coupling, const GridCoupling* whole,
                       const RadialMesh& mesh, const BuiltEarth& earth,
                       std::span<const double> delta,
                       std::span<const double> shearPerturbation,
                       std::span<const double> bulkPerturbation) {
  const auto& ranks = TheRanks();
  const auto& matScales = earth.material.ModelScales();
  // the values non-dimensional under the model's own scales, the
  // writer's
  const auto toModel = built.factor / matScales.Factor(built.objective);
  const auto write = [&](const std::string& suffix,
                         std::vector<io::NodeField> fields,
                         io::KernelFileOptions file) {
    if (!ranks.First()) return;
    const auto path = std::filesystem::path(stem.string() + suffix);
    const auto written = io::WriteNodeFields(path, whole ? *whole : coupling,
                                             mesh, earth.model, fields, file);
    if (!written) throw std::runtime_error(written.error().message);
  };
  const auto scaled = [&](std::vector<double> values) {
    for (auto& v : values) v *= toModel;
    if (ranks.Distributed()) values = GatherToFirst(values);
    return values;
  };
  if (block.Writes(KernelParameter::Viscosity)) {
    // the pairings with the run's lateral log-viscosity perturbation
    // and with a uniform one, ln eta raised by one wherever the
    // coupling carries memory nodes
    auto lateral = 0.0, uniform = 0.0;
    const auto ones = std::vector<double>(coupling.GridValueCount(), 1.0);
    auto kept = std::vector<std::vector<double>>{};
    auto fields = std::vector<io::NodeField>{};
    for (auto b = std::size_t{0}; b < kernels.logViscosityKernel3D.size(); ++b) {
      lateral += PairedOverRanks(coupling, kernels.logViscosityKernel3D[b], delta);
      uniform += PairedOverRanks(coupling, kernels.logViscosityKernel3D[b], ones);
      kept.push_back(scaled(kernels.logViscosityKernel3D[b]));
    }
    for (auto b = std::size_t{0}; b < kept.size(); ++b) {
      fields.push_back(io::NodeField{
          .name = std::format("K_ln_eta_{}", b),
          .meaning = std::format(
              "dJ per unit ln(viscosity) of branch {}, per unit volume", b),
          .dimensions = Dimensions{.mass = built.objective.mass,
                                   .length = built.objective.length - 3,
                                   .time = built.objective.time},
          .values = kept[b]});
    }
    write("_viscosity_kernels.nc", std::move(fields),
          {.name = std::format("viscosity kernels of {}", spec.name),
           .meta = {{"lateral_log_viscosity_pairing", built.factor * lateral},
                    {"uniform_log_viscosity_pairing", built.factor * uniform}}});
  }
  if (block.Writes(KernelParameter::Moduli)) {
    if (kernels.kappaKernel3D.size() != coupling.GridValueCount() ||
        kernels.muKernel3D.size() != coupling.GridValueCount()) {
      throw std::logic_error(
          "the sweep formed no moduli kernels over the coupling");
    }
    const auto pairing =
        PairedOverRanks(coupling, kernels.muKernel3D, shearPerturbation) +
        PairedOverRanks(coupling, kernels.kappaKernel3D, bulkPerturbation);
    const auto kappa = scaled(kernels.kappaKernel3D);
    const auto mu = scaled(kernels.muKernel3D);
    // per Pa of the modulus and per cubic metre
    const auto dims = Dimensions{.mass = built.objective.mass - 1,
                                 .length = built.objective.length + 1 - 3,
                                 .time = built.objective.time + 2};
    write("_moduli_kernels.nc",
          {io::NodeField{.name = "K_kappa",
                         .meaning = "dJ per Pa of the bulk modulus "
                                    "(absolute perturbation), per unit "
                                    "volume",
                         .dimensions = dims,
                         .values = kappa},
           io::NodeField{.name = "K_mu",
                         .meaning = "dJ per Pa of the unrelaxed shear "
                                    "modulus (absolute perturbation, "
                                    "the branch moduli as fixed "
                                    "fractions of it), per unit volume",
                         .dimensions = dims,
                         .values = mu}},
          {.name = std::format("moduli kernels of {}", spec.name),
           .meta = {{"lateral_moduli_pairing", built.factor * pairing}}});
  }
}

/// The kernels over a recorded radial run (the static problem's one
/// node, or a run through time), its snapshots from `evolution` on the
/// operator's rheology: the radial profiles of the viscosity and moduli
/// kernels, and with `kernels.lateral` their fields over the memory nodes'
/// grids too (LateralKernelSolver's radial route), written as a 3-D run's
/// are, the run's own lateral structure none.
std::string Sweep1D(const KernelInput& input, const Writing& w,
                    const Evolution& evolution,
                    std::span<const Evolution::Snapshot> snapshots,
                    std::span<const double> times,
                    const ViscoelasticOperator& op, const Rheology& rheology,
                    const BuiltEarth& earth, std::string_view iceRead,
                    std::string_view seaLevelRead, std::string_view read) {
  const auto& block = input.kernels;
  const auto& material = earth.material;
  const auto trajectory = ForwardTrajectory::FromSnapshots(snapshots);
  // about a migrating run, the sweeps linearise about its own shorelines
  const auto oceans = input.run.Migrating()
                          ? OceanFunctionsOf(snapshots)
                          : std::vector<std::vector<double>>{};
  const auto options = AdjointOptionsOf<AdjointSolver::Options>(input, w);
  const AdjointSolver adjoint(evolution, options);
  auto lateral = std::optional<LateralKernelSolver>{};
  if (block.lateral.value_or(false)) {
    const auto& solver = input.run.solver;
    lateral.emplace(evolution, options,
                    LateralKernelOptions{
                        .viscosity = block.Writes(KernelParameter::Viscosity),
                        .moduli = block.Writes(KernelParameter::Moduli),
                        .transform_degrees = solver.transform_degrees,
                        .cache_bytes = solver.cache_mb << 20});
  }
  // the run's lateral log viscosity at the grid values (zero: a radial
  // viscosity file holds none), for the viscosity kernels' pairing with it
  const auto delta =
      lateral && block.Writes(KernelParameter::Viscosity)
          ? io::SampleNodeField(lateral->Coupling(), DeltaOf(earth))
          : std::vector<double>{};
  return SweepEach<Evolution::Snapshot>(
      input, w, snapshots, iceRead, seaLevelRead, read,
      [&](const MeasurementSpec& spec, const Built& built,
          const std::filesystem::path& stem) {
        auto fields = std::optional<Kernels3D>{};
        if (lateral) {
          fields = lateral->FirstOrderKernels(times, built.measurements, trajectory,
                                              oceans);
        }
        const auto kernels =
            fields ? fields->ice
                   : adjoint.FirstOrderKernels(times, built.measurements, &trajectory,
                                               oceans);
        WriteSurfaceKernels(stem, w, block, spec, kernels, built);
        if (block.Writes(KernelParameter::Viscosity)) {
          WriteViscosityKernel1D(stem.string() + "_viscosity_kernel.txt", w,
                                 spec, built, kernels, rheology, material);
        }
        if (block.Writes(KernelParameter::Moduli)) {
          WriteModuliKernel1D(stem.string() + "_moduli_kernel.txt", w, spec,
                              built, kernels, op, material);
        }
        if (fields) {
          WriteFieldKernels(stem, block, spec, built, *fields, lateral->Coupling(),
                            nullptr, op.BaseMaterial().Mesh(), earth, delta, {}, {});
        }
      });
}

/// The 3-D kernels over a recorded 3-D run: the ice-history, load, tide
/// and moment kernels as the 1-D ones, the log-viscosity kernels and the
/// moduli kernels as planetmodel model files over the coupling's memory
/// nodes (gathered to the first rank on several), each with its pairing
/// against the run's own lateral structure (the log viscosity's against a
/// uniform perturbation too) in the metadata.
std::string Sweep3D(const KernelInput& input, const Writing& w,
                    const Evolution3D& evolution,
                    std::span<const Evolution3D::Snapshot> snapshots,
                    const ForwardTrajectory3D& trajectory,
                    std::span<const double> times, const Viscoelastic3D& op,
                    const BuiltEarth& earth, std::string_view iceRead,
                    std::string_view seaLevelRead, std::string_view read) {
  const auto& run = input.run;
  const auto& block = input.kernels;
  const auto& ranks = TheRanks();
  const auto oceans = run.Migrating() ? OceanFunctionsOf(snapshots)
                                      : std::vector<std::vector<double>>{};
  auto options = AdjointOptionsOf<AdjointSolver3D::Options>(input, w);
  options.max_coupled_iterations = run.solver.step_max_iterations;
  options.coupled_rtol = run.solver.step_rtol;
  const AdjointSolver3D adjoint(evolution, options);
  const auto whole = WholeCoupling(op, run.lmax);
  const auto& coupling = op.Coupling();
  const auto& mesh = op.GetRheology().BaseMaterial().Mesh();
  // the run's own log-viscosity perturbation at the grid values, for the
  // viscosity kernels' pairing with it
  const auto delta = block.Writes(KernelParameter::Viscosity)
                         ? io::SampleNodeField(coupling, DeltaOf(earth))
                         : std::vector<double>{};
  return SweepEach<Evolution3D::Snapshot>(
      input, w, snapshots, iceRead, seaLevelRead, read,
      [&](const MeasurementSpec& spec, const Built& built,
          const std::filesystem::path& stem) {
        const auto kernels = adjoint.FirstOrderKernels(
            times, built.measurements, &trajectory, oceans);
        if (ranks.First()) {
          WriteSurfaceKernels(stem, w, block, spec, kernels.ice, built);
        }
        WriteFieldKernels(stem, block, spec, built, kernels, coupling,
                          whole ? &*whole : nullptr, mesh, earth, delta,
                          op.UnrelaxedShearPerturbation(), op.BulkPerturbation());
      });
}

/// Refuses on several ranks what a 3-D kernel run forms on one process:
/// the kernels about a run under seismic sources (whose moment kernels the
/// sweeps form on one process), and the tide kernels of the continuous
/// sweep.
void RefuseOnRanks3D(const Json& top, const KernelInput& input) {
  const auto& ranks = TheRanks();
  if (!ranks.Distributed()) return;
  if (!input.run.forcing.seismic.empty()) {
    Fail(top.At("forcing").At("seismic"),
         std::format("gives seismic sources, and the 3-D kernels about a run "
                     "under them are formed on one process: run on one rank "
                     "(this run has {})",
                     ranks.size));
  }
  const auto& block = input.kernels;
  if (block.Writes(KernelParameter::Tide) &&
      block.method == AdjointMethod::Continuous) {
    const auto& list = top.At("kernels").At("parameters");
    for (const auto& item : list.AsArray()) {
      if (item.AsString() != "tide") continue;
      Fail(item, std::format("asks the tide kernels of the continuous 3-D "
                             "sweep, which are formed on one process: run on "
                             "one rank (this run has {}), or give "
                             "`kernels.method`: \"discrete\"",
                             ranks.size));
    }
  }
}

/// Where a kernel run's forward run comes from: the trajectory store of
/// `trajectory.directory` that holds this run complete, read back; that
/// store recorded by this run where the directory holds none (or an
/// incomplete one of this run); or, without a directory, the run in memory.
/// `run` is the run's description for the store.
struct Forward {
  std::optional<io::TrajectoryStore> store;  ///< none without a directory
  bool read = false;  ///< swept from a complete store, not run

  /// Surveys `trajectory.directory`, where given, for the run `run`
  /// describes (called only then), and opens its store for reading or
  /// for recording; the store's refusals named at the member.
  Forward(const Json& top, const TrajectoryBlock& block,
          const std::function<io::TrajectoryStore::Description()>& run) {
    if (block.directory.empty()) return;
    const auto described = run();
    if (SurveyTrajectory(top, block, described) ==
        io::TrajectoryStore::Standing::Complete) {
      store = OpenTrajectory(top, block, described);
      read = true;
    } else {
      store = RecordTrajectory(top, block, described);
    }
  }
  /// Whether this run records into the store.
  [[nodiscard]] bool Records() const { return store && !read; }

  /// The summary's account of the store, after the sweeps' (empty
  /// without one).
  [[nodiscard]] std::string Note() const {
    if (!store) return {};
    const auto at = std::format("the trajectory store {} ({})",
                                store->Directory().string(),
                                SizeOf(store->Bytes()));
    return read ? std::format("; the forward run read back from {}", at)
                : std::format("; the forward run recorded to {}", at);
  }
};

/// The kernels of the static problem on a radial Earth: the one-node run
/// of the radial evolution on a branchless rheology, recorded, or read
/// from the trajectory store of the run.
std::string KernelStatic1D(const std::filesystem::path& config,
                           const KernelInput& input, const Json& top,
                           const BuiltEarth& earth, std::string_view meaning) {
  const auto& run = input.run;
  const auto rheology = ElasticRheologyOf(earth, top);
  const ViscoelasticOperator op(rheology, run.lmax);
  const GRDSolver solver(SurfaceGrid(run.lmax), op.ElasticLoveNumbers(),
                         ParametersOf(run, earth.material));
  auto world = LoadStaticWorld(run, top, solver);
  const auto loads = StaticLoadsOf(run, top, solver.Grid());
  const auto times = std::vector<double>{0.0};
  const auto forward = Forward(top, input.trajectory, [&] {
    return TrajectoryOf(run, 1, times, solver);
  });
  auto options = Evolution::Options{};
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  options.record_stride = 1;
  if (forward.read) world.background = forward.store->Start();
  if (forward.Records()) {
    forward.store->PutStart(world.background);
    forward.store->Attach(options);
  }
  Evolution evolution(op, solver, world.background, options);
  const auto snapshots = forward.read
                             ? forward.store->Snapshots1D()
                             : SolveStatic(run, solver, world, loads, evolution);
  if (!snapshots.front().converged) {
    throw std::runtime_error(
        "the sea level equation did not converge; loosen solver.rtol");
  }
  if (forward.Records()) forward.store->Close();
  const auto w = WritingOf(config, run, solver);
  const auto read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                               InputsOf(loads.get()));
  return Sweep1D(input, w, evolution, snapshots, times, op, rheology, earth,
                 world.iceRead, world.seaLevelRead, read) +
         forward.Note();
}

/// The kernels of the static problem on lateral moduli: the one-node run
/// of the 3-D evolution on the stepper of a branchless rheology carrying
/// the moduli, distributed over radial segments on several ranks, recorded,
/// or read from the trajectory store of the run.
std::string KernelStatic3D(const std::filesystem::path& config,
                           const KernelInput& input, const Json& top,
                           const BuiltEarth& earth, std::string_view meaning) {
  const auto& run = input.run;
  RefuseOnRanks3D(top, input);
  const auto elastic = ElasticRheologyOf(earth, top);
  const Viscoelastic3D op(elastic, run.lmax,
                          [](double, double, double) { return 0.0; },
                          OperatorOptionsOf(run, earth, elastic));
  // the sea level grid blocks its transforms for the operator's cache
  const GRDSolver solver(
      SurfaceGrid(run.lmax, {.cache_bytes = op.GetOptions().cache_bytes}),
      op.ElasticLove(), ParametersOf(run, earth.material));
  auto world = LoadStaticWorld(run, top, solver);
  const auto loads = StaticLoadsOf(run, top, solver.Grid());
  const auto times = std::vector<double>{0.0};
  const auto forward = Forward(top, input.trajectory, [&] {
    return TrajectoryOf(run, 1, times, solver, op);
  });
  auto options = Evolution3D::Options{};
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  options.coupled_rtol = run.solver.step_rtol;
  options.max_coupled_iterations = run.solver.step_max_iterations;
  options.record_stride = 1;
  if (forward.read) world.background = forward.store->Start();
  if (forward.Records()) {
    forward.store->PutStart(world.background);
    forward.store->Attach(options);
  }
  Evolution3D evolution(op, solver, world.background, options);
  const auto snapshots = forward.read
                             ? forward.store->Snapshots3D()
                             : SolveStatic(run, solver, world, loads, evolution);
  if (!snapshots.front().converged) {
    throw std::runtime_error(
        "the sea level equation or the 3-D closure did not converge; loosen "
        "solver.rtol or solver.step_rtol, or raise "
        "solver.step_max_iterations");
  }
  if (forward.Records()) forward.store->Close();
  const auto trajectory =
      forward.store ? forward.store->Trajectory3D(snapshots)
                    : ForwardTrajectory3D::FromSnapshots(snapshots);
  const auto w = WritingOf(config, run, solver);
  const auto read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                               InputsOf(loads.get()));
  return Sweep3D(input, w, evolution, snapshots, trajectory, times, op, earth,
                 world.iceRead, world.seaLevelRead, read) +
         forward.Note();
}

/// The kernels of a radial run through time: the radial Maxwell rheology
/// of the deck or of a radial viscosity file, its initial sea level
/// calibrated first where asked, then the recorded run; or, from the
/// complete trajectory store of the run, its calibrated start and its
/// recording read back.
std::string Kernel1D(const std::filesystem::path& config,
                     const KernelInput& input, const Json& top,
                     const BuiltEarth& earth, std::string_view meaning) {
  const auto& run = input.run;
  const auto rheology = RheologyOf(earth, top);
  const ViscoelasticOperator op(rheology, run.lmax);
  const GRDSolver solver(SurfaceGrid(run.lmax), op.ElasticLoveNumbers(),
                         ParametersOf(run, earth.material));
  auto world = LoadWorld(run, top, solver);
  // the library's grid, a node where the forcing jumps repeated
  const auto times = StepGridOf(run, solver.SolverScales()).times;
  const auto stride = input.trajectory.record_stride;
  const auto forward = Forward(top, input.trajectory, [&] {
    return TrajectoryOf(run, stride, times, solver);
  });
  auto options = Evolution::Options{};
  options.scheme = run.solver.scheme;
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  HoldLand(options, world);
  if (forward.read) {
    world.background = forward.store->Start();
  } else if (run.Calibrating()) {
    // calibrated by runs that record nothing; the recording run follows
    Evolution calibrating(op, solver, world.background, options);
    static_cast<void>(
        CalibrateWorld(run, top, solver, world, calibrating, times));
  }
  options.record_stride = stride;
  if (forward.Records()) {
    forward.store->PutStart(world.background);
    forward.store->Attach(options);
  }
  Evolution evolution(op, solver, world.background, options);
  const auto drive = DriveOf<Evolution::Forcing>(run, top, solver, world);
  const auto snapshots =
      forward.read ? forward.store->Snapshots1D()
                   : RunDriven(evolution, times, drive, run.Migrating());
  for (const auto& snapshot : snapshots) {
    Check1D(snapshot, WhenOf(run, solver.SolverScales(), snapshot.t));
  }
  if (forward.Records()) forward.store->Close();
  const auto w = WritingOf(config, run, solver);
  const auto read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                               InputsOf(drive.loads.get()));
  return Sweep1D(input, w, evolution, snapshots, times, op, rheology, earth,
                 world.iceRead, world.seaLevelRead, read) +
         forward.Note();
}

/// The kernels of a 3-D run through time (a lateral viscosity, lateral
/// moduli or both), distributed over radial segments on several ranks:
/// recorded in memory, or into the trajectory store of
/// `trajectory.directory` (each rank's states in its own subdirectory) and
/// fetched back one at a time by the sweeps, or swept from the complete
/// store of the run there without the forward run.
std::string Kernel3D(const std::filesystem::path& config,
                     const KernelInput& input, const Json& top,
                     const BuiltEarth& earth, std::string_view meaning) {
  const auto& run = input.run;
  RefuseOnRanks3D(top, input);
  const auto rheology = RheologyOf(earth, top);
  const Viscoelastic3D op(rheology, run.lmax, DeltaOf(earth),
                          OperatorOptionsOf(run, earth, rheology));
  // the sea level grid blocks its transforms for the operator's cache
  const GRDSolver solver(
      SurfaceGrid(run.lmax, {.cache_bytes = op.GetOptions().cache_bytes}),
      op.ElasticLove(), ParametersOf(run, earth.material));
  auto world = LoadWorld(run, top, solver);
  // the library's grid, a node where the forcing jumps repeated
  const auto times = StepGridOf(run, solver.SolverScales()).times;
  const auto stride = input.trajectory.record_stride;
  const auto forward = Forward(top, input.trajectory, [&] {
    return TrajectoryOf(run, stride, times, solver, op);
  });
  auto options = Evolution3D::Options{};
  options.scheme = run.solver.scheme;
  options.rotation = run.rotation;
  options.rtol = run.solver.rtol;
  options.coupled_rtol = run.solver.step_rtol;
  options.max_coupled_iterations = run.solver.step_max_iterations;
  HoldLand(options, world);
  if (forward.read) {
    world.background = forward.store->Start();
  } else if (run.Calibrating()) {
    Evolution3D calibrating(op, solver, world.background, options);
    static_cast<void>(
        CalibrateWorld(run, top, solver, world, calibrating, times));
  }
  options.record_stride = stride;
  // recording into the store, each kept state goes to disk as the run
  // passes it, and the sweep fetches it back
  if (forward.Records()) {
    forward.store->PutStart(world.background);
    forward.store->Attach(options);
  }
  Evolution3D evolution(op, solver, world.background, options);
  const auto drive = DriveOf<Evolution3D::Forcing>(run, top, solver, world);
  const auto snapshots =
      forward.read ? forward.store->Snapshots3D()
                   : RunDriven(evolution, times, drive, run.Migrating());
  for (const auto& snapshot : snapshots) {
    Check3D(snapshot, WhenOf(run, solver.SolverScales(), snapshot.t));
  }
  if (forward.Records()) forward.store->Close();
  const auto trajectory =
      forward.store ? forward.store->Trajectory3D(snapshots)
                    : ForwardTrajectory3D::FromSnapshots(snapshots);
  const auto w = WritingOf(config, run, solver);
  const auto read = WithInputs(meaning, run, world.iceRead, world.seaLevelRead,
                               InputsOf(drive.loads.get()));
  return Sweep3D(input, w, evolution, snapshots, trajectory, times, op, earth,
                 world.iceRead, world.seaLevelRead, read) +
         forward.Note();
}

}  // namespace

std::string RunKernel(const std::filesystem::path& config) {
  const auto top = planetmodel::ReadJsonFile(config);
  auto input = ReadKernelInput(config);
  const auto earth = BuildEarth(input.run, top);
  PlaceSeismicSources(input.run, earth, top);
  const auto& run = input.run;
  const auto meaning = Meaning(run, earth);
  const auto& ranks = TheRanks();
  // a model with lateral structure has its kernels in the viscosity and
  // the moduli as fields, and only so
  if (earth.Lateral() && input.kernels.lateral == false) {
    Fail(top.At("kernels").At("lateral"),
         std::format("is false, and the model is {}: its kernels in the "
                     "viscosity and the moduli are fields over the Earth; "
                     "drop the member or give true",
                     earth.Dimension()));
  }
  const auto lateral = input.kernels.lateral.value_or(false);
  if (run.problem == Problem::Static) {
    // a viscosity file is refused in the static problem when the config is
    // read, so a lateral Earth here is one of lateral moduli
    if (earth.LateralModuli()) {
      return KernelStatic3D(config, input, top, earth, meaning);
    }
    RefuseThreeDSolver(top, "the static problem on a radial Earth", lateral);
    // one process's run: on several ranks, the first rank's alone
    if (!ranks.First()) return "";
    const RanksOverride alone(Ranks{});
    return KernelStatic1D(config, input, top, earth, meaning);
  }
  if (!earth.Lateral()) {
    RefuseThreeDSolver(top, "a 1-D run (a radial model)", lateral);
    if (!ranks.First()) return "";
    const RanksOverride alone(Ranks{});
    return Kernel1D(config, input, top, earth, meaning);
  }
  return Kernel3D(config, input, top, earth, meaning);
}

}  // namespace sl3dpp::app
