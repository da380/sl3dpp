/**
 * @file Perturbation.cpp
 * @brief The `perturbation` block: its reader, and the direction it gives
 * against a recorded run (Perturbation.hpp).
 */
#include "Perturbation.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <utility>

#include "Recorded.hpp"
#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/MemoryPoints.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/io/ModelField.hpp"

namespace sl3dpp::app {

namespace {

/// A model file member of `perturbation.model`: an object of `file` alone,
/// a `reference` refused (the file's field is the change as it stands).
std::filesystem::path ReadModelFileMember(
    const Json& member, const std::filesystem::path& directory,
    std::string_view holds) {
  if (!member.IsObject()) {
    Fail(member, std::format("is not an object: a model file, `file`, {}",
                             holds));
  }
  if (Has(member, "reference")) {
    Fail(member.At("reference"),
         "splits a model file's field about a radial reference, as the "
         "model's own file is split; a perturbation's field is the change as "
         "it stands, radial part and lateral part alike: drop the member");
  }
  Allow(member, {"file"});
  return PathOf(member, "file", directory);
}

}  // namespace

PerturbationBlock ReadPerturbationBlock(
    const Json& block, const Run& run,
    const std::filesystem::path& directory) {
  constexpr auto kMembers =
      "`forcing` (terms of `ice_change`, `load`, `tide` and `seismic`), "
      "`model` (`viscosity`, the change of ln viscosity, and `moduli`, the "
      "changes of kappa and mu) and `ocean` (the change of the state's sea "
      "level)";
  if (!block.IsObject()) {
    Fail(block, std::format("is not a block: `perturbation` is an object "
                            "holding a direction in the run's inputs, of {}",
                            kMembers));
  }
  Allow(block, {"forcing", "model", "ocean"});
  if (!run.gravitating) {
    Fail(block,
         "gives a direction for a tangent-linear run, which is not built for "
         "a run without gravity (`gravity`: \"none\")");
  }
  if (block.AsObject().empty()) {
    Fail(block, std::format("holds no direction: give at least one of {}",
                            kMembers));
  }
  auto out = PerturbationBlock{};
  if (Has(block, "forcing")) {
    // the run's own reader, on the run's clock and with its ocean, and the
    // run's own rules of what its state takes
    const auto& forcing = block.At("forcing");
    out.forcing = ReadForcing(forcing, directory, run.lmax, TermClockOf(run),
                              run.ocean.has_value());
    HoldForcingToState(run, out.forcing, forcing);
  }
  if (Has(block, "model")) {
    const auto& model = block.At("model");
    if (!model.IsObject()) {
      Fail(model,
           "is not a block: `perturbation.model` is an object of `viscosity` "
           "and `moduli`, each a model file of the parameter's change");
    }
    for (const auto* name : {"deck", "format", "burgers"}) {
      if (Has(model, name)) {
        Fail(model.At(name),
             "is not a perturbation's member: a direction in the Earth's "
             "parameters is `viscosity` (a model file of the change of ln "
             "viscosity) and `moduli` (a model file of the changes of kappa "
             "and mu, Pa); a direction in a deck's own columns is not built");
      }
    }
    Allow(model, {"viscosity", "moduli"});
    if (model.AsObject().empty()) {
      Fail(model, "holds no direction: give `viscosity` or `moduli`");
    }
    if (Has(model, "viscosity")) {
      const auto& viscosity = model.At("viscosity");
      if (!run.times) {
        Fail(viscosity,
             std::format("is a change of the viscosity, which {}; the static "
                         "problem's Earth is elastic and has none",
                         kNeedsQuasiStatic));
      }
      out.viscosity = ReadModelFileMember(
          viscosity, directory,
          "whose fields `ln_viscosity` and `ln_viscosity_branch_<b>` hold the "
          "changes of ln viscosity");
    }
    if (Has(model, "moduli")) {
      out.moduli = ReadModelFileMember(
          model.At("moduli"), directory,
          "whose fields `mu` and `kappa` hold the changes of the moduli, Pa");
    }
  }
  if (Has(block, "ocean")) {
    const auto& ocean = block.At("ocean");
    if (!run.ocean) {
      Fail(ocean,
           "is a change of the state's sea level, and the run has no ocean "
           "(no `ocean` block), and so no state");
    }
    if (!ocean.IsObject()) {
      Fail(ocean,
           "is not a block: `perturbation.ocean` is an object holding the "
           "change of the state's sea level, `sea_level` or `topography` as "
           "the run's `ocean` block gives its own");
    }
    if (Has(ocean, "ice")) {
      Fail(ocean.At("ice"),
           "is a change of the state's ice, the direction of which is not "
           "built: a change of the ice at the run's nodes is a term of "
           "`perturbation.forcing.ice_change`");
    }
    if (Has(ocean, "land")) {
      Fail(ocean.At("land"),
           "is a change of the land held, which is a mask, zero or one at "
           "each point, and has no direction");
    }
    const auto& source = run.ocean->seaLevel;
    const auto mine = source.Member();
    const auto other = mine == "sea_level" ? std::string_view{"topography"}
                                           : std::string_view{"sea_level"};
    if (Has(ocean, other)) {
      Fail(ocean.At(other),
           std::format(
               "is not the member the run's state gives its sea level by, "
               "which is `ocean.{}`: give the change of that, `{}` ({})",
               mine, mine,
               mine == "topography"
                   ? "a change of the topography is minus the sea level's, "
                     "the state's ice held"
                   : "a change of the sea level is minus the topography's"));
    }
    Allow(ocean, {mine});
    if (!Has(ocean, mine)) {
      Fail(ocean, std::format("holds no direction: give `{}`, the change of "
                              "the state's {}",
                              mine, source.Quantity()));
    }
    const auto& input = ocean.At(mine);
    out.seaLevel = ReadFileInput(input, directory, mine);
    AllowFileInput(input);
  }
  return out;
}

Run PerturbedForcingOf(const Run& run, const PerturbationBlock& perturbation,
                       const BuiltEarth& earth, const Json& block) {
  auto out = run;
  out.forcing = perturbation.forcing;
  out.fileJumps.clear();
  if (!Has(block, "forcing")) return out;
  PlaceSeismicSources(out, earth, block);
  FindFileJumps(out);
  return out;
}

namespace {

/// Refuses a jump of the perturbation's forcing that falls where the
/// library steps no pair of the run's own (StepGridOf: a node repeated or
/// a pair inserted where the run's forcing jumps): the left limit and the
/// right of such a jump would be two nodes the run does not have.
void RefuseJumpsTheRunDoesNotStep(const RecordedRun& recorded,
                                  const Run& forcing, const Json& block) {
  const auto& run = *recorded.run;
  const auto& steps = recorded.steps;
  const auto clock = SolverClockOf(run, recorded.solver->SolverScales());
  const auto& window = *run.times;
  for (const auto& jump : JumpsOf(forcing)) {
    const auto stepped = std::ranges::any_of(steps.jumps, [&](const auto& pair) {
      return std::abs(clock.At(steps.times[pair.first]) - jump.at) <=
             jump.tolerance;
    });
    if (stepped) continue;
    Fail(block.At("forcing").At(jump.list).At(jump.term).At(jump.member),
         std::format(
             "jumps at {}, where the run's own forcing does not, so that the "
             "run steps no left limit and right limit there: the "
             "tangent-linear run along a direction that jumps where its run "
             "does not is not built. Give the run's `forcing` a term that "
             "jumps there too (one of zero amplitude serves), or a direction "
             "continuous there",
             window.Label(window.Dated() ? window.Axis(0) + jump.at * KYR
                                         : jump.at * window.unit.seconds)));
  }
}

/// Adds a field to each node's of a direction's channel, which is made of
/// zeros where it was absent.
void AddAtEveryNode(std::vector<std::vector<double>>& channel,
                    std::span<const double> field, std::size_t nodes) {
  if (channel.empty()) {
    channel.assign(nodes, std::vector<double>(field.size(), 0.0));
  }
  for (auto& node : channel) {
    for (auto i = std::size_t{0}; i < field.size(); ++i) node[i] += field[i];
  }
}

}  // namespace

template <class Perturbation>
Direction<Perturbation> SurfaceDirectionOf(
    const RecordedRun& recorded, const PerturbationBlock& perturbation,
    const Run& forcing, const Json& block, bool total) {
  const auto& run = *recorded.run;
  const auto& solver = *recorded.solver;
  const auto& grid = solver.Grid();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto& steps = recorded.steps;
  const auto nodes = steps.times.size();
  const auto timed = run.times.has_value();
  auto out = Direction<Perturbation>{};
  auto& d = out.perturbation;
  if (timed) RefuseJumpsTheRunDoesNotStep(recorded, forcing, block);

  // the forcing beside the ice: the loads by their properties, the tide and
  // the sources, each read at the library's nodes as the run's own are
  const auto& terms = perturbation.forcing;
  if (terms.BesideTheIce()) {
    if (!terms.load.empty()) {
      out.loads = std::make_shared<const SurfaceLoads>(
          terms, grid, block.At("forcing"), timed);
      if (timed) {
        const auto clock = SolverClockOf(run, scales);
        auto times = steps.times;
        for (auto& t : times) t = clock.At(t);
        out.loads->CheckCovers(times);
      }
    }
    const auto channels = ForcingOf<Evolution::Forcing>(
        forcing, solver, {}, out.loads, {}, {}, timed);
    // a channel at every node, the left limit at the first of a pair
    const auto atNodes = [&](const auto& value, const auto& before,
                             auto& channel) {
      if (!value) return;
      for (auto k = std::size_t{0}; k < nodes; ++k) {
        const auto t = steps.times[k];
        channel.push_back(steps.Left(k) && before ? before(t) : value(t));
      }
    };
    atNodes(channels.directLoad, channels.directLoadBefore, d.directLoad);
    atNodes(channels.directLoadOutsideWaterBudget,
            channels.directLoadOutsideWaterBudgetBefore,
            d.directLoadOutsideWaterBudget);
    atNodes(channels.solidSurfaceChange, channels.solidSurfaceChangeBefore,
            d.solidSurfaceChange);
    atNodes(channels.tide, channels.tideBefore, d.tide);
    atNodes(channels.sources, channels.sourcesBefore, d.sources);
  }

  // the ice: the terms' change at each node, as the run's own ice is
  // formed there
  if (!terms.ice_change.empty()) {
    auto files = InputFiles(grid);
    const RunIce ice(*recorded.Ice(), terms.ice_change,
                     block.At("forcing").At("ice_change"), files);
    const auto held = !timed || run.ocean->at == WindowEnd::Start;
    const auto start = timed ? run.times->Axis(0) : ice.StateTime();
    for (auto k = std::size_t{0}; k < nodes; ++k) {
      auto change = std::vector<double>{};
      if (!timed) {
        change = ice.ChangeAt(ice.StateTime());
        out.iceChange = change;
      } else {
        const auto t = start + steps.times[k] * scales.Time();
        const auto before = steps.Left(k);
        if (held) {
          change = ice.ChangeAt(t, before);
        } else {
          // from the ice the run starts from, as the run's own
          change = ice.ChangeAt(start);
          const auto since = ice.ChangeSince(t, start, before);
          for (auto i = std::size_t{0}; i < change.size(); ++i) {
            change[i] += since[i];
          }
        }
      }
      for (auto& v : change) v /= length;
      d.iceThickness.push_back(std::move(change));
    }
    if (held) {
      // the state's ice held: the library's thickness direction moves the
      // background's ice with the first node's, which the direct load of
      // the first node's change at every node takes back
      const auto& ocean = recorded.Background().oceanFunction;
      auto load = d.iceThickness.front();
      for (auto i = std::size_t{0}; i < load.size(); ++i) {
        load[i] *= solver.IceDensity() * (1.0 - ocean[i]);
      }
      AddAtEveryNode(d.directLoad, load, nodes);
    }
  }

  // the state's sea level, read at the state's instant as the state is
  if (perturbation.seaLevel) {
    const auto& input = *perturbation.seaLevel;
    const auto& source = run.ocean->seaLevel;
    const auto& member = block.At("ocean").At(source.Member());
    auto files = InputFiles(grid);
    const auto file = files.Open(input, member);
    auto change = ReadValues(input, *file, member, recorded.Ice()->StateTime(),
                             StateReadOf(run));
    // a topography's change is minus the sea level's, the state's ice held
    const auto sign =
        source.kind == SeaLevelSource::Kind::Topography ? -1.0 : 1.0;
    for (auto& v : change) {
      if (!std::isfinite(v)) {
        Fail(member.At("file"),
             std::format("gives a change of the {} that is not a number "
                         "somewhere (a surface file marks uncovered points "
                         "so): fill it before the run",
                         source.Quantity()));
      }
      v *= sign / length;
    }
    if (run.Calibrating() && total) {
      out.targetSeaLevel = std::move(change);
    } else {
      d.initialSeaLevel = std::move(change);
    }
  }
  return out;
}

template Direction<TangentPerturbation> SurfaceDirectionOf<TangentPerturbation>(
    const RecordedRun&, const PerturbationBlock&, const Run&, const Json&,
    bool);
template Direction<TangentPerturbation3D>
SurfaceDirectionOf<TangentPerturbation3D>(const RecordedRun&,
                                          const PerturbationBlock&, const Run&,
                                          const Json&, bool);

namespace {

/// A model file's field against the run's Earth, a fault refused at the
/// member's `file`; nothing where the file does not hold it.
std::optional<io::ModelField> FieldOf(const std::filesystem::path& path,
                                      const BuiltEarth& earth,
                                      const std::string& name,
                                      std::optional<Dimensions> dimensions,
                                      const Json& member) {
  auto read = io::ModelField::ReadIfHeld(path, earth.model, name, dimensions);
  if (!read) Fail(member.At("file"), read.error().message);
  return std::move(*read);
}

/// The fields a perturbation of the viscosity is read from.
constexpr auto kLogViscosity = std::string_view{"ln_viscosity"};
constexpr auto kLogViscosityOfBranch = std::string_view{"ln_viscosity_branch_"};
constexpr auto kViscosityFields =
    "`ln_viscosity` (the change of the natural logarithm of every relaxing "
    "branch's viscosity alike: every relaxation time moved by it, the "
    "branches' moduli held) and `ln_viscosity_branch_<b>` (one branch's "
    "alone, b as `kernel` numbers its `K_branch_<b>`), pure numbers that add";

/// The branch a field `ln_viscosity_branch_<b>` names; nothing for another
/// name.
std::optional<std::size_t> BranchNamedBy(std::string_view name) {
  if (!name.starts_with(kLogViscosityOfBranch)) return std::nullopt;
  const auto digits = name.substr(kLogViscosityOfBranch.size());
  if (digits.empty() || digits.size() > 9 ||
      !std::ranges::all_of(digits,
                           [](char c) { return c >= '0' && c <= '9'; }) ||
      (digits.size() > 1 && digits.front() == '0')) {
    return std::nullopt;
  }
  auto b = std::size_t{0};
  for (const auto c : digits) b = 10 * b + static_cast<std::size_t>(c - '0');
  return b;
}

/// Whether a field's name is a parameter a prony or burgers deck states,
/// or a Burgers region of `model.burgers`.
bool IsStatedParameter(std::string_view name) {
  return name == "kelvin_viscosity" || name == "kelvin_modulus_fraction" ||
         name == "steady_viscosity" ||
         name.starts_with("relaxation_strength") ||
         name.starts_with("relaxation_time") ||
         name.starts_with("ln_kelvin_") || name.starts_with("ln_steady_") ||
         name.starts_with("ln_relaxation_");
}

/// The changes of ln viscosity of `perturbation.model.viscosity`, pure
/// numbers read as stored: every branch's alike (`ln_viscosity`) and each
/// branch's own (`ln_viscosity_branch_<b>`), which add. The file's other
/// fields are held against these: `mu` and `kappa` are the moduli's, which
/// one file may give beside them; a field `viscosity`, a parameter a prony
/// or burgers deck states, and any other are refused at the member's
/// `file`, each named, as is a file that holds none of the fields read.
struct ViscosityFields {
  std::optional<io::ModelField> all;
  std::map<std::size_t, io::ModelField> branch;

  /// Whether every field is a function of the radius alone.
  [[nodiscard]] bool IsRadial() const {
    return (!all || all->IsRadial()) &&
           std::ranges::all_of(branch, [](const auto& held) {
             return held.second.IsRadial();
           });
  }
};
ViscosityFields ViscosityFieldsOf(const PerturbationBlock& perturbation,
                                  const BuiltEarth& earth, const Json& member) {
  const auto& where = member.At("file");
  const auto names = io::ModelField::FieldNames(perturbation.viscosity);
  if (!names) Fail(where, names.error().message);
  auto out = ViscosityFields{};
  for (const auto& name : *names) {
    if (name == "mu" || name == "kappa") continue;
    const auto branch = BranchNamedBy(name);
    if (name == kLogViscosity || branch) {
      auto field =
          FieldOf(perturbation.viscosity, earth, name, std::nullopt, member);
      if (!field) continue;
      if (branch) {
        out.branch.emplace(*branch, std::move(*field));
      } else {
        out.all = std::move(*field);
      }
      continue;
    }
    if (name == "viscosity") {
      Fail(where,
           "holds a field `viscosity`, and a perturbation of the viscosity is "
           "a change of its natural logarithm, read from the field "
           "`ln_viscosity` (a pure number: delta ln eta): name the field so");
    }
    if (IsStatedParameter(name)) {
      Fail(where,
           std::format(
               "holds a field `{}`, a parameter a prony or burgers deck (or "
               "`model.burgers`) states: a direction in those parameters (a "
               "prony deck's strengths and times, a Burgers body's steady "
               "viscosity and its Kelvin element's viscosity and modulus "
               "fraction) is not built, as `kernel`'s kernels in them are "
               "not. The directions built are in the branches' viscosities: "
               "{}",
               name, kViscosityFields));
    }
    Fail(where, std::format("holds a field `{}`, which a perturbation of the "
                            "viscosity does not read: its fields are {}",
                            name, kViscosityFields));
  }
  if (!out.all && out.branch.empty()) {
    Fail(where, std::format("holds none of the fields a perturbation of the "
                            "viscosity is read from: {}",
                            kViscosityFields));
  }
  return out;
}

/// Refuses a field of one branch's viscosity for a branch the run's
/// rheology does not have.
void RefuseMissingBranches(const ViscosityFields& fields, std::size_t branches,
                           const Json& member) {
  for (const auto& [b, field] : fields.branch) {
    if (b < branches) continue;
    Fail(member.At("file"),
         std::format(
             "holds `{}{}`, and the run's rheology has {} branch{}, numbered "
             "from 0 as `kernel`'s `K_branch_<b>` (a Maxwell body's one is "
             "branch 0; a Burgers body's slower term 0 and its faster 1; a "
             "prony deck's in the deck's order)",
             kLogViscosityOfBranch, b, branches, branches == 1 ? "" : "es"));
  }
}

/// Refuses a change of one branch's viscosity at a point that relaxes and
/// does not have the branch.
[[noreturn]] void RefuseAbsentBranch(const Json& member, std::size_t b,
                                     double depthKm) {
  Fail(member.At("file"),
       std::format(
           "changes `{}{}` at {:.6g} km depth, where the rheology relaxes and "
           "has no branch {} (a Maxwell layer beside Burgers bodies has "
           "branch 0 alone, as `kernel` writes its kernel in `K_branch_0`): "
           "give a change of that branch that is zero there",
           kLogViscosityOfBranch, b, depthKm, b));
}

/// The changes of the moduli of `perturbation.model.moduli`, Pa: `kappa`
/// and `mu`, either alone.
struct ModuliFields {
  std::optional<io::ModelField> kappa, mu;
};
ModuliFields ModuliFieldsOf(const PerturbationBlock& perturbation,
                            const BuiltEarth& earth, const Json& member) {
  auto out = ModuliFields{
      .kappa = FieldOf(perturbation.moduli, earth, "kappa", ModulusDimensions,
                       member),
      .mu = FieldOf(perturbation.moduli, earth, "mu", ModulusDimensions,
                    member)};
  if (!out.kappa && !out.mu) {
    Fail(member.At("file"),
         "holds neither `mu` nor `kappa`, the fields a perturbation of the "
         "moduli is read from: their changes, Pa");
  }
  return out;
}

/// Refuses a field that varies laterally for a radial run.
void RefuseLateral(bool radial, const Json& member,
                   std::string_view parameter, std::string_view input) {
  if (radial) return;
  Fail(member.At("file"),
       std::format(
           "gives a change of the {} that varies laterally (a field stored on "
           "an angular grid), and the run is 1-D: the tangent-linear run of a "
           "radial run along lateral structure is not built. `kernel`'s "
           "lateral kernels (`kernels.lateral`) paired with the change give "
           "a measurement's; or give the run's own `{}` as a file stored on "
           "an angular grid, which makes the run 3-D",
           parameter, input));
}

}  // namespace

void CheckEarthDirection(const PerturbationBlock& perturbation,
                         const BuiltEarth& earth, bool lateral,
                         const Json& block) {
  if (!perturbation.OfTheEarth()) return;
  const auto& model = block.At("model");
  if (!perturbation.viscosity.empty()) {
    const auto& member = model.At("viscosity");
    const auto fields = ViscosityFieldsOf(perturbation, earth, member);
    if (!lateral) {
      RefuseLateral(fields.IsRadial(), member, "viscosity", "model.viscosity");
    }
  }
  if (!perturbation.moduli.empty()) {
    const auto& member = model.At("moduli");
    const auto fields = ModuliFieldsOf(perturbation, earth, member);
    if (lateral) return;
    if (fields.kappa) {
      RefuseLateral(fields.kappa->IsRadial(), member, "bulk modulus", "model.moduli");
    }
    if (fields.mu) {
      RefuseLateral(fields.mu->IsRadial(), member, "shear modulus", "model.moduli");
    }
  }
}

void AddEarthDirection(TangentPerturbation& direction,
                       const PerturbationBlock& perturbation,
                       const Recorded1D& recorded, const Json& block) {
  if (!perturbation.OfTheEarth()) return;
  const auto& earth = *recorded.earth;
  const auto& rheology = *recorded.rheology;
  const auto& material = rheology.BaseMaterial();
  const auto& mesh = material.Mesh();
  const auto& model = block.At("model");
  if (!perturbation.viscosity.empty()) {
    const auto& member = model.At("viscosity");
    const auto fields = ViscosityFieldsOf(perturbation, earth, member);
    RefuseLateral(fields.IsRadial(), member, "viscosity", "model.viscosity");
    const auto branches = rheology.BranchCount();
    RefuseMissingBranches(fields, branches, member);
    // delta itau_b = -itau_b delta ln eta_b at the nodes, the branch's
    // modulus held: every branch by the field of all, each by its own
    const auto ngll = mesh.NGll();
    const auto lengths = material.ModelScales();
    for (auto b = std::size_t{0}; b < branches; ++b) {
      const auto& itau = rheology.BranchAt(b).itau;
      const auto own = fields.branch.find(b);
      auto change = std::vector<double>(itau.size(), 0.0);
      for (auto e = 0; e < mesh.ElementCount(); ++e) {
        const auto layer = static_cast<std::size_t>(mesh.LayerOf(e));
        for (auto q = 0; q < ngll; ++q) {
          const auto i = static_cast<std::size_t>(e) * ngll + q;
          const auto r = mesh.Radius(e, q);
          if (itau[i] == 0.0) {
            // a point that relaxes without this branch takes no change of it
            auto relaxes = false;
            for (auto k = std::size_t{0}; k < branches; ++k) {
              relaxes = relaxes || rheology.BranchAt(k).itau[i] != 0.0;
            }
            if (relaxes && own != fields.branch.end() &&
                own->second.ValueIn(layer, r) != 0.0) {
              RefuseAbsentBranch(
                  member, b, (material.Radius() - r) * lengths.Length() / 1e3);
            }
            continue;
          }
          auto log = fields.all ? fields.all->ValueIn(layer, r) : 0.0;
          if (own != fields.branch.end()) log += own->second.ValueIn(layer, r);
          change[i] = -itau[i] * log;
        }
      }
      direction.itau.push_back(std::move(change));
    }
  }
  if (!perturbation.moduli.empty()) {
    const auto& member = model.At("moduli");
    const auto fields = ModuliFieldsOf(perturbation, earth, member);
    const auto& points = recorded.op->Points();
    const auto pa = material.ModelScales().Factor(ModulusDimensions);
    const auto count = static_cast<std::size_t>(mesh.ElementCount()) *
                       points.count;
    // at the memory points, (element, point); the fluid has no shear
    const auto sample = [&](const io::ModelField& field, bool shear) {
      auto out = std::vector<double>(count, 0.0);
      for (auto e = 0; e < mesh.ElementCount(); ++e) {
        if (shear && material.IsFluid(e)) continue;
        const auto layer = static_cast<std::size_t>(mesh.LayerOf(e));
        for (auto p = 0; p < points.count; ++p) {
          out[static_cast<std::size_t>(e) * points.count + p] =
              field.ValueIn(layer, points.RadiusAt(mesh, e, p)) / pa;
        }
      }
      return out;
    };
    if (fields.kappa) {
      RefuseLateral(fields.kappa->IsRadial(), member, "bulk modulus", "model.moduli");
      direction.material.kappa = sample(*fields.kappa, false);
    }
    if (fields.mu) {
      RefuseLateral(fields.mu->IsRadial(), member, "shear modulus", "model.moduli");
      direction.material.mu = sample(*fields.mu, true);
    }
  }
}

std::vector<double> ShearChangeAtNodes(const PerturbationBlock& perturbation,
                                       const Recorded1D& recorded,
                                       const Json& block) {
  if (perturbation.moduli.empty()) return {};
  const auto& member = block.At("model").At("moduli");
  const auto fields = ModuliFieldsOf(perturbation, *recorded.earth, member);
  if (!fields.mu) return {};
  RefuseLateral(fields.mu->IsRadial(), member, "shear modulus",
                "model.moduli");
  const auto& material = recorded.rheology->BaseMaterial();
  const auto& mesh = material.Mesh();
  const auto pa = material.ModelScales().Factor(ModulusDimensions);
  const auto ngll = mesh.NGll();
  auto out = std::vector<double>(
      static_cast<std::size_t>(mesh.ElementCount()) * ngll, 0.0);
  for (auto e = 0; e < mesh.ElementCount(); ++e) {
    if (material.IsFluid(e)) continue;
    const auto layer = static_cast<std::size_t>(mesh.LayerOf(e));
    for (auto q = 0; q < ngll; ++q) {
      out[static_cast<std::size_t>(e) * ngll + q] =
          fields.mu->ValueIn(layer, mesh.Radius(e, q)) / pa;
    }
  }
  return out;
}

void AddEarthDirection(TangentPerturbation3D& direction,
                       const PerturbationBlock& perturbation,
                       const Recorded3D& recorded, const Json& block) {
  if (!perturbation.OfTheEarth()) return;
  const auto& earth = *recorded.earth;
  const auto& coupling = recorded.op->Coupling();
  const auto& material = recorded.op->GetRheology().BaseMaterial();
  const auto& mesh = material.Mesh();
  const auto& model = block.At("model");
  // a field at every grid value of the coupling's memory nodes, each node
  // read within its own element's layer
  const auto sample = [&](const io::ModelField& field, double factor) {
    auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      const auto layer =
          static_cast<std::size_t>(mesh.LayerOf(coupling.ElementOfNode(rn)));
      const auto r = coupling.RadiusOfNode(rn);
      const auto at = coupling.FieldOffset(rn);
      const auto& grid = coupling.GridOf(rn);
      const auto nphi = grid.Longitudes().size();
      for (auto i = std::size_t{0}; i < grid.CoLatitudes().size(); ++i) {
        for (auto j = std::size_t{0}; j < nphi; ++j) {
          out[at + i * nphi + j] =
              factor * field.ValueIn(layer, r, grid.CoLatitudes()[i],
                                     grid.Longitudes()[j]);
        }
      }
    }
    return out;
  };
  if (!perturbation.viscosity.empty()) {
    const auto& member = model.At("viscosity");
    const auto fields = ViscosityFieldsOf(perturbation, earth, member);
    const auto branches = recorded.op->GetRheology().BranchCount();
    RefuseMissingBranches(fields, branches, member);
    if (fields.all) direction.logViscosity = sample(*fields.all, 1.0);
    if (!fields.branch.empty()) {
      // one branch's: delta itau_b = -itau_b delta ln eta_b over the grid
      // values, the other branches' zero
      const auto values = coupling.GridValueCount();
      const auto minusItau = TangentLinearSolver3D::ItauOfLogViscosity(
          *recorded.op, std::vector<double>(values, 1.0));
      const auto lengths = material.ModelScales();
      direction.itau.assign(branches, std::vector<double>(values, 0.0));
      for (const auto& [b, field] : fields.branch) {
        const auto log = sample(field, 1.0);
        for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
          const auto at = coupling.FieldOffset(rn);
          const auto& grid = coupling.GridOf(rn);
          const auto count =
              grid.CoLatitudes().size() * grid.Longitudes().size();
          for (auto i = at; i < at + count; ++i) {
            if (minusItau[b][i] != 0.0) {
              direction.itau[b][i] = minusItau[b][i] * log[i];
              continue;
            }
            auto relaxes = false;
            for (auto k = std::size_t{0}; k < branches; ++k) {
              relaxes = relaxes || minusItau[k][i] != 0.0;
            }
            if (relaxes && log[i] != 0.0) {
              RefuseAbsentBranch(
                  member, b,
                  (material.Radius() - coupling.RadiusOfNode(rn)) *
                      lengths.Length() / 1e3);
            }
          }
        }
      }
    }
  }
  if (!perturbation.moduli.empty()) {
    const auto& member = model.At("moduli");
    const auto fields = ModuliFieldsOf(perturbation, earth, member);
    const auto scalesOf = material.ModelScales();
    const auto pa = scalesOf.Factor(ModulusDimensions);
    // the elements of this rank's range that the coupling carries no
    // memory nodes for: a change of the moduli there is not taken
    auto carried = std::set<int>{};
    for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
      carried.insert(coupling.ElementOfNode(rn));
    }
    const auto [first, last] = coupling.GetOptions().elements;
    const auto end = last < 0 ? mesh.ElementCount() : last;
    const auto& points = coupling.Points();
    const auto& surface = recorded.solver->Grid();
    const auto uncovered = [&](const io::ModelField& field,
                               std::string_view name) {
      for (auto e = first; e < end; ++e) {
        // a 3-D run takes no lateral moduli in the fluid, its own or a
        // direction's; the element holding the centre is an element like
        // any other (its memory points are interior to it)
        if (carried.contains(e) || material.IsFluid(e)) continue;
        const auto layer = static_cast<std::size_t>(mesh.LayerOf(e));
        for (auto p = 0; p < points.count; ++p) {
          const auto r = points.RadiusAt(mesh, e, p);
          auto moved = false;
          if (field.IsRadial()) {
            moved = field.ValueIn(layer, r) != 0.0;
          } else {
            for (const auto theta : surface.CoLatitudes()) {
              for (const auto phi : surface.Longitudes()) {
                moved = moved || field.ValueIn(layer, r, theta, phi) != 0.0;
              }
            }
          }
          if (!moved) continue;
          Fail(member.At("file"),
               std::format(
                   "changes {} at {:.6g} km depth, in an element where the "
                   "3-D run carries no memory nodes ({}): its tangent-linear "
                   "run takes a change of the moduli where the run's lateral "
                   "structure lives alone (its viscous elements and those "
                   "its own lateral moduli perturb), and elsewhere is not "
                   "built. Give a change that is zero there",
                   name, (material.Radius() - r) * scalesOf.Length() / 1e3,
                   "an elastic element with no lateral moduli of the run's "
                   "own"));
        }
      }
    };
    if (fields.kappa) {
      uncovered(*fields.kappa, "kappa");
      direction.material.kappa = sample(*fields.kappa, 1.0 / pa);
    }
    if (fields.mu) {
      uncovered(*fields.mu, "mu");
      direction.material.mu = sample(*fields.mu, 1.0 / pa);
    }
  }
}

std::vector<std::pair<std::string, std::string>> DescribePerturbation(
    const Run& run, const PerturbationBlock& perturbation) {
  auto out = std::vector<std::pair<std::string, std::string>>{};
  const auto listed = [&out](std::string name, const auto& terms) {
    if (terms.empty()) return;
    auto text = std::string{};
    for (const auto& term : terms) {
      if (!text.empty()) text += "; ";
      text += Describe(term);
    }
    out.emplace_back(std::move(name), std::move(text));
  };
  const auto& forcing = perturbation.forcing;
  listed("ice_change", forcing.ice_change);
  listed("load", forcing.load);
  listed("tide", forcing.tide);
  listed("seismic", forcing.seismic);
  if (!perturbation.viscosity.empty()) {
    out.emplace_back("viscosity",
                     std::format("{} (the change of ln viscosity)",
                                 perturbation.viscosity.string()));
  }
  if (!perturbation.moduli.empty()) {
    out.emplace_back("moduli",
                     std::format("{} (the changes of kappa and mu, Pa)",
                                 perturbation.moduli.string()));
  }
  if (perturbation.seaLevel) {
    out.emplace_back(std::string{run.ocean->seaLevel.Member()},
                     DescribeInput(*perturbation.seaLevel,
                                   "the state's instant"));
  }
  return out;
}

}  // namespace sl3dpp::app
