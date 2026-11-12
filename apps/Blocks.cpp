#include "Blocks.hpp"

#include <cmath>
#include <format>

namespace sl3dpp::app {

Problem ReadProblem(const Json& top) {
  if (!Has(top, "problem")) {
    Fail(top,
         "has no `problem`: \"static\" (one instant) or \"quasi_static\" (a "
         "run through the nodes of a `times` block)");
  }
  const auto& member = top.At("problem");
  const auto problem = Choice<Problem>(
      member,
      {{"static", Problem::Static}, {"quasi_static", Problem::QuasiStatic}});
  if (problem == Problem::Static && Has(top, "times")) {
    Fail(top.At("times"),
         "is a run's time nodes, which the static problem does not take: "
         "set \"problem\": \"quasi_static\" for a run through time");
  }
  if (problem == Problem::QuasiStatic && !Has(top, "times")) {
    Fail(member,
         "is the quasi-static problem, a run through time, which needs a "
         "`times` block");
  }
  return problem;
}

DeckFormat FormatOf(ModelFormat format) {
  switch (format) {
    case ModelFormat::Gia:
      return GiaFormat();
    case ModelFormat::Mineos:
      return MineosFormat();
    case ModelFormat::Elastic:
      break;
  }
  return DeckFormat("elastic", {"rho", "vpv", "vsv", "vph", "vsh", "eta"}, 0);
}

ModelBlock ReadModelBlock(const Json& block,
                          const std::filesystem::path& directory) {
  if (!block.IsObject()) {
    Fail(block,
         "is not a block: `model` is an object of `deck` (the deck's path), "
         "`format`, `moduli` and `viscosity`");
  }
  Allow(block, {"deck", "format", "moduli", "viscosity"});
  auto out = ModelBlock{};
  out.deck = PathOf(block, "deck", directory);
  if (Has(block, "format")) {
    out.format = Choice<ModelFormat>(block.At("format"),
                                     {{"maxwell", ModelFormat::Gia},
                                      {"gia", ModelFormat::Gia},
                                      {"mineos", ModelFormat::Mineos},
                                      {"elastic", ModelFormat::Elastic}});
  }
  if (Has(block, "moduli")) {
    const auto& moduli = block.At("moduli");
    Allow(moduli, {"file", "reference"});
    out.moduli = PathOf(moduli, "file", directory);
    if (Has(moduli, "reference")) {
      out.moduli_file_radial =
          Choice<bool>(moduli.At("reference"),
                       {{"lateral_mean", false}, {"file_radial", true}});
    }
  }
  if (Has(block, "viscosity")) {
    const auto& viscosity = block.At("viscosity");
    Allow(viscosity, {"file", "reference"});
    out.viscosity = PathOf(viscosity, "file", directory);
    if (Has(viscosity, "reference")) {
      out.viscosity_file_radial =
          Choice<bool>(viscosity.At("reference"),
                       {{"geometric_mean", false}, {"file_radial", true}});
    }
  }
  return out;
}

namespace {

/// A finite number.
double Finite(const Json& item) {
  const auto value = item.AsNumber();
  if (!std::isfinite(value)) Fail(item, "is not a finite number");
  return value;
}

/// The before-present form of a `times` block.
TimesBlock ReadDates(const Json& block) {
  for (const auto* name : {"start_kyr", "end_kyr", "at_kyr"}) {
    if (Has(block, name)) {
      Fail(block.At(name),
           "counts forwards on a run's own clock, the form of `response`'s "
           "times; this command's times are dates before present: "
           "`start_kyr_bp`, `end_kyr_bp` and `nodes`");
    }
  }
  Allow(block, {"start_kyr_bp", "end_kyr_bp", "nodes"});
  auto out = TimesBlock{};
  out.start_kyr_bp = Finite(block.At("start_kyr_bp"));
  out.end_kyr_bp =
      Has(block, "end_kyr_bp") ? Finite(block.At("end_kyr_bp")) : 0.0;
  out.nodes = Integer(block, "nodes");
  if (!(out.start_kyr_bp > out.end_kyr_bp)) {
    Fail(Has(block, "end_kyr_bp") ? block.At("end_kyr_bp") : block,
         "is not later than start_kyr_bp: dates count back from the "
         "present, so the window's end is the smaller");
  }
  if (out.nodes < 2) Fail(block.At("nodes"), "is fewer than two nodes");
  return out;
}

}  // namespace

TimesBlock ReadTimesBlock(const Json& block, TimesForm form) {
  if (form == TimesForm::BeforePresent) return ReadDates(block);
  for (const auto* name : {"start_kyr_bp", "end_kyr_bp"}) {
    if (Has(block, name)) {
      Fail(block.At(name),
           "is a date before present, the form of `sea-level`'s times; "
           "`response`'s times count forwards on the run's own clock: "
           "`start_kyr`, `end_kyr` and `nodes`, or `at_kyr`");
    }
  }
  Allow(block, {"start_kyr", "end_kyr", "nodes", "at_kyr"});
  const auto even = Has(block, "end_kyr") || Has(block, "nodes");
  const auto listed = Has(block, "at_kyr");
  if (even && listed) {
    Fail(block,
         "gives the nodes twice: either `end_kyr` and `nodes` (evenly "
         "spaced from `start_kyr`) or `at_kyr` (the nodes listed)");
  }
  if (!even && !listed) {
    Fail(block,
         "gives no nodes: either `end_kyr` and `nodes` (evenly spaced from "
         "`start_kyr`) or `at_kyr` (the nodes listed)");
  }
  auto out = TimesBlock{};
  if (listed) {
    if (Has(block, "start_kyr")) {
      Fail(block.At("start_kyr"),
           "is given by `at_kyr`, whose first entry is the start");
    }
    const auto& at = block.At("at_kyr");
    for (const auto& item : at.AsArray()) {
      const auto value = Finite(item);
      if (!out.nodes_kyr.empty() && !(value > out.nodes_kyr.back())) {
        Fail(item,
             "is not later than the node before it: the nodes are "
             "strictly increasing");
      }
      out.nodes_kyr.push_back(value);
    }
    if (out.nodes_kyr.size() < 2) Fail(at, "is fewer than two nodes");
    for (auto n = std::size_t{1}; n < out.nodes_kyr.size(); ++n) {
      out.steps_kyr.push_back(out.nodes_kyr[n] - out.nodes_kyr[n - 1]);
    }
    return out;
  }
  for (const auto& [name, other] :
       {std::pair{"end_kyr", "nodes"}, std::pair{"nodes", "end_kyr"}}) {
    if (!Has(block, name)) {
      Fail(block, std::format("gives `{}` without `{}`: evenly spaced nodes "
                              "need both",
                              other, name));
    }
  }
  const auto start =
      Has(block, "start_kyr") ? Finite(block.At("start_kyr")) : 0.0;
  const auto end = Finite(block.At("end_kyr"));
  const auto nodes = Integer(block, "nodes");
  if (!(end > start)) {
    Fail(block.At("end_kyr"), "is not later than start_kyr");
  }
  if (nodes < 2) Fail(block.At("nodes"), "is fewer than two nodes");
  const auto spacing = (end - start) / (nodes - 1);
  for (auto n = 0; n < nodes; ++n) {
    out.nodes_kyr.push_back(n + 1 == nodes ? end : start + n * spacing);
  }
  out.steps_kyr.assign(static_cast<std::size_t>(nodes - 1), spacing);
  return out;
}

std::string_view SeaLevelSource::Member() const {
  return kind == Kind::SeaLevel ? "sea_level" : "topography";
}

std::string_view SeaLevelSource::Quantity() const {
  return kind == Kind::SeaLevel ? "sea level" : "topography";
}

OceanBlock ReadOceanBlock(const Json& block,
                          const std::filesystem::path& directory,
                          Problem problem) {
  if (!block.IsObject()) {
    Fail(block,
         "is not a block: `ocean` is an object holding the sea level, "
         "`sea_level` or `topography`");
  }
  if (Has(block, "calibrate_initial_sea_level")) {
    Fail(block.At("calibrate_initial_sea_level"),
         "is not a member: the calibration is `ocean.calibration` (`method`, "
         "`iterations`, `rtol`), which the sea level's \"at\": \"end\" turns "
         "on");
  }
  const auto timed = problem == Problem::QuasiStatic;
  if (timed) {
    Allow(block,
          {"shorelines", "sea_level", "topography", "land", "calibration"});
  } else {
    for (const auto* name : {"shorelines", "calibration"}) {
      if (Has(block, name)) {
        Fail(block.At(name),
             std::format("belongs to a run through time, which {}: the "
                         "static problem solves the linear equation on its "
                         "sea level's own shorelines, with no initial sea "
                         "level to calibrate",
                         kNeedsQuasiStatic));
      }
    }
    Allow(block, {"sea_level", "topography", "land"});
  }
  const auto given = Has(block, "sea_level");
  if (given && Has(block, "topography")) {
    Fail(block,
         "gives both `sea_level` and `topography`: give the one quantity the "
         "file holds");
  }
  if (!given && !Has(block, "topography")) {
    Fail(block,
         "gives no sea level: `sea_level` (a sea level, used as it stands) or "
         "`topography` (converted with the run's own ice at the same "
         "instant)");
  }
  auto out = OceanBlock{};
  auto& source = out.seaLevel;
  source.kind =
      given ? SeaLevelSource::Kind::SeaLevel : SeaLevelSource::Kind::Topography;
  const auto& input = block.At(source.Member());
  source.input = ReadFileInput(input, directory, source.Member());
  if (timed) {
    if (Has(input, "date_kyr_bp")) {
      Fail(input.At("date_kyr_bp"),
           "dates the sea level, as the static problem reads it; the "
           "quasi-static problem reads it at an end of its window, `at`");
    }
    Allow(input, {"file", "format", "variable", "at"});
    if (!Has(input, "at")) {
      Fail(input,
           "has no `at`: the end of the window the sea level belongs to, "
           "\"start\" (the initial sea level) or \"end\" (the sea level the "
           "run reaches, against which the initial sea level is calibrated)");
    }
    source.at = Choice<WindowEnd>(
        input.At("at"), {{"start", WindowEnd::Start}, {"end", WindowEnd::End}});
  } else {
    if (Has(input, "at")) {
      Fail(input.At("at"),
           "is an end of a run's window, which the static problem does not "
           "have: it reads a file with a time axis at its `date_kyr_bp`");
    }
    Allow(input, {"file", "format", "variable", "date_kyr_bp"});
    source.input.date = ReadDate(input, "date_kyr_bp");
  }
  if (Has(block, "land")) {
    const auto& land = block.At("land");
    out.land = ReadFileInput(land, directory, "land");
    Allow(land, {"file", "format", "variable"});
  }
  if (Has(block, "shorelines")) {
    out.shorelines = Choice<Shorelines>(
        block.At("shorelines"),
        {{"frozen", Shorelines::Frozen}, {"migrating", Shorelines::Migrating}});
  }
  out.calibrate = timed && source.at == WindowEnd::End;
  if (Has(block, "calibration")) {
    const auto& calibration = block.At("calibration");
    if (!out.calibrate) {
      Fail(calibration,
           "calibrates the initial sea level against the sea level at the "
           "window's end, which the sea level's \"at\": \"end\" gives; with "
           "\"at\": \"start\" the sea level is the initial one");
    }
    Allow(calibration, {"method", "iterations", "rtol"});
    if (Has(calibration, "method")) {
      out.calibration_method = Choice<CalibrationMethod>(
          calibration.At("method"),
          {{"fixed_point", CalibrationMethod::FixedPoint},
           {"gradient", CalibrationMethod::Gradient}});
    }
    out.calibration_iterations = Integer(calibration, "iterations", 10);
    if (out.calibration_iterations < 1) {
      Fail(calibration.At("iterations"), "is fewer than one run");
    }
    out.calibration_rtol = Number(calibration, "rtol", 1e-4);
    if (!(out.calibration_rtol > 0.0)) {
      Fail(calibration.At("rtol"), "is not a positive tolerance");
    }
  }
  return out;
}

SolverBlock SolverDefaults(SolverOf command) {
  auto out = SolverBlock{};
  if (command == SolverOf::Response) {
    out.step_rtol = 1e-10;
    out.step_max_iterations = 200;
  }
  return out;
}

SolverBlock ReadSolverBlock(const Json& block, int lmax, SolverOf command) {
  const auto ocean = command == SolverOf::SeaLevel;
  if (!ocean && Has(block, "rtol")) {
    Fail(block.At("rtol"),
         "is the sea level solves' tolerance, which `response` (no ocean) "
         "does not have; its 3D solves' tolerance is `step_rtol`");
  }
  if (ocean) {
    Allow(block, {"rtol", "step_rtol", "step_max_iterations",
                  "transform_degrees", "cache_mb", "scheme", "step"});
  } else {
    Allow(block, {"step_rtol", "step_max_iterations", "transform_degrees",
                  "cache_mb", "scheme", "step"});
  }
  auto out = SolverDefaults(command);
  if (ocean) {
    out.rtol = Number(block, "rtol", out.rtol);
    if (!(out.rtol > 0.0)) {
      Fail(block.At("rtol"), "is not a positive tolerance");
    }
  }
  out.step_rtol = Number(block, "step_rtol", out.step_rtol);
  if (!(out.step_rtol > 0.0)) {
    Fail(block.At("step_rtol"), "is not a positive tolerance");
  }
  out.step_max_iterations =
      Integer(block, "step_max_iterations", out.step_max_iterations);
  if (out.step_max_iterations < 1) {
    Fail(block.At("step_max_iterations"), "is not a positive count");
  }
  if (Has(block, "transform_degrees")) {
    for (const auto& item : block.At("transform_degrees").AsArray()) {
      const auto value = item.AsNumber();
      if (value != std::floor(value) || value < 2 || value > lmax) {
        Fail(item, std::format("is not a degree in [2, lmax = {}]", lmax));
      }
      out.transform_degrees.push_back(static_cast<int>(value));
    }
  }
  const auto cache_mb = Integer(block, "cache_mb", 0);
  if (cache_mb < 0) {
    Fail(block.At("cache_mb"), "is not a cache size in MiB (0 or more)");
  }
  out.cache_mb = static_cast<std::size_t>(cache_mb);
  if (Has(block, "scheme")) {
    out.scheme =
        Choice<Scheme>(block.At("scheme"),
                       {{"exponential_trapezoid", Scheme::ExponentialTrapezoid},
                        {"backward_euler", Scheme::BackwardEuler}});
  }
  if (Has(block, "step")) {
    using StepSolver = Viscoelastic3D::Options::StepSolver;
    out.step = Choice<StepSolver>(
        block.At("step"),
        {{"fixed_point", StepSolver::FixedPoint},
         {"conjugate_gradients", StepSolver::ConjugateGradients}});
  }
  return out;
}

CheckpointBlock ReadCheckpointBlock(const Json& block,
                                    const std::filesystem::path& directory) {
  Allow(block, {"directory", "every", "max_walltime_min"});
  auto out = CheckpointBlock{};
  out.directory = PathOf(block, "directory", directory);
  out.every = Integer(block, "every");
  if (out.every < 1) Fail(block.At("every"), "is not a positive stride");
  out.max_walltime_min = Number(block, "max_walltime_min", 0.0);
  return out;
}

OutputBlock ReadOutputBlock(const Json& block,
                            const std::filesystem::path& directory) {
  if (!block.IsObject()) {
    Fail(block,
         "is not a block: `output` is an object, the surface file its "
         "`surface`");
  }
  Allow(block, {"surface", "exchange", "volume", "exchange_nodes"});
  auto out = OutputBlock{};
  out.surface = PathOf(block, "surface", directory);
  if (Has(block, "exchange")) {
    out.exchange = PathOf(block, "exchange", directory);
  }
  if (Has(block, "volume")) out.volume = PathOf(block, "volume", directory);
  if (Has(block, "exchange_nodes")) {
    for (const auto& item : block.At("exchange_nodes").AsArray()) {
      const auto value = item.AsNumber();
      if (value != std::floor(value) || value < 0) {
        Fail(item, "is not a node index (an integer, 0 or more)");
      }
      out.exchange_nodes.push_back(static_cast<int>(value));
    }
  }
  return out;
}

RotationalFeedback RotationOf(const Json& word) {
  return Choice<RotationalFeedback>(word, {{"none", RotationalFeedback::None},
                                           {"axial", RotationalFeedback::Axial},
                                           {"full", RotationalFeedback::Full}});
}

bool HydrostaticMomentsOf(const Json& word) {
  return Choice<bool>(word, {{"input", false}, {"hydrostatic", true}});
}

std::string_view NameOf(RotationalFeedback rotation) {
  switch (rotation) {
    case RotationalFeedback::None:
      return "none";
    case RotationalFeedback::Axial:
      return "axial";
    case RotationalFeedback::Full:
      break;
  }
  return "full";
}

std::string_view NameOf(Shorelines shorelines) {
  return shorelines == Shorelines::Frozen ? "frozen" : "migrating";
}

}  // namespace sl3dpp::app
