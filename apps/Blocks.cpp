#include "Blocks.hpp"

#include <cmath>
#include <format>
#include <initializer_list>
#include <string_view>
#include <utility>

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

DiscretisationBlock ReadDiscretisationBlock(const Json& top, int least) {
  if (Has(top, "lmax")) {
    Fail(top.At("lmax"),
         "is not a top-level member: it belongs in `discretisation.lmax`");
  }
  if (!Has(top, "discretisation")) {
    Fail(top,
         "has no `discretisation` block, which gives the degree of the run, "
         "`lmax`");
  }
  const auto& block = top.At("discretisation");
  if (!block.IsObject()) {
    Fail(block, "is not a block: `discretisation` is an object of `lmax`");
  }
  Allow(block, {"lmax"});
  const auto out = DiscretisationBlock{.lmax = Integer(block, "lmax")};
  if (out.lmax < least) Fail(block.At("lmax"), "is not a degree");
  return out;
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

/// The two forms of the times, as a refusal names them.
constexpr auto kWindowForms = std::string_view{
    "dates before present (`start_kyr_bp`, `end_kyr_bp` and `nodes`, or "
    "`at_kyr_bp`) or times counted forwards on the run's own clock (`unit` "
    "with `start`, `end` and `nodes`, or `at`)"};

/// The dated form of the times.
Window ReadDates(const Json& block) {
  Allow(block, {"start_kyr_bp", "end_kyr_bp", "nodes", "at_kyr_bp"});
  auto out = Window{};
  out.calendar = Calendar::Dated;
  if (Has(block, "at_kyr_bp")) {
    if (Has(block, "start_kyr_bp")) {
      Fail(block.At("start_kyr_bp"),
           "is given by `at_kyr_bp`, whose first entry is the start");
    }
    if (Has(block, "end_kyr_bp") || Has(block, "nodes")) {
      Fail(block,
           "gives the nodes twice: either `start_kyr_bp`, `end_kyr_bp` and "
           "`nodes` (evenly spaced) or `at_kyr_bp` (the dates listed)");
    }
    out.listed = true;
    const auto& at = block.At("at_kyr_bp");
    for (const auto& item : at.AsArray()) {
      const auto value = Finite(item);
      if (!out.nodes.empty() && !(value < out.nodes.back())) {
        Fail(item,
             "is not later than the date before it: the dates run from the "
             "oldest, strictly decreasing in kyr BP");
      }
      out.nodes.push_back(value);
    }
    if (out.nodes.size() < 2) Fail(at, "is fewer than two nodes");
    return out;
  }
  const auto start = Finite(block.At("start_kyr_bp"));
  const auto end =
      Has(block, "end_kyr_bp") ? Finite(block.At("end_kyr_bp")) : 0.0;
  const auto nodes = Integer(block, "nodes");
  if (!(start > end)) {
    Fail(Has(block, "end_kyr_bp") ? block.At("end_kyr_bp") : block,
         "is not later than start_kyr_bp: dates count back from the "
         "present, so the window's end is the smaller");
  }
  if (nodes < 2) Fail(block.At("nodes"), "is fewer than two nodes");
  const auto spacing = (start - end) / (nodes - 1);
  for (auto n = 0; n < nodes; ++n) {
    out.nodes.push_back(n + 1 == nodes ? end : start - n * spacing);
  }
  return out;
}

/// The forwards form of the times, on the run's own clock.
Window ReadForwards(const Json& block) {
  Allow(block, {"unit", "start", "end", "nodes", "at"});
  if (!Has(block, "unit")) {
    Fail(block, std::format("counts forwards on the run's own clock and "
                            "gives no `unit` for its times: {}",
                            kTimeUnits));
  }
  auto out = Window{};
  out.calendar = Calendar::Forwards;
  out.unit = ReadTimeUnit(block.At("unit"));
  const auto even = Has(block, "end") || Has(block, "nodes");
  const auto listed = Has(block, "at");
  if (even && listed) {
    Fail(block,
         "gives the nodes twice: either `end` and `nodes` (evenly spaced "
         "from `start`) or `at` (the nodes listed)");
  }
  if (!even && !listed) {
    Fail(block,
         "gives no nodes: either `end` and `nodes` (evenly spaced from "
         "`start`) or `at` (the nodes listed)");
  }
  if (listed) {
    if (Has(block, "start")) {
      Fail(block.At("start"),
           "is given by `at`, whose first entry is the start");
    }
    out.listed = true;
    const auto& at = block.At("at");
    for (const auto& item : at.AsArray()) {
      const auto value = Finite(item);
      if (!out.nodes.empty() && !(value > out.nodes.back())) {
        Fail(item,
             "is not later than the node before it: the nodes are strictly "
             "increasing");
      }
      out.nodes.push_back(value);
    }
    if (out.nodes.size() < 2) Fail(at, "is fewer than two nodes");
    return out;
  }
  for (const auto& [name, other] :
       {std::pair{"end", "nodes"}, std::pair{"nodes", "end"}}) {
    if (!Has(block, name)) {
      Fail(block, std::format("gives `{}` without `{}`: evenly spaced nodes "
                              "need both",
                              other, name));
    }
  }
  const auto start = Has(block, "start") ? Finite(block.At("start")) : 0.0;
  const auto end = Finite(block.At("end"));
  const auto nodes = Integer(block, "nodes");
  if (!(end > start)) Fail(block.At("end"), "is not later than `start`");
  if (nodes < 2) Fail(block.At("nodes"), "is fewer than two nodes");
  const auto spacing = (end - start) / (nodes - 1);
  for (auto n = 0; n < nodes; ++n) {
    out.nodes.push_back(n + 1 == nodes ? end : start + n * spacing);
  }
  return out;
}

}  // namespace

double Window::Axis(std::size_t n) const {
  return Dated() ? -nodes[n] * KYR : nodes[n] * unit.seconds;
}

std::vector<double> Window::Times(double scale) const {
  auto out = std::vector<double>{};
  if (!Dated()) {
    const auto toTime = unit.seconds / scale;
    for (const auto node : nodes) out.push_back((node - Start()) * toTime);
    return out;
  }
  if (listed) {
    for (auto n = std::size_t{0}; n < nodes.size(); ++n) {
      out.push_back((Axis(n) - Axis(0)) / scale);
    }
    return out;
  }
  const auto count = static_cast<int>(nodes.size());
  const auto span = (Start() - End()) * KYR;
  for (auto n = 0; n < count; ++n) {
    out.push_back(span * n / (count - 1) / scale);
  }
  return out;
}

std::string Window::Label(double axis) const {
  return Dated() ? std::format("{:.6g} kyr BP", 0.0 - axis / KYR)
                 : std::format("{:.6g} {}", axis / unit.seconds, unit.name);
}

Window ReadWindow(const Json& block) {
  if (!block.IsObject()) {
    Fail(block, std::format("is not a block: `times` is an object, the nodes "
                            "as {}",
                            kWindowForms));
  }
  // times in kyr on a run's own clock without a unit, each named with the
  // member the forwards form gives it
  for (const auto& [name, wanted] : {std::pair{"start_kyr", "start"},
                                     std::pair{"end_kyr", "end"},
                                     std::pair{"at_kyr", "at"}}) {
    if (Has(block, name)) {
      Fail(block.At(name),
           std::format("counts forwards on a run's own clock in kyr without "
                       "its unit: times on the run's clock are `{}` with "
                       "`unit` (\"kyr\", \"yr\" or \"day\"; `start`, "
                       "`end` and `nodes`, or `at`), and dates before present "
                       "are `start_kyr_bp`, `end_kyr_bp` and `nodes`, or "
                       "`at_kyr_bp`",
                       wanted));
    }
  }
  const auto first = [&block](std::initializer_list<const char*> names) {
    for (const auto* name : names) {
      if (Has(block, name)) return std::string_view{name};
    }
    return std::string_view{};
  };
  const auto dated = first({"start_kyr_bp", "end_kyr_bp", "at_kyr_bp"});
  const auto forwards = first({"unit", "start", "end", "at"});
  if (!dated.empty() && !forwards.empty()) {
    Fail(block.At(forwards),
         std::format("counts forwards on the run's own clock, beside `{}`, a "
                     "date before present: the times are given in one of two "
                     "forms, {}",
                     dated, kWindowForms));
  }
  if (!dated.empty()) return ReadDates(block);
  if (!forwards.empty()) return ReadForwards(block);
  Fail(block, std::format("gives no window: the nodes are {}", kWindowForms));
}

std::string_view SeaLevelSource::Member() const {
  return kind == Kind::SeaLevel ? "sea_level" : "topography";
}

std::string_view SeaLevelSource::Quantity() const {
  return kind == Kind::SeaLevel ? "sea level" : "topography";
}

OceanBlock ReadOceanBlock(const Json& block,
                          const std::filesystem::path& directory,
                          Problem problem, Calendar calendar) {
  if (!block.IsObject()) {
    Fail(block,
         "is not a block: `ocean` is an object holding the state, the sea "
         "level (`sea_level` or `topography`) and the ice (`ice`)");
  }
  if (Has(block, "calibrate_initial_sea_level")) {
    Fail(block.At("calibrate_initial_sea_level"),
         "is not a member: the calibration is `ocean.calibration` (`method`, "
         "`iterations`, `rtol`), which the state's \"at\": \"end\" turns on");
  }
  const auto timed = problem == Problem::QuasiStatic;
  // the state's files are read at its date in the static problem and in a
  // run counting forwards, at an end of the window in a dated run
  const auto forwards = timed && calendar == Calendar::Forwards;
  const auto dated = timed && !forwards;
  if (forwards) {
    Allow(block, {"at", "date_kyr_bp", "shorelines", "sea_level", "topography",
                  "ice", "land", "calibration"});
  } else if (timed) {
    if (Has(block, "date_kyr_bp")) {
      Fail(block.At("date_kyr_bp"),
           "dates the state, as the static problem and a run counting "
           "forwards on its own clock read it; a dated run puts the state "
           "at an end of its window, `at` (\"start\" or \"end\")");
    }
    Allow(block, {"at", "shorelines", "sea_level", "topography", "ice", "land",
                  "calibration"});
  } else {
    if (Has(block, "at")) {
      Fail(block.At("at"),
           "is an end of a run's window, which the static problem does not "
           "have: it reads the state at its date, `date_kyr_bp`");
    }
    if (Has(block, "calibration")) {
      Fail(block.At("calibration"),
           std::format("calibrates an initial sea level, which {}: the "
                       "static problem has none, its state being the sea "
                       "level at its own instant",
                       kNeedsQuasiStatic));
    }
    Allow(block, {"date_kyr_bp", "shorelines", "sea_level", "topography",
                  "ice", "land"});
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
         "`topography` (converted with the state's ice, `ice`)");
  }
  if (!Has(block, "ice")) {
    Fail(block,
         "gives no ice, `ice`: the state is the sea level and the ice on the "
         "body at one instant, the ice a file input of its thickness (a "
         "change of it is a term of `forcing.ice_change`)");
  }
  auto out = OceanBlock{};
  auto& source = out.seaLevel;
  source.kind =
      given ? SeaLevelSource::Kind::SeaLevel : SeaLevelSource::Kind::Topography;
  // the state's members are read at the block's instant, not their own
  for (const auto& name : {source.Member(), std::string_view{"ice"}}) {
    const auto& input = block.At(name);
    if (!input.IsObject()) continue;  // ReadFileInput says what it is
    if (Has(input, "date_kyr_bp")) {
      Fail(input.At("date_kyr_bp"),
           dated ? "is not a member: the state's date is the block's, in a "
                   "dated run an end of the window, `ocean.at`"
                 : "is not a member: the state's date is the block's, "
                   "`ocean.date_kyr_bp`");
    }
    if (Has(input, "at")) {
      Fail(input.At("at"),
           timed ? "is not a member: the end of the window the state belongs "
                   "to is the block's, `ocean.at`"
                 : "is not a member: the state's instant is the block's, in "
                   "the static problem its date, `ocean.date_kyr_bp`");
    }
  }
  const auto& input = block.At(source.Member());
  source.input = ReadFileInput(input, directory, source.Member());
  AllowFileInput(input);
  const auto& ice = block.At("ice");
  out.ice = ReadFileInput(ice, directory, "ice_thickness");
  AllowFileInput(ice);
  if (timed) {
    if (!Has(block, "at")) {
      Fail(block,
           "has no `at`: the end of the window the state belongs to, "
           "\"start\" (the initial state) or \"end\" (the state the run "
           "reaches, against whose sea level the initial sea level is "
           "calibrated)");
    }
    out.at = Choice<WindowEnd>(
        block.At("at"), {{"start", WindowEnd::Start}, {"end", WindowEnd::End}});
    if (forwards && out.at == WindowEnd::End) {
      Fail(block.At("at"),
           "puts the state at the window's end, whose ice at the start is "
           "read back along dated ice histories (`thickness` terms) and "
           "against whose sea level the initial sea level is calibrated: a "
           "run counting forwards on its own clock has no such history, its "
           "ice changing by `change` files without a time axis and caps "
           "alone. Give the state at the start, \"start\", or the dated form "
           "of `times` (`start_kyr_bp`, `end_kyr_bp` and `nodes`, or "
           "`at_kyr_bp`)");
    }
  }
  if (!dated) out.date_kyr_bp = ReadDate(block, "date_kyr_bp");
  if (Has(block, "land")) {
    const auto& land = block.At("land");
    out.land = ReadFileInput(land, directory, "land");
    AllowFileInput(land);
  }
  if (Has(block, "shorelines")) {
    out.shorelines = Choice<Shorelines>(
        block.At("shorelines"),
        {{"frozen", Shorelines::Frozen}, {"migrating", Shorelines::Migrating}});
  }
  out.calibrate = timed && out.at == WindowEnd::End;
  if (Has(block, "calibration")) {
    const auto& calibration = block.At("calibration");
    if (!out.calibrate) {
      Fail(calibration,
           "calibrates the initial sea level against the state's sea level "
           "at the window's end, which \"at\": \"end\" gives; with \"at\": "
           "\"start\" the state's sea level is the initial one");
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

SolverBlock ReadSolverBlock(const Json& block, int lmax) {
  Allow(block, {"rtol", "step_rtol", "step_max_iterations",
                "transform_degrees", "cache_mb", "scheme", "step"});
  auto out = SolverBlock{};
  out.rtol = Number(block, "rtol", out.rtol);
  if (!(out.rtol > 0.0)) {
    Fail(block.At("rtol"), "is not a positive tolerance");
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
  Allow(block, {"surface", "exchange", "volume", "exchange_nodes", "fields",
                "smoothing"});
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
  if (Has(block, "fields")) {
    const auto& listed = block.At("fields");
    if (!listed.IsArray()) {
      Fail(listed,
           std::format("is not a list: `output.fields` lists the optional "
                       "fields the surface file adds, of {}",
                       kOptionalFields));
    }
    for (const auto& item : listed.AsArray()) {
      const auto field = Choice<OptionalField>(
          item, {{"horizontal_displacement",
                  OptionalField::HorizontalDisplacement}});
      if (out.Asks(field)) Fail(item, "is listed twice");
      out.fields.push_back(field);
    }
  }
  if (Has(block, "smoothing")) {
    const auto& smoothing = block.At("smoothing");
    if (!smoothing.IsObject()) {
      Fail(smoothing,
           "is not a block: `output.smoothing` is an object of `kind` and "
           "`width_deg`, as {\"kind\": \"gaussian\", \"width_deg\": 2}");
    }
    Allow(smoothing, {"kind", "width_deg"});
    auto read = Smoothing{};
    read.kind = Choice<Smoothing::Kind>(smoothing.At("kind"),
                                        {{"gaussian", Smoothing::Kind::Gaussian}});
    const auto width = Number(smoothing, "width_deg");
    if (!(width > 0.0) || !std::isfinite(width)) {
      Fail(smoothing.At("width_deg"), "is not a positive angle in degrees");
    }
    read.width_deg = width;
    out.smoothing = read;
  }
  return out;
}

RotationalFeedback RotationOf(const Json& word) {
  return Choice<RotationalFeedback>(word, {{"none", RotationalFeedback::None},
                                           {"axial", RotationalFeedback::Axial},
                                           {"full", RotationalFeedback::Full}});
}

bool GravityOf(const Json& word) {
  return Choice<bool>(word, {{"full", true}, {"none", false}});
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
