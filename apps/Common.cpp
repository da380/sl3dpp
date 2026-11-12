#include "Common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <numbers>
#include <optional>
#include <planetmodel/json.hpp>
#include <stdexcept>
#include <utility>

#include "Ranks.hpp"
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/LoveNumbers.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/SurfaceGrid.hpp"

namespace sl3dpp::app {

std::vector<std::string_view> SeaLevelRunMemberNames() {
  return {"problem", "model",    "lmax",    "times", "forcing",
          "ocean",   "rotation", "moments", "solver"};
}

void AllowTopLevel(const Json& top, std::span<const std::string_view> allowed) {
  // the members a block holds, by the block they belong in
  constexpr auto kInBlocks = std::array{
      std::pair{std::string_view{"model_format"}, "`model.format`"},
      std::pair{std::string_view{"viscosity"}, "`model.viscosity`"},
      std::pair{std::string_view{"state"},
                "`ocean`, as `sea_level` or `topography` and `ice`"},
      std::pair{std::string_view{"sea_level"}, "`ocean.sea_level`"},
      std::pair{std::string_view{"topography"}, "`ocean.topography`"},
      std::pair{std::string_view{"date_kyr_bp"}, "`ocean.date_kyr_bp`"},
      std::pair{std::string_view{"at"}, "`ocean.at`"},
      std::pair{std::string_view{"ice"},
                "`ocean.ice` (the state's ice) or, a change of it, "
                "`forcing.ice_change`"},
      std::pair{std::string_view{"ice_history"},
                "`forcing.ice_change`, as a `thickness` term"},
      std::pair{std::string_view{"ice_change"}, "`forcing.ice_change`"},
      std::pair{std::string_view{"change"}, "`forcing.ice_change`"},
      std::pair{std::string_view{"scheme"}, "`solver.scheme`"},
      std::pair{std::string_view{"rtol"}, "`solver.rtol`"},
      std::pair{std::string_view{"coupled"},
                "`solver`, as `step_rtol`, `step_max_iterations`, "
                "`transform_degrees` and `cache_mb`"},
      std::pair{std::string_view{"shorelines"}, "`ocean.shorelines`"},
      std::pair{std::string_view{"calibration"}, "`ocean.calibration`"},
      std::pair{std::string_view{"calibrate_initial_sea_level"},
                "`ocean.calibration`"}};
  for (const auto& [key, value] : top.AsObject()) {
    if (std::ranges::find(allowed, key) != allowed.end()) continue;
    for (const auto& [name, block] : kInBlocks) {
      if (key == name) {
        Fail(value,
             std::format("is not a top-level member: it belongs in {}", block));
      }
    }
  }
  Allow(top, allowed);
}

SeaLevelRun ReadSeaLevelRun(const Json& top,
                            const std::filesystem::path& directory) {
  auto out = SeaLevelRun{};
  out.problem = ReadProblem(top);
  const auto timed = out.problem == Problem::QuasiStatic;
  const auto& model = top.At("model");
  if (!timed && model.IsObject() && Has(model, "viscosity")) {
    Fail(model.At("viscosity"),
         std::format("is a viscosity, which {}; the static problem's Earth "
                     "is elastic and has no use for it",
                     kNeedsQuasiStatic));
  }
  out.model = ReadModelBlock(model, directory);
  if (timed && out.model.viscosity.empty() &&
      out.model.format != ModelFormat::Gia) {
    Fail(model.At("format"),
         "names a deck format with no viscosity column, which the "
         "quasi-static problem needs: give a \"maxwell\" deck with a "
         "viscosity, or a `model.viscosity` file");
  }
  out.lmax = Integer(top, "lmax");
  if (out.lmax < 1) Fail(top.At("lmax"), "is not a degree");
  if (timed) {
    out.times = ReadTimesBlock(top.At("times"), TimesForm::BeforePresent);
  }
  out.forcing = ReadSeaLevelForcing(
      top.At("forcing"), directory,
      timed ? std::optional<double>{out.times->start_kyr_bp}
            : std::optional<double>{});
  if (!Has(top, "ocean")) {
    Fail(top,
         "has no `ocean` block: the sea level equation needs its state, the "
         "sea level, `ocean.sea_level` (a sea level, used as it stands) or "
         "`ocean.topography` (converted with the state's ice), and the ice, "
         "`ocean.ice`, at one instant");
  }
  out.ocean = ReadOceanBlock(top.At("ocean"), directory, out.problem);
  // the terms of the ice change held to the state's instant
  for (auto k = std::size_t{0}; k < out.forcing.ice_change.size(); ++k) {
    const auto& term = out.forcing.ice_change[k];
    const auto& member = IceChangeMember(top).At(k).At(term.Member());
    if (!timed && term.kind == IceChangeTerm::Kind::Thickness &&
        !out.ocean.date_kyr_bp) {
      Fail(member,
           "measures its change from the state's date, `ocean.date_kyr_bp`, "
           "which the `ocean` block does not give (a state of files without "
           "a time axis has none): give the change itself, a `change` term");
    }
    if (timed && term.kind != IceChangeTerm::Kind::Thickness &&
        out.ocean.at == WindowEnd::End) {
      Fail(member,
           std::format("is {}, which the state at the window's end "
                       "(`ocean.at`: \"end\") does not take: the ice changes "
                       "are then thickness histories, `thickness` terms, each "
                       "read back from the end",
                       term.kind == IceChangeTerm::Kind::Change
                           ? "a change taken as it stands"
                           : "a cap of ice change"));
    }
  }
  if (Has(top, "rotation")) out.rotation = RotationOf(top.At("rotation"));
  if (Has(top, "moments")) {
    out.hydrostatic_moments = HydrostaticMomentsOf(top.At("moments"));
  }
  if (Has(top, "solver")) {
    const auto& solver = top.At("solver");
    if (!timed) {
      if (Has(solver, "scheme")) {
        Fail(solver.At("scheme"),
             std::format("sets the time step, which {}", kNeedsQuasiStatic));
      }
      // the static problem takes the 3-D step on a laterally varying
      // moduli file, which BuildEarth reads; without one it is refused here
      if (out.model.moduli.empty()) {
        for (const auto* name : {"step", "step_rtol", "step_max_iterations",
                                 "transform_degrees", "cache_mb"}) {
          if (Has(solver, name)) {
            Fail(solver.At(name),
                 std::format("sets the 3-D step, which {} or lateral moduli "
                             "(`model.moduli`): the static problem on a "
                             "radial Earth is one solve of the sea level "
                             "equation",
                             kNeedsQuasiStatic));
          }
        }
      }
    }
    out.solver = ReadSolverBlock(solver, out.lmax, SolverOf::SeaLevel);
  }
  return out;
}

void RefuseThreeDSolver(const Json& top, std::string_view what) {
  if (!Has(top, "solver")) return;
  const auto& solver = top.At("solver");
  for (const auto* name : {"step", "step_rtol", "step_max_iterations",
                           "transform_degrees", "cache_mb"}) {
    if (Has(solver, name)) {
      Fail(solver.At(name),
           std::format("sets the 3-D step, which {} does not have", what));
    }
  }
}

std::string BuiltEarth::Dimension() const {
  if (Lateral()) {
    return std::format(
        "3-D ({})",
        LateralViscosity() && LateralModuli()
            ? "lateral viscosity, lateral moduli"
            : (LateralViscosity() ? "lateral viscosity" : "lateral moduli"));
  }
  if (viscosity && moduli) {
    return "1-D (the viscosity and moduli files are radial)";
  }
  if (viscosity) return "1-D (the viscosity file is radial)";
  if (moduli) return "1-D (the moduli file is radial)";
  return "1-D";
}

BuiltEarth BuildEarth(const SeaLevelRun& run, const Json& top) {
  const auto& modelMember = top.At("model");
  const auto& deckMember = modelMember.At("deck");
  const auto deck = Deck::Read(run.model.deck, FormatOf(run.model.format));
  if (!deck) Fail(deckMember, deck.error().message);
  auto model = EarthModel::FromDeck(*deck);
  if (!model) Fail(deckMember, model.error().message);
  auto mesh_options = RadialMesh::Options{};
  mesh_options.lmax = run.lmax;
  const auto mesh = RadialMesh::Create(*model, mesh_options);
  if (!mesh) Fail(deckMember, mesh.error().message);
  auto material = Material::Create(*model, *mesh);
  if (!material) Fail(deckMember, material.error().message);
  auto out = BuiltEarth{std::move(*model), std::move(*material), {}, {}};
  if (!run.model.viscosity.empty()) {
    auto read = io::ModelViscosity::Read(
        run.model.viscosity, out.model,
        {.reference = run.model.viscosity_file_radial
                          ? io::ViscosityReference::FileRadial
                          : io::ViscosityReference::GeometricMean});
    if (!read) {
      Fail(modelMember.At("viscosity").At("file"), read.error().message);
    }
    out.viscosity.emplace(std::move(*read));
  }
  if (!run.model.moduli.empty()) {
    auto read = io::ModelModuli::Read(
        run.model.moduli, out.model,
        {.reference = run.model.moduli_file_radial
                          ? io::ModulusReference::FileRadial
                          : io::ModulusReference::LateralMean});
    if (!read) Fail(modelMember.At("moduli").At("file"), read.error().message);
    out.moduli.emplace(std::move(*read));
  }
  return out;
}

Rheology RheologyOf(const BuiltEarth& earth, const Json& top) {
  const auto& modelMember = top.At("model");
  auto rheology =
      earth.viscosity
          ? Rheology::Maxwell(earth.material, earth.viscosity->Viscosity())
          : Rheology::FromModel(earth.material, earth.model);
  if (!rheology) {
    if (earth.viscosity) {
      Fail(modelMember.At("viscosity").At("file"), rheology.error().message);
    }
    Fail(modelMember.At("deck"),
         std::format("gives no viscosity, which the quasi-static problem "
                     "needs: {}",
                     rheology.error().message));
  }
  return std::move(*rheology);
}

Viscoelastic3D::Options OperatorOptionsOf(const SeaLevelRun& run,
                                          const BuiltEarth& earth,
                                          const Rheology& rheology) {
  const auto& solver = run.solver;
  auto lateral = Viscoelastic3D::Options::LateralModuli{};
  if (earth.LateralModuli()) {
    lateral = {.shear = earth.moduli->Shear(), .bulk = earth.moduli->Bulk()};
  }
  // On several ranks under lateral moduli, the rank cuts balance the
  // transform work of the elements the coupling carries: the viscous ones
  // and those the moduli perturb.
  auto active = std::vector<char>{};
  if (earth.LateralModuli() && TheRanks().Distributed()) {
    auto probe = Viscoelastic3D::Options{};
    probe.transform_degrees = solver.transform_degrees;
    probe.cache_bytes = solver.cache_mb << 20;
    probe.moduli = lateral;
    const SurfaceGrid grid(run.lmax, {.nmax = 2,
                                      .oversampling = probe.oversampling,
                                      .parallel = probe.parallel,
                                      .matrix_kernel = probe.matrix_kernel,
                                      .cache_bytes = probe.cache_bytes});
    active = Viscoelastic3D::ActiveElementsFor(rheology, run.lmax, grid, probe);
  }
  auto options = OperatorOptions(run.lmax, solver.transform_degrees,
                                 solver.cache_mb, rheology, active);
  options.solver = solver.step;
  options.moduli = lateral;
  return options;
}

std::function<double(double, double, double)> DeltaOf(const BuiltEarth& earth) {
  if (earth.viscosity) return earth.viscosity->Delta();
  return [](double, double, double) { return 0.0; };
}

SeaLevelParameters ParametersOf(const SeaLevelRun& run,
                                const Material& material) {
  auto parameters = SeaLevelParameters{};
  if (run.hydrostatic_moments) {
    parameters =
        WithHydrostaticMoments(parameters, FluidTidalNumber(material, 2));
  }
  return parameters;
}

const Json& SeaLevelMember(const SeaLevelRun& run, const Json& top) {
  return top.At("ocean").At(run.ocean.seaLevel.Member());
}

const Json& IceMember(const Json& top) { return top.At("ocean").At("ice"); }

const Json& IceChangeMember(const Json& top) {
  return top.At("forcing").At("ice_change");
}

std::shared_ptr<const io::FieldHistory> InputFiles::Open(const FileInput& input,
                                                        const Json& object) {
  // one file for several inputs is read, resampled and held once
  for (const auto& [read, file] : read_) {
    if (SameFile(read, input)) {
      CheckVariable(input, *file, object);
      return file;
    }
  }
  auto file = std::make_shared<const io::FieldHistory>(
      OpenFileInput(input, *grid_, object));
  read_.emplace_back(input, file);
  return file;
}

std::vector<double> IceThicknessAt(const io::FieldHistory& ice,
                                   std::string_view variable, double time) {
  auto thickness = ice.Values(variable, time);
  for (auto v : thickness) {
    if (!std::isfinite(v)) {
      throw std::runtime_error(std::format(
          "the ice's {} is not a number somewhere at {:g} kyr BP (a surface "
          "file marks uncovered points so): fill it before the run",
          variable, -time / KYR));
    }
  }
  return thickness;
}

RunIce::RunIce(const SeaLevelRun& run, const Json& top, InputFiles& files,
               double stateTime)
    : grid_(&files.Grid()) {
  const auto& member = IceMember(top);
  const auto state = files.Open(run.ocean.ice, member);
  try {
    state_ = IceThicknessAt(*state, run.ocean.ice.variable, stateTime);
  } catch (const std::runtime_error& error) {
    Fail(member.At("file"), error.what());
  }
  if (run.forcing.ice_change.empty()) return;
  block_ = &IceChangeMember(top);
  for (auto k = std::size_t{0}; k < run.forcing.ice_change.size(); ++k) {
    auto held = Term{};
    held.term = run.forcing.ice_change[k];
    const auto& term = held.term;
    held.member = &block_->At(k).At(term.Member());
    const auto& item = *held.member;
    switch (term.kind) {
      case IceChangeTerm::Kind::Thickness:
      case IceChangeTerm::Kind::Change: {
        held.file = files.Open(term.file, item);
        const auto thickness = term.kind == IceChangeTerm::Kind::Thickness;
        if (thickness && !held.file->Span()) {
          Fail(item, std::format("names a file without a time axis ({}), "
                                 "whose change from the state's instant is "
                                 "identically zero: a thickness term needs "
                                 "a file with a time axis",
                                 term.file.file.string()));
        }
        if (!term.dated && !thickness) {
          CheckDate(term.file, *held.file, item, "date_kyr_bp");
        }
        if (thickness) held.reference = ValuesOf(held, stateTime);
        // the static problem's term is read once, at its own date (a file
        // without a time axis at any date: the present's)
        if (!term.dated) {
          held.fixed = TermAt(held, -term.file.date.value_or(0.0) * KYR);
        }
        break;
      }
      case IceChangeTerm::Kind::CapThickness:
      case IceChangeTerm::Kind::CapFraction: {
        // the Gaussian cap's weight times the amount, of the state's ice
        // for a fraction
        const auto mask = CapField(*grid_, term.site, term.width, 1.0);
        const auto fraction = term.kind == IceChangeTerm::Kind::CapFraction;
        held.fixed.resize(mask.size());
        for (auto i = std::size_t{0}; i < mask.size(); ++i) {
          held.fixed[i] = term.amount * mask[i] * (fraction ? state_[i] : 1.0);
          // a cap of thickness has no edge: where it melts, it takes no
          // more than the ice the state holds there
          if (!fraction) {
            held.fixed[i] = std::max(held.fixed[i], -std::max(state_[i], 0.0));
          }
        }
        break;
      }
    }
    terms_.push_back(std::move(held));
  }
}

std::vector<double> RunIce::ValuesOf(const Term& term, double time) const {
  try {
    return IceThicknessAt(*term.file, term.term.file.variable, time);
  } catch (const std::runtime_error& error) {
    Fail(term.member->At("file"), error.what());
  }
}

std::vector<double> RunIce::TermAt(const Term& term, double time) const {
  if (!term.fixed.empty()) {
    const auto& history = term.term.history;
    if (history.Empty()) return term.fixed;
    // a cap's history scales its change, at the time's place on the run's
    // clock (kyr from the window's start)
    const auto factor = history.At(*history.start_kyr_bp + time / KYR);
    auto out = term.fixed;
    for (auto& v : out) v *= factor;
    return out;
  }
  auto values = ValuesOf(term, time);
  if (term.term.kind == IceChangeTerm::Kind::Thickness) {
    for (auto i = std::size_t{0}; i < values.size(); ++i) {
      values[i] -= term.reference[i];
    }
  }
  return values;
}

std::vector<double> RunIce::ChangeAt(double time) const {
  auto out = std::vector<double>{};
  // the first term is taken as it stands, the others added to it
  for (const auto& term : terms_) {
    auto change = TermAt(term, time);
    if (out.empty()) {
      out = std::move(change);
    } else {
      for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += change[i];
    }
  }
  if (out.empty()) out = grid_->ZeroField();
  return out;
}

std::vector<double> RunIce::ChangeSince(double time, double since) const {
  auto out = std::vector<double>{};
  for (const auto& term : terms_) {
    // a cap: the same at every time, but for its history
    if (!term.file && term.term.history.Empty()) continue;
    auto change = term.file ? ValuesOf(term, time) : TermAt(term, time);
    const auto before =
        term.file ? ValuesOf(term, since) : TermAt(term, since);
    for (auto i = std::size_t{0}; i < change.size(); ++i) {
      change[i] -= before[i];
    }
    if (out.empty()) {
      out = std::move(change);
    } else {
      for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += change[i];
    }
  }
  if (out.empty()) out = grid_->ZeroField();
  return out;
}

std::vector<double> RunIce::At(double time) const {
  auto out = ChangeAt(time);
  for (auto i = std::size_t{0}; i < out.size(); ++i) out[i] += state_[i];
  return out;
}

double RunIce::Tolerance() const {
  const auto largest =
      state_.empty() ? 0.0 : *std::ranges::max_element(state_);
  return largest > 0.0 ? 1e-9 * largest : 1e-9;
}

void RunIce::CheckHeld(std::span<const double> times, bool dated) const {
  if (terms_.empty()) return;
  auto worst = 0.0, when = 0.0;
  auto where = std::size_t{0};
  for (const auto time : times) {
    const auto change = ChangeAt(time);
    for (auto i = std::size_t{0}; i < change.size(); ++i) {
      const auto held = std::max(state_[i], 0.0) + change[i];
      if (held < worst) {
        worst = held;
        where = i;
        when = time;
      }
    }
  }
  if (worst >= -Tolerance()) return;
  const auto degrees = 180.0 / std::numbers::pi;
  const auto nphi = grid_->Longitudes().size();
  const auto latitude = 90.0 - grid_->CoLatitudes()[where / nphi] * degrees;
  const auto longitude = grid_->Longitudes()[where % nphi] * degrees;
  Fail(*block_,
       std::format("removes more ice than the state holds, by {:.4g} m at "
                   "worst (at latitude {:.4g}, longitude {:.4g}{})",
                   -worst, latitude, longitude,
                   dated ? std::format(", {:.6g} kyr BP", 0.0 - when / KYR)
                         : std::string{}));
}

std::string DescribeIceChange(const SeaLevelRun& run,
                              std::string_view separator) {
  auto out = std::string{};
  for (const auto& term : run.forcing.ice_change) {
    if (!out.empty()) out += separator;
    out += Describe(term);
  }
  return out;
}

void CheckStateDate(const SeaLevelRun& run, const Json& top,
                    InputFiles& files) {
  const auto& ocean = top.At("ocean");
  // the state's members, the first file with a time axis named
  auto timed = std::optional<std::filesystem::path>{};
  const auto look = [&](const FileInput& input, const Json& member) {
    if (!timed && files.Open(input, member)->Span()) timed = input.file;
  };
  look(run.ocean.seaLevel.input, SeaLevelMember(run, top));
  look(run.ocean.ice, IceMember(top));
  if (run.ocean.land) look(*run.ocean.land, ocean.At("land"));
  if (timed && !run.ocean.date_kyr_bp) {
    Fail(ocean, std::format("holds a file with a time axis ({}): give the "
                            "state's date, `date_kyr_bp`, at which every "
                            "member's file with a time axis is read",
                            timed->string()));
  }
  if (!timed && run.ocean.date_kyr_bp) {
    Fail(ocean.At("date_kyr_bp"),
         "dates a state whose files have no time axis, the same at every "
         "date: drop it");
  }
}

LandMask ReadLand(const SeaLevelRun& run, const Json& top, InputFiles& files,
                  double time) {
  if (!run.ocean.land) return {};
  const auto& input = *run.ocean.land;
  const auto& member = top.At("ocean").At("land");
  const auto file = files.Open(input, member);
  auto values = std::vector<double>{};
  try {
    values = file->Values(input.variable, time);
  } catch (const std::runtime_error& error) {
    Fail(member.At("file"), error.what());
  }
  auto out = LandMask{};
  out.held.reserve(values.size());
  for (auto v : values) {
    if (!std::isfinite(v)) {
      Fail(member,
           "gives a land field that is not a number somewhere (a surface "
           "file marks uncovered points so): fill it before the run");
    }
    out.held.push_back(v > 0.5 ? 1 : 0);
  }
  out.read = std::format("land {}", DescribeInput(input, "held at every date"));
  return out;
}

std::vector<double> SeaLevelFrom(const SeaLevelRun& run, const Json& top,
                                 const io::FieldHistory& file,
                                 std::span<const double> ice, double time,
                                 const SeaLevelParameters& parameters) {
  const auto& source = run.ocean.seaLevel;
  auto sea = file.Values(source.input.variable, time);
  if (source.kind == SeaLevelSource::Kind::Topography) {
    sea = SeaLevelFromTopography(std::move(sea),
                                 std::vector<double>(ice.begin(), ice.end()),
                                 parameters);
  }
  for (auto v : sea) {
    if (!std::isfinite(v)) {
      Fail(SeaLevelMember(run, top).At("file"),
           std::format("gives a sea level that is not a number somewhere at "
                       "{:g} kyr BP (a surface file marks uncovered points "
                       "so): fill it before the run",
                       -time / KYR));
    }
  }
  return sea;
}

namespace {

/// The slice a run through time reads an input at: its window, or one of
/// its ends, from a file with a time axis; any date from a static file.
std::string WindowSlice(const SeaLevelRun& run, const io::FieldHistory& file,
                        std::optional<WindowEnd> end) {
  const auto& window = *run.times;
  const auto timeAxis = file.Span().has_value();
  if (!end) {
    return timeAxis ? std::format("{:g} to {:g} kyr BP", window.start_kyr_bp,
                                  window.end_kyr_bp)
                    : std::string{"static"};
  }
  const auto start = *end == WindowEnd::Start;
  const auto name = start ? "start" : "end";
  return timeAxis ? std::format("the window's {}, {:g} kyr BP", name,
                                start ? window.start_kyr_bp : window.end_kyr_bp)
                  : std::format("static, at the window's {}", name);
}

/// Refuses a state's file whose time axis leaves out the window's end,
/// where the state is read: its end values are held beyond its span,
/// which would make a file ending early the calibration's target
/// silently. `member` is the input's.
void CheckReachesEnd(const SeaLevelRun& run, const io::FieldHistory& file,
                     const Json& member) {
  const auto end_kyr_bp = run.times->end_kyr_bp;
  const auto end_si = -end_kyr_bp * KYR;
  if (const auto span = file.Span(); span.has_value()) {
    const auto slack = 1e-9 * KYR;
    if (end_si < span->first - slack || end_si > span->second + slack) {
      Fail(member.At("file"),
           std::format("spans {:g} to {:g} kyr BP, which leaves out the "
                       "window's end ({:g} kyr BP), where the state is read, "
                       "the calibration's target",
                       -span->first / KYR, -span->second / KYR, end_kyr_bp));
    }
  }
}

/// The state's sea level at its instant, SI, a topography converted with
/// the state's ice.
std::vector<double> StateSeaLevel(const SeaLevelRun& run, const Json& top,
                                  const SeaLevelParameters& parameters,
                                  const io::FieldHistory& sea,
                                  const RunIce& ice) {
  const auto time = run.ocean.at == WindowEnd::Start
                        ? -run.times->start_kyr_bp * KYR
                        : -run.times->end_kyr_bp * KYR;
  try {
    return SeaLevelFrom(run, top, sea, ice.State(), time, parameters);
  } catch (const std::runtime_error& error) {
    Fail(SeaLevelMember(run, top).At("file"),
         std::format("gives no sea level at the state's instant: {}",
                     error.what()));
  }
}

/// The ice's change from the ice the run starts from, SI, at SI time
/// `time`: with the state at the window's start, the change itself (from
/// the state's own ice); with the state at its end, each term's change
/// since the window's start (`start_si`).
std::vector<double> ChangeFromStart(const RunIce& ice, bool stateAtStart,
                                    double start_si, double time) {
  return stateAtStart ? ice.ChangeAt(time) : ice.ChangeSince(time, start_si);
}

}  // namespace

Loaded LoadWorld(const SeaLevelRun& run, const Json& top,
                 const GRDSolver& solver) {
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto& window = *run.times;
  const auto atStart = run.ocean.at == WindowEnd::Start;
  const auto start_si = -window.start_kyr_bp * KYR;
  const auto state_si = atStart ? start_si : -window.end_kyr_bp * KYR;
  auto files = InputFiles(grid);
  const auto& source = run.ocean.seaLevel;
  const auto sea = files.Open(source.input, SeaLevelMember(run, top));
  const auto iceFile = files.Open(run.ocean.ice, IceMember(top));
  if (!atStart) {
    CheckReachesEnd(run, *sea, SeaLevelMember(run, top));
    CheckReachesEnd(run, *iceFile, IceMember(top));
  }
  auto ice = std::make_shared<const RunIce>(run, top, files, state_si);
  // the ice at every node the run steps: no change takes more than the
  // state holds
  auto nodes = std::vector<double>{};
  for (const auto t : MakeTimes(run, scales)) {
    nodes.push_back(start_si + t * scales.Time());
  }
  ice->CheckHeld(nodes, true);
  // the change present at the start, where the state is there: a load at
  // the first node, or the ice the start's solve takes
  auto changeAtStart = 0.0;
  if (atStart) {
    for (const auto v : ice->ChangeAt(start_si)) {
      changeAtStart = std::max(changeAtStart, std::abs(v));
    }
  }
  // the ice the run starts from: the state's, or, with the state at the
  // window's end, the state's with the change back to the start
  auto ice0 = atStart ? ice->State() : ice->At(start_si);
  // the state's sea level: the initial sea level, or the calibration's
  // target and first guess
  auto sl_nd = StateSeaLevel(run, top, solver.Parameters(), *sea, *ice);
  auto ice_nd = std::vector<double>(fieldSize);
  for (auto i = std::size_t{0}; i < fieldSize; ++i) {
    ice_nd[i] = ice0[i] / length;
    sl_nd[i] /= length;
  }
  const auto land = ReadLand(run, top, files, start_si);
  auto background =
      solver.MakeState(std::move(ice_nd), std::move(sl_nd), land.held);
  auto iceRead =
      DescribeInput(run.ocean.ice, WindowSlice(run, *iceFile, run.ocean.at));
  auto seaRead = std::format(
      "{} {}", source.Quantity(),
      DescribeInput(source.input, WindowSlice(run, *sea, run.ocean.at)));
  if (!land.read.empty()) seaRead += ", " + land.read;
  return {sea,
          std::move(ice),
          std::move(ice0),
          std::move(background),
          std::move(iceRead),
          std::move(seaRead),
          changeAtStart};
}

std::vector<double> FinalSeaLevelOf(const SeaLevelRun& run, const Json& top,
                                    const GRDSolver& solver,
                                    const Loaded& world) {
  auto sea = StateSeaLevel(run, top, solver.Parameters(), *world.seaLevel,
                           *world.ice);
  const auto length = solver.SolverScales().Length();
  for (auto& v : sea) v /= length;
  return sea;
}

double DateKyrBp(const SeaLevelRun& run, const Scales& scales, double t) {
  return run.times->start_kyr_bp - t * scales.Time() / KYR;
}

std::vector<double> MakeTimes(const SeaLevelRun& run, const Scales& scales) {
  const auto& window = *run.times;
  auto times = std::vector<double>{};
  const auto span_s = (window.start_kyr_bp - window.end_kyr_bp) * KYR;
  for (auto n = 0; n < window.nodes; ++n) {
    times.push_back(span_s * n / (window.nodes - 1) / scales.Time());
  }
  return times;
}

std::function<std::vector<double>(double)> IceLoadOf(
    const SeaLevelRun& run, const GRDSolver& solver, const Loaded& world) {
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto time = scales.Time();
  const auto start_si = -run.times->start_kyr_bp * KYR;
  const auto atStart = run.ocean.at == WindowEnd::Start;
  const auto rho_ice = solver.IceDensity();
  return [&world, atStart, fieldSize, length, time, start_si,
          rho_ice](double t) {
    const auto change =
        ChangeFromStart(*world.ice, atStart, start_si, start_si + t * time);
    const auto& C = world.background.oceanFunction;
    auto load = std::vector<double>(fieldSize);
    for (auto i = std::size_t{0}; i < fieldSize; ++i) {
      load[i] = rho_ice * (1.0 - C[i]) * change[i] / length;
    }
    return load;
  };
}

std::function<std::vector<double>(double)> IceThicknessOf(
    const SeaLevelRun& run, const GRDSolver& solver, const Loaded& world) {
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto time = scales.Time();
  const auto start_si = -run.times->start_kyr_bp * KYR;
  const auto atStart = run.ocean.at == WindowEnd::Start;
  // the background's ice plus the change from world.ice0, which the
  // frozen load is measured from too (IceLoadOf)
  return [&world, atStart, length, time, start_si](double t) {
    auto thickness =
        ChangeFromStart(*world.ice, atStart, start_si, start_si + t * time);
    const auto& background = world.background.iceThickness;
    for (auto i = std::size_t{0}; i < thickness.size(); ++i) {
      thickness[i] = background[i] + thickness[i] / length;
    }
    return thickness;
  };
}

IceHistory HistoryOf(const SeaLevelRun& run, const GRDSolver& solver,
                     const Loaded& world) {
  return run.ocean.shorelines == Shorelines::Frozen
             ? IceHistory::Frozen(IceLoadOf(run, solver, world))
             : IceHistory::Migrating(IceThicknessOf(run, solver, world));
}

namespace {

CalibrationOptions CalibrationOptionsOf(const SeaLevelRun& run) {
  auto options = CalibrationOptions{};
  options.method = run.ocean.calibration_method;
  options.iterations = run.ocean.calibration_iterations;
  options.rtol = run.ocean.calibration_rtol;
  options.adjoint.rtol = run.solver.rtol;
  options.adjoint3D.rtol = run.solver.rtol;
  options.adjoint3D.coupled_rtol = run.solver.step_rtol;
  options.adjoint3D.max_coupled_iterations = run.solver.step_max_iterations;
  return options;
}

template <class Calibration>
CalibrationSummary Adopt(const SeaLevelRun& run, const Json& top,
                         const GRDSolver& solver, Loaded& world,
                         Calibration calibration) {
  for (const auto& snapshot : calibration.snapshots) {
    if (!snapshot.converged) {
      throw std::runtime_error(std::format(
          "the calibration's last run did not converge at {:.6g} kyr BP (its "
          "sea level solve, or in 3-D its coupled fixed point); loosen "
          "solver.rtol or solver.step_rtol",
          DateKyrBp(run, solver.SolverScales(), snapshot.t)));
    }
  }
  if (!calibration.converged) {
    const auto& ocean = top.At("ocean");
    Fail(Has(ocean, "calibration") ? ocean.At("calibration") : ocean.At("at"),
         std::format("calibrated without converging in {} runs (relative "
                     "update {:.3g}, tolerance {:.3g}); raise "
                     "ocean.calibration.iterations",
                     calibration.updates.size(), calibration.updates.back(),
                     run.ocean.calibration_rtol));
  }
  world.background = std::move(calibration.state);
  const auto length = solver.SolverScales().Length();
  // one run under frozen shorelines, one per update when migrating
  return {calibration.misfits.front() * length,
          calibration.misfits.back() * length, calibration.updates.size()};
}

}  // namespace

void StartFromIceHistory(const SeaLevelRun& run, const GRDSolver& solver,
                         Loaded& world, std::span<const double> seaLevelGuess) {
  const auto length = solver.SolverScales().Length();
  world.ice0 = run.ocean.at == WindowEnd::Start
                   ? world.ice->State()
                   : world.ice->At(-run.times->start_kyr_bp * KYR);
  auto ice = world.ice0;
  for (auto& v : ice) v /= length;
  // the land held stays the run's own
  const auto land = world.background.forcedLand;
  world.background = solver.MakeState(
      std::move(ice),
      std::vector<double>(seaLevelGuess.begin(), seaLevelGuess.end()), land);
}

namespace {

/// The calibration's start, pyslfp's construction: the run's ice at the
/// window's start under the target sea level, as the guess.
std::vector<double> PrepareCalibration(const SeaLevelRun& run, const Json& top,
                                       const GRDSolver& solver,
                                       Loaded& world) {
  auto target = FinalSeaLevelOf(run, top, solver, world);
  StartFromIceHistory(run, solver, world, target);
  return target;
}

}  // namespace

CalibrationSummary CalibrateWorld(const SeaLevelRun& run, const Json& top,
                                  const GRDSolver& solver, Loaded& world,
                                  Evolution& evolution,
                                  std::span<const double> times) {
  const auto target = PrepareCalibration(run, top, solver, world);
  evolution.SetBackgroundState(world.background);
  return Adopt(
      run, top, solver, world,
      CalibrateInitialSeaLevel(evolution, times, HistoryOf(run, solver, world),
                               target, CalibrationOptionsOf(run)));
}

CalibrationSummary CalibrateWorld(const SeaLevelRun& run, const Json& top,
                                  const GRDSolver& solver, Loaded& world,
                                  Evolution3D& evolution,
                                  std::span<const double> times) {
  const auto target = PrepareCalibration(run, top, solver, world);
  evolution.SetBackgroundState(world.background);
  return Adopt(
      run, top, solver, world,
      CalibrateInitialSeaLevel(evolution, times, HistoryOf(run, solver, world),
                               target, CalibrationOptionsOf(run)));
}

}  // namespace sl3dpp::app
