#include "Common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <iostream>
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

std::vector<std::string_view> RunMemberNames() {
  return {"problem", "discretisation", "model",   "times",  "forcing",
          "ocean",   "rotation",       "moments", "solver", "gravity"};
}

void AllowTopLevel(const Json& top, std::span<const std::string_view> allowed) {
  // the members a block holds, by the block they belong in
  constexpr auto kInBlocks = std::array{
      std::pair{std::string_view{"lmax"}, "`discretisation.lmax`"},
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
                "`ocean.calibration`"},
      std::pair{std::string_view{"adjoint"}, "`kernels.method`"},
      std::pair{std::string_view{"measurements"}, "`kernels.measurements`"},
      std::pair{std::string_view{"record_stride"},
                "`trajectory.record_stride`"},
      std::pair{std::string_view{"parameters"}, "`kernels.parameters`"},
      std::pair{std::string_view{"method"}, "`kernels.method`"}};
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

Run ReadRun(const Json& top, const std::filesystem::path& directory) {
  auto out = Run{};
  out.problem = ReadProblem(top);
  const auto timed = out.problem == Problem::QuasiStatic;
  const auto& model = top.At("model");
  if (!timed && model.IsObject() && Has(model, "viscosity")) {
    Fail(model.At("viscosity"),
         std::format("is a viscosity, which {}; the static problem's Earth "
                     "is elastic and has no use for it",
                     kNeedsQuasiStatic));
  }
  if (!timed && model.IsObject() && Has(model, "burgers")) {
    Fail(model.At("burgers"),
         std::format("is a rheology, Burgers layers, which {}; the static "
                     "problem's Earth is elastic and has no use for it",
                     kNeedsQuasiStatic));
  }
  out.model = ReadModelBlock(model, directory);
  if (timed && out.model.viscosity.empty() &&
      (out.model.format == ModelFormat::Mineos ||
       out.model.format == ModelFormat::Elastic)) {
    Fail(model.At("format"),
         "names a deck format with no viscosity column, which the "
         "quasi-static problem needs: give a \"maxwell\" deck with a "
         "viscosity, a \"prony\" or \"burgers\" deck, or a "
         "`model.viscosity` file");
  }
  out.lmax = ReadDiscretisationBlock(top, 1).lmax;
  // the terms' times are given as the window's are: dates in a dated run,
  // times on the run's own clock in its unit in one counting forwards
  auto clock = std::optional<TermClock>{};
  if (timed) {
    out.times = ReadWindow(top.At("times"));
    clock = out.times->Dated()
                ? TermClock{.form = TermClock::Form::Dated,
                            .start_kyr_bp = out.times->Start(),
                            .unit = {}}
                : TermClock{.form = TermClock::Form::Forwards,
                            .start_kyr_bp = std::nullopt,
                            .unit = out.times->unit};
  }
  // the state is optional: without an `ocean` block the body has none
  const auto ocean = Has(top, "ocean");
  // without gravity there is no sea level, and so no state
  if (Has(top, "gravity")) out.gravitating = GravityOf(top.At("gravity"));
  if (!out.gravitating && ocean) {
    Fail(top.At("ocean"),
         "is the state of the sea level equation, which a run without "
         "gravity (`gravity`: \"none\") does not have: drop the block, the "
         "body then having no ocean");
  }
  out.forcing =
      ReadForcing(top.At("forcing"), directory, out.lmax, clock, ocean);
  if (ocean) {
    out.ocean = ReadOceanBlock(
        top.At("ocean"), directory, out.problem,
        timed ? out.times->calendar : Calendar::Dated);
  }
  if (!out.gravitating && !out.forcing.tide.empty()) {
    // the forcing of a run without gravity: surface loads (their weight)
    // and seismic sources
    Fail(top.At("forcing").At("tide"),
         "is a tide, an external potential, which a run without gravity "
         "(`gravity`: \"none\") does not take");
  }
  // the calibration's runs are under the ice alone
  if (out.Calibrating() && out.forcing.BesideTheIce()) {
    const auto& forcing = top.At("forcing");
    Fail(forcing.At(!out.forcing.load.empty()
                        ? "load"
                        : (!out.forcing.tide.empty() ? "tide" : "seismic")),
         "is a forcing beside the ice, which a calibrated initial sea "
         "level does not take: the calibration's runs are under the ice "
         "alone. Give the state at the window's start (`ocean.at`: "
         "\"start\")");
  }
  // the terms of the ice change held to the state's instant (an ice
  // change has an ocean)
  for (auto k = std::size_t{0}; k < out.forcing.ice_change.size(); ++k) {
    const auto& term = out.forcing.ice_change[k];
    const auto& member = IceChangeMember(top).At(k).At(term.Member());
    if (!timed && term.kind == IceChangeTerm::Kind::Thickness &&
        !out.ocean->date_kyr_bp) {
      Fail(member,
           "measures its change from the state's date, `ocean.date_kyr_bp`, "
           "which the `ocean` block does not give (a state of files without "
           "a time axis has none): give the change itself, a `change` term");
    }
    if (timed && term.kind != IceChangeTerm::Kind::Thickness &&
        out.ocean->at == WindowEnd::End) {
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
  // without gravity the rotational feedback is none, and only none: the
  // inertia's change is read from the degree-2 potential, which a run
  // without gravity has not
  if (!out.gravitating) out.rotation = RotationalFeedback::None;
  if (Has(top, "rotation")) out.rotation = RotationOf(top.At("rotation"));
  if (!out.gravitating && out.rotation != RotationalFeedback::None) {
    Fail(top.At("rotation"),
         std::format("is \"{}\", a rotational feedback, which a run without "
                     "gravity (`gravity`: \"none\") does not take: the "
                     "inertia's change is read from the degree-2 potential, "
                     "which it has not; give \"none\" (the default without "
                     "gravity)",
                     NameOf(out.rotation)));
  }
  if (Has(top, "moments")) {
    out.hydrostatic_moments = HydrostaticMomentsOf(top.At("moments"));
    if (!out.gravitating && out.hydrostatic_moments) {
      Fail(top.At("moments"),
           "sets the moments of inertia from the model's hydrostatic fluid "
           "limit for the rotational feedback, which a run without gravity "
           "(`gravity`: \"none\") does not have");
    }
  }
  if (Has(top, "solver")) {
    const auto& solver = top.At("solver");
    if (!timed) {
      if (Has(solver, "scheme")) {
        Fail(solver.At("scheme"),
             std::format("sets the time step, which {}", kNeedsQuasiStatic));
      }
      // the static problem takes the 3-D step on a laterally varying
      // moduli file, which BuildEarth reads; without one it is refused here,
      // but for the grids' members where the kernels are lateral, whose
      // sweep expands on the grids
      if (out.model.moduli.empty()) {
        const auto transforms = LateralKernelsAsked(top);
        for (const auto* name : {"step", "step_rtol", "step_max_iterations",
                                 "transform_degrees", "cache_mb"}) {
          const auto grid = std::string_view{name} == "transform_degrees" ||
                            std::string_view{name} == "cache_mb";
          if (transforms && grid) continue;
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
    out.solver = ReadSolverBlock(solver, out.lmax);
  }
  // the gradient calibration's gradient under migrating shorelines comes
  // from the discrete adjoint, which a scheme of several stages has none
  // of (StepRecipe.hpp); under frozen shorelines it runs no sweep
  if (out.Calibrating() && out.Migrating() &&
      out.ocean->calibration_method == CalibrationMethod::Gradient &&
      IsMultiStage(out.solver.scheme)) {
    Fail(top.At("ocean").At("calibration").At("method"),
         "is the gradient calibration, whose gradient under migrating "
         "shorelines is the discrete adjoint's, which is not built for "
         "`solver.scheme` \"sdirk2\" (its exact derivative runs through "
         "each step's stages): give \"fixed_point\", whose calibrated state "
         "is the same, or a scheme of one stage, \"exponential_trapezoid\" "
         "or \"backward_euler\"");
  }
  return out;
}

bool LateralKernelsAsked(const Json& top) {
  if (!Has(top, "kernels")) return false;
  const auto& kernels = top.At("kernels");
  if (!kernels.IsObject() || !Has(kernels, "lateral")) return false;
  const auto& lateral = kernels.At("lateral");
  return lateral.IsBool() && lateral.AsBool();
}

void RefuseThreeDSolver(const Json& top, std::string_view what,
                        bool transforms) {
  if (!Has(top, "solver")) return;
  const auto& solver = top.At("solver");
  for (const auto* name : {"step", "step_rtol", "step_max_iterations",
                           "transform_degrees", "cache_mb"}) {
    const auto transform = std::string_view{name} == "transform_degrees" ||
                           std::string_view{name} == "cache_mb";
    if (transforms && transform) continue;
    if (Has(solver, name)) {
      Fail(solver.At(name),
           std::format("sets the 3-D step, which {} does not have", what));
    }
  }
}

std::function<LoveModuli(const BodyPoint&)> ModuliOf(const BuiltEarth& earth) {
  const auto* model = &earth.model;
  auto shear = std::function<double(double, double, double)>{};
  auto bulk = std::function<double(double, double, double)>{};
  if (earth.LateralModuli()) {
    shear = earth.moduli->Shear();
    bulk = earth.moduli->Bulk();
  }
  return [model, shear, bulk](const BodyPoint& at) {
    auto m = LoveModuliAt(*model, at.radius);
    if (!shear && !bulk) return m;
    const auto r = at.radius / model->ModelScales().Length();
    const auto s = shear ? shear(r, at.colatitude, at.longitude) : 0.0;
    const auto b = bulk ? bulk(r, at.colatitude, at.longitude) : 0.0;
    // the perturbations of the Voigt-averaged moduli, an isotropic change
    const auto mu = s * (m.A + m.C - 2.0 * m.F + 5.0 * m.N + 6.0 * m.L) / 15.0;
    const auto kappa = b * (4.0 * m.A + m.C + 4.0 * m.F - 4.0 * m.N) / 9.0;
    m.A += kappa + 4.0 * mu / 3.0;
    m.C += kappa + 4.0 * mu / 3.0;
    m.F += kappa - 2.0 * mu / 3.0;
    m.L += mu;
    m.N += mu;
    return m;
  };
}

void PlaceSeismicSources(Run& run, const BuiltEarth& earth, const Json& top) {
  if (std::ranges::none_of(run.forcing.seismic, [](const SeismicTerm& term) {
        return term.kind == SeismicTerm::Kind::FaultFile;
      })) {
    return;
  }
  // the nodes on the terms' clock (an onset between two, where the run
  // reads the file too, lies within the span they cover)
  const auto times =
      run.times ? NodesOnTheClock(*run.times) : std::vector<double>{};
  PlaceFaultFiles(run.forcing.seismic, earth.model.SurfaceRadius(),
                  ModuliOf(earth), top.At("forcing"), times);
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

std::map<std::size_t, Rheology::Kelvin> KelvinOf(const Run& run,
                                                 const EarthModel& model,
                                                 const Json& top) {
  auto out = std::map<std::size_t, Rheology::Kelvin>{};
  if (run.model.burgers.empty()) return out;
  const auto& member = top.At("model").At("burgers");
  const auto b = model.Boundaries();
  const auto surface = model.SurfaceRadius();
  const auto tolerance = 1e-9 * surface;
  const auto km = [&](double r) { return (surface - r) / 1e3; };
  // a layer as a refusal names it: its index and depths
  const auto named = [&](std::size_t layer) {
    return std::format("layer {} ({:g} to {:g} km deep)", layer,
                       km(b[layer + 1]), km(b[layer]));
  };
  // the boundary at a depth, or the refusal of a depth inside a layer
  const auto boundaryAt = [&](const Json& depth, double d) {
    for (auto j = std::size_t{0}; j < b.size(); ++j) {
      if (std::abs(surface - d - b[j]) <= tolerance) return j;
    }
    const auto r = surface - d;
    if (!(r > b.front() && r < b.back())) {
      Fail(depth, std::format("reaches {:g} km deep, beyond the deck, whose "
                              "depths are 0 to {:g} km",
                              d / 1e3, km(b.front())));
    }
    Fail(depth,
         std::format("puts an end at {:g} km, inside {}: a Burgers region is "
                     "whole layers of the deck, its ends on the deck's "
                     "boundaries",
                     d / 1e3, named(model.LayerOf(r))));
  };
  const auto& items = member.AsArray();
  for (auto k = std::size_t{0}; k < run.model.burgers.size(); ++k) {
    const auto& region = run.model.burgers[k];
    const auto& depth = items[k].At("depth_km");
    const auto upper = boundaryAt(depth, region.top);
    const auto lower = boundaryAt(depth, region.bottom);
    for (auto layer = lower; layer < upper; ++layer) {
      const auto viscous =
          model.Has("viscosity") && model.Has("viscosity", layer);
      if (model.IsFluid(layer) || !viscous) {
        Fail(depth,
             std::format("covers {}, which {}: a Burgers body is built over a "
                         "layer's Maxwell viscosity",
                         named(layer),
                         model.IsFluid(layer) ? "is fluid"
                                              : "is elastic, holding no "
                                                "viscosity"));
      }
      out[layer] = Rheology::Kelvin{
          .modulus_fraction = region.kelvin_modulus_fraction,
          .viscosity = region.kelvin_viscosity /
                       model.ModelScales().Factor(ViscosityDimensions)};
    }
  }
  return out;
}

BuiltEarth BuildEarth(const Run& run, const Json& top) {
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
  auto out = BuiltEarth{std::move(*model), std::move(*material), {}, {}, {}};
  out.kelvin = KelvinOf(run, out.model, top);
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
          : Rheology::FromModel(earth.material, earth.model, earth.kelvin);
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

Viscoelastic3D::Options OperatorOptionsOf(const Run& run,
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

SeaLevelParameters ParametersOf(const Run& run,
                                const Material& material) {
  auto parameters = SeaLevelParameters{};
  if (run.hydrostatic_moments) {
    parameters =
        WithHydrostaticMoments(parameters, FluidTidalNumber(material, 2));
  }
  return parameters;
}

const Json& SeaLevelMember(const Run& run, const Json& top) {
  return top.At("ocean").At(run.ocean->seaLevel.Member());
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
      CheckOutsideSpan(input, *file, object);
      return file;
    }
  }
  auto file = std::make_shared<const io::FieldHistory>(
      OpenFileInput(input, *grid_, object));
  read_.emplace_back(input, file);
  return file;
}

std::vector<double> IceThicknessAt(const FileInput& input,
                                   const io::FieldHistory& file,
                                   const Json& member, double time,
                                   std::string_view where) {
  auto thickness = ReadValues(input, file, member, time, where);
  for (auto v : thickness) {
    if (!std::isfinite(v)) {
      Fail(member.At("file"),
           std::format("the ice's {} is not a number somewhere at {:g} kyr BP "
                       "(a surface file marks uncovered points so): fill it "
                       "before the run",
                       input.variable, 0.0 - time / KYR));
    }
  }
  return thickness;
}

std::string StateReadOf(const Run& run) {
  if (!run.times || !run.times->Dated()) {
    return "the state's date, `ocean.date_kyr_bp`, where the state is read";
  }
  return run.ocean->at == WindowEnd::Start
             ? "the window's start, where the state is read"
             : "the window's end, where the state is read, the "
               "calibration's target";
}

namespace {

/// Where a term of the ice change is read, as a refusal states it: at
/// each node's date in a dated run, at its own date in the static problem.
std::string_view TermReadOf(const IceChangeTerm& term) {
  return term.dated ? "a node's date, where the run reads it"
                    : "its date, `date_kyr_bp`, where it is read";
}

}  // namespace

RunIce::RunIce(const Run& run, const Json& top, InputFiles& files,
               double stateTime)
    : grid_(&files.Grid()) {
  const auto& member = IceMember(top);
  const auto state = files.Open(run.ocean->ice, member);
  state_ = IceThicknessAt(run.ocean->ice, *state, member, stateTime,
                          StateReadOf(run));
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
        if (term.held && held.file->Span()) {
          Fail(item,
               std::format("names a file with a time axis ({}), a dated "
                           "history of the change, which makes the problem "
                           "a dated one: give the dated form of `times` "
                           "(`start_kyr_bp`, `end_kyr_bp` and `nodes`, or "
                           "`at_kyr_bp`); a run counting forwards on its own "
                           "clock holds a `change` file of one field, "
                           "without a time axis, through the run",
                           term.file.file.string()));
        }
        if (!term.dated && !term.held && !thickness) {
          CheckDate(term.file, *held.file, item, "date_kyr_bp");
        }
        if (thickness) {
          held.reference = ValuesOf(
              held, stateTime,
              "the state's instant, from which its change is measured");
        }
        // the static problem's term is read once, at its own date (a file
        // without a time axis at any date: the present's), and so is a
        // held one
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

std::vector<double> RunIce::ValuesOf(const Term& term, double time,
                                     std::string_view where) const {
  return IceThicknessAt(term.term.file, *term.file, *term.member, time, where);
}

std::vector<double> RunIce::TermAt(const Term& term, double time) const {
  if (!term.fixed.empty()) {
    const auto& history = term.term.history;
    if (history.Empty()) return term.fixed;
    // a cap's history scales its change, at the time's place on the terms'
    // clock: kyr from the window's start, its dates turned so (a dated
    // run's axis is the calendar's), or the run's own clock in its unit
    const auto factor =
        history.At(history.start_kyr_bp
                       ? *history.start_kyr_bp + time / KYR
                       : time / history.unit.value().seconds);
    auto out = term.fixed;
    for (auto& v : out) v *= factor;
    return out;
  }
  auto values = ValuesOf(term, time, TermReadOf(term.term));
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
    auto change = term.file ? ValuesOf(term, time, TermReadOf(term.term))
                            : TermAt(term, time);
    const auto before =
        term.file ? ValuesOf(term, since,
                             "the window's start, from which the run's change "
                             "is measured")
                  : TermAt(term, since);
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

void RunIce::CheckHeld(std::span<const double> times,
                       const Window* window) const {
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
                   window ? ", " + window->Label(when) : std::string{}));
}

std::string DescribeIceChange(const Run& run,
                              std::string_view separator) {
  auto out = std::string{};
  for (const auto& term : run.forcing.ice_change) {
    if (!out.empty()) out += separator;
    out += Describe(term);
  }
  return out;
}

void CheckStateDate(const Run& run, const Json& top,
                    InputFiles& files) {
  const auto& ocean = top.At("ocean");
  // the state's members, the first file with a time axis named
  auto timed = std::optional<std::filesystem::path>{};
  const auto look = [&](const FileInput& input, const Json& member) {
    if (!timed && files.Open(input, member)->Span()) timed = input.file;
  };
  look(run.ocean->seaLevel.input, SeaLevelMember(run, top));
  look(run.ocean->ice, IceMember(top));
  if (run.ocean->land) look(*run.ocean->land, ocean.At("land"));
  if (timed && !run.ocean->date_kyr_bp) {
    Fail(ocean, std::format("holds a file with a time axis ({}): give the "
                            "state's date, `date_kyr_bp`, at which every "
                            "member's file with a time axis is read",
                            timed->string()));
  }
  if (!timed && run.ocean->date_kyr_bp) {
    Fail(ocean.At("date_kyr_bp"),
         "dates a state whose files have no time axis, the same at every "
         "date: drop it");
  }
}

namespace {

/// The land held at SI time `time` of the land's file (ReadValues, `where`
/// the read): its field cut at one half, a value that is not a number
/// refused at `member`.
std::vector<char> LandAt(const FileInput& input, const io::FieldHistory& file,
                         const Json& member, double time,
                         std::string_view where) {
  const auto values = ReadValues(input, file, member, time, where);
  auto out = std::vector<char>{};
  out.reserve(values.size());
  for (auto v : values) {
    if (!std::isfinite(v)) {
      Fail(member,
           "gives a land field that is not a number somewhere (a surface "
           "file marks uncovered points so): fill it before the run");
    }
    out.push_back(v > 0.5 ? 1 : 0);
  }
  return out;
}

}  // namespace

LandMask ReadLand(const Run& run, const Json& top, InputFiles& files,
                  double time, std::string_view where,
                  std::string_view slice) {
  if (!run.ocean->land) return {};
  const auto& input = *run.ocean->land;
  const auto& member = top.At("ocean").At("land");
  const auto file = files.Open(input, member);
  auto out = LandMask{};
  out.held = LandAt(input, *file, member, time, where);
  out.read = std::format(
      "land {}", DescribeInput(input, OnTimeAxis(input, *file)
                                          ? slice
                                          : std::string_view{"held at every "
                                                             "date"}));
  return out;
}

std::vector<double> SeaLevelFrom(const Run& run, const Json& top,
                                 const io::FieldHistory& file,
                                 std::span<const double> ice, double time,
                                 const SeaLevelParameters& parameters) {
  const auto& source = run.ocean->seaLevel;
  auto sea = ReadValues(source.input, file, SeaLevelMember(run, top), time,
                        StateReadOf(run));
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
                       0.0 - time / KYR));
    }
  }
  return sea;
}

namespace {

/// The slice a run through time reads a state's input at: in a dated run
/// the end of its window the state belongs to, from a file with a time
/// axis; in one counting forwards the state's date; any date from a static
/// file.
std::string WindowSlice(const Run& run, const io::FieldHistory& file,
                        WindowEnd end) {
  const auto& window = *run.times;
  const auto timeAxis = file.Span().has_value();
  if (!window.Dated()) {
    return timeAxis ? std::format("{:g} kyr BP",
                                  run.ocean->date_kyr_bp.value_or(0.0))
                    : std::string{"static"};
  }
  const auto start = end == WindowEnd::Start;
  const auto name = start ? "start" : "end";
  return timeAxis ? std::format("the window's {}, {:g} kyr BP", name,
                                start ? window.Start() : window.End())
                  : std::format("static, at the window's {}", name);
}

/// The state's instant on its files' axes, SI seconds, in a run through
/// time: a dated run's end of the window `ocean.at` names (the calendar's
/// axis is the window's), or the state's date in one counting forwards (a
/// state of files without a time axis at any date: the present's).
double StateTime(const Run& run) {
  const auto& window = *run.times;
  if (!window.Dated()) return -run.ocean->date_kyr_bp.value_or(0.0) * KYR;
  return run.ocean->at == WindowEnd::Start ? window.Axis(0)
                                           : window.Axis(window.Size() - 1);
}

/// The state's sea level at its instant, SI, a topography converted with
/// the state's ice.
std::vector<double> StateSeaLevel(const Run& run, const Json& top,
                                  const SeaLevelParameters& parameters,
                                  const io::FieldHistory& sea,
                                  const RunIce& ice) {
  return SeaLevelFrom(run, top, sea, ice.State(), StateTime(run), parameters);
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

Loaded LoadWorld(const Run& run, const Json& top,
                 const GRDSolver& solver) {
  if (!run.ocean) {
    auto out = Loaded{};
    out.background = solver.NoOceanState();
    return out;
  }
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto& window = *run.times;
  const auto atStart = run.ocean->at == WindowEnd::Start;
  // the window's start on the run's axis, and the state's instant on its
  // files' (the same axis in a dated run)
  const auto start_si = window.Axis(0);
  const auto state_si = StateTime(run);
  auto files = InputFiles(grid);
  // a run counting forwards reads its state at its own date, held to its
  // files as the static problem's is
  if (!window.Dated()) CheckStateDate(run, top, files);
  const auto& source = run.ocean->seaLevel;
  const auto sea = files.Open(source.input, SeaLevelMember(run, top));
  const auto iceFile = files.Open(run.ocean->ice, IceMember(top));
  // the state's sea level is held to its file's span first, as the block
  // lists it (with the state at the window's end, the calibration's target)
  static_cast<void>(CoveredTime(source.input, *sea, SeaLevelMember(run, top),
                                state_si, StateReadOf(run)));
  auto ice = std::make_shared<const RunIce>(run, top, files, state_si);
  // the times the run reads the forcing at, each once: the library's grid
  // (StepGridOf), an onset between nodes among them
  auto steps = StepGridOf(run, scales).times;
  steps.erase(std::ranges::unique(steps).begin(), steps.end());
  // the ice at every one of them: no change takes more than the state
  // holds
  auto nodes = std::vector<double>{};
  for (const auto t : steps) nodes.push_back(start_si + t * scales.Time());
  ice->CheckHeld(nodes, &window);
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
  // The land held. A land field on a time axis is read on the calendar:
  // under migrating shorelines in a dated run at every node's date (the
  // background takes the first's), the ocean function following both
  // flotation and the land held at each node; under frozen ones at the
  // state's instant, whose ocean function the run keeps throughout; in a
  // run counting forwards at the state's date. A static field is the same
  // at every date.
  auto land = LandMask{};
  auto landHeld = std::function<std::vector<char>(double)>{};
  if (run.ocean->land) {
    const auto& input = *run.ocean->land;
    const auto& member = top.At("ocean").At("land");
    const auto file = files.Open(input, member);
    if (window.Dated() && run.Migrating() && OnTimeAxis(input, *file)) {
      // every node's, read before the run: the run asks at the library's
      // nodes alone
      auto masks = std::make_shared<
          std::vector<std::pair<double, std::vector<char>>>>();
      for (const auto t : steps) {
        masks->emplace_back(
            t, LandAt(input, *file, member, start_si + t * scales.Time(),
                      "a node's date, where the run reads it"));
      }
      land.held = masks->front().second;
      land.read = std::format("land {}",
                              DescribeInput(input, "each node's date"));
      landHeld = [masks](double t) {
        const auto found =
            std::ranges::find(*masks, t, [](const auto& m) { return m.first; });
        if (found == masks->end()) {
          throw std::logic_error(
              "the land held is asked at a time that is not a node of the run");
        }
        return found->second;
      };
    } else {
      const auto migratingDated = window.Dated() && run.Migrating();
      land = ReadLand(
          run, top, files, migratingDated ? start_si : state_si,
          migratingDated ? "the window's start, where the run starts"
                         : StateReadOf(run),
          WindowSlice(run, *file,
                      migratingDated ? WindowEnd::Start : run.ocean->at));
    }
  }
  auto background =
      solver.MakeState(std::move(ice_nd), std::move(sl_nd), land.held);
  auto iceRead = DescribeInput(run.ocean->ice,
                               WindowSlice(run, *iceFile, run.ocean->at));
  auto seaRead = std::format(
      "{} {}", source.Quantity(),
      DescribeInput(source.input, WindowSlice(run, *sea, run.ocean->at)));
  if (!land.read.empty()) seaRead += ", " + land.read;
  return {sea,
          std::move(ice),
          std::move(ice0),
          std::move(background),
          std::move(iceRead),
          std::move(seaRead),
          changeAtStart,
          std::move(landHeld)};
}

std::vector<double> FinalSeaLevelOf(const Run& run, const Json& top,
                                    const GRDSolver& solver,
                                    const Loaded& world) {
  auto sea = StateSeaLevel(run, top, solver.Parameters(), *world.seaLevel,
                           *world.ice);
  const auto length = solver.SolverScales().Length();
  for (auto& v : sea) v /= length;
  return sea;
}

std::string WhenOf(const Run& run, const Scales& scales, double t) {
  const auto& window = *run.times;
  return window.Label(window.Axis(0) + t * scales.Time());
}

std::vector<double> MakeTimes(const Run& run, const Scales& scales) {
  return run.times->Times(scales.Time());
}

std::vector<double> NodesOnTheClock(const Window& window) {
  auto out = std::vector<double>{};
  for (const auto node : window.nodes) {
    out.push_back(window.Dated() ? window.Start() - node : node);
  }
  return out;
}

std::vector<OnsetBetween> OnsetsBetweenNodes(const Run& run) {
  auto out = std::vector<OnsetBetween>{};
  if (!run.times) return out;
  const auto nodes = NodesOnTheClock(*run.times);
  const auto& seismic = run.forcing.seismic;
  for (auto k = std::size_t{0}; k < seismic.size(); ++k) {
    const auto& term = seismic[k];
    if (!term.onset) continue;
    const auto at = *term.onset;
    const auto onset = [&term](double t) { return term.OnsetAt(t); };
    const auto shared = std::ranges::any_of(
        out, [&](const OnsetBetween& o) { return term.OnsetAt(o.at); });
    if (std::ranges::any_of(nodes, onset) || shared || at < nodes.front() ||
        at > nodes.back()) {
      continue;
    }
    const auto after = std::ranges::upper_bound(nodes, at) - nodes.begin();
    out.push_back(
        {.term = k, .at = at, .node = static_cast<std::size_t>(after) - 1});
  }
  std::ranges::sort(out, {}, &OnsetBetween::at);
  return out;
}

StepGrid StepGridOf(const Run& run, const Scales& scales) {
  auto out = StepGrid{};
  const auto times = MakeTimes(run, scales);
  const auto clock = SolverClockOf(run, scales);
  const auto nodes = NodesOnTheClock(*run.times);
  const auto between = OnsetsBetweenNodes(run);
  auto next = between.begin();
  const auto add = [&out](double t, std::size_t window) {
    out.times.push_back(t);
    out.window.push_back(window);
  };
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    // the onsets between the node before and this one: a pair each, at the
    // onset's solver time
    for (; next != between.end() && next->node + 1 == n; ++next) {
      const auto t = (next->at - clock.zero) / clock.rate;
      add(t, StepGrid::kInserted);
      add(t, StepGrid::kInserted);
    }
    // a seismic term's onset at a node after the first: its left limit
    const auto jumps =
        n > 0 &&
        std::ranges::any_of(run.forcing.seismic, [&](const SeismicTerm& term) {
          return term.OnsetAt(nodes[n]);
        });
    if (jumps) add(times[n], n);
    out.library.push_back(out.times.size());
    add(times[n], n);
  }
  return out;
}

SolverClock SolverClockOf(const Run& run, const Scales& scales) {
  if (!run.times || run.times->Dated()) {
    return {.zero = 0.0, .rate = scales.Time() / KYR};
  }
  return {.zero = run.times->Start(),
          .rate = scales.Time() / run.times->unit.seconds};
}

std::function<std::vector<double>(double)> IceLoadOf(
    const Run& run, const GRDSolver& solver, const Loaded& world) {
  const auto& grid = solver.Grid();
  const auto fieldSize = grid.FieldSize();
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto time = scales.Time();
  const auto start_si = run.times->Axis(0);
  const auto atStart = run.ocean->at == WindowEnd::Start;
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
    const Run& run, const GRDSolver& solver, const Loaded& world) {
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto time = scales.Time();
  const auto start_si = run.times->Axis(0);
  const auto atStart = run.ocean->at == WindowEnd::Start;
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

IceHistory HistoryOf(const Run& run, const GRDSolver& solver,
                     const Loaded& world) {
  return run.ocean->shorelines == Shorelines::Frozen
             ? IceHistory::Frozen(IceLoadOf(run, solver, world))
             : IceHistory::Migrating(IceThicknessOf(run, solver, world));
}

namespace {

CalibrationOptions CalibrationOptionsOf(const Run& run) {
  auto options = CalibrationOptions{};
  options.method = run.ocean->calibration_method;
  options.iterations = run.ocean->calibration_iterations;
  options.rtol = run.ocean->calibration_rtol;
  // the gradient's sweep is the library's default, the discrete adjoint
  // (ReadRun refuses the gradient under a scheme of several stages)
  options.adjoint.rtol = run.solver.rtol;
  options.adjoint3D.rtol = run.solver.rtol;
  options.adjoint3D.coupled_rtol = run.solver.step_rtol;
  options.adjoint3D.max_coupled_iterations = run.solver.step_max_iterations;
  return options;
}

template <class Calibration>
CalibrationSummary Adopt(const Run& run, const Json& top,
                         const GRDSolver& solver, Loaded& world,
                         Calibration calibration) {
  for (const auto& snapshot : calibration.snapshots) {
    if (!snapshot.converged) {
      throw std::runtime_error(std::format(
          "the calibration's last run did not converge at {} (its "
          "sea level solve, or in 3-D its coupled fixed point); loosen "
          "solver.rtol or solver.step_rtol",
          WhenOf(run, solver.SolverScales(), snapshot.t)));
    }
  }
  if (!calibration.converged) {
    const auto& ocean = top.At("ocean");
    Fail(Has(ocean, "calibration") ? ocean.At("calibration") : ocean.At("at"),
         std::format("calibrated without converging in {} runs (relative "
                     "update {:.3g}, tolerance {:.3g}); raise "
                     "ocean.calibration.iterations",
                     calibration.updates.size(), calibration.updates.back(),
                     run.ocean->calibration_rtol));
  }
  world.background = std::move(calibration.state);
  const auto length = solver.SolverScales().Length();
  // one run under frozen shorelines, one per update when migrating
  return {calibration.misfits.front() * length,
          calibration.misfits.back() * length, calibration.updates.size()};
}

}  // namespace

void StartFromIceHistory(const Run& run, const GRDSolver& solver,
                         Loaded& world, std::span<const double> seaLevelGuess) {
  const auto length = solver.SolverScales().Length();
  world.ice0 = run.ocean->at == WindowEnd::Start
                   ? world.ice->State()
                   : world.ice->At(run.times->Axis(0));
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
std::vector<double> PrepareCalibration(const Run& run, const Json& top,
                                       const GRDSolver& solver,
                                       Loaded& world) {
  auto target = FinalSeaLevelOf(run, top, solver, world);
  StartFromIceHistory(run, solver, world, target);
  return target;
}

}  // namespace

CalibrationSummary CalibrateWorld(const Run& run, const Json& top,
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

CalibrationSummary CalibrateWorld(const Run& run, const Json& top,
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

namespace {

/// The share of a run's loads, below which the degree-1 part removed
/// without gravity is rounding and goes unremarked.
constexpr auto kDegreeOneRounding = 1e-12;

/// Without gravity, the degree-1 part of the surface loads, a net force on
/// the body, which nothing balances: the evolution removes it whatever its
/// size (Evolution.hpp). Measured as the evolution sees the loads
/// (DegreeOneShare of the loads applied, every term summed) at each of
/// `times` (the terms' clock), its largest share is said in one line on
/// the first rank's standard error, with the time (through `label`), where
/// it is above rounding.
void NoteDegreeOneLoads(const SurfaceLoads& loads, const SurfaceGrid& grid,
                        std::span<const double> times,
                        const std::function<std::string(std::size_t)>& label) {
  auto largest = 0.0;
  auto at = std::size_t{0};
  for (auto n = std::size_t{0}; n < times.size(); ++n) {
    const auto node = loads.At(times[n]);
    auto total = std::vector<double>(grid.FieldSize(), 0.0);
    for (const auto* part : {&node.inside, &node.outside}) {
      for (auto i = std::size_t{0}; i < part->size(); ++i) {
        total[i] += (*part)[i];
      }
    }
    const auto share = DegreeOneShare(grid.Expand(total), grid.LMax());
    if (share > largest) {
      largest = share;
      at = n;
    }
  }
  if (largest <= kDegreeOneRounding || !TheRanks().First()) return;
  std::cerr << std::format(
      "sl3d: forcing.load: the loads' degree-1 part ({:.3g} of their size{}) "
      "is a net force on the body, which nothing balances without gravity: "
      "it is removed, and the run answers the loads less it\n",
      largest, label ? " at " + label(at) : std::string{});
}

}  // namespace

template <class EvolutionForcing>
EvolutionForcing ForcingOf(
    const Run& run, const GRDSolver& solver,
    const std::function<std::vector<double>(double)>& iceLoad,
    std::shared_ptr<const SurfaceLoads> loads) {
  const auto& scales = solver.SolverScales();
  const auto length = scales.Length();
  const auto clock = SolverClockOf(run, scales);
  const auto toLoad = scales.Factor(Dimensions{.mass = 1, .length = -2});
  const auto ice = static_cast<bool>(iceLoad);
  auto forcing = EvolutionForcing{};
  if (loads) {
    // a node's loads are built once, whichever is asked first
    struct Built {
      std::optional<double> t;
      SurfaceLoads::Node node;
    };
    const auto built = std::make_shared<Built>();
    const auto at = [built, loads,
                     clock](double t) -> const SurfaceLoads::Node& {
      if (!built->t || *built->t != t) {
        built->node = loads->At(clock.At(t));
        built->t = t;
      }
      return built->node;
    };
    auto any = [&](auto member) {
      return std::ranges::any_of(run.forcing.load, member);
    };
    const auto inside =
        any([](const SurfaceLoadTerm& term) { return term.water_budget; });
    const auto outside =
        any([](const SurfaceLoadTerm& term) { return !term.water_budget; });
    const auto layered = any(
        [](const SurfaceLoadTerm& term) { return term.moves_solid_surface; });
    if (inside || ice) {
      forcing.directLoad = [at, iceLoad, ice, inside, toLoad](double t) {
        auto out = ice ? iceLoad(t) : std::vector<double>{};
        if (!inside) return out;
        const auto& given = at(t).inside;
        if (out.empty()) out.assign(given.size(), 0.0);
        for (auto i = std::size_t{0}; i < out.size(); ++i) {
          out[i] += given[i] / toLoad;
        }
        return out;
      };
    }
    if (outside) {
      forcing.directLoadOutsideWaterBudget = [at, toLoad](double t) {
        auto out = at(t).outside;
        for (auto& v : out) v /= toLoad;
        return out;
      };
    }
    if (layered) {
      forcing.solidSurfaceChange = [at, length](double t) {
        auto out = at(t).layer;
        for (auto& v : out) v /= length;
        return out;
      };
    }
  } else if (ice) {
    forcing.directLoad = iceLoad;
  }
  if (!run.forcing.tide.empty()) {
    const auto toPotential = scales.Factor(PotentialDimensions);
    forcing.tide = [tide = run.forcing.tide, grid = &solver.Grid(), clock,
                    toPotential](double t) {
      auto out = TideAt(tide, *grid, clock.At(t));
      for (auto& c : out) c /= toPotential;
      return out;
    };
  }
  if (!run.forcing.seismic.empty()) {
    const auto surfaceRadius = solver.Radius() * length;
    const auto energy =
        scales.Factor(Dimensions{.mass = 1, .length = 2, .time = -2});
    // the sources at a time and their left limits there, which differ at
    // an onset alone (a pair of StepGridOf's)
    auto sourcesAt = [seismic = run.forcing.seismic, surfaceRadius, clock,
                      length, energy](double t, bool before) {
      auto sources =
          SeismicSourcesAt(seismic, clock.At(t), surfaceRadius, before);
      for (auto& source : sources) {
        source.at.radius /= length;
        for (auto* c : {&source.moment.rr, &source.moment.tt,
                        &source.moment.pp, &source.moment.rt,
                        &source.moment.rp, &source.moment.tp}) {
          *c /= energy;
        }
      }
      return sources;
    };
    forcing.sources = [sourcesAt](double t) { return sourcesAt(t, false); };
    forcing.sourcesBefore = [sourcesAt](double t) {
      return sourcesAt(t, true);
    };
  }
  return forcing;
}

template Evolution::Forcing ForcingOf<Evolution::Forcing>(
    const Run&, const GRDSolver&,
    const std::function<std::vector<double>(double)>&,
    std::shared_ptr<const SurfaceLoads>);
template Evolution3D::Forcing ForcingOf<Evolution3D::Forcing>(
    const Run&, const GRDSolver&,
    const std::function<std::vector<double>(double)>&,
    std::shared_ptr<const SurfaceLoads>);

template <class EvolutionForcing>
Drive<EvolutionForcing> DriveOf(const Run& run, const Json& top,
                                const GRDSolver& solver, const Loaded& world) {
  const auto migrating = run.Migrating();
  auto out = Drive<EvolutionForcing>{};
  // the ice's grounded load under frozen shorelines; a body without an
  // ocean has no ice
  if (run.ocean && !migrating) out.directLoad = IceLoadOf(run, solver, world);
  if (migrating) out.thickness = IceThicknessOf(run, solver, world);
  // The forcing beside the ice: the surface loads by their properties, the
  // tide and the seismic sources. Under frozen shorelines the ice's load
  // joins the loads in the water budget; under migrating ones the ice is
  // given as its thickness, and a change present at the window's start is
  // taken through the forcing's entries (with nothing beside the ice, an
  // empty forcing), so that the start is solved. A run without an ocean
  // is forced beside the ice alone.
  out.forced = run.forcing.BesideTheIce() ||
               (migrating && world.ChangedAtStart());
  if (!run.forcing.load.empty()) {
    out.loads = std::make_shared<const SurfaceLoads>(run.forcing,
                                                     solver.Grid(),
                                                     top.At("forcing"));
    // a `field` file read through time is held to its span at every node
    // before the run reads it there
    const auto& scales = solver.SolverScales();
    const auto clock = SolverClockOf(run, scales);
    auto times = MakeTimes(run, scales);
    for (auto& t : times) t = clock.At(t);
    out.loads->CheckCovers(times);
    if (!run.gravitating) {
      const auto& window = *run.times;
      NoteDegreeOneLoads(
          *out.loads, solver.Grid(), times,
          [&window](std::size_t n) { return window.Label(window.Axis(n)); });
    }
  }
  if (out.forced) {
    out.forcing =
        ForcingOf<EvolutionForcing>(run, solver, out.directLoad, out.loads);
  }
  return out;
}

template Drive<Evolution::Forcing> DriveOf<Evolution::Forcing>(
    const Run&, const Json&, const GRDSolver&, const Loaded&);
template Drive<Evolution3D::Forcing> DriveOf<Evolution3D::Forcing>(
    const Run&, const Json&, const GRDSolver&, const Loaded&);

template <class EvolutionT>
std::vector<typename EvolutionT::Snapshot> RunDriven(
    EvolutionT& evolution, std::span<const double> times,
    const Drive<typename EvolutionT::Forcing>& drive, bool migrating) {
  if (migrating && drive.forced) {
    return evolution.RunMigrating(times, drive.thickness, drive.forcing);
  }
  if (drive.forced) return evolution.Run(times, drive.forcing);
  if (migrating) return evolution.RunMigrating(times, drive.thickness);
  return evolution.Run(times, drive.directLoad);
}

template <class EvolutionT>
std::vector<typename EvolutionT::Snapshot> ResumeDriven(
    EvolutionT& evolution, const typename EvolutionT::Checkpoint& from,
    std::span<const double> remaining,
    const Drive<typename EvolutionT::Forcing>& drive, bool migrating) {
  if (migrating && drive.forced) {
    return evolution.ResumeMigrating(from, remaining, drive.thickness,
                                     drive.forcing);
  }
  if (drive.forced) return evolution.Resume(from, remaining, drive.forcing);
  if (migrating) {
    return evolution.ResumeMigrating(from, remaining, drive.thickness);
  }
  return evolution.Resume(from, remaining, drive.directLoad);
}

template std::vector<Evolution::Snapshot> RunDriven<Evolution>(
    Evolution&, std::span<const double>, const Drive<Evolution::Forcing>&,
    bool);
template std::vector<Evolution3D::Snapshot> RunDriven<Evolution3D>(
    Evolution3D&, std::span<const double>, const Drive<Evolution3D::Forcing>&,
    bool);
template std::vector<Evolution::Snapshot> ResumeDriven<Evolution>(
    Evolution&, const Evolution::Checkpoint&, std::span<const double>,
    const Drive<Evolution::Forcing>&, bool);
template std::vector<Evolution3D::Snapshot> ResumeDriven<Evolution3D>(
    Evolution3D&, const Evolution3D::Checkpoint&, std::span<const double>,
    const Drive<Evolution3D::Forcing>&, bool);

void Check1D(const Evolution::Snapshot& snapshot, std::string_view when) {
  if (!snapshot.converged) {
    throw std::runtime_error(std::format(
        "the sea level solve did not converge at {} (shoreline cells "
        "flipping back and forth defeat the migrating solve's "
        "tolerance); loosen solver.rtol",
        when));
  }
}

void Check3D(const Evolution3D::Snapshot& snapshot, std::string_view when,
             double stepRtol) {
  if (!snapshot.converged) {
    // the step's own iteration as Viscoelastic3D::StepInfo::Describe
    // states it
    const auto info =
        Viscoelastic3D::StepInfo{.iterations = snapshot.coupledIterations,
                                 .residual = snapshot.coupledResidual};
    throw std::runtime_error(std::format(
        "the coupled fixed point or its sea level solve did not "
        "converge at {}: the fixed point's {} (solver.step_rtol; under "
        "migrating shorelines, cells flipping back and forth can "
        "defeat the tolerances); raise solver.step_max_iterations, or "
        "loosen solver.step_rtol or solver.rtol",
        when, info.Describe(stepRtol)));
  }
}

StaticWorld LoadStaticWorld(const Run& run, const Json& top,
                            const GRDSolver& solver) {
  if (!run.ocean) {
    auto out = StaticWorld{};
    out.background = solver.NoOceanState();
    return out;
  }
  const auto& source = run.ocean->seaLevel;
  const auto length = solver.SolverScales().Length();
  auto files = InputFiles(solver.Grid());
  CheckStateDate(run, top, files);
  const auto sea = files.Open(source.input, SeaLevelMember(run, top));
  const auto iceFile = files.Open(run.ocean->ice, IceMember(top));
  // a state without a time axis is read at any date: the present's
  const auto date = run.ocean->date_kyr_bp;
  const auto time = -date.value_or(0.0) * KYR;
  // the state's sea level is held to its file's span first, as the block
  // lists it
  static_cast<void>(CoveredTime(source.input, *sea, SeaLevelMember(run, top),
                                time, StateReadOf(run)));
  const RunIce ice(run, top, files, time);
  ice.CheckHeld(std::span(&time, 1), nullptr);
  // the state's ice, and its sea level: given, or by the flotation
  // construction with the solver's own densities from the topography and
  // the state's ice
  auto iceNd = ice.State();
  auto seaNd = SeaLevelFrom(run, top, *sea, iceNd, time, solver.Parameters());
  for (auto& v : iceNd) v /= length;
  for (auto& v : seaNd) v /= length;
  const auto land =
      ReadLand(run, top, files, time, StateReadOf(run),
               std::format("{:g} kyr BP", date.value_or(0.0)));
  auto out = StaticWorld{};
  out.background =
      solver.MakeState(std::move(iceNd), std::move(seaNd), land.held);
  out.change = ice.ChangeAt(time);
  const auto slice = [&date](const io::FieldHistory& file) {
    return file.Span() ? std::format("{:g} kyr BP", date.value_or(0.0))
                       : std::string{"static"};
  };
  out.iceRead = DescribeInput(run.ocean->ice, slice(*iceFile));
  out.seaLevelRead = std::format("{} {}", source.Quantity(),
                                 DescribeInput(source.input, slice(*sea)));
  if (!land.read.empty()) out.seaLevelRead += ", " + land.read;
  return out;
}

std::shared_ptr<const SurfaceLoads> StaticLoadsOf(const Run& run,
                                                  const Json& top,
                                                  const SurfaceGrid& grid) {
  if (run.forcing.load.empty()) return nullptr;
  auto out = std::make_shared<const SurfaceLoads>(run.forcing, grid,
                                                  top.At("forcing"), false);
  if (!run.gravitating) {
    const auto instant = std::vector<double>{0.0};
    NoteDegreeOneLoads(*out, grid, instant, {});
  }
  return out;
}

template <class EvolutionT>
std::vector<typename EvolutionT::Snapshot> SolveStatic(
    const Run& run, const GRDSolver& solver, const StaticWorld& world,
    std::shared_ptr<const SurfaceLoads> loads, EvolutionT& evolution) {
  const auto length = solver.SolverScales().Length();
  const auto& background = world.background;
  const auto& change = world.change;
  const auto migrating = run.Migrating();
  auto iceLoad = std::function<std::vector<double>(double)>{};
  if (run.ocean && !migrating) {
    auto load = std::vector<double>(change.size());
    for (auto i = std::size_t{0}; i < load.size(); ++i) {
      load[i] = solver.IceDensity() * (1.0 - background.oceanFunction[i]) *
                change[i] / length;
    }
    iceLoad = [load = std::move(load)](double) { return load; };
  }
  const auto forcing = ForcingOf<typename EvolutionT::Forcing>(
      run, solver, iceLoad, std::move(loads));
  const auto one = std::vector<double>{0.0};
  if (migrating) {
    auto thickness = background.iceThickness;
    for (auto i = std::size_t{0}; i < thickness.size(); ++i) {
      thickness[i] += change[i] / length;
    }
    return evolution.RunMigrating(
        one, [thickness = std::move(thickness)](double) { return thickness; },
        forcing);
  }
  return evolution.Run(one, forcing);
}

template std::vector<Evolution::Snapshot> SolveStatic<Evolution>(
    const Run&, const GRDSolver&, const StaticWorld&,
    std::shared_ptr<const SurfaceLoads>, Evolution&);
template std::vector<Evolution3D::Snapshot> SolveStatic<Evolution3D>(
    const Run&, const GRDSolver&, const StaticWorld&,
    std::shared_ptr<const SurfaceLoads>, Evolution3D&);

Rheology ElasticRheologyOf(const BuiltEarth& earth, const Json& top) {
  auto elastic = Rheology::Create(earth.material, {});
  if (!elastic) Fail(top.At("model").At("deck"), elastic.error().message);
  return std::move(*elastic);
}

std::string Meaning(const Run& run, const BuiltEarth& earth) {
  auto out = earth.Dimension();
  if (!run.times) {
    // the static problem at its one instant, in the place of the window
    out += ", elastic, static";
    out += run.ocean ? std::format(", shorelines {}",
                                   NameOf(run.ocean->shorelines))
                     : std::string{", no ocean"};
  } else {
    const auto& window = *run.times;
    out += window.Dated()
               ? std::format(", viscoelastic, {} nodes {} to {} kyr BP",
                             window.Size(), window.Start(), window.End())
               : std::format(", viscoelastic, {} nodes {} to {} {} on the "
                             "run's clock",
                             window.Size(), window.Start(), window.End(),
                             window.unit.name);
    if (!run.ocean) {
      out += ", no ocean";
    } else {
      out += std::format(", shorelines {}", NameOf(run.ocean->shorelines));
      if (run.Calibrating()) out += ", initial sea level calibrated";
    }
  }
  if (!run.gravitating) out += ", no gravity";
  out += std::format(", rotation {}", NameOf(run.rotation));
  if (run.hydrostatic_moments) out += ", moments hydrostatic";
  return out;
}

std::string WithInputs(std::string_view meaning, const Run& run,
                       std::string_view iceRead, std::string_view seaLevelRead,
                       std::span<const std::string> inputs) {
  auto out = run.ocean ? std::format("{}; ice {}, {}", meaning, iceRead,
                                     seaLevelRead)
                       : std::string{meaning};
  if (const auto change = DescribeIceChange(run, " + "); !change.empty()) {
    out += "; ice change " + change;
  }
  if (const auto beside = Summary(run.forcing); !beside.empty()) {
    out += "; " + beside;
  }
  for (const auto& read : inputs) out += "; " + read;
  return out;
}

std::vector<std::string> InputsOf(const SurfaceLoads* loads) {
  return loads != nullptr ? loads->Inputs() : std::vector<std::string>{};
}

}  // namespace sl3dpp::app
