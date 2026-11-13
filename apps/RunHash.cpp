#include "RunHash.hpp"

#include <format>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "sl3dpp/Viscoelastic.hpp"
#include "sl3dpp/Viscoelastic3D.hpp"

namespace sl3dpp::app {

namespace {

/// The outputs' `ocean` metadata of a run without an ocean, the hash's
/// word for it.
constexpr auto kNoOcean = std::string_view{"none"};

}  // namespace

void Fnv1a::Add(std::span<const std::byte> bytes) {
  for (const auto b : bytes) {
    hash_ ^= static_cast<std::uint64_t>(b);
    hash_ *= 1099511628211ull;
  }
}

void Fnv1a::AddText(std::string_view text) {
  Add(std::as_bytes(std::span(text)));
}

void Fnv1a::AddFile(const std::filesystem::path& path) {
  auto in = std::ifstream(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error(std::format("cannot read {}", path.string()));
  }
  auto block = std::vector<char>(std::size_t{1} << 20);
  while (in) {
    in.read(block.data(), static_cast<std::streamsize>(block.size()));
    Add(std::as_bytes(
        std::span(block.data(), static_cast<std::size_t>(in.gcount()))));
  }
}

void AddRunMembers(Fnv1a& hash, const Run& run) {
  hash.AddFile(run.model.deck);
  hash.AddValue(run.model.format);
  hash.AddValue(run.lmax);
  if (!run.model.viscosity.empty()) hash.AddFile(run.model.viscosity);
  hash.AddValue(run.model.viscosity_file_radial);
  // the state's files, or the body's having no ocean
  if (run.ocean) {
    hash.AddFile(run.ocean->seaLevel.input.file);
    hash.AddFile(run.ocean->ice.file);
  } else {
    hash.Add(std::as_bytes(std::span(kNoOcean)));
  }
  // the ice change, term by term: its kind, its file's contents and
  // variable, or a cap's description (its place, size, amount and
  // history) and those numbers whole, which the description rounds to six
  // digits
  hash.AddValue(run.forcing.ice_change.size());
  for (const auto& term : run.forcing.ice_change) {
    hash.AddValue(term.kind);
    if (!term.file.file.empty()) {
      hash.AddFile(term.file.file);
      hash.Add(std::as_bytes(std::span(term.file.variable)));
    } else {
      const auto described = Describe(term);
      hash.Add(std::as_bytes(std::span(described)));
      for (const auto value : {term.site.latitude, term.site.longitude,
                               term.width, term.amount}) {
        hash.AddValue(value);
      }
      hash.AddValue(term.history.times.size());
      for (auto k = std::size_t{0}; k < term.history.times.size(); ++k) {
        hash.AddValue(term.history.times[k]);
        hash.AddValue(term.history.factors[k]);
      }
    }
  }
  if (run.times) {
    // the nodes: evenly spaced dates by their window and count; listed
    // dates, or times on the run's own clock, by their form, their unit
    // and the nodes as given, so that the same physical nodes in another
    // form or another unit are another run (their solver times may differ
    // in the last bit)
    const auto& window = *run.times;
    if (window.Dated() && !window.listed) {
      hash.AddValue(window.Start());
      hash.AddValue(window.End());
      hash.AddValue(static_cast<int>(window.Size()));
    } else {
      const auto form =
          std::string_view{window.Dated() ? "at_kyr_bp" : "forwards"};
      hash.Add(std::as_bytes(std::span(form)));
      hash.Add(std::as_bytes(std::span(window.unit.name)));
      hash.AddValue(window.Size());
      for (const auto node : window.nodes) hash.AddValue(node);
      // a run counting forwards reads its state at its own date
      if (!window.Dated() && run.ocean && run.ocean->date_kyr_bp) {
        hash.AddValue(*run.ocean->date_kyr_bp);
      }
    }
  } else {
    // the static problem: its one instant, the state's date
    hash.AddText("static");
    if (run.ocean && run.ocean->date_kyr_bp) {
      hash.AddValue(*run.ocean->date_kyr_bp);
    }
  }
  hash.AddValue(run.solver.scheme);
  hash.AddValue(run.rotation);
  hash.AddValue(run.solver.step_rtol);
  hash.AddValue(run.solver.step_max_iterations);
  hash.AddValue(run.hydrostatic_moments);
  hash.AddValue(run.solver.rtol);
  // the members below enter only when set away from their defaults: a run
  // that leaves them there hashes the members above alone
  if (run.Migrating()) hash.AddValue(Shorelines::Migrating);
  if (run.ocean && run.ocean->at != WindowEnd::Start) {
    hash.AddValue(run.ocean->at);
  }
  if (!run.solver.transform_degrees.empty()) {
    hash.AddValue(run.solver.transform_degrees.size());
    for (const auto degree : run.solver.transform_degrees) {
      hash.AddValue(degree);
    }
  }
  if (run.ocean) {
    const auto& ice = run.ocean->ice;
    const auto& sea = run.ocean->seaLevel;
    // the state's variables and quantity, where they are other than the
    // ice thickness and the topography under their own names
    if (ice.variable != "ice_thickness") {
      hash.Add(std::as_bytes(std::span(ice.variable)));
    }
    // the land held, where the run is told one
    if (run.ocean->land) {
      hash.AddFile(run.ocean->land->file);
      hash.Add(std::as_bytes(std::span(run.ocean->land->variable)));
    }
    if (sea.kind != SeaLevelSource::Kind::Topography ||
        sea.input.variable != "topography") {
      hash.AddValue(sea.kind);
      hash.Add(std::as_bytes(std::span(sea.input.variable)));
    }
  }
  // the forcing beside the ice, where there is one
  // (a term's description less its file's path, and the file's contents:
  // a run directory may move)
  for (const auto& term : run.forcing.load) {
    auto unplaced = term;
    unplaced.term.field.file.clear();
    unplaced.term.file.clear();
    const auto described = Describe(unplaced);
    hash.Add(std::as_bytes(std::span(described)));
    // and its numbers whole, which the description rounds
    for (const auto value :
         {term.term.site.latitude, term.term.site.longitude, term.term.width,
          term.term.radius, term.term.amplitude, term.term.time_origin,
          term.density}) {
      hash.AddValue(value);
    }
    for (const auto* values :
         {&term.term.history.times, &term.term.history.factors}) {
      for (const auto value : *values) hash.AddValue(value);
    }
    if (term.term.kind == LoadTerm::Kind::Field) {
      hash.AddFile(term.term.field.file);
    } else if (term.term.kind == LoadTerm::Kind::Coefficients) {
      hash.AddFile(term.term.file);
    }
  }
  // the tide, coefficient by coefficient, its numbers whole
  if (!run.forcing.tide.empty()) {
    hash.AddValue(run.forcing.tide.size());
    for (const auto& term : run.forcing.tide) {
      hash.AddValue(term.degree);
      hash.AddValue(term.order);
      hash.AddValue(term.amplitude);
      hash.AddValue(term.imaginary);
      hash.AddValue(term.history.times.size());
      for (auto k = std::size_t{0}; k < term.history.times.size(); ++k) {
        hash.AddValue(term.history.times[k]);
        hash.AddValue(term.history.factors[k]);
      }
    }
  }
  if (run.Calibrating()) {
    hash.AddValue(run.ocean->calibration_method);
    hash.AddValue(run.ocean->calibration_iterations);
    hash.AddValue(run.ocean->calibration_rtol);
  }
  if (!run.model.moduli.empty()) {
    hash.AddFile(run.model.moduli);
    hash.AddValue(run.model.moduli_file_radial);
  }
  if (run.solver.step != Viscoelastic3D::Options::StepSolver::FixedPoint) {
    hash.AddValue(run.solver.step);
  }
  if (!run.gravitating) hash.AddText("gravity none");
}

std::string TrajectoryHashOf(const Run& run) {
  auto hash = Fnv1a{};
  hash.AddText("sl3d trajectory store");
  AddRunMembers(hash, run);
  // the seismic sources, term by term, their numbers whole
  hash.AddValue(run.forcing.seismic.size());
  for (const auto& term : run.forcing.seismic) {
    hash.AddValue(term.kind);
    for (const auto value :
         {term.site.latitude, term.site.longitude, term.depth, term.moment.rr,
          term.moment.tt, term.moment.pp, term.moment.rt, term.moment.rp,
          term.moment.tp, term.strike, term.dip, term.rake, term.length,
          term.width, term.scalarMoment}) {
      hash.AddValue(value);
    }
    hash.AddValue(term.alongStrike);
    hash.AddValue(term.downDip);
    hash.AddValue(term.onset.has_value());
    if (term.onset) hash.AddValue(*term.onset);
    hash.AddValue(term.start_kyr_bp.has_value());
    if (term.start_kyr_bp) hash.AddValue(*term.start_kyr_bp);
    hash.AddText(term.unit.name);
    hash.AddValue(term.history.times.size());
    for (auto k = std::size_t{0}; k < term.history.times.size(); ++k) {
      hash.AddValue(term.history.times[k]);
      hash.AddValue(term.history.factors[k]);
    }
  }
  // the dates the static problem reads its terms' files at, and every file
  // input's format
  for (const auto& term : run.forcing.ice_change) {
    hash.AddValue(term.file.format);
    hash.AddValue(term.file.date.has_value());
    if (term.file.date) hash.AddValue(*term.file.date);
  }
  for (const auto& term : run.forcing.load) {
    hash.AddValue(term.term.field.format);
    hash.AddValue(term.term.field.date.has_value());
    if (term.term.field.date) hash.AddValue(*term.term.field.date);
    hash.AddValue(term.term.before_present);
    hash.AddValue(term.term.unit.seconds);
  }
  if (run.ocean) {
    hash.AddValue(run.ocean->seaLevel.input.format);
    hash.AddValue(run.ocean->ice.format);
    if (run.ocean->land) hash.AddValue(run.ocean->land->format);
  }
  // the memory points of the radial and the 3-D operators, which the
  // program leaves at the library's defaults
  hash.AddValue(ViscoelasticOperator::Options{}.memory_points);
  hash.AddValue(Viscoelastic3D::Options{}.memory_points);
  return std::format("sl3d run fnv1a-64 {:016x}", hash.Value());
}

}  // namespace sl3dpp::app
