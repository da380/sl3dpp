/**
 * @file MakeSyntheticPerturbation.cpp
 * @brief Writes the examples' synthetic perturbations of the viscosity:
 * planetmodel model files on the PREM-like deck's body whose field
 * `ln_viscosity` holds a CHANGE of the natural logarithm of the viscosity, a
 * pure number, the form `sl3d tangent` reads a direction in the viscosity
 * in (`perturbation.model.viscosity`), so that the tangent configs run
 * with no data to fetch.
 *
 * Both are a toy weak zone in the upper mantle, the viscosity lowered by
 * up to a factor of ten in a smooth radial envelope peaking at 250 km
 * depth and vanishing below 660 km:
 *
 * - `synthetic_weak_zone.nc`: under the whole Earth, a function of the
 *   radius alone, which a radial run takes;
 * - `synthetic_weak_patch.nc`: under a cap about Hudson Bay, varying
 *   laterally, which a 3-D run takes.
 *
 * The build runs it from build/examples, reading the deck in data/ and
 * writing the two files into data/ (the default directory), each replaced
 * only where its content changed (KeepUnchanged.hpp):
 *
 *   MakeSyntheticPerturbation [directory]
 */
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <numbers>
#include <planetmodel/error.hpp>
#include <planetmodel/grid.hpp>
#include <planetmodel/modelfile.hpp>
#include <planetmodel/units.hpp>
#include <planetmodel/vocabulary.hpp>
#include <string>
#include <vector>

#include "Common.hpp"
#include "KeepUnchanged.hpp"

namespace {

constexpr auto kLmax = 16;
constexpr auto kSurface = 6371.0e3;

/// The upper-mantle envelope of the radius (metres): a bump at 250 km
/// depth, 150 km wide, cut off smoothly by 660 km.
double Envelope(double r) {
  const auto d = kSurface - r;
  if (d <= 0.0 || d >= 660.0e3) return 0.0;
  const auto bump = std::exp(-std::pow((d - 250.0e3) / 150.0e3, 2));
  const auto fade =
      d < 560.0e3
          ? 1.0
          : 0.5 * (1.0 + std::cos(std::numbers::pi * (d - 560.0e3) / 100.0e3));
  return bump * fade;
}

/// The change of ln viscosity under the whole Earth: a factor of ten
/// weaker at the envelope's peak.
double WeakZone(double r) { return -std::numbers::ln10 * Envelope(r); }

/// The same under a cap about Hudson Bay (60 N, 275 E), 20 degrees wide.
double WeakPatch(double r, double theta, double phi) {
  constexpr auto deg = std::numbers::pi / 180.0;
  return WeakZone(r) *
         examples::Cap(theta, phi, 30.0 * deg, 275.0 * deg, 20.0 * deg);
}

/// Writes a model file of the field `ln_viscosity` over the model's solid
/// layers, each layer's from `source`, and keeps `output` where its
/// content did not change. planetmodel's vocabulary does not name the
/// field, and the file states its spec: a scalar, dimensionless, and what
/// it is.
void Write(const std::filesystem::path& output,
           const sl3dpp::EarthModel& model, const std::string& name,
           const std::function<planetmodel::LayerSource()>& source) {
  namespace pm = planetmodel;
  const auto& boundaries = model.Boundaries();
  auto file = pm::ModelFile(
      pm::Skeleton(std::vector<double>(boundaries.begin(), boundaries.end())));
  file.SetName(name);
  file.SetMeta("code", "sl3dpp");
  const auto& skeleton = file.GetSkeleton();
  const auto layers = static_cast<int>(skeleton.NLayers());
  file.AddPart(0, layers - 1, pm::ElementEdges(skeleton, 0, layers - 1, 100.0e3),
               5, pm::GaussLegendre(kLmax));
  auto sources = std::map<int, pm::LayerSource>{};
  for (auto layer = 0; layer < layers; ++layer) {
    if (model.IsFluid(static_cast<std::size_t>(layer))) continue;
    sources.emplace(layer, source());
  }
  file.AddField("ln_viscosity",
                pm::FieldSpec{pm::SCALAR, pm::units::DIMENSIONLESS,
                              "change of the natural logarithm of the "
                              "viscosity of every relaxing branch alike"},
                sources);
  // beside the target first: the target is replaced only where the
  // content changed, so a rebuild leaves an unchanged input's bytes alone
  const auto beside = examples::Beside(output);
  pm::WriteModelFile(beside, file);
  const auto replaced = examples::ReplaceIfChanged(beside, output);
  std::cout << (replaced ? "wrote " : "kept, unchanged, ") << output.string()
            << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  const auto directory =
      argc > 1 ? std::filesystem::path{argv[1]}
               : examples::DataFile("synthetic_weak_zone.nc").parent_path();
  try {
    const auto earth = examples::Earth::Make(kLmax, 1e21);
    Write(directory / "synthetic_weak_zone.nc", earth.model,
          "a weak zone in the upper mantle: the change of ln viscosity",
          [] { return planetmodel::LayerSource::Radial(WeakZone); });
    Write(directory / "synthetic_weak_patch.nc", earth.model,
          "a weak patch in the upper mantle under Hudson Bay: the change of "
          "ln viscosity",
          [] { return planetmodel::LayerSource::Lateral(WeakPatch); });
  } catch (const planetmodel::Error& error) {
    std::cerr << "MakeSyntheticPerturbation: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
