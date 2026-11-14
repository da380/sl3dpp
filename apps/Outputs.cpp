#include "Outputs.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <numbers>
#include <set>
#include <stdexcept>

#include "Ranks.hpp"
#include "sl3dpp/GridCoupling.hpp"
#include "sl3dpp/Version.hpp"
#include "sl3dpp/io/KernelWriter.hpp"

namespace sl3dpp::app {

std::string Today() {
  const auto now = std::chrono::floor<std::chrono::days>(
      std::chrono::system_clock::now());
  return std::format("{:%Y-%m-%d}", now);
}

void CreateParent(const std::filesystem::path& path) {
  if (const auto parent = path.parent_path(); !parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

planetmodel::AngularGrid AngularGridOf(const SurfaceGrid& grid) {
  return planetmodel::AngularGrid(
      std::vector<double>(grid.CoLatitudes().begin(), grid.CoLatitudes().end()),
      std::vector<double>(grid.Longitudes().begin(), grid.Longitudes().end()),
      "gauss_legendre", grid.LMax(),
      std::vector<double>(grid.CoLatitudeWeights().begin(),
                          grid.CoLatitudeWeights().end()));
}

const LoadTerm* SingleCap(const Forcing& forcing) {
  return forcing.load.size() == 1 &&
                 forcing.load.front().term.kind == LoadTerm::Kind::Cap
             ? &forcing.load.front().term
             : nullptr;
}

const TideTerm* SingleTide(const Forcing& forcing) {
  return forcing.tide.size() == 1 ? &forcing.tide.front() : nullptr;
}

void AddForcingCounts(planetmodel::SurfaceFile& file, const Forcing& forcing) {
  file.meta.emplace(
      "load_terms",
      std::vector<long long>{static_cast<long long>(forcing.load.size())});
  file.meta.emplace(
      "tide_coefficients",
      std::vector<long long>{static_cast<long long>(forcing.tide.size())});
  if (const auto* cap = SingleCap(forcing)) {
    file.meta.emplace("load_latitude_deg",
                      std::vector<double>{cap->site.latitude});
    file.meta.emplace("load_longitude_deg",
                      std::vector<double>{cap->site.longitude});
    file.meta.emplace("load_width_deg", std::vector<double>{cap->width * 180.0 /
                                                            std::numbers::pi});
    file.meta.emplace("load_amplitude_kg_m2",
                      std::vector<double>{cap->amplitude});
  }
  if (const auto* single = SingleTide(forcing)) {
    file.meta.emplace("tide_degree", std::vector<long long>{single->degree});
    file.meta.emplace("tide_order", std::vector<long long>{single->order});
    file.meta.emplace("tide_amplitude_m2_s2",
                      std::vector<double>{single->amplitude});
    file.meta.emplace("tide_imaginary_m2_s2",
                      std::vector<double>{single->imaginary});
  }
}

void CheckOutputMembers(const Json& output, const OutputBlock& block,
                        std::optional<std::size_t> nodes, std::string_view run,
                        std::string_view volumeRefusal) {
  const auto perNode =
      block.exchange.string().find(kNodeField) != std::string::npos;
  if (!nodes) {
    if (perNode) {
      Fail(output.At("exchange"),
           std::format("holds `{{node}}`, which names a table per time node, "
                       "a member that {}",
                       kNeedsQuasiStatic));
    }
    if (Has(output, "exchange_nodes")) {
      Fail(output.At("exchange_nodes"),
           std::format("chooses time nodes, a member that {}",
                       kNeedsQuasiStatic));
    }
    return;
  }
  if (Has(output, "volume")) Fail(output.At("volume"), volumeRefusal);
  if (Has(output, "exchange") && !perNode) {
    Fail(output.At("exchange"),
         std::format("lacks `{{node}}`: {} writes a table per time node, "
                     "`{{node}}` replaced by the node's index",
                     run));
  }
  if (Has(output, "exchange_nodes")) {
    const auto& chosen = output.At("exchange_nodes");
    if (!Has(output, "exchange")) {
      Fail(chosen, "chooses exchange tables, which need `output.exchange`");
    }
    const auto last = static_cast<int>(*nodes) - 1;
    for (auto k = std::size_t{0}; k < block.exchange_nodes.size(); ++k) {
      if (block.exchange_nodes[k] > last) {
        Fail(chosen.At(k), std::format("is not a node index in [0, {}]", last));
      }
    }
  }
}

SurfaceGrid VectorGridOf(const SurfaceGrid& grid) {
  if (grid.NMax() >= 1) return grid;
  const SurfaceGrid vector(grid.LMax(), {.nmax = 1});
  if (!std::ranges::equal(vector.CoLatitudes(), grid.CoLatitudes()) ||
      !std::ranges::equal(vector.Longitudes(), grid.Longitudes())) {
    throw std::logic_error(
        "the grid of upper index 1 has other points than the run's");
  }
  return vector;
}

SurfaceVectorField HorizontalOf(const SurfaceGrid& grid,
                                std::span<const double> V,
                                std::span<const double> W) {
  return DisplacementField(VectorGridOf(grid), {}, V, W);
}

std::vector<double> SmoothedOf(const SurfaceGrid& grid,
                               std::span<const double> field,
                               std::span<const double> weights) {
  auto coefficients = grid.Expand(field);
  for (auto l = 0; l <= grid.LMax(); ++l) {
    for (auto m = 0; m <= l; ++m) {
      coefficients[SurfaceGrid::Index(l, m)] *=
          weights[static_cast<std::size_t>(l)];
    }
  }
  return grid.Synthesise(coefficients);
}

namespace {

/// A field made from another, `from`: its support, time interpolation and
/// dimensions, under its own name, values and meaning.
planetmodel::FileVariable MadeFrom(const planetmodel::FileVariable& from,
                                   std::string name, std::vector<double> values,
                                   std::string meaning) {
  auto out = from;
  out.name = std::move(name);
  out.values = std::move(values);
  out.attributes["long_name"] = std::move(meaning);
  return out;
}

/// A field's meaning, its `long_name`.
std::string MeaningOf(const planetmodel::FileVariable& field) {
  return field.Text("long_name").value_or(field.name);
}

/// Applies `f` to each node's slice of a field (a static field's one) and
/// lays the results end to end.
template <class F>
std::vector<double> EachSlice(const planetmodel::FileVariable& field,
                              std::size_t size, F&& f) {
  auto out = std::vector<double>{};
  out.reserve(field.values.size());
  for (auto at = std::size_t{0}; at < field.values.size(); at += size) {
    const auto made = f(std::span(field.values).subspan(at, size));
    out.insert(out.end(), made.begin(), made.end());
  }
  return out;
}

/// The east and north components of the displacement of the potentials
/// V and W (null: none), node by node, laid end to end.
std::pair<std::vector<double>, std::vector<double>> HorizontalFields(
    const SurfaceGrid& grid, const planetmodel::FileVariable& V,
    const planetmodel::FileVariable* W) {
  const auto size = grid.FieldSize();
  const auto vector = VectorGridOf(grid);
  auto east = std::vector<double>{}, north = std::vector<double>{};
  for (auto at = std::size_t{0}; at < V.values.size(); at += size) {
    const auto u = HorizontalOf(
        vector, std::span(V.values).subspan(at, size),
        W == nullptr ? std::span<const double>{}
                     : std::span<const double>(W->values).subspan(at, size));
    east.insert(east.end(), u.east.begin(), u.east.end());
    north.insert(north.end(), u.north.begin(), u.north.end());
  }
  return {std::move(east), std::move(north)};
}

/// The meanings of the east and north components.
constexpr auto kEastMeaning = std::string_view{
    "the tangential displacement's component to the east, along the "
    "geographic east unit vector at each point (horizontal_displacement)"};
constexpr auto kNorthMeaning = std::string_view{
    "the tangential displacement's component to the north, along the "
    "geographic north unit vector at each point (horizontal_displacement)"};

}  // namespace

void AddAskedFields(planetmodel::SurfaceFile& file, const OutputBlock& output,
                    const SurfaceGrid& grid) {
  const auto horizontal = output.Asks(OptionalField::HorizontalDisplacement);
  if (!horizontal && !output.smoothing) return;
  const auto size = grid.FieldSize();
  auto& variables = file.variables;
  const auto* toroidal = variables.contains("toroidal_potential")
                             ? &variables.at("toroidal_potential")
                             : nullptr;
  auto added = std::vector<planetmodel::FileVariable>{};
  if (horizontal) {
    const auto& V = variables.at("tangential_potential");
    auto [east, north] = HorizontalFields(grid, V, toroidal);
    added.push_back(MadeFrom(V, "displacement_east", std::move(east),
                             std::string{kEastMeaning}));
    added.push_back(MadeFrom(V, "displacement_north", std::move(north),
                             std::string{kNorthMeaning}));
    file.meta.emplace(
        "horizontal_displacement",
        std::string{
            "displacement_east and displacement_north are the components of "
            "the tangential surface displacement u_t = grad_1 V - r x grad_1 "
            "W (V the tangential_potential, W the toroidal_potential, zero "
            "where the file has none, grad_1 the gradient on the unit "
            "sphere, r the radial unit vector) along the geographic east and "
            "north unit vectors at each grid point, synthesised from the "
            "potentials' expansions to lmax; in the frame of no net rotation "
            "of the surface, as W is; the Gauss-Legendre grid has no point "
            "at a pole, where east and north are undefined"});
  }
  if (const auto& smoothing = output.smoothing) {
    const auto weights = CapMeanWeights(grid.LMax(), smoothing->Width());
    const auto smooth = [&](std::span<const double> field) {
      return SmoothedOf(grid, field, weights);
    };
    for (const auto name : kSmoothedFields) {
      const auto found = variables.find(std::string{name});
      if (found == variables.end()) continue;
      const auto& field = found->second;
      added.push_back(MadeFrom(
          field, std::string{name} + std::string{kSmoothedSuffix},
          EachSlice(field, size, smooth),
          MeaningOf(field) + ", smoothed (smoothing)"));
    }
    if (horizontal) {
      // the vector smoothed as its potentials are: the components of the
      // smoothed potentials' displacement
      const auto find = [&added](std::string_view name) {
        return &*std::ranges::find(added, name, &planetmodel::FileVariable::name);
      };
      const auto* V = find("tangential_potential_smoothed");
      const auto* W = toroidal == nullptr ? nullptr
                                          : find("toroidal_potential_smoothed");
      auto [east, north] = HorizontalFields(grid, *V, W);
      const auto& like = variables.at("tangential_potential");
      auto eastField =
          MadeFrom(like, "displacement_east_smoothed", std::move(east),
                   std::string{kEastMeaning} +
                       ", of the smoothed potentials: the vector smoothed "
                       "(smoothing)");
      auto northField =
          MadeFrom(like, "displacement_north_smoothed", std::move(north),
                   std::string{kNorthMeaning} +
                       ", of the smoothed potentials: the vector smoothed "
                       "(smoothing)");
      added.push_back(std::move(eastField));
      added.push_back(std::move(northField));
    }
    file.meta.emplace("smoothing_kind", std::string{"gaussian"});
    file.meta.emplace("smoothing_width_deg",
                      std::vector<double>{smoothing->width_deg});
    file.meta.emplace("smoothing_degree_weights", weights);
    file.meta.emplace(
        "smoothing",
        std::string{
            "each <field>_smoothed is <field> convolved over the sphere with "
            "the Gaussian cap w(d) = exp(-d^2 / (2 s^2)), d the great-circle "
            "distance and s = smoothing_width_deg (the cap's standard "
            "deviation), normalised to unit integral: the value at a point "
            "is the field's mean over the cap about it, the mean a cap "
            "measurement of width s reads there (to rounding, at every "
            "width); the field's expansion to lmax with degree l weighted by "
            "smoothing_degree_weights[l] = int_0^pi w(d) P_l(cos d) sin d dd "
            "/ int_0^pi w(d) sin d dd, synthesised. The response's fields "
            "are smoothed, the state's and the forcing's (the initial sea "
            "level, the ocean functions, the ice change, the loads applied) "
            "are not; relative_sea_level_change_smoothed is the field "
            "smoothed over the whole sphere, land included, not a mean over "
            "the ocean; displacement_east_smoothed and "
            "displacement_north_smoothed are the components of the smoothed "
            "potentials' displacement, the vector smoothed alike"});
  }
  for (auto& field : added) {
    const auto name = field.name;
    if (!variables.emplace(name, std::move(field)).second) {
      throw std::logic_error(
          std::format("the surface file already holds {}", name));
    }
  }
}

std::string SharedScalars(double radius, double gravity, int lmax, bool lateral,
                          const Forcing& forcing) {
  auto out = std::format("radius {:.17g}\n", radius);
  out += std::format("surface_gravity {:.17g}\n", gravity);
  out += std::format("lmax {}\n", lmax);
  out += std::format("lateral {}\n", lateral ? 1 : 0);
  out += std::format("load_terms {}\ntide_coefficients {}\n",
                     forcing.load.size(), forcing.tide.size());
  if (const auto* cap = SingleCap(forcing)) {
    out += std::format(
        "cap_latitude_deg {:.17g}\ncap_longitude_deg {:.17g}\ncap_width_deg "
        "{:.17g}\ncap_amplitude_kg_m2 {:.17g}\n",
        cap->site.latitude, cap->site.longitude,
        cap->width * 180.0 / std::numbers::pi, cap->amplitude);
  }
  if (const auto* single = SingleTide(forcing)) {
    out += std::format(
        "tide_degree {}\ntide_order {}\ntide_amplitude_m2_s2 {:.17g} {:.17g}\n",
        single->degree, single->order, single->amplitude, single->imaginary);
  }
  return out;
}

std::vector<ExchangeColumn> DisplacementAndPotential(
    std::span<const std::complex<double>> U,
    std::span<const std::complex<double>> Phi) {
  return {{.symbol = "U",
           .quantity = "the radial displacement U (m)",
           .coefficients = U},
          {.symbol = "Phi",
           .quantity = "the potential perturbation Phi (m^2 s^-2)",
           .coefficients = Phi}};
}

void WriteExchangeTable(const std::filesystem::path& path,
                        std::string_view heading,
                        std::span<const std::string> provenance,
                        std::string_view scalars, int lmax,
                        std::span<const ExchangeColumn> columns) {
  // the quantities as a list: "a and b", "a, b and c"
  auto quantities = std::string{};
  for (auto k = std::size_t{0}; k < columns.size(); ++k) {
    if (k > 0) quantities += k + 1 == columns.size() ? " and " : ", ";
    quantities += columns[k].quantity;
  }
  CreateParent(path);
  auto out = std::ofstream(path);
  out << std::format(
      "# {}: the m >= 0 spherical harmonic coefficients of {}, complex "
      "orthonormal harmonics with the Condon-Shortley phase of a real field, "
      "f_(l,-m) = (-1)^m conj(f_lm)\n",
      heading, quantities);
  out << std::format("# code sl3dpp version {} date {}\n", Version(), Today());
  for (const auto& line : provenance) out << "# " << line << '\n';
  out << scalars;
  out << "columns l m";
  for (const auto& column : columns) {
    out << std::format(" {}_re {}_im", column.symbol, column.symbol);
  }
  out << '\n';
  for (auto l = 0; l <= lmax; ++l) {
    for (auto m = 0; m <= l; ++m) {
      const auto k = SurfaceGrid::Index(l, m);
      out << std::format("{} {}", l, m);
      for (const auto& column : columns) {
        const auto c = column.coefficients[k];
        out << std::format(" {:.17g} {:.17g}", c.real(), c.imag());
      }
      out << '\n';
    }
  }
  if (!out) {
    throw std::runtime_error(std::format("cannot write {}", path.string()));
  }
}

std::vector<std::pair<int, std::filesystem::path>> ExchangePaths(
    const OutputBlock& output, std::size_t nodes) {
  auto chosen = std::set<int>(output.exchange_nodes.begin(),
                              output.exchange_nodes.end());
  if (output.exchange_nodes.empty()) {
    for (auto n = 0; n < static_cast<int>(nodes); ++n) chosen.insert(n);
  }
  const auto width = std::to_string(nodes - 1).size();
  const auto pattern = output.exchange.string();
  auto out = std::vector<std::pair<int, std::filesystem::path>>{};
  for (const auto n : chosen) {
    auto path = pattern;
    const auto index = std::format("{:0{}}", n, width);
    for (auto at = path.find(kNodeField); at != std::string::npos;
         at = path.find(kNodeField, at + index.size())) {
      path.replace(at, kNodeField.size(), index);
    }
    out.emplace_back(n, std::filesystem::path{path});
  }
  return out;
}

Viscoelastic3D::LoadCallable3D ColumnsOf(
    std::vector<std::complex<double>> coefficients, double factor) {
  return [coefficients = std::move(coefficients), factor](
             int l, double) -> std::optional<std::vector<std::complex<double>>> {
    auto columns = std::vector<std::complex<double>>(l + 1);
    for (auto m = 0; m <= l; ++m) {
      columns[m] = factor * coefficients[SurfaceGrid::Index(l, m)];
    }
    return columns;
  };
}

namespace {

/// VolumeFieldsOf at the memory nodes of `coupling`, the reference's
/// coupling or one over part of its nodes: every field, empty where the
/// coupling holds no node.
VolumeFields VolumeFieldsOn(const GridCoupling& coupling,
                            const ElasticPerturbation3D& reference,
                            const Viscoelastic3D::Solutions& x,
                            const io::ModelModuli* lateral) {
  const auto any = coupling.NodeCount() > 0;
  auto fields =
      any ? io::ResponseFieldsAtNodes(
                coupling,
                [&](int l) -> const DegreeOperator& {
                  return reference.Spheroidal(l);
                },
                [&](int l) -> const ToroidalOperator& {
                  return reference.Toroidal(l);
                },
                x)
          : io::ResponseNodeFields{};
  const auto sampled = [&](const auto& f) {
    return any ? io::SampleNodeField(coupling, f) : std::vector<double>{};
  };
  auto out = VolumeFields{};
  out.emplace_back("radial_displacement", std::move(fields.radialDisplacement));
  out.emplace_back("deviatoric_strain", std::move(fields.deviatoricStrain));
  if (lateral != nullptr) {
    if (lateral->Shear()) {
      out.emplace_back("mu_relative", sampled(lateral->Shear()));
    }
    if (lateral->Bulk()) {
      out.emplace_back("kappa_relative", sampled(lateral->Bulk()));
    }
  }
  return out;
}

}  // namespace

VolumeFields VolumeFieldsOf(const ElasticPerturbation3D& reference,
                            const Viscoelastic3D::Solutions& x,
                            const io::ModelModuli* lateral) {
  return VolumeFieldsOn(reference.Coupling(), reference, x, lateral);
}

VolumeFields GatheredVolumeFields(const ElasticPerturbation3D& reference,
                                  const Viscoelastic3D::Solutions& x,
                                  const io::ModelModuli* lateral,
                                  std::pair<int, int> elements) {
  if (!TheRanks().Distributed()) return VolumeFieldsOf(reference, x, lateral);
  // the reference's nodes in this rank's elements, the element holding
  // the centre left out as the reference leaves it
  auto options = reference.Coupling().GetOptions();
  const auto end = elements.second;
  options.elements = {
      std::min(std::max(options.elements.first, elements.first), end), end};
  const GridCoupling own(reference.GetRheology(), reference.LMax(),
                         reference.Coupling().Grid(), options);
  auto fields = VolumeFieldsOn(own, reference, x, lateral);
  for (auto& [name, values] : fields) values = GatherToFirst(values);
  return fields;
}

void WriteVolumeFile(const std::filesystem::path& path,
                     const ElasticPerturbation3D& reference,
                     const VolumeFields& fields, const RadialMesh& mesh,
                     const EarthModel& model, std::string_view name) {
  auto written = std::vector<io::NodeField>{};
  const auto strainDims = Dimensions{};
  for (const auto& [field, values] : fields) {
    auto meaning = std::string{};
    auto dims = Dimensions{};
    if (field == "radial_displacement") {
      meaning = "radial displacement of the elastic response";
      dims = Dimensions{.length = 1};
    } else if (field == "deviatoric_strain") {
      meaning = "the deviatoric strain's magnitude sqrt(e:e)";
      dims = strainDims;
    } else if (field == "mu_relative") {
      meaning = "the relative lateral perturbation of the shear modulus, "
                "mu(x) / mu(r) - 1";
    } else {
      meaning = "the relative lateral perturbation of the bulk modulus, "
                "kappa(x) / kappa(r) - 1";
    }
    written.push_back(
        {.name = field, .meaning = meaning, .dimensions = dims, .values = values});
  }
  CreateParent(path);
  const auto done = io::WriteNodeFields(path, reference.Coupling(), mesh, model,
                                        written, {.name = std::string{name}});
  if (!done) throw std::runtime_error(done.error().message);
}

}  // namespace sl3dpp::app
