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
