#include "sl3dpp/io/KernelWriter.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <planetmodel/modelfile.hpp>
#include <utility>
#include <vector>

namespace sl3dpp::io {

namespace {

/// One contiguous run of viscous elements: a part of the file.
struct Run {
  int firstElement = 0;
  int lastElement = 0;
  int firstNode = 0;                 ///< its first memory node
  std::vector<double> lefts;         ///< each element's left edge (nd)
  double right = 0.0;                ///< the run's top (nd)
};

std::vector<Run> RunsOf(const GridCoupling& coupling, const RadialMesh& mesh) {
  auto runs = std::vector<Run>{};
  const auto np = coupling.Points().count;
  for (auto node = 0; node < coupling.NodeCount(); node += np) {
    const auto e = coupling.ElementOfNode(node);
    if (!runs.empty() && e == runs.back().lastElement + 1) {
      runs.back().lastElement = e;
    } else {
      runs.push_back({e, e, node, {}, 0.0});
    }
    runs.back().lefts.push_back(mesh.Left(e));
    runs.back().right = mesh.Right(e);
  }
  return runs;
}

/// One kernel as the file reads it: a lateral interpolant of the SI
/// volume density per memory node, read radially through each element's
/// memory points by their own polynomial.
struct Sampler {
  const GridCoupling* coupling = nullptr;
  const std::vector<Run>* runs = nullptr;
  std::vector<planetmodel::LateralInterpolant> nodes;
  double length = 1.0;  ///< the model's length scale, metres

  double operator()(double rSI, double theta, double phi) const {
    const auto r = rSI / length;
    // the run holding r, then its element, right-continuously
    const auto& runs = *this->runs;
    auto run = runs.begin();
    while (run + 1 != runs.end() && (run + 1)->lefts.front() <= r) ++run;
    const auto clamped = std::clamp(r, run->lefts.front(), run->right);
    const auto i =
        std::upper_bound(run->lefts.begin(), run->lefts.end(), clamped) -
        run->lefts.begin() - 1;
    const auto np = coupling->Points().count;
    const auto base =
        run->firstNode + static_cast<int>(i) * np;
    // the element's own polynomial through its memory points
    auto value = 0.0;
    for (auto p = 0; p < np; ++p) {
      auto cardinal = 1.0;
      const auto rp = coupling->RadiusOfNode(base + p);
      for (auto q = 0; q < np; ++q) {
        if (q == p) continue;
        const auto rq = coupling->RadiusOfNode(base + q);
        cardinal *= (clamped - rq) / (rp - rq);
      }
      value += cardinal * nodes[static_cast<std::size_t>(base + p)]
                              .Value(theta, phi);
    }
    return value;
  }
};

}  // namespace

std::expected<void, IoError> WriteKernels(const std::filesystem::path& path,
                                          const Viscoelastic3D& op,
                                          const EarthModel& model,
                                          const std::vector<KernelField>& kernels,
                                          const KernelFileOptions& options) {
  const auto& coupling = op.Coupling();
  const auto& mesh = op.GetRheology().BaseMaterial().Mesh();
  const auto& grid = coupling.Grid();
  const auto& scales = model.ModelScales();
  const auto length = scales.Length();
  const auto fieldSize = grid.FieldSize();
  const auto nodeCount = coupling.NodeCount();
  if (kernels.empty()) {
    return std::unexpected(IoError{"no kernels to write"});
  }
  for (const auto& kernel : kernels) {
    if (kernel.name.empty()) {
      return std::unexpected(IoError{"a kernel needs a name"});
    }
    if (kernel.values.size() !=
        static_cast<std::size_t>(nodeCount) * fieldSize) {
      return std::unexpected(IoError{std::format(
          "{}: {} values for {} memory nodes of {} grid points each",
          kernel.name, kernel.values.size(), nodeCount, fieldSize)});
    }
  }
  const auto np = coupling.Points().count;
  if (np < 2) {
    return std::unexpected(IoError{std::format(
        "{} memory points per element cannot form a file element (two or "
        "more nodes)",
        np)});
  }

  try {
    const auto& boundaries = model.Boundaries();
    auto file = planetmodel::ModelFile(planetmodel::Skeleton(
        std::vector<double>(boundaries.begin(), boundaries.end())));
    if (!options.name.empty()) file.SetName(options.name);
    file.SetMeta("code", "sl3dpp");
    for (const auto& [key, value] : options.meta) file.SetMeta(key, value);

    // The coupling's own grid, poleless Gauss-Legendre.
    const auto fileGrid = planetmodel::AngularGrid(
        {grid.CoLatitudes().begin(), grid.CoLatitudes().end()},
        {grid.Longitudes().begin(), grid.Longitudes().end()}, "gauss_legendre",
        static_cast<int>(grid.CoLatitudes().size()) - 1,
        std::vector<double>(grid.CoLatitudeWeights().begin(),
                            grid.CoLatitudeWeights().end()));

    // One part per contiguous run of viscous elements, at the mesh's own
    // edges, its nodes the GLL points of each element. The writer tiles
    // the whole body, so the layers between and around the runs become
    // coarse filler parts of one two-node element per layer, holding the
    // density alone.
    const auto& skeleton = file.GetSkeleton();
    const auto runs = RunsOf(coupling, mesh);
    auto layers = std::vector<int>{};
    auto filler = [&](int first, int last) {
      file.AddPart(first, last,
                   planetmodel::ElementEdges(skeleton, first, last,
                                             skeleton.Span()),
                   2, fileGrid);
    };
    auto next = 0;
    for (const auto& run : runs) {
      const auto first = mesh.LayerOf(run.firstElement);
      const auto last = mesh.LayerOf(run.lastElement);
      if (next < first) filler(next, first - 1);
      auto edges = std::vector<double>{};
      for (const auto left : run.lefts) edges.push_back(left * length);
      edges.push_back(run.right * length);
      file.AddPart(first, last, std::move(edges),
                   static_cast<std::size_t>(np), fileGrid);
      for (auto layer = first; layer <= last; ++layer) layers.push_back(layer);
      next = last + 1;
    }
    if (next < static_cast<int>(skeleton.NLayers())) {
      filler(next, static_cast<int>(skeleton.NLayers()) - 1);
    }

    // The model's density, the radial reference a reader orients by.
    auto rho = std::map<int, planetmodel::LayerSource>{};
    for (auto layer = 0; layer < static_cast<int>(skeleton.NLayers());
         ++layer) {
      rho.emplace(layer, planetmodel::LayerSource::Radial(
                             [&model, layer](double r) {
                               return model.EvaluateInLayer(
                                   "rho", static_cast<std::size_t>(layer), r);
                             }));
    }
    file.AddField("rho", rho);

    for (const auto& kernel : kernels) {
      // The SI volume density at each memory node: the angular density
      // over the node's radial weight and r^2, re-dimensionalised. The
      // grid has no poles; each pole takes its nearest ring's mean.
      const auto factor =
          scales.Factor(kernel.objective) / (length * length * length);
      auto sampler = Sampler{&coupling, &runs, {}, length};
      sampler.nodes.reserve(static_cast<std::size_t>(nodeCount));
      const auto nphi = grid.Longitudes().size();
      auto row = std::vector<double>(fieldSize);
      for (auto node = 0; node < nodeCount; ++node) {
        const auto r = coupling.RadiusOfNode(node);
        const auto scale = factor / (coupling.WeightOfNode(node) * r * r);
        const auto at = kernel.values.subspan(
            static_cast<std::size_t>(node) * fieldSize, fieldSize);
        for (auto x = std::size_t{0}; x < fieldSize; ++x) {
          row[x] = at[x] * scale;
        }
        auto poles = std::array<double, 2>{0.0, 0.0};
        for (auto j = std::size_t{0}; j < nphi; ++j) {
          poles[0] += row[j];
          poles[1] += row[fieldSize - nphi + j];
        }
        poles[0] /= static_cast<double>(nphi);
        poles[1] /= static_cast<double>(nphi);
        sampler.nodes.emplace_back(fileGrid, row, poles,
                                   planetmodel::LateralRule::Linear,
                                   kernel.name);
      }

      const auto spec = planetmodel::FieldSpec{
          .character = planetmodel::SCALAR,
          .dimensions =
              planetmodel::Dimensions{.mass = kernel.objective.mass,
                                      .length = kernel.objective.length - 3,
                                      .time = kernel.objective.time},
          .meaning = kernel.meaning};
      auto sources = std::map<int, planetmodel::LayerSource>{};
      for (const auto layer : layers) {
        sources.emplace(layer, planetmodel::LayerSource::Lateral(
                                   [sampler](double r, double theta,
                                             double phi) {
                                     return sampler(r, theta, phi);
                                   }));
      }
      file.AddField(kernel.name, spec, sources,
                    {.lateral = planetmodel::LateralRule::Linear});
    }

    planetmodel::WriteModelFile(path, file);
    return {};
  } catch (const planetmodel::Error& error) {
    return std::unexpected(IoError{error.what()});
  }
}

}  // namespace sl3dpp::io
