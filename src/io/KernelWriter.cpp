#include "sl3dpp/io/KernelWriter.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <planetmodel/gll.hpp>
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

}  // namespace

std::expected<void, IoError> WriteKernels(const std::filesystem::path& path,
                                          const Viscoelastic3D& op,
                                          const EarthModel& model,
                                          const std::vector<KernelField>& kernels,
                                          const KernelFileOptions& options) {
  return WriteKernels(path, op.Coupling(),
                      op.GetRheology().BaseMaterial().Mesh(), model, kernels,
                      options);
}

std::expected<void, IoError> WriteKernels(
    const std::filesystem::path& path, const GridCoupling& coupling,
    const RadialMesh& mesh, const EarthModel& model,
    const std::vector<KernelField>& kernels, const KernelFileOptions& options) {
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
    if (kernel.values.size() != coupling.GridValueCount()) {
      return std::unexpected(IoError{std::format(
          "{}: {} values for {} memory nodes of {} grid points in all",
          kernel.name, kernel.values.size(), nodeCount,
          coupling.GridValueCount())});
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

    // The coupling's lmax grid, poleless Gauss-Legendre.
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
      // Per layer, the values at the layer's own points: every file node
      // of every element, element by element — a shared edge radius
      // appears once per side and keeps each element's own value, which
      // is how the format holds a piecewise field — then the poles,
      // which the grid lacks, each the nearest ring's mean. Each value
      // is the SI volume density: the element's own polynomial through
      // its memory points of the angular density over the radial weight
      // and r^2, re-dimensionalised.
      const auto factor =
          scales.Factor(kernel.objective) / (length * length * length);
      // every node on the lmax grid (a deeper block's carried there)
      const auto values = coupling.OnSurfaceGrid(kernel.values);
      const auto nphi = grid.Longitudes().size();
      auto sources = std::map<int, planetmodel::LayerSource>{};
      for (const auto& run : runs) {
        for (auto layer = mesh.LayerOf(run.firstElement);
             layer <= mesh.LayerOf(run.lastElement); ++layer) {
          auto rows = std::vector<double>{};
          auto poles = std::vector<double>{};
          auto row = std::vector<double>(fieldSize);
          auto density = std::vector<double>(
              static_cast<std::size_t>(np) * fieldSize);
          for (auto e = run.firstElement; e <= run.lastElement; ++e) {
            if (mesh.LayerOf(e) != layer) continue;
            const auto base =
                run.firstNode + (e - run.firstElement) * np;
            for (auto p = 0; p < np; ++p) {
              const auto r = coupling.RadiusOfNode(base + p);
              const auto scale =
                  factor / (coupling.WeightOfNode(base + p) * r * r);
              const auto at = std::span(values).subspan(
                  static_cast<std::size_t>(base + p) * fieldSize, fieldSize);
              for (auto x = std::size_t{0}; x < fieldSize; ++x) {
                density[static_cast<std::size_t>(p) * fieldSize + x] =
                    at[x] * scale;
              }
            }
            const auto nodes = planetmodel::GllNodes(
                mesh.Left(e) * length, mesh.Right(e) * length,
                static_cast<std::size_t>(np));
            for (const auto rFile : nodes) {
              std::ranges::fill(row, 0.0);
              for (auto p = 0; p < np; ++p) {
                auto cardinal = 1.0;
                const auto rp = coupling.RadiusOfNode(base + p) * length;
                for (auto q = 0; q < np; ++q) {
                  if (q == p) continue;
                  const auto rq = coupling.RadiusOfNode(base + q) * length;
                  cardinal *= (rFile - rq) / (rp - rq);
                }
                for (auto x = std::size_t{0}; x < fieldSize; ++x) {
                  row[x] +=
                      cardinal *
                      density[static_cast<std::size_t>(p) * fieldSize + x];
                }
              }
              rows.insert(rows.end(), row.begin(), row.end());
              auto north = 0.0;
              auto south = 0.0;
              for (auto j = std::size_t{0}; j < nphi; ++j) {
                north += row[j];
                south += row[fieldSize - nphi + j];
              }
              poles.push_back(north / static_cast<double>(nphi));
              poles.push_back(south / static_cast<double>(nphi));
            }
          }
          rows.insert(rows.end(), poles.begin(), poles.end());
          sources.emplace(layer,
                          planetmodel::LayerSource::LateralValues(
                              std::move(rows), planetmodel::Frame::Spherical));
        }
      }

      const auto spec = planetmodel::FieldSpec{
          .character = planetmodel::SCALAR,
          .dimensions =
              planetmodel::Dimensions{.mass = kernel.objective.mass,
                                      .length = kernel.objective.length - 3,
                                      .time = kernel.objective.time},
          .meaning = kernel.meaning};
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
