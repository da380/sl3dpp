#include "sl3dpp/io/KernelWriter.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <planetmodel/gll.hpp>
#include <planetmodel/modelfile.hpp>
#include <span>
#include <utility>
#include <vector>

namespace sl3dpp::io {

namespace {

/// One contiguous run of carried elements: a part of the file. A run
/// that starts above the bottom of its first layer (the centre element
/// is never carried) or ends below the top of its last is padded to
/// the boundary by one file element whose values are the nearest
/// carried node's (constant in radius), so that the part tiles its
/// layers as the format requires.
struct Run {
  int firstElement = 0;
  int lastElement = 0;
  int firstNode = 0;                 ///< its first memory node
  std::vector<double> lefts;         ///< each element's left edge (nd)
  double right = 0.0;                ///< the run's top (nd)
  bool padBelow = false, padAbove = false;
  double bottom = 0.0, top = 0.0;    ///< the layers' boundaries (nd)
};

std::vector<Run> RunsOf(const GridCoupling& coupling, const RadialMesh& mesh,
                        std::span<const double> boundariesNd) {
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
  for (auto& run : runs) {
    run.bottom = boundariesNd[static_cast<std::size_t>(mesh.LayerOf(run.firstElement))];
    run.top = boundariesNd[static_cast<std::size_t>(mesh.LayerOf(run.lastElement)) + 1];
    run.padBelow = run.lefts.front() > run.bottom;
    run.padAbove = run.right < run.top;
  }
  return runs;
}

}  // namespace

std::vector<double> SampleNodeField(
    const GridCoupling& coupling,
    const std::function<double(double, double, double)>& f) {
  auto out = std::vector<double>(coupling.GridValueCount(), 0.0);
  if (!f) return out;
  for (auto rn = 0; rn < coupling.NodeCount(); ++rn) {
    const auto r = coupling.RadiusOfNode(rn);
    const auto at = coupling.FieldOffset(rn);
    const auto& grid = coupling.GridOf(rn);
    const auto nphi = grid.Longitudes().size();
    for (auto i = std::size_t{0}; i < grid.CoLatitudes().size(); ++i) {
      for (auto j = std::size_t{0}; j < nphi; ++j) {
        out[at + i * nphi + j] = f(r, grid.CoLatitudes()[i], grid.Longitudes()[j]);
      }
    }
  }
  return out;
}

ResponseNodeFields ResponseFieldsAtNodes(
    const GridCoupling& coupling,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    const Viscoelastic3D::Solutions& x) {
  using Complex = std::complex<double>;
  auto displacement = coupling.ZeroSpectral();
  displacement.scalars.emplace_back(coupling.ScalarCoefficientCount(), Complex{});
  auto strain = coupling.ZeroSpectral();
  for (const auto& [l, xl] : x.spheroidal) {
    coupling.AddDisplacementOfSolution(displacement.scalars[0], l, spheroidal(l), xl);
    const auto w = x.toroidal.find(l);
    const auto hasW = l >= 1 && w != x.toroidal.end();
    coupling.AddStrainOfSolution(strain, l, spheroidal(l), xl,
                                 hasW ? &toroidal(l) : nullptr,
                                 hasW ? std::span<const Complex>(w->second)
                                      : std::span<const Complex>{});
  }
  const auto u = coupling.Synthesise(displacement);
  const auto e = coupling.Synthesise(strain);
  const auto values = coupling.GridValueCount();
  ResponseNodeFields out;
  out.radialDisplacement = u.scalars[0];
  out.deviatoricStrain.resize(values);
  for (auto i = std::size_t{0}; i < values; ++i) {
    // the Frobenius norm of a real symmetric traceless tensor from its
    // stored spin -2, -1 and 0 components (SpinComponents.hpp): the +2
    // and +1 components are the conjugates, A(00) = 2 A(-+)
    out.deviatoricStrain[i] = std::sqrt(2.0 * std::norm(e.c[0][i]) +
                                        4.0 * std::norm(e.c[1][i]) +
                                        6.0 * e.r[i] * e.r[i]);
  }
  return out;
}

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
  auto fields = std::vector<NodeField>{};
  for (const auto& kernel : kernels) {
    fields.push_back({.name = kernel.name,
                      .meaning = kernel.meaning,
                      .dimensions = Dimensions{.mass = kernel.objective.mass,
                                               .length = kernel.objective.length - 3,
                                               .time = kernel.objective.time},
                      .values = kernel.values});
  }
  return WriteNodeFields(path, coupling, mesh, model, fields, options);
}

std::expected<void, IoError> WriteNodeFields(
    const std::filesystem::path& path, const GridCoupling& coupling,
    const RadialMesh& mesh, const EarthModel& model,
    const std::vector<NodeField>& kernels, const KernelFileOptions& options) {
  const auto& grid = coupling.Grid();
  const auto& scales = model.ModelScales();
  const auto length = scales.Length();
  const auto fieldSize = grid.FieldSize();
  const auto nodeCount = coupling.NodeCount();
  if (kernels.empty()) {
    return std::unexpected(IoError{"no fields to write"});
  }
  for (const auto& kernel : kernels) {
    if (kernel.name.empty()) {
      return std::unexpected(IoError{"a field needs a name"});
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
    auto boundariesNd = std::vector<double>(boundaries.begin(), boundaries.end());
    for (auto& b : boundariesNd) b /= length;
    const auto runs = RunsOf(coupling, mesh, boundariesNd);
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
      if (run.padBelow) edges.push_back(run.bottom * length);
      for (const auto left : run.lefts) edges.push_back(left * length);
      edges.push_back(run.right * length);
      if (run.padAbove) edges.push_back(run.top * length);
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
      // is the SI value: the element's own polynomial through its memory
      // points of the pointwise value, re-dimensionalised.
      const auto factor = scales.Factor(kernel.dimensions);
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
          // the file's elements of this layer in the run: the padding
          // element below (its values the first element's innermost
          // node's), the carried elements, the padding above
          struct FileElement {
            int source;     ///< the carried element supplying the values
            double left, right;  ///< the file element's edges (nd)
            int constantNode;    ///< -1: the source's polynomial; else that node
          };
          auto elements = std::vector<FileElement>{};
          if (run.padBelow && mesh.LayerOf(run.firstElement) == layer) {
            elements.push_back({run.firstElement, run.bottom, run.lefts.front(), 0});
          }
          for (auto e = run.firstElement; e <= run.lastElement; ++e) {
            if (mesh.LayerOf(e) != layer) continue;
            elements.push_back({e, mesh.Left(e), mesh.Right(e), -1});
          }
          if (run.padAbove && mesh.LayerOf(run.lastElement) == layer) {
            elements.push_back({run.lastElement, run.right, run.top, np - 1});
          }
          for (const auto& element : elements) {
            const auto e = element.source;
            const auto base =
                run.firstNode + (e - run.firstElement) * np;
            for (auto p = 0; p < np; ++p) {
              const auto scale = factor;
              const auto at = std::span(values).subspan(
                  static_cast<std::size_t>(base + p) * fieldSize, fieldSize);
              for (auto x = std::size_t{0}; x < fieldSize; ++x) {
                density[static_cast<std::size_t>(p) * fieldSize + x] =
                    at[x] * scale;
              }
            }
            const auto nodes = planetmodel::GllNodes(
                element.left * length, element.right * length,
                static_cast<std::size_t>(np));
            for (const auto rFile : nodes) {
              std::ranges::fill(row, 0.0);
              if (element.constantNode >= 0) {
                const auto p = static_cast<std::size_t>(element.constantNode);
                for (auto x = std::size_t{0}; x < fieldSize; ++x) {
                  row[x] = density[p * fieldSize + x];
                }
              } else {
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
          .dimensions = planetmodel::Dimensions{.mass = kernel.dimensions.mass,
                                                .length = kernel.dimensions.length,
                                                .time = kernel.dimensions.time},
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
