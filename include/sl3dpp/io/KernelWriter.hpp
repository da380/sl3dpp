#pragma once

/**
 * @file KernelWriter.hpp
 * @brief Kernels out: the 3D sensitivity kernels of Kernels3D written
 * as a planetmodel model file, for the Python to read
 * back, draw and integrate.
 *
 * A kernel arrives as Kernels3D holds it: an angular density on the
 * coupling's grid at each radial memory node, the radial quadrature
 * folded in (dJ = sum_node int K delta dOmega). The file stores it per
 * unit volume — each node's values divided by the node's radial weight
 * (GridCoupling::WeightOfNode) and by r^2 — in SI, so that
 * dJ = int K delta dV under the file's own scales, the convention of
 * planetmodel's write_kernel example. The field states the objective's
 * dimensions less three powers of length, and a meaning of the caller's:
 * the library ships no kernel names.
 *
 * Layout: one part per contiguous run of viscous elements, its edges
 * the mesh's own (RadialMesh::Left/Right), its nodes the GLL points of
 * each element, written on the coupling's grid with the BILINEAR
 * lateral rule (a peaked field; the bicubic rings). The file
 * stores one value per node with a shared element-edge radius held
 * once per side, so the field is piecewise: with GLL memory points the
 * file's nodes are the memory nodes and the values pass as they are;
 * with Gauss memory points (the default) each element's nodes take its
 * own polynomial through its Gauss values. A read at a memory node and
 * a grid point returns the written value to rounding, with one reading
 * convention to know: planetmodel resolves a shared radius to the
 * element above within a layer, and at a LAYER boundary each side is
 * read through its own layer's field. The grid has no poles,
 * so each pole's stored value is the nearest colatitude ring's mean.
 * The model's density rides along as the radial reference.
 */

#include <expected>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "sl3dpp/Viscoelastic3D.hpp"
#include "sl3dpp/io/Error.hpp"

namespace sl3dpp::io {

/// One kernel to write: named and described by the caller, with the
/// objective's SI dimensions (the written field states these less three
/// powers of length, a density per unit volume).
struct KernelField {
  std::string name;
  std::string meaning;
  Dimensions objective = {};
  /// The Kernels3D layout, [node * Grid().FieldSize() + x],
  /// non-dimensional under the model's scales.
  std::span<const double> values;
};

struct KernelFileOptions {
  /// The model file's name attribute, empty for none.
  std::string name = {};
  /// Numeric meta_<key> attributes, for a reader's cross-checks.
  std::map<std::string, double> meta = {};
};

/// Writes the kernels of `op`'s memory nodes over `model`'s geometry;
/// refused when a kernel's size does not match the coupling or the file
/// cannot be written.
[[nodiscard]] std::expected<void, IoError> WriteKernels(
    const std::filesystem::path& path, const Viscoelastic3D& op,
    const EarthModel& model, const std::vector<KernelField>& kernels,
    const KernelFileOptions& options = {});

}  // namespace sl3dpp::io
