#pragma once

/**
 * @file KernelWriter.hpp
 * @brief Fields over the memory nodes out: the 3D sensitivity kernels of
 * Kernels3D, and any other scalar field a 3D run holds at the coupling's
 * nodes (a response's radial displacement, a strain invariant, the
 * lateral structure itself), written as a planetmodel model file for
 * the Python to read back, draw (planetmodel's cutaways and sections)
 * and integrate.
 *
 * A kernel arrives as Kernels3D holds it: its density per unit volume
 * on the coupling's grid at each radial memory node. The file stores it
 * in SI, so that dJ = int K delta dV under the file's own scales, the
 * convention of planetmodel's write_kernel example. The field states the objective's
 * dimensions less three powers of length, and a meaning of the caller's:
 * the library ships no kernel names.
 *
 * Layout: one part per contiguous run of viscous elements, its edges
 * the mesh's own (RadialMesh::Left/Right), its nodes the GLL points of
 * each element, written on the coupling's lmax grid with the BILINEAR
 * lateral rule (a peaked field; the bicubic rings). With the transforms
 * cut with depth, a deeper block's nodes are carried to that grid first
 * (GridCoupling::OnSurfaceGrid: its own degrees, zero padded). The file
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
#include <functional>
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
  /// The Kernels3D layout, [FieldOffset(node) + x] (each node on its
  /// block's grid, carried to the lmax grid for the file): the kernel's
  /// density per unit volume, non-dimensional under the model's scales.
  std::span<const double> values;
};

struct KernelFileOptions {
  /// The model file's name attribute, empty for none.
  std::string name = {};
  /// Numeric meta_<key> attributes, for a reader's cross-checks.
  std::map<std::string, double> meta = {};
};

/// Any scalar field over the memory nodes to write: a value per grid
/// point of every node in the Kernels3D layout, non-dimensional under
/// the model's scales, written in SI with the dimensions given:
/// pointwise values, written as they are (a kernel's are its density
/// per unit volume, as the library holds it).
struct NodeField {
  std::string name;
  std::string meaning;
  Dimensions dimensions = {};
  std::span<const double> values;
};

/// A function of (r, theta, phi), r non-dimensional, sampled at every
/// grid point of every memory node of a coupling, in the NodeField
/// layout: a lateral structure beside the response it shapes.
[[nodiscard]] std::vector<double> SampleNodeField(
    const GridCoupling& coupling,
    const std::function<double(double, double, double)>& f);

/// The fields of degree solutions at the memory nodes of a coupling, in
/// the NodeField layout, non-dimensional: the radial displacement U,
/// and the deviatoric strain's magnitude sqrt(e:e). The operators give
/// the dof layout the solutions use (a static solver's or the
/// stepper's unrelaxed systems, at the coupling's lmax).
struct ResponseNodeFields {
  std::vector<double> radialDisplacement;
  std::vector<double> deviatoricStrain;
};
[[nodiscard]] ResponseNodeFields ResponseFieldsAtNodes(
    const GridCoupling& coupling,
    const std::function<const DegreeOperator&(int)>& spheroidal,
    const std::function<const ToroidalOperator&(int)>& toroidal,
    const Viscoelastic3D::Solutions& x);

/// Writes fields over the memory nodes of a coupling as a model file
/// (the layout the file comment describes); refused when a field's size
/// does not match the coupling or the file cannot be written.
[[nodiscard]] std::expected<void, IoError> WriteNodeFields(
    const std::filesystem::path& path, const GridCoupling& coupling,
    const RadialMesh& mesh, const EarthModel& model,
    const std::vector<NodeField>& fields,
    const KernelFileOptions& options = {});

/// Writes the kernels of `op`'s memory nodes over `model`'s geometry;
/// refused when a kernel's size does not match the coupling or the file
/// cannot be written.
[[nodiscard]] std::expected<void, IoError> WriteKernels(
    const std::filesystem::path& path, const Viscoelastic3D& op,
    const EarthModel& model, const std::vector<KernelField>& kernels,
    const KernelFileOptions& options = {});

/// The same over a coupling and its rheology's mesh rather than an
/// operator's: a whole-mesh coupling, say, on the rank that writes a
/// radially distributed run's kernels gathered from every rank.
[[nodiscard]] std::expected<void, IoError> WriteKernels(
    const std::filesystem::path& path, const GridCoupling& coupling,
    const RadialMesh& mesh, const EarthModel& model,
    const std::vector<KernelField>& kernels,
    const KernelFileOptions& options = {});

}  // namespace sl3dpp::io
