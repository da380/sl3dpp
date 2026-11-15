# The sl3dpp library

Each `sl3d` command is a few calls into the C++23 library, which offers
more: any rheology, measurement, forcing or load history a code can
build, the operators and solvers on their own, and the derivatives of
all of them. How it works is in [design.md](design.md), the derivatives
in [adjoints.md](adjoints.md), and the API reference is generated from
the headers (`SL3DPP_BUILD_DOCS`, [building.md](building.md)). Linking it
from CMake is in [building.md](building.md), "Using it from CMake".

## Overview

The public headers are under `include/sl3dpp/`; `detail/` is not part of
the API.

| Header | Contents |
| --- | --- |
| `Deck.hpp` | Tabulated radial models: planetmodel decks (`GiaFormat`), its prony and burgers decks read by its own readers with their checks (`PronyFormat`, `BurgersFormat`), mineos (`MineosFormat`), custom column layouts (`DeckFormat`), and the legacy sl3d/gia1d `prem.simp.vm1` format (`Deck::ReadLegacySl3d`) |
| `EarthModel.hpp`, `Scales.hpp` | The radial model, its layering and gravity, and non-dimensionalisation: the elastic columns its own cubic splines, the columns of planetmodel's viscoelastic decks joined as planetmodel joins them (`planetmodel::DeckLayers`; `ViscoelasticDeckKinds`, `Options::kinds`: the viscosities and relaxation times in their logarithm, geometric between knots) |
| `RadialMesh.hpp`, `Material.hpp` | The spectral-element radial mesh (GLL nodes, each degree acting on the elements above its truncation radius), and the material on its nodes, with its directions of variation (`MaterialDirection`, `Variation`) |
| `MemoryPoints.hpp` | Where the internal variables live in each element (Gauss or GLL points), the shear term integrated there, and the volume pairing of a radial kernel with a perturbation (`VolumePairing`) |
| `Rheology.hpp`, `StepWeights.hpp` | Relaxing shear branches: a Maxwell body from a viscosity profile, or general branches; Burgers layers (a Kelvin element in series with the Maxwell one) as their two exact Prony terms (`Rheology::Burgers`, `BurgersPronyTerms`); the body a model's own fields state (`Rheology::FromModel`: a Maxwell deck's viscosity, a burgers deck's Maxwell and Burgers layers, a prony deck's branches, any number of them with a relaxed modulus), and a Maxwell model's layers made Burgers bodies (its overload with Kelvin elements); the time schemes' step weights |
| `DegreeOperator.hpp`, `ToroidalOperator.hpp`, `LoveNumbers.hpp` | Per-degree spheroidal and toroidal systems, with gravity or without; Love numbers, and their derivatives and kernels in κ and μ |
| `SurfaceGrid.hpp`, `ShConventions.hpp`, `Geography.hpp` | The Gauss–Legendre surface grid and its transforms; spherical-harmonic conventions; geographic coordinates |
| `Fingerprint.hpp` | Elastic sea-level fingerprints, SI in and out |
| `SeaLevel.hpp` | The sea-level solver (linear, generalised, nonlinear), on a state with an ocean or without one (`NoOceanState`), and its adjoint; a state's sea level from a topography (`SeaLevelFromTopography`) |
| `Viscoelastic.hpp`, `ToroidalViscoelastic.hpp`, `Evolution.hpp` | The 1D viscoelastic operator and its time steps, and its toroidal partner under interior sources; the coupled 1D evolution under any forcing, frozen or migrating shorelines |
| `PointSource.hpp`, `Fault.hpp` | Interior sources: point moment tensors and stress gluts on shells as forces on the degree systems (several point sources at once, those at one radius formed together: `PointSourcesForces`), and the strain at a point; point sources laid over a planar fault (`PlanarFaultSources`, `DoubleCouple`), and over a fault given by its slip, planetmodel's fault file (`FaultSourcesOf`, at the points of planetmodel's slice quadrature, its geometry formed once, each moment the elastic tensor there contracted with the slip and the normal, `SlipMoment`, from a model's moduli at a radius, `LoveModuliAt`; each slice's tensors, combined at a date's rows as planetmodel combines the slip, `FaultSources::MomentsAt`) |
| `InitialSeaLevel.hpp` | The final sea level misfit and its gradient; the initial sea level calibrated against a final one |
| `Viscoelastic3D.hpp`, `GridCoupling.hpp`, `SpinComponents.hpp`, `Evolution3D.hpp` | The 3D operator under a lateral viscosity and lateral moduli (`Options::moduli`), each step closed by the within-step fixed point or by conjugate gradients (`Options::solver`), and its static solve on a rheology without branches (`SolveElastic`); the grid coupling and its spin algebra; the coupled 3D evolution and its instantaneous response (`Instantaneous`), frozen or migrating shorelines |
| `ElasticPerturbation3D.hpp` | The static elastic problem under lateral δμ, δκ: its first-order (Born) response, the full solve of the laterally heterogeneous problem by preconditioned conjugate gradients (`SolveLateral`) for loads, body tides and measurement duals, and the transpose of its lateral force (`ForceTranspose`, the independent check of the static kernels, which are `LateralKernelSolver`'s) |
| `Adjoint.hpp`, `Adjoint3D.hpp` | Adjoint sweeps and kernels, 1D and 3D, by the continuous adjoint or the discrete transpose: in the ice history, surface loads, a tide, the initial sea level, the moment tensors, the inverse relaxation times (and in 3D the log viscosity) and the moduli; the second-order adjoints (`SecondOrderKernels`) |
| `LateralKernels.hpp` | The kernels in lateral structure (the log viscosity, κ(x), μ(x) as fields over the memory nodes' grids) about any run, the static problem's (a run of one node) included, by one entry point (`LateralKernelSolver`): about a radial Earth the radial sweeps with the grid expansion, no 3D solve; about a laterally heterogeneous one the 3D sweeps |
| `TangentLinear.hpp`, `TangentLinear3D.hpp` | Tangent-linear runs of the coupled evolutions, 1D and 3D, and the second-order runs |
| `Measurements.hpp` | Measurement duals (cap means, point values, expansion coefficients, displacements along a direction at a point or smoothed over a cap as a vector, baselines and angles between sites, rotation components), the degree weights of the cap mean about every point (a field smoothed over the cap), the displacement as a vector field of its east, north and upward components, and the decay time of a site's series as a fitted, kernel-ready datum |
| `Collectives.hpp`, `Distributed.hpp` | Radial distribution: the collectives a distributed run needs, as callbacks, and their MPI form (with `SL3DPP_WITH_MPI`) |
| `Version.hpp` | The library's version and the linked planetmodel's |
| `io/FieldHistory.hpp` | `io::FieldHistory`: a planetmodel surface file's fields on the solver's grid at any time |
| `io/ModelViscosity.hpp` | `io::ModelViscosity`: a model file's 3D viscosity as a radial reference and a lateral log-perturbation |
| `io/ModelModuli.hpp` | `io::ModelModuli`: a model file's `mu` and `kappa` (either alone) as relative lateral perturbations; `io::WriteModuli`, a radial model's moduli with lateral perturbations written as a model file |
| `io/KernelWriter.hpp` | Fields over the 3D memory nodes as a model file: `io::WriteKernels` (3D kernels), `io::WriteNodeFields` (any field, such as `io::ResponseFieldsAtNodes`' response inside the body) |
| `io/TrajectoryStore.hpp` | `io::TrajectoryStore`: a forward run's recording on disk, written as the run goes and read back by the sweeps ([trajectory-store.md](trajectory-store.md)) |
| `io/Binary.hpp`, `io/Error.hpp` | The raw binary records of the trajectory store and of `sl3d`'s checkpoints; `io::IoError`, the planetmodel file readers' and writers' error |

## Using the library

A 1D run, from a planetmodel GIA deck to sea-level histories:

```cpp
#include "sl3dpp/Deck.hpp"
#include "sl3dpp/EarthModel.hpp"
#include "sl3dpp/Evolution.hpp"
#include "sl3dpp/Material.hpp"
#include "sl3dpp/RadialMesh.hpp"
#include "sl3dpp/Rheology.hpp"
#include "sl3dpp/SeaLevel.hpp"
#include "sl3dpp/Viscoelastic.hpp"

using namespace sl3dpp;

// The Earth model: elastic structure and viscosity from the deck (SI).
const auto deck = Deck::Read("earth.deck", GiaFormat()).value();
const auto model = EarthModel::FromDeck(deck).value();
const auto mesh = RadialMesh::Create(model, {.lmax = lmax}).value();
const auto material = Material::Create(model, mesh).value();
const auto rheology = Rheology::FromModel(material, model).value();

// The viscoelastic operator and the sea-level solver.
const auto op = ViscoelasticOperator(rheology, lmax);
const auto solver = GRDSolver(SurfaceGrid(lmax), op.ElasticLoveNumbers(),
                              SeaLevelParameters{});
// The present-day ice thickness and sea level, on solver.Grid().
const auto state = solver.MakeState(iceThickness, seaLevel);

// The evolution under an ice-load history directLoad(t), sampled at `times`.
auto evolution = Evolution(op, solver, state, {});
const auto snapshots = evolution.Run(times, directLoad);
// Each snapshot: sea-level change, displacement, potential, angular velocity.
```

`lmax` is the caller's, and so are the state's fields, `times` and
`directLoad`, in the solver's units (`examples/ExampleEvolution.cpp`
makes them). The same lines read planetmodel's other viscoelastic decks,
`Deck::Read(path, PronyFormat())` or `BurgersFormat()`, `FromModel` then
giving the generalised Maxwell or Burgers bodies they state; and a
Maxwell deck's layers become Burgers bodies with
`Rheology::FromModel(material, model, kelvin)`, `kelvin` a Kelvin element
(`Rheology::Kelvin`: its modulus fraction and viscosity) per layer. The
operator and solver are borrowed by the evolution and must outlive it.
The other entry points follow the same pattern:

- **Fingerprints.** `Fingerprint(lmax, ComputeLoveNumbers(material, lmax))`,
  then `SetState(iceThickness, seaLevel)` and `SolveForIceChange(change)`,
  all in SI. On laterally varying elastic moduli,
  `Evolution3D(op, solver, state).Instantaneous(load)`, `op` a
  `Viscoelastic3D` with `Options::moduli` set, in the solver's units.
  Under any forcing, frozen or migrating, the static problem is a run of
  one node, `Run(std::vector{t0}, forcing)` or
  `RunMigrating(std::vector{t0}, ice, forcing)` of either evolution, as
  `sl3d solve` solves it.
- **3D.** `Viscoelastic3D(rheology, lmax, delta)`, with `delta(r, theta, phi)`
  the log-viscosity perturbation, driven by `Evolution3D` as above.
- **Kernels.** Run the evolution with `{.record_stride = n}`, build a
  `ForwardTrajectory` (or `ForwardTrajectory3D`) from the snapshots
  (`FromSnapshots`), and call
  `AdjointSolver(evolution).FirstOrderKernels(times, measurements, &trajectory)`
  (`AdjointSolver3D` in 3D), with `measurements` the duals of the observed
  quantities, keyed by time node. The kernels as fields in the lateral
  structure, about a radial run or a 3D one alike, the static problem's
  as a run of one node:
  `LateralKernelSolver(evolution, options, {.moduli = true})
  .FirstOrderKernels(times, measurements, trajectory)`, laid out on its
  `Coupling()`.

Model construction returns `std::expected`; solver failures are reported in
the returned diagnostics or thrown. The example drivers
([examples/README.md](../examples/README.md)) are complete programs of
each kind.

## Files: planetmodel from C++

sl3dpp reads and writes the file formats of
[planetmodel](https://github.com/da380/planetmodel). Linking
`sl3dpp::sl3dpp` brings planetmodel's headers with it, so a code can read
and write the same files beside the library
(`<planetmodel/surfacefile.hpp>`, `<planetmodel/modelfile.hpp>`,
`<planetmodel/deck.hpp>`). The `io/` classes connect them to the
library's own types (Overview, above), and `Deck::Read` reads decks.
`SeaLevelFromTopography` builds a state's sea level from an ICE-xG
topography and ice thickness. `examples/MakeSyntheticEarth.cpp` writes a
surface file, and `examples/Common.hpp` reads one.

## Units and conventions

- Decks are read in SI, and every internal component works in
  non-dimensional units. `Fingerprint` and `LoveNumbers` are SI at their
  interface. The time-dependent drivers (`Evolution`,
  `Evolution3D`, `AdjointSolver`, `AdjointSolver3D`) take and return
  times, loads and fields in the sea-level solver's units: convert with
  `solver.SolverScales()`. Their time grids increase, a time repeated
  once for a jump of the forcing (an earthquake's onset): its left limit,
  then its right ([design.md](design.md) §5); a caller whose own nodes
  miss an onset steps a pair there beside them, as `sl3d` does.
- Spherical harmonics are complex, fully normalised, with the Condon–Shortley
  phase (GSHTrans's convention); `ShConventions.hpp` converts to and from real
  coefficients.
- Fields live on the solver's Gauss–Legendre grid, colatitude-major.
- A kernel in a field of the Earth is a density per unit volume, which
  the `VolumePairing` functions pair with a perturbation; a kernel on the
  surface pairs by ∫ · dS ([adjoints.md](adjoints.md), "The kernels'
  convention").
- Details, including sign conventions, are in
  [design.md](design.md) (§2, §4, §9); the code style and testing
  conventions are in [conventions.md](conventions.md).
- Over MPI ranks the library calls no MPI itself: the 3D operator takes
  its radial cuts, segments and `Collectives`, which `Distributed.hpp`
  fills from a communicator ([radial-partitioning.md](radial-partitioning.md)
  §9).
