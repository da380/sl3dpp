# Radial substructuring: building and measuring the distributed 3D step

*Written 25 October 2026 against commit 6205daa (develop), with the
measurements of [multicore-scaling.md](multicore-scaling.md). A dated
record of how the radial distribution was built and what it measured;
the maintained account is
[radial-partitioning.md](../radial-partitioning.md).*

## The plan, in summary

On the two-socket server a 3D step at lmax 128 is fastest on one socket,
and GSHTrans keeps one level of threading, so one process cannot run a
group of row-threaded transforms per socket (multicore-scaling.md §3,
§6). Every large array of the step is indexed by radial memory node and
only the per-degree solves couple the nodes, along a band, so the plan
cut the radial mesh into contiguous segments, one per MPI rank with
OpenMP threads inside, each rank holding and transforming its own nodes,
and solved the degree systems by substructuring: each segment eliminates
its interior, and the cut nodes' Schur system couples them
(radial-partitioning.md §1–§4). A serial prototype on a synthetic system
with the degree systems' structure (`benchmarks/substructure_prototype.cpp`:
240 elements, 258 right-hand sides) agreed with the direct solve at
rounding, its work summed over the parts about 1.3 times the direct
solve's. The plan and the proposal that mapped it onto the code are built
as §7, §8 and §10–§12 record. As built, every rank solves the sea level
equation and rank 0's results are shared to all (not the top rank
solving and broadcasting), and the cuts balance the transform work, not
the memory nodes.

## 7. The substructured solve, built (25 October 2026, against 7a14aaa)

`detail::BandSubstructure` (`include/sl3dpp/detail/BandSubstructure.hpp`)
is the substructuring algebra (radial-partitioning.md §3) on any
symmetric band matrix given its separators: each
segment's interior factorised on its own (banded Cholesky, falling back to
banded LU as `DegreeOperator` does), the coupling columns K_II⁻¹ K_IΓ kept,
the separators' system assembled and factorised at its own bandwidth, and a
solve of one interior solve per segment per right-hand side. It refuses
separators that do not separate (a band entry coupling two segments, or a
segment and a non-adjacent separator).

`DegreeOperator::Options::cuts` and `ToroidalOperator::Options::cuts`
(element indices: the mesh is cut at each one's bottom node) switch a
degree system to it; cuts at or below the degree's first element are
dropped, so a high degree keeps only the upper segments. The scaled band
is substructured, the degree-1 frame (the removed surface φ dof) and the
toroidal pins (identity rows, applied before) carry over unchanged, and a
cut node in the fluid core has no toroidal dof and separates nothing.
`ViscoelasticOperator::Options::radial_cuts` and
`Viscoelastic3D::Options::radial_cuts` pass cuts to every degree system,
the 3D reference operator's included. `RadialMesh::BalancedCuts(parts,
weights)` picks cuts at element boundaries inside layers, balancing summed
element weights. `DegreeOperator::Apply` (the assembled operator) gives
residuals.

Agreement with the direct solve to 1e-13 relative, the plan's target, is
not to be had (`tests/TestSubstructure.cpp`). The two solves have the
same residuals, but the degree systems determine their solutions only to their
conditioning, and the direct solve's own error is as large. At lmax 64 the
substructured and direct solutions differ by up to 1.4e-10 at degree 1,
1e-11 at the lowest degrees and below 1e-12 from degree 10. The gate is
therefore residuals within 3x of the direct solve's, and agreement 1e-9
below degree 10 and 1e-11 above, for every spheroidal and toroidal degree
to 64. Also checked: cuts at the solid–fluid boundaries (algebraically
fine), the 1D unrelaxed and effective Love tables (1e-12), and a coupled 3D
lateral evolution with every system substructured (1e-10, the fixed
point's tolerance).

Cost (`benchmarks/substructure_timing`, prem.simp.vm1, one thread, every
spheroidal and toroidal degree with 2(l + 1) columns as a pass solves
them):

| lmax | parts | build | solves | solves / direct | balance | segments at lmax | agreement |
|---|---|---|---|---|---|---|---|
| 64 | 1 (direct) | 0.025 s | 0.257 s | 1 | 1 | 1 | — |
| 64 | 4 | 0.045 s | 0.292 s | 1.13 | 0.63 | 2 | 1.8e-11 |
| 64 | 16 | 0.048 s | 0.292 s | 1.14 | 0.62 | 8 | 2.7e-11 |
| 128 | 1 (direct) | 0.060 s | 1.175 s | 1 | 1 | 1 | — |
| 128 | 4 | 0.109 s | 1.352 s | 1.15 | 0.54 | 2 | 1.1e-10 |
| 128 | 32 | 0.121 s | 1.305 s | 1.11 | 0.52 | 13 | 1.0e-10 |

The total work is 5–15% above the direct solves'. Balance is the share
of a perfect split that the rank holding the largest segment carries,
weighted by columns. Cutting into equal element counts balances the
solves poorly, about 0.55: the high degrees, which carry the most columns,
keep only the upper segments. The solves are a few per cent of a pass, and
the cuts are set by the transforms, the pointwise work and the memory
(radial-partitioning.md §7), so the solves' imbalance is accepted. A
rank's interior solves fall with its share of the dofs either way.

MPI on this laptop: the system OpenMPI (4.1.6) has no development headers
(`libopenmpi-dev` is not installed). `~/miniforge` has OpenMPI 5.0.10,
whose wrapper builds with the system compiler as `OMPI_CXX=g++
~/miniforge/bin/mpicxx`, and its `mpirun` runs.

## 8. The distributed solve over MPI (25 October 2026)

MPI is an optional part of the library: the CMake option `SL3DPP_WITH_MPI`
links it and defines `SL3DPP_HAVE_MPI`, which guards the MPI sections
(`include/sl3dpp/Distributed.hpp`, `src/Distributed.cpp`, the MPI test
and benchmark). The solvers stay MPI-free. `BandSubstructure` takes the
segments a process owns and a `combine` callback that sums a vector over
the processes. Construction factorises the owned segments and combines S
(K_SS from the owner of segment 0, each segment's reduction from its
owner). Each solve reduces the owned interiors, combines g, solves S
redundantly on every rank, and back-substitutes the owned interiors.
`DegreeOperator` and `ToroidalOperator` pass `Options::segments` (numbered
over the whole mesh; a degree whose first element lies above some cuts
renumbers past them) and `Options::combine`. A toroidal cut node in the
fluid core gets an empty separator, so the segment numbering stays one per
cut. `Distributed.hpp` supplies `SumOver(comm)` (MPI_Allreduce) and
`ShareMesh(mesh, comm)`: one segment per rank, from `BalancedCuts`.

Here every process passes the whole right-hand side; only its own rows
and the separators' (from segment 0's owner) are read. In the distributed
3D step (§11) each rank's force holds its own elements' contributions at
its cut nodes, partial sums, and the separators' right-hand side is
combined as shares (`RightHandSide::Shares`).

`TestDistributed` runs under `mpiexec` on 2, 3 and 4 ranks. It gathers each
rank's part of the solution of every spheroidal and toroidal degree to lmax
32 and holds the result to the serial gate (residuals within 3x of the
direct solve's; agreement 1e-9 below degree 10, 1e-11 above). The suite
passes on all three rank counts, and the whole suite passes in the MPI build.

`distributed_solves` (prem.simp.vm1, lmax 128, every spheroidal and
toroidal degree with 2(l + 1) columns, one thread per rank, this laptop):

| ranks | slowest rank's solves | one-process direct solves |
|---|---|---|
| 1 | 1.08 s | 1.09 s |
| 2 | 1.16 s | 1.27 s (two direct runs at once) |
| 4 | 0.90 s | 1.64 s (four at once) |

The solves alone barely scale, as §7's balance measure predicts: the high
degrees, which carry most of the columns, keep only the top segment. A
distributed pass gains from its transforms and pointwise work, which split
evenly by memory node. The solves are a few per cent of a pass, so by
estimate their share would grow from about 3% to about 10% on 4 ranks. Two levers if that
matters: weight the cuts towards the surface (it costs the transforms'
balance), or give the top segment's rank more threads for its degree
loop. The communication is one all-reduce of 3 (P − 1) (2l + 2) values per
degree per solve. Batching the degrees of a pass into one all-reduce is
the obvious step for many ranks.

Building with MPI on this laptop: the apt OpenMPI has no development
headers. The MPICH 5.0.0 of PETSc's install (~/dev/petsc-install) works,
but its `lib` also holds netCDF, HDF5 and zlib under the system's sonames.
FindMPI was therefore pointed at a directory of links to `mpi*.h` and
`libmpi*.so` alone (README, *Building in detail*), so the run path finds
the system netCDF.

The step itself followed: the transforms cut with depth (§10) and the
distributed step (§11).

## 10. Transforms cut with depth: the node blocks (25 October 2026)

A transform degree per radial partition, chosen so that nothing is lost,
with the cuts balanced on the transform work rather than on node counts.
Built on one process first; the MPI distribution (§11) follows it.

**Why nothing is lost.** Degree l's system has dofs only above its
truncation radius, (r/a)^(l + 1) = eps under the default graded rule
(`RadialMesh::StartElement`). So a memory node's strain carries no degree
above L(r) = ln(1/eps) / ln(a/r) - 1, and the memory force it produces above
L(r) has no rows to enter. Yet every node was transformed at lmax. At the
default eps = 1e-8 on prem.simp.vm1:

| depth | 440 km | 845 km | 1560 km | 2710 km |
|---|---|---|---|---|
| L(r) | 256 | 128 | 64 | 32 |

Below about lmax 30 every mantle node needs lmax, so the blocks only
matter above that.

**What was built.** `GridCoupling::Options::transform_degrees` (and
`Viscoelastic3D::Options::transform_degrees`, sl3d's
`coupled.transform_degrees`): each viscous element's memory nodes take
the least listed degree, or lmax, at or above the largest degree whose
subset reaches the element (`GridCoupling::TransformDegrees`, which needs
no grid). That degree only grows outwards, so each degree's nodes form one
contiguous block with its own `SurfaceGrid` (and Wigner table) at the same
oversampling. Every field over the nodes lays the blocks end to end
(`FieldOffset(node)`, `FieldSizeOf(node)`, `GridValueCount()`); a node's
coefficients are its block's degree's, a prefix of lmax's in GSHTrans's
(l, m) order, so carrying them between blocks is zero padding
(`OnSurfaceGrid`, which the kernel writer uses to put every node on the
lmax grid). The batched transforms run block by block; the degree loops
start each degree at its first node whose block carries it. With no
degrees listed there is one block and the step is bitwise the same
(step_timing fingerprint 36dbd27521e4fddf, lmax 32, one thread).

**The one approximation, and David's point about it.** The lateral
product W(x) d(x) is formed on the block's grid. The force at degrees up
to L needs W up to degree 2L; the 3/2 rule's grid at L integrates it
exactly for W to about degree L, where the lmax grid did so for the deep
nodes to well beyond 2L. As David put it, this depends on the viscosity's
spectrum, not the fields': the lateral viscosity is typically smoother
than the solution, whose resolution (and the shorelines') sets lmax. For
smooth patterns the difference is at round-off (below); a rough deep
viscosity would want higher listed degrees there, or a per-block
oversampling chosen from its spectrum (not built).

**Measurements** (laptop, 8 threads pinned to cores; step_timing's
Y_{2,±2} log-viscosity of amplitude 0.6, Gauss memory points; the last
column is the relative change of the final surface sums):

| lmax, degrees | TensorField | analyse | synthesise | per pass | startStep | peak RSS | surface sums |
|---|---|---|---|---|---|---|---|
| 64, none | 0.74 GB | 0.254 s | 0.227 s | 0.754 s | 0.80 s | 6.31 GB | — |
| 64, 32 48 | 0.65 GB | 0.216 s | 0.184 s | 0.661 s | 0.65 s | 5.68 GB | 5e-16 |
| 64, 16 24 … 56 | 0.63 GB | 0.204 s | 0.185 s | 0.639 s | 0.61 s | 5.53 GB | 5e-16 |
| 128, none | 4.13 GB | 3.28 s | 2.26 s | 7.17 s | 8.32 s | 34.0 GB | — |
| 128, 32 48 … 112 | 2.78 GB | 2.06 s | 1.47 s | 4.65 s | 5.03 s | 24.2 GB | 3e-16 |

At lmax 128 the step takes 0.65 of the time and 0.71 of the memory; the
(L + 1)^3 model of `TransformWeights` predicts 0.61 for the transforms,
which take 0.63 (analysis) and 0.65 (synthesis): the FFTs and per-node
overheads, which shrink more slowly than L^3, make the deep blocks a
little dearer than the model. At lmax 64 three quarters of the nodes still
need lmax, and the gain is 15%. It grows with lmax (the model gives 0.42
at 256). The listed degrees need only be fine enough: seven at lmax 128
(every 16) or two at lmax 64 leave little on the table.

**Tests** (`tests/TestNodeBlocks.cpp`, lmax 16 with eps = 1e-2 so that
blocks form at a test's size): the blocks follow the truncation and
`TransformDegrees` / `TransformWeights` agree with them; each block's
transforms invert, and `OnSurfaceGrid` equals the lmax grid's synthesis of
the zero-padded coefficients; a step without lateral viscosity equals the
single grid's to 1e-12; with a smooth pattern of a decade's contrast it
differs by 3.8e-9, of a third of that amplitude by 5e-12 (the iteration's
tolerance); and the 3D adjoint on the blocks gives the single grid's
spherical kernels, ice kernels and the pairing of its log-viscosity kernel
with a smooth direction to 6e-14 without lateral viscosity and 1.3e-9
with it.

**Balancing the cuts.** `ShareMesh` balances the cuts on
`TransformWeights` rather than on elements (built with the distributed
step, §11). Computed for prem.simp.vm1, with every
element at its own degree L(e) and the cuts at whole elements: the
busiest rank's transform work over the mean, and the separator values
the redundant Schur solves sum per pass (4 dofs per separator node,
2(l + 1) columns per degree, only the separators above each degree's
first element):

| ranks | weighted by element | by memory node | by work, lmax 128 | by work, lmax 256 | summed per pass, lmax 256 |
|---|---|---|---|---|---|
| 2 | 1.84 | 1.70 | 1.00 | 1.00 | 2 MB |
| 4 | 1.91 | 1.74 | 1.00 | 1.01 | 6 MB |
| 8 | 2.32 | 1.76 | 1.01 | 1.03 | 14 MB |
| 16 | 2.32 | 1.76 | 1.04 | 1.04 | 29 MB |
| 32 | 2.40 | 1.76 | 1.12 | 1.20 | 59 MB |
| 64 | 2.56 | 1.92 | 1.28 | 1.37 | 119 MB |
| 128 | 2.56 | 2.56 | 1.92 | 1.37 | 238 MB |

(The element and node columns are at lmax 128.) Two limits near a hundred
ranks: whole elements cannot balance (the mesh is worth about 200
full-degree elements, so a rank gets one or two surface elements), and
the separator sum grows with the ranks, since every rank solves the
small cut-node system redundantly; a distributed solve of that
block-tridiagonal system (cyclic reduction, neighbour exchanges) would
replace it there.

## 11. The 3D step distributed (25 October 2026)

Built with callbacks rather than MPI calls in the core, the sea level
equation solved on every rank, and per-rank checkpoints in the app layer
(§12).

**The pieces.**
- `Collectives` (`Collectives.hpp`): `sum`, `max` and `share` (the first
  rank's values to all), each in place on a span; empty means one
  process. `Distributed.hpp`'s `CollectivesOver(comm)` makes them from
  MPI_Allreduce and MPI_Bcast, and `ShareMesh` returns them with the cuts
  and the rank's segment. The core never includes `mpi.h`.
- `Viscoelastic3D::Options::segments` and `collectives`, beside
  `radial_cuts`. The rank's coupling then holds the memory nodes of its
  segment's elements only (`GridCoupling::Options::elements`), and its
  degree systems solve its segment only.
- The degree solves split at their one collective
  (`BeginSolve` / `FinishSolve`, over `BandSubstructure::Condense` /
  `Complete`). `SolveAll` begins every degree's solve threaded, sums all
  the cut nodes' right-hand sides in one call on the main thread, and
  finishes them threaded. The forces arrive in shares (each rank's own
  elements' rows, partial at the cut nodes: `RightHandSide::Shares`); the
  unit loads arrive whole. A rank's solution holds its segment and the cut
  nodes, zero elsewhere (`Holds`, `MaskToShare`).
- Every MPI call is on the main thread, outside parallel regions
  (MPI_THREAD_FUNNELED is enough).

**The collectives of a pass**, in order: one sum of the cut nodes' parts
of every degree system (the one sizeable one; §10's table gives its size);
two sums of the surface columns (U and phi, (lmax + 1)(lmax + 2)/2
complex values each, in `Surface`); one of the inertia moment; the
first-rank share of the load and centrifugal coefficients (and, under
migrating shorelines, the ocean function); a max of three values for the
convergence test; and, once Anderson mixing engages, a sum of its 2m Gram
values. The first pass of a step adds a max of one flag (whether StartStep's
force may be used), and a degree's first unit solve one sum.

**Why the sea level loads are shared.** Every rank solves the sea level
equation from the same summed inputs, but each builds its own grids and
FFTW plans (by measurement), which may round differently; under migrating
shorelines a cell at the flotation threshold could then flip on one rank
and not another. Rank 0's loads (and ocean function) therefore serve all
ranks, for half a megabyte per pass at lmax 128.

**Tests.** Without MPI, `tests/TestRadialRanks.cpp` runs two and three
ranks as threads of one process (each collective a rendezvous reducing in
rank order) against the one-process run on the same cuts: two lateral
steps with the transforms cut with depth (surface response to 5e-16, each
rank's memory to 2.5e-14), and the coupled evolution, frozen and migrating
(4 cells flip; sea level to 1.3e-15, the ocean functions equal). Under
MPI, `tests/TestDistributed.cpp` makes the same comparisons on 2, 3 and 4
ranks: up to 1.8e-13 for the step (MPI sums in its own order) and 5e-15
for the evolutions. The one-process path is bitwise unchanged
(fingerprint 36dbd27521e4fddf).

**Measured on the laptop** (one socket, 8 cores, `distributed_step`,
lmax 64, transform degrees 32 48, phases only):

| ranks x threads | memory nodes per rank | respond | analyse | startStep | peak RSS per rank |
|---|---|---|---|---|---|
| 1 x 8 | 980 | 0.550 s | 0.217 s | 0.66 s | 5.44 GB |
| 2 x 4 | 564, 416 | 0.549 s | 0.191 s | 0.56 s | 3.10 GB |
| 4 x 2 | 356, 208 × 3 | 0.612 s | 0.199 s | 0.53 s | 1.89 GB |
| 8 x 1 | 252, 104 × 7 | 0.824 s | 0.260 s | 0.66 s | 1.25 GB |

On one socket the ranks cannot be faster: they share its cores and
memory. What they show: the memory per rank falls with the ranks, each
keeping a fixed part of roughly half a gigabyte at lmax 64 (libraries,
tables, the 1D reference operator, MPI's buffers; not itemised yet); the
transform work balances (the deep rank holds more, cheaper nodes); and
two ranks of four threads cost nothing. Pure MPI is slower here for two
reasons: one-thread ranks stream the whole Wigner table each (the
node-block pattern of multicore-scaling.md §6), and the high degrees,
whose dofs lie near the surface, are solved by the top rank alone on its
one thread.

**Not yet.** The measurement that matters, on the server: 2 x 32 against
1 x 64 on one node, then across nodes. The top rank's share of the
high-degree solves, if the server shows it. (The 3D adjoint and `sl3d`
followed: §12.)

## 12. The 3D adjoint and sl3d over ranks (25 October 2026)

**The adjoint** (`src/Adjoint3D.cpp`) needed little: its reversed sweep
steps the same distributed machinery (StartStep, Respond, Surface,
InertiaMoment, Converge), its viscosity-kernel accumulation runs over the
rank's own nodes, and its jump uses the 1D reference operator's unit
responses, which every rank holds whole. What changed: the equilibrium's
inertia moment is the rank's share of the degree-0 column, summed; after
every sea level solve (the jump's fingerprint, the equilibrium, each
reversed pass) rank 0's results are shared, as in the forward run, so the
ice kernels are every rank's alike bitwise; and a sea level solve's
failure is decided over all ranks, so that all throw together. Each
rank's viscosity kernels are its own nodes'; laid end to end in rank order
they are the whole mesh's. Tests: three thread ranks and 2-4 MPI ranks
against the one-process sweep, frozen and migrating: ice kernels to 5e-15,
log-viscosity kernels to 4e-14.

**sl3d** (`apps/Ranks.hpp`, docs/sl3d-configs.md "Over MPI ranks"): built
with SL3DPP_WITH_MPI, `main` starts MPI (MPI_THREAD_FUNNELED) and the 3D
commands run over the ranks of MPI_COMM_WORLD; the others refuse several
ranks. `OperatorOptions` gives the rank its segment (ShareMesh balanced on
the transform work) and the collectives. Rank 0 writes every output and
prints; a failure on any rank calls MPI_Abort, since the others would
wait at their next collective. Checkpoints: each rank writes its
own into `rank_RRRR` of the checkpoint directory (rank 0's with the
histories), the rank count enters the checkpoints' hash, a run resumes
from the newest node every rank has (rank 0's newest two shared, each
rank flagging which it lacks), a set the ranks do not share is cleared,
and the wall-time stop is decided over all ranks (their clocks differ).
The spilled trajectory goes to `rank_RRRR` of its directory. The kernel
command gathers each viscosity kernel to rank 0, which writes it over a
whole-mesh coupling (`io::WriteKernels`'s coupling overload). The 3D
evolution now shares rank 0's whole sea level response too, so that every
rank's snapshots are the same and whatever is decided from them (a
calibration's next run) is decided alike.

Tests: `tests/TestAppsDistributed.cpp` on 2 and 3 ranks: evolve and kernel
against the one-process run (rank 0 alone, as one rank), a run stopped at
every checkpoint and resumed from the ranks' subdirectories (bitwise the
uninterrupted one), and the per-rank spilled trajectory (bitwise the
in-memory one). By hand, the evolve example at lmax 16 (8 nodes) through
`mpiexec -n 2 sl3d evolve`: within 3.4e-12 of the one-process run (whose
solves are direct, not substructured), the resumed run bitwise the same.

**Not yet, for the server:** the timing runs; the memory each rank keeps
beyond its segment (about half a gigabyte at lmax 64, not itemised); and
`sl3d`'s calibration and kernel paths at scale. A rank's checkpoints are
raw binary for its machine, as before.
