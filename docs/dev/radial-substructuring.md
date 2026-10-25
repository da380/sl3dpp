# Radial substructuring: a plan for distributing the 3D step

*Written 25 October 2026 against commit 6205daa (develop), with the
measurements of [multicore-scaling.md](multicore-scaling.md). A plan; the
code then was a serial prototype of the per-degree solve
(`benchmarks/substructure_prototype.cpp`, §4). Step 1 of §5 was built the
same day (§7), the distributed solve over MPI (§8) and, on David's
direction, the transforms cut with depth (§10), the distributed 3D
step (§11), and the distributed adjoint and sl3d (§12).*

## 1. Why radius

On the two-socket server a 3D step at lmax 128 is fastest on one socket's
32 cores, and the second socket makes it slower (multicore-scaling.md §3).
Within a pass the transforms take about three quarters of the time and the
banded solves a few per cent; once per step `StartStep` takes about a
fifth. Memory is the other limit: one recorded 3D state is about 21 GB at
lmax 256 and 98 GB at lmax 512.

Every large array of the step is indexed by radial memory node first:
the memory tensors m_k, the grid strain, the weights beta_k and W, the
inverse relaxation times, the spectral strain and forces. The transforms
act node by node (a batch over nodes), the pointwise products point by
point, and the force assembly and strain recovery element by element. Only
the per-degree solves couple the nodes, and they couple them along a band.
Cutting the radial mesh into P contiguous segments therefore divides the
transform work, the pointwise work and the memory by P, with the solves the
one place where segments must talk. That is the case for MPI over radius,
with one rank per socket (or per node) and OpenMP inside each rank, as now.

It also answers the one-socket limit directly. GSHTrans threads each
transform over colatitude rows, and keeps one level of threading (its
`Execution::TeamSize`: no threads inside an active parallel region), so a
single process cannot run "one group of rows-threaded transforms per
socket". Threading the transforms over node blocks within a process was
measured on 25 October and is 45% slower than rows at lmax 64 on 8 threads
(multicore-scaling.md §6): every thread then streams the whole stored
Wigner table. A rank per socket gives each socket its own nodes, its own
copy of the table in its own L3 and memory, and GSHTrans its rows inside.

## 2. What each rank owns

- **Its elements and their memory nodes**: m_k, the strain on the grid and
  in coefficients, beta_k, W, itau(x), delta(x); the recorded states of
  its nodes (the adjoint's `on_state` store writes one file per rank).
- **Its rows of every degree system**: the dofs of its elements, the node
  shared with the segment below (a separator) belonging to both.
- **The surface** belongs to the top segment's rank: the surface dofs (U,
  V, phi at r = a), the load and tide channels enter there, and the sea
  level solver runs there. Its inputs from the rest are the inertia moment
  (the degree-0 moment row runs over every radial dof: a sum over ranks)
  and nothing else; its outputs (the load and centrifugal coefficients,
  (lmax + 1)(lmax + 2) / 2 complex values each) go to every rank.

Contiguous ranges of elements are chosen to balance memory nodes (the
transforms and the memory both scale with them), cut at element
boundaries inside a layer — never at a layer or solid–fluid boundary,
where the dofs are split or constrained (`DegreeOperator::Interfaces`).

## 3. One pass, distributed

1. Synthesise the strain, multiply by W, analyse: each rank its own
   nodes. No communication.
2. Assemble each degree's force: element-local; a separator's row receives
   contributions from both sides.
3. Solve each degree by substructuring (§4): one exchange of the reduced
   right-hand sides at the separators and of the separator solutions.
4. The sea level coupling on the top rank: surface values are already
   there; the inertia moment is one all-reduce of a scalar; the channel
   coefficients are broadcast back.
5. Add the channels (each rank holds its part of each degree's unit
   responses) and recover the strain: element-local.
6. Mix and test: the Anderson mixer's Gram system and residual norms are
   all-reduces of a few numbers; the update itself is local.

Per pass that is one exchange at the separators, a broadcast of the channel
coefficients and a handful of scalar reductions. At lmax 128 the separator
exchange is 3 (lmax + 1)(lmax + 2) real values per separator for the
spheroidal systems, about 0.4 MB, against a pass of about 3 s: the
communication is negligible on any interconnect. `StartStep` is local
except that it ends with a pass's first force (no extra communication).

## 4. The degree solve by substructuring, and the prototype

Number a segment's dofs as interior (I) and separator (Γ: the one shared
node at each internal cut, 3 dofs for the spheroidal systems, 1 for the
toroidal). Each segment eliminates its interior,

    S_p = K_ΓΓ^p − K_ΓI^p (K_II^p)⁻¹ K_IΓ^p,   g_p = b_Γ^p − K_ΓI^p (K_II^p)⁻¹ b_I^p,

the separators solve the assembled S x_Γ = g (block tridiagonal, 3 × 3
blocks, size 3(P − 1)), and each segment back-substitutes,

    x_I = (K_II^p)⁻¹ b_I − ((K_II^p)⁻¹ K_IΓ^p) x_Γ,

with both terms kept from the reduction, so each right-hand side costs one
interior solve. The interior factorisations and the coupling columns
(K_II⁻¹ K_IΓ, at most six) are per (degree, scheme, dt), cached exactly
as the effective systems are now.

`benchmarks/substructure_prototype.cpp` does this serially on a synthetic
system with the degree systems' structure: a 1D spectral-element assembly,
five GLL nodes per element, three coupled components per node in node-major
order (band 14, as `DegreeOperator`'s), each element a random symmetric
positive definite block scaled over ten decades across the radius (the real
systems' condition numbers reach 2.5e11 at degree 1). With 240 elements
(2883 dofs) and 258 right-hand sides (degree 128's 2(l + 1) columns), on
this laptop, one thread:

| Parts | max \|x − x_direct\| / max \|x_direct\| | residual (direct 3.9e-15) | Schur dofs | solves per part |
|---|---|---|---|---|
| 1 | 0 | 3.9e-15 | 0 | 38 ms |
| 2 | 3e-20 | 3.9e-15 | 3 | 28 ms |
| 8 | 1.3e-16 | 3.9e-15 | 21 | 7.6 ms |
| 32 | 9.8e-16 | 4.2e-15 | 93 | 1.3 ms |

The direct solve takes 31–45 ms (timing noise at this size). The
substructured work summed over parts is about 1.3 times the direct solve,
and the per-part time falls as 1/P; the interface system is tiny (the
prototype solves it densely, which is fine to P ≈ 32; a block-tridiagonal
elimination is linear in P beyond that). Agreement is at rounding: the
Schur complement of a positive definite matrix is positive definite and no
worse conditioned, and the elimination is the banded Cholesky's own
arithmetic regrouped.

What the real systems add, each local:

- **Degree-dependent truncation.** Degree l uses the elements from
  `FirstElement(l)` up, so at high degree the deep segments hold no dofs
  and drop out of that degree's Schur system. The solves are a few per
  cent of a pass, so the imbalance costs little; the transforms, which do
  not truncate, stay balanced.
- **The fluid core.** Systems whose Cholesky fails fall back to banded LU
  now; a segment does the same, and the Schur system is then solved by LU.
  Cuts stay out of the fluid regions' boundaries (§2).
- **Degree one.** The centre-of-mass frame removes the surface phi dof
  (the top segment's); the toroidal pins remove one dof per solid region,
  in whichever segment holds it.

## 5. Order of work

1. **Substructured `DegreeOperator::Solve`, serial**, behind an option (the
   segment cuts), gated against the direct solve on the real systems at
   lmax 64 for every degree (agreement to 1e-13 relative, residuals equal)
   and through the 3D step's fingerprint. This is where the assembly,
   the LU fall-back and the pins meet the algebra above; it needs nothing
   from MPI.
2. **Segment-offset data in GridCoupling and Viscoelastic3D**: the node
   arrays and the per-degree solutions addressed by segment, still in one
   process. Bitwise against the current step.
3. **MPI**: one rank per segment, the loops over segments replaced by
   ranks, the collectives of §3. The 1D code stays as it is.
4. **The server**: 2 ranks × 32 threads on one node against 1 × 32 and
   1 × 64 (multicore-scaling.md's sweep), then two nodes.

Alongside: the adjoints reuse the step and inherit the distribution; the
checkpoint and trajectory stores become per rank, with the root writing
the run's metadata.

## 6. What it leaves

The sea level solve stays on one rank: two-dimensional and cheap now, it
should be measured at lmax 512 before it is distributed. Within a rank the
transforms keep GSHTrans's row threading, which scales on one socket; the
forward transform's per-thread accumulators (GSHTrans `docs/open-issues.md`
§4) still matter there, and a decomposition over orders m would serve both
designs.

## 7. Step 1, built (25 October 2026, against 7a14aaa)

`detail::BandSubstructure` (`include/sl3dpp/detail/BandSubstructure.hpp`)
is §4's algebra on any symmetric band matrix given its separators: each
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

The gate (`tests/TestSubstructure.cpp`) found that this plan's "1e-13
relative" target was optimistic. The two solves have the same residuals,
but the degree systems determine their solutions only to their
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
the cuts must balance the memory nodes (the transforms, the pointwise work
and the memory, §2), which do not truncate, so the solves' imbalance is
accepted. A rank's interior solves fall with its share of the dofs either
way.

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

Every process passes the whole right-hand side; only its own rows and the
separators' (from segment 0's owner) are read. When the 3D step's force
assembly is distributed, each rank will have its own elements'
contributions at its cut nodes, partial sums. Then the separators' right-hand
side should be combined as shares rather than taken from one owner: a
small change in `Solve`, to make when step 2 lands.

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

What remains is §5's steps 2 and 3: each rank holding its own memory
nodes, transforms, force assembly and strain recovery; the sea level
solve on the top rank; and the per-pass collectives of §3.

## 9. Steps 2–3 against the code: a proposal (25 October 2026)

What the 3D step does today, with the places a distributed one changes.
Line numbers are against the working tree of 25 October (after §8).

**The memory nodes.** `GridCoupling`'s constructor makes one node per
memory point of every element with an active branch
(`src/GridCoupling.cpp:37-46`). Every grid-sized array is
`[node * FieldSize + x]`: the tensors (`TensorField`, `SpectralTensor`),
`Viscoelastic3D::itau_` and `muBranch_` (built at
`src/Viscoelastic3D.cpp:120-160`), the step's weights, memory and buffers
(`StartStep`, `src/Viscoelastic3D.cpp:576`). Proposal: a
`GridCoupling::Options` element range [first, end), the nodes of those
elements alone. Every node loop then runs over the rank's own nodes
without further change: transforms, pointwise products, `AdvanceMemory`,
the itau kernels. The whole range is today's coupling, bitwise.

**The force and the solves.** `SolveAll` (`src/Viscoelastic3D.cpp:276`)
takes the forces from `SpheroidalForce` / `ToroidalForce`
(`src/GridCoupling.cpp:279`), which sum over the nodes into each degree's
rows. On a rank these are its own elements' rows, partial at the cut
nodes, where both sides contribute. `BandSubstructure::Solve` therefore
needs the separators' right-hand side as shares, summed by the combine,
rather than taken from segment 0's owner (§8): a flag. The solves are §8's.
`AddStrainOfSolution` (`src/GridCoupling.cpp:223`) reads the solution at
its own elements' dofs, which a rank has: its interiors and every cut node.

**The channels.** `AddUnitChannels` adds the load and tide responses by
linearity from unit solves cached per degree. Those are distributed solves
too, with the load at the surface rows (the top rank's interior).

**What must be summed or maximised over the ranks.**
- `OpenStep::InertiaMoment` (`src/Viscoelastic3D.cpp:772`): the degree-0
  moment row against the solution. Each rank sums its own dofs, the cut
  nodes counted once, then a sum.
- `Surface` (`src/Viscoelastic3D.cpp:914`): the top dofs belong to the
  top rank. A sum of the per-degree surface columns, zeros elsewhere,
  gives them to every rank: (lmax + 1)(lmax + 2) complex values per
  component.
- `OpenStep::Converge` (`src/Viscoelastic3D.cpp:818`): the convergence
  test is a sup-norm ratio (a max over the ranks), and the Anderson mixer
  (`detail/AndersonMixer.hpp`) needs its Gram matrix and right-hand side
  summed over the ranks; the update itself is local.
- `SeaLevelEvolution3D::March`'s pass (`src/Evolution3D.cpp:302-340`):
  with the surface columns and the moment on every rank, every rank can
  run the sea level solve redundantly. It is 2D, cheap and deterministic,
  so the ranks agree without a broadcast. That is simpler than solving on
  the top rank and broadcasting the load and centrifugal coefficients,
  and costs nothing measurable.

So the solver core needs a sum and a max over the ranks, not only §8's
sum. Proposal: replace `combine` with a small struct of the two
callbacks, `{sum, max}`, which `Distributed.hpp` fills from a
communicator. The alternative is MPI calls in the core under
`#ifdef SL3DPP_HAVE_MPI`. The callbacks keep every MPI line in one file,
and they let a single-process test pass an identity (or a two-"rank"
emulation) and check bitwise identity with today's step.

**Order.** (a) The share flag in `BandSubstructure::Solve`. (b) The
`GridCoupling` element range, with the whole range bitwise as now.
(c) `Viscoelastic3D::Options` gains the range, segments and reductions;
`InertiaMoment`, `Surface`, `Converge` and the unit channels use them.
(d) `SeaLevelEvolution3D` runs as is on every rank. (e) An MPI test: a
lateral evolution on 2–4 ranks equal to the one-process one to the fixed
point's tolerance. (f) The step on the server: 2 ranks × 32 threads
against 1 × 64. Then the 3D adjoint (`src/Adjoint3D.cpp`: its jump, the
reference's unit responses and the kernel accumulation over nodes, which
is local) and the per-rank checkpoint and trajectory stores.

## 10. Transforms cut with depth: the node blocks (25 October 2026)

The first of the steps agreed after §9 (David, 25 October): a transform
degree per radial partition, chosen so that nothing is lost, with the cuts
later balanced on the transform work rather than on node counts. Built on
one process first; the MPI distribution of §9 follows it.

**Why nothing is lost.** Degree l's system has dofs only above its
truncation radius, (r/a)^l = eps under the default graded rule
(`RadialMesh::StartElement`). So a memory node's strain carries no degree
above L(r) = ln(1/eps) / ln(a/r), and the memory force it produces above
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

**Next** (step 2 of the agreed order): `ShareMesh` balanced on
`TransformWeights` rather than one per element, then the element range
per rank (step 3). Computed on 25 October for prem.simp.vm1, with every
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

## 11. The 3D step distributed (step 3, 25 October 2026)

Built on David's answers to §9's questions: callbacks rather than MPI in
the core (Q2), the sea level equation solved on every rank (Q3), per-rank
checkpoints (Q4, for the app layer, not built yet).

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
transform work balances (the deep rank holds more,
cheaper nodes); and two ranks of four threads cost nothing. Pure MPI is
slower here for two reasons: one-thread ranks stream the whole Wigner
table each (the §1 node-block pattern), and the high degrees, whose dofs
lie near the surface, are solved by the top rank alone on its one thread.

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
wait at their next collective. Checkpoints, per Q4: each rank writes its
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
