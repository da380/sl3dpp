# Radial partitioning

How sl3dpp splits the 3D problem over contiguous radial segments: the
substructured degree solves and their distributed form, the transforms
cut with depth, the balance of the cuts, the communication, and the rest
of the step and adjoint run over MPI ranks. This is the maintained
account; the measurements and the order in which it was built are in
[dev/radial-substructuring.md](https://github.com/da380/sl3dpp/blob/main/docs/dev/radial-substructuring.md),
the formulation in [design.md](design.md), and running `sl3d` over ranks in
[sl3d-configs.md](sl3d-configs.md).

## 1. Why radius

Every large array of the 3D step is indexed by radial memory node first:
the memory tensors, the strain on the grid and in coefficients, the step
weights, the inverse relaxation times. Each node's work is its own: the
transforms act node by node, the pointwise products point by point, the
force assembly and strain recovery element by element. Only the
per-degree solves couple the nodes, and they couple them along a band,
neighbour to neighbour in radius. Cutting the radial mesh into P
contiguous segments therefore divides the transform work, the pointwise
work and the memory by P, and leaves the solves as the one place where
segments meet, through the nodes at the cuts alone. No global transposes
are needed, which is what makes radius, rather than angle, the natural
direction to split.

A cut is an element boundary inside a layer, never at a model boundary or
a solid–fluid one, where a node carries split one-sided values or a
constraint (`RadialMesh::BalancedCuts` chooses only such boundaries).

## 2. The degree systems

For each degree l, the spheroidal system (U, V and φ at each node; fewer
components in the fluid) and the toroidal one (W at each solid node) are
symmetric band systems in node-major order: with five GLL nodes per
element the spheroidal band has kd = 14 superdiagonals. Each is scaled
symmetrically to unit diagonal, A → D A D, and factorised by banded
Cholesky, or banded LU where Cholesky fails (systems with fluid regions).

Degree l keeps only the elements above its truncation radius,
r_l = a ε^(1/(l+1)) under the default graded rule (ε = 1e-8): its first
element `FirstElement(l)` rises with l, so the high degrees have no dofs
in the deep mantle. A node at radius r therefore carries degrees up to
about

    L(r) = ln(1/ε) / ln(a/r) − 1,

which at ε = 1e-8 is 256 at 440 km depth, 128 at 845 km, 64 at 1560 km
and 32 at 2710 km.

## 3. Substructuring

Let the cut nodes be c_1 < … < c_{P−1}. A separator Γ_k is the set of
dofs of cut node c_k (consecutive in the node-major numbering); segment p
is the set of dofs strictly between its two cuts. An element couples
only its own nodes, so a segment's interior I_p couples only to its two
adjacent separators (checked at construction). Writing the system with
each segment's interior eliminated,

    S   = K_ΓΓ − Σ_p K_ΓI^p (K_II^p)⁻¹ K_IΓ^p
    g   = b_Γ  − Σ_p K_ΓI^p (K_II^p)⁻¹ b_I^p
    S x_Γ = g
    x_I^p = (K_II^p)⁻¹ b_I^p − ((K_II^p)⁻¹ K_IΓ^p) x_Γ

The Schur complement S of a positive definite matrix is positive definite
and no worse conditioned; it couples each separator only to its
neighbours, so it is block tridiagonal with blocks the separators' sizes
(3 × 3 spheroidal, 1 × 1 toroidal) and is factorised as a band.
Construction factorises each interior block and keeps the coupling
columns (K_II^p)⁻¹ K_IΓ^p, one per adjacent separator dof (at most six,
spheroidal); a right-hand side then costs one interior solve per segment,
the separators' solve, and the back-substitution, which reuses both
terms. The total work is 5–15% above the direct solve's.

Three features of the real systems enter locally:

- **Truncation.** A cut at or below `FirstElement(l)` is dropped for
  degree l: the segments below hold no dofs of that degree, and the first
  segment with dofs takes the dropped ones' place. The separators of
  degree l are the cuts above its first element.
- **Fluid regions.** A cut node in a fluid layer has no toroidal dof: its
  separator is empty, and the segments on either side are independent
  toroidal problems.
- **Degree one.** The toroidal gauge pins one dof per solid region; the
  pins are applied to the band before it is substructured, in whichever
  segment holds them.

## 4. The distributed solve

Each process owns one segment (or a contiguous run of them), factorises
its interiors, and forms its part of S: its segment's terms
K_ΓI (K_II)⁻¹ K_IΓ, with the separators' own block K_ΓΓ entered by one
process (the owner of the first segment). One sum over the processes then
gives every process S, which each factorises: it has 3(P − 1) dofs, far
smaller than any segment.

A solve needs g, and one sum gives it too: each process contributes
b_Γ's part it holds less its segment's term K_ΓI (K_II)⁻¹ b_I. A
right-hand side is held in one of two ways (`RightHandSide`):

- **Whole**, on every process: a unit load, a tide. The separators' rows
  are then read from one process.
- **Shares**: a force assembled element by element over each process's
  own elements. The separator rows are split between the two processes
  that share the cut, since both its elements contribute, and the sum adds
  the shares.

After the sum every process solves for x_Γ itself and back-substitutes its
own interiors. Its solution holds its own segment and every separator,
zero elsewhere. A process "holds" its interiors and, on one process, the
separators (`Holds`, `MaskToShare`), so that the held parts sum over the
processes to the whole solution. That is how a reduction over all dofs
(the degree-0 inertia moment) is formed: each process's share, summed.

A solve is split at its one collective (`BeginSolve`, the sum,
`FinishSolve`), so that the 3D step condenses every degree's systems in
parallel over threads and sums all their separators' parts in a single
collective per pass, from the main thread. No MPI call is made inside a
threaded region (MPI_THREAD_FUNNELED suffices).

## 5. Accuracy

The substructured solve is the banded factorisation's arithmetic
regrouped, and its residuals are the direct solve's to within a small
factor. Its solutions agree with the direct ones only as far as the
systems determine them. Their condition numbers reach 2.5e11 at degree
1, where the two solutions differ by up to 1.4e-10 relative, and the
direct one is no closer to the truth (both residuals are at round-off);
above degree 10 they agree to 1e-11. A distributed run against one
process on the same cuts differs only by the order of its sums: 5e-16 to
2e-13 in the 3D step's surface response and memory, 5e-15 in the coupled
evolution's sea level.

## 6. The transforms cut with depth

**What is kept.** A memory node in element e carries strain only in the
degrees whose subset reaches e, those with `FirstElement(l)` ≤ e; let
L(e) be the largest. The memory force the node produces in degrees above
L(e) has no rows to enter. So transforming the node at any degree
L_b ≥ L(e), on a grid of that degree, leaves every degree system's input
unchanged, except through the quadrature of the lateral product.

**The quadrature.** The node's force at degree l ≤ L_b is a projection

    ∫ W(x) s(x) Ȳ_lm(x) dΩ

of the strain s, band-limited at L_b, times the pointwise weights W,
which carry the lateral viscosity through the step's relaxation weights.
If W is band-limited at K, the integrand has degree at most 2L_b + K. The
grid's Gauss–Legendre colatitudes (n_θ of them) integrate exactly to
degree 2n_θ − 1, and its longitudes likewise. The 3/2 rule, n_θ ≈ 1.5 L_b
+ 1, is therefore exact for K up to about L_b. On the lmax grid a deep
node's product was exact to K ≈ 3 lmax − 2 L(e). The only change is in
W's components between those bounds. That is a matter of the viscosity's
spectrum, not the fields': lateral viscosity is typically smoother than
the solution, whose resolution and the shorelines set lmax. Measured: a
smooth degree-2 pattern changes the surface response by 3e-16 at lmax
128; a decade of contrast in such a pattern, with the truncation loosened
so that blocks form at lmax 16, by 3.8e-9. A deep viscosity rougher than
the solution there would want the listed degrees raised.

**The blocks.** `Options::transform_degrees` lists the degrees allowed.
Each viscous element's nodes take the least listed degree, or lmax, at or
above L(e) (`GridCoupling::TransformDegrees`, which needs no grid). L(e)
only grows outwards, so each degree's nodes form one contiguous block,
with its own grid and Wigner table at the same oversampling. Every field
over the nodes lays the blocks end to end (`FieldOffset`, `FieldSizeOf`,
`GridValueCount`). A node's coefficients are its block's degree's, a
prefix of lmax's in the (l, m) order, so carrying a field between blocks
is zero padding (`OnSurfaceGrid` puts every node on the lmax grid, for
output). With no degrees listed there is one block and the step is
unchanged bit for bit.

**The gain.** A transform at degree L costs about (L + 1)³ per node, its
grid (L + 1)². At lmax 128 with the degrees 32, 48, …, 112 the model
predicts 0.61 of the transform work, and the transforms take 0.63–0.65 of
their former time. The step takes 0.65 of its former time and 0.71 of its
memory. The model gives 0.42 at lmax 256. Below lmax ≈ 30 every mantle
node needs lmax under ε = 1e-8, and nothing is cut.

## 7. Balancing the cuts

The cuts balance the transform work, not the element count. Each
element's weight is its memory points times ((L_e + 1)/(lmax + 1))³ at its
transform degree (`GridCoupling::TransformWeights`). It is a formula, not a
timing, so every process computes the same cuts; `BalancedCuts` places
them at the admissible element boundaries nearest the equal shares. On
prem.simp.vm1 the busiest rank's transform work over the mean is:

| ranks | lmax 128 | lmax 256 |
|---|---|---|
| 2–16 | ≤ 1.04 | ≤ 1.04 |
| 32 | 1.12 | 1.20 |
| 64 | 1.28 | 1.37 |
| 128 | 1.92 | 1.37 |

Weighting by elements or by memory nodes instead gives 1.7–3.4 once the
transforms are cut with depth. The cuts concentrate near the surface, where
the work is. Two limits appear near a hundred ranks:

- **Granularity.** Cuts fall at whole elements, and the whole mesh is
  worth about 200 full-degree elements at lmax 128, so at 128 ranks a rank
  gets one or two surface elements and the balance goes.
- **The solves.** These cuts do not balance the solves. The high degrees
  have dofs only near the surface, so the top rank solves them alone. This
  is a few per cent of a pass with many threads per rank, and visible with
  one.

## 8. Communication

Per pass the one sizeable exchange is the sum of the separators' parts of
every degree system. For degree l that is (its separators) × (dofs per
separator) × 2(l + 1) real columns. Summed over degrees, with the cuts
balanced on the work:

| ranks | lmax 128 | lmax 256 |
|---|---|---|
| 16 | 7.5 MB | 29 MB |
| 64 | 31 MB | 119 MB |
| 128 | 62 MB | 238 MB |

It grows with the ranks because every process solves S whole, and so
needs every separator's g. On a 100 Gb/s network the sum is about 1% of a
rank's pass at lmax 128 on 16 ranks, rising to about 13% at lmax 256 on 64
(estimates). Beyond tens of ranks the remedies are:

- **Gather instead of sum.** Each process contributes to two separators
  alone, so gathering those pieces rather than summing the full vector
  roughly halves the traffic.
- **Overlap.** Begin the sum of one group of degrees while the next group
  condenses.
- **Distribute S itself.** Solve the block-tridiagonal S across the
  processes by cyclic reduction, in log P neighbour exchanges.

Each fits the callback interface below. The other collectives of a pass
are small:
- two sums of the surface values (U and φ, (lmax + 1)(lmax + 2)/2 complex
  values each);
- one sum of the inertia moment;
- the share of the sea level results from rank 0 (below);
- a maximum of three numbers for the convergence test;
- once Anderson mixing engages, a sum of its 2m Gram values.

Constructing a degree system sums its S once.

## 9. The distributed step, adjoint and program

The library calls no MPI itself. `Viscoelastic3D::Options` takes the cuts,
the segments this process owns, and `Collectives`: in-place sum, maximum,
and share from the first process. `Distributed.hpp` (built with
SL3DPP_WITH_MPI) fills them from a communicator, and `ShareMesh` gives a
rank its segment.

- **What a rank holds.** Its coupling holds the memory nodes of its own
  elements only (`GridCoupling::Options::elements`). The transforms, the
  pointwise products, the memory update, the force assembly and the strain
  recovery are all its own.
- **The sea level equation.** Every rank solves it from the same summed
  surface values. Each rank's transforms may round differently (it plans
  its own FFTs), so rank 0's results are shared to all after every solve:
  loads, centrifugal coefficients, ocean function and response. Every
  segment is then forced alike, a cell at the flotation threshold flips on
  all ranks or none, and whatever is decided from the snapshots, such as a
  calibration's next run, is decided alike.
- **Convergence.** The residual's maximum and the Anderson Gram system's
  sum are taken over the ranks, so the iteration takes the same course
  everywhere.
- **The 3D adjoint** runs the same way. Each rank sweeps its own nodes
  back over its own forward trajectory, and its viscosity kernels are its
  own nodes'; laid end to end in rank order they are the mesh's. The ice
  kernels are every rank's alike.
- **The `sl3d` program** runs `evolve` and `kernel` under `mpiexec`.
  - Rank 0 writes the outputs.
  - Each rank checkpoints its own segment into its own subdirectory. A
    run resumes from the newest node every rank has, over the same rank
    count only.
  - The viscosity kernels are gathered to rank 0 for writing.
  - A failure on any rank aborts the job.

One process with the threads inside it is the intended shape: on a
one-socket laptop, eight one-thread ranks were 50% slower per pass than
one eight-thread rank, since each streams the whole Wigner table and the
top one solves the high degrees alone.

## 10. In the code

| Piece | Where |
|---|---|
| The substructured band solve, its phases, the shares | `detail/BandSubstructure.hpp` |
| Cuts, segments, split solves, held parts | `DegreeOperator`, `ToroidalOperator` (`Options::cuts`, `segments`, `combine`; `BeginSolve`, `FinishSolve`, `Holds`, `MaskToShare`) |
| The collectives, right-hand sides, pending solves | `Collectives.hpp` |
| Node blocks, element ranges, work weights | `GridCoupling` (`Options::transform_degrees`, `elements`; `TransformDegrees`, `TransformWeights`, `OnSurfaceGrid`) |
| Admissible balanced cuts | `RadialMesh::BalancedCuts` |
| The distributed step | `Viscoelastic3D::Options` (`radial_cuts`, `segments`, `collectives`, `transform_degrees`) |
| MPI | `Distributed.hpp` (`ShareMesh`, `CollectivesOver`), `apps/Ranks.hpp` |
| Tests | `TestSubstructure`, `TestNodeBlocks`, `TestRadialRanks` (threads as ranks), `TestDistributed` and `TestAppsDistributed` (under mpiexec) |
| Benchmarks | `substructure_timing`, `step_timing` (degrees argument), `distributed_solves`, `distributed_step` |
