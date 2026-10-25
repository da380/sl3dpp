# Radial substructuring: a plan for distributing the 3D step

*Written 25 October 2026 against commit 6205daa (develop), with the
measurements of [multicore-scaling.md](multicore-scaling.md). A plan, not
built: the only code is a serial prototype of the per-degree solve
(`benchmarks/substructure_prototype.cpp`, §4).*

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
