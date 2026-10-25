# Multicore scaling: first measurements on a two-socket node

*Written 24 October 2026 against commit faea9f5 (main). The first runs of
sl3dpp on a large shared-memory node, what they show, and the plans that
follow. Nothing here is implemented beyond §1; the plans are for later
work, each to be measured before and after with the sweeps below. §6 adds
the work of 25 October: plan 1 built, plan 3's node blocks measured.*

## 1. The machines, and the OpenBLAS crash (fixed)

| | Server | Laptop |
|---|---|---|
| CPU | 2 × AMD EPYC 9334 (Zen 4), 32 cores each, 3.9 GHz boost | AMD Ryzen AI 7 PRO 350 (Zen 5), 8 cores, 5.1 GHz boost |
| Logical CPUs | 128 (two per core); `nproc` reports 128 | 16 |
| NUMA nodes | 2 | 1 |
| Compiler | GCC 13.2, Release | GCC 13.3, Release |
| BLAS | OpenBLAS 0.3.21, pthread build, `MAX_THREADS=128` | OpenBLAS 0.3.26, OpenMP build, `MAX_THREADS=64` |

The first `sl3d fingerprint` on the server segfaulted inside OpenBLAS
("precompiled NUM_THREADS exceeded"; "too many memory regions"). The
pthread build starts a worker pool of one thread per core when it loads,
and each worker holds a slot of a fixed buffer table (twice
`MAX_THREADS`); the per-degree loops then called LAPACK's banded routines
from every OpenMP thread at once, overflowed into OpenBLAS's buggy
overflow path, and crashed (OpenBLAS issue #5958, fixed only on its
development branch). `OPENBLAS_NUM_THREADS=1` avoided it but would slow
other codes on the same machine. The fix (commit 6459e62) gave sl3dpp its
own banded Cholesky and LU (`detail/BandSolvers.hpp`, LAPACK's unblocked
algorithms with its storage and pivoting), so no BLAS is called inside the
library's threads. Against LAPACK on the real per-degree systems (degrees
0 to 1024 at lmax 1024): identical pivots, backward errors about 2e-16 for
both; the solutions differ by up to 4e-9, inside what the systems' condition
numbers (up to 2.5e11 at degree 1) allow. On the server the fingerprint
then ran in 0.69 s (2.14 s under the environment-variable workaround),
and every `sl3d` command completed.

## 2. Placement and thread count on a small problem

`examples/configs/evolve.json` (3D, lmax 16), whole run:

| Machine | Threads | Placement | Time |
|---|---|---|---|
| Laptop | 1 | | 8.8 s |
| Laptop | 8 | default | 2.8 s |
| Server | 1 | | 18.3 s |
| Server | 8 | default | 12.0 s |
| Server | 8 | `OMP_PLACES=cores OMP_PROC_BIND=close` | 4.4 s |
| Server | 32 | default | 9.9 s |
| Server | 128 | default | 28.4 s (43 s in a first run) |

- A server core is about 2.1 times slower than a laptop core here.
- Pinning made the 8-thread run 2.7 times faster: unpinned, the threads
  wander across both sockets and away from their memory.
- OpenMP's default of one thread per logical CPU puts two threads on each
  core; with that, unpinned, and with too little work for 128 threads at
  lmax 16, the 128-thread run is the slowest.

## 3. `step_timing` at lmax 128 on the server

`step_timing tests/data/prem.simp.vm1 128 2 2`, `OMP_PLACES=cores`, peak
RSS 34 GB, one TensorField 4.13 GB (1388 memory nodes, grid 193 × 385).
Seconds:

| Binding, threads | per pass | analyse | synthesise | solves | startStep |
|---|---|---|---|---|---|
| close, 16 | 3.77 | 1.68 | 1.20 | 0.17 | 4.98 |
| close, 32 | **2.89** | 1.50 | 0.74 | 0.08 | 4.51 |
| close, 64 | 4.16 | 2.34 | 0.99 | 0.07 | 6.35 |
| spread, 16 | 4.83 | 2.44 | 1.75 | (noise) | 6.55 |
| spread, 32 | 4.02 | 1.98 | 1.26 | 0.05 | 5.58 |
| spread, 64 | 4.11 | 2.30 | 0.99 | 0.06 | 6.20 |

(`solves` is respond less its other parts, so small values are noisy; one
came out negative. The 64-thread runs under both bindings gave bitwise the
same state, as they place the threads alike.)

- **One socket is fastest.** 32 threads under `close` (socket 0's cores)
  beat all 64 by a factor of 1.4, and `close` beat `spread` at 16 and 32.
  Threads on the second socket reach across for arrays first touched on
  the first.
- **The transforms dominate and stop scaling.** Analyse and synthesise are
  about three quarters of `respond`. Analyse goes 1.68, 1.50, 2.34 s at
  16, 32, 64 threads. GSHTrans's forward transform gives each thread a
  private accumulator over the whole coefficient array for its colatitudes
  and then reduces them (in parallel, block by block), so the accumulators
  and the reduction's reads grow with the thread count. GSHTrans expects
  this: its comment at `GSHTrans/SphericalGrid.hpp:1242` (at the pinned
  commit) and its `docs/open-issues.md` §4, "Open questions for the
  64-core target", say the shape stops scaling well before 128 threads and
  leave the replacement, and the chunk rule on multi-socket machines, to
  measurements on such a machine. These are the first.
- **The banded solves scale** (0.17, 0.08, 0.07 s).
- **`StartStep` is about a fifth of a step and does not scale**; §4.

The 32-thread step is 20 s (7 passes and `StartStep`), so at 64 cores
the node does no better than one socket.

## 4. Why `StartStep` does not scale

`Viscoelastic3D::StartStep` (`src/Viscoelastic3D.cpp:493`), once per step:

1. copies the start strain's coefficients (`s0_`, spectral, small);
2. allocates and zero-fills, on one thread, `weight_` (nodes × grid
   doubles, 0.8 GB at lmax 128), the grid workspace `grid_`
   (`coupling_.ZeroTensor()`, 4.13 GB) and, per relaxing branch,
   `memory_[k]` (4.13 GB) and `beta_[k]` (0.8 GB): about 10 GB for the one
   Maxwell branch of these runs. `ZeroTensor` (`src/GridCoupling.cpp:64`)
   is `std::vector::assign`, serial;
3. computes the step weights, the decayed memory and the fixed memory force
   per grid point (`ParallelElements`, threaded);
4. analyses the fixed memory force (`fixed_`): one transform;
5. forms the first pass's lateral force from the grid strain in hand and
   analyses it (`analysed_`): a second transform, which is the first
   pass's own analysis moved here (that pass then skips its synthesis).

`Finish` (`src/Viscoelastic3D.cpp:696`) moves `memory_` and `grid_` into
the new state, so nothing is copied there, but the old state's arrays are
freed when it is replaced, and the next `StartStep` allocates fresh ones.

Splitting the measured `startStep` (which includes freeing the step):

| Machine, lmax, threads | startStep | two analyses | remainder |
|---|---|---|---|
| laptop, 64, 1 | 1.69 | 0.96 | 0.73 |
| laptop, 64, 8 | 0.92 | 0.29 | 0.63 |
| laptop, 64, 8, large pages | 0.52 | 0.26 | 0.27 |
| server, 128, close 16 / 32 / 64 | 4.98 / 4.51 / 6.35 | 3.36 / 3.00 / 4.69 | 1.6 / 1.5 / 1.7 |

The remainder hardly moves with the thread count: it is item 2. Requesting
transparent huge pages (`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`, the
laptop's setting being `[madvise]`) cut it from 0.63 to 0.27 s, so about
60% is the kernel faulting fresh memory in 4 KB pages on one thread, and
most of the rest is the serial memset (1.8 GB at lmax 64). The same
setting made a whole pass 23% faster (0.57 to 0.44 s), more than the 10%
the README quotes for an older build. The two analyses carry the
transforms' scaling problem (§3) into `StartStep`.

A second cost is placement: every array of item 2 is first touched by the
main thread, so on a multi-socket node all of them, `grid_` (the
workspace of every pass) and `weight_` (read by every pass's pointwise
product) among them, live on one socket's memory.

## 5. Plans

In order of expected value against effort; each to be measured with §3's
sweep (and §4's split) before and after, and the results added here.

1. **Reuse the step's arrays across steps.** Keep `weight_`, `beta_` and a
   grid workspace for the whole run (in the evolution, or as a workspace
   `StartStep` is handed), and let a step take over the previous state's
   `m[k]` (the decayed memory is formed elementwise, so in place is safe)
   through an overload taking the state by value; the adjoints, which
   replay recorded states, keep the `const&` form. No fresh pages after
   the first step, nothing freed per step.
2. **First touch in the threads that use the arrays.** Where arrays are
   still created, create them without value-initialisation and write them
   first inside `ParallelElements` (whose static schedule hands each
   thread the same block on every pass), so the pages land on the socket
   that uses them and the faulting is threaded. `memory_[k]` and
   `beta_[k]` are wholly overwritten by item 3's loop and need no zeroing;
   `grid_` and `weight_` accumulate over branches and can be written,
   not added to, for the first. Needs either reuse (plan 1) or an
   allocation that does not zero (a default-initialising allocator, or
   unique arrays), which touches `GridCoupling::TensorField`'s type.
3. **Transform threading.** The largest share. Either through GSHTrans,
   answering its open questions (§3) with these measurements as an
   open-issues item: the alternatives it lists are blocks of m, the batch
   axis (here the radial memory nodes) and a two-dimensional split; or by
   sl3dpp threading over radial-node chunks itself, with a serial GSHTrans
   call per chunk and thread-local workspaces (the performance notes of 17
   October, item 3). Likely with groups of threads by socket on
   multi-socket nodes.
4. **Huge pages by default.** Until plans 1 and 2 land, document
   `glibc.malloc.hugetlb=1` more prominently, or have the large
   allocations `madvise(MADV_HUGEPAGE)` themselves. Measure on the server
   first: `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` with §3's close-32 run.
5. **Thread counts by work.** Loops over lmax + 1 degrees or a few dozen
   rows should not start more threads than they have work for: a
   `num_threads` cap from the work size, so that small problems on large
   nodes stop slowing down with more threads.

Advice that already follows, in README *Running*: one thread per physical
core, pinned, one NUMA node per run, and one job per NUMA node with
`numactl` to use a whole node.

## 6. 25 October: plan 1 built, plan 3's node blocks measured

Laptop, Release, `step_timing tests/data/prem.simp.vm1 <lmax> <steps>`,
8 threads with `OMP_PLACES=cores OMP_PROC_BIND=close`.

**Plan 1, reuse of the step's arrays, is built.** `Viscoelastic3D` keeps
one step's worth of grid-sized arrays: a closed step hands back its
weights (`~OpenStep`), `Recycle(State&&)` hands back the tensors of a state
the caller is done with, and `StartStep` takes from these before it
allocates, writing every array before reading it (the first branch assigns
where the others add, `0.0 + v` as the accumulation onto zeros gave). The
3D evolution and the continuous 3D adjoint recycle each state a step
replaces and release the arrays at the end of a run; `step_timing`
recycles like the evolution. The one-thread state is bitwise the old
(fingerprint `36dbd27521e4fddf` at lmax 32, three steps, before and after),
and `Lateral.RecycledArraysGiveBitwiseTheFreshSteps` holds a recycling run
to the allocating one.

| lmax | startStep before → after | whole steps before → after |
|---|---|---|
| 64 (3 steps) | 1.14 → 0.77 s | 17.1 → 15.7 s |
| 96 (2 steps) | 3.69 → 2.71 s | 37.5 → 36.8 s |

What `StartStep` has left is mostly its two analyses (about 1 s each at
lmax 96): its allocation remainder was ~1 s there and is now the decayed
memory and weights loop. On the server the remainder was 1.5–1.7 s of a
~20 s step at lmax 128, so expect a few per cent there. Peak RSS is
unchanged (the arrays the old code freed and reallocated are the ones kept).

**Plan 2 (first touch) is not built.** With plan 1 the fresh pages are
faulted once per run, not per step; placing them by socket still needs
`TensorField`'s vectors to skip value-initialisation (a type change through
the coupling), and matters only for runs spanning sockets — which §3 says
to avoid, and which radial substructuring (below) would replace.

**Plan 3: node blocks lose.** Threading the tensor transforms over blocks
of radial nodes (each block one sequential GSHTrans call, GSHTrans's
workspaces being `thread_local`) instead of GSHTrans's colatitude rows was
built and measured at lmax 64: analyse 0.346 s against 0.238 s with rows,
synthesise 0.317 against 0.215, the pass 0.98 against 0.75 s. With the
Wigner values stored, each thread of a node block streams the whole table
(about 16 MB here, this chip's entire L3; ~128 MB at lmax 128), where rows
give each thread its own colatitudes' slice. It was removed again. Two
consequences: a "rows inside, groups of nodes per socket" hybrid cannot be
built inside one process, because GSHTrans keeps one level of threading
(`Execution::TeamSize`: no threads inside an active parallel region); and
the natural way to give each socket its own nodes and its own copy of the
table is a process per socket — see
[radial-substructuring.md](radial-substructuring.md). Within a socket the
open question stays GSHTrans's (its `docs/open-issues.md` §4): the forward
transform's per-thread accumulators, where a decomposition over orders m
would write disjoint coefficients and read only its own orders' Wigner
values.

**Plan 5: thread counts.** On the laptop the 3D example (`evolve.json`,
lmax 16) takes 9.5 s on one thread, 4.9 s on 4, 4.3 s on 8 and 5.9 s on 16
(two threads per core): the loss past the physical cores is the shared
core, which a cap by work size would not remove. The README's advice (one
thread per physical core, pinned) is the remedy; no cap was built.

