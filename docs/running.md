# Running sl3dpp

How to run the `sl3d` program and the library well on a machine: the
threads and where they run, the memory a run needs, MPI ranks, long
runs, and the choice of a time step and scheme. Every config member named here is in the configuration reference,
[sl3d-configs.md](sl3d-configs.md); the build options are in
[building.md](building.md).

## Threads and where they run

sl3dpp is threaded with OpenMP throughout, and how the threads are placed
matters as much as how many there are. `lscpu` (on Linux) gives a
machine's layout: its sockets, cores per socket, threads per core and NUMA
nodes.

- **One thread per physical core.** OpenMP starts one thread per logical
  CPU by default, which on a machine with simultaneous multithreading
  (hyperthreading: two logical CPUs to a core) is two threads per core. The
  two share the core's floating-point units, so for sl3dpp the second
  gains little and adds to every synchronisation. Set `OMP_NUM_THREADS` to
  the number of physical cores, or fewer.
- **Pin the threads.** Unpinned threads are moved between cores by the
  operating system, and on a machine with several sockets (NUMA nodes)
  away from the memory they first touched, which then crosses between
  sockets on every access. Pinning keeps each thread on one core:

  ```
  export OMP_NUM_THREADS=<physical cores> OMP_PLACES=cores OMP_PROC_BIND=close
  ```

  On a two-socket node this made an 8-thread 3D run 2.7 times faster. The
  variables apply to every OpenMP program started from that shell, so set
  them per run, or in a job script, where other codes want other settings.
- **One NUMA node per run.** `close` puts threads on neighbouring cores,
  filling one socket before the next; `spread` divides them between the
  sockets. sl3dpp's large arrays live in the memory of whichever socket
  first touched them, so threads on another socket reach across for them,
  and a 3D run is fastest on the cores of one NUMA node (`lscpu`
  lists them; usually one socket). On a two-socket node with 32 cores per
  socket, a 3D step at lmax 128 ran 1.4 times faster on one socket's 32
  cores (`close`) than on all 64, and `close` beat `spread` at 16 and 32
  threads. To use a whole node, run one job per NUMA node, each with its
  memory kept there too (when the job fits in that node's memory):

  ```
  export OMP_NUM_THREADS=<cores per NUMA node> OMP_PLACES=cores OMP_PROC_BIND=close
  numactl --cpunodebind=0 --membind=0 sl3d solve first.json &
  numactl --cpunodebind=1 --membind=1 sl3d solve second.json &
  ```

  or spread one 3D run over the node, one MPI rank per socket (Over MPI
  ranks, below).
- **Small problems want fewer threads.** The parallel loops run over
  degrees (lmax + 1 of them), latitude rows and radial nodes, and below a
  degree of a few dozen there is too little work to share among the cores
  of a large node: more threads then make a run slower, not faster.
- **Cutting the transforms with depth.** `Viscoelastic3D::Options::
  transform_degrees` (sl3d's `solver.transform_degrees`) transforms each
  memory node only to the degrees its solution carries there under the
  degree truncation. At lmax 128, the list 32, 48, ..., 112 made a step
  1.5 times faster and cut its peak memory from 34 to 24 GB, with the
  surface response unchanged to round-off under a smooth viscosity
  pattern; a deep viscosity rougher than the solution there would want
  the listed degrees higher. Below about lmax 30 every mantle node needs
  lmax and nothing is cut ([radial-partitioning.md](radial-partitioning.md)
  §6).
- **Measuring.** `step_timing` (built with `-DSL3DPP_BUILD_BENCHMARKS=ON`)
  times each phase of a 3D step and reports the peak memory. Run it at the
  production degree over a few thread counts and both bindings, and take
  the fastest per-pass time (it needs about 34 GB at lmax 128):

  ```
  for bind in close spread; do for threads in 16 32 64; do
    OMP_NUM_THREADS=$threads OMP_PLACES=cores OMP_PROC_BIND=$bind \
      build/benchmarks/step_timing tests/data/prem.simp.vm1 128 2 2 \
      > step_${bind}_${threads}.txt
  done; done
  grep -H "per pass" step_*.txt
  ```

- **Reproducibility.** Threaded runs of the 3D step reproduce one-thread
  runs bitwise: `step_timing`'s state fingerprint is the same at 1, 4, 16
  and 32 threads, with and without Anderson mixing. This holds for the
  default loop kernel; with `matrix_kernel` the sums are the BLAS's, whose
  grouping may depend on its thread count. FFTW's measured plans are each
  process's own, chosen by timing, so where two processes' plans differ
  the last bits may too: two runs of one config, even on one thread, agree
  to about 1e-14 relative (sl3d's kernel tables and 3D runs among what
  differs; many outputs come out bitwise the same), and a run over MPI
  ranks agrees with one process to round-off (below), not bitwise.
- **BLAS.** Any BLAS works on any number of cores, OpenBLAS's pthread and
  OpenMP builds alike, with no environment variables to set: the library's
  threaded loops make no BLAS calls (the per-degree banded solves are its
  own).

## Memory

- **Size.** In 3D each relaxing branch holds its internal variables on a
  grid at every radial memory node: one such tensor is 0.7 GB at lmax 64,
  4 GB at 128, 21 GB at 256 and 98 GB at 512, and a step holds several.
  Viscosity kernels need the forward states: recording every `n`-th of `N`
  time nodes holds about `N/n + n` of them during the sweep, fewest near
  `n = √N` (`trajectory.record_stride`). With `trajectory.directory` the
  kept states live on disk and the sweep reads each back as it reaches it
  (Long runs, below).
- **Huge pages.** Fresh memory is faulted in one 4 KB page at a time by the
  allocating thread. Where transparent huge pages are on request only
  (`/sys/kernel/mm/transparent_hugepage/enabled` reads `[madvise]`), glibc
  2.35+ can request them:

  ```
  GLIBC_TUNABLES=glibc.malloc.hugetlb=1 OMP_NUM_THREADS=8 ./ExampleLateralViscosity
  ```

  The 3D step reuses its large arrays from step to step, so they are
  faulted in once per run: at lmax 64 on 8 threads huge pages make the 3D
  run about 1.5% faster, with unchanged results.
- **Memory points.** The internal variables live at the Gauss points of each
  radial element by default; `memory_points = MemoryPointSet::Gll` places
  them at the GLL nodes, as the Fortran codes and pyslfp do, for direct
  comparison.

## Over MPI ranks

Built with `SL3DPP_WITH_MPI` ([building.md](building.md), The MPI build),
`sl3d solve`, `sl3d kernel` and `sl3d tangent` run under `mpiexec`, the
same config on every rank:

```
OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close \
  mpiexec -n 4 -bind-to core:32 sl3d solve sea_level_3d.json
```

(the binding flags are MPICH's; another MPI names them otherwise).

- **What is distributed.** A 3D run, split over radius, one contiguous
  radial segment per rank with its threads inside: the static problem on
  lateral moduli, the runs through time, the kernels about them, by
  either sweep, and the tangent-linear runs. A 1D run is the first rank's
  alone, the other ranks
  returning at once, and `love-numbers` refuses several ranks. `kernel`
  refuses on several ranks the 3D kernels its sweeps form on one process:
  those about a run under seismic sources, and the continuous sweep's
  tide kernels. The library pieces behind this (the 3D step and its
  lateral solves, the continuous adjoint, the tangent-linear runs, the
  discrete transposes and the material kernels) are in
  [radial-partitioning.md](radial-partitioning.md) §9 and
  [adjoints.md](adjoints.md) §17.
- **Placement.** One rank per socket (or per node) with threads inside is
  the intended shape. Measured with `distributed_step` at lmax 128 on 32
  of a two-socket node's cores, each rank pinned to its own cores: the
  response phase of a pass is 1.4 times faster with two ranks of 16
  threads than with one process of 32, and 2.2 times faster with eight
  ranks of four, the forward transform scaling with each rank's share of
  the nodes rather than with the threads inside a rank. On a one-socket
  laptop eight one-thread ranks were 50% slower per pass than one
  eight-thread rank.
- **Results.** They agree with a one-process run to round-off of the
  radially substructured solves (about 1e-12 relative), not bitwise; an
  iterative solve that closes a pass earlier or later on the ranks than
  on one process moves the fields within its tolerance (up to about
  1e-10 relative at sl3d's default tolerances of 1e-9).
- **Files.** Rank 0 writes every output and prints the summary. Each rank
  writes its checkpoints and its trajectory-store states into its own
  `rank_RRRR` subdirectory, and a run resumes, or a store is swept, over
  the same number of ranks only ([sl3d-configs.md](sl3d-configs.md),
  "Over MPI ranks").

## Long runs

- **Checkpoints.** A `checkpoint` block makes a `solve` run through time
  write its state every `every` nodes and resume from the newest
  checkpoint when the same config is run again; `"restart": false` starts
  it over. With `max_walltime_min` the run stops cleanly at the first
  checkpoint written after that many minutes, with exit status 0 and a
  summary saying where it stopped, so a chain of batch jobs running one
  config completes the run:

  ```json
  "checkpoint": {"directory": "checkpoints", "every": 4, "max_walltime_min": 1380}
  ```

  The limit is checked only as a checkpoint is written, so set it below
  the batch system's by more than `every` nodes' stepping. A checkpoint is
  bound to its run by a hash of the config's computation and its input
  files' contents; the static problem and a run under seismic sources
  take none ([sl3d-configs.md](sl3d-configs.md), "Checkpoints and
  restarts").
- **The trajectory store.** A `trajectory` block with a `directory` makes
  `solve` record its forward run into a store on disk, from which `kernel`
  sweeps for any number of measurements, and about which `tangent` runs
  for any number of directions, without running the forward problem
  again, the kept 3D states read back one at a time: the lever on
  memory at large degree. With a `checkpoint` block beside it, a run
  stopped at the wall-time limit leaves its store incomplete to the node
  it stopped at, and the job that resumes it continues the store from the
  checkpoint's node: the chain of jobs that completes the run completes
  the store, the uninterrupted run's file by file. A resumption whose
  store lacks a record before its checkpoint, or has none, is refused
  (`"restart": false` starts over and records it whole). A `kernel` run
  given an absent or empty directory, or an incomplete store of its run,
  records the store there itself. The block is in
  [sl3d-configs.md](sl3d-configs.md) ("The trajectory block", with the
  store's size on disk), and the store's format in
  [trajectory-store.md](trajectory-store.md).

## Choosing a step and scheme

A run through time is as accurate as its steps are short against the
relaxation times of the layers that matter to what is measured, and how
short they must be depends on the time scheme (`solver.scheme`,
`Evolution::Options::scheme`). No step suits every model: the relaxation
times follow the viscosity, which varies by orders of magnitude between
models and within one. `sl3d solve` therefore states the number to read a
step against, at the end of its summary and in its output's metadata:

```
...; wrote out/timestep_et_n10/timestep_et_n10.nc; its longest step 9.27 times the shortest relaxation time (0.108 kyr, at 2886 km depth in layer 2, 2738 to 2888 km deep)
```

the run's longest step over the shortest relaxation time of the model as
built, τ_min (the least η/μ over every branch, depth and, under a lateral
viscosity, place), with where it is. `sl3d kernel` states the same for a
continuous kernel. A point whose relaxation time is long against the step
is resolved by any scheme; a point whose time is short against it is
stiff, its internal variable slaved to the strain, and the schemes differ
in what they do there.

**The schemes.**

| `solver.scheme` | order | solves a step | at a stiff point |
| --- | --- | --- | --- |
| `"sdirk2"` (the default) | 2 | 2 (one factorisation, both stages) | damped, second order |
| `"exponential_trapezoid"` | 2 | 1 | acts as backward Euler: damped, first order on the slow modes the point carries |
| `"backward_euler"` | 1 | 1 | damped |

In 3D a solve is a step's within-step iteration, and Sdirk2 takes about
twice the passes (43 a step against the trapezoid's 21 on the lateral
convergence test, design.md §5).

**What the ratio means, measured.** The time-step study
(examples/configs/README.md, `timestep_study.py`; every number below is
its own) runs a 10 kyr deglaciation on the prem8 deck's 1e21 Pa s Maxwell
mantle, τ_min = 0.108 kyr, at steps of 1 to 1/32 kyr, and measures the
errors of the cap-mean sea level at two dates (written a / b), of its
derivative in ln η by each adjoint, and of its derivative along an ice
pulse, against a truth extrapolated from forward runs:

| h/τ_min | value, trapezoid | value, Sdirk2 | dJ/d ln η discrete, trapezoid | Sdirk2 | dJ/d ln η continuous, trapezoid | Sdirk2 |
| --- | --- | --- | --- | --- | --- | --- |
| 9.3 | 0.22 / 0.50 | 0.015 / 0.043 | 4.0 / 1.7 | 0.38 / 0.26 | 14 / 24 | 1.9 / 1.0 |
| 4.6 | 0.059 / 0.18 | 0.0023 / 0.010 | 1.4 / 1.3 | 0.079 / 0.060 | 5.5 / 8.5 | 0.52 / 0.24 |
| 2.3 | 0.015 / 0.051 | 0.00052 / 0.0025 | 0.40 / 0.49 | 0.020 / 0.015 | 1.7 / 2.5 | 0.13 / 0.057 |
| 1.2 | 0.0038 / 0.013 | 0.00012 / 0.00063 | 0.10 / 0.14 | 0.0048 / 0.0037 | 0.44 / 0.64 | 0.034 / 0.014 |
| 0.58 | 0.00094 / 0.0033 | 0.000030 / 0.00016 | 0.026 / 0.036 | 0.0012 / 0.0010 | 0.11 / 0.16 | 0.0086 / 0.0034 |

(magnitudes, in per cent). Backward Euler's values are 0.38 / 0.75% off
at 9.3 and halve with the step.

- *The forward run.* Both second-order schemes fall by four at each
  halving once the ratio is near one. At long steps the trapezoid is
  backward Euler on the stiff points and its error is larger than its
  order suggests; Sdirk2's values are 12 to 15 times closer at a ratio of
  9.3 and 20 to 30 times closer below 2.3. At twice the solves it is to
  be held against the trapezoid at half the step: there too it is the
  closer on this problem at every step measured, by 4 times at 9.3
  (0.015 / 0.043% against the trapezoid's 0.059 / 0.18% at 4.6) and by 5
  to 8 times at 0.58. After an abrupt change of the forcing over a weak
  layer (an unloading, a seismic onset) the gap is wider: the trapezoid
  is first order there, Sdirk2 second, 7 to 15 times closer over a
  1e19 Pa s layer (design.md §5). On the closed-form Maxwell sphere
  without gravity, at 16 steps a relaxation time, the trapezoid, exact
  for a strain linear across a step, is the closer (7e-9 off against
  Sdirk2's 5.6e-8, design.md §5); the gravitating problem of the table
  does not show that, Sdirk2 the closer there down to the shortest step
  measured.
- *The discrete kernel* (`kernels.method` `"discrete"`, the default) is
  the exact gradient of the stepped run under every scheme, so its error
  is the forward run's own: choosing the step and scheme for the forward
  run chooses them for the kernel. It reads larger against a kernel than
  against the value (dJ/d ln η is 8 to 16% of J here, the part the step's
  error lives in): 4.0 / 1.7% under the trapezoid at 9.3, 0.38 / 0.26%
  under Sdirk2, and along the ice pulse 11 / 8.9% against 1.2 / 1.1%.
  No layer stands out: 3.1 to 4.2% under the trapezoid and 0.30 to 0.41%
  under Sdirk2 at the first date.
- *The continuous kernel* (`"continuous"`) solves a second problem, an
  adjoint that relaxes after each measurement on τ_min itself, whatever
  the forward run resolves. Under the trapezoid it needs steps short
  against τ_min: 14 / 24% off at 9.3 and 33 / 41% in its worst layer,
  1.7 / 2.5% at 2.3, 0.44 / 0.64% at 1.2. Under Sdirk2 the stages damp
  that relaxation and it is 1.9 / 1.0% off at 9.3 (4.6 / 1.7% in its
  worst layer, 5.2 / 4.4% along the pulse), falling at second order.

The numbers are this problem's; the ratio carries them to another model
only roughly (with the viscosity scaled by 4 and by 1/4 the continuous
kernel's error at one ratio moved by up to a factor of two), and not at
all where the layer that sets τ_min is one the measurement does not feel:
a thin weak layer can put the ratio in the hundreds while the run is
converged in everything measured, or leave it there unconverged in what
that layer controls.

**What to do.**

1. Read the ratio `solve` states for the run, and where τ_min is.
2. Take the discrete adjoint for kernels (the default). It carries the
   forward run's accuracy and nothing else, at any ratio and under either
   scheme.
3. Keep the default, `"sdirk2"`, where the ratio is well above one or
   the forcing changes abruptly: at twice the cost of a step it was more
   accurate than the trapezoid at half the step on every case measured
   here. Take `"exponential_trapezoid"` where the ratio is about one or
   below: both schemes are then converged to well under a per cent in
   the values, and the trapezoid costs half as much a step.
4. Take the continuous adjoint as a check of the time discretisation (its
   agreement with the discrete kernel says both the forward run and the
   adjoint's relaxation are resolved) or where it is wanted for itself;
   under the trapezoid only at a ratio near one or below, under Sdirk2 at
   longer steps too.
5. Settle convergence by halving the step, not by the ratio: a quantity
   has converged when each halving changes it about a quarter as much as
   the one before, its error then about a third of the last change
   (examples/configs/README.md, "Judging convergence"). The ratio says
   where to start; the halving says where to stop.
