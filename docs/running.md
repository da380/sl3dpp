# Running sl3dpp

How to run the `sl3d` program and the library well on a machine: the
threads and where they run, the memory a run needs, MPI ranks, and long
runs. Every config member named here is in the configuration reference,
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
`sl3d solve` and `sl3d kernel` run under `mpiexec`, the same config on
every rank:

```
OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close \
  mpiexec -n 4 -bind-to core:32 sl3d solve sea_level_3d.json
```

(the binding flags are MPICH's; another MPI names them otherwise).

- **What is distributed.** A 3D run, split over radius, one contiguous
  radial segment per rank with its threads inside: the static problem on
  lateral moduli, the runs through time, and the kernels about them, by
  either sweep. A 1D run is the first rank's alone, the other ranks
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
  sweeps for any number of measurements without running the forward
  problem again, the kept 3D states read back one at a time: the lever on
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
