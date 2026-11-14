# The trajectory store

A trajectory store is a forward run's recording on disk: everything the
adjoint sweeps take from the run, written node by node as the run
produces it, so that the sweeps of any number of measurements run later,
in other processes, without running the forward problem again. The
library's `sl3dpp::io::TrajectoryStore` (`include/sl3dpp/io/TrajectoryStore.hpp`)
writes and reads it; `sl3d solve` records into it and `sl3d kernel`
sweeps from it (sl3d-configs.md, "The `trajectory` block"). This document
is the store's specification: its directory layout, its files byte by
byte, what the run hash must cover, and what a reader must check. Version
1 is described.

## What a store holds

For a run of N nodes (the static problem's one node, or a run through
time's nodes), recorded at stride s (every s-th node and the last are
*kept*):

- the surface state the run started from, its *background* (a calibrated
  initial sea level's, where the run calibrates);
- for every node, its *surface record*: the node's time, the sea level
  change, the displacement, the potential, the potentials of the
  tangential displacement, the angular velocity, the total-load and
  centrifugal coefficients the solid body answered, the ocean function of
  a run under migrating shorelines, and the 3D closure's diagnostics;
- for every kept node, on every process, its *state record*: the
  process's internal state at the node and, under interior sources, the
  sources' forces (and in 1D the toroidal systems' state).

That is what the sweeps of `AdjointSolver` and `AdjointSolver3D` read
from a `ForwardTrajectory` or `ForwardTrajectory3D` built from the run's
snapshots, together with what the measurements are evaluated on and the
evolution's background state.

## Directory layout

```
<directory>/
  trajectory.manifest          the manifest (the first process's)
  start.record                 the background (the first process's)
  node_000000.surface          one surface record per node
  node_000001.surface
  ...
  rank_0000/                   one subdirectory per process, by rank
    node_000000.state          one state record per kept node
    node_000002.state
    ...
  rank_0001/
    ...
```

Node numbers are written with at least six digits (`node_{:06}`), ranks
with at least four (`rank_{:04}`); a run on one process has `rank_0000`
alone. A store holds nothing else. Every file is written whole under a
temporary name in the same directory, `.<name>.partial`
(`.node_000003.surface.partial`, `rank_0000/.node_000002.state.partial`,
`.trajectory.manifest.partial`), and renamed into place, without fsync:
a killed run leaves at most partial files and no half record under a
final name.

The names above, their partials and the `rank_RRRR` subdirectories are
the store's own. A directory holding anything else is not a store, and a
writer refuses it rather than remove what it holds.

## Encodings

Every number is little-endian:

| Name | Bytes | Meaning |
| --- | --- | --- |
| `u64` | 8 | unsigned integer |
| `i64` | 8 | signed integer, two's complement |
| `f64` | 8 | IEEE 754 binary64 |
| `c128` | 16 | a complex number: its real part, then its imaginary part, each `f64` |
| `vec` | 8 + 8n | `u64` n, then n `f64` |
| `cvec` | 8 + 16n | `u64` n, then n `c128` |
| `text` | 8 + n | `u64` n, then n bytes |
| `flag` | 8 | `u64`, 0 or 1 |
| `degrees` | | `u64` count, then per entry `i64` degree and `vec` values, in increasing degree |
| `cdegrees` | | `u64` count, then per entry `i64` degree and `cvec` values, in increasing degree |

A value behind a *presence flag* is a `flag` followed, when it is 1, by
the value.

Every field is in the units the library holds it in: the sea level
solver's non-dimensional units (`GRDSolver::SolverScales`) for the times,
surface fields, angular velocity and background, and the material's
non-dimensional units for the load and tide coefficients and the states
(`Evolution::Snapshot`). The arrays are laid out as the library's types
lay them out; the types are named below with each field.

## The manifest

`trajectory.manifest`, in this order, ending exactly after its last
field:

| Field | Encoding | Meaning |
| --- | --- | --- |
| tag | 8 bytes | `SL3DTRAJ` (ASCII) |
| byte order | `u64` | 0x0102030405060708, written little-endian: the bytes 08 07 06 05 04 03 02 01 |
| version | `u64` | 1 |
| complete | `flag` | 1 once the recording finished, 0 until then |
| hash | `text` | the run hash (below) |
| writer | `text` | the program that wrote it, for provenance (`sl3dpp <version>`) |
| dimension | `u64` | 1 (`Evolution`) or 3 (`Evolution3D`) |
| times | `vec` | the N node times, increasing (one, at zero, for the static problem) |
| stride | `u64` | s >= 1 |
| kept | N `u64` | 1 where the node is kept (n mod s = 0, or n = N - 1), 0 elsewhere |
| migrating | `flag` | the shorelines migrate: every surface record holds an ocean function |
| sources | `flag` | interior sources: s = 1, every state record holds the forces and a state with its strain |
| tide | `flag` | a prescribed tide forces the run |
| field size | `u64` | F, the points of a surface field (`SurfaceGrid::FieldSize`) |
| cuts | `u64` C, then C `i64` | the radial cuts of the mesh (`Viscoelastic3D::Options::radial_cuts`); none on one process |
| parts | `u64` R, then R times three `u64` | per rank: its first segment, its count of segments (contiguous), and its grid values (`GridCoupling::GridValueCount`; 0 in 1D) |
| start bytes | `u64` | the size of `start.record` |
| surface bytes | N `u64` | the size of each `node_NNNNNN.surface` |
| state bytes | R times N `u64` | rank by rank, the size of each `rank_RRRR/node_NNNNNN.state`; 0 where the node is not kept |

The sizes are 0 until the recording is complete. A reader checks the tag,
the byte order (a big-endian file reads the mark back byte-swapped, and is
refused as such) and the version before reading further: a later version
may lay out what follows otherwise.

## Records

Every record file is a 40-byte header followed by its payload:

| Field | Encoding | Meaning |
| --- | --- | --- |
| tag | 8 bytes | `SL3DTREC` (ASCII) |
| kind | `u64` | 1 the start, 2 a surface record, 3 a state record |
| node | `u64` | the node (0 for the start) |
| rank | `u64` | the writing process's rank (0 for the start and the surface records) |
| length | `u64` | P, the payload's bytes: the file is 40 + P bytes |

### The start (`start.record`, kind 1)

The background, `GRDSolver::State`:

| Field | Encoding |
| --- | --- |
| ice thickness | `vec` (F) |
| sea level | `vec` (F) |
| ocean function | `vec` (F) |
| ocean area | `f64` |
| land held | `u64` M, then M bytes, 1 where the point is held as land (M = 0: none held) |

### A surface record (`node_NNNNNN.surface`, kind 2)

The node's snapshot beside its state (`Evolution::Snapshot`,
`Evolution3D::Snapshot`):

| Field | Encoding | Meaning |
| --- | --- | --- |
| t | `f64` | the node's time |
| converged | `flag` | the node's solves met their tolerances |
| coupled iterations | `i64` | the 3D closure's passes (0 in 1D) |
| coupled residual | `f64` | its last residual (0 in 1D) |
| solve iterations | `i64` | its conjugate-gradient iterations (0 in 1D) |
| angular velocity | 3 `f64` | x, y, z |
| relative sea level change | `vec` (F) | |
| displacement | `vec` (F) | radial |
| potential | `vec` (F) | |
| tangential potential | `vec` (F) | V |
| toroidal potential | `vec` (F or 0) | W, empty where the run has none |
| ocean function | `vec` (F or 0) | empty unless the shorelines migrate |
| load coefficients | `cvec` | the total load the body answered (`SurfaceGrid::Index` layout, m >= 0) |
| tide coefficients | `cvec` | the centrifugal and tidal potential it answered, likewise |

### A state record (`rank_RRRR/node_NNNNNN.state`, kind 3)

One per kept node and process. In 1D (`Evolution`):

| Field | Encoding | Meaning |
| --- | --- | --- |
| forces | presence flag; `degrees` spheroidal, `degrees` toroidal | the sources' forces (`RadialSourceForces`), under sources |
| toroidal state | presence flag; `f64` t, `degrees` m, `degrees` strain | the toroidal systems' state (`ToroidalViscoelastic::State`), under sources |
| state | presence flag; `f64` t, `degrees` m, `u64` count then per entry `i64` degree and `i64` columns, `degrees` strain | the internal state (`ViscoelasticState`: `m`, `columns`, `strain`; the strain empty unless recorded with it, as under sources) |

In 3D (`Evolution3D`), the process's own radial segments:

| Field | Encoding | Meaning |
| --- | --- | --- |
| forces | presence flag; `cdegrees` spheroidal, `cdegrees` toroidal | the sources' forces (`DegreeFields`), under sources |
| state | presence flag; `f64` t, `u64` K then K tensors, presence flag then a tensor (the strain), presence flag then a spectral tensor (the strain's coefficients) | the internal state (`Viscoelastic3D::State`; the strain and its coefficients absent unless recorded with them, as under sources) |

A tensor (`GridCoupling::TensorField`) is `cvec` c[0] (spin -2), `cvec`
c[1] (spin -1), `vec` r (spin 0) and `u64` S then S `vec` scalar
companions; a spectral tensor (`GridCoupling::SpectralTensor`) is `cvec`
c[0], `cvec` c[1], `cvec` r and `u64` S then S `cvec`. A 3D reader reads
the forces of every node with the snapshots (they lead the record, so the
state need not be read for them) and each state as the sweep reaches it.

## The run hash

The manifest's hash is the recorder's: opaque bytes that change with
whatever changes the forward run's numbers, so that a store is read only
by the run that recorded it. The principle: the hash covers what
determines the forward trajectory, and nothing of the reverse sweep, so
that one recorded run serves every sweep of it. It must cover the model
(the deck's and any lateral files' contents and how they are read), the
degree, the nodes (in the form that fixes their times to the bit), the
forcing (every term's numbers whole and its files' contents), the
ocean's state and its files, the shorelines, the initial sea level's
calibration, the rotation and the moments, the solves' tolerances and
budgets, the time step, the 3D step's solver and, where the forward run
steps the 3D operator (a lateral viscosity or lateral moduli), its
transforms, and the memory points of the operators. It must not cover
what only steers the run, what only a sweep reads, or what is checked
beside it: the paths themselves (a run directory may move), the record
stride and the processes' layout (the manifest states both, and a reader
checks them as their own faults), output, checkpoints, the sweeps'
measurements and methods, and on a radial run the transforms
(`solver.transform_degrees`), which there set only the grids the reverse
sweep expands its lateral kernels on (`kernels.lateral`).

`sl3d` writes `sl3d run fnv1a-64 <16 hexadecimal digits>`: FNV-1a over
the members of `apps/RunHash.cpp` (AddRunMembers, which its checkpoints
hash too, then the seismic sources, the static problem's dates of its
terms, the file inputs' formats and the memory points), the transforms'
cache left out because it does not change the numbers, and the
transforms' degrees entering on a 3D run alone.

## Writing

1. Survey the directory (below). Write only into a directory that is
   absent, empty, or holds an incomplete store with the run's hash; remove
   that store's own files first, the files before the `rank_RRRR`
   subdirectories, and nothing else. Every process surveys before any
   removes or writes.
2. The first process creates the directory and writes the manifest with
   complete = 0 and every size 0. Every process creates its `rank_RRRR`.
3. Write the start record, then, node by node as the run passes them, the
   surface records (the first process) and each kept node's state record
   (every process), each whole and renamed into place.
4. Once every record is in place, gather every process's state record
   sizes to the first, which rewrites the manifest with complete = 1 and
   every size. A recording that stops before this is recognised by its
   manifest's complete = 0.

A recording stopped with its run (a run that checkpoints and stops at a
wall-time limit) is continued by the resumed run from the node of the
checkpoint it resumes from, k:

1. Survey the directory as above: it must hold an incomplete store with
   the run's hash, recorded as the run is described (the dimension,
   times, field size, migrating, sources and tide flags, the stride and
   the processes' layout, as a reader checks a complete store).
2. Every process checks the records the run wrote to node k, each whole
   (its header's tag, kind, node, rank and length those its name gives,
   the length its file's): the first process the start and the surface
   records of nodes 0 to k, each process its own state records of the
   kept nodes among them. A store short of one is refused, nothing
   removed: a recording begun at node k would lack the nodes before it.
3. Every process has checked before any removes: the first removes the
   surface records of the nodes after k and its partial files, each
   process its own state records after k and its partial files. The
   manifest is left as it stands, incomplete.
4. The run records from node k + 1 on, the kept nodes those of the whole
   run (every s-th from node 0, whatever k is), and completes the
   manifest as above, the records to node k counted at their files'
   sizes.

A record is written whole before its node's checkpoint, so the records
to k are there whenever the checkpoint of k is; no manifest field marks
how far an incomplete recording reached, the records' presence and
headers saying it. The store a continued recording completes is the
uninterrupted run's, file by file.

The library writes the records from the evolutions' own hooks:
`Options::on_snapshot` hands it each node's snapshot (in 1D with its state
where kept), and in 3D `Options::on_state` hands it each kept state
instead of leaving it on the snapshot, so that the run holds no more than
it does without a store. A continued recording (`TrajectoryStore::
Continue`) sets the resumed evolution's `Options::record_offset` to k, so
that its stride counts the whole run's nodes, and puts each node it is
handed, the resumed call's own, at its node of the whole run.

## Reading

A reader checks, in this order, refusing each fault as its own:

1. the directory holds nothing but the store's own names, and a
   manifest;
2. the manifest's tag, byte order and version;
3. the hash equals the run's (another run's store, complete or not);
4. complete = 1 (an incomplete store is refused for reading; a recorder
   may replace it, or the resumed run continue it);
5. the dimension, times (bitwise), field size and the migrating, sources
   and tide flags equal the run's;
6. the stride equals the run's;
7. the processes' layout: the rank count, the cuts and every rank's part
   equal the run's;
8. the start, every surface record and the process's own state records
   exist at the sizes the manifest gives (a missing file and a file of
   another size are distinct faults);

and, as it reads each record, that its header's tag, kind, node, rank and
length are the ones its name and the manifest give, that every count
fits in the bytes the record has left, and that the payload is read to
its end.

## Size on disk

With L the degree, F = 2 (L + 1)^2 the points of a surface field (the
Gauss-Legendre grid of L + 1 colatitudes and 2 (L + 1) longitudes), N
nodes and stride s (k = floor((N - 1) / s) + 1 or + 2 kept nodes, the
last always kept):

- surface records: about N (8 F f + 16 (L + 1)(L + 2)) bytes, f = 4
  fields, 5 with a toroidal potential, one more under migrating
  shorelines;
- 1D states: about k * 24 K E (L + 1)(L + 2) bytes, K the relaxing
  branches and E the radial mesh's points (elements times GLL points):
  m holds K x 3 x E values per column, 2 (l + 1) columns at degree l.
  Under sources every node is kept with its strain (as large again), the
  toroidal state and strain and the sources' forces: about five and a
  half times the internal variables alone (at L = 32 on the prem8 deck,
  15.2 MB of m and as much strain, 28.6 MB of toroidal state and strain,
  24.2 MB of forces);
- 3D states: about k * 40 K G bytes over all processes, G the grid values
  of the memory nodes (about the memory nodes times F at lmax; fewer with
  `transform_degrees`), each process holding its own nodes' share: a
  memory tensor is two complex and one real value per grid value. Under
  sources each state carries its strain and its coefficients too, and
  the record the sources' forces.

The 3D states dominate: at lmax 64, F = 8450, a memory tensor takes about
340 kB per memory node, so a state of one branch on a mesh of 300 memory
nodes is about 100 MB, and a run of 81 nodes at stride 9 (ten states)
about 1 GB. Measured at lmax 8 (examples/configs/README.md): a 1D state
of the prem8 mesh, 346 kB; a 3D one, 2 MB.
