# Developer notes

Dated records kept from sl3dpp's development: the derivations, design
choices and measurements as they were made, with the dates and the order of
the work left in. They are not maintained as reference and parts describe
code that has since changed; [design.md](../design.md) is the current
account of how sl3dpp works, and the code is the authority.

- [lateral-coupling.md](lateral-coupling.md): the pseudospectral lateral
  coupling of the 3D problem, its algebra, and the within-step convergence
  and Fortran-gate measurements.
- [adjoints.md](adjoints.md): the discrete and continuous adjoints and the
  kernels, 1D and 3D, with the design of the discrete 3D transpose (not
  built).
- [walkthrough-3d-step.md](walkthrough-3d-step.md): a map of one 3D step
  through the code, its costs, and the performance work on it.
- [multicore-scaling.md](multicore-scaling.md): the first measurements on
  a two-socket node (thread placement, the 3D step's phases by thread
  count, why `StartStep` does not scale) and the plans that follow.
