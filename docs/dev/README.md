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
  kernels, 1D and 3D, the discrete 3D transpose, the tangent-linear runs,
  the second-order adjoints, and the material derivatives and kernels.
- [walkthrough-3d-step.md](walkthrough-3d-step.md): a map of one 3D step
  through the code as of 17 October 2026, its costs, and the performance
  work on it.
- [multicore-scaling.md](multicore-scaling.md): the first measurements on
  a two-socket node (thread placement, the 3D step's phases by thread
  count, why `StartStep` does not scale) and the plans that follow.
- [pygeoinf-integration.md](pygeoinf-integration.md): a plan for posing
  and solving inverse problems through pygeoinf v2: spaces, operators,
  what sl3dpp and pygeoinf would each need, and the order of work. Of
  sl3dpp's part, the tangent-linear model and the discrete 3D transpose
  are built; the bindings and the rest are not.
- [radial-substructuring.md](radial-substructuring.md): the building of
  the radial distribution of the 3D step on 25 October 2026: the
  substructured solves, the distributed solve, the transforms cut with
  depth, the distributed step, adjoint and `sl3d`, with the measurements
  of each. The maintained account is
  [radial-partitioning.md](../radial-partitioning.md).
