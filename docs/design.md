# sl3dpp design

What sl3dpp is and how it works: the formulation, the discretisation, the
numerical choices and the rules its results are held to. This page states
the design as it stands. The derivations and convergence measurements of
[the 3D coupling](https://github.com/da380/sl3dpp/blob/main/docs/dev/lateral-coupling.md) and [the adjoints](https://github.com/da380/sl3dpp/blob/main/docs/dev/adjoints.md),
and a [code map of one 3D step](https://github.com/da380/sl3dpp/blob/main/docs/dev/walkthrough-3d-step.md), are dated
developer notes in `docs/dev`; code style and testing conventions are in
[conventions.md](conventions.md).

## 1. Scope

sl3dpp computes the response of a self-gravitating, rotating, compressible
viscoelastic Earth to surface loading, with spherically symmetric (1D)
structure or laterally varying viscosity (3D), forward and adjoint.

| Capability | Where |
| --- | --- |
| Load, tidal and traction Love numbers of transversely isotropic radial models | `LoveNumbers.hpp`, `DegreeOperator.hpp`, `ToroidalOperator.hpp` |
| The sea level equation: linear, generalised (extra channels), nonlinear (migrating shorelines), with rotational feedback, and its exact adjoint | `SeaLevel.hpp` |
| Elastic sea-level fingerprints, SI at the interface | `Fingerprint.hpp` |
| 1D viscoelastic evolution under an ice history, frozen or migrating shorelines | `Viscoelastic.hpp`, `Evolution.hpp` |
| The initial sea level calibrated against a final one (the classical update or the adjoint gradient) | `InitialSeaLevel.hpp` |
| 3D (lateral-viscosity) evolution under an ice history, frozen or migrating shorelines | `Viscoelastic3D.hpp`, `GridCoupling.hpp`, `Evolution3D.hpp` |
| Sensitivity kernels with respect to the ice history and viscosity, by the continuous adjoint or the exact discrete transpose (1D and 3D) | `Adjoint.hpp`, `Adjoint3D.hpp`, `Measurements.hpp` |
| Tangent-linear runs, 1D and 3D: the derivative of the responses along a direction in the loads, the ice thickness, the initial sea level and the inverse relaxation times (laterally varying, or a log-viscosity field, in 3D), and the second-order run along two directions | `TangentLinear.hpp`, `TangentLinear3D.hpp` |
| Second-order adjoints, 1D and 3D: the Hessian's action on a direction, as the derivatives of the kernels | `Adjoint.hpp`, `Adjoint3D.hpp` (`SecondOrderKernels`) |
| Material derivatives and kernels in κ, μ and ρ: the static problem's variation operator, Love-number derivatives and kernels (gravity self-consistent in ρ); its first-order (Born) response to lateral δμ(x), δκ(x) and the lateral kernels of a surface measurement; the time-stepped tangent-linear runs along the moduli and (1D) the density; the κ, μ and (1D) ρ kernels of the evolutions by both sweeps (discrete exact, continuous at the commute level), and their derivatives in the second-order sweeps | `Material.hpp` (`MaterialDirection`, `Variation`), `DegreeOperator.hpp` (`ApplyVariation`, `MaterialGradient`), `LoveNumbers.hpp` (`ComputeLoveNumberDerivatives`, `ComputeLoveNumberKernels`), `ElasticPerturbation3D.hpp`, `TangentLinear.hpp`, `TangentLinear3D.hpp` (`material`), `Adjoint.hpp`, `Adjoint3D.hpp` (`Options::material_kernels`) |
| The 3D evolution, its continuous adjoint and its tangent-linear runs distributed over contiguous radial segments, one per MPI rank (optional) | `Collectives.hpp`, `Distributed.hpp`, [radial-partitioning.md](radial-partitioning.md) |
| Reading planetmodel decks, model files and surface files; writing kernels | `Deck.hpp`, `io/` |
| The `sl3d` program: the common capabilities from JSON configs | `apps/`, [sl3d-configs.md](sl3d-configs.md) |

Deliberately out of scope:

- **Frequency-domain work.** Everything is in the time domain, so every
  operator is real: the degree systems are real factorisations, and complex
  coefficient columns are solved as [Re | Im] right-hand-side pairs.
  Complex moduli and complex-frequency Love numbers would need a complex
  degree operator.
- **Bulk relaxation**, and relaxation in a transversely isotropic medium
  (§14).
- **Model generation and conversion.** Radial and 3D models, tomography
  conversion and units are planetmodel's; ice histories come from pyslfp
  (`export_ice_history`). sl3dpp reads their files (§13).
- **Inversion drivers.** The kernels are the deliverable; an inversion
  loop belongs to an external framework.

## 2. Formulation

- **Elastic response.** Quasi-static, compressible, self-gravitating
  loading of a transversely isotropic radial model with fluid regions, as
  the weak form of Al-Attar & Tromp (2014) eq. (D76) for the spheroidal
  scalars (U, V, φ; φ alone in fluids, with the stratification term and
  explicit fluid–solid interface couplings), and its toroidal analogue for
  W on solid regions (Myhill & Al-Attar). The exterior is closed by the
  Dirichlet-to-Neumann term at the surface; a domain truncated above the
  centre is closed by the interior continuation at its base. The
  formulation follows pyslfp's `love_numbers`, whose `pyslfp_theory.tex`
  is the authoritative statement of signs and normalisations: g = ∂_r Φ >
  0, the sea level ξ = −u − (φ + ψ)/g + c, and the Love numbers
  normalised with the radial model's body triple (b, g, G).
- **Rheology.** Generalised Maxwell (Prony) branches in shear on an
  isotropic elastic background where any branch is active:
  T = κ tr(ε) I + 2μ∞ d + Σ_k 2μ_k (d − m_k), dm_k/dt = (d − m_k)/τ_k,
  with μ∞ = μ_U − Σ_k μ_k. A Maxwell body is one branch with μ_1 = μ_U and
  τ = η/μ_U, planetmodel's `viscosity` convention.
- **Sea level.** The gravitationally consistent sea level equation of
  pyslfp's `core.sea_level`: a fixed point on the surface load with
  Anderson mixing, exact-quadrature surface integrals, mass conservation by
  a uniform ocean shift, and rotational feedback by first-order
  quasi-static Liouville theory solved in closed form at degree 2 (axial
  and transverse, no Chandler wobble, the hydrostatic-moment guard of
  pyslfp). The evolutions use the linear equation with frozen shorelines,
  or the nonlinear (Kendall-style) solve at every step with migrating
  shorelines (`RunMigrating`, pyslfp's `_run_migrating`): the step's
  memory field enters through the response channels and the step finishes
  under the converged state's grounded-ice and ocean load. In 3D the
  nonlinear solve runs in every pass of the combined fixed point (§6). A
  state may carry a prescribed ocean function, which the linear solves use
  as given.
- **The initial sea level.** `InitialSeaLevel.hpp` lifts pyslfp's
  calibration (pyslfp 3.0.0): the initial sea level whose run reaches a
  given final one. Under frozen shorelines the problem is linear and one
  run gives it exactly, the shorelines kept as a prescribed ocean
  function; under migrating ones it iterates, by the classical update
  (Kendall et al. 2005) or steepest descent with the adjoint gradient (Yu
  et al. 2025, §3.9), Anderson-mixed.
- **Lateral viscosity.** η(r, θ, φ) = η̄(r) exp(δ(r, θ, φ)) about a radial
  reference, so the inverse relaxation time is itau(x) = itau_ref(r)
  exp(−δ(x)), positive whatever δ is.
- **Adjoints.** Yu et al. (2025): the continuous adjoint equations have the
  forward problem's form, with finite jumps of the adjoint internal
  variables at each observation; kernels in the multiplier form (§7).

## 3. Radial discretisation

- **One global mesh.** A single spectral-element mesh of Gauss–Lobatto–
  Legendre nodes (5 per element by default) serves every degree and every
  grid field. Every model boundary is an element boundary, and each span
  is divided uniformly into elements no wider than 0.1 a/(lmax + 1)
  (planetmodel's rule). `RadialMesh::CreateGraded` (pyslfp's
  `graded_mesh`) widens the elements with depth instead, each degree
  keeping the uniform rule's resolution over its own subset; the
  benchmarks (`step_timing`, `fortran_gate`, `substructure_timing`,
  `distributed_*`) use it, the `sl3d` program and the examples the uniform
  rule. Nodal arrays are stored per element, so a shared node carries one
  value from each side.
- **Degree truncation is a subset, not a sub-mesh.** A degree-l solve
  acts on the elements above its truncation radius, snapped to an element
  boundary. The pseudospectral 3D coupling needs every degree's
  coefficients on the same radial nodes, which per-degree meshes would
  break. Two truncation rules are offered: `TruncationRule::Graded`,
  r_l = a ε^(1/(l+1)) (pyslfp's, the default) and `TruncationRule::Legacy`,
  r_l = a (1 − 20/(l + 0.5)) (the old Fortran's).
- **Low degrees.** At l = 1 the surface potential dof is removed (the
  centre-of-mass frame, §8); at l = 0 there is no tangential dof, fluids
  are μ = 0 elastic media, and U(0) = 0 at a centre.
- **Factorisation.** Each degree system is symmetric, stored as an upper
  band, scaled symmetrically by its diagonal and factorised once: banded
  Cholesky, falling back to banded LU where the matrix is not positive
  definite (the semi-definite fluid limit). The toroidal blocks, pinned at
  l = 1 (§8), are SPD and use banded Cholesky. Factorisations are cached
  per (scheme, step size) and reused across a run. With radial cuts
  (`radial_cuts`, `RadialMesh::BalancedCuts`) every degree system is solved
  by substructuring instead (`detail/BandSubstructure.hpp`): each segment
  of the mesh eliminates its interior on its own and the cut nodes' dofs
  solve together. The answers are the same, with the direct solve's
  residuals. This is the per-degree algebra of a distributed solve (§12).
- **Non-gravitating variant.** `DegreeOperator` with `gravitating = false`
  drops the gravity and potential terms (U, V alone) for the
  non-gravitating tier of the cross-code benchmarks, mirroring pyslfp's
  `traction_numbers`.

## 4. Angular discretisation and storage

- **Basis.** GSHTrans's generalised spherical harmonics throughout:
  complex, fully normalised, Condon–Shortley phase,
  Y^N_lm = √((2l+1)/4π) d^l_{Nm} e^{imφ}, forward transform ∫ conj(Y) f dΩ
  — the old Fortran's basis. The real orthonormal pyshtools convention of
  pyslfp and planetmodel exists only in the converters of
  `ShConventions.hpp`, which the rotational feedback's degree-2, order-1
  terms and the expansion-coefficient measurements use.
- **Grids.** Fields live on Gauss–Legendre grids (`SurfaceGrid`, a cheap
  handle to one immutable GSHTrans grid, shared where the resolution is
  the same). The sea level grid is band-exact; the 3D coupling's grid
  carries spins up to 2 and is oversampled by 3/2 so that the pointwise
  products are dealiased (the Fortran never dealiased). With
  `transform_degrees` the memory nodes fall into blocks by depth, each on
  a grid of its own degree, just covering the degrees its solution
  carries under the truncation ([radial-partitioning.md](radial-partitioning.md) §6).
- **Reality reduction.** The strain and memory tensors are real, so of the
  five canonical trace-free components (spins −2..2) only spins −2 and −1
  are stored, as complex fields, and spin 0 as a real field; spins +1 and
  +2 follow from A^{(N)}_{lm} = (−1)^m conj(A^{(−N)}_{l,−m}) where needed.
- **m ≥ 0 columns.** At the degree boundary each degree carries l + 1
  coefficient columns indexed by m ≥ 0 (m = 0 real). Any pairing of two
  such column sets weights m > 0 twice — (1, 2, 2, ...) — against a real
  dof pairing; transposes halve or double accordingly.
- **Batching.** Transforms are issued once per spin component and node
  block with the block's radial memory nodes as the batch, threaded by
  GSHTrans; its BLAS matrix kernel is an option.

## 5. Time stepping

- **State.** The evolving state is the internal variables m_k alone; the
  displacement is slaved through one elastic solve, as in pyslfp and
  AdGIA.
- **Schemes.** The exponential trapezoid (the default; second order, no
  stability limit, one solve per step) with h = Δt itau, e = e^{−h},
  φ₁ = (1 − e)/h, α = φ₁ − e, β = 1 − φ₁, and m⁺ = e m + α dⁿ + β dⁿ⁺¹;
  backward Euler and exponential Euler as alternatives. Step sizes are not
  limited by the shortest Maxwell time. `ViscoelasticOperator` also offers
  an adaptive step controller (embedded exponential Euler); the evolution
  drivers step on the caller's time grid.
- **The step.** The implicit step eliminates m⁺ into equilibrium, giving
  an effective operator per (scheme, Δt), forced by the load channels and
  the memory force of m̃ = e m + α dⁿ, finished through cached unit load
  and tide responses. An unrelaxed elastic solve at the start time gives
  the initial strain.
- **Memory points.** The internal variables live at the Gauss points of
  each radial element by default (`MemoryPointSet::Gauss`), the GLL nodes
  as an option (`MemoryPointSet::Gll`, the Fortran's and pyslfp's choice,
  for direct comparison). The N-point Gauss rule matches the (N+1)-point
  GLL rule's exactness for the memory-force integral and stores 80% of the
  memory. Everything that lives with m — itau, the step weights, their
  h-derivatives, the kernel accumulators — lives at the same points.
- **The shear split.** The energy is split as the continuum does,
  ∫ μ∞|ε|² + Σ_k ∫ μ_k |ε − m_k|²: the long-term part assembled at the GLL
  nodes, each branch's at the memory points. The effective operator is
  then K(μ∞) + Σ_k Dᵀ W (μ_k w_k) D with w_k = 1 − β_k, positive term by
  term for any step (a `PointShear` added to the long-term material).
- **Memory windows.** In 1D each degree's internal variables span only its
  memory window: its elements from the first to the last with a relaxing
  node.
- **Restart.** Both evolutions hand out checkpoints (`Checkpoint`,
  `on_checkpoint`) and resume from them (`Resume`); a resumed run on one
  thread is bitwise the uninterrupted one.

## 6. The 3D coupling

- **The split.** The effective operator with lateral viscosity couples all
  (l, m). It is split into the radial reference, factorised per degree as
  in 1D, and a lateral remainder applied pseudospectrally on the right-hand
  side: synthesise the strain to the grid, multiply pointwise by the weight
  field W(x) = Σ_k μ_k (β_k(x) − β_k,ref(r)), analyse, and assemble
  through the memory-force rows. The step weights come from h(x) =
  Δt itau(x) pointwise. Toroidal motion is driven through the same rows,
  for l ≥ 1.
- **The within-step iteration.** The iterate is the strain's coefficients.
  Each pass is one banded back-solve per degree plus the transforms. A
  step starts as Picard and switches to Anderson mixing (depth 5) for the
  rest of the step once a pass shrinks the residual by less than half
  (`IterationControl::anderson_rate`); defaults: relative tolerance 1e-11,
  at most 40 passes. A non-finite pass ends the iteration unconverged.
- **With sea level.** `SeaLevelEvolution3D` folds the ocean load and
  rotation into one combined fixed point per step (`coupled_rtol`,
  `max_coupled_iterations`); a node that does not converge is reported in
  its snapshot. Under migrating shorelines (`RunMigrating`) each pass
  solves the nonlinear sea level equation from the background with the
  pass's memory and lateral responses, so the ocean function follows
  flotation inside the fixed point, and the load channel carries the
  converged state's load relative to the background. With no lateral
  perturbation this is the 1D migrating run, shorelines included.
- **Envelope.** Measured on a smooth degree-1 pattern
  ([dev/lateral-coupling.md](https://github.com/da380/sl3dpp/blob/main/docs/dev/lateral-coupling.md) §5a): peak-to-peak contrasts
  up to 1000x converge at Δt/τ up to 2 (in about 150 passes at the hard
  end); at Δt/τ = 10, 10x converges and 100x does not within 200 passes.
  The pass counts do not depend on lmax, so the hard cases are hard in the
  continuum, not by truncation. Fields with sharp lateral edges alias in
  the pointwise products and need a higher lmax.
- **The radial reference.** Whatever radial viscosity the rheology
  carries. Read from a planetmodel model file (`io::ModelViscosity`), it is
  by default the lateral geometric mean at each radius, so δ has zero
  lateral mean; the file's own radial reference is the alternative.
- **Memory.** One internal-variable tensor (one branch, every memory node)
  is about 0.7 GB at lmax 64, 4 GB at 128, 21 GB at 256 and 98 GB at 512,
  and a step holds several; README, *Memory*, has the levers.

## 7. Adjoints and kernels

- **Two adjoints, one interface.** `AdjointSolver` offers the continuous
  adjoint (the default and preferred path) and the discrete adjoint (the
  exact transpose of the time-stepped forward map), with one result type.
  The discrete gradient is exact for the discretised problem; the two
  differ by the discretise–transpose commute error, which is large exactly
  when the forward run is under-resolved, so their disagreement is a
  resolution diagnostic. `AdjointSolver3D` offers both too: the
  continuous adjoint (the default) and the discrete transpose of the
  coupled lateral-viscosity map — the transposed within-step fixed point
  solved by the same iteration, with the force assembly as the transpose
  of the strain evaluation under the energy pairing; exact against the 3D
  tangent-linear run at 1e-8 and second order to solver precision in the
  Taylor test. The discrete 3D sweep is single-process.
- **Measurements.** Linear functionals of one node's outputs,
  J = a² ∫ g ξ dΩ + a² ∫ g_u u dΩ + a² ∫ g_φ φ dΩ + g_ω·ω, keyed by node
  index into the time grid. `Measurements.hpp` builds the duals: cap means,
  point values, expansion coefficients, rotation components, and the decay
  time of a site's series (Al-Attar & Tromp 2014, Appendix C) as per-node
  weights.
- **Ice-history kernels.** `IceKernels` carries pyslfp's three views: the
  discrete view (duals of the load at each node), the distributional view
  (a density in time plus the instantaneous amplitude of each
  measurement; Yu et al. 2025 §3.7), and the quelled view (the kernel
  against the thickness rate on each interval).
- **About a migrating trajectory.** Given the ocean function of every node
  of a migrating run, the sweeps (1D and 3D) linearise about it: the adjoint stays
  linear with the time-reversed C(t) as its coefficients (Yu et al. 2025,
  §3.3; the shoreline-motion singular terms cancel), the grounding factors
  are per node, and the initial sea level kernel
  ρ_w Σₙ (Cₙ − C₀) loadDualsₙ is formed (zero under frozen shorelines).
- **Viscosity kernels**, in the multiplier form of Yu et al. (2025), as
  pyslfp: in 1D, dJ/d itau = ∫ W_k m†/itau (d − m) dt per branch, mapped to
  dJ/dη by dη = −η² d(itau)/μ; in 3D, μ_k m†:(d − m)/itau per branch and
  memory point, with the contraction on the reduced storage
  r² [12 A₀B₀ + 8 Re(A₋₁B̄₋₁) + 4 Re(A₋₂B̄₋₂)] — the metric fixed by
  requiring the angle-uniform limit to reproduce the 1D force weights
  (4/3, k², k²(k² − 2)). The 3D results are the itau kernel on (memory
  node × grid), its spherical reduction, and the log-viscosity kernel
  −itau(x) K.
- **Tangent-linear runs** (`TangentLinearSolver`, the forward mode,
  pyslfp's `tangent_linear_ice_history`): the derivative of a run's
  responses along a direction in the direct load, the ice thickness, the
  initial sea level and the inverse relaxation times, about a recorded
  trajectory (frozen, or a migrating run's ocean functions held). It is
  the forward loop once more — the shorelines prescribed, the direct load
  the perturbation's grounded by each node's C relative to the first's,
  ρ_w (Cₙ − C₀) δS₀ for the initial sea level — with each opened step's
  memory state shifted by a drive bilinear in the trajectory and δitau:
  the scheme weights' h-derivatives (Discrete; the exact transpose of the
  discrete sweep, gated by the pairing identity at 1e-8) or the continuous
  q_k = (d − m_k) δitau_k/itau_k discretised by the scheme (Continuous;
  the transpose of the continuous sweep, converging to the discrete at
  the scheme's order). The forward trajectory is served forwards from
  its checkpoints by exact forward steps, which amplify nothing. The
  returned snapshots can record the tangent-linear state, which the
  second-order run (`RunSecondOrder`, F''(x)[u, v]: the same loop with no
  direct load and the drive bilinear in the two tangent trajectories and
  the forward one, plus the weights' second derivatives by the discrete
  method) reads. Under the sharp flotation rule the loads and the initial
  sea level enter no second derivative. The directions may have a part in
  the moduli (`SecondOrderDirection::materialU/V`): each direction's
  material forces on the other's tangent trajectory, its full solution
  recovered from the recorded snapshot and its decayed memory with its
  own itau drive (docs/dev/adjoints.md §15).
- **3D tangent-linear runs** (`TangentLinearSolver3D`): the same
  derivative of the coupled lateral-viscosity evolution, along loads,
  thickness, initial sea level and the laterally varying itau(x) (or a
  log-viscosity field), and the second-order run; the 1D drives applied
  pointwise on the memory grid through `OpenStep::AddMemory`, the
  within-step fixed point unchanged. The discrete run is the exact
  derivative of the coupled forward (gated against central differences
  with Richardson to 1e-6); the continuous run is the transpose of the
  continuous 3D sweep to the commute error. The second-order run takes
  material directions too (`SecondOrderDirection3D::materialU/V`), with
  the base step's lateral weights in the effective shear.
- **Second-order adjoints** (`AdjointSolver::SecondOrderKernels`, pyslfp's
  `second_order_adjoint_ice_history`): the kernels and their derivative
  along a direction v, the Hessian's action on v, as `KernelDerivatives`
  (the thickness and initial-sea-level views' derivatives and the itau
  kernel's). Given v's recorded tangent-linear run, the sweep carries a
  second dual state forced by the first sweep's and δitau. Discrete: the
  exact derivative of the exact gradient, the Hessian symmetric to
  rounding. Continuous: the second-order adjoint equations with the ocean
  functions held, the derivative state stepped with the continuous drive
  of the adjoint trajectory. A Hessian-vector product costs the forward
  run, v's tangent-linear run and the second-order sweep. With
  `Options::material_kernels` the material kernels and their derivatives
  along v are carried too (`KernelDerivatives::kappaKernel/muKernel`), and
  v may have a part in the moduli (`SecondOrderKernels(..., material)`),
  by both methods — the derivative of the effective systems on the first
  duals as a source before the unit-response contraction, and the memory
  force's variation on the first dual strains; in the continuous sweep v's
  material forces on the adjoint trajectory, the jump's fingerprint moving
  with the unrelaxed table (docs/dev/adjoints.md §15).
- **3D second-order adjoints** (`AdjointSolver3D::SecondOrderKernels`):
  the continuous sweep carrying the derivative adjoint state along v,
  stepped with the continuous drive of the adjoint trajectory through the
  same `AddMemory`; returns `KernelDerivatives3D` (the views' derivatives,
  the itau kernel's derivative and its spherical reduction, and the
  log-viscosity kernel's derivative, the Hessian-vector product of the
  log parameterisation). Both methods: the discrete sweep carries the
  derivative dual state through the same transposed fixed point, forced
  by the derivative of the step's lateral weight on the first dual
  strain, the Hessian exact to the solvers' tolerance
  (docs/dev/adjoints.md §14). Material directions and the material
  kernels' derivatives as in 1D, by both methods, single-process
  (docs/dev/adjoints.md §15).
- **Material derivatives** (`MaterialDirection`: δκ, δμ, δρ on the mesh's
  nodes). `Material::Variation` is the exact linearisation of
  `Material::Shifted`, gravity's variation integrated from δρ so that the
  background gravity, the stratification and the interface terms move
  with it; `DegreeOperator::ApplyVariation` applies the degree system's
  variation and `ComputeLoveNumberDerivatives` gives the Love numbers'.
  The time-stepped tangent-linear runs take directions in κ and μ
  (`TangentPerturbation::material`; `TangentPerturbation3D::material` on
  the memory nodes' grid values), the branch moduli fixed fractions of
  the unrelaxed modulus and the relaxation times held: the variations of
  each step's effective system and memory force enter as forces on the
  step (`AddForce`), and the start's unrelaxed response to the variation
  seeds the start. Density directions are refused there, and lateral δρ
  is excluded in 3D.
- **The lateral Born response** (`ElasticPerturbation3D`): the first-order
  response of the static elastic problem to lateral δμ(x) and δκ(x),
  applied through the grid pipeline as the lateral viscosity is (δμ on
  the deviatoric strain, δκ on the dilatation) and solved on the radial
  reference systems; it couples degrees and orders by the products'
  selection rules, and spheroidal to toroidal motion.
- **Material kernels** (`Options::material_kernels`): dJ/dκ and dJ/dμ,
  nodal in 1D (`IceKernels::kappaKernel`, `muKernel`) and on the
  coupling's grid in 3D with spherical reductions
  (`Kernels3D::kappaKernel3D`, `muKernel3D`). By the discrete sweeps they
  are the exact transposes of the material tangents (in 1D through the
  degree form's nodal gradient, `DegreeOperator::MaterialGradient`; in 3D
  through the grid strains' energy contraction and the dilatations); by
  the continuous ones, the time quadrature of the adjoint displacement's
  pairing with the forward solution and memory plus lumped pairings at
  the observations, at the commute level. They need the recorded
  trajectory, the memory points at the GLL nodes and no rotational
  feedback. The static kernels: `ComputeLoveNumberKernels` (the Love
  numbers' in κ(r), μ(r) per channel and degree) and
  `ElasticPerturbation3D::StaticKernels` with `SolveSurfaceDual` (the
  lateral kernels of a static surface measurement).
- **Forward recordings.** A forward run keeps every stride-th node's
  internal variables (`record_stride`) and every node's loads; the
  adjoints regenerate a stored node's strain by one elastic solve
  (`record_strain` keeps it instead). In 1D the sweep rebuilds the gaps by
  exact backward steps (`StepBack`), whose round-off grows like exp(Σh)
  across a gap. In 3D it rebuilds them by forward steps from the kept
  states, holding N/stride + stride states, with no stability limit on the
  stride; the recorded states can be handed to a store as they are made
  (`on_state`) and fetched back during the sweep.

## 8. Degree one

- **Spheroidal translations** are fixed by the centre-of-mass frame: the
  surface degree-1 potential dof is removed, so the centre of mass of Earth
  and load stays at the origin. The frame is physical, so the degree-1
  radial displacement, potential and sea level the code computes —
  relative to the centre of mass — are well defined.
- **Toroidal rotations.** Lateral viscosity drives degree-1 toroidal
  motion, so the toroidal block runs from l = 1. Its quasi-static operator
  is singular: rigid rotation (W ∝ r) carries no strain, one null vector
  per order in each solid region that fluid separates (mantle, inner
  core). The right-hand side is torque-free, so the system is consistent,
  and one W dof per solid region is pinned (the region's top node by
  default, `pin_fraction`). The pinned matrix is SPD and the pinned solve
  solves the full system; moving the pin changes only the rigid rotation.
- **The rule: don't measure gauge.** Measured, benchmarked and tested
  quantities must be invariant under the toroidal rigid rotations; this is
  also the solvability condition of the adjoint equations. A horizontal
  displacement at a point is gauge; a change in distance between sites is
  not. Sea level, radial displacement, potential and rotation do not see
  the rigid rotations, so every current measurement type qualifies; a
  horizontal measurement type would have to be built rotation-invariant.

## 9. Units and non-dimensionalisation

- **SI in, SI out, non-dimensional inside.** Readers and SI construction
  options non-dimensionalise on entry; returns and writers
  re-dimensionalise on exit; components exchange non-dimensional numbers
  (`EarthModel::Dimensionless`). `Fingerprint`, `LoveNumbers` and the `io/`
  classes are SI at their interfaces; the evolution and adjoint drivers
  work in the sea level solver's units (`SeaLevelSolver::SolverScales()`).
- **The scheme** is planetmodel's `geophysical`: length the body's surface
  radius, density the Earth's mean density ρ̄, time fixed by G ρ̄ T² = 1
  (about 1648 s), so G = 1 internally. `Scales.hpp` mirrors planetmodel's
  units, and `UnitsParity` holds the constants and the geophysical factors
  equal to planetmodel's exactly.
- **Across codes**, non-dimensional numbers are never compared directly:
  pyslfp's internal scheme differs (T ≈ 805 s; the non-dimensional
  potential scales as T²), so comparisons anchor in SI, channel by channel
  (§10).

## 10. Comparison rules for cross-code work

These hold for the fixture gates here and for the cross-code benchmarks
(AdGIA's benchmark hierarchy among them).

1. **Coefficient space or grid-independent functionals.** Fields are
   compared through their expansion coefficients (via the documented basis
   conversion) or through functionals that do not depend on the grid — the
   rotation vector, mean sea level, surface integrals, cap means — never by
   matching grids point for point.
2. **Band-limited inputs for tight gates.** Tight comparisons use smooth,
   band-limited inputs, including a smooth ocean-function override in the
   sea level equation, so that results are grid-independent to spectral
   accuracy. The sharp flotation ocean function is compared separately, at
   the tolerance its coastline quadrature permits: pyslfp's own spread
   between its grids at lmax 32 is 0.44% in ocean area, 0.049 of the
   largest sea-level coefficient, and a factor of about 2 in the
   ill-determined transverse rotation.
3. **Gauge-invariant quantities only** (§8). Degree 1 is compared only in
   a shared, physical frame; against the Fortran sl3d, whose degree-1
   frame is unsettled, degree 1 is excluded.
4. **Anchor in SI**, channel by channel (§9).
5. **References.** pyslfp is the reference for the elastic problem; the
   Fortran sl3d serves the time-dependent comparisons, at tolerances judged
   per case. Elastic Love numbers from an updated Fortran build sit about
   1% off pyslfp, a known and unexplained discrepancy.
6. **Fixtures.** Reference values are produced by the Python scripts in
   `scripts/` (driving pyslfp and planetmodel) and committed to
   `tests/data/` with a provenance header; the tests read them and need no
   Python.

## 11. Validation

The suite runs mostly at small degree (lmax ≤ 8; some fixture gates and
the substructuring and distribution tests go to 16–64) and few steps, each
test on one thread: the gates are identities, convergence orders and
fixture comparisons, which small problems pin as tightly as large ones.

| Gate | Compared against | Fixture | Tolerance |
| --- | --- | --- | --- |
| `GravityGate` | planetmodel's mass, moment of inertia and g(r) | `planetmodel_gravity.txt` | 1e-13 relative |
| `LoveNumbersGate` | pyslfp's generalised Love numbers | `pyslfp_love_numbers.txt` | 1e-10 relative |
| `SeaLevelGate.GeneralisedSolveMatchesPyslfp` | pyslfp, band-limited generalised solve | `pyslfp_generalised_sea_level.txt` | 5e-9 in coefficients |
| `SeaLevelGate.SharpOceanFingerprintNearPyslfp` | pyslfp, sharp flotation ocean | `pyslfp_fingerprint.txt` | 8e-2 in coefficients, 1% ocean area |
| `EvolutionGate.StepperMatchesPyslfp` | pyslfp's 1D stepper, Love-number histories | `pyslfp_stepper.txt` | 1e-9 |
| `EvolutionGate.CoupledEvolutionMatchesPyslfp` | pyslfp's coupled 1D GIA run | `pyslfp_evolution.txt` | 1e-5 |
| `MigratingGate.RunMisfitAndCalibrationMatchPyslfp` | pyslfp's migrating-shoreline run, the gradient of a final sea level misfit, and both calibrations' misfit histories | `pyslfp_migrating.txt` | ocean functions equal; 1e-9 (run), 1e-10 (gradient), 1e-7 (histories) |
| `AdjointGate` | pyslfp's ice-history load duals (both adjoints) and itau kernels | `pyslfp_kernels.txt` | 1e-5 |
| `E1Gate.TractionNumbersMatchPyslfp` | pyslfp's traction numbers, gravitating and not | `e1_traction_numbers.txt` | 1e-10 |
| `DecayTime.MatchesPyslfpOnTheFixture` | pyslfp's decay-time estimator | `decay_time.txt` | 1e-6 relative |
| `ModelViscosity` | planetmodel's Python reading of model files | `viscosity_*.nc`, `viscosity_probes_*.txt` | 1e-11 |
| `LateralGate` | the 1D path, with zero lateral variation | — | 1e-11 (pipeline), 1e-10 (step), 1e-8 (coupled evolution) |

Beyond the fixtures:

- **3D.** `Lateral` checks second-order self-convergence of the lateral
  step, threaded against serial runs, the degree-1 toroidal block (torque-
  free force, pin independence), and resumption.
- **Tangent-linear runs.** Linear in the loads (the run along a load
  perturbation equals the difference of two forward runs at 1e-8); the
  discrete run and the discrete adjoint exact transposes (the pairing
  identity with the load duals and the itau kernel at 1e-8, node 0 and a
  checkpointed trajectory included); the continuous drive converging to
  the discrete at the scheme's order; about a migrating trajectory, the
  run along thickness and initial sea level perturbations that flip no
  cell equal to central differences at 1e-7; the weights' second
  h-derivatives against differences of the first; the second-order run
  as the derivative of the first-order one (central differences at
  second order, Richardson to 1e-6), symmetric in its directions, zero
  without a viscosity perturbation, and continuous converging to
  discrete at the scheme's order.
- **3D tangent-linear runs.** Zero-lateral equal to the 1D run at 1e-7;
  linear in the loads with lateral structure at 1e-8; the discrete run
  along a log-viscosity direction against central differences of the
  coupled forward at second order with Richardson to 1e-6, and the
  continuous run against the continuous 3D kernel's pairing at the
  commute level; the second-order run as the derivative of the first
  with the log parameterisation's second variation, symmetric.
- **Second-order adjoints.** The carried kernels equal to the first-order
  sweep's at 1e-12; central differences of the gradient along v against
  the derivatives at second order with Richardson to 1e-6, both schemes;
  the Hessian symmetric at 1e-9 and adjoint to the second-order
  tangent-linear run at 1e-9, frozen and migrating; the continuous
  derivatives converging to the discrete block by block.
- **3D second-order adjoints.** Zero-lateral equal to the 1D continuous
  second-order sweep at 1e-6 (kernels and derivatives); with lateral
  structure, the carried kernels equal to the first-order sweep's at
  1e-10, and the Hessian's symmetry, its adjointness to the 3D
  second-order run and its agreement with central differences of the
  continuous kernels all at the commute level, improving under
  refinement.
- **Adjoints.** The discrete gradient identity to 1e-8 under both schemes;
  finite differences with Richardson extrapolation of the itau kernels;
  the continuous adjoint converging to the discrete under refinement;
  checkpointed recordings reproducing full ones; in 3D, the zero-lateral
  sweep equal to the 1D one at 1e-6, the log-viscosity kernel against
  finite differences at the commute level, and a Taylor test of the
  gradient remainder.
- **The discrete 3D transpose.** The pairing identity with the 3D
  tangent-linear run at 1e-8 (both schemes, loads and a log-viscosity
  direction, stride-2 trajectories); zero-lateral equal to the 1D
  discrete sweep at 1e-8; the machine-grade Taylor test (second order at
  every step, Richardson to 1e-7); the continuous gap closing under
  refinement.
- **Material derivatives (static).** Gravity's variation exact for
  polynomial density directions; the variation operator against central
  differences of the assembled system at 1e-9; the Love-number
  derivatives against central differences of the shifted material's
  numbers, Richardson to 1e-7 for every channel, degree and axial number
  under μ, κ and ρ directions; the reciprocity identities differentiated.
- **Lateral elastic perturbation (Born).** Angle-uniform directions
  reproducing the 1D Love-number derivatives at 1e-8 through the grid
  pipeline; the perturbation force symmetric at 1e-10; the selection
  rules of a single-harmonic direction, spheroidal and toroidal.
- **Material tangent, 1D.** The time-stepped tangent-linear along a
  direction in κ and μ (branch moduli as fixed fractions, times held)
  against central differences on the shifted material, both schemes,
  second order with Richardson at 1e-6.
- **Material tangent, 3D.** Angle-uniform directions in κ and μ with
  lateral viscosity on against central differences on the shifted
  material (both schemes); the zero-lateral bridge to the 1D material
  tangent at 1e-7.
- **Material kernels, 1D.** The nodal material gradient against the
  variation operator at 1e-12 for every degree; the discrete κ and μ
  kernels paired with a direction against the material tangent at 1e-10
  and against central differences on the shifted material, both
  schemes.
- **Material kernels, 3D.** The discrete κ(x) and μ(x) kernels paired
  with lateral directions against the 3D material tangent at 1e-8 with
  lateral viscosity on, both schemes; the zero-lateral bridge to the 1D
  material kernels at 1e-7.
- **Continuous material kernels, 1D and 3D.** Against the exact
  discrete ones on one problem under refinement: a per-cent gap at
  2.5 kyr steps falling at the schemes' orders, both schemes.
- **Static material kernels.** The Love-number kernels in κ and μ
  against the Love-number derivatives at 1e-10 for every channel and
  degree; the 3D static kernels as the transpose of the Born force at
  1e-10.
- **Writers and units.** Kernel files round-trip through planetmodel's
  reader at 1e-11; `UnitsParity` holds the constants equal.
- **External.** The Fortran sl3d's `visco_test` with lateral viscosity
  on: `benchmarks/fortran_gate` with `scripts/run_fortran_gate.sh`, the
  lmax-32 reference committed as `tests/data/fortran_visco_test.txt`. The
  two codes — explicit rate-form stepping on a uniform mesh against
  implicit stepping on the graded mesh — agree at 3–13e-4 relative in the
  surface U and φ coefficients over 5 kyr, for every degree l ≥ 2.

## 12. Parallelism

- **Shared memory, OpenMP.** The target is a large shared-memory node
  (around a hundred cores and the memory of §6). Threaded by default on
  OpenMP's thread count (`OMP_NUM_THREADS`); `parallel = false` options
  force a component serial for a caller already inside a parallel region.
  The parallel axes: degree loops (independent factorisations and solves),
  element and latitude loops for pointwise grid work, and GSHTrans's
  threaded transforms. The Anderson mixer threads its sweeps with
  per-thread partial sums added in thread order.
- **No BLAS inside the threads.** The banded factorisations and solves
  are the library's own (`detail/BandSolvers.hpp`, LAPACK's unblocked
  algorithms with its storage and pivoting), because a BLAS called from
  every thread of a degree loop at once fails on many-core nodes; README,
  *Running*, has the detail.
- **Reproducibility.** One-thread runs are bitwise reproducible; threaded
  runs agree with them to round-off.
- **Distributed over radius (MPI, optional).** A production 3D problem
  (lmax 256) fits one large node, but a step is fastest on one socket, and
  larger problems want the memory of several nodes. With the CMake option
  `SL3DPP_WITH_MPI` (which defines `SL3DPP_HAVE_MPI` and guards the MPI
  sections) the 3D step runs over contiguous radial segments, one per
  rank, with OpenMP threads inside each
  ([radial-partitioning.md](radial-partitioning.md); the measurements in
  [dev/radial-substructuring.md](https://github.com/da380/sl3dpp/blob/main/docs/dev/radial-substructuring.md)).
  Every large array is indexed by radial node and the pseudospectral
  products are pointwise within a node, so each rank holds and transforms
  its own segment's memory nodes alone, and no global transposes are
  needed; the cuts balance the transform work
  (`GridCoupling::TransformWeights`). The degree systems are substructured
  over the segments: each rank solves its own, and the cut nodes' parts of
  every degree's system are summed in one collective per pass. The
  library calls no MPI itself: `Viscoelastic3D` takes `Collectives`
  (sum, max and share callbacks), which `Distributed.hpp` fills from a
  communicator. The sea level equation is solved on every rank alike and
  the first rank's solution shared. The continuous 3D adjoint and the 3D
  tangent-linear runs run the same way, each rank sweeping its own nodes
  over its own trajectory; the discrete 3D sweep, the material tangent and
  the material kernels are single-process. `sl3d`'s `evolve` and `kernel`
  run under `mpiexec` with per-rank checkpoints
  ([sl3d-configs.md](sl3d-configs.md)).

## 13. Dependencies and boundaries

| Library | Used for |
| --- | --- |
| GSHTrans | grids, generalised spherical harmonic transforms, Wigner functions |
| GaussQuad | Gauss and Gauss–Lobatto quadrature, Legendre polynomials |
| Interpolation | the per-layer splines of `EarthModel` |
| FFTWpp, NumericConcepts | through GSHTrans |
| planetmodel | decks, model files, surface files, units, JSON configs |
| LAPACK, BLAS | small serial solves: the decay fit and Anderson mixing |
| MPI (optional, `SL3DPP_WITH_MPI`) | the collectives of the radial distribution (`Distributed.hpp`) and `sl3d` over ranks |
| netCDF-C | through planetmodel |

- **Pins.** The sibling libraries are pinned to exact commits
  (`SL3DPP_<NAME>_PIN` in `CMakeLists.txt`), so every commit of sl3dpp
  builds against the same sources; `scripts/bump_dependencies.py` moves the
  pins, and a weekly CI workflow builds against the libraries' current
  `main`. The libraries themselves track each other's `main` between
  releases.
- **Internal representation is sl3dpp's own.** The model, mesh, operators
  and fields are designed for this code (GLL nodal fields, degree columns,
  spectral layouts). File formats are a boundary that the readers and
  writers translate across.
- **The `io/` classes** keep planetmodel's types out of the library's own
  headers: `io::ModelViscosity` reads a planetmodel model file into a
  Maxwell rheology and a δ functor; `io::WriteKernels` writes 3D kernels
  as a model file (per unit volume, SI); `io::FieldHistory` reads surface
  files onto the run's grid. `Deck` delegates parsing and validation to
  planetmodel's deck reader; `Deck::ReadLegacySl3d` reads the old
  `prem.simp.vm1` format for comparison with the Fortran. planetmodel is
  linked publicly, so a code using sl3dpp can read and write its files
  beside it.
- **The program.** `apps/` holds the `sl3d` program: strict JSON configs
  read with planetmodel's reader, one subcommand per capability, outputs
  with provenance, and disk checkpoints for the evolutions.

## 14. Known limitations

What is not built, or holds only within limits:

- **The discrete 3D sweeps.** The discrete 3D adjoint, first and second
  order, is single-process (refused when distributed); a material
  direction of either 3D second-order sweep is single-process too.
- **Strong lateral contrasts at long steps.** The within-step iteration's
  envelope (§6); a better reference profile (per degree, or worst case) or
  an implicit–explicit split would extend it, and neither is built.
- **Shoreline flips inside the 3D fixed point.** Under migrating
  shorelines the 3D combined fixed point re-derives the ocean function in
  every pass, so a cell sitting at the flotation threshold can flip between
  passes and hold the residual above a tight `coupled_rtol`; such a node is
  reported unconverged rather than accepted. The tests' marine caps converge
  at 1e-12. The 3D gradient calibration uses the continuous adjoint by
  default (`CalibrationOptions::adjoint3D`; the discrete sweep serves
  single-process), whose kernel is exact only as the step goes to zero;
  the iteration's fixed point is still the calibrated state, since the
  kernel is linear in the residual.
- **Nonlinear (stress-dependent) rheologies.** Not built; they would fold
  into the same within-step fixed point
  ([dev/lateral-coupling.md](https://github.com/da380/sl3dpp/blob/main/docs/dev/lateral-coupling.md) §7a).
- **Transversely isotropic relaxation.** The elastic structure may be
  transversely isotropic, but the material must be isotropic wherever a
  branch relaxes. What a relaxed TI medium is — the relaxed bulk response,
  and whether moduli relax on different time-scales — is open.
- **Bulk relaxation** is not modelled.
- **Second-order runs and adjoints** take no density direction (the
  form's second variation in the density is not built), and the
  shoreline terms of pyslfp's smoothed flotation rule are not ported
  (sl3dpp has the sharp rule only).
- **Material derivatives.** The time-stepped material tangents and
  kernels need the memory points at the GLL nodes and no rotational
  feedback; a density direction is taken with the sea level engine's
  gravity and radius held (`SeaLevelParameters::body_from_table`;
  docs/dev/adjoints.md §16), and lateral density variations are excluded
  in 3D.
- **The `sl3d` program** exposes none of the discrete 3D kernels, the
  tangent-linear runs, the Hessian products and the material kernels;
  they are library calls.
- **Adaptive stepping** exists in `ViscoelasticOperator` but not in the
  evolution drivers.
- **Sharp lateral structure** (cratons, slabs) aliases in the pointwise
  products and needs a higher lmax than its smooth part; a δ given as a
  truncated expansion overshoots.
- **The truncation-radius rule** (§3) defaults to the graded rule; the
  choice between the two rules has not been measured.
- **Scaling past one socket.** On a two-socket node a 3D step at lmax 128
  is fastest on one socket's cores, and 32 threads were only 1.3 times
  faster than 16. The transforms (analyse and synthesise, three quarters
  of a pass) stop scaling first: GSHTrans threads the colatitude rows of
  each chunk and sums per-thread accumulators, and the analysis gets
  slower beyond 32 threads. The large arrays are not placed socket by
  socket (they are reused from step to step, so allocated once per run,
  but not first touched in the threads that use them), and the once-per-step
  start (`StartStep`) is mostly two analyses, which carry the transforms'
  limit. README, *Running*, has the advice that follows, and
  [dev/multicore-scaling.md](https://github.com/da380/sl3dpp/blob/main/docs/dev/multicore-scaling.md) the
  measurements. One process per socket, over MPI, is built (§12,
  [radial-partitioning.md](radial-partitioning.md)); its timing on the
  server is still to be measured.
- **Memory at lmax 512.** A 3D run there needs the recorded states on disk
  (the `on_state` store) and a large node.
- **Inversion.** There are no inversion drivers or optimiser interface: the
  library supplies forward runs, kernels and gradients for one to use.
- **An unexplained elastic discrepancy with the Fortran.** Elastic Love
  numbers from an updated build of the Fortran sit about 1% from pyslfp's,
  more than numerics; the cause has not been found. sl3dpp agrees with
  pyslfp (§11), which is the authority for the elastic problem, and Fortran
  comparisons are made for the time-dependent problems only.
