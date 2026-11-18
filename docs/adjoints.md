# Adjoints: discrete and continuous

*The adjoints and kernels of sl3dpp, 1D and 3D. The 1D path follows
pyslfp's adjoint machinery (`viscoelastic/evolution.py`: discrete and
continuous adjoints, quelling, 1D viscous kernels, checkpointing) and
Yu et al. 2025 (equation references below are to that paper); the 3D
path is sl3dpp's own. Sections 6a and 6b give the 3D continuous sweep
and the design of the discrete 3D transpose; §8–§17 the tangent-linear
runs, the second-order adjoints, the discrete 3D transpose (§12), the
material derivatives and kernels, and their distribution over radius;
§21 the kernels in lateral structure about a radial background; §22 the
derivatives of a run whose initial sea level is calibrated; §1a the
derivatives under Sdirk2, the scheme of two stages. The overview is
[design.md](design.md) §7; choosing a step and a scheme for a run and
its kernels is in [running.md](running.md), "Choosing a step and
scheme".*

## 1. Both formulations, and when each is accurate

Two adjoints are implemented as options, as pyslfp and AdGIA carry,
behind one interface and one result type so every gate runs on both
alike:

- **Discrete**: the exact transpose of the time-stepped forward map.
  Gradient identity to solver tolerance for any perturbation on the
  date grid; gated by dot-product tests at machine precision. Being
  the exact transpose of THIS code's step, it is also the
  porting-bug detector against pyslfp's (algebraically identical)
  implementation.
- **Continuous**: the adjoint equations of the continuum problem,
  discretised. They have the same form as the forward problem (Yu et
  al. §3.3), so between observations the adjoint state is stepped by
  the SAME machinery as a zero-load forward run in reversed time, with
  finite jumps of the adjoint internal variables at each observation
  (eq. 80). It never sees the stepper's internals, so it survives a
  change of scheme — and of discretisation.

**Accuracy.** Both kernels converge to the same limit, the derivative
of the problem continuous in time, at second order in the time step,
layer by layer; they differ in what their error at a given step is.

The discrete kernel is the exact gradient of what the program computes:
its error is the forward run's own time-discretisation error, at any
step, rheology and shoreline treatment. Against the kernel it reads
larger than against the value, a viscosity kernel being the derivative
of the small part of a value that the viscosity sets (dJ / d ln eta is
8–16% of J for the measurements below), the part the step's error lives
in.

The continuous kernel solves a second problem. Each measurement forces
the adjoint impulsively, and after it the adjoint state relaxes on the
model's SHORTEST relaxation time, which the forward run, forced
smoothly, never has to resolve; and the kernel is a time quadrature of
the product of the forward and adjoint fields, each discretised on its
own. Under the exponential trapezoid — exact for a strain linear across
a step, acting as backward Euler on a mode whose relaxation time is
short against the step — **the continuous
kernel is accurate when the steps are short against the shortest
relaxation time of the layers that matter**, and its error is set by
their ratio. On the prem8 deck's 1e21 Pa s Maxwell mantle (its
shortest relaxation time 0.108 kyr; the kernel paired with a uniform
change of ln eta, for cap means of the sea level 6 and 10 kyr into a
10 kyr deglaciation, against central differences of forward runs: the
time-step study, examples/configs/README.md) it is 0.4–0.6% at a ratio
of 1.2, 1.7–2.5% at 2.3 and 14–24% at 9.3, within about a factor of two of
that at the same ratio with the viscosity scaled by 4 or by 1/4, and far
larger in a weak layer (each layer's own kernel the further off the
shorter its relaxation time: 33–41% at 9.3 in the layer at the base of
the mantle), while at 9.3 the forward values are good to 0.2–0.5% and
the discrete kernel to 1.7–4.0%.

So `sl3d kernel` takes the discrete adjoint by default
(`kernels.method`), and the continuous one where asked, as a full
route: it is the continuum theory discretised and converges to the same
kernels. When a continuous kernel is asked for through time, `sl3d
kernel` states the run's longest step over the shortest relaxation time
of the model as built, and where that time is (the layer, the depth
and, under a lateral viscosity, the place;
`ViscoelasticOperator::ShortestRelaxation`,
`Viscoelastic3D::ShortestRelaxation`), as information. The library's
sweeps take either method through `Options::method`, the discrete by
default as in `sl3d kernel`.

**Agreement of the two kernels is a check of the time discretisation.**
They discretise the derivative differently and share its limit, so
where they agree the step resolves both the forward run and the
adjoint's relaxation; where they disagree the continuous kernel's steps
are long against the shortest relaxation time, while the discrete
kernel carries the forward run's error alone. A kernel's convergence is
judged by halvings of the step that each change it about a quarter as
much as the one before (second order), not by one small change: errors
of opposite signs (in a cap's degrees, say) can cancel at coarse steps.

Under Sdirk2 (§1a) the same holds with a smaller error in each: the
discrete kernel is the exact gradient of the two-stage run, 0.38 / 0.26%
off at a ratio of 9.3 on the study's problem, and the continuous one,
its adjoint's relaxation damped by the stages, 1.9 / 1.0%. Which scheme
and step to take for a run and its kernels is in
[running.md](running.md), "Choosing a step and scheme".

## 1a. The derivatives under Sdirk2

Under `Scheme::Sdirk2` (design.md §5) a step of length Δt is two
backward Euler stages of length γ Δt, γ = 1 − 1/√2: the first from mⁿ to
tⁿ + γ Δt, closed by the sea level equation under the forcing of the
step's two nodes combined (below), giving Y₁; the second from (1 − r) mⁿ + r Y₁, r = (1 − γ)/γ, closed
at tⁿ⁺¹, giving mⁿ⁺¹. Every derivative of the one-stage schemes is built
for it, 1D and 3D, frozen and migrating shorelines, with the rotational
feedback, a tide, interior sources, jumps of the forcing (a zero-length
step is one stage under every scheme), lateral viscosity and moduli,
over radial segments: the tangent-linear runs, the discrete adjoint and
the discrete second-order sweeps, which are derivatives of the stepped
map, and the continuous adjoint, which steps the adjoint state by the
same stages.

**The stepped map's derivatives, stage by stage.** A stage is a
one-stage step of its family from a memory state that is a constant
combination of states, so the derivative of a step is its stages'
derivatives chained through the combinations (`detail/StepStages.hpp`):

- *Forwards* (the tangent-linear runs, §8 and §10, and the second-order
  runs): the tangent state is stepped by the recipe as the forward state
  is (`StepByRecipe`), each stage opened from the same combination of
  tangent states, driven by the backward Euler weights' derivatives at
  γ Δt about the forward stage's own states (`FirstOrderDrive`, the
  material forces, the second derivatives' drive, each called with the
  stage's family and length), and closed under the direction as the
  stage takes it and the stage's own shorelines.
- *Backwards* (the discrete sweeps, §3 and §12, first and second order):
  the stages are transposed last to first, each the one-stage step's own
  transpose (the dual of the stage's end state in, the dual of its
  memory state out), and the dual of a memory state is handed to the
  states it combines, times their coefficients
  (`detail::TransposeByRecipe`): the second stage's dual goes r of it to
  Y₁ and 1 − r of it to mⁿ, and the first stage's adds to the latter.
  The node's measurement forces the stage that ends at the node alone.

The one-stage schemes are one stage of themselves, and take the same
code, operation for operation.

**The first stage's state is the stage repeated.** The derivatives pair
and drive each stage with the forward stage's states: Y₁ and its strain
beside the nodes'. The recording keeps the nodes' states alone; beside
each node it keeps what closed the first stage of the step ending there
(`Evolution::Snapshot::stages`, `ForwardTrajectory::stages`): its
total-load and centrifugal coefficients, two coefficient vectors, its
ocean function under migrating shorelines, and its sources' forces under
interior sources. Y₁ is then the stage again from mⁿ under those
coefficients (`TrajectoryCursor::Step`, `detail::ForwardStages3D`): one
stage solve with no sea level equation to close, the forward run's own
operations on the same numbers in 1D (the state is the run's to
rounding) and its own within-step iteration in 3D (to that iteration's
tolerance). Keeping Y₁ instead would double the recording, which is what
limits a 3D kernel run; repeating the stage costs one stage solve a step
of the sweep, beside its two transposed ones. A recording with gaps
(`record_stride` above 1) is served the same way under either adjoint:
a state between kept ones is the run's steps again, stage by stage, each
under its own recorded closure; in 1D the first stages those steps take
are held with the gap's states and handed to the sweep, which takes no
stage twice. A trajectory that does not give the
stages' closures is refused by name. A recorded tangent-linear run
(`Options::record`), which a second-order run about it reads, keeps its
own state at each first stage, a tangent stage not being rebuilt from
its closure alone.

**Shorelines.** About a migrating run each stage is linearised about its
own shorelines, as each node is about the node's: the sweeps and the
tangent runs take the run's ocean functions as one argument, the nodes'
and the first stages' (`OceanFunctions`, `OceanFunctions::Of` the run's
snapshots), and refuse a migrating Sdirk2 run's without the stages'.

**The forcing inside a step.** A run reads its forcing at its nodes
alone, under every scheme. The first stage ends at tⁿ + γ Δt, which is no
node's time, and no callback is called there: the stage takes 1 − γ of
the forcing at node n, as the step from there reads it (the right limit
at a repeated pair), and γ of the forcing at node n + 1, as that node
reads it (its left limit where it is the first of a pair), term by term
(`detail::AtStage` in `detail/TimeGrid.hpp`, the one place that decides
it): the direct loads, the layer on the solid surface, the tide, the ice
thickness, frozen and migrating, and the sources, whose moments at the
two nodes are taken times 1 − γ and γ. A land mask that changes in time
is the end node's. The derivative runs hold the forcing by its node
values and take the same combination: a direction's value at the first
stage is 1 − γ of its value at node n and γ of its value at node n + 1,
and the stage's kernel (the dual of the forcing the stage took) is
handed to those two nodes with the same weights. The node kernels of
`Kernels` are therefore the exact derivative of the stepped run in its
forcing's node values, which are all it depends on, whatever the forcing
does between nodes; and the tangent-linear run, the derivative along a
direction in them, is the discrete sweep's transpose to rounding. Under
migrating shorelines the thickness view grounds the stage's part by the
stage's own ocean function, ρ_i (1 − C_s), the initial sea level kernel
takes ρ_w (C_s − C₀) of it, and the quelled view pairs with the
differences of the thickness between nodes.

**Gates.** `TestSdirk2Derivatives`: under a load and a migrating ice
history curved in time across every step, the run is the run under their
piecewise-linear interpolants through its nodes (to rounding in 1D and
3D, no callback read between nodes), and the tangent-linear run along a
curved direction's node values is the run's derivative (2e-12 against
differences, the thickness kernel alike); the tangent-linear run against
differences of the forward run (the loads and the ice to 2e-12, the
problem linear in them; the relaxation times and the moduli at second
order in the difference step, Richardson to 5e-8), the discrete sweep
its transpose (8e-15 to 3e-12 in 1D, 8e-16 to 7e-14 in 3D under lateral
viscosity, and to 2e-13 from a stride-2 recording), about a migrating run
whose shorelines move inside the steps (7e-12 against differences, the
thickness and initial sea level kernels to 1e-14), across an onset under
interior sources (the moment kernels to 6e-13, the kernels about the
source-driven run to 2e-11) and a load jump with a tide, a load outside
the water budget and a layer (2e-13), the discrete Hessian symmetric and
equal to the second-order run (7e-13 frozen, 5e-12 migrating, 4e-13 in
3D) and the derivative of the gradient; in 3D also under lateral moduli
with migrating shorelines. `TestLateralKernels` holds the radial route's
lateral kernels to the 3D sweep's at zero lateral structure,
`TestDistributed` the sweeps, the tangent run and the second-order sweep
over two to four ranks to the one process's, `TestApps` `sl3d kernel`'s
discrete viscosity kernel to the difference of two `sl3d solve`s.

**The continuous sweep.** It steps the adjoint state through the reversed
step by the same stages (`StepByRecipe`), each closed by the memory-only
adjoint equilibrium, the toroidal adjoint beside it under the same
recipe, in 1D and 3D; the jumps at the measurements, the equilibria and
the kernels' node quadrature are the one-stage schemes' unchanged (about
a migrating trajectory, the run's shorelines within a step are read as
below). On the same ramp as
the forward gates at steps 9.3 times the shortest relaxation time, its
viscosity kernel is 1.4% off against the trapezoid's continuous 3.4%,
and converges at second order to the derivative central differences of
the forward runs give (0.32% and 0.077% at the next two halvings,
TestAdjoint); its moment kernels, where the adjoint's relaxation after a
measurement matters most, are 15 to 30 times closer than the trapezoid's
at the same steps. On the time-step study's problem (§1,
examples/configs/README.md) at 1 kyr, 9.3 times the shortest relaxation
time, its continuous kernel is +1.9 / −1.0% off uniformly (the
trapezoid's +14 / +24%), 4.6 / 1.7% in its worst layer (33 / 41%) and
−5.2 / −4.4% along the ice pulse (+25 / +41%), falling at second order.
The kernels in the relaxation times and the moduli stay the node
quadrature of the forward and adjoint fields. A quadrature consistent
with the stages would integrate the product of
the two sweeps' within-step trajectories, each given by its stages; the
forward's stages fall at tⁿ + γ Δt and the adjoint's at tⁿ⁺¹ − γ Δt, so
it needs both sweeps' stage states and each interpolated to the other's
times. It is not built.

**The continuous sweep about migrating shorelines.** The adjoint's
coefficient at each time is the run's ocean function at that time (Yu et
al. 2025, §3.3), and a run solves for its shorelines at its nodes and, under
Sdirk2, at the inner stages (tⁿ + γ Δt) alone. The continuous sweep reads
both (`OceanFunctions`, refused without the stages' as the discrete sweep
is), in 1D and 3D:

- *Its reversed stages.* The last stage of the reversed step from node
  n + 1 closes at node n, under the node's ocean function. The inner one
  closes at tⁿ⁺¹ − γ Δt, where the run did not solve; it takes the run's
  ocean function linear in time between the solves either side of that
  time, the inner stage's at tⁿ + γ Δt and node n + 1's
  (`detail::OceanFunctionWithinStep`): in a cell whose shoreline moved
  before the stage the stage's value, in one that moved after it a
  coefficient between 0 and 1 (the adjoint's closure is linear in it).
- *Its ice views.* The kernels in the thickness and the initial sea level
  are time integrals of the load kernel against ρ_i (1 − C(t)) and
  ρ_w (C(t) − C₀). Their quadrature within a step is the scheme's own:
  each inner stage carries its share b_i Δt of the step (Sdirk2's
  (1 − γ) Δt; `detail::StageQuadratureWeights`) at its own time, its load
  kernel there the adjoint density linear between the step's two nodes,
  and that part is grounded by the stage's own shorelines and handed to
  the step's nodes 1 − γ and γ, as the discrete sweep grounds and hands its
  stage's part (`detail::ContinuousStageLoads`, `AssembleIceViews`). A cell
  whose shoreline moves inside a step thus enters the views as the run
  had it there.

*Order.* An ocean function jumps in time where a cell's flotation flips,
inside some step, and a run samples it at its solves' times alone, so
every kernel about such a run, discrete or continuous, under either
scheme, carries an error of first order in the step in the cells that
flip, against the derivative of the problem continuous in time (zero in
mean over where in a step the flips fall, but not small for a few cells).
Under the exponential trapezoid the two sweeps sample the shorelines
alike, at the nodes, and their difference is of second order. Under
Sdirk2 the discrete sweep's samples are the run's, at the nodes and the
inner stages, while the continuous sweep's inner stages fall between them,
and in the cells that flip inside a step the two differ at first order;
where the shorelines move only at nodes (a solved start's among them) the
difference is of second order. On the marine cap of `TestMigratingDerivatives`
(a solved start on a 1e22 Pa s mantle, two cells flipping inside steps)
the continuous kernels (the ice views and the relaxation-time kernel) are
1e-3 to 7e-3 from the discrete at 5 nodes and 2e-7 to 2.4e-6 at 2049,
halving with the step from about 129 nodes; under the trapezoid 1e-2 to 2.4e-2 at 5 nodes and
5e-8 to 1e-7 at 2049. Either sweep's ice views, under either scheme,
differ from a run's at 2049 nodes by 1e-5 to 2e-4 at 513 and 1025 nodes,
not monotonically: the first-order error both carry is larger than what
separates them. A kernel at one node whose support floats within a step
of it is the extreme case: at 5 nodes the thickness at a node with no
grounded ice still loads the run through the inner stage of the step
ending there, 39% of the derivative, which the continuous sweep meets
through that stage's shorelines (the kernels 2.1% from central
differences; `Migrating.TrajectoryKernelsUnderSdirk2`).

**Not built under Sdirk2, the default scheme, each refused by name with
its cure:** the continuous
second-order sweep, whose derivative state's drive is a one-stage
step's, and with it the continuous drive of a tangent-linear run along
the relaxation times (`TangentLinearSolver::Options::method`
`Continuous` with an `itau` or log-viscosity direction) and the
continuous second-order tangent-linear run, which serve that sweep
alone. The discrete second-order sweep and its runs (the default
method) are built, and a tangent-linear run along the forcing or the
moduli is the same by either method. The continuous second order is had
by stating a one-stage scheme, `Scheme::ExponentialTrapezoid`, in the
evolution's `Options::scheme` (§9, §11): the refusals say so.

## 2. Scope

1. **1D kernels** (the pyslfp port): ice-history kernels for objectives
   J = Σ_i ∫ g_i · (ξ, u, φ)(t_i) dS + g_ω·ω(t_i), by both sweeps;
   inverse-relaxation-time (itau) kernels, both ways, with the
   viscosity mapping; checkpointing (record stride + forward rebuilds
   of the gaps); the kernel views of §4.
2. **3D kernels**: the continuous adjoint of the coupled 3D
   evolution and the grid viscosity kernel μ_k m†:(d − m)/itau (§6,
   §6a), and the discrete transpose of the reduced pipeline, designed in
   §6b and built (§12).
   **Both kernel granularities are deliverables**:
   the full 3D viscosity kernel on the (memory nodes × angular grid),
   AND the spherically averaged 1D radial kernel as pyslfp provides —
   the latter both from the 1D sweeps directly and as the spherical
   average of the 3D kernel, whose agreement in the zero-lateral limit
   is a free consistency gate between the two pipelines.
3. Linearisation about a migrating-shoreline trajectory (pyslfp's
   `ocean_functions`), 1D and 3D: the sweeps' `oceanFunctions` argument
   (design.md §7). Decay-time functionals (pyslfp `functionals.py`), a
   front end over Measurement: `DecayTimeMeasurements` in
   `Measurements.hpp`.

## 3. The 1D sweeps (pyslfp's algebra, verbatim)

The authoritative statements are pyslfp `evolution.py` docstrings and
code; what follows fixes the sl3dpp mapping only.

- **Discrete** (`adjoint_ice_history`): one reversed sweep carrying the
  duals (m†, d†) per degree. Per step: the duals of the end solve
  gather through the strain transpose (`Degree::DualFromScalars` — the
  plain transpose of `Degree::Strain` under the nodal pairing); the
  unit load/tide responses contract them to load/tide coefficient
  duals; those join the measurement duals through the engine adjoint
  (`GRDSolver::SolveAdjoint`, the sea-level solver's transpose); its channel
  duals force one symmetric effective solve per degree; the weight
  recursion closes m† and d†. The start is the transposed
  instantaneous elastic solve. Yu et al. §3; pyslfp lines 1554–1663.
- **Continuous** (`continuous_adjoint_ice_history`): reversed zero-load
  stepping by `StartStep`/`Finish` with the memory responses through
  the SLE's response channels — exactly the forward loop with no ice
  load. At an observation: the instantaneous adjoint fingerprint via
  the forward channel mapping (ζ_d = g_ξ, ζ^u = −g_u, ζ^φ = −g g_φ,
  L = −g (g_ω − Ψ* g_φ), with Ψ* the adjoint of the centrifugal
  operator), then the radial jump m† += itau d̃† with d̃† the strain of
  the unrelaxed solve under the converged (λ, ψ†) and the SPLIT load
  channels (`DegreeOperator::Load(LoadPart::Force/Potential)`) carrying
  displacement and potential measurements. The smooth kernel density is
  the memory-only adjoint equilibrium's sea level, kept one-sided
  across each observation (the density jumps there).
- **itau kernels.** Discrete: per step, the h-derivatives of the scheme
  weights (`ExponentialTrapezoidWeightDerivatives`,
  `BackwardEulerWeightDerivative`) pair the sweep's duals with the
  recorded forward internal variables and strains, and the effective
  modulus's dependence pairs the transposed and forward solutions
  through the branch bilinear (pyslfp lines 1603–1627). Continuous:
  dJ/d itau(r) = ∫ W_k m†/itau (d − m) dt over the reversed nodes with
  the split trapezoid (the adjoint state jumps at observations; the
  self-adjoint channel mapping poses the adjoint in the g-weighted
  energy form, so the carried radial state is g × the true multiplier —
  hence a division by g in material units). `Kernels::Viscosity`
  maps to dJ/dη via dη = −η² d(itau)/μ per branch, as pyslfp.
- **Checkpointing.** The forward `record_stride` stores the internal
  state every stride-th node (the first and the last always) and the
  per-node load/tide coefficients; a `TrajectoryCursor`
  (`detail/TrajectoryCursor.hpp`) serves the states to every 1D
  derivative run, the reverse sweeps (first and second order, discrete
  and continuous, the lateral kernels' too) and the tangent-linear runs
  (§8). Recordings keep the internal variables only, as in pyslfp and
  AdGIA; the cursor regenerates each stored state's strain by one elastic
  solve under that node's recorded loads, and the 3D sweep does the same
  per node. A state between two kept ones is rebuilt by the run's own
  steps forward from the kept state below, each finished under its end
  node's recorded total-load and centrifugal coefficients, which are all
  the radial step consumed: the sea level solve, a migrating run's ocean
  functions, the rotation and a tide reach it only through them, and a
  jump's two nodes are a step of zero length between their recorded
  sides. Forward steps damp, so the rebuilt states are the run's to
  rounding at any stride (the step at −dt would invert a step exactly,
  but amplify rounding by exp(h) per step, h = dt/τ, without bound
  across a gap on a weak layer). A reverse sweep entering a gap rebuilds
  it up to the node it asks for and holds those states until it has
  passed below them, so it holds the kept states and at most a stride's
  more, about N/stride + stride of N nodes, fewest near stride √N; it
  pays one step per state of a gap (one effective solve a degree, as
  the run), once. A step of Sdirk2 is rebuilt stage by stage, its first
  stage under the stage's own recorded coefficients (§1a). Not rebuilt:
  a run under interior sources, whose states are not the loads' alone
  (§19): it is recorded at every node, a stride other than 1 refused.
  `TestTrajectoryCursor` holds it on a layer of 1e19 Pa s, 254 of its
  relaxation times a 1 kyr step, at strides 2, 6 and 5 (an uneven last
  gap) with a repeated node in the gaps, under loads and a tide and about
  a migrating run: every kernel of both first-order sweeps, the
  second-order sweeps' kernel derivatives and the tangent-linear runs
  they read equal the full recording's within 1.4e-11 (discrete) and
  1.2e-10 (continuous), the departure the kept states' regenerated
  strain alone (their elastic solve against the run's step, a rounding
  the layer's relaxed d − m amplifies): with the strain kept
  (`record_strain`) the kernels are the full recording's to the bit.
  `sl3d kernel` on a ball of a 1e19 Pa s mantle likewise (TestApps).

## 4. Result type and views

`Kernels` carries the same three views as pyslfp, so the gates and
any inversion driver are formulation-blind:

- **Discrete view**: `directLoadKernels[n]` (dual of the direct load grid at
  node n), `thickness[n]` (grounding factor ρ_i (1−C) applied, the
  baseline −ρ_i (1−C) Σ_n directLoadKernels folded into node 0), `baseline`,
  `initialSeaLevel` (≡ 0 under frozen shorelines). Exact pairing on
  the time grid.
- **Distributional view** (Yu et al. §3.7, eq. 81): per-time `density`
  (converges under grid refinement) plus `instantaneous[i]` delta
  amplitudes — the unrelaxed fingerprint adjoint of each measurement,
  independent of the time grid.
- **Quelled view** (Backus): `rate[k]`, the kernel against the
  thickness RATE on interval k — the cumulative tail Σ_{n>k} of the
  thickness duals; bounded, with finite jumps at observations only;
  the initial-thickness dual vanishes identically.
- `itauKernel` per branch over (element, gll) when a trajectory is
  supplied, and `Viscosity(rheology)` mapping it to the viscosity's.

### The kernels' convention

Every kernel with respect to a field in the Earth — the relaxation
times', the viscosity's, the moduli's, in the time-stepped and the
static problems, their derivatives in the second-order sweeps — is a
**density per unit volume**, held as samples of a function at the
points where the parameter lives. With K the kernel and δ the
perturbation of the parameter,

    dJ = ∫ K δ dV.

In 3D the samples are K(x) on each memory node's grid, and

    dJ ≈ Σ_node w_node r_node² ∫ K δ dΩ

(`VolumePairing(coupling, K, δ)`; w the node's radial quadrature
weight). In 1D the samples are K(r) at the memory points or the nodes,
and for the spherically symmetric perturbation δ(r)

    dJ = 4π ∫ K(r) δ(r) r² dr ≈ Σ_p W_p K(r_p) δ(r_p),

W_p = 4π r_p² times the point's radial quadrature weight
(`MemoryPoints::VolumeWeights`, `VolumePairing(points, mesh, K, δ)`).
Neither the r² nor the 4π is in the kernel: both belong to the volume.
So a 1D kernel is the same function as the 3D kernel of the same
model, and a 3D kernel's reduction per node (`…Spherical`,
`SphericalMean`) is its mean over the sphere, the kernel of an
angle-uniform perturbation.

The values are non-dimensional in the material's scheme: the
objective's unit per unit of the parameter and of volume. The programs
write them in SI, the objective's unit per parameter unit per m³.

The discrete and the continuous sweeps both approximate these
functions. The sweeps accumulate the gradient with respect to each
point value, which is the kernel times the point's volume weight, and
divide the weights out on return; the pairing above with the weights
restored is then the exact discrete derivative for the discrete
sweeps. A node at the centre has no volume: its sample is the element's
polynomial through the other nodes and its gradient entry is not kept.
For a kernel accumulated at the points themselves the entry is small
but not zero by construction: at a GLL node at r = 0 it is the node's
quadrature weight (the degree forms carry no r² there) times products of
the solution's departures from regularity at the centre, spectrally
small, at rounding on production meshes. For the nodal relaxation-time
kernel of Gauss memory points in an element at the centre that relaxes
it is not small, and the pairing misses the sensitivity to the value at
the centre alone (about 1e-5 of that element's). In 3D no memory node sits at the centre: the
Gauss points are interior to the element holding it, which the coupling
carries like any other, and at the GLL points that element is not
carried (§13k, "The element holding the centre").
The kernels with respect to the ice history are fields on the surface
grid and pair by ∫ · dS (below).

Conventions (sl3dpp's, not pyslfp's): times are the forward driver's —
increasing, SLE solver units, a node repeated once for a jump of the
forcing (§20); measurements are keyed by NODE INDEX into the time grid
(no float matching), an observation at a repeated time on the right
node; duals are non-dimensional SLE-unit
densities pairing by ∫ · dS on the solver grid, the angular-velocity
dual by the dot product; densities/rates are per unit SLE time.

## 5. Architecture

- `Adjoint.hpp`: `Measurement`, `Kernels`, `KernelDerivatives`,
  `ForwardTrajectory` (built from recorded snapshots) and
  `AdjointSolver`, which borrows a `Evolution` (the operator,
  solver, state and unit ratios the sweeps need). Its
  `FirstOrderKernels` runs either 1D sweep by `Options::method`
  (`Adjoint.cpp`); its `SecondOrderKernels` the second-order sweeps
  (`AdjointSecondOrder.cpp`, §9). The pieces they share (the dual
  packers, the node weights, the measurement check) are in
  `detail/AdjointInternals.hpp`, and the trajectory cursor, which the
  tangent-linear runs read too, in `detail/TrajectoryCursor.hpp` (§3).
- `Adjoint3D.hpp`: `ForwardTrajectory3D`, `Kernels3D`,
  `KernelDerivatives3D` and `AdjointSolver3D` over a
  `Evolution3D`. `FirstOrderKernels` and `SecondOrderKernels` run
  the continuous sweep, one core in `Adjoint3D.cpp` (§6a, §11);
  `Options::method = Discrete` the discrete transpose in
  `AdjointDiscrete3D.cpp` (§12).
- `TangentLinear.hpp`, `TangentLinear3D.hpp`: `TangentLinearSolver` and
  `TangentLinearSolver3D`, the forward mode (§8, §10).
- Internals access is by friendship; nothing speculative is made
  public. `ViscoelasticOperator` befriends `AdjointSolver`,
  `AdjointSolver3D` (the 3D continuous sweep's jump) and
  `TangentLinearSolver`, and `ViscoelasticStep` befriends
  `TangentLinearSolver` (`Viscoelastic.hpp`); `Viscoelastic3D` and its
  `OpenStep` befriend `AdjointSolver3D` and `TangentLinearSolver3D`
  (`Viscoelastic3D.hpp`); each evolution befriends its adjoint and
  tangent-linear solvers. The degree workspace
  (`detail/ViscoelasticDegree.hpp`) carries the transposes
  (`DualFromScalars`, `momentRow`, `ApplyWeights`); the operator caches
  the split unit responses (`SplitUnitResponses`) and the scheme weights
  with their h-derivatives (`SchemeWeightsOf`, `WeightsOfOrder`).

## 6. The 3D path

- **Continuous**: the adjoint equations' form is again the forward's,
  so the reversed sweep reuses the coupled machinery of Evolution3D —
  zero-load reversed stepping with the
  within-step lateral iteration intact, jumps at observations through
  the unrelaxed 3D solve, the lateral term entering the fixed point
  exactly as forward. The viscosity kernel is the 3D form of Yu et al.
  2025's multiplier integral, as in pyslfp and the 1D sweep (not the
  Fortran's rate-form kernel expression): per branch and memory point,
  μ_k m†:(d − m)/itau, accumulated over the reversed nodes with the
  split trapezoid. The contraction pairs the adjoint memory with the
  forward d − m, and its metric on the reduced storage is fixed by the 1D
  limit (§6a), not taken from GSHTrans's tensor types. The reality
  reduction gives the derived spins by sign·conj for free at the
  contraction, and the adjoint fields are real tensors too.
- **Discrete option**: the exact transpose of the reduced pipeline —
  Scatter/batched-synthesis/pointwise-multiply/analysis/force-rows in
  reverse, the m ≥ 0 column reduction and the spin orbit relation
  transposed consistently (the pipeline is ordered for it), and the
  transpose of the converged within-step fixed point
  solved by the same Anderson-mixed iteration on the transposed
  system (as the SLE adjoint is). This is the machine-precision gate
  for the 3D machinery, and the sweeps' default beside the continuous
  one (`AdjointMethod::Discrete`, `Options::method`'s default, as of
  `kernels.method` in `sl3d kernel`).
- Gate chain: zero-lateral limit — the 3D sweeps must reproduce the 1D
  sweeps' kernels (machine precision for the discrete, tight for the
  continuous); dot-product identity for the discrete 3D transpose; FD
  checks on the grid viscosity kernel; discrete-vs-continuous ladder
  over lateral contrast (the check of the time discretisation, §1 — a
  research measurement alongside `lateral-coupling.md` §5a).

## 6a. The 3D continuous sweep as built

`Adjoint3D.hpp/cpp`: `AdjointSolver3D` (the continuous sweep; the
discrete transpose is built too, §12), `ForwardTrajectory3D` (a recording
at any `Evolution3D::Options::record_stride`: the sweep rebuilds
the gaps by forward steps from the kept states), and
`Kernels3D`: the 1D `Kernels` views, `itauKernel3D` (per branch, the
density per unit volume over memory node × coupling grid, §4), its mean
over the sphere `itauKernelSpherical` (the 1D bridge),
and `logViscosityKernel3D` = −itau(x)·K (the operator's delta
parameterisation).

The pieces, each mirroring a gated 1D counterpart:
- the observation JUMP is the 1D one (the unrelaxed problem is
  laterally uniform): the elastic adjoint fingerprint's channels drive
  the reference unit/split responses, the strain enters the pipeline
  via `AddStrainOfSolution` + `Synthesise`, and
  m† += itau(x)·d̃†(x) pointwise with the laterally varying itau;
- between observations the reversed stepping is Evolution3D's coupled
  pass loop verbatim with no ice load (`OpenStep::Respond` /
  `AddChannels` / `StrainOf` / `Finish` under the Anderson mixer);
- the kernel contraction's null-basis metric is FIXED by the 1D limit:
  on the reduced storage, C(x) = r²·[12·A₀B₀ + 8 Re(A₋₁B̄₋₁) +
  4 Re(A₋₂B̄₋₂)] — derived by requiring the angle-uniform Parseval
  reduction to reproduce the force weights (4/3, k², k²(k²−2)) through
  `ComponentsFromScalars`, and verified by the zero-lateral gate;
- the covariant pairing ratio of the 1D sweep (the pyslfp
  units-covariance fix) carries over unchanged.

GATES (TestAdjoint3D): zero-lateral — ice kernels and the spherical
reduction equal the 1D continuous sweep's at 1e-6 through the whole
pipeline; lateral — the full 3D log-viscosity kernel against central
differences of the coupled forward along an analytic direction, at the
commute level (< 10%) in the resolved regime and improving under step
refinement; and the canonical Taylor test: |J(δ+hp) − J| = O(h) and the
gradient-corrected remainder |J(δ+hp) − J − h⟨K,p⟩| at second order on
the curvature-dominated end of the h-sweep, monotone, and inside the
commute floor (0.1·h·|⟨K,p⟩|) everywhere. The machine-grade Taylor
test, second order to solver precision at every h, is the discrete
transpose's (§12).

## 6b. The discrete 3D transpose: design (built, §12)

The step is LINEAR throughout, so its exact transpose is mechanical
once the pairings are fixed. Write the forward within-step pass (the
code of Evolution3D::Run and OpenStep) as d' = Exit K⁻¹ [Entry(m̃ +
Δβ d) + units·closure(f, S K⁻¹ Entry(...))] with
- Entry(T) = SpheroidalForce/ToroidalForce(Analyse(μ_k-scaled T)) — the
  grid entry with branch moduli,
- Exit(u) = Synthesise(AddStrainOfSolution(u)) — the strain exit,
- K the reference effective systems (symmetric), S the surface reads,
  closure the (linear) generalised SLE, Δβ = diag(β_k(x) − β_ref_k(r)).

*Note.* The forward code iterates on the
strain's COEFFICIENTS s, not its grid values d. A pass is
s' = AddStrainOfSolution(K⁻¹[fixed + Analyse(W · Synthesise(s)) +
channels]), where fixed = Analyse(Σ_k μ_k m̃_k) is formed once per step
and W = Σ_k μ_k Δβ_k(x) is a single weight field. The step map, and so
everything in this section, is the same; only the code's split differs:
Exit stops at the coefficients, Synthesise opens the next pass, and
Entry's constant part is applied once per step. A transposed iteration can likewise run on coefficient
duals.

**The load-bearing duality**: Entry = Exitᵀ under the ENERGY pairing
⟨S, T⟩_W0 = Σ_node Wq(node) · r² [12 S₀T₀ + 8 Re(S₋₁T̄₋₁) +
4 Re(S₋₂T̄₋₂)] on the reduced components against the REAL dof pairing
with m ≥ 0 column weights (1, 2, 2, ...) — the SAME metric the
continuous kernel contraction derived from the 1D force weights, and
which the zero-lateral kernel gate already validated. With duals
carried in this pairing, the transposed pass is the forward pass with
the force/strain pair swapped and the SLE replaced by its
adjoint:

  passᵀ(y): x† = Entry-with-Δβμ-weights(y) … K⁻¹ (symmetric) …
            unit-response contraction → SLE ADJOINT → surface-dual +
            moment-row injections → K⁻¹ … Exit → Δβ-weighted tensor.

The transposed step: w = d† + Σ_k β_k(x) m†_k; solve y = Mᵀy + w by the
SAME Anderson-mixed iteration (the transpose of the converged fixed
point is the transposed system's solution, as the SLE adjoint already
does); then one application of the constant part's transpose yields
m̃†_k (→ the e/α recursion closing m† and d†, pointwise fields) and the
node's load dual; the measurement duals enter the forcing w through one
SLE-adjoint chain per node, as in the 1D discrete sweep.

The gates of the transpose as built are in §12.

## 7. The pieces and their gates

a–d are described in §3 and §6a, e in §12.

a. **Discrete 1D sweep** (ice kernels, full Measurement). GATES:
   dot-product/gradient identity at solver tolerance against the
   forward driver for random perturbations on the time grid; pyslfp
   fixture comparison (identical algebra — near machine precision).
b. **Continuous 1D sweep**. GATES: agreement with (a) at
   discretisation accuracy, tightening under dt refinement; the
   singular-part separation and quelled-view identities (pyslfp's
   `test_kernel_singular_parts_are_separated`, `test_quelled_rate_kernels`).
c. **itau/viscosity kernels + checkpointing**, both sweeps. GATES: FD
   kernel checks; checkpointed == full-record to rounding at strides
   whose gaps span many relaxation times; pyslfp fixtures.
d. **3D continuous sweep + grid viscosity kernel** (§6). GATES:
   zero-lateral limit; FD checks; Δt refinement.
e. **3D discrete transpose** (§12). GATES: dot-product at machine
   precision; zero-lateral == 1D discrete; the contrast-ladder
   diagnostic.

Fixtures: `scripts/make_kernel_fixtures.py` driving pyslfp, committed
under `tests/data/` with provenance headers, like the other fixtures.

## 8. The tangent-linear runs

The forward mode, ported from pyslfp's `tangent_linear_ice_history`
(its docstring is the authoritative statement). With the
trajectory's ocean functions held the forward problem is linear in its
loads, so the derivative along (δf, δI, δS₀, δitau) is the forward
problem once more:

- the shorelines prescribed from the trajectory (`oceanFunctionOverride`
  on every generalised solve, as the sweeps do);
- the direct load δfₙ + ρ_i (1 − Cₙ) δIₙ − ρ_i (1 − C₀) δI₀ +
  ρ_w (Cₙ − C₀) δS₀, with the start the instantaneous elastic response to
  its first node (zero for thickness and sea level directions);
- per step, the opened step's decayed memory state shifted by a drive
  (`ViscoelasticStep::AddMemory`), bilinear in the trajectory's (m, d) at
  the step's ends and δitau at the memory points (the forward map of
  `ItauKernelsAtNodes`' transpose, `ItauPerturbationAtPoints`):
  - Discrete: dh (de m⁰ + dα d⁰ + dβ d¹) for the trapezoid, dw dh (m⁰ − d¹)
    for backward Euler, dh = dt δitau — the forward mode of the scheme
    weights, whose reverse mode is the discrete sweep's itau pairing;
  - Continuous: α q⁰ + β q¹ (trapezoid) or (1 − w) q¹ (backward Euler)
    with qⁿ = (dⁿ − mⁿ_k) δitau_k/itau_k — the continuous drive entering
    like the strain, whose transpose is the continuous sweep.

The two coincide in the loads and differ in itau by the order of the
scheme. The forward trajectory is served forwards by the sweeps'
`TrajectoryCursor` (§3): stored states where the recording kept them
(strain regenerated by one elastic solve under the recorded loads), the
run's forward steps under the recorded loads between, the newest two
held. With `Options::record` every snapshot
keeps the tangent-linear state with its strain and the node's load and
centrifugal coefficients: what the second-order runs read.

Weights: `StepWeights.hpp` carries the second h-derivatives (d2e = e,
d2α = d2φ₁ − e, d2β = −d2φ₁ with d2φ₁ = −(e + 2 dφ₁)/h and its series
below the switch; 2/(1 + h)³ for backward Euler), cached by the operator
beside the weights and their first derivatives (`WeightsOfOrder`).

GATES (`TestTangentLinear`): linear in the loads at 1e-8 against the
difference of two forward runs; the discrete pair exact transposes — the
objective of the run along (δf, δitau) equals the pairing of the load
duals (node 0 included) and the itau kernel at 1e-8, each argument alone
too, and a stride-2 trajectory reproduces the full one at 1e-10; the
continuous drive converging to the discrete at the scheme's order (gap
< 0.2, falling by > 3 as the steps halve, η = 1e22 Pa s, 10 kyr ramp);
about a migrating marine-cap trajectory, the run along thickness and
initial-sea-level perturbations that flip no flotation cell equal to
central differences of migrating runs at 1e-7, both methods, and the
kernels' thickness and initial-sea-level views pairing to the same
number; the initial sea level inert under frozen shorelines.

**The program.** `sl3d tangent` makes these runs (and the 3D ones, §10)
by the discrete method, or by the continuous one where its config asks
(`tangent.method`), along a direction a config's `perturbation` block
gives as the inputs are given, about `solve`'s run of the same config;
a direction in the viscosity is one in the branches' relaxation times,
every branch's alike or one branch's, on every rheology. The gates of its
files against pairs of `sl3d solve` runs, of its measurements' changes
against `sl3d kernel`'s kernels paired with the perturbation, and of the
continuous run against the discrete one as the steps shorten, are
`TestAppsTangent`'s ([sl3d-configs.md](sl3d-configs.md),
tangent).

**The second-order run** (`RunSecondOrder`; pyslfp's
`second_order_tangent_linear_ice_history` under the sharp flotation
rule): F''(x)[u, v] as the forward problem with no direct load, the
shorelines prescribed, and the drive the sum of the first-order drive
taken with v's recorded tangent trajectory and u's δitau, with u's
trajectory and v's δitau, with the forward trajectory and the second
variation itau_uv (empty for a parameterisation linear in itau), and, by
the discrete method, the second derivatives of the scheme weights with
the forward trajectory and both perturbations (dh_u dh_v (d2e m⁰ +
d2α d⁰ + d2β d¹), or d2w dh_u dh_v (m⁰ − d¹)). With the ocean functions
held the map is affine in the loads and the initial sea level, so those
directions enter only through the tangent trajectories. The smoothed
flotation rule's shoreline terms (pyslfp's `flotation_curvature`) are
not ported: sl3dpp has no smoothed rule. Both runs share one loop
(`March`: prescribed shorelines, a direct load per node, a memory
increment per step).

GATES: the second-order run along (u, v) is the derivative, at the base
point along u, of the first-order run along v — central differences
with the loads moved by ε δu and the rheology by ε δitau_u converge at
second order (ratio in 3–5) and Richardson lands on it at 1e-6, both
schemes — and is symmetric in u and v at 1e-10; it vanishes identically
without a viscosity perturbation, about frozen and migrating runs alike;
by the continuous method it converges to the discrete one at the
scheme's order (gap 0.11 → 0.027 from 17 to 33 nodes over 10 kyr at
η = 1e22; the coarser grids, h ≈ 1, are pre-asymptotic); unrecorded
tangent runs are refused.

## 9. The second-order adjoint

`AdjointSolver::SecondOrderKernels` (`AdjointSecondOrder.cpp`; pyslfp's
`second_order_adjoint_ice_history` and `_second_order_continuous_sweep`
under the sharp flotation rule, Yu et al. 2025 §4): the kernels of the
objective and their derivative along a direction v — the action of the
objective's Hessian on v, returned as `KernelDerivatives` (the thickness
and initial-sea-level views' derivatives in the kernels' own pairings,
and the itau kernel's). Inputs: the recorded forward trajectory, v's
tangent-linear run recorded at every node by the same method, v's
δitau, and a migrating run's ocean functions. The sweep that gives the
kernels is itself a function of the parameters, through the forward
trajectory and the relaxation times; differentiating it along v gives a
second sweep of the same form, forced by the first sweep's state and
δitau, with v's tangent trajectory entering where the forward one enters
the kernels.

- **Discrete** (the exact derivative of the exact gradient; the Hessian
  symmetric to rounding): per reversed step, beside the first sweep's
  (m†, d†), the derivative duals (dm†, dd†). The first sweep's solve is
  the first-order one; the second sweep's dual of the step's solution is
  dd† + Σ B dm† + Bh dh m† through the strain transpose plus the source
  Σ Bh dh W s_y (the effective system's derivative acting on the first
  sweep's solution), contracted with the unit responses into load and
  tide duals and passed through the SLE adjoint with no measurement; its
  radial solve closes dm̃† = dm† + W d s_y. The pairings g (the itau
  kernel) and dg (its derivative: the same expression with each factor
  replaced in turn by its derivative — the dual state's, the tangent
  trajectory's (v m⁰, v d⁰, v d¹), and the weights' second derivatives
  times dh) accumulate at the memory points; the recursion closes both
  states (dm† = E dm̃† + Eh dh m̃†, dd† = Σ A dm̃† + Ah dh m̃†). The start
  is the transposed instantaneous solve for both dual states. Weights in
  the trapezoid's (E, A, B) form, backward Euler as (w, 0, 1 − w).
- **Continuous**: the adjoint state and its derivative are both stepped
  by the forward machinery in reversed time; the derivative state's step
  carries the continuous first-order drive built from the adjoint
  trajectory (`TangentLinearSolver::FirstOrderDrive` with the primed
  pre-step state and the post-step state); at an observation both jump
  by the instantaneous adjoint strain, over itau for the state and by
  δitau for its derivative; the viscosity pairing is differentiated with
  the derivative state and v's tangent trajectory in place, less
  (δitau/itau) times the first-order pairing. Assembly as the first
  sweep's (split node weights, lumps on the kernels only).

The shoreline terms of pyslfp's smoothed flotation rule are not ported.
The sweeps' shared pieces (dual packers, node weights, measurement
check) are in `detail/AdjointInternals.hpp`, the trajectory cursor in
`detail/TrajectoryCursor.hpp`.

GATES (`TestSecondOrderAdjoint`): the kernels the second-order sweep
carries equal the first-order sweep's at 1e-12, both methods, about
frozen and migrating runs; along v, central differences of the gradient
paired with u converge at second order to the derivatives paired with u
and Richardson lands on them at 1e-6, both schemes; ⟨H u, w⟩ = ⟨H w, u⟩
at 1e-9 and equals ⟨g, F''[u, w]⟩ from the second-order tangent-linear
run at 1e-9, frozen and migrating; the continuous derivatives converge
to the discrete block by block (ice, viscosity) and their mismatch with
the continuous second-order run falls at the scheme's order (17 → 33
nodes, η = 1e22), the discrete pair matching at 1e-9 at every spacing.

## 10. The 3D tangent-linear runs

`TangentLinearSolver3D` (`TangentLinear3D.hpp/cpp`; sl3dpp's own — pyslfp
has no 3D): the derivative of the coupled lateral-viscosity evolution
along a direction in the direct load, the ice thickness, the initial sea
level and the laterally varying inverse relaxation times, and the
second-order run along two directions. The direction in itau is per
branch over the coupling's grid values (the layout of
`Kernels3D::itauKernel3D`), or a log-viscosity field δδ(x) which maps
to it by δitau_k = −itau_k(x) δδ (`ItauOfLogViscosity`; the second
variation of the log parameterisation, itau u v, by
`SecondVariationOfLogViscosity`).

The algebra is the 1D one pointwise. The coupled step is linear in its
loads and its memory with coefficients that depend on itau(x) through
the pointwise weights alone: the decayed memory e(x) m + α(x) d⁰, the
lateral weight Σ μ_k (β_k(x) − β_ref) in the fixed point's force, and
the finish m⁺ = m̃ + β(x) d. Differentiating each in itau(x) gives
terms that together are exactly the 1D drive dh (de m⁰ + dα d⁰ + dβ d¹)
added to the decayed memory — `OpenStep::AddMemory`, which raises the
fixed memory force Σ μ_k m̃_k and the end state alike, so the within-step
iteration, the reference effective systems and the sea level closure
run unchanged (the lateral weight W δd is the trajectory's). The
continuous drive α q⁰ + β q¹, q = (d − m) δitau/itau, is pointwise too.
The forward trajectory is served forwards by `ForwardCursor3D`: kept
states as recorded, the gaps rebuilt by forward coupled steps under the
recorded loads with the run's own iteration (the adjoint's rebuild, in
the forward direction). Snapshots are `Evolution3D`'s, recording
the tangent state with its strain for the second-order run.

GATES (`TestTangentLinear3D`, lmax 8): zero-lateral — with δ = 0 at the
GLL memory points the 3D run equals the 1D run along loads and an
angle-uniform itau perturbation at 1e-7, both methods; with lateral
structure on, linear in the loads at 1e-8 against two coupled runs; the
discrete run along a log-viscosity direction p is the exact derivative
of the coupled forward (central differences with the operator's δ moved
by ε p converge at second order, Richardson to 1e-6), and the continuous
run pairs with the continuous 3D kernel ⟨K_δ, p⟩ at the commute level
(< 10 %); the second-order run along loads and log-viscosity directions
(with itau_uv) is the derivative along u of the first-order run along v
at the moved operator (second order, Richardson to 1e-6), symmetric at
1e-9, and zero without a viscosity perturbation.

## 11. The 3D second-order adjoint

`AdjointSolver3D::SecondOrderKernels`: the continuous sweep of §6a runs
as one core (`Sweep`) that carries, when given v's recorded 3D
tangent run and δitau(x), a second carried state — the derivative of
the adjoint state along v — beside the first. The derivative state is
stepped by the same reversed coupled machinery, its step driven by the
continuous first-order drive of the adjoint trajectory
(`TangentLinearSolver3D::FirstOrderDrive` with the primed pre-step
adjoint state and the post-step one, through `OpenStep::AddMemory`);
both states jump at an observation by the instantaneous adjoint strain,
over itau(x) and by δitau(x); and the pointwise pairing is
differentiated — the derivative state against the forward d − m, the
adjoint state against v's tangent d − m, less (δitau/itau) times the
pairing itself — with the same null-basis metric. The result is
`KernelDerivatives3D`: the thickness and initial-sea-level views'
derivatives, the itau kernel's derivative on the grid and its spherical
reduction, and the log-viscosity kernel's derivative
−(δitau K + itau dK), the Hessian-vector product of the log
parameterisation. `FirstOrderKernels` is the same sweep with no
tangent run, so the first-order gates guard it too.

GATES (`TestSecondOrderAdjoint3D`, lmax 8): zero-lateral — with δ = 0 at
the GLL points the kernels and the derivatives (thickness views,
spherical reductions of the itau kernel and its derivative) equal the
1D continuous second-order sweep's at 1e-6 along a direction
angle-uniform in itau; lateral — the carried kernels equal the
first-order sweep's at 1e-10, and ⟨H u, w⟩ against ⟨H w, u⟩, against
⟨g, F''[u, w]⟩ from the 3D second-order run (the log parameterisation's
uv term accounted for), and against central differences of the
continuous kernels along u, all agree below 10 % at 9 nodes and improve
at 17, on the η = 1e22 mantle — the continuous formulation's commute
level, as the first-order 3D gates. The machine-grade version is the
discrete 3D second-order sweep (§14); the first-order discrete 3D
transpose is §12.

## 12. The discrete 3D transpose as built

`AdjointSolver3D` with `Options::method = Discrete`
(`AdjointDiscrete3D.cpp`): the exact transpose of the coupled
lateral-viscosity time-stepped map, the §6b design realised with no new
low-level transposes. The load-bearing duality holds as stated there:
under the energy pairing ⟨S, T⟩ = Σ_node Wq r² ∫dΩ [12 S₀T₀ +
8 Re(S₋₁T̄₋₁) + 4 Re(S₋₂T̄₋₂)] on grid tensors and the Hermitian m ≥ 0
pairing (m > 0 weighted twice) on degree columns, Entry (force assembly
of a tensor's analysis) is Exitᵀ (synthesis of the strain of a
solution), the pointwise weight W and the reference systems K are
self-adjoint, and the sea level equation has its exact adjoint. So the
transposed pass is the forward pass with the force and strain legs
swapped and the SLE solve replaced by its adjoint:

    z ↦ A( d_e† + W ⊙ Exit( K⁻¹ [ Entry(z) + Surfᵀ SLE_adj( Uᵀ Entry(z), g ) ] ) ),

run on the spectral dual z = A(y) by the step's own iteration
(`OpenStep::Converge` with a start, opened on the forward step's start
state for its weight field, effective systems and unit responses),
with the measurement duals g a constant source in every pass as the
direct load is in the forward; one more pass at the converged dual gives
the byproducts. Per reversed step: d_e† = d† + Σ β_k(x) m†_k (the finish
transposed); the transposed fixed point; the load dual f† =
directLoadDual + synth(σ†)/r²; m̃†_k = m†_k + μ_k Exit(x†) (the fixed
memory force transposed); the itau kernel as the energy pairing of m̃†
with dt (dE m + dA d⁰ + dB d¹) — the tangent-linear drive's derivative —
as an angular density (Wq r² folded in); and the recursion m† = E m̃†,
d† = Σ A m̃† pointwise. The start is the transposed instantaneous elastic
solve on the unrelaxed systems. The column duals carry no halving or
doubling: the Hermitian column pairing matches the solver's real-field
coefficient pairing (unlike the 1D real columns, which halve and double).
Distributed over radius (§17). Its second-order form is
§14 (the continuous one is §11).

GATES (`TestDiscreteAdjoint3D`, lmax 8, lateral structure on): the
pairing identity with the discrete tangent-linear run at 1e-8 for both
schemes, along loads (node 0 included) and a log-viscosity direction,
each alone and together, and a stride-2 trajectory giving the same at
1e-9; zero-lateral at the GLL points equal to the 1D discrete sweep at
1e-8 (load duals and the spherical reduction of the itau kernel); the
machine-grade Taylor test — |J(δ + h p) − J − h⟨K_δ, p⟩| falls at second
order (rate in 1.9–2.1) at every h from 0.2 to 0.025, Richardson on
central differences landing at 1e-7; and the continuous sweep's gap to
the discrete one falling by > 2.5 as the steps halve, in the loads and
in the log-viscosity kernel (the check of the time discretisation, §1,
in 3D).

## 13. Material derivatives and kernels

New design, not a port (pyslfp has no material derivatives): the static
elastic problem (§13a), its lateral Born response (§13b), the
time-stepped tangents in the moduli (§13c, §13d), their exact transposes,
the material kernels (§13e, §13f), the continuous kernels (§13g) and the
static kernels (§13h).

### 13a. The static elastic problem

The parameters are the material's moduli at its nodes —
`MaterialDirection`: δκ, δμ on the mesh's (element, gll) nodes in
material units, either empty — and the derivative is taken of the
computed map through `Material::Shifted(d, ε)`, whose linearisation
`Material::Variation(d)` is exact: the moduli's variation (A, C, F, L, N
from δκ, δμ). The density, gravity and the fluid flags are held; the
density is not a parameter of the derivatives (§13l).

`DegreeOperator` assembles its band from a `Coefficients` set
(`AssembleBand`), the base one from the material and the variation's
from `VariationCoefficients`: the elastic block's moduli replaced by
their variations, with the gravitational couplings, the fluid
stratification, the fluid-solid interface terms, the exterior closures
and the point shear (fixed branches) carrying none.
`ApplyVariation(dm, x)` is (dK·dm) x; the forcings do not depend on the
moduli. `ComputeLoveNumberDerivatives(material, lmax, direction)` solves
dx = −K⁻¹ dK x per degree and forcing and reads the surface off as the
numbers are; it returns a LoveNumbers-shaped table of derivatives
(surface_gravity, radius and G zero).

GATES (`TestMaterialDerivatives`, prem8 at lmax 6): the variation
operator against central differences of K(m ± εd)x, Richardson to 1e-9
at every degree, and a direction of the wrong size refused; the
Love-number derivatives against central differences of the shifted
material's numbers for directions in μ, κ alone and together,
Richardson to 1e-7 per kind of number for every channel, degree and
axial number, the differences second order where resolved; the
reciprocity identities differentiated (g δh_φ = δk_u and the two others)
at 1e-10; an empty direction gives zeros.

### 13b. The lateral Born response

`ElasticPerturbation3D` (`ElasticPerturbation3D.hpp/cpp`): the
first-order (Born) response of the static gravitating elastic problem to
lateral δμ(x) and δκ(x) over the grid values. A lateral δμ enters the
bilinear form exactly as the lateral viscosity weight enters a step —
a pointwise weight on the deviatoric strain, fed back through the force
assembly (`Entry(δμ ⊙ Exit(x))`), which in the angle-uniform limit is
the 1D shear block exactly — and δκ weights the dilatation r div u =
X + Y, carried by a scalar companion of `GridCoupling`
(`SynthesiseScalar`/`AnalyseScalar` in the spin-0 layouts,
`AddDilatationOfSolution`, `DilatationForce`: the bulk term κ(X+Y)(X'+Y')
of the degree form, pointwise in angle). Then δx = −K⁻¹ (dK) x with the
reference (radial, unrelaxed) systems, spheroidal and toroidal: a
lateral perturbation couples the degrees by the selection rules of the
products and couples spheroidal to toroidal motion. The nodes are the
GLL nodes of every solid element above the one holding the centre (that
element's first GLL node is the centre itself, where the spherical
strain components, which divide by r, are not evaluated: §13k, "The
element holding the centre"); they come from a throwaway
Maxwell rheology over the isotropic Voigt average of the material (a
branch may not be active where the material is anisotropic), while the
reference systems and the perturbation's meaning — an isotropic
increment of whatever anisotropy the material has — are the material's
own. `SolveLoad` gives the reference response to a surface mass load by
its m ≥ 0 coefficients; `Force`, `Respond`, `Solve`, `Surface`,
`SurfaceToroidal` and the Hermitian `Pair` complete the interface. No
δρ (§13l).

GATES (`TestElasticPerturbation3D`, prem8 at lmax 6): angle-uniform
δμ(r) and δκ(r) reproduce §13a's exact 1D Love-number derivatives
(U, V, φ at the surface, every degree) at 1e-8 through the whole
pipeline, with no coupling into other orders or toroidal motion; the
perturbation force is symmetric under the Hermitian pairing at 1e-10 for
lateral μ and κ directions, alone and together; a single-harmonic
direction f(r) P_L^M cos Mφ couples a degree-l order-m load only to the
degrees |l − L|..l + L and orders m ± M, spheroidal and toroidal alike,
with the toroidal response present. The lateral response beyond the
angle-uniform limit rests on the pipeline (gated through the viscosity
path) and these structural identities; its independent check is the
cross-code comparison with AdGIA, which is what this is for.

### 13c. The material tangent, 1D

`TangentPerturbation::material` (a `PointMaterialDirection`: δκ and δμ
at the memory points) runs the 1D
tangent-linear along a direction of the moduli with the branch moduli as
fixed fractions f_k = μ_k/μ_U of the unrelaxed modulus at the points and
the relaxation times held (compose with `itau` for a viscosity held
instead). A perturbation of the moduli is a field at the memory points
that enters by being integrated against the strain and the dilatation
there, the rule of the 3D stepper's lateral moduli; the forward problem
takes the same perturbation as `ViscoelasticOperator::Options::moduli`,
and the tangent is its derivative. At the GLL points the points are the
nodes and the direction is a nodal one (`AtMemoryPoints` carries a nodal
direction to the points). The material enters a step in two places and
the start in one, and each is a force on the step's own systems:

- the effective system K_eff of a step (whose shear at the points is
  μ_U − Σ_k μ_k (1 − w_k) with w_k the scheme's weight there): its
  variation along δμ_eff = δμ (1 − Σ_k f_k (1 − w_k)), δκ at the points,
  applied to the FULL base solution x_n of the step
  (`DegreeOperator::ApplyPointVariation`), negated;
- the memory force Σ_k μ_k ⟨m̃_k, ·⟩ of the base decayed memory m̃ =
  e m + α d₀ with δμ_k = f_k δμ in place of μ_k;
- the start's unrelaxed solve: −K_U⁻¹ (δK_U) x₀ about the base start
  solution, whose surface values close the start's sea level solve and
  whose strain is added to the primed start strain (the trapezoid's
  α d₀ reads it).

Both step terms go in through `ViscoelasticStep::AddForce` (a force per
degree added to the memory force before the effective solve; the cached
memory solutions invalidated), so the step's memory surface — what the
sea level closure already reads — carries them, and the closure needs no
Love-table derivative: acting on the full solution, the first term holds
the material derivative of the load response too (adding δh_eff σ as a
response channel would double count). The scheme weights depend on itau
and dt only, so the memory recursion is unchanged.

Rotational feedback, axial or full, needs nothing further for the
moduli. The base solution the force acts on holds the response to the
centrifugal potential, so the force carries the tidal Love numbers'
derivative as it carries the load numbers'; and the rotation block of
the sea level solve closes about the material response through the
channels it reads for the memory — the degree-2 surface displacement
and potential and the degree-0 inertia moment, the start's unrelaxed
response included — with the base model's numbers in its closed form.

GATE (`TangentLinear.MaterialDirectionMatchesDifferences`): along
a smooth direction in κ and μ on the solid, both schemes, the tangent
run matches central differences of the coupled run on the shifted
material — `Material::Shifted` with its rheology rebuilt at the same
fractions and times — at second order (ratio in (3, 5)) with Richardson
landing at 1e-6 of the derivative. The steps are 1% and 0.5% of the
moduli, wide enough for the truncation error to stand above the solves'
1e-12 noise: the functional moves by parts in 1e-12 per unit step, and
at narrower steps the ratio check reads noise.
With rotational feedback on, axial and full, the same holds along the
moduli (the same test and its 3D twin), the discrete kernels stay the
tangent's transpose (`Adjoint.MaterialKernelsAreTheTangentTranspose`,
`DiscreteAdjoint3D.MaterialKernelsPairWithTheTangent`), the continuous
ones converge to them, and the second-order blocks are exact and
symmetric (`SecondOrderAdjoint.MaterialBlocksAreExactAndSymmetric`,
`SecondOrderAdjoint3D.DiscreteMaterialHessianIsExactUnderRotation`).
`TangentLinear.MaterialDirectionMatchesDifferencesAtGaussPoints` holds
the same at the default memory points, along a direction in κ and μ at
the Gauss points, against differences of the run with `Options::moduli`
perturbed.

### 13d. The material tangent, 3D

`TangentPerturbation3D::material` (`MaterialDirection3D`: δκ(x), δμ(x)
over the coupling's grid values, material units) runs the 3D tangent-linear along a lateral direction of
the moduli, with the same meaning as in 1D (branch moduli as fixed
fractions, relaxation times held). The direction lives on the memory
nodes, which is where the operator's lateral structure lives: an
elastic region to be perturbed laterally takes a branch of negligible
inverse relaxation time, so that it carries memory nodes. The three
terms are the 1D ones through the grid pipeline, entering the opened
step by `OpenStep::AddForce` (an analysed tensor joining the fixed
memory force, and per-degree spheroidal forces added after the
assembly), so the within-step fixed point and the sea level closure
carry them as they carry the memory force:

- the effective system's variation on the full base solution — the
  Born force of §13b with the weight δμ_eff(x) = δμ(x)(1 − Σ_k f_k
  β_k(x)), the step's OWN lateral weights β_k(x) (a lateral viscosity
  makes the effective modulus's variation lateral even for a radial δμ)
  — on the base node's grid strain (the cursor's state carries it; a
  step's converged solution satisfies the unrelaxed equilibrium with its
  new memory, so the base solutions a bulk direction needs come from one
  unrelaxed solve under the recorded loads);
- the memory force of the BASE step's decayed memory m̃ = e m + α d₀
  with δμ_k = f_k δμ: a base step opened on the cursor's previous state
  (the tangent's own step holds the tangent's memory, which is not it);
- the bulk term through the scalar pipeline (`DilatationForce` of
  −δκ times the dilatation of the base solutions);

and the start's unrelaxed response −K_U⁻¹ (δK_U) x₀ by the same
pipeline, closing the start's sea level solve and seeding the start
strain. Rotational feedback as in 1D (§13c): the closure reads the
material response's surface values and inertia moment, the start's
included.

GATES (`TestTangentLinear3D`): `MaterialDirectionMatchesDifferences`
— with lateral viscosity ON, along an angle-uniform direction in κ and
μ on the memory nodes, the 3D tangent matches central differences of
the coupled 3D run on the shifted material at second order with
Richardson at 1e-6, both schemes (the lateral β_k(x) factor exercised),
and `MaterialDirectionMatchesDifferencesAtGaussPoints` the same at the
default memory nodes against the run with `Options::moduli` perturbed;
`ZeroLateralMaterialMatchesThe1DRun` — with δ = 0 the 3D run equals the
1D material tangent at 1e-7, both schemes, at the GLL and at the Gauss
points, and a lateral direction of the same angular mean is a different
run. The
lateral part of the shear term is the pipeline of §13b and of the
viscosity path; its exact gate is the pairing identity with the discrete
3D material kernels (`DiscreteAdjoint3D.MaterialKernelsPairWithTheTangent`,
§13f).

### 13e. The material kernels, 1D

`AdjointSolver::Options::material_kernels` has the discrete sweep form
`Kernels::kappaKernel` and `muKernel` — the kernels in κ(r) and
μ(r) over the full mesh in material units, per unit volume (§4): the
exact transpose of
`TangentPerturbation::material`. The sweep already solves, per step,
the transposed effective system for the dual y_n of the step's full
solution (the duals of the next memory through the strain transpose,
the sea level closure's response duals, the inertia dual); the material
enters the tangent as forces on that solve, so the kernels are the
pairings of y_n with each force's material derivative:

- −(dK_eff) x_n: the gradient of the degree form in the moduli at the
  memory points for the pair (y_n, x_n), `DegreeOperator::PointGradient`
  — the dilatation product for κ and the deviatoric-scalar form the
  memory force uses for μ, at each point with its quadrature weight,
  over EVERY element of the system — the shear scaled by the effective
  factor 1 − Σ_k f_k (1 − w_k) at the points; x_n from one unrelaxed
  solve under the node's memory and recorded loads;
- the memory force of the base decayed memory m̃ with δμ_k = f_k δμ:
  Σ_k f_k W_q Σ_s w_s d_s(y_n) m̃_{k,s}, on the memory window;
- the start: the transposed unrelaxed solve under the start duals and
  the start's response duals, paired with the base start solution.

The gradient lives on `DegreeOperator`, not on the operator's per-degree
workspace, because that workspace covers the memory window only (the
relaxing elements) and the kernels span the elastic lithosphere too.
Discrete exact, continuous at the commute level (below, §13g); needs the
trajectory; formed under any rotational feedback; the bulk pairing costs
one unrelaxed solve per step. The kernels in κ and μ are held at the memory
points as samples of the kernel per unit volume, and pair with a
`PointMaterialDirection` by `VolumePairing` (§4); their jumps across
the mesh's interfaces are kept, since no point is shared between
elements. As functions the kernels of the two memory point sets agree
to 1e-6 or better (`Adjoint.MaterialKernelsAreFunctionsOfRadius`).

GATES (`TestAdjoint`): `MaterialGradientIsTheTransposeOfTheVariation`
— for random solution pairs of every degree the gradient paired with a
random direction equals the pair's product with §13a's variation
operator at 1e-12; `MaterialKernelsAreTheTangentTranspose` — the
kernels paired with a smooth direction in κ and μ equal the material
tangent's objective change at 1e-10, both schemes, with the memory
points at the GLL nodes and at the Gauss points, and at the GLL points
central differences on the shifted material land on them at second
order with Richardson at 1e-6; the refusals.

### 13f. The material kernels, 3D

`AdjointSolver3D::Options::material_kernels` has the discrete 3D sweep
form `Kernels3D::kappaKernel3D` and `muKernel3D` — the kernels in κ(x)
and μ(x) in the itau kernel's layout, per unit volume (§4) — and their
means over the sphere, the
exact transpose of `TangentPerturbation3D::material`. The transposed
fixed point already yields, per step, the dual solution y_n of the
step's full solution (the force of the end strain's dual, the lateral
weight's transpose and the closure's response duals through the
effective systems); the material tangent adds its forces to that very
solve (an analysed tensor into the fixed force, per-degree bulk
forces), so the kernels are the pairings of y_n with the tangent's grid
tensors:

- the shear: W_q r² [−(1 − Σ_k f_k β_k(x)) C(e_n, d_n) + Σ_k f_k C(e_n,
  m̃_k)] with C the (12, 8, 4) energy contraction the itau kernel uses,
  e_n the dual solution's grid strain, d_n the recorded forward strain
  and m̃_k the base decayed memory the opened step holds;
- the bulk: −W_q D(y_n) D(x_n), the dilatations through the scalar
  companion (`AddDilatationOfSolution`, `SynthesiseScalar`), x_n from
  one unrelaxed solve under the node's memory and recorded loads;
- the start: the transposed unrelaxed solve under the start duals and
  the start's response duals (one more solve than the ice kernels
  need), paired with the base start solution.

Discrete exact, continuous at the commute level (below, §13g); needs the
trajectory; distributed (§17).

GATES (`TestDiscreteAdjoint3D`): `MaterialKernelsPairWithTheTangent` —
with lateral viscosity on, the kernels paired with LATERAL directions in
κ and μ (a radial profile times caps and harmonics) equal the 3D
material tangent's objective change at 1e-8, κ alone, μ alone and
together, both schemes: the exact gate of the lateral shear term of
tangent and kernel alike; the spherical reductions pair with
angle-uniform directions as the full kernels do; the refusals.
`ZeroLateralMaterialKernelsMatchThe1D` — with δ = 0 the spherical
reductions equal the 1D discrete material kernels at the memory nodes
at 1e-7, both schemes.

The derivative machinery in the material is closed in both dimensions:
Love-number derivatives (§13a), the lateral Born response (§13b), the
time-stepped tangents (§13c, §13d) and their exact transposes, the κ and
μ kernels (§13e, §13f), each gated against the previous rung.

### 13g. The continuous material kernels, 1D and 3D

`material_kernels` works under `AdjointMethod::Continuous` too. The
continuous adjoint problem has the forward form, so at a node the
adjoint displacement λ is the unrelaxed solve under the adjoint
memory and the closed adjoint load (what `smoothEquilibrium` closes),
and the true multiplier is λ times the itau kernel's units constant
(and, in 1D, its Hermitian column weights: m = 0 real once, its
imaginary part not at all, m > 0 twice — derived from the itau pairing,
whose calibration fixes the multiplier). The kernels are then

- the time quadrature, with the itau kernel's trapezoid half-weights on
  both sides of each node, of −(the material gradient of λ and the
  forward solution x_n) + (the memory force's variation with the FULL
  forward memory m_n): no effective factor and no decayed memory, those
  being the discrete form's;
- plus, at each observation, the lumped pairing of the instantaneous
  adjoint solution (the jump's, kept from `applyJump`) with x_n and m_n,
  without a time weight: the delta the adjoint displacement carries
  there, whose jump in the adjoint memory the viscosity path already
  took.

In 3D the same with the grid strains' energy contraction and the
dilatations through the scalar companion (as the discrete 3D kernels),
no column weights (the grid pairing is the full real one). Cost: one
more unrelaxed solve per node than the discrete form.

GATES: `Adjoint.ContinuousMaterialKernelsAgreeAndConverge` and
`DiscreteAdjoint3D.ContinuousMaterialKernelsAgreeAndConverge` — ONE
problem at every resolution (a ramp of fixed fields, measurements at the
same times), against the exact discrete kernels paired with a direction
(lateral in 3D, lateral viscosity on): at 2.5 kyr steps the gap is 0.9%
(trapezoid) and 5.6% / 0.5% (backward Euler, 1D / 3D), falling under
refinement at the schemes' orders (1D trapezoid 0.95% → 0.24% → 0.06%
over 5, 9, 17 nodes; backward Euler 5.6% → 2.9% → 1.5%).

### 13h. The static kernels

For the static elastic problem the kernel is the start term alone.
`ComputeLoveNumberKernels` (`LoveNumbers.hpp`): per channel (load force and potential pieces,
tangential, tide) and degree, dh_l/dκ(r), dh_l/dμ(r) and the l and k
alike, nodal over the full mesh, by the unit surface duals' solutions
paired with the forcings' through `DegreeOperator::MaterialGradient`;
gated against `ComputeLoveNumberDerivatives` along a direction at 1e-10
for every channel and degree
(`MaterialDerivatives.LoveNumberKernelsPairToTheDerivatives`). In 3D
a measurement's kernels in κ(x) and μ(x) are those of the evolution
asked for one solve (§19, *The elastic problem*), through
`LateralKernelSolver` (§21): about a radial body its radial route,
under lateral moduli the 3D sweeps; with an ocean, the rotational
feedback, sources, every measurement and either point set.

Beneath that route, `ElasticPerturbation3D::ForceTranspose(y, x)` is the
transpose of its `Force`, per unit volume on its grid: with y the solve
of a surface measurement's dual (`SolveSurfaceDual`, or `SolveLateral`
under a lateral background) and x the forward solution, minus it is
that measurement's kernels in the static problem without an ocean or
rotational feedback. It is not a route of its own but the route's
independent check (§19, §21). It is gated as the transpose of `Force`
under the Hermitian pairing at 1e-10 and by the dual solve being the
adjoint of the surface readout
(`ElasticPerturbation3D.ForceTransposePairsWithTheForce`), and
under a lateral background with both solutions from the full solve
(§13j) against central differences of the measurement at 1e-5
(`KernelsUnderLateralStructureMatchFiniteDifferences`).

### 13j. The full 3D static solve

The laterally heterogeneous static problem itself, (K_ref + dK) x = f,
not only its Born linearisation: `ElasticPerturbation3D::SolveLateral`.
The method is that of Maitra & Al-Attar (2019) and Myhill, Maitra &
Al-Attar (2026) for Poisson's equation in aspherical planets, carried
to the elastic operator: the full operator's action is matrix-free —
the reference degree systems' `Apply` (banded, per degree and order)
plus the lateral force of §13b through the grid pipeline (`Apply(x,
δμ, δκ)`) — and the reference solve K_ref⁻¹, block diagonal over the
degrees with the factorisations already held, is the preconditioner.
The operator is symmetric (§13b's gate) and, for perturbations that
leave the moduli positive, definite, so the iteration is preconditioned
conjugate gradients, started from the reference response x₀ = K_ref⁻¹
f; the first preconditioned residual is then the Born correction, so
the Born response is the method's own first step and the iteration
continues to the exact answer. The toroidal degree-1 rigid rotations,
which the pinned reference solve removes and the lateral force never
excites, stay out. The residual reported is the preconditioned energy
norm relative to the forcing's, ⟨r, K_ref⁻¹ r⟩/⟨f, K_ref⁻¹ f⟩ under the
Hermitian pairing; `SolveInfo` carries its history, the count and a
flag should a curvature come out non-positive (the operator pair then
not definite, the iteration continued on magnitudes).

The forcing is any `Forces` pair: `LoadForces` (a surface mass load),
`TideForces` (an external potential of every degree from 2 through
`DegreeOperator::Tide(power)` — the static body tide of a laterally
heterogeneous body; `sl3d solve` solves that problem by the stepper,
§13k),
`SurfaceDualForces` (a measurement's dual), or their sum, with toroidal
forces allowed. Because the operator is symmetric, the solve of a
measurement's dual forces under the lateral structure is the adjoint
solution at that structure, and the transpose of `Force` on it and the
lateral forward solution, negated (`-ForceTranspose(y, x)`), is the
measurement's gradient in μ(x) and κ(x) at the laterally heterogeneous
model, discrete-exact: the kernels the route gives there (the 3D sweeps
of a one-node `Evolution3D` on the same moduli, through
`LateralKernelSolver`, §19, §21), which it checks.

GATES (`TestElasticPerturbation3D`, prem8 at lmax 6): the converged
solution's full, unpreconditioned residual at 1e-9 of the forcing; its
departure from the reference-plus-Born response second order in the
contrast (halving a 10% lateral δμ, δκ quarters it: ratios 4.0
spheroidal, 3.9 toroidal), the first-order check against §13b; an
angle-uniform 10% shift of an isotropic reference (the Voigt average of
PREM — the shifted material must keep the reference's anisotropy, so the
reference is made isotropic) equals the exact 1D solve on the shifted
material at 1e-9 at every degree and order of a load and of a tide
(every degree from 2),
with no toroidal motion; the kernels under a lateral background (10% in
μ and κ) paired with a lateral direction equal the central difference of
a surface potential measurement through the full solve at 1e-5 (the
difference's own error). Convergence to 1e-10 against a degree-2 shear
pattern of 5, 10, 20 and 40 per cent of the local shear modulus takes
6, 7, 9 and 15 iterations, geometric throughout; the solve of the
10% kernel gate converges in about 20. Cost on 16 threads (the
disabled `TimingAtHigherDegree`, `SL3DPP_LMAX`): a 10% degree-2
pattern and a cap-like load at lmax 16, 32 and 64 take 0.4, 2.4 and
15 s for six iterations, the lateral force (the grid transforms) the
whole of it; the reference solve is 0.07 s at lmax 64. Realistic
contrasts (10% or less) are well inside what the reference
preconditioner handles, as the papers found for Poisson's equation.

Not built: aspherical geometry by particle relabelling (the papers'
mapping ξ, which turns the geometry into laterally varying tensorial
coefficients of the same form), lateral
δρ (the hydrostatic background does not admit it, §13l), and
distribution of `SolveLateral` itself (OpenMP over the grid pipeline
only; the static problem over MPI ranks is the stepper's, below and
§13m).

The program: `sl3d solve` (apps/SolveCommand.cpp) runs this problem in
its static problem without an ocean and without rotational feedback, on
a model file's lateral moduli (io::ModelModuli: the relative
perturbations of its `mu` and `kappa` about their lateral means or a
stored radial reference), as the zero-length step of the stepper on a
branchless rheology (§13k; its combined fixed point, or the nested
closure of §13n), distributed over radial segments, with the radial
model's response beside it; it writes the surface fields, the surface
coefficients in the benchmarks' exchange form, and the response at the
memory nodes (the stepper's conjugate-gradient elastic solve under the
static solve's loads) as a model file (io::WriteNodeFields,
io::ResponseFieldsAtNodes) for planetmodel's cutaways. ExampleStatic3D
and `configs/no_ocean.json` run it on the synthetic 3D elastic Earth the
build writes (MakeSyntheticModuli, io::WriteModuli). In the quasi-static
problem (a `times` block) the same command steps the viscoelastic
response through time instead, its first node this problem's solve
(docs/sl3d-configs.md).

### 13k. Lateral moduli in the 3D stepper

The same lateral δμ(x), δκ(x) in the quasi-static viscoelastic
problem: `Viscoelastic3D::Options::moduli`, relative perturbations
sampled like the log-viscosity — μ_U(x) = μ̄_U(r)(1 + s(x)) with every
branch modulus scaled alike (fractions and relaxation times held, the
material derivatives' convention), κ(x) = κ̄(r)(1 + b(x)). With them
set the coupling carries every solid element where they are non-zero
beside the viscous ones (`Viscoelastic3D::ActiveElementsFor`,
`GridCoupling::Options::active_elements`; an elastic element's memory
is inert), the element holding the centre included at the Gauss points
(below), so a perturbation can live in the lithosphere or the inner
core, down to the centre, as well as the Maxwell mantle. The fluid is
not covered: the
fluid form (κ, ρ, g; no deviatoric strain) has no grid pipeline, so a
lateral κ in the outer core is not representable.

The step is one fixed point (§5 of lateral-coupling.md): the
lateral moduli join the lateral viscosity in its pointwise weight. The
effective shear of the shifted body is (1 + s)(μ̄_U − Σ_k μ̄_k β_k(x)),
so the weight gains −s(x) μ̄_eff(x) with μ̄_eff(x) = μ̄_U − Σ_k μ̄_k β_k(x)
the reference moduli under the LOCAL weights (`StartStep`), the fixed
memory force gains the factor (1 + s(x)) on every branch modulus, and
m⁺ = m̃ + β d is as before (β is the relaxation time's). The bulk term
is new in kind: a lateral δκ weights the dilatation, which the
deviatoric iterate does not carry, so `SpectralTensor`/`TensorField`
carry an optional scalar companion (`scalar`: the dilatation's
coefficients and grid values), which `Synthesise`/`Analyse` carry when
present, `StrainOf` fills (`AddDilatationOfSolution` beside the strain),
`Respond` weights by −δκ(x) and assembles through `DilatationForce`
(`Viscoelastic3D::SpheroidalForces`), and the mixer and residual treat
as a fourth block. A strain cache under a lateral bulk modulus must
carry it (a cache without it is rebuilt by a solve); the serialiser
writes it.

The UNRELAXED solve (a step's start without a cache, `SolveElastic`,
`ElasticStrain`, `PrimeStrain`) is laterally heterogeneous, so it
is the preconditioned conjugate-gradient solve of §13j on the
unrelaxed reference systems — shared code, `detail/LateralSolve.hpp`,
which `ElasticPerturbation3D::SolveLateral` also uses — with the
memory and channel forces as the right-hand side
(`LateralElasticSolve`, `AddChannelForces`; `elastic_rtol` 1e-12;
`LastElasticSolve` reports it). Where the material is anisotropic the
perturbation is an isotropic increment of its Voigt shear; at the
memory points the reference's unrelaxed shear is the interpolated
long-term modulus plus the branches' at the points (`UnrelaxedShearAtNodes`).

GATES (`TestLateralModuli3D`, PREM's Voigt average with a Maxwell
mantle, lmax 6): smooth 10% shifts of shear and bulk over the whole
solid above the centre element at the GLL points equal the 1D stepper
on the shifted rheology at 8e-12 (U, V, φ at every degree and order,
no toroidal motion), both schemes, two steps from rest; at Gauss
points a per-element constant 10% shift of the Maxwell region (where
every shear term is a point term) is exact likewise at 9e-12; the
unrelaxed solve of a zero state under a load with angle-varying shear
and bulk equals the static lateral solve on the same fields at 1e-9,
toroidal motion included; with lateral moduli and lateral viscosity on
together the step self-converges at second order (ratio 3.98). The
within-step passes grow from 2 (zero lateral) to about 10 at 10%
moduli contrast — the lateral elastic term is a stronger coupling than
mild viscosity and contracts under Picard at roughly 0.1 a pass; the
step by conjugate gradients (§13n) is the alternative.

**The element holding the centre.** At the Gauss memory points the
coupling carries it like any other element: under lateral moduli where
they are non-zero there or where it relaxes, under a lateral viscosity
where it relaxes. What divides by r is the spherical form of a tensor
at a memory node: the five deviatoric strain components are
combinations of U, V, W and their r ∂r over r (`DeviatoricComponents`,
`AddStrainOfSolution`), an internal-variable state's components its
scalars over r (`ComponentsFromScalars`, `AddScalarState`), and the 3D
kernels are densities, the accumulated gradient over the node's volume
weight w r². The dilatation r div u = r ∂r U + 2U − k²V does not
divide, nor do the force rows, which take the factor of r back (the
pointwise rows under jac·w·r, the derivative rows with r²), so that the
assembled force is the 1D degree form's, whose scalars never divide.
r = 0 is a regular singular point: the regular solution makes the
numerators vanish with r (U and V go as r^(l−1) at degree l, and the
rigid translation at l = 1, U = V, carries no strain), so the
components are bounded and the spherical form is only a change of
variables away from the radial one. The Gauss points are interior (the
innermost at 0.069 of the element's width for five GLL nodes), so no
point at r = 0 is ever evaluated, and the division by r and the
multiplication by r that undoes it cost rounding alone: with zero
lateral moduli a run equals the run without them to rounding on bodies
viscous to the centre, and an angle-uniform shift reaching the centre
equals the 1D stepper to rounding (gates below). At the GLL points the
element's first node is the centre itself, where the components are
0/0: the coupling never carries that element (its memory is inert
where it does not relax; a lateral perturbation of the moduli there
is not applied), and a coupling over a range holding it is refused, by
name, where it relaxes (`GridCoupling`'s constructor), lateral
viscosity or lateral moduli alike; the Gauss points, the default, take
every such body. `ElasticPerturbation3D`, whose nodes are GLL nodes,
starts above that element.

GATES, on bodies viscous to their centre where not said otherwise (lmax
6 for the forward gates, 8 for the derivatives; the default Gauss
points):
`Lateral.TheElementHoldingTheCentreIsCarriedAtTheGaussPoints` — on the
Maxwell ball the Gauss points carry that element under every rule (the
rheology's, `all_solid`, a mask), none of its nodes at r = 0, and the
GLL points are refused there by name, a range above it built; on prem8
(an elastic inner core) the GLL points carry every solid element above
it and the Gauss points every solid element;
`LateralModuli3D.ZeroModuliOnBodiesViscousToTheirCentreAreTheRunWithout`
— lateral moduli set to zero equal the run without them on the
homogeneous Maxwell ball and on PREM with a Maxwell inner core, four
steps from rest, at 0 (Sdirk2) and 1.2e-13 and 3.8e-13 (trapezoid), the
unrelaxed solve at 4.0e-12 and 3.2e-12;
`LateralModuli3D.ShiftsReachingTheCentreMatchThe1DStepperOnABallViscousToItsCentre`
— on the Maxwell ball a shear shift constant per element (25% in the
element holding the centre) equals the 1D stepper on the shifted
rheology at 3.9e-12 (both schemes; the unrelaxed solve 4.6e-12), smooth
shifts of shear and bulk the 1D stepper given them at its points at
4.3e-12, the centre element's own shift moving the run by 9.9e-7;
`TangentLinear3D.DirectionsInTheCentreMatchDifferencesOnABallViscousToItsCentre`
— on the ball meshed in 1500 km elements, directions confined to the
element holding the centre (κ and μ; ln η) match central differences at
second order (ratio 4.00), Richardson within 1e-8 of the tangent, both
schemes; `DiscreteAdjoint3D.PairsWithTheTangentOnABallViscousToItsCentre`
— the discrete load, itau, κ and μ kernels pair with the tangent along
loads, ln η and the moduli concentrated about the centre to 1e-10 at
worst, both schemes, the kernels non-zero in that element;
`DiscreteAdjoint3D.ContinuousGapClosesOnABallViscousToItsCentre` — the
continuous sweep's gap falls by 3.6 to 3.8 as the steps halve;
`SecondOrderAdjoint3D.DiscreteMaterialHessianIsExactOnABallViscousToItsCentre`
— symmetric and adjoint to the second-order run to 1e-13 relative;
`TangentTest.In3DOnABallViscousToItsCentreTheViscosityIsTheDecks` —
`sl3d tangent` along a uniform ln η equals the difference of two `solve`
runs on the deck's viscosity scaled at 1.3e-6 (Maxwell ball) and 1.1e-6
(Burgers ball), the gate's tolerance 1e-5, and the kernel's uniform
pairing is the tangent's change at 1e-13;
`TangentTest.In3DAChangeOfTheModuliAtTheCentreIsAChangeLikeAnyOther` —
on prem8 under a lateral viscosity alone (an elastic inner core the run
carries no nodes in) a change of the moduli about the centre is refused
at the member, naming the depth of the centre element's first memory
point, as a change in any element the run does not carry; and over two
and three ranks
`AppsOverRanks.OnABallViscousToItsCentreSolveKernelAndTangentMatchOneProcess`
— `solve`, `kernel` and `tangent` on the ball are the one process's at
5e-12, 3e-11 and 8e-12.

**The sweeps under lateral moduli.** Three things change
for the tangent, the adjoints and the second-order sweeps, and nothing
else: every branch modulus the sweeps pair with or differentiate
through carries the shear factor 1 + s(x) (`Viscoelastic3D::
ShearFactor`: the itau kernels' μ_k m†:(d − m), the discrete
recursion's μ_k e† terms, the derivative dual's, the lateral weight's
derivative Σ μ_k dβ dh; the fractions f_k are unchanged, so every
material drive and pairing is); every unrelaxed solve of forces is the
lateral one (`SolveUnrelaxed`: the material drives' responses, the
tangent solutions, the start duals); and the discrete transposed fixed
point carries the dilatation's block (its `entry` adds the dilatation's
force where the iterate carries the scalar companion, its `exit`'s
strain carries the dilatation, weighted by −δκ(x) in the pass).

The one structural change is the INSTANTANEOUS problem. The unrelaxed
operator is laterally heterogeneous, so the elastic sea-level
problem at a node — the forward run's start, the tangent's node 0, the
discrete sweep's start transpose, the continuous sweep's jumps and
equilibria — is a coupled fixed point like a step's, not a Love-number
closure with response channels. `Viscoelastic3D::StartStep` accepts
dt = 0: a zero-length step (β = 0, m̃ = m, the effective systems the
unrelaxed ones, the lateral weight −δμ_U), whose within-step fixed point
with the sea level engine IS the instantaneous coupled solve. Under
lateral moduli the evolution's start is such a step (`Evolution3D`,
one coupled pass for the start and the steps), the tangent's
node 0 likewise with the material drive's start forces on it
(`MaterialForcing3D::ApplyStart`), the discrete sweep runs its step
transpose down to n = 0 with dt = 0 instead of its start block (the
channel duals at a lateral start are the unit load and tide vectors
paired with the lateral solution, `lateralChannelDuals`), and the
continuous sweep's jumps and equilibria are zero-length reversed steps
(`instantaneous`, `lateralJump`, `lateralJumpDerivative`, and
`reversedStep(…, 0.0, …)` for the equilibria, the derivative state's
with v's material source). Without lateral moduli the Love-number
closures are kept.

The evolution's start is also a solve of its own,
`Evolution3D::Instantaneous(load)`: the instantaneous elastic
response of the sea level problem to a direct load, the linear equation
on the background state's shorelines coupled to the rotational feedback,
by the zero-length step's fixed point (or the nested closure of §13n)
under lateral moduli and by the unrelaxed Love numbers without — the
elastic sea-level fingerprint of a laterally heterogeneous Earth, which
`sl3d solve` solves in the static problem on a lateral moduli
file. `Run`'s first node is the same code (`Start`), so `Run(times,
load)[0]` is `Instantaneous(load(times[0]), times[0])`; it is distributed
over radial segments as the run is. The check of the transverse
rotational feedback against the model's fluid limit, which a run
through time makes for its long steps, does not apply to it.

GATES under lateral moduli (10% shear, 5% bulk, angle-varying, with
lateral viscosity): the coupled evolution with angle-uniform moduli
equals the evolution on the shifted rheology at every node including
the start, rotation on, at 1e-9 (`LateralModuli3D.
CoupledEvolutionMatchesTheShiftedRheology`); the discrete sweep pairs
with the tangent-linear run at 1e-8 for loads and a log-viscosity
direction, both schemes (`DiscreteAdjoint3D.PairsWithTheTangentUnderLateralModuli`);
the discrete κ(x), μ(x) kernels pair with the material tangent at 1e-8
(`MaterialKernelsPairWithTheTangentUnderLateralModuli`); the tangent
along a lateral material direction matches central differences of the
coupled run with the background moduli shifted, Richardson at 1e-6
(`TangentLinear3D.MaterialDirectionMatchesDifferencesUnderLateralModuli`);
the discrete second-order sweep is symmetric and adjoint to the
second-order run to 1e-21 relative (`SecondOrderAdjoint3D.
DiscreteMaterialHessianIsExactUnderLateralModuli`); the continuous
sweep's gap to the discrete closes under refinement at ratio 4 for
loads and log viscosity (`ContinuousGapClosesUnderLateralModuli`), and
the continuous material
kernels and continuous second-order sweep converge likewise
(`ContinuousMaterialKernelsAgreeUnderLateralModuli`,
`ContinuousMaterialHessianConvergesUnderLateralModuli`). The
instantaneous response equals the run's first node to the bit, with and
without lateral moduli, rotation off and full, both step solvers
(`LateralModuli3D.InstantaneousIsTheRunsFirstNode`); with the moduli set
but zero it is the radial fingerprint, to rounding (1e-12; bitwise in the
test's one process)
(`InstantaneousWithoutLateralStructureIsTheRadialFingerprint`); 10%
shear and 5% bulk perturbations move it by 2.3e-2 of the sea level's
largest magnitude, the two step solvers agreeing at 1e-9
(`InstantaneousUnderLateralModuliDepartsFromTheRadialFingerprint`).

Not covered: lateral δκ in the fluid. Distribution over radial
segments: §13m.

### 13l. Density: not offered

sl3dpp offers no derivatives with respect to the density, radial or
lateral: its material directions, tangents and kernels are in the moduli
(and, through the rheology, the relaxation times), the density a fixed
part of the forward model. A lateral density would not even be
consistent with the forward problem. The equations sl3dpp solves are
those of a hydrostatically pre-stressed body: the reference gravity
g(r), the reference potential and its Hessian, and the stratification
r²(dρ/dr)/g all come from a radial density in equilibrium. A laterally
varying density is not in hydrostatic equilibrium with a radial gravity
field. What the field does — ρ(x) multiplying the displacement in the
gravitational couplings while g, the reference potential and the
stratification stay radial, the self-attraction and the pre-stress of
δρ neglected — is a modelling choice with no consistent limit inside a
spherically layered reference: making the gravity self-consistent
(carrying Φ as a variable of the whole problem, equilibrium and
first-order together) still leaves the pressure wrong, because ∇p =
ρ∇φ₀ has no solution on a spherically layered reference with a lateral
ρ; the consistent routes are an aspherical figure (particle
relabelling, not built) or a non-hydrostatic pre-stress, which is
AdGIA's new theory. So 3D density belongs on the AdGIA side, and
sl3dpp says only what it can defend: the moduli laterally, the density
radially.

### 13m. The lateral solves over radial segments

The stepper's lateral moduli run over radial segments like the rest
of it (docs/radial-partitioning.md). The grid pipeline is local to
each process's nodes — the lateral weights and the companions are per
node — so what is distributed is the conjugate-gradient start solve:
its operator's reference part assembles the whole vector from the
held shares (`MaskToShare` then the sum collective, one per degree),
applies the band and masks the product back to the share, the layout
the force shares have; its pairings are each process's part (its
solution, holding its segments and the cut nodes, against the shares)
summed by the collective (`detail::LateralSystem::sum`); the channel
forces take the unit vectors as shares. The static 3D problem over ranks
is the stepper on a branchless rheology (`Rheology::Create(material,
{})` with `moduli` set; `SolveElastic` is then the static lateral
solve); `ElasticPerturbation3D` itself stays single-process.

GATES: two and three threads-as-ranks (`RadialRanks.
LateralModuliStepsMatchTheOneProcessSteps`): the start solve's
iteration count (to within one: a residual at the tolerance in another
summation order may take one more), two lateral steps' surface
responses and memory slices, and the branchless static solve, against
the one process at 1e-9 (measured 6e-12, the conjugate-gradient solves
at 1e-12); the continuous and discrete sweeps on three
ranks with lateral moduli and the axial rotational feedback
(`LateralModuliAdjointsMatchTheOneProcessSweeps`): ice kernels and
viscosity-kernel slices at 1e-9 (measured 3e-12); the
instantaneous response on three ranks, on the Maxwell and on the
branchless rheology, both step solvers, at 1e-9 (measured 2e-16 by the
fixed point and 1.4e-12 by the conjugate gradients;
`InstantaneousResponsesMatchTheOneProcess`); under
mpiexec on two and three ranks
(`Distributed.LateralModuliStepsMatchTheOneProcessSteps`), likewise.

**Cost, and the active elements.** A step's pass on 16
threads at lmax 64 (prem8, Maxwell mantle, the `DISABLED_StepTiming`
probe of `TestLateralModuli3D`): 0.66 s with lateral viscosity alone
(1108 memory nodes), 0.96 s with a lateral shear over the whole solid
(1700 nodes: the lithosphere, crust and inner core join); the one
scalar companion of a lateral bulk modulus adds two scalar transforms a
pass, and a scalar transform costs about as much as a spin one. The
coupling therefore carries, beside the viscous
elements, only the solid elements where a perturbation is non-zero at
some memory point (`Viscoelastic3D::ActiveElementsFor`, a probe of the
whole solid sampling the functions; `GridCoupling::Options::
active_elements`), so a perturbation confined to the mantle costs no
lithosphere or inner-core nodes.

Two properties of the distribution, which the program's static problem
on lateral moduli relies on. Whether an operator has lateral moduli is a
property of the RUN, not of a rank: with the moduli confined to the
upper mantle, an inner rank's segment holds no element where they are
non-zero, its coupling no node, and its sampled arrays are empty — but
every rank must take the lateral route, since the solve is collective
(`HasLateralModuli` reports the option, not the samples;
`RadialRanks.ConfinedLateralModuliMatchTheOneProcessStatic` holds a
node-less rank to the one-process answer). And the rank cuts balance
the transform work of the coupling's nodes, which for a branchless
rheology the rheology does not know: the program weights the active
elements the moduli make (`OperatorOptions` takes them), or the
partition is degenerate and the solves stall.

### 13n. The step by conjugate gradients, and the nested closure

A 3D step is a linear system in the degree solutions,

    (K_eff − Pᵀ W S) x = f,

K_eff the reference effective systems of the (scheme, dt), W(x) the
step's pointwise lateral weight (the viscosity's Σ μ̄_k(β_k(x) −
β_ref,k) and the moduli's −s μ̄_eff, with the bulk companion −κ̄ b on
the dilatation), S the strain of the solutions, Pᵀ the assembly, and
f the fixed memory force plus any added forces and the channels'. The
operator is the 3D effective stiffness: symmetric positive definite
whatever the contrast, since the effective modulus μ_U(1+s) −
Σ μ_k(1+s)β_k(x) is positive pointwise. The within-step fixed point
(§13k, lateral-coupling.md §5) iterates it as x = K_eff⁻¹(f + Pᵀ W S x)
on the strain — Picard, Anderson-mixed — which contracts only as fast
as the lateral term is small against the reference: at 100x contrasts
and dt ≥ 2τ it takes 65–150 passes, and at dt = 10τ it fails.
`Viscoelastic3D::Options::solver = ConjugateGradients` closes the same
system by the preconditioned conjugate gradients of §13j instead
(`detail/LateralSolve.hpp`, the routine the static solve and the
unrelaxed start solve use), K_eff⁻¹ the preconditioner, from the
Picard iterate (the reference response, or a warm start): guaranteed
to converge, and in practice in fewer iterations than the fixed point
takes passes everywhere measured. An iteration is one grid pipeline
(synthesis, weight, analysis, assembly) like a pass, plus the reference
band products on the columns (hand-written, threaded over the degrees:
a BLAS call inside the OpenMP loops starts a second thread pool on the
server)
and the preconditioner's reference solve, plus the Krylov vector
algebra; at lmax 16, where the pipeline is 4 ms, an iteration costs
about 1.3 passes, and the ladder's second steps still take less time
at every contrast (lateral-coupling.md §5a''); at production sizes the
pipeline dominates both. `OpenStep::Forces` builds
f, `OpenStep::Apply` the operator, `OpenStep::Solve` the solve
(`solve_rtol` on the relative residual in the preconditioned energy
norm √(⟨r, K_eff⁻¹ r⟩/⟨f, K_eff⁻¹ f⟩), `solve_max_iterations` the
budget), `OpenStep::Close` a step under known channels either way, and
`Step` reads the option. Distributed, the operator is the §13m one
(shares assembled by the sum collective per degree, the band applied,
the product masked back; pairings summed), so the step's solve runs
over radial segments like the start solve.

**The closure outside.** The sea level equation makes the loads depend
on the solutions. The fixed point folded both into one iteration on
the strain. With the solve inside, `OpenStep::Nest` runs the driver's
closure (the sea level engine on the surface response, as before: a
`Closure` maps the CHANNEL-FREE solutions to the next channels) as an
outer iteration: pass n hands the closure the last solution less its
channels, solves the step under the new channels warm-started from the
last solution WITH the new channels (the Picard iterate, which already
carries the 1D Love-number response to the load change), and the
outer residual is that warm start's residual before the solve — how
far the last solutions with the new channels are from solving the
step, relative to its forces, in the solve's own measure. Converged
when it is at most `coupled_rtol`. The inner solves are inexact: each
reduces its start's residual by `solve_outer_factor` (1e-2) at least,
never below `solve_rtol`, so the inner work follows the outer
progress. The outer map is Picard, Anderson-mixed on the channel-free
solutions by the same adaptive rule as the strain fixed point (the
`IterationControl`'s memory and rate); in the cases measured the outer
map contracts by ~1e-3 a pass and Picard never hands over. The outer
contraction is the 3D departure of the load response from the 1D Love
numbers the engine closes with, times the ocean's feedback — small
unless the lateral structure is strong at the surface.

**Who uses what.** The evolution's steps and its lateral start (the
zero-length step, `Instantaneous` too), the tangent-linear run's steps
and start, the
continuous sweep's reversed steps and its instantaneous problems
(equilibria, jumps) all go through `Nest` under the option; the
forward regenerations in the sweeps (a kept trajectory's steps rebuilt
under recorded loads) through `Close`. The DISCRETE transpose follows
the option too. It transposes the converged linear solve, which is the
same system whichever way the forward solved it, and its operator is
the forward step's own: in the dual solutions y the transposed step is

    (K_eff − Pᵀ W S) y = b₀ + Surfᵀ SLE†(Fᵀ r, g),

b₀ the forces of the dual strain's source (and any added), F the unit
load and tide forces, Surfᵀ the surface duals placed on the surface
rows and the degree-0 moment row, and r = y less the reference response
of the surface duals — the reference response of the forces, so Fᵀ r is
the forces' contraction with the unit responses, what the transposed
fixed point hands the engine's adjoint. Under the fixed point the pass
of §13f iterates it on the dual strain; under conjugate gradients the
sea level adjoint is the outer closure on r exactly as `Nest` runs the
forward's (the solve is `OpenStep::Solve` on the same step, warm-started
from r with the new surface duals' reference response, inexact by
`solve_outer_factor`, the outer residual the warm start's, Anderson by
the control's rule), for the first dual state and the second-order
sweep's derivative state alike. Distributed, Fᵀ r is summed from each
rank's rows.

**Measured (lateral-coupling.md §5a'').** The §5a ladder at lmax 16,
the second step: at 3x, 8/13/17 iterations against 12/21/21 passes at
dt/τ = 0.5/2/10; at 10x, 12/23/34 against 20/29/50; at 100x, 23/46/90
against 26/65/fail; at 1000x, 42/83/183 against 49/138/fail. The
nested evolution at a 3x viscosity contrast with 10%/5% moduli and the
full rotational feedback (`LateralModuli3D.NestedConjugateGradient
EvolutionMatchesTheFixedPoint`, agreement to 1e-9 at every node): 5
passes and 8 iterations at the start against the fixed point's 10
passes, 7 passes and 20–22 iterations a step against 20 — the nested
closure costs about 1.4 passes' worth per fixed-point pass at this
weak contrast (each outer pass also pays one operator application for
its residual and one reference solve for its scale), and the fixed
point stays the default. Where the fixed point grinds or fails, the
conjugate-gradient step is the one to use.

GATES: `Lateral.ConjugateGradientStepMatchesTheFixedPoint` (10x,
dt = 2τ, two steps, solutions to 1e-9, and the operator applied to the
fixed point's solution returns the step's forces to 1e-9),
`LateralModuli3D.ConjugateGradientStepMatchesTheFixedPointUnderModuli`
(viscosity, shear and bulk together), the nested evolution gate above,
`TangentLinear3D.MaterialDirectionMatchesDifferencesUnderConjugate
GradientSteps` (the tangent along a lateral material direction against
central differences of nested runs, Richardson at 1e-6),
`DiscreteAdjoint3D.ContinuousMaterialKernelsAgreeUnderConjugate
GradientSteps` (the continuous sweep's reversed steps and instantaneous
problems nested, the discrete transpose nested too, the gap closing
under refinement), `DiscreteAdjoint3D.PairsWithTheTangentUnderConjugate
GradientSteps`, `DiscreteAdjoint3D.MaterialKernelsPairWithTheTangent
UnderConjugateGradientSteps` and `SecondOrderAdjoint3D.DiscreteMaterial
HessianIsExactUnderConjugateGradientSteps` (the nested transpose exact:
the kernels pair with the nested tangent and the Hessian with the
second-order run at 1e-8), and the two radial-rank gates and the
mpiexec lateral gate run the steps and the evolution on the
conjugate-gradient solver. Probe: `Lateral.DISABLED_StepSolverLadder`.

## 14. The discrete 3D second-order adjoint

`AdjointSolver3D::SecondOrderKernels` under `AdjointMethod::Discrete`
(AdjointDiscrete3D.cpp; the 3D form of the 1D discrete second-order sweep
of §9): the exact derivative of the exact gradient of §12 along a
direction v, from v's recorded DISCRETE tangent-linear run and its
δitau(x). The transposed sweep carries the derivative dual state
(dm†_k, dd†) beside (m†_k, d†), and every piece of §12 is differentiated
in place:

- the dual of the end strain: dd† + Σ_k β_k(x) dm†_k + Σ_k dβ_k(x) dh_k(x)
  m†_k, with dh_k = dt δitau_k(x) and dβ_k the weight's h-derivative
  (dB of the trapezoid's (E, A, B), −dw for backward Euler);
- the derivative state's transposed fixed point: the SAME iteration as
  the first's (reference systems, lateral weight W(x), the SLE adjoint
  with no measurement duals), with its source dde + dW(x) e, where
  dW = Σ_k μ_k(r) dβ_k(x) dh_k(x) is the derivative of the step's
  lateral weight Σ_k μ_k (β_k(x) − β^ref_k(r)) and e the first dual
  strain — the only term with no 1D counterpart in form (in 1D it is the
  effective system's derivative acting on the first sweep's solution,
  Σ_k B_h dh μ_k S y; here the reference is held and the lateral weight
  carries all of the itau dependence, so the derivative lands pointwise);
- the dual of the decayed memory, dm̃†_k = dm†_k + μ_k e', with e' the
  derivative dual strain; the itau pairing's derivative, pointwise in the
  energy contraction: ⟨dm̃†, dE m + dA d⁰ + dB d¹⟩ + ⟨m̃†, dE vm + dA vd⁰
  + dB vd¹⟩ + dh ⟨m̃†, d²E m + d²A d⁰ + d²B d¹⟩ with (vm, vd⁰, vd¹) the
  tangent run's states; and the recursion dm† = E dm̃† + dE dh m̃†,
  dd† = Σ_k (A dm̃†_k + dA dh m̃†_k);
- the start: the transposed instantaneous solve for both dual states.

Returns `KernelDerivatives3D` as the continuous sweep does: the thickness
and initial-sea-level views of the derivative load duals (no lumps, as in
1D), the itau kernel's derivative with its spherical reduction, and the
log-viscosity kernel's derivative −(δitau K + itau dK). Cost: two
transposed fixed points per node instead of one. Distributed (§17);
the material kernels and their derivatives alongside (§15).

GATES (`TestSecondOrderAdjoint3D`, lmax 8, prem8):
`DiscreteZeroLateralMatchesThe1DDiscreteSweep` — with δ = 0 at the GLL
points the kernels and derivatives equal the 1D discrete second-order
sweep's at 1e-7 (thickness views, spherical reductions of the itau
kernel and its derivative). `DiscreteLateralHessianIsExact` — lateral
structure on (δ of amplitude 0.6, η = 1e22, 9 nodes over 10 kyr, a
sea-level measurement at the end): the carried kernels equal the
first-order discrete sweep's bitwise; ⟨H u, w⟩ against ⟨H w, u⟩ agree at
1.2e-13 (gate 1e-8); against ⟨g, F''[u, w]⟩ from the discrete 3D
second-order tangent-linear run (the uv term of the log parameterisation
accounted for) at 1.5e-13 (gate 1e-8); and central differences of the
discrete first-order kernels along u (δ and the thickness moved
together, w's itau direction taken at the moved operator) paired with w
converge at second order (ratio 3.99994 over eps 0.02 → 0.01), Richardson
landing at 7.6e-11 (gate 1e-6). The continuous sweep's gates (§11) are
unchanged; the two are the same problem's commute pair, as in 1D.

## 15. The material in the second-order runs

The directions u, v of the second-order machinery (§8–§12, §14) gain a
part in the moduli — `MaterialDirection` (1D, nodal) / `MaterialDirection3D`
(3D, grid values): δκ, δμ with the branch moduli as fixed fractions and
the relaxation times held — so that the Hessian's
action covers every pairing of (ice, itau, κ, μ): `SecondOrderDirection::
materialU/materialV`, `AdjointSolver::SecondOrderKernels(..., material)`,
`KernelDerivatives::kappaKernel/muKernel` (and the 3D forms with their
spherical reductions). New design, no pyslfp counterpart. Everything
runs under any rotational feedback. The
moduli's directions and kernels are held at the memory points, GLL or
Gauss, in 1D (§13c, §13e) as on the 3D coupling's memory nodes. The 3D
forms are distributed (§17).

**The second-order tangent run** (`TangentLinearSolver::RunSecondOrder`,
3D alike). The step map is affine in the moduli, so the second variation
is the bilinear pairing of each direction's material forces with the
other's tangent trajectory: for the ordered pairs (a, b) = (u, v), (v, u)
with a material in a, the forces −dK_eff[δm_a] x'_b + Σ_k f_k δμ_a
⟨m̃'_{k,b} + drive_b⟩ enter each step through `AddForce` (1D) /
`OpenStep::AddForce` (3D), where x'_b is b's FULL tangent solution at the
node — recovered from b's recorded snapshot by one unrelaxed solve under
its loads and memory, plus b's own unrelaxed material force's response
−K_U⁻¹(dK_U[δm_b] x_n − MemVar(δμ_b, m⁺_n)) on the base when b has one
(the unrelaxed and effective forms of the material force are equal, the
end solution satisfying unrelaxed equilibrium with its new memory) — and
m̃'_b + drive_b is b's decayed tangent memory with b's own itau drive, read
off a step opened on b's tangent state (`mem_` / `memory_` after
`AddMemory`): the drive's dB d¹ part is the mixed moduli–relaxation-time
term ∂²K_eff/∂μ∂h x_n in disguise. The start response is a's unrelaxed
variation on b's start solution, summed over the pairs. In 3D the shear
weight is δμ_a (1 − Σ_k f_k β_k(x)) with the BASE step's lateral weights
and the strain is the recorded tangent strain; `ForwardCursor3D`
serves the node before its current one, so the itau and material drives
of one step may ask for (node − 1, node) in either order.

**The discrete sweeps** (1D `DiscreteSecondOrder`, 3D `DiscreteSweep`).
By ⟨H v, u⟩ = ⟨λ, forcing''(u, v)⟩: the parts of forcing'' linear in u's
tangent trajectory transpose into SOURCES of the derivative dual state,
the rest pair with the first duals directly. Along a material v: the
dual of the full tangent solution receives −dK_eff[δm_v] y_n (the
derivative of the effective systems on the first sweep's solution; in 3D
the Born force on the first dual strain with the step's lateral weights,
plus the bulk term), added BEFORE the unit-response contraction so that it
reaches the sea level adjoint — no table derivative is needed, for the
same reason the forward tangent needs none; the derivative dual of the
decayed memory receives f_k δμ_v S y_n (the memory force's variation on
the first dual strain), which then flows through the kernel pairing's
B-weight terms (written as the decayed-memory duals against the end
strains) and the E/A recursion; the start likewise with the unrelaxed
systems. The material kernels' derivatives are the pairings of the
derivative duals with the forward trajectory plus the first duals with
v's tangent trajectory — its full solution (recovered as above) and its
decayed memory with v's full discrete drive (whose dB d¹ part is the
effective factor's derivative along v's itau) — and the start terms
alike. Exact: the Hessian symmetric and equal to the second-order run to
the solvers' tolerance.

**The continuous sweeps** (1D `ContinuousSecondOrder`, 3D `Sweep`). The
second-order adjoint equations are the derivative of the continuous
adjoint equations, which have the forward form, so along a material v the
derivative state is driven by v's material forces on the ADJOINT
trajectory — the tangent's drive with the adjoint states as the base: on
each reversed step's end solution and decayed memory (`AddForce` on the
derivative state's step, the 3D shear weight with the step's own lateral
weights), on the equilibrium solutions at the observations (an extra
response channel of the closure), and on the instantaneous jump
solutions, whose fingerprint moves with the unrelaxed table — carried as
the response channel of v's unrelaxed variation on the jump solution (the
forward start's pattern), its sea level the derivative LUMP of the
derivative duals (a material v is the one direction whose derivative
duals carry lumps). Three further pieces: the
itau pairing's branch weight μ_k moves with δμ (the term δμ_v/μ_U times
the pairing); the derivative state's strain after a jump must carry v's
material response (what `StartStep` would regenerate lacks it; the
trapezoid's α d⁰ reads it — without it the convergence to the discrete
derivatives drops to first order); and the material kernels' derivatives
pair the derivative displacement with the forward trajectory and the
adjoint displacement with v's tangent solution and END memory (no drive:
the continuous pairing is with the full forward memory). At the commute
level, falling at the scheme's order under refinement.

GATES. 1D: `TangentLinear.SecondOrderRunInTheMaterial` — u, v in the
loads, itau and the moduli (1 % of the moduli): the second-order run
against central differences of the first-order run on the shifted
material (the rheology rebuilt at the same fractions, the inverse times
and loads moved), second order with Richardson at 1e-6, both schemes,
symmetric, the moduli–loads block non-zero;
`SecondOrderAdjoint.MaterialBlocksAreExactAndSymmetric` — the discrete
sweep with the material kernels on: carried kernels at 1e-12,
⟨H u, w⟩ = ⟨H w, u⟩ = ⟨g, F''[u, w]⟩ at 1e-9, central differences of the
full gradient along w (material shifted, itau and thickness moved) paired
with u at second order, Richardson at 1e-6, both schemes;
`SecondOrderAdjoint.ContinuousConvergesToDiscrete` carries the
material blocks: ice, viscosity and material gaps and the mismatch with
the continuous second-order run all fall by > 3 from 17 to 33 nodes
(along a material-only v the gaps are 0.06 %, 0.23 %, 0.04 % at 17 nodes,
ratio 4 over 17 → 33 → 65). 3D:
`TangentLinear3D.SecondOrderRunInTheMaterial` (loads, lateral log
viscosity, angle-uniform moduli; lateral viscosity on; second order,
Richardson 1e-6, symmetric);
`SecondOrderAdjoint3D.DiscreteMaterialHessianIsExactAndSymmetric`
(lateral directions in everything: carried kernels at 1e-10, symmetry and
the adjoint identity with the discrete second-order run at 1e-8);
`DiscreteZeroLateralMatchesThe1DDiscreteSweep` and
`ZeroLateralMatchesThe1DSweep` with an angle-uniform material v and
the material kernels (their spherical reductions and derivatives against
the 1D at 1e-7 / 1e-6); `ContinuousMaterialHessianConverges` (lateral
directions, the continuous blocks against the discrete and the mismatch
with the continuous run below 10 % at 9 nodes and smaller at 17).

## 17. The derivatives over radial segments

The derivative paths run under the radial distribution of
[radial-partitioning.md](radial-partitioning.md), as the forward step,
the continuous 3D adjoint and the 3D tangent-linear runs do:

- **The discrete 3D transpose** (first and second order, §12, §14, with
  the material of §15): each rank transposes its own memory nodes' part
  of every pass; the duals of the load and centrifugal coefficients are
  the ranks' PARTIAL contractions of the unit responses (whole) with the
  forces (held in shares, each rank its own rows) summed over the ranks;
  every rank solves the sea level adjoint alike from them and rank 0's
  answers serve all, a failure failing on all; the surface duals and the
  degree-0 moment row join the forces on the rank holding their rows
  (`DegreeOperator::Holds`); the transposed fixed point's residual is the
  ranks' maximum, as the forward's. The forward step's conventions
  (`Evolution3D::March`) transposed.
- **The 3D material tangent, the 3D material kernels of both sweeps and
  the material directions of the 3D second-order sweeps** need nothing
  of their own: their pieces — the Born force assembled on the rank's own
  nodes, the bulk term's dilatation through the scalar companion, the
  unrelaxed responses through `SolveForces`, the closures' extra response
  channels through the collective `Surface` and the summed moment — are
  the forward step's.
- **Threads.** The derivative loops are threaded through
  `detail::ForEach` / `ForEachTask` (the 3D sweeps over nodes
  and grid values, the 1D sweeps over degrees), as are the static Born
  perturbation's pointwise loops.

GATES: `RadialRanks.DerivativesMatchTheOneProcess` (threads as ranks,
three ranks, the 1e22 mantle with lateral viscosity, GLL memory nodes):
the 3D tangent-linear runs of both methods along loads, a lateral log
viscosity and lateral moduli; the first-order sweeps of both methods
with the material kernels; the second-order sweeps of both methods along
that direction with the kernels' derivatives — the ice views on every
rank against the one process at 1e-11, and each rank's slices of the 3D
kernels and derivatives on the same cuts at 1e-9 (measured 1e-12: the
sweeps close at the coupled tolerance of 1e-11, in another summation
order).
`Distributed.DerivativesMatchTheOneProcess` is the same under mpiexec
(TestDistributed), on 2, 3 and 4 ranks.
Not distributed: the static Born perturbation (`ElasticPerturbation3D`),
a small static problem run at benchmark size on one process.

## 18. The tide and the tangential displacement

**The forcing.** A run is driven by a direct load f_n and a
prescribed tidal potential psi_n at each node (`Evolution::Forcing`),
with an ocean or without one (a state whose ocean function vanishes),
under frozen shorelines or migrating ones. A tide is taken relative to
the centre of mass, so psi_n is degree 2 and up (`kLowestTideDegree`):
a degree-0 potential exerts no force and a degree-1 one moves the whole
body without deforming it, and every entry (the sea level solves, the
evolutions, the operators' tide channels, the tangent-linear
directions, `ElasticPerturbation3D::TideForces`) refuses a part below
degree 2. A frozen run is linear in
both, so the tangent-linear run along (delta f, delta psi) is the run
under them as its forcing (`TangentPerturbation::tide`,
`TangentPerturbation3D::tide`). The tide enters a node twice: the sea
level solve, where the body answers it through the tidal Love numbers
and the potential itself moves the equipotentials, and the radial
steps, whose tide channel carries psi_n beside the centrifugal
potential of the rotation. A migrating run takes the tide in the same
two places, its nonlinear solve moving the shorelines by the sea
surface the tide shifts (`GRDSolver::NonlinearForcing::tide`). Under
the sharp flotation rule its derivative along the tide is the frozen
one about the run's own ocean functions C_n, each node's sea level
solve over C_n: no new term, and so every class below takes the tide
about a migrating run as it takes a load, by the ocean functions it is
given.

**The tide kernels.** The discrete sweeps transpose both entries:
with tau_n the sea level adjoint's tide dual at node n and
psi-dagger_n the dual of the node's step channel (the dual the
centrifugal exit already reads),

    tideKernels[n] = tau_n + psi-dagger_n,
    delta J = sum_n <tideKernels[n], delta psi_n>,

in the L2 coefficient pairing of real fields, its entries below degree
2 zero: the kernel is the dual over the tides admitted, as tau_n is
(`AdjointResponse::tideDual`). Without rotational feedback psi-dagger_n
still carries the steps' dependence on the tide; the feedback adds
nothing of its own, the centrifugal potential being a function of the
response. The kernels in the Earth's parameters need
no change: the recorded trajectory holds the potential the steps saw,
the tide included.

**The tangential displacement.** The sea level solve returns the
potential V of the tangential displacement, u_t = grad_1 V:

    V = l (sigma) + l_u (z_u) + l_phi (z_phi) + l_t (psi + psi_c) + V_r,

sigma the converged total load, z_u and z_phi the split load channels,
psi_c the centrifugal potential and V_r a ready response (a step's
memory, and in 3D its lateral part, read at the surface as U's and
Phi's are). V feeds nothing back, so its dual lambda_V leaves the fixed
point untouched: it joins the dual load by l, the centrifugal dual by
l_t at degree 2, the tide dual by l_t, and is itself the dual of V_r,
which the sweeps place on the surface row of V in the transposed solve
as they do U's and Phi's. A measurement of V is
`Measurement::tangentialPotentialDual`; one of the tangential
displacement itself, int w . grad_1 V dS, has the dual -div_1 w.

**Displacement measurements.** A measurement of the surface
displacement is a vector weight w on the surface, J = int w . u dS,
with

    u = U r + grad_1 V - r x grad_1 W,

V and W the spheroidal and toroidal potentials of the tangential part.
In the canonical basis (Phinney & Burridge 1973; Dahlen & Tromp 1998,
appendix C; Crawford et al. 2017, eqs 2.51-2.60) the components expand
in generalised spherical harmonics of upper index -1, 0, 1,

    u^0_lm = U_lm,   u^(+-)_lm = sqrt(l (l + 1) / 2) (V_lm +- i W_lm),

so a weight's components give its duals algebraically: the radial
component is the dual of U, and with w = w_r r + grad_1 a - r x grad_1 b
the pairing is sum_lm l (l + 1) (a V + b W), the duals of V and W having
the coefficients sqrt(l (l + 1) / 2) (w^+ + w^-)_lm and
-i sqrt(l (l + 1) / 2) (w^+ - w^-)_lm (`Measurement::displacementDual`,
`tangentialPotentialDual`, `toroidalPotentialDual`).
`DisplacementMeasurement` builds them from a weight given by its east,
north and up components on a grid carrying upper index 1, and
`DisplacementField` is the inverse map, the vector field of U, V and W.
`PointDisplacementMeasurement` is the weight concentrated at a site
along a direction, band-limited as the point dual is, and needs no
transform: the upward part is the grid's reproducing kernel
K(x . x0), the horizontal part its derivative along the direction,
(x . d) K'(x . x0), on V, and along d x r on W.
`CapDisplacementMeasurement` reads the displacement smoothed over a
Gaussian cap as a vector field at the cap's centre: the same kernels with
degree l weighted by the cap's degree weights (`CapMeanWeights`), so that
it reads the displacement of the potentials U, V and W each smoothed
over the cap, the field `sl3d solve`'s `output.smoothing` writes, at the
centre along the direction.

A rigid rotation of a solid region carries no strain, so the toroidal
degree 1 of a solution is the solver's gauge, and an adjoint force that
applies a torque has no solution to drive. A measurement must therefore
not depend on the rigid rotation. The displacement measurements leave
the degree 1 out of the dual of W: they read the displacement in the
frame of no net rotation of the surface (`NetRotation`,
`WithoutNetRotation` and `RigidRotationPotential` move a W between
frames). `BaselineMeasurement` and `AngleMeasurement` read the change of
the distance between two sites and of the angle at one site between the
lines to two others: sums of point measurements whose forces balance,
which no rigid motion alters and no frame enters.

**The toroidal potential.** A radial model under a load and a tide has
no toroidal displacement: W is absent from its responses, and its
sweeps pass over the dual of W. Lateral structure makes one. The sea
level does not see W and no Love number makes it, so the sea level
solve carries it alone, from a ready channel to the response and its
dual back (`toroidalResponse`, `Response::toroidalPotential`; empty
when there is none). The 3D evolution and tangent-linear run read W at
the surface of the toroidal systems' solutions; the discrete 3D sweep
places its dual on their surface rows.

**The adjoint traction.** The continuous sweeps pose the adjoint as a
forward problem, each measurement's dual a forcing of it: the sea
level's a direct load, the potential's the attraction piece of a load,
and a displacement measurement's a traction on the surface, the weight
itself (times g, in the energy form the sweeps carry). The channels
are in the units of a load, a unit being the traction -g times it, as
a load presses down: the radial part enters as the traction piece of
the load -w_r, the tangential part as the spheroidal traction of
potential -a and the toroidal traction of potential -b, whose
coefficients at degree l are those of minus the duals of V and W over
l (l + 1). The body answers a unit spheroidal traction through h_v,
l_v and k_v, with h_v = l (l + 1) l_u and k_v = g l (l + 1) l_phi by the
reciprocity of the symmetric form; the jump at an observation solves
the unrelaxed sea level equation under it
(`GeneralisedForcing::tangentialTraction`) and adds the channel's unit
response (the third column of `SplitUnitResponses`) to the radial
solution whose strain bumps the adjoint internal variables. A toroidal
traction moves no load and no potential in the radial reference: its
unrelaxed toroidal response joins the jump's solution through its
strain alone. Under lateral moduli both tractions' forces join the
instantaneous fixed point's (`OpenStep::AddToroidalForce` for the
toroidal one); they carry no mass, and the rotational feedback needs no
term of theirs. The attraction piece joins it as a force too, but
carries the load's mass: its degree-0 inertia moment, a^2 / 2 int z_phi
dS (`GRDSolver::SurfaceMassMoment`), which the Love-number closure adds
itself for `potentialLoad` (the shell term), is given to the fixed
point's sea level solve beside the body's. On lateral moduli under the
axial and the full feedback, measured in the potential with a degree-0
part, the elastic problem's continuous second-order sweep (the jump and
its derivative along the moduli) is symmetric and equal to the
second-order run and to the discrete sweep within 3e-12, and to central
differences of the continuous first-order kernels within 1.2e-8
(Richardson; `SecondOrderAdjoint3D.PotentialMeasurementUnderRotationAndLateralModuli`).

**What takes them.** The tangent-linear runs, 1D and 3D, take the tide,
about a frozen or a migrating run, and return V, and in 3D W. Every
sweep, discrete and continuous, 1D and 3D, takes a displacement
measurement in full. The discrete sweeps form the tide kernels, first
and second order, about either run; the continuous ones when
`Options::tide_kernels` asks (in 3D on one process).
`TestMigratingDerivatives` holds the tide about a forced migrating run
(the marine cap's, loads, a layer and a source beside it): the tangent
along a tide of every degree from 2 against central differences of
runs that flip no cell (1e-7), the discrete tide kernels and the other
kernels about that run as the tangents' transposes (1e-8), the
second-order sweep symmetric and equal to the second-order run with the
tide in its directions, the tide kernels' derivative along the
relaxation times (in 3D the log viscosity) against central differences
of the first-order tide kernels (Richardson to 1e-6), and the
continuous tide kernels and their derivatives converging to the
discrete as the steps are halved, 1D and 3D.

## 19. Interior sources as parameters

**The tangent.** A frozen run is linear in its interior sources (the
point stress gluts and the stress gluts on shells of docs/design.md
§6a), so the tangent-linear run along a perturbation of them is the run
under the perturbation: `TangentPerturbation::sources` and
`stressGluts`, per node. It treats them as the evolution does, their
spheroidal forces on each step and the toroidal systems stepped beside
the loop (`ToroidalViscoelastic`).

**The moment kernels.** A source's forces on a step's systems are
linear in its moment tensor, and the dual of a step's forces is the
step's dual solution, so with y_n the dual solution of the step ending
at node n (of the unrelaxed start, at n = 0)

    delta J = sum_n y_n . delta f_n = sum_n sum_p K_np : delta M_p(t_n).

A point source's force on the coefficient (l, m) of an unknown is the
conjugate of M contracted with the strain of that unknown's unit mode,
so K_np is the strain at the point of the dual displacement: a stress
glut works on the strain, and its kernel is the adjoint strain. In the
real columns the sweep carries, the dual solution's m > 0 columns are
halved before `PointStrain` reads them, the conjugate pairing counting
those orders twice. The spheroidal dual solutions are the sweep's own.
The toroidal ones come from the transpose of the toroidal recurrence
(`ToroidalViscoelastic::StepTransposed`, `StartTransposed`), forced by
the dual of the surface W alone: a measurement that does not read the
toroidal potential has no toroidal part in its moment kernel.
`AdjointSolver::Options::moment_points` gives the points, and
`Kernels::momentKernels[n][p]` the tensors, in the solver's units.

**The relaxation times about a source-driven run.** Under sources the
toroidal systems carry internal variables as the spheroidal ones do,
so the run's dependence on the relaxation times has a toroidal part.
Such a run is recorded at every node (`record_stride` 1), each kept
state with its strain, the toroidal state beside it
(`Snapshot::toroidalState`, `ForwardTrajectory::toroidal`): the strain
of a source-driven solution is not the elastic solve under the recorded
loads alone, so it is kept and not regenerated. The tangent-linear run
steps the toroidal systems under the toroidal twin of the first-order
drive (`ToroidalViscoelastic::FirstOrderDrive`: the scheme weights'
derivatives on the toroidal scalars, or the continuous drive), with no
force of their own unless the sources are perturbed too. The discrete
sweep carries the toroidal duals back whenever the trajectory is
source-driven, forced by the dual of the surface W, and adds each
step's pairing of them with the forward toroidal states to the kernel
in the inverse relaxation times, the spheroidal sweep's expression on
the two toroidal scalars (`StepTransposed` with an `ItauPairing`).

**The moduli about a source-driven run.** The sources' forces do not
depend on the moduli (a stress glut is given), so the material tangent
and kernels of §13 hold with two additions. The base solutions the
material forces act on, and the kernels pair with, are those of the
source-driven run: the recorded forces (`Snapshot::sourceForces`,
`ForwardTrajectory::sourceForces`) join the base step where it is
redone and the base start's unrelaxed solve. And the toroidal systems
have their own share, in the shear modulus alone: with S the strain
operator of the two toroidal scalars and W_c their force weights, the
toroidal stiffness is S^T (W_c mu w) S at the points (w the
quadrature), so a perturbation delta mu forces a step by

    -S^T (W_c (1 - sum_k f_k b_k) delta mu w) S x
        + sum_k S^T (W_c f_k delta mu w) m~_k,

x the base step's toroidal solution, m~ its decayed memory, f_k the
branch fractions and b_k the scheme's end weights, and the start by
the unrelaxed systems under -S^T (W_c delta mu w) S x_0
(`ToroidalViscoelastic::MaterialForce`, `StartMaterialResponse`). The
sweep adds the transposes to the shear kernel, the dual toroidal
solution's strain paired with the same two terms (`StepTransposed`
with a `MaterialPairing`, `StartTransposed` with the base start). The
toroidal workspace spans every element of its system for this, the
elastic regions included.

**The continuous sweep.** The continuous adjoint is a forward problem
in reversed time, and in the energy form it carries a forcing's kernel
is the adjoint solution's own work against it, as the load kernel is
the adjoint sea level. With lambda the adjoint displacement at a node
(the unrelaxed solve under the adjoint memory and the closed adjoint
load; at an observation the jump's instantaneous solution, with no time
weight) and c the constant between the sweep's pairing and the
material's:

- the moment kernel is c times the strain of lambda at the source
  point, toroidal part included;
- the tide kernel at degree l, order m is c times lambda's pairing with
  the degree's unit tide force, less r^2 / g times the adjoint load's
  coefficient: the tide moves the solid through its force and the sea
  level through its own potential, and the second acts on the adjoint
  load (the measurement's sea level dual and the ocean's);
- the toroidal adjoint is the toroidal systems in reversed time, with no
  force between observations and a jump at each by the unrelaxed
  toroidal response to the adjoint traction of a measurement of W
  (`ToroidalViscoelastic::Jump`).

About a source-driven trajectory the kernels in the Earth's parameters
gain their toroidal parts: the multiplier integrand W m-dagger (d - m) /
itau of the toroidal internal variables (`AccumulateItau`), and in the
shear modulus the adjoint toroidal displacement paired with the forward
toroidal solution and memory (`AccumulateMaterial`); the spheroidal
pairings read the forward solution under its sources' forces. All of
these carry the continuous method's commute error, falling at the
scheme's order. The moment and tide kernels cost one unrelaxed solve
per node side; `Options::tide_kernels` asks for the second.

**In 3D.** The 3D tangent-linear run takes a perturbation of the
sources (`TangentPerturbation3D::sources`, `stressGluts`): their forces
join each step's, and the start is the zero-length step whether or not
the moduli vary laterally. The discrete 3D sweep forms the moment
kernels from each step's dual solutions, spheroidal and toroidal
together (`AdjointSolver3D::Options::moment_points`, on one process);
in the stepper's complex m >= 0 columns they are the coefficients of a
real field as they stand.

A source-driven 3D run is recorded as the radial one is, at every
node with each state's strain and the sources' forces
(`Evolution3D::Snapshot::sourceForces`,
`ForwardTrajectory3D::sourceForces`). The 3D internal variables are
tensors on the grid, the toroidal part within them, so the tangent
and the sweeps in the relaxation times need nothing more: with the
strain kept, the drive and the pairings read a source-driven
trajectory as any other, by both methods.

For the moduli the 3D tangent and both sweeps redo a node's full
forward solution (its dilatation is what the bulk term reads; the
deviatoric strain is the recorded one), and about a source-driven run
that solution is under the sources' forces too: the unrelaxed solve of
the recorded forces is added to the solve under the recorded loads.

The continuous 3D sweep forms the moment kernels as the radial one
does: the strain at the points of the adjoint displacement at each node
side under the time weights, and of the jump's solution at each
observation without one (under lateral moduli the instantaneous fixed
point's).

Its tide kernels are the radial sweep's construction too
(`AdjointSolver3D::Options::tide_kernels`): the adjoint displacement's
spheroidal columns paired with each degree's unit tide force, less
r^2 / g times the adjoint load, whose coefficients at an observation
are those of the measurement's sea level dual and the ocean's share of
the jump's sea level.

**Second order.** The run is linear in its sources, so the Hessian has
no source-source block; the sources enter the second derivative through
the mixed blocks, with the relaxation times and the moduli. A direction
that perturbs the sources is carried, as a load's is, by its recorded
tangent-linear run. The discrete 3D second-order sweep takes a
source-driven trajectory and such directions in the relaxation times
and the sources: the derivative duals' solutions give the derivatives
of the moment kernels along v (`KernelDerivatives::momentKernels`, the
source rows of the Hessian's action), by the same strain at the points,
and the viscosity rows are formed as about any trajectory. The
second-order tangent-linear run takes the same.

The moduli join them: where a second-order run or sweep rebuilds a
solution, the base's is under the recorded sources' forces and a
tangent run's that perturbs the sources is under its own, which its
recorded snapshots carry. The continuous 3D second-order sweep takes
the same trajectories and directions, at its commute level, without the
moment rows.

A note on the continuous sweeps' jump along a direction in the moduli
(§15): the derivative of the instantaneous adjoint solution is the
unrelaxed response to the direction's force on it, closed by the sea
level equation, and the closure returns two channels to the solution,
the ocean's load and, under the rotational feedback, the centrifugal
potential of the derivative's rotation. Both are added, in 1D and in
3D; a measurement of the displacement or of the tangential potentials
depends on the second at the per-cent level.

**The radial model's second order.** Its toroidal systems step on
their own (`ToroidalViscoelastic`), so their second order is their own
too, beside the spheroidal one. With the step written as the decayed
memory m~_k = e_k m_k + alpha_k d_0 (plus a drive), the effective solve
K_eff W = F(m~) + f and the update m_k' = m~_k + beta_k d_1:

- *The second-order run.* The drive on a step is each direction's
  perturbation of the inverse relaxation times on the other's toroidal
  tangent trajectory (`FirstOrderDrive` on its recorded states), the
  two on the base trajectory through the weights' second derivatives
  (`SecondWeightDrive`, the discrete flavour), and each direction's
  shear modulus on the other's tangent step (`MaterialForce` on that
  step formed again: its decayed memory with its drive, its solution
  under its sources' forces and its own material force). A tangent
  run records its toroidal states and its perturbed sources' forces
  for this (`detail/ToroidalSecondOrder.hpp` forms the steps again).
- *The discrete sweep.* The transposed recurrence carries the
  derivatives of its duals beside the duals
  (`StepTransposedSecond`, `StartTransposedSecond`): the end weights'
  derivatives act on the memory duals, the effective modulus's
  variation (delta mu's share, and the end weights' through delta
  itau) on the dual solution as a source of its derivative, and the
  branch moduli's variation on the dual solution's strain as a source
  of the memory duals' derivatives. The dual of the surface W is the
  measurement's own and no parameter moves it. The kernel in the
  inverse relaxation times is the dual of the decayed memory with the
  forward step's derivative in h; its derivative has that pairing with
  the derivative dual, with v's tangent states, and with the weights'
  second derivatives. The shear kernel's derivative pairs the
  derivative dual solution with the base step and the dual solution
  with v's tangent step; the end weights' derivatives in the share are
  in the tangent step's decayed memory, whose drive carries them.
- *The continuous sweep.* The toroidal adjoint state is stepped in
  reversed time by the forward machinery, and its derivative likewise
  under v's drive on the adjoint steps and v's material force on them;
  at a measurement of W the derivative state jumps by v's perturbation
  of the inverse relaxation times on the jump solution's strain and by
  the strain of the jump solution's own derivative (the unrelaxed
  response to delta mu on it). The pairings take the derivative state
  with the forward toroidal trajectory, the state with v's, and the
  integrand's own dependence (the branch weight with delta mu, the
  relaxation time with delta itau).

Both sweeps return the moment kernels and their derivatives along v
where points are given: the strain at the points of the dual solutions
(discrete) or of the adjoint displacements, regular and lumped
(continuous), spheroidal and toroidal, and of their derivatives. The
toroidal systems are swept wherever the base run or v's twists, so the
block of the sources against the parameters is formed about a
load-driven run too.

The 3D continuous second-order sweep forms its moment rows the same
way.

**The elastic problem.** A time grid of one node is the elastic problem:
the evolution's run is its instantaneous solve, and the tangent-linear
runs, both sweeps and the second-order sweeps take the one node, in 1D
and in 3D with or without lateral moduli. Each reduces to its start
term: the transposed unrelaxed solve (discrete) or the jump (continuous)
— the two then agree to rounding, there being no step to commute —
with the start's pairings for the kernels in the moduli and the source
moments and their derivatives. The thickness kernel is the
instantaneous one; the rate and density views, which divide by a node's
interval, are zero. A rheology with no relaxing layer is taken alike,
and so is one with no branch at all (the static problem's): the sweeps
size their arrays at the memory points by the points themselves, not by
a branch. The static problems therefore have no sweeps of their own:
load, tide and source kernels, kernels in kappa and mu about a laterally
heterogeneous background, and the Hessian's action, are those of the
evolution asked for one solve, and the lateral kernels about a radial
one are `LateralKernelSolver`'s at one node (§21). Without an ocean or
rotational feedback the kernels in kappa and mu have an independent
form, minus the transpose of `ElasticPerturbation3D::Force` on the
static adjoint and forward solutions (`ForceTranspose`, §13h, §13j),
which checks them:
`StaticDerivatives.OneNodeSweepsReproduceTheForceTranspose` holds the
one-node sweeps' kernels in kappa and mu, both methods, to that
transpose's means over the sphere at every node of the static coupling
at 1e-11, under a load and under a body tide measured in U, V and phi,
and `…UnderLateralModuli` to the transpose itself at every grid value
of the coupling at 1e-11, the static solutions from the full lateral
solves. The 3D one-node route forms its kernels at the coupling's
memory nodes only, which on a branchless rheology are where the
lateral moduli are non-zero.

**What is formed on one process.** The 3D moment kernels of both
sweeps and the continuous sweep's tide kernels are formed on one
process (the discrete sweep's tide kernels are distributed).

## 20. Unequal steps, solved starts and forced trajectories

The derivative classes read every step's size from the time grid (the
per-dt caches of weights and effective systems, the cursors' rebuilt
steps, the trapezoid node weights and the continuous sweeps' split
weights), so a grid of unequal steps needs nothing further.
`TestUnevenSteps` holds them on steps 1 : 3 : 0.5 : 2 : 1.5 kyr, both
schemes, 1D and under lateral viscosity in 3D: the tangent along the
loads and the thickness the difference of two runs at 1e-8; along the
relaxation times (the log viscosity in 3D) and the moduli the
derivative of the run, central differences at second order with
Richardson at 1e-6; the discrete sweeps the tangents' transposes at
1e-8 (every kernel, and a stride-2 recording: in 3D at 1e-9, in 1D the
relaxation-time kernel at 1e-10 of the full recording's, measured
4e-13); the
distributional and
quelled views equal to the thickness view; the discrete Hessian
symmetric and equal to the second-order run (1e-9 in 1D, 1e-8 in 3D,
where they agree to parts in 1e12), which is itself
the derivative of the first-order run (Richardson at 1e-6); the
continuous forms converging to the discrete at the scheme's order as
every step is halved (rates near 4 for the trapezoid and 2 for
backward Euler, the even grids' — the load, relaxation-time and moduli
kernels, the continuous tangent and the continuous Hessian); and a
decay time fitted over the uneven window, its weights the fit's
derivative and its kernel the forward runs' derivative.

A migrating run whose start is solved — under an ice change at the
first node, or under forcings beside the ice — is linearised about its
ocean functions node by node, the start's included; the thickness
direction is relative to the first node's (TangentPerturbation::
iceThickness), the derivative with the background's ice moved with the
first node's. With the background held, the derivative along the
history adds the run along rho_i (1 - C) dI_0 at every node, and its
kernel at the first node is the thickness view less the baseline.
`TestMigratingDerivatives` holds both readings against central
differences of migrating runs that flip no cell (1e-7), the viscosity
and the moduli about such runs (Richardson at 1e-6; 1e-5 for the 1D
moduli, whose step that flips no cell leaves the solves' noise at parts
in 1e6), the discrete kernels as the transposes (1e-8) and the
Hessian's symmetry and equality with the second-order run about a
forced trajectory, and every continuous kernel (the loads inside and
outside the water budget, the layer, the thickness, the initial sea
level, the moments, the relaxation times or log viscosity, the moduli)
and the continuous Hessian converging to the discrete at second order,
1D and 3D, under the exponential trapezoid; under Sdirk2, where the
shorelines move inside steps, the continuous kernels converge to the
discrete at first order (§1a), and the 3D sweep's about the solved start
is held to halve at least with the step.

**A jump of the forcing.** A time repeated once in the grid, times[k + 1]
= times[k], is a jump of the forcing there (an earthquake's onset,
`Forcing::sourcesBefore` the left limits of the sources; a step of a
surface term, `directLoadBefore`, `directLoadOutsideWaterBudgetBefore`,
`solidSurfaceChangeBefore`, `tideBefore` and a migrating run's
`iceThicknessBefore`): node k the
left limit, node k + 1 the right, joined by a step of zero length. At
h = 0 the weights are exact (e = 1, α = β = 0, every effective weight
one), so the step is the static elastic solve with the internal
variables held, and its effective systems, unit responses and Love
numbers are the unrelaxed ones (`EffectiveMaterial`, `PointShearModulus`
and `EffectiveLoveNumbers` at dt = 0). The internal variables are
continuous across the jump and the strain, the displacement and the sea
level jump by that solve, the continuum's statement. Every derivative
run sees an ordinary step of zero length, with the forward trajectory
recorded at both nodes: the discrete transposes and tangents transpose
and differentiate it as any step (its weights' h-derivatives meet
dh = dt d(itau) = 0); the continuous sweeps' split trapezoid gives the
zero-length interval no weight, so each side of the jump pairs with its
own interval through its own recorded state, one-sided values of an
integrand that jumps there, and an observation at the right node takes
its jump there as at any node; the checkpointed cursor steps back across
the pair by the run's zero-length step (m held) under the pair's two
recorded sides. A source at the left node
and one at the right are two parameters, each with its moment kernel,
and so are a surface term's two sides, each with its load, tide or
thickness kernel (the derivative runs read the recorded per-node loads
and the directions node by node, a migrating run's each side's own ocean
function). By the continuous method each side is an end node of its
interval, its quadrature the half step on its side alone: its smooth
part tends to the discrete kernel at the scheme's order in absolute
terms, at first order relative to its own size (O(h)), where no
observation's lump at the node leads it; the two sides' sum, the kernel
of a change continuous across the jump, at the scheme's order. The
quelled view's zero-length interval, `rate[k]`, is the kernel of the
jump of the ice there, I(t+) - I(t-), the limit of h_k (dI/dt)_k: the
tail of the thickness duals from the right node on, by both methods.
`TestTangentLinear` (DerivativesAcrossAnOnset, TheContinuousSweepAcross
AnOnset, SecondOrderRunAcrossAnOnset), `TestSecondOrderAdjoint`,
`TestDiscreteAdjoint3D`, `TestTangentLinear3D`, `TestSecondOrderAdjoint3D`,
`TestLateralKernels` and `TestDistributed` hold them with an onset at an
interior node measured at both nodes of the pair: the discrete sweeps
the tangents' transposes (1e-8), the tangents central differences'
limits (Richardson at 1e-6), the discrete Hessian symmetric and the
second-order run's (1e-8), and the continuous kernels and Hessian rows
converging to the discrete at second order through the jump. Their
`...AcrossALoadJump` twins (`TestTangentLinear`, `TestSecondOrderAdjoint`,
`TestDiscreteAdjoint3D`, `TestSecondOrderAdjoint3D`, `TestLateralKernels`)
and `MigratingDerivatives.AcrossAnIceJump` hold the same with every
surface term jumping at the pair, each side perturbed apart, the run
recorded sparsely through it; `TestTrajectoryCursor` the sweeps about a
sparse recording through a load and tide jump and a migrating ice step
equal to the full recording's; `TestDistributed` the run and its sweeps
under mpiexec through a load step and an onset.

## 21. Lateral kernels about a radial background

The kernels of a measurement in LATERAL structure — each branch's log
viscosity, the bulk modulus κ(x) and the unrelaxed shear modulus μ(x),
as fields over the memory nodes' angular grids — are the starting point
of a 3D inversion, and about a radial Earth they need no 3D solve (Al-Attar
& Tromp 2014; Crawford et al. 2018): the forward and adjoint problems
separate by degree, coupled only through the sea level equation, and a
kernel is the pointwise product of the forward and adjoint fields
expanded in space, accumulated over the time steps.

**What the radial route computes.** The radial sweeps (§3, §13e, §13g,
§19) run as they do for the radial kernels, and at each place where they
pair an adjoint field with a forward one they hand both, degree by
degree, to an accumulator (`detail/LateralKernels.hpp`) that expands them
on the grids of a `GridCoupling` over the radial model's memory points
(`AddScalarState` for the spheroidal internal variables and strain
scalars, `AddToroidalScalarState` for the toroidal ones, whose components
are those of a strain of W alone, `AddStrainOfSolution` and
`AddDilatationOfSolution` for solutions), contracts them pointwise with
the energy metric of §6a and adds them with the 3D sweeps' weights:

| Kernel | Method | Adjoint field | Forward field | Weight |
| --- | --- | --- | --- | --- |
| itau, branch k | continuous | the adjoint memory m†_k | d − m_k | pairing × time weight × μ_k Wq r² / itau_k |
| itau, branch k | discrete | the dual of the decayed memory, m̃†_k | dE m_k + dA d⁰ + dB d¹ of the step | dt Wq r² |
| μ | continuous | the strain of the adjoint displacement λ | Σ_k f_k m_k − d | pairing × time weight × Wq r² |
| μ | discrete | the strain of the step's dual solution y | Σ_k f_k m̃_k − (1 − Σ_k f_k b_k) d¹ | Wq r² |
| κ | both | the dilatation D(λ) or D(y) (the scalar companion) | D(x), x the full forward solution | minus μ's, without r² |

with the observations' lumped pairings (continuous, no time weight) and
the start's transposed unrelaxed solve (discrete, every weight one) as in
§13e–§13g. The log-viscosity kernel is −itau K_itau; the means over the
sphere are kept beside the fields (`itauKernelSpherical`,
`kappaKernelSpherical`, `muKernelSpherical`), and the result is a
`Kernels3D` in the 3D sweeps' layout, pairing and units (§4), its `ice`
the radial sweep's own kernels.

**Why it is the 3D sweep.** With δ = 0 and no lateral moduli the 3D
step's lateral weight vanishes, its fixed point closes in one pass, and
every grid field the 3D sweeps hold — the forward state and strain, the
adjoint state, the duals, the dual and adjoint solutions — is the
synthesis of the degree-domain field the radial sweep holds at the same
node of the same step; the scheme weights, the branch moduli and
fractions, the effective factor and itau are radial. The 3D kernel
integrands are therefore pointwise products of syntheses of the radial
sweep's fields, at the same nodes in time with the same weights, which is
what the accumulator forms. The scalings are fixed by the pairings: the
continuous adjoint state and the solutions are natural coefficients; a
discrete sweep's dual of a solution pairs by the real-column dot
product, which counts each order m > 0 twice, so the 3D dual is it over
(1, 2, 2, …); and the discrete dual of the decayed memory carries the
memory pairing's weight k_s Wq of each scalar besides (k_s the force
weights k²(k²−2), 4/3, k²), which the energy metric of §6a reproduces in
the angle-uniform reduction — dividing it out gives the 3D sweep's dual
m̃† = m† + μ_k Exit(x†). The radial kernels are these fields' means.

The 3D integrands read every field over every node of the coupling: the
forward strain in the moduli kernels is the full solution's (the radial
state carries strain on its memory window only, the viscous elements, so
the elastic elements' comes from the solution), and the toroidal fields
enter beside the spheroidal ones. A measurement of the toroidal potential
W makes a toroidal adjoint even about a load-driven run, and a seismic
source a toroidal forward run; their products with the spheroidal fields
integrate to zero over the sphere but not pointwise, so the radial
sweeps carry the toroidal adjoint wherever a measurement reads W
(the toroidal partner of §19, its decayed-memory duals read off
`ToroidalViscoelastic::StepTransposed`), and the forward toroidal states
of a source-driven run are the trajectory's.

**Where the nodes are.** `LateralCoupling`: the operator's memory points
at its lmax, over the viscous elements when the viscosity's kernels
alone are asked for, and over every solid element when the moduli's are
(an elastic element's viscosity kernel is then zero), the element
holding the centre included at the Gauss points and left out at the
GLL points (§13k, "The element holding the centre") — the elements the
3D operator carries for a lateral viscosity and for lateral moduli over
the whole solid, and at the GLL points `ElasticPerturbation3D`'s. The
transforms may be cut by depth
(`transform_degrees`) as the 3D classes' are.

**The entry point.** `LateralKernelSolver` (`LateralKernels.hpp`) is the
one route to lateral kernels, the static problem's included: built on an
`Evolution` it runs the radial sweeps with the grid expansion, through
time or at one node (the static problem); built on an `Evolution3D` it
runs the 3D sweeps, at one node the static problem under lateral
moduli. Its `Coupling()` is the layout either way. It takes an ocean,
the rotational feedback, sources, every measurement and either point
set.

**The static problem's check.** Without an ocean or rotational feedback
the static kernels in κ and μ have an independent form: minus the
transpose of `ElasticPerturbation3D::Force` (§13b, §13j) on the static
adjoint and forward solutions (`ForceTranspose`, §13h), not a route of
its own but the check of this one. On a radial body the two agree at
every grid value within 4e-12 (κ) and 1.3e-12 (μ) at lmax 8, and
1.5e-12 / 2.7e-13 at lmax 32 and 64
(`LateralKernels.TheStaticRouteAgreesWithTheTransposeOfTheForce`), at
the same cost: 0.29 s against 0.30 s at lmax 32 and 2.3 s against 2.2 s
at lmax 64 on four threads (the route's time includes its sweep's sea
level solves; the expansions dominate both). Under lateral moduli the
3D sweeps at one node agree with it likewise (§19).

**Gates** (`TestLateralKernels`, prem8 with a 1e21 Pa s mantle at lmax 8):

- *Zero lateral structure.* Against the 3D sweep of the same method on
  a `Viscoelastic3D` with δ ≡ 0 (its nodes the viscous elements'), and
  against one whose memory nodes cover the whole solid (a lateral shear of
  one part in 1e100, which no solve sees and whose forces stay normal
  numbers), every field at every grid value, the load kernels included:
  a melting load on the analytic ocean measured in the sea level and the
  displacement, GLL and Gauss points, both schemes; the full rotational
  feedback with every field measured at three nodes (the toroidal
  potential among them) and W alone at a fourth, on uneven steps; no
  ocean, a load and a tide, the axial feedback; a growing point source,
  with and without a measurement of W, both schemes; migrating
  shorelines with a solved start (a load and a layer at the first node);
  and the static problem on a branchless and on the Maxwell rheology,
  with an ocean and without. The discrete route equals the 3D sweep at
  3e-14 – 3.4e-12, the continuous at 9e-14 – 3.3e-11 (the 3D fixed
  point's and conjugate gradients' tolerances, 1e-12); the load kernels
  within 1.5e-14.
- *Differences of the 3D run.* The kernels paired with lateral
  perturbations of the log viscosity, the shear and the bulk modulus,
  each of degree 2 and of degree 6, against central differences of the 3D
  forward run with the perturbation (steps of 2% and 1%, Richardson): the
  discrete kernels at 2e-11 – 5e-10, the continuous at their commute
  error, 0.15–2.9% at 2 kyr steps on a 1e22 Pa s mantle, falling by 3.4–4.4
  as the steps halve.
- *Means and recordings.* The means over the sphere are the radial
  kernels to 2e-12 (itau at the GLL points, κ and μ at both point sets,
  about a source-driven run); a recording kept at every second node
  gives the lateral kernels of the full recording at 2e-12, as it gives
  the radial ones.

**Cost.** The expansions are the route's cost: per node, two tensor
syntheses for each branch's viscosity kernel (the adjoint and the
forward field, both sides of a node in one where the state does not jump
between them) and two tensor and two scalar syntheses for the moduli's,
and no fixed point. On four threads, ten 1 kyr steps of a 1e21 Pa s
mantle, a sea-level and a displacement measurement (the disabled probe
`LateralKernels.CostAgainstTheAlternatives`, `SL3DPP_LMAX`), against the
radial kernels alone and the 3D sweep at zero lateral structure with its
forward run (its nodes over the whole solid for the moduli):

| lmax | kernels | method | radial kernels | radial route | 3D sweep + 3D run | recordings, radial / 3D |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | viscosity | discrete | 0.05 s | 0.16 s | 0.58 + 0.29 s | 24 / 160 MB |
| 16 | viscosity | continuous | 0.06 s | 0.19 s | 0.37 + 0.29 s | 24 / 160 MB |
| 16 | viscosity, moduli | discrete | 0.10 s | 0.46 s | 2.09 + 0.41 s | 24 / 246 MB |
| 16 | viscosity, moduli | continuous | 0.16 s | 0.65 s | 2.92 + 0.41 s | 24 / 246 MB |
| 32 | viscosity | discrete | 0.37 s | 1.20 s | 5.12 + 2.09 s | 167 / 1200 MB |
| 32 | viscosity | continuous | 0.44 s | 1.33 s | 3.01 + 2.09 s | 167 / 1200 MB |
| 32 | viscosity, moduli | discrete | 0.63 s | 3.51 s | 18.1 + 3.3 s | 167 / 1842 MB |
| 32 | viscosity, moduli | continuous | 1.13 s | 5.00 s | 25.9 + 3.3 s | 167 / 1842 MB |

The radial run itself is 0.07 s at lmax 16 and 0.33 s at 32 against the
3D run's 0.29–0.41 s and 2.1–3.3 s. So the lateral kernels about a radial
Earth cost three to six times its radial kernels; with the forward runs
counted, 2.5–4.7 times less than the 3D route for the viscosity's and
4.8–5.6 times less with the moduli's, the gain growing with lmax, on a
recording seven to eleven times smaller; the kernels themselves (each a field over
the nodes' grids, 44 MB for the viscosity's and 134 MB with the moduli's
at lmax 32 over 568 and 872 nodes) are the same size either way. The
transforms dominate, and the by-depth transform degrees cut them as they
cut the 3D route's.

**Over ranks.** A radial run is one process's (`sl3d` runs it on the
first rank), and so are its lateral kernels. The expansion is local to
each node, so it would distribute over radial segments with nothing
new: every rank runs the radial sweep (cheap, and deterministic up to the
transforms' rounding), its accumulator holding a coupling over its own
elements (`GridCoupling::Options::elements`, cut by `TransformWeights`),
and the fields gathered at the end as the 3D app gathers its own. Not
built.

**Not built: the lateral direction.** The tangent-linear run along a
lateral perturbation about a radial Earth (the Born response through
time) does not separate by degree: the perturbation's force on each
step, Entry(δW ⊙ Exit(x)) and its bulk companion, couples every degree
and order and turns spheroidal motion toroidal, so the tangent state is
a 3D field however radial the background. It is the 3D tangent-linear
run (§10, §13d) on a δ ≡ 0 operator, whose within-step fixed point closes
at once; a dedicated route would save that fixed point's second pass and
the 3D forward run (the base fields synthesised from the radial
trajectory, as here), not the tangent's own transforms. The second-order
sweeps along such a direction need that run.

## 22. The derivatives of a calibrated run

A run may be CALIBRATED (`InitialSeaLevel.hpp`): its initial sea level
S0 is not given but found, so that the run reaches a given sea level
S_f at its last node (the present day's, usually). The sweeps of the
sections above take S0 as a parameter: their kernels are the PARTIAL
derivatives of an objective J, the initial sea level held, with the
initial sea level kernel k0 = ∂J/∂S0 beside them. A calibrated run's S0
is found again when a parameter changes, so the derivative of what the
run computes is the TOTAL one, with the change of S0 in it. Under
frozen shorelines the run does not depend on S0 at all (k0 ≡ 0), and
the two are one. Under migrating shorelines they are not: S0 places the
initial shorelines and, through the cells that flood or ground, every
later load. This section is Yu et al. 2025, §3.9, eqs. (85)–(92), in
the discrete setting.

**The converged condition.** Write the final sea level as S0 + ξ(t_f; p,
S0), p the run's parameters (the ice thickness at the nodes, the
relaxation times, the moduli, any other forcing) and ξ the sea level
change. The calibration weights the misfit by w ≥ 0, sets S0 on the
points P where w > 0 and holds it elsewhere (with no weights, P is the
sphere). Its converged state satisfies

    S0 + ξ(t_f; p, S0) = S_f   on P                                (22.1)

whichever method reached it, the classical update (eq. 85) or steepest
descent on the misfit (eqs. 86–88), mixed or not: the derivative below
is of (22.1), not of an iteration. With A = ∂ξ(t_f)/∂S0 and 𝒫 the
restriction to P (as a projection: zero off P),

    𝒫 (I + A) 𝒫 δS0 = 𝒫 (δS_f − (∂ξ(t_f)/∂p) δp),   δS0 = 0 off P.   (22.2)

**The multiplier.** With ⟨·,·⟩ the surface pairing of the duals (§4,
∫ · dS on the solver's sphere), δJ = (∂J/∂p) δp + ⟨k0, δS0⟩, and (22.2)
gives

    δJ = (∂J/∂p) δp − ⟨λ, (∂ξ(t_f)/∂p) δp⟩ + ⟨λ, δS_f⟩,
    λ = 𝒫 (k0 − Aᵀ λ).                                             (22.3)

λ is the multiplier of the constraint (22.1), SL0† of eq. (89), a field
in the duals' pairing, supported on P. Two things follow from the
sweeps as built. First, ⟨λ, ξ(t_f)⟩ is a sea level measurement at the
last node with the dual λ (`Measurement::relativeSeaLevelDual`, which
pairs with the node's ξ by ∫ · dS: no weight or factor between them),
so Aᵀλ is that measurement's initial sea level kernel and −(∂ξ(t_f)/∂p)ᵀλ
its kernels in p, negated. Second, the sweeps are linear in their
measurements. So ONE sweep, forced by J's measurements less the sea
level measurement λ at the last node, gives

- in every parameter of the run, the total kernel
  ∂J/∂p − (∂ξ(t_f)/∂p)ᵀ λ: the thickness views, the load kernels, the
  tide and moment kernels, the relaxation-time (viscosity) and moduli
  kernels, each in its own layout and pairing, nothing about them
  changed;
- as its initial sea level kernel, k0 − Aᵀλ, which on P is λ again by
  (22.3), and off P the total derivative in the initial sea level held
  there.

That sweep's forcing is the adjoint load of eq. (92): the measurements'
and, at the time-reversed present, a singular load proportional to the
multiplier, which depends on the adjoint sea level over the whole
window (eq. 91) and so closes by iteration. And λ = ∂J/∂S_f on P: the
kernel of the objective in the sea level the run is calibrated to.

**The iteration.** (22.3) is a linear fixed point for λ. From λ = 0 the
first sweep gives k0 on P, the partial run's; each sweep after it,
forced by the measurements less the current λ at the last node, returns
𝒫 (k0 − Aᵀλ), and the iteration stops when the largest change of λ is
below a tolerance times its largest value
(`SolveCalibrationMultiplier`, `CalibrationMultiplierOptions`). These
sweeps need the initial sea level kernel alone, which is assembled from
the load kernels and the run's ocean functions (§4): they take no
recorded trajectory and form no relaxation-time, moduli, tide or moment
kernel, so each costs about one forward run. The full sweep, with the
trajectory, runs once, at the end (`ComputeCalibratedKernels`). The
plain iteration contracts at the rate of A, which acts only through the
cells whose ocean function changes along the run: measured 0.25 per
sweep on the tests' marine cap (1D and 3D; 21 sweeps to 1e-12) and 0.10
to 0.16 on the synthetic deglaciation of the examples (9 to 10 sweeps
to 1e-8). The relation is linear and A of low rank, so Anderson mixing
(the calibration's own mixer, `detail/AndersonMixer.hpp`, depth 5 by
default) ends it sooner: 5 sweeps to 1e-12 on the marine cap, 6 to 7 to
1e-8 on the examples. An iteration that runs out of sweeps returns its
last iterate with `converged` false, and `sl3d kernel` refuses it.

**Conventions.**

- *Pairings.* λ, k0 and the measurement duals are non-dimensional
  solver-unit densities on the surface grid, paired by ∫ · dS (§4).
  The calibration's weights do not enter the pairing: they choose P,
  and nothing else of them is in (22.1).
- *The ice.* The parameters p are the run's own: the thickness at each
  node, in the thickness view of §4 (the first node's moving the ice
  the run starts from, its baseline folded in), which is what
  ∂ξ(t_f)/∂p is taken in, so the total thickness kernels are in that
  view, and the baseline of the total sweep is the total derivative in
  the ice the run starts from with every node's held. The constraint is
  on the sea level alone. A calibrated run's state also holds the ice
  at the last node, but that is the history's own value there, not a
  condition solved for: it brings no second multiplier, and changes
  ∂ξ(t_f)/∂p in no way. A state's ice raised with the history's changes
  held moves every node and the start alike, leaves the loads as they
  were, and acts through the target alone (where the target is a
  topography converted with that ice), by ⟨λ, δS_f⟩.
- *The scheme.* By the discrete method k0 and Aᵀ are the exact
  transposes of the stepped run under every scheme. Under Sdirk2 each
  step's first stage has its own shorelines, about which the sweep
  grounds that stage's part of the initial sea level kernel (§1a,
  `OceanFunctions::stages`); the multiplier's measurement sits at the last
  node, a node value like any other. The total kernels are then the
  derivative of the calibrated run as the program computes it, to the
  calibration's and the iteration's tolerances. By the continuous
  method k0 and Aᵀ are the continuous sweep's approximations (it reads
  the run's shorelines at its nodes and inner stages, §1a), λ is the
  fixed point of that approximate relation, and the total kernels
  converge to the discrete ones as the continuous kernels do: at second
  order in the step, at first order under Sdirk2 where the shorelines
  move inside steps. A total kernel is the partial one
  less a correction of comparable size, so it keeps fewer of the
  continuous sweep's digits than either: on the tests' stiff mantle,
  under the exponential trapezoid, 37%, 8.4% and 2.1% at 5, 9 and 17
  nodes where the discrete ones hold 1e-7.
- *The calibration's tolerance.* (22.1) is assumed to hold. A
  calibration stopped at `rtol` leaves a residual of that order, and
  the kernels are the calibrated run's to that order: tighten the
  calibration to tighten them.
- *Flotation.* As every kernel about a migrating run, these are
  derivatives while no cell's flotation flips, the calibrated start's
  included.

**The linearised calibration.** The same derivative in the forward mode
is the tangent-linear run of a calibrated run: the change of its
responses along a direction δp of its parameters and a change δS_f of its
target, with the change of the calibrated initial sea level in it. That
change is the solution of (22.2),

    𝒫 (I + A) 𝒫 δS0 = 𝒫 (δS_f − (∂ξ(t_f)/∂p) δp),

and each term is a tangent-linear run (§8, §10) read at the last node:
(∂ξ(t_f)/∂p) δp the run along δp with the initial sea level held, and
A δS0 the run along an initial sea level δS0 alone, which needs no
recorded trajectory. The equation is the transpose of the multiplier's
(22.3) (I + A against I + Aᵀ, on the same points), and is solved by the
same iteration: δS0 = 𝒫 (b − A δS0) with b the right side, from zero
(whose image is b itself, at no run), stopped when the largest change of
δS0 is below a tolerance times its largest value, Anderson-mixed as the
multiplier's is; one loop serves both (`SolveCalibrationTangent`, its
options the multiplier's, `CalibrationMultiplierOptions`). The run along
δp and the converged δS0 together is then the total tangent-linear run
(`ComputeCalibratedTangent`), and for any objective

    ⟨g, δ(responses)⟩ = (total kernels) · δp + ⟨λ, δS_f⟩,

its duals g paired with that run on the left, the kernels of
`ComputeCalibratedKernels` paired with the direction and the multiplier
with the target's change on the right: the two modes of one derivative,
each the other's check. Where the weights vanish the calibration holds
the initial sea level, and a direction's own initial sea level is its
change there (and is refused where the calibration sets it). Under frozen
shorelines A vanishes: δS0 is the right side, no pass runs, and the run
along δp is the total run. By the discrete tangent-linear runs it is the
derivative of the calibrated run as the program computes it, under every
scheme, each stage of Sdirk2 about its own shorelines.

Gates (`TestCalibratedKernels`): 1D, the marine cap, Sdirk2 and the
exponential trapezoid, a direction in the thickness at every node, the
relaxation times and the moduli with a change of the target: the total
tangent-linear run's objective against the total kernels' pairing 1e-12
and 3e-12 (the partial run misses it by 6% and 5%), in 7 runs (the run
along the direction, 5 along an initial sea level, the final one); the
change of the calibrated initial sea level against the difference of
calibrations afresh 6e-11 along the thickness and 4e-13 along the
target; the plain iteration reaches it in more runs. Frozen shorelines:
one run, δS0 against calibrations afresh 3e-13. 3D under lateral
viscosity, Sdirk2, a thickness direction and a lateral direction of the
log viscosity with the target moved: 5e-14 against the total kernels'
pairing, the partial run 24% from it. Through the program
(`TestAppsTangent`): `sl3d tangent`'s total run against pairs of `sl3d
solve` runs, each calibrating again, 8e-13 along a thickness history,
6e-9 along the target and 4e-7 along ln η (the differences' own error),
and against `sl3d kernel`'s total kernels paired with the perturbation,
2e-10; over radial segments (`TestAppsDistributed`) the one process's to
5e-12.

**Second order.** The second-order kernels of a calibrated migrating
run would be the derivative along a direction of the total kernels,
with the derivative of λ in it: not built. `SecondOrderKernels` (§9,
§11, §14) takes the initial sea level as a parameter and gives the
derivative of the partial kernels. So that this is not taken for the
calibrated run's, an evolution carries a mark of a state calibrated
under migrating shorelines (`Evolution::Calibrated`:
`CalibrateInitialSeaLevel` sets it, `SetBackgroundState` clears it, and
`sl3d kernel` sets it on the evolutions it builds about such a state),
and the second-order sweeps about a marked evolution, given ocean
functions, are refused naming what is not built
(`RefuseCalibratedSecondOrderKernels`) unless their
`Options::initial_sea_level_held` states that the partial kernels'
derivative is what is wanted. Under frozen shorelines the sweep is the
calibrated run's, and nothing is marked.

**Over ranks.** The multiplier's sweeps are the 3D sweeps, collective
as they are (§17); every rank holds the same surface fields, and
iterates alike.

**Gates** (`TestCalibratedKernels`; every difference between runs
calibrated afresh to 1e-13 of the sea level that flip no cell, the
start's included). 1D, the marine cap melting over five steps, discrete
sweeps, Sdirk2 and the exponential trapezoid: the total kernels against
the difference in the thickness at every node 8e-13 and 6e-13, in the
target (the multiplier) 1e-11 and 4e-13, in the relaxation times 4e-8
and 5e-8 and in the moduli 4e-8 and 8e-8 (Richardson's value of two
central differences; its remainder and the solves' noise are what is
left); the partial kernels miss the same differences by 21%, 25% and
22%. The final sweep's initial sea level kernel is the multiplier to
1e-10. A calibration weighted on half the cap: the thickness 8e-12, and
the initial sea level kernel off the weighted points against runs
calibrated from a guess moved there, 1e-11. Frozen shorelines: no sweep
for the multiplier, the kernels the one sweep's, 1e-12 against the
re-calibrated frozen runs. 3D under lateral viscosity, Sdirk2, four
nodes: the thickness 2e-10 and a lateral direction of the log viscosity
2e-9, where the partial kernels miss by 104% and 14%. Over radial
segments (`TestDistributed`, `TestRadialRanks`) the multiplier, the
total ice kernels and each rank's log-viscosity kernel are the one
process's to 1e-11. Through the program (`TestApps`): the total kernels
of `sl3d kernel` against pairs of `sl3d solve` runs, each calibrating
again, in the ice at a node 4e-14, in the target 2e-13, in a uniform
change of ln η 1e-6.
