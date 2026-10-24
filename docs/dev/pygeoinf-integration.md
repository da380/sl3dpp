# Inverse problems with pygeoinf: a plan

*Written 24 October 2026 against sl3dpp ebfc3d5 (develop) and pygeoinf's
`refactor` branch at 4844c5f, where v2 lives as `pygeoinf2/`, unfinished
and unreleased. A planning note: nothing here is built. It is to be taken
up again once pygeoinf v2 is released, when the D-numbers (pygeoinf's
`pygeoinf2/DECISIONS.md`) and the file references below need re-checking
against the released code. §8 lists features pygeoinf v2 could build in
for this use.*

## 1. The question, and the short answer

Can sl3dpp formulate and solve inverse problems through pygeoinf, with the
heavy computation in C++ here and pygeoinf conducting the inversion through
its matrix-free methods? Yes, and the two fit well: sl3dpp's adjoints
already return *derivatives*, which is the form pygeoinf v2 is built around
(D-26), and pyslfp's `FingerPrintOperator`
(`pyslfp/linear_operators/physics.py:131`) is the template: an L2 operator
whose forward and adjoint are solves, lifted to a Sobolev space by
`LinearOperator.from_formal_adjoint`.

The C++ effort should go into the solves and the changes of representation,
not into the spaces as such (§3).

## 2. What each side has

**pygeoinf v2.**

- The backend contract (`CURRENT_STATE.md` §6): a `CoordinateSpace` needs
  `dim`, `to_components`, `from_components`, `apply_gram`, `solve_gram`
  and `_key`. A coordinate-free backend supplies a `HilbertSpace` and a
  mass operator, as `backends/mfem_hilbert.py` does.
- An adjoint code enters through
  `LinearOperator.from_derivative_callables` (`algebra/operators.py:1662`):
  it hands over `d (A x, y) / d c_x` and the library applies the one Gram
  solve (D-26, D-88). `from_formal_adjoint` lifts an L2 operator to a
  weighted space.
- `Operator.at(x)` returns a `Linearization`, value and derivative sharing
  one backend call (D-29). `apply_block` and `_adjoint_apply_block` are the
  batching hooks (`algebra/operators.py:991`).
- Sobolev spaces on the sphere are diagonal-metric spaces (D-85, D-86).
  `sem1d.ball` and `sem1d.layered.Layered.ball` give Sobolev spaces on
  balls and annuli in the eigenbasis of `1 - div(L^2 grad)` on a padded
  GLL mesh, layers independent with jumps, continuity by conditioning
  (D-115 to D-126); the numerics are `planetmodel.randomfield`'s, in
  Python.
- `MaximumAPosteriori` (L-BFGS by default), `NewtonCG`,
  `TrustRegionNewton`, `gauss_newton_hessian`, `random_eig` and the
  low-rank operators, Backus-Gilbert, the data-space and model-space
  normal operators.

**sl3dpp.**

- Forward evolutions, 1D and 3D (`Evolution.hpp`, `Evolution3D.hpp`), and
  measurement duals (`Measurements.hpp`; `Measurement`,
  `include/sl3dpp/Adjoint.hpp:67`).
- `IceKernels::loadDuals` (`include/sl3dpp/Adjoint.hpp:97`):
  `dJ = sum_n int loadDuals[n] df_n dS`, an L2 density per time node.
- `Kernels3D::logViscosityKernel3D` (`include/sl3dpp/Adjoint3D.hpp:99`):
  the radial quadrature folded in, `dJ = sum_node int K dδ dOmega`, which
  is already a derivative in pygeoinf's sense.
- 3D has only the continuous adjoint (`include/sl3dpp/Adjoint3D.hpp:129`);
  the discrete 3D transpose is designed in [adjoints.md](adjoints.md) §6b
  and not built.
- δ enters `Viscoelastic3D` as a callable sampled at construction
  (`include/sl3dpp/Viscoelastic3D.hpp:105`).
- No tangent-linear model and no Python bindings. design.md §1 already
  places inversion drivers in an external framework.

## 3. Where the cost is, and the boundary rule

In an eigenbasis Sobolev space the space itself costs almost nothing: the
Gram map, its inverse, white noise and the covariance `A^-s` are
elementwise operations on a component array. What costs is

- the PDE solves: forward, tangent and adjoint;
- changing representation between the model's components and sl3dpp's
  internal grids: the map `P` from components to what the solver consumes
  (the ice load on the SLE grid at each time node; δ at the memory points
  on the coupling grid), and its transpose `P^T` from a kernel to
  derivative components.

Sizes for a 3D viscosity problem: a parameter field at lmax 32 with ten
radial modes in each of two layers has about 22 000 components; δ on the
coupling grid of a forward run at lmax 128 (about a thousand memory points,
oversampling 1.5) has of order 10^8 values, about 0.6 GB. Hence the rule:

**Only component vectors and data vectors cross the language boundary.**
`P` and `P^T` run in C++, on sl3dpp's grids, with GSHTrans's batched
transforms; the kernels never leave C++ at full size. The pygeoinf-facing
spaces can then be thin Python objects, and mostly pygeoinf's own.

## 4. Spaces

### 4a. The sphere: reuse pygeoinf's

`symmetric_space.sphere.Sobolev(lmax, order, length_scale, radius=a,
grid="GLQ")` is the right object and nothing in it is expensive. What C++
supplies is the conversion between pyshtools' real orthonormal
coefficients (no Condon-Shortley phase, D-86) and GSHTrans's complex
m >= 0 columns (`ShConventions.hpp`), as two functions: one for values and
one for duals. They differ by the (1, 2, 2, ...) column weights that bit
stage 5 twice, which is D-26 in another guise. `check_coordinates` and
`check_representer` catch it: the pairing axiom (D-18) fails by exactly
that factor.

### 4b. The ice history: a space-time tensor space

The model is `I(x, t)`. pygeoinf deliberately has no tensor product of
spaces, there being "no backend-independent representation" (D-23), but a
backend can supply one. `H^s(S^2) ⊗ H^q(T)` is diagonal in (spherical
harmonic × a temporal eigenbasis), so it is a `DiagonalMetricSpace` whose
metric is the product of the two symbols: about fifty lines. The temporal
basis could be `sem1d.interval`, which allows a time scale that varies
along the history.

The three views of the ice kernels correspond to three parameterisations:

| kernel view | pairs with |
|---|---|
| discrete, `loadDuals[n]` | values at the time nodes |
| distributional, `density` + `instantaneous` | a continuous-time basis: `int phi_k density dt + sum_i phi_k(t_i) lump_i` |
| quelled, `rate[k]` | a prior on the thickness rate `dI/dt` |

Start with the node values, which are exact on the time grid.

### 4c. Annuli and layers, for the log-viscosity

The parameter lives in the viscous layers only (not the elastic
lithosphere, not the fluid core), with jumps allowed where the model has
interfaces. Two routes:

**Route A (share the definition).** The model space is
`sem1d.layered.Layered.ball`, its layer bounds read from the same
planetmodel deck as sl3dpp's mesh so that they coincide. At setup Python
hands C++, layer by layer and degree by degree, the radial modes evaluated
at sl3dpp's memory points: a matrix `R_l` of shape (points, modes of degree
l), small. C++ then does

    P(c)(r_j, θ, φ) = sum_{l,m,k} c_lmk R_l[j, k] Y_lm(θ, φ)
    P^T(K)_lmk      = sum_j R_l[j, k] int K_j Y_lm dOmega

on the coupling grid. Because `Kernels3D` folds the radial quadrature in,
`P^T` is a pure transpose and its output is the derivative components.
Priors with `pointwise_std=`, jump operators, conditioning on continuity
and the shell, section and profile plots come free. `R_l` is reachable
today only through the space's `basis` (planetmodel's `SpectralBasis.evaluate`);
§8 item 9 asks for it as public API.

**Route B (port).** `RadialOperatorFamily` and the eigenbasis in C++,
parity-gated against `planetmodel.randomfield` to rounding, as sl3dpp was
gated against pyslfp. The costs: a symmetric eigensolver per degree, which
reopens the LAPACK question just closed for the band solves (6459e62), and
re-implementing the truncation, padding, Robin and fitting decisions
(D-118 to D-124). If it is done, planetmodel's C++ library is arguably the
home rather than sl3dpp: it owns `randomfield` in Python and the GLL and
radial machinery in C++, and YSpecpp and AdGIA could share it.

Route A first; Route B only if the `sl3d` program needs priors without
Python, or measurement says the Python side matters.

## 5. Operators

**The response operator.** The model goes to the response fields at the
observation nodes (sea level, displacement, potential, rotation), the
codomain a direct sum of Lebesgue spheres. Its adjoint takes dual fields to
one `Measurement` per node, runs the sweep and applies `P^T`. The adjoint
of an observation on an L2 sphere is exactly the L2 dual a `Measurement`
expects, so pyslfp's and pygeoinf's observation operators (tide gauges,
GRACE, altimetry path averages, point evaluation through the NUFFT, D-89)
compose on top unchanged. For large runs, a narrower variant keeps the
measurements in C++ (`Measurements.hpp`): a data vector out, weights `y`
in, and `sum_i y_i dual_i` built on the C++ side.

**Linear: ice to data,** under a fixed Earth and frozen shorelines. Build it
with `from_derivative_callables`. Override `apply_block` and
`_adjoint_apply_block` so that k measurement sets go through one reverse
sweep: the degree solves already take several right-hand sides and the
transforms batch. The randomised methods and the data-space normal
operator `J C J* + R` want exactly this.

**Nonlinear: δ to data.** Implement `_linearize`: a recording forward run
(stride about sqrt(N), or spilled to disk through `on_state`) returning the
value and a derivative whose adjoint is the `AdjointSolver3D` sweep under
the weighted measurements. L-BFGS on the MAP objective needs nothing more:
`MaximumAPosteriori` uses only `derivative.adjoint`
(pygeoinf's `inference/laplace.py:199`). Gauss-Newton-CG, the
model-space Laplace covariance and `random_eig` of the prior-preconditioned
Hessian need `J v`, the tangent model, as well; but see the next item.

**The data-space route needs only adjoints.** GIA data are few (10^2 to
10^4) against models of 10^4 to 10^6 components. With the kernels
`k_i = J* e_i` of every datum in hand,

    (J C J*)_ij = (k_i, C k_j)_X        (J v)_i = (k_i, v)_X

so the data-space normal matrix is formed exactly, `J` acts exactly through
the stored kernels, and the Gauss-Newton step and the Laplace posterior in
the data-space formalism need no tangent model: one adjoint per datum per
outer iteration, fewer sweeps when batched. pygeoinf has the pieces
(`LinearOperator.from_vectors`, whose adjoint is `J`). The pair `J`, `J*`
built this way is exactly adjoint by construction, which also removes the
Krylov consistency problem of §7 item 1 for the continuous 3D kernels.

**Joint ice and viscosity.** Nearly free: one sweep returns both kernels,
and the model space is a `DirectSum`.

## 6. What sl3dpp would build

1. Python bindings (§9).
2. `P` and `P^T` for the sphere (a conversion, §4a), the space-time ice
   space (§4b) and the layered annulus (§4c), batched.
3. A measurement set: forward pairing to a data vector, and
   `Weighted(y) -> map<node, Measurement>` for the adjoint.
4. Batched adjoint sweeps: k objectives through one reverse sweep, 1D and
   3D.
5. δ supplied as values on the memory-point grid, reusing the reference's
   flavours: the reference effective operators do not depend on δ, yet a
   new model means a new `Viscoelastic3D` today.
6. A tangent-linear model of the 3D evolution with respect to δ. δ enters
   only through `itau_ref exp(-δ)`, so the tangent is the same step with an
   extra source term in the memory update, and `OpenStep::Converge` serves
   it. Gated by a Taylor test and by `(J v, y) = (v, J* y)`.
7. The discrete 3D transpose ([adjoints.md](adjoints.md) §6b), for the
   Krylov and low-rank methods (§7 item 1).

## 7. Things that will bite

1. **Adjoint consistency.** pygeoinf's Krylov, Lanczos and randomised
   methods assume the adjoint belongs to the operator as implemented. The
   continuous adjoint differs from it at the level of the discretisation:
   CG on the normal equations then sees a non-symmetric operator and
   stalls, and the solvers are strict by default (D-8), so they raise. The
   continuous adjoint is right for the gradient of a MAP objective (the
   preferred formulation, the right problem); the discrete one is right for
   the linear algebra; their gap stays the resolution diagnostic. The data
   route of §5 sidesteps this.
2. **Tolerances.** The forward is linear only to its SLE and coupled
   tolerances, 1e-9 by default (`Evolution3D.hpp` Options); pygeoinf's
   outer default is 1e-8 (D-8). That is too close: loosen the outer to
   about 1e-6 or tighten the inner.
3. **Units.** A constant factor in a gradient does not change convergence,
   but it shifts the balance of prior and likelihood in a posterior. The
   cross-project non-dimensionalisation (pyslfp's T = 805 s against
   planetmodel's 1648 s) must be settled first; the bindings take and
   return one documented frame and convert inside C++.
4. **Which kernel to feed.** `logViscosityKernel3D` is derivative form:
   feed it to `P^T`. `io::WriteKernels` divides by the node weight times
   r^2 to give a per-unit-volume field, which is a gradient, for plotting:
   never feed that one. `loadDuals` are L2 densities and need the
   quadrature-weighted analysis.
5. **Two forward runs per MAP evaluation.** The objective's value goes
   through `chi_squared` (pygeoinf's `inference/problem.py:165`) and its
   derivative calls `forward.at` again (`inference/laplace.py:192-208`);
   the line searches call the functional and then `at` at the same point
   (`numerics/line_search.py:135, 197`). Without a cache every trial costs
   two forward runs and an adjoint. §8 item 1.
6. **The contrast envelope.** The within-step lateral iteration converges
   only within an envelope (design.md §6: 1000x at Δt/τ up to 2; at
   Δt/τ = 10, 10x converges and 100x does not), and an optimiser step can
   leave it. A forward that does not converge must fail loudly, and the
   line search must back off (§8 item 2). Moving the spherical mean of δ
   into the reference between outer iterations (the geometric-mean default
   of the model-file reader) keeps the iteration nearer the middle.
7. **Processes and threads.** pygeoinf's `n_jobs` runs joblib's process
   pool (`parallel.py`); a single 3D state is gigabytes (21 GB at lmax
   256). An sl3dpp operator must never be pickled into workers: batch in
   C++ instead (§8 item 7). Release the GIL around every C++ call, and keep
   the evolution, grid and trajectory alive from the Python objects that
   use them; pygeoinf's MFEM lessons (`CURRENT_STATE.md` §6) apply as
   written.

## 8. Features pygeoinf v2 could build in

Checked against `refactor` at 4844c5f. In order of how much they matter
here.

1. **One forward run per point.** `MaximumAPosteriori.objective` is built
   by `Functional.from_callables(space, value, derivative=...)` with no
   `linearize=` (`inference/laplace.py:192-208`), so value and derivative
   each run the forward. Wanted: the objective (and any other objective
   assembled from a forward problem) shares one `forward.at(model)` through
   `linearize=`; and a library-level last-point cache for an operator
   (by identity, then by a hash of the components), since the line searches
   evaluate the value and then `at` at one point.
2. **An evaluation-failure protocol.** An exception type pygeoinf
   recognises, raised by a forward that cannot converge, which the line
   searches treat as a step too long and the optimisers report rather than
   crash on. Armijo backtracking already survives a NaN or infinite value,
   because the comparison fails (`numerics/line_search.py:135-136`); the
   strong-Wolfe zoom with such values is untested, and an exception
   propagates and loses the optimiser's state after hours of forward runs.
3. **The data-space formalism for nonlinear problems.**
   `MaximumAPosteriori.linearized_normal` fixes `formalism="model_space"`
   (`inference/laplace.py`, the `NormalOperator` call). Wanted: a data-space
   choice for Gauss-Newton steps and the Laplace posterior, and the mirror
   of `with_probed_adjoint` (D-34): a derivative assembled from `dim(Y)`
   adjoint applications through `_adjoint_apply_block`, an exact
   `J`, `J*` pair from kernels (§5).
4. **A low-rank Laplace posterior.** The large-scale standard: the
   leading eigenpairs `(V, D)` of the prior-preconditioned misfit Hessian
   `C^1/2 J* R^-1 J C^1/2` by `random_eig`, two batched sweeps per probe,
   and the posterior `C^1/2 (I - V D (D + I)^-1 V*) C^1/2` with sampling,
   pointwise variance and no Krylov solve per application; `D` also gives
   the information gain and the data's resolution. Present: `random_eig`,
   `LowRankEig`, `WoodburyPreconditioner`, `low_rank_surrogate` (which
   truncates the prior's and error's factors, a different thing). Absent:
   this posterior.
5. **`check_operator` with a tolerance and a measured defect.**
   `testing.check_operator` (`testing.py:466`) takes no tolerance. Wanted:
   `rtol=`, and the relative defect of the adjoint identity returned or
   reported, so that a test can assert 1e-8 for a discrete adjoint and
   record the defect of a continuous one as it falls under refinement.
6. **Blocks everywhere, with a size hint.** Every consumer that applies an
   operator many times (the range finder and `random_eig`, `matrix()`, the
   data-space normal operator, trace and diagonal estimation, Backus-Gilbert)
   goes through `apply_block` and `_adjoint_apply_block`; and an operator
   can state a preferred block size, since a batched sweep's memory grows
   with the block.
7. **Process safety.** An operator can declare that it must not be sent to
   worker processes, the declaration carried through the expression nodes,
   so that `parallel_map` refuses or runs it in one block rather than
   pickling a 20 GB evolution because a caller set `n_jobs`.
8. **A Kronecker space of diagonal-metric factors.** D-23 revisited
   narrowly: for coordinate spaces with diagonal metrics (every symmetric
   space and every `sem1d` space, in components) the product is
   representable on components, with the metric the outer product of the
   diagonals and separable `spectral_operator` and `sobolev_measure`. For
   space-time ice priors (§4b), sphere times `sem1d.interval`. sl3dpp can
   supply it as a backend if pygeoinf would rather not.
9. **A separable-basis export from `sem1d`.** For Route A (§4c): per
   layer and degree, the radial modes at given radii, with the component
   ordering (degree; cosine orders 0..l; sine orders 1..l; radial index
   innermost) and the harmonic convention as a stated contract, and a test
   that synthesis from the exported arrays reproduces `evaluate`.
10. **Optimiser checkpoint and resume.** The descent methods keep their
    state internally (`numerics/optimization.py:229-282`), with no
    per-iteration callback and no way to save it and resume. Runs on a
    walltime-limited node with evaluations of minutes to hours need both.
11. **Resource-owning linearisations.** A `Linearization` that holds a
    recorded trajectory can be released explicitly, and the optimisers
    drop the old one before forming the next where the algorithm allows.
12. **Accuracy negotiation for inexact operators** (a design question,
    lower priority). Inexact Krylov theory lets the inner tolerance grow as
    the outer residual falls; a solver that told the operator the accuracy
    it needs per application would let sl3dpp set its SLE and coupled
    tolerances per call. At least: a strict solver that stalls at an
    operator's accuracy floor could say so rather than report a plain
    failure to converge.

## 9. Bindings and packaging

nanobind with scikit-build-core, in a `python/` directory behind a CMake
option: a module of raw C++ objects (`sl3dpp._core`) and a module of
pygeoinf adapters, or the adapters in a package of their own. In-process
bindings beat driving the `sl3d` program through configs and files because
the expensive setup persists between applications: the Wigner tables, the
Love-number caches, the reference flavours. The file route stays for batch
kernel production on the server. Pin a pygeoinf commit until v2 is
released.

## 10. Order of work

1. Bindings, and the elastic fingerprint as a response operator on
   pygeoinf's sphere; `check_space`, `check_coordinates`,
   `check_operator` pass; parity with pyslfp's `FingerPrintOperator`.
2. The 1D viscoelastic ice history: the space-time space, batched discrete
   adjoints, a data-space linear inversion.
3. δ by values, Route A with `P` and `P^T`, L-BFGS MAP on a synthetic
   lateral anomaly from relative sea level and uplift rates; then the
   data-space Gauss-Newton of §5 with the continuous kernels.
4. The tangent model and the discrete 3D transpose: model-space
   Gauss-Newton-CG, the low-rank Laplace posterior, the joint ice and
   viscosity problem.
5. Route B, only if wanted.

## 11. Open decisions

- The unit frame at the bindings, and with it the cross-project
  non-dimensionalisation.
- Route A or B for the annuli; for B, sl3dpp or planetmodel's C++ library.
- Where the Python layer lives: in sl3dpp, or a package of its own.
- Whether pyslfp moves to pygeoinf v2 first, so that the parity tests of
  §10 run on one version across the stack.
