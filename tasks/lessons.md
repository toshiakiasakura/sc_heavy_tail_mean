# Lessons

Accumulated gotchas so the same mistake isn't repeated. Newest first.

## `Meta.parseall` does NOT throw on a syntax error 2026-08-02 (verification tooling)

- Used as a cheap "does this file still parse?" gate after bulk edits, `Meta.parseall(read(f,String))`
  is **worthless on its own**: it returns an `Expr(:toplevel, …)` whose args contain
  `Expr(:error, ParseError(...))` nodes, and *returns normally*. Two files with a stray `"""` sailed
  through a `Meta.parseall(...); println("parses OK")` check and only failed later at `include` time —
  the same class of false-pass as the scoringutils column check (`nrow > 0` passing with every metric
  gone). **Assert on the parsed content, not on the absence of an exception:**

  ```julia
  ex = Meta.parseall(read(f, String))
  any(a -> a isa Expr && a.head in (:error, :incomplete), ex.args) && error("parse error in $f")
  ```
- The bug both times was the same mechanical one: a scripted block replacement whose replacement text
  ended with `"""` while the retained text began with `"""`, producing an empty docstring `"""\n"""`
  followed by an orphaned signature line. When splicing around Julia docstrings by line range, check
  the boundary lines of BOTH the removed and the retained text.

## The per-cell dispersion RE was REMOVED — it is not identifiable here 2026-08-02 (`joint_model.jl` §4.3) (user request)

**Do not re-add a per-age-pair-cell dispersion random effect without reading this.** It has now been
implemented and withdrawn THREE times (2026-07-11 add+revert same day; 2026-07-30 `-hd` flat
hierarchy; 2026-08-02 `-rhs` regularised horseshoe). The current model is block-linear × week only,
`log d_{ij,t} = β[bl,t]`.

- **The horseshoe's τ₀ response is a CLIFF, not a gradient.** Measured at origin 2021-05-09 h1, one
  fit per step: τ₀=0.1 ⇒ τ posterior 7–15 prior SDs out, slab inflated to c=29.3/2.93 until it never
  bound, λ never left its init (a plain hierarchical RE wearing horseshoe clothes). τ₀=0.01 ⇒ nominal
  scale cut ≈4× but realised within-block spread cut only ≈5–12%, because the posterior routes around
  τ₀ through `z` (implied sd(z) rose 0.55→2.13 negbin, 0.31→1.08 hweibull — `δ = τλ̃z` and `z ~ N(0,1)`
  is free). τ₀=0.005 ⇒ monotone but modest (−48% c→c to −15% a→a). τ₀=0.001 ⇒ RE extinguished
  outright (within-block SD 0.000, m_eff 0.00, τ back inside its prior). **There is no setting that
  selects a few informed cells and shrinks the rest** — which is the entire point of a horseshoe.
- **The cause is identifiability, not tuning.** 49 ordered cells per week, many of them empty (the
  per-week hurdle-Weibull cells are frequently p⁰=1 throughout). The data do not inform a per-cell
  dispersion; the block mean is what the window supports. Reaching for a different prior family, or a
  per-block/per-week scale, does not change that.
- **A shrinkage diagnostic that looks reasonable can still be measuring nothing.** The 11j panels
  rendered fine at every τ₀ — legends, bands, sensible-looking ranked-δ staircases. What exposed the
  degeneracy was comparing the *realised* within-block SD against the previous generation's, not the
  shrinkage statistics computed from the fitted horseshoe's own parameters.
- **The verification that the RE is gone is `plot_within_block_sd`.** The current generation's
  within-block SD of log-dispersion must be **identically 0** (block-constant by construction), with
  the retained `-hd` line non-zero beside it. A cheap structural assertion beats eyeballing a map.
- **Reverting the hierarchy is NOT the same as returning to the pre-hierarchy model.** Commit
  `9a801db` bundled three independent changes under one token bump — `-hd` (the hierarchy), `-p0`
  (fitted hurdle p⁰, +588 latents and a Binomial roster term, weighted path only) and `-gi` (the
  generation interval became sampled `w_mu`/`w_sigma` in Stage 2) — and `0e56fc7` then changed
  `ref_bin` 1→4 and `susc_inf_sd_prior` **without bumping the token at all**. Only `p0` and `gi` are
  reflected in the current token `temporal-gsar-cut-sc-p0-gi`. Consequence for the `dt_intermediate_old/`
  cache (token `temporal-gsar-cut-sc`, a complete 63-origin × 4-horizon × 2-family grid): its
  **unweighted-negbin Stage 1 targets the same posterior** as the reverted model (NegBin has no
  hurdle, and its clamp [-4,5] never moved — verified by comparing latent sets, which are identical),
  but its **hurdle-Weibull Stage 1 does not** (no `p0f`, κ clamp was [-3,3]) and **none of its Stage 2
  does** (no `w_mu`/`w_sigma`, old `ref_bin`, old prior). Check the parameter names in the chain, not
  the filename token, when asking whether a cached fit matches the current model.
- **Bundling unrelated changes into one token bump is the root cause of that mess.** One token
  component per independent change, and bump it whenever the posterior moves.


## τ₀ is per-family and can only be set EMPIRICALLY 2026-08-02 (`framework.jl`, `src/tune_tau0.jl`) (user request)

> **SUPERSEDED 2026-08-02 (same day):** the horseshoe was removed entirely and `src/tune_tau0.jl` deleted — see the dispersion-RE entry at the top. The methodological point (measure shrinkage from fitted chains; P&V's τ₀ formula does not transfer to a non-linear likelihood) still stands and is why the removal was justified rather than assumed.

- **Piironen & Vehtari's τ₀ formula does not apply to this model, and reaching for it would have been
  wrong.** `τ₀ = p₀/(D−p₀)·σ/√n` is derived for a LINEAR model: it needs a residual scale `σ` and a
  sample size `n`. A NegBin / hurdle-Weibull likelihood on counts and duration histograms has
  neither. The user caught this ("this is not a linear regression") before it was acted on.
- **A prior-predictive Monte Carlo of `m_eff` is well-defined but answers the wrong question.** It
  was cheap to build (`prior_shrinkage_reference`) and gives a clean τ₀-vs-escape curve — measured
  E[escape] = 1.8% at τ₀=0.1, 9.8% at τ₀=0.348 — but it describes only the PRIOR. At τ₀=0.1 the
  fitted τ came back 7–15 prior SDs out, i.e. the likelihood simply overwhelms the prior, so
  prior-predictive escape says nothing about realised escape. **Measure it from fitted chains.**
  Keep the prior-predictive band anyway: it is the reference `m_eff` must be read against, because
  `shrink` is a slab-vs-spike fraction with a NON-ZERO prior baseline (≈0.998 ⇒ `m_eff` ≈ 1 of 49).
- **One τ₀ cannot serve both degree families.** Measured at origin 2021-05-09 h1 under a shared
  τ₀=0.1: per-cell multiplier `τ·λ̃` 1.61 (NegBin φ) vs 0.73 (Weibull κ); within-block SD of
  log-dispersion 0.5–2.4 vs 0.05–0.42. Split into `disp_re_scale_prior_unweighted` / `_weighted` and
  read ONLY through `disp_tau0_prior(cfg, dm)` — the two are tuned independently and diverge.
- **The escape-fraction panels are NOT comparable across families.** `shrink = c²/(c²+τ²λ²)` is
  relative to each fit's own slab, and `c` differed 10× (29.3 negbin vs 2.93 hweibull). So hweibull
  showed 20× the escape while having *less than half* the absolute RE. Comparing families requires
  the multiplier `τ·λ̃` or the within-block SD, never the escape fraction.
- **Tuning a prior scale REQUIRES the cache token to encode it.** `contacts_label` did not, so a
  refit at a new τ₀ writes the same filename and `fit_or_load_stage1`'s `isfile` short-circuit
  reloads the previous step — the same footgun as γ_SAR 2026-07-13 and `stage1_pathfinder_runs`
  2026-07-30, which is now three times. `_tau0_tag` puts both values in the token so steps are
  non-colliding by construction and stay side by side for comparison.
- **Knock-on: a load-time `const` token goes stale the moment a prior it encodes becomes tunable.**
  `CONTACTS_TOKEN` is built from the DEFAULT `FrameworkConfig`, so every helper that receives a `cfg`
  now defaults to `contacts_label(cfg)` instead; `CONTACTS_TOKEN` is only for helpers with no `cfg`
  in scope.
- **A docstring inserted between an existing docstring and its definition breaks the module load**
  with "cannot document the following expression" — the first docstring binds to the second *string
  literal*. Cost one full load cycle. When adding a documented helper near another, put it entirely
  ABOVE the neighbouring docstring, not between it and its function.
- **`include` in a script resolves relative to the SCRIPT's directory, not `cwd`.** A driver in the
  scratchpad cannot `include("forecast_utils.jl")` even when run from `src/`; pass the source dir
  explicitly (`ENV["SRCDIR"]`). Separately, the framework's data paths still require `cwd == src/`,
  so both have to be right at once.

## Regularised horseshoe on the Stage-1 dispersion RE 2026-08-02 (`joint_model.jl`, `framework.jl`, 10j/11j viz, `inst/3` §4.3) (user request)

> **SUPERSEDED 2026-08-02 (same day):** the horseshoe is no longer in the model — see the dispersion-RE entry at the top. Kept because the AD findings below are general (they apply to ANY global×local×slab composition, and to `_softclamp` vs `_softcap` generally), and because the code is recoverable from commit 7cd11c2.

- **The composition form is an AD constraint, not style — and three of the four natural ways to write
  it are wrong.** The RE multiplier is `τ·λ̃ = c·u/√(c²+u²)` with `u = τλ`. Measured with
  `ReverseDiff.gradient` at the corner cases: the literal eq.-11 `c²λ²/(c²+τ²λ²)` NaNs when `λ²`
  overflows; an `exp(_softclamp(log(u), …))` guard NaNs **in the gradient** at `u=0`
  (`d log u/du = 1/u → ∞`) *and* floors the value at 4.7e-14 instead of 0; `c·√w` with
  `w = u²/(c²+u²)` NaNs in the gradient at `w=0`. Only `c*u/sqrt(c^2+u^2)` is finite in value **and**
  gradient at both ends (at `u=0`: grad `[0, 0.38, 0, 0]`).
- **`u = 0` is the corner that matters, and it is the one a smoke test misses.** Under a working
  horseshoe *most* cells sit at full shrinkage, so a NaN there poisons the whole gradient every
  iteration — and both rejected guards return a perfectly finite **value** at that point. A "does the
  model evaluate?" check passes while the fit is dead. Assert `all(isfinite, grad)`, not just
  `isfinite(logp)`, and assert it AT the full-shrinkage corner.
- **`_softcap(x, hi) = hi − softplus(hi − x)`, NOT `_softclamp(x, 0, hi)`.** `_softclamp` is only
  ≈identity *deep* in the interior; near `lo = 0` it distorts —
  `_softclamp(0.7, 0, 1e6) = softplus(0.7) = 1.10` — which would put a floor of ≈log 2 under λ and
  destroy the shrinkage the horseshoe exists to provide. The upper-only form is exact there
  (`1e6 − softplus(1e6 − 0.7) = 0.7`) and inherits the same Inf-safety and ReverseDiff-safety.
- **`logpdf(TDist, x)` → −Inf once `x²` overflows (`x > 1.34e154`) — but so does `logpdf(Normal, x)`,
  at exactly the same point.** That is upstream of the composition, shared by every `z`/`z_c`/`z_k`
  latent already in the model, and NOT fixable by `_softcap`. Don't mistake it for a horseshoe
  defect; the honest gate asserts the two behave identically there.
- **A LogDensityFunction built over an UNLINKED VarInfo silently misreads an unconstrained vector.**
  This cost a full debug cycle: `logdensity_and_gradient(f, x)` returned NaN at *every* point,
  including a tame one, because `tau`/`c2` were being read as constrained values and landed negative.
  The model was healthy the whole time (`logjoint` finite, priors finite, moments finite). Build it
  as `LogDensityFunction(model, DynamicPPL.getlogjoint, link!!(VarInfo(rng, model), model); adtype)`.
  **When every point fails, suspect the harness before the model.**
- **`λ = 1` and `c² = s²` are the EXACT prior modes in the unconstrained coordinates**, for any ν and
  s. For `v = log λ` under half-t_ν: `dlogp/dv = 0 ⇔ ν + e^{2v} = (ν+1)e^{2v} ⇔ λ = 1`. For
  `w = log c²` under `InvGamma(ν/2, νs²/2)`: `dlogp/dw = 0 ⇔ e^{−w} = 1/s² ⇔ c² = s²`. So pinning
  them in `_stage1_init` is principled, not a fudge — and it reproduces the pre-horseshoe geometry,
  which is why the 2026-07-30 `z_init_scale = 0.1` sweep carries over unchanged.
- **`lam` MUST NOT be initialised from its prior.** `_stage1_init`'s filter is `startswith(name, "z")`,
  which `lam`/`c2` do not match, so they would have been prior draws. A half-t₃ over 588 cells
  routinely yields values in the tens — several cells start *inside the slab* with a full-strength RE
  before the likelihood has said anything. Also note index-range inits would now be doubly wrong:
  adding `lam`/`c2` and shrinking `tau` shifts every block (`z_kappa` was 415:1002 of 1590).
- **The tail exponent IS the restoring force — that is why half-t₃ converges better than half-Cauchy.**
  In `v = log λ` the half-t_ν log-density behaves as `−ν·v`, so measured `dlogp/dv → −3.000` for ν=3
  vs `−1.000` for ν=1, and `q99.99` drops from 6366 to 28 (227× lighter). Not folklore — check it
  numerically before claiming a prior "converges better".
- **`shrink = c²/(c²+τ²λ²)` has a NON-ZERO prior baseline (≈0.998), so `m_eff` has a prior floor of
  order 1 per 49-cell week, not 0.** Reporting "m_eff = 3" without the prior-predictive band would
  badly overstate how many cells escaped. `prior_shrinkage_reference` Monte-Carlos it through the
  identical formulas; every panel draws the band. Also: this quantity is the *slab-vs-spike* fraction,
  NOT Piironen & Vehtari's `κ_j = 1/(1+nσ⁻²τ²λ̃²)`, which is scaled by data information. Same
  direction, different denominator — don't quote one as the other.
- **`generated_quantities` CANNOT read a previous-token chain.** `stage1_moment_draws` runs the
  *current* `model_degree`; against an `-hd` chain with no `lam`/`c2` and a vector `tau` it errors
  or, worse, silently re-draws the missing latents from the prior and returns plausible numbers.
  Every cross-generation comparison must go through the `reconstruct_*` mirrors in `10j_viz_utils.jl`,
  which is why `_read_disp_chain` keeps an explicit `legacy` branch keyed on the presence of `c2`.
- **Token `-hd` → `-rhs`; the old files are kept ON PURPOSE.** Both generations coexist on disk by
  filename, which is what makes the before/after possible with zero refits of the old. Also replaced
  the seven hard-coded token defaults across `8j`/`10j` with `CONTACTS_TOKEN`/`CONTACTS_TOKEN_HD` in
  `framework.jl`, so the next bump lands in one place.
- **Name collision watch:** `model_degree` already binds `c` (the GP level intercept), so the slab
  scale is `c_slab`. Shadowing it would silently corrupt the contact mean, not error.
- **`ETp` promotion:** `tau`/`c2` are SCALARS now, so it is `typeof(tau)`/`typeof(c2)`, not `eltype`.
  Miss one and `K1/K2/G` are allocated `Float64`, cutting the tape for that latent — silently.
- **Open question flagged, not resolved:** Pathfinder fits a single multivariate normal, and a
  regularised-horseshoe posterior is spike-and-slab-shaped and heavy-tailed in `log λ` in ~1600–2200
  dimensions. The λ draws may be closer to the approximation than to the posterior. Confirm with one
  `cfg.stage1_use_nuts = true` run at the pilot origin before trusting the shrinkage numbers.

## scoringutils drops metric COLUMNS on drifted quantile levels — and `score_wis` was never once executed 2026-07-30 (`scoring.jl`, `CLAUDE.md`)

- **Context: `score_wis` had ZERO runtime coverage.** R was installed, `scoringutils` was not, so the
  entire WIS path had never run — through every prior session of this framework. Worth asking, for
  any "verified" pipeline: which paths has the environment *silently prevented* from executing?
- **The old CLAUDE.md gotcha was wrong in two ways.** It said fp drift from
  `collect(0.05:0.05:0.95)` breaks `interval_coverage_50` "while the 90% endpoints still work".
  Measured (`verify_scoring.jl`): (a) **`collect` on a Julia range does NOT drift** — 0/19 levels
  differ from their 2-dp rounding, because ranges carry `TwicePrecision` internals; the risk lives in
  constructions like `cumsum(fill(0.05,19))`. (b) The 90% endpoints do **not** survive. One drifted
  endpoint makes the whole interval set asymmetric, so scoringutils refuses **`wis`,
  `overprediction`, `underprediction`, `dispersion`, `ae_median` AND both coverages** — `by_model`
  came back as literally `["model","scale","bias"]`.
- **The columns are ABSENT, not NA, and R emits only WARNINGS**, so `score_wis` returns a
  well-formed frame and Julia raises nothing. **`nrow(sc) > 0` passes with every metric gone** —
  which is exactly what the original smoke test asserted. *Assert on columns and their VALUES, never
  on row count.* `score_wis` now hard-errors on any missing metric column, naming the dropped
  columns and printing which endpoints (0.05/0.25/0.5/0.75/0.95) were actually matched.
- **Test coverage against its NOMINAL level, not just "populated".** First attempt built synthetic
  draws centred exactly on `observed` ⇒ coverage 1.0 at *every* level, which cannot tell a working
  50% interval from a broken one. Drawing `observed` as an **independent draw from the same
  predictive law** over 560 units recovers 0.498/0.896 vs nominal 0.50/0.90 — a broken 0.25/0.75
  match cannot produce 0.498.
- **`try`/`catch` opens a new scope: a flag set in the `catch` does not escape.** `threw = false`
  outside, `threw = true` inside the catch ⇒ a *local*; the probe reported "no error raised" when the
  guard had fired correctly. Same soft-scope trap as a bare top-level `for`. Wrap the probe in a
  **function that returns** the outcome.

## `fit_concurrency` silently returned 1 on macOS 2026-07-30 (`joint_model.jl`)

- **`Sys.free_memory()` is NOT "available memory" on darwin.** `_mem_available_gib` read
  `/proc/meminfo`, which **does not exist on macOS**; the bare `catch` swallowed the `SystemError`
  and fell through to `Sys.free_memory()` = libuv's `uv_get_free_memory()` = **truly-free pages
  only**. macOS deliberately keeps that near zero (memory is held as inactive/cached, not freed):
  **measured 2.09 GiB on an idle 32 GiB machine**. Then
  `mem_cap = floor((2.09 − reserve 4.0)/1.0) = 0` ⇒ `fit_concurrency() = max(1, min(cpu_cap, 0)) = 1`
  ⇒ **fully serial fitting, with no warning**. The `max(1, …)` floor made it safe, not correct.
- **Fix:** a `Sys.isapple()` branch parsing `vm_stat` — **free + inactive + speculative** pages ×
  page size, the darwin analogue of Linux `MemAvailable`. Measured 14.0 GiB vs 2.09 GiB on the same
  idle machine ⇒ `mem_cap` 0 → 10. **`purgeable` is deliberately NOT added**: it is a subset of
  active/inactive, so it would double-count. The Linux branch is now explicitly `Sys.islinux()`
  rather than try-and-fall-through.
- **It was invisible because the notebooks store zero outputs.** `8j` printed the value and nothing
  else, and all three notebooks are committed with no outputs, so no artefact ever recorded it.
  Added `fit_concurrency_report()` → `(; concurrency, mem_cap, cpu_cap, avail_gib, nthreads,
  cpu_threads, binding)` and wired it into 8j's setup cell. **Print the breakdown, not the number** —
  `binding` names which cap actually bit.
- **Second darwin under-report, documented NOT fixed:** on Apple silicon `Sys.CPU_THREADS` reports
  **performance cores only** (`hw.perflevel0.logicalcpu` = 4) while `hw.ncpu` = 10 (P+E). So
  `cpu_cap = min(nthreads, CPU_THREADS−1)` saturates at 3. That is a defensible cap for compute-bound
  Pathfinder fits (E-cores contribute little, oversubscription hurts), so it is left alone — pass
  `max_concurrent` to override. **A return of 1 usually means `Threads.nthreads()==1`** (Julia
  started without `-t`/`JULIA_NUM_THREADS`), not the memory cap; the report says which.
- **Lesson beyond this function:** a resource heuristic that silently degrades to the safe-but-slow
  branch is worse than one that errors. Measure it and print the breakdown before trusting it to
  size a multi-hour run.

## Formal model: hierarchical dispersion + fitted p⁰ + estimated GI + t₀+h antibody 2026-07-30 (`inst/5_formal_pathfinder_impl.md`)

- **Dispersion is now a two-level hierarchy with a SHARED per-week scale.**
  `log_disp_{ij,t} = β[bl,t] + τ_t·z[pcode,t]`, non-centred, `pcode=(i−1)A+j` over the **49 ordered**
  cells (directional blocks ⇒ ordered, not the 28 unordered pairs the contact *mean* uses).
  `tau ~ filldist(N⁺(0, cfg.disp_re_scale_prior[2]), Tn)` — **one τ per week, shared across the four
  blocks**. The user rejected a per-block `σ_XY,t`: child→child has only `2×2=4` ordered cells and
  the scale is re-estimated every week, so 12 SDs from 4 observations each would just sit on the
  prior. This re-derives the **same conclusion as the 2026-07-11 attempt** (reverted then, restored
  now) — `git show a9a2953` is the reference implementation; restore it rather than re-deriving.
- **Prior scale 0.5, NOT the old 0.109.** The 2026-07-11 value was E[τ²]-matched to the observed
  *between-block* homogeneity of κ; τ here scales *within-block between-cell* spread, which has
  never been measured. 0.5 ⇒ a typical cell within ≈[0.37,2.7]× its block mean at ±2 SD.
  **Watch BOTH directions** (10j `plot_tau_over_weeks`): τ hugging the prior ⇒ the RE is not earning
  its place; τ far ABOVE the prior ⇒ `m + τ·z` saturating the `log_kappa` clamp.

## κ soft-clamp WIDENED `[-3,3]` → `[-4.3,5]` 2026-07-30 (`joint_model.jl`, `10j_viz_utils.jl`, spec) (user request)

- **Symptom:** after the hierarchy landed, EVERY fitted Weibull κ sat exactly on `0.0498` — the
  lower clamp — where the pre-hierarchy block-only fits had κ ≈ 0.88–1.03. Alongside it, τ_t medians
  ≈2 against a prior median of 0.34.
- **The τ reading was a SYMPTOM, not the cause — and the obvious fix was the wrong one.** A four-way
  sweep of the τ prior (1e-6 / 0.109 / 0.25 / 0.5) came out **non-monotone**: 1e-6 and 0.25 gave sane
  block means `[0.53,1.98,1.81,-1.46]` with κ interior, while **0.109 and 0.5 gave block means of
  −441 and −247** — ≈900 prior SDs from `Normal(0,0.5)`, i.e. a diverged LBFGS path, not a posterior.
  Non-monotone ⇒ optimiser-path luck, not a prior-scale effect. Tightening τ (the "obvious" fix,
  and the one the seam text originally recommended) would NOT have helped.
- **Mechanism:** the soft-clamp creates a genuinely FLAT region. Once `β + τ·z` leaves the bounds, κ
  is constant, the Weibull likelihood has zero gradient in β and τ, and LBFGS wanders arbitrarily
  far. The per-cell RE makes that region far easier to reach because 588 `z`'s now multiply τ. **A
  clamp that binds is not merely biased — it can destabilise the optimiser**, which is a stronger
  failure than the γ_SAR clamp-compression case of 2026-07-13.
- **FLOOR IS −4.446, NOT −5.14 — I got this wrong once; check EVERY `gamma` on the path, not the
  first one.** I widened to `lo=−5.0` having verified only `λ = μ/gamma(1+1/κ)`, and two sweeps then
  crashed inside `draws_to_chains`. `_weibull_moments` *also* forms
  `CV² = gamma(1+2/κ)/gamma(1+1/κ)²`, and `1+2/κ` overflows at **twice the κ**:

  | quantity | gamma arg | overflows at | log κ floor |
  |---|---|---|---|
  | `λ = μ/gamma(1+1/κ)` | `1+1/κ` | κ ≲ 0.00586 | −5.14 |
  | `CV² = gamma(1+2/κ)/gamma(1+1/κ)²` | `1+2/κ` | κ ≲ 0.01172 | **−4.446** |

  At `logκ=−5` λ is still finite (1.5e-263) so a λ-only check PASSES, but `CV² = Inf/Inf = NaN`
  propagates into `⟨k²⟩` and aborts the fit. Settled on **`lo=−4.3`** (≈0.15 margin). Failure mode
  differs from the λ one: NaN silently poisons the moments rather than throwing `Weibull: θ>0`.
  `diag_multipath.jl` now asserts finiteness at −4.3/−4.44 and non-finiteness at −4.5/−5.0, so a
  future mis-widening fails in seconds instead of after a multi-minute fit.
  The NegBin φ clamp has no analogous constraint (its moments are polynomial in `1/φ`) and is
  unchanged at `[-4,5]`.
- **The mirror must move in lockstep.** `reconstruct_dispersion_draws` hardcodes
  `lo, hi = weighted ? (-4.3,5.0) : (-4.0,5.0)`; leaving it at `(-3,3)` would silently break the
  reconstruct-matches-model invariant (which is checked to 3.9e-16, so it WOULD have been caught —
  but only if the check is re-run).
- **Hurdle p⁰ is FITTED** (weighted path only): `p0f ~ filldist(Beta(1,1), A*A, Tn)` with
  `n_zero ~ Binomial(n, p⁰)`. **The data was already there and dead**: `AgePairData.n` /`ds.n` was
  populated and never read anywhere, and `whist_nobs` (`degree_dist.jl:40`) was defined and never
  called. `n_zero = n − whist_nobs(pos_weight)` is EXACT (verified 588/588 cells, three independent
  routes). Do **not** use `dd_count.y[1]` — the zero bin is only pushed `if n_zero > 0`, so a cell
  with no zeros has no `x==0` row; `sum(y[x .== 0])` is the safe idiom.
- **Binomial written as the raw kernel** `nz*log(p0) + (nn−nz)*log1p(-p0)`, not
  `logpdf(Binomial(n,p0), nz)`: the `log C(n,k)` normaliser is data-only so the posterior is
  identical, it keeps `lgamma` out of a 49×Tn loop that runs on every gradient eval, and it avoids
  any question about AD through `binomlogpdf`. **Skip cells with `n == 0`** — there the stored
  `p0 = 1.0` is fabricated (`degree_agepair.jl:194`), not observed. Cells with `n>0, n_zero==n` are
  real data and must be kept.
- **The two degree models' Stage-1 parameter spaces now DIFFER** — only the weighted path declares
  `p0f` (+588 latents). Anything inferring regime from parameter names must account for it; it is
  also the cheapest way to tell the chain shapes apart.
- **Generation interval estimated** (Munday 2023 Eq 2 + Table 1). `w_σ` is the **LOG-VARIANCE**, not
  the log-SD: Table 1 builds its prior centre as `log((sd/mean)²+1)`, which *is* σ²_log, even though
  Eq 2 passes it as the second CDF argument and p.8 calls it a "log-standard-deviation". Taking the
  Table-1 reading makes the prior mean reproduce 5 d/5 d **exactly**, so the estimated-GI model
  **nests** the fixed-GI one (verified: `gen_interval_pmf_log(gen_interval_logparams(5,5)…) ==
  gen_interval_pmf(5,5)`). The sdlog reading would imply 4.5 d/3.5 d at the same centre.
- **The paper's printed GI prior is not a valid statement** — `w_μ` has prior mean −0.683 yet is
  written `T[0,]` with a *negative* prior SD (−0.1366). A negative meanlog is REQUIRED (a 5-day GI is
  shorter than a week), so only `w_σ` is truncated and `abs()` guards the SD.
- **Return latents POST-clamp.** `model_transmission` returns `w_mu`/`w_sigma` *after* `_softclamp`,
  so `gen_interval_pmf_log(w_mu[d], w_sigma[d])` downstream reproduces the fit exactly. Same
  convention `gamma_sar`/`susc`/`inf` already used. Returning raw latents would have silently
  desynced the forecast from the fit at the clamp boundary.
- **`w` is now PER DRAW — three consumers had to change together**: `fit_stage2_pooled` (stores
  `w_mu`/`w_sigma`), `two_stage_forecast` (the renewal lag sum `acc` **moves inside the draw loop**;
  it could no longer be hoisted per horizon), and `10j fit_window_infection_draws`. Miss any one and
  it silently uses the PRIOR-CENTRE GI while the fit used the posterior.
- **Antibody at t₀+h — forecast NGM only.** New `WindowData.antibody_fc` (A×H) field rather than
  widening `antibody`, so `antibody[:, t]` keeps its meaning in the Stage-2 fit loop and no existing
  `[:, end]` silently changes meaning. The fit loop stays t₀-anchored on the user's reasoning: its
  infection outcomes only exist up to t₀, so shifting it contributes nothing. `reproduction_draws`
  (8j viz) shifted too, so the plotted R describes the NGM the forecast actually uses.
- **`weekly_antibody` zero-fill now warns.** An unmatched week returned `0.0`, indistinguishable from
  genuinely zero prevalence — and zero antibody means FULL susceptibility, so a zero-filled forecast
  column inflates the forecast silently. It now collects the unmatched weeks and `@warn`s **once per
  call** (per week, not per cell — a missing week is missing for all A bins). The zero fill is kept
  as the value on purpose: `NaN`/`missing` would propagate into the NGM.
- **Cache token `…-sc` → `…-sc-hd-p0-gi`.** All 504 `8j_s1_*` + 1008 `8j_s2_*` are stale; full refit
  required. The token is SHARED by `stage1_path`/`stage2_path`, so Stage 1 cannot be spared even
  though `-gi` is Stage-2-only. Regenerate the 9j caches too — the forecast changed (per-draw `w`,
  t₀+h antibody) independently of the chains.
- **`gi_moments_days` lives in `renewal.jl`, not `8j_viz_utils.jl`.** First written as a viz helper,
  which broke any non-viz consumer: `forecast_utils.jl` does not include the `*_viz_utils.jl` files.
  Pure functions about a model parameterisation belong beside the parameterisation.
- **The space-form `@view` trap bit again, in VIZ code this time.** `quantile(@view τ[:, t], 0.05)`
  fails with *"Invalid use of @view macro: argument must be a reference expression"* — the macro
  greedily swallows the trailing `0.05`. The single-argument `median(@view τ[:, t])` on the adjacent
  line is fine, which is what makes it easy to miss. Worse, the error surfaces from `Base.Docs.docm`
  (docstring expansion) with **no line number for the offending call**, so it reads like a docstring
  problem. Use the `view(τ, :, t)` function form everywhere in an argument list — the rule is not
  specific to `_cell_moments!` where it was first documented.

## Null + no-interaction baselines and the log score 2026-07-30 (`inst/6_null_interaction_model.md`)

- **Read the existing forecast path before "adding" future contacts.** The spec asks that the
  no-interaction model "be allowed to use the future mean contacts". It already is: for horizon `h`,
  `fit_or_load_stage2` fits Stage 1 on `WeeklyWindow(origin + 7h)` and freezes the forecast NGM at
  `Cstar_end[m]` = the **origin+h** contact matrix (infections/antibody stay at the origin). That
  turned a feared re-architecture into a one-line `contact_star` method. **The null model is the
  only exception** — its `c̄` comes from the *origin* window's focal weeks and is reused for every
  horizon, which is what "fixed while forecasting" means.
- **Stage-1 chains carry no NGM token**, so `(NegBinAgePair, DiagonalMeanNGM)` reuses
  `8j_s1_unweighted-negbin_*` verbatim: the no-interaction model costs **zero** Stage-1 refits, and
  neither baseline invalidates a cached file (their Stage-2 tokens are new). Only the
  `9j_assembly_*` cache self-invalidates, on the model-label change.
- **`contact_star` broadcasts `base_contact` elementwise**, so a builder that needs the cell index
  (diagonal-only) must add a **`contact_star` method**, not a `base_contact` method — and must
  return a **dense** `Matrix`, because `fit_stage2_pooled` stores into `Vector{Matrix{Float64}}`
  (a `Diagonal` does not fit the slot).
- **Diagonal C* ⇒ `susc_a·inf_a` is only identified as a product**, hence `inf ≡ 1`. Drop `sig_i`/`z_i`
  from the parameter space entirely rather than sampling-and-ignoring them: an unused latent stays
  prior-driven and pollutes the Pathfinder approximation. Use plain `ones(A)` (not
  `ones(typeof(sig_s), A)` — `one(::Type{TrackedReal})` is not reliably defined) since no gradient
  flows through a constant.
- **The null model's `c̄` is unidentified with γ_SAR — by design.** `N = γ_SAR·fs_a·c̄·inf_b`, so the
  two enter only as a product and the *forecasts are invariant* to the `/A` convention; `c̄` only
  fixes what γ_SAR **means**. Measured: `c̄`×10 ⇒ posterior median γ_SAR ×**0.105** (the 5% excess is
  the log-normal prior's pull, not error) and median R unchanged to 0.3%. Corollary: use **M = 1**
  dummy moment draw with `n_draw = 100×100`, not 100 identical draws — there is no contact
  uncertainty to propagate, so 100 refits would add only fit-to-fit noise at 100× the cost.
- **`scoringutils` computes `log_score` ONLY for the sample forecast class** (`as_forecast_sample`
  → `scoringRules::logs_sample`, KDE) — never for quantile forecasts, so it cannot be a flag on the
  existing `score_wis`. It needs a second R path over the raw draws. Two traps there: the KDE cannot
  consume the `±Inf` draws `two_stage_forecast` deliberately keeps, and `log_shift` returns `NaN` on
  the negative draws a Gaussian fan contains (the quantile path rarely hits this because the 5%
  quantile is usually positive). Build the log-scale copy explicitly with `log(pmax(·,0)+1)`, and
  **count and print** every dropped/censored draw. Score **one origin at a time**: `score()` returns
  one row per forecast unit, so the accumulated table stays small while a single global sample table
  would be ~10⁸ rows.
- **Relative log score must be a DIFFERENCE, not a ratio** — a log score is not sign-stable (it goes
  negative wherever the predictive density exceeds 1), unlike WIS.
- **Naming trap**: "log-scale WIS" (`transform_forecasts(log_shift)`) is WIS after a log *transform*
  and is **not** a log score. The repo had the former since 2026-07-12 and none of the latter.
- Per-model panel figures (`plot_ratio`, `plot_ratio_bins`, `plot_lengthscales`) were hard-coded
  `layout = (2, 2)` from when there were exactly four models — they silently lose panels at six.
  Fixed with `panel_grid(np)`; this is the 2026-07-15 "derive the layout" lesson biting again.
- `9j_viz_utils.jl:22` had documented `unweighted-negbin|mean` as *standing in* for Munday's
  no-interaction reference. Now that a real one exists, `REF_MODEL` is
  `unweighted-negbin|mean-diagonal`, which rebases every relative-WIS figure.

## susc/inf smoothing REMOVED entirely → independent per-bin offsets 2026-07-13 (`joint_model.jl`, `framework.jl`, spec, 10j) (user request)

- **The relative susc/inf age profiles are no longer smoothed at all.** The `A-1` non-reference
  log-offsets are now **independent per age bin**: `susc = vcat(1, exp.(_softclamp.(sig_s .* z_s, log0.05, log20)))`
  (likewise `inf`), `z_s`/`z_i ~ filldist(Normal(0,1), A-1)`. This is the pre-`4607021` form (the GP was
  built on top of exactly this). The end of the GP→RW1→RW2→GP experiment sequence — now **none**.
- **What was removed** from `model_transmission` (`joint_model.jl`): the shared-length-scale RBF GP block —
  `log_rho_si`, `ρ_si`, `Ksi`, `Lsi`, and the `Lsi *` mixing (`sig·(Lsi·z)` → `sig·z`). Config
  (`framework.jl`): dropped `susc_inf_gp_len_prior` (its only consumer `log_rho_si` is gone) and
  **renamed** `susc_inf_gp_sd_prior` → `susc_inf_sd_prior` for the marginal-SD prior of `sig_s`/`sig_i`
  (still SEPARATE per profile).
- **Prior + clamp LOOSENED (same day, user request)** to allow real age variation in susc/inf:
  `susc_inf_sd_prior` **`N⁺(0.1,0.05²)` → `N⁺(0.5,0.25²)`** (marginal SD ≈0.5 ⇒ ±2 SD ⇒ typical
  susc/inf ∈ `[0.37,2.7]`, was `[0.80,1.25]`), and the soft-clamp **`[log0.2,log5]` → `[log0.05,log20]`**
  (≈`[-3,3]`; hard-bounds susc/inf ∈ `[0.05,20]`, was `[0.2,5]`). The clamp now sits at ~±6 SD (prior ±2
  SD well interior) and **re-aligns with the `-sc` cache-token docstring**, which already described the
  clamp as `≈[0.05,20]` (the `-sc` marker still holds — a clamp still exists, just wider).
- **KEPT (method-agnostic):** the soft-clamp itself (NGM safety bound, not smoothing), the reference-bin
  pinning `susc[1]=inf[1]=1`, and `sig_s`/`sig_i`/`z_s`/`z_i`. Because the GP kernel had **unit diagonal**,
  removing it leaves the per-bin marginal SD unchanged (`=sig`) — the calibration holds; the profile is
  just **rougher** (neighbours no longer pulled together). `sigma_inf` (obs-noise SD) is untouched.
- **Downstream unchanged, nothing breaks:** `model_transmission` still returns
  `(; susc, inf, F, gamma_sar, sigma_inf)` and `fit_stage2_pooled` still stores the `N×A` susc/inf draws,
  so `load_transmission_draws` (8j_viz), 10j `make_susc_inf_fig`, and 9j `plot_ratio(tr.susc/tr.inf)` all
  keep working (rougher profiles). No viz ever read the smoothing latents. `plot_lengthscales` (9j) is the
  **contact-degree** GP (ρ_diag/ρ_gap/ρ_time), unrelated — untouched.
- **Cache: `-sm` token DROPPED** in `contacts_label` → `temporal-gsar-cut-sc` / `pooled-gsar-cut-sc`
  (tag must reflect the parameter space; keeping a "smoothing" marker on unsmoothed fits would be a
  footgun). The 256 existing Stage-1 chains (`8j_s1_*_…-sc-sm_*`) are susc/inf-independent, so they were
  **renamed** `-sc-sm`→`-sc` on disk and reused as-is (no refit) — mirrors the `-sm`-ADD precedent in
  reverse (`for f in 8j_s1_*-sc-sm_*.jld2; do mv "$f" "${f/-sc-sm_/-sc_}"; done`). **Zero** Stage-2
  `8j_s2_*` files existed, so nothing to delete; re-run `prefit_stage2!` (8j) to regenerate under `-sc`.

## susc/inf smoothing REVERTED RW2/IWP → shared-length-scale RBF GP 2026-07-12 (`joint_model.jl`, `framework.jl`, spec, 10j)

- **Undo of the RW1/RW2 experiment below (user request).** The relative susc/inf age profiles are again
  smoothed by a **shared-length-scale squared-exponential (RBF) GP** — the commit-`4607021` form. ONE
  length-scale `log_rho_si ~ Normal(log1.5, 0.5²)` (soft-clamp `[log0.5, log6]`) is **shared** by susc
  and inf; the marginal scales `sig_s`, `sig_i ~ N⁺(cfg.susc_inf_gp_sd_prior…)` are **separate**. The
  offset is `susc/inf = vcat(1, exp.(_softclamp.(sig·(Lsi·z), log0.2, log5)))`, `Lsi = chol(K(ρ_si)+1e-4·I)`,
  `K_{mn}=exp(-(m-n)²/2ρ_si²)`. `K` has **unit diagonal** ⇒ per-bin marginal SD = `sig` (the kernel only
  correlates neighbours, doesn't inflate spread).
- **Marginal SD TIGHTENED vs the GP era: `susc_inf_gp_sd_prior = (0.1, 0.05)`** (GP-era was `N⁺(0.2,0.1²)`),
  carrying over the recent "susc/inf age variation is empirically small" intent. `sig≈0.1` ⇒ typical
  susc/inf ∈ [0.80,1.25] at ±2 SD; neighbour corr ≈0.8 at ρ_si≈1.5 ⇒ adjacent-bin ratio ≈1.15× (≪2×).
- **GP is STATIONARY** (marginal SD constant across age) — the key contrast with RW1/RW2, whose variance
  *grew* with distance from the reference bin. So the oldest "70+" bin is **no looser in level** than
  near-reference bins; the RW far-field diffuseness (its 95% level band `[0.37,2.7]`) is gone.
- **What was removed:** the `_iwp_offsets` helper; the `age_mid` 5th arg of `model_transmission` (back to
  `model_transmission(Cstar_weeks, wd, w, cfg)`) and its `age_mid = cis_age_midpoints()` line in
  `fit_stage2_pooled`; the `susc_inf_rw_sd_prior` config field (→ `susc_inf_gp_len_prior` +
  `susc_inf_gp_sd_prior`). Raw Stage-2 latents changed (`tau_*`/`v0_*`/2×(A-1) `z_*` → `log_rho_si` +
  `sig_*` + (A-1) `z_*`), but `model_transmission` still returns `(; susc, inf, F, gamma_sar, sigma_inf)`,
  so `generated_quantities`/viz (10j `make_susc_inf_fig`, `load_transmission_draws`) are untouched.
- **KEEP (method-agnostic):** the **1e-4 jitter** on `Lsi` (not 1e-6 — `K` is near rank-1 at the upper
  ρ_si clamp and the Stage-2 Pathfinder is not try/caught, so a `PosDefException` aborts the whole fit;
  mirrors the `Kt` temporal kernel in `model_degree`); the `_softclamp` Inf-safe nested form.
- **Cache:** tag `…-sc-sm` **unchanged** (mechanism-agnostic marker). The existing `8j_s2_*_-sc-sm_*`
  pooled chains are RW2-smoothed ⇒ **stale**; they were **deleted** so Stage 2 re-fits under the GP.
  `8j_s1_*` (smoothing-independent) are reused as-is. Re-run the 8j Stage-2 prefit.

## γ_SAR prior WIDENED 2026-07-12 (`framework.jl`, `joint_model.jl`, spec)

- **`gamma_sar_prior` went from the data-calibrated `(log0.33, 0.56)` to the weakly-informative
  `(log0.27, 1.05)`** so the prior *widely covers* γ_SAR∈[0.05,1.5] (user request). Log-SD ≈doubles
  (0.56→1.05); 90% γ_SAR band widens ≈[0.13,0.83] → [0.048,1.52] (γ span ~6× → ~30×). Centre log(0.27)
  is the **geometric mean of [0.05,1.5]** (≈ the old calibrated 0.33), so both endpoints sit just inside
  the central 90% (≈±1.6σ). The old 18-chain calibration now only informs the centre; the data dominate.
- **Softclamp `[log0.02, log5]` is UNCHANGED** and still correct: it sits at ≈−2.5σ/+2.8σ (tails
  ≈0.7%/0.3%), i.e. outside the widened 90% band, so it contains the prior without distorting the
  covered range. Don't tighten it to "match" the old prior.
- **Only the config default value changed** — no field rename, no `-gsar`/`-sc`/`-cut` cache-tag bump, no
  Stage-1 (`8j_s1_*`) impact (γ_SAR is a Stage-2 latent). Existing Stage-2 `8j_s2_*` chains predate the
  new prior, so **re-fit** to see its effect. `load_transmission_draws` / softclamp reads are untouched.

## RW2 / IWP smoothing of relative susc/inf 2026-07-12 — ⚠️ SUPERSEDED (reverted to the shared-length-scale RBF GP above; kept for history) (`joint_model.jl`, `framework.jl`, spec)

- **The relative susc/inf age profiles are smoothed by a second-order RANDOM WALK (RW2), i.e. a 2nd-order
  Integrated Wiener Process (IWP) for the irregular age bins — not RW1, not a GP.** (RW2 superseded a
  short-lived RW1 the same day, which had superseded a squared-exponential GP — see git history if the
  RW1 `cumsum(τ√Δ·z)` form or the GP `Lsi`/`ρ_si`/`Ksi` names resurface.) The offset is built by the
  helper `_iwp_offsets(τ, v0, δ, z)` (a non-mutating `cumsum`-based 2×2 state recursion, AD-safe under
  ReverseDiff), then `susc/inf = exp.(_softclamp.(o_•, log0.2, log5))`, so `o[1]=0 ⇒ susc[1]=inf[1]=1`
  (reference bin anchors the walk). **RW2 penalises the profile's CURVATURE (second differences), not
  its slope (RW1) or level (GP)** ⇒ a much smoother age profile.
- **Exact IWP, not a naive double-cumsum.** Over a normalised gap δ the increment covariance is the
  exact continuous-time IWP `τ²[δ³/3 δ²/2; δ²/2 δ]` (Cholesky-factored in the helper: value gets
  `τ·δ^{3/2}/√3·z₁`, slope `τ·√δ·(√3/2·z₁+½·z₂)`), NOT `o=cumsum(cumsum(...))`. This is what "IWP for
  irregular age bins" means; the state carries a slope `v` alongside the value `o`. `z_•` is a
  **2×(A-1)** matrix (two innovations per step), not a length-(A-1) vector.
- **Free initial slope.** `o_1=0` pins only the *level* at the reference bin; the *slope* `v_1` is free
  (`v0_• ~ Normal(0,1)`, scaled `τ_•/√(A-1)`) — the linear null-space of an intrinsic RW2. Don't pin
  `v_1=0` (that would force a flat start and drop the RW2's linear component).
- **Separate variances, distance-scaled steps.** `τ_s`,`τ_i ~ truncated(Normal(cfg.susc_inf_rw_sd_prior…);
  lower=0)` are TWO distinct latents (the "separately estimated variances"), one shared prior form; each
  also scales its profile's `v0`. `Δ = diff(age_mid)` are the adjacent age-bin MIDPOINT gaps,
  **normalised to unit mean** (`Δn = Δ ./ (sum(Δ)/(A-1))`) so `τ` ≈ typical per-step innovation SD.
- **RW2 far-field is MUCH more diffuse than RW1** (doubly integrated): equal-gap approx
  `Var(o_2)=τ²/2` (SD≈0.035, near-reference VERY tight, `susc_2∈[0.96,1.04]`) but `Var(o_A)≈78·τ²`
  (SD≈0.44 at τ=0.05 — the oldest bin's *level* is the least-informed, 95% band ≈`[0.37,2.7]`, though its
  *shape* stays smooth). Intended (curvature-penalised, level-diffuse). The
  `_softclamp(o, log0.2, log5)` binds increasingly at the oldest bins and still hard-bounds
  susc/inf ∈ [0.2,5]. **2×2 recursion ⇒ no dense Cholesky ⇒ no PosDef/jitter concern.**
- **τ prior tightened TWICE (0.2,0.1)→(0.1,0.05)→(0.05,0.025) to bound the ADJACENT-bin ratio ≲2×**
  (`|Δo|≤log2≈0.693`, 2026-07-12). Watch out: RW2's per-step jump SD is NOT uniform — the slope
  *accumulates*, so it grows to `≈3τ` at the oldest (widest-gap) bins (worst step, actual midpoints
  `[6,13,20,29.5,42,59.5,74.5]`). 400k-draw MC of `_iwp_offsets` over the real normalised gaps: the
  fraction of prior draws with SOME adjacent step >2× fell `≈32%→≈7%→≈0.3%` across the two halvings; at
  (0.05,0.025) the worst-step increment SD≈0.17 ⇒ ±2SD adjacent ratio `exp(0.34)≈1.4×`. Sizing lever is
  τ ALONE (v0 and the clamp are secondary). TRADE-OFF of the 2nd halving: the far-field oldest *level* SD
  (`√78·τ`) drops `≈0.88→0.50` (95% band `[0.37,2.7]`), so it no longer spans the full clamp — older-age
  structure is now MORE constrained, but still above the data-preferred log-SD ≈0.22–0.28 (see "surfaced,
  not fixed" below), so not starved; it just accumulates smoothly over 6 steps rather than jumping in one bin.
- **Midpoints must NOT be fetched inside the model body** — `cis_age_midpoints()` reads a CSV. Compute
  `age_mid = cis_age_midpoints()` ONCE in `fit_stage2_pooled` and pass it as the 5th positional arg of
  `model_transmission(Cstar_weeks, wd, w, cfg, age_mid)`. (User declined adding an `age_mid` field to
  the `WindowData` struct, so it's threaded as a model arg instead — canonical midpoints
  `[6,13,20,29.5,42,59.5,74.5]`, 70+ → 74.5, same source the degree GP uses at `joint_model.jl:81`.)
- **τ/v0 are NOT stored.** `model_transmission`'s return `(; susc, inf, F, gamma_sar, sigma_inf)` is
  unchanged; `fit_stage2_pooled` reads susc/inf from `generated_quantities`, so RW2 flows into the
  stored `N×A` arrays automatically — no storage/viz change (10j `make_susc_inf_fig` reads stored draws).
- **Cache: shared tag `-sc-sm` (mechanism-agnostic "smoothing" marker); the 256 `8j_s1_*` chains were
  RENAMED `-sc`→`-sc-sm`** rather than refit (Stage 1 is untouched). `-sm` covered the earlier RW1 and
  now the RW2/IWP with **no further rename** (the smoother lives entirely in Stage 2; switching RW1→RW2
  changes only the Stage-2 posterior). Only `contacts_label` (single source; the 8j notebook literal
  uses it dynamically) + the hardcoded default `contacts="temporal-gsar-cut-sc-sm"` literals in
  `8j_viz_utils.jl`/`10j_viz_utils.jl` carry the tag. 0 `8j_s2_*` files existed, so nothing stale; any
  pre-existing s2 pooled files (RW1- or GP-smoothed) carry differently-smoothed susc/inf and must not be
  reused.

## Two-stage cut inference + γ_SAR revert 2026-07-12 (`joint_model.jl`, `ngm.jl`, `framework.jl`, viz utils, 8j/9j/10j) — inst/4_cut_Bayes.md

- **The single joint `model_joint` was SPLIT into a two-stage CUT inference.** `model_degree(dm, ds,
  pop, cfg)` fits the contact-degree GP alone and returns per-week **raw moments** `(; K1, K2, G)`
  (each a length-`Tn` `Vector{Matrix}`), NOT `Cstar` — so the fit is **NGM-independent** and one
  Stage-1 chain serves both builders (the builder is applied downstream via `contact_star(nb, …)`).
  `model_transmission(Cstar_weeks, wd, w, cfg)` fits the infection/renewal block conditioning on a
  **fixed** `Cstar_weeks`. The cut Monte Carlo: impute `cfg.n_stage1_post=100` Stage-1 draws, re-fit
  Stage 2 per draw keeping `cfg.n_stage2_draws=100`, **pool** 100×100 = 10 000 infection draws → WIS.
- **Stage-1 latents are nb-independent, so μ no longer depends on the NGM builder.** Under the old
  joint model the infection likelihood fed back into the GP posterior, so μ differed slightly per nb.
  Now the two builders of a degree family reconstruct **identical** μ (10j §2b/§3 draw overlapping
  lines — correct, not a bug). Consequence: `stage1_chain_path` drops the ngm token (`8j_s1_<degree>_…`).
- **γ REVERT: the C\* S̄-normalisation was removed and `γ`→`γ_SAR`.** No more `C* → C*/S̄` decoupling;
  C\* feeds the NGM raw and `log_gamma_sar`/`gamma_sar` is again the **per-contact SAR** (reproduces the
  reference cell `N_11 = susc₁·inf₁ = γ_SAR`), comparable across origins. Prior back to
  `gamma_sar_prior=(log0.33,0.56)`; `build_ngm(…; gamma_sar=…)` kwarg (was `γ`). This *reverts* the
  2026-07-11 `-gnorm` entry below — that whole "decouple γ from contact scale" change is undone.
- **`fit_stage2_pooled` returns `(; gamma_sar, susc[N×A], inf[N×A], F, sigma_inf, post_index[N],
  Cstar_end[n_post])`.** `Cstar_end[m]` = Stage-1 draw `m`'s origin-week (`[end]`) C\* — the ONLY C\*
  the forecast NGM needs. The full per-week C\* trajectory is **not** stored; the 10j fit-window
  diagnostic (`fit_window_infection_draws`) rebuilds it on demand from the Stage-1 chain
  (`stage1_moment_draws` → `contact_star`). This works ONLY because `stage1_moment_draws` is
  deterministic (even-grid subsample of the reloaded chain) so `post_index[d]=m` aligns with the same
  Stage-1 draw at pooling time and at reconstruction time — keep both call sites using the same
  `(ds, pop, cfg, n_post)`. (Weibull moments depend on `ds.p0`, so pass the SAME h-window `apd`.)
- **Cache tag bumped `temporal/pooled-gnorm` → `temporal/pooled-gsar-cut`** (`contacts_label`). Two new
  artefact families: `8j_s1_<degree>_<contacts>_<origin>_h<h>.jld2` (key `result`, GP chain, NO ngm
  token) and `8j_s2_<degree>_<ngm>_<contacts>_<origin>_h<h>.jld2` (key `pooled`). Old single-file
  `8j_chn_*-gnorm` joint chains are structurally disjoint (differently-scaled `log_gamma`), left on disk.
- **Sampler (user decision):** PF + PF for this preliminary run, but keep Stage-1 switchable to NUTS —
  `cfg.stage1_use_nuts` (8j's `STAGE1_USE_NUTS`) threads into `fit_stage1`; Stage 2 is always Pathfinder
  (100 cheap fits/cell). The forecast is unchanged in shape (`A×H×N`, N pooling-agnostic to `scoring.jl`).
- **Function renames (update all call sites):** `fit_joint`→`fit_stage1`+`fit_stage2_pooled`;
  `iterated_forecast`/`posterior_forecast`→`two_stage_forecast`; `fit_or_load_chain`→`fit_or_load_stage1`
  +`fit_or_load_stage2`; `prefit_chains(_streaming)!`→`prefit_stage1!`+`prefit_stage2!`(+`prefit_two_stage!`).
  Viz: `chain_path`→`stage1_chain_path`/`stage2_pooled_path`; `load_transmission_draws` now reads susc/inf/
  `gamma_sar` from the Stage-2 pooled file and ρ from the Stage-1 chain; `reproduction_draws` drops its
  `apd` arg (reads the pooled file + eigen-decomposes). `reconstruct_mu_draws`/`reconstruct_dispersion_draws`
  read the Stage-1 chain (logic unchanged; tag/path only).

## C\* normalisation to decouple γ from contact scale + γ_SAR→γ rename 2026-07-11 (`joint_model.jl`, `ngm.jl`, `framework.jl`, `8j_viz_utils.jl`)  — REVERTED 2026-07-12 (see the two-stage cut entry above)

- **Problem:** `γ_SAR` and the C\* scale/eigenvalue were posterior-correlated — since `N = γ·(fs⊙C*)⊙inf'`
  depends only on the **product** `γ·C*`, a uniform `C*→κC*, γ→γ/κ` leaves `N` (hence Rt, every forecast)
  identical; only the CoMix degree likelihood weakly breaks the tie, so raising Rt could be met by raising
  `γ` *or* pulling up the whole contact matrix.
- **Fix:** inside `model_joint`, after `Cstar_weeks` is built and **before** the infection loop, normalise
  by one window-constant `S̄` = fit-window-averaged, pop-weighted mean contact intensity
  (`S̄ = mean(sum(wpop[a]*Cstar_weeks[t][a,b] …) for t in (smax+1):Tn)`, `wpop = wd.pop/sum(wd.pop)`;
  floor `max(S̄,1e-8)`), then `Cstar_weeks = [C./S̄ for C in Cstar_weeks]`. `S̄` is homogeneous degree-1 in
  C\*, so the renewal likelihood becomes **scale-invariant in C\*** → the contact LEVEL moves into `γ`
  (data-identified by infections) and C\* feeds the NGM only its temporal change.
- **Why it's results-neutral:** the transform is level-preserving in `N`, so Rt / forecasts / WIS are
  materially unchanged — only the level *decomposition* is cleaned up. All consumers (`posterior_forecast`,
  `iterated_forecast`, `reproduction_draws`) flow through `build_ngm → N` and stay consistent with **no
  logic change** (rename only). Normalise at the `Cstar` stage, **NOT** in the μ level `c`/`c_vec` —
  `reconstruct_mu_draws` / the 10j heatmap read raw μ against empirical contacts and must stay physical.
- **Rename `γ_SAR → γ`:** Unicode `γ` for the transformed scalar / local / return field / `build_ngm`
  kwarg (avoids colliding with `SpecialFunctions.gamma` in `_weibull_moments`), ASCII `log_gamma` for the
  Turing param (chain column, mirrors `log_kappa`), ASCII `gamma_prior` for the config field. `γ` is no
  longer a per-contact SAR — it is the absolute NGM level `γ = susc₁·inf₁·S̄ ≈ Rt/ρ(C̃*)`, **window-relative**
  (not comparable across origins).
- **Prior — measure it, don't guess.** Reconstructing `new-γ = susc₁·inf₁·S̄` per draw from the pre-norm
  `temporal` chains (6 origins × 4 combos) gave `S̄` medians 4.8 / 172.6 / 1.6 / 3.7 (negbin·mean /
  negbin·neigh / hweibull·mean / hweibull·neigh) — **S̄ spans ~100×** because the neighbourhood size-biased
  ⟨k²⟩/⟨k⟩ explodes — but **new-γ is O(1) for ALL four** (medians 0.69–1.20), because `susc₁·inf₁` (fit
  jointly with C\*) shrinks to compensate the builder scale (0.159 negbin·mean → **0.0071** negbin·neigh).
  So a **single** prior `gamma_prior=(log(0.8),0.5)` serves all combos; the existing softclamp
  `[log0.02,log5]` already contains the range (max q95 = 2.08) — **no clamp widening needed** (the naïve
  `0.33·S̄` would have implied per-combo priors spanning 0.5→57 — wrong, because it ignores the
  builder-compensating `susc₁·inf₁`).
- **Cache-bust (mandatory):** `contacts_label` bumped `temporal-gsar`→`temporal-gnorm` /
  `pooled-gsar`→`pooled-gnorm`. The filename does not encode the transmission block, so a differently-scaled
  `γ` (`log_gamma`) would silently reload stale `-gsar`/`temporal` chains without the bump. Keep the viz
  mirrors (`load_transmission_draws` reads `chn[:log_gamma]`; default `contacts`; all `build_ngm(…; γ=…)`
  call sites) in sync. Old chains kept, not deleted; a re-fit populates the `-gnorm` caches.

## Absolute γ_SAR + reference-normalised susc/inf 2026-07-11 (`joint_model.jl`, `ngm.jl` — transmission reparam)

- **What changed:** the transmission block dropped the confounded level pair `μ_s ~ Beta(24,24)` /
  `μ_i ~ Beta(4,12)` (only their sum was identified, and it fought C*'s scale) for a single **absolute
  transmissibility** `log_gamma_sar ~ Normal(log0.33, 0.56)` → `gamma_sar = exp(softclamp(·,log0.02,log5))`.
  `susc`/`inf` are now **relative**, normalised so reference **bin 1 ("2-10") = 1**:
  `susc = vcat(one(sig_s), exp.(sig_s .* z_s))` with `z_s ~ filldist(Normal(0,1), A-1)` (shrunk from `A`;
  no redundant reference offset). NGM: `N_ab = gamma_sar · susc_a·(1+(F-1)A_a) · C*_ab · inf_b` via
  `build_ngm(...; gamma_sar=1.0)` (kwarg, default 1 keeps the convenience/test method valid). This is the
  analysis-plan `γ_SAR` + baseline-"2-10" form (docx wins); §3.2/§6 updated.
- **NGM ordering: follow the paper/code, NOT the docx's `diag(γ_inf)C diag(γ_sus)`.** Munday 2023 Eq 3 is
  `N = diag(s) C diag(i)` — susceptibility on the susceptible **row** `a`, infectivity on the infectious
  **column** `b`. The docx transposes this (a typo vs its own source); keep the existing code convention.
- **γ_SAR prior is CALIBRATED, not prior-implied.** The naive "current level" `exp(μ_s+μ_i)` from prior
  MEANS (≈2.1) is doubly wrong: (a) the data pull `μ_s≈0.2, μ_i≈0.05` (well below the Beta means 0.5/0.25),
  and (b) under bin-1 normalisation γ_SAR must reproduce the **reference cell** `N_11 = susc[1]·inf[1]`, NOT
  the geometric-mean level `exp(μ_s+μ_i)≈1.31`. Reading 18 pre-reparam `dt_intermediate_age_pair_temporal_GP`
  chains (both degree models × 9 origins), `susc[1]·inf[1]` had **median 0.33, log-SD 0.56** (weighted ~0.43,
  unweighted ~0.20) ⇒ prior `Normal(log0.33, 0.56)`, 90% γ_SAR∈[0.13,0.83].
- **Param space changed ⇒ cache-bust.** `contacts_label` bumped `…-hn`→`…-hn-gsar` (framework.jl) + the 4
  hardcoded `contacts=` viz defaults (8j/10j). Pre-`-gsar` chains carry `mu_s`/`mu_i` and lack
  `log_gamma_sar`, and `z_s`/`z_i` are the wrong length — so `generated_quantities` / `load_transmission_draws`
  reconstruct wrongly. The filename does NOT encode the transmission block, so a transmission change would
  silently reload stale chains WITHOUT the tag bump — always bump.
- **`load_transmission_draws` must mirror the model:** `gamma_sar=exp.(softclamp.(chn[:log_gamma_sar],…))`,
  `susc = hcat(ones(nd), exp.(sig_s .* z_s))` (z_s is now `nd×(A-1)`), col 1 pinned to 1; returns
  `gamma_sar`. All three `build_ngm` NGM call sites (`joint_model` infection loop + `posterior_forecast` +
  `iterated_forecast`, plus `reproduction_draws`) pass `gamma_sar`.
- **Surfaced, not fixed:** fitted `σ_s≈0.22–0.28`, `σ_i≈0.14–0.24` ≫ the tight `N⁺(0.1,0.02)` prior, with
  3–5× age profiles carried by extreme `z` (`z_s[1]≈−4.5`). Kept the tight offset prior for a minimal
  reparam; widening `σ_s/σ_i` (spread carried by `σ`, `z∼N(0,1)`) is the recommended next step.

## Half-Normal per-week dispersion RE scale 2026-07-11 (`src/joint_model.jl` — τ prior form + per-week τ_t)

- **The per-age-pair RE scale τ changed twice, same day:** (1) log-Normal `log_tau ~ Normal(log0.25,0.5)`
  → tightened to `Normal(log0.10,0.30)`; then (2) replaced by a **half-Normal** and made **per-week**:
  `tau ~ filldist(truncated(Normal(0.0, cfg.disp_re_scale); lower=0), Tn)` in the per-week branch
  (one `tau[t]` per window week, iid; passed as `tau[t]` into `_cell_moments!`). The **pooled** branch
  keeps a single scalar `tau`. The `exp(softclamp(log_tau,-4,1))` transform is **gone** — a half-Normal
  is already ≥0, so `τ = tau` directly.
- **Scale σ = `cfg.disp_re_scale` = 0.109 is E[τ²]-matched, NOT Var(τ)-matched.** For a random-effect
  scale the meaningful "equivalent variance" is the marginal RE variance `Var(τz)=E[τ²]=σ²`. Matching
  E[τ²] of the prior log-Normal `exp(Normal(log0.10,0.30))` (E[τ²]=exp(2log0.10+2·0.30²)=0.01197) gives
  σ=√0.01197≈**0.109**, which also keeps the typical τ magnitude ~0.10. Matching **Var(τ)** instead gives
  σ≈**0.053** — rejected: half-Normal has mode 0, so Var-matching nearly extinguishes the RE. (Formula in
  the framework.jl comment.)
- **Empirical basis (why shrink hard):** the pre-hierarchical block-only temporal-GP fits in
  `dt_intermediate_age_pair_temporal_GP/` (tag `temporal`, **no** `z_*`/`log_tau` latents — 48 = 4 blocks
  × 12 weeks only) show κ **homogeneous** across blocks (0.88–1.03, log-SD≈0.06, ~2× week swing) and φ
  typical ~0.28 but volatile (0.05–18, **per-week swings up to 71×**). So per-pair REs should shrink onto
  the block mean. (Reconstruct via `reconstruct_dispersion_draws(...; contacts="temporal", save_dir=…)`.)
- **Param name `log_tau`→`tau` ⇒ cache-bust.** `contacts_label` bumped `temporal-hdisp`→`temporal-hdisp-hn`
  / `pooled-hdisp`→`pooled-hdisp-hn` (framework.jl), plus the 4 hardcoded `contacts="temporal-hdisp"`
  defaults in 8j/10j viz utils and `reconstruct_mu_draws`/`reconstruct_dispersion_draws`. A prior-VALUE
  change alone wouldn't bust the cache (same param space) — but a prior-FORM/name change does.
- **`reconstruct_dispersion_draws` reads τ AFTER `wk` is known:** `τ = wk===nothing ? vec(Array(chn[:tau]))
  : vec(Array(chn[Symbol("tau[$wk]")]))` (pooled scalar vs per-week `tau[wk]`). `D=size(chn,1)*size(chn,3)`
  (was `length(τ)`, no longer available before the read). `ETp` promote uses `eltype(tau)` not `typeof(τ)`.
- **Verified** (`scratchpad/verify_tau.jl`): both κ/φ models build, sample `tau[1..12]`, full logjoint finite.
  Full refit under the new `-hn` tag is still required to regenerate chains (old `-hdisp` chains are stale).

## Hierarchical dispersion 2026-07-11 (`src/joint_model.jl` — block mean + age-pair random effect on φ/κ)

- **The block dispersion (NegBin `log_k` / Weibull `log_kappa`) is now HIERARCHICAL: a block MEAN +
  a per-age-pair random effect.** It is a strictly *additive* extension of the old per-block-per-week
  `4×Tn` array: that array is **kept** (same name `log_k`/`log_kappa`, same `Normal(0,1|0.5)` prior)
  but reinterpreted as the block MEAN β; on top of it each ordered age pair `(i,j)` deviates by
  `τ·z[pcode]`, `pcode=(i-1)A+j ∈ 1..A²`. So `log_disp_{ij,t} = β[bl,t] + τ·z_disp[pcode,t]`,
  `bl=2(block_of i −1)+block_of j`. New latents: `z_kappa`/`z_k ~ filldist(Normal(0,1), A*A, Tn)`
  (per-week, **2-D** = 49×Tn, reconstructable) and a **single scalar** `log_tau ~
  Normal(cfg.disp_re_scale_prior=(log0.25,0.5))`, `τ = exp(softclamp(log_tau,-4,1))`, shared across
  **all** blocks and weeks (user's choice: per-week + single shared scale). Dispersion is still
  per-week (not temporally smoothed) — unlike the mean field.
- **User decisions that shaped it (don't silently "improve"):** (i) **per-week** not time-invariant —
  β and z_disp are re-drawn each week; (ii) **single shared τ** not per-block — one scalar, more
  identifiable than a per-block scale (child→child has only 4 ordered pairs); (iii) **49 ordered
  pairs** (directional), forced by the 4 **directional** blocks (child→adult ≠ adult→child), not 28
  unordered; self-pairs `(i,i)` included.
- **`_cell_moments!` signature changed** `(…, dispv)` → `(…, βv, zv, τ)`: it now computes
  `logd = βv[bl] + τ·zv[pcode]` per cell (bl and pcode both integer, data-independent ⇒ ReverseDiff-safe),
  then the same `exp(softclamp(logd, …))`. `didx` still indexes the degree DATA arrays. `ll` accumulator
  is `zero(eltype(K1))` (=ETp) — ETp now promotes `τ` too (`promote_type(typeof(c), eltype(Fld), typeof(τ))`
  per-week; `promote_type(eltype(μ), typeof(τ))` pooled).
- **Reconstruction contract changed `ndraws×4` → `ndraws×A×A`** (`reconstruct_dispersion_draws`,
  `10j_viz_utils.jl`). It now needs `cfg` (for `block_of`) and `grid` (for `A`) as kwargs, reads the
  block means (`log_*[bl,t]` per-week / column-major `log_*[bi,bj]` pooled), the scalar `log_tau`, and
  the per-pair `z_*[p,t]`/`z_*[p]` (space-tolerant regex, read columns by EXACT stored name — MCMCChains
  prints `z_k[1, 2]` with a space), then rebuilds per cell `exp(softclamp(β[bl]+τ·z[pcode], lo, hi))`.
  The one consumer (10j §4 CCDF grid, `agepair_ccdf_panel`) changed `view(κdraws,:,bl)` → `view(κdraws,:,i,j)`
  and its call passes `grid=grid, cfg=cfg`. `reconstruct_mu_draws` is UNAFFECTED (μ ⟂ dispersion).
- **Cache-bust via `contacts_label`** (`framework.jl`): `"temporal"→"temporal-hdisp"`,
  `"pooled"→"pooled-hdisp"` — the param space gained `z_*`/`log_tau`, so old-named chains lack them and
  mis-reconstruct. Bumped the three hardcoded `contacts="temporal"` viz defaults to `"temporal-hdisp"`
  (`chain_path`, `load_transmission_draws` in 8j_viz_utils.jl; `reconstruct_mu_draws`,
  `reconstruct_dispersion_draws` in 10j_viz_utils.jl). Pre-hierarchical chains archived to
  `dt_intermediate_bf_hdisp/`; refit under the new label.
- **Pooled regime (inactive) preserved as-is:** it keeps the `2×2` block-mean matrix, and the
  model's `vec(2×2)`→`bl` is **column-major** (off-diagonal blocks labelled by that order; harmless as
  the block-mean prior is exchangeable). The reconstruction pooled branch mirrors this exactly
  (`β[:, r+2(c-1)]`) — do NOT reuse the old row-major `2(bi-1)+bj` there. The **active per-week path is
  unambiguous** (`4×Tn` rows indexed directly by `bl`).
- **Spec updated**: `inst/3_preliminary_model_struct.md` §4.1/§4.2 (per-cell `φ_{ij}`/`κ_{ij}`), §4.3
  (rewritten: hierarchical block mean + shared-scale age-pair RE), §6 sampling block (new `β_t`,
  `log_tau`, `z^disp_t`), §10 (`disp_re_scale_prior`, `temporal-hdisp` label), §11 (seam partially
  relaxed).

## Separable spatio-temporal GP 2026-07-10 (`src/joint_model.jl` — per-week contact mean gains a temporal axis)

- **The per-week (`constant_contacts=false`) contact-mean GP is now separable spatio-*temporal*,
  replacing the per-week-iid regime.** The 28-age-pair spatial RBF (ρ_diag/ρ_gap, `Lp`) is unchanged;
  what changed is that the weekly fields are no longer iid. Added a temporal RBF over week indices
  `1:Tn`: `Kt[s,t]=exp(-(s-t)²/(2ρ_time²))`, `Lt=chol(Sym(Kt)+1e-4·I).L`, with a **shared** `ρ_time`
  (`log_rho_time ~ Normal(gp_time_len_prior=(log4,0.5))`, soft-clamp `[log0.5,log26]` weeks). The
  structure field is matrix-normal, **precomputed once** before the week loop:
  `Fld = η .* (Lp * z * Lt')` (P×Tn), so `Cov(vec R)=η²(Kt⊗Kage)` — each age-pair a temporally-correlated
  GP, each week the spatial RBF. `ρ_diag=ρ_gap` recovers the isotropic spatial kernel; `ρ_time→0` iid,
  `→∞` pooled.
- **The temporal coupling means you CANNOT slice `z[:,t]` per week anymore** — week `t`'s column of
  `Lp*z*Lt'` mixes ALL columns of `z`. Precompute `Fld` once, then `μ = _mu_matrix(c_vec[t] .+ @view Fld[:,t])`.
  Don't re-apply `η` in the loop (`Fld` already carries it). Fix the eltype: `ETp = promote_type(typeof(c),
  eltype(Fld))` — `c` is now a **scalar** (`typeof`, not `eltype`).
- **Decoupled temporal LEVEL (user's choice), not just a scalar c.** The overall weekly level is
  `c_t = c + σ_c·(Lt·z_c)`: a stored scalar intercept `c ~ Normal(c0,3)` (kept stored so the viz mirror
  reads it directly — no data-derived `c0` recompute) plus a 1-D temporal GP with its OWN amplitude
  `σ_c` (`log_sigma_c ~ Normal(gp_level_scale_prior=(0,0.5))`, soft-clamp `[-3,2]`, mirrors η), sharing
  `Lt`. This frees `η` to govern age-structure amplitude only. Dispersion (`log_kappa`/`log_k`, `4×Tn`)
  stays per-week iid — temporal smoothing is on the **mean field only**.
- **`Kt` jitter is 1e-4, NOT 1e-6.** At the upper clamp (ρ_time≈26 over a 12-week window) `Kt` is near
  rank-1; the Pathfinder call in `fit_joint` is **not** try/caught (unlike NUTS), so a `PosDefException`
  aborts the whole fit. 1e-4 keeps the near-pooled limit reachable without failing the Cholesky.
- **Parameter space changed ⇒ label bumped `"weekly"→"temporal"`** (`contacts_label`, framework.jl). New
  latents (`log_rho_time`, `log_sigma_c`, `z_c`, scalar `c` instead of `c[t]`) mean old per-week-iid
  `"weekly"` chains would mis-reconstruct. The two hardcoded `contacts="weekly"` defaults in
  `8j_viz_utils.jl` (`chain_path`, `load_transmission_draws`) and `10j_viz_utils.jl` (`reconstruct_mu_draws`,
  `reconstruct_dispersion_draws`) were bumped to `"temporal"` too. Pre-temporal chains live in
  `dt_intermediate_bf_temporal_GP/`; the orphaned `9j_*_weekly.jld2` caches are simply not read under
  the new label.
- **Viz mirror `reconstruct_mu_draws` (10j) rewritten** — regime detection is now `log_rho_time`-first:
  (1) present ⇒ temporal (scalar `c`, `log_sigma_c`, `z_c`, 2-D `z[p,t]`; infer `Tn=max t`; per draw
  build `Lt(ρ_time)`; `R[:,wk]=η·(Lp·(z·Lt[wk,:]))`, `c_wk = c + σ_c·(Lt[wk,:]·z_c)` — needs the **full**
  `z` matrix + `z_c`, not week wk's column, since `Lt[wk,:]` mixes weeks `1..wk`); (2) `c[\d+]` ⇒ legacy
  per-week iid; (3) pooled. `reconstruct_dispersion_draws` needed no logic change (dispersion still `4×Tn`).
- **`load_transmission_draws` (8j) returns `rho_time` too** (`NaN` for pooled); the 9j length-scale panel
  (`collect_transmission_structure` rho-store `nO×2→nO×3`, `plot_lengthscales` 3 series solid/dash/dot)
  now plots ρ_diag/ρ_gap (age-yrs) + ρ_time (weeks) on one axis.
- **Spec updated**: `inst/3_preliminary_model_struct.md` §5 (separable spatio-temporal kernel + matrix-normal
  field + decoupled level), §6 sampling block, §10 config/outputs, §11 (temporal structure now *implemented*;
  remaining seams = per-week dispersion, longer-memory/non-separable kernel).

## Anisotropic diagonal-coordinate GP 2026-07-09 (`src/joint_model.jl` — age-pair contact-mean kernel)

- **The age-pair GP kernel is now anisotropic in DIAGONAL coordinates.** The old isotropic RBF had
  one shared length-scale ρ acting equally on both age coordinates. It's now rotated 45° into
  `u=(mid_a+mid_b)/√2` (along the main diagonal = **total age**) and `v=(mid_a−mid_b)/√2` (across it
  = **age gap** / assortativity), each with its **own** length-scale: `ρ_diag` on `u`, `ρ_gap` on
  `v`. Kernel: `Kp[m,n]=exp(-((su[m]-su[n])²/(2ρ_diag²) + (df[m]-df[n])²/(2ρ_gap²)))`. The **√2
  normalisation is load-bearing**: the rotation is orthonormal so `(Δu)²+(Δv)²=(Δx)²+(Δy)²`, hence
  `ρ_diag=ρ_gap` reduces **exactly** to the old isotropic kernel (verified: `max|Kiso−Kani|~0`) — do
  not drop the `/√2` (it keeps the `[3,45]`-yr soft-clamp bounds and `gp_len_prior` meaningful in
  the same age-year units).
- **Chain param rename `log_rho` → `log_rho_diag` + `log_rho_gap`** (both `~ Normal(cfg.gp_len_prior…)`
  — a **shared** prior, no new config field; both soft-clamped to `[log 3, log 45]`). This changes the
  parameter space ⇒ every cached `8j_chn_*.jld2` is stale and was deleted; refit.
- **Viz mirrors MUST rotate coordinates identically** (the reconstruct-matches-`generated_quantities`
  invariant): `10j_viz_utils.jl` (`reconstruct_mu_draws`) rebuilds `su`/`df` and the two-length-scale
  kernel from `chn[:log_rho_diag]`/`[:log_rho_gap]`; `8j_viz_utils.jl` (`load_transmission_draws`) now
  returns **both** `rho_diag` and `rho_gap` (was a single `rho`) — its only in-code consumer,
  `9j_forecast_diagnostics.ipynb`'s length-scale panel, plots two series per config (ρ_diag solid,
  ρ_gap dashed). Anything reading the old `.rho` field will break.
- **Spec updated**: `inst/3_preliminary_model_struct.md` §5 (kernel eq + rotation), §6 sampling
  statement, and the §-`8j` config/output bullets now describe the two diagonal length-scales.

## Model improvements 2026-07-09 (`src/joint_model.jl` — relative pop · collapsed Weibull · clamp-free/ReverseDiff)

- **Relative-population offset fixes the μ-saturation degeneracy (the fix to the 10j lesson
  below).** The reciprocity offset is now `logpop = log.(wd.pop ./ wd.pop[1])` (reference bin
  index 1, "2-10"), not `log.(wd.pop)`. It's an *exact* reparametrisation (μ invariant; `c0`
  auto-shifts +log pop₁ since `c0 = mean(log_emp .- logpop')` reuses `logpop`), but it rescales the
  latent level `c`/`c0` from ≈−15.6 to **O(1)** — which removes the pressure that let Pathfinder
  drift `c` up into the μ clamp. Confirmed under ReverseDiff, origin 2020-11-08, NegBin|mean:
  `c[origin] ≈ −0.82`, `μ=Cstar ≈ 0.05–2.7` (interior), **not** 403. Keep it relative — do not drop
  the per-`j` `logpop[j]` term (reciprocity needs it, up to an additive constant).
- **`clamp` replaced by a smooth soft-clamp, not deleted (ReverseDiff + numerical stability).**
  The `clamp` *function* is gone from `model_joint` (only the integer thread-cap `clamp` in
  `prefit_chains!` remains), but the bound *ranges* stay — via an interior-preserving softclamp
  (with a sign-branched stable `_softplus`). **NOTE (2026-07-11): the formula was reformulated to
  the Inf-safe nested form** `_softclamp(x,lo,hi) = lo + softplus((hi − softplus(hi − x)) − lo)`;
  the original `x − softplus(x−hi) + softplus(lo−x)` returns **NaN** at `x=±Inf` (`Inf − Inf`), so
  when `tau`/`τ·zv`/the GP field overflowed to Inf on a stray step, `κ=exp(softclamp(Inf))` became
  NaN and `Weibull(κ,λ)` threw `DomainError α>0` — killing the Pathfinder run (seen: origin
  2020-10-18, weighted-hweibull, **neighbourhood** NGM, whose `gamma(1+2/κ)` second moment drives
  the optimiser into those extremes). The nested form saturates ±Inf to ≈hi/≈lo; interiors agree to
  <3e-3. It is the **identity in the interior** (so the relative-pop-rescaled O(1) latents
  are undistorted) and only saturates stray LBFGS excursions; applied to `ρ,η` (`log 3..log 45`,
  `-3..2`), `μ` (`-8..6`), Weibull `κ` (`-3..3`), NegBin `kk` (`-4..5`) — the same ranges the old
  `clamp`s used. AD is switched to **ReverseDiff**:
  `fit_joint`/`fit_or_load_chain`/`iterated_forecast`/`prefit_chains!` take an
  `adtype = AutoReverseDiff()` kwarg threaded into **both** Pathfinder and NUTS (Pathfinder
  previously ignored `adtype`); `main_utils.jl` does `using ReverseDiff` so the DynamicPPL AD
  extension loads, and `ReverseDiff` is now a project dep. **Viz mirrors must track the model** —
  `10j_viz_utils.jl` / `8j_viz_utils.jl` apply the same `_softclamp` + relative-pop (else
  reconstruction diverges from `generated_quantities`).
- **Why the soft bounds are load-bearing for Weibull (not just cosmetic).** A plain-`exp` (no
  bound) HurdleWeibull fit **aborts**: an unbounded LBFGS step sends `μ = exp(rvec+logpop)` to
  `0.0`/`Inf`/`NaN`, so the Weibull scale `λ = μ/gamma(1+1/κ)` goes non-finite and `Weibull(κ,λ)`
  **throws** `DomainError θ>0` (a thrown error kills the whole Pathfinder run — it is not a
  gracefully-rejected step). NegBin is immune (it tolerates `μ→0`), which is why the *unweighted*
  model fits clamp-free but the *weighted* one does not. Point-guards (κ lower floor, `λ+floatmin`)
  do **not** suffice because the `NaN` originates upstream in the shared `μ`; bounding `μ` (and `κ`)
  via `_softclamp` is what keeps `λ` finite-positive. Don't “simplify” the soft-clamps back to plain
  `exp`.
- **Collapsed Weibull likelihood (`WeightedDegreeHist`).** `pos_weight` is now a
  `WeightedDegreeHist{x::Float64, y::Int}` (value→count) instead of a raw `Vector{Float64}`; the
  weighted degrees live on a finite duration-weight lattice, so collapsing is lossless. The hurdle
  likelihood uses `calculate_loglikelihood(pos, Weibull(κ,λ)) = Σ y·logpdf` (new continuous method
  in `turing_utils.jl`), mirroring the NegBin `DegreeDist` path. Consumers use `whist_mean`/`isempty`
  (not `mean`/`length`): `build_degree_stats` (μW init), `pool_over_time` (merge via `merge_whist`,
  not `vcat`), and the 10j hweibull observed-mean cell. Verified `Σlogpdf(raw)==calculate_loglik`.
- **Model changed ⇒ all cached `8j_chn_*.jld2` are stale** (numeric meaning of `c` changed; Weibull
  likelihood changed). `fit_or_load_chain`/`prefit_chains!` skip existing files, so they must be
  **deleted** and re-fit — a silent reuse would mis-reconstruct.

## Diagnostics (`src/10j_*`)

- **Reconstructing the smoothed contact mean μ from a cached chain — validate against the model,
  and beware degenerate/saturated fits.** `10j_viz_utils.jl:reconstruct_mu_draws` rebuilds
  `μ_{i→j} = exp(rvec[pair_index] + logpop_j)` (per-week `c[t]`, `z[·,t]`, `ρ`, `η`; clamp-free and
  relative-pop after the 2026-07-09 change above) exactly as `model_joint` does. It was verified to
  match `generated_quantities(model, chn)`'s `q.Cstar[t]` element-for-element (for `MeanNGM`,
  `Cstar == ⟨k⟩ == μ` on the NegBin path). Two traps: (1) MCMCChains names the 2-D field with a
  **space** — `z[1, 12]` not `z[1,12]` — so the regex must allow `\s*` around the comma; `c`/`z`
  are **per-week** (`c[t]`, `z[p,t]`), not pooled. (2) *[RESOLVED by the relative-pop offset above.]*
  Historically `logpop = log.(wd.pop)` (raw England pop ~1e7 ⇒ `log ≈ 15.6`) forced a healthy fit
  to need `c ≈ -15`; the old Pathfinder chains drifted to `c ≈ -5.45`, pinning `c+logpop ≈ 10 →
  clamp 6 → μ ≡ 403` for every cell (forecast fans still looked sane because a constant `C*` is
  absorbed into `susc·inf·F`; the age-pair-mean panel exposed it). The relative offset + clamp
  removal make the healthy `c` O(1), so this no longer occurs.

## Preliminary forecasting framework (inst/1, `src/8j_*` + `src/framework.jl` …)

- **Per-week contact degree + time-varying renewal NGM (`constant_contacts=false`).**
  The pooled one-`C*`-per-window model was generalised so the age-pair mean is estimated
  **per window week** and the renewal NGM `N(t)` varies through contacts as well as antibody.
  Design that kept the two swap-axes intact: `model_joint` **branches** on
  `cfg.constant_contacts` and produces `Cstar_weeks::Vector{Matrix}` of length `Tn` either way
  (pooled ⇒ `fill(C*, Tn)`; per-week ⇒ one `C*_t` per week), then a **shared** renewal loop
  uses `Cstar_weeks[t]`. Per-week latents: **per-week level `c_t` and field `z_t`**
  (`c ~ N(c0,3)^Tn`, `z ~ N(0,1)^{28×Tn}`) with `ρ,η` and the 28×28 Cholesky **shared** across
  weeks (one factorisation, reused) — independent weekly GP draws, **no temporal smoothing**;
  dispersion `φ/κ` per week × block, stored **`4×Tn`** (block-linear rows × week). Gotchas that
  bit / were avoided:
  (i) **Empty per-week Weibull cells** have `p⁰=1 ⇒ ⟨k⟩=⟨k²⟩=0 ⇒ 0/0` in the neighbourhood
  `k2/k1`; guard `base_contact(::NeighbourhoodDegreeNGM,…) = k1>0 ? (k2/k1)*g : zero(k1)`
  (NegBin keeps `k1=μ>0`, never hits it). Per-week has *many* fully-empty cells, so this is
  load-bearing, not theoretical. (ii) **c0 / `log_emp`** stays the *pooled* grand mean in both
  regimes so the prior centre is identical (build_degree_stats always pools for `log_emp`, then
  returns pooled `A×A` **or** raw `[t,i,j]` weekly arrays). (iii) **Keep per-week latents ≤ 2-D;
  3-D `filldist` breaks `generated_quantities`.** A `2×2×Tn` dispersion (`ProductDistribution{3}`)
  samples/Pathfinders fine but `generated_quantities` can't reconstruct it from the chain
  (`hasvalue(vals,vn,dist)` unimplemented → *"No value was provided for the variable log_k"*), so
  dispersion is `4×Tn` (block-linear `bl=2(bi−1)+bj`), `z` is `P×Tn`, `c` is `Tn` — all ≤2-D. The
  changed parameter space also means stale pooled `.jld2` mis-reconstruct, so `contacts_label(cfg)`
  (`"weekly"`/`"pooled"`) is in the chain filename to keep the caches disjoint.
  (iv) **Forecast picks the origin-week slice** `q.Cstar[end]` (last week of that fit's degree
  window = contacts at `t₀+h−1` in the contact-updated iterate). (v) **Alignment**: base
  `fit_joint` shares ds & wd windows (exact); the forecast pre-fit pairs an h-shifted degree
  window with baseline `wd0` **positionally** (ds week k ↔ infection week k) — the same
  approximation the pooled scheme already made, and the meaningful contact for the step is the
  `[end]` slice. Reuse the `K1/K2/G` buffers across weeks — `contact_star` materialises a fresh
  `C*_t` each iteration so there's no aliasing.

- **Reconstructing per-cell params from a cached chain (10j viz): use the EXACT stored
  parameter name — MCMCChains prints matrix indices with a space after the comma.** A `2-D`
  Turing param `log_k ~ filldist(…, 4, Tn)` is stored in the chain as
  `"log_k[1, 2]"` (space after the comma), **not** `"log_k[1,2]"`. Rebuilding the symbol as
  `Symbol("log_k[$bl,$wk]")` throws `ArgumentError: index log_k[1,12] not found` even though the
  column exists. `reconstruct_mu_draws` sidesteps this for `z` by iterating `names(chn,:parameters)`
  and indexing `chn[Symbol(n)]` with the *actual* name string; `reconstruct_dispersion_draws`
  (added for the 10j §4 age-pair degree-CCDF grid) does the same — regex-scan the param names
  (`^log_k\[(\d+)\s*,\s*(\d+)\]$`, `\s*` tolerates the space), then read each matched column by its
  own name. The scalar/1-D reads (`c[$wk]`, `log_eta`) have no comma so `Symbol("c[$wk]")` is fine.
- **Plotting the estimated NegBin degree CCDF per draw: build it from `pdf` on a bounded integer
  grid, NOT `ccdf(::PoissonMixture,·)`.** The custom `Distributions.ccdf(d::PoissonMixture,k)` in
  `turing_utils.jl` is `@memoize`d and recurses up to `k_max=20_000` (summing `pdf` from `k` to
  `k_max`); across ~200 draws × 49 cells that is ~10⁸ `pdf` evals and hangs. Instead evaluate
  `pdf.(NegBin(μ,k), 0:kmax)` on a bounded grid (`kmax` = max observed degree in the cell),
  reverse-cumsum for the tail, and — to match the OBSERVED `plot_ccdf!(dd)` convention, which strips
  the zero bin and normalises over positives — divide by `(1−P₀)` so the CCDF is **conditional on
  ≥1** (starts at 1 at the smallest degree). The Weibull hurdle path needs no truncation: its
  positive-part `Weibull(κ, μ/Γ(1+1/κ))` CCDF already matches the positive-only observed CCDF.

## Reverted hierarchical dispersion → block-only 2026-07-11 (`joint_model.jl` + viz + spec — user request)
- **SUPERSEDES the "Hierarchical dispersion" and "Half-Normal per-week dispersion RE scale" entries
  above (both 2026-07-11).** Per user request the per-cell dispersion is back to **block-only**:
  `log_disp_{ij} = dispv[bl]` (just the `4×Tn` / `2×2` block-linear mean `log_kappa`/`log_k`), with
  NO per-age-pair random effect and NO shared scale. Removed `tau`, `z_kappa`/`z_k`, and the
  `disp_re_scale` config field; `_cell_moments!` went back to `(K1,K2,G,μ,didx,dispv)`.
- **The absolute-γ_SAR reparam was KEPT** (it shipped in the same commit `a9a2953` but is
  independent): `log_gamma_sar` + relative susc/inf (bin-1 = 1, `z_s`/`z_i` length `A-1`),
  `build_ngm(…; gamma_sar=…)`, `gamma_sar_prior`. Only the dispersion hierarchy was undone.
- **Cache tag bumped** `temporal/pooled-hdisp-hn-gsar` → `temporal/pooled-gsar` (`contacts_label`).
  This alone isolates the stale hierarchical `.jld2` (they carry `tau`/`z_kappa`/`z_k`; the reverted
  model would silently ignore those extra columns and reconstruct a DIFFERENT dispersion) — the new
  tag means they are simply never reloaded. **A re-fit is required** to populate the `-gsar` caches;
  old `-hdisp-hn-gsar` chains were left on disk (not deleted).
- Companion reverts: `reconstruct_dispersion_draws` (10j_viz_utils.jl) back to `ndraws×4`
  block-linear (dropped `cfg`/`grid` args); the 10j §4 CCDF-grid caller back to `bl`-indexed
  `view(κdraws,:,bl)` + `agepair_ccdf_panel(…, cfg; …)`; `_softclamp` fix-comment and the CLAUDE.md
  gotcha updated to cite `log_kappa`/`log_k` (not the now-gone `tau`) as the overflow example.
- **Lesson**: `a9a2953` bundled two orthogonal changes (dispersion hierarchy + γ_SAR) in one commit,
  which made this a *surgical partial* revert rather than a `git revert`. Prefer one concern per
  commit so either can be backed out cleanly.

## Loosened γ_SAR prior [0.001,10] — clamp compression + cache-token footguns 2026-07-13 (user request)
- **User asked to loosen the γ_SAR prior "from 0.001 to 10".** Motivation surfaced from 9j's
  `plot_gamma`: the negbin|neighbourhood posterior median sat at ~0.021 — **right on the old
  softclamp lower bound `log0.02`** — with an implausibly *tight* 90% CI (~[0.0202,0.0208]). That
  narrow-CI-pinned-at-the-bound signature is **softclamp compression**, not genuine certainty: the
  softplus squashes latent values approaching the bound, so the estimate can't move and its spread
  collapses. Whenever a posterior parks exactly on a clamp with a suspiciously tiny CI, suspect the
  clamp, not the data.
- **Change** (`framework.jl` `gamma_sar_prior`, `joint_model.jl` `model_transmission`, spec §"Update
  2026-07-12"/§6 prior list): prior `Normal(log0.27,1.05²)` [90% γ∈[0.048,1.52]] → `Normal(log0.1,1.8²)`
  [90% γ∈[0.0052,1.93]]; softclamp `[log0.02,log5]` → `[log0.001,log10]`. Centre = geometric mean of
  the requested [0.001,10]; σ=1.8 puts the clamp at ±2.56σ (outside the 90% band), i.e. the codebase
  idiom "clamp = outer safety bound, weakly-informative prior lives inside it". The old 0.021 estimate
  now sits at only −0.87σ (≈19th pctile), mid-bulk.
- **Cache footgun**: the Stage-2 caches are keyed only by `contacts_label(cfg)` (`temporal-gsar-cut-sc-sm`),
  which does **NOT encode the prior/clamp**, and `prefit_stage2!` skips on bare `isfile(path)`
  (joint_model.jl ~L600). So editing the prior alone ⇒ the 512 cached `8j_s2_*` chains are silently
  reused and results don't change. **Must delete `8j_s2_*` and re-run `prefit_stage2!`.** Did NOT bump
  the token: it is shared with `stage1_path`, so bumping would also orphan the γ_SAR-independent
  `8j_s1_*` Stage-1 GP fits and force a needless (expensive) Stage-1 refit. Deleting only s2 is the
  minimal correct action. (Contrast the 2026-07-11 dispersion revert, where a token bump WAS right
  because the model's saved columns changed.)

## Contact-only relative reproduction number in 9j — new diagnostic 2026-07-13 (user request)

- **What**: added a **contacts-only relative R** figure alongside the "contact & transmission" R in 9j.
  The two are now separate PNGs (user asked to split them 2026-07-13): full-NGM R →
  `res/9j_reproduction_number.png`, contacts-only relative R → `res/9j_contact_reproduction_number.png`
  (was briefly a `layout=(1,2)` composite of both into `9j_reproduction_number.png`).
  The relative R = `ρ(C*_origin)/ρ(C*_first origin)`, the dominant eigenvalue of the *bare*
  contact matrix C* alone (drop γ_SAR/susc/inf/antibody), normalised to the first forecast origin so every
  model passes through 1.0 there. Isolates how contact structure alone drove transmissibility vs a baseline
  week — the transmission scalings (which are per-origin-fit constants) cancel in the ratio; only C* varies.
- **Where**: `contact_reproduction_draws` in `8j_viz_utils.jl` (mirrors `reproduction_draws` but takes
  `max real(eigvals(Cstar_end[m]))` of C* only, no `wd`/`build_ngm`); collector
  `relative_contact_reproduction_over_time` + cached `_or_load` (cache `9j_relrt_<contacts>_h<h>.jld2`,
  distinct from the full-R `9j_rt_*`) + `plot_relative_contact_reproduction` in `9j_viz_utils.jl`.
- **Cheap by design**: the origin-week C* is ALREADY stored per-draw as `pooled.Cstar_end[m]` in every
  Stage-2 `8j_s2_*` file, so this needs **no Stage-1 reload/refit** — just eigvals of the stored matrices.
  Uses the `n_post` distinct Stage-1 C* matrices (contact structure is NGM-independent, Stage-1 only),
  not the pooled Np draws.
- **Gotcha carried over**: plot the real Date-bearing series BEFORE the `hline!(1.0)` reference, else the
  numeric axis locks in first and mangles the date ticks (same as `plot_reproduction`). Reference anchor =
  first forecast origin was the user's explicit choice (vs earliest fit week / per-window re-anchoring).
- **Observed model-free line 2026-07-13 (user request)**: added ONE extra line to the relative-contact
  figure = the RAW observed weekly mean-contact matrix's relative R, `ρ(emean_t)/ρ(emean_first origin)`,
  with NO GP / NO reciprocity / NO fit. `emean` is already computed by `prepare_degree_data` (the observed
  per-cell mean incl. zeros, `AgePairData.emp_mean[t,i,j]`), so the observed line just eigen-decomposes
  `emp_mean[end,:,:]` of the SAME horizon-h window (`WeeklyWindow(origin+7h)`, same `cfg.seed`) the Stage-1
  fit uses — guaranteeing its week/binning matches the model's `Cstar_end`. `observed_contact_reproduction_over_time`
  (+ cached `_or_load`, cache `9j_obsrt_<contacts>_h<h>.jld2`) in `9j_viz_utils.jl`; overlaid via the new
  `observed=` kwarg of `plot_relative_contact_reproduction` as a black step (`:steppost`, matching the four
  fitted step curves; black diamonds set it apart). It's the data-only baseline the fitted C* curves smooth.

## Fitting period capped at end-2021 — and why the data-derived origin bound lies 2026-07-15 (user request)

- **What**: `available_forecast_origins` (`degree_agepair.jl`) gained an `origin_max::Union{Nothing,Date}=nothing`
  kwarg (`tmax = min(tmax, week_start(origin_max))`); 8j and 9j both pass `FIT_END = Date(2021,12,31)` from their
  setup cells ⇒ **63 origins, 2020-10-18 … 2021-12-26** (was 107, … 2022-10-30). User's choice: the cap bounds the
  **origin**, not the target — the last window's h=1..4 targets legitimately run to 2022-01-23.
- **Why the uncapped bound is wrong** (the real gotcha): `tmax = last_contact_week − max(horizons)` trusts
  `maximum(craw.date)`, but **CoMix-UK has a hole**. The main panel stops **2022-03-02**; an isolated block runs
  **2022-11-16 … 2022-11-28** and *that block alone* drags `last_contact_week` from 2022-01-30 out to 2022-10-30.
  Origins ~2022-03-06 onward therefore roll through windows with **no contact data at all**. Don't read
  `extrema(craw.date)` as "the panel runs to here" — check monthly row counts.
- **Second, uncaught limit**: inc2prev `infections`/`gen_dab` end **2022-03-26**. `weekly_infections`/
  `weekly_antibody` (`infection_data.jl`) `continue` on unmatched weeks into **pre-zeroed** arrays, so origins past
  that are **silently zero-filled** — no error, just fake zero-infection windows. Nothing bounds origins by the
  infection series' *end*; only `infection_start` bounds the start.
- **No refit needed**: `8j_s1_*`/`8j_s2_*` are keyed by origin *in the filename*, so a cap only drops origins —
  the 63 kept fits stay valid (verified 504/504 Stage-1, 1008/1008 Stage-2 present). The four `9j_*` caches store
  their `origins` vector and self-invalidate on mismatch; `contacts_label(cfg)` does not (and should not) encode
  the range.
- **`period_summary` mislabels the tail** (pre-existing, `9j_viz_utils.jl`): `PERIODS` ends **2021-11-24**, and the
  `missing` bucket prints as `"(pre-Lockdown 2)"` — so origins 2021-11-28 … 2021-12-26 tally under that wrong
  label. `plot_rwis_by_period` is safe (drops `missing`). Left unfixed; flag if it matters.
- **NotebookEdit gotcha**: it rewrites the edited cell's `source` as a single JSON **string** and strips the EOF
  newline, exploding the git diff (whole cell shown as rewritten). Renormalise afterwards — `source` back to a
  list of `\n`-terminated lines, `json.dump(..., indent=1, ensure_ascii=False)` + trailing newline — to keep
  notebook diffs line-wise and reviewable.

## Plots.jl panel figures: derive the layout, never hard-code it 2026-07-15 (9j: 9 → 15 forecast panels)

- **A hard-coded `layout` silently caps a `n`-panel figure.** `plot_forecast_panels` took `n::Integer = 9` but
  ended in `layout = (3, 3)`, so any `n > 9` died with `When doing layout, n (9) < n_override (12)` — pointing at
  the `plot(...)` line, not at the `n` the caller passed. A previous session hit exactly this at `n = 12` and left
  the error stored in notebook cell `4109ada9`. If a panel count is a kwarg, the tile grid must be **derived**:
  `nrow = max(1, floor(Int, sqrt(np))); ncol = ceil(Int, np / nrow)` (9 → 3×3, 12 → 3×4, 15 → 3×5 — exact, no
  blanks, and wide grids suit a date x-axis). Size per-cell too (430×330 keeps 9 at the former 1300×1000).
- **`plot_title` adds an extra, series-less subplot.** `length(fig.subplots)` is panels **+1** whenever
  `plot_title` is set — a 15-panel figure reports 16. Assert on `count(sp -> !isempty(sp.series_list), ...)`
  instead; I wrote a wrong assertion first and briefly mistook it for a real failure.
- **Medians don't commute with a weighted average.** `susc`/`inf` super-group ratios are pop-weighted means of the
  per-bin ones **exactly, per draw** — but *not* after taking medians (`inf` differs by up to 11%, `susc` ~1%,
  tracking how skewed each posterior is). Verify such a refinement identity on the **raw draws** (`< 1e-10`), never
  on the stored `med` — an assertion on medians tests nothing and will fail for honest reasons.
- **`markerstrokewidth = 0` when markers are small and colour carries meaning.** The default black 1px stroke
  swamps a `ms = 1.5` marker, so a 7-line `:viridis` age palette rendered as 7 near-black lines — and, worse, 7
  identical **legend swatches**, which is the only thing identifying the series. Bit `plot_ratio_bins`.
- The Date-axis gotcha already noted for `plot_ratio` applies to any new facet: plot the 1.0 reference as a
  **Date-valued series first**; a leading `hline!` initialises a numeric axis and collapses the dates.

## Overlaying incommensurable series on one axis 2026-07-15 (9j: inc2prev R over the relative contact R)

- **When the user asks for a mixed-units overlay, say so in the figure — don't refuse and don't hide it.**
  `plot_relative_contact_reproduction`'s steps are a dimensionless ratio to a baseline week; inc2prev's
  national R is an absolute R. Same symbol, two meanings, and both hover near 1 — so a shared axis makes them
  *look* comparable. The user asked for the shared axis knowingly ("even though those two show not a
  comparative quantity"), which is legitimate: the shapes ARE worth comparing, only the vertical gap is
  meaningless. The fix is labelling, not refusal — ylabel says `MIXED UNITS`, the series label says
  `ABSOLUTE, different quantity`, and the single 1.0 line is labelled with BOTH jobs it does
  ("baseline week (steps) & R = 1 (inc2prev)"). A shared axis at least leaves the conflation visible;
  `twinx` hides it behind two independently-scaled ranges that invite reading a scaling artifact as
  agreement. (I built `twinx` first, from an earlier answer to the same question — the reversal was cheap
  because the overlay was one `national::Bool` kwarg, not a new function.)
- **A `ylabel` is measured against the axis HEIGHT** (it is rotated 90°). The 62-char mixed-units label
  clipped "MIXED UNITS" clean off the top of a 950×520 figure; ~39 chars is the budget here. Watch this
  whenever a label grows to explain something.
- **`marker = :circle` doubles `series_list`.** Plots emits a `scatter` *and* a `path` per marked series, so
  a 6-curve panel reports 11 entries. I asserted `== 6` and it failed on correct code (same family as the
  `plot_title` extra-subplot trap above). Assert on the **non-empty `:label`s** — what the reader actually
  sees — not on raw series counts.
- `pad_margins` sets left/bottom only; a right-hand axis label needs an explicit `right_margin` on the base
  `plot(...)`. (Moot now the twin is gone, but it will bite the next `twinx`.)
- Substantive result the overlay was built to show: relative contact R correlates with national R at only
  **Pearson ≈ 0.36–0.40 / Spearman ≈ 0.36–0.44** (n=63 origins, h=1) — and the RAW observed weekly means score
  the same (0.359), so that ceiling is the contact data's, not any modelling choice's. Contacts swing 0.5–2.8×
  while R stays in 0.75–1.33; the gaps line up with Alpha, Delta and the vaccine rollout.

## Two "reference age bins" coexisted — the model gauge and a viz renormalisation 2026-08-04 (9j)

- **"Change the reference age bin from 2-15 to 25-34" was already half-done, in a different place.** The
  *estimation* reference has been `cfg.ref_bin = 4` = `"25-34"` since 2026-07-31 (`framework.jl`,
  applied at `joint_model.jl` `susc = vcat(offs_s[1:ref-1], one(sig_s), offs_s[ref:end])`). The `2-15`
  the user was reading off the figures was a **separate post-hoc renormalisation living only in
  `collect_transmission_structure`**, which divided every draw by the pop-weighted `(2-10, 11-15)`
  super-group. Before touching anything, check whether the name in the request refers to the model or to
  the plot — the two had drifted apart and the docstring at the division site was the only thing
  recording it.
- **The renormalisation was deliberately gauge-invariant, and the fix knowingly gives that up.** Dividing
  by the 2-15 super-group made the figures independent of `ref_bin`; dividing by `V[:, ref_bin]` makes
  them *be* the gauge. Since `V[:, ref_bin] ≡ 1` exactly in the stored draws, the division is the
  identity — the "ratio" store is now literally the raw pooled draws, and 9j's per-bin medians equal
  10j's `make_susc_inf_fig` medians to 0.0. That equality is the cheapest end-to-end check that the
  rewiring landed; assert it.
- **Deleting a docstring claim is part of the change.** The old text advertised "GAUGE-INVARIANT to the
  model's reference-bin choice". Left in place it would have been actively false and would have sent the
  next reader looking for a bug. Same for the notebook's `2-15` comment block.
- **Viz-layer only ⇒ no cache invalidation.** `contacts_label` does not encode `ref_bin`, so a change to
  the *model* reference would silently reuse stale `8j_s2_*` chains — but a change to the *plot*
  denominator touches nothing. Verified: 0 of 2016 `8j_s[12]_*` artefacts modified by the rerun. Always
  confirm which side of that line a "reference" change falls on.
- **A super-group decomposition has to be rebuilt around the new reference.** With `2-15` as baseline the
  other groups were `16-49`/`>50`; with `25-34` the reference sits *inside* `16-49`. `supergroup_split`
  now splits whichever base group contains `ref_bin` into (before, [ref], after) — for the default grid
  `((1,2),(3,),(4,),(5,),(6,7))` = `2-15 / 16-24 / 25-34 / 35-49 / 50+`. Group names are rebuilt from
  `grid.LO`/`grid.HI` mirroring `cis_age_grid`'s own `LAB` construction, so a singleton group reproduces
  `grid.LAB[i]` exactly (and the last group is now `"50+"`, not the hand-written `">50"`).
- **Don't plot the reference group as a series.** It is identically 1.0 with a zero-width ribbon. The
  dashed grey 1.0 line already *is* that group — give it the label (`"25-34 (ref)"`) and skip the series.
- **A `$(…)` inside a `"""docstring"""` is interpolated at parse time.** Documenting the name-construction
  rule as `"$(LO[first])-$(HI[last])"` threw `UndefVarError: LO not defined` on include — the file parsed
  fine, so the parse-check passed and it only surfaced under the full stack. Write such examples with
  placeholder brackets, or escape the `$`.
- **`plot_title` costs one extra subplot.** `length(fig.subplots)` is panels + 1 whenever `plot_title` is
  set: a 3×2 figure reports 7. I asserted `== 6` against correct code. Same family as the
  `marker = :circle` doubling of `series_list` noted above — never assert on raw Plots container counts
  without accounting for these.
- Stale-label bug found in passing and fixed: `10j_viz_utils.jl`'s susc/inf figure title hard-coded
  `grid.LAB[1]`, so it printed `ref bin "2-10" = 1` over a figure whose 1.0 line had been bin 4 since
  2026-07-31. `cfg` was already in scope. `inst/analysis_plan_heavy_tail_mean.md` still names `(2-10)` as
  the baseline age group — left alone (the docx is source of truth), but it is stale.
