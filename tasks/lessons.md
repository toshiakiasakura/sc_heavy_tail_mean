# Lessons

Accumulated gotchas so the same mistake isn't repeated. Newest first.

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
  `_softclamp(x,lo,hi) = x - softplus(x-hi) + softplus(lo-x)` (with a sign-branched stable
  `_softplus`). It is the **identity in the interior** (so the relative-pop-rescaled O(1) latents
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
