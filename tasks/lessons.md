# Lessons

Accumulated gotchas so the same mistake isn't repeated. Newest first.

## susc/inf smoothing REMOVED entirely → independent per-bin offsets 2026-07-13 (`joint_model.jl`, `framework.jl`, spec, 10j) (user request)

- **The relative susc/inf age profiles are no longer smoothed at all.** The `A-1` non-reference
  log-offsets are now **independent per age bin**: `susc = vcat(1, exp.(_softclamp.(sig_s .* z_s, log0.2, log5)))`
  (likewise `inf`), `z_s`/`z_i ~ filldist(Normal(0,1), A-1)`. This is the pre-`4607021` form (the GP was
  built on top of exactly this). The end of the GP→RW1→RW2→GP experiment sequence — now **none**.
- **What was removed** from `model_transmission` (`joint_model.jl`): the shared-length-scale RBF GP block —
  `log_rho_si`, `ρ_si`, `Ksi`, `Lsi`, and the `Lsi *` mixing (`sig·(Lsi·z)` → `sig·z`). Config
  (`framework.jl`): dropped `susc_inf_gp_len_prior` (its only consumer `log_rho_si` is gone) and
  **renamed** `susc_inf_gp_sd_prior` → `susc_inf_sd_prior` (the marginal-SD prior `N⁺(0.1,0.05²)` for
  `sig_s`/`sig_i` survives, still SEPARATE per profile).
- **KEPT (method-agnostic):** the soft-clamp `[log0.2,log5]` on the offset (NGM safety bound, not
  smoothing), the reference-bin pinning `susc[1]=inf[1]=1`, and `sig_s`/`sig_i`/`z_s`/`z_i`. Because the
  GP kernel had **unit diagonal**, removing it leaves the per-bin marginal SD unchanged (`=sig`) — the
  `[0.80,1.25]`/`[0.2,5]` calibration holds; the profile is just **rougher** (neighbours no longer pulled
  together). `sigma_inf` (obs-noise SD) is untouched — not a smoothing latent.
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
