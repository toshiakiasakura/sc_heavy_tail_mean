# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Epidemiological research project fitting heavy-tailed count distributions to social contact survey data (**CoMix UK** during COVID-19). The goal is to model the degree distribution of daily contacts — decomposed into home vs. non-home — jointly with per-contact duration, and to track these distributions longitudinally. (A **Danon 2013** strand is now *legacy*: its raw data `dt_Leon_Danon_2013/` was removed as no longer needed, so `danon_utils.jl` and `4j_Danon_degree_duration.ipynb` are orphaned — code kept, not runnable.)

A second, newer strand (`inst/3_preliminary_model_struct.md`, `src/framework.jl` and friends) **extends** the age-stratified renewal / next-generation-matrix (NGM) forecasting model of Munday et al. 2023 (`inst/pcbi.1011453.pdf`; the upstream R+Stan reference `CovidAgeGroupForecast/` is **no longer vendored in-tree** — port provenance survives only in source comments and `tasks/lessons.md`). Here the age-pair **contact-degree distributions** feed the NGM directly (mean or neighbourhood/excess degree), driving a weekly renewal-equation forecast of age-stratified infections. Where the analysis plan (`inst/analysis_plan_heavy_tail_mean.docx`, converted to `.md`) differs from the reference code, **the docx wins**.

**Language**: Julia 1.12.4 (with embedded R via RCall for MGLM Dirichlet–multinomial regression and `scoringutils` WIS scoring).

## Environment & Commands

The canonical environment is the devcontainer (`.devcontainer/Dockerfile`, based on `quay.io/jupyter/datascience-notebook:julia-1.12.4`). R packages `socialmixr`, `MGLM`, `broom`, `tableone`, `scoringutils`, etc. are installed there.

`Project.toml` gained a `[compat]` section on 2026-08-05 (there was none before), pinning `julia = "1.12"` plus the packages that define the fit's numerics — so a Manifest/Julia mismatch is now a resolve error rather than the silent drift that let `Manifest.toml` keep claiming `julia_version = "1.11.1"` while the container ran 1.12.4.

**AD backend**: gradients for *both* stages go through `cfg.ad_backend` — `:mooncake` (default), `:reversediff` or `:forwarddiff` — resolved in one place by `_resolve_adtype`/`ad_type` (`src/framework.jl`). `main_utils.jl` loads Mooncake and ReverseDiff unconditionally so either is selectable at runtime. Mooncake pays a one-off rule build per model *type* per process (~66 s NegBin / 14 s hurdle-Weibull / 15 s Stage 2), which `prefit_stage1!` warms serially before its thread fan-out; a sysimage does **not** remove that (see Gotchas).

```bash
# Install Julia deps (run once, from /workdir)
julia --project=/workdir -e 'using Pkg; Pkg.instantiate()'

# Build sysimage — speeds up `using Turing/Plots/...` from ~minutes to seconds.
# Takes 15–30 min; produces /workdir/sysimage.so (~1 GB).
julia --project=/workdir /workdir/build_sysimage.jl

# Run Julia / notebooks with the sysimage
julia --project=/workdir --sysimage=/workdir/sysimage.so

# Execute a notebook headless
jupyter nbconvert --to notebook --execute src/<name>.ipynb --inplace

# One-time R dep for forecast scoring (src/scoring.jl)
Rscript -e 'install.packages("scoringutils")'
```

`JULIA_NUM_THREADS=12` and `JULIA_DEPOT_PATH=/home/jovyan/.julia:/opt/julia` are set by the devcontainer.

**No formal test suite.** Verify changes by running affected notebook cells, or by spot-checking with `pdf` / `ccdf` / `mean` calls on constructed distributions. The `precompile_script.jl` exercises the main libraries (Distributions, DataFrames, Plots, Turing) and doubles as a smoke test.

## Repository Layout

```
src/
  main_utils.jl              Canonical preamble — single include covers all sub-includes & using-statements
  utils.jl                   Plot defaults, value_counts, model_abbr / setting_pretty dicts
  data_setup.jl              Raw CSV/Arrow → normalised DegreeDist DataFrames (CoMix pipeline)
  degree_dist.jl             DegreeDist struct {x, y}; contact_degrees; duration weighting (K=5)
  fit_utils.jl               Post-processing across models: mean/CI extraction, WAIC summaries
  turing_models.jl           Turing @model definitions (ZeroInf, ZeroTrunc, convoluted, hierarchical)
  turing_utils.jl            Chain extraction, convergence checks (ESS>200, Rhat<1.1)
  mglm_utils.jl              R-backed MGLM Dirichlet–multinomial fit/predict wrappers
  bnb_utils.jl               BNB-specific fitting helpers
  comix_uk_time_series.jl    Longitudinal pipeline — per-wave 2-week windowed DegreeDist; inc2prev week anchor
  danon_utils.jl             [LEGACY] Danon 2013 loader (K=4 duration scale) — raw data removed, not runnable
  vis_utils.jl               Shared plotting helpers (marker sizing, log ticks, panel layouts)
  distributions/             Custom Distributions.jl-compatible library
    main.jl                    Entry point — `include("distributions/main.jl")` loads all of these
    poisson_mixture.jl         NegBin, PoissonLogNormal, PoissonLomax + ZeroTrunc wrappers
    bnb.jl                     BNB(α,β,r) and BNB2(m,v,η) — Beta Negative Binomial
    zeroinf.jl                 ZeroInfDist, ZeroInfConvolutedDist
    zerotrunc.jl               ZeroTrunc variants; ZeroTruncConvolutedDist (also in zeroinf.jl — second def wins)
    kernel.jl                  InterpKDEDistribution — KDE as a Distributions.jl prior
    helper.jl                  ZeroInf → ZeroTrunc conversion utilities
    plot.jl                    Log-log PDF/CCDF plotting

  # --- forecasting framework (spec: inst/3_preliminary_model_struct.md) — second preamble via forecast_utils.jl ---
  forecast_utils.jl          Forecast preamble — includes main_utils + CoMix pipeline + all framework modules below
  framework.jl               Swap-axis types (ContactDegreeModel, NGMBuilder), WeeklyWindow, FrameworkConfig, containers, CIS age grid
  infection_data.jl          Weekly infections (rolling-sum × pop) + gen_dab antibody from inc2prev estimates
  degree_agepair.jl          Age-pair (7×7 CIS bin) per-cell weekly contact-degree assembly (reuses 7j binning)
  renewal.jl                 Generation-interval PMF (log-normal) + weekly renewal iteration / frozen-contact forecast
  ngm.jl                     NGM builders (Mean / Neighbourhood / DiagonalMean / Null): C* = builder C0 used directly, no post-hoc balancing + leaky susceptibility
  joint_model.jl             Two-stage cut: model_degree (Stage 1) + model_transmission (Stage 2); fit_stage1/stage2_inputs/fit_stage2_pooled; two_stage_forecast; prefit_stage1!/prefit_stage2!; null_contact_level/null_moment_draws
  scoring.jl                 WIS via R scoringutils (RCall, quantile format) + LOG SCORE via the sample class (to_sample_long/score_logs) + native sample CRPS cross-check
  8j_viz_utils.jl            Read-only 8j viz — reconstructs susc/inf/ρ from cached joint-model chains (no model rebuild)
  9j_viz_utils.jl            Forecast assembly cache + scoring reports + all 8j/9j figures (REF_MODEL, panel_grid)
  10j_viz_utils.jl           Model diagnostics — μ/dispersion reconstruction from Stage-1 chains (stays on the original four combos); `_read_disp_chain` carries a legacy `-hd` branch for the retained hierarchical chains
  11j_viz_utils.jl           Weekly identifiability (mean vs neighbourhood moment timelines) + `plot_within_block_sd` (current vs legacy `-hd`; the current line must be identically 0)
  12j_viz_utils.jl           Stage-1 NUTS chain convergence: `load_nuts_chain`, `convergence_table`/`_summary` (per-block worst ESS + split-R̂), `hmc_health` (adds E-BFMI and `frac_at_cap`), 7 figures, `convergence_verdict`

  <N>j_*.ipynb               Numbered notebook entry points:
                               1j data explore · 2j duration · 3j effective degree · 4j Danon [legacy] ·
                               5j group contacts · 6j fitting dist · 7j weekly age-pair · 8j forecast fitting ·
                               9j forecast diagnostics/scoring · 10j model diagnostics · 11j weekly identifiability + shrinkage ·
                               12j Stage-1 NUTS chain convergence (one origin × both degree models) ·
                               13j model diagnostics, h=1 ONLY — a variant of 10j for generations fitted at a single
                                 horizon. Generated from 10j by `tmp/build_13j_nb.py` (never hand-edit); the h1 lever is
                                 `FrameworkConfig(horizons=1:1)` and figures go to `res/13j/`. EXISTS BECAUSE running 10j
                                 against h1-only artefacts does not fail loudly — `two_stage_forecast` → `fit_or_load_stage2`
                                 FITS a missing artefact, silently launching NUTS for h=2,3,4.
                             NOTE notebooks are stored PLAIN (no outputs, `execution_count = null`) — verify with
                             `jupyter nbconvert --execute --output-dir <scratch>`, never `--inplace`.

dt_comix_no_public/          Primary CoMix-UK data — contacts_uk.arrow / part_uk.arrow (+ csv/qs mirrors). gitignored (local only)
dt_surveys/                  [ABSENT] Not in tree; legacy loaders in data_setup.jl/utils.jl still reference ../dt_surveys/ paths (e.g. 2014 Read China)
dt_intermediate/             Fitted Turing chains (JLD2): date-stamped dist fits + 8j_s1_*/8j_s2_* forecast artefacts; cached df_dds*.csv
                               ⚠ As of 2026-08-06 this holds ONLY df_dds.csv + df_dds_settings.csv — no chains. The 504-file
                               Pathfinder grid under CONTACTS_TOKEN_PF actually lives in dt_intermediate_GP_RBP/; other
                               generations are in dt_intermediate_hierarchical{,_ver1}/, dt_intermediate_age_block_ver1/,
                               dt_intermediate_old/. Check before assuming a token resolves anything.
inc2prev/                    Git submodule (github.com/epiforecasts/inc2prev). Supplies the CIS age/population grid
                               (data-processed/populations.csv) and infection + antibody estimates (outputs/estimates_age_ab.csv)
inst/                        Spec markdown — 2_weekly_age_pair.md (→7j), 3_preliminary_model_struct.md (→8j+framework),
                               analysis_plan_heavy_tail_mean.md/.docx (overall plan, docx wins), pcbi.1011453.pdf (Munday 2023), media/
                               — read these for task source-of-truth
# [REMOVED] dt_Leon_Danon_2013/ (Danon raw data) and CovidAgeGroupForecast/ (Munday R+Stan reference) are no longer in the tree.
res/                         Output artefacts (PNGs, score CSVs)
tasks/                       Working plans (todo.md) and accumulated lessons (lessons.md)
```

## Architecture: Key Patterns

**Two preambles**: general notebooks/scripts start with `include("main_utils.jl")` from `src/` (all `using` statements + custom distributions + DegreeDist + Turing stack). The **forecasting framework** uses `include("forecast_utils.jl")` instead, which pulls in `main_utils.jl`, the CoMix pipeline (`data_setup.jl`, `comix_uk_time_series.jl`), and all framework modules in dependency order. Always run scripts from `src/` (relative paths, including `../dt_comix_no_public/…` and `../inc2prev/…`, assume `cwd == src/`).

**Custom distributions** extend `Distributions.DiscreteUnivariateDistribution`. Pattern:
```julia
Base.@kwdef struct MyDist <: DiscreteUnivariateDistribution
    α::Float64; β::Float64
end
Distributions.logpdf(d::MyDist, k::Int64) = begin
    @unpack α, β = d   # Parameters.jl
    ...
end
```
Use `@memoize` (Memoization.jl) on expensive `pdf`/`ccdf` methods called inside MCMC likelihoods.

**Turing models** sample parameters in log-space (`log_m`, `log_σ`) and `exp()` them inside the model body to enforce positivity for HMC/NUTS. They use `Turing.@addlogprob!` to inject manually-computed log-likelihoods rather than `x ~ dist`. MCMC entry for the distribution-fitting strand: `fit_model_with_forward_mode(model, n_sample)` with `Random.seed!(1236)`.

**Convoluted distribution** (primary CoMix model): `ZeroInfConvolutedDist` convolves a home distribution (`ZeroInfDist(PoissonLogNormal)`) with a non-home distribution (`ZeroInfDist(PoissonLomax)` or `ZeroInfDist(BNB2)`), fitted jointly to both marginals.

**DataFrame transforms** use DataFramesMeta macros throughout: `@subset`, `@transform!`, `@select`, `@rename!`, `@pipe`, `@chain`, `@byrow`. Unicode identifiers (`π0`, `μ`, `σ`, `α`, `θ`, `ξ`) are normal — don't ASCII-fy them.

**Saving chains**: `JLD2.@save` / `JLD2.@load` to `dt_intermediate/` with date-stamped filenames like `2020-04-12_zeroinf_bnb2_nhm.jld2`.

**CoMix duration scale**: CoMix uses K=5 bins with midpoints `(2.5, 10, 37.5, 150, ∞)` min (see `_DURATION_T_MID` in `degree_dist.jl`). *(Legacy: the removed Danon strand used K=4 midpoints `(5, 20, 45, ≥60)` min, `d_max = 60`, `_DANON_T_MID` in `danon_utils.jl` — do not remap one scale onto the other if you ever revive Danon.)*

## Architecture: Forecasting Framework

The composable forecast pipeline (spec: `inst/3_preliminary_model_struct.md` — a reverse-engineered spec of the coded model; driver notebook: `src/8j_preliminary_forecast.ipynb`) is built around **two swappable axes** that are chosen once per fit and dispatched deterministically. This is the design that must be preserved when adding models.

- **Axis 1 — `ContactDegreeModel`** (how the age-pair contact degree is modelled): `NegBinAgePair` (unweighted integer counts), `HurdleWeibullAgePair` (duration-weighted, hurdle), or `NoContactDegree` (none — the null model). `is_weighted(dm)` branches the likelihood; `needs_stage1(dm)` is `false` only for `NoContactDegree`.
- **Axis 2 — `NGMBuilder`** (how the NGM's per-capita contact `C0` is formed from the degree distribution's raw moments `⟨k⟩, ⟨k²⟩`): `MeanNGM` (`C0 = ⟨k⟩`), `NeighbourhoodDegreeNGM` (`C0 = ⟨k²⟩/⟨k⟩ = m(1+CV²)`, the excess/size-biased degree), `DiagonalMeanNGM` (`C* = diagm(diag(⟨k⟩))`) or `NullNGM` (`C*` a fixed uniform constant). `fix_infectivity(nb)` is `true` only for `DiagonalMeanNGM`.

The "**four ways**" of the preliminary analysis is the 2×2 grid of the first two options on each axis. **Two baselines** (`inst/6_null_interaction_model.md`, 2026-07-30) sit alongside it as two further pairings of the same axes:

- **no-interaction** `unweighted-negbin|mean-diagonal` = `NegBinAgePair × DiagonalMeanNGM`. Only `diag(C*)` enters the NGM, so age groups don't infect each other; age-dependent infectivity is pinned to `inf ≡ 1` because a diagonal NGM identifies only the product `susc_a·inf_a`. It **reuses the NegBin Stage-1 chains verbatim** (`8j_s1_*` carry no ngm token) ⇒ no extra Stage-1 fits. This is now `REF_MODEL`, the relative-skill reference in `9j_viz_utils.jl`.
- **null** `no-contact|null` = `NoContactDegree × NullNGM`. No contact data at all: `C*` is a uniform constant `c̄` = roster-weighted mean *unweighted* contacts per participant-day over the origin's 8 focal fit weeks, ÷ A (`null_contact_level`), held fixed across horizons. Stage 1 is skipped entirely (`stage2_inputs` returns one `null_moment_draws` draw with the full 10 000 Stage-2 samples). Forecasts are **invariant** to `c̄` — it only sets γ_SAR's scale (verified: `c̄`×10 ⇒ γ_SAR ×0.105, R unchanged to 0.3%).

**Two-stage cut inference** (`inst/4_cut_Bayes.md`, 2026-07-12): the former single joint `@model` was split. **Stage 1** `model_degree(dm, ds, pop, cfg)` fits the contact-degree GP alone and returns per-week raw moments `(; K1, K2, G)` — **NGM-independent**, so one fit serves both builders (the builder is applied downstream via `contact_star(nb,…)`). **Stage 2** `model_transmission(Cstar_weeks, wd, w, cfg, nb=MeanNGM())` fits the infection block conditioning on a *fixed* `Cstar_weeks` from one Stage-1 draw (the trailing `nb` is consulted **only** for `fix_infectivity`; the C* functional was already applied upstream). The cut Monte Carlo: `fit_stage1` → `stage1_moment_draws` (100 draws) → `fit_stage2_pooled` (100 Stage-2 draws each) → pool 100×100 = **10 000** infection draws for WIS. `dm`/`nb` remain fixed args. There is **no feedback** from infections to the contact GP, so μ is NGM-independent. `stage2_inputs(dm,…)` is the single fork between this path and the **null** bypass (no Stage 1; one constant-C* draw × 10 000 Stage-2 samples).

**Fit path**: `fit_stage1` runs **NUTS** (`cfg.stage1_use_nuts`, the default since 2026-08-05), initialised from the Pathfinder mean via `_pf_mean_init` — so Pathfinder still runs first and the cost is **additive**; set `stage1_use_nuts=false` for the Pathfinder-only preliminary. NUTS is configured explicitly (`cfg.stage1_nuts_adapts=1000`, `_draws=500`, `_target_accept=0.95`, `_max_depth=10`) because a bare `NUTS()` would derive only 125 warmup iterations in 389/977 dimensions. One chain per fit ⇒ **no R̂**; health comes from `_nuts_diagnostics` (divergences, `frac_at_max_depth`, `min_ess`). **Stage 2 is always Pathfinder and has no NUTS path at all** (100 cheap fits per Stage-1 draw — the point of the cut). Seed `cfg.seed = 1236`. Both stages' gradients come from `cfg.ad_backend` (default `:mooncake`). Drivers: `prefit_stage1!` then `prefit_stage2!` (or `prefit_two_stage!`); forecast `two_stage_forecast`. Artefacts: `8j_s1_<degree>_<contacts>_<origin>_h<h>.jld2` (Stage-1 chain, key `result`, plus self-describing `sampler`/`diag`/`ad_backend` keys, no ngm token) and `8j_s2_<degree>_<ngm>_<contacts>_<origin>_h<h>.jld2` (Stage-2 pooled, key `pooled`). `CONTACTS_TOKEN` now carries `-s0-m32-t0-ar1-nuts`; the existing 504/1512-file Pathfinder generation is reached by **`CONTACTS_TOKEN_PF`**, which became a LITERAL `"temporal-gsar-cut-sc-p0-gi"` when `-s0` landed — it now differs by a model component as well as the sampler, so no `cfg` reproduces it (same directory; suffix alone distinguishes them).

**Transmission core** (Julia ports of the Munday-2023 Stan model `multi-option-contact-model.stan`; the `CovidAgeGroupForecast/` reference tree is no longer vendored — `renewal.jl`/`infection_data.jl` comments cite the original file:line): weekly renewal `I(t) = Σ_{s=1}^{smax} w(s)·N(t)·I(t−s)`, `smax=4`; `N_ab = γ_SAR · full_susceptibility_a · C*_ab · inf_b` with leaky antibody protection `full_susceptibility_a = susc_a·(1+(F−1)·A_a(t))`. `γ_SAR` (`build_ngm(…; gamma_sar=…)`) is the **per-contact secondary attack rate**; `C*` is the builder's `C0` used **directly** and **un-normalised** (the `-gnorm` C*/S̄ decoupling was reverted 2026-07-12), and reciprocity lives in the contact-mean estimation, so there is **no** post-hoc symmetrisation (see Gotchas). Forecasts freeze the NGM at the origin week and iterate the renewal forward.

**Weekly grid**: inc2prev-aligned, Sunday-start, anchored at `WEEK_ANCHOR = 2021-03-21`, Wednesday mid-date labels. A `WeeklyWindow(origin; n_fit=8, smax=4, horizons=1:4)` yields `lag_weeks` (renewal history) ++ `fit_weeks` = `all_weeks`, plus horizon `forecast_weeks`. Age grid = the 7 England CIS `age_school` bins (`2-10 … 70+`) from `inc2prev/data-processed/populations.csv`.

**Scoring** (`scoring.jl`): primary metric is **WIS** via R `scoringutils` v2 on quantile-format forecasts (`to_quantile_long` → `score_wis`); native sample CRPS is a cheap cross-check. A **log score** (`to_sample_long` → `score_logs`, `inst/6`) is reported alongside — `scoringutils` defines it only for the **sample** class, so it is a second R path over the raw draws, scored one origin at a time, on both scales. Don't confuse it with the existing "log-scale WIS", which is WIS after a log *transform*. Outputs go to `res/8j_*` (scores + `8j_logscore_*`).

Contacts are **per-week** by default (`constant_contacts=false`, the notebook setting): the age-pair mean is estimated separately each window week by a separable spatio-temporal GP (per-week level `c_t` + matrix-normal field, shared `ρ_diag/ρ_time`, `η`, `σ_c`), so the renewal `N(t)` varies through that week's `C*_t` as well as antibody.

**The structure field is SUM-TO-ZERO over the 28 age pairs within each week** (`-s0`, 2026-08-05). `R = η·(Q·La·z·Ltᵀ)` with `Q = _sum_zero_basis(28)` the constant Helmert basis of 1^⊥ and `La = chol(Qᵀ·Kp·Q + 1e-6·I)`, so `Cov(vec R) = η²·(Kt ⊗ M·Kp·M)` — the same GP *conditioned*, not approximated. Without it the field's per-week mean was a second copy of `c_t` and `η`/`σ_c` were confounded (worse as `ρ` grows, since `Kp → J`). `z` is `27×Tn`, so Stage 1 is **389** (NegBin) / **977** (hurdle-Weibull) latents (`z_c` is `Tn−1` since `-t0`). Two consequences: `η` is no longer exactly the marginal SD (the field's SD is `η·sqrt(diag(M·Kp·M))` — measured ×0.71–×1.08 at the `gp_len_prior` mode under `-m32`, so `gp_scale_prior` was left alone), and **do not renormalise `Ap` by `tr(Ap)/P` to "restore" it** — as `ρ→∞` that is dominated by the jitter and degenerates the field to white noise, inverting the correct limit (field → 0, `c_t` carries everything).

**Both GP kernels are MATÉRN 3/2, and the spatial kernel smooths BOTH directions** (`-m32`, 2026-08-05). The age pair is rotated 45° into `u = (mid_a+mid_b)/√2` (total age) and `v = (mid_a−mid_b)/√2` (age gap); `Kp[m,n] = m32(|Δu|/ρ_diag)·m32(|Δv|/ρ_gap)` with `m32(x) = (1+√3x)·exp(−√3x)` — a separable product of two 1-D Matérn 3/2 factors, so it is unit-diagonal by construction and `ρ_diag = ρ_gap` does **not** give an isotropic Matérn (that identity held only for the old squared exponential). **The TEMPORAL kernel is no longer Matérn 3/2** — see `-ar1` below. Both `log_rho_diag` and `log_rho_gap` draw from the single `gp_len_prior`. **Why the kernel family changed**: the squared exponential's eigenvalues decay super-exponentially, so `Kt` went numerically rank-4-of-12 as ρ_time drifted and `z[:,1]` got pinned ~60× tighter than the other columns — the geometry behind 100 % max-tree-depth and min ESS 1.9–5.4/500 in the 2026-08-05 NUTS pilot. Measured: `Kt` keeps **full rank 12 at every ρ_time** in `RHO_TIME_BOUNDS`; `Ap` keeps rank 27 across all of `RHO_BOUNDS` (min eigenvalue 2.5e-2 at the `gp_len_prior` mode vs 2.9e-5 for SE, and 2.1e-3 vs **1.5e-8** — below the 1e-6 jitter — at +2σ); `cholesky(Ap + 1e-6·I)` clean at all 625 points of a 25×25 (ρ_diag, ρ_gap) grid. This **supersedes `-diag`**, which for a few hours the same day dropped `log_rho_gap` (389/977) and did not fix the mixing problem. ⚠ The ρ priors moved with it — `gp_len_prior` N(log 20, 0.35²) and `gp_time_len_prior` N(log 2, 0.35²) — and the tight time prior is what keeps ρ_time inside the identified region; the two changes are complementary, don't revert one alone. (An InverseGamma variant `-ig` was tried and REVERTED on 2026-08-06 — see `tasks/lessons.md`; it improved mixing but let hurdle-Weibull's ρ_time drift to 47–66 wk.)


**The TEMPORAL correlation is AR(1)** (`-ar1`, 2026-08-06, user request). `Kt[s,t] = φ^|s−t|`, which *is* an AR(1) correlation matrix (equivalently the exponential / Matérn 1/2 kernel), replaces the Matérn 3/2 in the **time direction only** — the spatial kernel is untouched. No structural change was needed: the separable matrix-normal already gives every age pair its own temporal trajectory under one shared amplitude `η` ("AR(1) per age pair, sharing the variance"), and the pairs stay correlated across age through `La`. `log_rho_time` → **`phi_time`**, sampled `Beta(cfg.ar1_phi_prior...)` = **Uniform(0,1)** by default — deliberately weak, and the first thing to revisit. ⚠ The latent count is **unchanged at 389/977**, so the count cannot date a chain; the in-chain signal is the name (`phi_time` ⇒ `-ar1`, `log_rho_time` ⇒ earlier) plus the token. **Why**: measured at *matched effective rank* (same temporal pooling), AR(1) gives `Kt` min eigenvalue 2.7e-05 → 5.1e-03 and `Lt` column spread 94.2 → 23.1 at effrank 1.08, and the LEVEL's `Lc` spread is essentially **flat at 1.6–1.8 across every φ** where Matérn 3/2 degrades to 13.5. Mechanism: AR(1) is **Markov** — tridiagonal precision, polynomially-decaying eigenvalues — so it keeps spectral mass in the non-constant directions even at φ = 0.995, where Matérn 3/2's spectrum has collapsed and `Lt`'s first column absorbs the field. It is a wash in NegBin's regime (ρ_time ≈ 2.2) and helps hurdle-Weibull, whose likelihood wants near-constant contacts. ⚠ This is a **modelling** change too: AR(1) paths are non-differentiable and memory is LONGER at long lag (at matched lag-1 0.785, lag-4 is 0.380 vs 0.140). ⚠ It does **not** change what the likelihood wants — **a high φ is the measurement, not a failure**. ⚠ `RHO_TIME_BOUNDS` and the temporal soft-clamp are gone from this path (φ ∈ (0,1) by construction, φ^k cannot overflow); the constant is retained only for replaying archived chains.

**The weekly LEVEL is SUM-TO-ZERO over the window weeks** (`-t0`, 2026-08-06). `cₜ = c + σ_c·(Qt·Lc·z_c)` with `Qt = _sum_zero_basis(Tn)` and `Lc = chol(Qtᵀ·Kt·Qt + 1e-4·I)` — the temporal analogue of `-s0`, same machinery, so `Cov(σ_c·dev) = σ_c²·(Mt·Kt·Mt)` is the same GP *conditioned* on the deviation summing to zero. `z_c` goes `Tn` → `Tn−1` ⇒ **389/977** latents. Why: `c` and the time-mean of `σ_c·(Lt·z_c)` were two parameterisations of one quantity — measured on all four `-m32` chains at corr **−1.000 exactly**, SD(c) ≈ SD(dev) ≈ 0.38–0.73 against SD(sum) = **0.007**. ⚠ **LEVEL ONLY** — do *not* also project the structure field's time axis: `R`'s per-pair mean over weeks duplicates nothing, so constraining it would force every pair's structure to average to zero across the window (a model restriction, not a reparameterisation). ⚠ 389/977 coincides with `-diag`'s counts; distinguish by `log_rho_gap` (present) and `z_c` = Tn−1.

**Dispersion is BLOCK-LINEAR × WEEK, with no per-cell term** (`inst/3` §4.3; reverted 2026-08-02). `log d_{ij,t} = β[bl,t]`, i.e. `log_k ~ filldist(Normal(0,1.0), 4, Tn)` (NegBin dispersion `φ`) / `log_kappa ~ filldist(Normal(0,0.5), 4, Tn)` (Weibull shape `κ`), so all 49 ordered cells inside a child/adult block share one value each week. Stage-1 latent count: **389** (NegBin) / **977** (hurdle-Weibull). A per-cell random effect lived here from 2026-07-30 to 2026-08-02 — first a flat hierarchy `τ_t·z_{ij,t}` (`-hd`), then a regularised horseshoe `τ·λ̃_{ij,t}·z_{ij,t}` (`-rhs`, Piironen & Vehtari 2017 eq. 11) — and BOTH were removed as unidentifiable; do not reinstate without reading `tasks/lessons.md` first. The `-hd` chains are retained in `dt_intermediate_hierarchical/` and are still read by `plot_within_block_sd`/`plot_tau_over_weeks`. `constant_contacts=true` recovers the pooled one-`C*`-per-window preliminary. `model_degree` branches on the flag but returns per-week moments `(; K1, K2, G)` (length `Tn`) either way; the forecast NGM uses the origin-week slice via `Cstar_end[m]` in the Stage-2 pooled result.

## Known Gotchas

- **The AD backend is NOT in the cache token — the artefact records it instead.** `contacts_label` encodes the *sampler* (`-nuts`) but deliberately not `cfg.ad_backend`: the target density is the same function and AD only supplies its gradient, so encoding it would fork the 504/1512-file grids for no scientific difference. But draws are **not** bit-identical across backends (different accumulation order ⇒ chaotically different LBFGS/NUTS paths), so a partially-refitted grid is **mixed-provenance**. Every `8j_s1_*` written since 2026-08-05 carries an `ad_backend` key; audit with `countmap([jldopen(p) do f; haskey(f,"ad_backend") ? f["ad_backend"] : :legacy end for p in glob("8j_s1_*", "../dt_intermediate")])` and regenerate the whole grid under one backend before publishing.
- **`NegBin`'s fields are parametric and must stay that way** (`src/distributions/poisson_mixture.jl`). With abstract `::Real` fields Mooncake ran the 402-dim Stage-1 model at **2.7 grad/s vs ReverseDiff's 41.8** — a 15× regression — while the concrete-typed hurdle-Weibull path was already 9× *faster*. Parameterising took it to 482 grad/s with the log-density unchanged to the last bit. The other legacy distributions here still use `::Real`; they are off the AD hot path (ForwardDiff-fitted) and are deliberately left alone.
- **A sysimage does not remove Mooncake's per-model warm-up.** `:Mooncake` in `build_sysimage.jl` bakes in Mooncake's own inference/codegen, but `build_rrule` keys on the concrete `DynamicPPL.Model` type, which doesn't exist until `forecast_utils.jl` is included at runtime. `prefit_stage1!` warms one fit per *degree-model type* (a `Set{DataType}`, not the old single `Ref{Bool}` that only ever warmed `dms[1]`) before its `Threads.@spawn` fan-out — deriving inside the fan-out is correct but parks every other worker on Mooncake's global lock while it holds a semaphore slot.
- `Lomax` struct is defined twice in `src/distributions/poisson_mixture.jl` — don't add a third.
- `ZeroTruncConvolutedDist` appears in both `zeroinf.jl` and `zerotrunc.jl` — the second definition silently overwrites the first.
- `Distributions.cov(d)` in this codebase returns the **coefficient of variation**, not covariance — non-standard naming.
- *(Legacy Danon)* `C_Duration` codes in Danon's `Contact_data.csv` are `{-1, 0, 1, 2, 3}`, offset by one from the codebook's `{1..4}` (`-1 → NA`, `0..3 → 1..4`); `C_Number ∈ {-1, 0}` rows (43) are anomalous and dropped. Data now removed — kept only as reference if the strand is revived.
- **inc2prev `estimates_age_ab.csv` schema**: `name=="infections"` is a **daily per-capita proportion** — multiply by `population` and 7-day-sum for weekly counts. `name=="gen_dab"` is antibody prevalence ∈[0,1] but its `date` column is **empty** — dates are reconstructed from `t_index` via the `infections` rows' `t_index→date` map (see `infection_data.jl`).
- **Turing + Weibull/exp over/underflow**: `κ = exp(log_kappa)` can go non-finite on aggressive Pathfinder/optim steps → `DomainError: Weibull α>0`. Log-parameters (`log_kappa`, `log_k`, `log μ` in `model_degree`; `log_gamma_sar` in `model_transmission`) are soft-**clamped** (`_softclamp`); the mode stays interior so gradients are unaffected. Keep those clamps when editing either model. **`_softclamp` must stay the Inf-safe nested form** `lo + softplus((hi − softplus(hi − x)) − lo)` — the naive `x − softplus(x−hi) + softplus(lo−x)` returns **NaN** at `x=±Inf` (`Inf − Inf`), so when a raw log-latent (`log_kappa`/`log_k`) or the GP field overflows to Inf, `κ=exp(softclamp(Inf))` becomes NaN and the Weibull throws (aborting the whole fit). This bit the **neighbourhood** NGM hardest (its `gamma(1+2/κ)` second moment pushes the optimiser to extremes); see `tasks/lessons.md` 2026-07-11. (`_softcap`, the upper-only sibling, went with the horseshoe's `lam` on 2026-08-02.)
- **The forecast already uses FUTURE contacts.** For horizon `h`, `fit_or_load_stage2` fits Stage 1 on `WeeklyWindow(origin + 7h)` and the forecast NGM is frozen at `Cstar_end[m]` = the **origin+h** contact matrix, while infections/antibody stay at the origin. This is deliberate (the "contact-updated iterate", Munday's design), not a leak to fix — and it is why `inst/6`'s "the no-interaction model is allowed to use the future mean contacts" needed no forecast-path change. The **null** model is the sole exception: its `c̄` is computed from the *origin* window's focal weeks and reused for every horizon.
- **Log score ≠ log-scale WIS.** `scoringutils` computes `log_score` only for the **sample** forecast class, never for quantile forecasts — so it cannot be added as a flag to `score_wis`. `score_logs` is a separate R path over the raw draws. Its KDE cannot consume the `±Inf` draws `two_stage_forecast` deliberately keeps, and `log_shift` returns `NaN` on the negative draws a Gaussian fan contains; both are sanitised, **counted, and printed** rather than silently dropped.
- **Per-model panel figures must derive their layout** (`panel_grid` in `9j_viz_utils.jl`). `plot_ratio` / `plot_ratio_bins` / `plot_lengthscales` were hard-coded `layout = (2, 2)` while there were exactly four models and silently lose panels at six.
- **scoringutils v2 quantile levels**: keep the `round(q; digits=2)` in `to_quantile_long` — scoringutils matches interval endpoints by **exact Float64 equality**, and one drifted endpoint makes the whole set asymmetric. Measured 2026-07-30 (`verify_scoring.jl`): injecting `0.75 → 0.7500000000000001` drops **`wis`, `overprediction`, `underprediction`, `dispersion`, `ae_median` and *both* coverages** from `score_wis`'s frames — `by_model` comes back as just `["model","scale","bias"]`. The columns are **absent, not NA**, and R emits only *warnings*, so `score_wis` returns normally: a `nrow(sc) > 0` smoke test passes with every metric gone. Assert on the **columns and their values**, never on row count. (Two earlier claims here were wrong: `collect(0.05:0.05:0.95)` does **not** drift — Julia ranges use `TwicePrecision`, 0/19 levels off — the risk comes from constructions like `cumsum(fill(0.05,19))`; and the 90% endpoints do **not** survive.) Correct wiring reproduces nominal coverage: 0.498/0.896 vs 0.50/0.90 over 560 units.
- **Neighbourhood NGM on empty cells (per-week)**: with `constant_contacts=false` many per-week Weibull cells are fully empty (`p⁰=1 ⇒ ⟨k⟩=⟨k²⟩=0`), so the raw `k2/k1` is `0/0=NaN`. `base_contact(::NeighbourhoodDegreeNGM,…)` is guarded `k1>0 ? (k2/k1)*g : zero(k1)`; keep the guard. (NegBin keeps `k1=μ>0`, so it never triggers.)
- **No post-hoc reciprocity balancing** (removed 2026-07-08): reciprocity is baked into the contact-*mean* only — `log μ_{i→j} = r_{min,max} + log N_j` ⟹ `N_i·μ_{i→j} = N_j·μ_{j→i}` exactly. `contact_star(nb, K1, K2, G)` returns the builder's `C0` unchanged (no `pop` arg, no `reciprocity_balance` function). So the **Mean NGM** inherits exact reciprocity from μ, but the **Neighbourhood NGM**'s size-biased `C*` is *not* reciprocal — intended. Don't reinstate the symmetrisation from the reference/docx (this decision overrides the docx total-contact balance). Neighbourhood `.jld2` chains predating this are stale.
- `CovidAgeGroupForecast/` was the upstream **reference** (Munday 2023 R+Stan), never a dependency, and is **no longer in the tree**. Its Stan diverged from both the paper and its own "joint" file (see `tasks/lessons.md` for the specific divergences); don't treat those ports as ground truth over the analysis-plan docx. Source comments in `renewal.jl`/`infection_data.jl` still cite its original file:line for provenance.

## Workflow

Project-level instructions live alongside specs in `inst/`: `2_weekly_age_pair.md` (→ notebook 7j), `3_preliminary_model_struct.md` (→ notebook 8j + forecasting framework), and the overarching `analysis_plan_heavy_tail_mean.md`/`.docx`. When given a task, **read the corresponding `inst/*.md` first** — it's the source of truth (docx wins over reference code).

- **Plan first** for non-trivial work (3+ steps or architectural decisions). Write the plan to `tasks/todo.md` with checkable items. Resolve fork-in-the-road decisions in the plan before editing files.
- **Use subagents** liberally for research, exploration, and parallel analysis — keep the main context window clean.
- **Verify before declaring done**: run the affected notebook cell(s), or spot-check distributions via `pdf` / `ccdf` / `mean`. Diff against `main` when relevant.
- **Minimal impact**: changes should touch only what's necessary. Don't refactor adjacent code unless the task requires it.
- **Capture lessons**: after any user correction, append the pattern to `tasks/lessons.md` so the same mistake isn't repeated. Read it before starting forecast-framework work — it holds hard-won reference/scoring/Turing gotchas.
- **Autonomous bug fixing**: when given a bug report with logs/errors/failing tests, just fix it — don't ask for hand-holding.
