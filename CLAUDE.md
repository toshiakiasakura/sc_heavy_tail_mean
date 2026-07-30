# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Epidemiological research project fitting heavy-tailed count distributions to social contact survey data (**CoMix UK** during COVID-19). The goal is to model the degree distribution of daily contacts — decomposed into home vs. non-home — jointly with per-contact duration, and to track these distributions longitudinally. (A **Danon 2013** strand is now *legacy*: its raw data `dt_Leon_Danon_2013/` was removed as no longer needed, so `danon_utils.jl` and `4j_Danon_degree_duration.ipynb` are orphaned — code kept, not runnable.)

A second, newer strand (`inst/3_preliminary_model_struct.md`, `src/framework.jl` and friends) **extends** the age-stratified renewal / next-generation-matrix (NGM) forecasting model of Munday et al. 2023 (`inst/pcbi.1011453.pdf`; the upstream R+Stan reference `CovidAgeGroupForecast/` is **no longer vendored in-tree** — port provenance survives only in source comments and `tasks/lessons.md`). Here the age-pair **contact-degree distributions** feed the NGM directly (mean or neighbourhood/excess degree), driving a weekly renewal-equation forecast of age-stratified infections. Where the analysis plan (`inst/analysis_plan_heavy_tail_mean.docx`, converted to `.md`) differs from the reference code, **the docx wins**.

**Language**: Julia 1.11.1 (with embedded R via RCall for MGLM Dirichlet–multinomial regression and `scoringutils` WIS scoring).

## Environment & Commands

The canonical environment is the devcontainer (`.devcontainer/Dockerfile`, based on `quay.io/jupyter/datascience-notebook:x86_64-julia-1.11.1`). R packages `socialmixr`, `MGLM`, `broom`, `tableone`, `scoringutils`, etc. are installed there.

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
  10j_viz_utils.jl           Model diagnostics — μ reconstruction from Stage-1 chains (stays on the original four combos)

  <N>j_*.ipynb               Numbered notebook entry points:
                               1j data explore · 2j duration · 3j effective degree · 4j Danon [legacy] ·
                               5j group contacts · 6j fitting dist · 7j weekly age-pair · 8j forecast fitting ·
                               9j forecast diagnostics/scoring · 10j model diagnostics

dt_comix_no_public/          Primary CoMix-UK data — contacts_uk.arrow / part_uk.arrow (+ csv/qs mirrors). gitignored (local only)
dt_surveys/                  [ABSENT] Not in tree; legacy loaders in data_setup.jl/utils.jl still reference ../dt_surveys/ paths (e.g. 2014 Read China)
dt_intermediate/             Fitted Turing chains (JLD2): date-stamped dist fits + 8j_chn_<degree>_<ngm>_<contacts>_<origin>_h<h>.jld2 forecast chains; cached df_dds*.csv
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

**Fit path**: `fit_stage1` runs `Pathfinder.pathfinder` (or NUTS when `cfg.stage1_use_nuts=true`, initialised from the Pathfinder mean); Stage 2 is always Pathfinder (100 cheap fits per Stage-1 draw). Seed `cfg.seed = 1236`. Drivers: `prefit_stage1!` then `prefit_stage2!` (or `prefit_two_stage!`); forecast `two_stage_forecast`. Artefacts: `8j_s1_<degree>_<contacts>_<origin>_h<h>.jld2` (Stage-1 chain, key `result`, no ngm token) and `8j_s2_<degree>_<ngm>_<contacts>_<origin>_h<h>.jld2` (Stage-2 pooled, key `pooled`).

**Transmission core** (Julia ports of the Munday-2023 Stan model `multi-option-contact-model.stan`; the `CovidAgeGroupForecast/` reference tree is no longer vendored — `renewal.jl`/`infection_data.jl` comments cite the original file:line): weekly renewal `I(t) = Σ_{s=1}^{smax} w(s)·N(t)·I(t−s)`, `smax=4`; `N_ab = γ_SAR · full_susceptibility_a · C*_ab · inf_b` with leaky antibody protection `full_susceptibility_a = susc_a·(1+(F−1)·A_a(t))`. `γ_SAR` (`build_ngm(…; gamma_sar=…)`) is the **per-contact secondary attack rate**; `C*` is the builder's `C0` used **directly** and **un-normalised** (the `-gnorm` C*/S̄ decoupling was reverted 2026-07-12), and reciprocity lives in the contact-mean estimation, so there is **no** post-hoc symmetrisation (see Gotchas). Forecasts freeze the NGM at the origin week and iterate the renewal forward.

**Weekly grid**: inc2prev-aligned, Sunday-start, anchored at `WEEK_ANCHOR = 2021-03-21`, Wednesday mid-date labels. A `WeeklyWindow(origin; n_fit=8, smax=4, horizons=1:4)` yields `lag_weeks` (renewal history) ++ `fit_weeks` = `all_weeks`, plus horizon `forecast_weeks`. Age grid = the 7 England CIS `age_school` bins (`2-10 … 70+`) from `inc2prev/data-processed/populations.csv`.

**Scoring** (`scoring.jl`): primary metric is **WIS** via R `scoringutils` v2 on quantile-format forecasts (`to_quantile_long` → `score_wis`); native sample CRPS is a cheap cross-check. A **log score** (`to_sample_long` → `score_logs`, `inst/6`) is reported alongside — `scoringutils` defines it only for the **sample** class, so it is a second R path over the raw draws, scored one origin at a time, on both scales. Don't confuse it with the existing "log-scale WIS", which is WIS after a log *transform*. Outputs go to `res/8j_*` (scores + `8j_logscore_*`).

Contacts are **per-week** by default (`constant_contacts=false`, the notebook setting): the age-pair mean is estimated separately each window week by a separable spatio-temporal GP (per-week level `c_t` + matrix-normal field, shared `ρ_diag/ρ_gap/ρ_time`, `η`, `σ_c`; dispersion per week × block), so the renewal `N(t)` varies through that week's `C*_t` as well as antibody. `constant_contacts=true` recovers the pooled one-`C*`-per-window preliminary. `model_degree` branches on the flag but returns per-week moments `(; K1, K2, G)` (length `Tn`) either way; the forecast NGM uses the origin-week slice via `Cstar_end[m]` in the Stage-2 pooled result.

## Known Gotchas

- `Lomax` struct is defined twice in `src/distributions/poisson_mixture.jl` — don't add a third.
- `ZeroTruncConvolutedDist` appears in both `zeroinf.jl` and `zerotrunc.jl` — the second definition silently overwrites the first.
- `Distributions.cov(d)` in this codebase returns the **coefficient of variation**, not covariance — non-standard naming.
- *(Legacy Danon)* `C_Duration` codes in Danon's `Contact_data.csv` are `{-1, 0, 1, 2, 3}`, offset by one from the codebook's `{1..4}` (`-1 → NA`, `0..3 → 1..4`); `C_Number ∈ {-1, 0}` rows (43) are anomalous and dropped. Data now removed — kept only as reference if the strand is revived.
- **inc2prev `estimates_age_ab.csv` schema**: `name=="infections"` is a **daily per-capita proportion** — multiply by `population` and 7-day-sum for weekly counts. `name=="gen_dab"` is antibody prevalence ∈[0,1] but its `date` column is **empty** — dates are reconstructed from `t_index` via the `infections` rows' `t_index→date` map (see `infection_data.jl`).
- **Turing + Weibull/exp over/underflow**: `κ = exp(log_kappa)` can go non-finite on aggressive Pathfinder/optim steps → `DomainError: Weibull α>0`. Log-parameters (`log_kappa`, `log_k`, `log μ` in `model_degree`; `log_gamma_sar` in `model_transmission`) are soft-**clamped** (`_softclamp`); the mode stays interior so gradients are unaffected. Keep those clamps when editing either model. **`_softclamp` must stay the Inf-safe nested form** `lo + softplus((hi − softplus(hi − x)) − lo)` — the naive `x − softplus(x−hi) + softplus(lo−x)` returns **NaN** at `x=±Inf` (`Inf − Inf`), so when a raw log-latent (`log_kappa`/`log_k`) or the GP field overflows to Inf, `κ=exp(softclamp(Inf))` becomes NaN and the Weibull throws (aborting the whole fit). This bit the **neighbourhood** NGM hardest (its `gamma(1+2/κ)` second moment pushes the optimiser to extremes); see `tasks/lessons.md` 2026-07-11.
- **The forecast already uses FUTURE contacts.** For horizon `h`, `fit_or_load_stage2` fits Stage 1 on `WeeklyWindow(origin + 7h)` and the forecast NGM is frozen at `Cstar_end[m]` = the **origin+h** contact matrix, while infections/antibody stay at the origin. This is deliberate (the "contact-updated iterate", Munday's design), not a leak to fix — and it is why `inst/6`'s "the no-interaction model is allowed to use the future mean contacts" needed no forecast-path change. The **null** model is the sole exception: its `c̄` is computed from the *origin* window's focal weeks and reused for every horizon.
- **Log score ≠ log-scale WIS.** `scoringutils` computes `log_score` only for the **sample** forecast class, never for quantile forecasts — so it cannot be added as a flag to `score_wis`. `score_logs` is a separate R path over the raw draws. Its KDE cannot consume the `±Inf` draws `two_stage_forecast` deliberately keeps, and `log_shift` returns `NaN` on the negative draws a Gaussian fan contains; both are sanitised, **counted, and printed** rather than silently dropped.
- **Per-model panel figures must derive their layout** (`panel_grid` in `9j_viz_utils.jl`). `plot_ratio` / `plot_ratio_bins` / `plot_lengthscales` were hard-coded `layout = (2, 2)` while there were exactly four models and silently lose panels at six.
- **scoringutils v2 quantile levels**: round the quantile levels before handing them to R — fp drift from `collect(0.05:0.05:0.95)` makes `0.25`/`0.75` not match exactly, so `interval_coverage_50` silently fails while the 90% endpoints still work.
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
