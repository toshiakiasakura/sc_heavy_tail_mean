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
  ngm.jl                     NGM builders (MeanNGM / NeighbourhoodDegreeNGM): C* = builder C0 used directly, no post-hoc balancing + leaky susceptibility
  joint_model.jl             The one joint Turing @model (degree + infection likelihoods); Pathfinder→NUTS fit; posterior forecast
  scoring.jl                 WIS via R scoringutils (RCall, quantile format) + native sample CRPS cross-check
  8j_viz_utils.jl            Read-only 8j viz — reconstructs susc/inf/ρ from cached joint-model chains (no model rebuild)

  <N>j_*.ipynb               Numbered notebook entry points:
                               1j data explore · 2j duration · 3j effective degree · 4j Danon [legacy] ·
                               5j group contacts · 6j fitting dist · 7j weekly age-pair · 8j preliminary forecast

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

- **Axis 1 — `ContactDegreeModel`** (how the age-pair contact degree is modelled): `NegBinAgePair` (unweighted integer counts) or `HurdleWeibullAgePair` (duration-weighted, hurdle). `is_weighted(dm)` branches the likelihood.
- **Axis 2 — `NGMBuilder`** (how the NGM's per-capita contact `C0` is formed from the degree distribution's raw moments `⟨k⟩, ⟨k²⟩`): `MeanNGM` (`C0 = ⟨k⟩`) or `NeighbourhoodDegreeNGM` (`C0 = ⟨k²⟩/⟨k⟩ = m(1+CV²)`, the excess/size-biased degree).

The "**four ways**" of the preliminary analysis is the 2×2 grid of these two axes.

**One joint Turing model** (`model_joint` in `joint_model.jl`) serves all four combos — the contact-degree likelihood *and* the infection renewal likelihood live in the same `@model`. `nb::NGMBuilder` adds no parameters (pure dispatch, passed as a fixed arg); `dm::ContactDegreeModel` selects the degree likelihood + latent shape/dispersion via `if is_weighted(dm)`. Both are fixed model arguments so the parameter space is well-defined per fit.

**Fit path**: `fit_joint` runs `Pathfinder.pathfinder(model)` for a parsimonious init, then initialises `NUTS` from the Pathfinder posterior mean (falls back to Pathfinder draws if NUTS fails). Seed is `cfg.seed = 1236`.

**Transmission core** (Julia ports of the Munday-2023 Stan model `multi-option-contact-model.stan`; the `CovidAgeGroupForecast/` reference tree is no longer vendored — `renewal.jl`/`infection_data.jl` comments cite the original file:line): weekly renewal `I(t) = Σ_{s=1}^{smax} w(s)·N(t)·I(t−s)`, `smax=4`; `N_ab = full_susceptibility_a · C*_ab · inf_b` with leaky antibody protection `full_susceptibility_a = susc_a·(1+(F−1)·A_a(t))`. `C*` is the builder's `C0` used **directly** — reciprocity lives in the contact-mean estimation, so there is **no** post-hoc symmetrisation (see Gotchas). Forecasts freeze the NGM at the origin week and iterate the renewal forward.

**Weekly grid**: inc2prev-aligned, Sunday-start, anchored at `WEEK_ANCHOR = 2021-03-21`, Wednesday mid-date labels. A `WeeklyWindow(origin; n_fit=8, smax=4, horizons=1:4)` yields `lag_weeks` (renewal history) ++ `fit_weeks` = `all_weeks`, plus horizon `forecast_weeks`. Age grid = the 7 England CIS `age_school` bins (`2-10 … 70+`) from `inc2prev/data-processed/populations.csv`.

**Scoring** (`scoring.jl`): primary metric is **WIS** via R `scoringutils` v2 on quantile-format forecasts (`to_quantile_long` → `score_wis`); native sample CRPS is a cheap cross-check. Outputs go to `res/8j_*`.

Contacts are **per-week** by default (`constant_contacts=false`, the notebook setting): the age-pair mean is estimated separately each window week as an independent GP draw sharing the length-scale `ρ` and marginal scale `η` (per-week level `c_t` + field `z_t`; dispersion per week × block) — **no temporal smoothing** — so the renewal `N(t)` varies through that week's `C*_t` as well as antibody. `constant_contacts=true` recovers the pooled one-`C*`-per-window preliminary (a *temporally-correlated* RW1/Matérn mean is the remaining swap-in seam). `model_joint` branches on the flag but returns `Cstar` as a length-`Tn` vector either way; the forecast NGM uses the origin-week slice `q.Cstar[end]`.

## Known Gotchas

- `Lomax` struct is defined twice in `src/distributions/poisson_mixture.jl` — don't add a third.
- `ZeroTruncConvolutedDist` appears in both `zeroinf.jl` and `zerotrunc.jl` — the second definition silently overwrites the first.
- `Distributions.cov(d)` in this codebase returns the **coefficient of variation**, not covariance — non-standard naming.
- *(Legacy Danon)* `C_Duration` codes in Danon's `Contact_data.csv` are `{-1, 0, 1, 2, 3}`, offset by one from the codebook's `{1..4}` (`-1 → NA`, `0..3 → 1..4`); `C_Number ∈ {-1, 0}` rows (43) are anomalous and dropped. Data now removed — kept only as reference if the strand is revived.
- **inc2prev `estimates_age_ab.csv` schema**: `name=="infections"` is a **daily per-capita proportion** — multiply by `population` and 7-day-sum for weekly counts. `name=="gen_dab"` is antibody prevalence ∈[0,1] but its `date` column is **empty** — dates are reconstructed from `t_index` via the `infections` rows' `t_index→date` map (see `infection_data.jl`).
- **Turing + Weibull/exp underflow**: `κ = exp(log_kappa)` can underflow to `0.0` on aggressive Pathfinder/optim steps → `DomainError: Weibull α>0`. Log-parameters (`log_kappa`, `log_k`, `log μ`) are **clamped** inside `model_joint`; the mode stays interior so gradients are unaffected. Keep those clamps when editing the model.
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
