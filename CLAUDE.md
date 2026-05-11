# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Epidemiological research project fitting heavy-tailed count distributions to social contact survey data (primarily **CoMix UK** during COVID-19, and **Danon 2013**). The goal is to model the degree distribution of daily contacts — decomposed into home vs. non-home — jointly with per-contact duration, and to track these distributions longitudinally.

**Language**: Julia 1.11.1 (with embedded R via RCall for MGLM Dirichlet–multinomial regression).

## Environment & Commands

The canonical environment is the devcontainer (`.devcontainer/Dockerfile`, based on `quay.io/jupyter/datascience-notebook:x86_64-julia-1.11.1`). R packages `socialmixr`, `MGLM`, `broom`, `tableone`, etc. are installed there.

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
```

`JULIA_NUM_THREADS=12` and `JULIA_DEPOT_PATH=/home/jovyan/.julia:/opt/julia` are set by the devcontainer.

**No formal test suite.** Verify changes by running affected notebook cells, or by spot-checking with `pdf` / `ccdf` / `mean` calls on constructed distributions. The `precompile_script.jl` exercises the main libraries (Distributions, DataFrames, Plots, Turing) and doubles as a smoke test.

## Repository Layout

```
src/
  main_utils.jl              Canonical preamble — single include covers all sub-includes & using-statements
  utils.jl                   Plot defaults, value_counts, model_abbr / setting_pretty dicts
  data_setup.jl              Raw CSV → normalised DegreeDist DataFrames (CoMix pipeline)
  degree_dist.jl             DegreeDist struct {x, y}; contact_degrees; duration weighting (K=5)
  fit_utils.jl               Post-processing across models: mean/CI extraction, WAIC summaries
  turing_models.jl           Turing @model definitions (ZeroInf, ZeroTrunc, convoluted, hierarchical)
  turing_utils.jl            Chain extraction, convergence checks (ESS>200, Rhat<1.1)
  mglm_utils.jl              R-backed MGLM Dirichlet–multinomial fit/predict wrappers
  bnb_utils.jl               BNB-specific fitting helpers
  comix_uk_time_series.jl    Longitudinal pipeline — per-wave 2-week windowed DegreeDist
  danon_utils.jl             Danon 2013 loader + Danon-specific weighting (K=4 duration scale)
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
  <N>j_*.ipynb               Numbered notebook entry points (1j data explore, 2j duration, 3j effective degree, 4j Danon)

dt_surveys/                  Raw CoMix UK CSVs + reference JSON
dt_Leon_Danon_2013/          Danon 2013 raw data (Contact_data.csv, Person_data.csv, WKW_data.csv)
dt_intermediate/             Per-wave fitted Turing chains (JLD2, date-stamped)
inst/                        Plan/spec markdown (1_…, 2_…, 3_…, 4_…) — read these for task source-of-truth
inc2prev/                    External R package, vendored as a submodule
res/                         Output artefacts
tasks/                       Working plans (todo.md) and accumulated lessons (lessons.md)
```

## Architecture: Key Patterns

**Single preamble**: every script and notebook starts with `include("main_utils.jl")` from `src/` — this loads all `using` statements and all sub-includes. Always run scripts from `src/` (relative paths assume `cwd == src/`).

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

**Turing models** sample parameters in log-space (`log_m`, `log_σ`) and `exp()` them inside the model body to enforce positivity for HMC/NUTS. They use `Turing.@addlogprob!` to inject manually-computed log-likelihoods rather than `x ~ dist`. MCMC entry: `fit_model_with_forward_mode(model, n_sample)` with `Random.seed!(1236)`.

**Convoluted distribution** (primary CoMix model): `ZeroInfConvolutedDist` convolves a home distribution (`ZeroInfDist(PoissonLogNormal)`) with a non-home distribution (`ZeroInfDist(PoissonLomax)` or `ZeroInfDist(BNB2)`), fitted jointly to both marginals.

**DataFrame transforms** use DataFramesMeta macros throughout: `@subset`, `@transform!`, `@select`, `@rename!`, `@pipe`, `@chain`, `@byrow`. Unicode identifiers (`π0`, `μ`, `σ`, `α`, `θ`, `ξ`) are normal — don't ASCII-fy them.

**Saving chains**: `JLD2.@save` / `JLD2.@load` to `dt_intermediate/` with date-stamped filenames like `2020-04-12_zeroinf_bnb2_nhm.jld2`.

**Danon vs CoMix duration scale**: CoMix uses K=5 bins with midpoints `(2.5, 10, 37.5, 150, ∞)` min (see `_DURATION_T_MID` in `degree_dist.jl`). Danon uses K=4 bins with midpoints `(5, 20, 45, ≥60)` min and `d_max = 60` (`_DANON_T_MID` in `danon_utils.jl`). Do not forcibly remap one onto the other. The Danon loader reshapes into the project schema (`part_id_d`, `date`, `cnt_home`, `duration_multi`) so it can call existing K-parametrised code; `date` is a constant placeholder because Danon is single-shot per responder.

## Known Gotchas

- `Lomax` struct is defined twice in `src/distributions/poisson_mixture.jl` — don't add a third.
- `ZeroTruncConvolutedDist` appears in both `zeroinf.jl` and `zerotrunc.jl` — the second definition silently overwrites the first.
- `Distributions.cov(d)` in this codebase returns the **coefficient of variation**, not covariance — non-standard naming.
- `C_Duration` codes in Danon's `Contact_data.csv` are `{-1, 0, 1, 2, 3}`, offset by one from the codebook's `{1..4}`. `-1 → NA`, `0..3 → 1..4`. `C_Number ∈ {-1, 0}` rows (43 of them) are anomalous; the Danon loader drops them.

## Workflow

Project-level instructions live alongside specs in `inst/` (numbered `1_…`, `2_…`, `3_…`, `4_…`). When given a task, **read the corresponding `inst/N_*.md` first** — it's the source of truth.

- **Plan first** for non-trivial work (3+ steps or architectural decisions). Write the plan to `tasks/todo.md` with checkable items. Resolve fork-in-the-road decisions in the plan before editing files.
- **Use subagents** liberally for research, exploration, and parallel analysis — keep the main context window clean.
- **Verify before declaring done**: run the affected notebook cell(s), or spot-check distributions via `pdf` / `ccdf` / `mean`. Diff against `main` when relevant.
- **Minimal impact**: changes should touch only what's necessary. Don't refactor adjacent code unless the task requires it.
- **Capture lessons**: after any user correction, append the pattern to `tasks/lessons.md` so the same mistake isn't repeated.
- **Autonomous bug fixing**: when given a bug report with logs/errors/failing tests, just fix it — don't ask for hand-holding.
