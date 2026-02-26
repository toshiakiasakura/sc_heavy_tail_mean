# GitHub Copilot Instructions

## Project Overview

Epidemiological research project fitting heavy-tailed count distributions to social contact data from the **CoMix UK** survey (COVID-19 era). The goal is to model the degree distribution of daily contacts, decomposed into home vs. non-home contacts, and track these distributions longitudinally.

**Language**: Julia 1.11.1 — dependencies declared in `Project.toml` (use `Pkg.instantiate()` to install).

## Architecture

```
src/
  distributions/       ← Custom Distributions.jl-compatible library
    main.jl              entry point — include this file
    poisson_mixture.jl   NegBin, PoissonLogNormal, PoissonLomax + ZeroTrunc wrappers
    bnb.jl               BNB (α,β,r) and BNB2 (m,v,η) — Beta Negative Binomial
    zeroinf.jl           ZeroInfDist, ZeroInfConvolutedDist, ZeroTruncConvolutedDist
    zerotrunc.jl         ZeroTruncNegBin, ZeroTruncPoissonLogNormal, ZeroTruncPoissonLomax
    kernel.jl            InterpKDEDistribution (KDE as a Distributions.jl prior)
    helper.jl            ZeroInf → ZeroTrunc conversion utilities
    plot.jl              Log-log PDF/CCDF plotting
  main_utils.jl        Canonical preamble: all using-statements + all includes
  utils.jl             Misc helpers: plot defaults, value_counts, model_abbr dict
  data_setup.jl        Raw CSV → normalised DegreeDist DataFrames
  degree_dist.jl       DegreeDist struct {x::Vector{Int64}, y::Vector{Int64}} + plotting
  fit_utils.jl         Post-processing: summarise results, extract means/CIs across models
  turing_models.jl     Turing @model definitions (ZeroInf, ZeroTrunc, convoluted, hierarchical)
  turing_utils.jl      Chain extraction, convergence checks (ESS>200, Rhat<1.1)
  comix_uk_time_series.jl  Longitudinal pipeline: per-wave 2-week windowed DegreeDist
  1j_data_explore.ipynb    Primary interactive notebook entry point

dt_surveys/      ← Raw CoMix UK CSV files + reference JSON
dt_intermediate/ ← Per-wave fitted chain JLD2 files (date-stamped)
inst/            ← Output placeholder
```

## Code Style

- **Custom distributions** extend `Distributions.DiscreteUnivariateDistribution` via `Base.@kwdef struct`; dispatch methods as `Distributions.logpdf(d::MyType, k::Int64) = ...`.
- **`@unpack`** (Parameters.jl) used to destructure distribution fields inside `logpdf`: `@unpack α, β, r = d`.
- **Model parameters sampled in log-space**: `log_m`, `log_σ`, etc., with explicit `exp()` in the model body to enforce positivity for HMC/NUTS.
- **All Turing models use `Turing.@addlogprob!`** to inject manually-computed log-likelihoods rather than `x ~ dist` syntax.
- **`@memoize`** (Memoization.jl) on expensive `pdf`/`ccdf` methods called inside MCMC likelihoods.
- **DataFrame transforms** use DataFramesMeta macros: `@subset`, `@transform!`, `@rename!`, `@pipe`, `@chain`, `@byrow`.
- **Unicode identifiers** for parameters: `π0`, `μ`, `σ`, `α`, `θ`, `ξ`.
- **Relative paths from `src/`**: data reads use `"../dt_surveys/"` — always run scripts from within `src/`.

## Key Patterns

**Convoluted distribution model** (primary fitting approach):
- Home contacts: `ZeroInfDist(PoissonLogNormal(μ, σ))`
- Non-home contacts: `ZeroInfDist(PoissonLomax(α, θ))` or `ZeroInfDist(BNB2(m, v, η))`
- All contacts: `ZeroInfConvolutedDist` (convolution of the above), fitted jointly to both marginals

**Standard script preamble** (from `src/`) — single include covers everything:
```julia
include("main_utils.jl")   # loads all using-statements + all sub-includes
```

**MCMC entry point**: `fit_model_with_forward_mode(model, n_sample)` — uses `NUTS()` with `Random.seed!(1236)`.

**Saving/loading chains**: `JLD2.@save`/`JLD2.@load` to `dt_intermediate/` with date-stamped filenames.

## Known Issues / Watch Out

- `Lomax` struct is defined twice in [src/distributions/poisson_mixture.jl](../src/distributions/poisson_mixture.jl) — avoid adding a third.
- `ZeroTruncConvolutedDist` appears in both `zeroinf.jl` and `zerotrunc.jl` — the second definition silently overwrites the first.
- `Distributions.cov(d)` returns the **coefficient of variation**, not covariance — non-standard naming.
- No formal test suite; verify changes by running the notebook or spot-checking with `pdf`/`ccdf` calls.

## Build / Environment

```julia
# Install all dependencies (from project root)
using Pkg; Pkg.instantiate()

# Scripts must be run from src/ — relative paths assume this working directory
```

## Agent Workflow

- Plan before implementing any non-trivial change (3+ steps).
- Verify correctness by running affected notebook cells or `pdf`/`ccdf` spot-checks.
- Keep changes minimal and targeted — avoid touching unrelated files.
- After any user correction, note the lesson to avoid repeating the mistake.

## Key Dependencies

`Distributions`, `Turing`, `DynamicPPL`, `Pathfinder`, `MCMCChains`, `KernelDensity`, `Interpolations`, `DataFrames`, `DataFramesMeta`, `CSV`, `Plots`, `StatsPlots`, `StatsBase`, `QuadGK`, `Memoization`, `Parameters`, `LogExpFunctions`, `SpecialFunctions`, `LaTeXStrings`, `JLD2`, `Dates`, `GLM`, `XLSX`, `Pipe`
