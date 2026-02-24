# GitHub Copilot Instructions

## Project Overview

Epidemiological research project fitting heavy-tailed count distributions to social contact data from the **CoMix UK** survey (COVID-19 era). The goal is to model the degree distribution of daily contacts, decomposed into home vs. non-home contacts, and track these distributions longitudinally.

**Language**: Julia 1.11.1 (no project-local `Project.toml` — packages are installed in the global/shared environment)

## Architecture

```
distributions/   ← Custom Distributions.jl-compatible library (include via distributions/main.jl)
  poisson_mixture.jl  NegBin, PoissonLogNormal, PoissonLomax + their ZeroTrunc wrappers
  zeroinf.jl          ZeroInfDist, ZeroInfConvolutedDist, ZeroTruncConvolutedDist
  zerotrunc.jl        ZeroTruncNegBin, ZeroTruncPoissonLogNormal, ZeroTruncPoissonLomax
  kernel.jl           InterpKDEDistribution (KDE as a Distributions.jl prior)
  helper.jl           ZeroInf → ZeroTrunc conversion utilities
  plot.jl             Log-log PDF/CCDF plotting

src/
  data_setup.jl        Raw CSV → normalised DegreeDist DataFrames
  degree_dist.jl       DegreeDist struct {x::Vector{Int64}, y::Vector{Int64}} + plotting
  turing_models.jl     Turing @model definitions (ZeroInf, ZeroTrunc, convoluted, hierarchical)
  turing_utils.jl      Chain extraction, convergence checks (ESS>200, Rhat<1.1)
  comix_uk_time_series.jl  Longitudinal pipeline: per-wave 2-week windowed DegreeDist
  1j_data_explore.ipynb    Primary interactive notebook entry point

dt_surveys/      ← Raw CoMix UK CSV files + reference JSON
inst/            ← Output placeholder (currently empty)
```

## Code Style

- **Custom distributions** extend `Distributions.DiscreteUnivariateDistribution` / `ContinuousUnivariateDistribution` via `Base.@kwdef struct`; add methods with `Distributions.logpdf(d::MyType, x) = ...` dispatch.
- **Model parameters sampled in log-space**: `log_m_ga`, `log_σ_ln`, etc., with explicit `exp()` in the model body to enforce positivity for HMC/NUTS.
- **All Turing models use `Turing.@addlogprob!`** to inject manually-computed log-likelihoods rather than `x ~ dist` syntax.
- **`@memoize`** (Memoize.jl) is applied to expensive `pdf`/`ccdf` methods called repeatedly inside MCMC likelihoods.
- **DataFrame transforms** use DataFramesMeta macros: `@subset`, `@transform!`, `@rename!`, `@pipe`, `@chain`, `@byrow`.
- **Unicode identifiers** for parameters: `π0`, `μ`, `σ`, `α`, `θ`, `ξ`.
- **Relative paths from `src/`**: data reads use `"../dt_surveys/"` — always run scripts from within `src/`.

## Key Patterns

**Convoluted distribution model** (primary fitting approach):
- Home contacts: `ZeroInfDist(PoissonLogNormal(μ, σ))`
- Non-home contacts: `ZeroInfDist(PoissonLomax(α, θ))`
- All contacts: `ZeroInfConvolutedDist` (convolution of the above), fitted jointly to both marginals

**Including the distribution library**:
```julia
include("../distributions/main.jl")   # loads all 5 sub-files
```

**Standard script preamble** (from `src/`):
```julia
include("../distributions/main.jl")
include("data_setup.jl")
include("degree_dist.jl")
include("turing_models.jl")
include("turing_utils.jl")
```

**MCMC entry point**: `fit_model_with_forward_mode(model, n_sample)` — uses `NUTS()` with `Random.seed!(1236)`.

## Known Issues / Watch Out

- `Lomax` struct is defined twice in [distributions/poisson_mixture.jl](../distributions/poisson_mixture.jl) — avoid adding a third.
- `ZeroTruncConvolutedDist` appears in both `zeroinf.jl` and `zerotrunc.jl` — the second definition silently overwrites the first.
- `Distributions.cov(d)` returns the **coefficient of variation**, not covariance — non-standard naming.
- No formal test suite; verify changes by running the notebook or spot-checking with `pdf`/`ccdf` calls.

## Key Dependencies

`Distributions`, `Turing`, `MCMCChains`, `KernelDensity`, `Interpolations`, `DataFrames`, `DataFramesMeta`, `CSV`, `Plots`, `StatsBase`, `QuadGK`, `Memoize`, `LogExpFunctions`, `SpecialFunctions`, `LaTeXStrings`, `Dates`
