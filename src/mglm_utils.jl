######################################################################
###### Wrapper for R's MGLM Dirichlet-multinomial regression  ########
######################################################################
#
# Talks to R via `RCall`. The R-side parameterisation is
#
#     α_ik = exp(x_i' β_k),    μ_ik = α_ik / Σ_j α_ij,
#
# which differs from the softmax / log-α₀ split used by Turing's
# `model_dm_logdeg` but yields equivalent fitted proportions μ.

using RCall
using DataFrames
using Distributions

"""
Fit a Dirichlet-multinomial regression with R's `MGLM::MGLMreg`.

Inputs:
- `X::AbstractMatrix{<:Real}` — design matrix `[1 log(n)]` from
  `prepare_dm_inputs`. The leading all-ones intercept column is dropped
  before being passed to R; `MGLMreg` adds its own intercept from the formula.
- `Y::AbstractMatrix{<:Integer}` — count matrix, size `N × K`.
- `intercept_only::Bool` — if `true`, fit the null `Y ~ 1` (no `log(n)` term).
  Used as the AIC/BIC comparator.

Returns a NamedTuple:
- `β`        — coefficient matrix `P × K` (rows = predictors, cols = categories).
- `SE`       — standard errors, same shape as `β`.
- `logL`     — log-likelihood at the MLE.
- `AIC`, `BIC` — information criteria.
- `iter`     — Newton-Raphson iteration count.
- `wald`     — DataFrame of per-predictor Wald χ² and p-values (`m@test`).
- `fitted`   — fitted proportions on the training rows (`N × K`).
- `r_object` — the raw `MGLMreg` S4 object, useful for `summary()`, `predict()`,
  or `saveRDS`.
"""
function fit_mglm_dm(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Integer};
                     intercept_only::Bool = false)
    R"suppressPackageStartupMessages(library(MGLM))"

    Y_int = Matrix{Int}(Y)
    @rput Y_int

    if intercept_only
        R"""
        d <- data.frame(.dummy = rep(1, nrow(Y_int)))
        m <- suppressWarnings(MGLMreg(Y_int ~ 1, data = d, dist = "DM"))
        """
    else
        # Drop the intercept column. `prepare_dm_inputs` always produces
        # X = [1 log(n)] with size (N, 2); fall back to using all columns
        # past the first if a caller supplies extra predictors later.
        size(X, 2) >= 2 || error("X must have at least 2 columns (intercept + ≥1 predictor)")
        Xpred = Matrix{Float64}(X[:, 2:end])
        log_n = vec(Xpred[:, 1])
        @rput log_n
        R"""
        d <- data.frame(log_n = log_n)
        m <- suppressWarnings(MGLMreg(Y_int ~ log_n, data = d, dist = "DM"))
        """
    end

    β        = rcopy(R"as.matrix(m@coefficients)")
    SE       = rcopy(R"as.matrix(m@SE)")
    logL     = rcopy(R"as.numeric(m@logL)")
    AIC_val  = rcopy(R"as.numeric(m@AIC)")
    BIC_val  = rcopy(R"as.numeric(m@BIC)")
    iter     = rcopy(R"as.integer(m@iter)")
    wald     = rcopy(R"as.data.frame(m@test)")
    fitted   = rcopy(R"as.matrix(m@fitted)")
    r_object = R"m"

    return (; β = β, SE = SE, logL = logL, AIC = AIC_val, BIC = BIC_val,
              iter = iter, wald = wald, fitted = fitted, r_object = r_object)
end

"""
Compute fitted DM category proportions μ from MGLM coefficients on a new
design matrix `X`.

Inputs:
- `β::AbstractMatrix` — `P × K`, as returned by `fit_mglm_dm`.
- `X::AbstractMatrix` — `N × P`. For the null fit (`intercept_only = true`),
  pass `reshape(ones(N), N, 1)`.

Returns `μ::Matrix{Float64}` of size `N × K` with rows summing to 1.
"""
function mglm_dm_proportions(β::AbstractMatrix, X::AbstractMatrix)
    size(X, 2) == size(β, 1) ||
        error("X has $(size(X, 2)) columns but β has $(size(β, 1)) rows; they must match")
    log_α = X * β                              # N × K
    M     = maximum(log_α; dims = 2)           # row-wise stabiliser
    α     = exp.(log_α .- M)
    return α ./ sum(α; dims = 2)
end

"""
Compare a full MGLM fit against an intercept-only null fit.

Returns a NamedTuple `(ΔAIC, ΔBIC, prefer)` where `Δ = null - full` (positive
favours the full model) and `prefer ∈ ("full", "null")` is decided by ΔAIC.
"""
function mglm_compare(full::NamedTuple, null::NamedTuple)
    ΔAIC = null.AIC - full.AIC
    ΔBIC = null.BIC - full.BIC
    prefer = ΔAIC > 0 ? "full" : "null"
    return (; ΔAIC = ΔAIC, ΔBIC = ΔBIC, prefer = prefer)
end

"""
Print a compact diagnostic summary of an MGLM DM fit (replacement for
`summarize_dm_fit` in the MGLM path).
"""
function mglm_dm_show(fit::NamedTuple; io::IO = stdout)
    println(io, "MGLM Dirichlet-multinomial fit")
    println(io, "  iterations: ", fit.iter,
              "    logL: ",  round(fit.logL; digits = 3),
              "    AIC: ",   round(fit.AIC;  digits = 3),
              "    BIC: ",   round(fit.BIC;  digits = 3))
    println(io, "  coefficients (P × K):")
    show(io, "text/plain", fit.β);  println(io)
    println(io, "  standard errors:")
    show(io, "text/plain", fit.SE); println(io)
    println(io, "  per-predictor Wald test:")
    show(io, "text/plain", fit.wald); println(io)
    return nothing
end

"""
Save the underlying R `MGLMreg` object to `path` as an RDS file. Use this to
persist fits for later inspection in R (e.g. `readRDS(path)`).
"""
function save_mglm_fit(fit::NamedTuple, path::AbstractString)
    R"saveRDS($(fit.r_object), $path)"
    return path
end

"""
Predict, per degree `n`, the DM marginal distribution implied by an MGLM fit.

For each `n` in `n_grid`, the fitted concentration is α(n) = exp(X(n) · β),
and the marginal of `Y_k` is `BetaBinomial(n, α_k, α₀ − α_k)` with

    E[Y_k/n]   = μ_k = α_k / α₀
    Var[Y_k/n] = μ_k (1 − μ_k) (n + α₀) / (n (1 + α₀))

The band is computed analytically as `μ_k ± z · σ_k(n)` (Gaussian
approximation), then clamped to `[0, 1]`. `z = quantile(Normal(),
1 − α_level/2)` (≈ 1.96 at the 5% level).

Returns:

- `μ::Matrix{Float64}` `(length(n_grid) × K)` — predicted mean proportions.
- `lower::Matrix{Float64}` — lower band edge (clamped to `[0, 1]`).
- `upper::Matrix{Float64}` — upper band edge (clamped to `[0, 1]`).

`fit.β` decides the design: shape `(1, K)` ⇒ intercept-only, shape `(2, K)` ⇒
intercept + `log(n)`. Other shapes raise an error.
"""
function mglm_dm_predict(fit::NamedTuple, n_grid::AbstractVector{<:Integer};
                         α_level::Real = 0.05)
    P = size(fit.β, 1); K = size(fit.β, 2)
    Xg = if P == 1
        reshape(ones(length(n_grid)), :, 1)
    elseif P == 2
        hcat(ones(length(n_grid)), log.(collect(n_grid)))
    else
        error("expected β with 1 or 2 rows; got $(P)")
    end
    α  = exp.(Xg * fit.β)                              # L × K
    α0 = reshape(vec(sum(α; dims = 2)), :, 1)          # L × 1
    μ  = α ./ α0                                       # L × K
    n  = reshape(Float64.(collect(n_grid)), :, 1)      # L × 1
    var = μ .* (1 .- μ) .* ((n .+ α0) ./ (n .* (1 .+ α0)))
    σ   = sqrt.(var)
    z   = quantile(Normal(), 1 - α_level / 2)
    lower = clamp.(μ .- z .* σ, 0.0, 1.0)
    upper = clamp.(μ .+ z .* σ, 0.0, 1.0)
    return (; μ = μ, lower = lower, upper = upper)
end
