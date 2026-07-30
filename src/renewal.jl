# renewal.jl — weekly renewal-equation transmission dynamics (port of
# CovidAgeGroupForecast stan/multi-option-contact-model.stan:104-109,263-276).
#
# I(t) = Σ_{s=1}^{smax} w(s) · N(t) · I(t-s)  = N(t) · Σ_s w(s) I(t-s)
# with a single NGM per target week applied to the smax lagged infection vectors.

"""
    gen_interval_logparams(mean_days, sd_days) -> (w_mu, w_sigma)

Log-parameters of the weekly generation-interval log-normal, moment-matched to a natural-scale
`mean_days`/`sd_days`. Follows Munday 2023 Table 1's own construction:

    w_sigma = log((sd/mean)^2 + 1)      # the LOG-VARIANCE, σ²_log  (NOT the log-SD)
    w_mu    = log(mean) − w_sigma/2     # the meanlog

⚠ `w_sigma` is the log-VARIANCE. Table 1 names the pair "log-mean and log-variance" and builds
its prior centre exactly as above, but Eq 2 then writes `F_LNorm(s, w_mu, w_sigma)` and the p. 8
prose calls `w_sigma` a "log-standard-deviation". We take the Table-1 reading — i.e.
`sdlog = sqrt(w_sigma)` — because it is self-consistent, and because it makes the prior mean
reproduce the stated 5-day mean / 5-day SD exactly. (Reading it as the sdlog would imply a
4.5-day mean and 3.5-day SD at the same prior centre.)

Times are in WEEKS (the model's time unit); `mean_days`/`sd_days` are converted here.
"""
function gen_interval_logparams(mean_days::Real, sd_days::Real)
    wm = mean_days / 7; ws = sd_days / 7
    v  = log((ws / wm)^2 + 1)          # w_sigma: LOG-VARIANCE
    return (log(wm) - v / 2, v)        # (w_mu, w_sigma)
end

"""
    gi_moments_days(w_mu, w_sigma) -> (; mean_days, sd_days)

Natural-scale mean and SD, in DAYS, of the generation-interval log-normal — the inverse of
`gen_interval_logparams`, so `w_sigma` is the LOG-VARIANCE (⇒ `sdlog = √w_sigma`; see there).

    mean = exp(w_mu + w_sigma/2),   sd = mean·√(exp(w_sigma) − 1)      [weeks, ×7 ⇒ days]

Broadcasts, so it maps the per-draw `w_mu`/`w_sigma` vectors of a Stage-2 pooled result straight
to interpretable days for the 9j GI panel.
"""
function gi_moments_days(w_mu, w_sigma)
    m_wk  = @. exp(w_mu + w_sigma / 2)
    sd_wk = @. m_wk * sqrt(expm1(w_sigma))            # expm1 for accuracy at small w_sigma
    return (; mean_days = 7 .* m_wk, sd_days = 7 .* sd_wk)
end

"""
    gen_interval_pmf_log(meanlog, logvar; smax=4)

Discretised weekly generation-interval PMF, Munday 2023 Eq 2 (`inst/pcbi.1011453.pdf`, p. 6):

    w(s) = (F(s) − F(s−1)) / F(smax),   s = 1..smax

with `F` the CDF of `LogNormal(meanlog, sqrt(logvar))`. Because `F(0) = 0`, the denominator
equals the numerator sum exactly, so `w` is a proper right-truncated PMF with no further
normalisation step (dividing by `F(smax)` and by `sum(w)` are the same operation here).

**AD-generic on purpose** — no `Float64` annotations, no preallocated concrete buffers — because
`meanlog`/`logvar` are sampled latents in `model_transmission` and arrive as tracked reals under
ReverseDiff. Keep it that way.
"""
function gen_interval_pmf_log(meanlog::Real, logvar::Real; smax::Int = 4)
    d  = LogNormal(meanlog, sqrt(logvar))
    Fs = [cdf(d, float(s)) for s in 1:smax]                       # F(0) = 0 exactly
    w  = [s == 1 ? Fs[1] : Fs[s] - Fs[s - 1] for s in 1:smax]
    return w ./ Fs[end]
end

"""
    gen_interval_pmf(mean_days, sd_days; smax=4)

Discretised weekly generation-interval PMF for a FIXED natural-scale mean/SD in days — the
moment-matched wrapper around `gen_interval_logparams` + `gen_interval_pmf_log`.

Since 2026-07-30 the generation interval is **estimated** inside `model_transmission`
(`w_mu`, `w_sigma` latents, §3.1), so this is no longer the fitting path; it survives for the
prior centre, for defaults, and for diagnostics that want the prior-mean PMF. At
`(mean_days, sd_days) = (5, 5)` it returns the same `w` the model produced before the GI was
estimated, so the estimated-GI model nests the fixed-GI one at its prior mean.
"""
gen_interval_pmf(mean_days::Real, sd_days::Real; smax::Int = 4) =
    gen_interval_pmf_log(gen_interval_logparams(mean_days, sd_days)...; smax = smax)

"""
    renewal_next(N, Imat, t, w)

Predicted next-generation infection vector for week `t`:
`N · Σ_{s=1}^{length(w)} w[s]·Imat[:, t-s]`. `Imat` is `A × T`; requires
`t > length(w)`. AD-friendly (no in-place mutation of tracked arrays).
"""
function renewal_next(N::AbstractMatrix, Imat::AbstractMatrix, t::Int, w::AbstractVector)
    acc = w[1] .* Imat[:, t - 1]
    for s in 2:length(w)
        acc = acc .+ w[s] .* Imat[:, t - s]
    end
    return N * acc
end

"""
    forecast_forward(N_origin, I_seed, w, H)

Deterministic `A × H` forecast, iterating the renewal `H` weeks with the NGM frozen
at the origin (reference stan:309-314). `I_seed` is `A × K` infection history whose
last column is the origin week; predictions are appended and reused as lags.

⚠ Currently UNUSED — `two_stage_forecast` (`joint_model.jl`) supersedes it with its own
inline iteration. If you revive it, note that since 2026-07-30 the generation interval is a
per-draw quantity (§3.1): `w` must come from that draw's `(w_mu, w_sigma)` via
`gen_interval_pmf_log`, NOT from `cfg.gen_mean_days`/`gen_sd_days`, which are only the prior
centre.
"""
function forecast_forward(N_origin::AbstractMatrix, I_seed::AbstractMatrix,
                          w::AbstractVector, H::Int)
    A = size(N_origin, 1); smax = length(w)
    hist = collect(float.(I_seed))
    out = zeros(A, H)
    for f in 1:H
        Tcur = size(hist, 2)
        acc = zeros(A)
        for s in 1:smax
            acc .+= w[s] .* hist[:, Tcur - s + 1]
        end
        pred = N_origin * acc
        out[:, f] = pred
        hist = hcat(hist, pred)
    end
    return out
end
