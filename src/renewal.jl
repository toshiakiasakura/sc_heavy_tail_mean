# renewal.jl — weekly renewal-equation transmission dynamics (port of
# CovidAgeGroupForecast stan/multi-option-contact-model.stan:104-109,263-276).
#
# I(t) = Σ_{s=1}^{smax} w(s) · N(t) · I(t-s)  = N(t) · Σ_s w(s) I(t-s)
# with a single NGM per target week applied to the smax lagged infection vectors.

"""
    gen_interval_pmf(mean_days, sd_days; smax=4)

Discretised weekly generation-interval PMF from a log-normal, renormalised over
`1:smax` (reference stan:104-109). `mean_days`/`sd_days` are converted to weeks;
the log-normal is parameterised correctly (`sdlog = sqrt(log(CV²+1))`) — note the
reference passes the log-variance where sd is expected; here we use the proper form.
Returns `w` with `w[s]` the weight on lag `s` weeks.
"""
function gen_interval_pmf(mean_days::Real, sd_days::Real; smax::Int = 4)
    wm = mean_days / 7; ws = sd_days / 7
    sdlog2 = log((ws / wm)^2 + 1)
    d = LogNormal(log(wm) - sdlog2 / 2, sqrt(sdlog2))
    w = [cdf(d, float(s)) - cdf(d, float(s - 1)) for s in 1:smax]
    return w ./ sum(w)
end

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
