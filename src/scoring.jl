# scoring.jl — forecast scoring. Primary: WIS (weighted interval score) via the R
# `scoringutils` package (RCall), on quantile-format forecasts. Secondary: a native
# sample CRPS as a cheap cross-check.

"""
    to_quantile_long(fc, truth, model_label, win, cfg, age_labels)

Long quantile-format table for scoringutils: one row per
(age × horizon × quantile_level). `fc` is `A×H×D` posterior-predictive draws;
`truth` is `A×H` realized weekly infections.
"""
function to_quantile_long(fc::AbstractArray{<:Real,3}, truth::AbstractMatrix,
                          model_label::AbstractString, win::WeeklyWindow,
                          cfg::FrameworkConfig, age_labels::Vector{String})
    A, H, _ = size(fc)
    qs = cfg.quantiles
    rows = DataFrame(model = String[], forecast_date = Date[], target_date = Date[],
                     horizon = Int[], age_group = String[], quantile_level = Float64[],
                     predicted = Float64[], observed = Float64[])
    for a in 1:A, h in 1:H
        draws = collect(fc[a, h, :])
        obs = float(truth[a, h])
        for q in qs
            # round the quantile level so scoringutils finds exact 0.25/0.75/0.05/0.95
            # (fp drift from 0.05:0.05:0.95 otherwise breaks interval coverage).
            push!(rows, (model_label, win.origin, win.forecast_weeks[h], h,
                         age_labels[a], round(q; digits = 2), quantile(draws, q), obs))
        end
    end
    return rows
end

"""
    score_wis(df_quant)

Run R `scoringutils` on a quantile-format table. Scores are computed on **both the
natural and the log scale** (`transform_forecasts(fun=log_shift, offset=1)`; inst/1e) —
each returned frame carries a `scale ∈ {"natural","log"}` column, and WIS is
aggregated **by horizon** across all forecast dates/origins (`by_model_h`), so the
headline metric is the log-scale, by-horizon WIS (`scale=="log"`). Returns
`(; by_model, by_model_h, by_model_dt, by_model_dt_h, by_model_h_age, by_model_dt_h_age)`
with WIS + components, bias, and interval coverage; `by_model_h_age` adds the age-stratified
(model × horizon × age_group) WIS used by the 9j Fig-3-style diagnostics, `by_model_dt_h_age`
is the same at the FULL forecast unit (+ `forecast_date`, 21 168 rows — added for 15j, which must
sum WIS over exactly the margins an energy score collapses), and
`n_negative`/`negative_by_model` report predicted quantiles that fell below zero.

⚠ **The two scales are scored from different objects.** The Gaussian observation fan of
`two_stage_forecast` can push low quantiles below zero, which the natural scale must see and
penalise, but which `log_shift` cannot represent (scoringutils v2 errors on it). Natural scale =
raw quantiles; log scale = a copy truncated at 0, with the truncation counted, returned and
`@warn`ed. Verified identical to `transform_forecasts(fun = log_shift, offset = 1)` when no
quantile is negative. Requires `scoringutils` v2.
"""
function score_wis(df_quant::DataFrame)
    @rput df_quant
    R"""
    suppressMessages(library(scoringutils))
    suppressMessages(library(data.table))
    dt <- as.data.table(df_quant)
    FU <- c("model","forecast_date","target_date","horizon","age_group")

    # NEGATIVE PREDICTED QUANTILES. `two_stage_forecast` adds a GAUSSIAN observation fan, so a
    # sufficiently overdispersed cell puts its lower quantiles below zero. WIS on the NATURAL scale
    # handles that correctly and should see the raw fan — being wrong about impossible values is
    # exactly what it must penalise. `log_shift` cannot: log(x+1) is undefined below 0, and
    # scoringutils v2 ERRORS rather than warning (it did not always), which is what broke the
    # 2026-08-07 run. So the two scales are now scored from two objects: natural from the raw
    # quantiles, log from a copy truncated at 0. Truncation is COUNTED and returned, never silent.
    # On data with no negatives this reproduces `transform_forecasts(fun = log_shift, offset = 1)`
    # to 1e-12, verified.
    # NON-FINITE PREDICTED QUANTILES. `two_stage_forecast` deliberately KEEPS +/-Inf draws (a
    # pathological Pathfinder draw can give a supercritical NGM, and fabricating a NaN there would
    # hide it), so a wholly degenerate forecast unit arrives here as Inf at every quantile level.
    # scoringutils cannot score that, and — worse — a single non-finite value makes it silently DROP
    # `bias` from EVERY returned frame while emitting only a warning, which is how 9j came to die
    # three cells later on a missing column rather than here. Whole UNITS are dropped, never
    # individual quantiles: removing part of a fan would leave an asymmetric interval set, which
    # breaks WIS itself (see the `round(q; digits=2)` note below for the same failure mode).
    # Counted and returned per model, because "this model produced an unusable forecast for N% of
    # its units" is a more important result than the WIS of the remainder.
    dt[, .bad := !is.finite(predicted)]
    dt[, .unit_bad := any(.bad), by = FU]
    n_nonfinite  <- sum(dt$.bad)
    nonfinite_rows <- if (n_nonfinite > 0) as.data.frame(dt[(.unit_bad),
                          .(units_dropped = uniqueN(.SD), rows = .N), by = .(model), .SDcols = FU]) else
                          data.frame(model = character(0), units_dropped = integer(0), rows = integer(0))
    units_total <- as.data.frame(dt[, .(units_total = uniqueN(.SD)), by = .(model), .SDcols = FU])
    dt <- dt[!(.unit_bad)][, c(".bad", ".unit_bad") := NULL]

    n_neg    <- sum(dt$predicted < 0)
    neg_rows <- if (n_neg > 0) as.data.frame(dt[predicted < 0,
                    .(n = .N, worst = min(predicted)), by = .(model)]) else
                    data.frame(model = character(0), n = integer(0), worst = numeric(0))
    dt_pos <- data.table::copy(dt); dt_pos[predicted < 0, predicted := 0]

    mkfq <- function(d) as_forecast_quantile(d, forecast_unit = FU, observed = "observed",
                            predicted = "predicted", quantile_level = "quantile_level")
    sc_nat <- score(mkfq(dt))
    sc_log <- score(transform_forecasts(mkfq(dt_pos), fun = log_shift, offset = 1, append = FALSE))
    sc_nat[, scale := "natural"]; sc_log[, scale := "log"]
    sc <- rbind(sc_nat, sc_log, fill = TRUE)
    # `rbind` on a `scores` object drops the `metrics` attribute that summarise_scores() requires.
    attr(sc, "metrics") <- attr(sc_nat, "metrics"); class(sc) <- class(sc_nat)
    by_model    <- as.data.frame(summarise_scores(sc, by = c("model","scale")))
    by_model_h  <- as.data.frame(summarise_scores(sc, by = c("model","horizon","scale")))
    by_model_dt <- as.data.frame(summarise_scores(sc, by = c("model","forecast_date","scale")))
    by_model_dt_h <- as.data.frame(summarise_scores(sc, by = c("model","forecast_date","horizon","scale")))
    # age-stratified WIS by horizon (Fig 3C analog): one WIS per model × horizon × age bin
    by_model_h_age <- as.data.frame(summarise_scores(sc, by = c("model","horizon","age_group","scale")))
    # FULL forecast unit (model x origin x horizon x age). Added 2026-10-15 for 15j, which compares
    # WIS against the multivariate energy score: the matched comparator for "ES over the 7 age bins
    # at (model, origin, horizon)" is the SUM of WIS over those 7 bins at that origin and horizon,
    # and `by_model_h_age` has already averaged over origins before you can take that sum. Scoring
    # two rules on two different aggregations would conflate the aggregation with the metric. The
    # grouping is 1-to-1 with the forecast unit, so nothing is actually aggregated here.
    by_model_dt_h_age <- as.data.frame(summarise_scores(sc,
        by = c("model","forecast_date","horizon","age_group","scale")))
    """
    by_model       = rcopy(R"by_model")
    by_model_h     = rcopy(R"by_model_h")
    by_model_dt    = rcopy(R"by_model_dt")
    by_model_dt_h  = rcopy(R"by_model_dt_h")
    by_model_h_age = rcopy(R"by_model_h_age")
    by_model_dt_h_age = rcopy(R"by_model_dt_h_age")
    n_neg          = Int(rcopy(R"n_neg"))
    neg_by_model   = rcopy(R"neg_rows")
    n_nonfinite    = Int(rcopy(R"n_nonfinite"))
    nonfinite_by_model = rcopy(R"nonfinite_rows")
    units_by_model     = rcopy(R"units_total")

    if n_nonfinite > 0
        df = leftjoin(nonfinite_by_model, units_by_model; on = :model)
        df.pct_units_dropped = round.(100 .* df.units_dropped ./ df.units_total; digits = 1)
        @warn "score_wis: $(n_nonfinite) predicted quantiles were NON-FINITE (±Inf). The whole " *
              "forecast UNIT was dropped in each case — a partial fan would break WIS's interval " *
              "structure. These models are NOT scored on the same units as the others:" df
    end

    # Report the truncation loudly. 159 of 9576 rows on the 2026-08-07 three-origin smoke, ALL of
    # them `weighted-hweibull|neighbourhood` at quantile levels 0.05-0.30, worst -2.03e6 against a
    # typical positive forecast of 2.7e4 — that is not rounding noise, it is one model's predictive
    # fan being far too wide, and the log-scale WIS for it is a truncated quantity.
    if n_neg > 0
        @warn "score_wis: $(n_neg) of $(nrow(df_quant)) predicted quantiles were < 0 and were " *
              "TRUNCATED AT 0 for the log scale only (the natural scale scores the raw fan). " *
              "Concentrated in:" neg_by_model
    end

    # scoringutils DROPS a metric's column when its computation fails, and reports the failure as an
    # R *warning* — so this function would otherwise return a perfectly well-formed frame with the
    # headline metric missing, and `nrow(sc) > 0` would pass. Measured 2026-07-30: one drifted
    # quantile endpoint (0.75 -> 0.7500000000000001) makes the interval set asymmetric and reduces
    # `by_model` to ["model","scale","bias"] — wis, both coverages, all 3 WIS components and
    # ae_median gone. Fail loudly instead; see CLAUDE.md "scoringutils v2 quantile levels".
    # `bias` is in this list since 2026-08-07: it was the ONE metric not asserted on, and it was
    # exactly the one scoringutils dropped (on non-finite predictions), so the frame came back
    # well-formed and 9j died three cells later inside a plotting function. Assert on every metric
    # any downstream figure reads, across EVERY returned frame — a metric can survive in `by_model`
    # and vanish from `by_model_h`.
    required = ["wis", "overprediction", "underprediction", "dispersion", "bias",
                "interval_coverage_50", "interval_coverage_90", "ae_median"]
    # EVERY returned frame must be in this tuple. A metric can survive in `by_model` and vanish
    # from `by_model_h`, so a frame left out of the sweep is a frame whose dropped columns nobody
    # notices -- which is the exact failure this assert exists for.
    missing_cols = union([setdiff(required, names(f)) for f in
                          (by_model, by_model_h, by_model_dt, by_model_dt_h, by_model_h_age,
                           by_model_dt_h_age)]...)
    if !isempty(missing_cols)
        qs = sort(unique(df_quant.quantile_level))
        error("""
              score_wis: scoringutils dropped metric column(s) $(missing_cols).
              This is almost always non-exact quantile levels: interval endpoints are matched by
              exact Float64 equality, so a single drifted level breaks the whole symmetric-interval
              set (and with it `wis` itself), emitting only an R warning.
              Levels received ($(length(qs))): $(qs)
              Endpoints present — 0.05:$(0.05 in qs) 0.25:$(0.25 in qs) 0.5:$(0.5 in qs) \
              0.75:$(0.75 in qs) 0.95:$(0.95 in qs)
              Fix: round the levels (`to_quantile_long` does this via `round(q; digits=2)`).""")
    end
    return (; by_model, by_model_h, by_model_dt, by_model_dt_h, by_model_h_age,
              by_model_dt_h_age,
              n_negative = n_neg, negative_by_model = neg_by_model,
              n_nonfinite, nonfinite_by_model, units_by_model)
end

"""Native sample CRPS (Gneiting–Raftery energy form, O(M log M)) — cross-check."""
function crps_sample(x::AbstractVector, y::Real)
    M = length(x); M == 0 && return NaN
    t1 = mean(abs.(x .- y))
    xs = sort(x)
    t2 = 0.0
    @inbounds for i in 1:M
        t2 += (2i - M - 1) * xs[i]
    end
    return t1 - t2 / (M^2)
end

"""Mean native CRPS over all age×horizon cells of an `A×H×D` forecast array."""
function mean_crps(fc::AbstractArray{<:Real,3}, truth::AbstractMatrix)
    A, H, _ = size(fc)
    vals = [crps_sample(collect(fc[a, h, :]), truth[a, h]) for a in 1:A, h in 1:H]
    return mean(vals)
end

# ── Multivariate proper scoring rules (15j) ───────────────────────────────────────────
# The univariate `crps_sample` above scores one cell at a time. These two score a whole
# `d`-vector jointly, which is what lets 15j ask whether the 7 age bins (or the 4 horizons)
# move together as the model says they should. Both take `X` as `d × M` — one COLUMN per
# draw — which matches `scoringRules::es_sample(y, dat)` and keeps a draw's coordinates
# contiguous in Julia's column-major memory (measured 0.095 s at d=7, M=10_000).

"""
    _kbn(s, c, x) -> (s, c)

One Kahan–Babuška–Neumaier compensated-summation step: add `x` to the running sum `s` carrying the
lost low-order bits in `c`; the sum is `s + c`.

⚠ **This is not premature polish — it is what makes the energy score's own defining identity hold
at this grid's `M`.** `energy_score`'s pair term sums `M(M−1)/2` square roots (5e7 terms at
`M = 10 000`) and the score is then a DIFFERENCE of two nearly equal quantities, so naive
accumulation is amplified twice over. Measured on the 2026-09-16 forecast grid at `d = 1`, where
the energy score must equal `crps_sample` exactly: naive summation disagreed by up to **8.3e-6**
relative over 21 168 real groups, compensated summation by **≤1e-13**.
  Note why a cross-check against `scoringRules::es_sample` does NOT catch this: that function sums
the same O(M²) way, so both implementations carry the same error and agree to 1e-14 while both are
off. Only the comparison against the well-conditioned O(M log M) `crps_sample` exposes it.
"""
@inline function _kbn(s::Float64, c::Float64, x::Float64)
    t = s + x
    c += ifelse(abs(s) >= abs(x), (s - t) + x, (x - t) + s)
    return t, c
end

"""
    energy_score(X, y) -> Float64

Sample energy score (Gneiting–Raftery, β=1) of the `d × M` draw matrix `X` against the
observation `y` (length `d`):

    ES = (1/M)·Σ_m‖x_m − y‖  −  (1/(2M²))·Σ_i Σ_j‖x_i − x_j‖

⚠ **The `1/M²` pair-term convention is load-bearing.** It is the biased Gneiting estimator, and
it is exactly what `crps_sample` above and `scoringRules::es_sample` both use — so **at `d == 1`
this function equals `crps_sample(vec(X), y[1])` to floating-point exactness**. That identity is
the cheapest tripwire available for this whole file (15j asserts it over 10 584 real groups at
`rtol = 1e-10`); switching to the unbiased `1/(M(M−1))` form would silently break it.

⚠ **The estimator's bias depends on `M`**, so two groups scored at different `M` are two
different estimators. Callers comparing models must equalise `M` first — see 15j's
`mv_scores`, which also explains why that is *not* the same thing as aligning draws.

⚠ **Not robust to extreme draws, and the natural scale is not safe.** ES is a mean of Euclidean
norms, so one pathological draw dominates it: measured on the 2026-09-16 forecast grid, a single
3.8e109 draw takes the natural-scale ES of its cell to 6.1e101 against a grid median of ~3.9e4.
WIS is structurally immune because it sees only the 5–95% quantiles; ES sees every draw. Score
on the log scale (`log(x+1)`, as `score_wis` does via `log_shift`), not the natural one.
"""
function energy_score(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    d, M = size(X)
    length(y) == d || throw(DimensionMismatch(
        "energy_score: X is $(d)×$(M) (d × M) but y has length $(length(y)); " *
        "X must carry ONE COLUMN PER DRAW"))
    M == 0 && return NaN
    t1 = 0.0; c1 = 0.0
    @inbounds for m in 1:M
        s = zero(Float64)
        for k in 1:d
            s += abs2(float(X[k, m]) - float(y[k]))
        end
        t1, c1 = _kbn(t1, c1, sqrt(s))
    end
    t1 = (t1 + c1) / M
    # Σ_i Σ_j over ALL ordered pairs = 2·Σ_{i<j} (the i==j terms are 0), so the half-sum
    # doubled and divided by 2M² is just the half-sum divided by M².
    t2 = 0.0; c2 = 0.0
    @inbounds for i in 1:M
        for j in (i + 1):M
            s = zero(Float64)
            for k in 1:d
                s += abs2(float(X[k, i]) - float(X[k, j]))
            end
            t2, c2 = _kbn(t2, c2, sqrt(s))
        end
    end
    return t1 - (t2 + c2) / (M^2)
end

"""
    variogram_score(X, y; p = 0.5) -> Float64

Sample variogram score of order `p` of the `d × M` draw matrix `X` against `y`:

    VS_p = Σ_i Σ_j ( |y_i − y_j|^p − (1/M)·Σ_m |x_mi − x_mj|^p )²

over all ordered pairs with uniform weights — `scoringRules::vs_sample(y, dat, p)`'s convention.
The full double sum (not the `i<j` half) is what pins the normalisation, and is the one thing
only the R cross-check can confirm; 15j runs it.

Complexity is **`O(M·d²)`**, not ES's `O(M²·d)` — at `d=28, M=10_000` that is 7.8e6 against
2.8e9, i.e. VS is effectively free and several `p` can be computed for nothing.

⚠ **VS is shift-invariant and is NOT proper for the marginals.** It sees only pairwise
*differences*, so a forecast biased uniformly high is not penalised at all. It therefore
complements the energy score and must never replace it — and the distinction belongs in the
figure's own title, not only in this docstring.

⚠ **`p` is a dependence-weighting knob, not a robustness one.** Lowering it to 0.5 does *not*
make the natural scale usable: measured, the cell that sends ES to 6.1e101 sends VS(p=0.5) to
1.2e103. Score on the log scale and read `p` as a choice about which pairs to emphasise.

Returns `NaN` at `d == 1` (a one-coordinate field has no pairwise structure) rather than `0.0`,
so it cannot average silently into a summary.
"""
function variogram_score(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; p::Real = 0.5)
    d, M = size(X)
    length(y) == d || throw(DimensionMismatch(
        "variogram_score: X is $(d)×$(M) (d × M) but y has length $(length(y)); " *
        "X must carry ONE COLUMN PER DRAW"))
    (M == 0 || d == 1) && return NaN
    tot = 0.0
    @inbounds for i in 1:d
        for j in (i + 1):d
            acc = 0.0; ca = 0.0
            for m in 1:M
                acc, ca = _kbn(acc, ca, abs(float(X[i, m]) - float(X[j, m]))^p)
            end
            dev = abs(float(y[i]) - float(y[j]))^p - (acc + ca) / M
            tot += abs2(dev)
        end
    end
    return 2 * tot          # the i<j half-sum doubled = the full ordered double sum
end
