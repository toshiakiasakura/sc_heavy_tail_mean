# scoring.jl — forecast scoring. Primary: WIS (weighted interval score) via the R
# `scoringutils` package (RCall), on quantile-format forecasts. Secondary: a native
# sample CRPS as a cheap cross-check.
#
# Also here (inst/6_null_interaction_model.md): the LOG SCORE. Note the distinction —
# `score_wis` already reports WIS on a "log scale", but that is WIS computed after a log
# TRANSFORM of the forecasts (`transform_forecasts(log_shift)`), which is a different thing from
# the logarithmic SCORE. `scoringutils` can only compute the log score for the SAMPLE forecast
# class (`as_forecast_sample`; it is undefined for quantile forecasts), so `to_sample_long` /
# `score_logs` below run the raw posterior-predictive draws through that second R path.

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
`(; by_model, by_model_h, by_model_dt, by_model_dt_h, by_model_h_age)` with WIS +
components, bias, and interval coverage; `by_model_h_age` adds the age-stratified
(model × horizon × age_group) WIS used by the 9j Fig-3-style diagnostics, and
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
    # quantiles, log from a copy truncated at 0. Truncation is COUNTED and returned, never silent —
    # the same rule `to_sample_long` follows for its own sanitisations. On data with no negatives
    # this reproduces `transform_forecasts(fun = log_shift, offset = 1)` to 1e-12, verified.
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
    """
    by_model       = rcopy(R"by_model")
    by_model_h     = rcopy(R"by_model_h")
    by_model_dt    = rcopy(R"by_model_dt")
    by_model_dt_h  = rcopy(R"by_model_dt_h")
    by_model_h_age = rcopy(R"by_model_h_age")
    n_neg          = Int(rcopy(R"n_neg"))
    neg_by_model   = rcopy(R"neg_rows")

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
    required = ["wis", "overprediction", "underprediction", "dispersion",
                "interval_coverage_50", "interval_coverage_90", "ae_median"]
    missing_cols = setdiff(required, names(by_model))
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
              n_negative = n_neg, negative_by_model = neg_by_model)
end

# ======================================================================================
# Log score — scoringutils SAMPLE class (inst/6_null_interaction_model.md)
# ======================================================================================

"""
    to_sample_long(fc, truth, model_label, win, cfg, age_labels; n_sample=1000)

Long **sample**-format table for `scoringutils::as_forecast_sample`: one row per
(age × horizon × retained draw). `fc` is `A×H×D` posterior-predictive draws, `truth` is `A×H`
realized weekly infections. Columns:
`model, forecast_date, target_date, horizon, age_group, sample_id, predicted, observed`.

Two sanitisations, both **counted and returned** rather than silently applied:

- **Non-finite draws are dropped.** `two_stage_forecast` deliberately keeps `±Inf` draws (a
  pathological Pathfinder draw can give a supercritical NGM) rather than fabricating NaN, and the
  KDE behind the log score cannot consume them. Dropping them makes the retained fan *narrower*
  than the true predictive, so the count matters — report it, never hide it.
- **Draws are thinned** to `n_sample` per cell on a deterministic even grid (the
  `stage1_moment_draws` idiom). The full 10 000 draws × 7 ages × 4 horizons × 63 origins × 6 models
  is ~10⁸ rows to hand to R; the KDE is stable far below that.

Returns `(; df, n_dropped, n_total)`.
"""
function to_sample_long(fc::AbstractArray{<:Real,3}, truth::AbstractMatrix,
                        model_label::AbstractString, win::WeeklyWindow,
                        cfg::FrameworkConfig, age_labels::Vector{String};
                        n_sample::Int = 1000)
    A, H, D = size(fc)
    rows = DataFrame(model = String[], forecast_date = Date[], target_date = Date[],
                     horizon = Int[], age_group = String[], sample_id = Int[],
                     predicted = Float64[], observed = Float64[])
    n_dropped = 0
    for a in 1:A, h in 1:H
        pool = filter(isfinite, collect(@view fc[a, h, :]))     # order preserved ⇒ systematic thin
        n_dropped += D - length(pool)
        isempty(pool) && continue
        keep = min(n_sample, length(pool))
        idx  = round.(Int, range(1, length(pool); length = keep))
        obs  = float(truth[a, h])
        for (s, k) in enumerate(idx)
            push!(rows, (model_label, win.origin, win.forecast_weeks[h], h,
                         age_labels[a], s, float(pool[k]), obs))
        end
    end
    return (; df = rows, n_dropped = n_dropped, n_total = A * H * D)
end

"Metric columns `scoringutils` returns for the sample forecast class (whichever are present)."
const _SAMPLE_METRICS = [:log_score, :crps, :dss, :bias, :mad,
                         :overprediction, :underprediction, :dispersion]

"Mean of every present sample metric, grouped by `by` (mirrors R `summarise_scores`)."
function _summarise_sample(sc::DataFrame, by::Vector{Symbol})
    cols = intersect(_SAMPLE_METRICS, propertynames(sc))
    return sort(combine(groupby(sc, by),
                        [c => (x -> mean(skipmissing(x))) => c for c in cols]), by)
end

"""One R round-trip: per-forecast-unit sample scores on the natural AND log scale."""
function _score_sample_r(df_sample::DataFrame)
    @rput df_sample
    R"""
    suppressMessages(library(scoringutils))
    suppressMessages(library(data.table))
    dt <- as.data.table(df_sample)
    FU <- c("model","forecast_date","target_date","horizon","age_group")
    .mk <- function(d) as_forecast_sample(d, forecast_unit = FU, observed = "observed",
                                          predicted = "predicted", sample_id = "sample_id")
    sc_nat <- as.data.frame(suppressWarnings(score(.mk(dt))))
    sc_nat$scale <- "natural"
    # Log-scale copy built EXPLICITLY rather than with transform_forecasts(log_shift): individual
    # sample draws can be negative (draws = pred + sigma*randn), and log_shift would return NaN.
    # pmax(., 0) censors at the model's support (infections cannot be negative).
    dtl <- copy(dt)
    dtl[, predicted := log(pmax(predicted, 0) + 1)][, observed := log(observed + 1)]
    sc_log <- as.data.frame(suppressWarnings(score(.mk(dtl))))
    sc_log$scale <- "log"
    sc_all <- rbind(sc_nat, sc_log)
    """
    return rcopy(R"sc_all")
end

"""
    score_logs(fc_store, truth_store, wins, labels, cfg, age_labels; n_sample=1000)

**Log score** (plus the free sample-class metrics `crps`/`dss`/`bias`/`mad`) via R `scoringutils`
on sample-format forecasts — the companion to `score_wis`, which cannot produce a log score because
the metric is undefined for the quantile forecast class.

`fc_store[(origin, label)]` is the `A×H×D` fan from `assemble_or_load_forecasts`;
`truth_store[origin]` is the matching `A×H` realized infections. Scoring runs **one origin at a
time** (one `@rput` + `score()` per origin) rather than as a single global transfer: `score()`
returns one row per forecast unit, so the accumulated per-unit table is only a few thousand rows
even though the sample tables handed to R are ~10⁵ rows each.

Scores are computed on **both the natural and the log scale**, so the returned frames mirror
`score_wis`'s shapes exactly (`by_model`, `by_model_h`, `by_model_dt`, `by_model_dt_h`,
`by_model_h_age`, each with a `scale` column), plus:

- `per_unit` — the full per-(model × origin × target × horizon × age × scale) score table;
- `dropped` / `floored` — the sanitisation tallies from `to_sample_long` and the `pmax(·,0)`
  censoring, as `(n, total, frac)` NamedTuples;
- `n_nonfinite_logscore` — units whose KDE log score came back non-finite (the observation fell
  outside the retained fan). These propagate into the means, so a non-zero count must be reported.

R warnings from `score()` are suppressed (the KDE chatter would repeat once per origin); the
pathologies that matter are counted explicitly and returned instead.
"""
function score_logs(fc_store, truth_store, wins, labels, cfg::FrameworkConfig,
                    age_labels::Vector{String}; n_sample::Int = 1000, verbose::Bool = true)
    parts = DataFrame[]
    n_drop = 0; n_tot = 0; n_neg = 0; n_pred = 0
    for win in wins
        haskey(truth_store, win.origin) || continue
        truth = truth_store[win.origin]
        tabs = DataFrame[]
        for lbl in labels
            haskey(fc_store, (win.origin, lbl)) || continue
            s = to_sample_long(fc_store[(win.origin, lbl)], truth, lbl, win, cfg, age_labels;
                               n_sample = n_sample)
            n_drop += s.n_dropped; n_tot += s.n_total
            isempty(s.df) || push!(tabs, s.df)
        end
        isempty(tabs) && continue
        df_sample = vcat(tabs...)
        n_neg  += count(<(0), df_sample.predicted)
        n_pred += nrow(df_sample)
        push!(parts, _score_sample_r(df_sample))
    end
    isempty(parts) && error("score_logs: no (origin, model) fans found in fc_store")
    per_unit = vcat(parts...)

    n_nonfinite = :log_score in propertynames(per_unit) ?
                  count(!isfinite, skipmissing(per_unit.log_score)) : 0
    if verbose
        println("log score: ", nrow(per_unit), " scored units over ", length(parts), " origins ",
                "(", n_sample, " draws/cell)")
        println("  non-finite forecast draws dropped: ", n_drop, "/", n_tot,
                " (", round(100 * n_drop / max(n_tot, 1); digits = 3), "%)")
        println("  negative draws censored at 0 for the log scale: ", n_neg, "/", n_pred,
                " (", round(100 * n_neg / max(n_pred, 1); digits = 3), "%)")
        n_nonfinite > 0 && @warn "non-finite KDE log scores (observation outside the retained fan)" n_nonfinite
    end

    return (; by_model       = _summarise_sample(per_unit, [:model, :scale]),
              by_model_h     = _summarise_sample(per_unit, [:model, :horizon, :scale]),
              by_model_dt    = _summarise_sample(per_unit, [:model, :forecast_date, :scale]),
              by_model_dt_h  = _summarise_sample(per_unit, [:model, :forecast_date, :horizon, :scale]),
              by_model_h_age = _summarise_sample(per_unit, [:model, :horizon, :age_group, :scale]),
              per_unit       = per_unit,
              dropped        = (; n = n_drop, total = n_tot, frac = n_drop / max(n_tot, 1)),
              floored        = (; n = n_neg,  total = n_pred, frac = n_neg / max(n_pred, 1)),
              n_nonfinite_logscore = n_nonfinite)
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
