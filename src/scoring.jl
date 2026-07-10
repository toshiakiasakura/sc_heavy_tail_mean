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
`(; by_model, by_model_h, by_model_dt, by_model_dt_h, by_model_h_age)` with WIS +
components, bias, and interval coverage; `by_model_h_age` adds the age-stratified
(model × horizon × age_group) WIS used by the 9j Fig-3-style diagnostics.
Requires `scoringutils` v2 (`as_forecast_quantile`/`transform_forecasts`/`score`).
"""
function score_wis(df_quant::DataFrame)
    @rput df_quant
    R"""
    suppressMessages(library(scoringutils))
    suppressMessages(library(data.table))
    dt <- as.data.table(df_quant)
    fq <- as_forecast_quantile(
        dt,
        forecast_unit = c("model","forecast_date","target_date","horizon","age_group"),
        observed = "observed", predicted = "predicted", quantile_level = "quantile_level")
    # append a log-scale copy (scale column: "natural" + "log"); score both (inst/1e)
    fq <- transform_forecasts(fq, fun = log_shift, offset = 1)
    sc <- score(fq)
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
    return (; by_model, by_model_h, by_model_dt, by_model_dt_h, by_model_h_age)
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
