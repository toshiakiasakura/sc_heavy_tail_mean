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

Run R `scoringutils` on a quantile-format table. Returns `(; by_model, by_model_h)`
DataFrames with WIS and its components, bias, and interval coverage. Requires the
`scoringutils` package (see scripts install step). Targets the v2 API
(`as_forecast_quantile` / `score` / `summarise_scores`).
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
    sc <- score(fq)
    by_model   <- as.data.frame(summarise_scores(sc, by = c("model")))
    by_model_h <- as.data.frame(summarise_scores(sc, by = c("model","horizon")))
    """
    by_model   = rcopy(R"by_model")
    by_model_h = rcopy(R"by_model_h")
    return (; by_model, by_model_h)
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
