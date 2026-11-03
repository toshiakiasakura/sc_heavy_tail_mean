# 18j_viz_utils.jl — helpers for `18j_scenario_bars.ipynb` (2026-11-02, user request).
#
# 17j's ranking figure sorts the series separately in each horizon panel, so one series sits on a
# different row in every panel. 18j draws the SAME ranking with the rows fixed: horizons as columns
# sharing y ticks, the two no-contact series first, then four bars per contact model. A second figure,
# `plot_scenario_dots`, is a Cleveland dot plot of the same frame at h1 and h4: one row per contact model
# with its four scenarios, under one no-contact row with its two.
#
# Requires `forecast_utils.jl` → `9j_viz_utils.jl` → `15j_viz_utils.jl` → `17j_viz_utils.jl`. The input
# is 17j's `scenario_rank_frame` (read back from its CSV by the notebook); nothing here fits or reads
# `dt_intermediate`.

using Printf

# The series that orders the contact models, and each model's four bars, top to bottom.
const BAR_ORDER_SCENARIO = "variant + contact"
const BAR_CONTACT_ROWS = ("contact (frozen)"           => "contact (frozen) only",
                          "variant + contact (frozen)" => "contact (frozen) – variant",
                          "contact"                    => "contact only",
                          "variant + contact"          => "contact – variant")
const BAR_NULL_ROWS = (("no contact", "no contact – no variant"), ("variant", "no contact – variant"))

_bar_short(m) = replace(m, "unweighted-" => "", "weighted-" => "", "neighbourhood" => "nbhd")

"""
    scenario_bar_contact_order(rk; scenario = BAR_ORDER_SCENARIO) -> (; order, models, levels)

The order of the contact models: score each by its `scenario` mean WIS averaged over the horizons in
`rk`, rank each level of each factor (degree, NGM) by the BEST score of any model with that level, and
sort by (degree rank, NGM rank). So the degree models stay together, and a model can sit above one that
scores better (`models` holds the scores, `levels` the per-level best).
"""
function scenario_bar_contact_order(rk::DataFrame; scenario::AbstractString = BAR_ORDER_SCENARIO)
    models = combine(groupby(@subset(rk, :scenario .== scenario), :model), :wis => mean => :mean_wis)
    models.degree = first.(split.(models.model, "|"))
    models.ngm    = last.(split.(models.model, "|"))
    levels = vcat([rename(combine(groupby(models, f), :mean_wis => minimum => :best), f => :level)
                   for f in (:degree, :ngm)]...; source = :factor => ["degree", "ngm"])
    best = Dict(levels.level .=> levels.best)
    sort!(models, [order(:degree; by = l -> best[l]), order(:ngm; by = l -> best[l])])
    return (; order = String.(models.model), models, levels = sort!(levels, [:factor, :best]))
end

"""
    scenario_bar_rows(contact_order) -> DataFrame

The y axis, one row per tick from the top: `y`, `tick`, `model`, `scenario` (`missing` on a contact
model's heading row, which draws no bar).
"""
function scenario_bar_rows(contact_order)
    rows = NamedTuple[]
    y = 0.0
    for (sc, tick) in BAR_NULL_ROWS
        push!(rows, (; y = y -= 1, tick, model = NULL_LABEL, scenario = sc))
    end
    for m in contact_order
        push!(rows, (; y = y -= 1.5, tick = _bar_short(m), model = m, scenario = missing))
        for (sc, tick) in BAR_CONTACT_ROWS
            push!(rows, (; y = y -= 1, tick, model = m, scenario = sc))
        end
    end
    return DataFrame(rows)
end

"""
    plot_scenario_bars_by_horizon(rk, cfg; title_suffix, scale) -> Plot

One column per horizon, sharing y ticks (`scenario_bar_rows`, in `scenario_bar_contact_order`): each
series as a horizontal bar of its mean WIS, labelled with the value and its rank within the horizon.
Colour = scenario (`SCENARIO_STYLE`; frozen paler), as in `plot_scenario_rank_by_horizon`. The x-axes all
start at 0 but are NOT shared, because WIS roughly triples from h1 to h4: compare rows within a column.
"""
function plot_scenario_bars_by_horizon(rk::DataFrame, cfg; title_suffix::AbstractString = "",
                                       scale::AbstractString = WIS_SCALE)
    rows = scenario_bar_rows(scenario_bar_contact_order(rk).order)
    bars = dropmissing(rows, :scenario)
    ylim = (minimum(rows.y) - 0.7, maximum(rows.y) + 0.7)
    panels = Plots.Plot[]
    for (k, h) in enumerate(cfg.horizons)
        s = innerjoin(bars, @subset(rk, :horizon .== h); on = [:model, :scenario])
        @assert nrow(s) == nrow(bars) "plot_scenario_bars_by_horizon: h$h has $(nrow(s)) of $(nrow(bars)) bars"
        p = plot(; title = "horizon $h", titlefontsize = 10, xlabel = "mean WIS ($scale)", legend = false,
                 yticks = (rows.y, k == 1 ? rows.tick : fill("", nrow(rows))), ytickfontsize = 7,
                 xlim = (0, maximum(s.wis) * 1.3), ylim = ylim)
        for r in eachrow(s)
            colr = SCENARIO_STYLE[r.scenario][1]
            bar!(p, [r.y], [r.wis]; orientation = :horizontal, color = colr, linecolor = colr,
                 fillalpha = (occursin("frozen", r.scenario) ? 0.45 : 0.95), bar_width = 0.75)
            annotate!(p, r.wis, r.y, text(@sprintf(" %.3f (#%d)", r.wis, r.rank), 6, :left))
        end
        push!(panels, p)
    end
    return plot(panels...; layout = (1, length(panels)), link = :y, size = (1500, 820),
                plot_title = "18j — mean WIS ($(scale) scale) of all model × scenario series$(title_suffix); (#) = rank within horizon, pale = frozen",
                plot_titlefontsize = 10)
end

"""
    scenario_dots_frame(rk; horizons = (1, 4)) -> DataFrame

The dot plot's values: `rk` at `horizons`, with `rel_wis` = mean WIS ÷ the mean WIS of "no contact" (no
variant) at the same horizon. This is a ratio of means, not a mean of per-origin ratios.
"""
function scenario_dots_frame(rk::DataFrame; horizons = (1, 4))
    d = @subset(rk, in.(:horizon, Ref(horizons)))
    ref = select(@subset(d, :scenario .== "no contact"), :horizon, :wis => :wis_ref)
    @assert sort(ref.horizon) == sort(collect(horizons)) "scenario_dots_frame: need one no-contact row per horizon"
    d = innerjoin(d, ref; on = :horizon)
    d.rel_wis = d.wis ./ d.wis_ref
    return d
end

"""
    plot_scenario_dots(rk; horizons = (1, 4), title_suffix, scale, panel_width = 500) -> Plot

Cleveland dot plot of `scenario_dots_frame`: a no-contact row with its two `BAR_NULL_ROWS` dots, then one
row per contact model (in `scenario_bar_contact_order`) with one dot per `BAR_CONTACT_ROWS` scenario; one
column per horizon sharing both axes, and a legend cell. The line at 1 is no contact – no variant.
Colour and marker = `SCENARIO_STYLE`. Frozen is drawn first as a larger open diamond and conditional on
top as a smaller filled circle, so that a coinciding pair shows as a circle inside a diamond.
"""
function plot_scenario_dots(rk::DataFrame; horizons = (1, 4), title_suffix::AbstractString = "",
                            scale::AbstractString = WIS_SCALE, panel_width::Real = 500)
    models = [NULL_LABEL; scenario_bar_contact_order(rk).order]
    groups = (first.(BAR_CONTACT_ROWS)..., first.(BAR_NULL_ROWS)...)   # frozen first: drawn underneath
    n_dots = length(BAR_NULL_ROWS) + (length(models) - 1) * length(BAR_CONTACT_ROWS)
    y = Dict(m => -i for (i, m) in enumerate(models))
    style(sc) = (c = SCENARIO_STYLE[sc]; occursin("frozen", sc) ?
                 (; marker = c[3], ms = 9, markercolor = :white, markerstrokecolor = c[1], markerstrokewidth = 1.5) :
                 (; marker = c[3], ms = 5.5, markercolor = c[1], markerstrokecolor = c[1], markerstrokewidth = 0.5))
    ys = [y[m] for m in models]
    dots = scenario_dots_frame(rk; horizons)
    lo, hi = extrema(dots.rel_wis)
    pad = 0.08 * (hi - lo)
    panels = Plots.Plot[]
    for (k, h) in enumerate(horizons)
        d = @subset(dots, :horizon .== h, in.(:model, Ref(models)), in.(:scenario, Ref(groups)))
        @assert nrow(d) == n_dots "plot_scenario_dots: h$h has $(nrow(d)) of $n_dots dots"
        p = plot(; title = "horizon $h", titlefontsize = 10, xlabel = "relative WIS ($scale)", legend = false,
                 yticks = (ys, k == 1 ? [m == NULL_LABEL ? "no contact" : _bar_short(m) for m in models] : fill("", length(ys))),
                 ytickfontsize = 8,
                 xlim = (lo - pad, hi + pad), ylim = (minimum(ys) - 0.6, maximum(ys) + 0.6))
        hline!(p, ys; color = :grey70, ls = :dot, lw = 0.8, label = "")
        vline!(p, [1.0]; color = :grey40, lw = 1, label = "")
        for sc in groups, r in eachrow(@subset(d, :scenario .== sc))
            scatter!(p, [r.rel_wis], [y[r.model]]; label = "", style(sc)...)
        end
        push!(panels, p)
    end
    leg = plot(; framestyle = :none, grid = false, xticks = false, yticks = false, legend = :left,
               legendfontsize = 8, foreground_color_legend = nothing)
    for (sc, lab) in (BAR_NULL_ROWS..., BAR_CONTACT_ROWS...)
        scatter!(leg, [NaN], [NaN]; label = lab, style(sc)...)
    end
    n, leg_width = length(panels), 250
    width = n * panel_width + leg_width
    return plot(panels..., leg; layout = Plots.grid(1, n + 1; widths = [fill(panel_width / width, n); leg_width / width]), size = (width, 520),
                plot_title = "18j — mean WIS relative to no contact – no variant (= 1) at horizons $(join(horizons, ", ", " and "))$(title_suffix)",
                plot_titlefontsize = 10)
end

"""
    scenario_age_rel_wis(by_h_age; scale = WIS_SCALE) -> DataFrame

`score_wis(…).by_model_h_age` at `scale`, with `rel_wis` = mean WIS ÷ the mean WIS of "no contact" (no
variant) in the same age group and horizon: a ratio of means over origins.
"""
function scenario_age_rel_wis(by_h_age::DataFrame; scale::AbstractString = WIS_SCALE)
    d = select(@subset(by_h_age, :scale .== scale), :model => :series, :horizon, :age_group, :wis)
    ref = select(@subset(d, :series .== NULL_LABEL), :horizon, :age_group, :wis => :wis_ref)
    d = innerjoin(d, ref; on = [:horizon, :age_group])
    d.rel_wis = d.wis ./ d.wis_ref
    return d
end

"""
    plot_scenario_age_heatmap(rel, rk, grid; horizons = (1, 4), cap, title_suffix, scale) -> Plot

One row per horizon, one column: `rel_wis` (`scenario_age_rel_wis`) by age group (y, youngest at the bottom)
and series (x, in the order of the bars: the scenario on the tick, the contact model above its columns). One
colour scale, white at 1, blue below and red above. Every cell prints its value, so `cap` (the largest distance
from 1 the colours resolve; default the data's) never hides one.
"""
function plot_scenario_age_heatmap(rel::DataFrame, rk::DataFrame, grid; horizons = (1, 4), cap = nothing,
                                   title_suffix::AbstractString = "", scale::AbstractString = WIS_SCALE)
    rows = dropmissing(scenario_bar_rows(scenario_bar_contact_order(rk).order), :scenario)
    name = Dict((r.model, r.scenario) => r.series for r in eachrow(rk))
    series = [name[(r.model, r.scenario)] for r in eachrow(rows)]
    headings = [(_bar_short(m), mean(findall(==(m), rows.model))) for m in unique(rows.model) if m != NULL_LABEL]
    breaks = [j + 0.5 for j in 1:(nrow(rows) - 1) if rows.model[j] != rows.model[j + 1]]
    A, n = grid.N, length(series)
    val = Dict((r.series, r.horizon, r.age_group) => r.rel_wis for r in eachrow(rel))
    d = cap === nothing ? maximum(abs.(rel.rel_wis .- 1)) : cap
    panels = Plots.Plot[]
    for (k, h) in enumerate(horizons)
        M = [val[(s, h, grid.LAB[a])] for a in 1:A, s in series]
        p = heatmap(1:n, 1:A, M; c = cgrad(:RdBu; rev = true), clims = (1 - d, 1 + d), title = "horizon $h",
                    titlefontsize = 10, ylabel = "age group", yticks = (1:A, grid.LAB), ytickfontsize = 8,
                    xticks = (1:n, k == length(horizons) ? rows.tick : fill("", n)), xtickfontsize = 7, xrotation = 45,
                    colorbar_title = "relative WIS", grid = false, ylim = (0.5, A + 1))
        vline!(p, breaks; color = :black, lw = 0.8, label = "")
        for a in 1:A, j in 1:n
            annotate!(p, j, a, text(@sprintf("%.2f", M[a, j]), 5, abs(M[a, j] - 1) > 0.8d ? :white : :black))
        end
        for (lab, x) in headings
            annotate!(p, x, A + 0.75, text(lab, 8, :center))
        end
        push!(panels, p)
    end
    capped = cap === nothing ? "" : "; colours capped at 1 ± $(cap)"
    return plot(panels...; layout = (length(panels), 1), size = (1350, 860),
                plot_title = "18j — mean WIS ($(scale) scale) relative to no contact – no variant, by age$(title_suffix)$(capped)",
                plot_titlefontsize = 10)
end

"""
    scenario_coverage_by_origin(fc_store, truth, grid; level = 0.95, horizons = (1, 4)) -> DataFrame

From the fans' draws, per (origin, series) of `fc_store` × horizon: `coverage` = the share of the age groups
whose observed value (`truth[origin]`, `A×H`) lies inside the central `level` interval. It is computed from
the draws because the scored quantiles stop at 5–95%, so scoringutils cannot give 95%.
"""
function scenario_coverage_by_origin(fc_store, truth, grid; level::Real = 0.95, horizons = (1, 4))
    α = round((1 - level) / 2; digits = 4)     # exact 0.05, as `to_quantile_long` rounds its levels
    rows = NamedTuple[]
    for ((origin, series), fc) in fc_store, h in horizons
        n = 0
        for a in 1:grid.N
            lo, hi = quantile(view(fc, a, h, :), [α, 1 - α])
            @assert !isnan(lo) && !isnan(hi) "scenario_coverage_by_origin: NaN bound at $origin $series h$h age $a"
            n += lo <= truth[origin][a, h] <= hi
        end
        push!(rows, (; series, forecast_date = origin, horizon = h, coverage = n / grid.N))
    end
    return sort!(DataFrame(rows), [:series, :horizon, :forecast_date])
end

# The four series coloured in the over-time figure. No contact is black rather than its `SCENARIO_STYLE`
# grey, which would not stand out from the other series' grey.
const OVER_TIME_HIGHLIGHT = ((NULL_LABEL,                                "no contact – no variant",          :black),
                             (NULL_VARIANT_LABEL,                        "no contact – variant",             :firebrick),
                             ("weighted-hweibull|neighbourhood",         "hweibull|nbhd: contact only",      :steelblue),
                             ("weighted-hweibull|neighbourhood-variant", "hweibull|nbhd: contact – variant", :darkorange))

"One over-time panel: the other series in grey underneath, then `OVER_TIME_HIGHLIGHT` on top."
function _over_time_panel(d::DataFrame, col::Symbol; title, ylabel, ylim)
    p = plot(; title, titlefontsize = 9, ylabel, xlabel = "forecast origin", legend = false, xrotation = 45,
             ylim, ytickfontsize = 8, xtickfontsize = 7)
    named = first.(OVER_TIME_HIGHLIGHT)
    for s in groupby(@subset(d, .!in.(:series, Ref(named))), :series)
        plot!(p, s.forecast_date, s[!, col]; color = :grey80, lw = 0.6, label = "")
    end
    for (series, _, colr) in OVER_TIME_HIGHLIGHT
        s = @subset(d, :series .== series)
        @assert nrow(s) > 0 "_over_time_panel: no rows for $series"
        plot!(p, s.forecast_date, s[!, col]; color = colr, lw = 1.6, label = "")
    end
    return p
end

"""
    plot_scenario_wis_coverage_over_time(by_dt_h, cov, grid; horizons, nominal, title_suffix, scale) -> Plot

One row per horizon. Left: per-origin mean WIS (`by_dt_h`, with `series` and `forecast_date`), bands labelled with
9j's `PERIODS`. Right: per-origin coverage (`scenario_coverage_by_origin`), same bands, with `nominal` dashed; it
moves in steps of 1/A, so lines often coincide. Both have `OVER_TIME_HIGHLIGHT` coloured and the others grey.
"""
function plot_scenario_wis_coverage_over_time(by_dt_h::DataFrame, cov::DataFrame, grid; horizons = (1, 4),
                                              nominal::Real = 0.95, title_suffix::AbstractString = "",
                                              scale::AbstractString = WIS_SCALE)
    panels = Plots.Plot[]
    for h in horizons
        w = sort(@subset(by_dt_h, :scale .== scale, :horizon .== h), :forecast_date)
        c = sort(@subset(cov, :horizon .== h), :forecast_date)
        dmin, dmax = extrema(w.forecast_date)
        @assert extrema(c.forecast_date) == (dmin, dmax) "plot_scenario_wis_coverage_over_time: windows differ at h$h"
        l = _over_time_panel(w, :wis; title = "horizon $h — mean WIS", ylabel = "mean WIS ($scale)",
                             ylim = (0, 1.25 * maximum(w.wis)))
        shade_periods!(l, dmin, dmax; fontsize = 6)
        r = _over_time_panel(c, :coverage; title = "horizon $h — $(round(Int, 100nominal))% coverage",
                             ylabel = "share of $(grid.N) age groups covered", ylim = (0, 1.03))
        hline!(r, [nominal]; color = :grey30, ls = :dash, lw = 1, label = "")   # after the dates, see 14j §7
        shade_periods!(r, dmin, dmax; labels = false)
        push!(panels, l, r)
    end
    leg = plot(; framestyle = :none, grid = false, xticks = false, yticks = false, legend = :left,
               legendfontsize = 8, foreground_color_legend = nothing)
    for (_, lab, colr) in OVER_TIME_HIGHLIGHT
        plot!(leg, [NaN], [NaN]; color = colr, lw = 1.6, label = lab)
    end
    n_other = length(unique(cov.series)) - length(OVER_TIME_HIGHLIGHT)
    plot!(leg, [NaN], [NaN]; color = :grey80, lw = 0.6, label = "other $(n_other) series")
    @assert length(panels) == 4 "plot_scenario_wis_coverage_over_time: the layout holds two horizons"
    return plot(panels..., leg; layout = @layout([[a b; c d] e{0.16w}]), size = (1450, 820),
                plot_title = "18j — mean WIS and coverage over time$(title_suffix)", plot_titlefontsize = 10)
end

"18j's own cache of the age-level scores and coverage — small frames only, no fans or quantiles."
age_coverage_cache_path(cfg::FrameworkConfig; save_dir::AbstractString = "../dt_intermediate") =
    joinpath(save_dir, "18j_age_coverage_$(contacts_label(cfg)).jld2")

"""
    scenario_age_coverage_or_load(cfg, sources, series, grid; horizons, origin_max, cache_path, rebuild)
        -> (; by_model_h_age, by_model_dt_h, cov95, cov90)

The age and over-time figures' inputs: `score_wis` frames and draw-based 95%/90% coverage for `series` at
`horizons` over origins ≤ `origin_max`. `sources` holds one `(path, labels, frozen)` per assembly cache, read with
`load_9j_assembly`, which cannot fit. The result is cached at `cache_path` and reused while `horizons`, `origin_max`
and `series` match and no source is newer than it. Comparing mtimes matters because 9j's caches check names only, so
one that survives a change of contents would otherwise be served silently (CLAUDE.md's inc2prev-schema Gotcha).
"""
function scenario_age_coverage_or_load(cfg::FrameworkConfig, sources, series, grid; horizons, origin_max::Date,
                                       cache_path::AbstractString = age_coverage_cache_path(cfg), rebuild::Bool = false)
    want_h, want_s = collect(horizons), sort(collect(series))
    if !rebuild && isfile(cache_path)
        c = load(cache_path)
        newer = [p for (p, _, _) in sources if isfile(p) && mtime(p) > mtime(cache_path)]
        if c["horizons"] == want_h && c["origin_max"] == origin_max && c["series"] == want_s && isempty(newer)
            println("age scores and coverage: loaded cache ", cache_path)
            return (; by_model_h_age = c["by_model_h_age"], by_model_dt_h = c["by_model_dt_h"],
                      cov95 = c["cov95"], cov90 = c["cov90"])
        end
        @warn "18j cache stale (horizons/origins/series changed, or a source is newer) — rebuilding" cache_path newer
    end
    keep = Set(want_s)   # drops `no-contact|null@frozen`, in the frozen cache but not one of the 22 series (as in 17j)
    qs = DataFrame[]; c95 = DataFrame[]; c90 = DataFrame[]
    for (path, labels, frozen) in sources
        a = load_9j_assembly(cfg; labels, cache_path = path, grid)
        q, fc = rename_series(a.qall, a.fc_store, frozen)
        fans = Dict(k => v for (k, v) in fc if k[1] <= origin_max && k[2] in keep)
        truth = assembly_truth(q, grid).truth
        push!(c95, scenario_coverage_by_origin(fans, truth, grid; level = 0.95, horizons))
        push!(c90, scenario_coverage_by_origin(fans, truth, grid; level = 0.90, horizons))
        push!(qs, @subset(q, :forecast_date .<= origin_max, in.(:horizon, Ref(horizons)), in.(:model, Ref(keep))))
        println(basename(path), ": ", length(fans), " fans")
    end
    GC.gc()
    sc = score_wis(vcat(qs...))
    res = (; by_model_h_age = sc.by_model_h_age, by_model_dt_h = sc.by_model_dt_h, cov95 = vcat(c95...), cov90 = vcat(c90...))
    _atomic_jldsave(cache_path; res..., horizons = want_h, origin_max, series = want_s)
    println("age scores and coverage: wrote cache ", cache_path)
    return res
end
