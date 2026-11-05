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

"""
    scenario_assembly_sources(cfg; save_dir = "../dt_intermediate") -> Vector{Tuple{String,Vector{String},Bool}}

The four forecast caches behind the 22 scenario series, one `(path, labels, frozen)` each, in 17j's order:
9j's conditional cache, the frozen one, the variant one and the variant-frozen one. The last has no null model.
Shared by 18j and `19r_export_wis_age.jl`, so the list exists once.
"""
function scenario_assembly_sources(cfg::FrameworkConfig; save_dir::AbstractString = "../dt_intermediate")
    combos = vcat([(dm, nb) for dm in (NegBinAgePair(), HurdleWeibullAgePair())
                            for nb in (MeanNGM(), NeighbourhoodDegreeNGM())],
                  [(NegBinAgePair(),   DiagonalMeanNGM()),      # no-interaction
                   (NoContactDegree(), NullNGM())])             # null, in 9j's order
    lab(dm, nb) = string(degree_label(dm), "|", ngm_label(nb))
    labels_c = [lab(dm, nb) for (dm, nb) in combos]
    labels_v = [lab(dm, VariantNGM(nb)) for (dm, nb) in combos]
    return [(joinpath(save_dir, "9j_assembly_$(contacts_label(cfg)).jld2"), labels_c, false),
            (frozen_cache_path(cfg; save_dir), labels_c, true),
            (variant_cache_path(cfg; frozen = false, save_dir), labels_v, false),
            (variant_cache_path(cfg; frozen = true, save_dir),
             [lab(dm, VariantNGM(nb)) for (dm, nb) in combos if needs_stage1(dm)], true)]
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

# ══════════════════════════════════════════════════════════════════════════════════════
# Infections over time with the h1–H forecast fans (2026-11-05, user request), in the form of the
# COVIDhub ensemble figure: the observed series, and from every `every`-th origin a short median
# trajectory with its 50% and 95% bands. Two series, hweibull|nbhd with contact only and with contact +
# variant, labelled as in `OVER_TIME_HIGHLIGHT`. The fans overlap, so every pair of colours has to separate, not just
# neighbours: the colours are the first three slots of the dataviz reference palette (blue, orange, aqua), which pass
# its validator on ALL pairs (worst CVD ΔE 9.2, normal-vision 24.0). The X11 steelblue / darkorange / purple they
# replaced failed it (steelblue reads grey, purple is too dark). Aqua is below 3:1 against white, so it relies on its legend label.
# ══════════════════════════════════════════════════════════════════════════════════════

const FAN_SERIES = (("weighted-hweibull|neighbourhood",         "hweibull|nbhd: contact only",      "#2a78d6"),
                    ("weighted-hweibull|neighbourhood-variant", "hweibull|nbhd: contact – variant", "#eb6834"))
# The second figure adds negbin|mean with contact – variant, in the palette's third slot. Its series list
# contains FAN_SERIES, so the notebook caches this one list and draws both figures from it.
const FAN_SERIES_NEGBIN = (FAN_SERIES..., ("unweighted-negbin|mean-variant", "negbin|mean: contact – variant", "#1baf7a"))
const FAN_LEVELS = (0.025, 0.25, 0.5, 0.75, 0.975)

_qcol(l) = Symbol("q", l)

"18j's own cache of the fan quantiles — per series × origin × horizon × age, no draws."
fan_quantile_cache_path(cfg::FrameworkConfig; save_dir::AbstractString = "../dt_intermediate") =
    joinpath(save_dir, "18j_fan_quantiles_$(contacts_label(cfg)).jld2")

"""
    scenario_fan_quantiles_or_load(cfg, sources, series, grid; origin_max, levels, cache_path, rebuild)
        -> (; quant, observed)

`quant`: the `levels` quantiles (columns `q0.025` …) of each `series`' forecast per origin ≤ `origin_max` ×
horizon × age group, the 7 CIS bins plus `"all"` (the quantile of the per-draw SUM over ages, as in 17j's
panels). `observed`: the truth 9j scored against (`assembly_truth`), one row per target week × age group, with
`"all"` the sum over the bins. Only the `sources` holding a requested series are read, with `load_9j_assembly`,
which cannot fit. Cached at `cache_path` and validated as `scenario_age_coverage_or_load` is.
"""
function scenario_fan_quantiles_or_load(cfg::FrameworkConfig, sources, series, grid; origin_max::Date,
                                        levels = FAN_LEVELS, cache_path::AbstractString = fan_quantile_cache_path(cfg),
                                        rebuild::Bool = false)
    want_s, want_l = sort(collect(series)), collect(levels)
    use = [(p, l, f) for (p, l, f) in sources if any(s -> s in Set(series_name.(l, f)), want_s)]
    if !rebuild && isfile(cache_path)
        c = load(cache_path)
        newer = [p for (p, _, _) in use if isfile(p) && mtime(p) > mtime(cache_path)]
        if c["series"] == want_s && c["origin_max"] == origin_max && c["levels"] == want_l && isempty(newer)
            println("fan quantiles: loaded cache ", cache_path)
            return (; quant = c["quant"], observed = c["observed"])
        end
        @warn "18j fan cache stale (series/origins/levels changed, or a source is newer) — rebuilding" cache_path newer
    end
    A, H = grid.N, length(cfg.horizons)
    ages = vcat(grid.LAB, "all")
    rows = NamedTuple[]; obs = nothing
    for (path, labels, frozen) in use
        a = load_9j_assembly(cfg; labels, cache_path = path, grid)
        q, fc = rename_series(a.qall, a.fc_store, frozen)
        for ((origin, s), f) in fc
            (origin <= origin_max && s in want_s) || continue
            for (k, lab) in enumerate(ages), h in 1:H
                v = k <= A ? f[k, h, :] : vec(sum(f[:, h, :]; dims = 1))
                push!(rows, merge((; series = s, forecast_date = origin, horizon = h, age_group = lab),
                                  NamedTuple(_qcol(l) => _fq(v, l) for l in want_l)))
            end
        end
        truth = assembly_truth(q, grid).truth
        o = DataFrame([(; week = origin + Week(h), age_group = lab,
                          observed = k <= A ? M[k, h] : sum(M[:, h]))
                       for (origin, M) in truth, h in 1:H, (k, lab) in enumerate(ages) if origin <= origin_max])
        u = unique(o)
        @assert nrow(unique(u, [:week, :age_group])) == nrow(u) "scenario_fan_quantiles_or_load: two observations of one week in $(basename(path))"
        obs === nothing ? (obs = u) :
            @assert sort(obs, [:week, :age_group]) == sort(u, [:week, :age_group]) "scenario_fan_quantiles_or_load: truth differs between sources"
        println(basename(path), ": read")
    end
    GC.gc()
    quant = sort!(DataFrame(rows), [:series, :forecast_date, :horizon, :age_group])
    @assert Set(quant.series) == Set(want_s) "scenario_fan_quantiles_or_load: series missing from the sources: $(setdiff(want_s, quant.series))"
    observed = sort!(obs, [:age_group, :week])
    _atomic_jldsave(cache_path; quant, observed, series = want_s, origin_max, levels = want_l)
    println("fan quantiles: wrote cache ", cache_path)
    return (; quant, observed)
end

"""
    plot_scenario_fans_over_time(fq, grid; series = FAN_SERIES, every = 4, ncol = 2, yscale = :log10, ycap, title_suffix) -> Plot

One panel per age group (all ages, then the CIS bins), filled row by row into `ncol` columns (8 panels ⇒ 4 × 2). Each panel shows the observed infections (black)
and, from every `every`-th origin in `fq.quant`, each series' median at h = 1..H (dots) with its 50% (dark) and
95% (light) bands. `fq` is `scenario_fan_quantiles_or_load`'s result. The y-axis is capped at `ycap ×` the panel's
observed maximum (default 10 on the log scale, 3 on the linear one; `legend_pos` places the legend in the first panel) and, on the log scale, floored at its minimum ÷
`ycap`; 95% bands crossing either limit are COUNTED in the title.
"""
function plot_scenario_fans_over_time(fq, grid; series = FAN_SERIES, every::Integer = 4, ncol::Integer = 2,
                                      yscale::Symbol = :log10, legend_pos = :bottomright,
                                      ycap::Real = (yscale == :identity ? 3.0 : 10.0),
                                      title_suffix::AbstractString = "")
    @assert yscale in (:log10, :identity) "plot_scenario_fans_over_time: yscale must be :log10 or :identity"
    logy = yscale == :log10
    fmt(y) = y >= 1e6 ? @sprintf("%gM", y / 1e6) : y >= 1e3 ? @sprintf("%gk", y / 1e3) : @sprintf("%g", y)
    lo95, lo50, med, hi50, hi95 = _qcol.(FAN_LEVELS)
    origins = sort(unique(fq.quant.forecast_date))[1:every:end]
    H = maximum(fq.quant.horizon)
    wmin, wmax = first(origins) + Week(1), last(origins) + Week(H)
    ages = vcat("all", grid.LAB)
    nrow_ = cld(length(ages), ncol)
    panels = Plots.Plot[]
    for (r, lab) in enumerate(ages)
        o = sort(@subset(fq.observed, :age_group .== lab, :week .>= wmin, :week .<= wmax), :week)
        @assert o.week == collect(wmin:Week(1):wmax) "plot_scenario_fans_over_time: observed weeks have gaps for $lab"
        @assert !logy || all(>(0), o.observed) "plot_scenario_fans_over_time: a non-positive observation for $lab on a log axis"
        qd = @subset(fq.quant, :age_group .== lab, in.(:forecast_date, Ref(origins)), in.(:series, Ref(first.(series))))
        @assert nrow(qd) == length(series) * length(origins) * H "plot_scenario_fans_over_time: missing fans for $lab"
        # the caps; the axis shrinks to the data inside them
        ymax = ycap * maximum(o.observed)
        ymin = logy ? minimum(o.observed) / ycap : 0.0
        ytop = min(ymax, max(maximum(o.observed), maximum(filter(isfinite, qd[!, hi95]))))
        ybot = logy ? max(ymin, min(minimum(o.observed), minimum(filter(isfinite, qd[!, lo95])))) : 0.0
        ylim = logy ? (ybot / 1.15, 1.15 * ytop) : (0.0, 1.05 * ytop)
        n_off = count(d -> any(>(ymax), d[!, hi95]) || (logy && any(<(ymin), d[!, lo95])),
                      groupby(qd, [:series, :forecast_date]))
        p = plot(; titlefontsize = 9, yscale, ylim, yformatter = fmt, minorgrid = logy,
                 legend = (r == 1 ? legend_pos : false),   # default: empty on the log axis, autumn 2021 runs near the top
                 legendfontsize = 7, xtickfontsize = 7, ytickfontsize = 7,
                 xlabel = (r > (nrow_ - 1) * ncol ? "week" : ""),
                 title = (lab == "all" ? "all ages" : "age $lab") *
                         (n_off > 0 ? "  [$(n_off) 95% band(s) cut at the y-axis limits]" : ""))
        plot!(p, week_mid.(o.week), o.observed; color = :black, lw = 1.4, marker = :circle, ms = 2.2,
              msc = :black, label = "observed")
        clip(v) = max.(v, ylim[1])   # a log axis cannot draw a band edge at or below 0
        for (s, slab, colr) in series
            for (j, origin) in enumerate(origins)
                d = sort(@subset(qd, :series .== s, :forecast_date .== origin), :horizon)
                @assert d.horizon == collect(1:H) "plot_scenario_fans_over_time: $s at $origin lacks a horizon"
                x = week_mid.(origin .+ Week.(d.horizon))
                m = d[!, med]
                plot!(p, x, m; ribbon = (m .- clip(d[!, lo95]), d[!, hi95] .- m), color = colr,
                      fillalpha = 0.15, lw = 0, label = "")
                plot!(p, x, m; ribbon = (m .- clip(d[!, lo50]), d[!, hi50] .- m), color = colr,
                      fillalpha = 0.35, lw = 1.0, marker = :circle, ms = 2.5, msc = colr,
                      label = (j == 1 ? slab : ""))
            end
        end
        plot!(p, week_mid.(o.week), o.observed; color = :black, lw = 1.4, label = "")   # again, on top of the fans
        shade_periods!(p, wmin, wmax; labels = (r <= ncol), fontsize = 6)   # after the Date series; labels on the top row
        push!(panels, p)
    end
    return plot(panels...; layout = (nrow_, ncol), size = (850 * ncol, 330 * nrow_),
                plot_title = "18j — weekly infections ($(logy ? "log" : "linear") scale) and h1–$(H) forecasts from " *
                             "every $(every)th origin (dots = median, bands = 50% / 95%)$(title_suffix)",
                plot_titlefontsize = 10)
end
