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

# The phase bands are 17j's `PERIODS_MAIN` (`shade_periods_main!`, `17j_viz_utils.jl`), the user's table.

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
    panels, leg = _scenario_dots_panels(rk; horizons, scale)
    n, leg_width = length(panels), 250
    width = n * panel_width + leg_width
    return plot(panels..., leg; layout = Plots.grid(1, n + 1; widths = [fill(panel_width / width, n); leg_width / width]), size = (width, 520),
                plot_title = "18j — mean WIS relative to no contact – no variant (= 1) at horizons $(join(horizons, ", ", " and "))$(title_suffix)",
                plot_titlefontsize = 10)
end

"""
    _scenario_dots_panels(rk; horizons = (1, 4), scale = WIS_SCALE, titles = nothing) -> (panels, legend_cell)

The panels of `plot_scenario_dots` (one per horizon) and its legend cell, lifted out 2026-11-11 so
`plot_summary_pre_omicron` can place them in its own layout. `titles` overrides the "horizon h" titles.
"""
function _scenario_dots_panels(rk::DataFrame; horizons = (1, 4), scale::AbstractString = WIS_SCALE, titles = nothing,
                               frozen_msw::Real = 1.5)   # the open diamond's rim (the publication summary: thinner)
    models = [NULL_LABEL; scenario_bar_contact_order(rk).order]
    groups = (first.(BAR_CONTACT_ROWS)..., first.(BAR_NULL_ROWS)...)   # frozen first: drawn underneath
    n_dots = length(BAR_NULL_ROWS) + (length(models) - 1) * length(BAR_CONTACT_ROWS)
    y = Dict(m => -i for (i, m) in enumerate(models))
    style(sc) = (c = SCENARIO_STYLE[sc]; occursin("frozen", sc) ?
                 (; marker = c[3], ms = 9, markercolor = :white, markerstrokecolor = c[1], markerstrokewidth = frozen_msw) :
                 (; marker = c[3], ms = 5.5, markercolor = c[1], markerstrokecolor = c[1], markerstrokewidth = 0.5))
    ys = [y[m] for m in models]
    dots = scenario_dots_frame(rk; horizons)
    lo, hi = extrema(dots.rel_wis)
    pad = 0.08 * (hi - lo)
    panels = Plots.Plot[]
    for (k, h) in enumerate(horizons)
        d = @subset(dots, :horizon .== h, in.(:model, Ref(models)), in.(:scenario, Ref(groups)))
        @assert nrow(d) == n_dots "plot_scenario_dots: h$h has $(nrow(d)) of $n_dots dots"
        p = plot(; title = (titles === nothing ? "horizon $h" : titles[k]), titlefontsize = 10, xlabel = "relative WIS ($scale)", legend = false,
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
    return panels, leg
end

"The publication summary's text sizes (user, 2026-11-18: larger than `PUB_FONT`, on a 1700 × 1000 canvas)."
const SUMMARY_FONT = (; title = 17, guide = 17, tick = 15, legend = 15, phase = 10)
"The open diamonds' rim in the publication summary (user: thinner than the dot plot's 1.5)."
const SUMMARY_FROZEN_MSW = 0.8

"Panels C–E's line style for scenario `sc` (and the summary's legend): `SCENARIO_STYLE` colour; frozen dashed + open diamond."
function _usage_line_style(sc::AbstractString)
    c = SCENARIO_STYLE[sc][1]; fz = occursin("frozen", sc)
    return (; color = c, lw = 1.8, ls = (fz ? :dash : :solid), marker = (fz ? :diamond : :circle), ms = (fz ? 5.5 : 4),
            markercolor = (fz ? :white : c), markerstrokecolor = c, markerstrokewidth = (fz ? SUMMARY_FROZEN_MSW : 0))
end

"""
    _usage_coverage_bias_panels(rk, model, cfg; scale = WIS_SCALE) -> Vector{Plot}

The PUBLICATION summary's panels C–E (2026-11-18, user: "a different model use for hweibull|nbhd, consistent with panels A
and B"): 50% coverage, 90% coverage and bias by horizon of ONE contact `model` under its four data usages (`BAR_CONTACT_ROWS`)
plus the two no-contact series (`BAR_NULL_ROWS`), read from 17j's pre-Omicron ranking `rk`. Each series in its panel-A/B
style — `SCENARIO_STYLE` colour, frozen as a DASHED line with open diamonds, the rest solid with filled circles — so A/B's
legend cell serves all five panels. Asserts every series has every horizon.
"""
function _usage_coverage_bias_panels(rk::DataFrame, model::AbstractString, cfg; scale::AbstractString = WIS_SCALE)
    hs = collect(cfg.horizons)
    ser = vcat([(NULL_LABEL, sc) for (sc, _) in BAR_NULL_ROWS], [(model, sc) for sc in first.(BAR_CONTACT_ROWS)])
    d = @subset(rk, :scale .== scale)
    panels = Plots.Plot[]
    for (col_, yl, ref, ylab) in ((:interval_coverage_50, (0, 1), 0.5, "coverage"), (:interval_coverage_90, (0, 1), 0.9, "coverage"),
                                  (:bias, (-1, 1), 0.0, "bias"))
        p = plot(; xlabel = "horizon (weeks)", ylabel = ylab, ylim = yl, xticks = hs, legend = false)
        hline!(p, [ref]; color = :gray, ls = :dot, label = "")
        for (m, sc) in ser
            r = sort(@subset(d, :model .== m, :scenario .== sc), :horizon)
            @assert r.horizon == hs "_usage_coverage_bias_panels: $m / $sc lacks a horizon"
            plot!(p, r.horizon, r[!, col_]; label = "", _usage_line_style(sc)...)
        end
        push!(panels, p)
    end
    return panels
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
    plot_scenario_age_heatmap(rel, rk, grid; horizons = (1, 4), cap, title_suffix, scale, group_by = :model,
                              publication = false) -> Plot

One row per horizon, one column: `rel_wis` (`scenario_age_rel_wis`) by age group (y, youngest at the bottom)
and series (x). One colour scale, white at 1, blue below and red above. Every cell prints its value, so `cap` (the
largest distance from 1 the colours resolve; default the data's) never hides one. The two no-contact columns come first;
then the contact series in blocks, separated by lines and headed above:
- `group_by = :model` — a block per contact model (in `scenario_bar_contact_order`), the four data usages inside
  (the order of the bars; the scenario on the tick);
- `group_by = :usage` (2026-11-18, user) — a block per data usage, contact – variant, contact only, contact (frozen) –
  variant, contact (frozen) only, the contact models inside (same model order; the model on the tick).
`publication = true` (2026-11-18, user): `_pub_fonts!`/`_pub_head!` panel heads A, B…, x tick labels under every
panel, cell values at 10 pt (white on the darker cells), block headings at 14 pt, colour-bar title "relative WIS" (user
 dropped the cap note) and no figure title.
"""
function plot_scenario_age_heatmap(rel::DataFrame, rk::DataFrame, grid; horizons = (1, 4), cap = nothing,
                                   title_suffix::AbstractString = "", scale::AbstractString = WIS_SCALE,
                                   group_by::Symbol = :model, publication::Bool = false)
    @assert group_by in (:model, :usage) "plot_scenario_age_heatmap: group_by must be :model or :usage"
    order = scenario_bar_contact_order(rk).order
    if group_by == :model
        rows = dropmissing(scenario_bar_rows(order), :scenario)
        rows.group = [m == NULL_LABEL ? "" : _bar_short(m) for m in rows.model]
    else
        nul = [(; tick, model = NULL_LABEL, scenario = sc, group = "") for (sc, tick) in BAR_NULL_ROWS]
        use = [(; tick = _bar_short(m), model = m, scenario = sc, group = lab)
               for (sc, lab) in reverse(collect(BAR_CONTACT_ROWS)) for m in order]   # contact – variant first
        rows = DataFrame(vcat(nul, use))
    end
    name = Dict((r.model, r.scenario) => r.series for r in eachrow(rk))
    series = [name[(r.model, r.scenario)] for r in eachrow(rows)]
    @assert allunique(series) && length(series) == length(unique(rk.series)) "plot_scenario_age_heatmap: a series is missing or repeated"
    headings = [(g, mean(findall(==(g), rows.group))) for g in unique(rows.group) if g != ""]
    breaks = [j + 0.5 for j in 1:(nrow(rows) - 1) if rows.group[j] != rows.group[j + 1]]
    A, n = grid.N, length(series)
    val = Dict((r.series, r.horizon, r.age_group) => r.rel_wis for r in eachrow(rel))
    d = cap === nothing ? maximum(abs.(rel.rel_wis .- 1)) : cap
    capped = cap === nothing ? "" : "; colours capped at 1 ± $(cap)"
    F = SUMMARY_FONT   # (user) every text larger than `PUB_FONT`: ticks 15, axis and panel titles 17
    panels = Plots.Plot[]
    for (k, h) in enumerate(horizons)
        M = [val[(s, h, grid.LAB[a])] for a in 1:A, s in series]
        lab_x = publication || k == length(horizons)
        p = heatmap(1:n, 1:A, M; c = cgrad(:balance), clims = (1 - d, 1 + d),   # cmocean, blue < 1 < red (user)
                    title = "horizon $h",
                    # publication: the age labels and title are hand-drawn after the layout (below), close to the axis
                    titlefontsize = 10, ylabel = (publication ? "" : "age group"),
                    yticks = (1:A, publication ? fill("", A) : grid.LAB), ytickfontsize = 8,
                    xticks = (1:n, lab_x ? rows.tick : fill("", n)), xtickfontsize = 7, xrotation = 45,
                    colorbar_title = "relative WIS",   # (user) the cap note dropped from the publication title too
                    grid = false, ylim = (0.5, A + 1))
        vline!(p, breaks; color = :black, lw = 0.8, label = "")
        for a in 1:A, j in 1:n
            annotate!(p, j, a, text(@sprintf("%.2f", M[a, j]), publication ? 11 : 5, abs(M[a, j] - 1) > (publication ? 0.5 : 0.8) * d ? :white : :black,   # (user) larger cell values; white on the darker cells
                                    (publication ? (PUB_FAMILY,) : ())...))
        end
        for (lab, x) in headings
            annotate!(p, x, A + 0.75, text(lab, publication ? 16 : 8, :center, (publication ? (PUB_FAMILY,) : ())...))   # (user) larger block headings
        end
        if publication
            _pub_fonts!(p; font = F)
            plot!(p; colorbar_titlefontsize = F.guide, colorbar_titlefontfamily = PUB_FAMILY,
                  colorbar_tickfontsize = F.tick, colorbar_tickfontfamily = PUB_FAMILY)
            _pub_head!(p, k, "Mean WIS ($(scale) scale) relative to no contact – no variant, by age, horizon $(h)"; font = F, y = 1.03, dx = 0.025)
        end
        push!(panels, p)
    end
    if publication
        fig = plot(panels...; layout = (length(panels), 1), size = (1600, 660 * length(panels)),   # wide enough for the 16 pt block headings
                   # the 45° x labels reach left of the first column and below the panel
                   left_margin = 33Plots.mm, right_margin = 4Plots.mm, top_margin = 9Plots.mm, bottom_margin = 10Plots.mm)
        # (user) the age labels 3.2 mm from the axis, as in the summary figure: GR puts tick labels 1.5% of the FIGURE's
        # width away (~10 mm here) — so they are drawn by hand from the computed plot areas
        Plots.prepare_output(fig)
        gap = 3.2; labw = maximum(length.(grid.LAB)) * 0.6 * F.tick * 0.254   # mm (≈ 0.6 em per character)
        for k in eachindex(panels)
            pa = Plots.plotarea(fig[k]); w = Plots.width(pa).value
            yr(y) = (y - 0.5) / (A + 0.5)   # the y axis runs 0.5 … A + 1
            for a in 1:A
                annotate!(fig, (-gap / w, yr(a)), text(grid.LAB[a], F.tick, :right, :vcenter, PUB_FAMILY); subplot = k)
            end
            annotate!(fig, (-(gap + labw + 13) / w, yr((A + 1) / 2)),   # clear of the labels (measured)
                      text("age group", F.guide, :center, :center, PUB_FAMILY; rotation = 90); subplot = k)
        end
        return fig
    end
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
the user's `PERIODS_MAIN` (17j). Right: per-origin coverage (`scenario_coverage_by_origin`), same bands, with `nominal` dashed; it
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
        shade_periods_main!(l, dmin, dmax; fontsize = 6)
        r = _over_time_panel(c, :coverage; title = "horizon $h — $(round(Int, 100nominal))% coverage",
                             ylabel = "share of $(grid.N) age groups covered", ylim = (0, 1.03))
        hline!(r, [nominal]; color = :grey30, ls = :dash, lw = 1, label = "")   # after the dates, see 14j §7
        shade_periods_main!(r, dmin, dmax; labels = false)
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
# The second figure adds negbin|mean with contact – variant, in the palette's third slot, and (2026-11-16, user request)
# the null model, no contact – no variant, in the palette's FOURTH slot, yellow (user's choice, replacing grey).
# ⚠ The validator FAILS yellow beside the orange series when every pair overlaps, as fans do (normal-vision ΔE 13.7
# < 15): the legend label carries the identity. Its series list contains FAN_SERIES, so the notebook caches this one
# list and draws both figures from it.
const FAN_SERIES_NEGBIN = (FAN_SERIES..., ("unweighted-negbin|mean-variant", "negbin|mean: contact – variant", "#1baf7a"),
                           (NULL_LABEL, "no contact (null)", "#eda100"))
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
`ycap`; 95% bands crossing either limit are COUNTED in the title. `bands = false` (2026-11-16) draws the medians only,
and then the medians set the y-range and are what the caps count.
"""
function plot_scenario_fans_over_time(fq, grid; series = FAN_SERIES, every::Integer = 4, ncol::Integer = 2,
                                      yscale::Symbol = :log10, legend_pos = :bottomright,
                                      ycap::Real = (yscale == :identity ? 3.0 : 10.0),
                                      title_suffix::AbstractString = "", bands::Bool = true)
    @assert yscale in (:log10, :identity) "plot_scenario_fans_over_time: yscale must be :log10 or :identity"
    logy = yscale == :log10
    origins = sort(unique(fq.quant.forecast_date))[1:every:end]
    H = maximum(fq.quant.horizon)
    ages = vcat("all", grid.LAB)
    nrow_ = cld(length(ages), ncol)
    panels = Plots.Plot[]
    for (r, lab) in enumerate(ages)
        push!(panels, _fan_panel(fq, lab, series, origins; H, yscale, ycap, bands,
                                 legend = (r == 1 ? legend_pos : false),   # default: empty on the log axis, autumn 2021 runs near the top
                                 xlabel = (r > (nrow_ - 1) * ncol ? "week" : ""),
                                 title = (lab == "all" ? "all ages" : "age $lab"), phase_labels = (r <= ncol)))
    end
    return plot(panels...; layout = (nrow_, ncol), size = (850 * ncol, 330 * nrow_),
                plot_title = "18j — weekly infections ($(logy ? "log" : "linear") scale) and h1–$(H) forecasts from " *
                             "every $(every)th origin (dots = median$(bands ? ", bands = 50% / 95%" : "; no intervals"))$(title_suffix)",
                plot_titlefontsize = 10)
end

"""
    _fan_panel(fq, lab, series, origins; H, yscale, ycap, bands, legend, xlabel, title, phase_labels, titlefontsize) -> Plot

ONE panel of `plot_scenario_fans_over_time` for age group `lab` (lifted out 2026-11-16 for `plot_fan_patterns`): the
observed infections, then each series' median at h = 1..H from every origin in `origins`, with the 50% / 95% bands when
`bands`, under the y-caps described there (bands ⇒ the 95% limits set the range, else the medians); cut fans are counted in
the title. A series tuple is `(series, label, colour)` with an optional 4th element, its line style (default `:solid`).
"""
function _fan_panel(fq, lab, series, origins; H::Integer, yscale::Symbol = :log10,
                    ycap::Real = (yscale == :identity ? 3.0 : 10.0), bands::Bool = true, legend = false,
                    xlabel = "", title = "", phase_labels::Bool = false, titlefontsize = 9,
                    obs_from::Union{Nothing,Date} = nothing, obs_to::Union{Nothing,Date} = nothing,
                    top_frac::Real = 1.0, phase_kw = (;))
    logy = yscale == :log10
    fmt(y) = y >= 1e6 ? @sprintf("%gM", y / 1e6) : y >= 1e3 ? @sprintf("%gk", y / 1e3) : @sprintf("%g", y)
    lo95, lo50, med, hi50, hi95 = _qcol.(FAN_LEVELS)
    wmin, wmax = first(origins) + Week(1), last(origins) + Week(H)
    # `obs_from`: start the observed series earlier, e.g. at the first origin's first fit week (needs those weeks in
    # `fq.observed`, see `extend_fan_observed`)
    obs_from === nothing || (@assert obs_from <= wmin; wmin = obs_from)
    # `obs_to`: run the observed series on past the last fan, e.g. to the last pre-Omicron origin's h4 target week
    obs_to === nothing || (@assert obs_to >= wmax; wmax = obs_to)
    o = sort(@subset(fq.observed, :age_group .== lab, :week .>= wmin, :week .<= wmax), :week)
    @assert o.week == collect(wmin:Week(1):wmax) "plot_scenario_fans_over_time: observed weeks have gaps for $lab"
    @assert !logy || all(>(0), o.observed) "plot_scenario_fans_over_time: a non-positive observation for $lab on a log axis"
    qd = @subset(fq.quant, :age_group .== lab, in.(:forecast_date, Ref(origins)), in.(:series, Ref(first.(series))))
    @assert nrow(qd) == length(series) * length(origins) * H "plot_scenario_fans_over_time: missing fans for $lab"
    # the caps; the axis shrinks to the data inside them
    ymax = ycap * maximum(o.observed)
    ymin = logy ? minimum(o.observed) / ycap : 0.0
    # without bands (`bands = false`) the medians alone set the range and are what the cap can cut
    up, dn = bands ? (hi95, lo95) : (med, med)
    ytop = min(ymax, max(maximum(o.observed), maximum(filter(isfinite, qd[!, up]))))
    ybot = logy ? max(ymin, min(minimum(o.observed), minimum(filter(isfinite, qd[!, dn])))) : 0.0
    ylim = logy ? (ybot / 1.15, 1.15 * ytop) : (0.0, 1.05 * ytop)
    # `top_frac` < 1: the data fill only that bottom fraction of the panel (in its own scale) — headroom for phase names
    top_frac < 1 && (ylim = logy ? (ylim[1], ylim[1] * (ylim[2] / ylim[1])^(1 / top_frac)) : (ylim[1], ylim[2] / top_frac))
    n_off = count(d -> any(>(ymax), d[!, up]) || (logy && any(<(ymin), d[!, dn])),
                  groupby(qd, [:series, :forecast_date]))
    p = plot(; titlefontsize, yscale, ylim, yformatter = fmt, minorgrid = logy,
             legend, legendfontsize = 7, xtickfontsize = 7, ytickfontsize = 7, xlabel,
             title = title * (n_off > 0 ? "  [$(n_off) $(bands ? "95% band(s)" : "median trajectorie(s)") cut at the y-axis limits]" : ""))
    plot!(p, week_mid.(o.week), o.observed; color = :black, lw = 1.4, marker = :circle, ms = 2.2,
          msc = :black, label = "observed")
    clip(v) = max.(v, ylim[1])   # a log axis cannot draw a band edge at or below 0
    for t in series
        s, slab, colr = t[1], t[2], t[3]
        ls = length(t) >= 4 ? t[4] : :solid
        for (j, origin) in enumerate(origins)
            d = sort(@subset(qd, :series .== s, :forecast_date .== origin), :horizon)
            @assert d.horizon == collect(1:H) "plot_scenario_fans_over_time: $s at $origin lacks a horizon"
            x = week_mid.(origin .+ Week.(d.horizon))
            m = d[!, med]
            if bands
                plot!(p, x, m; ribbon = (m .- clip(d[!, lo95]), d[!, hi95] .- m), color = colr,
                      fillalpha = 0.15, lw = 0, label = "")
                plot!(p, x, m; ribbon = (m .- clip(d[!, lo50]), d[!, hi50] .- m), color = colr,
                      fillalpha = 0.35, lw = 1.0, ls, marker = :circle, ms = 2.5, msc = colr,
                      label = (j == 1 ? slab : ""))
            else
                plot!(p, x, m; color = colr, lw = 1.6, ls, marker = :circle, ms = 2.5, msc = colr,
                      label = (j == 1 ? slab : ""))
            end
        end
    end
    plot!(p, week_mid.(o.week), o.observed; color = :black, lw = 1.4, label = "")   # again, on top of the fans
    shade_periods_main!(p, wmin, wmax; labels = phase_labels, fontsize = 6, phase_kw...)   # after the Date series
    return p
end

# ── Three comparisons, all ages, one row each (2026-11-16, user request) ─────────────────────────────────────
# Colour follows the SERIES's data, never the row: hweibull|nbhd contact – variant is orange and contact only blue in every
# row (as in `FAN_SERIES`). Row 1 adds no contact – variant in aqua: palette slots 1–3, which pass the validator on ALL
# pairs, needed because its bands overlap. No four-colour set passes all pairs, so row 2 is encoded instead as colour =
# data (variant orange, only blue) × style = contacts (conditional solid, frozen DASHED, 17j's frozen convention), and
# row 3 as colour = degree model (hweibull orange, negbin violet `#6250d6`; that pair passes all-pairs) × line style =
# NGM builder (nbhd solid, mean DOTTED). (Rows 2 and 3 were swapped on 2026-11-16, user request.) Dotted and dashed never share a row.
const FAN_PATTERNS = (
    ("Pattern 1 — hweibull|nbhd: dropping one data source (median, 50% / 95%)", true,
     (("weighted-hweibull|neighbourhood-variant", "contact – variant",    "#eb6834"),
      ("weighted-hweibull|neighbourhood",         "contact only",         "#2a78d6"),
      (NULL_VARIANT_LABEL,                        "no contact – variant", "#1baf7a"))),
    ("Pattern 2 — hweibull|nbhd: future vs frozen contacts (median only)", false,
     (("weighted-hweibull|neighbourhood-variant",                      "contact – variant",          "#eb6834", :solid),
      (series_name("weighted-hweibull|neighbourhood-variant", true),   "contact (frozen) – variant", "#eb6834", :dash),
      ("weighted-hweibull|neighbourhood",                              "contact only",               "#2a78d6", :solid),
      (series_name("weighted-hweibull|neighbourhood", true),           "contact (frozen) only",      "#2a78d6", :dash))),
    ("Pattern 3 — contact – variant, by contact model (median only)", false,
     (("weighted-hweibull|neighbourhood-variant", "hweibull|nbhd", "#eb6834", :solid),
      ("weighted-hweibull|mean-variant",          "hweibull|mean", "#eb6834", :dot),
      ("unweighted-negbin|neighbourhood-variant", "negbin|nbhd",   "#6250d6", :solid),
      ("unweighted-negbin|mean-variant",          "negbin|mean",   "#6250d6", :dot))))
"Every series any fan figure draws — the ONE list the fan cache is built for."
const FAN_PATTERN_SERIES = unique(vcat(collect(first.(FAN_SERIES_NEGBIN)),   # `collect`: vcat does not splat a Tuple
                                         [first(t) for (_, _, ser) in FAN_PATTERNS for t in ser]))

"""
    plot_fan_patterns(fq; patterns = FAN_PATTERNS, every = 4, yscale = :log10, title_suffix = "") -> Plot

One row per pattern `(title, bands, series)`, all ages only: `_fan_panel(fq, "all", …)` with the pattern's own legend and
y-range (bands ⇒ the 95% limits, else the medians). Phase names on the top row, the week label on the bottom one.
"""
function plot_fan_patterns(fq; patterns = FAN_PATTERNS, every::Integer = 4, yscale::Symbol = :log10,
                           title_suffix::AbstractString = "")
    origins = sort(unique(fq.quant.forecast_date))[1:every:end]
    H = maximum(fq.quant.horizon)
    n = length(patterns)
    panels = [_fan_panel(fq, "all", ser, origins; H, yscale, bands, legend = :bottomright,
                         xlabel = (k == n ? "week" : ""), title = ttl, phase_labels = (k == 1), titlefontsize = 10)
              for (k, (ttl, bands, ser)) in enumerate(patterns)]
    return plot(panels...; layout = (n, 1), size = (1300, 420 * n), left_margin = 6Plots.mm,
                plot_title = "18j — weekly infections, all ages ($(yscale == :log10 ? "log" : "linear") scale), and h1–$(H) " *
                             "forecasts from every $(every)th origin$(title_suffix)",
                plot_titlefontsize = 11)
end

# ── Variant-specific γ_SAR over time (2026-11-11) ─────────────────────────────────────────────
# The per-contact secondary attack rate of the variant models, one γ_SAR per variant category, read by
# 9j's `collect_variant_gamma` (read-only: `isfile` → `load`, never fits). 9j's `plot_variant_gamma` draws
# it on a log axis; this is the linear twin, one panel per model with its own y-scale.

const _DEGREE_OF = Dict(degree_label(dm) => dm for dm in (NegBinAgePair(), HurdleWeibullAgePair(), NoContactDegree()))
const _NGM_OF = Dict(ngm_label(nb) => nb for nb in (MeanNGM(), NeighbourhoodDegreeNGM(), DiagonalMeanNGM(), NullNGM()))

"""
    variant_gamma_combos(models) -> Vector{Tuple}

`(dm, VariantNGM(nb))` for each base label `"<degree>|<ngm>"` in `models`, asserting that the pair
round-trips to `"<base>-variant"`, the label its Stage-2 files carry.
"""
function variant_gamma_combos(models)
    map(models) do m
        d, n = split(m, "|")
        c = (_DEGREE_OF[d], VariantNGM(_NGM_OF[n]))
        @assert string(degree_label(c[1]), "|", ngm_label(c[2])) == m * "-variant"
        c
    end
end

"""
    plot_variant_gamma_over_time(df, models; ylim = (0.0, 1.0),
                                 free = ("weighted-hweibull|mean", "unweighted-negbin|neighbourhood"),
                                 ycap = 2.5, title_suffix = "") -> Plot

One panel per base label in `models` (in that order), x = forecast origin, one line per variant category
(median + 90% band, `VARIANT_COLS`) on the LEFT axis, and each category's genome share in the origin week
(`share_t0`, dashed, same colour) on a RIGHT axis in PERCENT, 0–100. `df` is `collect_variant_gamma`'s frame,
whose `model` column carries the `-variant` suffix. Gaps are origins at which a category had no renewal of
its own; a pooled category's share is the pooled share.

The left axis is LINEAR and fixed at `ylim` (user, 2026-11-11), except for the models in `free` — hweibull|mean,
whose γ runs past 1, and negbin|nbhd, whose γ ≈ 0.005 would sit on the floor of a 0–1 axis: there the axis
fits the bands (+5%) but stops at `ycap ×` the largest median. γ_SAR multiplies C*, whose units differ by
degree model and NGM builder, so even the fixed panels are not comparable with one another. Bands cut by an
axis limit are counted in the panel title.
"""
function plot_variant_gamma_over_time(df::DataFrame, models; ylim = (0.0, 1.0),
                                      free = ("weighted-hweibull|mean", "unweighted-negbin|neighbourhood"),
                                      ycap::Real = 2.5, title_suffix::AbstractString = "")
    nr, nc = panel_grid(length(models))
    built = [_variant_gamma_panel(df, m; ylim, free, ycap, legend = (k == 1 ? :topleft : false),
                                  ylabel = ((k - 1) % nc == 0 ? "γ_SAR,v (solid)" : ""), share_key = (k == 1))
             for (k, m) in enumerate(models)]
    fig = plot(first.(built)...; layout = (nr, nc), size = (600 * nc, 380 * nr), left_margin = 6Plots.mm,
               right_margin = 10Plots.mm, bottom_margin = 6Plots.mm,
               plot_title = "18j — variant-specific secondary attack rate γ_SAR,v (h = 1, median + 90%) and variant " *
                            "share at the origin$(title_suffix)",
               plot_titlefontsize = 10)
    _add_share_axes!(fig, last.(built); ylabel_at = [k % nc == 0 || k == length(models) for k in eachindex(models)])
    return fig
end

"""
    _variant_gamma_panel(df, m; ylim, free, ycap, legend, ylabel, share_key, title = default, xlabel = "") -> (panel, shares)

ONE γ_SAR,v panel of `plot_variant_gamma_over_time` for base label `m` (lifted out 2026-11-11 for the merged
18j figure): the y-axis rule (`ylim`, or for `m ∈ free` the bands +5% capped at `ycap ×` the largest median),
one NaN-padded median + 90% line per category, and the cut-band count in the title. `shares` is what
`_add_share_axes!` draws on the right axis; `share_key` adds that axis's legend entry to this panel.
"""
function _variant_gamma_panel(df::DataFrame, m; ylim, free, ycap::Real, legend, ylabel, share_key::Bool,
                              title = nothing, xlabel = "", titlefontsize = 9, scale::Real = 1.0)
    cats = vcat([VARIANT_RESIDUAL], COG_GROUPS[2:end])
    d = @subset(df, :model .== m * "-variant")
    @assert nrow(d) > 0 "_variant_gamma_panel: no rows for $m"
    # `scale` (2026-11-17): 100 draws γ in % (the publication figure); the returned shares stay fractions
    scale == 1 || (d = transform(d, [:gamma_med, :gamma_lo, :gamma_hi] .=> (x -> scale .* x); renamecols = false))
    yl = m in free ? (0.0, min(ycap * maximum(d.gamma_med), 1.05 * maximum(d.gamma_hi))) : ylim
    n_cut = count(>(yl[2]), d.gamma_hi)
    ttl = (title === nothing ? _bar_short(m) * " – variant" : title) *
          (n_cut > 0 ? "  [$(n_cut) band(s) cut at the top]" : "")
    p = plot(; title = ttl, titlefontsize, ylim = yl, ylabel, xlabel,
             legend, legendfontsize = 7, xrotation = 45,
             xtickfontsize = 7, ytickfontsize = 7)
    origins = sort(unique(d.origin))
    sh = Tuple{String,Vector{Date},Vector{Float64}}[]
    for nm in cats
        s = @subset(d, :variant .== nm)
        nrow(s) == 0 && continue
        # NaN at the origins where the category is absent, so the line BREAKS there instead of bridging
        ix = Dict(o => i for (i, o) in enumerate(s.origin))
        pick(col) = [haskey(ix, o) ? s[ix[o], col] : NaN for o in origins]
        med, lo, hi = pick(:gamma_med), pick(:gamma_lo), min.(pick(:gamma_hi), yl[2])
        plot!(p, origins, med; ribbon = (med .- lo, hi .- med), fillalpha = 0.18,
              color = get(VARIANT_COLS, nm, :black), lw = 1.6, marker = :circle, ms = 2, msw = 0, label = nm)
        push!(sh, (nm, origins, pick(:share_t0)))
    end
    # legend key for the right axis (an invisible point; the shares themselves go on the twin)
    share_key && plot!(p, origins[1:1], [NaN]; color = :black, ls = :dash, lw = 1.0, label = "share at origin, % (right)")
    return (p, sh)
end

"""
    _add_share_axes!(fig, shares; ylabel_at, idx = eachindex(shares))

The variant shares (dashed) and their right-hand % axis (0–100) on subplots `idx` of the COMPOSED `fig` — a twin made
inside a standalone panel is an inset subplot, which composing panels into a layout does not carry over reliably.
`ylabel_at[k]` says whether the k-th one carries the axis label.
⚠ The shares are drawn on the HOST, rescaled so 0–100% spans the host's 0…ymax, and the `twinx` only carries the tick
labels (2026-11-17). Drawn on the twin they were misplaced in x: the twin's plot area does not reserve the host's
left y-label room, so it starts further left and the same date lands earlier — a week's offset unseen while the axis
began a week before the first origin, ~7 weeks once it began at the first fit week. Asserts the host's y-axis starts
at 0, which is what makes the two scales line up.
"""
function _add_share_axes!(fig, shares; ylabel_at, idx = eachindex(shares))
    for (k, (i, sh)) in enumerate(zip(idx, shares))
        yl = Plots.ylims(fig[i])
        @assert yl[1] == 0 "_add_share_axes!: the host's y-axis must start at 0 for the share scale to line up"
        for (nm, x, y) in sh
            plot!(fig[i], x, yl[2] .* y; color = get(VARIANT_COLS, nm, :black), ls = :dash, lw = 1.0, label = "")
        end
        plot!(fig[i]; ylims = yl)                       # the shares must not move the host's axis
        q = twinx(fig[i])
        plot!(q; ylim = (0.0, 100.0), yticks = 0:20:100,
              ylabel = (ylabel_at[k] ? "share at origin, % (dashed)" : ""),
              legend = false, grid = false, ytickfontsize = 7)
        isempty(sh) || plot!(q, sh[1][2][1:1], [NaN]; label = "")   # an empty twin may not draw its axis
    end
    return fig
end

# ── Relative susceptibility and infectivity of the variant models (2026-11-11) ─────────────────
# 9j's `collect_transmission_structure` cannot read a `-variant` file (its `load_transmission_draws` takes
# `gamma_sar` as a vector, and it also loads every Stage-1 chain), so the susc/inf half is restated here
# from the Stage-2 file alone, with the SAME arithmetic: pop-weighted super-groups from `supergroup_split`
# divided per draw by the reference bin `cfg.ref_bin`. The stores have 9j's shape, so `plot_ratio` draws them.

"""
    susc_inf_ratio_stores(combos, origins, cfg; grid, h = 1, save_dir) -> (; susc, inf, sg_names, ref_lab, labels, frame)

READ-ONLY (`stage2_path` → assert `isfile` → `load`; nothing fits). For each `(dm, nb)` × origin, the median
and 5/95% of the per-draw susceptibility and infectivity super-group ratios to `cfg.ref_bin`, as
`Dict(label => (med, lo, hi))` of `nO × nG` matrices keyed by `"<degree>|<ngm>"` (9j's
`collect_transmission_structure` shape, for `plot_ratio`), plus the same in a long `frame` for the CSV.
Asserts the reference column is identically 1 and that no model's infectivity is pinned (`plot_ratio`
is called without `pinned`).
"""
function susc_inf_ratio_stores(combos, origins, cfg::FrameworkConfig; grid = cis_age_grid(), h::Integer = 1,
                               save_dir::AbstractString = "../dt_intermediate")
    ref = cfg.ref_bin
    sg_groups, sg_names = supergroup_split(grid, ref)
    nO, nG = length(origins), length(sg_groups)
    labels = [string(degree_label(dm), "|", ngm_label(nb)) for (dm, nb) in combos]
    mk() = Dict(l => (med = fill(NaN, nO, nG), lo = fill(NaN, nO, nG), hi = fill(NaN, nO, nG)) for l in labels)
    susc, inf = mk(), mk()
    rows = NamedTuple[]
    for ((dm, nb), lbl) in zip(combos, labels)
        pinned = true
        for (oi, o) in enumerate(origins)
            p = stage2_path(dm, nb, o, h; contacts = contacts_label(cfg), save_dir = save_dir)
            @assert isfile(p) "susc_inf_ratio_stores: missing $p"
            pooled = load(p, "pooled")
            pinned &= all(==(1.0), pooled.inf)
            for (q, V, dst) in (("susceptibility", pooled.susc, susc), ("infectivity", pooled.inf, inf))
                den = view(V, :, ref)
                @assert all(==(1.0), den) "susc_inf_ratio_stores: $lbl $o $q reference bin is not ≡ 1"
                r = aggregate_supergroups(V, grid.POP; groups = sg_groups) ./ den
                for g in 1:nG
                    m, lo, hi = median(r[:, g]), quantile(r[:, g], 0.05), quantile(r[:, g], 0.95)
                    dst[lbl].med[oi, g] = m; dst[lbl].lo[oi, g] = lo; dst[lbl].hi[oi, g] = hi
                    push!(rows, (; model = lbl, origin = o, h, quantity = q, age_group = sg_names[g],
                                   ratio_med = m, ratio_lo = lo, ratio_hi = hi))
                end
            end
        end
        @assert !pinned "susc_inf_ratio_stores: $lbl has infectivity pinned ≡ 1 — pass `pinned` to plot_ratio"
    end
    return (; susc, inf, sg_names, ref_lab = grid.LAB[ref], labels, frame = DataFrame(rows))
end

"`susc_inf_ratio_stores` for the `-variant` twins of the base labels in `models` (`variant_gamma_combos`)."
collect_variant_susc_inf(models, origins, cfg::FrameworkConfig; kw...) =
    susc_inf_ratio_stores(variant_gamma_combos(models), origins, cfg; kw...)

"Log-axis ticks at the powers of 2 inside `yl`, labelled as plain ratios (\"0.25\", \"1\", \"4\") rather than 10^x."
function _pow2_ticks(yl; frac::Bool = false)
    tk = filter(t -> yl[1] <= t <= yl[2], 2.0 .^ (-6:6))
    return (tk, [t >= 1 ? string(Int(t)) : frac ? "1/$(Int(1 / t))" : string(t) for t in tk])   # frac: "1/8" for 0.125
end

"""
    plot_variant_transmission_grid(vg, tr, models, origins; free, ycap, title_suffix) -> Plot

ONE figure (2026-11-11): rows γ_SAR,v · susceptibility · infectivity, one column per base label in `models`
(all `-variant`). Row 1 is `_variant_gamma_panel` with the variant shares on a right axis in %
(`_add_share_axes!`); rows 2–3 are 9j's `_ratio_panel` on `tr.susc` / `tr.inf` (`collect_variant_susc_inf`),
log axis with `_pow2_ticks`, one range per row from that row's medians (`_median_ylims`), so a wide band
can run off the panel; no prior band (user's choice). The γ row keeps the 0–1 axis except for `free` models. Column titles on the top row,
row labels on the first column, x label on the bottom row; legends: variants in the γ row, age groups in
the susceptibility row.
"""
function plot_variant_transmission_grid(vg::DataFrame, tr, models, origins;
                                        free = ("weighted-hweibull|mean", "unweighted-negbin|neighbourhood"),
                                        ycap::Real = 2.5, title_suffix::AbstractString = "")
    nc = length(models)
    lbls = [m * "-variant" for m in models]
    @assert all(in(tr.labels), lbls) "plot_variant_transmission_grid: susc/inf stores lack $(setdiff(lbls, tr.labels))"
    pri = nothing   # no prior 90% band (user, 2026-11-11)
    built = [_variant_gamma_panel(vg, m; ylim = (0.0, 1.0), free, ycap, legend = (k == 1 ? :topleft : false),
                                  ylabel = (k == 1 ? "γ_SAR,v (solid)" : ""), share_key = (k == 1), titlefontsize = 10)
             for (k, m) in enumerate(models)]
    panels = Plots.Plot[first.(built)...]
    for (r, (q, st)) in enumerate((("susceptibility", tr.susc), ("infectivity", tr.inf)))
        yl = _median_ylims(vcat([vec(st[l].med) for l in lbls]...); log = true)
        for (k, l) in enumerate(lbls)
            push!(panels, _ratio_panel(st, l, origins; gnames = tr.sg_names, ref_lab = tr.ref_lab, pri, yl,
                                       yticks = _pow2_ticks(yl), title = "",
                                       ylabel = (k == 1 ? "$q ratio to $(tr.ref_lab)" : ""),
                                       xlabel = (r == 2 ? "forecast origin" : ""),
                                       legend = (r == 1 && k == 1 ? :topleft : false), first_panel = (k == 1),
                                       marker_kw = (; msw = 0)))   # no black outline: it swamps the 2-px markers and legend swatches
        end
    end
    # ⚠ `save_show` → `pad_margins` OVERRIDES left/bottom margins on every subplot (8 / 12 mm by default); the notebook
    # passes `l = 16Plots.mm` there, because the infectivity row's "0.125" ticks push its label off the left edge.
    fig = plot(panels...; layout = (3, nc), size = (540 * nc, 380 * 3), left_margin = 16Plots.mm,
               right_margin = 10Plots.mm, bottom_margin = 12Plots.mm,
               plot_title = "18j — variant models: γ_SAR,v with variant share (%, right), and susceptibility / " *
                            "infectivity ratio to $(tr.ref_lab) (log) (h = 1, median + 90%)$(title_suffix)",
               plot_titlefontsize = 11)
    _add_share_axes!(fig, last.(built); ylabel_at = [k == nc for k in 1:nc], idx = 1:nc)
    return fig
end

# ── Summary figure: scenario dots + coverage/bias (left) and the 19r Tweedie forest plot (right) (2026-11-11) ──
# The forest plot is 19r's `19r_fixed_effects_tweedie.png` redrawn with Plots from `res/19r_fixed_effects.csv`
# (19r's ggplot figure is untouched), with the DERIVED block moved to the middle and cut to the two
# future-vs-frozen contrasts (user's choice).

"Blocks of the forest plot, top to bottom, and their terms (`X_TERMS` / `DERIVED_TERMS` in `19r_utils.R`)."
const FOREST_BLOCKS = ("contact – variant usage" => ["ctc_future", "ctc_frozen", "variant", "ctc_future_variant", "ctc_frozen_variant"],
                       "derived"                 => ["future vs frozen (variant)", "future vs frozen (no variant)"],
                       "contact model"           => ["duration", "nbhd", "duration_nbhd", "diagonal"])
"19r's horizon colours (light → dark), as in `19r_fixed_effects_tweedie.png`."
const FOREST_H_COLS = ("#7fa7c9", "#3d8bd1", "#1c4f96", "#08306b")

"""
    tweedie_forest_frame(eff; family = "tweedie", blocks = FOREST_BLOCKS, sort_h = 4) -> DataFrame

The rows of the forest plot from 19r's `19r_fixed_effects.csv` (`eff`): one row per term × horizon with
`block`, `term`, `horizon`, `ratio_median/lo/hi` (median and 90% interval of exp(b)) and `rank` — the term's
position within its block, sorted by the `sort_h` median, largest WIS reduction first (19r's rule). Asserts
every listed term has every horizon.
"""
function tweedie_forest_frame(eff::DataFrame; family::AbstractString = "tweedie", blocks = FOREST_BLOCKS, sort_h::Integer = 4)
    d = @subset(eff, :family .== family)
    hs = sort(unique(d.horizon))
    out = DataFrame[]
    for (bname, terms) in blocks
        b = @subset(d, in.(:term, Ref(terms)))
        for t in terms
            @assert sort(@subset(b, :term .== t).horizon) == hs "tweedie_forest_frame: $t lacks a horizon"
        end
        ord = sort(@subset(b, :horizon .== sort_h), :ratio_median).term
        b = transform(b, :term => ByRow(t -> findfirst(==(t), ord)) => :rank)
        b.block .= bname
        push!(out, select(b, :block, :term, :horizon, :ratio_median, :ratio_lo, :ratio_hi, :rank))
    end
    return sort!(vcat(out...), [order(:block; by = bl -> findfirst(p -> first(p) == bl, collect(blocks))), :rank, :horizon])
end

"""
    _tweedie_forest_panel(f; title, xlabel, ticks) -> Plot

ONE forest-plot subplot from `tweedie_forest_frame`: a row per term (block by block, a labelled header row
and a grey separator between blocks), the horizons dodged top (h1) → bottom (h4) as median + 90% interval in
`FOREST_H_COLS`, log x-axis with plain-value ticks, dashed line at 1 (< 1 = lowers WIS).
"""
function _tweedie_forest_panel(f::DataFrame; title = "19r — Tweedie: effects on WIS by horizon",
                               xlabel = "multiplicative effect on log-scale WIS (median, 90% interval; < 1 = better)",
                               ticks = [0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75],
                               header_font = (8, Plots.default(:fontfamily)))   # (size, family) of the block headers
    hs = sort(unique(f.horizon))
    @assert length(hs) <= length(FOREST_H_COLS) "_tweedie_forest_panel: more horizons than colours"
    off = Dict(h => 0.3 - 0.6 * (i - 1) / max(length(hs) - 1, 1) for (i, h) in enumerate(hs))   # h1 on top
    # y positions: a header row per block, then its terms, going down
    ypos = Dict{String,Float64}(); headers = Tuple{String,Float64}[]; seps = Float64[]
    y = 0.0
    blocks = unique(f.block)
    for (k, bl) in enumerate(blocks)
        k > 1 && (push!(seps, y + 0.4); y -= 0.3)
        push!(headers, (bl, y)); y -= 1
        for t in unique(sort(@subset(f, :block .== bl), :rank).term)
            ypos[t] = y; y -= 1
        end
    end
    xl = (min(minimum(f.ratio_lo), 1.0) / 1.04, max(maximum(f.ratio_hi), 1.0) * 1.04)
    tk = filter(t -> xl[1] <= t <= xl[2], ticks)
    terms = collect(keys(ypos))
    p = plot(; title, titlefontsize = 10, xlabel, xguidefontsize = 9, xscale = :log10, xlim = xl,
             xticks = (tk, string.(tk)), yticks = ([ypos[t] for t in terms], terms), ytickfontsize = 8,
             ylim = (y + 0.4, 0.6), legend = :topright, legendfontsize = 7, legendtitle = "horizon",
             legendtitlefontsize = 7, grid = :x)
    vline!(p, [1.0]; color = :grey40, ls = :dash, lw = 1, label = "")
    isempty(seps) || hline!(p, seps; color = :grey75, lw = 0.8, label = "")
    for (i, h) in enumerate(hs)
        s = @subset(f, :horizon .== h)
        yy = [ypos[t] + off[h] for t in s.term]
        xs = reduce(vcat, [[lo, hi, NaN] for (lo, hi) in zip(s.ratio_lo, s.ratio_hi)])
        plot!(p, xs, reduce(vcat, [[v, v, NaN] for v in yy]); color = FOREST_H_COLS[i], lw = 1.6, label = "")
        scatter!(p, s.ratio_median, yy; color = FOREST_H_COLS[i], ms = 3.5, msw = 0, label = "h$h")
    end
    for (bl, yh) in headers
        annotate!(p, xl[1] * 1.01, yh, text(bl, header_font[1], :left, header_font[2]))   # the figure's font, not GR's sans default (a colour here came out PURPLE in Helvetica)
    end
    return p
end

"""
    _tweedie_forest_blocks(f; xlabel, ticks, font = PUB_FONT) -> (; blocks, spans)

The PUBLICATION forest plot (2026-11-18, user: "similar to `19r_fixed_effects_tweedie.png`"): the rows of
`tweedie_forest_frame` as ONE framed sub-panel per block, the block name in bold at the panel's top right (user: inside the
panel, not 19r's side strips), sharing one log x axis whose tick labels and title appear under the last block only, a y grid
line per term, and the horizon legend inside the top block, bottom right (user). Same marks as
`_tweedie_forest_panel` (h1 on top, `FOREST_H_COLS`). `spans[k]` = block k's y-axis length in term rows: give each block a
plot area ∝ its span and the horizons are stacked equally tightly in every block (user).
"""
function _tweedie_forest_blocks(f::DataFrame; ticks = [0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75], font = PUB_FONT,
                                xlabel = "multiplicative effect on log-scale WIS (median, 90% interval; < 1 = better)")
    hs = sort(unique(f.horizon))
    @assert length(hs) <= length(FOREST_H_COLS) "_tweedie_forest_blocks: more horizons than colours"
    off = Dict(h => 0.3 - 0.6 * (i - 1) / max(length(hs) - 1, 1) for (i, h) in enumerate(hs))   # h1 on top
    xl = (min(minimum(f.ratio_lo), 1.0) / 1.04, max(maximum(f.ratio_hi), 1.0) * 1.04)
    tk = filter(t -> xl[1] <= t <= xl[2], ticks)
    names = unique(f.block)
    blocks = Plots.Plot[]; spans = Float64[]
    top = -0.4   # NO extra headroom for the block name (user): it sits in the empty top-right corner
    for (k, bl) in enumerate(names)
        b = @subset(f, :block .== bl)
        terms = unique(sort(b, :rank).term)
        ypos = Dict(t => -Float64(i) for (i, t) in enumerate(terms))
        last = k == length(names)
        p = plot(; framestyle = :axes, xscale = :log10, xlim = xl,   # (user) left + bottom axes only, as in A and B
                 xticks = (tk, last ? string.(tk) : fill("", length(tk))), xlabel = (last ? xlabel : ""),
                 # the term names are ANNOTATED left of the axis, not tick labels: tick labels size each panel's left
                 # edge to its own longest name, so the three blocks' x axes would not line up (the caller sets ONE margin)
                 yticks = ([ypos[t] for t in terms], fill("", length(terms))), ylim = (-length(terms) - 0.6, top),
                 grid = :x, legend = false)   # (user) ticks and x grid as in A and B; a dotted line per row, as A's
        hline!(p, [ypos[t] for t in terms]; color = :grey70, ls = :dot, lw = 0.8, label = "")
        vline!(p, [1.0]; color = :grey40, lw = 1, label = "")   # (user) the reference: A's solid grey line
        for (i, h) in enumerate(hs)
            s = @subset(b, :horizon .== h)
            yy = [ypos[t] + off[h] for t in s.term]
            xs = reduce(vcat, [[lo, hi, NaN] for (lo, hi) in zip(s.ratio_lo, s.ratio_hi)])
            plot!(p, xs, reduce(vcat, [[v, v, NaN] for v in yy]); color = FOREST_H_COLS[i], lw = 2, label = "")
            scatter!(p, s.ratio_median, yy; color = FOREST_H_COLS[i], ms = 4.5, msw = 0, label = "")
        end
        _pub_fonts!(p; font)
        for t in terms
            annotate!(p, xl[1] / 1.03, ypos[t], text(t, font.tick, :right, :vcenter, PUB_FAMILY))   # clear of the tick marks
        end
        annotate!(p, xl[2] / 1.012, top - 0.08, text(bl, font.legend, :right, :top, PUB_FAMILY_BOLD))   # the block name, top right
        push!(blocks, p); push!(spans, length(terms) + 0.6 + top)
    end
    # (user) the horizon legend INSIDE the top block, bottom RIGHT (empty there), drawn by hand: "horizon", then a line + dot
    # and "h…" per horizon, h1 on top; x at fractions of the log axis, y in term rows from the block's bottom
    b1 = blocks[1]; n1 = spans[1] - 0.6 + 0.4   # the first block's term count (span = n + 0.6 + top, top = −0.4)
    xat(f) = xl[1] * (xl[2] / xl[1])^f
    y0 = -n1 - 0.6 + 0.35; dy = 0.42
    annotate!(b1, xat(0.80), y0 + dy * length(hs), text("horizon", font.legend, :left, :vcenter, PUB_FAMILY))
    for (i, h) in enumerate(hs)
        yy = y0 + dy * (length(hs) - i)
        plot!(b1, [xat(0.80), xat(0.86)], [yy, yy]; color = FOREST_H_COLS[i], lw = 2, label = "")
        scatter!(b1, [xat(0.83)], [yy]; color = FOREST_H_COLS[i], ms = 4.5, msw = 0, label = "")
        annotate!(b1, xat(0.875), yy, text("h$h", font.legend, :left, :vcenter, PUB_FAMILY))
    end
    return (; blocks, spans)
end

"""
    _summary_legend_cell(font) -> Plot

The publication summary's ONE legend for A–E, drawn by hand (a GR legend's line sample is too short to show a dash): per
entry C–E's line (frozen dashed) through A/B's marker, then the label, centred in the cell.
"""
function _summary_legend_cell(font)
    leg = plot(; framestyle = :none, grid = false, xticks = false, yticks = false, legend = false, xlim = (0, 1), ylim = (0, 1))
    entries = (BAR_NULL_ROWS..., BAR_CONTACT_ROWS...)
    for (i, (sc, lab)) in enumerate(entries)
        yy = 0.5 + 0.13 * ((length(entries) + 1) / 2 - i)
        st = _usage_line_style(sc); fz = occursin("frozen", sc)
        # (user) the dash must read as a dash: GR's pattern on a short sample showed one dash a side, half hidden by the
        # marker — so a longer sample, and a frozen entry's dashes drawn as separate segments (`dash` long, `gap` apart)
        x0, x1 = 0.0, 0.22
        if fz
            dash, gap = 0.035, 0.022
            xs = Float64[]; ys = Float64[]
            for a in x0:(dash + gap):x1
                append!(xs, [a, min(a + dash, x1), NaN]); append!(ys, [yy, yy, NaN])
            end
            plot!(leg, xs, ys; color = st.color, lw = st.lw, ls = :solid, label = "")
        else
            plot!(leg, [x0, x1], [yy, yy]; color = st.color, lw = st.lw, label = "")
        end
        scatter!(leg, [(x0 + x1) / 2], [yy]; marker = st.marker, ms = (fz ? 9 : 5.5), markercolor = st.markercolor,
                 markerstrokecolor = st.markerstrokecolor, markerstrokewidth = (fz ? SUMMARY_FROZEN_MSW : 0.5), label = "")
        annotate!(leg, 0.26, yy, text(lab, font.legend, :left, :vcenter, PUB_FAMILY))
    end
    return leg
end

"""
    _plot_summary_publication(rk, eff, cfg; usage_model, W = 1700, H = 1000) -> Plot

`plot_summary_pre_omicron(…; publication = true)` (2026-11-18, user). Left: a 2 × 3 grid of equal cells — A, B (relative
WIS at h1, h4; `_scenario_dots_panels`), the legend (`_summary_legend_cell`); C, D, E (`_usage_coverage_bias_panels`) —
whose height is set so E's x axis lines up with F's (an empty cell below takes the rest). Right: F, the forest plot
(`_tweedie_forest_blocks`), each block's plot area ∝ its rows (a `Plots.grid`'s `heights` share out the PLOT AREAS, so the
horizons are equally tight in every block). Text in `SUMMARY_FONT`; heads A–F placed from the computed plot areas.
The layout is computed twice (`Plots.prepare_output`): the first pass measures where E and F end.
"""
function _plot_summary_publication(rk::DataFrame, eff::DataFrame, cfg::FrameworkConfig;
                                   usage_model::AbstractString = "weighted-hweibull|neighbourhood", W::Integer = 1700, H::Integer = 1000)
    font = SUMMARY_FONT
    um = _bar_short(usage_model)
    # (user) three LETTERED groups — A the two dot plots, B the three coverage/bias panels, C the forest plot — and a short
    # title above every panel
    titles = ["horizon 1", "horizon 4", "50% coverage", "90% coverage", "bias", ""]   # C: the letter only (user)
    ff = tweedie_forest_frame(eff)
    # subplot indices: the two dot plots, the legend, the three coverage/bias panels, the blank, then the forest blocks + legend
    iA, iB, iL, iC, iD, iE, iF = 1, 2, 3, 4, 5, 6, 8
    # (user) the y tick labels close to the axis: GR puts them 1.5% (× 1.5 with outward ticks) of the FIGURE's width away,
    # ~10 mm here, with no setting for it — so they and the y titles they push out are drawn by hand, `ygap` mm from the axis,
    # and each panel reserves its own left margin for them (widths ESTIMATED from the character count: `em` × the font's
    # size per character, 0.254 mm per pt — Helvetica letters ≈ 0.67 em, numbers ≈ 0.6 em)
    f_pad = 0.5    # term rows: the empty cell under F
    ygap = 3.2   # (user) as far from the axis as C's term names
    # Plots pads every panel 2 mm + (for y tick labels, even BLANK ones) 1 mm + 1% of the figure width — net that out of the
    # margin reserved for the hand-drawn labels (`_update_min_padding!` in Plots' GR backend)
    tickpad = 3.0 + 0.01 * W * 0.254
    labw(labs, em) = maximum(length.(labs)) * em * font.tick * 0.254
    fmt(v) = replace(@sprintf("%.1f", v), "-" => "−")
    cov_t = collect(0:0.2:1); bias_t = collect(-1:0.5:1)
    yhand = Dict(iC => (cov_t, fmt.(cov_t), "coverage", 0.6), iD => (cov_t, fmt.(cov_t), "coverage", 0.6),
                 iE => (bias_t, fmt.(bias_t), "bias", 0.6))
    bline = 10.0   # mm: B's heading line sits this far above its panel titles (user: "B hweibull|nbhd", then the titles)
    leg_font = 13  # (user) the A–B legend a little smaller than `font.legend`, so A's panels get the width
    leg_w = 84.0 * leg_font / 15   # mm: the legend's plot area — its longest label ("contact (frozen) – variant") and line sample
    function compose(s, wA, wB)
        # the panels are rebuilt on each pass: composing a figure takes over its panels' subplots
        dp, _ = _scenario_dots_panels(rk; horizons = (1, 4), titles = ["", ""], frozen_msw = SUMMARY_FROZEN_MSW)
        yhand[iA] = (dp[1][1][:yaxis][:ticks]..., "", 0.67)   # the model names of the dot plot
        cb = _usage_coverage_bias_panels(rk, usage_model, cfg)
        fp = _tweedie_forest_blocks(ff; font, xlabel = "multiplicative effect on log-scale WIS\n(median, 90% interval; < 1 = better)")
        foreach(p -> _pub_fonts!(p; font), (dp..., cb...))
        blank = plot(; framestyle = :none, grid = false, xticks = false, yticks = false, legend = false,
                     background_color_subplot = :transparent, background_color_inside = :transparent)
        fblank = plot(; framestyle = :none, grid = false, xticks = false, yticks = false, legend = false,
                      background_color_subplot = :transparent, background_color_inside = :transparent)
        nb = length(fp.blocks)
        l = Plots.grid(1, 2; widths = [0.65, 0.35])
        # rows: dot plots + legend (widths `wA`: the two dot plots equal, the legend as wide as its text); the three
        # coverage/bias panels (widths `wB`, set so their PLOT AREAS are equal — user); a blank row (ratio `s`) that lifts
        # them to F's x axis
        l[1, 1] = Plots.grid(3, 1; heights = [1, 1, s] ./ (2 + s))
        l[1, 1][1, 1] = Plots.grid(1, 3; widths = wA ./ sum(wA))
        l[1, 1][2, 1] = Plots.grid(1, 3; widths = wB ./ sum(wB))
        # F's blocks, then an EMPTY cell where its legend row was (the legend is inside the top block now): F keeps its
        # height and B's alignment target (user: do not stretch F)
        l[1, 2] = Plots.grid(nb + 1, 1; heights = vcat(fp.spans, f_pad) ./ (sum(fp.spans) + f_pad))
        fig = plot(dp..., _summary_legend_cell(merge(font, (; legend = leg_font))), cb..., blank, fp.blocks..., fblank; layout = l,
                   size = (W, H), left_margin = 4Plots.mm, right_margin = 3Plots.mm, top_margin = 2Plots.mm, bottom_margin = 2Plots.mm)
        for (i, (v, labs, ylab, em)) in yhand   # blank the real labels and title; reserve room for the hand-drawn ones
            plot!(fig[i]; yticks = (v, fill("", length(v))), ylabel = "",
                  left_margin = max(ygap + labw(labs, em) + (isempty(ylab) ? 0 : 10) + (i == iA ? 3 : 1) - tickpad, 0) * Plots.mm)
        end
        # A: three x ticks (five collide at this width; the line marks 1) and ONE axis title under both panels (below)
        for i in (iA, iB); plot!(fig[i]; xticks = [0.8, 1.2, 1.6], xlabel = ""); end
        # (user) wider panels from the white space between them: A's two panels nearly touch (B's blank tick labels still get
        # Plots' padding, which the negative margin takes back), the legend and B's panels lose their right margins
        plot!(fig[iA]; right_margin = 0Plots.mm); plot!(fig[iB]; left_margin = -2Plots.mm)
        plot!(fig[iL]; left_margin = 0Plots.mm, right_margin = 0Plots.mm)
        for i in (iC, iD); plot!(fig[i]; right_margin = 0Plots.mm); end
        for i in (iA, iB, iL); plot!(fig[i]; top_margin = 11Plots.mm, bottom_margin = 10Plots.mm); end   # incl. A's axis title
        # (user) more room between A and B, and B's heading line ("B hweibull|nbhd") above its panel titles
        for i in (iC, iD, iE); plot!(fig[i]; top_margin = (16 + bline) * Plots.mm, bottom_margin = 5Plots.mm); end
        plot!(fig[7]; top_margin = 0Plots.mm, bottom_margin = 0Plots.mm)
        plot!(fig[iF + nb]; top_margin = 0Plots.mm, bottom_margin = 0Plots.mm)
        for k in 1:nb
            # ONE left margin for the annotated term names, so the blocks' x axes line up; a block above the last
            # reserves room for its BLANK tick labels, which the negative bottom margin takes back; the last needs room
            # for the two-line axis title
            plot!(fig[iF - 1 + k]; left_margin = 60Plots.mm, top_margin = (k == 1 ? 11 : 1.5) * Plots.mm,
                  bottom_margin = (k == nb ? 12 : -4) * Plots.mm)
        end
        Plots.prepare_output(fig)
        return fig, nb
    end
    bottom(sp) = (pa = Plots.plotarea(sp); (pa.x0[2] + pa.a[2]).value)   # mm from the top (x0 = the top-left corner)
    pw(sp) = Plots.width(Plots.plotarea(sp)).value
    # (1) widths, from one measuring pass. Top row: the legend `leg_w` wide, the two dot plots sharing the rest equally.
    # Lower row: Plots sizes its plot areas as if each sat behind the column's widest left padding (A's labels), so C's
    # comes out wider by the padding it does not use (e) — ask for e less for C, and the three match.
    fig, nb = compose(0.25, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    Ptop = pw(fig[iA]) + pw(fig[iB]) + pw(fig[iL])
    @assert Ptop > leg_w + 60 "_plot_summary_publication: no room for the dot plots beside the legend"
    wA = [(Ptop - leg_w) / 2, (Ptop - leg_w) / 2, leg_w]
    e = pw(fig[iC]) - pw(fig[iD])
    T = (pw(fig[iC]) + pw(fig[iD]) + pw(fig[iE])) / 3
    wB = [T - e, T, T]
    # (2) height: the blank row's ratio s by secant until E's plot area ends where F's last block does
    offset(f, n) = bottom(f[iE]) - bottom(f[iF - 1 + n])
    s0 = 0.25; fig, nb = compose(s0, wA, wB); o0 = offset(fig, nb)
    s1 = 0.4;  fig, nb = compose(s1, wA, wB); o1 = offset(fig, nb)
    for _ in 1:6
        abs(o1) < 0.2 && break
        s2 = max(s1 - o1 * (s1 - s0) / (o1 - o0), 0.01)
        s0, o0 = s1, o1
        s1 = s2; fig, nb = compose(s1, wA, wB); o1 = offset(fig, nb)
    end
    # A's one x title, centred under both dot plots, where GR puts an axis title (~16 mm below the axis at this size)
    let pa = Plots.plotarea(fig[iA]), pb = Plots.plotarea(fig[iB])
        w = Plots.width(pa).value; h = Plots.height(pa).value
        xc = ((pb.x0[1] + pb.a[1]) - pa.x0[1]).value / 2 / w
        annotate!(fig, (xc, -16.0 / h), text("relative WIS (log)", font.guide, :center, :center, PUB_FAMILY); subplot = iA)
    end
    # the hand-drawn y tick labels and titles, in each plot area's relative coordinates
    for (i, (v, labs, ylab, em)) in yhand
        pa = Plots.plotarea(fig[i]); w = Plots.width(pa).value
        y0, y1 = Plots.ylims(fig[i])
        for (t, lab) in zip(v, labs)
            annotate!(fig, (-ygap / w, (t - y0) / (y1 - y0)), text(lab, font.tick, :right, :vcenter, PUB_FAMILY); subplot = i)
        end
        isempty(ylab) || annotate!(fig, (-(ygap + labw(labs, em) + 7.5) / w, 0.5),   # (user) clear of the tick labels
                                   text(ylab, font.guide, :center, :center, PUB_FAMILY; rotation = 90); subplot = i)
    end
    maximum(pw(fig[i]) for i in (iC, iD, iE)) - minimum(pw(fig[i]) for i in (iC, iD, iE)) < 0.5 ||
        @warn "_plot_summary_publication: the coverage/bias panels are not equally wide" [pw(fig[i]) for i in (iC, iD, iE)]
    off = bottom(fig[iE]) - bottom(fig[iF - 1 + nb])
    abs(off) < 0.5 || @warn "_plot_summary_publication: E ends $(round(off; digits = 1)) mm off F"
    # each panel's title at its plot area's left, the baseline 3 mm above it; each GROUP's bold letter at the top-LEFT corner of
    # its first panel (left of the axis labels), on the same baseline — mm from the computed layout → relative coordinates
    for (k, i) in enumerate((iA, iB, iC, iD, iE, iF))
        pa = Plots.plotarea(fig[i]); w = Plots.width(pa).value; h = Plots.height(pa).value
        y = 1 + 3.0 / h
        isempty(titles[k]) || annotate!(fig, (0.0, y), text(titles[k], font.title, :left, :bottom, PUB_FAMILY); subplot = i)
        if i in (iA, iC, iF)
            x0 = (Plots.bbox(fig[i]).x0[1] - pa.x0[1]).value / w + 1.0 / w
            letter = Dict(iA => "A", iC => "B", iF => "C")[i]
            yl = i == iC ? y + bline / h : y   # B: the letter and the model on their own line, the titles below (user)
            annotate!(fig, (x0, yl), text(letter, font.title + 2, :left, :bottom, PUB_FAMILY_BOLD); subplot = i)
            i == iC && annotate!(fig, (x0 + 7.5 / w, yl), text("Coverage and bias for $(um)", font.title, :left, :bottom, PUB_FAMILY); subplot = i)
        end
    end
    return fig
end

"""
    plot_summary_pre_omicron(rk, wis_pre, eff, cfg; models, title_suffix = "") -> Plot

ONE figure (2026-11-11). Left column: `_scenario_dots_panels` at h1 and h4 (relative WIS, with its legend cell)
above `_variant_coverage_bias_panels` (50% / 90% coverage, bias, for the contact – variant `models` vs no
contact and no contact + variant, from `variant_coverage_bias_frame(wis_pre, models)`). Right column:
`_tweedie_forest_panel(tweedie_forest_frame(eff))`. `rk` = 17j's pre-Omicron ranking, `wis_pre` =
`scenario_wis_by_horizon_until(…)`, `eff` = 19r's `19r_fixed_effects.csv`. `publication = true` (2026-11-18, user): panel
heads A–F above the panels (`_pub_head!`), Helvetica and `PUB_FONT`, no figure title — save with `save_show(…; pad = false)`.
In that style C–E show `usage_model` under each data usage (`_usage_coverage_bias_panels`, from `rk`; `models` is then unused)
and F is `_tweedie_forest_blocks`.
"""
function plot_summary_pre_omicron(rk::DataFrame, wis_pre::DataFrame, eff::DataFrame, cfg::FrameworkConfig;
                                  models, title_suffix::AbstractString = "", publication::Bool = false,
                                  usage_model::AbstractString = "weighted-hweibull|neighbourhood")
    publication && return _plot_summary_publication(rk, eff, cfg; usage_model)
    dp, leg = _scenario_dots_panels(rk; horizons = (1, 4), titles = ["relative WIS, horizon 1", "relative WIS, horizon 4"])
    cb = _variant_coverage_bias_panels(variant_coverage_bias_frame(wis_pre, models), models, cfg)
    fp = _tweedie_forest_panel(tweedie_forest_frame(eff))
    # `@layout` cannot size a nested group, so the grid is built directly: left column 64%, its top row
    # (two dot panels + legend cell) and bottom row (three coverage/bias panels) sized separately.
    l = Plots.grid(1, 2; widths = [0.64, 0.36])
    l[1, 1] = Plots.grid(2, 1)
    l[1, 1][1, 1] = Plots.grid(1, 3; widths = [0.39, 0.39, 0.22])
    l[1, 1][2, 1] = Plots.grid(1, 3)
    # ⚠ `save_show` → `pad_margins` resets every subplot's left/bottom margin; the notebook passes `l = 12Plots.mm`,
    # which the 50% coverage panel's y-label needs.
    return plot(dp..., leg, cb..., fp; layout = l, size = (2300, 1100), left_margin = 12Plots.mm,
                bottom_margin = 12Plots.mm,
                plot_title = "Pre-Omicron summary: relative WIS (18j), coverage and bias (17j), and Tweedie fixed effects (19r)$(title_suffix)",
                plot_titlefontsize = 11)
end

# ── One-column summary of hweibull|nbhd – variant, with England vaccination coverage (2026-11-16) ─────────────
const VACC_PATH = joinpath(@__DIR__, "..", "dt_vaccination", "england_vaccination_by_age.csv")

"`\"05_11\"` → (5, 11), `\"90+\"` → (90, 120): a dashboard age band's first and last year (120 = the grid's open end)."
function _vacc_band_years(b::AbstractString)
    endswith(b, "+") && return (parse(Int, chop(b)), 120)
    lo, hi = parse.(Int, split(b, "_"))
    return (lo, hi)
end

"""
    vaccination_coverage_cis(grid; path = VACC_PATH) -> DataFrame

England's first- and second-dose coverage per day × CIS age bin, from `dt_vaccination/england_vaccination_by_age.csv`
(5-year dashboard bands with their NIMS denominators; `fetch_vaccination_by_age.py`). A band's people are assumed
uniform over its single years, so it contributes the share `f` of its years that fall in the bin:
`coverage(bin) = Σ f·cov·pop / Σ f·pop`. The grid's 2-10 bin therefore gets 6/7 of 05_11 (ages 5–10; ages 2–4 are in
no band and never eligible in this period) and 11-15 gets 1/7 of 05_11 plus 12_15. Columns `date`, `bin`, `bin_label`,
`cov_first`, `cov_second`, `population` (the bin's NIMS denominator).
"""
function vaccination_coverage_cis(grid; path::AbstractString = VACC_PATH)
    v = CSV.read(path, DataFrame; stringtype = String)
    rows = NamedTuple[]
    for g in groupby(v, :date), k in 1:grid.N
        num1 = num2 = den = 0.0
        for r in eachrow(g)
            lo, hi = _vacc_band_years(r.age_band)
            f = max(0, min(hi, grid.HI[k]) - max(lo, grid.LO[k]) + 1) / (hi - lo + 1)
            f == 0 && continue
            num1 += f * r.cov_first * r.population; num2 += f * r.cov_second * r.population; den += f * r.population
        end
        @assert den > 0 "vaccination_coverage_cis: no band overlaps $(grid.LAB[k])"
        push!(rows, (; date = Date(g.date[1]), bin = k, bin_label = grid.LAB[k], cov_first = num1 / den,
                       cov_second = num2 / den, population = den))
    end
    return sort!(DataFrame(rows), [:bin, :date])
end

"""
    plot_variant_summary_column(fq, vg, tr, vacc, grid; model, every = 4, title_suffix = "") -> Plot

ONE column, one panel per row, a shared date axis (tick labels on the bottom row only), for the contact – variant
model `model` (a base label, e.g. `"weighted-hweibull|neighbourhood"`):
1. forecast fans, all ages (`_fan_panel`, median + 50/95%; x = target week), from every `every`-th origin: the model
   with contact – variant, with contact only, and no contact – variant (Pattern 1's comparison and colours);
2. γ_SAR,v by variant with the variant shares (%) on a right axis (`_variant_gamma_panel` + `_add_share_axes!`);
3–4. susceptibility and infectivity, ratio to the reference bin (`_ratio_panel` on `tr.susc` / `tr.inf`, log axis with
   `_pow2_ticks`, no prior band);
   rows 2–4 and 6–7 are per-ORIGIN quantities from the h = 1 fit (fitted to the 8 weeks up to the origin) and are drawn
   at `week_mid(origin)`, the same Thursday convention as the fans, so a vertical line reads one week on every row;
5. England vaccination coverage in the susc/inf age groups (`supergroup_split`, population-weighted over the CIS bins of
   `vaccination_coverage_cis`; each group in its colour from row 3, the reference grey), first dose solid, second
   dashed (x = date);
6–7. (with `contacts`, 14j's `14j_contact_mean_timeline.csv`) 14j's `14j_contact_quantities_by_block.png`: per age block,
   the fitted ⟨k⟩ (negbin), (1−p⁰)·μ and excess degree (hweibull), median + 90% (at week_mid(origin), h = 1 chain).
last. (with `wis` = 17j's per-origin scores; 2026-11-17) row 1's three series' WIS per origin, averaged over `wis_horizons`
   (`_pattern_wis_panel`, at week_mid(origin)).
`vg` = `collect_variant_gamma`'s frame, `tr` = `collect_variant_susc_inf`'s result, `fq` = the fan cache. Panel titles
sit INSIDE the panels and the rows touch (user, 2026-11-16) through negative margins — save with
`save_show(…; pad = false)`, or its `pad_margins` resets them.
"""
function plot_variant_summary_column(fq, vg::DataFrame, tr, vacc::DataFrame, grid; model::AbstractString,
                                     every::Integer = 4, title_suffix::AbstractString = "", contacts = nothing,
                                     fans_again_at::Union{Nothing,Integer} = nothing, wis = nothing, wis_horizons = 1:4)
    lbl = model * "-variant"
    ax = _summary_x_axis(fq, vg, tr, model; every)
    vg, fan_origins, H, x0, x1 = ax.vg, ax.fan_origins, ax.H, ax.x0, ax.x1
    short = _bar_short(model)
    # 1–2
    # row 1 = Pattern 1's comparison for this model, in its colours: contact – variant, contact only, no contact – variant
    fan_series = ((lbl,                "$(short): contact – variant", "#eb6834"),
                  (model,              "$(short): contact only",      "#2a78d6"),
                  (NULL_VARIANT_LABEL, "no contact – variant",        "#1baf7a"))
    @assert all(in(Set(fq.quant.series)), first.(fan_series)) "plot_variant_summary_column: a fan series is not in the cache"
    fan_panel(ttl; phases = true) = _fan_panel(fq, "all", fan_series, fan_origins; H, bands = true, legend = :bottomright,
                                               title = ttl, phase_labels = phases, titlefontsize = 10)
    fan_title = "(a) weekly infections, all ages, and h1–$(H) forecasts from every $(every)th origin (median, 50% / 95%)"
    p1 = fan_panel(fan_title)
    pp = _variant_param_panels(vg, tr, model, ax.xo; letters = ('b', 'c', 'd'))
    p2, sh, p3, p4 = pp.gamma, pp.share, pp.susc, pp.inf
    # 5 — the SAME age groups as rows 3–4 (`supergroup_split`), in THEIR colours: each group's colour is read off the
    # susceptibility panel's series, so a colour means one age group in both rows; the reference group, a grey line
    # there, is grey here. Groups are population-weighted over their CIS bins (the bins' NIMS denominators).
    ref = findfirst(==(tr.ref_lab), grid.LAB)
    groups, gnames = supergroup_split(grid, ref)
    @assert gnames == tr.sg_names "plot_variant_summary_column: the age groups differ from the susc/inf stores'"
    gcol = Dict(String(sr[:label]) => sr[:linecolor] for sr in p3[1].series_list if sr[:label] in gnames)
    gcol[tr.ref_lab] = RGB(0.45, 0.45, 0.45)
    vv = @subset(vacc, :date .>= x0, :date .<= x1)
    p5 = plot(; title = "(e) vaccination coverage, England, by age (UKHSA dashboard archive)", titlefontsize = 10,
              ylabel = "share of age group", ylim = (0, 1.05), yticks = (0:0.2:1, ["$(round(Int, 100t))%" for t in 0:0.2:1]),
              legend = :left, legendfontsize = 6)
    for (g, nm) in zip(groups, gnames)
        s = combine(groupby(@subset(vv, in.(:bin, Ref(collect(g)))), :date),
                    [:cov_first, :population] => ((c, w) -> sum(c .* w) / sum(w)) => :cov_first,
                    [:cov_second, :population] => ((c, w) -> sum(c .* w) / sum(w)) => :cov_second)
        sort!(s, :date)
        plot!(p5, s.date, s.cov_first; color = gcol[nm], lw = 1.6, label = nm)
        plot!(p5, s.date, s.cov_second; color = gcol[nm], lw = 1.6, ls = :dash, label = "")
    end
    plot!(p5, [x0], [NaN]; color = :black, lw = 1.2, label = "first dose")
    plot!(p5, [x0], [NaN]; color = :black, lw = 1.2, ls = :dash, label = "second dose")
    panels = [p1, p2, p3, p4, p5]
    # 6–7: 14j's `14j_contact_quantities_by_block.png`, one row per age block (`contacts` = 14j's
    # `14j_contact_mean_timeline.csv`: median and 90% band of the fitted block contact rate per origin, h = 1)
    if contacts !== nothing
        # plain-text exponents: the figure's font has no glyph for ⁰ or ² (they render as empty boxes)
        qs = (("unweighted-negbin", "K1",     "unweighted-negbin — mean degree ⟨k⟩"),
              ("weighted-hweibull", "K1",     "weighted-hweibull — (1 − p0)·μ"),
              ("weighted-hweibull", "excess", "weighted-hweibull — excess degree (⟨k^2⟩/⟨k⟩)·g"))
        qcols = collect(palette(:Dark2_8))[1:length(qs)]          # 14j's `_model_cols`
        for (k, (b, desc)) in enumerate(sort(unique(select(contacts, :block, :block_desc)), :block) |> eachrow)
            sub = @subset(contacts, :block .== b, :origin .>= x0, :origin .<= x1)
            @assert all(sub.week .== week_mid.(sub.origin)) "plot_variant_summary_column: 14j's `week` is not week_mid(origin)"
            p = plot(; title = "($(Char('e' + k))) fitted contact rate in the origin week, $(desc) (h = 1 chain)", titlefontsize = 10,
                     ylabel = "per participant-day", legend = (k == 1 ? :topright : false), legendfontsize = 6,
                     ylim = (0, 1.08 * maximum(sub.hi)))
            for ((d, q, lab), c) in zip(qs, qcols)
                s_ = sort(@subset(sub, :degree .== d, :quantity .== q), :origin)
                @assert nrow(s_) > 0 "plot_variant_summary_column: no $(d) $(q) rows for block $(b)"
                plot!(p, s_.week, s_.med; ribbon = (s_.med .- s_.lo, s_.hi .- s_.med), fillalpha = 0.2,
                      color = c, lw = 1.6, label = lab)
            end
            push!(panels, p)
        end
    end
    # (2026-11-17, user request) LAST row: the per-origin WIS of row (a)'s three series, averaged over `wis_horizons`
    # (`wis` = 17j's per-origin scores, as for `plot_fans_and_parameters(…; layout = :parameters)`)
    wis === nothing ||
        push!(panels, _pattern_wis_panel(wis, fan_series, ax; horizons = wis_horizons, letter = Char('a' + length(panels)),
                                         phase_labels = false))   # row (a) names the phases; here they would rise into the row above
    # (2026-11-17, user request, exploratory) row (a) drawn AGAIN at row `fans_again_at`, so the fans sit right beside
    # the rows below the parameters too; every panel letter after it moves down by one
    if fans_again_at !== nothing
        @assert 3 <= fans_again_at <= length(panels) + 1 "plot_variant_summary_column: fans_again_at must leave row 2 (γ, with its twin) in place"
        # no phase names: rotated, they rise out of the panel into the row above (rows touch)
        insert!(panels, fans_again_at, fan_panel(fan_title * " — a repeat of row (a)"; phases = false))
        for (k, p) in enumerate(panels)
            plot!(p; title = replace(p[1][:title], r"^\([a-z]\)" => "($(Char('a' + k - 1)))"))
        end
    end
    n = length(panels)
    is_fan = [k == 1 || k == fans_again_at for k in 1:n]
    # no x-axis label: the bottom row's ticks are dates; the fans' title goes bottom-left (their top carries the phases)
    _finish_rows!(panels, ax; labelled = [k == n for k in 1:n], bottom_title = is_fan)
    fig = plot(panels...; layout = (n, 1), size = (1300, 300 * n),
               plot_title = "$(short) – variant: forecasts, transmission parameters and vaccination$(title_suffix)",
               plot_titlefontsize = 11)
    _add_share_axes!(fig, [sh]; ylabel_at = [true], idx = [2])
    plot!(fig[n + 1]; xlims = ax.xl)                     # the share twin, on the same axis
    _set_row_margins!(fig, 1:n; first_row = [k == 1 for k in 1:n], last_row = [k == n for k in 1:n],
                      left = fill(14Plots.mm, n),    # 10 mm clipped the (c)/(d)/vaccination y-labels (2026-11-17)
                      right = fill(22Plots.mm, n),   # right: room for row 2's right-axis label
                      twin = (n + 1, 2))
    return fig
end

"""
    plot_fans_and_parameters(fq, vg, tr; model, patterns = FAN_PATTERNS, every = 4, layout,
                             wis = nothing, wis_horizons = 1:4, obs_from = nothing, obs_to = nothing) -> Plot

PUBLICATION figures (2026-11-17, user; styled after Fig 1 of PLOS Comput Biol 10.1371/journal.pcbi.1011200), in Helvetica
(`PUB_FAMILY`): every panel has its bold letter and title ABOVE it (left-aligned annotations), its OWN monthly date labels, minor ticks at each week's
`week_mid` (`_minor_xticks!`), large fonts (`PUB_FONT`) and no figure title (that is the caption's job). Two layouts:
- `:fans` — the three fan patterns of `plot_fan_patterns` (all ages, `_fan_panel`), A–C, on EXACTLY the weeks
  `obs_from` (the first origin's first fit week) … `obs_to` (the last origin's h`H` target week), both Monday week
  starts and both required; `fq.observed` must hold them (`extend_fan_observed`). The fans stay every `every`-th origin;
  minor ticks on every observed week.
- `:parameters` — A Pattern 1's per-origin WIS averaged over `wis_horizons` (`_pattern_wis_panel`; `wis` required), then
  `model` – variant's B γ_SAR,v in % with each variant's share at the origin (%) dashed on the SAME axis, C relative
  susceptibility and D relative infectivity (ratio to the reference bin, log axis, "1/8"-style ticks; infectivity fixed
  at 1/8 … 8) — `_variant_param_panels(…; publication = true)` — on EXACTLY the weeks of the first … last origin
  (points at `week_mid(origin)`); minor ticks on every origin.
The phase names sit INSIDE panel A (`top_frac` headroom + `label_frac`). Save with `save_show(…; pad = false)`: the
figure sets its own margins.
"""
function plot_fans_and_parameters(fq, vg::DataFrame, tr; model::AbstractString, patterns = FAN_PATTERNS,
                                  every::Integer = 4, layout::Symbol,
                                  wis = nothing, wis_horizons = 1:4, obs_from::Union{Nothing,Date} = nothing,
                                  obs_to::Union{Nothing,Date} = nothing)
    @assert layout in (:fans, :parameters) "plot_fans_and_parameters: layout must be :fans or :parameters"
    @assert (wis !== nothing) == (layout == :parameters) "plot_fans_and_parameters: `wis` is required for, and only for, :parameters"
    @assert obs_to === nothing || layout == :fans "plot_fans_and_parameters: `obs_to` is for :fans only"
    @assert length(patterns) == 3 "plot_fans_and_parameters: expects three fan patterns"
    ax0 = _summary_x_axis(fq, vg, tr, model; every, x_from = obs_from)
    if layout == :fans
        @assert obs_from !== nothing && obs_to !== nothing "plot_fans_and_parameters: :fans needs obs_from and obs_to"
        @assert dayofweek(obs_from) == 1 && dayofweek(obs_to) == 1 "plot_fans_and_parameters: obs_from/obs_to must be Mondays"
    end
    # whole Monday–Sunday weeks: the points sit at their week_mid (Thursday)
    xrange = layout == :fans ? (obs_from, obs_to + Day(6)) : (first(ax0.origins), last(ax0.origins) + Day(6))
    ax = _summary_x_axis(fq, vg, tr, model; every, x_from = obs_from, xrange)
    F = PUB_FONT
    # the names INSIDE panel A, in the figure's font; a rotated name is CENTRED on `label_frac` (its :center halign runs
    # along the text), so with the data below `PUB_TOP_FRAC` the names fill the band above it
    phase_kw = (; fontsize = F.phase, label_frac = PUB_PHASE_FRAC, label_family = PUB_FAMILY)
    if layout == :fans
        panels = [begin
                      p = _fan_panel(fq, "all", ser, ax.fan_origins; H = ax.H, bands, legend = :bottomright, obs_from, obs_to,
                                     phase_labels = (k == 1), top_frac = (k == 1 ? PUB_TOP_FRAC : 1.0), phase_kw)
                      plot!(p; ylabel = "weekly infections")
                  end for (k, (_, bands, ser)) in enumerate(patterns)]
        heads = [replace(ttl, r"^Pattern \d+ — " => "") for (ttl, _, _) in patterns]
        minor_x = week_mid.(collect(obs_from:Week(1):obs_to))   # every observed week ⊇ every origin
    else
        hs = collect(wis_horizons)
        pw = _pattern_wis_panel(wis, first(patterns)[3], ax; horizons = hs, letter = 'a',
                                top_mult = 1.1 / PUB_TOP_FRAC, phase_kw)
        pp = _variant_param_panels(ax.vg, tr, model, ax.xo; letters = ('b', 'c', 'd'), publication = true)
        panels = [pw, pp.gamma, pp.susc, pp.inf]
        short = _bar_short(model)
        heads = ["Forecast skill of $(short): WIS (log scale) per forecast origin, mean over h$(first(hs))–$(last(hs))",
                 "Per-contact secondary attack rate by variant, $(short) – variant",
                 "Relative susceptibility, ratio to $(tr.ref_lab)",
                 "Relative infectivity, ratio to $(tr.ref_lab)"]
        minor_x = ax.xo
    end
    @assert issubset(ax.xo, minor_x)
    n = length(panels)
    tks = collect(Date(year(ax.x0), month(ax.x0), 1) + Month(1):Month(1):ax.x1)        # the 1st of every month
    # month names, with the year on the first tick and every January only — "Sep 2020" on all 16 months overlaps
    tlab = [k == 1 || month(t) == 1 ? Dates.format(t, "u yyyy") : Dates.format(t, "u") for (k, t) in enumerate(tks)]
    for (k, p) in enumerate(panels)
        plot!(p; xlims = ax.xl, xrotation = 0, xticks = (Dates.value.(tks), tlab), title = "",
              guidefontsize = F.guide, tickfontsize = F.tick, legendfontsize = F.legend,
              guidefontfamily = PUB_FAMILY, tickfontfamily = PUB_FAMILY, legendfontfamily = PUB_FAMILY)
        if layout == :fans   # every decade labelled (10k, 100k, 1M, …): GR thins them once panel A has its headroom
            yl = Plots.ylims(p)
            dec = 10.0 .^ (ceil(Int, log10(yl[1])):floor(Int, log10(yl[2])))
            plot!(p; yticks = (dec, [y >= 1e6 ? "$(round(Int, y / 1e6))M" : "$(round(Int, y / 1e3))k" for y in dec]))
        end
        _minor_xticks!(p, minor_x)
        _pub_head!(p, k, heads[k])
    end
    # (user, 2026-11-18) the parameter rows closer together; the fans keep their spacing
    tight = layout == :parameters
    return plot(panels...; layout = (n, 1), size = (1100, (tight ? 310 : 340) * n),
                left_margin = 8Plots.mm, right_margin = 4Plots.mm, top_margin = (tight ? 7 : 9) * Plots.mm,
                bottom_margin = (tight ? 0 : 3) * Plots.mm)
end


# ── Contact rate per age block + relative contact reproduction number, merged (2026-11-18, user) ──────────────────
"The three contact models of `plot_contact_rate_reproduction`: (NGM label, degree, 14j quantity, legend) — each NGM's input."
const CR_MODELS = (("unweighted-negbin|mean",          "unweighted-negbin", "K1",     "negbin: mean degree"),
                   ("weighted-hweibull|mean",          "weighted-hweibull", "K1",     "hweibull: mean degree"),
                   ("weighted-hweibull|neighbourhood", "weighted-hweibull", "excess", "hweibull: nbhd degree"))   # legends (user)

"""
    plot_contact_rate_reproduction(tl, rn; origins) -> (; fig, data)

PUBLICATION figure, 1 column × 3 rows, merging 14j's `14j_contact_quantities_by_block.png` and 9j's
`9j_contact_reproduction_number.png` over `origins` (user: pre-Omicron). Read-only: `tl` = 14j's
`14j_contact_mean_timeline.csv`, `rn` = 14j's `14j_reproduction_number.csv` (its `contact_rel_R` panel — the same quantity
9j draws — and `national_R`).
- A, B — the fitted contact rate per participant-day in the origin week, children then adults (median, 90% band; the h = 1
  chain, at `week_mid(origin)`): each NGM's contact input — negbin's mean degree, hweibull's duration-weighted mean degree
  (1 − p0)·μ, and hweibull's excess degree.
- C — the contacts-only relative reproduction number ρ(C*)/ρ(C*_ref) (ref = the first origin; median, 90% band) of the
  three models, at their contact week (origin + 1, `week_mid`), with inc2prev's national R (England, ABSOLUTE) in black —
  two different quantities on one axis, as in 9j. The model-free observed series of 9j is left out (user).
One colour per model in all three rows (14j's Dark2 order); the model legend in B. Grey phase bands (`shade_periods_main!`)
on all three rows, the names inside A (user, 2026-11-18). Monthly ticks and minor ticks on every week. Save with
`save_show(…; pad = false)`. `data` = the plotted rows (one frame, `panel` ∈ {children, adults, contact_rel_R, national_R}).
"""
function plot_contact_rate_reproduction(tl::DataFrame, rn::DataFrame; origins)
    os = Set(origins)
    cols = collect(palette(:Dark2_8))[1:length(CR_MODELS)]
    x0, x1 = first(origins), last(origins) + Week(1) + Day(6)          # the origin weeks and the last contact week
    xl = Dates.value.((x0, x1))
    panels = Plots.Plot[]; heads = String[]; data = DataFrame[]
    for (k, b) in enumerate(sort(unique(tl.block)))
        sub = @subset(tl, :block .== b, in.(:origin, Ref(os)))
        desc = only(unique(sub.block_desc))
        # (2026-11-18, user: phase bands on all three rows) the names sit inside A, above its data (`PUB_TOP_FRAC`), so the
        # model legend moves to B
        p = plot(; ylabel = "contacts per participant-day", legend = (k == 2 ? :topright : false),
                 ylim = (0, (k == 1 ? 1.05 / PUB_TOP_FRAC : 1.4) * maximum(sub.hi)))
        for ((_, d, q, lab), c) in zip(CR_MODELS, cols)
            s = sort(@subset(sub, :degree .== d, :quantity .== q), :origin)
            @assert s.origin == origins "plot_contact_rate_reproduction: $d $q lacks an origin in block $b"
            @assert all(s.week .== week_mid.(s.origin)) "plot_contact_rate_reproduction: 14j's `week` is not week_mid(origin)"
            plot!(p, s.week, s.med; ribbon = (s.med .- s.lo, s.hi .- s.med), fillalpha = 0.2, color = c, lw = 1.8,
                  marker = :circle, ms = 2, msw = 0, label = lab)
            push!(data, DataFrame(panel = String(split(desc)[1]), series = lab, origin = s.origin, x = s.week,
                                  med = s.med, lo = s.lo, hi = s.hi))
        end
        shade_periods_main!(p, x0, x1; labels = (k == 1), fontsize = PUB_FONT.phase, label_frac = PUB_PHASE_FRAC,
                            label_family = PUB_FAMILY)   # after the Date series (see `shade_periods!`)
        push!(panels, p)
        push!(heads, "Fitted contact rate in the origin week, " * replace(desc, "bins " => "age groups ", r"\+(?=\d)" => ", "))   # "70+" keeps its +
    end
    r = @subset(rn, :panel .== "contact_rel_R", in.(:origin, Ref(os)), in.(:label, Ref(first.(CR_MODELS))))
    p = plot(; ylabel = "relative contact Rt\nnational Rt", legend = :topright,
             ylim = (0, 1.45 * max(maximum(r.hi), maximum(r.national_R))))
    hline!(p, [1.0]; color = :grey50, ls = :dash, lw = 1, label = "")
    for ((m, _, _, lab), c) in zip(CR_MODELS, cols)
        s = sort(@subset(r, :label .== m), :origin)
        @assert s.origin == origins "plot_contact_rate_reproduction: $m lacks an origin"
        @assert all(s.week_mid .== week_mid.(s.contact_week)) && all(s.contact_week .== s.origin .+ Week(1))
        plot!(p, s.week_mid, s.med; ribbon = (s.med .- s.lo, s.hi .- s.med), fillalpha = 0.2, color = c, lw = 1.8,
              marker = :circle, ms = 2, msw = 0, label = _bar_short(m))   # the NGM model: negbin|mean, …
        push!(data, DataFrame(panel = "contact_rel_R", series = m, origin = s.origin, x = s.week_mid, med = s.med, lo = s.lo, hi = s.hi))
    end
    nr = sort(combine(groupby(r, [:origin, :week_mid]), :national_R => (v -> (@assert allequal(v); first(v))) => :national_R), :origin)
    plot!(p, nr.week_mid, nr.national_R; color = :black, lw = 1.8, label = "national R (absolute)")   # (user)
    push!(data, DataFrame(panel = "national_R", series = "inc2prev", origin = nr.origin, x = nr.week_mid,
                          med = nr.national_R, lo = missing, hi = missing))
    shade_periods_main!(p, x0, x1; labels = false)
    push!(panels, p)
    push!(heads, "Relative contact reproduction number (ref. = first origin) and national R")
    tks = collect(Date(year(x0), month(x0), 1) + Month(1):Month(1):x1)
    tlab = [k == 1 || month(t) == 1 ? Dates.format(t, "u yyyy") : Dates.format(t, "u") for (k, t) in enumerate(tks)]
    for (k, p) in enumerate(panels)
        plot!(p; xlims = xl, xrotation = 0, xticks = (Dates.value.(tks), tlab))
        _pub_fonts!(p)
        _minor_xticks!(p, week_mid.(collect(first(origins):Week(1):last(origins) + Week(1))))   # every origin and contact week
        _pub_head!(p, k, heads[k])
    end
    fig = plot(panels...; layout = (length(panels), 1), size = (1100, 330 * length(panels)), left_margin = 8Plots.mm,
               right_margin = 4Plots.mm, top_margin = 7Plots.mm, bottom_margin = 0Plots.mm)
    return (; fig, data = reduce(vcat, data))
end

"""
    _pattern_wis_panel(wis, series, ax; horizons = 1:4, letter, scale = WIS_SCALE) -> Plot

Per-origin WIS (`wis` = 17j's `17j_scenarios_scores_by_model_date_horizon.csv` with `series`, `forecast_date`,
`horizon`, `scale`, `wis`, each row already the mean over the 7 age bins) of the fan `series` tuples, in their colours,
AVERAGED over `horizons` (user, 2026-11-17: h1–4) — every origin must carry every horizon, so each point is a mean of
the same number of scores. A WIS belongs to its ORIGIN, so it is drawn at `week_mid(origin)` like the parameter rows,
over the same origins (`ax.origins`, the pre-Omicron window — the x-range itself runs H weeks further, for the fans).
"""
function _pattern_wis_panel(wis::DataFrame, series, ax; horizons = 1:4, letter::Char, scale::AbstractString = WIS_SCALE,
                            phase_labels::Bool = true, top_mult::Union{Nothing,Real} = nothing, phase_kw = (;))
    hs = collect(horizons)
    w = @subset(wis, :scale .== scale, in.(:horizon, Ref(hs)), in.(:series, Ref(first.(series))),
                in.(:forecast_date, Ref(Set(ax.origins))))   # the parameter rows' origins: the x-range runs past them
    @assert nrow(unique(w, [:series, :forecast_date, :horizon])) == nrow(w) "_pattern_wis_panel: duplicate scores"
    m = combine(groupby(w, [:series, :forecast_date]), :wis => mean => :wis, nrow => :n_h)
    @assert all(==(length(hs)), m.n_h) "_pattern_wis_panel: an origin lacks a horizon"
    hl = hs == collect(first(hs):last(hs)) ? "h$(first(hs))–$(last(hs))" : join("h" .* string.(hs), ", ")
    p = plot(; title = "($(letter)) WIS ($(scale) scale) per forecast origin, averaged over $(hl)",
             titlefontsize = 10, ylabel = "mean WIS ($(scale))", legend = (phase_labels ? :right : :topright), legendfontsize = 7,
             ylim = (0, something(top_mult, phase_labels ? 1.55 : 1.1) * maximum(m.wis)), ytickfontsize = 7)   # headroom: the phase names
    for t in series
        s = sort(@subset(m, :series .== t[1]), :forecast_date)
        @assert s.forecast_date == ax.origins "_pattern_wis_panel: $(t[1]) lacks an origin"
        plot!(p, week_mid.(s.forecast_date), s.wis; color = t[3], lw = 1.6, marker = :circle, ms = 2, msw = 0, label = t[2])
    end
    # (user, 2026-11-17) the phase bands, over the panel's whole date range — after the Date series (see `shade_periods!`)
    shade_periods_main!(p, ax.x0, ax.x1; labels = phase_labels, fontsize = 6, phase_kw...)
    return p
end

"""
    extend_fan_observed(fq, grid; from::Date, inf = load_raw_infection_inputs()) -> NamedTuple

`fq` with its `observed` series extended back to the Monday week `from` (e.g. the first origin's first fit week,
`first_origin − (n_fit − 1)` weeks), which the fan cache lacks — it holds only the forecast TARGET weeks. The extra weeks
come from the outcome itself, `weekly_infections` on inc2prev's `estimates_age.csv` (the 7 CIS bins and their sum as
`"all"`). READ-ONLY (no fit). Tripwire: the same reader must reproduce the cached series on every overlapping week, to
1e-9 relative, so the joined series is one quantity.
"""
function extend_fan_observed(fq, grid; from::Date, inf = load_raw_infection_inputs())
    @assert dayofweek(from) == 1 "extend_fan_observed: `from` is not a Monday week start"
    first_cached = minimum(fq.observed.week)
    weeks = collect(from:Week(1):maximum(fq.observed.week))
    M, _ = weekly_infections(inf.df, weeks, grid)
    @assert all(>(0), M) "extend_fan_observed: a week outside the infection estimates (zero-filled)"
    labs = vcat(collect(grid.LAB), ["all"])
    new = DataFrame([(; week = w, age_group = lab, observed = k <= grid.N ? M[k, t] : sum(M[:, t]))
                     for (t, w) in enumerate(weeks) for (k, lab) in enumerate(labs)])
    j = innerjoin(fq.observed, rename(new, :observed => :obs_new); on = [:week, :age_group])
    @assert nrow(j) == nrow(fq.observed) "extend_fan_observed: a cached week is not on the reread grid"
    @assert maximum(abs.(j.observed .- j.obs_new) ./ j.observed) < 1e-9 "extend_fan_observed: the reread outcome ≠ the cached truth"
    observed = sort!(vcat(@subset(new, :week .< first_cached), fq.observed), [:age_group, :week])
    return (; quant = fq.quant, observed)
end

"""
    _summary_x_axis(fq, vg, tr, model; every) -> NamedTuple

The ONE date axis of the summary figures. `vg` comes back with its `origin` moved to `week_mid(origin)`, and `xo` is the
same for the susc/inf rows: per-ORIGIN quantities (h = 1 fit, fitted to the 8 weeks up to the origin) sit at the mid-date
of the origin week, the Thursday convention of the fans' points, so a vertical line reads one week on every row. The
range `x0`…`x1` spans the parameter origins and the fans (every `every`-th origin, out to `H` weeks ahead), ±1 week.
`xrange = (x0, x1)` sets the range EXACTLY instead (2026-11-17: the split column figures, each spanning whole Monday–Sunday
weeks).
"""
function _summary_x_axis(fq, vg::DataFrame, tr, model::AbstractString; every::Integer,
                         x_from::Union{Nothing,Date} = nothing, xrange::Union{Nothing,Tuple{Date,Date}} = nothing)
    lbl = model * "-variant"
    @assert lbl in tr.labels "_summary_x_axis: $lbl not in the susc/inf stores"
    fan_origins = sort(unique(fq.quant.forecast_date))[1:every:end]
    origins = sort(unique(@subset(vg, :model .== lbl).origin))
    @assert all(o -> dayofweek(o) == 1, origins) "_summary_x_axis: origins are not Monday week starts"
    H = maximum(fq.quant.horizon)
    x0 = min(first(origins), first(fan_origins), something(x_from, first(origins))) - Day(7)
    x1 = max(last(origins), last(fan_origins) + Week(H)) + Day(7)
    xrange === nothing || ((x0, x1) = xrange)
    return (; vg = transform(vg, :origin => ByRow(week_mid) => :origin), fan_origins, origins, xo = week_mid.(origins),
            H, x0, x1, xl = (Dates.value(x0), Dates.value(x1)))
end

"""
    _ref_legend_in_age_order!(p, gnames, ref_lab) -> p

Puts the reference group's legend entry ("<ref_lab> (ref)") in AGE order among the groups (user: 2-15, 16-24, 25-34 (ref),
35-49, 50+). `_ratio_panel` must DRAW the reference first — a Date-valued series is what makes the axis a date axis — and
an extra legend-only series would shift every later group's automatic colour, so the entry is moved in the subplot's
series list AFTER plotting (the legend follows that list; colours are already fixed).
"""
function _ref_legend_in_age_order!(p, gnames, ref_lab::AbstractString)
    sl = p[1].series_list
    i = findfirst(sr -> sr[:label] == "$(ref_lab) (ref)", sl)
    k = findfirst(==(ref_lab), gnames)
    (i === nothing || k === nothing || k == length(gnames)) && return p
    r = popat!(sl, i)
    j = findfirst(sr -> sr[:label] == gnames[k + 1], sl)   # the next older group
    j === nothing ? push!(sl, r) : insert!(sl, j, r)
    return p
end

"""
    _variant_param_panels(vg, tr, model, xo; letters) -> (; gamma, share, susc, inf)

γ_SAR,v (`_variant_gamma_panel`, 0–1, shares returned for `_add_share_axes!`), then susceptibility and infectivity as
ratios to `tr.ref_lab` (`_ratio_panel`, log axis with `_pow2_ticks`, no prior band, no marker outline), for `model` –
variant. `vg` and `xo` must already be on `week_mid` (`_summary_x_axis`). `letters` label the three panels.
"""
function _variant_param_panels(vg::DataFrame, tr, model::AbstractString, xo; letters, publication::Bool = false)
    lbl = model * "-variant"
    # `publication` (2026-11-17, user): γ in %, its shares dashed on the SAME axis (no twin), "relative …" labels,
    # infectivity fixed at 1/8 … 8 and fraction tick labels
    pg, sh = _variant_gamma_panel(vg, model; ylim = (publication ? (0.0, 100.0) : (0.0, 1.0)), free = (), ycap = 2.5, legend = :right,
                                  ylabel = (publication ? "variant SAR, % (solid)\nshare of variant, % (dashed)" : "γ_SAR,v (solid)"),
                                  share_key = !publication, titlefontsize = 10, scale = (publication ? 100 : 1),
                                  title = "($(letters[1])) per-contact secondary attack rate by variant, fitted at each origin (h = 1 fit)")
    if publication
        for (nm, x, y) in sh
            plot!(pg, x, 100 .* y; color = get(VARIANT_COLS, nm, :black), ls = :dash, lw = 1.2, label = "")
        end
        plot!(pg, sh[1][2][1:1], [NaN]; color = :black, ls = :dash, lw = 1.2, label = "share of variant")   # short: one legend row spaces every entry by the LONGEST label
        # (user, 2026-11-18) the legend VERTICAL at the right, in the empty band between Delta's solid and dashed lines
        plot!(pg; ylims = (0.0, 105.0), yticks = 0:20:100, legend = (0.878, 0.675))   # (x of the text, y of the first row): centred on 60 %, the text at C's x (user)
    end
    ratio(st, q, letter, leg; yl = _median_ylims(vec(st[lbl].med); log = true)) =
        _ratio_panel(st, lbl, xo; gnames = tr.sg_names, ref_lab = tr.ref_lab, pri = nothing, yl,
                     yticks = _pow2_ticks(yl; frac = publication), title = "($(letter)) $(q) ratio to $(tr.ref_lab), fitted at each origin (h = 1 fit, log axis)",
                     ylabel = (publication ? "relative $(q)" : "$(q) ratio"), xlabel = "", legend = leg, first_panel = true, titlefontsize = 10,
                     marker_kw = (; msw = 0))
    # publication (user, 2026-11-18): VERTICAL legends at the right, where the panel is empty — susceptibility at the bottom
    # right (below the curves from Sep), infectivity (now with its own legend) at the top right of its 1/16 … 16 axis
    susc = ratio(tr.susc, "susceptibility", letters[2], :bottomright)
    inf = publication ? ratio(tr.inf, "infectivity", letters[3], (0.878, 0.985); yl = (1 / 16, 16.0)) :   # top, clear of the bands; C's x (user)
                        ratio(tr.inf, "infectivity", letters[3], false)
    publication && foreach(p -> _ref_legend_in_age_order!(p, tr.sg_names, tr.ref_lab), (susc, inf))
    return (; gamma = pg, share = sh, susc, inf)
end

"""
    _finish_rows!(panels, ax; labelled, bottom_title)

The shared date axis and the inside titles of the summary figures. Every panel gets the SAME range and the same monthly
ticks — hiding labels with a formatter is not enough, each panel would place its own (unlabelled) gridlines at different
dates — labelled only where `labelled[k]`. The title moves INSIDE the panel (user, 2026-11-16) so the rows can touch:
top-left, or bottom-left where `bottom_title[k]` (a fan panel, whose top can carry the phase names); `title_y[k]` lowers a
top-left title (the WIS panel with phase names: below them, at 0.75).
"""
function _finish_rows!(panels, ax; labelled, bottom_title, title_y = fill(nothing, length(panels)), minor_x = nothing)
    tks = collect(Date(year(ax.x0), month(ax.x0), 1) + Month(1):Month(1):ax.x1)        # the 1st of every month
    for (k, p) in enumerate(panels)
        plot!(p; xlims = ax.xl, xrotation = 0,
              xticks = (Dates.value.(tks), labelled[k] ? Dates.format.(tks, "u yyyy") : fill("", length(tks))),
              xtickfontsize = 7)
        minor_x === nothing || _minor_xticks!(p, minor_x)
        ttl = p[1][:title]
        plot!(p; title = "")
        annotate!(p, bottom_title[k] ? (0.01, 0.04) : (0.01, something(title_y[k], 0.97)),
                  text(ttl, 9, :left, bottom_title[k] ? :bottom : :top, Plots.default(:fontfamily)))   # the figure's font
    end
    return panels
end

"""
    _minor_xticks!(p, xs; frac = 0.015)

Minor x-ticks at the dates `xs` (user, 2026-11-17: one per forecast origin, at `week_mid(origin)` — where every per-origin
point sits): Plots can only space minor ticks EVENLY between the major ones, so they are drawn as short inward marks
from the bottom of the plot area, `frac` of its height (in the axis's own scale, so a log axis works too). The panel's
x minor grid — evenly spaced, so at odd dates — is switched off, and the y-limits are pinned so the marks do not move them.
"""
function _minor_xticks!(p, xs; frac::Real = 0.015)   # half a major tick (user: minor < major)
    yl = Plots.ylims(p)
    logy = p[1][:yaxis][:scale] == :log10
    y1 = logy ? yl[1] * (yl[2] / yl[1])^frac : yl[1] + frac * (yl[2] - yl[1])
    x = Float64.(Dates.value.(xs))
    plot!(p, vec([x x fill(NaN, length(x))]'), vec(repeat([yl[1] y1 NaN], length(x))');
          color = :black, lw = 0.5, label = "", primary = false)
    plot!(p; ylims = yl, xminorgrid = false)
    return p
end

"""
    _set_row_margins!(fig, idx; first_row, last_row, left, right, twin = nothing)

Margins on the COMPOSED figure's subplots `idx` — set on standalone panels, `plot(panels…)` copies the FIRST panel's onto
every row (measured: the last row came out with row 1's −5 mm and its tick labels clipped). Rows TOUCH (user,
2026-11-16): a row without tick labels still reserves ~5 mm for them and has a ~2 mm pad above its plot area, so the
margins go negative (−5 mm below, −2 mm above); the first row keeps 6 mm on top for the phase names and the last row
8 mm below for its date labels. ⚠ Save with `save_show(…; pad = false)`: its `pad_margins` would reset them.
`twin = (subplot, host)` gives a `twinx` subplot its host's right/top/bottom margins.
"""
function _set_row_margins!(fig, idx; first_row, last_row, left, right, twin = nothing)
    tb(k) = ((first_row[k] ? 6 : -2) * Plots.mm, (last_row[k] ? 8 : -5) * Plots.mm)
    for (k, i) in enumerate(idx)
        t, b = tb(k)
        plot!(fig[i]; left_margin = left[k], right_margin = right[k], top_margin = t, bottom_margin = b)
    end
    if twin !== nothing
        s, h = twin
        k = findfirst(==(h), collect(idx))
        t, b = tb(k)
        plot!(fig[s]; right_margin = right[k], top_margin = t, bottom_margin = b)
    end
    return fig
end
