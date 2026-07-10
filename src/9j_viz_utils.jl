# 9j_viz_utils.jl — detailed implementation behind the 9j forecast-diagnostics notebook.
#
# Keeps `9j_forecast_diagnostics.ipynb` thin: the notebook chooses inputs, calls these
# builders, and displays/saves the returned figures; all the aggregation + plotting detail
# lives here. Mirrors the paper-style evaluation of Munday et al. 2023 (inst/pcbi.1011453):
# relative WIS, bias, age-stratified relative WIS, per-period skill, interval coverage,
# and the NGM-eigenvalue reproduction number.
#
# Reference model = `unweighted-negbin|mean` (the "no-interaction" analog); WIS on the LOG
# scale is the headline (robust — the neighbourhood-NGM natural-scale WIS blows up).
#
# Depends on the forecasting framework (`forecast_utils.jl`) and the read-only chain helpers
# in `8j_viz_utils.jl` (`reproduction_draws`, `chain_path`, …), included below.

include("8j_viz_utils.jl")

const REF_MODEL = "unweighted-negbin|mean"   # relative-WIS reference (Munday "no-interaction")
const WIS_SCALE = "log"                       # headline scale for WIS / coverage frames

# ── Pandemic periods (Munday 2023, Table 2 / Gimma et al.) ────────────────────────────
# Named UK COVID phases used to aggregate forecast skill over the epidemic timeline.
# Boundaries verbatim from Table 2, correcting the OCR typo in the Lockdown-3 start
# (printed "2021-05-01" → 2021-01-05). Origins before Lockdown 2 (the autumn-2020 tier
# system) fall outside Gimma's periods → `missing`.
const PERIODS = [
    ("Lockdown 2",              Date(2020, 11, 5),  Date(2020, 12, 2)),
    ("Lockdown 2 Easing",       Date(2020, 12, 3),  Date(2020, 12, 19)),
    ("Christmas",               Date(2020, 12, 20), Date(2021, 1, 4)),
    ("Lockdown 3",              Date(2021, 1, 5),   Date(2021, 3, 8)),
    ("Lockdown 3 Schools open", Date(2021, 3, 9),   Date(2021, 3, 28)),
    ("Lockdown 3 Easing",       Date(2021, 3, 29),  Date(2021, 9, 30)),
    ("Opening up",              Date(2021, 10, 1),  Date(2021, 11, 24)),
]
const PERIOD_ORDER = first.(PERIODS)

"Named pandemic period (Table 2) containing `d`, or `missing` if outside all periods."
function period_of(d::Date)
    for (nm, lo, hi) in PERIODS
        lo <= d <= hi && return nm
    end
    return missing
end

"Print a tally of forecast `origins` per named period (origins before Lockdown 2 → pre-L2)."
function period_summary(origins)
    tally = Dict{Union{String,Missing},Int}()
    for o in origins
        p = period_of(o); tally[p] = get(tally, p, 0) + 1
    end
    println("forecast origins per period:")
    for p in PERIOD_ORDER
        haskey(tally, p) && println("  ", rpad(p, 26), tally[p])
    end
    haskey(tally, missing) && println("  ", rpad("(pre-Lockdown 2)", 26), tally[missing])
    return tally
end

# ── small utilities ───────────────────────────────────────────────────────────────────
"Save `fig` to `path` and display it inline; returns `fig`."
save_show(fig, path::AbstractString) = (savefig(fig, path); display(fig); fig)

"Relative WIS `wis[h] / wis_ref[h]` aligned to horizons `hz` (NaN where the ref is absent)."
_relwis(wis, hz, ref_by_h) = wis ./ [get(ref_by_h, h, NaN) for h in hz]

# ── Fig 3 analog: relative WIS + bias (A,B) and age-stratified relative WIS (C) ────────
"""
    plot_rwis_bias_by_horizon(scores, labels4, model_cols, cfg; ref, scale) -> Plot

Two-panel figure: (A) relative WIS by horizon (ratio to `ref`), (B) forecast bias by
horizon; one series per model. rWIS < 1 ⇒ better than the reference; bias ≈ 0 ⇒ unbiased.
"""
function plot_rwis_bias_by_horizon(scores, labels4, model_cols, cfg;
                                   ref::AbstractString = REF_MODEL, scale::AbstractString = WIS_SCALE)
    bmh   = @subset(scores.by_model_h, :scale .== scale)
    ref_h = Dict(r.horizon => r.wis for r in eachrow(@subset(bmh, :model .== ref)))

    a = plot(; xlabel = "horizon (weeks)", ylabel = "relative WIS (vs $(ref))",
             title = "A. relative WIS by horizon", titlefontsize = 9,
             legend = :topleft, legendfontsize = 6, xticks = 1:length(cfg.horizons))
    hline!(a, [1.0]; color = :gray, ls = :dash, label = "")
    for (ci, m) in enumerate(labels4)
        s = sort(@subset(bmh, :model .== m), :horizon)
        plot!(a, s.horizon, _relwis(s.wis, s.horizon, ref_h);
              color = model_cols[ci], lw = 2, marker = :circle, ms = 3, label = m)
    end

    b = plot(; xlabel = "horizon (weeks)", ylabel = "bias",
             title = "B. forecast bias by horizon", titlefontsize = 9,
             legend = false, xticks = 1:length(cfg.horizons))
    hline!(b, [0.0]; color = :gray, ls = :dash, label = "")
    for (ci, m) in enumerate(labels4)
        s = sort(@subset(bmh, :model .== m), :horizon)
        plot!(b, s.horizon, s.bias; color = model_cols[ci], lw = 2, marker = :circle, ms = 3, label = m)
    end

    return plot(a, b; layout = (1, 2), size = (1000, 430),
                plot_title = "9j — relative WIS & bias by horizon (log scale, vs $(ref))",
                plot_titlefontsize = 11)
end

"""
    plot_rwis_by_age_horizon(scores, labels4, model_cols, grid, cfg; ref, scale) -> Plot

One small panel per CIS age bin of relative WIS vs horizon (ratio to `ref`), one series
per model — the age-disaggregated relative score of Munday Fig 3 B/C.
"""
function plot_rwis_by_age_horizon(scores, labels4, model_cols, grid, cfg;
                                  ref::AbstractString = REF_MODEL, scale::AbstractString = WIS_SCALE)
    bmha = @subset(scores.by_model_h_age, :scale .== scale)
    panels = Plots.Plot[]
    for (ai, ag) in enumerate(grid.LAB)
        ref_h = Dict(r.horizon => r.wis for r in eachrow(@subset(bmha, :model .== ref, :age_group .== ag)))
        p = plot(; title = ag, titlefontsize = 8, xlabel = "horizon", ylabel = "rel. WIS",
                 legend = (ai == 1 ? :topleft : false), legendfontsize = 5,
                 xticks = 1:length(cfg.horizons))
        hline!(p, [1.0]; color = :gray, ls = :dash, label = "")
        for (ci, m) in enumerate(labels4)
            s = sort(@subset(bmha, :model .== m, :age_group .== ag), :horizon)
            isempty(s) && continue
            plot!(p, s.horizon, _relwis(s.wis, s.horizon, ref_h);
                  color = model_cols[ci], lw = 1.6, marker = :circle, ms = 2, label = m)
        end
        push!(panels, p)
    end
    return plot(panels...; layout = (2, 4), size = (1300, 620),
                plot_title = "9j — age-stratified relative WIS by horizon (log scale, vs $(ref))",
                plot_titlefontsize = 11)
end

# ── Fig 4 analog: relative WIS vs horizon, faceted by pandemic period ─────────────────
"""
    plot_rwis_by_period(scores, labels4, model_cols, cfg; ref, scale) -> Plot

One panel per pandemic period (origins mapped via `period_of`); x = horizon, one line per
model, y = relative WIS (ratio to `ref`) after averaging log-scale WIS over the origins in
that period.
"""
function plot_rwis_by_period(scores, labels4, model_cols, cfg;
                             ref::AbstractString = REF_MODEL, scale::AbstractString = WIS_SCALE)
    bdth = @subset(scores.by_model_dt_h, :scale .== scale)
    bdth = @transform(bdth, :period = period_of.(:forecast_date))
    bdth = @subset(bdth, .!ismissing.(:period))
    agg  = combine(groupby(bdth, [:model, :period, :horizon]), :wis => mean => :wis)

    present = [p for p in PERIOD_ORDER if p in agg.period]
    panels = Plots.Plot[]
    for (pk, per) in enumerate(present)
        ref_h = Dict(r.horizon => r.wis for r in eachrow(@subset(agg, :model .== ref, :period .== per)))
        p = plot(; title = per, titlefontsize = 8, xlabel = "horizon (weeks)",
                 ylabel = "relative WIS", legend = (pk == 1 ? :topleft : false),
                 legendfontsize = 5, xticks = 1:length(cfg.horizons))
        hline!(p, [1.0]; color = :gray, ls = :dash, label = "")
        for (ci, m) in enumerate(labels4)
            s = sort(@subset(agg, :model .== m, :period .== per), :horizon)
            isempty(s) && continue
            plot!(p, s.horizon, _relwis(s.wis, s.horizon, ref_h);
                  color = model_cols[ci], lw = 1.8, marker = :circle, ms = 2, label = m)
        end
        push!(panels, p)
    end
    nc = min(3, length(panels)); nr = cld(length(panels), nc)
    return plot(panels...; layout = (nr, nc), size = (390 * nc, 300 * nr),
                plot_title = "9j — relative WIS by horizon per pandemic period (log scale, vs $(ref))",
                plot_titlefontsize = 10)
end

# ── Fig 5 analog: central-interval coverage (calibration) ─────────────────────────────
"""
    plot_interval_coverage(scores, labels4, model_cols, cfg; scale) -> Plot

Empirical 50% / 90% central-interval coverage by horizon (from scoringutils'
`interval_coverage_50/90`), one series per model, with the nominal reference lines.
"""
function plot_interval_coverage(scores, labels4, model_cols, cfg; scale::AbstractString = WIS_SCALE)
    covd = sort(@subset(scores.by_model_h, :scale .== scale), [:model, :horizon])
    panels = Plots.Plot[]
    for (lvl, col) in ((50, :interval_coverage_50), (90, :interval_coverage_90))
        p = plot(; title = "$(lvl)% central interval", titlefontsize = 9,
                 xlabel = "horizon (weeks)", ylabel = "empirical coverage",
                 legend = (lvl == 50 ? :bottomleft : false), legendfontsize = 6,
                 xticks = 1:length(cfg.horizons), ylim = (0, 1))
        hline!(p, [lvl / 100]; color = :gray, ls = :dash, label = "nominal $(lvl)%")
        for (ci, m) in enumerate(labels4)
            s = sort(@subset(covd, :model .== m), :horizon)
            plot!(p, s.horizon, s[!, col]; color = model_cols[ci], lw = 2,
                  marker = :circle, ms = 3, label = m)
        end
        push!(panels, p)
    end
    return plot(panels...; layout = (1, 2), size = (1000, 440),
                plot_title = "9j — forecast calibration: 50% & 90% interval coverage",
                plot_titlefontsize = 11)
end

# ── Reproduction number over time: dominant NGM eigenvalue ────────────────────────────
"""
    reproduction_over_time(combos, labels4, wins, cfg; grid, raw, h, save_dir, verbose)
        -> Dict(label => (; med, lo, hi))

For each forecast origin and model, the median and 90% band of the reproduction number
(`reproduction_draws`, dominant eigenvalue of the frozen origin-week NGM at horizon `h`).
Reloads window/degree data + cached chains per origin — read-only, no re-fit.
"""
function reproduction_over_time(combos, labels4, wins, cfg;
                                grid = cis_age_grid(), raw,
                                h::Integer = 1, save_dir::AbstractString = "../dt_intermediate",
                                verbose::Bool = true)
    nO = length(wins)
    store = Dict(l => (med = fill(NaN, nO), lo = fill(NaN, nO), hi = fill(NaN, nO)) for l in labels4)
    t0 = time()
    for (oi, win_o) in enumerate(wins)
        wd_o  = load_window_data(win_o; grid = grid)
        apd_o = prepare_degree_data(win_o, cfg; grid = grid, setting = :all,   # h=1 window = origin
                                    df_part_raw = raw.df_part, craw_raw = raw.craw)
        for ((dm, nb), lbl) in zip(combos, labels4)
            R = reproduction_draws(dm, nb, apd_o, wd_o, cfg, win_o; h = h, save_dir = save_dir)
            R === nothing && continue
            store[lbl].med[oi] = median(R)
            store[lbl].lo[oi]  = quantile(R, 0.05)
            store[lbl].hi[oi]  = quantile(R, 0.95)
        end
        verbose && (oi % 5 == 0 || oi == nO) &&
            println("  R(t) origin $oi/$nO (", win_o.origin, ")  elapsed ",
                    round(Int, time() - t0), "s")
    end
    return store
end

"""
    plot_reproduction(store, labels4, model_cols, origins; h) -> Plot

One panel of the reproduction number over time (one line + 90% ribbon per model, R=1
threshold). `store` is `reproduction_over_time`'s output; `origins` its window origins.
"""
function plot_reproduction(store, labels4, model_cols, origins; h::Integer = 1)
    fig = plot(; xlabel = "forecast origin",
               ylabel = "reproduction number R  (dominant NGM eigenvalue)",
               title = "9j — reproduction number over time by model (h=$h, 90% CI)",
               size = (950, 520), legend = :topleft, xrotation = 45)
    hline!(fig, [1.0]; color = :gray, ls = :dash, label = "R = 1")
    x = week_mid.(origins)
    for (ci, lbl) in enumerate(labels4)
        s = store[lbl]
        plot!(fig, x, s.med; color = model_cols[ci], lw = 1.8, marker = :circle, ms = 2,
              ribbon = (s.med .- s.lo, s.hi .- s.med), fillalpha = 0.12, label = lbl)
    end
    return fig
end
