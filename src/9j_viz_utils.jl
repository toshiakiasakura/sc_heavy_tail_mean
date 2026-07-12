# 9j_viz_utils.jl — detailed implementation behind the 9j forecast-diagnostics notebook.
#
# Keeps `9j_forecast_diagnostics.ipynb` thin: the notebook chooses inputs, calls these
# builders, and displays/saves the returned figures; all the aggregation + plotting detail
# lives here. Two groups of helpers:
#   1. Forecast assembly + on-disk cache (`assemble_or_load_forecasts`) and scoring report
#      (`report_forecast_scores`) — the slow origin×combo chain-reload loop is cached so the
#      notebook reruns in seconds.
#   2. Figures: the 8j-style skill/forecast/transmission panels reproduced in 9j, plus the
#      paper-style evaluation of Munday et al. 2023 (inst/pcbi.1011453) — relative WIS, bias,
#      age-stratified relative WIS, per-period skill, interval coverage, and the
#      NGM-eigenvalue reproduction number.
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
"""
    pad_margins(fig; l, b) -> fig

Secure outer margins on `fig` (applied to every subplot) so the x-/y-axis labels aren't
clipped at the figure's bottom/left edge — the GR default leaves too little room for the
rotated date ticks and the long y-labels. Mutates and returns `fig`.
"""
pad_margins(fig; l = 8Plots.mm, b = 12Plots.mm) = plot!(fig; left_margin = l, bottom_margin = b)

"Secure label margins (`pad_margins`), save `fig` to `path` and display it inline; returns `fig`."
save_show(fig, path::AbstractString) = (pad_margins(fig); savefig(fig, path); display(fig); fig)

"Relative WIS `wis[h] / wis_ref[h]` aligned to horizons `hz` (NaN where the ref is absent)."
_relwis(wis, hz, ref_by_h) = wis ./ [get(ref_by_h, h, NaN) for h in hz]

# ── Forecast assembly + on-disk cache ─────────────────────────────────────────────────
"""
    assemble_or_load_forecasts(wins, combos, cfg; grid, raw, use_nuts, save_dir,
                               cache_path, rebuild) -> (; qall, fc_store, crps, skipped)

Reload the cached 8j chains for every `origin × combo` and assemble the forecast products
used downstream: `qall` (long quantile table for WIS scoring), `fc_store`
(`(origin,label) → A×H×D` forecast fans for the forecast-vs-observed panels), `crps`
(per-origin native CRPS cross-check) and `skipped` (origin×combo pairs whose reload threw).

This loop (`iterated_forecast` → `fit_or_load_chain` → `generated_quantities`, per origin ×
combo) is the notebook's dominant cost, yet fully determined by the cached chains — so the
result is **cached to `cache_path`** (JLD2) and reused. The cache stores `origins`/`labels`
alongside the products and is treated as **stale** (rebuilt) if either changed; `rebuild=true`
forces a fresh reload. Missing chains fall through to `iterated_forecast`'s own re-fit — run
8j first so the chains exist.
"""
function assemble_or_load_forecasts(wins, combos, cfg;
                                    grid = cis_age_grid(), raw,
                                    use_nuts::Bool = false,
                                    save_dir::AbstractString = "../dt_intermediate",
                                    cache_path::AbstractString =
                                        joinpath(save_dir, "9j_assembly_$(contacts_label(cfg)).jld2"),
                                    rebuild::Bool = false)
    labels  = [string(degree_label(dm), "|", ngm_label(nb)) for (dm, nb) in combos]
    origins = [w.origin for w in wins]
    if !rebuild && isfile(cache_path)
        c = load(cache_path)
        if c["origins"] == origins && c["labels"] == labels
            println("assembly: loaded cache ", cache_path, " (", size(c["qall"], 1), " quantile rows)")
            return (; qall = c["qall"], fc_store = c["fc_store"], crps = c["crps"], skipped = c["skipped"])
        end
        @warn "assembly cache stale (origins/labels changed) — rebuilding" cache_path
    end

    qtabs     = DataFrame[]
    fc_store  = Dict{Tuple{Date,String},Array{Float64,3}}()
    crps_rows = NamedTuple[]
    skipped   = Tuple{Date,String}[]
    t0 = time()
    for (oi, win_o) in enumerate(wins)
        wd_o    = load_window_data(win_o; grid = grid)
        truth_o = load_forecast_truth(win_o; grid = grid)
        # this origin's 4 contact/degree windows (reuse the single raw read); discarded after.
        apd_o = [prepare_degree_data(
                     WeeklyWindow(win_o.origin + Day(7 * h);
                                  n_fit = cfg.n_fit, smax = cfg.smax, horizons = cfg.horizons),
                     cfg; grid = grid, setting = :all,
                     df_part_raw = raw.df_part, craw_raw = raw.craw)
                 for h in cfg.horizons]
        for (dm, nb) in combos
            lbl = string(degree_label(dm), "|", ngm_label(nb))
            try   # keep the multi-origin run alive if a single origin×combo reload is pathological
                fc = iterated_forecast(dm, nb, wd_o, cfg, win_o;
                                       grid = grid, setting = :all, use_nuts = use_nuts,
                                       save_dir = save_dir, apd_by_h = apd_o)
                fc_store[(win_o.origin, lbl)] = fc
                push!(qtabs, to_quantile_long(fc, truth_o, lbl, win_o, cfg, grid.LAB))
                push!(crps_rows, (origin = win_o.origin, model = lbl, mean_crps = mean_crps(fc, truth_o)))
            catch err
                push!(skipped, (win_o.origin, lbl))
                @warn "skipped origin×combo" origin=win_o.origin model=lbl exception=err
            end
        end
        if oi % 5 == 0 || oi == length(wins)
            println("  origin $oi/$(length(wins)) (", win_o.origin, ")  elapsed ",
                    round(Int, time() - t0), "s")
        end
    end
    qall = vcat(qtabs...)
    crps = DataFrame(crps_rows)
    println("quantile rows: ", size(qall), "   (", length(wins), " origins × ", length(combos),
            " combos; skipped ", length(skipped), ")")
    jldsave(cache_path; qall, fc_store, crps, skipped, origins, labels)
    println("assembly: wrote cache ", cache_path)
    return (; qall, fc_store, crps, skipped)
end

# ── Scoring report ────────────────────────────────────────────────────────────────────
"""
    report_forecast_scores(scores, wins, crps) -> by_mh_log

Print the headline log-scale WIS (by model, and by model × horizon) and the mean native
CRPS cross-check, write the four `res/8j_scores_*.csv` frames (both scales), and return the
by-model × horizon log-scale frame `by_mh_log` used by the figure builders.
"""
function report_forecast_scores(scores, wins, crps)
    by_mh_log = sort(@subset(scores.by_model_h, :scale .== "log"), [:model, :horizon])
    by_m_log  = sort(@subset(scores.by_model,   :scale .== "log"), :wis)

    println("\n===== log-scale WIS by model (aggregated over horizons & ", length(wins), " origins) =====")
    show(by_m_log, allcols = true); println()
    println("\n===== log-scale WIS by model × horizon (aggregated over origins) =====")
    show(by_mh_log, allcols = true); println()

    crps_df = sort(combine(groupby(crps, :model), :mean_crps => mean => :mean_crps), :mean_crps)
    println("\nmean native CRPS (avg over origins):")
    show(crps_df, allrows = true); println()

    CSV.write("../res/8j_scores_by_model.csv", scores.by_model)                   # both scales
    CSV.write("../res/8j_scores_by_model_horizon.csv", scores.by_model_h)         # both scales × horizon
    CSV.write("../res/8j_scores_by_model_date.csv", scores.by_model_dt)           # both scales × origin
    CSV.write("../res/8j_scores_by_model_date_horizon.csv", scores.by_model_dt_h) # both scales × origin × horizon
    return by_mh_log
end

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
        # horizon-h contact window ends at origin+h (contacts observed h wks ahead), matching how
        # the (origin, h) chain was fit — so generated_quantities' Cstar[end] is coherent.
        apd_o = prepare_degree_data(
                    WeeklyWindow(win_o.origin + Day(7 * h); n_fit = cfg.n_fit, smax = cfg.smax,
                                 horizons = cfg.horizons),
                    cfg; grid = grid, setting = :all,
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
    reproduction_over_time_or_load(combos, labels4, wins, cfg; grid, raw, h, save_dir,
                                   cache_path, rebuild, verbose) -> store

Cached wrapper around `reproduction_over_time`. The R(t) loop reloads window/degree data and
the cached chain per origin × combo (slow), but is deterministic given the cached chains — so
its `store` is cached to `dt_intermediate/9j_rt_<contacts>_h<h>.jld2` with the same
validate-or-rebuild pattern as `assemble_or_load_forecasts`. The cache self-invalidates when
`origins`/`labels` change; `rebuild=true` forces a fresh compute.
"""
function reproduction_over_time_or_load(combos, labels4, wins, cfg;
        grid = cis_age_grid(), raw, h::Integer = 1, save_dir::AbstractString = "../dt_intermediate",
        cache_path::AbstractString = joinpath(save_dir, "9j_rt_$(contacts_label(cfg))_h$(h).jld2"),
        rebuild::Bool = false, verbose::Bool = true)
    origins = [w.origin for w in wins]
    if !rebuild && isfile(cache_path)
        c = load(cache_path)
        if c["origins"] == origins && c["labels"] == labels4
            println("R(t): loaded cache ", cache_path)
            return c["store"]
        end
        @warn "R(t) cache stale (origins/labels changed) — rebuilding" cache_path
    end
    store = reproduction_over_time(combos, labels4, wins, cfg;
                                   grid = grid, raw = raw, h = h, save_dir = save_dir, verbose = verbose)
    jldsave(cache_path; store, origins, labels = labels4, h)
    return store
end

const _NATIONAL_EST_PATH = joinpath(@__DIR__, "..", "inc2prev", "outputs", "estimates_national.csv")

"""
    national_R(; path, region, d0, d1) -> (; date, med, lo, hi)

Daily inc2prev national R estimate (`name=="R"`, `variable==region`, default England) as
median + 90% band (q5/q95), sorted by date and optionally clipped to `[d0, d1]`. Overlaid on
`plot_reproduction` as an external reference for the NGM-derived R.
"""
function national_R(; path::AbstractString = _NATIONAL_EST_PATH, region::AbstractString = "England",
                    d0 = nothing, d1 = nothing)
    df = CSV.read(path, DataFrame)
    r  = @subset(df, :name .== "R", :variable .== region)
    r  = @transform(r, :date = _todate_safe.(:date))
    d0 !== nothing && (r = @subset(r, :date .>= d0))
    d1 !== nothing && (r = @subset(r, :date .<= d1))
    r = sort(r, :date)
    return (; date = r.date, med = r.median, lo = r.q5, hi = r.q95)
end

"""
    plot_reproduction(store, labels4, model_cols, origins; h) -> Plot

Reproduction number over time. Each model's R is drawn as a **step function** (`:steppost` —
held constant forward from each origin week, matching the NGM frozen at that origin) with a
90% ribbon, over the inc2prev **national R** (England) reference curve. R=1 threshold dashed.
`store` is `reproduction_over_time`'s output; `origins` its window origins.
"""
function plot_reproduction(store, labels4, model_cols, origins; h::Integer = 1)
    x = week_mid.(origins .+ Day(7 * h))   # R derives from the origin+h contact week; plot it there
    fig = plot(; xlabel = "contact week (origin + $(h) wk)",
               ylabel = "reproduction number R  (dominant NGM eigenvalue)",
               title = "9j — reproduction number over time by model (h=$h, step; 90% CI)",
               size = (950, 520), legend = :topleft, xrotation = 45, ylims = (0, 3))
    # Plot real Date-bearing series FIRST so the x-axis is established as a date axis; only
    # THEN add the R=1 hline. A leading synthetic 2-point line on an empty ylims-fixed plot
    # mangles the date ticks (numeric axis locks in before the real dates).
    natR = national_R(; d0 = first(x), d1 = last(x))
    if !isempty(natR.date)
        plot!(fig, natR.date, natR.med; color = :black, lw = 2, label = "inc2prev national R (England)",
              ribbon = (natR.med .- natR.lo, natR.hi .- natR.med), fillalpha = 0.10)
    end
    for (ci, lbl) in enumerate(labels4)
        s = store[lbl]
        plot!(fig, x, s.med; color = model_cols[ci], lw = 1.8, linetype = :steppost,
              marker = :circle, ms = 2, ribbon = (s.med .- s.lo, s.hi .- s.med),
              fillalpha = 0.12, label = lbl)
    end
    hline!(fig, [1.0]; color = :gray, ls = :dash, label = "R = 1")  # threshold, after dates set
    return fig
end

# ══ 8j-style diagnostic figures (reproduced in 9j) ═════════════════════════════════════
# WIS skill, forecast fans and fitted transmission structure — same titles / `res/8j_*.png`
# output names as the 8j notebook, moved here so the 9j cells are one-line `save_show` calls.

"""
    plot_wis_by_horizon(scores, wins, cfg; scale) -> Plot

Mean log-scale WIS vs horizon, one line per model (aggregated over all origins).
"""
function plot_wis_by_horizon(scores, wins, cfg; scale::AbstractString = WIS_SCALE)
    bmh = sort(@subset(scores.by_model_h, :scale .== scale), [:model, :horizon])
    Hn  = length(cfg.horizons)
    fig = plot(; xlabel = "horizon (weeks)", ylabel = "mean log-scale WIS",
               title = "8j — log-scale WIS by horizon ($(length(wins)) origins)",
               size = (760, 420), legend = :topleft, xticks = 1:Hn)
    for m in unique(bmh.model)
        sub = sort(@subset(bmh, :model .== m), :horizon)
        plot!(fig, sub.horizon, sub.wis; marker = :circle, lw = 2, label = m)
    end
    return fig
end

"""
    plot_wis_four_ways(scores, labels4, cfg; scale) -> Plot

Grouped bar of mean log-scale WIS, model on the x-axis, dodged by horizon (lower = better).
"""
function plot_wis_four_ways(scores, labels4, cfg; scale::AbstractString = WIS_SCALE)
    bmh = @subset(scores.by_model_h, :scale .== scale)
    Hn  = length(cfg.horizons)
    # rows = model, cols = horizon; WIS as bar height (not the model index — the old bug)
    Mwis = [only(@subset(bmh, :model .== m, :horizon .== h).wis) for m in labels4, h in 1:Hn]
    return groupedbar(Mwis; bar_position = :dodge,
                      xticks = (1:length(labels4), labels4), xrotation = 20,
                      label = reshape(["h=$h" for h in 1:Hn], 1, :),
                      ylabel = "mean log-scale WIS (lower = better)", legend = :topleft,
                      title = "8j — log-scale WIS by model × horizon",
                      size = (950, 480), bottom_margin = 14Plots.mm, left_margin = 6Plots.mm)
end

"""
    plot_wis_over_time(scores; scale) -> Plot

Mean log-scale WIS over the forecast period, one line per model — full-period skill.
"""
function plot_wis_over_time(scores; scale::AbstractString = WIS_SCALE)
    by_dt = sort(@subset(scores.by_model_dt, :scale .== scale), [:model, :forecast_date])
    fig = plot(; xlabel = "forecast origin", ylabel = "mean log-scale WIS",
               title = "8j — log-scale WIS over the available period",
               size = (900, 420), legend = :topleft)
    for m in unique(by_dt.model)
        sub = @subset(by_dt, :model .== m)
        plot!(fig, sub.forecast_date, sub.wis; lw = 2, marker = :circle, ms = 2, label = m)
    end
    return fig
end

"""
    plot_wis_by_horizon_over_time(scores, labels4, model_cols, cfg; ref, scale) -> Plot

Relative WIS as a time series (one line per config), faceted by horizon (2×2): within each
horizon × forecast_date the log-scale WIS is ratioed to the reference model `ref`, so the
reference sits on the 1.0 line and values < 1 beat it.
"""
function plot_wis_by_horizon_over_time(scores, labels4, model_cols, cfg;
                                       ref::AbstractString = REF_MODEL, scale::AbstractString = WIS_SCALE)
    bdth = @subset(scores.by_model_dt_h, :scale .== scale)
    panels = Plots.Plot[]
    for (k, h) in enumerate(cfg.horizons)
        sub_h = @subset(bdth, :horizon .== h)
        ref_by_date = Dict(r.forecast_date => r.wis for r in eachrow(@subset(sub_h, :model .== ref)))
        p = plot(; title = "horizon $h (wk ahead)", titlefontsize = 8, xlabel = "forecast origin",
                 ylabel = "relative WIS (vs $(ref))", legend = (k == 1 ? :topleft : false),
                 legendfontsize = 6, xrotation = 45)
        for (ci, m) in enumerate(labels4)
            s = sort(@subset(sub_h, :model .== m), :forecast_date)
            rel = [ (haskey(ref_by_date, d) && ref_by_date[d] != 0) ? w / ref_by_date[d] : NaN
                    for (w, d) in zip(s.wis, s.forecast_date) ]
            plot!(p, s.forecast_date, rel; color = model_cols[ci], lw = 1.5,
                  marker = :circle, ms = 2, label = m, ylim=[0,3.0])
        end
        hline!(p, [1.0]; color = :gray, ls = :dash, label = "")  # ref = 1, after dates set
        push!(panels, p)
    end
    return plot(panels...; layout = (2, 2), size = (1150, 780),
                plot_title = "8j — relative WIS over time, by horizon (log scale, vs $(ref))",
                plot_titlefontsize = 11)
end

"""
    plot_forecast_panels(fc_store, wins, labels4, model_cols, cfg; grid, n) -> Plot

Forecast vs observed at `n` evenly-spaced origins. Each panel: one observed series (the
fit-week history ++ the realized target weeks) overlaid with the four configs' total-infection
forecast fans (median + 90% band). Reloads window/truth data for the selected origins.
"""
function plot_forecast_panels(fc_store, wins, labels4, model_cols, cfg;
                              grid = cis_age_grid(), n::Integer = 9)
    origins = [w.origin for w in wins]
    sel = pick_origins(origins; n = n)
    qs_lo, qs_hi = 0.05, 0.95
    H = length(cfg.horizons)
    panels = Plots.Plot[]
    for (pi, origin) in enumerate(sel)
        win   = wins[findfirst(==(origin), origins)]
        wd    = load_window_data(win; grid = grid)          # observed history (A × all_weeks)
        truth = load_forecast_truth(win; grid = grid)       # observed target weeks (A × H)
        # one continuous observed line: fit weeks (cols smax+1:end) ++ the H forecast weeks
        x_hist = week_mid.(win.fit_weeks)
        y_hist = vec(sum(wd.I_mean[:, (cfg.smax + 1):end]; dims = 1))
        x_fore = week_mid.(win.forecast_weeks)
        y_fore = [sum(truth[:, h]) for h in 1:H]
        p = plot(; title = string(origin), titlefontsize = 7, xrotation = 45,
                 legend = (pi == 1 ? :topleft : false), legendfontsize = 5)
        plot!(p, vcat(x_hist, x_fore), vcat(y_hist, y_fore);
              color = :black, lw = 2, marker = :circle, ms = 2, label = "observed")
        vline!(p, [week_mid(win.origin)]; color = :gray, ls = :dash, lw = 1, label = "")
        for (ci, lbl) in enumerate(labels4)
            haskey(fc_store, (origin, lbl)) || continue      # skipped origin×combo → gap
            tot = dropdims(sum(fc_store[(origin, lbl)]; dims = 1); dims = 1)   # H × draws
            med = [median(tot[h, :]) for h in 1:H]
            lo  = [quantile(tot[h, :], qs_lo) for h in 1:H]
            hi  = [quantile(tot[h, :], qs_hi) for h in 1:H]
            plot!(p, x_fore, med; color = model_cols[ci], lw = 1.6,
                  ribbon = (med .- lo, hi .- med), fillalpha = 0.10, label = (pi == 1 ? lbl : ""))
        end
        push!(panels, p)
    end
    return plot(panels...; layout = (3, 3), size = (1300, 1000),
                plot_title = "8j — total-infection forecast (four ways) vs observed, by origin (90% band)",
                plot_titlefontsize = 11)
end

# ── Fitted transmission structure (susceptibility / infectivity / GP length-scales) ────
"""
    collect_transmission_structure(labels4, origins; grid, h) -> (; susc, inf, rho, gamma)

Per-model × origin summary (median + 90% band) of the fitted transmission structure from the
cached `h`-chains: `susc`/`inf` are ratios of the 16-49 and >50 super-groups to 2-15
(≡ 1 by construction); `rho` holds the three GP length-scales (col 1 ρ_diag total-age, col 2
ρ_gap age-gap, both age-yrs; col 3 ρ_time weeks — `NaN` for pooled chains); `gamma` holds the
scalar absolute-transmissibility level γ (window-relative). `susc`/`inf` stores are
`Dict(label => (med, lo, hi))` of `nO × 2` matrices, `rho` of `nO × 3`, `gamma` of `nO × 1`;
missing chains leave `NaN` gaps. Reuses `load_transmission_draws` + `aggregate_supergroups`.
"""
function collect_transmission_structure(labels4, origins; grid = cis_age_grid(), h::Integer = 1)
    nO = length(origins)
    mkstore(k) = Dict(l => (med = fill(NaN, nO, k), lo = fill(NaN, nO, k), hi = fill(NaN, nO, k))
                      for l in labels4)
    susc_store, inf_store, rho_store, gamma_store = mkstore(2), mkstore(2), mkstore(3), mkstore(1)
    for lbl in labels4, (oi, origin) in enumerate(origins)
        d = load_transmission_draws(lbl, origin, h)     # nothing if chain missing → leaves NaN gap
        d === nothing && continue
        for (V, dst) in ((d.susc, susc_store), (d.inf, inf_store))
            sg = aggregate_supergroups(V, grid.POP)      # ndraws × 3 (2-15, 16-49, >50)
            r  = sg[:, 2:3] ./ sg[:, 1]                  # ratios vs 2-15
            for g in 1:2
                dst[lbl].med[oi, g] = median(r[:, g])
                dst[lbl].lo[oi, g]  = quantile(r[:, g], 0.05)
                dst[lbl].hi[oi, g]  = quantile(r[:, g], 0.95)
            end
        end
        for (g, rv) in enumerate((d.rho_diag, d.rho_gap, d.rho_time))  # ρ_diag,ρ_gap,ρ_time → cols 1,2,3
            all(isnan, rv) && continue                   # ρ_time absent (pooled) → leave NaN
            rho_store[lbl].med[oi, g] = median(rv)
            rho_store[lbl].lo[oi, g]  = quantile(rv, 0.05)
            rho_store[lbl].hi[oi, g]  = quantile(rv, 0.95)
        end
        γ = d.γ                                          # scalar-per-draw (not per-age) → no super-groups
        gamma_store[lbl].med[oi, 1] = median(γ)
        gamma_store[lbl].lo[oi, 1]  = quantile(γ, 0.05)
        gamma_store[lbl].hi[oi, 1]  = quantile(γ, 0.95)
    end
    return (; susc = susc_store, inf = inf_store, rho = rho_store, gamma = gamma_store)
end

"""
    plot_ratio(store, labels4, origins, ttl) -> Plot

2×2 facet (one panel per config) of a super-group ratio-to-2-15 store from
`collect_transmission_structure` (susceptibility or infectivity): the 16-49 and >50 series
with 90% ribbons, referenced to 1.0 (the 2-15 baseline).
"""
function plot_ratio(store, labels4, origins, ttl::AbstractString)
    gnames = ["16-49", ">50"]
    ps = Plots.Plot[]
    for (k, lbl) in enumerate(labels4)
        p = plot(; title = lbl, titlefontsize = 8, xlabel = "forecast origin",
                 ylabel = "ratio to 2-15", legend = (k == 1 ? :topright : false),
                 legendfontsize = 6, xrotation = 45)
        # Reference at 1 as a Date-valued series FIRST → establishes the date x-axis.
        # (A leading `hline!` here initialises a numeric axis and collapses the Dates.)
        plot!(p, [first(origins), last(origins)], [1.0, 1.0]; color = :gray, ls = :dash, label = "")
        for g in 1:2
            m, lo, hi = store[lbl].med[:, g], store[lbl].lo[:, g], store[lbl].hi[:, g]
            plot!(p, origins, m; lw = 1.8, marker = :circle, ms = 2, label = gnames[g],
                  ribbon = (m .- lo, hi .- m), fillalpha = 0.15)
        end
        push!(ps, p)
    end
    return plot(ps...; layout = (2, 2), size = (1150, 780), plot_title = ttl, plot_titlefontsize = 11)
end

"""
    plot_lengthscales(rho, labels4, origins; h) -> Plot

2×2 facet of the separable spatio-temporal-GP length-scales per config: ρ_diag (total-age, solid)
and ρ_gap (age-gap, dashed) in age-years, plus ρ_time (temporal, dotted) in weeks, each with a 90%
ribbon. The three share one axis (units age-yrs / weeks; ρ_time ∈ [0.5,26]w, the spatial ρ ∈ [3,45]y);
ρ_time is absent (NaN, not plotted) for pooled chains.
"""
function plot_lengthscales(rho, labels4, origins; h::Integer = 1)
    rho_dirs = ["ρ_diag (total age, yr)", "ρ_gap (age gap, yr)", "ρ_time (weeks)"]
    rho_ls   = [:solid, :dash, :dot]
    panels = Plots.Plot[]
    for lbl in labels4
        p = plot(; title = lbl, titlefontsize = 8, xlabel = "forecast origin",
                 ylabel = "GP length-scale (age-yrs / weeks)", legend = (lbl == labels4[1] ? :topright : false),
                 legendfontsize = 6, xrotation = 45, ylims = (0, 50))
        for g in 1:3
            m, lo, hi = rho[lbl].med[:, g], rho[lbl].lo[:, g], rho[lbl].hi[:, g]
            all(isnan, m) && continue                    # skip ρ_time for pooled chains
            plot!(p, origins, m; lw = 1.8, marker = :circle, ms = 2, ls = rho_ls[g], label = rho_dirs[g],
                  ribbon = (m .- lo, hi .- m), fillalpha = 0.15)
        end
        push!(panels, p)
    end
    return plot(panels...; layout = (2, 2), size = (1150, 780),
                plot_title = "8j — separable GP length-scales ρ_diag / ρ_gap / ρ_time over time (h=$h)",
                plot_titlefontsize = 11)
end

"""
    plot_gamma(store, labels4, model_cols, origins; h) -> Plot

Absolute-transmissibility γ over the forecast origins, one line per model config (median + 90%
ribbon) in a single panel — γ is a scalar-per-draw (one value per model×origin), so unlike
`plot_ratio` there are no per-age super-groups to facet. `store` is the `gamma` field of
`collect_transmission_structure` (`Dict(label => (med, lo, hi))` of `nO × 1` matrices).

NOTE: γ is **window-relative** — each window normalises `C*` to unit fit-window mean intensity,
so γ carries the absolute NGM level *for that window* and its cross-origin level is partly a
normalisation artefact. Read it as a within-origin level, not a clean temporal trend. γ has no
natural reference level (unlike the ratio=1 / R=1 lines), so none is drawn.
"""
function plot_gamma(store, labels4, model_cols, origins; h::Integer = 1)
    # Plot the real Date-bearing series directly (no leading synthetic/`hline!` line) so the
    # x-axis stays a date axis — see the gotcha in `plot_ratio` / `plot_reproduction`.
    fig = plot(; xlabel = "forecast origin", ylabel = "γ (absolute transmissibility, window-relative)",
               title = "8j — absolute transmissibility γ over time by model (h=$h; 90% CI)",
               size = (950, 520), legend = :topright, xrotation = 45)
    for (ci, lbl) in enumerate(labels4)
        m, lo, hi = store[lbl].med[:, 1], store[lbl].lo[:, 1], store[lbl].hi[:, 1]
        plot!(fig, origins, m; color = model_cols[ci], lw = 1.8, marker = :circle, ms = 2,
              ribbon = (m .- lo, hi .- m), fillalpha = 0.12, label = lbl)
    end
    return fig
end
