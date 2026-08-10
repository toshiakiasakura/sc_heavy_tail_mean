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
# Reference model = `unweighted-negbin|mean-diagonal`, the real NO-INTERACTION model added
# 2026-07-30 (inst/6_null_interaction_model.md). Until then `unweighted-negbin|mean` *stood in* for
# Munday's no-interaction reference because no diagonal-only variant existed; it does now, so the
# relative-skill baseline is the genuine article. WIS on the LOG scale is the headline (robust —
# the neighbourhood-NGM natural-scale WIS blows up).
#
# Depends on the forecasting framework (`forecast_utils.jl`) and the read-only two-stage helpers
# in `8j_viz_utils.jl` (`reproduction_draws`, `load_transmission_draws`, …), included below.

include("8j_viz_utils.jl")

const REF_MODEL = "unweighted-negbin|mean-diagonal"  # relative-skill ref = the no-interaction model
const WIS_SCALE = "log"                       # headline scale for WIS / coverage frames
const LOGS_SCALE = "natural"                  # headline scale for the log score (see report_logscore)

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

# Compact labels for the named periods, used when shading them onto a narrow date-axis panel.
const PERIOD_ABBR = Dict(
    "Lockdown 2"              => "L2",
    "Lockdown 2 Easing"       => "L2 easing",
    "Christmas"               => "Xmas",
    "Lockdown 3"              => "L3",
    "Lockdown 3 Schools open" => "L3 schools",
    "Lockdown 3 Easing"       => "L3 easing",
    "Opening up"              => "Opening up",
)

"""
    shade_periods!(p, dmin, dmax; alpha, fontsize, labels) -> p

Overlay the named UK COVID `PERIODS` (Table 2 — the same bands `plot_rwis_by_period` facets on)
as alternating translucent vertical spans on a Date-axis panel `p`, clipped to `[dmin, dmax]`,
each annotated (rotated 90°) with its abbreviated name near the panel top. Call this **after** the
data series are plotted so the Date axis and y-limits are already established (a leading numeric
overlay would collapse the date axis — see the date-axis gotcha). Bands use a low `fillalpha` so
the lines underneath stay legible.
"""
function shade_periods!(p, dmin::Date, dmax::Date; alpha::Real = 0.10,
                        fontsize::Integer = 5, labels::Bool = true)
    yl   = Plots.ylims(p)
    ytxt = yl[1] + 0.93 * (yl[2] - yl[1])
    cols = (:gray, :steelblue)
    j = 0
    for (nm, lo, hi) in PERIODS
        plo = max(lo, dmin); phi = min(hi, dmax)
        plo <= phi || continue
        j += 1
        vspan!(p, [plo, phi]; color = cols[mod1(j, 2)], fillalpha = alpha,
               linealpha = 0, label = "")
        labels || continue
        mid = plo + Day(div((phi - plo).value, 2))
        annotate!(p, mid, ytxt,
                  text(get(PERIOD_ABBR, nm, nm), fontsize, :center, :bottom; rotation = 90))
    end
    return p
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

"""
    panel_grid(np) -> (nrow, ncol)

Near-square panel layout for `np` subplots. DERIVE the layout, never hard-code it
(`tasks/lessons.md` 2026-07-15): the per-model facets were `(2, 2)` while there were exactly four
models, and silently lose panels the moment the model count changes (it is now six).
"""
function panel_grid(np::Int)
    nr = max(1, floor(Int, sqrt(np)))
    return (nr, ceil(Int, np / nr))
end

# ── Forecast assembly + on-disk cache ─────────────────────────────────────────────────
"""
    assemble_or_load_forecasts(wins, combos, cfg; grid, raw, save_dir,
                               cache_path, rebuild) -> (; qall, fc_store, truth_store, crps, skipped)

Reload the cached 8j chains for every `origin × combo` and assemble the forecast products
used downstream: `qall` (long quantile table for WIS scoring), `fc_store`
(`(origin,label) → A×H×D` forecast fans for the forecast-vs-observed panels), `truth_store`
(`origin → A×H` realized infections, so `score_logs` needn't re-read inc2prev per origin), `crps`
(per-origin native CRPS cross-check) and `skipped` (origin×combo pairs whose reload threw).

This loop (`two_stage_forecast` → `fit_or_load_stage2` → pooled draws, per origin × combo) is the
notebook's dominant cost, yet fully determined by the cached Stage-1 chains + Stage-2 pooled files —
so the result is **cached to `cache_path`** (JLD2) and reused. The cache stores `origins`/`labels`
alongside the products and is treated as **stale** (rebuilt) if either changed; `rebuild=true`
forces a fresh reload. Missing artefacts fall through to `two_stage_forecast`'s own re-fit — run
8j first so the Stage-1/Stage-2 files exist.
"""
function assemble_or_load_forecasts(wins, combos, cfg;
                                    grid = cis_age_grid(), raw,
                                    save_dir::AbstractString = "../dt_intermediate",
                                    cache_path::AbstractString =
                                        joinpath(save_dir, "9j_assembly_$(contacts_label(cfg)).jld2"),
                                    rebuild::Bool = false)
    labels  = [string(degree_label(dm), "|", ngm_label(nb)) for (dm, nb) in combos]
    origins = [w.origin for w in wins]
    if !rebuild && isfile(cache_path)
        c = load(cache_path)
        # `truth_store` was added with the log score (2026-07-30); a cache written before that is
        # missing the key, so treat it as stale rather than KeyError-ing downstream.
        if c["origins"] == origins && c["labels"] == labels && haskey(c, "truth_store")
            println("assembly: loaded cache ", cache_path, " (", size(c["qall"], 1), " quantile rows)")
            return (; qall = c["qall"], fc_store = c["fc_store"], truth_store = c["truth_store"],
                      crps = c["crps"], skipped = c["skipped"])
        end
        @warn "assembly cache stale (origins/labels changed, or pre-truth_store) — rebuilding" cache_path
    end

    qtabs       = DataFrame[]
    fc_store    = Dict{Tuple{Date,String},Array{Float64,3}}()
    truth_store = Dict{Date,Matrix{Float64}}()
    crps_rows   = NamedTuple[]
    skipped     = Tuple{Date,String}[]
    t0 = time()
    for (oi, win_o) in enumerate(wins)
        wd_o    = load_window_data(win_o; grid = grid)
        truth_o = load_forecast_truth(win_o; grid = grid)
        truth_store[win_o.origin] = Float64.(truth_o)
        # this origin's 4 contact/degree windows (reuse the single raw read); discarded after.
        apd_o = [prepare_degree_data(degree_window(win_o.origin, h, cfg), cfg;
                                     grid = grid, setting = :all,
                                     df_part_raw = raw.df_part, craw_raw = raw.craw)
                 for h in cfg.horizons]
        for (dm, nb) in combos
            lbl = string(degree_label(dm), "|", ngm_label(nb))
            try   # keep the multi-origin run alive if a single origin×combo reload is pathological
                fc = two_stage_forecast(dm, nb, wd_o, cfg, win_o;
                                        grid = grid, setting = :all,
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
    jldsave(cache_path; qall, fc_store, truth_store, crps, skipped, origins, labels)
    println("assembly: wrote cache ", cache_path)
    return (; qall, fc_store, truth_store, crps, skipped)
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

"""
    report_logscore(logs, wins; scale = LOGS_SCALE) -> by_mh

Print the headline log score (by model, and by model × horizon), restate the sanitisation tallies,
and write the four `res/8j_logscore_*.csv` frames (both scales). Returns the by-model × horizon
frame on `scale`, used by the log-score figure builders.

The headline scale is **natural**, not log: unlike WIS — whose natural-scale values are dominated by
the neighbourhood NGM's blow-ups — the log score is already a density-based quantity, and its
log-scale variant additionally depends on the `pmax(·,0)+1` censoring applied in `score_logs`. Both
scales are written to CSV.
"""
function report_logscore(logs, wins; scale::AbstractString = LOGS_SCALE)
    by_mh = sort(@subset(logs.by_model_h, :scale .== scale), [:model, :horizon])
    by_m  = sort(@subset(logs.by_model,   :scale .== scale), :log_score)

    println("\n===== $(scale)-scale LOG SCORE by model (aggregated over horizons & ",
            length(wins), " origins; lower = better) =====")
    show(by_m, allcols = true); println()
    println("\n===== $(scale)-scale log score by model × horizon (aggregated over origins) =====")
    show(by_mh, allcols = true); println()
    println("\nsanitisation: dropped ", logs.dropped.n, "/", logs.dropped.total,
            " non-finite draws (", round(100 * logs.dropped.frac; digits = 3), "%); censored ",
            logs.floored.n, "/", logs.floored.total, " negative draws at 0 for the log scale (",
            round(100 * logs.floored.frac; digits = 3), "%); ",
            logs.n_nonfinite_logscore, " non-finite unit scores")

    CSV.write("../res/8j_logscore_by_model.csv", logs.by_model)
    CSV.write("../res/8j_logscore_by_model_horizon.csv", logs.by_model_h)
    CSV.write("../res/8j_logscore_by_model_date.csv", logs.by_model_dt)
    CSV.write("../res/8j_logscore_by_model_date_horizon.csv", logs.by_model_dt_h)
    return by_mh
end

# ── Log-score figures (siblings of the WIS ones) ──────────────────────────────────────
"""`plot_logscore_by_horizon(logs, wins, cfg; scale)` — mean log score vs horizon, one line per
model (sibling of `plot_wis_by_horizon`; lower = better)."""
function plot_logscore_by_horizon(logs, wins, cfg; scale::AbstractString = LOGS_SCALE)
    bmh = sort(@subset(logs.by_model_h, :scale .== scale), [:model, :horizon])
    fig = plot(; xlabel = "horizon (weeks)", ylabel = "mean log score (lower = better)",
               title = "8j — $(scale)-scale log score by horizon ($(length(wins)) origins)",
               size = (760, 420), legend = :topleft, xticks = 1:length(cfg.horizons))
    for m in unique(bmh.model)
        sub = sort(@subset(bmh, :model .== m), :horizon)
        plot!(fig, sub.horizon, sub.log_score; marker = :circle, lw = 2, label = m)
    end
    return fig
end

"""`plot_logscore_by_model_horizon(logs, labels, cfg; scale)` — grouped bars, model × horizon
(sibling of `plot_wis_by_model_horizon`)."""
function plot_logscore_by_model_horizon(logs, labels, cfg; scale::AbstractString = LOGS_SCALE)
    bmh = @subset(logs.by_model_h, :scale .== scale)
    Hn  = length(cfg.horizons)
    M   = [only(@subset(bmh, :model .== m, :horizon .== h).log_score) for m in labels, h in 1:Hn]
    return groupedbar(M; bar_position = :dodge,
                      xticks = (1:length(labels), labels), xrotation = 20,
                      label = reshape(["h=$h" for h in 1:Hn], 1, :),
                      ylabel = "mean log score (lower = better)", legend = :topleft,
                      title = "8j — $(scale)-scale log score by model × horizon",
                      size = (950, 480), bottom_margin = 14Plots.mm, left_margin = 6Plots.mm)
end

"""`plot_logscore_over_time(logs; scale)` — mean log score per forecast origin, one line per model
(sibling of `plot_wis_over_time`)."""
function plot_logscore_over_time(logs; scale::AbstractString = LOGS_SCALE)
    by_dt = sort(@subset(logs.by_model_dt, :scale .== scale), [:model, :forecast_date])
    fig = plot(; xlabel = "forecast origin", ylabel = "mean log score (lower = better)",
               title = "8j — $(scale)-scale log score over the available period",
               size = (900, 420), legend = :topleft)
    for m in unique(by_dt.model)
        sub = @subset(by_dt, :model .== m)
        plot!(fig, sub.forecast_date, sub.log_score; lw = 2, marker = :circle, ms = 2, label = m)
    end
    return fig
end

"""
    plot_rel_logscore_by_horizon(logs, labels, model_cols, cfg; ref, scale) -> Plot

Log score **relative to `ref`**, by horizon — the log-score analogue of panel A of
`plot_rwis_bias_by_horizon`. Reported as a **difference** (`logs_m − logs_ref`), not a ratio: a log
score is not sign-stable (it turns negative wherever the predictive density exceeds 1), so a ratio
is uninterpretable. Below the zero line = better than the reference.
"""
function plot_rel_logscore_by_horizon(logs, labels, model_cols, cfg;
                                      ref::AbstractString = REF_MODEL,
                                      scale::AbstractString = LOGS_SCALE)
    bmh   = @subset(logs.by_model_h, :scale .== scale)
    ref_h = Dict(r.horizon => r.log_score for r in eachrow(@subset(bmh, :model .== ref)))
    fig = plot(; xlabel = "horizon (weeks)", ylabel = "Δ log score (vs $(ref))",
               title = "9j — log score relative to $(ref) ($(scale) scale; lower = better)",
               titlefontsize = 10, legend = :topleft, legendfontsize = 6,
               size = (860, 440), xticks = 1:length(cfg.horizons))
    hline!(fig, [0.0]; color = :gray, ls = :dash, label = "")
    for (ci, m) in enumerate(labels)
        s = sort(@subset(bmh, :model .== m), :horizon)
        isempty(s) && continue
        d = s.log_score .- [get(ref_h, h, NaN) for h in s.horizon]
        plot!(fig, s.horizon, d; color = model_cols[ci], lw = 2, marker = :circle, ms = 3, label = m)
    end
    return fig
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
        # R reads the Stage-2 pooled file directly (Cstar_end = contacts at origin+h, antibody at
        # origin) — no degree-data rebuild needed under the two-stage cut.
        for ((dm, nb), lbl) in zip(combos, labels4)
            R = reproduction_draws(dm, nb, wd_o, cfg, win_o; h = h, save_dir = save_dir)
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
    weekly_national_R(; region) -> Dict{Date,Float64}

Weekly inc2prev national R keyed by `week_start` (Sunday grid key): the daily `national_R`
median averaged over each week. Used by `plot_wis_vs_rt` to look up the Rt at a forecast
target week. Reuses `national_R` (no second CSV path).
"""
function weekly_national_R(; region::AbstractString = "England")
    natR = national_R(; region = region)
    sums = Dict{Date,Float64}(); cnts = Dict{Date,Int}()
    for (d, m) in zip(natR.date, natR.med)
        (d === nothing || !isfinite(m)) && continue
        w = week_start(d)
        sums[w] = get(sums, w, 0.0) + m
        cnts[w] = get(cnts, w, 0) + 1
    end
    return Dict(w => sums[w] / cnts[w] for w in keys(sums))
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

# ── Relative contact reproduction number: ρ(C*) / ρ(C*_ref), contacts only ────────────
"""
    relative_contact_reproduction_over_time(combos, labels4, wins, cfg; grid, h, save_dir, verbose)
        -> Dict(label => (; med, lo, hi))

For each forecast origin and model, the median and 90% band of the **relative contact reproduction
number** `ρ(C*_origin) / ρ(C*_ref)`, where `ρ(C*)` is the dominant eigenvalue of the origin-week
contact matrix alone (`contact_reproduction_draws` — no γ_SAR/susc/inf/antibody) and the reference is
the **first forecast origin** (`wins[1]`). Each origin's per-draw ρ is divided by the reference
origin's *median* ρ (the reference window is the fixed anchor), so every model's curve passes through
1.0 at the first origin. Contact-only ⇒ isolates how contact structure alone drove transmissibility
relative to the baseline week. Reads the Stage-2 pooled files directly — read-only, no re-fit.
"""
function relative_contact_reproduction_over_time(combos, labels4, wins, cfg;
                                                 grid = cis_age_grid(),
                                                 h::Integer = 1,
                                                 save_dir::AbstractString = "../dt_intermediate",
                                                 verbose::Bool = true)
    nO = length(wins)
    # pass 1: raw per-draw spectral radii ρ(C*) per origin per label (nothing where a file is missing)
    raw_rho = Dict(l => Vector{Union{Nothing,Vector{Float64}}}(nothing, nO) for l in labels4)
    t0 = time()
    for (oi, win_o) in enumerate(wins)
        for ((dm, nb), lbl) in zip(combos, labels4)
            ρ = contact_reproduction_draws(dm, nb, cfg, win_o; h = h, save_dir = save_dir)
            ρ === nothing && continue
            raw_rho[lbl][oi] = ρ
        end
        verbose && (oi % 5 == 0 || oi == nO) &&
            println("  contact R(t) origin $oi/$nO (", win_o.origin, ")  elapsed ",
                    round(Int, time() - t0), "s")
    end
    # pass 2: per label, anchor to the first origin with finite data, then normalise
    store = Dict(l => (med = fill(NaN, nO), lo = fill(NaN, nO), hi = fill(NaN, nO)) for l in labels4)
    for lbl in labels4
        ref = NaN
        for oi in 1:nO
            r = raw_rho[lbl][oi]
            if r !== nothing && isfinite(median(r)) && median(r) > 0
                ref = median(r)
                break
            end
        end
        isfinite(ref) || continue
        for oi in 1:nO
            r = raw_rho[lbl][oi]
            r === nothing && continue
            rel = r ./ ref
            store[lbl].med[oi] = median(rel)
            store[lbl].lo[oi]  = quantile(rel, 0.05)
            store[lbl].hi[oi]  = quantile(rel, 0.95)
        end
    end
    return store
end

"""
    relative_contact_reproduction_over_time_or_load(combos, labels4, wins, cfg; grid, h, save_dir,
                                                    cache_path, rebuild, verbose) -> store

Cached wrapper around `relative_contact_reproduction_over_time`, caching its `store` to
`dt_intermediate/9j_relrt_<contacts>_h<h>.jld2` with the same origins/labels self-invalidation as
`reproduction_over_time_or_load` (distinct filename ⇒ no clash with the `9j_rt_*` full-R cache).
"""
function relative_contact_reproduction_over_time_or_load(combos, labels4, wins, cfg;
        grid = cis_age_grid(), h::Integer = 1, save_dir::AbstractString = "../dt_intermediate",
        cache_path::AbstractString = joinpath(save_dir, "9j_relrt_$(contacts_label(cfg))_h$(h).jld2"),
        rebuild::Bool = false, verbose::Bool = true)
    origins = [w.origin for w in wins]
    if !rebuild && isfile(cache_path)
        c = load(cache_path)
        if c["origins"] == origins && c["labels"] == labels4
            println("contact R(t): loaded cache ", cache_path)
            return c["store"]
        end
        @warn "contact R(t) cache stale (origins/labels changed) — rebuilding" cache_path
    end
    store = relative_contact_reproduction_over_time(combos, labels4, wins, cfg;
                                                    grid = grid, h = h, save_dir = save_dir, verbose = verbose)
    jldsave(cache_path; store, origins, labels = labels4, h)
    return store
end

"""
    observed_contact_reproduction_over_time(wins, cfg; grid, raw, h, setting, verbose) -> Vector{Float64}

Model-free companion to `relative_contact_reproduction_over_time`: the **observed** relative contact
reproduction number per origin, `ρ(Ê_t)/ρ(Ê_ref)`, where `Ê_t` is the RAW empirical mean-contact matrix
(`AgePairData.emp_mean` — the observed mean number of contacts bin `i` reports with bin `j`, incl. zeros)
for the contact week `origin+h`, taken **directly from the survey data with no GP / no reciprocity
balancing / no model fit at all**. Uses the same horizon-`h` window (`WeeklyWindow(origin+7h)`) and
seed the Stage-1 fit uses, so its week/binning matches the model's `Cstar_end` exactly. `ρ` is the
dominant (Perron) eigenvalue of the nonnegative matrix; the series is anchored to the first origin with
finite, positive `ρ` (=1.0 there), like the model lines. One number per origin (no band — it is a point
estimate from the data). Pass `raw = load_raw_contact_inputs()` to reuse the single CoMix read.
"""
function observed_contact_reproduction_over_time(wins, cfg;
        grid = cis_age_grid(), raw, h::Integer = 1, setting::Symbol = :all,
        verbose::Bool = true)
    nO = length(wins)
    ρ  = fill(NaN, nO)
    t0 = time()
    for (oi, win_o) in enumerate(wins)
        apd = prepare_degree_data(degree_window(win_o.origin, h, cfg), cfg;
                                   grid = grid, setting = setting,
                                   df_part_raw = raw.df_part, craw_raw = raw.craw)
        E = apd.emp_mean[end, :, :]              # observed mean-contact matrix at week origin+h
        ρ[oi] = maximum(real(eigvals(E)))        # Perron root of the raw observed contact matrix
        verbose && (oi % 5 == 0 || oi == nO) &&
            println("  obs contact R(t) origin $oi/$nO (", win_o.origin, ")  elapsed ",
                    round(Int, time() - t0), "s")
    end
    ref = NaN
    for oi in 1:nO
        if isfinite(ρ[oi]) && ρ[oi] > 0
            ref = ρ[oi]; break
        end
    end
    return isfinite(ref) ? ρ ./ ref : ρ
end

"""
    observed_contact_reproduction_over_time_or_load(wins, cfg; grid, raw, h, setting, save_dir,
                                                    cache_path, rebuild, verbose) -> Vector{Float64}

Cached wrapper around `observed_contact_reproduction_over_time`, caching the observed relative-R series
to `dt_intermediate/9j_obsrt_<contacts>_h<h>.jld2` (distinct filename ⇒ no clash with `9j_rt_*` /
`9j_relrt_*`). Self-invalidates when `origins` change; `rebuild=true` forces a fresh compute.
"""
function observed_contact_reproduction_over_time_or_load(wins, cfg;
        grid = cis_age_grid(), raw, h::Integer = 1, setting::Symbol = :all,
        save_dir::AbstractString = "../dt_intermediate",
        cache_path::AbstractString = joinpath(save_dir, "9j_obsrt_$(contacts_label(cfg))_h$(h).jld2"),
        rebuild::Bool = false, verbose::Bool = true)
    origins = [w.origin for w in wins]
    if !rebuild && isfile(cache_path)
        c = load(cache_path)
        if c["origins"] == origins
            println("obs contact R(t): loaded cache ", cache_path)
            return c["rel"]
        end
        @warn "obs contact R(t) cache stale (origins changed) — rebuilding" cache_path
    end
    rel = observed_contact_reproduction_over_time(wins, cfg; grid = grid, raw = raw,
                                                  h = h, setting = setting, verbose = verbose)
    jldsave(cache_path; rel, origins, h)
    return rel
end

"""
    plot_relative_contact_reproduction(store, labels4, model_cols, origins; h, observed) -> Plot

Relative contact reproduction number over time — `ρ(C*)/ρ(C*_ref)` (contacts only, reference = first
origin). Same step-function/90%-ribbon layout and x-axis as `plot_reproduction`; the anchor is a dashed
line at 1.0, and the left axis has no fixed `ylims`. Pass `observed` (from
`observed_contact_reproduction_over_time`) to overlay a single model-free line built from the RAW weekly
observed mean-contact matrices (`emean`, no GP / no model estimate) — same first-origin anchor, drawn as
a black step (matching the model curves, black diamonds to set it apart).

`national=true` overlays the inc2prev national R (`national_R`, England) on the **same, shared** axis,
absolute and unrescaled. The two are **not commensurable** and sharing an axis does not make them so: the
steps are a dimensionless ratio to the baseline contact week, the red curve is an absolute R. Both happen
to sit near 1, which is exactly the trap — the single 1.0 line is the baseline week for the steps *and*
the epidemic threshold for the red curve at once (its label says both). Read the SHAPES against each
other (does contact-driven transmissibility turn when R turns?); the vertical gap between them means
nothing. Sharing the axis also squeezes R's ~0.75–1.35 range against the steps' far wider swing — the
price of one frame, and why `national=false` (a twin axis is the other way out, but it invites reading a
scaling artifact as agreement; a shared axis at least leaves the conflation visible).
"""
function plot_relative_contact_reproduction(store, labels4, model_cols, origins;
                                            h::Integer = 1, observed = nothing,
                                            national::Bool = true,
                                            region::AbstractString = "England")
    x = week_mid.(origins .+ Day(7 * h))   # same contact-week x as plot_reproduction, so panels align
    fig = plot(; xlabel = "contact week (origin + $(h) wk)",
               # Keep the shared-axis label SHORT: rotated 90° it is measured against the axis
               # HEIGHT, and anything much past the original's ~39 chars clips off the top.
               ylabel = national ? "ρ(C*)/ρ(C*_ref)  &  R  — MIXED UNITS" :
                                   "relative contact R  (ρ(C*) / ρ(C*_ref))",
               title = "9j — relative contact reproduction number (contacts only; ref = first origin; 90% CI)",
               size = (950, 520), legend = :topleft, xrotation = 45)
    # Real Date-bearing series FIRST (establish the date axis), then the reference hline — same
    # date-tick gotcha as plot_reproduction.
    for (ci, lbl) in enumerate(labels4)
        s = store[lbl]
        plot!(fig, x, s.med; color = model_cols[ci], lw = 1.8, linetype = :steppost,
              marker = :circle, ms = 2, ribbon = (s.med .- s.lo, s.hi .- s.med),
              fillalpha = 0.12, label = lbl)
    end
    if observed !== nothing
        # model-free observed weekly means, as a step (matching the model curves); black diamonds
        # keep it distinguishable from the coloured fitted C* steps.
        plot!(fig, x, observed; color = :black, lw = 2.2, ls = :solid, linetype = :steppost,
              marker = :diamond, ms = 3, label = "observed weekly means (raw)")
    end
    # Shared axis, absolute and unrescaled — see the docstring: the units are mixed on purpose.
    # Daily (not :steppost) and unmarked, so the smooth red curve reads as the external reference it
    # is rather than as a fifth step function.
    natR = national ? national_R(; region = region, d0 = first(x), d1 = last(x)) : nothing
    drew_nat = natR !== nothing && !isempty(natR.date)
    if drew_nat
        plot!(fig, natR.date, natR.med; color = :firebrick, lw = 2,
              ribbon = (natR.med .- natR.lo, natR.hi .- natR.med), fillalpha = 0.10,
              label = "inc2prev national R ($(region)) — ABSOLUTE, different quantity")
    end
    # ONE line at 1.0 doing two jobs once the axis is shared: baseline week for the steps, epidemic
    # threshold for the red curve. Spell both out — the coincidence is the figure's main trap.
    hline!(fig, [1.0]; color = :gray, ls = :dash,
           label = drew_nat ? "1.0 — baseline week (steps) & R = 1 (inc2prev)" :
                              "reference (first origin)")
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
    plot_wis_by_model_horizon(scores, labels4, cfg; scale) -> Plot

Grouped bar of mean log-scale WIS, model on the x-axis, dodged by horizon (lower = better).
"""
function plot_wis_by_model_horizon(scores, labels4, cfg; scale::AbstractString = WIS_SCALE)
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
    plot_wis_diff_over_time(scores, labels, model_cols, cfg; ref, scale) -> Plot

**Cumulative** WIS difference over time, faceted by horizon: within each horizon the per-origin
log-scale WIS gap `wis - wis_ref` (against a caller-chosen `ref`, default
`weighted-hweibull|neighbourhood`) is accumulated in date order, so each line is the running total
skill deficit/surplus and its **endpoint = the whole-period WIS difference** vs `ref`. A line ending
below 0 beat `ref` cumulatively; a steadily-rising line loses a little every week. Origins where the
`ref` WIS is absent are skipped (they contribute 0 to the running sum). The `ref` line itself is
omitted (identically 0). `models` restricts which lines are drawn (default the two mean-NGM
configs); each keeps its colour from its index in the full `labels` list, so colours stay
consistent with the other figures. Contrast `plot_wis_by_horizon_over_time`, which ratios per-date
against the global `REF_MODEL`. The named UK COVID `PERIODS` (the same Table-2 bands
`plot_rwis_by_period` facets on) are overlaid as translucent shaded spans with labels via
`shade_periods!`, to read the cumulative gap against the epidemic timeline.
"""
function plot_wis_diff_over_time(scores, labels, model_cols, cfg;
                                 ref::AbstractString = "weighted-hweibull|neighbourhood",
                                 models::Vector{<:AbstractString} = ["unweighted-negbin|mean",
                                                                     "weighted-hweibull|mean"],
                                 scale::AbstractString = WIS_SCALE)
    bdth = @subset(scores.by_model_dt_h, :scale .== scale)
    panels = Plots.Plot[]
    for (k, h) in enumerate(cfg.horizons)
        sub_h = @subset(bdth, :horizon .== h)
        ref_by_date = Dict(r.forecast_date => r.wis for r in eachrow(@subset(sub_h, :model .== ref)))
        p = plot(; title = "horizon $h (wk ahead)", titlefontsize = 8, xlabel = "forecast origin",
                 ylabel = "cumulative WIS diff", legend = (k == 1 ? :topleft : false),
                 legendfontsize = 6, xrotation = 45)
        for (ci, m) in enumerate(labels)
            (m == ref || m ∉ models) && continue         # ref ≡ 0; keep only requested models
            s = sort(@subset(sub_h, :model .== m), :forecast_date)
            dts = Date[]; cum = Float64[]; acc = 0.0
            for (w, d) in zip(s.wis, s.forecast_date)    # running sum of the per-origin gap, in date order
                haskey(ref_by_date, d) || continue
                acc += w - ref_by_date[d]
                push!(dts, d); push!(cum, acc)
            end
            plot!(p, dts, cum; color = model_cols[ci], lw = 1.5,
                  marker = :circle, ms = 2, label = m)
        end
        hline!(p, [0.0]; color = :gray, ls = :dash, label = "")  # ref = 0, after dates set
        pdates = collect(keys(ref_by_date))              # the origins actually plotted (ref present)
        isempty(pdates) || shade_periods!(p, minimum(pdates), maximum(pdates))  # named-period bands
        push!(panels, p)
    end
    nr, nc = panel_grid(length(cfg.horizons))
    return plot(panels...; layout = (nr, nc), size = (1150, 780),
                plot_title = "9j — cumulative WIS difference over time, by horizon (log scale; < 0 beats $(ref))",
                plot_titlefontsize = 11)
end

"""
    plot_wis_diff_over_time_cumh(scores, labels, model_cols, cfg; ref, models, scale) -> Plot

Horizon-**cumulative** twin of `plot_wis_diff_over_time`, matched to the M-SAP (multi-horizon
aggregated) target. Instead of one panel *per* horizon `h`, panel `H` sums each origin's WIS over
the horizon *set* `1:H` before differencing against `ref` — so the panels are `h=1`, `h=1:2`,
`h=1:3`, `h=1:4` (the `h=1` panel is identical to `plot_wis_diff_over_time`'s). Within each panel
the per-origin multi-horizon gap `Σ_{h≤H} wis − Σ_{h≤H} wis_ref` is accumulated in date order, so a
line's endpoint = the whole-period, all-horizons-through-H WIS difference vs `ref`; < 0 ⇒ beats it
cumulatively across both time and lead-times. Only origins with **all** `H` horizons present
contribute (so the horizon sum is comparable origin-to-origin); the named `PERIODS` are shaded via
`shade_periods!`. `ref`/`models`/`scale` behave as in `plot_wis_diff_over_time`.
"""
function plot_wis_diff_over_time_cumh(scores, labels, model_cols, cfg;
                                      ref::AbstractString = "weighted-hweibull|neighbourhood",
                                      models::Vector{<:AbstractString} = ["unweighted-negbin|mean",
                                                                          "weighted-hweibull|mean"],
                                      scale::AbstractString = WIS_SCALE)
    bdth = @subset(scores.by_model_dt_h, :scale .== scale)
    hs   = sort(collect(cfg.horizons))
    panels = Plots.Plot[]
    for (k, H) in enumerate(hs)
        # sum WIS over the horizon set 1:H per (model, origin); keep only origins with all H present
        sub = @subset(bdth, :horizon .<= H)
        agg = combine(groupby(sub, [:model, :forecast_date]), :wis => sum => :wis, nrow => :nh)
        agg = @subset(agg, :nh .== H)
        ref_by_date = Dict(r.forecast_date => r.wis for r in eachrow(@subset(agg, :model .== ref)))
        ttl = H == 1 ? "horizon 1 (wk ahead)" : "horizons 1–$H (cumulative)"
        p = plot(; title = ttl, titlefontsize = 8, xlabel = "forecast origin",
                 ylabel = "cumulative WIS diff (Σ h≤$H)", legend = (k == 1 ? :topleft : false),
                 legendfontsize = 6, xrotation = 45)
        for (ci, m) in enumerate(labels)
            (m == ref || m ∉ models) && continue         # ref ≡ 0; keep only requested models
            s = sort(@subset(agg, :model .== m), :forecast_date)
            dts = Date[]; cum = Float64[]; acc = 0.0
            for (w, d) in zip(s.wis, s.forecast_date)    # running sum of the per-origin multi-horizon gap
                haskey(ref_by_date, d) || continue
                acc += w - ref_by_date[d]
                push!(dts, d); push!(cum, acc)
            end
            plot!(p, dts, cum; color = model_cols[ci], lw = 1.5,
                  marker = :circle, ms = 2, label = m)
        end
        hline!(p, [0.0]; color = :gray, ls = :dash, label = "")  # ref = 0, after dates set
        pdates = collect(keys(ref_by_date))              # origins actually plotted (ref present)
        isempty(pdates) || shade_periods!(p, minimum(pdates), maximum(pdates))  # named-period bands
        push!(panels, p)
    end
    nr, nc = panel_grid(length(cfg.horizons))
    return plot(panels...; layout = (nr, nc), size = (1150, 780),
                plot_title = "9j — horizon-cumulative WIS difference over time (M-SAP; log scale; < 0 beats $(ref))",
                plot_titlefontsize = 11)
end

"""
    plot_wis_vs_rt(scores, rt_by_week, cfg; models, scale) -> Plot

Scatter of per-origin log-scale WIS against the England Rt at the forecast **target** week
(`week_start(origin) + 7h`), one series per model in `models`, **faceted by horizon** (2×2 over
`cfg.horizons`). Answers "does skill degrade as transmission rises?". `rt_by_week` is the Dict from
`weekly_national_R()`. A dashed vertical line marks Rt = 1 (the growth/decline threshold).
"""
function plot_wis_vs_rt(scores, rt_by_week, cfg;
                        models::Vector{<:AbstractString} = ["unweighted-negbin|mean",
                                                            "weighted-hweibull|neighbourhood"],
                        scale::AbstractString = WIS_SCALE)
    palette = Dict("unweighted-negbin|mean" => :steelblue,
                   "weighted-hweibull|neighbourhood" => :purple)
    panels = Plots.Plot[]
    for (k, h) in enumerate(cfg.horizons)
        bdth = @subset(scores.by_model_dt_h, :scale .== scale, :horizon .== h)
        p = plot(; title = "horizon $h (wk ahead)", titlefontsize = 8,
                 xlabel = "England Rt at target week (origin + $h wk)", ylabel = "WIS (log scale)",
                 legend = (k == 1 ? :topleft : false), legendfontsize = 6)
        for m in models
            s = @subset(bdth, :model .== m)
            xs = Float64[]; ys = Float64[]
            for r in eachrow(s)
                tw = week_start(r.forecast_date) + Day(7h)
                haskey(rt_by_week, tw) || continue
                push!(xs, rt_by_week[tw]); push!(ys, r.wis)
            end
            scatter!(p, xs, ys; label = m, ms = 4, msw = 0.5, color = get(palette, m, :grey40))
        end
        vline!(p, [1.0]; color = :gray, ls = :dash, label = "Rt = 1")
        push!(panels, p)
    end
    nr, nc = panel_grid(length(cfg.horizons))
    return plot(panels...; layout = (nr, nc), size = (1150, 780),
                plot_title = "9j — WIS vs England Rt at target week, by horizon (log scale)",
                plot_titlefontsize = 11)
end

"""
    plot_wis_scatter(scores, cfg; xmodel, ymodel, scale) -> Plot

Per-origin WIS **scatter** of two models against each other, **faceted by horizon** (2×2 over
`cfg.horizons`). Each point is one forecast origin: `x` = `xmodel`'s log-scale WIS,
`y` = `ymodel`'s, paired on `forecast_date` from `by_model_dt_h` (age-aggregated, one WIS per
model × origin × horizon). The dashed `y = x` line is the tie: points **below** it are origins
where `ymodel` scored lower (= better) than `xmodel`, points above are where it did worse. Axes
share an equal square range so the diagonal is at 45°. Default pairing is
`weighted-hweibull|neighbourhood` (y) vs `unweighted-negbin|mean` (x).
"""
function plot_wis_scatter(scores, cfg;
                          xmodel::AbstractString = "unweighted-negbin|mean",
                          ymodel::AbstractString = "weighted-hweibull|neighbourhood",
                          scale::AbstractString = WIS_SCALE)
    bdth = @subset(scores.by_model_dt_h, :scale .== scale)
    panels = Plots.Plot[]
    for (k, h) in enumerate(cfg.horizons)
        sub_h = @subset(bdth, :horizon .== h)
        xby = Dict(r.forecast_date => r.wis for r in eachrow(@subset(sub_h, :model .== xmodel)))
        yby = Dict(r.forecast_date => r.wis for r in eachrow(@subset(sub_h, :model .== ymodel)))
        xs = Float64[]; ys = Float64[]
        for d in sort(collect(keys(xby)))       # paired origins only (both models present)
            haskey(yby, d) || continue
            push!(xs, xby[d]); push!(ys, yby[d])
        end
        p = plot(; title = "horizon $h (wk ahead)", titlefontsize = 8,
                 xlabel = "WIS: $(xmodel)", ylabel = "WIS: $(ymodel)",
                 legend = (k == 1 ? :topleft : false), legendfontsize = 6,
                 xguidefontsize = 6, yguidefontsize = 6)
        if !isempty(xs)
            lim = (0.0, maximum(vcat(xs, ys)) * 1.05)          # equal square range → 45° diagonal
            plot!(p, collect(lim), collect(lim); color = :gray, ls = :dash,
                  label = "y = x (tie)", xlims = lim, ylims = lim)
            scatter!(p, xs, ys; label = "$(length(xs)) origins", ms = 4, msw = 0.5,
                     color = :purple, alpha = 0.7)
        end
        push!(panels, p)
    end
    nr, nc = panel_grid(length(cfg.horizons))
    return plot(panels...; layout = (nr, nc), size = (1150, 820),
                plot_title = "9j — per-origin WIS: $(ymodel) vs $(xmodel), by horizon " *
                             "($(scale) scale; below diagonal ⇒ $(ymodel) better)",
                plot_titlefontsize = 10)
end

"""
    plot_forecast_panels(fc_store, wins, labels4, model_cols, cfg; grid, n) -> Plot

Forecast vs observed at `n` evenly-spaced origins. Each panel: one observed series (the
fit-week history ++ the realized target weeks) overlaid with each config's total-infection
forecast fans (median + 90% band). Reloads window/truth data for the selected origins.

The tile grid and figure size are DERIVED from the panel count (`pick_origins` clamps it to
`min(n, length(wins))`), so `n` is a free knob: a hard-coded `layout` throws
`When doing layout, n (…) < n_override (…)` the moment `n` exceeds it.
"""
function plot_forecast_panels(fc_store, wins, labels4, model_cols, cfg;
                              grid = cis_age_grid(), n::Integer = 15)
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
            med = [_fmed(tot[h, :]) for h in 1:H]                              # finite-robust (fan may be ±Inf)
            lo  = [_fq(tot[h, :], qs_lo) for h in 1:H]
            hi  = [_fq(tot[h, :], qs_hi) for h in 1:H]
            plot!(p, x_fore, med; color = model_cols[ci], lw = 1.6,
                  ribbon = (med .- lo, hi .- med), fillalpha = 0.10, label = (pi == 1 ? lbl : ""))
        end
        push!(panels, p)
    end
    # Tile grid from the panel count: floor(√)-cols / ceil-rows favours a tall grid
    # (9 → 3×3, 12 → 4×3, 15 → 5×3). 430×330 per cell keeps 9 at the former 1300×1000 figure.
    np   = length(panels)
    ncol = max(1, floor(Int, sqrt(np)))
    nrow = ceil(Int, np / ncol)
    return plot(panels...; layout = (nrow, ncol), size = (430 * ncol, 330 * nrow),
                plot_title = "8j — total-infection forecast vs observed, by origin (90% band)",
                plot_titlefontsize = 11)
end

# ── Fitted transmission structure (susceptibility / infectivity / GP length-scales) ────
"""
    collect_transmission_structure(labels4, origins, cfg; grid, h)
        -> (; susc, inf, susc_bin, inf_bin, rho, gamma, gi, F, sigma_inf,
              inf_pinned, F_pinned, sg_names, ref_lab)

Per-model × origin summary (median + 90% band) of the fitted transmission structure from the
two-stage artefacts. `susc`/`inf` are pop-weighted age super-group values, `susc_bin`/`inf_bin` the
same at full per-age-bin resolution, both divided per draw by the **model's own reference bin**
`cfg.ref_bin` (default 4 = "25-34"). That is the gauge the transmission block is identified in —
`model_transmission` splices an exact `1.0` in at `ref_bin` (joint_model.jl) — so this is the SAME
baseline 10j's `make_susc_inf_fig` plots against, and the per-bin stores are in fact the raw pooled
draws (the division is the identity). Bin `cfg.ref_bin` is therefore a flat 1.0 with a zero-width
band in every per-bin figure.

CHANGED 2026-08-04 (user request): the denominator was the pop-weighted 2-15 super-group, chosen so
the figures were gauge-invariant to `ref_bin`. That invariance is deliberately given up in exchange
for reading against the estimation procedure's own reference. `supergroup_split` accordingly splits
the super-group containing `ref_bin` so the reference is a group of its own — `sg_names` carries the
resulting names (`2-15 / 16-24 / 25-34 / 35-49 / 50+` for the default grid) and `ref_lab =
grid.LAB[cfg.ref_bin]` the reference label, both threaded into the plot functions rather than
re-derived there.

`rho` holds the three GP length-scales (col 1 ρ_diag total-age and col 2 ρ_gap age-gap, both
age-yrs; col 3 the AR(1) φ, dimensionless — `NaN` for pooled chains); `gamma` holds the per-contact
secondary attack rate γ_SAR and `sigma_inf` the infection-likelihood observation SD (added
2026-08-09 — it was stored in every `8j_s2_*` but never surfaced).

`inf_pinned` / `F_pinned` are `Dict(label => Bool)` flags saying whether that model's `inf` / `F` is
a **pin rather than an estimate** — `model_transmission` sets `inf = ones(A)` under
`fix_infectivity(nb)` (the diagonal-NGM baseline identifies only `susc_a·inf_a`) and `F = 1.0`
always. Detected from the draws (`all(== 1.0)`), so a label rename cannot silently un-mark it. Pass
them to `plot_ratio`/`plot_F` so a pinned flat line is annotated as such instead of reading as a
fitted result that happens to equal 1. ⚠ A pinned `inf` also means that model's `susc` absorbs the
whole `susc·inf` product and is NOT on the same footing as the other models' `susc`.

Stores are `Dict(label => (med, lo, hi))` of `nO × length(sg_names)` matrices for
`susc`/`inf`, `nO × grid.N` for the `*_bin` pair, `nO × 3` for `rho`, `nO × 1` for `gamma`; missing
artefacts leave `NaN` gaps. Reuses `load_transmission_draws` + `supergroup_split` +
`aggregate_supergroups`.
"""
function collect_transmission_structure(labels4, origins, cfg; grid = cis_age_grid(),
                                        h::Integer = 1)
    nO = length(origins)
    ref = cfg.ref_bin
    sg_groups, sg_names = supergroup_split(grid, ref)      # reference bin isolated as its own group
    nG = length(sg_groups)
    mkstore(k) = Dict(l => (med = fill(NaN, nO, k), lo = fill(NaN, nO, k), hi = fill(NaN, nO, k))
                      for l in labels4)
    susc_store, inf_store, rho_store, gamma_store = mkstore(nG), mkstore(nG), mkstore(3), mkstore(1)
    gi_store = mkstore(2)                               # GI: col 1 = mean (days), col 2 = SD (days)
    F_store  = mkstore(1)                               # leaky antibody-protection factor F (scalar per draw)
    sinf_store = mkstore(1)                             # observation SD σ_inf (scalar per draw)
    susc_bin_store, inf_bin_store = mkstore(grid.N), mkstore(grid.N)
    # PINNED vs FITTED. `model_transmission` pins `inf = ones(A)` whenever `fix_infectivity(nb)`
    # (the no-interaction / DiagonalMeanNGM baseline: a diagonal NGM identifies only the product
    # susc_a·inf_a, so infectivity is not separately estimable) and pins `F = 1.0` for every model
    # (the antibody term is off since 2026-08-04). A pin renders as a flat 1.0 line that is
    # indistinguishable from a fitted parameter that happens to sit at 1 — so detect it and let the
    # figures SAY so. Detected from the DRAWS (`all(== 1.0)`, exact because the pin is a literal
    # `one`/`1.0`, not an estimate) rather than by parsing `lbl` for "mean-diagonal": the label is a
    # display string and a rename would silently un-mark the pin.
    inf_pinned = Dict(l => true for l in labels4)
    F_pinned   = Dict(l => true for l in labels4)
    seen       = Dict(l => false for l in labels4)
    for lbl in labels4, (oi, origin) in enumerate(origins)
        # ⚠ `contacts` MUST be passed explicitly. `load_transmission_draws` defaults it to
        # `CONTACTS_TOKEN`, which `framework.jl` builds from a LITERAL `stage1_use_nuts = true` and
        # so ALWAYS ends `-nuts`, no matter what this `cfg` says. Omitting it here (the state until
        # 2026-08-09) silently pointed every lookup at the NUTS generation — which does not exist on
        # disk — so `load_transmission_draws` returned `nothing` for all 63 origins × 6 models and
        # EVERY figure fed by this function (susc, inf, susc_bin, inf_bin, rho, gamma, gi, F) came
        # out blank: all-NaN stores, panels containing nothing but their reference line. There is no
        # warning, because a missing Stage-2 artefact is a legitimate "skipped origin×combo".
        # This is the same bug class fixed across `10j_viz_utils.jl` on 2026-08-08 (see the
        # `CONTACTS_TOKEN` entry in CLAUDE.md's Gotchas); 9j was missed in that sweep.
        # Symptom to watch for if it ever regresses: this function returning in well under a second.
        d = load_transmission_draws(lbl, origin, h; contacts = contacts_label(cfg))
        d === nothing && continue                       # genuinely missing → leaves a NaN gap
        seen[lbl] = true
        inf_pinned[lbl] &= all(==(1.0), d.inf)          # exact: the pin is a literal `one`, not a fit
        F_pinned[lbl]   &= all(==(1.0), d.F)
        for (V, dst, dstb) in ((d.susc, susc_store, susc_bin_store),
                               (d.inf,  inf_store,  inf_bin_store))
            den = view(V, :, ref)                        # the model's gauge — identically 1 per draw
            sg  = aggregate_supergroups(V, grid.POP; groups = sg_groups)   # ndraws × nG
            r   = sg ./ den                              # ratios vs the reference bin
            for g in 1:nG
                dst[lbl].med[oi, g] = median(r[:, g])
                dst[lbl].lo[oi, g]  = quantile(r[:, g], 0.05)
                dst[lbl].hi[oi, g]  = quantile(r[:, g], 0.95)
            end
            rb = V ./ den                                # ndraws × A — each bin vs its own draw's ref bin
            for a in 1:grid.N
                dstb[lbl].med[oi, a] = median(rb[:, a])
                dstb[lbl].lo[oi, a]  = quantile(rb[:, a], 0.05)
                dstb[lbl].hi[oi, a]  = quantile(rb[:, a], 0.95)
            end
        end
        for (g, rv) in enumerate((d.rho_diag, d.rho_gap, d.rho_time))  # ρ_diag,ρ_gap,ρ_time → cols 1,2,3
            all(isnan, rv) && continue                   # ρ_time absent (pooled) → leave NaN
            rho_store[lbl].med[oi, g] = median(rv)
            rho_store[lbl].lo[oi, g]  = quantile(rv, 0.05)
            rho_store[lbl].hi[oi, g]  = quantile(rv, 0.95)
        end
        γ = d.gamma_sar                                  # scalar-per-draw (not per-age) → no super-groups
        gamma_store[lbl].med[oi, 1] = median(γ)
        gamma_store[lbl].lo[oi, 1]  = quantile(γ, 0.05)
        gamma_store[lbl].hi[oi, 1]  = quantile(γ, 0.95)
        Fd = d.F                                          # leaky antibody-protection factor (scalar per draw)
        F_store[lbl].med[oi, 1] = median(Fd)
        F_store[lbl].lo[oi, 1]  = quantile(Fd, 0.05)
        F_store[lbl].hi[oi, 1]  = quantile(Fd, 0.95)
        si = d.sigma_inf                                  # infection-likelihood observation SD
        sinf_store[lbl].med[oi, 1] = median(si)
        sinf_store[lbl].lo[oi, 1]  = quantile(si, 0.05)
        sinf_store[lbl].hi[oi, 1]  = quantile(si, 0.95)
        # generation interval (estimated since 2026-07-30) → natural scale, DAYS. Col 1 = mean, 2 = SD.
        gm = gi_moments_days(d.w_mu, d.w_sigma)
        for (g, v) in enumerate((gm.mean_days, gm.sd_days))
            gi_store[lbl].med[oi, g] = median(v)
            gi_store[lbl].lo[oi, g]  = quantile(v, 0.05)
            gi_store[lbl].hi[oi, g]  = quantile(v, 0.95)
        end
    end
    # A label with no artefact at all must not be reported as "pinned" — `inf_pinned` starts `true`
    # and an all-missing model would never have it cleared, turning a MISSING model into a
    # confident claim about its parameterisation. `seen` is what distinguishes the two.
    for l in labels4
        seen[l] || (inf_pinned[l] = false; F_pinned[l] = false)
    end
    return (; susc = susc_store, inf = inf_store, susc_bin = susc_bin_store,
              inf_bin = inf_bin_store, rho = rho_store, gamma = gamma_store, gi = gi_store,
              F = F_store, sigma_inf = sinf_store, inf_pinned, F_pinned,
              sg_names, ref_lab = grid.LAB[ref])
end

"""
    plot_gen_interval(gi, labels4, origins, cfg; h=1) -> Plot

One panel per config (layout via `panel_grid`) of the **estimated generation interval** over the rolling
origins: posterior median + 90% band of its natural-scale mean and SD in DAYS, against the prior
band (grey) implied by `cfg.gen_mean_days`/`gen_sd_days` and `cfg.gen_prior_rel_sd`.

This is the identifiability read for the `w` ↔ `γ_SAR` confounding (§3.1): the two are only
separated by this informative prior, so
- posterior band ≈ prior band  ⇒ the data say nothing about the GI, it is carrying prior only;
- posterior pinned at a soft-clamp with a *narrow* band ⇒ clamp compression, not certainty
  (the failure mode γ_SAR hit at log 0.02 in 2026-07-13 — suspect the clamp, not the data).
"""
function plot_gen_interval(gi, labels4, origins, cfg; h::Integer = 1)
    # Prior reference band: each log-parameter swept to its own ±1.645σ with the OTHER held at its
    # centre, then mapped to days. This is a CONDITIONAL band, not the true joint marginal (the GI
    # mean depends on both w_mu and w_sigma), so it is labelled as such — it is a visual reference
    # for "has the posterior moved?", not a calibrated interval.
    wmu0, wv0 = gen_interval_logparams(cfg.gen_mean_days, cfg.gen_sd_days)
    r = cfg.gen_prior_rel_sd
    zq = 1.6448536269514722                                    # 90% two-sided normal quantile
    pri_mean = gi_moments_days(wmu0 .+ zq .* abs(wmu0) * r .* [-1, 0, 1], fill(wv0, 3)).mean_days
    pri_sd   = gi_moments_days(fill(wmu0, 3), wv0 .+ zq .* abs(wv0) * r .* [-1, 0, 1]).sd_days
    series = [("GI mean (days)", 1, :solid, pri_mean), ("GI SD (days)", 2, :dash, pri_sd)]
    panels = Plots.Plot[]
    for lbl in labels4
        p = plot(; title = lbl, titlefontsize = 8, xlabel = "forecast origin",
                 ylabel = "generation interval (days)",
                 legend = (lbl == labels4[1] ? :topright : false),
                 legendfontsize = 6, xrotation = 45)
        # Date-valued series FIRST — a leading hline!/hspan! locks in a numeric axis and mangles
        # the date ticks (the trap plot_ratio/plot_reproduction already document).
        for (nm, g, ls, pri) in series
            m, lo, hi = gi[lbl].med[:, g], gi[lbl].lo[:, g], gi[lbl].hi[:, g]
            all(isnan, m) && continue
            plot!(p, origins, m; lw = 1.8, marker = :circle, ms = 2, markerstrokewidth = 0,
                  ls = ls, label = nm, ribbon = (m .- lo, hi .- m), fillalpha = 0.15)
            plot!(p, origins, fill(pri[2], length(origins)); lw = 1.0, ls = :dot, color = :grey40,
                  label = (nm == series[1][1] ? "prior centre & 90% (other param fixed)" : ""),
                  ribbon = (fill(pri[2] - pri[1], length(origins)),
                            fill(pri[3] - pri[2], length(origins))),
                  fillalpha = 0.08, fillcolor = :grey60)
        end
        push!(panels, p)
    end
    nr, nc = panel_grid(length(panels))          # derive it — six models now, not four
    return plot(panels...; layout = (nr, nc), size = (575 * nc, 390 * nr),
                plot_title = "8j — ESTIMATED generation interval vs prior (h=$h); " *
                             "posterior≈prior ⇒ GI not identified by the data",
                plot_titlefontsize = 10)
end

"""
    _median_ylims(vals; ref, pad, log) -> (lo, hi)

y-limits spanning the finite `vals` (typically ratio **medians**) and the `ref` reference line,
padded `pad` on each side. Used to scale the ratio panels to their median lines while letting wide
90% ribbons run **off-panel** (Plots clips to `ylims`), rather than letting a single huge CI
compress every median to a flat line. With `log = true` the padding is **multiplicative** (and
non-positive values dropped) so the limits stay valid on a `:log10` axis. Falls back to `(0, 2)`
(`(0.5, 2)` on log) when nothing usable is finite.
"""
function _median_ylims(vals; ref::Real = 1.0, pad::Real = 0.05, log::Bool = false)
    v = filter(isfinite, vals)
    log && (v = filter(>(0), v))                 # a log axis can't show ≤ 0
    isempty(v) && return log ? (0.5, 2.0) : (0.0, 2.0)
    lo = min(minimum(v), ref); hi = max(maximum(v), ref)
    if log
        (hi <= lo) && return (lo / 1.5, hi * 1.5)   # all-equal medians ⇒ give the flat line room
        f = (hi / lo)^pad                           # multiplicative pad, symmetric in log
        return (lo / f, hi * f)
    end
    m = pad * (hi - lo)
    m == 0 && (m = 0.05 * max(abs(hi), 1.0))     # all-equal medians ⇒ give the flat line room
    return (lo - m, hi + m)
end

"""
    _prior_ratio_band(cfg; n, q) -> (lo, med, hi)

Prior quantiles of a **single non-reference** susc/inf ratio, by simulation from the exact
generative form in `model_transmission`: `sig ~ N⁺(cfg.susc_inf_sd_prior...)`,
`z ~ N(0,1)`, ratio `= exp(_softclamp(sig·z, log 0.05, log 20))`.

Simulated rather than derived because `sig·z` is a half-normal × normal product (no closed form)
and because the soft-clamp is part of the construction — a band computed from the unclamped
log-normal would not be the prior the model actually uses. Seeded from `cfg.seed`, so the band is
reproducible and does not move between renders.

This is the identifiability read for the age profile, and the reason it matters here specifically:
`susc_inf_sd_prior` was set to N⁺(0, 0.25²) on 2026-07-31 *so that the profile can shrink to
no-variation when the data are silent*. Without a prior reference on the panel there is no way to
tell a fitted flat profile from a prior-driven one — exactly the read `plot_gen_interval` already
provides for the generation interval.

⚠ It is a **per-bin** band. The super-group series in `plot_ratio` are population-weighted means of
several bins, whose prior is tighter (averaging), so the band is a conservative reference there —
labelled as per-bin on the plot rather than silently reused.
"""
function _prior_ratio_band(cfg; n::Integer = 200_000, q = (0.05, 0.5, 0.95))
    rng = Random.Xoshiro(cfg.seed)
    sig = rand(rng, truncated(Normal(cfg.susc_inf_sd_prior...); lower = 0), n)
    z   = randn(rng, n)
    off = exp.(_softclamp.(sig .* z, log(0.05), log(20.0)))
    return Tuple(quantile(off, collect(q)))
end

"""
    _prior_gamma_band(cfg; q) -> (lo, med, hi)

Prior quantiles of γ_SAR: `exp(_softclamp(Normal(cfg.gamma_sar_prior...), log 0.001, log 10))`.
Analytic (a normal quantile pushed through the same clamp the model applies), so no simulation.
"""
function _prior_gamma_band(cfg; q = (0.05, 0.5, 0.95))
    μ, σ = cfg.gamma_sar_prior
    zq = 1.6448536269514722                        # 90% two-sided normal quantile
    lq = (μ - zq * σ, μ, μ + zq * σ)
    return Tuple(exp(_softclamp(x, log(0.001), log(10.0))) for x in lq)
end

"""
    plot_ratio(store, labels4, origins, ttl; gnames, ref_lab, cfg, pinned) -> Plot

One panel per config of a super-group ratio store from `collect_transmission_structure`
(susceptibility or infectivity): each super-group with a 90% ribbon, referenced to 1.0 — the
model's own reference bin `ref_lab` (= `grid.LAB[cfg.ref_bin]`, default "25-34"). Pass `gnames` and
`ref_lab` straight from that call's `sg_names`/`ref_lab` so the two never drift apart.

`ref_lab` is one of the super-groups (`supergroup_split` isolates it) but is NOT drawn as a series:
it is identically 1.0 with a zero-width ribbon, so the dashed grey reference line carries its label
instead. The shared y-limits are set from the **medians** only (`_median_ylims`) so a wide 90% band
on one series runs off-panel instead of flattening every median line.

⚠ **The y-axis is `:log10` (fixed 2026-08-09).** These are RATIOS, constructed in
`model_transmission` as `exp(_softclamp(sig·z, …))` — multiplicatively symmetric about 1, so a
halving and a doubling are the same distance from the reference and must plot that way. On the
linear axis this used to use, the measured median range over the completed grid (**0.12 – 18.1**)
put the entire sub-1 half — *reduced* susceptibility, half the parameter's range by construction —
into a sliver at the bottom, and squashed `mean-diagonal` (0.19–1.9) to a flat line beside
`negbin|neighbourhood` (→18.1). Both per-bin companions (`plot_ratio_bins`,
`plot_susc_inf_bins_ci`) were already `:log10`; this figure was the odd one out.

`cfg` (optional) adds the grey **prior 90% band** from `_prior_ratio_band` — posterior inside it
⇒ the age profile is prior-driven, not estimated. `pinned` (optional, `tr.inf_pinned`) marks the
models whose parameter is a PIN rather than a fit; their panel title is flagged and the prior band
is suppressed, because a pinned value has no prior to be read against.
"""
function plot_ratio(store, labels4, origins, ttl::AbstractString;
                    gnames, ref_lab::AbstractString, cfg = nothing, pinned = nothing)
    yl = _median_ylims(reduce(vcat, [vec(store[l].med) for l in labels4]); log = true)
    pri = cfg === nothing ? nothing : _prior_ratio_band(cfg)
    ps = Plots.Plot[]
    for (k, lbl) in enumerate(labels4)
        is_pin = pinned !== nothing && get(pinned, lbl, false)
        p = plot(; title = is_pin ? "$lbl  [PINNED ≡ 1, not fitted]" : lbl, titlefontsize = 8,
                 titlefontcolor = is_pin ? :firebrick : :black,
                 xlabel = "forecast origin",
                 ylabel = "ratio to $(ref_lab) (log)", legend = (k == 1 ? :topright : false),
                 legendfontsize = 6, xrotation = 45, yscale = :log10, ylims = yl)
        # Reference at 1 as a Date-valued series FIRST → establishes the date x-axis.
        # (A leading `hline!` here initialises a numeric axis and collapses the Dates.)
        plot!(p, [first(origins), last(origins)], [1.0, 1.0]; color = :gray, ls = :dash,
              label = "$(ref_lab) (ref)")
        # Prior band UNDER the posterior series, and only where a prior exists. Drawn as a ribbon on
        # a two-point date series so it cannot disturb the date axis (same trap as the ref line).
        if pri !== nothing && !is_pin
            xs = [first(origins), last(origins)]
            plot!(p, xs, fill(pri[2], 2); color = :grey40, ls = :dot, lw = 1.0,
                  label = (k == 1 ? "prior 90% (per bin)" : ""),
                  ribbon = (fill(pri[2] - pri[1], 2), fill(pri[3] - pri[2], 2)),
                  fillalpha = 0.10, fillcolor = :grey60)
        end
        for g in eachindex(gnames)
            gnames[g] == ref_lab && continue     # ≡ 1 by construction — the dashed line above IS it
            m, lo, hi = store[lbl].med[:, g], store[lbl].lo[:, g], store[lbl].hi[:, g]
            plot!(p, origins, m; lw = 1.8, marker = :circle, ms = 2, label = gnames[g],
                  ribbon = (m .- lo, hi .- m), fillalpha = 0.15)
        end
        push!(ps, p)
    end
    nr, nc = panel_grid(length(ps))
    return plot(ps...; layout = (nr, nc), size = (575 * nc, 390 * nr),
                plot_title = ttl, plot_titlefontsize = 11)
end

"""
    plot_ratio_bins(store, labels4, origins, ttl; grid, ref_lab) -> Plot

One panel per config of a PER-AGE-BIN ratio store (`susc_bin`/`inf_bin` from
`collect_transmission_structure`) — the age-resolved refinement of `plot_ratio`, sharing its
`ref_lab` baseline so the two figures read against the same 1.0 reference. Since that reference is
the model's own gauge, bin `cfg.ref_bin` sits exactly on the dashed 1.0 line. Median lines only:
seven overlapping 90% ribbons are unreadable, so the bands stay in the `plot_ratio` figure. The
y-axis is **log10** — the ratios are multiplicative, so a bin at 2× and one at 0.5× sit
symmetrically about the 1.0 reference; shared limits come from the medians
(`_median_ylims(; log = true)`).
"""
function plot_ratio_bins(store, labels4, origins, ttl::AbstractString; grid = cis_age_grid(),
                         ref_lab::AbstractString)
    cols = palette(:viridis, grid.N)     # age is ordinal → perceptually ordered palette
    yl = _median_ylims(reduce(vcat, [vec(store[l].med) for l in labels4]); log = true)
    ps = Plots.Plot[]
    for (k, lbl) in enumerate(labels4)
        p = plot(; title = lbl, titlefontsize = 8, xlabel = "forecast origin",
                 ylabel = "ratio to $(ref_lab) (log)", legend = (k == 1 ? :topright : false),
                 legendfontsize = 5, background_color_legend = RGBA(1, 1, 1, 0.7),
                 xrotation = 45, yscale = :log10, ylims = yl)
        # Reference at 1 as a Date-valued series FIRST → establishes the date x-axis (see plot_ratio).
        plot!(p, [first(origins), last(origins)], [1.0, 1.0]; color = :gray, ls = :dash, label = "")
        for a in 1:grid.N
            # markerstrokewidth = 0: the default black 1px stroke swamps a 1.5px marker and hides
            # the age colour — including in the legend swatches, which is what identifies the lines.
            plot!(p, origins, store[lbl].med[:, a]; color = cols[a], lw = 1.5,
                  marker = :circle, ms = 1.5, markerstrokewidth = 0, label = grid.LAB[a])
        end
        push!(ps, p)
    end
    nr, nc = panel_grid(length(ps))
    return plot(ps...; layout = (nr, nc), size = (575 * nc, 390 * nr),
                plot_title = ttl, plot_titlefontsize = 11)
end

"""
    plot_susc_inf_bins_ci(susc_bin, inf_bin, lbl, origins, grid; groups, ref_lab) -> Plot

For ONE model `lbl`, a **3×2** grid of the finest age-dependent transmission structure WITH 90%
CIs: rows = age-bin groups (default the 7 CIS bins split 2/2/3), columns = susceptibility (left)
and infectivity (right). Each panel plots every bin in its group as a ratio to the `ref_lab`
baseline (the same denominator as `plot_ratio`/`plot_ratio_bins` — the model's own `cfg.ref_bin`)
over the rolling origins — posterior median line + 5–95% ribbon. Splitting the bins keeps ≤3
overlapping ribbons per panel readable, which is exactly why `plot_ratio_bins` drew medians only.
Consumes the `susc_bin`/`inf_bin` stores from `collect_transmission_structure` (they already carry
`lo`/`hi`). A bin whose series is all-NaN (missing chain) is skipped.

`groups` is the PANEL ROW SPLIT only — unrelated to `supergroup_split`'s pop-weighted super-groups.
The reference bin is drawn like any other: a flat 1.0 line with no visible ribbon, which is the
visual marker of the gauge.
"""
function plot_susc_inf_bins_ci(susc_bin, inf_bin, lbl, origins, grid;
                               groups = [[1, 2], [3, 4], [5, 6, 7]],
                               ref_lab::AbstractString)
    cols = palette(:viridis, grid.N)             # age is ordinal → perceptually ordered palette
    quantities = [("susceptibility", susc_bin), ("infectivity", inf_bin)]
    # Figure-wide y-limits from the MEDIANS only (both quantities), on a LOG10 axis (ratios are
    # multiplicative) so the panels stay comparable and a wide 90% ribbon runs off-panel rather
    # than compressing every median line.
    yl = _median_ylims(reduce(vcat, [vec(susc_bin[lbl].med), vec(inf_bin[lbl].med)]); log = true)
    panels = Plots.Plot[]
    for grp in groups                            # rows = age-bin groups; row-major fill ⇒ (sus, inf) per row
        for (qname, store) in quantities
            s = store[lbl]
            p = plot(; title = "$qname — bins $(grid.LAB[first(grp)])…$(grid.LAB[last(grp)])",
                     titlefontsize = 8, xlabel = "forecast origin",
                     ylabel = "ratio to $(ref_lab) (log)",
                     legend = :topright, legendfontsize = 6, xrotation = 45,
                     yscale = :log10, ylims = yl)
            # Reference at 1 as a Date-valued series FIRST → establishes the date x-axis (see plot_ratio).
            plot!(p, [first(origins), last(origins)], [1.0, 1.0]; color = :gray, ls = :dash, label = "")
            for a in grp
                m, lo, hi = s.med[:, a], s.lo[:, a], s.hi[:, a]
                all(isnan, m) && continue
                plot!(p, origins, m; color = cols[a], lw = 1.6, marker = :circle, ms = 1.5,
                      markerstrokewidth = 0, label = grid.LAB[a],
                      ribbon = (m .- lo, hi .- m), fillalpha = 0.15, fillcolor = cols[a])
            end
            push!(panels, p)
        end
    end
    return plot(panels...; layout = (length(groups), length(quantities)),
                size = (575 * length(quantities), 360 * length(groups)),
                plot_title = "8j — age-dependent susceptibility & infectivity " *
                             "(ratio to $(ref_lab), 90% CI) — $lbl",
                plot_titlefontsize = 11)
end

"""
    plot_lengthscales(rho, labels4, origins; h) -> Plot

**TWO panels per config**, because the quantities have different units: a SPATIAL panel with ρ_diag
(total-age, solid) and ρ_gap (age-gap, dashed) in **age-years**, and a TEMPORAL panel with ρ_time in
**weeks**, each with a 90% ribbon. ρ_time is absent (NaN, not plotted) for pooled chains. Configs
with NO Stage-1 contact fit at all (the NULL model) have no length-scales and are skipped entirely
rather than drawn as blank panels.

`cfg` (optional) draws the identifiability limit on the temporal panel: a ρ_time at or beyond the
window length `n_fit + h` means the field is effectively constant across the window, so the
parameter has stopped being identified. See `gp_time_len_prior`.
"""
function plot_lengthscales(rho, labels4, origins; h::Integer = 1, cfg = nothing)
    # ⚠ TWO DIFFERENT QUANTITIES, TWO AXES. Columns 1–2 are the SPATIAL GP length-scales in
    # age-years (0–50); column 3 is ρ_time in WEEKS. Drawing all three on one axis (the state until
    # 2026-08-06) mixes units on a single "age-yrs / weeks" label and hides the temporal series in
    # the bottom of a 0–50 age-year range. The split was introduced by `-ar1`, when column 3 was a
    # dimensionless φ ∈ (0,1); `-m32t` (2026-08-10) put ρ_time back in weeks but the split is KEPT,
    # because the units never did match.
    spatial_dirs = ["ρ_diag (total age, yr)", "ρ_gap (age gap, yr)"]
    spatial_ls   = [:solid, :dash]
    shown = [l for l in labels4 if any(isfinite, rho[l].med)]   # drop contact-fit-free configs
    panels = Plots.Plot[]
    for lbl in shown
        ps = plot(; title = "$(lbl) — spatial", titlefontsize = 8, xlabel = "forecast origin",
                  ylabel = "GP length-scale (age-yrs)", legend = (lbl == shown[1] ? :topright : false),
                  legendfontsize = 6, xrotation = 45, ylims = (0, 50))
        for g in eachindex(spatial_dirs)
            m, lo, hi = rho[lbl].med[:, g], rho[lbl].lo[:, g], rho[lbl].hi[:, g]
            all(isnan, m) && continue
            plot!(ps, origins, m; lw = 1.8, marker = :circle, ms = 2, ls = spatial_ls[g],
                  label = spatial_dirs[g], ribbon = (m .- lo, hi .- m), fillalpha = 0.15)
        end
        push!(panels, ps)

        m, lo, hi = rho[lbl].med[:, 3], rho[lbl].lo[:, 3], rho[lbl].hi[:, 3]
        # The window length is where ρ_time stops being identified, and it is also the natural top
        # of the axis. `cfg` is optional, so fall back to the project default n_fit = 8.
        Tn_win = (cfg === nothing ? 8 : cfg.n_fit) + h
        # `filter` first, then test emptiness — `all(isnan, hi)` would not catch a hi that mixes NaN
        # with a non-finite value, and `maximum` of an empty collection throws.
        fin_hi = filter(isfinite, hi)
        yhi = max(Tn_win * 1.15, isempty(fin_hi) ? 0.0 : maximum(fin_hi) * 1.05)
        pt = plot(; title = "$(lbl) — temporal", titlefontsize = 8, xlabel = "forecast origin",
                  ylabel = "ρ_time (weeks)", legend = (lbl == shown[1] ? :topright : false),
                  legendfontsize = 6, xrotation = 45, ylims = (0, yhi))
        if !all(isnan, m)                                # NaN for pooled / contact-free chains
            plot!(pt, origins, m; lw = 1.8, marker = :circle, ms = 2, ls = :dot, color = :black,
                  label = "ρ_time", ribbon = (m .- lo, hi .- m), fillalpha = 0.15)
            # ρ_time ≥ the window length is the pooled/unidentified limit (the field is effectively
            # constant across the window); mark it so a posterior sitting there is visible rather
            # than merely "high". This is the check `gp_time_len_prior` asks for at every refit.
            hline!(pt, [Tn_win]; ls = :dash, color = :red, lw = 1,
                   label = (lbl == shown[1] ? "window = $(Tn_win) wk" : false))
        end
        push!(panels, pt)
    end
    isempty(panels) && return plot(; title = "no chains with finite length-scales")
    return plot(panels...; layout = (length(shown), 2), size = (1150, 390 * length(shown)),
                plot_title = "8j — spatial (age-yrs) and temporal (weeks) GP length-scales (h=$h)",
                plot_titlefontsize = 11)
end

"""
    plot_gamma(store, labels4, model_cols, origins; h) -> Plot

Per-contact secondary attack rate γ_SAR over the forecast origins, one line per model config
(median + 90% ribbon) in a single panel — γ_SAR is a scalar-per-draw (one value per model×origin),
so unlike `plot_ratio` there are no per-age super-groups to facet. `store` is the `gamma` field of
`collect_transmission_structure` (`Dict(label => (med, lo, hi))` of `nO × 1` matrices).

Under the two-stage cut, C* is NOT normalised, so γ_SAR is the per-contact secondary attack rate
(it reproduces the reference cell N_{ref,ref} = susc_r·inf_r directly) and IS comparable across
origins. γ_SAR has no natural reference level (unlike the ratio=1 / R=1 lines), so none is drawn.

⚠ **The y-axis is `:log10` and no longer capped (fixed 2026-08-09).** γ_SAR is constructed as
`exp(_softclamp(log_gamma_sar, log 0.001, log 10))` — a multiplicative scale spanning four orders of
magnitude — and the old linear `ylims = (0, 2.0)` destroyed both ends of it. Measured on the
completed grid: `negbin|neighbourhood` sits ON the lower clamp (median 0.001 at some origins, 1.9%
of its draws at the bound), which on a linear 0–2 axis is visually indistinguishable from zero and
from any other small value; meanwhile medians and bands past 2.0 were silently clipped away. The
`_softclamp` bounds are now drawn as dashed red lines, so clamp compression — the failure mode this
parameter has hit twice (the log 0.02 bound in 2026-07-13, and now log 0.001) — is visible on the
figure instead of having to be inferred from the draws.

`cfg` (optional) adds the grey prior 90% band from `_prior_gamma_band`: a posterior sitting inside
it means γ_SAR is carrying prior, which given the documented γ_SAR ↔ generation-interval
confounding (§3.1) is the thing worth knowing.
"""
function plot_gamma(store, labels4, model_cols, origins; h::Integer = 1, cfg = nothing)
    # Plot the real Date-bearing series directly (no leading synthetic/`hline!` line) so the
    # x-axis stays a date axis — see the gotcha in `plot_ratio` / `plot_reproduction`.
    # ⚠ A log axis cannot show ≤ 0, and `lo` can legitimately reach the 0.001 clamp — so limits come
    # from the finite POSITIVE values only, via the same `_median_ylims(...; log = true)` the ratio
    # figures use, widened to include the clamp bounds so they are always on-panel.
    vals = reduce(vcat, [vec(store[l].med) for l in labels4])
    cl   = (0.001, 10.0)                                   # `model_transmission` soft-clamp bounds
    yl   = _median_ylims(vcat(vals, collect(cl)); ref = 1.0, log = true)
    fig = plot(; xlabel = "forecast origin", ylabel = "γ_SAR (per-contact SAR, log)",
               title = "8j — secondary attack rate γ_SAR over time by model (h=$h; 90% CI)",
               size = (950, 520), legend = :topright, xrotation = 45,
               yscale = :log10, ylims = yl)
    for (ci, lbl) in enumerate(labels4)
        m, lo, hi = store[lbl].med[:, 1], store[lbl].lo[:, 1], store[lbl].hi[:, 1]
        all(isnan, m) && continue
        plot!(fig, origins, m; color = model_cols[ci], lw = 1.8, marker = :circle, ms = 2,
              ribbon = (m .- lo, hi .- m), fillalpha = 0.12, label = lbl)
    end
    if cfg !== nothing
        pri = _prior_gamma_band(cfg)
        xs  = [first(origins), last(origins)]
        plot!(fig, xs, fill(pri[2], 2); color = :grey40, ls = :dot, lw = 1.0,
              label = "prior 90%", ribbon = (fill(pri[2] - pri[1], 2), fill(pri[3] - pri[2], 2)),
              fillalpha = 0.10, fillcolor = :grey60)
    end
    # Soft-clamp bounds LAST (after the date-valued series, so the x-axis is already a date axis).
    hline!(fig, collect(cl); ls = :dash, color = :red, lw = 1,
           label = "softclamp [0.001, 10]")
    return fig
end

"""
    plot_sigma_inf(store, labels4, model_cols, origins; h=1) -> Plot

Infection-likelihood **observation SD σ_inf** over the rolling forecast origins, one line per model
(median + 90% band). `store` is the `sigma_inf` field of `collect_transmission_structure`.

`sigma_inf ~ N⁺(0.05, 0.025²)` in `model_transmission` is the noise scale of the weekly infection
likelihood — the only fitted Stage-2 scalar that had **no figure at all**: it has been written into
every `8j_s2_*` since the cut landed, but `load_transmission_draws` dropped it before 2026-08-09, so
nothing downstream could see it. It is worth a panel because it is the model's own estimate of how
well it fits the infection series: σ_inf far above its prior centre means the renewal step is not
tracking the data and the residual is being absorbed as noise, which is exactly the situation in
which good WIS would be coming from a wide predictive rather than an accurate one.
"""
function plot_sigma_inf(store, labels4, model_cols, origins; h::Integer = 1)
    fig = plot(; xlabel = "forecast origin", ylabel = "σ_inf (infection observation SD)",
               title = "9j — infection-likelihood observation SD σ_inf (h=$h; 90% CI)",
               size = (950, 520), legend = :topright, xrotation = 45)
    for (ci, lbl) in enumerate(labels4)
        m, lo, hi = store[lbl].med[:, 1], store[lbl].lo[:, 1], store[lbl].hi[:, 1]
        all(isnan, m) && continue
        plot!(fig, origins, m; color = model_cols[ci], lw = 1.8, marker = :circle, ms = 2,
              ribbon = (m .- lo, hi .- m), fillalpha = 0.12, label = lbl)
    end
    # Prior centre + 90%, so "has it moved off the prior?" is readable straight off the panel —
    # the same identifiability idiom as `plot_gen_interval`. Truncated at 0, hence `max(·, 0)`.
    pm, ps = 0.05, 0.025
    zq = 1.6448536269514722
    xs = [first(origins), last(origins)]
    plot!(fig, xs, fill(pm, 2); color = :grey40, ls = :dot, lw = 1.0, label = "prior 90%",
          ribbon = (fill(pm - max(pm - zq * ps, 0.0), 2), fill(zq * ps, 2)),
          fillalpha = 0.10, fillcolor = :grey60)
    return fig
end

"""
    plot_F(store, labels4, model_cols, origins; h=1) -> Plot

Antibody-protection factor **F** over the rolling forecast origins, one line per model (median + 90%
band). `store` is the `F` field of `collect_transmission_structure` (an `nO × 1` med/lo/hi store of
the pooled per-draw `F`). F is the LEAKY antibody-protection factor in `full_susceptibility_a(t) =
susc_a·(1 + (F−1)·A_a(t))`: F = 0 ⇒ antibodies fully protect, F = 1 ⇒ no protection. This is the
direct analog of `plot_gamma` for the γ_SAR level.

⚠ **CHANGED 2026-08-04 (user request): F is no longer estimated.** `model_transmission` PINS
`F = 1.0` to disable the antibody term (the `Beta(5,1)` prior is commented out there), so every
model's line is a flat 1.0 with a zero-width band and all six coincide — that degenerate line IS
the pin, not a fitting result or a convergence failure. The panel is kept as the visible evidence
that the term is off. `ylims` is `(0, 1.05)` rather than `(0, 1)` precisely so the pinned line does
not sit on the top border and vanish (it also fixes the pre-existing silent-clipping hazard: an F
outside (0,1) used to be clipped away without warning).
"""
function plot_F(store, labels4, model_cols, origins; h::Integer = 1, pinned = nothing)
    # Plot the real Date-bearing series directly (no leading synthetic/`hline!` line) so the
    # x-axis stays a date axis — see the gotcha in `plot_ratio` / `plot_reproduction`.
    # The "PINNED" title used to be a hard-coded string, which would have quietly LIED the moment
    # F was restored to `~ Beta(5,1)`. It is now driven by `pinned` (`tr.F_pinned`), detected from
    # the draws, so the panel can only claim a pin that is actually in the artefacts.
    all_pinned = pinned !== nothing && !isempty(labels4) && all(get(pinned, l, false) for l in labels4)
    ttl = all_pinned ? "9j — antibody-protection factor F (h=$h) — PINNED at 1.0 (term off)" :
          pinned === nothing ? "9j — antibody-protection factor F (h=$h)" :
                               "9j — antibody-protection factor F (h=$h) — FITTED"
    fig = plot(; xlabel = "forecast origin", ylabel = "F (leaky antibody-protection factor)",
               title = ttl,
               size = (950, 520), legend = :topright, xrotation = 45, ylims = (0, 1.05))
    for (ci, lbl) in enumerate(labels4)
        m, lo, hi = store[lbl].med[:, 1], store[lbl].lo[:, 1], store[lbl].hi[:, 1]
        all(isnan, m) && continue
        plot!(fig, origins, m; color = model_cols[ci], lw = 1.8, marker = :circle, ms = 2,
              ribbon = (m .- lo, hi .- m), fillalpha = 0.12, label = lbl)
    end
    return fig
end
