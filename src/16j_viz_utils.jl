# 16j_viz_utils.jl — helpers for `16j_antibody_window.ipynb`.
#
# 14j §6c draws antibody prevalence over the whole CoMix span. 16j zooms to ONE forecast origin —
# 10j's, 2021-05-09 — and to exactly the weeks the framework touches there: the `smax` renewal-lag
# weeks, the `n_fit` fit weeks and the forecast-target weeks of `WeeklyWindow`. In those weeks it
# overlays three things:
#
#   • the weekly antibody covariate `A_a(t)` the FRAMEWORK builds from `estimates_age_ab.csv`
#     (`load_window_data` → `weekly_antibody`: `wd.antibody` over lag+fit weeks, `wd.antibody_fc`
#     at the forecast targets) — what entered the NGM's leaky factor `(1 + (F−1)·A_a(t))` while
#     `_EST_PATH` named the `_ab` file (until 2026-09-16);
#   • the daily `gen_dab` it is averaged from (median, 90% band);
#   • the ONS antibody positivity the `_ab` fit was trained on, rebinned to the CIS bins exactly as
#     `read_ab()` does (14j's `ons_antibody_to_cis`).
#
#   §A  `antibody_window_frame` — one tidy (bin × week) frame: segment, framework A, matched ONS
#   §B  `antibody_window_summary` — per bin × segment agreement in percentage points
#   §C  `incidence_window_frame` / `calibration_window_frame` / `forecast_window_frame` — observed
#       weekly infections over the whole window, each model's in-sample fitted incidence over the
#       fit weeks (10j's `fit_window_infection_draws`), and its forecast fan at the targets, all on
#       a right-hand axis
#   §D  `make_antibody_window_fig` — one panel per CIS bin, segments shaded
#   §E  `ons_sgtf_share` / `make_variant_incidence_fig` — the same incidence against the England
#       S-gene-target-failure share (Alpha proxy) and its S-positive complement (≈ Delta from
#       late April 2021). England-level ONLY: ONS publishes no variant split by age.
#   §F  `ons_variant_mix` / `make_variant_mix_fig` — every variant category's share over the whole
#       CoMix span (2020-10 → 2022-01), stacked; §E and §F share ONE release reader,
#       `_ons_variant_releases`.
#
# Requires (in this order): `forecast_utils.jl`, `9j_viz_utils.jl` (`panel_grid`), then
# `14j_viz_utils.jl` (`ons_antibody_age`, `ons_antibody_fit_vintage`, `ons_antibody_to_cis`,
# `i2p_gen_dab`, `_i2p_est_path`, `_model_cols`). The forecast fans come from 9j's assembly cache via
# 15j's `load_9j_assembly` (`isfile` → `load` → validate), so nothing here can fit.
#
# ⚠ THE INCIDENCE AND THE ANTIBODY COME FROM DIFFERENT inc2prev FILES, deliberately. The forecasts
# are the LIVE grid's, fitted to `estimates_age.csv` (`_EST_PATH`), so their observed comparator is
# that file's weekly infections — the truth 9j scored them against (the notebook asserts the cache's
# truth equals `load_forecast_truth`). The antibody is `estimates_age_ab.csv`'s, the only file that
# carries any. With `F = 1` the antibody never entered those forecasts; the overlay shows timing, not
# a causal link.
#
# ⚠ THE FORECAST-WEEK `A` IS THE REALISED FUTURE VALUE. `wd.antibody_fc[:, h]` is `gen_dab` at
# t₀+h, read from the same posterior that saw data past the origin — the same "contact-updated
# iterate" convention CLAUDE.md records for `Cstar_end`. It is not a forecast of antibody.
#
# ⚠ INERT TODAY. `_EST_PATH` names `estimates_age.csv` (no `gen_dab` ⇒ `A ≡ 0`) and
# `model_transmission` pins `F = 1.0`, so `(1 + (F−1)·A) ≡ 1` whatever `A` holds. 16j reads the
# `_ab` file EXPLICITLY; it shows what the previous Stage-2 generation (`dt_intermediate_residual/`)
# was handed, and what re-enabling `F` would feed — not anything the live grid uses.

"Which `WeeklyWindow` segment a week belongs to, in plotting order."
const _AB_SEGMENTS = ("lag", "fit", "forecast")

"""
    antibody_window_frame(win, wd, obs; grid) -> DataFrame

One row per (CIS bin, window week) over `win.all_weeks ++ win.forecast_weeks`:
`segment ∈ _AB_SEGMENTS`, `week` (Monday start), `week_mid`, `A_framework` (proportion, from `wd`)
and the matched ONS observation (`ons_value`/`ons_lo`/`ons_hi`, `ons_start_date`; `NaN`/`missing`
where none) from `obs` (`ons_antibody_to_cis`).

ONS weeks run Monday–Sunday, and so do the framework's since 2026-10-22 (`WEEK_ANCHOR`), so an ONS
week is matched to the framework week containing its Thursday mid-date — the SAME week, all 7 days.
(On the old Sunday grid this was the week starting the day before, 6 of 7 days shared.) Asserted
one-to-one, so a calendar change cannot silently pair two ONS weeks with one framework week.
"""
function antibody_window_frame(win::WeeklyWindow, wd, obs::DataFrame; grid = cis_age_grid())
    @assert wd.weeks == win.all_weeks "antibody_window_frame: `wd` was not built from `win`"
    @assert size(wd.antibody_fc, 2) == length(win.forecast_weeks)
    onsix = Dict{Tuple{Int,Date},Int}()
    for (k, r) in enumerate(eachrow(obs))
        key = (r.bin, week_start(r.date_mid))
        @assert !haskey(onsix, key) "antibody_window_frame: two ONS weeks map to framework week \
            $(key[2]) in bin $(key[1]) — the ONS ↔ framework week matching is no longer one-to-one"
        onsix[key] = k
    end
    seg = vcat(fill("lag", length(win.lag_weeks)), fill("fit", length(win.fit_weeks)),
               fill("forecast", length(win.forecast_weeks)))
    weeks = vcat(win.all_weeks, win.forecast_weeks)
    Amat  = hcat(wd.antibody, wd.antibody_fc)
    rows = NamedTuple[]
    for a in 1:grid.N, (t, w) in enumerate(weeks)
        k = get(onsix, (a, w), 0)
        o = k == 0 ? nothing : obs[k, :]
        push!(rows, (bin = a, bin_label = grid.LAB[a], segment = seg[t], week = w,
                     week_mid = week_mid(w), A_framework = Amat[a, t],
                     ons_start_date = o === nothing ? missing : o.start_date,
                     ons_value = o === nothing ? NaN : o.value,
                     ons_lo    = o === nothing ? NaN : o.value_lo,
                     ons_hi    = o === nothing ? NaN : o.value_hi))
    end
    return DataFrame(rows)
end

"""
    antibody_window_summary(fr) -> DataFrame

Per (bin, segment): weeks with an ONS match, median signed and absolute `A_framework − ONS` in
percentage points, and how many weeks the framework value falls OUTSIDE ONS's published 95% CI.
Bins without ONS data (`<16`) report `n_matched = 0` and `NaN`s rather than being dropped.
"""
function antibody_window_summary(fr::DataFrame)
    g = groupby(transform(fr, [:A_framework, :ons_value] => ((a, o) -> 100 .* (a .- o)) => :d_pp,
                          [:A_framework, :ons_lo, :ons_hi] =>
                              ByRow((a, l, h) -> isfinite(l) && !(l <= a <= h)) => :outside),
                [:bin, :bin_label, :segment])
    out = combine(g, :d_pp => (d -> count(isfinite, d)) => :n_matched,
                  :d_pp => (d -> (f = filter(isfinite, d); isempty(f) ? NaN : median(f))) => :median_diff_pp,
                  :d_pp => (d -> (f = filter(isfinite, d); isempty(f) ? NaN : median(abs.(f)))) => :median_absdiff_pp,
                  :outside => sum => :n_outside_95)
    out.segment_order = [findfirst(==(s), _AB_SEGMENTS) for s in out.segment]
    return select(sort(out, [:bin, :segment_order]), Not(:segment_order))
end

"""
    incidence_window_frame(win, wd, truth; grid) -> DataFrame (bin, bin_label, segment, week,
                                                               week_mid, observed)

The observed OUTCOME over the whole window — ONS CIS positives × pop (`weekly_cis_positives`):
`wd.I_mean` over `win.all_weeks` (lag + fit) and `truth` (`load_forecast_truth`, `A × H`) at the
forecast targets. Both come from the same `_cis_frame()`, so they are the series the forecasts were
fitted to and scored against.
"""
function incidence_window_frame(win::WeeklyWindow, wd, truth::AbstractMatrix; grid = cis_age_grid())
    @assert wd.weeks == win.all_weeks "incidence_window_frame: `wd` was not built from `win`"
    @assert size(truth) == (grid.N, length(win.forecast_weeks))
    seg = vcat(fill("lag", length(win.lag_weeks)), fill("fit", length(win.fit_weeks)),
               fill("forecast", length(win.forecast_weeks)))
    weeks = vcat(win.all_weeks, win.forecast_weeks)
    M = hcat(wd.I_mean, truth)
    return DataFrame([(bin = a, bin_label = grid.LAB[a], segment = seg[t], week = w,
                       week_mid = week_mid(w), observed = M[a, t])
                      for a in 1:grid.N for (t, w) in enumerate(weeks)])
end

"""
    forecast_window_frame(fc_store, origin, labels, win; grid, q=(0.05, 0.5, 0.95)) -> DataFrame

Each model's pooled forecast fan at `origin` (`fc_store[(origin, label)]`, `A × H × draws`, from the
9j assembly cache) summarised per (label, bin, horizon) to `lo`/`med`/`hi` at quantiles `q`, placed
at the forecast weeks' `week_mid`. Quantiles over the 10 000 draws of ONE (bin, week) cell, so the
band is a genuine pointwise 90% predictive interval.
"""
function forecast_window_frame(fc_store, origin::Date, labels, win::WeeklyWindow;
                               grid = cis_age_grid(), q = (0.05, 0.5, 0.95))
    rows = NamedTuple[]
    for l in labels
        F = get(fc_store, (origin, l), nothing)
        F === nothing && error("forecast_window_frame: no fan for ($origin, $l) in the cache")
        @assert size(F, 1) == grid.N && size(F, 2) == length(win.forecast_weeks)
        for a in 1:grid.N, (h, w) in enumerate(win.forecast_weeks)
            v = @view F[a, h, :]
            push!(rows, (label = String(l), bin = a, bin_label = grid.LAB[a], h = h, week = w,
                         week_mid = week_mid(w), lo = quantile(v, q[1]), med = quantile(v, q[2]),
                         hi = quantile(v, q[3])))
        end
    end
    return DataFrame(rows)
end

"""
    calibration_window_frame(store, labels, win; grid, q=(0.05, 0.5, 0.95)) -> DataFrame

Each model's IN-SAMPLE fitted incidence over the fit weeks — `store[label]` is 10j's
`fit_window_infection_draws` output (`A × n_fit × draws`) — summarised to `lo`/`med`/`hi` per
(label, bin, fit week). Same schema as `forecast_window_frame` minus `h`.

⚠ **Not the same quantity as the forecast band.** This is the MEAN of Stage 2's infection
likelihood given the OBSERVED lags (a one-step-ahead fitted mean, no observation noise), so its band
is posterior uncertainty on that mean; the forecast band is posterior-PREDICTIVE (mean + noise, and
self-iterated). 10j §1 makes the same distinction. A narrow calibration band beside a wide forecast
band is expected, not a contradiction.
"""
function calibration_window_frame(store::AbstractDict, labels, win::WeeklyWindow;
                                  grid = cis_age_grid(), q = (0.05, 0.5, 0.95))
    rows = NamedTuple[]
    for l in labels
        X = get(store, l, nothing)
        X === nothing && continue                       # a missing artefact: drawn as absent, not as zero
        @assert size(X, 1) == grid.N && size(X, 2) == length(win.fit_weeks) "calibration_window_frame: \
            `$l` has size $(size(X)), expected ($(grid.N), $(length(win.fit_weeks)), draws)"
        for a in 1:grid.N, (c, w) in enumerate(win.fit_weeks)
            v = @view X[a, c, :]
            push!(rows, (label = String(l), bin = a, bin_label = grid.LAB[a], week = w,
                         week_mid = week_mid(w), lo = quantile(v, q[1]), med = quantile(v, q[2]),
                         hi = quantile(v, q[3])))
        end
    end
    return DataFrame(rows)
end

"Tick formatter for weekly infection counts: 12000 → \"12k\"."
_kfmt(y) = y >= 1000 ? "$(round(Int, y / 1000))k" : "$(round(Int, y))"

"""
    _incidence_overlay!(ax, b, inc_obs, inc_fc, inc_cal, fc_labels, fc_cols) -> ymax

Draw bin `b`'s incidence on axis `ax`, shared by BOTH 16j figures so they cannot drift apart:
observed (black ○, all weeks), each label's CALIBRATION (dashed ◇, faint band, fit weeks) and
FORECAST (solid ○, band, targets) — 10j §1's styles. `inc_fc`/`inc_cal` may be `nothing`. Returns
the largest value drawn (observed or a 95% bound) so the caller can set the axis limit; the caller
also owns the axis labels, so the same series can sit on a left or a right axis.
"""
function _incidence_overlay!(ax, b::Integer, inc_obs::DataFrame, inc_fc, inc_cal, fc_labels, fc_cols)
    o = sort(inc_obs[inc_obs.bin .== b, :], :week)
    _sel(fr_) = fr_ === nothing ? DataFrame(label = String[], hi = Float64[]) :
                fr_[(fr_.bin .== b) .& in.(fr_.label, Ref(fc_labels)), :]
    q, qc = _sel(inc_fc), _sel(inc_cal)
    for (k, l) in enumerate(fc_labels)
        cl = nrow(qc) == 0 ? qc : sort(qc[qc.label .== l, :], :week)
        nrow(cl) > 0 && plot!(ax, cl.week_mid, cl.med; ribbon = (cl.med .- cl.lo, cl.hi .- cl.med),
                              fillalpha = 0.07, color = fc_cols[k], lw = 1.3, ls = :dash,
                              markershape = :diamond, markersize = 2.4,
                              markercolor = :white, markerstrokecolor = fc_cols[k], label = "")
        ql = nrow(q) == 0 ? q : sort(q[q.label .== l, :], :h)
        nrow(ql) > 0 && plot!(ax, ql.week_mid, ql.med; ribbon = (ql.med .- ql.lo, ql.hi .- ql.med),
                              fillalpha = 0.12, color = fc_cols[k], lw = 1.5, markershape = :circle,
                              markersize = 2.4, markerstrokecolor = fc_cols[k], label = "")
    end
    plot!(ax, o.week_mid, o.observed; color = :black, lw = 1.8, markershape = :circle,
          markersize = 2.4, label = "")
    return maximum(vcat(o.observed, q.hi, qc.hi))
end

"Legend entries for `_incidence_overlay!`'s series, on a NaN-data legend panel."
function _incidence_legend!(lg, fc_labels, fc_cols, with_cal::Bool; axis::AbstractString = "")
    pre = isempty(axis) ? "" : "$(axis) — "
    plot!(lg, [NaN], [NaN]; color = :black, lw = 1.8, markershape = :circle, markersize = 3,
          label = "$(pre)observed infections / week\n(estimates_age.csv, the forecasts' truth)")
    for (k, l) in enumerate(fc_labels)
        plot!(lg, [NaN], [NaN]; ribbon = [NaN], fillalpha = 0.12, color = fc_cols[k], lw = 1.5,
              markershape = :circle, markersize = 3, markerstrokecolor = fc_cols[k], label = "$(l)")
    end
    plot!(lg, [NaN], [NaN]; color = :gray30, lw = 1.5, markershape = :circle, markersize = 3,
          label = "solid ○ = forecast (median, 90% predictive)")
    with_cal && plot!(lg, [NaN], [NaN]; color = :gray30, lw = 1.3, ls = :dash,
          markershape = :diamond, markersize = 3, markercolor = :white, markerstrokecolor = :gray30,
          label = "dashed ◇ = calibration: in-sample fitted\nmean (median, 90% CrI, h=1 chain)")
    return lg
end

"""
    make_antibody_window_fig(fr, gd, win; grid, res_dir="../res",
                             fname="16j_antibody_window.png") -> Plots.Plot

One panel per CIS bin (layout from `panel_grid`), x = the window's weeks only. Background bands mark
the three `WeeklyWindow` segments (renewal lag · fit · forecast); a dashed line marks where the
forecast starts (the end of origin week t₀). Per panel:

- **`A_a(t)` the framework builds** — one square per week at its Thursday `week_mid`, joined;
- **`gen_dab`** — daily median and pointwise 90% band (daily quantiles, nothing averaged), plus the
  daily MEAN dashed. ⚠ `weekly_antibody` averages the posterior **mean** column, and the child-bin
  posteriors are right-skewed, so there the squares sit ~3 pp ABOVE the median — on the dashed line,
  which is what they average. Drawing only the median would make that look like a bug;
- **ONS** — black points with the published 95% CI at their Thursday mid-dates (the fit's vintage).

With `inc_obs` (`incidence_window_frame`) and `inc_fc` (`forecast_window_frame`) each panel gains a
RIGHT-hand axis (`twinx`) of weekly infections: the observed series over all 16 weeks (black,
circles) and each of `fc_labels`' forecast median + 90% band over the 4 targets (`fc_cols`, solid,
○). With `inc_cal` (`calibration_window_frame`) each model's IN-SAMPLE fitted median + 90% band is
added over the fit weeks in the same colour, DASHED with ◇ and a fainter band — 10j §1's convention
for keeping calibration and forecast apart. The
antibody series are then drawn in GREY so the model colours stay unambiguous. Right-axis range is per
panel, from 0 to the largest observed value or 95% bound shown.

The legend gets its own panel in the grid's spare slot when there is one (7 bins ⇒ a 2×4 grid);
otherwise it sits in the first panel. The y-range is per panel and fitted to the data (bins sit anywhere from ~20% to ~95% in this
window, so one shared 0–100 axis would flatten every weekly movement); the value range is in each
panel's tick labels. Child bins have no ONS data and are titled so.
"""
function make_antibody_window_fig(fr::DataFrame, gd::DataFrame, win::WeeklyWindow;
                                  inc_obs = nothing, inc_fc = nothing, inc_cal = nothing,
                                  fc_labels = String[],
                                  fc_cols = [:steelblue, :darkorange, :seagreen, :purple],
                                  grid = cis_age_grid(), res_dir::AbstractString = "../res",
                                  fname::AbstractString = "16j_antibody_window.png")
    seg_w = Dict("lag" => win.lag_weeks, "fit" => win.fit_weeks, "forecast" => win.forecast_weeks)
    seg_c = Dict("lag" => :gray80, "fit" => :lightsteelblue, "forecast" => :navajowhite)
    seg_l = Dict("lag" => "renewal lag ($(length(win.lag_weeks)) wk)",
                 "fit" => "fit window ($(length(win.fit_weeks)) wk)",
                 "forecast" => "forecast targets ($(length(win.forecast_weeks)) wk)")
    lo = first(win.all_weeks); hi = last(win.forecast_weeks) + Day(7)
    split = win.origin + Day(7)                      # end of origin week t₀ = start of forecast
    ticks = vcat(win.all_weeks, win.forecast_weeks, [hi])
    with_inc = inc_obs !== nothing
    with_inc && @assert length(fc_cols) >= length(fc_labels) "make_antibody_window_fig: one colour per forecast label"
    # Antibody in grey when incidence is overlaid, so every colour on the panel names a MODEL.
    col_m, col_A = with_inc ? :gray50 : _model_cols(2)[2], :black
    xl = (Dates.value(lo), Dates.value(hi))
    np = grid.N; nr, nc = panel_grid(np)             # DERIVED — never a literal
    panels = Any[]
    for b in 1:np
        f  = fr[fr.bin .== b, :]
        g  = gd[(gd.bin .== b) .& (lo .<= gd.date .< hi), :]
        haso = any(isfinite, f.ons_value)
        vals = 100 .* vcat(f.A_framework, g.q5, g.q95, filter(isfinite, f.ons_lo), filter(isfinite, f.ons_hi))
        ylo, yhi = extrema(vals); pad = max(2.0, 0.08 * (yhi - ylo))
        yl = (max(0.0, ylo - pad), min(100.0, yhi + 2.2pad))   # extra top room for the legend
        pnl = plot(; title = haso ? "age $(grid.LAB[b])" : "age $(grid.LAB[b]) — NO ONS data (<16)",
                     titlefontsize = 8, xrotation = 60, xtickfontsize = 5,
                     xlabel = (b > np - nc) ? "week (Mon start)" : "",
                     ylabel = (mod1(b, nc) == 1) ? "antibody positive (%)" : "",
                     legend = false)
        # Date series FIRST (a leading numeric overlay collapses the Date axis), bands after.
        y = 100 .* g.q50
        plot!(pnl, g.date, y; ribbon = (y .- 100 .* g.q5, 100 .* g.q95 .- y), fillalpha = 0.25,
              color = col_m, lw = 1.2, label = "gen_dab daily (median, 90% CrI)")
        plot!(pnl, g.date, 100 .* g.mean; color = col_m, lw = 0.9, ls = :dash, label = "gen_dab daily mean")
        plot!(pnl, f.week_mid, 100 .* f.A_framework; color = col_A, lw = 1.0, markershape = :square,
              markersize = 2.6, markercolor = :white, markerstrokecolor = col_A,
              label = "A_a(t) the framework builds (weekly mean of gen_dab)")
        if haso
            m  = isfinite.(f.ons_value)
            ox = Date.(f.ons_start_date[m]) .+ Day(3)
            yo = 100 .* f.ons_value[m]
            scatter!(pnl, ox, yo; yerror = (yo .- 100 .* f.ons_lo[m], 100 .* f.ons_hi[m] .- yo),
                     markersize = 2.6, markercolor = :darkred, markerstrokecolor = :darkred,
                     label = "ONS 179 ng/ml, rebinned (95% CI) — what the fit read")
        end
        for s in _AB_SEGMENTS
            ws = seg_w[s]
            vspan!(pnl, [first(ws), last(ws) + Day(7)]; color = seg_c[s], fillalpha = 0.25,
                   linealpha = 0, label = b == 1 ? seg_l[s] : "")
        end
        vline!(pnl, [split]; color = :black, ls = :dash, lw = 1, label = b == 1 ? "forecast starts (end of t₀ week)" : "")
        plot!(pnl; ylims = yl, xlims = xl, xticks = (Dates.value.(ticks), Dates.format.(ticks, "d u")))
        if with_inc
            # RIGHT axis: Date series first again (its own axis collapses the same way), then limits.
            ax = twinx(pnl)
            ymax = _incidence_overlay!(ax, b, inc_obs, inc_fc, inc_cal, fc_labels, fc_cols)
            plot!(ax; ylims = (0, 1.12 * ymax), xlims = xl, legend = false, grid = false,
                  ytickfontsize = 6, yformatter = _kfmt,
                  ylabel = (mod1(b, nc) == nc || b == np) ? "infections / week" : "")
        end
        push!(panels, pnl)
    end
    # LEGEND PANEL: the same styles on NaN data, in the grid's spare slot (every data panel is busy
    # in every corner over a 16-week window). With no spare slot, fall back to panel 1.
    lg = plot(; framestyle = :none, legend = :left, legendfontsize = 7, grid = false,
                xticks = false, yticks = false)
    plot!(lg, [NaN], [NaN]; ribbon = [NaN], fillalpha = 0.25, color = col_m, lw = 1.2,
          label = "gen_dab daily (median, 90% CrI)")
    plot!(lg, [NaN], [NaN]; color = col_m, lw = 0.9, ls = :dash, label = "gen_dab daily mean")
    plot!(lg, [NaN], [NaN]; color = col_A, lw = 1.0, markershape = :square, markersize = 3,
          markercolor = :white, markerstrokecolor = col_A,
          label = "A_a(t) the framework builds\n(weekly mean of gen_dab's MEAN)")
    scatter!(lg, [NaN], [NaN]; markercolor = :darkred, markerstrokecolor = :darkred, markersize = 3,
             label = "ONS 179 ng/ml, rebinned to CIS bins\n(95% CI) — what the fit read")
    for s in _AB_SEGMENTS
        plot!(lg, [NaN], [NaN]; seriestype = :shape, color = seg_c[s], fillalpha = 0.4,
              linealpha = 0, label = seg_l[s])
    end
    plot!(lg, [NaN], [NaN]; color = :black, ls = :dash, lw = 1, label = "forecast starts (end of t₀ week)")
    with_inc && _incidence_legend!(lg, fc_labels, fc_cols, inc_cal !== nothing; axis = "RIGHT axis")
    if nr * nc > np
        push!(panels, lg)
        for _ in 1:(nr * nc - np - 1); push!(panels, plot(; framestyle = :none)); end
    else
        plot!(panels[1]; legend = :topleft, legendfontsize = 5)
    end
    ptitle = "Antibody prevalence around forecast origin $(win.origin) (10j's window), ENGLAND — " *
             "what estimates_age_ab.csv hands the framework, against the ONS data it was fitted to" *
             (with_inc ? "\nRIGHT axis: weekly infections — observed (estimates_age.csv), each model's " *
                         (inc_cal === nothing ? "" : "in-sample calibration (dashed) and ") *
                         "forecast from this origin   ·   ⚠ F = 1, so antibody never entered them" : "") * "\n" *
             "⚠ forecast-week A is the REALISED gen_dab at t₀+h (contact-updated-iterate convention), " *
             "not a forecast   ·   ⚠ inert today: _EST_PATH = estimates_age.csv (A ≡ 0) and F = 1"
    fig = plot(panels...; layout = (nr, nc), size = (430 * nc, 400 * nr),
               left_margin = 8Plots.mm, bottom_margin = 12Plots.mm,
               top_margin = 6Plots.mm,
               # `top_margin` does not reserve room for `plot_title` — this does. Three title lines
               # with incidence overlaid, two without.
               plot_titlevspan = with_inc ? 0.075 : 0.05,
               right_margin = (with_inc ? 12 : 2) * Plots.mm,   # room for the right-axis labels
               plot_title = ptitle, plot_titlefontsize = 8)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, fname))
    return fig
end

##########################################################################
# §E — the variant overlay: S-gene target failure (Alpha proxy) vs S-positive
##########################################################################
#
# The readers (`ons_sgtf_share`, `_ons_variant_releases`, `_GENE_CLASS`, the data paths and
# `ONS_DELTA_PERIOD_START`) live in `ons_variant.jl`, shared with 1j. Moved verbatim 2026-10-21.
include(joinpath(@__DIR__, "ons_variant.jl"))

"""
    make_variant_incidence_fig(inc_obs, inc_fc, inc_cal, var, win; fc_labels, fc_cols, grid,
                               res_dir="../res", fname="16j_variant_incidence.png") -> Plots.Plot

§1's incidence (observed · calibration · forecast, drawn by the SAME `_incidence_overlay!`) on the
LEFT axis of one panel per CIS bin, and on the RIGHT axis the England SGTF share (Alpha proxy,
crimson) and its S-positive complement (≈ Delta from late April 2021, brown dash-dot) from
`ons_sgtf_share` — the SAME curve in every panel, because ONS publishes no variant split by age.
A vertical line marks ONS's own Alpha → Delta period boundary (`ONS_DELTA_PERIOD_START`); the three
`WeeklyWindow` segments are shaded and the forecast start dashed, as in §1.
"""
function make_variant_incidence_fig(inc_obs::DataFrame, inc_fc, inc_cal, var::DataFrame,
                                    win::WeeklyWindow; fc_labels = String[],
                                    fc_cols = [:steelblue, :darkorange, :seagreen, :purple],
                                    grid = cis_age_grid(), res_dir::AbstractString = "../res",
                                    fname::AbstractString = "16j_variant_incidence.png")
    seg_w = Dict("lag" => win.lag_weeks, "fit" => win.fit_weeks, "forecast" => win.forecast_weeks)
    seg_c = Dict("lag" => :gray80, "fit" => :lightsteelblue, "forecast" => :navajowhite)
    seg_l = Dict("lag" => "renewal lag ($(length(win.lag_weeks)) wk)",
                 "fit" => "fit window ($(length(win.fit_weeks)) wk)",
                 "forecast" => "forecast targets ($(length(win.forecast_weeks)) wk)")
    lo = first(win.all_weeks); hi = last(win.forecast_weeks) + Day(7)
    split = win.origin + Day(7)
    ticks = vcat(win.all_weeks, win.forecast_weeks, [hi])
    xl = (Dates.value(lo), Dates.value(hi))
    col_a, col_s = :crimson, :saddlebrown
    v = sort(var[(lo .<= var.date .< hi), :], :date)
    np = grid.N; nr, nc = panel_grid(np)
    panels = Any[]
    for b in 1:np
        pnl = plot(; title = "age $(grid.LAB[b])", titlefontsize = 8, xrotation = 60, xtickfontsize = 5,
                     xlabel = (b > np - nc) ? "week (Mon start)" : "",
                     ylabel = (mod1(b, nc) == 1) ? "infections / week" : "", legend = false)
        # Date series FIRST (a leading numeric overlay collapses the Date axis), then bands/lines.
        ymax = _incidence_overlay!(pnl, b, inc_obs, inc_fc, inc_cal, fc_labels, fc_cols)
        for s in _AB_SEGMENTS
            ws = seg_w[s]
            vspan!(pnl, [first(ws), last(ws) + Day(7)]; color = seg_c[s], fillalpha = 0.25, linealpha = 0, label = "")
        end
        vline!(pnl, [split]; color = :black, ls = :dash, lw = 1, label = "")
        plot!(pnl; ylims = (0, 1.12 * ymax), xlims = xl, yformatter = _kfmt, ytickfontsize = 6,
              xticks = (Dates.value.(ticks), Dates.format.(ticks, "d u")))
        ax = twinx(pnl)
        plot!(ax, v.date, 100 .* v.share_sgtf; color = col_a, lw = 2.0, label = "")
        plot!(ax, v.date, 100 .* v.share_spos; color = col_s, lw = 2.0, ls = :dashdot, label = "")
        vline!(ax, [ONS_DELTA_PERIOD_START]; color = col_a, ls = :dot, lw = 1.4, label = "")
        plot!(ax; ylims = (0, 105), yticks = 0:20:100, xlims = xl, legend = false, grid = false,
              ytickfontsize = 6, ylabel = (mod1(b, nc) == nc || b == np) ? "share of identified positives (%)" : "")
        push!(panels, pnl)
    end
    lg = plot(; framestyle = :none, legend = :left, legendfontsize = 7, grid = false, xticks = false, yticks = false)
    _incidence_legend!(lg, fc_labels, fc_cols, inc_cal !== nothing; axis = "LEFT axis")
    plot!(lg, [NaN], [NaN]; color = col_a, lw = 2.0, label = "RIGHT axis — SGTF share (Alpha proxy)")
    plot!(lg, [NaN], [NaN]; color = col_s, lw = 2.0, ls = :dashdot, label = "RIGHT axis — S-positive share\n(≈ Delta from late Apr 2021)")
    plot!(lg, [NaN], [NaN]; color = col_a, ls = :dot, lw = 1.4,
          label = "ONS variant period: Alpha → Delta\n($(ONS_DELTA_PERIOD_START))")
    for s in _AB_SEGMENTS
        plot!(lg, [NaN], [NaN]; seriestype = :shape, color = seg_c[s], fillalpha = 0.4, linealpha = 0, label = seg_l[s])
    end
    plot!(lg, [NaN], [NaN]; color = :black, ls = :dash, lw = 1, label = "forecast starts (end of t₀ week)")
    if nr * nc > np
        push!(panels, lg)
        for _ in 1:(nr * nc - np - 1); push!(panels, plot(; framestyle = :none)); end
    else
        plot!(panels[1]; legend = :topleft, legendfontsize = 5)
    end
    ptitle = "Weekly infections around forecast origin $(win.origin) (10j's window), ENGLAND — observed, " *
             "in-sample calibration and forecast, against the Alpha/Delta mix\n" *
             "RIGHT axis: ONS CIS S-gene-target-failure share (Alpha proxy) among identified positives — " *
             "⚠ England-level, NOT age-specific: ONS publishes no variant split by age; the same curve is in every panel"
    fig = plot(panels...; layout = (nr, nc), size = (430 * nc, 400 * nr),
               left_margin = 8Plots.mm, bottom_margin = 12Plots.mm, top_margin = 6Plots.mm,
               right_margin = 12Plots.mm, plot_titlevspan = 0.06,
               plot_title = ptitle, plot_titlefontsize = 8)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, fname))
    return fig
end

##########################################################################
# §F — the variant mix across the whole CoMix span (England, not by age)
##########################################################################

# `_VARIANT_OF`, `_VARIANT_ORDER`, `ONS_VARIANT_PERIODS` and `ons_variant_mix` live in `ons_variant.jl`.

# Why ONS variant data are absent where they are — quoted, so a gap is never left unexplained.
# The second is ONS's own note on technical table 1c (verified in the 2021-09-03 release): the
# modelled variant tables were FROZEN at 2021-06-06 … 07-17 from 30 July, and resumed with the
# 2021-12-17 release (data from 2021-10-31). inc2prev re-scraped the frozen table weekly, which is
# why `cis_variants.csv` repeats it 20 times; nothing is wrong with those rows.
const _ONS_VARIANT_GAP_NOTES = [
    (Date(2020, 11, 22), "no ONS variant data\n(SGTF reliable from\nmid-Nov 2020)"),
    (Date(2021, 10, 31), "ONS stopped updating its\nvariant tables from 30 Jul 2021:\n\"99% of positive cases are\ncompatible with the Delta\nvariant\"")]

"""
    make_variant_mix_fig(mix; span, origins, window=nothing, res_dir="../res",
                         fname="16j_variant_mix.png") -> Plots.Plot

ONE wide panel: the daily England variant mix (`ons_variant_mix`) as a STACKED share over `span`,
monthly ticks via 14j's `_origin_span_xaxis!`. Lumped days are drawn hatched. Overlays:
ONS's own period boundaries (`ONS_VARIANT_PERIODS`, dotted), the first/last CoMix forecast origin
(grey dotted), and optionally `window = (lo, hi)` — 16j §1–§2's 16 weeks — as a light band. Days
before the first ONS variant data are labelled, not left as an unexplained gap.
"""
function make_variant_mix_fig(mix::DataFrame; span::Tuple{Date,Date}, origins::AbstractVector{Date},
                              window = nothing, res_dir::AbstractString = "../res",
                              fname::AbstractString = "16j_variant_mix.png")
    lo, hi = span
    days = sort(unique(mix.date[(lo .<= mix.date .<= hi)]))
    lut = Dict((r.date, r.variant) => r.share for r in eachrow(mix))
    pnl = plot(; size = (1500, 560), legend = :outerright, legendfontsize = 7,
                 ylabel = "share of gene-pattern-identified positives (%)", ylims = (0, 100),
                 yticks = 0:20:100, xrotation = 45, bottom_margin = 10Plots.mm, left_margin = 8Plots.mm,
                 top_margin = 8Plots.mm,
                 title = "ONS COVID-19 Infection Survey — variant mix, ENGLAND (not age-specific), " *
                         "across the CoMix forecast span\neach category is a TaqPath gene pattern under " *
                         "ONS's name for that period (SGTF = S-gene target failure); per day the latest release",
                 titlefontsize = 9)
    # Stack each CONTIGUOUS run of days on its own: a single series across a data gap would fill
    # the gap by linear interpolation and draw variant shares ONS never published.
    runs = Vector{Vector{Date}}()
    for d in days
        (isempty(runs) || d - last(last(runs)) > Day(1)) ? push!(runs, [d]) : push!(last(runs), d)
    end
    shown = Set{String}()
    for run in runs
        cum = zeros(length(run))
        for (c, lab, col) in _VARIANT_ORDER
            sh = [get(lut, (d, c), 0.0) for d in run]
            any(>(0), sh) || continue
            lower = copy(cum); cum .+= sh
            plot!(pnl, run, 100 .* cum; fillrange = 100 .* lower, color = col, lw = 0, fillalpha = 0.9,
                  fillstyle = c == "lumped_other" ? :/ : nothing, label = c in shown ? "" : lab)
            push!(shown, c)
        end
    end
    # Gaps: before the first data day, and between runs — shaded and EXPLAINED.
    gaps = Tuple{Date,Date}[]
    first(days) > lo && push!(gaps, (lo, first(days) - Day(1)))
    for k in 1:(length(runs) - 1)
        push!(gaps, (last(runs[k]) + Day(1), first(runs[k + 1]) - Day(1)))
    end
    last(days) < hi && push!(gaps, (last(days) + Day(1), hi))
    for (g0, g1) in gaps
        vspan!(pnl, [g0, g1]; color = :gray92, linealpha = 0, label = "")
        note = something(findfirst(n -> n[1] == g1 + Day(1), _ONS_VARIANT_GAP_NOTES), 0)
        txt = note == 0 ? "no ONS variant data" : _ONS_VARIANT_GAP_NOTES[note][2]
        annotate!(pnl, g0 + (g1 - g0) ÷ 2, 50, text(txt, 7, :gray25))
    end
    # The 16j §1–§2 window as two LINES: a translucent band disappears under the opaque stack.
    window === nothing || vline!(pnl, [window[1], window[2]]; color = :steelblue, ls = :dash, lw = 2,
                                 label = "16j §1–§2 window (origin $(window[3]))")
    for (k, (nm, d)) in enumerate(ONS_VARIANT_PERIODS)
        vline!(pnl, [d]; color = :black, ls = :dot, lw = 1.2, label = k == 1 ? "ONS variant period starts" : "")
        annotate!(pnl, d + Day(2), 97, text(nm, 7, :left, :black))
    end
    vline!(pnl, [minimum(origins), maximum(origins)]; color = :gray40, ls = :dash, lw = 1,
           label = "first / last CoMix forecast origin")
    _origin_span_xaxis!(pnl, (lo, hi))
    plot!(pnl; ylims = (0, 100))
    mkpath(res_dir)
    savefig(pnl, joinpath(res_dir, fname))
    return pnl
end
