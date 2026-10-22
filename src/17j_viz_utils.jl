# 17j_viz_utils.jl — FROZEN-CONTACT (h0, unconditional) vs CONDITIONAL forecast comparison
# (2026-10-17, user request).
#
# The CONDITIONAL forecast (9j) is the contact-updated iterate: horizon h is fitted on contacts up
# to t₀+h and its NGM is frozen at the t₀+h contact matrix — it is TOLD the future contacts. The
# FROZEN forecast (`two_stage_forecast(…; frozen = true)`) is fitted on contacts up to t₀ only
# (`FROZEN_H = 0`) and iterates every horizon with C* (and antibody) frozen at t₀. Same iterator,
# same noise seeds, same median lag plug — the two differ only in their contacts and fit.
#
# Requires `forecast_utils.jl` → `9j_viz_utils.jl` → `15j_viz_utils.jl` (for `load_9j_assembly`, the
# read-only reader of 9j's conditional cache). 17j never builds the conditional assembly itself.
#
# ⚠ `assemble_or_load_forecasts(…; frozen = true)` FITS ON MISS, like the conditional path. Call
# `assert_h0_artefacts` first — that assert is the no-refit guard.

const SCHEMES      = ("conditional", "frozen")
const SCHEME_STYLE = Dict("conditional" => :solid, "frozen" => :dash)

"17j's own frozen assembly cache — NEVER 9j's default `9j_assembly_*` (a frozen call there would clobber it)."
frozen_cache_path(cfg::FrameworkConfig; save_dir::AbstractString = "../dt_intermediate") =
    joinpath(save_dir, "17j_assembly_frozen_$(contacts_label(cfg)).jld2")

"""
    assert_h0_artefacts(wins, combos, cfg; save_dir) -> n_checked

Error unless every frozen-contact artefact exists: the `8j_s1_*_h0` chain of each Stage-1 degree
model and the `8j_s2_*_h0` pooled file of each combo, for every origin. Without it a miss inside
`two_stage_forecast(…; frozen = true)` falls through to `fit_or_load_stage2` and FITS, serially,
writing into the live token.
"""
function assert_h0_artefacts(wins, combos, cfg::FrameworkConfig;
                             save_dir::AbstractString = "../dt_intermediate")
    tag = contacts_label(cfg)
    want = String[]
    for w in wins, (dm, nb) in combos
        needs_stage1(dm) &&
            push!(want, stage1_path(dm, w.origin, FROZEN_H; contacts = tag, save_dir = save_dir))
        push!(want, stage2_path(dm, nb, w.origin, FROZEN_H; contacts = tag, save_dir = save_dir))
    end
    unique!(want)
    miss = filter(!isfile, want)
    isempty(miss) || error("""
        17j: $(length(miss)) of $(length(want)) frozen-contact (h$(FROZEN_H)) artefacts are missing,
        e.g. $(basename.(first(miss, 3))).
        Run 8j §2b (FIT_H0=true) for the same FIT_END/ORIGIN_MIN first — 17j would otherwise REFIT them.""")
    return length(want)
end

"""
    check_h0_artefact_shapes(wins, combos, cfg; save_dir) -> DataFrame

Open every h0 artefact and check it IS an h0 one, not merely named like one: a Stage-1 chain must
carry `n_fit − 1` `z_c` coordinates (the level is sum-to-zero over `Tn = n_fit + h` weeks, so this
is `Tn − 1` and pins `h = 0`), and a Stage-2 pooled file must hold one `Cstar_end` per Stage-1 draw.
Returns one row per file with `ok`.
"""
function check_h0_artefact_shapes(wins, combos, cfg::FrameworkConfig;
                                  save_dir::AbstractString = "../dt_intermediate")
    tag = contacts_label(cfg)
    rows = NamedTuple[]
    for dm in unique(first.(combos)), w in wins
        needs_stage1(dm) || continue
        p = stage1_path(dm, w.origin, FROZEN_H; contacts = tag, save_dir = save_dir)
        nm = string.(names(load(p, "result")))
        nzc = count(startswith("z_c["), nm)
        push!(rows, (; stage = "s1", file = basename(p), n = nzc, want = cfg.n_fit - 1,
                       ok = nzc == cfg.n_fit - 1))
    end
    for (dm, nb) in combos, w in wins
        p = stage2_path(dm, nb, w.origin, FROZEN_H; contacts = tag, save_dir = save_dir)
        pooled = load(p, "pooled")
        n = length(pooled.Cstar_end)
        push!(rows, (; stage = "s2", file = basename(p), n = n, want = pooled.n_post,
                       ok = n == pooled.n_post &&
                            all(m -> size(m) == (size(pooled.susc, 2), size(pooled.susc, 2)),
                                pooled.Cstar_end)))
    end
    return DataFrame(rows)
end

# ── Scores side by side ────────────────────────────────────────────────────────────────
"""
    scheme_compare_frames(scores_c, scores_f; scale) -> (; by_h, by_dt_h)

Join the two schemes' `score_wis` frames on their unit keys. `by_h` is model × horizon (`wis_*`,
`ratio = wis_frozen / wis_conditional`, both coverages and biases); `by_dt_h` is model × origin ×
horizon with the PAIRED difference `diff = wis_frozen − wis_conditional` (< 0 ⇒ frozen better).
Inner joins: a unit present in only one scheme is dropped, and the count is returned so it is seen.
"""
function scheme_compare_frames(scores_c, scores_f; scale::AbstractString = WIS_SCALE)
    pick(df, keys, sfx) = rename(select(@subset(df, :scale .== scale), vcat(keys,
                                 [:wis, :bias, :interval_coverage_50, :interval_coverage_90])),
                                 [:wis => Symbol("wis_", sfx), :bias => Symbol("bias_", sfx),
                                  :interval_coverage_50 => Symbol("cov50_", sfx),
                                  :interval_coverage_90 => Symbol("cov90_", sfx)])
    kh = [:model, :horizon]
    by_h = innerjoin(pick(scores_c.by_model_h, kh, "conditional"),
                     pick(scores_f.by_model_h, kh, "frozen"); on = kh)
    by_h.ratio = by_h.wis_frozen ./ by_h.wis_conditional
    sort!(by_h, kh)
    kd = [:model, :forecast_date, :horizon]
    c = pick(scores_c.by_model_dt_h, kd, "conditional")
    f = pick(scores_f.by_model_dt_h, kd, "frozen")
    by_dt_h = innerjoin(c, f; on = kd)
    by_dt_h.diff = by_dt_h.wis_frozen .- by_dt_h.wis_conditional
    sort!(by_dt_h, kd)
    n_dropped = nrow(c) + nrow(f) - 2 * nrow(by_dt_h)
    n_dropped == 0 || @warn "17j: $(n_dropped) model×origin×horizon units scored in only one scheme — dropped from the paired frame"
    return (; by_h, by_dt_h, n_dropped)
end

"""
    paired_ratio_ci(by_dt_h; B, level, seed) -> DataFrame

Per model × horizon, `ratio = mean(wis_frozen) / mean(wis_conditional)` over origins with a
`level` bootstrap interval from resampling ORIGINS (both schemes resampled together, so the
interval is for the PAIRED comparison). Origins are weekly and autocorrelated, so the interval is
optimistic — read it as a noise guide, not a test.
"""
function paired_ratio_ci(by_dt_h::DataFrame; B::Int = 2000, level::Real = 0.90, seed::Int = 1236)
    rng = MersenneTwister(seed)
    a = (1 - level) / 2
    rows = NamedTuple[]
    for g in groupby(by_dt_h, [:model, :horizon])
        wc, wf = g.wis_conditional, g.wis_frozen
        n = length(wc)
        boots = [begin
                     ix = rand(rng, 1:n, n)
                     mean(@view wf[ix]) / mean(@view wc[ix])
                 end for _ in 1:B]
        push!(rows, (; model = g.model[1], horizon = g.horizon[1], n_origins = n,
                       ratio = mean(wf) / mean(wc),
                       lo = quantile(boots, a), hi = quantile(boots, 1 - a),
                       frac_frozen_better = mean(wf .< wc)))
    end
    return sort!(DataFrame(rows), [:model, :horizon])
end

# ── Figures ────────────────────────────────────────────────────────────────────────────
"""
    plot_scheme_wis_by_horizon(cmp, labels, model_cols, cfg) -> Plot

One panel per model (layout from `panel_grid`): mean log-scale WIS by horizon, conditional solid vs
frozen dashed. Panels share the y-axis so the levels compare across models too.
"""
function plot_scheme_wis_by_horizon(cmp, labels, model_cols, cfg)
    yl = extrema(vcat(cmp.by_h.wis_conditional, cmp.by_h.wis_frozen))
    yl = (0.0, yl[2] * 1.08)
    panels = Plots.Plot[]
    for (ci, m) in enumerate(labels)
        s = sort(@subset(cmp.by_h, :model .== m), :horizon)
        p = plot(; title = m, titlefontsize = 8, xlabel = "horizon (weeks)", ylabel = "WIS (log)",
                 xticks = collect(cfg.horizons), ylim = yl, legend = (ci == 1 ? :topleft : false),
                 legendfontsize = 6)
        for sch in SCHEMES
            plot!(p, s.horizon, s[!, Symbol("wis_", sch)]; color = model_cols[ci], lw = 2,
                  ls = SCHEME_STYLE[sch], marker = (sch == "frozen" ? :diamond : :circle), ms = 3,
                  label = sch)
        end
        push!(panels, p)
    end
    nr, nc = panel_grid(length(panels))
    return plot(panels...; layout = (nr, nc), size = (380 * nc, 320 * nr),
                plot_title = "17j — WIS by horizon: conditional (contacts to t₀+h) vs frozen (contacts to t₀)",
                plot_titlefontsize = 10)
end

"""
    plot_scheme_ratio_by_horizon(ci_frame, labels, model_cols, cfg; null_label) -> Plot

`mean WIS_frozen / mean WIS_conditional` by horizon per model, with the paired-origin bootstrap
band. > 1 ⇒ knowing the future contacts helped. The NULL model's C* is the same constant in both
schemes, so its line is the Monte-Carlo noise floor of the comparison — drawn grey and dashed.
"""
function plot_scheme_ratio_by_horizon(ci_frame, labels, model_cols, cfg;
                                      null_label::AbstractString = "no-contact|null")
    p = plot(; xlabel = "horizon (weeks)", ylabel = "WIS frozen / WIS conditional (log scale WIS)",
             title = "> 1 ⇒ the future contacts helped; null = noise floor (same C* in both schemes)",
             titlefontsize = 8, legend = :topleft, legendfontsize = 6,
             xticks = collect(cfg.horizons), size = (850, 520))
    hline!(p, [1.0]; color = :gray, ls = :dot, label = "")
    for (ci, m) in enumerate(labels)
        s = sort(@subset(ci_frame, :model .== m), :horizon)
        isnull = m == null_label
        plot!(p, s.horizon, s.ratio; ribbon = (s.ratio .- s.lo, s.hi .- s.ratio), fillalpha = 0.10,
              color = isnull ? :grey40 : model_cols[ci], ls = isnull ? :dash : :solid, lw = 2,
              marker = :circle, ms = 3, label = isnull ? "$(m) (noise floor)" : m)
    end
    plot!(p; plot_title = "17j — relative WIS of the frozen-contact forecast (90% paired bootstrap)",
          plot_titlefontsize = 10)
    return p
end

"""
    plot_scheme_diff_over_time(cmp, labels, model_cols, cfg; models) -> Plot

Per-origin paired difference `WIS_frozen − WIS_conditional` (log scale), one panel per horizon,
for `models`; > 0 ⇒ the conditional forecast was better at that origin. Pandemic phases shaded
(after the series — `shade_periods!` must come last).
"""
function plot_scheme_diff_over_time(cmp, labels, model_cols, cfg;
                                    models = ["unweighted-negbin|mean", "weighted-hweibull|mean",
                                              "weighted-hweibull|neighbourhood", "no-contact|null"])
    panels = Plots.Plot[]
    for (k, h) in enumerate(cfg.horizons)
        sub = @subset(cmp.by_dt_h, :horizon .== h)
        p = plot(; title = "horizon $h (wk ahead)", titlefontsize = 8, xlabel = "forecast origin",
                 ylabel = "WIS frozen − conditional", legend = (k == 1 ? :topleft : false),
                 legendfontsize = 6, xrotation = 45)
        for (ci, m) in enumerate(labels)
            m in models || continue
            s = sort(@subset(sub, :model .== m), :forecast_date)
            plot!(p, s.forecast_date, s.diff; color = model_cols[ci], lw = 1.3, marker = :circle,
                  ms = 2, label = m)
        end
        hline!(p, [0.0]; color = :gray, ls = :dash, label = "")
        isempty(sub.forecast_date) ||
            shade_periods!(p, minimum(sub.forecast_date), maximum(sub.forecast_date))
        push!(panels, p)
    end
    nr, nc = panel_grid(length(panels))
    return plot(panels...; layout = (nr, nc), size = (1150, 780),
                plot_title = "17j — per-origin WIS difference, frozen − conditional (log scale; > 0 ⇒ conditional better)",
                plot_titlefontsize = 10)
end

"""
    plot_scheme_coverage(cmp, labels, model_cols, cfg) -> Plot

50% and 90% central-interval coverage by horizon; colour = model, line style = scheme.
"""
function plot_scheme_coverage(cmp, labels, model_cols, cfg)
    panels = Plots.Plot[]
    for lvl in (50, 90)
        p = plot(; title = "$(lvl)% central interval (solid conditional, dashed frozen)",
                 titlefontsize = 8, xlabel = "horizon (weeks)", ylabel = "empirical coverage",
                 ylim = (0, 1), xticks = collect(cfg.horizons),
                 legend = (lvl == 50 ? :bottomleft : false), legendfontsize = 6)
        hline!(p, [lvl / 100]; color = :gray, ls = :dot, label = "nominal")
        for (ci, m) in enumerate(labels)
            s = sort(@subset(cmp.by_h, :model .== m), :horizon)
            for sch in SCHEMES
                plot!(p, s.horizon, s[!, Symbol("cov", lvl, "_", sch)]; color = model_cols[ci],
                      ls = SCHEME_STYLE[sch], lw = 1.8, marker = :circle, ms = 2,
                      label = sch == "conditional" ? m : "")
            end
        end
        push!(panels, p)
    end
    return plot(panels...; layout = (1, 2), size = (1100, 460),
                plot_title = "17j — calibration under each scheme", plot_titlefontsize = 10)
end

"""
    plot_scheme_rank_by_horizon(cmp, labels, model_cols, cfg; ref) -> Plot

Relative WIS against `ref` (default `REF_MODEL`) by horizon, one panel per scheme on a shared
y-axis: does the model ORDERING survive taking the future contacts away?
"""
function plot_scheme_rank_by_horizon(cmp, labels, model_cols, cfg; ref::AbstractString = REF_MODEL)
    rel = Dict{String,DataFrame}()
    for sch in SCHEMES
        col = Symbol("wis_", sch)
        r = Dict(row.horizon => row[col] for row in eachrow(@subset(cmp.by_h, :model .== ref)))
        rel[sch] = transform(cmp.by_h, [:horizon, col] => ByRow((h, w) -> w / get(r, h, NaN)) => :rwis)
    end
    yl = extrema(filter(isfinite, vcat(rel["conditional"].rwis, rel["frozen"].rwis)))
    yl = (min(yl[1], 1.0) * 0.95, max(yl[2], 1.0) * 1.05)
    panels = Plots.Plot[]
    for (k, sch) in enumerate(SCHEMES)
        p = plot(; title = sch, titlefontsize = 9, xlabel = "horizon (weeks)",
                 ylabel = "relative WIS (vs $(ref))", ylim = yl, xticks = collect(cfg.horizons),
                 legend = (k == 1 ? :topleft : false), legendfontsize = 6)
        hline!(p, [1.0]; color = :gray, ls = :dash, label = "")
        for (ci, m) in enumerate(labels)
            s = sort(@subset(rel[sch], :model .== m), :horizon)
            plot!(p, s.horizon, s.rwis; color = model_cols[ci], lw = 2, marker = :circle, ms = 3,
                  label = m)
        end
        push!(panels, p)
    end
    return plot(panels...; layout = (1, 2), size = (1100, 460),
                plot_title = "17j — relative skill under each scheme (log-scale WIS)",
                plot_titlefontsize = 10)
end

"""
    plot_scheme_forecast_panels(fc_c, fc_f, wins, lbl, cfg; grid, inf, n) -> Plot

Total-infection forecast fans (median + 90% band) of model `lbl` under BOTH schemes, at `n` evenly
spaced origins, over the observed series (fit weeks ++ realised targets). Pass
`inf = load_raw_infection_inputs()` to read the estimates CSV once rather than per panel.
"""
function plot_scheme_forecast_panels(fc_c, fc_f, wins, lbl::AbstractString, cfg;
                                     grid = cis_age_grid(), inf = nothing, n::Integer = 9)
    origins = [w.origin for w in wins]
    sel = pick_origins(origins; n = n)
    H = length(cfg.horizons)
    cols = Dict("conditional" => :steelblue, "frozen" => :darkorange)
    panels = Plots.Plot[]
    for (pi, origin) in enumerate(sel)
        win   = wins[findfirst(==(origin), origins)]
        wd    = inf === nothing ? load_window_data(win; grid = grid) :
                                  load_window_data(win, inf.df, inf.tmap; grid = grid)
        truth = load_forecast_truth(win; grid = grid)
        x_hist = week_mid.(win.fit_weeks)
        y_hist = vec(sum(wd.I_mean[:, (cfg.smax + 1):end]; dims = 1))
        x_fore = week_mid.(win.forecast_weeks)
        y_fore = [sum(truth[:, h]) for h in 1:H]
        p = plot(; title = string(origin), titlefontsize = 7, xrotation = 45,
                 legend = (pi == 1 ? :topleft : false), legendfontsize = 5)
        plot!(p, vcat(x_hist, x_fore), vcat(y_hist, y_fore);
              color = :black, lw = 2, marker = :circle, ms = 2, label = "observed")
        vline!(p, [week_mid(win.origin)]; color = :gray, ls = :dash, lw = 1, label = "")
        for (sch, store) in (("conditional", fc_c), ("frozen", fc_f))
            haskey(store, (origin, lbl)) || continue
            tot = dropdims(sum(store[(origin, lbl)]; dims = 1); dims = 1)
            med = [_fmed(tot[h, :]) for h in 1:H]
            lo  = [_fq(tot[h, :], 0.05) for h in 1:H]
            hi  = [_fq(tot[h, :], 0.95) for h in 1:H]
            plot!(p, x_fore, med; color = cols[sch], lw = 1.6, ls = SCHEME_STYLE[sch],
                  ribbon = (med .- lo, hi .- med), fillalpha = 0.12,
                  label = (pi == 1 ? sch : ""))
        end
        push!(panels, p)
    end
    nr, nc = panel_grid(length(panels))
    return plot(panels...; layout = (nr, nc), size = (430 * nc, 330 * nr),
                plot_title = "17j — $(lbl): conditional vs frozen forecast (90% band)",
                plot_titlefontsize = 10)
end

# ══════════════════════════════════════════════════════════════════════════════════════
# SIX-SCENARIO COMPARISON (2026-10-21, user request) — the variant-specific renewal (`VariantNGM`,
# inst/3 §12; fitted in 9j §V) against the existing schemes, per contact model:
#   1 no contact                 `no-contact|null`                 (9j conditional cache)
#   2 contact                    `<c>`                             (9j conditional cache)
#   3 contact (frozen)           `<c>` frozen                      (17j frozen cache)
#   4 variant                    `no-contact|null-variant`         (17j variant cache)
#   5 variant + contact          `<c>-variant`                     (17j variant cache)
#   6 variant + contact (frozen) `<c>-variant` frozen              (17j variant frozen cache)
# Scenarios 1 and 4 use no contact data, so they are the same series in every contact panel.
# A frozen series is named `<label>@frozen` — a SERIES name for scoring/plotting only; never pass
# it to `stage1_chain_path`/`stage2_pooled_path`, which split on `|` and would ignore the suffix.
# ══════════════════════════════════════════════════════════════════════════════════════

const SCENARIO_ORDER = ["no contact", "contact", "contact (frozen)",
                        "variant", "variant + contact", "variant + contact (frozen)"]
const SCENARIO_STYLE = Dict(
    "no contact"                 => (:grey40,     :solid, :circle),
    "contact"                    => (:steelblue,  :solid, :circle),
    "contact (frozen)"           => (:steelblue,  :dash,  :diamond),
    "variant"                    => (:firebrick,  :solid, :circle),
    "variant + contact"          => (:darkorange, :solid, :circle),
    "variant + contact (frozen)" => (:darkorange, :dash,  :diamond))
const NULL_LABEL         = "no-contact|null"
const NULL_VARIANT_LABEL = "no-contact|null-variant"

"Own caches for the variant forecasts — never 9j's or 17j's frozen-vs-conditional ones."
variant_cache_path(cfg::FrameworkConfig; frozen::Bool, save_dir::AbstractString = "../dt_intermediate") =
    joinpath(save_dir, "17j_assembly_variant$(frozen ? "_frozen" : "")_$(contacts_label(cfg)).jld2")

"`series_name(label, frozen)` — the scoring/plotting name of one scenario series."
series_name(label::AbstractString, frozen::Bool) = frozen ? string(label, "@frozen") : String(label)

"""
    scenario_table(contact_labels) -> DataFrame

One row per (contact panel × scenario): `contact` (the contact combo label), `scenario`
(`SCENARIO_ORDER`), `label` (the model label), `frozen`, `series` (`series_name`).
"""
function scenario_table(contact_labels)
    rows = NamedTuple[]
    for c in contact_labels
        for (sc, lbl, fz) in (("no contact", NULL_LABEL, false), ("contact", c, false),
                              ("contact (frozen)", c, true), ("variant", NULL_VARIANT_LABEL, false),
                              ("variant + contact", c * "-variant", false),
                              ("variant + contact (frozen)", c * "-variant", true))
            push!(rows, (; contact = String(c), scenario = sc, label = String(lbl), frozen = fz,
                           series = series_name(lbl, fz)))
        end
    end
    return DataFrame(rows)
end

"""
    rename_series(qall, fc_store, frozen) -> (qall, fc_store)

Copy of one assembly's products with every model renamed to its `series_name` (identity for a
conditional assembly). The quantile frame and the fan store stay keyed alike.
"""
function rename_series(qall::DataFrame, fc_store, frozen::Bool)
    q = copy(qall); q.model = series_name.(q.model, frozen)
    fc = Dict((k[1], series_name(k[2], frozen)) => v for (k, v) in fc_store)
    return q, fc
end

"""
    scenario_pair_ratios(by_dt_h, st; B, level, seed, scale) -> DataFrame

Paired contrasts per contact panel × horizon, `ratio = mean WIS(num) / mean WIS(den)` over origins
with a `level` origin-bootstrap band (both series resampled together), on `scale`:
- `vs no contact` for scenarios 2–6 (relative skill against scenario 1);
- the named contrasts `variant gain` (4/1, 5/2, 6/3), `contact gain` (2/1, 5/4) and
  `frozen penalty` (3/2, 6/5).
`by_dt_h` is `score_wis(...).by_model_dt_h` over the series names. Origins are weekly and
autocorrelated, so read the band as a noise guide, not a test.
"""
function scenario_pair_ratios(by_dt_h::DataFrame, st::DataFrame; B::Int = 2000, level::Real = 0.90,
                              seed::Int = 1236, scale::AbstractString = WIS_SCALE)
    rng = MersenneTwister(seed)
    a = (1 - level) / 2
    d = @subset(by_dt_h, :scale .== scale)
    look = Dict((r.model, r.forecast_date, r.horizon) => r.wis for r in eachrow(d))
    sname(c, sc) = only(@subset(st, :contact .== c, :scenario .== sc).series)
    contrasts = vcat([("vs no contact", sc, "no contact") for sc in SCENARIO_ORDER[2:end]],
                     [("variant gain", "variant", "no contact"),
                      ("variant gain", "variant + contact", "contact"),
                      ("variant gain", "variant + contact (frozen)", "contact (frozen)"),
                      ("contact gain", "contact", "no contact"),
                      ("contact gain", "variant + contact", "variant"),
                      ("frozen penalty", "contact (frozen)", "contact"),
                      ("frozen penalty", "variant + contact (frozen)", "variant + contact")])
    rows = NamedTuple[]
    for c in unique(st.contact), (kind, num, den) in contrasts, h in sort(unique(d.horizon))
        sn, sd = sname(c, num), sname(c, den)
        origins = sort(unique(@subset(d, :model .== sd, :horizon .== h).forecast_date))
        wn = [get(look, (sn, o, h), NaN) for o in origins]
        wd = [get(look, (sd, o, h), NaN) for o in origins]
        ok = isfinite.(wn) .& isfinite.(wd)
        @assert all(ok) "scenario_pair_ratios: $(count(!, ok)) origin(s) at h=$h lack a score for $sn or $sd"
        n = length(wn)
        boots = [begin ix = rand(rng, 1:n, n); mean(@view wn[ix]) / mean(@view wd[ix]) end for _ in 1:B]
        push!(rows, (; contact = c, contrast = kind, num = num, den = den, horizon = h, n_origins = n,
                       ratio = mean(wn) / mean(wd), lo = quantile(boots, a), hi = quantile(boots, 1 - a),
                       frac_num_better = mean(wn .< wd)))
    end
    return DataFrame(rows)
end

"""
    plot_scenario_wis_by_horizon(by_h, st, cfg; scale) -> Plot

One panel per contact model (layout from `panel_grid`), the six scenarios' mean WIS by horizon on a
shared y-axis. Scenarios 1 and 4 (no contact data) are identical in every panel.
"""
function plot_scenario_wis_by_horizon(by_h::DataFrame, st::DataFrame, cfg; scale::AbstractString = WIS_SCALE,
                                      subtitle::AbstractString = "")
    d = @subset(by_h, :scale .== scale)
    j = innerjoin(d, st; on = :model => :series)
    yl = (0.0, maximum(j.wis) * 1.08)
    panels = Plots.Plot[]
    for (k, c) in enumerate(unique(st.contact))
        p = plot(; title = c, titlefontsize = 8, xlabel = "horizon (weeks)", ylabel = "WIS ($scale)",
                 xticks = collect(cfg.horizons), ylim = yl, legend = (k == 1 ? :topleft : false),
                 legendfontsize = 6)
        for sc in SCENARIO_ORDER
            s = sort(@subset(j, :contact .== c, :scenario .== sc), :horizon)
            col, ls, mk = SCENARIO_STYLE[sc]
            plot!(p, s.horizon, s.wis; color = col, ls = ls, marker = mk, ms = 3, lw = 2, label = sc)
        end
        push!(panels, p)
    end
    nr, nc = panel_grid(length(panels))
    return plot(panels...; layout = (nr, nc), size = (400 * nc, 340 * nr),
                plot_title = "17j — WIS by horizon, six scenarios per contact model ($(scale) scale)$(subtitle)",
                plot_titlefontsize = 10)
end

"""
    plot_scenario_rel_wis(ratios, cfg) -> Plot

Relative WIS against scenario 1 (no contact, no variant) by horizon, one panel per contact model,
with the 90% paired origin-bootstrap band. < 1 ⇒ better than the no-contact null.
"""
function plot_scenario_rel_wis(ratios::DataFrame, cfg)
    r = @subset(ratios, :contrast .== "vs no contact")
    yl = (min(minimum(r.lo), 1.0) * 0.97, max(maximum(r.hi), 1.0) * 1.03)
    panels = Plots.Plot[]
    for (k, c) in enumerate(unique(r.contact))
        p = plot(; title = c, titlefontsize = 8, xlabel = "horizon (weeks)",
                 ylabel = "WIS / WIS(no contact)", ylim = yl, xticks = collect(cfg.horizons),
                 legend = (k == 1 ? :topleft : false), legendfontsize = 6)
        hline!(p, [1.0]; color = :grey40, ls = :solid, lw = 1, label = "no contact")
        for sc in SCENARIO_ORDER[2:end]
            s = sort(@subset(r, :contact .== c, :num .== sc), :horizon)
            col, ls, mk = SCENARIO_STYLE[sc]
            plot!(p, s.horizon, s.ratio; ribbon = (s.ratio .- s.lo, s.hi .- s.ratio), fillalpha = 0.10,
                  color = col, ls = ls, marker = mk, ms = 3, lw = 2, label = sc)
        end
        push!(panels, p)
    end
    nr, nc = panel_grid(length(panels))
    return plot(panels...; layout = (nr, nc), size = (400 * nc, 340 * nr),
                plot_title = "17j — relative WIS vs no contact (log-scale WIS; 90% paired origin bootstrap)",
                plot_titlefontsize = 10)
end

"""
    plot_variant_gain_over_time(by_dt_h, st, cfg; entries, scale) -> Plot

Per-origin paired difference `WIS(variant scenario) − WIS(its non-variant twin)`, one panel per
horizon: the no-contact pair (4 − 1) thick black, each contact model's conditional pair (5 − 2)
thin coloured. < 0 ⇒ the variant split helped at that origin. `entries` = `(variant, date)` pairs
marking the first origin each category got its own renewal (dotted verticals); phases shaded last.
"""
function plot_variant_gain_over_time(by_dt_h::DataFrame, st::DataFrame, cfg; entries = Tuple{String,Date}[],
                                     scale::AbstractString = WIS_SCALE)
    d = @subset(by_dt_h, :scale .== scale)
    look = Dict((r.model, r.forecast_date, r.horizon) => r.wis for r in eachrow(d))
    contacts = unique(st.contact)
    ccols = collect(palette(:Dark2_8))[mod1.(1:length(contacts), 8)]
    sname(c, sc) = only(@subset(st, :contact .== c, :scenario .== sc).series)
    panels = Plots.Plot[]
    for (k, h) in enumerate(cfg.horizons)
        origins = sort(unique(@subset(d, :horizon .== h).forecast_date))
        p = plot(; title = "horizon $h (wk ahead)", titlefontsize = 8, xlabel = "forecast origin",
                 ylabel = "WIS variant − non-variant", legend = (k == 1 ? :bottomleft : false),
                 legendfontsize = 5, xrotation = 45)
        for (ci, c) in enumerate(contacts)
            y = [look[(sname(c, "variant + contact"), o, h)] - look[(sname(c, "contact"), o, h)] for o in origins]
            plot!(p, origins, y; color = ccols[ci], lw = 1.0, alpha = 0.8, label = "$(c): (5) − (2)")
        end
        c1 = first(contacts)
        y = [look[(sname(c1, "variant"), o, h)] - look[(sname(c1, "no contact"), o, h)] for o in origins]
        plot!(p, origins, y; color = :black, lw = 2.2, label = "no contact: (4) − (1)")
        hline!(p, [0.0]; color = :gray, ls = :dash, label = "")
        for (nm, dt) in entries
            vline!(p, [dt]; color = :purple, ls = :dot, lw = 1, label = "")
            annotate!(p, dt, Plots.ylims(p)[1] + 0.05 * (Plots.ylims(p)[2] - Plots.ylims(p)[1]),
                      text(" " * nm, 5, :left, :bottom, :purple; rotation = 90))
        end
        isempty(origins) || shade_periods!(p, minimum(origins), maximum(origins))
        push!(panels, p)
    end
    nr, nc = panel_grid(length(panels))
    return plot(panels...; layout = (nr, nc), size = (1250, 820),
                plot_title = "17j — per-origin variant gain ($(scale)-scale WIS; < 0 ⇒ the variant split helped; dotted = a category enters)",
                plot_titlefontsize = 10)
end

"""
    plot_scenario_coverage(by_h, st, cfg) -> Plot

50% (top row) and 90% (bottom row) central-interval coverage by horizon, one column per contact model.
"""
function plot_scenario_coverage(by_h::DataFrame, st::DataFrame, cfg; scale::AbstractString = WIS_SCALE,
                                subtitle::AbstractString = "")
    j = innerjoin(@subset(by_h, :scale .== scale), st; on = :model => :series)
    contacts = unique(st.contact)
    panels = Plots.Plot[]
    for lvl in (50, 90), (k, c) in enumerate(contacts)
        col_ = Symbol("interval_coverage_", lvl)
        p = plot(; title = "$(c) — $(lvl)%", titlefontsize = 7, xlabel = "horizon (weeks)",
                 ylabel = "coverage", ylim = (0, 1), xticks = collect(cfg.horizons),
                 legend = (lvl == 50 && k == 1 ? :bottomleft : false), legendfontsize = 5)
        hline!(p, [lvl / 100]; color = :gray, ls = :dot, label = "nominal")
        for sc in SCENARIO_ORDER
            s = sort(@subset(j, :contact .== c, :scenario .== sc), :horizon)
            colr, ls, mk = SCENARIO_STYLE[sc]
            plot!(p, s.horizon, s[!, col_]; color = colr, ls = ls, marker = mk, ms = 2, msc = colr, lw = 1.6, label = sc)
        end
        push!(panels, p)
    end
    nc = length(contacts)
    return plot(panels...; layout = (2, nc), size = (330 * nc, 700),
                plot_title = "17j — 50% / 90% interval coverage, six scenarios$(subtitle)", plot_titlefontsize = 10)
end

"""
    takeover_origins(wins, cc, cfg; groups, k) -> Vector{Tuple{String,Vector{Date},Date}}

For each COG-UK group in `groups`, the `k` forecast origins centred on its TAKEOVER origin — the first
origin whose origin-week genome share (`variant_set(…).share0`, data up to that origin only) is ≥ 50%
— shifted to stay inside the origin grid. Default: Alpha (Dec 2020), Delta (May 2021), Omicron BA.1
(Dec 2021), 5 each. Each entry is `(group, origins, takeover origin)`.
"""
function takeover_origins(wins, cc, cfg::FrameworkConfig;
                          groups = ["Alpha", "Delta", "Omicron BA.1"], k::Integer = 5)
    origins = [w.origin for w in wins]
    rule = variant_rule(cfg)
    sh(o, g) = begin
        vs = variant_set(cc, o, rule)
        i = findfirst(m -> g in m, vs.members)
        # a group pooled with others (residual/dominant) has no share of its own: count its genomes
        i === nothing ? 0.0 : (length(vs.members[i]) == 1 ? vs.share0[i] :
            cc.n[_cog_row(cc, o), findfirst(==(g), cc.groups)] / sum(cc.n[_cog_row(cc, o), :]))
    end
    out = Tuple{String,Vector{Date},Date}[]
    for g in groups
        i50 = findfirst(o -> sh(o, g) >= 0.5, origins)
        @assert i50 !== nothing "takeover_origins: $g never reaches 50% over the origin grid"
        lo = clamp(i50 - k ÷ 2, 1, length(origins) - k + 1)
        push!(out, (g, origins[lo:(lo + k - 1)], origins[i50]))
    end
    @assert allunique(reduce(vcat, [o[2] for o in out])) "takeover_origins: windows overlap"
    return out
end

"""
    plot_scenario_forecast_panels(fc_store, wins, st, contact, cfg; groups, scenarios, grid, inf) -> Plot

9j's `plot_forecast_panels` for the six scenarios of ONE contact model: per origin, the observed
total infections (fit weeks ++ realised targets) and each scenario's forecast fan (median + 90% band;
frozen scenarios dashed). `groups` = `takeover_origins(…)` lays the panels out as one COLUMN per
takeover and one ROW per origin; the first origin at ≥ 50% is marked in its title. Each panel's y-axis
is capped at `ycap ×` the largest observed value it shows (the variant fans at the Omicron origins run
100–1000× above the data and would otherwise flatten every observed line); the number of fans whose
upper band crosses the cap is COUNTED in the title, never hidden silently.
"""
function plot_scenario_forecast_panels(fc_store, wins, st::DataFrame, contact::AbstractString, cfg;
                                       groups, scenarios = SCENARIO_ORDER,
                                       grid = cis_age_grid(), inf = nothing, ycap::Real = 3.0)
    fmt(y) = abs(y) >= 1e6 ? string(round(y / 1e6; digits = 1), "M") : string(round(Int, y / 1e3), "k")
    all_o = [w.origin for w in wins]
    H = length(cfg.horizons)
    k = length(first(groups)[2])
    @assert all(length(g[2]) == k for g in groups) "plot_scenario_forecast_panels: groups differ in length"
    panels = Plots.Plot[]
    for r in 1:k, (gname, gorig, g50) in groups                # row-major ⇒ one column per takeover
        origin = gorig[r]
        win   = wins[findfirst(==(origin), all_o)]
        wd    = inf === nothing ? load_window_data(win; grid = grid) :
                                  load_window_data(win, inf.df, inf.tmap; grid = grid)
        truth = load_forecast_truth(win; grid = grid)
        x_hist = week_mid.(win.fit_weeks)
        y_hist = vec(sum(wd.I_mean[:, (cfg.smax + 1):end]; dims = 1))
        x_fore = week_mid.(win.forecast_weeks)
        y_fore = [sum(truth[:, h]) for h in 1:H]
        first_panel = isempty(panels)
        ymax = ycap * maximum(vcat(y_hist, y_fore))
        p = plot(; titlefontsize = 7, xrotation = 45, ylim = (0, ymax),
                 legend = (first_panel ? :topleft : false), legendfontsize = 5, yformatter = fmt)
        plot!(p, vcat(x_hist, x_fore), vcat(y_hist, y_fore);
              color = :black, lw = 2, marker = :circle, ms = 2, label = "observed")
        vline!(p, [week_mid(win.origin)]; color = :gray, ls = :dash, lw = 1, label = "")
        n_off = 0
        for sc in scenarios
            sname = only(@subset(st, :contact .== contact, :scenario .== sc).series)
            haskey(fc_store, (origin, sname)) || continue
            tot = dropdims(sum(fc_store[(origin, sname)]; dims = 1); dims = 1)
            med = [_fmed(tot[h, :]) for h in 1:H]
            lo  = [_fq(tot[h, :], 0.05) for h in 1:H]
            hi  = [_fq(tot[h, :], 0.95) for h in 1:H]
            colr, ls, _ = SCENARIO_STYLE[sc]
            plot!(p, x_fore, med; color = colr, lw = 1.5, ls = ls, ribbon = (med .- lo, hi .- med),
                  fillalpha = 0.08, label = (first_panel ? sc : ""))
            n_off += any(>(ymax), filter(isfinite, hi))
        end
        plot!(p; title = "$(gname) — $(origin)$(origin == g50 ? "  (first ≥50%)" : "")" *
                         (n_off > 0 ? "  [$(n_off) fan(s) off-scale]" : ""))
        push!(panels, p)
    end
    nc = length(groups)
    return plot(panels...; layout = (k, nc), size = (520 * nc, 300 * k),
                plot_title = "17j — $(contact): total infections around each takeover, six scenarios (median + 90% band; dashed = frozen)",
                plot_titlefontsize = 10)
end

"""
    plot_scenario_wis_over_time(by_dt_h, st, contact, cfg; ref, entries, scale) -> Plot

One figure per contact model, one ROW per horizon. Left: each scenario's per-origin WIS (`scale`, log by
default) — `plot_wis_by_horizon_over_time`'s form. Right: the running sum over origins of
`WIS(scenario) − WIS(ref)` (default `ref` = "no contact", drawn as the 0 line) — `plot_wis_diff_over_time`'s
form, so a line's endpoint is the whole-period WIS difference and a line below 0 beat the reference
cumulatively. Dotted verticals mark where a variant category enters (`entries`); phases shaded last.
"""
function plot_scenario_wis_over_time(by_dt_h::DataFrame, st::DataFrame, contact::AbstractString, cfg;
                                     ref::AbstractString = "no contact", entries = Tuple{String,Date}[],
                                     scale::AbstractString = WIS_SCALE)
    d = @subset(by_dt_h, :scale .== scale)
    sname(sc) = only(@subset(st, :contact .== contact, :scenario .== sc).series)
    panels = Plots.Plot[]
    for (k, h) in enumerate(cfg.horizons)
        sub = @subset(d, :horizon .== h)
        pl = plot(; title = "horizon $h — WIS ($scale)", titlefontsize = 8, ylabel = "WIS ($scale)",
                  legend = (k == 1 ? :topleft : false), legendfontsize = 5, xrotation = 45, ylim = (0, Inf),
                  xlabel = (h == last(cfg.horizons) ? "forecast origin" : ""))
        for sc in SCENARIO_ORDER
            s = sort(@subset(sub, :model .== sname(sc)), :forecast_date)
            colr, ls, _ = SCENARIO_STYLE[sc]
            plot!(pl, s.forecast_date, s.wis; color = colr, ls = ls, lw = 1.4, label = sc)
        end
        ref_by = Dict(r.forecast_date => r.wis for r in eachrow(@subset(sub, :model .== sname(ref))))
        pr = plot(; title = "horizon $h — cumulative WIS − WIS($ref)", titlefontsize = 8,
                  ylabel = "cumulative diff", legend = false, xrotation = 45,
                  xlabel = (h == last(cfg.horizons) ? "forecast origin" : ""))
        for sc in SCENARIO_ORDER
            sc == ref && continue
            s = sort(@subset(sub, :model .== sname(sc)), :forecast_date)
            dts = Date[]; cum = Float64[]; acc = 0.0
            for (w, dt) in zip(s.wis, s.forecast_date)
                haskey(ref_by, dt) || continue
                acc += w - ref_by[dt]
                push!(dts, dt); push!(cum, acc)
            end
            colr, ls, _ = SCENARIO_STYLE[sc]
            plot!(pr, dts, cum; color = colr, ls = ls, lw = 1.6, label = sc)
        end
        hline!(pr, [0.0]; color = SCENARIO_STYLE[ref][1], ls = :solid, lw = 1, label = "")
        for (nm, dt) in entries
            vline!(pr, [dt]; color = :purple, ls = :dot, lw = 1, label = "")
        end
        if !isempty(ref_by)
            dmin, dmax = extrema(collect(keys(ref_by)))
            shade_periods!(pl, dmin, dmax; labels = (k == 1))
            shade_periods!(pr, dmin, dmax; labels = (k == 1))
        end
        push!(panels, pl, pr)
    end
    H = length(cfg.horizons)
    return plot(panels...; layout = (H, 2), size = (1400, 330 * H),
                plot_title = "17j — $(contact): WIS over time by horizon, six scenarios ($(scale) scale; right: cumulative vs $(ref), < 0 ⇒ better; dotted = a variant category enters)",
                plot_titlefontsize = 10)
end

"""
    scenario_period_label(d) -> String

`period_of(d)` (Munday 2023 Table 2, 9j), plus two explicit buckets for the origins outside every
phase — `"before Lockdown 2"` and `"after Opening up"`. 9j drops those; here they are kept because
the latter holds the Omicron origins. Never reuse 9j's `"(pre-Lockdown 2)"` label for both: it
mislabels the five late origins.
"""
function scenario_period_label(d::Date)
    p = period_of(d)
    ismissing(p) || return p
    return d < PERIODS[1][2] ? "before Lockdown 2" : "after Opening up"
end
const SCENARIO_PERIOD_ORDER = vcat(["before Lockdown 2"], PERIOD_ORDER, ["after Opening up"])

"""
    scenario_wis_by_period(by_dt_h, st; scale) -> DataFrame

Mean WIS (`scale`, log by default) per contact model × scenario × period × horizon over the origins in
that period, with `n_origins`. Periods from `scenario_period_label`.
"""
function scenario_wis_by_period(by_dt_h::DataFrame, st::DataFrame; scale::AbstractString = WIS_SCALE)
    d = @subset(by_dt_h, :scale .== scale)
    d = innerjoin(d, select(st, :series, :contact, :scenario); on = :model => :series)
    d.period = scenario_period_label.(d.forecast_date)
    return sort!(combine(groupby(d, [:contact, :scenario, :period, :horizon]),
                         :wis => mean => :wis, :forecast_date => (x -> length(unique(x))) => :n_origins),
                 [:contact, :period, :scenario, :horizon])
end

"""
    plot_scenario_wis_by_period(agg, contact, cfg; scale) -> Plot

ABSOLUTE mean WIS by horizon, one panel per pandemic period (`SCENARIO_PERIOD_ORDER`, those present),
the six scenarios as lines — the absolute-scale twin of 9j's `plot_rwis_by_period`. Each panel has its
own y-axis from 0 (periods differ by an order of magnitude); `n` origins in the title.
"""
function plot_scenario_wis_by_period(agg::DataFrame, contact::AbstractString, cfg; scale::AbstractString = WIS_SCALE)
    a = @subset(agg, :contact .== contact)
    present = [p for p in SCENARIO_PERIOD_ORDER if p in a.period]
    panels = Plots.Plot[]
    for (k, per) in enumerate(present)
        ap = @subset(a, :period .== per)
        p = plot(; title = "$(per) (n = $(maximum(ap.n_origins)))", titlefontsize = 8,
                 xlabel = "horizon (weeks)", ylabel = "WIS ($scale)", xticks = collect(cfg.horizons),
                 ylim = (0, maximum(ap.wis) * 1.08), legend = (k == 1 ? :topleft : false), legendfontsize = 5)
        for sc in SCENARIO_ORDER
            s = sort(@subset(ap, :scenario .== sc), :horizon)
            colr, ls, mk = SCENARIO_STYLE[sc]
            plot!(p, s.horizon, s.wis; color = colr, ls = ls, marker = mk, ms = 3, msc = colr, lw = 1.8, label = sc)
        end
        push!(panels, p)
    end
    nc = min(3, length(panels)); nr = cld(length(panels), nc)
    return plot(panels...; layout = (nr, nc), size = (420 * nc, 320 * nr),
                plot_title = "17j — $(contact): mean WIS by horizon per pandemic period, six scenarios ($(scale) scale)",
                plot_titlefontsize = 10)
end

"""
    plot_scenario_age_fans(fc_store, wins, st, contact, origin, cfg; grid, inf, ycap) -> Plot

AGE-STRATIFIED forecast at one `origin` for one contact model: one panel per CIS age bin plus an
all-ages panel; in each, the observed weekly infections (fit weeks ++ realised targets) and the six
scenarios' fans (median + 90% band; frozen dashed). y capped at `ycap ×` the panel's observed maximum,
with fans crossing it COUNTED in the panel title.
"""
function plot_scenario_age_fans(fc_store, wins, st::DataFrame, contact::AbstractString, origin::Date, cfg;
                                grid = cis_age_grid(), inf = nothing, ycap::Real = 3.0)
    fmt(y) = abs(y) >= 1e6 ? string(round(y / 1e6; digits = 1), "M") : string(round(Int, y / 1e3), "k")
    win   = wins[findfirst(w -> w.origin == origin, wins)]
    wd    = inf === nothing ? load_window_data(win; grid = grid) :
                              load_window_data(win, inf.df, inf.tmap; grid = grid)
    truth = load_forecast_truth(win; grid = grid)
    H = length(cfg.horizons); A = size(truth, 1)
    x_hist = week_mid.(win.fit_weeks); x_fore = week_mid.(win.forecast_weeks)
    sname(sc) = only(@subset(st, :contact .== contact, :scenario .== sc).series)
    panels = Plots.Plot[]
    for a in vcat(collect(1:A), 0)                       # 0 = all ages
        rows = a == 0 ? (1:A) : (a:a)
        y_hist = vec(sum(wd.I_mean[rows, (cfg.smax + 1):end]; dims = 1))
        y_fore = vec(sum(truth[rows, :]; dims = 1))
        ymax = ycap * maximum(vcat(y_hist, y_fore))
        p = plot(; xrotation = 45, ylim = (0, ymax), yformatter = fmt, titlefontsize = 8,
                 legend = (a == 1 ? :topleft : false), legendfontsize = 5)
        plot!(p, vcat(x_hist, x_fore), vcat(y_hist, y_fore); color = :black, lw = 2, marker = :circle,
              ms = 2, label = "observed")
        vline!(p, [week_mid(origin)]; color = :gray, ls = :dash, lw = 1, label = "")
        n_off = 0
        for sc in SCENARIO_ORDER
            key = (origin, sname(sc))
            haskey(fc_store, key) || continue
            f = fc_store[key]                             # A × H × draws
            tot = dropdims(sum(f[rows, :, :]; dims = 1); dims = 1)
            med = [_fmed(tot[h, :]) for h in 1:H]
            lo  = [_fq(tot[h, :], 0.05) for h in 1:H]
            hi  = [_fq(tot[h, :], 0.95) for h in 1:H]
            colr, ls, _ = SCENARIO_STYLE[sc]
            plot!(p, x_fore, med; color = colr, ls = ls, lw = 1.5, ribbon = (med .- lo, hi .- med),
                  fillalpha = 0.08, label = sc)
            n_off += any(>(ymax), filter(isfinite, hi))
        end
        plot!(p; title = (a == 0 ? "all ages" : "age $(grid.LAB[a])") *
                         (n_off > 0 ? "  [$(n_off) fan(s) off-scale]" : ""))
        push!(panels, p)
    end
    nc = 4; nr = cld(length(panels), nc)
    return plot(panels...; layout = (nr, nc), size = (400 * nc, 330 * nr),
                plot_title = "17j — $(contact): age-stratified forecast at origin $(origin), six scenarios (median + 90% band; dashed = frozen)",
                plot_titlefontsize = 10)
end

"""
    _point_forecast_inputs(fc_store, st, contact, origin, cfg; inf, grid, n_before, n_after) -> NamedTuple

Everything one origin's point-forecast panels need: the observed weekly infections `I_obs` (A × weeks)
over `n_before` weeks up to the origin and `n_after` past it, the six scenario fans (A × H × draws), and
the x positions. Asserts every fan exists and that `I_obs` reproduces the window's fit weeks
(`load_window_data`) and scored targets (`load_forecast_truth`) EXACTLY — the long series IS the truth the
forecasts were fitted to and scored against.
"""
function _point_forecast_inputs(fc_store, st::DataFrame, contact::AbstractString, origin::Date, cfg;
                                inf, grid, n_before::Int, n_after::Int)
    H = length(cfg.horizons)
    @assert n_after >= H "_point_forecast_inputs: n_after = $n_after does not cover the $H horizons"
    win   = WeeklyWindow(origin; n_fit = cfg.n_fit, smax = cfg.smax, horizons = cfg.horizons)
    weeks = collect((origin - Week(n_before - 1)):Week(1):(origin + Week(n_after)))
    I_obs, _ = weekly_infections(inf.df, weeks, grid)
    @assert all(>(0), sum(I_obs; dims = 1)) "_point_forecast_inputs: observed weeks outside the estimates around $origin"
    wd = load_window_data(win, inf.df, inf.tmap; grid = grid)
    wi = Dict(w => k for (k, w) in enumerate(weeks))
    fit_w = [w for w in win.fit_weeks if haskey(wi, w)]
    @assert I_obs[:, [wi[w] for w in fit_w]] == wd.I_mean[:, [findfirst(==(w), wd.weeks) for w in fit_w]] "_point_forecast_inputs: observed ≠ window fit weeks at $origin"
    @assert I_obs[:, [wi[w] for w in win.forecast_weeks]] == load_forecast_truth(win; grid = grid) "_point_forecast_inputs: observed ≠ scored truth at $origin"
    sname(sc) = only(@subset(st, :contact .== contact, :scenario .== sc).series)
    fans = Dict(sc => (k = (origin, sname(sc));
                       @assert haskey(fc_store, k) "_point_forecast_inputs: no fan for $(k)"; fc_store[k])
                for sc in SCENARIO_ORDER)
    x_obs  = week_mid.(weeks)
    months = [d for d in firstdayofmonth(first(x_obs)):Month(1):last(x_obs) if d >= first(x_obs)]
    return (; origin, win, weeks, I_obs, fans, x_obs, months, H, i0 = wi[origin],
              x_fc = vcat(week_mid(origin), week_mid.(win.forecast_weeks)))
end

"""
    _point_forecast_panel!(frame, d, lab, rows; title, ycap, xfmt, legend, small, xlabel) -> Plot

One panel: the observed infections summed over `rows` (black, squares), a dashed line at the origin, and
each scenario's median (no band) at h = 1..H as a DOTTED line of OPEN markers from the origin's observed
value, as in Fig 4 of Bracher et al. 2021 — colour and marker from `SCENARIO_STYLE`, so they match every
other §6 figure (frozen = same colour, diamond). The median is of the per-draw SUM over `rows`. y
capped at `ycap ×` the observed maximum, medians above the drawn limit COUNTED in the title. Month ticks
labelled with `xfmt` (`nothing` = unlabelled). `msw` is the marker rim width — GR draws strokes thicker
in a larger figure, so the by-age figure passes a thinner one. Appends the plotted numbers to `frame`.
"""
function _point_forecast_panel!(frame::DataFrame, d, lab::AbstractString, rows; title::AbstractString,
                                ycap::Real, xfmt, legend = false, small::Bool = false, xlabel = "",
                                msw::Real = (small ? 0.6 : 0.9))
    fmt(y) = abs(y) >= 1e6 ? string(round(y / 1e6; digits = 1), "M") : string(round(Int, y / 1e3), "k")
    H = d.H
    y_obs = vec(sum(d.I_obs[rows, :]; dims = 1))
    meds  = Dict(sc => (tot = dropdims(sum(d.fans[sc][rows, :, :]; dims = 1); dims = 1);
                        [_fmed(tot[h, :]) for h in 1:H]) for sc in SCENARIO_ORDER)
    ymax  = ycap * maximum(y_obs)                         # the cap; the axis shrinks to the data under it
    ytop  = 1.05 * min(ymax, maximum(vcat(y_obs, filter(isfinite, reduce(vcat, values(meds))))))
    xlab  = xfmt === nothing ? fill("", length(d.months)) : Dates.format.(d.months, xfmt)
    lw, ms = small ? (1.2, 3.5) : (1.8, 5.5)    # big markers, thin rims (`msw`): a thick rim hides circle vs diamond
    p = plot(; ylim = (0, ytop), yformatter = fmt, xticks = (Dates.value.(d.months), xlab), xlabel,
             titlefontsize = (small ? 8 : 10), tickfontsize = (small ? 6 : 8), legend,
             legendfontsize = 7)
    plot!(p, d.x_obs, y_obs; color = :black, lw, marker = :square, ms = (small ? 2.5 : 4.0), label = "observed")
    vline!(p, [week_mid(d.origin)]; color = :black, ls = :dash, lw = 1, label = "")
    append!(frame, DataFrame(origin = d.origin, age_group = lab, series = "observed", week = d.weeks,
                             value = y_obs, horizon = missing))
    n_off = 0
    for sc in SCENARIO_ORDER
        colr, _, mk = SCENARIO_STYLE[sc]
        plot!(p, d.x_fc, vcat(y_obs[d.i0], meds[sc]); color = colr, ls = :dot, lw, marker = mk, ms,
              markercolor = :white, markerstrokecolor = colr, markerstrokewidth = msw, label = sc)
        append!(frame, DataFrame(origin = d.origin, age_group = lab, series = sc,
                                 week = d.win.forecast_weeks, value = meds[sc], horizon = collect(1:H)))
        n_off += count(>(ytop), filter(isfinite, meds[sc]))
    end
    plot!(p; title = title * (n_off > 0 ? "  [$(n_off) median(s) off-scale]" : ""))
    return p
end

_point_frame() = DataFrame(origin = Date[], age_group = String[], series = String[], week = Date[],
                           value = Float64[], horizon = Union{Missing,Int}[])

"""
    plot_scenario_point_forecasts(fc_store, st, contact, origins, cfg; inf, grid, tags,
                                  n_before = 12, n_after = 6, ycap = 3.0) -> (; fig, frame)

Fig 4 of Bracher et al. 2021 (Nat Commun 12:5173; `ref/s41467-021-25207-0.pdf`) for the six scenarios of
ONE contact model, ALL AGES: one panel per origin — observed weekly infections, a dashed line at the
origin, and each scenario's POINT forecast (median, no band) at h = 1..H. See `_point_forecast_panel!`
and `_point_forecast_inputs` (which asserts the observed series against the scored truth). `frame` is
the plotted numbers, long. The age-stratified twin is `plot_scenario_point_forecasts_by_age`.
"""
function plot_scenario_point_forecasts(fc_store, st::DataFrame, contact::AbstractString,
                                       origins::Vector{Date}, cfg; inf, grid = cis_age_grid(),
                                       tags = Dict{Date,String}(), n_before::Int = 12,
                                       n_after::Int = 6, ycap::Real = 3.0)
    frame = _point_frame()
    panels = map(enumerate(origins)) do (j, origin)
        d   = _point_forecast_inputs(fc_store, st, contact, origin, cfg; inf, grid, n_before, n_after)
        tag = haskey(tags, origin) ? " ($(tags[origin]))" : ""
        _point_forecast_panel!(frame, d, "all", 1:grid.N; ycap, xfmt = "u yyyy", xlabel = "time",
                               legend = (j == 1 ? :topleft : false),
                               title = "($(Char('a' + j - 1))) Infections, England, forecasts from $(origin)$(tag)")
    end
    fig = plot(panels...; layout = (1, length(panels)), size = (560 * length(panels), 400),
               plot_title = "17j — $(contact): point forecasts (median) of the six scenarios, all ages",
               plot_titlefontsize = 10)
    return (; fig, frame)
end

"""
    plot_scenario_point_forecasts_by_age(fc_store, st, contact, origin, cfg; inf, grid, tag,
                                         n_before = 12, n_after = 6, ycap = 3.0) -> (; fig, frame)

The age-stratified point-forecast figure for ONE origin: a large all-ages panel on top, and below it a
2 × 4 grid of small panels — the seven CIS bins and a legend. Same panel as
`plot_scenario_point_forecasts`. Each panel's median is of the per-draw sum over ITS ages, so the age
panels do not add up to the total.
"""
function plot_scenario_point_forecasts_by_age(fc_store, st::DataFrame, contact::AbstractString,
                                              origin::Date, cfg; inf, grid = cis_age_grid(),
                                              tag::AbstractString = "", n_before::Int = 12,
                                              n_after::Int = 6, ycap::Real = 3.0)
    A = grid.N; nc = 4
    @assert A + 1 == 2nc "plot_scenario_point_forecasts_by_age: the 2 × $nc grid holds $(2nc - 1) age panels + a legend, not $A"
    frame = _point_frame()
    d = _point_forecast_inputs(fc_store, st, contact, origin, cfg; inf, grid, n_before, n_after)
    big = _point_forecast_panel!(frame, d, "all", 1:A; ycap, xfmt = "u yyyy", legend = false, msw = 0.5,
                                 title = "All ages — forecasts from $(origin)$(isempty(tag) ? "" : " ($tag)")")
    small = [_point_forecast_panel!(frame, d, grid.LAB[a], a:a; ycap, small = true, msw = 0.35,
                                    xfmt = (a > nc ? "u" : nothing), title = "age $(grid.LAB[a])")
             for a in 1:A]
    leg = plot(; framestyle = :none, grid = false, xticks = false, yticks = false, legend = :left,
               legendfontsize = 8, foreground_color_legend = nothing)
    plot!(leg, [NaN], [NaN]; color = :black, lw = 1.8, marker = :square, ms = 4.0, label = "observed")
    for sc in SCENARIO_ORDER
        colr, _, mk = SCENARIO_STYLE[sc]
        plot!(leg, [NaN], [NaN]; color = colr, ls = :dot, lw = 1.8, marker = mk, ms = 5.5,
              markercolor = :white, markerstrokecolor = colr, markerstrokewidth = 0.5, label = sc)
    end
    # the 2 × 4 cells spelled out: `grid(2, 4)` inside `@layout` would call this function's `grid` kwarg
    fig = plot(big, small..., leg; layout = @layout([a{0.5h}; [b c d e; f g h i]]), size = (1300, 1000),
               plot_title = "17j — $(contact): point forecasts (median) of the six scenarios, by age",
               plot_titlefontsize = 11)
    return (; fig, frame)
end

"""
    scenario_wis_by_horizon_until(by_dt_h, origin_max) -> DataFrame

`by_model_h`-shaped frame (`model, horizon, scale, wis, bias, interval_coverage_50/90, n_origins`)
restricted to origins ≤ `origin_max`: the mean over those origins of each per-origin score. Each origin carries the same 7 age bins, so over all
origins this reproduces `score_wis(…).by_model_h` exactly (the notebook asserts it).
"""
function scenario_wis_by_horizon_until(by_dt_h::DataFrame, origin_max::Date)
    d = @subset(by_dt_h, :forecast_date .<= origin_max)
    return sort!(combine(groupby(d, [:model, :horizon, :scale]), :wis => mean => :wis, :bias => mean => :bias,
                         :interval_coverage_50 => mean => :interval_coverage_50,
                         :interval_coverage_90 => mean => :interval_coverage_90,
                         :forecast_date => (x -> length(unique(x))) => :n_origins), [:model, :scale, :horizon])
end

"""
    scenario_rank_frame(wis_by_h, st; scale) -> DataFrame

One row per DISTINCT series × horizon (22 series: the 5 contact models × 4 contact scenarios + the 2
no-contact ones — the no-contact model has only "no contact" and "variant"): its mean WIS (`scale`),
`rank` within the horizon (1 = lowest WIS), the base model (`-variant` stripped) and the scenario.
`wis_by_h` is `by_model_h`-shaped (e.g. `scenario_wis_by_horizon_until`).
"""
function scenario_rank_frame(wis_by_h::DataFrame, st::DataFrame; scale::AbstractString = WIS_SCALE)
    meta = unique(select(st, :series, :label, :scenario))
    meta.model = replace.(meta.label, "-variant" => "")
    d = innerjoin(@subset(wis_by_h, :scale .== scale), select(meta, :series, :model, :scenario);
                  on = :model => :series, renamecols = "" => "_base")
    rename!(d, :model => :series, :model_base => :model, :scenario_base => :scenario)
    @assert nrow(unique(d, :series)) == nrow(meta) "scenario_rank_frame: series without a score"
    d = transform(groupby(d, :horizon), :wis => (w -> invperm(sortperm(w))) => :rank)
    return sort!(d, [:horizon, :rank])
end

"""
    plot_scenario_rank_by_horizon(rk, cfg; title_suffix) -> Plot

One panel per horizon: every series as a horizontal bar of its mean WIS, sorted best (rank 1) at the
top. Bar colour = scenario (`SCENARIO_STYLE`; the frozen scenarios are paler versions of their
conditional colour), label = `model · scenario`.
"""
function plot_scenario_rank_by_horizon(rk::DataFrame, cfg; title_suffix::AbstractString = "",
                                       scale::AbstractString = WIS_SCALE)
    short(m) = replace(m, "unweighted-" => "", "weighted-" => "", "neighbourhood" => "nbhd")
    panels = Plots.Plot[]
    for h in cfg.horizons
        s = sort(@subset(rk, :horizon .== h), :rank)
        n = nrow(s)
        y = n:-1:1                                        # rank 1 at the top
        p = plot(; title = "horizon $h", titlefontsize = 9, xlabel = "mean WIS ($scale)",
                 yticks = (collect(y), ["$(r). $(short(m)) · $(sc)" for (r, m, sc) in zip(s.rank, s.model, s.scenario)]),
                 ytickfontsize = 6, legend = false, xlim = (0, maximum(s.wis) * 1.12), ylim = (0.4, n + 0.6))
        for (yi, r) in zip(y, eachrow(s))
            colr = SCENARIO_STYLE[r.scenario][1]
            bar!(p, [yi], [r.wis]; orientation = :horizontal, color = colr, linecolor = colr,
                 fillalpha = (occursin("frozen", r.scenario) ? 0.45 : 0.95), bar_width = 0.75)
            annotate!(p, r.wis, yi, text(" " * string(round(r.wis; digits = 3)), 5, :left))
        end
        push!(panels, p)
    end
    nr, nc = panel_grid(length(panels))
    return plot(panels...; layout = (nr, nc), size = (880 * nc, 620 * nr), left_margin = 12Plots.mm,
                plot_title = "17j — ranking of all model × scenario series by mean WIS ($(scale) scale)$(title_suffix); colour = scenario, pale = frozen",
                plot_titlefontsize = 10)
end

"""
    plot_scenario_bias(by_h, st, cfg; scale, subtitle) -> Plot

scoringutils' `bias` (∈ [−1, 1]; > 0 ⇒ over-prediction, < 0 ⇒ under-prediction) by horizon, one panel per
contact model (layout from `panel_grid`), the six scenarios as lines; shared y-axis [−1, 1] with 0 dashed.
`by_h` is `by_model_h`-shaped (`score_wis(…).by_model_h`, or `scenario_wis_by_horizon_until` for a sub-period).
"""
function plot_scenario_bias(by_h::DataFrame, st::DataFrame, cfg; scale::AbstractString = WIS_SCALE,
                            subtitle::AbstractString = "")
    j = innerjoin(@subset(by_h, :scale .== scale), st; on = :model => :series)
    panels = Plots.Plot[]
    for (k, c) in enumerate(unique(st.contact))
        p = plot(; title = c, titlefontsize = 8, xlabel = "horizon (weeks)", ylabel = "bias",
                 ylim = (-1, 1), xticks = collect(cfg.horizons), legend = (k == 1 ? :bottomleft : false),
                 legendfontsize = 5)
        hline!(p, [0.0]; color = :gray, ls = :dot, label = "")
        for sc in SCENARIO_ORDER
            s = sort(@subset(j, :contact .== c, :scenario .== sc), :horizon)
            colr, ls, mk = SCENARIO_STYLE[sc]
            plot!(p, s.horizon, s.bias; color = colr, ls = ls, marker = mk, ms = 3, msc = colr, lw = 1.8, label = sc)
        end
        push!(panels, p)
    end
    nr, nc = panel_grid(length(panels))
    return plot(panels...; layout = (nr, nc), size = (400 * nc, 340 * nr),
                plot_title = "17j — bias by horizon, six scenarios per contact model (> 0 over-, < 0 under-prediction)$(subtitle)",
                plot_titlefontsize = 10)
end
