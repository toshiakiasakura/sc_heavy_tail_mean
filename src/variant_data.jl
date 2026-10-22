# variant_data.jl — COG-UK England variant composition for the VARIANT-SPECIFIC renewal (`VariantNGM`,
# inst/3 §12; 2026-10-21). Included by forecast_utils.jl after framework.jl / infection_data.jl.
#
# Source: `dt_cog_uk/cog_uk_england_weekly.csv` (`dt_cog_uk/fetch_cog_uk_counts.py`) — England genomes per
# MONDAY sample-date week by constellation group. No source gives infections by variant OR by age, so the
# England-wide genome share is applied to every age bin (`I_v,a(t) = p_v(t)·I_a(t)`).
#
# ⚠ WEEK ALIGNMENT: a framework week is Sunday-start; it is read from the COG-UK Monday week starting the
#   NEXT day (6 of its 7 days). `cog_week` is the one place that rule is written down.
# ⚠ The set of categories is chosen ONCE, at the origin (`variant_set`), and is used for every horizon and
#   every week of the window: no variant data after t₀ enters a fit or a forecast (user, 2026-10-21).
# ⚠ HYSTERESIS (user, 2026-10-21): inclusion by the 4-week-n / origin-share rule, exit only after all
#   n_fit fit weeks sit below `variant_drop_share` — so the set at an origin depends on the history.

const COG_VARIANT_PATH = joinpath(@__DIR__, "..", "dt_cog_uk", "cog_uk_england_weekly.csv")
"""COG-UK constellation groups, in the order categories are listed (the catch-all first)."""
const COG_GROUPS = ["Other", "Alpha", "Delta", "Omicron BA.1", "Omicron BA.2", "Omicron other"]
"""Name of the pooled category: COG-UK "Other" plus every group failing the inclusion rule."""
const VARIANT_RESIDUAL = "residual"

"""
    load_cog_variant_counts(path = COG_VARIANT_PATH) -> (; weeks, groups, n)

Weekly England genome counts, wide: `weeks` the contiguous Monday weeks of the file, `groups` =
`COG_GROUPS`, `n[w, g]` genomes (a missing (week, group) row is 0 — the CSV is sparse). Asserts every
week is a Monday, every group is known and no week is absent altogether.
"""
function load_cog_variant_counts(path::AbstractString = COG_VARIANT_PATH)
    df = CSV.read(path, DataFrame)
    @assert all(c -> c in names(df), ["week_start", "group", "n_genomes"]) "load_cog_variant_counts: unexpected columns $(names(df))"
    wk = Date.(df.week_start)
    @assert all(dayofweek.(wk) .== 1) "load_cog_variant_counts: a week_start is not a Monday"
    @assert issubset(Set(df.group), Set(COG_GROUPS)) "load_cog_variant_counts: unknown groups $(setdiff(Set(df.group), Set(COG_GROUPS)))"
    weeks = collect(minimum(wk):Week(1):maximum(wk))
    @assert Set(wk) == Set(weeks) "load_cog_variant_counts: weeks with no genome at all: $(setdiff(Set(weeks), Set(wk)))"
    wi = Dict(w => i for (i, w) in enumerate(weeks)); gi = Dict(g => j for (j, g) in enumerate(COG_GROUPS))
    n = zeros(Int, length(weeks), length(COG_GROUPS))
    for r in eachrow(df)
        n[wi[Date(r.week_start)], gi[r.group]] += r.n_genomes
    end
    return (; weeks, groups = copy(COG_GROUPS), n)
end

"""`cog_week(d)` — the COG-UK Monday week a framework (Sunday-start) week `d` is read from: `d + 1 day`."""
function cog_week(d::Date)
    @assert dayofweek(d) == 7 "cog_week: $d is not a Sunday week start"
    return d + Day(1)
end

_cog_row(cc, d::Date) = begin
    w = cog_week(d)
    i = searchsortedfirst(cc.weeks, w)
    @assert i <= length(cc.weeks) && cc.weeks[i] == w "COG-UK counts have no week $w (framework week $d)"
    i
end

variant_rule(cfg::FrameworkConfig) =
    (; n_min = cfg.variant_n_min, n_weeks = cfg.variant_n_weeks, share_min = cfg.variant_share_min,
       drop_share = cfg.variant_drop_share, drop_weeks = cfg.n_fit)

"""
    _variant_states(cc, rule) -> (; on, resid_on)

The HYSTERESIS state machine over every COG-UK week `i` (each read as a forecast origin), scanned from
the first week of the file so that the state at an origin depends only on data up to that origin:

- a named group (anything but "Other") is switched ON at `i` when it passes the INCLUSION rule there
  (genomes over weeks `i−n_weeks+1 … i` ≥ `n_min`, or week-`i` share ≥ `share_min`);
- once on, it stays on until an origin at which its share was < `drop_share` in EVERY one of the
  `drop_weeks` (= `n_fit`) weeks `i−drop_weeks+1 … i` — the origin's fit weeks;
- after a drop it re-enters only by passing the inclusion rule again.

The RESIDUAL ("Other" + every named group that is off at `i`, pooled with week-`i` membership) runs
the same machine, starting ON. `on[i, j]` is per group column (the "Other" column is unused).
"""
function _variant_states(cc, rule::NamedTuple)
    W, G = size(cc.n)
    tot = vec(sum(cc.n; dims = 2))
    share(rows, cols) = [sum(cc.n[r, cols]) / tot[r] for r in rows]
    passes(i, cols) = i >= rule.n_weeks &&
        (sum(cc.n[(i - rule.n_weeks + 1):i, cols]) >= rule.n_min || share(i:i, cols)[1] >= rule.share_min)
    below(i, cols) = i >= rule.drop_weeks &&
        all(share((i - rule.drop_weeks + 1):i, cols) .< rule.drop_share)
    jo = findfirst(==("Other"), cc.groups)
    on = falses(W, G); resid_on = falses(W)
    prev = falses(G); prev_r = true
    for i in 1:W
        for j in 1:G
            j == jo && continue
            on[i, j] = passes(i, [j]) || (prev[j] && !below(i, [j]))
        end
        rc = [j for j in 1:G if j == jo || !on[i, j]]
        resid_on[i] = passes(i, rc) || (prev_r && !below(i, rc))
        prev = on[i, :]; prev_r = resid_on[i]
    end
    return (; on, resid_on)
end

"""
    variant_set(cc, origin, rule) -> (; names, members, n_window, share0)

The categories at forecast origin `origin` (a Sunday week start), from `_variant_states`: every named
COG-UK group whose state is ON gets its own renewal (inclusion: 4-week genomes ≥ `n_min` OR origin-week
share ≥ `share_min`; drop, with HYSTERESIS, only after `drop_weeks` = `n_fit` fit weeks all below
`drop_share`). "Other" + the groups that are off pool into `VARIANT_RESIDUAL`, which keeps its own
renewal while its own state is on, and otherwise merges into the included group with the largest
origin-week share. Categories are returned residual first, then in `COG_GROUPS` order.
`n_window` (genomes over the inclusion window) and `share0` are per category.
"""
function variant_set(cc, origin::Date, rule::NamedTuple)
    i0 = _cog_row(cc, origin)
    @assert i0 - max(rule.n_weeks, rule.drop_weeks) + 1 >= 1 "variant_set: COG-UK counts start too late for origin $origin"
    st = _variant_states(cc, rule)
    tot0 = sum(cc.n[i0, :])
    @assert tot0 > 0 "variant_set: no genomes in the origin week of $origin"
    named = [g for (j, g) in enumerate(cc.groups) if g != "Other" && st.on[i0, j]]
    resid = [g for g in cc.groups if !(g in named)]
    names = String[]; members = Vector{String}[]
    if isempty(named) || st.resid_on[i0]
        push!(names, VARIANT_RESIDUAL); push!(members, resid)
        for g in named; push!(names, g); push!(members, [g]); end
    else                                       # residual off: fold it into the dominant included group
        jn = [findfirst(==(g), cc.groups) for g in named]
        dom = named[argmax(cc.n[i0, jn])]
        for g in named
            push!(names, g); push!(members, g == dom ? vcat([g], resid) : [g])
        end
    end
    win = (i0 - rule.n_weeks + 1):i0
    cat_n(rows, mem) = sum(cc.n[rows, [findfirst(==(g), cc.groups) for g in mem]])
    n_window = [cat_n(win, m) for m in members]
    share0   = [cat_n(i0:i0, m) / tot0 for m in members]
    return (; names, members, n_window, share0)
end

"""
    variant_split(win, cc, cfg) -> VariantSplit

The variant composition of window `win` over `win.all_weeks`, with the categories chosen at
`win.origin` by `variant_set`. Per week: genome share per category (sums to 1), genomes sequenced, and
the binomial sampling variance of each share, `p̃(1−p̃)/n` with `p̃ = (n_v+½)/(n+1)` (> 0 even when
`n_v = 0`, which the split likelihood needs).
"""
function variant_split(win::WeeklyWindow, cc, cfg::FrameworkConfig)
    rule = variant_rule(cfg)
    vs = variant_set(cc, win.origin, rule)
    weeks = collect(win.all_weeks)
    V, T = length(vs.names), length(weeks)
    cols = [[findfirst(==(g), cc.groups) for g in m] for m in vs.members]
    @assert sort(reduce(vcat, cols)) == collect(1:length(cc.groups)) "variant_split: categories do not partition the COG-UK groups"
    share = zeros(V, T); se2 = zeros(V, T); n = zeros(Int, T)
    for (t, d) in enumerate(weeks)
        i = _cog_row(cc, d)
        n[t] = sum(cc.n[i, :])
        @assert n[t] > 0 "variant_split: no genomes in the COG-UK week of $d"
        for v in 1:V
            nv = sum(cc.n[i, cols[v]])
            share[v, t] = nv / n[t]
            pt = (nv + 0.5) / (n[t] + 1)
            se2[v, t] = pt * (1 - pt) / n[t]
        end
    end
    @assert all(abs.(sum(share; dims = 1) .- 1) .< 1e-12) "variant_split: shares do not sum to 1"
    return VariantSplit(vs.names, vs.members, weeks, share, n, se2, win.origin, rule)
end

"""`with_variants(wd, vs)` — `wd` with its `variant` field set; asserts the two cover the same weeks."""
function with_variants(wd::WindowData, vs::VariantSplit)
    @assert vs.weeks == wd.weeks "with_variants: variant weeks $(first(vs.weeks))…$(last(vs.weeks)) ≠ window weeks $(first(wd.weeks))…$(last(wd.weeks))"
    return WindowData(wd.weeks, wd.A, wd.I_mean, wd.I_sd, wd.antibody, wd.antibody_fc,
                      wd.pop, wd.prop, wd.labels, vs)
end

"""`variant_set_table(wins, cc, cfg)` — one row per origin: the categories, the residual's members and
each category's origin-week share and window genome count (for reporting the rule's decisions)."""
function variant_set_table(wins, cc, cfg::FrameworkConfig)
    rule = variant_rule(cfg)
    rows = map(wins) do w
        vs = variant_set(cc, w.origin, rule)
        (; origin = w.origin, n_categories = length(vs.names),
           categories = join(vs.names, "; "),
           members = join([n * "=" * join(m, "+") for (n, m) in zip(vs.names, vs.members)], "; "),
           share_t0 = join([string(n, " ", round(100s; digits = 2), "%") for (n, s) in zip(vs.names, vs.share0)], "; "),
           n_window = join([string(n, " ", k) for (n, k) in zip(vs.names, vs.n_window)], "; "))
    end
    return DataFrame(rows)
end
