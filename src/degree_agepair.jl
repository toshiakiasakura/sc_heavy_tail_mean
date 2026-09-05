# degree_agepair.jl — assemble per-cell weekly contact-degree data over the 7 CIS
# age bins (contactor/participant bin i × contactee bin j), reusing the age-pair
# binning of src/7j_weekly_age_pair.ipynb §2–§3.
#
# For each (week t, participant bin i, contactee bin j) we build:
#   - dd_count : DegreeDist of integer contact counts INCLUDING zeros (NegBin path).
#   - pos_weight : positive duration-weighted degrees (hurdle-Weibull path).
#   - p0 : empirical zero probability; n : sampled participant-days with part_bin i.
# Zeros come from the participant-day ROSTER (df_part), not the contact table.

const _ARROW_PATH = joinpath(@__DIR__, "..", "dt_comix_no_public", "contacts_uk.arrow")

# ---- age-interval parsers (from 7j §2; robust to Int-typed or String cells) ----
# `:cnt_age_est_min`/`:cnt_age_est_max` come back from contacts_uk.arrow as
# `Union{Missing,Int64}`, so handle numeric cells directly — `String(::Int64)` throws.
_toint7j(s) = ismissing(s) ? nothing :
              s isa Real   ? round(Int, s) :
              s == "NA"    ? nothing : tryparse(Int, String(s))

"Parse participant age-group string \"lo-hi\" → (lo,hi); NA/unparseable → (0,120)."
function parse_age_interval(s)
    (ismissing(s) || s == "NA") && return (0, 120)
    parts = split(String(s), "-")
    length(parts) == 2 || return (0, 120)
    a = tryparse(Int, parts[1]); b = tryparse(Int, parts[2])
    (a === nothing || b === nothing) ? (0, 120) : (a, b)
end

"Contactee interval from (est_min, est_max); NA/unparseable → (0,120)."
function interval_from_minmax(mn, mx)
    a = _toint7j(mn); b = _toint7j(mx)
    (a === nothing || b === nothing) ? (0, 120) : (a, b)
end

"Group (\"mass\") vs individually-reported contact, from `:cnt_mass` in
contacts_uk.arrow (values \"mass\"/\"individual\"). Missing → not group."
_is_group_contact(v) = !ismissing(v) && (String(v) == "mass")

"""
    assign_age_bin(a, b, grid, rng) -> Int | nothing

Assign a reported age interval `[a,b]` to one of `grid`'s CIS bins. An interval overlapping
several bins is resolved by a **single population-weighted draw** from `rng`; one lying entirely
below the grid's first bin returns `nothing` (the caller drops it), and one overlapping none but
sitting above the grid falls back to bin 1.

⚠ **This is the ONLY place the rule is written down, deliberately.** It was a closure inside
`prepare_degree_data` until 14j needed the same mapping to split participants into the model's
child/adult blocks (`block_of`, joint_model.jl). A second copy would be exactly the failure this
repo keeps hitting — a rule fixed in one mirror and missed in another — and it would be silent,
because the only intervals whose bin is ambiguous are the ones straddling a boundary: CoMix's
`12-17` participant group spans bin 2 (`11-15`) and bin 3 (`16-24`), i.e. precisely the 15/16
child/adult cut.

⚠ **`rng` IS CONSUMED, so CALL ORDER IS PART OF THE CONTRACT.** `prepare_degree_data` draws for
participants first and contacts second from ONE `MersenneTwister(cfg.seed)`; reordering or adding
a call changes every downstream bin assignment and hence every fitted chain's data.
"""
function assign_age_bin(a, b, grid, rng)
    b < grid.LO[1] && return nothing
    cand = [j for j in 1:grid.N if a <= grid.HI[j] && b >= grid.LO[j]]
    isempty(cand)     && return 1
    length(cand) == 1 && return cand[1]
    return sample(rng, cand, Weights(grid.POP[cand]))
end

"""
    load_raw_contact_inputs(; arrow_path=_ARROW_PATH)

Read the CoMix-UK participant roster and contact table **once** (the two full reads
`prepare_degree_data` would otherwise repeat per window). Returns `(; df_part, craw)`
to pass straight into `prepare_degree_data(...; df_part_raw=…, craw_raw=…)`. Also
exposes the contact date span via `extrema(skipmissing(craw.date))`.
"""
function load_raw_contact_inputs(; arrow_path::AbstractString = _ARROW_PATH)
    _, df_part = read_comix_uk_contact_raw()
    craw = read_arrow_df(arrow_path;
        cols = [:part_wave_uid, :date, :cnt_home, :cnt_minutes_max, :cnt_total_time,
                :cnt_mass, :cnt_age_est_min, :cnt_age_est_max])
    return (; df_part, craw)
end

"""
    available_forecast_origins(cfg; grid=cis_age_grid(), craw=nothing,
                               infection_start=Date(2020,8,2), step_weeks=1,
                               origin_max=nothing)

Rolling Sunday-start forecast origins the current data support. Lower bound: the
12-week fit/lag window (`origin−11wk … origin`) must lie within the INFECTION series
(`infection_start` = first full inc2prev week) — the infection window is still
`n_fit + smax` weeks even though the CONTACT window is only `n_fit` (`-w8`,
2026-08-09: `prepare_degree_data` spans `win.fit_weeks`, so contacts are needed
only back to `origin − n_fit + 1 + min(horizons)`, a weaker bound that never binds).
Upper bound: the contact-updated iterate needs contact data out to
`origin + max horizon` weeks, so the last origin is
`last_contact_week − max(horizons)`. `craw` is the raw contact table
(from `load_raw_contact_inputs`); if `nothing` it is read.

`origin_max` caps the last origin at `week_start(origin_max)` (`nothing` = data-derived
bound only). The data-derived bound alone is **too generous** and callers should cap it:
CoMix's main panel stops 2022-03-02, but a stray 2022-11-16…2022-11-28 block drags
`last_contact_week` — and hence `tmax` — out to 2022-10-30, so origins from ~2022-03-06
roll through a window with no contact data. inc2prev is a second, uncaught limit: its
`infections`/`gen_dab` end 2022-03-26 and `weekly_infections`/`weekly_antibody`
(`infection_data.jl`) skip unmatched weeks into pre-zeroed arrays, so later origins are
silently zero-filled rather than erroring. 8j/9j pass `origin_max=Date(2021,12,31)`
(⇒ last origin 2021-12-26).
"""
function available_forecast_origins(cfg::FrameworkConfig; grid = cis_age_grid(),
                                    craw = nothing,
                                    infection_start::Date = Date(2020, 8, 2),
                                    step_weeks::Int = 1,
                                    origin_max::Union{Nothing,Date} = nothing)
    craw === nothing && (craw = load_raw_contact_inputs().craw)
    cweeks = week_start.(collect(skipmissing(craw.date)))
    last_contact_week = maximum(cweeks)
    lookback = cfg.n_fit - 1 + cfg.smax                         # all_weeks[1] = origin − lookback
    tmin = week_start(infection_start) + Day(7 * lookback)
    tmax = last_contact_week - Day(7 * maximum(cfg.horizons))
    origin_max !== nothing && (tmax = min(tmax, week_start(origin_max)))
    return collect(tmin:Day(7 * step_weeks):tmax)
end

"""
    degree_window(origin, h, cfg) -> WeeklyWindow

The `WeeklyWindow` to hand `prepare_degree_data` for horizon `h`: the ORIGIN's window with its
horizons truncated to `1:h`, so that `fit_weeks ++ forecast_weeks` = `[t₀−n_fit+1 … t₀+h]`.
`h = 0` (`horizons = 1:0`, an empty range) gives the bare `n_fit` fit weeks — what a diagnostic
wants when it needs observed contacts at the origin itself with no horizon tail.

⚠ **USE THIS RATHER THAN `WeeklyWindow(origin + Day(7h))`.** The shifted-origin form was the
convention until `-w8h` (2026-08-09) and it is now WRONG in a way nothing detects: it would span
`[t₀+h−n_fit+1 … t₀+h+4]`, which has the right *length* at `h = 4` and the wrong *dates* at every
horizon. `stage2_inputs` asserts `apd.weeks[1:n_fit] == win0.fit_weeks` for exactly this reason.
"""
degree_window(origin::Date, h::Integer, cfg::FrameworkConfig) =
    WeeklyWindow(origin; n_fit = cfg.n_fit, smax = cfg.smax, horizons = 1:h)

"""
    prepare_degree_data(win, cfg; grid, setting=:all, arrow_path=_ARROW_PATH)

Build `AgePairData` for the span **`win.fit_weeks ++ win.forecast_weeks`** — i.e.
`[origin−n_fit+1 … origin+h]`, length `n_fit + h`, where `h` is however many horizons `win` was
built with (`degree_window(origin, h, cfg)` above is how to build it).
`setting ∈ (:all,:home,:nonhome)`. Ambiguous/missing ages are assigned by a single seeded
population-weighted draw (`MersenneTwister(cfg.seed)`), exactly as in 7j.

⚠ **NOT `all_weeks` (`-w8h`, 2026-08-09, user request).** This used to span all 12 weeks of
`win.all_weeks` — the `smax` renewal-lag weeks as well as the `n_fit` fitting weeks — and the first
`smax` of them were fitted and then **discarded**: `model_transmission`'s likelihood runs
`for t in (smax+1):Tn`, so only the fitting weeks' `C*` ever reaches the renewal. The lag weeks
exist to supply `I(t−s)` history, which is an INFECTION-side need; the contact block has no use for
them. Stage-1 latents go from a flat 389/977 to `5 + 32(n_fit+h)` = 293/325/357/389 (NegBin) and
`5 + 81(n_fit+h)` = 734/815/896/977 (hurdle-Weibull).

The window is **anchored at the origin and extended to the horizon**, not slid forward with it: a
brief intermediate version (`-w8`, same day) used the sliding `[t₀−n_fit+1+h … t₀+h]`, which is
exactly the `n_fit` weeks the renewal consumes but leaves the earliest fit weeks in no chain at all
— visible as 10j §2c losing the first half of its μ timeline. The last `h` columns beyond what the
renewal reads are fitted-but-unused; that cost is accepted deliberately.

⚠ Consequence for callers: **the contact window and the infection window have DIFFERENT LENGTHS**
(`n_fit + h` vs `n_fit + smax`) and cannot be indexed by a shared `t`. Everything pairing a per-week
`C*` with a per-week infection quantity must offset — `Cstar_weeks[t − smax + h]` — and both sites
derive `h` as `length(Cstar_weeks) − n_fit` rather than being passed it: see `model_transmission`
and `fit_window_infection_draws` (`10j_viz_utils.jl`). `load_window_data` is unchanged and still
spans `win.all_weeks`.

Pass `df_part_raw`/`craw_raw` (from `load_raw_contact_inputs()`) to reuse a single
read of the CoMix participant roster and contact table across many windows — this
avoids re-reading/re-joining the full Arrow on every call (matters for a rolling
multi-origin run). Both are filtered to the window's dates internally, so the cached
frames are never mutated.
"""
function prepare_degree_data(win::WeeklyWindow, cfg::FrameworkConfig;
                             grid = cis_age_grid(), setting::Symbol = :all,
                             arrow_path::AbstractString = _ARROW_PATH,
                             df_part_raw = nothing, craw_raw = nothing)
    weeks = vcat(win.fit_weeks, win.forecast_weeks)   # `-w8h`: [t₀−n_fit+1 … t₀+h] (see docstring)
    wkset = Dict(w => k for (k, w) in enumerate(weeks))
    A = grid.N; T = length(weeks)
    dmin = minimum(weeks); dmax = maximum(weeks) + Day(6)

    rng = MersenneTwister(cfg.seed)
    # `assign_age_bin` (above) is the shared rule; this closure only pins `grid`/`rng`. The two
    # call sites below (participants, then contacts) draw from this ONE stream in that order —
    # do not reorder them, or every ambiguous age's bin changes.
    assign_bin(a, b) = assign_age_bin(a, b, grid, rng)

    # participant-day roster with a drawn participant bin (reuse cached read if given)
    df_part = df_part_raw === nothing ? read_comix_uk_contact_raw()[2] : df_part_raw
    df_part = @subset(df_part, (:date .>= dmin) .& (:date .<= dmax))
    piv = parse_age_interval.(df_part.part_age)
    df_part[!, :part_bin] = [assign_bin(a, b) for (a, b) in piv]
    df_part = @subset(df_part, .!isnothing.(:part_bin))
    df_part[!, :part_bin] = Int.(df_part.part_bin)
    df_part[!, :wk] = week_start.(df_part.date)
    df_part = @subset(df_part, in.(:wk, Ref(Set(weeks))))
    part_lookup = unique(@select(df_part, :part_id_d, :date, :part_bin, :wk))

    n_roster = zeros(Int, T, A)
    for r in eachrow(part_lookup)
        n_roster[wkset[r.wk], r.part_bin] += 1
    end

    # contact table with contactee bin, duration and setting (reuse cached read if given)
    craw = craw_raw === nothing ?
        read_arrow_df(arrow_path;
            cols = [:part_wave_uid, :date, :cnt_home, :cnt_minutes_max, :cnt_total_time,
                    :cnt_mass, :cnt_age_est_min, :cnt_age_est_max]) :
        craw_raw
    craw = @subset(craw, (:date .>= dmin) .& (:date .<= dmax))
    craw = @select(craw,
        :part_id_d      = :part_wave_uid,
        :date,
        :cnt_home,
        :cnt_mass,
        :duration_multi = _uk_duration_multi.(:cnt_minutes_max, :cnt_total_time),
        :cnt_age_est_min, :cnt_age_est_max)
    standardise_cnt_home_values!(craw)
    dfA = innerjoin(craw, part_lookup, on = [:part_id_d, :date])
    civ = [interval_from_minmax(mn, mx) for (mn, mx) in zip(dfA.cnt_age_est_min, dfA.cnt_age_est_max)]
    dfA[!, :cnt_bin] = [assign_bin(a, b) for (a, b) in civ]
    dfA = @subset(dfA, .!isnothing.(:cnt_bin))
    dfA[!, :cnt_bin] = Int.(dfA.cnt_bin)
    if setting === :home
        dfA = @subset(dfA, :cnt_home .== "true")
    elseif setting === :nonhome
        dfA = @subset(dfA, :cnt_home .== "false")
    end
    # Group ("mass") contacts have no recorded duration; assign the explicit,
    # later-estimable group weight `w_dur_group` (inst/1e D4) rather than letting them
    # fall through the NA→<5min path. Individually-reported contacts use the duration bin.
    dfA[!, :w] = ifelse.(_is_group_contact.(dfA.cnt_mass),
                         cfg.w_dur_group,
                         duration_weight.(dfA.duration_multi, cfg.d_max))

    # per participant-day × week × cell: contact count and summed weight
    g = combine(groupby(dfA, [:part_id_d, :date, :wk, :part_bin, :cnt_bin]),
                nrow => :cnt, :w => sum => :wsum)
    gidx = Dict{Tuple{Int,Int,Int},DataFrame}()
    for sub in groupby(g, [:wk, :part_bin, :cnt_bin])
        gidx[(wkset[sub.wk[1]], sub.part_bin[1], sub.cnt_bin[1])] = DataFrame(sub)
    end

    dd_count   = Array{DegreeDist,3}(undef, T, A, A)
    pos_weight = Array{WeightedDegreeHist,3}(undef, T, A, A)
    p0   = zeros(T, A, A)
    nn   = zeros(Int, T, A, A)
    emean = zeros(T, A, A)
    ecv2  = zeros(T, A, A)

    for t in 1:T, i in 1:A, j in 1:A
        nrost = n_roster[t, i]
        nn[t, i, j] = nrost
        sub = get(gidx, (t, i, j), nothing)
        if sub === nothing || nrost == 0
            dd_count[t, i, j]   = DegreeDist([0], [max(nrost, 0)], true)
            pos_weight[t, i, j] = WeightedDegreeHist(Float64[], Int64[])
            p0[t, i, j] = 1.0
            continue
        end
        counts_pos = Int.(sub.cnt)
        n_pos  = length(counts_pos)
        n_zero = nrost - n_pos
        cm = countmap(counts_pos)
        xs = collect(keys(cm)); ys = collect(values(cm))
        if n_zero > 0
            push!(xs, 0); push!(ys, n_zero)
        end
        ord = sortperm(xs)
        dd_count[t, i, j]   = DegreeDist(Int.(xs[ord]), Int.(ys[ord]), true)
        pos_weight[t, i, j] = WeightedDegreeHist(Float64.(sub.wsum))
        p0[t, i, j] = n_zero / nrost
        allcnt = vcat(counts_pos, zeros(Int, n_zero))
        m = mean(allcnt); v = var(allcnt; corrected = false)
        emean[t, i, j] = m
        ecv2[t, i, j]  = m > 0 ? v / m^2 : 0.0
    end

    return AgePairData(setting, weeks, A, dd_count, pos_weight, p0, nn, emean, ecv2)
end

"""
    pool_over_time(apd)

Collapse the per-week cells to one pooled cell per (i,j) — used when
`cfg.constant_contacts = true`. Returns `(; dd_count, pos_weight, p0, n, emean, ecv2)`
as `A×A` arrays.
"""
function pool_over_time(apd::AgePairData)
    A = apd.A
    dd  = Array{DegreeDist,2}(undef, A, A)
    pw  = Array{WeightedDegreeHist,2}(undef, A, A)
    p0  = zeros(A, A); nn = zeros(Int, A, A)
    em  = zeros(A, A); ec = zeros(A, A)
    for i in 1:A, j in 1:A
        merged = merge_dd(vcat([dd_to_df(apd.dd_count[t, i, j]) for t in eachindex(apd.weeks)]...))
        dd[i, j] = merged
        pw[i, j] = merge_whist([apd.pos_weight[t, i, j] for t in eachindex(apd.weeks)])
        ntot = sum(apd.n[t, i, j] for t in eachindex(apd.weeks))
        nn[i, j] = ntot
        nzero = sum(dd[i, j].y[dd[i, j].x .== 0])
        p0[i, j] = ntot > 0 ? nzero / ntot : 1.0
        em[i, j] = ntot > 0 ? sum(dd[i, j].x .* dd[i, j].y) / ntot : 0.0
        # CV^2 of counts incl zeros
        if ntot > 0 && em[i, j] > 0
            m2 = sum((dd[i, j].x .^ 2) .* dd[i, j].y) / ntot
            ec[i, j] = (m2 - em[i, j]^2) / em[i, j]^2
        end
    end
    return (; dd_count = dd, pos_weight = pw, p0 = p0, n = nn, emean = em, ecv2 = ec)
end
