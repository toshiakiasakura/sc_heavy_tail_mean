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

# ---- age-interval parsers (verbatim from 7j §2) ----
_toint7j(s) = (ismissing(s) || s == "NA") ? nothing : tryparse(Int, String(s))

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
                               infection_start=Date(2020,8,2), step_weeks=1)

Rolling Sunday-start forecast origins the current data support. Lower bound: the
12-week fit/lag window (`origin−11wk … origin`) must lie within the infection series
(`infection_start` = first full inc2prev week). Upper bound: the contact-updated
iterate needs contact data out to `origin + max horizon` weeks, so the last
origin is `last_contact_week − max(horizons)`. `craw` is the raw contact table
(from `load_raw_contact_inputs`); if `nothing` it is read.
"""
function available_forecast_origins(cfg::FrameworkConfig; grid = cis_age_grid(),
                                    craw = nothing,
                                    infection_start::Date = Date(2020, 8, 2),
                                    step_weeks::Int = 1)
    craw === nothing && (craw = load_raw_contact_inputs().craw)
    cweeks = week_start.(collect(skipmissing(craw.date)))
    last_contact_week = maximum(cweeks)
    lookback = cfg.n_fit - 1 + cfg.smax                         # all_weeks[1] = origin − lookback
    tmin = week_start(infection_start) + Day(7 * lookback)
    tmax = last_contact_week - Day(7 * maximum(cfg.horizons))
    return collect(tmin:Day(7 * step_weeks):tmax)
end

"""
    prepare_degree_data(win, cfg; grid, setting=:all, arrow_path=_ARROW_PATH)

Build `AgePairData` for the 12-week span `win.all_weeks`. `setting ∈ (:all,:home,
:nonhome)`. Ambiguous/missing ages are assigned by a single seeded
population-weighted draw (`MersenneTwister(cfg.seed)`), exactly as in 7j.

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
    weeks = win.all_weeks
    wkset = Dict(w => k for (k, w) in enumerate(weeks))
    A = grid.N; T = length(weeks)
    dmin = minimum(weeks); dmax = maximum(weeks) + Day(6)

    rng = MersenneTwister(cfg.seed)
    AGE_MIN = grid.LO[1]
    overlapping(a, b) = [j for j in 1:A if a <= grid.HI[j] && b >= grid.LO[j]]
    function assign_bin(a, b)
        b < AGE_MIN && return nothing
        cand = overlapping(a, b)
        isempty(cand)     && return 1
        length(cand) == 1 && return cand[1]
        return sample(rng, cand, Weights(grid.POP[cand]))
    end

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
