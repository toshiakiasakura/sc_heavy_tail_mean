# infection_data.jl — the framework's infection-side inputs on the Monday-start weekly grid.
#
# OUTCOME (2026-10-22, user request): ONS CIS positivity by age (`inc2prev/data-processed/cis_age.csv`,
# England, daily) × population — `weekly_cis_positives`. It REPLACED inc2prev's modelled infections
# as `WindowData.I_mean` (renewal history, Stage-2 target, scoring truth). It is a PREVALENCE used
# as-is where the renewal expects incidence; see `weekly_cis_positives`. Everything below about
# `estimates_age.csv` now concerns ANTIBODY only (all zero, F pinned) and `weekly_infections`, which
# the 14j/16j audits still read.
#
# SOURCE: `estimates_age.csv` since 2026-09-16 (user request — errors found in the use of the
# `_ab` variant). The SCHEMA is identical in both files: `name` selects the quantity, `variable`
# is the age group, `mean`/`sd` the daily estimate, `population` the age-group size, `t_index` a
# 1-based day index. What differs is the ROW SET, in two ways that both matter:
#
#   1. `estimates_age.csv` carries NO ANTIBODY AT ALL — no `gen_dab`, no `est_ab`. Those rows
#      exist only in the `_ab` variants. `weekly_antibody` therefore returns all-zero against
#      this source; see its docstring for why that is harmless TODAY and when it would stop being.
#   2. The `infections` estimates THEMSELVES differ. The `_ab` file is a joint fit to prevalence
#      AND antibody data, so its infection posterior is not the same quantity. Measured over
#      2020-10-01…2022-01-31: median 8.6% relative difference in the mean, p90 76%. Switching
#      files is a REFIT, not a relabelling.
#
# Both files start 2020-07-31; `estimates_age.csv` runs to 2023-01-31 against the `_ab` file's
# 2022-03-26. `infections` carries `date` in both, and the `t_index → Date` map built from those
# rows is retained for `gen_dab`, which does NOT carry `date` wherever it exists at all (matches
# CovidAgeGroupForecast run_forecasts_for_infections.R:75).
#
# ⚠ CACHE. Changing this path invalidates every `8j_s2_*` and the four derived `9j_assembly_*`/
# `9j_rt_*`/`9j_relrt_*`/`9j_obsrt_*` caches — and NOTHING DETECTS THAT. Every skip-check in the
# codebase is `isfile`, and `assemble_or_load_forecasts` validates only `origins`/`labels`, which
# this switch leaves unchanged (origins come from the contact data and a hard-coded
# `infection_start`). The stale files must be DELETED by hand. Stage 1 is unaffected:
# `model_degree` takes contact statistics and `pop`, and `pop` is `grid.POP` from
# `data-processed/populations.csv` — a different file, identical between the two variants.
#
# ⚠ The previous generation, fitted against `estimates_age_ab.csv`, is preserved byte-identical
# in `dt_intermediate_residual/` (504 `8j_s1_*` + 1512 `8j_s2_*`). Nothing in a filename or cache
# token records which source a fit used — `contacts_label` encodes only `constant_contacts` and
# `stage1_use_nuts` — so that directory IS the provenance record.

const _EST_PATH = joinpath(@__DIR__, "..", "inc2prev", "outputs", "estimates_age.csv")

_toint_safe(x) = x isa Integer ? Int(x) :
                 x isa AbstractFloat ? Int(round(x)) :
                 ismissing(x) ? -1 : (v = tryparse(Int, String(x)); v === nothing ? -1 : v)
_tofloat_safe(x) = x isa Real ? Float64(x) : (ismissing(x) ? NaN : parse(Float64, String(x)))
_todate_safe(x) = x isa Date ? x : (ismissing(x) || x == "" ? nothing : Date(String(x)))

"""Load the estimates CSV and build a `t_index → Date` map from the infections rows."""
function _load_estimates(path::AbstractString = _EST_PATH)
    df = CSV.read(path, DataFrame)
    inf = @subset(df, :name .== "infections")
    tmap = Dict{Int,Date}()
    for r in eachrow(inf)
        d = _todate_safe(r.date)
        d === nothing && continue
        tmap[_toint_safe(r.t_index)] = d
    end
    return df, tmap
end

_age_index_map(grid) = Dict(l => i for (i, l) in enumerate(grid.LAB))

# ONS CIS modelled PCR positivity by age (England, daily, 118 stacked publication vintages): the
# framework OUTCOME since 2026-10-22. See `cis_weekly_positivity` / `weekly_cis_positives`.
const _CIS_AGE_PATH = joinpath(@__DIR__, "..", "inc2prev", "data-processed", "cis_age.csv")

"""
    cis_weekly_positivity(weeks; grid, path, vintage=:latest, geography, level, cis_df)
        -> (; P, LO, HI, n_days, n_vintages, vintage_range)

Weekly ONS CIS PCR positivity by CIS age bin, `A × length(weeks)`, as a **proportion**. The
`× population` happens once in `inc2prev_age_timeseries`, so the raw proportion stays inspectable.

⚠ **THE FRAMEWORK OUTCOME IS BUILT FROM THIS** (2026-10-22): `weekly_cis_positives` multiplies `P` by
`grid.POP` and feeds Stage 2 and scoring. It lived in `14j_viz_utils.jl` until then, as a §6c
comparator, and was moved here verbatim.

This is the genuinely OBSERVED series — `inc2prev/stan/inc2prev.stan:176` is
`prev ~ normal(odcases, combined_sigma)`, i.e. `proportion_pos` is literally what both estimate files
were fitted to. (For the REPORTED case counts that sit beside it in the incidence figure, see
`ukhsa_weekly_cases`. An earlier version of this docstring said age-stratified reported counts "do
not exist for this comparison, and UKHSA's live API carries `age: \"all\"` only" — that was true of
the retired `api.coronavirus.data.gov.uk` and is FALSE of its replacement, where
`COVID-19_cases_rateRollingMean` is published in 5-year bands. A note that data does not exist is a
measurement with a date on it; this one had gone stale and closed off a figure.)

⚠ **`cis_age.csv` IS NOT A SERIES — IT IS 118 STACKED PUBLICATION VINTAGES OF ONE.** 34 692
England/`age_school` rows collapse to 6307 `(date, bin)` cells. Read without a collapse it silently
averages first-release and revised estimates together, and the early ONS revisions are large.
`vintage = :latest` keeps the maximum `publication_date` per `(start_date, lower_age_limit)`;
`:first` gives the real-time view and is deliberately NOT the default, because inc2prev's own fit
consumed the revised series.

⚠ **`start_date == end_date` is ASSERTED, not assumed.** True on all 34 692 rows today, which is what
makes this a DAILY series that a weekly mean is well defined over. The sibling files (`cis.csv`,
`cis_variants.csv`) and future vintages can carry multi-day windows; averaging a windowed row as if
it were a day double-counts with no error at all.

⚠ **WEEKLY MEAN, NEVER SUM.** Positivity is a STOCK. This mirrors `weekly_antibody`'s aggregation and
is the OPPOSITE of `weekly_infections`'. See the §G2 banner.

⚠ **Empty cells are `NaN`, not `0.0` — deliberately the opposite of `weekly_antibody`.** That one
zero-fills because a NaN would propagate into the NGM and poison every downstream draw, and it warns
as the tripwire. Nothing downstream of *this* function is a model: it feeds a plot, where `NaN` is
drawn as a gap and a zero would draw a line to the floor of a log axis.

⚠ **`weeks` MUST BE MONDAY-ALIGNED** (week starts of the framework grid) — the lookup is
`haskey(wkset, week_start(d))`, so a vector of Thursday `week_mid` labels matches nothing and returns all-NaN, silently (`incidence_age_shares`'
trap verbatim).

⚠ Bins key on `lower_age_limit` against `grid.LO`, NOT on a label against `grid.LAB` as
`weekly_infections` does — `cis_age.csv` has no label column. Two join keys for the same seven bins,
so the key set is asserted once here. Values parse through `_tofloat_safe`/`_todate_safe`: the column
types are inferred over ALL geographies and levels, so one `"NA"` anywhere in the 7 MB file turns
`proportion_pos` into a `String` column (measured: 0 missing in England/`age_school` today).

`LO`/`HI` are the same weekly mean of the published `proportion_pos_low_95`/`_high_95`. They are
carried for the agreement diagnostic and are NOT drawn — see `make_inc2prev_timeline_fig`.
"""
function cis_weekly_positivity(weeks::AbstractVector{Date}; grid = cis_age_grid(),
                               path::AbstractString = _CIS_AGE_PATH,
                               vintage::Symbol = :latest,
                               geography::AbstractString = "England",
                               level::AbstractString = "age_school",
                               cis_df = nothing)
    vintage in (:latest, :first) ||
        throw(ArgumentError("cis_weekly_positivity: vintage must be :latest or :first, got :$vintage"))
    df = cis_df
    if df === nothing
        isfile(path) || error("cis_weekly_positivity: no such file $path — is the inc2prev submodule " *
                              "checked out? (`git submodule update --init`)")
        df = CSV.read(path, DataFrame)
    end
    keep = [string(g) == geography && string(l) == level
            for (g, l) in zip(df.geography, df.level)]
    any(keep) || error("cis_weekly_positivity: no rows with geography=\"$geography\" level=\"$level\"")
    sub = @view df[keep, :]

    A = grid.N; T = length(weeks)
    wkset = Dict(w => k for (k, w) in enumerate(weeks))
    loidx = Dict(lo => i for (i, lo) in enumerate(grid.LO))

    best  = Dict{Tuple{Date,Int}, Tuple{Date,Float64,Float64,Float64}}()
    los   = Set{Int}()
    vints = Set{Date}()
    for r in eachrow(sub)
        sd = _todate_safe(r.start_date); sd === nothing && continue
        ed = _todate_safe(r.end_date)
        @assert ed === nothing || ed == sd "cis_weekly_positivity: $path has a row spanning \
            $sd…$ed. Every England/age_school row is a SINGLE day today (verified on all 34 692), \
            which is what makes the weekly MEAN below well defined; averaging a multi-day window as \
            if it were one day double-counts silently. Refusing."
        lo = _toint_safe(r.lower_age_limit); push!(los, lo)
        pub = _todate_safe(r.publication_date); pub === nothing && continue
        push!(vints, pub)
        k = (sd, lo)
        prev = get(best, k, nothing)
        newer = prev === nothing || (vintage === :latest ? pub > prev[1] : pub < prev[1])
        newer || continue
        best[k] = (pub, _tofloat_safe(r.proportion_pos),
                        _tofloat_safe(r.proportion_pos_low_95),
                        _tofloat_safe(r.proportion_pos_high_95))
    end
    @assert los == Set(grid.LO) "cis_weekly_positivity: $path carries lower_age_limit $(sort(collect(los))) \
        but the CIS grid is $(grid.LO). These are joined by NUMBER, so a changed bin set would drop or \
        mis-assign bins rather than error."
    ndates = length(Set(k[1] for k in keys(best)))
    @assert length(best) == ndates * A "cis_weekly_positivity: the $vintage-vintage collapse left \
        $(length(best)) (date, bin) cells for $ndates dates × $A bins — the panel is ragged, so some \
        date is missing a bin and the weekly mean there would be taken over fewer bins than it looks."

    Ps = zeros(A, T); Ls = zeros(A, T); Hs = zeros(A, T); n_days = zeros(Int, A, T)
    for ((d, lo), (_, p, l, h)) in best
        w = week_start(d); haskey(wkset, w) || continue
        a = get(loidx, lo, 0); a == 0 && continue
        isfinite(p) || continue
        t = wkset[w]
        Ps[a, t] += p; Ls[a, t] += l; Hs[a, t] += h; n_days[a, t] += 1
    end
    P = fill(NaN, A, T); LOm = fill(NaN, A, T); HIm = fill(NaN, A, T)
    for a in 1:A, t in 1:T
        n_days[a, t] == 0 && continue
        P[a, t]   = Ps[a, t] / n_days[a, t]
        LOm[a, t] = Ls[a, t] / n_days[a, t]
        HIm[a, t] = Hs[a, t] / n_days[a, t]
    end
    return (; P, LO = LOm, HI = HIm, n_days, n_vintages = length(vints),
              vintage_range = (minimum(vints), maximum(vints)))
end

"""
    weekly_cis_positives(cis, weeks, grid; vintage = :latest) -> (I_mean, I_sd)

The framework OUTCOME (2026-10-22, user request): the number of people in England currently PCR
positive, by CIS age bin and Monday week, `A × length(weeks)` — weekly MEAN ONS CIS positivity
(`cis_weekly_positivity`) × `grid.POP`. It replaces inc2prev's modelled infections
(`weekly_infections`) everywhere `WindowData.I_mean` is read: the renewal history, the Stage-2
likelihood target and the scoring truth.

⚠ **IT IS A PREVALENCE (a stock), USED AS-IS where the renewal expects incidence** (user's choice).
Weekly positivity × pop is roughly (mean detectable duration ≈ 8.7 d)/7 ≈ 1.24 weekly incidences,
lagged and smoothed; γ_SAR and the generation interval absorb that. Never read γ_SAR from these fits
as a per-contact attack rate on the incidence scale.

`I_sd` = weekly mean of the daily `(HI − LO)/(2·1.96)` × POP — inc2prev's own CrI → SD rule
(`R/model.R`). Daily values come from one smooth ONS model, so the SDs are AVERAGED (perfect
correlation), not added in quadrature as independent days would be.

⚠ ASSERTS 7 days and a finite, positive value in EVERY cell. `cis_weekly_positivity` NaN-fills an
empty cell and averages a part-week silently; here either is an error, because it is what the
origin grid's lower bound (`available_forecast_origins`' `infection_start`, the first full Monday
week 2020-08-17) exists to prevent.

`cis` is the raw `cis_age.csv` frame (`load_raw_infection_inputs().cis`), read once per process.
"""
function weekly_cis_positives(cis::AbstractDataFrame, weeks::AbstractVector{Date}, grid;
                              vintage::Symbol = :latest)
    pos = cis_weekly_positivity(weeks; grid = grid, cis_df = cis, vintage = vintage)
    bad = findall(pos.n_days .!= 7)
    @assert isempty(bad) "weekly_cis_positives: $(length(bad)) (bin, week) cell(s) without 7 days of \
        CIS data, first at week $(weeks[bad[1][2]]) (n_days = $(pos.n_days[bad[1]])). CIS by age \
        covers 2020-08-14 … 2023-01-31; an origin needs all of `all_weeks`/`forecast_weeks` inside it."
    @assert all(p -> isfinite(p) && p > 0, pos.P) "weekly_cis_positives: non-finite or zero positivity"
    I_mean = pos.P .* grid.POP
    I_sd   = (pos.HI .- pos.LO) ./ (2 * 1.96) .* grid.POP
    @assert all(isfinite, I_sd) && all(>=(0), I_sd) "weekly_cis_positives: bad CrI-derived SD"
    return I_mean, I_sd
end

"""Read `cis_age.csv` (raw; `cis_weekly_positivity` does the geography/level/vintage cut)."""
_load_cis(path::AbstractString = _CIS_AGE_PATH) =
    (isfile(path) || error("no such file $path — is the inc2prev submodule checked out?");
     CSV.read(path, DataFrame))

# ONE read of `cis_age.csv` per process, shared by every `load_window_data`/`load_forecast_truth`
# call, so none of their ~25 call sites had to change signature when the outcome moved to the CIS.
# Locked because `prefit_*!` call the data provider from concurrent tasks. Treated read-only.
const _CIS_CACHE = Ref{Union{Nothing,DataFrame}}(nothing)
const _CIS_LOCK  = ReentrantLock()
_cis_frame() = lock(_CIS_LOCK) do
    _CIS_CACHE[] === nothing && (_CIS_CACHE[] = _load_cis())
    _CIS_CACHE[]
end

"""Weekly infection counts and SDs (A × length(weeks)).

Weekly count = 7-day sum of daily incidence proportion × population
(CovidAgeGroupForecast make_infs_weekly.R:20). SDs combined in quadrature across
the week (independence approximation — flagged in inst/1a)."""
function weekly_infections(df::DataFrame, weeks::Vector{Date}, grid)
    A = grid.N; T = length(weeks)
    wkset = Dict(w => k for (k, w) in enumerate(weeks))
    ageidx = _age_index_map(grid)
    I_mean = zeros(A, T); I_var = zeros(A, T)
    inf = @subset(df, :name .== "infections")
    for r in eachrow(inf)
        d = _todate_safe(r.date); d === nothing && continue
        w = week_start(d); haskey(wkset, w) || continue
        a = get(ageidx, String(r.variable), 0); a == 0 && continue
        t = wkset[w]; pop = _tofloat_safe(r.population)
        I_mean[a, t] += _tofloat_safe(r.mean) * pop
        I_var[a, t]  += (_tofloat_safe(r.sd) * pop)^2
    end
    return I_mean, sqrt.(I_var)
end

# Fires the "this source has no antibody" notice ONCE per session rather than once per call.
# The absence is a property of the FILE, not of the window, so every one of the ~126 call sites in
# a full grid run would otherwise report the same fact — and `weekly_antibody`'s per-week warning
# would list EVERY week of EVERY window, burying the run in output that says one thing.
const _NO_ANTIBODY_NOTIFIED = Ref(false)

function _notify_no_antibody()
    _NO_ANTIBODY_NOTIFIED[] && return nothing
    _NO_ANTIBODY_NOTIFIED[] = true
    # Real line breaks, not `\` continuations: joined, this was ONE 387-character line, which
    # `export_notebook_pdf.py` refuses rather than wrap (14j cell "§6 NGM" prints it).
    @info """weekly_antibody: the estimates source carries NO `gen_dab` rows — antibody is
             all-zero for every week and every age bin. This is `estimates_age.csv`, which has no
             antibody at all (only the `_ab` variants do), and it is HARMLESS while the leaky
             term is disabled: `model_transmission` pins `F = 1.0`, so the NGM's antibody factor
             `(1 + (F-1)*A)` is exactly 1 whatever `A` contains.
             ⚠ RE-ENABLING `F ~ Beta(5, 1)` (joint_model.jl) against this source would mean ZERO
             antibody protection everywhere, silently — F would be inferred from a covariate that
             is identically 0. Point `_EST_PATH` back at an `_ab` variant first.""" source = _EST_PATH
    return nothing
end

"""Weekly antibody prevalence A_a(t) ∈ [0,1] (A × length(weeks)), averaged over the
week from `gen_dab`; dates recovered from `t_index` via `tmap`.

⚠ **THE CURRENT SOURCE (`estimates_age.csv`) HAS NO `gen_dab` ROWS AT ALL**, so this returns an
all-zero matrix and emits `_notify_no_antibody()`'s one-time `@info`. That is inert today only
because `model_transmission` pins `F = 1.0` — read that notice before re-enabling the term.
The no-rows case is handled SEPARATELY from the per-week gap below on purpose: "this file has no
antibody" and "this window straddles the end of the antibody series" are different facts and want
different reports.

⚠ Where the source DOES carry antibody, unmatched weeks are filled with `0.0`, which is
INDISTINGUISHABLE from genuinely zero antibody prevalence — and zero antibody means full
susceptibility, so a zero-filled column inflates whatever it feeds. Nothing bounds forecast
origins by the `gen_dab` series' end (the sibling problem `weekly_infections` has,
tasks/lessons.md 2026-07-15). Since 2026-07-30 this **warns** once per call listing the offending
weeks; `what` labels them in the message (e.g. "forecast target"), and `warn_unmatched=false`
silences it for callers that expect gaps.

The zero fill is retained as the VALUE on purpose: a `NaN`/`missing` would propagate silently
into the NGM and poison every downstream draw. The warning is the tripwire, not the fill."""
function weekly_antibody(df::DataFrame, tmap::Dict{Int,Date}, weeks::Vector{Date}, grid;
                         warn_unmatched::Bool = true, what::AbstractString = "window")
    A = grid.N; T = length(weeks)
    gd = @subset(df, :name .== "gen_dab")
    # Source-level absence: report once and return, WITHOUT the per-week warning — every week is
    # "unmatched" here, and that message would name the file's contents as if it were a gap.
    isempty(gd) && (_notify_no_antibody(); return zeros(A, T))
    wkset = Dict(w => k for (k, w) in enumerate(weeks))
    ageidx = _age_index_map(grid)
    acc = [Float64[] for _ in 1:A, _ in 1:T]
    for r in eachrow(gd)
        d = get(tmap, _toint_safe(r.t_index), nothing); d === nothing && continue
        w = week_start(d); haskey(wkset, w) || continue
        a = get(ageidx, String(r.variable), 0); a == 0 && continue
        push!(acc[a, wkset[w]], _tofloat_safe(r.mean))
    end
    AB = zeros(A, T)
    for a in 1:A, t in 1:T
        AB[a, t] = isempty(acc[a, t]) ? 0.0 : mean(acc[a, t])
    end
    if warn_unmatched
        # report per WEEK (not per cell): a missing week is missing for every age bin, so a
        # per-cell warning would fire A times for the same cause.
        missing_wk = [weeks[t] for t in 1:T if all(isempty, @view acc[:, t])]
        isempty(missing_wk) || @warn "weekly_antibody: no gen_dab rows for these $what weeks — " *
                                     "antibody zero-filled ⇒ FULL susceptibility there" weeks = missing_wk
    end
    return AB
end

"""
    load_raw_infection_inputs(; path=_EST_PATH)

Read the infection-side inputs **once**: the inc2prev estimates CSV with its `t_index → Date` map
(now used for ANTIBODY only — `estimates_age.csv` carries none, so `A ≡ 0`), and the ONS CIS
by-age frame `cis` that the OUTCOME comes from (`weekly_cis_positives`). Returns `(; df, tmap, cis)`;
pass `inf.df, inf.tmap` straight into `load_window_data(win, inf.df, inf.tmap; grid)`, which reads
the CIS from the same per-process cache (`_cis_frame`). Every frame is treated read-only, so it is
safe to reuse across origins and to read concurrently from a background prefetch task.
"""
function load_raw_infection_inputs(; path::AbstractString = _EST_PATH)
    df, tmap = _load_estimates(path)
    return (; df, tmap, cis = _cis_frame())
end

"""Assemble `WindowData` from a **pre-loaded** estimates frame `df` and `t_index → Date`
map `tmap` (from `load_raw_infection_inputs()`), skipping the per-call CSV read. Byte-identical
to the `path`-reading method; safe to call concurrently (read-only over `df`/`cis`).

`I_mean`/`I_sd` are the ONS CIS positives × population (`weekly_cis_positives`, 2026-10-22) — NOT
inc2prev's modelled infections, which `weekly_infections` still reads for the 14j/16j audits. The
field names are unchanged so no reader had to move; read them as "people currently positive"."""
function load_window_data(win::WeeklyWindow, df::DataFrame, tmap::Dict{Int,Date};
                          grid = cis_age_grid(), cis::AbstractDataFrame = _cis_frame())
    weeks = win.all_weeks
    I_mean, I_sd = weekly_cis_positives(cis, weeks, grid)
    AB = weekly_antibody(df, tmap, weeks, grid; what = "window")
    # Antibody at the FORECAST TARGET weeks t₀+h, for the frozen forecast NGM (§3.2). A gap here
    # matters more than one in the fit window — it drives every horizon-h prediction — so it gets
    # its own labelled warning.
    AB_fc = weekly_antibody(df, tmap, win.forecast_weeks, grid; what = "FORECAST TARGET")
    return WindowData(weeks, grid.N, I_mean, I_sd, AB, AB_fc, grid.POP, grid.PROP, grid.LAB)
end

"""Assemble `WindowData` for the full 12-week (lags + fit) span of `win` — `win.all_weeks`, and
UNCHANGED by `-w8h` (2026-08-09), which reshaped only the CONTACT window (`prepare_degree_data`,
now `win.fit_weeks ++ win.forecast_weeks` = `[t₀−n_fit+1 … t₀+h]`). The renewal needs `smax` weeks of infection history that carry no `C*`, so the
two windows differ in length and are paired with a `smax` offset; see `model_transmission`. Reads the
estimates CSV from `path`. Backward-compatible wrapper around the pre-loaded-frame method."""
function load_window_data(win::WeeklyWindow; path::AbstractString = _EST_PATH, grid = cis_age_grid())
    df, tmap = _load_estimates(path)
    return load_window_data(win, df, tmap; grid = grid)
end

"""Realized weekly CIS positives × population (A × n_horizons) for the forecast target weeks
— the scoring `true_value`. Same quantity as `load_window_data`'s `I_mean` (`weekly_cis_positives`)."""
function load_forecast_truth(win::WeeklyWindow; grid = cis_age_grid(),
                             cis::AbstractDataFrame = _cis_frame())
    I_mean, _ = weekly_cis_positives(cis, win.forecast_weeks, grid)
    return I_mean
end
