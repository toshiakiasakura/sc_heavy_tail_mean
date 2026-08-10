# infection_data.jl — load age-stratified infection + antibody estimates from
# inc2prev and aggregate to the inc2prev-aligned weekly grid.
#
# estimates_age_ab.csv schema: `name` selects the quantity (infections, gen_dab, …),
# `variable` is the age group, `mean`/`sd` the daily estimate, `population` the
# age-group size, `t_index` a 1-based day index. `infections` carries `date`;
# `gen_dab` (antibody prevalence) does NOT — its date is recovered from `t_index`
# via the infections rows (matches CovidAgeGroupForecast run_forecasts_for_infections.R:75).

const _EST_PATH = joinpath(@__DIR__, "..", "inc2prev", "outputs", "estimates_age_ab.csv")

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

"""Weekly antibody prevalence A_a(t) ∈ [0,1] (A × length(weeks)), averaged over the
week from `gen_dab`; dates recovered from `t_index` via `tmap`.

⚠ Unmatched weeks are filled with `0.0`, which is INDISTINGUISHABLE from genuinely zero antibody
prevalence — and zero antibody means full susceptibility, so a zero-filled column inflates
whatever it feeds. Nothing bounds forecast origins by the `gen_dab` series' end (the sibling
problem `weekly_infections` has, tasks/lessons.md 2026-07-15). Since 2026-07-30 this **warns**
once per call listing the offending weeks; `what` labels them in the message (e.g. "forecast
target"), and `warn_unmatched=false` silences it for callers that expect gaps.

The zero fill is retained as the VALUE on purpose: a `NaN`/`missing` would propagate silently
into the NGM and poison every downstream draw. The warning is the tripwire, not the fill."""
function weekly_antibody(df::DataFrame, tmap::Dict{Int,Date}, weeks::Vector{Date}, grid;
                         warn_unmatched::Bool = true, what::AbstractString = "window")
    A = grid.N; T = length(weeks)
    wkset = Dict(w => k for (k, w) in enumerate(weeks))
    ageidx = _age_index_map(grid)
    acc = [Float64[] for _ in 1:A, _ in 1:T]
    gd = @subset(df, :name .== "gen_dab")
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

Read the inc2prev estimates CSV **once** and build the `t_index → Date` map (the read
`load_window_data` would otherwise repeat on every origin). Returns `(; df, tmap)` to pass
straight into `load_window_data(win, inf.df, inf.tmap; grid)`, mirroring
`load_raw_contact_inputs()`. The frame is treated read-only — `weekly_infections`/
`weekly_antibody` `@subset` it into copies — so it is safe to reuse across origins and to
read concurrently from a background prefetch task.
"""
function load_raw_infection_inputs(; path::AbstractString = _EST_PATH)
    df, tmap = _load_estimates(path)
    return (; df, tmap)
end

"""Assemble `WindowData` from a **pre-loaded** estimates frame `df` and `t_index → Date`
map `tmap` (from `load_raw_infection_inputs()`), skipping the per-call CSV read. Byte-identical
to the `path`-reading method; safe to call concurrently (read-only over `df`)."""
function load_window_data(win::WeeklyWindow, df::DataFrame, tmap::Dict{Int,Date};
                          grid = cis_age_grid())
    weeks = win.all_weeks
    I_mean, I_sd = weekly_infections(df, weeks, grid)
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

"""Realized weekly infection counts (A × n_horizons) for the forecast target weeks
— the scoring `true_value`."""
function load_forecast_truth(win::WeeklyWindow; path::AbstractString = _EST_PATH, grid = cis_age_grid())
    df, _ = _load_estimates(path)
    I_mean, _ = weekly_infections(df, win.forecast_weeks, grid)
    return I_mean
end
