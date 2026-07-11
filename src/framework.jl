# framework.jl — composable forecasting framework: types, window, config, containers.
#
# Two swappable axes (analysis plan inst/1a):
#   ContactDegreeModel  — how the age-pair contact degree distribution is modelled
#   NGMBuilder          — how the next-generation matrix is built from that distribution
# See src/joint_model.jl for the joint Turing model that consumes both.

##########################################################################
# Swap axis 1: contact-degree model
##########################################################################
abstract type ContactDegreeModel end
struct NegBinAgePair        <: ContactDegreeModel end   # unweighted (integer counts)
struct HurdleWeibullAgePair <: ContactDegreeModel end   # duration-weighted (hurdle)

is_weighted(::NegBinAgePair)        = false
is_weighted(::HurdleWeibullAgePair) = true
degree_label(::NegBinAgePair)        = "unweighted-negbin"
degree_label(::HurdleWeibullAgePair) = "weighted-hweibull"

##########################################################################
# Swap axis 2: NGM builder
##########################################################################
abstract type NGMBuilder end
struct MeanNGM                <: NGMBuilder end   # C0 = mean degree
struct NeighbourhoodDegreeNGM <: NGMBuilder end   # C0 = excess degree m(1+CV^2)

ngm_label(::MeanNGM)                = "mean"
ngm_label(::NeighbourhoodDegreeNGM) = "neighbourhood"

##########################################################################
# inc2prev-aligned weekly grid (Sunday-start, anchor 2021-03-21)
##########################################################################
const WEEK_ANCHOR = Date(2021, 3, 21)
week_index(d::Date) = fld((d - WEEK_ANCHOR).value, 7)
week_start(d::Date) = WEEK_ANCHOR + Day(week_index(d) * 7)  # Sunday of the week
week_mid(d::Date)   = week_start(d) + Day(3)                # Wednesday label

"""
    WeeklyWindow(origin; n_fit=8, smax=4, horizons=1:4)

A single forecast window. `origin` is snapped to its Sunday-start week and becomes
the last fitting week. `fit_weeks` are the `n_fit` weeks ending at `origin`;
`lag_weeks` are the `smax` weeks of renewal history immediately before them;
`all_weeks = lag_weeks ++ fit_weeks` (chronological). `forecast_weeks` are the
`horizons`-ahead target weeks after `origin`.
"""
struct WeeklyWindow
    origin::Date
    fit_weeks::Vector{Date}
    lag_weeks::Vector{Date}
    all_weeks::Vector{Date}
    smax::Int
    horizons::UnitRange{Int}
    forecast_weeks::Vector{Date}
end

function WeeklyWindow(origin::Date; n_fit::Int = 8, smax::Int = 4, horizons = 1:4)
    o = week_start(origin)
    fit_weeks = [o - Day(7 * (n_fit - 1 - k)) for k in 0:(n_fit - 1)]      # ascending, last = o
    lag_weeks = [o - Day(7 * (n_fit - 1 + smax - k)) for k in 0:(smax - 1)] # smax weeks before fit_weeks[1]
    all_weeks = vcat(lag_weeks, fit_weeks)
    forecast_weeks = [o + Day(7 * h) for h in horizons]
    WeeklyWindow(o, fit_weeks, lag_weeks, all_weeks, smax, horizons, forecast_weeks)
end

n_all_weeks(w::WeeklyWindow)  = length(w.all_weeks)
fit_offset(w::WeeklyWindow)   = length(w.lag_weeks)  # index (0-based) of fit_weeks[1] within all_weeks

##########################################################################
# Configuration
##########################################################################
Base.@kwdef struct FrameworkConfig
    d_max::Float64        = 240.0     # duration-weight cap (>4h ⇒ weight 1)
    w_dur_group::Float64  = 2.5 / 240 # group-contact duration weight (inst/1e; fixed now, estimated later)
    smax::Int             = 4         # renewal weekly lags
    n_fit::Int            = 8         # fitting weeks
    horizons::UnitRange{Int} = 1:4    # forecast horizons (weeks)
    constant_contacts::Bool = true    # lean: one latent mean per cell (vs per-week RW1)
    seed::Int             = 1236
    gen_mean_days::Float64 = 5.0      # generation-interval mean (reference 5-day)
    gen_sd_days::Float64   = 5.0      # generation-interval sd
    child_bins::Int       = 2         # bins 1..child_bins ("2-10","11-15") are "child"
    quantiles::Vector{Float64} = collect(0.05:0.05:0.95)
    n_forecast_draws::Int = 200       # posterior draws retained for scoring
    # --- separable spatio-temporal GP smoothing of the age-pair mean (inst/1e, §5) ---
    gp_len_prior::Tuple{Float64,Float64}   = (log(15.0), 0.5)  # log-ρ Normal(μ,σ), age-years; shared by BOTH spatial diagonal length-scales (ρ_diag=total-age, ρ_gap=age-gap)
    gp_scale_prior::Tuple{Float64,Float64} = (0.0, 0.5)        # log-η Normal(μ,σ), GP marginal scale (age-pair field)
    gp_time_len_prior::Tuple{Float64,Float64}   = (log(4.0), 0.5)  # log-ρ_time Normal(μ,σ), weeks; temporal length-scale (shared across age-pairs), per-week regime only
    gp_level_scale_prior::Tuple{Float64,Float64} = (0.0, 0.5)      # log-σ_c Normal(μ,σ), amplitude of the decoupled temporal level GP c_t = c + σ_c·(Lt·z_c)
    # --- absolute transmissibility γ (§3.2/§6; analysis-plan reparam + C* normalisation) ---
    gamma_prior::Tuple{Float64,Float64} = (log(0.8), 0.5) # log-γ Normal(μ,σ): the single absolute transmissibility scalar. C* is normalised inside model_joint to unit fit-window mean intensity (C*→C*/S̄), so γ = susc₁·inf₁·S̄ ≈ Rt/ρ(C̃*) carries the absolute level and is decoupled from the contact scale (no longer a per-contact SAR; window-relative, not comparable across origins). Calibrated 2026-07-11 by reconstructing new-γ = susc₁·inf₁·S̄ per draw from the pre-gsar dt_intermediate_age_pair_temporal_GP chains, 6 origins × all 4 (degree×NGM) combos: new-γ medians 0.69 (hweibull·mean), 0.73 (hweibull·neigh), 0.83 (negbin·mean), 1.20 (negbin·neigh); per-combo log-SD 0.12–0.34. A SINGLE prior serves all four (geo-mean centre 0.84; S̄ itself spans ~100× — negbin·neigh S̄≈173 — but susc₁·inf₁ compensates the builder scale, so γ stays O(1)). 90% γ∈[0.35,1.8] sits interior to the softclamp [log0.02,log5].
end

"""`contacts_label(cfg)` — tags the contact/model regime for chain-cache filenames so fits with
different parameter spaces never reload each other's stale chains. The suffix is a running version
tag (the filename does not otherwise encode dispersion/transmission structure): `-gnorm` (2026-07-11,
C*-normalisation + γ rename — C* divided by its unit fit-window mean intensity inside `model_joint`,
transmission scalar renamed `log_gamma_sar`→`log_gamma`/`gamma_sar`→`γ` and its prior recalibrated,
§3.2/§6). Predecessor `-gsar` (absolute `log_gamma_sar`, susc/inf bin-1 relative, `z_s`/`z_i` length
`A-1`) used a NON-normalised C*; the `-gnorm` chains carry a differently-scaled γ (`log_gamma`), so
they are disjoint from `-gsar`, from `-hdisp-hn-gsar` (which also carry `z_k`/`z_kappa`+`tau`), and
from the pre-`gsar` `"temporal"`/`"pooled"` chains (`mu_s`/`mu_i`), and reload none of them."""
contacts_label(cfg::FrameworkConfig) = cfg.constant_contacts ? "pooled-gnorm" : "temporal-gnorm"

##########################################################################
# Age grid — CIS "age_school" bins from inc2prev populations (England).
# Matches src/7j_weekly_age_pair.ipynb §2 exactly.
##########################################################################
const _POP_PATH = joinpath(@__DIR__, "..", "inc2prev", "data-processed", "populations.csv")

"""
    cis_age_grid(; path=_POP_PATH)

Return `(; LO, HI, POP, LAB, N, PROP)` for the 7 England CIS `age_school` bins
(lo ≥ 2): lower/upper limits, populations, labels, count, and population
proportions (`PROP = POP/sum(POP)`).
"""
function cis_age_grid(; path::AbstractString = _POP_PATH)
    pop_df  = CSV.read(path, DataFrame)
    pop_age = @subset(pop_df, :level .== "age_school", :geography .== "England")
    _asint(x)   = x isa AbstractString ? parse(Int, x)     : Int(x)
    _asfloat(x) = x isa AbstractString ? parse(Float64, x) : Float64(x)
    pop_age = @transform(pop_age, :lo = _asint.(:lower_age_limit), :pop = _asfloat.(:population))
    pop_age = @subset(pop_age, :lo .>= 2)
    sort!(pop_age, :lo)
    LO  = pop_age.lo
    HI  = vcat(LO[2:end] .- 1, 120)
    POP = pop_age.pop
    N   = length(LO)
    LAB = [LO[j] == LO[end] ? "$(LO[j])+" : "$(LO[j])-$(HI[j])" for j in 1:N]
    return (; LO, HI, POP, LAB, N, PROP = POP ./ sum(POP))
end

"""
    cis_age_midpoints(; grid=cis_age_grid())

Numeric age coordinate for each CIS bin, used as the input to the spatial GP kernel
(inst/1e). Each bin takes its interval midpoint `(lo+hi)/2`; the open-ended **70+**
bin is fixed to **74.5** (the observed mean age of 70+ CoMix participants, ~74.1–74.7).
For the default England grid this returns `[6.0, 13.0, 20.0, 29.5, 42.0, 59.5, 74.5]`.
"""
function cis_age_midpoints(; grid = cis_age_grid())
    mid = [(grid.LO[j] + grid.HI[j]) / 2 for j in 1:grid.N]
    mid[end] = 74.5                     # 70+ open-ended → assumed mean age (inst/1e)
    return mid
end

##########################################################################
# Containers
##########################################################################
"""Per-cell weekly contact-degree data for the window (both model families).

Indexed `[t, i, j]` over `T = length(all_weeks)` weeks, `A` participant bins
`i` (contactor / susceptible), `A` contactee bins `j` (infectious).

- `dd_count[t,i,j]` :: DegreeDist — integer contact counts incl. zeros (NegBin path).
- `pos_weight[t,i,j]` :: WeightedDegreeHist — collapsed (value→count) histogram of positive
  duration-weighted degrees (hurdle path); `whist_mean`/`isempty` accessors.
- `p0[t,i,j]` — empirical zero probability in the cell (weighted path).
- `n[t,i,j]` — sampled participant-days with part_bin i in week t (cell denominator).
- `emp_mean`, `emp_cv2` — empirical per-capita mean and CV^2 (fallback / init).
"""
struct AgePairData
    setting::Symbol
    weeks::Vector{Date}
    A::Int
    dd_count::Array{DegreeDist,3}
    pos_weight::Array{WeightedDegreeHist,3}
    p0::Array{Float64,3}
    n::Array{Int,3}
    emp_mean::Array{Float64,3}
    emp_cv2::Array{Float64,3}
end

"""Weekly window data for the transmission/renewal fit.

`I_mean`, `I_sd` :: (A × T) weekly infection counts and SDs (rolling-sum × population).
`antibody` :: (A × T) weekly antibody prevalence A_a(t) ∈ [0,1].
`pop`, `prop` :: (A) populations and proportions. `weeks` chronological (all_weeks).
"""
struct WindowData
    weeks::Vector{Date}
    A::Int
    I_mean::Matrix{Float64}
    I_sd::Matrix{Float64}
    antibody::Matrix{Float64}
    pop::Vector{Float64}
    prop::Vector{Float64}
    labels::Vector{String}
end
