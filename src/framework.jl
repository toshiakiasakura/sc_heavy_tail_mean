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
    n_forecast_draws::Int = 200       # posterior draws retained per fan for the stored (capped) assembly cache
    # --- two-stage cut inference (inst/4_cut_Bayes.md) ---
    # Stage 1 (contact degree) is fit once; `n_stage1_post` of its posterior draws are carried into
    # Stage 2 (infection), which is re-fit conditioning on each and yields `n_stage2_draws` samples.
    # The 100×100 = 10_000 pooled infection draws form the predictive distribution scored by WIS.
    n_stage1_post::Int    = 100       # Stage-1 posterior draws imputed into Stage 2 (the "100 posteriors")
    n_stage2_draws::Int   = 100       # Stage-2 samples kept per Stage-1 draw (the "100 samples each")
    stage1_use_nuts::Bool = false     # Stage-1 sampler: false = Pathfinder (preliminary), true = NUTS (later)
    # --- separable spatio-temporal GP smoothing of the age-pair mean (inst/1e, §5) ---
    gp_len_prior::Tuple{Float64,Float64}   = (log(15.0), 0.5)  # log-ρ Normal(μ,σ), age-years; shared by BOTH spatial diagonal length-scales (ρ_diag=total-age, ρ_gap=age-gap)
    gp_scale_prior::Tuple{Float64,Float64} = (0.0, 0.5)        # log-η Normal(μ,σ), GP marginal scale (age-pair field)
    gp_time_len_prior::Tuple{Float64,Float64}   = (log(4.0), 0.5)  # log-ρ_time Normal(μ,σ), weeks; temporal length-scale (shared across age-pairs), per-week regime only
    gp_level_scale_prior::Tuple{Float64,Float64} = (0.0, 0.5)      # log-σ_c Normal(μ,σ), amplitude of the decoupled temporal level GP c_t = c + σ_c·(Lt·z_c)
    # --- secondary attack rate γ_SAR (§3.2/§6; analysis-plan per-contact SAR, non-normalised C*) ---
    gamma_sar_prior::Tuple{Float64,Float64} = (log(0.1), 1.8) # log-γ_SAR Normal(μ,σ): the per-contact secondary attack rate. C* is NOT normalised (the -gnorm C*→C*/S̄ decoupling was reverted 2026-07-12, inst/4_cut_Bayes.md), so γ_SAR reproduces the reference cell N_11 = susc₁·inf₁ = γ_SAR directly. LOOSENED 2026-07-13 to span γ_SAR∈[0.001,10] (softclamp bounds below): the earlier (log0.27, 1.05) prior [90% γ_SAR∈[0.048,1.52]] and softclamp lower bound log0.02 were actively pinning the low-γ configs — the negbin|neighbourhood posterior median (~0.021) sat right on the log0.02 clamp with an implausibly tight CI (clamp compression). New centre log(0.1) = geometric mean of [0.001,10] with log-SD 1.8 ⇒ 90% γ_SAR∈[0.0052,1.93], weakly-informative across the full range. The softclamp [log0.001,log10] now sits at ≈±2.56σ (outside the 90% band, tails ≈0.5% each), so it comfortably contains the prior and stops biasing the low tail. NOTE: this change invalidates cached 8j_s2_* Stage-2 chains (the contacts token does not encode the prior) — delete them and re-run prefit_stage2! to regenerate; Stage-1 8j_s1_* chains are γ_SAR-independent and unaffected.
    # --- independent per-bin marginal SD of the relative susc/inf age profile (Stage-2 transmission block; NO cross-bin smoothing) ---
    susc_inf_sd_prior::Tuple{Float64,Float64} = (0.5, 0.25) # N⁺(mean,sd) marginal-SD prior for the susc/inf offset sig·z (INDEPENDENT per age bin — the shared-length-scale RBF GP `sig·(Lsi·z)` was removed 2026-07-13, user request); used SEPARATELY by sig_s (susc) and sig_i (inf) — two distinct latents, one shared prior form. LOOSENED 2026-07-13 from N⁺(0.1,0.05²) → N⁺(0.5,0.25²) (user request): marginal SD ≈0.5 ⇒ ±2 SD ≈±1.0 in log ⇒ typical susc/inf ≈[0.37,2.7] (was [0.80,1.25]) — allows real age variation in inherent susceptibility/infectivity. Soft-clamp LOOSENED 2026-07-13 to [log0.05,log20] (≈[-3,3]) ⇒ hard-bounds susc/inf ∈ [0.05,20] (±2 SD well interior; clamp at ~±6 SD) — matches the wider prior and re-aligns with the `-sc` cache-token docstring. (The `susc_inf_gp_len_prior` length-scale field was dropped with the GP.)
end

"""`contacts_label(cfg)` — tags the contact/model regime for chain-cache filenames so fits with
different parameter spaces never reload each other's stale chains. The suffix is a running version
tag (the filename does not otherwise encode dispersion/transmission structure): `-gsar-cut`
(2026-07-12, inst/4_cut_Bayes.md — the JOINT fit was split into a two-stage CUT inference: Stage 1
fits the contact-degree GP alone (saved `8j_s1_*`), Stage 2 re-fits the infection block conditioning
on each of 100 Stage-1 draws (pooled `8j_s2_*`). The C*-normalisation (`C*→C*/S̄`) was REVERTED, so C*
feeds the NGM at its raw level and the transmissibility scalar is again the per-contact SAR
`log_gamma_sar`/`gamma_sar`). The `-sc` suffix (2026-07-12) marks the Stage-2 susc/inf **soft-clamp**
`exp(softclamp(σ·z, -3, 3))` (bounds relative susceptibility/infectivity to ≈[0.05,20] so a stray
Pathfinder draw can't blow the NGM up); it changes the Stage-2 posterior, so the pre-`-sc` `8j_s2_*`
files carry unclamped susc/inf and must not be reused. The `-cut` artefacts (`8j_s1_*`/`8j_s2_*`) are
structurally disjoint from the single-file `-gnorm` joint chains (`8j_chn_*`, differently-scaled
`log_gamma`), so nothing reloads the stale ones. (The `-sm` suffix, 2026-07-12, formerly marked the
Stage-2 susc/inf smoothing — first RW1/RW2, then a shared-length-scale RBF GP — and was RETIRED
2026-07-13 when the smoothing was removed (user request), reverting the tag to `-sc`. Because the
smoothing lived only in Stage 2, the existing `8j_s1_*` Stage-1 chains were renamed `-sc-sm`→`-sc` on
disk and reused as-is, not refit; the smoothed `8j_s2_*` pooled files were stale and are regenerated
under `-sc`.)"""
contacts_label(cfg::FrameworkConfig) = cfg.constant_contacts ? "pooled-gsar-cut-sc" : "temporal-gsar-cut-sc"

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
