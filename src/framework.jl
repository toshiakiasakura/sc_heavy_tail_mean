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
# NULL model (inst/6_null_interaction_model.md): no contact likelihood at all — Stage 1 is skipped
# and C* is a fixed uniform constant (see `null_contact_level` / `null_moment_draws` in
# joint_model.jl). Paired with `NullNGM`.
struct NoContactDegree      <: ContactDegreeModel end   # no social contact data used

is_weighted(::NegBinAgePair)        = false
is_weighted(::HurdleWeibullAgePair) = true
is_weighted(::NoContactDegree)      = false
degree_label(::NegBinAgePair)        = "unweighted-negbin"
degree_label(::HurdleWeibullAgePair) = "weighted-hweibull"
degree_label(::NoContactDegree)      = "no-contact"

"""`needs_stage1(dm)` — does this degree model require a Stage-1 (contact-degree GP) fit? `false`
only for `NoContactDegree`, whose C* is a data-derived constant rather than a fitted quantity; the
prefit drivers skip Stage 1 (and `build_degree_stats`) entirely for it."""
needs_stage1(::ContactDegreeModel) = true
needs_stage1(::NoContactDegree)    = false

##########################################################################
# Swap axis 2: NGM builder
##########################################################################
abstract type NGMBuilder end
struct MeanNGM                <: NGMBuilder end   # C0 = mean degree
struct NeighbourhoodDegreeNGM <: NGMBuilder end   # C0 = excess degree m(1+CV^2)
# --- baseline builders (inst/6_null_interaction_model.md) ---
struct NullNGM                <: NGMBuilder end   # C0 = a fixed uniform constant (no contact data)
struct DiagonalMeanNGM        <: NGMBuilder end   # C0 = mean degree, off-diagonal zeroed

ngm_label(::MeanNGM)                = "mean"
ngm_label(::NeighbourhoodDegreeNGM) = "neighbourhood"
ngm_label(::NullNGM)                = "null"
ngm_label(::DiagonalMeanNGM)        = "mean-diagonal"

"""`fix_infectivity(nb)` — pin the relative age-dependent infectivity to `inf ≡ 1` (all bins)?

`true` only for `DiagonalMeanNGM` (the NO-INTERACTION model, inst/6). With a diagonal C* the NGM is
diagonal, `N_aa = γ_SAR·susc_a·(1+(F−1)A_a)·C*_aa·inf_a`, so `susc_a` and `inf_a` enter only through
their product and are individually non-identifiable — infectivity is fixed to 1 and the age profile
is carried entirely by `susc`. See `model_transmission` (joint_model.jl)."""
fix_infectivity(::NGMBuilder)      = false
fix_infectivity(::DiagonalMeanNGM) = true

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
    gen_mean_days::Float64 = 5.0      # generation-interval mean, days — since 2026-07-30 this is the PRIOR CENTRE for the estimated GI (§3.1), not a fixed value
    gen_sd_days::Float64   = 5.0      # generation-interval sd, days — likewise the prior centre
    gen_prior_rel_sd::Float64 = 0.2   # prior SD as a FRACTION of the prior mean, for both GI log-parameters: w_mu ~ Normal(w_mu0, |w_mu0|·r), w_sigma ~ N⁺(w_sigma0, |w_sigma0|·r). Munday 2023 p.8: "their prior was set to be normally distributed with a standard deviation of 20% of the mean". This width is load-bearing — w and γ_SAR are confounded (both scale the renewal predictor), and this informative prior is what identifies the pair. Do NOT loosen it without checking the 9j GI-vs-prior panel.
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
    stage1_pathfinder_runs::Int = 1   # Stage-1 Pathfinder paths. >1 ⇒ `multipathfinder` (independent LBFGS runs pooled by Pareto-smoothed importance resampling); 1 ⇒ single-path. RESET TO 1 on 2026-07-30 (user request, and the measurement agrees). It was briefly 4, to insure against the single-path divergence seen BEFORE the κ clamp was corrected to [-4.3,5]. Once the clamp was fixed the premise vanished: measured head-to-head on hurdle-Weibull, 5 seeds, corrected clamp — nruns=1 gave 0/5 diverged in 17–186 s; nruns=4 gave 0/3 diverged in 560–653 s, i.e. ~4–10× the cost for no divergence benefit, AND with Pareto k = 9.7/13.0/14.5 (≫0.7), so the importance resampling across paths was not valid anyway. High k is expected here: Pathfinder fits a NORMAL approximation in ~1000–1600 dimensions, where importance weights are near-degenerate by construction — multipathfinder is a poor fit for a model this size. Stability now comes from `stage1_z_init_scale` instead. NOTE the cache token does NOT encode the sampler, so changing this alone will silently reuse existing chains — delete them if you change it outside a token bump.
    stage1_z_init_scale::Float64 = 1.0 # SD of the N(0,σ²) initial values given to the STANDARD-NORMAL non-centred random terms (`z`, `z_c`, and `z_kappa`/`z_k`) at the start of the Stage-1 LBFGS path; ≤0 disables the explicit init and restores Pathfinder's own default (`UniformSampler(2)`, i.e. U(-2,2) per coordinate in unconstrained space). These blocks dominate Stage 1 (588 + 336 of 1580/2168 unconstrained coordinates) and are only weakly identified, so where the path STARTS largely decides where it ends. SET TO 1.0 on 2026-08-02 (user request): this is the z's OWN PRIOR, so the init is now a draw from the prior like every other latent rather than a deliberately shrunken one. WHY IT CHANGED: at τ₀=0.001 the fitted dispersion block came back with λ median 1.000, c 0.9995 and z SD 0.07 — i.e. every coordinate still sitting at its 0.1-scaled starting point, which is consistent with the RE being genuinely unidentifiable at that τ₀ but INDISTINGUISHABLE from Pathfinder never moving in those coordinates. Starting at the prior scale separates the two: if the RE still collapses from a diffuse start, the collapse is real. PRIOR VALUE 0.1 and its rationale (2026-07-30 sweep): a diffuse start costs DISTANCE — 588 coordinates each up to |2| from neutral under the old `UniformSampler(2)` default — which lengthens the path and lets early LBFGS steps swing τ before the likelihood constrains it; the observed failure then was τ collapsing to exactly 0. Non-centred parameterisation makes z=0 the exactly-neutral start (no RE at all), so a small σ was a mild perturbation off neutral and the RE had to be EARNED. That reasoning still stands as the risk of 1.0 — watch for τ→0 collapse. NOTE the cache token does NOT encode this, so changing it silently reuses existing chains: DELETE the affected `8j_s1_*` files before refitting (as was done for the τ₀=0.001 pair on 2026-08-02).
    # --- separable spatio-temporal GP smoothing of the age-pair mean (inst/1e, §5) ---
    gp_len_prior::Tuple{Float64,Float64}   = (log(15.0), 0.5)  # log-ρ Normal(μ,σ), age-years; shared by BOTH spatial diagonal length-scales (ρ_diag=total-age, ρ_gap=age-gap)
    gp_scale_prior::Tuple{Float64,Float64} = (0.0, 0.5)        # log-η Normal(μ,σ), GP marginal scale (age-pair field)
    gp_time_len_prior::Tuple{Float64,Float64}   = (log(4.0), 0.5)  # log-ρ_time Normal(μ,σ), weeks; temporal length-scale (shared across age-pairs), per-week regime only
    gp_level_scale_prior::Tuple{Float64,Float64} = (0.0, 0.5)      # log-σ_c Normal(μ,σ), amplitude of the decoupled temporal level GP c_t = c + σ_c·(Lt·z_c)
    # --- hierarchical contact-degree dispersion: block mean + REGULARISED HORSESHOE cell RE (§4.3) ---
    # τ₀ IS PER DEGREE FAMILY (2026-08-02, user request). The two families' dispersion RE sit at very
    # different scales — measured at origin 2021-05-09 h1 under τ₀=0.1: per-cell multiplier τ·λ̃ was
    # 1.61 (NegBin φ) vs 0.73 (Weibull κ), and within-block SD of log-dispersion 0.5–2.4 vs 0.05–0.42
    # — so one shared τ₀ cannot be right for both. Read them via `disp_tau0_prior(cfg, dm)`, never
    # directly, so the branch lives in exactly one place. Both are `N⁺(mean,sd)` for the GLOBAL
    # horseshoe scale τ (`tau ~ truncated(Normal(mean,sd); lower=0)`); half-Normal ⇒ already ≥0, so no
    # exp/softclamp transform. τ is ONE SCALAR FOR THE WHOLE FITTING WINDOW (it was one τ_t per week
    # before 2026-08-02 — a per-week scale is re-estimated 12× from 49 cells each and lands on
    # optimiser-path luck; tasks/lessons.md 2026-07-30) and is SHARED across the four child/adult
    # blocks (a per-block scale sees only that block's cells, and child→child has just 2×2=4).
    #
    # TUNING HISTORY (all measured at origin 2021-05-09 h1 via `src/tune_tau0.jl`; pre-horseshoe was
    # 0.5). Reported as multiplier τ·λ̃ = the per-cell RE size in log, and the within-block SD of
    # log-dispersion for adult→adult, which is what the RE actually delivers:
    #   τ₀=0.1   negbin τ·λ̃ 1.588, SD 0.877 | hweibull τ·λ̃ 0.724, SD 0.225
    #   τ₀=0.01  negbin τ·λ̃ 0.390, SD 0.830 | hweibull τ·λ̃ 0.195, SD 0.211
    # i.e. a 10× tighter τ₀ cut the nominal scale ≈4× but the REALISED spread only ≈5–12%: the
    # posterior routes around τ₀ through `z` (implied z SD rose 0.55→2.13 negbin, 0.31→1.08 hweibull),
    # since δ = τ·λ̃·z and z ~ N(0,1) is free. At τ₀=0.1 the z's were UNDER-dispersed (0.3–0.55) so
    # their own prior was not resisting at all; by 0.01 they sit at ≈1–2, i.e. the z prior has just
    # begun to bind. SET TO 0.001 (2026-08-02, user) to push well past that knee — at this scale
    # holding the same δ needs z ≈ 4–20, which its unit-normal prior charges tens to hundreds of nats
    # per cell across 588 cells, so this is where the RE should finally give way if it is going to.
    # WHY 0.1 was abandoned — measured, not assumed: at τ₀=0.1 the posterior τ came back at 1.57
    # (negbin) / 0.73 (hweibull), i.e. 7–15 prior SDs out, the slab inflated to c=29.3/2.93 until it
    # never bound, λ never left its init, and the within-block spread was essentially unchanged from
    # the pre-horseshoe flat hierarchy. The prior was simply being outbid: a posterior τ of 0.73 costs
    # ≈27 nats at τ₀=0.1 but ≈2665 nats at τ₀=0.01, so 0.01 should actually bite.
    #
    # ⚠ τ₀ CANNOT be set from Piironen & Vehtari's τ₀ = p₀/(D−p₀)·σ/√n — that is derived for a LINEAR
    # model with a residual scale σ and sample size n, neither of which exists for a NegBin /
    # hurdle-Weibull likelihood on counts and durations. A prior-predictive Monte Carlo of m_eff is
    # well-defined but describes only the PRIOR, and the pilot showed the likelihood overwhelming the
    # prior by 7–15 SDs. THE ESCAPE RATE MUST BE MEASURED FROM FITTED CHAINS — use `src/tune_tau0.jl`.
    # τ→0 recovers the block-only model exactly.
    disp_re_scale_prior_unweighted::Tuple{Float64,Float64} = (0.0, 0.001) # NegBinAgePair — dispersion φ
    disp_re_scale_prior_weighted::Tuple{Float64,Float64}   = (0.0, 0.001) # HurdleWeibullAgePair — Weibull shape κ
    disp_rhs_local_df::Float64 = 3.0   # degrees of freedom of the LOCAL scale's half-Student-t prior, `lam ~ truncated(TDist(df); lower=0)` (scale 1), one per ordered cell × week. SET TO 3, NOT the horseshoe's canonical 1 (= half-Cauchy), on user request 2026-08-02 for Pathfinder/NUTS convergence. The two differ only in tail weight, and the tail is where the sampler geometry is decided: in the unconstrained coordinate y = log λ the restoring gradient is EXACTLY −df (measured: −3.000 for df=3, −1.000 for df=1, at y = 5/10/20), so df=3 pulls a runaway local scale back 3× as hard, and the extreme tail is 227× lighter (q99.99 = 28 vs 6366). This is what stops the large-λ region becoming the flat, gradient-free trap that the κ soft-clamp created on 2026-07-30. Lowering this to 1 restores the textbook horseshoe and its sparser selection, at the cost of that geometry.
    disp_rhs_slab_scale::Float64 = 1.0 # slab scale `s` in `c² ~ InverseGamma(df/2, df·s²/2)`. SET TO 1 (user, 2026-08-02). This is the CAP on how far an escaped cell's log-dispersion may sit from its block mean: the RE multiplier τ·λ̃ = c·u/√(c²+u²) (u = τλ) is bounded above by c, so |δ| ≤ c·|z| exactly. With s=1 a fully escaped cell at |z|=2 deviates ≈±2.2 in log — a factor ≈9 — which stays well inside the composed value's soft-clamp ([-4.3,5] for log κ, [-4,5] for log φ). That bound is the reason the clamp no longer binds the way it did before the horseshoe (tasks/lessons.md 2026-07-30).
    disp_rhs_slab_df::Float64  = 4.0   # slab degrees of freedom `ν` in `c² ~ InverseGamma(ν/2, ν·s²/2)` (Piironen & Vehtari 2017 eq. 11); the marginal for a fully ESCAPED cell is t_ν(0, s). 4 is the paper's own default (user-confirmed 2026-08-02). ν→∞ pins c² at s² (a fixed slab, fewest moving parts, no slab-width learning); ν=1 gives a Cauchy slab that barely regularises. NOTE `c² = sqrt`'d in the model without an epsilon — safe because InverseGamma's exp(−νs²/2c²) factor gives an unbounded restoring gradient as c²→0 (in y = log c², dlogp/dy = −ν/2·2 + … → +∞), so c² cannot reach the sqrt's infinite-derivative point. Verify rather than assume: check c²'s 1st percentile in the 10j/11j globals panel.
    # --- secondary attack rate γ_SAR (§3.2/§6; analysis-plan per-contact SAR, non-normalised C*) ---
    gamma_sar_prior::Tuple{Float64,Float64} = (log(0.1), 1.8) # log-γ_SAR Normal(μ,σ): the per-contact secondary attack rate. C* is NOT normalised (the -gnorm C*→C*/S̄ decoupling was reverted 2026-07-12, inst/4_cut_Bayes.md), so γ_SAR reproduces the reference cell N_11 = susc₁·inf₁ = γ_SAR directly. LOOSENED 2026-07-13 to span γ_SAR∈[0.001,10] (softclamp bounds below): the earlier (log0.27, 1.05) prior [90% γ_SAR∈[0.048,1.52]] and softclamp lower bound log0.02 were actively pinning the low-γ configs — the negbin|neighbourhood posterior median (~0.021) sat right on the log0.02 clamp with an implausibly tight CI (clamp compression). New centre log(0.1) = geometric mean of [0.001,10] with log-SD 1.8 ⇒ 90% γ_SAR∈[0.0052,1.93], weakly-informative across the full range. The softclamp [log0.001,log10] now sits at ≈±2.56σ (outside the 90% band, tails ≈0.5% each), so it comfortably contains the prior and stops biasing the low tail. NOTE: this change invalidates cached 8j_s2_* Stage-2 chains (the contacts token does not encode the prior) — delete them and re-run prefit_stage2! to regenerate; Stage-1 8j_s1_* chains are γ_SAR-independent and unaffected.
    # --- independent per-bin marginal SD of the relative susc/inf age profile (Stage-2 transmission block; NO cross-bin smoothing) ---
    susc_inf_sd_prior::Tuple{Float64,Float64} = (0.0, 0.25) # N⁺(mean,sd) marginal-SD prior for the susc/inf offset sig·z (INDEPENDENT per age bin — the shared-length-scale RBF GP `sig·(Lsi·z)` was removed 2026-07-13, user request); used SEPARATELY by sig_s (susc) and sig_i (inf) — two distinct latents, one shared prior form. SET 2026-07-31 to N⁺(0,0.25²) (user request): a mode-at-0 half-normal — the conventional weakly-informative scale prior, so the age profile can SHRINK to no-variation (susc/inf ≡ reference) when the data carry no age signal, rather than being asserted at ≈0.5 log-SD. This differs from the 2026-07-13 loosening N⁺(0.1,0.05²)→N⁺(0.5,0.25²), which centred the scale at 0.5 (mode away from 0) and could not shrink flat — questionable for infectivity, whose age signal is weak. Upper reach ≈ unchanged (marginal SD still ≤~0.5, so realistic profiles stay well inside the clamp). Soft-clamp [log0.05,log20] (≈[-3,3]) ⇒ hard-bounds susc/inf ∈ [0.05,20] — unchanged. (The `susc_inf_gp_len_prior` length-scale field was dropped with the GP.)
    # --- susc/inf reference age bin (the gauge: susc[ref]=inf[ref]≡1, only the A-1 other bins estimated) ---
    ref_bin::Int = 4 # index of the CIS age bin fixed to 1 in the relative susc/inf profile. SET 2026-07-31 to 4 = "25-34" (user request), moved off the former 1 = "2-10": children are an extreme, poorly-identified, antibody-sparse anchor, whereas 25-34 is a large well-mixed adult group (Munday/Davies convention). The NGM likelihood is GAUGE-INVARIANT to this choice (rescaling all susc by c and γ_SAR by 1/c leaves N unchanged), so switching the reference acts ONLY through the priors — which bin is pinned vs. carries the log-offset, and what γ_SAR = N_{ref,ref} anchors to. Because it changes the Stage-2 posterior but the `contacts_label` cache token does not encode it, stale `8j_s2_*` chains must be regenerated after a change (Stage 1 is unaffected — it has no susc/inf).
end

"""
    disp_tau0_prior(cfg, dm) -> (mean, sd)

The `N⁺(mean, sd)` prior for the horseshoe's global scale τ, for THIS degree family. The single
access point for `disp_re_scale_prior_unweighted` / `_weighted` — the model, the 10j/11j prior bands
and the tuning harness all go through it, so the family branch exists once. Reading the fields
directly is a bug waiting to happen: the two are tuned independently and diverge.
"""
disp_tau0_prior(cfg::FrameworkConfig, dm::ContactDegreeModel) =
    is_weighted(dm) ? cfg.disp_re_scale_prior_weighted : cfg.disp_re_scale_prior_unweighted

# τ₀ formatted for the cache token. `contacts_label` does not otherwise encode the dispersion priors,
# so WITHOUT this a tuning refit of the same family writes the SAME filename and
# `fit_or_load_stage1`'s `isfile` short-circuit silently reloads the stale chain — the exact footgun
# tasks/lessons.md records twice (γ_SAR 2026-07-13, stage1_pathfinder_runs 2026-07-30). Encoding both
# values makes every tuning step non-colliding by construction and keeps prior steps side by side on
# disk for comparison. Dots are fine in these filenames (they already end `.jld2`).
_tau0_tag(cfg::FrameworkConfig) =
    string(cfg.disp_re_scale_prior_unweighted[2], "-", cfg.disp_re_scale_prior_weighted[2])

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
under `-sc`.)

`-hd-p0-gi` (2026-07-30, inst/5_formal_pathfinder_impl.md — the FORMAL model). Three parameter-space
changes across both stages: `-hd` the **h**ierarchical **d**ispersion (Stage 1 gains `tau` [one per
week, shared across blocks] and `z_kappa`/`z_k` [A²×Tn per-cell random terms]; `log_kappa`/`log_k` are
reinterpreted as block MEANS); `-p0` the **fitted hurdle zero probability** (Stage 1, weighted path
ONLY, gains `p0f` [A²×Tn] — so the two degree models' chains now differ in shape); `-gi` the estimated
**g**eneration **i**nterval (Stage 2 gains `w_mu`/`w_sigma`, and the pooled file gains the matching
per-draw vectors). ALL 504 `8j_s1_*` and 1008 `8j_s2_*` files under `…-sc` are stale — a full refit is
required. The token is SHARED by `stage1_path` and `stage2_path`, so Stage 1 cannot be spared even
though `-gi` is a Stage-2-only change. Regenerate the derived 9j caches (`9j_rt_*`, `9j_relrt_*`,
`9j_obsrt_*`) and any stored forecast assembly too: the forecast itself changed (per-draw `w`,
antibody at t₀+h) independently of the chains.

`-hd` → `-rhs` (2026-08-02, user request — the **r**egularised **h**orse**s**hoe). The Stage-1
dispersion RE keeps its non-centred block-mean + per-cell form but its scale is rebuilt as
Piironen & Vehtari (2017) eq. 11: `tau` collapses from a length-`Tn` per-week vector to a SINGLE
window-level scalar, and two new latents appear — `lam` (the per-ordered-cell × week local scale,
`A²×Tn`, half-t₃) and `c2` (the scalar slab, `InverseGamma(ν/2, νs²/2)`). Stage 1's unconstrained
dimension goes 1002→1580 (NegBin) and 1590→2168 (hurdle-Weibull). Stage 2 is untouched, but shares
the token, so its chains are re-keyed too.

**The `-hd-` artefacts are deliberately RETAINED, in `dt_intermediate_hierarchical/`** (2022 files:
504 `8j_s1_*`, 1512 `8j_s2_*`, plus the derived `9j_*` caches). Unlike previous bumps this is not a
"delete the stale files" migration: `src/11j_viz_utils.jl` reads both generations side by side to
compare the half-Normal RE against the horseshoe. Note they differ in BOTH token and directory, so a
cross-generation read needs `CONTACTS_TOKEN_HD` **and** `CONTACTS_SAVE_DIR_HD` — passing the token
alone against the default `save_dir` silently finds nothing. Existing WIS/log scores continue to
describe the `-hd-` chains until a full refit is run; as of the 2026-08-02 pilot only origin
2021-05-09 h1 exists under `-rhs`, in `dt_intermediate/`."""
contacts_label(cfg::FrameworkConfig) =
    (cfg.constant_contacts ? "pooled" : "temporal") * "-gsar-cut-sc-rhs" * _tau0_tag(cfg) * "-p0-gi"

"""Default contacts token for the read-only viz helpers that do NOT receive a `cfg`
(`stage1_chain_path`, `reconstruct_p0_draws`, `reconstruct_tau_draws`, …), so a token bump lands in
one place instead of the seven hard-coded copies that previously had to be edited in lockstep.
Resolves to the per-week (`constant_contacts=false`) regime — the setting every notebook uses.

⚠ It is a load-time constant built from the **default** `FrameworkConfig`, and since 2026-08-02 the
token encodes τ₀ (`_tau0_tag`). So during τ₀ tuning it goes stale the moment you construct a `cfg`
with non-default τ₀. Every helper that takes a `cfg` therefore defaults to `contacts_label(cfg)`
instead of this — prefer that. Use this only where no `cfg` is in scope, and pass `contacts = …`
explicitly when tuning (see `CONTACTS_TOKEN_HD` for the previous generation)."""
const CONTACTS_TOKEN = contacts_label(FrameworkConfig(constant_contacts = false))

"""The PREVIOUS generation's token (`-hd`, 2026-07-30 → 2026-08-02): flat hierarchical dispersion
with a per-week half-Normal τ_t and no horseshoe. Those chains are still on disk and are read side
by side with `CONTACTS_TOKEN` by the 11j old-vs-new shrinkage comparison, so this is a live constant,
not a historical note. Only the dispersion mirrors understand it — `stage1_moment_draws` does NOT
(see `_read_disp_chain`)."""
const CONTACTS_TOKEN_HD = "temporal-gsar-cut-sc-hd-p0-gi"

"""Where the `-hd` generation's chains live. They were moved out of `dt_intermediate/` into
`dt_intermediate_hierarchical/` (2022 files: 504 `8j_s1_*`, 1512 `8j_s2_*`, plus the derived `9j_*`
caches), so a cross-generation read needs BOTH a different token and a different directory — passing
`CONTACTS_TOKEN_HD` alone against the default `save_dir` silently finds nothing. Path is relative to
`src/`, matching the rest of the framework's `cwd == src/` convention."""
const CONTACTS_SAVE_DIR_HD = joinpath(@__DIR__, "..", "dt_intermediate_hierarchical")

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
`antibody` :: (A × T) weekly antibody prevalence A_a(t) ∈ [0,1] over `weeks` (= `win.all_weeks`).
`antibody_fc` :: (A × H) antibody at the FORECAST TARGET weeks `win.forecast_weeks` (t₀+h).
`pop`, `prop` :: (A) populations and proportions. `weeks` chronological (all_weeks).

Antibody is deliberately carried in TWO fields rather than one widened matrix (2026-07-30, §3.2):
`antibody[:, t]` keeps its exact meaning inside the Stage-2 fit loop, which is anchored at t₀,
while `antibody_fc[:, hi]` is what the *forecast* NGM uses — matching the contact window that
already ends at t₀+h. Widening `antibody` would have made `[:, end]` silently mean something new
at every existing call site.
"""
struct WindowData
    weeks::Vector{Date}
    A::Int
    I_mean::Matrix{Float64}
    I_sd::Matrix{Float64}
    antibody::Matrix{Float64}
    antibody_fc::Matrix{Float64}
    pop::Vector{Float64}
    prop::Vector{Float64}
    labels::Vector{String}
end
