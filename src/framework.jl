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
"""
Hard soft-clamp bounds for the GP length-scales, as `(lo, hi)` on the LOG scale.

SINGLE SOURCE OF TRUTH — do not re-spell these as literals. They are consumed by `model_degree`
(`src/joint_model.jl`) *and*, independently, by the read-only viz layer, which RECONSTRUCTS ρ and
`C*` from a raw chain by re-applying the same `_softclamp`: `8j_viz_utils.jl` (`reconstruct_*`) and
`10j_viz_utils.jl` (`reconstruct_mu_draws`). If the model's window moves and a viz site does not,
every reconstructed length-scale and contact matrix is silently wrong and NOTHING raises. That is
why these were hoisted out of five separate literals on 2026-08-05.

⚠ **`RHO_TIME_BOUNDS` IS DEAD CODE since `-ar1` (2026-08-06)** and is kept only so archived
pre-`-ar1` chains can still be replayed by the read-only mirrors. The temporal correlation is now
AR(1), `Kt[s,t] = φ^|s−t|` with `φ ~ Beta` — `φ ∈ (0,1)` by construction (Turing's bijector) and `φ^k`
cannot overflow, so there is nothing for a clamp to guard. `model_degree` no longer references it.
Do not "restore" a temporal clamp: it would be a modelling constraint disguised as a numerical guard,
which is exactly the mistake `RHO_BOUNDS` was widened to undo (see below).

`RHO_BOUNDS` (spatial) is still live and **INERT under the current prior**, kept as an overflow guard:
`gp_len_prior` N(log 20, 0.35²) puts its floor 10.5σ below and its ceiling 9.2σ above, so the clamp
cannot bind. That is the intended division of labour — the PRIOR restrains the length-scale, the clamp
only stops a stray optimiser step from overflowing `exp`.

⚠ That was NOT true of the previous generation, and the failure was measured. With
`gp_time_len_prior` at N(log 4, 0.5²) the four-cell NUTS pilot of 2026-08-05 put ρ_time at 24.1,
34.8, 48.0 and 62.7 weeks — +3.6σ to +5.5σ into the prior's tail, over a **12-week** window. The
docstring here previously asserted "the ceiling cannot be reached and the restraint is the prior";
that was an assertion, not a measurement, and the posterior went most of the way to the ceiling
anyway. The lesson stands independent of the numbers: **a length-scale is only restrained by its
prior if the likelihood is informative about it, and over 12 weeks it is not.** If a refit shows
ρ_time drifting up the tail again, the answer is a tighter prior or a shorter window — not a wider
clamp.

WIDENED 2026-08-05, `RHO_BOUNDS` `[log 3, log 45]` → `[log 0.5, log 500]`. The old window was
2.708 nats wide, which is narrower than `_softplus`'s transition width, so `_softclamp` had **no
interior** there (max derivative 0.600 anywhere) and was acting as a hard modelling constraint
disguised as a numerical guard — see the `_softclamp` note in `joint_model.jl`. It is not a
numerical guard: under `-m32` (2026-08-05) `cholesky(Symmetric(Ap) + 1e-6·I)` is clean at all 625
points of a 25×25 (ρ_diag, ρ_gap) grid spanning the whole window, `min eigval(Kp) = 1.4e-7` (PSD),
and `rank(Ap) = 27` everywhere INCLUDING the ρ = 500 ceiling — Matérn 3/2's tails are heavy enough
that `Kp → J` is not reached inside the clamp at all (min(Kp) is still 0.93 at ρ = 500; rank first
degrades around ρ ≈ 5000). As ρ→0, `Kp → I`. The real limit is identifiability, and that is held by
`gp_len_prior`, not by this clamp.

`RHO_TIME_BOUNDS` `[log 0.5, log 26]` → `[log 0.25, log 104]`, widened 2026-08-05 after a Pathfinder
posterior survey (4 origins × 2 degree models × 2 NGMs) found ρ_time **pinned against the old
ceiling**: median 22.8, max 25.87 against a bound of 26.0. The widening was correct as far as it went
— a clamp absorbing the likelihood's preference reports false confidence — but it did not identify
ρ_time, it only let the drift become visible. `gp_time_len_prior` N(log 2, 0.35²) is what actually
bounds it now. Cholesky is safe regardless: `cholesky(Kt + 1e-4·I)` succeeds to ρ_time = 1e6.

⚠ Widening is not free, and this is the mechanism to understand before touching either bound: a large
ρ_time drives `Kt` toward rank-1, `Lt`'s leading column then absorbs the whole field, and `z[:,1]`
becomes pinned ~60× tighter than the other columns — the geometry that produced 100% max-tree-depth
and min ESS 1.9–5.4 of 500 (tasks/lessons.md 2026-08-05). The `-m32` kernel swap attacks the same
mechanism from the other side (Matérn keeps `Kt` full rank 12 at every ρ_time in this window), but
the two are complementary, not redundant. Do not widen past the point where the model can still
discriminate.

⚠ PROVENANCE: the ρ_time survey was run with PATHFINDER (`stage1_use_nuts = false`) because it is
~30 s vs ~1 h per fit. In a flat direction crossed with a saturated clamp, Pathfinder's normal
approximation has no curvature to fit and can come back arbitrarily wide, so its raw ρ_time ≈ 1047 is
partly an artefact and should NOT be quoted as a posterior. The four NUTS values quoted above are the
trustworthy ones.
"""
const RHO_BOUNDS      = (log(0.5), log(500.0))   # ρ_diag / ρ_gap — age-years (shared window)
const RHO_TIME_BOUNDS = (log(0.25), log(104.0))  # ρ_time — weeks. DEAD since `-ar1`: kept only for replaying archived pre-`-ar1` chains

"""
Soft-clamp bounds and transition width for the estimated generation interval (`model_transmission`,
`src/joint_model.jl`). `w_mu` is the meanlog in WEEKS; `w_sigma` is the LOG-VARIANCE (so
`sdlog = √w_sigma`, see `gen_interval_logparams`).

WIDENED 2026-08-05, and this one was a REAL BIAS, not just a mixing problem. The old box
(`w_mu ∈ [log 1/7, log 3]`, `w_sigma ∈ [0.02, 4]`) is only 3.04 / 3.98 wide, and `w_sigma`'s prior
mode sits just 0.673 above its floor — under the old O(1) transition width the clamp displaced
`w_sigma` by **2.8 prior SDs**. At the INTENDED prior centre (`gen_mean_days = gen_sd_days = 5.0`)
the model was therefore using a generation interval of **6.91 d mean / 9.65 d sd**, not 5.0/5.0 —
a +38% / +93% inflation, present at every draw. The old comment claiming these bounds sit "far
outside the prior's ±2 SD" was simply not true of `w_sigma`. Since `w` and `γ_SAR` are confounded
(both scale the renewal predictor, `gamma_sar_prior`), that bias was being absorbed into γ_SAR.
With this box the effective prior reproduces 5.000 d / 5.000 d exactly and tracks the raw latents
to <0.06% across ±3 prior SDs.

THE UPPER BOUND ON `w_mu` IS LOAD-BEARING AND log(3) IS THE MAXIMUM — DO NOT WIDEN IT.
`gen_interval_pmf_log` divides by `F(smax) = F(4)`. `min F(4)` over the box is attained at exactly
this corner (`w_mu = log 3`, `w_sigma = 4`) and equals **0.5572**. Raising the bound collapses it:
`log 4 ⇒ 0.500`, `log 6 ⇒ 0.0021` (and 0.000 at small `w_sigma`), i.e. `w ./ F(4)` divides by ~0.
The LOWER bound is free to move (small meanlog ⇒ mass at short lags ⇒ `F(4) → 1`), so only it was
widened. `w_sigma`'s ceiling is likewise held at 4.0: it never binds (prior +4σ = 1.25) and raising
it to 8.0 would drop `min F(4)` to 0.5405, eroding the ≥0.55 guarantee for nothing.

`W_GI_SOFT` is a TIGHTER transition width than `_softclamp`'s 0.25 default, and it is needed here
because `w_sigma` is a VARIANCE: its floor is pinned near 0 by physics, so the distance from the
prior mode to the bound cannot be bought by widening (0.02 → 0.0002 buys 0.02, moving the clamped
mode only 0.7095 → 0.7083). Shrinking `s` is the only lever that reaches it: at 0.05 the gradient
at the mode is 1.000000 and the clamped value equals the raw latent to 6 d.p.
"""
const W_MU_BOUNDS    = (log(1 / 28), log(3.0))   # meanlog, weeks — UPPER bound is the F(4) guard
const W_SIGMA_BOUNDS = (0.002, 4.0)              # LOG-VARIANCE (sdlog = √w_sigma)
const W_GI_SOFT      = 0.05                      # transition width; tighter than the 0.25 default

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
    stage1_use_nuts::Bool = true      # Stage-1 sampler: true = NUTS (THE DEFAULT since 2026-08-05, user request — "use only NUTS in stage 1"), false = Pathfinder (the preliminary generation that produced the 504-file `-gi` grid). UNLIKE every other field in this block, this one IS encoded in the cache token (`contacts_label` appends `-nuts`), so flipping it does NOT silently reuse the Pathfinder chains — see the `-nuts` note in `contacts_label`. NUTS is initialised from the Pathfinder mean (`_pf_mean_init`), so Pathfinder still runs first and the cost is ADDITIVE. **STAGE 2 IS UNAFFECTED AND HAS NO NUTS PATH AT ALL** — `fit_stage2_pooled` only ever calls `pathfinder` (100 cheap fits per Stage-1 draw); that is by design (inst/4_cut_Bayes.md), not an oversight.
    ad_backend::Symbol    = :mooncake # AD backend for BOTH stages' gradients: :mooncake (default) | :reversediff | :forwarddiff. Resolved once by `_resolve_adtype`; see `ad_type`. DEFAULTED TO MOONCAKE 2026-08-05 on measurement, not preference — at origin 2021-05-09, gradients/s Mooncake vs ReverseDiff: Stage-1 negbin (402 dims) 482 vs 44 (10.9×), Stage-1 hurdle-Weibull (990 dims) 241 vs 27 (9.0×), Stage-2 transmission (18 dims) 30 685 vs 1 711 (17.9×). Gradients agree with ReverseDiff to ≤4e-14 relative on all three. Mooncake pays a one-off `build_rrule` cost per model TYPE per process (66 s negbin / 14 s hurdle-Weibull / 15 s Stage 2), which is nothing against the ~1e5 gradient evaluations a single Stage-1 NUTS fit needs — but it is why `prefit_stage1!` warms one fit per degree-model type BEFORE its thread fan-out. NOT encoded in the cache token (AD is a numerical means, not a model change); the backend is recorded inside each artefact instead — see `contacts_label`.
    # --- Stage-1 NUTS settings (2026-08-05; consulted only when `stage1_use_nuts`) ---
    # These exist because a bare `NUTS()` derives `n_adapts = min(1000, n_sample ÷ 2)`, which at the
    # former `n_sample = 250` gave 125 warmup iterations to adapt a step size and diagonal metric in
    # 389 (NegBin) / 977 (hurdle-Weibull) dimensions. Stan's default is 1000; 125 is not a tuning
    # choice, it is an accident of the convenience constructor.
    stage1_nuts_adapts::Int = 1000    # warmup iterations, DISCARDED and drawn ON TOP of `stage1_nuts_draws` (AbstractMCMC applies `discard_initial` before collecting N, so total work = adapts + draws).
    stage1_nuts_draws::Int  = 500     # KEPT draws per Stage-1 fit. Must stay ≥ `n_stage1_post` (=100) or `stage1_moment_draws` cannot fill the cut's 100 imputations; `fit_stage1` enforces that with a `max`.
    stage1_nuts_target_accept::Float64 = 0.95  # above NUTS' 0.65 default: the non-centred GP (`z`, `z_c`) crossed with the soft-clamped exponentials is moderately curved, and the clamp's flat region is exactly where a too-large step lands. RAISED 0.9→0.95 on 2026-08-06 (user request) because the `-t0` refit left divergences in the hurdle-Weibull cells (4 of 500 at 2020-11-15, 5 of 500 at 2021-05-09; NegBin 0 in both). WHY THIS IS THE RIGHT LEVER HERE and not elsewhere: the project's own rule (12j "divergence pairs") is that SCATTERED divergences mean the step size is slightly too large — `target_accept` — while CLUSTERED ones mean a funnel and want a parameterisation fix. Measured: the divergent draws' percentile ranks within `log_sigma_c`/`log_eta`/`log_rho_diag`/`log_rho_gap`/`log_rho_time`/‖z‖ span 46–94 points in every coordinate, i.e. no tail bunching ⇒ scattered. ⚠ Do NOT generalise this to a MIXING complaint: `tasks/lessons.md` records that `target_accept` is *not* the lever for low ESS (that was the `z[·,1]` boundary-week geometry), and raising it costs step size and therefore wall-clock. `-t0` bought the headroom to pay that (tree depth 6.98–7.00 against a cap of 10, fits 2.2–3.7× faster), which is what makes 0.95 affordable now and would not have been before.
    stage1_nuts_max_depth::Int = 10   # explicit rather than implicit so `_nuts_diagnostics` can report the saturating fraction against a known ceiling.
    stage1_pathfinder_runs::Int = 1   # Stage-1 Pathfinder paths. >1 ⇒ `multipathfinder` (independent LBFGS runs pooled by Pareto-smoothed importance resampling); 1 ⇒ single-path. RESET TO 1 on 2026-07-30 (user request, and the measurement agrees). It was briefly 4, to insure against the single-path divergence seen BEFORE the κ clamp was corrected to [-4.3,5]. Once the clamp was fixed the premise vanished: measured head-to-head on hurdle-Weibull, 5 seeds, corrected clamp — nruns=1 gave 0/5 diverged in 17–186 s; nruns=4 gave 0/3 diverged in 560–653 s, i.e. ~4–10× the cost for no divergence benefit, AND with Pareto k = 9.7/13.0/14.5 (≫0.7), so the importance resampling across paths was not valid anyway. High k is expected here: Pathfinder fits a NORMAL approximation in ~1000–1600 dimensions, where importance weights are near-degenerate by construction — multipathfinder is a poor fit for a model this size. Stability now comes from `stage1_z_init_scale` instead. NOTE the cache token does NOT encode THIS field (it encodes only `stage1_use_nuts`, since 2026-08-05), so changing it alone will silently reuse existing chains — delete them if you change it outside a token bump.
    stage1_z_init_scale::Float64 = 1.0 # SD of the N(0,σ²) initial values given to the STANDARD-NORMAL non-centred random terms (`z`, `z_c`) at the start of the Stage-1 LBFGS path; ≤0 disables the explicit init and restores Pathfinder's own default (`UniformSampler(2)`, i.e. U(-2,2) per coordinate in unconstrained space). These blocks dominate Stage 1 (`z` 324 + `z_c` 11 = 335 of 389/977 unconstrained coordinates) and are only weakly identified, so where the path STARTS largely decides where it ends. SET TO 1.0 on 2026-08-02: this is the z's OWN PRIOR, so the init is a draw from the prior like every other latent rather than a deliberately shrunken one. HISTORY: it was 0.1 while the dispersion carried a per-cell random effect (`z_kappa`/`z_k`, 588 further coordinates) — a diffuse start over that many weakly-identified coordinates lengthened the path and let early LBFGS steps swing the RE scale before the likelihood constrained it. That RE was removed on 2026-08-02 (dispersion is now block-linear × week only), so the argument for shrinking the init no longer applies and only the GP's own `z`/`z_c` remain. NOTE the cache token does NOT encode this, so changing it silently reuses existing chains: DELETE the affected `8j_s1_*` files before refitting.
    # --- separable spatio-temporal GP smoothing of the age-pair mean (inst/1e, §5) ---
    gp_len_prior::Tuple{Float64,Float64}   = (log(20.0), 0.35)   # log-ρ Normal(μ,σ), age-years — SHARED by BOTH spatial length-scales, `log_rho_diag` (total age) and `log_rho_gap` (age gap). SET 2026-08-05 (user request) alongside the `-m32` kernel swap and the restoration of the off-diagonal smoothing term. ⚠ UNITS: ρ lives on the rotated (su, df) scale, which is √2× an age difference, so ρ=20 is an effective age-difference length-scale of 20/√2 = 14.1 yr. WHAT IT ASSERTS: at the mode the kernel gives correlation 0.788 to the closest diagonal step (2-10→11-15, Δu=9.90) and 0.028 across the widest age gap — a MUCH smoother contact surface than the (log 4, 0.5) it replaces, which gave 0.047 at that same step, i.e. barely smoothed at all at its own mode. CONDITIONING (measured, kernel_static): under Matérn 3/2 the projected kernel `Ap` has rank 27 and min eigenvalue 2.5e-2 at the mode, 2.1e-3 at +2σ — 3-4 orders above the 1e-6 jitter, with effective rank 16.8/8.3/4.4 at −2σ/mode/+2σ. Under the OLD squared exponential the same ρ=20 gave min eigenvalue 2.9e-5 and, at +2σ, 1.5e-8, i.e. BELOW the jitter: the kernel swap is what makes this centre numerically safe, and the two changes should not be separated. ⚠ IN TENSION WITH THE EARLIER POSTERIOR, DELIBERATELY: the 2026-08-05 Pathfinder survey put ρ_diag≈7.9 and ρ_gap≈4.65, which are −2.7σ and −4.2σ here, so this prior is informative rather than weak. WATCH AT THE REFIT: if the posterior piles up against the LOWER edge, the data are disagreeing with the assumed smoothness and the centre should come down. HISTORY worth keeping: the SD was briefly 0.75 (with centre log 4) and that MEASURABLY BROKE Stage-1 NUTS — at 2021-05-09 h1 negbin, min ESS fell 118→1.8 of 500, step size 1.51e-2→1.17e-3, tree depth 8.40→10.00 (100% at cap), max R̂ 1.027→1.597. The worst blocks were log_rho_time and the z_c/z field it couples to through Lt: a looser length-scale prior let ρ_time drift into the near-pooled region where Kt goes low-rank, collapsing the map z↦Fld. That mechanism is exactly what `-m32` targeted, and what `-ar1` (2026-08-06) targets more directly still — AR(1)'s Markov spectrum keeps `Kt` well conditioned even in the near-pooled limit, so the temporal side is no longer restrained by a prior at all (`ar1_phi_prior` is Uniform). ⚠ That does NOT license loosening THIS spatial SD: the 0.75 regression above was measured, and the spatial kernel is unchanged.
    gp_scale_prior::Tuple{Float64,Float64} = (0.0, 0.5)        # log-η Normal(μ,σ), GP marginal scale (age-pair field)
    ar1_phi_prior::Tuple{Float64,Float64}   = (1.0, 1.0)   # Beta(a,b) on the AR(1) temporal coefficient φ ∈ (0,1) — REPLACES `gp_time_len_prior` under `-ar1` (2026-08-06, user request): the temporal correlation is now `Kt[s,t] = φ^|s−t|` (AR(1) ≡ exponential ≡ Matérn 1/2), not a Matérn 3/2 length-scale in weeks. (1.0, 1.0) is Uniform(0,1) — DELIBERATELY WEAK, and the first thing to revisit. WHY WEAK: tail-matching φ to the ρ_time prior it replaces gives Beta(10.22, 3.395) (90% interval [0.5445, 0.9137], reproducing the implied lag-1 correlation to 6.8e-05), but that puts P(φ>0.95) at 0.010 while hurdle-Weibull's likelihood wants φ ≈ 0.99+ — precisely the prior–likelihood conflict the reverted `-ig` experiment diagnosed as costing ESS in EVERY block (rho_time's own ESS tripled, 76–80 → 188–278, once the prior stopped fighting it; coordinates below ESS 100 fell 38 → 10). The numerical case for a tight temporal prior is also much weaker now: AR(1) is still well conditioned at φ = 0.995 (Lt column spread 33 against Matérn 3/2's 183 at comparable pooling), which is what makes a weak prior affordable here and was NOT true before. ⚠ φ→1 is the pooled/degenerate limit; the 1e-4 jitter on `Kt` is what keeps the Cholesky safe there, so do not reduce it.
    gp_level_scale_prior::Tuple{Float64,Float64} = (0.0, 0.5)      # log-σ_c Normal(μ,σ), amplitude of the decoupled temporal level GP c_t = c + σ_c·(Lt·z_c)
    # --- secondary attack rate γ_SAR (§3.2/§6; analysis-plan per-contact SAR, non-normalised C*) ---
    gamma_sar_prior::Tuple{Float64,Float64} = (log(0.1), 1.8) # log-γ_SAR Normal(μ,σ): the per-contact secondary attack rate. C* is NOT normalised (the -gnorm C*→C*/S̄ decoupling was reverted 2026-07-12, inst/4_cut_Bayes.md), so γ_SAR reproduces the reference cell N_11 = susc₁·inf₁ = γ_SAR directly. LOOSENED 2026-07-13 to span γ_SAR∈[0.001,10] (softclamp bounds below): the earlier (log0.27, 1.05) prior [90% γ_SAR∈[0.048,1.52]] and softclamp lower bound log0.02 were actively pinning the low-γ configs — the negbin|neighbourhood posterior median (~0.021) sat right on the log0.02 clamp with an implausibly tight CI (clamp compression). New centre log(0.1) = geometric mean of [0.001,10] with log-SD 1.8 ⇒ 90% γ_SAR∈[0.0052,1.93], weakly-informative across the full range. The softclamp [log0.001,log10] now sits at ≈±2.56σ (outside the 90% band, tails ≈0.5% each), so it comfortably contains the prior and stops biasing the low tail. NOTE: this change invalidates cached 8j_s2_* Stage-2 chains (the contacts token does not encode the prior) — delete them and re-run prefit_stage2! to regenerate; Stage-1 8j_s1_* chains are γ_SAR-independent and unaffected.
    # --- independent per-bin marginal SD of the relative susc/inf age profile (Stage-2 transmission block; NO cross-bin smoothing) ---
    susc_inf_sd_prior::Tuple{Float64,Float64} = (0.0, 0.25) # N⁺(mean,sd) marginal-SD prior for the susc/inf offset sig·z (INDEPENDENT per age bin — the shared-length-scale RBF GP `sig·(Lsi·z)` was removed 2026-07-13, user request); used SEPARATELY by sig_s (susc) and sig_i (inf) — two distinct latents, one shared prior form. SET 2026-07-31 to N⁺(0,0.25²) (user request): a mode-at-0 half-normal — the conventional weakly-informative scale prior, so the age profile can SHRINK to no-variation (susc/inf ≡ reference) when the data carry no age signal, rather than being asserted at ≈0.5 log-SD. This differs from the 2026-07-13 loosening N⁺(0.1,0.05²)→N⁺(0.5,0.25²), which centred the scale at 0.5 (mode away from 0) and could not shrink flat — questionable for infectivity, whose age signal is weak. Upper reach ≈ unchanged (marginal SD still ≤~0.5, so realistic profiles stay well inside the clamp). Soft-clamp [log0.05,log20] (≈[-3,3]) ⇒ hard-bounds susc/inf ∈ [0.05,20] — unchanged. (The `susc_inf_gp_len_prior` length-scale field was dropped with the GP.)
    # --- susc/inf reference age bin (the gauge: susc[ref]=inf[ref]≡1, only the A-1 other bins estimated) ---
    ref_bin::Int = 4 # index of the CIS age bin fixed to 1 in the relative susc/inf profile. SET 2026-07-31 to 4 = "25-34" (user request), moved off the former 1 = "2-10": children are an extreme, poorly-identified, antibody-sparse anchor, whereas 25-34 is a large well-mixed adult group (Munday/Davies convention). The NGM likelihood is GAUGE-INVARIANT to this choice (rescaling all susc by c and γ_SAR by 1/c leaves N unchanged), so switching the reference acts ONLY through the priors — which bin is pinned vs. carries the log-offset, and what γ_SAR = N_{ref,ref} anchors to. Because it changes the Stage-2 posterior but the `contacts_label` cache token does not encode it, stale `8j_s2_*` chains must be regenerated after a change (Stage 1 is unaffected — it has no susc/inf).
end

"""
    _resolve_adtype(sym::Symbol) -> ADTypes.AbstractADType

Map `cfg.ad_backend` to its concrete `ADTypes` object — the SINGLE SOURCE OF TRUTH that replaced
nine duplicated `adtype = AutoReverseDiff()` kwarg defaults in `joint_model.jl` (2026-08-05).

The `Auto*` names arrive via **Turing's re-export** (`Turing.jl` exports `AutoForwardDiff`,
`AutoReverseDiff`, `AutoMooncake`), which is already how `AutoReverseDiff()` was reached here —
`ADTypes` is not a direct dependency and does not need to be.

`AutoMooncake()` is `Base.@kwdef` with `config = nothing`, the documented default configuration; it
constructs to `AutoMooncake{Nothing}` and does NOT require `Mooncake` to be in scope. Mooncake must
still be LOADED (it is, unconditionally, in `main_utils.jl`) or the failure surfaces much later and
far less legibly, inside `DifferentiationInterface.prepare_gradient`.

`:enzyme` is deliberately absent: Enzyme was a dependency until 2026-08-05, referenced by no code,
and was removed with the Mooncake switch. Re-adding it means a `Project.toml` dep AND a `using`.
"""
function _resolve_adtype(sym::Symbol)
    sym === :mooncake    && return AutoMooncake()
    sym === :reversediff && return AutoReverseDiff()
    sym === :forwarddiff && return AutoForwardDiff()
    error("_resolve_adtype: unknown ad_backend $(sym) — expected :mooncake, :reversediff or :forwarddiff")
end

"""`ad_type(cfg)` — the AD backend object for `cfg`, used by BOTH stages.

One backend covers everything because Stage-1 Pathfinder, Stage-1 NUTS and Stage-2 Pathfinder all
build the *same* `DynamicPPL.LogDensityFunction(model, getlogjoint_internal, linked_vi; adtype)`,
and because Mooncake measured faster than ReverseDiff on all three targets (see `ad_backend`) —
there was no case for a per-stage split, so there is no `ad_backend_stage2`. Add one only if a
future measurement actually disagrees across stages."""
ad_type(cfg::FrameworkConfig) = _resolve_adtype(cfg.ad_backend)

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

`-hd` → `-rhs` → **NO SUFFIX** (2026-08-02). The per-cell dispersion random effect was briefly a
regularised horseshoe (`-rhs`, Piironen & Vehtari 2017 eq. 11: window-global `tau`, per-cell-per-week
`lam`, scalar slab `c2`) and has now been REMOVED ENTIRELY, along with the flat `-hd` hierarchy that
preceded it. Dispersion is back to the plain block-linear × week array `log_k`/`log_kappa` (`4×Tn`),
with NO per-cell term: `log d_{ij,t} = β[bl,t]`. Why: measured at origin 2021-05-09 h1 across
τ₀ ∈ {0.1, 0.01, 0.005, 0.001}, the horseshoe either had its prior outbid by the likelihood (τ landing
7–15 prior SDs out, the slab inflating until it never bound, λ never leaving its init) or, at 0.001,
extinguished the RE outright — a cliff rather than a usable shrinkage dial. Stage 1's unconstrained
dimension goes 1580→402 (NegBin) and 2168→990 (hurdle-Weibull). The token drops `-rhs<τ₀>` and keeps
`-p0-gi`, which is what distinguishes it from the pre-2026-07-30 `…-sc` generation: the fitted hurdle
p⁰ and the sampled generation interval are BOTH retained (see `inst/3` §4.2 and the `-gi` note above),
so `temporal-gsar-cut-sc-p0-gi` is NOT the same model as `temporal-gsar-cut-sc`.

`-nuts` (2026-08-05) — a SAMPLER component, appended when `cfg.stage1_use_nuts`. It is the first
token component that does not describe the model's parameter space: Stage 1's posterior is the
same target either way, but Pathfinder only *approximates* it with a single multivariate normal in
389/977 dimensions, so the draws differ and everything conditioned on them differs with it. The
component is added because `fit_or_load_stage1` short-circuits on bare `isfile`, so without it
flipping `stage1_use_nuts` would silently reload the 504 Pathfinder chains and change nothing —
the hazard that was already documented on the `stage1_pathfinder_runs` field.

The token is SHARED with `stage2_path`, and that is correct here (unlike the 2026-07-13 γ_SAR case,
where bumping would have orphaned Stage-1 chains for a Stage-2-only change): Stage 2 conditions on
Stage-1 draws, so a Stage-1 sampler change invalidates both. The Pathfinder generation keeps the
un-suffixed token and stays on disk untouched as the comparison baseline — read the two side by
side the way `11j_viz_utils.jl` already reads `CONTACTS_TOKEN_HD`.

**Both older generations are RETAINED on disk**: `-hd` in `dt_intermediate_hierarchical/` (2022 files)
and the pre-hierarchy `…-sc` in `dt_intermediate_old/` (1520 files). The `-hd` set is still read by
`src/10j_viz_utils.jl`/`11j_viz_utils.jl` to compare "flat hierarchy" against "no hierarchy"; it needs
`CONTACTS_TOKEN_HD` **and** `CONTACTS_SAVE_DIR_HD` together, since token and directory both differ.
The `dt_intermediate_old/` set is NOT reusable here even though its Stage 1 also has no RE — its
hurdle-Weibull Stage 1 lacks `p0f` and used the old κ clamp [-3,3], and all of its Stage 2 predates
`w_mu`/`w_sigma`, `ref_bin=4` and the `susc_inf_sd_prior` change.

`-s0` (2026-08-05) — **s**um-to-**0**: the Stage-1 age-pair structure field is now constrained to be
mean-zero over the 28 pairs within each week, `R = η·(Q·La·z·Ltᵀ)` with `Q` the constant Helmert
basis of 1^⊥ and `La = chol(Qᵀ·Kp·Q + 1e-6·I)` (`_sum_zero_basis`, `model_degree`). This is a
PARAMETER-SPACE change, not a sampler one: `z` goes `P×Tn` → `(P−1)×Tn`, so Stage 1 is **390**
(NegBin) / **978** (hurdle-Weibull) unconstrained dimensions, down from 402/990. Why: the field's
per-week mean was a second copy of `c_t = c + σ_c·(Lt·z_c)_t`, so η and σ_c were confounded — a ridge
that tightens as ρ grows (`Kp → J`), and a plausible contributor to the `max_depth = 10` saturation
measured on 2026-08-05 (`tasks/lessons.md`). The implied covariance is exactly `η²·(Kt ⊗ M·Kp·M)`,
i.e. the same GP conditioned, not an approximation.

Because it is a Stage-1 parameter-space change, **every `8j_s1_*` AND `8j_s2_*` file under the
previous token is stale** (Stage 2 conditions on Stage-1 draws), as is every derived 9j cache. It is
placed BEFORE the `-nuts` sampler component so the model/sampler split stays readable in the
filename. Note this stacks on the `-nuts` live migration that had not yet been fitted, so in practice
no completed grid is discarded — see `CONTACTS_TOKEN_PF`.

`-m32` (2026-08-05) — **Matérn 3/2** replaces the squared exponential in BOTH Stage-1 GP kernels, and
the **off-diagonal (age-gap) smoothing term is restored**. Spatially `Kp` is again a separable
anisotropic kernel in the rotated `(u, v)` = (total age, age gap) coordinates, now a product of two
1-D Matérn 3/2 factors; temporally `Kt` is a 1-D Matérn 3/2 over the window weeks. So `log_rho_gap`
is BACK and Stage 1 returns to **390** (NegBin) / **978** (hurdle-Weibull) unconstrained dimensions.

This supersedes `-diag`, which for a few hours the same day had dropped `log_rho_gap` and smoothed the
matrix diagonal only (389/977). Why it was reverted: a four-cell NUTS pilot (2 degree models × 2
origins × h1) showed `-diag` did not fix the mixing problem it was aimed at — min ESS at
negbin @ 2020-11-15 went 1.9 (pre-`s0`) → 4.0 (`-s0`) → 5.4 (`-diag`) of 500, with 100% of sampling
iterations at max tree depth in all four cells. The measured cause was not the spatial kernel at all
but ρ_time drifting to 24–63 weeks over a 12-week window, where the SQUARED-EXPONENTIAL `Kt` goes
numerically rank-4-of-12 and the map `z ↦ field` becomes ~60× anisotropic. Hence the kernel family
change rather than a further edit to the spatial structure.

Measured statically (kernel_static, no MCMC): `Kt` keeps FULL rank 12 at every ρ_time in
`RHO_TIME_BOUNDS` (SE fell to rank 10 by ρ_time = 4); `Ap` keeps rank 27 across all of `RHO_BOUNDS`
including the ceiling, with min eigenvalue 2.5e-2 at the `gp_len_prior` mode against 2.9e-5 for SE at
the same ρ; `Kp` is PSD (min eigenvalue 1.4e-7) and unit-diagonal exactly, and
`cholesky(Ap + 1e-6·I)` is clean at all 625 points of a 25×25 (ρ_diag, ρ_gap) grid.

Stage-1 parameter-space change again, so the same staleness rule applies: every `8j_s1_*` AND
`8j_s2_*` under any previous token is unreachable (none deleted). Both ρ priors changed with it
(`gp_len_prior` → N(log 20, 0.35²), `gp_time_len_prior` → N(log 2, 0.35²)); the token does not encode
priors, but no chain was ever fitted under `-s0` or `-diag`, so nothing collides. The read-only
mirrors (`reconstruct_mu_draws`, `load_transmission_draws`) carry the `log_rho_gap` sniff INVERTED —
a chain LACKING it is now the stale generation.

`-t0` (2026-08-06) — **t**ime sum-to-**0**: the per-week LEVEL's temporal deviation is now constrained
to be mean-zero over the Tn window weeks, `cₜ = c + σ_c·(Qt·Lc·z_c)` with `Qt = _sum_zero_basis(Tn)`
and `Lc = chol(Qtᵀ·Kt·Qt + 1e-4·I)` — the temporal analogue of `-s0`, using the identical machinery.
`z_c` goes Tn → Tn−1, so Stage 1 is **389** (NegBin) / **977** (hurdle-Weibull).

Why: `c` and the time-mean of `σ_c·(Lt·z_c)` were two parameterisations of the same quantity. The
flat direction that creates was measured on all four `-m32` chains at corr = **−1.000 exactly**, with
SD(c) ≈ SD(deviation) ≈ 0.38–0.73 against SD(their sum) = **0.007** — they cancel to 1–2% of their
own spread. `model_degree` had carried a note deferring this ("Held back so the Tn-axis change can be
measured separately") since the temporal GP landed; this is that change.

⚠ The latent counts 389/977 COINCIDE with the short-lived `-diag` generation's. The models are
unrelated: `-diag` had one spatial length-scale and a squared-exponential kernel, this has two and
Matérn 3/2. Distinguish by `log_rho_gap` (present here, absent under `-diag`) and by the token — the
`z_c` row count (Tn−1 vs Tn) is the other in-chain signal.

Applies to the LEVEL ONLY. The structure field keeps the full `Lt`: its per-pair mean over weeks
duplicates nothing, so constraining it would be a model restriction rather than a reparameterisation.

`-ar1` (2026-08-06) — **AR(1)** temporal correlation (user request). The time direction only: the
Matérn 3/2 temporal kernel is replaced by `Kt[s,t] = φ^|s−t|`, which IS an AR(1) correlation matrix
(equivalently the exponential / Matérn 1/2 kernel). The SPATIAL kernel is untouched, so this needed no
structural change — the separable matrix-normal already gives every age pair its own temporal
trajectory under one shared amplitude η ("AR(1) per age pair, sharing the variance"), and the pairs
remain correlated across age through `La`. `log_rho_time` → `phi_time`; the latent count is
**unchanged at 389/977**, so as with the reverted `-ig` the count cannot date a chain — the in-chain
signal is the NAME plus the token.

WHY, measured statically through the shipped code path, at MATCHED effective rank (the same amount of
temporal pooling, which is what the likelihood picks). `Lt sprd` is the column-scale ratio that broke
NUTS on 2026-08-05 (2.4 in healthy chains, 228–274 in the chains that would not mix); `Lc sprd` is its
analogue after the `-t0` projection, using the shipped Helmert `_sum_zero_basis`:

| effrank | Matérn 3/2 | AR(1) | Kt min eig | cond | Lt sprd | Lc sprd |
|---|---|---|---|---|---|---|
| ≈4.5 | ρ=2 | φ=0.785 | 5.0e-2 → 1.2e-1 | 83 → 47 | 2.4 → 2.6 | 1.7 → 1.6 |
| ≈1.4 | ρ=10 | φ=0.95 | 4.6e-4 → 2.6e-2 | 2.2e4 → 380 | 26.8 → 8.6 | 5.5 → 1.8 |
| ≈1.08 | ρ=26 | φ=0.99 | 2.7e-5 → 5.1e-3 | 4.3e5 → 2.3e3 | 94.2 → 23.1 | 12.5 → 1.8 |
| ≈1.03 | ρ=47 | φ=0.995 | 4.5e-6 → 2.6e-3 | 2.6e6 → 4.6e3 | 151.5 → 33.4 | 13.5 → 1.8 |

A wash in NegBin's regime (ρ_time ≈ 2.2); 4× better column spread and 2–3 orders better min
eigenvalue in hurdle-Weibull's (ρ_time 20–66 wk). The LEVEL benefits most: `Lc` spread is essentially
FLAT at 1.6–1.8 across every φ (worst 1.8 over 1000 values), where Matérn 3/2 degrades to 13.5.
MECHANISM: AR(1) is MARKOV — tridiagonal precision, eigenvalues decaying only polynomially — so it
keeps spectral mass in the non-constant directions even at φ = 0.995, where Matérn 3/2's spectrum has
collapsed onto one direction and `Lt`'s first column absorbs the whole field. This continues the
SE → Matérn 3/2 → Matérn 1/2 direction rather than reversing it.

⚠ A MODELLING change too, not only numerical: AR(1) paths are non-differentiable (rougher week to
week) and memory is LONGER at long lag — at matched lag-1 correlation 0.785, lag-4 is 0.380 against
Matérn 3/2's 0.140 and lag-8 is 0.115 against 0.008.
⚠ It does NOT change what the likelihood wants. Hurdle-Weibull prefers near-constant weekly contacts
(measured at ρ_time 20–27 wk under the tight log-normal and 47–66 wk under the reverted `-ig`); AR(1)
lets the sampler represent that regime without the geometry punishing it. A high φ is the
measurement, not a failure.
⚠ `RHO_TIME_BOUNDS` and the temporal soft-clamp are GONE from this path — φ ∈ (0,1) by construction
and φ^k cannot overflow. Stage-1 parameter-space change again, so every `8j_s1_*`/`8j_s2_*` under a
previous token is unreachable (none deleted).

Stage-1 parameter-space change, so the same staleness rule applies: every `8j_s1_*`/`8j_s2_*` under
any previous token is unreachable (none deleted)."""
contacts_label(cfg::FrameworkConfig) =
    (cfg.constant_contacts ? "pooled" : "temporal") * "-gsar-cut-sc-p0-gi-s0-m32-t0-ar1" *
    (cfg.stage1_use_nuts ? "-nuts" : "")

"""Default contacts token for the read-only viz helpers that do NOT receive a `cfg`
(`stage1_chain_path`, `reconstruct_p0_draws`, `reconstruct_tau_draws`, …), so a token bump lands in
one place instead of the seven hard-coded copies that previously had to be edited in lockstep.
Resolves to the per-week (`constant_contacts=false`) regime — the setting every notebook uses — and,
since `stage1_use_nuts` became the default on 2026-08-05, to the **NUTS** sampler generation, i.e. it
now carries the `-nuts` suffix.

⚠ **That is a live migration, not a no-op.** The complete 504/1512-file grid on disk is the
*Pathfinder* generation and is reached by `CONTACTS_TOKEN_PF` below. Until the NUTS grid has actually
been fitted, the read-only viz helpers that default to this constant (`stage1_chain_path`,
`reconstruct_p0_draws`, `reconstruct_tau_draws`, the 9j/10j/11j figures) will find **no files**.
Point them at `CONTACTS_TOKEN_PF` to read the old generation, exactly as `plot_within_block_sd`
already does with `CONTACTS_TOKEN_HD`.

It is a load-time constant built from the **default** `FrameworkConfig`. The token no longer encodes
any dispersion prior (it did while the horseshoe's τ₀ was being tuned), so it is stable again — but
helpers that take a `cfg` still default to `contacts_label(cfg)` rather than this, which stays the
right habit for any future field that does enter the token. Use this only where no `cfg` is in scope
(see `CONTACTS_TOKEN_HD` for the previous generation)."""
const CONTACTS_TOKEN = contacts_label(FrameworkConfig(constant_contacts = false))

"""The **Pathfinder** Stage-1 generation's token — what the 504 `8j_s1_*` / 1512 `8j_s2_*` files
currently in `dt_intermediate/` were fitted under, before `stage1_use_nuts` became the default on
2026-08-05.

⚠ **A LITERAL, no longer `contacts_label(…; stage1_use_nuts=false)`.** It became one when `-s0`
landed the same day: that generation predates the sum-to-zero constraint, so it now differs from the
current token by a MODEL component as well as the sampler, and no `cfg` can reproduce it. This is
the same pattern `CONTACTS_TOKEN_HD` already uses — a retained generation is named by its literal
string, because the config that produced it no longer exists. Its Stage 1 has the unconstrained
`z` (28×12 = 336 coordinates, 402/990 total), which is why `reconstruct_mu_draws` must branch on the
field construction rather than assume the current one.

This is a live constant, not a historical note — it is how the 9j/10j/11j viz reaches the existing
grid while the current generation is being fitted, and how the two are read side by side afterwards.
Unlike `CONTACTS_TOKEN_HD` it needs NO separate directory: both live in `dt_intermediate/`,
distinguished by the suffix alone."""
const CONTACTS_TOKEN_PF = "temporal-gsar-cut-sc-p0-gi"

"""The PREVIOUS generation's token (`-hd`, 2026-07-30 → 2026-08-02): flat hierarchical dispersion
with a per-week half-Normal τ_t (and, briefly after it, the `-rhs` horseshoe). Those chains are still
on disk and are read side by side with `CONTACTS_TOKEN` by `plot_within_block_sd`, which is now the
"did the RE actually go away" check — the current model's within-block SD of log-dispersion must be
identically 0 against a non-zero `-hd` line. So this is a live constant, not a historical note. Only
the dispersion mirrors understand it — `stage1_moment_draws` does NOT (see `_read_disp_chain`)."""
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
