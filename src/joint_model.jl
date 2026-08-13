# joint_model.jl — the TWO-STAGE (cut) forecast model + Pathfinder→NUTS fits + pooled forecast.
#
# The former single joint @model (contact-degree likelihood + infection likelihood together) was
# split into a CUT inference (inst/4_cut_Bayes.md):
#   • Stage 1 `model_degree`      — the contact-degree GP alone (NGM-independent; returns per-week
#                                    raw moments ⟨k⟩,⟨k²⟩,g). Fit once per (degree × origin × horizon).
#   • Stage 2 `model_transmission`— the infection/renewal block alone, conditioning on a FIXED C*
#                                    built from ONE Stage-1 draw. Re-fit for each of `n_stage1_post`
#                                    (=100) Stage-1 draws, keeping `n_stage2_draws` (=100) each; the
#                                    100×100 = 10_000 pooled draws are the infection predictive → WIS.
# Composability is preserved: `dm::ContactDegreeModel` selects the degree likelihood/dispersion in
# Stage 1; `nb::NGMBuilder` selects the NGM contact functional applied to the Stage-1 moments before
# Stage 2 (deterministic dispatch). The C*-normalisation (`C*→C*/S̄`) was REVERTED, so C* feeds the
# NGM at its raw level and the transmissibility scalar is again the per-contact SAR `gamma_sar`.

block_of(a::Int, cfg::FrameworkConfig) = a <= cfg.child_bins ? 1 : 2

# Numerically-stable softplus and an interior-preserving smooth "soft clamp".
# `_softclamp(x, lo, hi)` equals `x` in the interior (lo ≪ x ≪ hi) and saturates smoothly to
# ≈lo / ≈hi outside — a differentiable, ReverseDiff-safe replacement for `clamp` that keeps the
# exp-transformed rates/shapes/means finite during aggressive Pathfinder/LBFGS steps without
# distorting the well-scaled interior where good fits live. `_softplus` branches on the sign to
# avoid `exp` overflow (branch is on a value, so it is ReverseDiff-safe on an uncompiled tape).
#
# The clamp is applied as a NESTED upper-then-lower soft bound: `x` is first squashed against `hi`
# (`hi − softplus(hi − x)`, itself finite even at x=±Inf), then that result against `lo`. Every
# intermediate is individually finite, so a ±Inf input SATURATES to ≈hi/≈lo instead of producing
# the `Inf − Inf = NaN` of the naive `x − softplus(x−hi) + softplus(lo−x)` form. This matters:
# when the GP field or a raw dispersion latent (`log_kappa`/`log_k`) overflows to Inf on a stray
# optim step, `dispv`/`rvec` become Inf, and the old form's NaN would make `κ = exp(NaN)` NaN and abort
# the fit with `Weibull: α > 0 not satisfied`; the nested form keeps κ, λ, μ finite so the fit
# just sees a bad (finite) objective and backtracks.
#
# `s` IS THE TRANSITION WIDTH AND IT IS LOAD-BEARING (added 2026-08-05). The un-scaled form
# (equivalently s=1) inherits `_softplus`'s O(1) transition width, so "equals x in the interior" is
# only true when `hi − lo` is several nats. It was NOT true for the ρ length-scales, whose window
# was 2.708 nats wide (`[log 3, log 45]`): the derivative there never exceeded **0.600** ANYWHERE —
# there was no interior at all. Measured consequences at origin 2021-05-09, hurdle-Weibull:
#   • gradients attenuated ~1.9× at the mode, and curved everywhere (nonzero 2nd derivative
#     throughout), which bends the ρ↔z ridge in a way a diagonal metric cannot undo;
#   • `gp_len_prior = N(log 15, 0.5)` was delivered as ρ∈[10.6,19.0] at ±1sd, not [9.1,24.7];
#   • the floor BOUND: raw log_rho_gap = 1.5375 came out as ρ_gap = 7.22, not exp(1.5375) = 4.65.
# ⇒ `log_rho_gap`/`log_rho_diag` were 2 of the 15 worst-mixing coordinates (ESS 44.5/59.9 of 500).
#   (`log_rho_gap` was briefly removed later the same day by `-diag` and RESTORED by `-m32`; the
#   clamp lesson stands for both length-scales and for every other soft-clamped log-latent.)
# With s = 0.25 the same window is the identity to 4 s.f. and the min gradient over ρ∈[1,300] is
# 0.885. The wide clamps are unaffected (log_kappa 0.980→1.000, μ 0.997→1.000).
# Keep s well below `(hi − lo)`; s → 0 recovers a hard `clamp` (and a flat, hard-to-escape
# exterior), so 0.25 is deliberately moderate. Inf-safety is preserved: every intermediate is
# finite, `f(+Inf) = hi + s·log1p(exp(-(hi-lo)/s))` (≈5e-6 over `hi` at s=0.25) and `f(-Inf) = lo`.
_softplus(z) = z > zero(z) ? z + log1p(exp(-z)) : log1p(exp(z))
_softclamp(x, lo, hi, s = 0.25) = lo + s * _softplus((hi - s * _softplus((hi - x) / s) - lo) / s)

# Matérn 3/2 correlation at scaled separation `x = |Δ|/ρ`  (2026-08-05, replaced the squared
# exponential in BOTH GP kernels — the spatial `Kp` and, at the time, the temporal `Kt` in
# `model_degree`).
#
# ⚠ IT IS NOW THE SPATIAL KERNEL ONLY. The temporal factor went to AR(1) on 2026-08-06 (`-ar1`),
# back to Matérn 3/2 for one day on 2026-08-10 (`-m32t`), and back to AR(1) the same day when the
# smoke measured the pooled limit still being reached — see the kernel block in `model_degree`.
# `_m32` is still used by the read-only mirrors to replay both generations, so do not delete it if
# the temporal path stops calling it.
#
# WHY: the squared exponential's eigenvalues decay super-exponentially, so at the length-scales this
# model wants it goes numerically low-rank and the non-centred map `z ↦ field` becomes wildly
# anisotropic — a handful of `z` columns pinned by the data while the rest sit at prior. That is a
# geometry NUTS cannot step through, and it is what produced 100% max-tree-depth and min ESS 1.9–5.4
# of 500 across every cell of the 2026-08-05 pilot (tasks/lessons.md). Matérn 3/2 has polynomial
# spectral tails instead, so the same amount of smoothing costs far less conditioning. Measured on
# this design (`Tn=12`, 28 age-pairs, the real CIS midpoints):
#   • `Kt` keeps FULL rank 12 at EVERY ρ_time in `RHO_TIME_BOUNDS` (the SE kernel fell to rank 10 by
#     ρ_time = 4 and rank 4 by ρ_time = 63). At the `gp_time_len_prior` mode the column-scale spread
#     of `Lt` is 2.4, against 228–274 in the chains that failed to mix.
#   • `Ap`'s smallest eigenvalue at the `gp_len_prior` mode is 2.54e-2 — 25 000× the 1e-6 jitter,
#     where the SE kernel gave 2.95e-5 and, at +2σ, 1.53e-8, i.e. BELOW the jitter.
#
# AD: `x` is always `abs(constant)/ρ`. The `abs` is applied to parameter-free coordinates (`su`/`df`
# in space, `s-t` in time), so no non-differentiable operation is ever tracked and `_m32` is smooth
# in ρ everywhere, including on the matrix diagonal where the separation is 0.
_m32(x) = (1 + sqrt(3) * x) * exp(-sqrt(3) * x)

# Element type of the fitted hurdle p⁰ block, for the `ETp` promotion in `model_degree`.
# The NegBin path has no p⁰ (it models its zeros directly) and passes `nothing`; `Bool` is the
# identity for `promote_type` (`promote_type(T, Bool) == T` for every numeric T), so the NegBin
# promotion is unaffected.
_p0_eltype(::Nothing) = Bool
_p0_eltype(x) = eltype(x)

"""
    _unordered_pairs(A)

Return `(pair_list, pair_index)` for the reciprocity-structural mean (inst/1e).
`pair_list` is the vector of `A·(A+1)/2` unordered age pairs `(a,b)` with `a ≤ b`
(28 for `A=7`); `pair_index[i,j]` gives the row of `pair_list` for the representative
`(min(i,j), max(i,j))`, so `(i,j)` and `(j,i)` share one latent value.
"""
function _unordered_pairs(A::Int)
    pair_list = [(a, b) for a in 1:A for b in a:A]
    idx = Dict(p => k for (k, p) in enumerate(pair_list))
    pair_index = [idx[(min(i, j), max(i, j))] for i in 1:A, j in 1:A]
    return (pair_list, pair_index)
end

"""
    _sum_zero_basis(P)

`P × (P−1)` orthonormal basis of the complement of the all-ones vector: `QᵀQ = I`, `Qᵀ1 = 0`,
`QQᵀ = M = I − 11ᵀ/P`. Used by `model_degree` to constrain the age-pair structure field to be
**mean-zero over the P pairs within each week** (2026-08-05), which is what makes the code's
long-standing claim that σ_c carries the level and η "governs age-structure only" actually true.
Without it the field's per-week mean duplicates `c_t` exactly, and the two are confounded — a ridge
that tightens as ρ grows, because `Kp → J` (rank-1) in that limit (see `RHO_BOUNDS`' docstring).

**Helmert contrasts, NOT `qr(ones(P))`** — and that is load-bearing, not taste. QR's column signs
come from LAPACK. The *model* is invariant to a sign flip (it is absorbed by flipping the matching
row of `z`), but `reconstruct_mu_draws` is NOT: it rebuilds the basis to replay a saved chain, so a
basis that differed by a sign between the fitting run and the replay would silently produce a
different μ with nothing raising. Helmert is closed-form and byte-identical on every machine.

Verified orthonormal / annihilating-1 / `QQᵀ = M` to ≤4.4e-16 at P = 28.
"""
function _sum_zero_basis(P::Int)
    Q = zeros(P, P - 1)
    for k in 1:(P - 1)
        s = sqrt(k * (k + 1))
        Q[1:k, k] .= 1 / s
        Q[k + 1, k] = -k / s
    end
    return Q
end

"""
    build_degree_stats(dm, apd, cfg)

Precompute the fixed per-cell inputs the model needs: degree distributions, empirical
zero prob `p0`, and the prior-centre `log_emp` (log count-mean for NegBin; log
positive-weight mean for the hurdle).

Two contact regimes (`cfg.constant_contacts`):
- **pooled** (`true`): collapse the per-week cells to one pooled cell per `(i,j)`; the
  returned `dd_count`/`pos_weight`/`p0`/`n` are `A×A`.
- **per-week** (`false`): keep the raw `[t,i,j]` weekly arrays so the model can fit an
  independent age-pair GP per week (`model_degree`'s per-week branch).

`log_emp` (hence the GP prior-centre `c0`) is always the **pooled** grand mean, so the
prior is identical across regimes.
"""
function build_degree_stats(dm::ContactDegreeModel, apd::AgePairData, cfg::FrameworkConfig)
    p = pool_over_time(apd)                                                 # pooled: data + c0 centre
    A = apd.A
    log_emp = Matrix{Float64}(undef, A, A)
    for i in 1:A, j in 1:A
        base = if is_weighted(dm)
            isempty(p.pos_weight[i, j]) ? 1e-3 : whist_mean(p.pos_weight[i, j])   # μW init
        else
            max(p.emean[i, j], 1e-3)                                        # count-mean init
        end
        log_emp[i, j] = log(base)
    end
    pair_list, pair_index = _unordered_pairs(A)
    # Sum-to-zero basis for the structure field, precomputed ONCE. Both `sz_Q` and its transpose are
    # stored as plain dense `Matrix{Float64}`: an `Adjoint` wrapper inside the model body is exactly
    # what ReverseDiff cannot write a dense cotangent into (see the `La` densification note in
    # `model_degree`), and `Q'` would create one on every gradient evaluation.
    sz_Q = _sum_zero_basis(length(pair_list))
    common = (; log_emp = log_emp, A = A, weeks = apd.weeks,
                mid = cis_age_midpoints(), pair_list = pair_list, pair_index = pair_index,
                sz_Q = sz_Q, sz_Qt = Matrix(sz_Q'))
    if cfg.constant_contacts
        return (; dd_count = p.dd_count, pos_weight = p.pos_weight, p0 = p.p0,
                  n = p.n, common...)
    else
        return (; dd_count = apd.dd_count, pos_weight = apd.pos_weight, p0 = apd.p0,
                  n = apd.n, common...)                                     # raw [t,i,j] weekly arrays
    end
end

# ======================================================================================
# NULL model support (inst/6_null_interaction_model.md) — no contact fit at all.
# ======================================================================================
"""
    null_contact_level(apd, win0)

The NULL model's fixed per-cell contact level `c̄`: the roster-weighted mean number of **unweighted**
(no duration weights) contacts a participant-day reports over `win0`'s **8 focal fit weeks**,
divided across the `A` age bins so each row of the uniform `C*` sums to that average total —
keeping `γ_SAR` on the same per-contact scale as the mean-NGM models.

`apd.emp_mean[t,i,j]` is the mean unweighted count of bin-`i`→bin-`j` contacts per participant-day
(zeros included) and `apd.n[t,i,j] == n_roster[t,i]` for every `j`, so

    c̄ = Σ_{t∈fit, i} n[t,i] · Σ_j emp_mean[t,i,j] / Σ_{t∈fit, i} n[t,i] / A .

`apd` may be ANY of the window's horizon-`h` degree windows: since `-w8h` (2026-08-09) each is
`[t₀−n_fit+1 … t₀+h]` and so contains ALL of `win0.fit_weeks` as its first `n_fit` entries
(`stage2_inputs` asserts exactly that), and the row-sum over `j` is invariant to the seeded
contactee-bin draw — so `c̄` is the same whichever is passed. That is what makes the null constant
**fixed while forecasting** (spec: "the used average number should be fixed while forecasting").
⚠ The intermediate `-w8` window (sliding, `[t₀−n_fit+1+h … t₀+h]`) broke this: it held only
`n_fit − h` of the focal weeks, so `c̄` drifted with the horizon. Anchoring the window restored it.

NOTE the model's *forecasts* are invariant to the `/A` convention AND to `c̄` itself:
`N_ab = γ_SAR·fs_a·c̄·inf_b`, so `γ_SAR` and `c̄` enter only as a product and `γ_SAR` is freely
estimated (verified: `c̄`×10 ⇒ `γ_SAR`×0.105, R unchanged to 0.3%). The convention only fixes what
`γ_SAR` **means** (and where it sits under its prior).
"""
function null_contact_level(apd::AgePairData, win0::WeeklyWindow)
    A  = apd.A
    ts = findall(w -> w in win0.fit_weeks, apd.weeks)
    isempty(ts) && error("null_contact_level: none of win0.fit_weeks are in the degree window " *
                         "($(first(apd.weeks))–$(last(apd.weeks)))")
    # A partial overlap is an ANOMALY again under `-w8h` (the window is anchored at the origin, so
    # all `n_fit` focal weeks are present by construction) — restored after the sliding `-w8` window
    # made it the normal case for a few hours. If this fires, the caller built the wrong window.
    length(ts) == length(win0.fit_weeks) ||
        @warn "null_contact_level: only $(length(ts))/$(length(win0.fit_weeks)) focal weeks present"
    num = 0.0; den = 0.0
    for t in ts, i in 1:A
        n_ti = apd.n[t, i, 1]                                  # roster count (same for all j)
        num += n_ti * sum(apd.emp_mean[t, i, j] for j in 1:A)   # total contacts of bin i that week
        den += n_ti
    end
    return (den > 0 ? num / den : 0.0) / A
end

"""
    null_moment_draws(c0, A, Tn; M=1)

Stand-in for `stage1_moment_draws` on the NULL path: `M` identical "draws" whose per-week `K1` is
the constant matrix `fill(c0, A, A)`. `NullNGM`'s `base_contact` is the identity, so
`contact_star` returns that constant matrix for every week and every horizon. `K2`/`G` are unused
by `NullNGM` and are filled with `0`/`1` so the NamedTuple has the same shape Stage 1 returns.

`M = 1` by default: the null model has **no** contact-degree uncertainty to propagate, so repeating
identical Pathfinder fits would only add fit-to-fit approximation noise (at 100× the cost). Draw
parity with the other models is kept by raising `n_draw` instead (see `prefit_stage2!`).
"""
null_moment_draws(c0::Real, A::Int, Tn::Int; M::Int = 1) =
    [(; K1 = [fill(float(c0), A, A) for _ in 1:Tn],
        K2 = [zeros(A, A)           for _ in 1:Tn],
        G  = [ones(A, A)            for _ in 1:Tn]) for _ in 1:M]

# --- per-cell raw moments (⟨k⟩, ⟨k²⟩) + zero factor g, from the fitted params ---
# g scales the neighbourhood-degree C0 to condition on non-zero contacts (inst/1c,1d):
#   NegBin  g = 1/(1−P₀), P₀ = (φ/(φ+μ))^φ  — left-truncated fitted NegBin.
#   Weibull g = (1−p⁰)                       — empirical hurdle non-zero probability.
# The MeanNGM builder ignores g. ⚠ Floor (1−P₀) so near-empty cells (μ→0 ⇒ P₀→1)
# don't blow up; the relative-population offset keeps μ interior (see tasks/lessons.md).
const _P0_FLOOR = 1e-3
function _negbin_moments(m, k)
    P0 = (k / (k + m))^k
    g  = 1 / max(1 - P0, _P0_FLOOR)
    return (m, m + m^2 * (1 + 1 / k), g)                                      # ⟨k⟩, ⟨k²⟩, g
end
function _weibull_moments(μW, κ, p0)                                          # incl-zero raw moments
    cvw2 = gamma(1 + 2 / κ) / gamma(1 + 1 / κ)^2 - 1
    return ((1 - p0) * μW, (1 - p0) * μW^2 * (1 + cvw2), 1 - p0)              # ⟨k⟩, ⟨k²⟩, g
end

# ======================================================================================
# Stage 1 — contact-degree GP (NGM-INDEPENDENT). Returns per-week raw moments (⟨k⟩,⟨k²⟩,g),
# so ONE Stage-1 fit serves both NGM builders (the builder is applied downstream via
# `contact_star`). No S̄ normalisation, no transmission block — those live in Stage 2.
# `pop` is the per-bin population (for the reciprocity offset); `ds` = build_degree_stats output.
# ======================================================================================
@model function model_degree(dm::ContactDegreeModel, ds, pop, cfg::FrameworkConfig)
    A = ds.A
    Tn = length(ds.weeks)

    # ---- reciprocity-structural, GP-smoothed contact mean (inst/1e) ----
    # 28 unordered age pairs (a≤b) carry one symmetric log-rate r; reciprocity is exact
    # via the contactee-population offset  log μ_{i→j} = r_{min,max} + log(pop_j)
    # ⟹ pop_i·μ_{i→j} = pop_j·μ_{j→i}. The rate field is a separable-RBF GP over the
    # age-pair grid, non-centred as f = η·L·z (L = chol K).
    # The kernel is ANISOTROPIC in DIAGONAL coordinates and SEPARABLE (`-m32`, 2026-08-05): the age
    # pair (x,y)=(mid_a,mid_b) is rotated 45° into u=(x+y)/√2 (along the main diagonal = total age)
    # and v=(x−y)/√2 (across it = age gap), each with its OWN length-scale — ρ_diag on total age,
    # ρ_gap on the age gap (assortativity). `Kp` is the PRODUCT of two 1-D Matérn 3/2 factors, one
    # per direction, so it is unit-diagonal by construction and PSD as a product of PSD kernels.
    #
    # The off-diagonal (age-gap) direction was briefly removed earlier the same day (`-diag`) and is
    # RESTORED here; the squared exponential went with it (see `_m32` above for the measurement that
    # motivated the swap). Because a product of two 1-D Matérns is a separable process rather than a
    # 2-D Matérn, ρ_diag = ρ_gap does NOT recover an isotropic Matérn — that identity held for the
    # old squared-exponential form and no longer applies.
    #
    # Verified statically over the whole of `RHO_BOUNDS` (kernel_static, 200 isotropic ρ + a 25×25
    # anisotropic grid): min eigenvalue of `Kp` = 1.4e-7 (PSD), unit diagonal and symmetry exact to
    # 0.0, and `cholesky(Ap + 1e-6·I)` clean at all 625 (ρ_diag, ρ_gap) combinations. rank(Ap) = 27
    # everywhere in the window INCLUDING the ρ = 500 ceiling — the Matérn tails are heavy enough that
    # `Kp → J` is not reached inside the clamp at all (min(Kp) is still 0.93 at ρ = 500; rank first
    # drops, to 24, only around ρ ≈ 5000). Under `gp_len_prior` the effective rank is 16.8 at −2σ,
    # 8.3 at the mode and 4.4 at +2σ, with min eigenvalue 2.5e-2 at the mode.
    # `ρ_diag`, `ρ_gap`, `η` and the 27×27 Cholesky `La` are SHARED
    # across weeks. In the per-week regime the weekly fields are no longer iid: they are
    # coupled by a SEPARABLE temporal GP (§5) — a matrix-normal field R = η·(Q·La·z·Ltᵀ) with a
    # shared temporal Cholesky Lt(φ) and a decoupled, UNSMOOTHED level cₜ = c + σ_c·(Qt·z_c).
    # The population offset is taken RELATIVE to the reference bin (index 1, "2-10"): only
    # relative population matters for reciprocity, and a constant shift log(pop₁) cancels in
    # pop_i·μ_{i→j}=pop_j·μ_{j→i}, so exact reciprocity is preserved — but it rescales the
    # latent level c/c0 to O(1) (absolute log(pop)≈15.6 otherwise forces c≈−15.6 and, at the
    # old clamp, the degenerate μ≡403 saturation; see tasks/lessons.md).
    logpop = log.(pop ./ pop[1])
    log_rho_diag ~ Normal(cfg.gp_len_prior[1], cfg.gp_len_prior[2])   # total-age direction
    log_rho_gap  ~ Normal(cfg.gp_len_prior[1], cfg.gp_len_prior[2])   # age-gap direction (assortativity)
    log_eta ~ Normal(cfg.gp_scale_prior[1], cfg.gp_scale_prior[2])
    # UNITS: both ρ live on the rotated (`su`, `df`) scale, which is √2× an age difference — so
    # ρ = 20 is an effective age-difference length-scale of 20/√2 = 14.1 yr. The two share ONE prior
    # (`gp_len_prior`), as they did before `-diag`.
    ρ_diag = exp(_softclamp(log_rho_diag, RHO_BOUNDS...))   # length-scale (age-yrs), soft-bounded
    ρ_gap  = exp(_softclamp(log_rho_gap,  RHO_BOUNDS...))   # length-scale (age-yrs), soft-bounded
    η = exp(_softclamp(log_eta, -3.0, 2.0))               # GP marginal scale, soft-bounded
    c0 = mean(ds.log_emp .- logpop')                      # smooth mean-fn anchor (pooled c0)
    mid = ds.mid
    P = length(ds.pair_list)
    # rotated (diagonal / anti-diagonal) coordinates for the 28 pairs, √2-normalised. KEEP THE /√2:
    # the rotation is orthonormal, so (Δu)²+(Δv)² = (Δx)²+(Δy)² and both length-scales stay on the
    # same age-year footing as the un-rotated grid. Both are parameter-free, hence constant across
    # gradient evaluations — which is what keeps the `abs` inside `_m32` off the AD tape.
    su = [(mid[p[1]] + mid[p[2]]) / sqrt(2) for p in ds.pair_list]   # along-diagonal (total age)
    df = [(mid[p[1]] - mid[p[2]]) / sqrt(2) for p in ds.pair_list]   # across-diagonal (age gap)
    # 28×28 separable anisotropic Matérn 3/2 in diagonal coordinates. The product is exactly 1 when
    # m == n (both factors are `_m32(0) = 1`), so no diagonal special-case is needed.
    Kp = [_m32(abs(su[m] - su[n]) / ρ_diag) * _m32(abs(df[m] - df[n]) / ρ_gap)
          for m in 1:P, n in 1:P]
    # ---- SUM-TO-ZERO over the P pairs, within each week (2026-08-05) ----
    # `Q = ds.sz_Q` is the constant P×(P−1) Helmert basis of 1^⊥ (`_sum_zero_basis`), so QQᵀ = M =
    # I − 11ᵀ/P. Whitening in that subspace instead of the full one gives
    #     Cov(vec R) = η²·(Kt ⊗ Q·Ap·Qᵀ) = η²·(Kt ⊗ M·Kp·M),
    # i.e. the EXACT GP conditioned on mean_p R_{p,t} = 0 — not an approximation, and not a soft
    # penalty. Verified to 6.7e-16 against `M·Kp·M + jitter·M` at the range corners, with the field's
    # per-week mean zero to ≤7.4e-16.
    #
    # WHY: nothing previously constrained the field's per-week mean over the pairs, and that mean is
    # exactly what `c_t = c + σ_c·(Qt·z_c)_t` already parameterises — so η and σ_c were confounded,
    # increasingly so as ρ grows (`Kp → J` in that limit). The comment below
    # has claimed since the temporal GP landed that σ_c exists "so η governs age-structure only";
    # this is what makes that true. `z` drops from P×Tn to (P−1)×Tn ⇒ Stage 1 goes 402→390 (NegBin)
    # and 990→978 (hurdle-Weibull). The dimension saving is incidental; identifiability is the point.
    #
    # η IS NO LONGER THE MARGINAL SD. `Kp` has unit diagonal, `M·Kp·M` does not — the field's SD is
    # η·sqrt(diag(M·Kp·M)). Re-measured under `-m32` at the CURRENT `gp_len_prior` (log 20, 0.35):
    # ×0.709–1.076 (mean ×0.860) at the mode ρ=20, ×0.861–1.018 at −2σ and ×0.490–1.093 at +2σ.
    # Note the factor now exceeds 1 for some cells — under the squared exponential it was ≤1
    # everywhere, but Matérn's slower off-diagonal decay leaves cells that are anti-correlated with
    # the pair-mean, and projecting the mean out INFLATES those. The whole range still sits inside a
    # `gp_scale_prior` spanning ×0.61–×1.65 at ±1σ, so that prior is left alone — but this is now a
    # measured tolerance, not a negligible correction. Do NOT "fix" it by renormalising `Ap` by
    # tr(Ap)/P: as ρ→∞ that is a 0/0 dominated by the jitter and degenerates the field to WHITE
    # NOISE, inverting the correct limit (field → 0, `c_t` carrying everything).
    Ap = ds.sz_Qt * Kp * ds.sz_Q                        # (P−1)×(P−1) projected kernel
    # DENSE Cholesky factor (Matrix, not the LowerTriangular `.L`): the per-week structure field
    # forms the matrix product Q·La·z·Ltᵀ, and ReverseDiff cannot write a dense cotangent into a
    # triangular-typed factor (`… * Ltᵀ` → "cannot set index in the lower triangular part of an
    # UpperTriangular matrix"). Densifying both factors is the RD-safe form; gradients w.r.t.
    # ρ_diag/ρ_gap still flow through `Matrix(cholesky(...).L)`. (Verified vs triangular variants.)
    #
    # Jitter stays 1e-6, and under `-m32` it has a far wider margin than it did under the squared
    # exponential: `min eigval(Ap)` is 2.5e-2 at the `gp_len_prior` mode and 2.1e-3 at +2σ, versus
    # 2.9e-5 and 1.5e-8 (i.e. BELOW the jitter) for the SE kernel at the same ρ. Checked rather than
    # assumed — `cholesky(Ap + 1e-6·I)` is clean at all 625 points of a 25×25 (ρ_diag, ρ_gap) grid
    # spanning all of `RHO_BOUNDS`, and the worst case anywhere in the window is 1.4e-7 at the ρ=500
    # ceiling. That margin matters because the Pathfinder call is not try/caught (cf. `Kt`'s 1e-4).
    La = Matrix(cholesky(Symmetric(Ap) + 1e-6 * I).L)

    # per-cell log-rate → directional mean μ_{i→j}. Soft-clamped (not `clamp`, so ReverseDiff-safe)
    # to μ ∈ ≈[3e-4, 400]: the relative-population offset keeps the healthy log-rate O(1) (deep in
    # the interior, where _softclamp is the identity), so μ is undistorted; the soft bound only
    # stops a stray LBFGS step from over/under-flowing μ (which would make the Weibull scale
    # λ=μ/gamma non-finite and abort the fit).
    _mu_matrix(rvec) =
        [exp(_softclamp(rvec[ds.pair_index[i, j]] + logpop[j], -8.0, 6.0)) for i in 1:A, j in 1:A]

    # per-cell moments (⟨k⟩, ⟨k²⟩, zero factor g) + contact log-likelihood for one week's
    # μ matrix. `didx` indexes the (possibly weekly) degree arrays.
    #
    # DISPERSION IS BLOCK-LINEAR × WEEK, WITH NO PER-CELL TERM (§4.3). `dispv` is the length-4
    # block-linear log-dispersion for this week, indexed `bl = 2(bi−1)+bj ∈ {1,2,3,4}`, so every
    # ordered cell in a child/adult block shares one value:
    #
    #     log_disp_{ij,t} = dispv[bl(i,j)]
    #
    # A per-cell random effect lived here from 2026-07-30 to 2026-08-02 — first a flat non-centred
    # hierarchy (`τ_t·z_{ij,t}`), then a regularised horseshoe (Piironen & Vehtari 2017 eq. 11,
    # `τ·λ̃_{ij,t}·z_{ij,t}`). Both were REMOVED: measured across τ₀ ∈ {0.1, 0.01, 0.005, 0.001} at
    # origin 2021-05-09 h1, the horseshoe's global scale was simply outbid by the likelihood (τ
    # landing 7–15 prior SDs out, the slab inflating until it never bound, λ never leaving its init)
    # until τ₀ = 0.001, where the RE vanished outright — a cliff, not a usable shrinkage dial. With
    # 49 ordered cells per week and many of them empty, the per-cell dispersion was never identified
    # by the data; the block mean is what the window actually supports. See tasks/lessons.md.
    #
    # `p0v` (weighted path only; `nothing` for NegBin) is the per-cell FITTED hurdle zero
    # probability, replacing the empirical `ds.p0` plug-in (§4.2) — this is RETAINED.
    #
    # `dispv` and `p0v` are ≤ 2-D `filldist` slices (`4×Tn`, `A²×Tn`) so `generated_quantities` can
    # reconstruct them — a 3-D `filldist` cannot be (see build sites).
    function _cell_moments!(K1, K2, G, μ, didx, dispv, p0v)
        ll = zero(eltype(K1))                     # NOT eltype(μ): μ carries none of p0's type
        for i in 1:A, j in 1:A
            bl    = 2 * (block_of(i, cfg) - 1) + block_of(j, cfg)
            pcode = (i - 1) * A + j                        # ordered/directional, self-pairs included
            logd  = dispv[bl]                              # block-linear × week; no per-cell term
            if is_weighted(dm)
                # WIDENED 2026-07-30 from [-3,3] (κ∈[0.05,20]) to [-4.3,5] (κ∈[0.0136,148]).
                # The old bound was binding hard once the per-cell RE was added: every κ sat
                # exactly on 0.0498, which is the clamp-compression signature, and the flat
                # region it creates let the LBFGS path run away (block means reached −441, ~900
                # prior SDs). See tasks/lessons.md 2026-07-30. RETAINED when the per-cell RE was
                # removed on 2026-08-02: the clamp is a numerical guard, not part of the RE, and
                # (b) below is a hard floor regardless of what feeds `logd`. Widening it back would
                # only re-introduce a binding bound for no benefit.
                #
                # ⚠ THE LOWER BOUND IS NUMERICALLY LOAD-BEARING AND −4.45 IS THE HARD FLOOR.
                # It guards TWO different overflows, and the SECOND one binds much earlier — the
                # trap that first cost a run here:
                #   (a) λ = μ/gamma(1+1/κ)          needs 1+1/κ ≲ 171.6 ⇒ log κ ≳ −5.14
                #   (b) CV² = gamma(1+2/κ)/gamma(1+1/κ)²  (in `_weibull_moments`)
                #                                    needs 1+2/κ ≲ 171.6 ⇒ log κ ≳ −4.446  ← BINDS
                # Past (b) both gammas are Inf, so CV² = Inf/Inf = NaN, K2 goes NaN, and it
                # propagates silently into C* and the whole Stage-2 chain. Measured: −4.44 ⇒
                # gamma(170.6)=7.2e305, CV²=6.7e49 (finite); −4.50 ⇒ Inf ⇒ NaN. −4.3 keeps ~50
                # orders of headroom. Do NOT raise this to −5 "because λ is still finite" —
                # that was checked once and it broke (b).
                κ = exp(_softclamp(logd, -4.3, 5.0))       # shape ∈ ≈[0.0136, 148], soft-bounded
                λ = μ[i, j] / gamma(1 + 1 / κ)             # scale stays finite & >0 (μ, κ bounded)
                pos = didx === nothing ? ds.pos_weight[i, j] : ds.pos_weight[didx, i, j]
                isempty(pos) || (ll += calculate_loglikelihood(pos, Weibull(κ, λ)))  # collapsed histogram
                # ---- hurdle zero part: FITTED p⁰ with a Binomial roster likelihood (§4.2) ----
                # n = sampled participant-days for bin i in this week (constant in j);
                # n_zero = those with NO contact in cell (i,j) = n − (#participant-days with ≥1),
                # and `whist_nobs(pos)` is exactly that second count, so the split is exact.
                # The `log C(n, n_zero)` normaliser is dropped: it depends only on data, so the
                # posterior is unchanged, and it keeps lgamma out of a 49×Tn inner loop that runs
                # on every gradient evaluation. Written as the raw kernel rather than
                # `logpdf(Binomial(n, p0), nz)` to keep the AD surface trivial.
                p0 = p0v[pcode]
                nn = didx === nothing ? ds.n[i, j] : ds.n[didx, i, j]
                if nn > 0                                  # n == 0 ⇒ roster row absent ⇒ no trial
                    nz = nn - whist_nobs(pos)
                    ll += nz * log(p0) + (nn - nz) * log1p(-p0)
                end
                k1, k2, g = _weibull_moments(μ[i, j], κ, p0)
            else
                kk = exp(_softclamp(logd, -4.0, 5.0))      # dispersion ∈ ≈[0.018, 148], soft-bounded
                dd = didx === nothing ? ds.dd_count[i, j] : ds.dd_count[didx, i, j]
                ll += calculate_loglikelihood(dd, NegBin(μ[i, j], kk))
                k1, k2, g = _negbin_moments(μ[i, j], kk)
            end
            K1[i, j] = k1; K2[i, j] = k2; G[i, j] = g
        end
        return ll
    end

    # ---- contact-degree likelihood → per-week raw moments (K1=⟨k⟩, K2=⟨k²⟩, G=g) ----
    # Returned per week so the NGM builder can be applied downstream (Stage 2); Stage 1 is
    # NGM-independent. In the pooled regime the Tn entries alias one moment set.
    if cfg.constant_contacts
        # pooled: one latent field, one moment set reused for every renewal week.
        c ~ Normal(c0, 3.0)
        z ~ filldist(Normal(0, 1), P - 1)                 # 27 iid (non-centred, sum-to-zero GP)
        # Dispersion: block-linear only, no per-cell term (§4.3).
        if is_weighted(dm)
            log_kappa ~ filldist(Normal(0.0, 0.5), 2, 2)  # Weibull shape by child/adult block
            p0f ~ filldist(Beta(1.0, 1.0), A * A)         # fitted hurdle zero prob (weighted path only)
            β_disp = vec(log_kappa); p0v = p0f
        else
            log_k ~ filldist(Normal(0.0, 1.0), 2, 2)      # NegBin dispersion by block
            β_disp = vec(log_k); p0v = nothing            # NegBin models its zeros directly
        end
        # Constrained here too, though this regime is inactive: the same c↔field-mean confound
        # exists (one flat direction rather than Tn), and leaving the two branches structurally
        # different would be a trap for whoever revives `constant_contacts=true`. Note that
        # `reconstruct_mu_draws`' pooled branch is deliberately NOT updated to match — it reads the
        # `dt_intermediate_old/` generation, which is unconstrained.
        μ = _mu_matrix(c .+ η .* (ds.sz_Q * (La * z)))
        ETp = promote_type(eltype(μ), eltype(β_disp), _p0_eltype(p0v))
        K1 = Matrix{ETp}(undef, A, A); K2 = Matrix{ETp}(undef, A, A); G = Matrix{ETp}(undef, A, A)
        # vec(2×2)→bl is column-major (off-diagonal blocks bl=2/3 labelled by that order); harmless
        # as the block prior is exchangeable, and this regime is inactive. See §4.3.
        Turing.@addlogprob! _cell_moments!(K1, K2, G, μ, nothing, β_disp, p0v)
        K1w = [K1 for _ in 1:Tn]; K2w = [K2 for _ in 1:Tn]; Gw = [G for _ in 1:Tn]
        return (; K1 = K1w, K2 = K2w, G = Gw)
    else
        # per-week: SEPARABLE spatio-temporal field (§5). The age-pair field is smoothed over weeks
        # by an AR(1) over week indices 1:Tn, sharing one coefficient φ across all age-pairs; the
        # spatial kernel (ρ_diag, ρ_gap, η, La) is shared as before.
        #   • temporal kernel  Kt[s,t] = φ^|s−t|  (AR(1) ≡ exponential),  Lt = chol(Kt + jitter)
        #   • structure field  R = η·(Q·La·z·Ltᵀ)  (P×Tn)  ⟹ Cov(vec R) = η²·(Kt ⊗ M·Kage·M),
        #     each age-pair its OWN temporal path, each week the spatial kernel conditioned to
        #     sum to zero over the P pairs (see the `Ap`/`La` block above). "Independently per age
        #     pair" is the temporal factor; the pairs remain CORRELATED across age through `La`.
        #   • decoupled level  cₜ = c + σ_c·(Qt·z_c)  — scalar intercept c + an UNSMOOTHED (iid)
        #     weekly deviation, CONDITIONED TO SUM TO ZERO over the Tn weeks (`-t0`/`-lc0`, below).
        #     The spatial sum-to-zero constraint is what makes "η governs age-structure only" true
        #     rather than aspirational: without it the field's per-week mean is a second copy of cₜ
        #     and the two amplitudes are confounded.
        # ρ_diag/ρ_gap→0 ⇒ iid age-pairs, →∞ ⇒ pooled; φ→0 ⇒ iid weeks, φ→1 ⇒ pooled. Since `-lc0`
        # the φ limits describe the AGE-PAIR FIELD ONLY — the level is iid at every φ.
        # ---- TEMPORAL CORRELATION IS AR(1) (`-ar1`, 2026-08-06; RESTORED 2026-08-10) ----
        # `Kt[s,t] = φ^|s−t|` IS an AR(1) correlation matrix (equivalently the exponential / Matérn
        # 1/2 kernel), so "AR(1) per age pair, sharing the variance" needs no structural change: the
        # separable matrix-normal below already gives every pair its own temporal trajectory under
        # one shared amplitude η, and only the correlation FUNCTION moves. The SPATIAL kernel and
        # the iid level (`-lc0`) are untouched — a time-direction change only.
        #
        # ⚠ THIS BLOCK WAS SWAPPED TO MATÉRN 3/2 FOR ONE DAY (`-m32t`, 2026-08-10) AND SWAPPED BACK
        # THE SAME DAY, ON A MEASUREMENT. Read that round trip before touching this kernel again,
        # because the argument that briefly won is superficially reasonable and was WRONG for a
        # reason only a fit could expose.
        #
        # `-m32t`'s case was that two later changes had retired AR(1)'s justification: `-lc0` removed
        # the mechanism the φ→1 collapse was fatal through (the LEVEL's projected kernel
        # `Qtᵀ·Kt·Qt`, exactly 0 in the pooled limit however well conditioned `Kt` is), and `-w8h`
        # shortened the window to `n_fit + h` = 9–12 weeks, so the near-pooled regime was argued to
        # be "less worth representing". Both premises are true. The CONCLUSION did not follow, and
        # the 3-origin `-m32t` Pathfinder smoke refuted it directly: on the hurdle-Weibull path
        # **3 of 12 chains came back with ρ_time AT OR PAST THE LENGTH OF THEIR OWN WINDOW** —
        # 12.0 wk on a 9-week window, 10.2 on 9, 24.4 on 12 — i.e. an end-to-end within-window
        # correlation of 0.61–0.82 and a field collapsed to essentially one constant repeated across
        # the window. The pooled limit is still REACHED. "Less worth representing" was an inference
        # about reachability; the fit measured reachability and disagreed.
        #
        # So the operative question is not whether the pooled limit is visited (it is) but whether
        # it is SAFE to visit, and that is exactly what AR(1) buys. Measured at matched effective
        # rank — equal temporal pooling, so the comparison is not confounded by how much smoothing
        # each kernel applies. `spread` is `Lt`'s column-scale ratio, the quantity that broke NUTS on
        # 2026-08-05 (2.4 in healthy chains, 228–274 in the chains that would not mix):
        #
        #   effrank | Matérn 3/2 | AR(1)     | Kt min eig        | Lt spread   | Lc cond
        #   --------|------------|-----------|-------------------|-------------|-----------------
        #   ≈4.5    | ρ=2        | φ=0.785   | 5.0e-2 → 1.2e-1   | 2.4 → 2.6   | 61 → 21
        #   ≈1.4    | ρ=10       | φ=0.95    | 4.6e-4 → 2.6e-2   | 26.8 → 8.6  | 3.5e3 → 45
        #   ≈1.08   | ρ=26       | φ=0.99    | 2.7e-5 → 5.1e-3   | 94.2 → 23.1 | 1.6e4 → 55
        #   ≈1.03   | ρ=47       | φ=0.995   | 4.5e-6 → 2.6e-3   | 151.5→ 33.4 | 3.5e4 → 56
        #
        # A wash in NegBin's regime (φ ≈ 0.73, ρ_time ≈ 2); 4–6× better column spread and 2–3 orders
        # better min eigenvalue in hurdle-Weibull's. MECHANISM: AR(1) is MARKOV — tridiagonal
        # precision, eigenvalues decaying only polynomially — so it keeps spectral mass in the
        # non-constant directions even at φ = 0.995, where Matérn 3/2's spectrum has collapsed onto
        # one direction and `Lt`'s first column absorbs the whole field.
        #
        # ⚠ WHAT THIS DOES NOT FIX. Reverting the kernel does not stop hurdle-Weibull WANTING the
        # pooled limit — that is a property of its likelihood (p⁰ ≈ 0.95 leaves it nearly flat in
        # time), not of the kernel, and it has now surfaced under FOUR parameterisations: ρ_time
        # 20–27 wk under the pre-`-ar1` log-normal, 47–66 under the reverted `-ig`, φ → 0.9985–
        # 0.9998 under `-ar1`, and ρ_time past the window under `-m32t`. The indicated action for
        # that path is `constant_contacts = true`, which its posterior has now said five times — see
        # tasks/todo.md open items.
        # ⚠ A MODELLING difference, not only a numerical one: AR(1) paths are non-differentiable
        # (rougher week to week) and memory is LONGER at long lag — at matched lag-1 correlation
        # 0.785, lag-4 is 0.380 against Matérn 3/2's 0.140 and lag-8 is 0.115 against 0.008. That
        # was `-m32t`'s best argument and it is a real cost, accepted here for the conditioning.
        #
        # NO SOFT-CLAMP: φ ∈ (0,1) by construction (Turing's bijector), and φ^k cannot overflow, so
        # there is nothing for `_softclamp` to guard — unlike `log ρ_time`, where a stray LBFGS step
        # could overflow `exp`. `RHO_TIME_BOUNDS` is consequently UNUSED by this path (dead code
        # again, retained only so archived Matérn-temporal chains can be replayed by the mirrors).
        # Jitter stays 1e-4 (not 1e-6): the Pathfinder call is not try/caught, so a PosDefException
        # would abort the whole fit (see tasks/lessons.md).
        phi_time ~ Beta(cfg.ar1_phi_prior...)              # AR(1) coefficient, (0,1)
        Kt = [phi_time^abs(s - t) for s in 1:Tn, t in 1:Tn]
        Lt = Matrix(cholesky(Symmetric(Kt) + 1e-4 * I).L)   # DENSE (see La note above): Ltᵀ must not be a triangular type

        # ---- SUM-TO-ZERO over the Tn weeks, for the LEVEL (`-t0`, 2026-08-06) ----
        # `tz_Q = _sum_zero_basis(Tn)` is the constant Tn×(Tn−1) Helmert basis of 1^⊥, so Qt·Qtᵀ =
        # Mt = I − 11ᵀ/Tn. Whitening the temporal deviation in that subspace gives
        #     Cov(σ_c·dev) = σ_c²·(Qt·Qtᵀ) = σ_c²·Mt,
        # i.e. IID weekly deviations conditioned on Σ_t dev_t = 0 — exact, not an approximation, and
        # not a soft penalty. Identical construction to the spatial `-s0` above, one axis over.
        # Per-week marginal SD is σ_c·√(1 − 1/Tn) = 0.943–0.958·σ_c over Tn = 9..12 (`-w8h`), so
        # `gp_level_scale_prior` needs no rescaling.
        #
        # ---- THE LEVEL'S TEMPORAL SMOOTHING IS GONE (`-lc0`, 2026-08-09, user request) ----
        # This block used to whiten through `Lc = chol(Qtᵀ·Kt·Qt + 1e-4·I)`, i.e. the level was a
        # 1-D process on the SAME `Kt` as the field, giving Cov = σ_c²·(Mt·Kt·Mt). It now whitens
        # through the identity. **`phi_time` therefore reaches the likelihood ONLY through `Lt`, i.e.
        # only through the per-age-pair temporal correlation** — which is the whole point of the
        # change: one kernel per age pair and nothing else. It survived BOTH kernel swaps of
        # 2026-08-10 unchanged (each request was to swap the family, not to re-smooth the level).
        #
        # A second, structural benefit falls out, and it is why the pooled limit is no longer
        # dangerous. Under `-ar1` + `-t0` a φ→1 degeneracy destroyed whole fits, and it was a
        # property of THIS projection rather than of `Kt`: `Qtᵀ·J·Qt = 0` EXACTLY, so in the pooled
        # limit the projected kernel went to zero however well conditioned `Kt` was, `Lc` collapsed
        # to `chol(1e-4·I)`, and the weekly level died into the jitter (marginal SD per unit σ_c:
        # 0.958 at φ=0 → 0.260 at 0.99 → 0.028 at 0.9999). With `Lc` removed the level's amplitude is
        # φ-independent by construction and that failure mode cannot occur. ⚠ It does NOT rescue the
        # FIELD side — `Lt`'s first column still absorbs the field as φ→1 — which is what
        # `ar1_phi_prior` restrains, and which is why removing `Lc` did NOT make the kernel choice
        # irrelevant (the `-m32t` round trip of 2026-08-10 tested exactly that inference and the
        # smoke refuted it; see the kernel block above).
        #
        # WHY (unchanged by `-lc0` — the confound is with the deviation's MEAN, not its correlation):
        # `c` and the time-MEAN of the weekly deviation are two parameterisations of the same
        # quantity, and the flat direction that creates was measured on all four `-m32` chains at
        # corr(c, time-mean deviation) = −1.000 EXACTLY, with SD(c) ≈ SD(deviation) ≈ 0.38–0.73 but
        # SD(their sum) = 0.007 — the components cancel to 1–2% of their own spread while the mean
        # level itself is pinned by the data. `z_c` is Tn−1, not Tn. Stage-1 latent count under
        # `-w8h` (Tn = n_fit + h = 9..12): 5 + 32·Tn = **293/325/357/389** (NegBin), 5 + 81·Tn =
        # **734/815/896/977** (hurdle-Weibull) at h = 1..4.
        # ⚠ Do NOT date a chain by its dimension. 389/977 is ALSO what `-diag` and `-t0-ar1` had (at
        # their flat Tn = 12), and NEITHER kernel swap of 2026-08-10 changed a dimension at all —
        # each replaced one scalar by one scalar. Identify the model by the token and by which names
        # are present: `phi_time` ⇒ AR(1) (current, and `-ar1`), `log_rho_time` ⇒ a Matérn 3/2
        # temporal kernel (the one-day `-m32t` generation, and everything before `-ar1`).
        #
        # LEVEL ONLY — do NOT also project the structure field's time axis. `R`'s per-pair mean over
        # weeks duplicates nothing (no other parameter carries persistent age-pair structure), so
        # constraining it would force every pair's structure to average to zero across the window:
        # a model restriction, not a reparameterisation. The field keeps the full `Lt`.
        c ~ Normal(c0, 3.0)                               # scalar level intercept (stored)
        log_sigma_c ~ Normal(cfg.gp_level_scale_prior[1], cfg.gp_level_scale_prior[2])
        σ_c = exp(_softclamp(log_sigma_c, -3.0, 2.0))     # temporal-level amplitude, soft-bounded (mirrors η)
        tz_Q = _sum_zero_basis(Tn)                        # Tn×(Tn−1), constant — same helper as the spatial basis
        # NB `tz_Q` is the TEMPORAL basis; `ds.sz_Q`/`ds.sz_Qt` are the SPATIAL one and its transpose.
        # No Cholesky and no jitter here any more (`-lc0`): the whitening matrix IS the identity, so
        # there is no near-singular factor left to regularise on this axis. `Lt`'s 1e-4 stays.
        z_c ~ filldist(Normal(0, 1), Tn - 1)              # temporal-level raw (non-centred, sum-to-zero, IID)
        c_vec = c .+ σ_c .* (tz_Q * z_c)                   # per-week level cₜ, deviation sums to 0 over t

        # (P−1)×Tn, NOT P×Tn: the field lives in the sum-to-zero subspace. Keep the name `z` —
        # `_stage1_init` selects the non-centred blocks by the PREFIX `startswith(string(k), "z")`
        # and would silently return `nothing` (⇒ Pathfinder falls back to `UniformSampler(2)`, no
        # warning) if this were renamed to something outside that prefix.
        z ~ filldist(Normal(0, 1), P - 1, Tn)             # structure field raw (shared spatial+temporal kernel)
        # Dispersion: block-linear × week, NO per-cell term (§4.3). `log_k`/`log_kappa` is a 4×Tn
        # array (block-linear rows × week), so every ordered cell in a child/adult block shares that
        # week's value:  log_disp_{ij,t} = β[bl,t]. It is per-week iid — NOT temporally smoothed,
        # unlike the mean field. Everything ≤2-D so generated_quantities can reconstruct it (a 3-D
        # filldist can't be). The per-cell RE that sat here from 2026-07-30 to 2026-08-02 (flat
        # hierarchy, then regularised horseshoe) was removed — see `_cell_moments!`.
        if is_weighted(dm)
            log_kappa ~ filldist(Normal(0.0, 0.5), 4, Tn)      # shape by block-linear × week
            p0f ~ filldist(Beta(1.0, 1.0), A * A, Tn)          # fitted hurdle zero prob (weighted path only)
            β_disp = log_kappa; p0m = p0f
        else
            log_k ~ filldist(Normal(0.0, 1.0), 4, Tn)          # dispersion by block-linear × week
            β_disp = log_k; p0m = nothing                      # NegBin models its zeros directly
        end
        # precompute the whole spatio-temporal field ONCE (the temporal coupling means each
        # week's column depends on ALL columns of z, so it can't be sliced per week). Fld
        # already carries η; don't re-apply it below.
        Fld = η .* (ds.sz_Q * (La * z * Lt'))              # P×Tn, columns sum to 0 exactly
        # Miss any of these and K1/K2/G are allocated as Float64, which silently cuts the AD tape
        # for that latent. `β_disp` reaches K1/K2/G only through `_cell_moments!`, whose own
        # `zero(eltype(K1))` then has to carry it — so it must be in this promotion too.
        ETp = promote_type(typeof(c), eltype(Fld), eltype(β_disp), _p0_eltype(p0m))
        K1w = Vector{Matrix{ETp}}(undef, Tn)               # per-week raw moments (NGM applied downstream)
        K2w = Vector{Matrix{ETp}}(undef, Tn)
        Gw  = Vector{Matrix{ETp}}(undef, Tn)
        ll = zero(ETp)
        for t in 1:Tn
            K1 = Matrix{ETp}(undef, A, A); K2 = Matrix{ETp}(undef, A, A); G = Matrix{ETp}(undef, A, A)
            μ = _mu_matrix(c_vec[t] .+ @view Fld[:, t])
            # `view(...)` (function form), NOT a space-form `@view a, @view b`: in an argument
            # list the macro greedily swallows the following args ("Invalid use of @view macro").
            ll += _cell_moments!(K1, K2, G, μ, t, view(β_disp, :, t),
                                 p0m === nothing ? nothing : view(p0m, :, t))
            K1w[t] = K1; K2w[t] = K2; Gw[t] = G           # fresh matrices per week (not reused buffers)
        end
        Turing.@addlogprob! ll
        return (; K1 = K1w, K2 = K2w, G = Gw)
    end
end

# ======================================================================================
# Stage 2 — infection / renewal block, conditioning on a FIXED per-week C* trajectory.
# `Cstar_weeks` is one Stage-1 draw's moments run through `contact_star(nb, …)`. Since `-w8`
# (2026-08-09) it has length `Tn − smax` = `cfg.n_fit`, NOT `Tn`: the contact window spans only the
# renewal's FITTING weeks (`prepare_degree_data` → `win.fit_weeks`), while `wd` still carries the
# `smax` lag weeks of infection history. Since `-w8h` the contact window is `[t₀−n_fit+1 … t₀+h]`,
# so its LAST `n_fit` columns are the ones the renewal reads: `Cstar_weeks[k + h]` pairs with `wd`
# week `k + smax`, and the likelihood indexes it as `Cstar_weeks[t - cfg.smax + off]` with
# `off = length(Cstar_weeks) - cfg.n_fit` == h. The length relation is ASSERTED below — a silent
# off-by-`smax`/`h` here would score a forecast built on the wrong weeks' contacts.
# Samples the transmission latents and the renewal likelihood;
# C* is NOT re-scaled (the -gnorm S̄ decoupling was reverted), so `gamma_sar` is the per-contact
# secondary attack rate and reproduces the reference cell N_{ref,ref} = susc_ref·inf_ref = γ_SAR
# directly (ref = cfg.ref_bin, default 4 = "25-34"; formerly bin 1 = "2-10").
#
# `nb` is passed ONLY so the model can honour `fix_infectivity(nb)` (the NO-INTERACTION variant,
# inst/6); the C* functional itself has already been applied upstream. It defaults to `MeanNGM()`
# so older call sites keep the unconstrained behaviour.
# ======================================================================================
@model function model_transmission(Cstar_weeks, wd::WindowData, cfg::FrameworkConfig,
                                   nb::NGMBuilder = MeanNGM())
    A = wd.A
    Tn = length(wd.weeks)
    # `-w8h`: the contact window is `[t₀−n_fit+1 … t₀+h]`, length `n_fit + h`, while `wd` spans the
    # 12 `all_weeks`. The likelihood needs the LAST `n_fit` contact columns (contacts h weeks ahead
    # of their fit week), so the index offset IS h — and h is recoverable from the length rather
    # than passed in, which keeps this model's signature free of the horizon:
    off = length(Cstar_weeks) - cfg.n_fit
    # Checked, not assumed: the two week vectors are aligned POSITIONALLY and nothing downstream
    # would notice a wrong-length `Cstar_weeks` except by producing subtly wrong forecasts. (A
    # length check cannot catch a wrong-DATED window of the right length — `stage2_inputs` asserts
    # `apd_h.weeks[1:n_fit] == win0.fit_weeks` for that.)
    @assert 0 <= off <= maximum(cfg.horizons) "model_transmission: Cstar_weeks has \
$(length(Cstar_weeks)) weeks; expected n_fit + h = $(cfg.n_fit) + h for h in $(cfg.horizons) \
(see the `-w8h` note above)"

    # ---- transmission latents: per-contact SAR γ_SAR + relative susc/inf (analysis-plan form) ----
    # γ_SAR (per-contact secondary attack rate) carries the NGM level; inherent susceptibility &
    # infectivity are RELATIVE, normalised so the reference bin `cfg.ref_bin` (default 4 = "25-34") = 1
    # (the other A-1 bins estimated), so `z_s`/`z_i` have length A-1 and the fixed 1 is SPLICED in at
    # `ref_bin`. The choice of reference is a gauge (likelihood-invariant); it acts only through the
    # priors. NGM index convention: susc on susceptible row a, inf on
    # infectious column b (Munday Eq 3).
    log_gamma_sar ~ Normal(cfg.gamma_sar_prior[1], cfg.gamma_sar_prior[2])  # centre log(0.1), LOOSENED to 90% γ_SAR∈[0.0052,1.93]
    gamma_sar = exp(_softclamp(log_gamma_sar, log(0.001), log(10.0)))       # secondary attack rate, soft-bounded to [0.001,10] (was [0.02,5]; low bound was pinning negbin|neighbourhood ~0.021)

    # susc/inf are RELATIVE (bin `ref` = 1). The PRIOR controls the typical age spread and the
    # SOFT-CLAMP is a looser safety bound. The offset scale `sig ~ N⁺(0, 0.25²)` (SET 2026-07-31,
    # user request; a mode-at-0 half-normal — the conventional weakly-informative scale, so the age
    # profile can SHRINK to no-variation when the data are silent, rather than being asserted at ≈0.5
    # log-SD as the previous N⁺(0.5,0.25²) did; marginal SD still ≤~0.5 ⇒ realistic susc/inf stay well
    # inside the clamp) — admits real age variation but does not impose it.
    # The log-offset soft-clamp [log 0.05, log 20] ≈ [−3.0, +3.0] ⇒ HARD-bounds susc/inf ∈ [0.05, 20]
    # (LOOSENED 2026-07-13 from [log 0.2, log 5], user request — matching the wider prior, and now
    # re-aligned with the `-sc` cache-token docstring): the prior's ±2 SD is well interior (clamp at
    # ~±6 SD), so realistic profiles stay off it, but it still caps a stray Stage-2 Pathfinder draw
    # that would otherwise send `σ·z` to ±100 → `exp` ~1e8 → supercritical/Inf NGM (mirrors κ/γ_SAR).
    #
    # NO CROSS-BIN SMOOTHING (2026-07-13, user request): the A-1 non-reference offsets are INDEPENDENT
    # per age bin — `sig·z` with z ~ iid Normal(0,1). The shared-length-scale RBF GP (`log_rho_si`,
    # `Ksi`, `Lsi`, offset `sig·(Lsi·z)`) that previously correlated neighbouring bins was removed; only
    # the marginal-SD prior `sig_s`/`sig_i` and the soft-clamp remain, so the [0.37,2.7]/[0.05,20]
    # calibration above is per-bin (the GP had unit diagonal ⇒ dropping it leaves per-bin SD = sig).
    # See tasks/lessons.md 2026-07-13 (and the GP→RW1→RW2→GP history before it).
    sig_s ~ truncated(Normal(cfg.susc_inf_sd_prior[1], cfg.susc_inf_sd_prior[2]); lower = 0)
    z_s ~ filldist(Normal(0, 1), A - 1)                    # A-1 non-reference offsets (all bins ≠ ref), independent
    offs_s = exp.(_softclamp.(sig_s .* z_s, log(0.05), log(20.0)))                  # ∈ [0.05,20]
    susc = vcat(offs_s[1:cfg.ref_bin-1], one(sig_s), offs_s[cfg.ref_bin:end])       # susc[ref]=1; the A-1 offsets fill the other bins in order

    # NO-INTERACTION model (inst/6): with a DIAGONAL C* the NGM is diagonal, so
    # N_aa = γ_SAR·susc_a·(1+(F−1)A_a)·C*_aa·inf_a — susc_a and inf_a enter only through their
    # product and are individually non-identifiable. Infectivity is therefore pinned to 1 in every
    # bin and the whole age profile is carried by `susc`. `sig_i`/`z_i` are NOT sampled at all (an
    # unused latent would just be prior-driven noise in the Pathfinder approximation).
    if fix_infectivity(nb)
        inf = ones(A)                       # plain Float64: constant, no gradient flows through it
    else
        sig_i ~ truncated(Normal(cfg.susc_inf_sd_prior[1], cfg.susc_inf_sd_prior[2]); lower = 0)
        z_i ~ filldist(Normal(0, 1), A - 1)
        offs_i = exp.(_softclamp.(sig_i .* z_i, log(0.05), log(20.0)))                   # ∈ [0.05,20]
        inf = vcat(offs_i[1:cfg.ref_bin-1], one(sig_i), offs_i[cfg.ref_bin:end])         # inf[ref]=1
    end

    # ⚠ TEMPORARY (2026-08-04, user request): the ANTIBODY / TITRE TERM IS DISABLED by pinning the
    # leaky protection factor to F ≡ 1. Then full_susceptibility_a(t) = susc_a·(1 + 0·A_a(t)) = susc_a
    # EXACTLY, so `wd.antibody` drops out of the NGM entirely. `ngm.jl` is deliberately left alone —
    # `build_ngm`/`full_susceptibility` stay general and are simply called with F = 1.0, and the
    # multiplier (F−1) is exactly 0.0. That is only NaN-safe because `weekly_antibody` ZERO-fills
    # unmatched weeks rather than NaN-filling (infection_data.jl; 0.0*NaN would be NaN).
    #
    # F is NOT sampled — same reasoning as `fix_infectivity` above (inst/3 §"Fixed infectivity"):
    # an unused latent stays prior-driven and pollutes the Pathfinder approximation, so it is dropped
    # from the parameter space rather than merely ignored. It IS still returned (as the constant 1.0)
    # so every downstream consumer keeps working untouched: `fit_stage2_pooled`'s `q.F` → `pooled.F`,
    # `two_stage_forecast`, `reproduction_draws` (8j), `fit_window_infection_draws` (10j) and
    # `plot_F` (9j), whose panel now shows a flat 1.0 line that documents the term being off.
    #
    # ⚠ CACHE: this changes the Stage-2 posterior and `contacts_label` does NOT encode it (same trap
    # as `ref_bin`/`gamma_sar_prior`) — delete any `8j_s2_*` and the derived `9j_assembly_*`/`9j_rt_*`/
    # `9j_relrt_*`/`9j_obsrt_*` before refitting. Stage-1 `8j_s1_*` chains have no F and are
    # UNAFFECTED; do NOT bump the token, it is shared with Stage 1 and a bump would orphan them all.
    #
    # TO RESTORE the antibody term: swap the two lines below back.
    # F ~ Beta(5, 1)
    F = 1.0
    sigma_inf ~ truncated(Normal(0.05, 0.025); lower = 0)

    # ---- generation interval, ESTIMATED (2026-07-30; Munday 2023 Eq 2 + Table 1, §3.1) ----
    # Prior centres are the moment-matched log-parameters of cfg.gen_mean_days/gen_sd_days
    # (5d/5d ⇒ (w_mu0, w_sigma0) = (−0.6830, log2 = 0.6931)), with Munday p.8's "SD = 20% of the
    # prior mean". `w_sigma` is the LOG-VARIANCE (see gen_interval_logparams), so sdlog = √w_sigma.
    #
    # Only w_sigma is truncated at 0. The paper prints T[0,] on BOTH and a *negative* prior SD for
    # w_mu, which is not a valid statement — and truncating w_mu at 0 would be wrong regardless: a
    # 5-day GI is shorter than a week, so meanlog = −0.683 < 0 is REQUIRED. Hence `abs(...)` on the
    # SD and no truncation on w_mu. Deliberate, documented departure from the printed table.
    #
    # Soft-clamps are outer safety bounds (codebase idiom), far outside the prior's ±2 SD:
    # w_mu ∈ [log(1/7), log 3] ⇒ GI mean ≈ 1 day .. 3 weeks; w_sigma ∈ [0.02, 4]. At either bound
    # F(smax) ≥ 0.55, so gen_interval_pmf_log's division by F(smax) cannot blow up.
    #
    # ⚠ IDENTIFIABILITY: w and gamma_sar are confounded — both scale the renewal predictor, so
    # raising w₁ and lowering gamma_sar nearly compensate over an 8-week window. The 20% prior SD
    # is what keeps the pair identified; do NOT loosen it. Check the posterior against the prior
    # (9j `plot_gen_interval`): equal to the prior ⇒ the GI is adding nothing; parked on a clamp
    # with a tight CI ⇒ clamp compression, not certainty (cf. the γ_SAR≈0.021 episode).
    wmu0, wv0 = gen_interval_logparams(cfg.gen_mean_days, cfg.gen_sd_days)
    r = cfg.gen_prior_rel_sd
    w_mu ~ Normal(wmu0, abs(wmu0) * r)                                   # meanlog (weeks)
    w_sigma ~ truncated(Normal(wv0, abs(wv0) * r); lower = 0)            # LOG-VARIANCE
    # POST-CLAMP values — these, not the raw latents, are what the likelihood used, and they are
    # what gets returned/stored so `gen_interval_pmf_log(w_mu, w_sigma)` downstream (the forecast,
    # the 10j fit-window reconstruction, the 9j GI panel) reproduces this fit exactly. Same
    # convention as gamma_sar/susc/inf, which are also returned post-clamp.
    # `W_MU_BOUNDS`/`W_SIGMA_BOUNDS`/`W_GI_SOFT` (framework.jl) — read that docstring before touching
    # them. The upper `w_mu` bound is the `F(smax)` guard and log(3) is its maximum; the tighter
    # `W_GI_SOFT` is required because `w_sigma` is a VARIANCE whose floor is pinned near 0.
    w_mu_e    = _softclamp(w_mu,    W_MU_BOUNDS...,    W_GI_SOFT)
    w_sigma_e = _softclamp(w_sigma, W_SIGMA_BOUNDS..., W_GI_SOFT)
    w = gen_interval_pmf_log(w_mu_e, w_sigma_e; smax = cfg.smax)

    # ---- infection likelihood over the fitting weeks (t > smax); NGM uses week-t C* ----
    # (contacts vary by week; C*_t is the fixed `Cstar_weeks[t - smax + off]` — the contact window
    # starts at `wd`'s first FITTING week and runs `h` weeks past the origin, `-w8h`, so the renewal
    # reads its last `n_fit` columns. `wd.antibody[:, t]` is still passed but has NO effect
    # while F ≡ 1 — see the TEMPORARY block above.)
    for t in (cfg.smax + 1):Tn
        N = build_ngm(Cstar_weeks[t - cfg.smax + off], susc, inf, F, wd.antibody[:, t]; gamma_sar = gamma_sar)
        pred = renewal_next(N, wd.I_mean, t, w)
        for a in 1:A
            σ = sqrt((sigma_inf * wd.I_mean[a, t])^2 + wd.I_sd[a, t]^2)
            wd.I_mean[a, t] ~ Normal(pred[a], σ)
        end
    end

    return (; susc, inf, F, gamma_sar, sigma_inf, w_mu = w_mu_e, w_sigma = w_sigma_e)
end

# --------------------------------------------------------------------------------------
# Cache filenames. Stage-1 chains carry NO ngm token (NGM-independent ⇒ one fit serves both
# builders); Stage-2 pooled results carry both degree and ngm.
# --------------------------------------------------------------------------------------
stage1_path(dm::ContactDegreeModel, origin::Date, h::Integer; contacts::AbstractString,
            save_dir::AbstractString) =
    joinpath(save_dir, "8j_s1_$(degree_label(dm))_$(contacts)_$(origin)_h$(h).jld2")

stage2_path(dm::ContactDegreeModel, nb::NGMBuilder, origin::Date, h::Integer;
            contacts::AbstractString, save_dir::AbstractString) =
    joinpath(save_dir, "8j_s2_$(degree_label(dm))_$(ngm_label(nb))_$(contacts)_$(origin)_h$(h).jld2")

"""
    _atomic_jldsave(path; kwargs...)

`jldsave` to a sibling temporary file, then `mv` onto `path`. Use this for EVERY grid artefact.

A plain `jldsave(path; …)` writes in place, so a process killed mid-write leaves a TRUNCATED
`.jld2` — and every skip-check in this codebase (`fit_or_load_stage1`, `prefit_stage1!`/
`prefit_stage2!`, `8j_run_grid.jl`'s `s1_missing`/`s2_missing`, 11j's up-front assert) tests
`isfile`, so a truncated file counts as COMPLETE. The grid then reports itself finished and fails
later at read time, one cell at a time. Rare on a workstation, routine under Slurm: the time limit
arrives as SIGTERM→SIGKILL at an arbitrary instant, and NUTS artefacts are ~28/72 MB, i.e. a wide
window to be interrupted in. `mv` within one filesystem is a rename and therefore atomic — the
target either is the old file or the complete new one.

The temp name carries pid and thread id because `prefit_*!` fan out over threads, and lives beside
the target so the rename cannot cross a filesystem. `.tmp.*` does not match the `8j_s1_*`/`8j_s2_*`
globs the audit and readers use.
"""
function _atomic_jldsave(path::AbstractString; kwargs...)
    tmp = string(path, ".tmp.", getpid(), ".", Threads.threadid())
    try
        jldsave(tmp; kwargs...)
        mv(tmp, path; force = true)
    catch
        isfile(tmp) && rm(tmp; force = true)
        rethrow()
    end
    return path
end

"""
    _stage1_init(model, z_scale, rng, cfg)

Explicit starting point for the Stage-1 LBFGS path, as an **unconstrained** vector.

Every latent is drawn from its prior *except* two blocks, both shrunk toward the centre of the
**unconstrained** space because they are weakly identified and the LBFGS path largely ends where it
starts:

1. the standard-normal non-centred terms — any variable whose name starts with `z` (`z`, `z_c`) —
   drawn from `N(0, z_scale²)` instead of `N(0,1)`. These dominate the parameter space (251 of
   293/734 unconstrained coordinates at Tn = 9, i.e. h=1; Tn = n_fit + h varies 9..12 since `-w8h`);
2. the AR(1) coefficient `phi_time`, drawn as `logistic(N(0, cfg.stage1_phi_init_scale²))`
   (2026-08-10, user request) — i.e. **`N(0, 0.1²)` ON THE LOGIT SCALE**, which is where Turing's
   bijector puts it, giving φ ≈ 0.5 ± 0.025. Skipped when the model has no `phi_time`
   (`constant_contacts = true` has no temporal kernel).

⚠ The two are shrunk toward *different things* and it matters. `z` is identity-linked (a standard
normal is already unconstrained), so `N(0, 0.1²)` is 1/10 of prior scale, near the prior mean, and
`z = 0` means "no field". φ is LOGIT-linked, so `N(0, 0.1²)` unconstrained is φ ≈ 0.5 — the prior
median, *not* a small φ. "Start φ small" (φ₀ = 0.1, weeks nearly independent) is a different
operation and is NOT what this does.

**MEASURED on the cell that motivated it** (weighted-hweibull @ 2021-05-02 h1, through the shipped
code path with the driver's `Xoshiro(1236)`):

| | φ median | φ spread | log_eta | max abs(z) |
|---|---|---|---|---|
| prior-draw init (before) | **1.000000** | 7.2e-10 | −1.56 | 4.38 |
| `logistic(N(0,0.1²))` (now) | **0.820** | 4.4e-02 | −0.48 | 2.39 |

The boundary collapse is gone: φ comes back interior with real posterior spread, and the rest of the
fit is healthier too (`log_eta` closer to its prior mean, `max|z|` down). That matters beyond φ
itself because `_pf_mean_init` hands the Pathfinder mean to NUTS as `initial_params` — a finite logit
is a workable NUTS start, `logit(1.0)` is not.

⚠ **It does not IDENTIFY φ.** The initialisation probe (`inst/3` §12.6) held everything fixed but φ₀
and found the final φ largely determined by the start on both degree models (hurdle-Weibull
0.011→1.000 non-monotonically, NegBin 0.150→0.726). A defined, reproducible start replaces an
arbitrary one; it does not make the data informative. Read φ from NUTS, not from Pathfinder.

The name filter is a **prefix**, not a fixed list, so it automatically covers any future `z*` block;
it also covered the dispersion RE's `z_kappa`/`z_k` while that existed (2026-07-30 → 2026-08-02).

Returns `nothing` when `z_scale ≤ 0`, which leaves Pathfinder on its own `UniformSampler(2)` default
(U(-2,2) per unconstrained coordinate).

Built by round-tripping a `NamedTuple` through `InitFromParams` + `link!!` rather than by writing
into index ranges of the flat vector. The ranges happen to be contiguous, but that is an
implementation detail of DynamicPPL's variable ordering — it would shift the moment a latent is
added, reordered, or made conditional, and silently initialise the wrong block. The named round-trip
cannot go wrong that way.

Note these `z`s are identity-transformed under `link!!` (a standard Normal is already
unconstrained), so the requested SD is the SD *in the space Pathfinder optimises*, not merely on
the constrained scale.
"""
function _stage1_init(model, z_scale::Real, rng, cfg::FrameworkConfig)
    z_scale > 0 || return nothing
    vi = DynamicPPL.VarInfo(rng, model)
    nt = DynamicPPL.values_as(vi, NamedTuple)
    zk = filter(k -> startswith(string(k), "z"), keys(nt))
    isempty(zk) && return nothing
    vals = NamedTuple{Tuple(zk)}(Tuple(z_scale .* randn(rng, size(nt[k])) for k in zk))
    # φ, on the scale Turing's bijector actually optimises. Set CONSTRAINED here (InitFromParams
    # takes constrained values and the `link!!` below transforms) — so the normal draw is pushed
    # through the logistic, which is the inverse of the logit link. Doing it the other way round
    # would put N(0,0.1²) on the (0,1) scale and immediately leave the support.
    φs = cfg.stage1_phi_init_scale
    if φs > 0 && haskey(nt, :phi_time)
        u = φs * randn(rng)
        vals = merge(vals, (; phi_time = 1 / (1 + exp(-u))))
    end
    _, vi2 = DynamicPPL.init!!(rng, model, DynamicPPL.VarInfo(),
                               DynamicPPL.InitFromParams(merge(nt, vals)))
    # `collect(Float64, …)` is REQUIRED, not tidying: the InitFromParams round-trip yields a
    # `Vector{Real}` (non-concrete eltype), and Optimization.jl rejects that outright —
    # "Non-concrete element type inside of an `Array` detected. Element type: Real" — so the fit
    # dies before the first gradient. The vector itself is already correct at that point, which is
    # why a length/finiteness/SD check on it passes while the fit still fails.
    return collect(Float64, DynamicPPL.link!!(vi2, model)[:])
end

"""
    _fit_pathfinder(model, ndraws, nruns, rng, adtype; init=nothing)

Run Pathfinder on `model`, single- or multi-path depending on `nruns`.

`init` (single-path only) is an explicit unconstrained starting vector from `_stage1_init`;
`nothing` leaves Pathfinder on its `UniformSampler(2)` default. It is not forwarded to
`multipathfinder`, which derives its own per-path inits and throws if given both `init` and `nruns`.

**`nruns > 1` ⇒ `multipathfinder`** — `nruns` independent LBFGS paths, then Pareto-smoothed
importance resampling to `ndraws`. This is the robustness fix for Stage-1 divergence (2026-07-30):
a single path diverged into the soft-clamp's flat region in **2 of 5 seeds** on the hurdle-Weibull
path. A diverged path lands where the log-posterior is minuscule, so its importance weights are
negligible and the resampling all but discards it — provided at least one path is healthy.

Two API details that matter:
- `ndraws` is POSITIONAL for `multipathfinder` but a keyword for `pathfinder`.
- `nruns` has **no usable default** (it is `-1` unless `init` is supplied, which then throws), so it
  must be passed explicitly.
- `executor` is left at its `SequentialEx()` default **deliberately**: `prefit_stage1!` already fans
  fits out over Julia threads under a semaphore, so letting multipathfinder thread internally would
  oversubscribe and (per its own docs) requires a thread-safe `rng` and log-density.

Warns when the Pareto shape k > 0.7 — the standard threshold above which importance resampling is
unreliable, i.e. the paths disagree so badly that the pooled draws should not be trusted.
"""
function _fit_pathfinder(model, ndraws::Int, nruns::Int, rng, adtype; init = nothing)
    if nruns <= 1
        kw = (; ndraws = ndraws, rng = rng)
        adtype === nothing || (kw = merge(kw, (; adtype = adtype)))
        init === nothing   || (kw = merge(kw, (; init = init)))
        return pathfinder(model; kw...)
    end
    pf = adtype === nothing ?
        multipathfinder(model, ndraws; nruns = nruns, rng = rng) :
        multipathfinder(model, ndraws; nruns = nruns, rng = rng, adtype = adtype)
    k = try
        pf.psis_result === nothing ? nothing : pf.psis_result.pareto_shape
    catch
        nothing
    end
    (k !== nothing && k > 0.7) &&
        @warn "multipathfinder: Pareto k > 0.7 — importance resampling unreliable, paths disagree" k nruns
    return pf
end

"""
    _pf_mean_init(model, pf, rng) -> DynamicPPL.InitFromParams

Starting point for Stage-1 NUTS: the Pathfinder mean, as a **structured** `InitFromParams`.

⚠ THIS MUST NOT BE BUILT FROM CHAIN PARAMETER NAMES. `InitFromParams` resolves a `NamedTuple` by
**VarName symbol** (`hasvalue(params, vn, dist)`), not by the flattened label MCMCChains uses. The
former implementation here was

    pnames = names(pf.draws_transformed, :parameters)   # Symbol("z[1,1]"), Symbol("p0f[3,7]"), …
    InitFromParams(NamedTuple(zip(pnames, means)))

and every one of those keys FAILS to match its varname (`z`, `p0f`, …), so each array-valued latent
fell through to the default `fallback = InitFromPrior()` — **silently**, because falling back is
`InitFromParams`'s documented behaviour, not an error. `model_degree` declares only 6 scalars
against 384 (NegBin) / 972 (hurdle-Weibull) array coordinates, so NUTS was starting from the prior
in ~99% of the space while appearing to start from Pathfinder. Verified on DynamicPPL 0.39.15: a
`z ~ MvNormal(zeros(4), I)` seeded at 9.0 came back as prior draws, while a sibling scalar was
honoured. That defeats the whole point of `_stage1_init` (the `z*` blocks "dominate the parameter
space … and are only weakly identified, so where the path starts largely decides where it ends").

The mean is taken in the **UNCONSTRAINED** space — `pf.fit_distribution` is the ELBO-maximising
`MvNormal` Pathfinder actually fitted, so its `μ` is the approximation's mode. That is both the
natural "Pathfinder mean" and strictly better than averaging `pf.draws_transformed`, which would
average `p0f` on the constrained `[0,1]` scale. `multipathfinder` returns a mixture and exposes no
single `fit_distribution`, so fall back to the mean of its (also unconstrained) `draws`, which are
`dim × ndraws`.

The unconstrained vector is turned into a `NamedTuple` by the same `link!!`/`values_as` round-trip
`_stage1_init` uses, rather than by writing into index ranges — see that function's docstring for
why the named round-trip is the only form that cannot silently target the wrong block.
"""
function _pf_mean_init(model, pf, rng)
    u = hasproperty(pf, :fit_distribution) && pf.fit_distribution !== nothing ?
        collect(Float64, mean(pf.fit_distribution)) :
        vec(mean(pf.draws; dims = 2))                      # multipathfinder: mixture, no single fit
    vil = DynamicPPL.link!!(DynamicPPL.VarInfo(rng, model), model)
    length(u) == length(vil[:]) ||
        error("_pf_mean_init: Pathfinder dim $(length(u)) ≠ model unconstrained dim $(length(vil[:]))")
    nt = DynamicPPL.values_as(DynamicPPL.invlink!!(DynamicPPL.unflatten(vil, u), model), NamedTuple)
    # Guard the failure mode above: every model varname must be covered, or the missing ones would
    # be silently re-drawn from the prior. `values_as` is built FROM the model, so a mismatch here
    # means the round-trip itself broke — fail loudly rather than sample from a half-prior start.
    expect = keys(DynamicPPL.values_as(DynamicPPL.VarInfo(rng, model), NamedTuple))
    missing_keys = setdiff(expect, keys(nt))
    isempty(missing_keys) ||
        error("_pf_mean_init: init misses model varnames $(missing_keys) — would fall back to prior")
    lj = DynamicPPL.logjoint(model, nt)
    isfinite(lj) || error("_pf_mean_init: Pathfinder mean has non-finite logjoint ($lj)")
    return DynamicPPL.InitFromParams(nt)
end

"""
    _nuts_diagnostics(chn) -> NamedTuple

Post-fit health summary for a Stage-1 NUTS chain: divergence count, the fraction of transitions
that hit `max_depth` (a saturated tree means the sampler never terminated by U-turn, i.e. the step
size is too small for the geometry), and the minimum ESS over parameters.

**There is no R̂ here — Stage 1 samples ONE chain per fit** (`prefit_stage1!` already fans the 504
fits out over threads under a semaphore with BLAS pinned to 1, so per-fit chain threading would
oversubscribe; this is the same argument `_fit_pathfinder` makes for leaving multipathfinder's
`executor` sequential). Do not read `turing_utils.jl`'s `Rhat < 1.1` convergence check as applying
to these chains — it cannot, with one chain.

Every field is `missing` when the chain does not carry the corresponding internal, so this is safe
to call on a Pathfinder `Chains` too — which has **no `:internals` section at all**. Note
`names(chn, section)` is a bare `name_map[section]` lookup and therefore throws `KeyError` on a
missing section; go through `MCMCChains.sections` first rather than calling it speculatively.
"""
function _nuts_diagnostics(chn)
    intern = :internals in MCMCChains.sections(chn) ? names(chn, :internals) : Symbol[]
    getcol(s) = Symbol(s) in intern ? vec(Array(chn[:, Symbol(s), :])) : nothing
    div_col = getcol("numerical_error")
    depth   = getcol("tree_depth")
    ndiv = div_col === nothing ? missing : count(x -> x === true || x == 1, div_col)
    dmax = depth === nothing ? missing : maximum(depth)
    fmax = depth === nothing ? missing : count(==(dmax), depth) / length(depth)
    # `ess(chn)` returns a ChainDataFrame; read its NamedTuple directly (the `.nt.ess` idiom
    # MCMCChains itself uses) rather than round-tripping through DataFrame.
    min_ess = try
        minimum(skipmissing(MCMCChains.ess(chn).nt.ess))
    catch
        missing
    end
    return (; divergences = ndiv, max_tree_depth = dmax, frac_at_max_depth = fmax,
              min_ess, n_draws = size(chn, 1))
end

"""
    fit_stage1(dm, ds, pop, cfg; use_nuts=cfg.stage1_use_nuts, ndraws_pf, n_sample=250)

Stage-1 (contact-degree GP) fit: Pathfinder init → (optionally) NUTS on `model_degree`.
Returns `(; chn, model, pf, sampler, diag)`. `chn` is the Pathfinder approximate posterior
(default) or the NUTS chain; both carry the same GP parameter names. `sampler` is `:pathfinder` or
`:nuts` and `diag` is `_nuts_diagnostics(chn)` (all-`missing` on the Pathfinder path), so a saved
artefact can say which sampler produced it. `ndraws_pf`/`n_sample` are kept ≥ `cfg.n_stage1_post`
so there are always enough draws to impute into Stage 2.

Uses **multi-path** Pathfinder when `cfg.stage1_pathfinder_runs > 1` (see `_fit_pathfinder`);
`= 1` (the default since 2026-07-30) is single-path.

Single-path additionally starts from an explicit `_stage1_init(model, z_init_scale, rng, cfg)`: all
latents drawn from their priors except the non-centred `z*` blocks, drawn from `N(0, z_init_scale²)`
(see `_stage1_init`).
Set `z_init_scale = 0` to restore Pathfinder's diffuse `UniformSampler(2)` default. The init is
`rng`-dependent, so distinct seeds still explore distinct starting points.

**NUTS path (2026-08-05).** Pathfinder always runs first — NUTS is initialised from its mean via
`_pf_mean_init` (read that docstring: the previous chain-name construction was a silent no-op), so
the NUTS cost is ADDITIVE on top of the Pathfinder cost, not a replacement for it. The sampler is
built from explicit `cfg.stage1_nuts_*` settings rather than a bare `NUTS()`: the convenience
constructor derives `n_adapts = min(1000, n_sample ÷ 2)`, which at the old `n_sample = 250` gave
**125** warmup iterations to adapt a step size and metric in 293–389/734–977 dimensions (Stan's default is
1000). ONE chain per fit — see `_nuts_diagnostics` for why, and for what that costs in diagnostics.

A NUTS failure **propagates**. It used to be caught and replaced by `pf.draws_transformed`, which
`fit_or_load_stage1` then wrote under the NUTS filename with nothing to distinguish it — a
Pathfinder result wearing a NUTS name, and across a threaded 504-fit prefit the `@warn` is easy to
lose. `prefit_stage1!` has its own `try/catch` that counts the cell as `failed` and leaves no file,
so propagating keeps the cell refittable instead of silently poisoning the cache.
"""
function fit_stage1(dm::ContactDegreeModel, ds, pop, cfg::FrameworkConfig;
                    use_nuts::Bool = cfg.stage1_use_nuts,
                    ndraws_pf::Int = max(200, cfg.n_stage1_post),
                    n_sample::Int = max(cfg.stage1_nuts_draws, cfg.n_stage1_post),
                    nruns::Int = cfg.stage1_pathfinder_runs,
                    z_init_scale::Real = cfg.stage1_z_init_scale,
                    adtype = ad_type(cfg), rng = nothing)
    if rng === nothing
        Random.seed!(cfg.seed)
        rng = Random.default_rng()
    end
    model = model_degree(dm, ds, pop, cfg)
    # Explicit small init for the non-centred z blocks (single-path only — multipathfinder derives
    # its own inits per path, and passing `init` to it makes `nruns` throw; see _fit_pathfinder).
    init = nruns <= 1 ? _stage1_init(model, z_init_scale, rng, cfg) : nothing
    pf = _fit_pathfinder(model, ndraws_pf, nruns, rng, adtype; init = init)
    if !use_nuts
        return (; chn = pf.draws_transformed, model, pf, ad_backend = cfg.ad_backend,
                  sampler = :pathfinder, diag = _nuts_diagnostics(pf.draws_transformed))
    end
    nuts_init = _pf_mean_init(model, pf, rng)
    sampler = NUTS(cfg.stage1_nuts_adapts, cfg.stage1_nuts_target_accept;
                   max_depth = cfg.stage1_nuts_max_depth,
                   adtype = adtype === nothing ? Turing.DEFAULT_ADTYPE : adtype)
    # `n_sample` is the number of KEPT draws; `cfg.stage1_nuts_adapts` warmup iterations are drawn
    # and discarded ON TOP of it (AbstractMCMC applies `discard_initial` before collecting N).
    chn = sample(rng, model, sampler, n_sample; initial_params = nuts_init, progress = false)
    diag = _nuts_diagnostics(chn)
    if !ismissing(diag.divergences) && diag.divergences > 0
        @warn "Stage-1 NUTS: divergent transitions" divergences=diag.divergences n=diag.n_draws
    end
    if !ismissing(diag.frac_at_max_depth) && diag.frac_at_max_depth > 0.2
        @warn "Stage-1 NUTS: tree saturating at max_depth" frac=diag.frac_at_max_depth depth=diag.max_tree_depth
    end
    return (; chn, model, pf, sampler = :nuts, diag, ad_backend = cfg.ad_backend)
end

"""
    fit_or_load_stage1(path, dm, ds, pop, cfg; adtype, rng) -> (; chn)

Reload the Stage-1 chain at `path` if present, else fit (`fit_stage1`) and save
(`_atomic_jldsave(path; result=chn, sampler, diag, ad_backend, target_accept, nuts_adapts,
nuts_draws, phi_init_scale)` — temp file then rename, so a killed process cannot leave a truncated
artefact that the `isfile` skip counts as done). Idempotent skip ⇒ resumable prefit.

`sampler` (`:pathfinder`/`:nuts`), `diag` (`_nuts_diagnostics`), `ad_backend` (`cfg.ad_backend`),
`target_accept` (`cfg.stage1_nuts_target_accept`, raised 0.9→0.95 on 2026-08-06), the NUTS
iteration counts `nuts_adapts`/`nuts_draws` (`500→2000` on 2026-08-07) and `phi_init_scale`
(`cfg.stage1_phi_init_scale`, added 2026-08-10) are written alongside `result` so an artefact is
self-describing. The cache filename encodes the SAMPLER via `contacts_label`, but deliberately NOT
the AD backend, the NUTS tuning or the initialisation, so for those the file's own contents are the
*only* record — which is what makes a mixed-provenance grid auditable:

```julia
using JLD2, Glob, StatsBase
countmap([jldopen(p) do f; haskey(f, "ad_backend") ? f["ad_backend"] : :legacy; end
          for p in glob("8j_s1_*", "../dt_intermediate")])
```

Adding keys is backward compatible: every existing reader asks for `load(path, "result")` by name,
and files that predate a key simply do not have it — `haskey` first, as the snippet above does.

⚠ `phi_init_scale` is the one to read most carefully, because it is the only key here that changes
the answer SYSTEMATICALLY rather than chaotically. A different `ad_backend` perturbs the optimiser
path and gives a different draw from the same posterior; a different φ init can land the fit in a
different MODE (measured: 1.000000 vs 0.820 on weighted-hweibull @ 2021-05-02 h1). Chains that
differ only in this key are not interchangeable, and nothing outside the file says so.
"""
function fit_or_load_stage1(path::AbstractString, dm::ContactDegreeModel, ds, pop,
                            cfg::FrameworkConfig; adtype = ad_type(cfg), rng = nothing)
    isfile(path) && return (; chn = load(path, "result"))
    res = fit_stage1(dm, ds, pop, cfg; use_nuts = cfg.stage1_use_nuts, adtype = adtype, rng = rng)
    # `target_accept`, `nuts_adapts` and `nuts_draws` are recorded for the same reason as
    # `ad_backend`: `contacts_label` encodes only the SAMPLER (`-nuts`), not its tuning, so a grid
    # refitted in part after a tuning change is silently mixed-provenance. The file's own contents
    # are the only record. Audit exactly as for the backend, swapping the key.
    #   `nuts_draws` was added 2026-08-07 with the 500→2000 raise: the draw count is otherwise
    # recoverable only as `size(chn, 1)`, which means loading the whole (now ~28/72 MB) chain just to
    # ask how it was fitted. Under the Pathfinder path these three describe settings that were not
    # consulted — kept anyway so every artefact has the same key set and the audit needs no branch.
    #   `phi_init_scale` was added 2026-08-10 after this gap cost a near-miss. `stage1_phi_init_scale`
    # is deliberately NOT in the cache token (it is a starting value, not a model property), and it
    # changes NOTHING about the chain's parameter names or shapes — yet it moved the motivating cell
    # from φ = 1.000000 to 0.820. So a grid refitted in part across that change is mixed in a way
    # NOTHING visible records: same token, same dimensions, same key set. That is strictly worse than
    # the `ad_backend` case, where at least the draws differ chaotically rather than systematically.
    # Applies on BOTH paths — Pathfinder consumes it directly, NUTS inherits it through `_pf_mean_init`.
    _atomic_jldsave(path; result = res.chn, sampler = res.sampler, diag = res.diag,
                          ad_backend = res.ad_backend,
                          target_accept = cfg.stage1_nuts_target_accept,
                          nuts_adapts = cfg.stage1_nuts_adapts,
                          nuts_draws  = cfg.stage1_nuts_draws,
                          phi_init_scale = cfg.stage1_phi_init_scale)
    return (; chn = res.chn)
end

"""
    stage1_moment_draws(dm, ds, pop, cfg, s1_chn; n_post=cfg.n_stage1_post) -> Vector

Subsample `n_post` Stage-1 posterior draws and return their per-week raw moments via
`generated_quantities(model_degree(dm, ds, pop, cfg), s1_chn)`. Each element is a
`(; K1, K2, G)` of length-`Tn` `Vector{Matrix}` — NGM-independent, ready for
`contact_star(nb, …)` downstream. Draws are subsampled on an even grid (deterministic).
"""
function stage1_moment_draws(dm::ContactDegreeModel, ds, pop, cfg::FrameworkConfig, s1_chn;
                             n_post::Int = cfg.n_stage1_post)
    model = model_degree(dm, ds, pop, cfg)
    gq = vec(generated_quantities(model, s1_chn))
    gq = [q for q in gq if q !== nothing]
    isempty(gq) && error("stage1_moment_draws: no usable Stage-1 draws")
    keep = min(n_post, length(gq))
    idx = round.(Int, range(1, length(gq); length = keep))
    return [gq[k] for k in idx]
end

"""
    stage2_inputs(dm, apd_h, win0, wd0, cfg, s1_path; adtype, rng) -> (; md, n_draw, c0)

The contact moments Stage 2 conditions on, plus the per-fit draw count — the ONE place the null
path forks from the fitted path.

- `needs_stage1(dm)` (the fitted degree models): load-or-fit the Stage-1 chain at `s1_path` and take
  `cfg.n_stage1_post` moment draws, keeping `cfg.n_stage2_draws` samples from each Stage-2 fit.
- **NULL model**: no Stage 1, no `build_degree_stats`, no chain file — a single constant-C* draw
  from `null_moment_draws`, with the FULL `n_stage1_post × n_stage2_draws` samples taken from that
  one fit so the pooled predictive still has 10 000 draws like every other model.

`c0` is the null contact level (`NaN` on the fitted path), returned for logging/diagnostics.
"""
function stage2_inputs(dm::ContactDegreeModel, apd_h::AgePairData, win0::WeeklyWindow,
                       wd0::WindowData, cfg::FrameworkConfig, s1_path::AbstractString;
                       adtype = ad_type(cfg), rng = nothing)
    # ---- THE ONE PLACE THE WINDOW'S DATES ARE CHECKED (`-w8h`, 2026-08-09) ----
    # The degree window must be `degree_window(win0.origin, h, cfg)` = `[t₀−n_fit+1 … t₀+h]`, so its
    # first `n_fit` weeks ARE the origin's fit weeks. The predecessor convention
    # (`WeeklyWindow(origin + Day(7h))`, sliding) starts at `t₀−n_fit+1+h` instead, and at h=4 it
    # even has the right LENGTH — so a call site missed in the migration would produce a complete,
    # plausible, wrongly-dated forecast that no length check downstream could catch. Dates are only
    # in scope here, where both `apd_h` and `win0` are available; assert rather than trust.
    @assert length(apd_h.weeks) >= cfg.n_fit &&
            apd_h.weeks[1:cfg.n_fit] == win0.fit_weeks "stage2_inputs: degree window \
$(first(apd_h.weeks))…$(last(apd_h.weeks)) does not start at win0.fit_weeks \
$(first(win0.fit_weeks))…$(last(win0.fit_weeks)) — build it with `degree_window(origin, h, cfg)`"
    # `n_fit + h` CONTACT weeks, taken from the window itself: since `-w8h` the count varies with the
    # horizon, and the null path must hand Stage 2 the same number of C* matrices the fitted path
    # does or `model_transmission`'s length assertion fires.
    Tc = length(apd_h.weeks)
    if !needs_stage1(dm)
        c0 = null_contact_level(apd_h, win0)
        return (; md = null_moment_draws(c0, wd0.A, Tc),
                  n_draw = cfg.n_stage1_post * cfg.n_stage2_draws, c0 = c0)
    end
    ds = build_degree_stats(dm, apd_h, cfg)
    s1 = fit_or_load_stage1(s1_path, dm, ds, wd0.pop, cfg; adtype = adtype, rng = rng)
    md = stage1_moment_draws(dm, ds, wd0.pop, cfg, s1.chn; n_post = cfg.n_stage1_post)
    return (; md = md, n_draw = cfg.n_stage2_draws, c0 = NaN)
end

"""
    fit_stage2_pooled(nb, moment_draws, wd, cfg; n_draw=cfg.n_stage2_draws, base_seed, max_concurrent)

Cut-inference Stage 2: for EACH of the `M = length(moment_draws)` imputed Stage-1 draws, form
the fixed per-week `C*` (via `contact_star(nb, …)`), Pathfinder-fit `model_transmission`, and keep
`n_draw` infection draws. Pool the `M × n_draw` (= 100×100 = 10_000) draws.

⚠ `adtype` defaults to `stage2_ad_type(cfg)` (`:reversediff`), NOT `ad_type(cfg)` (`:mooncake`).
Stage 2 is 100 independent fits of an 18-dimension model, where per-fit setup dominates and
Mooncake costs 10.64 s vs ReverseDiff's 0.30 s per fit while parallelising 1.09× against 2.31× —
16.5 min vs 0.2 min per pooled cell. See `stage2_ad_backend`.

Returns a NamedTuple
`(; gamma_sar, susc, inf, F, sigma_inf, post_index, Cstar_end, n_post, n_draw)` where the first
five are the pooled per-draw infection parameters (`susc`/`inf` are `N×A`), `post_index[d]` is the
Stage-1 draw `d` came from, and `Cstar_end[m]` is Stage-1 draw `m`'s origin-week (`[end]`) `C*` — the
matrix the forecast NGM is built from. The `M` per-draw fits run under `Semaphore(max_concurrent)`
(each with its own deterministic RNG `base_seed + m`), writing disjoint preallocated slots.
"""
function fit_stage2_pooled(nb::NGMBuilder, moment_draws, wd::WindowData, cfg::FrameworkConfig;
                           n_draw::Int = cfg.n_stage2_draws, adtype = stage2_ad_type(cfg),
                           base_seed::Int = cfg.seed, max_concurrent::Int = 1)
    A = wd.A
    M = length(moment_draws)
    Cstar_end = Vector{Matrix{Float64}}(undef, M)
    per_m = Vector{Any}(undef, M)
    fit_m(m) = begin
        md = moment_draws[m]
        # Length comes from the STAGE-1 draw (`cfg.n_fit` contact weeks since `-w8`), NOT from
        # `length(wd.weeks)` (= n_fit + smax, the infection window). `model_transmission` asserts
        # the relation between the two.
        Tc = length(md.K1)
        Cstar_m = [Float64.(contact_star(nb, md.K1[t], md.K2[t], md.G[t])) for t in 1:Tc]
        Cstar_end[m] = Cstar_m[end]                    # last contact week = origin + h
        # GI is sampled INSIDE the model (no `w` arg); `nb` only selects `fix_infectivity`
        model = model_transmission(Cstar_m, wd, cfg, nb)
        rng = Random.Xoshiro(base_seed + m)
        pf = adtype === nothing ? pathfinder(model; ndraws = n_draw, rng = rng) :
                                  pathfinder(model; ndraws = n_draw, rng = rng, adtype = adtype)
        gq = vec(generated_quantities(model, pf.draws_transformed))
        nd = min(n_draw, length(gq))
        gs = Vector{Float64}(undef, nd); Fv = Vector{Float64}(undef, nd); sg = Vector{Float64}(undef, nd)
        wm = Vector{Float64}(undef, nd); wv = Vector{Float64}(undef, nd)
        su = Matrix{Float64}(undef, nd, A); infm = Matrix{Float64}(undef, nd, A)
        for d in 1:nd
            q = gq[d]
            gs[d] = q.gamma_sar; Fv[d] = q.F; sg[d] = q.sigma_inf
            wm[d] = q.w_mu; wv[d] = q.w_sigma          # per-draw GI log-parameters
            su[d, :] = q.susc; infm[d, :] = q.inf
        end
        per_m[m] = (; gamma_sar = gs, susc = su, inf = infm, F = Fv, sigma_inf = sg,
                      w_mu = wm, w_sigma = wv)
    end

    if max_concurrent <= 1 || M <= 1
        for m in 1:M; fit_m(m); end
    else
        K = clamp(max_concurrent, 1, Threads.nthreads())
        old_blas = LinearAlgebra.BLAS.get_num_threads()
        LinearAlgebra.BLAS.set_num_threads(1)
        try
            fit_m(1)                                       # warm compile before fan-out
            sem = Base.Semaphore(K)
            @sync for m in 2:M
                Threads.@spawn begin
                    Base.acquire(sem)
                    try fit_m(m) finally Base.release(sem) end
                end
            end
        finally
            LinearAlgebra.BLAS.set_num_threads(old_blas)
        end
    end

    gamma_sar  = reduce(vcat, (per_m[m].gamma_sar  for m in 1:M))
    F          = reduce(vcat, (per_m[m].F          for m in 1:M))
    sigma_inf  = reduce(vcat, (per_m[m].sigma_inf  for m in 1:M))
    susc       = reduce(vcat, (per_m[m].susc       for m in 1:M))
    inf        = reduce(vcat, (per_m[m].inf        for m in 1:M))
    w_mu       = reduce(vcat, (per_m[m].w_mu       for m in 1:M))
    w_sigma    = reduce(vcat, (per_m[m].w_sigma    for m in 1:M))
    post_index = reduce(vcat, (fill(m, length(per_m[m].gamma_sar)) for m in 1:M))
    return (; gamma_sar, susc, inf, F, sigma_inf, w_mu, w_sigma, post_index, Cstar_end,
              n_post = M, n_draw = n_draw)
end

# ---- parallel pre-fitting of the (origin × combo × horizon) two-stage artefacts -------
# Fits are mutually independent (each is one Pathfinder run on its own data), so we fan
# them out over Julia threads with a concurrency cap chosen to balance CPU and memory.
# Threading (not Distributed) keeps memory low — one process, shared compiled code — which
# matters here: each worker process would otherwise re-load/compile the whole Turing stack.

"""
Reclaimable RAM (GiB) on macOS, parsed from `vm_stat`: free + inactive + speculative pages.

`Sys.free_memory()` is NOT usable on darwin — it is libuv's `uv_get_free_memory()`, i.e. the
*truly free* page count only. macOS deliberately keeps that near zero (it holds memory as
inactive/cached rather than freeing it), so it reads ~2 GiB on an idle 32 GiB machine and
`fit_concurrency`'s memory cap floors to 0 ⇒ silently serial fitting. Free + inactive +
speculative is the darwin analogue of Linux's `MemAvailable` (inactive and speculative pages
are reclaimable, the latter being file read-ahead). `purgeable` is deliberately NOT added: it
is a subset of active/inactive, so counting it would double-count.
"""
function _darwin_available_gib()
    out = read(`vm_stat`, String)
    m = match(r"page size of (\d+) bytes", out)
    pagesize = m === nothing ? 4096 : parse(Int, m.captures[1])
    npages(label) = begin
        mm = match(Regex("^Pages " * label * ":\\s+(\\d+)\\.", "m"), out)
        mm === nothing ? 0 : parse(Int, mm.captures[1])
    end
    return (npages("free") + npages("inactive") + npages("speculative")) * pagesize / 2^30
end

"""
Memory headroom (GiB) inside a Slurm allocation, or `nothing` when not under Slurm / not
determinable.

`/proc/meminfo` is the WRONG source on a compute node: it reports the whole machine (LSHTM's new
nodes are 96 cores / 768 GB), not the cgroup the job is confined to. Under that reading
`fit_concurrency`'s memory cap never binds and `8j_run_grid.jl`'s `MEM_FLOOR_GIB` early-exit — the
guard that exists *because* the 63-origin run was OOM-killed twice — can never fire. The job is then
killed by the cgroup OOM killer with SIGKILL, which leaves nothing in the log at all.

Limit, in order of preference: cgroup v2 `memory.max` (resolved through `/proc/self/cgroup`, which
is what makes this work inside a Singularity container sharing the host's cgroup hierarchy) →
cgroup v1 `memory.limit_in_bytes` → Slurm's own `SLURM_MEM_PER_NODE` / `SLURM_MEM_PER_CPU ×
SLURM_CPUS_PER_TASK` (MB). Usage is **current** `VmRSS` from `/proc/self/status`, never
`Sys.maxrss()` — that is a high-water mark and monotone by construction, so it cannot say how much
has since been freed.

A cgroup with no limit reads `"max"` (v2) or a sentinel near `typemax` (v1); both are rejected so
the caller falls through to the ordinary reading.
"""
function _slurm_mem_available_gib()
    haskey(ENV, "SLURM_JOB_ID") || return nothing
    limit_bytes = nothing
    try
        # cgroup v2: /proc/self/cgroup is "0::<path>" and the controller file lives at
        # /sys/fs/cgroup<path>/memory.max. Walk up: the limit may be set on an ancestor.
        for line in eachline("/proc/self/cgroup")
            parts = split(line, ':'; limit = 3)
            length(parts) == 3 && parts[1] == "0" || continue
            dir = joinpath("/sys/fs/cgroup", lstrip(parts[3], '/'))
            while true
                f = joinpath(dir, "memory.max")
                if isfile(f)
                    v = strip(read(f, String))
                    if v != "max"                  # "max" = no limit here; an ancestor may set one
                        limit_bytes = parse(Int, v)
                        break
                    end
                end
                d = dirname(dir)
                (d == dir || !startswith(d, "/sys/fs/cgroup")) && break
                dir = d
            end
            break                                  # v2 has exactly one "0::" line
        end
    catch
    end
    if limit_bytes === nothing                                    # cgroup v1
        try
            f = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
            if isfile(f)
                v = parse(Int, strip(read(f, String)))
                v < (Int(1) << 60) && (limit_bytes = v)           # unlimited reads ~typemax
            end
        catch
        end
    end
    if limit_bytes === nothing                                    # Slurm's own accounting, MB
        try
            mb = if haskey(ENV, "SLURM_MEM_PER_NODE")
                parse(Float64, ENV["SLURM_MEM_PER_NODE"])
            elseif haskey(ENV, "SLURM_MEM_PER_CPU")
                parse(Float64, ENV["SLURM_MEM_PER_CPU"]) *
                    parse(Float64, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
            else
                0.0
            end
            mb > 0 && (limit_bytes = round(Int, mb * 2^20))
        catch
        end
    end
    limit_bytes === nothing && return nothing
    rss_bytes = 0
    try
        for line in eachline("/proc/self/status")
            startswith(line, "VmRSS:") && (rss_bytes = parse(Int, split(line)[2]) * 1024; break)
        end
    catch
        return nothing
    end
    return max(0.0, (limit_bytes - rss_bytes) / 2^30)
end

"""
Available RAM (GiB), per platform: **inside a Slurm allocation** the cgroup headroom
(`_slurm_mem_available_gib` — `/proc/meminfo` would report the whole compute node); Linux
`/proc/meminfo` `MemAvailable` (counts reclaimable cache); macOS `vm_stat`
free+inactive+speculative (see `_darwin_available_gib` — the old `/proc/meminfo`-then-
`Sys.free_memory` fallback under-reported by ~7× on darwin and forced `fit_concurrency` to 1);
anything else `Sys.free_memory`.
"""
function _mem_available_gib()
    let slurm = _slurm_mem_available_gib()      # `nothing` unless SLURM_JOB_ID is set
        slurm === nothing || return slurm
    end
    if Sys.islinux()
        try
            for line in eachline("/proc/meminfo")
                startswith(line, "MemAvailable:") && return parse(Int, split(line)[2]) / 2^20
            end
        catch
        end
    elseif Sys.isapple()
        try
            return _darwin_available_gib()
        catch
        end
    end
    return Sys.free_memory() / 2^30
end

"""
    fit_concurrency(; mem_per_fit_gib=1.0, reserve_gib=4.0)

How many joint fits to run at once, balancing CPU and memory: the minimum of the Julia
thread count, (physical cores − 1), and how many `mem_per_fit_gib`-sized fits fit in
available RAM after a `reserve_gib` headroom. Always ≥ 1.

Note the CPU cap is bounded by `Threads.nthreads()`, which is 1 unless Julia is started with
`-t`/`JULIA_NUM_THREADS` (the devcontainer sets 12) — so a return of 1 on a many-core machine
usually means the thread count, not the memory cap. `fit_concurrency_report()` says which.

Second darwin caveat (documented, NOT worked around): on Apple silicon `Sys.CPU_THREADS` reports
the *performance* cores only (`hw.perflevel0.logicalcpu`, e.g. 4) while `hw.ncpu` counts P+E
(e.g. 10), so `cpu_cap` saturates well below the apparent core count. That is a defensible cap
for compute-bound Pathfinder fits — E-cores contribute little and oversubscribing hurts — so it
is left as is; pass `max_concurrent` explicitly to override.
"""
function fit_concurrency(; mem_per_fit_gib::Real = 1.0, reserve_gib::Real = 4.0)
    mem_cap = floor(Int, max(0.0, _mem_available_gib() - reserve_gib) / mem_per_fit_gib)
    cpu_cap = min(Threads.nthreads(), max(1, Sys.CPU_THREADS - 1))
    return max(1, min(cpu_cap, mem_cap))
end

"""
    fit_concurrency_report(; mem_per_fit_gib=1.0, reserve_gib=4.0)

Diagnostic breakdown of `fit_concurrency`: which cap binds, and the memory reading behind it.
Returns `(; concurrency, mem_cap, cpu_cap, avail_gib, nthreads, cpu_threads, binding)`.
Print this before a long pre-fit — the value decides whether the run is serial or `n`-way, and
nothing in the notebooks records it.
"""
function fit_concurrency_report(; mem_per_fit_gib::Real = 1.0, reserve_gib::Real = 4.0)
    avail   = _mem_available_gib()
    mem_cap = floor(Int, max(0.0, avail - reserve_gib) / mem_per_fit_gib)
    cpu_cap = min(Threads.nthreads(), max(1, Sys.CPU_THREADS - 1))
    k       = max(1, min(cpu_cap, mem_cap))
    binding = mem_cap <= cpu_cap ? :memory : :cpu
    return (; concurrency = k, mem_cap, cpu_cap, avail_gib = avail,
              nthreads = Threads.nthreads(), cpu_threads = Sys.CPU_THREADS, binding)
end

"""
    prefit_stage1!(dms, wins, cfg; data_provider, save_dir, max_concurrent, adtype)

Fit every MISSING Stage-1 chain (`8j_s1_<degree>_<contacts>_<origin>_h<h>.jld2`) for the distinct
degree models `dms` × origins `wins` × horizons. Degree models with `needs_stage1(dm) == false`
(the NULL model) are dropped up front — they have no contact likelihood and never produce an
`8j_s1_*` file. Origins are processed sequentially (bounded
memory: one origin's data held at a time via `data_provider(oi, win) → (wd0, apd_by_h)`); within an
origin the (degree × horizon) fits fan out under `Semaphore(max_concurrent)`. Each fit gets its own
RNG and writes a distinct file, and BLAS is pinned to one thread during the parallel region. Cached
chains are skipped ⇒ resumable. Returns `(; fitted, failed, skipped, concurrency)`.
"""
function prefit_stage1!(dms, wins, cfg::FrameworkConfig; data_provider,
                        save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                        max_concurrent::Int = fit_concurrency(), adtype = ad_type(cfg))
    mkpath(save_dir)
    dms = filter(needs_stage1, collect(dms))   # the NULL degree model has no Stage 1 to fit
    isempty(dms) && return (; fitted = 0, failed = 0, skipped = 0, concurrency = 1)
    K   = clamp(max_concurrent, 1, Threads.nthreads())
    tag = contacts_label(cfg)
    fitted  = Threads.Atomic{Int}(0)
    failed  = Threads.Atomic{Int}(0)
    skipped = Threads.Atomic{Int}(0)
    warmed  = Set{DataType}()   # degree-model TYPES whose AD rule has already been built
    old_blas = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)
    try
        for (oi, win_o) in enumerate(wins)
            specs = NamedTuple[]
            for dm in dms, (hi, h) in enumerate(cfg.horizons)
                path = stage1_path(dm, win_o.origin, h; contacts = tag, save_dir = save_dir)
                isfile(path) ? Threads.atomic_add!(skipped, 1) : push!(specs, (; dm, hi, h, path))
            end
            isempty(specs) && continue
            wd0, apd_by_h = data_provider(oi, win_o)
            do_fit(s) = begin
                try
                    ds_h = build_degree_stats(s.dm, apd_by_h[s.hi], cfg)
                    fit_or_load_stage1(s.path, s.dm, ds_h, wd0.pop, cfg;
                                       adtype = adtype, rng = Random.Xoshiro(cfg.seed))
                    Threads.atomic_add!(fitted, 1)
                catch err
                    Threads.atomic_add!(failed, 1)
                    @warn "stage1 fit failed" origin=win_o.origin degree=degree_label(s.dm) h=s.h exception=(err, catch_backtrace())
                end
            end
            # Warm ONE fit per as-yet-unseen degree-model TYPE, SERIALLY, before the fan-out.
            # Each `typeof(dm)` is a distinct `model_degree` signature and so a distinct AD-rule
            # derivation — under Mooncake that is `build_rrule`, measured at 66 s (negbin) / 14 s
            # (hurdle-Weibull). This used to be a single `Ref{Bool}` set on `specs[1]`, and since
            # `specs` is built `for dm in dms, h in horizons` that is ALWAYS `dms[1]`: the second
            # degree model's rule was therefore derived INSIDE the `@spawn` region, where Mooncake
            # serialises concurrent derivations on a global lock and every other worker blocks —
            # while still holding its semaphore slot. Correct, but it stalls the fan-out and looks
            # like a hang, because nothing prints until the origin completes.
            # `warmed` is hoisted outside the origin loop on purpose: the rule cache is per-process
            # and the model TYPE does not vary with origin (only the values inside `ds` do).
            warm_idx = Int[]
            for (k, s) in enumerate(specs)
                typeof(s.dm) in warmed && continue
                push!(warmed, typeof(s.dm)); push!(warm_idx, k)
            end
            for k in warm_idx
                dt = @elapsed do_fit(specs[k])
                @info "prefit_stage1!: warm fit (AD rule built)" degree=degree_label(specs[k].dm) backend=cfg.ad_backend seconds=round(dt; digits = 1)
            end
            rest = @view specs[setdiff(1:length(specs), warm_idx)]
            sem = Base.Semaphore(K)
            t_origin = time()
            @sync for s in rest
                Threads.@spawn begin
                    Base.acquire(sem)
                    try do_fit(s) finally Base.release(sem) end
                end
            end
            # Heartbeat, mirroring `prefit_stage2!`'s per-origin line. Without it this function
            # prints NOTHING between the two warm-fit lines (first origin only) and the final
            # summary — over a 63-origin Stage-1 NUTS grid that is days of silence, and the artefact
            # count on disk becomes the only way to tell a running fit from a hung one.
            # ⚠ FULL COLLECTION PER ORIGIN, and it is not defensive clutter. Measured 2026-08-07 on
            # the 63-origin Pathfinder grid: RSS climbed monotonically 10.3 -> 13.6 -> 17.8 -> 20.7
            # -> 21.8 -> 23.7 -> 25.0 GiB over ~12 origins against 27.4 GiB of RAM, per-origin wall
            # time went 224 s -> 1134 s as the GC thrashed, and the process was OOM-killed at
            # origin 16 with no error, no stack trace and nothing in its log. The per-origin working
            # set (`wd0`, `apd_by_h`, and the `do_fit` closures over them) IS garbage once the
            # `@sync` returns, but Julia's heuristic will not run a full collection while the heap
            # looks healthy, so it accumulates. `GC.gc()` costs ~a second against a 4-minute origin.
            GC.gc()
            @info "prefit_stage1!: origin $(win_o.origin) done ($(oi)/$(length(wins)), $(length(specs)) fits, $(round(time() - t_origin; digits = 1)) s, $(fitted[]) fitted / $(failed[]) failed so far, RSS $(round(Sys.maxrss() / 2^30; digits = 1)) GiB)"
        end
    finally
        LinearAlgebra.BLAS.set_num_threads(old_blas)
    end
    @info "prefit_stage1!: $(fitted[]) fitted, $(skipped[]) skipped, $(failed[]) failed (concurrency=$K)"
    return (; fitted = fitted[], failed = failed[], skipped = skipped[], concurrency = K)
end

"""
    prefit_stage2!(combos, wins, cfg; data_provider, save_dir, max_concurrent, adtype)

Fit every MISSING Stage-2 pooled result (`8j_s2_<degree>_<ngm>_<contacts>_<origin>_h<h>.jld2`) for
combos × origins × horizons. Requires the matching Stage-1 chain (run `prefit_stage1!` first; a
missing one is fit on demand, under `adtype` — the STAGE-1 backend, distinct from the `s2_adtype`
the transmission fits use) — except for the NULL model, which has no Stage 1 and gets its
constant C* from `stage2_inputs`. Origins are processed sequentially; within an origin the (combo ×
horizon) cells run sequentially, each `fit_stage2_pooled` fanning out its `n_stage1_post` per-draw
fits under `Semaphore(max_concurrent)`. Cached pooled files are skipped ⇒ resumable.
"""
function prefit_stage2!(combos, wins, cfg::FrameworkConfig; data_provider,
                        save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                        max_concurrent::Int = fit_concurrency(), adtype = ad_type(cfg),
                        s2_adtype = stage2_ad_type(cfg))
    # `adtype` is the STAGE-1 backend: `stage2_inputs` may have to fit a missing Stage-1 chain, and
    # that leg must match how the rest of the Stage-1 grid was fitted. `s2_adtype` is the one the
    # 100 per-draw transmission fits actually use, and it is a DIFFERENT backend by default — see
    # `stage2_ad_backend`. Conflating them is what made one pooled cell take 16.5 minutes.
    mkpath(save_dir)
    K   = clamp(max_concurrent, 1, Threads.nthreads())
    tag = contacts_label(cfg)
    fitted = 0; failed = 0; skipped = 0
    for (oi, win_o) in enumerate(wins)
        todo = NamedTuple[]
        for (dm, nb) in combos, (hi, h) in enumerate(cfg.horizons)
            path = stage2_path(dm, nb, win_o.origin, h; contacts = tag, save_dir = save_dir)
            isfile(path) ? (skipped += 1) : push!(todo, (; dm, nb, hi, h, path))
        end
        isempty(todo) && continue
        wd0, apd_by_h = data_provider(oi, win_o)
        for s in todo
            try
                s1p = stage1_path(s.dm, win_o.origin, s.h; contacts = tag, save_dir = save_dir)
                inp = stage2_inputs(s.dm, apd_by_h[s.hi], win_o, wd0, cfg, s1p;
                                    adtype = adtype, rng = Random.Xoshiro(cfg.seed))
                pooled = fit_stage2_pooled(s.nb, inp.md, wd0, cfg; n_draw = inp.n_draw,
                                           adtype = s2_adtype,
                                           base_seed = cfg.seed + 1000 * s.h, max_concurrent = K)
                # Record the Stage-2 backend for the same reason Stage 1 records its own: it is
                # deliberately NOT in the filename, so the file is the only evidence of how the
                # draws were produced. Pathfinder's LBFGS path is chaotic, so a grid half-fitted
                # under each backend is not bit-comparable even though the gradients agree.
                _atomic_jldsave(s.path; pooled, ad_backend = cfg.stage2_ad_backend)
                fitted += 1
            catch err
                failed += 1
                @warn "stage2 fit failed" origin=win_o.origin degree=degree_label(s.dm) ngm=ngm_label(s.nb) h=s.h exception=(err, catch_backtrace())
            end
        end
        GC.gc()   # same reason as prefit_stage1! — see the note there
        @info "prefit_stage2!: origin $(win_o.origin) done ($fitted fitted, $failed failed, RSS $(round(Sys.maxrss() / 2^30; digits = 1)) GiB)"
    end
    return (; fitted, failed, skipped, concurrency = K)
end

"""
    prefit_two_stage!(combos, wins, cfg; data_provider, save_dir, max_concurrent, adtype)

Convenience driver: `prefit_stage1!` (for the distinct degree models in `combos`) then
`prefit_stage2!`. Returns `(; stage1, stage2)`.
"""
function prefit_two_stage!(combos, wins, cfg::FrameworkConfig; data_provider,
                           save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                           max_concurrent::Int = fit_concurrency(), adtype = ad_type(cfg))
    dms = unique(first.(combos))
    s1 = prefit_stage1!(dms, wins, cfg; data_provider = data_provider, save_dir = save_dir,
                        max_concurrent = max_concurrent, adtype = adtype)
    s2 = prefit_stage2!(combos, wins, cfg; data_provider = data_provider, save_dir = save_dir,
                        max_concurrent = max_concurrent, adtype = adtype)
    return (; stage1 = s1, stage2 = s2)
end

"""
    fit_or_load_stage2(dm, nb, wd0, cfg, win0, h; apd_h=nothing, grid, setting, save_dir, adtype)

Return the Stage-2 pooled NamedTuple for `(dm, nb, origin, h)`, reloading the cached `8j_s2_*` file
if present, else building it end-to-end: Stage-1 (load/fit on the horizon-`h` contact window ending
at `origin+h`) → `n_stage1_post` moment draws → `fit_stage2_pooled` → save. `apd_h` is the pre-built
`AgePairData` for that horizon window (built on demand if `nothing`). For the NULL model the
Stage-1 leg is replaced by a single constant-C* draw (see `stage2_inputs`).
"""
function fit_or_load_stage2(dm, nb, wd0::WindowData, cfg::FrameworkConfig, win0::WeeklyWindow,
                            h::Integer; apd_h = nothing, grid = cis_age_grid(),
                            setting::Symbol = :all,
                            save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                            adtype = ad_type(cfg), s2_adtype = stage2_ad_type(cfg))
    tag = contacts_label(cfg)
    s2p = stage2_path(dm, nb, win0.origin, h; contacts = tag, save_dir = save_dir)
    isfile(s2p) && return load(s2p, "pooled")
    if apd_h === nothing
        apd_h = prepare_degree_data(degree_window(win0.origin, h, cfg), cfg;
                                    grid = grid, setting = setting)
    end
    s1p = stage1_path(dm, win0.origin, h; contacts = tag, save_dir = save_dir)
    inp = stage2_inputs(dm, apd_h, win0, wd0, cfg, s1p; adtype = adtype, rng = Random.Xoshiro(cfg.seed))
    pooled = fit_stage2_pooled(nb, inp.md, wd0, cfg; n_draw = inp.n_draw, adtype = s2_adtype,
                               base_seed = cfg.seed + 1000 * h)
    _atomic_jldsave(s2p; pooled, ad_backend = cfg.stage2_ad_backend)
    return pooled
end

"""
    two_stage_forecast(dm, nb, wd0, cfg, win0; apd_by_h=nothing, grid, setting, save_dir, adtype)

Pooled `A × H × N` forecast (N = n_stage1_post·n_stage2_draws = 10_000). **Infections** are frozen at
the baseline `win0.origin` (t₀); for each horizon `h` the Stage-2 pooled draws for
`(dm, nb, origin, h)` are reloaded (or built), and per pooled draw `d` (from Stage-1 draw
`m = post_index[d]`) the NGM
`N = build_ngm(Cstar_end[m], susc[d], inf[d], F[d], wd0.antibody_fc[:,hi]; gamma_sar=gamma_sar[d])`
takes one renewal step against the history (observed lags up to t₀ plus the intervening horizons'
MEAN forecasts — per-draw coherence across horizons is undefined, mirroring the former
`iterated_forecast`). `apd_by_h[hi]` (optional) is the pre-built horizon-window `AgePairData`.

Two things changed 2026-07-30 (`inst/5_formal_pathfinder_impl.md`):
- **Antibody comes from the TARGET week t₀+h** (`antibody_fc[:, hi]`), not t₀ — matching the contact
  window, which already ends at t₀+h. §3.2.
- **The generation interval is per draw**, so the renewal-weighted lag sum `acc` is computed INSIDE
  the draw loop from that draw's `(w_mu, w_sigma)`. It can no longer be hoisted per horizon; hoisting
  it would silently apply one draw's GI to all of them.
"""
function two_stage_forecast(dm, nb, wd0::WindowData, cfg::FrameworkConfig, win0::WeeklyWindow;
                            apd_by_h = nothing, grid = cis_age_grid(), setting::Symbol = :all,
                            save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                            adtype = ad_type(cfg))
    A = wd0.A; H = length(cfg.horizons)
    mkpath(save_dir)
    hist = collect(float.(wd0.I_mean))                 # A × Tn0, last col = origin (t₀)
    cols = Vector{Matrix{Float64}}(undef, H)
    for (hi, h) in enumerate(cfg.horizons)
        apd_h  = apd_by_h === nothing ? nothing : apd_by_h[hi]
        pooled = fit_or_load_stage2(dm, nb, wd0, cfg, win0, h; apd_h = apd_h, grid = grid,
                                    setting = setting, save_dir = save_dir, adtype = adtype)
        Np  = length(pooled.gamma_sar)
        rng = MersenneTwister(cfg.seed + h)
        # The renewal-weighted history is now PER DRAW: the generation interval is estimated
        # (§3.1), so each pooled draw carries its own (w_mu, w_sigma) ⇒ its own w. It can no
        # longer be hoisted out of the loop as one horizon-constant `acc`.
        # Antibody comes from the TARGET week t₀+h (2026-07-30, §3.2), matching the contact
        # window that already ends at t₀+h — not from the origin.
        ab_h = wd0.antibody_fc[:, hi]
        draws_h = Array{Float64}(undef, A, Np)
        preds   = Array{Float64}(undef, A, Np)         # deterministic renewal mean per draw
        for d in 1:Np
            m = pooled.post_index[d]
            w_d = gen_interval_pmf_log(pooled.w_mu[d], pooled.w_sigma[d]; smax = cfg.smax)
            acc = zeros(A)
            for s in 1:cfg.smax
                acc .+= w_d[s] .* hist[:, end - s + 1]
            end
            N = build_ngm(pooled.Cstar_end[m], pooled.susc[d, :], pooled.inf[d, :],
                          pooled.F[d], ab_h; gamma_sar = pooled.gamma_sar[d])
            pred = N * acc
            for a in 1:A
                p = pred[a]
                preds[a, d] = p
                if isfinite(p)
                    σ = max(pooled.sigma_inf[d] * p, 1e-6)
                    draws_h[a, d] = p + σ * randn(rng)
                else
                    draws_h[a, d] = p                  # keep ±Inf; never fabricate NaN (Inf + Inf·randn)
                end
            end
        end
        cols[hi] = draws_h
        # ROBUST lag plug for the next horizon: median over FINITE draws per age. The pooled
        # predictive is heavy-tailed — a few pathological Pathfinder draws give an astronomically
        # supercritical N — and a MEAN plug lets one outlier poison the shared iteration for ALL
        # draws → mass Inf/NaN blow-up. The median ignores those outliers so the iteration stays
        # finite; individual pathological draws still blow up in their own fan column (handled by
        # the finite-robust quantiles downstream).
        step_plug = [begin
                         f = filter(isfinite, @view preds[a, :])
                         isempty(f) ? 0.0 : median(f)
                     end for a in 1:A]
        hist = hcat(hist, step_plug)                   # robust deterministic lag for the next horizon
    end
    K = minimum(size(c, 2) for c in cols)              # align draw count across horizons
    out = Array{Float64}(undef, A, H, K)
    for hi in 1:H, a in 1:A, d in 1:K
        out[a, hi, d] = cols[hi][a, d]
    end
    return out
end
