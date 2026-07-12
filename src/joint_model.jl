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
# just sees a bad (finite) objective and backtracks. The interior is unchanged to <3e-3.
_softplus(z) = z > zero(z) ? z + log1p(exp(-z)) : log1p(exp(z))
_softclamp(x, lo, hi) = lo + _softplus((hi - _softplus(hi - x)) - lo)

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
    common = (; log_emp = log_emp, A = A, weeks = apd.weeks,
                mid = cis_age_midpoints(), pair_list = pair_list, pair_index = pair_index)
    if cfg.constant_contacts
        return (; dd_count = p.dd_count, pos_weight = p.pos_weight, p0 = p.p0,
                  n = p.n, common...)
    else
        return (; dd_count = apd.dd_count, pos_weight = apd.pos_weight, p0 = apd.p0,
                  n = apd.n, common...)                                     # raw [t,i,j] weekly arrays
    end
end

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
    # The kernel is ANISOTROPIC in DIAGONAL coordinates: the age pair (x,y)=(mid_a,mid_b)
    # is rotated 45° into u=(x+y)/√2 (along the main diagonal = total age) and v=(x−y)/√2
    # (across the diagonal = age gap), each with its OWN length-scale — ρ_diag on the
    # total-age direction, ρ_gap on the age-gap direction (assortativity). Because the
    # rotation is orthonormal, (Δu)²+(Δv)² = (Δx)²+(Δy)², so ρ_diag=ρ_gap recovers the old
    # isotropic RBF exactly. `ρ_diag`, `ρ_gap`, `η` and the 28×28 Cholesky `Lp` are SHARED
    # across weeks. In the per-week regime the weekly fields are no longer iid: they are
    # coupled by a SEPARABLE temporal GP (§5) — a matrix-normal field R = η·(Lp·z·Ltᵀ) with a
    # shared temporal Cholesky Lt(ρ_time) and a decoupled temporal level cₜ = c + σ_c·(Lt·z_c).
    # The population offset is taken RELATIVE to the reference bin (index 1, "2-10"): only
    # relative population matters for reciprocity, and a constant shift log(pop₁) cancels in
    # pop_i·μ_{i→j}=pop_j·μ_{j→i}, so exact reciprocity is preserved — but it rescales the
    # latent level c/c0 to O(1) (absolute log(pop)≈15.6 otherwise forces c≈−15.6 and, at the
    # old clamp, the degenerate μ≡403 saturation; see tasks/lessons.md).
    logpop = log.(pop ./ pop[1])
    log_rho_diag ~ Normal(cfg.gp_len_prior[1], cfg.gp_len_prior[2])   # total-age direction
    log_rho_gap  ~ Normal(cfg.gp_len_prior[1], cfg.gp_len_prior[2])   # age-gap direction
    log_eta ~ Normal(cfg.gp_scale_prior[1], cfg.gp_scale_prior[2])
    ρ_diag = exp(_softclamp(log_rho_diag, log(3.0), log(45.0)))   # length-scale (age-yrs), soft-bounded
    ρ_gap  = exp(_softclamp(log_rho_gap,  log(3.0), log(45.0)))   # length-scale (age-yrs), soft-bounded
    η = exp(_softclamp(log_eta, -3.0, 2.0))               # GP marginal scale, soft-bounded
    c0 = mean(ds.log_emp .- logpop')                      # smooth mean-fn anchor (pooled c0)
    mid = ds.mid
    P = length(ds.pair_list)
    # rotated (diagonal / anti-diagonal) coordinates for the 28 pairs, √2-normalised
    su = [(mid[p[1]] + mid[p[2]]) / sqrt(2) for p in ds.pair_list]   # along-diagonal (total age)
    df = [(mid[p[1]] - mid[p[2]]) / sqrt(2) for p in ds.pair_list]   # across-diagonal (age gap)
    # 28×28 anisotropic separable RBF in diagonal coordinates
    Kp = [exp(-((su[m] - su[n])^2 / (2 * ρ_diag^2) + (df[m] - df[n])^2 / (2 * ρ_gap^2)))
          for m in 1:P, n in 1:P]
    # DENSE Cholesky factor (Matrix, not the LowerTriangular `.L`): the per-week structure field
    # forms the matrix product Lp·z·Ltᵀ, and ReverseDiff cannot write a dense cotangent into a
    # triangular-typed factor (`… * Ltᵀ` → "cannot set index in the lower triangular part of an
    # UpperTriangular matrix"). Densifying both factors is the RD-safe form; gradients w.r.t.
    # ρ_diag/ρ_gap still flow through `Matrix(cholesky(...).L)`. (Verified vs triangular variants.)
    Lp = Matrix(cholesky(Symmetric(Kp) + 1e-6 * I).L)

    # per-cell log-rate → directional mean μ_{i→j}. Soft-clamped (not `clamp`, so ReverseDiff-safe)
    # to μ ∈ ≈[3e-4, 400]: the relative-population offset keeps the healthy log-rate O(1) (deep in
    # the interior, where _softclamp is the identity), so μ is undistorted; the soft bound only
    # stops a stray LBFGS step from over/under-flowing μ (which would make the Weibull scale
    # λ=μ/gamma non-finite and abort the fit).
    _mu_matrix(rvec) =
        [exp(_softclamp(rvec[ds.pair_index[i, j]] + logpop[j], -8.0, 6.0)) for i in 1:A, j in 1:A]

    # per-cell moments (⟨k⟩, ⟨k²⟩, zero factor g) + contact log-likelihood for one week's
    # μ matrix. `didx` indexes the (possibly weekly) degree arrays; `dispv` is the length-4
    # block-linear dispersion for this context, indexed `bl = 2(bi−1)+bj ∈ {1,2,3,4}`
    # (kept 1-D per week: DynamicPPL's `generated_quantities` can't reconstruct a 3-D
    # `filldist`, so dispersion is a 2-D `4×Tn` array sliced per week, never `2×2×Tn`).
    function _cell_moments!(K1, K2, G, μ, didx, dispv)
        ll = zero(eltype(μ))
        for i in 1:A, j in 1:A
            bl = 2 * (block_of(i, cfg) - 1) + block_of(j, cfg)
            if is_weighted(dm)
                κ = exp(_softclamp(dispv[bl], -3.0, 3.0))   # shape ∈ ≈[0.05, 20], soft-bounded
                λ = μ[i, j] / gamma(1 + 1 / κ)              # scale stays finite & >0 (μ, κ bounded)
                pos = didx === nothing ? ds.pos_weight[i, j] : ds.pos_weight[didx, i, j]
                isempty(pos) || (ll += calculate_loglikelihood(pos, Weibull(κ, λ)))  # collapsed histogram
                p0 = didx === nothing ? ds.p0[i, j] : ds.p0[didx, i, j]
                k1, k2, g = _weibull_moments(μ[i, j], κ, p0)
            else
                kk = exp(_softclamp(dispv[bl], -4.0, 5.0))  # dispersion ∈ ≈[0.018, 148], soft-bounded
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
        z ~ filldist(Normal(0, 1), P)                     # 28 iid (non-centred GP)
        if is_weighted(dm)
            log_kappa ~ filldist(Normal(0.0, 0.5), 2, 2)  # Weibull shape by child/adult block
            disp = log_kappa
        else
            log_k ~ filldist(Normal(0.0, 1.0), 2, 2)      # NegBin dispersion by block
            disp = log_k
        end
        μ = _mu_matrix(c .+ η .* (Lp * z))
        ETp = eltype(μ)
        K1 = Matrix{ETp}(undef, A, A); K2 = Matrix{ETp}(undef, A, A); G = Matrix{ETp}(undef, A, A)
        Turing.@addlogprob! _cell_moments!(K1, K2, G, μ, nothing, vec(disp))   # 2×2 → block-linear 4
        K1w = [K1 for _ in 1:Tn]; K2w = [K2 for _ in 1:Tn]; Gw = [G for _ in 1:Tn]
        return (; K1 = K1w, K2 = K2w, G = Gw)
    else
        # per-week: SEPARABLE spatio-temporal GP (§5). The age-pair field is smoothed over
        # weeks by a temporal RBF over week indices 1:Tn, sharing one length-scale ρ_time
        # across all age-pairs; the spatial kernel (ρ_diag, ρ_gap, η, Lp) is shared as before.
        #   • temporal kernel  Kt[s,t] = exp(-(s-t)²/(2ρ_time²)),  Lt = chol(Kt + jitter)
        #   • structure field  R = η·(Lp·z·Ltᵀ)   (P×Tn)  ⟹ Cov(vec R) = η²·(Kt ⊗ Kage)
        #     each age-pair a temporally-correlated GP, each week the spatial RBF.
        #   • decoupled level  cₜ = c + σ_c·(Lt·z_c)  — scalar intercept c + a 1-D temporal GP
        #     with its OWN amplitude σ_c (so η governs age-structure only), sharing ρ_time.
        # ρ_diag=ρ_gap recovers the isotropic spatial kernel; ρ_time→0 ⇒ iid weeks, →∞ ⇒ pooled.
        log_rho_time ~ Normal(cfg.gp_time_len_prior[1], cfg.gp_time_len_prior[2])
        ρ_time = exp(_softclamp(log_rho_time, log(0.5), log(26.0)))   # weeks, soft-bounded
        # temporal Cholesky over the Tn window weeks. Jitter 1e-4 (not 1e-6): Kt is near
        # rank-1 at the upper clamp (near-pooled) and the Pathfinder call is not try/caught,
        # so a PosDefException would abort the whole fit (see tasks/lessons.md).
        Kt = [exp(-((s - t)^2) / (2 * ρ_time^2)) for s in 1:Tn, t in 1:Tn]
        Lt = Matrix(cholesky(Symmetric(Kt) + 1e-4 * I).L)   # DENSE (see Lp note above): Ltᵀ must not be a triangular type

        c ~ Normal(c0, 3.0)                               # scalar level intercept (stored)
        log_sigma_c ~ Normal(cfg.gp_level_scale_prior[1], cfg.gp_level_scale_prior[2])
        σ_c = exp(_softclamp(log_sigma_c, -3.0, 2.0))     # temporal-level amplitude, soft-bounded (mirrors η)
        z_c ~ filldist(Normal(0, 1), Tn)                  # temporal-level raw (non-centred)
        c_vec = c .+ σ_c .* (Lt * z_c)                     # per-week level cₜ (temporally smooth)

        z ~ filldist(Normal(0, 1), P, Tn)                 # structure field raw (shared spatial+temporal kernel)
        # dispersion 4×Tn (block-linear rows × week): 2-D so generated_quantities can
        # reconstruct it (a 3-D 2×2×Tn filldist can't be — see _cell_moments!).
        if is_weighted(dm)
            log_kappa ~ filldist(Normal(0.0, 0.5), 4, Tn)      # shape by block-linear × week
            disp = log_kappa
        else
            log_k ~ filldist(Normal(0.0, 1.0), 4, Tn)          # dispersion by block-linear × week
            disp = log_k
        end
        # precompute the whole spatio-temporal field ONCE (the temporal coupling means each
        # week's column depends on ALL columns of z, so it can't be sliced per week). Fld
        # already carries η; don't re-apply it below.
        Fld = η .* (Lp * z * Lt')                          # P×Tn
        ETp = promote_type(typeof(c), eltype(Fld))
        K1w = Vector{Matrix{ETp}}(undef, Tn)               # per-week raw moments (NGM applied downstream)
        K2w = Vector{Matrix{ETp}}(undef, Tn)
        Gw  = Vector{Matrix{ETp}}(undef, Tn)
        ll = zero(ETp)
        for t in 1:Tn
            K1 = Matrix{ETp}(undef, A, A); K2 = Matrix{ETp}(undef, A, A); G = Matrix{ETp}(undef, A, A)
            μ = _mu_matrix(c_vec[t] .+ @view Fld[:, t])
            ll += _cell_moments!(K1, K2, G, μ, t, @view disp[:, t])
            K1w[t] = K1; K2w[t] = K2; Gw[t] = G           # fresh matrices per week (not reused buffers)
        end
        Turing.@addlogprob! ll
        return (; K1 = K1w, K2 = K2w, G = Gw)
    end
end

# ======================================================================================
# Stage 2 — infection / renewal block, conditioning on a FIXED per-week C* trajectory.
# `Cstar_weeks` is one Stage-1 draw's moments run through `contact_star(nb, …)` (length Tn,
# positionally aligned to `wd`). Samples the transmission latents and the renewal likelihood;
# C* is NOT re-scaled (the -gnorm S̄ decoupling was reverted), so `gamma_sar` is the per-contact
# secondary attack rate and reproduces the reference cell N_11 = susc₁·inf₁ = γ_SAR directly.
# ======================================================================================
@model function model_transmission(Cstar_weeks, wd::WindowData, w, cfg::FrameworkConfig)
    A = wd.A
    Tn = length(wd.weeks)

    # ---- transmission latents: per-contact SAR γ_SAR + relative susc/inf (analysis-plan form) ----
    # γ_SAR (per-contact secondary attack rate) carries the NGM level; inherent susceptibility &
    # infectivity are RELATIVE, normalised so the reference bin 1 ("2-10") = 1 (bins 2..A estimated),
    # so `z_s`/`z_i` have length A-1. NGM index convention: susc on susceptible row a, inf on
    # infectious column b (Munday Eq 3).
    log_gamma_sar ~ Normal(cfg.gamma_sar_prior[1], cfg.gamma_sar_prior[2])  # centre log(0.33), calibrated
    gamma_sar = exp(_softclamp(log_gamma_sar, log(0.02), log(5.0)))         # secondary attack rate, soft-bounded

    # susc/inf are RELATIVE (bin 1 = 1) and SOFT-CLAMPED to ≈[0.05, 20]: the fitted age profiles
    # are only ~3-5× (deep in the interior where _softclamp is the identity), but a stray Stage-2
    # Pathfinder draw can send a raw offset `σ·z` to ±100 → `exp` gives susc/inf ~1e8, an
    # astronomically supercritical NGM and an Inf/NaN forecast. Bounding the log-offset caps that
    # tail at the source while leaving real profiles undistorted (mirrors the κ/γ_SAR clamps).
    sig_s ~ truncated(Normal(0.1, 0.02); lower = 0)
    z_s ~ filldist(Normal(0, 1), A - 1)                    # A-1 non-reference offsets (bins 2..A)
    susc = vcat(one(sig_s), exp.(_softclamp.(sig_s .* z_s, -3.0, 3.0)))  # susc[1]=1; ∈ ≈[0.05,20]

    sig_i ~ truncated(Normal(0.1, 0.02); lower = 0)
    z_i ~ filldist(Normal(0, 1), A - 1)
    inf = vcat(one(sig_i), exp.(_softclamp.(sig_i .* z_i, -3.0, 3.0)))   # inf[1]=1;  ∈ ≈[0.05,20]

    F ~ Beta(5, 1)
    sigma_inf ~ truncated(Normal(0.05, 0.025); lower = 0)

    # ---- infection likelihood over the fitting weeks (t > smax); NGM uses week-t C* ----
    # (antibody and contacts vary by week; C*_t is the fixed Cstar_weeks[t].)
    for t in (cfg.smax + 1):Tn
        N = build_ngm(Cstar_weeks[t], susc, inf, F, wd.antibody[:, t]; gamma_sar = gamma_sar)
        pred = renewal_next(N, wd.I_mean, t, w)
        for a in 1:A
            σ = sqrt((sigma_inf * wd.I_mean[a, t])^2 + wd.I_sd[a, t]^2)
            wd.I_mean[a, t] ~ Normal(pred[a], σ)
        end
    end

    return (; susc, inf, F, gamma_sar, sigma_inf)
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
    fit_stage1(dm, ds, pop, cfg; use_nuts=cfg.stage1_use_nuts, ndraws_pf, n_sample=250)

Stage-1 (contact-degree GP) fit: Pathfinder init → (optionally) NUTS on `model_degree`.
Returns `(; chn, model, pf)`. `chn` is the Pathfinder approximate posterior (default) or the
NUTS chain; both carry the same GP parameter names. `ndraws_pf`/`n_sample` are kept ≥
`cfg.n_stage1_post` so there are always enough draws to impute into Stage 2.
"""
function fit_stage1(dm::ContactDegreeModel, ds, pop, cfg::FrameworkConfig;
                    use_nuts::Bool = cfg.stage1_use_nuts,
                    ndraws_pf::Int = max(200, cfg.n_stage1_post),
                    n_sample::Int = max(250, cfg.n_stage1_post),
                    adtype = AutoReverseDiff(), rng = nothing)
    if rng === nothing
        Random.seed!(cfg.seed)
        rng = Random.default_rng()
    end
    model = model_degree(dm, ds, pop, cfg)
    pf = adtype === nothing ? pathfinder(model; ndraws = ndraws_pf, rng = rng) :
                              pathfinder(model; ndraws = ndraws_pf, rng = rng, adtype = adtype)
    if !use_nuts
        return (; chn = pf.draws_transformed, model, pf)
    end
    pnames = names(pf.draws_transformed, :parameters)
    means  = [mean(pf.draws_transformed[:, p, :]) for p in pnames]
    init   = DynamicPPL.InitFromParams(NamedTuple(zip(pnames, means)))
    sampler = adtype === nothing ? NUTS() : NUTS(; adtype = adtype)
    local chn
    try
        chn = sample(rng, model, sampler, n_sample; initial_params = init, progress = false)
    catch err
        @warn "Stage-1 NUTS failed; falling back to Pathfinder draws" err
        chn = pf.draws_transformed
    end
    return (; chn, model, pf)
end

"""
    fit_or_load_stage1(path, dm, ds, pop, cfg; adtype, rng) -> (; chn)

Reload the Stage-1 chain at `path` if present, else fit (`fit_stage1`) and save
(`jldsave(path; result=chn)`). Idempotent skip ⇒ resumable prefit.
"""
function fit_or_load_stage1(path::AbstractString, dm::ContactDegreeModel, ds, pop,
                            cfg::FrameworkConfig; adtype = AutoReverseDiff(), rng = nothing)
    isfile(path) && return (; chn = load(path, "result"))
    res = fit_stage1(dm, ds, pop, cfg; use_nuts = cfg.stage1_use_nuts, adtype = adtype, rng = rng)
    jldsave(path; result = res.chn)
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
    fit_stage2_pooled(nb, moment_draws, wd, cfg; n_draw=cfg.n_stage2_draws, base_seed, max_concurrent)

Cut-inference Stage 2: for EACH of the `M = length(moment_draws)` imputed Stage-1 draws, form
the fixed per-week `C*` (via `contact_star(nb, …)`), Pathfinder-fit `model_transmission`, and keep
`n_draw` infection draws. Pool the `M × n_draw` (= 100×100 = 10_000) draws. Returns a NamedTuple
`(; gamma_sar, susc, inf, F, sigma_inf, post_index, Cstar_end, n_post, n_draw)` where the first
five are the pooled per-draw infection parameters (`susc`/`inf` are `N×A`), `post_index[d]` is the
Stage-1 draw `d` came from, and `Cstar_end[m]` is Stage-1 draw `m`'s origin-week (`[end]`) `C*` — the
matrix the forecast NGM is built from. The `M` per-draw fits run under `Semaphore(max_concurrent)`
(each with its own deterministic RNG `base_seed + m`), writing disjoint preallocated slots.
"""
function fit_stage2_pooled(nb::NGMBuilder, moment_draws, wd::WindowData, cfg::FrameworkConfig;
                           n_draw::Int = cfg.n_stage2_draws, adtype = AutoReverseDiff(),
                           base_seed::Int = cfg.seed, max_concurrent::Int = 1)
    A = wd.A; Tn = length(wd.weeks)
    w = gen_interval_pmf(cfg.gen_mean_days, cfg.gen_sd_days; smax = cfg.smax)
    M = length(moment_draws)
    Cstar_end = Vector{Matrix{Float64}}(undef, M)
    per_m = Vector{Any}(undef, M)
    fit_m(m) = begin
        md = moment_draws[m]
        Cstar_m = [Float64.(contact_star(nb, md.K1[t], md.K2[t], md.G[t])) for t in 1:Tn]
        Cstar_end[m] = Cstar_m[end]
        model = model_transmission(Cstar_m, wd, w, cfg)
        rng = Random.Xoshiro(base_seed + m)
        pf = adtype === nothing ? pathfinder(model; ndraws = n_draw, rng = rng) :
                                  pathfinder(model; ndraws = n_draw, rng = rng, adtype = adtype)
        gq = vec(generated_quantities(model, pf.draws_transformed))
        nd = min(n_draw, length(gq))
        gs = Vector{Float64}(undef, nd); Fv = Vector{Float64}(undef, nd); sg = Vector{Float64}(undef, nd)
        su = Matrix{Float64}(undef, nd, A); infm = Matrix{Float64}(undef, nd, A)
        for d in 1:nd
            q = gq[d]
            gs[d] = q.gamma_sar; Fv[d] = q.F; sg[d] = q.sigma_inf
            su[d, :] = q.susc; infm[d, :] = q.inf
        end
        per_m[m] = (; gamma_sar = gs, susc = su, inf = infm, F = Fv, sigma_inf = sg)
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
    post_index = reduce(vcat, (fill(m, length(per_m[m].gamma_sar)) for m in 1:M))
    return (; gamma_sar, susc, inf, F, sigma_inf, post_index, Cstar_end, n_post = M, n_draw = n_draw)
end

# ---- parallel pre-fitting of the (origin × combo × horizon) two-stage artefacts -------
# Fits are mutually independent (each is one Pathfinder run on its own data), so we fan
# them out over Julia threads with a concurrency cap chosen to balance CPU and memory.
# Threading (not Distributed) keeps memory low — one process, shared compiled code — which
# matters here: each worker process would otherwise re-load/compile the whole Turing stack.

"Available RAM (GiB): `/proc/meminfo` `MemAvailable` (counts reclaimable cache), else `Sys.free_memory`."
function _mem_available_gib()
    try
        for line in eachline("/proc/meminfo")
            startswith(line, "MemAvailable:") && return parse(Int, split(line)[2]) / 2^20
        end
    catch
    end
    return Sys.free_memory() / 2^30
end

"""
    fit_concurrency(; mem_per_fit_gib=1.0, reserve_gib=4.0)

How many joint fits to run at once, balancing CPU and memory: the minimum of the Julia
thread count, (physical cores − 1), and how many `mem_per_fit_gib`-sized fits fit in
available RAM after a `reserve_gib` headroom. Always ≥ 1.
"""
function fit_concurrency(; mem_per_fit_gib::Real = 1.0, reserve_gib::Real = 4.0)
    mem_cap = floor(Int, max(0.0, _mem_available_gib() - reserve_gib) / mem_per_fit_gib)
    cpu_cap = min(Threads.nthreads(), max(1, Sys.CPU_THREADS - 1))
    return max(1, min(cpu_cap, mem_cap))
end

"""
    prefit_stage1!(dms, wins, cfg; data_provider, save_dir, max_concurrent, adtype)

Fit every MISSING Stage-1 chain (`8j_s1_<degree>_<contacts>_<origin>_h<h>.jld2`) for the distinct
degree models `dms` × origins `wins` × horizons. Origins are processed sequentially (bounded
memory: one origin's data held at a time via `data_provider(oi, win) → (wd0, apd_by_h)`); within an
origin the (degree × horizon) fits fan out under `Semaphore(max_concurrent)`. Each fit gets its own
RNG and writes a distinct file, and BLAS is pinned to one thread during the parallel region. Cached
chains are skipped ⇒ resumable. Returns `(; fitted, failed, skipped, concurrency)`.
"""
function prefit_stage1!(dms, wins, cfg::FrameworkConfig; data_provider,
                        save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                        max_concurrent::Int = fit_concurrency(), adtype = AutoReverseDiff())
    mkpath(save_dir)
    K   = clamp(max_concurrent, 1, Threads.nthreads())
    tag = contacts_label(cfg)
    fitted  = Threads.Atomic{Int}(0)
    failed  = Threads.Atomic{Int}(0)
    skipped = Threads.Atomic{Int}(0)
    warmed  = Ref(false)
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
            rest = specs
            if !warmed[]
                do_fit(specs[1]); warmed[] = true; rest = @view specs[2:end]   # warm compile before fan-out
            end
            sem = Base.Semaphore(K)
            @sync for s in rest
                Threads.@spawn begin
                    Base.acquire(sem)
                    try do_fit(s) finally Base.release(sem) end
                end
            end
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
missing one is fit on demand). Origins are processed sequentially; within an origin the (combo ×
horizon) cells run sequentially, each `fit_stage2_pooled` fanning out its `n_stage1_post` per-draw
fits under `Semaphore(max_concurrent)`. Cached pooled files are skipped ⇒ resumable.
"""
function prefit_stage2!(combos, wins, cfg::FrameworkConfig; data_provider,
                        save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                        max_concurrent::Int = fit_concurrency(), adtype = AutoReverseDiff())
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
                ds_h = build_degree_stats(s.dm, apd_by_h[s.hi], cfg)
                s1p  = stage1_path(s.dm, win_o.origin, s.h; contacts = tag, save_dir = save_dir)
                s1   = fit_or_load_stage1(s1p, s.dm, ds_h, wd0.pop, cfg;
                                          adtype = adtype, rng = Random.Xoshiro(cfg.seed))
                md   = stage1_moment_draws(s.dm, ds_h, wd0.pop, cfg, s1.chn; n_post = cfg.n_stage1_post)
                pooled = fit_stage2_pooled(s.nb, md, wd0, cfg; adtype = adtype,
                                           base_seed = cfg.seed + 1000 * s.h, max_concurrent = K)
                jldsave(s.path; pooled)
                fitted += 1
            catch err
                failed += 1
                @warn "stage2 fit failed" origin=win_o.origin degree=degree_label(s.dm) ngm=ngm_label(s.nb) h=s.h exception=(err, catch_backtrace())
            end
        end
        @info "prefit_stage2!: origin $(win_o.origin) done ($fitted fitted, $failed failed)"
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
                           max_concurrent::Int = fit_concurrency(), adtype = AutoReverseDiff())
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
`AgePairData` for that horizon window (built on demand if `nothing`).
"""
function fit_or_load_stage2(dm, nb, wd0::WindowData, cfg::FrameworkConfig, win0::WeeklyWindow,
                            h::Integer; apd_h = nothing, grid = cis_age_grid(),
                            setting::Symbol = :all,
                            save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                            adtype = AutoReverseDiff())
    tag = contacts_label(cfg)
    s2p = stage2_path(dm, nb, win0.origin, h; contacts = tag, save_dir = save_dir)
    isfile(s2p) && return load(s2p, "pooled")
    if apd_h === nothing
        win_h = WeeklyWindow(win0.origin + Day(7 * h); n_fit = cfg.n_fit, smax = cfg.smax,
                             horizons = cfg.horizons)
        apd_h = prepare_degree_data(win_h, cfg; grid = grid, setting = setting)
    end
    ds_h = build_degree_stats(dm, apd_h, cfg)
    s1p  = stage1_path(dm, win0.origin, h; contacts = tag, save_dir = save_dir)
    s1   = fit_or_load_stage1(s1p, dm, ds_h, wd0.pop, cfg; adtype = adtype, rng = Random.Xoshiro(cfg.seed))
    md   = stage1_moment_draws(dm, ds_h, wd0.pop, cfg, s1.chn; n_post = cfg.n_stage1_post)
    pooled = fit_stage2_pooled(nb, md, wd0, cfg; adtype = adtype, base_seed = cfg.seed + 1000 * h)
    jldsave(s2p; pooled)
    return pooled
end

"""
    two_stage_forecast(dm, nb, wd0, cfg, win0; apd_by_h=nothing, grid, setting, save_dir, adtype)

Pooled `A × H × N` forecast (N = n_stage1_post·n_stage2_draws = 10_000). Infections and antibody are
frozen at the baseline `win0.origin` (t₀); for each horizon `h` the Stage-2 pooled draws for
`(dm, nb, origin, h)` are reloaded (or built), and per pooled draw `d` (from Stage-1 draw
`m = post_index[d]`) the NGM `N = build_ngm(Cstar_end[m], susc[d], inf[d], F[d], antibody_t₀;
gamma_sar=gamma_sar[d])` takes one renewal step against the history (observed lags up to t₀ plus the
intervening horizons' MEAN forecasts — per-draw coherence across horizons is undefined, mirroring the
former `iterated_forecast`). `apd_by_h[hi]` (optional) is the pre-built horizon-window `AgePairData`.
"""
function two_stage_forecast(dm, nb, wd0::WindowData, cfg::FrameworkConfig, win0::WeeklyWindow;
                            apd_by_h = nothing, grid = cis_age_grid(), setting::Symbol = :all,
                            save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                            adtype = AutoReverseDiff())
    A = wd0.A; H = length(cfg.horizons)
    w = gen_interval_pmf(cfg.gen_mean_days, cfg.gen_sd_days; smax = cfg.smax)
    mkpath(save_dir)
    hist = collect(float.(wd0.I_mean))                 # A × Tn0, last col = origin (t₀)
    cols = Vector{Matrix{Float64}}(undef, H)
    for (hi, h) in enumerate(cfg.horizons)
        apd_h  = apd_by_h === nothing ? nothing : apd_by_h[hi]
        pooled = fit_or_load_stage2(dm, nb, wd0, cfg, win0, h; apd_h = apd_h, grid = grid,
                                    setting = setting, save_dir = save_dir, adtype = adtype)
        Np  = length(pooled.gamma_sar)
        rng = MersenneTwister(cfg.seed + h)
        acc = zeros(A)                                 # renewal-weighted history (fixed within horizon)
        for s in 1:cfg.smax
            acc .+= w[s] .* hist[:, end - s + 1]
        end
        draws_h = Array{Float64}(undef, A, Np)
        preds   = Array{Float64}(undef, A, Np)         # deterministic renewal mean per draw
        for d in 1:Np
            m = pooled.post_index[d]
            N = build_ngm(pooled.Cstar_end[m], pooled.susc[d, :], pooled.inf[d, :],
                          pooled.F[d], wd0.antibody[:, end]; gamma_sar = pooled.gamma_sar[d])
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
