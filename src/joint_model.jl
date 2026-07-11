# joint_model.jl — the joint Turing model (contact-degree likelihood + infection
# likelihood in one model) and the Pathfinder→NUTS fit + posterior forecast.
#
# Composability: one @model serves all four (degree model × NGM builder) combos —
# `dm::ContactDegreeModel` selects the degree likelihood + latent shape/dispersion,
# `nb::NGMBuilder` selects the NGM contact functional (deterministic dispatch). Both
# are fixed model arguments, so the parameter space is well-defined per fit.

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
  independent age-pair GP per week (`model_joint`'s per-week branch).

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

@model function model_joint(dm::ContactDegreeModel, nb::NGMBuilder,
                            ds, wd::WindowData, w, cfg::FrameworkConfig)
    A = wd.A
    Tn = length(wd.weeks)

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
    logpop = log.(wd.pop ./ wd.pop[1])
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

    # ---- contact-degree likelihood → per-week C* (Cstar_weeks[t]) ----
    if cfg.constant_contacts
        # pooled: one latent field, one C* reused for every renewal week.
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
        Cstar1 = contact_star(nb, K1, K2, G)
        Cstar_weeks = [Cstar1 for _ in 1:Tn]
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
        Cstar_weeks = Vector{Matrix{ETp}}(undef, Tn)
        K1 = Matrix{ETp}(undef, A, A); K2 = Matrix{ETp}(undef, A, A); G = Matrix{ETp}(undef, A, A)
        ll = zero(ETp)
        for t in 1:Tn
            μ = _mu_matrix(c_vec[t] .+ @view Fld[:, t])
            ll += _cell_moments!(K1, K2, G, μ, t, @view disp[:, t])
            Cstar_weeks[t] = contact_star(nb, K1, K2, G)
        end
        Turing.@addlogprob! ll
    end

    # ---- normalize C* by the fitting-window mean contact intensity (decouple γ) ----
    # S̄ = fit-window-averaged, population-weighted mean effective contacts per person. Homogeneous
    # degree-1 in C*, so the renewal likelihood becomes scale-invariant in C* — the absolute contact
    # LEVEL moves into γ and the NGM sees only temporal change in contacts. The degree likelihood
    # (raw μ) is unaffected; reconstruct_mu_draws / the 10j heatmap read raw μ, also unaffected.
    wpop    = wd.pop ./ sum(wd.pop)
    fitcols = (cfg.smax + 1):Tn                       # the n_fit infection-likelihood weeks
    S̄ = mean(sum(wpop[a] * Cstar_weeks[t][a, b] for a in 1:A, b in 1:A) for t in fitcols)
    S̄ = max(S̄, 1e-8)                                 # guard fully-empty windows
    Cstar_weeks = [C ./ S̄ for C in Cstar_weeks]       # normalized C* reused by infection loop + return

    # ---- transmission latents: absolute γ + relative susc/inf (analysis-plan reparam) ----
    # ONE absolute-transmissibility scalar γ carries the NGM level; inherent susceptibility &
    # infectivity are RELATIVE, normalised so the reference bin 1 ("2-10") = 1 (bins 2..A estimated).
    # This removes the old μ_s/μ_i level pair (confounded with each other and with C*'s scale — only
    # their sum was identified). With C* now normalised to unit fit-window mean intensity (above),
    # γ ≈ susc₁·inf₁·S̄ ≈ Rt/ρ(C̃*) is data-identified and decoupled from the contact scale (no longer
    # a per-contact SAR; it is window-relative, so not comparable across origins). The old per-bin
    # `susc=exp(μ_s+σ_s z_s)` became `susc=vcat(1, exp(σ_s z_s[2:A]))` (bin-1 reference), so `z_s`/`z_i`
    # shrink from length A to A-1. NGM index convention unchanged (susc on susceptible row a, inf on
    # infectious column b; Munday Eq 3).
    log_gamma ~ Normal(cfg.gamma_prior[1], cfg.gamma_prior[2])  # centre log(0.8), calibrated (tasks/lessons.md)
    γ = exp(_softclamp(log_gamma, log(0.02), log(5.0)))         # absolute transmissibility, soft-bounded

    sig_s ~ truncated(Normal(0.1, 0.02); lower = 0)
    z_s ~ filldist(Normal(0, 1), A - 1)                    # A-1 non-reference offsets (bins 2..A)
    susc = vcat(one(sig_s), exp.(sig_s .* z_s))            # susc[1] = 1 (relative inherent susceptibility)

    sig_i ~ truncated(Normal(0.1, 0.02); lower = 0)
    z_i ~ filldist(Normal(0, 1), A - 1)
    inf = vcat(one(sig_i), exp.(sig_i .* z_i))             # inf[1] = 1 (relative infectivity)

    F ~ Beta(5, 1)
    sigma_inf ~ truncated(Normal(0.05, 0.025); lower = 0)

    # ---- infection likelihood over the fitting weeks (t > smax); NGM uses week-t C* ----
    # (antibody and — now — contacts vary by week; C*_t is Cstar_weeks[t].)
    for t in (cfg.smax + 1):Tn
        N = build_ngm(Cstar_weeks[t], susc, inf, F, wd.antibody[:, t]; γ = γ)
        pred = renewal_next(N, wd.I_mean, t, w)
        for a in 1:A
            σ = sqrt((sigma_inf * wd.I_mean[a, t])^2 + wd.I_sd[a, t]^2)
            wd.I_mean[a, t] ~ Normal(pred[a], σ)
        end
    end

    return (; susc, inf, F, γ, sigma_inf, Cstar = Cstar_weeks)
end

"""
    fit_joint(dm, nb, ds, wd, cfg; n_sample=250, use_nuts=true, ndraws_pf=200)

Pathfinder init → (optionally) NUTS. Returns `(; chn, model, w, pf)`. If `use_nuts`
is false or NUTS fails, the Pathfinder approximate-posterior draws are returned as
`chn` (they carry the same parameter names).
"""
function fit_joint(dm::ContactDegreeModel, nb::NGMBuilder, ds, wd::WindowData,
                   cfg::FrameworkConfig; n_sample::Int = 250, use_nuts::Bool = true,
                   ndraws_pf::Int = 200, adtype = AutoReverseDiff(), rng = nothing)
    # `rng === nothing` keeps the original single-thread behaviour (seed the global RNG);
    # a supplied RNG (an isolated per-fit stream) makes the fit **thread-safe** for the
    # parallel pre-fit — no shared global-RNG mutation (see `prefit_chains!`).
    # `adtype` (default ReverseDiff — the clamp-free model is ReverseDiff-compatible) is
    # threaded into BOTH Pathfinder and NUTS; `nothing` restores each backend's own default.
    if rng === nothing
        Random.seed!(cfg.seed)
        rng = Random.default_rng()
    end
    w = gen_interval_pmf(cfg.gen_mean_days, cfg.gen_sd_days; smax = cfg.smax)
    model = model_joint(dm, nb, ds, wd, w, cfg)

    pf = adtype === nothing ? pathfinder(model; ndraws = ndraws_pf, rng = rng) :
                              pathfinder(model; ndraws = ndraws_pf, rng = rng, adtype = adtype)
    if !use_nuts
        return (; chn = pf.draws_transformed, model, w, pf)
    end

    # init NUTS from the Pathfinder posterior mean
    pnames = names(pf.draws_transformed, :parameters)
    means  = [mean(pf.draws_transformed[:, p, :]) for p in pnames]
    init   = DynamicPPL.InitFromParams(NamedTuple(zip(pnames, means)))
    sampler = adtype === nothing ? NUTS() : NUTS(; adtype = adtype)
    local chn
    try
        chn = sample(rng, model, sampler, n_sample; initial_params = init, progress = false)
    catch err
        @warn "NUTS failed; falling back to Pathfinder draws" err
        chn = pf.draws_transformed
    end
    return (; chn, model, w, pf)
end

"""
    posterior_forecast(model, chn, wd, cfg, w; ndraws)

Posterior-predictive `A × H × ndraws` forecast. Per draw: freeze the NGM at the
origin week (last week; antibody held at origin), iterate the renewal `H` weeks
(reference stan:309-314), and add observation noise `σ = sigma_inf·pred`.
"""
function posterior_forecast(model, chn, wd::WindowData, cfg::FrameworkConfig, w;
                            ndraws::Int = cfg.n_forecast_draws)
    gq = generated_quantities(model, chn)
    gq = vec(gq)
    keep = min(ndraws, length(gq))
    idx = round.(Int, range(1, length(gq); length = keep))
    A = wd.A; H = length(cfg.horizons); Tn = length(wd.weeks)
    seed_cols = (Tn - cfg.smax + 1):Tn                 # last smax weeks of history
    rng = MersenneTwister(cfg.seed)
    out = Array{Float64}(undef, A, H, keep)
    for (d, k) in enumerate(idx)
        q = gq[k]
        N_origin = build_ngm(q.Cstar[end], q.susc, q.inf, q.F, wd.antibody[:, Tn]; γ = q.γ)  # origin-week C*
        mean_path = forecast_forward(N_origin, wd.I_mean[:, seed_cols], w, H)
        for a in 1:A, h in 1:H
            σ = max(q.sigma_inf * mean_path[a, h], 1e-6)
            out[a, h, d] = mean_path[a, h] + σ * randn(rng)
        end
    end
    return out
end

"""
    fit_or_load_chain(path, dm, nb, ds, wd, cfg, w; use_nuts)

Return `(; chn, model)` for one joint fit, reloading a saved chain when `path`
exists (idempotent skip, cf. `bnb_utils.jl`). The model is always rebuilt (cheap,
deterministic) so `generated_quantities(model, chn)` works after a reload.
"""
function fit_or_load_chain(path::AbstractString, dm::ContactDegreeModel,
                           nb::NGMBuilder, ds, wd::WindowData, cfg::FrameworkConfig, w;
                           use_nuts::Bool = false, adtype = AutoReverseDiff(), rng = nothing)
    model = model_joint(dm, nb, ds, wd, w, cfg)
    if isfile(path)
        return (; chn = load(path, "result"), model)
    end
    res = fit_joint(dm, nb, ds, wd, cfg; use_nuts = use_nuts, adtype = adtype, rng = rng)
    jldsave(path; result = res.chn)
    return (; chn = res.chn, model)
end

# ---- parallel pre-fitting of the (origin × combo × horizon) chains --------------------
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
    prefit_chains!(combos, wins, wds, cfg, apd_by_h_all; grid, setting=:all,
                   use_nuts=false, save_dir, max_concurrent=fit_concurrency())

Fit every **missing** `(origin × combo × horizon)` joint chain in parallel (bounded to
`max_concurrent` concurrent fits) and save each to `save_dir/8j_chn_<…>.jld2`; cached
chains are skipped. Thread-safe by construction: each fit gets its own RNG and model, each
writes a distinct file, and BLAS is pinned to one thread during the parallel region to
avoid CPU oversubscription. One spec is fit sequentially first to warm the model/AD
compilation before fan-out. Afterwards `iterated_forecast` just reloads the cached chains.

`combos` is a vector of `(dm, nb)`; `wins`/`wds` are the per-origin windows and
`WindowData`; `apd_by_h_all[oi][hi]` is the pre-built `AgePairData` for origin `oi`,
horizon `hi`. Returns `(; requested, fitted, failed, concurrency)`.
"""
function prefit_chains!(combos, wins, wds, cfg::FrameworkConfig, apd_by_h_all;
                        grid = cis_age_grid(), setting::Symbol = :all,
                        use_nuts::Bool = false, adtype = AutoReverseDiff(),
                        save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                        max_concurrent::Int = fit_concurrency())
    mkpath(save_dir)
    specs = NamedTuple[]
    for (oi, win_o) in enumerate(wins), (dm, nb) in combos, (hi, h) in enumerate(cfg.horizons)
        path = joinpath(save_dir,
            "8j_chn_$(degree_label(dm))_$(ngm_label(nb))_$(contacts_label(cfg))_$(win_o.origin)_h$(h).jld2")
        isfile(path) || push!(specs, (; dm, nb, oi, hi, h, win_o, path))
    end
    total = length(combos) * length(wins) * length(cfg.horizons)
    isempty(specs) && return (; requested = 0, fitted = 0, failed = 0, concurrency = 0)

    K = clamp(max_concurrent, 1, Threads.nthreads())
    @info "prefit_chains!: fitting $(length(specs))/$total chains; concurrency=$K " *
          "(threads=$(Threads.nthreads()), cores=$(Sys.CPU_THREADS), " *
          "mem_avail=$(round(_mem_available_gib(); digits=1)) GiB)"

    fitted = Threads.Atomic{Int}(0)
    failed = Threads.Atomic{Int}(0)
    do_fit(s) = begin
        try
            ds_h = build_degree_stats(s.dm, apd_by_h_all[s.oi][s.hi], cfg)
            res  = fit_joint(s.dm, s.nb, ds_h, wds[s.oi], cfg; use_nuts = use_nuts,
                             adtype = adtype, rng = Random.Xoshiro(cfg.seed))
            jldsave(s.path; result = res.chn)
            Threads.atomic_add!(fitted, 1)
        catch err
            Threads.atomic_add!(failed, 1)
            @warn "prefit fit failed" origin=s.win_o.origin degree=degree_label(s.dm) ngm=ngm_label(s.nb) h=s.h exception=(err, catch_backtrace())
        end
    end

    old_blas = LinearAlgebra.BLAS.get_num_threads()
    LinearAlgebra.BLAS.set_num_threads(1)                 # avoid threads × BLAS oversubscription
    try
        do_fit(specs[1])                                  # warm compilation before fan-out
        if length(specs) > 1
            sem = Base.Semaphore(K)
            @sync for s in @view specs[2:end]
                Threads.@spawn begin
                    Base.acquire(sem)
                    try
                        do_fit(s)
                    finally
                        Base.release(sem)
                    end
                end
            end
        end
    finally
        LinearAlgebra.BLAS.set_num_threads(old_blas)
    end
    return (; requested = length(specs), fitted = fitted[], failed = failed[], concurrency = K)
end

"""
    iterated_forecast(dm, nb, wd0, cfg, win0; grid, setting, use_nuts, save_dir,
                      ndraws, apd_by_h)

Contact-updated iterated `A × H × K` forecast (spec inst/1d, points 2–3). Infections
and antibody are **frozen at the baseline** `win0.origin` (t₀); for each horizon
`h = 1..H` the contact/degree window slides to end at `t₀+h`, the joint model is
re-fit (or a saved chain reloaded), the NGM is refreshed, and one renewal step is
taken. The forecast for week `t₀+h` uses the observed history up to t₀ plus the
**mean** forecasts of the intervening weeks as renewal lags (per-draw coherence
across independent re-fits is undefined). One MCMC chain is saved per horizon under
`save_dir` as `8j_chn_<degree>_<ngm>_<origin>_h<h>.jld2`.

`apd_by_h` (optional) is a length-`H` vector of pre-built `AgePairData` for the
shifted windows — pass it to reuse the age-pair binning across the four combos.
"""
function iterated_forecast(dm::ContactDegreeModel, nb::NGMBuilder, wd0::WindowData,
                           cfg::FrameworkConfig, win0::WeeklyWindow;
                           grid = cis_age_grid(), setting::Symbol = :all,
                           use_nuts::Bool = false, adtype = AutoReverseDiff(),
                           save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                           ndraws::Int = cfg.n_forecast_draws, apd_by_h = nothing)
    A = wd0.A; H = length(cfg.horizons)
    w = gen_interval_pmf(cfg.gen_mean_days, cfg.gen_sd_days; smax = cfg.smax)
    mkpath(save_dir)
    hist = collect(float.(wd0.I_mean))                 # A × Tn0, last col = origin (t₀)
    cols = Vector{Matrix{Float64}}(undef, H)           # per-horizon A × keep draws
    for (hi, h) in enumerate(cfg.horizons)
        # contact/degree window ending at t₀ + h weeks (sliding; infections stay at t₀)
        origin_h = win0.origin + Day(7 * h)
        win_h = WeeklyWindow(origin_h; n_fit = cfg.n_fit, smax = cfg.smax, horizons = cfg.horizons)
        apd_h = apd_by_h === nothing ?
            prepare_degree_data(win_h, cfg; grid = grid, setting = setting) : apd_by_h[hi]
        ds_h = build_degree_stats(dm, apd_h, cfg)
        path = joinpath(save_dir,
            "8j_chn_$(degree_label(dm))_$(ngm_label(nb))_$(contacts_label(cfg))_$(win0.origin)_h$(h).jld2")
        fl = fit_or_load_chain(path, dm, nb, ds_h, wd0, cfg, w; use_nuts = use_nuts, adtype = adtype)

        gq = vec(generated_quantities(fl.model, fl.chn))
        keep = min(ndraws, length(gq))
        idx = round.(Int, range(1, length(gq); length = keep))
        rng = MersenneTwister(cfg.seed + h)
        draws_h = Array{Float64}(undef, A, keep)
        step_mean = zeros(A)
        for (d, k) in enumerate(idx)
            q = gq[k]
            N = build_ngm(q.Cstar[end], q.susc, q.inf, q.F, wd0.antibody[:, end]; γ = q.γ)   # origin-week C* (t₀+h); antibody frozen at t₀
            acc = zeros(A)
            for s in 1:cfg.smax
                acc .+= w[s] .* hist[:, end - s + 1]
            end
            pred = N * acc
            for a in 1:A
                σ = max(q.sigma_inf * pred[a], 1e-6)
                draws_h[a, d] = pred[a] + σ * randn(rng)
            end
            step_mean .+= pred
        end
        step_mean ./= keep
        cols[hi] = draws_h
        hist = hcat(hist, step_mean)                   # deterministic lag for the next horizon
    end
    K = minimum(size(c, 2) for c in cols)              # align draw count across horizons
    out = Array{Float64}(undef, A, H, K)
    for hi in 1:H, a in 1:A, d in 1:K
        out[a, hi, d] = cols[hi][a, d]
    end
    return out
end
