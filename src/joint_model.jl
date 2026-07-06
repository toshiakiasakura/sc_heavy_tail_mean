# joint_model.jl — the joint Turing model (contact-degree likelihood + infection
# likelihood in one model) and the Pathfinder→NUTS fit + posterior forecast.
#
# Composability: one @model serves all four (degree model × NGM builder) combos —
# `dm::ContactDegreeModel` selects the degree likelihood + latent shape/dispersion,
# `nb::NGMBuilder` selects the NGM contact functional (deterministic dispatch). Both
# are fixed model arguments, so the parameter space is well-defined per fit.

block_of(a::Int, cfg::FrameworkConfig) = a <= cfg.child_bins ? 1 : 2

"""
    build_degree_stats(dm, apd, cfg)

Pool the per-week cells (lean `constant_contacts=true`) and precompute the fixed
per-cell inputs the model needs: degree distributions, empirical zero prob `p0`,
and the prior-centre `log_emp` (log count-mean for NegBin; log positive-weight mean
for the hurdle).
"""
function build_degree_stats(dm::ContactDegreeModel, apd::AgePairData, cfg::FrameworkConfig)
    cfg.constant_contacts || error("per-week contacts not wired in the lean preliminary")
    p = pool_over_time(apd)
    A = apd.A
    log_emp = Matrix{Float64}(undef, A, A)
    for i in 1:A, j in 1:A
        base = if is_weighted(dm)
            isempty(p.pos_weight[i, j]) ? 1e-3 : mean(p.pos_weight[i, j])   # μW init
        else
            max(p.emean[i, j], 1e-3)                                        # count-mean init
        end
        log_emp[i, j] = log(base)
    end
    return (; dd_count = p.dd_count, pos_weight = p.pos_weight, p0 = p.p0,
              n = p.n, log_emp = log_emp, A = A)
end

# --- per-cell raw moments (⟨k⟩, ⟨k²⟩) from the fitted degree parameters ---
_negbin_moments(m, k)      = (m, m + m^2 * (1 + 1 / k))                       # ⟨k⟩, ⟨k²⟩
function _weibull_moments(μW, κ, p0)                                          # incl-zero raw moments
    cvw2 = gamma(1 + 2 / κ) / gamma(1 + 1 / κ)^2 - 1
    return ((1 - p0) * μW, (1 - p0) * μW^2 * (1 + cvw2))
end

@model function model_joint(dm::ContactDegreeModel, nb::NGMBuilder,
                            ds, wd::WindowData, w, cfg::FrameworkConfig)
    A = wd.A
    Tn = length(wd.weeks)

    # ---- contact-degree latents (constant over the window; RW1/GP swap-in later) ----
    z_mu ~ filldist(Normal(0, 1), A, A)
    # clamp guards against exp under/overflow on extreme optimiser excursions
    # (μ ∈ [3e-4, 400]); the mode is well interior so gradients are unaffected there.
    μ = exp.(clamp.(ds.log_emp .+ z_mu, -8.0, 6.0))

    if is_weighted(dm)
        log_kappa ~ filldist(Normal(0.0, 0.5), 2, 2)     # Weibull shape by child/adult block
    else
        log_k ~ filldist(Normal(0.0, 1.0), 2, 2)         # NegBin dispersion by block
    end

    ETp = eltype(μ)
    K1 = Matrix{ETp}(undef, A, A)
    K2 = Matrix{ETp}(undef, A, A)
    ll = zero(ETp)
    for i in 1:A, j in 1:A
        bi = block_of(i, cfg); bj = block_of(j, cfg)
        if is_weighted(dm)
            κ = exp(clamp(log_kappa[bi, bj], -3.0, 3.0))   # shape ∈ [0.05, 20]
            λ = μ[i, j] / gamma(1 + 1 / κ)
            pos = ds.pos_weight[i, j]
            isempty(pos) || (ll += sum(logpdf.(Weibull(κ, λ), pos)))
            k1, k2 = _weibull_moments(μ[i, j], κ, ds.p0[i, j])
        else
            kk = exp(clamp(log_k[bi, bj], -4.0, 5.0))       # dispersion ∈ [0.018, 148]
            ll += calculate_loglikelihood(ds.dd_count[i, j], NegBin(μ[i, j], kk))
            k1, k2 = _negbin_moments(μ[i, j], kk)
        end
        K1[i, j] = k1; K2[i, j] = k2
    end
    Turing.@addlogprob! ll

    # ---- transmission latents (reference priors, non-centred; stan:167-174) ----
    mu_s ~ Beta(24, 24)
    sig_s ~ truncated(Normal(0.1, 0.02); lower = 0)
    z_s ~ filldist(Normal(0, 1), A)
    susc = exp.(mu_s .+ sig_s .* z_s)

    mu_i ~ Beta(4, 12)
    sig_i ~ truncated(Normal(0.1, 0.02); lower = 0)
    z_i ~ filldist(Normal(0, 1), A)
    inf = exp.(mu_i .+ sig_i .* z_i)

    F ~ Beta(5, 1)
    sigma_inf ~ truncated(Normal(0.05, 0.025); lower = 0)

    # ---- NGM (reciprocity-balanced C* once; antibody varies by week) ----
    Cstar = contact_star(nb, K1, K2, wd.pop)

    # ---- infection likelihood over the fitting weeks (t > smax) ----
    for t in (cfg.smax + 1):Tn
        N = build_ngm(Cstar, susc, inf, F, wd.antibody[:, t])
        pred = renewal_next(N, wd.I_mean, t, w)
        for a in 1:A
            σ = sqrt((sigma_inf * wd.I_mean[a, t])^2 + wd.I_sd[a, t]^2)
            wd.I_mean[a, t] ~ Normal(pred[a], σ)
        end
    end

    return (; susc, inf, F, sigma_inf, Cstar)
end

"""
    fit_joint(dm, nb, ds, wd, cfg; n_sample=250, use_nuts=true, ndraws_pf=200)

Pathfinder init → (optionally) NUTS. Returns `(; chn, model, w, pf)`. If `use_nuts`
is false or NUTS fails, the Pathfinder approximate-posterior draws are returned as
`chn` (they carry the same parameter names).
"""
function fit_joint(dm::ContactDegreeModel, nb::NGMBuilder, ds, wd::WindowData,
                   cfg::FrameworkConfig; n_sample::Int = 250, use_nuts::Bool = true,
                   ndraws_pf::Int = 200, adtype = nothing)
    Random.seed!(cfg.seed)
    w = gen_interval_pmf(cfg.gen_mean_days, cfg.gen_sd_days; smax = cfg.smax)
    model = model_joint(dm, nb, ds, wd, w, cfg)

    pf = pathfinder(model; ndraws = ndraws_pf)
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
        chn = sample(model, sampler, n_sample; initial_params = init, progress = false)
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
        N_origin = build_ngm(q.Cstar, q.susc, q.inf, q.F, wd.antibody[:, Tn])
        mean_path = forecast_forward(N_origin, wd.I_mean[:, seed_cols], w, H)
        for a in 1:A, h in 1:H
            σ = max(q.sigma_inf * mean_path[a, h], 1e-6)
            out[a, h, d] = mean_path[a, h] + σ * randn(rng)
        end
    end
    return out
end
