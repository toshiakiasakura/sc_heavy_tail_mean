# joint_model.jl — the joint Turing model (contact-degree likelihood + infection
# likelihood in one model) and the Pathfinder→NUTS fit + posterior forecast.
#
# Composability: one @model serves all four (degree model × NGM builder) combos —
# `dm::ContactDegreeModel` selects the degree likelihood + latent shape/dispersion,
# `nb::NGMBuilder` selects the NGM contact functional (deterministic dispatch). Both
# are fixed model arguments, so the parameter space is well-defined per fit.

block_of(a::Int, cfg::FrameworkConfig) = a <= cfg.child_bins ? 1 : 2

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
    pair_list, pair_index = _unordered_pairs(A)
    return (; dd_count = p.dd_count, pos_weight = p.pos_weight, p0 = p.p0,
              n = p.n, log_emp = log_emp, A = A,
              mid = cis_age_midpoints(), pair_list = pair_list, pair_index = pair_index)
end

# --- per-cell raw moments (⟨k⟩, ⟨k²⟩) + zero factor g, from the fitted params ---
# g scales the neighbourhood-degree C0 to condition on non-zero contacts (inst/1c,1d):
#   NegBin  g = 1/(1−P₀), P₀ = (φ/(φ+μ))^φ  — left-truncated fitted NegBin.
#   Weibull g = (1−p⁰)                       — empirical hurdle non-zero probability.
# The MeanNGM builder ignores g. ⚠ Floor (1−P₀) so near-empty cells (μ→0 ⇒ P₀→1)
# don't blow up; the model's μ clamp keeps the mode interior (see tasks/lessons.md).
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
    # age-pair grid with a shared length-scale ρ, non-centred as f = η·L·z (L = chol K).
    logpop = log.(wd.pop)
    log_rho ~ Normal(cfg.gp_len_prior[1], cfg.gp_len_prior[2])
    log_eta ~ Normal(cfg.gp_scale_prior[1], cfg.gp_scale_prior[2])
    ρ = exp(clamp(log_rho, log(3.0), log(45.0)))          # length-scale (age-years)
    η = exp(clamp(log_eta, -3.0, 2.0))                    # GP marginal scale
    c0 = mean(ds.log_emp .- logpop')                      # smooth constant mean-fn anchor
    c ~ Normal(c0, 3.0)
    z ~ filldist(Normal(0, 1), length(ds.pair_list))      # 28 iid (non-centred GP)
    mid = ds.mid
    # 28×28 separable RBF (upper-triangular submatrix of the I₄₉ kernel over pairs)
    Kp = [exp(-((mid[p[1]] - mid[q[1]])^2 + (mid[p[2]] - mid[q[2]])^2) / (2 * ρ^2))
          for p in ds.pair_list, q in ds.pair_list]
    Lp = cholesky(Symmetric(Kp) + 1e-6 * I).L
    r = c .+ η .* (Lp * z)                                 # symmetric 28-vector log-rate
    # clamp guards exp under/overflow (μ ∈ [3e-4, 400]); the mode stays well interior.
    μ = [exp(clamp(r[ds.pair_index[i, j]] + logpop[j], -8.0, 6.0)) for i in 1:A, j in 1:A]

    if is_weighted(dm)
        log_kappa ~ filldist(Normal(0.0, 0.5), 2, 2)     # Weibull shape by child/adult block
    else
        log_k ~ filldist(Normal(0.0, 1.0), 2, 2)         # NegBin dispersion by block
    end

    ETp = eltype(μ)
    K1 = Matrix{ETp}(undef, A, A)
    K2 = Matrix{ETp}(undef, A, A)
    G  = Matrix{ETp}(undef, A, A)                          # per-cell zero factor (see helpers)
    ll = zero(ETp)
    for i in 1:A, j in 1:A
        bi = block_of(i, cfg); bj = block_of(j, cfg)
        if is_weighted(dm)
            κ = exp(clamp(log_kappa[bi, bj], -3.0, 3.0))   # shape ∈ [0.05, 20]
            λ = μ[i, j] / gamma(1 + 1 / κ)
            pos = ds.pos_weight[i, j]
            isempty(pos) || (ll += sum(logpdf.(Weibull(κ, λ), pos)))
            k1, k2, g = _weibull_moments(μ[i, j], κ, ds.p0[i, j])
        else
            kk = exp(clamp(log_k[bi, bj], -4.0, 5.0))       # dispersion ∈ [0.018, 148]
            ll += calculate_loglikelihood(ds.dd_count[i, j], NegBin(μ[i, j], kk))
            k1, k2, g = _negbin_moments(μ[i, j], kk)
        end
        K1[i, j] = k1; K2[i, j] = k2; G[i, j] = g
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
    Cstar = contact_star(nb, K1, K2, G, wd.pop)

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
                   ndraws_pf::Int = 200, adtype = nothing, rng = nothing)
    # `rng === nothing` keeps the original single-thread behaviour (seed the global RNG);
    # a supplied RNG (an isolated per-fit stream) makes the fit **thread-safe** for the
    # parallel pre-fit — no shared global-RNG mutation (see `prefit_chains!`).
    if rng === nothing
        Random.seed!(cfg.seed)
        rng = Random.default_rng()
    end
    w = gen_interval_pmf(cfg.gen_mean_days, cfg.gen_sd_days; smax = cfg.smax)
    model = model_joint(dm, nb, ds, wd, w, cfg)

    pf = pathfinder(model; ndraws = ndraws_pf, rng = rng)
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
        N_origin = build_ngm(q.Cstar, q.susc, q.inf, q.F, wd.antibody[:, Tn])
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
                           use_nuts::Bool = false, rng = nothing)
    model = model_joint(dm, nb, ds, wd, w, cfg)
    if isfile(path)
        return (; chn = load(path, "result"), model)
    end
    res = fit_joint(dm, nb, ds, wd, cfg; use_nuts = use_nuts, rng = rng)
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
                        use_nuts::Bool = false,
                        save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                        max_concurrent::Int = fit_concurrency())
    mkpath(save_dir)
    specs = NamedTuple[]
    for (oi, win_o) in enumerate(wins), (dm, nb) in combos, (hi, h) in enumerate(cfg.horizons)
        path = joinpath(save_dir,
            "8j_chn_$(degree_label(dm))_$(ngm_label(nb))_$(win_o.origin)_h$(h).jld2")
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
                             rng = Random.Xoshiro(cfg.seed))
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
`h = 1..H` the contact/degree window slides to end at `t₀+h−1`, the joint model is
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
                           use_nuts::Bool = false,
                           save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                           ndraws::Int = cfg.n_forecast_draws, apd_by_h = nothing)
    A = wd0.A; H = length(cfg.horizons)
    w = gen_interval_pmf(cfg.gen_mean_days, cfg.gen_sd_days; smax = cfg.smax)
    mkpath(save_dir)
    hist = collect(float.(wd0.I_mean))                 # A × Tn0, last col = origin (t₀)
    cols = Vector{Matrix{Float64}}(undef, H)           # per-horizon A × keep draws
    for (hi, h) in enumerate(cfg.horizons)
        # contact/degree window ending at t₀ + (h-1) weeks (sliding; infections stay at t₀)
        origin_h = win0.origin + Day(7 * (h - 1))
        win_h = WeeklyWindow(origin_h; n_fit = cfg.n_fit, smax = cfg.smax, horizons = cfg.horizons)
        apd_h = apd_by_h === nothing ?
            prepare_degree_data(win_h, cfg; grid = grid, setting = setting) : apd_by_h[hi]
        ds_h = build_degree_stats(dm, apd_h, cfg)
        path = joinpath(save_dir,
            "8j_chn_$(degree_label(dm))_$(ngm_label(nb))_$(win0.origin)_h$(h).jld2")
        fl = fit_or_load_chain(path, dm, nb, ds_h, wd0, cfg, w; use_nuts = use_nuts)

        gq = vec(generated_quantities(fl.model, fl.chn))
        keep = min(ndraws, length(gq))
        idx = round.(Int, range(1, length(gq); length = keep))
        rng = MersenneTwister(cfg.seed + h)
        draws_h = Array{Float64}(undef, A, keep)
        step_mean = zeros(A)
        for (d, k) in enumerate(idx)
            q = gq[k]
            N = build_ngm(q.Cstar, q.susc, q.inf, q.F, wd0.antibody[:, end])   # antibody frozen at t₀
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
