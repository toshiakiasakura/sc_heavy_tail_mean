# 10j_viz_utils.jl — read-only helper for the 10j single-time-point diagnostics.
#
# Reconstructs the GP-smoothed directional contact mean μ_{i→j} per posterior draw from a
# cached joint-model chain, WITHOUT rebuilding the model — mirroring the `load_transmission_draws`
# pattern in 8j_viz_utils.jl. μ is a deterministic transform of the raw sampled columns
# (log_rho, log_eta, per-week level c[t], per-week field z[·,t]); see `model_joint`
# (joint_model.jl:95–111, 158–182). Requires 8j_viz_utils.jl (for `chain_path`) to be included
# first. LinearAlgebra (cholesky/Symmetric/I) and `_unordered_pairs`/`cis_age_midpoints` come in
# via forecast_utils.jl.

"""
    reconstruct_mu_draws(lbl, origin, h; week_index=nothing, grid, contacts, save_dir)
        -> ndraws × A × A  |  nothing

Load the cached chain for `(lbl, origin, h)` and rebuild the smoothed directional contact-mean
matrix μ_{i→j} for one week, once per posterior draw:

    ρ  = exp(clamp(log_rho, log3, log45)),  η = exp(clamp(log_eta, -3, 2))
    Kp[p,q] = exp(-((mid_p1-mid_q1)² + (mid_p2-mid_q2)²)/(2ρ²))  over the 28 unordered pairs
    Lp = chol(Kp + 1e-6 I).L
    rvec = c[week] .+ η .* (Lp * z[:,week])
    μ[i,j] = exp(clamp(rvec[pair_index[i,j]] + log(pop_j), -8, 6))

`week_index` defaults to the last window week (the origin week the forecast NGM is frozen at).
Handles both the per-week regime (`c[t]`, `z[p,t]`; the cached `contacts="weekly"` chains) and
the pooled regime (scalar `c`, `z[p]`). Returns `nothing` when the chain file is missing.
"""
function reconstruct_mu_draws(lbl::AbstractString, origin::Date, h::Integer;
                              week_index::Union{Int,Nothing} = nothing,
                              grid,
                              contacts::AbstractString = "weekly",
                              save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    path = chain_path(lbl, origin, h; contacts = contacts, save_dir = save_dir)
    isfile(path) || return nothing
    chn = try
        load(path, "result")
    catch err
        @warn "could not load chain" path err
        return nothing
    end

    A = grid.N
    pair_list, pair_index = _unordered_pairs(A)
    P = length(pair_list)
    mid = cis_age_midpoints(; grid = grid)
    logpop = log.(grid.POP)

    ρ = exp.(clamp.(vec(Array(chn[:log_rho])), log(3.0), log(45.0)))
    η = exp.(clamp.(vec(Array(chn[:log_eta])), -3.0, 2.0))
    D = length(ρ)

    pnames = string.(names(chn, :parameters))
    # --- select the level c and field z for the requested week ---
    if any(n -> occursin(r"^c\[\d+\]$", n), pnames)          # per-week regime: c[t], z[p,t]
        weeks_present = sort(unique(parse(Int, match(r"^c\[(\d+)\]$", n).captures[1])
                                    for n in pnames if occursin(r"^c\[\d+\]$", n)))
        wk = week_index === nothing ? maximum(weeks_present) : week_index
        c_t = vec(Array(chn[Symbol("c[$wk]")]))              # D
        z_t = Matrix{Float64}(undef, D, P)                   # D × P (field for week wk)
        for n in pnames
            m = match(r"^z\[(\d+)\s*,\s*(\d+)\]$", n)
            m === nothing && continue
            p = parse(Int, m.captures[1]); t = parse(Int, m.captures[2])
            t == wk && (z_t[:, p] = vec(Array(chn[Symbol(n)])))
        end
    else                                                      # pooled regime: scalar c, z[p]
        c_t = vec(Array(chn[:c]))                            # D
        z_t = Matrix{Float64}(undef, D, P)
        for p in 1:P
            z_t[:, p] = vec(Array(chn[Symbol("z[$p]")]))
        end
    end

    μ = Array{Float64,3}(undef, D, A, A)
    for d in 1:D
        Kp = [exp(-((mid[p[1]] - mid[q[1]])^2 + (mid[p[2]] - mid[q[2]])^2) / (2 * ρ[d]^2))
              for p in pair_list, q in pair_list]
        Lp = cholesky(Symmetric(Kp) + 1e-6 * I).L
        rvec = c_t[d] .+ η[d] .* (Lp * @view z_t[d, :])
        for i in 1:A, j in 1:A
            μ[d, i, j] = exp(clamp(rvec[pair_index[i, j]] + logpop[j], -8.0, 6.0))
        end
    end
    return μ
end
