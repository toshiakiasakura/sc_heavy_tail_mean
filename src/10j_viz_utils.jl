# 10j_viz_utils.jl — read-only helper for the 10j single-time-point diagnostics.
#
# Reconstructs the GP-smoothed directional contact mean μ_{i→j} per posterior draw from a
# cached joint-model chain, WITHOUT rebuilding the model — mirroring the `load_transmission_draws`
# pattern in 8j_viz_utils.jl. μ is a deterministic transform of the raw sampled columns
# (log_rho_diag, log_rho_gap, log_eta; and, for the separable spatio-temporal regime, log_rho_time,
# log_sigma_c, scalar level c, temporal-level raw z_c, structure-field raw z[·,·]); see `model_joint`
# (joint_model.jl §5/§6). Requires 8j_viz_utils.jl (for `chain_path`) to be included first.
# LinearAlgebra (cholesky/Symmetric/I/dot) and `_unordered_pairs`/`cis_age_midpoints` come in via
# forecast_utils.jl.

"""
    reconstruct_mu_draws(lbl, origin, h; week_index=nothing, grid, contacts, save_dir)
        -> ndraws × A × A  |  nothing

Load the cached chain for `(lbl, origin, h)` and rebuild the smoothed directional contact-mean
matrix μ_{i→j} for one week, once per posterior draw:

    ρ_diag = exp(softclamp(log_rho_diag, log3, log45)),  ρ_gap = exp(softclamp(log_rho_gap, log3, log45))
    η = exp(softclamp(log_eta, -3, 2))  (mirrors model)
    u = (mid_p1+mid_p2)/√2 (total age),  v = (mid_p1-mid_p2)/√2 (age gap)        # 45° rotation
    Kp[p,q] = exp(-((u_p-u_q)²/(2ρ_diag²) + (v_p-v_q)²/(2ρ_gap²)))  over the 28 unordered pairs
    Lp = chol(Kp + 1e-6 I).L
    μ[i,j] = exp(softclamp(rvec[pair_index[i,j]] + log(pop_j / pop_ref), -8, 6))   (pop_ref = pop[1], "2-10")

with the per-week rate `rvec` built for the requested `week` (`wk`):

- **Separable spatio-temporal regime** (`log_rho_time` present; the cached `contacts="temporal"`
  chains): scalar intercept `c`, temporal-level GP and matrix-normal structure field share the
  temporal Cholesky `Lt` built from `ρ_time = exp(softclamp(log_rho_time, log0.5, log26))` over the
  full `Tn` window weeks. `Lt[wk,:]` (= column `wk` of `Ltᵀ`) mixes weeks `1..wk`, so the FULL field
  `z` (P×Tn) and level `z_c` (Tn) are needed, not just week `wk`:

      σ_c = exp(softclamp(log_sigma_c, -3, 2));   Kt[s,t]=exp(-(s-t)²/(2ρ_time²));  Lt=chol(Kt+1e-4 I).L
      rvec = ( c + σ_c·(Lt[wk,:]·z_c) )  .+  η·( Lp · (z · Lt[wk,:]) )

- **Legacy per-week iid** (`c[t]`, `z[p,t]`): `rvec = c[wk] .+ η .* (Lp * z[:,wk])`.
- **Pooled** (scalar `c`, `z[p]`): `rvec = c .+ η .* (Lp * z)`.

`week_index` defaults to the last window week (the origin week the forecast NGM is frozen at).
Returns `nothing` when the chain file is missing.
"""
function reconstruct_mu_draws(lbl::AbstractString, origin::Date, h::Integer;
                              week_index::Union{Int,Nothing} = nothing,
                              grid,
                              contacts::AbstractString = "temporal-gsar",
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
    logpop = log.(grid.POP ./ grid.POP[1])          # relative to reference bin (index 1, "2-10")

    ρ_diag = exp.(_softclamp.(vec(Array(chn[:log_rho_diag])), log(3.0), log(45.0)))   # soft-bounded, mirrors model
    ρ_gap  = exp.(_softclamp.(vec(Array(chn[:log_rho_gap])),  log(3.0), log(45.0)))
    η = exp.(_softclamp.(vec(Array(chn[:log_eta])), -3.0, 2.0))
    D = length(ρ_diag)
    # rotated (diagonal / anti-diagonal) coordinates for the 28 pairs, √2-normalised (mirrors model)
    su = [(mid[p[1]] + mid[p[2]]) / sqrt(2) for p in pair_list]   # along-diagonal (total age)
    df = [(mid[p[1]] - mid[p[2]]) / sqrt(2) for p in pair_list]   # across-diagonal (age gap)

    pnames = string.(names(chn, :parameters))
    # spatial Cholesky Lp for draw d (mirrors model) — shared by all regimes
    _Lp(d) = cholesky(Symmetric(
        [exp(-((su[m] - su[n])^2 / (2 * ρ_diag[d]^2) + (df[m] - df[n])^2 / (2 * ρ_gap[d]^2)))
         for m in 1:P, n in 1:P]) + 1e-6 * I).L
    μ = Array{Float64,3}(undef, D, A, A)

    if any(n -> n == "log_rho_time", pnames)                 # separable spatio-temporal regime
        # infer Tn from the structure-field names z[p,t] (z_c[t]/z_s/z_i don't match "^z\[")
        Tn = maximum(parse(Int, match(r"^z\[\d+\s*,\s*(\d+)\]$", n).captures[1])
                     for n in pnames if occursin(r"^z\[\d+\s*,\s*\d+\]$", n))
        wk = week_index === nothing ? Tn : week_index
        cc     = vec(Array(chn[:c]))                                                    # D scalar intercept
        σ_c    = exp.(_softclamp.(vec(Array(chn[:log_sigma_c])), -3.0, 2.0))            # D
        ρ_time = exp.(_softclamp.(vec(Array(chn[:log_rho_time])), log(0.5), log(26.0)))  # D
        Z  = Array{Float64,3}(undef, D, P, Tn)               # structure-field raw z[p,t]
        for n in pnames
            m = match(r"^z\[(\d+)\s*,\s*(\d+)\]$", n); m === nothing && continue
            Z[:, parse(Int, m.captures[1]), parse(Int, m.captures[2])] = vec(Array(chn[Symbol(n)]))
        end
        Zc = Matrix{Float64}(undef, D, Tn)                   # temporal-level raw z_c[t]
        for n in pnames
            m = match(r"^z_c\[(\d+)\]$", n); m === nothing && continue
            Zc[:, parse(Int, m.captures[1])] = vec(Array(chn[Symbol(n)]))
        end
        for d in 1:D
            Kt = [exp(-((s - t)^2) / (2 * ρ_time[d]^2)) for s in 1:Tn, t in 1:Tn]
            Lt = cholesky(Symmetric(Kt) + 1e-4 * I).L
            ltrow = Lt[wk, :]                                # row wk of Lt = column wk of Ltᵀ
            c_wk = cc[d] + σ_c[d] * dot(ltrow, @view Zc[d, :])
            rvec = c_wk .+ η[d] .* (_Lp(d) * (@view(Z[d, :, :]) * ltrow))    # cₜ + η·(Lp·(z·Lt[wk,:]))
            for i in 1:A, j in 1:A
                μ[d, i, j] = exp(_softclamp(rvec[pair_index[i, j]] + logpop[j], -8.0, 6.0))
            end
        end
        return μ
    end

    # --- legacy per-week iid (c[t], z[p,t]) or pooled (scalar c, z[p]): single P-vector field ---
    if any(n -> occursin(r"^c\[\d+\]$", n), pnames)          # per-week iid: c[t], z[p,t]
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

    for d in 1:D
        rvec = c_t[d] .+ η[d] .* (_Lp(d) * @view z_t[d, :])
        for i in 1:A, j in 1:A
            μ[d, i, j] = exp(_softclamp(rvec[pair_index[i, j]] + logpop[j], -8.0, 6.0))
        end
    end
    return μ
end

"""
    reconstruct_dispersion_draws(lbl, origin, h; weighted, week_index=nothing,
                                 contacts, save_dir) -> ndraws × 4  |  nothing

Load the cached chain for `(lbl, origin, h)` and rebuild the per-cell degree-model dispersion
(one value per block-linear index `bl = 2(block_of(i)−1) + block_of(j) ∈ {1,2,3,4}`) for one
week, once per posterior draw — the companion to `reconstruct_mu_draws` (same chain, same draw
order, so column `d` pairs with μ's draw `d`).

`weighted` selects the parameter and its soft-clamp bounds, mirroring `_cell_moments!`
(joint_model.jl):
- Weibull (`weighted=true`):  `κ = exp(softclamp(log_kappa, −3, 3))`
- NegBin  (`weighted=false`): `k = exp(softclamp(log_k,     −4, 5))`

Handles the per-week regime (`log_k[bl,t]` / `log_kappa[bl,t]`, the cached `contacts="temporal-gsar"`
chains — dispersion stays per-week × block, so this is unchanged by the spatio-temporal GP;
`week_index` defaults to the last window week, the origin week the NGM is frozen at) and the pooled
regime (`2×2` block matrix `log_k[bi,bj]`, mapped to `bl`). Returns `nothing` when the chain file is
missing. (Dispersion is block-linear only — the hierarchical per-age-pair RE was reverted 2026-07-11.)
"""
function reconstruct_dispersion_draws(lbl::AbstractString, origin::Date, h::Integer;
                                      weighted::Bool,
                                      week_index::Union{Int,Nothing} = nothing,
                                      contacts::AbstractString = "temporal-gsar",
                                      save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    path = chain_path(lbl, origin, h; contacts = contacts, save_dir = save_dir)
    isfile(path) || return nothing
    chn = try
        load(path, "result")
    catch err
        @warn "could not load chain" path err
        return nothing
    end

    base    = weighted ? "log_kappa" : "log_k"
    lo, hi  = weighted ? (-3.0, 3.0) : (-4.0, 5.0)        # soft-clamp bounds mirror _cell_moments!
    pnames  = string.(names(chn, :parameters))
    perweek = Regex("^" * base * raw"\[(\d+)\s*,\s*(\d+)\]$")
    D = length(vec(Array(chn[Symbol("log_eta")])))         # draw count (shared with μ reconstruction)
    disp = Matrix{Float64}(undef, D, 4)

    # collect (row, col, exact-name) for every disp param. Read via the EXACT stored name — MCMCChains
    # prints matrix indices as "log_k[1, 2]" (space after the comma), so a rebuilt "log_k[1,2]" misses.
    entries = [(parse(Int, m.captures[1]), parse(Int, m.captures[2]), n)
               for n in pnames for m in (match(perweek, n),) if m !== nothing]
    isempty(entries) && (@warn "no $base parameters in chain" path; return nothing)
    read_col!(bl, name) = (disp[:, bl] = exp.(_softclamp.(vec(Array(chn[Symbol(name)])), lo, hi)))

    if maximum(e[1] for e in entries) == 4                 # per-week: row = block-linear bl, col = week
        wk = week_index === nothing ? maximum(e[2] for e in entries) : week_index
        for (bl, t, name) in entries
            t == wk && read_col!(bl, name)
        end
    else                                                   # pooled: 2×2 block matrix log_*[bi,bj]
        for (bi, bj, name) in entries
            read_col!(2 * (bi - 1) + bj, name)
        end
    end
    return disp
end
