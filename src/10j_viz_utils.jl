# 10j_viz_utils.jl — read-only helper for the 10j single-time-point diagnostics.
#
# Reconstructs the GP-smoothed directional contact mean μ_{i→j} per posterior draw from a
# cached STAGE-1 chain, WITHOUT rebuilding the model — mirroring the `load_transmission_draws`
# pattern in 8j_viz_utils.jl. μ is a deterministic transform of the raw sampled columns
# (log_rho_diag, log_rho_gap, log_eta; and, for the separable spatio-temporal regime, log_rho_time,
# log_sigma_c, scalar level c, temporal-level raw z_c, structure-field raw z[·,·]); see `model_degree`
# (joint_model.jl §5/§6). Requires 8j_viz_utils.jl (for `stage1_chain_path`) to be included first.
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
                              contacts::AbstractString = "temporal-gsar-cut-sc-hd-p0-gi",
                              save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    path = stage1_chain_path(lbl, origin, h; contacts = contacts, save_dir = save_dir)
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
    reconstruct_dispersion_draws(lbl, origin, h; weighted, cfg, grid, week_index=nothing,
                                 contacts, save_dir) -> ndraws × A × A  |  nothing

Load the cached Stage-1 chain for `(lbl, origin, h)` and rebuild the **per-cell** degree-model
dispersion for one week, once per posterior draw — the companion to `reconstruct_mu_draws` (same
chain, same draw order, so slice `d` pairs with μ's draw `d`).

Mirrors the HIERARCHICAL construction of `_cell_moments!` (joint_model.jl §4.3) exactly — this
reconstruct-matches-model invariant is the standing rule for every viz mirror here:

    log_disp[i,j] = β[bl] + τ·z[pcode],   bl = 2(block_of(i)−1)+block_of(j),  pcode = (i−1)A+j
    disp[i,j]     = exp(softclamp(log_disp[i,j], lo, hi))

`weighted` selects the parameter family and its soft-clamp bounds:
- Weibull (`weighted=true`):  `κ = exp(softclamp(·, −5, 5))`, from `log_kappa` + `z_kappa`
- NegBin  (`weighted=false`): `k = exp(softclamp(·, −4, 5))`, from `log_k` + `z_k`

τ is the RE scale: `tau[wk]` per week (shared across blocks), or a scalar `tau` when pooled.
Returns `ndraws × A × A` (was `ndraws × 4` while dispersion was block-only, pre-2026-07-30), so
callers index `[:, i, j]`, not `[:, bl]`. `week_index` defaults to the last window week (the origin
week the NGM is frozen at). Returns `nothing` when the chain file is missing.
"""
function reconstruct_dispersion_draws(lbl::AbstractString, origin::Date, h::Integer;
                                      weighted::Bool, cfg, grid,
                                      week_index::Union{Int,Nothing} = nothing,
                                      contacts::AbstractString = "temporal-gsar-cut-sc-hd-p0-gi",
                                      save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    path = stage1_chain_path(lbl, origin, h; contacts = contacts, save_dir = save_dir)
    isfile(path) || return nothing
    chn = try
        load(path, "result")
    catch err
        @warn "could not load chain" path err
        return nothing
    end

    A      = grid.N
    bbase  = weighted ? "log_kappa" : "log_k"              # block-MEAN param
    zbase  = weighted ? "z_kappa"   : "z_k"                # per-ordered-cell random term
    lo, hi = weighted ? (-4.3, 5.0) : (-4.0, 5.0)          # soft-clamp bounds MIRROR _cell_moments!
                                                           # (κ widened -3,3 → -4.3,5 on 2026-07-30;
                                                           #  −4.45 is the hard floor, see there.
                                                           #  Keep these two in lockstep or the
                                                           #  reconstruct-matches-model check fails)
    pnames = string.(names(chn, :parameters))
    D = size(chn, 1) * size(chn, 3)                        # posterior draws (iter × chains)

    # --- block means β: D × 4 (block-linear bl). Read via the EXACT stored name — MCMCChains prints
    # matrix indices as "log_k[1, 2]" (SPACE after the comma), so a rebuilt "log_k[1,2]" would miss. ---
    β  = Matrix{Float64}(undef, D, 4)
    bw = Regex("^" * bbase * raw"\[(\d+)\s*,\s*(\d+)\]$")
    bentries = [(parse(Int, m.captures[1]), parse(Int, m.captures[2]), n)
                for n in pnames for m in (match(bw, n),) if m !== nothing]
    isempty(bentries) && (@warn "no $bbase parameters in chain" path; return nothing)
    if maximum(e[1] for e in bentries) == 4                # per-week: [bl, t]
        wk = week_index === nothing ? maximum(e[2] for e in bentries) : week_index
        for (bl, t, name) in bentries
            t == wk && (β[:, bl] = vec(Array(chn[Symbol(name)])))
        end
    else                                                   # pooled: 2×2 [row, col] → COLUMN-major
        wk = nothing                                       # bl = row + 2(col−1), matching `vec(log_*)`
        for (r, c, name) in bentries
            β[:, r + 2 * (c - 1)] = vec(Array(chn[Symbol(name)]))
        end
    end

    # --- RE scale τ: per-week `tau ~ filldist(HalfNormal, Tn)` ⇒ stored "tau[t]" (select week `wk`);
    # pooled draws a scalar `tau`. Resolved AFTER `wk` is known — the two regimes store it differently. ---
    τ = wk === nothing ? vec(Array(chn[:tau])) : vec(Array(chn[Symbol("tau[$wk]")]))

    # --- per-ordered-cell random term z: D × A² ---
    Z = Matrix{Float64}(undef, D, A * A)
    if wk === nothing                                      # pooled: 1-D z[p]
        zw1 = Regex("^" * zbase * raw"\[(\d+)\]$")
        for n in pnames
            m = match(zw1, n); m === nothing && continue
            Z[:, parse(Int, m.captures[1])] = vec(Array(chn[Symbol(n)]))
        end
    else                                                   # per-week: z[p, wk]
        zw = Regex("^" * zbase * raw"\[(\d+)\s*,\s*(\d+)\]$")
        for n in pnames
            m = match(zw, n); m === nothing && continue
            parse(Int, m.captures[2]) == wk &&
                (Z[:, parse(Int, m.captures[1])] = vec(Array(chn[Symbol(n)])))
        end
    end

    # --- per cell: log_disp = β[bl] + τ·z[pcode], then exp∘softclamp (mirrors _cell_moments!) ---
    disp = Array{Float64,3}(undef, D, A, A)
    for i in 1:A, j in 1:A
        bl    = 2 * (block_of(i, cfg) - 1) + block_of(j, cfg)
        pcode = (i - 1) * A + j
        for d in 1:D
            disp[d, i, j] = exp(_softclamp(β[d, bl] + τ[d] * Z[d, pcode], lo, hi))
        end
    end
    return disp
end

"""
    reconstruct_p0_draws(lbl, origin, h; grid, week_index=nothing, contacts, save_dir)
        -> ndraws × A × A  |  nothing

Fitted hurdle zero probability `p⁰_{ij}` per draw for one week, from a **weighted-path** Stage-1
chain (`p0f[pcode, t]`, §4.2). Returns `nothing` for a NegBin chain — that path has no `p0f` (it
models its zeros directly), which is also the cheapest way to tell the two chain shapes apart.

Companion diagnostic: compare against the empirical `n_zero/n` (`AgePairData.p0`) — they should
track closely wherever the roster count is large, which is the check that the Binomial in
`_cell_moments!` is wired to the right denominator.
"""
function reconstruct_p0_draws(lbl::AbstractString, origin::Date, h::Integer; grid,
                              week_index::Union{Int,Nothing} = nothing,
                              contacts::AbstractString = "temporal-gsar-cut-sc-hd-p0-gi",
                              save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    path = stage1_chain_path(lbl, origin, h; contacts = contacts, save_dir = save_dir)
    isfile(path) || return nothing
    chn = try
        load(path, "result")
    catch err
        @warn "could not load chain" path err
        return nothing
    end
    A = grid.N
    pnames = string.(names(chn, :parameters))
    D = size(chn, 1) * size(chn, 3)
    pw = r"^p0f\[(\d+)\s*,\s*(\d+)\]$"
    entries = [(parse(Int, m.captures[1]), parse(Int, m.captures[2]), n)
               for n in pnames for m in (match(pw, n),) if m !== nothing]
    if isempty(entries)                                    # pooled 1-D p0f[p], or a NegBin chain
        p1 = [(parse(Int, m.captures[1]), n)
              for n in pnames for m in (match(r"^p0f\[(\d+)\]$", n),) if m !== nothing]
        isempty(p1) && return nothing                      # NegBin path: no p0f by design
        out = Array{Float64,3}(undef, D, A, A)
        for (p, name) in p1
            i, j = fldmod1(p, A)                           # pcode = (i−1)A + j
            out[:, i, j] = vec(Array(chn[Symbol(name)]))
        end
        return out
    end
    wk = week_index === nothing ? maximum(e[2] for e in entries) : week_index
    out = Array{Float64,3}(undef, D, A, A)
    for (p, t, name) in entries
        t == wk || continue
        i, j = fldmod1(p, A)
        out[:, i, j] = vec(Array(chn[Symbol(name)]))
    end
    return out
end

"""
    reconstruct_tau_draws(lbl, origin, h; contacts, save_dir) -> ndraws × Tn  |  nothing

The dispersion random-effect scale τ_t for EVERY window week (`tau[t]`, one per week, shared
across the four child/adult blocks — §4.3). Half-Normal, so it is stored untransformed: no
exp/softclamp mirror needed here, unlike `reconstruct_dispersion_draws`.
Returns `nothing` if the chain is missing, or a `ndraws × 1` matrix for a pooled chain (scalar τ).
"""
function reconstruct_tau_draws(lbl::AbstractString, origin::Date, h::Integer;
                               contacts::AbstractString = "temporal-gsar-cut-sc-hd-p0-gi",
                               save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    path = stage1_chain_path(lbl, origin, h; contacts = contacts, save_dir = save_dir)
    isfile(path) || return nothing
    chn = try
        load(path, "result")
    catch err
        @warn "could not load chain" path err
        return nothing
    end
    pnames = string.(names(chn, :parameters))
    ent = [(parse(Int, m.captures[1]), n)
           for n in pnames for m in (match(r"^tau\[(\d+)\]$", n),) if m !== nothing]
    isempty(ent) && return reshape(vec(Array(chn[:tau])), :, 1)     # pooled: scalar τ
    Tn = maximum(first.(ent))
    D  = size(chn, 1) * size(chn, 3)
    out = Matrix{Float64}(undef, D, Tn)
    for (t, name) in ent
        out[:, t] = vec(Array(chn[Symbol(name)]))
    end
    return out
end

"""
    plot_tau_over_weeks(lbl, origin, cfg, weeks; h=1, save_dir) -> Plots.Plot | nothing

Posterior median + 90% band of the dispersion RE scale τ_t across the window weeks, against its
`N⁺(0, cfg.disp_re_scale_prior[2])` prior band (grey).

Read it as the "is the hierarchy earning its place?" check: **τ_t hugging the prior across all
weeks means the data are not informing the per-cell spread** — the random term is then just prior
noise flowing into ⟨k²⟩ (and amplified by the neighbourhood NGM). The documented fallbacks are a
tighter prior scale (0.109, the 2026-07-11 value) or a single scalar τ for the whole window.
τ_t → 0 recovers the block-only dispersion exactly.
"""
function plot_tau_over_weeks(lbl::AbstractString, origin::Date, cfg, weeks;
                             h::Integer = 1,
                             save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    τ = reconstruct_tau_draws(lbl, origin, h; contacts = contacts_label(cfg), save_dir = save_dir)
    τ === nothing && (@warn "no Stage-1 chain for τ panel" lbl origin; return nothing)
    Tn  = size(τ, 2)
    wks = length(weeks) == Tn ? collect(weeks) : collect(1:Tn)
    # `view(...)` function form, NOT the space-form `@view`: in an argument list with further args
    # the macro greedily swallows them (`quantile(@view τ[:, t], 0.05)` → "Invalid use of @view
    # macro"). Same trap the model's `_cell_moments!` call site documents.
    med = [median(view(τ, :, t)) for t in 1:Tn]
    lo  = [quantile(view(τ, :, t), 0.05) for t in 1:Tn]
    hi  = [quantile(view(τ, :, t), 0.95) for t in 1:Tn]
    # half-Normal N⁺(0,σ) prior quantiles: q(p) = σ·Φ⁻¹((1+p)/2)
    σp  = cfg.disp_re_scale_prior[2]
    pri = (med = σp * 0.6744897501960817, lo = σp * 0.06270677794321385, hi = σp * 1.959963984540054)
    p = plot(; title = "$lbl — dispersion RE scale τ_t (origin $origin, h=$h)", titlefontsize = 9,
             xlabel = "window week", ylabel = "τ  (log-scale SD of the per-cell RE)",
             legend = :topright, legendfontsize = 6, xrotation = 45, ylims = (0, max(σp * 2.5, maximum(hi) * 1.1)))
    # Date-valued series FIRST — a leading hline!/hspan! locks a numeric axis and mangles date ticks.
    plot!(p, wks, med; lw = 2, marker = :circle, ms = 3, markerstrokewidth = 0,
          ribbon = (med .- lo, hi .- med), fillalpha = 0.18, label = "posterior median & 90%")
    plot!(p, wks, fill(pri.med, Tn); lw = 1.2, ls = :dot, color = :grey40, label = "prior median & 90%",
          ribbon = (fill(pri.med - pri.lo, Tn), fill(pri.hi - pri.med, Tn)),
          fillalpha = 0.08, fillcolor = :grey60)
    return p
end

"""
    plot_dispersion_cells(lbl, origin, cfg, grid; weighted, h=1, week_index=nothing, save_dir)
        -> Plots.Plot | nothing

7×7 heatmap of the per-cell dispersion (posterior median of `κ_{ij}` / `φ_{ij}`) at one week, with
the child/adult **block boundary** drawn on top.

This is the direct picture of what the 2026-07-30 hierarchy bought: before it, every cell inside a
block was **identical** by construction, so each of the four block quadrants would be one flat
colour. Visible within-quadrant variation is the per-cell random term `τ_t·z_{ij,t}` doing work;
four flat quadrants mean τ_t has been shrunk to ~0 and the model has collapsed back to block-only
(cross-check with `plot_tau_over_weeks`).
"""
function plot_dispersion_cells(lbl::AbstractString, origin::Date, cfg, grid;
                               weighted::Bool, h::Integer = 1,
                               week_index::Union{Int,Nothing} = nothing,
                               save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    D = reconstruct_dispersion_draws(lbl, origin, h; weighted = weighted, cfg = cfg, grid = grid,
                                     week_index = week_index, contacts = contacts_label(cfg),
                                     save_dir = save_dir)
    D === nothing && return nothing
    A = grid.N
    med = [median(view(D, :, i, j)) for i in 1:A, j in 1:A]
    sym = weighted ? "κ" : "φ"
    p = heatmap(1:A, 1:A, med; c = :viridis, yflip = true,
                xticks = (1:A, grid.LAB), yticks = (1:A, grid.LAB), xrotation = 45,
                xlabel = "contactee age group j", ylabel = "participant age group i",
                title = "$lbl — per-cell dispersion $sym (median, week $(week_index === nothing ? "last" : week_index))",
                titlefontsize = 9, tickfontsize = 6)
    # child/adult block boundary (cfg.child_bins splits both axes)
    b = cfg.child_bins + 0.5
    plot!(p, [b, b], [0.5, A + 0.5]; lw = 2, color = :white, label = "")
    plot!(p, [0.5, A + 0.5], [b, b]; lw = 2, color = :white, label = "")
    return p
end

"""
    plot_p0_vs_empirical(lbl, origin, apd, cfg, grid; h=1, week_index=nothing, save_dir)
        -> Plots.Plot | nothing

Fitted hurdle zero probability `p⁰` against the empirical `n_zero/n`, one point per age-pair cell,
sized by roster count `n`. **Weighted (hurdle-Weibull) path only** — returns `nothing` for a NegBin
chain, which has no `p0f` by design.

This is the wiring check for the Binomial in `_cell_moments!`: with a flat `Beta(1,1)` and a large
`n`, the posterior mean must sit essentially on the empirical ratio. Systematic departure at large
`n` means the denominator is wrong (e.g. `whist_nobs` vs roster mismatch). Small-`n` cells legitimately
shrink toward 0.5, and all-zero cells legitimately land just BELOW 1 rather than on it — that is the
change that stops `base_contact`'s `k1>0` guard from firing.
"""
function plot_p0_vs_empirical(lbl::AbstractString, origin::Date, apd, cfg, grid;
                              h::Integer = 1, week_index::Union{Int,Nothing} = nothing,
                              save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    P = reconstruct_p0_draws(lbl, origin, h; grid = grid, week_index = week_index,
                             contacts = contacts_label(cfg), save_dir = save_dir)
    P === nothing && return nothing                    # NegBin path (no p0f) or missing chain
    A = grid.N
    wk = week_index === nothing ? length(apd.weeks) : week_index
    fitted = [median(view(P, :, i, j)) for i in 1:A, j in 1:A]
    emp    = [apd.p0[wk, i, j] for i in 1:A, j in 1:A]
    nn     = [apd.n[wk, i, j]  for i in 1:A, j in 1:A]
    keep   = vec(nn) .> 0
    p = plot(; title = "$lbl — fitted p⁰ vs empirical n₀/n (origin $origin, week $wk)",
             titlefontsize = 9, xlabel = "empirical n₀/n", ylabel = "posterior median p⁰",
             legend = :bottomright, legendfontsize = 6, xlims = (-0.02, 1.02), ylims = (-0.02, 1.02))
    plot!(p, [0, 1], [0, 1]; lw = 1, ls = :dash, color = :grey50, label = "y = x")
    scatter!(p, vec(emp)[keep], vec(fitted)[keep];
             ms = 2 .+ 4 .* sqrt.(vec(nn)[keep] ./ maximum(nn)), markerstrokewidth = 0,
             alpha = 0.65, label = "age-pair cells (size ∝ √n)")
    return p
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure builders for the 10j diagnostics notebook (moved out of the notebook so
# the notebook keeps only config + data prep + calls + display). All read-only:
# they reload cached joint-model chains and never re-fit. Requires the full
# forecast preamble (forecast_utils.jl) plus 8j_viz_utils.jl (stage1_chain_path / stage2_pooled_path).
#
# Most origin-week helpers take a small context bundle assembled once in the
# notebook:  oc = (; apd, t_o, t_o_est, origin, grid, cfg)
#   apd     — AgePairData for the origin window (last week == origin)
#   t_o     — origin week index (last of apd.weeks); observed overlays read column t_o
#   t_o_est — the h=1 chain's GP column for t₀ (= t_o − 1, contacts observed h wks ahead)
#   origin  — the forecast origin Date; grid — CIS age grid; cfg — FrameworkConfig
# ─────────────────────────────────────────────────────────────────────────────

# participant split (upper young / lower older) + per-participant colours for the
# §2 age-pair panels — 7-CIS-bin specific; overridable via the figure helpers.
const _PART_ROWS = ((1, 2, 3, 4), (5, 6, 7))
const _ROW_TITLE = ("participants 2-34", "participants 35+")
const _PART_COLS = [:steelblue, :darkorange, :seagreen, :purple, :crimson, :goldenrod, :teal]

"""
    _observed_cell_mean(apd, t, i, j, weighted) -> Float64

Empirical contact mean of age-pair cell `(i→j)` in week `t`, matched to the degree family's
scale: positive duration-weighted mean (`whist_mean(pos_weight)`, `NaN` if the cell is empty)
when `weighted`, else the per-capita count mean (`emp_mean`).
"""
function _observed_cell_mean(apd, t::Integer, i::Integer, j::Integer, weighted::Bool)
    if weighted
        pw = apd.pos_weight[t, i, j]
        return isempty(pw) ? NaN : whist_mean(pw)
    else
        return apd.emp_mean[t, i, j]
    end
end

"""
    fit_window_infection_draws(dm, nb, apd_o, wd, cfg, origin; h=1, save_dir)
        -> A × n_fit × draws  |  nothing

In-sample expected (fitted) infections over the fit window, from the two-stage artefacts of the
horizon-`h` fit: the MEAN of Stage 2's Normal infection likelihood (`model_transmission`). Per
pooled draw `d` (from Stage-1 draw `m = post_index[d]`),
`pred_t = build_ngm(Cstar_m[t], susc[d], inf[d], F[d], wd.antibody[:,t]; gamma_sar[d]) · Σ_s w[s]·wd.I_mean[:,t-s]`
using OBSERVED lags ⇒ one-step-ahead fitted mean (NOT the self-iterated forecast). Columns
`(smax+1):Tn` == the window's fit weeks. The per-week `Cstar_m` is rebuilt from the Stage-1 chain
(`stage1_moment_draws` → `contact_star(nb, …)`), in the SAME draw order the pooling used, so
`post_index` aligns. `apd_o[h]` is the h-window degree data (matches the cached Stage-1 chain).
Read-only (NO re-fit). Returns `nothing` when either artefact is missing/unloadable.
"""
function fit_window_infection_draws(dm::ContactDegreeModel, nb::NGMBuilder,
                                    apd_o, wd, cfg, origin::Date;
                                    h::Int = 1,
                                    save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    lbl = string(degree_label(dm), "|", ngm_label(nb))
    tag = contacts_label(cfg)
    s1p = stage1_chain_path(lbl, origin, h; contacts = tag, save_dir = save_dir)
    s2p = stage2_pooled_path(lbl, origin, h; contacts = tag, save_dir = save_dir)
    (isfile(s1p) && isfile(s2p)) || (@warn "no two-stage artefacts for fit-window fit" lbl; return nothing)
    ds   = build_degree_stats(dm, apd_o[h], cfg)              # h-window degree stats (matches cached chain)
    local md, pooled
    try
        s1chn  = load(s1p, "result")
        md     = stage1_moment_draws(dm, ds, wd.pop, cfg, s1chn; n_post = cfg.n_stage1_post)
        pooled = load(s2p, "pooled")
    catch err
        @warn "could not load two-stage artefacts for fit-window fit" lbl err; return nothing
    end
    A = wd.A; Tn = length(wd.weeks); fitcols = (cfg.smax + 1):Tn
    # per-Stage-1-draw per-week C* (nb applied) — reused across that draw's pooled infection draws.
    Cstar_by_m = [[contact_star(nb, md[m].K1[t], md[m].K2[t], md[m].G[t]) for t in 1:Tn]
                  for m in eachindex(md)]
    Np = length(pooled.gamma_sar)
    out = Array{Float64}(undef, A, length(fitcols), Np)
    for d in 1:Np
        m = pooled.post_index[d]
        # PER-DRAW generation interval (2026-07-30, §3.1): the GI is estimated in Stage 2, so
        # each pooled draw carries its own (w_mu, w_sigma). Rebuilding it from cfg here would
        # silently use the PRIOR CENTRE while the fit used the posterior. The stored values are
        # post-clamp, so this reproduces the model's `w` exactly.
        w_d = gen_interval_pmf_log(pooled.w_mu[d], pooled.w_sigma[d]; smax = cfg.smax)
        for (c, t) in enumerate(fitcols)
            N = build_ngm(Cstar_by_m[m][t], pooled.susc[d, :], pooled.inf[d, :],
                          pooled.F[d], wd.antibody[:, t]; gamma_sar = pooled.gamma_sar[d])
            out[:, c, d] = renewal_next(N, wd.I_mean, t, w_d)
        end
    end
    return out
end

"""
    agepair_panel(parts, ttl, μdraws, weighted, oc; part_cols=_PART_COLS) -> Plots.Plot

One §2 panel: overlay every participant in `parts` — GP-smoothed μ (median line + 90% ribbon,
from `μdraws` = ndraws×A×A) and the matching observed mean (× markers, `_observed_cell_mean` at
`oc.t_o`), coloured by participant; x-axis = contactee age group.
"""
function agepair_panel(parts, ttl::AbstractString, μdraws, weighted::Bool, oc;
                       part_cols = _PART_COLS)
    grid = oc.grid
    pnl = plot(; title = "$(ttl)  (lines = smoothed μ, 90%;  × = observed)", titlefontsize = 8,
               xticks = (1:grid.N, grid.LAB), xrotation = 45,
               xlabel = "contactee age group", ylabel = "contact mean μ",
               legend = :topright, legendfontsize = 6, legendtitle = "participant",
               legendtitlefontsize = 6)
    for i in parts
        obs = [_observed_cell_mean(oc.apd, oc.t_o, i, j, weighted) for j in 1:grid.N]
        med = [median(μdraws[:, i, j]) for j in 1:grid.N]
        lo  = [quantile(μdraws[:, i, j], 0.05) for j in 1:grid.N]
        hi  = [quantile(μdraws[:, i, j], 0.95) for j in 1:grid.N]
        c = part_cols[i]
        plot!(pnl, 1:grid.N, med; ribbon = (med .- lo, hi .- med), color = c, lw = 2,
              marker = :circle, ms = 3, fillalpha = 0.08, label = grid.LAB[i])
        scatter!(pnl, 1:grid.N, obs; color = c, marker = :x, ms = 5, msw = 2, label = "")
    end
    return pnl
end

"""
    make_agepair_fig(dm, nb, oc; part_rows=_PART_ROWS, row_titles=_ROW_TITLE, res_dir="../res")
        -> Plots.Plot | nothing

§2 — the 2×1 (upper young / lower older) observed-vs-smoothed-μ figure for one model. μ is
reconstructed at the origin week (`oc.t_o_est` column of the h=1 chain). Saves to
`res_dir/10j_agepair_mean_{negbin|hweibull}_<origin>.png`. `nothing` if the chain is missing.
"""
function make_agepair_fig(dm::ContactDegreeModel, nb::NGMBuilder, oc;
                          part_rows = _PART_ROWS, row_titles = _ROW_TITLE,
                          res_dir::AbstractString = "../res")
    grid = oc.grid
    lbl  = string(degree_label(dm), "|", ngm_label(nb))
    μdraws = reconstruct_mu_draws(lbl, oc.origin, 1; week_index = oc.t_o_est, grid = grid)
    μdraws === nothing && (@warn "no chain for $lbl @ $(oc.origin)"; return nothing)
    weighted = is_weighted(dm)
    panels = [agepair_panel(part_rows[r], row_titles[r], μdraws, weighted, oc)
              for r in 1:length(part_rows)]
    scale_note = weighted ? "positive duration-weighted degree" : "per-capita count"
    fig = plot(panels...; layout = (2, 1), size = (760, 820),
               left_margin = 10Plots.mm, bottom_margin = 12Plots.mm,   # room for x-/y-labels
               plot_title = "$(lbl) — observed vs smoothed μ ($(scale_note)), origin $(oc.origin)",
               plot_titlefontsize = 10)
    fname = weighted ? "10j_agepair_mean_hweibull_$(oc.origin).png" :
                       "10j_agepair_mean_negbin_$(oc.origin).png"
    savefig(fig, joinpath(res_dir, fname))
    return fig
end

"""
    make_mu_horizon_fig(dm, cells, tag, title_desc, apd_o, origin, grid, cfg, labels4,
                        model_cols; hz=collect(cfg.horizons), res_dir="../res") -> Plots.Plot

§2b — contact mean μ across horizons h1..h4 for one degree family, one panel per `cell`
`(i, j, panel_title)` (participant `i` → contactee `j`). The two lines per panel are the
mean- vs neighbourhood-NGM fits (coloured via `labels4`/`model_cols`); × = observed at the
forecast week t₀+h (`apd_o[k]`, whose last window week is t₀+h). μ for horizon `h` is read at
that horizon's own chain default (last) week — the C*-slice the forecast is frozen at.

Unifies the participant-row (`cells = [(PART_I, j, …) for j …]`, `tag="vs_horizon"`) and the
diagonal self-contact (`cells = [(i, i, …) for i …]`, `tag="diag_vs_horizon"`) figures. Saves to
`res_dir/10j_agepair_mu_<tag>_<degree>_<origin>.png`.
"""
function make_mu_horizon_fig(dm::ContactDegreeModel, cells, tag::AbstractString,
                             title_desc::AbstractString, apd_o, origin::Date, grid, cfg,
                             labels4, model_cols;
                             hz = collect(cfg.horizons), res_dir::AbstractString = "../res")
    weighted = is_weighted(dm)
    ncell = length(cells)

    # μ stats per NGM builder: reconstruct each (lbl, h) chain ONCE, slice each cell.
    stats = Dict{String,NTuple{3,Matrix{Float64}}}()          # ngm_label => (med, lo, hi), each |hz|×ncell
    for nb in (MeanNGM(), NeighbourhoodDegreeNGM())
        lbl = string(degree_label(dm), "|", ngm_label(nb))
        med = fill(NaN, length(hz), ncell); lo = copy(med); hi = copy(med)
        for (k, h) in enumerate(hz)
            μd = reconstruct_mu_draws(lbl, origin, h; grid = grid)   # default week = forecast week t₀+h
            μd === nothing && (@warn "no chain for $lbl @ $origin h$h"; continue)
            for (p, (ci_, cj_, _)) in enumerate(cells)
                col = @view μd[:, ci_, cj_]
                med[k, p] = median(col); lo[k, p] = quantile(col, 0.05); hi[k, p] = quantile(col, 0.95)
            end
        end
        stats[ngm_label(nb)] = (med, lo, hi)
    end

    # observed μ at each forecast week t₀+h (matched to the family's scale).
    obs = fill(NaN, length(hz), ncell)
    for (k, h) in enumerate(hz)
        apd = apd_o[k]
        @assert apd.weeks[end] == origin + Day(7 * h)
        for (p, (ci_, cj_, _)) in enumerate(cells)
            obs[k, p] = _observed_cell_mean(apd, lastindex(apd.weeks), ci_, cj_, weighted)
        end
    end

    panels = Any[]
    for (p, (_, _, ptitle)) in enumerate(cells)
        pnl = plot(; title = ptitle, titlefontsize = 8,
                   xticks = (hz, ["h$(h)" for h in hz]), xlabel = "horizon",
                   ylabel = "contact mean μ", legend = (p == 1 ? :best : false), legendfontsize = 5)
        for nb in (MeanNGM(), NeighbourhoodDegreeNGM())
            lbl = string(degree_label(dm), "|", ngm_label(nb))
            ci  = findfirst(==(lbl), labels4)                    # colour consistent with §1/§3
            med, lo, hi = stats[ngm_label(nb)]
            m = med[:, p]; l = lo[:, p]; u = hi[:, p]
            plot!(pnl, hz, m; ribbon = (m .- l, u .- m), color = model_cols[ci], lw = 2,
                  marker = :circle, ms = 3, fillalpha = 0.10, label = ngm_label(nb))
        end
        scatter!(pnl, hz, obs[:, p]; color = :black, marker = :x, ms = 5, msw = 2, label = "observed")
        push!(panels, pnl)
    end
    push!(panels, plot(; framestyle = :none))                    # 8th blank cell fills the 2×4 grid

    scale_note = weighted ? "positive duration-weighted degree" : "per-capita count"
    fig = plot(panels...; layout = (2, 4), size = (1400, 700),
               left_margin = 6Plots.mm, bottom_margin = 9Plots.mm,
               plot_title = "$(degree_label(dm)) — $(title_desc) " *
                            "($(scale_note)); lines = mean/neighbourhood NGM (90%), × observed; origin $(origin)",
               plot_titlefontsize = 9)
    savefig(fig, joinpath(res_dir, "10j_agepair_mu_$(tag)_$(degree_label(dm))_$(origin).png"))
    return fig
end

"""
    make_mu_timeline_fig(dm, cells, tag, title_desc, apd_h, origin, grid, cfg, labels4,
                         model_cols; h_chain=maximum(cfg.horizons), res_dir="../res") -> Plots.Plot

§2c — contact mean μ over the fit window + horizons h1..h4, reconstructed from the SINGLE
horizon-`h_chain` chain (default h4), for one degree family. One panel per `cell`
`(i, j, panel_title)`. Unlike §2b (`make_mu_horizon_fig`, which reads each horizon from its OWN
chain at that chain's forecast week), this traces μ over TIME from one fit: the h4 chain's per-week
GP spans exactly the origin window's 8 fit weeks (its `all_weeks[1:8]`, ending at t₀) ++ the 4
horizon weeks h1..h4 (`all_weeks[9:12]`, ending at t₀+4), because `constant_contacts = false`
estimates μ per week. It is the μ analogue of §1's h4 in-sample fit.

`reconstruct_mu_draws(lbl, origin, h_chain; week_index = t)` reads μ at each window week `t`;
observed comes from that same window `apd_h` (= `apd_o[h_chain]`) per week (`_observed_cell_mean`).
x-axis = week (Wed mid-date); the origin t₀ is marked with a rule (fit weeks left, horizons h1..h4
right). Two lines per panel = mean- vs neighbourhood-NGM (median + 90% ribbon); × = observed. Saves
to `res_dir/10j_agepair_mu_<tag>_<degree>_<origin>.png`.
"""
function make_mu_timeline_fig(dm::ContactDegreeModel, cells, tag::AbstractString,
                              title_desc::AbstractString, apd_h, origin::Date, grid, cfg,
                              labels4, model_cols;
                              h_chain::Integer = maximum(cfg.horizons),
                              res_dir::AbstractString = "../res")
    weighted = is_weighted(dm)
    ncell = length(cells)
    weeks = apd_h.weeks                       # Tn dates: origin's 8 fit weeks ++ horizons h1..h_chain
    Tn = length(weeks)
    xdate = week_mid.(weeks)

    # μ stats per NGM builder: reconstruct the SAME (lbl, h_chain) chain at EVERY window week t.
    stats = Dict{String,NTuple{3,Matrix{Float64}}}()          # ngm_label => (med, lo, hi), each Tn×ncell
    for nb in (MeanNGM(), NeighbourhoodDegreeNGM())
        lbl = string(degree_label(dm), "|", ngm_label(nb))
        med = fill(NaN, Tn, ncell); lo = copy(med); hi = copy(med)
        for t in 1:Tn
            μd = reconstruct_mu_draws(lbl, origin, h_chain; week_index = t, grid = grid)  # h4 chain, week t
            μd === nothing && (@warn "no chain for $lbl @ $origin h$h_chain"; break)
            for (p, (ci_, cj_, _)) in enumerate(cells)
                col = @view μd[:, ci_, cj_]
                med[t, p] = median(col); lo[t, p] = quantile(col, 0.05); hi[t, p] = quantile(col, 0.95)
            end
        end
        stats[ngm_label(nb)] = (med, lo, hi)
    end

    # observed μ per week (matched to the family's scale) from the h_chain degree window.
    obs = fill(NaN, Tn, ncell)
    for t in 1:Tn, (p, (ci_, cj_, _)) in enumerate(cells)
        obs[t, p] = _observed_cell_mean(apd_h, t, ci_, cj_, weighted)
    end

    panels = Any[]
    for (p, (_, _, ptitle)) in enumerate(cells)
        pnl = plot(; title = ptitle, titlefontsize = 8, xrotation = 45,
                   xlabel = "week (Wed mid-date)", ylabel = "contact mean μ",
                   legend = (p == 1 ? :best : false), legendfontsize = 5)
        vline!(pnl, [week_mid(origin)]; color = :gray, ls = :dash, lw = 1, label = "")   # origin t₀
        for nb in (MeanNGM(), NeighbourhoodDegreeNGM())
            lbl = string(degree_label(dm), "|", ngm_label(nb))
            ci  = findfirst(==(lbl), labels4)                    # colour consistent with §1/§2b/§3
            med, lo, hi = stats[ngm_label(nb)]
            m = med[:, p]; l = lo[:, p]; u = hi[:, p]
            plot!(pnl, xdate, m; ribbon = (m .- l, u .- m), color = model_cols[ci], lw = 2,
                  marker = :circle, ms = 2, fillalpha = 0.10, label = ngm_label(nb))
        end
        scatter!(pnl, xdate, obs[:, p]; color = :black, marker = :x, ms = 4, msw = 2, label = "observed")
        push!(panels, pnl)
    end
    push!(panels, plot(; framestyle = :none))                    # 8th blank cell fills the 2×4 grid

    scale_note = weighted ? "positive duration-weighted degree" : "per-capita count"
    fig = plot(panels...; layout = (2, 4), size = (1400, 700),
               left_margin = 6Plots.mm, bottom_margin = 12Plots.mm,
               plot_title = "$(degree_label(dm)) — $(title_desc) (from h$(h_chain) chain, $(scale_note)); " *
                            "lines = mean/neighbourhood NGM (90%), × observed; origin $(origin)",
               plot_titlefontsize = 9)
    savefig(fig, joinpath(res_dir, "10j_agepair_mu_$(tag)_$(degree_label(dm))_$(origin).png"))
    return fig
end

"""
    make_contactmatrix_fig(dm, nb, oc; res_dir="../res") -> Plots.Plot | nothing

§3 — 7×7 contact-matrix heatmaps, OBSERVED vs ESTIMATED (median reconstructed μ at the origin
week `oc.t_o_est`), two panels sharing one colour scale. Observed = `_observed_cell_mean` at
`oc.t_o`. Saves to `res_dir/10j_contactmatrix_<degree>_<ngm>_<origin>.png`.
"""
function make_contactmatrix_fig(dm::ContactDegreeModel, nb::NGMBuilder, oc;
                                res_dir::AbstractString = "../res")
    grid = oc.grid; A = grid.N
    lbl  = string(degree_label(dm), "|", ngm_label(nb))
    μdraws = reconstruct_mu_draws(lbl, oc.origin, 1; week_index = oc.t_o_est, grid = grid)
    μdraws === nothing && (@warn "no chain for $lbl @ $(oc.origin)"; return nothing)
    weighted = is_weighted(dm)
    obs = [_observed_cell_mean(oc.apd, oc.t_o, i, j, weighted) for i in 1:A, j in 1:A]
    est = [median(μdraws[:, i, j]) for i in 1:A, j in 1:A]
    cmax  = maximum(x for x in Iterators.flatten((obs, est)) if isfinite(x))
    clims = (0.0, cmax)                              # shared across both panels
    hm(M, ttl) = heatmap(1:A, 1:A, M; clims = clims, c = :viridis, yflip = true,
                         xticks = (1:A, grid.LAB), yticks = (1:A, grid.LAB), xrotation = 45,
                         title = ttl, titlefontsize = 9, aspect_ratio = :equal,
                         xlabel = "contactee age group j", ylabel = "participant age group i")
    scale_note = weighted ? "positive duration-weighted degree" : "per-capita count"
    fig = plot(hm(obs, "Observed"), hm(est, "Estimated (smoothed μ)"); layout = (1, 2),
               size = (1050, 470), left_margin = 10Plots.mm, bottom_margin = 12Plots.mm,  # room for x-/y-labels
               plot_title = "$(lbl) — contact matrix ($(scale_note)), origin $(oc.origin)",
               plot_titlefontsize = 10)
    savefig(fig, joinpath(res_dir, "10j_contactmatrix_$(degree_label(dm))_$(ngm_label(nb))_$(oc.origin).png"))
    return fig
end

# ── §4 age-pair degree-distribution CCDF (observed ● vs estimated 90% band) ──

"""Expand a `WeightedDegreeHist` (distinct value → count) to a raw vector of positive weighted degrees."""
_whist_to_vec(w::WeightedDegreeHist) = isempty(w) ? Float64[] :
    vcat((fill(x, y) for (x, y) in zip(w.x, w.y))...)

"""
    estimated_ccdf_band(μd, κd, weighted, xgrid) -> (med, lo, hi)

Per-x estimated CCDF band over the `D` posterior draws (`μd`, `κd` are length-`D`), matched to
each path's observed normalisation:
- `weighted`  → positive-part `Weibull(κ, λ=μ/Γ(1+1/κ))` CCDF (closed form, positive-only).
- `!weighted` → NegBin count CCDF CONDITIONAL ON ≥1 (matches `plot_ccdf!`, which strips the zero
  bin): the pdf is evaluated on a bounded integer grid `0:kmax`, reverse-cumsummed, divided by
  `(1−P₀)`. We do NOT call `ccdf(::PoissonMixture,·)` — it is memoised and recurses to k_max=20_000,
  far too slow across D×49 cells.
Returns `(med, lo, hi)` (0.05/0.5/0.95 quantiles across draws) aligned to `xgrid`.
"""
function estimated_ccdf_band(μd::AbstractVector, κd::AbstractVector, weighted::Bool, xgrid::AbstractVector)
    D = length(μd)
    C = Matrix{Float64}(undef, D, length(xgrid))
    if weighted
        for d in 1:D
            κ = κd[d]; λ = μd[d] / gamma(1 + 1 / κ)               # Weibull scale, as in _cell_moments!
            C[d, :] = ccdf.(Weibull(κ, λ), xgrid)
        end
    else
        kmax = Int(maximum(xgrid)); ks = collect(0:kmax)
        for d in 1:D
            pk   = pdf.(NegBin(μd[d], κd[d]), ks)                 # bounded grid; pdf is cheap & exact
            tail = reverse(cumsum(reverse(pk)))                   # tail[k+1] = Σ_{j≥k} pdf(j)
            denom = max(1 - pk[1], 1e-12)                         # 1 − P₀  ⇒ condition on ≥1
            C[d, :] = [tail[Int(k) + 1] / denom for k in xgrid]
        end
    end
    med = [median(view(C, :, m)) for m in eachindex(xgrid)]
    lo  = [quantile(view(C, :, m), 0.05) for m in eachindex(xgrid)]
    hi  = [quantile(view(C, :, m), 0.95) for m in eachindex(xgrid)]
    return med, lo, hi
end

"""
    agepair_ccdf_panel(i, j, μdraws, κdraws, weighted, oc; xlim=:auto) -> Plots.Plot

One small age-pair CCDF panel: observed CCDF markers (at `oc.t_o`) + estimated median line & 90%
ribbon (`estimated_ccdf_band`). `xlim` is the figure-wide shared x-range (from `_shared_xlim`).
`κdraws` is `ndraws × A × A` and sliced PER CELL — the dispersion has been hierarchical since
2026-07-30 (§4.3), so it varies within a child/adult block; the old `[:, bl]` block-linear slice
would collapse all cells of a block onto one value.
"""
function agepair_ccdf_panel(i, j, μdraws, κdraws, weighted::Bool, oc; xlim = :auto)
    grid = oc.grid; apd = oc.apd; t_o = oc.t_o
    κd = view(κdraws, :, i, j); μd = view(μdraws, :, i, j)
    pnl = plot(; title = "$(grid.LAB[i])→$(grid.LAB[j])", titlefontsize = 6, xaxis = :log10, xlim = xlim,
               yscale = weighted ? :log10 : :identity, ylim = weighted ? (1e-5, 1.0) : (-5.0, 0.0),
               left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,   # room for per-panel axis ticks
               legend = false, tickfontsize = 5, guidefontsize = 6, xlabel = "", ylabel = "")
    if weighted
        pos = sort(filter(>(0), _whist_to_vec(apd.pos_weight[t_o, i, j])))
        isempty(pos) && return pnl
        n = length(pos)
        scatter!(pnl, pos, (n .- (0:n-1)) ./ n; color = :black, ms = 2, msw = 0.0, label = "")
        xgrid = exp10.(range(log10(minimum(pos)), log10(maximum(pos)); length = 60))
        med, lo, hi = estimated_ccdf_band(μd, κd, true, xgrid)
        plot!(pnl, xgrid, med; ribbon = (med .- lo, hi .- med), color = :darkorange,
              lw = 1.5, fillalpha = 0.15, label = "")
    else
        dd = apd.dd_count[t_o, i, j]
        posx = dd.x[dd.x .> 0]
        isempty(posx) && return pnl
        plot_ccdf!(pnl, dd; color = :black, markersize = 2, markerstrokewidth = 0.0,
                   linealpha = 0.0, label = "")
        xgrid = 1:maximum(posx)
        med, lo, hi = estimated_ccdf_band(μd, κd, false, xgrid)
        lmed = log10.(max.(med, 1e-12)); llo = log10.(max.(lo, 1e-12)); lhi = log10.(max.(hi, 1e-12))
        plot!(pnl, collect(xgrid), lmed; ribbon = (lmed .- llo, lhi .- lmed),
              color = :darkorange, lw = 1.5, fillalpha = 0.15, label = "")
    end
    return pnl
end

"""
    _shared_xlim(weighted, oc) -> Tuple{Float64,Float64} | Symbol

Figure-wide shared log10 x-range = the global positive-degree span over all 49 cells at `oc.t_o`
(negbin: integer counts ≥1; hweibull: positive weighted degrees). `:auto` if no cell has data.
"""
function _shared_xlim(weighted::Bool, oc)
    apd = oc.apd; t_o = oc.t_o; A = oc.grid.N
    xs = Float64[]
    for i in 1:A, j in 1:A
        if weighted
            append!(xs, filter(>(0), _whist_to_vec(apd.pos_weight[t_o, i, j])))
        else
            dd = apd.dd_count[t_o, i, j]; append!(xs, dd.x[dd.x .> 0])
        end
    end
    isempty(xs) && return :auto
    return (minimum(xs), maximum(xs))
end

"""
    make_agepair_ccdf_fig(dm, nb, oc; res_dir="../res") -> Plots.Plot | nothing

§4 — 7×7 grid of observed-vs-estimated log-log CCDF panels for one model. Reconstructs μ and
block-linear dispersion at the origin week (`oc.t_o_est`) and compares the estimated degree-
distribution SHAPE (not just the §3 mean) with the empirical CCDF. Saves to
`res_dir/10j_degdist_<degree>_<ngm>_<origin>.png`. `nothing` if a chain is missing.
"""
function make_agepair_ccdf_fig(dm::ContactDegreeModel, nb::NGMBuilder, oc;
                               res_dir::AbstractString = "../res")
    grid = oc.grid; A = grid.N
    lbl  = string(degree_label(dm), "|", ngm_label(nb))
    μdraws = reconstruct_mu_draws(lbl, oc.origin, 1; week_index = oc.t_o_est, grid = grid)
    κdraws = reconstruct_dispersion_draws(lbl, oc.origin, 1; weighted = is_weighted(dm),
                                          cfg = oc.cfg, grid = grid, week_index = oc.t_o_est)
    (μdraws === nothing || κdraws === nothing) && (@warn "no chain for $lbl @ $(oc.origin)"; return nothing)
    weighted = is_weighted(dm)
    xlim_shared = _shared_xlim(weighted, oc)                     # one x-axis for the whole grid
    panels = [agepair_ccdf_panel(i, j, μdraws, κdraws, weighted, oc; xlim = xlim_shared)
              for i in 1:A for j in 1:A]
    scale_note = weighted ? "positive duration-weighted degree" : "count (CCDF | ≥1)"
    fig = plot(panels...; layout = (A, A), size = (1500, 1500),
               left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,   # room for per-panel axis ticks
               plot_title = "$(lbl) — age-pair degree CCDF ($(scale_note)): observed ● vs estimated (90%), origin $(oc.origin)",
               plot_titlefontsize = 11)
    savefig(fig, joinpath(res_dir, "10j_degdist_$(degree_label(dm))_$(ngm_label(nb))_$(oc.origin).png"))
    return fig
end

"""
    make_forecast_ci_fig(fc_store, fit_store, win, wd, truth, cfg, labels4, model_cols, origin;
                         fit_h=1, qs_lo=0.05, qs_hi=0.95, age_idx=nothing, age_desc="all ages",
                         file_tag="", res_dir="../res") -> Plots.Plot

§1 — total-infection forecast point + 90% CI for the four models at one origin, overlaid on
observed (all ages). Solid + ○ + ribbon = self-iterated forecast (`fc_store[lbl]`, H×draws over
the forecast weeks); dashed + ◇ + ribbon = in-sample fitted mean (`fit_store[lbl]`, A×n_fit×draws
over the fit weeks, from the horizon-`fit_h` chain — `fit_window_infection_draws(...; h=fit_h)`).
Both age-summed; the origin week is marked with a vertical rule. Saves to
`res_dir/10j_forecast_ci_<origin>_fit-h<fit_h><file_tag>.png`.

`age_idx` restricts the ages summed into the reported total (default `nothing` ⇒ all `A` bins);
pass e.g. the non-70+ bins to report the 2–69 total only. NOTE: this excludes 70+ only from the
*reported sum*, not from the coupled NGM dynamics — the cached forecast draws already propagate all
ages, so this is a report-side subset, not a 6-bin re-fit. `age_desc` labels the axis/title and
`file_tag` is appended to the PNG name so a subset figure never overwrites the all-ages one.
"""
function make_forecast_ci_fig(fc_store, fit_store, win, wd, truth, cfg, labels4, model_cols,
                              origin::Date; fit_h::Integer = 1, qs_lo = 0.05, qs_hi = 0.95,
                              age_idx = nothing, age_desc::AbstractString = "all ages",
                              file_tag::AbstractString = "", res_dir::AbstractString = "../res")
    H  = length(cfg.horizons)
    A  = size(wd.I_mean, 1)
    ai = age_idx === nothing ? (1:A) : age_idx          # age bins summed into the reported total
    x_hist = week_mid.(win.fit_weeks)
    y_hist = vec(sum(wd.I_mean[ai, (cfg.smax + 1):end]; dims = 1))   # history (included ages)
    x_fore = week_mid.(win.forecast_weeks)
    y_fore = [sum(truth[ai, h]) for h in 1:H]                        # realised targets (included ages)

    fig = plot(; title = "10j — total infections ($(age_desc)) vs observed: in-sample fit " *
                         "(h$(fit_h), dashed) + forecast (solid), origin $(origin) (90%)",
               titlefontsize = 8, xrotation = 45, legend = :topleft, size = (950, 540),
               left_margin = 8Plots.mm, bottom_margin = 14Plots.mm,   # room for y-label & rotated dates
               xlabel = "week (Wed mid-date)", ylabel = "weekly infections ($(age_desc))")
    plot!(fig, vcat(x_hist, x_fore), vcat(y_hist, y_fore);
          color = :black, lw = 2, marker = :circle, ms = 3, label = "observed")
    vline!(fig, [week_mid(win.origin)]; color = :gray, ls = :dash, lw = 1, label = "")

    # self-iterated forecast fans (solid + ○) over the forecast weeks (right of the origin line).
    for (ci, lbl) in enumerate(labels4)
        haskey(fc_store, lbl) || continue
        tot = dropdims(sum(fc_store[lbl][ai, :, :]; dims = 1); dims = 1)   # H × draws (over included ages)
        med = [_fmed(tot[h, :])     for h in 1:H]                    # finite-robust (fan may be ±Inf)
        lo  = [_fq(tot[h, :], qs_lo) for h in 1:H]
        hi  = [_fq(tot[h, :], qs_hi) for h in 1:H]
        plot!(fig, x_fore, med; color = model_cols[ci], lw = 1.8, marker = :circle, ms = 2,
              ribbon = (med .- lo, hi .- med), fillalpha = 0.12, label = lbl)
    end
    # in-sample fitted mean (dashed + ◇) over the fit weeks only (left of the origin line) ⇒
    # visually separate from the solid+○ forecast.
    for (ci, lbl) in enumerate(labels4)
        haskey(fit_store, lbl) || continue
        tot = dropdims(sum(fit_store[lbl][ai, :, :]; dims = 1); dims = 1)  # n_fit × draws (over included ages)
        med = [_fmed(tot[t, :])     for t in 1:length(x_hist)]       # finite-robust (fit may be huge)
        lo  = [_fq(tot[t, :], qs_lo) for t in 1:length(x_hist)]
        hi  = [_fq(tot[t, :], qs_hi) for t in 1:length(x_hist)]
        plot!(fig, x_hist, med; color = model_cols[ci], lw = 1.6, ls = :dash, marker = :diamond,
              ms = 3, ribbon = (med .- lo, hi .- med), fillalpha = 0.10, label = "")
    end
    plot!(fig, [first(x_hist)], [NaN]; color = :gray, lw = 1.6, ls = :dash, marker = :diamond, ms = 3,
          label = "in-sample fit (h$(fit_h)), 90%")   # proxy: dashed ⇒ fitted; colour ⇒ model
    savefig(fig, joinpath(res_dir, "10j_forecast_ci_$(origin)_fit-h$(fit_h)$(file_tag).png"))
    return fig
end

"""
    forecast_ci_age_panel(a, fc_store, fit_store, win, wd, truth, cfg, labels4, model_cols;
                          fit_h=1, qs_lo=0.05, qs_hi=0.95, ttl="", showleg=false) -> Plots.Plot

One §1b panel: the §1 forecast-CI content (`make_forecast_ci_fig`) restricted to a SINGLE age bin
`a` — observed weekly infections (fit-week history ++ realised targets), per-model self-iterated
forecast (solid + ○ + 90% band, right of the origin) and per-model in-sample fitted mean
(dashed + ◇ + 90% band, left of the origin). Same colours/styling as §1; `showleg` toggles the
per-model legend (only the first panel carries it, to avoid clutter across the grid).
"""
function forecast_ci_age_panel(a::Integer, fc_store, fit_store, win, wd, truth, cfg, labels4,
                               model_cols; fit_h::Integer = 1, qs_lo = 0.05, qs_hi = 0.95,
                               ttl::AbstractString = "", showleg::Bool = false)
    H      = length(cfg.horizons)
    x_hist = week_mid.(win.fit_weeks)
    y_hist = vec(wd.I_mean[a, (cfg.smax + 1):end])        # single-age history
    x_fore = week_mid.(win.forecast_weeks)
    y_fore = [truth[a, h] for h in 1:H]                   # single-age realised targets

    pnl = plot(; title = ttl, titlefontsize = 8, xrotation = 45,
               legend = (showleg ? :topleft : false), legendfontsize = 5,
               xlabel = "week (Wed mid-date)", ylabel = "weekly infections")
    plot!(pnl, vcat(x_hist, x_fore), vcat(y_hist, y_fore);
          color = :black, lw = 2, marker = :circle, ms = 2, label = "observed")
    vline!(pnl, [week_mid(win.origin)]; color = :gray, ls = :dash, lw = 1, label = "")

    # self-iterated forecast fans (solid + ○) over the forecast weeks (right of the origin line).
    for (ci, lbl) in enumerate(labels4)
        haskey(fc_store, lbl) || continue
        tot = fc_store[lbl][a, :, :]                      # H × draws (single age)
        med = [_fmed(tot[h, :])      for h in 1:H]        # finite-robust (fan may be ±Inf)
        lo  = [_fq(tot[h, :], qs_lo) for h in 1:H]
        hi  = [_fq(tot[h, :], qs_hi) for h in 1:H]
        plot!(pnl, x_fore, med; color = model_cols[ci], lw = 1.8, marker = :circle, ms = 2,
              ribbon = (med .- lo, hi .- med), fillalpha = 0.12, label = lbl)
    end
    # in-sample fitted mean (dashed + ◇) over the fit weeks (left of the origin line).
    for (ci, lbl) in enumerate(labels4)
        haskey(fit_store, lbl) || continue
        tot = fit_store[lbl][a, :, :]                     # n_fit × draws (single age)
        med = [_fmed(tot[t, :])      for t in 1:length(x_hist)]
        lo  = [_fq(tot[t, :], qs_lo) for t in 1:length(x_hist)]
        hi  = [_fq(tot[t, :], qs_hi) for t in 1:length(x_hist)]
        plot!(pnl, x_hist, med; color = model_cols[ci], lw = 1.6, ls = :dash, marker = :diamond,
              ms = 2, ribbon = (med .- lo, hi .- med), fillalpha = 0.10, label = "")
    end
    showleg && plot!(pnl, [first(x_hist)], [NaN]; color = :gray, lw = 1.6, ls = :dash,
                     marker = :diamond, ms = 3, label = "in-sample fit (h$(fit_h)), 90%")
    return pnl
end

"""
    make_forecast_ci_by_age_fig(fc_store, fit_store, win, wd, truth, cfg, labels4, model_cols,
                                origin, grid; fit_h=1, qs_lo=0.05, qs_hi=0.95, file_tag="_byage",
                                res_dir="../res") -> Plots.Plot

§1b — the §1 forecast-CI diagnostic broken out PER AGE GROUP: one panel per CIS age bin (7 panels
in a 2×4 grid, 8th cell blank), each the single-age analogue of `make_forecast_ci_fig` (observed
++ per-model self-iterated forecast solid+○+90% and in-sample fitted mean dashed+◇+90%). Free y per
panel (age magnitudes differ widely); colours/styling match §1; only the first panel carries the
per-model legend. Under the two-stage cut the cached draws propagate all 7 ages through the coupled
NGM, so each panel is that age's slice of the joint 7-age forecast. Saves to
`res_dir/10j_forecast_ci_<origin>_fit-h<fit_h><file_tag>.png`.
"""
function make_forecast_ci_by_age_fig(fc_store, fit_store, win, wd, truth, cfg, labels4, model_cols,
                                     origin::Date, grid; fit_h::Integer = 1, qs_lo = 0.05,
                                     qs_hi = 0.95, file_tag::AbstractString = "_byage",
                                     res_dir::AbstractString = "../res")
    A = grid.N
    panels = [forecast_ci_age_panel(a, fc_store, fit_store, win, wd, truth, cfg, labels4,
                                    model_cols; fit_h = fit_h, qs_lo = qs_lo, qs_hi = qs_hi,
                                    ttl = "age $(grid.LAB[a])", showleg = (a == 1))
              for a in 1:A]
    push!(panels, plot(; framestyle = :none))            # blank cell fills the 2×4 grid (A=7)
    fig = plot(panels...; layout = (2, 4), size = (1550, 760),
               left_margin = 6Plots.mm, bottom_margin = 12Plots.mm,
               plot_title = "10j — weekly infections BY AGE GROUP vs observed: in-sample fit " *
                            "(h$(fit_h), dashed) + forecast (solid), origin $(origin) (90%)",
               plot_titlefontsize = 10)
    savefig(fig, joinpath(res_dir, "10j_forecast_ci_$(origin)_fit-h$(fit_h)$(file_tag).png"))
    return fig
end

"""
    make_susc_inf_fig(combos, labels4, model_cols, origin, cfg, grid; h=1, res_dir="../res")
        -> Plots.Plot

§5 — age-specific RELATIVE susceptibility and infectivity (reference bin 1 "2-10" fixed = 1) for the
four models at one forecast origin. Reloads each model's Stage-2 pooled draws
(`load_transmission_draws`, 8j_viz_utils.jl) — `susc`/`inf` are `N×A` pooled draws relative to the
reference bin — and plots the per-age-group median + 90% band. Two panels (susceptibility |
infectivity); x = age group, one coloured line per model (colours consistent with §1–§4 via
`labels4`/`model_cols`). Under the two-stage cut susc/inf are fit in Stage 2 conditioning on that
model's `C*`, so — unlike the NGM-independent μ — they genuinely differ across all four combos.
Finite-robust quantiles (`_fmed`/`_fq`) because the pooled draws are heavy-tailed. Read-only
(no re-fit). Saved to `res/10j_susc_inf_<origin>.png`. Missing artefacts are skipped (their line
is dropped) with a warning.
"""
function make_susc_inf_fig(combos, labels4, model_cols, origin::Date, cfg, grid;
                           h::Integer = 1, res_dir::AbstractString = "../res")
    A   = grid.N
    tag = contacts_label(cfg)
    xs  = 1:A

    mk(sym, ttl, showleg) = begin
        pnl = plot(; title = ttl, titlefontsize = 9,
                   xticks = (xs, grid.LAB), xrotation = 45,
                   xlabel = "age group", ylabel = "relative $(sym == :susc ? "susceptibility" : "infectivity")",
                   legend = (showleg ? :topleft : false), legendfontsize = 6)
        hline!(pnl, [1.0]; color = :gray, ls = :dash, lw = 1, label = "")   # reference bin = 1
        for (ci, (dm, nb)) in enumerate(combos)
            lbl = string(degree_label(dm), "|", ngm_label(nb))
            td  = load_transmission_draws(lbl, origin, h; contacts = tag)
            td === nothing && (@warn "no Stage-2 pooled artefact for susc/inf" lbl origin; continue)
            V   = sym == :susc ? td.susc : td.inf          # N × A pooled draws
            med = [_fmed(view(V, :, a))        for a in 1:A]
            lo  = [_fq(view(V, :, a), 0.05)    for a in 1:A]
            hi  = [_fq(view(V, :, a), 0.95)    for a in 1:A]
            plot!(pnl, xs, med; ribbon = (med .- lo, hi .- med), color = model_cols[ci], lw = 2,
                  marker = :circle, ms = 3, fillalpha = 0.08, label = labels4[ci])
        end
        pnl
    end

    fig = plot(mk(:susc, "Relative susceptibility", true),
               mk(:inf,  "Relative infectivity",    false);
               layout = (1, 2), size = (1150, 500),
               left_margin = 9Plots.mm, bottom_margin = 12Plots.mm,
               plot_title = "10j — relative age-specific susceptibility & infectivity " *
                            "(ref bin \"$(grid.LAB[1])\" = 1), median + 90%, origin $(origin) (h$(h))",
               plot_titlefontsize = 10)
    savefig(fig, joinpath(res_dir, "10j_susc_inf_$(origin).png"))
    return fig
end
