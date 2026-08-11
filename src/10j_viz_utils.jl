# 10j_viz_utils.jl — read-only helper for the 10j single-time-point diagnostics.
#
# Reconstructs the GP-smoothed directional contact mean μ_{i→j} per posterior draw from a
# cached STAGE-1 chain, WITHOUT rebuilding the model — mirroring the `load_transmission_draws`
# pattern in 8j_viz_utils.jl. μ is a deterministic transform of the raw sampled columns
# (log_rho_diag, log_rho_gap, log_eta; and, for the separable spatio-temporal regime, phi_time,
# log_sigma_c, scalar level c, temporal-level raw z_c, structure-field raw z[·,·]); see `model_degree`
# (joint_model.jl §5/§6). Requires 8j_viz_utils.jl (for `stage1_chain_path`) to be included first.
# LinearAlgebra (cholesky/Symmetric/I/dot) and `_unordered_pairs`/`cis_age_midpoints` come in via
# forecast_utils.jl.

"""
    _stage1_gp_generation(pnames, contacts, path) -> (; temporal_is_ar1)  |  nothing

The three generation checks EVERY `model_degree` Stage-1 GP mirror in this file must apply before it
touches a draw, factored out (2026-08-10) so a new reader cannot ship with two of the three. Returns
`nothing` having already `@warn`ed — callers just `return nothing` — or the one FORK decision the
caller needs: which temporal kernel wrote this chain.

Extracted because the failure this repo actually suffers is a guard fixed in one mirror and missed in
another: the 2026-08-08 `contacts`-default sweep fixed five 10j call sites and missed
`collect_transmission_structure`, blanking all eight 9j transmission figures until 2026-08-09; and the
`occursin("-m32", …)` → `is_legacy_token` rewrite had to land in two functions in lockstep. These
guards exist precisely because the offending chain looks IDENTICAL to a current one, so a missed copy
fails silently.

⚠ NOT here, deliberately — the checks needing quantities only the caller has:
  • `-s0`: the `z` ROW count, which needs `P` from the grid (`reconstruct_gp_hyper_draws` makes its
    own, because η's MEANING depends on that projection while μ's shape does not);
  • `-t0`: the `z_c` COLUMN count, which needs `Tn`;
  • `-rhs` (`c2`): `_read_disp_chain`'s, about the dispersion block rather than the GP.
⚠ And NOT `load_transmission_draws` (8j_viz_utils.jl), which applies the same three checks with a
REFUSAL where this FORKS — 9j draws one figure per generation and mixing week-units onto a (0,1) axis
is worse there than a NaN panel. That asymmetry is deliberate and documented on both sides; keep the
two in step BY HAND rather than unifying them, which would pull 8j and all of 9j into a viz change.

# ---- which SPATIAL KERNEL generation is this chain? (`-m32`, 2026-08-05) ----
`model_degree` uses a SEPARABLE ANISOTROPIC kernel with TWO length-scales, so `log_rho_gap` must be
present. A chain lacking it was fitted under the short-lived `-diag` generation (diagonal-only
smoothing, one length-scale), and replaying it through the two-length-scale formula would silently
rebuild a DIFFERENT kernel — every μ / C* / contact matrix / CCDF wrong with nothing raised. The
`zrows` sniff at the call site cannot catch this (the `z` shape is identical to `-diag`'s), so it
needs its own. NOTE this guard was INVERTED on 2026-08-05: it previously refused chains that CARRIED
`log_rho_gap`.
⚠ It does NOT distinguish `-m32` from the pre-`-diag` squared-exponential generation, which also
carried two length-scales — that fork is the token check below, since `-m32` renamed it. Do not rely
on this sniff alone if you stage a chain under a hand-written filename.

# ---- which TEMPORAL kernel? ----
This is a FORK, not a guard, and deliberately so: BOTH temporal kernels are on disk and both must stay
replayable. It was introduced by `-m32t` (2026-08-10) and KEPT when that generation was reverted the
same day — which is precisely what it earned its place for, since the fork is what lets the `-m32t`
smoke chains still be read as the evidence for the revert.
  • `phi_time`     ⇒ AR(1) coefficient φ ∈ (0,1) — the CURRENT kernel, and the `-ar1` generation
    whose complete 504/1512-file grid is retained in `dt_intermediate_ar1/` (`CONTACTS_TOKEN_AR1`).
  • `log_rho_time` ⇒ Matérn 3/2 length-scale in weeks — the one-day `-m32t` generation and every
    generation before `-ar1`.
Getting it wrong is silent and severe in either direction: a length-scale of 26 weeks read as a
correlation of 26, or a correlation of 0.99 read as a 0.99-week length-scale. The NAME is a reliable
discriminator here (unlike the spatial kernel family, which needs the token) BECAUSE the two
parameterisations chose different names. `-m32t` changed no dimension, so the count cannot help.

# ---- which KERNEL FAMILY? (`-m32`, 2026-08-05) ----
A pre-`-diag` chain is PARAMETRICALLY IDENTICAL to a current one — same names, same shapes, two
length-scales — but its kernel was the SQUARED EXPONENTIAL. No sniff over `pnames` can tell them
apart, so this forks on the TOKEN, which `-m32` renamed for exactly that reason. Without it,
`contacts = CONTACTS_TOKEN_PF` — a supported way to read the retained Pathfinder grid — would
silently replay SE draws through a Matérn kernel. Current-generation tokens have Matérn 3/2 BY
CONSTRUCTION; only a RETAINED legacy token has to prove it (see `is_legacy_token`). This was a bare
`occursin("-m32", contacts)` until the accumulated token prefix was dropped on 2026-08-09 — which
would have rejected every current chain.
"""
function _stage1_gp_generation(pnames::AbstractVector{<:AbstractString},
                               contacts::AbstractString, path::AbstractString)
    if !any(n -> n == "log_rho_gap", pnames)
        @warn "chain has no `log_rho_gap`, i.e. the `-diag` diagonal-only spatial kernel. \
               Refusing to reconstruct rather than replay it through the two-length-scale kernel." path
        return nothing
    end
    temporal_is_ar1 = any(n -> n == "phi_time", pnames)
    if !temporal_is_ar1 && !any(n -> n == "log_rho_time", pnames)
        @warn "chain carries neither `log_rho_time` nor `phi_time` — its temporal kernel cannot be \
               identified, so it is not a per-week `model_degree` Stage-1 chain. Refusing to \
               reconstruct." path
        return nothing
    end
    if is_legacy_token(contacts) && !occursin("-m32", contacts)
        @warn "contacts token `$contacts` predates `-m32`, so this chain's kernel was the squared \
               exponential, not Matérn 3/2. The chain columns are indistinguishable from a current \
               one, so this cannot be detected from the chain — refusing on the token." path
        return nothing
    end
    return (; temporal_is_ar1)
end

"""
    reconstruct_mu_draws(lbl, origin, h; week_index=nothing, grid, contacts, save_dir)
        -> ndraws × A × A  |  nothing

Load the cached chain for `(lbl, origin, h)` and rebuild the smoothed directional contact-mean
matrix μ_{i→j} for one week, once per posterior draw:

    ρ_diag = exp(softclamp(log_rho_diag, RHO_BOUNDS...));  ρ_gap = exp(softclamp(log_rho_gap, RHO_BOUNDS...))
    η = exp(softclamp(log_eta, -3, 2))  (mirrors model)
    u = (mid_p1+mid_p2)/√2 (total age);  v = (mid_p1-mid_p2)/√2 (age gap)
    m32(x) = (1+√3 x)·exp(-√3 x)                                    # Matérn 3/2 (`-m32`, 2026-08-05)
    Kp[p,q] = m32(|u_p-u_q|/ρ_diag) · m32(|v_p-v_q|/ρ_gap)          # 28 unordered pairs, separable
    Lp = chol(Kp + 1e-6 I).L
    μ[i,j] = exp(softclamp(rvec[pair_index[i,j]] + log(pop_j / pop_ref), -8, 6))   (pop_ref = pop[1], "2-10")

with the per-week rate `rvec` built for the requested `week` (`wk`):

- **Separable spatio-temporal regime** (the cached `contacts="temporal"` chains): scalar intercept
  `c`, weekly level and matrix-normal structure field share the temporal Cholesky `Lt` over the full
  `Tn` window weeks. `Lt[wk,:]` (= column `wk` of `Ltᵀ`) mixes weeks `1..wk`, so the FULL field
  `z` (P×Tn) and level `z_c` (Tn−1) are needed, not just week `wk`:

      σ_c = exp(softclamp(log_sigma_c, -3, 2));   Lt = chol(Kt + 1e-4 I).L
      Qt = _sum_zero_basis(Tn)                                              # `-t0`, level only
      rvec = ( c + σ_c·(Qt·z_c)[wk] )  .+  η·( Lp · (z · Lt[wk,:]) )        # `-lc0`: level is iid

  ⚠ TWO INDEPENDENT GENERATION FORKS, decided by DIFFERENT evidence. Both are live because all four
  combinations exist on disk, and replaying a chain through the wrong branch silently returns a
  plausible but different μ.

  1. **Which temporal kernel** — forked on the chain's own PARAMETER NAME, which is reliable here
     because the two parameterisations chose different ones:

         phi_time     present ⇒ Kt[s,t] = phi_time^|s−t|         # AR(1) — CURRENT, read CONSTRAINED
         log_rho_time present ⇒ ρ_time = exp(softclamp(log_rho_time, RHO_TIME_BOUNDS...))
                                Kt[s,t] = m32(|s−t|/ρ_time)      # `-m32t` and everything pre-`-ar1`

  2. **Whether the level is whitened** — forked on the `-lc0` marker in `contacts`, because here the
     parameter names are IDENTICAL across generations and only the token distinguishes them. From
     `-lc0` (2026-08-09) the level is iid-with-sum-to-zero as written above, so the temporal
     parameter enters ONLY through `Lt`; before it the deviation went through
     `Lc = chol(Qtᵀ·Kt·Qt + 1e-4 I).L` and the temporal kernel acted on the level too.

- **Legacy per-week iid** (`c[t]`, `z[p,t]`): `rvec = c[wk] .+ η .* (Lp * z[:,wk])`.
- **Pooled** (scalar `c`, `z[p]`): `rvec = c .+ η .* (Lp * z)`.

`week_index` defaults to the last window week (the origin week the forecast NGM is frozen at).
Returns `nothing` when the chain file is missing.
"""
function reconstruct_mu_draws(lbl::AbstractString, origin::Date, h::Integer;
                              week_index::Union{Int,Nothing} = nothing,
                              grid,
                              contacts::AbstractString = CONTACTS_TOKEN,
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

    pnames = string.(names(chn, :parameters))

    # The three SHARED generation checks — (1) the `-diag` spatial-kernel refusal, (2) the
    # `phi_time`-vs-`log_rho_time` temporal FORK, (3) the pre-`-m32` kernel-family refusal on the
    # TOKEN. All three live in `_stage1_gp_generation` (above) so this mirror and
    # `reconstruct_gp_hyper_draws` cannot drift apart; read its docstring for why each exists and
    # which evidence — chain name or token — it is allowed to use. The `-s0`/`-t0` sniffs below stay
    # here: they need `P` and `Tn`, which only this caller has.
    gen = _stage1_gp_generation(pnames, contacts, path)
    gen === nothing && return nothing                    # it has already warned
    temporal_is_ar1 = gen.temporal_is_ar1

    # `RHO_BOUNDS` (framework.jl), NOT literals — this MUST track `model_degree` or every
    # reconstructed μ / C* is silently wrong. See the constants' docstring.
    ρ_diag = exp.(_softclamp.(vec(Array(chn[:log_rho_diag])), RHO_BOUNDS...))   # soft-bounded, mirrors model
    ρ_gap  = exp.(_softclamp.(vec(Array(chn[:log_rho_gap])),  RHO_BOUNDS...))   # soft-bounded, mirrors model
    η = exp.(_softclamp.(vec(Array(chn[:log_eta])), -3.0, 2.0))
    D = length(ρ_diag)
    # rotated (diagonal / anti-diagonal) coordinates for the 28 pairs, √2-normalised (mirrors model).
    su = [(mid[p[1]] + mid[p[2]]) / sqrt(2) for p in pair_list]   # along-diagonal  (total age)
    dfp = [(mid[p[1]] - mid[p[2]]) / sqrt(2) for p in pair_list]  # across-diagonal (age gap)

    # ---- which GP field generation is this chain? (`-s0`, 2026-08-05) ----
    # `model_degree` constrains the structure field to sum to zero over the P pairs each week, so
    # `z` has P−1 = 27 rows, not P = 28. Chains written before that have 28, and getting it wrong is
    # silent: `Z` below is allocated at a size derived from the GRID, so replaying a 27-row chain
    # under the 28-row formula would leave row 28 as uninitialised memory and every μ / C* / CCDF
    # would contain garbage without so much as a warning.
    #
    # P−1 IS NOW THE ONLY ACCEPTED COUNT. `-s0` landed before `-m32` and was never reverted, so any
    # chain the token guard above admits is necessarily sum-to-zero; a P-row chain reaching here is
    # a pre-`-s0` file staged under a current-token filename, which the token fork cannot see. The
    # unconstrained branch that used to handle it was therefore both unreachable in normal use and,
    # since `-m32`, actively wrong (it would rebuild a squared-exponential generation's field with a
    # Matérn kernel), so it is gone rather than kept as a trap. Restore it only alongside a
    # kernel-family branch.
    zrows = maximum((parse(Int, match(r"^z\[(\d+)", n).captures[1])
                     for n in pnames if occursin(r"^z\[\d+", n)); init = 0)
    if zrows == 0
        @warn "chain has no `z[...]` structure-field columns — not a `model_degree` Stage-1 chain" path
        return nothing
    elseif zrows != P - 1
        @warn "chain's structure field has $zrows rows, expected $(P-1) (sum-to-zero `-s0`) for \
               A=$A. $(zrows == P ? "This is the pre-`-s0` unconstrained generation, whose kernel \
               also predates `-m32`. " : "")Refusing to reconstruct rather than guess." path
        return nothing
    end
    sz_Q = _sum_zero_basis(P)                    # SAME helper the model uses — never re-derive it

    # Spatial whitening for draw d (mirrors model): `Q·chol(Qᵀ·Kp·Q + 1e-6·I)`, giving Cov = M·Kp·M.
    # Maps a (P−1)-vector of `z` to the P-vector field.
    function _Lp(d)
        Kp = [_m32(abs(su[m] - su[n]) / ρ_diag[d]) * _m32(abs(dfp[m] - dfp[n]) / ρ_gap[d])
              for m in 1:P, n in 1:P]
        sz_Q * cholesky(Symmetric(sz_Q' * Kp * sz_Q) + 1e-6 * I).L
    end
    μ = Array{Float64,3}(undef, D, A, A)

    # Separable spatio-temporal regime — the only one the guard above admits. (The per-week-iid and
    # pooled branches at the foot of this function are consequently unreachable in normal use; they
    # predate the guard and are left as documentation of those chain layouts.)
    if temporal_is_ar1 || any(n -> n == "log_rho_time", pnames)
        # infer Tn from the structure-field names z[p,t] (z_c[t]/z_s/z_i don't match "^z\[")
        Tn = maximum(parse(Int, match(r"^z\[\d+\s*,\s*(\d+)\]$", n).captures[1])
                     for n in pnames if occursin(r"^z\[\d+\s*,\s*\d+\]$", n))
        wk = week_index === nothing ? Tn : week_index
        # BOUNDS GUARD — `Tn` is GENERATION- AND HORIZON-DEPENDENT since `-w8h` (n_fit+h = 9..12 now,
        # a flat 12 under `-t0-ar1`), so a
        # caller iterating weeks across generations will overshoot. Without this the overshoot dies
        # deep inside the draw loop on `Lt[wk, :]` (`BoundsError` on an 8×8 LowerTriangular) — loud,
        # but from a stack frame that says nothing about which chain or which week. Refusing here
        # matches every other generation check in this function, and matches
        # `_read_disp_chain`'s guard so the two mirrors behave the same way.
        if !(1 <= wk <= Tn)
            @warn "week_index $wk outside the chain's $Tn weeks — refusing to reconstruct" path
            return nothing
        end
        cc     = vec(Array(chn[:c]))                                                    # D scalar intercept
        σ_c    = exp.(_softclamp.(vec(Array(chn[:log_sigma_c])), -3.0, 2.0))            # D
        # The temporal parameter, read the way ITS OWN generation stored it (see the fork above):
        #   • AR(1) (current, and `-ar1`): `phi_time` is stored CONSTRAINED in (0,1) — no exp, no
        #     softclamp, because `model_degree` has none either (φ^k cannot overflow).
        #   • `-m32t` and pre-`-ar1`: `log_rho_time` is a raw log-latent, so it must be soft-clamped
        #     with `RHO_TIME_BOUNDS` and exponentiated — mirroring the `model_degree` of ITS OWN
        #     generation. Getting this wrong is the silent-length-scale failure the constants'
        #     docstring warns about, which is why `RHO_TIME_BOUNDS` is retained though dead.
        # `tpar` carries whichever it is; `temporal_is_ar1` says how to turn it into `Kt` below.
        tpar = temporal_is_ar1 ? vec(Array(chn[:phi_time])) :
               exp.(_softclamp.(vec(Array(chn[:log_rho_time])), RHO_TIME_BOUNDS...))    # D
        Z  = Array{Float64,3}(undef, D, zrows, Tn)           # structure-field raw z[p,t]; `zrows`, NOT `P` — see the generation sniff above
        for n in pnames
            m = match(r"^z\[(\d+)\s*,\s*(\d+)\]$", n); m === nothing && continue
            Z[:, parse(Int, m.captures[1]), parse(Int, m.captures[2])] = vec(Array(chn[Symbol(n)]))
        end
        # ---- `-t0` (2026-08-06): the LEVEL's temporal deviation is sum-to-zero over the weeks ----
        # `z_c` therefore has Tn−1 columns, not Tn, and the level is rebuilt through the temporal
        # Helmert basis. A chain with Tn columns is the pre-`-t0` generation; refuse rather than
        # guess, exactly as the `zrows` sniff does spatially. (The STRUCTURE FIELD is unaffected —
        # it keeps the full `Lt` — so `Z` above is still Tn-wide.)
        zc_cols = maximum((parse(Int, match(r"^z_c\[(\d+)\]$", n).captures[1])
                           for n in pnames if occursin(r"^z_c\[", n)); init = 0)
        if zc_cols != Tn - 1
            @warn "chain's temporal level has $zc_cols z_c columns, expected $(Tn-1) (sum-to-zero \
                   `-t0`). $(zc_cols == Tn ? "This is the pre-`-t0` generation. " : "")Refusing to \
                   reconstruct rather than guess." path
            return nothing
        end
        Zc = Matrix{Float64}(undef, D, zc_cols)              # temporal-level raw z_c[t], Tn−1 wide
        for n in pnames
            m = match(r"^z_c\[(\d+)\]$", n); m === nothing && continue
            Zc[:, parse(Int, m.captures[1])] = vec(Array(chn[Symbol(n)]))
        end
        tz_Q = _sum_zero_basis(Tn)                           # SAME helper the model uses — never re-derive
        # ---- `-lc0` (2026-08-09): the LEVEL lost its temporal whitening ----
        # From `-lc0` the level is `c + σ_c·(Qt·z_c)` — iid deviations conditioned to sum to zero, so
        # the temporal parameter reaches μ ONLY through `Lt`. Before it, the deviation was whitened
        # through `Lc = chol(Qtᵀ·Kt·Qt + 1e-4·I)` and the temporal kernel acted on the level as well.
        # Both generations are on disk (`CONTACTS_TOKEN_AR1`), the parameter NAMES are identical in
        # both, and replaying one with the other's algebra returns a plausible-but-wrong μ with no
        # error — so THIS branch is taken on the TOKEN, which is the only thing that distinguishes
        # them. (Contrast the kernel-family fork above, which the parameter NAMES do distinguish.)
        # The two forks are independent and all four combinations occur on disk: the current
        # `temporal-w8h-lc0` and the superseded `temporal-w8-lc0` grid (AR(1) + iid), the one-day
        # `-m32t` smoke (m32 + iid), `-t0-ar1` (AR(1) + Lc) and the `-m32-t0` NUTS pilots (m32 + Lc).
        level_is_iid = !is_legacy_token(contacts) || occursin("-lc0", contacts)
        for d in 1:D
            # `Kt` mirrors whichever `model_degree` wrote this chain — see the fork above.
            Kt = temporal_is_ar1 ? [tpar[d]^abs(s - t) for s in 1:Tn, t in 1:Tn] :
                                   [_m32(abs(s - t) / tpar[d]) for s in 1:Tn, t in 1:Tn]
            Lt = cholesky(Symmetric(Kt) + 1e-4 * I).L
            ltrow = Lt[wk, :]                                # row wk of Lt = column wk of Ltᵀ
            # level: cₜ = c + σ_c·(tz_Q·z_c)_wk  (`-lc0`), or ·(tz_Q·Lc·z_c)_wk before it
            dev = if level_is_iid
                dot(view(tz_Q, wk, :), @view Zc[d, :])
            else
                Lc = cholesky(Symmetric(transpose(tz_Q) * Kt * tz_Q) + 1e-4 * I).L
                dot(view(tz_Q, wk, :), Lc * (@view Zc[d, :]))
            end
            c_wk = cc[d] + σ_c[d] * dev
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
        z_t = Matrix{Float64}(undef, D, zrows)               # D × zrows (field for week wk)
        for n in pnames
            m = match(r"^z\[(\d+)\s*,\s*(\d+)\]$", n)
            m === nothing && continue
            p = parse(Int, m.captures[1]); t = parse(Int, m.captures[2])
            t == wk && (z_t[:, p] = vec(Array(chn[Symbol(n)])))
        end
    else                                                      # pooled regime: scalar c, z[p]
        c_t = vec(Array(chn[:c]))                            # D
        z_t = Matrix{Float64}(undef, D, zrows)
        for p in 1:zrows
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
    reconstruct_gp_hyper_draws(lbl, origin, h; grid, contacts, save_dir)
        -> (; rho_diag, rho_gap, eta, sigma_c, phi, rho_time, temporal_is_ar1, Tn, ndraws,
              sampler, phi_init_scale, diag, ess_bulk, rhat, path)  |  nothing

Per-draw posterior of the Stage-1 GP SMOOTHING hyperparameters from a cached chain, without
rebuilding the model — the scalar companion to `reconstruct_mu_draws` (same chain, same draw order).
`model_degree` samples these five ONCE PER FIT and applies them to all 28 age pairs and all `Tn`
window weeks, so they are the whole smoothing story in five numbers:

    ρ_diag = exp(softclamp(log_rho_diag, RHO_BOUNDS...))   # total-age direction, ROTATED age-yrs
    ρ_gap  = exp(softclamp(log_rho_gap,  RHO_BOUNDS...))   # age-gap direction,  ROTATED age-yrs
    η      = exp(softclamp(log_eta,     -3, 2))            # age-structure field amplitude
    σ_c    = exp(softclamp(log_sigma_c, -3, 2))            # weekly-level amplitude
    φ      = phi_time                                      # AR(1) lag-1 corr — READ RAW, (0,1)

⚠ `phi_time` is stored CONSTRAINED: no `exp`, no `_softclamp`, because `model_degree` has neither
(`φ^k` cannot overflow, so `RHO_TIME_BOUNDS` is dead on this path). The three log-latents MUST go
through `_softclamp` with the model's own bounds or every reported value is silently wrong at the
edges — the same mirrors-match-model invariant `reconstruct_mu_draws` and `_read_disp_chain` carry.
`c` is deliberately absent: it is the level INTERCEPT, not a smoothing parameter, and its prior
centre is data-dependent (`c0 = mean(log_emp − logpop)`), so it is not comparable across windows.
Two units caveats that belong to the READER of these numbers, not to this function: both ρ live on
the √2-ROTATED coordinates, so an effective age-DIFFERENCE length-scale is `ρ/√2`; and since `-s0`
η is no longer exactly the field's marginal SD (that is `η·sqrt(diag(M·Kp·M))`, ×0.71–1.08 at the
`gp_len_prior` mode), so η and σ_c each compare across chains but not directly with each other.

**`contacts` has NO DEFAULT, and that is deliberate.** Every sibling reader here defaults to
`CONTACTS_TOKEN`, which is a compile-time literal that always ends `-nuts` and silently ignores
`cfg` — the bug that blanked four 10j figures on 2026-08-08 and all eight 9j transmission figures on
2026-08-09 (CLAUDE.md's `CONTACTS_TOKEN` gotcha). Nothing depends on this function yet, so the
footgun is made UNREPRESENTABLE rather than documented. Pass `contacts_label(cfg)`, or go through
`collect_gp_hyper`.

**Generations.** The three shared checks via `_stage1_gp_generation` (refuses `-diag` chains and
pre-`-m32` tokens, FORKS on the temporal parameter name), plus its own `-s0` check: η is an age-
STRUCTURE amplitude only because the field is projected to sum to zero over the 28 pairs, so a
pre-`-s0` chain's η means something else and must not be tabulated beside a current one. The temporal
fork fills exactly one of the two temporal fields:

    temporal_is_ar1 == true  ⇒ `phi` populated,      `rho_time` all-NaN   # `-ar1` and current
    temporal_is_ar1 == false ⇒ `rho_time` populated, `phi`      all-NaN   # `-m32t`, pre-`-ar1`

so a consumer must be NaN-safe on both (use `_fmed`/`_fq`), and a figure must never put weeks on a
(0,1) axis. Keeping the fork rather than refusing is what lets the retained `-m32t` smoke chains be
read as the evidence for reverting `-m32t`.

**Provenance, because the token does not carry it.** `sampler`, `phi_init_scale` and `diag` are read
straight out of the artefact (`fit_or_load_stage1` has written them since 2026-08-05 / 2026-08-10),
via `haskey` so an older artefact yields `missing` rather than throwing. ⚠ Read `sampler` before
quoting φ: measured, a PATHFINDER φ is largely a function of its starting value (0.011 → 1.000 across
five inits on one cell), so φ is a posterior summary only on the NUTS path.

`ess_bulk`/`rhat` are `Dict{Symbol,Float64}` over the SAMPLED columns (`missing` if `summarystats`
fails), so a 90% interval resting on ~200 effective draws can be read as such. They are computed on
the log-scale latents and apply unchanged to the exponentiated values: both statistics are RANK-based
and ranks are invariant under the monotone `exp ∘ _softclamp`.

Returns `nothing` when the file is missing (silently — the standing contract of every reader here) or
when it is unreadable / of a refused generation (with a warning).
"""
function reconstruct_gp_hyper_draws(lbl::AbstractString, origin::Date, h::Integer;
                                    grid,
                                    contacts::AbstractString,      # NO DEFAULT — see the docstring
                                    save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    path = stage1_chain_path(lbl, origin, h; contacts = contacts, save_dir = save_dir)
    isfile(path) || return nothing
    # `jldopen` + `haskey`, never a bare `load(path, key)`: the provenance keys were added over time
    # (`sampler`/`diag` 2026-08-05, `phi_init_scale` 2026-08-10) and a retained artefact predating one
    # of them must come back `missing`, not throw. Same pattern as 12j's `load_nuts_chain`.
    chn, sampler, phi_init, dg = try
        jldopen(path, "r") do f
            (f["result"],
             haskey(f, "sampler")        ? f["sampler"]        : missing,
             haskey(f, "phi_init_scale") ? f["phi_init_scale"] : missing,
             haskey(f, "diag")           ? f["diag"]           : missing)
        end
    catch err
        @warn "could not load chain" path err
        return nothing
    end

    pnames = string.(names(chn, :parameters))
    gen = _stage1_gp_generation(pnames, contacts, path)   # shared guards + the temporal fork
    gen === nothing && return nothing

    A = grid.N
    pair_list, _ = _unordered_pairs(A)
    P = length(pair_list)
    # ---- `-s0` (2026-08-05): the field sums to zero over the P pairs, so `z` has P−1 rows ----
    # Unlike the μ mirror this function never touches `z`, so nothing here would BREAK on a 28-row
    # chain — which is exactly why the check has to be explicit: η's meaning is "amplitude of the
    # PROJECTED field", and reporting a pre-`-s0` η in the same table as a current one would compare
    # two different quantities under one name.
    zrows = maximum((parse(Int, match(r"^z\[(\d+)", n).captures[1])
                     for n in pnames if occursin(r"^z\[\d+", n)); init = 0)
    if zrows != P - 1
        @warn "chain's structure field has $zrows rows, expected $(P-1) (sum-to-zero `-s0`) for \
               A=$A — refusing to report η, whose meaning depends on that projection." path
        return nothing
    end
    # `Tn` from the `z[p,t]` column names. VARIES 9..12 across one origin's four horizon chains since
    # `-w8h` (Tn = n_fit + h), where it used to be a flat 12 — so it is reported, not assumed.
    tcols = [parse(Int, m.captures[1]) for n in pnames
             for m in (match(r"^z\[\d+\s*,\s*(\d+)\]$", n),) if m !== nothing]
    if isempty(tcols)
        @warn "chain has no 2-D `z[p,t]` columns — not a per-week `model_degree` chain" path
        return nothing
    end
    Tn = maximum(tcols)

    # MIRRORS `model_degree` — `RHO_BOUNDS` from framework.jl, and the (-3, 2) pair as the literals
    # the model itself writes for both amplitudes. Keep these in lockstep with §5/§6 of joint_model.jl.
    ρ_diag = exp.(_softclamp.(vec(Array(chn[:log_rho_diag])), RHO_BOUNDS...))
    ρ_gap  = exp.(_softclamp.(vec(Array(chn[:log_rho_gap])),  RHO_BOUNDS...))
    η      = exp.(_softclamp.(vec(Array(chn[:log_eta])),      -3.0, 2.0))
    σ_c    = exp.(_softclamp.(vec(Array(chn[:log_sigma_c])),  -3.0, 2.0))
    D = length(ρ_diag)
    # The temporal parameter, read the way ITS OWN generation stored it (see the fork). The unused
    # one is all-NaN rather than absent, so consumers have a fixed field set and NaN-safe summaries
    # (`_fmed`/`_fq`) do the right thing on either generation.
    φ  = gen.temporal_is_ar1 ? vec(Array(chn[:phi_time])) : fill(NaN, D)
    ρt = gen.temporal_is_ar1 ? fill(NaN, D) :
         exp.(_softclamp.(vec(Array(chn[:log_rho_time])), RHO_TIME_BOUNDS...))

    # Per-scalar mixing. Computed on the SAMPLED (log/constrained) columns; `summarystats` is wrapped
    # because it is a convenience, not the payload — a version skew here must not cost the estimates.
    syms = [:log_rho_diag, :log_rho_gap, :log_eta, :log_sigma_c]
    gen.temporal_is_ar1 && push!(syms, :phi_time)
    ess_bulk, rhat = missing, missing
    try
        nt = summarystats(chn[:, syms, :]).nt         # sub-chain: 5 columns, not the 293–977 of them
        ks = Symbol.(string.(nt.parameters))          # zip against the RETURNED order, never assume
        # `haskey` per statistic, as `convergence_table` (12j) does — the set `summarystats` returns
        # depends on the number of chains, and one chain per fit means no R̂ at all on some versions.
        haskey(nt, :ess_bulk) && (ess_bulk = Dict(zip(ks, Float64.(collect(nt.ess_bulk)))))
        haskey(nt, :rhat)     && (rhat     = Dict(zip(ks, Float64.(collect(nt.rhat)))))
    catch err
        @warn "summarystats failed; ESS/R̂ reported as missing" path err
    end

    return (; rho_diag = ρ_diag, rho_gap = ρ_gap, eta = η, sigma_c = σ_c, phi = φ, rho_time = ρt,
              temporal_is_ar1 = gen.temporal_is_ar1, Tn, ndraws = D,
              sampler, phi_init_scale = phi_init, diag = dg, ess_bulk, rhat, path)
end

"""
    _prior_hyper(cfg, par; q=(0.05,0.5,0.95)) -> (; dist, band, bounds, label, logscale, sym, pretty)

The prior of ONE reported hyperparameter, keyed by `par ∈ (:rho_diag, :rho_gap, :eta, :sigma_c, :phi)`
and read from `cfg` — never from a literal — so a prior change lands in one place and cannot desync
between the table and the two figures.

  • `band`  = the (5%, 50%, 95%) quantiles of the prior the model ACTUALLY uses, i.e. Normal quantiles
    pushed through the SAME `_softclamp` + `exp` the model applies (9j's `_prior_gamma_band` pattern:
    a band from the unclamped log-normal would not be that prior).
  • `dist`  = the UNCLAMPED distribution, for the density CURVE and for `prior_pctl`. It is NOT the
    same as `band`, and never exactly: `_softclamp` is a softplus pair, so it shifts every point by
    O(exp(−distance/s)) rather than being the identity anywhere. Measured against the analytic
    quantiles — ρ by ≤7e-6 relative (its bounds are ~10σ out), the amplitudes by 8e-5 at the median
    and 2.3e-3 at the 95th, because `log_eta`'s (−3, 2) window puts +1.645σ only 1.18 nats below the
    upper bound. Since `_softclamp` is monotone, a percentile read off `dist` at a clamped value is
    wrong only by that same fraction, which is why `prior_pctl` uses the analytic CDF rather than
    inverting the clamp. Never use `dist` for `band` — the band is the prior the sampler SAW.
  • `bounds`= the soft-clamp bounds, drawn as dashed red lines. For `:phi` they are `(0, 1)`, the
    SUPPORT: 1.0 is the POOLED LIMIT (contacts constant across the window, the parameter unidentified),
    not a clamp — `RHO_TIME_BOUNDS` and the temporal soft-clamp are dead on the AR(1) path. Label it
    as such wherever it is drawn.
  • `sym`   = the chain column, so `ess_bulk`/`rhat` can be looked up without a second mapping.

⚠ THE BAND IS ONLY HONEST FOR CHAINS FITTED UNDER `cfg`'s PRIORS. `contacts_label` encodes no prior
at all, so pointing a figure at a retained generation (`CONTACTS_TOKEN_AR1` spans Uniform → Beta(2,2)
→ Beta(3,3) for φ, and a different `gp_len_prior` before `-m32`) would draw today's band against
another generation's draws with nothing raised. Hence `label`, which every figure PRINTS, so a
mismatch is at least visible. (`plot_tau_over_weeks` hard-codes its σ for the mirror-image reason:
ITS chains are legacy, so `cfg` would be the wrong source.)
"""
function _prior_hyper(cfg, par::Symbol; q = (0.05, 0.5, 0.95))
    # Push the Normal quantiles through the model's own clamp+exp, rather than using LogNormal's
    # quantiles, so the band IS the prior the sampler saw.
    _band(μ, σ, lo, hi) = Tuple(exp(_softclamp(μ + quantile(Normal(), p) * σ, lo, hi)) for p in q)
    if par === :rho_diag || par === :rho_gap
        μ, σ = cfg.gp_len_prior                      # ONE prior SHARED by both spatial length-scales
        return (; dist = LogNormal(μ, σ), band = _band(μ, σ, RHO_BOUNDS...),
                  bounds = exp.(RHO_BOUNDS), logscale = true,
                  sym = par === :rho_diag ? :log_rho_diag : :log_rho_gap,
                  pretty = par === :rho_diag ? "ρ_diag" : "ρ_gap",
                  # ⚠ ASCII `^2`, not the superscript ²: GR has no glyph for U+00B2/U+00B3/U+207B and
                  # prints "glyph missing from current font" while dropping it from the PNG. Greek
                  # (ρ η σ φ) and √ ÷ DO render — it is only the superscripts that are missing.
                  label = "log ρ ~ N(log $(round(exp(μ); digits = 1)), $(σ)^2)")
    elseif par === :eta || par === :sigma_c
        μ, σ = par === :eta ? cfg.gp_scale_prior : cfg.gp_level_scale_prior
        return (; dist = LogNormal(μ, σ), band = _band(μ, σ, -3.0, 2.0),
                  bounds = (exp(-3.0), exp(2.0)), logscale = true,
                  sym = par === :eta ? :log_eta : :log_sigma_c,
                  pretty = par === :eta ? "η" : "σ_c",
                  label = "log $(par === :eta ? "η" : "σ_c") ~ N($μ, $(σ)^2)")   # ASCII ^2 — see above
    elseif par === :phi
        a, b = cfg.ar1_phi_prior
        d = Beta(a, b)
        return (; dist = d, band = Tuple(quantile(d, p) for p in q), bounds = (0.0, 1.0),
                  logscale = false, sym = :phi_time, pretty = "φ",
                  label = "φ ~ Beta($a, $b)")
    end
    error("_prior_hyper: no prior registered for `$par`")
end

# The five reported hyperparameters, in the order every §7 table and figure uses: the three smoothing
# parameters first (the section's subject), then the two amplitudes. These are BOTH the `_prior_hyper`
# keys AND the `reconstruct_gp_hyper_draws` field names — deliberately kept identical so consumers can
# `getproperty(r, par)` without a lookup table. Keep them in step if either side gains a parameter.
const GP_HYPER_PARS = (:rho_diag, :rho_gap, :phi, :eta, :sigma_c)

"""
    collect_gp_hyper(dms, origin, cfg; grid, hz, contacts, save_dir)
        -> Dict{Tuple{String,Int},NamedTuple}

`reconstruct_gp_hyper_draws` over `dms × hz` at ONE origin, keyed `(degree_label, h)` — the store the
§7 table and both §7 figures consume, so the chains are deserialised once instead of three times
(`collect_transmission_structure`'s role in 9j, one origin wide instead of 63).

Keyed by DEGREE LABEL, not by `"<degree>|<ngm>"`, because Stage 1 is NGM-INDEPENDENT: there are TWO
chains per horizon, not four, and making that structural stops 10j's four combos turning into four
identical loads of the same file. Missing chains are warned about and simply absent from the store
(the `isfile || return nothing` contract), so consumers iterate `keys(store)` and never assume 8
entries.
"""
function collect_gp_hyper(dms, origin::Date, cfg; grid,
                          hz = collect(cfg.horizons),
                          contacts::AbstractString = contacts_label(cfg),
                          save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    store = Dict{Tuple{String,Int},NamedTuple}()
    for dm in dms, h in hz
        deg = degree_label(dm)
        # The `|mean` is a FORMALITY: `stage1_chain_path` drops the ngm token entirely.
        r = reconstruct_gp_hyper_draws("$(deg)|mean", origin, h;
                                       grid = grid, contacts = contacts, save_dir = save_dir)
        if r === nothing
            @warn "no Stage-1 chain for the GP hyperparameters" deg origin h contacts
        else
            store[(deg, h)] = r
        end
    end
    return store
end

"""
    gp_hyper_table(store, cfg; contacts, csv_path=nothing) -> DataFrame

§7's summary: one row per (degree family × horizon × hyperparameter) — 5 × 8 = 40 rows at the standard
two-family / h1–h4 store — with the posterior beside the PRIOR it was fitted under.

Columns: `degree, h, Tn, ndraws, sampler, phi_init_scale, divergences, parameter, median, q05, q95,
prior_q05, prior_med, prior_q95, prior_pctl, ess_bulk, rhat, contacts`.

`prior_pctl = cdf(prior, posterior_median)` gives ONE uniform semantics across the LogNormal and Beta
priors alike: 0.5 means the data moved nothing, `< 0.05` means the posterior median has been pushed
into the prior's lower tail. That is the numeric form of the standing question in `gp_len_prior`'s
docstring — "if the posterior piles up against the LOWER edge, the data are disagreeing with the
assumed smoothness" — whose earlier Pathfinder answer was quoted as −2.7σ / −4.2σ.

Quantiles go through `_fmed`/`_fq` (8j_viz_utils.jl) because ONE of the two temporal fields is
all-NaN by construction (see `reconstruct_gp_hyper_draws`' fork): a `-m32t` chain has no φ, and a
plain `median` would throw rather than report a blank row.

`contacts` is a COLUMN as well as part of the filename: this table IS a generation's measurement, and
the token is the only thing that distinguishes two of them at one origin (`-lc0` changed no parameter
name and no dimension). Same reasoning as `8j_s1_*` recording `ad_backend`.

⚠ Writes a CSV only when `csv_path` is given — deliberately unlike this file's `make_*_fig` helpers,
which `savefig` internally. A figure's PNG is a by-product of displaying it; a DataFrame IS the
artefact, so persisting it is the notebook's decision and a check script can exercise this without
writing into `res/`.

⚠ RAW SAMPLED PARAMETERS ONLY. Derived readings — `ρ/√2` (the age-DIFFERENCE length-scale, since ρ
lives on the √2-rotated coordinates), φ's e-folding time `−1/log φ`, the end-to-end correlation
`φ^(Tn−1)`, and the field's true marginal SD `η·sqrt(diag(M·Kp·M))` (η is not it, since `-s0`) — are
deliberately NOT rows here (2026-08-10 scope decision); `Tn` is carried per row so `φ^(Tn−1)` is one
step away, and the two units caveats live in §7's markdown.
"""
function gp_hyper_table(store, cfg; contacts::AbstractString = contacts_label(cfg),
                        csv_path::Union{Nothing,AbstractString} = nothing)
    rows = NamedTuple[]
    for (deg, h) in sort(collect(keys(store)))
        r = store[(deg, h)]
        ndiv = (r.diag === missing || !hasproperty(r.diag, :divergences)) ? missing : r.diag.divergences
        for par in GP_HYPER_PARS
            pri = _prior_hyper(cfg, par)
            v   = getproperty(r, par)          # `GP_HYPER_PARS` == the reader's field names
            med = _fmed(v)
            # `prior_pctl` is undefined for a parameter this generation did not sample (all-NaN).
            # Read off the UNCLAMPED prior: `_softclamp` is monotone, so the only error is the clamp
            # shift itself — measured ≤2.3e-3 for the amplitudes, ≤6.4e-6 for ρ (see `_prior_hyper`).
            pctl = isfinite(med) ? cdf(pri.dist, med) : missing
            push!(rows, (; degree = deg, h = h, Tn = r.Tn, ndraws = r.ndraws,
                           sampler = string(r.sampler), phi_init_scale = r.phi_init_scale,
                           divergences = ndiv, parameter = pri.pretty,
                           median = med, q05 = _fq(v, 0.05), q95 = _fq(v, 0.95),
                           prior_q05 = pri.band[1], prior_med = pri.band[2], prior_q95 = pri.band[3],
                           prior_pctl = pctl,
                           ess_bulk = r.ess_bulk === missing ? missing :
                                      get(r.ess_bulk, pri.sym, missing),
                           rhat = r.rhat === missing ? missing : get(r.rhat, pri.sym, missing),
                           contacts = contacts))
        end
    end
    tbl = DataFrame(rows)
    csv_path === nothing || CSV.write(csv_path, tbl)
    return tbl
end

# Which of the four 10j model labels a degree family owns, so a family keeps the colour it has in
# §1/§2b/§3. Stage 1 is NGM-independent, so the `|mean` entry stands for the whole family.
# `labels4 === nothing` (a caller with no model list to hand) falls back to the per-family default,
# which is the SAME colour the standard `labels4`/`model_cols` pairing gives — keep the two in step.
const _GP_FAMILY_FALLBACK = Dict("unweighted-negbin" => :steelblue, "weighted-hweibull" => :seagreen)
function _gp_family_colour(deg::AbstractString, labels4, model_cols)
    (labels4 === nothing || model_cols === nothing) && return get(_GP_FAMILY_FALLBACK, deg, :black)
    i = findfirst(==("$(deg)|mean"), labels4)
    return i === nothing ? get(_GP_FAMILY_FALLBACK, deg, :black) : model_cols[i]
end

"""
    plot_gp_hyper_horizons(store, origin, cfg, labels4, model_cols; hz, res_dir="../res")
        -> Plots.Plot | nothing

§7a — the five Stage-1 GP hyperparameters across the horizon RE-FITS h1..h4 at one origin, in THREE
panels, both degree families overlaid.

**Three panels because the units do not mix** — the split `plot_lengthscales` makes in 9j, on the same
parameters, for the same reason: ρ in (rotated) age-years, φ dimensionless on (0,1), the amplitudes in
log-scale SD. φ drawn on a 0–50 age-year axis is an invisible line along the bottom.

  1. spatial (`:log10`) — ρ_diag solid, ρ_gap dashed, 90% ribbons;
  2. temporal (`ylims = (0,1)`) — φ, with **φ = 1 marked**: the pooled limit at which contacts are
     constant across the window and the parameter stops being identified, so a chain sitting ON the
     boundary reads as such rather than merely as "near the top". The weighted path has reached that
     limit under every temporal parameterisation tried (see `ar1_phi_prior`);
  3. amplitude (`:log10`) — η solid, σ_c dashed, with the `[e⁻³, e²]` soft-clamp bounds drawn, because
     BOTH pinning at the e⁻³ floor together with φ ≈ 1 is the measured signature of the Uniform-φ
     divergence mode. `plot_gamma` establishes the idiom: make clamp compression visible on the panel
     instead of leaving it to be inferred from the draws.

**Both families in ONE figure**, unlike §2b/§2c which split by family — those split because μ is a
per-capita COUNT mean for negbin and a duration-WEIGHTED mean for hweibull. These five are on
identical scales for both families, and the negbin-vs-hweibull contrast in φ is the section's
headline, so separating them would put the two numbers you must compare on different pages.

⚠ y-LIMITS ARE SET EXPLICITLY from the posteriors and the prior band, NOT from the clamp bounds, and
that takes code: **`hline!` EXPANDS a panel's limits to include its value — it is not clipped by
default.** Measured on the first render, drawing ρ's `[0.5, 500]` clamp without explicit `ylims`
stretched the axis over three decades and squashed every median into the middle one. `plot_gamma`
widens to include its bounds and can afford to (γ_SAR genuinely spans four decades); ρ cannot. With
limits pinned, the bound lines are clipped and so become visible exactly when a posterior runs out to
one — which is the point of drawing them — and the numbers are in the legend either way.

The grey band is the prior 90% from `_prior_hyper`, whose `label` is printed so a prior/chain mismatch
is visible (see that function's warning); the plot title states the SAMPLER, because a Pathfinder φ is
partly a function of its initialisation. Returns `nothing` on an empty store.
"""
function plot_gp_hyper_horizons(store, origin::Date, cfg, labels4, model_cols;
                                hz = collect(cfg.horizons), res_dir::AbstractString = "../res")
    isempty(store) && (@warn "plot_gp_hyper_horizons: empty store — nothing to draw"; return nothing)
    degs = sort(unique(first.(collect(keys(store)))))
    # Median/90% per (family, horizon) for one field, NaN where that chain is absent or the parameter
    # belongs to the other temporal generation.
    function series(deg, field)
        med = Float64[]; lo = Float64[]; hi = Float64[]
        for h in hz
            r = get(store, (deg, h), nothing)
            v = r === nothing ? [NaN] : getproperty(r, field)
            push!(med, _fmed(v)); push!(lo, _fq(v, 0.05)); push!(hi, _fq(v, 0.95))
        end
        return med, lo, hi
    end
    xt   = (hz, ["h$h" for h in hz])                   # integer horizons, as `make_mu_horizon_fig`
    xlim = (first(hz) - 0.25, last(hz) + 0.25)         # so a single-horizon store (13j) still renders
    samplers = join(sort(unique(string(store[k].sampler) for k in keys(store))), "/")

    # y-LIMITS MUST BE SET EXPLICITLY, from the posteriors and the prior band only. `hline!` EXPANDS a
    # panel's limits to include its value — it does NOT get clipped by default — so drawing the ρ
    # soft-clamp bounds without this stretched the axis over 0.5–500 and squashed every median into the
    # middle decade (measured on the first render). With explicit limits the bound lines are clipped and
    # so become visible exactly when a posterior runs out to one, which is the point of drawing them.
    function _lims(pri, fields; pad = 1.35)
        vs = Float64[pri.band[1], pri.band[3]]
        for deg in degs, f in fields, h in hz
            r = get(store, (deg, h), nothing); r === nothing && continue
            v = getproperty(r, f)
            append!(vs, (_fq(v, 0.05), _fq(v, 0.95)))
        end
        vs = filter(x -> isfinite(x) && x > 0, vs)
        isempty(vs) && return :auto
        return (minimum(vs) / pad, maximum(vs) * pad)
    end

    # ---- panel 1: the two spatial length-scales -------------------------------------------------
    pri_r = _prior_hyper(cfg, :rho_diag)               # ONE prior shared by ρ_diag and ρ_gap
    ps = plot(; title = "spatial GP length-scales", titlefontsize = 9, xlabel = "horizon (re-fit)",
              ylabel = "ρ (rotated age-yrs; ÷√2 for age difference)", yscale = :log10,
              ylims = _lims(pri_r, (:rho_diag, :rho_gap)),
              xticks = xt, xlims = xlim, legend = :outertop, legendfontsize = 6, legendcolumns = 2)
    # Prior band FIRST as a flat ribbon (`plot_gamma`'s pattern), so the posteriors draw over it.
    plot!(ps, [first(hz), last(hz)], fill(pri_r.band[2], 2); ls = :dot, color = :grey40, lw = 1,
          ribbon = (fill(pri_r.band[2] - pri_r.band[1], 2), fill(pri_r.band[3] - pri_r.band[2], 2)),
          fillcolor = :grey60, fillalpha = 0.10, label = "prior 90%: $(pri_r.label)")
    for deg in degs, (field, lsty, nm) in ((:rho_diag, :solid, "ρ_diag"), (:rho_gap, :dash, "ρ_gap"))
        med, lo, hi = series(deg, field)
        all(isnan, med) && continue
        plot!(ps, hz, med; lw = 1.8, ls = lsty, marker = :circle, ms = 3,
              color = _gp_family_colour(deg, labels4, model_cols),
              ribbon = (med .- lo, hi .- med), fillalpha = 0.12, label = "$(nm) — $(deg)")
    end
    hline!(ps, collect(pri_r.bounds); ls = :dash, color = :red, lw = 1,
           label = "softclamp [$(round(pri_r.bounds[1]; digits = 2)), $(Int(round(pri_r.bounds[2])))]")

    # ---- panel 2: the AR(1) temporal coefficient ------------------------------------------------
    pri_p = _prior_hyper(cfg, :phi)
    # (0, 1.03), not (0, 1): φ = 1 is the whole point of this panel and on a hard (0,1) axis its rule
    # lands ON the frame, where a posterior sitting at 0.993 is indistinguishable from one at 1.000.
    pt = plot(; title = "AR(1) temporal coefficient", titlefontsize = 9, xlabel = "horizon (re-fit)",
              ylabel = "φ (lag-1 correlation between weeks)", ylims = (0, 1.03),
              xticks = xt, xlims = xlim, legend = :outertop, legendfontsize = 6, legendcolumns = 2)
    plot!(pt, [first(hz), last(hz)], fill(pri_p.band[2], 2); ls = :dot, color = :grey40, lw = 1,
          ribbon = (fill(pri_p.band[2] - pri_p.band[1], 2), fill(pri_p.band[3] - pri_p.band[2], 2)),
          fillcolor = :grey60, fillalpha = 0.10, label = "prior 90%: $(pri_p.label)")
    for deg in degs
        med, lo, hi = series(deg, :phi)
        all(isnan, med) && continue                    # a Matérn-temporal (`-m32t`) chain has no φ
        plot!(pt, hz, med; lw = 1.8, marker = :circle, ms = 3,
              color = _gp_family_colour(deg, labels4, model_cols),
              ribbon = (med .- lo, hi .- med), fillalpha = 0.12, label = "φ — $(deg)")
    end
    # φ = 1: the pooled limit, NOT a clamp (φ ∈ (0,1) by construction on this path).
    hline!(pt, [1.0]; ls = :dash, color = :red, lw = 1, label = "φ = 1 (pooled limit)")

    # ---- panel 3: the two amplitudes ------------------------------------------------------------
    # η and σ_c are SEPARATE `cfg` fields (`gp_scale_prior`, `gp_level_scale_prior`) that happen to
    # share N(0, 0.5²) today — so read both and draw one band when they agree, two when they do not,
    # rather than silently showing η's band under a σ_c line.
    pri_e = _prior_hyper(cfg, :eta)
    pri_s = _prior_hyper(cfg, :sigma_c)
    pa = plot(; title = "GP amplitudes", titlefontsize = 9, xlabel = "horizon (re-fit)",
              ylabel = "amplitude (log-scale SD)", yscale = :log10,
              ylims = _lims(pri_e, (:eta, :sigma_c)),   # explicit, so the clamp lines are CLIPPED
              xticks = xt, xlims = xlim, legend = :outertop, legendfontsize = 6, legendcolumns = 2)
    amp_priors = pri_e.band == pri_s.band ?
                 ((pri_e, "prior 90% (η, σ_c): $(pri_e.label)", :dot),) :
                 ((pri_e, "prior 90% η: $(pri_e.label)", :dot),
                  (pri_s, "prior 90% σ_c: $(pri_s.label)", :dashdot))
    for (pri, lab, lsty) in amp_priors
        plot!(pa, [first(hz), last(hz)], fill(pri.band[2], 2); ls = lsty, color = :grey40, lw = 1,
              ribbon = (fill(pri.band[2] - pri.band[1], 2), fill(pri.band[3] - pri.band[2], 2)),
              fillcolor = :grey60, fillalpha = 0.10, label = lab)
    end
    for deg in degs, (field, lsty, nm) in ((:eta, :solid, "η"), (:sigma_c, :dash, "σ_c"))
        med, lo, hi = series(deg, field)
        all(isnan, med) && continue
        plot!(pa, hz, med; lw = 1.8, ls = lsty, marker = :circle, ms = 3,
              color = _gp_family_colour(deg, labels4, model_cols),
              ribbon = (med .- lo, hi .- med), fillalpha = 0.12, label = "$(nm) — $(deg)")
    end
    hline!(pa, collect(pri_e.bounds); ls = :dash, color = :red, lw = 1,
           label = "softclamp [exp(-3), exp(2)]")   # ASCII: GR has no ⁻³/² glyph, see `_prior_hyper`

    fig = plot(ps, pt, pa; layout = (1, 3), size = (1560, 520),
               left_margin = 7Plots.mm, bottom_margin = 9Plots.mm,
               plot_title = "10j — Stage-1 GP hyperparameters across horizon re-fits, origin " *
                            "$(origin) (median + 90%, $(samplers))",
               plot_titlefontsize = 10)
    savefig(fig, joinpath(res_dir, "10j_gp_hyper_horizons_$(origin).png"))
    return fig
end

"""
    plot_gp_hyper_marginals(store, origin, cfg; h=1, res_dir="../res") -> Plots.Plot | nothing

§7b — prior vs posterior MARGINALS of the five hyperparameters at ONE horizon: a `nfamilies × 5` grid,
rows = degree family, columns = ρ_diag, ρ_gap, φ, η, σ_c.

This is the panel §7's two documented questions are actually about, both being questions about
marginal SHAPE against an informative prior rather than about a trend across horizons:

  • `gp_len_prior` is deliberately informative and deliberately in tension with the 2026-08-05
    Pathfinder survey (ρ_diag ≈ 7.9, ρ_gap ≈ 4.65 — −2.7σ / −4.2σ under it). Its standing instruction
    is to bring the centre down if the posterior piles up on the LOWER edge.
  • φ's boundary: whether the mass is against 1.0, and for which degree family.

Each panel title carries the posterior median and the prior percentile at it, so the figure is
self-contained.

⚠ **HISTOGRAM, NOT A KDE, AND THAT IS LOAD-BEARING FOR φ.** φ's posterior can sit hard against 1.0
(measured medians 0.990–0.994 on the weighted path); a kernel density smears mass PAST the boundary
and so UNDERSTATES exactly the pile-up this panel exists to show. φ is binned on an explicit
`range(0, 1; length = 51)` for the same reason. The four log-normal parameters are binned in LOG space
on a `:log10` axis — the scale they are constructed on (`exp` of a Normal) and where their prior is
symmetric — and `normalize = :pdf` makes the bar heights directly comparable with the overlaid prior
density in the same units.

Grey dotted curve = the prior density of the reported quantity (`_prior_hyper(...).dist`); grey
dash-dot line = the prior median; red dashed = the soft-clamp bounds, with φ's single line at 1.0
relabelled the pooled limit. A parameter this generation did not sample (φ on a `-m32t` chain) gets an
annotated empty panel rather than a crash.
"""
function plot_gp_hyper_marginals(store, origin::Date, cfg; h::Integer = 1,
                                 labels4 = nothing, model_cols = nothing,
                                 res_dir::AbstractString = "../res")
    degs = sort(unique(d for (d, hh) in keys(store) if hh == h))
    if isempty(degs)
        @warn "plot_gp_hyper_marginals: no chains at this horizon" origin h
        return nothing
    end
    panels = Plots.Plot[]
    for (ri, deg) in enumerate(degs), (ci, par) in enumerate(GP_HYPER_PARS)
        r   = store[(deg, h)]
        pri = _prior_hyper(cfg, par)
        v   = filter(isfinite, getproperty(r, par))   # `GP_HYPER_PARS` == the reader's field names
        ttl = "$(deg) — $(pri.pretty)"
        if isempty(v)
            # The other temporal generation's parameter: state that rather than drawing nothing.
            push!(panels, plot(; framestyle = :box, title = "$(ttl): not sampled", titlefontsize = 7,
                               legend = false, xticks = false, yticks = false))
            continue
        end
        med  = median(v)
        pctl = cdf(pri.dist, med)
        # x-LIMITS EXPLICITLY, from the posterior and the prior 90% only — `vline!` EXPANDS the axis
        # rather than being clipped, so drawing the soft-clamp bounds without this stretched every ρ
        # panel across 0.5–500 and reduced the posterior to a spike (measured on the first render).
        xl = pri.logscale ?
             (min(minimum(v), pri.band[1]) / 1.6, max(maximum(v), pri.band[3]) * 1.6) :
             (0.0, 1.03)          # φ: the full support plus headroom, so the φ=1 rule is ON-panel
        pnl  = plot(; title = @sprintf("%s  med %.3g (prior pctl %.2f)", ttl, med, pctl),
                    titlefontsize = 7, legend = (ri == 1 && ci == 1 ? :topright : false),
                    legendfontsize = 5, xlabel = pri.pretty, ylabel = (ci == 1 ? "density" : ""),
                    xlims = xl, xscale = pri.logscale ? :log10 : :identity)
        # Bins on the parameter's OWN scale: log-spaced for the log-normal four (their axis is
        # :log10, so linear bins would render as wildly unequal bars), linear on (0,1) for φ.
        bins = if pri.logscale
            lo, hi = extrema(v)
            exp.(range(log(lo) - 1e-6, log(hi) + 1e-6; length = 41))
        else
            range(0, 1; length = 51)                   # the FULL support, so the boundary is on-panel
        end
        histogram!(pnl, v; bins = bins, normalize = :pdf, alpha = 0.55, lw = 0,
                   color = _gp_family_colour(deg, labels4, model_cols), label = "posterior")
        xs = pri.logscale ?
             exp.(range(log(min(minimum(v), pri.band[1])) - 0.3,
                        log(max(maximum(v), pri.band[3])) + 0.3; length = 300)) :
             range(0, 1; length = 300)
        plot!(pnl, xs, pdf.(pri.dist, xs); color = :grey30, ls = :dot, lw = 1.5,
              label = "prior $(pri.label)")
        vline!(pnl, [pri.band[2]]; color = :grey40, ls = :dashdot, lw = 1, label = "prior median")
        # Clamp bounds / support. Plots clips them, so they appear only when the posterior is near one.
        if par === :phi
            vline!(pnl, [1.0]; color = :red, ls = :dash, lw = 1, label = "φ = 1 (pooled limit)")
        else
            vline!(pnl, collect(pri.bounds); color = :red, ls = :dash, lw = 1, label = "softclamp")
        end
        push!(panels, pnl)
    end
    fig = plot(panels...; layout = (length(degs), length(GP_HYPER_PARS)),
               size = (350 * length(GP_HYPER_PARS), 310 * length(degs)),
               left_margin = 6Plots.mm, bottom_margin = 8Plots.mm,
               plot_title = "10j — Stage-1 GP hyperparameters: posterior vs prior, origin " *
                            "$(origin), h=$(h) (Tn = $(store[(degs[1], h)].Tn))",
               plot_titlefontsize = 10)
    savefig(fig, joinpath(res_dir, "10j_gp_hyper_marginals_h$(h)_$(origin).png"))
    return fig
end

"""
    _read_disp_chain(lbl, origin, h; weighted, week_index, contacts, save_dir, A)
        -> (; β, Z, τ, wk, lo, hi, D, legacy)  |  nothing

Chain reader behind `reconstruct_dispersion_draws`: pulls the dispersion latents out of a cached
Stage-1 chain for ONE week, without composing them.

Returns `β` (D×4 block-linear log-dispersion), the resolved week `wk` (`nothing` ⇒ pooled chain),
the soft-clamp bounds `lo`/`hi` for this degree family, and `D` draws.

**Reads BOTH cache generations**, told apart by whether the chain carries a per-cell random term:

- **current** (`legacy = false`): dispersion is block-linear × week only, `log_disp_{ij,t} = β[bl,t]`.
  `Z` and `τ` come back `nothing`.
- **legacy `-hd`** (`legacy = true`, `CONTACTS_TOKEN_HD` in `CONTACTS_SAVE_DIR_HD`): the flat
  non-centred hierarchy, `log_disp_{ij,t} = β[bl,t] + τ_t·z_{ij,t}`. `Z` is D×A², `τ` is length-D.

The legacy branch is kept ALIVE ON PURPOSE: it is what lets `plot_within_block_sd` compare "flat
hierarchy" against "no hierarchy" without refitting the old chains. `stage1_moment_draws` cannot do
this — it runs `generated_quantities` against the CURRENT `model_degree`, which has no `tau`/`z_k` to
read, so it would either error or silently re-draw them from the prior. Cross-generation work must go
through this mirror.

A `-rhs` horseshoe chain (the 2026-08-02 generation, identified by `c2`) is rejected with a warning:
that parameter space no longer exists in `model_degree`.
"""
function _read_disp_chain(lbl::AbstractString, origin::Date, h::Integer;
                          weighted::Bool, week_index::Union{Int,Nothing},
                          contacts::AbstractString, save_dir::AbstractString, A::Int)
    path = stage1_chain_path(lbl, origin, h; contacts = contacts, save_dir = save_dir)
    isfile(path) || return nothing
    chn = try
        load(path, "result")
    catch err
        @warn "could not load chain" path err
        return nothing
    end

    bbase  = weighted ? "log_kappa" : "log_k"              # block-linear dispersion param
    zbase  = weighted ? "z_kappa"   : "z_k"                # per-cell RE — legacy `-hd` chains only
    lo, hi = weighted ? (-4.3, 5.0) : (-4.0, 5.0)          # soft-clamp bounds MIRROR _cell_moments!
                                                           # (κ widened -3,3 → -4.3,5 on 2026-07-30
                                                           #  and RETAINED after the RE was removed;
                                                           #  −4.45 is the hard floor, see there.
                                                           #  Keep these two in lockstep or the
                                                           #  reconstruct-matches-model check fails)
    pnames = string.(names(chn, :parameters))
    D = size(chn, 1) * size(chn, 3)                        # posterior draws (iter × chains)

    # --- block dispersion β: D × 4 (block-linear bl). Read via the EXACT stored name — MCMCChains
    # prints matrix indices as "log_k[1, 2]" (SPACE after the comma), so a rebuilt "log_k[1,2]" misses.
    β  = Matrix{Float64}(undef, D, 4)
    bw = Regex("^" * bbase * raw"\[(\d+)\s*,\s*(\d+)\]$")
    bentries = [(parse(Int, m.captures[1]), parse(Int, m.captures[2]), n)
                for n in pnames for m in (match(bw, n),) if m !== nothing]
    isempty(bentries) && (@warn "no $bbase parameters in chain" path; return nothing)
    if maximum(e[1] for e in bentries) == 4                # per-week: [bl, t]
        Tn_chn = maximum(e[2] for e in bentries)
        wk = week_index === nothing ? Tn_chn : week_index
        # BOUNDS GUARD, and it is load-bearing since `-w8h` (2026-08-09) made Tn vary with both the
        # generation and the horizon (n_fit+h = 9..12 now, a flat 12 under `-t0-ar1`): with
        # `wk > Tn_chn` NO entry matches below, `β` is left as the
        # UNINITIALISED `Matrix{Float64}(undef, …)` it was allocated as, and the caller gets garbage
        # dispersion values with no error and no warning. `plot_within_block_sd` walks t = 1:12 across
        # both generations and relies on this returning `nothing` to stop at the shorter one.
        if !(1 <= wk <= Tn_chn)
            @warn "week_index $wk outside the chain's $Tn_chn weeks" path
            return nothing
        end
        for (bl, t, name) in bentries
            t == wk && (β[:, bl] = vec(Array(chn[Symbol(name)])))
        end
    else                                                   # pooled: 2×2 [row, col] → COLUMN-major
        wk = nothing                                       # bl = row + 2(col−1), matching `vec(log_*)`
        for (r, c, name) in bentries
            β[:, r + 2 * (c - 1)] = vec(Array(chn[Symbol(name)]))
        end
    end

    # A 2-D `name[p, t]` block for week `wk`, or the 1-D `name[p]` form in the pooled regime.
    function _cellmat(base)
        M = Matrix{Float64}(undef, D, A * A)
        if wk === nothing
            w1 = Regex("^" * base * raw"\[(\d+)\]$")
            hit = false
            for n in pnames
                m = match(w1, n); m === nothing && continue
                M[:, parse(Int, m.captures[1])] = vec(Array(chn[Symbol(n)])); hit = true
            end
            return hit ? M : nothing
        end
        w2 = Regex("^" * base * raw"\[(\d+)\s*,\s*(\d+)\]$")
        hit = false
        for n in pnames
            m = match(w2, n); m === nothing && continue
            parse(Int, m.captures[2]) == wk || continue
            M[:, parse(Int, m.captures[1])] = vec(Array(chn[Symbol(n)])); hit = true
        end
        return hit ? M : nothing
    end

    if any(n -> n == "c2", pnames)
        @warn "chain carries `c2` — this is a `-rhs` regularised-horseshoe chain, a parameter space \
               `model_degree` no longer has (removed 2026-08-02). Refit under the current token." path
        return nothing
    end

    Z = _cellmat(zbase)
    if Z === nothing                                       # current generation: no per-cell RE
        return (; β, Z = nothing, τ = nothing, wk, lo, hi, D, legacy = false)
    end
    # legacy `-hd`: τ_t per week (`tau[wk]`), or a bare scalar in the legacy pooled regime.
    τ = wk === nothing ? vec(Array(chn[:tau])) : vec(Array(chn[Symbol("tau[$wk]")]))
    return (; β, Z, τ, wk, lo, hi, D, legacy = true)
end

"""
    reconstruct_dispersion_draws(lbl, origin, h; weighted, cfg, grid, week_index=nothing,
                                 contacts, save_dir) -> ndraws × A × A  |  nothing

Load the cached Stage-1 chain for `(lbl, origin, h)` and rebuild the per-cell degree-model dispersion
for one week, once per posterior draw — the companion to `reconstruct_mu_draws` (same chain, same
draw order, so slice `d` pairs with μ's draw `d`).

Mirrors `_cell_moments!` (joint_model.jl §4.3) exactly — this reconstruct-matches-model invariant is
the standing rule for every viz mirror here:

    log_disp[i,j] = β[bl],                bl = 2(block_of(i)−1)+block_of(j)
    disp[i,j]     = exp(softclamp(log_disp[i,j], lo, hi))

`weighted` selects the parameter family and its soft-clamp bounds:
- Weibull (`weighted=true`):  `κ = exp(softclamp(·, −4.3, 5))`, from `log_kappa`
- NegBin  (`weighted=false`): `k = exp(softclamp(·, −4, 5))`, from `log_k`

⚠ **The result is CONSTANT WITHIN EACH BLOCK by construction**, since 2026-08-02 — all 4 (or 9, 12,
24) ordered cells of a child/adult block share one value. Flat quadrants in `plot_dispersion_cells`
are now the CORRECT reading, not the clamp-compression failure signature they were while the per-cell
random effect existed. The `ndraws × A × A` shape is kept (rather than reverting to `ndraws × 4`) so
callers keep indexing `[:, i, j]`, and so a legacy chain can be read into the same container.

For a legacy `-hd` chain (`CONTACTS_TOKEN_HD` + `CONTACTS_SAVE_DIR_HD`) the per-cell term `τ_t·z_{ij}`
IS applied, so those draws do vary within a block — that contrast is exactly what
`plot_within_block_sd` measures. `week_index` defaults to the last window week (the origin week the
NGM is frozen at). Returns `nothing` when the chain file is missing.
"""
function reconstruct_dispersion_draws(lbl::AbstractString, origin::Date, h::Integer;
                                      weighted::Bool, cfg, grid,
                                      week_index::Union{Int,Nothing} = nothing,
                                      contacts::AbstractString = contacts_label(cfg),
                                      save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    r = _read_disp_chain(lbl, origin, h; weighted = weighted, week_index = week_index,
                         contacts = contacts, save_dir = save_dir, A = grid.N)
    r === nothing && return nothing
    A = grid.N
    disp = Array{Float64,3}(undef, r.D, A, A)
    for i in 1:A, j in 1:A
        bl    = 2 * (block_of(i, cfg) - 1) + block_of(j, cfg)
        pcode = (i - 1) * A + j
        for d in 1:r.D
            δ = r.legacy ? r.τ[d] * r.Z[d, pcode] : 0.0    # current model has NO per-cell term
            disp[d, i, j] = exp(_softclamp(r.β[d, bl] + δ, r.lo, r.hi))
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
                              contacts::AbstractString = CONTACTS_TOKEN,
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

The dispersion random-effect **global** scale τ. Half-Normal, so it is stored untransformed: no
exp/softclamp mirror needed here, unlike `reconstruct_dispersion_draws`.

⚠ **LEGACY READER.** The current model has no dispersion random effect at all (removed 2026-08-02),
so there is no `tau` in a current chain and this returns `nothing` for one. It stays alive for the
`-hd` chains in `CONTACTS_SAVE_DIR_HD`, which stored `tau[t]` per week — pass `CONTACTS_TOKEN_HD`
**and** that `save_dir`. Returned as an `ndraws × Tn` matrix (`Tn = 1` for the legacy pooled regime),
so callers need not branch. Returns `nothing` if the chain is missing or carries no `tau`.
"""
function reconstruct_tau_draws(lbl::AbstractString, origin::Date, h::Integer;
                               contacts::AbstractString = CONTACTS_TOKEN,
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
`N⁺(0, 0.5)` prior band (grey) — the scale those `-hd` chains were actually fit under, hard-coded
here rather than read from `cfg`, whose τ₀ has since moved twice and is now per-family.

⚠ **LEGACY PANEL.** The current model has no dispersion random effect, so there is no τ to plot and
this returns `nothing` for a current chain. It is kept because it still reads the `-hd` chains in
`CONTACTS_SAVE_DIR_HD` (its defaults point there), and running it on those documents what was given
up when the RE was removed on 2026-08-02: how large the per-cell scale actually was, week by week,
and whether it varied at all.
"""
function plot_tau_over_weeks(lbl::AbstractString, origin::Date, cfg, weeks;
                             h::Integer = 1,
                             contacts::AbstractString = CONTACTS_TOKEN_HD,
                             save_dir::AbstractString = CONTACTS_SAVE_DIR_HD)
    τ = reconstruct_tau_draws(lbl, origin, h; contacts = contacts, save_dir = save_dir)
    τ === nothing && (@warn "no Stage-1 chain for τ panel" lbl origin contacts; return nothing)
    Tn = size(τ, 2)
    if Tn == 1
        @warn "plot_tau_over_weeks: τ is a scalar in this chain — no week axis to plot. This panel \
               only applies to the legacy `-hd` chains (CONTACTS_TOKEN_HD/CONTACTS_SAVE_DIR_HD)." lbl contacts
        return nothing
    end
    wks = length(weeks) == Tn ? collect(weeks) : collect(1:Tn)
    # `view(...)` function form, NOT the space-form `@view`: in an argument list with further args
    # the macro greedily swallows them (`quantile(@view τ[:, t], 0.05)` → "Invalid use of @view
    # macro"). Same trap the model's `_cell_moments!` call site documents.
    med = [median(view(τ, :, t)) for t in 1:Tn]
    lo  = [quantile(view(τ, :, t), 0.05) for t in 1:Tn]
    hi  = [quantile(view(τ, :, t), 0.95) for t in 1:Tn]
    # half-Normal N⁺(0,σ) prior quantiles: q(p) = σ·Φ⁻¹((1+p)/2). The `-hd` chains were fit under
    # the OLD prior scale 0.5, not the current cfg value — hard-code it so the band is honest.
    σp  = 0.5
    pri = (med = σp * 0.6744897501960817, lo = σp * 0.06270677794321385, hi = σp * 1.959963984540054)
    p = plot(; title = "$lbl — LEGACY per-week dispersion RE scale τ_t (origin $origin, h=$h)",
             titlefontsize = 9,
             xlabel = "window week", ylabel = "τ_t  (log-scale SD of the per-cell RE)",
             legend = :topright, legendfontsize = 6, xrotation = 45, ylims = (0, max(σp * 2.5, maximum(hi) * 1.1)))
    # Date-valued series FIRST — a leading hline!/hspan! locks a numeric axis and mangles date ticks.
    plot!(p, wks, med; lw = 2, marker = :circle, ms = 3, markerstrokewidth = 0,
          ribbon = (med .- lo, hi .- med), fillalpha = 0.18, label = "posterior median & 90%")
    plot!(p, wks, fill(pri.med, Tn); lw = 1.2, ls = :dot, color = :grey40, label = "prior median & 90% (σ=0.5)",
          ribbon = (fill(pri.med - pri.lo, Tn), fill(pri.hi - pri.med, Tn)),
          fillalpha = 0.08, fillcolor = :grey60)
    return p
end

"""
    plot_dispersion_cells(lbl, origin, cfg, grid; weighted, h=1, week_index=nothing, save_dir)
        -> Plots.Plot | nothing

7×7 heatmap of the per-cell dispersion (posterior median of `κ_{ij}` / `φ_{ij}`) at one week, with
the child/adult **block boundary** drawn on top.

⚠ **Four flat quadrants are the CORRECT result** for the current model. Dispersion is block-linear ×
week with no per-cell term, so every cell inside a child/adult block is identical by construction and
this panel is really a 2×2 read-out drawn on a 7×7 grid. It is kept because it is the clearest visual
confirmation of that, and because pointing it at the legacy `-hd` chains
(`save_dir = CONTACTS_SAVE_DIR_HD`, `contacts = CONTACTS_TOKEN_HD`) shows the within-quadrant
variation the per-cell RE used to produce — the before/after in one picture.

(Between 2026-07-30 and 2026-08-02 the reading was the opposite: flat quadrants then meant the RE
scale had collapsed to ~0. Don't carry that interpretation over.)
"""
function plot_dispersion_cells(lbl::AbstractString, origin::Date, cfg, grid;
                               weighted::Bool, h::Integer = 1,
                               week_index::Union{Int,Nothing} = nothing,
                               contacts::AbstractString = contacts_label(cfg),
                               save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    D = reconstruct_dispersion_draws(lbl, origin, h; weighted = weighted, cfg = cfg, grid = grid,
                                     week_index = week_index, contacts = contacts,
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
`pred_t = build_ngm(Cstar_m[t−smax+h], susc[d], inf[d], F[d], wd.antibody[:,t]; gamma_sar[d]) · Σ_s w[s]·wd.I_mean[:,t-s]`
using OBSERVED lags ⇒ one-step-ahead fitted mean (NOT the self-iterated forecast). Columns
`(smax+1):Tn` == the window's fit weeks; the `−smax+h` on `Cstar_m` is the `-w8h` window offset (the
contact window spans `n_fit + h` weeks ending at t₀+h, the infection window `n_fit + smax` ending at
t₀, and the renewal reads the contact window's last `n_fit` columns). The per-week `Cstar_m` is rebuilt from the Stage-1 chain
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
    # `-w8h` (2026-08-09): the CONTACT window is `n_fit + h` weeks `[t₀−n_fit+1 … t₀+h]`, the
    # INFECTION window `Tn` = n_fit + smax, so the renewal reads `Cstar`'s LAST `n_fit` columns:
    # index `t − smax + off` against `wd`'s week `t`, with `off = Tc − n_fit` == h. This MUST match
    # `model_transmission` exactly or the panel plots a model that was never fitted.
    Tc  = length(md[1].K1)
    off = Tc - cfg.n_fit
    (0 <= off <= maximum(cfg.horizons)) ||
        (@warn "fit-window fit: Stage-1 chain has $Tc contact weeks, expected n_fit + h = $(cfg.n_fit) + h"; return nothing)
    Cstar_by_m = [[contact_star(nb, md[m].K1[t], md[m].K2[t], md[m].G[t]) for t in 1:Tc]
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
            N = build_ngm(Cstar_by_m[m][t - cfg.smax + off], pooled.susc[d, :], pooled.inf[d, :],
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
    # `contacts` MUST come from the notebook's cfg. Omitting it falls through to the module-level
    # `CONTACTS_TOKEN`, which `framework.jl` builds from `FrameworkConfig()`'s LITERAL
    # `stage1_use_nuts = true` — i.e. always the `-nuts` generation, whatever cfg or ENV say. That
    # is how every μ figure below went silently blank against the Pathfinder grid (2026-08-08).
    μdraws = reconstruct_mu_draws(lbl, oc.origin, 1; week_index = oc.t_o_est, grid = grid,
                                  contacts = contacts_label(oc.cfg))
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
            # `contacts` from cfg, never the `CONTACTS_TOKEN` default — see make_agepair_fig.
            μd = reconstruct_mu_draws(lbl, origin, h; grid = grid,   # default week = forecast week t₀+h
                                      contacts = contacts_label(cfg))
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
chain at that chain's forecast week), this traces μ over TIME from one fit: the h`h_chain` chain's
per-week GP spans `[t₀−n_fit+1 … t₀+h_chain]` (`-w8h`, 2026-08-09), so at h4 that is the origin's 8
fit weeks ++ the 4 horizon weeks — 12 points with t₀ at position 8. (The `smax` renewal-lag weeks
that used to precede them are gone; a same-day intermediate, `-w8`, slid the window instead of
anchoring it and left this figure with only 8 points starting at t₀−3, which is what prompted the
correction.) `constant_contacts = false` is what makes μ per-week at all. It is the μ analogue of
§1's h4 in-sample fit.

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
    weeks = apd_h.weeks                       # n_fit + h_chain dates, [t₀−n_fit+1 … t₀+h_chain] (`-w8h`)
    Tn = length(weeks)
    xdate = week_mid.(weeks)

    # μ stats per NGM builder: reconstruct the SAME (lbl, h_chain) chain at EVERY window week t.
    stats = Dict{String,NTuple{3,Matrix{Float64}}}()          # ngm_label => (med, lo, hi), each Tn×ncell
    for nb in (MeanNGM(), NeighbourhoodDegreeNGM())
        lbl = string(degree_label(dm), "|", ngm_label(nb))
        med = fill(NaN, Tn, ncell); lo = copy(med); hi = copy(med)
        for t in 1:Tn
            # `contacts` from cfg, never the `CONTACTS_TOKEN` default — see make_agepair_fig.
            μd = reconstruct_mu_draws(lbl, origin, h_chain; week_index = t, grid = grid,  # h4 chain, week t
                                      contacts = contacts_label(cfg))
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
    # `contacts` from cfg, never the `CONTACTS_TOKEN` default — see make_agepair_fig.
    μdraws = reconstruct_mu_draws(lbl, oc.origin, 1; week_index = oc.t_o_est, grid = grid,
                                  contacts = contacts_label(oc.cfg))
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
    # `contacts` from cfg, never the `CONTACTS_TOKEN` default — see make_agepair_fig. (The κ call
    # below already threaded cfg through; this one did not, which is how the mismatch hid.)
    μdraws = reconstruct_mu_draws(lbl, oc.origin, 1; week_index = oc.t_o_est, grid = grid,
                                  contacts = contacts_label(oc.cfg))
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

§5 — age-specific RELATIVE susceptibility and infectivity (reference bin `cfg.ref_bin`, default
4 = "25-34", fixed = 1) for the
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
                            "(ref bin \"$(grid.LAB[cfg.ref_bin])\" = 1), median + 90%, origin $(origin) (h$(h))",
               plot_titlefontsize = 10)
    savefig(fig, joinpath(res_dir, "10j_susc_inf_$(origin).png"))
    return fig
end
