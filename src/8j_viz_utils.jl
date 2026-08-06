# 8j_viz_utils.jl — read-only visualisation helpers for the 8j preliminary forecast (two-stage cut).
#
# Two artefact families per (origin, horizon):
#   • Stage-1 GP chain  `8j_s1_<degree>_<contacts>_<origin>_h<h>.jld2` (key "result") — the
#     contact-degree GP latents (NGM-independent ⇒ NO ngm token). μ / ρ / dispersion reconstruct
#     from here (see `model_degree` in joint_model.jl §5/§6, and 10j_viz_utils.jl).
#   • Stage-2 pooled    `8j_s2_<degree>_<ngm>_<contacts>_<origin>_h<h>.jld2` (key "pooled") — the
#     100×100 pooled infection draws `(; gamma_sar, susc, inf, F, sigma_inf, post_index, Cstar_end)`
#     (see `fit_stage2_pooled`). susc/inf/γ_SAR/R read from here.
# `lbl = "<degree>|<ngm>"` splits into the two tokens.

"""
    _fmed(v) / _fq(v, q)

Finite-robust median / quantile: reduce over the FINITE entries of `v` only, returning `NaN` when the
whole vector is non-finite. The natural-scale pooled forecast is heavy-tailed — a few pathological
draws can be ±Inf — so a plain `median`/`quantile` throws (`quantiles undefined in presence of NaNs`).
These plot the well-behaved central line/band and leave a gap where a cell is wholly non-finite.
"""
_fmed(v) = (f = filter(isfinite, v); isempty(f) ? NaN : median(f))
_fq(v, q) = (f = filter(isfinite, v); isempty(f) ? NaN : quantile(f, q))

"""
    stage1_chain_path(lbl, origin, h; contacts, save_dir)

Cached Stage-1 GP-chain filepath for `lbl` (`"<degree>|<ngm>"`), `origin`, horizon `h`. The
ngm token is dropped (Stage 1 is NGM-independent), so both builders of a degree family share it.
"""
function stage1_chain_path(lbl::AbstractString, origin::Date, h::Integer;
                           contacts::AbstractString = CONTACTS_TOKEN,
                           save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    deg, _ = split(lbl, "|")
    joinpath(save_dir, "8j_s1_$(deg)_$(contacts)_$(origin)_h$(h).jld2")
end

"""
    stage2_pooled_path(lbl, origin, h; contacts, save_dir)

Cached Stage-2 pooled-result filepath for `lbl` (`"<degree>|<ngm>"`), `origin`, horizon `h`.
"""
function stage2_pooled_path(lbl::AbstractString, origin::Date, h::Integer;
                            contacts::AbstractString = CONTACTS_TOKEN,
                            save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    deg, ngm = split(lbl, "|")
    joinpath(save_dir, "8j_s2_$(deg)_$(ngm)_$(contacts)_$(origin)_h$(h).jld2")
end

"""
    _group_matrix(chn, sym) -> ndraws × K

Stack the bracket-indexed columns `sym[1] … sym[K]` of a chain into a draws×K matrix,
ordered by the bracketed index. Avoids depending on `MCMCChains.group` being exported.
"""
function _group_matrix(chn, sym::Symbol)
    nm = string(sym)
    cols = filter(p -> startswith(string(p), nm * "["), names(chn, :parameters))
    _idx(p) = parse(Int, match(r"\[(\d+)\]", string(p)).captures[1])
    cols = sort(cols; by = _idx)
    return reduce(hcat, [vec(Array(chn[c])) for c in cols])
end

"""
    load_transmission_draws(lbl, origin, h; contacts, save_dir) -> (; susc, inf, gamma_sar, rho…) | nothing

Load the two-stage artefacts for `(lbl, origin, h)` and return per-draw transmission structure.
The infection block (susc / inf / `gamma_sar`) comes from the **Stage-2 pooled** file — susc/inf are
the stored `N×A` pooled draws (relative to the reference bin `cfg.ref_bin`, default 4 = "25-34", = 1), `gamma_sar` the pooled per-contact
secondary-attack-rate draws (N = 10_000). The GP length-scales come from the **Stage-1 chain**:
`rho_diag`/`rho_gap[d] = exp(softclamp(log_rho_diag|log_rho_gap,…))`; `phi_time[d]` is the AR(1)
coefficient read CONSTRAINED from the chain (dimensionless — NOT a length-scale in weeks, so it
must not share an axis with the two spatial ρ; see `plot_lengthscales`)
(total-age, age-gap and — in the separable spatio-temporal regime — temporal directions;
`phi_time` is `NaN` for pooled chains). Note susc/inf/gamma_sar (pooled, ~10_000 draws) and the ρ
(Stage-1, ~200 draws) have different draw counts — they are consumed by separate figures.

Also returns the per-draw **generation-interval** log-parameters `w_mu`/`w_sigma` from the pooled
file (estimated since 2026-07-30, §3.1; stored POST-clamp, so
`gen_interval_pmf_log(w_mu[d], w_sigma[d])` reproduces that draw's `w` exactly). Convert to natural
scale with `gi_moments_days`. These come from Stage 2, so they are present for the NULL model too.

Returns `nothing` when the **Stage-2** file is missing/unreadable (skipped origin×combo). A missing
**Stage-1** chain is tolerated and yields all-`NaN` ρ: the NULL model (`no-contact|null`,
inst/6) has no contact fit at all, but its infection block is still worth plotting.
"""
function load_transmission_draws(lbl::AbstractString, origin::Date, h::Integer;
                                 contacts::AbstractString = CONTACTS_TOKEN,
                                 save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    s2p = stage2_pooled_path(lbl, origin, h; contacts = contacts, save_dir = save_dir)
    s1p = stage1_chain_path(lbl, origin, h; contacts = contacts, save_dir = save_dir)
    isfile(s2p) || return nothing
    pooled = try
        load(s2p, "pooled")
    catch err
        @warn "could not load the Stage-2 pooled artefact" s2p err
        return nothing
    end
    susc = pooled.susc; inf = pooled.inf; gamma_sar = pooled.gamma_sar   # Stage-2 pooled draws (N×A / N)
    chn = isfile(s1p) ? (try load(s1p, "result") catch err
                             @warn "could not load the Stage-1 chain; ρ set to NaN" s1p err
                             nothing
                         end) : nothing
    if chn === nothing                                                   # NULL model: no contact fit
        nan1 = fill(NaN, 1)
        return (; susc, inf, gamma_sar, F = pooled.F, rho_diag = nan1, rho_gap = nan1, phi_time = nan1,
                  w_mu = pooled.w_mu, w_sigma = pooled.w_sigma)          # GI/F are Stage-2, always there
    end
    s1names = string.(names(chn, :parameters))
    # `-diag` chains (2026-08-05, short-lived) LACK `log_rho_gap`: their spatial kernel smoothed the
    # matrix diagonal only with a single length-scale, so `rho_diag` means something different there
    # and reporting it beside current-token length-scales would silently mix generations. Same guard
    # as `reconstruct_mu_draws` — and note it was INVERTED when `-m32` restored the second ρ.
    # The token check catches the pre-`-diag` SQUARED-EXPONENTIAL generation, which is parametrically
    # identical to the current one (same names, same shapes) and so invisible to any `s1names` sniff.
    # ρ is on the same scale in both, but it means a different correlation function, so plotting the
    # two together on one axis would silently mix kernel families. Same fork as `reconstruct_mu_draws`.
    # `-ar1` (2026-08-06): the temporal parameter must be `phi_time`; a chain carrying
    # `log_rho_time` is pre-`-ar1` and its weeks-valued length-scale must not be read as a
    # correlation. Spatial guard (`log_rho_gap` + `-m32`) is unchanged.
    if !("log_rho_gap" in s1names) || !("phi_time" in s1names) || !occursin("-m32", contacts)
        @warn "Stage-1 chain is not the `-m32` generation (no `log_rho_gap`, or pre-`-m32` token); \
               ρ set to NaN" s1p contacts
        nan1 = fill(NaN, 1)
        return (; susc, inf, gamma_sar, F = pooled.F, rho_diag = nan1, rho_gap = nan1, phi_time = nan1,
                  w_mu = pooled.w_mu, w_sigma = pooled.w_sigma)
    end
    # `RHO_BOUNDS`/`RHO_TIME_BOUNDS` (framework.jl), NOT literals — this MUST track `model_degree`
    # or every reconstructed length-scale is silently wrong. See the constants' docstring.
    rho_diag = exp.(_softclamp.(vec(Array(chn[:log_rho_diag])), RHO_BOUNDS...))  # total-age dir, mirrors model
    rho_gap  = exp.(_softclamp.(vec(Array(chn[:log_rho_gap])),  RHO_BOUNDS...))  # age-gap dir, mirrors model
    # `-ar1`: the temporal parameter is the AR(1) coefficient φ ∈ (0,1), NOT a length-scale in
    # weeks — stored constrained, so read it directly (no exp/softclamp; RHO_TIME_BOUNDS is dead).
    # ⚠ It is returned as `phi_time`, not `rho_time`: the units differ from rho_diag/rho_gap
    # (dimensionless correlation vs age-years), so it must NOT share their axis in 9j.
    phi_time = ("phi_time" in s1names) ? vec(Array(chn[:phi_time])) : fill(NaN, length(rho_diag))
    w_mu = pooled.w_mu; w_sigma = pooled.w_sigma      # per-draw GI log-params (post-clamp)
    return (; susc, inf, gamma_sar, F = pooled.F, rho_diag, rho_gap, phi_time, w_mu, w_sigma)
end

"""
    supergroup_split(grid, ref_bin; base) -> (; groups, names)

Age super-groups for the ratio figures, with the **reference bin isolated as its own group**.
Starts from `base` (default `((1,2),(3,4,5),(6,7))` = 2-15 / 16-49 / 50+) and splits whichever
group contains `ref_bin` into up to three parts — the bins before it, `(ref_bin,)` alone, the bins
after — dropping the empty ones. For the 7 CIS bins and the default `cfg.ref_bin = 4` this gives
`((1,2),(3,),(4,),(5,),(6,7))` = 2-15 / 16-24 / **25-34** / 35-49 / 50+, so the group carrying the
model's gauge is a single bin that is identically 1 (see `collect_transmission_structure`).
`ref_bin = 1` ⇒ `((1,),(2,),(3,4,5),(6,7))`, and so on for any bin.

`names` mirror the `LAB` construction in `cis_age_grid` — open-ended (`"50+"`) when the group runs
to the last bin, else `"<LO[first]>-<HI[last]>"` — so a singleton group reproduces `grid.LAB[i]`
exactly. `groups` comes back in the tuple-of-tuples shape `aggregate_supergroups` takes as its
`groups` kwarg, ready to pass straight through.
"""
function supergroup_split(grid, ref_bin::Integer;
                          base = ((1, 2), (3, 4, 5), (6, 7)))
    groups = Tuple{Vararg{Int}}[]
    for g in base
        idx = collect(g)
        if ref_bin in idx
            for part in (filter(<(ref_bin), idx), [ref_bin], filter(>(ref_bin), idx))
                isempty(part) || push!(groups, Tuple(part))
            end
        else
            push!(groups, Tuple(idx))
        end
    end
    names = [last(g) == grid.N ? "$(grid.LO[first(g)])+" :
             "$(grid.LO[first(g)])-$(grid.HI[last(g)])" for g in groups]
    return (; groups = Tuple(groups), names)
end

"""
    aggregate_supergroups(V, POP; groups) -> ndraws × length(groups)

Population-weighted aggregation of a per-draw per-bin matrix `V` (`ndraws × A`) to the
super-groups in `groups` (default `((1,2),(3,4,5),(6,7))` = 2-15 / 16-49 / >50 for the
7 CIS bins; the ratio figures pass `supergroup_split(…).groups` instead). `POP` is the per-bin
population vector (`grid.POP`).
"""
function aggregate_supergroups(V::AbstractMatrix, POP::AbstractVector;
                               groups = ((1, 2), (3, 4, 5), (6, 7)))
    out = Matrix{Float64}(undef, size(V, 1), length(groups))
    for (g, idx) in enumerate(groups)
        cols = collect(idx)
        w = collect(float.(POP[cols]))
        out[:, g] = (V[:, cols] * w) ./ sum(w)
    end
    return out
end

"""
    pick_origins(origins; n=9) -> Vector{Date}

Evenly-spaced subset (≤ `n`) of `origins` for panel tiling.
"""
pick_origins(origins::AbstractVector{Date}; n::Int = 9) =
    origins[unique(round.(Int, range(1, length(origins); length = min(n, length(origins)))))]

"""
    reproduction_draws(dm, nb, wd, cfg, win; h=1, save_dir) -> Vector{Float64} | nothing

Per-pooled-draw reproduction number `R` for one forecast origin/model: the dominant eigenvalue of
the horizon-`h` next-generation matrix, exactly the NGM the forecast is frozen at (`Cstar_end` =
contacts at origin+h, antibody at the TARGET week origin+h). Reloads the cached **Stage-2 pooled**
file (NO re-fit); for each pooled draw `d` (from Stage-1 draw `m = post_index[d]`) builds
`N = build_ngm(Cstar_end[m], susc[d], inf[d], F[d], wd.antibody_fc[:,hi]; gamma_sar=gamma_sar[d])`
and takes `max real(eigvals(N))` (the NGM is nonnegative ⇒ its Perron root is real & positive).
The antibody column moved from the origin to origin+h on 2026-07-30 (§3.2) so this R describes the
same NGM `two_stage_forecast` actually uses — keep the two in step.
Returns `nothing` when the pooled file is missing/unreadable, so callers can leave a gap.
"""
function reproduction_draws(dm::ContactDegreeModel, nb::NGMBuilder, wd, cfg, win;
                            h::Integer = 1,
                            save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    lbl  = string(degree_label(dm), "|", ngm_label(nb))
    path = stage2_pooled_path(lbl, win.origin, h; contacts = contacts_label(cfg), save_dir = save_dir)
    isfile(path) || return nothing
    pooled = try
        load(path, "pooled")
    catch err
        @warn "could not load pooled result for reproduction number" path err
        return nothing
    end
    Np = length(pooled.gamma_sar)
    R = Vector{Float64}(undef, Np)
    hi = findfirst(==(h), collect(cfg.horizons))          # antibody column for THIS horizon
    ab_h = hi === nothing ? wd.antibody[:, end] : wd.antibody_fc[:, hi]
    for d in 1:Np
        m = pooled.post_index[d]
        N = build_ngm(pooled.Cstar_end[m], pooled.susc[d, :], pooled.inf[d, :],
                      pooled.F[d], ab_h; gamma_sar = pooled.gamma_sar[d])
        R[d] = maximum(real(eigvals(N)))
    end
    return R
end

"""
    contact_reproduction_draws(dm, nb, cfg, win; h=1, save_dir) -> Vector{Float64} | nothing

Per-Stage-1-draw **contact-only** reproduction number: the dominant eigenvalue of the origin-week
contact matrix `C*` ALONE — `ρ(C*) = max real(eigvals(Cstar_end[m]))` — dropping γ_SAR,
susceptibility, infectivity and antibody entirely (unlike `reproduction_draws`, which diagonalises
the full NGM). Reloads the cached **Stage-2 pooled** file read-only and uses its stored `Cstar_end`
(the same origin-week C* the forecast NGM is frozen at), so **no Stage-1 refit/reload** is needed.
`C*` is nonnegative ⇒ its Perron root is real & positive. Returns the `n_post` distinct Stage-1 C*
spectral radii (contact structure depends only on Stage 1), or `nothing` if the file is missing.
Feed to `relative_contact_reproduction_over_time` to normalise against a reference origin.
"""
function contact_reproduction_draws(dm::ContactDegreeModel, nb::NGMBuilder, cfg, win;
                                    h::Integer = 1,
                                    save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    lbl  = string(degree_label(dm), "|", ngm_label(nb))
    path = stage2_pooled_path(lbl, win.origin, h; contacts = contacts_label(cfg), save_dir = save_dir)
    isfile(path) || return nothing
    pooled = try
        load(path, "pooled")
    catch err
        @warn "could not load pooled result for contact reproduction number" path err
        return nothing
    end
    M = pooled.n_post
    ρ = Vector{Float64}(undef, M)
    for m in 1:M
        ρ[m] = maximum(real(eigvals(pooled.Cstar_end[m])))   # Perron root of the bare C*
    end
    return ρ
end
