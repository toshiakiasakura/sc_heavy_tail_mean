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
                           contacts::AbstractString = "temporal-gsar-cut-sc",
                           save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    deg, _ = split(lbl, "|")
    joinpath(save_dir, "8j_s1_$(deg)_$(contacts)_$(origin)_h$(h).jld2")
end

"""
    stage2_pooled_path(lbl, origin, h; contacts, save_dir)

Cached Stage-2 pooled-result filepath for `lbl` (`"<degree>|<ngm>"`), `origin`, horizon `h`.
"""
function stage2_pooled_path(lbl::AbstractString, origin::Date, h::Integer;
                            contacts::AbstractString = "temporal-gsar-cut-sc",
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
the stored `N×A` pooled draws (relative to reference bin 1 = 1), `gamma_sar` the pooled per-contact
secondary-attack-rate draws (N = 10_000). The GP length-scales come from the **Stage-1 chain**:
`rho_diag`/`rho_gap`/`rho_time[d] = exp(softclamp(log_rho_diag|log_rho_gap|log_rho_time,…))`
(diagonal/total-age, age-gap and — in the separable spatio-temporal regime — temporal directions;
`rho_time` is `NaN` for pooled chains). Note susc/inf/gamma_sar (pooled, ~10_000 draws) and the ρ
(Stage-1, ~200 draws) have different draw counts — they are consumed by separate figures.
Returns `nothing` when the **Stage-2** file is missing/unreadable (skipped origin×combo). A missing
**Stage-1** chain is tolerated and yields all-`NaN` ρ: the NULL model (`no-contact|null`,
inst/6) has no contact fit at all, but its infection block is still worth plotting.
"""
function load_transmission_draws(lbl::AbstractString, origin::Date, h::Integer;
                                 contacts::AbstractString = "temporal-gsar-cut-sc",
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
        return (; susc, inf, gamma_sar, rho_diag = nan1, rho_gap = nan1, rho_time = nan1)
    end
    rho_diag = exp.(_softclamp.(vec(Array(chn[:log_rho_diag])), log(3.0), log(45.0)))  # total-age dir, mirrors model
    rho_gap  = exp.(_softclamp.(vec(Array(chn[:log_rho_gap])),  log(3.0), log(45.0)))  # age-gap dir
    rho_time = ("log_rho_time" in string.(names(chn, :parameters))) ?                  # temporal dir (weeks); NaN if pooled
        exp.(_softclamp.(vec(Array(chn[:log_rho_time])), log(0.5), log(26.0))) : fill(NaN, length(rho_diag))
    return (; susc, inf, gamma_sar, rho_diag, rho_gap, rho_time)
end

"""
    aggregate_supergroups(V, POP; groups) -> ndraws × length(groups)

Population-weighted aggregation of a per-draw per-bin matrix `V` (`ndraws × A`) to the
super-groups in `groups` (default `((1,2),(3,4,5),(6,7))` = 2-15 / 16-49 / >50 for the
7 CIS bins). `POP` is the per-bin population vector (`grid.POP`).
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
contacts at origin+h, antibody held at the origin). Reloads the cached **Stage-2 pooled** file
(NO re-fit); for each pooled draw `d` (from Stage-1 draw `m = post_index[d]`) builds
`N = build_ngm(Cstar_end[m], susc[d], inf[d], F[d], wd.antibody[:,end]; gamma_sar=gamma_sar[d])` and
takes `max real(eigvals(N))` (the NGM is nonnegative ⇒ its Perron root is real & positive).
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
    for d in 1:Np
        m = pooled.post_index[d]
        N = build_ngm(pooled.Cstar_end[m], pooled.susc[d, :], pooled.inf[d, :],
                      pooled.F[d], wd.antibody[:, end]; gamma_sar = pooled.gamma_sar[d])
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
