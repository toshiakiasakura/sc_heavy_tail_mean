# 8j_viz_utils.jl — read-only visualisation helpers for the 8j preliminary forecast.
#
# Pulls fitted transmission parameters (susceptibility, infectivity, GP length-scale)
# out of the cached joint-model chains WITHOUT rebuilding the model: susc/inf/ρ are
# deterministic transforms of raw sampled columns (see `model_joint` in joint_model.jl),
# so they are reconstructed directly from the chain. Included from the 8j notebook.
#
# Chains are saved by `iterated_forecast`/`prefit_chains!` as
#   dt_intermediate/8j_chn_<degree>_<ngm>_<origin>_h<h>.jld2   (jldsave(path; result=chn))
# where <degree> = degree_label(dm), <ngm> = ngm_label(nb) — the same tokens the model
# label `lbl = "<degree>|<ngm>"` splits into.

"""
    chain_path(lbl, origin, h; save_dir)

Cached-chain filepath for model label `lbl` (`"<degree>|<ngm>"`), forecast `origin`
and horizon `h`. `lbl` joins the two tokens with `|`; the filename joins them with `_`.
"""
function chain_path(lbl::AbstractString, origin::Date, h::Integer;
                    contacts::AbstractString = "temporal-gsar",
                    save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    deg, ngm = split(lbl, "|")
    joinpath(save_dir, "8j_chn_$(deg)_$(ngm)_$(contacts)_$(origin)_h$(h).jld2")
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
    load_transmission_draws(lbl, origin, h; save_dir) -> (; susc, inf, gamma_sar, rho) | nothing

Load the cached chain for `(lbl, origin, h)` and return per-draw transmission draws
reconstructed from raw sampled columns (post-reparam: absolute `gamma_sar` + susc/inf RELATIVE to
reference bin 1 = 1): `gamma_sar[d] = exp(softclamp(log_gamma_sar,…))`,
`susc[d,:] = [1, exp(sig_s·z_s[1..A-1])]`, `inf[d,:] = [1, exp(sig_i·z_i[1..A-1])]`
(`ndraws × A` each, column 1 pinned to 1), and the GP length-scales `rho_diag`/`rho_gap`/`rho_time[d] =
exp(softclamp(log_rho_diag|log_rho_gap|log_rho_time,…))` (`ndraws` each; diagonal/total-age,
age-gap and — in the separable spatio-temporal regime — temporal directions; mirrors model).
`rho_time` is `NaN` for pooled chains (no `log_rho_time` parameter).
Returns `nothing` when the file is missing or unreadable (skipped origin×combo),
so callers can leave a gap.
"""
function load_transmission_draws(lbl::AbstractString, origin::Date, h::Integer;
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
    gamma_sar = exp.(_softclamp.(vec(Array(chn[:log_gamma_sar])), log(0.02), log(5.0)))  # absolute transmissibility, mirrors model
    sig_s = vec(Array(chn[:sig_s])); z_s = _group_matrix(chn, :z_s)   # z_s: ndraws × (A-1)
    sig_i = vec(Array(chn[:sig_i])); z_i = _group_matrix(chn, :z_i)
    susc = hcat(ones(size(z_s, 1)), exp.(sig_s .* z_s))  # ndraws × A, col 1 = 1 (relative to ref bin 1)
    inf  = hcat(ones(size(z_i, 1)), exp.(sig_i .* z_i))
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
    reproduction_draws(dm, nb, apd, wd, cfg, w; h=1, save_dir) -> Vector{Float64} | nothing

Per-draw reproduction number `R` for one forecast origin/model: the dominant eigenvalue
of the **origin-week** next-generation matrix, exactly the NGM the forecast is frozen at
(`posterior_forecast`/`iterated_forecast` use `q.Cstar[end]` with antibody held at the
origin). `apd` is the raw age-pair `AgePairData` for the origin window (as returned by
`prepare_degree_data`); it is turned into the model's degree stats with
`build_degree_stats(dm, apd, cfg)` — mirroring `iterated_forecast`. Reloads the cached `h`
chain via `fit_or_load_chain` (rebuilds the model so `generated_quantities` works), then
for each posterior draw builds
`N = build_ngm(q.Cstar[end], q.susc, q.inf, q.F, wd.antibody[:,end]; gamma_sar=q.gamma_sar)` and takes
`max real(eigvals(N))` (the NGM is nonnegative, so its Perron root is real & positive).

`h=1` is the direct 1-week-ahead fit (contacts observed up to the origin). Returns
`nothing` when the chain file is missing, so callers can leave a gap.
"""
function reproduction_draws(dm::ContactDegreeModel, nb::NGMBuilder, apd, wd, cfg, win;
                            h::Integer = 1,
                            save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    lbl  = string(degree_label(dm), "|", ngm_label(nb))
    path = chain_path(lbl, win.origin, h; contacts = contacts_label(cfg), save_dir = save_dir)
    isfile(path) || return nothing
    ds = build_degree_stats(dm, apd, cfg)                                 # raw AgePairData → model stats
    wpmf = gen_interval_pmf(cfg.gen_mean_days, cfg.gen_sd_days; smax = cfg.smax)  # model's 7th arg is the GI PMF
    res = try
        fit_or_load_chain(path, dm, nb, ds, wd, cfg, wpmf; use_nuts = false)  # reload chain, rebuild model
    catch err
        @warn "could not load chain for reproduction number" path err
        return nothing
    end
    gq = vec(generated_quantities(res.model, res.chn))                     # per-draw (; susc, inf, F, Cstar)
    R = Float64[]
    for q in gq
        q === nothing && continue
        N = build_ngm(q.Cstar[end], q.susc, q.inf, q.F, wd.antibody[:, end]; gamma_sar = q.gamma_sar)  # mirror posterior_forecast
        push!(R, maximum(real(eigvals(N))))
    end
    return R
end
