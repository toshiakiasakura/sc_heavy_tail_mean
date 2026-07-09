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
                    contacts::AbstractString = "weekly",
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
    load_transmission_draws(lbl, origin, h; save_dir) -> (; susc, inf, rho) | nothing

Load the cached chain for `(lbl, origin, h)` and return per-draw transmission draws
reconstructed from raw sampled columns:
`susc[d,a] = exp(mu_s + sig_s·z_s[a])`, `inf[d,b] = exp(mu_i + sig_i·z_i[b])`
(`ndraws × A` each), and `rho[d] = exp(softclamp(log_rho,…))` (`ndraws`; mirrors model).
Returns `nothing` when the file is missing or unreadable (skipped origin×combo),
so callers can leave a gap.
"""
function load_transmission_draws(lbl::AbstractString, origin::Date, h::Integer;
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
    mu_s = vec(Array(chn[:mu_s])); sig_s = vec(Array(chn[:sig_s]))
    z_s  = _group_matrix(chn, :z_s)                       # ndraws × A
    mu_i = vec(Array(chn[:mu_i])); sig_i = vec(Array(chn[:sig_i]))
    z_i  = _group_matrix(chn, :z_i)
    susc = exp.(mu_s .+ sig_s .* z_s)                     # ndraws × A
    inf  = exp.(mu_i .+ sig_i .* z_i)
    rho  = exp.(_softclamp.(vec(Array(chn[:log_rho])), log(3.0), log(45.0)))   # soft-bounded, mirrors model
    return (; susc, inf, rho)
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
