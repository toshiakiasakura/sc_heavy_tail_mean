# 12j_viz_utils.jl — Stage-1 NUTS chain convergence, visually and quantitatively.
#
# Read-only diagnostic: everything here reads a CACHED Stage-1 chain and never rebuilds or refits a
# model (same contract as 8j/10j/11j_viz_utils.jl). Companion to 12j_chain_convergence.ipynb.
#
# WHY THIS EXISTS. Stage 1 became a NUTS fit on 2026-08-05 (`cfg.stage1_use_nuts`), and the whole
# point of paying for NUTS over Pathfinder is that its draws are asymptotically exact — which is
# only worth anything if the chain actually converged. `_nuts_diagnostics` (joint_model.jl) gives
# the three numbers cheap enough to compute inside a 504-fit prefit (divergences,
# `frac_at_max_depth`, `min_ess`); this file is the expensive, per-origin follow-up.
#
# ⚠ ONE CHAIN PER FIT — so there is NO between-chain R̂, exactly as `_nuts_diagnostics` documents.
# What IS available, and is used throughout here, is **rank-normalised SPLIT-R̂** (Vehtari et al.
# 2021): MCMCDiagnosticTools defaults to `split_chains = 2`, so `rhat(chn)` on a single chain
# compares its own first and second halves and therefore does detect a chain that has not settled.
# It cannot detect two chains stuck in different modes — no single-chain diagnostic can. Read a
# clean split-R̂ here as "this chain is stationary", never as "the posterior was explored".
#
# Thresholds default to the MODERN rank-normalised ones (R̂ < 1.01, ESS ≥ 100 per chain; Vehtari
# 2021), which are stricter than the ESS > 200 / R̂ < 1.1 rule in `turing_utils.jl` — that rule
# predates this strand and was written for the multi-chain distribution fits.

using Statistics, StatsBase, DataFrames, Dates, Printf

# Pilot/production chains both live under a `dt_intermediate*` directory; the pilot generation from
# tmp/pilot_nuts_timing.jl is kept apart from the production grid so the two never mix.
const NUTS_PILOT_DIR = joinpath(@__DIR__, "..", "dt_intermediate_nuts_pilot")

"""
    load_nuts_chain(deg, origin, h; save_dir=NUTS_PILOT_DIR, contacts=CONTACTS_TOKEN)
        -> (; kept, full, n_adapt, diag, sampler, ad_backend, path)

Load a cached Stage-1 NUTS chain for degree label `deg` (`"unweighted-negbin"` /
`"weighted-hweibull"`), forecast `origin` and horizon `h`.

Looks for the **pilot** artefact `pilot_<deg>_<origin>_h<h>.jld2` first, then falls back to the
production `8j_s1_*` path via `stage1_chain_path`. Errors if neither exists — a missing chain must
fail loudly, not come back as blank panels (the 11j lesson).

Returns:
- `kept`    — the production draw set (the 500 draws a real fit keeps and Stage 2 imputes from)
- `full`    — the full instrumented chain INCLUDING warmup, or `nothing`. Only the pilot artefact
              carries this; production fits discard warmup inside `sample`, so warmup-phase figures
              (`plot_adaptation`) are simply unavailable for them.
- `n_adapt` — warmup length, recovered as `size(full,1) - size(kept,1)`; `missing` without `full`.

NOTE on the pilot's chain positions: position 1 of `full` is the INITIAL sample (`state.i = 0`),
which carries no HMC statistics at all — its `tree_depth`/`n_steps`/`step_size` are `missing`.
Every function here skips it. See tmp/pilot_nuts_timing.jl for why the adaptation index lags the
chain position by one.
"""
function load_nuts_chain(deg::AbstractString, origin::Date, h::Integer;
                         save_dir::AbstractString = NUTS_PILOT_DIR,
                         contacts::AbstractString = CONTACTS_TOKEN)
    pilot = joinpath(save_dir, "pilot_$(deg)_$(origin)_h$(h).jld2")
    path  = isfile(pilot) ? pilot : stage1_chain_path("$(deg)|mean", origin, h; contacts = contacts)
    isfile(path) || error("load_nuts_chain: no chain for $(deg) @ $(origin) h$(h).\n" *
                          "  looked for $(pilot)\n  and         $(path)")
    kept, full, diag, sampler, adb = jldopen(path, "r") do f
        (f["result"],
         haskey(f, "full_instrumented_chain") ? f["full_instrumented_chain"] : nothing,
         haskey(f, "diag")        ? f["diag"]        : missing,
         haskey(f, "sampler")     ? f["sampler"]     : missing,
         haskey(f, "ad_backend")  ? f["ad_backend"]  : missing)
    end
    n_adapt = full === nothing ? missing : size(full, 1) - size(kept, 1)
    return (; kept, full, n_adapt, diag, sampler, ad_backend = adb, path)
end

# `names(chn, section)` is a bare name_map lookup and throws KeyError on a missing section, so every
# internals access in this file goes through these two guards (the `_nuts_diagnostics` pattern).
_internals(chn) = :internals in MCMCChains.sections(chn) ? names(chn, :internals) : Symbol[]
function _icol(chn, s)
    Symbol(s) in _internals(chn) || return nothing
    v = vec(Array(chn[:, Symbol(s), :]))
    return collect(skipmissing(v))          # drops the pilot's statless initial sample
end

# A Chains' backing array is `Union{Missing,Float64}` WHOLESALE — even for parameter columns, which
# never actually contain a missing. Anything typed `AbstractVector{<:Real}` (StatsBase.autocor,
# StatsBase.ordinalrank) rejects that eltype, so parameter columns are read through here.
_pcol(chn, p) = collect(skipmissing(vec(Array(chn[:, Symbol(p), :]))))

"""
    param_group(p) -> String

Parameter block a chain column belongs to: `"z[3,7]" → "z"`. `model_degree` samples 6 scalars
(`c`, `log_eta`, `log_rho_diag`, `log_rho_gap`, `phi_time`, `log_sigma_c`) plus the blocks
`z_c` (Tn−1), `z` ((P−1)×Tn) and the dispersion (`log_k` for NegBin, `log_kappa`+`p0f` for
hurdle-Weibull) — i.e. `5 + 32·Tn` and `5 + 81·Tn` coordinates. **`Tn = n_fit + h` since `-w8h`, so
the count VARIES WITH THE HORIZON: 293/325/357/389 (NegBin) and 734/815/896/977 (hurdle-Weibull) at
h = 1..4.**

The count has moved five times and is NOT a reliable generation stamp: `-s0` dropped `z` from 28 to
27 rows (402/990 → 390/978 at Tn=12), `-diag` briefly removed the `log_rho_gap` scalar (→ 389/977),
`-m32` restored it (→ 390/978) while swapping both GP kernels to Matérn 3/2, `-t0` (2026-08-06) took
`z_c` from Tn to Tn−1 (→ 389/977 — ⚠ coinciding exactly with `-diag`'s, though the models are
unrelated), and `-w8h` (2026-08-09) took Tn from a flat `n_fit + smax` = 12 to `n_fit + h` = 9..12.
⚠ At h = 4 that is 389/977 again, i.e. **numerically identical to the `-t0-ar1` generation** — the
count cannot distinguish them at all, only the token can.
In-chain signals that DO discriminate: `log_rho_gap` present (absent under `-diag`), `z_c` = Tn−1
(Tn before `-t0`), and the `z` column count = Tn, which now VARIES ACROSS THE FOUR HORIZON CHAINS of
one origin (9/10/11/12) where it used to be a constant 12. `-lc0`, which landed with `-w8h`, changes
no name and no dimension at all — only the token records it. Archived pilot
chains: `dt_intermediate_nuts_pilot_pre_s0/` has 28 `z` rows, `dt_intermediate_nuts_pilot_diag/` has
27 rows and NO `log_rho_gap`. Per-parameter tables are unreadable at this size, so almost everything
here is reported BY GROUP.
"""
param_group(p) = String(split(String(Symbol(p)), "[")[1])

"""
    convergence_table(chn) -> DataFrame

Per-parameter convergence statistics: `parameter`, `group`, `mean`, `std`, `mcse`, `ess_bulk`,
`ess_tail`, `rhat`. One row per chain column.

`ess_bulk`/`ess_tail` are the rank-normalised bulk and tail effective sample sizes, and `rhat` is
rank-normalised split-R̂ with `kind = :rank` (the max of its bulk and tail variants) — all straight
from `MCMCChains.summarystats`, which computes exactly this set. Bulk and tail are reported
separately on purpose: bulk governs the posterior mean, tail governs the quantiles, and it is the
TAIL that the forecast's 90% intervals and the WIS score depend on.
"""
function convergence_table(chn)
    nt  = summarystats(chn).nt
    tbl = DataFrame(parameter = String.(String.(Symbol.(nt.parameters))))
    tbl.group = param_group.(tbl.parameter)
    for k in (:mean, :std, :mcse, :ess_bulk, :ess_tail, :rhat)
        tbl[!, k] = haskey(nt, k) ? collect(nt[k]) : fill(missing, nrow(tbl))
    end
    return tbl
end

"""
    convergence_summary(tbl; ess_min=100, rhat_max=1.01) -> DataFrame

Collapse `convergence_table` to one row per parameter group: block size, the WORST (minimum) bulk
and tail ESS, the worst split-R̂, and how many coordinates fail each threshold.

Defaults are the rank-normalised recommendations (Vehtari 2021): `rhat_max = 1.01`, and
`ess_min = 100` per chain — the level at which R̂ and the ESS estimates themselves become reliable.
Reporting the worst rather than the mean is deliberate: a block of 324 coordinates whose median ESS
is 400 is not converged if one coordinate sits at 5.
"""
function convergence_summary(tbl::DataFrame; ess_min::Real = 100, rhat_max::Real = 1.01)
    _min(v) = (w = collect(skipmissing(v)); isempty(w) ? missing : minimum(w))
    _max(v) = (w = collect(skipmissing(v)); isempty(w) ? missing : maximum(w))
    g = combine(groupby(tbl, :group),
                nrow => :n,
                :ess_bulk => _min => :ess_bulk_min,
                :ess_tail => _min => :ess_tail_min,
                :rhat     => _max => :rhat_max,
                :ess_bulk => (v -> count(x -> x < ess_min, skipmissing(v))) => :n_low_ess,
                :rhat     => (v -> count(x -> x > rhat_max, skipmissing(v))) => :n_high_rhat)
    return sort(g, :ess_bulk_min)
end

"""
    hmc_health(chn) -> NamedTuple

Sampler-level health of a NUTS chain, beyond what `_nuts_diagnostics` reports:

- `divergences`, `frac_div` — a divergence means the leapfrog integrator broke down; ANY divergence
  in the sampling phase biases the posterior and is not fixable by drawing longer.
- `mean_depth`, `obs_max_depth`, `frac_at_obs_max`, `frac_at_cap` — ⚠ read these two fractions
  carefully, they are NOT the same thing. `obs_max_depth` is the deepest tree the chain actually
  built and `frac_at_obs_max` the share of iterations that reached it; that share can be large and
  mean nothing if the observed max sits BELOW the configured ceiling. Only `frac_at_cap` — the
  share at `cap = cfg.stage1_nuts_max_depth`, and `missing` unless `cap` is passed — is the
  saturation that matters: a tree at the ceiling did not terminate by U-turn, so it was cut off
  mid-trajectory. Correct but inefficient, at 2^cap gradients per iteration.
  (`_nuts_diagnostics` reports the observed-max flavour, which is why it read 1.0 on the 25-adapt
  wiring checks.)
- `step_size`, `accept_rate` — the adapted step size and realised acceptance (compare against
  `cfg.stage1_nuts_target_accept = 0.95`).
- `ebfmi` — energy-based Bayes fraction of missing information, `mean(ΔE²)/var(E)` over the
  marginal energy. **Below ~0.3 (Betancourt 2016) the momentum resampling is not moving the chain
  between energy levels**, i.e. heavy tails the sampler cannot climb. This is the one HMC pathology
  that leaves NO trace in divergences or R̂, which is why it is here and not in `_nuts_diagnostics`.
- `grads` — total leapfrog steps = total gradient evaluations, the contention-free cost measure.

Every field is `missing` when the chain lacks the corresponding internal, so this is safe on a
Pathfinder `Chains` (which has no `:internals` section at all).
"""
function hmc_health(chn; cap::Union{Nothing,Integer} = nothing)
    depth, divg = _icol(chn, "tree_depth"), _icol(chn, "numerical_error")
    eps_, acc   = _icol(chn, "step_size"),  _icol(chn, "acceptance_rate")
    nstep, E    = _icol(chn, "n_steps"),    _icol(chn, "hamiltonian_energy")
    dmax  = depth === nothing ? missing : Int(maximum(depth))
    ebfmi = (E === nothing || length(E) < 2 || var(E) == 0) ? missing :
            mean(abs2, diff(E)) / var(E)
    ndiv  = divg === nothing ? missing : count(x -> x === true || x == 1, divg)
    return (; n_draws     = size(chn, 1),
              divergences = ndiv,
              frac_div    = ndiv  === missing ? missing : ndiv / length(divg),
              mean_depth  = depth === nothing ? missing : mean(depth),
              obs_max_depth   = dmax,
              frac_at_obs_max = depth === nothing ? missing : count(==(dmax), depth) / length(depth),
              frac_at_cap     = (depth === nothing || cap === nothing) ? missing :
                                count(==(cap), depth) / length(depth),
              cap         = cap === nothing ? missing : Int(cap),
              step_size   = eps_  === nothing ? missing : last(eps_),
              accept_rate = acc   === nothing ? missing : mean(acc),
              ebfmi, grads = nstep === nothing ? missing : sum(nstep))
end

# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------

"""
    plot_trace_running(chn, params; title="") -> Plots.Plot

Trace of each parameter in `params` with its RUNNING mean overlaid (crimson).

The running mean is the point of the figure. A raw trace of 500 correlated draws is hard to read;
a running mean that is still drifting at the end of the chain is unambiguous evidence the chain has
not settled, and it is the visual counterpart of split-R̂.
"""
function plot_trace_running(chn, params; title::AbstractString = "")
    ps = [p for p in params if Symbol(p) in names(chn, :parameters)]
    isempty(ps) && return plot(; title = "no matching parameters", framestyle = :none)
    panels = map(ps) do p
        v  = _pcol(chn, p)
        rm = cumsum(v) ./ (1:length(v))
        pl = plot(v; lw = 0.6, alpha = 0.75, c = :steelblue, label = "", ylabel = String(Symbol(p)))
        plot!(pl, rm; lw = 2, c = :crimson, label = "")
        pl
    end
    plot(panels...; layout = (length(panels), 1), size = (760, 170 * length(panels)),
         plot_title = title, link = :x, xlabel = "draw")
end

"""
    plot_rank(chn, params; nsplit=4, nbins=20, title="") -> Plots.Plot

Rank plots (Vehtari et al. 2021) — the recommended replacement for trace plots.

Pool every draw of a parameter, replace it by its rank, then split the chain into `nsplit`
consecutive segments and histogram each segment's ranks. If the chain is stationary, each segment
holds a uniform spread of ranks and all `nsplit` histograms are flat at the dashed line. A segment
skewed low or high is a chain that was still moving through the parameter space — the same signal
split-R̂ turns into a number, but showing WHERE in the chain it happened.

With one chain the segments are of that chain, so this diagnoses stationarity, not multi-modality.
"""
function plot_rank(chn, params; nsplit::Int = 4, nbins::Int = 20, title::AbstractString = "")
    ps = [p for p in params if Symbol(p) in names(chn, :parameters)]
    isempty(ps) && return plot(; title = "no matching parameters", framestyle = :none)
    panels = Plots.Plot[]
    for p in ps
        v = _pcol(chn, p)
        r = StatsBase.ordinalrank(v) ./ length(v)          # ranks on (0,1]
        seg = max(1, length(v) ÷ nsplit)
        pl  = plot(; ylabel = String(Symbol(p)), legend = (p === first(ps) ? :outertop : false),
                     legendcolumns = nsplit)
        edges = collect(range(0, 1; length = nbins + 1))
        mids  = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
        for s in 1:nsplit
            idx = ((s - 1) * seg + 1):min(s * seg, length(v))
            hh  = StatsBase.fit(Histogram, r[idx], edges)      # qualify: Distributions exports `fit`
            plot!(pl, mids, hh.weights; seriestype = :steppost, lw = 1.5, label = "seg $(s)")
        end
        hline!(pl, [seg / nbins]; ls = :dash, c = :black, lw = 1, label = "")
        push!(panels, pl)
    end
    plot(panels...; layout = (length(panels), 1), size = (760, 170 * length(panels)),
         plot_title = title, xlabel = "normalised rank")
end

"""
    plot_acf(chn, params; maxlag=50, title="") -> Plots.Plot

Autocorrelation of each parameter against lag, with the ±2/√N white-noise band.

Reads directly as sampling efficiency: ESS ≈ N / (1 + 2Σρ_k), so a parameter whose ACF is still
above the band at lag 50 is contributing a small fraction of its nominal 500 draws.
"""
function plot_acf(chn, params; maxlag::Int = 50, title::AbstractString = "")
    ps = [p for p in params if Symbol(p) in names(chn, :parameters)]
    isempty(ps) && return plot(; title = "no matching parameters", framestyle = :none)
    N  = size(chn, 1)
    pl = plot(; xlabel = "lag", ylabel = "autocorrelation", title = title, size = (760, 420))
    for p in ps
        v = _pcol(chn, p)
        plot!(pl, 0:maxlag, StatsBase.autocor(v, 0:min(maxlag, N - 1)); lw = 1.6,
              label = String(Symbol(p)))
    end
    hline!(pl, [0.0]; c = :black, lw = 0.8, label = "")
    hline!(pl, [2 / sqrt(N), -2 / sqrt(N)]; c = :grey, ls = :dash, lw = 1, label = "±2/√N")
    pl
end

"""
    plot_ess_rhat_dist(tbl; ess_min=100, rhat_max=1.01, title="") -> Plots.Plot

Three panels over ALL parameters: bulk-ESS histogram, tail-ESS histogram, split-R̂ histogram, each
with its threshold marked. This is how a ~300–1000-coordinate chain gets read at a glance — the
per-parameter table is for the offenders the histograms reveal.
"""
function plot_ess_rhat_dist(tbl::DataFrame; ess_min::Real = 100, rhat_max::Real = 1.01,
                            title::AbstractString = "")
    _v(c) = collect(skipmissing(tbl[!, c]))
    p1 = histogram(_v(:ess_bulk); bins = 40, c = :steelblue, lc = :white, label = "",
                   xlabel = "bulk ESS", ylabel = "parameters")
    vline!(p1, [ess_min]; c = :crimson, ls = :dash, lw = 2, label = "min $(ess_min)")
    p2 = histogram(_v(:ess_tail); bins = 40, c = :seagreen, lc = :white, label = "",
                   xlabel = "tail ESS")
    vline!(p2, [ess_min]; c = :crimson, ls = :dash, lw = 2, label = "min $(ess_min)")
    p3 = histogram(_v(:rhat); bins = 40, c = :darkorange, lc = :white, label = "",
                   xlabel = "split-R̂")
    vline!(p3, [rhat_max]; c = :crimson, ls = :dash, lw = 2, label = "max $(rhat_max)")
    plot(p1, p2, p3; layout = (1, 3), size = (1150, 340), plot_title = title, legend = :topright)
end

"""
    plot_adaptation(full, n_adapt; title="") -> Plots.Plot | nothing

Warmup diagnostics over the FULL instrumented chain: step size (log scale), tree depth, and
divergence markers, with the adaptation boundary drawn in.

Returns `nothing` when `full === nothing` — production fits discard warmup inside `sample`, so only
the pilot artefacts can show this. What to look for: the step size should stop moving at the
boundary (dual averaging finalises there) and the tree depth should FALL as it settles. A depth
that stays pinned at the cap after the boundary means the fit is paying 2^max_depth gradients per
iteration for the whole sampling phase.
"""
function plot_adaptation(full, n_adapt; title::AbstractString = "")
    full === nothing && return nothing
    eps_, depth, divg = _icol(full, "step_size"), _icol(full, "tree_depth"), _icol(full, "numerical_error")
    (eps_ === nothing || depth === nothing) && return nothing
    it   = 1:length(eps_)                      # position 1 (statless init) already dropped by _icol
    bnd  = n_adapt === missing ? nothing : n_adapt
    p1 = plot(it, eps_; yscale = :log10, lw = 1.5, c = :steelblue, label = "",
              ylabel = "step size", xlabel = "")
    p2 = plot(it, depth; lw = 0.8, c = :seagreen, alpha = 0.8, label = "",
              ylabel = "tree depth", xlabel = "iteration")
    if divg !== nothing
        di = findall(x -> x === true || x == 1, divg)
        isempty(di) || scatter!(p2, di, depth[di]; ms = 3, c = :crimson, msw = 0,
                                label = "divergence ($(length(di)))")
    end
    if bnd !== nothing
        vline!(p1, [bnd]; c = :black, ls = :dash, lw = 1.5, label = "end of adaptation")
        vline!(p2, [bnd]; c = :black, ls = :dash, lw = 1.5, label = "")
    end
    plot(p1, p2; layout = (2, 1), size = (900, 520), plot_title = title, link = :x)
end

"""
    plot_energy(chn; title="") -> Plots.Plot | nothing

The energy plot (Betancourt 2016): the marginal energy density π_E (centred `hamiltonian_energy`)
against the energy TRANSITION density π_ΔE (its successive differences).

If π_ΔE is visibly narrower than π_E, momentum resampling cannot move the chain across the energy
range the posterior occupies — the sampler is failing to reach the tails, and no amount of extra
draws fixes it. The E-BFMI in `hmc_health` is the scalar version of this picture.
"""
function plot_energy(chn; title::AbstractString = "")
    E = _icol(chn, "hamiltonian_energy")
    (E === nothing || length(E) < 3) && return nothing
    Ec, dE = E .- mean(E), diff(E)
    pl = histogram(Ec; bins = 40, normalize = :pdf, c = :steelblue, lc = :white, alpha = 0.6,
                   label = "π_E (marginal)", xlabel = "energy (centred)", ylabel = "density")
    histogram!(pl, dE .- mean(dE); bins = 40, normalize = :pdf, c = :darkorange, lc = :white,
               alpha = 0.6, label = "π_ΔE (transition)")
    eb = mean(abs2, dE) / var(E)
    plot!(pl; title = title * @sprintf("  E-BFMI = %.2f%s", eb, eb < 0.3 ? "  ** < 0.3 **" : ""),
          size = (760, 420))
end

"""
    plot_divergence_pairs(chn, pairs; title="") -> Plots.Plot

Bivariate scatter of each `(x, y)` in `pairs`, with divergent transitions overplotted in red.

Divergences that CLUSTER in one region of a pair — classically the neck of a funnel between a scale
parameter and the raw variates it multiplies — localise the geometry the sampler could not handle.
Scattered divergences usually just mean the step size is a little too large. `model_degree` is
non-centred throughout (`z`, `z_c` are standard normal), which is exactly the reparameterisation
that removes funnels, so this figure is the check that the reparameterisation did its job.
"""
function plot_divergence_pairs(chn, pairs; title::AbstractString = "")
    divg = _icol(chn, "numerical_error")
    di   = divg === nothing ? Int[] : findall(x -> x === true || x == 1, divg)
    panels = map(pairs) do (px, py)
        (Symbol(px) in names(chn, :parameters) && Symbol(py) in names(chn, :parameters)) ||
            return plot(; title = "missing $(px)/$(py)", framestyle = :none)
        x, y = _pcol(chn, px), _pcol(chn, py)
        pl = scatter(x, y; ms = 2, msw = 0, c = :steelblue, alpha = 0.45, label = "",
                     xlabel = String(Symbol(px)), ylabel = String(Symbol(py)))
        isempty(di) || scatter!(pl, x[di], y[di]; ms = 4, msw = 0, c = :crimson,
                                label = "divergent ($(length(di)))")
        pl
    end
    n = length(panels)
    plot(panels...; layout = (1, n), size = (380 * n, 360), plot_title = title)
end

"""
    convergence_verdict(deg, hh, tbl; ess_min=100, rhat_max=1.01) -> Bool

Print a pass/fail block for one chain and return whether every check passed.

The checks, in the order they matter: divergences in the KEPT draws (a hard fail — those draws are
biased), E-BFMI ≥ 0.3, split-R̂ ≤ `rhat_max` everywhere, and bulk/tail ESS ≥ `ess_min` everywhere.
`frac_at_max` is reported but is NOT a failure: a saturated tree costs gradients, it does not
invalidate draws.
"""
function convergence_verdict(deg::AbstractString, hh, tbl::DataFrame;
                             ess_min::Real = 100, rhat_max::Real = 1.01)
    ok(b) = b ? "PASS" : "FAIL"
    nlo   = count(x -> x < ess_min, skipmissing(tbl.ess_bulk)) +
            count(x -> x < ess_min, skipmissing(tbl.ess_tail))
    nhi   = count(x -> x > rhat_max, skipmissing(tbl.rhat))
    c_div = hh.divergences === missing || hh.divergences == 0
    c_eb  = hh.ebfmi === missing || hh.ebfmi >= 0.3
    c_rh  = nhi == 0
    c_ess = nlo == 0
    println("── convergence verdict: $(deg) ", "─"^(46 - length(deg)))
    @printf("  %s  divergences in kept draws : %s\n", ok(c_div), string(hh.divergences))
    @printf("  %s  E-BFMI >= 0.3             : %s\n", ok(c_eb),
            hh.ebfmi === missing ? "missing" : @sprintf("%.2f", hh.ebfmi))
    @printf("  %s  split-R̂ <= %.2f           : %d of %d parameters above\n", ok(c_rh), rhat_max,
            nhi, nrow(tbl))
    @printf("  %s  ESS >= %d (bulk & tail)   : %d of %d checks below\n", ok(c_ess), ess_min,
            nlo, 2 * nrow(tbl))
    @printf("  ····  tree depth %.2f (deepest built %s, %.0f%% of iterations there; %s at the cap), accept %.3f, %s gradients\n",
            hh.mean_depth, string(hh.obs_max_depth), 100 * hh.frac_at_obs_max,
            hh.frac_at_cap === missing ? "cap not supplied" :
                @sprintf("%.0f%%", 100 * hh.frac_at_cap),
            hh.accept_rate, string(hh.grads))
    all((c_div, c_eb, c_rh, c_ess))
end
