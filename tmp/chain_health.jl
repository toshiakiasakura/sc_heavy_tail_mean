# chain_health.jl — Stage-1 NUTS health over the WHOLE grid, not the four pilot cells.
#
# `_nuts_diagnostics` is computed at fit time and stored under the `diag` key, so the distribution
# of divergences / max-depth saturation / min ESS across all 504 fits is available WITHOUT loading a
# single chain (jldopen + a NamedTuple read touches only the file index). That matters: at 2000
# draws the chains are ~28 MB (NegBin) / ~72 MB (hurdle-Weibull), so loading them all would be 25 GB
# of I/O to answer a question the artefacts already answer.
#
# Split-R̂ is NOT in `diag` (one chain per fit ⇒ MCMCChains reports no R̂), so it is computed here on
# an opt-in subsample: RHAT_N chains are loaded, split in half, and scored with
# `MCMCChains.rhat` on the rank-normalised split — the same quantity 12j reports, but over a
# sample of the grid rather than one origin.
#
#   julia --project=/workdir /workdir/tmp/chain_health.jl
#   STAGE1_USE_NUTS=false RHAT_N=0 julia … tmp/chain_health.jl     # rehearsal grid, skip split-R̂

cd("/workdir/src")
include("/workdir/src/forecast_utils.jl")
using JLD2, Printf, Statistics, Dates, MCMCChains

const SAVE_DIR = get(ENV, "SAVE_DIR", "../dt_intermediate")
const RHAT_N   = parse(Int, get(ENV, "RHAT_N", "8"))   # chains to load for split-R̂; 0 disables

STAGE1_USE_NUTS = get(ENV, "STAGE1_USE_NUTS", "true") == "true"
cfg  = FrameworkConfig(constant_contacts = false, stage1_use_nuts = STAGE1_USE_NUTS)
tag  = contacts_label(cfg)

paths = filter(p -> occursin("_$(tag)_", p) && startswith(basename(p), "8j_s1_"),
               readdir(SAVE_DIR; join = true))
isempty(paths) && (println("no Stage-1 artefacts for token $(tag) in $(SAVE_DIR)"); exit(1))

# Filenames are 8j_s1_<degree>_<token>_<origin>_h<h>.jld2; the degree label is the one field that
# varies between the two very different-sized models, so report per degree as well as overall.
degree_of(p) = occursin("8j_s1_unweighted-negbin_", basename(p)) ? "unweighted-negbin" :
               occursin("8j_s1_weighted-hweibull_", basename(p)) ? "weighted-hweibull" : "other"

rows = NamedTuple[]
for p in paths
    d = jldopen(p) do f; haskey(f, "diag") ? f["diag"] : nothing; end
    d === nothing && continue
    g(k) = (v = get(d, k, missing); v === nothing ? missing : v)
    push!(rows, (; degree = degree_of(p), file = basename(p),
                   div = g(:divergences), frac = g(:frac_at_max_depth),
                   ess = g(:min_ess), n = g(:n_draws)))
end

println("="^96)
println("chain_health | token $(tag) | $(length(rows)) artefacts with a `diag` key of $(length(paths))")
println("="^96)

fmt(v) = v === missing ? "  --" : @sprintf("%.3g", v)
function report(lbl, rs)
    isempty(rs) && return
    ess  = collect(skipmissing(getfield.(rs, :ess)))
    divs = collect(skipmissing(getfield.(rs, :div)))
    frac = collect(skipmissing(getfield.(rs, :frac)))
    @printf("\n%s  (n = %d)\n", lbl, length(rs))
    if !isempty(ess)
        q = quantile(ess, [0.0, 0.05, 0.5, 0.95, 1.0])
        @printf("  min ESS      min %.1f  p5 %.1f  median %.1f  p95 %.1f  max %.1f\n", q...)
        @printf("               %d/%d fits below 100,  %d/%d below the ESS>200 bar\n",
                count(<(100), ess), length(ess), count(<(200), ess), length(ess))
    end
    if !isempty(divs)
        @printf("  divergences  total %d over %d fits;  %d fits with any (%.1f%%), worst %d\n",
                sum(divs), length(divs), count(>(0), divs), 100count(>(0), divs) / length(divs),
                maximum(divs))
    end
    if !isempty(frac)
        @printf("  at max_depth median %.3f, p95 %.3f, max %.3f;  %d fits above the 0.2 warn line\n",
                median(frac), quantile(frac, 0.95), maximum(frac), count(>(0.2), frac))
    end
    # Name the worst few — an aggregate hides whether a bad tail is one pathological origin or a
    # systematic problem, and those call for different responses.
    worst = sort(filter(r -> r.ess !== missing, rs), by = r -> r.ess)
    for r in first(worst, min(5, length(worst)))
        @printf("    worst  %-72s ESS %7.1f  div %s\n", r.file, r.ess, fmt(r.div))
    end
end

report("ALL", rows)
for deg in unique(getfield.(rows, :degree)); report(deg, filter(r -> r.degree == deg, rows)); end

# ---------------------------------------------------------------------------------------------
# Split-R̂ on a subsample. This is the quantity the 2000-draw raise was meant to discriminate:
# the failures previously sat at HIGH ESS (max R̂ 1.085 at ESS 1023), which points at
# first-half/second-half drift rather than autocorrelation. If the >1.01 count now falls roughly in
# proportion to the extra draws it was autocorrelation after all; if it holds, drift is confirmed.
# ---------------------------------------------------------------------------------------------
if RHAT_N > 0
    println("\n", "-"^96)
    println("split-R̂ on $(min(RHAT_N, length(paths))) chains (loaded; this is the expensive part)")
    println("-"^96)
    sample = paths[round.(Int, range(1, length(paths); length = min(RHAT_N, length(paths))))]
    for p in sample
        chn = load(p, "result")
        tbl = try
            DataFrame(MCMCChains.ess_rhat(chn))
        catch e
            println("  ", basename(p), ": ess_rhat failed — ", sprint(showerror, e)[1:min(80, end)])
            continue
        end
        ecol = Symbol(first(intersect(String.(names(tbl)), ["ess_bulk", "ess"])))
        @printf("  %-72s rows %4d  R̂>1.01 %4d  maxR̂ %.3f  minESS %7.1f\n",
                basename(p), size(chn, 1), count(>(1.01), tbl.rhat), maximum(tbl.rhat),
                minimum(tbl[!, ecol]))
    end
end
