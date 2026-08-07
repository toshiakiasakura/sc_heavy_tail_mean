# check_grid.jl — the gate between 8j and 9j/10j/11j.
#
# WHY THIS EXISTS. `fit_or_load_stage2` fits on miss, and 9j wraps every origin×combo in a
# `try/catch` that only *logs*. So a grid with one missing artefact does not fail when 9j runs — it
# silently refits, serially (`fit_or_load_stage2` passes no `max_concurrent`, so the 100 per-draw
# Pathfinder fits run one at a time), and under the NUTS generation that means a full 1000-adapt
# Stage-1 fit inside a notebook cell. Counting the files first is the whole defence.
#
# It also audits PROVENANCE. `contacts_label` encodes only the sampler (`-nuts`), deliberately not
# the AD backend or the NUTS tuning, so a grid refitted in part across a backend/target_accept/draw
# change is mixed and nothing in the filename says so. The artefacts carry `ad_backend`,
# `target_accept`, `nuts_adapts` and `nuts_draws`; this reads them without loading the chains.
#
#   julia --project=/workdir /workdir/tmp/check_grid.jl
#   STAGE1_USE_NUTS=false ORIGIN_MIN=2021-04-25 FIT_END=2021-05-09 julia … tmp/check_grid.jl
#
# Same three env knobs as 8j/9j, same defaults. Exit code 0 = complete and uniform, 1 = otherwise.

cd("/workdir/src")
include("/workdir/src/forecast_utils.jl")
using JLD2, Printf, StatsBase, Dates

const SAVE_DIR = get(ENV, "SAVE_DIR", "../dt_intermediate")
const MAXLIST  = parse(Int, get(ENV, "MAXLIST", "25"))   # how many missing paths to print

STAGE1_USE_NUTS = get(ENV, "STAGE1_USE_NUTS", "true") == "true"
cfg  = FrameworkConfig(constant_contacts = false, stage1_use_nuts = STAGE1_USE_NUTS)
grid = cis_age_grid()
raw  = load_raw_contact_inputs()

FIT_END = Date(get(ENV, "FIT_END", "2021-12-31"))
origins = available_forecast_origins(cfg; grid = grid, craw = raw.craw, origin_max = FIT_END)
haskey(ENV, "ORIGIN_MIN") && filter!(>=(Date(ENV["ORIGIN_MIN"])), origins)

combos = vcat([(dm, nb) for dm in (NegBinAgePair(), HurdleWeibullAgePair())
                        for nb in (MeanNGM(), NeighbourhoodDegreeNGM())],
              [(NegBinAgePair(), DiagonalMeanNGM()), (NoContactDegree(), NullNGM())])
dms = filter(needs_stage1, unique(first.(combos)))
tag = contacts_label(cfg)

println("="^96)
println("check_grid | token $(tag)")
println("  save_dir $(SAVE_DIR) | $(length(origins)) origins $(first(origins))…$(last(origins))" *
        " | horizons $(cfg.horizons) | $(length(combos)) combos")
println("="^96)

s1_paths = [stage1_path(dm, o, h; contacts = tag, save_dir = SAVE_DIR)
            for o in origins, dm in dms, h in cfg.horizons]
s2_paths = [stage2_path(dm, nb, o, h; contacts = tag, save_dir = SAVE_DIR)
            for o in origins, (dm, nb) in combos, h in cfg.horizons]

# ⚠ `ok` is mutated from inside top-level `for` bodies, which are SOFT SCOPE: a bare `ok = false`
# there silently creates a new local and leaves the global `true`, so the script would print
# "GRID COMPLETE AND UNIFORM" over a broken grid and exit 0. That false pass has bitten this project
# repeatedly (see tasks/lessons.md). Every write goes through `fail!`, which is a function and
# therefore has no soft-scope ambiguity at all.
ok = true
fail!() = (global ok = false)

for (lbl, paths) in (("Stage 1", s1_paths), ("Stage 2", s2_paths))
    miss = filter(!isfile, vec(paths))
    @printf("%s: %d / %d present%s\n", lbl, length(paths) - length(miss), length(paths),
            isempty(miss) ? "" : "   ** $(length(miss)) MISSING **")
    if !isempty(miss)
        fail!()
        for p in first(miss, MAXLIST); println("    missing  ", basename(p)); end
        length(miss) > MAXLIST && println("    … and $(length(miss) - MAXLIST) more")
    end
end

# ---------------------------------------------------------------------------------------------
# Provenance. `jldopen` + `haskey` reads only the file's index — the (7–72 MB) chain is never
# touched, so this is fast enough to run over all 504.
# ---------------------------------------------------------------------------------------------
println("\n--- Stage-1 provenance (index-only reads; chains not loaded) ---")
present = filter(isfile, vec(s1_paths))
if isempty(present)
    println("  (no Stage-1 artefacts to audit)")
else
    keys_of_interest = ("sampler", "ad_backend", "target_accept", "nuts_adapts", "nuts_draws")
    tally = Dict(k => Dict{Any,Int}() for k in keys_of_interest)
    for p in present
        jldopen(p) do f
            for k in keys_of_interest
                v = haskey(f, k) ? f[k] : :ABSENT
                tally[k][v] = get(tally[k], v, 0) + 1
            end
        end
    end
    for k in keys_of_interest
        vals = sort(collect(tally[k]), by = x -> -x[2])
        uniform = length(vals) == 1
        uniform || fail!()
        @printf("  %-14s %s%s\n", k, join(["$(v) × $(n)" for (v, n) in vals], ",  "),
                uniform ? "" : "   ** MIXED PROVENANCE **")
    end
    # Stage 2 records only its own backend (there is no chain, sampler or tuning to describe — it is
    # always Pathfinder). It is a DIFFERENT backend from Stage 1 by default, so a `:mooncake` here
    # means the cell predates the 2026-08-07 split and cost ~16.5 min instead of ~12 s.
    s2_present = filter(isfile, vec(s2_paths))
    if !isempty(s2_present)
        t2 = Dict{Any,Int}()
        for p in s2_present
            v = jldopen(p) do f; haskey(f, "ad_backend") ? f["ad_backend"] : :ABSENT; end
            t2[v] = get(t2, v, 0) + 1
        end
        vals = sort(collect(t2), by = x -> -x[2])
        length(vals) == 1 || fail!()
        @printf("  %-14s %s%s\n", "s2 ad_backend",
                join(["$(v) × $(n)" for (v, n) in vals], ",  "),
                length(vals) == 1 ? "" : "   ** MIXED PROVENANCE **")
    end

    # The draw count is the one setting that is neither in the filename nor (for pre-2026-08-07
    # artefacts) in the file, so cross-check it against the chain itself on a small sample.
    sample = present[round.(Int, range(1, length(present); length = min(4, length(present))))]
    for p in sample
        n = size(load(p, "result"), 1)
        @printf("  chain rows     %-58s %d\n", basename(p), n)
        if STAGE1_USE_NUTS && n != cfg.stage1_nuts_draws
            println("    ** expected $(cfg.stage1_nuts_draws) kept draws **"); fail!()
        end
    end
end

println("\n", ok ? "GRID COMPLETE AND UNIFORM — safe to run 9j/10j/11j" :
                   "GRID INCOMPLETE OR MIXED — do NOT run 9j (it refits on miss, silently)")
exit(ok ? 0 : 1)
