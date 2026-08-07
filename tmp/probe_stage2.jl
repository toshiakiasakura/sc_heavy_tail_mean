# probe_stage2.jl — why is Stage 2 slow?
#
# OBSERVED (2026-08-07 smoke, Pathfinder generation, 3 origins): the first `prefit_stage2!` cell —
# ONE (degree, ngm, origin, horizon), i.e. 100 per-draw Pathfinder fits — took ~19 minutes, with the
# kernel at 17.4 GiB RSS and only ~167 % CPU on a 10-core box despite `max_concurrent = 9`. The
# previous (`-hd`, pre-Mooncake) generation wrote all 1512 of these files in 4.9 h ≈ 12 s each.
#
# Low CPU with a wide semaphore means the workers are BLOCKED, not computing. The hypothesis is
# Mooncake's `build_rrule`: CLAUDE.md records it as "a one-off per model TYPE per process", which is
# true of Stage 1 (`prefit_stage1!` warms it deliberately) — but Stage 2 constructs a fresh
# `model_transmission` per Stage-1 draw, and if the rule is re-derived per construction then each of
# the 100 fits pays ~15 s AND serialises on Mooncake's global lock. 100 × 15 s ≈ 25 min, which is
# the right order for what was seen.
#
# This measures, rather than assumes:
#   1. Is `LogDensityFunction(...; adtype)` construction cached across two IDENTICAL models?
#      (the decisive test — if the second is as slow as the first, the rule is per-construction)
#   2. Serial per-fit cost, first fit vs the rest, under Mooncake.
#   3. Whether `max_concurrent > 1` actually buys anything under Mooncake.
#   4. The same three under ReverseDiff, which is what the 4.9 h generation used.
#
#   julia --project=/workdir /workdir/tmp/probe_stage2.jl [n_draws]
#
# Reads an existing Stage-1 chain from the smoke grid; fits nothing that is kept.

cd("/workdir/src")
include("/workdir/src/forecast_utils.jl")
using Printf, Random, Dates, Statistics, LinearAlgebra

const M      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8   # Stage-1 draws to fit (of the real 100)
const ORIGIN = Date(get(ENV, "PROBE_ORIGIN", "2021-04-25"))
const NDRAW  = parse(Int, get(ENV, "PROBE_NDRAW", "100"))    # Stage-2 draws per fit (production 100)

say(s) = (println(s); flush(stdout))
LinearAlgebra.BLAS.set_num_threads(1)

cfg  = FrameworkConfig(constant_contacts = false, stage1_use_nuts = false)
grid = cis_age_grid()
raw  = load_raw_contact_inputs()
infd = load_raw_infection_inputs()
win0 = WeeklyWindow(ORIGIN; n_fit = cfg.n_fit, smax = cfg.smax, horizons = cfg.horizons)
wd0  = load_window_data(win0, infd.df, infd.tmap; grid = grid)
apd1 = prepare_degree_data(WeeklyWindow(ORIGIN + Day(7); n_fit = cfg.n_fit, smax = cfg.smax,
                                        horizons = cfg.horizons),
                           cfg; grid = grid, setting = :all,
                           df_part_raw = raw.df_part, craw_raw = raw.craw)
dm, nb = NegBinAgePair(), MeanNGM()
s1p = stage1_path(dm, ORIGIN, 1; contacts = contacts_label(cfg), save_dir = "../dt_intermediate")
isfile(s1p) || error("no Stage-1 chain at $(s1p) — run the smoke first")

inp = stage2_inputs(dm, apd1, win0, wd0, cfg, s1p; adtype = ad_type(cfg),
                    rng = Random.Xoshiro(cfg.seed))
say("Stage-1 draws available: $(length(inp.md));  probing with M=$(M), n_draw=$(NDRAW)")
md = inp.md[1:min(M, length(inp.md))]

# One C*, built exactly as `fit_stage2_pooled` builds it per draw (the `Float64.` matters — it is
# what makes `Cstar_m::Vector{Matrix{Float64}}` and hence the model type stable across draws).
# The generation interval is sampled INSIDE `model_transmission`, so there is no `w` argument.
cs1 = [Float64.(contact_star(nb, md[1].K1[t], md[1].K2[t], md[1].G[t]))
       for t in eachindex(md[1].K1)]

# =============================================================================================
# 1. Is the AD rule cached across two IDENTICAL model constructions? THE decisive measurement.
# =============================================================================================
say("\n" * "="^94)
say("1. LogDensityFunction construction (this is where DifferentiationInterface derives the rule)")
say("="^94)
for backend in (:mooncake, :reversediff)
    c   = FrameworkConfig(constant_contacts = false, stage1_use_nuts = false, ad_backend = backend)
    adt = ad_type(c)
    ts  = Float64[]
    for rep in 1:3
        mdl = model_transmission(cs1, wd0, c, nb)
        vil = DynamicPPL.link!!(DynamicPPL.VarInfo(Random.Xoshiro(c.seed), mdl), mdl)
        push!(ts, @elapsed DynamicPPL.LogDensityFunction(mdl, DynamicPPL.getlogjoint_internal, vil;
                                                         adtype = adt))
    end
    @printf("  %-12s  rep1 %7.2f s   rep2 %7.2f s   rep3 %7.2f s   %s\n", backend, ts...,
            ts[2] > 0.25 * ts[1] ? "** NOT CACHED — re-derived every construction **" : "cached")
end

# =============================================================================================
# 2/3. Serial vs parallel `fit_stage2_pooled`, per backend.
# =============================================================================================
say("\n" * "="^94)
say("2. fit_stage2_pooled: serial vs parallel, per backend  (M=$(M) of the production 100)")
say("="^94)
@printf("  %-12s %10s %10s %10s %14s\n", "backend", "K=1", "K=9", "speedup", "→ 100 draws")
for backend in (:mooncake, :reversediff)
    c   = FrameworkConfig(constant_contacts = false, stage1_use_nuts = false, ad_backend = backend)
    adt = ad_type(c)
    # Warm both paths once so the numbers are steady-state, not first-call compilation.
    fit_stage2_pooled(nb, md[1:1], wd0, c; n_draw = NDRAW, adtype = adt,
                      base_seed = c.seed, max_concurrent = 1)
    t1 = @elapsed fit_stage2_pooled(nb, md, wd0, c; n_draw = NDRAW, adtype = adt,
                                    base_seed = c.seed, max_concurrent = 1)
    t9 = @elapsed fit_stage2_pooled(nb, md, wd0, c; n_draw = NDRAW, adtype = adt,
                                    base_seed = c.seed, max_concurrent = 9)
    @printf("  %-12s %9.1fs %9.1fs %9.2f× %11.1f min\n", backend, t1, t9, t1 / t9,
            (t9 / length(md)) * 100 / 60)
end

say("\nProduction Stage 2 is 1512 cells of 100 draws each; the last column × 1512 is the grid cost.")
