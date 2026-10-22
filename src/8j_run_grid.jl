# 8j_run_grid.jl — headless, BOUNDED-BATCH, resumable driver for the 8j two-stage grid.
#
# Mirrors cells 2/4/5/7/8 of `8j_preliminary_forecast.ipynb` and cell 27 of
# `9j_forecast_diagnostics.ipynb`: same `FrameworkConfig`, same six `combos`, same data providers,
# same `prefit_stage1!`/`prefit_stage2!` calls. Since 2026-10-22 ONE run fits EVERY scenario 17j §6
# scores — the conditional grid, the frozen-contact h0 grid and the variant-specific renewal — so
# a single HPC submission (s1 array, then s2 array) produces the whole set. The one thing it adds
# is the reason it exists — it processes a BOUNDED batch of origins and then EXITS, so a
# supervisor (`hpc/run_grid.sh`, locally or inside the Slurm/Singularity job) can restart it in a
# FRESH PROCESS.
#
# FIT SETS (`SCENARIOS`, default all three), per origin:
#   Stage 1  2 degree models × horizons 1:4 (conditional, variant)  +  × h0 (frozen)    = 10
#   Stage 2  6 combos × 1:4 (conditional)                                              = 24
#            6 combos × h0 (frozen)                                                    =  6
#            6 VariantNGM combos × 1:4 (variant)                                       = 24
#            5 contact VariantNGM combos × h0 (variant AND frozen)                     =  5
# ⇒ 10 + 59 per origin; 670 + 3953 over the 67-origin grid. Stage 1 fits h1–4 and h0 in ONE
# fan-out per origin (one `prefit_stage1!` call over `S1_HORIZONS`), so the two h0 chains never run
# 2-wide on their own — the barrier problem `queue = true` was written for in 8j §2b.
#
# WHY. Measured 2026-08-07 (tasks/lessons.md §"The 63-origin grid OOMs a single process") and again
# 2026-08-08: a single process fitting origins back-to-back grows ~1.08 GiB PER ORIGIN — the
# per-origin working set (`wd0`, `apd_by_h`, the `do_fit` closures over them) is garbage once the
# `@sync` returns, but Julia will not run a full collection while the heap looks healthy. Both runs
# were OOM-killed (2026-08-08: cgroup `oom_kill 1`, peak 26.8 GiB of 27 GB, at origin 37/63) with no
# error and nothing in the log, because SIGKILL leaves nothing behind. `prefit_stage1!`/
# `prefit_stage2!` already call `GC.gc()` per origin, and that was measured NOT to be enough
# (1.08 GiB/origin with it, vs 1.2 without) — it keeps per-origin wall time flat but does not stop
# the climb. A PROCESS BOUNDARY resets the heap unconditionally, which no in-process fix can, and a
# Jupyter kernel cannot have one at all: a cell cannot restart its own process. Hence a .jl driver.
#
# ⚠ This DUPLICATES notebook cells 2/4/5/7, so the two can drift. A change to `cfg` or `combos` must
# be made in both. (`tmp/run_nb_headless.jl` evaluates the notebook's own cells and so cannot drift;
# it is the alternative if the duplication ever bites.) The resolved token and origin count are
# printed on every launch so a drift shows up in the log rather than as a silently forked grid.
#
# EXIT CODES — the supervisor's protocol:
#    0  did work, more remains          → relaunch in a fresh process
#   10  nothing pending for this PHASE  → phase complete
#    3  PHASE=s2 but every remaining origin is blocked on missing Stage-1 chains → stop (a failure)
#  else crash (OOM, fit error)          → supervisor stops rather than powering through
#
# ENVIRONMENT (all optional):
#   PHASE=s1|s2       which stage to advance                        (default s1)
#   SCENARIOS         comma list ⊆ conditional,frozen,variant       (default all three)
#   STAGE1_USE_NUTS   Stage-1 sampler                               (default "true" ⇒ NUTS)
#   AD_BACKEND        Stage-1 AD backend                            (default mooncake)
#   MAX_ORIGINS       origins per PROCESS — the memory bound        (default 8)
#   CHUNK             origins per prefit! call — memory checkpoint   (default 4)
#   MEM_FLOOR_GIB     exit early below this MemAvailable            (default 8.0)
#   ORIGIN_STRIDE     how many disjoint slices the origins split into (default 1 = no slicing)
#   ORIGIN_OFFSET     which slice THIS process owns, 0-based         (default 0)
#   DRY_RUN=1         resolve + print everything, exit 10 before any fit (login-node preflight)
#   S1_CONCURRENCY / S2_CONCURRENCY / FIT_END / ORIGIN_MIN  — exactly as the notebook reads them
#
# ORIGIN_STRIDE/ORIGIN_OFFSET exist for the Slurm ARRAY (hpc/s1_array.slurm): each array task owns a
# round-robin slice of the origins and therefore a DISJOINT set of artefact paths. Without them every
# task takes `pending[1:MAX_ORIGINS]` from the same disk view, so N tasks would fit the SAME chains
# concurrently and race each other's writes. Round-robin rather than contiguous blocks because
# origins differ in cost and interleaving averages that out across tasks.
#
# STAGE1_USE_NUTS defaults to "true", as in 8j. (It defaulted to "false" until 2026-10-22, to RESUME
# the Pathfinder `-ar1` generation; nothing is left to resume.) It is in the cache token, so getting
# it wrong does not error — it starts a different generation and refits from scratch.
#
#   julia --project=/workdir /workdir/src/8j_run_grid.jl

ENV["GKSwstype"] = "100"   # headless GR, as the notebook does
cd(@__DIR__)               # every relative path in the framework assumes cwd == src/

include("forecast_utils.jl")   # single preamble: base + CoMix pipeline + forecasting framework
using Random, Statistics, Dates, Printf

say(args...) = (println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] ", args...);
                flush(stdout); flush(stderr))

# ───────────────────────────────────────────────────────────── config (notebook cells 2 and 4)
STAGE1_USE_NUTS = get(ENV, "STAGE1_USE_NUTS", "true") == "true"
AD_BACKEND      = Symbol(get(ENV, "AD_BACKEND", "mooncake"))
PHASE           = get(ENV, "PHASE", "s1")
MAX_ORIGINS     = parse(Int,     get(ENV, "MAX_ORIGINS",  "8"))
CHUNK           = parse(Int,     get(ENV, "CHUNK",        "4"))
MEM_FLOOR       = parse(Float64, get(ENV, "MEM_FLOOR_GIB", "8.0"))
PHASE in ("s1", "s2") || error("PHASE must be \"s1\" or \"s2\", got $(repr(PHASE))")

const KNOWN_SCENARIOS = ("conditional", "frozen", "variant")
SCENARIOS = Set(String.(strip.(split(get(ENV, "SCENARIOS", join(KNOWN_SCENARIOS, ",")), ","; keepempty = false))))
isempty(SCENARIOS) && error("SCENARIOS selected nothing")
for sc in SCENARIOS
    sc in KNOWN_SCENARIOS || error("unknown scenario $(repr(sc)) in SCENARIOS — known: $(join(KNOWN_SCENARIOS, ", "))")
end

cfg  = FrameworkConfig(constant_contacts = false, stage1_use_nuts = STAGE1_USE_NUTS,
                       ad_backend = AD_BACKEND)
grid = cis_age_grid()

FIT_END = Date(get(ENV, "FIT_END", "2022-01-24"))   # ⇒ 67 Monday origins, 2020-10-19 … 2022-01-24
raw = load_raw_contact_inputs()              # England-only roster (`england_only`)
inf = load_raw_infection_inputs()            # (; df, tmap) — read estimates_age.csv once
# COG-UK variant composition — the variant sets need it, and it is an UNTRACKED file, so a cluster
# copy of the repo lacks it unless it was uploaded alongside `dt_comix_no_public/`.
cog = if "variant" in SCENARIOS
    isfile(COG_VARIANT_PATH) || error("SCENARIOS includes \"variant\" but $(COG_VARIANT_PATH) is missing — " *
                                      "upload dt_cog_uk/ with the repo (hpc/README.md)")
    load_cog_variant_counts()
else
    nothing
end
FORECAST_ORIGINS = available_forecast_origins(cfg; grid = grid, craw = raw.craw,
                                              origin_max = FIT_END)
haskey(ENV, "ORIGIN_MIN") && filter!(>=(Date(ENV["ORIGIN_MIN"])), FORECAST_ORIGINS)
@assert !isempty(FORECAST_ORIGINS) "ORIGIN_MIN/FIT_END selected no origins"
wins = [WeeklyWindow(o; n_fit = cfg.n_fit, smax = cfg.smax, horizons = cfg.horizons)
        for o in FORECAST_ORIGINS]

# ── this process's ORIGIN SLICE (Slurm array partition; identity by default) ──────────────────
# Applied BEFORE the pending filter and used for every count below, so `wins` means "the origins
# this process is responsible for" everywhere downstream — including the `remaining` count that
# decides the exit code, which must be about THIS slice and not the whole grid.
ORIGIN_STRIDE = parse(Int, get(ENV, "ORIGIN_STRIDE", "1"))
ORIGIN_OFFSET = parse(Int, get(ENV, "ORIGIN_OFFSET", "0"))
ORIGIN_STRIDE >= 1 || error("ORIGIN_STRIDE must be >= 1, got $ORIGIN_STRIDE")
0 <= ORIGIN_OFFSET < ORIGIN_STRIDE ||
    error("ORIGIN_OFFSET must be in 0:$(ORIGIN_STRIDE - 1), got $ORIGIN_OFFSET")
wins = wins[(ORIGIN_OFFSET + 1):ORIGIN_STRIDE:end]
@assert !isempty(wins) "slice $ORIGIN_OFFSET/$ORIGIN_STRIDE selected no origins"

# ───────────────────────────────────────────────────────────── combos (notebook cell 5, 9j cell 23)
combos = vcat([(dm, nb) for dm in (NegBinAgePair(), HurdleWeibullAgePair())
                        for nb in (MeanNGM(), NeighbourhoodDegreeNGM())],
              [(NegBinAgePair(),   DiagonalMeanNGM()),      # no-interaction (inst/6)
               (NoContactDegree(), NullNGM())])             # null (inst/6)
# The variant models wrap every combo (the wrapper forwards `contact_star`/`fix_infectivity`); the
# frozen scheme needs C*, so its variant set drops the null model — exactly 9j §V.
vcombos         = [(dm, VariantNGM(nb)) for (dm, nb) in combos]
vcombos_contact = filter(c -> needs_stage1(c[1]), vcombos)

# Stage 1 fans out whole (degree × horizon) FITS; Stage 2 fans out 100 fits of an 18-dimension
# transmission model within one cell, so 1 GiB/fit is generous there and the CPU cap should bind.
S1_CONCURRENCY = parse(Int, get(ENV, "S1_CONCURRENCY",
                       string(fit_concurrency(mem_per_fit_gib = STAGE1_USE_NUTS ? 4.0 : 2.0))))
S2_CONCURRENCY = parse(Int, get(ENV, "S2_CONCURRENCY", string(fit_concurrency())))

const SAVE_DIR = "../dt_intermediate"
const TAG      = contacts_label(cfg)
const DMS      = filter(needs_stage1, unique(first.(combos)))   # the NULL model has no Stage 1

# ── FIT SETS ─────────────────────────────────────────────────────────────────────────────────────
# Stage 1 serves BOTH h1–4 scenarios (conditional, variant) from the same chains, and the h0 ones
# from the frozen chains, so its horizons are a union. Stage 2 is one set per scenario cell type.
# Each set carries its own `horizons`, because a data provider's `apd_by_h` must be ALIGNED to the
# horizons of the call it serves (`prefit_stage1!`'s docstring) — the h0 window is not `cfg.horizons`.
const S1_HORIZONS = vcat(("conditional" in SCENARIOS || "variant" in SCENARIOS) ? collect(cfg.horizons) : Int[],
                         "frozen" in SCENARIOS ? [FROZEN_H] : Int[])
const S2_SETS = filter(st -> st.on, [
    (name = "conditional",    combos = combos,          horizons = collect(cfg.horizons), variant = false,
     on = "conditional" in SCENARIOS),
    (name = "frozen",         combos = combos,          horizons = [FROZEN_H],            variant = false,
     on = "frozen" in SCENARIOS),
    (name = "variant",        combos = vcombos,         horizons = collect(cfg.horizons), variant = true,
     on = "variant" in SCENARIOS),
    (name = "variant-frozen", combos = vcombos_contact, horizons = [FROZEN_H],            variant = true,
     on = "variant" in SCENARIOS && "frozen" in SCENARIOS)])

s1_paths(w) = [stage1_path(dm, w.origin, h; contacts = TAG, save_dir = SAVE_DIR)
               for dm in DMS, h in S1_HORIZONS]
s2_paths(w, st) = [stage2_path(dm, nb, w.origin, h; contacts = TAG, save_dir = SAVE_DIR)
                   for (dm, nb) in st.combos, h in st.horizons]
s1_missing(w) = count(!isfile, s1_paths(w))
s2_missing(w) = sum(st -> count(!isfile, s2_paths(w, st)), S2_SETS)
const N_S1 = length(DMS) * length(S1_HORIZONS)                                  # 10 with all scenarios
const N_S2 = sum(st -> length(st.combos) * length(st.horizons), S2_SETS)        # 59 with all scenarios

say("8j_run_grid | PHASE=", PHASE, " | pid ", getpid(), " | threads ", Threads.nthreads())
say("  sampler     : Stage 1 ", cfg.stage1_use_nuts ? "NUTS" : "Pathfinder", ", Stage 2 Pathfinder")
say("  ad_backend  : Stage 1 ", cfg.ad_backend, ", Stage 2 ", cfg.stage2_ad_backend)
say("  cache token : ", TAG)
say("  origins     : ", length(FORECAST_ORIGINS), " available (", first(FORECAST_ORIGINS), " … ",
    last(FORECAST_ORIGINS), ", capped at ", FIT_END, ")")
# The slice's origin LIST, not just its length: it is the only evidence in the log that the array
# tasks partitioned the grid rather than duplicating it. Two logs that share an origin are the bug.
say("  slice       : ", ORIGIN_OFFSET, "/", ORIGIN_STRIDE, " ⇒ ", length(wins), " origin(s): ",
    join(string.(getfield.(wins, :origin)), ", "))
say("  concurrency : s1 = ", S1_CONCURRENCY, ", s2 = ", S2_CONCURRENCY,
    " | batch = ", MAX_ORIGINS, " origins, chunk = ", CHUNK, ", mem floor = ", MEM_FLOOR, " GiB")
say("  scenarios   : ", join(sort(collect(SCENARIOS)), ", "), " | s1 horizons ", S1_HORIZONS)
say("  grid now    : s1 ", sum(w -> N_S1 - s1_missing(w), wins), "/", N_S1 * length(wins),
    ", s2 ", sum(w -> N_S2 - s2_missing(w), wins), "/", N_S2 * length(wins),
    " | MemAvailable ", round(_mem_available_gib(); digits = 1), " GiB")
for st in S2_SETS   # per-set scope, so the preflight shows exactly what this run will produce
    n = length(st.combos) * length(st.horizons) * length(wins)
    say("    s2 ", rpad(st.name, 15), ": ", n - sum(w -> count(!isfile, s2_paths(w, st)), wins), "/", n,
        "  (", length(st.combos), " models × h ", st.horizons, ")")
end

# ───────────────────────────────────────────────────────────── pick this process's batch
pending = if PHASE == "s1"
    [w for w in wins if s1_missing(w) > 0]
else
    # Stage 2 only for origins whose Stage 1 is COMPLETE. An origin missing a Stage-1 chain sends
    # `stage2_inputs` down `fit_or_load_stage1`, which fits it SERIALLY (it passes no
    # `max_concurrent`, so the 100 per-draw Pathfinder fits run one at a time, ~9× slower) and does
    # NOT error — the silent-refit trap in CLAUDE.md's Gotchas. Naming the blocked origins turns a
    # mystery slowdown into a line in the log.
    ready   = [w for w in wins if s2_missing(w) > 0 && s1_missing(w) == 0]
    blocked = [w for w in wins if s2_missing(w) > 0 && s1_missing(w) > 0]
    isempty(blocked) || say("  NOTE: ", length(blocked), " origin(s) held back — Stage 1 incomplete: ",
                            join(string.(getfield.(blocked, :origin)), ", "))
    ready
end

if isempty(pending)
    # s2 with origins held back is NOT complete: report it as a failure (exit 3), so `run_grid.sh`
    # stops and Slurm marks the task FAILED, rather than an s2 job submitted before Stage 1 finished
    # "completing" with nothing fitted.
    if PHASE == "s2" && any(w -> s2_missing(w) > 0, wins)
        nb = count(w -> s2_missing(w) > 0, wins)
        if get(ENV, "DRY_RUN", "0") == "1"     # the preflight: report it, but exit 10 so a loop never spins
            say("DRY_RUN=1 — nothing READY for PHASE=s2: ", nb, " origin(s) blocked on Stage 1 (a real run exits 3)")
            exit(10)
        end
        say("*** nothing READY for PHASE=s2, but ", nb,
            " origin(s) still need Stage 2 and are blocked on Stage 1 — run/finish s1 first")
        exit(3)
    end
    say("nothing pending for PHASE=", PHASE, " — phase complete")
    exit(10)
end

batch = pending[1:min(length(pending), MAX_ORIGINS)]
say("  pending     : ", length(pending), " origin(s); this process takes ", length(batch),
    " (", first(batch).origin, " … ", last(batch).origin, ")")

# DRY_RUN=1 stops HERE — after the token, the slice, the concurrency and the batch have been
# resolved and printed, but before a single fit. This is the HPC preflight (`hpc/README.md` §3):
# the one thing worth checking on a login node is that the resolved CACHE TOKEN and slice are the
# ones intended, and that check must not cost an hour of NUTS on a shared login node.
#   Exit 10, not 0: if DRY_RUN is ever left set inside a Slurm job, `hpc/run_grid.sh` reads 0 as
# "did work, relaunch" and would spin forever. 10 ends the loop cleanly.
if get(ENV, "DRY_RUN", "0") == "1"
    say("DRY_RUN=1 — resolved and stopping before any fit")
    exit(10)
end

# ───────────────────────────────────────────────────────────── data providers (8j cells 7/8, 9j cell 26)
# One origin's datasets: window infections (+ the origin's `VariantSplit` for the variant
# sets) and one contact/degree window PER ENTRY of `horizons` — aligned to the horizons of the call
# the provider serves. Pure and deterministic given the shared read-only `inf`/`raw`/`cog` reads.
# `oi` is unused — `prefit_*!` pass it as the index into the vector they were given.
make_provider(horizons; variant::Bool = false) = (oi, win_o) -> begin
    wd = load_window_data(win_o, inf.df, inf.tmap; grid = grid)                  # reuse the single CSV reads
    variant && (wd = with_variants(wd, variant_split(win_o, cog, cfg)))          # categories fixed at t₀
    (wd, [prepare_degree_data(degree_window(win_o.origin, h, cfg), cfg;
                              grid = grid, setting = :all,
                              df_part_raw = raw.df_part, craw_raw = raw.craw)    # reuse the single Arrow read
          for h in horizons])
end

# ───────────────────────────────────────────────────────────── run, chunk by chunk
# Sub-chunking inside the process buys a memory checkpoint between groups of origins, so a process
# that is heading for the wall stops and says so instead of being SIGKILLed. It costs one extra
# "warm" round per chunk in Stage 1: `prefit_stage1!` keeps its `warmed` set PER CALL, so the first
# fit of each degree model in a chunk runs serially rather than in the fan-out. That is ~2 serial
# fits per chunk — NOT a Mooncake `build_rrule` rebuild, which is cached per PROCESS and so is paid
# once here regardless.
#
# ⚠ `n_fitted` is a Ref, not an Int, because the loop below is a TOP-LEVEL `for` and therefore SOFT
# SCOPE: a bare `did += r.fitted` there does NOT touch the global — Julia makes `did` a fresh local,
# so the `+=` reads an undefined variable and throws `UndefVarError` at the end of the first chunk.
# (It warns, but only at lowering time, buried above the fit output.) Mutating a container has no
# soft-scope ambiguity at all — the same reason `tmp/check_grid.jl` routes every write to its `ok`
# flag through a function. This project has been bitten by soft scope repeatedly; see the note there.
t_start = time()
const n_fitted = Ref(0)
for (ci, chunk) in enumerate(Iterators.partition(batch, CHUNK))
    chunk = collect(chunk)
    say("chunk ", ci, ": origins ", first(chunk).origin, " … ", last(chunk).origin,
        " (", length(chunk), ")")
    t0 = time()
    if PHASE == "s1"
        # ONE call over h1–4 AND h0: every (degree × horizon) fit of an origin shares one fan-out.
        r = prefit_stage1!(DMS, chunk, cfg; data_provider = make_provider(S1_HORIZONS),
                           horizons = S1_HORIZONS, save_dir = SAVE_DIR, max_concurrent = S1_CONCURRENCY)
        n_fitted[] += r.fitted
        say("  s1 ", S1_HORIZONS, ": ", r.fitted, " fitted / ", r.skipped, " skipped / ", r.failed,
            " FAILED in ", round(Int, time() - t0), "s")
    else
        for st in S2_SETS
            t1 = time()
            r = prefit_stage2!(st.combos, chunk, cfg;
                               data_provider = make_provider(st.horizons; variant = st.variant),
                               horizons = st.horizons, save_dir = SAVE_DIR, max_concurrent = S2_CONCURRENCY)
            n_fitted[] += r.fitted
            say("  s2 ", rpad(st.name, 15), ": ", r.fitted, " fitted / ", r.skipped, " skipped / ",
                r.failed, " FAILED in ", round(Int, time() - t1), "s")
        end
    end

    GC.gc(); GC.gc()   # second pass finalises what the first one queued
    avail = _mem_available_gib()
    # ⚠ `Sys.maxrss()` is a HIGH-WATER MARK and monotone by construction — it can neither confirm
    # nor refute accumulation. MemAvailable (free + reclaimable) is the figure that decides.
    say("  MemAvailable ", round(avail; digits = 1), " GiB (process maxrss ",
        round(Sys.maxrss() / 2^30; digits = 1), " GiB)")
    if avail < MEM_FLOOR
        say("  *** MemAvailable below MEM_FLOOR_GIB=", MEM_FLOOR,
            " — stopping this process early; the supervisor will start a fresh one")
        break
    end
end

remaining = PHASE == "s1" ? count(w -> s1_missing(w) > 0, wins) :
                            count(w -> s2_missing(w) > 0 && s1_missing(w) == 0, wins)
say("process done: ", n_fitted[], " artefact(s) in ", round(Int, time() - t_start),
    "s | origins still pending for PHASE=", PHASE, ": ", remaining)
blocked_left = PHASE == "s2" ? count(w -> s2_missing(w) > 0 && s1_missing(w) > 0, wins) : 0
blocked_left > 0 && remaining == 0 &&
    say("*** ", blocked_left, " origin(s) still blocked on Stage 1 — exiting 3, not 10")
exit(remaining > 0 ? 0 : blocked_left > 0 ? 3 : 10)
