# probe_stage2b.jl — WHERE does Mooncake lose Stage 2?
#
# probe_stage2.jl measured the whole `fit_stage2_pooled`: Mooncake 10.8 s/draw with NO parallel
# speedup (1.09× at K=9), ReverseDiff 0.3 s/draw with 2.31×. That is a 36× gap, while
# `framework.jl`'s `ad_backend` docstring records Mooncake as **17.9× FASTER** on this very model
# (30 685 vs 1 711 grad/s, `tmp/verify_adtype.jl`). Both cannot describe the same quantity.
#
# The difference between those two measurements is everything Pathfinder does around the gradient:
# a fresh `model_transmission` and a fresh `LogDensityFunction` per Stage-1 draw. So split the cost
# into (a) prepared-gradient throughput and (b) per-fit setup, and check whether the raw gradient
# advantage survives at all. Also count the model's dimension — "18 dims" is the figure the old
# bench was taken at, and the transmission block has changed since.
#
#   julia --project=/workdir /workdir/tmp/probe_stage2b.jl

cd("/workdir/src")
include("/workdir/src/forecast_utils.jl")
using Printf, Random, Dates, Statistics, LinearAlgebra

const ORIGIN = Date(get(ENV, "PROBE_ORIGIN", "2021-04-25"))
const LDP    = DynamicPPL.LogDensityProblems     # transitive-only dep — reach it through DynamicPPL
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
inp = stage2_inputs(dm, apd1, win0, wd0, cfg, s1p; adtype = ad_type(cfg),
                    rng = Random.Xoshiro(cfg.seed))
md  = inp.md
mkC(m) = [Float64.(contact_star(nb, md[m].K1[t], md[m].K2[t], md[m].G[t])) for t in eachindex(md[m].K1)]

say("\n" * "="^94)
say("Stage-2 transmission model: prepared-gradient rate vs per-fit setup")
say("="^94)
@printf("  %-12s %8s %12s %14s %12s %14s\n",
        "backend", "dims", "prep #1 (s)", "prep #2..5 (s)", "grad/s", "pathfinder (s)")
for backend in (:mooncake, :reversediff)
    c   = FrameworkConfig(constant_contacts = false, stage1_use_nuts = false, ad_backend = backend)
    adt = _resolve_adtype(backend)

    build(m) = begin
        mdl = model_transmission(mkC(m), wd0, c, nb)
        vil = DynamicPPL.link!!(DynamicPPL.VarInfo(Random.Xoshiro(c.seed), mdl), mdl)
        (mdl, vil)
    end
    # #1 pays the rule derivation; #2..5 are what draws 2..100 of a real cell actually pay, and use
    # a DIFFERENT C* each time — same types, different values, exactly as fit_stage2_pooled does.
    m1, v1 = build(1)
    t_prep1 = @elapsed ldf = DynamicPPL.LogDensityFunction(m1, DynamicPPL.getlogjoint_internal, v1;
                                                           adtype = adt)
    t_prepN = let ts = Float64[]
        for m in 2:5
            mm, vv = build(m)
            push!(ts, @elapsed DynamicPPL.LogDensityFunction(mm, DynamicPPL.getlogjoint_internal,
                                                             vv; adtype = adt))
        end
        mean(ts)
    end
    x0 = collect(Float64, v1[:])
    LDP.logdensity_and_gradient(ldf, x0)                       # warm the call itself
    n  = 200
    gps = n / @elapsed(for _ in 1:n; LDP.logdensity_and_gradient(ldf, x0); end)

    mdl2 = model_transmission(mkC(2), wd0, c, nb)
    pathfinder(mdl2; ndraws = 100, rng = Random.Xoshiro(c.seed), adtype = adt)   # warm
    t_pf = @elapsed pathfinder(model_transmission(mkC(3), wd0, c, nb);
                               ndraws = 100, rng = Random.Xoshiro(c.seed), adtype = adt)
    @printf("  %-12s %8d %12.2f %14.3f %14.0f %14.2f\n",
            backend, length(x0), t_prep1, t_prepN, gps, t_pf)
end

say("""
Read: `prep #2..5` and `pathfinder` are the per-Stage-1-draw costs a real cell pays 100 times.
`grad/s` is the quantity framework.jl's ad_backend docstring quotes — if Mooncake still wins there
while losing the `pathfinder` column, the loss is per-fit overhead, not gradient throughput.""")
