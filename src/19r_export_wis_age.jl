# 19r_export_wis_age.jl — the per-cell scores behind `19r_wis_bam.ipynb` (2026-11-03, user request).
#
# 19r regresses log-scale WIS for each (series, origin, horizon, age bin) on the scenario and contact-model
# features, in R with mgcv `bam`. No CSV holds WIS at that resolution: 17j writes it averaged over age, and 18j caches
# only `by_model_h_age`. This script scores the 22 scenario series from the same four forecast caches 18j reads
# (`scenario_assembly_sources`) and writes one row per forecast unit.
#
# Reads with `load_9j_assembly`, which cannot fit, and calls nothing that does. Needs no CoMix or inc2prev read.
# Run from `src/` after 17j:
#     julia --project=.. --sysimage=../sysimage.so 19r_export_wis_age.jl

ENV["GKSwstype"] = "100"
include("forecast_utils.jl")
include("9j_viz_utils.jl")
include("15j_viz_utils.jl")
include("17j_viz_utils.jl")
include("18j_viz_utils.jl")
using Statistics, DataFrames, CSV

cfg = FrameworkConfig(constant_contacts = false)
@assert contacts_label(cfg) == "temporal-w8h-lc0-eng-nuts"
PRE_OMICRON_END = Date(2021, 11, 1)                # = 17j cell 24 / 18j
HORIZONS = collect(cfg.horizons)
grid = cis_age_grid()

# The 22 series and their scenario columns, from 17j's own per-origin scores.
sc17 = CSV.read("../res/17j_scenarios_scores_by_model_date_horizon.csv", DataFrame; stringtype = String)
meta = unique(select(sc17, :series, :scenario, :label, :frozen))
@assert nrow(meta) == 22 && allunique(meta.series)
keep = Set(meta.series)

qs = DataFrame[]
for (path, labels, frozen) in scenario_assembly_sources(cfg)
    a = load_9j_assembly(cfg; labels, cache_path = path, grid)
    q, _ = rename_series(a.qall, a.fc_store, frozen)
    push!(qs, @subset(q, :forecast_date .<= PRE_OMICRON_END, in.(:horizon, Ref(HORIZONS)), in.(:model, Ref(keep))))
    println(basename(path), ": ", length(unique(qs[end].model)), " series")
    a = nothing; GC.gc()
end
qall = vcat(qs...)
@assert Set(qall.model) == keep "series missing from the caches: $(setdiff(keep, Set(qall.model)))"

# Truth is model-invariant (asserted by `assembly_truth`); `target_date` comes with it.
truth = unique(select(qall, :forecast_date, :target_date, :horizon, :age_group, :observed))
@assert allunique(select(truth, :forecast_date, :horizon, :age_group)) "observed is not model-invariant"

sc = score_wis(qall)
d = select(@subset(sc.by_model_dt_h_age, :scale .== "log"), :model => :series, :forecast_date, :horizon, :age_group, :wis)
d = innerjoin(innerjoin(d, meta; on = :series), truth; on = [:forecast_date, :horizon, :age_group])

n_orig = length(unique(d.forecast_date))
@assert n_orig == 55 "expected 55 pre-Omicron origins, got $(n_orig)"
@assert nrow(d) == 22 * n_orig * length(HORIZONS) * grid.N "got $(nrow(d)) rows"
@assert allunique(select(d, :series, :forecast_date, :horizon, :age_group))
@assert Set(d.age_group) == Set(grid.LAB)
@assert all(>(0), d.wis) && all(>(0), d.observed)

# Tripwire for a transposed age or horizon index: the mean over the 7 bins is 17j's per-origin WIS.
let m = combine(groupby(d, [:series, :forecast_date, :horizon]), :wis => mean => :wis_mean),
    j = innerjoin(m, select(@subset(sc17, :scale .== "log"), :series, :forecast_date, :horizon, :wis);
                  on = [:series, :forecast_date, :horizon])
    @assert nrow(j) == nrow(m) "rows missing from 17j's CSV"
    rel = maximum(abs.(j.wis_mean .- j.wis) ./ j.wis)
    @assert rel < 1e-9 "per-age WIS does not average to 17j's WIS (max rel diff $(rel))"
    println("checked: mean over age = 17j's log WIS for ", nrow(j), " series × origin × horizon (max rel diff ", rel, ")")
end

sort!(d, [:horizon, :series, :forecast_date, :age_group])
select!(d, :series, :label, :scenario, :frozen, :forecast_date, :target_date, :horizon, :age_group, :wis, :observed)
out = joinpath("../dt_intermediate", "19r_wis_age_time_$(contacts_label(cfg)).csv")
CSV.write(out, d)
println("wrote ", out, ": ", nrow(d), " rows; log WIS range ", extrema(d.wis))
