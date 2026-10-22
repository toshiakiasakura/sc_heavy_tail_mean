# 15j_viz_utils.jl — helpers for `15j_multivariate_scoring.ipynb`.
#
# 9j scores the six forecast models with WIS alone, and WIS is UNIVARIATE: it scores each
# (origin × horizon × age bin) cell on its own and averages afterwards. Nothing in the tree asked
# whether a model gets the JOINT forecast right — whether the 7 CIS age bins move together as it
# says they should. 15j adds the two standard multivariate proper scoring rules for that question
# and compares them against WIS on identical forecasts.
#
# ⚠ SINCE 2026-10-16 THE ONLY MULTIVARIATE MARGIN REPORTED IS `"age"` — the 7-bin CIS vector at a
# fixed horizon. The horizon-spanning margins (`"time"`, `"age-time"`) were dropped because
# `two_stage_forecast` fits every horizon separately, with its own RNG and a MEDIAN lag plug, so the
# model emits no cross-horizon joint predictive to score. See `MARGIN_ORDER` for the full argument
# and `decouple_check` for the measurement that settled it. Both margins stay COMPUTABLE behind an
# explicit `margins =`; they are simply not reported.
#
#   §A  `load_9j_assembly` / `assembly_truth` — read 9j's assembly cache, validate it, recover truth
#   §B  `_margin_groups` / `_mv_fill!` / `mv_scores` — the ES/VS grid over the reported margins
#   §C  `wis_long` / `wis_aggregate` — the MATCHED WIS comparator, summed over the same margins
#   §D  `summarise_mvscores` / `summarise_mvscores_by_horizon` / `score_concordance` — headline
#       tables, the per-horizon tables §7 ranks on, and rank agreement
#   §E  `coordinate_shares` — is the log scale enough, or is one coordinate carrying the score?
#   §F  `decouple_check` — the permutation test that measures how much dependence these scores see
#   §G  `es_cross_check_r` / `verify_es_reduces_to_crps` / `log_scale_matches_wis` /
#       `scoringutils_semantics_check` — the verification suite 15j §5 runs
#   §H  the nine figure builders (one of which the notebook calls four times) + `report_mvscores`
#
# Requires (in this order): `forecast_utils.jl`, then `9j_viz_utils.jl`.
# From 9j this file uses exactly three names — `REF_MODEL` and `WIS_SCALE` (default arguments only,
# so they are evaluated at CALL time), `panel_grid` (every layout; never hard-code one) and
# `shade_periods!` (the Munday-2023 phase bands, in `plot_score_diff_over_time` only). All three are
# referenced from inside function bodies or default arguments, so this file loads standalone.
# ⚠ Do NOT introduce a top-level `const` built from any of them — that would evaluate at include
# time and break standalone loading. It is why 14j's `_period_cols` is a function, and the same
# rule applies here.
# The notebook — not this file — calls `save_show` on each returned `Plot`, following 9j's
# convention that the helper builds figures and the notebook decides where they go.
#
# ⚠ NOTHING HERE FITS, AND NOTHING HERE TOUCHES `dt_intermediate` EXCEPT ONE `isfile` + `load` OF A
# SINGLE NAMED `.jld2`. There is no `two_stage_forecast`, `fit_or_load_*`, `prefit_*`,
# `assemble_or_load_forecasts`, `load_forecast_truth`, `load_window_data` or `prepare_degree_data`
# call anywhere in this file (15j §5 asserts that mechanically with a grep, because a claim in a
# header comment is not a check). Those paths FIT ON MISS and write into `../dt_intermediate` under
# the live token — CLAUDE.md's first Gotcha, and how 9j comes to grind for days.
#   A consequence worth stating, because it is the reason 15j needs no `contacts =` discipline:
# there is no `reconstruct_*` / `stage1_chain_path` / `stage2_pooled_path` call either, so the
# blank-but-plausible `CONTACTS_TOKEN`-default family of bugs cannot apply here at all.
#
# ⚠ THE SCORES ARE ON THE LOG SCALE, AND FOR THESE TWO RULES THAT IS NOT A PREFERENCE.
# `score_wis` only ever drops NON-FINITE predictions, and WIS is structurally immune to the tail
# anyway because `to_quantile_long` keeps just the 5–95% quantiles. ES and VS use EVERY draw, and
# the 2026-09-16 grid contains extreme but perfectly FINITE draws — up to 3.80e109 — that every
# existing guard passes. Measured at the worst cell (2021-08-29, weighted-hweibull|neighbourhood,
# h=3): natural-scale ES 6.1e101 and VS(p=0.5) 1.2e103, against 1.30 and 7.86 under `log(x+1)`.
# So a natural-scale panel is not a worse view of the same thing, it is a view of one draw.

using Glob, Random, Statistics, StatsBase

####################################################################################################
# §A — the assembly cache: read it, validate it, recover truth
####################################################################################################

"The six model colours 9j's figures use, so a model keeps ONE colour across `res/8j_*`, `res/9j_*`
and `res/15j_*`. ⚠ This duplicates the literal in `9j_forecast_diagnostics.ipynb` cell 4, which
cannot be imported because it lives in a notebook cell; the clean fix is to move it into
`9j_viz_utils.jl` and have 9j use it from there. Deliberately NOT 14j's Dark2 `_model_cols` — 15j's
figures are read beside 9j's, not beside 14j's."
const MODEL_COLS = [:steelblue, :darkorange, :seagreen, :purple, :firebrick, :grey40]

"""
The margin choices REPORTED, in reporting order. The name says which axes form the multivariate
vector: `"age"` scores the 7-bin CIS vector (one score per origin × horizon) and `"none"` is the
univariate baseline where ES degenerates to `crps_sample`.

⚠ **`"time"` and `"age-time"` were REMOVED from this tuple on 2026-10-16 (user request), and the
reason is structural rather than editorial.** `two_stage_forecast` fits every horizon with its own
`fit_or_load_stage2` and its own `MersenneTwister(cfg.seed + h)`, and plugs the intervening lag with
the MEDIAN over draws — so **the model emits no cross-horizon joint predictive at all**. A vector
spanning the horizon axis is a definition this file would be imposing, not a property the forecast
has, and `decouple_check` measured exactly that (ES moved ≤0.06% when the horizon pairing was
destroyed). Reporting such a score invites a reader to interpret a number the model never produced.
Both margins remain COMPUTABLE — `_margin_groups` and `wis_aggregate` still carry their branches —
so passing `margins = ("time", "age-time")` explicitly still works; they are simply not reported.

⚠ **`"none"` is NOT a reported view and must not be "tidied" away either.** It is the univariate
baseline that carries this notebook's sharpest verification (ES ≡ `crps_sample` at `d == 1`,
asserted over 10 584 rows) and the only place WIS and CRPS are compared on an identical cell.
"""
const MARGIN_ORDER = ("none", "age")

"Variogram orders computed. **`0.5` is the HEADLINE** and is named in every column, label and title;
`1.0` exists only to show whether the model ranking is `p`-stable, which is free because VS is
`O(M·d²)` against ES's `O(M²·d)`."
const VS_P = (0.5, 1.0)

"Headline scale for the multivariate scores. ES, VS and WIS must be transforms of the same quantity
or the comparison is between scales rather than between rules — but this is a LITERAL and not
`WIS_SCALE`, because `WIS_SCALE` lives in `9j_viz_utils.jl`, which this file does not require at
load time (see the header). The notebook asserts `WIS_SCALE == MV_SCALE == \"log\"` instead, which
keeps the tie checked rather than merely intended."
const MV_SCALE = "log"

"`|draw|` above this is flagged as pathological. 1e8 is ~130× the largest observed weekly infection
count in the grid (7.6e5), so nothing legitimate reaches it. Reported, NEVER filtered — see the
header note; this is the guard class `score_wis` cannot see."
const EXTREME_DRAW = 1e8

"""
    load_9j_assembly(cfg; labels, save_dir, cache_path, grid, n_draw_expected) -> NamedTuple

Read 9j's forecast assembly cache and validate it. Returns
`(; qall, fc_store, crps, origins, labels, path, bytes, mtime, n_draw)`.

This is `isfile` → `load` → check, and **nothing else**. It deliberately does NOT call
`assemble_or_load_forecasts` (`9j_viz_utils.jl:152`), which needs the CoMix `raw` read and, on a
miss, falls through to `two_stage_forecast` → `fit_or_load_stage2` → `fit_or_load_stage1` and
**fits**, serially, writing into the live token. Refusing to have that function in 15j's call graph
at all is the no-refit guarantee.

`labels` is a REQUIRED kwarg with no default so a caller cannot silently skip the check; the
notebook builds it independently from the same `combos` literal 9j uses, which makes the comparison
a real cross-check rather than a tautology. `origins` is deliberately NOT re-derived — that would
need `available_forecast_origins`, i.e. the CoMix read this notebook otherwise avoids — so its
SHAPE is asserted instead (length, endpoints, sortedness, exact weekly spacing), which catches
every failure mode a re-derivation would.

⚠ **This cache validates `origins` and `labels` only — names, not contents.** On 2026-09-16 the
infection source moved from `estimates_age_ab.csv` to `estimates_age.csv`; origins and labels were
both unchanged, so a stale cache would have been served silently and had to be deleted by hand
(`tasks/lessons.md`, and CLAUDE.md's inc2prev-schema Gotcha). A docstring warning nobody runs is
worth less than two `mtime` comparisons, so this function compares the cache's mtime against the
estimates CSV and against the newest `8j_s2_*` artefact and `@warn`s if either is newer.
"""
function load_9j_assembly(cfg::FrameworkConfig; labels::AbstractVector{<:AbstractString},
                          save_dir::AbstractString = "../dt_intermediate",
                          cache_path::AbstractString =
                              joinpath(save_dir, "9j_assembly_$(contacts_label(cfg)).jld2"),
                          grid = cis_age_grid(), n_draw_expected::Integer = 10_000,
                          est_path::AbstractString = joinpath(@__DIR__, "..", "inc2prev",
                                                              "outputs", "estimates_age.csv"))
    isfile(cache_path) || error("""
        15j found no 9j assembly cache at
            $(cache_path)
        15j is a READER: it cannot build one, by design. Run 9j for token `$(contacts_label(cfg))`
        first (its cell 4 writes this file), or point `cache_path` at the generation you meant.""")

    c = load(cache_path)
    for k in ("qall", "fc_store", "crps", "skipped", "origins", "labels")
        haskey(c, k) || error("15j: assembly cache is missing key `$(k)` — $(cache_path)")
    end
    origins = c["origins"]::Vector{Date}
    got_lab = c["labels"]::Vector{String}

    # --- labels: a real cross-check against an independently built list -------------------------
    if got_lab != collect(labels)
        error("""
            15j: assembly cache model labels disagree with this notebook's `combos`.
              only in cache    : $(setdiff(got_lab, labels))
              only in notebook : $(setdiff(labels, got_lab))
              cache order      : $(got_lab)
              notebook order   : $(collect(labels))
            Order matters as well as membership — the figures key colour on it.""")
    end

    # --- origins: validate the SHAPE rather than re-deriving (see docstring) --------------------
    length(origins) == 67 || @warn "15j: expected the 67-origin grid" n = length(origins)
    issorted(origins) || error("15j: cache origins are not sorted")
    bad = findall(!=(Day(7)), diff(origins))
    isempty(bad) || error("15j: cache origins are not a contiguous weekly grid; gaps after " *
                          "$(origins[bad])")

    # --- the fans themselves --------------------------------------------------------------------
    fc_store = c["fc_store"]::Dict{Tuple{Date,String},Array{Float64,3}}
    want = [(o, l) for o in origins for l in got_lab]
    miss = [k for k in want if !haskey(fc_store, k)]
    isempty(miss) || error("15j: assembly cache is missing $(length(miss)) of $(length(want)) " *
                           "(origin, model) fans, e.g. $(first(miss, 3))")
    A, H = grid.N, length(cfg.horizons)
    for k in want
        sz = size(fc_store[k])
        sz == (A, H, n_draw_expected) || error("""
            15j: fan $(k) has size $(sz), expected $((A, H, n_draw_expected)).
            ⚠ The draw count is not cosmetic. `two_stage_forecast` sets
            `K = minimum(size(c,2) for c in cols)` (joint_model.jl:1918) and then copies by
            POSITIONAL PREFIX, so one short horizon silently truncates every horizon to its first
            K draws — and the draw axis is block-major (d = (m−1)·100 + s), so a prefix is a
            biased subsample of the Stage-1 blocks, not a thinning.""")
    end

    skipped = c["skipped"]
    if !isempty(skipped)
        @warn "15j: 9j recorded skipped origin×combo pairs — the models are NOT on common units, " *
              "and `mv_scores`' M-equalisation cannot repair a wholly absent fan" skipped
    end

    # --- names-not-contents: make the warning an actual check ----------------------------------
    cache_mt = mtime(cache_path)
    stale = String[]
    if isfile(est_path) && mtime(est_path) > cache_mt
        push!(stale, "$(basename(est_path)) ($(Dates.unix2datetime(mtime(est_path))))")
    end
    # The frozen-contact `_h0` cells (17j, 2026-10-17) are not inputs to 9j's CONDITIONAL cache —
    # counting them would raise this staleness warning every time the h0 grid is (re)fitted.
    s2 = filter(p -> !endswith(p, "_h$(FROZEN_H).jld2"), glob("8j_s2_*", save_dir))
    if !isempty(s2)
        newest = maximum(mtime, s2)
        newest > cache_mt && push!(stale, "the newest 8j_s2_* artefact " *
                                          "($(Dates.unix2datetime(newest)))")
    end
    if !isempty(stale)
        @warn """
            15j: the 9j assembly cache may be STALE. It validates `origins` and `labels` only —
            names, not contents — and both were unchanged by the 2026-09-16 switch from
            `estimates_age_ab.csv` to `estimates_age.csv`, so a stale cache is served silently.
            Newer than the cache ($(Dates.unix2datetime(cache_mt))): $(join(stale, "; ")).
            If the inputs really changed, DELETE this cache (and 9j's `9j_rt_*`/`9j_relrt_*`/
            `9j_obsrt_*`) and re-run 9j.""" cache_path
    end

    return (; qall = c["qall"], fc_store, crps = c["crps"], origins, labels = got_lab,
              path = cache_path, bytes = filesize(cache_path), mtime = cache_mt,
              n_draw = n_draw_expected)
end

"""
    assembly_truth(qall, grid; horizons) -> (; truth, frame, n_cells)

Recover the realized weekly infection counts from the assembly cache's own quantile table.

`to_quantile_long` (`scoring.jl:12`) writes the observation into every row as `observed`, so the
truth 9j actually scored against is already in `qall` and **no inc2prev read is needed** — which is
what keeps 15j a one-file reader. Returns `truth::Dict{Date,Matrix{Float64}}` (`A×H` per origin,
rows ordered by `grid.LAB`), the tidy `frame`, and the cell count.

The function asserts the observation is **model-invariant** — exactly one distinct `observed` per
(forecast_date, horizon, age_group). That single check simultaneously proves every model was scored
against the same truth and that the unit sets coincide, which is the precondition for comparing
any two scoring rules across models at all.
"""
function assembly_truth(qall::DataFrame, grid; horizons = 1:4)
    frame = unique(select(qall, [:forecast_date, :horizon, :age_group, :observed]))
    dup = combine(groupby(frame, [:forecast_date, :horizon, :age_group]), nrow => :n)
    bad = @subset(dup, :n .> 1)
    nrow(bad) == 0 || error("15j: `observed` is not model-invariant in $(nrow(bad)) cells — the " *
                            "models were scored against different truth, e.g. $(first(bad, 3))")
    all(frame.observed .>= 0) || error("15j: negative `observed` — these are counts, and the log " *
                                       "scale transforms the observation without truncating it")

    A, H = grid.N, length(horizons)
    lut = Dict((r.forecast_date, r.horizon, r.age_group) => r.observed for r in eachrow(frame))
    origins = sort(unique(frame.forecast_date))
    truth = Dict{Date,Matrix{Float64}}()
    for o in origins
        M = Matrix{Float64}(undef, A, H)
        for a in 1:A, h in 1:H
            key = (o, h, grid.LAB[a])
            haskey(lut, key) || error("15j: no observation for $(key)")
            M[a, h] = lut[key]
        end
        truth[o] = M
    end
    n_cells = length(origins) * A * H
    nrow(frame) == n_cells || error("15j: truth has $(nrow(frame)) rows, expected $(n_cells)")
    return (; truth, frame, n_cells)
end

####################################################################################################
# §B — the multivariate score grid
####################################################################################################

"""
    _margin_groups(margin, A, H) -> Vector{@NamedTuple{horizon, age_group, idx}}

The groups one `(model, origin)` fan contributes under `margin`. `idx` lists the `(age, horizon)`
cells forming the multivariate vector, in a FIXED order (`age` fastest) so a run is reproducible
and the R cross-check lines up. `horizon`/`age_group` are the group's free keys and are `nothing`
for whichever axis is *inside* the vector.

Both ES and uniformly-weighted VS are permutation-invariant, so the cell order cannot change a
score — it is pinned for comparability, not correctness.
"""
function _margin_groups(margin::AbstractString, A::Int, H::Int)
    T = @NamedTuple{horizon::Union{Nothing,Int}, age_group::Union{Nothing,Int},
                    idx::Vector{Tuple{Int,Int}}}
    out = T[]
    if margin == "none"
        for h in 1:H, a in 1:A
            push!(out, (horizon = h, age_group = a, idx = [(a, h)]))
        end
    elseif margin == "time"                       # vector = the H horizons, for one age bin
        # ⚠ NOT in `MARGIN_ORDER` — the model emits no cross-horizon joint predictive (separate
        # Stage-2 fit + RNG per horizon), so this vector is a definition, not a model property.
        # Retained because it is correct and reachable via an explicit `margins =`; see MARGIN_ORDER.
        for a in 1:A
            push!(out, (horizon = nothing, age_group = a, idx = [(a, h) for h in 1:H]))
        end
    elseif margin == "age"                        # vector = the A age bins, at one horizon
        for h in 1:H
            push!(out, (horizon = h, age_group = nothing, idx = [(a, h) for a in 1:A]))
        end
    elseif margin == "age-time"                   # vector = all A·H cells
        # ⚠ NOT in `MARGIN_ORDER`, same reason as "time" — it spans the horizon axis.
        push!(out, (horizon = nothing, age_group = nothing,
                    idx = [(a, h) for h in 1:H for a in 1:A]))
    else
        error("15j: unknown margin `$(margin)`; expected one of $(MARGIN_ORDER)")
    end
    return out
end

"Apply the reporting scale to one value. `\"log\"` is `log(x + 1)` on a copy truncated at 0 — the
same expression `scoringutils::log_shift(x, offset = 1)` evaluates, so the multivariate scores sit
on exactly WIS's scale. ⚠ `log(x+1)` and `log1p(x)` are NOT the same function (at `x = 1e-17` the
first is 0 and the second is 1e-17); writing the one `log_shift` uses makes that identity exact by
construction rather than by coincidence."
@inline function _apply_scale(v::Float64, scale::AbstractString)
    scale == "natural" && return (v, false)
    t = v < 0
    return (log(max(v, 0.0) + 1.0), t)
end

"""
    _mv_fill!(buf, fc, idx, scale, cols) -> (M, n_trunc)

Fill `buf` (`d × ≥length(cols)`) with the transformed draws of cells `idx` over draw indices `cols`,
then compact away any draw COLUMN holding a non-finite coordinate. Returns the surviving column
count and the number of coordinates truncated at 0 by the log scale.

⚠ **The granularity is the draw column, never the coordinate.** One non-finite coordinate makes the
whole `d`-vector unusable (`‖x_m − y‖` is `Inf`/`NaN`, poisoning both ES terms), and dropping a
single coordinate instead would change `d` between draws, which is not a score at all.
"""
function _mv_fill!(buf::Matrix{Float64}, fc::Array{Float64,3}, idx::Vector{Tuple{Int,Int}},
                   scale::AbstractString, cols::AbstractVector{Int})
    d = length(idx)
    ntr = 0
    @inbounds for (m, c) in enumerate(cols)
        for k in 1:d
            a, h = idx[k]
            v, t = _apply_scale(fc[a, h, c], scale)
            buf[k, m] = v
            t && (ntr += 1)
        end
    end
    n = length(cols)
    keep = 0
    @inbounds for m in 1:n
        ok = true
        for k in 1:d
            if !isfinite(buf[k, m])
                ok = false
                break
            end
        end
        if ok
            keep += 1
            if keep != m
                for k in 1:d
                    buf[k, keep] = buf[k, m]
                end
            end
        end
    end
    return keep, ntr
end

"""
    _clean_draws(fc) -> Vector{Int}

Draw indices that are finite in EVERY cell of the fan. Used to equalise `M` across models before
scoring (see `mv_scores`); on a healthy grid this is `1:D`.
"""
function _clean_draws(fc::Array{Float64,3})
    A, H, D = size(fc)
    out = Int[]
    @inbounds for m in 1:D
        ok = true
        for h in 1:H, a in 1:A
            if !isfinite(fc[a, h, m])
                ok = false
                break
            end
        end
        ok && push!(out, m)
    end
    return out
end

"Take `n` of the clean draw indices `ci`, spread EVENLY over the vector. ⚠ The draw axis is
block-major — `d = (m−1)·n_stage2 + s`, 100 contiguous blocks of 100 — so a CONTIGUOUS PREFIX is a
biased subsample: measured, `1:1000` carries 10 of the 100 Stage-1 blocks and shifts a cell mean
4% (4593 vs 4784). An even spread over the sorted clean set touches every block, which is
`_stratified_draws`' rule (`14j_viz_utils.jl`) applied without needing `post_index`. `1:100:10_000`
would also work, but only by the accident `n_stage1_post == n_stage2_draws`."
function _even_draws(ci::Vector{Int}, n::Integer)
    n >= length(ci) && return ci
    n <= 1 && return ci[1:1]
    return ci[round.(Int, range(1, length(ci); length = n))]
end

"""
    mv_scores(fc_store, truth, labels, origins; grid, cfg, scales, margins, vs_p) -> (; scores, audit)

The energy and variogram score of every `(model, origin, margin, scale, group)` on the grid, with
the matched per-cell CRPS alongside.

`scores` columns: `model, origin, margin, scale, horizon, age_group, d, M, n_draw_dropped,
n_trunc, n_extreme, max_abs_draw, es, vs_p05, vs_p1, crps_sum, crps_mean`. `horizon`/`age_group`
are `missing` for whichever axis the vector spans.

**`M` is equalised across models before scoring.** ES's sample estimator is biased, and its bias
depends on `M` (the `−1/(2M²)ΣΣ` term), so two groups scored at different `M` are two different
estimators and the comparison would partly be of `M`. Each model's usable draws at an origin are
those finite in every cell (`_clean_draws`); every model at that origin is then cut to the smallest
such count, taking draws evenly over the block-major axis (`_even_draws`).
⚠ This is **not** aligning draws. Draw index `d` means nothing across models — each has its own
Stage-1/Stage-2 fits and its own observation RNG — so nothing is being matched up; the cut only
equalises the estimator's bias.

⚠ `scales` defaults to `("log",)`. The natural scale is available and is worth computing ONCE for
the audit (it is the evidence that WIS's quantile clipping is what makes WIS immune to the extreme
finite draws), but it must never be plotted or quoted as a headline — see the file header.

`fast_univariate` (default `true`) computes the `margin = "none"` energy score as `crps_sample`
rather than through the O(M²) kernel. The two are the SAME NUMBER — that is the documented identity
in `energy_score` — and the shortcut is worth having because the univariate baseline is 28 of the
40 groups a fan contributes and was measured at ~5300 s of the sweep's ~9000 s CPU. Pass `false` to
force every group through the general kernel, which is what 15j §5's exactness check does.
"""
function mv_scores(fc_store, truth, labels, origins; grid = cis_age_grid(), cfg,
                   scales = (MV_SCALE,), margins = MARGIN_ORDER, vs_p = VS_P,
                   fast_univariate::Bool = true, verbose::Bool = true)
    A, H = grid.N, length(cfg.horizons)
    @assert 0.5 in vs_p "15j 'report the variogram at p = 0.5' — p=0.5 must be in vs_p, got $(vs_p)"

    # ---- M equalisation, one cheap scan ------------------------------------------------------
    clean = Dict{Tuple{Date,String},Vector{Int}}()
    for o in origins, l in labels
        clean[(o, l)] = _clean_draws(fc_store[(o, l)])
    end
    mcommon = Dict(o => minimum(length(clean[(o, l)]) for l in labels) for o in origins)
    usecols = Dict(k => _even_draws(v, mcommon[k[1]]) for (k, v) in clean)
    if verbose
        nd = sort(unique(values(mcommon)))
        println("M equalisation: draws used per origin = ", length(nd) == 1 ? string(nd[1]) :
                string(minimum(nd), " … ", maximum(nd), " (", length(nd), " distinct)"))
    end

    groupspec = Dict(m => _margin_groups(m, A, H) for m in margins)
    dmax = maximum(length(g.idx) for m in margins for g in groupspec[m])

    jobs = [(o, l) for o in origins for l in labels]
    nchunk = max(1, min(Threads.nthreads(), length(jobs)))
    chunks = [jobs[i:nchunk:end] for i in 1:nchunk]
    parts = Vector{Vector{NamedTuple}}(undef, nchunk)

    @sync for (ci, ch) in enumerate(chunks)
        Threads.@spawn begin
            # One buffer per CHUNK, not per thread id: task migration makes `threadid()` indexing
            # unsafe on 1.12, and chunk-local state sidesteps the question entirely.
            buf = Matrix{Float64}(undef, dmax, maximum(values(mcommon)))
            rows = NamedTuple[]
            for (o, l) in ch
                fc = fc_store[(o, l)]
                T = truth[o]
                cols = usecols[(o, l)]
                D = size(fc, 3)
                for margin in margins, g in groupspec[margin]
                    idx = g.idx
                    d = length(idx)
                    y_nat = [T[a, h] for (a, h) in idx]
                    nex = 0
                    mx = 0.0
                    @inbounds for (a, h) in idx, m in cols
                        v = abs(fc[a, h, m])
                        v > mx && (mx = v)
                        v > EXTREME_DRAW && (nex += 1)
                    end
                    for scale in scales
                        M, ntr = _mv_fill!(buf, fc, idx, scale, cols)
                        X = view(buf, 1:d, 1:M)
                        y = [first(_apply_scale(v, scale)) for v in y_nat]
                        cr = [crps_sample(view(buf, k, 1:M), y[k]) for k in 1:d]
                        # EXACT fast path at d == 1: the energy score of a one-coordinate vector
                        # IS `crps_sample` (same 1/M² convention), and `crps_sample` gets it in
                        # O(M log M) where `energy_score` pays O(M²). This is not an
                        # approximation, and it is not free of obligation either: the identity is
                        # what licenses the shortcut, so 15j §5 verifies it on a sample of real
                        # groups THROUGH the O(M²) path before relying on it here.
                        #   It matters because the univariate baseline is 28 of the 40 groups a
                        # fan contributes, so it was ~60% of the whole sweep's cost (measured:
                        # ~5300 s of ~9000 s CPU) to recompute a number already in `cr`.
                        es = (d == 1 && fast_univariate) ? cr[1] : energy_score(X, y)
                        vs = Dict(p => variogram_score(X, y; p = p) for p in vs_p)
                        push!(rows, (model = l, origin = o, margin = margin, scale = scale,
                                     horizon = g.horizon === nothing ? missing : g.horizon,
                                     age_group = g.age_group === nothing ? missing :
                                                 grid.LAB[g.age_group],
                                     d = d, M = M,
                                     n_draw_dropped = D - M,
                                     n_trunc = ntr, n_extreme = nex, max_abs_draw = mx,
                                     es = es,
                                     vs_p05 = vs[0.5], vs_p1 = get(vs, 1.0, NaN),
                                     crps_sum = sum(cr), crps_mean = mean(cr)))
                    end
                end
            end
            parts[ci] = rows
        end
    end

    scores = DataFrame(reduce(vcat, parts))
    # Deterministic row order regardless of how the chunks were scheduled. `horizon`/`age_group`
    # carry `missing` for whichever axis is inside the vector, and `isless` has no total order
    # across `missing` and a String, so sort on explicit surrogate keys instead of the raw columns.
    scores[!, :_hk] = coalesce.(scores.horizon, -1)
    scores[!, :_ak] = [ismissing(x) ? "" : x for x in scores.age_group]
    sort!(scores, [:margin, :scale, :model, :origin, :_hk, :_ak])
    select!(scores, Not([:_hk, :_ak]))

    audit = combine(groupby(@subset(scores, :margin .== "none", :scale .== first(scales)),
                            :model),
                    nrow => :n_cells,
                    :M => sum => :n_draws_used,
                    :n_extreme => sum => :n_extreme,
                    :max_abs_draw => maximum => :max_abs_draw,
                    :M => minimum => :M_min, :M => maximum => :M_max)
    if "log" in scales
        lg = @subset(scores, :margin .== "none", :scale .== "log")
        audit = leftjoin(audit, combine(groupby(lg, :model), :n_trunc => sum => :n_trunc_log);
                         on = :model)
    end
    audit = leftjoin(audit, DataFrame(model = collect(labels),
                                      n_draws_clean = [sum(length(clean[(o, l)]) for o in origins)
                                                       for l in labels],
                                      n_draws_total = [length(origins) * A * H *
                                                       size(fc_store[(first(origins), l)], 3)
                                                       for l in labels]); on = :model)

    nd = sum(scores.n_draw_dropped)
    nd == 0 || @warn "15j: $(nd) draw columns were dropped for a non-finite coordinate. ES's " *
                     "bias depends on M, so check `audit.M_min`/`M_max` before comparing models." 
    ne = sum(@subset(scores, :margin .== "none", :scale .== first(scales)).n_extreme)
    ne == 0 || @warn """
        15j: $(ne) draws exceed |$(EXTREME_DRAW)| — finite, so every existing guard passes them,
        and `score_wis` never sees them because `to_quantile_long` keeps only the 5–95% quantiles.
        ES and VS use every draw, so the NATURAL scale is not usable for those models. This is
        reported, not filtered: see `audit.max_abs_draw`.""" 
    return (; scores, audit)
end

####################################################################################################
# §C — the matched WIS comparator
####################################################################################################

"The log-scale WIS at the FULL forecast unit, renamed onto `mv_scores`' key schema."
function wis_long(scores; scale::AbstractString = MV_SCALE)
    haskey(scores, :by_model_dt_h_age) || error("""
        15j: `score_wis` did not return `by_model_dt_h_age`. That frame is the matched comparator
        for a multivariate score and was added to `scoring.jl` for 15j — an older `scoring.jl` is
        loaded.""")
    wl = @subset(scores.by_model_dt_h_age, :scale .== scale)
    return rename(select(wl, [:model, :forecast_date, :horizon, :age_group, :wis]),
                  :forecast_date => :origin)
end

"""
    wis_aggregate(wl, margin) -> DataFrame

Sum the full-unit WIS over **exactly** the margins the energy score's vector spans, giving
`model, origin, margin, horizon, age_group, wis_sum` on `mv_scores`' keys.

This is the whole reason `scoring.jl` gained `by_model_dt_h_age`: a metric comparison is only
honest if the aggregation is held fixed, and `by_model_h_age` averages over origins *before* you
can sum over age, which would conflate the aggregation choice with the metric.
"""
function wis_aggregate(wl::DataFrame, margin::AbstractString)
    if margin == "none"
        out = @transform(wl, :wis_sum = :wis)
        out = select(out, [:model, :origin, :horizon, :age_group, :wis_sum])
    elseif margin == "time"                       # ⚠ not in MARGIN_ORDER — see that docstring
        out = combine(groupby(wl, [:model, :origin, :age_group]), :wis => sum => :wis_sum)
        out[!, :horizon] = Vector{Union{Missing,Int}}(missing, nrow(out))
    elseif margin == "age"
        out = combine(groupby(wl, [:model, :origin, :horizon]), :wis => sum => :wis_sum)
        out[!, :age_group] = Vector{Union{Missing,String}}(missing, nrow(out))
    elseif margin == "age-time"                   # ⚠ not in MARGIN_ORDER — see that docstring
        out = combine(groupby(wl, [:model, :origin]), :wis => sum => :wis_sum)
        out[!, :horizon] = Vector{Union{Missing,Int}}(missing, nrow(out))
        out[!, :age_group] = Vector{Union{Missing,String}}(missing, nrow(out))
    else
        error("15j: unknown margin `$(margin)`; known: none / time / age / age-time " *
              "(reported: $(MARGIN_ORDER))")
    end
    out[!, :margin] .= margin
    out[!, :horizon] = Vector{Union{Missing,Int}}(out.horizon)
    out[!, :age_group] = Vector{Union{Missing,String}}(out.age_group)
    return select(out, [:model, :origin, :margin, :horizon, :age_group, :wis_sum])
end

"All four margins' matched WIS, stacked."
wis_aggregate_all(wl::DataFrame; margins = MARGIN_ORDER) =
    reduce(vcat, (wis_aggregate(wl, m) for m in margins))

####################################################################################################
# §D — headline tables and rank agreement
####################################################################################################

"The metric columns reported, in order. `wis_sum` is the matched comparator; `vs_p05` is the
HEADLINE variogram score and `vs_p1` is only the `p`-stability check."
const MV_METRICS = (:wis_sum, :es, :vs_p05, :vs_p1, :crps_sum)

"Pretty metric labels, with `p` named so no reader has to go looking for which order produced a
number."
mv_metric_label(m::Symbol) = m === :wis_sum  ? "WIS (summed)"   :
                             m === :es       ? "energy score"   :
                             m === :vs_p05   ? "variogram p=0.5" :
                             m === :vs_p1    ? "variogram p=1"   :
                             m === :crps_sum ? "Σ CRPS"          : string(m)

"Join `mv_scores`' frame to the matched WIS on the shared keys, one row per scored group."
function mv_join_wis(mv::DataFrame, wisagg::DataFrame; scale::AbstractString = MV_SCALE)
    s = @subset(mv, :scale .== scale)
    out = leftjoin(s, wisagg; on = [:model, :origin, :margin, :horizon, :age_group],
                   matchmissing = :equal)
    nmiss = count(ismissing, out.wis_sum)
    nmiss == 0 || @warn "15j: $(nmiss) of $(nrow(out)) groups found no matched WIS row" 
    nrow(out) == nrow(s) || error("15j: the WIS join changed the row count " *
                                  "($(nrow(s)) → $(nrow(out))) — a key is not unique")
    return out
end

"""
    summarise_mvscores(joined; ref, metrics) -> DataFrame

Per (margin, metric, model): `n_groups`, `score_mean`, `score_median`, `rel_mean`, `rel_median`,
`rank_mean`, `rank_median`.

⚠ **Relative skill is the MEDIAN of per-group ratios, not the ratio of means**, and both are
returned. Even on the log scale a mean is outlier-sensitive — the cache's own `crps` frame has one
cell at 5.14e7 against ~5.5e4 for its peers — so when `rel_mean` and `rel_median` disagree about a
winner, **that disagreement is the result**: it says one origin carries the mean. Print both.
"""
function summarise_mvscores(joined::DataFrame; ref::AbstractString = REF_MODEL,
                            metrics = MV_METRICS)
    rows = NamedTuple[]
    for margin in unique(joined.margin)
        sub = @subset(joined, :margin .== margin)
        for met in metrics
            # Skip a metric that has nothing to say for this margin. VS is NaN at `d == 1` by
            # construction (a one-coordinate field has no pairwise structure), so the univariate
            # baseline legitimately has no variogram score — and a row of NaNs would still get a
            # rank, which is worse than no row.
            vals = sub[!, met]
            all(x -> ismissing(x) || !isfinite(x), vals) && continue
            refrows = @subset(sub, :model .== ref)
            rl = Dict((r.origin, r.horizon, r.age_group) => r[met] for r in eachrow(refrows))
            for m in unique(sub.model)
                ms = @subset(sub, :model .== m)
                v = collect(skipmissing(ms[!, met]))
                isempty(v) && continue
                rr = Float64[]
                for r in eachrow(ms)
                    rv = get(rl, (r.origin, r.horizon, r.age_group), missing)
                    (ismissing(rv) || ismissing(r[met]) || !isfinite(rv) || rv == 0) && continue
                    push!(rr, r[met] / rv)
                end
                vf = filter(isfinite, v)
                push!(rows, (margin = margin, metric = String(met),
                             metric_label = mv_metric_label(met), model = m,
                             n_groups = length(vf),
                             score_mean = isempty(vf) ? NaN : mean(vf),
                             score_median = isempty(vf) ? NaN : median(vf),
                             rel_mean = isempty(rr) ? NaN : mean(rr),
                             rel_median = isempty(rr) ? NaN : median(rr)))
            end
        end
    end
    out = DataFrame(rows)
    # ranks (1 = best = lowest score) within each margin × metric
    out[!, :rank_median] = zeros(Int, nrow(out))
    out[!, :rank_mean] = zeros(Int, nrow(out))
    for g in groupby(out, [:margin, :metric])
        g.rank_median .= invperm(sortperm(g.score_median))
        g.rank_mean   .= invperm(sortperm(g.score_mean))
    end
    return sort(out, [:margin, :metric, :rank_median])
end

"""
    summarise_mvscores_by_horizon(joined; ref, metrics, margin) -> DataFrame

`summarise_mvscores` sliced by HORIZON: the same per-model table, computed independently within
each lead time, with a `:horizon` column prepended to the result. This is what
`plot_score_rank_bump_by_horizon` ranks on.

Why per-horizon at all: with the horizon-spanning margins gone (see `MARGIN_ORDER`), lead time is
the only free axis left on the one reported multivariate margin, and the ranking genuinely moves
along it. Measured on the 2026-10-15 grid at `margin = "age"`, 63 origins per (model, horizon):
**by MEAN, Σ WIS, Σ CRPS and the energy score give the identical ordering at all four horizons,
while both variogram orders do not.** That contrast is the cleanest evidence in the notebook that
VS measures a different quantity — and it sits beside the known reason, that VS is shift-invariant
and so not proper for the marginals.

⚠ **This is a WRAPPER, not a generalisation of `summarise_mvscores`' grouping.** Each call sees one
horizon, so the ranker's existing `groupby(out, [:margin, :metric])` already produces per-horizon
ranks with no change — and the four other call sites of `summarise_mvscores` are untouched.

⚠ **Subset `:margin` first, THEN `:horizon`, in two calls.** `joined.horizon` is
`Union{Missing,Int}` — `wis_aggregate` writes `missing` sentinels for the marginalised frames — and
a single `@subset(joined, :margin .== margin, :horizon .== h)` evaluates the horizon predicate over
every row, including the `missing` ones, and throws. `plot_rel_score_by_horizon` uses the same
two-step pattern for the same reason.
"""
function summarise_mvscores_by_horizon(joined::DataFrame; ref::AbstractString = REF_MODEL,
                                       metrics = MV_METRICS, margin::AbstractString = "age")
    sub = @subset(joined, :margin .== margin)              # margin FIRST — see the docstring
    nrow(sub) == 0 && error("15j: no rows at margin `$(margin)` — nothing to rank by horizon")
    hz = sort(unique(skipmissing(sub.horizon)))
    isempty(hz) && error("15j: margin `$(margin)` has no horizon axis — it is inside the vector")
    out = DataFrame[]
    for h in hz
        sh = @subset(sub, :horizon .== h)                  # ...horizon SECOND
        s = summarise_mvscores(sh; ref = ref, metrics = metrics)
        s[!, :horizon] .= h
        push!(out, s)
    end
    res = reduce(vcat, out)
    return sort(select(res, vcat([:horizon], Symbol.(names(res, Not(:horizon))))),
                [:horizon, :metric, :rank_median])
end


"""
    score_concordance(joined, summ; metrics) -> (; model_level, group_level, discordant)

Does the energy score rank the models the way WIS does, and does it rank the *origins* the way WIS
does?

- `model_level`: Spearman/Kendall between each metric pair over the **6 model medians**, with
  `n_discordant` out of the 15 model pairs. ⚠ A rank correlation on n = 6 is coarse — the
  discordant-pair count is the more readable number and is why it is reported alongside.
- `group_level`: the same correlations computed **per model over its 63–1764 groups**, which
  answers the better-powered question: given a model, do the two rules agree about which origins
  it forecast badly?
"""
function score_concordance(joined::DataFrame, summ::DataFrame; metrics = MV_METRICS)
    pairs = [(a, b) for (i, a) in enumerate(metrics) for b in metrics[(i + 1):end]]
    mrows = NamedTuple[]
    for margin in unique(summ.margin), (a, b) in pairs
        sa = @subset(summ, :margin .== margin, :metric .== String(a))
        sb = @subset(summ, :margin .== margin, :metric .== String(b))
        (nrow(sa) == 0 || nrow(sb) == 0) && continue
        j = innerjoin(select(sa, [:model, :score_median]),
                      select(sb, [:model, :score_median]); on = :model, makeunique = true)
        nrow(j) < 3 && continue
        x, y = j.score_median, j.score_median_1
        nd = count(((i, k),) -> (x[i] - x[k]) * (y[i] - y[k]) < 0,
                   [(i, k) for i in 1:nrow(j) for k in (i + 1):nrow(j)])
        push!(mrows, (margin = margin, metric_x = String(a), metric_y = String(b),
                      n_models = nrow(j), spearman = corspearman(x, y),
                      kendall = corkendall(x, y), n_pairs = binomial(nrow(j), 2),
                      n_discordant = nd))
    end
    grows = NamedTuple[]
    for margin in unique(joined.margin), (a, b) in pairs
        sub = @subset(joined, :margin .== margin)
        for m in unique(sub.model)
            ms = @subset(sub, :model .== m)
            x = ms[!, a]; y = ms[!, b]
            ok = .!ismissing.(x) .& .!ismissing.(y) .&
                 isfinite.(coalesce.(x, NaN)) .& isfinite.(coalesce.(y, NaN))
            sum(ok) < 10 && continue
            push!(grows, (margin = margin, model = m, metric_x = String(a), metric_y = String(b),
                          n = sum(ok),
                          spearman = corspearman(Float64.(x[ok]), Float64.(y[ok]))))
        end
    end
    md = DataFrame(mrows)
    return (; model_level = md, group_level = DataFrame(grows),
              discordant = isempty(md) ? md : sort(@subset(md, :n_discordant .> 0),
                                                   :n_discordant, rev = true))
end

####################################################################################################
# §E — is the log scale enough?
####################################################################################################

"""
    coordinate_shares(fc_store, truth, labels, origins; grid, cfg, scale) -> DataFrame

Mean share of the squared error `‖x − y‖²` carried by each coordinate, over draws and groups:
`axis ∈ {"age","horizon"}, coord, label, share, share_sd, uniform`.

This is the measurement behind "is the log scale enough, or is the 7-dim age score really a score
of the two biggest bins?". Decision rule for 15j: if every share sits within ~2× of uniform (1/7,
1/4) the log scale suffices and no per-coordinate standardisation is needed; if one coordinate
carries more than ~40%, the vector is effectively one-dimensional and a truth-derived
standardisation would be required. **Measured on this grid: log shares run 9.7–20.8% across age,
i.e. comfortably inside the rule** — so no third scale is enabled.
"""
function coordinate_shares(fc_store, truth, labels, origins; grid = cis_age_grid(), cfg,
                           scale::AbstractString = MV_SCALE)
    A, H = grid.N, length(cfg.horizons)
    rows = NamedTuple[]
    for (axis, n, lab) in (("age", A, grid.LAB), ("horizon", H, string.(1:H)))
        acc = [Float64[] for _ in 1:n]
        for o in origins, l in labels
            fc = fc_store[(o, l)]
            T = truth[o]
            D = size(fc, 3)
            if axis == "age"
                for h in 1:H
                    s = zeros(n)
                    for a in 1:A
                        yv, _ = _apply_scale(T[a, h], scale)
                        @inbounds for m in 1:D
                            v, _ = _apply_scale(fc[a, h, m], scale)
                            isfinite(v) && (s[a] += abs2(v - yv))
                        end
                    end
                    tot = sum(s)
                    tot > 0 && for a in 1:A; push!(acc[a], s[a] / tot); end
                end
            else
                for a in 1:A
                    s = zeros(n)
                    for h in 1:H
                        yv, _ = _apply_scale(T[a, h], scale)
                        @inbounds for m in 1:D
                            v, _ = _apply_scale(fc[a, h, m], scale)
                            isfinite(v) && (s[h] += abs2(v - yv))
                        end
                    end
                    tot = sum(s)
                    tot > 0 && for h in 1:H; push!(acc[h], s[h] / tot); end
                end
            end
        end
        for k in 1:n
            push!(rows, (axis = axis, coord = k, label = lab[k],
                         share = mean(acc[k]), share_sd = std(acc[k]), uniform = 1 / n))
        end
    end
    return DataFrame(rows)
end

####################################################################################################
# §F — the decoupling (permutation) test
####################################################################################################

"""
    decouple_check(fc_store, truth, labels, origins; grid, cfg, margins, origin_stride,
                   n_shuffle, scale) -> (; results, corrs)

Permute **each coordinate's draws independently**, which preserves every marginal exactly and
destroys all dependence between coordinates, then re-score. `results` carries `es`/`es_shuf`,
`vs_p05`/`vs_p05_shuf` and the relative changes; `corrs` carries the mean and max |Pearson
correlation| between coordinates of the unshuffled draw vector.

**What this is for.** `margins` defaults to `("age",)` alone, because `"age"` is the only margin
whose vector the model actually produces jointly. Across AGE there *is* a genuine shared-parameter
joint — one `post_index[d] → Cstar_end[m]`, one `susc`/`inf`/`gamma_sar`/GI per draw, and
`pred = N * acc` couples all seven bins — so a non-zero answer here is possible, and what it
measures is how much of that joint survives the iid-across-age observation layer that dominates it
(mean |corr| across age ≈ 0.07–0.10).

⚠ Across the HORIZON axis the model has *no cross-horizon joint predictive at all*: each horizon is
a separate `fit_or_load_stage2` with its own pooled draws and its own `MersenneTwister(cfg.seed + h)`
(`joint_model.jl:1872-1875`), and the intervening-horizon lag plug is the MEDIAN over finite draws
(`1912-1915`) — a point summary, so no draw-level path is shared. `two_stage_forecast`'s own
docstring says it (`1851-1853`). **That is why the `"time"` and `"age-time"` margins were dropped
from `MARGIN_ORDER` on 2026-10-16**, and running this check on them was the evidence: ES moved
≤0.06% and VS ≤0.8% when the horizon pairing was destroyed, i.e. there was no coupling there to
destroy. Pass `margins = ("time",)` to reproduce that measurement; do not read it as a finding
about the data.

⚠ **No tolerance is asserted on the effect size.** A hard threshold on a measured empirical
quantity turns a finding into a flaky test. The function `@warn`s above 5% — ~6× the largest effect
measured on this grid — and the warning says the notebook's narrative is now wrong, which is what
is actually at risk.

⚠ **Seeded per group, not globally.** A single global RNG would make the result depend on iteration
order, and this function is threaded; "reproducible only when single-threaded" is not reproducible.
"""
function decouple_check(fc_store, truth, labels, origins; grid = cis_age_grid(), cfg,
                        margins = ("age",), origin_stride::Integer = 7,
                        n_shuffle::Integer = 3, scale::AbstractString = MV_SCALE,
                        warn_rel::Real = 0.05)
    A, H = grid.N, length(cfg.horizons)
    ors = collect(origins)[1:origin_stride:end]
    groupspec = Dict(m => _margin_groups(m, A, H) for m in margins)
    dmax = maximum(length(g.idx) for m in margins for g in groupspec[m])

    jobs = [(o, l) for o in ors for l in labels]
    nchunk = max(1, min(Threads.nthreads(), length(jobs)))
    chunks = [jobs[i:nchunk:end] for i in 1:nchunk]
    rparts = Vector{Vector{NamedTuple}}(undef, nchunk)
    cparts = Vector{Vector{NamedTuple}}(undef, nchunk)

    @sync for (ci, ch) in enumerate(chunks)
        Threads.@spawn begin
            D = size(fc_store[first(jobs)], 3)
            buf = Matrix{Float64}(undef, dmax, D)
            sbf = Matrix{Float64}(undef, dmax, D)
            rows = NamedTuple[]; crows = NamedTuple[]
            for (o, l) in ch
                fc = fc_store[(o, l)]
                T = truth[o]
                cols = 1:size(fc, 3)
                for margin in margins, g in groupspec[margin]
                    idx = g.idx; d = length(idx)
                    M, _ = _mv_fill!(buf, fc, idx, scale, cols)
                    X = view(buf, 1:d, 1:M)
                    y = [first(_apply_scale(T[a, h], scale)) for (a, h) in idx]
                    es0 = energy_score(X, y); vs0 = variogram_score(X, y; p = 0.5)
                    C = cor(transpose(Matrix(X)))
                    od = [abs(C[i, j]) for i in 1:d for j in 1:d if i != j]
                    push!(crows, (model = l, origin = o, margin = margin, d = d,
                                  mean_abs_cor = isempty(od) ? NaN : mean(od),
                                  max_abs_cor = isempty(od) ? NaN : maximum(od)))
                    for r in 1:n_shuffle
                        rng = MersenneTwister(hash((cfg.seed, "15j-decouple", l, o, margin, r)))
                        copyto!(view(sbf, 1:d, 1:M), X)
                        for k in 1:d
                            v = view(sbf, k, 1:M)
                            shuffle!(rng, v)
                        end
                        Xs = view(sbf, 1:d, 1:M)
                        # The entire validity claim, checkable to 0.0: a permutation of each row
                        # leaves that row's multiset — hence every marginal — bit-identical.
                        for k in 1:d
                            @assert sort(collect(view(Xs, k, 1:M))) ==
                                    sort(collect(view(X, k, 1:M))) "15j decouple: the shuffle " *
                                    "changed a marginal (model $(l), origin $(o), $(margin), row $(k))"
                        end
                        push!(rows, (model = l, origin = o, margin = margin, scale = scale,
                                     shuffle = r, d = d, M = M, es = es0, vs_p05 = vs0,
                                     es_shuf = energy_score(Xs, y),
                                     vs_p05_shuf = variogram_score(Xs, y; p = 0.5)))
                    end
                end
            end
            rparts[ci] = rows; cparts[ci] = crows
        end
    end

    res = DataFrame(reduce(vcat, rparts))
    res = @transform(res, :rel_es = :es_shuf ./ :es .- 1, :rel_vs = :vs_p05_shuf ./ :vs_p05 .- 1)
    sort!(res, [:margin, :model, :origin, :shuffle])
    corrs = sort(DataFrame(reduce(vcat, cparts)), [:margin, :model, :origin])

    for g in groupby(res, [:model, :margin])
        for (col, nm) in ((:rel_es, "ES"), (:rel_vs, "VS(p=0.5)"))
            v = filter(isfinite, collect(skipmissing(g[!, col])))
            isempty(v) && continue
            if abs(median(v)) > warn_rel
                @warn """
                    15j decouple: destroying the draw pairing moved $(nm) by a median
                    $(round(100 * median(v); digits = 2))% for $(first(g.model)) over
                    $(first(g.margin)) — above the $(round(100 * warn_rel; digits = 1))% notice
                    level. The scores are still correct; what is now wrong is the notebook's
                    NARRATIVE, which says these rules see almost no dependence on this grid.
                    Rewrite §8 rather than adjusting the threshold."""
            end
        end
    end
    return (; results = res, corrs)
end

####################################################################################################
# §G — the guarded R oracle
####################################################################################################

"Is an R package available? Used to make the cross-checks optional."
function _has_r_pkg(name::AbstractString)
    try
        return rcopy(R"isTRUE(requireNamespace($name, quietly = TRUE))")::Bool
    catch
        return false
    end
end

"""
    es_cross_check_r(fc_store, truth, labels, origins; grid, cfg, n_per_margin, scale, seed)
        -> Union{Nothing,DataFrame}

Cross-check the native `energy_score`/`variogram_score` against `scoringRules::es_sample` and
`vs_sample` on real groups. Returns `nothing` (with an `@info`) when `scoringRules` is absent.

⚠ `scoringRules` v1.1.3 is installed in this image but is **NOT declared in
`.devcontainer/Dockerfile`** (which installs only socialmixr, jsonlite, here, checkpoint, cmocean,
tableone, broom, MGLM) and is not in CLAUDE.md's R list. So it must not be a hard dependency: the
native implementation is the engine and R is only the oracle.

⚠ **This is the only thing that pins the variogram normalisation** — the full ordered double sum
versus the `i<j` half is a factor of 2, and nothing else in the repo would notice. Measured
2026-10-15 at d = 4/7/28 and M = 1000…10 000: ES and VS(p = 0.5, 1.0) agree with `scoringRules` to
≤3.2e-14 relative (1.9e-13 on the 2026-10-16 run's 16 groups).

⚠ The margins it samples are a LITERAL covering all four, **not `MARGIN_ORDER`**, so `"time"` and
`"age-time"` groups appear in this table even though they are no longer reported. That is on
purpose: the primitive must be correct at every `d`, and restricting the oracle to the reported
margins would drop the d = 4 and d = 28 coverage for no gain.

⚠ The problem with the R route was never data volume, it was FORMAT: `scoringutils`' multivariate
sample class wants one row per (unit × dimension × sample_id) — ~1.06e8 rows for this grid — while
a 28×10 000 Float64 matrix is 2.2 MB and crosses RCall instantly. Do not re-derive the wrong
conclusion from the row count.
"""
function es_cross_check_r(fc_store, truth, labels, origins; grid = cis_age_grid(), cfg,
                          n_per_margin::Integer = 4, scale::AbstractString = MV_SCALE,
                          seed::Integer = 15)
    Threads.threadid() == 1 || error("15j: RCall is not thread-safe — call the cross-check from " *
                                     "the main task only")
    if !_has_r_pkg("scoringRules")
        @info "15j: skipping the R cross-check — `scoringRules` is not installed. It is present " *
              "in the dev image but is NOT declared in .devcontainer/Dockerfile, so it is " *
              "deliberately not a hard dependency. The native scores are the engine either way; " *
              "the verified agreement (<=3.2e-14 at d=4/7/28) is recorded in this function's " *
              "docstring and in 15j §5."
        return nothing
    end
    R"suppressMessages(library(scoringRules))"
    A, H = grid.N, length(cfg.horizons)
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    # ⚠ A LITERAL, deliberately NOT `MARGIN_ORDER`. This is an oracle for the PRIMITIVE, which has
    # to be right at every `d`, so it samples d = 1 / 4 / 7 / 28 even though only d = 1 and d = 7
    # are reported. Narrowing it to the reported margins would quietly halve the dimensional
    # coverage of the ONE check that pins the variogram normalisation (see the docstring).
    for margin in ("none", "time", "age", "age-time")
        gs = _margin_groups(margin, A, H)
        for _ in 1:n_per_margin
            o = rand(rng, collect(origins)); l = rand(rng, collect(labels)); g = rand(rng, gs)
            fc = fc_store[(o, l)]; T = truth[o]; idx = g.idx; d = length(idx)
            buf = Matrix{Float64}(undef, d, size(fc, 3))
            M, _ = _mv_fill!(buf, fc, idx, scale, 1:size(fc, 3))
            X = Matrix(view(buf, 1:d, 1:M))
            y = [first(_apply_scale(T[a, h], scale)) for (a, h) in idx]
            @rput X; @rput y
            esr = rcopy(R"es_sample(y, X)")
            esj = energy_score(X, y)
            vsr = d == 1 ? NaN : rcopy(R"vs_sample(y, X, p = 0.5)")
            vsj = variogram_score(X, y; p = 0.5)
            relf(a, b) = (isnan(a) && isnan(b)) ? 0.0 : abs(a - b) / max(abs(b), eps())
            push!(rows, (margin = margin, model = l, origin = o, d = d, M = M,
                         es_julia = esj, es_R = esr, es_rel = relf(esj, esr),
                         vs_julia = vsj, vs_R = vsr, vs_rel = relf(vsj, vsr)))
        end
    end
    out = DataFrame(rows)
    worst = maximum(vcat(out.es_rel, filter(isfinite, out.vs_rel)))
    worst < 1e-9 || error("15j: native scores disagree with scoringRules by $(worst) relative — " *
                          "the most likely cause is the variogram normalisation (full ordered " *
                          "double sum vs the i<j half, a factor of 2)")
    println("R cross-check: ", nrow(out), " real groups, worst relative disagreement ", worst)
    return out
end

"""
    verify_es_reduces_to_crps(fc_store, truth, labels, origins; grid, cfg, n, scale, seed)
        -> (; frame, max_rel)

Check, on `n` randomly chosen REAL cells, that `energy_score` on a 1-coordinate vector returns
exactly what `crps_sample` returns — forcing the general O(M²) kernel rather than the fast path.

**This is the check that licenses `mv_scores(…; fast_univariate = true)`**, so it must not be run
through the shortcut it justifies: that would be circular. It is also the single most informative
assert in 15j, because one identity exercises the `1/M²` convention, the `d × M` orientation, the
scale transform and the truth lookup at once.

Tolerance is RELATIVE, and the two measured values are worth knowing because they say something
about the data rather than about the code:

- **log scale: ≤3.3e-13** over 400 real cells (`rtol = 1e-10` passes comfortably).
- **natural scale: ~2.3e-9** — and it CANNOT be made better. The pair term sums 5e7 square roots;
  where a cell contains a 3.8e109 draw beside ~1e5 ones, Float64 cannot represent the small
  contributions at all, so two algebraically identical algorithms (this O(M²) kernel and
  `crps_sample`'s sorted O(M log M) rearrangement) necessarily lose different amounts. That is one
  more consequence of the extreme finite draws, not a defect — call this on the log scale, which is
  the scale everything is reported on, and pass a looser `rtol` if you want the natural-scale
  figure as evidence.

⚠ Both numbers moved by 2.5e7× when `energy_score` gained compensated summation (`_kbn`): naive
accumulation gave 8.3e-6 on the LOG scale. A cross-check against `scoringRules::es_sample` does not
catch that, because it sums the same O(M²) way and is wrong in the same direction.
"""
function verify_es_reduces_to_crps(fc_store, truth, labels, origins; grid = cis_age_grid(), cfg,
                                   n::Integer = 300, scale::AbstractString = MV_SCALE,
                                   seed::Integer = 1236, rtol::Real = 1e-10)
    A, H = grid.N, length(cfg.horizons)
    rng = MersenneTwister(seed)
    rows = NamedTuple[]
    for _ in 1:n
        o = rand(rng, collect(origins)); l = rand(rng, collect(labels))
        a = rand(rng, 1:A); h = rand(rng, 1:H)
        fc = fc_store[(o, l)]
        buf = Matrix{Float64}(undef, 1, size(fc, 3))
        M, _ = _mv_fill!(buf, fc, [(a, h)], scale, 1:size(fc, 3))
        X = view(buf, 1:1, 1:M)
        y = first(_apply_scale(truth[o][a, h], scale))
        es = energy_score(X, [y])                 # the GENERAL kernel, deliberately
        cr = crps_sample(view(buf, 1, 1:M), y)
        push!(rows, (model = l, origin = o, age = grid.LAB[a], horizon = h, M = M,
                     es = es, crps = cr, rel = abs(es - cr) / max(abs(cr), eps())))
    end
    out = DataFrame(rows)
    mx = maximum(out.rel)
    mx < rtol || error("15j: `energy_score` at d=1 does not reproduce `crps_sample` " *
                       "(max rel $(mx) over $(n) real cells). The 1/M² pair-term convention is " *
                       "the usual cause — the unbiased 1/(M(M−1)) form breaks this identity, and " *
                       "with it `mv_scores`' fast_univariate path.")
    println("ES == crps_sample at d=1 on ", n, " real cells (general O(M²) kernel): max rel = ", mx)
    return (; frame = out, max_rel = mx)
end

"""
    scoringutils_semantics_check(fc_store, truth, labels, origins; grid, cfg, scale)
        -> Union{Nothing,NamedTuple}

Confirm that `scoringutils`' `joint_across` means what this whole notebook assumes — "which margins
form the multivariate vector" — by scoring ONE `margin="age"` group (d = 7, so 70 000 long-format
rows: trivial) through `as_forecast_multivariate_sample(..., joint_across = "age_group")` →
`score()` and comparing against the native value. Returns `nothing` if the 2.2.0 multivariate API
is absent.

⚠ This check was on `margin="time"` / `joint_across = "horizon"` until 2026-10-16 and MOVED rather
than being deleted when the horizon-spanning margins were dropped from `MARGIN_ORDER`. It is the
only thing that pins `joint_across`'s meaning, and that meaning is the assumption the whole
notebook rests on — so it has to live on a margin the notebook still reports.
"""
function scoringutils_semantics_check(fc_store, truth, labels, origins; grid = cis_age_grid(),
                                      cfg, scale::AbstractString = MV_SCALE)
    Threads.threadid() == 1 || error("15j: RCall is not thread-safe — main task only")
    _has_r_pkg("scoringutils") || (@info "15j: scoringutils absent, skipping"; return nothing)
    has_mv = try
        rcopy(R"""exists("as_forecast_multivariate_sample", where = asNamespace("scoringutils"))""")::Bool
    catch
        false
    end
    if !has_mv
        @info "15j: this scoringutils has no multivariate sample class (a 2.2.0 feature) — " *
              "skipping the semantics check; the scoringRules cross-check still covers the maths."
        return nothing
    end
    o = first(origins); l = first(labels)
    fc = fc_store[(o, l)]; T = truth[o]
    A = grid.N
    idx = [(a, 1) for a in 1:A]          # the AGE vector at h = 1 — `margin = "age"`, d = A
    buf = Matrix{Float64}(undef, A, size(fc, 3))
    M, _ = _mv_fill!(buf, fc, idx, scale, 1:size(fc, 3))
    X = Matrix(view(buf, 1:A, 1:M))
    y = [first(_apply_scale(T[a, h], scale)) for (a, h) in idx]
    ags = [grid.LAB[a] for (a, _) in idx]
    df = DataFrame(age_group = repeat(ags, outer = M), sample_id = repeat(1:M, inner = A),
                   predicted = vec(X), observed = repeat(y, outer = M))
    @rput df
    R"""
    suppressMessages(library(scoringutils))
    f  <- as_forecast_multivariate_sample(df, joint_across = "age_group",
             forecast_unit = "age_group", observed = "observed",
             predicted = "predicted", sample_id = "sample_id")
    sc <- as.data.frame(score(f))
    """
    sc = rcopy(R"sc")
    esr = first(skipmissing(sc.energy_score)); vsr = first(skipmissing(sc.variogram_score))
    esj = energy_score(X, y); vsj = variogram_score(X, y; p = 0.5)
    rel(a, b) = abs(a - b) / max(abs(b), eps())
    println("scoringutils `joint_across = \"age_group\"` semantics check (", l, " @ ", o, ", d=", A,
            ", M=", M, "):")
    println("   energy_score    native ", esj, "  scoringutils ", esr, "  rel ", rel(esj, esr))
    println("   variogram_score native ", vsj, "  scoringutils ", vsr, "  rel ", rel(vsj, vsr))
    rel(esj, esr) < 1e-9 || error("15j: `joint_across` does not mean what 15j assumes — the " *
                                  "energy score disagrees by $(rel(esj, esr)) relative")
    return (; es_native = esj, es_r = esr, vs_native = vsj, vs_r = vsr)
end

"""
    log_scale_matches_wis(qall; n = 20_000, rtol = 1e-12) -> NamedTuple

Check that `log(max(x, 0) + 1)` is the same transform `scoringutils` applies to WIS, i.e.
`transform_forecasts(fun = log_shift, offset = 1)` — so that "ES and VS are scored on a log scale
like WIS" is a verified statement rather than an intention. If `log_shift`'s **offset** or **base**
convention ever changed, ES/VS and WIS would drift onto different scales while both were still
called "log", and nothing downstream would notice.

⚠ **The claim is agreement to a few ulp, NOT bit-identity, and the distinction was learned the hard
way.** `scoringutils::log_shift` evaluates `log(x + offset, base = exp(1))`, which is the same
*mathematical* transform, but R and Julia call different libm implementations of `log`: measured on
20 000 real `qall.predicted` values, **3 disagreed, by 1 ulp (max |Δ| = 1.8e-15 on values of
magnitude ~12, i.e. ~1.5e-16 relative)**. An earlier spot-check on six hand-picked values returned
`identical() == TRUE` and was simply too small a sample to see it.
  That is why the tolerance is relative and loose by floating-point standards but still four orders
of magnitude tighter than any convention error could hide in: a wrong `offset` (0 instead of 1) or a
wrong `base` changes the result by O(1), not by an ulp. The check discriminates exactly what it is
meant to.
"""
function log_scale_matches_wis(qall::DataFrame; n::Integer = 20_000, seed::Integer = 7,
                               rtol::Real = 1e-12)
    Threads.threadid() == 1 || error("15j: RCall is not thread-safe — main task only")
    rng = MersenneTwister(seed)
    v = qall.predicted
    idx = length(v) <= n ? collect(eachindex(v)) : rand(rng, eachindex(v), n)
    x = max.(float.(v[idx]), 0.0)
    @rput x
    R"""suppressMessages(library(scoringutils)); r <- log_shift(x, offset = 1)"""
    r = rcopy(R"r")
    j = log.(x .+ 1)
    adiff = abs.(j .- r)
    rdiff = adiff ./ max.(abs.(r), eps())
    n_ne  = count(!, isequal.(j, r))
    ulps  = [a == 0 ? 0.0 : a / eps(max(abs(b), floatmin())) for (a, b) in zip(adiff, r)]
    maxrel = maximum(rdiff)
    maxrel < rtol || error("""
        15j: `log(x+1)` does not reproduce scoringutils' `log_shift(x, offset = 1)`:
        max RELATIVE difference $(maxrel) over $(length(x)) values exceeds rtol $(rtol)
        (max |Δ| = $(maximum(adiff)), max ulp distance = $(maximum(ulps))).
        An ulp-scale disagreement is expected — R and Julia use different libm `log`
        implementations. A difference this large is not: suspect the OFFSET or the BASE, either of
        which shifts the result by O(1). ES/VS and WIS would then be on two different scales while
        both were still called "log".""")
    @printf("log scale == WIS's log_shift(offset=1): %d of %d values differ, max |Δ| %.2e, max %.1f ulp, max rel %.2e\n", n_ne, length(x), maximum(adiff), maximum(ulps), maxrel)
    return (; n = length(x), n_differing = n_ne, max_abs_diff = maximum(adiff),
              max_ulp = maximum(ulps), max_rel_diff = maxrel)
end

####################################################################################################
# §H — figures and the report
####################################################################################################

"Model → colour, keyed on the cache's label order so a model carries the same colour here as in
`res/8j_*` and `res/9j_*`."
_model_col(labels, m) = MODEL_COLS[findfirst(==(m), labels)]

"""
    plot_rel_score_by_horizon(joined, labels, cfg; ref, metrics) -> Plot

The **ES-over-age** view: one panel per metric (matched Σ-WIS, energy score, variogram p=0.5),
x = horizon, y = the median across origins of the per-group ratio to `ref`. One line per model.

Three panels on shared axes *is* the comparison — same models, same colours, same x, three rules.
"""
function plot_rel_score_by_horizon(joined::DataFrame, labels, cfg; ref::AbstractString = REF_MODEL,
                                   metrics = (:wis_sum, :es, :vs_p05))
    sub = @subset(joined, :margin .== "age")
    hz = sort(unique(skipmissing(sub.horizon)))
    panels = Plots.Plot[]
    for (pi, met) in enumerate(metrics)
        p = plot(; xlabel = "horizon (weeks)", ylabel = "relative $(mv_metric_label(met))",
                 title = "$(('A' + pi - 1)). $(mv_metric_label(met))", titlefontsize = 9,
                 legend = (pi == 1 ? :topleft : false), legendfontsize = 5, xticks = hz)
        for m in labels
            ys = Float64[]
            for h in hz
                a = @subset(sub, :model .== m, :horizon .== h)
                b = @subset(sub, :model .== ref, :horizon .== h)
                j = innerjoin(select(a, [:origin, met]), select(b, [:origin, met]);
                              on = :origin, makeunique = true)
                r = filter(isfinite, j[!, met] ./ j[!, Symbol(string(met, "_1"))])
                push!(ys, isempty(r) ? NaN : median(r))
            end
            plot!(p, hz, ys; color = _model_col(labels, m), lw = 2, marker = :circle, ms = 3,
                  markerstrokewidth = 0, label = m)
        end
        hline!(p, [1.0]; color = :gray, ls = :dash, label = "")   # after the series
        push!(panels, p)
    end
    return plot(panels...; layout = panel_grid(length(panels)), size = (420 * length(panels), 430),
                plot_title = "15j — ES OVER AGE (7-bin vector): relative skill by horizon " *
                             "(log scale, median ratio vs $(ref))", plot_titlefontsize = 10)
end

"""
    plot_score_rank_bump(summ, labels; stat) -> Plot

The direct answer to "does the energy score rank the models the way WIS does?" — one panel per
margin, x = the three rules at THAT margin (matched Σ-WIS, energy score, variogram p=0.5), y = rank
(1 = best).

⚠ **The margin is held fixed within a panel, deliberately.** An earlier version put all four
margins on one axis, and its crossings read as "the rules disagree" when they were really showing
the *margin* changing — the x positions varied the rule and the aggregation at the same time, so
neither effect could be read off it. Comparing rules is only meaningful at a fixed aggregation
(which is the same reason `scoring.jl` gained `by_model_dt_h_age`), and the across-margin story is
then readable as the difference BETWEEN panels.

Rank is the only honest common scale across rules whose units differ, which is why this is a bump
chart and not a dual-axis comparison of raw scores. The univariate `none` panel carries no
variogram score (VS is undefined at `d = 1`) and shows two rules.
"""
function plot_score_rank_bump(summ::DataFrame, labels; stat::Symbol = :score_median)
    rankcol = stat === :score_median ? :rank_median : :rank_mean
    rules = ((:wis_sum, "Σ WIS"), (:es, "ES"), (:vs_p05, "VS\np=0.5"))
    margins = [m for m in MARGIN_ORDER if nrow(@subset(summ, :margin .== m)) > 0]
    panels = Plots.Plot[]
    for (pi, mg) in enumerate(margins)
        sm = @subset(summ, :margin .== mg)
        keep = [(mt, lb) for (mt, lb) in rules if nrow(@subset(sm, :metric .== String(mt))) > 0]
        p = plot(; title = "margin = $(mg)  (d = $(mg == "none" ? 1 : mg == "time" ? 4 :
                                                  mg == "age" ? 7 : 28))",
                 titlefontsize = 8, ylabel = (pi == 1 ? "rank (1 = best)" : ""), yflip = true,
                 xticks = (1:length(keep), [lb for (_, lb) in keep]),
                 yticks = 1:length(labels), xtickfontsize = 6,
                 legend = (pi == 1 ? :outerbottom : false), legendfontsize = 5,
                 xlims = (0.7, length(keep) + 0.3))
        for m in labels
            ys = Float64[]
            for (mt, _) in keep
                r = @subset(sm, :metric .== String(mt), :model .== m)
                push!(ys, nrow(r) == 0 ? NaN : float(first(r[!, rankcol])))
            end
            plot!(p, 1:length(keep), ys; color = _model_col(labels, m), lw = 2.2,
                  marker = :circle, ms = 5, markerstrokewidth = 0, label = m)
        end
        push!(panels, p)
    end
    return plot(panels...; layout = panel_grid(length(panels)),
                size = (430 * min(length(panels), 2), 400 * cld(length(panels), 2)),
                plot_title = "15j — within each margin, do the rules rank the six models the " *
                             "same way? (log scale, by $(stat))", plot_titlefontsize = 9)
end

"""
    plot_score_rank_bump_by_horizon(summ_h, labels; margin, stats, rules) -> Plot

The per-HORIZON bump chart: columns are lead times 1..4, rows are the two summary statistics
(`:score_median` on top, `:score_mean` below), x = the five rules at `margin = "age"`
(Σ WIS, Σ CRPS, ES, VS p=0.5, VS p=1), y = rank 1..6 with 1 at the top. Takes
`summarise_mvscores_by_horizon`'s frame.

**Both statistics are in ONE figure because their disagreement is the result.** Measured on the
2026-10-15 grid, 63 origins per (model, horizon):

| ranked by | Σ WIS | Σ CRPS | ES | VS p=0.5 | VS p=1 |
|---|---|---|---|---|---|
| mean — same ordering at h=1,2,3,4? | YES | YES | YES | no | no |
| median — same ordering across h?   | no  | no  | no  | no | no |

Two things to read off it, both of which belong in any caption:

1. **The three proper, marginal-sensitive rules agree perfectly across lead time on the mean** —
   `weighted-hweibull|neighbourhood < weighted-hweibull|mean < no-contact|null <
   unweighted-negbin|mean < unweighted-negbin|neighbourhood < unweighted-negbin|mean-diagonal` at
   every horizon. The variogram score does not. ⚠ At h=4 by mean, `vs_p1` ranks **`no-contact|null`
   FIRST** — a model with a constant `C*` and no contact data at all wins. That is not a finding
   about the null model, it is **VS's impropriety for the marginals demonstrated**: VS is
   shift-invariant and sees only pairwise differences, so a uniformly biased forecast is not
   penalised. Read VS as a dependence diagnostic, never as a skill ranking.
2. **The mean ranking cannot move with the aggregation and the median ranking can.** The mean over
   groups of a sum over collapsed cells is the total divided by the group count, and the total is
   aggregation-invariant; a median of sums is not a sum of medians. Summing cells averages away
   each model's per-cell right skew, at different rates per model, so near-tied models swap.

`crps_sum` appears here but deliberately NOT in `plot_score_rank_bump`'s rules: at `margin="none"`
`es ≡ crps_sum ≡ crps_mean` exactly, so it would draw a duplicate line on the only panel where it
would be new.
"""
function plot_score_rank_bump_by_horizon(summ_h::DataFrame, labels; margin::AbstractString = "age",
                                         stats = (:score_median, :score_mean),
                                         rules = ((:wis_sum, "Σ WIS"), (:crps_sum, "Σ CRPS"),
                                                  (:es, "ES"), (:vs_p05, "VS\np=0.5"),
                                                  (:vs_p1, "VS\np=1")))
    sm0 = @subset(summ_h, :margin .== margin)
    nrow(sm0) == 0 && error("15j: `summ_h` has no rows at margin `$(margin)`")
    hz = sort(unique(sm0.horizon))
    panels = Plots.Plot[]
    for (si, stat) in enumerate(stats)
        rankcol = stat === :score_median ? :rank_median : :rank_mean
        for (hi, h) in enumerate(hz)
            sh = @subset(sm0, :horizon .== h)
            keep = [(mt, lb) for (mt, lb) in rules if nrow(@subset(sh, :metric .== String(mt))) > 0]
            first_panel = (si == 1 && hi == 1)
            p = plot(; title = "h = $(h) (wk ahead) — by $(stat === :score_median ? "median" : "mean")",
                     titlefontsize = 8, ylabel = (hi == 1 ? "rank (1 = best)" : ""), yflip = true,
                     xticks = (1:length(keep), [lb for (_, lb) in keep]),
                     yticks = 1:length(labels), xtickfontsize = 6, ytickfontsize = 6,
                     legend = (first_panel ? :outerbottom : false), legendfontsize = 5,
                     xlims = (0.7, length(keep) + 0.3), ylims = (0.5, length(labels) + 0.5))
            for m in labels
                ys = Float64[]
                for (mt, _) in keep
                    r = @subset(sh, :metric .== String(mt), :model .== m)
                    push!(ys, nrow(r) == 0 ? NaN : float(first(r[!, rankcol])))
                end
                plot!(p, 1:length(keep), ys; color = _model_col(labels, m), lw = 2.0,
                      marker = :circle, ms = 4.5, markerstrokewidth = 0, label = m)
            end
            push!(panels, p)
        end
    end
    return plot(panels...; layout = (length(stats), length(hz)),
                size = (330 * length(hz), 430 * length(stats)),
                plot_title = "15j — does the ranking hold across LEAD TIME? (margin = $(margin), " *
                             "log scale; top row median, bottom row mean)",
                plot_titlefontsize = 9)
end


"""
    plot_score_vs_wis(joined, labels; margin, metric) -> Plot

Per-group agreement between the matched Σ-WIS and the energy score, **one panel per model** —
six colours in a single all-pairs scatter is past the point of legibility, so the fix is faceting.
Spearman ρ is annotated in each panel.
"""
function plot_score_vs_wis(joined::DataFrame, labels; margin::AbstractString = "age",
                           metric::Symbol = :es)
    sub = @subset(joined, :margin .== margin)
    panels = Plots.Plot[]
    for m in labels
        ms = @subset(sub, :model .== m)
        x = Float64[]; y = Float64[]
        for r in eachrow(ms)
            (ismissing(r.wis_sum) || ismissing(r[metric])) && continue
            (r.wis_sum > 0 && isfinite(r.wis_sum) && r[metric] > 0 && isfinite(r[metric])) || continue
            push!(x, r.wis_sum); push!(y, r[metric])
        end
        rho = length(x) > 9 ? corspearman(x, y) : NaN
        p = plot(; title = string(m, "\nSpearman ρ = ", isnan(rho) ? "n/a" :
                                  string(round(rho; digits = 3))),
                 titlefontsize = 7, xlabel = "Σ WIS", ylabel = mv_metric_label(metric),
                 legend = false, xscale = :log10, yscale = :log10)
        scatter!(p, x, y; color = _model_col(labels, m), ms = 1.8, markerstrokewidth = 0, alpha = 0.5)
        push!(panels, p)
    end
    return plot(panels...; layout = panel_grid(length(panels)), size = (1180, 700),
                plot_title = "15j — per-group agreement: Σ WIS vs the energy score " *
                             "(margin = $(margin), log scale, $(nrow(sub) ÷ max(length(labels),1)) " *
                             "groups per model)", plot_titlefontsize = 10)
end

"""
    plot_es_vs_sumcrps(joined, labels; margins) -> Plot

Energy score against the sum of the per-coordinate CRPS of the same cells, one panel per
multivariate margin, with Spearman ρ and the fitted log-log slope. Since 2026-10-16 `margins`
defaults to the single reported multivariate margin `("age",)`, so this is normally ONE panel —
that is the margin reduction, not a facet that failed to render.

⚠ **This is a rank/monotonicity check, NOT an equality check.** The energy score is a *norm* and
Σ CRPS is a *sum*: for `d` coordinates of comparable scale ES grows like `√d·scale` while Σ CRPS
grows like `d·scale`, so the ratio is `O(1/√d)` and *should* be well below 1 for a correct
implementation. Equality holds only at `d = 1`, which is what 15j §5's exact-reduction assert
tests. Do not put a tolerance on the ratio.
"""
function plot_es_vs_sumcrps(joined::DataFrame, labels; margins = ("age",))
    panels = Plots.Plot[]
    for mg in margins
        sub = @subset(joined, :margin .== mg)
        x = Float64[]; y = Float64[]
        p = plot(; title = "margin = $(mg)  (d = $(isempty(sub) ? 0 : first(sub.d)))",
                 titlefontsize = 8, xlabel = "Σ CRPS over the same cells", ylabel = "energy score",
                 legend = (mg == first(margins) ? :topleft : false), legendfontsize = 5,
                 xscale = :log10, yscale = :log10)
        for m in labels
            ms = @subset(sub, :model .== m)
            xa = Float64[]; ya = Float64[]
            for r in eachrow(ms)
                (r.crps_sum > 0 && isfinite(r.crps_sum) && r.es > 0 && isfinite(r.es)) || continue
                push!(xa, r.crps_sum); push!(ya, r.es)
            end
            append!(x, xa); append!(y, ya)
            scatter!(p, xa, ya; color = _model_col(labels, m), ms = 1.8, markerstrokewidth = 0,
                     alpha = 0.5, label = m)
        end
        if length(x) > 9
            rho = corspearman(x, y)
            lx = log10.(x); ly = log10.(y)
            b = cov(lx, ly) / var(lx)
            plot!(p; title = "margin = $(mg)  (d = $(first(sub.d)))\nρ = $(round(rho; digits=3)), " *
                             "log-log slope = $(round(b; digits=3)), " *
                             "median ES/ΣCRPS = $(round(median(y ./ x); digits = 3))")
        end
        push!(panels, p)
    end
    return plot(panels...; layout = panel_grid(length(panels)),
                size = (max(430 * length(panels), 620), 450),
                plot_title = "15j — how much of the energy score is just the marginals? " *
                             "(a RANK check: ES is a norm and Sum-CRPS a sum, so the ratio is O(1/sqrt(d)))",
                plot_titlefontsize = 9)
end

"""
    plot_coordinate_shares(shares) -> Plot

Mean share of the squared error carried by each coordinate, with the uniform reference. Single hue
— these are magnitudes of one quantity, not categories.
"""
function plot_coordinate_shares(shares::DataFrame)
    panels = Plots.Plot[]
    for (ax, xl) in (("age", "CIS age bin"), ("horizon", "horizon (weeks)"))
        s = sort(@subset(shares, :axis .== ax), :coord)
        nrow(s) == 0 && continue
        p = plot(; title = "share of squared error by $(ax)", titlefontsize = 9, xlabel = xl,
                 ylabel = "mean share of |x-y|^2", legend = :topright, legendfontsize = 6,
                 xticks = (1:nrow(s), s.label), xrotation = (ax == "age" ? 45 : 0),
                 ylims = (0, max(maximum(s.share) * 1.25, 1.6 / nrow(s))))
        bar!(p, 1:nrow(s), s.share; color = :steelblue, alpha = 0.85, linecolor = :steelblue,
             label = "")
        hline!(p, [first(s.uniform)]; color = :firebrick, ls = :dash,
               label = "uniform = 1/$(nrow(s))")
        hline!(p, [2 * first(s.uniform)]; color = :grey50, ls = :dot, label = "2× uniform")
        push!(panels, p)
    end
    return plot(panels...; layout = panel_grid(length(panels)), size = (520 * length(panels), 420),
                plot_title = "15j — is the log scale enough? (no coordinate should dominate the squared error)",
                plot_titlefontsize = 10)
end

"""
    plot_decoupling(dec, labels) -> Plot

How much of each score comes from the dependence between coordinates: the relative change in ES and
in VS(p=0.5) when every coordinate's draws are permuted independently (marginals exactly
preserved). Point = median over origins × shuffles, whisker = the 5–95% range.

A band hugging 0 means the rule is reading the marginals and almost nothing else — which for the
horizon margin is the *expected* answer, because the model produces no cross-horizon joint at all.
"""
function plot_decoupling(dec, labels)
    res = dec.results
    # ONE x-range for every panel. The whole point of this figure is that all three margins give
    # the same answer — ≈0 — and three independently auto-scaled axes make each panel look like it
    # has structure, at three different zoom levels, which is the opposite of what the data says.
    allv = filter(isfinite, vcat(collect(skipmissing(res.rel_es)), collect(skipmissing(res.rel_vs))))
    lim = isempty(allv) ? 1.0 : 100 * maximum(abs, quantile(allv, [0.02, 0.98]))
    lim = max(lim, 0.5)
    panels = Plots.Plot[]
    for mg in unique(res.margin)
        sub = @subset(res, :margin .== mg)
        p = plot(; title = "margin = $(mg)", titlefontsize = 9,
                 xlabel = "relative change (%)", ylabel = "",
                 yticks = (1:length(labels), labels), ytickfontsize = 6,
                 xlims = (-lim, lim),
                 legend = (mg == first(unique(res.margin)) ? :bottomright : false),
                 legendfontsize = 5)
        for (oi, (col, nm, c)) in enumerate(((:rel_es, "energy score", :steelblue),
                                             (:rel_vs, "variogram p=0.5", :darkorange)))
            for (mi, m) in enumerate(labels)
                v = filter(isfinite, collect(skipmissing(@subset(sub, :model .== m)[!, col])))
                isempty(v) && continue
                yy = mi + (oi - 1.5) * 0.22
                lo, hi = quantile(v, 0.05), quantile(v, 0.95)
                # Clamp to the shared axis rather than letting a whisker run off it unremarked;
                # an arrowless line that stops at the frame would read as a real endpoint.
                plot!(p, [clamp(100lo, -lim, lim), clamp(100hi, -lim, lim)], [yy, yy];
                      color = c, lw = 1.4, label = "")
                scatter!(p, [100 * median(v)], [yy]; color = c, ms = 4, markerstrokewidth = 0,
                         label = (mi == 1 ? nm : ""))
            end
        end
        vline!(p, [0.0]; color = :gray, ls = :dash, label = "")
        push!(panels, p)
    end
    return plot(panels...; layout = panel_grid(length(panels)), size = (430 * length(panels), 420),
                plot_title = "15j — how much dependence do these rules actually see? " *
                             "(log scale; ≈0 means the score is reading the marginals)",
                plot_titlefontsize = 10)
end

"""
    score_diff_frame(joined; metric, ref, models, margin) -> DataFrame

The per-origin gap `score - score_ref` and its running total in date order, for one `metric`.
Columns `model, origin, horizon, metric, diff, cum_diff`. This is the data behind
`plot_score_diff_over_time`, exposed separately so the figures' content reaches a CSV — a PDF
cannot be read numerically, and these series exist nowhere else.

Origins where `ref` has no score are skipped and contribute nothing to the running sum, matching
`plot_wis_diff_over_time`'s behaviour exactly.
"""
function score_diff_frame(joined::DataFrame; metric::Symbol, ref::AbstractString,
                          models, margin::AbstractString = "age")
    sub = @subset(joined, :margin .== margin)
    rows = NamedTuple[]
    for h in sort(unique(skipmissing(sub.horizon)))
        sh = @subset(sub, :horizon .== h)
        refmap = Dict(r.origin => r[metric] for r in eachrow(@subset(sh, :model .== ref)))
        for m in models
            m == ref && continue                       # the reference is identically 0
            acc = 0.0
            for r in eachrow(sort(@subset(sh, :model .== m), :origin))
                rv = get(refmap, r.origin, missing)
                (ismissing(rv) || ismissing(r[metric]) ||
                 !isfinite(rv) || !isfinite(r[metric])) && continue
                d = r[metric] - rv
                acc += d
                push!(rows, (model = m, origin = r.origin, horizon = h, metric = String(metric),
                             diff = d, cum_diff = acc))
            end
        end
    end
    return DataFrame(rows)
end

"""
    plot_score_diff_over_time(joined, labels, cfg; metric, ref, models, margin) -> Plot

**Cumulative** score difference over time, faceted by horizon — 15j's twin of 9j's
`plot_wis_diff_over_time` (`res/9j_wis_diff_over_time.png`), with the same reference and the same
drawn set plus `no-contact|null`, so the two figures read directly against each other. Within each
horizon the per-origin gap `score - score_ref` is accumulated in date order, so each line's
**endpoint is the whole-period difference** vs `ref`: below 0 beat it cumulatively, a steadily
rising line loses a little every week. The `ref` line is omitted (identically 0).

Called once per metric. `metric` is any column of the joined frame — `:wis_sum`, `:es`, `:vs_p05`,
`:vs_p1`, `:crps_sum`.

⚠ **Σ WIS here is a SUM over the 7 age bins; 9j's `by_model_dt_h.wis` is a MEAN over them.** A
constant factor of 7, so the shapes are identical and the two figures compare exactly — but the
y-axis magnitudes do not, and the axis label says so.

⚠ **For the two variogram panels this is a weaker statement than for WIS or ES.** VS is
shift-invariant and not proper for the marginals: a uniformly biased forecast is not penalised at
all. A rising VS line means the model's pairwise DEPENDENCE structure is drifting from the truth's,
not that it forecasts worse. `plot_score_rank_bump_by_horizon`'s docstring carries the concrete
demonstration (`no-contact|null` ranks first under `vs_p1` at h=4).

⚠ Date-axis ordering, as in 9j: all series first, THEN `hline!`, and `shade_periods!` last of all —
a leading numeric overlay collapses the Date axis, and `shade_periods!` reads `Plots.ylims`.

⚠ Two departures from 9j, both legibility fixes for flaws visible in `res/9j_wis_diff_over_time.png`:
`markerstrokewidth = 0` (at `ms = 2` the default black stroke swallows the fill and every series
reads as the same dark line), and **the band LABELS are suppressed on the legend panel**. The
latter is the only fix that works: `shade_periods!` must run last, and it writes the band names as
annotations, so they land on top of the legend whatever `background_color_legend` says. The
shading itself is unchanged on every panel.
"""
function plot_score_diff_over_time(joined::DataFrame, labels, cfg; metric::Symbol,
                                   ref::AbstractString = "weighted-hweibull|neighbourhood",
                                   models::Vector{<:AbstractString} = ["unweighted-negbin|mean",
                                                                       "weighted-hweibull|mean",
                                                                       "no-contact|null"],
                                   margin::AbstractString = "age")
    ref in labels || error("15j: reference `$(ref)` is not among the models")
    sub = @subset(joined, :margin .== margin)
    nrow(sub) == 0 && error("15j: no rows at margin `$(margin)`")
    ylab = metric === :wis_sum ? "cumulative Σ WIS diff (summed over the 7 age bins)" :
                                 "cumulative $(mv_metric_label(metric)) diff"
    panels = Plots.Plot[]
    for (k, h) in enumerate(cfg.horizons)
        sh = @subset(sub, :horizon .== h)
        refmap = Dict(r.origin => r[metric] for r in eachrow(@subset(sh, :model .== ref)))
        p = plot(; title = "horizon $h (wk ahead)", titlefontsize = 8, xlabel = "forecast origin",
                 ylabel = ylab, ylabelfontsize = 7, xlabelfontsize = 8,
                 legend = (k == 1 ? :topleft : false), legendfontsize = 6,
                 background_color_legend = :white, foreground_color_legend = nothing,
                 xrotation = 45, bottom_margin = 10Plots.mm, left_margin = 4Plots.mm)
        for m in labels
            (m == ref || m ∉ models) && continue       # ref ≡ 0; keep only the requested models
            dts = Date[]; cum = Float64[]; acc = 0.0
            for r in eachrow(sort(@subset(sh, :model .== m), :origin))   # running sum, date order
                rv = get(refmap, r.origin, missing)
                (ismissing(rv) || ismissing(r[metric]) ||
                 !isfinite(rv) || !isfinite(r[metric])) && continue
                acc += r[metric] - rv
                push!(dts, r.origin); push!(cum, acc)
            end
            isempty(dts) && continue
            # ⚠ `markerstrokewidth = 0`: at ms = 2 the default black stroke swallows the fill and
            # every series reads as the same dark line — the flaw in res/9j_wis_diff_over_time.png.
            plot!(p, dts, cum; color = _model_col(labels, m), lw = 1.5, marker = :circle, ms = 2.2,
                  markerstrokewidth = 0, label = m)
        end
        hline!(p, [0.0]; color = :gray, ls = :dash, label = "")   # ref = 0, AFTER the dates are set
        pdates = collect(keys(refmap))                            # origins the ref actually covers
        # ⚠ `labels = false` on the LEGEND panel. `shade_periods!` writes its rotated band names as
        # ANNOTATIONS, and it must run last (it reads `Plots.ylims`), so they are drawn OVER the
        # legend however the legend is styled — `background_color_legend` cannot save it. That is
        # the collision in res/9j_wis_diff_over_time.png. The band SHADING still appears on every
        # panel and the four panels share an x-axis, so dropping the text on one costs nothing.
        isempty(pdates) || shade_periods!(p, minimum(pdates), maximum(pdates);
                                          labels = (k != 1))      # last of all
        push!(panels, p)
    end
    nr, nc = panel_grid(length(cfg.horizons))
    # ⚠ Second line, not a longer first one: a single-line plot_title this wide is CLIPPED at both
    # ends by Plots rather than shrunk, so the caveat would silently vanish from the figure.
    sub2 = (metric === :vs_p05 || metric === :vs_p1) ?
           "⚠ VS is shift-invariant and NOT proper for the marginals — read this as dependence " *
           "drift, not skill" :
           "margin = $(margin) (the 7-bin age vector), log scale"
    return plot(panels...; layout = (nr, nc), size = (1150, 820),
                plot_title = "15j — cumulative $(mv_metric_label(metric)) difference over time, " *
                             "by horizon;  < 0 beats $(ref)\n$(sub2)",
                plot_titlefontsize = 9)
end


"""
    report_mvscores(mv, joined, summ, conc, shares, dec; summ_h, diffs, res_dir) -> DataFrame

Print the headline tables and write the CSVs (the count is printed, so it cannot go stale again).
Returns the summary frame.

Every table is printed with **median and mean side by side** — see `summarise_mvscores` for why
their disagreement is itself a result rather than a nuisance, and
`plot_score_rank_bump_by_horizon` for the sharpest instance of it on this grid.

`summ_h` (`summarise_mvscores_by_horizon`) and `diffs` (a stacked `score_diff_frame`) are optional
and each adds one CSV. They are the content of the two figure families added on 2026-10-16, and a
PDF cannot be read numerically, so a figure whose numbers reach no CSV is a dead end.
"""
function report_mvscores(mv, joined::DataFrame, summ::DataFrame, conc, shares::DataFrame, dec;
                         summ_h::Union{Nothing,DataFrame} = nothing,
                         diffs::Union{Nothing,DataFrame} = nothing,
                         res_dir::AbstractString = "../res")
    mkpath(res_dir)
    println("\n===== log-scale scores by model, per margin (rank 1 = best) =====")
    for mg in MARGIN_ORDER
        s = @subset(summ, :margin .== mg)
        nrow(s) == 0 && continue
        println("\n--- margin = ", mg, " (vector spans: ",
                mg == "none" ? "nothing — univariate baseline, ES ≡ crps_sample" :
                mg == "age"  ? "the 7 CIS age bins, at one horizon" :
                mg == "time" ? "the 4 horizons (NOT REPORTED — see MARGIN_ORDER)" :
                               "all 28 age × horizon cells (NOT REPORTED — see MARGIN_ORDER)",
                ") ---")
        show(select(sort(s, [:metric, :rank_median]),
                    [:metric_label, :model, :n_groups, :score_median, :score_mean,
                     :rel_median, :rel_mean, :rank_median, :rank_mean]);
             allrows = true, allcols = true, truncate = 36)
        println()
    end

    println("\n===== draw audit (the guard class `score_wis` cannot see) =====")
    show(mv.audit; allrows = true, allcols = true, truncate = 36); println()

    println("\n===== rank agreement between the rules (6 model medians; 15 pairs) =====")
    show(conc.model_level; allrows = true, allcols = true, truncate = 30); println()

    println("\n===== coordinate shares (is the log scale enough?) =====")
    show(shares; allrows = true, allcols = true); println()

    println("\n===== decoupling: median relative change when the pairing is destroyed =====")
    dsum = combine(groupby(dec.results, [:margin, :model]),
                   :rel_es => (x -> median(filter(isfinite, collect(skipmissing(x))))) => :med_rel_es,
                   :rel_vs => (x -> median(filter(isfinite, collect(skipmissing(x))))) => :med_rel_vs)
    dcor = combine(groupby(dec.corrs, [:margin, :model]),
                   :mean_abs_cor => mean => :mean_abs_cor)
    show(leftjoin(dsum, dcor; on = [:margin, :model]); allrows = true, allcols = true,
         truncate = 36); println()

    if summ_h !== nothing
        println("\n===== the SAME table sliced by horizon (margin = age) =====")
        for h in sort(unique(summ_h.horizon))
            println("\n--- horizon ", h, " (wk ahead) ---")
            show(select(sort(@subset(summ_h, :horizon .== h), [:metric, :rank_median]),
                        [:metric_label, :model, :n_groups, :score_median, :score_mean,
                         :rank_median, :rank_mean]);
                 allrows = true, allcols = true, truncate = 36)
            println()
        end
        # The headline contrast: which rules give ONE ordering across all four lead times?
        println("\nis the model ordering identical at every horizon?")
        for met in unique(summ_h.metric)
            sm = @subset(summ_h, :metric .== met)
            om = Set(tuple(sort(@subset(sm, :horizon .== h), :rank_median).model...)
                     for h in sort(unique(sm.horizon)))
            od = Set(tuple(sort(@subset(sm, :horizon .== h), :rank_mean).model...)
                     for h in sort(unique(sm.horizon)))
            println("   ", rpad(met, 10), " by median: ", length(om) == 1 ? "YES" : "NO",
                    "   by mean: ", length(od) == 1 ? "YES" : "NO")
        end
    end

    nwritten = 7
    CSV.write(joinpath(res_dir, "15j_mvscores.csv"), mv.scores)
    CSV.write(joinpath(res_dir, "15j_mvscores_summary.csv"), summ)
    CSV.write(joinpath(res_dir, "15j_rank_concordance.csv"), conc.model_level)
    CSV.write(joinpath(res_dir, "15j_es_vs_sumcrps.csv"),
              select(joined, [:model, :origin, :margin, :horizon, :age_group, :d, :M,
                              :wis_sum, :es, :vs_p05, :vs_p1, :crps_sum, :crps_mean]))
    CSV.write(joinpath(res_dir, "15j_decoupling.csv"), dec.results)
    CSV.write(joinpath(res_dir, "15j_draw_audit.csv"), mv.audit)
    CSV.write(joinpath(res_dir, "15j_coordinate_shares.csv"), shares)
    if summ_h !== nothing
        CSV.write(joinpath(res_dir, "15j_mvscores_summary_by_horizon.csv"), summ_h); nwritten += 1
    end
    if diffs !== nothing
        CSV.write(joinpath(res_dir, "15j_score_diff_over_time.csv"), diffs); nwritten += 1
    end
    println("\nwrote ", nwritten, " CSVs to ", res_dir)
    return summ
end
