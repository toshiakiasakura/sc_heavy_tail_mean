# 14j_viz_utils.jl — helpers for `14j_publication_figures.ipynb`.
#
# 14j is the PUBLICATION notebook, not a diagnostic: 8j–13j all answer "is the fit healthy?",
# and nothing in the tree answered "what do the contact data look like, and what does the fitted
# model say the mean contact rate did over time?". These things live here:
#
#   §A  `audit_stage1_grid`  — are the chains on disk compatible with the CURRENT model?
#                              (this is the replacement for the vanished `tmp/check_grid.jl`)
#   §B  `marginal_degree_data` — per-participant-day contact degree, split child/adult by the
#                              MODEL's own block rule, in three weighting variants
#   §C  `model_NegBinDegree` / `model_HurdleWeibullDegree` — the two marginal likelihoods
#   §D  `fit_marginal_models` — fit them, write chains + a summary table to `res/`
#   §E  figures — degree pdf/ccdf panels, and the fitted contact mean over time
#   §F  `_period_cols`/`_period_group` + `contact_duration_shares` — pandemic-phase colouring, and
#                              §5's raw home vs non-home share of contact DURATION
#   §G  `_perron_share` + `incidence_age_shares` — §6's expected age distribution (the dominant
#                              eigenvector of C*) and its inc2prev observed comparator
#   §H  the three round-5 figures
#
# Requires (in this order): `forecast_utils.jl`, `8j_viz_utils.jl` (`stage1_chain_path`),
# `10j_viz_utils.jl` (`reconstruct_mu_draws`, `reconstruct_p0_draws`, `_stage1_gp_generation`).
# `9j_viz_utils.jl` (`PERIODS`/`PERIOD_ORDER`/`period_of`/`shade_periods!`) is a RUNTIME-ONLY
# dependency of §E/§F/§H: every reference to it sits inside a function body, so this file still
# loads standalone. Do NOT introduce a top-level `const` built from `PERIODS` — that is exactly
# why `_period_cols` is a function and not a constant.
#
# ⚠ NOTHING HERE FITS OR WRITES A GRID ARTEFACT. Every Stage-1 reader is
# `isfile(path) || return nothing`, and no function calls `two_stage_forecast` / `fit_or_load_*`.
# That is deliberate: those paths FIT ON MISS and write into `../dt_intermediate` under the live
# token (CLAUDE.md's first Gotcha — it is how 9j grinds and how 10j cell 4 used to race a running
# grid). The only sampling 14j does is §D's 2–3 parameter marginal fits, which go to `res/`.

##########################################################################
# §A — Stage-1 grid audit
##########################################################################

const _S1_DEGREES = ("unweighted-negbin", "weighted-hweibull")

"Latent count `model_degree` produces for `deg` over a `Tn`-week window (`-w8h`/`-s0`/`-t0`):
5 shared GP scalars + 32·Tn (NegBin) or 81·Tn (hurdle-Weibull)."
_expected_npar(deg::AbstractString, Tn::Integer) =
    deg == "unweighted-negbin" ? 5 + 32Tn : 5 + 81Tn

"""
    _s1_meta(path) -> NamedTuple

The self-describing keys `fit_or_load_stage1` writes alongside `result`, read WITHOUT
deserialising the chain (JLD2 fetches only the named datasets, so this is ~1 ms against the
~3 s a full `load(path, "result")` costs). A key absent from the file comes back `:ABSENT`,
which is a reportable finding rather than an error: `phi_pf_max`/`phi_pf_override` only exist
on artefacts written after 2026-08-11, so a whole grid reading `:ABSENT` dates the generation.
"""
function _s1_meta(path::AbstractString)
    jldopen(path, "r") do f
        g(k) = haskey(f, k) ? f[k] : :ABSENT
        d = g("diag")
        (; sampler = g("sampler"), ad_backend = g("ad_backend"),
           target_accept = g("target_accept"), nuts_adapts = g("nuts_adapts"),
           nuts_draws = g("nuts_draws"), phi_init_scale = g("phi_init_scale"),
           phi_pf_max = g("phi_pf_max"), phi_pf_override = g("phi_pf_override"),
           divergences  = d === :ABSENT ? missing : d.divergences,
           min_ess      = d === :ABSENT ? missing : d.min_ess,
           max_depth    = d === :ABSENT ? missing : d.max_tree_depth,
           frac_at_max  = d === :ABSENT ? missing : d.frac_at_max_depth,
           n_draws      = d === :ABSENT ? missing : d.n_draws)
    end
end

"""
    _s1_structure(path, deg, Tn, A, contacts) -> NamedTuple

Load the chain and check its parameter block against what the CURRENT `model_degree` would
write. Expensive (a full deserialise), which is why `audit_stage1_grid` samples by default.

The three generation gates come from **`_stage1_gp_generation`** (`10j_viz_utils.jl`) rather
than being restated here — the `-diag` refusal, the `phi_time`-vs-`log_rho_time` temporal fork
and the pre-`-m32` token refusal. That helper exists precisely so a new reader cannot ship with
two of the three; read its docstring before adding a fourth check.

The shape checks that stay local are the ones needing `A`/`Tn`, exactly as in
`reconstruct_mu_draws`: `z` rows = P−1 (`-s0`), `z` cols = Tn, `z_c` = Tn−1 (`-t0`), the
dispersion block `4·Tn`, and `p0f` = A²·Tn on the weighted path only.
"""
function _s1_structure(path::AbstractString, deg::AbstractString, Tn::Integer, A::Integer,
                       contacts::AbstractString)
    chn = try
        load(path, "result")
    catch err
        return (; ok = false, note = "unreadable ($(typeof(err)))")
    end
    chn isa MCMCChains.Chains ||
        return (; ok = false, note = "`result` is $(nameof(typeof(chn))), not a Chains — the PPL " *
                                     "stack (forecast_utils.jl) must be loaded before auditing")
    pn = string.(names(chn, :parameters))
    P  = length(first(_unordered_pairs(A)))

    gen = _stage1_gp_generation(pn, contacts, path)
    gen === nothing && return (; ok = false, note = "failed a generation gate (see warning)")

    zr = [parse(Int, m.captures[1]) for n in pn
          for m in (match(r"^z\[(\d+)\s*,\s*(\d+)\]$", n),) if m !== nothing]
    zt = [parse(Int, m.captures[2]) for n in pn
          for m in (match(r"^z\[(\d+)\s*,\s*(\d+)\]$", n),) if m !== nothing]
    isempty(zr) && return (; ok = false, note = "no `z[p,t]` structure field")

    nz_c  = count(n -> occursin(r"^z_c\[\d+\]$", n), pn)
    ndisp = count(n -> startswith(n, deg == "unweighted-negbin" ? "log_k[" : "log_kappa["), pn)
    np0f  = count(n -> startswith(n, "p0f["), pn)
    want_p0f = deg == "unweighted-negbin" ? 0 : A * A * Tn

    problems = String[]
    maximum(zr) == P - 1 || push!(problems, "z rows $(maximum(zr))≠$(P-1)")
    maximum(zt) == Tn    || push!(problems, "z cols $(maximum(zt))≠$Tn")
    nz_c  == Tn - 1      || push!(problems, "z_c $nz_c≠$(Tn-1)")
    ndisp == 4Tn         || push!(problems, "dispersion $ndisp≠$(4Tn)")
    np0f  == want_p0f    || push!(problems, "p0f $np0f≠$want_p0f")
    length(pn) == _expected_npar(deg, Tn) ||
        push!(problems, "npar $(length(pn))≠$(_expected_npar(deg, Tn))")

    return (; ok = isempty(problems), note = isempty(problems) ? "ok" : join(problems, "; "),
              npar = length(pn), ndraws = size(chn, 1),
              temporal = gen.temporal_is_ar1 ? "ar1" : "matern32")
end

"""
    audit_stage1_grid(cfg; grid, contacts, save_dir, origins, structural, res_dir)
        -> (; files, coverage, meta, verdict)

Is the Stage-1 grid on disk compatible with the CURRENT model, and is it one generation?

This replaces `tmp/check_grid.jl`, which CLAUDE.md still points at but which left the tree with
the rest of `tmp/`. Three separate questions, answered separately because they fail differently:

1. **Coverage** — which (degree × origin × horizon) cells exist. A partial grid is normal here
   (the hurdle-Weibull path is still being filled in), so missing cells are REPORTED, not failed.
2. **Provenance** — every setting that changes the draws but is deliberately absent from the
   cache token, so that a partially-refitted grid is detectable. `sampler`, `ad_backend`,
   `target_accept`, `nuts_adapts`, `nuts_draws`, `phi_init_scale` and `phi_pf_max` are audited
   for UNIFORMITY; a split here means chains that are not interchangeable.
   ⚠ **`phi_pf_override` is TALLIED ONLY, never audited for uniformity** — it is a per-cell
   OUTCOME (did this fit's Pathfinder φ sit at the boundary?) and is expected to be `true` on a
   handful of weighted-path cells. A large count is a modelling signal, not a provenance failure.
3. **Structure** — the parameter block against `model_degree`. `structural` selects how much of
   this to pay for: `:sample` (default) checks one file per degree × horizon, `:all` checks
   every file (~3 s each), `:none` skips it. The sample is enough when provenance is uniform,
   because the structure is fixed by the code that wrote the file; if provenance is NOT uniform
   the report says to re-run with `:all`.

`frac_at_max_depth` is reported against `cfg.stage1_nuts_max_depth` but is **not** a failure
condition, and it is easy to misread: `_nuts_diagnostics` computes it at the chain's OBSERVED
maximum depth, so `1.0` at depth 7 under a cap of 10 means the sampler settled at a constant
tree depth — healthy — and not that it saturated.

Writes `res_dir/14j_grid_audit.csv`. `verdict` is true when every present file passes its
structural check, provenance is uniform, and no chain has a divergence.
"""
function audit_stage1_grid(cfg::FrameworkConfig;
                           grid = cis_age_grid(),
                           contacts::AbstractString = contacts_label(cfg),
                           save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                           origins::Union{Nothing,AbstractVector{Date}} = nothing,
                           craw = nothing,
                           origin_max::Date = Date(2021, 12, 31),
                           structural::Symbol = :sample,
                           res_dir::AbstractString = "../res")
    structural in (:none, :sample, :all) ||
        error("structural must be :none, :sample or :all (got $structural)")
    origins === nothing && (origins = available_forecast_origins(cfg; grid = grid, craw = craw,
                                                                 origin_max = origin_max))
    A  = grid.N
    hs = collect(cfg.horizons)

    rows = NamedTuple[]
    # One structural probe per (degree, horizon) under :sample — the first PRESENT file, so the
    # probe never lands on a hole.
    probed = Set{Tuple{String,Int}}()
    for deg in _S1_DEGREES, o in origins, h in hs
        path = stage1_chain_path("$deg|mean", o, h; contacts = contacts, save_dir = save_dir)
        if !isfile(path)
            push!(rows, (; degree = deg, origin = o, h = h, present = false,
                           sampler = missing, ad_backend = missing, target_accept = missing,
                           nuts_adapts = missing, nuts_draws = missing, phi_init_scale = missing,
                           phi_pf_max = missing, phi_pf_override = missing,
                           divergences = missing, min_ess = missing, max_depth = missing,
                           frac_at_max = missing, struct_ok = missing, struct_note = "absent"))
            continue
        end
        m  = _s1_meta(path)
        Tn = cfg.n_fit + h
        do_struct = structural === :all ||
                    (structural === :sample && !((deg, h) in probed))
        s = if do_struct
            push!(probed, (deg, h)); _s1_structure(path, deg, Tn, A, contacts)
        else
            (; ok = missing, note = "not checked")
        end
        push!(rows, (; degree = deg, origin = o, h = h, present = true,
                       m.sampler, m.ad_backend, m.target_accept, m.nuts_adapts, m.nuts_draws,
                       m.phi_init_scale, m.phi_pf_max, m.phi_pf_override,
                       m.divergences, m.min_ess, m.max_depth, m.frac_at_max,
                       struct_ok = s.ok, struct_note = s.note))
    end
    files = DataFrame(rows)

    have = @subset(files, :present)
    coverage = combine(groupby(files, [:degree, :h])) do g
        p = g[g.present, :]
        (; n_present = nrow(p), n_expected = nrow(g),
           first_origin = nrow(p) == 0 ? missing : minimum(p.origin),
           last_origin  = nrow(p) == 0 ? missing : maximum(p.origin))
    end

    prov_keys = (:sampler, :ad_backend, :target_accept, :nuts_adapts, :nuts_draws,
                 :phi_init_scale, :phi_pf_max)
    meta = Dict{Symbol,Any}(k => countmap(have[!, k]) for k in prov_keys)
    meta[:phi_pf_override] = countmap(have.phi_pf_override)          # TALLY ONLY — see docstring
    prov_uniform = all(length(meta[k]) <= 1 for k in prov_keys)

    struct_checked = @subset(have, .!ismissing.(:struct_ok))
    struct_ok = nrow(struct_checked) == 0 ? false : all(struct_checked.struct_ok)
    ndiv = nrow(have) == 0 ? 0 : sum(skipmissing(have.divergences))
    verdict = struct_ok && prov_uniform && ndiv == 0 && nrow(have) > 0

    mkpath(res_dir)
    CSV.write(joinpath(res_dir, "14j_grid_audit.csv"), files)

    println("=== 14j Stage-1 grid audit — token `", contacts, "` in ", save_dir, " ===")
    println("present ", nrow(have), " / ", nrow(files), " cells over ", length(origins),
            " origins × horizons ", hs)
    for r in eachrow(coverage)
        @printf("  %-18s h%d  %3d/%3d  %s … %s\n", r.degree, r.h, r.n_present, r.n_expected,
                string(r.first_origin), string(r.last_origin))
    end
    println("provenance (must be uniform):")
    for k in prov_keys
        println("  ", rpad(string(k), 16), meta[k], length(meta[k]) > 1 ? "   ⚠ MIXED" : "")
    end
    println("  ", rpad("phi_pf_override", 16), meta[:phi_pf_override], "   (per-cell outcome; tally only)")
    if nrow(have) > 0
        println("health: divergences=", ndiv,
                "  min_ess=", round(minimum(skipmissing(have.min_ess)); digits = 1),
                "  max tree depth=", maximum(skipmissing(have.max_depth)),
                " (cap ", cfg.stage1_nuts_max_depth, ")")
    end
    println("structure (", structural, "): ", nrow(struct_checked), " chains checked, ",
            struct_ok ? "all match the current model_degree" : "MISMATCH")
    for r in eachrow(@subset(struct_checked, .!:struct_ok))
        println("   ⚠ ", r.degree, " ", r.origin, " h", r.h, ": ", r.struct_note)
    end
    !prov_uniform && structural !== :all &&
        println("   ⚠ provenance is mixed — re-run with structural = :all before trusting this")
    println(verdict ? "VERDICT: compatible with the current model structure" :
                      "VERDICT: NOT clean — see the warnings above")
    return (; files, coverage, meta, verdict)
end

##########################################################################
# §B — marginal (non-age-pair) contact degree, split child/adult
##########################################################################

# Block naming, in ONE place: `_MD_BLOCK_SYM` builds the Dict keys (`:adult_child`) and
# `_MD_BLOCK_NAME` the human titles ("adults → children"). Indexed by `block_of`, so a change
# to `cfg.child_bins` moves which BINS are in a block without touching either name.
const _MD_BLOCK_SYM  = ("child", "adult")
const _MD_BLOCK_NAME = ("children", "adults")
# contactor → contactee, in the order the §3 pairs figure stacks its rows.
const _MD_PAIR_KEYS  = (:child_child, :child_adult, :adult_child, :adult_adult)

"""
    marginal_degree_data(cfg; grid, date_from, date_to, raw)
        -> Dict{Symbol,NamedTuple}

Per-participant-day contact degree over `[date_from, date_to]`, in three weighting variants, at two
levels of resolution:

| key | what |
|---|---|
| `:child`, `:adult` | the **marginal** degree — every contact a participant of that block reported |
| `:child_child` … `:adult_adult` | split by the **contactee's** block too, read *contactor → contactee* |

**Both splits are `block_of(bin, cfg)`**, i.e. `cfg.child_bins = 2` ⇒ child = CIS bins 1–2
(`2-10`, `11-15`, ages 2–15), adult = bins 3–7 (16+). That is the same rule `model_degree`'s
dispersion blocks use — `bl = 2·(block_of(i)−1) + block_of(j)` in `_cell_moments!` — so the four
pairs here ARE the model's four dispersion blocks, and every 14j section shares one definition.
Bins come from `assign_age_bin` (degree_agepair.jl), which resolves an ambiguous age interval by a
population-weighted draw: that matters for participants because CoMix's `12-17` group straddles the
15/16 cut, and for contactees because most reported contactee ages are intervals.

⚠ **`assign_age_bin` CONSUMES `rng`, so the call order is part of the contract.** Participants
first, contactees second, from ONE `MersenneTwister(cfg.seed)` — `prepare_degree_data`'s order.
Appending the contactee draws *after* the participant ones is why adding the pairs left the
`:child`/`:adult` marginals bit-identical to what this function returned before they existed.

⚠ **THE FOUR PAIRS DO NOT SUM TO THE MARGINAL, DELIBERATELY.** A marginal keeps every contact; a
pair needs the contactee's bin, and `assign_age_bin` returns `nothing` for an age reported entirely
below the grid's first bin (under-2s). Those contacts are dropped from the pairs and counted in the
marginal's `n_contacts_unbinned`, so

    n_contacts(:adult) == n_contacts(:adult_child) + n_contacts(:adult_adult) +
                          n_contacts_unbinned(:adult)

holds exactly — the notebook asserts it per block per window. The marginals were **not** recomputed
on binned contacts to force the pairs to sum to them: the marginal is the observed degree
distribution, and quietly discarding part of it for a tidier identity would change a published
figure for a bookkeeping convenience.

**The three weightings** mirror `prepare_degree_data`'s per-contact weight exactly:

| variant | group ("mass") contact | individually-reported contact |
|---|---|---|
| `count`      | 1                     | 1                                          |
| `w_group`    | `cfg.w_dur_group`     | `duration_weight(duration_multi, cfg.d_max)` |
| `w_nogroup`  | **0**                 | as above                                     |

⚠ `cfg.w_dur_group` is `2.5/240`, numerically IDENTICAL to what `duration_weight` assigns a
missing duration (the NA→<5 min fallback). So `w_group` is not a special convention — it is the
ordinary one — and the contrast that carries information is against `w_nogroup`.

⚠ **The two weighted series differ far less than the raw counts suggest, and that is the point.**
Over 2021-07-01…2021-12-31 group contacts are ~51% of the adult contact table, yet zeroing them
moves the mean weighted degree only 1.334 → 1.310 (−1.8%) and the zero fraction 0.1772 → 0.1842,
because each carries weight 0.0104. Where they DO matter is the tail (adult maximum 43.7 → 34.8)
and the handful of participant-days whose only contacts were group ones. This is the analysis
plan's own claim — "a group contact's contributions are significantly diminished by contact
duration weights" — made visible, so read the two curves in the CCDF tail, not at the mode.

**Zeros come from the participant-day roster**, not the contact table, exactly as
`prepare_degree_data` does: a participant-day with no contacts contributes a 0 to all three. For a
PAIR that means a participant-day with no contact *into that contactee block* — which is why the
off-diagonal pairs have much higher zero fractions than the marginals, and why their hurdle `p⁰` is
the parameter carrying most of the signal.

Each entry carries `dd` (a `DegreeDist` over integer counts, zeros included), `wg`/`wn`
(`WeightedDegreeHist` over the POSITIVE weighted degrees) and the raw per-participant-day
vectors, plus the counts §D's likelihoods need and the `caption` §E's figures title with.

⚠ `raw` is `load_raw_contact_inputs()`'s `(; df_part, craw)` — the RAW arrow contact table, not
`read_comix_uk_contact_raw()`'s joined frame, which is what this function used until the pairs
needed `:cnt_age_est_min`/`:cnt_age_est_max` (the joined frame does not select them). The two cover
identical contact rows: `(part_wave_uid, date)` is unique in `part_uk.arrow` (276 482 of 276 482),
so the joined frame's inner join neither drops nor duplicates any of the 1 325 891 contacts, and
`duration_multi` is derived by the same `_uk_duration_multi` on both paths. Taking the raw one also
lets §1–§3 and §4 share the single `load_raw_contact_inputs()` the notebook already does.
"""
function marginal_degree_data(cfg::FrameworkConfig;
                              grid = cis_age_grid(),
                              date_from::Date = Date(2021, 7, 1),
                              date_to::Date   = Date(2021, 12, 31),
                              raw = nothing)
    raw === nothing && (raw = load_raw_contact_inputs())
    inwin(d) = !ismissing(d) && (date_from <= d <= date_to)

    # ---- participants: ONE seeded stream, DRAWN FIRST (see the rng contract above) ----
    rng = MersenneTwister(cfg.seed)
    df_part = @subset(raw.df_part, inwin.(:date))
    piv = parse_age_interval.(df_part.part_age)
    df_part[!, :part_bin] = [assign_age_bin(a, b, grid, rng) for (a, b) in piv]
    df_part = @subset(df_part, .!isnothing.(:part_bin))
    df_part[!, :part_bin] = Int.(df_part.part_bin)
    df_part[!, :blk] = block_of.(df_part.part_bin, Ref(cfg))
    roster = unique(@select(df_part, :part_id_d, :date, :part_bin, :blk))

    # ---- contacts: built exactly as `prepare_degree_data` builds them, then binned SECOND ----
    dfA = @select(@subset(raw.craw, inwin.(:date)),
                  :part_id_d      = :part_wave_uid,
                  :date, :cnt_mass,
                  :duration_multi = _uk_duration_multi.(:cnt_minutes_max, :cnt_total_time),
                  :cnt_age_est_min, :cnt_age_est_max)
    dfA = innerjoin(dfA, roster, on = [:part_id_d, :date])
    civ = [interval_from_minmax(mn, mx) for (mn, mx) in zip(dfA.cnt_age_est_min, dfA.cnt_age_est_max)]
    cbin = [assign_age_bin(a, b, grid, rng) for (a, b) in civ]
    # 0 = contactee age below the grid (`assign_age_bin` → nothing). Kept as a sentinel rather than
    # filtered here, so the marginals keep every contact and the shortfall stays countable.
    dfA[!, :cnt_blk] = [c === nothing ? 0 : block_of(Int(c), cfg) for c in cbin]

    dfA[!, :grp] = _is_group_contact.(dfA.cnt_mass)
    w_ind = duration_weight.(dfA.duration_multi, cfg.d_max)
    dfA[!, :wg] = ifelse.(dfA.grp, cfg.w_dur_group, w_ind)
    dfA[!, :wn] = ifelse.(dfA.grp, 0.0,             w_ind)

    # One entry from a roster slice (which supplies the zeros) and its matching contact slice.
    function _entry(sym, ttl, cap, rsub, csub, extra)
        agg = combine(groupby(csub, [:part_id_d, :date]),
                      nrow => :k, :wg => sum => :zg, :wn => sum => :zn)
        j  = leftjoin(rsub, agg, on = [:part_id_d, :date])
        k  = Int.(coalesce.(j.k, 0))
        zg = Float64.(coalesce.(j.zg, 0.0))
        zn = Float64.(coalesce.(j.zn, 0.0))
        bins = sort(unique(rsub.part_bin))
        return merge((; group = sym, title = ttl, caption = cap, n = length(k),
                        bins = bins, bin_labels = grid.LAB[bins],
                        window = (date_from, date_to),
                        k = k, zg = zg, zn = zn,
                        dd = DegreeDist(k),
                        wg = WeightedDegreeHist(filter(>(0), zg)),
                        wn = WeightedDegreeHist(filter(>(0), zn)),
                        n_zero_k = count(==(0), k),
                        n_zero_g = count(<=(0), zg),
                        n_zero_n = count(<=(0), zn),
                        n_group_contacts = count(csub.grp),
                        n_contacts       = nrow(csub)), extra)
    end

    binlist(b) = join(grid.LAB[[i for i in 1:grid.N if block_of(i, cfg) == b]], ", ")
    out = Dict{Symbol,NamedTuple}()
    for (b, sym, ttl) in ((1, :child, "children"), (2, :adult, "adults"))
        rsub = @subset(roster, :blk .== b)
        csub = @subset(dfA,    :blk .== b)
        out[sym] = _entry(sym, ttl, "$(ttl) ($(binlist(b)))", rsub, csub,
                          (; block = b, pair = nothing,
                             n_contacts_unbinned = count(==(0), csub.cnt_blk)))
    end
    for (pb, cb) in ((1, 1), (1, 2), (2, 1), (2, 2))
        pn, cn = _MD_BLOCK_NAME[pb], _MD_BLOCK_NAME[cb]
        sym  = Symbol(_MD_BLOCK_SYM[pb], "_", _MD_BLOCK_SYM[cb])
        ttl  = "$(pn) → $(cn)"
        cap  = "$(ttl)  (participants $(binlist(pb))  ·  contactees $(binlist(cb)))"
        out[sym] = _entry(sym, ttl, cap,
                          @subset(roster, :blk .== pb),
                          @subset(dfA, (:blk .== pb) .& (:cnt_blk .== cb)),
                          (; block = pb, pair = (pb, cb), n_contacts_unbinned = 0))
    end
    return out
end

##########################################################################
# §C — the two marginal likelihoods
##########################################################################

"""
    model_NegBinDegree(dd)

Plain NegBin on the integer contact degree, zeros included — the marginal counterpart of
`model_degree`'s unweighted path and of the analysis plan's `P = NegBin(μ, k)`. NOT
zero-inflated: the framework models its zeros through the NegBin itself, and mixing in a
separate `π0` (the `model_ZeroInfNegativeBinomial` convention of 6j) would not be comparable
with the fitted grid.

`log_m`/`log_k` are soft-clamped for the same reason `model_degree`'s are: Pathfinder's LBFGS
takes aggressive early steps, and `loggamma(k + y)` at an overflowed `k` aborts the whole fit.
The mode stays interior, so gradients are unaffected.
"""
@model function model_NegBinDegree(dd::DegreeDist)
    log_m ~ Normal(0.0, 2.0)
    log_k ~ Normal(0.0, 1.0)
    m = exp(_softclamp(log_m, -8.0, 6.0))
    k = exp(_softclamp(log_k, -6.0, 6.0))
    Turing.@addlogprob! calculate_loglikelihood(dd, NegBin(m, k))
end

"""
    model_HurdleWeibullDegree(w, n_zero, n_pos)

Hurdle-Weibull on the duration-weighted degree: a zero probability `p0` plus a Weibull over the
strictly positive values, which is `HurdleWeibullAgePair`'s likelihood with the age-pair and
week structure removed.

Parameterised by the **mean of the positive part** rather than the Weibull scale, and the scale
derived as `λ = μ / Γ(1 + 1/κ)` — mirroring `_cell_moments!` (joint_model.jl) so the fitted
`μ`, `κ` and `p0` mean the same things they mean in the grid, and so the incl-zero mean
`(1−p0)·μ` is directly comparable with the `K1` that `_weibull_moments` returns.

The `-4.3` lower clamp on `log_kappa` is the framework's, and it is load-bearing: below it
`Γ(1 + 2/κ)` overflows to `Inf`, `CV² = Inf/Inf` goes `NaN`, and the fit dies.
"""
@model function model_HurdleWeibullDegree(w::WeightedDegreeHist, n_zero::Int, n_pos::Int)
    p0        ~ Beta(1.0, 1.0)            # same prior as `p0f` in model_degree
    log_mu    ~ Normal(0.0, 2.0)
    log_kappa ~ Normal(0.0, 0.5)          # same prior as the block-linear `log_kappa`
    κ = exp(_softclamp(log_kappa, -4.3, 3.0))
    μ = exp(_softclamp(log_mu, -8.0, 6.0))
    λ = μ / gamma(1 + 1 / κ)
    Turing.@addlogprob! n_zero * log(p0) + n_pos * log1p(-p0) +
                        calculate_loglikelihood(w, Weibull(κ, λ))
end

##########################################################################
# §D — fit and store to res/
##########################################################################

_q3(v) = (median(v), quantile(v, 0.05), quantile(v, 0.95))

"""
    fit_marginal_models(md, cfg; res_dir="../res", n_sample=2000)
        -> (; negbin, hweibull, summary)

Fit both §C models to one age group's data and write the results to **`res/`**:

    res/14j_fit_negbin_<group>_<from>_<to>.jld2      keys: chain, n_obs, n_zero, window, group
    res/14j_fit_hweibull_<group>_<from>_<to>.jld2    (+ n_pos)

`fit_model_with_forward_mode` (turing_models.jl) is the repo's standard entry point for this
strand — `NUTS()` from a Pathfinder-mean init under `Random.seed!(1236)`. These are 2–3
parameter models, so the cost is seconds.

The returned `summary` has one row per (group, family, quantity) with median and 5/95%, and
includes the derived quantities the figures and the text need: the NegBin mean and CV², and for
the hurdle-Weibull both the positive-part mean `μ` and the **incl-zero mean `(1−p0)·μ`**, which
is the quantity comparable with §E's timeline.
"""
function fit_marginal_models(md::NamedTuple, cfg::FrameworkConfig;
                             res_dir::AbstractString = "../res", n_sample::Integer = 2000)
    mkpath(res_dir)
    d0, d1 = md.window
    tag = "$(md.group)_$(d0)_$(d1)"

    n_pos_g = whist_nobs(md.wg)
    println("--- 14j fitting $(md.title): n=$(md.n) participant-days, ",
            "zero counts $(md.n_zero_k), zero weighted $(md.n_zero_g) ---")

    chn_nb = fit_model_with_forward_mode(model_NegBinDegree(md.dd), n_sample; progress = false)
    chn_hw = fit_model_with_forward_mode(
        model_HurdleWeibullDegree(md.wg, md.n_zero_g, n_pos_g), n_sample; progress = false)

    jldsave(joinpath(res_dir, "14j_fit_negbin_$(tag).jld2");
            chain = chn_nb, group = String(md.group), window = md.window,
            n_obs = md.n, n_zero = md.n_zero_k, model = "NegBin(m,k) on integer degree")
    jldsave(joinpath(res_dir, "14j_fit_hweibull_$(tag).jld2");
            chain = chn_hw, group = String(md.group), window = md.window,
            n_obs = md.n, n_zero = md.n_zero_g, n_pos = n_pos_g,
            model = "hurdle-Weibull(p0, mu, kappa) on duration-weighted degree (with group contacts)")

    m  = exp.(_softclamp.(vec(Array(chn_nb[:log_m])), -8.0, 6.0))
    kk = exp.(_softclamp.(vec(Array(chn_nb[:log_k])), -6.0, 6.0))
    p0 = vec(Array(chn_hw[:p0]))
    μ  = exp.(_softclamp.(vec(Array(chn_hw[:log_mu])), -8.0, 6.0))
    κ  = exp.(_softclamp.(vec(Array(chn_hw[:log_kappa])), -4.3, 3.0))

    emp_cv2(v) = mean(v) > 0 ? var(v; corrected = false) / mean(v)^2 : NaN
    rows = NamedTuple[]
    add!(fam, q, v) = (t = _q3(v); push!(rows,
        (; group = String(md.group), family = fam, quantity = q,
           median = t[1], q05 = t[2], q95 = t[3])))
    add!("negbin", "m (mean)", m)
    add!("negbin", "k (dispersion)", kk)
    add!("negbin", "CV2", 1 ./ m .+ 1 ./ kk)
    add!("hweibull", "p0", p0)
    add!("hweibull", "mu (positive-part mean)", μ)
    add!("hweibull", "kappa (shape)", κ)
    add!("hweibull", "incl-zero mean (1-p0)*mu", (1 .- p0) .* μ)

    emp = [("count",     Float64.(md.k),  md.n_zero_k),
           ("w_group",   md.zg,           md.n_zero_g),
           ("w_nogroup", md.zn,           md.n_zero_n)]
    for (nm, v, nz) in emp
        for (q, val) in (("empirical mean", mean(v)), ("empirical CV2", emp_cv2(v)),
                         ("empirical zero fraction", nz / md.n))
            push!(rows, (; group = String(md.group), family = nm, quantity = q,
                           median = val, q05 = NaN, q95 = NaN))
        end
    end
    push!(rows, (; group = String(md.group), family = "data", quantity = "n participant-days",
                   median = float(md.n), q05 = NaN, q95 = NaN))

    summary = DataFrame(rows)
    return (; negbin = chn_nb, hweibull = chn_hw, summary)
end

##########################################################################
# §E — figures
##########################################################################

# The repo carries TWO incompatible log-axis conventions: the discrete plotters
# (`plot_pdf!`/`plot_ccdf!` on a DegreeDist, and the fitted-distribution overlays in
# distributions/plot.jl) put `log10(y)` on a LINEAR axis with faked ticks, while the continuous
# ones (`plot_ccdf_continuous!`, `plot_pdf_hist!`) put y on a `yscale = :log10` axis. Overlaying
# the two families on one panel puts a series at y = −2 next to one at y = 0.01 and is silently
# wrong. 14j needs discrete counts, continuous weighted degrees and both fitted curves on the
# SAME panel, so everything below uses the house `log10(y)` convention.

const _W_COLS = (count = :black, w_group = :darkorange, w_nogroup = :purple)

"y-axis ticks for a log10-transformed value plotted on a linear axis (the house convention)."
_log10_ticks(lo::Int, hi::Int = 0) =
    (collect(lo:hi), [L"10^{%$e}" for e in lo:hi])

"""
    _surv_gt(vals, n_total) -> (x, S)

Empirical survival `S(v) = P(X > v)` at each distinct positive value, normalised over
**`n_total` participant-days including the zeros**. Unconditional on purpose: the fitted
overlays are unconditional too (`ccdf(NegBin, k)`; `(1−p0)·ccdf(Weibull, z)`), so the two sit on
one axis with no zero-truncation gymnastics, and the height of the leftmost point IS the
non-zero fraction.
"""
function _surv_gt(vals::AbstractVector{<:Real}, n_total::Integer)
    xs = sort!(Float64.(filter(>(0), vals)))
    isempty(xs) && return (Float64[], Float64[])
    u = unique(xs); n = length(xs)
    S = [(n - searchsortedlast(xs, v)) / n_total for v in u]
    keep = S .> 0
    return (u[keep], S[keep])
end

"""
    _logbin_density(vals, n_total; nbins) -> (centre, density)

Density of the positive values on LOG-spaced bins, normalised over `n_total` (so it integrates
to the non-zero fraction). Log bins rather than `plot_pdf_hist!`'s fixed linear width because
these are heavy-tailed and the panel's x axis is logarithmic — linear bins would put almost
every observation in the first one.
"""
function _logbin_density(vals::AbstractVector{<:Real}, n_total::Integer; nbins::Integer = 22)
    xs = Float64.(filter(>(0), vals))
    isempty(xs) && return (Float64[], Float64[])
    lo, hi = minimum(xs), maximum(xs)
    lo == hi && return ([lo], [length(xs) / n_total])
    edges = 10 .^ range(log10(lo) - 1e-9, log10(hi) + 1e-9; length = nbins + 1)
    ctr  = sqrt.(edges[1:end-1] .* edges[2:end])
    dens = [count(v -> edges[i] <= v < edges[i+1], xs) / n_total / (edges[i+1] - edges[i])
            for i in eachindex(ctr)]
    keep = dens .> 0
    return (ctr[keep], dens[keep])
end

"Pointwise posterior median and 90% band of the columns of `M` (draws × grid)."
function _band(M::AbstractMatrix)
    ngrid = size(M, 2)
    ([median(view(M, :, c)) for c in 1:ngrid],
     [quantile(view(M, :, c), 0.05) for c in 1:ngrid],
     [quantile(view(M, :, c), 0.95) for c in 1:ngrid])
end

"""
    _curve_pair_band(f, ndraw, ngrid; nsub) -> (band_a, band_b)

Posterior median + 90% band for the TWO curves `f(i)` returns for draw `i`, evaluating `f` once
per draw. The pdf and the ccdf of one fitted distribution come from the same evaluation, and for
the NegBin that evaluation is thousands of `loggamma` calls — so computing them separately would
double the cost of the most expensive thing in this file.
"""
function _curve_pair_band(f, ndraw::Integer, ngrid::Integer; nsub::Integer = 200)
    idx = round.(Int, range(1, ndraw; length = min(nsub, ndraw)))
    A = Array{Float64}(undef, length(idx), ngrid)
    B = Array{Float64}(undef, length(idx), ngrid)
    for (r, i) in enumerate(idx)
        a, b = f(i); A[r, :] = a; B[r, :] = b
    end
    return _band(A), _band(B)
end

"Strictly-positive finite entries of `v` — the ones that survive a log10."
_finite_pos(v) = Float64[x for x in v if isfinite(x) && x > 0]

"""
    _log_yrange(vs...; max_decades = 7) -> (lo_exp, hi_exp)

Integer log10 limits covering every series in `vs`, capped at `max_decades` BELOW the maximum.

The cap is the whole point. A fitted NegBin evaluated out to the observed maximum degree (~4000
for CoMix adults, whose mass-contact reports run into the thousands) returns densities around
1e-40; letting that set the axis compresses all four real series into the top 5% of the panel
and the figure reads as four flat lines. The empirical series cannot go below 1/n by
construction, so the visible range is set by the data and the fitted curves are clipped into it.
"""
function _log_yrange(vs...; max_decades::Real = 7)
    all = reduce(vcat, [_finite_pos(v) for v in vs]; init = Float64[])
    isempty(all) && return (-1, 0)
    hi = log10(maximum(all))
    lo = max(log10(minimum(all)), hi - max_decades)
    return (Int(floor(lo)), Int(ceil(hi)))
end

"Apply the log10 range from `_log_yrange` to a panel, with matching decade ticks."
function _apply_log_yaxis!(pl, (ylo, yhi))
    plot!(pl; ylims = (ylo - 0.25, yhi + 0.25), yticks = _log10_ticks(ylo, yhi))
end

"""
Median line + 90% ribbon in log10 space. Points below `10^ylo` are dropped rather than clamped:
clamping would draw a false floor along the bottom of the panel where the fitted curve has
actually left the plotted range.
"""
function _plot_log_band!(pl, x, med, lo, hi; color, label, ls = :solid, ylo::Real = -Inf)
    thr = isfinite(ylo) ? 10.0^ylo : 0.0
    keep = (med .> thr) .& (lo .> 0) .& isfinite.(med)
    any(keep) || return pl
    xk, mk = x[keep], med[keep]
    lk = max.(lo[keep], thr); hk = hi[keep]
    lm = log10.(mk)
    plot!(pl, xk, lm; ribbon = (lm .- log10.(lk), log10.(hk) .- lm),
          color = color, lw = 2, ls = ls, fillalpha = 0.15, label = label)
end

"""
    degree_panels(md, fit, cfg; kmax_mult=3) -> (p_pdf, p_ccdf)

The two panels for one age group: log–log pdf and log–log ccdf, each carrying the three
empirical series (unweighted count, duration-weighted with group contacts, duration-weighted
with group contacts zeroed) and the two fitted curves (NegBin on the counts, hurdle-Weibull on
the weighted-with-group series), with 90% posterior bands.

Everything is normalised over ALL participant-days, zeros included — see `_surv_gt`.
"""
function degree_panels(md::NamedTuple, fit::NamedTuple; kmax_cap::Integer = 5000)
    n = md.n
    # Span the data and no further — CoMix's mass-contact reports push the observed maximum into
    # the thousands, and every extra grid point costs `loggamma` on every posterior draw.
    kmax = clamp(maximum(md.k), 200, kmax_cap)

    m  = exp.(_softclamp.(vec(Array(fit.negbin[:log_m])), -8.0, 6.0))
    kk = exp.(_softclamp.(vec(Array(fit.negbin[:log_k])), -6.0, 6.0))
    p0 = vec(Array(fit.hweibull[:p0]))
    μ  = exp.(_softclamp.(vec(Array(fit.hweibull[:log_mu])), -8.0, 6.0))
    κ  = exp.(_softclamp.(vec(Array(fit.hweibull[:log_kappa])), -4.3, 3.0))

    kgrid = 1:kmax
    zpos  = filter(>(0), md.zg)
    zgrid = 10 .^ range(log10(minimum(zpos)), log10(maximum(zpos)); length = 120)

    # NegBin: build the pmf once per draw and cumulate it, rather than calling the memoised
    # recursive `ccdf(::PoissonMixture, k)`, which walks from k all the way up to its
    # `k_max = 20_000` sentinel on every fresh distribution. `_curve_pair_band` evaluates each
    # draw ONCE and splits the two curves out of it.
    nb_curves(i) = (d = NegBin(m[i], kk[i]);
                    pmf = [pdf(d, k) for k in 0:kmax];
                    (pmf[2:end], max.(0.0, 1 .- cumsum(pmf)[1:end-1])))   # pdf on 1:kmax, P(X>k)
    (nb_pdf_med, nb_pdf_lo, nb_pdf_hi), (nb_s_med, nb_s_lo, nb_s_hi) =
        _curve_pair_band(nb_curves, length(m), kmax)

    hw_curves(i) = (d = Weibull(κ[i], μ[i] / gamma(1 + 1 / κ[i]));
                    ((1 - p0[i]) .* pdf.(d, zgrid), (1 - p0[i]) .* ccdf.(d, zgrid)))
    (hw_pdf_med, hw_pdf_lo, hw_pdf_hi), (hw_s_med, hw_s_lo, hw_s_hi) =
        _curve_pair_band(hw_curves, length(p0), length(zgrid))

    # ---- pdf panel ----
    # NOTE ON UNITS: the count series is a pmf on unit-width integer bins, which IS a density,
    # so the three empirical series are dimensionally comparable. They are NOT on comparable
    # LEVELS, though: the weighted degrees live on a support ~400× narrower, so their density is
    # correspondingly higher. Compare each series against its own fitted curve, and compare the
    # three with each other by SHAPE. The ccdf panel is the one to read levels off.
    pk    = md.dd.x .> 0
    k_pmf = md.dd.y[pk] ./ n
    xg, dg = _logbin_density(md.zg, n)
    xn, dn = _logbin_density(md.zn, n)
    ylo_pdf, yhi_pdf = _log_yrange(k_pmf, dg, dn)          # EMPIRICAL series only set the range

    p_pdf = plot(; xaxis = :log10, xlabel = "degree", ylabel = "density (pmf on unit bins)",
                   legend = :bottomleft, legendfontsize = 6,
                   title = "pdf", titlefontsize = 9)
    scatter!(p_pdf, md.dd.x[pk], log10.(k_pmf);
             color = _W_COLS.count, ms = 3, msw = 0, label = "observed count")
    isempty(xg) || scatter!(p_pdf, xg, log10.(dg); color = _W_COLS.w_group, ms = 3, msw = 0,
                            marker = :diamond, label = "duration-weighted (with group)")
    isempty(xn) || scatter!(p_pdf, xn, log10.(dn); color = _W_COLS.w_nogroup, ms = 3, msw = 0,
                            marker = :utriangle, label = "duration-weighted (group w=0)")
    _plot_log_band!(p_pdf, collect(kgrid), nb_pdf_med, nb_pdf_lo, nb_pdf_hi;
                    color = _W_COLS.count, label = "NegBin fit", ylo = ylo_pdf)
    _plot_log_band!(p_pdf, zgrid, hw_pdf_med, hw_pdf_lo, hw_pdf_hi;
                    color = _W_COLS.w_group, label = "hurdle-Weibull fit", ylo = ylo_pdf)
    _apply_log_yaxis!(p_pdf, (ylo_pdf, yhi_pdf))

    # ---- ccdf panel ----
    surv = [_surv_gt(v, n) for v in (Float64.(md.k), md.zg, md.zn)]
    ylo_c, yhi_c = _log_yrange((s[2] for s in surv)...)

    p_ccdf = plot(; xaxis = :log10, xlabel = "degree", ylabel = "P(degree > x)",
                    legend = :bottomleft, legendfontsize = 6,
                    title = "ccdf", titlefontsize = 9)
    for ((x, S), col, lb) in zip(surv,
                                 (_W_COLS.count, _W_COLS.w_group, _W_COLS.w_nogroup),
                                 ("observed count", "duration-weighted (with group)",
                                  "duration-weighted (group w=0)"))
        isempty(x) || plot!(p_ccdf, x, log10.(S); color = col, lw = 1.4, alpha = 0.85, label = lb)
    end
    _plot_log_band!(p_ccdf, collect(kgrid), nb_s_med, nb_s_lo, nb_s_hi;
                    color = _W_COLS.count, label = "NegBin fit", ls = :dash, ylo = ylo_c)
    _plot_log_band!(p_ccdf, zgrid, hw_s_med, hw_s_lo, hw_s_hi;
                    color = _W_COLS.w_group, label = "hurdle-Weibull fit", ls = :dash, ylo = ylo_c)
    _apply_log_yaxis!(p_ccdf, (ylo_c, yhi_c))

    return (p_pdf, p_ccdf)
end

"""
    make_degree_fig(md, fit; res_dir="../res") -> Plots.Plot

One group's publication panel pair — a marginal (`:adult`, `:child`) or one contactor→contactee
pair (`:adult_child`, …). Saves `res_dir/14j_degree_<group>_<from>_<to>.png`.

The plot title comes from the entry's own `caption`, which spells out which participant bins (and,
for a pair, which contactee bins) the panel covers — so a figure cannot be mistaken for the marginal
of the same participant block.

⚠ The window is IN THE FILENAME. 14j fits more than one window, and without it the second window
would silently overwrite the first — the fit `.jld2` names have always carried it, so only the
figures were exposed.
"""
function make_degree_fig(md::NamedTuple, fit::NamedTuple; res_dir::AbstractString = "../res")
    p1, p2 = degree_panels(md, fit)
    d0, d1 = md.window
    fig = plot(p1, p2; layout = (1, 2), size = (1050, 430),
               left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
               plot_title = "CoMix UK $(md.caption), $(d0) – $(d1); " *
                            "n = $(md.n) participant-days",
               plot_titlefontsize = 9)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, "14j_degree_$(md.group)_$(d0)_$(d1).png"))
    return fig
end

"""
    make_degree_combined_fig(mds, fits; res_dir="../res") -> Plots.Plot

The 2×2 publication figure — adults on the top row, children on the bottom, pdf then ccdf.
Saves `res_dir/14j_degree_combined_<from>_<to>.png` (window in the name — see `make_degree_fig`).
"""
function make_degree_combined_fig(mds::Dict{Symbol,<:NamedTuple}, fits::Dict{Symbol,<:NamedTuple};
                                  res_dir::AbstractString = "../res")
    panels = Any[]
    for g in (:adult, :child)
        p1, p2 = degree_panels(mds[g], fits[g])
        plot!(p1; title = "$(mds[g].title) — pdf")
        plot!(p2; title = "$(mds[g].title) — ccdf")
        push!(panels, p1, p2)
    end
    d0, d1 = mds[:adult].window
    fig = plot(panels...; layout = (2, 2), size = (1150, 830),
               left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
               plot_title = "CoMix UK contact degree, $(d0) – $(d1) — observed vs duration-weighted " *
                            "(group contacts included / zeroed), with NegBin and hurdle-Weibull fits",
               plot_titlefontsize = 9)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, "14j_degree_combined_$(d0)_$(d1).png"))
    return fig
end

"""
    make_degree_pairs_fig(mds, fits; res_dir="../res") -> Plots.Plot

The contactor→contactee twin of `make_degree_combined_fig`: **one row per block pair** in the order
child→child, child→adult, adult→child, adult→adult, with the same pdf / ccdf column pair per row.

These four pairs are `model_degree`'s four dispersion blocks (`bl = 2·(block_of(i)−1) + block_of(j)`,
`_cell_moments!`), so this figure is the marginal figure decomposed exactly the way the fitted model
decomposes it — and the panel each row's dispersion parameter is estimated from.

⚠ The off-diagonal rows are dominated by ZEROS: a participant-day with no contact into the other
block contributes a 0, so `adult → child` has a zero fraction several times the adult marginal's.
That is real, not a data problem — the ccdf's leftmost height IS the non-zero fraction, and it is
what the hurdle `p⁰` fits.

⚠ These four do NOT sum to the two marginals — see `marginal_degree_data`'s docstring. Saves
`res_dir/14j_degree_pairs_<from>_<to>.png` (window in the name, as everywhere in 14j).
"""
function make_degree_pairs_fig(mds::Dict{Symbol,<:NamedTuple}, fits::Dict{Symbol,<:NamedTuple};
                               res_dir::AbstractString = "../res")
    panels = Any[]
    for g in _MD_PAIR_KEYS
        p1, p2 = degree_panels(mds[g], fits[g])
        plot!(p1; title = "$(mds[g].title) — pdf")
        plot!(p2; title = "$(mds[g].title) — ccdf")
        push!(panels, p1, p2)
    end
    d0, d1 = mds[first(_MD_PAIR_KEYS)].window
    # ⚠ Wider and with a bigger left margin than the 2×2 marginal figure, and a SHORTER title. At
    # (1150, 1600) Plots scaled the guide fonts to the taller canvas: the y-axis labels ran off the
    # left edge and the 2×2 figure's title — which fits on one line at 830 px tall — was clipped at
    # both ends. Neither failure raises; both are only visible in the rendered PNG.
    fig = plot(panels...; layout = (length(_MD_PAIR_KEYS), 2), size = (1250, 1650),
               left_margin = 12Plots.mm, right_margin = 4Plots.mm, bottom_margin = 5Plots.mm,
               plot_title = "CoMix UK contact degree by contactor → contactee block, $(d0) – $(d1) " *
                            "— observed vs duration-weighted, NegBin and hurdle-Weibull fits",
               plot_titlefontsize = 9)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, "14j_degree_pairs_$(d0)_$(d1).png"))
    return fig
end

"""
    _age_blocks(cfg, grid) -> (; blk, blocks, wts)

The child/adult split and the within-block population weights, in ONE place.

`blk[i]` is `block_of(i, cfg)` (1 = child, 2 = adult), `blocks` carries `(code, symbol, description)`
per block, and `wts[i] = POP_i / Σ_{i' in the same block} POP_{i'}` — so a block value is the
population-average contact rate for a person in that block, not an unweighted average over bins of
very different size.

Shared because the FITTED aggregate (`collect_origin_contact_means`) and the OBSERVED aggregate
(`observed_contact_means`) are overlaid on the same axis: if their weights or their block membership
ever diverged, the two lines would stop being comparable and nothing would raise.
"""
function _age_blocks(cfg::FrameworkConfig, grid)
    blk = [block_of(i, cfg) for i in 1:grid.N]
    _binlist(b) = join(grid.LAB[blk .== b], "+")
    blocks = ((1, :child, "children (bins " * _binlist(1) * ")"),
              (2, :adult, "adults (bins "   * _binlist(2) * ")"))
    wts = [grid.POP[i] / sum(grid.POP[blk .== blk[i]]) for i in 1:grid.N]
    return (; blk, blocks, wts)
end

"""
    _p0_draws_valid(p0d) -> Bool

Is every `p0f` coordinate in `p0d` (ndraws × A × A) an actual posterior?

A coordinate that is **identically 0.0 (or 1.0) across every draw** is not one. Under the hurdle
Binomial likelihood `nz·log(p0) + (nn−nz)·log1p(−p0)` with a `Beta(1,1)` prior, a cell the data
push toward the boundary still returns *varying* values around 1e-4 — never several thousand
bit-identical zeros. So the test is exact and cannot false-positive on a legitimate fit.

⚠ **WHY THIS EXISTS (2026-09-04).** The chain `weighted-hweibull @ 2021-02-14 h1`, the last file
of an interrupted transfer from the HPC, arrived with **61 of its 441 `p0f` coordinates
identically zero over all 2000 draws** — an unwritten array region, while `μ`, `κ`, the GP
scalars and the *other* `p0f` cells in the same chain were healthy. Since `K1 = (1−p0)·μ`, each
dead cell silently contributes its full `μ`, and the child totals at that origin inflated ×2.5
(2.837 → 7.141). It read as a real epidemiological signal and was not: the NegBin panel is flat
at that origin and only steps up at 2021-03-07, when English schools actually reopened.

Nothing else caught it. The file was normal size, so `isfile` passed; the dimensions were right,
so `_s1_structure` passed; the parameter names were right, so `_stage1_gp_generation` passed.
This is CLAUDE.md's "a truncated `.jld2` counts as complete" hazard in CONTENT rather than
length. The refilled 504-file grid is clean (0 of 63 h1 chains affected), so this guard is a
REGRESSION NET rather than a live fix — but the grid arrives by transfer, the failure is silent,
and it has cost one misreading already.
"""
function _p0_draws_valid(p0d::AbstractArray{<:Real,3})
    for j in CartesianIndices(axes(p0d)[2:3])
        v = view(p0d, :, j)
        first(v) == 0 || first(v) == 1 || continue
        all(==(first(v)), v) && return false
    end
    return true
end

"""
    _origin_quantities(dm, cfg, origin; grid, h, contacts, save_dir)
        -> (; quantities, excluded)

Read ONE origin's Stage-1 chain and build its per-draw `ndraws × A × A` contact matrices — the
whole of `collect_origin_contact_means`' chain-reading half, lifted out on 2026-09-10 so §6's
eigenvector figure could reuse it without a second pass over the grid.

`quantities` is a `Vector{Tuple{String,Array{Float64,3}}}` of `(quantity, C)` pairs:

    NegBin           "K1"     C[d,i,j] = μ[d,i,j]
    hurdle-Weibull   "K1"     C[d,i,j] = (1 − p0[d,i,j]) · μ[d,i,j]
    hurdle-Weibull   "excess" C[d,i,j] = base_contact(NeighbourhoodDegreeNGM(), _weibull_moments(…)...)

i.e. exactly the matrices §4 row-sums over contactee `j` and §6 eigen-decomposes. The chain is read
at week column `cfg.n_fit` (the origin week — since `-w8h` t₀ sits there in EVERY horizon's chain).

⚠ **THE EXTRACTION CONTRACT IS `isempty(quantities)` ⟺ THE OLD BARE `continue`.** All three skip
paths — no chain, no `p0f`, degenerate `p0f` — return no quantities, and the caller must write
`isempty(qs) && continue` AFTER draining `excluded`. Note that the first two skips record NOTHING
and warn NOTHING, deliberately: a missing chain is a legitimate gap (a partial grid renders as a gap
and lengthens by itself), and logging it would turn one into 126 warnings.

⚠ **THIS FUNCTION WARNS, SO IT MUST HAVE EXACTLY ONE CALL SITE.** Two collectors calling it would
double every `@warn` and duplicate every `excluded` row. §6 is therefore a KWARG on
`collect_origin_contact_means`, never a second collector.

⚠ `contacts` defaults to `contacts_label(cfg)`, never `CONTACTS_TOKEN` — see the caller's docstring.
`lbl`'s `MeanNGM()` is not a choice: Stage-1 artefacts carry no ngm token, so the NGM half of the
label is ignored by `stage1_chain_path`. Do not "fix" it to take an `nb`; that would change the
FILENAME read.
"""
function _origin_quantities(dm::ContactDegreeModel, cfg::FrameworkConfig, o::Date;
                            grid = cis_age_grid(),
                            h::Integer = 1,
                            contacts::AbstractString = contacts_label(cfg),
                            save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    lbl = string(degree_label(dm), "|", ngm_label(MeanNGM()))   # s1 path ignores the ngm half
    weighted = is_weighted(dm)
    deg = degree_label(dm)
    A = grid.N
    wk = cfg.n_fit          # bound ONCE: it is read four times, incl. inside a `reason` string that
                            # must never drift out of step with the week actually read
    quantities = Tuple{String,Array{Float64,3}}[]
    excluded = NamedTuple[]

    μd = reconstruct_mu_draws(lbl, o, h; week_index = wk, grid = grid,
                              contacts = contacts, save_dir = save_dir)
    μd === nothing && return (; quantities, excluded)
    D = size(μd, 1)
    if weighted
        p0d = reconstruct_p0_draws(lbl, o, h; grid = grid, week_index = wk,
                                   contacts = contacts, save_dir = save_dir)
        p0d === nothing && return (; quantities, excluded)
        if !_p0_draws_valid(p0d)
            push!(excluded, (; degree = deg, quantity = "K1+excess", origin = o,
                               reason = "p0f has coordinates identically 0/1 across all draws " *
                                        "— an unwritten artefact region, not a posterior"))
            @warn "14j: dropping origin from the $deg timeline — degenerate p0f" origin=o
            return (; quantities, excluded)
        end
        @assert size(p0d, 1) == D "μ and p0 draw counts disagree ($D vs $(size(p0d,1))) — the two \
            readers are indexed [d,i,j] in lockstep, so this is a chain that cannot be paired"
        # `_weibull_moments`' first return is exactly this, and needs no κ — so K1 survives a
        # chain whose dispersion cannot be read, while `excess` (which does need κ) does not.
        push!(quantities, ("K1", (1 .- p0d) .* μd))
        κd = reconstruct_dispersion_draws(lbl, o, h; weighted = true, cfg = cfg, grid = grid,
                                          week_index = wk, contacts = contacts,
                                          save_dir = save_dir)
        if κd === nothing
            push!(excluded, (; degree = deg, quantity = "excess", origin = o,
                               reason = "log_kappa unreadable at week $wk — K1 kept, " *
                                        "excess degree dropped for this origin"))
            @warn "14j: no dispersion draws — dropping the excess-degree point" origin=o
        else
            @assert size(κd, 1) == D "μ and κ draw counts disagree ($D vs $(size(κd,1)))"
            # THE FRAMEWORK'S OWN FUNCTIONS, not an algebraic restatement of them: `_weibull_moments`
            # (joint_model.jl) for (⟨k⟩, ⟨k²⟩, g) and `base_contact` (ngm.jl) for the builder's C0.
            # That is the standing reconstruct-matches-model rule — a re-derived `(1−p⁰)·μ·(1+CV²)`
            # here would silently stop tracking `base_contact` the first time it changed.
            EX = Array{Float64,3}(undef, D, A, A)
            for d in 1:D, i in 1:A, j in 1:A
                k1, k2, g = _weibull_moments(μd[d, i, j], κd[d, i, j], p0d[d, i, j])
                EX[d, i, j] = base_contact(NeighbourhoodDegreeNGM(), k1, k2, g)
            end
            push!(quantities, ("excess", EX))
        end
    else
        push!(quantities, ("K1", μd))
    end
    return (; quantities, excluded)
end

"""
    _perron_share(C) -> (; v, reason)

Dominant right (Perron) eigenvector of the nonnegative matrix `C`, sign-fixed and normalised to
sum 1 — or `v = nothing` plus a `reason` string when `C` does not admit one.

`v[i]` is the stationary share of age bin `i` under the map `x ↦ C·x`. With `C = C*` that is the
age distribution the CONTACT STRUCTURE ALONE would drive: the renewal is `I(t) = N·Σ_s w(s)·I(t−s)`
with `N[a,b] = γ_SAR · susc_a·(1+(F−1)A_a) · C*[a,b] · inf_b` (`ngm.jl`, `renewal.jl`), so the right
Perron vector of the FULL `N` is the stable age distribution of infections — and `C*` alone drops
γ_SAR, susceptibility, infectivity, `F` and antibody, **every one of which is age-varying**. This is
therefore the contact-only analogue of `contact_reproduction_draws`' `ρ(C*)`, NOT the model's
predicted case age-distribution. Say so wherever it is plotted.

⚠ **NEVER `abs.(v)`.** For a strictly positive `C` — which every healthy fitted `C*` is, since
`μ = exp(_softclamp(·,−8,6))`, `p0f ~ Beta(1,1)` is stored constrained and `κ = exp(_softclamp(·,−4.3,5))` —
Perron–Frobenius makes the dominant eigenvector real, simple and single-signed, so `abs` is a no-op.
Where the vector is NOT single-signed we did not compute the Perron vector at all, and `abs` turns
that failure into a plausible probability vector that sums to 1 and plots beautifully. That is the
blank-but-plausible pathology this file already guards three times. The sign is fixed from
`sum(real(v))` and positivity is then CHECKED.

Refuses on: a non-finite entry; a non-positive entry (a single-draw `p⁰ = 1` gives `k1 = 0`, so
`base_contact` takes its zero branch and `C` becomes reducible — `_p0_draws_valid` only catches
coordinates dead across EVERY draw); a complex dominant eigenpair; or a residual
`‖Cv − λv‖∞ / (λ‖v‖∞) > 1e-8`.
"""
function _perron_share(C::AbstractMatrix)
    all(isfinite, C) || return (; v = nothing, reason = "non-finite entry in C")
    all(>(0), C) || return (; v = nothing,
                              reason = "non-positive entry in C — the Perron vector of a reducible " *
                                       "matrix is not unique")
    F = eigen(Matrix{Float64}(C))
    i = argmax(real.(F.values))          # Perron root: strictly largest in modulus ⇒ also in ℜ
    λ = F.values[i]
    w = F.vectors[:, i]
    r = real.(w)
    abs(imag(λ)) <= 1e-8 * abs(λ) || return (; v = nothing, reason = "dominant eigenvalue is complex")
    maximum(abs, imag.(w)) <= 1e-8 * maximum(abs, r) ||
        return (; v = nothing, reason = "dominant eigenvector is complex")
    sum(r) < 0 && (r = -r)               # fix the arbitrary sign — do NOT abs
    all(>(0), r) || return (; v = nothing, reason = "dominant eigenvector is not single-signed")
    λr = real(λ)
    res = maximum(abs, C * r .- λr .* r) / (λr * maximum(r))
    res <= 1e-8 || return (; v = nothing, reason = "residual ‖Cv−λv‖ = $(round(res; sigdigits = 3))")
    return (; v = r ./ sum(r), reason = "")
end

"""
    _push_eigen_rows!(eigrows, eigexcl, C, key, o, grid) -> nothing

Summarise one origin's `ndraws × A × A` matrices into per-bin Perron shares.

`key` is the NamedTuple identifying the series, PREPENDED to every emitted row: `(; degree, quantity)`
for §6's `C*` frame and `(; degree, ngm)` for its NGM twin. It is a parameter rather than two
positional strings because the NGM frame's second axis really is the **NGM builder** — writing it
into a `quantity` column would overload that field the way a `"weighted-hweibull-excess"`
pseudo-label would overload `degree` (the standing "never sniff a model property out of a label"
rule). With `key = (; degree, quantity)` the emitted schema and column order are unchanged.

`_perron_share` is applied PER DRAW (the eigenvector is a nonlinear function of `C*`, so
decomposing a median matrix is not the median of the decompositions), failing draws are DROPPED and
the rest kept, then each component is reduced to its median and 5/95th percentiles.

⚠ **The median and its band are rescaled by the SAME scalar `1/sum(med)`.** Per-component medians of
vectors that each sum to 1 do not themselves sum to 1; renormalising the median alone would leave
`lo`/`hi` on the old scale and the band would stop bracketing the plotted line. `sum_before` records
what the raw median summed to (typically 1 ± 1e-3).

⚠ A partly-failing origin keeps its good draws and writes ONE `eigen_excluded` row with `n_bad` and
the modal reason; only `n_bad == n_draws` drops the series. A dropped series must never be silent.
`ratio` (`max(C)/min(C)`, worst over the kept draws) and `vmin` are carried because `1 + CV²` has a
large dynamic range at the κ floor — the same root cause that forced `obs_clip_q = 0.99` onto
`_TL_SPECS` panel 3 — and a `C` spanning many orders of magnitude collapses the Perron vector onto
one coordinate: arithmetically correct, scientifically empty.
"""
function _push_eigen_rows!(eigrows::Vector{NamedTuple}, eigexcl::Vector{NamedTuple},
                           C::AbstractArray{<:Real,3}, key::NamedTuple, o::Date, grid)
    D, A = size(C, 1), size(C, 2)
    good = Vector{Vector{Float64}}()
    reasons = String[]
    ratio = 0.0
    for d in 1:D
        M = view(C, d, :, :)
        r = _perron_share(M)
        if r.v === nothing
            push!(reasons, r.reason)
        else
            push!(good, r.v)
            ratio = max(ratio, maximum(M) / minimum(M))
        end
    end
    n_bad = D - length(good)
    if isempty(good)
        push!(eigexcl, merge(key, (; origin = o, n_draws = D, n_bad = n_bad,
                                     reason = isempty(reasons) ? "no draws" : _modal(reasons))))
        @warn "14j §6: no usable Perron vector — dropping series" series=key origin=o
        return nothing
    end
    n_bad > 0 && push!(eigexcl, merge(key, (; origin = o, n_draws = D, n_bad = n_bad,
                                              reason = _modal(reasons))))
    V = reduce(hcat, good)'                                     # n_good × A
    med = [median(view(V, :, i)) for i in 1:A]
    lo  = [quantile(view(V, :, i), 0.05) for i in 1:A]
    hi  = [quantile(view(V, :, i), 0.95) for i in 1:A]
    s = sum(med)
    for i in 1:A
        push!(eigrows, merge(key, (; origin = o, week = week_mid(o),
                                     bin = i, bin_label = grid.LAB[i],
                                     med = med[i] / s, lo = lo[i] / s, hi = hi[i] / s,
                                     sum_before = s, vmin = minimum(med) / s, ratio = ratio,
                                     n_draws = D, n_bad = n_bad)))
    end
    return nothing
end

"Most frequent element of a non-empty vector (the modal exclusion reason)."
_modal(v::AbstractVector) = isempty(v) ? "" : argmax(countmap(v))

"""
    collect_origin_contact_means(dm, cfg; grid, h, origins, contacts, save_dir, with_eigen)
        -> (; bins, blocks, eigen, excluded, eigen_excluded)

Total fitted contacts per contactor, read from the Stage-1 chains, at two levels:

- `bins`   — one row per (origin × contactor CIS bin), the raw seven-bin detail
- `blocks` — one row per (origin × child/adult block), the headline quantity
- `eigen`  — §6's per-bin Perron shares, EMPTY unless `with_eigen = true` (see below)
- `excluded` — origins dropped by the `p0f` guard, with the reason (see below)
- `eigen_excluded` — series/draws dropped by `_perron_share`, a SEPARATE frame

For each origin the chain is read at **week column `cfg.n_fit`** — the origin week. Since
`-w8h` every horizon's degree window is anchored at `t₀ − n_fit + 1`, so t₀ sits at column
`n_fit` in *every* chain regardless of `h` (this is the same constant 10j asserts as
`t_o_est == cfg.n_fit`; it was `t_o − 1` only while the window slid with the horizon).

The per-cell quantity is the **incl-zero raw first moment** `K1`, matching `_weibull_moments`:

    NegBin           K1[i,j] = μ[i,j]
    hurdle-Weibull   K1[i,j] = (1 − p0[i,j]) · μ[i,j]

and the total for contactor `i` is `Σ_j K1[i,j]`. μ already carries `log(pop_j / pop_ref)`, so
that row sum is contacts per participant in bin `i` — "combining all contact means for each
contactor". The block value is the **population-weighted** mean of those totals over the bins in
the block, with blocks from `block_of(i, cfg)` — the same child/adult rule §1–§3 use:

    block_b = Σ_{i ∈ b} w_i · total_i,     w_i = POP_i / Σ_{i' ∈ b} POP_{i'}

⚠ **Both levels are summarised PER DRAW, in one pass.** The block interval is the interval of the
aggregate, not a combination of the per-bin intervals — those would be wrong, since the bins
within a block are strongly correlated through the shared GP and summing their quantiles would
overstate the spread.

⚠ `contacts` defaults to `contacts_label(cfg)` and NOT to `CONTACTS_TOKEN`. The latter is a
compile-time constant built from the default `FrameworkConfig`, and leaving it as the default on
readers like these is the bug that blanked five 10j figures and all eight 9j transmission panels.

Origins with no chain are SKIPPED, not failed, so a partial grid renders a gap and lengthens by
itself as chains are inserted. Origins failing `_p0_draws_valid` are also skipped, but are
RECORDED in `excluded` so the caller can print them — a dropped origin must never be silent.

`with_eigen = true` additionally fills `eigen` with `_push_eigen_rows!`'s per-bin dominant-eigenvector
shares of the SAME matrices (§6's "expected age distribution"). It is a kwarg on this function rather
than a second collector for one reason: the chain read is the entire cost — a 2000-draw JLD2 load
plus 2000 Cholesky factorisations inside `reconstruct_mu_draws`, per origin per model — while the
eigendecompositions are seconds. A separate collector would double that AND double every `@warn`
`_origin_quantities` emits. It defaults to `false` so the function's observable behaviour is
unchanged for callers that predate §6.

⚠ `eigen_excluded` is a SEPARATE frame, not another `quantity` value in `excluded`.
`DataFrame(excluded)` needs one schema across all elements, and §4's cell prints `excluded` as
"nothing was excluded ⇒ every chain is a genuine posterior" — folding §6's per-draw failures in
would make §4 warn about §6's problems.
"""
function collect_origin_contact_means(dm::ContactDegreeModel, cfg::FrameworkConfig;
                                      grid = cis_age_grid(),
                                      h::Integer = 1,
                                      origins::AbstractVector{Date},
                                      contacts::AbstractString = contacts_label(cfg),
                                      save_dir::AbstractString = joinpath(@__DIR__, "..",
                                                                          "dt_intermediate"),
                                      with_eigen::Bool = false)
    deg = degree_label(dm)
    A = grid.N

    blk, blocks, wts = _age_blocks(cfg, grid)                   # SHARED with observed_contact_means

    binrows = NamedTuple[]; blockrows = NamedTuple[]; excluded = NamedTuple[]
    eigrows = NamedTuple[]; eigexcl = NamedTuple[]
    q3(v) = (median(v), quantile(v, 0.05), quantile(v, 0.95))
    rowsum(X) = dropdims(sum(X; dims = 3); dims = 3)            # ndraws × A, summed over contactee j
    for o in origins
        # ONE read per origin; §4 row-sums the matrices and §6 eigen-decomposes them.
        qs, ex = _origin_quantities(dm, cfg, o; grid = grid, h = h, contacts = contacts,
                                    save_dir = save_dir)
        append!(excluded, ex)
        isempty(qs) && continue     # ⟺ the three bare `continue`s this loop body used to have
        for (qty, C) in qs
            tot = rowsum(C)
            for i in 1:A
                m, l, u = q3(view(tot, :, i))
                push!(binrows, (; degree = deg, quantity = qty, origin = o, week = week_mid(o),
                                  bin = i, bin_label = grid.LAB[i], block = blk[i],
                                  med = m, lo = l, hi = u))
            end
            for (b, sym, lab) in blocks
                idx = findall(==(b), blk)
                agg = tot[:, idx] * wts[idx]                    # per-draw population-weighted mean
                m, l, u = q3(agg)
                push!(blockrows, (; degree = deg, quantity = qty, origin = o, week = week_mid(o),
                                    block = b, block_label = String(sym), block_desc = lab,
                                    med = m, lo = l, hi = u))
            end
            with_eigen && _push_eigen_rows!(eigrows, eigexcl, C, (; degree = deg, quantity = qty),
                                            o, grid)
        end
    end
    return (; bins = DataFrame(binrows), blocks = DataFrame(blockrows),
              eigen = DataFrame(eigrows), excluded = DataFrame(excluded),
              eigen_excluded = DataFrame(eigexcl))
end

"""
    observed_contact_means(cfg; grid, origins, raw) -> (; bins, blocks, apd)

The RAW observed contact means the §4 fitted lines are overlaid on — one row per
(degree × origin × bin) and per (degree × origin × block), matching
`collect_origin_contact_means`'s frames so the two join on those keys.

Both quantities are **incl-zero per-participant-day means**, summed over contactee bins, which is
what `K1` is:

    unweighted   obs[i] = Σ_j emp_mean[t,i,j]
    weighted     obs[i] = Σ_j (1 − p0[t,i,j]) · whist_mean(pos_weight[t,i,j])      (empty cell ⇒ 0)

⚠ **NOT `_observed_cell_mean(…; weighted = true)`** (`10j_viz_utils.jl`), which returns
`whist_mean(pw)` alone — the mean of the POSITIVE weighted degrees. That is the correct comparator
for **μ**, and 10j uses it against μ. §4 plots `K1 = (1−p0)·μ`, so the observed must carry the same
`(1−p0)` factor; without it the overlay sits systematically high by a factor of ~4 on the weighted
path (p⁰ ≈ 0.75 per cell), which would read as the model badly under-fitting rather than as a
units mismatch.

⚠ **ONE `prepare_degree_data` call covers the whole timeline.** `origins` are consecutive weekly
Sundays, so `WeeklyWindow(last(origins); n_fit = length(origins), horizons = 1:0)` has
`fit_weeks == origins` exactly and `prepare_degree_data` returns one cell per origin week. Building
a window per origin would repeat the same two Arrow reads 63 times.

⚠ **This is the MODEL's own pipeline, deliberately** — same binning, same duration weights, same
roster-derived zeros — so the overlay is like-for-like rather than an independently-computed number
that would differ for uninteresting reasons. It costs 0.35% of contacts (3061 of 865108 over
2020-10-18…2021-12-26), whose contactee age cannot be assigned to a CIS bin and which
`prepare_degree_data` drops.
"""
function observed_contact_means(cfg::FrameworkConfig;
                                grid = cis_age_grid(),
                                origins::AbstractVector{Date},
                                raw = nothing)
    isempty(origins) && return (; bins = DataFrame(), blocks = DataFrame(), apd = nothing)
    raw === nothing && (raw = load_raw_contact_inputs())
    A = grid.N
    blk, blocks, wts = _age_blocks(cfg, grid)                   # SHARED with the fitted collector

    win = WeeklyWindow(last(origins); n_fit = length(origins), smax = cfg.smax, horizons = 1:0)
    @assert win.fit_weeks == collect(origins) "origins are not consecutive weekly Sundays — the \
        single-window shortcut above does not hold; build one window per origin instead"
    apd = prepare_degree_data(win, cfg; grid = grid,
                              df_part_raw = raw.df_part, craw_raw = raw.craw)

    # `whist_mean`'s second-moment sibling — `degree_dist.jl` has no `whist_mom2`, and one figure is
    # not reason enough to add a method to a core file.
    _whist_mom2(w) = isempty(w) ? NaN : sum(w.x .^ 2 .* w.y) / sum(w.y)

    # (K1, excess) for one cell. The weighted branch goes through `base_contact` — the SAME framework
    # functional the fitted side uses — on empirical moments in place of fitted ones, so a change to
    # the builder moves both series together or neither.
    function _cell(t, i, j, weighted)
        weighted || return (apd.emp_mean[t, i, j], NaN)
        pw = apd.pos_weight[t, i, j]
        isempty(pw) && return (0.0, 0.0)                        # empty cell contributes nothing
        p0 = apd.p0[t, i, j]
        k1 = (1 - p0) * whist_mean(pw)
        k2 = (1 - p0) * _whist_mom2(pw)
        return (k1, base_contact(NeighbourhoodDegreeNGM(), k1, k2, 1 - p0))
    end

    binrows = NamedTuple[]; blockrows = NamedTuple[]
    for (deg, weighted) in (("unweighted-negbin", false), ("weighted-hweibull", true))
        qts = weighted ? ("K1", "excess") : ("K1",)
        for (t, o) in enumerate(origins)
            cells = [_cell(t, i, j, weighted) for i in 1:A, j in 1:A]
            for (qi, qty) in enumerate(qts)
                tot = [sum(j -> cells[i, j][qi], 1:A) for i in 1:A]
                for i in 1:A
                    push!(binrows, (; degree = deg, quantity = qty, origin = o, week = week_mid(o),
                                      bin = i, bin_label = grid.LAB[i], block = blk[i],
                                      observed = tot[i]))
                end
                for (b, sym, lab) in blocks
                    idx = findall(==(b), blk)
                    push!(blockrows, (; degree = deg, quantity = qty, origin = o, week = week_mid(o),
                                        block = b, block_label = String(sym), block_desc = lab,
                                        observed = sum(wts[idx] .* tot[idx])))
                end
            end
        end
    end
    # `apd` is returned so §5 can assert its own roster against `apd.n[t,i,1]` (the roster count
    # for bin `i`, constant in `j`) WITHOUT a second `prepare_degree_data` call. That assertion is
    # what proves §5's `assign_age_bin` draws are bit-identical to the ones behind §4's observed
    # overlay — two independent rng streams that must agree because they use the same source frame,
    # the same date predicate and the same roster `unique`.
    return (; bins = DataFrame(binrows), blocks = DataFrame(blockrows), apd = apd)
end

# The two §4 figures differ ONLY in what they key their series on — child/adult blocks, or the seven
# CIS bins. Everything else (fitted line + band, observed markers, phase shading, panel furniture) is
# identical and lives in `_timeline_panel!`, so the two cannot drift in how they draw the
# fitted-vs-observed comparison. Same motivation as `_stage1_gp_generation` in 10j.

# (degree, quantity, y label, panel title, per-panel `obs_clip_q` default).
#
# Panels 1–2 are `MeanNGM`'s C0 for the two degree models; panel 3 is `NeighbourhoodDegreeNGM`'s C0
# for the weighted one — the two NGM builders' inputs from the SAME Stage-1 draws, which is the axis
# the forecasting grid is a 2×2 of. Extending this tuple is all it takes to add a panel: the figure
# derives its layout and height from `length(_TL_SPECS)` rather than hard-coding a row count (the
# `panel_grid` lesson from `9j_viz_utils.jl`, where a fixed `layout = (2,2)` silently dropped panels
# once a fifth model existed).
#
# ⚠ Panel 3 carries a y-cap by DEFAULT where panels 1–2 do not. The observed excess is an empirical
# SECOND moment over positive weighted degrees, so a single mass-contact report — already the reason
# the per-bin figure needs a cap on the mean — dominates it far harder after squaring.
const _TL_SPECS = (("unweighted-negbin", "K1", "contacts / participant-day",
                    "unweighted-negbin — ⟨k⟩ summed over contactee bins  (mean NGM C₀)",
                    nothing),
                   ("weighted-hweibull", "K1", "duration-weighted / participant-day",
                    "weighted-hweibull — (1−p⁰)·μ summed over contactee bins  (mean NGM C₀)",
                    nothing),
                   ("weighted-hweibull", "excess", "excess degree / participant-day",
                    "weighted-hweibull — excess degree (⟨k²⟩/⟨k⟩)·g summed  (neighbourhood NGM C₀)",
                    0.99))

"An axis with no series has no limits, so `shade_periods!`/`annotate!` would throw on it."
function _timeline_empty_panel(deg, ttl)
    pnl = plot(; xlims = (0, 1), ylims = (0, 1), framestyle = :box, legend = false,
                 title = ttl, titlefontsize = 8, xticks = false, yticks = false)
    annotate!(pnl, 0.5, 0.5, text("no Stage-1 chains on disk for $deg", 9, :gray))
    return pnl
end

"""
    _timeline_panel!(pnl, sub, obs, keycol, keys, labels, colours; showlegend, band, ms, phase_labels)

One §4 panel: a fitted median + 90% band per key, the matching observed series as `×` markers, and
the named UK COVID phases shaded behind.

⚠ **`shade_periods!` is called LAST, and that is required, not stylistic.** Its docstring: call it
after the data series so the Date axis and y-limits are already established — a leading numeric
overlay collapses the date axis. 14j shaded first while it used fixed windows; with `PERIODS` it
must not.

Observed gets ONE legend entry rather than one per key: with seven bins the legend would otherwise
be 14 rows and eat the panel.
"""
function _timeline_panel!(pnl, sub::AbstractDataFrame, obs, keycol::Symbol, keys, labels, colours;
                          showlegend::Bool = true, band::Real = 0.15, ms::Real = 3,
                          phase_labels::Bool = true, obs_clip_q = nothing)
    dmin, dmax = extrema(sub.week)
    first_obs = true
    n_above = 0
    for (k, lab, col) in zip(keys, labels, colours)
        sf = sort(sub[sub[!, keycol] .== k, :], :week)
        nrow(sf) == 0 && continue
        plot!(pnl, sf.week, sf.med; ribbon = (sf.med .- sf.lo, sf.hi .- sf.med),
              color = col, lw = 2, fillalpha = band, label = showlegend ? lab : "")
        obs === nothing && continue
        so = sort(obs[obs[!, keycol] .== k, :], :week)
        nrow(so) == 0 && continue
        scatter!(pnl, so.week, so.observed; color = col, marker = :x, ms = ms, msw = 1.2,
                 label = (showlegend && first_obs) ? "observed (raw data)" : "")
        first_obs = false
    end
    # ZERO BASELINE (2026-09-10, user request). Every series here is a rate per participant-day and
    # every `lo` is a 5th percentile of a positive quantity, so nothing is clipped — but an auto lower
    # limit starts the child panel near 2 and the adult panel near 1, which visually exaggerates every
    # wiggle and makes the three quantities look further apart than they are.
    # ⚠ THIS IS THE ONE PLACE TO FIX IT. All three §4 figures (`make_contact_mean_timeline_fig`,
    # `make_contact_mean_bins_fig`, `make_contact_quantities_fig`) route through this helper; do not
    # add per-figure `ylims`.
    ylo = 0.0
    _, yhi = Plots.ylims(pnl)
    # OPTIONAL ROBUST Y-CAP. The observed series is a weekly per-participant mean over a roster that
    # can be small — 2020-11-29 has just 65 child participant-days, one of which reported a mass
    # contact, giving an observed 33.8 against a median of 4.2 across all bins and weeks. Left alone
    # that single point sets the axis and flattens every line into the bottom fifth of the panel.
    # Capping at `quantile(observed, obs_clip_q)` (never below the fitted bands, which must always be
    # fully visible) restores the panel; the count of points pushed off-scale is RETURNED so the
    # caller can state it on the panel rather than hiding it.
    if obs_clip_q !== nothing && obs !== nothing && nrow(obs) > 0
        cap = max(maximum(sub.hi), quantile(obs.observed, obs_clip_q))
        n_above = count(>(cap), obs.observed)
        yhi = cap
    end
    # Headroom BEFORE shading: `shade_periods!` reads the panel's y-limits and writes its rotated
    # phase names at 93% of the range, which lands on top of the series (and of the legend) unless
    # a clear band is opened first. 25% is enough for the longest label ("Opening up") at size 5.
    ylims!(pnl, (ylo, ylo + 1.25 * (yhi - ylo)))
    shade_periods!(pnl, dmin, dmax; alpha = 0.10, fontsize = 5, labels = phase_labels)
    return n_above
end

"""
    _make_timeline_fig(tl, obs, keycol, keys, labels, colours, fname, ptitle; …) -> Plots.Plot

One stacked panel per `_TL_SPECS` entry, built from `_timeline_panel!`. `tl` is a fitted frame from
`collect_origin_contact_means` and `obs` the matching frame from `observed_contact_means` (or
`nothing`); each panel is selected by **both** `:degree` and `:quantity`, since the weighted model
contributes two of them (`K1` and `excess`).

The layout, the figure height and which panel carries the rotated phase names are all DERIVED from
`length(_TL_SPECS)`; `size_ = nothing` (the default) sizes the figure to the panel count. Phase
labels go on the BOTTOM panel only and the legend on the top one — every panel shares one calendar
and one set of series, so each belongs in exactly one place.

`obs_clip_q` given at figure level applies to every panel; left `nothing`, each panel falls back to
its own `_TL_SPECS` default (which is how the excess panel gets a cap in the block figure while the
two mean panels keep an uncapped axis).
"""
function _make_timeline_fig(tl::DataFrame, obs, keycol::Symbol, keys, labels, colours,
                            fname::AbstractString, ptitle::AbstractString;
                            res_dir::AbstractString = "../res", band::Real = 0.15,
                            ms::Real = 3, size_ = nothing, legendfontsize::Integer = 7,
                            obs_clip_q = nothing)
    panels = Any[]
    npanel = length(_TL_SPECS)
    for (pi, (deg, qty, ylab, ttl, spec_clip)) in enumerate(_TL_SPECS)
        sub = @subset(tl, :degree .== deg, :quantity .== qty)
        if nrow(sub) == 0
            push!(panels, _timeline_empty_panel(deg, ttl)); continue
        end
        pnl = plot(; xlabel = "forecast origin (Wed mid-date)", ylabel = ylab,
                     title = ttl, titlefontsize = 8, xrotation = 45,
                     legend = pi == 1 ? :topleft : false, legendfontsize = legendfontsize)
        ob = obs === nothing || nrow(obs) == 0 ? nothing :
             @subset(obs, :degree .== deg, :quantity .== qty)
        # Legend on the TOP panel, phase names on the BOTTOM one. Every panel shares one calendar and
        # one set of series, so each belongs in exactly one place — and putting them in the same
        # panel makes the rotated phase text collide with the legend box.
        n_above = _timeline_panel!(pnl, sub, ob, keycol, keys, labels, colours;
                                   showlegend = pi == 1, band = band, ms = ms,
                                   phase_labels = pi == npanel,
                                   obs_clip_q = obs_clip_q === nothing ? spec_clip : obs_clip_q)
        n_above > 0 && plot!(pnl; title = ttl * "   [$(n_above) observed pt$(n_above == 1 ? "" : "s") above axis]")
        push!(panels, pnl)
    end
    fig = plot(panels...; layout = (npanel, 1), size = size_ === nothing ? (1100, 410 * npanel) : size_,
               left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, top_margin = 5Plots.mm,
               plot_title = ptitle, plot_titlefontsize = 9)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, fname))
    return fig
end

"""
    make_contact_mean_timeline_fig(tl; obs, grid, res_dir="../res") -> Plots.Plot

§4's headline figure: fitted contact rate by **child/adult block**, median and 90% band, with the
raw observed values overlaid as `×` and the named UK COVID phases shaded behind. One panel per
`_TL_SPECS` entry — the two mean-NGM `C₀` panels (NegBin, hurdle-Weibull) and the hurdle-Weibull
**excess-degree** panel, which is the neighbourhood-NGM `C₀` from the same Stage-1 draws.

`tl` is the `blocks` frame from `collect_origin_contact_means` — its intervals are the intervals OF
THE AGGREGATE (summarised per draw upstream), not a combination of the per-bin ones. `obs` is the
matching `blocks` frame from `observed_contact_means`. Both carry a `quantity` column
(`"K1"` / `"excess"`), which is half of each panel's selector.

The phase bands come from **9j's `PERIODS`/`shade_periods!`**, i.e. the same Munday-2023 Table 2
boundaries `plot_wis_diff_over_time` uses — reused rather than restated, so the two notebooks can
never disagree about when a phase started. `PERIODS` runs 2020-11-05 … 2021-11-24, so origins
outside that (the first ~3 weeks and last ~5 of this grid) are legitimately unshaded, exactly as in
9j.

Saves `res_dir/14j_contact_mean_timeline.png` and the plotted table as
`14j_contact_mean_timeline.csv` (with the `observed` column joined in when `obs` is given).
"""
function make_contact_mean_timeline_fig(tl::DataFrame; obs = nothing, grid = cis_age_grid(),
                                        res_dir::AbstractString = "../res")
    # legend text = the block's own description ("children (bins 2-10+11-15)"), taken from the frame
    # so the bin membership shown always matches `cfg.child_bins` rather than a hard-coded string.
    desc(b, fallback) = (r = @subset(tl, :block_label .== b);
                         nrow(r) == 0 ? fallback : first(r.block_desc))
    fig = _make_timeline_fig(tl, obs, :block_label,
                             ("child", "adult"),
                             [desc("child", "children"), desc("adult", "adults")],
                             (:darkorange, :steelblue),
                             "14j_contact_mean_timeline.png",
                             "Fitted contact rate by age block (lines, 90%) vs raw observed (×), " *
                             "Stage-1 chains at the origin week (h=1); bands = UK COVID phases";
                             res_dir = res_dir)
    out = obs === nothing ? tl :
          leftjoin(tl, select(obs, [:degree, :quantity, :origin, :block_label, :observed]),
                   on = [:degree, :quantity, :origin, :block_label])
    CSV.write(joinpath(res_dir, "14j_contact_mean_timeline.csv"),
              sort(out, [:degree, :quantity, :origin, :block_label]))
    return fig
end

"""
    make_contact_mean_bins_fig(tl; obs, grid, res_dir="../res") -> Plots.Plot

The per-age-group twin of `make_contact_mean_timeline_fig`: the same panels, but one fitted line per
CIS bin (seven, coloured by `_PART_COLS`) plus their observed series. Kept a twin deliberately — both
figures go through `_make_timeline_fig`, so a panel added to `_TL_SPECS` appears in both and the two
cannot drift.

With 14 series per panel the ribbons are thinned (`fillalpha = 0.06`) and the observed markers
shrunk, so the lines stay readable; the observed series share ONE legend entry. Read this figure for
which age groups drive the block lines, and the block figure for the headline.

`obs_clip_q = 0.99` is passed at FIGURE level here, so it overrides the per-panel defaults and caps
all three panels — the per-bin observed series is a weekly mean over a roster that can be as small as
65 participant-days, and one mass-contact report otherwise flattens every line.

Saves `res_dir/14j_contact_mean_timeline_bins.png`, beside the `…_bins.csv` written by the notebook.
"""
function make_contact_mean_bins_fig(tl::DataFrame; obs = nothing, grid = cis_age_grid(),
                                    res_dir::AbstractString = "../res")
    return _make_timeline_fig(tl, obs, :bin, 1:grid.N, grid.LAB, _PART_COLS,
                              "14j_contact_mean_timeline_bins.png",
                              "Fitted contact rate by CIS age bin (lines, 90%) vs raw observed (×), " *
                              "Stage-1 chains at the origin week (h=1); bands = UK COVID phases";
                              res_dir = res_dir, band = 0.06, ms = 2, legendfontsize = 6,
                              obs_clip_q = 0.99)
end

##########################################################################
# §F — pandemic-phase colouring, and §5's raw home/non-home duration share
##########################################################################

"Bucket for weeks `PERIODS` does not cover (3 origins before it, 5 after, on the current grid)."
const _OUTSIDE_PHASES = "outside phases"

"""
    _period_cols() -> Dict{String,Any}

One colour per named UK COVID phase, plus `"outside phases"` for weeks the table does not cover.

⚠ **A FUNCTION, NOT A `const`.** `PERIODS`/`PERIOD_ORDER` live in `9j_viz_utils.jl`, which this
file's header does NOT list as a dependency — every other 9j reference here (`shade_periods!`,
`PERIODS`) is likewise inside a function body, so `14j_viz_utils.jl` still loads standalone. A
top-level `const` would evaluate at include time and break that.

Sized from `length(PERIOD_ORDER)`, never a 7-long literal: append an eighth period to `PERIODS` and
an eighth colour appears (the `panel_grid` lesson — the hard-coded `layout = (2,2)` that silently
lost panels at six models). The palette is QUALITATIVE, because these are identities and not a
ranking; the out-of-range bucket is grey AND dashed at the call site, so it is never colour-alone.
"""
function _period_cols()
    n = length(PERIOD_ORDER)
    d = Dict{String,Any}(zip(PERIOD_ORDER, collect(palette(:tab10, max(n, 3)))[1:n]))
    d[_OUTSIDE_PHASES] = :gray65
    return d
end

"""
    _period_group(d) -> String

`period_of(d)`, or the single bucket `"outside phases"`.

⚠ Do NOT reuse 9j's `period_summary` bucket `"(pre-Lockdown 2)"`: `PERIODS` spans
2020-11-05 … 2021-11-24 while the origin grid runs 2020-10-18 … 2021-12-26, so **3 origins fall
before and 5 fall after**, and that label mislabels five of the eight. Classify the ORIGIN (Sunday),
matching `period_summary(origins)`, not the Wednesday `week_mid` label — the two agree on this grid
but need not in general.
"""
_period_group(d::Date) = (p = period_of(d); p === missing ? _OUTSIDE_PHASES : p)

"Phases actually present in `origins`, in PERIOD_ORDER, with the outside bucket last if non-empty."
function _periods_present(origins)
    g = unique(_period_group.(origins))
    return vcat([p for p in PERIOD_ORDER if p in g], _OUTSIDE_PHASES in g ? [_OUTSIDE_PHASES] : String[])
end

"""
    contact_duration_shares(cfg; grid, origins, raw) -> (; bins, blocks)

§5. The share of reported CONTACT DURATION spent at home vs away, per week, by CIS age bin and by
child/adult block — **straight from the raw CoMix table**: no chain, no fit, no model.

Duration is the framework's own weight put back on its natural scale:
`duration_weight(_uk_duration_multi(cnt_minutes_max, cnt_total_time), cfg.d_max) * cfg.d_max`
minutes, with group (`"mass"`) contacts taking `cfg.w_dur_group * cfg.d_max` = 2.5 min exactly as
`prepare_degree_data` assigns them. Shares are scale-free, so the `* cfg.d_max` only makes
`min_per_pd` readable.

Measured facts this figure rests on (whole raw table, 2020-10-18 … 2021-12-31):
home is **29 % of contacts but 72 % of duration**; **all 674 908 group contacts are non-home and all
have missing duration**, so the group lever moves the non-home side only, by 1.2–1.4 pp.

⚠ **ALWAYS PASS `cfg.d_max`.** `duration_weight(d, d_max::Real = 300)` defaults to **300** while
`cfg.d_max` is **240** — omitting it rescales level 4 from 0.625 to 0.5 with no error at all. Same
class as the `CONTACTS_TOKEN` default gotcha.

⚠ **`standardise_cnt_home_values!` MUTATES ITS ARGUMENT.** `raw.craw` carries `cnt_home` as a raw
`Int64` 0/1 and is shared with §1–§4 and §6; call the standardiser only on the `@select`ed copy
(`prepare_degree_data` gets this right by rebinding `craw` twice first). The resulting value set is
ASSERTED to be exactly `{"true","false"}`: a schema change to e.g. `"Yes"/"No"` would make
`:cnt_home .== "true"` silently all-false and report a 0 % home share.

⚠ **Blocks are POPULATION-WEIGHTED** with `_age_blocks`' `wts`, matching §4 — otherwise "children"
would denote one population in §4 and another in §5 of the same publication. The pooled
`Σminutes/Σminutes` alternative is reported as `share_pooled` so the choice is visible rather than
hidden. Bins with `n_pd == 0` are dropped and the weights RE-NORMALISED over the bins present
(recorded as `n_bins_present`), else a block share jumps silently when a bin empties.

⚠ `assign_age_bin` CONSUMES an rng. This helper owns its own `MersenneTwister(cfg.seed)` and draws
for **participants only** — home/non-home and duration are contact-level attributes, so no contactee
bin is involved and there is no second call site. It uses the same source frame, the same date
predicate and the same `unique(...)` roster as `prepare_degree_data`, so its participant draws are
bit-identical to §4's; the notebook asserts that via `n_pd == apd.n[t, i, 1]`.

⚠ **`n_dur_na` counts individually-reported contacts with NO duration** (~4 250 over this span).
`duration_weight` maps them to level 1 = 2.5 min, numerically indistinguishable from a group
contact, so they must be counted rather than assumed away.
"""
function contact_duration_shares(cfg::FrameworkConfig; grid = cis_age_grid(),
                                 origins::AbstractVector{Date}, raw = nothing)
    isempty(origins) && return (; bins = DataFrame(), blocks = DataFrame())
    raw === nothing && (raw = load_raw_contact_inputs())
    A = grid.N
    blk, blocks, wts = _age_blocks(cfg, grid)                   # SHARED with §4
    weeks = collect(origins)
    wkset = Dict(w => t for (t, w) in enumerate(weeks))
    dmin, dmax = first(weeks), last(weeks) + Day(6)

    rng = MersenneTwister(cfg.seed)
    df_part = @subset(raw.df_part, (:date .>= dmin) .& (:date .<= dmax))
    piv = parse_age_interval.(df_part.part_age)
    df_part[!, :part_bin] = [assign_age_bin(a, b, grid, rng) for (a, b) in piv]
    df_part = @subset(df_part, .!isnothing.(:part_bin))
    df_part[!, :part_bin] = Int.(df_part.part_bin)
    df_part[!, :wk] = week_start.(df_part.date)
    df_part = @subset(df_part, in.(:wk, Ref(Set(weeks))))
    roster = unique(@select(df_part, :part_id_d, :date, :part_bin, :wk))

    dfC = @select(@subset(raw.craw, (:date .>= dmin) .& (:date .<= dmax)),
                  :part_id_d      = :part_wave_uid,
                  :date, :cnt_home, :cnt_mass,
                  :duration_multi = _uk_duration_multi.(:cnt_minutes_max, :cnt_total_time))
    standardise_cnt_home_values!(dfC)                            # on THE COPY, never on raw.craw
    @assert Set(unique(dfC.cnt_home)) ⊆ Set(["true", "false"]) "cnt_home did not standardise to \
        {true,false} — the `.== \"true\"` predicate below would silently report a 0% home share"
    dfC = innerjoin(dfC, roster, on = [:part_id_d, :date])
    grp = _is_group_contact.(dfC.cnt_mass)
    dfC[!, :grp] = grp
    dfC[!, :na]  = .!grp .& _is_dur_na.(dfC.duration_multi)      # individually-reported, no duration
    dfC[!, :min] = ifelse.(grp, cfg.w_dur_group,
                           duration_weight.(dfC.duration_multi, cfg.d_max)) .* cfg.d_max
    dfC[!, :setting] = ifelse.(dfC.cnt_home .== "true", "home", "non-home")
    dfC[!, :t] = getindex.(Ref(wkset), dfC.wk)

    npd = zeros(Int, length(weeks), A)
    for r in eachrow(roster); npd[wkset[r.wk], r.part_bin] += 1; end

    SET = ("home", "non-home")
    minutes = zeros(length(weeks), A, 2); ncnt = zeros(Int, length(weeks), A, 2)
    ngrp = zeros(Int, length(weeks), A, 2); nna = zeros(Int, length(weeks), A, 2)
    for r in eachrow(dfC)
        s = r.setting == "home" ? 1 : 2
        minutes[r.t, r.part_bin, s] += r.min
        ncnt[r.t, r.part_bin, s]    += 1
        r.grp && (ngrp[r.t, r.part_bin, s] += 1)
        r.na  && (nna[r.t, r.part_bin, s]  += 1)
    end

    binrows = NamedTuple[]; blockrows = NamedTuple[]
    for (t, o) in enumerate(weeks)
        for i in 1:A
            tot = sum(minutes[t, i, :])
            for (s, nm) in enumerate(SET)
                push!(binrows, (; origin = o, week = week_mid(o), bin = i, bin_label = grid.LAB[i],
                                  block = blk[i], setting = nm,
                                  min_per_pd = npd[t, i] == 0 ? NaN : minutes[t, i, s] / npd[t, i],
                                  share = tot == 0 ? NaN : minutes[t, i, s] / tot,
                                  share_pooled = tot == 0 ? NaN : minutes[t, i, s] / tot,
                                  minutes = minutes[t, i, s], n_pd = npd[t, i],
                                  n_contacts = ncnt[t, i, s], n_group = ngrp[t, i, s],
                                  n_dur_na = nna[t, i, s]))
            end
        end
        for (b, sym, lab) in blocks
            idx = [i for i in 1:A if blk[i] == b && npd[t, i] > 0]  # re-normalise over bins PRESENT
            w = isempty(idx) ? Float64[] : wts[idx] ./ sum(wts[idx])
            lev = [isempty(idx) ? NaN : sum(w .* [minutes[t, i, s] / npd[t, i] for i in idx])
                   for s in 1:2]
            bidx = [i for i in 1:A if blk[i] == b]
            pool = [sum(minutes[t, bidx, s]) for s in 1:2]
            for (s, nm) in enumerate(SET)
                push!(blockrows, (; origin = o, week = week_mid(o), block = b,
                                    block_label = String(sym), block_desc = lab, setting = nm,
                                    min_per_pd = lev[s],
                                    share = sum(lev) == 0 ? NaN : lev[s] / sum(lev),
                                    share_pooled = sum(pool) == 0 ? NaN : pool[s] / sum(pool),
                                    minutes = pool[s], n_pd = sum(npd[t, bidx]),
                                    n_contacts = sum(ncnt[t, bidx, s]),
                                    n_group = sum(ngrp[t, bidx, s]),
                                    n_dur_na = sum(nna[t, bidx, s]),
                                    n_bins_present = length(idx)))
            end
        end
    end
    return (; bins = DataFrame(binrows), blocks = DataFrame(blockrows))
end

##########################################################################
# §G — the expected age distribution: Perron shares and inc2prev incidence
##########################################################################

"""
    incidence_age_shares(cfg; grid, origins, inf) -> DataFrame

§6's observed comparator: the age distribution of **incidence** per week, from the inc2prev
age-stratified estimates. One row per `(origin, week, bin, bin_label, infections, share)`, where
`share` is that week's infections in the bin over the week's total across all bins.

This is the ONLY place 14j touches the infection series at all, and nothing in the repo normalised
infections into an age proportion before — every existing use age-sums (`plot_forecast_panels`) or
slices one bin (`forecast_ci_age_panel`).

Reuses `weekly_infections` (`infection_data.jl`) untouched: unlike `load_window_data` it takes an
ARBITRARY `Vector{Date}` of Sunday week-starts, so the whole 63-origin span is ONE call over ONE CSV
read — the same "one pass, not 63 windows" rule `observed_contact_means` follows for contacts. It
returns absolute weekly counts (daily per-capita `mean` × the row's `population`, bucketed by
`week_start`) and matches ages by LABEL, so the CSV's alphabetical `variable` ordering is irrelevant.

⚠ **`weeks` MUST BE SUNDAY-ALIGNED.** The lookup is `haskey(wkset, week_start(d))`, so a vector of
Wednesday `week_mid` labels matches nothing and returns **all zeros, silently**. Build from
`origins`; apply `week_mid` only at plot time.

⚠ **`weekly_infections` ZERO-FILLS unmatched weeks** into a pre-zeroed array (the trap flagged in
`degree_agepair.jl`: the estimates end 2022-03-26), so a week outside the data normalises to
`0/0 = NaN`. The assertion below is the guard.

Point estimate from `mean` only, no band. `I_sd` is an explicit independence approximation across
the week; the CSV's `q5`/`q95` are *daily per-capita* quantiles, and summing quantiles is not the
quantile of a sum — nor does a ratio-of-sums have a closed form from marginal SDs. No consumer in
this repo has ever put a band on observed infections.

⚠ **ENGLAND vs UK.** inc2prev infections are **England**; the CoMix contact side is **UK**. Nothing
to fix in code, but any figure putting the two on one axis must say so.
"""
function incidence_age_shares(cfg::FrameworkConfig; grid = cis_age_grid(),
                              origins::AbstractVector{Date}, inf = nothing)
    isempty(origins) && return DataFrame()
    inf === nothing && (inf = load_raw_infection_inputs())
    weeks = collect(origins)
    I_mean, _ = weekly_infections(inf.df, weeks, grid)           # A × T absolute weekly counts
    tot = vec(sum(I_mean; dims = 1))
    @assert all(>(0), tot) "weekly_infections returned an all-zero column — those weeks are outside \
        the inc2prev span and were zero-filled silently; the share would be 0/0 = NaN"
    rows = NamedTuple[]
    for (t, o) in enumerate(weeks), i in 1:grid.N
        push!(rows, (; origin = o, week = week_mid(o), bin = i, bin_label = grid.LAB[i],
                       infections = I_mean[i, t], share = I_mean[i, t] / tot[t]))
    end
    return DataFrame(rows)
end

##########################################################################
# §H — the three round-5 figures
##########################################################################

"""
    _model_cols(n) -> Vector

`n` qualitative colours for `n` MODELS, from one place.

§4's transposed figure, §6's per-phase figures and §7's reproduction panels all colour by model (or
by the quantity that maps 1:1 onto a model), and the same model must get the same colour in all of
them — a reader carrying the key from one publication figure to the next is the whole reason those
figures sit in one notebook. Spelled out at each call site, that consistency drifts the first time
one of them is edited.

⚠ **Take the FIRST `n` entries, do not RESAMPLE.** `palette(:Dark2_8, 3)` interpolates across the
whole scheme and lands on its last entry — Dark2's **grey** `RGB(0.4,0.4,0.4)` — for the third model.
§7 draws two BLACK reference lines (the inc2prev national R, and the model-free observed step) in the
same panels as the model curves, so a grey model is the one colour in the scheme that must not be
chosen. The literal first three are teal / orange / purple, all well clear of black and of each
other. Resampling is only the fallback past the scheme's 8 colours.
"""
_model_cols(n::Integer) = (p = collect(palette(:Dark2_8));
                           n <= length(p) ? p[1:n] : collect(palette(:Dark2_8, n)))

"""
    make_contact_quantities_fig(tl, cfg; grid, res_dir) -> fig

§4's block figure TRANSPOSED: one panel per age block, all three `_TL_SPECS` series overlaid inside
it, so the reader compares what the MEAN-NGM and the NEIGHBOURHOOD-NGM are fed *within* one block
rather than across panels. Same frame, same draws, no new chain reads.

**Fitted only, no observed `×`.** Two reasons: the comparison here is fitted-vs-fitted (what each
builder consumes), and `_timeline_panel!`'s `obs_clip_q` is a PANEL-WIDE quantile — well defined
when a panel holds one quantity, meaningless when it holds three on different scales. The
fitted-vs-observed comparison is the job of the two existing §4 figures.

**Raw values on a shared linear axis, not indexed.** Measured over all 63 origins: the child panel
spans 2.65–15.07 and the adult panel 0.89–7.65 across all three series — factors of ~6 and ~8.6, so
every series is legible unscaled. Indexing to a base week would also destroy the point, which is
that the weighted mean sits ~3× below the count mean; that gap IS the duration weighting.

⚠ The three series carry three different units (contacts, duration-weighted contacts, excess
duration-weighted contacts) on one axis. State it in the caption; it is a deliberate comparison of
magnitudes, not an accident.

⚠ The series key is `(degree, quantity)`, so a transient `:series` column is synthesised HERE and
never stored: `degree` must stay a real `degree_label` (CLAUDE.md's "never sniff a model property
out of a label" — the same rule that made §4's excess a `quantity` and not a pseudo-label).
"""
function make_contact_quantities_fig(tl::DataFrame, cfg::FrameworkConfig; grid = cis_age_grid(),
                                     res_dir::AbstractString = "../res")
    _, blocks, _ = _age_blocks(cfg, grid)
    keys_ = [string(d, "|", q) for (d, q, _, _, _) in _TL_SPECS]
    labels = [string(d, " — ", q == "excess" ? "excess degree (⟨k²⟩/⟨k⟩)·g" :
                                d == "unweighted-negbin" ? "⟨k⟩" : "(1−p⁰)·μ")
              for (d, q, _, _, _) in _TL_SPECS]
    cols = _model_cols(length(_TL_SPECS))
    df = copy(tl)
    df[!, :series] = string.(df.degree, "|", df.quantity)
    df = @subset(df, in.(:series, Ref(Set(keys_))))

    panels = Any[]
    npanel = length(blocks)
    for (pi, (b, sym, lab)) in enumerate(blocks)
        sub = @subset(df, :block .== b)
        if nrow(sub) == 0
            push!(panels, _timeline_empty_panel(String(sym), lab)); continue
        end
        pnl = plot(; xlabel = "forecast origin (Wed mid-date)",
                     ylabel = "per participant-day", title = lab, titlefontsize = 9,
                     xrotation = 45, legend = pi == 1 ? :topleft : false, legendfontsize = 7)
        # `obs = nothing` is handled by `_timeline_panel!` (its `obs === nothing && continue` and the
        # `obs !== nothing` guard on the clip block), so the helper is reused untouched.
        _timeline_panel!(pnl, sub, nothing, :series, keys_, labels, cols;
                         showlegend = pi == 1, phase_labels = pi == npanel)
        push!(panels, pnl)
    end
    fig = plot(panels...; layout = (npanel, 1), size = (1100, 430 * npanel),
               left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, top_margin = 5Plots.mm,
               plot_title = "Fitted contact rate per age block — the three Stage-1 quantities " *
                            "side by side (lines = median, ribbon = 90%); bands = UK COVID phases",
               plot_titlefontsize = 9)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, "14j_contact_quantities_by_block.png"))
    return fig
end

"""
    make_duration_share_fig(ds, cfg; grid, res_dir) -> fig

§5. Share of reported contact DURATION at home vs away, weekly, one panel per age block.

The two lines sum to 1 by construction, so the figure is read as "how much of a typical
participant-day's contact time was spent at home". y is fixed to `[0, 1]` with headroom above for
the rotated phase names — `shade_periods!` writes them at 93 % of the y-range and would otherwise
put them on top of the series.

⚠ `shade_periods!` is called LAST, as its docstring requires: a leading numeric overlay collapses
the Date axis.
"""
function make_duration_share_fig(ds::DataFrame, cfg::FrameworkConfig; grid = cis_age_grid(),
                                 res_dir::AbstractString = "../res")
    _, blocks, _ = _age_blocks(cfg, grid)
    SET = ("home", "non-home"); COLS = (:seagreen, :indianred)
    panels = Any[]
    npanel = length(blocks)
    for (pi, (b, sym, lab)) in enumerate(blocks)
        sub = @subset(ds, :block .== b)
        pnl = plot(; xlabel = "week (Wed mid-date)", ylabel = "share of contact duration",
                     title = lab, titlefontsize = 9, xrotation = 45,
                     legend = pi == 1 ? :topleft : false, legendfontsize = 7)
        dmin, dmax = extrema(sub.week)
        for (s, c) in zip(SET, COLS)
            sf = sort(@subset(sub, :setting .== s), :week)
            plot!(pnl, sf.week, sf.share; color = c, lw = 2,
                  label = pi == 1 ? s : "", marker = :circle, ms = 2)
        end
        ylims!(pnl, (0.0, 1.25))                    # headroom for the rotated phase names
        yticks!(pnl, 0.0:0.2:1.0)
        shade_periods!(pnl, dmin, dmax; alpha = 0.10, fontsize = 5, labels = pi == npanel)
        push!(panels, pnl)
    end
    fig = plot(panels...; layout = (npanel, 1), size = (1100, 400 * npanel),
               left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, top_margin = 5Plots.mm,
               plot_title = "Share of reported contact DURATION at home vs away, by age block — " *
                            "raw CoMix UK, population-weighted within block; bands = UK COVID phases",
               plot_titlefontsize = 9)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, "14j_duration_share.png"))
    return fig
end

"Draw one age-distribution panel: thin per-week lines coloured by phase + a thick per-phase median."
function _age_dist_panel!(pnl, sub::AbstractDataFrame, grid, pcols, phases; showlegend::Bool)
    for r in groupby(sort(sub, [:origin, :bin]), :origin)
        p = _period_group(first(r.origin))
        plot!(pnl, r.bin, r.med; color = pcols[p], lw = 1, alpha = 0.40,
              ls = p == _OUTSIDE_PHASES ? :dash : :solid, label = "")
    end
    for p in phases                                    # thick medians ON TOP of the thin lines
        sp = @subset(sub, _period_group.(:origin) .== p)
        nrow(sp) == 0 && continue
        m = combine(groupby(sp, :bin), :med => median => :m)
        sort!(m, :bin)
        plot!(pnl, m.bin, m.m; color = pcols[p], lw = 3,
              ls = p == _OUTSIDE_PHASES ? :dash : :solid,
              label = showlegend ? p : "")
    end
    plot!(pnl, 1:grid.N, grid.PROP; color = :black, lw = 2, ls = :dot,
          label = showlegend ? "population share" : "")
    return pnl
end

"""
    make_age_distribution_fig(eig; inc, cfg, grid, res_dir) -> fig

§6. The **expected age distribution** over time: the dominant right (Perron) eigenvector of each
week's estimated 7×7 contact matrix `C*`, normalised to sum 1 — one thin line per week coloured by
its UK COVID phase, plus a thick per-phase median. One panel per `_TL_SPECS` quantity, and, when
`inc` is given, a further panel drawing the inc2prev **age distribution of incidence** exactly the
same way, so the observed distribution and the three contact-implied ones are read off one axis.

⚠ **This is NOT the model's predicted case age-distribution.** `C*` alone drops γ_SAR,
susceptibility, infectivity, `F` and antibody — every one of which is age-varying. It is the
contact-only analogue of `contact_reproduction_draws`' `ρ(C*)`. See `_perron_share`.

⚠ **The incidence panel is ENGLAND (inc2prev); the contact panels are UK (CoMix).**

⚠ **No `shade_periods!` here** — the x-axis is a numeric age bin, not a Date. `shade_periods!`
`vspan!`s with Dates and reads `Plots.ylims`; on this axis it would throw or corrupt the panel.
The phase lives in the line colour instead, which is why `_period_cols` exists.

Panel count is DERIVED (`length(_TL_SPECS)` + the incidence panel), never hard-coded, and all
panels share one y-limit — that comparison is the whole figure.
"""
function make_age_distribution_fig(eig::DataFrame; inc = nothing, grid = cis_age_grid(),
                                   res_dir::AbstractString = "../res",
                                   specs = [(d, q, t) for (d, q, _, t, _) in _TL_SPECS],
                                   seriescol::Symbol = :quantity,
                                   fname::AbstractString = "14j_age_distribution.png",
                                   ptitle::AbstractString =
                                       "Expected age distribution: dominant eigenvector of C* " *
                                       "(thin = each week, thick = phase median) vs observed incidence")
    pcols = _period_cols()
    allo = unique(vcat(eig.origin, inc === nothing ? Date[] : inc.origin))
    phases = _periods_present(allo)
    ymax = maximum(vcat(eig.med, inc === nothing ? Float64[] : inc.share, grid.PROP))
    # HEADROOM + a 3-COLUMN legend. There are `length(PERIOD_ORDER)` + 2 entries, and a single-column
    # legend box is tall enough to cover the two CHILD bins of the first panel outright — every series
    # starts at the left edge, so a default `:topleft` box hides exactly the part of the figure that
    # carries the signal, silently. Opening the axis and laying the legend out wide-and-short keeps it
    # clear of the data; same idea as `_timeline_panel!`'s 1.25× before `shade_periods!`.
    ytop = 1.42 * ymax

    panels = Any[]
    npanel = length(specs) + (inc === nothing ? 0 : 1)
    for (pi, (deg, qty, ttl)) in enumerate(specs)
        sub = @subset(eig, :degree .== deg, cols(seriescol) .== qty)
        if nrow(sub) == 0
            push!(panels, _timeline_empty_panel(deg, ttl)); continue
        end
        pnl = plot(; xlabel = "CIS age bin", ylabel = "share of the stable distribution",
                     title = ttl, titlefontsize = 8,
                     xticks = (1:grid.N, grid.LAB), xrotation = 45, ylims = (0.0, ytop),
                     legend = pi == 1 ? :topleft : false, legendfontsize = 6, legend_columns = 3)
        _age_dist_panel!(pnl, sub, grid, pcols, phases; showlegend = pi == 1)
        push!(panels, pnl)
    end
    if inc !== nothing
        sub = rename(select(inc, [:origin, :bin, :share]), :share => :med)
        pnl = plot(; xlabel = "CIS age bin", ylabel = "share of weekly infections",
                     title = "OBSERVED — inc2prev age-stratified incidence (England), " *
                             "weekly infections ÷ that week's total",
                     titlefontsize = 8, xticks = (1:grid.N, grid.LAB), xrotation = 45,
                     ylims = (0.0, ytop), legend = false)
        _age_dist_panel!(pnl, sub, grid, pcols, phases; showlegend = false)
        push!(panels, pnl)
    end
    fig = plot(panels...; layout = (npanel, 1), size = (1000, 380 * npanel),
               left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, top_margin = 5Plots.mm,
               plot_title = ptitle, plot_titlefontsize = 9)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, fname))
    return fig
end

##########################################################################
# §I — the NGM version of §6: eigenvector of N, not of the bare C*
##########################################################################

"""
    ngm_antibody_columns(cfg; grid, origins, h, inf) -> A × length(origins)

Antibody prevalence at each origin's horizon-`h` FORECAST TARGET week — column `t` is byte-identical
to `load_window_data(WeeklyWindow(origins[t], …), …).antibody_fc[:, h]`, which is the column
`reproduction_draws` feeds `build_ngm`.

ONE call, not 63. `weekly_antibody` takes an arbitrary `Vector{Date}` of Sunday week-starts (only
`load_window_data` is window-scoped), so `[o + Day(7h) for o in origins]` gets the whole grid in
**13 ms** off a single `load_raw_infection_inputs()`. Sixty-three `load_window_data` calls cost ~2 s
and additionally build a 12-week `antibody` matrix and an `I_mean` that this figure does not use —
and fire `weekly_antibody`'s unmatched-week `@warn` at 126 sites instead of one. Same "one pass, not
63 windows" rule `incidence_age_shares` and `observed_contact_means` already follow.

⚠ `weekly_antibody` ZERO-FILLS weeks it cannot match, and a zero antibody column means FULL
susceptibility rather than missing data — so the zero-column assertion below is the guard, not a
formality. (Measured: 0 unmatched over the 63 h=1 target weeks, which end 2022-01-02, inside the
inc2prev span 2020-07-31…2022-03-26.)
"""
function ngm_antibody_columns(cfg::FrameworkConfig; grid = cis_age_grid(),
                              origins::AbstractVector{Date}, h::Integer = 1, inf = nothing)
    inf === nothing && (inf = load_raw_infection_inputs())
    wks = [o + Day(7 * h) for o in origins]
    AB = weekly_antibody(inf.df, inf.tmap, wks, grid; what = "NGM target")
    @assert !any(t -> all(iszero, view(AB, :, t)), axes(AB, 2)) "weekly_antibody zero-filled a \
        target week — that reads downstream as FULL susceptibility, not as missing data"
    return AB
end

"""
    _stratified_draws(post_index, n_post, n_per_block) -> Vector{Int}

Draw indices covering EVERY Stage-1 block, `n_per_block` from each.

⚠ **Never subsample the pooled draws by position.** `pooled` holds `n_post` (=100) consecutive
blocks of `n_draw` (=100) draws, and `Cstar_end` varies ONLY across blocks — so a contiguous prefix
`1:1000` carries just **10 of the 100** contact matrices, and a coarse stride `1:1000:10_000` also
only 10, while looking like a 1000- and a 10-draw sample of the posterior respectively. (Measured:
`1:100:10_000` happens to be fine — it lands on the first draw of every block — but that is an
accident of `n_draw == n_post`, not a property to rely on.) Stratifying on `post_index` is the fix,
and `collect_ngm_age_eigen` asserts the block coverage rather than trusting it.
"""
function _stratified_draws(post_index::AbstractVector{<:Integer}, n_post::Integer,
                           n_per_block::Integer)
    idx = Int[]
    for m in 1:n_post
        blk = findall(==(m), post_index)
        isempty(blk) && continue
        k = min(n_per_block, length(blk))
        # `range(a, b; length = 1)` THROWS when a != b ("endpoints differ"), so k == 1 needs its own
        # branch — and k == 1 is a real call (the notebook's γ_SAR check wants one draw per block).
        append!(idx, k <= 1 ? blk[1:1] : blk[round.(Int, range(1, length(blk); length = k))])
    end
    return idx
end

"The 2×2-grid combos whose C* matches each `_TL_SPECS` entry, so every NGM panel has a C* twin."
const _NGM_COMBOS = ((NegBinAgePair(),        MeanNGM()),
                     (HurdleWeibullAgePair(), MeanNGM()),
                     (HurdleWeibullAgePair(), NeighbourhoodDegreeNGM()))

"""
    collect_ngm_age_eigen(cfg; grid, origins, h, combos, contacts, save_dir, n_per_block, inf)
        -> (; eigen, eigen_excluded, n_used)

§6's NGM twin: the dominant right (Perron) eigenvector of the FULL next-generation matrix
`N = γ_SAR · susc_a·(1+(F−1)A_a) · C*_ab · inf_b` per origin, normalised to sum 1 — the stable age
distribution of infections including the age-varying transmission terms that §6's bare-`C*` panels
drop. Rows are keyed `(degree, ngm, origin, bin)` and otherwise share the `C*` frame's schema, so the
same figure functions draw both.

**γ_SAR cannot change the answer.** It is a scalar multiplier applied outermost by `build_ngm`, and
`N v = λ v ⟺ (γN) v = (γλ) v`, so the eigenVECTOR is exactly invariant to it. What separates this
figure from §6's `C*` one is therefore precisely `susc`, `inf`, `F` and the antibody column — nothing
else. (The notebook asserts the invariance on real draws.)

READ-ONLY, and deliberately built like `contact_reproduction_draws`: `stage2_pooled_path` →
`isfile || skip` → `load(path, "pooled")`. It NEVER calls `fit_or_load_stage2` / `two_stage_forecast`,
which fit on miss and write into the live token. ⚠ This does make 14j a **Stage-2 reader**, which it
was not before; the no-refit property is unchanged, but the dependency is new.

⚠ **`Cstar_end[m]` is the contact matrix at week `origin + h`, NOT the origin week** — see the
trailing comment on its assignment in `fit_stage2_pooled` (three docstrings in the repo said
"origin-week" and were wrong until 2026-09-10). §6's `C*` panels read the Stage-1 chain at week
column `cfg.n_fit` = t₀, so at h=1 these panels are ONE WEEK LATER, and from 100 pooled blocks rather
than 2000 chain draws. Say so wherever the two are shown together; they are not a like-for-like
overlay.

⚠ **`Cstar_end` is indexed by the Stage-1 draw `m ∈ 1:n_post`; `susc`/`inf`/`F`/`gamma_sar` by the
pooled draw `d ∈ 1:Np`.** The bridge is `m = post_index[d]`. Getting this wrong silently pairs one
draw's contact matrix with another's transmission parameters.

⚠ **The pooled file cannot be re-guarded by `_p0_draws_valid`** — its `Cstar_end` has already
absorbed whatever `p0f` the Stage-1 chain held. The notebook asserts that every origin here also
passed §6's `C*` guard, which is the same chain.
"""
function collect_ngm_age_eigen(cfg::FrameworkConfig; grid = cis_age_grid(),
                               origins::AbstractVector{Date}, h::Integer = 1,
                               combos = _NGM_COMBOS,
                               contacts::AbstractString = contacts_label(cfg),
                               save_dir::AbstractString = joinpath(@__DIR__, "..",
                                                                   "dt_intermediate"),
                               n_per_block::Integer = 10, inf = nothing)
    isempty(origins) && return (; eigen = DataFrame(), eigen_excluded = DataFrame(), n_used = 0)
    AB = ngm_antibody_columns(cfg; grid = grid, origins = origins, h = h, inf = inf)
    A = grid.N
    eigrows = NamedTuple[]; eigexcl = NamedTuple[]
    n_used = 0
    for (dm, nb) in combos
        lbl = string(degree_label(dm), "|", ngm_label(nb))
        for (t, o) in enumerate(origins)
            path = stage2_pooled_path(lbl, o, h; contacts = contacts, save_dir = save_dir)
            isfile(path) || continue                     # a partial grid renders as a gap
            pooled = try
                load(path, "pooled")
            catch err
                @warn "14j §6: could not load pooled result" path err
                continue
            end
            idx = _stratified_draws(pooled.post_index, pooled.n_post, n_per_block)
            # The null model collapses to ONE C*; guard before indexing `Cstar_end`.
            @assert pooled.n_post >= 1 && length(pooled.Cstar_end) == pooled.n_post
            @assert length(unique(pooled.post_index[idx])) == pooled.n_post "the subsample missed a \
                Stage-1 block — see `_stratified_draws`, a plain stride does exactly this"
            N = Array{Float64,3}(undef, length(idx), A, A)
            ab = view(AB, :, t)
            for (r, d) in enumerate(idx)
                m = pooled.post_index[d]                 # Cstar_end is per-m, the rest per-d
                N[r, :, :] = build_ngm(pooled.Cstar_end[m], pooled.susc[d, :], pooled.inf[d, :],
                                       pooled.F[d], ab; gamma_sar = pooled.gamma_sar[d])
            end
            n_used = length(idx)
            _push_eigen_rows!(eigrows, eigexcl, N,
                              (; degree = degree_label(dm), ngm = ngm_label(nb)), o, grid)
        end
    end
    return (; eigen = DataFrame(eigrows), eigen_excluded = DataFrame(eigexcl), n_used)
end

"""
    make_age_distribution_by_phase_fig(eig; inc, cfg, grid, res_dir, seriescol, specs, fname, ptitle)

§6 TRANSPOSED: one panel per pandemic phase, all sources overlaid inside it — observed incidence in
black, then one line per model. Where `make_age_distribution_fig` colours by phase and panels by
model (so it reads as "how did each model's age distribution move over time?"), this one colours by
model and panels by phase, which is the layout that puts each model directly beside observed
*within* a period.

⚠ **Colour means a different thing in the two figures** — phase there, source here. Say so in the
caption; a reader carrying the other figure's key over to this one will misread every line.

Each line is the MEDIAN across that phase's weeks, with a light ribbon for the interquartile range
across those weeks, so it is visible whether a phase's weeks agree or the median is hiding a spread.
`grid.PROP` is the dashed no-age-structure null. Panel layout comes from `panel_grid`, never a
literal, and the phase list from `_periods_present` — so a phase with no origins simply does not get
a panel, and adding one to `PERIODS` adds one here.

⚠ No `shade_periods!`: the x-axis is a numeric age bin, not a Date.

`seriescol` selects the column identifying a series — `:quantity` for the `C*` frame, `:ngm` for the
NGM one — so this ONE function draws both.
"""
function make_age_distribution_by_phase_fig(eig::DataFrame; inc = nothing, grid = cis_age_grid(),
                                            res_dir::AbstractString = "../res",
                                            seriescol::Symbol = :quantity,
                                            specs = [(d, q, "$d — $q") for (d, q, _, _, _) in _TL_SPECS],
                                            fname::AbstractString = "14j_age_distribution_by_phase.png",
                                            ptitle::AbstractString =
                                                "Expected age distribution by pandemic phase — each " *
                                                "model's C* eigenvector vs observed incidence " *
                                                "(line = phase median, ribbon = IQR across weeks)")
    allo = unique(vcat(eig.origin, inc === nothing ? Date[] : inc.origin))
    phases = _periods_present(allo)
    # NOT `cols` — that would shadow DataFramesMeta's `cols()`, which `@subset` below needs to
    # resolve `seriescol` at runtime. The shadowing is silent and the failure is a MethodError deep
    # inside the macro expansion.
    scols = _model_cols(length(specs))
    ymax = maximum(vcat(eig.med, inc === nothing ? Float64[] : inc.share, grid.PROP))

    # Median + IQR across the weeks of one phase, for one series. Returns `nothing` if that phase has
    # no rows for it (a phase can predate a model's first chain).
    function band(df, valcol, os)
        sub = df[in.(df.origin, Ref(Set(os))), :]
        nrow(sub) == 0 && return nothing
        g = combine(groupby(sub, :bin),
                    valcol => median => :m, valcol => (x -> quantile(x, 0.25)) => :lo,
                    valcol => (x -> quantile(x, 0.75)) => :hi)
        return sort(g, :bin)
    end

    panels = Any[]
    for (pi, ph) in enumerate(phases)
        os = [o for o in allo if _period_group(o) == ph]
        pnl = plot(; xlabel = "CIS age bin", ylabel = "share",
                     title = "$ph  ($(length(os)) week$(length(os) == 1 ? "" : "s"))",
                     titlefontsize = 8, xticks = (1:grid.N, grid.LAB), xrotation = 45,
                     ylims = (0.0, 1.30 * ymax),
                     legend = pi == 1 ? :topleft : false, legendfontsize = 5)
        if inc !== nothing
            b = band(inc, :share, os)
            b === nothing || plot!(pnl, b.bin, b.m; ribbon = (b.m .- b.lo, b.hi .- b.m),
                                   color = :black, lw = 2.5, fillalpha = 0.12,
                                   label = pi == 1 ? "observed incidence" : "")
        end
        for ((deg, key, lab), c) in zip(specs, scols)
            b = band(@subset(eig, :degree .== deg, cols(seriescol) .== key), :med, os)
            b === nothing && continue
            plot!(pnl, b.bin, b.m; ribbon = (b.m .- b.lo, b.hi .- b.m),
                  color = c, lw = 2, fillalpha = 0.12, label = pi == 1 ? lab : "")
        end
        plot!(pnl, 1:grid.N, grid.PROP; color = :black, lw = 1.5, ls = :dot,
              label = pi == 1 ? "population share" : "")
        push!(panels, pnl)
    end
    nr, nc = panel_grid(length(panels))               # DERIVE the layout — never a literal
    # 420 px of row height is not enough: the next row's panel TITLE lands on this row's rotated
    # x-tick labels and its "CIS age bin" axis label. Only visible in the rendered PNG.
    fig = plot(panels...; layout = (nr, nc), size = (420 * nc, 420 * nr),
               left_margin = 8Plots.mm, bottom_margin = 12Plots.mm, top_margin = 6Plots.mm,
               plot_title = ptitle, plot_titlefontsize = 9)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, fname))
    return fig
end

##########################################################################
# §J — §7's reproduction number: the NGM R and the contact-only R
##########################################################################

"""
    observed_contact_rel_or_read(wins, cfg; grid, raw, h, save_dir) -> Vector{Float64}

9j's MODEL-FREE relative contact reproduction number `ρ(Ê_t)/ρ(Ê_ref)` — the Perron root of the RAW
empirical mean-contact matrix per origin, anchored at the first origin — **read** from 9j's cache
when one is there for these exact origins, and computed fresh when not.

⚠ **This exists so that 14j never writes outside `res/`.** The obvious call,
`observed_contact_reproduction_over_time_or_load`, `jldsave`s `9j_obsrt_<token>_h<h>.jld2` on a miss.
That is a derived viz cache and not a grid artefact, so nothing would be corrupted — but "14j writes
only to `res/`" is a cheap invariant to keep and an expensive one to re-establish once broken. The
cache validates on `origins` ALONE (it is model-free), so 9j's existing file is a legitimate hit for
14j's identical 63 origins.

⚠ A miss costs 63 `prepare_degree_data` calls — the slow path. Run 9j first, or accept the wait.

⚠ **Do NOT reach for `reproduction_over_time_or_load` / `relative_contact_reproduction_over_time_or_load`
the same way.** Those two key their filename on `(contacts, h)` but validate the MODEL SET from the
payload, so 14j's three combos would overwrite 9j's six-combo store and start a rebuild ping-pong.
§7 calls their uncached inner functions instead.
"""
function observed_contact_rel_or_read(wins, cfg; grid = cis_age_grid(), raw,
                                      h::Integer = 1,
                                      save_dir::AbstractString = joinpath(@__DIR__, "..",
                                                                          "dt_intermediate"))
    origins = [w.origin for w in wins]
    path = joinpath(save_dir, "9j_obsrt_$(contacts_label(cfg))_h$(h).jld2")
    if isfile(path)
        c = try load(path) catch err; @warn "14j §7: unreadable obs-contact-R cache" path err; nothing end
        if c !== nothing && get(c, "origins", nothing) == origins
            println("§7: reusing 9j's model-free observed contact-R cache (read-only)")
            return c["rel"]
        end
        @warn "14j §7: 9j's observed contact-R cache is for different origins — recomputing (not caching)"
    end
    return observed_contact_reproduction_over_time(wins, cfg; grid = grid, raw = raw, h = h)
end

"""
    reproduction_frame(rt_store, crt_store, labels, origins; h, obs) -> DataFrame

Tidy long table behind §7's figure — one row per `(panel, label, origin)` with `med`/`lo`/`hi`, plus
the model-free observed series and the inc2prev national R averaged onto the same weekly grid.

9j writes no CSV for these figures; 14j puts a table behind every PNG, and this is that table. `panel`
is `"ngm_R"` (absolute, dominant eigenvalue of the frozen NGM) or `"contact_rel_R"` (the
`ρ(C*)/ρ(C*_ref)` ratio) — two different quantities, which is exactly why they are a column and not
two `med` columns side by side.
"""
function reproduction_frame(rt_store, crt_store, labels, origins::AbstractVector{Date};
                            h::Integer = 1, obs = nothing)
    natw = weekly_national_R()
    rows = NamedTuple[]
    for (panel, store) in (("ngm_R", rt_store), ("contact_rel_R", crt_store))
        for lbl in labels, (oi, o) in enumerate(origins)
            cw = o + Day(7 * h)                       # the contact week the quantity describes
            push!(rows, (; panel, label = lbl, origin = o, contact_week = cw,
                           week_mid = week_mid(cw),
                           med = store[lbl].med[oi], lo = store[lbl].lo[oi], hi = store[lbl].hi[oi],
                           national_R = get(natw, week_start(cw), NaN)))
        end
    end
    if obs !== nothing
        for (oi, o) in enumerate(origins)
            cw = o + Day(7 * h)
            push!(rows, (; panel = "contact_rel_R", label = "observed-raw", origin = o,
                           contact_week = cw, week_mid = week_mid(cw),
                           med = obs[oi], lo = NaN, hi = NaN,
                           national_R = get(natw, week_start(cw), NaN)))
        end
    end
    return DataFrame(rows)
end

"""
    make_reproduction_fig(rt_store, crt_store, labels, origins; obs, h, res_dir, national) -> fig

§7. 9j's TWO reproduction figures stacked into one, restricted to §6's three model combos.

- top    — the full NGM R, `ρ(N)` of the NGM frozen at origin+h, over the inc2prev national R
           (England) and the R = 1 threshold.
- bottom — the contact-only R, `ρ(C*_t)/ρ(C*_ref)`, anchored at the first origin, with the model-free
           `ρ(Ê_t)/ρ(Ê_ref)` from the raw empirical contact matrix.

Stacked rather than written as two files because the whole question is vertical: the same three
models, the same colours and one shared x-axis make *"does contact structure alone explain the way R
moved?"* a direct read. That is the contact-degree strand's premise, and §6 answers its
age-composition half.

⚠ **THE BOTTOM PANEL HAS MIXED UNITS, DELIBERATELY** (`national = true`, 9j's default and the user's
choice). The steps are a dimensionless ratio to the baseline contact week; the national R is an
absolute R; and the single 1.0 line is the baseline week for one and the epidemic threshold for the
other at once. 9j's `plot_relative_contact_reproduction` docstring calls that coincidence the
figure's main trap, and it is right: read the SHAPES against each other, never the vertical gap.
`national = false` drops the overlay if a reader cannot be trusted with the caveat.

⚠ **Both panels come from `plot_reproduction` / `plot_relative_contact_reproduction` VERBATIM** —
only the titles are overridden afterwards. Those two carry a lot of earned detail (`:steppost` held
forward from each origin, the 90% ribbons, the "plot Date-bearing series BEFORE `hline!` or the date
ticks mangle" ordering, the rotated-y-label length limit), and hand-reproducing any of it here is how
it rots out of step with 9j.
"""
function make_reproduction_fig(rt_store, crt_store, labels, origins::AbstractVector{Date};
                               obs = nothing, h::Integer = 1,
                               res_dir::AbstractString = "../res", national::Bool = true)
    cols = _model_cols(length(labels))       # SAME colours these models carry in §4 and §6
    p_ngm = plot_reproduction(rt_store, labels, cols, origins; h = h)
    p_rel = plot_relative_contact_reproduction(crt_store, labels, cols, origins;
                                               h = h, observed = obs, national = national)
    plot!(p_ngm; title = "Full NGM — R = ρ(γ_SAR·susc·(1+(F−1)A)·C*·inf), frozen at origin+$h",
                 titlefontsize = 9)
    plot!(p_rel; title = "Contacts only — ρ(C*)/ρ(C*_ref), ref = first origin" *
                         (national ? "   [national R overlaid: MIXED UNITS, read shapes only]" : ""),
                 titlefontsize = 9)
    fig = plot(p_ngm, p_rel; layout = (2, 1), size = (1050, 980),
               left_margin = 10Plots.mm, bottom_margin = 10Plots.mm, top_margin = 5Plots.mm,
               plot_title = "Reproduction number by model (h=$h, step; 90% CI) — " *
                            "transmission included, then contacts alone",
               plot_titlefontsize = 9)
    mkpath(res_dir)
    savefig(fig, joinpath(res_dir, "14j_reproduction_number.png"))
    return fig
end
