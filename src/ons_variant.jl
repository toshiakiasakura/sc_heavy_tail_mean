# ons_variant.jl — England ONS CIS variant (TaqPath gene-pattern) readers, shared by 16j and 1j.
#
# MOVED VERBATIM from `16j_viz_utils.jl` §E/§F on 2026-10-21 so that 1j can read the ONS SGTF series
# without loading 16j's dependency chain (forecast_utils → 9j → 14j → 15j). Nothing here plots.
#   `ons_sgtf_share`        daily SGTF / S-positive share (16j §2)
#   `ons_variant_mix`       daily share of every `_VARIANT_OF` category (16j §3, 1j)
#   `_ons_variant_releases` the ONE release reader behind both
#
# Requires: CSV, DataFrames, Dates, Statistics (all from `main_utils.jl`) and `_todate_safe` /
# `_tofloat_safe` from `infection_data.jl`. Paths resolve from `@__DIR__`, so the cwd does not matter.
#
# ⚠ NOT AGE-SPECIFIC. ONS publishes no Alpha/Delta proportion by age over time: the CIS variant
# tables are England / nation / region only, and ONS's own variant-by-age analysis (technical
# article, 9 Feb 2023) defines the variant BY PERIOD (Alpha to 17 May 2021, Delta from 18 May).
# Age-split SGTF is public only in PHE technical briefings for Dec 2020 – Jan 2021, outside this
# window. So the share is England-level and the SAME curve is drawn in every age panel.

const _ONS_VAR_EARLY_PATH = joinpath(@__DIR__, "..", "dt_ons_variant", "ons_variant_england_daily.csv")
const _CIS_VARIANTS_PATH  = joinpath(@__DIR__, "..", "inc2prev", "data-processed", "cis_variants.csv")
const _CIS_NATIONAL_PATH  = joinpath(@__DIR__, "..", "inc2prev", "data-processed", "cis.csv")
"ONS's Alpha → Delta period boundary (first Delta day), from its 9 Feb 2023 variant-by-age article."
const ONS_DELTA_PERIOD_START = Date(2021, 5, 18)

# ONS's labels are TaqPath GENE PATTERNS renamed over time; this is the one map onto the two
# classes the SGTF proxy needs. Anything else (Omicron labels, "too low") is not a class here.
const _GENE_CLASS = Dict(
    "uk_variant_compatible"                => "sgtf",
    "alpha_uk_variant_compatible"          => "sgtf",
    "alpha_variant_compatible"             => "sgtf",
    "not_compatible_with_uk_variant"       => "s_pos",
    "not_compatible_with_alpha_uk_variant" => "s_pos",
    "delta_variant_compatible"             => "s_pos",
    "virus_too_low_for_variant_to_be_identifiable" => "undetermined")

"""
    ons_sgtf_share(; dates, early_path, cis_path, nat_path) -> DataFrame

DAILY England share of S-gene-target-failure (SGTF, the Alpha proxy) among gene-pattern-identified
CIS positives, with its S-positive complement:
`date, sgtf, s_pos, share_sgtf, share_spos, publication_date, source, scale_fixed`.

**Sources, one vintage rule.** `inc2prev/data-processed/cis_variants.csv` (technical workbooks
from 2021-05-21) and `dt_ons_variant/ons_variant_england_daily.csv` (the earlier England and
technical releases, from 2020-11-22; see its fetch script). For each date the value is taken from
the LATEST RELEASE THAT CARRIES BOTH `sgtf` AND `s_pos` for it, and both classes come from that one
release — mixing two vintages would mix revisions into one ratio. (Releases rank by publication
date, then technical over England, then version.)

**Why one map works.** Every ONS label is a TaqPath gene pattern: "UK/new/Alpha variant compatible"
= ORF1ab+N without S = SGTF; "not compatible …" and `delta_variant_compatible` = S-positive
(`_GENE_CLASS`). "Too low" is excluded (pattern undetermined); the January-2021 "other" class,
which lumps S-positive with undetermined, carries no `s_pos` and so never supplies a share.

⚠ **S-POSITIVE IS NOT DELTA BEFORE ~LATE APRIL 2021** — it is every non-Alpha lineage. It reads as
Delta only once Alpha and Delta are essentially all infections. The column is `share_spos`, and
every legend says "S-positive", never "Delta".

⚠ **TWO RELEASES OF `cis_variants.csv` ARE 100× TOO SMALL** — publications 2021-06-11 and
2021-06-18 (measured: summed variant positivity / ONS national positivity = 0.0099 and 0.0096,
against 0.91–1.09 for every other release). Each release is scale-tested against
`cis.csv`'s England national positivity exactly as the fetch script tests its own files, fixed by
×100 or ÷100 where it is off by that, and the row's `scale_fixed` says so. The SHARE is unaffected
either way (a ratio within one release), but the absolute `sgtf`/`s_pos` would be.
"""
function ons_sgtf_share(; dates::AbstractVector{Date},
                        early_path::AbstractString = _ONS_VAR_EARLY_PATH,
                        cis_path::AbstractString = _CIS_VARIANTS_PATH,
                        nat_path::AbstractString = _CIS_NATIONAL_PATH)
    pair = Dict(k => v for (k, v) in _GENE_CLASS if v in ("sgtf", "s_pos"))
    rel, scale = _ons_variant_releases(pair; early_path, cis_path, nat_path)
    # Per date: the latest release carrying BOTH classes.
    rows = NamedTuple[]
    for d in dates
        best = nothing
        for (key, cell) in rel
            (haskey(cell, (d, "sgtf")) && haskey(cell, (d, "s_pos"))) || continue
            (best === nothing || key > best) && (best = key)
        end
        best === nothing && continue
        f = scale[best]; a = rel[best][(d, "sgtf")] * f; s = rel[best][(d, "s_pos")] * f
        push!(rows, (date = d, sgtf = a, s_pos = s, share_sgtf = a / (a + s), share_spos = s / (a + s),
                     publication_date = best[1], source = best[4], scale_fixed = f != 1.0))
    end
    out = DataFrame(rows)
    nrow(out) > 0 && @assert all(abs.(out.share_sgtf .+ out.share_spos .- 1) .< 1e-12)
    return out
end

"""
    _ons_variant_releases(classof; early_path, cis_path, nat_path) -> (rel, scale)

The ONE reader behind `ons_sgtf_share` (§2) and `ons_variant_mix` (§3). Every England row of
`dt_ons_variant/ons_variant_england_daily.csv` and `cis_variants.csv` is grouped by RELEASE —
key `(publication_date, is_technical, version, source)`, which sorts so the max is the latest — and
its label mapped through `classof::Dict{String,String}`; unmapped labels are kept only for the
scale test. Returns `rel[key][(date, class)] = value` and `scale[key]`, the factor (1, 100 or 0.01)
that brings each release onto ONS's national positivity.

⚠ **The scale test sums EVERY label of a release**, not just the mapped classes: an Omicron-era
release seen through a pre-Omicron map would otherwise sum to a fraction of positivity and fail.
⚠ Two labels mapping to one class inside one release with different values is an ERROR — it would
mean the map merges two things the release keeps apart.
"""
function _ons_variant_releases(classof::AbstractDict;
                               early_path::AbstractString = _ONS_VAR_EARLY_PATH,
                               cis_path::AbstractString = _CIS_VARIANTS_PATH,
                               nat_path::AbstractString = _CIS_NATIONAL_PATH)
    for p in (early_path, cis_path, nat_path)
        isfile(p) || error("_ons_variant_releases: no such file $p" *
                           (p == early_path ? " — run `python3 -I dt_ons_variant/fetch_ons_variant.py`" : ""))
    end
    # ONS national England positivity, latest vintage per date — the scale reference.
    nat = Dict{Date,Tuple{Date,Float64}}()
    for r in eachrow(CSV.read(nat_path, DataFrame))
        (string(r.level) == "national" && string(r.geography) == "England") || continue
        d = _todate_safe(r.start_date); p = _todate_safe(r.publication_date)
        (d === nothing || p === nothing) && continue
        (haskey(nat, d) && nat[d][1] >= p) && continue
        nat[d] = (p, _tofloat_safe(r.proportion_pos))
    end
    rel    = Dict{Tuple{Date,Bool,Int,String},Dict{Tuple{Date,String},Float64}}()
    alltot = Dict{Tuple{Date,Bool,Int,String},Dict{Date,Float64}}()
    _tot!(key, d, v) = (t = get!(alltot, key, Dict{Date,Float64}()); t[d] = get(t, d, 0.0) + v)
    function _put!(key, d, cls, v)
        cell = get!(rel, key, Dict{Tuple{Date,String},Float64}())
        k = (d, cls)
        haskey(cell, k) && !isapprox(cell[k], v; rtol = 1e-12) &&
            error("_ons_variant_releases: release $(key) has two different `$cls` values for $d — two labels of one class in one release")
        cell[k] = v
    end
    for r in eachrow(CSV.read(early_path, DataFrame))
        v   = string(r.version)
        key = (Date(string(r.publication_date)), startswith(v, "tec"), parse(Int, split(v, "_v")[2]),
               "dt_ons_variant:" * v)
        d = Date(string(r.date)); x = Float64(r.proportion_pos)
        _tot!(key, d, x)
        cls = get(classof, string(r.label), nothing)
        cls === nothing || _put!(key, d, cls, x)
    end
    for r in eachrow(CSV.read(cis_path, DataFrame))
        string(r.geography) == "England" || continue
        d = _todate_safe(r.start_date); p = _todate_safe(r.publication_date)
        (d === nothing || p === nothing) && continue
        @assert _todate_safe(r.end_date) == d "_ons_variant_releases: cis_variants row spans $d … $(r.end_date)"
        key = (p, true, 0, "cis_variants"); x = _tofloat_safe(r.proportion_pos)
        _tot!(key, d, x)
        cls = get(classof, string(r.variant), nothing)
        cls === nothing || _put!(key, d, cls, x)
    end
    # Per-release SCALE test, over EVERY label: summed per date / national positivity, median.
    scale = Dict{eltype(keys(rel)),Float64}()
    for key in keys(rel)
        tot = alltot[key]
        ratios = [t / nat[d][2] for (d, t) in tot if haskey(nat, d) && nat[d][2] > 0]
        isempty(ratios) && (scale[key] = 1.0; continue)
        m = median(ratios)
        f = m < 0.05 ? 100.0 : m > 20 ? 0.01 : 1.0
        @assert 0.5 <= m * f <= 1.7 "_ons_variant_releases: release $(key) does not match national positivity at any 100× scale (median ratio $(m))"
        scale[key] = f
    end
    return rel, scale
end

##########################################################################
# §F — variant categories and the daily mix
##########################################################################

# ONS's variant labels → display category. Every category is a TaqPath GENE PATTERN under the name
# ONS gave it in that period; the two "S-positive …" catch-alls are named for what they CONTAIN,
# not relabelled as a single variant:
#   * `not_compatible_with_(alpha_)uk_variant` (to early Jun 2021 releases) is every non-Alpha
#     lineage — wild type, and increasingly Delta from ~May 2021;
#   * `not_compatible_with_omicron_ba_1_variant` and the BA.2 labels are S-POSITIVE in the Omicron
#     era: residual Delta, then BA.2 (BA.2 is S-gene positive, so "BA.2 compatible" includes Delta).
#   * `other_not_new_variant_compatible` (the three Jan-2021 releases) lumps S-positive with
#     too-low-to-identify, and is drawn hatched, never split.
const _VARIANT_OF = Dict(
    "uk_variant_compatible"                    => "alpha",
    "alpha_uk_variant_compatible"              => "alpha",
    "alpha_variant_compatible"                 => "alpha",
    "not_compatible_with_uk_variant"           => "spos_pre_omicron",
    "not_compatible_with_alpha_uk_variant"     => "spos_pre_omicron",
    "delta_variant_compatible"                 => "delta",
    "omicron_variant_compatible"               => "omicron_ba1",
    "omicron_ba_1_variant_compatible"          => "omicron_ba1",
    "not_compatible_with_omicron_ba_1_variant" => "spos_omicron_era",
    "compatible_with_omicron_ba_2_variant"     => "spos_omicron_era",
    "omicron_ba_2_variant_compatible"          => "spos_omicron_era",
    "other_not_new_variant_compatible"         => "lumped_other")

"Stack order (bottom → top) and legend text for `_VARIANT_OF`'s categories."
const _VARIANT_ORDER = [
    ("spos_pre_omicron", "S-positive, non-Alpha: wild type (→ Delta from ~May 2021)", :gray60),
    ("alpha",            "Alpha (SGTF)",                                               :crimson),
    ("delta",            "Delta (S-positive)",                                         :saddlebrown),
    ("omicron_ba1",      "Omicron BA.1 (SGTF) — the Oct–mid-Nov 2021 sliver is\nbackground SGTF among Delta, not Omicron", :darkviolet),
    ("spos_omicron_era", "S-positive, Omicron era: residual Delta → BA.2",             :peru),
    ("lumped_other",     "other — S-positive + undetermined lumped (Jan-2021 releases)", :gray80)]

"ONS's variant PERIODS (technical article, 9 Feb 2023): first day of each, for the boundary lines."
const ONS_VARIANT_PERIODS = [("Alpha", Date(2020, 12, 8)), ("Delta", Date(2021, 5, 18)),
                             ("BA.1", Date(2021, 12, 14))]

"""
    ons_variant_mix(; dates, early_path, cis_path, nat_path) -> DataFrame

DAILY England share of each variant category (`_VARIANT_OF`) among gene-pattern-IDENTIFIED CIS
positives ("too low" excluded): long frame `date, variant, share, n_identified, publication_date,
source, lumped, scale_fixed`, shares summing to 1 per date (asserted).

⚠ **A category is a GENE PATTERN, so its name can run ahead of the virus.** "Omicron BA.1
compatible" = SGTF from the 2021-12-17 release, whose window starts 2021-10-31: the ~2–3% it shows
in Oct–mid-Nov 2021 is the background SGTF rate among Delta, not Omicron (first UK Omicron cases
late Nov 2021). Likewise "S-positive, non-Alpha" is wild type before ~May 2021, not Delta.

Per date, the LATEST release that identifies at least one category for it supplies EVERY category
of that date (one release per day, so no ratio mixes vintages — `ons_sgtf_share`'s rule).
`n_identified` records how many it identified: ONE is legitimate and common — after 2021-07-17 ONS
stops publishing an Alpha row and Delta is 100% of identified positives until the Omicron labels
arrive. Requiring two would silently blank Aug–Oct 2021. Where no release
does, the latest release with Alpha + the Jan-2021 lumped "other" is used and `lumped = true`:
Alpha's share there is of Alpha + (S-positive AND undetermined), so it reads LOW. Dates with
neither (before 2020-11-22, the first ONS SGTF data) are absent, not zero.
"""
function ons_variant_mix(; dates::AbstractVector{Date},
                         early_path::AbstractString = _ONS_VAR_EARLY_PATH,
                         cis_path::AbstractString = _CIS_VARIANTS_PATH,
                         nat_path::AbstractString = _CIS_NATIONAL_PATH)
    rel, scale = _ons_variant_releases(_VARIANT_OF; early_path, cis_path, nat_path)
    cats(cell, d) = [c for (c, _, _) in _VARIANT_ORDER if haskey(cell, (d, c))]
    rows = NamedTuple[]
    for d in dates
        # Prefer the latest release that identifies >= 1 category WITHOUT the lumped class; only if
        # none exists, the latest one carrying the lumped class — and then the lumped class STAYS IN
        # the denominator (dropping it would report Alpha as 100% of a sum it is a minority of).
        best = nothing
        for (key, cell) in rel
            cs = cats(cell, d)
            (!("lumped_other" in cs) && length(cs) >= 1) || continue
            (best === nothing || key > best) && (best = key)
        end
        lumped = false
        if best === nothing
            for (key, cell) in rel
                "lumped_other" in cats(cell, d) || continue
                (best === nothing || key > best) && (best = key)
            end
            lumped = best !== nothing
        end
        best === nothing && continue
        cell = rel[best]; f = scale[best]
        cs = cats(cell, d)
        tot = sum(cell[(d, c)] for c in cs)
        for c in cs
            push!(rows, (date = d, variant = c, share = cell[(d, c)] / tot, n_identified = length(cs),
                         publication_date = best[1],
                         source = best[4], lumped = lumped, scale_fixed = f != 1.0))
        end
    end
    out = DataFrame(rows)
    if nrow(out) > 0
        s = combine(groupby(out, :date), :share => sum => :s)
        @assert all(abs.(s.s .- 1) .< 1e-12) "ons_variant_mix: shares do not sum to 1"
    end
    return out
end
