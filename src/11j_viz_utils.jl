# 11j_viz_utils.jl — weekly identifiability of contactee mean vs neighbourhood-mean degree,
#                    and how the dispersion random term SHRINKS under the regularised horseshoe.
#
# Read-only diagnostic (no refit, no Stage 2). Two families of figure:
#
# (1) MOMENT TIMELINES. Reconstructs the per-week age-pair raw moments (; K1, K2, G) from a cached
#     Stage-1 chain (`8j_s1_*`) via `stage1_moment_draws`, then forms the two NGM per-cell C0
#     functionals per week with 90% CIs across Stage-1 draws:
#       • mean          ⟨k⟩         = K1                                        (MeanNGM C0)
#       • neighbourhood ⟨k²⟩/⟨k⟩·g = contact_star(NeighbourhoodDegreeNGM(), …)  (size-biased/excess)
#     Only `constant_contacts=false` (the per-week spatio-temporal GP) has weekly fluctuation to
#     diagnose — the pooled regime aliases one moment set across all weeks. Comparing the weekly
#     wobble/CI of the *neighbourhood* line (driven by the second moment, hence by the dispersion
#     random term) against the *mean* line (level only) isolates that term's contribution.
#
# (2) SHRINKAGE. Since 2026-08-02 the dispersion RE is a REGULARISED HORSESHOE (§4.3):
#         log d_{ij,t} = β[bl,t] + τ·λ̃_{ij,t}·z_{ij,t},   λ̃² = c²λ²/(c² + τ²λ²)
#     with a window-global τ, a per-cell-per-week local scale λ ~ half-t₃, and a slab c. These panels
#     show WHERE the shrinkage acts, for BOTH degree families — the NegBin dispersion φ and the
#     hurdle-Weibull shape κ (κ is that path's overdispersion parameter: SMALLER κ = heavier tail =
#     MORE dispersion, so the two families read in opposite directions).
#
# ⚠ `stage1_moment_draws` runs `generated_quantities` against the CURRENT `model_degree`, so it can
# only read CURRENT-token chains. Anything comparing cache generations must go through the
# `reconstruct_*` mirrors in 10j_viz_utils.jl, which carry an explicit legacy branch.
#
# Companion to 11j_weekly_identifiability.ipynb. Requires 8j_viz_utils.jl (`stage1_chain_path`) and
# 10j_viz_utils.jl (`reconstruct_rhs_components`, `reconstruct_dispersion_draws`) loaded first, and
# the forecast preamble (`forecast_utils.jl`) for the framework symbols.

using Statistics, Dates, Random

"""
    moment_timeline_stats(dm, apd_h, pop, cfg, s1chn, grid)
        -> (; weeks, mean=(med,lo,hi), neigh=(med,lo,hi))

Per-week, per age-pair-cell posterior summary of the two NGM contact functionals, reconstructed
read-only from a cached Stage-1 chain `s1chn`. `apd_h` is the degree window this chain was fit on
(`prepare_degree_data(WeeklyWindow(origin + Day(7h)), …)`); `pop` is the CIS population vector.

Returns `weeks::Vector{Date}` (length `Tn`) and two `(med, lo, hi)` triples, each a `Tn×A×A` array:
`mean` = ⟨k⟩ = K1 (MeanNGM C0); `neigh` = ⟨k²⟩/⟨k⟩·g (NeighbourhoodDegreeNGM C0). `med` is the
across-draw median, `lo`/`hi` the 5%/95% quantiles (90% CI). Indexed `[t, i, j]` = week, contactor
i, contactee j. No model is refit — `stage1_moment_draws` only runs `generated_quantities`.
"""
function moment_timeline_stats(dm::ContactDegreeModel, apd_h, pop, cfg::FrameworkConfig, s1chn, grid)
    ds  = build_degree_stats(dm, apd_h, cfg)
    md  = stage1_moment_draws(dm, ds, pop, cfg, s1chn; n_post = cfg.n_stage1_post)
    weeks = ds.weeks
    Tn = length(weeks); A = grid.N; D = length(md)

    # Per-draw stacks of the two functionals. Neighbourhood is computed once per (m,t) over the whole
    # 7×7 (contact_star broadcasts base_contact) — not per cell — so the k1>0 guard is applied once.
    meanstack  = Array{Float64}(undef, D, Tn, A, A)
    neighstack = similar(meanstack)
    for m in 1:D, t in 1:Tn
        K1 = md[m].K1[t]; K2 = md[m].K2[t]; G = md[m].G[t]
        meanstack[m, t, :, :]  .= K1
        neighstack[m, t, :, :] .= contact_star(NeighbourhoodDegreeNGM(), K1, K2, G)
    end

    summ(stack) = begin
        med = Array{Float64}(undef, Tn, A, A); lo = similar(med); hi = similar(med)
        for t in 1:Tn, i in 1:A, j in 1:A
            col = view(stack, :, t, i, j)
            med[t, i, j] = median(col)
            lo[t, i, j]  = quantile(col, 0.05)
            hi[t, i, j]  = quantile(col, 0.95)
        end
        (; med, lo, hi)
    end

    return (; weeks, mean = summ(meanstack), neigh = summ(neighstack))
end

"""
    plot_moment_timeline(dm, j, origin, cfg, grid, raw, wd; h=1, res_dir="../res", logy=false)

7-panel figure (one panel per contactor age i=1:A) of the weekly **mean** ⟨k⟩ and
**neighbourhood-mean** ⟨k²⟩/⟨k⟩ contact degree for the fixed **contactee** column `j`, each with a
90% CI ribbon over the Stage-1 posterior, on a Wednesday-mid-date x-axis. Mirrors
`make_mu_timeline_fig` (10j_viz_utils.jl) but plots the true moment functionals (route A:
`stage1_moment_draws` → `contact_star`), not the raw μ draws — which matters for HurdleWeibull where
`K1 = (1−p⁰)·μW ≠ μW`.

Reads the cached Stage-1 chain `8j_s1_<degree>_<contacts>_<origin>_h<h>.jld2`; if it is missing,
emits a warning and returns a single blank panel (graceful, like the other read-only 10j helpers).
Saves `11j_moment_timeline_<degree>_contactee<lab>_<origin>.png` to `res_dir` and returns the figure.
`logy=true` switches to a log10 y-axis (offered because the size-biased neighbourhood mean ≥ mean).
`show_n=true` (default) overlays, on a secondary right axis in each panel, the per-cell sample size
`n_pos = n_roster·(1−p⁰)` (positive contacts in cell (i,j) per week) as faint gray bars — so the CI
width can be read against the informative sample size (thin bars under a wide ribbon ⇒ sample-starved).
"""
function plot_moment_timeline(dm::ContactDegreeModel, j::Integer, origin::Date, cfg::FrameworkConfig,
                              grid, raw, wd; h::Integer = 1, res_dir::AbstractString = "../res",
                              logy::Bool = false, show_n::Bool = true,
                              contacts::AbstractString = contacts_label(cfg))
    A   = grid.N
    lbl = string(degree_label(dm), "|", ngm_label(MeanNGM()))     # ngm token irrelevant to Stage 1
    s1p = stage1_chain_path(lbl, origin, h; contacts = contacts)
    if !isfile(s1p)
        @warn "plot_moment_timeline: no Stage-1 chain — returning empty figure" model=degree_label(dm) origin h path=s1p
        return plot(; framestyle = :none,
                    title = "no chain: $(degree_label(dm)), contactee $(grid.LAB[j]) @ $(origin)")
    end
    s1chn = load(s1p, "result")
    apd_h = prepare_degree_data(WeeklyWindow(origin + Day(7 * h); n_fit = cfg.n_fit, smax = cfg.smax,
                                             horizons = cfg.horizons), cfg;
                                grid = grid, setting = :all,
                                df_part_raw = raw.df_part, craw_raw = raw.craw)
    st     = moment_timeline_stats(dm, apd_h, wd.pop, cfg, s1chn, grid)
    NP     = cell_npos(apd_h)
    xdate  = week_mid.(st.weeks)
    yscale = logy ? :log10 : :identity

    panels = Any[]
    for i in 1:A
        pnl = plot(; title = "contactor $(grid.LAB[i])", titlefontsize = 8, xrotation = 45,
                   xlabel = "week (Wed mid-date)", ylabel = "contacts / participant-day",
                   yscale = yscale, legend = (i == 1 ? :best : false), legendfontsize = 5)
        # Date-valued series FIRST (date-axis gotcha), then the vertical origin rule.
        mmed = st.mean.med[:, i, j];  mlo = st.mean.lo[:, i, j];  mhi = st.mean.hi[:, i, j]
        nmed = st.neigh.med[:, i, j]; nlo = st.neigh.lo[:, i, j]; nhi = st.neigh.hi[:, i, j]
        plot!(pnl, xdate, mmed; ribbon = (mmed .- mlo, mhi .- mmed), color = :steelblue, lw = 2,
              marker = :circle, ms = 2, fillalpha = 0.12, label = "mean ⟨k⟩")
        plot!(pnl, xdate, nmed; ribbon = (nmed .- nlo, nhi .- nmed), color = :darkorange, lw = 2,
              marker = :circle, ms = 2, fillalpha = 0.12, label = "neighbourhood ⟨k^2⟩/⟨k⟩")
        vline!(pnl, [week_mid(origin)]; color = :gray, ls = :dash, lw = 1, label = "")   # origin t₀
        if show_n
            # Per-cell sample size behind the ribbons, on a secondary right axis (see `cell_npos`):
            # thin bars under a wide ribbon = the identifiability being sample-starved.
            npos = [round(Int, NP[t, i, j]) for t in 1:length(st.weeks)]
            ax2  = twinx(pnl)
            bar!(ax2, xdate, npos; color = :gray, fillalpha = 0.15, linewidth = 0, label = "",
                 legend = false, ylabel = "n_pos", yguidefontsize = 6, ytickfontsize = 5,
                 ylims = (0, max(1, maximum(npos)) * 1.05))
        end
        push!(panels, pnl)
    end
    push!(panels, plot(; framestyle = :none))                     # 8th blank cell fills the 2×4 grid

    fig = plot(panels...; layout = (2, 4), size = (1400, 700),
               left_margin = 6Plots.mm, bottom_margin = 12Plots.mm,
               plot_title = "11j — weekly mean vs neighbourhood degree, contactee $(grid.LAB[j]) — " *
                            "$(degree_label(dm)) (origin $(origin); h$(h) chain; 90% CI; " *
                            "gray bars = n_pos = positive contacts/cell/week)",
               plot_titlefontsize = 9)
    savefig(fig, joinpath(res_dir,
            "11j_moment_timeline_$(degree_label(dm))$(_tau_tag(cfg, dm))_contactee$(grid.LAB[j])_$(origin).png"))
    return fig
end

# ======================================================================================
# Shrinkage of the dispersion random term under the regularised horseshoe (§4.3)
# ======================================================================================

"""
    _tau_tag(cfg, dm) -> String

Filename suffix naming this figure's τ₀ tuning step, e.g. `"_tau0.005"`. Without it every step
overwrites the previous one's PNGs and the whole point of a sweep — comparing steps side by side —
is lost. Mirrors what `_tau0_tag` does for the chain filenames.
"""
_tau_tag(cfg::FrameworkConfig, dm::ContactDegreeModel) = string("_tau", disp_tau0_prior(cfg, dm)[2])

"""
    cell_npos(apd) -> Tn × A × A

Per-cell **informative sample size**: `n_pos = n_roster·(1−p⁰)`, the number of participant-days in
cell `(i,j)` that week which recorded at least one contact. This is the count the cell's dispersion
— and hence the second moment `⟨k²⟩` the neighbourhood NGM divides by `⟨k⟩` — is actually estimated
from, so it is the right x-axis for every "is the shrinkage data-driven?" question.

⚠ `apd.n[t,i,j]` is the ROSTER count and is **invariant in `j`** (`null_contact_level`'s docstring
states `apd.n[t,i,j] == n_roster[t,i]`), so all of the `j`-variation here enters through the
empirical zero fraction `(1−p⁰)`. `apd.p0` is the EMPIRICAL zero fraction, not the weighted path's
fitted `p0f`, so this quantity is identical for both degree models — which is what makes it a fair
common x-axis for comparing them.
"""
cell_npos(apd) = apd.n .* (1 .- apd.p0)

"""
    prior_shrinkage_reference(cfg, dm; A=7, nsim=20_000, rng=Random.Xoshiro(1236))
        -> (; shrink_q, meff_q, mult_q)

Prior-predictive distribution of the shrinkage statistics for ONE degree family (τ₀ is per-family
since 2026-08-02, via `disp_tau0_prior`), by drawing `τ ~ N⁺(0,τ₀)`,
`c² ~ InverseGamma(ν/2, νs²/2)` and `λ ~ half-t_df` and pushing them through the *same* formulas the
posterior panels use. Returns `(median, q05, q95)` triples for the per-cell `shrink`, the per-cell
multiplier `τλ̃`, and `m_eff = Σ_{ij}(1 − shrink)` over an `A²`-cell week.

**This is not optional decoration.** `shrink = c²/(c²+τ²λ²)` is a slab-vs-spike fraction, so its
baseline is NOT zero: at the prior medians a typical cell already sits at `shrink ≈ 0.998`, which
leaves `m_eff` with a prior floor of order 1 per 49-cell week. Reporting "m_eff = 3" without this
band would badly overstate how many cells have genuinely escaped. Every `m_eff`/`shrink` panel here
draws it.

⚠ This is the PRIOR only. It cannot be used to *set* τ₀ against an escape target: the fitted
posterior overwhelmed this prior by 7–15 SDs at τ₀=0.1 (measured 2026-08-02), and Piironen &
Vehtari's analytic `τ₀ = p₀/(D−p₀)·σ/√n` does not apply here — it assumes a linear model with a
residual scale σ and sample size n, neither of which a NegBin / hurdle-Weibull likelihood on counts
and durations has. To choose τ₀, MEASURE the realised escape from fitted chains (`src/tune_tau0.jl`).

Cheap — no chain, no data, milliseconds.
"""
function prior_shrinkage_reference(cfg::FrameworkConfig, dm::ContactDegreeModel; A::Int = 7,
                                   nsim::Int = 20_000, rng = Random.Xoshiro(1236))
    t0 = disp_tau0_prior(cfg, dm)
    dτ = truncated(Normal(t0[1], t0[2]); lower = 0)
    dc = InverseGamma(cfg.disp_rhs_slab_df / 2, cfg.disp_rhs_slab_df * cfg.disp_rhs_slab_scale^2 / 2)
    dλ = truncated(TDist(cfg.disp_rhs_local_df); lower = 0)
    ncell = A * A
    shr  = Vector{Float64}(undef, nsim * ncell)
    mul  = similar(shr)
    meff = Vector{Float64}(undef, nsim)
    k = 0
    for s in 1:nsim
        τ = rand(rng, dτ); c = sqrt(rand(rng, dc)); acc = 0.0
        for _ in 1:ncell
            λ  = rand(rng, dλ)
            mt = _rhs_mult(τ, λ, c)                 # the SAME helper the mirrors use
            sh = 1 - (mt / c)^2
            k += 1; shr[k] = sh; mul[k] = mt; acc += 1 - sh
        end
        meff[s] = acc
    end
    q(v) = (median(v), quantile(v, 0.05), quantile(v, 0.95))
    return (; shrink_q = q(shr), meff_q = q(meff), mult_q = q(mul))
end

"""
    _rhs_all_weeks(dm, origin, cfg, grid; h, contacts, save_dir)
        -> (; shrink, delta, mult, disp, tau, c_slab, Tn) | nothing

`reconstruct_rhs_components` over EVERY window week, stacked as `Tn`-vectors of `D×A×A` arrays.
`Tn = cfg.smax + cfg.n_fit` by construction (`WeeklyWindow`), so no data reload is needed.
"""
function _rhs_all_weeks(dm::ContactDegreeModel, origin::Date, cfg::FrameworkConfig, grid;
                        h::Integer = 1, contacts::AbstractString = contacts_label(cfg),
                        save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    lbl = string(degree_label(dm), "|", ngm_label(MeanNGM()))
    Tn  = cfg.smax + cfg.n_fit
    out = [reconstruct_rhs_components(lbl, origin, h; weighted = is_weighted(dm), cfg = cfg,
                                      grid = grid, week_index = t, contacts = contacts,
                                      save_dir = save_dir) for t in 1:Tn]
    any(isnothing, out) && return nothing
    return (; shrink = [o.shrink for o in out], delta = [o.delta for o in out],
              mult = [o.mult for o in out], disp = [o.disp for o in out],
              tau = out[1].tau, c_slab = out[1].c_slab, Tn)
end

_blank(msg) = plot(; framestyle = :none, title = msg, titlefontsize = 9)

"""
    plot_shrinkage_cells(dm, origin, cfg, grid; h=1, week_index=nothing, contacts, save_dir, res_dir)

7×7 heatmap of the median **escape fraction** `1 − shrink` per ordered cell for one week, with the
child/adult block boundary drawn — the same geometry as `plot_dispersion_cells` (10j) so the two can
be read side by side.

Bright = the cell has escaped the global shrinkage into the slab and carries its own dispersion;
dark = it has been pulled onto its block mean. A uniformly dark grid is the horseshoe's DESIGNED
default (the spike shrinks everything), so the signal here is a small number of individually bright
cells — not overall brightness. Contrast with the old flat hierarchy, where within-block variation
was uniform by construction.
"""
function plot_shrinkage_cells(dm::ContactDegreeModel, origin::Date, cfg::FrameworkConfig, grid;
                              h::Integer = 1, week_index::Union{Int,Nothing} = nothing,
                              contacts::AbstractString = contacts_label(cfg),
                              save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                              res_dir::AbstractString = "../res")
    lbl = string(degree_label(dm), "|", ngm_label(MeanNGM()))
    wk  = week_index === nothing ? cfg.smax + cfg.n_fit : week_index
    r = reconstruct_rhs_components(lbl, origin, h; weighted = is_weighted(dm), cfg = cfg,
                                   grid = grid, week_index = wk, contacts = contacts,
                                   save_dir = save_dir)
    r === nothing && return _blank("no horseshoe chain: $(degree_label(dm)) @ $origin")
    A   = grid.N
    esc = [median(view(r.shrink, :, i, j)) for i in 1:A, j in 1:A]
    esc = 1 .- esc
    sym = is_weighted(dm) ? "κ" : "φ"
    p = heatmap(1:A, 1:A, esc; c = :viridis, yflip = true, clims = (0, 1),
                xticks = (1:A, grid.LAB), yticks = (1:A, grid.LAB), xrotation = 45,
                xlabel = "contactee age group j", ylabel = "participant age group i",
                title = "$(degree_label(dm)) — escape fraction 1−shrink for $sym (week $wk, origin $origin)",
                titlefontsize = 9, size = (700, 560), right_margin = 5Plots.mm)
    # child/adult block boundary (bins 1..cfg.child_bins are "child")
    b = cfg.child_bins + 0.5
    vline!(p, [b]; color = :white, lw = 2, label = "")
    hline!(p, [b]; color = :white, lw = 2, label = "")
    savefig(p, joinpath(res_dir, "11j_shrinkage_cells_$(degree_label(dm))$(_tau_tag(cfg, dm))_$(origin)_wk$(wk).png"))
    return p
end

# Fixed y-range for the escape-fraction panels: 1e-4 … 1, decade ticks. Shared by every τ₀ step so
# the figures from a sweep can be laid side by side (user request 2026-08-02).
const _ESC_YLIM   = (1e-4, 1.0)
# ASCII tick labels on purpose: GR's default font has no glyphs for Unicode superscripts, so
# "10⁻⁴" renders as tofu boxes in the PNG. Same reason the title below says "tau0", not "τ₀" —
# the subscript ₀ is missing too, although plain τ/κ/φ/⟨k⟩ render fine.
const _ESC_YTICKS = ([1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                     ["10^-4", "10^-3", "10^-2", "10^-1", "10^0"])

"""
    plot_shrinkage_vs_n(dm, origin, cfg, grid, raw; h=1, contacts, save_dir, res_dir)

**The headline check.** Median escape fraction `1 − shrink` (log y) against the per-cell informative
sample size `n_pos` (log x) over all `A²×Tn` cell-weeks, coloured by child/adult block, with the
prior-predictive median drawn as a dashed rule.

If the shrinkage is data-driven the cloud rises with `n_pos`: sample-starved cells sit at the prior
level (fully shrunk onto their block mean), well-sampled cells escape. **Escape concentrated at LOW
`n_pos` is the warning sign** — the horseshoe letting prior noise through exactly where there is no
data, which then flows into `⟨k²⟩` and gets amplified by the neighbourhood NGM. That is the
pathology 11j exists to detect.

The y-axis is pinned to `1e-4 … 1` with decade ticks across every τ₀ step, so panels from a tuning
sweep are directly comparable; points below the floor are clamped to it and counted in the title.
"""
function plot_shrinkage_vs_n(dm::ContactDegreeModel, origin::Date, cfg::FrameworkConfig, grid, raw;
                             h::Integer = 1, contacts::AbstractString = contacts_label(cfg),
                             save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                             res_dir::AbstractString = "../res")
    st = _rhs_all_weeks(dm, origin, cfg, grid; h = h, contacts = contacts, save_dir = save_dir)
    st === nothing && return _blank("no horseshoe chain: $(degree_label(dm)) @ $origin")
    apd_h = prepare_degree_data(WeeklyWindow(origin + Day(7 * h); n_fit = cfg.n_fit, smax = cfg.smax,
                                             horizons = cfg.horizons), cfg;
                                grid = grid, setting = :all,
                                df_part_raw = raw.df_part, craw_raw = raw.craw)
    NP = cell_npos(apd_h)
    A  = grid.N
    xs = Dict(b => Float64[] for b in 1:4); ys = Dict(b => Float64[] for b in 1:4)
    for t in 1:st.Tn, i in 1:A, j in 1:A
        bl = 2 * (block_of(i, cfg) - 1) + block_of(j, cfg)
        # 0.5 floor so fully-empty cells stay visible on a log axis rather than dropping out
        push!(xs[bl], max(NP[t, i, j], 0.5))
        push!(ys[bl], max(1 - median(view(st.shrink[t], :, i, j)), 1e-6))
    end
    pri = prior_shrinkage_reference(cfg, dm; A = A)
    lab = ["child→child", "child→adult", "adult→child", "adult→adult"]
    col = [:steelblue, :darkorange, :seagreen, :purple]
    sym = is_weighted(dm) ? "κ" : "φ"
    # FIXED y-range 1e-4 … 1 with decade ticks. The escape fraction moves by orders of magnitude as
    # τ₀ is tuned, so an auto-scaled axis silently rescales between steps and makes two figures from
    # a sweep look alike when they are not — pinning the decades is what makes them comparable.
    # Points below the floor are CLAMPED to it (so they stay visible at the bottom edge) and COUNTED
    # in the title rather than silently dropped off-axis.
    nclip = sum(count(<(_ESC_YLIM[1]), ys[b]) for b in 1:4)
    for b in 1:4
        ys[b] = max.(ys[b], _ESC_YLIM[1])
    end
    p = plot(; xscale = :log10, yscale = :log10, legend = :bottomright, legendfontsize = 6,
             xlabel = "n_pos = informative participant-days in cell (log)",
             ylabel = "escape fraction 1 − shrink (log)",
             ylims = _ESC_YLIM, yticks = _ESC_YTICKS,
             title = "$(degree_label(dm)) — $sym escape vs sample size, all $(A*A)×$(st.Tn) cell-weeks " *
                     "(origin $origin, tau0=$(disp_tau0_prior(cfg, dm)[2])" *
                     (nclip > 0 ? "; $nclip pts clamped to the 1e-4 floor)" : ")"),
             titlefontsize = 9, size = (900, 600))
    for b in 1:4
        isempty(xs[b]) && continue
        scatter!(p, xs[b], ys[b]; ms = 3, msw = 0, alpha = 0.55, color = col[b], label = lab[b])
    end
    hline!(p, [max(1 - pri.shrink_q[1], _ESC_YLIM[1])]; color = :grey30, ls = :dash, lw = 1.5,
           label = "prior median escape")
    savefig(p, joinpath(res_dir, "11j_shrinkage_vs_n_$(degree_label(dm))$(_tau_tag(cfg, dm))_$(origin).png"))
    return p
end

"""
    plot_re_ranked(dm, origin, cfg, grid; h=1, week_index=nothing, contacts, save_dir, res_dir)

The 49 ordered cells' RE deviation `δ = τ·λ̃·z` (posterior median + 90%) for one week, **ranked by
|median|**, with the slab ceiling `±c` and the spike level `±median(τλ̃)` drawn as reference rules.

This is the spike-and-slab signature: a working regularised horseshoe shows a **flat floor of cells
pinned near 0 and a handful of spikes reaching toward the ceiling**. A smooth staircase across all 49
means it has degenerated into a plain hierarchical random effect with no selection. The subtitle
reports how many cells' 90% CI excludes 0.
"""
function plot_re_ranked(dm::ContactDegreeModel, origin::Date, cfg::FrameworkConfig, grid;
                        h::Integer = 1, week_index::Union{Int,Nothing} = nothing,
                        contacts::AbstractString = contacts_label(cfg),
                        save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                        res_dir::AbstractString = "../res")
    lbl = string(degree_label(dm), "|", ngm_label(MeanNGM()))
    wk  = week_index === nothing ? cfg.smax + cfg.n_fit : week_index
    r = reconstruct_rhs_components(lbl, origin, h; weighted = is_weighted(dm), cfg = cfg,
                                   grid = grid, week_index = wk, contacts = contacts,
                                   save_dir = save_dir)
    r === nothing && return _blank("no horseshoe chain: $(degree_label(dm)) @ $origin")
    A = grid.N
    med = Float64[]; lo = Float64[]; hi = Float64[]; nm = String[]
    for i in 1:A, j in 1:A
        v = view(r.delta, :, i, j)
        push!(med, median(v)); push!(lo, quantile(v, 0.05)); push!(hi, quantile(v, 0.95))
        push!(nm, "$(grid.LAB[i])→$(grid.LAB[j])")
    end
    ord = sortperm(abs.(med); rev = true)
    med, lo, hi, nm = med[ord], lo[ord], hi[ord], nm[ord]
    nsig  = count(k -> lo[k] > 0 || hi[k] < 0, eachindex(med))
    cmed  = median(r.c_slab)
    spike = median(r.mult)                       # typical τλ̃ across cells & draws
    sym   = is_weighted(dm) ? "κ" : "φ"
    p = plot(; legend = :topright, legendfontsize = 6, xrotation = 90, xtickfontsize = 5,
             xticks = (1:length(med), nm), xlabel = "ordered cell i→j (ranked by |median δ|)",
             ylabel = "δ = log $sym − block mean",
             title = "$(degree_label(dm)) — RE deviation ranked, week $wk (origin $origin); " *
                     "$nsig/$(length(med)) cells' 90% CI excludes 0",
             titlefontsize = 9, size = (1200, 600), bottom_margin = 14Plots.mm)
    plot!(p, 1:length(med), med; yerror = (med .- lo, hi .- med), seriestype = :scatter,
          ms = 3, msw = 0.6, color = :steelblue, label = "median & 90%")
    hline!(p, [0]; color = :black, lw = 1, label = "")
    hline!(p, [cmed, -cmed]; color = :firebrick, ls = :dash, lw = 1.2, label = "slab ceiling ±c")
    hline!(p, [spike, -spike]; color = :grey40, ls = :dot, lw = 1.2, label = "spike ±median(τλ̃)")
    savefig(p, joinpath(res_dir, "11j_re_ranked_$(degree_label(dm))$(_tau_tag(cfg, dm))_$(origin)_wk$(wk).png"))
    return p
end

"""
    plot_meff_over_weeks(dm, origin, cfg, grid; h=1, contacts, save_dir, res_dir)

Effective number of **unshrunk** cells per window week, `m_eff,t = Σ_{ij}(1 − shrink_{ij,t})`, as a
posterior median + 90% band over the grey **prior-predictive** band from
`prior_shrinkage_reference`, against the `A²`-cell ceiling.

The compact "how sparse is the dispersion?" summary. Above the prior band ⇒ the data have genuinely
singled out cells. Inside it ⇒ the horseshoe has collapsed to block-only-plus-noise and is not
earning its place (documented fallback: raise this family's τ₀ via `disp_tau0_prior`).
"""
function plot_meff_over_weeks(dm::ContactDegreeModel, origin::Date, cfg::FrameworkConfig, grid;
                              h::Integer = 1, contacts::AbstractString = contacts_label(cfg),
                              save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"),
                              res_dir::AbstractString = "../res")
    st = _rhs_all_weeks(dm, origin, cfg, grid; h = h, contacts = contacts, save_dir = save_dir)
    st === nothing && return _blank("no horseshoe chain: $(degree_label(dm)) @ $origin")
    A = grid.N; D = size(st.shrink[1], 1)
    med = Float64[]; lo = Float64[]; hi = Float64[]
    for t in 1:st.Tn
        me = [sum(1 - st.shrink[t][d, i, j] for i in 1:A, j in 1:A) for d in 1:D]
        push!(med, median(me)); push!(lo, quantile(me, 0.05)); push!(hi, quantile(me, 0.95))
    end
    pri = prior_shrinkage_reference(cfg, dm; A = A)
    sym = is_weighted(dm) ? "κ" : "φ"
    p = plot(; legend = :topright, legendfontsize = 6,
             xlabel = "window week index (1 = earliest lag week)",
             ylabel = "m_eff = Σ(1 − shrink)",
             title = "$(degree_label(dm)) — effective unshrunk $sym cells per week (origin $origin, h=$h)",
             titlefontsize = 9, size = (900, 500))
    plot!(p, 1:st.Tn, med; ribbon = (med .- lo, hi .- med), lw = 2, marker = :circle, ms = 3,
          markerstrokewidth = 0, color = :steelblue, fillalpha = 0.18, label = "posterior median & 90%")
    plot!(p, 1:st.Tn, fill(pri.meff_q[1], st.Tn);
          ribbon = (fill(pri.meff_q[1] - pri.meff_q[2], st.Tn),
                    fill(pri.meff_q[3] - pri.meff_q[1], st.Tn)),
          lw = 1.2, ls = :dot, color = :grey40, fillalpha = 0.10, fillcolor = :grey60,
          label = "prior-predictive median & 90%")
    hline!(p, [A * A]; color = :firebrick, ls = :dash, lw = 1, label = "ceiling = $(A*A) cells")
    savefig(p, joinpath(res_dir, "11j_meff_$(degree_label(dm))$(_tau_tag(cfg, dm))_$(origin).png"))
    return p
end

"""
    plot_within_block_sd(dm, origin, cfg, grid; h=1, tokens, save_dirs, res_dir)

**Old vs new.** Per week and per child/adult block, the SD of `log(dispersion)` ACROSS the cells of
that block (posterior median + 90%), one line per cache generation — the flat `-hd` hierarchy versus
the regularised horseshoe.

This is the direct measurement of what the horseshoe changed: it should COMPRESS within-block spread
everywhere except in blocks containing a genuinely escaped cell. It works with no refit because both
generations sit on disk and `reconstruct_dispersion_draws` reads both — this is exactly why that
function keeps its legacy branch. (It cannot be done through `stage1_moment_draws`, which runs
`generated_quantities` against the current model only.)

⚠ The two generations differ in **both** token and directory: the `-hd` chains were moved to
`dt_intermediate_hierarchical/`. `save_dirs` is parallel to `tokens` for that reason — passing the
old token against the current `save_dir` silently finds nothing. A missing generation degrades to a
warning plus that line being omitted.

⚠ The current-generation token comes from `contacts_label(cfg)`, NOT the `CONTACTS_TOKEN` const.
Since the token encodes τ₀, the const (built from the DEFAULT `FrameworkConfig` at load time) points
at whatever τ₀ the default happens to carry — which during tuning is not the step the caller pinned.
Using it here silently dropped the `-rhs` line from every panel while leaving the `-hd` line intact,
so the figure looked plausible and merely showed no shrinkage at all.
"""
function plot_within_block_sd(dm::ContactDegreeModel, origin::Date, cfg::FrameworkConfig, grid;
                              h::Integer = 1,
                              tokens = (CONTACTS_TOKEN_HD, contacts_label(cfg)),
                              save_dirs = (CONTACTS_SAVE_DIR_HD,
                                           joinpath(@__DIR__, "..", "dt_intermediate")),
                              res_dir::AbstractString = "../res")
    lbl = string(degree_label(dm), "|", ngm_label(MeanNGM()))
    A   = grid.N; Tn = cfg.smax + cfg.n_fit
    cols = (:grey40, :steelblue)
    nice = ("-hd (flat τ_t)", "-rhs (horseshoe)")
    blab = ["child→child", "child→adult", "adult→child", "adult→adult"]
    panels = Any[]
    for bl in 1:4
        cells = [(i, j) for i in 1:A, j in 1:A if 2 * (block_of(i, cfg) - 1) + block_of(j, cfg) == bl]
        pnl = plot(; title = blab[bl], titlefontsize = 8, xlabel = "window week",
                   ylabel = "SD of log dispersion within block",
                   legend = (bl == 1 ? :best : false), legendfontsize = 5)
        for (k, tok) in enumerate(tokens)
            med = fill(NaN, Tn); lo = fill(NaN, Tn); hi = fill(NaN, Tn); got = false
            for t in 1:Tn
                dd = reconstruct_dispersion_draws(lbl, origin, h; weighted = is_weighted(dm),
                                                  cfg = cfg, grid = grid, week_index = t,
                                                  contacts = tok, save_dir = save_dirs[k])
                dd === nothing && break
                got = true
                D = size(dd, 1)
                s = [std([log(dd[d, i, j]) for (i, j) in cells]) for d in 1:D]
                med[t] = median(s); lo[t] = quantile(s, 0.05); hi[t] = quantile(s, 0.95)
            end
            got || (@warn "plot_within_block_sd: no chain for this generation" lbl tok dir=save_dirs[k];
                    continue)
            plot!(pnl, 1:Tn, med; ribbon = (med .- lo, hi .- med), lw = 2, marker = :circle,
                  ms = 2, markerstrokewidth = 0, color = cols[k], fillalpha = 0.12,
                  label = nice[k])
        end
        push!(panels, pnl)
    end
    sym = is_weighted(dm) ? "κ" : "φ"
    # (2,2) is safe to hard-code here — unlike the per-MODEL figures in 9j, the panel count is
    # STRUCTURALLY 4: `block_of` is binary (child/adult), so bl = 2(bi−1)+bj always spans exactly
    # {1,2,3,4} regardless of `cfg.child_bins` or the number of age bins.
    fig = plot(panels...; layout = (2, 2), size = (1100, 700),
               left_margin = 6Plots.mm, bottom_margin = 8Plots.mm,
               plot_title = "$(degree_label(dm)) — within-block spread of log $sym, " *
                            "flat hierarchy vs regularised horseshoe (origin $origin, h=$h)",
               plot_titlefontsize = 9)
    savefig(fig, joinpath(res_dir, "11j_within_block_sd_$(degree_label(dm))$(_tau_tag(cfg, dm))_$(origin).png"))
    return fig
end
