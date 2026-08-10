# 11j_viz_utils.jl — weekly identifiability of contactee mean vs neighbourhood-mean degree,
#                    and the within-block spread of the degree model's dispersion.
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
# (2) DISPERSION SPREAD. Since 2026-08-02 the dispersion is block-linear × week with NO per-cell
#     term, `log d_{ij,t} = β[bl,t]` (§4.3). `plot_within_block_sd` compares that against the
#     retained `-hd` chains, for BOTH degree families — the NegBin dispersion φ and the
#     hurdle-Weibull shape κ (κ is that path's overdispersion parameter: SMALLER κ = heavier tail =
#     MORE dispersion, so the two families read in opposite directions).
#
# ⚠ `stage1_moment_draws` runs `generated_quantities` against the CURRENT `model_degree`, so it can
# only read CURRENT-token chains. Anything comparing cache generations must go through the
# `reconstruct_*` mirrors in 10j_viz_utils.jl, which carry an explicit legacy branch.
#
# Companion to 11j_weekly_identifiability.ipynb. Requires 8j_viz_utils.jl (`stage1_chain_path`) and
# 10j_viz_utils.jl (`reconstruct_dispersion_draws`) loaded first, and
# the forecast preamble (`forecast_utils.jl`) for the framework symbols.

using Statistics, Dates, Random

"""
    moment_timeline_stats(dm, apd_h, pop, cfg, s1chn, grid)
        -> (; weeks, mean=(med,lo,hi), neigh=(med,lo,hi))

Per-week, per age-pair-cell posterior summary of the two NGM contact functionals, reconstructed
read-only from a cached Stage-1 chain `s1chn`. `apd_h` is the degree window this chain was fit on
(`prepare_degree_data(degree_window(origin, h, cfg), …)`, i.e. `[t₀−n_fit+1 … t₀+h]`, length
`n_fit + h` — NOT the shifted `WeeklyWindow(origin + Day(7h))` this used to say); `pop` is the CIS
population vector.

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
    apd_h = prepare_degree_data(degree_window(origin, h, cfg), cfg;
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
            "11j_moment_timeline_$(degree_label(dm))_contactee$(grid.LAB[j])_$(origin).png"))
    return fig
end

# ======================================================================================
# Dispersion of the degree model — is there any within-block structure left?
# ======================================================================================
#
# The per-cell dispersion random effect was REMOVED on 2026-08-02 (see joint_model.jl §4.3), so the
# shrinkage panels that lived here — escape fraction, m_eff, ranked RE, escape-vs-n_pos, and the
# prior-predictive reference — no longer have a quantity to plot: with `log d_{ij,t} = β[bl,t]` there
# is no `τ`, no `λ`, no slab `c`, and no deviation `δ` to shrink. They were deleted rather than left
# returning `nothing`, and are recoverable from commit 7cd11c2.
#
# `plot_within_block_sd` below survives them, and is now the VERIFICATION that the RE really is gone:
# the current generation's within-block SD of log-dispersion must be identically 0.

"""
    cell_npos(apd) -> A×A (or Tn×A×A)

Per-cell count of participant-days with at least one contact, `n·(1−p⁰)` — the sample size that
actually informs a cell's dispersion, as opposed to the roster count `n` (which includes the
all-zero participant-days absorbed by the hurdle). Kept as the natural x-axis for any per-cell
diagnostic scatter.
"""
cell_npos(apd) = apd.n .* (1 .- apd.p0)

"""
    plot_within_block_sd(dm, origin, cfg, grid; h=1, tokens, save_dirs, res_dir)

**Old vs new.** Per week and per child/adult block, the SD of `log(dispersion)` ACROSS the cells of
that block (posterior median + 90%), one line per cache generation — the flat `-hd` hierarchy versus
the current no-random-effect model.

**This is the verification that the per-cell RE is gone.** The current generation's line must be
**identically 0** at every week in every block, because `log d_{ij,t} = β[bl,t]` gives every cell in
a block the same value by construction; the `-hd` line shows the spread the RE used to produce.
Anything non-zero on the current line means a per-cell term survived the 2026-08-02 revert.

It works with no refit because both generations sit on disk and `reconstruct_dispersion_draws` reads
both — this is exactly why that function keeps its legacy branch. (It cannot be done through
`stage1_moment_draws`, which runs `generated_quantities` against the current model only.)

⚠ The two generations differ in **both** token and directory: the `-hd` chains live in
`dt_intermediate_hierarchical/`. `save_dirs` is parallel to `tokens` for that reason — passing the
old token against the current `save_dir` silently finds nothing. A missing generation degrades to a
warning plus that line being omitted.

⚠ The current-generation token comes from `contacts_label(cfg)`, NOT the `CONTACTS_TOKEN` const.
Using the const here once silently dropped the current line from every panel while leaving the `-hd`
line intact, so the figure still rendered with a legend and merely appeared to show no spread at all.
Keep it as `contacts_label(cfg)`.
"""
function plot_within_block_sd(dm::ContactDegreeModel, origin::Date, cfg::FrameworkConfig, grid;
                              h::Integer = 1,
                              tokens = (CONTACTS_TOKEN_HD, contacts_label(cfg)),
                              save_dirs = (CONTACTS_SAVE_DIR_HD,
                                           joinpath(@__DIR__, "..", "dt_intermediate")),
                              res_dir::AbstractString = "../res")
    lbl = string(degree_label(dm), "|", ngm_label(MeanNGM()))
    A = grid.N
    # X-axis extent = the LONGEST window in play, 12 weeks. That is `smax + n_fit` for the legacy
    # `-hd` generation and `n_fit + max(horizons)` for the current one (`-w8h`, 2026-08-09) — equal
    # by coincidence, since smax == max(horizons) == 4. A chain fitted at h < 4 is SHORTER (n_fit+h),
    # so its line simply stops early: the per-token loop below breaks when
    # `reconstruct_dispersion_draws` returns `nothing`, which `_read_disp_chain`'s bounds guard now
    # does for `week_index` past the chain's own Tn (it used to return uninitialised garbage).
    Tn = cfg.smax + cfg.n_fit
    cols = (:grey40, :steelblue)
    nice = ("-hd (flat τ_t·z RE)", "current (block only)")
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
                            "flat hierarchy vs no random effect (origin $origin, h=$h)",
               plot_titlefontsize = 9)
    savefig(fig, joinpath(res_dir, "11j_within_block_sd_$(degree_label(dm))_$(origin).png"))
    return fig
end
