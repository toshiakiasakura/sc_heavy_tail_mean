# 11j_viz_utils.jl — weekly identifiability of contactee mean vs neighbourhood-mean degree.
#
# Read-only diagnostic (no refit, no Stage 2). Reconstructs the per-week age-pair raw moments
# (; K1, K2, G) from a cached Stage-1 chain (`8j_s1_*`) via `stage1_moment_draws`, then forms the two
# NGM per-cell C0 functionals per week with 90% CIs across Stage-1 draws:
#   • mean          ⟨k⟩            = K1                             (MeanNGM C0)
#   • neighbourhood ⟨k²⟩/⟨k⟩·g    = contact_star(NeighbourhoodDegreeNGM(), …)  (size-biased/excess)
# Only `constant_contacts=false` (the per-week hierarchical spatio-temporal GP) has weekly
# fluctuation to diagnose — the pooled regime aliases one moment set across all weeks. Comparing the
# weekly wobble/CI of the *neighbourhood* line (driven by the second moment / per-week variance term
# τ_t) against the *mean* line (level only) isolates the contribution of that variance term.
#
# Companion to 11j_weekly_identifiability.ipynb. Requires 8j_viz_utils.jl (`stage1_chain_path`)
# loaded first, and the forecast preamble (`forecast_utils.jl`) for the framework symbols.

using Statistics, Dates

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
                              logy::Bool = false, show_n::Bool = true)
    A   = grid.N
    lbl = string(degree_label(dm), "|", ngm_label(MeanNGM()))     # ngm token irrelevant to Stage 1
    s1p = stage1_chain_path(lbl, origin, h; contacts = contacts_label(cfg))
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
            # Per-cell sample size behind the ribbons, on a secondary right axis: n_pos = number of
            # POSITIVE contacts in cell (i,j) that week = n_roster·(1−p⁰). This is the genuinely
            # per-cell (j-varying) informative count the second moment — hence the neighbourhood CI —
            # is estimated from; thin bars under a wide ribbon = the identifiability being sample-starved.
            npos = [round(Int, apd_h.n[t, i, j] * (1 - apd_h.p0[t, i, j])) for t in 1:length(st.weeks)]
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
