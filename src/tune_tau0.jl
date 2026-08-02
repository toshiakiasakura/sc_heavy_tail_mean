# tune_tau0.jl — empirical τ₀ tuning for the regularised-horseshoe dispersion RE (§4.3).
#
# WHY THIS EXISTS. τ₀ (the horseshoe's global shrinkage scale) cannot be set analytically here.
# Piironen & Vehtari's `τ₀ = p₀/(D−p₀)·σ/√n` is derived for a LINEAR model with a residual scale σ
# and sample size n; this is a NegBin / hurdle-Weibull likelihood on counts and durations, which has
# neither. A prior-predictive Monte Carlo of `m_eff` (`prior_shrinkage_reference`) is well-defined
# but describes only the PRIOR — and measured at τ₀ = 0.1 the posterior τ came back 7–15 prior SDs
# out, i.e. the likelihood simply overwhelms the prior. So the escape rate has to be MEASURED from
# fitted chains, one τ₀ at a time. That is what this does.
#
# Fits Stage 1 at ONE origin/horizon for each degree family under the current `cfg`, then reports
# every quantity needed to decide whether τ₀ should move, and in which direction. Because
# `contacts_label` now encodes both τ₀ values (`_tau0_tag`), successive tuning steps write distinct
# filenames and stay side by side on disk — no stale-reload risk, and `report_only=true` re-reads any
# previous step without refitting.
#
# Usage (from src/):
#   include("tune_tau0.jl")
#   tune_tau0(FrameworkConfig(constant_contacts = false))                       # fit + report
#   tune_tau0(cfg; report_only = true)                                          # re-read a past step
#   compare_tau0(cfg_a, cfg_b)                                                  # two steps side by side
#
# Requires the forecast preamble plus 8j/10j/11j_viz_utils.jl (see `_tune_requires`).

using Statistics, Printf, Dates, Random

const _TUNE_ORIGIN = Date(2021, 5, 9)   # the pilot origin; every τ₀ step uses it so steps are comparable
const _TUNE_H      = 1

"""
    tau0_report(dm, origin, cfg, grid, h; save_dir) -> NamedTuple | nothing

Read-only summary of one fitted chain's horseshoe behaviour. Everything comes from
`reconstruct_rhs_components` / `_rhs_all_weeks` (10j/11j), so this cannot drift from the model.

Fields: `tau`/`c` posterior triples, `tau_sds` (how many prior SDs the τ posterior median sits from
0 — the "is the prior being outbid?" number), `lam_med`/`lam_max`, `mult_med` (the actual per-cell RE
size in log), `escape_*`, `n_escaped` (cells with median escape > 0.5), `meff` per week,
`wbsd` (within-block SD of log dispersion, the quantity judged "too large"), and `n_signif`
(cells whose δ 90% CI excludes 0 — 33/49 for hweibull at τ₀ = 0.1).
"""
function tau0_report(dm::ContactDegreeModel, origin::Date, cfg::FrameworkConfig, grid, h::Integer;
                     save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    lbl = string(degree_label(dm), "|", ngm_label(MeanNGM()))
    tok = contacts_label(cfg)
    A   = grid.N
    Tn  = cfg.smax + cfg.n_fit
    st = _rhs_all_weeks(dm, origin, cfg, grid; h = h, contacts = tok, save_dir = save_dir)
    st === nothing && return nothing
    r  = reconstruct_rhs_components(lbl, origin, h; weighted = is_weighted(dm), cfg = cfg,
                                    grid = grid, week_index = Tn, contacts = tok, save_dir = save_dir)
    r === nothing && return nothing

    q(v) = (median(v), quantile(v, 0.05), quantile(v, 0.95))
    D = size(r.shrink, 1)

    # λ across every cell × week, straight from the chain (the mirrors only expose one week at a time)
    chn = load(stage1_chain_path(lbl, origin, h; contacts = tok, save_dir = save_dir), "result")
    lam = Float64[]
    for p in 1:A*A, t in 1:Tn
        append!(lam, vec(Array(chn[Symbol("lam[$p, $t]")])))
    end

    escape = [1 - median(view(st.shrink[t], :, i, j)) for t in 1:Tn, i in 1:A, j in 1:A]
    meff   = [median([sum(1 - st.shrink[t][d, i, j] for i in 1:A, j in 1:A) for d in 1:D])
              for t in 1:Tn]

    # within-block SD of log dispersion, pooled over the four blocks at the origin week
    wb = Float64[]
    for bl in 1:4
        cells = [(i, j) for i in 1:A, j in 1:A if 2*(block_of(i, cfg)-1) + block_of(j, cfg) == bl]
        push!(wb, median([std([log(r.disp[d, i, j]) for (i, j) in cells]) for d in 1:D]))
    end

    nsig = count(1:A*A) do p
        i, j = fldmod1(p, A)
        v = view(r.delta, :, i, j)
        quantile(v, 0.05) > 0 || quantile(v, 0.95) < 0
    end

    return (; tau = q(r.tau), c = q(r.c_slab),
              tau_sds = median(r.tau) / disp_tau0_prior(cfg, dm)[2],
              lam_med = median(lam), lam_max = maximum(lam),
              mult_med = median(r.mult),
              escape_med = median(escape), escape_max = maximum(escape),
              n_escaped = count(>(0.5), escape[Tn, :, :]),
              meff = meff, wbsd = wb, n_signif = nsig, ncell = A * A, Tn = Tn)
end

_fmt3(t) = @sprintf("%.4g [%.4g, %.4g]", t...)

"""
    tune_tau0(cfg; origin, h, grid, raw, wd, report_only=false, save_dir) -> Dict

Fit (or re-read) Stage 1 at one origin for BOTH degree families under `cfg`'s τ₀ values, and print
the tuning table. Returns the per-family reports so successive steps can be compared in code.

`report_only = true` skips fitting and only reads what is already on disk — use it to re-print an
earlier step, or after `compare_tau0`.
"""
function tune_tau0(cfg::FrameworkConfig; origin::Date = _TUNE_ORIGIN, h::Integer = _TUNE_H,
                   grid = cis_age_grid(), raw = nothing, wd = nothing,
                   report_only::Bool = false,
                   save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    tok = contacts_label(cfg)
    println("\n", "="^100)
    @printf("τ₀ TUNING STEP — origin %s, h%d, token %s\n", origin, h, tok)
    @printf("   τ₀  negbin = %-8g   hweibull = %-8g   |  slab s=%g ν=%g, local ν=%g\n",
            cfg.disp_re_scale_prior_unweighted[2], cfg.disp_re_scale_prior_weighted[2],
            cfg.disp_rhs_slab_scale, cfg.disp_rhs_slab_df, cfg.disp_rhs_local_df)
    println("="^100)

    if !report_only
        raw === nothing && (raw = load_raw_contact_inputs())
        wd  === nothing && (wd = load_window_data(WeeklyWindow(origin; n_fit = cfg.n_fit,
                                                               smax = cfg.smax,
                                                               horizons = cfg.horizons); grid = grid))
        apd_h = prepare_degree_data(WeeklyWindow(origin + Day(7 * h); n_fit = cfg.n_fit,
                                                 smax = cfg.smax, horizons = cfg.horizons), cfg;
                                    grid = grid, setting = :all,
                                    df_part_raw = raw.df_part, craw_raw = raw.craw)
        for dm in (NegBinAgePair(), HurdleWeibullAgePair())
            ds = build_degree_stats(dm, apd_h, cfg)
            p  = stage1_path(dm, origin, h; contacts = tok, save_dir = save_dir)
            t0 = time()
            fit_or_load_stage1(p, dm, ds, wd.pop, cfg; rng = Random.Xoshiro(cfg.seed))
            @printf("  fitted %-18s %6.1f s  %s\n", degree_label(dm), time() - t0,
                    isfile(p) ? "" : "(MISSING!)")
        end
    end

    out = Dict{String,Any}()
    for dm in (NegBinAgePair(), HurdleWeibullAgePair())
        rep = tau0_report(dm, origin, cfg, grid, h; save_dir = save_dir)
        if rep === nothing
            @warn "no chain to report" model=degree_label(dm) token=tok
            continue
        end
        out[degree_label(dm)] = rep
        sym = is_weighted(dm) ? "κ" : "φ"
        println("\n── ", degree_label(dm), "  (dispersion parameter $sym,  τ₀ = ",
                disp_tau0_prior(cfg, dm)[2], ") ", "─"^28)
        @printf("   τ  posterior      %s      = %.2f × τ₀\n", _fmt3(rep.tau), rep.tau_sds)
        @printf("   c = √c²           %s\n", _fmt3(rep.c))
        @printf("   λ                 median %.4g   max %.4g\n", rep.lam_med, rep.lam_max)
        @printf("   multiplier τ·λ̃    median %.4g          <- the per-cell RE size, in log\n",
                rep.mult_med)
        @printf("   escape 1−shrink   median %.4g   max %.4g   cells>0.5: %d/%d\n",
                rep.escape_med, rep.escape_max, rep.n_escaped, rep.ncell)
        @printf("   m_eff / week      %s\n", join(round.(rep.meff; digits = 2), " "))
        @printf("   within-block SD   %s   (c→c, c→a, a→c, a→a)\n",
                join(round.(rep.wbsd; digits = 3), "  "))
        @printf("   δ 90%% CI excludes 0 in %d/%d cells\n", rep.n_signif, rep.ncell)
    end
    return out
end

"""
    compare_tau0(cfgs...; origin, h, grid, save_dir)

Print one row per (family, τ₀ step) across several already-fitted configs — the table τ₀ is chosen
from. Read-only: every config's chains must already exist (run `tune_tau0` for each first).
"""
function compare_tau0(cfgs::FrameworkConfig...; origin::Date = _TUNE_ORIGIN, h::Integer = _TUNE_H,
                      grid = cis_age_grid(),
                      save_dir::AbstractString = joinpath(@__DIR__, "..", "dt_intermediate"))
    @printf("\n%-18s %8s %22s %20s %10s %10s %9s %9s %8s\n",
            "model", "τ₀", "τ posterior", "c = √c²", "λ med", "τ·λ̃ med", "escape", "m_eff", "sig/49")
    println("─"^128)
    for cfg in cfgs, dm in (NegBinAgePair(), HurdleWeibullAgePair())
        rep = tau0_report(dm, origin, cfg, grid, h; save_dir = save_dir)
        if rep === nothing
            @printf("%-18s %8g   <no chain for token %s>\n",
                    degree_label(dm), disp_tau0_prior(cfg, dm)[2], contacts_label(cfg))
            continue
        end
        @printf("%-18s %8g %22s %20s %10.4g %10.4g %9.4g %9.2f %5d/%d\n",
                degree_label(dm), disp_tau0_prior(cfg, dm)[2],
                _fmt3(rep.tau), _fmt3(rep.c), rep.lam_med, rep.mult_med,
                rep.escape_med, median(rep.meff), rep.n_signif, rep.ncell)
    end
    println()
end
