######################################################
##### BNB utility functions for bi-weekly analysis ###
######################################################

"""
    load_dds(df_dds; strat="non-home", late_from=nothing)

Filter a pre-loaded `df_dds` DataFrame by `strat` (e.g. `"non-home"`, `"home"`, `"all"`)
and optionally restrict to rows whose `:date >= late_from`, then merge into a `DegreeDist`.

# Arguments
- `df_dds`      : DataFrame with columns `:date`, `:key`, `:strat`, `:x`, `:y`
- `strat`       : contact stratum to keep (default `"non-home"`)
- `late_from`   : `Date` lower-bound on `:date`; `nothing` keeps all dates

# Returns
`DegreeDist` of the merged count data.
"""
function load_dds(df_dds::DataFrame;
                  strat::String = "non-home",
                  late_from::Union{Date, Nothing} = nothing)::DegreeDist
    df = @subset(df_dds, :strat .== strat)
    if !isnothing(late_from)
        df = @subset(df, :date .>= late_from)
    end
    return merge_dd(df)
end


"""
    fit_biweekly_zeroinf_bnb2!(df_dds, dir_; strat="non-home", n_sample=1000)

Fit a `ZeroInfBNB2` model to every bi-weekly window in `df_dds`.
Skips windows whose output file already exists (safe to re-run).

# Arguments
- `df_dds`    : long DataFrame with columns `:key` (date string), `:strat`, `:x`, `:y`
- `dir_`      : output directory for JLD2 chain files
- `strat`     : contact stratum to fit (default `"non-home"`)
- `n_sample`  : number of NUTS posterior samples (default `1000`)
"""
function fit_biweekly_zeroinf_bnb2!(df_dds::DataFrame, dir_::String;
                                     strat::String = "non-home",
                                     n_sample::Int = 1000)
    keys_ = sort(unique(df_dds.key))
    for k in keys_
        path = "$(dir_)/$(k)_zeroinf_bnb2_nhm.jld2"
        if isfile(path)
            println("Already fitted, skipping: $k")
            continue
        end
        dd = @subset(df_dds, :key .== k, :strat .== strat) |> DegreeDist
        println("Fitting: $k  (n = $(sum(dd.y)))")
        chn = fit_model_with_forward_mode(model_ZeroInfBNB2(dd), n_sample; progress=false)
        jldsave(path, result=chn)
        println("  ✓ saved → $path")
    end
    println("All waves done.")
end


"""
    extract_biweekly_means(keys_, dir_, df_dds; strat="non-home") -> DataFrame

Load each per-wave chain from `dir_` and return a tidy DataFrame with:
- Posterior median mean and tail index `α = v + 1`, plus 95 % credible intervals
- `mean_raw`   : empirical mean of all observed degrees
- `mean_trunc` : empirical mean with degrees capped at 50 before averaging

Reuses `create_capped_frequencies` from `comix_uk_time_series.jl`.

Columns: `:key`, `:mean`, `:mean_l`, `:mean_u`, `:α`, `:α_l`, `:α_u`,
         `:mean_raw`, `:mean_trunc`
"""
function extract_biweekly_means(keys_::AbstractVector, dir_::String,
                                 df_dds::DataFrame;
                                 strat::String = "non-home")::DataFrame
    rows = []
    for k in keys_
        path = "$(dir_)/$(k)_zeroinf_bnb2_nhm.jld2"
        chn  = load(path, "result")

        dists = get_vec_ZeroInfBNB2_from_chn(chn)
        ms = mean.(dists)
        αs = map(d -> d.v + 1, dists)

        m_l, m_med, m_u = quantile(ms, [0.025, 0.5, 0.975])
        α_l, α_med, α_u = quantile(αs, [0.025, 0.5, 0.975])

        # Empirical means from the observed degree distribution for this wave
        df_k       = @subset(df_dds, :key .== k, :strat .== strat)
        mean_raw   = mean(DegreeDist(df_k))
        mean_trunc = mean(create_capped_frequencies(df_k) |> DegreeDist)

        push!(rows, (key=Date(k), mean=m_med, mean_l=m_l, mean_u=m_u,
                                  α=α_med,    α_l=α_l,    α_u=α_u,
                                  mean_raw=mean_raw, mean_trunc=mean_trunc))
    end
    return sort!(DataFrame(rows), :key)
end


"""
    plot_temporal_mean_alpha(df_means) -> Plot

Two-panel figure:
- (top)    three mean series overlaid on the same axes:
             · ZeroInfBNB2 posterior median with 95 % CI ribbon (blue)
             · empirical raw mean (orange)
             · empirical truncated mean, degrees capped at 50 (green dashed)
- (bottom) tail index α with reference lines at α = 1 and α = 2
"""
function plot_temporal_mean_alpha(df_means::DataFrame)
    pl_mean = plot(
        df_means.key, df_means.mean;
        ribbon    = (df_means.mean .- df_means.mean_l, df_means.mean_u .- df_means.mean),
        fillalpha = 0.2,
        ylabel    = "Mean contacts",
        label     = "ZeroInfBNB2 posterior median",
        title     = "Bi-weekly non-home contacts",
        xrotation = 45,
        marker    = :circle, markersize = 4,
        legend    = :topright,
        color     = :steelblue,
    )
    plot!(pl_mean, df_means.key, df_means.mean_raw;
        label  = "Raw mean",
        marker = :circle, markersize = 4,
        color  = :darkorange,
    )
    plot!(pl_mean, df_means.key, df_means.mean_trunc;
        label     = "Truncated mean (cap 50)",
        marker    = :diamond, markersize = 4,
        color     = :green,
        linestyle = :dash,
    )

    pl_α = plot(
        df_means.key, df_means.α;
        ribbon    = (df_means.α .- df_means.α_l, df_means.α_u .- df_means.α),
        fillalpha = 0.3,
        xlabel    = "Date",
        ylabel    = "α (tail index)",
        label     = "α = v + 1",
        xrotation = 45,
        marker    = :circle, markersize = 4,
        legend    = :topright,
        color     = :crimson,
        ylim      = [0.9, 2.5],
    )
    hline!(pl_α, [1.0]; linestyle=:dash, color=:grey, label="α = 1")
    hline!(pl_α, [2.0]; linestyle=:dot,  color=:grey, label="α = 2")

    return plot(pl_mean, pl_α;
        layout      = (2, 1),
        size        = (800, 700),
        left_margin = 5Plots.mm,
    )
end


"""
    fit_settings_zeroinf_bnb2!(df_dds, dir_; settings, late_from, n_sample=1000)

Fit a `ZeroInfBNB2` model for each stratum in `settings` using data on or after
`late_from`. Saves one JLD2 file per setting; skips if the file already exists.

# Arguments
- `df_dds`     : long DataFrame with columns `:key`, `:strat`, `:x`, `:y`, `:date`
- `dir_`       : output directory for JLD2 chain files
- `settings`   : vector of stratum strings, e.g. `["work", "school", "other"]`
- `late_from`  : `Date` lower-bound on `:date`
- `n_sample`   : number of NUTS posterior samples (default `1000`)
"""
function fit_settings_zeroinf_bnb2!(df_dds::DataFrame, dir_::String;
                                     settings::AbstractVector{String} = ["work", "school", "other"],
                                     late_from::Date = Date(2021, 7, 1),
                                     n_sample::Int   = 1000)
    for s in settings
        path = "$(dir_)/chn_zeroinf_bnb2_$(s)_late.jld2"
        if isfile(path)
            println("Already fitted, skipping: $s")
            continue
        end
        dds = load_dds(df_dds; strat=s, late_from=late_from)
        println("Fitting: $s  (n = $(sum(dds.y)))")
        chn = fit_model_with_forward_mode(model_ZeroInfBNB2(dds), n_sample; progress=false)
        jldsave(path, result=chn)
        println("  ✓ saved → $path")
    end
    println("All settings done.")
end


"""
    plot_pdf_by_setting(df_dds, dir_; settings, late_from) -> Plot

Load per-setting ZeroInfBNB2 chains from `dir_` and overlay empirical PDF and
fitted PDF for each stratum on a single log-PDF plot.
"""
function plot_pdf_by_setting(df_dds::DataFrame, dir_::String;
                              settings::AbstractVector{String} = ["work", "school", "other"],
                              late_from::Date = Date(2021, 7, 1))
    pl = plot(xlim=[0, 30], ylim=[-5, 0],
              title="PDF by setting ($(Dates.format(late_from, "u yyyy")) – Mar 2022)",
              legend=:topright)
    for (i, s) in enumerate(settings)
        col = palette(:default)[i]
        dds = load_dds(df_dds; strat=s, late_from=late_from)
        chn = load("$(dir_)/chn_zeroinf_bnb2_$(s)_late.jld2", "result")
        bnb = get_ZeroInfBNB2(chn)
        plot_pdf!(pl, dds; markersize=2, markerstrokewidth=0.1,
                  color=col, label="$s (data)")
        plot_pdf!(pl, bnb; seriestype=:line, linewidth=1.5, ls=:dash,
                  color=col, label="$s (fit)")
    end
    return pl
end


"""
    plot_ccdf_by_setting(df_dds, dir_; settings, late_from) -> Plot

Load per-setting ZeroInfBNB2 chains from `dir_` and overlay empirical CCDF and
fitted CCDF for each stratum on a single log-x CCDF plot.
"""
function plot_ccdf_by_setting(df_dds::DataFrame, dir_::String;
                               settings::AbstractVector{String} = ["work", "school", "other"],
                               late_from::Date = Date(2021, 7, 1))
    pl = plot(xaxis=:log10,
              title="CCDF by setting ($(Dates.format(late_from, "u yyyy")) – Mar 2022)",
              legend=:topright)
    for (i, s) in enumerate(settings)
        col = palette(:default)[i]
        dds = load_dds(df_dds; strat=s, late_from=late_from)
        chn = load("$(dir_)/chn_zeroinf_bnb2_$(s)_late.jld2", "result")
        bnb = get_ZeroInfBNB2(chn)
        plot_ccdf!(pl, dds; markersize=2, markerstrokewidth=0.1,
                   color=col, label="$s (data)")
        plot_ccdf!(pl, bnb; seriestype=:line, linewidth=1.5, ls=:dash,
                   color=col, label="$s (fit)")
    end
    return pl
end


"""
    fit_biweekly_zeroinf_bnb2_settings!(df_dds, dir_; settings, n_sample=1000)

Fit `ZeroInfBNB2` to every bi-weekly window × setting combination in `df_dds`.
Saves `\$(k)_zeroinf_bnb2_\$(s).jld2` per wave `k` and setting `s`; skips existing files.
Waves with fewer than 10 observations are skipped.

The inner loop over bi-weekly keys is parallelised with `Threads.@threads`.
Each thread runs its own task-local RNG (Julia ≥ 1.7), so `Random.seed!` inside
`fit_model_with_forward_mode` is thread-safe.
"""
function fit_biweekly_zeroinf_bnb2_settings!(df_dds::DataFrame, dir_::String;
                                              settings::AbstractVector{String} = ["work", "school", "other"],
                                              n_sample::Int = 1000)
    keys_ = sort(unique(df_dds.key))
    log_lock = ReentrantLock()
    Threads.@threads for s in settings
        for k in keys_
            path = "$(dir_)/$(k)_zeroinf_bnb2_$(s).jld2"
            if isfile(path)
                lock(log_lock) do
                    println("  Already fitted, skipping: $k ($s)")
                end
                continue
            end
            df_k = @subset(df_dds, :key .== k, :strat .== s)
            isempty(df_k) && continue
            dd = DegreeDist(df_k)
            if sum(dd.y) < 10
                lock(log_lock) do
                    println("  Too few obs, skipping: $k ($s)")
                end
                continue
            end
            lock(log_lock) do
                println("  Fitting: $k  (n = $(sum(dd.y)), thread $(Threads.threadid()))")
            end
            chn = fit_model_with_forward_mode(model_ZeroInfBNB2(dd), n_sample; progress=false)
            jldsave(path, result=chn)
            lock(log_lock) do
                println("  ✓ saved → $path")
            end
        end
    end
    println("All settings × waves done.")
end


"""
    extract_biweekly_means_settings(keys_, dir_, df_dds; settings) -> DataFrame

Like `extract_biweekly_means` but loops over `settings`, loading
`\$(k)_zeroinf_bnb2_\$(s).jld2`. Returns a long DataFrame with an extra `:strat`
column. Waves whose chain file is absent are silently skipped.

Columns: `:key`, `:strat`, `:mean`, `:mean_l`, `:mean_u`, `:α`, `:α_l`, `:α_u`,
         `:mean_raw`, `:mean_trunc`
"""
function extract_biweekly_means_settings(keys_::AbstractVector, dir_::String,
                                          df_dds::DataFrame;
                                          settings::AbstractVector{String} = ["work", "school", "other"])::DataFrame
    rows = []
    for s in settings
        for k in keys_
            path = "$(dir_)/$(k)_zeroinf_bnb2_$(s).jld2"
            isfile(path) || continue
            chn = load(path, "result")

            dists = get_vec_ZeroInfBNB2_from_chn(chn)
            ms    = mean.(dists)
            αs    = map(d -> d.v + 1, dists)
            m_l, m_med, m_u = quantile(ms, [0.025, 0.5, 0.975])
            α_l, α_med, α_u = quantile(αs, [0.025, 0.5, 0.975])

            df_k = @subset(df_dds, :key .== k, :strat .== s)
            isempty(df_k) && continue
            mean_raw   = mean(DegreeDist(df_k))
            mean_trunc = mean(create_capped_frequencies(df_k) |> DegreeDist)

            push!(rows, (key=Date(k), strat=s,
                         mean=m_med, mean_l=m_l, mean_u=m_u,
                         α=α_med,    α_l=α_l,    α_u=α_u,
                         mean_raw=mean_raw, mean_trunc=mean_trunc))
        end
    end
    return sort!(DataFrame(rows), [:strat, :key])
end


"""
    plot_temporal_mean_alpha_settings(df_means; settings) -> Plot

Four-panel comparison across settings:
- (top-left)     ZeroInfBNB2 estimated mean with 95 % CI ribbon per setting
- (top-right)    empirical raw mean per setting
- (bottom-left)  empirical truncated mean (cap 50) per setting
- (bottom-right) tail index α overlaid per setting; reference lines at α = 1 and α = 2
"""
function plot_temporal_mean_alpha_settings(df_means::DataFrame;
                                            settings::AbstractVector{String} = ["work", "school", "other"])
    pl_est   = plot(; ylabel="Estimated mean",    title="Posterior mean",          xrotation=45, legend=:topright,
                    ylim=[0, 8])
    pl_raw   = plot(; ylabel="Raw mean",          title="Empirical raw mean",      xrotation=45, legend=:topright)
    pl_trunc = plot(; ylabel="Truncated mean",    title="Empirical mean (cap 50)", xrotation=45, legend=:topright)
    pl_α     = plot(; ylabel="α (tail index)",    title="Tail index α",            xrotation=45,
                      legend=:topright, ylim=[0.8, 2.5])
    #hline!(pl_α, [1.0]; linestyle=:dash, color=:grey, label="α = 1")
    #hline!(pl_α, [2.0]; linestyle=:dot,  color=:grey, label="α = 2")

    for (i, s) in enumerate(settings)
        col  = palette(:default)[i]
        df_s = @subset(df_means, :strat .== s)
        isempty(df_s) && continue

        plot!(pl_est, df_s.key, df_s.mean;
              ribbon    = (df_s.mean .- df_s.mean_l, df_s.mean_u .- df_s.mean),
              fillalpha = 0.2, color=col, marker=:circle, markersize=3, label=s)

        plot!(pl_raw, df_s.key, df_s.mean_raw;
              color=col, marker=:circle, markersize=3, label=s)

        plot!(pl_trunc, df_s.key, df_s.mean_trunc;
              color=col, marker=:diamond, markersize=3, linestyle=:dash, label=s)

        plot!(pl_α, df_s.key, df_s.α;
              ribbon    = (df_s.α .- df_s.α_l, df_s.α_u .- df_s.α),
              fillalpha = 0.2, color=col, marker=:circle, markersize=3, label=s)
    end

    return plot(pl_est, pl_raw, pl_trunc, pl_α;
        layout        = (2, 2),
        size          = (1100, 800),
        left_margin   = 5Plots.mm,
        bottom_margin = 6Plots.mm,
    )
end


"""
    plot_ccdf_diagnostics(df_dds, keys_, dir_; group_size=4, strat="non-home") -> Plot

Overlay empirical CCDF and fitted ZeroInfBNB2 CCDF for every bi-weekly wave,
grouped into panels of `group_size` waves each.
"""
function plot_ccdf_diagnostics(df_dds::DataFrame, keys_::AbstractVector, dir_::String;
                                group_size::Int = 4,
                                strat::String   = "non-home")
    suffix     = strat == "non-home" ? "nhm" : strat
    key_groups = [keys_[i:min(i+group_size-1, end)] for i in 1:group_size:length(keys_)]
    palette4   = [1, 2, 3, 4]

    subplots = []
    for grp in key_groups
        pl = plot(; xaxis=:log10, ylim=[-4, 0], xlim=[1, 10_000],
                    legend=:bottomleft,
                    xlabel="Contacts", ylabel="log10 CCDF",
                    title=strat)
        for (ci, k) in enumerate(grp)
            path = "$(dir_)/$(k)_zeroinf_bnb2_$(suffix).jld2"
            isfile(path) || continue
            dd    = @subset(df_dds, :key .== k, :strat .== strat) |> DegreeDist
            chn   = load(path, "result")
            d_fit = get_ZeroInfBNB2(chn)
            col   = palette4[ci]
            plot_ccdf!(pl, dd;    markersize=1.5, markerstrokewidth=0.0, color=col, label=k)
            plot_ccdf!(pl, d_fit; color=col, linewidth=1.5, linestyle=:dash, label="")
        end
        push!(subplots, pl)
    end

    ncols = 2
    nrows = ceil(Int, length(key_groups) / ncols)
    return plot(subplots...;
        layout=(nrows, ncols),
        size=(ncols * 420, nrows * 320),
        left_margin=4Plots.mm, bottom_margin=4Plots.mm,
    )
end


"""
    plot_means_overlay_by_setting(df_means; settings) -> Plot

One panel per setting, each overlaying three mean series:
- ZeroInfBNB2 posterior median with 95 % CI ribbon
- empirical raw mean
- empirical truncated mean (degrees capped at 50)
"""
function plot_means_overlay_by_setting(df_means::DataFrame;
                                        settings::AbstractVector{String} = ["work", "school", "other"])
    subplots = []
    for s in settings
        df_s = @subset(df_means, :strat .== s)
        isempty(df_s) && continue

        pl = plot(;
            title         = s,
            ylabel        = "Mean contacts",
            xrotation     = 45,
            legend        = :topright,
        )

        plot!(pl, df_s.key, df_s.mean;
              ribbon    = (df_s.mean .- df_s.mean_l, df_s.mean_u .- df_s.mean),
              fillalpha = 0.2,
              color     = :steelblue,
              marker    = :circle, markersize = 3,
              label     = "Estimated (ZeroInfBNB2)")

        plot!(pl, df_s.key, df_s.mean_raw;
              color  = :darkorange,
              marker = :circle, markersize = 3,
              label  = "Raw mean")

        plot!(pl, df_s.key, df_s.mean_trunc;
              color     = :green,
              marker    = :diamond, markersize = 3,
              linestyle = :dash,
              label     = "Truncated mean (cap 50)")

        push!(subplots, pl)
    end

    nrows = length(subplots)
    plot!(subplots[1], ylim=[0,6.5])
    plot!(subplots[2], ylim=[0,1])
    plot!(subplots[3], ylim=[0,2])
    return plot(subplots...;
        layout        = (nrows, 1),
        size          = (900, nrows * 320),
        left_margin   = 5Plots.mm,
        bottom_margin = 8Plots.mm,
    )
end


"""
    plot_settings_sum_vs_nonhome(df_means_settings, df_dds; settings) -> Plot

Sum the estimated means (with 95 % CI), raw means, and truncated means across
`settings` per wave, then overlay against empirical non-home means (raw and
truncated, cap 50) derived directly from `df_dds`.

CI for the sum is obtained by summing per-setting lower/upper credible bounds,
which is exact for independent posteriors under additive transformations.
"""
function plot_settings_sum_vs_nonhome(df_means_settings::DataFrame,
                                       df_dds::DataFrame;
                                       settings::AbstractVector{String} = ["work", "school", "other"])
    keys_ = sort(unique(df_means_settings.key))   # Vector{Date}

    # --- sum across settings per wave ---
    sum_rows = []
    for k in keys_
        df_k = @subset(df_means_settings, :key .== k, :strat .∈ Ref(settings))
        nrow(df_k) == 0 && continue
        push!(sum_rows, (
            key            = k,
            mean_sum       = sum(df_k.mean),
            mean_sum_l     = sum(df_k.mean_l),
            mean_sum_u     = sum(df_k.mean_u),
            mean_raw_sum   = sum(df_k.mean_raw),
            mean_trunc_sum = sum(df_k.mean_trunc),
        ))
    end
    df_sum = DataFrame(sum_rows)

    # --- empirical non-home means from raw df_dds ---
    nhm_rows = []
    for k in keys_
        df_k = @subset(df_dds, :key .== string(k), :strat .== "non-home")
        isempty(df_k) && continue
        push!(nhm_rows, (
            key             = k,
            mean_raw_nhm    = mean(DegreeDist(df_k)),
            mean_trunc_nhm  = mean(create_capped_frequencies(df_k) |> DegreeDist),
        ))
    end
    df_nhm  = DataFrame(nhm_rows)
    df_plot = innerjoin(df_sum, df_nhm; on = :key)

    label_sum = join(settings, "+")
    pl = plot(;
        ylabel    = "Mean contacts",
        title     = "Sum of settings ($(label_sum)) vs. non-home",
        xrotation = 45,
        legend    = :topright,
        ylim=[0, 9]
    )

    # summed estimated posterior median + CI
    plot!(pl, df_plot.key, df_plot.mean_sum;
          ribbon    = (df_plot.mean_sum .- df_plot.mean_sum_l,
                       df_plot.mean_sum_u .- df_plot.mean_sum),
          fillalpha = 0.2, color = :steelblue,
          marker = :circle, markersize = 3,
          label  = "Σ estimated ($(label_sum))")

    # summed empirical raw
    plot!(pl, df_plot.key, df_plot.mean_raw_sum;
          color = :steelblue, marker = :utriangle, markersize = 3,
          linestyle = :dash, label = "Σ raw ($(label_sum))")

    # summed empirical truncated
    plot!(pl, df_plot.key, df_plot.mean_trunc_sum;
          color = :steelblue, marker = :diamond, markersize = 3,
          linestyle = :dot, label = "Σ trunc cap50 ($(label_sum))")

    # non-home empirical raw
    plot!(pl, df_plot.key, df_plot.mean_raw_nhm;
          color = :crimson, marker = :circle, markersize = 3,
          label = "Non-home raw")

    # non-home empirical truncated
    plot!(pl, df_plot.key, df_plot.mean_trunc_nhm;
          color = :crimson, marker = :diamond, markersize = 3,
          linestyle = :dash, label = "Non-home trunc (cap 50)")

    return plot(pl; size = (900, 400), left_margin = 5Plots.mm, bottom_margin = 8Plots.mm)
end
