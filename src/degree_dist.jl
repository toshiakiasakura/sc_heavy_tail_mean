Base.@kwdef mutable struct DegreeDist
	x::Vector{Int64} # Degree
	y::Vector{Int64} # Corresponding count
	include_zero::Bool = true
end

function DegreeDist(cnt::Vector{Int64}; include_zero = true)::DegreeDist
	cnt = cnt |> countmap
	x, y = collect(pairs(cnt)) |> (x -> (first.(x), last.(x)))
	ind = sortperm(x)
	return DegreeDist(x[ind], y[ind], include_zero)
end

function DegreeDist(cnt::Vector{Int64}, n_part::Int64; include_zero = true)::DegreeDist
	n_cnt1more = length(cnt)
	dd = DegreeDist(cnt; include_zero = include_zero)
	insert!(dd.x, 1, 0)
	insert!(dd.y, 1, n_part - n_cnt1more)
	return dd
end

dd_to_df(dd::DegreeDist)::DataFrame = DataFrame(x = dd.x, y = dd.y)
function dd_to_df(dd::DegreeDist, strat::String)::DataFrame
	df = dd_to_df(dd)
	df[!, :strat] .= strat
	return df
end
dd_to_line_vec(dd::DegreeDist)::Vector = vcat([fill(x, y) for (x, y) in zip(dd.x, dd.y)]...)

function DegreeDist(df::DataFrame)::DegreeDist
	if "x" in names(df) && "y" in names(df)
		if length(df[:, :x]) == length(unique(df[:, :x]))
			return DegreeDist(df[:, :x], df[:, :y], true)
		else
			error("x values are not unique")
		end
	else
		error("DataFrame does not have x and y columns")
	end
end


"""Merge all rows of a degree-dist DataFrame into a single `DegreeDist`
by summing `y` over duplicate `x` values.

Example:
    period_keys = filter(k -> startswith(k, "2021"), unique(df_dds_nhm[:, :key]))
    dd = @subset(df_dds_nhm, in.(:key, Ref(period_keys))) |> merge_dd
"""
function merge_dd(df_dds::DataFrame)::DegreeDist
    agg = combine(groupby(df_dds, :x), :y => sum => :y)
    sort!(agg, :x)
    return DegreeDist(agg[:, :x], agg[:, :y], true)
end

Base.length(dd::DegreeDist) = length(dd.x)
Base.maximum(dd::DegreeDist) = maximum(dd.x)
Base.iterate(p::DegreeDist) = (p, nothing)
Base.iterate(p::DegreeDist, nothing) = nothing
Distributions.rand(dd::DegreeDist, n::Int64) = sample(dd.x, Weights(dd.y), n)
Distributions.mean(dd::DegreeDist) = sum(dd.x .* dd.y) / sum(dd.y)
Base.sum(dd::DegreeDist) = sum(dd.y)

"""Given, a vector of a probablity mass function,
return a ccdf distribution.
"""
function obtain_ccdf(y::Vector)
	return cumsum(y[end:-1:begin])[end:-1:begin]
end

function obtain_ccdf(dd::DegreeDist)::Vector{Float64}
	y = dd.y / sum(dd.y)
	y_ccdf = obtain_ccdf(y)
	return y_ccdf
end

"""
See also `turing_utils.jl` which contains
- `plot_single_setting`
- `plot_multiple_settings`
"""
function plot_ccdf!(pl::Plots.Plot, dd::DegreeDist; ytk_digit = 6, kwds...)
	if dd.include_zero == true
		dd = deepcopy(dd)
		dd.x = dd.x[2:end]
		dd.y = dd.y[2:end]
	end
	y_ccdf = obtain_ccdf(dd)
	ind_incl = 1:ytk_digit
	ytk = (
		[0, -1, -2, -3, -4, -5][ind_incl],
		[L"1", L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}"][ind_incl])
	plot!(pl, dd.x, log10.(y_ccdf); marker = :circle, yticks = ytk,
		kwds...)
end
plot_ccdf!(pl::Plots.Plot, x::Vector{Int64}; kwds...) = plot_ccdf!(pl, DegreeDist(x); kwds...)

function plot_ccdf(dd::DegreeDist; kwds...)::Plots.Plot
	y_ccdf = obtain_ccdf(dd)
	pl = plot(; xaxis = :log10, xlabel = "log10(k)", ylabel = "log10(ccdf(k))",
		xlim = [0.1, 10000],
	)
	scatter!(pl, dd.x, log10.(y_ccdf); kwds...)
	return pl
end
plot_ccdf(x::Vector{Int64}; kwds...)::Plots.Plot = plot_ccdf(DegreeDist(x); kwds...)

function plot_pdf!(pl::Plots.Plot, dd::DegreeDist; conv_log10 = true, ytk_digit = 6, kwds...)
	y_pdf = dd.y / sum(dd.y)
	y_pdf = conv_log10 == true ? log10.(y_pdf) : y_pdf
	ind_incl = 1:ytk_digit
	if conv_log10 == true
		ytk = (
			[0, -1, -2, -3, -4, -5][ind_incl],
			[L"1", L"10^{-1}", L"10^{-2}", L"10^{-3}", L"10^{-4}", L"10^{-5}"][ind_incl])
	else
		ytk = true
	end
	plot!(pl, dd.x, y_pdf; marker = :circle, yticks=ytk, kwds...)
end
plot_pdf!(pl::Plots.Plot, x::Vector{Int64}; kwds...) = plot_pdf!(pl, DegreeDist(x); kwds...)

function plot_pdf(dd::DegreeDist; kwds...)
	pl = plot(xlabel = "k", ylabel = "pdf", xlim = [0, 20])
	plot_pdf!(pl, dd; kwds...)
	return pl
end
plot_pdf(x::Vector{Int64}; kwds...) = plot_pdf(DegreeDist(x); kwds...)

"""
Args:
- df_dd: DegreeDist type of dataframe.
"""
function plot_single_survey(df_dd::DataFrame)
	pl1 = plot_pdf_single_survey(df_dd)
	pl2 = plot_ccdf_single_survey(df_dd)
	plot(pl1, pl2; size = (800, 400))
end

function plot_pdf_single_survey!(pl::Plots.Plot, df_dd::DataFrame)
	for strat in ["all", "home", "non-home"]
		dd = @subset(df_dd, :strat .== strat) |> DegreeDist
		plot_pdf!(pl, dd, label = strat, markersize = 2.5, markerstrokewidth = 0.0)
	end
	pl
end

function plot_pdf_single_survey(df_dd::DataFrame)
	pl = plot(; xlim = [0, 50], ylim = [-4, 0])
	plot_pdf_single_survey!(pl, df_dd)
end

function plot_ccdf_single_survey!(pl::Plots.Plot, df_dd::DataFrame)
	for strat in ["all", "home", "non-home"]
		dd = @subset(df_dd, :strat .== strat) |> DegreeDist
		plot_ccdf!(pl, dd, label = strat, markersize = 2.5, markerstrokewidth = 0.0)
	end
	pl
end

function plot_ccdf_single_survey(df_dd::DataFrame)
	pl = plot(; xaxis = :log10, ylim = [-4, 0], xlim = [0.1, 10_000])
	plot_ccdf_single_survey!(pl, df_dd)
end

function plot_pdf_across_survey(df_dd::DataFrame; col = :key)
	#pl = plot(; xlim = [0, 50], ylim = [0, 0.3])
	pl = plot(; xlim = [0, 50], ylim = [-5, 0])
	for gdf in groupby(df_dd, col)
		dd = DegreeDist(gdf |> DataFrame)
		plot_pdf!(
			pl,
			dd,
			label = unique(gdf[:, col])[1],
			markersize = 2.5,
			markerstrokewidth = 0.0,
			#conv_log10 = false
		)
	end
	pl
end

function plot_ccdf_across_survey(df_dd::DataFrame; col = :key, kwds...)
	pl = plot(; xaxis = :log10, ylim = [-5, 0], xlim = [0.1, 10_000], kwds...)
	for gdf in groupby(df_dd, col)
		dd = DegreeDist(gdf |> DataFrame)
		plot_ccdf!(pl, dd, label = unique(gdf[:, col])[1], markersize = 2.5, markerstrokewidth = 0.0)
	end
	pl
end


###################################################
##### Effective (duration-weighted) degree     ####
###################################################

# Bin midpoints (minutes) for `:duration_multi` levels 1..5.
# Level 5 (>4h) is open-ended; the effective duration is capped at d_max so w=1.
const _DURATION_T_MID = (2.5, 10.0, 37.5, 150.0, Inf)

_is_dur_na(v) = ismissing(v) || (v isa AbstractString && v == "NA")
_dur_to_int(v) = v isa AbstractString ? parse(Int, v) : Int(v)

"""
    duration_weight(d, d_max=300)

Per-contact weight from `:duration_multi` ∈ {1..5} and maximum duration `d_max`
(in minutes; default 300). Missing / `"NA"` → treated as <5 min (level 1) per
inst/2_effective_contact_degree.md.

Reference midpoints (min): 1→2.5, 2→10, 3→37.5, 4→150, 5→cap at d_max (>4h ⇒ w=1).
The closed form is `w = 1 - (d_max - t_mid)/d_max = t_mid/d_max`, with `t_mid`
clipped at `d_max` for level 5. Requires `d_max ≥ 240`.
"""
function duration_weight(d, d_max::Real=300)
    d_eff = _is_dur_na(d) ? 1 : _dur_to_int(d)
    1 <= d_eff <= 5 || error("unexpected :duration_multi value $d")
    t = min(_DURATION_T_MID[d_eff], float(d_max))
    return t / d_max
end

const _DURATION_LABELS = ("<5 min", "5–15 min", "15 min–1 h", "1–4 h", ">4 h")

"""
    print_duration_weights(d_max; io=stdout)

Print `duration_weight(d, d_max)` for each `:duration_multi` level d ∈ 1..5.
"""
function print_duration_weights(d_max::Real; io::IO = stdout)
    println(io, "duration_weight(d, d_max = $d_max min):")
    for d in 1:5
        t_mid = isfinite(_DURATION_T_MID[d]) ? _DURATION_T_MID[d] : float(d_max)
        @printf(io, "  d=%d  %-11s  t_mid=%5.1f min  w=%.6f\n",
                d, _DURATION_LABELS[d], t_mid, duration_weight(d, d_max))
    end
end

"""
    contact_degrees(df, df_part; setting, weighted=false, d_max=300)

Vector of degrees, one entry per (`:part_id_d`, `:date`) row in `df_part`.

- `setting ∈ (:all, :home, :nonhome)` filters via `:cnt_home` (`"true"`/`"false"`).
- `weighted=false` → `Vector{Int}` (count of contacts).
- `weighted=true`  → `Vector{Float64}` (sum of `duration_weight`, NA → <5min).

Participant-days with no contacts in the chosen setting receive 0 / 0.0.
"""
function contact_degrees(df::DataFrame, df_part::DataFrame;
                         setting::Symbol, weighted::Bool=false, d_max::Real=300)
    sub = setting === :home    ? @subset(df, :cnt_home .== "true")  :
          setting === :nonhome ? @subset(df, :cnt_home .== "false") :
          setting === :all     ? df :
          error("setting must be :all, :home, or :nonhome")

    keys_part = unique(@select(df_part, :part_id_d, :date))

    if weighted
        sub = @transform(sub, :w = duration_weight.(:duration_multi, d_max))
        agg = combine(groupby(sub, [:part_id_d, :date]), :w => sum => :deg)
        joined = leftjoin(keys_part, agg, on = [:part_id_d, :date])
        return coalesce.(joined.deg, 0.0)
    else
        agg = combine(groupby(sub, [:part_id_d, :date]), nrow => :deg)
        joined = leftjoin(keys_part, agg, on = [:part_id_d, :date])
        return coalesce.(joined.deg, 0)
    end
end

"""
    dm_expected_weight(fit, n, d_max)

E[w | n] under the 2j Dirichlet-multinomial fit. `fit.β` is the (P × K=5)
coefficient matrix from `fit_mglm_dm`. For each participant-day degree `n`,
proportion vector p(n) is the softmax of `[1 log(n)] · fit.β`; the expected
weight is `sum(p(n) .* w_levels(d_max))`.
"""
function dm_expected_weight(fit::NamedTuple, n::Integer, d_max::Real)
    n == 0 && return 0.0
    P = size(fit.β, 1)
    Xn = P == 1 ? reshape([1.0], 1, 1) : reshape([1.0, log(Float64(n))], 1, 2)
    p = mglm_dm_proportions(fit.β, Xn)               # 1 × 5
    w_levels = (duration_weight(k, d_max) for k in 1:5)
    return sum(p[1, k] * w for (k, w) in enumerate(w_levels))
end

"""
    contact_degrees_dm_imputed(df, df_part, fit; setting, d_max=300)

Vector of weighted degrees per (`:part_id_d`, `:date`). NA contacts are
imputed softly: each contributes `dm_expected_weight(fit, n, d_max)` where `n`
is the participant-day's total contact count in the chosen setting (NA included).

`fit` is an MGLM DM fit produced by `fit_mglm_dm(X, Y)` on the same setting,
where `X = [1 log(n)]` and `Y` is the 5-column duration count matrix from
`prepare_dm_inputs(...; outcome=:duration_multi, K=5, dropna_keep_n=true)`.

`setting` must be `:home` or `:nonhome`. For an "all" effective degree,
combine the two settings per (participant, date).
"""
function contact_degrees_dm_imputed(df::DataFrame, df_part::DataFrame,
                                    fit::NamedTuple;
                                    setting::Symbol, d_max::Real=300)
    sub = setting === :home    ? @subset(df, :cnt_home .== "true")  :
          setting === :nonhome ? @subset(df, :cnt_home .== "false") :
          error(":all is not supported here; combine :home + :nonhome instead")

    sub = @transform(sub,
        :w_obs = ifelse.(_is_dur_na.(:duration_multi),
                         0.0,
                         duration_weight.(:duration_multi, d_max)),
        :is_na = _is_dur_na.(:duration_multi))

    agg = combine(groupby(sub, [:part_id_d, :date]),
                  :w_obs => sum => :w_sum,
                  :is_na => sum => :n_na,
                  nrow         => :n_tot)

    agg = @transform(agg,
        :deg = :w_sum .+ :n_na .* dm_expected_weight.(Ref(fit), :n_tot, d_max))

    keys_part = unique(@select(df_part, :part_id_d, :date))
    joined = leftjoin(keys_part, @select(agg, :part_id_d, :date, :deg),
                      on = [:part_id_d, :date])
    return coalesce.(joined.deg, 0.0)
end


###################################################
##### Continuous-degree plotting (variants 2,3) ###
###################################################

"""Empirical CCDF on continuous values, log-y axis. Uses only positive entries."""
function plot_ccdf_continuous!(plt::Plots.Plot, x::AbstractVector{<:Real};
                               label = "", color = :auto, kwds...)
    xs = sort(filter(>(0), x))
    isempty(xs) && return plt
    n = length(xs)
    ccdf = (n .- (0:n-1)) ./ n           # 1, (n-1)/n, …, 1/n
    plot!(plt, xs, ccdf;
          yscale = :log10,
          xlabel = "weighted degree", ylabel = "CCDF",
          label = label, color = color, kwds...)
end

plot_ccdf_continuous(x::AbstractVector{<:Real}; kwds...) =
    plot_ccdf_continuous!(plot(), x; kwds...)

"""Histogram-based PDF on continuous values, semilog-y, fixed bin width."""
function plot_pdf_hist!(plt::Plots.Plot, x::AbstractVector{<:Real};
                        label = "", color = :auto, binwidth::Real = 0.5, kwds...)
    isempty(x) && return plt
    edges = 0:binwidth:(maximum(x) + binwidth)
    histogram!(plt, x;
               bins = edges, normalize = :pdf,
               yscale = :log10,
               xlabel = "weighted degree", ylabel = "density",
               label = label, color = color, alpha = 0.4, kwds...)
end

plot_pdf_hist(x::AbstractVector{<:Real}; kwds...) =
    plot_pdf_hist!(plot(), x; kwds...)

function plot_pdf_hist_single_survey(x_all, x_home, x_non; binwidth = 0.5)
    p = plot(; xlim = [0, 50])
    plot_pdf_hist!(p, x_all;  label = "all",      color = :black, binwidth = binwidth)
    plot_pdf_hist!(p, x_home; label = "home",     color = :red,   binwidth = binwidth)
    plot_pdf_hist!(p, x_non;  label = "non-home", color = :blue,  binwidth = binwidth)
    return p
end

function plot_ccdf_continuous_single_survey(x_all, x_home, x_non)
    p = plot(; xaxis = :log10, xlim = [0.1, 10_000])
    plot_ccdf_continuous!(p, x_all;  label = "all",      color = :black)
    plot_ccdf_continuous!(p, x_home; label = "home",     color = :red)
    plot_ccdf_continuous!(p, x_non;  label = "non-home", color = :blue)
    return p
end

# Three-weighting overlay (one panel per setting). Compares unweighted,
# weighted (NA → <5min), and weighted (DM-imputed NA) for the same
# `(part_id_d, date)` cells.
function plot_pdf_hist_by_weighting(x_unw, x_w, x_w_imp; binwidth = 0.5)
    p = plot(; xlim = [0, 50])
    plot_pdf_hist!(p, x_unw;   label = "unweighted",       color = :black,  binwidth = binwidth)
    plot_pdf_hist!(p, x_w;     label = "weighted (NA→<5)", color = :orange, binwidth = binwidth)
    plot_pdf_hist!(p, x_w_imp; label = "weighted (DM)",    color = :purple, binwidth = binwidth)
    return p
end

function plot_ccdf_continuous_by_weighting(x_unw, x_w, x_w_imp)
    p = plot(; xaxis = :log10, xlim = [0.1, 10_000])
    plot_ccdf_continuous!(p, x_unw;   label = "unweighted",       color = :black)
    plot_ccdf_continuous!(p, x_w;     label = "weighted (NA→<5)", color = :orange)
    plot_ccdf_continuous!(p, x_w_imp; label = "weighted (DM)",    color = :purple)
    return p
end

"""
    plot_weighting_compare(x_unw, x_w, x_w_imp; setting_label, binwidth=0.5)

PDF + CCDF panels overlaying the three weighting variants. Returns a single
`Plots.Plot` titled with `setting_label`.
"""
function plot_weighting_compare(x_unw, x_w, x_w_imp;
                                setting_label::String, binwidth::Real = 0.5)
    p_pdf  = plot_pdf_hist_by_weighting(x_unw, x_w, x_w_imp; binwidth = binwidth)
    p_ccdf = plot_ccdf_continuous_by_weighting(x_unw, x_w, x_w_imp)
    plot(p_pdf, p_ccdf; layout = (1, 2), size = (900, 400),
         plot_title = "Weighting comparison — $setting_label")
end

"""
    plot_ccdf_dmax_compare(x_unw, x_w, x_w_dmax2, x_imp, x_imp_dmax2;
                           setting_label, d_max=300, d_max2=240)

CCDF-only overlay comparing the three weighting variants at two `d_max` values.
Solid lines are `d_max`; dashed lines are `d_max2`. Unweighted appears once.
"""
function plot_ccdf_dmax_compare(x_unw, x_w, x_w_dmax2, x_imp, x_imp_dmax2;
                                setting_label::String,
                                d_max::Real = 300, d_max2::Real = 240)
    p = plot(; xaxis = :log10, xlim = [0.1, 10_000],
              size = (700, 450),
              title = "Weighting + d_max sensitivity — $setting_label")
    plot_ccdf_continuous!(p, x_unw;        label = "unweighted",
                          color = :black)
    plot_ccdf_continuous!(p, x_w;          label = "weighted (NA→<5), d_max=$d_max",
                          color = :orange, linestyle = :solid)
    plot_ccdf_continuous!(p, x_w_dmax2;    label = "weighted (NA→<5), d_max=$d_max2",
                          color = :orange, linestyle = :dash)
    plot_ccdf_continuous!(p, x_imp;        label = "weighted (DM), d_max=$d_max",
                          color = :purple, linestyle = :solid)
    plot_ccdf_continuous!(p, x_imp_dmax2;  label = "weighted (DM), d_max=$d_max2",
                          color = :purple, linestyle = :dash)
    return p
end
