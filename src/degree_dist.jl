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
		xlim = [1, 10000],
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
	pl = plot(; xaxis = :log10, ylim = [-4, 0], xlim = [1, 10_000])
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
	pl = plot(; xaxis = :log10, ylim = [-5, 0], xlim = [1, 10_000], kwds...)
	for gdf in groupby(df_dd, col)
		dd = DegreeDist(gdf |> DataFrame)
		plot_ccdf!(pl, dd, label = unique(gdf[:, col])[1], markersize = 2.5, markerstrokewidth = 0.0)
	end
	pl
end
