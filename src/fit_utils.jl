#####################################
##### Post-processing functions #####
#####################################
function plot_zero_pct(df_dds::DataFrame)
	df_obs = combine(groupby(df_dds, [:key, :strat])) do gdf
		n_answer = gdf[:, :y] |> sum
		n0 = gdf[gdf.x .== 0, :y][1]
		(n_answer = n_answer, n0 = n0)
	end
	@transform!(df_obs, :pct0 = :n0 ./ :n_answer * 100);

	# Non-home
	df_obs_nhm = @subset(df_obs, :strat .== "non-home")
	keys_ = df_obs_nhm[:, :key]
	xtk = ((1:length(keys_)) .- 0.5, keys_)
	bar(keys_, df_obs_nhm[:, :pct0],
		xrotation = 90, xticks = xtk, label = "", title = "Non-home contacts",
		ylabel = "Percentage of zero answers", ylim = [0, 100]) |> display

	df_obs_nhm = @subset(df_obs, :strat .== "all")
	keys_ = df_obs_nhm[:, :key]
	xtk = ((1:length(keys_)) .- 0.5, keys_)
	bar(keys_, df_obs_nhm[:, :pct0],
		xrotation = 90, xticks = xtk, label = "", title = "All contacts",
		ylabel = "Percentage of zero answers", ylim = [0, 100])
end

function summarise_res_one_strat(res::Dict, strat, key)
	model_names = get_model_names()
	df_tmp = DataFrame()
	for model_name in model_names
		dd = res["dds"][strat]
		chn = res["chns_$strat"][model_name]
		dist = get_ZeroInfDist(chn, model_name)
		dists = get_vec_ZeroInfDist_from_chn(chn, model_name)
		n_sample = sum(dd.y)
		mean_raw = mean(dd)
		mean_ = mean(dist)
		mean_set = mean.(dists)
		cond = isnan.(mean_set)
		if (any(cond) == true) & (all(cond) == true)
			println("All NaN: $key, strat: $strat, model: $model_name")
			mean_l = NaN
			mean_u = NaN
		elseif any(cond) == true
			println("NaN presence: $key, strat: $strat, model: $model_name")
			mean_l, mean_u = quantile(mean_set[.!cond], [0.025, 0.975])
		else
			mean_l, mean_u = quantile(mean_set, [0.025, 0.975])
		end
		waic_ = calc_waic(dists, dd)
		r = DataFrame(key = key, strat = strat, model = model_name,
			n_sample = n_sample,
			mean_raw = mean_raw, mean = mean_, mean_l = mean_l, mean_u = mean_u,
			waic = waic_,
		)

		df_tmp = vcat(df_tmp, r)
	end
	return df_tmp
end

"""
Note:
See `read_master_with_fit_summary` for creating a sequences of data.
"""
function create_summary_stat_one_data(key; dir_="../dt_intermediate")
	path = "$(dir_)/$(key)_chns.jld2"
	if isfile(path) == false
		return DataFrame()
	end
	res = load(path)["result"];

	df_sum = DataFrame()
	for strat in ["home", "non-home"]
		df_tmp = summarise_res_one_strat(res, strat, key)
		df_sum = vcat(df_sum, df_tmp)
	end
	return df_sum
end

function calc_waic(dists::Vector{T}, dd::DegreeDist
) where {T <: DiscreteUnivariateDistribution}
	lppd = 0
	p_waic = 0
	for i in 1:length(dd.x)
		ll = logpdf.(dists, dd.x[i])

		lppd1 = ll .|> exp |> mean |> log
		lppd += lppd1 * dd.y[i]

		p_waic1 = ll |> var
		p_waic += p_waic1 * dd.y[i]
	end
	return -2 * (lppd - p_waic)
end

"""
WAIC from a per-draw, per-observation log-likelihood matrix.

`loglik` has shape `(S, N)` — `S` posterior draws by `N` observations.

Returns a `NamedTuple`:
- `waic`     — −2 · (Σ lppd_i − Σ p_waic_i)
- `lppd`     — Σ_i log(mean_s exp(loglik[s, i]))
- `p_waic`   — Σ_i var_s(loglik[s, i])
- `elpd_i`   — vector of per-obs `lppd_i − p_waic_i` (length N)
- `se`       — `sqrt(N · var_i(−2 · elpd_i))`, the standard error of WAIC
"""
function calc_waic_from_loglik(loglik::AbstractMatrix)
	S, N = size(loglik)
	lppd_i   = [logsumexp(view(loglik, :, i)) - log(S) for i in 1:N]
	p_waic_i = [var(view(loglik, :, i))                for i in 1:N]
	elpd_i   = lppd_i .- p_waic_i
	waic     = -2 * sum(elpd_i)
	se       = sqrt(N * var(-2 .* elpd_i))
	return (waic = waic, lppd = sum(lppd_i), p_waic = sum(p_waic_i),
	        elpd_i = elpd_i, se = se)
end

"""
Standard error of the WAIC difference between two models on the same data,
given their per-observation `elpd_i` vectors.
"""
function calc_waic_diff_se(elpd_a::AbstractVector, elpd_b::AbstractVector)
	@assert length(elpd_a) == length(elpd_b)
	d = -2 .* (elpd_a .- elpd_b)
	return sqrt(length(d) * var(d))
end

# TODO: delete it.
#function calc_waic_weights(df_waic::DataFrame, model_names::Vector)
#	mat_w = df_waic[:, model_names] |> Matrix
#	mat_w = mat_w .- minimum(mat_w, dims = 2)
#	mat_w = exp.(-0.5 .* mat_w)
#	mat_w = mat_w ./ sum(mat_w, dims = 2)
#	df_w = DataFrame(mat_w, model_names)
#	df_w[!, :n_answer] = df_waic.n_answer;
#	return df_w
#end

"""Add waic_weight for data frame.
"""
function flag_minimum_IC(df_res::DataFrame, ic::Symbol)::DataFrame
	f_ic = Symbol("fmin_$(ic)")
	weight_ic = Symbol("weight_$(ic)")

	@transform!(df_res, :key_strat = :key .* "_" .* :strat)

	df_tmp = @by df_res :key_strat @astable begin
		:model = :model
		min_ = minimum($ic)
		$f_ic = $ic .== min_
		$weight_ic = exp.(-($ic .- min_)/2) /
					 sum(exp.(- ($ic .- min_) ./ 2))
	end
	df_new = innerjoin(df_res, df_tmp, on = [:key_strat, :model])
	return df_new
end

function get_model_names()
	models = [model_ZeroInfNegativeBinomial, model_ZeroInfPoissonLogNormal,
		model_ZeroInfPoissonLomax]
	model_names = get_dist_name_from_model.(models);
	return model_names
end

#################################
##### EVI related functions #####
#################################
function read_EVI_summary(df_n_obs; filter_quantile = false)
	df_EVI = CSV.read("../dt_intermediate/EVI_summary.csv", DataFrame)
	if filter_quantile == true
		df_EVI = @subset(df_EVI, :q .== 0.98)
	end
	df_EVI = @pipe leftjoin(df_EVI, df_n_obs, on = :key) |>
				   sort(_, :n_answer; rev = false)
	@transform!(df_EVI, :m_l = :mean - :lower, :m_u = :upper - :mean)
	return df_EVI
end

function plot_mean_excess_function(x_raw::Vector{Int64})
	qs = [(0.95:0.001:0.999)...]
	qs_v = quantile.(Ref(x_raw), qs)
	mef = [mean(x_raw[x_raw .> q] .- q) for q in qs_v]
	plot(qs, mef)
end

function obtain_peak_over_threshold_values(x_raw::Vector{Int64}, q::Float64)
	q_v = quantile(x_raw, q)
	x_thres = x_raw[x_raw .> q_v]
	x = x_thres .- q_v
	return x
end

function fit_GP_to_q(dd::DegreeDist, q::Float64)::DataFrameRow
	x = dd_to_line_vec(dd)
	q_v = quantile(x, q)
	x = obtain_peak_over_threshold_values(x, q)
	chn = fit_model_GP(x, model_GeneralizedPareto; n_samples = 2000)
	res = extract_chain_info(chn)
	m, l, h = res[2, [:mean, :lower, :upper]]
	return DataFrame(q = q, q_v = q_v, mean = m, lower = l, upper = h)[1, :]
end

function EVI_estimate_for_qs(dd::DegreeDist;
	qs = [0.95, 0.97, 0.99, 0.995, 0.999])::DataFrame
	res = DataFrame()
	for q in qs
		r = fit_GP_to_q(dd, q)
		push!(res, r)
	end
	return res
end

function collect_estimates_across_studies(df_dd::DataFrame, keys_;
	strat = "non-home", kwds...)
	res_sum = DataFrame()
	for k in keys_
		dd = @subset(df_dd, :key .== k, :strat .== strat) |> DegreeDist
		res = EVI_estimate_for_qs(dd; kwds...)
		res[!, :key] .= k
		res_sum = vcat(res_sum, res)
	end
	res_sum
end

function plot_EVI_across_surveys(df_EVI; col_ytick = :key, title = "Non-home contacts")
	ylim = [0.5, length(df_EVI.key |> unique) + 0.5]
	pl = plot(ylim = ylim,
		title = title,
		ytickfontsize = 8,
	)
	yticks = (1:nrow(df_EVI), df_EVI[:, col_ytick])
	scatter!(pl, df_EVI[:, :mean], 1:nrow(df_EVI), xerr = (df_EVI[:, :m_l], df_EVI[:, :m_u]),
		xlabel = "Extreme value index", yticks = yticks,
		label = "", color = :black, markerstrokewidth = 0.5)
	vline!(pl, [0], colour = :black, linestyle = :dash, label = "")
	return pl
end

function table_styl_annotate!(
	pl::Plots.Plot, labels::AbstractVector, title::String, x_pos::Real; dy::Real = 1.2)

	y_pos = length(labels) + dy
	annotate!(pl, x_pos, y_pos, text(boldstring(title), :black, :left, :centre, :bold, 8, "Helvetica"))
	for (i, lbl) in enumerate(labels)
		annotate!(pl, x_pos, i, text(lbl, :black, :left, :centre, 8, "Helvetica"))
	end
end

function plot_EVI_forest_multi_q(df_EVI::DataFrame;
    col_ytick = :key, title = "", show_yticks = true)

    qs = sort(unique(df_EVI.q))
    df_plot = @subset(df_EVI, :q .∈ Ref(qs))

    # Get unique keys and prepare spacing
    keys_unique = unique(df_plot.key)
    n_keys = length(keys_unique)
    n_q = length(qs)

    # Calculate y positions: each key gets n_q positions with spacing
    y_spacing = 1.0  # spacing between different q within same key
    row_height = n_q * y_spacing + 0.5  # total height per key including gap

    @transform!(df_plot, :m_l = :mean - :lower, :m_u = :upper - :mean)

    # Create y-axis positions
    ylim = [0.0, n_keys * row_height + 0.5]
    ytick_positions = Float64[]
    ytick_labels = String[]

    pl = plot(ylim = ylim,
        title = title,
        ytickfontsize = 8,
        xlabel = "Extreme value index",
        legend = :topright,
    )

    colors = [RGB(0.0, 0.45, 0.70), RGB(0.90, 0.62, 0.0), RGB(0.0, 0.62, 0.45),
              RGB(0.80, 0.47, 0.65), RGB(0.34, 0.71, 0.91)]
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle]

    for (q_idx, q_val) in enumerate(qs)
        df_q = @subset(df_plot, :q .== q_val)
        sort!(df_q, order(:key, by = x -> findfirst(==(x), keys_unique)))
        y_positions = [(findfirst(==(k), keys_unique) - 1) * row_height + 1 + (q_idx - 1) * y_spacing
                       for k in df_q.key]

        scatter!(pl, df_q.mean, y_positions,
            xerr = (df_q.m_l, df_q.m_u),
            label = "q = $q_val",
            color = colors[mod1(q_idx, length(colors))],
            marker = markers[mod1(q_idx, length(markers))],
            markerstrokewidth = 0.4,
            markersize = 3)
    end
    # Set y-axis ticks at the center of each key's group
    for (i, key) in enumerate(keys_unique)
        y_center = (i - 1) * row_height + 1 + (n_q - 1) * y_spacing / 2
        push!(ytick_positions, y_center)
        if show_yticks
            push!(ytick_labels, string(key))
        else
            push!(ytick_labels, "")
        end
    end

    plot!(pl, yticks = (ytick_positions, ytick_labels))
    vline!(pl, [0], colour = :black, linestyle = :dash, label = "")
    return pl
end

function plot_EVI_sensitivity_qs(df_EVI)
	clean_survey_key_names!(df_EVI)
	df_EVI_ana = @subset(df_EVI, @byrow (:key in rem_lis) == false);
	df_EVI_hm = @subset(df_EVI_ana, :strat .== "home");
	df_EVI_nhm = @subset(df_EVI_ana, :strat .== "non-home");
	pl_nhm = plot_EVI_forest_multi_q(df_EVI_nhm, title = "Non-home contacts")
	pl_hm = plot_EVI_forest_multi_q(df_EVI_hm, show_yticks=false, title = "Home contacts")
	pl = plot(pl_nhm, pl_hm, size=(600,650), dpi = 300)
end

#############################################
##### Meta-regression-related functions #####
#############################################
function prep_fmnl_vars(df_res::DataFrame)
	df_master = read_survey_master_data()
	add_vis_cols_to_master!(df_master)
	df_mas = @select(df_master, :key, :group_c, :mode, :cutoff_less90);
	clean_survey_key_names!(df_mas)

	df_res_nhm = @subset(df_res, :strat .== "non-home")
	df_ana = prepare_ana_for_fmnl(df_res_nhm)
	df_ana = leftjoin(df_ana, df_mas, on = :key);

	# TODO: check it.
	@transform!(df_ana,
		:mode = replace.(:mode, "OP" => "NA", "PI" => "P"),
		:cutoff_less90 = replace.(:cutoff_less90, "Yes" => "c-Yes", "No" => "c-No"))
	@transform!(df_ana,
		:key_pri = string.(rpad.(
				:key, 15), "/", :group_c, "/", :mode, "/", :cutoff_less90)
	)
	# Set a reference
	mode_cate = CategoricalArray(df_ana[:, :mode])
	levels!(mode_cate, ["P", "I", "O", "OP", "NA"])
	df_ana[:, :mode_cate] = mode_cate;
	return df_ana
end

function prepare_ana_for_fmnl(df_mer_nh)
	df_ana = unstack(df_mer_nh, :key, :model, :weight_waic)
	df_ana = leftjoin(df_ana,
		unique(df_mer_nh, [:key])[:, [:key, :n_sample]],
		on = :key)
	@transform!(df_ana, :logX = log10.(:n_sample))
	return df_ana
end

function add_vis_cols_to_master!(df_master::DataFrame)
	rep_dic = Dict("No" => "G-No", "Yes" => "G-Yes", missing => "NA")
	df_master[!, :group_c] = replace(df_master.group_contacts, rep_dic...)
	rep_dic = Dict(
		"Paper" => "P", "Interview" => "I", "Online" => "O",
		"Online and Paper" => "OP", "missing" => "NA",
		"Paper and Interview" => "PI",
	)
	df_master[!, :mode] = replace(df_master.mode_of_questionnaire, rep_dic...)
	nothing
end

"""This convert wide format probs to cumsum according to the model names.
"""
function create_tab_cum(df_tab::DataFrame, models::Vector)
	df_tab_cum = copy(df_tab)
	for i in 1:length(models)
		# sum of each model weights for stacked bar plot.
		df_tab_cum[!, models[i]] = sum(eachcol(df_tab[:, models[begin:i]]))
		df_tab_cum[!, models[i]] = round.(df_tab_cum[!, models[i]], digits = 2) .* 100
	end
	df_tab_cum
end

"""
Args:
- df_ana: long-format of dataframe containing `key`, `model` nad `weight_waic`.
"""
function create_stacked_bar_weights(df_ana::DataFrame, models, df_n_obs::DataFrame)
	df_tab = unstack(df_ana, :key, :model, :weight_waic)
	df_tab_cum = create_tab_cum(df_tab, models)
	df_tab_cum = @pipe leftjoin(df_tab_cum, df_n_obs, on = :key) |>
					   sort(_, :all; rev = false)
	df_tab_cum
end

function plot_bar_waic_pretty(df_obs::DataFrame, df_res::DataFrame, df_EVI::DataFrame)
	df_ana = prep_fmnl_vars(df_res)
	sort!(df_ana, :n_sample)

	ytk = ["" for _ in 1:nrow(df_ana)];  #df_ana[:, :key];
	pls = plot_bar_waic(df_obs, df_res, ytk = ytk, df_EVI = df_EVI)
	annotate!(pls[1], (1.0, 1.04), text("Non-home contacts", :black, :left, :center, 12, "Helvetica"))
	annotate!(pls[3], (1.0, 1.04), text("Home contacts", :black, :left, :center, 12, "Helvetica"))

	x_base = -37.0
	dx = -36
	dy = 1.5
	cutoff = replace.(df_ana[:, :cutoff_less90], "c-Yes" => "≤90", "c-No" => "91+")
	table_styl_annotate!(pls[1], cutoff, "Cap of\nanswers", x_base; dy = dy)
	s_mode = replace.(df_ana[:, :mode], "P" => "Paper", "I" => "Interview", "O" => "Online")
	table_styl_annotate!(pls[1], s_mode, "Survey\nmode", x_base + dx; dy = dy)
	grp_cate = replace.(df_ana[:, :group_c], "G-No" => "No", "G-Yes" => "Yes")
	table_styl_annotate!(pls[1], grp_cate, "Group\ncontacts", x_base + 2*dx; dy = dy)
	table_styl_annotate!(pls[1], df_ana[:, :n_sample], "Sample\nsize", x_base + 3*dx; dy = dy)
	df_tmp = copy(df_ana)
	table_styl_annotate!(pls[1], df_tmp[:, :key], "Study", x_base + 4*dx - 33)
	plot!(pls[1], left_margin = 80Plots.mm)

	annotate!(pls[1], x_base*5 - 20, nrow(df_ana) + 3.0, text("A", :black, :left, :center, 18, "Helvetica"))
	annotate!(pls[1], (0.1, 1.1), text("B", :black, :left, :center, 18, "Helvetica"))
	annotate!(pls[3], (0.1, 1.1), text("C", :black, :left, :center, 18, "Helvetica"))
	layout = @layout [a b c d]
	return plot(pls..., layout = layout,
		right_margin = 5.0Plots.mm,
		top_margin = 15.0Plots.mm,
		size = (1000, 600),
		dpi = 200, fig = :png,
	)
end

function plot_bar_waic(df_obs, df_res; ytk = nothing, df_EVI = nothing)
	model_names = get_model_names()

	tab_n_obs = unstack(df_obs, :key, :strat, :n_answer);

	df_res_tmp = @subset(df_res, :strat .== "home")
	df_tab_cum_hm = create_stacked_bar_weights(df_res_tmp, model_names, tab_n_obs;
	)
	df_res_tmp = @subset(df_res, :strat .== "non-home")
	df_tab_cum_nhm = create_stacked_bar_weights(df_res_tmp, model_names, tab_n_obs)
	df_tab_cum_nhm[:, :key_empty] .= "";

	ytk = isnothing(ytk) ? df_tab_cum_nhm[:, :key] : ytk
	labels = ["ZInf-NB", "ZInf-PLN", "ZInf-PLomax"]
	pl1 = plot_stacked_bar(df_tab_cum_hm, model_names;
		title = "", #"Home contacts",
		right_margin = 0Plots.mm,
		yticks = (1:length(ytk), ytk),
		labels = labels,
		legend_columns = 3,
		legend = (-0.8, -0.15), #-0.22),
	)

	pl2 = plot_stacked_bar(df_tab_cum_nhm, model_names;
		col_ytick = "key_empty",
		right_margin = -5.0Plots.mm,
		left_margin = -7.0Plots.mm,
		bottom_margin = 15Plots.mm,
		title = "", # "Non-home contacts",
		labels = nothing,
	)
	if isnothing(df_EVI)
		layout = @layout [a b]
		return plot(pl1, pl2, layout = layout, right_margin = 10Plots.mm, size = (800, 600))
	else
		df_EVI[:, :empty] .= ""
		df_EVI_hm = @subset(df_EVI, :strat .== "home")
		df_EVI_nhm = @subset(df_EVI, :strat .== "non-home")

		pl3 = plot_EVI_across_surveys(df_EVI_hm; col_ytick = "empty", title = "") #"Home contacts")
		pl4 = plot_EVI_across_surveys(df_EVI_nhm; col_ytick = "empty", title = "") #"Non-home contacts")

		adj_m = -9.0Plots.mm
		plot!(pl1, left_margin = adj_m)
		plot!(pl3, left_margin = adj_m)
		plot!(pl4, left_margin = adj_m)

		return [pl2, pl4, pl1, pl3]
	end
end

"""
Example:
```
df_sum = CSV.read("../dt_intermediate_refit/fit_summary.csv", DataFrame)
df_res = flag_minimum_IC(df_sum, :waic);
df_res_home = @subset(df_res, :strat .== "non-home")
df_tab_cum = create_stacked_bar_weights(df_res_home, model_names, tab_n_obs;
)
plot_stacked_bar(df_tab_cum, model_names;
	labels=model_names)
```
"""
function plot_stacked_bar(tab::DataFrame, stacked_cols::Vector;
	col_ytick = :key,
	right_margin = 60Plots.mm,
	legend = (1.15, 0.9),
	title = "",
	labels = nothing,
	xlabel = "WAIC weight (%)",
	kwds...,
)::Plots.Plot
	yticks_ = (1:nrow(tab), tab[:, col_ytick])
	pl = plot(;
		xlim = [0, 100],
		legend = legend,
		right_margin = right_margin,
		yticks = yticks_,
		xlabel = xlabel,
		title = title,
		kwds...,
	)
	color_lis = [7, 16, 13]
	for (i, c) in enumerate(stacked_cols)
		label = isnothing(labels) ? false : labels[i]
		bar!(pl, 1:nrow(tab), tab[:, c];
			permute = (:x, :y),
			label = label,
			color = color_lis[i],
			z_order = 1,
		)
	end
	return pl
end

function repeated_univariate_fmnl_reg(df_ana::DataFrame)
	pred, Y, x_names = one_hot_encoding_multi_vars(df_ana)
	chn1 = sample(model_fmnl(pred[:, [1, 2]], Y), NUTS(), 2000; progress = false)
	chn2 = sample(model_fmnl(pred[:, [1, 3]], Y), NUTS(), 2000; progress = false)
	chn3 = sample(model_fmnl(pred[:, [1, 4, 5]], Y), NUTS(), 2000; progress = false)
	chn4 = sample(model_fmnl(pred[:, [1, 6]], Y), NUTS(), 2000; progress = false)

	# Exclude intercept part from chain results.
	chn_res1 = extract_chain_info(chn1)[[2, 4], :]
	chn_res1[:, :var_name] .= x_names[2]
	chn_res2 = extract_chain_info(chn2)[[2, 4], :]
	chn_res2[:, :var_name] .= x_names[3]
	chn_res3 = extract_chain_info(chn3)[[2, 3, 5, 6], :]
	chn_res3[:, :var_name] .= x_names[4]
	chn_res3[3:4, :var_name] .= x_names[5]
	chn_res4 = extract_chain_info(chn4)[[2, 4], :]
	chn_res4[:, :var_name] .= x_names[6]

	chn_res = vcat(chn_res1, chn_res2, chn_res3, chn_res4)
	# Reorder to be used in `forestplot_fmnl_multi_vars`
	chn_res = chn_res[[1, 3, 5, 6, 9, 2, 4, 7, 8, 10], :]
	#chn_res = chn_res[[1, 3, 5, 7, 2, 4, 6, 8], :]
	return chn_res
end

"""
Args:
- chn_res: DataFrame from `extract_chain_info`, removing constant terms.
	first n_β should be for ZInf-NB over ZInf-PLomax.
"""
function forestplot_fmnl_multi_vars(chn_res::DataFrame, x_names::Vector; yticks = true)
	n_β = nrow(chn_res) ÷ 2
	est = chn_res[1:end, :median]
	xerr_l = chn_res[1:end, :xerr_l]
	xerr_u = chn_res[1:end, :xerr_u]

	yvals = [1, 3, 5, 6, 8]
	ytk_ind = [
		"Larger vs smaller",
		"Yes vs no",
		"Interview vs paper", "Online vs paper",
		"≤90 vs 91+",
	]
	ytk_cate = [
		boldstring("Log10 of sample size"),
		boldstring("Group contact"),
		boldstring("Survey mode"),
		boldstring("Cap of answers"),
	]
	ytk = yticks==true ? (yvals, ytk_ind) : (yvals, ["", "", "", "", ""])
	dy = 0.15
	pl = plot(xlabel = "Log ratio", yticks = ytk, ylim = [0.5, n_β + 4 + 0.5])
	if yticks==true
		[annotate!(pl, -9.3, y,
			text(ytk_cate[i], :black, :right, 9, "Helvetica"))
		 for (i, y) in enumerate([2, 4, 7, 9])]
	end
	inds = 1:n_β
	plot!(pl, est[inds], yvals .- dy, xerr = (xerr_l[inds], xerr_u[inds]),
		seriestype = :scatter, markershape = :square,
		label = " ZInf-NB over ZInf-PLomax",
	)
	inds = (n_β+1):(2*n_β)
	plot!(pl, est[inds], yvals .+ dy, xerr = (xerr_l[inds], xerr_u[inds]),
		seriestype = :scatter, markershape = :square,
		label = " ZInf-PLN over ZInf-PLomax",
	)
	vline!(pl, [0], color = :black, ls = :dash, label = false, z_order = 1, alpha = 0.5)
	plot!(pl, size = (400, 300), grid = false,
		legend = (0.1, -0.2), bottom_margin = 15Plots.mm)
end


function plot_meta_reg(chn_res, chn_res_multi)
	n_x = length(x_names)
	pl1 = forestplot_fmnl_multi_vars(chn_res, x_names[2:end])
	pl2 = forestplot_fmnl_multi_vars(chn_res_multi[Not([1, n_x+1]), :], x_names[2:end]; yticks = false)

	xtk = [-6.0, -3.0, 0.0, 3.0, 6.0]
	xtk = (xtk, xtk)
	kwds = (xlim = [-8.0, 6.5], xticks = xtk, titlefontsize = 12,
		ytickfontsize = 9, xtickfontsize = 9)
	plot!(pl1; legend_columns = 2, top_margin = 5Plots.mm, left_margin = 7Plots.mm,
		legendfontsize = 9, kwds...)
	plot!(pl2; legend = false, right_margin = 10Plots.mm,
		kwds...)
	annotate!(pl1, 0.0, 9.75, text("Univariate", :black, :left, :center, 11, "Helvetica"))
	annotate!(pl2, 0.0, 9.75, text("Multivariate", :black, :left, :center, 11, "Helvetica"))
	plot(pl1, pl2, layout = (1, 2), size = (600, 450))
end

#####################################
##### Fitting-related functions #####
#####################################
"""
Args:
- df_dds: Merged dds DataFrame.
"""
function fit_surveys(df_dds::DataFrame; dir_ = "../dt_intermediate")
	keys = df_dds.key |> unique
	for k in keys
		path = "$(dir_)/$(k)_chns.jld2"
		if isfile(path) == true
			println("Path present: ", path)
			continue
		end

		df_tmp = @subset(df_dds, :key .== k)
		dds = Dict(
			"home" => @subset(df_tmp, :strat .== "home") |> DegreeDist,
			"non-home" => @subset(df_tmp, :strat .== "non-home") |> DegreeDist,
		)
		println("Start fitting: $k")
		res = fit_hm_nhm_dds(dds)
		println("Finished fitting: $k")
		jldsave(path, result = res)
	end
end

function fit_hm_nhm_dds(dds::Dict)
	models = [model_ZeroInfNegativeBinomial, model_ZeroInfPoissonLogNormal,
		model_ZeroInfPoissonLomax]

	res = Dict("dds" => dds, "chns_home" => Dict(), "chns_non-home" => Dict())
	lis_k_model = [[k, model_func]
				   for k in ["home", "non-home"]
				   for model_func in models]
	Threads.@threads for v in lis_k_model
		k, model_func = v
		dd = dds[k]
		model = model_func(dd)
		med, chn = get_median_parms_from_model(model)
		chn = fit_model_with_forward_mode(model, 2000; iparms = med, progress = false)
		res["chns_$k"][get_dist_name_from_model(model_func)] = chn
	end
	return res
end

"""This function returns a median value of each model parameter
using a pathfinder for fast computation.

Args:
- model: A Turing model after sampling with data supplied.
"""
function get_median_parms_from_model(model)
	chn = pathfinder(model; ndraws = 1000).draws_transformed
	median_params = Dict(p => median(chn[p]) for p in names(chn, :parameters))
	return (median_params, chn)
end

function plot_pdf_validate(key, strat)
	res = load("../dt_intermediate/$(key)_chns.jld2")["result"];
	model_names = get_model_names()

	pl = plot(; xlim = [0, 50], ylim = [-5, 0])
	dd = res["dds"][strat]
	plot_pdf!(pl, dd, label = "Observed", markersize = 2.5, markerstrokewidth = 0.0)
	for m in model_names
		d = get_ZeroInfDist(res["chns_$(strat)"][m], m)
		plot_pdf!(pl, d, label = m)
	end
	pl
end

function plot_ccdf_validate(key, strat)
	res = load("../dt_intermediate/$(key)_chns.jld2")["result"];
	model_names = get_model_names()

	pl = plot(; xaxis = :log10, ylim = [-5, 0], xlim = [1, 10_000])
	dd = res["dds"][strat]
	plot_ccdf!(pl, dd, label = "Observed", markersize = 2.5, markerstrokewidth = 0.0)
	for m in model_names
		d = get_ZeroInfDist(res["chns_$(strat)"][m], m)
		plot_ccdf!(pl, d, label = m)
	end
	pl
end

function plot_pdf_ccdf_validate(key)
	pl1 = plot_pdf_validate(key, "home")
	pl2 = plot_ccdf_validate(key, "home")
	pl3 = plot_pdf_validate(key, "non-home")
	pl4 = plot_ccdf_validate(key, "non-home")
	plot(pl1, pl2, pl3, pl4, layout = (2, 2), size = (800, 800), title = key)
end

function plot_ccdf_poisson_lomax!(
	pl::Plots.Plot, key::String, strat::String, label::String, color=:black)
    res = load("../dt_intermediate/$(key)_chns.jld2")["result"]
    dd = res["dds"][strat]
    plot_ccdf!(pl, dd, label = label,
        color = color,
        markersize = 2.5,
        markerstrokewidth = 0.0)

    # Plot fitted ZeroInf-PoissonLomax
    d = get_ZeroInfDist(res["chns_$(strat)"]["ZeroInfPoissonLomax"], "ZeroInfPoissonLomax")
    plot_ccdf!(pl, d,
        label = "",
        color = color,
        linewidth = 1,
        linestyle = :dash)

    return pl
end

function plot_multi_ccdf_poisson_lomax(keys_; kwds...)
	pl = plot(; xlim=[1, 10_000], ylim=[-4, 0], kwds...)
	for (i, k) in enumerate(keys_)
		label = replace(k, "_" => " ") |> transform_comix2
		plot_ccdf_poisson_lomax!(pl, k, "non-home", label, i)
	end
	return pl
end