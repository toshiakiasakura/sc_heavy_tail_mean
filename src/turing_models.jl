###################################
###### ZeroInf fitting models #####
###################################
function fit_model_with_forward_mode(model, n_sample; iparms = Dict(), progress = true)
	Random.seed!(1236)
	sampler = NUTS()
	if isempty(iparms)
		# Use Pathfinder to find good initial parameters (mean of approximate posterior)
		pf_result = pathfinder(model; progress = progress)
		param_names = names(pf_result.draws_transformed, :parameters)
		mean_values = [mean(pf_result.draws_transformed[:, p, :]) for p in param_names]
		pf_mean_nt = NamedTuple(zip(param_names, mean_values))
		init_strategy = DynamicPPL.InitFromParams(pf_mean_nt)
	else
		init_strategy = DynamicPPL.InitFromParams(NamedTuple(iparms))
	end
	@time chn = sample(model, sampler, n_sample;
		progress = progress, initial_params = init_strategy)
end

@model function model_ZeroInfNegativeBinomial(dd::DegreeDist)
	log_m_ga ~ Normal(0.0, 2)
	log_k_ga ~ Normal(0.0, 1)
	m = exp(log_m_ga)
	k = exp(log_k_ga)
	π0 ~ Beta(1.5, 1.5)
	zeroInf = ZeroInfDist(π0, NegBin(m, k))
	ll = calculate_loglikelihood(dd, zeroInf)
	Turing.@addlogprob! ll
end

@model function model_ZeroInfPoissonLogNormal(dd::DegreeDist)
	μ_obs_ln ~ Normal(0, 1.0)
	log_σ_ln ~ Normal(0, 1.0)
	π0 ~ Beta(1.5, 1.5)
	cond = is_outside_cutoff_PoissonLogNormal(μ_obs_ln, log_σ_ln)
	if cond == true
		Turing.@addlogprob! -Inf # Reject sample point immediately
		return
	end

	μ_ln, σ_ln = PoissonLogNormal_convert(μ_obs_ln, log_σ_ln)
	zeroInf = ZeroInfDist(π0, PoissonLogNormal(μ_ln, σ_ln))
	ll = calculate_loglikelihood(dd, zeroInf)
	Turing.@addlogprob! ll
end

@model function model_ZeroInfPoissonLomax(dd::DegreeDist)
	log_α_lo ~ Normal(0, 1.0)
	log_β_lo ~ Normal(0, 2.0)
	α = exp(log_α_lo)
	β = exp(log_β_lo)
	π0 ~ Beta(1.5, 1.5)
	cond = is_outside_cutoff_PoissonLomax(log_α_lo, log_β_lo)
	if cond == true
		Turing.@addlogprob! -Inf # Reject sample point immediately
		return
	end
	zeroInf = ZeroInfDist(π0, PoissonLomax(α, β))
	ll = calculate_loglikelihood(dd, zeroInf)
	Turing.@addlogprob! ll
end

"""Fit a Beta Negative Binomial (BNB2 reparameterisation) to a `DegreeDist`.

Parameters sampled in log-space:
- `log_m_bnb`: log mean  (m > 0)
- `log_v_bnb`: log v = α-1  (v > 0, so α > 1)
- `log_η_bnb`: log tailness index η  (η > 0)

Example:
```julia
dd = merge_dd(@subset(df_dds_nhm, in.(:key, Ref(period_keys))))
chn = fit_model_with_forward_mode(model_BNB2(dd), 1000)
d   = get_BNB2(chn)
```
"""
@model function model_BNB2(dd::DegreeDist)
	log_m_bnb ~ Normal(log(5.0), 1.5)
	log_v_bnb ~ Normal(0.0, 1.5)
	log_η_bnb ~ Normal(0.0, 1.5)

	m = exp(log_m_bnb)
	v = exp(log_v_bnb)
	η = exp(log_η_bnb)
	dist = BNB2(; m = m, v = v, η = η)
	ll = calculate_loglikelihood(dd, dist)
	Turing.@addlogprob! ll
end

@model function model_ZeroInfConvDist(dd_all::DegreeDist, dd_hm::DegreeDist, prior_dic)
	μ_obs_ln ~ prior_dic["hm_p1"]
	log_σ_ln ~ prior_dic["hm_p2"]
	π0_ln ~ Beta(1.5, 1.5)

	log_α_lo ~ prior_dic["nhm_p1"]
	log_β_lo ~ prior_dic["nhm_p2"]
	π0_lo ~ Beta(1.5, 1.5)

	μ_ln, σ_ln = PoissonLogNormal_convert(μ_obs_ln, log_σ_ln)
	α_lo, β_lo = exponential_convert(log_α_lo, log_β_lo)

	d_hm = ZeroInfDist(π0_ln, PoissonLogNormal(μ_ln, σ_ln))
	d_nhm = ZeroInfDist(π0_lo, PoissonLomax(α_lo, β_lo))
	conv_dist = ZeroInfConvolutedDist(d_hm, d_nhm, maximum(dd_hm))
	ll = calculate_loglikelihood(dd_all, conv_dist)
	Turing.@addlogprob! ll
end


###########################################
###### Validation of estimation methods ###
###########################################
@model function model_PoissonLogNormal(dd::DegreeDist)
	μ_obs_ln ~ Normal(0, 1.0)
	log_σ_ln ~ Normal(0, 1.0)
	μ_ln, σ_ln = PoissonLogNormal_convert(μ_obs_ln, log_σ_ln)

	dist = PoissonLogNormal(μ_ln, σ_ln)
	ll = calculate_loglikelihood(dd, dist)
	Turing.@addlogprob! ll
end

@model function model_PoissonLomax(dd::DegreeDist)
	log_α_lo ~ Normal(0, 1.0)
	log_β_lo ~ Normal(0, 2.0)
	α = exp(log_α_lo)
	β = exp(log_β_lo)

	dist = PoissonLomax(α, β)
	ll = calculate_loglikelihood(dd, dist)
	Turing.@addlogprob! ll
end

@model function model_hierarchical_PoissonLogNormal(x::Vector{Int64})
	μ_obs_ln ~ Normal(0, 1.0)
	log_σ_ln ~ Normal(0, 1.0)
	μ_ln, σ_ln = PoissonLogNormal_convert(μ_obs_ln, log_σ_ln)
	λ ~ filldist(LogNormal(μ_ln, σ_ln), length(x))
	for i in eachindex(x)
		x[i] ~ Poisson(λ[i])
	end
end

@model function model_hierarchical_PoissonLomax(x::Vector{Int64})
	log_α_lo ~ Normal(0, 1.0)
	log_β_lo ~ Normal(0, 2.0)
	α = exp(log_α_lo)
	β = exp(log_β_lo)
	λ ~ filldist(Lomax(α, β), length(x))
	for i in eachindex(x)
		x[i] ~ Poisson(λ[i])
	end
end

##################################################
###### Fractional multinomial distributions ######
##################################################

"""Fractional multinomial distributions.
"""
@model function model_fmnl(x::Matrix, y::Matrix)
	n_x, n_col_x = size(x)
	β1 ~ filldist(Normal(0, 3), n_col_x)
	β2 ~ filldist(Normal(0, 3), n_col_x)

	η1 = x * β1
	η2 = x * β2
	scale = [logsumexp([0.0, η1[i], η2[i]]) for i in 1:n_x]
	log_α1 = η1 .- scale
	log_α2 = η2 .- scale
	log_α3 = - scale
	log_αs = hcat(log_α1, log_α2, log_α3)
	for i in 1:n_x
		Turing.@addlogprob! sum(y[i,:] .* log_αs[i, :])
	end
end

function calculate_fmnl_probs(x::AbstractMatrix, β1::AbstractVector, β2::AbstractVector)
	η1 = x * β1
	η2 = x * β2
	scale = [logsumexp([0.0, η1[i], η2[i]]) for i in 1:size(x, 1)]
	log_α1 = η1 .- scale
	log_α2 = η2 .- scale
	log_α3 = - scale
	αs = hcat(log_α1, log_α2, log_α3) .|> exp
    return αs
end

"""
Args:
- df_ana: DataFrame after `prepare_ana_for_fmnl`
"""
function one_hot_encoding_multi_vars(df_ana::DataFrame)
	model_names = get_model_names()
	df_ana[:, :y_dummy] .= 1
	f = @formula(y_dummy ~  1 + log10(n_sample) + group_c + mode_cate + cutoff_less90)
	f = apply_schema(f, schema(f, df_ana))
	f |> display
	resp, pred = modelcols(f, df_ana)
	_, x_names = coefnames(f);
	Y = df_ana[:, model_names] |> Matrix{Float64};
	return (pred, Y, x_names)
end

function get_β_med(chn::Chains, n_x::Int)
	chn_res = extract_chain_info(chn)
	β1_med = chn_res[1:n_x, :median]
	β2_med = chn_res[(n_x+1):(2*n_x), :median]
	return (β1_med, β2_med)
end

# TODO: df_obs is needed
function pred_fmnl_multi_vars(chn::Chains, pred::Matrix)
	n_x = size(pred, 2)
	β1_med, β2_med = get_β_med(chn, n_x)
	pred_waic = calculate_fmnl_probs(pred, β1_med, β2_med)
	df_pred = DataFrame(pred_waic, model_names)
	df_pred_cum = create_tab_cum(df_pred, model_names);
	df_pred_cum[:, :key] = df_ana[:, :key];
	df_all_obs = @subset(df_obs, :strat .== "all")
	df_pred_cum = @pipe leftjoin(df_pred_cum, df_all_obs, on = :key) |>
						sort(_, :n_answer; rev = false);
	return df_pred_cum
end

##########################################
##### Extreme value index estimation #####
##########################################

@model function model_GeneralizedPareto(x::Vector{Float64})
	σ ~ Gamma(1, 1)
	ξ ~ Normal(0, 5)
	GP = GeneralizedPareto(σ, ξ)
	for i in eachindex(x)
		x[i] ~ GP
	end
end

function fit_model_GP(x::Vector{Float64}, model::Function;
	n_samples = 1000, progress = false)::Chains
	return sample(model(x), NUTS(max_depth = 10), n_samples;
		progress = progress)
end