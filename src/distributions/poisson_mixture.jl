##### Poisson Mixture distributions #####
abstract type PoissonMixture <: DiscreteUnivariateDistribution end
Base.length(p::PoissonMixture)=1
Base.iterate(p::PoissonMixture) = (p, nothing)
Base.iterate(p::PoissonMixture, nothing) = nothing

# PARAMETRIC, not `m::Real; k::Real` — this is load-bearing for AD PERFORMANCE, not tidying.
# DO NOT revert it to abstract fields.
#
# `NegBin` is the Stage-1 unweighted contact likelihood (`joint_model.jl` `_cell_moments!` →
# `calculate_loglikelihood`), so it is constructed 49×Tn times per gradient evaluation. Abstract
# `::Real` fields make it type-unstable, which ReverseDiff shrugs off (it boxes into TrackedReal
# regardless) but which **Mooncake cannot** — Mooncake is source-to-source and needs the primal to
# infer. Measured 2026-08-05, Stage-1 model at origin 2021-05-09, 402 unconstrained dims:
#
#            abstract ::Real fields          parametric (this)
#   negbin   2.7 grad/s  (ReverseDiff 41.8 — a 15× REGRESSION)   482 grad/s  (10.9× FASTER)
#
# For contrast the hurdle-Weibull path, which uses the concrete `Distributions.Weibull`, was
# already 9× faster under Mooncake before this change — that asymmetry is what indicted the struct.
# The log-density is unchanged to the last bit (-444771.5765202815), so this is purely a typing fix.
#
# The other legacy distributions here still use `::Real`. They are off the framework path and are
# fitted with ForwardDiff, so they are deliberately left alone (minimal impact). `ZeroTruncNegBin`
# (`zerotrunc.jl`) wraps a `NegBin` field and so is now abstractly-typed in that field — harmless,
# as it too is legacy-only; revisit only if it ever joins an AD hot path.
# Both fields stay independently parameterised so a promotion never silently widens the other.
Base.@kwdef struct NegBin{Tm<:Real,Tk<:Real} <: PoissonMixture
	m::Tm
	k::Tk
end

Base.@kwdef struct PoissonLogNormal <: PoissonMixture
	μ::Real
	σ::Real
end

Base.@kwdef struct PoissonLomax <: PoissonMixture
	α::Real # shape
	θ::Real # scale
end

@memoize function Distributions.ccdf(
    d::PoissonMixture,
    k::Int64;
    k_max=20_000
)
    if k > k_max
        error("Increase k_max")
	elseif k == k_max
		return pdf(d, k)
	else
		return Distributions.ccdf(d, k+1; k_max=k_max) + pdf(d, k)
    end
end

mean100(d::PoissonMixture) = sum(k * pdf(d, k) for k in 0:100)
var100(d::PoissonMixture) = sum( (k-mean(d))^2 * pdf(d, k) for k in 0:100)

##### Negative Binomial distribution #####
Distributions.mean(d::NegBin) = d.m
Distributions.cov(d::NegBin) = sqrt(1/d.m + 1/d.m + 1/d.k)
Distributions.var(d::NegBin) = d.m + d.m^2/d.k

function Distributions.logpdf(d::NegBin, y::Int64)
	return loggamma(d.k+y) - loggamma(d.k) - loggamma(y+1) +
		   d.k*log(d.k/(d.m+d.k)) + y*log(d.m/(d.m+d.k))
end

function Distributions.rand(d::NegBin, n::Int64 = 1)
	negbin = NegativeBinomial(d.k, d.k/(d.m + d.k))
	return rand(negbin, n)
end
Distributions.rand(d::NegBin) = rand(d, 1)[1]

##### Piosson-Lognormal distributions #####
Distributions.mean(d::PoissonLogNormal) = exp(d.μ + d.σ^2 / 2)
Distributions.cov(d::PoissonLogNormal) = sqrt(1 / mean(d) + exp(d.σ^2) - 1)
Distributions.var(d::PoissonLogNormal) = mean(d) + mean(d)^2 * (exp(d.σ^2) - 1)

function Distributions.rand(d::PoissonLogNormal, n::Int64 = 1)
	lognorm = LogNormal(d.μ, d.σ)
	return rand.(Poisson.(rand(lognorm, n)))
end
Distributions.rand(d::PoissonLogNormal) = rand(d, 1)[1]

function Distributions.logpdf(d::PoissonLogNormal, k::Int64)
	@unpack μ, σ = d
	function integrand(x)
		l_int = -loggamma(k+1) + (k-1)*log(x) - x - (log(x)-μ)^2/(2σ^2)
		return exp(l_int)
	end
	# Avoid the integrand to be 0.
	k_tmp = k == 0 ? 1 : k
	int, err = quadgk(u -> integrand(u * k_tmp) * k_tmp, 1e-6, Inf, rtol = 1e-8)
	ret = log(int * 1/(σ*√(2π)))
	return ret
end

##### Lomax distribution #####
struct Lomax <: ContinuousUnivariateDistribution
	α::Real # shape
	θ::Real # scale
end
Distributions.rand(d::Lomax) = (1 / rand() ^ (1/d.α) - 1) * d.θ
Distributions.rand(rng::AbstractRNG, d::Lomax) = rand(rng)
Distributions.minimum(d::Lomax) = 0.0
Distributions.maximum(d::Lomax) = Inf

function Distributions.logpdf(d::Lomax, x::Real)::Real
	@unpack α, θ = d
	if x < 0
		return -Inf
	else
		return log(α) - log(θ) - (α+1)*log(1+x/θ)
	end
end
Distributions.pdf(d::Lomax, x::Real) = exp(logpdf(d, x))
Distributions.ccdf(d::Lomax, x::Real) = (1 + x/d.θ)^(-d.α)
Distributions.cdf(d::Lomax, x::Real) = 1 - ccdf(d, x)

##### Poisson-Lomax distributions #####
Distributions.mean(d::PoissonLomax) = d.α > 1 ? d.θ / (d.α-1) : NaN
function Distributions.var(d::PoissonLomax)
	@unpack α, θ = d
	if α <= 1
		return NaN
	elseif α <= 2
		return Inf
	else
		return mean(d) + θ^2 * α / (α - 1)^2 / (α - 2)
	end
end

function Distributions.cov(d::PoissonLomax)
	@unpack α, θ = d
	if α <= 1
		return NaN
	elseif α <= 2
		return Inf
	else
		return √(1 / mean(d) + α / (α - 2))
	end
end

function Distributions.rand(d::PoissonLomax, n::Int64 = 1)
	@unpack α, θ = d
	y = rand(n)
	x = (1 ./ y .^ (1/α) .- 1) .* θ
	return rand.(Poisson.(x))
end
Distributions.rand(d::PoissonLomax) = rand(d, 1)[1]

function Distributions.logpdf(d::PoissonLomax, k::Int64)
	@unpack α, θ = d
	function integrand(λ)
		l_int = k*log(λ) - λ - loggamma(k+1) + log(α) + α*log(θ) - (α+1)*log(λ+θ)
		return (exp(l_int))
	end
	# Avoid the integrand to be 0.
	k_tmp = k == 0 ? 1 : k
	int, err = quadgk(u -> integrand(u * k_tmp) * k_tmp, 1e-6, Inf, rtol = 1e-8)
	return log(int)
end
Distributions.cdf(d::PoissonLomax, k::Int64)::Real = sum(pdf(d, i) for i in 0:k)


#####################################
###### Mixture distributions ########
#####################################
struct Lomax <: ContinuousUnivariateDistribution
	α::Real # shape
	θ::Real # scale
end
Distributions.rand(d::Lomax) = (1 / rand() ^ (1/d.α) - 1) * d.θ
Distributions.rand(rng::AbstractRNG, d::Lomax) = rand(rng)
Distributions.minimum(d::Lomax) = 0.0
Distributions.maximum(d::Lomax) = Inf

function Distributions.logpdf(d::Lomax, x::Real)::Real
	@unpack α, θ = d
	if x < 0
		return -Inf
	else
		return log(α) - log(θ) - (α+1)*log(1+x/θ)
	end
end
Distributions.pdf(d::Lomax, x::Real) = exp(logpdf(d, x))
Distributions.ccdf(d::Lomax, x::Real) = (1 + x/d.θ)^(-d.α)
Distributions.cdf(d::Lomax, x::Real) = 1 - ccdf(d, x)
