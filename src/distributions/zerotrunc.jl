abstract type ZeroTruncPoissonMixture <: DiscreteUnivariateDistribution end
Base.length(p::ZeroTruncPoissonMixture)=1
Base.iterate(p::ZeroTruncPoissonMixture) = (p, nothing)
Base.iterate(p::ZeroTruncPoissonMixture, nothing) = nothing

"""
Example:
```
negbin = ZeroTruncNegBin(;m=m, k=k)
```
"""
Base.@kwdef struct ZeroTruncNegBin <: ZeroTruncPoissonMixture
	m::Real
	k::Real
	d::NegBin = NegBin(m, k)
	p0::Real = pdf(d, 0)
end

Base.@kwdef struct ZeroTruncPoissonLogNormal <: ZeroTruncPoissonMixture
	μ::Real
	σ::Real
	d::PoissonLogNormal = PoissonLogNormal(μ, σ)
	p0::Real = pdf(d, 0)
end

Base.@kwdef struct ZeroTruncPoissonLomax <: ZeroTruncPoissonMixture
	α::Real
	θ::Real
	d::PoissonLomax = PoissonLomax(α, θ)
	p0::Real = pdf(d, 0)
end

function Distributions.logpdf(d::ZeroTruncPoissonMixture, y::Int64)
	return logpdf(d.d, y) - log(1 - d.p0)
end
Distributions.pdf(d::ZeroTruncPoissonMixture, y::Int64) = exp(logpdf(d, y))

function Distributions.cdf(d::ZeroTruncPoissonMixture, k::Int64)
	if k == 0
		error("k must be greater than 0 for ZeroTruncPoissonMixture")
	end
	return (cdf(d.d, k) - d.p0) / (1 - d.p0)
end

function Distributions.ccdf(d::ZeroTruncPoissonMixture, k::Int64)
	if k == 1
		return 1
	end
	# NOTE: Avoid computation error for very small value.
	#       This only relates to visualisation purpose.
	#p = (1 - cdf(d.d, k - 1)) / (1-d.p0)
	p = ccdf(d.d, k) / (1-d.p0)
	return p < 0 ? NaN : p
end
Distributions.mean(d::ZeroTruncPoissonMixture) = mean(d.d) / (1 - d.p0)
Distributions.std(d::ZeroTruncPoissonMixture) = sqrt(var(d))
function Distributions.var(d::ZeroTruncPoissonMixture)
	Pu = 1 - d.p0
	m = mean(d.d)
	v = var(d.d)
	return ((m^2 + v) * Pu - m^2) / Pu^2
end

function Distributions.cov(d::ZeroTruncPoissonMixture)
	Pu = 1 - d.p0
	return sqrt((1 + cov(d.d)^2) * Pu - 1)
end

function Distributions.rand(d::ZeroTruncPoissonMixture)
	r = rand(d.d)[1]
	for i in 1:10_000
		if r == 0
			r = rand(d.d)[1]
		else
			return r
		end
	end
	error("Failed to generate a truncated one.")
end
Distributions.rand(d::ZeroTruncPoissonMixture, n::Int64) = [rand(d) for _ in 1:n]

##### Zero-Truncated BNB #####

Base.@kwdef struct ZeroTruncBNB <: ZeroTruncPoissonMixture
    α::Real
    β::Real
    r::Real
    d::BNB  = BNB(α, β, r)
	p0::Real = pdf(d, 0)
end

Base.@kwdef struct ZeroTruncBNB2 <: ZeroTruncPoissonMixture
    m::Real
    v::Real
    η::Real
    d::BNB2  = BNB2(; m=m, v=v, η=η)
	p0::Real = pdf(d, 0)
end
zero_trunc(bnb2::BNB2) = ZeroTruncBNB2(m = bnb2.m, v = bnb2.v, η = bnb2.η)

# BNB rand returns a scalar (not a vector like the Poisson-mixture variants),
# so override rand to avoid the [1] indexing in the abstract method.
function Distributions.rand(d::Union{ZeroTruncBNB, ZeroTruncBNB2})
    for _ in 1:10_000
        r = rand(d.d)
        r > 0 && return r
    end
    error("Failed to generate a non-zero sample from ZeroTruncBNB.")
end
Distributions.rand(d::Union{ZeroTruncBNB, ZeroTruncBNB2}, n::Int64) = [rand(d) for _ in 1:n]

"""Zero truncated convoluted distribution #####
This is based on ZeroInfConvolutedDist.
"""
Base.@kwdef struct ZeroTruncConvolutedDist <: DiscreteUnivariateDistribution
	conv_d::ZeroInfConvolutedDist
	p0::Real = pdf(conv_d, 0)
end
Distributions.pdf(d::ZeroTruncConvolutedDist, k::Int64) = pdf(d.conv_d, k) / (1 - d.p0)
Distributions.logpdf(d::ZeroTruncConvolutedDist, k::Int64) = log(pdf(d,k))
Distributions.ccdf(d::ZeroTruncConvolutedDist, k::Int64) = ccdf(d.conv_d, k) / (1 - d.p0)