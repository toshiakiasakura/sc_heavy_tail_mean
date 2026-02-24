Base.@kwdef mutable struct ZeroInfDist <: DiscreteUnivariateDistribution
	π0::Real
	d::DiscreteUnivariateDistribution
end

##### ZeroInfDist #####
Distributions.mean(d::ZeroInfDist) = (1 - d.π0) * mean(d.d)

@memoize function Distributions.logpdf(d::ZeroInfDist, y::Int64)
	log_one_minus_pi0 = log1p(-d.π0)
	if y == 0
		return logaddexp(log(d.π0), log_one_minus_pi0 + logpdf(d.d, 0))
	else
		return logpdf(d.d, y) + log_one_minus_pi0
	end
end
Distributions.pdf(d::ZeroInfDist, y::Int64) = exp(logpdf(d, y))

function Distributions.rand(d::ZeroInfDist)
	if rand() < d.π0
		return 0
	else
		return rand(d.d)
	end
end
Distributions.rand(d::ZeroInfDist, n::Int64) = [rand(d) for _ in 1:n]

###### Convoluted of home and non-home contact distribution for all contact distribution #####
Base.@kwdef struct ZeroInfConvolutedDist <: DiscreteUnivariateDistribution
	d_hm::ZeroInfDist
	d_nhm::ZeroInfDist
	k_h_max::Int64
end
Distributions.logpdf(d::ZeroInfConvolutedDist, k::Int64) = log(pdf(d, k))

@memoize function Distributions.pdf(conv_d::ZeroInfConvolutedDist, k::Int64)
	@unpack d_hm, d_nhm, k_h_max = conv_d
	k_max = minimum((k, k_h_max))

	p = 0.0
	for i in 0:k_max
		lp = logpdf(d_hm, i) + logpdf(d_nhm, k - i)
		p += exp(lp)
	end
	return p
end

function Distributions.ccdf(
    d::ZeroInfConvolutedDist,
    k::Int64;
    k_max=10_000
)
    if k > k_max
        error("Increase k_max")
    end
    return sum(pdf(d, i) for i in k:k_max)
end

Base.@kwdef struct ZeroTruncConvolutedDist <: DiscreteUnivariateDistribution
	conv_d::ZeroInfConvolutedDist
	p0::Real = pdf(conv_d, 0)
end
Distributions.pdf(d::ZeroTruncConvolutedDist, k::Int64) = pdf(d.conv_d, k) / (1 - d.p0)
Distributions.logpdf(d::ZeroTruncConvolutedDist, k::Int64) = log(pdf(d,k))
Distributions.ccdf(d::ZeroTruncConvolutedDist, k::Int64) = ccdf(d.conv_d, k) / (1 - d.p0)