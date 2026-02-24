using Interpolations
using KernelDensity

"""Kernel density functions to use the posteiror samples as priors.
See https://discourse.julialang.org/t/how-to-use-a-distribution-estimated-via-kernel-density-estimation-kde-as-a-prior-in-turing-jl/124504/2
for the original code.
"""
struct InterpKDEDistribution{T <: Real, K <: KernelDensity.InterpKDE} <: ContinuousUnivariateDistribution
	kde::K
end

function InterpKDEDistribution(k::KernelDensity.InterpKDE)
	T = eltype(k.kde.x)
	return InterpKDEDistribution{T, typeof(k)}(k)
end
InterpKDEDistribution(k::KernelDensity.UnivariateKDE) = InterpKDEDistribution(KernelDensity.InterpKDE(k))
Distributions.minimum(d::InterpKDEDistribution) = first(only(Interpolations.bounds(d.kde.itp.itp)))
Distributions.maximum(d::InterpKDEDistribution) = last(only(Interpolations.bounds(d.kde.itp.itp)))
Distributions.pdf(d::InterpKDEDistribution, x::Real) = pdf(d.kde, x)
Distributions.logpdf(d::InterpKDEDistribution, x::Real) = log(pdf(d, x))

function Random.rand(rng::Random.AbstractRNG, d::InterpKDEDistribution)
	(; kde) = d
	knots = Interpolations.knots(kde.itp.itp)
	cdf = cumsum(pdf.(Ref(kde), knots))
	u = rand(rng)
	return knots[findlast(u .> cdf)]
end