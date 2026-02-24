# Wrapper for the ZeroTruncPoissonMixture
function ZeroTrunc(d::DataType, p1, p2)
	if d == NegBin
		return ZeroTruncNegBin(; m = p1, k = p2)
	elseif d == PoissonLogNormal
		return ZeroTruncPoissonLogNormal(; μ = p1, σ = p2)
	elseif d == PoissonLomax
		return ZeroTruncPoissonLomax(; α = p1, θ = p2)
	else
		error("Unsupported distribution type for ZeroTrunc")
	end
end

function convert_ZeroInf_to_ZeroTrunc(d::ZeroInfDist)
	if typeof(d.d) == NegBin
		return ZeroTruncNegBin(; m = d.d.m, k=d.d.k)
	elseif typeof(d.d) == PoissonLogNormal
		return ZeroTruncPoissonLogNormal(; μ = d.d.μ, σ=d.d.σ)
	elseif typeof(d.d) == PoissonLomax
		return ZeroTruncPoissonLomax(; α = d.d.α, θ=d.d.θ)
	else
		error("Unsupported distribution type for convert_ZeroInf_to_ZeroTrunc")
	end
end
