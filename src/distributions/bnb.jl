##### Beta Negative Binomial distribution #####
#
# BNB  — Wikipedia parameterisation (α, β, r)
# BNB2 — Reparameterised form (m, v, η) where
#          m = rβ/(α-1)  (mean),  v = α-1,  η = (β+r)/√(βr) - 2  (overdispersion)
#
# Reference: inst/BNB.md

##### BNB — (α, β, r) parameterisation #####
Base.@kwdef struct BNB <: DiscreteUnivariateDistribution
    α::Real   # Beta shape 1
    β::Real   # Beta shape 2
    r::Real   # NegBin dispersion
end
Base.length(d::BNB) = 1
Base.iterate(d::BNB) = (d, nothing)
Base.iterate(d::BNB, ::Nothing) = nothing

# PMF in log-space:
#   log f(k|α,β,r) = [loggamma(r+k) - loggamma(k+1) - loggamma(r)]
#                  + [loggamma(α+r) + loggamma(β+k) - loggamma(α+β+r+k)]
#                  - [loggamma(α) + loggamma(β) - loggamma(α+β)]
function Distributions.logpdf(d::BNB, k::Int64)
    @unpack α, β, r = d
    if k < 0 || α <= 0 || β <= 0 || r <= 0
        return -Inf
    end
    return (loggamma(r + k) - loggamma(k + 1) - loggamma(r)) +
           (loggamma(α + r) + loggamma(β + k) - loggamma(α + β + r + k)) +
           (loggamma(α + β) - loggamma(α) - loggamma(β))
end
Distributions.pdf(d::BNB, k::Int64) = exp(logpdf(d, k))

# Mean: E[X] = rβ/(α-1), requires α > 1
function Distributions.mean(d::BNB)
    d.α > 1 ? d.r * d.β / (d.α - 1) : NaN
end

# Variance: rβ(α-1+r)(α+β-1) / ((α-1)²(α-2)), requires α > 2
function Distributions.var(d::BNB)
    @unpack α, β, r = d
    if α <= 1
        return NaN
    elseif α <= 2
        return Inf
    else
        return r * β * (α - 1 + r) * (α + β - 1) / ((α - 1)^2 * (α - 2))
    end
end

# Coefficient of variation (CV), matching codebase convention for cov()
function Distributions.cov(d::BNB)
    m = mean(d)
    v = var(d)
    (isnan(m) || isinf(v)) ? NaN : sqrt(v) / m
end

# CCDF: recursive memoized accumulation (same pattern as PoissonMixture)
@memoize function Distributions.ccdf(d::BNB, k::Int64; k_max = 20_000)
    if k > k_max
        error("Increase k_max")
    elseif k == k_max
        return pdf(d, k)
    else
        return Distributions.ccdf(d, k + 1; k_max = k_max) + pdf(d, k)
    end
end
Distributions.cdf(d::BNB, k::Int64) = sum(pdf(d, i) for i in 0:k)

# Sampling via the Beta-NegBin mixture construction:
#   p ~ Beta(α, β),  X|p ~ NegativeBinomial(r, p)
function Distributions.rand(d::BNB)
    @unpack α, β, r = d
    p = rand(Beta(α, β))
    return rand(NegativeBinomial(r, p))
end
Distributions.rand(d::BNB, n::Int64) = [rand(d) for _ in 1:n]

##### BNB2 — (m, v, η) reparameterisation #####
#
# Inverse mapping:
#   α = v + 1
#   β = √(mv)/2 · (η + 2 + √(η²+4η))
#   r = √(mv)/2 · (η + 2 − √(η²+4η))
#
# Constraints: m > 0, v > 0, η ≥ 0  →  r > 0, β > 0, α > 1

Base.@kwdef struct BNB2 <: DiscreteUnivariateDistribution
    m::Real        # mean
    v::Real        # α - 1  (> 0)
    η::Real        # overdispersion tail index  (≥ 0)
    d::BNB = begin
        α = v + 1
        sq = sqrt(m * v) / 2
        disc = sqrt(η^2 + 4η)
        β = sq * (η + 2 + disc)
        # Use algebraic identity (η+2 - disc)*(η+2 + disc) = 4  to avoid
        # catastrophic cancellation when η is large:
        r = sq * 4 / (η + 2 + disc)
        BNB(; α = α, β = β, r = r)
    end
end
Base.length(d::BNB2) = 1
Base.iterate(d::BNB2) = (d, nothing)
Base.iterate(d::BNB2, ::Nothing) = nothing

# Delegate all methods to the inner BNB
Distributions.logpdf(d::BNB2, k::Int64) = logpdf(d.d, k)
Distributions.pdf(d::BNB2, k::Int64)    = pdf(d.d, k)
Distributions.mean(d::BNB2)             = mean(d.d)
Distributions.var(d::BNB2)              = var(d.d)
Distributions.cov(d::BNB2)             = cov(d.d)
Distributions.ccdf(d::BNB2, k::Int64; k_max = 20_000) = ccdf(d.d, k; k_max = k_max)
Distributions.cdf(d::BNB2, k::Int64)   = cdf(d.d, k)
Distributions.rand(d::BNB2)            = rand(d.d)
Distributions.rand(d::BNB2, n::Int64)  = rand(d.d, n)
