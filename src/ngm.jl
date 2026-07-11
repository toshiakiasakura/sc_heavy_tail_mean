# ngm.jl — build the next-generation matrix from the age-pair contact-degree
# distribution. The NGM *builder* is the second swap axis (deterministic dispatch).
#
#   N_ab(t) = γ_SAR · full_susceptibility_a(t) · C*_ab(t) · inf_rate_b
#   full_susceptibility_a(t) = susceptibility_a · (1 + (F-1)·A_a(t))   (leaky, stan:260)
# γ_SAR is the single absolute-transmissibility scalar (analysis-plan reparam); susc/inf are RELATIVE,
# normalised to reference bin 1 ("2-10") = 1, so γ_SAR reproduces the reference cell N_11 directly.
#
# C*_ab = per-capita effective contacts from bin a to bin b (mean or excess degree).
# There is NO post-hoc reciprocity symmetrisation: reciprocity is carried entirely by
# the contact-mean estimation (log μ_{i→j} = r_{min,max} + log N_j ⟹ N_i μ_{i→j} =
# N_j μ_{j→i}; see joint_model.jl). The mean builder therefore inherits exact
# reciprocity from μ; the neighbourhood (size-biased) builder does not — this is by design.

# ---- the ONLY line that differs between builders ----
# Inputs: the raw moments of the (zero-included) degree distribution — K1 = ⟨k⟩
# (mean), K2 = ⟨k²⟩ (second moment) — plus a per-cell zero factor `g` that makes the
# neighbourhood degree condition on non-zero contacts (spec inst/1c, inst/1d). This
# unifies NegBin and hurdle-Weibull; the mean builder ignores `g`.
#   g = 1/(1−P₀) for NegBin (left-truncated fitted NegBin, P₀=(φ/(φ+μ))^φ, floored);
#   g = (1−p⁰)  for Weibull (empirical hurdle non-zero probability).
"""`base_contact(builder, k1, k2, g)` — per-capita effective contact C0 from the raw
moments `k1=⟨k⟩`, `k2=⟨k²⟩` and the per-cell zero factor `g`. `MeanNGM` returns the
mean; `NeighbourhoodDegreeNGM` returns the size-biased degree ⟨k²⟩/⟨k⟩ times `g`
(the configuration-network C0 among non-zero degrees, docx `Ccf=(z²/z)⊙Pnz`)."""
base_contact(::MeanNGM, k1, k2, g)                = k1
# `k1>0` guard: per-week Weibull cells with an empirical p⁰=1 (no observed contacts that
# week) give ⟨k⟩=⟨k²⟩=0, so the raw `k2/k1` is 0/0=NaN. Such a cell contributes no
# transmission, so return 0. (NegBin keeps k1=μ>0, so it always takes the first branch.)
base_contact(::NeighbourhoodDegreeNGM, k1, k2, g) = k1 > 0 ? (k2 / k1) * g : zero(k1)   # size-biased × zero factor

"""`full_susceptibility(susc, F, A_col)` — leaky antibody susceptibility vector
`susc .* (1 .+ (F-1).*A_col)` for one week's antibody prevalence `A_col`."""
full_susceptibility(susc::AbstractVector, F::Real, A_col::AbstractVector) =
    susc .* (1 .+ (F - 1) .* A_col)

"""
    contact_star(builder, K1, K2, G)

Per-capita effective contact matrix `C*` for the given builder, from the `A×A`
raw-moment matrices `K1=⟨k⟩`, `K2=⟨k²⟩` and per-cell zero factors `G`. This is simply
the builder's per-cell contact `C0` — there is no post-hoc reciprocity symmetrisation
(reciprocity is carried by the contact-mean estimation; see the module header).
NGM-independent of transmission, so compute once per fit and reuse across weeks.
"""
function contact_star(builder::NGMBuilder, K1::AbstractMatrix, K2::AbstractMatrix,
                      G::AbstractMatrix)
    return base_contact.(Ref(builder), K1, K2, G)
end

"""
    build_ngm(Cstar, susc, inf, F, A_col; gamma_sar=1.0)

7×7 next-generation matrix for one week from a precomputed `C*`:
`N_ab = gamma_sar · full_susceptibility_a · C*_ab · inf_b`. `gamma_sar` is the absolute
transmissibility scalar (susc/inf relative to reference bin 1); `gamma_sar=1` recovers the
pre-reparam form.
"""
function build_ngm(Cstar::AbstractMatrix, susc::AbstractVector, inf::AbstractVector,
                   F::Real, A_col::AbstractVector; gamma_sar::Real = 1.0)
    fs = full_susceptibility(susc, F, A_col)
    return gamma_sar .* ((fs .* Cstar) .* inf')
end

"""Convenience: build `C*` then the NGM in one call (used in tests)."""
function build_ngm(builder::NGMBuilder, K1::AbstractMatrix, K2::AbstractMatrix,
                   G::AbstractMatrix, susc::AbstractVector, inf::AbstractVector,
                   F::Real, A_col::AbstractVector; gamma_sar::Real = 1.0)
    return build_ngm(contact_star(builder, K1, K2, G), susc, inf, F, A_col; gamma_sar = gamma_sar)
end
