# ngm.jl — build the next-generation matrix from the age-pair contact-degree
# distribution. The NGM *builder* is the second swap axis (deterministic dispatch).
#
#   N_ab(t) = full_susceptibility_a(t) · C*_ab(t) · inf_rate_b
#   full_susceptibility_a(t) = susceptibility_a · (1 + (F-1)·A_a(t))   (leaky, stan:260)
#
# C0_ab = per-capita effective contacts from bin a to bin b (mean or excess degree);
# C*_ab = reciprocity-balanced per-capita contacts (docx: N_a μ_ab = N_b μ_ba).

# ---- the ONLY line that differs between builders ----
# Inputs are the raw moments of the (zero-included) degree distribution:
#   K1 = ⟨k⟩ (mean), K2 = ⟨k²⟩ (second moment). This unifies NegBin and hurdle-Weibull.
"""`base_contact(builder, k1, k2)` — per-capita effective contact C0 from the first
two raw moments `k1=⟨k⟩`, `k2=⟨k²⟩` of the degree distribution."""
base_contact(::MeanNGM, k1, k2)                = k1
base_contact(::NeighbourhoodDegreeNGM, k1, k2) = k2 / k1        # ⟨k²⟩/⟨k⟩ = m(1+CV²)

"""
    reciprocity_balance(C0, pop)

Total-contact reciprocity (analysis plan): return `C*` with
`C*_ab = (pop_a·C0_ab + pop_b·C0_ba) / (2·pop_a)`, so `pop_a·C*_ab = pop_b·C*_ba`.
(The reference instead imposes per-capita symmetry; the docx specifies total balance.)
"""
function reciprocity_balance(C0::AbstractMatrix, pop::AbstractVector)
    T = pop .* C0                       # T_ab = pop_a · C0_ab  (row scaling)
    Tsym = (T .+ T') ./ 2
    return Tsym ./ pop                  # divide row a by pop_a
end

"""`full_susceptibility(susc, F, A_col)` — leaky antibody susceptibility vector
`susc .* (1 .+ (F-1).*A_col)` for one week's antibody prevalence `A_col`."""
full_susceptibility(susc::AbstractVector, F::Real, A_col::AbstractVector) =
    susc .* (1 .+ (F - 1) .* A_col)

"""
    contact_star(builder, K1, K2, pop)

Reciprocity-balanced per-capita contact matrix `C*` for the given builder, from the
`A×A` raw-moment matrices `K1=⟨k⟩`, `K2=⟨k²⟩`. NGM-independent of transmission, so
compute once per fit and reuse across weeks.
"""
function contact_star(builder::NGMBuilder, K1::AbstractMatrix, K2::AbstractMatrix,
                      pop::AbstractVector)
    C0 = base_contact.(Ref(builder), K1, K2)
    return reciprocity_balance(C0, pop)
end

"""
    build_ngm(Cstar, susc, inf, F, A_col)

7×7 next-generation matrix for one week from a precomputed `C*`:
`N_ab = full_susceptibility_a · C*_ab · inf_b`.
"""
function build_ngm(Cstar::AbstractMatrix, susc::AbstractVector, inf::AbstractVector,
                   F::Real, A_col::AbstractVector)
    fs = full_susceptibility(susc, F, A_col)
    return (fs .* Cstar) .* inf'
end

"""Convenience: build `C*` then the NGM in one call (used in tests)."""
function build_ngm(builder::NGMBuilder, K1::AbstractMatrix, K2::AbstractMatrix,
                   susc::AbstractVector, inf::AbstractVector, F::Real,
                   A_col::AbstractVector, pop::AbstractVector)
    return build_ngm(contact_star(builder, K1, K2, pop), susc, inf, F, A_col)
end
