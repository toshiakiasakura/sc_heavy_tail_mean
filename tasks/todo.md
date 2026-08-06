# TODO — AR(1) temporal correlation (`-ar1`) — 2026-08-06 — DONE, refitted

Committed `079e3b4`. `Kt[s,t] = φ^|s−t|` (AR(1) ≡ exponential ≡ Matérn 1/2) replaces the Matérn 3/2 in
the time direction only; spatial kernel untouched; `phi_time ~ Uniform(0,1)`; 389/977 unchanged.
Full rationale, conditioning table and refit outcome are in `src/framework.jl`'s token block and
`tasks/lessons.md`.

**Refit vs the `-t0` @ 0.95 generation it replaced:** better in 3 of 4 cells (min ESS 61.0→126.2,
47.6→78.4, 30.7→56.4, 59.7→48.5), sub-100 coords 38→29, zero divergences everywhere, depth 7.00.
NegBin gained most — NOT the prediction. `Lc` spread flat at 1.8 as predicted; `Lt` spread 147 at
hurdle-Weibull's φ = 0.9998, because that is past the φ ≤ 0.999 grid I swept.

## THE NEXT ACTION, and it is not another temporal prior

**Run hurdle-Weibull with `constant_contacts = true`.** Its temporal process has now collapsed to
exactly constant under a prior that does not fight it (effrank 1.00, lag-8 corr 0.999,
P(φ>0.99)=1.000) — the third independent measurement of the same preference across three kernels and
three priors. NegBin is unaffected (φ = 0.891, effrank 2.07) and should stay per-week. That asymmetry
— the two degree families genuinely disagreeing about temporal structure — is the finding to act on.

Secondary: the `phi_time` prior is deliberately Uniform and is the first knob to revisit IF a reason
appears that is not "the posterior is high" — a high φ is the measurement.

---

## Still open (not in this change)

1. **`target_accept = 0.95` cost ESS.** Divergences 4/5 → 0/1, but sub-100 coordinates 3 → 38 and
   min ESS 115.9 → 47.6 (negbin @ 2020-11-15), 69.6 → 30.7 (hweibull @ 2020-11-15). Worth revisiting
   once the η↔ρ ridge is addressed, at which point 0.95 would be cheap.
2. **The η↔ρ_diag ridge** (corr +0.40…+0.45 in all four chains) — still the binding constraint.
3. **split-R̂ failures sit at HIGH ESS** (max R̂ 1.085 at ESS 1023), so they are first-half/second-half
   drift, not autocorrelation. More kept draws is the cheap discriminating experiment.
4. **13j/Stage-2/10j run** — `src/13j_model_diagnostics_h1.ipynb` and `tmp/install_s1_stage2_h1.jl`
   are built and verified but were interrupted; re-run after this refit lands.
