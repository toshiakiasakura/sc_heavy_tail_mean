# TODO — temporal sum-to-zero on the weekly level (`-t0`) — 2026-08-06

## Context: measured, not assumed

`joint_model.jl` has carried this note since the temporal GP landed:

> STILL CONFOUNDED, DELIBERATELY LEFT: `c` and the temporal mean of `σ_c·(Lt·z_c)` duplicate each
> other the same way — one flat direction on the Tn axis, fixable with the identical
> `_sum_zero_basis` machinery. Held back so the Tn-axis change can be measured separately.

Measured on all four `-m32` chains (2 degree models × 2 origins):

| chain | corr(c, time-mean of σ_c·Lt·z_c) | SD(c) | SD(dev) | **SD(sum)** |
|---|---|---|---|---|
| negbin @ 2020-11-15 | **−1.000** | 0.379 | 0.379 | **0.007** |
| negbin @ 2021-05-09 | **−1.000** | 0.379 | 0.379 | **0.007** |
| hweibull @ 2020-11-15 | **−1.000** | 0.668 | 0.668 | **0.007** |
| hweibull @ 2021-05-09 | **−1.000** | 0.725 | 0.725 | **0.007** |

The two components cancel to **1–2 % of their individual spread**. The mean weekly level is pinned by
the data (SD 0.007) while its two parameterisations wander ±0.4–0.7 along an exactly anti-correlated
ridge. This is a textbook flat direction, present in **every** chain — the single best-evidenced
geometry defect currently in the model.

**Scope, stated honestly.** This fixes the `c`/`z_c` ridge. It will *not* fix hurdle-Weibull's
100 %-at-cap, whose worst blocks are `log_rho_gap`, `z`, `log_eta`, `log_rho_diag`; the same
diagnostic found an η↔ρ_diag ridge there (r = +0.47) that this change does not touch. Expect it to
help NegBin's remaining split-R̂ failures (24–31 of 390, with `z_c` among the worst-mixing blocks)
and to help hurdle-Weibull less.

Two other findings from the same sweep, recorded so they are not re-litigated:
- **ρ_time is NOT in a ridge**: |corr(log_rho_time, log‖z‖)| ≤ 0.06 and |corr(log_rho_time, c)| ≤ 0.13
  in all four chains. Its value is a statement about the data, not a geometric artefact.
- **The μ soft-clamp is nowhere near binding**, and the two responses are on a *comparable* scale
  once the hurdle hives off the zeros: empirical cell means median 0.74 (weighted) vs 0.76
  (unweighted), with 4.6–6.7 nats of margin to the lower clamp and 4.7–5.5 to the upper. The raw
  per-person duration-weighted degree is much smaller than the count, but that does not propagate to
  what `model_degree` parameterises. No clamp change needed.

---

## 1. The change — `src/joint_model.jl`, `model_degree`

Apply the **existing** `_sum_zero_basis` machinery to the Tn axis of the **level only**:

```julia
Qt = _sum_zero_basis(Tn)                                   # Tn × (Tn−1), Helmert basis of 1^⊥
Lc = Matrix(cholesky(Symmetric(transpose(Qt) * Kt * Qt) + 1e-4 * I).L)
z_c ~ filldist(Normal(0, 1), Tn - 1)                       # was Tn
c_vec = c .+ σ_c .* (Qt * (Lc * z_c))                      # Σ_t (c_vec − c) = 0 exactly
```

Implied covariance `σ_c²·(Mt·Kt·Mt)` with `Mt = I − 11ᵀ/Tn` — the same temporal GP *conditioned* on
the deviation summing to zero, not an approximation. Identical construction to `-s0`, one axis over.

**Only the level.** Do **NOT** also project the structure field's time axis. `R`'s per-pair mean over
weeks duplicates nothing — no other parameter carries persistent age-pair structure — so constraining
it would force every pair's structure to average to zero across the window. That is a model
restriction, not a reparameterisation, and it would be wrong.

Latent count: `z_c` goes Tn → Tn−1, so **390/978 → 389/977**. ⚠ Numerically the same as the
short-lived `-diag` generation's counts; **not** the same model. Tell them apart by `log_rho_gap`
(present here, absent under `-diag`) and by the token.

**Token** — `contacts_label`, the only executable copy: `…-gi-s0-m32` → `…-gi-s0-m32-t0`.

## 2. Lockstep

- **`10j_viz_utils.jl` mirror** (`reconstruct_mu_draws`) rebuilds the level independently:
  `c_wk = cc[d] + σ_c[d] * dot(Lt[wk,:], Zc[d,:])` → `cc[d] + σ_c[d] * dot(Qt[wk,:], Lc * Zc[d,:])`.
  Read `Zc` at Tn−1 columns. The field's temporal factor stays the full `Lt` — unchanged. This mirror
  has no runtime cross-check except `tmp/verify_sumzero.jl`; it must move in lockstep.
- **`tmp/verify_12j.jl`** — `expect_npar` hard-codes `+ Tn` for `z_c`. Derive the `z_c` row count from
  the chain like `zrows`, so the check stays generation-agnostic.
- **Counts 390/978 → 389/977** and `z_c` block size `Tn` → `Tn−1` in: `framework.jl`,
  `joint_model.jl`, `12j_viz_utils.jl`, `CLAUDE.md`, `inst/3_preliminary_model_struct.md`,
  `tmp/build_12j_nb.py` (then regenerate the notebook), `tmp/verify_sumzero.jl`, `tmp/verify_wiring.jl`.
- `_stage1_init`/`_pf_mean_init` select by name prefix, so a shape change cannot misalign them.

## 3. Docs

`framework.jl` token block (a `-t0` entry), the `model_degree` comment that currently defers this
change, `CLAUDE.md`, `inst/3` §5, and a `tasks/lessons.md` entry recording the −1.000 measurement.

## 4. Commit

One commit: the model change plus all lockstep and docs, with the correlation table in the message.

## 5. Refit

Archive `-m32` first (`dt_intermediate_nuts_pilot_m32/`, `tmp/pilot_baseline_m32/`,
`res/12j_m32_baseline/`), then the same 4 cells via `tmp/pilot_nuts_timing.jl`
(2 degree models × {2020-11-15, 2021-05-09} × h1), then 12j on both origins.

## Verification — ALL DONE (commit `e9b2606`)

- [x] Static: `Mt·Kt·Mt` reproduced to **7.772e-16**; deviation sums to zero to **2.609e-15**; `Lc`
      Cholesky clean at all 200 ρ_time (min eigenvalue 4.163e-07); flat direction SD **0.578 → 4.743e-17**.
- [x] Gradient smoke: **389/977** dims, `z_c` = 11, `log_rho_gap` present, Mooncake finite.
- [x] `tmp/verify_sumzero.jl` — mirror vs model **2.946e-15**.
- [x] `tmp/verify_wiring.jl` — ALL PASS.
- [x] Post-fit ridge test — see below. Note the planned "corr must no longer be −1.000" is **vacuous
      under `-t0`**: the deviation's time-mean is zero *by construction*, so the correlation is 0/0.
      The test that carries the information is SD(c), which was predicted to land near 0.007.
- [x] 12j both origins (28 figures) + `tmp/verify_12j.jl` exit 0, ALL PASS.

## Outcome

**The deliverable — ridge removed, all four cells.** Sum-to-zero exact (max|time-mean deviation|
6.2e-18 … 6.0e-17), and SD(c) collapses onto the predicted pinned scale:

| chain | SD(c) `-m32` → `-t0` | factor | ESS(c) |
|---|---|---|---|
| negbin @ 2020-11-15 | 0.379 → **0.0063** | 60× | 217 → **984** |
| negbin @ 2021-05-09 | 0.379 → **0.0073** | 52× | 292 → **1113** |
| hweibull @ 2020-11-15 | 0.668 → **0.0066** | 102× | 228 → **1234** |
| hweibull @ 2021-05-09 | 0.725 → **0.0070** | 103× | 307 → **805** |

**Hurdle-Weibull's 100 %-at-cap DID resolve — the "not expected" scope note above was wrong.**
Mean depth 10.00/9.99 → **6.99/6.98**, 0 % at cap; step size ≈10× larger; fit 8316/7949 s →
**2232/2201 s**; min ESS 18.2/45.4 → **69.6/99.8**; coords ESS<100 16/978 and 8/978 → **1/977** each.
The scope note reasoned from *which block mixed worst*, which does not hold under a single global
step size — see `tasks/lessons.md` 2026-08-06.

NegBin: depth 8.95 → 7.00, fits 2.2× faster, min ESS 135.5 → 115.9 and 112.6 → 69.3 (ESS *per second*
1.9× and 1.4× better; the per-draw drop is what 4× shorter trajectories do at a fixed 500-draw budget).
`z_c` left the worst-mixing blocks for hweibull (236 → 415, 207 → 438); for negbin it is still the
worst block but above threshold (116/119).

**All four verdicts: NOT CONVERGED**, in every case on split-R̂. Failures rose (31→37, 24→39,
98→126, 79→136). But they **do not track low ESS**: among coordinates with ESS ≥ 400, 6.5–13.9 % fail,
and the largest R̂ per hweibull chain is at **ESS 1349 (1.060)** and **ESS 1023 (1.085)**. That is
first-half/second-half drift, not autocorrelation — a different defect from the one ESS measures.
Divergences: negbin 0/0; hweibull 0→4 and 3→5.

**Next step, in order.** (1) Re-run with more kept draws — now cheap at 2.2–3.7× the speed — which
discriminates short-run drift from genuine non-stationarity and is the only way to settle the R̂
signal. (2) Then the η↔ρ_diag ridge, now the binding constraint in *all four* chains
(corr +0.40…+0.45), not just hweibull.

---

## Still open (not in this change)

1. **Where weekly variation belongs in the hurdle model.** hweibull has 588 free `p0f` zero-probs
   (one per cell × week, `Beta(1,1)`, unstructured in time) *and* the GP field. NegBin has no such
   escape valve. If weekly behaviour change shows up as "more people at zero contacts" rather than
   "shorter contacts", p⁰ absorbs it and ρ_time → 20–27 wk legitimately. Decomposing the weekly
   change in the cell mean into its p⁰ and μ parts would settle it.
   Unchanged by `-t0`: hweibull's ρ_time is still **27.1 / 21.0 weeks** (negbin **2.30 / 2.19**).
2. **The η↔ρ_diag ridge** — the classic GP amplitude/length-scale trade-off. No longer hweibull-only:
   after `-t0` it is present in **all four** chains at corr **+0.40 … +0.45**, and it is now the
   binding constraint (worst coords are `log_rho_gap` 69.3, `log_rho_diag` 69.6, `log_eta` 99.8).
3. **Whether hweibull should simply run `constant_contacts=true`**, which is what its posterior says.
4. **Whether the split-R̂ signal survives a longer run.** R̂ failures sit at high ESS (up to 1349), so
   they are drift, not autocorrelation. `-t0` made the fits 2.2–3.7× faster, so more kept draws is
   now the cheap discriminating experiment.
