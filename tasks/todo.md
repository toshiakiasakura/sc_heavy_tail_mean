# TODO — InverseGamma priors on all three GP length scales (`-ig`) — 2026-08-06

## The change

All three GP length-scale priors move from log-Normal (a `Normal` on `log ρ`) to **InverseGamma on ρ
itself** (user request). Calibrated by **tail-matching the current 90 % intervals**, so the change is
in the prior FAMILY, not in its location or spread — any behaviour change is attributable to the
shape alone.

| scale | old | new | achieved 5 % / 50 % / 95 % |
|---|---|---|---|
| `rho_diag`, `rho_gap` | `Normal(log 20, 0.35²)` on log ρ | `InverseGamma(8.5814, 156.2941)` | 11.246 / 18.944 / 35.569 |
| `rho_time` | `Normal(log 2, 0.35²)` on log ρ | `InverseGamma(8.5814, 15.6294)` | 1.1246 / 1.8944 / 3.5569 |

α is shared because both current priors have σ = 0.35, so both need the same 95 %/5 % ratio (3.1629);
only β scales (β_t = β_s/10). Targets reproduced exactly.

**Measured tail behaviour — the reason this needed checking before committing.** InverseGamma's right
tail is polynomial (∝ ρ^−α−1) and therefore HEAVIER than log-normal's, which matters because
ρ_time drifting to 20–27 weeks has been this model's recurring failure:

| P(ρ_time >) | log-Normal | InverseGamma | ratio |
|---|---|---|---|
| 4 wk | 2.38e-02 | 2.76e-02 | 1.2× |
| 11 wk (window extent) | 5.56e-07 | 4.04e-05 | 73× |
| 26 wk (the observed drift) | 1.17e-13 | 5.20e-08 | 446 000× |

Relatively much heavier; in ABSOLUTE terms still ~1 in 25 000 past the window, so tail-matching kept
it tight enough that the drift should not reopen. **This is the thing to check at the refit** — if
ρ_time climbs past ~4 wk, the prior family is the cause and α must rise.

At the other end InverseGamma does what it is chosen for: P(ρ_time < 0.5) falls 3.73e-05 → 4.50e-07
(83× less mass at the white-noise boundary), P(ρ_diag < 5) likewise 3.73e-05 → 4.50e-07.

## 1. Model — `src/joint_model.jl` `model_degree`

Sample ρ **directly** (Turing's bijector handles the positivity constraint and the Jacobian):

```julia
rho_diag ~ InverseGamma(cfg.gp_len_prior...)
rho_gap  ~ InverseGamma(cfg.gp_len_prior...)
rho_time ~ InverseGamma(cfg.gp_time_len_prior...)
ρ_diag = exp(_softclamp(log(rho_diag), RHO_BOUNDS...))       # clamp KEPT — see below
ρ_gap  = exp(_softclamp(log(rho_gap),  RHO_BOUNDS...))
ρ_time = exp(_softclamp(log(rho_time), RHO_TIME_BOUNDS...))
```

⚠ **Keep the soft-clamps.** They exist because Pathfinder's LBFGS overflows `exp` on aggressive steps
(`tasks/lessons.md` 2026-07-11, the `_softclamp` Inf-safety gotcha). Turing still samples these in an
unconstrained log space internally, so an extreme unconstrained draw still produces an extreme ρ; the
prior discourages it but does not bound it. `log(rho)` is safe because the bijector guarantees ρ > 0.

**Latent count is UNCHANGED at 389 / 977** — still three scalars, only their names and densities move.
⚠ So the count no longer distinguishes generations; the in-chain signal is the NAME (`rho_gap`, no
`log_` prefix) plus the token.

## 2. Priors and token — `src/framework.jl`

- `gp_len_prior      = (8.5814, 156.2941)`  ⚠ semantics change: now **(α, β) of an InverseGamma**, not (μ, σ) of a log-Normal. Same `Tuple{Float64,Float64}` type, so nothing breaks at compile time — the docstring must carry the warning.
- `gp_time_len_prior = (8.5814, 15.6294)`
- `contacts_label`: `…-s0-m32-t0` → `…-s0-m32-t0-ig`.

## 3. Lockstep — the sampled names lose their `log_` prefix

`log_rho_diag`/`log_rho_gap`/`log_rho_time` → `rho_diag`/`rho_gap`/`rho_time`, and readers must stop
exponentiating. Sites (from a repo-wide grep):

| file | hits | note |
|---|---|---|
| `src/10j_viz_utils.jl` | 12 | the read-only μ mirror; **no runtime cross-check except `tmp/verify_sumzero.jl`** |
| `src/joint_model.jl` | 10 | the model + docs |
| `src/8j_viz_utils.jl` | 8 | `load_transmission_draws`, 3 return paths |
| `src/12j_viz_utils.jl` | 4 | SCALARS / PAIRS |
| `tmp/verify_12j.jl` | 6 | |
| `tmp/build_12j_nb.py` | 4 | then regenerate `src/12j_chain_convergence.ipynb` |
| `src/framework.jl` | 6 | docstrings |
| `tmp/verify_sumzero.jl`, `tmp/install_s1_stage2_h1.jl` | 1, 2 | |

**Guards.** Generation detection currently keys on `log_rho_gap` being PRESENT plus the `-m32` token
(10j and 8j mirrors, `tmp/verify_*`). Both must move to `rho_gap` + `-ig`, or every new chain is
refused and every stale one accepted — the same inversion hazard as the `-m32` change.

## 4. Stale artefacts

`dt_intermediate/` currently holds 4 `8j_s1_*` (installed from the 0.95 pilots) and 1 `8j_s2_*`
(written before the Stage-2 run was killed). The token bump makes them unreachable, but delete them
anyway so the directory does not accumulate a second unusable generation.

## 5. Docs

`framework.jl` prior docstrings + token block, `joint_model.jl` model comments, `CLAUDE.md`,
`inst/3_preliminary_model_struct.md`, and a `tasks/lessons.md` entry recording the tail measurement.

## 6. Commit, then refit

One commit. Then archive the 0.95 baseline and refit the same 4 cells
(2 degree models × {2020-11-15, 2021-05-09} × h1) via `tmp/pilot_nuts_timing.jl`.

## Verification

- [ ] Calibration reproduced: `quantile(InverseGamma(α,β), [0.05,0.95])` == the targets to <1e-3.
- [ ] Gradient smoke: **389/977** dims, names `rho_diag`/`rho_gap`/`rho_time` present and
      `log_rho_*` ABSENT, Mooncake finite, ∂logp/∂ each ρ nonzero.
- [ ] `tmp/verify_sumzero.jl` — mirror vs model to <1e-10 (the only guard on the 10j mirror).
- [ ] `tmp/verify_wiring.jl` — token, AD dispatch, artefacts.
- [ ] Refit: **ρ_time posterior must stay ≲ 4 wk.** If it climbs, the heavier tail is the cause
      (measured above) and α must rise — that is the one predicted failure mode.
- [ ] 12j both origins + `tmp/verify_12j.jl`.

## Success criteria — RESULT (refit 2026-08-06)

**Sampling improved substantially; the ρ_time acceptance test FAILED for hurdle-Weibull.** Both are
real and they point the same way.

| cell | coords ESS<100 (0.95 → `-ig`) | min ESS | ρ_time median (0.95 → `-ig`) | ρ_time ESS |
|---|---|---|---|---|
| negbin @ 2020-11-15 | 3 → **2** | 47.6 → **68.9** | 2.27 → 2.24 ✓ | 129.2 |
| negbin @ 2021-05-09 | 10 → **5** | 61.0 → 58.2 | 2.13 → 2.16 ✓ | 107.2 |
| hweibull @ 2020-11-15 | 14 → **1** | 30.7 → **93.1** | 26.79 → **66.20** ✗ | **277.7** |
| hweibull @ 2021-05-09 | 11 → **2** | 59.7 → **75.8** | 20.42 → **47.30** ✗ | **188.3** |

Sub-100 coordinates fell **38 → 10** overall, divergences 1 → 0, depth still 7.0–7.5 with 0 % at cap.

**The ρ_time failure is NOT a prior-calibration failure, and the documented remedy ("α must rise") is
the wrong reading.** ρ_time was ALREADY at 20–27 wk for hurdle-Weibull under the tight log-normal —
i.e. **+7σ into that prior's tail**. `-ig` did not create the drift; its heavier polynomial tail
merely stopped the prior from fighting the likelihood so hard, and ρ_time moved to where the
likelihood actually wants it (~47–66 wk). The decisive evidence is that **ρ_time's own ESS more than
tripled** (76–80 → 188–278) and every other block improved with it: a prior–likelihood conflict was
costing mixing everywhere, and removing it helped.

So raising α would buy a passing number by fighting the data — exactly what `framework.jl`'s
`RHO_BOUNDS` docstring warns against ("a length-scale is only restrained by its prior if the
likelihood is informative about it"). ⚠ Update that "α must rise" note: the honest conclusion is that
**the hurdle-Weibull likelihood wants near-constant contacts over a 12-week window**, which is open
item 1/3 below (the p⁰-versus-μ decomposition), not a prior problem.

NegBin is untouched by all of this (2.24/2.16, unchanged), so the two degree families genuinely
disagree about temporal structure — that is the finding.

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
