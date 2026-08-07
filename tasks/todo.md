# TODO — full-grid run of the `-ar1` generation (2026-08-07)

Plan: `.claude/plans/prancy-sprouting-liskov.md`. Two generations, both from an empty
`dt_intermediate/`:

1. **Rehearsal** — Pathfinder for *both* stages (`STAGE1_USE_NUTS=false`), token
   `temporal-gsar-cut-sc-p0-gi-s0-m32-t0-ar1`, then 8j → 9j → 10j → 11j.
2. **Formal** — Stage 1 NUTS **1000 warmup + 2000 kept draws** (user request; was 500), Stage 2
   Pathfinder, token `…-ar1-nuts`, monitored; then the same notebooks + 12j.

The two tokens differ, so both live in `dt_intermediate/` without collision.

## Phase 0 — done

- [x] `scoringutils` 2.2.0 installed (was absent; 9j's `score_wis`/`score_logs` would have hard-failed)
- [x] `STAGE1_USE_NUTS` read from ENV in 8j/9j/10j/11j — **the landmine**: all four build their own
      `cfg`, and a token mismatch does not error, it silently refits (see the new CLAUDE.md gotcha)
- [x] `FIT_END`/`ORIGIN_MIN` from ENV in 8j and 9j, defaults reproducing the 63-origin set
- [x] 8j calls `prefit_stage1!` and `prefit_stage2!` separately, with per-stage concurrency —
      `fit_concurrency()` assumes 1 GiB/fit but Stage-1 NUTS peaks at 3.4–4.0 GiB, so its 9 would
      have over-subscribed 27 GiB of RAM by ~4×
- [x] `stage1_nuts_draws` 500 → 2000; `nuts_adapts`/`nuts_draws` now recorded in every artefact
- [x] per-origin `@info` heartbeat in `prefit_stage1!` (it printed nothing for the whole grid before)
- [x] `tmp/check_grid.jl` (completeness + provenance gate), `tmp/watch_grid.sh` (progress logger)

## Phase 1 smoke — found a 17-day regression, fixed

The 3-origin Pathfinder smoke got through Stage 1 fine (24 chains, 12.2 min, 8-way fan-out) and then
wrote **one** Stage-2 cell in 19 min at 167 % CPU. Measured cause: **Mooncake is the wrong backend
for Stage 2** — 17.9× faster per gradient (the documented number reproduces) but 35× slower per
`pathfinder()` fit, and it does not parallelise (1.09× at K=9). Stage 2 is 100 fits of an 18-dim
model per cell, so per-fit setup is everything. 16.5 min/cell ⇒ **17 days** for the grid.

Fixed by splitting the backend per stage: `cfg.stage2_ad_backend = :reversediff` (0.2 min/cell ⇒
**5.0 h**, matching the 4.9 h the `-hd` generation took). Stage 1 keeps Mooncake. Full write-up in
`tasks/lessons.md`.

## Budget

| leg | fits | wall |
|---|---|---|
| smoke, 3 origins, Pathfinder, 8j→11j | 24 s1 + 72 s2 | ~1.2 h |
| rehearsal 8j (63 origins) | 504 + 1512 | ~10 h |
| rehearsal 9j/10j/11j | — | ~2–3 h |
| **formal Stage 1 (NUTS 1000+2000)** | 504 | **~88 h at 4-way** |
| formal Stage 2 | 1512 | ~8–14 h |
| formal 9j/10j/11j/12j | — | ~2–3 h |

Per-fit NUTS cost is the pilot's Pathfinder + warmup unchanged, sampling ×4: 1692/2509/2806/3027 s
(mean ×1.60 on 500 draws) ⇒ 351 CPU-hours. Everything is resumable — both prefit drivers skip
existing artefacts — so raising `S1_CONCURRENCY` mid-run costs nothing but a restart.

## Still open (not in this change)

1. **`target_accept = 0.95` cost ESS.** Divergences 4/5 → 0/1, but sub-100 coordinates 3 → 38 and
   min ESS 115.9 → 47.6 (negbin @ 2020-11-15), 69.6 → 30.7 (hweibull @ 2020-11-15). Worth revisiting
   once the η↔ρ ridge is addressed, at which point 0.95 would be cheap.
2. **The η↔ρ_diag ridge** (corr +0.40…+0.45 in all four chains) — still the binding constraint.
3. **split-R̂ failures sit at HIGH ESS** (max R̂ 1.085 at ESS 1023), so they are first-half/second-half
   drift, not autocorrelation. ⇒ **The 2000-draw formal run answers this as a by-product**: if the
   R̂>1.01 count falls roughly in proportion to the extra draws it was autocorrelation, if it holds it
   is drift.
4. **Run hurdle-Weibull with `constant_contacts = true`.** Its temporal process collapsed to exactly
   constant under a prior that does not fight it (φ = 0.9985–0.9998, effrank 1.00, lag-8 corr 0.999)
   — the third independent measurement of the same preference across three kernels and three priors.
   NegBin is unaffected (φ = 0.891, effrank 2.07) and should stay per-week. Deferred until the two
   generations above are on disk, since it is a third token (`pooled-…`).
5. The `phi_time` prior is deliberately Uniform and is the first knob to revisit IF a reason appears
   that is not "the posterior is high" — a high φ is the measurement.

## Correction to the record

`tasks/lessons.md` / earlier notes described the interrupted Stage-2 attempts as ending in a
Pathfinder stack trace. They did not: both `tmp/stage2_h1.log` and `tmp/stage2_ig.log` end in
`signal 15: Terminated` at `install_s1_stage2_h1.jl:90`, i.e. they were killed by hand. There is no
known Stage-2 bug, and `prefit_stage2!` has never been observed to fail.
