# 9j transmission-parameter visualisation — match the current parameter construction

Measured on the completed Pathfinder grid (`temporal-gsar-cut-sc-p0-gi-s0-m32-t0-ar1`,
504/1512, 2026-08-09), h=1, all 63 origins × 6 models.

## What the model actually constructs (`model_transmission`, joint_model.jl:622)

| latent | derived quantity | visualised? |
|---|---|---|
| `log_gamma_sar` | `gamma_sar = exp(softclamp(·, log 0.001, log 10))` | `plot_gamma` — **wrong scale** |
| `sig_s`, `z_s` (A−1) | `susc = exp(softclamp(sig_s·z_s, log .05, log 20))`, ref spliced = 1 | `plot_ratio` — **wrong scale** |
| `sig_i`, `z_i` (A−1) | `inf`, or **pinned `ones(A)`** when `fix_infectivity(nb)` | plotted, **pin not marked** |
| `sigma_inf` | infection-likelihood observation SD | **not visualised at all** |
| `F` | **pinned 1.0** (antibody term off) | `plot_F` — pin marked ✓ |
| `w_mu`, `w_sigma` | generation interval | `plot_gen_interval` ✓ (with prior band) |

## Findings

0. **THE PRIMARY BUG — every one of these figures was BLANK.**
   `collect_transmission_structure` called `load_transmission_draws(lbl, origin, h)` with no
   `contacts` argument, so it defaulted to `CONTACTS_TOKEN` — always `-nuts`, a generation that
   does not exist on disk. All 378 lookups returned `nothing`, every store stayed NaN, and all
   eight figures rendered as bare reference lines. **No warning**: a missing Stage-2 artefact is a
   legitimate skipped origin×combo. Same bug fixed across `10j_viz_utils.jl` on 2026-08-08; 9j was
   missed. Tell: the collector runs in **0.4 s** when blind vs **11.4 s** when reading the grid.
   Findings 1–5 below are real but were *invisible* until this was fixed.

1. **`plot_ratio` uses a LINEAR y-axis** (9j_viz_utils.jl:1342 — `_median_ylims` without
   `log = true`, no `yscale`) for a quantity built as `exp(·)`, i.e. multiplicatively symmetric
   about 1. Measured median range across the grid: **0.12 – 18.1**. On one shared linear axis the
   entire sub-1 half — *reduced* susceptibility, half the parameter's range by construction —
   collapses into a sliver, and `mean-diagonal` (0.19–1.9) is invisible beside
   `negbin|neighbourhood` (→18.1). Its own per-bin companions `plot_ratio_bins` and
   `plot_susc_inf_bins_ci` are already `:log10`, so the headline figure contradicts them.
2. **`plot_gamma` is linear with a hard `ylims = (0, 2.0)`** for γ_SAR = `exp(softclamp(·))`.
   Measured: `negbin|neighbourhood` sits ON the lower clamp 0.001 (1.9% of draws), others reach
   past 2.0 and are silently clipped. Both tails are invisible.
3. **`sigma_inf` is fitted and stored in every `8j_s2_*`** (`fit_stage2_pooled`, joint_model.jl:1176)
   but `load_transmission_draws` drops it, so no store and no figure exist. Median 0.18–0.21.
4. **Pinned parameters are drawn exactly like fitted ones.** `inf ≡ 1` under `fix_infectivity`
   is a *pin* (a diagonal NGM identifies only `susc_a·inf_a`), yet renders as an ordinary flat
   line. Worse, that model's `susc` therefore absorbs the whole product and is **not on the same
   footing** as the other models' susc — while sharing an axis with them.
5. **No prior reference on susc/inf or γ_SAR**, though `plot_gen_interval` establishes exactly that
   idiom, and `susc_inf_sd_prior` was set to N⁺(0, 0.25²) precisely so the age profile *can* shrink
   to flat when the data are silent — with no way to see whether it did.

Clamp pile-up is otherwise negligible (susc/inf 0.0–0.3% at the bounds), so this is a scale and
annotation problem, not a refit.

## Tasks

- [x] **`9j_viz_utils.jl::collect_transmission_structure` — pass `contacts = contacts_label(cfg)`**
- [x] `8j_viz_utils.jl::load_transmission_draws` — also return `sigma_inf` (all 3 return sites)
- [x] `9j_viz_utils.jl::collect_transmission_structure` — `sigma_inf` store; detect pinned
      `inf`/`F` from the draws themselves (`all(== 1.0)`) rather than parsing labels
- [x] `_prior_ratio_band` / `_prior_gamma_band` helpers — prior 90% bands from `cfg`
- [x] `plot_ratio` — log10 axis, prior band, pinned/gauge annotation
- [x] `plot_gamma` — log10 axis, soft-clamp lines, prior band, no hard cap
- [x] `plot_sigma_inf` — new figure
- [x] 9j notebook cell 17 — pass `cfg`/`pinned`, add the σ_inf figure
- [x] Verify by rendering the affected figures headless

Viz layer only — no refit, no cache invalidated.

---

# 2026-08-09 — Stage-1: origin-anchored contact window, iid weekly level, φ ~ Beta(3,3)

Three user-requested changes to `model_degree`, landed together as one cache generation.
**Cache token: `temporal-w8h-lc0` (+`-nuts`)** — superseded on 2026-08-10 by
`temporal-w8h-lc0-m32t`, see the section below. The nine accumulated historical suffixes were
dropped at the same time (also user request); retained generations are the literals
`CONTACTS_TOKEN_AR1` / `_PF` / `_HD` in `framework.jl`.

## What changed

1. **`-w8h`** — `prepare_degree_data` spans `win.fit_weeks ++ win.forecast_weeks` of
   `degree_window(origin, h, cfg)` = `[t₀−n_fit+1 … t₀+h]` (9/10/11/12 wk), not the 12 `all_weeks`
   of a horizon-shifted window. The `smax` lag weeks were fitted and discarded:
   `model_transmission` indexes `C*` only over `t = smax+1 … Tn`. Stage-1 latents go from a flat
   389/977 to **293/325/357/389** and **734/815/896/977**. The infection window is unchanged at 12,
   so the two are paired as `Cstar_weeks[t − smax + h]` ↔ `wd` week `t`, with `h` derived from the
   length and the dates asserted in `stage2_inputs`.
   ⚠ Superseded a same-day sliding variant `[t₀−n_fit+1+h … t₀+h]` (`-w8`), which was exactly the
   weeks the renewal reads but left the earliest fit weeks in no chain — 10j §2c showed 8 points
   instead of 12. That generation's 3-origin smoke grid is still on disk under `temporal-w8-lc0`.
2. **`-lc0`** — `c_t = c + σ_c·(Qt·z_c)`: the level's `Lc = chol(Qtᵀ·Kt·Qt + 1e-4·I)` whitening is
   gone, so the weekly level is iid (still sum-to-zero, `-t0` retained) and `phi_time` reaches the
   likelihood only through `Lt`. No name and no dimension change — the token is the only record.
3. **`ar1_phi_prior` (2,2) → (3,3)** — `P(φ>0.99)` 2.98e-4 → 9.85e-6; logit tail `e^{-2u}` → `e^{-3u}`.

## Done

- [x] `degree_agepair.jl` — `fit_weeks` window + docstrings
- [x] `joint_model.jl` — `Cstar_weeks[t − smax]` + length assertion; `fit_stage2_pooled` takes its
      length from the Stage-1 draw; `stage2_inputs` null path uses `cfg.n_fit`;
      `null_contact_level` partial-overlap documented, warning dropped
- [x] `joint_model.jl::model_degree` — `Lc` removed, comment blocks rewritten
- [x] `framework.jl` — prior, short token, `is_legacy_token`, `CONTACTS_TOKEN_AR1`
- [x] `10j_viz_utils.jl` — `C*` offset, `-lc0` replay branch, `_read_disp_chain` bounds guard
- [x] `8j/10j` generation sniffs rewritten against `is_legacy_token` (a bare `occursin("-m32", …)`
      would reject every current chain now that the token is short)
- [x] `11j`/`12j` — Tn is generation-dependent; docstrings
- [x] `tmp/run_grid_batched.sh` — token derived from `framework.jl`, not hardcoded
- [x] Docs: `inst/3` §2.2/§5/§6/§10/§11/§12.4, `CLAUDE.md`, `tasks/lessons.md`
- [x] Static verification (scratch script): window = 8 wk ending at origin+h for every h; Stage-1
      dims 261/653; level identical at φ=0.1 vs 0.99 and sum-to-zero; `Cstar` length assertion
      fires on a wrong-length input; null path returns `n_fit` matrices; `P(φ>0.99) = 9.85e-6`

## Measured — 3-origin Pathfinder smoke, SLIDING `-w8` window (superseded; token `temporal-w8-lc0`)

⚠ Measured on the sliding window, i.e. `Tn = 8` flat, before the anchored `-w8h` correction. Kept
because the φ result is the interesting one and is unlikely to reverse; re-measure under `-w8h`.

Ran clean: **24/24 Stage-1 in 1477 s** (concurrency 5), **72/72 Stage-2 in 583 s** (concurrency 9),
0 failed. Chain shape confirms `-w8`: `z` is 27×8, `z_c` is 7, 261 latents (NegBin) / 653
(hurdle-Weibull) — the 264/656 stored columns include 3 MCMCChains internals.

Divergence census (`max|z| > 10` or `|log_eta| > 5`), all 24 chains:

| family | n | diverged | φ median (max) | φ > 0.99 | max\|z\| | σ_c median | log_eta median |
|---|---|---|---|---|---|---|---|
| unweighted-negbin | 12 | **0** | 0.577 (0.656) | 0 | 2.51 | 0.57 | 0.38 |
| weighted-hweibull | 12 | **0** | 0.472 (0.529) | **0** | 1.62 | 0.24 | 0.55 |

**The weighted path no longer approaches the φ→1 boundary.** Under `-t0-ar1` its φ posterior median
was 0.9985–0.9998 with 11.5 % of the 252 chains above 0.99 and σ_c inflated to 0.48 among those;
here the *maximum* over 12 chains is 0.529 and σ_c sits at 0.24. That is the regime `-lc0` and
Beta(3,3) were aimed at.

⚠ **Do not promote this to a finding yet.** It is 12 chains at 3 adjacent origins, not the 252-chain
census, and it is confounded with `-w8`: 8 weeks of temporal data instead of 12 is also less to pool
over, so some of the drop in φ may be the shorter window rather than the level change or the prior.
Separating them needs either the full grid or a deliberate `-w8`-only / `-lc0`-only pair at one
origin.

## Open

- [ ] **Re-run the divergence census over the FULL grid** once one exists under the new token — the
      0/24 above is the smoke set, and the `-t0-ar1` baseline it is being compared against is 252
      chains over 63 origins. `inst/3` §12.4.
- [ ] **Attribute the φ drop** between `-w8` (less temporal data), `-lc0` (φ no longer identified by
      the level) and Beta(3,3) (prior). One origin × hurdle-Weibull × each lever in isolation would
      settle it; nothing downstream depends on the answer, but §5's ⚠ box makes a mechanistic claim
      that this either confirms or complicates.
- [ ] **Production refit** under `temporal-w8-lc0-nuts` — the user's call to launch. Stage 1 should
      be materially cheaper than the 219→351 CPU-h estimate at `Tn = 12`, since the parameter space
      is a third smaller, but that has not been measured.
- [ ] Point 9j/10j/11j at `CONTACTS_TOKEN_AR1` if the existing Pathfinder grid needs reading while
      the new one is fitted (`ENV["STAGE1_USE_NUTS"]` alone no longer reaches it — the token differs
      by model, not just sampler).

---

# 2026-08-10 — Stage-1: the temporal kernel reverts to Matérn 3/2 (`-m32t`)

User request: "revert the AR(1) to Matérn 3/2 kernel while the structure should be kept remained
(i.e. the kernel should be applied independently for each age pair, and no smoothing kernel on the
level term). The prior for the length scale should be appropriately chosen considering the scale of
time."

**Cache token: `temporal-w8h-lc0-m32t` (+`-nuts`).** Confirmed with the user before implementing:
prior `N(log 2, 0.35²)`, and the AR(1) path **removed** rather than kept behind a config switch.

⚠ **THIS GENERATION LASTED ONE DAY AND WAS REVERTED** — see the section below. Kept here because
its smoke is the measurement that decided the matter, and its 24 s1 / 72 s2 artefacts are on disk.

## What changed

1. **`-m32t`** — `Kt[s,t] = m32(|s−t|/ρ_time)` with `log_rho_time ~ Normal(cfg.gp_time_len_prior…)`
   soft-clamped to `RHO_TIME_BOUNDS`, replacing `phi_time ~ Beta(cfg.ar1_phi_prior…)` and
   `Kt = φ^|s−t|`. Both GP kernels are Matérn 3/2 again, as under `-m32`. **Time direction only** —
   the spatial kernel, the separable matrix-normal (one trajectory per age pair under a shared η,
   pairs correlated across age through `La`), `-s0`/`-t0`/`-lc0` and the `-w8h` window are all
   untouched. The structure the user asked to keep is the structure that predates `-ar1`.
2. **`gp_time_len_prior` restored at `(log 2, 0.35)`**; `ar1_phi_prior` deleted. The value equals
   what `-m32` set, but it was re-derived against the *current* 9–12 week window — see below.
3. **`RHO_TIME_BOUNDS` is live again** (dead code only while `-ar1` had a bounded φ), and **inert**:
   −5.9σ / +11.3σ under this prior, so it is an `exp`-overflow guard, not a modelling constraint.

⚠ The latent count is **unchanged** (one scalar for one scalar): 293/325/357/389 and
734/815/896/977. Unlike `-m32`/`-t0`/`-lc0`, though, this change **renames** the parameter, so a
chain proves its own generation — which is why the 10j mirror *forks* on the name (keeping the
retained `-ar1` grid replayable) while 8j/9j *refuse* (one figure per generation).

## Why the prior is `N(log 2, 0.35²)` — the part the user asked to be reasoned about

| | ρ_time |
|---|---|
| 90% prior interval | [1.12, 3.56] wk |
| correlation at lag 1 / 2 / 4 / 8 | 0.785 / 0.483 / 0.140 / 0.008 |
| `Lt` column spread at q05 / med / q95 | 1.39 / 2.41 / 5.17 |
| P(ρ_time > 9 wk), the shortest window | 8.7e-6 |

1. **Matches the measurement** — the 252-fit `-ar1` survey put NegBin at φ median 0.726 (max 0.844),
   which at matched lag-1 correlation is ρ_time 1.6–2.5 wk. The prior sits on the posterior the data
   produced rather than fighting it (the failure `-ig` diagnosed).
2. **Keeps the geometry healthy** — `Lt` column spread is what broke NUTS on 2026-08-05 (2.4 healthy,
   228–274 in chains that would not mix). Across this prior's band it is 1.4–5.2; at ρ_time = 26 it
   is 94.
3. **Excludes the unidentified region** — over 9–12 weeks, ρ_time ≳ 9 wk is the pooled limit. The
   2026-08-05 drift to 24–63 wk happened under N(log 4, 0.5²), a wider prior on a *longer* window.

Rejected after pricing the same way: centre 3 wk (P(ρ>9) = 8.5e-4, spread to 9.7) and SD 0.5
(P(ρ>9) = 1.3e-3) — both defensible, neither better motivated.

## Done

- [x] `joint_model.jl::model_degree` — kernel swap, clamp restored, comment block rewritten to argue
      the revert instead of `-ar1`
- [x] `framework.jl` — `gp_time_len_prior` restored, `ar1_phi_prior` deleted, `RHO_TIME_BOUNDS`
      un-deadened, token → `-m32t`, `-m32t` section added to `contacts_label`'s history
- [x] `10j_viz_utils.jl` — temporal kernel **forked on the chain's parameter name**; the level fork
      still on the token; both labelled with which evidence they use
- [x] `8j_viz_utils.jl` — returns `rho_time` (weeks, softclamped) not `phi_time`; guard requires
      `log_rho_time` and refuses `-ar1` chains
- [x] `9j_viz_utils.jl` — `plot_lengthscales` temporal panel is ρ_time in weeks with the window
      length drawn as the identifiability limit (the two-panel split from `-ar1` is KEPT — the units
      never matched); notebook passes `cfg`
- [x] `12j_viz_utils.jl` + `12j` notebook `SCALARS` — `:phi_time` → `:log_rho_time`
- [x] Docs: `CLAUDE.md`, `inst/3` §5/§6/§10/§12.5, `tasks/lessons.md`
- [x] Static verification (`scratchpad/check_m32t.jl`): token forks; `ar1_phi_prior` absent and the
      clamp inert; `Kt` unit-diagonal/symmetric/PSD/full-rank at 40 ρ_time × Tn 9–12 with a clean
      Cholesky; dims 293/325/357/389 and 734/815/896/977 with `log_rho_time` in and `phi_time` out;
      gradients finite with **zero** null partials; the 10j mirror reproduces the model's own μ at
      all Tn weeks; and an archived `-t0-ar1` chain still replays through the name-fork

## Measured — 3-origin Pathfinder smoke (`temporal-w8h-lc0-m32t`)

**24/24 Stage-1 in 26 min** (concurrency 6), **0 failed**. Divergence census over all 24 chains
(`max|z| > 10` or `|log_eta| > 5`): **NO DIVERGENCES** — max|z| 3.31, |log_eta| ≤ 0.85 everywhere.

| family | n | div | ρ_time med | ρ_time max | prior z | σ_c med | max\|z\| |
|---|---|---|---|---|---|---|---|
| unweighted-negbin | 12 | **0** | 1.44 wk | 1.82 | **−0.95** | 0.45 | 2.65 |
| weighted-hweibull | 12 | **0** | 5.31 wk | 29.5 | **+2.60** | 0.38 | 3.31 |

**ρ_time against its prior — the check `gp_time_len_prior` asks for, and the two families disagree.**

- **NegBin sits BELOW the centre and is nowhere near the window.** All 12 chain medians fall in
  1.03–1.71 wk, i.e. prior z −1.90…−0.44; 0/12 above q95, 2/12 below q05, 0/12 at or past the
  window length. The data want *slightly less* temporal smoothing than the prior asserts — the
  opposite of a pile-up. **By the criterion set when the prior was chosen, the centre stays.** It
  also corroborates the derivation: the `-ar1` survey's φ median 0.726 predicted ρ_time ≈ 2.0 wk at
  matched lag-1, and the direct estimate is 1.44.
- **Hurdle-Weibull pulls hard upward, as expected, and this is now the fourth independent
  measurement of the same preference.** 6/12 above prior q95, 3/12 at or past the window length
  (12.0 wk at Tn=9, 10.2 at Tn=9, 24.4 at Tn=12), worst chain z = +7.14. Its p⁰ ≈ 0.95 likelihood
  is nearly flat in time, so it wants the pooled limit whatever the parameterisation: ρ_time 20–27
  wk under the pre-`-ar1` log-normal, 47–66 under `-ig`, φ → 0.9985–0.9998 under `-ar1`, and now
  ρ_time past the window again. The prior restrains it rather than following it — deliberately —
  but this IS a live prior–likelihood conflict of the kind `-ig` diagnosed as costing ESS.

⚠ **These are PATHFINDER medians and must not be quoted as a posterior for ρ_time.** The project's
own 2026-08-05 note is explicit: in a flat direction Pathfinder's normal approximation has no
curvature to fit and can come back arbitrarily wide or displaced (it returned ρ_time ≈ 1047 once).
That caveat bites hardest exactly where the conflict is — the hurdle-Weibull path. **The NUTS smoke
is the trustworthy read**, and the +7.14σ chain is the one to look at there first.

⚠ Confound against the superseded sliding-`-w8` smoke (φ max 0.529, i.e. no upward pull at all):
that ran under Beta(3,3) on φ with Tn = 8 flat. Beta(3,3) is far more restrictive near the pooled
limit than N(log 2, 0.35²) is at ρ_time = 24 wk, and the window differed too — so the two smokes are
not a clean comparison of kernels.

## Open

- [x] **Pathfinder smoke** — 24/24, 0 divergences; census above. NegBin criterion satisfied, centre
      unchanged.
- [ ] **10j §2c** — 12 points with t₀ at position 8. Outstanding since the `-w8h` rework; never run.
- [ ] **NUTS smoke** — same 3 origins. Outstanding since 2026-08-09.
- [ ] The `-w8h`/`-lc0` open items above carry over unchanged: the full-grid divergence census, and
      attributing the φ drop (now a ρ_time question) between window length, `-lc0` and the prior.

---

# 2026-08-10 (later) — reverted: the temporal kernel goes back to AR(1)

User request, after reading the `-m32t` census: "stop the fitting process and revert it back to the
AR(1)." **Cache token back to `temporal-w8h-lc0` (+`-nuts`)** — safe to reuse, because that
generation was never fitted (its smoke was killed at one file, which was deleted), so nothing on disk
has ever carried it.

## Why — the `-m32t` premise was testable and the smoke refuted it

`-m32t` argued that `-lc0` and `-w8h` had made the near-pooled temporal regime unreachable enough not
to matter. Both premises are true; the conclusion was not. Its own smoke put **3 of 12
hurdle-Weibull chains at or past the length of their own window** (12.0 wk on 9, 10.2 on 9, 24.4 on
12 — end-to-end within-window correlation 0.61–0.82, a field collapsed to one constant), under a
prior placing 8.7e-6 of its mass beyond 9 weeks. The pooled limit is still reached, so the question
is whether it is **safe** to visit — which is the original AR(1) argument (`Kt` min eigenvalue
2.7e-5 → 5.1e-3, `Lt` column spread 94.2 → 23.1 at matched effective rank).

⚠ **The revert does not fix the hurdle-Weibull pull**, and should not be expected to. That is its
likelihood (p⁰ ≈ 0.95 ⇒ nearly flat in time), not the kernel, now seen under four parameterisations.
The indicated action for that path is `constant_contacts = true`.

⚠ **NegBin was never the problem** and the two parameterisations agree on it quantitatively: φ median
0.726 under `-ar1` ⇒ ρ_time ≈ 2.0 wk at matched lag-1, against `-m32t`'s measured 1.44 wk with all 12
chains below the prior centre. Kernel choice here is a hurdle-Weibull question.

## Done

- [x] `joint_model.jl::model_degree` — `phi_time ~ Beta(ar1_phi_prior…)`, `Kt = φ^|s−t|`, clamp gone;
      the comment block now records the round trip and the measurement that ended it
- [x] `framework.jl` — `ar1_phi_prior` restored (⚠ flagged NEVER FITTED AT SCALE at (3,3)),
      `gp_time_len_prior` removed, `RHO_TIME_BOUNDS` dead again, token back to `temporal-w8h-lc0`
- [x] `8j/9j/12j` mirrors back to `phi_time`; `plot_lengthscales`' temporal panel back to φ ∈ (0,1)
      with the φ=1 line — **the two-panel split is KEPT** (weeks and age-years never matched either)
- [x] `10j_viz_utils.jl` — **fork logic unchanged**; only the "which is current" wording moved. This
      is what keeps the `-m32t` chains readable, i.e. what keeps the evidence for this revert alive
- [x] Docs: `CLAUDE.md`, `inst/3` §5/§6/§10 + new §12.6, `tasks/lessons.md`

## Kept from the `-m32t` commit (deliberately NOT reverted)

- 13j's missing `stage1_use_nuts` — it defaulted to `true`, pointed at a token with no artefacts and,
  since 13j calls `two_stage_forecast`, would have silently **refit serially** rather than erred
- 10j / 13j Notes that still described the superseded sliding window (`week_index = t_o − 1`)
- `joint_model.jl` stale latent counts (261/653 → per-horizon 293–389 / 734–977)
- `plot_lengthscales`' two-panel split and its `filter`-then-`isempty` robustness fix
- The `-m32t` census in `inst/3` §12.5 and above — it is the evidence for this revert

## Open

## Measured — head-to-head Pathfinder smoke, AR(1) vs Matérn on THE SAME 24 chains

24/24 s1 in 22 min, 72/72 s2 in 9 min. Same 3 origins × 4 horizons × 2 degree models, same window,
same seed, same AD backend — only the temporal correlation function differs. Compared on the two
scales that mean the same thing in both parameterisations: **lag-1 correlation** (φ for AR(1);
`m32(1/ρ_time)` for Matérn) and **end-to-end correlation** across the window (lag Tn−1), which is
what "the field has collapsed to a constant" actually means.

| family | kernel | div | lag-1 med | end-to-end med | nominal med | max\|z\| |
|---|---|---|---|---|---|---|
| unweighted-negbin | AR(1) | **0/12** | 0.487 | 0.001 | φ 0.49 | 2.82 |
| unweighted-negbin | Matérn | **0/12** | 0.660 | 0.000 | ρ 1.44 wk | 2.65 |
| weighted-hweibull | AR(1) | **0/12** | 0.748 | 0.070 | φ 0.75 | 4.29 |
| weighted-hweibull | Matérn | **0/12** | 0.942 | 0.148 | ρ 5.31 wk | 3.31 |

**Zero divergences under either kernel** — the revert is safe on that criterion, and so was `-m32t`.

**NegBin is indifferent to the kernel**, confirming this is a hurdle-Weibull question: no chain gets
anywhere near collapse under either (end-to-end ≤ 0.014 AR(1), ≤ 0.003 Matérn; 0/12 above 0.5 both).

**Hurdle-Weibull — a genuine improvement, but NOT a clean win.** Chains with end-to-end ≥ 0.5 fall
**3/12 → 1/12**, and AR(1) pools less in 7 of the 12 paired chains, including both of Matérn's worst
cases (0.679 → 0.244 and 0.815 → 0.001). ⚠ **But its single worst chain is worse**: at
2021-05-02 h1 AR(1) returned **φ = 1.000000 in every draw** (min = max — a point mass exactly on the
boundary) where Matérn stopped at ρ_time 8.3–13.9 with real spread. So the revert trades three
moderate collapses for one total one.

⚠ **Beta(3,3) did not prevent that boundary pile-up**, on the very first smoke, despite having
density → 0 at φ = 1 — which is the whole reason it was chosen over Uniform. Worth knowing before
anyone proposes a sixth prior.

⚠ **It is NOT the `-ar1` Uniform-prior catastrophe**: that chain's `log_eta` is −1.56 (≈ −3.1 prior
SD, against the −238 that generation produced), `log_sigma_c` −1.17, max\|z\| 4.29 — healthy by every
other measure, which is why it does not trip the divergence criterion. A φ pinned at exactly 1.0 with
zero spread is also the classic signature of **Pathfinder's normal approximation collapsing in a flat
direction**, which this repo already warns about under `RHO_TIME_BOUNDS`. Whether it is a real
posterior mode is exactly what the NUTS smoke answers.

⚠ AR(1)'s longer long-lag memory shows up as predicted (it was `-m32t`'s best argument): across all
24 pairs AR(1) has the *higher* end-to-end correlation in 16, though at NegBin's values (0.012 vs
0.002) that is a structural property, not a pathology.

## Measured — φ's INITIALISATION PROBE: φ is not identified by the data under Pathfinder

Prompted by "should φ also start from a 0.1 range, like `z`?". Answer: **no**, and the probe that
settles it is more important than the question. Same cell, same seed (`Xoshiro(1236)`, the driver's),
everything held fixed except φ's *starting value*. The shipped result reproduces exactly (φ₀ = 0.654
⇒ φ = 1.000000, log_eta −1.56, max|z| 4.38 — matching the artefact).

| φ₀ | hurdle-Weibull φ final | NegBin φ final |
|---|---|---|
| 0.100 | **0.011** | 0.150 |
| 0.300 | 0.670 | 0.319 |
| 0.500 (= "SD 0.1 on the logit scale") | 0.997 | 0.395 |
| 0.654 (prior draw, SHIPPED) | **1.000000** | 0.551 |
| 0.900 | 0.710 | 0.726 |

**The final φ is largely a function of where it started, on BOTH degree models.** Hurdle-Weibull
spans 0.011–1.000 and is not even monotone in the start; NegBin spans 0.150–0.726, monotone, with
only ~30% shrinkage toward the middle. Setting φ's init would therefore not fix the boundary
pile-up — **it would choose the answer**. Note the literal reading of the question is the worst
option tested: φ is logit-linked, so `N(0, 0.1²)` unconstrained means φ = 0.5 ± 0.025, and 0.5 lands
at 0.997.

**Why this is NOT analogous to the `z` fix.** There a diffuse start dropped the optimiser into the
saturated `[-8,6]` exponent clamp with no gradient to escape, and there WAS a right answer
(max|z| ≈ 3.5) the shrunk start could reach, after which results were stable. φ has no dead zone —
bounded support, finite gradient — and no stable answer to find. `z` also mattered because it is 251
of 293 coordinates entering composed as `η·(Q·La·z·Ltᵀ)`; φ is one scalar with no composed amplitude.

⚠ **CORRECTION to the head-to-head census above.** It reported NegBin as corroborating the kernel
correspondence (`φ = 0.726 ⇒ ρ_time ≈ 2.0 wk` against a measured 1.44). That is weaker than stated,
in two ways. (i) The 0.726 comes from the 252-fit **Uniform-prior** survey — a different prior from
the Beta(3,3) used here, so it is not a matched comparison. (ii) On the SAME 24 chains the two
parameterisations do **not** agree: AR(1) gives lag-1 0.487, Matérn gives 0.660. Combined with this
probe, the honest statement is: **NegBin is far better behaved than hurdle-Weibull — no boundary
collapse, no window collapse — but its φ is not well identified either, and no single Pathfinder φ
should be read as a posterior summary for either path.**

⚠ Consequence for everything φ-shaped that has been quoted from Pathfinder fits, including the
252-fit `-ar1` survey: those medians partly measure the optimiser's starting distribution. The
prior still does real work (Uniform → 0.726, Beta(2,2) → 0.568, Beta(3,3) → 0.487, all at prior
median 0.5), so it is not purely init-driven — but the data is the weakest of the three inputs.

## Open

- [ ] **NUTS smoke** — running under `temporal-w8h-lc0-nuts` (8/24 at ~2 h). Now the decisive run,
      and the specific question is sharper than before: NUTS starts from the Pathfinder mean, so for
      2021-05-02 h1 it starts AT φ = 1.0. **Does it walk away?** If it does, Pathfinder's φ is an
      artefact throughout and the `constant_contacts` conclusion needs re-deriving from NUTS. If it
      does not, the pooled limit is a genuine mode.
- [ ] **Re-examine whether φ should be reported from Pathfinder fits at all**, or only from NUTS.
- [ ] **NUTS smoke** — same 3 origins. Outstanding since 2026-08-09, and the trustworthy read on
      whether the pooled limit is a genuine posterior mode or a Pathfinder artefact.
- [ ] **`constant_contacts = true` for hurdle-Weibull** — now the fifth independent indication.
- [ ] Beta(3,3) has still never been fitted at scale; the 0/252 divergence result is the (2,2)
      measurement under `Tn = 12` and an AR(1) level.
