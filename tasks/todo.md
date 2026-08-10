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
**Cache token: `temporal-w8h-lc0` (+`-nuts`)** — the nine accumulated historical suffixes were
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
