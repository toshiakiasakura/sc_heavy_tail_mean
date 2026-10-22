# 7. Data sources of the forecasting framework (audit, 2026-10-22)

What every Stage-1 / Stage-2 / scoring input is, where it is read, and what changed on the
`refactor` branch (user request: England-only contacts, Monday weeks, origins through Jan 2022).
Cache token of the new generation: `temporal-w8h-lc0-eng` (+`-nuts`). The outcome was briefly switched
to ONS CIS positivity and switched back the same day (see "Stage 2 and scoring").

## Summary

| role | before (`temporal-w8h-lc0-nuts`) | after (`temporal-w8h-lc0-eng-nuts`) |
|---|---|---|
| Stage 1 contacts | CoMix **UK** participants | CoMix **England** participants (`england_only`) |
| Stage 2 target, renewal history, scoring truth | inc2prev **modelled infections** (`estimates_age.csv`, 7-day sum × pop) | unchanged — now with a full-coverage assert per week |
| Antibody covariate `A` | `estimates_age.csv` `gen_dab` (absent ⇒ `A ≡ 0`, `F = 1`) | **removed** with `F` (2026-10-22) |
| Population / age grid | `populations.csv` `age_school`, 7 bins 2-10 … 70+ | unchanged |
| Variant composition | COG-UK England, Monday weeks, read from the week starting the day after each Sunday week | same file, now the SAME week (`cog_week` = identity) |
| Week grid | Sunday-start, anchor 2021-03-21 | **Monday-start, anchor 2021-03-22**, Thursday mid labels |
| Forecast origins | 63, 2020-10-18 … 2021-12-26 | **67, 2020-10-19 … 2022-01-24** |

## Stage 1 — CoMix contacts

- **Files**: `dt_comix_no_public/part_uk.arrow` (276 482 participant-days, 466 columns) and
  `contacts_uk.arrow` (1 325 891 contacts). Read by `read_comix_uk_raw_contacts_and_part`
  (`data_setup.jl`) via `load_raw_contact_inputs` (`degree_agepair.jl`).
- **Geography**: `area_3_name` (ITL1 region, 0 missing, 12 values). England = the nine English
  regions; `NON_ENGLAND = (Scotland, Wales, Northern Ireland)`. `contacts_uk.arrow` carries only
  `country == "uk"`, so the filter is on participants; contacts follow by the inner join on
  `(part_id, date)` (0 orphan contacts measured).
  - England keeps **234 320 / 276 482 participant-days (84.75%)** and 1 134 313 / 1 325 891 contacts.
  - Filter per ROW: 650 participants report from different nations in different waves.
  - Do **not** use `area_3_code` (164 non-English rows coded 1 = North East), `uk_region1` (28 408
    missing), `uk_stdregion`/`area_2_name` (mixed regions such as "South West & Wales") or
    `part_region_uk` (all missing).
- **Span**: main panel 2020-03-23 … Wed 2022-03-02, plus a stray 2022-11-16 … 11-28 block. The last
  complete Monday week is 2022-02-21 ⇒ the last origin with a complete h=4 contact week is
  **2022-01-24**.
- The descriptive notebooks (1j–7j) still read the UK roster: `england_only` is applied only on the
  framework path (`load_raw_contact_inputs` and `prepare_degree_data`'s fallback).

## Stage 2 and scoring — the outcome

- **File**: `inc2prev/outputs/estimates_age.csv`, `name == "infections"` — inc2prev's modelled daily
  per-capita infections by the 7 CIS bins, England, **2020-07-31 … 2023-01-31**, fitted to ONS CIS
  positivity alone. Weekly count = 7-day sum × population (`weekly_infections`); SDs in quadrature.
- **Coverage**: `load_window_data`/`load_forecast_truth` assert every week read is fully covered
  (`_assert_infection_coverage`) — `weekly_infections` zero-fills a missing day silently. First full
  Monday week 2020-08-03 ⇒ first origin **2020-10-19** (`available_forecast_origins`'
  `infection_start`); an origin a week earlier fails the assert.
- **Not `estimates_age_ab.csv`** (the joint positivity + antibody fit, the outcome until 2026-09-16):
  1j's comparison shows its weekly infections nearly FLAT through the spring-2021 trough in 2-10, 11-15,
  16-24 and 35-49 (up to ~10–20× the positivity-only fit), with the totals within a few %.
- **Considered and reverted (2026-10-22): ONS CIS positivity × population** (`cis_age.csv`,
  `weekly_cis_positives`, still in `infection_data.jl`). inc2prev's fitted prevalence reproduces it
  closely (weekly ratio median 0.996, 90% within 0.89–1.11; 1j), but it is a PREVALENCE: 0.6–2.3× the
  weekly infections week to week around a steady-state D/7 = 1.24 (D = 8.7 days detectable, inc2prev's
  `prob_detectable.csv`), a scale and lag the renewal would have to absorb. ⚠ There are no observed
  positive COUNTS by age in the tree.

## Covariates

- **Antibody** `A_a(t)`: none. It was `estimates_age.csv` `gen_dab` — absent from that file, so `A ≡ 0`
  under a pinned `F = 1` — and both were removed from the code on 2026-10-22.
- **Population**: `inc2prev/data-processed/populations.csv` (`grid.POP`), England, unchanged.
- **Variants** (`VariantNGM`): `dt_cog_uk/cog_uk_england_weekly.csv`, COG-UK England genomes per
  Monday sample week by constellation group (untracked; must be uploaded with the repo for HPC runs).

## Audits that still read inc2prev infections

14j §6 (`incidence_age_shares`) and §6c, 16j's inc2prev panels and 1j's comparisons set inc2prev's
fits beside the CIS data. They read `weekly_infections` on purpose.

## Removed (2026-10-22)

`dt_ons_variant/` (ONS SGTF gene-pattern shares, 16j §2–§3 and 1j) and `dt_ukhsa/` (UKHSA lineage %,
reported cases and single-year populations — 1j, 14j §6c's fourth line and §6d) were deleted: variant
proportions come from COG-UK alone (`dt_cog_uk/`, which also holds 1j's processed weekly shares,
`cog_uk_england_weekly_share.csv`). Both folders were untracked, so git holds no copy.
