# 7. Data sources of the forecasting framework (audit, 2026-10-22)

What every Stage-1 / Stage-2 / scoring input is, where it is read, and what changed on the
`refactor` branch (user request: England-only contacts, ONS CIS outcome, Monday weeks, origins
through Jan 2022). Cache token of the new generation: `temporal-w8h-lc0-eng` (+`-nuts`).

## Summary

| role | before (`temporal-w8h-lc0-nuts`) | after (`temporal-w8h-lc0-eng-nuts`) |
|---|---|---|
| Stage 1 contacts | CoMix **UK** participants | CoMix **England** participants (`england_only`) |
| Stage 2 target, renewal history, scoring truth | inc2prev **modelled infections** (`estimates_age.csv`, 7-day sum × pop) | **ONS CIS positivity by age × pop** (`cis_age.csv`, weekly mean × pop) |
| Antibody covariate `A` | `estimates_age.csv` `gen_dab` (absent ⇒ `A ≡ 0`, `F = 1`) | unchanged |
| Population / age grid | `populations.csv` `age_school`, 7 bins 2-10 … 70+ | unchanged |
| Variant composition | COG-UK England, Monday weeks, read from the week starting the day after each Sunday week | same file, now the SAME week (`cog_week` = identity) |
| Week grid | Sunday-start, anchor 2021-03-21 | **Monday-start, anchor 2021-03-22**, Thursday mid labels |
| Forecast origins | 63, 2020-10-18 … 2021-12-26 | **65, 2020-11-02 … 2022-01-24** |

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

- **File**: `inc2prev/data-processed/cis_age.csv` — ONS COVID-19 Infection Survey **modelled** PCR
  positivity (MRP model, with 95% credible interval), England, `level == "age_school"`, the 7 CIS
  bins keyed by `lower_age_limit` ∈ {2, 11, 16, 25, 35, 50, 70} (= `grid.LO`). Daily rows
  (`start_date == end_date`, asserted), **2020-08-14 … 2023-01-31**, 118 stacked publication
  vintages (1–7 per date × bin).
- **Reader**: `cis_weekly_positivity` (moved verbatim from `14j_viz_utils.jl` to `infection_data.jl`)
  keeps the LATEST vintage per (date, bin) and takes the weekly MEAN (positivity is a stock).
- **Outcome**: `weekly_cis_positives` → `I_mean = P × POP` (people currently positive),
  `I_sd = mean daily (HI − LO)/(2·1.96) × POP`. It asserts 7 days and a finite, positive value in
  every (bin, week) cell.
- ⚠ **Prevalence used as incidence (user's choice).** The renewal and its likelihood are unchanged,
  so γ_SAR and the generation interval absorb the stock/flow mismatch (weekly positivity × pop ≈
  1.24 weekly incidences given a ~8.7-day mean detectable duration, and lagged). γ_SAR from these fits
  is not a per-contact attack rate on the incidence scale.
- ⚠ There are **no observed positive COUNTS by age** in the tree. ONS's unweighted England counts
  (`dt_ons_variant/ons_cis_england_sample_counts_14day.csv`) have no age split and end 2021-03-13.
- **First origin**: the 12-week infection window (4 lag + 8 fit) must start on or after the first
  full Monday week of CIS data, 2020-08-17 ⇒ origin **2020-11-02** (`available_forecast_origins`'
  `infection_start`). An origin a week earlier fails the coverage assert.

## Covariates

- **Antibody** `A_a(t)`: `inc2prev/outputs/estimates_age.csv` `gen_dab` — absent from this file, so
  `A ≡ 0`; inert while `F = 1` is pinned (unchanged; see CLAUDE.md's schema gotcha).
- **Population**: `inc2prev/data-processed/populations.csv` (`grid.POP`), England, unchanged.
- **Variants** (`VariantNGM`): `dt_cog_uk/cog_uk_england_weekly.csv`, COG-UK England genomes per
  Monday sample week by constellation group (untracked; must be uploaded with the repo for HPC runs).

## Audits that still read inc2prev infections

14j §6 (`incidence_age_shares`) and §6c/§6d, and 16j's inc2prev panels, compare inc2prev's fits
with the CIS and UKHSA data. They read `weekly_infections` on purpose and are unchanged.
