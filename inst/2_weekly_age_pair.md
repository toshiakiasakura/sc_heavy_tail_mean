# 2 — 7j Weekly age-pair mean & excess degree

## Context

6j §8/§8b decomposes the duration-weighted degree by **age pair**
*(participant CIS bin → contactee CIS bin)* for a single aggregated window
(Jan–Jun 2021), summarising each cell's fitted Weibull by mean and CV. This task
adds a **longitudinal** view in a new notebook `src/7j_weekly_age_pair.ipynb`:
track, **per week**, the **mean** and the **excess degree**
`m(1+CV²) = ⟨k²⟩/⟨k⟩` (mean of the size-biased degree) for each age pair, plus the
**number of sampled participant-days** per pair per week so coverage is visible
alongside the trends.

Confirmed design choices:
- **Empirical moments** per week/cell (not per-cell Weibull fits — unstable on
  sparse weekly cells; the excess is a pure moment formula).
- **Full CoMix-UK span** 2020-03-23 → 2021-06-23 (all waves).
- **Positives-only per cell** (zero-excluded): a participant-day enters cell *i→j*
  only if it had ≥1 contact to bin *j*, so zero answers are excluded by construction.

## Reused building blocks (do not reinvent)

- `read_comix_uk_contact_raw()` (`src/data_setup.jl:789`) — full-span contacts +
  participants (`part_id_d`, `part_age`, `date`).
- `read_arrow_df` / `standardise_cnt_home_values!` / `_uk_duration_multi`
  (`src/data_setup.jl`) — build the contactee-age contact table `craw`, as 6j cell
  `2c5c78a3`.
- `duration_weight(d, DMAX=240)` (`src/degree_dist.jl:215`) — per-contact weight.
- `inc2prev_week_anchor()` (`src/comix_uk_time_series.jl:56`) — 2021-03-21
  Sunday-start week anchor.
- 6j §8 bin logic (cell `6eeca66a`): `age_school` population grid,
  `parse_age_interval`, `interval_from_minmax`, `overlapping`, `assign_bin`
  (population-weighted draw for ambiguous ages), `is_ambiguous`.

## Age grid & sampling rules

- **Grid** = `age_school` rows of `inc2prev/data-processed/populations.csv`
  (England): `2-10, 11-15, 16-24, 25-34, 35-49, 50-69, 70+`.
- **Ages < 2 ignored**: grid already starts at 2 (`@subset :lo ≥ 2`); any reported
  age interval lying entirely below 2 is dropped (`assign_bin → nothing`).
- **Age-category / ambiguous ages**: when a reported interval overlaps several
  bins (contactee "0–17", missing / "Don't know", etc.), draw the bin by
  **weighted random sampling ∝ bin population** — `sample(rng, cand,
  Weights(CIS_POP[cand]))`, single seed `MersenneTwister(1236)`.
- **Age pairs** are formed for **contactor (participant) × contactee** bins:
  `part_bin` drawn once per participant-day, `cnt_bin` once per contact.

## Weekly binning

inc2prev-aligned 7-day weeks (Sunday-start), anchored at `inc2prev_week_anchor()`
(2021-03-21); each date tagged by its **mid-date** (Wednesday):

```
week_index(d)  = fld((d - WK_ANCHOR).value, 7)
week_mid_of(d) = WK_ANCHOR + Day(week_index(d)*7 + 3)
```

## Per-cell statistics (empirical, zero-excluded)

For each **week × (participant bin i → contactee bin j) × setting**, take the
per-participant-day weighted degree (sum of `duration_weight` over that day's
contacts into the cell — positive by construction) and compute:

- `n`  = number of sampled participant-days (the requested sample count),
- `mean` = m,
- `cv`   = std / mean,
- `excess` = m(1 + cv²).

## Notebook cells

1. **md** — overview.
2. **code** — `include("main_utils.jl")` + `data_setup.jl` + `comix_uk_time_series.jl`
   + `vis_utils.jl`; `default_plot_setting()`.
3. **§1 code** — `df, df_part = read_comix_uk_contact_raw()` (full span); `DMAX=240`;
   week helpers.
4. **§2 code** — `age_school` bins (`lo ≥ 2`); bin-parse/assign helpers; `assign_bin`
   returns `nothing` for intervals entirely < 2.
5. **§3 code** — build `craw`; draw `part_bin`/`cnt_bin`, drop `nothing`; `dfA` with
   per-contact `week_mid`.
6. **§3 code** — `week_cell_stats(setting)` → `statsall = vcat(home, nonhome)`.
7. **§4 code** — per-setting 7×7 grid (row = participant bin, col = contactee bin);
   each panel: weekly **mean** (solid black) + **excess** (dashed red), markers,
   shared axes, `MIN_CELL = 10` display filter. Save
   `res/7j_agepair_mean_excess_weekly_{home,nonhome}.png`.
8. **§4 code** — `me_grid(:home)`, `me_grid(:nonhome)`.
9. **§5 code** — 7×7 counts grid: weekly `n` per pair, home (blue) vs non-home
   (red), log-y, all weeks. Save `res/7j_agepair_counts_weekly.png`.
10. **md** — notes/takeaways.

## Outputs

- `src/7j_weekly_age_pair.ipynb` (new).
- `res/7j_agepair_mean_excess_weekly_home.png`,
  `res/7j_agepair_mean_excess_weekly_nonhome.png`,
  `res/7j_agepair_counts_weekly.png`.
- No existing source files modified.

## Verification

- Execute headless:
  `jupyter nbconvert --to notebook --execute src/7j_weekly_age_pair.ipynb --inplace`.
- Spot-check `statsall`: `excess ≥ mean` everywhere; `n ≥ 1`; weeks span
  2020-03 → 2021-06; diagonal home cells densest.
- Confirm the three PNGs render and inline grids show plausible trends (non-home
  cells show the widest mean↔excess gap).
