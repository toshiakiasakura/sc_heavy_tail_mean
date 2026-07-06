# 1a — Preliminary composable forecasting framework

Implementation spec for `inst/1_preliminary_framework.md`. Source paper:
`inst/pcbi.1011453.pdf` (Munday et al. 2023). Full model: `inst/analysis_plan_heavy_tail_mean.docx`.

## Context

Extend the Munday et al. 2023 age-stratified COVID-19 forecasting model. The
paper is a weekly renewal-equation / next-generation-matrix (NGM) model,

```
I(t) = Σ_{s=1}^{s_max} w(s) · N(t−s) · I(t−s),   N(t) = γ·diag(s)·C(t)·diag(i)
```

fit to age-stratified infection estimates on a rolling 8-week window and scored
with CRPS/bias/coverage. The analysis-plan **extension** replaces the plugged-in
contact matrix `C(t)` with **age-pair contact-degree distributions** and derives
the NGM two ways: from cell **means** (age-stratified matrix) and from
**neighbourhood/excess degree** `⟨k²⟩/⟨k⟩` (configuration network,
Saumell-Mendiola 2012 Phys Rev E).

**This is the preliminary run**: implement the framework, fit **one** forecast
origin (first week of 2021) on an **8-week** window, and report **forecasting
scores four ways**. Fitting uses **Pathfinder.jl** for init and **Turing.jl**
for the formal fit. The contact-degree and renewal/NGM parts are **composable**
(to be added to / swapped later).

### Confirmed decisions
1. **"Four ways" = the 2×2 model grid**: {Unweighted **Negative Binomial**,
   Weighted **Hurdle-Weibull**} degree models × {**Mean** NGM,
   **Neighbourhood-degree** NGM}. Four score rows.
2. **Joint single model**: one Turing model carries the contact-data likelihood
   *and* the infection likelihood (faithful to the paper). Composability is kept
   *inside* the joint model (see Architecture).
3. **Lean skeleton first**: both real degree models and both real NGM methods
   end-to-end producing the true 4-way scores, but with **simplified temporal
   smoothing** (random-walk instead of Matérn GP) and a **reduced
   transmission-parameter set**. GP + full hierarchy drop in later via the seams.

## Grounded facts (from exploration)

- 7 CIS age bins `("2-10","11-15","16-24","25-34","35-49","50-69","70+")`;
  England `age_school` populations from `inc2prev/data-processed/populations.csv`
  = `[6_254_603, 3_370_248, 5_950_637, 7_596_145, 10_853_151, 13_618_246, 7_679_719]`.
- `inc2prev/outputs/estimates_age_ab.csv`: `name=="infections"` is **daily** per
  age (2020-07-31→2022-03-26) with `mean,sd,q5..q95`; `name=="est_ab"` is
  **weekly** (2020-12-10→2021-12-02) = antibody prevalence `A_a(t)`.
- Week grid: Sunday-start, `inc2prev_week_anchor() = Date(2021,3,21)`
  (`src/comix_uk_time_series.jl:56`); `week_index(d)=fld((d-anchor).value,7)`.
- **Renewal machinery does not exist in Julia** — port from
  `inc2prev/stan/functions/rt.stan` (`discretised_gamma_pmf`, Cori
  `update_infectiousness`). No NGM, serial interval, infection loader, or CRPS
  exist yet. **No GP package** installed. `Pathfinder`, `Turing`, `Distributions`,
  `ForwardDiff`, `Optim`, `RCall`, `KernelDensity` are present; R `scoringutils`
  is **not** installed.

### Reused building blocks (do not reinvent)
- `contact_degrees(df, df_part; setting, weighted, d_max)` and
  `duration_weight(d, d_max)` — `src/degree_dist.jl:249,215`. `DegreeDist`,
  `merge_dd`.
- `fit_model_with_forward_mode(model, n; iparms)` — Pathfinder→NUTS, seed 1236,
  `src/turing_models.jl:4`; `@model` patterns beside it (e.g.
  `model_ZeroInfNegativeBinomial:21`); `calculate_loglikelihood(dd,dist)`
  `src/turing_utils.jl:29`; `is_chains_converged` `src/turing_utils.jl:162`.
- `NegBin(m,k)`: `mean=m`, `var=m+m²/k` ⇒ `CV²=1/m+1/k`
  (`src/distributions/poisson_mixture.jl`); hurdle/zero-trunc helpers in
  `src/distributions/zerotrunc.jl` + `helper.jl`.
- Age-pair binning from 6j cell `6eeca66a` / 7j: `age_school` grid (`lo ≥ 2`),
  `parse_age_interval`, `interval_from_minmax`, `assign_bin` (population-weighted
  draw for ambiguous ages, `MersenneTwister(1236)`) — spec in
  `inst/2_weekly_age_pair.md`.
- `read_comix_uk_contact_raw()` (`src/data_setup.jl:789`) — full-span contacts +
  participants; `inc2prev_week_anchor()` (`src/comix_uk_time_series.jl:56`).

## Concrete preliminary run

- **Forecast origin week** = first Sunday-start week of 2021 =
  **2021-01-03 → 2021-01-09** (mid Wed 2021-01-06).
- **8-week fit window** ending at the origin (paper convention: fit up to and
  including the forecast date): weeks starting
  `2020-11-15, 11-22, 11-29, 12-06, 12-13, 12-20, 12-27, 2021-01-03`.
- **Infections**: aggregate the daily inc2prev `infections` series to those 8
  Sunday-start weeks (sum `mean`; combine `sd` in quadrature) per age →
  `I_mean[a,t]`, `I_sd[a,t]`. First 4 weeks seed the `s_max=4` renewal; weeks 5–8
  give the in-window infection likelihood.
- **Antibody**: `A_a(t)` from `est_ab`; weeks predating 2020-12-10 (W1–W3)
  **backfilled constant** with the earliest available value.
- **Forecast horizons 1–4 weeks** → target weeks
  `2021-01-10, 01-17, 01-24, 01-31`; `C(t)` and `A_a(t)` **held at origin**
  (no future-data leakage). Score against realized weekly-aggregated infections.
- **CoMix waves in window** derived empirically from `date` of
  `read_comix_uk_contact_raw()` over 2020-11-15→2021-01-09 (do not hard-code
  wave numbers).

## Architecture — composable joint model

Two swap axes via Julia abstract types + dispatch:

```julia
abstract type ContactDegreeModel end
struct NegBinAgePair        <: ContactDegreeModel end   # unweighted
struct HurdleWeibullAgePair <: ContactDegreeModel end   # duration-weighted

abstract type NGMBuilder end
struct MeanNGM                <: NGMBuilder end
struct NeighbourhoodDegreeNGM <: NGMBuilder end
```

- The **NGM axis is a deterministic function** `build_ngm(::NGMBuilder, μ, cv2, aux, t)`
  → 7×7 matrix; it adds no parameters, so it is passed as an argument into the
  joint model and swapped freely.
- The **degree axis is the model variant**: two `@model` functions
  (`model_joint_negbin`, `model_joint_hweibull`) that sample the latent per-cell
  weekly mean structure and add the contact-data likelihood; both take an
  `nb::NGMBuilder` argument used only through `build_ngm`. → **2 model defs × 2
  builders = 4 joint fits.** Shared transmission-sampling + renewal-likelihood
  code lives in a helper `include`d by both.

Joint model body (both variants):
1. Sample latent contact means `log μ_xy(t)` (lean: **RW1** over the 8 weeks,
   per cell, non-centred) + dispersion/shape pooled by child/adult block.
2. Add **contact-data likelihood** from CoMix (NegBin on integer degrees via
   `calculate_loglikelihood`; Hurdle-Weibull accumulates `logpdf(Weibull,·)` on
   the positive weighted per-participant-day degrees, zeros carried by empirical
   `p0`).
3. Sample **transmission latents** (lean reduced set): `s_inh` (7, sum-to-zero,
   hierarchical σ), antibody effectiveness `F ~ Beta`, overall `log γ_SAR`,
   obs-noise `CV`; **fix infectivity `i=1`** for the preliminary (swap-in later).
4. Build per-week `N(t) = γ_SAR·diag(s(t))·C*(t)·diag(i)` with
   `s_a(t)=s_inh,a·(1+(F−1)·A_a(t))` and `C*` from `build_ngm(nb,…)`.
5. Add **infection likelihood** `I_obs[a,t] ~ Normal(Î_a(t), σ_a(t))`,
   `Î` from the renewal, `σ_a(t)=sqrt(I_sd² + (CV·I_mean)²)`.

Fit with `fit_model_with_forward_mode` (Pathfinder init → NUTS). Forecast by
posterior draws iterating the renewal 1–4 weeks with held `C`,`A`.

### NGM math (concrete)
Per cell (x=participant/susceptible age, y=contactee/infectious age): latent
per-capita mean `m_xy`, `CV²_xy`, empirical `p0_xy`; populations `N_x`.
- Effective per-capita contacts (the only differing line):
  - `MeanNGM`: `C0_xy = m_xy`.
  - `NeighbourhoodDegreeNGM`: `C0_xy = m_xy·(1+CV²_xy) = ⟨k²⟩/⟨k⟩`
    — NegBin closed form `m_xy + 1 + m_xy/k`; Hurdle-Weibull uses positive-part
    moments (the `(1−p0)` cancels), matching the 7j empirical `excess = m(1+cv²)`.
- Reciprocity (`N_x·μ_xy = N_y·μ_yx`) by symmetrising totals:
  `T_xy = N_x·C0_xy`, `C*_xy = (T_xy + T_yx)/(2·N_x)`.
- Then `s_a(t) = s_inh,a·(1+(F−1)·A_a(t))` and
  `N_ab(t) = γ_SAR·s_a(t)·C*_ab(t)·i_b`.

### Serial interval & renewal
Port `rt.stan`: `gen_interval_pmf(mean_days, sd_days; s_max=4, step=7)` via
gamma-CDF differences renormalised over `1..4`. Keep `mean/sd` **configurable**;
flag that the inc2prev default (3.64d/3.08d) collapses to ~all mass on `s=1`
weekly — real serial-interval params (analysis plan cites Sang Woo Park 2024
medRxiv) to be confirmed before scientific interpretation.

### Scoring (native Julia, no R dependency)
Over posterior-predictive samples: `crps_sample` (Gneiting–Raftery energy form),
`bias_sample` (scoringutils convention, range [-1,1]), `covered` (50% & 90%
central intervals). `score(pred, truth)` aggregates CRPS/bias/coverage over
7 ages × 4 horizons → one row per model variant.

## File layout (new; do NOT edit `main_utils.jl`)

The driver notebook includes `main_utils.jl` then these explicitly (convention):
- `src/framework.jl` — abstract types, subtypes, `WeeklyWindow`, containers,
  `nameof` helpers.
- `src/infection_data.jl` — loader for `estimates_age_ab.csv` (infections +
  est_ab) with weekly aggregation to the Sunday grid.
- `src/degree_agepair.jl` — age-pair binning reused from 6j/7j, weekly Sunday
  binning, per-cell weekly degree assembly (`contact_degrees(...; weighted)`),
  empirical `p0`, counts.
- `src/renewal.jl` — `gen_interval_pmf`, renewal iteration/forecast (port of
  `rt.stan`).
- `src/ngm.jl` — `build_ngm(::NGMBuilder,…)`, reciprocity, susceptibility, `NGMAux`.
- `src/joint_model.jl` — `model_joint_negbin`, `model_joint_hweibull`, shared
  transmission+renewal likelihood helper, `fit_joint`, posterior-predictive.
- `src/scoring.jl` — `crps_sample`, `bias_sample`, `covered`, `score`.
- `src/8j_preliminary_forecast.ipynb` — driver: load data, define window, run the
  4 joint fits (2 degree models × 2 NGM builders), emit the 4-row score table +
  diagnostic plots under `res/8j_*`.

### Four output score rows
| degree model | NGM method | mean CRPS | mean bias | cov50 | cov90 |
|---|---|---|---|---|---|
| Unweighted (NegBin) | Mean | … | … | … | … |
| Unweighted (NegBin) | Neighbourhood | … | … | … | … |
| Weighted (Hurdle-Weibull) | Mean | … | … | … | … |
| Weighted (Hurdle-Weibull) | Neighbourhood | … | … | … | … |

(Optionally also emit the 16-row per-horizon breakdown, h = 1..4.)

## Lean simplifications (swap-in path noted)
- Temporal: **RW1** per cell (not Matérn GP). GP swaps in inside the `@model`.
- Transmission: **fix `i=1`**; keep `s_inh, F, γ_SAR, CV`.
- `p0` empirical per cell (per analysis plan). `C(t)` latent but strongly pinned
  by the CoMix likelihood.
- Single window, single origin, plug-in posterior for forecasts.

## Implementation phases
1. `src/infection_data.jl` + `src/framework.jl` + `src/degree_agepair.jl`
   (data + skeleton); smoke-test degree assembly on the window.
2. `src/renewal.jl` + `src/ngm.jl` + `src/scoring.jl` (deterministic pieces;
   unit-check `gen_interval_pmf` sums to 1, CRPS vs a hand case, reciprocity).
3. `src/joint_model.jl` — the two `@model`s; Pathfinder→NUTS on one variant.
4. `src/8j_preliminary_forecast.ipynb` — run all four, produce the score table.

## Verification
- `gen_interval_pmf` sums to 1; reciprocity `N_x·C*_xy == N_y·C*_yx`; NGM
  entries ≥ 0; `NeighbourhoodDegreeNGM` entries ≥ `MeanNGM` entries.
- Each joint fit passes `is_chains_converged` (ESS>200, Rhat<1.1) or divergences
  are reported; Pathfinder init used.
- Weekly-aggregated infections match a manual sum of daily inc2prev rows for one
  age/week; antibody backfill only touches W1–W3.
- `crps_sample` matches a small closed-form check; coverage ∈ [0,1].
- Notebook executes headless:
  `jupyter nbconvert --to notebook --execute src/8j_preliminary_forecast.ipynb --inplace`
  and produces the 4-row score table plus `res/8j_*` plots.

## Open items folded as defaults (revisit before scientific use)
Serial-interval params (Park 2024 vs inc2prev default); antibody backfill vs
`A=0` for W1–W3; contact-index orientation (participant = susceptible `a`,
contactee = infectious `b`); whether reciprocity constrains the excess matrix or
only the mean; fixing `i=1`.
