# Heavy-tailed social contact degree distributions for age-specific short-term forecasting of SARS-CoV-2 incidence in England

Research code accompanying the manuscript *(in preparation)*. It estimates **age-pair–stratified
contact *degree distributions*** from the CoMix UK social contact survey, converts them into
next-generation matrices (NGMs), and uses those in a weekly age-stratified renewal equation to
produce and score short-term forecasts of SARS-CoV-2 incidence in England.

- **Repository:** <https://github.com/toshiakiasakura/sc_heavy_tail_mean>
- **Language:** Julia 1.12, with R called through `RCall` for forecast scoring (`scoringutils`)
  and Dirichlet–multinomial regression (`MGLM`)
- **Archived version / DOI:** *to be minted on acceptance (see [Code availability](#code-availability))*

This README follows the *Guidelines for authors submitting code & software*
(<https://www.nature.com/documents/GuidelinesCodePublication.pdf>). The checklist items map onto
sections as follows.

| Guideline item | Section |
|---|---|
| Key operations performed by the code | [1. Overview](#1-overview) |
| Key characteristics: algorithms, dependencies | [1. Overview](#1-overview), [3. System requirements](#3-system-requirements) |
| Readme listing all documentation provided | [2. Repository contents](#2-repository-contents) |
| Installation guide: OS, language, dependencies, non-standard hardware, install time | [3. System requirements](#3-system-requirements), [4. Installation guide](#4-installation-guide) |
| Test dataset / example data and external dependencies | [5. Data](#5-data) |
| Demo on example data with typical run time | [6. Demo](#6-demo) |
| Instructions for use, and reproduction of published results | [7. Instructions for use](#7-instructions-for-use) |
| Licence of use | [9. Licence](#9-licence) |
| Code availability statement | [Code availability](#code-availability) |

---

## 1. Overview

### The task the code addresses

Age-stratified contact *matrices* summarise social mixing by its **mean** only. This code asks
whether the **heterogeneity beyond the age-stratified mean** — the heavy right tail of the number
of contacts per person per day, and the type (duration, group setting) of those contacts — carries
information that improves epidemic forecasts. It does so by fitting alternative contact models
inside an otherwise fixed forecasting framework and comparing their out-of-sample forecast skill,
extending the age-specific forecasting model of Munday et al. (2023).

### Key operations

1. **Contact data preparation** (`src/data_setup.jl`, `src/degree_dist.jl`,
   `src/comix_uk_time_series.jl`, `src/degree_agepair.jl`)
   Reads the CoMix UK participant and contact tables, derives each participant-day's **contact
   degree**, optionally **duration-weights** each contact (5 duration bins with midpoints
   2.5/10/37.5/150/240 min, capped at 4 h), assigns participants and contactees to the 7 ONS CIS
   `age_school` bins (2–10, 11–15, 16–24, 25–34, 35–49, 50–69, 70+), and assembles per-week,
   per-age-pair degree data on an inc2prev-aligned Sunday-start weekly grid.

2. **Heavy-tailed degree distribution fitting** (`src/distributions/`, `src/turing_models.jl`,
   notebooks `1j`–`7j`)
   A library of `Distributions.jl`-compatible count distributions — negative binomial,
   Poisson–log-normal, Poisson–Lomax, beta–negative-binomial, plus zero-inflated, zero-truncated and
   **convoluted** (home ⊛ non-home) wrappers — fitted by Hamiltonian Monte Carlo (Turing.jl) and
   compared by WAIC and by log–log CCDF fit in the tail.

3. **Two-stage ("cut") Bayesian inference for the forecasting model** (`src/joint_model.jl`,
   `src/framework.jl`)
   - **Stage 1 — contact degree.** Per age pair *and* per week, the mean contact degree is a
     separable spatio-temporal Gaussian process: a sum-to-zero weekly level plus a matrix-normal
     structure field with a **Matérn 3/2 × Matérn 3/2** kernel over rotated age coordinates (total
     age, age gap) and an **AR(1)** temporal correlation, sum-to-zero over the 28 age pairs within
     each week. Reciprocity is imposed in the mean (`log μ_{i→j} = r_{min,max} + log N_j`).
     Two likelihoods are offered: **negative binomial** on raw counts, and a **hurdle–Weibull** on
     duration-weighted degree. Sampling is NUTS (1000 adapt + 2000 draws, target accept 0.95,
     max tree depth 10), initialised from a Pathfinder mean. 293–977 latent parameters per fit.
   - **Stage 2 — transmission.** Conditioning on fixed Stage-1 draws (the *cut*), a weekly renewal
     equation `I(t) = Σ_{s=1}^{4} w(s)·N(t)·I(t−s)` with
     `N_ab = γ_SAR · susc_a · (1 + (F−1)·A_a(t)) · C*_ab · inf_b` (leaky antibody protection) is
     fitted by Pathfinder. 100 Stage-1 draws × 100 Stage-2 draws are pooled into 10 000 posterior
     forecast draws.

4. **NGM construction** (`src/ngm.jl`) — the per-capita contact term `C*` is built from the degree
   distribution's raw moments ⟨k⟩, ⟨k²⟩ by a swappable builder: `MeanNGM` (`C0 = ⟨k⟩`),
   `NeighbourhoodDegreeNGM` (excess/size-biased degree, `C0 = ⟨k²⟩/⟨k⟩`), `DiagonalMeanNGM`
   (no between-age-group transmission) and `NullNGM` (a constant, contact-data-free baseline).

5. **Forecasting and scoring** (`src/renewal.jl`, `src/scoring.jl`) — the NGM is frozen at the
   forecast origin and the renewal iterated 1–4 weeks ahead. Forecasts are scored against inc2prev
   age-stratified infection estimates by the **weighted interval score** (R `scoringutils` v2, on
   both the natural and the log scale; log-scale WIS is the headline), cross-checked by a native
   sample CRPS.

The design is deliberately **two swappable axes** — a contact-degree model × an NGM builder —
evaluated over a 2×2 grid plus two baselines (a no-interaction and a null model), across
**63 weekly forecast origins × 4 horizons**.

---

## 2. Repository contents

```
src/                       all analysis code (Julia) — see the notebook map below
  main_utils.jl            preamble 1: general analyses (distributions, DegreeDist, Turing stack)
  forecast_utils.jl        preamble 2: the forecasting framework (includes main_utils.jl)
  distributions/           custom Distributions.jl-compatible heavy-tailed count distributions
  framework.jl             swappable axes, weekly windows, FrameworkConfig, CIS age grid
  infection_data.jl        weekly infections and antibody prevalence from inc2prev estimates
  degree_agepair.jl        age-pair weekly contact-degree assembly
  renewal.jl  ngm.jl       generation interval + renewal iteration; NGM builders
  joint_model.jl           Stage-1 / Stage-2 models, fitting drivers, pooled forecast
  scoring.jl               WIS via R scoringutils + native CRPS
  8j_run_grid.jl           headless, resumable, memory-bounded driver for the full grid
  *_viz_utils.jl           read-only diagnostic and figure code for notebooks 8j–14j
  build_comix_uk.jl        one-time CoMix -> CoMix-UK data build (CSV -> Arrow)
inst/                      specifications — the source of truth for the analysis (see below)
hpc/                       Slurm + Singularity layer for running the grid on an HPC cluster
tasks/                     working plan (todo.md) and accumulated implementation notes (lessons.md)
.devcontainer/             the canonical, reproducible environment (Docker + VS Code devcontainer)
Project.toml, Manifest.toml  pinned Julia environment
build_sysimage.jl          optional PackageCompiler sysimage build (start-up time only)
precompile_script.jl       sysimage precompile workload; doubles as a smoke test
dt_comix_no_public/        CoMix input data — NOT redistributed (gitignored); see section 5
inc2prev/                  git submodule: population and infection/antibody estimates
dt_intermediate/           fitted MCMC artefacts (.jld2) — NOT redistributed
res/, res_summary/         generated figures and score tables
```

### Documentation provided

| File | Contents |
|---|---|
| `README.md` | this file — installation, demo, and instructions for use |
| `inst/analysis_plan_heavy_tail_mean.md` / `.docx` | the analysis plan; **the `.docx` is authoritative** where it and the code disagree |
| `inst/3_preliminary_model_struct.md` | full specification of the forecasting framework and both stages |
| `inst/2_weekly_age_pair.md` | specification of the weekly age-pair contact assembly (notebook 7j) |
| `inst/pcbi.1011453.pdf` | Munday et al. (2023), the model this work extends |
| `hpc/README.md` | runbook for the Slurm + Singularity execution of the fitting grid |
| `CLAUDE.md` | developer notes: architecture, design decisions and their measured justifications |
| `tasks/lessons.md` | chronological record of modelling decisions, dead ends and their evidence |

### Notebook map

Notebooks are numbered in execution order and stored **without outputs** (`execution_count: null`).

| Notebook | Purpose |
|---|---|
| `1j_data_explore.ipynb` | CoMix UK data exploration and the UK subset build |
| `2j_proportion_duration_physical.ipynb` | contact duration and physical-contact composition |
| `3j_effective_degree_distribution.ipynb` | duration-weighted ("effective") degree distributions |
| `4j_Danon_degree_duration.ipynb` | **legacy** — its raw data is no longer in the tree; not runnable |
| `5j_group_contacts.ipynb` | mass/group contact reports and their weighting |
| `6j_fitting_dist.ipynb` | heavy-tailed distribution fitting and comparison |
| `7j_weekly_age_pair.ipynb` | weekly age-pair contact assembly (spec: `inst/2_weekly_age_pair.md`) |
| `8j_preliminary_forecast.ipynb` | **fitting** — runs the two-stage grid (`src/8j_run_grid.jl` is the headless equivalent) |
| `9j_forecast_diagnostics.ipynb` | forecast assembly, WIS scoring, transmission-parameter figures |
| `10j_model_diagnostics.ipynb` | Stage-1 model diagnostics (§1–§6) and GP hyperparameters vs prior (§7) |
| `11j_weekly_identifiability.ipynb` | mean vs neighbourhood-degree identifiability over weeks |
| `12j_chain_convergence.ipynb` | Stage-1 NUTS convergence: ESS, split-R̂, E-BFMI, tree depth |
| `13j_model_diagnostics_h1.ipynb` | a horizon-1-only variant of 10j |
| `14j_publication_figures.ipynb` | **publication figures**, plus a full audit of the fitted grid (§0) |

---

## 3. System requirements

### Operating systems tested

| Environment | OS | Status |
|---|---|---|
| VS Code devcontainer (`.devcontainer/Dockerfile`) | Linux, from `quay.io/jupyter/datascience-notebook:julia-1.12.4` | **canonical — all results produced here** |
| LSHTM HPC, Slurm + Singularity (`hpc/Dockerfile.hpc`) | Linux, Singularity image built from the same Julia 1.12.4 base | used for the fitting grid only |

The code contains no OS-specific calls and is expected to run on any platform supporting Julia
1.12 and R, but only the two environments above have been tested. A Docker-capable machine is
therefore the recommended route on macOS and Windows.

### Software dependencies

- **Julia 1.12** (`[compat] julia = "1.12"`; the container ships 1.12.4). The full package set is
  pinned in `Manifest.toml`; `Project.toml` additionally pins the packages that determine the
  fit's numerics: `Turing 0.42`, `DynamicPPL 0.39`, `Distributions 0.25`, `Pathfinder 0.9`,
  `Mooncake 0.5`, `ReverseDiff 1`. Other direct dependencies include `MCMCChains`, `JLD2`,
  `DataFrames(Meta)`, `Arrow`, `CSV`, `Plots`/`StatsPlots`, `RCall`, `IJulia`.
- **R ≥ 4.0**, reached from Julia through `RCall`. Required packages: **`scoringutils` (v2 — v1 is
  not compatible; 2.2.0 used here)**, `MGLM`, `data.table`. The devcontainer additionally installs
  `socialmixr`, `broom`, `tableone`, `jsonlite`, `here`.
- **Jupyter** (supplied by the base image) for the `.ipynb` entry points.
- **Docker** (for the devcontainer route) or **Singularity + Slurm** (for the HPC route).

### Hardware

No non-standard hardware (no GPU, no accelerator) is required. However, the **full fitting grid is
computationally heavy** and its resource envelope is not that of a typical laptop:

| Task | CPU | Memory | Notes |
|---|---|---|---|
| Notebooks `1j`–`7j`, `9j`–`14j` (analysis, diagnostics, figures) | 2–4 cores | 8–16 GB | runs comfortably on a desktop |
| Stage 1 of the full grid (504 chains) | ≥ 8 cores; the devcontainer sets `JULIA_NUM_THREADS=12` | ≥ 32 GB, and ≈4 GB per concurrent fit | HPC array job: 8 CPUs, 64 GB, 24 h per task |
| Stage 2 of the full grid (1512 pooled cells) | ≥ 16 cores | 32 GB | HPC array job: 16 CPUs, 32 GB, 6 h per task |

⚠ **Memory grows with the number of forecast origins processed in one process** (measured
≈1.08 GiB per origin, and not reclaimed by `GC.gc()`). The driver therefore fits a bounded batch of
origins and exits; a supervisor script relaunches it in a fresh process. Do not remove this
structure — a single process fitting all 63 origins was OOM-killed at origin 37 with 26.8 GiB
resident.

---

## 4. Installation guide

### Option A — devcontainer (recommended, and the environment all results were produced in)

Requires Docker and (optionally) VS Code with the Dev Containers extension.

```bash
git clone --recurse-submodules https://github.com/toshiakiasakura/sc_heavy_tail_mean.git
cd sc_heavy_tail_mean
# VS Code: "Reopen in Container" — the image build and Pkg.instantiate() run automatically
```

The `postCreateCommand` runs `Pkg.instantiate()` for you. Equivalently, without VS Code:

```bash
docker build -t sc-heavy-tail -f .devcontainer/Dockerfile .
docker run -it --rm -v "$PWD":/workdir -w /workdir sc-heavy-tail \
    julia --project=/workdir -e 'using Pkg; Pkg.instantiate()'
```

### Option B — existing Julia 1.12 + R installation

```bash
git clone --recurse-submodules https://github.com/toshiakiasakura/sc_heavy_tail_mean.git
cd sc_heavy_tail_mean
julia --project=. -e 'using Pkg; Pkg.instantiate()'
Rscript -e 'install.packages(c("scoringutils", "MGLM", "data.table"))'   # scoringutils v2
```

If the submodule was not cloned recursively: `git submodule update --init inc2prev`.

### Optional — sysimage

Reduces `using Turing/Plots/...` start-up from minutes to seconds. It affects **start-up time
only**, not results, and is not required.

```bash
julia --project=. build_sysimage.jl          # 15–30 min, produces ./sysimage.so (~1 GB)
julia --project=. --sysimage=./sysimage.so   # use it
```

### Typical install times

Approximate, on a current workstation with a broadband connection:

| Step | Time |
|---|---|
| `Pkg.instantiate()` + first-use precompilation of the Turing stack | ~20–40 min |
| Devcontainer image build (includes the R packages above) | ~30–60 min |
| R `scoringutils` alone, into an existing R | ~2–5 min |
| Optional sysimage build | 15–30 min |
| HPC Singularity image (`hpc/build_image.ps1`: build + docker2singularity) | 50–100 min; needs ~40 GB free disk |

---

## 5. Data

### Inputs

| Source | Files | Role | Redistributed here? |
|---|---|---|---|
| **CoMix UK** social contact survey | `dt_comix_no_public/{part,contacts}.csv` → `{part_uk,contacts_uk}.arrow` | participant-days and their reported contacts (the explanatory data) | **No** — see below |
| **inc2prev** (from the ONS COVID-19 Infection Survey) | `inc2prev/outputs/estimates_age_ab.csv` | age-stratified estimated infection incidence and antibody prevalence in England (the outcome data) | **Yes** — openly available, pinned as a git submodule |
| **inc2prev** | `inc2prev/data-processed/populations.csv` | CIS age bins and England population sizes | **Yes** — openly available, pinned as a git submodule |

The **inc2prev** inputs are external and versioned: the submodule is pinned to commit
`78e3d8717219c4238fb9dc971405d0562695a6f3` of <https://github.com/epiforecasts/inc2prev>. Only the
two files above are read. Note two schema quirks handled in `src/infection_data.jl`:
`name == "infections"` rows are a **daily per-capita proportion** (multiplied by population and
7-day-summed for weekly counts), and `name == "gen_dab"` (antibody) rows have an **empty `date`
column**, so dates are reconstructed from `t_index`.

The **CoMix UK** individual-level data are **not redistributed in this repository**
(`dt_comix_no_public/` is gitignored) because they are participant-level social contact records
held under the CoMix study's data-sharing terms. The UK survey is described in Gimma A, Munday JD,
Wong KLM, et al. *Changes in social contacts in England during the COVID-19 pandemic between March
2020 and March 2021 as measured by the CoMix survey.* PLoS Med. 2022;19(3):e1003907. Access should
be requested from the CoMix data custodians at the London School of Hygiene & Tropical Medicine;
the code expects the consortium's released tables (`part.csv`, `contacts.csv`, and the data
dictionary `dd_v1238_20230328.xlsx`) placed in `dt_comix_no_public/`.

Once those files are in place, build the UK subset once:

```bash
cd src && julia --project=.. build_comix_uk.jl   # filter country == "uk", write CSV + Arrow
```

### Test dataset

Because the contact data cannot be redistributed, the repository does not ship an example CoMix
extract. Two forms of test material are available instead, and both exercise the code paths
end-to-end:

1. **A self-contained numerical smoke test with no restricted data at all** — the heavy-tailed
   distribution library, the generation-interval discretisation, the NGM builders and the full
   Turing/AD sampling stack, exercised on synthetic values with analytically checkable answers
   (see [Demo](#6-demo), Demo 1).
2. **A 3-origin subset of the real analysis**, selected by environment variables and reproducing
   exactly the code path of the full run (Demo 2). This requires CoMix access.

The external dependencies of both are exactly those in
[System requirements](#3-system-requirements); results are deterministic given the fixed seed
(`FrameworkConfig.seed = 1236`) and the pinned `Manifest.toml`.

---

## 6. Demo

### Demo 1 — self-contained smoke test (no restricted data; ~2–5 min after installation)

```bash
julia --project=. precompile_script.jl
```

Exercises `Distributions`, `DataFrames`, `Plots` and `Turing` — including three short NUTS runs,
one per AD backend the framework selects between — and an asserted Arrow round-trip. It returns
exit status 0 with no error output on success.

For a direct check of the distribution library and the forecasting core on synthetic values —
still with no restricted data — run from `src/`:

```bash
cd src && julia --project=.. -e '
    include("forecast_utils.jl")                 # the whole framework preamble

    pl = PoissonLomax(2.0, 1.5)                  # heavy-tailed (power-law) count distribution
    zi = ZeroInfDist(0.3, pl)                    # with 30% structural zeros
    @show pdf(zi, 0) pdf(zi, 5) mean(zi)
    @show pdf(pl, 10) pdf(pl, 100) pdf(pl, 1000) # the tail

    w = gen_interval_pmf(5.0, 5.0; smax = 4)     # weekly generation-interval PMF
    @show w sum(w)

    K1 = [1.0 0.5; 0.5 2.0]                      # synthetic raw moments <k> and <k^2>,
    K2 = [3.0 1.0; 1.0 8.0]; G = ones(2, 2)      # and per-cell zero factors
    @show contact_star(MeanNGM(), K1, K2, G)
    @show contact_star(NeighbourhoodDegreeNGM(), K1, K2, G)
'
```

*Expected output:*

- `mean(zi)` = 1.05 exactly (= (1 − π₀)·θ/(α − 1)), and finite probabilities throughout;
- a tail that falls by roughly three orders of magnitude per decade of *k* — the Lomax power law
  `k^−(α+1)` at α = 2 — rather than exponentially;
- a 4-element generation-interval PMF summing to 1;
- `MeanNGM` returning `K1` unchanged and `NeighbourhoodDegreeNGM` returning
  `[3.0 2.0; 2.0 4.0]` = `K2 ./ K1`, which is elementwise ≥ the mean degree, as the excess
  (size-biased) degree must be.

*Expected run time:* < 1 min with a sysimage, ~2–3 min without (dominated by package load).

### Demo 2 — 3 forecast origins of the real pipeline (requires CoMix access)

First, a zero-cost preflight that resolves the configuration, prints the cache token and the list
of origins it would fit, and exits **10** without fitting anything (seconds):

```bash
cd src
DRY_RUN=1 STAGE1_USE_NUTS=true ORIGIN_MIN=2021-04-25 FIT_END=2021-05-09 \
    julia --project=.. 8j_run_grid.jl
```

Then the fit itself — 3 origins × 4 horizons × 2 degree models = 24 Stage-1 chains, followed by
Stage 2:

```bash
cd src
export STAGE1_USE_NUTS=true ORIGIN_MIN=2021-04-25 FIT_END=2021-05-09
PHASE=s1 julia --project=.. 8j_run_grid.jl     # relaunch while it exits 0; exit 10 = phase done
PHASE=s2 julia --project=.. 8j_run_grid.jl
```

*Expected output:* 24 files `dt_intermediate/8j_s1_<degree>_temporal-w8h-lc0-nuts_<origin>_h<h>.jld2`
and 72 files `8j_s2_<degree>_<ngm>_..._h<h>.jld2`, each a JLD2 archive carrying the chain plus
self-describing provenance keys (`sampler`, `ad_backend`, `target_accept`, `nuts_adapts`,
`nuts_draws`, `phi_init_scale`, `phi_pf_max`). Progress, the resolved token and per-fit NUTS
diagnostics (divergences, minimum ESS, tree depth) are printed to stdout.

*Expected run time:* Stage 1 dominates, at ≈28/42/47/50 min per fit at horizons 1/2/3/4
(≈17 CPU-hours for the 24 chains), run concurrently across available threads — **≈3–6 h of
wall-clock on a 12-thread workstation**, depending on how many concurrent fits memory allows.
Stage 2 adds ≈0.2 min per pooled cell (≈15 min for the 72 cells). Setting
`STAGE1_USE_NUTS=false` substitutes Pathfinder for NUTS in Stage 1 and is far faster, but produces
a **different generation of artefacts** (a different cache token) that must not be mixed with
NUTS results.

⚠ The exit-code protocol is deliberate: `0` = work done and more remains (relaunch in a **fresh
process**, which is what bounds memory), `10` = this phase is complete, anything else = a genuine
failure. `hpc/run_grid.sh` implements the loop.

---

## 7. Instructions for use

### Running on your own data

All framework behaviour is set through one immutable configuration struct, `FrameworkConfig`
(`src/framework.jl`), whose defaults are those used in the manuscript: `n_fit = 8` fitting weeks,
`smax = 4` renewal lags, `horizons = 1:4`, `constant_contacts = false` (per-week contact means),
`seed = 1236`, `n_stage1_post = 100` × `n_stage2_draws = 100` pooled draws, Stage 1 by NUTS
(1000 adapt + 2000 draws, `target_accept = 0.95`, `max_depth = 10`) with the Mooncake AD backend,
Stage 2 by Pathfinder with ReverseDiff.

To substitute your own contact survey, provide participant and contact tables in the CoMix column
convention (see `src/data_setup.jl` and `src/build_comix_uk.jl`) and your own weekly incidence and
population files in the inc2prev column convention (`src/infection_data.jl`). The age grid comes
from `cis_age_grid()` and the weekly grid is anchored at `WEEK_ANCHOR = 2021-03-21`
(Sunday-start weeks, Wednesday mid-date labels).

**Always run scripts and notebooks from `src/`** — every relative path (`../dt_comix_no_public/…`,
`../inc2prev/…`, `../dt_intermediate/…`) assumes `cwd == src/`.

Notebooks are executed headless with:

```bash
cd src
jupyter nbconvert --to notebook --execute 10j_model_diagnostics.ipynb \
    --ExecutePreprocessor.timeout=-1 --output-dir /tmp/nb_out
```

`--ExecutePreprocessor.timeout=-1` is required (the 30 s default kills the preamble). Do **not**
use `--inplace`: notebooks are stored without outputs by design.

### Reproducing the published results

The pipeline is: **build data → fit the grid → score and diagnose → make figures**.

1. **Build the CoMix UK subset** (once): `cd src && julia --project=.. build_comix_uk.jl`.

2. **Fit the grid.** 63 weekly origins (2020-10-18 … 2021-12-26) × 4 horizons, for the two
   Stage-1 degree models (504 Stage-1 chains) and six model combinations (1512 Stage-2 artefacts).
   Either open `src/8j_preliminary_forecast.ipynb`, or run the headless driver under a supervisor:

   ```bash
   cd src
   export STAGE1_USE_NUTS=true FIT_END=2021-12-31
   while PHASE=s1 julia --project=.. 8j_run_grid.jl; do :; done   # stops on exit 10
   while PHASE=s2 julia --project=.. 8j_run_grid.jl; do :; done
   ```

   On a cluster, use `hpc/` instead: edit `hpc/config.sh` (account, project directory, image, and
   the generation selectors), then `sbatch hpc/s1_array.slurm` and, once it completes,
   `sbatch hpc/s2_array.slurm`. Each 16-task array partitions the origins round-robin via
   `ORIGIN_STRIDE`/`ORIGIN_OFFSET` so that tasks write disjoint artefacts. See `hpc/README.md`.

   **Total cost of the full grid (measured):** Stage 1 ≈ 351 CPU-hours (per-fit ≈1690/2510/2810/3030 s
   at horizons 1–4); Stage 2 ≈ 5 h of wall-clock at the default per-cell concurrency (≈0.2 min per
   pooled cell × 1512 cells). On the 16-task HPC arrays Stage 1 is roughly a day of wall-clock.
   ⚠ Stage 2's AD backend is load-bearing: with Mooncake instead of ReverseDiff the same 1512 cells
   would take ≈17 days, because Stage 2 is 100 independent fits of an 18-dimensional model per cell
   and per-fit setup, not gradient throughput, dominates.

3. **Audit the fitted grid before analysing it.** Notebook `14j` §0 (`audit_stage1_grid`) checks
   coverage (504/504), structural agreement with the current model, and **uniformity of
   provenance** across `sampler`, `ad_backend`, `target_accept`, `nuts_adapts`, `nuts_draws`,
   `phi_init_scale` and `phi_pf_max`. Posterior draws are not bit-identical across AD backends or
   initialisation settings, so a partially refitted grid is mixed-provenance and must be
   regenerated under one setting before publication.

4. **Score and diagnose.** Run in order: `9j` (forecast assembly and WIS scoring), `10j` (Stage-1
   model diagnostics), `11j` (weekly identifiability), `12j` (NUTS convergence). Outputs are
   written to `res/`.

5. **Publication figures.** Run `14j_publication_figures.ipynb`. It writes
   `res/14j_*` and `res/14j_fit_*.csv`, and triggers no fitting.

### Environment variables

| Variable | Read by | Default | Meaning |
|---|---|---|---|
| `STAGE1_USE_NUTS` | `8j`, `9j`, `10j`, `11j`, `13j`, `14j`, `8j_run_grid.jl` | `true` in the notebooks, `false` in `8j_run_grid.jl` | selects the Stage-1 sampler **and hence which generation of cached artefacts is read** |
| `FIT_END` | `8j`, `9j`, `14j`, `8j_run_grid.jl` | `2021-12-31` (⇒ 63 origins) | last forecast origin |
| `ORIGIN_MIN` | `8j`, `9j`, `8j_run_grid.jl` | unset (all origins) | first forecast origin |
| `PHASE` | `8j_run_grid.jl` | `s1` | which stage to advance (`s1` or `s2`) |
| `MAX_ORIGINS`, `CHUNK`, `MEM_FLOOR_GIB` | `8j_run_grid.jl` | `8`, `4`, `8.0` | per-process memory bounds |
| `ORIGIN_STRIDE`, `ORIGIN_OFFSET` | `8j_run_grid.jl` | `1`, `0` | disjoint origin slices for a Slurm array |
| `S1_CONCURRENCY`, `S2_CONCURRENCY` | `8j`, `8j_run_grid.jl` | derived from cores and free memory | concurrent fits |
| `DRY_RUN` | `8j_run_grid.jl` | unset | resolve and print, then exit 10 without fitting |
| `ORIGIN_12J`, `ORIGIN_13J` | `12j`, `13j` | `2021-05-09` | the single origin those notebooks examine |
| `AD_BACKEND` | `8j_run_grid.jl` | `mooncake` | Stage-1 AD backend |

⚠ `STAGE1_USE_NUTS` and `FIT_END` **must agree between the fitting step and every downstream
notebook**. They select the cache token; pointing a downstream notebook at a generation that was
never fitted does not raise an error — the framework would silently refit it. The diagnostic
notebooks therefore assert up front that every required artefact exists; do not remove those
assertions.

---

## 8. Reproducibility and verification

- **Determinism.** All sampling is seeded (`FrameworkConfig.seed = 1236`; the distribution-fitting
  strand uses `Random.seed!(1236)`). Results are reproducible given the same `Manifest.toml`,
  Julia version and AD backend. They are **not** bit-identical across AD backends or across
  different Stage-1 initialisation settings, both of which are recorded in every artefact.
- **Environment pinning.** `Project.toml` pins `julia = "1.12"` and the numerics-defining packages;
  `Manifest.toml` pins the complete dependency tree; the devcontainer and HPC images pin the
  interpreter and the R packages. A Julia/Manifest mismatch is a hard resolve error by design.
- **Artefact provenance.** Each fitted chain stores its own sampler, AD backend and sampler
  settings; `audit_stage1_grid` (notebook `14j` §0) verifies these are uniform across the grid.
- **No formal unit-test suite.** Verification is by (i) `precompile_script.jl` as a smoke test,
  (ii) the diagnostic notebooks `9j`–`13j` (convergence, identifiability, prior–posterior
  comparison, forecast calibration), and (iii) targeted spot-checks of constructed distributions
  via `pdf` / `ccdf` / `mean`. Several internal consistency checks are asserted at run time — for
  example, that the contact and infection week vectors align, that per-block degree marginals equal
  the sum of their age-pair components plus unbinnable contacts, and that the neighbourhood
  (excess) degree is never below the mean degree.
- **Known non-runnable code.** `src/danon_utils.jl` and `src/4j_Danon_degree_duration.ipynb` belong
  to a discontinued analysis strand whose raw data is not in the tree. They are retained for
  provenance and are **not** part of the published analysis.

---

## 9. Licence

*Not yet assigned.* No `LICENSE` file is currently present, so the code is under default copyright
and is being shared for peer review only. An OSI-approved licence (MIT is intended) will be added
to the repository root, and stated here and in the manuscript's Code Availability statement, before
publication.

Third-party components keep their own licences: the `inc2prev` submodule is licensed by its
authors (see `inc2prev/LICENSE`), and the CoMix data are governed by the CoMix study's data-sharing
terms rather than by this repository's licence.

---

## Code availability

The code that generates all results in the manuscript is available at
<https://github.com/toshiakiasakura/sc_heavy_tail_mean>. The exact version used will be archived
with a DOI on Zenodo at acceptance and cited in the manuscript; this README will be updated with
the DOI and the corresponding tag. No restrictions on code availability are anticipated beyond the
licence, which will be OSI-approved.

Individual-level CoMix UK contact data are not included in this repository and are available from
the CoMix study custodians at the London School of Hygiene & Tropical Medicine under their
data-sharing terms; all other inputs (inc2prev estimates and populations) are openly available and
are included as a pinned submodule.

## Citation

Please cite the accompanying manuscript *(in preparation)* and, where the software itself is used,
the archived release (DOI to follow).

This work builds directly on:

- Munday JD, Abbott S, Meakin S, Funk S. *Evaluating the use of social contact data to produce
  age-specific short-term forecasts of SARS-CoV-2 incidence in England.* PLoS Comput Biol.
  2023;19(9):e1011453. <https://doi.org/10.1371/journal.pcbi.1011453>
- Gimma A, Munday JD, Wong KLM, et al. *Changes in social contacts in England during the COVID-19
  pandemic between March 2020 and March 2021 as measured by the CoMix survey.* PLoS Med.
  2022;19(3):e1003907. <https://doi.org/10.1371/journal.pmed.1003907>
- Abbott S, Funk S, et al. *inc2prev*: <https://github.com/epiforecasts/inc2prev>

## Contact

Toshiaki Asakura, London School of Hygiene & Tropical Medicine —
<https://github.com/toshiakiasakura>. Please open an issue on the repository for questions about
running the code.
