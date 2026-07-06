# 1d — Implementation plan for the 8j fixes (`inst/1c_fix_for_preliminary.md`)

Executable plan for the three fixes in `1c`. Source-of-truth ordering: the
analysis-plan docx > `1c` > `1a`/`1b`. Two design forks were confirmed with the user
(2026-07-06) and are recorded under **Decisions** below.

## Context

The preliminary 8j forecast (`src/8j_preliminary_forecast.ipynb` + framework modules)
currently:
- computes the **neighbourhood-degree NGM** from the *zero-included* size-biased
  degree ⟨k²⟩/⟨k⟩, so the empirical zero fraction `p0` **cancels** and has no effect
  (`ngm.jl` `base_contact(::NeighbourhoodDegreeNGM,k1,k2)=k2/k1`; cf. `1b §3`);
- forecasts all 4 horizons from **one** fit at a single origin with the NGM
  **frozen** at the origin (`joint_model.jl:posterior_forecast`; `1b §9`).

`1c` requires: (1) build the neighbourhood NGM **among those with >0 contacts** so
the zero fraction re-enters; (2) forecast assuming **only the contact matrix is
available at h−1** and **re-estimate at each iteration**, saving MCMC per horizon;
(3) forecast **4 weeks forward** from the single baseline.

Intended outcome: the four-way grid stays non-degenerate and the neighbourhood NGM
reflects the hurdle/zero structure; the forecast mimics a real-time run where the
degree distribution is refreshed weekly while infections are not yet observed past
the baseline.

## Decisions (confirmed with user)

1. **Neighbourhood C0 = docx configuration-network form** `Ccf = (z²/z) ⊙ Pnz`
   (size-biased degree × per-cell non-zero factor). This overrides the "(1−p0)
   cancels" reading in `1a`/`1b §3`.
2. **Forecast scheme = contact-updated iterate**: single baseline `t₀`; infections
   and antibody frozen at `t₀`; for each `h=1..4` re-fit with the contact/degree
   window ended at `t₀+h−1`, refresh the NGM, take one renewal step; save MCMC per
   horizon.

## Point 1 — neighbourhood-degree NGM among non-zero degrees

Preserve the two-axis design: the **degree family** produces per-cell moments **plus
a zero factor `g`**; the **NGM builder** stays a pure moment functional. Only the
neighbourhood builder uses `g`; the mean builder is unchanged.

**Raw moments (unchanged; `joint_model.jl:_negbin_moments`/`_weibull_moments`):**
- NegBin: ⟨k⟩=μ, ⟨k²⟩=μ+μ²(1+1/φ).
- Weibull: ⟨k⟩=(1−p⁰)μ_W, ⟨k²⟩=(1−p⁰)μ_W²(1+CV²_W),
  CV²_W=Γ(1+2/κ)/Γ(1+1/κ)²−1.

**Per-cell zero factor `g_ab` (new):**
- NegBin: `g = 1/(1 − P₀)`, `P₀ = (φ/(φ+μ))^φ` (model-based zero prob of the fitted
  NegBin — "left-truncate the fitted NegBin").
- Weibull: `g = (1 − p⁰)`, `p⁰` empirical (`ds.p0[i,j]`, the hurdle zero prob).

**C0 (`ngm.jl`):**
```julia
base_contact(::MeanNGM, k1, k2, g)                = k1            # ⟨k⟩ — unchanged
base_contact(::NeighbourhoodDegreeNGM, k1, k2, g) = (k2/k1) * g  # size-biased × zero factor
```
which yields the confirmed forms:
- NegBin neighbourhood: `(μ+1+μ/φ)/(1−P₀)`  = ⟨k²⟩/⟨k⟩ ÷ (1−P₀).
- Weibull neighbourhood: `μ_W(1+CV²_W)·(1−p⁰)`.

**⚠ Numerical guard (must-have).** For NegBin, μ→0 ⇒ P₀→1 ⇒ `g→∞` (near-empty cells
blow up). Floor the denominator: `g = 1/max(1−P₀, ε)`, `ε≈1e-3`, computed *inside*
the model where the existing μ clamp (`exp∈[3e-4,400]`) already bounds it — mirrors
the Weibull/κ clamps in `tasks/lessons.md`. Add a verification check on `max(C*)` in
sparse cells; if a sparse cell still dominates the NGM, report it (candidate
fallback: the symmetric multiply form `(k2/k1)·(1−P₀)` — but implement the ÷ form as
chosen).

**Files:**
- `src/ngm.jl` — add `g` arg to `base_contact` (both builders); thread a per-cell
  `G` matrix through `contact_star(nb, K1, K2, G, pop)` and the convenience
  `build_ngm(builder, K1, K2, G, susc, inf, F, A_col, pop)`.
- `src/joint_model.jl` — in the `model_joint` cell loop, also accumulate `G[i,j]`
  (dispatch on `is_weighted(dm)` using μ, φ/κ, `ds.p0`), then
  `Cstar = contact_star(nb, K1, K2, G, wd.pop)`. Reciprocity balance unchanged.

## Points 2 & 3 — contact-updated iterated 4-week forecast with saved chains

The joint model already takes degree stats `ds` and infection window `wd` as
**separate** args, so decoupling "contact to h−1 / infections to t₀" needs **no model
change** — pass a shifted `ds` with the baseline `wd`.

**New `iterated_forecast(dm, nb, wd0, cfg, win0; grid, setting, use_nuts, save_dir,
label)` in `joint_model.jl`** (keep `posterior_forecast` for reference):
- `hist = collect(wd0.I_mean)` (A×12, last col = `t₀`); `w = gen_interval_pmf(...)`.
- For `h = 1..4`:
  - `win_h = WeeklyWindow(win0.origin + Day(7*(h-1)); n_fit, smax, horizons)` — a
    12-week contact window **ending at `t₀+h−1`** (slides forward).
  - `ds_h = build_degree_stats(dm, prepare_degree_data(win_h, cfg; grid, setting), cfg)`.
  - **Fit-or-load** `model = model_joint(dm, nb, ds_h, wd0, w, cfg)`:
    `path = save_dir/"8j_chn_$(degree_label(dm))_$(ngm_label(nb))_$(win0.origin)_h$(h).jld2"`;
    `isfile(path)` → `chn = load(path,"result")`, else
    `chn = fit_joint(...).chn; jldsave(path; result=chn)`
    (mirrors `bnb_utils.jl:47-55`).
  - `gq = generated_quantities(model, chn)` → per-draw `q`.
  - **One renewal step** for week `t₀+h`, NGM refreshed, antibody frozen at `t₀`:
    `N = build_ngm(q.Cstar, q.susc, q.inf, q.F, wd0.antibody[:, end])`;
    `pred = N * Σ_{s=1}^{smax} w[s]·hist[:, end-s+1]`; noise
    `σ = max(q.sigma_inf*pred, 1e-6)` → `out[:, h, d]`.
  - Append the **mean** forecast to `hist` (`hist = hcat(hist, meandraws(out[:,h,:]))`)
    so horizon `h+1` uses it as a lag (plug-in-mean iterated path; per-draw coherence
    across independent re-fits is undefined).
- Return `out :: A × H × ndraws`.

**Driver (`src/8j_preliminary_forecast.ipynb`) changes:**
- Build `wd0`/`truth` from `WeeklyWindow(Date(2021,1,3))` **once** (unchanged).
- **Precompute** `apd_h` for `h=1..4` once (shared across all four combos → avoid 16
  redundant Arrow reads), then `ds_h[dm]` per family.
- Loop the 4 combos: `fc = iterated_forecast(dm, nb, wd0, cfg, win0; …,
  save_dir="../dt_intermediate")`; then `to_quantile_long`/`mean_crps` as today.
- Scoring, WIS table, and fan plot **unchanged** (`fc` still A×H×draws;
  `truth = load_forecast_truth(win0)`).
- Keep `USE_NUTS=false` default (16 fits; Pathfinder tractable, save/skip makes
  re-runs cheap). Seed forecast noise `MersenneTwister(cfg.seed + h)`.

**Files:** `src/joint_model.jl` (new `iterated_forecast` + fit-or-load helper),
`src/8j_preliminary_forecast.ipynb` (precompute + combo loop). `dt_intermediate/`
gains 16 `8j_chn_*.jld2` files.

## Doc & lessons updates
- `inst/1b_8j_model_structure.md` — update §3 (neighbourhood C0 = size-biased × zero
  factor; NegBin `/(1−P₀)`, Weibull `×(1−p⁰)`) and §9 (contact-updated iterate, not
  frozen-NGM); note it overrides the "(1−p0) cancels" statement.
- `tasks/lessons.md` — add: (i) NegBin neighbourhood `/(1−P₀)` blow-up + floor;
  (ii) contact/infection decoupling via separate `ds`/`wd` args; (iii) per-horizon
  chain save/skip convention.
- `tasks/todo.md` — add a `8j — 1c fixes` block.

## Verification
- **Unit:** `gen_interval_pmf` sums to 1; reciprocity `props_a·C*_ab==props_b·C*_ba`;
  `C*≥0` and **finite** (spot-check sparse child↔70+ cells for blow-up);
  Neighbourhood ≥ Mean per cell for both families.
- **Point 1:** on a constructed cell, confirm NegBin neighbourhood
  `=(μ+1+μ/φ)/(1−P₀)` and Weibull `=μ_W(1+CV²_W)(1−p⁰)` numerically; confirm the four
  combos give **four distinct** contact matrices (no Mean≡Neighbourhood collapse).
- **Points 2/3:** run the driver headless (`jupyter nbconvert --to notebook --execute
  --inplace`); confirm 16 `8j_chn_*.jld2` written; a second run **loads** them (no
  re-fit); `fc` is A×4×draws; finite 4-row WIS table + `res/8j_*` refreshed.
- Diff `res/8j_scores_by_model.csv` vs the pre-fix run to confirm the scheme changed
  the scores (expect less over-prediction of the lockdown-3 decline).

## Out of scope
Keep the lean skeleton (constant-within-window contacts, reduced transmission block,
RW1/GP still the swap-in seam). No change to data loaders, scoring, or the age grid.
