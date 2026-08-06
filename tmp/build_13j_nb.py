#!/usr/bin/env python3
"""Generate src/13j_model_diagnostics_h1.ipynb from src/10j_model_diagnostics.ipynb.

13j is 10j wired for a SINGLE horizon (h=1). It exists because 10j structurally needs h1-h4
(cell 4 forecasts over cfg.horizons, cell 5 builds a fit_store_h4, 2b/2c read mu across h1-h4
and from "the h4 chain"), and running it against h1-only artefacts does NOT fail loudly:
two_stage_forecast -> fit_or_load_stage2 FITS when an artefact is missing, so it would silently
launch full NUTS Stage-1 fits for h=2,3,4.

The h1 lever is one field: FrameworkConfig(horizons = 1:1). Everything downstream derives from it
  - H = length(cfg.horizons)            (10j_viz_utils.jl)
  - win.forecast_weeks                  (framework.jl)
  - make_mu_horizon_fig(...; hz = collect(cfg.horizons))
  - make_mu_timeline_fig(...; h_chain = maximum(cfg.horizons))
so NO change to 10j_viz_utils.jl is needed and none is made here.

RES13: figure filenames hard-code the "10j_" prefix INSIDE 10j_viz_utils.jl (only tag/file_tag/
res_dir are caller-controlled), so a default-res_dir run would overwrite the existing res/10j_*.png.
Every saving call therefore gets res_dir = RES13. The three non-saving helpers
(plot_dispersion_cells, plot_tau_over_weeks, plot_p0_vs_empirical) are left alone.

Regenerate with:  python3 tmp/build_13j_nb.py     (never hand-edit the notebook)
"""
import json
import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "10j_model_diagnostics.ipynb")
DST = os.path.join(os.path.dirname(__file__), "..", "src", "13j_model_diagnostics_h1.ipynb")

nb = json.load(open(SRC))
cells = nb["cells"]
_applied = []


def sub(idx, old, new, why):
    """Replace `old` with `new` in cell `idx`, asserting exactly one match."""
    src = "".join(cells[idx]["source"])
    n = src.count(old)
    if n != 1:
        sys.exit(f"FATAL cell {idx}: {n} matches (expected 1) for {old[:70]!r}\n  ({why})")
    cells[idx]["source"] = (src.replace(old, new)).splitlines(keepends=True)
    _applied.append((idx, why))


def replace_cell(idx, text, why):
    cells[idx]["source"] = text.strip("\n").splitlines(keepends=True)
    _applied.append((idx, why))


# ── 0. header ────────────────────────────────────────────────────────────────
replace_cell(0, """
# 13j — single-time-point model diagnostics, **h = 1 only**, two-stage cut

A **horizon-1** variant of `10j_model_diagnostics.ipynb`. Identical in content; it differs in one
config field, `horizons = 1:1`.

**Why it exists.** 10j structurally needs Stage-1 and Stage-2 artefacts at **h1–h4**: it forecasts
over `cfg.horizons`, builds an in-sample fit store from the h4 artefacts, and §2c reconstructs μ
from "the single h4 chain". Crucially, running 10j against h1-only artefacts does **not** fail
loudly — `two_stage_forecast` → `fit_or_load_stage2` **fits when an artefact is missing**, so it
would silently launch full NUTS Stage-1 fits for h = 2, 3, 4. This notebook exists so an h1-only
generation of chains can be diagnosed without that trap.

**What is lost relative to 10j**, stated plainly:

- §1/§1b show a **1-week** forecast fan, not 4, and the in-sample fit is drawn once (from the h=1
  chain) rather than twice (h=1 and h=4);
- §2b's "μ across horizons" collapses to a **single** horizon point per cell — the panel survives
  but carries much less information;
- §2c's timeline is reconstructed from the **h1** chain, so it spans the 8 fit weeks ++ **h1**
  (12 weeks) rather than ++ h1–h4. Still a genuine per-week μ trace from one fit, just a shorter
  forecast reach.

§3 (contact matrices), §4 (degree distributions), §5 (susc/inf) and §6 (dispersion, p⁰) anchor at
the origin week t₀ and read the h=1 chain in 10j too, so they are **unchanged**.

`ORIGIN` is settable via `ENV["ORIGIN_13J"]` (mirroring 12j's `ORIGIN_12J`). Figures go to
**`res/13j/`** — the filenames still carry the `10j_` prefix (it is hard-coded in
`10j_viz_utils.jl`), so the separate directory is what stops them overwriting 10j's own output.
""", "13j header")

# ── 1. preamble: RES13 ───────────────────────────────────────────────────────
sub(1, 'mkpath("../res")',
    'mkpath("../res")\n\n'
    '# Figure filenames hard-code the "10j_" prefix inside 10j_viz_utils.jl (only tag/file_tag/\n'
    '# res_dir are caller-controlled), so writing to the default "../res" would OVERWRITE 10j\'s\n'
    '# own PNGs for this origin. Every saving call below passes res_dir = RES13.\n'
    'const RES13 = "../res/13j"\n'
    'mkpath(RES13)',
    "define RES13")

# ── 2. config: horizons = 1:1, ORIGIN from env ───────────────────────────────
sub(2, "cfg  = FrameworkConfig(constant_contacts = false)",
       "# `horizons = 1:1` IS the h1-only lever — every horizon-dependent consumer derives from it\n"
       "# (H = length(cfg.horizons), win.forecast_weeks, make_mu_horizon_fig's `hz`,\n"
       "# make_mu_timeline_fig's `h_chain`). It does NOT enter `contacts_label`, and WeeklyWindow's\n"
       "# all_weeks comes from n_fit/smax alone, so the cache token and the Stage-1 contact window\n"
       "# are byte-for-byte what a full h1-h4 run would use: artefacts stay forward-compatible.\n"
       "cfg  = FrameworkConfig(constant_contacts = false, horizons = 1:1)",
       "horizons = 1:1")
sub(2, "# The single forecast date this notebook diagnoses (all four models cached for it, h1–h4).\n"
       "#ORIGIN = Date(2020, 11, 22)\n"
       "ORIGIN = Date(2021, 5, 9)",
       "# The single forecast date this notebook diagnoses (all four models cached for it at h=1).\n"
       "# Override with ORIGIN_13J=YYYY-MM-DD, as 12j does with ORIGIN_12J.\n"
       "ORIGIN = Date(get(ENV, \"ORIGIN_13J\", \"2021-05-09\"))",
       "ORIGIN from env")

# ── 4. one degree window, not four ───────────────────────────────────────────
sub(4, "# this origin's 4 contact/degree windows (one per horizon; reuse the single raw read).",
       "# this origin's contact/degree windows, one per horizon — a SINGLE window here (h=1).",
       "apd_o comment")

# ── 5. drop the h4 fit store ─────────────────────────────────────────────────
sub(5, "# Built from the h=1 AND h=4 artefacts — §1 overlays each as its own figure.",
       "# Built from the h=1 artefacts only — this notebook is h1-only, so there is no h4 companion.",
       "fit store comment")
sub(5, 'fit_store    = _assemble_fit_store(1)\n'
       'fit_store_h4 = _assemble_fit_store(4)\n'
       'println("assembled fit-window fits (h1): ", collect(keys(fit_store)))\n'
       'println("assembled fit-window fits (h4): ", collect(keys(fit_store_h4)))',
       'fit_store = _assemble_fit_store(1)\n'
       'println("assembled fit-window fits (h1): ", collect(keys(fit_store)))',
       "drop fit_store_h4")

# ── 6/7. §1 — single figure ──────────────────────────────────────────────────
sub(6, "\n\nThe in-sample fitted mean (dashed) is drawn twice — once reconstructed from each model's **h=1**\n"
       "artefacts and once from its **h=4** artefacts — as two otherwise-identical figures.",
       "\n\nOnly the **h=1** in-sample reconstruction is drawn here (10j draws a second figure from the h=4\n"
       "artefacts; this notebook has none).", "§1 md")
sub(7, "# §1 forecast point + 90% CI — four ways, single origin (make_forecast_ci_fig, 10j_viz_utils.jl).\n"
       "# Solid + ○ = self-iterated forecast; dashed + ◇ = in-sample fitted mean. Two figures differing\n"
       "# only in which chain the in-sample fit is reconstructed from: the h=1 vs the h=4 chain.\n"
       "display(make_forecast_ci_fig(fc_store, fit_store,    win, wd, truth, cfg, labels4, model_cols, ORIGIN; fit_h = 1))\n"
       "display(make_forecast_ci_fig(fc_store, fit_store_h4, win, wd, truth, cfg, labels4, model_cols, ORIGIN; fit_h = 4))",
       "# §1 forecast point + 90% CI — four ways, single origin (make_forecast_ci_fig, 10j_viz_utils.jl).\n"
       "# Solid + ○ = self-iterated forecast; dashed + ◇ = in-sample fitted mean, from the h=1 chain.\n"
       "display(make_forecast_ci_fig(fc_store, fit_store, win, wd, truth, cfg, labels4, model_cols, ORIGIN;\n"
       "                             fit_h = 1, res_dir = RES13))", "§1 fig")

# ── 8/9. §1b — single figure ─────────────────────────────────────────────────
sub(8, "Drawn twice — once with the in-sample fit reconstructed from each model's **h=1** artefacts and once\n"
       "from its **h=4** artefacts — mirroring §1's two figures. Saved to\n"
       "`res/10j_forecast_ci_<origin>_fit-h<fit_h>_byage.png`.",
       "Drawn once, with the in-sample fit reconstructed from each model's **h=1** artefacts. Saved to\n"
       "`res/13j/10j_forecast_ci_<origin>_fit-h1_byage.png`.", "§1b md")
sub(9, "# Reuses the already-loaded fc_store / fit_store(_h4) / truth / wd (NO re-fit); each panel is the\n"
       "# single-age slice of §1's figure (make_forecast_ci_by_age_fig, 10j_viz_utils.jl). Two figures\n"
       "# mirroring §1: in-sample fit from the h=1 vs the h=4 chain (the forecast/observed are identical).\n"
       "# NOTE: age panels are slices of the coupled 7-age forecast, not seven separate single-age models.\n"
       "display(make_forecast_ci_by_age_fig(fc_store, fit_store,    win, wd, truth, cfg, labels4, model_cols, ORIGIN, grid; fit_h = 1))\n"
       "display(make_forecast_ci_by_age_fig(fc_store, fit_store_h4, win, wd, truth, cfg, labels4, model_cols, ORIGIN, grid; fit_h = 4))",
       "# Reuses the already-loaded fc_store / fit_store / truth / wd (NO re-fit); each panel is the\n"
       "# single-age slice of §1's figure (make_forecast_ci_by_age_fig, 10j_viz_utils.jl), in-sample\n"
       "# fit from the h=1 chain.\n"
       "# NOTE: age panels are slices of the coupled 7-age forecast, not seven separate single-age models.\n"
       "display(make_forecast_ci_by_age_fig(fc_store, fit_store, win, wd, truth, cfg, labels4, model_cols,\n"
       "                                    ORIGIN, grid; fit_h = 1, res_dir = RES13))", "§1b fig")

# ── 12/13. §2 age-pair ───────────────────────────────────────────────────────
sub(12, "make_agepair_fig(NegBinAgePair(), MeanNGM(), oc)",
        "make_agepair_fig(NegBinAgePair(), MeanNGM(), oc; res_dir = RES13)", "§2 negbin")
sub(13, "make_agepair_fig(HurdleWeibullAgePair(), MeanNGM(), oc)",
        "make_agepair_fig(HurdleWeibullAgePair(), MeanNGM(), oc; res_dir = RES13)", "§2 hweibull")

# ── 14-17. §2b — one horizon ─────────────────────────────────────────────────
for i in (14, 16):
    sub(i, "h1–h4", "h1", f"§2b md {i}")
sub(15, "# 2b. Contact mean μ from participants 11-15 (bin 2) across horizons h1..h4, per contactee age",
        "# 2b. Contact mean μ from participants 11-15 (bin 2) at horizon h1, per contactee age", "§2b hdr")
sub(15, '#     week = the C*-slice its forecast is frozen at); observed from apd_o[h]. See make_mu_horizon_fig.',
        '#     week = the C*-slice its forecast is frozen at); observed from apd_o[h]. See make_mu_horizon_fig.\n'
        '#     h1-ONLY: `hz` defaults to collect(cfg.horizons) == [1], so each panel carries ONE point.',
        "§2b note")
sub(15, 'tdesc = "μ from 11-15 over h1–h4 by contactee"', 'tdesc = "μ from 11-15 at h1 by contactee"',
    "§2b tdesc")
sub(15, "display(make_mu_horizon_fig(NegBinAgePair(), partic_cells, \"vs_horizon\", tdesc,\n"
        "                            apd_o, ORIGIN, grid, cfg, labels4, model_cols))          # Fig A — count mean\n"
        "display(make_mu_horizon_fig(HurdleWeibullAgePair(), partic_cells, \"vs_horizon\", tdesc,\n"
        "                            apd_o, ORIGIN, grid, cfg, labels4, model_cols))          # Fig B — positive weighted mean",
        "display(make_mu_horizon_fig(NegBinAgePair(), partic_cells, \"vs_horizon\", tdesc,\n"
        "                            apd_o, ORIGIN, grid, cfg, labels4, model_cols;\n"
        "                            res_dir = RES13))                                        # Fig A — count mean\n"
        "display(make_mu_horizon_fig(HurdleWeibullAgePair(), partic_cells, \"vs_horizon\", tdesc,\n"
        "                            apd_o, ORIGIN, grid, cfg, labels4, model_cols;\n"
        "                            res_dir = RES13))                                        # Fig B — positive weighted mean",
        "§2b figs")
sub(17, "# 2b (cont.). Diagonal (self-contact) μ_{i→i} for each of the 7 age groups across horizons h1..h4",
        "# 2b (cont.). Diagonal (self-contact) μ_{i→i} for each of the 7 age groups at horizon h1", "§2b diag hdr")
sub(17, 'tdesc_diag = "diagonal (self-contact) μ over h1–h4"',
        'tdesc_diag = "diagonal (self-contact) μ at h1"', "§2b diag tdesc")
sub(17, "display(make_mu_horizon_fig(NegBinAgePair(), diag_cells, \"diag_vs_horizon\", tdesc_diag,\n"
        "                            apd_o, ORIGIN, grid, cfg, labels4, model_cols))          # Fig A — count mean\n"
        "display(make_mu_horizon_fig(HurdleWeibullAgePair(), diag_cells, \"diag_vs_horizon\", tdesc_diag,\n"
        "                            apd_o, ORIGIN, grid, cfg, labels4, model_cols))          # Fig B — positive weighted mean",
        "display(make_mu_horizon_fig(NegBinAgePair(), diag_cells, \"diag_vs_horizon\", tdesc_diag,\n"
        "                            apd_o, ORIGIN, grid, cfg, labels4, model_cols;\n"
        "                            res_dir = RES13))                                        # Fig A — count mean\n"
        "display(make_mu_horizon_fig(HurdleWeibullAgePair(), diag_cells, \"diag_vs_horizon\", tdesc_diag,\n"
        "                            apd_o, ORIGIN, grid, cfg, labels4, model_cols;\n"
        "                            res_dir = RES13))                                        # Fig B — positive weighted mean",
        "§2b diag figs")

# ── 18-21. §2c — timeline from the h1 chain ──────────────────────────────────
sub(18, "## 2c. Contact mean μ over the fit window + horizons h1–h4, from the **h4 chain** (11-15 → contactee)",
        "## 2c. Contact mean μ over the fit window + h1, from the **h1 chain** (11-15 → contactee)", "§2c md")
sub(20, "### 2c (cont.) — diagonal (self-contact) μ_{i→i}(t) from the h4 chain",
        "### 2c (cont.) — diagonal (self-contact) μ_{i→i}(t) from the h1 chain", "§2c diag md")
sub(19, "# 2c. μ_{11-15→j}(t) over the fit window + horizons h1..h4, reconstructed from the SINGLE h4 chain\n"
        "#     (its per-week GP spans the origin's 8 fit weeks ++ the 4 horizon weeks). Contrast with §2b,\n"
        "#     which reads each horizon from its own chain. Two figures by degree family. make_mu_timeline_fig\n"
        "#     reuses the §2b participant-cell list (partic_cells); apd_o[end] is the h4 degree window.",
        "# 2c. μ_{11-15→j}(t) over the fit window + h1, reconstructed from the SINGLE h1 chain (its per-week\n"
        "#     GP spans the origin's 8 fit weeks ++ the 1 horizon week = 12 weeks with the 4 renewal lags).\n"
        "#     `h_chain` defaults to maximum(cfg.horizons) == 1, and apd_o[end] == apd_o[1] is the h1 degree\n"
        "#     window. Shorter forecast reach than 10j's h4 version, but the same per-week μ trace from ONE\n"
        "#     fit — contrast with §2b, which reads each horizon from its own chain.",
        "§2c hdr")
sub(19, 'tdesc_tl = "μ from 11-15 over fit weeks + h1–h4"', 'tdesc_tl = "μ from 11-15 over fit weeks + h1"',
    "§2c tdesc")
sub(19, "display(make_mu_timeline_fig(NegBinAgePair(), partic_cells, \"h4timeline\", tdesc_tl,\n"
        "                             apd_o[end], ORIGIN, grid, cfg, labels4, model_cols))        # Fig A — count mean\n"
        "display(make_mu_timeline_fig(HurdleWeibullAgePair(), partic_cells, \"h4timeline\", tdesc_tl,\n"
        "                             apd_o[end], ORIGIN, grid, cfg, labels4, model_cols))        # Fig B — positive weighted mean",
        "display(make_mu_timeline_fig(NegBinAgePair(), partic_cells, \"h1timeline\", tdesc_tl,\n"
        "                             apd_o[end], ORIGIN, grid, cfg, labels4, model_cols;\n"
        "                             res_dir = RES13))                                          # Fig A — count mean\n"
        "display(make_mu_timeline_fig(HurdleWeibullAgePair(), partic_cells, \"h1timeline\", tdesc_tl,\n"
        "                             apd_o[end], ORIGIN, grid, cfg, labels4, model_cols;\n"
        "                             res_dir = RES13))                                          # Fig B — positive weighted mean",
        "§2c figs")
sub(21, "# 2c (cont.). Diagonal μ_{i→i}(t) from the single h4 chain over the fit window + horizons h1..h4.",
        "# 2c (cont.). Diagonal μ_{i→i}(t) from the single h1 chain over the fit window + h1.", "§2c diag hdr")
sub(21, 'tdesc_diag_tl = "diagonal (self-contact) μ over fit weeks + h1–h4"',
        'tdesc_diag_tl = "diagonal (self-contact) μ over fit weeks + h1"', "§2c diag tdesc")
sub(21, "display(make_mu_timeline_fig(NegBinAgePair(), diag_cells, \"diag_h4timeline\", tdesc_diag_tl,\n"
        "                             apd_o[end], ORIGIN, grid, cfg, labels4, model_cols))        # Fig A — count mean\n"
        "display(make_mu_timeline_fig(HurdleWeibullAgePair(), diag_cells, \"diag_h4timeline\", tdesc_diag_tl,\n"
        "                             apd_o[end], ORIGIN, grid, cfg, labels4, model_cols))        # Fig B — positive weighted mean",
        "display(make_mu_timeline_fig(NegBinAgePair(), diag_cells, \"diag_h1timeline\", tdesc_diag_tl,\n"
        "                             apd_o[end], ORIGIN, grid, cfg, labels4, model_cols;\n"
        "                             res_dir = RES13))                                          # Fig A — count mean\n"
        "display(make_mu_timeline_fig(HurdleWeibullAgePair(), diag_cells, \"diag_h1timeline\", tdesc_diag_tl,\n"
        "                             apd_o[end], ORIGIN, grid, cfg, labels4, model_cols;\n"
        "                             res_dir = RES13))                                          # Fig B — positive weighted mean",
        "§2c diag figs")

# ── 23/25/26/28. §3-§5 — res_dir only ────────────────────────────────────────
sub(23, "f = make_contactmatrix_fig(dm, nb, oc)",
        "f = make_contactmatrix_fig(dm, nb, oc; res_dir = RES13)", "§3")
sub(25, "make_agepair_ccdf_fig(NegBinAgePair(), MeanNGM(), oc)",
        "make_agepair_ccdf_fig(NegBinAgePair(), MeanNGM(), oc; res_dir = RES13)", "§4 negbin")
sub(26, "make_agepair_ccdf_fig(HurdleWeibullAgePair(), MeanNGM(), oc)",
        "make_agepair_ccdf_fig(HurdleWeibullAgePair(), MeanNGM(), oc; res_dir = RES13)", "§4 hweibull")
sub(28, "display(make_susc_inf_fig(combos, labels4, model_cols, ORIGIN, cfg, grid; h = 1))",
        "display(make_susc_inf_fig(combos, labels4, model_cols, ORIGIN, cfg, grid;\n"
        "                          h = 1, res_dir = RES13))", "§5")

# §6's three helpers (plot_dispersion_cells / plot_tau_over_weeks / plot_p0_vs_empirical) never
# savefig, so they cannot collide and are deliberately left untouched.

# ── 31. notes ────────────────────────────────────────────────────────────────
sub(31, "- **Single origin** (2020-11-15); no re-fit — cached 8j two-stage artefacts reloaded via",
        "- **h = 1 ONLY** (`cfg.horizons = 1:1`) — see the header for exactly what that costs relative\n"
        "  to 10j. Figures are written to `res/13j/` (filenames keep the hard-coded `10j_` prefix).\n"
        "- **Single origin**, `ENV[\"ORIGIN_13J\"]`; no re-fit — cached two-stage artefacts reloaded via",
        "notes")

# ── global markdown pass: figure paths in PROSE ──────────────────────────────
# The section headers document each figure's saved filename. Those references are stale twice over
# after the edits above: the tag changed (h4timeline -> h1timeline) and the directory changed
# (res/ -> res/13j/). Counts are asserted so this cannot silently drift if 10j gains a figure.
def md_pass(old, new, expect, why):
    hits = 0
    for c in cells:
        if c["cell_type"] != "markdown":
            continue
        s = "".join(c["source"])
        if old in s:
            hits += s.count(old)
            c["source"] = s.replace(old, new).splitlines(keepends=True)
    if hits != expect:
        sys.exit(f"FATAL md_pass {why}: {hits} hits, expected {expect} for {old!r}")
    _applied.append(("md", f"{why} x{hits}"))


# Two prose lines describe the FIGURE's own horizon span (not 10j's, which cell 0 discusses by
# contrast and must keep saying h1-h4). Fixed individually so cell 0's deliberate mentions survive.
sub(18, "origin t₀ (fit weeks to its left, horizons h1–h4 to its right).",
        "origin t₀ (fit weeks to its left, horizon h1 to its right).", "§2c span")
sub(20, "over the origin's fit weeks ++ horizons h1–h4.",
        "over the origin's fit weeks ++ horizon h1.", "§2c diag span")

md_pass("h4timeline", "h1timeline", 2, "timeline tag in prose")
md_pass("res/10j_", "res/13j/10j_", 7, "figure dir in prose")   # res/13j/... already written is safe

# ── plain-notebook hygiene ───────────────────────────────────────────────────
for c in cells:
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []

json.dump(nb, open(DST, "w"), indent=1, ensure_ascii=False)
open(DST, "a").write("\n")
print(f"wrote {os.path.normpath(DST)} with {len(cells)} cells, {len(_applied)} edits applied")
