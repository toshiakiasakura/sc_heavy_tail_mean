#!/usr/bin/env bash
# run_diagnostics.sh <pf|nuts> — gate, then 9j → 10j → 11j (→ 12j for nuts), headless.
#
# The gate is the point of the script. `fit_or_load_stage2` fits on miss and 9j's per-cell
# try/catch only logs, so running the diagnostics against an incomplete grid does not fail — it
# silently refits, serially. `tmp/check_grid.jl` counts 504/1512 for the active token first and
# stops the run if anything is missing or the provenance keys are mixed.
#
# STOP-ON-FIRST-FAILURE is deliberate: 10j and 11j read what 9j has already cached, and a broken
# 9j would otherwise be diagnosed three notebooks later.
set -u
MODE="${1:?usage: run_diagnostics.sh <pf|nuts>}"
case "$MODE" in
  pf)   export STAGE1_USE_NUTS=false; NBS="9j_forecast_diagnostics 10j_model_diagnostics 11j_weekly_identifiability" ;;
  nuts) export STAGE1_USE_NUTS=true;  NBS="9j_forecast_diagnostics 10j_model_diagnostics 11j_weekly_identifiability 12j_chain_convergence" ;;
  *)    echo "mode must be pf or nuts"; exit 2 ;;
esac

SCR="${SCRATCH:-/tmp/claude-1000/-workdir/807a82ea-386b-4606-b76b-9ba4dcaf2878/scratchpad}/nb_${MODE}_diag"
mkdir -p "$SCR"
cd /workdir/src

echo "=== gate: tmp/check_grid.jl [$MODE] $(date '+%F %T') ==="
julia --project=/workdir /workdir/tmp/check_grid.jl || { echo "*** GATE FAILED — not running diagnostics ***"; exit 1; }

for nb in $NBS; do
  echo "=================== $nb  $(date '+%F %T') ==================="
  t0=$(date +%s)
  # 12j reads dt_intermediate_nuts_pilot/ FIRST and only falls back to the production artefact, so
  # while the four -ar1 pilots are there it would silently diagnose a 500-draw pilot chain instead
  # of the 2000-draw production one and report it as the grid's convergence. Refuse rather than
  # move data implicitly — archiving a generation is a decision, not a side effect.
  if [ "$nb" = "12j_chain_convergence" ]; then
    export ORIGIN_12J="${ORIGIN_12J:-2021-05-09}"
    for d in unweighted-negbin weighted-hweibull; do
      p="/workdir/dt_intermediate_nuts_pilot/pilot_${d}_${ORIGIN_12J}_h1.jld2"
      [ -f "$p" ] && { echo "*** $p exists — 12j would read the PILOT chain, not the production one."; \
                       echo "    Archive it first:  mv /workdir/dt_intermediate_nuts_pilot /workdir/dt_intermediate_nuts_pilot_ar1pilot"; \
                       exit 1; }
    done
  fi
  jupyter nbconvert --to notebook --execute "$nb.ipynb" \
      --ExecutePreprocessor.timeout=-1 --output-dir "$SCR" \
    || { echo "*** $nb FAILED after $(( $(date +%s) - t0 ))s ***"; exit 1; }
  echo "--- $nb OK in $(( $(date +%s) - t0 ))s ---"
done
echo "DIAGNOSTICS COMPLETE [$MODE] $(date '+%F %T')"
