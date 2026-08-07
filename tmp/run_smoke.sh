#!/usr/bin/env bash
# Phase-1 smoke: 3 origins, Pathfinder both stages, the full 8j -> 9j -> 10j -> 11j chain.
set -u
cd /workdir/src
export STAGE1_USE_NUTS=false ORIGIN_MIN=2021-04-25 FIT_END=2021-05-09
SCR="$1"
for nb in 8j_preliminary_forecast 9j_forecast_diagnostics 10j_model_diagnostics 11j_weekly_identifiability; do
  echo "=================== $nb  $(date '+%H:%M:%S') ==================="
  if [ "$nb" = "9j_forecast_diagnostics" ]; then
    echo "--- gate: tmp/check_grid.jl ---"
    julia --project=/workdir /workdir/tmp/check_grid.jl || { echo "GATE FAILED — stopping"; exit 1; }
  fi
  t0=$(date +%s)
  jupyter nbconvert --to notebook --execute "$nb.ipynb" \
      --ExecutePreprocessor.timeout=-1 --output-dir "$SCR/nb_smoke" \
    || { echo "*** $nb FAILED after $(( $(date +%s) - t0 ))s ***"; exit 1; }
  echo "--- $nb OK in $(( $(date +%s) - t0 ))s ---"
done
echo "SMOKE COMPLETE"
