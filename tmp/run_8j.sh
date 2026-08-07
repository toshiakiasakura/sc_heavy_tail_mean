#!/usr/bin/env bash
# run_8j.sh <pf|nuts> — execute the 8j fitting notebook headless for one generation.
#
#   pf    STAGE1_USE_NUTS=false  ⇒ token …-t0-ar1        (Pathfinder both stages, the rehearsal)
#   nuts  STAGE1_USE_NUTS=true   ⇒ token …-t0-ar1-nuts   (Stage 1 NUTS 1000+2000, Stage 2 Pathfinder)
#
# The two tokens differ, so the generations coexist in dt_intermediate/ and neither overwrites the
# other. Both prefit drivers skip existing artefacts, so re-running this after an interruption
# resumes; that is also how S1_CONCURRENCY can be changed mid-grid (kill, re-launch, keep the
# finished fits).
#
# `--ExecutePreprocessor.timeout=-1` is mandatory: nbconvert's 30 s per-cell default would kill
# cell 1's `include("forecast_utils.jl")`, never mind a multi-day fit cell. `--output-dir` (never
# `--inplace`) keeps the tracked notebook plain.
#
# Extra env passed straight through: S1_CONCURRENCY, S2_CONCURRENCY, ORIGIN_MIN, FIT_END.
set -u
MODE="${1:?usage: run_8j.sh <pf|nuts>}"
case "$MODE" in
  pf)   export STAGE1_USE_NUTS=false ;;
  nuts) export STAGE1_USE_NUTS=true  ;;
  *)    echo "mode must be pf or nuts"; exit 2 ;;
esac

SCR="${SCRATCH:-/tmp/claude-1000/-workdir/807a82ea-386b-4606-b76b-9ba4dcaf2878/scratchpad}/nb_${MODE}"
mkdir -p "$SCR"
cd /workdir/src

echo "=== 8j [$MODE] start $(date '+%F %T') | STAGE1_USE_NUTS=$STAGE1_USE_NUTS ==="
echo "    S1_CONCURRENCY=${S1_CONCURRENCY:-auto} S2_CONCURRENCY=${S2_CONCURRENCY:-auto}" \
     "ORIGIN_MIN=${ORIGIN_MIN:-unset} FIT_END=${FIT_END:-default}"
t0=$(date +%s)
jupyter nbconvert --to notebook --execute 8j_preliminary_forecast.ipynb \
    --ExecutePreprocessor.timeout=-1 --output-dir "$SCR"
rc=$?
el=$(( $(date +%s) - t0 ))
printf '=== 8j [%s] rc=%d after %dh%02dm %s ===\n' \
       "$MODE" "$rc" $(( el / 3600 )) $(( (el % 3600) / 60 )) "$(date '+%F %T')"
exit $rc
