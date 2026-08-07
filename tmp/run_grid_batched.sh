#!/usr/bin/env bash
# run_grid_batched.sh <pf|nuts> [origins_per_batch] — run the 63-origin 8j grid in batches, each
# batch a FRESH Julia process.
#
# WHY BATCHES. Measured 2026-08-07 on the Pathfinder grid: a single process fitting origins
# back-to-back grew monotonically 10.3 -> 25.0 GiB over ~12 origins against 27.4 GiB of RAM, slowed
# from 224 s to 1134 s per origin as the GC thrashed, and was OOM-killed at origin 16 — silently,
# with no error in its own log, because SIGKILL leaves nothing behind. `prefit_stage1!` now calls
# `GC.gc()` per origin, which should hold RSS flat; this script is the belt to that braces. A
# process that exits every N origins CANNOT accumulate, whatever the underlying cause, and the run
# is fully resumable (both prefit drivers skip existing artefacts) so a batch boundary costs only
# the ~30 s package load.
#
# Each batch runs 8j over its own origin slice — Stage 1 then Stage 2 for those origins — so the
# total work is identical to one long run, just partitioned.
#
#   tmp/run_grid_batched.sh pf 8
#
# Progress: tmp/run_8j_<mode>_batchNN.log per batch, plus the shared
# tmp/grid_progress_<token>.log from watch_grid.sh (started separately, or by tmp/launch_8j.sh).
set -u
MODE="${1:?usage: run_grid_batched.sh <pf|nuts> [origins_per_batch]}"
N_PER="${2:-8}"

case "$MODE" in
  pf)   export STAGE1_USE_NUTS=false ;;
  nuts) export STAGE1_USE_NUTS=true  ;;
  *)    echo "mode must be pf or nuts"; exit 2 ;;
esac

# The 63 production origins: weekly Sunday-start, 2020-10-18 … 2021-12-26. Derived here rather than
# hard-coded as a list so it stays in step with `available_forecast_origins`; 8j re-derives it too
# and intersects with ORIGIN_MIN/FIT_END, so a drift would shrink a batch, never corrupt one.
FIRST=2020-10-18
N_ORIGINS=63

date_at() { date -u -d "$FIRST + $(( $1 * 7 )) days" +%Y-%m-%d; }

n_batches=$(( (N_ORIGINS + N_PER - 1) / N_PER ))
echo "=== batched grid [$MODE] : $N_ORIGINS origins in $n_batches batches of $N_PER, $(date '+%F %T') ==="

for ((b = 0; b < n_batches; b++)); do
    lo=$(( b * N_PER ))
    hi=$(( lo + N_PER - 1 )); [ "$hi" -ge "$N_ORIGINS" ] && hi=$(( N_ORIGINS - 1 ))
    export ORIGIN_MIN="$(date_at $lo)"
    export FIT_END="$(date_at $hi)"
    log="/workdir/tmp/run_8j_${MODE}_batch$(printf '%02d' $((b+1))).log"
    echo "--- batch $((b+1))/$n_batches : origins $ORIGIN_MIN … $FIT_END  $(date '+%T')  -> $(basename "$log")"
    t0=$(date +%s)
    ( cd /workdir/src && julia --project=/workdir /workdir/tmp/run_nb_headless.jl \
        8j_preliminary_forecast.ipynb ) > "$log" 2>&1
    rc=$?
    el=$(( $(date +%s) - t0 ))
    s1=$(ls -1 /workdir/dt_intermediate/ | grep -c '^8j_s1_')
    s2=$(ls -1 /workdir/dt_intermediate/ | grep -c '^8j_s2_')
    printf '    batch %d rc=%d in %dh%02dm | grid now s1=%d/504 s2=%d/1512 | peak RSS %s\n' \
           $((b+1)) "$rc" $(( el / 3600 )) $(( (el % 3600) / 60 )) "$s1" "$s2" \
           "$(grep -o 'RSS [0-9.]* GiB' "$log" | tail -1)"
    # rc != 0 is worth stopping on: a batch that dies has either hit the same memory wall (so the
    # next one will too) or found a real fit error, and both want looking at, not powering through.
    if [ "$rc" -ne 0 ]; then
        echo "*** batch $((b+1)) exited rc=$rc — stopping. Tail of $log:"; tail -25 "$log"; exit "$rc"
    fi
done
# FINAL SWEEP over ALL origins, unrestricted. Batches partition the origin list, so anything a
# batch missed — an early exit, a per-fit failure, a boundary arithmetic slip — would otherwise never
# be revisited and would surface only as a gap in check_grid.jl after the whole run. Both prefit
# drivers skip existing artefacts, so when the grid is already complete this costs one package load
# and nothing else. It is the difference between "the batches finished" and "the grid is complete".
unset ORIGIN_MIN FIT_END
log="/workdir/tmp/run_8j_${MODE}_sweep.log"
echo "--- final sweep : all $N_ORIGINS origins  $(date '+%T')  -> $(basename "$log")"
t0=$(date +%s)
( cd /workdir/src && julia --project=/workdir /workdir/tmp/run_nb_headless.jl \
    8j_preliminary_forecast.ipynb ) > "$log" 2>&1
rc=$?
el=$(( $(date +%s) - t0 ))
s1=$(ls -1 /workdir/dt_intermediate/ | grep -c '^8j_s1_')
s2=$(ls -1 /workdir/dt_intermediate/ | grep -c '^8j_s2_')
printf '    sweep rc=%d in %dh%02dm | grid now s1=%d/504 s2=%d/1512\n' \
       "$rc" $(( el / 3600 )) $(( (el % 3600) / 60 )) "$s1" "$s2"
[ "$rc" -ne 0 ] && { echo "*** sweep exited rc=$rc — tail of $log:"; tail -25 "$log"; exit "$rc"; }

echo "=== ALL BATCHES COMPLETE [$MODE] $(date '+%F %T') ==="
if [ "$s1" -ge 504 ] && [ "$s2" -ge 1512 ]; then
    echo "=== GRID COMPLETE: s1=$s1/504 s2=$s2/1512 ==="
else
    echo "*** GRID INCOMPLETE after sweep: s1=$s1/504 s2=$s2/1512 — run tmp/check_grid.jl ***"; exit 1
fi
