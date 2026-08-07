#!/usr/bin/env bash
# launch_8j.sh <pf|nuts> — start the 8j fitting run headless, plus its progress watcher.
#
# A script rather than an inline compound command on purpose: the inline form kept losing newlines
# through the shell wrapper, silently turning `sleep 1` + `cd …` into `sleep 1 cd …`, which fails
# and short-circuits the `&&` so the fit never starts while the watcher happily reports 0/504.
set -u
MODE="${1:?usage: launch_8j.sh <pf|nuts>}"
case "$MODE" in
  pf)   export STAGE1_USE_NUTS=false; TOKEN="temporal-gsar-cut-sc-p0-gi-s0-m32-t0-ar1" ;;
  nuts) export STAGE1_USE_NUTS=true;  TOKEN="temporal-gsar-cut-sc-p0-gi-s0-m32-t0-ar1-nuts" ;;
  *)    echo "mode must be pf or nuts"; exit 2 ;;
esac

# Refuse to start a second copy: both prefit drivers skip existing artefacts, so a duplicate would
# not corrupt anything, but two processes would halve the throughput and double the memory.
if pgrep -f "[r]un_nb_headless.jl" > /dev/null; then
  echo "*** a run_nb_headless.jl is already running — not starting another"; pgrep -af "[r]un_nb_headless.jl"; exit 1
fi
pkill -f "[w]atch_grid.sh" 2>/dev/null || true

cd /workdir/src
nohup julia --project=/workdir /workdir/tmp/run_nb_headless.jl 8j_preliminary_forecast.ipynb \
      > "/workdir/tmp/run_8j_${MODE}.log" 2>&1 &
fitpid=$!
nohup /workdir/tmp/watch_grid.sh "$TOKEN" 504 1512 600 > "/workdir/tmp/watch_${MODE}.log" 2>&1 &
watchpid=$!

echo "8j [$MODE] pid $fitpid  -> /workdir/tmp/run_8j_${MODE}.log"
echo "watcher   pid $watchpid -> /workdir/tmp/grid_progress_${TOKEN}.log"
echo "token     $TOKEN"
sleep 3
ps -o pid=,etime=,args= -p "$fitpid" || { echo "*** the fit process died immediately"; exit 1; }
