#!/usr/bin/env bash
# watch_grid.sh — progress logger for a long 8j pre-fit run.
#
# `prefit_stage1!`/`prefit_stage2!` emit one line per ORIGIN, and Turing's own progress bar is off
# (`sample(...; progress = false)`), so the artefact count on disk is the only fine-grained signal
# of how a multi-day grid is doing. This turns it into a rate and an ETA, and records RSS — the
# number that decides whether S1_CONCURRENCY can be raised on the next (resumable) restart.
#
#   tmp/watch_grid.sh <token> [expected_s1] [expected_s2] [interval_s]
#     token         the contacts token, e.g. temporal-gsar-cut-sc-p0-gi-s0-m32-t0-ar1-nuts
#     expected_s1   default 504   (63 origins × 2 degree models × 4 horizons)
#     expected_s2   default 1512  (63 origins × 6 combos × 4 horizons)
#     interval_s    default 600
#
# Appends to tmp/grid_progress_<token>.log and echoes the same line. Run under nohup; stop with
# `kill`. Safe to start before the fits do — a zero count just reports 0.
set -u

TOKEN="${1:?usage: watch_grid.sh <token> [n_s1] [n_s2] [interval_s]}"
N1="${2:-504}"
N2="${3:-1512}"
IVL="${4:-600}"
DIR=/workdir/dt_intermediate
LOG="/workdir/tmp/grid_progress_${TOKEN}.log"

count() { ls -1 "$DIR" 2>/dev/null | grep -c -- "^8j_${1}_.*_${TOKEN}_[0-9-]*_h[0-9]*\.jld2$"; }

t_start=$(date +%s)
c1_start=$(count s1); c2_start=$(count s2)
based=0     # has the rate baseline been re-anchored to the first observed progress?
printf '=== watch_grid %s | start s1=%d/%d s2=%d/%d | every %ds ===\n' \
       "$TOKEN" "$c1_start" "$N1" "$c2_start" "$N2" "$IVL" | tee -a "$LOG"

while true; do
    now=$(date +%s); el=$(( now - t_start ))
    c1=$(count s1); c2=$(count s2)
    # Re-anchor the baseline the FIRST time progress is seen. The watcher is usually started before
    # the fits are (package load alone is minutes), and averaging the rate over that dead time makes
    # the ETA meaningless — measured 29.9 h against an actual ~4 h because 28 idle minutes were in
    # the denominator. From here the rate is over fitting time only.
    if [ "$based" -eq 0 ] && { [ "$c1" -gt "$c1_start" ] || [ "$c2" -gt "$c2_start" ]; }; then
        based=1; t_start=$(( now - 1 )); c1_start=$c1; c2_start=$c2; el=1
        echo "    (rate baseline anchored at first observed progress)" | tee -a "$LOG"
    fi
    # Rate over THIS run only (c*_start), so a resumed run is not flattered by what it inherited.
    d1=$(( c1 - c1_start )); d2=$(( c2 - c2_start ))
    stage="s1"; done_n=$d1; todo_n=$(( N1 - c1 ))
    if [ "$c1" -ge "$N1" ]; then stage="s2"; done_n=$d2; todo_n=$(( N2 - c2 )); fi

    if [ "$el" -gt 0 ] && [ "$done_n" -gt 0 ]; then
        eta=$(awk -v t="$el" -v d="$done_n" -v r="$todo_n" 'BEGIN{printf "%.1f", r*t/d/3600}')
        rate=$(awk -v t="$el" -v d="$done_n" 'BEGIN{printf "%.1f", d*3600/t}')
    else
        eta="--"; rate="0.0"
    fi

    # MemAvailable is the honest figure (free + reclaimable), not MemFree.
    read -r _ memtot _  < <(grep MemTotal     /proc/meminfo)
    read -r _ memav _   < <(grep MemAvailable /proc/meminfo)
    used=$(awk -v t="$memtot" -v a="$memav" 'BEGIN{printf "%.1f", (t-a)/1048576}')
    tot=$(awk  -v t="$memtot"              'BEGIN{printf "%.1f", t/1048576}')
    jrss=$(ps -o rss= -C julia 2>/dev/null | awk '{s+=$1} END{printf "%.1f", s/1048576}')
    load=$(awk '{print $1}' /proc/loadavg)

    printf '[%s] s1 %d/%d (%s%%) | s2 %d/%d (%s%%) | %s %s/h ETA %sh | mem %s/%sG julia %sG | load %s\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" \
        "$c1" "$N1" "$(awk -v a=$c1 -v b=$N1 'BEGIN{printf "%.1f", 100*a/b}')" \
        "$c2" "$N2" "$(awk -v a=$c2 -v b=$N2 'BEGIN{printf "%.1f", 100*a/b}')" \
        "$stage" "$rate" "$eta" "$used" "$tot" "${jrss:-0.0}" "$load" | tee -a "$LOG"

    # Stop once BOTH stages are complete — otherwise this would log forever after the run ends.
    [ "$c1" -ge "$N1" ] && [ "$c2" -ge "$N2" ] && { echo "GRID COMPLETE" | tee -a "$LOG"; break; }
    sleep "$IVL"
done
