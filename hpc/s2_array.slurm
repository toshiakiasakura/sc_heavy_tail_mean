#!/bin/bash
#SBATCH --array=0-15
#SBATCH --job-name=sc_s2
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=lsh2200567@lshtm.ac.uk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=6:00:00
#SBATCH --output=logs/s2_%A_%a.log
#
# Stage 2 of the 8j grid: 100 Pathfinder fits of the 18-dimension transmission model per cell,
# 6 combos × 4 horizons = 24 cells per origin, conditioning on the Stage-1 draws.
#
# ⚠ RUN ONLY AFTER STAGE 1 IS COMPLETE. The driver holds back any origin whose Stage-1 chains are
#   missing and names it in the log — because `stage2_inputs` would otherwise fall through to
#   `fit_or_load_stage1`, which fits SERIALLY and does not error (CLAUDE.md's silent-refit trap).
#   Under an array that also means the holding-back task would fit chains another task owns.
# ⚠ ORIGIN_MIN/FIT_END/STAGE1_USE_NUTS must be IDENTICAL to the Stage-1 run — they select the
#   generation. config.sh is shared precisely so they cannot drift.
# ⚠ Same array rules as Stage 1: start at 0, contiguous, submit from $PROJECT_DIR with logs/ present.
#
# SIZING (measured, CLAUDE.md): ~0.2 min per pooled cell with the ReverseDiff Stage-2 backend at
# max_concurrent=9 ⇒ the whole 1512-cell grid is ~5 h in one process; 16 tasks finish inside an
# hour. 1 GiB per concurrent fit is generous, so 32 GB covers 16-way concurrency.

set -euo pipefail

# ⚠ NOT `dirname $0` — Slurm runs a spool-dir copy of this script; see s1_array.slurm.
CFG=""
for d in "${SLURM_SUBMIT_DIR:-.}/hpc" "${SLURM_SUBMIT_DIR:-.}" "$(dirname "$(readlink -f "$0")")"; do
    [ -f "$d/config.sh" ] && { CFG="$d/config.sh"; break; }
done
[ -n "$CFG" ] || { echo "FATAL: hpc/config.sh not found (submit from \$PROJECT_DIR)" >&2; exit 1; }
echo "config     : $CFG"
. "$CFG"

pwd; hostname; date
echo "time started"

# See s1_array.slurm: ORIGIN_STRIDE pins the partition for a single-slice re-run; otherwise it is
# derived from the array, which must then be 0-based and contiguous.
TASK_MIN="${SLURM_ARRAY_TASK_MIN:-0}"
if [ -n "${ORIGIN_STRIDE:-}" ]; then
    TASK_COUNT="${ORIGIN_STRIDE}"
    echo "slice      : stride PINNED by ORIGIN_STRIDE=${ORIGIN_STRIDE} (single-slice re-run)"
else
    TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-$(( ${SLURM_ARRAY_TASK_MAX:-0} - TASK_MIN + 1 ))}"
    if [ "$TASK_MIN" -ne 0 ]; then
        echo "FATAL: --array must start at 0 (got min=$TASK_MIN). To re-run ONE slice, pin the" >&2
        echo "       stride: ORIGIN_STRIDE=<original> sbatch --array=<id>-<id> $0" >&2
        exit 1
    fi
fi
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$TASK_COUNT" ]; then
    echo "FATAL: task id ${SLURM_ARRAY_TASK_ID:-0} >= stride ${TASK_COUNT} — no such slice." >&2
    exit 1
fi

# Defaults so `set -u` does not abort when run by hand; see s1_array.slurm.
CPUS="${SLURM_CPUS_PER_TASK:-1}"
DEPOT_NAME="julia_depot_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-0}"
DEPOT_HOST="${DEPOT_ROOT}/${DEPOT_NAME}"
DEPOT_CONT="/workdir/${DEPOT_NAME}"
trap 'echo "removing ${DEPOT_HOST}"; rm -rf "${DEPOT_HOST}"' EXIT
mkdir -p "${DEPOT_HOST}" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/dt_intermediate"

echo "image      : ${IMAGE}"
echo "project    : ${PROJECT_DIR}"
echo "depot      : ${DEPOT_HOST}"
echo "slice      : ${SLURM_ARRAY_TASK_ID:-0}/${TASK_COUNT}"
echo "sampler    : STAGE1_USE_NUTS=${STAGE1_USE_NUTS} (token selector), Stage 2 always Pathfinder"
[ -f "${IMAGE}" ] || { echo "FATAL: image not found: ${IMAGE}" >&2; exit 1; }

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
fi
module load singularity

singularity exec --bind "${PROJECT_DIR}:/workdir" \
    --env JULIA_DEPOT_PATH="${DEPOT_CONT}:/opt/julia" \
    --env JULIA_NUM_THREADS="${CPUS}" \
    --env PHASE="s2" \
    --env STAGE1_USE_NUTS="${STAGE1_USE_NUTS}" \
    --env AD_BACKEND="${AD_BACKEND}" \
    --env ORIGIN_STRIDE="${TASK_COUNT}" \
    --env ORIGIN_OFFSET="${SLURM_ARRAY_TASK_ID:-0}" \
    --env FIT_END="${FIT_END}" \
    ${ORIGIN_MIN:+--env ORIGIN_MIN="${ORIGIN_MIN}"} \
    --env S2_CONCURRENCY="${CPUS}" \
    --env MAX_ORIGINS="${MAX_ORIGINS}" \
    --env CHUNK="${CHUNK}" \
    --env MEM_FLOOR_GIB="${MEM_FLOOR_GIB}" \
    "${IMAGE}" \
    /bin/sh /workdir/hpc/run_grid.sh

echo "time ended"
date
