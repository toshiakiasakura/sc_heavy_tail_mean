#!/bin/bash
#SBATCH --array=0-15
#SBATCH --job-name=sc_s1
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=lsh2200567@lshtm.ac.uk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/s1_%A_%a.log
#
# Stage 1 of the 8j grid: the contact-degree GP, NUTS, 2 degree models × 4 horizons per origin.
#
# ⚠ #SBATCH lines are parsed BEFORE the script runs, so they cannot read hpc/config.sh. Change the
#   array size / resources / mail address HERE (or override on the command line:
#   `sbatch --array=0-7 --time=48:00:00 hpc/s1_array.slurm`).
# ⚠ THE ARRAY MUST START AT 0 and be contiguous — the origin slice is 0-based and every task's
#   offset must be < the task count, or origins are silently left unfitted. Checked below.
# ⚠ Submit FROM $PROJECT_DIR with logs/ already created: --output cannot create its directory, and
#   a missing one fails the task instantly with nothing written anywhere.
#
# SIZING (measured, CLAUDE.md): a Stage-1 NUTS fit takes 1692/2509/2806/3027 s at h=1..4, and an
# origin is 8 such fits. --cpus-per-task=8 lets one origin's 8 fits run concurrently ⇒ ~1 h per
# origin, bounded by the slowest. 63 origins over 16 tasks = 4 origins each ⇒ ~5 h, comfortably
# inside 24 h. Memory: ~4 GiB per concurrent NUTS fit × 8 + the shared data ⇒ 64 GB.

set -euo pipefail

# ⚠ NOT `dirname $0`: Slurm copies the batch script to a node-local spool directory
# (/var/spool/slurmd/job<id>/slurm_script), so `$0` does not point into the repo. $SLURM_SUBMIT_DIR
# is where sbatch was invoked — the README says submit from $PROJECT_DIR. Candidates in order, with
# the script's own directory last so the file still works when run by hand on the login node.
CFG=""
for d in "${SLURM_SUBMIT_DIR:-.}/hpc" "${SLURM_SUBMIT_DIR:-.}" "$(dirname "$(readlink -f "$0")")"; do
    [ -f "$d/config.sh" ] && { CFG="$d/config.sh"; break; }
done
[ -n "$CFG" ] || { echo "FATAL: hpc/config.sh not found (submit from \$PROJECT_DIR)" >&2; exit 1; }
echo "config     : $CFG"
. "$CFG"

pwd; hostname; date
echo "time started"

# The partition: OFFSET = this task's id, STRIDE = how many tasks the origins were split across.
# Normally STRIDE is derived from the array itself, which is why the array must be 0-based and
# contiguous. Setting ORIGIN_STRIDE explicitly PINS it — the way to re-run one failed slice without
# re-running the whole array (`ORIGIN_STRIDE=16 sbatch --array=5-5 hpc/s1_array.slurm`), since a
# one-element array cannot infer the stride of the run it belongs to.
TASK_MIN="${SLURM_ARRAY_TASK_MIN:-0}"
if [ -n "${ORIGIN_STRIDE:-}" ]; then
    TASK_COUNT="${ORIGIN_STRIDE}"
    echo "slice      : stride PINNED by ORIGIN_STRIDE=${ORIGIN_STRIDE} (single-slice re-run)"
else
    TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-$(( ${SLURM_ARRAY_TASK_MAX:-0} - TASK_MIN + 1 ))}"
    if [ "$TASK_MIN" -ne 0 ]; then
        echo "FATAL: --array must start at 0 (got min=$TASK_MIN). Offsets are 0-based; a 1-based" >&2
        echo "       array leaves slice 0's origins unfitted and pushes the last task out of" >&2
        echo "       range. To re-run ONE slice of an earlier run, pin the stride instead:" >&2
        echo "       ORIGIN_STRIDE=<original> sbatch --array=<id>-<id> $0" >&2
        exit 1
    fi
fi
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$TASK_COUNT" ]; then
    echo "FATAL: task id ${SLURM_ARRAY_TASK_ID:-0} >= stride ${TASK_COUNT} — no such slice." >&2
    exit 1
fi

# Private, writable Julia depot per array task (docx §2-3-1). Two paths for ONE directory: the
# container sees /workdir, the host sees $PROJECT_DIR.
#   ⚠ The docx's `rm -rf "${TEMP_DEPOT}"` "does not work" (§1-4) for exactly this reason — it uses
# the /workdir path OUTSIDE the container, where it does not exist. Remove the HOST path.
# Defaults so `set -u` does not abort when this is run by hand (no array, or no Slurm at all) —
# the depot name only has to be unique per concurrent task.
CPUS="${SLURM_CPUS_PER_TASK:-1}"
DEPOT_NAME="julia_depot_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-0}"
DEPOT_HOST="${DEPOT_ROOT}/${DEPOT_NAME}"
DEPOT_CONT="/workdir/${DEPOT_NAME}"
trap 'echo "removing ${DEPOT_HOST}"; rm -rf "${DEPOT_HOST}"' EXIT   # also fires on the time-limit SIGTERM
mkdir -p "${DEPOT_HOST}" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/dt_intermediate"

echo "image      : ${IMAGE}"
echo "project    : ${PROJECT_DIR}"
echo "depot      : ${DEPOT_HOST}"
echo "slice      : ${SLURM_ARRAY_TASK_ID:-0}/${TASK_COUNT}"
echo "sampler    : STAGE1_USE_NUTS=${STAGE1_USE_NUTS}, AD_BACKEND=${AD_BACKEND}"
[ -f "${IMAGE}" ] || { echo "FATAL: image not found: ${IMAGE}" >&2; exit 1; }

# `module` is a shell function inherited from the submitting login shell (sbatch exports the
# environment by default). If it did not survive, initialise it rather than dying on "command not
# found" — under `set -e` that would kill the task before a single fit.
if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh          # an `if` condition is exempt from set -e; a bare `||` is not
fi
module load singularity

# ⚠ JULIA_DEPOT_PATH uses a SINGLE colon. The docx's "…::/opt/julia" contains an EMPTY entry, which
# Julia expands to the DEFAULT depots — i.e. the shared ~/.julia on the host home, concurrently
# written by every array task. That is the conflict behind the docx §2-3 `IOError -116`.
#   No Pkg.instantiate() at run time either: the depot is complete inside the image, compute nodes
# have no network, and docx §1-3-6 reached the same conclusion the hard way.
singularity exec --bind "${PROJECT_DIR}:/workdir" \
    --env JULIA_DEPOT_PATH="${DEPOT_CONT}:/opt/julia" \
    --env JULIA_NUM_THREADS="${CPUS}" \
    --env PHASE="s1" \
    --env STAGE1_USE_NUTS="${STAGE1_USE_NUTS}" \
    --env AD_BACKEND="${AD_BACKEND}" \
    --env ORIGIN_STRIDE="${TASK_COUNT}" \
    --env ORIGIN_OFFSET="${SLURM_ARRAY_TASK_ID:-0}" \
    --env FIT_END="${FIT_END}" \
    ${ORIGIN_MIN:+--env ORIGIN_MIN="${ORIGIN_MIN}"} \
    --env S1_CONCURRENCY="${CPUS}" \
    --env MAX_ORIGINS="${MAX_ORIGINS}" \
    --env CHUNK="${CHUNK}" \
    --env MEM_FLOOR_GIB="${MEM_FLOOR_GIB}" \
    "${IMAGE}" \
    /bin/sh /workdir/hpc/run_grid.sh

echo "time ended"
date
