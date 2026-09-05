#!/bin/sh
# hpc/run_grid.sh — the IN-CONTAINER runner. Invoked as the last argument of `singularity exec`.
#
# Implements the supervisor protocol that src/8j_run_grid.jl's header defines:
#
#     0  did work, more remains for this slice  → relaunch in a FRESH process
#    10  nothing pending for this PHASE         → this task's slice is complete
#  else  crash (OOM, fit error)                 → stop, do not power through
#
# The relaunch is not decoration. A single process fitting origins back-to-back grows ~1.08 GiB per
# origin — measured, and measured NOT to be fixed by the per-origin GC.gc() the prefit functions
# already do. Only a process boundary resets the heap. `8j_run_grid.jl` therefore does at most
# MAX_ORIGINS origins and exits 0; this loop supplies the boundary. The Slurm ARRAY is what supplies
# the parallelism (each task owns a disjoint ORIGIN_OFFSET/ORIGIN_STRIDE slice) — the two are
# independent and both are needed.
#
# Everything else — PHASE, the slice, concurrency, the token-defining STAGE1_USE_NUTS — arrives as
# environment from the .slurm file, so this script is identical for Stage 1 and Stage 2.
#
# /bin/sh on purpose: the docx invokes `/bin/sh -c` and the image's shell is not guaranteed to be
# bash. No bashisms below.

set -u

echo "run_grid.sh: host $(hostname) | PHASE=${PHASE:-?} | slice ${ORIGIN_OFFSET:-0}/${ORIGIN_STRIDE:-1}"
echo "run_grid.sh: JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-<unset>}"
echo "run_grid.sh: JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-<unset>}"

# Relative paths in the framework assume cwd == src/ (the driver also cd's itself; this makes the
# assumption visible from the outside).
cd /workdir/src || exit 1

pass=0
while : ; do
    pass=$((pass + 1))
    echo "run_grid.sh: === pass ${pass} ==============================================="
    julia --project=/workdir /workdir/src/8j_run_grid.jl
    rc=$?
    case "$rc" in
        0)  echo "run_grid.sh: pass ${pass} did work, more remains — fresh process"
            ;;
        10) echo "run_grid.sh: slice complete after ${pass} pass(es)"
            exit 0
            ;;
        *)  echo "run_grid.sh: driver exited ${rc} after ${pass} pass(es) — STOPPING" >&2
            exit "$rc"
            ;;
    esac
done
