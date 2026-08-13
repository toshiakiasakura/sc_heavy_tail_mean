# hpc/config.sh — site + run settings, sourced by both .slurm files.
#
# The ONLY file to edit when the account, project directory, image or grid selection changes.
# Sourced, never executed: no shebang, no side effects beyond setting variables.

# ── site ────────────────────────────────────────────────────────────────────────────────────
HPC_USER="${HPC_USER:-lsh2200567}"
MAIL_USER="${MAIL_USER:-${HPC_USER}@lshtm.ac.uk}"

# Project directory ON THE HPC — the repo, the data and the .simg all live here, and it is what
# gets bind-mounted to /workdir inside the container.
PROJECT_DIR="${PROJECT_DIR:-/home/${HPC_USER}/prj_sc_heavy_tail_mean}"

# The Singularity image, as produced by hpc/build_image.ps1 and uploaded by WinSCP.
# ⚠ Set this to the actual file name docker2singularity printed — it embeds the build date, and a
# grid is only comparable within one image.
IMAGE_NAME="${IMAGE_NAME:-sc-heavy-tail-hpc-20260812.simg}"
IMAGE="${IMAGE:-${PROJECT_DIR}/${IMAGE_NAME}}"

# Where each array task's private, writable Julia depot is created. The bind-mounted project dir is
# the docx default and keeps the depot visible from inside the container without a second --bind.
# Point it at node-local scratch (e.g. /tmp) if home-directory quota or NFS contention bites.
DEPOT_ROOT="${DEPOT_ROOT:-${PROJECT_DIR}}"

# ── the generation being fitted ─────────────────────────────────────────────────────────────
# ⚠ These decide WHICH GRID is written. `stage1_use_nuts` is what appends `-nuts` to the cache
# token, so STAGE1_USE_NUTS=false here does not "run the old sampler" — it starts a DIFFERENT
# generation and refits from scratch. true = the formal NUTS run, token `temporal-w8h-lc0-nuts`.
STAGE1_USE_NUTS="${STAGE1_USE_NUTS:-true}"
AD_BACKEND="${AD_BACKEND:-mooncake}"        # Stage 1 only; Stage 2 is always ReverseDiff (cfg)

# Origin selection — must MATCH between the s1 and s2 jobs, or Stage 2 looks for Stage-1 chains that
# were never fitted (and, per CLAUDE.md, silently fits them serially instead of failing).
FIT_END="${FIT_END:-2021-12-31}"            # ⇒ 63 origins, last 2021-12-26
# ORIGIN_MIN is deliberately UNSET by default (all origins). Export it for a smoke run, e.g.
#   ORIGIN_MIN=2021-04-25 FIT_END=2021-04-25 sbatch --array=0-0 hpc/s1_array.slurm
# ORIGIN_MIN="2021-04-25"

# ── per-process memory bounds (see 8j_run_grid.jl's header) ─────────────────────────────────
# A single process grows ~1.08 GiB per origin and the heap is only reset by a PROCESS BOUNDARY, so
# each task does MAX_ORIGINS origins and exits; hpc/run_grid.sh relaunches it until the slice is
# done. Small numbers here cost one Julia startup each (~1 min against a ~1 h origin).
MAX_ORIGINS="${MAX_ORIGINS:-2}"
CHUNK="${CHUNK:-1}"
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-8}"         # cgroup-aware since the _slurm_mem_available_gib change
