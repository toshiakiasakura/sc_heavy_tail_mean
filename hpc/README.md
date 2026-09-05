# Running the 8j grid on the LSHTM HPC (Slurm + Singularity)

Everything here fits the pattern in `B4a07_lang_Slurm_singularity_LSHTM_hpc.docx`: build a Docker
image on Windows → convert with **docker2singularity** → `sbatch` a job that `module load
singularity` and `singularity exec --bind /home/<user>/<project>:/workdir`, with a **per-job Julia
depot**. Three places where this departs from the docx are marked ⚠ below — two deliberate design
choices (§"Departures") and one plain fix to the docx's build command, which no longer runs as
written.

Scope: **the 8j two-stage grid only** (Stage 1 contact-degree NUTS, Stage 2 pooled transmission).
Scoring and diagnostics (9j–13j) stay on the devcontainer.

| file | what it is |
|---|---|
| `Dockerfile.hpc` | the HPC image — lean, with the Julia depot **baked in** at `/opt/julia` |
| `build_image.ps1` | Windows: `docker build` + `docker2singularity` → `.simg` |
| `config.sh` | the only file to edit for account / paths / image / grid selection |
| `s1_array.slurm`, `s2_array.slurm` | the two array jobs |
| `run_grid.sh` | in-container runner: relaunches `src/8j_run_grid.jl` per its exit-code protocol |

---

## 1. Build the image (Windows, Docker Desktop running)

```powershell
cd C:\Users\aflub\Documents\sc_heavy_tail_mean
pwsh hpc/build_image.ps1
```

30–60 min for the build (R packages, `Pkg.instantiate()` + `Pkg.precompile()`), then 20–40 min for
the conversion. The `.simg` lands in `C:\Users\aflub\Downloads`; the script prints its name.

⚠ **Disk.** The docker image is ~12.6 GB and `docker2singularity` stages an uncompressed tar of it
inside the Docker VM before building the squashfs, so **keep ~40 GB free on C:** for the round trip.
Docker Desktop's VHDX does not shrink on its own afterwards — reclaim with `docker system prune -a`
(and, if it matters, Docker Desktop → *Troubleshoot → Clean / Purge data*). The `.simg` itself is
only ~3–5 GB, because squashfs compresses.

**Why a separate image.** `.devcontainer/Dockerfile` is an interactive image: Node.js, Claude Code,
RTK, no `scoringutils`, and a Julia depot that lives in an empty named volume — so a job built from
it would try to download and precompile the whole Turing stack on a compute node with no network.
`Dockerfile.hpc` bakes packages *and* precompile caches into `/opt/julia` and makes them
world-readable (Singularity runs as the **host** user, not `jovyan`).

⚠ **Rebuild whenever `Manifest.toml` changes.** At run time `/workdir` supplies
`Project.toml`/`Manifest.toml` from the bind mount while the packages come from the image. If they
disagree, Pkg wants to resolve and cannot. The preflight in §3 is the check.

⚠ **The docx's conversion command no longer works as written on a current Docker Desktop.**
`singularityware/docker2singularity` ships a Docker 18.09.8 client (API 1.39) and modern daemons
refuse anything below 1.40 — it dies with `client version 1.39 is too old`. `build_image.ps1` passes
`-e DOCKER_API_VERSION=1.41`, which pins the version the old client *advertises* and is enough;
the calls it makes (`images`/`save`/`inspect`) are unchanged. Verified against Docker Desktop
29.6.2. If you run the docx command by hand, add that flag.

## 2. Transfer (WinSCP + PuTTY)

Into `/home/<user>/prj_sc_heavy_tail_mean/` (= `PROJECT_DIR` in `config.sh`):

| what | note |
|---|---|
| the repo (`src/`, `hpc/`, `Project.toml`, `Manifest.toml`, `inst/`, `tasks/`) | `git clone` on the login node is fine |
| `dt_comix_no_public/part_uk.arrow`, `contacts_uk.arrow` | **gitignored — WinSCP only** |
| `inc2prev/data-processed/populations.csv`, `inc2prev/outputs/estimates_age_ab.csv` | the submodule; only these two files are read |
| the `.simg` | ~3–5 GB (squashfs of the 12.6 GB image) |

Then, on the login node:

```bash
cd ~/prj_sc_heavy_tail_mean
mkdir -p logs dt_intermediate res
chmod +x hpc/*.sh hpc/*.slurm
head -1 hpc/run_grid.sh | od -c | head -1     # must end \n, NOT \r \n
```

⚠ **Line endings.** `core.autocrlf=true` is set in this repo, and WinSCP transfers the *working
tree* — a CRLF copy of these scripts fails on Linux with `$'\r': command not found`, which is an
opaque way to lose a job. `.gitattributes` pins `*.sh`/`*.slurm`/`Dockerfile*` to LF so this cannot
happen; the `od -c` above is the one-second confirmation. (`dos2unix hpc/*.sh hpc/*.slurm` if it
ever does.)

⚠ **Check the quota before you start.** A Stage-1 NUTS chain is ~28 MB (NegBin) / ~72 MB
(hurdle-Weibull) at `nuts_draws = 2000`, and the grid is 4+4 of them per origin × 63 origins ≈
**25 GB**, plus the Stage-2 pooled files and the ~4–5 GB image. `quota -s` / `df -h ~` first; a
full filesystem part-way through a 5 h array is an expensive way to find out. (Since artefacts are
now written temp-then-rename, running out of space fails the *temp* write and the target path is
never touched — a loud, resumable failure rather than a truncated file the next run would skip as
complete.)

⚠ **Check what is already in `dt_intermediate/` first.** `prefit_stage1!` **skips existing
artefacts**, so a stale chain under the same token is silently kept, never overwritten. If any
`8j_s1_*temporal-w8h-lc0-nuts*` predates the current φ-init, move it aside — see CLAUDE.md,
"`stage1_phi_init_scale` is the SHARPER version of that hazard".

## 3. Preflight (login node, ~5 min) — docx §2-2 / §1-3-7

```bash
cd ~/prj_sc_heavy_tail_mean
module load singularity
singularity exec --bind $PWD:/workdir \
    --env JULIA_DEPOT_PATH="/workdir/.julia_preflight:/opt/julia" \
    --env JULIA_NUM_THREADS=4 \
    ./sc-heavy-tail-hpc-<date>.simg /bin/bash
```

Inside:

```bash
julia --project=/workdir -e 'using Pkg; Pkg.status()'      # must resolve, download NOTHING
R -e 'library(scoringutils); packageVersion("scoringutils")'
cd /workdir/src && julia --project=/workdir -e 'include("forecast_utils.jl"); println("preamble ok")'
```

Then a **dry run** of the real driver — it resolves the token, the slice and the batch, prints them,
and stops before a single fit (`DRY_RUN=1`, exit 10). Cheap enough for a login node, which a real
fit is *not*: an hour of NUTS on a shared login node is the docx's "Top 10 Mistakes" §1.

```bash
cd /workdir/src
DRY_RUN=1 PHASE=s1 STAGE1_USE_NUTS=true AD_BACKEND=mooncake \
ORIGIN_STRIDE=16 ORIGIN_OFFSET=5 \
julia --project=/workdir /workdir/src/8j_run_grid.jl
```

Check three lines: `cache token : temporal-w8h-lc0-nuts` (a different token = a different
generation — fix `STAGE1_USE_NUTS` before launching), `sampler : Stage 1 NUTS, Stage 2 Pathfinder`,
and `slice : 5/16 ⇒ … origin(s): …` listing the origins that task will own. Delete
`.julia_preflight` afterwards.

Then let the **first array task** be the real smoke — as a job, not on the login node:

```bash
ORIGIN_MIN=2021-04-25 FIT_END=2021-04-25 sbatch --array=0-0 hpc/s1_array.slurm
```

One origin, 8 Stage-1 chains, ~1 h. Confirm the artefacts and their provenance keys (§6) before
submitting the full array.

## 4. Submit

```bash
cd ~/prj_sc_heavy_tail_mean          # --output paths are relative to here, and logs/ must exist
sbatch hpc/s1_array.slurm            # 16 tasks × ~4 origins ≈ 5 h
# ... wait for Stage 1 to be COMPLETE, then:
sbatch hpc/s2_array.slurm            # 16 tasks, under an hour
```

Each task owns a **round-robin slice** of the 63 origins (`ORIGIN_OFFSET=$SLURM_ARRAY_TASK_ID`,
`ORIGIN_STRIDE=$SLURM_ARRAY_TASK_COUNT`), so no two tasks ever write the same artefact. The slice and
its origin list are printed at the top of every task log — that print is the evidence the partition
was disjoint.

⚠ **The array must start at 0** (`--array=0-15`, not `1-16`); both scripts refuse otherwise, because
a 1-based array would leave slice 0's origins unfitted.

To resize: `sbatch --array=0-7 hpc/s1_array.slurm` (fewer, longer tasks).

To re-run **one** failed slice, pin the stride of the run it belonged to — a one-element array
cannot infer it, and getting it wrong would fit a different set of origins:

```bash
ORIGIN_STRIDE=16 sbatch --array=5-5 --export=ALL hpc/s1_array.slurm
```

Resubmitting the whole array also works and is safe: it is resumable and skips everything already
fitted.

Each task loops `8j_run_grid.jl` in **fresh processes** (`run_grid.sh`), `MAX_ORIGINS=2` at a time.
That is not tidiness: a single process grows ~1.08 GiB per origin and only a process boundary resets
it — the 63-origin run was OOM-killed twice for this.

## 5. Monitor

```bash
squeue --user=$USER                       # running / pending
squeue --user=$USER --start                # expected start time
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State
tail -f logs/s1_<jobid>_0.log
ls dt_intermediate/8j_s1_*temporal-w8h-lc0-nuts* | wc -l    # target 504 (s1), 1512 for 8j_s2_*
scancel <jobid>
```

## 6. Collect and gate

Bring `dt_intermediate/` back with WinSCP, then **before any analysis** confirm the grid is of one
provenance (CLAUDE.md: the AD backend and φ-init are deliberately *not* in the cache token, so a
part-refitted grid is mixed with nothing in the filename to show it):

```julia
using Glob, JLD2, StatsBase
key(p, k) = jldopen(f -> haskey(f, k) ? f[k] : :ABSENT, p)
ps = glob("8j_s1_*temporal-w8h-lc0-nuts*", "dt_intermediate")
countmap(key.(ps, "ad_backend"));  countmap(key.(ps, "phi_init_scale"));  countmap(key.(ps, "sampler"))
```

Each must be a **single** value. `:ABSENT` means a pre-2026-08-10 file survived the clean-out.

---

## Departures from the docx

Two are deliberate design choices; the third (§1) is a fix — the docx's own conversion command dies
on a current Docker Desktop without `-e DOCKER_API_VERSION=1.41`.

1. **`rm -rf` uses the HOST path.** Docx §1-4 notes *"Unfortunately, rm -rf part does not work"* — its
   `TEMP_DEPOT` is `/workdir/julia_depot_…`, which exists only *inside* the container, while the `rm`
   runs outside it. The `.slurm` files keep both names (`DEPOT_CONT` for `--env`, `DEPOT_HOST` for
   the filesystem) and remove the host one from a `trap … EXIT`, so it is cleaned up on the
   time-limit SIGTERM too.
2. **`JULIA_DEPOT_PATH="<per-task depot>:/opt/julia"` — one colon.** The docx's `"…::/opt/julia"`
   contains an empty entry, which Julia expands to the *default* depots, i.e. the shared `~/.julia`
   on the host home that every array task would write at once. That is the conflict behind the docx
   §2-3 `IOError -116`. For the same reason there is **no `Pkg.instantiate()` at run time** (docx
   §1-3-4): the depot is already complete in the image, and docx §1-3-6 reached the same conclusion
   empirically.

## Code that had to change for the cluster

| where | why |
|---|---|
| `src/8j_run_grid.jl` — `ORIGIN_STRIDE`/`ORIGIN_OFFSET` | array tasks must own disjoint origins; without it every task takes `pending[1:MAX_ORIGINS]` from the same disk view and races. |
| `src/8j_run_grid.jl` — `DRY_RUN=1` | the login-node preflight: resolve and print the token, slice and batch, then exit 10 before any fit. |
| `src/joint_model.jl` — `_slurm_mem_available_gib` | `/proc/meminfo` reports the whole 768 GB node, not the job's cgroup, so `MEM_FLOOR_GIB` and `fit_concurrency`'s memory cap could never bind and the OOM guard was inert. |
| `src/joint_model.jl` — `_atomic_jldsave` | a time-limit kill mid-`jldsave` leaves a truncated `.jld2` that every `isfile` skip-check counts as complete; write-then-rename makes the artefact appear atomically. |

None of them touches a model, a prior, a `FrameworkConfig` default or the cache token: the grid this
produces is the grid the devcontainer would have produced.
