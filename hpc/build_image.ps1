<#
.SYNOPSIS
    Build the HPC Docker image and convert it to a Singularity .simg (docx §1-3-3 / §2-1).

.DESCRIPTION
    Run from the REPO ROOT on the Windows box, with Docker Desktop running:

        pwsh hpc/build_image.ps1

    Two steps, both slow the first time:
      1. docker build -f hpc/Dockerfile.hpc      (~30-60 min: R packages + Pkg.instantiate/precompile)
      2. docker2singularity                      (~10-20 min, needs --privileged and the docker socket)

    The .simg lands in -OutputDir (default C:\Users\aflub\Downloads) and is what you upload to the
    HPC with WinSCP. Its name is chosen by docker2singularity from the image tag; the script prints
    the file it produced — copy that name into hpc/config.sh as IMAGE_NAME.

.PARAMETER Tag
    Docker image tag. Default sc-heavy-tail-hpc:<yyyyMMdd>. The date is part of the provenance:
    an artefact grid is only comparable within one image.

.PARAMETER OutputDir
    Windows directory that receives the .simg.

.PARAMETER SkipBuild
    Convert an image that is already built.
#>
param(
    [string]$Tag       = ("sc-heavy-tail-hpc:" + (Get-Date -Format "yyyyMMdd")),
    [string]$OutputDir = "C:\Users\aflub\Downloads",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

# Repo root = parent of this script's directory. Docker build context must be the root: the
# Dockerfile COPYs Project.toml/Manifest.toml from there.
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
Write-Host "repo root : $repoRoot"
Write-Host "tag       : $Tag"
Write-Host "output    : $OutputDir"

if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

if (-not $SkipBuild) {
    Write-Host "`n=== 1/2  docker build (this takes 30-60 min) ============================="
    docker build -f hpc/Dockerfile.hpc -t $Tag .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
}

# Sanity-check BEFORE spending 20 minutes converting. Both flags mirror how Slurm will run it:
#   --entrypoint   `singularity exec` bypasses the image entrypoint (no start.sh, no conda hooks),
#                  so check the depot under exactly that condition rather than a friendlier one;
#   JULIA_DEPOT_PATH  the run-time arrangement — an empty WRITABLE depot in front of the baked
#                  read-only one. If resolution needs anything from the network this fails here,
#                  on a machine that has network, instead of on a compute node that does not.
Write-Host "`n=== check  baked depot resolves the way Slurm will ======================="
docker run --rm --entrypoint julia -e JULIA_DEPOT_PATH="/tmp/probe_depot:/opt/julia" $Tag `
    --project=/workdir -e 'using Pkg; Pkg.status(); using Turing, Pathfinder, Mooncake, RCall; println("depot ok")'
if ($LASTEXITCODE -ne 0) { throw "baked depot does not resolve - do not ship this image" }

Write-Host "`n=== 2/2  docker2singularity (docx 1-3-3) ================================="
$before = @(Get-ChildItem -Path $OutputDir -Filter *.simg -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Name)

# ⚠ -e DOCKER_API_VERSION is an addition to the docx recipe, and without it this step FAILS on a
# current Docker Desktop: the docker2singularity image ships a Docker 18.09.8 client (API 1.39) and
# the modern daemon refuses anything below 1.40 — "client version 1.39 is too old". Pinning the
# version the client ADVERTISES is enough; the calls it actually makes (images/save/inspect) are
# unchanged. Verified against Docker Desktop 29.6.2 on 2026-08-12.
docker run -v /var/run/docker.sock:/var/run/docker.sock `
           -v "${OutputDir}:/output" `
           -e DOCKER_API_VERSION=1.41 `
           --privileged -t --rm `
           singularityware/docker2singularity $Tag
if ($LASTEXITCODE -ne 0) { throw "docker2singularity failed" }

$new = @(Get-ChildItem -Path $OutputDir -Filter *.simg |
         Where-Object { $before -notcontains $_.Name } |
         Sort-Object LastWriteTime -Descending)

if ($new.Count -eq 0) {
    Write-Warning "No new .simg appeared in $OutputDir - check the docker2singularity output above."
} else {
    $simg = $new[0]
    Write-Host "`nBuilt: $($simg.FullName)  ($([math]::Round($simg.Length/1GB,2)) GB)"
    Write-Host "Next:"
    Write-Host "  1. WinSCP it to  /home/<user>/prj_sc_heavy_tail_mean/"
    Write-Host "  2. Set IMAGE_NAME=$($simg.Name)  in hpc/config.sh"
    Write-Host "  3. Follow hpc/README.md from 'Preflight'"
}
