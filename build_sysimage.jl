using Pkg
Pkg.activate(@__DIR__)

using PackageCompiler

packages = [
    # Data
    :AbstractMCMC,
    :CSV,
    :CategoricalArrays,
    :DataFrames,
    :DataFramesMeta,
    :Distributions,
    :Glob,
    :GLM,
    :Images,
    :Interpolations,
    :Arrow,
    :JLD2,
    :KernelDensity,
    :LaTeXStrings,
    :LogExpFunctions,
    :Memoization,
    :Memoize,
    :Parameters,
    :Pipe,
    :QuadGK,
    :RCall,
    :SpecialFunctions,
    :StatsBase,
    :StringEncodings,
    :XLSX,
    # Visualisation
    :Plots,
    :StatsPlots,
    # Bayesian / MCMC
    :DynamicPPL,
    :MCMCChains,
    :Mooncake,       # default AD backend (cfg.ad_backend); bakes in Mooncake's own inference/codegen
    :Pathfinder,
    :ReverseDiff,    # fallback AD backend — was never listed here, so RD's load cost was paid every session
    :Turing,
    # Jupyter
    :IJulia,
]

# NOTE (2026-08-05): including :Mooncake bakes in Mooncake's own inference and codegen paths — most
# of its fixed cost — but NOT the derived rule for `model_degree`. `build_rrule` keys on the concrete
# `DynamicPPL.Model` type, which does not exist until `forecast_utils.jl` is included at runtime. So
# `prefit_stage1!`'s per-degree-model warm-up is still required WITH a sysimage, not instead of one.
sysimage_path = joinpath(@__DIR__, "sysimage.so")

pkg_list = join(string.(packages), ", ")
@info "Building sysimage at $sysimage_path — this takes ~15–30 minutes"
@info "Packages: $pkg_list"

create_sysimage(
    packages;
    sysimage_path,
    precompile_execution_file = joinpath(@__DIR__, "precompile_script.jl"),
)

@info "Done. Start Julia with:  julia --sysimage sysimage.so"
