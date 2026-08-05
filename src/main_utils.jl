using CategoricalArrays
#using ColorSchemes
using CSV
using DataFrames
using DataFramesMeta
using Dates
using Distributions
using Glob
using GLM
using Images
using Arrow
using JLD2
using LaTeXStrings
using LinearAlgebra
using LogExpFunctions
using Memoization
using Parameters
using Plots
import Pipe: @pipe
using Pkg
using Printf
using QuadGK
using Random
using RCall
using SpecialFunctions
using StatsBase
using StatsPlots
using StringEncodings
using Test
using XLSX

# Turing related packages
using Base.Threads
using Pathfinder
using DynamicPPL
using Turing
using ReverseDiff          # loads the Turing/DynamicPPL ReverseDiff AD extension (AutoReverseDiff)
using Mooncake             # 2026-08-05: loads DynamicPPLMooncakeExt / BijectorsMooncakeExt /
                           # DifferentiationInterfaceMooncakeExt ⇒ AutoMooncake() works. This is the
                           # forecasting framework's DEFAULT backend (`cfg.ad_backend`, framework.jl).
                           # UNCONDITIONAL, like ReverseDiff above: `using` is only legal at top
                           # level, so loading it lazily from `_resolve_adtype` would need an
                           # `@eval Main using …` plus an `invokelatest` world-age barrier — the
                           # freshly-loaded extension methods are invisible to the already-compiled
                           # caller. Notebooks 1j–7j pay the load cost without using AD, which is the
                           # same trade-off `using ReverseDiff`/`Turing`/`Pathfinder` already make.
using MCMCChains

include("distributions/main.jl")
include("utils.jl")
include("degree_dist.jl")
include("fit_utils.jl")
include("turing_utils.jl")
include("turing_models.jl")