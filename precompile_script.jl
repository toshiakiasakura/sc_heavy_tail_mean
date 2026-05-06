using CategoricalArrays
using CSV
using DataFrames
using DataFramesMeta
using Dates
using Distributions
using Glob
using GLM
using Images
using JLD2
using LaTeXStrings
using LinearAlgebra
using LogExpFunctions
using Memoization
using Parameters
using Plots
using Pipe
using Printf
using QuadGK
using Random
using RCall
using SpecialFunctions
using StatsBase
using StatsPlots
using StringEncodings
using XLSX

using Base.Threads
using DynamicPPL
using MCMCChains
using Pathfinder
using Turing

# Exercise Distributions
d = Normal(0, 1)
rand(d, 10)
pdf(d, 0.0)

# Exercise DataFrames
df = DataFrame(x = 1:5, y = rand(5))
@transform df :z = :x .+ 1

# Exercise Plots (trigger compilation of the plot pipeline)
p = plot(rand(10))
scatter!(p, rand(10))

# Exercise Turing (minimal model to trigger compilation)
@model function demo(y)
    μ ~ Normal(0, 1)
    y ~ Normal(μ, 1)
end

model = demo(1.0)
chain = sample(model, NUTS(0.65), 50; progress=false)
