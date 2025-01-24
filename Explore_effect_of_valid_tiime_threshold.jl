using Pkg; Pkg.activate(".")
include("$(pwd())/src/HybridRCforNLONS.jl")
using Statistics, LinearAlgebra, Plots, DataFrames, DelimitedFiles, CSV, Arrow
import .HybridRCforNLONS: valid_time, sqr_even_indices, valid_time
plotlyjs()

#load in the trajectory data.
