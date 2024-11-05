# Hybrid_RC_for_NLONS_paper_code
This is the code that can be used to prodiuce the results and figures of our paper "Modeling nonlinear oscillator networks using physics-informed hybrid reservoir computing".

## Julia project use:
All package information is contained in the project.toml and manifest.toml files, such that you can create a julia project from within the main repo directory and it will automatically load the correct packages, the following packages and versions were used.

Arrow v2.7.2
CSV v0.10.14
DataFrames v1.6.1
DelimitedFiles v1.9.1
Distributions v0.25.109
DynamicalSystems v3.3.17
OrdinaryDiffEq v6.84.0
PlotlyJS v0.18.13
Plots v1.40.4
Revise v3.5.14
LinearAlgebra
Random
Statistics v1.10.0


using OrdinaryDiffEq, Random, Statistics, Distributions, LinearAlgebra, CSV, Arrow, DataFrames, DelimitedFiles, DynamicalSystems, Plots
using Plots, PlotlyJS, DelimtedFiles, Statistics
using OrdinaryDiffEq, Random, DataFrames, CSV, Arrow
using Distributions, Statistics,  Random, LinearAlgebra, OrdinaryDiffEq, Arrow, CSV, DataFrames

#to remove from the project when sweep is done.
ChaosTools v3.1.2
ColorSchemes v3.25.0
DSP v0.7.9
DelayEmbeddings v2.7.4
Distances v0.10.11
GR v0.73.6
LaTeXStrings v1.3.1
PlotlyBase v0.8.19
ProgressMeter v1.10.2
ScikitLearn v0.7.0
StatsBase v0.33.21
Zygote
ProgressLogging
ForwardDiff
DynamicAxisWarping
