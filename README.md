# Hybrid_RC_for_NLONS_paper_code

This is the code that can be used to prodiuce the results and figures of our paper "Modeling nonlinear oscillator networks using physics-informed hybrid reservoir computing".

## Julia project use:

Package information is contained in the project.toml and manifest.toml files, such that you can create a julia project from within the main repo directory and it will automatically load the correct packages, the following packages and versions were used:

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

## Task specific directories:

The code is grouped into directories for the two tasks and for the supplementary figures that are distinct from the task figures. Each folder has its own readme explaining what the code produces and how to run it.

All code should be run from the top level directory (where this readme is), as input/output paths are defined assuming that is the current working directory. Otherwise they will need changing appropriately in each script.

## src directory:
'HybridRCforNLONS.jl' is the source code script, containing the module 'HybridRCforNLONS' that defines all the functions for creating and testing standard and hybrid reservoirs as used in the paper. It also contains dynamical system implementations for ground truth generation and for the expert ODE model in the hybrid reservoir. Some utility functions to compress csv files, and compute normalised mean square error and valid time are also present.

## Lorenz_Example
'Lorenz_Example.jl' is short script comparing a standard and hybrid reservoir's prediction of the 3D lorenz system, using the structs/functions that are used in the main tasks to demonstrate the common workflow used. Reservoir and ground truth parameters are modifiable, along with the trajectory span lengths for the training/warmup/test split.