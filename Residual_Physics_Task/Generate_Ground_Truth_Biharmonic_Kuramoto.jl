using Pkg; Pkg.activate(".")
include("$(pwd())/src/HybridRCforNLONS.jl")
import .HybridRCforNLONS: biharmonic_kuramoto, biharmonic_kuramoto_p, biharmonic_kuramoto_ic, reset_condition1, reset_affect1!, reset_condition2, reset_affect2!, generate_arrow, phasetoxy, xytophase,kuramoto_order2
using OrdinaryDiffEq, Random, DataFrames, CSV, Arrow

#four conditions: synchronous, asynchronous, heteroclinic cycles and self consistent partial synchrony,.

# Parameters:
# a=0.2. N=10. K=1. nat freqs drawn from a very close distribution: Lorentzian(0.0, 0.1), Gamma2=pi
# Synch: Gamma1=6.283 (2pi)
# Asynch: Gamma1=pi
# Heteroclinic Cycles: Gamma1=1.300
# Self Consistent Partial Synchrony: Gamma1=1.5

#these are therefore 4 models for now. with some given random seed generating the natural frequencies from a lorentzian distribution. (1234+i) with i = [1,2,3,4] for synch, asynch, HC, SCPS

output_path="$(pwd())/Residual_Physics_Task/Settings_and_GroundTruth/"

γ_1s=[2*Float64(pi),Float64(pi),1.3,1.5,Float64(pi),Float64(pi),Float64(pi)]
γ_2=Float64(pi)
a=0.2
N=10 #number of oscillators
K=1.0 #coupling strength
tspan=(0.0,6200.0) #one large trajectory for each model will be generated. To be split into training, warmup and test segments as required.
dt=0.1 #time step.
μ=0.0 #mean of the natural frequency distribution
Δω=0.01 #width of the natural frequency distribution

callback=CallbackSet(
    VectorContinuousCallback(reset_condition1,reset_affect1!,N),
    VectorContinuousCallback(reset_condition2,reset_affect2!,N)
    )
    
cases=["Synch","Asynch","HeteroclinicCycles","SelfConsistentPartialSynchrony","Asynch_Fast","Asynch_Fast_2","Asynch_Fast_10"]
for (cidx,case) in enumerate([cases[7]])
    cidx=7
    rng=MersenneTwister(1234+cidx)
    if case=="Asynch_Fast"
        Δω=0.05
        K=5.0
    elseif case=="Asynch_Fast_2"
        Δω=0.02
        K=2.0
    elseif case=="Asynch_Fast_10"
        Δω=0.1
        K=10.0
    end
    base_params=biharmonic_kuramoto_p(rng,N,μ,Δω,K,a,γ_1s[cidx],γ_2)
    ic=biharmonic_kuramoto_ic(N) #same initial conditions for every run (internally this uses a MersenneTwister rng with seed 1234)
    prob=ODEProblem(biharmonic_kuramoto,ic,tspan,base_params;callback=callback)
    gt_data=permutedims(reduce(hcat,solve(prob,Rodas4P(),dtmax=1/32,adaptive=true,saveat=dt).u))
    name="Biharmonic_Kuramoto_$(case)_ground_truth_data"
    CSV.write(output_path*name*".csv",DataFrame(gt_data,:auto),writeheader=true)
    generate_arrow(name,output_path)
    rm(output_path*name*".csv")
end

#test: read and plot trajectories:
using Plots
plots=Vector{Any}()
default(titlefontsize=15)
default(xlabelfontsize=11)

for case in cases
    gt_read=permutedims(Matrix(DataFrame(Arrow.Table(output_path*"Biharmonic_Kuramoto_$(case)_ground_truth_data.arrow.lz4"))))
    gt_read=[phasetoxy(gt_read[:,i]) for i in 1:size(gt_read,2)]
    gt_read=reduce(hcat,gt_read)
    push!(plots,plot(collect(0:0.1:299.9),gt_read'[1:3000,1:10],legend=false,title="$(case)",xlabel="time s",xlims=(0.0,200),bottom_margin=10Plots.mm))
    push!(plots,plot(abs.(kuramoto_order2(xytophase(gt_read[1:20,1:3000]),10)[1,:])))
end
plot(plots...,size=(2000,1000),layout=(7,2))

#plot_order
