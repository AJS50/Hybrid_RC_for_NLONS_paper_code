using Pkg; Pkg.activate(".")
include("$(pwd())/src/HybridRCforNLONS.jl")
using OrdinaryDiffEq, Random, Statistics, Distributions, LinearAlgebra, Plots
import HybridRCforNLONS: cartesian_kuramoto, cartesian_kuramoto_p, normalised_error, generate_ODE_data_task2, ESN, Hybrid_ESN, train_reservoir!, predict!, ingest_data!, initialise_reservoir!, valid_time, sqr_even_indices
plotlyjs()

## Generate lorenz data
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3])-u[2]
    du[3] = u[1]*u[2]-β*u[3]
end

p=[10.0, 28.0, 8/3]
u0 = [-5.0, 0.0,20.0]
tspan = (-100.0, 400.0)
dt=0.1
t = collect(tspan[1]:dt:tspan[2])
prob=ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=t)
sol=sol[1001:end] #remove initial transient
#check ground truth
plot(sol, idxs=(1,2,3), linewidth=1, legend=false,color=:black)

#divide into train and test.
train_data=reduce(hcat,sol[1:1000].u)
target_data=reduce(hcat,sol[2:1001].u)
warmup_data=reduce(hcat,sol[1901:2000].u)
test_data=reduce(hcat,sol[2000:end].u)
test_length=size(test_data,2)

##create standard and hybrid reservoirs
res_size=250
mean_degree=3
data_length=length(u0)
spectral_radius=0.6
input_scaling=0.25
g=1.0
regularisation_strength=0.3
NLAT=sqr_even_indices
standard_reservoir=ESN(res_size, mean_degree, data_length, spectral_radius, input_scaling, g, regularisation_strength, NLAT)

initialisation_rng=MersenneTwister(42)
initialise_reservoir!(initialisation_rng,standard_reservoir)

function expert_model(du,u,p,t)
    σ, ρ, β = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3])-u[2]
    du[3] = u[1]*u[2]-β*u[3]
end

model_dimension=3
knowledge_ratio=0.5
noise_distribution=Normal(0.0,0.01) #as in Pathak et al (2018)
parameter_errors=(rand(initialisation_rng,noise_distribution,3))
expert_parameters=p.*(parameter_errors.+1)
hybrid_reservoir=Hybrid_ESN(res_size, mean_degree, data_length, model_dimension, knowledge_ratio, spectral_radius, input_scaling, g, regularisation_strength, NLAT,expert_model,expert_parameters,dt)

initialise_reservoir!(initialisation_rng,hybrid_reservoir)

## train_reservoirs
ingest_data!(standard_reservoir,train_data)
train_reservoir!(standard_reservoir,target_data)

ingest_data!(hybrid_reservoir,train_data)
train_reservoir!(hybrid_reservoir,target_data)

##predict.
ingest_data!(standard_reservoir,warmup_data)
standard_prediction=predict!(standard_reservoir,test_length,false,false)
ingest_data!(hybrid_reservoir,warmup_data)
hybrid_prediction=predict!(hybrid_reservoir,test_length,false,false)

#plot results
xlimits=(-10.0,15)
plot(range(start=-10+dt, stop=0,step=dt),warmup_data[1,1:end], linewidth=1, label="Warmup",color=:black,xlabel="time (s)",xlims=xlimits);
plot!(range(start=0,stop=test_length*dt,length=test_length),test_data[1,:], linewidth=1, label="Ground Truth",color=:purple,linestyle=:dash);
plot!(range(start=0,stop=test_length*dt,length=test_length),standard_prediction[1,:], linewidth=1, label="Standard",color=:blue);
plot!(range(start=0,stop=test_length*dt,length=test_length),hybrid_prediction[1,:], linewidth=1, label="Hybrid",color=:red)
