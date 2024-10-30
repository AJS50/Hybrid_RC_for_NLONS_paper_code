using Pkg; Pkg.activate(".")
include("$(pwd())/src/HybridRCforNLONS.jl")
using OrdinaryDiffEq, Random, Statistics, Distributions, LinearAlgebra, CSV, Arrow, DataFrames, DelimitedFiles, DynamicalSystems, Plots
import HybridRCforNLONS: valid_time, reset_affect1!, reset_condition1, reset_affect2!, reset_condition2, phasetoxy, xytophase, lorentzian_nat_freqs, sqr_even_indices

### Define new multistep reservoir struct. 'steps_per_step' is the number of internal steps per external step.
mutable struct ESN_Multi
    res_size::Int64
    mean_degree::Float64
    data_length::Int64
    spectral_radius::Float64
    input_scaling::Float64
    state_history::Matrix{Float64}
    current_state::Vector{Float64}
    input_weight_matrix::Matrix{Float64}
    output_weight_matrix::Matrix{Float64}
    reservoir_weight_matrix::Matrix{Float64}
    g::Float64 #scaling similar to spectral radius but for entire update function as per "Inubushi et al 2021" Reservoir Computing textbook.
    regularisation_strength::Float64
    NLAT::Function #nonlinear activation function between reservoir state and output computation
    prediction_state_history::Matrix{Float64}
    steps_per_step::Int64

    function ESN_Multi(res_size::Int64, mean_degree::Int64, data_length::Int64, spectral_radius::Float64, input_scaling::Float64, g::Float64, regularisation_strength::Float64, NLAT::Function,steps_per_step::Int64)
        new(res_size,
            mean_degree,
            data_length, 
            spectral_radius, 
            input_scaling, 
            zeros(Float64, res_size, data_length), 
            Vector{Float64}(undef, res_size), 
            zeros(res_size, data_length), 
            Matrix{Float64}(undef, data_length, res_size), 
            Matrix{Float64}(undef, res_size, res_size), 
            g, 
            regularisation_strength, 
            NLAT, 
            Matrix{Float64}(undef,res_size,1),
            steps_per_step
            )
        
    end
end

function update_state!(reservoir::ESN_Multi,input::Vector{Float64})
    for i in 1:reservoir.steps_per_step
        if i==1 #take in input data instance on first step.
            reservoir.current_state=tanh.(reservoir.g.*(reservoir.reservoir_weight_matrix*reservoir.current_state.+reservoir.input_weight_matrix*input))
        else #propagate internal state using only internal connectivity matrix for the update.
            reservoir.current_state=tanh.(reservoir.g.*(reservoir.reservoir_weight_matrix*reservoir.current_state))
        end
    end
end

function initialise_reservoir!(rng,reservoir::ESN_Multi)
    #create reservoir weight matrix:
    reservoir.reservoir_weight_matrix=zeros(reservoir.res_size,reservoir.res_size)
    edge_probability=reservoir.mean_degree/reservoir.res_size

    decision_samples=rand(rng,reservoir.res_size,reservoir.res_size)

    for i in 1:reservoir.res_size
        for j in 1:reservoir.res_size
            if decision_samples[i,j]<=edge_probability
                #weights uniform sample ∈ [-1,1]
                reservoir.reservoir_weight_matrix[i,j]=2*rand(rng)-1
            end
        end
    end

    #get current spectral radius
    res_ρ=maximum(abs.(eigvals(reservoir.reservoir_weight_matrix)))
    #scale to desired spectral radius
    reservoir.reservoir_weight_matrix .*= reservoir.spectral_radius/res_ρ
    
    #create input weight matrix
    for i in 1:reservoir.res_size
        reservoir.input_weight_matrix[i,rand(rng,1:reservoir.data_length)]=reservoir.input_scaling*(2*rand(rng)-1)
    end

    #reset state history
    reservoir.state_history=Matrix{Float64}(undef,reservoir.res_size,1)
    #reset current state
    reservoir.current_state=zeros(reservoir.res_size)
end


function ingest_data!(reservoir::ESN_Multi,data::Matrix{Float64}) #for training or warmup... sets state history size to length of data being processed.
    number_of_data_points=size(data,2)
    reservoir.state_history=Matrix{Float64}(undef,reservoir.res_size,number_of_data_points)
    reservoir.current_state=zeros(reservoir.res_size)
    reservoir.state_history[:,1]=reservoir.current_state
    for i in 2:number_of_data_points
        update_state!(reservoir,data[:,i])
        reservoir.state_history[:,i]=reservoir.current_state
    end
end

function train_reservoir!(reservoir::ESN_Multi,target_data::Matrix{Float64})
    #get res states post non linear transform:
    reservoir_states=reservoir.NLAT(reservoir.state_history)
    #compute output weight matrix
    reservoir.output_weight_matrix=((reservoir_states*reservoir_states' + reservoir.regularisation_strength*I)\(reservoir_states*target_data'))'
end

function compute_output(reservoir::ESN_Multi)
    return reservoir.output_weight_matrix*reservoir.NLAT(reservoir.current_state)
end

function predict!(reservoir::ESN_Multi,num_steps,save_states=false,phase=true)
    prediction=Matrix{Float64}(undef,reservoir.data_length,num_steps)
    if save_states
        reservoir.prediction_state_history=Matrix{Float64}(undef,reservoir.res_size,num_steps)
    end
    #runs from current state (end of state history) onwards
    if phase
        prediction[:,1]=phasetoxy(xytophase(compute_output(reservoir)))
        for i in 2:num_steps
            update_state!(reservoir,phasetoxy(xytophase(prediction[:,i-1])))
            prediction[:,i]=phasetoxy(xytophase(compute_output(reservoir)))
            if save_states
                reservoir.prediction_state_history[:,i]=reservoir.current_state
            end
        end
    else
        prediction[:,1]=compute_output(reservoir)
        for i in 2:num_steps
            update_state!(reservoir,prediction[:,i-1])
            prediction[:,i]=compute_output(reservoir)
            if save_states
                reservoir.prediction_state_history[:,i]=reservoir.current_state
            end
        end
    end
    return(prediction)
end

#generate ground truth heteroclinic cycles data
#using DynamicalSystems.jl format.
function biharmonic_kuramoto_ic_for_DS(N::Int64;range::Vector{Float64}=[0.0,0.5],random=false)
    if random
        upper=range[2]*(2*π)-π
        lower=range[1]*(2*π)-π
        return [lower+(upper-lower)*rand(rng) for i in 1:N]
    else
        ic_rng=Random.MersenneTwister(1234)
        θs=[2*π*rand(ic_rng)-π for i in 1:N]
        return θs
    end
end
function biharmonic_kuramoto_p_for_DS(rng,N::Int64,μ::Float64, Δω::Float64,K::Float64,a::Float64,γ_1::Float64,γ_2::Float64)
    ωs=lorentzian_nat_freqs(N,μ,Δω,rng)
    return [ωs...,K,a,γ_1,γ_2]
end
function biharmonic_kuramoto_for_DS!(du,u,p,t)
    N=Int64(size(u,1))
    θs=u[1:N]
    ωs=p[1:N]
    K=p[N+1]
    a=p[N+2]
    γ_1=p[N+3]
    γ_2=p[N+4]
    for i in 1:N
        du[i]=ωs[i] + (K/N)*sum([(sin(θs[j]-θs[i]+γ_1)+a*sin(2*θs[j]-2*θs[i]+γ_2)) for j in 1:N])
    end
    return nothing
end
cases=["Synch","Asynch","HeteroclinicCycles","SelfConsistentPartialSynchrony"]
γ_1s=[2*Float64(pi),Float64(pi),1.3,1.5]
γ_2=Float64(pi)
a=0.2
N=10
K=1.0
tspan=(0.0,6200.0)
dt=0.1
μ=0.0
Δω=0.01
reset_callback=CallbackSet(
        VectorContinuousCallback(reset_condition1,reset_affect1!,N),
        VectorContinuousCallback(reset_condition2,reset_affect2!,N)
        )

#build 4 coupled ODE systems for the 4 different cases.
systems=Vector{CoupledODEs}(undef,length(cases))
for (cidx,case) in enumerate(cases)
    rng=MersenneTwister(1234+cidx)
    base_params=biharmonic_kuramoto_p_for_DS(rng,N,μ,Δω,K,a,γ_1s[cidx],γ_2)
    ic=biharmonic_kuramoto_ic_for_DS(N) #same initial conditions for every run (internally this uses a MersenneTwister rng with seed 1234)
    diffeq=(alg=Tsit5(),adaptive=true,dtmax=1/32,callback=reset_callback)
    systems[cidx]=CoupledODEs(biharmonic_kuramoto_for_DS!,ic, base_params;diffeq,t0=0.0)
end


total_time=400.5
sampling_time=0.1
#generate ground truth data for the heteroclinic cycles case (3)
Y,t=trajectory(systems[3],total_time,Ttr=2.2,Δt=sampling_time)


#reshape and transform into phase components.
data=reshape(phasetoxy(Matrix(Y)'[:,:]),20,4006)
# plot(data[1,:])
# plot(data[2,:])

#segment into training, warmup and test data.
target_data=data[:,2:1001]
warmup_data=data[:,1200:1300]
test_data=data[:,1301:2000]
training_data=data[:,1:1000]

#compute valid times for 30 reservoir instances using 1 to 10 internal steps.
vts=Array{Float64}(undef,30,10)
for i in 1:10
    multi_res=ESN_Multi(300,3,20,0.5,0.15,1.00,0.000001,sqr_even_indices,i)
    for j in 1:30
        initialise_reservoir!(MersenneTwister(1234+j),multi_res)
        ingest_data!(multi_res,training_data)
        train_reservoir!(multi_res,target_data)
        ingest_data!(multi_res,warmup_data)
        prediction=predict!(multi_res,700,false,true)
        vts[j,i]=valid_time(0.4,test_data[:,:],prediction[:,:],0.1)
    end
end

#using the optimised parameters from the grid search results. 
vts_opt_sr_is_reg=Array{Float64}(undef,30,10)
for i in 1:10
    multi_res=ESN_Multi1(300,3,20,0.05,0.05,1.00,0.0001,sqr_even_indices,i)
    for j in 1:30
        initialise_reservoir!(MersenneTwister(1234+j),multi_res)
        ingest_data!(multi_res,training_data)
        train_reservoir!(multi_res,target_data)
        ingest_data!(multi_res,warmup_data)
        prediction=predict!(multi_res,700,false,true)
        vts_opt_sr_is_reg[j,i]=valid_time(0.4,test_data[:,:],prediction[:,:],0.1)
    end
end

#checking 'effective spectral radius' when using multiple internal steps
test_res=ESN_Multi(300,3,20,0.05,0.05,1.00,0.0001,sqr_even_indices,1)
initialise_reservoir!(MersenneTwister(1234),test_res)
A=test_res.reservoir_weight_matrix
#GET SPECTRAL RADIUS OF A AND A*A AND A*A*A ETC..
for i in 1:10
    println("A^$i: spectral radius =  ",maximum(abs.(eigvals(A^i))))
    println("base SR^$i: = ",0.05^i) #baseline SR is 0.05
    println("rescaled SR = ",exp(log(0.05)/i))
    x=exp(log(0.05)/i)
    println("rescaled effective spectral radius = ",x^i)
end

#using reservoirs with constant effective spectral radius at 0.05
vts_opt_sr_is_reg_constsr=Array{Float64}(undef,30,10)
for i in 1:10
    spectral_radius=exp(log(0.05)/i)
    println("Spectral Radius: ",spectral_radius)
    multi_res=ESN_Multi(300,3,20,spectral_radius,0.05,1.00,0.0001,sqr_even_indices,i)
    for j in 1:30
        initialise_reservoir!(MersenneTwister(1234+j),multi_res)
        if j==1 
            println("effective SR: ",maximum(abs.(eigvals(multi_res.reservoir_weight_matrix^i))))
        end
        ingest_data!(multi_res,training_data)
        train_reservoir!(multi_res,target_data)
        ingest_data!(multi_res,warmup_data)
        prediction=predict!(multi_res,700,false,true)
        vts_opt_sr_is_reg_constsr[j,i]=valid_time(0.4,test_data[:,:],prediction[:,:],0.1)
    end
end

# testing on lorenz data
ds = Systems.lorenz()
sampling_time=0.05
total_time=200
Y_lorenz,t_lorenz=trajectory(ds,total_time,Ttr=2.2,Δt=sampling_time)
data=permutedims(Matrix(Y_lorenz))
# plot(data[1,:])
# plot(data[2,:])

#segment into training, warmup and test data.
training_data=data[:,1:1000]
target_data=data[:,2:1001]
warmup_data=data[:,1200:1300]
test_data=data[:,1301:2000]

vts_lorenz=Array{Float64}(undef,30,10)
for i in 1:10
    lorenz_res=ESN_Multi(300,3,3,0.5,0.15,1.00,0.0001,sqr_even_indices,i)
    reservoirs=[deepcopy(lorenz_res) for q in 1:30] #was using Threads, so created separate reservoir for each instance.
    predictions=Array{Float64}(undef,3,700,30)
    for j in 1:30
        initialise_reservoir!(MersenneTwister(1234+j),reservoirs[j])
        ingest_data!(reservoirs[j],training_data)
        train_reservoir!(reservoirs[j],target_data)
        ingest_data!(reservoirs[j],warmup_data)
        predictions[:,:,j]=predict!(reservoirs[j],700,false,false)
        vts_lorenz[j,i]=valid_time(0.4,test_data[:,:],predictions[:,:,j],0.05)
    end
end

#save each of the valid time matrices in csv files:
using DelimitedFiles
writedlm("/Users/as15635/Documents/Projects/KnowledgeReservoirs2/test/MultiStepReservoirs/vts_lorenz.csv",vts_lorenz,',')
writedlm("/Users/as15635/Documents/Projects/KnowledgeReservoirs2/test/MultiStepReservoirs/vts.csv",vts,',')
writedlm("/Users/as15635/Documents/Projects/KnowledgeReservoirs2/test/MultiStepReservoirs/vts_opt_sr_is_reg.csv",vts_opt_sr_is_reg,',')
writedlm("/Users/as15635/Documents/Projects/KnowledgeReservoirs2/test/MultiStepReservoirs/vts_opt_sr_is_reg_constsr.csv",vts_opt_sr_is_reg_constsr,',')




