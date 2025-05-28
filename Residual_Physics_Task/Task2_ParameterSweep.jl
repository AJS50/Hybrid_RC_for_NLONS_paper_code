cd("..")
println("script started pre loading project and packages")
using Pkg; Pkg.activate("$(pwd())")
println("project activated")
flush(stdout)
Pkg.instantiate()
println("project instantiated")
flush(stdout)
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0  # Disable automatic precompilation
println("precompilation disabled")
flush(stdout)
include("$(pwd())/src/HybridRCforNLONS.jl")
println("Hybrid_RC_for_NLONS_paper_code loaded")
flush(stdout)
using OrdinaryDiffEq, Random, Statistics, Distributions, LinearAlgebra, CSV, Arrow, DataFrames, DelimitedFiles
import .HybridRCforNLONS: cartesian_kuramoto, cartesian_kuramoto_p, normalised_error, generate_ODE_data_task2, generate_arrow, ESN, Hybrid_ESN, train_reservoir!, predict!, ingest_data!, initialise_reservoir!, phasetoxy,xytophase,valid_time, sqr_even_indices

println("packages loaded")
flush(stdout)
println("Threads: ",Base.Threads.nthreads())
flush(stdout)

arrayindex=parse(Int,ARGS[1]) #where in the parameter sweep are we? (1-20)
# arrayindex=1

psweep_name=ARGS[2] #to select parameter settings according to the settings csv files. See settings files names for correct names.
# psweep_name="InputScaling"

ground_truth_case=parse(Int64,ARGS[3]) # regimes: 1.Synch, 2.Asynch, 3.Heteroclinic, 4.SCPS, 5.fast asynchronous
# ground_truth_case=5

model_type=ARGS[6] # ODE, Hybrid, Standard.
# model_type="Hybrid"# ODE, Hybrid, Standard.

num_instantiations=40 #how many reservoir or ODE instantiations to test. reduce for quick tests.
# num_instantiations=ARGS[5]

num_tests=20 #how many test spans to predict. maximum 20, as ground truth is always split into 20 warmup-test segments.
# num_tests=ARGS[6]

input_path="$(pwd())/Residual_Physics_Task/Settings_and_GroundTruth/"
# input_path=ARGS[7] #path to settings and ground truth files

# output_path="$(pwd())/Residual_Physics_Task/"
output_path=ARGS[5] #path to parent folder to store output valid times and trajectories. Will generate subfolders for each parameter.

#create parameter specific subfolder in the output path.
save_path=output_path*psweep_name*"/"
if isdir(save_path)
    println("Directory $(psweep_name) exists")
else
    mkdir(save_path)
end

cases=["Synch","Asynch","HeteroclinicCycles","SelfConsistentPartialSynchrony","Asynch_Fast"]
case=cases[ground_truth_case]
γ_1s=[2*Float64(pi),Float64(pi),1.3,1.5, Float64(pi)]
γ_1=γ_1s[ground_truth_case]
γ_2=Float64(pi)
a=0.2

settings=readdlm(input_path*psweep_name*"_sweep_settings.csv",',',header=true)[1]
N,K,system,μ,Δω,res_size,scaling,knowledge_ratio,data_dim,model_dim,spectral_radius,mean_degree,dt, K_err, omega_err, reg_param=settings[arrayindex,:] 
g=1.0
system=getfield(HybridRCforNLONS,Symbol(system))

if case=="Asynch_Fast"
    Δω=0.05
    K=5.0
end

#we are testing the residual physics by using ground truth from the biharmonic model.
#base parameters are therefore from the standard kuramoto model for the ODE and hybrid expert model.
base_params=cartesian_kuramoto_p(MersenneTwister(1234+ground_truth_case),N,μ,Δω,K)
#create set of modified/innacurate model parameters for the 20 cases (reservoirs or ODE's)
#error distributions to sample from based on this run's setting s
ω_err_dist=Normal(0.0,omega_err)
K_err_dist=Normal(0.0,K_err)

#multiplicative parameter error is still present in the residual physics task (task 2)
modified_params=Vector{Any}(undef,num_instantiations)
for i in 1:num_instantiations
    modified_params[i]=deepcopy(base_params)
    modified_params[i][1:N].*=1.0.+rand(ω_err_dist,N)
    modified_params[i][N+1]*=1.0+rand(K_err_dist)
end

#load ground truth for error calculation
ground_truth=permutedims(Matrix(DataFrame(Arrow.Table(input_path*"Biharmonic_Kuramoto_$(case)_ground_truth_data.arrow.lz4"))))
#transform ground truth to xy components (biharmonic gt data is saved in phase form.)
ground_truth=[phasetoxy(ground_truth[:,i]) for i in 1:size(ground_truth,2)]
ground_truth=reduce(hcat,ground_truth)

train_len=1000
warmup_len=100
test_len=2500
shift=500

#segment the ground truth data into training, warmup and test data. 20 warmup-test spans in total.
training_data=ground_truth[:,1:train_len]
target_data=ground_truth[:,2:train_len+1]
warmup_test_data=ground_truth[:,train_len+train_len+1:end]
test_data=Array{Array{Float64,2},1}(undef,20)
warmup_data=Array{Array{Float64,2},1}(undef,20)
for i in 1:20
    test_data[i]=warmup_test_data[:,shift+1+(test_len+shift)*(i-1):(test_len+shift)+(test_len+shift)*(i-1)]
    warmup_data[i]=warmup_test_data[:,shift+1-warmup_len+(test_len+shift)*(i-1):shift+(test_len+shift)*(i-1)]
end

#save trajectories for inspection? reasonably large storage required. approx 300Gb for 20x2500step tests, 40 reservoirs, 10 oscillators.
save_trajectories=false

#for reservoir initialisation.
reservoir_rng=MersenneTwister(1234+arrayindex)

#for valid time computation
threshold=0.4

#run the tests!
if model_type=="ODE"
    #for each of the parameter combinations, for each of the test spans, start from it's initial condition, and run a simulation until time = 250.0. with saveat=0.1
    dt=0.1
    #20 tests in rows, each column is a reservoir/instance.
    valid_times=Array{Float64,2}(undef,num_tests,num_instantiations)
    println("running tests")
    for test_num in 1:num_tests
        ode_prediction=Array{Float64,2}(undef,test_len,num_instantiations*data_dim)
        for run_num in 1:num_instantiations
            sol=generate_ODE_data_task2(system,test_data[test_num][:,1],modified_params[run_num],(0.0,249.9),1e7,dt)
            sol=permutedims(reduce(hcat,sol.u))
            ode_prediction[:,1+(data_dim*(run_num-1)):data_dim+(data_dim*(run_num-1))]=sol
            valid_times[test_num,run_num]=valid_time(threshold,permutedims(sol),test_data[test_num],dt)
        end
        if save_trajectories
            local name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_predictions_test_$(test_num)_array_index_$(arrayindex)"
            CSV.write(save_path*name*".csv",DataFrame(ode_prediction,:auto),writeheader=true)
            generate_arrow(name,save_path)
            rm(save_path*name*".csv")
        end
        println("test number $(test_num) complete")
        flush(stdout)

    end
    #save the valid times for each reservoir/instance across the 20 tests.
    name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_valid_times_array_index_$(arrayindex)"
    CSV.write(save_path*name*".csv",DataFrame(valid_times,:auto),writeheader=true)

elseif model_type=="Standard"
    #create 40 standard ESN's with the modified parameters and the same reservoir size and connectivity.
    #train each of them on the training data, and then predict the 20 test spans.
    #train all the reservoirs first
    reservoirs=Vector{ESN}()
    println("loading reservoirs")
    for res_idx in 1:num_instantiations
        push!(reservoirs,ESN(res_size,mean_degree,data_dim,spectral_radius,scaling,g,reg_param,sqr_even_indices))
        initialise_reservoir!(reservoir_rng,reservoirs[res_idx])
        ingest_data!(reservoirs[res_idx],training_data)
        train_reservoir!(reservoirs[res_idx],target_data)
    end
    println("reservoirs loaded")
    #20 tests in rows, each column is a reservoir/instance.
    valid_times=Array{Float64,2}(undef,num_tests,num_instantiations)
    println("running tests")
    for test_num in 1:num_tests
        test_prediction=Array{Float64,2}(undef,test_len,num_instantiations*data_dim)
        for run_num in 1:num_instantiations
            ingest_data!(reservoirs[run_num],warmup_data[test_num])
            standard_prediction=predict!(reservoirs[run_num],test_len,false,true)
            test_prediction[:,1+(data_dim*(run_num-1)):data_dim+(data_dim*(run_num-1))]=permutedims(standard_prediction)
            valid_times[test_num,run_num]=valid_time(threshold,standard_prediction,test_data[test_num],dt)
        end
        if save_trajectories
            local name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_predictions_test_$(test_num)_array_index_$(arrayindex)"
            CSV.write(save_path*name*".csv",DataFrame(test_prediction,:auto),writeheader=true)
            generate_arrow(name,save_path)
            rm(save_path*name*".csv")
        end
        println("test number $(test_num) complete")
        flush(stdout)
    end
    #save the valid times for each reservoir/instance across the 20 tests.
    name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_valid_times_array_index_$(arrayindex)"
    CSV.write(save_path*name*".csv",DataFrame(valid_times,:auto),writeheader=true)
# 
elseif model_type=="Hybrid"
    # create 40 hybrid ESN's with the modified parameters and the same reservoir size and connectivity.
    # train each of them on the training data, and then predict the 20 test spans.
    reservoirs=Vector{Hybrid_ESN}()
    println("loading reservoirs")
    for res_idx in 1:num_instantiations
        push!(reservoirs,Hybrid_ESN(res_size,mean_degree,model_dim,data_dim,knowledge_ratio,spectral_radius,scaling,g,reg_param,sqr_even_indices,system,modified_params[res_idx],dt))
        initialise_reservoir!(reservoir_rng,reservoirs[res_idx])
        ingest_data!(reservoirs[res_idx],training_data)
        train_reservoir!(reservoirs[res_idx],target_data)
    end
    println("reservoirs loaded")
    #20 tests in rows, each column is a reservoir/instance.
    valid_times=Array{Float64,2}(undef,num_tests,num_instantiations)
    println("running tests")
    for test_num in 1:num_tests
        test_prediction=Array{Float64,2}(undef,test_len,num_instantiations*data_dim)
        for run_num in 1:num_instantiations
            ingest_data!(reservoirs[run_num],warmup_data[test_num])
            hybrid_prediction=predict!(reservoirs[run_num],test_len,false,true)
            test_prediction[:,1+(data_dim*(run_num-1)):data_dim+(data_dim*(run_num-1))]=permutedims(hybrid_prediction)
            valid_times[test_num,run_num]=valid_time(threshold,hybrid_prediction,test_data[test_num],dt)
        end
        if save_trajectories
            local name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_predictions_test_$(test_num)_array_index_$(arrayindex)"
            CSV.write(save_path*name*".csv",DataFrame(test_prediction,:auto),writeheader=true)
            generate_arrow(name,save_path)
            rm(save_path*name*".csv")
        end
        println("test number $(test_num) complete")
        flush(stdout)
    end
    #save the valid times for each reservoir/instance across the 20 tests.
    name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_valid_times_array_index_$(arrayindex)"
    CSV.write(save_path*name*".csv",DataFrame(valid_times,:auto),writeheader=true)
end
