using Pkg; Pkg.activate(".")
include("$(pwd())/src/HybridRCforNLONS.jl")
using OrdinaryDiffEq, Random, Statistics, Distributions, LinearAlgebra, CSV, Arrow, DataFrames, DelimitedFiles
import HybridRCforNLONS: cartesian_kuramoto, cartesian_kuramoto_p, normalised_error, generate_ODE_data, generate_arrow, ESN, Hybrid_ESN, train_reservoir!, predict!, ingest_data!, initialise_reservoir!, valid_time,phasetoxy,sqr_even_indices

arrayindex=parse(Int,ARGS[1]) #where in the grid search? (1-8)
# arrayindex=1

psweep_name="GridSearch"

ground_truth_case=parse(Int64,ARGS[2]) # 1.Synch, 2.Asynch, 3.Heteroclinic, 4.SCPS
# ground_truth_case=3 # 1.Synch, 3.Heteroclinic, 4.SCPS (2.asynch not used)

# model_type=ARGS[3] #Hybrid, Standard.
model_type="Standard"

input_path="$(pwd())/Residual_Physics_Task/Settings_and_GroundTruth/"
# input_path=ARGS[4] #where the settings and ground truth data is stored.

output_path="$(pwd())/Residual_Physics_Task/"
# output_path=ARGS[5] #path to parent location to store output. GridSearch folder will be created here.

save_path=output_path*psweep_name*"/"
if isdir(save_path)
    println("Directory $(psweep_name) exists")
else
    mkdir(save_path)
end

num_instantiations=40 #number of reservoirs to create and test

num_tests=20 #number of test spans to predict. maximum 20, as ground truth is split into 20 warmup-test segments.
cases=["Synch","Asynch","HeteroclinicCycles","SelfConsistentPartialSynchrony"]

case=cases[ground_truth_case]
γ_1s=[2*Float64(pi),Float64(pi),1.3,1.5]
γ_1=γ_1s[ground_truth_case]
γ_2=Float64(pi)
a=0.2

settings=readdlm(input_path*psweep_name*"_sweep_settings.csv",',',header=true)[1]
N,K,system,μ,Δω,res_size,scaling,knowledge_ratio,data_dim,model_dim,spectral_radius,mean_degree,dt, K_err, omega_err, reg_param=settings[arrayindex,:] 
g=1.0
system=getfield(HybridRCforNLONS,Symbol(system))

#base parameters are for the standard kuramoto model, as this is used by the hybrid expert model and ODE model.
base_params=cartesian_kuramoto_p(MersenneTwister(1234+ground_truth_case),N,μ,Δω,K)
#create set of modified/innacurate model parameters for the 40 cases (reservoirs or ODE's)
#error distributions to sample from based on this run's settings
ω_err_dist=Normal(0.0,omega_err)
K_err_dist=Normal(0.0,K_err)

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

reservoir_rng=MersenneTwister(1234+arrayindex)

#save trajectores for inspection or other error metric computation?
save_trajectories=false

#threhold for valid time calculation
threshold=0.4

#run the tests!
if model_type=="ODE" #kept from parameter sweep if needed, but ODE was not tested on the grid search in the paper.
    #for each of the parameter combinations, for each of the test spans, start from it's initial condition, and run a simulation until time = 250.0. with saveat=0.1
    dt=0.1
    #20 tests in rows, each column is a reservoir/instance.
    valid_times_per_test_per_reservoir=Array{Float64,2}(undef,num_tests,num_instantiations)
    for test_num in 1:num_tests
        println("model_type: $model_type test: ",test_num)
        ode_prediction=Array{Float64,2}(undef,test_len,num_instantiations*data_dim)
        for run_num in 1:num_instantiations
            sol=generate_ODE_data(system,test_data[test_num][:,1],modified_params[run_num],(0.0,249.9),1e7,dt)
            sol=permutedims(reduce(hcat,sol.u))
            ode_prediction[:,1+(data_dim*(run_num-1)):data_dim+(data_dim*(run_num-1))]=sol
            valid_times_per_test_per_reservoir[test_num,run_num]=valid_time(threshold,permutedims(sol),test_data[test_num],dt)
        end
        if save_trajectories
            local name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_predictions_test_$(test_num)_array_index_$(arrayindex)"
            CSV.write(save_path*name*".csv",DataFrame(ode_prediction,:auto),writeheader=true)
            generate_arrow(name,save_path)
            rm(save_path*name*".csv")
        end
    end
    #save the valid times for each reservoir/instance across the 20 tests. (rows=tests, columns=reservoirs)
    name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_valid_times_array_index_$(arrayindex)"
    CSV.write(save_path*name*".csv",DataFrame(valid_times_per_test_per_reservoir,:auto),writeheader=true)

elseif model_type=="Standard"
    #create 20 standard ESN's with the modified parameters and the same reservoir size and connectivity.
    #train each of them on the training data, and then predict the 20 test spans. save the predictions as csv's.
    #train all the reservoirs first
    reservoirs=Vector{ESN}()
    for res_idx in 1:num_instantiations
        push!(reservoirs,ESN(res_size,mean_degree,data_dim,spectral_radius,scaling,g,reg_param,sqr_even_indices))
        initialise_reservoir!(reservoir_rng,reservoirs[res_idx])
        ingest_data!(reservoirs[res_idx],training_data)
        train_reservoir!(reservoirs[res_idx],target_data)
    end
    #20 tests in rows, each column is a reservoir/instance.
    valid_times_per_test_per_reservoir=Array{Float64,2}(undef,num_tests,num_instantiations)
    for test_num in 1:num_tests
        println("model_type: $model_type test: ",test_num)
        test_prediction=Array{Float64,2}(undef,test_len,num_instantiations*data_dim)
        for run_num in 1:num_instantiations
            ingest_data!(reservoirs[run_num],warmup_data[test_num])
            standard_prediction=predict!(reservoirs[run_num],test_len,false,true)
            test_prediction[:,1+(data_dim*(run_num-1)):data_dim+(data_dim*(run_num-1))]=permutedims(standard_prediction)
            valid_times_per_test_per_reservoir[test_num,run_num]=valid_time(threshold,standard_prediction,test_data[test_num],dt)
        end
        if save_trajectories
            local name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_predictions_test_$(test_num)_array_index_$(arrayindex)"
            CSV.write(save_path*name*".csv",DataFrame(test_prediction,:auto),writeheader=true)
            generate_arrow(name,save_path)
            rm(save_path*name*".csv")
        end
    end
    #save the valid times for each reservoir/instance across the 20 tests. (rows=tests, columns=reservoirs)
    name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_valid_times_array_index_$(arrayindex)"
    CSV.write(save_path*name*".csv",DataFrame(valid_times_per_test_per_reservoir,:auto),writeheader=true)
# 
elseif model_type=="Hybrid"
    # create 20 hybrid ESN's with the modified parameters and the same reservoir size and connectivity.
    # train each of them on the training data, and then predict the 20 test spans. save the predictions as csv's.
    reservoirs=Vector{Hybrid_ESN}()
    for res_idx in 1:num_instantiations
        push!(reservoirs,Hybrid_ESN(res_size,mean_degree,model_dim,data_dim,knowledge_ratio,spectral_radius,scaling,g,reg_param,sqr_even_indices,system,modified_params[res_idx],dt))
        initialise_reservoir!(reservoir_rng,reservoirs[res_idx])
        ingest_data!(reservoirs[res_idx],training_data)
        train_reservoir!(reservoirs[res_idx],target_data)
    end
    #20 tests in rows, each column is a reservoir/instance.
    valid_times_per_test_per_reservoir=Array{Float64,2}(undef,num_tests,num_instantiations)
    for test_num in 1:num_tests
        println("model_type: $model_type test: ",test_num)
        test_prediction=Array{Float64,2}(undef,test_len,num_instantiations*data_dim)
        for run_num in 1:num_instantiations
            ingest_data!(reservoirs[run_num],warmup_data[test_num])
            hybrid_prediction=predict!(reservoirs[run_num],test_len,false,true)
            test_prediction[:,1+(data_dim*(run_num-1)):data_dim+(data_dim*(run_num-1))]=permutedims(hybrid_prediction)
            valid_times_per_test_per_reservoir[test_num,run_num]=valid_time(threshold,hybrid_prediction,test_data[test_num],dt)
        end
        if save_trajectories
            local name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_predictions_test_$(test_num)_array_index_$(arrayindex)"
            CSV.write(save_path*name*".csv",DataFrame(test_prediction,:auto),writeheader=true)
            generate_arrow(name,save_path)
            rm(save_path*name*".csv")
        end
    end
    #save the valid times for each reservoir/instance across the 20 tests. (rows=tests, columns=reservoirs)
    name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_valid_times_array_index_$(arrayindex)"
    CSV.write(save_path*name*".csv",DataFrame(valid_times_per_test_per_reservoir,:auto),writeheader=true)
end