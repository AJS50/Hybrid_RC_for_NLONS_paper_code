using Pkg; Pkg.activate(".")
include("$(pwd())/src/HybridRCforNLONS.jl")
using OrdinaryDiffEq, Random, Statistics, Distributions, LinearAlgebra, CSV, Arrow, DataFrames, DelimitedFiles
import HybridRCforNLONS: cartesian_kuramoto, cartesian_kuramoto_p, normalised_error, generate_ODE_data_task1, generate_arrow, ESN, Hybrid_ESN, train_reservoir!, predict!, ingest_data!, initialise_reservoir!,sqr_even_indices

#The parameter sweeps were run using SLURM array jobs, with arrayindex being the ${SLURM_ARRAY_TASK_ID} from the job.

#read in command line arguments
# arrayindex=10
arrayindex=parse(Int64,ARGS[1]) #where in the parameter sweep are we?  (1-20)
psweep_name="SpectralRadius"
# psweep_name=ARGS[2] #read in name of parameter sweep to be run. this will be used to load in the appropriate csv's with parameter settings. See settings file titles for correct naming.
base_model=7
# base_model=parse(Int64,ARGS[3]) #which of the three regimes is used as ground truth and expert ODE model ? #Model2=asynchronous, Model7=multi-frequency, Model16=synchronous
input_path="$(pwd())/Parameter_Error_Task/Settings_and_GroundTruth/"
# input_path=ARGS[4] #location of csv's with parameter settings and ground truth data.
output_path="$(pwd())/Parameter_Error_Task/"
# output_path=ARGS[5] #path to store results in.
# eval_type="Standard"
eval_type=ARGS[2] #read in type of evaluation to be done. "ODE" or "Standard" or "Hybrid" (to allow parallisation across jobs.) each will regenerate the ground truth for now, but will see how much better Arrow is for storage and consider saving it.
num_reservoirs=40
# num_reservoirs=parse(Int64,ARGS[7]) #how many reservoir or ODE instantiations to test. reduce to run quick test.
num_tests=20 #number of test spans to predict, maximum 20 as the ground truth data is always the same. reduce to run quick test.

output_path=output_path*"/"*psweep_name*"/"
if !isdir(output_path)
    mkdir(output_path)
end


#save trajectories? will require large storage for all of the task 1 parameter sweep trajectories. Conservatively ~700GB.
save_trajectories=false
#still want array index dependent random seed for reservoir initialisation. but not for base parameters as they are expert model/ground truth specific
model_names=["asynchronous","multi-frequency","synchronous"]
for j in 1:3 #loop over all three instantiations of the model regime.
    rng=MersenneTwister(1234+arrayindex) # for reservoir initialisation
    settings=readdlm(input_path*psweep_name*"_sweep_settings.csv",',',header=true)[1]
    N,K_,model,μ,Δω,res_size,scaling,knowledge_ratio,data_dim,model_dim,spectral_radius,mean_degree,dt, K_err, omega_err=settings[arrayindex,:] 
    model=getfield(HybridRCforNLONS,Symbol(model))
    reg_param=0.000001
    g=1.0
    #using Model (base_model) set the natural frequencies, and K.
    Ks=[1.0,2.0,4.0]
    param_seed=1234+base_model+j-1
    param_rng=MersenneTwister(param_seed)
    if base_model==2
        K=Ks[1]
        models=["2","2_1","2_2"] #to label particular instance of the regime. 
        idx=1 #to select model regime name when saving.
    elseif base_model==7
        K=Ks[2]
        models=["7","7_1","7_2"]
        idx=2
    elseif base_model==16
        K=Ks[3]
        models=["16","16_1","16_2"]
        idx=3
    end

    base_sub_model=models[j]
    if j==1 #first regime instance was found from the array index rng parameters
        base_params=cartesian_kuramoto_p(param_rng,N,μ,Δω,K)
    else #the other two sampled from manually specified distributions.
        base_params=cartesian_kuramoto_p(param_rng,N,μ,Δω,K)
        if base_model == 2 # - Model 2: Nat Freqs within [-1,+1], K=1.0 (asynch)
            base_params[1:5]=2.0.*rand(param_rng,5).-1.0
            base_params[6]=1.0
        elseif base_model == 7 # - Model 7: 4 Nat Freqs within [-1,+1], with outlier >3Hz.  K=2.0, (asynch with high frequency)
            base_params[1:4]=2.0.*rand(param_rng,4).-1.0
            x=rand()-0.5
            base_params[5]=(rand(param_rng).+3.0)*sign(x)
            base_params[6]=2.0
        elseif base_model == 16 # - Model 16: 4 Nat Freqs within [-1,+1] K=4.0 (synchronised)
            base_params[1:5]=2.0.*rand(param_rng,5).-1.0
            base_params[6]=4.0
        end
    end

    println("base_params_$(j): ",base_params)

    ground_truth=permutedims(Matrix(DataFrame(Arrow.Table(input_path*"Model_$(model_names[idx])_$(j-1)_ground_truth_data.arrow.lz4"))))

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

    #create set of modified/innacurate model parameters for the 20 cases (reservoirs or ODE's)
    #create error distributions to sample from based on this run's settings
    ω_err_dist=Normal(0.0,omega_err)
    K_err_dist=Normal(0.0,K_err)
    #multiplicative error. new parameters stored in modified_params to be used for each hybrid reservoir/DOE instance.
    modified_params=Vector{Any}(undef,num_reservoirs)
    for i in 1:num_reservoirs
        modified_params[i]=deepcopy(base_params)
        modified_params[i][1:N].*=1.0.+rand(ω_err_dist,N)
        modified_params[i][N+1]*=1.0+rand(K_err_dist)
    end

    #run the parameter sweep for the given 'eval type' ODE, Standard ESN or Hybrid ESN.

    if eval_type=="ODE"
        #for each of the parameter combinations, for each of the test spans, start from it's initial condition, and run a simulation until time = 250.0. with saveat=0.1 (save interval)
        #hcat the output solution.u and save it as a csv.
        dt=0.1
        #20 tests in rows, each column is a reservoir/instance.
        norm_errors_per_test_per_reservoir=Array{Float64,2}(undef,num_tests,num_reservoirs)
        for test_num in 1:num_tests
            ode_prediction=Array{Float64,2}(undef,test_len,num_reservoirs*data_dim)
            for run_num in 1:num_reservoirs
                sol=generate_ODE_data_task1(cartesian_kuramoto,test_data[test_num][:,1],modified_params[run_num],(0.0,249.9),1e7,dt)
                sol=permutedims(reduce(hcat,sol.u))
                ode_prediction[:,1+(data_dim*(run_num-1)):data_dim+(data_dim*(run_num-1))]=sol
                norm_errors_per_test_per_reservoir[test_num,run_num]=mean(normalised_error(permutedims(sol),test_data[test_num]))

            end
            #save the trajectory in an arrow file if required. (will create roughly 600GB for all of the task 1 parameter sweep data)
            if save_trajectories
                local name=psweep_name*"_task1_$(model_names[idx])_$(j-1)"*"_"*eval_type*"predictions_test_$(test_num)_array_index_$(arrayindex)"
                CSV.write(output_path*name*".csv",DataFrame(ode_prediction,:auto),writeheader=true)
                generate_arrow(name,output_path)
                rm(output_path*name*".csv")
            end
        end
        #save the normalised errors for each reservoir/instance across the 20 tests in a csv.
        name=psweep_name*"_task1_$(model_names[idx])_$(j-1)"*"_"*eval_type*"norm_errors_array_index_$(arrayindex)"
        CSV.write(output_path*name*".csv",DataFrame(norm_errors_per_test_per_reservoir,:auto),writeheader=true)

    elseif eval_type=="Standard"
        #create 40 standard ESN's with the modified parameters and the same reservoir size and connectivity.
        #train each of them on the training data, and then predict the 20 test spans. save the predictions as csv's.
        #train all the reservoirs first
        reservoirs=Vector{ESN}()
        for res_idx in 1:num_reservoirs
            push!(reservoirs,ESN(res_size,mean_degree,data_dim,spectral_radius,scaling,g,reg_param,sqr_even_indices))
            initialise_reservoir!(rng,reservoirs[res_idx])
            ingest_data!(reservoirs[res_idx],training_data)
            train_reservoir!(reservoirs[res_idx],target_data)
        end
        #20 tests in rows, each column is a reservoir/instance.
        norm_errors_per_test_per_reservoir=Array{Float64,2}(undef,num_tests,num_reservoirs)
        for test_num in 1:num_tests
            test_prediction=Array{Float64,2}(undef,test_len,num_reservoirs*data_dim)
            for run_num in 1:num_reservoirs
                ingest_data!(reservoirs[run_num],warmup_data[test_num])
                standard_prediction=predict!(reservoirs[run_num],test_len,false,true)
                test_prediction[:,1+(data_dim*(run_num-1)):data_dim+(data_dim*(run_num-1))]=permutedims(standard_prediction)
                norm_errors_per_test_per_reservoir[test_num,run_num]=mean(normalised_error(standard_prediction,test_data[test_num]))
            end
            if save_trajectories
                local name=psweep_name*"_task1_$(model_names[idx])_$(j-1)"*"_"*eval_type*"predictions_test_$(test_num)_array_index_$(arrayindex)"
                CSV.write(output_path*name*".csv",DataFrame(test_prediction,:auto),writeheader=true)
                generate_arrow(name,output_path)
                rm(output_path*name*".csv")
            end
        end
        #save the normalised errors for each reservoir/instance across the 20 tests.
        name=psweep_name*"_task1_$(model_names[idx])_$(j-1)"*"_"*eval_type*"norm_errors_array_index_$(arrayindex)"
        CSV.write(output_path*name*".csv",DataFrame(norm_errors_per_test_per_reservoir,:auto),writeheader=true)
    # 
    elseif eval_type=="Hybrid"
        # create 40 hybrid ESN's with the modified parameters and the same reservoir size and connectivity.
        # train each of them on the training data, and then predict the 20 test spans. save the predictions as csv's.
        reservoirs=Vector{Hybrid_ESN}()
        for res_idx in 1:num_reservoirs
            push!(reservoirs,Hybrid_ESN(res_size,mean_degree,model_dim,data_dim,knowledge_ratio,spectral_radius,scaling,g,reg_param,sqr_even_indices,model,modified_params[res_idx],dt))
            initialise_reservoir!(rng,reservoirs[res_idx])
            ingest_data!(reservoirs[res_idx],training_data)
            train_reservoir!(reservoirs[res_idx],target_data)
        end
        #20 tests in rows, each column is a reservoir/instance.
        norm_errors_per_test_per_reservoir=Array{Float64,2}(undef,num_tests,num_reservoirs)
        for test_num in 1:num_tests
            test_prediction=Array{Float64,2}(undef,test_len,num_reservoirs*data_dim)
            for run_num in 1:num_reservoirs
                ingest_data!(reservoirs[run_num],warmup_data[test_num])
                hybrid_prediction=predict!(reservoirs[run_num],test_len,false,true)
                test_prediction[:,1+(data_dim*(run_num-1)):data_dim+(data_dim*(run_num-1))]=permutedims(hybrid_prediction)
                norm_errors_per_test_per_reservoir[test_num,run_num]=mean(normalised_error(hybrid_prediction,test_data[test_num]))
            end
            if save_trajectories
                local name=psweep_name*"_task1_$(model_names[idx])_$(j-1)"*"_"*eval_type*"predictions_test_$(test_num)_array_index_$(arrayindex)"
                CSV.write(output_path*name*".csv",DataFrame(test_prediction,:auto),writeheader=true)
                generate_arrow(name,output_path)
                rm(output_path*name*".csv")
            end
        end
        #save the normalised errors for each reservoir/instance across the 20 tests.
        name=psweep_name*"_task1_$(model_names[idx])_$(j-1)"*"_"*eval_type*"norm_errors_array_index_$(arrayindex)"
        CSV.write(output_path*name*".csv",DataFrame(norm_errors_per_test_per_reservoir,:auto),writeheader=true)
    end
end
