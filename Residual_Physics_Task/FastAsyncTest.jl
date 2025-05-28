using Pkg; Pkg.activate(".")
include("$(pwd())/src/HybridRCforNLONS.jl")
using OrdinaryDiffEq, Random, Statistics, Distributions, LinearAlgebra, Plots, DataFrames, DelimitedFiles, CSV, Arrow
import .HybridRCforNLONS: cartesian_kuramoto, cartesian_kuramoto_p, normalised_error, generate_ODE_data_task2, ESN, Hybrid_ESN, train_reservoir!, predict!, ingest_data!, initialise_reservoir!, valid_time, sqr_even_indices, valid_time
import .HybridRCforNLONS: biharmonic_kuramoto_p, biharmonic_kuramoto, biharmonic_kuramoto_ic, reset_condition1, reset_affect1!, reset_condition2, reset_affect2!, generate_arrow, xytophase,phasetoxy, kuramoto_order2
plotlyjs()
arrayindex=parse(Int,ARGS[1]) #where in the parameter sweep are we? (1-20)
arrayindex=20

psweep_name=ARGS[2] #to select parameter settings according to the settings csv files. See settings files names for correct names.
psweep_name="SpectralRadius"

ground_truth_case=parse(Int64,ARGS[3]) # regimes: 1.Synch, 2.Asynch, 3.Heteroclinic, 4.SCPS
ground_truth_case=5

model_type=ARGS[4] # ODE, Hybrid, Standard.
model_type="Hybrid"# ODE, Hybrid, Standard.

num_instantiations=10 #how many reservoir or ODE instantiations to test. reduce for quick tests.
# num_instantiations=ARGS[5]

num_tests=5 #how many test spans to predict. maximum 20, as ground truth is always split into 20 warmup-test segments.
# num_tests=ARGS[6]

input_path="$(pwd())/Residual_Physics_Task/Settings_and_GroundTruth/"
# input_path=ARGS[7] #path to settings and ground truth files

output_path="$(pwd())/Residual_Physics_Task/"
# output_path=ARGS[8] #path to parent folder to store output valid times and trajectories. Will generate subfolders for each parameter.

save_path=output_path*psweep_name*"/"
if isdir(save_path)
    println("Directory $(psweep_name) exists")
else
    mkdir(save_path)
end


## Creating faster Asynchronous model by increasing the omegas (i.e. increasing Δ, and increasing K by the same factor. Effectively scaling time)
# Try 5 times faster.
settings=readdlm(input_path*psweep_name*"_sweep_settings.csv",',',header=true)[1]
N,K,system,μ,Δω,res_size,scaling,knowledge_ratio,data_dim,model_dim,spectral_radius,mean_degree,dt, K_err, omega_err, reg_param=settings[arrayindex,:] 
g=1.0
system=getfield(HybridRCforNLONS,Symbol(system))
Δω
K
μ
#faster async:
# Δω=0.05
# K=5.0

cases=["Synch","Asynch","HeteroclinicCycles","SelfConsistentPartialSynchrony","Asynch_Fast"]
case=cases[ground_truth_case]
γ_1s=[2*Float64(pi),Float64(pi),1.3,1.5,Float64(pi)]
γ_1=γ_1s[ground_truth_case]
γ_2=Float64(pi)
a=0.2

cidx=ground_truth_case
case=cases[cidx]

callback=CallbackSet(
    VectorContinuousCallback(reset_condition1,reset_affect1!,N),
    VectorContinuousCallback(reset_condition2,reset_affect2!,N)
    )

if case=="Asynch_Fast"
    Δω=0.05
    K=5.0
end

rng=MersenneTwister(1234+cidx)
base_params=biharmonic_kuramoto_p(rng,N,μ,Δω,K,a,γ_1s[ground_truth_case],γ_2)
ic=biharmonic_kuramoto_ic(N) #same initial conditions for every run (internally this uses a MersenneTwister rng with seed 1234)
tspan=(0.0,6200.0)
# prob=ODEProblem(biharmonic_kuramoto,ic,tspan,base_params;callback=callback)
# gt_data=permutedims(reduce(hcat,solve(prob,Tsit5(),dtmax=1/32,adaptive=true,saveat=dt).u))
# import .HybridRCforNLONS: phasetoxy
# ground_truth=[phasetoxy(gt_data'[:,i]) for i in 1:size(gt_data',2)]
# ground_truth=reduce(hcat,ground_truth)

#save the gt data as the correct ground truth:
# name=psweep_name*"_Biharmonic_Kuramoto_$(case)_ground_truth_data"
# CSV.write(save_path*name*".csv",DataFrame(gt_data,:auto),writeheader=true)
# generate_arrow(name,save_path)
# rm(save_path*name*".csv")
#instead load the bad gt from the HPC.
ground_truth=Matrix(DataFrame(Arrow.Table("/Users/as15635/Documents/Projects/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/Settings_and_GroundTruth/Biharmonic_Kuramoto_Asynch_Fast_ground_truth_data.arrow.lz4")))
ground_truth=[phasetoxy(ground_truth'[:,i]) for i in 1:size(ground_truth',2)]
ground_truth=reduce(hcat,ground_truth)

# plot(gt_data[1:1000,1:10])
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
# plot(training_data[1:10,:]')
# plot(test_data[1][1:10,:]')
# plot(1:100,warmup_data[1][1:10,:]')
# plot!(101:2600,test_data[1][1:10,:]',color=:purple)
# plot!(xlims=(80,120))


#we are testing the residual physics by using ground truth from the biharmonic model.
#base parameters are therefore from the standard kuramoto model for the ODE and hybrid expert model.
base_params=cartesian_kuramoto_p(MersenneTwister(1234+ground_truth_case),N,μ,Δω,K)
#create set of modified/innacurate model parameters for the 20 cases (reservoirs or ODE's)
#error distributions to sample from based on this run's settings
ω_err_dist=Normal(0.0,omega_err)
K_err_dist=Normal(0.0,K_err)

#multiplicative parameter error is still present in the residual physics task (task 2)
modified_params=Vector{Any}(undef,num_instantiations)
for i in 1:num_instantiations
    modified_params[i]=deepcopy(base_params)
    modified_params[i][1:N].*=1.0.+rand(ω_err_dist,N)
    modified_params[i][N+1]*=1.0+rand(K_err_dist)
end



#save trajectories for inspection? reasonably large storage required. approx 300Gb for 20x2500step tests, 40 reservoirs, 10 oscillators.
save_trajectories=true

#for reservoir initialisation.
reservoir_rng=MersenneTwister(1234+arrayindex)

#for valid time computation
threshold=0.4

for model_type in ["ODE","Standard","Hybrid"]
#run the tests!
if model_type=="ODE"
    #for each of the parameter combinations, for each of the test spans, start from it's initial condition, and run a simulation until time = 250.0. with saveat=0.1
    dt=0.1
    @show "doing ODE"
    #20 tests in rows, each column is a reservoir/instance.
    valid_times=Array{Float64,2}(undef,num_tests,num_instantiations)
    for test_num in 1:num_tests
        ode_prediction=Array{Float64,2}(undef,test_len,num_instantiations*data_dim)
        for run_num in 1:num_instantiations
            @show test_data[test_num][:,1]
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
    end
    #save the valid times for each reservoir/instance across the 20 tests.
    name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_valid_times_array_index_$(arrayindex)"
    CSV.write(save_path*name*".csv",DataFrame(valid_times,:auto),writeheader=true)

elseif model_type=="Standard"
    #create 40 standard ESN's with the modified parameters and the same reservoir size and connectivity.
    #train each of them on the training data, and then predict the 20 test spans.
    #train all the reservoirs first
    reservoirs=Vector{ESN}()
    for res_idx in 1:num_instantiations
        push!(reservoirs,ESN(res_size,mean_degree,data_dim,spectral_radius,scaling,g,reg_param,HybridRCforNLONS.sqr_even_indices))
        initialise_reservoir!(reservoir_rng,reservoirs[res_idx])
        ingest_data!(reservoirs[res_idx],training_data)
        train_reservoir!(reservoirs[res_idx],target_data)
    end
    #20 tests in rows, each column is a reservoir/instance.
    valid_times=Array{Float64,2}(undef,num_tests,num_instantiations)
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
    end
    #save the valid times for each reservoir/instance across the 20 tests.
    name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_valid_times_array_index_$(arrayindex)"
    CSV.write(save_path*name*".csv",DataFrame(valid_times,:auto),writeheader=true)
# 
elseif model_type=="Hybrid"
    # create 40 hybrid ESN's with the modified parameters and the same reservoir size and connectivity.
    # train each of them on the training data, and then predict the 20 test spans.
    @show "doing hybrid"
    reservoirs=Vector{Hybrid_ESN}()
    for res_idx in 1:num_instantiations
        push!(reservoirs,Hybrid_ESN(res_size,mean_degree,model_dim,data_dim,knowledge_ratio,spectral_radius,scaling,g,reg_param,HybridRCforNLONS.sqr_even_indices,system,modified_params[res_idx],dt))
        initialise_reservoir!(reservoir_rng,reservoirs[res_idx])
        ingest_data!(reservoirs[res_idx],training_data)
        train_reservoir!(reservoirs[res_idx],target_data)
    end
    #20 tests in rows, each column is a reservoir/instance.
    valid_times=Array{Float64,2}(undef,num_tests,num_instantiations)
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
    end
    #save the valid times for each reservoir/instance across the 20 tests.
    name=psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_valid_times_array_index_$(arrayindex)"
    CSV.write(save_path*name*".csv",DataFrame(valid_times,:auto),writeheader=true)
end
end 


#read in the data and plot the hybrid prediction against the ground truth 
arrayindex=10
instance_number=10
test_num=5
case=cases[5]
cols=["black","blue","red"]
to_plot_for_particular_regime=Vector{Any}()
for (midx,model_type) in enumerate(["ODE","Standard","Hybrid"])
    case
    traj=Matrix(DataFrame(Arrow.Table(save_path*psweep_name*"_"*model_type*"_Biharmonic_Kuramoto_$(case)_predictions_test_$(test_num)_array_index_$(arrayindex).arrow.lz4")))
    test_traj=test_data[test_num]'
    p=plot(test_traj[:,1:10],label="",color=:purple);
    plot!(p,traj[:,1+20*(instance_number-1):10+20*(instance_number-1)],color=cols[midx],xlims=(0,100),size=(800,400),legend=nothing,title=model_type,label="");
    plot!(p,xlims=(0,1000))
    push!(to_plot_for_particular_regime,p)
end




bigplot=plot(to_plot_for_particular_regime...,layout=(2,2),plot_title="SpectralRadius - index $(arrayindex)")
width,height=bigplot.attr[:size]
Plots.prepare_output(bigplot)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(bigplot),"/Users/as15635/Documents/Projects/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/Figures/$(case)_trajectory_$(psweep_name)_instance_$(instance_number)_test_$(test_num)_array_index_$(arrayindex)_WrongGTonHPC.pdf",width=width,height=height)


#checking kuramoto order parameter of each in the fast asynch case and the normal async case to see if that is a good metric.
async_slow_hybrid_trajectory=Matrix(DataFrame(Arrow.Table(save_path*psweep_name*"_Hybrid_Biharmonic_Kuramoto_Asynch_predictions_test_1_array_index_$(arrayindex).arrow.lz4")))[:,1:20]
async_fast_hybrid_trajectory=Matrix(DataFrame(Arrow.Table(save_path*psweep_name*"_Hybrid_Biharmonic_Kuramoto_Asynch_Fast_predictions_test_1_array_index_$(arrayindex).arrow.lz4")))[:,1:20]
async_slow_ode_trajectory=Matrix(DataFrame(Arrow.Table(save_path*psweep_name*"_ODE_Biharmonic_Kuramoto_Asynch_predictions_test_1_array_index_$(arrayindex).arrow.lz4")))[:,1:20]
async_fast_ode_trajectory=Matrix(DataFrame(Arrow.Table(save_path*psweep_name*"_ODE_Biharmonic_Kuramoto_Asynch_Fast_predictions_test_1_array_index_$(arrayindex).arrow.lz4")))[:,1:20]
async_slow_standard_trajectory=Matrix(DataFrame(Arrow.Table(save_path*psweep_name*"_Standard_Biharmonic_Kuramoto_Asynch_predictions_test_1_array_index_$(arrayindex).arrow.lz4")))[:,1:20]
async_fast_standard_trajectory=Matrix(DataFrame(Arrow.Table(save_path*psweep_name*"_Standard_Biharmonic_Kuramoto_Asynch_Fast_predictions_test_1_array_index_$(arrayindex).arrow.lz4")))[:,1:20]

# async_slow_ground_truth=test_data[1];
async_slow_ground_truth

# async_fast_ground_truth=test_data[1]
async_fast_ground_truth

async_slow_hybrid_order=kuramoto_order2(xytophase(async_slow_hybrid_trajectory'),10)[1,:]
async_fast_hybrid_order=kuramoto_order2(xytophase(async_fast_hybrid_trajectory'),10)[1,:]
async_slow_ode_order=kuramoto_order2(xytophase(async_slow_ode_trajectory'),10)[1,:]
async_fast_ode_order=kuramoto_order2(xytophase(async_fast_ode_trajectory'),10)[1,:]
async_slow_standard_order=kuramoto_order2(xytophase(async_slow_standard_trajectory'),10)[1,:]
async_fast_standard_order=kuramoto_order2(xytophase(async_fast_standard_trajectory'),10)[1,:]
async_slow_gt_order=kuramoto_order2(xytophase(async_slow_ground_truth),10)[1,:]
async_fast_gt_order=kuramoto_order2(xytophase(async_fast_ground_truth),10)[1,:]

p=plot(abs.(async_slow_gt_order),label="Ground Truth Slow",color=:purple)
plot!(p,abs.(async_slow_hybrid_order),label="Hybrid Slow",color=:red)
plot!(p,abs.(async_slow_ode_order),label="ODE Slow",color=:black)
plot!(p,abs.(async_slow_standard_order),label="Standard Slow",color=:blue)
plot!(p,ylabel="Synchrony",xlabel="Time Step",title="Asynchronous Slow")

#plot and save the figure.
width,height=p.attr[:size]
Plots.prepare_output(p)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),"$(pwd())/Slow_Asynchronous_Order_$(psweep_name)_instance_$(instance_number)_test_$(test_num)_array_index_$(arrayindex).pdf",width=width,height=height)


p=plot(abs.(async_fast_gt_order),label="Ground Truth Fast",color=:purple)
plot!(p,abs.(async_fast_standard_order),label="Standard Fast",color=:blue)
plot!(p,abs.(async_fast_hybrid_order),label="Hybrid Fast",color=:red)
plot!(p,abs.(async_fast_ode_order),label="ODE Fast",color=:black)
plot!(p,ylabel="Synchrony",xlabel="Time Step",title="Asynchronous Fast")

width,height=p.attr[:size]
Plots.prepare_output(p)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),"$(pwd())/Fast_Asynchronous_Order_$(psweep_name)_instance_$(instance_number)_test_$(test_num)_array_index_$(arrayindex).pdf",width=width,height=height)