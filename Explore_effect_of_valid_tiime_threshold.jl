using Pkg; Pkg.activate(".")
include("$(pwd())/src/HybridRCforNLONS.jl")
using Statistics, LinearAlgebra, Plots, DataFrames, DelimitedFiles, CSV, Arrow
import .HybridRCforNLONS: valid_time, sqr_even_indices, valid_time
using ProgressLogging
plotlyjs()

#load in the trajectory data for all three modes and ground truth for a given psweep, arrindex.
#The data is all tests and reservoirs
gt_input_path="/user/home/as15635/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/Settings_and_GroundTruth/"
traj_input_path="/user/work/as15635/output_data/ExtKuramoto/"
output_path="/user/work/as15635/output_data/ExtKuramoto/VT_threshold_sweep/"

regimes=["Asynch_Fast"]#,"Synch","Asynch_Fast","HeteroclinicCycles","SelfConsistentPartialSynchrony"]

parameter="SpectralRadius"
test_range=1:20
instantiation_range=1:40
num_instantiations=instantiation_range[end]
arrayindex=9
train_len=1000
warmup_len=100
test_len=2500
shift=500
data_dimension=20
dt=0.1
models=["ODE","Hybrid","Standard"]

thresholds=collect(0.25:0.25:3.5)
trajectory_store=Array{Float64}(undef,test_len,data_dimension*num_instantiations)

per_threshold_test_wise_valid_times_per_regime=Array{Float64}(undef,length(regimes),length(thresholds),length(models),length(test_range),num_instantiations)
@progress for (r_idx,regime) in enumerate(regimes)
    #load ground truth in xy space
    ground_truth=permutedims(Matrix(DataFrame(Arrow.Table(gt_input_path*"Biharmonic_Kuramoto_$(regime)_ground_truth_data.arrow.lz4"))))
    ground_truth=[phasetoxy(ground_truth[:,i]) for i in 1:size(ground_truth,2)]
    ground_truth=reduce(hcat,ground_truth)

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
    

    for (t_idx,threshold) in enumerate(thresholds)
        for (m_idx,model) in enumerate(models)
            # for each test at this array index, load the trajectories of the instances, and compute the valid time.
            for test_num in test_range
                if regime!="Asynch_Fast"
                    trajectory_store=Matrix(DataFrame(Arrow.Table(traj_input_path*parameter*"/$(parameter)"*model*"_ExtKuramoto_"*regime*"_predictions_test_$(test_num)_arrayindex_$(arrayindex).arrow.lz4")))
                else
                    trajectory_store=Matrix(DataFrame(Arrow.Table(traj_input_path*parameter*"/$(parameter)"*"_"*model*"_Biharmonic_Kuramoto_"*regime*"_predictions_test_$(test_num)_array_index_$(arrayindex).arrow.lz4")))
                end
                for instantiation in instantiation_range
                    trajectory=trajectory_store[:,1+20*(instantiation-1):20+20*(instantiation-1)]
                    # @show size(trajectory)
                    # @show size(test_data[test_num])
                    per_threshold_test_wise_valid_times_per_regime[r_idx,t_idx,m_idx,test_num,instantiation]=valid_time(threshold,permutedims(trajectory),test_data[test_num],dt)
                end
            end
        end
    end 
end


SpectralRadii=collect(0.1:0.1:2.0)
Inputscales=SpectralRadii



per_threshold_test_wise_valid_times_per_regime

p=plot(thresholds,mean(mean(per_threshold_test_wise_valid_times_per_regime[1,:,1,:,:],dims=4),dims=5)[:,1,1],color=:black,linewidth=3,label="ODE");
plot!(p,thresholds,mean(mean(per_threshold_test_wise_valid_times_per_regime[1,:,2,:,:],dims=4),dims=5)[:,1,1],color=:red,linewidth=3,label="Hybrid");
plot!(p,thresholds,mean(mean(per_threshold_test_wise_valid_times_per_regime[1,:,3,:,:],dims=4),dims=5)[:,1,1],color=:blue,linewidth=3,label="Standard");
plot!(p,ylabel="Mean t<sup>*</sup> (s)",xlabel="t<sup>*</sup> threshold",legend=:outertopright,left_margin=20pt);
plot!(p,title="$(regimes[1]), $(parameter)=$(SpectralRadii[arrayindex])",titlefontsize=14);
plot!(p,title="Asynchronous, Spectral radius = $(SpectralRadii[arrayindex])",titlefontsize=14)




width,height=p.attr[:size]
Plots.prepare_output(p)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),input_path*"Figures/vary_threshold_$(regimes[1])_$(parameter)_array_index_$(arrayindex).pdf")

