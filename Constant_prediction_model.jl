using Pkg; Pkg.activate(".")
include("$(pwd())/src/HybridRCforNLONS.jl")
using OrdinaryDiffEq, Random, Statistics, Distributions, LinearAlgebra, Plots, DataFrames, DelimitedFiles, CSV, Arrow
import .HybridRCforNLONS: normalised_error, valid_time, xytophase,phasetoxy,generate_arrow

plotlyjs()

save_path="/Users/as15635/Documents/Projects/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/ConstantModelPredictions/"

#load ground truth cases from the arrow files
gt_sync=Matrix(DataFrame(Arrow.Table("/Users/as15635/Documents/Projects/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/Settings_and_GroundTruth/Biharmonic_Kuramoto_Synch_ground_truth_data.arrow.lz4")))
gt_sync=[phasetoxy(gt_sync'[:,i]) for i in 1:size(gt_sync',2)]
gt_sync=reduce(hcat,gt_sync)
gt_AS=Matrix(DataFrame(Arrow.Table("/Users/as15635/Documents/Projects/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/Settings_and_GroundTruth/Biharmonic_Kuramoto_Asynch_ground_truth_data.arrow.lz4")))
gt_AS=[phasetoxy(gt_AS'[:,i]) for i in 1:size(gt_AS',2)]
gt_AS=reduce(hcat,gt_AS)
gt_HC=Matrix(DataFrame(Arrow.Table("/Users/as15635/Documents/Projects/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/Settings_and_GroundTruth/Biharmonic_Kuramoto_HeteroclinicCycles_ground_truth_data.arrow.lz4")))
gt_HC=[phasetoxy(gt_HC'[:,i]) for i in 1:size(gt_HC',2)]
gt_HC=reduce(hcat,gt_HC)
gt_SCPS=Matrix(DataFrame(Arrow.Table("/Users/as15635/Documents/Projects/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/Settings_and_GroundTruth/Biharmonic_Kuramoto_SelfConsistentPartialSynchrony_ground_truth_data.arrow.lz4")))
gt_SCPS=[phasetoxy(gt_SCPS'[:,i]) for i in 1:size(gt_SCPS',2)]
gt_SCPS=reduce(hcat,gt_SCPS)
gt_AF=Matrix(DataFrame(Arrow.Table("/Users/as15635/Documents/Projects/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/Settings_and_GroundTruth/Biharmonic_Kuramoto_Asynch_Fast_ground_truth_data.arrow.lz4")))
gt_AF=[phasetoxy(gt_AF'[:,i]) for i in 1:size(gt_AF',2)]
gt_AF=reduce(hcat,gt_AF)
gts=[gt_sync,gt_AS,gt_HC,gt_SCPS,gt_AF]

function constant_model(du,u,p,t)
    du .= 0.0
end

regimes=["sync","AS","HC","SCPS","AF"]

num_tests=20
dt=0.1
data_dim=20
threshold=0.4
train_len=1000
warmup_len=100
test_len=2500
shift=500

for (ridx,regime) in enumerate(regimes)
    gt=gts[ridx]
    #segment the ground truth data into training, warmup and test data. 20 warmup-test spans in total.
    training_data=gt[:,1:train_len]
    target_data=gt[:,2:train_len+1]
    warmup_test_data=gt[:,train_len+train_len+1:end]
    test_data=Array{Array{Float64,2},1}(undef,20)
    warmup_data=Array{Array{Float64,2},1}(undef,20)
    for i in 1:num_tests
        test_data[i]=warmup_test_data[:,shift+1+(test_len+shift)*(i-1):(test_len+shift)+(test_len+shift)*(i-1)]
        warmup_data[i]=warmup_test_data[:,shift+1-warmup_len+(test_len+shift)*(i-1):shift+(test_len+shift)*(i-1)]
    end
    valid_times=Vector{Float64}(undef,num_tests)
    
    for (tidx,test) in enumerate(test_data)
        const_prediction=Array{Float64,2}(undef,test_len,data_dim)
        u0=test[:,1]
        tspan=(0.0,249.9)
        prob=ODEProblem(constant_model,u0,tspan)
        pred=solve(prob,saveat=dt)
        pred=permutedims(reduce(hcat,pred.u))
        #calculate the valid times and save
        @show size(permutedims(pred))
        @show size(test)
        # @show pred
        valid_times[tidx]=valid_time(threshold,permutedims(pred),test,dt)
        const_prediction[:,1:data_dim]=pred
        if tidx==1
            local name="ConstModel"*"_Biharmonic_Kuramoto_$(regime)_predictions_test_$(tidx)"
            CSV.write(save_path*name*".csv",DataFrame(const_prediction,:auto),writeheader=true)
            generate_arrow(name,save_path)
            rm(save_path*name*".csv")
        end
    end
    #save the valid times:
    name="ConstModel"*"_Biharmonic_Kuramoto_$(regime)_valid_times"
    CSV.write(save_path*name*".csv",DataFrame(reshape(valid_times,1,length(valid_times)),:auto),writeheader=true)
      
end


#plot on equivalent to figure 9?
