using Pkg; Pkg.activate(".")
using Plots, Arrow, DataFrames
using ColorSchemes
plotlyjs()

function xytophase(xys)
    N=Int64(size(xys)[1]/2)
    phases=atan.(xys[N+1:2*N,:],xys[1:N,:])
    return phases
end

function phasetoxy(phases)
    xys=reduce(vcat,vcat(cos.(phases),sin.(phases)))
    return xys
end

arrayindex=ARGS[1]
# arrayindex=1

psweep_name=ARGS[2]
# psweep_name="InputScaling"


ground_truth_regime=ARGS[3]
# ground_truth_regime="SelfConsistentPartialSynchrony" #Asynchronous, HeteroclinicCycles, SelfConsistentPartialSynchrony
if ground_truth_regime=="Synchronous"
    gt_index=1
elseif ground_truth_regime=="Asynchronous"
    gt_index=2
elseif ground_truth_regime=="HeteroclinicCycles"
    gt_index=3
else
    gt_index=4
end

#for titles.
gt_type=["Synchronous","Asynchronous","Heteroclinic Cycles","Partial Synchrony"]
#for file names.
base_models=["Synch","Asynch","HeteroclinicCycles","SelfConsistentPartialSynchrony"]
base_model=base_models[gt_index]

test_num=ARGS[5]
# test_num="1"
test_num_num=parse(Int64,test_num)

instance_number=parse(Int64,ARGS[6])
# instance_number=1

input_path="$(pwd())/Residual_Physics_Task/"
# input_path=ARGS[7]



label_fontsize=20
title_fontsize=20
xtick_fontsize=16
SIZE_COLS=1020
SIZE_ROWS=400
margin_bottom=20Plots.mm
margin_top=5Plots.mm
margin_left=10Plots.mm
margin_right=1Plots.mm

t=1500 #how many time steps to plot.

#read in ground truth:
ground_truth=permutedims(Matrix(DataFrame(Arrow.Table(input_path*"Settings_and_GroundTruth/"*"Biharmonic_Kuramoto_$(base_model)_ground_truth_data.arrow.lz4"))))
ground_truth=[phasetoxy(ground_truth[:,i]) for i in 1:size(ground_truth,2)]
ground_truth=reduce(hcat,ground_truth)

train_len=1000
warmup_len=100
test_len=2500
shift=500

training_data=ground_truth[:,1:train_len]
target_data=ground_truth[:,2:train_len+1]
warmup_test_data=ground_truth[:,train_len+train_len+1:end]
test_data=Array{Array{Float64,2},1}(undef,20)
warmup_data=Array{Array{Float64,2},1}(undef,20)
for i in 1:20
    test_data[i]=warmup_test_data[:,shift+1+(test_len+shift)*(i-1):(test_len+shift)+(test_len+shift)*(i-1)]
    warmup_data[i]=warmup_test_data[:,shift+1-warmup_len+(test_len+shift)*(i-1):shift+(test_len+shift)*(i-1)]
end

ode_std_hybrid_on_particular_regime=Vector{Any}()
for model in ["ODE","Standard","Hybrid"]
    plots=Vector{Any}()

    #read in data
    trajectory=Matrix(DataFrame(Arrow.Table(input_path*"$(psweep_name)/"*"$(psweep_name)"*"_"*"$(model)_Biharmonic_Kuramoto_$(base_model)_predictions_test_$(test_num)_array_index_$(arrayindex).arrow.lz4")))

    if model=="ODE"
        col=:black
    elseif model=="Hybrid"
        col=:red
    elseif model=="Standard"
        col=:blue
    end

    #plot all 10 oscillator's x-component trajectories
    p=plot()
    for i in 1:10
        if i==1&&model=="ODE"
            plot!(p,collect(0.1:0.1:t/10),permutedims(test_data[test_num_num])[1:t,i],label="Ground truth",ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,colour=:purple,linestyle=:dash)#palette=:ColorSchemes.Purples,linestyle=:dash)
            plot!(p,collect(0.1:0.1:t/10),trajectory[1:t,i+(20*(instance_number-1))],label=model,ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,color=col,legend=true)
            plot!(p,ylabel="x<sub>i</sub>",xlabel="Time (s)",ylabelfontsize=label_fontsize,xlabelfontsize=label_fontsize,titlefontsize=title_fontsize,tickfontsize=xtick_fontsize,legendfontsize=xtick_fontsize, label="",title=gt_type[gt_index],legend=:topright,xticks=([0,50,100,150,200],["0","50","100","150"]),yticks=([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5],["-1.5","-1.0","0.5","0.0","0.5","1.0","1.5"]))
            plot!(p,bottom_margin=margin_bottom,top_margin=margin_top,left_margin=margin_left,right_margin=margin_right,label="")
        elseif i==1&&(model=="Hybrid"||model=="Standard")
            plot!(p,collect(0.1:0.1:t/10),permutedims(test_data[test_num_num])[1:t,i],label="",ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,colour=:purple,linestyle=:dash)#palette=:ColorSchemes.Purples,linestyle=:dash)
            plot!(p,collect(0.1:0.1:t/10),trajectory[1:t,i+(20*(instance_number-1))],label=model,ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,color=col,legend=true)
            plot!(p,ylabel="x<sub>i</sub>",xlabel="Time (s)",ylabelfontsize=label_fontsize,xlabelfontsize=label_fontsize,titlefontsize=title_fontsize,tickfontsize=xtick_fontsize,legendfontsize=xtick_fontsize, label="",title=gt_type[gt_index],legend=:topright,xticks=([0,50,100,150,200],["0","50","100","150"]),yticks=([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5],["-1.5","-1.0","0.5","0.0","0.5","1.0","1.5"]))
            plot!(p,bottom_margin=margin_bottom,top_margin=margin_top,left_margin=margin_left,right_margin=margin_right,label="")
        else
            plot!(p,collect(0.1:0.1:t/10),permutedims(test_data[test_num_num])[1:t,i],label="",ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,colour=:purple,linestyle=:dash)#palette=:ColorSchemes.Purples,linestyle=:dash)
            plot!(p,collect(0.1:0.1:t/10),trajectory[1:t,i+(20*(instance_number-1))],label="",ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,color=col)
            plot!(p,ylabel="x<sub>i</sub>",xlabel="Time (s)",ylabelfontsize=label_fontsize,xlabelfontsize=label_fontsize,titlefontsize=title_fontsize,tickfontsize=xtick_fontsize,legendfontsize=xtick_fontsize, label="",title=gt_type[gt_index],legend=:topright,xticks=([0,50,100,150,200],["0","50","100","150"]),yticks=([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5],["-1.5","-1.0","0.5","0.0","0.5","1.0","1.5"]))
            plot!(p,bottom_margin=margin_bottom,top_margin=margin_top,left_margin=margin_left,right_margin=margin_right,label="")
        end
    end
    push!(plots,p)
    push!(ode_std_hybrid_on_particular_regime,plots[1])
end

### the trajectory plots for the supplementary info (S11-S14) were taken from the InputScaling sweep, array index 1, test 1, instance 1.
### the trajectory plots for S15 were taken from the InputScaling sweep, array index 19, test 1, instance 1.
### exact trajectories will vary due to use of the default random seed in the modified_params generation.
#plot and save the figure.
p=plot(ode_std_hybrid_on_particular_regime...,layout=(2,2),size=(SIZE_COLS*2,SIZE_ROWS*2),legend=:bottom)
width,height=p.attr[:size]
Plots.prepare_output(p)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),input_path*"Figures/$(base_model)_trajectory_$(pwsweep_name)_instance_$(instance_number)_test_$(test_num)_array_index_$(arrayindex).pdf",width=width,height=height)
