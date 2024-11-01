using Pkg; Pkg.activate(".")
using Plots, Arrow, DataFrames
plotlyjs()

arrayindex=10
arrayindex=ARGS[1]

psweep_name="OmegaErrorLarge"
psweep_name=ARGS[2]

ground_truth_regime="asynchronous_0"#asynchronous_0" # asynchronous_0, _1, _2. synchronous_0, _1, _2. multi-frequency_0, _1, _2.
if ground_truth_regime=="asynchronous_0"||ground_truth_regime=="asynchronous_1"||ground_truth_regime=="asynchronous_2"
    gt_index=1
elseif ground_truth_regime=="synchronous_0"||ground_truth_regime=="synchronous_1"||ground_truth_regime=="synchronous_2"
    gt_index=3
else
    gt_index=2
end

test_num="1"
test_num=ARGS[5]
test_num_num=parse(Int64,test_num)

input_path="$(pwd())/Parameter_Error_Task/"
input_path=ARGS[6]

instance_number=2
instance_number=parse(Int64,ARGS[7])

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

ode_std_hybrid_on_particular_regime=Vector{Any}()
for model in ["ODE","Hybrid","Standard"]
    base_model=ground_truth_regime
    gt_type=["Asynchronous","Multi-Frequency","Synchronous"]
    #read in ground truth:
    ground_truth=permutedims(Matrix(DataFrame(Arrow.Table(input_path*"Settings_and_GroundTruth/"*"Model_$(ground_truth_regime)_ground_truth_data.arrow.lz4"))))

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
    
    #read in data
    trajectory=Matrix(DataFrame(Arrow.Table(input_path*"$(psweep_name)/"*"$(psweep_name)"*"_task1_$(ground_truth_regime)_$(model)predictions_test_$(test_num)_array_index_$(arrayindex).arrow.lz4")))
    if model=="ODE"
        col=:black
    elseif model=="Hybrid"
        col=:red
    elseif model=="Standard"
        col=:blue
    end

    # function xytophase(xys)
    #     N=Int64(size(xys)[1]/2)
    #     phases=atan.(xys[N+1:2*N,:],xys[1:N,:])
    #     return phases
    # end

    # function phasetoxy(phases)
    #     xys=reduce(vcat,vcat(cos.(phases),sin.(phases)))
    #     return xys
    # end
    # renormed_trajectories=copy(permutedims(trajectory))
    #         @show size(trajectory)
    #         for i in 1:2
    #             single_res_trajectory=permutedims(trajectory)[(i-1)*10+1:i*10,:]
    #             if i==1
    #                 @show size(single_res_trajectory)
    #             end
    #             rephasenormedtrajectory=copy(single_res_trajectory)
    #             if i==1
    #                 @show size(rephasenormedtrajectory)
    #             end
    #             # pre_norm_error=normalised_error(trajectory, test_data[test_num_num])
    #             for (colidx,sample) in enumerate(eachcol(single_res_trajectory))
    #                 #check difference between pre norm error and post norm error.
    #                 # @show pre_norm_error
    #                 # phase_sample=sample
    #                 phase_sample=xytophase(sample)
                
    #                 #@show size(phase_sample)
    #                 # normed_xy_sample=phase_sample
    #                 normed_xy_sample=phasetoxy(phase_sample)
    #                 if colidx==1000&&i==2&&model=="Standard"
    #                     # @show sample
    #                     # @show phase_sample
    #                     # @show normed_xy_sample
    #                 end
    #                 #@show size(normed_xy_sample)
    #                 rephasenormedtrajectory[:,colidx]=normed_xy_sample
    #             end
    #             # renormed_errors[test_num_num,i]=mean(normalised_error(rephasenormedtrajectory, test_data[test_num_num]))
                
    #             renormed_trajectories[(i-1)*10+1:i*10,:]=rephasenormedtrajectory
    
    #             # println("instance ",i)
    #             # print("Pre norm error: ", mean(pre_norm_error), " Post norm error: ",mean(norm_error), "\n")
    #         end
    #         renormed_trajectories=permutedims(renormed_trajectories)
    #         trajectory=renormed_trajectories
    cols=cgrad(:darkrainbow, collect(0:0.1:1))
    p=plot()
    for i in 1:5
        if i==1&&model=="ODE"
            plot!(p,collect(0.1:0.1:t/10),permutedims(test_data[test_num_num])[1:t,i],label="Ground truth",ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,color=:purple)#,linestyle=:dash)                
            plot!(p,collect(0.1:0.1:t/10),trajectory[1:t,[(instance_number-1)*10+i]],label=model,ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,color=col)
            plot!(p,ylabel="x<sub>i</sub>",xlabel="Time (s)",ylabelfontsize=label_fontsize,xlabelfontsize=label_fontsize,titlefontsize=title_fontsize,tickfontsize=xtick_fontsize,legendfontsize=xtick_fontsize, title=gt_type[gt_index],legend=false,xticks=([0,50,100,150,200,250,300,350,400,450,500],["0","50","100","150","200","250","300","350","400","450","500"]),yticks=([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5],["-1.5","-1.0","0.5","0.0","0.5","1.0","1.5"]))
            plot!(p,bottom_margin=margin_bottom,top_margin=margin_top,left_margin=margin_left,right_margin=margin_right)
        elseif i==1&&(model=="Standard"||model=="Hybrid")
            plot!(p,collect(0.1:0.1:t/10),permutedims(test_data[test_num_num])[1:t,i],label="",ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,color=:purple)#,linestyle=:dash)
            plot!(p,collect(0.1:0.1:t/10),trajectory[1:t,[(instance_number-1)*10+i]],label=model,ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,color=col)
            plot!(p,ylabel="x<sub>i</sub>",xlabel="Time (s)",ylabelfontsize=label_fontsize,xlabelfontsize=label_fontsize,titlefontsize=title_fontsize,tickfontsize=xtick_fontsize,legendfontsize=xtick_fontsize, title=gt_type[gt_index],legend=false,xticks=([0,50,100,150,200,250,300,350,400,450,500],["0","50","100","150","200","250","300","350","400","450","500"]),yticks=([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5],["-1.5","-1.0","0.5","0.0","0.5","1.0","1.5"]))
            plot!(p,bottom_margin=margin_bottom,top_margin=margin_top,left_margin=margin_left,right_margin=margin_right)
        else 
            plot!(p,collect(0.1:0.1:t/10),permutedims(test_data[test_num_num])[1:t,i],label="",ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,color=:purple)#,linestyle=:dash)
            plot!(p,collect(0.1:0.1:t/10),trajectory[1:t,[(instance_number-1)*10+i]],label="",ylims=(-1.5,1.5),size=(SIZE_COLS,SIZE_ROWS),linewidth=3,color=col)
            plot!(p,ylabel="x<sub>i</sub>",xlabel="Time (s)",ylabelfontsize=label_fontsize,xlabelfontsize=label_fontsize,titlefontsize=title_fontsize,tickfontsize=xtick_fontsize,legendfontsize=xtick_fontsize, title=gt_type[gt_index],legend=false,xticks=([0,50,100,150,200,250,300,350,400,450,500],["0","50","100","150","200","250","300","350","400","450","500"]),yticks=([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5],["-1.5","-1.0","0.5","0.0","0.5","1.0","1.5"]))
            plot!(p,bottom_margin=margin_bottom,top_margin=margin_top,left_margin=margin_left,right_margin=margin_right)
        end
    end
    push!(ode_std_hybrid_on_particular_regime,p)

end
    
### the trajectory plots for the supplementary info (S8-S10) were taken from OmegeErrorLarge, array index 10, test 1, reservoir instance 2.
#plot and save the figure.
p=plot(ode_std_hybrid_on_particular_regime...,layout=(2,2),size=(SIZE_COLS*2,SIZE_ROWS*2),legend=:bottom, title="a_0_new_tsit_ins2_seprun")
width,height=p.attr[:size]
Plots.prepare_output(p)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),input_path*"Figures/$(ground_truth_regime)_trajectory_$(psweep_name)_instance_$(instance_number)_test_$(test_num)_array_index_$(arrayindex).pdf",width=width,height=height)

