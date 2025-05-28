using Pkg; Pkg.activate(".")
using Plots, Statistics, CSV, Arrow, DelimitedFiles, DataFrames, Plots.Measures, Printf

#read in args specifyign the sweep and conditions to plot.
psweep_name="all"
# psweep_name = ARGS[1]
original_base_model="all"
# original_base_model = ARGS[2]
model_type="all"
# model_type = ARGS[3]
test_num_range=1:20
# test_num_range=parse(Int64,ARGS[6]):parse(Int64,ARGS[7])
input_path_base="$(pwd())/Residual_Physics_Task/"
input_path_base="/user/work/as15635/output_data/ExtKuramoto/"
# input_path_base=ARGS[8]
num_reservoirs=40
# num_reservoirs=parse(Int64,ARGS[9])

plotlyjs()

tickfontsize=16
labelfontsize=20
title_fontsize=20
legendfontsize=20
titlefontsize=20
default(titlefontsize=20)

plot_vector=Vector{Any}()

case_letters=[["<b>a</b>","<b>e</b>","<b>i</b>","<b>m</b>"],["<b>b</b>","<b>f</b>","<b>j</b>","<b>n</b>"],["<b>c</b>","<b>g</b>","<b>k</b>","<b>o</b>"],["<b>d</b>","<b>h</b>","<b>l</b>","<b>p</b>"],["<b>b</b>","<b>f</b>","<b>j</b>","<b>n</b>"]]
for arrindex in [1,5,3,4,2]
    if original_base_model=="all"
        four_models=["Synch", "Asynch","HeteroclinicCycles","SelfConsistentPartialSynchrony","Asynch_Fast"]
        base_model=four_models[arrindex]
    end 

    #the valid times are saved in CSVs on a per array index basis for each model type for each regime
    # each csv file has the number of columns equal to the number of reservoirs/instances of the ODEs that were tested.
    # And in each row the individual test trajectory valid times.

    if model_type=="all"
        models=["ODE","Standard","Hybrid"]
        colours=[:black,:blue,:red]
    else
        models=[model_type]
        if model_type=="ODE"
            colours=[:black]
        elseif model_type=="Standard"
            colours=[:blue]
        else
            colours=[:red]
        end
    end

    if psweep_name=="all"
        parameters_to_plot=["SpectralRadius","InputScaling","Regularisation","ReservoirSize"]
    else
        parameters_to_plot=[psweep_name]
    end

    #tick label formatting
    pvalues_dict=Dict("OmegaError"=>collect(range(0.004,0.08,20)),"OmegaErrorLarge"=>collect(range(0.1,0.48,20)),"KError"=>collect(range(0.004,0.08,20)),"KErrorLarge"=>collect(range(0.1,0.48,20)),"ReservoirSize"=>collect(range(50,1000,20)),"InputScaling"=>collect(range(0.1,2.0,20)),"InputScalingLarge"=>collect(range(1.05,2.0,20)),"SpectralRadius"=>collect(range(0.1,2.0,20)),"MeanDegree"=>collect(range(2,21,20)),"KnowledgeRatio"=>collect(range(0.05,1.0,20)),"Regularisation"=>[0.5,0.25,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001])
    pticks_labels=[string.(pvalues_dict[i][1:2:end]) for i in parameters_to_plot]
    pticks_dict=Dict(collect(zip(parameters_to_plot,pticks_labels)))
    pticks_locations=Dict(collect(zip(parameters_to_plot,[pvalues_dict[i][1:2:end] for i in parameters_to_plot])))
    RS_ticks=string.([@sprintf "%.0f" i for i in pvalues_dict["ReservoirSize"]])
    pticks_dict["ReservoirSize"]=(RS_ticks)[1:3:end]
    pticks_locations["ReservoirSize"]=pvalues_dict["ReservoirSize"][1:3:end]

    model_titles=["Synchronous", "Asynchronous Slow","Heteroclinic Cycles","Partial Synchrony","Asynchronous Fast"]
    parameters_sweep_names=["Spectral radius ","Input scaling","Regularisation","D<sub>r</sub>"]
    
    for (pidx,param) in enumerate(parameters_to_plot)
        println("plotting ", param)
        input_path = input_path_base*param*"/"
        p=scatter();
        #set to range tested. (array index range is the number of tested parameter settings in the parameter sweeps. Regularisation was 10.)
        if param=="Regularisation"
            global arrayindex_range=collect(1:1:10)
        else
            global arrayindex_range=collect(1:1:20)
        end

        for (idx,model) in enumerate(models)
            colour=colours[idx]
            collected_mean_errors=Array{Float64,2}(undef, length(arrayindex_range), num_reservoirs)
            for arrayindex in arrayindex_range
                # if arrayindex==15||arrayindex==16
                    # arrayindex_replace=14
                    # norm_errors=Matrix(DataFrame(CSV.read(input_path*param*"_"*model*"_"*"Biharmonic_Kuramoto"*"_"*base_model*"_valid_times_array_index_$(arrayindex_replace).csv",DataFrame)))
                    # mean_over_tests=mean(norm_errors[test_num_range,:],dims=1)
                    # collected_mean_errors[arrayindex-arrayindex_range[1]+1,:]=mean_over_tests 
                # else 
                    if base_model=="Asynch_Fast"
                        norm_errors=Matrix(DataFrame(CSV.read(input_path*param*"_"*model*"_"*"Biharmonic_Kuramoto"*"_"*base_model*"_valid_times_array_index_$(arrayindex).csv",DataFrame)))
                        mean_over_tests=mean(norm_errors[test_num_range,:],dims=1)
                        collected_mean_errors[arrayindex-arrayindex_range[1]+1,:]=mean_over_tests    
                    else
                        norm_errors=Matrix(DataFrame(CSV.read("/user/work/as15635/output_data/ExtKuramoto/"*"New_Error_Metrics/"*param*"_"*model*"_"*"ExtKuramoto"*"_"*base_model*"_mean_valid_times_arrayindex_$(arrayindex).csv",DataFrame)))
                        mean_over_tests=mean(norm_errors[test_num_range,:],dims=1)
                        collected_mean_errors[arrayindex-arrayindex_range[1]+1,:]=mean_over_tests
                    end
                # end
            end
            scatter!(p,[get(pvalues_dict,param,"..")[arrayindex_range] for i in 1:size(collected_mean_errors,1)],collected_mean_errors,xticks=(pticks_locations[param],pticks_dict[param]), color=colour,label=nothing,markersize=2,markerstrokecolor=:match,markeralpha=0.45,markerstrokewidth=0.0,size=(560,480),dpi=300);
            if arrindex>=2
                if arrindex==4&&pidx==1
                    plot!(p,get(pvalues_dict,param,"..")[arrayindex_range],title=model_titles[arrindex],titlefontsize=title_fontsize,mean(collected_mean_errors,dims=2),label=model,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=:right,xlabel=parameters_sweep_names[pidx],ylabel="", xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize,size=(560,480),margin=(5mm),dpi=300);
                elseif pidx==1 
                    plot!(p,get(pvalues_dict,param,"..")[arrayindex_range],title=model_titles[arrindex],titlefonsize=title_fontsize,mean(collected_mean_errors,dims=2),label=nothing,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=false,xlabel=parameters_sweep_names[pidx],ylabel="", xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize,size=(560,480),margin=(5mm),dpi=300);
                else   
                    plot!(p,get(pvalues_dict,param,"..")[arrayindex_range],titlefontsize=title_fontsize,mean(collected_mean_errors,dims=2),label=nothing,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=false,xlabel=parameters_sweep_names[pidx],ylabel="", xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize,size=(560,480),margin=(5mm),dpi=300);
                end
                if param=="Regularisation"
                    plot!(xscale=:log10)
                end
            else
                if arrindex==4&&pidx==1
                    plot!(p,get(pvalues_dict,param,"..")[arrayindex_range],titlefontsize=title_fontsize,mean(collected_mean_errors,dims=2),label=model,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=:right,xlabel=parameters_sweep_names[pidx],ylabel="Mean t<sup>*</sup> (s)", xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize,size=(560,480),margin=(5mm),dpi=300);
                elseif pidx==1
                    plot!(p,get(pvalues_dict,param,"..")[arrayindex_range],title=model_titles[arrindex],titlefontsize=title_fontsize,mean(collected_mean_errors,dims=2),label=nothing,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=false,xlabel=parameters_sweep_names[pidx],ylabel="Mean t<sup>*</sup> (s)", xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize,size=(560,480),margin=(5mm),dpi=300);
                else
                    plot!(p,get(pvalues_dict,param,"..")[arrayindex_range],titlefontsize=title_fontsize,mean(collected_mean_errors,dims=2),label=nothing,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=false,xlabel=parameters_sweep_names[pidx],ylabel="Mean t<sup>*</sup> (s)", xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize,size=(560,480),margin=(5mm),dpi=300);
                end
                if param=="Regularisation"
                    plot!(xscale=:log10)
                end
            end
        end
        
        if arrindex==1
            plot!(p,ylims=(0.0,250.0))
        elseif arrindex==2
            plot!(p,ylims=(0.0,7.0))
        elseif arrindex==3
            plot!(p,ylims=(0.0,2.5))
        elseif arrindex==4||arrindex==5
            plot!(p,ylims=(0.0,2.5))
        end
        if param=="Regularisation"
            # annotate!(p,log10(get(pvalues_dict,param,"..")[end]*1.55),(Plots.ylims(p[1])[2])*0.95,text(case_letters[arrindex][pidx],20))
        else
            # annotate!(p,get(pvalues_dict,param,"..")[1]*1.55,(Plots.ylims(p[1])[2])*0.95,text(case_letters[arrindex][pidx],20))
        end

        yticks_p=yticks(p)
        plot!(p,yticks=(yticks_p[1][1],string.(yticks_p[1][2])))
        display(p)
        push!(plot_vector,p)
    end
end


#load const model data and add to the figure appropriately:

const_HC_vts=Matrix(DataFrame(CSV.read("/user/home/as15635/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/ConstantModelPredictions/ConstModel_Biharmonic_Kuramoto_HC_valid_times.csv",DataFrame)))
const_HC_mean=mean(const_HC_vts[:,test_num_range],dims=2)[1,1]
const_HC_std=std(const_HC_vts[:,test_num_range],dims=2)[1,1]
const_AF_vts=Matrix(DataFrame(CSV.read("/user/home/as15635/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/ConstantModelPredictions/ConstModel_Biharmonic_Kuramoto_AF_valid_times.csv",DataFrame)))
const_AF_mean=mean(const_AF_vts[:,test_num_range],dims=2)[1,1]
const_AF_std=std(const_AF_vts[:,test_num_range],dims=2)[1,1]
const_AS_vts=Matrix(DataFrame(CSV.read("/user/home/as15635/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/ConstantModelPredictions/ConstModel_Biharmonic_Kuramoto_AS_valid_times.csv",DataFrame)))
const_AS_mean=mean(const_AS_vts[:,test_num_range],dims=2)[1,1]
const_AS_std=std(const_AS_vts[:,test_num_range],dims=2)[1,1]
const_sync_vts=Matrix(DataFrame(CSV.read("/user/home/as15635/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/ConstantModelPredictions/ConstModel_Biharmonic_Kuramoto_sync_valid_times.csv",DataFrame)))
const_sync_mean=mean(const_sync_vts[:,test_num_range],dims=2)[1,1]
const_sync_std=std(const_sync_vts[:,test_num_range],dims=2)[1,1]
const_SCPS_vts=Matrix(DataFrame(CSV.read("/user/home/as15635/Hybrid_RC_for_NLONS_paper_code/Residual_Physics_Task/ConstantModelPredictions/ConstModel_Biharmonic_Kuramoto_SCPS_valid_times.csv",DataFrame)))
const_SCPS_mean=mean(const_SCPS_vts[:,test_num_range],dims=2)[1,1]
const_SCPS_std=std(const_SCPS_vts[:,test_num_range],dims=2)[1,1]

#the plot vector is ordered as follows: [sync, AF, HC,SCPS] with four plots per entry
#so for each one, i need to just plot a repeat of the mean const vt with a std ribbon around it, in green

#need the ticks
pvalues_dict=Dict("OmegaError"=>collect(range(0.004,0.08,20)),"OmegaErrorLarge"=>collect(range(0.1,0.48,20)),"KError"=>collect(range(0.004,0.08,20)),"KErrorLarge"=>collect(range(0.1,0.48,20)),"ReservoirSize"=>collect(range(50,1000,20)),"InputScaling"=>collect(range(0.1,2.0,20)),"InputScalingLarge"=>collect(range(1.05,2.0,20)),"SpectralRadius"=>collect(range(0.1,2.0,20)),"MeanDegree"=>collect(range(2,21,20)),"KnowledgeRatio"=>collect(range(0.05,1.0,20)),"Regularisation"=>[0.5,0.25,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001])
parameters_to_plot=["SpectralRadius","InputScaling","Regularisation","ReservoirSize"]
pticks_labels=[string.(pvalues_dict[i][1:2:end]) for i in parameters_to_plot]
pticks_dict=Dict(collect(zip(parameters_to_plot,pticks_labels)))
pticks_locations=Dict(collect(zip(parameters_to_plot,[pvalues_dict[i][1:2:end] for i in parameters_to_plot])))
RS_ticks=string.([@sprintf "%.0f" i for i in pvalues_dict["ReservoirSize"]])
pticks_dict["ReservoirSize"]=(RS_ticks)[1:3:end]
pticks_locations["ReservoirSize"]=pvalues_dict["ReservoirSize"][1:3:end]

for i in 1:4
    param=parameters_to_plot[i]
    if i == 3
        plot!(plot_vector[i],get(pvalues_dict,param,"..")[1:10],repeat([const_sync_mean],10),color=:green,linewidth=3,ribbon=const_sync_std,fillalpha=0.3,label="")
    else
        plot!(plot_vector[i],get(pvalues_dict,param,"..")[1:20],repeat([const_sync_mean],20),color=:green,linewidth=3,ribbon=const_sync_std,fillalpha=0.3,label="")
    end
end

for i in 5:8
    param=parameters_to_plot[i-4]
    if i == 7
        plot!(plot_vector[i],get(pvalues_dict,param,"..")[1:10],repeat([const_AF_mean],10),color=:green,linewidth=3,ribbon=const_AF_std,fillalpha=0.3,label="")
    else
        plot!(plot_vector[i],get(pvalues_dict,param,"..")[1:20],repeat([const_AF_mean],20),color=:green,linewidth=3,ribbon=const_AF_std,fillalpha=0.3,label="")
    end
end

for i in 9:12
    param=parameters_to_plot[i-8]
    if i == 11
        plot!(plot_vector[i],get(pvalues_dict,param,"..")[1:10],repeat([const_HC_mean],10),color=:green,linewidth=3,ribbon=const_HC_std,fillalpha=0.3,label="")
    else
        plot!(plot_vector[i],get(pvalues_dict,param,"..")[1:20],repeat([const_HC_mean],20),color=:green,linewidth=3,ribbon=const_HC_std,fillalpha=0.3,label="")
    end
end

for i in 13:16
    param=parameters_to_plot[i-12]
    if i == 15
        plot!(plot_vector[i],get(pvalues_dict,param,"..")[1:10],repeat([const_SCPS_mean],10),color=:green,linewidth=3,ribbon=const_SCPS_std,fillalpha=0.3,label="Constant")
    else
        plot!(plot_vector[i],get(pvalues_dict,param,"..")[1:20],repeat([const_SCPS_mean],20),color=:green,linewidth=3,ribbon=const_SCPS_std,fillalpha=0.3,label="")
    end
end

for i in 17:20
    param=parameters_to_plot[i-16]
    if i == 19
        plot!(plot_vector[i],get(pvalues_dict,param,"..")[1:10],repeat([const_AS_mean],10),color=:green,linewidth=3,ribbon=const_AS_std,fillalpha=0.3,label="",ylims=(0.0,40),yticks=(0:10:40))
    else
        plot!(plot_vector[i],get(pvalues_dict,param,"..")[1:20],repeat([const_AS_mean],20),color=:green,linewidth=3,ribbon=const_AS_std,fillalpha=0.3,label="",ylims=(0.0,40),yticks=(0:10:40))
    end
end



# plot(plot_vector...,layout=(4,4),size=(2240,1920))



#save the plot.
p=plot(reshape(permutedims(reshape(plot_vector,4,5)),1,20)[1,:]...,size=(2800,1920),margin=2Plots.mm,left_margin=15mm,bottom_margin=14mm,legend_position=(0.915,0.97))
width, height = p.attr[:size]
Plots.prepare_output(p)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),"$(pwd())/Residual_Physics_Task/Figures/residual_physics_parameter_sweep_withConstModel.pdf",width=width,height=height)

#fast async only plot
p=plot(plot_vector[5:8]...,size=(1080,1020),layout=(2,2),margin=2Plots.mm,left_margin=15mm,bottom_margin=14mm,legend_position=(0.915,0.97))
width, height = p.attr[:size]
Plots.prepare_output(p)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),"$(pwd())/Residual_Physics_Task/Figures/residual_physics_parameter_sweep_async_fast.pdf",width=width,height=height)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),"$(pwd())/Residual_Physics_Task/Figures/residual_physics_parameter_sweep_async_fast_with_constmodel.pdf",width=width,height=height)
