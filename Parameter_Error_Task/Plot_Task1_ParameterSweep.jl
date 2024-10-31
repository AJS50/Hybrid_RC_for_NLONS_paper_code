using Pkg; Pkg.activate(".")
using Plots, Statistics, CSV, Arrow, DelimitedFiles, DataFrames, Plots.Measures, Printf

#read in args specifying the sweep and conditions to plot.
psweep_name="all" #all sweeps can be plotted at once.
# psweep_name = ARGS[1]

model_type="all" #ODE, Standard, Hybrid all plotted together.
# model_type = ARGS[2]


arrayindex_range=1:3 #number of parameter settings tested.
# arrayindex_range=parse(Int64,ARGS[3]):parse(Int64,ARGS[4])

test_num_range=1:6 #three regime instantiations, each 20 test spans. so 60 overall.
# test_num_range=parse(Int64,ARGS[5]):3*parse(Int64,ARGS[6])


input_path_base="$(pwd())/Parameter_Error_Task/" #parent folder with normalised error data in parameter specific folders.
# input_path_base=ARGS[7]
num_reservoirs=1
# num_reservoirs=parse(Int64,ARGS[8])

three_models=[["2","2_1","2_2"],["7","7_1","7_2"],["16","16_1","16_2"]]
case_letters=["b","c","a"]
case_letters_IS=["e","f","d"]


plotlyjs()

subplot_collection=Vector{Any}()
subplot_collection_IScombined=Vector{Any}()
for arrindex in 1:3 #plotting all regimes at once.
    #the normalised errors are saved in CSVs on a per array index (parameter setting) basis for each model type for each model instance.
    # each csv file has a number of columns equal to the number of reservoirs/instances of the ODEs that were tested. (40)
    # And in each row the idividual test trajectories mean normalised error scores. (20)

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
        parameters_to_plot=[ "KError","KErrorLarge", "OmegaError","OmegaErrorLarge", "ReservoirSize", "InputScaling", "KnowledgeRatio", "InputScalingLarge", "SpectralRadius","MeanDegree"]
        parameters_to_plot=[ "KError"]
    else
        parameters_to_plot=[psweep_name]
    end

    #tick label formatting.
    pvalues_dict=Dict("OmegaError"=>collect(range(0.004,0.08,20)),"OmegaErrorLarge"=>collect(range(0.1,0.48,20)),"KError"=>collect(range(0.004,0.08,20)),"KErrorLarge"=>collect(range(0.1,0.48,20)),"ReservoirSize"=>collect(range(50,1000,20)),"InputScaling"=>collect(range(0.05,1.0,20)),"InputScalingLarge"=>collect(range(1.05,2.0,20)),"SpectralRadius"=>collect(range(0.1,2.0,20)),"MeanDegree"=>collect(range(2,21,20)),"KnowledgeRatio"=>collect(range(0.05,1.0,20)))
    pticks_labels=[string.(pvalues_dict[i][1:2:end]) for i in parameters_to_plot]
    pticks_dict=Dict(collect(zip(parameters_to_plot,pticks_labels)))
    RS_ticks=string.([@sprintf "%.0f" i for i in pvalues_dict["ReservoirSize"]])
    pticks_dict["ReservoirSize"]=(RS_ticks)[1:3:end] 


    model_titles=["Asynchronous", "MultiFrequency", "Synchronous"]
    model_names=["asynchronous","multi-frequency","synchronous"]
    parameters_sweep_names=["σ<sub>K</sub>","σ<sub>K</sub>", "σ<sub>ω</sub>","σ<sub>ω</sub>", "D<sub>r</sub>", "Input scaling", "Knowledge ratio", "Input Scaling Large", "Spectral radius","<d>"]

    tickfontsize=16
    labelfontsize=20
    legendfontsize=20
    titlefontsize=20

    for (pnidx,param) in enumerate(parameters_to_plot)
        println("plotting ", param)
        input_path = input_path_base*param #for all but mean degree case used for ODE runs on parameters not affecting it.
        p=Plots.scatter();
        for (idx,model) in enumerate(models)
            colour=colours[idx]
            collected_mean_errors=Array{Float64,2}(undef, length(arrayindex_range), num_reservoirs)
            for arrayindex in arrayindex_range
                #use Mean degree sweep data for ODE model for these parameters as they do not effect it.       
                if model=="ODE"&&param in ["ReservoirSize", "InputScaling", "KnowledgeRatio", "InputScalingLarge", "SpectralRadius", "MeanDegree"]  
                    norm_errors_1=Matrix(DataFrame(CSV.read(input_path_base*"MeanDegree/"*"MeanDegree"*"_task1_$(model_names[arrindex])_0"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_2=Matrix(DataFrame(CSV.read(input_path_base*"MeanDegree/"*"MeanDegree"*"_task1_$(model_names[arrindex])_1"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_3=Matrix(DataFrame(CSV.read(input_path_base*"MeanDegree/"*"MeanDegree"*"_task1_$(model_names[arrindex])_2"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors=vcat(vcat(norm_errors_1,norm_errors_2),norm_errors_3)
                else
                    norm_errors_1=Matrix(DataFrame(CSV.read(input_path*"/"*param*"_task1_$(model_names[arrindex])_0"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_2=Matrix(DataFrame(CSV.read(input_path*"/"*param*"_task1_$(model_names[arrindex])_1"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_3=Matrix(DataFrame(CSV.read(input_path*"/"*param*"_task1_$(model_names[arrindex])_2"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors=vcat(vcat(norm_errors_1,norm_errors_2),norm_errors_3)
                end   
                mean_over_tests=mean(norm_errors[test_num_range,:],dims=1)
                collected_mean_errors[arrayindex-arrayindex_range[1]+1,:]=mean_over_tests    
            end
            scatter!(p,[get(pvalues_dict,param,"..")[arrayindex_range] for i in 1:size(collected_mean_errors,1)],collected_mean_errors, color=colour,label=nothing,markersize=2,markerstrokecolor=:match,markeralpha=0.45,markerstrokewidth=0.0,ylims=(0.0,2.5),size=(2830,2010));
            #draw mean line for each case and shaded area for the standard deviation.
            #only label the legend for the first model instance. otherwise will repeat.
            if arrindex==2
                plot!(p,get(pvalues_dict,param,"..")[arrayindex_range],mean(collected_mean_errors,dims=2),label=model,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=:topright,xlabel=parameters_sweep_names[pnidx],ylabel="",title=model_titles[arrindex], xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize,titlefontsize=titlefontsize, margin=(5mm),size=(720,480));#title=model_titles[arrindex],size=(2830,2010));
            elseif arrindex==1
                plot!(p,get(pvalues_dict,param,"..")[arrayindex_range],mean(collected_mean_errors,dims=2),label=nothing,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=false,xlabel=parameters_sweep_names[pnidx],ylabel="",title=model_titles[arrindex], xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize,titlefontsize=titlefontsize, margin=(5mm),size=(720,480));#title=model_titles[arrindex],size=(2830,2010));
            else
                plot!(p,get(pvalues_dict,param,"..")[arrayindex_range],mean(collected_mean_errors,dims=2),label=nothing,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=false,xlabel=parameters_sweep_names[pnidx],ylabel="Mean NMSE", title=model_titles[arrindex],xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize,titlefontsize=titlefontsize, margin=(5mm),left_margin=15Plots.mm,size=(720,480));#title=model_titles[arrindex],size=(2830,2010));
            end
            #add lower case letters.
            annotate!(get(pvalues_dict,param,"..")[1],[2.4],text(case_letters[arrindex],20))

        end

        yticks_p=yticks(p)
        #manually set ticks.
        plot!(p,yticks=(yticks_p[1][1],String.(yticks_p[1][2])),xticks=(pvalues_dict[param][arrayindex_range][1:2:end],pticks_dict[param]))
        if param=="SpectralRadius"
            plot!(p,xticks=([0.1,0.5,1.0,1.5,2.0],string.([0.1,0.5,1.0,1.5,2.0])))
        end
        push!(subplot_collection,p)
    end

    # plotting full range input scaling graph; "InputScaling" and "InputScalingLarge" together)
    parameters_to_plot=["InputScaling"]
    for param in parameters_to_plot
        println("plotting ", param)

        p=scatter();
        for (idx,model) in enumerate(models)
            # @show idx
            colour=colours[idx]
            collected_mean_errors=Array{Float64,2}(undef, length(arrayindex_range)*2, num_reservoirs)

            #adding input scaling data
            param="InputScaling"
            input_path = input_path_base*param
            for arrayindex in arrayindex_range
                #use Mean degree sweep data for ODE model for these parameters as they do not effect it.       
                if model=="ODE"&&param in ["ReservoirSize", "InputScaling", "KnowledgeRatio", "InputScalingLarge", "SpectralRadius", "MeanDegree"]  
                    norm_errors_1=Matrix(DataFrame(CSV.read(input_path_base*"MeanDegree/"*"MeanDegree"*"_task1_$(model_names[arrindex])_0"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_2=Matrix(DataFrame(CSV.read(input_path_base*"MeanDegree/"*"MeanDegree"*"_task1_$(model_names[arrindex])_1"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_3=Matrix(DataFrame(CSV.read(input_path_base*"MeanDegree/"*"MeanDegree"*"_task1_$(model_names[arrindex])_2"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors=vcat(vcat(norm_errors_1,norm_errors_2),norm_errors_3)
                else
                    norm_errors_1=Matrix(DataFrame(CSV.read(input_path*"/"*param*"_task1_$(model_names[arrindex])_0"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_2=Matrix(DataFrame(CSV.read(input_path*"/"*param*"_task1_$(model_names[arrindex])_1"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_3=Matrix(DataFrame(CSV.read(input_path*"/"*param*"_task1_$(model_names[arrindex])_2"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors=vcat(vcat(norm_errors_1,norm_errors_2),norm_errors_3)
                end   
                mean_over_tests=mean(norm_errors[test_num_range,:],dims=1)
                collected_mean_errors[arrayindex-arrayindex_range[1]+1,:]=mean_over_tests    
            end
            #adding inputscalinglarge data
            param="InputScalingLarge"
            input_path = input_path_base*param
            for arrayindex in arrayindex_range
                #use Mean degree sweep data for ODE model for these parameters as they do not effect it.       
                if model=="ODE"&&param in ["ReservoirSize", "InputScaling", "KnowledgeRatio", "InputScalingLarge", "SpectralRadius", "MeanDegree"]  
                    norm_errors_1=Matrix(DataFrame(CSV.read(input_path_base*"MeanDegree/"*"MeanDegree"*"_task1_$(model_names[arrindex])_0"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_2=Matrix(DataFrame(CSV.read(input_path_base*"MeanDegree/"*"MeanDegree"*"_task1_$(model_names[arrindex])_1"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_3=Matrix(DataFrame(CSV.read(input_path_base*"MeanDegree/"*"MeanDegree"*"_task1_$(model_names[arrindex])_2"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors=vcat(vcat(norm_errors_1,norm_errors_2),norm_errors_3)
                else
                    norm_errors_1=Matrix(DataFrame(CSV.read(input_path*"/"*param*"_task1_$(model_names[arrindex])_0"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_2=Matrix(DataFrame(CSV.read(input_path*"/"*param*"_task1_$(model_names[arrindex])_1"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors_3=Matrix(DataFrame(CSV.read(input_path*"/"*param*"_task1_$(model_names[arrindex])_2"*"_"*model*"norm_errors_array_index_$(arrayindex).csv",DataFrame)))
                    norm_errors=vcat(vcat(norm_errors_1,norm_errors_2),norm_errors_3)
                end   
                mean_over_tests=mean(norm_errors[test_num_range,:],dims=1)
                collected_mean_errors[arrayindex+arraindex_range[2]-arrayindex_range[1]+1,:]=mean_over_tests   

            end

            scatter!(p,[vcat(get(pvalues_dict,"InputScaling","..")[1:1:20],get(pvalues_dict,"InputScalingLarge","..")[1:1:20]) for i in 1:size(collected_mean_errors,1)],collected_mean_errors, color=colour,label=nothing,markersize=2,markerstrokecolor=:match,markeralpha=0.45,markerstrokewidth=0.0,ylims=(0.0,2.5),size=(5630,2010));
            
            #draw mean line for each case and shaded area for the standard deviation.
            #only label first model series as will repeat in legend otherwise.
            if arrindex==2
                plot!(p,vcat(get(pvalues_dict,"InputScaling","..")[1:1:20],get(pvalues_dict,"InputScalingLarge","..")[1:1:20]),mean(collected_mean_errors,dims=2),label=nothing,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=:topright,xlabel=parameters_sweep_names[6],ylabel="", xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize, margin=(5mm),size=(720,480));#title=model_titles[arrindex],#,size=(5630,2010) );
            elseif arrindex==1
                plot!(p,vcat(get(pvalues_dict,"InputScaling","..")[1:1:20],get(pvalues_dict,"InputScalingLarge","..")[1:1:20]),mean(collected_mean_errors,dims=2),label=nothing,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=false,xlabel=parameters_sweep_names[6],ylabel="", xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize, margin=(5mm),size=(720,480));#title=model_titles[arrindex],#,size=(5630,2010) );
            else
                plot!(p,vcat(get(pvalues_dict,"InputScaling","..")[1:1:20],get(pvalues_dict,"InputScalingLarge","..")[1:1:20]),mean(collected_mean_errors,dims=2),label=nothing,color=colour,linewidth=3,ribbon=std(collected_mean_errors,dims=2),fillalpha=0.3,legend=false,xlabel=parameters_sweep_names[6],ylabel="Mean NMSE",xtickfontsize=tickfontsize,ytickfontsize=tickfontsize,legendfontsize=legendfontsize, xlabelfontsize=labelfontsize,ylabelfontsize=labelfontsize,margin=(5mm),left_margin=15Plots.mm,size=(720,480));#title=model_titles[arrindex],#,size=(5630,2010) );
            end
            
            annotate!(p,vcat(get(pvalues_dict,"InputScaling","..")[1:1:20],get(pvalues_dict,"InputScalingLarge","..")[1:1:20])[1],[2.4],text(case_letters_IS[arrindex],20))
        end
        yticks_p=yticks(p)

        plot!(p,yticks=(yticks_p[1][1],string.(yticks_p[1][2])),xticks=([0.05,0.5,1.0,1.5,2.0],string.([0.05,0.5,1.0,1.5,2.0])))

        push!(subplot_collection_IScombined,p)
    end
end

psweep_idxs=collect(1:1:10)
parameters_to_plot=[ "KError"]#,"KErrorLarge", "OmegaError","OmegaErrorLarge", "ReservoirSize", "InputScaling", "KnowledgeRatio", "InputScalingLarge", "SpectralRadius","MeanDegree"]
subplot_collection
#Save supplementary figures S1-S7.
for i in psweep_idxs #the shifts of 20 and 10 here, (and specific ordering) will pick out the particular parameter sweep, and plot the synchronous, asynchronous and multifrequency models next to each other in that order.
    p=Plots.plot(subplot_collection[[i+20,i,i+10]]...,layout=(1,3),size=(2160,480),link=:y,bottom_margin=15Plots.mm,legend_position=(0.91,0.954))
    width, height = p.attr[:size]
    Plots.prepare_output(p)
    display(p)
    PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),"$(pwd())/Parameter_Error_Task/Figures/"*parameters_to_plot[i]*"_"*"task1_parametersweep"*"_"*model_type*"_mean_norm_error.pdf",width=width, height=height)
end

#save Figure 8: Parameter sweep of spectral radius and Input scaling combined.
for i in psweep_idxs[9]
    p=plot(vcat(subplot_collection[[i+20,i,i+10,]],subplot_collection_IScombined[[3,1,2]])...,layout=(2,3),size=(2160,960),link=:y,bottom_margin=15Plots.mm,legend_position=(0.91,0.954))
    width, height = p.attr[:size]
    Plots.prepare_output(p)
    display(p)
    PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),"$(pwd())/Parameter_Error_Task/Figures/"*"IS_and_SR_task1_parametersweep"*"_"*model_type*"_mean_norm_error.pdf",width=width,height=height)
end
