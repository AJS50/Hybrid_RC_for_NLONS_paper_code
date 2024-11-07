using Pkg; Pkg.activate(".")
using Plots, Statistics, CSV, Arrow, DelimitedFiles, DataFrames, Plots.Measures, Printf
plotlyjs() 

psweep_name="GridSearch"

num_reservoirs=40
arrayindex_range=1:8 #8 parameter sets.
test_num_range=1:20 #20 tests

#load all valid time data from the grid search (over regimes, parameter sets, and reservoir instances.) and take the mean over tests.
global all_data=Vector{Any}()
for ai in arrayindex_range
    for model_type in ["Standard","Hybrid"]
        input_path_base="$(pwd())/Residual_Physics_Task/GridSearch/"
        four_models=["Synch", "Asynch","HeteroclinicCycles","SelfConsistentPartialSynchrony"]
        for arrindex in [1,3,4] #across the three tested regimes. 1=Synch, 3=Heteroclinic, 4=SCPS
            global base_model=four_models[arrindex]
            #load in the valid times;
            val_times_this_ai=Array{Float64,2}(undef,length(test_num_range),num_reservoirs)
            val_times_this_ai[:,:]=Matrix(DataFrame(CSV.read(input_path_base*psweep_name*"_$(model_type)_"*"Biharmonic_Kuramoto"*"_"*base_model*"_valid_times_array_index_$(ai).csv",DataFrame)))
            mean_over_tests_val_times=mean(val_times_this_ai,dims=1)
            push!(all_data,mean_over_tests_val_times[1,:]) #collecting all valid time data together to plot below.
        end
        
    end
end

all_data
#split into regime specific data vectors.
#Data structure: all_data has 48 entries. 3 regimes x 2 model types x 8 parameter sets.
#they are interleaved a bit awkwardly.
sync_data=all_data[1:3:48];
HC_data=all_data[2:3:48];
SCPS_data=all_data[3:3:48];

#plot formatting.
fill_alpha=0.2
line_width=2
default(tickfontsize=20)
tickfontsize=20
default(titlefontsize=24)
title_fontsize=24
default(labelfontsize=24)
labelfontsize=24
default(legendfontsize=20)
legendfontsize=20

#print out max and min valid times for each model type and regime across parameter sets.
show_max_mins=false

#set of 0,1 tuples showing 0=low, 1=high for the three parameters explored here.
high_low_indicators=reduce(vcat,([i for i in Iterators.product((1,2),(1,2),(1,2))]))
high_low_letters_R=("R<sup>−</sup>","R<sup>+</sup>")
high_low_letters_S=("S<sup>−</sup>","S<sup>+</sup>")
high_low_letters_I=("I<sup>−</sup>","I<sup>+</sup>")
high_low_colors=(:black,:black) #color code the high and low states of the parameters?.
p_sync=scatter();
p_HC=scatter();
p_SCPS=scatter();
x_indices=collect(1:2:16)
#not actually jitter anymore. just points within a range of +-0.5, evenly spaced
x_indices_jitter=[(1.0.*collect(range(start=0,stop=1.0,length=40)).+i.-0.5) for i in x_indices]
#max and minimum valid times across the parameter sets.
if show_max_mins
    println("Standard reservoir maximum, Synchronous: ", maximum(maximum.(sync_data[x_indices])) )
    println("Hybrid reservoir maximum, Synchronous: ", maximum(maximum.(sync_data[x_indices.+1])))
    println("Standard reservoir maximum, Synchronous: ", maximum(maximum.(HC_data[x_indices])) )
    println("Hybrid reservoir maximum, Synchronous: ", maximum(maximum.(HC_data[x_indices.+1])) )
    println("Standard reservoir maximum, Synchronous: ", maximum(maximum.(SCPS_data[x_indices])) )
    println("Hybrid reservoir maximum, Synchronous: ", maximum(maximum.(SCPS_data[x_indices.+1])))
    println("Standard reservoir minimum, Synchronous: ", minimum(minimum.(sync_data[x_indices])) )
    println("Hybrid reservoir minimum, Synchronous: ", minimum(minimum.(sync_data[x_indices.+1])))
    println("Standard reservoir minimum, Synchronous: ", minimum(minimum.(HC_data[x_indices])) )
    println("Hybrid reservoir minimum, Synchronous: ", minimum(minimum.(HC_data[x_indices.+1])) )
    println("Standard reservoir minimum, Synchronous: ", minimum(minimum.(SCPS_data[x_indices])) )
    println("Hybrid reservoir minimum, Synchronous: ", minimum(minimum.(SCPS_data[x_indices.+1])))
end
#plot the results!
for (i,x) in enumerate(x_indices)
    if i==1
        scatter!(p_sync,x_indices_jitter[i],sync_data[x],color=:blue,label="Standard",ylims=(0.0,251.0),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>","<b>B</b>","<b>C</b>","<b>D</b>","<b>E</b>","<b>F</b>","<b>G</b>","<b>H</b>"]),ylabel=L"\textrm{Mean~}t^*\textrm{~(s)}",xlabel=L"\textrm{Parameter~set}",dpi=300);
        plot!(p_sync,x_indices_jitter[i],[mean(sync_data[x]) for i in eachindex(sync_data[x])],ribbon=std(sync_data[x]),color=:blue,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
        scatter!(p_sync,x_indices_jitter[i],sync_data[x+1],color=:red,label="Hybrid",ylims=(0.0,251.0),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>","<b>B</b>","<b>C</b>","<b>D</b>","<b>E</b>","<b>F</b>","<b>G</b>","<b>H</b>"]),yticks=([0,50,100,150,200,250],[L"0",L"50",L"100",L"150",L"200",L"250"]));
        plot!(p_sync,x_indices_jitter[i],[mean(sync_data[x+1]) for i in eachindex(sync_data[x+1])],ribbon=std(sync_data[x+1]),color=:red,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
        scatter!(p_HC,x_indices_jitter[i],HC_data[x],color=:blue,label="",ylims=(0.0,2.5),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>","<b>B</b>","<b>C</b>","<b>D</b>","<b>E</b>","<b>F</b>","<b>G</b>","<b>H</b>"]),ylabel="Mean t<sup>*</sup> (s)",xlabel="Parameter set",dpi=300);
        plot!(p_HC,x_indices_jitter[i],[mean(HC_data[x]) for i in eachindex(HC_data[x])],ribbon=std(HC_data[x]),color=:blue,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
        scatter!(p_HC,x_indices_jitter[i],HC_data[x+1],color=:red,label="",ylims=(0.0,2.5),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>","<b>B</b>","<b>C</b>","<b>D</b>","<b>E</b>","<b>F</b>","<b>G</b>","<b>H</b>"]),yticks=([0.0,0.5,1.0,1.5,2.0,2.5],["0","0.5","1.0","1.5","2.0","2.5"]));#,yticks=([0,1,2,3],[L"0",L"1",L"2",L"3"]));
        plot!(p_HC,x_indices_jitter[i],[mean(HC_data[x+1]) for i in eachindex(HC_data[x+1])],ribbon=std(HC_data[x+1]),color=:red,label=nothing,fillalpha=fill_alpha,linewidth=line_width); 
        scatter!(p_SCPS,x_indices_jitter[i],SCPS_data[x],color=:blue,label="",ylims=(0.0,2.5),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>","<b>B</b>","<b>C</b>","<b>D</b>","<b>E</b>","<b>F</b>","<b>G</b>","<b>H</b>"]),ylabel="Mean t<sup>*</sup> (s)",xlabel="Parameter set",dpi=300);
        plot!(p_SCPS,x_indices_jitter[i],[mean(SCPS_data[x]) for i in eachindex(SCPS_data[x])],ribbon=std(SCPS_data[x]),color=:blue,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
        scatter!(p_SCPS,x_indices_jitter[i],SCPS_data[x+1],color=:red,label="",ylims=(0.0,2.5),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>","<b>B</b>","<b>C</b>","<b>D</b>","<b>E</b>","<b>F</b>","<b>G</b>","<b>H</b>"]),yticks=([0.0,0.5,1.0,1.5,2.0,2.5],["0","0.5","1.0","1.5","2.0","2.5"]));#,yticks=([0,1,2,3],[L"0",L"1",L"2",L"3"]));
        plot!(p_SCPS,x_indices_jitter[i],[mean(SCPS_data[x+1]) for i in eachindex(SCPS_data[x+1])],ribbon=std(SCPS_data[x+1]),color=:red,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
    else
        scatter!(p_sync,x_indices_jitter[i],sync_data[x],color=:blue,label=nothing,ylims=(0.0,251.0),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>","<b>B</b>","<b>C</b>","<b>D</b>","<b>E</b>","<b>F</b>","<b>G</b>","<b>H</b>"]),ylabel="Mean t<sup>*</sup> (s)",xlabel="Parameter set",dpi=300);
        plot!(p_sync,x_indices_jitter[i],[mean(sync_data[x]) for i in eachindex(sync_data[x])],ribbon=std(sync_data[x]),color=:blue,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
        scatter!(p_sync,x_indices_jitter[i],sync_data[x+1],color=:red,label=nothing,ylims=(0.0,251.0),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>: "*high_low_letters_R[high_low_indicators[1][1]]*high_low_letters_S[high_low_indicators[1][2]]*high_low_letters_I[high_low_indicators[1][3]],"<b>B</b>: "*high_low_letters_R[high_low_indicators[2][1]]*high_low_letters_S[high_low_indicators[2][2]]*high_low_letters_I[high_low_indicators[2][3]],"<b>C</b>: "*high_low_letters_R[high_low_indicators[3][1]]*high_low_letters_S[high_low_indicators[3][2]]*high_low_letters_I[high_low_indicators[3][3]],"<b>D</b>: "*high_low_letters_R[high_low_indicators[4][1]]*high_low_letters_S[high_low_indicators[4][2]]*high_low_letters_I[high_low_indicators[4][3]],"<b>E</b>: "*high_low_letters_R[high_low_indicators[5][1]]*high_low_letters_S[high_low_indicators[5][2]]*high_low_letters_I[high_low_indicators[5][3]],"<b>F</b>: "*high_low_letters_R[high_low_indicators[6][1]]*high_low_letters_S[high_low_indicators[6][2]]*high_low_letters_I[high_low_indicators[6][3]],"<b>G</b>: "*high_low_letters_R[high_low_indicators[7][1]]*high_low_letters_S[high_low_indicators[7][2]]*high_low_letters_I[high_low_indicators[7][3]],"<b>H</b>: "*high_low_letters_R[high_low_indicators[8][1]]*high_low_letters_S[high_low_indicators[8][2]]*high_low_letters_I[high_low_indicators[8][3]]]),yticks=([0,50,100,150,200,250],["0","50","100","150","200","250"]));
        plot!(p_sync,x_indices_jitter[i],[mean(sync_data[x+1]) for i in eachindex(sync_data[x+1])],ribbon=std(sync_data[x]),color=:red,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
        scatter!(p_HC,x_indices_jitter[i],HC_data[x],color=:blue,label=nothing,ylims=(0.0,2.5),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>","<b>B</b>","<b>C</b>","<b>D</b>","<b>E</b>","<b>F</b>","<b>G</b>","<b>H</b>"]),ylabel="Mean t<sup>*</sup> (s)",xlabel="Parameter set",dpi=300);
        plot!(p_HC,x_indices_jitter[i],[mean(HC_data[x]) for i in eachindex(HC_data[x])],ribbon=std(HC_data[x]),color=:blue,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
        scatter!(p_HC,x_indices_jitter[i],HC_data[x+1],color=:red,label=nothing,ylims=(0.0,2.5),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>: "*high_low_letters_R[high_low_indicators[1][1]]*high_low_letters_S[high_low_indicators[1][2]]*high_low_letters_I[high_low_indicators[1][3]],"<b>B</b>: "*high_low_letters_R[high_low_indicators[2][1]]*high_low_letters_S[high_low_indicators[2][2]]*high_low_letters_I[high_low_indicators[2][3]],"<b>C</b>: "*high_low_letters_R[high_low_indicators[3][1]]*high_low_letters_S[high_low_indicators[3][2]]*high_low_letters_I[high_low_indicators[3][3]],"<b>D</b>: "*high_low_letters_R[high_low_indicators[4][1]]*high_low_letters_S[high_low_indicators[4][2]]*high_low_letters_I[high_low_indicators[4][3]],"<b>E</b>: "*high_low_letters_R[high_low_indicators[5][1]]*high_low_letters_S[high_low_indicators[5][2]]*high_low_letters_I[high_low_indicators[5][3]],"<b>F</b>: "*high_low_letters_R[high_low_indicators[6][1]]*high_low_letters_S[high_low_indicators[6][2]]*high_low_letters_I[high_low_indicators[6][3]],"<b>G</b>: "*high_low_letters_R[high_low_indicators[7][1]]*high_low_letters_S[high_low_indicators[7][2]]*high_low_letters_I[high_low_indicators[7][3]],"<b>H</b>: "*high_low_letters_R[high_low_indicators[8][1]]*high_low_letters_S[high_low_indicators[8][2]]*high_low_letters_I[high_low_indicators[8][3]]]),yticks=([0.0,0.5,1.0,1.5,2.0,2.5],["0","0.5","1.0","1.5","2.0","2.5"]));#,yticks=([0,1,2,3],[L"0",L"1",L"2",L"3"]));
        plot!(p_HC,x_indices_jitter[i],[mean(HC_data[x+1]) for i in eachindex(HC_data[x+1])],ribbon=std(HC_data[x+1]),color=:red,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
        scatter!(p_SCPS,x_indices_jitter[i],SCPS_data[x],color=:blue,label=nothing,ylims=(0.0,2.5),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>","<b>B</b>","<b>C</b>","<b>D</b>","<b>E</b>","<b>F</b>","<b>G</b>","<b>H</b>"]),ylabel="Mean t<sup>*</sup> (s)",xlabel="Parameter set",dpi=300);
        plot!(p_SCPS,x_indices_jitter[i],[mean(SCPS_data[x]) for i in eachindex(SCPS_data[x])],ribbon=std(SCPS_data[x]),color=:blue,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
        scatter!(p_SCPS,x_indices_jitter[i],SCPS_data[x+1],color=:red,label=nothing,ylims=(0.0,2.5),markeralpha=0.7,markersize=5,markerstrokewidth=0.0,minorgrid=true,size=(2160,480),xticks=([1,3,5,7,9,11,13,15],["<b>A</b>: "*high_low_letters_R[high_low_indicators[1][1]]*high_low_letters_S[high_low_indicators[1][2]]*high_low_letters_I[high_low_indicators[1][3]],"<b>B</b>: "*high_low_letters_R[high_low_indicators[2][1]]*high_low_letters_S[high_low_indicators[2][2]]*high_low_letters_I[high_low_indicators[2][3]],"<b>C</b>: "*high_low_letters_R[high_low_indicators[3][1]]*high_low_letters_S[high_low_indicators[3][2]]*high_low_letters_I[high_low_indicators[3][3]],"<b>D</b>: "*high_low_letters_R[high_low_indicators[4][1]]*high_low_letters_S[high_low_indicators[4][2]]*high_low_letters_I[high_low_indicators[4][3]],"<b>E</b>: "*high_low_letters_R[high_low_indicators[5][1]]*high_low_letters_S[high_low_indicators[5][2]]*high_low_letters_I[high_low_indicators[5][3]],"<b>F</b>: "*high_low_letters_R[high_low_indicators[6][1]]*high_low_letters_S[high_low_indicators[6][2]]*high_low_letters_I[high_low_indicators[6][3]],"<b>G</b>: "*high_low_letters_R[high_low_indicators[7][1]]*high_low_letters_S[high_low_indicators[7][2]]*high_low_letters_I[high_low_indicators[7][3]],"<b>H</b>: "*high_low_letters_R[high_low_indicators[8][1]]*high_low_letters_S[high_low_indicators[8][2]]*high_low_letters_I[high_low_indicators[8][3]]]),yticks=([0.0,0.5,1.0,1.5,2.0,2.5],["0","0.5","1.0","1.5","2.0","2.5"]));#,yticks=([0,1,2,3],[L"0",L"1",L"2",L"3"]));
        plot!(p_SCPS,x_indices_jitter[i],[mean(SCPS_data[x+1]) for i in eachindex(SCPS_data[x+1])],ribbon=std(SCPS_data[x+1]),color=:red,label=nothing,fillalpha=fill_alpha,linewidth=line_width);
    end
end
plot!(p_sync,tickfontsize=tickfontsize,ylabelfontsize=labelfontsize,titlefontsize=title_fontsize,title="Synchronous",xlabelfontsize=labelfontsize,legend=:false,legendfontsize=legendfontsize,bottom_margin=14Plots.mm,left_margin=15Plots.mm,reshape(repeat(x_indices.+1,inner=2),2,8),reshape(repeat([0,350.0],8),2,8),color=:black,linestyle=:dash,label=nothing); 
plot!(p_HC,tickfontsize=tickfontsize,ylabelfontsize=labelfontsize,titlefontsize=title_fontsize,title="Heteroclinic Cycles",xlabelfontsize=labelfontsize,legend=:false,legendfontsize=legendfontsize,bottom_margin=14Plots.mm,left_margin=15Plots.mm,reshape(repeat(x_indices.+1,inner=2),2,8),reshape(repeat([0,350.0],8),2,8),color=:black,linestyle=:dash,label=nothing) ;
plot!(p_SCPS,tickfontsize=tickfontsize,ylabelfontsize=labelfontsize,titlefontsize=title_fontsize,title="Partial Synchrony",xlabelfontsize=labelfontsize,legend=:topright,legendfontsize=legendfontsize,bottom_margin=14Plots.mm,left_margin=15Plots.mm,reshape(repeat(x_indices.+1,inner=2),2,8),reshape(repeat([0,350.0],8),2,8),color=:black,linestyle=:dash,label=nothing) ;

#Figure 10 grid search plot and save:
combined_plot=plot((p_sync,p_HC,p_SCPS)...,layout=(3,1),size=(2000,1540))
p=combined_plot
width,height=p.attr[:size]
Plots.prepare_output(p)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),"$(pwd())/Residual_Physics_Task/Figures/residual_physics_grid_search.pdf",width=width,height=height)

##plot the regime specific results individually:
# display(plot(p_sync))
# display(plot(p_HC))
# display(plot(p_SCPS))
# savefig(p_sync,"/Users/as15635/Documents/Projects/KnowledgeReservoirs2/test/ExtendedKuramoto/Local_Paper_figures/Waterfall_Synchronous.pdf")
# savefig(p_HC,"/Users/as15635/Documents/Projects/KnowledgeReservoirs2/test/ExtendedKuramoto/Local_Paper_figures/Waterfall_HC.pdf")
# savefig(p_SCPS,"/Users/as15635/Documents/Projects/KnowledgeReservoirs2/test/ExtendedKuramoto/Local_Paper_figures/Waterfall_SCPS.pdf")
