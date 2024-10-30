using Pkg; Pkg.activate(".")
using Plots, PlotlyJS, DelimtedFiles, Statistics

input_path="./" #location of the files output from Mutlistep_Reservoir_test.jl.

#load the vts from the csvs
vts_lorenz= readdlm(input_path*"vts_lorenz.csv",',')
vts= readdlm(input_path*"vts.csv",',')
vts_opt_sr_is_reg= readdlm(input_path*"vts_opt_sr_is_reg.csv",',')
vts_opt_sr_is_reg_constsr= readdlm(input_path*"vts_opt_sr_is_reg_constsr.csv",',')

plotlyjs()

#formatting
default(fontfamily="Helvetica")
marker_size=2.5
marker_alpha=0.6
markerstrokewidth=0
default(legendfontsize=12)
default(labelfontsize=16)
default(tickfontsize=16)

plot(mean(vts_lorenz,dims=1)[1,:],ylabel="t<sup>*</sup> (s)",xlabel="Internal steps",color=:blue,linewidth=3,label="Lorenz",size=(800,700),ribbon=(std(vts_lorenz,dims=1)[1,:],std(vts_lorenz,dims=1)[1,:]),fillalpha=0.2,xticks=([2,4,6,8,10],["2","4","6","8","10"]),yticks=([0,2,4,6,8],["0","2","4","6","8"]))
plot!(legend=:topright)
scatter!([[i for j in 1:30] for i in eachindex(eachcol(vts_lorenz))],vts_lorenz,label="",color=:blue,markersize=marker_size,markeralpha=marker_alpha,markerstrokewidth=markerstrokewidth)
plot!(mean(vts,dims=1)[1,:],color=:red,linewidth=3,label="Heteroclinic Cycles",ribbon=(std(vts,dims=1)[1,:],std(vts,dims=1)[1,:]),fillalpha=0.2)
scatter!([[i for j in 1:30] for i in eachindex(eachcol(vts))],vts,label="",color=:red,markersize=marker_size,markeralpha=marker_alpha,markerstrokewidth=markerstrokewidth)
plot!(mean(vts_opt_sr_is_reg,dims=1)[1,:],color=:green,linewidth=3,label="Heteroclinic Cycles. Opt",ribbon=(std(vts_opt_sr_is_reg,dims=1)[1,:],std(vts_opt_sr_is_reg,dims=1)[1,:]),fillalpha=0.2,linestyle=:dash)
scatter!([[i for j in 1:30] for i in eachindex(eachcol(vts_opt_sr_is_reg))],vts_opt_sr_is_reg,label="",color=:green,markersize=marker_size,markeralpha=marker_alpha,markerstrokewidth=markerstrokewidth)
plot!(mean(vts_opt_sr_is_reg_constsr,dims=1)[1,:],color=:purple,linewidth=3,label="Heteroclinic Cycles. Opt - constant effective SR",ribbon=(std(vts_opt_sr_is_reg_constsr,dims=1)[1,:],std(vts_opt_sr_is_reg_constsr,dims=1)[1,:]),fillalpha=0.2)
scatter!([[i for j in 1:30] for i in eachindex(eachcol(vts_opt_sr_is_reg_constsr))],vts_opt_sr_is_reg_constsr,label="",color=:purple,markersize=marker_size,markeralpha=marker_alpha,markerstrokewidth=markerstrokewidth)
savefig("Multistep_res_results_SansSerif.pdf")