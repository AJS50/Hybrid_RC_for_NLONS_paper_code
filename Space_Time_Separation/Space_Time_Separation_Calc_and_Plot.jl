include("$(pwd())/src/HybridRCforNLONS.jl")
using Plots, Arrow, DataFrames
input_path="$(pwd())/Residual_Physics_Task/Settings_and_GroundTruth/"
output_path="$(pwd())/Residual_Physics_Task/"

function xytophase(xys)
    N=Int64(size(xys)[1]/2)
    phases=atan.(xys[N+1:2*N,:],xys[1:N,:])
    return phases
end

function phasetoxy(phases)
    xys=reduce(vcat,vcat(cos.(phases),sin.(phases)))
    return xys
end

function distance(x,y)
    return sqrt(sum((x-y).^2))
end

#load ground truth data from residual physics task's biharmonic Kuramoto model and convert to xy components.
synch_ground_truth=permutedims(Matrix(DataFrame(Arrow.Table(input_path*"Biharmonic_Kuramoto_$("Synch")_ground_truth_data.arrow.lz4"))))
synch_ground_truth=[phasetoxy(synch_ground_truth[:,i]) for i in 1:size(synch_ground_truth,2)]
synch_ground_truth=reduce(hcat,synch_ground_truth)
sgt=permutedims(synch_ground_truth)

hc_ground_truth=permutedims(Matrix(DataFrame(Arrow.Table(input_path*"Biharmonic_Kuramoto_$("HeteroclinicCycles")_ground_truth_data.arrow.lz4"))))
hc_ground_truth=[phasetoxy(hc_ground_truth[:,i]) for i in 1:size(hc_ground_truth,2)]
hc_ground_truth=reduce(hcat,hc_ground_truth)
hgt=permutedims(hc_ground_truth)

scps_ground_truth=permutedims(Matrix(DataFrame(Arrow.Table(input_path*"Biharmonic_Kuramoto_$("SelfConsistentPartialSynchrony")_ground_truth_data.arrow.lz4"))))
scps_ground_truth=[phasetoxy(scps_ground_truth[:,i]) for i in 1:size(scps_ground_truth,2)]
scps_ground_truth=reduce(hcat,scps_ground_truth)
scpsgt=permutedims(scps_ground_truth)

async_ground_truth=permutedims(Matrix(DataFrame(Arrow.Table(input_path*"Biharmonic_Kuramoto_$("Asynch")_ground_truth_data.arrow.lz4"))))
async_ground_truth=[phasetoxy(async_ground_truth[:,i]) for i in 1:size(async_ground_truth,2)]
async_ground_truth=reduce(hcat,async_ground_truth)
agt=permutedims(async_ground_truth)

cases_gts=[sgt,agt,hgt,scpsgt]

## prepare plot.:
plotlyjs()
cases=["Synchronous","Asynchronous","Heteroclinic Cycles","Partial Synchrony"]
space_time_plots=Vector{Any}()
for (idx,case_gt) in enumerate(cases_gts)
    #every tenth point:
    time_distances_10th=Array{Float64,2}(undef,200,200)
    space_distances_10th=Array{Float64,2}(undef,200,200)
    for i in 1:200
        for j in 1:200
            space_distances_10th[i,j]=distance(case_gt[i*300,:],case_gt[j*300,:])
            time_distances_10th[i,j]=abs(i*300-j*300)*dt
        end
    end
    p=plot(scatter(time_distances_10th,space_distances_10th,size=(1400,800),color=:purple,markerstrokewidth=0.00,markersize=1),legend=false)
    plot!(p,xlabel="Time separation (s)",ylabel="Euclidean distance",title="$(cases[idx])")
    plot!(p,tickfontsize=12,titlefontsize=18,labelfontsize=15,xticks=([0,1000,2000,3000,4000,5000,6000],["0","1000","2000","3000","4000","5000","6000"]),yticks=([0,1,2,3,4,5,6],["0","1","2","3","4","5","6"]))
    push!(space_time_plots,p)
end

#plot and save.
p=plot(space_time_plots...,layout=(2,2),margin=5Plots.mm,bottom_margin=10Plots.mm)
width,height=p.attr[:size]
Plots.prepare_output(p)
PlotlyJS.savefig(Plots.plotlyjs_syncplot(p),output_path*"SpaceTimeSeparationPlots.pdf",width=width,height=height)