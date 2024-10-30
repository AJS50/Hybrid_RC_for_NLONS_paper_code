using Pkg; Pkg.activate(".")
include("$(pwd())/src/HybridRCforNLONS.jl")
import .HybridRCforNLONS: cartesian_kuramoto, cartesian_kuramoto_ic, generate_ground_truth_data, generate_arrow
using OrdinaryDiffEq, Random, CSV, Arrow, DataFrames

#read in command line arguments
idx=1
idx=parse(Int64,ARGS[1]) #read in index defining the ground truth model regime: 1= regime 2=asynchronous, 2=regime 7=multi-frequency, 3=regime16=synchronous 
input_path="./"
input_path=ARGS[2]
output_path="./"
output_path=ARGS[3] #read in absolute path to store results in.

N=5 #number of oscillators
tspan=(0.0,6200.0) #one large trajectory for each model will be generated. To be split into training, warmup and test segments as required.
dt=0.1 #time step.
models=[["2","2_1","2_2"],["7","7_1","7_2"],["16","16_1","16_2"]] #three instantiations of each model type.
model_names=["asynchronous","multi-frequency","synchronous"]
i=[2,7,16][idx] #the three model types.
for j in 1:3
    model=models[idx][j]
    println("model=",model," running ODE sim")
    rng=MersenneTwister(1234+i+j-1)
    println("rng: ",rng)

    #set parameters. First instance of each regime was found from its array index. 
    if j==1
        settings=readdlm(input_path*psweep_name*"_sweep_settings.csv",',',header=true)[1]
        N,K,model,μ,Δω,res_size,scaling,knowledge_ratio,data_dim,model_dim,spectral_radius,mean_degree,dt, K_err, omega_err=settings[1,:] 
        base_params=cartesian_kuramoto_p(rng,N,μ,Δω,K)
    else
    #the other two are specified to match the quality of the regime manually, but with parameters still sampled randomly.
        base_params=ones(Float64,6)
        if i == 2 # - Model 2: Nat Freqs within [-1,+1], K=1.0 (asynch)
            base_params[1:5]=2.0.*rand(rng,5).-1.0
            base_params[6]=1.0 #set K.
        elseif i == 7 # - Model 7: 4 Nat Freqs within [-1,+1], with outlier >3Hz.  K=2.0, (phase locked with high frequency)
            base_params[1:4]=2.0.*rand(rng,4).-1.0
            x=rand()-0.5
            base_params[5]=(rand(rng).+3.0)*sign(x)
            base_params[6]=2.0 #set K.
        elseif i == 16 # - Model 16: 4 Nat Freqs within [-1,+1] K=4.0 (synchronised)
            base_params[1:5]=2.0.*rand(rng,5).-1.0
            base_params[6]=4.0 #set K.
        end
    end
    println("params: ",base_params)
    #generate ground truth data
    #cartesian_kuramoto is in phase-component space. so with N=5, the state vector is 10-dimensional. 
    #arranged such that 1:5 are the x components and 6:10 are the y components.
    ic=cartesian_kuramoto_ic(N) #same initial conditions for every run (internally this uses a MersenneTwister rng with seed 1234)
    gt_data=generate_ground_truth_data(cartesian_kuramoto,ic,base_params,tspan,1e7,dt)
    gt_data=permutedims(reduce(hcat,gt_data.u))

    #write to csv and compress as arrow file.
    name="Model_$(model_names[idx])_$(j-1)_ground_truth_data"
    println("writing csv")
    CSV.write(output_path*name*".csv",DataFrame(gt_data,:auto),writeheader=true)
    println("writing arrow")
    generate_arrow(name,output_path)
    println("deleting csv")
    rm(output_path*name*".csv")
end

#test output: open data and plot
using Plots
j=0 #model instantiation number
gt_read=Matrix(DataFrame(Arrow.Table(output_path*"Model_$(model_names[idx])_$(j)_ground_truth_data.arrow.lz4")))
plot(gt_read[1:1000,1:5],label="",color=:blue,ylims=(-1.0,1.0))
plot!(gt_read[1:1000,6:10],label="",color=:red)

