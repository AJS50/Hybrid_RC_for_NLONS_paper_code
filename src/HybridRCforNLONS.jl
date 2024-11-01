module HybridRCforNLONS

    using Distributions, Statistics,  Random, LinearAlgebra, OrdinaryDiffEq, Arrow, CSV, DataFrames

    #dynamical systems
    export lorenz, lorenz_ic, lorenz_p, lotka, lotka_ic, sin_2d, sin_2d_ic, cartesian_kuramoto, cartesian_kuramoto_ic, cartesian_kuramoto_p, biharmonic_kuramoto_ic, biharmonic_kuramoto_p, biharmonic_kuramoto, cartesian_biharmonic_kuramoto
    #reservoirs and reservoir state modifiers
    export xytophase, phasetoxy, sqr_even_indices,ESN, Hybrid_ESN, update_state!, initialise_reservoir!, ingest_data!, train_reservoir!, compute_output, predict!, ModelStep!
    #evaluation methods
    export normalised_error, valid_time
    #data processing
    export generate_ground_truth_data,generate_ODE_data_task1,generate_ODE_data_task2, generate_arrow, generate_ODE_data_ExtKuramoto
    #biharmonic kuramoto callbacks and order funcs:
    export reset_condition1, reset_affect1!, reset_condition2, reset_affect2!, complex_form, kuramoto_order2, get_order_params

    """
    standard 3D Lorenz dynamical system. Chaotic with parameters: σ=10, ρ=28, β=8/3. du[1]=σ(u[2]-u[1]), du[2]=u[1](ρ-u[3])-u[2], du[3]=u[1]u[2]-βu[3] 
    """
    function lorenz(du,u,p,t)
        σ=p[1]
        ρ=p[2]
        β=p[3]
        du[1]=σ*(u[2]-u[1])
        du[2]=u[1]*(ρ-u[3])-u[2]
        du[3]=u[1]*u[2]-β*u[3]
    end

    """
    returns default initial conditions for lorenz system [1.0,0.0,25.0]. If random=true, then returns initial conditions samples from range[1] to range[2] for each dimension.
    """
    function lorenz_ic(range::Vector{Float64};random=false)
        if random
            return [range[1]+(range[2]-range[1])*rand(rng) for i in 1:3]
        else
            return [1.0,0.0,25.0]
        end
    end

    """
    returns default lorenz parameter set [σ,ρ,β] = [10.0,28.0,8/3]
    """
    function lorenz_p()
        return [10.0,28.0,8/3]
    end

    function cartesian_kuramoto(du,u,p,t)
        N=Int64(length(u)/2)
        xs=u[1:N]
        ys=u[N+1:end]
        ωs=p[1:N]
        K=p[N+1]
        for i in 1:N
            #x components
            du[i]=-ωs[i]*ys[i] - (K/N)*ys[i]*sum([ys[j]*xs[i]-xs[j]*ys[i] for j in 1:N])
            #y components
            du[i+N]=ωs[i]*xs[i] + (K/N)*xs[i]*sum([ys[j]*xs[i]-xs[j]*ys[i] for j in 1:N])
        end
    end

    """
    Returns default seeded initial phase components using internal random number generator (MersenneTwister(), seed 1234). Phases θ_i, i=1,...,N sampled by default on ~U(-π ,π) then x, y components calculated as cos(θ), sin(θ) respectively. When random=true: the range specified here (range∈[0,1]^2) will be transformed to specify a range of initial phases within [-π,π]. 
    """
    function cartesian_kuramoto_ic(N::Int64;range::Vector{Float64}=[0.0,0.5],random=false)
        #use same random seed for initial conditions of ground truth data. the parameters will vary. but should start from the same point as variability in init' cond's across the 20 test segments will provide the variation. 
        if random
            upper=range[2]*(2*π)-π
            lower=range[1]*(2*π)-π
            return [lower+(upper-lower)*rand(rng) for i in 1:N]
        else
            ic_rng=Random.MersenneTwister(1234)
            θs=[2*π*rand(ic_rng)-π for i in 1:N]
            return vcat(cos.(θs),sin.(θs))
        end
    end

    function cartesian_kuramoto_p(rng,N::Int64,μ::Float64, Δω::Float64,K::Float64)
        ωs=lorentzian_nat_freqs(N,μ,Δω,rng)
        return [ωs...,K]
    end

    function lorentzian_nat_freqs(N,μ,Δω,rng) #generate N lorentzian distributed natural frequencies
        location=μ
        width=Δω
        L=Cauchy(location,width)
        ωs=Vector(rand(rng,L,N))
        return(ωs)
    end

    """
    standard Lotka-Volterra predator-prey dynamical system. du[1]=αu[1]-βu[1]u[2], du[2]=δu[1]u[2]-γu[2]
    """
    function lotka(du,u,p,t)
        α=p[1]
        β=p[2]
        γ=p[3]
        δ=p[4]
        du[1]=α*u[1]-β*u[1]*u[2]
        du[2]=δ*u[1]*u[2]-γ*u[2]
    end


    """
    returns default initial conditions for lotka_volterra system [0.1,0.5]. If random=true, then returns initial conditions sampled from range[1] to range[2] for each dimension.
    """
    function lotka_ic(range::Vector{Float64};random=false)
        if random
            return [range[1]+(range[2]-range[1])*rand(rng) for i in 1:2]
        else
            return [0.1,0.5]
        end
    end

    """
    returns default lotka parameter set [α,β,γ,δ] = [1.0,1.0,1.0,1.0]
    """
    function lotka_p()
        return [1.0,1.0,1.0,1.0]
    end

    function biharmonic_kuramoto_ic(N::Int64;range::Vector{Float64}=[0.0,0.5],random=false)
        #use same random seed for initial conditions of ground truth data. the parameters will vary. but should start from the same point as variability in init' cond's across the 20 test segments will provide the variation. 
        if random
            upper=range[2]*(2*π)-π
            lower=range[1]*(2*π)-π
            return [lower+(upper-lower)*rand(rng) for i in 1:N]
        else
            ic_rng=Random.MersenneTwister(1234)
            θs=[2*π*rand(ic_rng)-π for i in 1:N]
            return θs
        end
    end

    function biharmonic_kuramoto_p(rng,N::Int64,μ::Float64, Δω::Float64,K::Float64,a::Float64,γ_1::Float64,γ_2::Float64)
        ωs=lorentzian_nat_freqs(N,μ,Δω,rng)
        return [ωs...,K,a,γ_1,γ_2]
    end

    function biharmonic_kuramoto(du,u,p,t)
        N=Int64(size(u,1))
        θs=u[1:N]
        ωs=p[1:N]
        K=p[N+1]
        a=p[N+2]
        γ_1=p[N+3]
        γ_2=p[N+4]
        for i in 1:N
            du[i]=ωs[i] + (K/N)*sum([(sin(θs[j]-θs[i]+γ_1)+a*sin(2*θs[j]-2*θs[i]+γ_2)) for j in 1:N])
        end
    end

    ### STANDARD RESERVOIR IMPLEMENTATION



    """
    Non linear function applied to reservoir states before ouput layer.
        This squares the value of all even index states, and leaves the odd index states unchanged.
    """
    function sqr_even_indices(x)
        y=copy(x)
        for i in 2:2:size(y)[1]
            y[i,:]=y[i,:].^2
        end
        return(y)
    end

    function xytophase(xys)
        N=Int64(size(xys)[1]/2)
        phases=atan.(xys[N+1:2*N,:],xys[1:N,:])
        return phases
    end

    function phasetoxy(phases)
        xys=reduce(vcat,vcat(cos.(phases),sin.(phases)))
        return xys
    end

    mutable struct ESN
        res_size::Int64
        mean_degree::Float64
        data_length::Int64
        spectral_radius::Float64
        input_scaling::Float64
        state_history::Matrix{Float64}
        current_state::Vector{Float64}
        input_weight_matrix::Matrix{Float64}
        output_weight_matrix::Matrix{Float64}
        reservoir_weight_matrix::Matrix{Float64}
        g::Float64 #scaling similar to spectral radius but for entire update function as per "Inubushi et al 2021" ReservoirComputing textbook.
        regularisation_strength::Float64
        NLAT::Function #nonlinear activation function between reservoir state and output computation
        prediction_state_history::Matrix{Float64}

        function ESN(res_size::Int64, mean_degree::Int64, data_length::Int64, spectral_radius::Float64, input_scaling::Float64, g::Float64, regularisation_strength::Float64, NLAT::Function)
            new(res_size,
                mean_degree,
                data_length, 
                spectral_radius, 
                input_scaling, 
                zeros(Float64, res_size, data_length), 
                Vector{Float64}(undef, res_size), 
                zeros(res_size, data_length), 
                Matrix{Float64}(undef, data_length, res_size), 
                Matrix{Float64}(undef, res_size, res_size), 
                g, 
                regularisation_strength, 
                NLAT, 
                Matrix{Float64}(undef,res_size,1)
            )
        end
    end

    function update_state!(reservoir::ESN,input::Vector{Float64})
        reservoir.current_state=tanh.(reservoir.g.*(reservoir.reservoir_weight_matrix*reservoir.current_state.+reservoir.input_weight_matrix*input))
    end

    function initialise_reservoir!(rng,reservoir::ESN)
        #create reservoir weight matrix:
        reservoir.reservoir_weight_matrix=zeros(reservoir.res_size,reservoir.res_size)
        edge_probability=reservoir.mean_degree/reservoir.res_size

        decision_samples=rand(rng,reservoir.res_size,reservoir.res_size)

        for i in 1:reservoir.res_size
            for j in 1:reservoir.res_size
                if decision_samples[i,j]<=edge_probability
                    #weights uniform sample ∈ [-1,1]
                    reservoir.reservoir_weight_matrix[i,j]=2*rand(rng)-1
                end
            end
        end

        #get current spectral radius
        res_ρ=maximum(abs.(eigvals(reservoir.reservoir_weight_matrix)))
        #scale to desired spectral radius
        reservoir.reservoir_weight_matrix .*= reservoir.spectral_radius/res_ρ
        
        #create input weight matrix
        for i in 1:reservoir.res_size
            reservoir.input_weight_matrix[i,rand(rng,1:reservoir.data_length)]=reservoir.input_scaling*(2*rand(rng)-1)
        end

        #reset state history
        reservoir.state_history=Matrix{Float64}(undef,reservoir.res_size,1)
        #reset current state
        reservoir.current_state=zeros(reservoir.res_size)
    end

    # function ingest_data!(reservoir::ESN,data::Matrix{Float64}) #for training or warmup... sets state history size to length of data being processed.
    #     number_of_data_points=size(data,2)
    #     reservoir.state_history=Matrix{Float64}(undef,reservoir.res_size,number_of_data_points)
    #     reservoir.current_state=zeros(reservoir.res_size)
    #     reservoir.state_history[:,1]=reservoir.current_state
    #     for i in 1:number_of_data_points
    #         update_state!(reservoir,data[:,i])
    #         reservoir.state_history[:,i]=reservoir.current_state
    #     end
    # end

    function ingest_data!(reservoir::ESN,data::Matrix{Float64}) #for training or warmup... sets state history size to length of data being processed.
        number_of_data_points=size(data,2)
        reservoir.state_history=Matrix{Float64}(undef,reservoir.res_size,number_of_data_points)
        reservoir.current_state=zeros(reservoir.res_size)
        reservoir.state_history[:,1]=reservoir.current_state
        for i in 2:number_of_data_points
            update_state!(reservoir,data[:,i])
            reservoir.state_history[:,i]=reservoir.current_state
        end
    end

    function train_reservoir!(reservoir::ESN,target_data::Matrix{Float64})
        #get res states post non linear transform:
        reservoir_states=reservoir.NLAT(reservoir.state_history)
        #compute output weight matrix
        reservoir.output_weight_matrix=((reservoir_states*reservoir_states' + reservoir.regularisation_strength*I)\(reservoir_states*target_data'))'
    end

    function compute_output(reservoir::ESN)
        return reservoir.output_weight_matrix*reservoir.NLAT(reservoir.current_state)
    end

    function predict!(reservoir::ESN,num_steps,save_states=false,phase=true)
        prediction=Matrix{Float64}(undef,reservoir.data_length,num_steps)
        if save_states
            reservoir.prediction_state_history=Matrix{Float64}(undef,reservoir.res_size,num_steps)
        end
        #runs from current state (end of state history) onwards
        if phase
            prediction[:,1]=phasetoxy(xytophase(compute_output(reservoir)))
            for i in 2:num_steps
                update_state!(reservoir,phasetoxy(xytophase(prediction[:,i-1])))
                prediction[:,i]=phasetoxy(xytophase(compute_output(reservoir)))
                if save_states
                    reservoir.prediction_state_history[:,i]=reservoir.current_state
                end
            end
        else
            prediction[:,1]=compute_output(reservoir)
            for i in 2:num_steps
                update_state!(reservoir,prediction[:,i-1])
                prediction[:,i]=compute_output(reservoir)
                if save_states
                    reservoir.prediction_state_history[:,i]=reservoir.current_state
                end
            end
        end
        return(prediction)
    end


    ### HYBRID RESERVOIR IMPLEMENTATIONS

    mutable struct Hybrid_ESN
        res_size::Int64
        mean_degree::Float64
        data_length::Int64
        model_dim::Int64
        knowledge_ratio::Float64
        spectral_radius::Float64
        input_scaling::Float64
        state_history::Matrix{Float64}
        model_state_history::Matrix{Float64}
        current_state::Vector{Float64}
        current_model_state::Vector{Float64}
        input_weight_matrix::Matrix{Float64}
        output_weight_matrix::Matrix{Float64}
        reservoir_weight_matrix::Matrix{Float64}
        g::Float64 #scaling similar to spectral radius but for entire update function as per "Inubushi et al 2021" ReservoirComputing textbook.
        regularisation_strength::Float64
        NLAT::Function #nonlinear activation function between reservoir state and output computation
        prediction_state_history::Matrix{Float64}
        prediction_model_state_history::Matrix{Float64}
        model::Function
        model_parameters::Vector{Float64}
        dt::Float64

        function Hybrid_ESN(res_size::Int64, mean_degree::Int64, data_length::Int64, model_dim::Int64, knowledge_ratio::Float64, spectral_radius::Float64, input_scaling::Float64, g::Float64, regularisation_strength::Float64, NLAT::Function,model::Function,model_parameters::Vector{Float64},dt::Float64)
            new(res_size,
                mean_degree,
                data_length, 
                model_dim,
                knowledge_ratio,
                spectral_radius, 
                input_scaling, 
                zeros(Float64, res_size, data_length),
                zeros(Float64,model_dim,data_length),
                Vector{Float64}(undef, res_size), 
                Vector{Float64}(undef, model_dim),
                zeros(res_size, data_length+model_dim), 
                Matrix{Float64}(undef, data_length, res_size+model_dim), 
                Matrix{Float64}(undef, res_size, res_size), 
                g, 
                regularisation_strength, 
                NLAT, 
                Matrix{Float64}(undef,res_size,1),
                Matrix{Float64}(undef,model_dim,1),
                model,
                deepcopy(model_parameters),
                dt
            )
        end
    end

    function update_state!(reservoir::Hybrid_ESN,input::Vector{Float64})
        reservoir.current_state=tanh.(reservoir.g.*(reservoir.reservoir_weight_matrix*reservoir.current_state.+reservoir.input_weight_matrix*input))
    end

    function ModelStep!(reservoir::Hybrid_ESN)
        prob=ODEProblem(reservoir.model,reservoir.current_model_state,(0.0,reservoir.dt),reservoir.model_parameters)
        sol=solve(prob,Tsit5())
        return(sol.u[end])
    end


    function initialise_reservoir!(rng,reservoir::Hybrid_ESN)
        #create reservoir weight matrix:
        reservoir.reservoir_weight_matrix=zeros(reservoir.res_size,reservoir.res_size)
        edge_probability=reservoir.mean_degree/reservoir.res_size

        decision_samples=rand(rng,reservoir.res_size,reservoir.res_size)

        for i in 1:reservoir.res_size
            for j in 1:reservoir.res_size
                if decision_samples[i,j]<=edge_probability
                    #weights uniform sample ∈ [-1,1]
                    reservoir.reservoir_weight_matrix[i,j]=2*rand(rng)-1
                end
            end
        end

        #get current spectral radius
        res_ρ=maximum(abs.(eigvals(reservoir.reservoir_weight_matrix)))
        #scale to desired spectral radius
        reservoir.reservoir_weight_matrix .*= reservoir.spectral_radius/res_ρ
        
        #create input weight matrix
        decision_samples=rand(rng,reservoir.res_size)
        for (idx,sample) in enumerate(decision_samples)
            if sample <= reservoir.knowledge_ratio #KR % of connections are from the model. (which is concatentated to the input BEFORE the data, hence the model connections being made on [i,1:model_dim])
                reservoir.input_weight_matrix[idx,rand(1:reservoir.model_dim)]=reservoir.input_scaling*(2*rand(rng)-1)
            else
                reservoir.input_weight_matrix[idx,reservoir.model_dim+rand(rng,1:reservoir.data_length)]=reservoir.input_scaling*(2*rand(rng)-1)
            end
        end

        #reset state histories
        reservoir.state_history=Matrix{Float64}(undef,reservoir.res_size,1)
        reservoir.model_state_history=Matrix{Float64}(undef,reservoir.model_dim,1)
        #reset current state
        reservoir.current_state=zeros(reservoir.res_size)
        reservoir.current_model_state=zeros(reservoir.model_dim)

    end

    function ingest_data!(reservoir::Hybrid_ESN,data::Matrix{Float64}) #for training or warmup... sets state history size to length of data being processed.
        number_of_data_points=size(data,2)
        reservoir.state_history=Matrix{Float64}(undef,reservoir.res_size,number_of_data_points)
        reservoir.model_state_history=Matrix{Float64}(undef,reservoir.model_dim,number_of_data_points)
        reservoir.current_state=zeros(reservoir.res_size)
        reservoir.state_history[:,1]=reservoir.current_state
        for (i,instance) in enumerate(eachcol(data))
            reservoir.current_model_state=instance
            reservoir.model_state_history[:,i]=ModelStep!(reservoir)
        end

        combined_input_data=vcat(reservoir.model_state_history,data)
        for i in 1:number_of_data_points
            update_state!(reservoir,combined_input_data[:,i])
            reservoir.state_history[:,i]=reservoir.current_state
        end
    end

    function train_reservoir!(reservoir::Hybrid_ESN,target_data::Matrix{Float64})
        #get res states post non linear transform:
        reservoir_states=reservoir.NLAT(reservoir.state_history)
        augmented_states=vcat(reservoir.model_state_history,reservoir_states)
        #compute output weight matrix
        reservoir.output_weight_matrix=((augmented_states*augmented_states' + reservoir.regularisation_strength*I)\(augmented_states*target_data'))'
    end

    function compute_output(reservoir::Hybrid_ESN)
        return reservoir.output_weight_matrix*vcat(reservoir.current_model_state,reservoir.NLAT(reservoir.current_state))
    end

    function predict!(reservoir::Hybrid_ESN,num_steps,save_states=false,phase=true)
        prediction=Matrix{Float64}(undef,reservoir.data_length,num_steps)
        if save_states
            reservoir.prediction_state_history=Matrix{Float64}(undef,reservoir.res_size,num_steps)
            reservoir.prediction_state_history[:,1]=reservoir.current_state
            reservoir.prediction_model_state_history=Matrix{Float64}(undef,reservoir.model_dim,num_steps)
            reservoir.prediction_model_state_history[:,1]=reservoir.current_model_state
        end
        #runs from current state (end of state history) onwards
        if phase
            prediction[:,1]=phasetoxy(xytophase(compute_output(reservoir)))
            for i in 2:num_steps
                reservoir.current_model_state=phasetoxy(xytophase(prediction[:,i-1]))
                reservoir.current_model_state=phasetoxy(xytophase(ModelStep!(reservoir)))
                update_state!(reservoir,vcat(phasetoxy(xytophase(reservoir.current_model_state)),phasetoxy(xytophase(prediction[:,i-1]))))
                prediction[:,i]=phasetoxy(xytophase(compute_output(reservoir)))

                if save_states
                    reservoir.prediction_state_history[:,i-1]=reservoir.current_state
                    reservoir.prediction_model_state_history[:,i-1]=reservoir.current_model_state
                end
            end
        else
            prediction[:,1]=compute_output(reservoir)
            for i in 2:num_steps
                reservoir.current_model_state=prediction[:,i-1]
                reservoir.current_model_state=ModelStep!(reservoir)
                update_state!(reservoir,vcat(reservoir.current_model_state,prediction[:,i-1]))
                prediction[:,i]=compute_output(reservoir)
        
                if save_states
                    reservoir.prediction_state_history[:,i-1]=reservoir.current_state
                    reservoir.prediction_model_state_history[:,i-1]=reservoir.current_model_state
                end
            end
        end
        return(prediction)
    end

    ###EVALUATION FUNCTIONS

    function normalised_error(u_pred,u_target)
        l2_norm=sqrt.(sum((u_target.-u_pred).^2,dims=1))
        n=size(u_target,2)
        norm_error=l2_norm./sqrt((1/n).*sum(sum(u_target.^2,dims=1)))
        return norm_error
    end

    function valid_time(threshold,u_pred,u_target,dt)
        norm_error=normalised_error(u_pred,u_target)
        valid_time_index=findfirst(x->(x>threshold),norm_error)
        if valid_time_index===nothing
            valid_time_index=CartesianIndex(1,length(norm_error))
        end
        tsteps=range(0,length=size(u_target,2),step=dt)
        return tsteps[valid_time_index[2]]
    end

    ### DATA PROCESSING FUNCTIONS

    function generate_ground_truth_data(system::Function,ic::Vector{Float64},p,tspan::Tuple{Float64,Float64},maxiters::Float64,dt::Float64)
        prob=ODEProblem(system,ic,tspan,p)
        sol=solve(prob,alg=Tsit5(),saveat=dt,adaptive=false,dt=1/1024,maxiters=maxiters)
        return sol
    end

    #for parameter error task
    function generate_ODE_data_task1(system::Function,ic::Vector{Float64},p,tspan::Tuple{Float64,Float64},maxiters::Float64,dt::Float64)
        prob=ODEProblem(system,ic,tspan,p)
        sol=solve(prob,alg=Tsit5(),saveat=dt,adaptive=false,dt=1/1024,maxiters=maxiters)
        return sol
    end

    #for residual physics task
    function generate_ODE_data_task2(system::Function,ic::Vector{Float64},p,tspan::Tuple{Float64,Float64},maxiters::Float64,dt::Float64)
        prob=ODEProblem(system,ic,tspan,p)
        sol=solve(prob,alg=Rosenbrock23(),saveat=dt,adaptive=true,dtmax=1/32,maxiters=maxiters)
        return sol
    end

    function generate_ODE_data_ExtKuramoto(system::Function,ic::Vector{Float64},p,tspan::Tuple{Float64,Float64},maxiters::Float64,dt::Float64)
        N=length(ic)
        callback=CallbackSet(
            VectorContinuousCallback(reset_condition1,reset_affect1!,N),
            VectorContinuousCallback(reset_condition2,reset_affect2!,N)
            )
        prob=ODEProblem(system,ic,tspan,p;callback=callback)
        sol=solve(prob,alg=Rodas4P(),saveat=dt,adaptive=true,dtmax=1/64,maxiters=maxiters)
        return sol
    end

    #From David Barton:
    function generate_arrow(name, data_path; force = false, compress = true)
        filename = joinpath(data_path, name)
        arrow_file = filename * ".arrow"
        if compress
            arrow_file = arrow_file * ".lz4"
        end
        csv_file = filename * ".csv"
        if !isfile(arrow_file) || force
            data = CSV.read(csv_file, DataFrame; normalizenames = true)
            if compress
                Arrow.write(arrow_file, data, compress = :lz4)
            else
                Arrow.write(arrow_file, data)
            end
        end
        return nothing
    end


    ## for extended kuramoto (non cartesian so needs resets)
    function reset_condition1(out,u,t,integrator)
        N=length(integrator.u)
        for i in 1:N
            out[i]=pi-u[i]
        end
    end

    function reset_affect1!(integrator,event_idx)
        integrator.u[event_idx]=-pi
    end

    function reset_condition2(out,u,t,integrator)
        N=length(integrator.u)
        for i in 1:N
            out[i]=-pi-u[i]
        end
    end

    function reset_affect2!(integrator,event_idx)
        integrator.u[event_idx]=pi
    end

    function complex_form(theta_t)
        exp.(im.*theta_t)
    end

    function kuramoto_order2(solution,number_neurons)
        N=number_neurons
        #@show typeof(solution)
        #@show size(solution)
        (1/N) .* sum(complex_form,solution,dims=1)
    end

    function get_order_params(data)
        # phases,_=get_phase_portrait(data)
        phases=data[:,:]
        N=size(phases,1)
        @show(N)
        order=kuramoto_order2(phases,N)
        return order
    end

end
