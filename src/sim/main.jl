# This code is the main script to run 2d hydrodynamic simulations of nematopolar fluids.

include("InputParams.jl")
include("kernels.jl")     # stress, Fourier velocity solve, P/Q update
include("operators.jl")   # spatial scheme (:fft or :jacobi): derivatives, force balance, density fluxes

function main()
    # Dedicated random stream for the initial noise: GPU set-up (e.g. Metal kernel compilation spawning
    # tasks) draws from the global stream, so seeding that one gave backend-dependent noise.
    # Xoshiro(seed) yields the same numbers as Random.seed!(seed) did on the CPU.
    rng = Xoshiro(seed)
    @inbounds begin
        # ------ VARIABLE INITIALIZATION ------ #
        hasP = is_polar(Orientation_trait)
        hasQ = is_nematic(Orientation_trait)
        hasOrientation = has_orientation(Orientation_trait)

        # Initialize time and next storage time
        t = 0;
        NextStoreTime = 0;
        global Δt = Δt_ini
        # Perform adaptive time-step every Δt_check (DF column t_check)
        Δt_Check   = Δt_check;
        NextCheck  = 0;
        eps        = TF(1e-8); # Small constant (TF keeps the adaptive Δt in Float32 on Metal).
        
        # Dynamical fields
        ρ     = @zeros(N, N)         # Density field
        P     = hasP ? @zeros(N, N, 2) : nothing      # Polar field:    1 → x, 2 → y
        Q     = hasQ ? @zeros(N, N, 2) : nothing      # Nematic field:  1 → Qxx=-Qyy, 2 → xy=yx
        V     = @zeros(N, N, 2)      # Velocity field: 1 → x, 2 → y
        σ_nv  = @zeros(N, N, 4)      # Non-viscous stress tensor: 1 → xx, 2 → xy, 3 → yx, 4 → yy
        h     = hasP ? @zeros(N, N, 2) : nothing      # Molecular field for polar field
        H     = hasQ ? @zeros(N, N, 2) : nothing      # Molecular field for nematic field
        
        # Copy of device arrays on host.
        ρ_host  = zeros(TF, N, N)            # Copy of ρ on host
        P_host  = hasP ? zeros(TF, N, N, 2) : nothing         # Copy of P on host
        Q_host  = hasQ ? zeros(TF, N, N, 2) : nothing         # Copy of Q on host
        V_host  = zeros(TF, N, N, 2)         # Copy of V on host
        θQ      = hasQ ? zeros(TF, N, N) : nothing            # Orientation field of Q on host (used to check stationary homogeneous states)
        mask    = hasP ? zeros(Bool, N, N) : nothing          # Mask for polar defects (strings)

        # Initialize derivatives
        ∇P    = hasP ? @zeros(N, N, 4) : nothing     # 1 → ∂xPx, 2 → ∂xPy, 3 → ∂yPx, 4 → ∂yPy
        ΔP    = hasP ? @zeros(N, N, 2) : nothing     # Laplacian of Px, Py
        ∇Q    = hasQ ? @zeros(N, N, 4) : nothing     # 1 → ∂xQ1, 2 → ∂xQ2, 3 → ∂yQ1, 4 → ∂yQ2
        ΔQ    = hasQ ? @zeros(N, N, 2) : nothing     # Laplacian of Q1, Q2
        ∇ρ    = @zeros(N, N, 2)     # 1 → xx, 2 → yy
        Δρ    = @zeros(N, N)        # Laplacian of ρ
        ∇V    = @zeros(N, N, 4)     # 1 → xx, 2 → xy, 3 → yx, 4 → yy
        ∂xρVx = @zeros(N, N)        # Advection flux along x
        ∂yρVy = @zeros(N, N)        # Advection flux along x
        
        # Factors for FFT derivatives
        factor_∂x  = TA(zeros(TC, Lkx, N))
        factor_∂y  = TA(zeros(TC, Lkx, N))
        factor_Δ   = @zeros(Lkx, N)
        # Initialize the FFT factors (such as F[∂x], F[∂x^2], ...) on the reciprocal grid (spectral scheme only)
        is_spectral(Solver_trait) &&
            @parallel blocks_FFT threads compute_FFTderivative_factors!(factor_∂x, factor_∂y, factor_Δ, kx, ky, kx2, ky2)

        # FFT
        FV  = TA(zeros(TC, Lkx, N, 2))   # FFT of V
        Fσ  = TA(zeros(TC, Lkx, N, 4))   # FFT of σ
        # Work bundle for the spatial scheme (operators.jl). Spectral: FFT plans, derivative factors,
        # and three reusable FFT buffers (Rtmp real; F0, Fa complex) so the time loop allocates
        # nothing. Finite differences: the force G = ∇·σ and the Jacobi iteration's second array.
        work = (W = W, Wi = Wi, factor_∂x = factor_∂x, factor_∂y = factor_∂y, factor_Δ = factor_Δ,
                FV = FV, Fσ = Fσ,
                Rtmp = @zeros(N, N), F0 = TA(zeros(TC, Lkx, N)), Fa = TA(zeros(TC, Lkx, N)),
                G = @zeros(N, N, 2), v_temp = @zeros(N, N, 2))
        

        # ------ INITIAL CONDITIONS ------ #
        # Homogeneous density for initial condition.
        ρHSS = ρ0

        # Initialize the system on host then copy to GPU.
        # First, check whether a file already exists in the data folder. If so,
        # resume the simulation from there.
        # if !isempty(readdir(file))
        #     # Use Glob to match the file pattern
        #     file_list = glob("data_*.jld", file)
        #     # Extract numbers from file names and find the largest. To do so,
        #     # define a custom criterion for maximum() via the do ... end block.
        #     latest_num = maximum(file_list) do f
        #         # Extract file number. From the file list, remove the path and keep only the
        #         # basename (local file name). With regex r, extract the number.
        #         filenum = match(r"data_(\d+)\.jld", basename(f))
        #         # Convert the captured number to an Int.
        #         filenum !== nothing ? parse(Int, filenum.captures[1]) : -1
        #     end
        #     latest_file = string(file,"data_",@sprintf("%08d", latest_num),".jld")
        #     println("File already exists. Resuming simulation from ",basename(latest_file),"\n")
        #     # Import data from latest file
        #     ρ_host  = load(string(latest_file), "C") # Density (concentration C) field
        #     P_host  = load(string(latest_file), "P") # Polar field
        #     Q_host  = load(string(latest_file), "Q") # Nematic field
        #     # Renormalize densities to ρ0 and save them back to initial condition
        #     norm     = mean(ρ_host)
        #     @. ρ_host = ρ_host/norm*ρ0
        #     num = latest_num + 1
        #     # Increase NextStoreTime to avoid overwriting the initial condition.
        #     NextStoreTime += Δt_Store
        # # If no data file exists, initialize the system.
        # else
            # Initial conditions for density fields and polar field, respectively.
            Iρ  = zeros(TF, N, N)
            ICP = hasP ? zeros(TF, N, N, 2) : nothing
            ICQ = hasQ ? zeros(TF, N, N, 2) : nothing
            # Initialize noise for density fields and for polar field, respectively.
            Noiseρ  = rand(rng, N, N)
            NoiseP  = hasP ? rand(rng, N, N, 2) : nothing
            NoiseQ  = hasQ ? rand(rng, N, N, 2) : nothing
            get_CentredNoise!(Noiseρ, η0)   
            get_CentredNoisePQ!(NoiseP, ηP)
            get_CentredNoisePQ!(NoiseQ, ηQ)

            if Initialization=="Homogeneous"
                Iρ[:,:] .= 1.0
            # Polarized initial conditions along x direction,
            # according to the energy minima.
            elseif Initialization=="Polarized" && hasP && hasQ
                S1 = √((αQ/βQ) * (1.0 + χ^2/(4.0*βp*αQ)))
                Iρ[:,:] .= 1.0
                ICQ[:,:,1] .= 0.5*S1
                ICP[:,:,1] .= √(0.5*S1*abs(χ)/βp)
            # Polar only: |p|² = α_p/β_p at ρ = ρ0.
            elseif Initialization=="Polarized" && hasP
                Iρ[:,:] .= 1.0
                ICP[:,:,1] .= √(αp/βp)
            # Nematic only: Tr Q² = 2 Q1² = α_Q/(2β_Q) at ρ = ρ0 (the χ = 0 case of the above).
            elseif Initialization=="Polarized" && hasQ
                Iρ[:,:] .= 1.0
                ICQ[:,:,1] .= 0.5*√(αQ/βQ)
            # Initialize the system in a loop configuration.
            # Fields are polarized along x, and they exhibit a defect 
            # loop.
            elseif Initialization=="Loop"&& hasP && hasQ
                S1 = √((αQ/βQ) * (1.0 + χ^2/(4.0*βp*αQ)))
                Iρ[:,:] .= 1.0
                ICQ[:,:,1] .= 0.5*S1
                ICP[:,:,1] .= √(0.5*S1*abs(χ)/βp)
                get_Loop!(Iρ, ICP, ICQ, 0.25*L, 0.1)
            else
                error("initialisation = $Initialization is not available with orientation = $(df[:orientation]): " *
                      "Polarized needs a polar or nematic field, Loop needs nematopolar.")
            end
            @. ρ_host[:,:]   = ρHSS  * (Iρ + Noiseρ)
            hasP ? (@. P_host[:,:,:] = ICP + NoiseP) : nothing
            hasQ ? (@. Q_host[:,:,:] = ICQ + NoiseQ) : nothing
            # Initialize filename identifier
            num = 0
        # end
        # Copy from host to CUDA arrays
        copyto!(ρ,ρ_host)
        hasP ? copyto!(P,P_host) : nothing
        hasQ ? copyto!(Q,Q_host) : nothing

        # ------ RUN SIMULATION ------ #
        println("Simulation starts...")
        start_time = time()
        while (t <= t_end + 5*Δt) # to be "sure" that t_end is saved
            # ----- Store data ----- # 
            if (t >= NextStoreTime)

                if any(isnan, ρ)
                    println("Error: NaN. Stop simulation.")
                    return 1    # If there are errors, stop the simulations.
                end

                filename = string(file,"/Data/",@sprintf("%010d", t),".jld")   # Output filename is data_XXXXX.jld
                save_simulation_state(filename, ρ, V, P, Q, ρ_host, V_host, P_host, Q_host)
                # Compute elapsed time in seconds
                elapsed_time = time()-start_time
                # Print on .out file
                println("Now at time: " * @sprintf("%.2f", t) * " of " * @sprintf("%.2f", t_end) * ". Elapsed time [s] = " * @sprintf("%d", elapsed_time))
                NextStoreTime += Δt_Store
            end

            # ----- Time-evolution ----- # 
            # 1/ Non-viscous stress in real space. The derivatives ∇ρ, Δρ, ∇P, ΔP, ∇Q, ΔQ come
            # from the selected scheme (spectral or finite differences).
            compute_gradients!(Solver_trait, ρ, ∇ρ, Δρ, P, ∇P, ΔP, Q, ∇Q, ΔQ, work)
            # Compute chemical potential, molecular field, stress
            @parallel blocks threads Compute_stress!(σ_nv, h, H, ρ, ∇ρ, P, ∇P, ΔP, Q, ∇Q, ΔQ)

            # 2/ Solve the force balance ξV − (∇²V + ∇(∇·V)) = ∇·σ for V and ∇V
            # (exactly in Fourier space for :fft, by Jacobi iteration for :jacobi).
            solve_velocity!(Solver_trait, V, ∇V, σ_nv, work)

            # Adaptive timestep check.
            if (t >= NextCheck)
                max_V =  @views mapreduce( (x, y) -> x^2 + y^2, max, V[:,:,1], V[:,:,2] ) 
                global Δt = min(Δt*TF(1.25), Δt_max, TF(0.05)*Δ/(sqrt(max_V)+eps))
                # A NaN velocity makes Δt NaN, which would end the time loop as if t_end were
                # reached; stop with an error instead.
                if !isfinite(Δt)
                    println("Error: non-finite time step (NaN or Inf in the velocity) at t = ", t, ". Stop simulation.")
                    return 1
                end
                NextCheck += Δt_Check;
            end

            # 3/ Dynamics of the polar field
            if hasOrientation
                @parallel blocks threads Update_PQ!(P, Q, h, H, ∇P, ∇Q, V, ∇V, ρ, Δt)
            end
            
            # 4/ Dynamics of the density fields
            # Advective fluxes ∂x(ρVx), ∂y(ρVy) from the selected scheme
            density_advection!(Solver_trait, ∂xρVx, ∂yρVy, ρ, V, work)
            # Update the density fields.
            @. ρ += Δt*(-∂xρVx  -∂yρVy  + D0*Δρ - Rd*(ρ-ρ0))
            t += Δt
        end
    end
    end_time = time()
    println("Simulation finished!")

    # Elapsed time in seconds
    elapsed_time = end_time-start_time
    hours   = floor(elapsed_time/3600)
    minutes = floor((elapsed_time-3600*hours)/60)
    seconds = floor(elapsed_time-3600*hours-60*minutes)
    @printf("Elapsed time: %d hours, %d minutes and %d seconds.\n", hours, minutes, seconds)
    return nothing
end

# Start simulation
# A failed run (main returns 1) exits with a non-zero status, so Slurm reports FAILED.
main() == 1 && exit(1)
