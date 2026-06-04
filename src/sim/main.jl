# This code is the main script to run 2d hydrodynamic simulations of nematopolar fluids.

include("InputParams.jl")
include("kernels.jl")

function main()
    Random.seed!(seed)
    @inbounds begin
        # ------ VARIABLE INITIALIZATION ------ #
        hasP = is_polar(Orientation_trait)
        hasQ = is_nematic(Orientation_trait)
        hasOrientation = has_orientation(Orientation_trait)

        # Initialize time and next storage time
        t = 0;
        NextStoreTime = 0;
        global Δt = Δt_ini
        # Perform adaptive time-step every Δt_check
        Δt_Check   = 1.0;
        NextCheck  = 0;
        eps        = 1e-8;     # Small constant.
        
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
        # Initialize the FFT factors (such as F[∂x], F[∂x^2], ...) on the recprocal grid
        @parallel blocks_FFT threads compute_FFTderivative_factors!(factor_∂x, factor_∂y, factor_Δ, kx, ky, kx2, ky2)

        # FFT
        Fρ  = TA(zeros(TC, Lkx, N))      # FFT of ρ
        FV  = TA(zeros(TC, Lkx, N, 2))   # FFT of V
        FP  = hasP ? TA(zeros(TC, Lkx, N, 2)) : nothing   # FFT of P
        FQ  = hasP ? TA(zeros(TC, Lkx, N, 2)) : nothing   # FFT of Q
        Fσ  = TA(zeros(TC, Lkx, N, 4))   # FFT of σ
        

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
            Noiseρ  = rand(N, N)
            NoiseP  = hasP ? rand(N, N, 2) : nothing
            NoiseQ  = hasP ? rand(N, N, 2) : nothing
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
                println("Error: the system was not correctly initialized.\n")
                return nothing
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
                    print("Error: NaN. Stop simulation.")
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
            # 1/ We compute the non-viscous stress in real space. To do so,
            # we first compute all gradient terms in Fourier space, then perform
            # the inverse transform.  
            # Fourier-transform of the fields
            Fρ  .= W * ρ
            
            # Polar and nematic fields → F[P], F[Q] → ∇P, ∇Q, ΔP in real space
            for i=1:2
                if hasP
                    @views FP[:,:,i]    .= W  * P[:,:,i]
                    @views ∇P[:,:,i]    .= Wi * (factor_∂x .* FP[:,:,i]) # Update ∂x P_{x,y} → Components 1,2
                    @views ∇P[:,:,2+i]  .= Wi * (factor_∂y .* FP[:,:,i]) # Update ∂y P_{x,y} → Components 3,4
                    @views ΔP[:,:,i]    .= Wi * (factor_Δ  .* FP[:,:,i]) # Update Laplacian of P
                end
                if hasQ 
                    @views FQ[:,:,i]    .= W  * Q[:,:,i]
                    @views ∇Q[:,:,i]    .= Wi * (factor_∂x .* FQ[:,:,i]) # Update ∂x Q_{1,2} → Components 1,2
                    @views ∇Q[:,:,2+i]  .= Wi * (factor_∂y .* FQ[:,:,i]) # Update ∂y Q_{1,2} → Components 3,4
                    @views ΔQ[:,:,i]    .= Wi * (factor_Δ  .* FQ[:,:,i]) # Update Laplacian of Q
                end
            end

            # Actin concentration ρ → F[ρ] → ∇ρ in real space
            @views ∇ρ[:,:,1] .= Wi * (factor_∂x .* Fρ)
            @views ∇ρ[:,:,2] .= Wi * (factor_∂y .* Fρ)
            Δρ        .= Wi * (factor_Δ  .* Fρ)
            # Compute chemical potential, molecular field, stress
            @parallel blocks threads Compute_stress!(σ_nv, h, H, ρ, ∇ρ, P, ∇P, ΔP, Q, ∇Q, ΔQ)
            
            # 2/ We Fourier-transform the non-viscous stress and determine the velocity
            # in Fourier space.
            for i=1:4
                @views Fσ[:,:,i] .= W * σ_nv[:,:,i]
            end
            # Update the velocity in Fourier space, then back to real space
            @parallel blocks_FFT threads UpdateVelocity_Fourier!(FV, Fσ, kx, ky) 
            for i=1:2
                @views ∇V[:,:,i]   .= Wi * (factor_∂x .* FV[:,:,i]) # Update ∂x V_{x,y} → Components 1,2
                @views ∇V[:,:,i+2] .= Wi * (factor_∂y .* FV[:,:,i]) # Update ∂y V_{x,y} → Components 3,4
                @views V[:,:,i]    .= Wi * FV[:,:,i]
            end

            # Adaptive timestep check.
            if (t >= NextCheck)
                max_V =  @views mapreduce( (x, y) -> x^2 + y^2, max, V[:,:,1], V[:,:,2] ) 
                global Δt = minimum([Δt*1.25, Δt_max, 0.05*Δ/(sqrt(max_V)+eps)])
                NextCheck += Δt_Check;
            end

            # 3/ Dynamics of the polar field
            if hasOrientation
                @parallel blocks threads Update_PQ!(P, Q, h, H, ∇P, ∇Q, V, ∇V, ρ, Δt)
            end
            
            # 4/ Dynamics of the density fields
            # Advective terms in Fourier → real space
            @views ∂xρVx .= Wi * (factor_∂x .* (W * (ρ .* V[:,:,1]) ) )
            @views ∂yρVy .= Wi * (factor_∂y .* (W * (ρ .* V[:,:,2]) ) )
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
main()