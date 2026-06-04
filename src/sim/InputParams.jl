idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
file = ENV["path_to_data"]
mkpath(string(file,"/Data/") )
println(idx) 


using DelimitedFiles, CSV, DataFrames, Dates, Printf, JLD2, ParallelStencil, Random, FFTW, Statistics, Random, Roots, Glob
# CUDA, Metal, AMDGPU, oneAPI, KernelAbstractions, Threads, Polyester
@show const USE_GPU = Base.parse(Bool, ENV["use_gpu"])#true # false #true
@show const TF = Sys.isapple() && USE_GPU ? Float32 : Float64
@show const TI = Int64
@show const TC = Sys.isapple() && USE_GPU ? ComplexF32 : ComplexF64

@static if USE_GPU
    @static if Sys.isapple()
        @init_parallel_stencil(Metal, TF, 2, inbounds=true)
        const TA = Metal.MltArray
    else
        @init_parallel_stencil(CUDA, TF, 2, inbounds=true)
        const TA = CUDA.CuArray
    end
else
    @init_parallel_stencil(Threads, TF, 2, inbounds=true)
        const TA = Array
end
@show TA
@show @current_hardware

include("Utils.jl")

dir_df = @__DIR__
df = CSV.read(joinpath(dir_df,"DF.csv"), DataFrame)[idx,:]

# ------ Initialization of simulation grids ------- #
# System size (square)
const N::TI  = TF(df[:N]);    # Must be a power of 2!
const Δ::TF  = TF(df[:dx])
const Δ2::TF = Δ^2

const L::TF  = N*Δ;


# Time step, final time, storage time interval.
const Δt_ini::TF = TF(df[:dt_ini]);
const Δt_max::TF = TF(df[:dt_max]);
const t_end::TF = df[:t_end];
const Δt_Store = TF(df[:t_print]);
const Δt_check = TF(df[:t_check]);

# Space grid [-L/2,L/2]x[-L/2,L/2]
x  = Data.Array([-L/2 + i*Δ for i = 0:N-1])
y  = Data.Array([-L/2 + i*Δ for i = 0:N-1])
# Reciprocal-space grid
kx  = Data.Array(2*pi*rfftfreq(N, 1/Δ)); # rfft() creates N/2+1 qx-values
ky  = Data.Array(2*pi*fftfreq(N, 1/Δ))   # fft() creates N qy-values
const Lkx::TI = length(kx);                   # N/2+1
# Squared reciprocal vectors
kx2 = kx.*kx
ky2 = ky.*ky
# FFT tools to calculate derivatives
W  = plan_rfft(@ones(N, N))    # Fourier-transform matrix operator
Wi = inv(W)                                   # Inverse-Fourier transform

# ------ GPU parameters ↔ Real space mapping ------- #
# We divide the (N, N) grid into Bx*By blocks, which are further divided into wraps.
# We assign a wrap to each point of the real space grid. 
WrapsT::Int = 16                            # N. of wraps per block
B::Int = ceil(Int, N/WrapsT)               # N. of blocks along x
# Each block has size (WrapsT * WrapsT). 
threads = (WrapsT, WrapsT)               # Dimension of each block
blocks = (B, B)                          # Dimension of grid

# ------ GPU parameter ↔ Reciprocal space mapping ------ #
# Note that we first do rfft along x, then fft along y, hence the asymmetric grid
# The +1 is for the k=0 mode.
blocks_FFT = (div(B,2)+1, B)

# ------ System parameters ------ #
@show const Rd::TF  = 1/TF(df[:tau])         # Renewal rate ( τ^{-1} )
@show const D0::TF  = TF(df[:D])        # Diffusivity
# Orientation Field(s)
@show const Orientation_trait = get_trait(Symbol(df[:orientation]))
# Free energy parameters
@show const a::TF   = TF(df[:a])          # Steric coefficient
@show const αp::TF  = TF(df[:alpha_p])          # Pre-factor of p^2 term
@show const βp::TF  = TF(df[:beta_p])          # Pre-factor of p^4 term
@show const kp::TF  = TF(df[:kappa_p])       # Elastic constant for p
@show const αQ::TF  = TF(df[:alpha_q])          # Pre-factor of Q^2 term
@show const βQ::TF  = TF(df[:beta_q])          # Pre-factor of Q^4 term
@show const kQ::TF  = TF(df[:kappa_q])       # Elastic constant for Q
@show const χ::TF   = TF(df[:chi])          # Nematopolar coupling
# Polar/nematic field dynamics
@show const ν1::TF = TF(df[:nu])           # Flow alignment for P
@show const ν2::TF = TF(df[:nu2])           #
@show const λ::TF  = TF(df[:lambda])           # Flow alignment for Q
@show const Γ::TF  = TF(df[:gamma_p])           # Rotational viscosity for P, Q
@show const εp::TF = TF(df[:epsi_p])          # Active extensile term
@show const εQ::TF = TF(df[:epsi_q])           # Active extensile term
# Active stress
@show const ζρ::TF  = TF(df[:zeta])          # Isotropic contractility
@show const ζp::TF  = TF(df[:zeta_p])          # Anisotropic stress coefficients (polar)
@show const ζp2::TF = TF(df[:zeta_p2]) 
@show const ζQ::TF  = TF(df[:zeta_q])          # Anisotropic stress coefficient (nematic)
# Viscosity
@show const ξ::TF   = 1.0#TF(df[:xi])
# Density
@show const ρ0::TF  = TF(df[:rho0])          
# Initialization
@show const Initialization = "Homogeneous" # Choose between "Homogeneous", "Polarized","Loop"
@show const seed::TI = TI(df[:seed])                # Seed for random number generator
@show const η0::TF   = TF(df[:eta_rho])            # Noise amplitude for initial conditions on density
@show const ηP::TF   = TF(df[:eta_p])
@show const ηQ::TF   = TF(df[:eta_q])

println("Parameters have been initialized.")