idx = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
file = ENV["path_to_data"]
mkpath(string(file,"/Data/") )
println(idx) 


include("utils.jl")   # provides ensure_installed(...) + the trait helpers

# Self-provision: install any missing (non-stdlib) dependencies before loading them.
ensure_installed("CSV", "DataFrames", "JLD2", "ParallelStencil", "FFTW", "Roots", "Glob")

using DelimitedFiles, CSV, DataFrames, Dates, Printf, JLD2, ParallelStencil, Random, FFTW, Statistics, Random, Roots, Glob
@show const USE_GPU = Base.parse(Bool, ENV["use_gpu"])#true # false #true
@show const TF = Sys.isapple() && USE_GPU ? Float32 : Float64
@show const TI = Int64
@show const TC = Sys.isapple() && USE_GPU ? ComplexF32 : ComplexF64

# The GPU backend package must be present before @init_parallel_stencil imports it.
USE_GPU && ensure_installed(Sys.isapple() ? "Metal" : "CUDA")

@static if USE_GPU
    @static if Sys.isapple()
        @init_parallel_stencil(Metal, TF, 2, inbounds=true)
        const TA = Metal.MtlArray
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

dir_df = @__DIR__
df = CSV.read(joinpath(dir_df,"DF.csv"), DataFrame)[idx,:]

# Optional-column getter: tables written before a column existed still load, with its default.
_ip_get(k, default) = (k in propertynames(df)) ? df[k] : default

# ------ Spatial scheme / velocity solver ------- #
# :fft (spectral + Fourier solve; CUDA and CPU) | :jacobi (finite differences + iteration; any
# backend, including Metal, which has no FFT). See operators.jl.
@show const Solver_trait = get_solver(Symbol(_ip_get(:solver, "fft")))
# Iterative-solver settings (used by :jacobi only)
@show const cce_base::Int       = round(Int, _ip_get(:cce_base, 10))    # first convergence-check interval
@show const cce_cap::Int        = round(Int, _ip_get(:cce_cap, 100))    # largest check interval
@show const max_iter::Int       = round(Int, _ip_get(:max_iter, 100000))
@show const error_threshold::TF = TF(_ip_get(:error_threshold, 1e-6))

# Memory hygiene: a partial garbage collection, GC.gc(false), every `gc_every` time steps keeps the
# process's memory bounded under a cluster memory limit (the CPU backend allocates a little at every
# kernel launch, i.e. ~1 MB per step with :jacobi). 0 disables it.
@show const gc_every::Int = round(Int, _ip_get(:gc_every, 20))

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
# Created only for a spectral scheme: the finite-difference schemes take no transform, which is
# what lets them run on Metal (Metal.jl has no plan_rfft).
W  = is_spectral(Solver_trait) ? plan_rfft(@ones(N, N)) : nothing   # Fourier-transform operator
Wi = is_spectral(Solver_trait) ? inv(W)                 : nothing   # inverse transform

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
@show const Initialization = String(df[:initialisation])   # "Homogeneous", "Polarized" or "Loop"
@show const seed::TI = TI(df[:seed])                # Seed for random number generator
@show const η0::TF   = TF(df[:eta_rho])            # Noise amplitude for initial conditions on density
@show const ηP::TF   = TF(df[:eta_p])
@show const ηQ::TF   = TF(df[:eta_q])

println("Parameters have been initialized.")