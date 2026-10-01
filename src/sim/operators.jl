# operators.jl — the spatial scheme: field derivatives, force balance, and density fluxes.
#
# One switch, Solver_trait (DF.csv column `solver`, see utils.jl), selects a whole discretisation:
#   FFTSolver    (:fft)    spectral derivatives + exact per-mode Fourier velocity solve.
#                          Needs the FFT plans W, Wi → CUDA and CPU.
#   JacobiSolver (:jacobi) 2nd-order central finite differences (periodic) + Jacobi iteration of the
#                          same force balance. Takes no transform → also runs on Metal / AMD.
#
# Both solve the compressible Darcy–Brinkman force balance (viscosity η = 1, friction ξ)
#     ξ V − ( ∇²V + ∇(∇·V) ) = G ,     G = ∇·σ_nv ,
# whose Fourier solution is UpdateVelocity_Fourier! (kernels.jl).
#
# ── Adding a scheme ──────────────────────────────────────────────────────────────────────────────
#  1. utils.jl: define `struct MySolver <: VelocitySolver end`, add `:mysolver => MySolver` to
#     SOLVERS, and set `is_spectral(::MySolver) = true` if it uses the FFT plans W, Wi.
#  2. Here: implement the three methods below for ::MySolver. Each receives the device arrays plus
#     `w`, the work bundle built in main.jl (FFT plans/factors/buffers and real scratch arrays).
#        compute_gradients!(::MySolver, ρ, ∇ρ, Δρ, P, ∇P, ΔP, Q, ∇Q, ΔQ, w)
#            fill ∇ρ (…,2), Δρ, and when P/Q are not `nothing`: ∇P, ∇Q (…,4: ∂xF1,∂xF2,∂yF1,∂yF2)
#            and ΔP, ΔQ (…,2).
#        solve_velocity!(::MySolver, V, ∇V, σ_nv, w)
#            solve the force balance above for V (…,2) and fill ∇V (…,4, same layout), from the
#            non-viscous stress σ_nv (…,4: xx, xy, yx, yy). V holds the previous step on entry
#            (a free initial guess for iterative schemes).
#        density_advection!(::MySolver, ∂xρVx, ∂yρVy, ρ, V, w)
#            fill the advective fluxes ∂x(ρVx) and ∂y(ρVy).
#  3. GenInputParams.pluto.jl: add its name to ENUM_SOLVER so it can be selected.
#  Nothing in main.jl or kernels.jl changes. Check a new scheme against :fft on a short run.
# ────────────────────────────────────────────────────────────────────────────────────────────────
# Included AFTER kernels.jl (uses ξ, Δ, Δ2, N, kx, ky, blocks/threads from InputParams.jl).

using LinearAlgebra: mul!

# ================================================================================================
# Spectral scheme (:fft). Allocation-free: every transform writes into a buffer of `w` with mul!.
#   w.Rtmp (N×N real)      real scratch: input of forward transforms, output of inverse ones
#   w.F0   (Lkx×N complex) spectrum of the field being differentiated
#   w.Fa   (Lkx×N complex) scratch input of the inverse (c2r) transform, which overwrites it
# ================================================================================================

# out ← F⁻¹[fac · F0]   (out may be a slice: the result passes through w.Rtmp)
@inline function _spec_deriv!(out, fac, w)
    w.Fa .= fac .* w.F0
    mul!(w.Rtmp, w.Wi, w.Fa)
    copyto!(out, w.Rtmp)
    return nothing
end

# Gradient and Laplacian of component i of a two-component field g (P or Q).
function _spec_grad_lap!(∇g, Δg, g, i, w)
    copyto!(w.Rtmp, view(g, :, :, i)); mul!(w.F0, w.W, w.Rtmp)
    _spec_deriv!(view(∇g, :, :, i),     w.factor_∂x, w)   # ∂x g_i → components 1,2
    _spec_deriv!(view(∇g, :, :, 2 + i), w.factor_∂y, w)   # ∂y g_i → components 3,4
    _spec_deriv!(view(Δg, :, :, i),     w.factor_Δ,  w)   # Laplacian of g_i
    return nothing
end

function compute_gradients!(::FFTSolver, ρ, ∇ρ, Δρ, P, ∇P, ΔP, Q, ∇Q, ΔQ, w)
    for i = 1:2
        P !== nothing && _spec_grad_lap!(∇P, ΔP, P, i, w)
        Q !== nothing && _spec_grad_lap!(∇Q, ΔQ, Q, i, w)
    end
    mul!(w.F0, w.W, ρ)                                    # r2c leaves ρ untouched
    _spec_deriv!(view(∇ρ, :, :, 1), w.factor_∂x, w)
    _spec_deriv!(view(∇ρ, :, :, 2), w.factor_∂y, w)
    _spec_deriv!(Δρ,                w.factor_Δ,  w)
    return nothing
end

function solve_velocity!(::FFTSolver, V, ∇V, σ_nv, w)
    for i = 1:4
        copyto!(w.Rtmp, view(σ_nv, :, :, i)); mul!(w.Fa, w.W, w.Rtmp); copyto!(view(w.Fσ, :, :, i), w.Fa)
    end
    @parallel blocks_FFT threads UpdateVelocity_Fourier!(w.FV, w.Fσ, kx, ky)
    for i = 1:2
        copyto!(w.F0, view(w.FV, :, :, i))
        _spec_deriv!(view(∇V, :, :, i),     w.factor_∂x, w)   # ∂x V_{x,y} → components 1,2
        _spec_deriv!(view(∇V, :, :, i + 2), w.factor_∂y, w)   # ∂y V_{x,y} → components 3,4
        w.Fa .= w.F0; mul!(w.Rtmp, w.Wi, w.Fa); copyto!(view(V, :, :, i), w.Rtmp)
    end
    return nothing
end

function density_advection!(::FFTSolver, ∂xρVx, ∂yρVy, ρ, V, w)
    @views w.Rtmp .= ρ .* V[:, :, 1]; mul!(w.F0, w.W, w.Rtmp)
    w.Fa .= w.factor_∂x .* w.F0; mul!(∂xρVx, w.Wi, w.Fa)
    @views w.Rtmp .= ρ .* V[:, :, 2]; mul!(w.F0, w.W, w.Rtmp)
    w.Fa .= w.factor_∂y .* w.F0; mul!(∂yρVy, w.Wi, w.Fa)
    return nothing
end

# ================================================================================================
# Finite-difference scheme (:jacobi). Periodic, central, 2nd order; index wrap by ternary
# (branch-light, safe on every backend).
# ================================================================================================

# Diagonal of the discretised force-balance operator, hoisted out of the iteration: ξ + 6/Δ².
const inv_diag = TF(1) / (ξ + TF(6) / Δ2)

@parallel_indices (i, j) function fd_grad_lap_scalar!(∇f, Δf, f)
    i_ = (i == 1) ? N : i-1;  ip = (i == N) ? 1 : i+1
    j_ = (j == 1) ? N : j-1;  jp = (j == N) ? 1 : j+1
    ∇f[i,j,1] = (f[ip,j] - f[i_,j]) * TF(0.5) / Δ
    ∇f[i,j,2] = (f[i,jp] - f[i,j_]) * TF(0.5) / Δ
    Δf[i,j]   = (f[ip,j] + f[i_,j] + f[i,jp] + f[i,j_] - TF(4)*f[i,j]) / Δ2
    return nothing
end

# Two-component field F: ∇F = (∂xF1, ∂xF2, ∂yF1, ∂yF2), ΔF = (ΔF1, ΔF2).
@parallel_indices (i, j) function fd_grad_lap_vec!(∇F, ΔF, F)
    i_ = (i == 1) ? N : i-1;  ip = (i == N) ? 1 : i+1
    j_ = (j == 1) ? N : j-1;  jp = (j == N) ? 1 : j+1
    for c in 1:2
        ∇F[i,j,c]   = (F[ip,j,c] - F[i_,j,c]) * TF(0.5) / Δ
        ∇F[i,j,2+c] = (F[i,jp,c] - F[i,j_,c]) * TF(0.5) / Δ
        ΔF[i,j,c]   = (F[ip,j,c] + F[i_,j,c] + F[i,jp,c] + F[i,j_,c] - TF(4)*F[i,j,c]) / Δ2
    end
    return nothing
end

# Gradient only, for ∇V.
@parallel_indices (i, j) function fd_grad_vec!(∇F, F)
    i_ = (i == 1) ? N : i-1;  ip = (i == N) ? 1 : i+1
    j_ = (j == 1) ? N : j-1;  jp = (j == N) ? 1 : j+1
    for c in 1:2
        ∇F[i,j,c]   = (F[ip,j,c] - F[i_,j,c]) * TF(0.5) / Δ
        ∇F[i,j,2+c] = (F[i,jp,c] - F[i,j_,c]) * TF(0.5) / Δ
    end
    return nothing
end

# Force G = ∇·σ_nv (σ components 1 = xx, 2 = xy, 3 = yx, 4 = yy).
@parallel_indices (i, j) function fd_force_div!(G, σ)
    i_ = (i == 1) ? N : i-1;  ip = (i == N) ? 1 : i+1
    j_ = (j == 1) ? N : j-1;  jp = (j == N) ? 1 : j+1
    G[i,j,1] = (σ[ip,j,1] - σ[i_,j,1]) * TF(0.5)/Δ + (σ[i,jp,2] - σ[i,j_,2]) * TF(0.5)/Δ
    G[i,j,2] = (σ[ip,j,3] - σ[i_,j,3]) * TF(0.5)/Δ + (σ[i,jp,4] - σ[i,j_,4]) * TF(0.5)/Δ
    return nothing
end

@parallel_indices (i, j) function fd_density_advection!(∂xρVx, ∂yρVy, ρ, V)
    i_ = (i == 1) ? N : i-1;  ip = (i == N) ? 1 : i+1
    j_ = (j == 1) ? N : j-1;  jp = (j == N) ? 1 : j+1
    ∂xρVx[i,j] = (ρ[ip,j]*V[ip,j,1] - ρ[i_,j]*V[i_,j,1]) * TF(0.5) / Δ
    ∂yρVy[i,j] = (ρ[i,jp]*V[i,jp,2] - ρ[i,j_]*V[i,j_,2]) * TF(0.5) / Δ
    return nothing
end

# One Jacobi sweep of the discretised force balance. Per component, the anisotropic Laplacian
# (2∂x² + ∂y² for Vx) uses the 4 edge neighbours and the grad-div cross term the 4 corners.
@parallel_indices (i, j) function iterate_velocity_jac!(vnew, v, G)
    i_ = (i == 1) ? N : i-1;  ip = (i == N) ? 1 : i+1
    j_ = (j == 1) ? N : j-1;  jp = (j == N) ? 1 : j+1
    vnew[i,j,1] = inv_diag * ( G[i,j,1] + ( TF(2)*(v[i_,j,1] + v[ip,j,1]) + (v[i,j_,1] + v[i,jp,1]) +
                    (v[ip,jp,2] + v[i_,j_,2] - v[i_,jp,2] - v[ip,j_,2]) * TF(0.25) ) / Δ2 )
    vnew[i,j,2] = inv_diag * ( G[i,j,2] + ( TF(2)*(v[i,j_,2] + v[i,jp,2]) + (v[i_,j,2] + v[ip,j,2]) +
                    (v[ip,jp,1] + v[i_,j_,1] - v[i_,jp,1] - v[ip,j_,1]) * TF(0.25) ) / Δ2 )
    return nothing
end

# Sweep until the largest change between two sweeps is below error_threshold. The check is a
# global reduction, so it runs every `interval` sweeps, starting at cce_base and doubling up to
# cce_cap: about log(T) + T/cce_cap checks for T sweeps.
function iterate_to_convergence!(v, v_temp, G)
    incre = 0; max_error = TF(1); interval = cce_base
    while max_error > error_threshold && incre < max_iter
        for _ = 1:interval
            @parallel blocks threads iterate_velocity_jac!(v_temp, v, G)
            @parallel blocks threads iterate_velocity_jac!(v, v_temp, G)
        end
        incre += interval
        max_error = mapreduce((x, y) -> abs(x - y), max, v, v_temp)
        interval = min(2*interval, cce_cap)
    end
    return incre, max_error
end

function compute_gradients!(::JacobiSolver, ρ, ∇ρ, Δρ, P, ∇P, ΔP, Q, ∇Q, ΔQ, w)
    @parallel blocks threads fd_grad_lap_scalar!(∇ρ, Δρ, ρ)
    P !== nothing && @parallel blocks threads fd_grad_lap_vec!(∇P, ΔP, P)
    Q !== nothing && @parallel blocks threads fd_grad_lap_vec!(∇Q, ΔQ, Q)
    return nothing
end

function solve_velocity!(::JacobiSolver, V, ∇V, σ_nv, w)
    @parallel blocks threads fd_force_div!(w.G, σ_nv)
    iterate_to_convergence!(V, w.v_temp, w.G)            # V from the previous step is the guess
    @parallel blocks threads fd_grad_vec!(∇V, V)
    return nothing
end

function density_advection!(::JacobiSolver, ∂xρVx, ∂yρVy, ρ, V, w)
    @parallel blocks threads fd_density_advection!(∂xρVx, ∂yρVy, ρ, V)
    return nothing
end
