# This code contains the GPU kernels for the 2d hydrodynamic simulations.

# ------ INITIALIZATION ------ #
# Initialize all factors for Fourier transforms on the reciprocal grid.
@parallel_indices (i, j) function compute_FFTderivative_factors!(factor_∂x, factor_∂y, factor_Δ, kx, ky, kx2, ky2)

    if i <= Lkx
        factor_∂x[i,j]  = im * kx[i]
        factor_∂y[i,j]  = im * ky[j]
        factor_Δ[i,j]   = -kx2[i]-ky2[j]
    end

    return nothing
end

# Function to initialize a defect loop in a polarized configuration.
# r0 is the radius of the loop, w is the width of the defect region.
# Note that Iρ is a CPU array.
function get_Loop!(Iρ, ICP, ICQ, r0, w)
    for i in (1:N)
        for j in (1:N)
            # Distance from the centre
            r = √((i*Δ - 0.5*L)^2 + (j*Δ - 0.5*L)^2)
            # Invert the polar field inside the defect region 
            polar_sign = (r<r0-0.5*w) ? -1 : 1
            # Squared distance from the defect loop centre line
            dr2=(r-r0)^2
            # Shrink the fields P, Q close to zero at the defect core
            # using a smooth Schwarz kernel.
            factor   = (dr2<w^2) ? (1.0 - 1.0*exp(-w^2/(w^2-dr2))) : 1.0
            factor_ρ = (dr2<w^2) ? (1.0 - 0.1*exp(-w^2/(w^2-dr2))) : 1.0
            # Adαpt the fields accordingly
            ICP[i,j,:] .= ICP[i,j,:]*polar_sign*factor
            ICQ[i,j,:] .= ICQ[i,j,:]*factor
            Iρ[i,j]     =    Iρ[i,j]*factor_ρ
        end
    end
    return nothing
end

# Initialize centred noise for the density fields.
function get_CentredNoise!(ξ, Eps)
    @. ξ      = Eps*(2*ξ-1.)
    ξ_Average = sum(ξ)/length(ξ) 
    ξ       .-= ξ_Average
end

# Initialize centred noise for the polar field. Note that the noise has both x,y components.
@inline function get_CentredNoisePQ!(ξ::AbstractArray, Eps)
    @. ξ     = Eps*(2*ξ-1.)
    # Center noise along both X and Y directions
    ξ_AverageX = sum(ξ[:,:,1])/length(ξ[:,:,1]) # X
    ξ[:,:,1] .-= ξ_AverageX
    ξ_AverageY = sum(ξ[:,:,2])/length(ξ[:,:,2]) # Y
    ξ[:,:,2] .-= ξ_AverageY
end

@inline function get_CentredNoisePQ!(::Nothing, ::Any)
    return nothing
end


# ==============================================================================
# Kernel design philosophy
# ==============================================================================
#
# The kernel is written unconditionally for the most general case (IsNematoPolar).
# Simpler configurations (IsNone, IsPolar, IsNematic) are handled automatically
# by the compiler without any runtime branching.
#
# This works in three steps:
#
# 1. LOAD — dispatch on Nothing vs real array
#
#       @inline load_P(P, i, j)         = P[i,j,1], P[i,j,2]
#       @inline load_P(::Nothing, i, j) = TF(0), TF(0)
#
#    If P is nothing, Px and Py are TF(0) literals in the kernel scope.
#
# 2. COMPUTE — write all physics unconditionally
#
#       hx = -ρ^2*(βp*P2 - αp*ρ*ρ0_rec)*Px + kp*ΔPx*ρ^2 + ...
#
#    No if/else, no trait guards, no helper functions needed.
#
# 3. ELIMINATE — LLVM constant-folds the zero arithmetic at compile time
#
#    For IsNone or IsNematic, Px=0, Py=0, ... so the compiler sees:
#       P2 = 0*0 + 0*0  → 0
#       hx = -ρ^2*(βp*0 - αp*ρ*ρ0_rec)*0 + kp*0*ρ^2 + ...  → 0
#    and eliminates those terms entirely from the compiled kernel.
#
#    This is safe as long as ρ is finite (no NaN/Inf in the density field),
#    since LLVM cannot fold 0 * NaN → 0 under IEEE 754.
#
# The result is 4 fully optimised kernels (one per trait specialisation)
# compiled from a single readable source, with no runtime overhead.
# ==============================================================================

# ── Local-variable loaders ───────────────────────────────────────────────────── 
# Using ntuple with Val(N) triggers loop unrolling inside the LLVM compiler stage. 
# The generated machine code will directly load the values into registers without any loop overhead 
# or tuple creation penalties, matching the performance of a hand-written versions.

@inline function load_field(F, i, j, ::Val{N}) where {N}
    return ntuple(k -> F[i,j,k], Val(N))
end
# Generic Loader for a Nothing fallback
@inline function load_field(::Nothing, i, j, ::Val{N}) where {N}
    return ntuple(k -> TF(0), Val(N))
end

# ── Write on device ───────────────────────────────────────────────────── 
@inline function save_field!(F, i, j, v1, v2)
    F[i, j, 1] = v1
    F[i, j, 2] = v2
    return nothing
end
# If the array is Nothing, this function compiles into literally nothing!
@inline save_field!(::Nothing, i, j, v1, v2) = nothing


# ------ DYNAMICS ------ #
@parallel_indices (i, j) function Compute_stress!(σ_nv, h, H, ρ_, ∇ρ, P, ∇P, ΔP, Q, ∇Q, ΔQ)

    # Load local variable
    # # Density
    ρ      = ρ_[i,j];
    ρ0_rec = 1.0/ρ0;
    ∂xρ  = ∇ρ[i,j,1];  ∂yρ  = ∇ρ[i,j,2];

    # # Polarity
    Px, Py                  = load_field(P, i, j, Val(2))
    ∂xPx, ∂xPy, ∂yPx, ∂yPy  = load_field(∇P, i, j, Val(4))
    ΔPx, ΔPy                = load_field(ΔP, i, j, Val(2))

    # # Nematic
    Q1, Q2                  = load_field(Q, i, j, Val(2))
    ∂xQ1, ∂xQ2, ∂yQ1, ∂yQ2  = load_field(∇Q, i, j, Val(4))
    ΔQ1, ΔQ2                = load_field(ΔQ, i, j, Val(2))

    # Squared modulus of the polarization field and its gradient
    P2  = Px*Px + Py*Py
    P4  = P2*P2
    ∇P2 = ∂xPx^2+∂xPy^2+∂yPx^2+∂yPy^2
    QQ  = 2*(Q1*Q1 + Q2*Q2)                # Trace of Q^2
    Q4  = QQ*QQ                            # Trace of Q^4
    ∇QQ = 2*(∂xQ1^2+∂xQ2^2+∂yQ1^2+∂yQ2^2)

    # Compute chemical potential and molecular field
    μ_ij = a*ρ^3 + ρ*(-1.5*ρ*ρ0_rec*αp*P2 + 0.5*βp*P4 + kp*∇P2
                        -1.5*ρ*ρ0_rec*αQ*QQ +     βQ*Q4 + kQ*∇QQ
                        +χ*(Q1*(Py*Py-Px*Px)-2*Q2*Px*Py) );

    # Molecular field for polar field
    hx = -ρ^2*(βp*P2-αp*ρ*ρ0_rec)*Px + kp*ΔPx*ρ^2 + 2*kp*ρ*(∂xρ*∂xPx+∂yρ*∂yPx) + ρ^2*χ*(Q2*Py+Q1*Px);
    hy = -ρ^2*(βp*P2-αp*ρ*ρ0_rec)*Py + kp*ΔPy*ρ^2 + 2*kp*ρ*(∂xρ*∂xPy+∂yρ*∂yPy) + ρ^2*χ*(Q2*Px-Q1*Py); 
    save_field!(h, i, j, hx, hy)
    
    # Molecular filed for nematic field
    H1 = -2*ρ^2*(2*βQ*QQ-αQ*ρ*ρ0_rec)*Q1 + 2*kQ*ΔQ1*ρ^2 + 4*kQ*ρ*(∂xρ*∂xQ1+∂yρ*∂yQ1) + ρ^2*χ*0.5*(Px^2-Py^2); 
    H2 = -2*ρ^2*(2*βQ*QQ-αQ*ρ*ρ0_rec)*Q2 + 2*kQ*ΔQ2*ρ^2 + 4*kQ*ρ*(∂xρ*∂xQ2+∂yρ*∂yQ2) + ρ^2*χ*Px*Py;           
    save_field!(H, i, j, H1, H2)

    # f - ρμ = -(Hydrostatic pressure)
    f_ρμ = 0.25*a*ρ^4 + ρ^2*(-0.5*αp*ρ*ρ0_rec*P2 + 0.25*βp*P4 + 0.5*kp*∇P2 
                                -0.5*αQ*ρ*ρ0_rec*QQ +  0.5*βQ*Q4 + 0.5*kQ*∇QQ 
                                +0.5*χ*(Q1*(Py*Py-Px*Px)-2*Q2*Px*Py)) -ρ*μ_ij;
                        
    # Non-viscous stress tensor:
    # σ_nv = sym Ericksen + flow aligment + active + hydrostatic pressure + anti-sym part of Ericksen
    σ_anti      = 0.5*(Px*hy-hx*Py) + 2*(Q1*H2-H1*Q2);
    # xx component of the non-viscous stress
    σ_nv[i,j,1] = f_ρμ + (- ρ^2*kp*(∂xPx^2+∂xPy^2)   + ν1*Px*hx + ν2*(Px*hx+Py*hy) 
                            - ρ^2*kQ*2*(∂xQ1^2+∂xQ2^2) + 2*λ*H1 
                            - ζρ*ρ^3 - ρ*(ζp*(Px*Px-0.5*P2) + 0.5*ζp2*P2 + ζQ*Q1) );
    # yy component of the non-viscous stress
    σ_nv[i,j,4] = f_ρμ + (- ρ^2*kp*(∂yPx^2+∂yPy^2) + ν1*Py*hy + ν2*(Px*hx+Py*hy)
                            - ρ^2*kQ*2*(∂yQ1^2+∂yQ2^2) - 2*λ*H1
                            - ζρ*ρ^3 - ρ*(ζp*(Py*Py-0.5*P2) + 0.5*ζp2*P2 - ζQ*Q1) );
    # Symmetric xy component of the non-viscous stress
    σ_symm      = (- ρ^2*(kp*(∂xPx*∂yPx+∂xPy*∂yPy) + 2*kQ*(∂xQ1*∂yQ1+∂xQ2*∂yQ2)) 
                            + 0.5*ν1*(Px*hy+Py*hx) + 2*λ*H2 - ρ*(ζp*Px*Py + ζQ*Q2));
    σ_nv[i,j,2] = σ_symm + σ_anti;
    σ_nv[i,j,3] = σ_symm - σ_anti;

    return nothing
end


# Determine the velocity field in Fourier space from the Fourier transform of the non-viscous stress.
@parallel_indices (i, j) function UpdateVelocity_Fourier!(FV, Fσ, kx_, ky_)  

    if i <= Lkx
        # Define Fourier factors and derivatives
        kx = kx_[i]; kx2 = kx*kx;
        ky = ky_[j]; ky2 = ky*ky;
        kxy = kx*ky
        Fσxx = Fσ[i,j,1]; Fσxy = Fσ[i,j,2]; 
        Fσyx = Fσ[i,j,3]; Fσyy = Fσ[i,j,4];
        
        # Relate the Fourier transform of the stress to the Fourier transform of the velocity
        fact = 1.0/(2*kx2+ky2+ξ)
        # Fourier transform of Vy
        FVy   = im / (2*ky2-kxy*kxy*fact+kx2+ξ) * (ky*Fσyy + kx*Fσyx - kxy*(kx*Fσxx+ky*Fσxy)*fact)
        # Fourier transform of Vx
        FV[i,j,1] = ( im*(kx*Fσxx+ky*Fσxy)-kxy*FVy)*fact
        FV[i,j,2] = FVy
    end

    return nothing
end


# Dynamics of the polar field. We update P with Euler integration.
@parallel_indices (i, j) function Update_PQ!(P, Q, h, H, ∇P, ∇Q, V, ∇V, ρ_, Δt)
    # Actin density
    ρ = ρ_[i,j];
    
    # Initialize velocity, shear rate and vorticity.
    Vx  = V[i,j,1]; Vy = V[i,j,2]
    Vxx = ∇V[i,j,1]
    Vxy = 0.5*(∇V[i,j,2]+∇V[i,j,3])
    Vyy = ∇V[i,j,4]
    ωxy = 0.5*(∇V[i,j,2]-∇V[i,j,3])
    ωyx = -ωxy
    # Initialize polar, nematic and molecular fields.
    # # Polarity
    Px, Py                  = load_field(P, i, j, Val(2))
    ∂xPx, ∂xPy, ∂yPx, ∂yPy  = load_field(∇P, i, j, Val(4))
    Q1, Q2                  = load_field(Q, i, j, Val(2))
    ∂xQ1, ∂xQ2, ∂yQ1, ∂yQ2  = load_field(∇Q, i, j, Val(4))
    hx, hy                  = load_field(h, i, j, Val(2))
    H1, H2                  = load_field(H, i, j, Val(2))

    # Dynamics of the polar field
    Px_next = Px + Δt*(-Vx*∂xPx - Vy*∂yPx - ωxy*Py - ν1*(Px*Vxx+Py*Vxy) - ν2*Px*(Vxx+Vyy) + Γ*hx + εp*ρ*Px)
    Py_next = Py + Δt*(-Vx*∂xPy - Vy*∂yPy - ωyx*Px - ν1*(Px*Vxy+Py*Vyy) - ν2*Py*(Vxx+Vyy) + Γ*hy + εp*ρ*Py)
    save_field!(P, i, j, Px_next, Py_next)
    # Dynamics of the nematic field
    Q1_next = Q1 + Δt*(-Vx*∂xQ1 - Vy*∂yQ1 - 2*ωxy*Q2 + Γ*H1 - λ*(Vxx-Vyy) + εQ*ρ*Q1)
    Q2_next = Q2 + Δt*(-Vx*∂xQ2 - Vy*∂yQ2 - 2*ωyx*Q1 + Γ*H2 - 2*λ*Vxy     + εQ*ρ*Q2)
    save_field!(Q, i, j, Q1_next, Q2_next)

    return nothing
end