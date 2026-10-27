# Physics tests of HydraFluids. Each case runs in a temporary copy of src/sim with its own DF.csv
# (src/sim/DF.csv is never touched), in a separate Julia process, since the parameters are
# constants. CPU only, N = 32: a few minutes in all, mostly compilation.
#
#   torque      With κp = κQ = 0, the antisymmetric stress vanishes: the free energy is invariant
#               under a joint rotation of P and Q, so it exerts no bulk torque.
#   energy      Without renewal and activity: over one short step, dF/dt equals minus the
#               dissipation (energy_balance, diffusion = mu); over a run, F only decreases
#               (energy_mu, energy_fick). All couplings, flow alignment included, are tested.
#   relaxation  Near the isotropic state (κ = 0, uniform density), P and Q grow at the rates
#               ρ0²αp/Γp and ρ0²αQ/ΓQ, with independent Γp and ΓQ.
#
# From the repository root:
#     julia test/physics.jl                    # all cases
#     julia test/physics.jl torque energy      # chosen cases only

const SIM = normpath(joinpath(@__DIR__, "..", "src", "sim"))
include(joinpath(SIM, "utils.jl"))                     # ensure_installed
ensure_installed("CSV", "DataFrames", "JLD2", "FFTW")
using JLD2, FFTW, Printf

const COLUMNS = ["fn", "N", "dx", "dt_max", "dt_ini", "t_check", "t_print", "t_end",
    "initialisation", "eta_rho", "eta_p", "eta_q", "seed", "orientation", "a", "zeta", "D", "tau",
    "rho0", "alpha_p", "beta_p", "kappa_p", "zeta_p", "zeta_p2", "gamma_p", "nu", "nu2", "epsi_p",
    "alpha_q", "beta_q", "kappa_q", "zeta_q", "gamma_q", "lambda", "epsi_q", "chi", "solver",
    "diffusion"]

# Common values; each case overrides some. Columns not listed are 0. tau = 1e30: no renewal.
base_row() = Dict{String,Any}("fn" => 1, "N" => 32, "dx" => 0.05, "dt_max" => 0.005,
    "dt_ini" => 0.001, "t_check" => 0.1, "t_print" => 1.0, "t_end" => 2.0,
    "initialisation" => "Homogeneous", "eta_rho" => 0.01, "eta_p" => 0.1, "eta_q" => 0.1,
    "seed" => 1, "orientation" => "nematopolar", "a" => 4/3, "D" => 0.0, "tau" => 1e30,
    "rho0" => 1.0, "alpha_p" => 0.2, "beta_p" => 0.5, "kappa_p" => 1e-3, "gamma_p" => 1.0,
    "alpha_q" => 0.3, "beta_q" => 0.6, "kappa_q" => 1e-3, "gamma_q" => 1.0, "chi" => 0.2,
    "solver" => "fft", "diffusion" => "fick")

function sim_copy(row)
    tmp = mktempdir()
    sim = mkpath(joinpath(tmp, "sim"))
    for f in readdir(SIM)
        endswith(f, ".jl") && cp(joinpath(SIM, f), joinpath(sim, f))
    end
    open(joinpath(sim, "DF.csv"), "w") do io
        println(io, join(COLUMNS, ","))
        println(io, join((get(row, c, 0.0) for c in COLUMNS), ","))
    end
    return sim, joinpath(tmp, "out")
end

# Run `script` (a file in the copy) on the CPU; returns (success, path of the log).
function run_sim(sim, out, script)
    env = copy(ENV)
    env["SLURM_ARRAY_TASK_ID"] = "1"; env["path_to_data"] = out; env["use_gpu"] = "false"
    env["HYDRAFLUIDS_COMMIT"] = "physics-test"
    project = dirname(Base.active_project())
    cmd = setenv(`$(Base.julia_cmd()) --startup-file=no --project=$project -t auto $script`, env; dir = sim)
    log = joinpath(dirname(out), "run.log")
    ok = success(pipeline(cmd; stdout = log, stderr = log))
    ok || println("--- end of the log:\n", join(last(readlines(log), 20), "\n"))
    return ok, log
end

# Snapshots of a run, sorted by time; the file name is the time (rounded to an integer).
function snapshots(out)
    data = joinpath(out, "Data")
    files = sort(filter(endswith(".jld"), readdir(data; join = true)))
    return [(parse(Int, splitext(basename(f))[1]), JLD2.load(f)) for f in files]
end

# Spectral gradient of a periodic field (x along the first index, as in the code).
function grad(f, dx)
    k = 2π .* fftfreq(size(f, 1), 1 / dx)
    F = fft(f)
    return real(ifft(im .* k .* F)), real(ifft(im .* transpose(k) .* F))
end

# Total free energy of a snapshot: f_ρ + f_p + f_Q + f_pQ as in GenInputParams.pluto.jl, with
# (∂Q)² = (∂γQαβ)(∂γQαβ) = 2[(∂Q1)² + (∂Q2)²].
function free_energy(s, r)
    ρ = s["C"]; dx = r["dx"]; ρ0 = r["rho0"]
    f = r["a"] / 4 .* ρ .^ 4
    if haskey(s, "P")
        px, py = s["P"][:, :, 1], s["P"][:, :, 2]
        p2 = px .^ 2 .+ py .^ 2
        gp = zeros(size(ρ))
        for c in (px, py), d in grad(c, dx); gp .+= d .^ 2; end
        f .+= ρ .^ 2 .* (-r["alpha_p"] / 2 .* ρ ./ ρ0 .* p2 .+ r["beta_p"] / 4 .* p2 .^ 2 .+ r["kappa_p"] / 2 .* gp)
    end
    if haskey(s, "Q")
        q1, q2 = s["Q"][:, :, 1], s["Q"][:, :, 2]
        QQ = 2 .* (q1 .^ 2 .+ q2 .^ 2)
        gq = zeros(size(ρ))
        for c in (q1, q2), d in grad(c, dx); gq .+= 2 .* d .^ 2; end
        f .+= ρ .^ 2 .* (-r["alpha_q"] / 2 .* ρ ./ ρ0 .* QQ .+ r["beta_q"] / 2 .* QQ .^ 2 .+ r["kappa_q"] / 2 .* gq)
    end
    if haskey(s, "P") && haskey(s, "Q")
        px, py = s["P"][:, :, 1], s["P"][:, :, 2]
        q1, q2 = s["Q"][:, :, 1], s["Q"][:, :, 2]
        f .+= r["chi"] / 2 .* ρ .^ 2 .* (q1 .* (py .^ 2 .- px .^ 2) .- 2 .* q2 .* px .* py)
    end
    return sum(f) * dx^2
end

# ------------------------------------------------------------------------------------------------
# torque: call Compute_stress! once on random fields and compare σ_xy with σ_yx.
const TORQUE_DRIVER = raw"""
include("InputParams.jl"); include("kernels.jl")
rng = Xoshiro(2)
r(dims...) = Data.Array(TF.(rand(rng, dims...) .- 0.5))
ρ = Data.Array(TF.(0.5 .+ rand(rng, N, N)))
P, Q, ΔP, ΔQ = r(N, N, 2), r(N, N, 2), r(N, N, 2), r(N, N, 2)
∇ρ, ∇P, ∇Q = r(N, N, 2), r(N, N, 4), r(N, N, 4)
σ = @zeros(N, N, 4); h = @zeros(N, N, 2); H = @zeros(N, N, 2)
if isdefined(Main, :Diffusion_trait)          # current kernels also store μ
    @parallel blocks threads Compute_stress!(σ, h, H, nothing, ρ, ∇ρ, P, ∇P, ΔP, Q, ∇Q, ΔQ)
else                                         # kernels before the μ argument
    @parallel blocks threads Compute_stress!(σ, h, H, ρ, ∇ρ, P, ∇P, ΔP, Q, ∇Q, ΔQ)
end
s = Array(σ)
println("TORQUE ", maximum(abs, s[:, :, 2] .- s[:, :, 3]), " ", maximum(abs, s))
"""

function test_torque()
    row = merge(base_row(), Dict("kappa_p" => 0.0, "kappa_q" => 0.0, "nu" => 0.3, "nu2" => 0.2,
                                 "lambda" => 0.4, "zeta" => -1.0, "zeta_p" => 0.5, "zeta_q" => 0.7))
    sim, out = sim_copy(row)
    write(joinpath(sim, "torque.jl"), TORQUE_DRIVER)
    ok, log = run_sim(sim, out, "torque.jl")
    ok || return (false, "run failed")
    line = only(filter(startswith("TORQUE"), readlines(log)))
    asym, scale = parse.(Float64, split(line)[2:3])
    return (asym <= 1e-12 * scale, @sprintf("max|σxy-σyx| = %.2e, max|σ| = %.2e", asym, scale))
end

# ------------------------------------------------------------------------------------------------
# energy: free energy at every snapshot of a passive run without renewal.
function energy_case(over)
    row = merge(base_row(), Dict("t_end" => 15.0), over)
    sim, out = sim_copy(row)
    ok, _ = run_sim(sim, out, "main.jl")
    ok || return (false, "run failed")
    F = [free_energy(s, row) for (_, s) in snapshots(out)]
    dF = diff(F)
    tol = 1e-10 * maximum(abs, F)
    passed = all(dF .<= tol) && F[end] < F[1]
    return (passed, @sprintf("F: %.6e → %.6e over %d snapshots, largest increase %.1e",
                             F[1], F[end], length(F), max(maximum(dF), 0.0)))
end

# energy balance: one short Euler step from smooth random fields, compared with the dissipation
#   dF/dt = -∫ [ ξ|v|² + 2 v_αβ v_αβ + |h|²/Γp + (H1² + H2²)/(2ΓQ) + D|∇μ|² ]      (η = 1)
# for diffusion = mu, no renewal, no activity, flow alignment included. A wrong factor in any
# reactive coupling (rotation, antisymmetric stress, flow alignment) or relaxation breaks it.
const BALANCE_DRIVER = raw"""
include("InputParams.jl"); include("kernels.jl"); include("operators.jl")
rng = Xoshiro(3)
function smooth(amp, offset)                    # a few low Fourier modes
    f = fill(offset, N, N)
    for _ in 1:6
        kx, ky = rand(rng, -2:2), rand(rng, -2:2); ph = 2π*rand(rng); a = amp*(rand(rng) - 0.5)
        f .+= [a*cos(2π*(kx*i + ky*j)/N + ph) for i in 1:N, j in 1:N]
    end
    return TF.(f)
end
ρ = Data.Array(smooth(0.2, 1.0))
P = Data.Array(cat(smooth(0.4, 0.3), smooth(0.4, 0.1); dims = 3))
Q = Data.Array(cat(smooth(0.3, 0.2), smooth(0.3, 0.0); dims = 3))
∇ρ, Δρ = @zeros(N, N, 2), @zeros(N, N)
∇P, ΔP, ∇Q, ΔQ = @zeros(N, N, 4), @zeros(N, N, 2), @zeros(N, N, 4), @zeros(N, N, 2)
V, ∇V, σ = @zeros(N, N, 2), @zeros(N, N, 4), @zeros(N, N, 4)
h, H, μ, Δμ = @zeros(N, N, 2), @zeros(N, N, 2), @zeros(N, N), @zeros(N, N)
∂xρVx, ∂yρVy = @zeros(N, N), @zeros(N, N)
fx, fy, fΔ = TA(zeros(TC, Lkx, N)), TA(zeros(TC, Lkx, N)), @zeros(Lkx, N)
@parallel blocks_FFT threads compute_FFTderivative_factors!(fx, fy, fΔ, kx, ky, kx2, ky2)
work = (W = W, Wi = Wi, factor_∂x = fx, factor_∂y = fy, factor_Δ = fΔ,
        FV = TA(zeros(TC, Lkx, N, 2)), Fσ = TA(zeros(TC, Lkx, N, 4)),
        Rtmp = @zeros(N, N), F0 = TA(zeros(TC, Lkx, N)), Fa = TA(zeros(TC, Lkx, N)),
        G = @zeros(N, N, 2), v_temp = @zeros(N, N, 2))
function free_energy()                          # uses ∇P, ∇Q of the current fields
    p2 = P[:, :, 1].^2 .+ P[:, :, 2].^2; QQ = 2 .* (Q[:, :, 1].^2 .+ Q[:, :, 2].^2)
    gp = sum(∇P .^ 2; dims = 3)[:, :, 1]; gq = 2 .* sum(∇Q .^ 2; dims = 3)[:, :, 1]
    f = a/4 .* ρ.^4 .+ ρ.^2 .* (-αp/2 .* ρ ./ ρ0 .* p2 .+ βp/4 .* p2.^2 .+ kp/2 .* gp
                                .- αQ/2 .* ρ ./ ρ0 .* QQ .+ βQ/2 .* QQ.^2 .+ kQ/2 .* gq
                                .+ χ/2 .* (Q[:, :, 1] .* (P[:, :, 2].^2 .- P[:, :, 1].^2) .- 2 .* Q[:, :, 2] .* P[:, :, 1] .* P[:, :, 2]))
    return sum(f) * Δ2
end
compute_gradients!(Solver_trait, ρ, ∇ρ, Δρ, P, ∇P, ΔP, Q, ∇Q, ΔQ, work)
F0 = free_energy()
@parallel blocks threads Compute_stress!(σ, h, H, μ, ρ, ∇ρ, P, ∇P, ΔP, Q, ∇Q, ΔQ)
ΔD = diffusion_laplacian!(Diffusion_trait, Solver_trait, Δρ, Δμ, μ, work)
solve_velocity!(Solver_trait, V, ∇V, σ, work)
k = 2π .* FFTW.fftfreq(N, 1/Δ); Fμ = FFTW.fft(μ)
μx = real(FFTW.ifft(im .* k .* Fμ)); μy = real(FFTW.ifft(im .* transpose(k) .* Fμ))
vxy = (∇V[:, :, 2] .+ ∇V[:, :, 3]) ./ 2
diss = sum(ξ .* (V[:, :, 1].^2 .+ V[:, :, 2].^2) .+ 2 .* (∇V[:, :, 1].^2 .+ ∇V[:, :, 4].^2 .+ 2 .* vxy.^2)
           .+ inv_gammap .* (h[:, :, 1].^2 .+ h[:, :, 2].^2) .+ inv_gammaQ/2 .* (H[:, :, 1].^2 .+ H[:, :, 2].^2)
           .+ D0 .* (μx.^2 .+ μy.^2)) * Δ2
dt = TF(1e-7)
@parallel blocks threads Update_PQ!(P, Q, h, H, ∇P, ∇Q, V, ∇V, ρ, dt)
density_advection!(Solver_trait, ∂xρVx, ∂yρVy, ρ, V, work)
@. ρ += dt*(-∂xρVx - ∂yρVy + D0*ΔD - Rd*(ρ - ρ0))
compute_gradients!(Solver_trait, ρ, ∇ρ, Δρ, P, ∇P, ΔP, Q, ∇Q, ΔQ, work)
println("BALANCE ", (free_energy() - F0)/dt, " ", -diss)
"""

function test_energy_balance()
    row = merge(base_row(), Dict("diffusion" => "mu", "D" => 1e-3, "nu" => 0.3, "nu2" => 0.2,
                                 "lambda" => 0.4, "gamma_p" => 1.5, "gamma_q" => 0.7, "chi" => 1.0))
    sim, out = sim_copy(row)
    write(joinpath(sim, "balance.jl"), BALANCE_DRIVER)
    ok, log = run_sim(sim, out, "balance.jl")
    ok || return (false, "run failed")
    line = only(filter(startswith("BALANCE"), readlines(log)))
    dFdt, expected = parse.(Float64, split(line)[2:3])
    return (abs(dFdt / expected - 1) < 1e-5,
            @sprintf("dF/dt = %.8e, -dissipation = %.8e", dFdt, expected))
end

test_energy_mu() = energy_case(Dict("diffusion" => "mu", "D" => 1e-3, "nu" => 0.3, "nu2" => 0.2,
                                    "lambda" => 0.4, "gamma_p" => 1.5, "gamma_q" => 0.7))
test_energy_fick() = energy_case(Dict("orientation" => "nematic", "diffusion" => "fick", "D" => 0.0,
                                      "lambda" => 0.4, "gamma_q" => 0.7, "chi" => 0.0,
                                      "alpha_p" => 0.0, "beta_p" => 0.0, "kappa_p" => 0.0,
                                      "gamma_p" => 0.0, "eta_p" => 0.0))

# ------------------------------------------------------------------------------------------------
# relaxation: exponential growth of P and Q from small noise about the isotropic state.
function test_relaxation()
    row = merge(base_row(), Dict("eta_rho" => 0.0, "eta_p" => 1e-4, "eta_q" => 1e-4,
        "kappa_p" => 0.0, "kappa_q" => 0.0, "chi" => 0.0, "alpha_p" => 0.3, "beta_p" => 1.0,
        "alpha_q" => 0.5, "beta_q" => 1.0, "gamma_p" => 1.5, "gamma_q" => 2.0,
        "dt_max" => 0.002, "t_end" => 3.0))
    sim, out = sim_copy(row)
    ok, _ = run_sim(sim, out, "main.jl")
    ok || return (false, "run failed")
    snaps = Dict(snapshots(out))
    rate(f) = log(sqrt(sum(abs2, snaps[3][f])) / sqrt(sum(abs2, snaps[1][f]))) / 2
    rp, rq = rate("P"), rate("Q")
    ep = row["rho0"]^2 * row["alpha_p"] / row["gamma_p"]
    eq = row["rho0"]^2 * row["alpha_q"] / row["gamma_q"]
    passed = abs(rp / ep - 1) < 0.01 && abs(rq / eq - 1) < 0.01
    return (passed, @sprintf("P: %.4f (expected %.4f), Q: %.4f (expected %.4f)", rp, ep, rq, eq))
end

const CASES = ["torque" => test_torque, "energy_balance" => test_energy_balance, "energy_mu" => test_energy_mu,
               "energy_fick" => test_energy_fick, "relaxation" => test_relaxation]

function main(args)
    chosen = isempty(args) ? CASES :
             filter(c -> any(a -> startswith(first(c), a), lowercase.(args)), CASES)
    results = Pair{String,Tuple{Bool,String}}[]
    for (name, f) in chosen                  # one Julia process at a time
        println("running ", name, " ..."); flush(stdout)
        push!(results, name => f())
    end
    println()
    for (name, (passed, msg)) in results
        println(rpad(name, 16), rpad(passed ? "PASS" : "FAIL", 6), msg)
    end
    all(first(last(r)) for r in results) || exit(1)
end

main(ARGS)
