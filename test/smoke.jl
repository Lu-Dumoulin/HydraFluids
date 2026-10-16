# Smoke test of HydraFluids: a tiny run (N = 64, t = 2) on each backend, each in a temporary
# copy of src/sim, so that src/sim/DF.csv is never touched. Checks that the packages load, the
# kernels compile, snapshots are written and every field is finite. Run it before a job array.
#
# From the repository root:
#     julia test/smoke.jl              # CPU, plus CUDA (if nvidia-smi exists) or Metal (macOS)
#     julia test/smoke.jl cpu cuda     # chosen backends only
# The runs use the active Julia environment (packages missing there are installed, as usual).

const SIM = normpath(joinpath(@__DIR__, "..", "src", "sim"))
const N = 64

# one tiny polar run; columns not listed here are 0. Metal has no FFT: it uses jacobi.
# t_print >= 1: file names hold the time as an integer, so shorter intervals overwrite files.
function smoke_row(backend)
    Dict("fn" => "1", "N" => "$N", "dx" => "0.015625", "dt_max" => "0.01", "dt_ini" => "0.001",
         "t_check" => "0.1", "t_print" => "1.0", "t_end" => "2.0", "initialisation" => "Homogeneous",
         "eta_rho" => "0.001", "eta_p" => "0.001", "seed" => "1", "orientation" => "polar",
         "a" => "1.3333333333333333", "zeta" => "-1.0", "D" => "0.0001", "tau" => "1.0", "rho0" => "1.0",
         "alpha_p" => "0.1", "beta_p" => "0.1", "kappa_p" => "0.0001", "gamma_p" => "1.0",
         "solver" => backend == "metal" ? "jacobi" : "fft")
end

# the columns of a parameter table, as written by GenInputParams.pluto.jl (src/sim/DF.csv
# itself is not in git), plus the optional column solver
const COLUMNS = ["fn", "N", "dx", "dt_max", "dt_ini", "t_check", "t_print", "t_end",
    "initialisation", "eta_rho", "eta_p", "eta_q", "seed", "orientation", "a", "zeta", "D", "tau",
    "rho0", "alpha_p", "beta_p", "kappa_p", "zeta_p", "zeta_p2", "gamma_p", "nu", "nu2", "epsi_p",
    "alpha_q", "beta_q", "kappa_q", "zeta_q", "gamma_q", "lambda", "epsi_q", "chi", "solver"]

function write_table(path, backend)
    header = COLUMNS
    row = smoke_row(backend)
    open(path, "w") do io
        println(io, join(header, ","))
        println(io, join((get(row, String(c), "0.0") for c in header), ","))
    end
end

function all_finite(file)
    try
        @eval import JLD2                                       # loaded now, used in the latest world
        d = Base.invokelatest(f -> Main.JLD2.load(f), file)
        return all(all(isfinite, v) for v in values(d) if v isa AbstractArray{<:Real})
    catch err
        @warn "could not read $(basename(file)) with JLD2" exception = err
        return missing
    end
end

function smoke(backend)
    tmp = mktempdir()
    sim = mkpath(joinpath(tmp, "sim"))
    for f in readdir(SIM)
        endswith(f, ".jl") && cp(joinpath(SIM, f), joinpath(sim, f))
    end
    write_table(joinpath(sim, "DF.csv"), backend)
    out = joinpath(tmp, "out")
    env = copy(ENV)
    env["SLURM_ARRAY_TASK_ID"] = "1"
    env["path_to_data"] = out
    env["use_gpu"] = backend == "cpu" ? "false" : "true"
    try                                                       # the copy has no .git: pass the commit
        env["HYDRAFLUIDS_COMMIT"] = readchomp(pipeline(`git -C $SIM rev-parse --short HEAD`; stderr = devnull))
    catch                                                     # no git here: commit.txt, if any, is used
    end
    isfile(joinpath(SIM, "commit.txt")) && cp(joinpath(SIM, "commit.txt"), joinpath(sim, "commit.txt"))
    project = dirname(Base.active_project())
    cmd = setenv(`$(Base.julia_cmd()) --startup-file=no --project=$project -t 2 main.jl`, env; dir = sim)
    log = joinpath(tmp, "run.log")
    t = @elapsed ok = success(pipeline(cmd; stdout = log, stderr = log))
    data = joinpath(out, "Data")
    snaps = isdir(data) ? filter(endswith(".jld"), readdir(data; join = true)) : String[]
    finite = isempty(snaps) ? missing : all_finite(last(snaps))
    passed = ok && length(snaps) >= 2 && finite === true
    setup = joinpath(out, "setup.txt")                        # written by check_setup
    isfile(setup) && println("--- $backend setup:\n", read(setup, String))
    passed || println("--- end of the log of the $backend run:\n", join(last(readlines(log), 15), "\n"))
    return (; backend, passed, ok, snapshots = length(snaps), finite, seconds = round(t; digits = 1))
end

function main(args)
    backends = isempty(args) ? ["cpu"; Sys.isapple() ? ["metal"] :
                                       Sys.which("nvidia-smi") === nothing ? String[] : ["cuda"]] :
                               lowercase.(args)
    results = [smoke(b) for b in backends]
    println("\nbackend   result  exit ok  snapshots  finite  time [s]")
    for r in results
        println(rpad(r.backend, 10), rpad(r.passed ? "PASS" : "FAIL", 8), rpad(r.ok, 9),
                rpad(r.snapshots, 11), rpad(r.finite, 8), r.seconds)
    end
    all(r.passed for r in results) || exit(1)
end

main(ARGS)
