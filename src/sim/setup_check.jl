# What a run uses, printed in the log and written next to the data: included by InputParams.jl
# once the backend is set up (it needs USE_GPU and TF). Reads only metadata: well under a second.

"""
    check_setup(outdir) -> String

Print, and write to `outdir/setup.txt`, the Julia version, threads, host, HydraFluids git
commit, package versions, backend, float type and (on CUDA) the GPU. Nothing here can stop a
run: a missing piece of information is reported as unknown.
"""
function check_setup(outdir::AbstractString)
    io = IOBuffer()
    println(io, "Julia ", VERSION, ", threads ", Threads.nthreads(), ", host ", gethostname())
    # the commit: from git, else from HYDRAFLUIDS_COMMIT or a commit.txt written when the code was
    # copied (copies, as on the cluster or in test/smoke.jl, have no .git)
    commit = try
        readchomp(pipeline(`git -C $(@__DIR__) rev-parse --short HEAD`; stderr=devnull))
    catch
        file = joinpath(@__DIR__, "commit.txt")
        get(ENV, "HYDRAFLUIDS_COMMIT", isfile(file) ? strip(read(file, String)) : "unknown")
    end
    println(io, "HydraFluids commit ", commit)
    wanted = ("CSV", "DataFrames", "JLD2", "ParallelStencil", "FFTW", "Roots", "Glob", "CUDA", "Metal")
    for info in sort(collect(values(Pkg.dependencies())); by = i -> i.name)
        info.name in wanted && println(io, "  ", rpad(info.name, 16), info.version)
    end
    backend = USE_GPU ? (Sys.isapple() ? "Metal" : "CUDA") : "CPU threads"
    println(io, "backend ", backend, ", float type ", TF)
    try
        if USE_GPU && !Sys.isapple()
            C = Main.CUDA
            dev = C.device()
            println(io, "CUDA functional ", C.functional(), ", device ", C.name(dev), ", memory ",
                    round(C.totalmem(dev) / 2^30; digits = 1), " GiB, runtime ", C.runtime_version())
        elseif USE_GPU
            println(io, "Metal device ", Main.Metal.current_device().name)
        end
    catch err
        println(io, "GPU information unavailable: ", sprint(showerror, err))
    end
    s = String(take!(io))
    print(s)
    try
        mkpath(outdir)
        write(joinpath(outdir, "setup.txt"), s)
    catch err
        @warn "could not write setup.txt" exception = err
    end
    return s
end
