#julia4 --machinefile=$PBS_NODEFILE -- run100.jl 1 2 110
#run on all available cores using:
#addprocs(readcsv(ENV["PBS_NODEFILE"], ASCIIString)[:, 1])

if nprocs() < 2
    if ARGS[4] == "slurm"
        using ClusterManagers
        np=parse(Int, ENV["SLURM_NTASKS"])
        addprocs(SlurmManager(np))
    elseif ARGS[4] == "pbs"
        addprocs(readcsv(ENV["PBS_NODEFILE"], ASCIIString)[:, 1])
    end
end

@everywhere args = @fetchfrom 1 ARGS

include(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "teststat200/TestAsymptoticDistribution.jl"))
@everywhere Ctrue = parse(Int, args[1])
@everywhere Calternative = parse(Int, args[2])
teststat = pmap((b) -> Brun(b, Ctrue, Calternative, showpower=true), 1:parse(Int, args[3]))
writecsv("seq$(Ctrue)$(Calternative).csv", teststat)
run(`sed -i "s/(//g" seq$(Ctrue)$(Calternative).csv`)
run(`sed -i "s/)//g" seq$(Ctrue)$(Calternative).csv`)
