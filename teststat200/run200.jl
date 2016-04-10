#julia4 --machinefile=$PBS_NODEFILE -- run100.jl 1 2 110
#run on all available cores using:
addprocs(readcsv(ENV["PBS_NODEFILE"], ASCIIString)[:, 1])
@everywhere args = @fetchfrom 1 ARGS

include(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "examples/TestAsymptoticDistribution.jl"))
@everywhere Ctrue = parse(Int, args[1])
@everywhere Calternative = parse(Int, args[2])
teststat = pmap((b) -> Brun(b, Ctrue, Calternative), 1:parse(Int, args[3]))
writecsv("teststat$(Ctrue)$(Calternative).csv", teststat)
run(`sed -i "s/(//g" teststat$(Ctrue)$(Calternative).csv`)
run(`sed -i "s/)//g" teststat$(Ctrue)$(Calternative).csv`)
