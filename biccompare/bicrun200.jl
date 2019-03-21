#julia4 --machinefile=$PBS_NODEFILE -- run100.jl 1 2 110
#run on all available cores using:
#addprocs(readcsv(ENV["PBS_NODEFILE"], ASCIIString)[:, 1])
@everywhere args = @fetchfrom 1 ARGS
import LatentGaussianMixtureModel
#include(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "biccompare/runbic.jl"))
include(joinpath(dirname(pathof(LatentGaussianMixtureModel)), "..", "biccompare/runbic.jl"))
@everywhere C=parse(Int,args[2])
teststat = pmap((b) -> Brun(b, C), 1:parse(Int, args[1]))
writecsv("bic$(C).csv", teststat)
run(`sed -i "s/\[//g" bic$(C).csv`)
run(`sed -i "s/\]//g" bic$(C).csv`)
