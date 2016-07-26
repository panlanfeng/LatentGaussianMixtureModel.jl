#julia4 --machinefile=$PBS_NODEFILE -- run100.jl 1 2 110
#run on all available cores using:
addprocs(readcsv(ENV["PBS_NODEFILE"], ASCIIString)[:, 1])
@everywhere args = @fetchfrom 1 ARGS

include(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "challenge/challengesim.jl"))
Xraw=readcsv(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "challenge/X.csv"))
groupindex_raw=readcsv(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "challenge/groupindex.csv"), Int)[:,1]
@everywhere X = @fetchfrom 1 Xraw
@everywhere groupindex = @fetchfrom 1 groupindex_raw

teststat = pmap((b) -> Brun(b, X, groupindex), 1:parse(Int, args[1]))
writecsv("teststat.csv", teststat)
run(`sed -i "s/(//g" teststat.csv`)
run(`sed -i "s/)//g" teststat.csv`)
