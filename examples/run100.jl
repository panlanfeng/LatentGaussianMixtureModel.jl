#run on all available cores using:
include(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "examples/TestAsymptoticDistribution.jl"))
@everywhere Ctrue = 2
@everywhere Calternative = 3
teststat = pmap((b) -> Brun(b, Ctrue, Calternative), 1:100)
writecsv("teststat$(Calternative)$(Ctrue).csv", teststat)
