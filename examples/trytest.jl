import LatentGaussianMixtureModel

b=4
Ctrue=2
include("datagen.jl")

lrt = LatentGaussianMixtureModel.EMtest(X, Y, groupindex, Ctrue, ntrials=2, debuginfo=true, ctauparallel=true, ngh=100)
