module LatentGaussianMixtureModel


using Distributions
using StatsBase
using NLopt
using FastGaussQuadrature
using Yeppp
import Yeppp: add!, exp!, log!
export gmm, latentgmm, latentgmm_ctau, marginallikelihood, loglikelihoodratio, loglikelihoodratio_ctau, maxposterior, decidepenalty
include("arithmetic.jl")
include("utility.jl")
# package code goes here

end # module
