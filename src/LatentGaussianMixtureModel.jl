module LatentGaussianMixtureModel


using Distributions
using StatsBase
using NLopt
using FastGaussQuadrature
using Yeppp
import Yeppp: add!, exp!, log!
import StatsBase: RealArray, RealVector, RealArray, IntegerArray, IntegerVector, IntegerMatrix, IntUnitRange
export gmm, latentgmm, latentgmm_ctau, marginallikelihood, loglikelihoodratio, loglikelihoodratio_ctau, maxposterior, decidepenalty
include("arithmetic.jl")
include("utility.jl")
include("EM.jl")
# package code goes here

end # module
