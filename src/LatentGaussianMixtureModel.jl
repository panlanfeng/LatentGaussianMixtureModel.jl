module LatentGaussianMixtureModel


using Distributions
using StatsBase, StatsFuns
using NLopt
using GaussQuadrature
using Yeppp
#import Yeppp: add!, exp!, log!
import StatsBase: RealArray, RealVector, RealArray, IntegerArray, IntegerVector, IntegerMatrix, IntUnitRange
export gmm, predictgamma, latentgmm, latentgmm_ctau, latentgmmEM, latentgmmFI, marginallikelihood, loglikelihoodratio, loglikelihoodratio_ctau, loglikelihoodratioEM, maxposterior, decidepenalty
include("arithmetic.jl")
include("utility.jl")
include("MCEM.jl")
include("EM.jl")
include("FractionalImputation.jl")
# package code goes here

end # module
