module LatentGaussianMixtureModel


using Distributions
using StatsBase, StatsFuns
using NLopt
using FastGaussQuadrature
using Yeppp
#import Yeppp: add!, exp!, log!
import StatsBase: RealArray, RealVector, RealArray, IntegerArray, IntegerVector, IntegerMatrix, IntUnitRange
import StatsFuns: logπ
export gmm, predictgamma, asymptoticdistribution, latentgmmEM, marginallikelihood, loglikelihoodratioEM, decidepenalty
include("arithmetic.jl")
include("utility.jl")
#include("MCEM.jl")
include("EM.jl")
#include("FractionalImputation.jl")
# package code goes here

end # module
