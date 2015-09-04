module LatentGaussianMixtureModel


using Distributions
using StatsBase
using NLopt
if !("FastGaussQuadrature" in keys(Pkg.installed()))
    Pkg.clone("https://github.com/ajt60gaibb/FastGaussQuadrature.jl.git")
end
using FastGaussQuadrature
using Yeppp
import Yeppp: add!, exp!, log!
export eachrepeat, gmm, latentgmm, latentgmm_ctau, marginallikelihood, loglikelihoodratio, loglikelihoodratio_ctau, maxposterior
include("utility.jl")
# package code goes here

end # module
