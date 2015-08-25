module LatentGaussianMixtureModel


using Distributions
using StatsBase
using NLopt
if !("FastGaussQuadrature" in keys(Pkg.installed()))
    Pkg.clone("https://github.com/panlanfeng/FastGaussQuadrature.jl.git")
end
using FastGaussQuadrature
import Yeppp: add!, exp!, log!
export gmm, latentgmm, latentgmm_ctau, marginallikelihood, loglikelihoodratio, loglikelihoodratio_ctau
include("utility.jl")
# package code goes here

end # module
