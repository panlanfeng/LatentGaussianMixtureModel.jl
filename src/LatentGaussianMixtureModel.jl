module LatentGaussianMixtureModel

using StatsModels
using Distributions
using StatsBase, StatsFuns
using FastGaussQuadrature
using Yeppp
using Printf
using Distributed

import GaussianMixtureTest: gmm, pn, decidepenalty
import StatsBase: RealArray, RealVector, RealMatrix, IntegerArray, IntegerVector, IntegerMatrix, IntUnitRange
import StatsFuns: logπ


import StatsBase: coef, confint, deviance, loglikelihood, nobs, stderror, vcov, predict, fit!, model_response, RegressionModel, coeftable

export LGMModel, ModelTable,
    initialize!, latentgmm, EMtest, asymptoticdistribution,
    ranefmixture, infomatrix, multiplefit!, ranef!,
    FDR, detect, latexprint,
    fit!, coef, confint, deviance, loglikelihood, nobs, stderror, vcov, predict, model_response, coeftable


import Base.show
using DataFrames

include("tableprint.jl")
include("model.jl")

#include("MCEM.jl")

include("arithmetic.jl")
include("utility.jl")
include("EM.jl")
#include("FractionalImputation.jl")

PKGVERSION=v"0.4.1"
export PKGVERSION
end # module
