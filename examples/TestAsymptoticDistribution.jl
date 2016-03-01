#find out the test statistic when m0=1
import LatentGaussianMixtureModel
import Distributions
include(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "examples/datagen.jl"))
#Brun calculate the statistic for one data set;
#b is the the random number seed, from 1 to 100
@everywhere function Brun(b::Integer, Ctrue::Integer, Calternative::Integer; debuginfo::Bool=false, ntrials::Int=25)
    n = 500
    srand(100)
    n_ij = round(Int64, Distributions.rand(Distributions.Poisson(5), n).+Distributions.rand(Distributions.Exponential(95), n))
    N = sum(n_ij); J=2

    showpower = Ctrue>=Calternative
    X, Y, groupindex, wi_true, mu_true, sigmas_true, betas_true, m, gamma_true = datagen(n, Ctrue, N, J, n_ij, b=b, adjustmean=false, showpower=showpower)
    
    lr=LatentGaussianMixtureModel.EMtest(X, Y, groupindex, Calternative-1, ntrials=ntrials, debuginfo=debuginfo, ctauparallel=false, ngh=100)
    println("Mission $b completed with lr=$lr")
    lr
end
