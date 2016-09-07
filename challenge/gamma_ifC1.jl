#find out the test statistic when m0=1
import LatentGaussianMixtureModel
import Distributions, StatsBase
import Yeppp

@everywhere using LatentGaussianMixtureModel
@everywhere using Distributions, Yeppp, StatsBase

#Brun calculate the statistic for one data set;
#b is the the random number seed, from 1 to 100
@everywhere function Brun(b::Integer, X::Matrix{Float64},
    groupindex::Vector; debuginfo::Bool=false)

    mu_true = [-0.979]
    wi_true =  [1.0]
    sigmas_true = [0.26]
    betas_true=[0.019498134778826972,0.007110782868926634,0.030919753747641616,-0.26957245719168493,-0.5257588591653842,-0.6314135703754931,-0.799509687993618,0.07786105411635978,0.12059632526169425,0.22504173963135948]

    #Randomly data generation based on the setting on datagen.jl
    srand(b * 100)
    n = Int(maximum(groupindex))
    m = MixtureModel(map((u, v) -> Normal(u, v), mu_true, sigmas_true), wi_true)
    gamma_true = rand(m, n)

    prob = exp(gamma_true[groupindex] .+ X*betas_true)
    prob= prob ./ (1 .+ prob)
    Y = Bool[rand(Binomial(1, prob[i])) == 1 for i in 1:length(groupindex)];

    m = LGMModel(X, Y, groupindex, 2)
    LatentGaussianMixtureModel.fit!(m)
    LatentGaussianMixtureModel.ranef!(m)
    sort(m.gammaprediction)[1:10]
end
