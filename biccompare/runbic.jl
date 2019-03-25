#find out the test statistic when m0=1
import LatentGaussianMixtureModel
import Distributions, StatsBase
import Yeppp
using Distributed

@everywhere using LatentGaussianMixtureModel
@everywhere using Distributions, Yeppp, StatsBase, Random


#Brun calculate the statistic for one data set;
#b is the the random number seed, from 1 to 100
@everywhere function Brun(b::Int,Ctrue::Int;
    debuginfo::Bool=false,ncomponent::Int=5,Cn::Real=3.0)
    nF = 282
    Random.seed!(100)
    n_ij = round.(Int64, rand(Poisson(55), 282).+rand(Exponential(95), 282))
    N = sum(n_ij)

    groupindex = inverse_rle(1:nF, n_ij)
    J=2  #42 #beta dimension
    Random.seed!(200*b)
    X = rand(Distributions.Normal(), N, 2)
    betas_true=ones(J)

    mu_all = log(1/0.779 - 1)
    if Ctrue == 2
        mu_true = [mu_all - 1.0, mu_all + 0.8]
        wi_true =  [.6, .4]
        sigmas_true = [1.2, .8]
    elseif Ctrue == 3
        mu_true = [mu_all - 2.0, mu_all + 1.0, mu_all + 3.5]
        wi_true = [.3, .4, .3]
        sigmas_true = [1.2, .8, .9]
    end

    #Randomly data generation based on the setting on datagen.jl
    Random.seed!(b * 100)
    m = MixtureModel(map((u, v) -> Normal(u, v), mu_true, sigmas_true), wi_true)
    gamma_true = rand(m, nF)

    prob = exp.(gamma_true[groupindex] .+ X*betas_true)
    prob= prob ./ (1 .+ prob)
    Y = Bool[rand(Binomial(1, prob[i])) == 1 for i in 1:N];
    X = X .- mean(X, dims=1);

    vb=fill(-Inf, ncomponent)
    for kcom in 1:ncomponent
        # res = LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, kcom, maxiteration=1000, tol=0.001)
        res = LatentGaussianMixtureModel.latentgmmrepeat(X, Y, groupindex, kcom, ntrials=25)
        vb[kcom]=res[5]#-kcom*log(nF)*Cn/2.0
    end
    bic,C=findmax(vb)
    println("Mission $b chooses $C components.")
    return vb
end
