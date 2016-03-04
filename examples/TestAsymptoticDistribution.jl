#find out the test statistic when m0=1
import LatentGaussianMixtureModel
import Distributions, StatsBase
import Yeppp

@everywhere using LatentGaussianMixtureModel
@everywhere using Distributions, Yeppp, StatsBase


#Brun calculate the statistic for one data set;
#b is the the random number seed, from 1 to 100
@everywhere function Brun(b::Integer, Ctrue::Integer, Calternative::Integer; debuginfo::Bool=false, ntrials::Int=25)
    nF = 282
    srand(100)
    n_ij = round(Int64, rand(Poisson(55), 282).+rand(Exponential(95), 282))
    N = sum(n_ij)

    groupindex = inverse_rle(1:nF, n_ij)
    J=2  #42 #beta dimension
    srand(200*b)
    X = rand(TDist(8), N, 2) .* [14.71 4.47] ./ std(TDist(8))
    betas_true=[0.0274, 0.00878]
    if Ctrue == 1
        mu_true = [log(1/0.779-1)]
        wi_true = [1.0]
        sigmas_true = [0.5]
    elseif Ctrue == 2
        mu_all = log(1/0.779-1)
        mu_true = [mu_all - 2.0, mu_all + 2.0;] 
        wi_true = [0.5, 0.5;]
        sigmas_true = [1.2, 0.8;]
    elseif Ctrue == 3
        mu_true = [log(1/0.779 - 1) - 4.0, log(1/0.779 - 1) + 1.0, log(1/0.779 - 1) + 4.0;]
        wi_true = [.3, .4, .3]
        sigmas_true = [1.2, .8, .9]
    end
    if Ctrue == Calternative
        mu_all = log(1/0.779 - 1)
        if Ctrue == 2
            mu_true = [mu_all - 0.5, mu_all + 0.5] 
            wi_true =  [.6, .4]
            sigmas_true = [1.2, .8]
        elseif Ctrue == 3
            mu_true = [mu_all - 1.0, mu_all + 1.0, mu_all + 3.0;]
            wi_true = [.3, .4, .3]
            sigmas_true = [.7, .8, .9]
        end
    end
    
    #Randomly data generation based on the setting on datagen.jl
    srand(b * 100)
    m = MixtureModel(map((u, v) -> Normal(u, v), mu_true, sigmas_true), wi_true)
    gamma_true = rand(m, nF)
    
    prob = exp(gamma_true[groupindex] .+ X*betas_true)
    prob= prob ./ (1 .+ prob)
    Y = Bool[rand(Binomial(1, prob[i])) == 1 for i in 1:N];
    X = X .- mean(X, 1);

    lr=LatentGaussianMixtureModel.EMtest(X, Y, groupindex, Calternative-1, ntrials=ntrials, debuginfo=debuginfo, ctauparallel=false, ngh=100)
    println("Mission $b completed with lr=$lr")
    lr
end
