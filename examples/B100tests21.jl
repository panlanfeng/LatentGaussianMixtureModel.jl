#find out the test statistic when m0=1
import LatentGaussianMixtureModel
import Distributions
import Yeppp
import datagen3comp
@everywhere using LatentGaussianMixtureModel
@everywhere using Distributions, Yeppp
@everywhere using datagen1comp

#Brun calculate the statistic for one data set;
#b is the the random number seed, from 1 to 100
@everywhere function Brun(b::Integer)
    #Randomly data generation based on the setting on datagen.jl
    srand(b * 100)

    gamma_true = rand(Normal(mu_true, sigmas_true), nF)
    prob = exp(gamma_true[facility] .+ X*beta_true)
    prob= prob ./ (1 .+ prob)
    Y = Array(Bool, N)
    for i in 1:N
        if rand(Binomial(1, prob[i])) == 1
            Y[i] = true
        else
            Y[i] = false
        end
    end

    lr = loglikelihoodratio(X, Y, facility, nF, 2)
    maximum(lr)
end

#run on all available cores using:
#
teststat= pmap(Brun, 1:100)
writecsv("teststat21.csv", teststat)
