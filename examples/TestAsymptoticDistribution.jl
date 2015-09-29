#find out the test statistic when m0=1
import LatentGaussianMixtureModel
import Distributions
import Yeppp

@everywhere using LatentGaussianMixtureModel
@everywhere using Distributions, Yeppp


#Brun calculate the statistic for one data set;
#b is the the random number seed, from 1 to 100
@everywhere function Brun(b::Integer, Ctrue::Integer, Calternative::Integer; debuginfo=false)
    nF = 282
    srand(100)
    n_ij = round(Int64, rand(Poisson(5), 282).+rand(Exponential(45), 282))
    N = sum(n_ij)

    facility = eachrepeat(1:nF, n_ij)
    J=2  #42 #beta dimension
    srand(100)
    X = rand(Normal(), (N, J))
    beta_true=ones(J) #rand(Normal(0,1), J)
    if Ctrue == 1
        mu_true = [log(1/0.779 - 1)]
        wi_true = [1.0]
        sigmas_true = [1.2]
    elseif Ctrue == 2
        mu_true = [log(1/0.779 - 1)/2 - 2.0, log(1/0.779 - 1)/2 + 2.0]
        wi_true = [.5, .5]
        sigmas_true = [1.2, .8]
    elseif Ctrue == 3
        mu_true = [log(1/0.779 - 1)/3 - 4.0, log(1/0.779 - 1)/3 + 1.0, log(1/0.779 - 1)/3 + 4.0;]
        wi_true = [.3, .4, .3]
        sigmas_true = [1.2, .8, .9]
    end
    
    #Randomly data generation based on the setting on datagen.jl
    srand(b * 100)
    m = MixtureModel(map((u, v) -> Normal(u, v), mu_true, sigmas_true), wi_true)
    gamma_true = rand(m, nF)
    
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

    lr=loglikelihoodratio(X, Y, facility, Calternative, ntrials=25, debuginfo=debuginfo)
    println("Mission $b completed with lr=$lr")
    lr
end
