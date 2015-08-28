#find out the test statistic when m0=1


@everywhere using LatentGaussianMixtureModel
@everywhere using Distributions
if VERSION >=v"0.4.0-dev"
    @everywhere import datagen3
else
    require(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "examples/datagen3comp.jl"))
end

#Brun calculate the statistic for one data set;
#b is the the random number seed, from 1 to 100
@everywhere function Brun(b::Integer)
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

    #Initialized one component model using max posterior model
    gamma_init, beta_init, sigmas_init = maxposterior(X, Y, facility)
    wi_init, mu_init, sigmas_init, ml1 = gmm(gamma_init, 2, ones(2)/2, percentile(gamma_init, [25,75]), ones(2))

    #two component model
    #M=20000
    re = latentgmm(X, Y, facility, nF, 2, beta_init, wi_init, mu_init, sigmas_init, Mmax=10000,initial_iteration=10)
    # m = Normal( mu_true,  sigmas_true)
    #m2 = MixtureModel(Normal[Normal(re2[4][1], re2[5][1]), Normal(re2[4][2], re2[5][2])], re2[3])

    #mll2 = marginallikelihood(re2[2], X, Y, facility,nF, m2, M)
    gamma_hat = vec(mean(re[6], 2))
    lr = loglikelihoodratio(X, Y, facility, nF, 3, re[4], re[1], re[2], re[3], gamma_hat, minimum(gamma_hat), maximum(gamma_hat), ml_base=re[5])
    maximum(lr)

end

#run on all available cores using:
#
#teststat= pmap(Brun, 1:100)
#writecsv("teststat32C3.csv", teststat)
