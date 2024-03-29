import Distributions, StatsBase, Random
@everywhere using Distributions, StatsBase, Random
@everywhere function datagen(n::Int, Ctrue::Int, N::Int, J::Int, n_ij::Vector{Int64}; b::Int=round(Int,rand()*1000), adjustmean::Bool=false, showpower::Bool=false, Xstd::Real=1.0)

    Random.seed!(200*b)
    groupindex = StatsBase.inverse_rle(1:n, n_ij)
    X = rand(Distributions.Normal(), N, 2)
    betas_true=ones(J)
    if adjustmean
        mu_all = -log(1/0.81 - 1)
        if Ctrue == 1
            mu_true = [mu_all]
            wi_true = [1.0]
            sigmas_true = [0.3493]
        elseif Ctrue == 2
            mu_true = [mu_all - 1.0, mu_all + 1.2]
            wi_true = [.3, .7]
            sigmas_true = [.6, .5]
        elseif Ctrue == 3
            mu_true = [mu_all - 1.0, mu_all, mu_all + 2.]
            wi_true = [.2, .5, .3]
            sigmas_true = [.6, .4, .6]
        end
    elseif showpower
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
    else
        mu_all = log(1/0.779 - 1)
        if Ctrue == 1
            mu_true = [mu_all]
            wi_true = [1.0]
            sigmas_true = [0.5]
        elseif Ctrue == 2
            mu_true = [mu_all - 2.0, mu_all + 2.0]
            wi_true =  [.5, .5]
            sigmas_true = [1.2, .8]
        elseif Ctrue == 3
            mu_true = [mu_all - 4.0, mu_all + 1.0, mu_all + 4.0]
            wi_true = [.3, .4, .3]
            sigmas_true = [1.2, .8, .9]
        end
    end

    #Randomly data generation based on the setting on datagen.jl
    Random.seed!(b * 100)
    m = Distributions.MixtureModel(map((u, v) -> Distributions.Normal(u, v), mu_true, sigmas_true), wi_true)
    gamma_true = rand(m, n)

    prob = exp.(gamma_true[groupindex] .+ X*betas_true)
    prob= prob ./ (1 .+ prob)
    Y = Bool[Distributions.rand(Distributions.Binomial(1, prob[i])) == 1 for i in 1:N];
    X = X .- mean(X, dims=1);
    return(X, Y, groupindex, wi_true, mu_true, sigmas_true, betas_true, m, gamma_true)
end
