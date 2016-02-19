
using Distributions, StatsBase


n = 282
srand(100)
n_ij = round(Int64, rand(Poisson(55), 282).+rand(Exponential(95), 282))
N = sum(n_ij)

groupindex = inverse_rle(1:n, n_ij)
J=2  #beta dimension
srand(200*b)
X = rand(TDist(8), N, 2) .* [14.71 .447] ./ std(TDist(8))
betas_true=[0.0274, 0.0878]
if Ctrue == 1
    mu_true = [-1.54]
    wi_true = [1.0]
    sigmas_true = [0.3493]
elseif Ctrue == 2
    mu_true = [-2.0858,-1.4879]
    wi_true = [0.0828,0.9172]
    sigmas_true = [0.6735,0.2931]
elseif Ctrue == 3
    mu_true = [log(1/0.779 - 1) - 4.0, log(1/0.779 - 1) + 1.0, log(1/0.779 - 1) + 4.0;]
    wi_true = [.3, .4, .3]
    sigmas_true = [1.2, .8, .9]
end

#Randomly data generation based on the setting on datagen.jl
srand(b * 100)
m = MixtureModel(map((u, v) -> Normal(u, v), mu_true, sigmas_true), wi_true)
gamma_true = rand(m, n)

prob = exp(gamma_true[groupindex] .+ X*betas_true)
prob= prob ./ (1 .+ prob)
Y = Bool[rand(Binomial(1, prob[i])) == 1 for i in 1:N];
X = X .- mean(X, 1);
