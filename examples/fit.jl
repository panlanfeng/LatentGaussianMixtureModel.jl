
import LatentGaussianMixtureModel
using Distributions, StatsBase
using Yeppp
import PyPlot; const plt = PyPlot

b=7272
Ctrue=2

nF = 282
srand(100)
n_ij = round(Int64, rand(Poisson(25), nF).+rand(Exponential(95), nF));
N = sum(n_ij)

groupindex = inverse_rle(1:nF, n_ij);
J=2  #42 #beta dimension
srand(200*b)
X = rand(TDist(8), N, 2) .* [14.71 4.47] ./ std(TDist(8));
betas_true=[0.0274, 0.00878]
if Ctrue == 1
    mu_true = [log(1/0.779 - 1)]
    wi_true = [1.0]
    sigmas_true = [1.2]
elseif Ctrue == 2
    mu_true = [-2.0858,-1.4879]
    wi_true = [0.0828,0.9172]
    sigmas_true = [0.6735,0.2931]
elseif Ctrue == 3
    mu_true = [log(1/0.779 - 1)/3 - 4.0, log(1/0.779 - 1)/3 + 1.0, log(1/0.779 - 1)/3 + 4.0;]
    wi_true = [.3, .4, .3]
    sigmas_true = [1.2, .8, .9]
end

#Randomly data generation based on the setting on datagen.jl
srand(b * 100)
m = MixtureModel(map((u, v) -> Normal(u, v), mu_true, sigmas_true), wi_true)
#gamma_true = rand(m, nF);
gamma_true = [rand(m.components[1], 24), rand(m.components[2], 258);];
#gamma_true=readcsv("/Users/lanfengpan/fa/transplant/gammaprediction.csv")[:,1]

prob = exp(gamma_true[groupindex] .+ X*betas_true);
prob= prob ./ (1 .+ prob);
Y = Bool[rand(Binomial(1, prob[i])) == 1 for i in 1:N];

#trand=LatentGaussianMixtureModel.asymptoticdistribution(X, Y, groupindex, wi_true, mu_true, sigmas_true, betas_true);

C0=Ctrue
nF = maximum(groupindex)
an1 = 1/nF

srand(b);wi_init, mu_init, sigmas_init, betas_init, ml_tmp = LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, 1, [1., 1.], [1.0], [0.], [1.], maxiteration=200, debuginfo=false, Qmaxiteration=6, an=an1)

 gamma_init = LatentGaussianMixtureModel.predictgamma(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init);

wi_init, mu_init, sigmas_init, ml_tmp = LatentGaussianMixtureModel.gmm(gamma_init, C0, an=an1)

@time wi, mu, sigmas, betas, ml_C0 = LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, C0, betas_init, wi_init, mu_init, sigmas_init, maxiteration=2000, an=an1, debuginfo=false, sn=std(gamma_init).*ones(C0), tol=.001, Qmaxiteration=6, pl=false,ptau=false)

 trand=LatentGaussianMixtureModel.asymptoticdistribution(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init);

#LatentGaussianMixtureModel.EMtest(X, Y, groupindex, Ctrue, ntrials=2, debuginfo=true, ctauparallel=true, ngh=100)

mhat = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)
println(mhat)
println("ml_C0=",ml_C0)
xs = linspace(-4, 2, 400)
dentrue = pdf(m, xs)
denhat = pdf(mhat, xs)
#gammaprediction = readcsv("/Users/lanfengpan/fa/transplant/gammaprediction.csv")
plt.plot(xs, dentrue, "k-", xs, denhat, "b--")
nothing
