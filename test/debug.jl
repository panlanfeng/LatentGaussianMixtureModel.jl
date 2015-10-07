import LatentGaussianMixtureModel
using Distributions, StatsBase
using Yeppp
b=1
Ctrue=2

nF = 282
srand(100)
n_ij = round(Int64, rand(Poisson(5), 282).+rand(Exponential(45), 282))
N = sum(n_ij)

facility = inverse_rle(1:nF, n_ij)
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
srand(b * 100)
for i in 1:N
    if rand(Binomial(1, prob[i])) == 1
        Y[i] = true
    else
        Y[i] = false
    end
end
#lr=loglikelihoodratio(X, Y, facility, Calternative, ntrials=25, debuginfo=debuginfo)




ncomponent1=3
C0 = ncomponent1 - 1
C1 = ncomponent1 
nF = maximum(facility)
an1 = 0.0 # 1/nF
gamma_init, beta_init, sigmas_tmp = LatentGaussianMixtureModel.maxposterior(X, Y, facility)
wi_init, mu_init, sigmas_init, ml_tmp = LatentGaussianMixtureModel.gmm(gamma_init, C0, ones(C0)/C0, quantile(gamma_init, linspace(0, 1, C0+2)[2:end-1]), ones(C0), an=an1, maxiter=1)

 srand(100);@time LatentGaussianMixtureModel.latentgmm(X, Y, facility, C0, beta_init, wi_init, mu_init, sigmas_init, Mmax=5000, initial_iteration=10, maxiteration=3, an=an1, sn=std(gamma_init).*ones(C0));



vtau=[.5, .3, .1]
ngh=1000
Mctau=1000
gamma0 = vec(mean(gamma_mat, 2))    
mingamma = minimum(gamma0)
maxgamma = maximum(gamma0)

lr = zeros(length(vtau), C0)
or = sortperm(mu_init)
wi0 = wi_init[or]
mu0 = mu_init[or]
sigmas0 = sigmas_init[or]
betas0 = betas_init
an = 0.0 # decidepenalty(wi0, mu0, sigmas0, nF)

N,J=size(X)
sample_gamma_mat = zeros(nF, Mctau)
sumlogmat = zeros(nF, ngh*ncomponent1)
llvec = zeros(N)
llvecnew = zeros(N)
xb = zeros(N)

whichtosplit=2
ind = [1:whichtosplit, whichtosplit:C0;]
mu_lb = [mingamma, (mu0[1:(C0-1)] .+ mu0[2:C0])./2;]
mu_ub = [(mu0[1:(C0-1)] .+ mu0[2:C0])./2, maxgamma;]
mu_lb = mu_lb[ind]
mu_ub = mu_ub[ind]
sigmas_lb = 0.25 .* sigmas0[ind]
sigmas_ub = 2 .* sigmas0[ind]

tau=.1

ind = [1:whichtosplit, whichtosplit:C0;]
using FastGaussQuadrature
ngh=1000
ghx, ghw = gausshermite(ngh)
ntrials=3
wi_C1 = wi0[ind]
wi_C1[whichtosplit] = wi_C1[whichtosplit]*tau
wi_C1[whichtosplit+1] = wi_C1[whichtosplit+1]*(1-tau)

wi = repmat(wi_C1, 1, 4*ntrials)
mu = zeros(ncomponent1, 4*ntrials)
sigmas = ones(ncomponent1, 4*ntrials)
betas = repmat(betas0, 1, 4*ntrials)
ml = -Inf .* ones(4*ntrials) 

i=4
srand(i)
mu[:, i] = rand(ncomponent1) .* (mu_ub .- mu_lb) .+ mu_lb
sigmas[:, i] = rand(ncomponent1) .* (sigmas_ub .- sigmas_lb) .+ sigmas_lb

srand(100);LatentGaussianMixtureModel.latentgmm_ctau(X, Y, facility, ncomponent1, betas0, wi[:, i], mu[:, i], sigmas[:, i], whichtosplit, tau, ghx, ghw, mu_lb=mu_lb,mu_ub=mu_ub, maxiteration=30, Mmax=50, M_discard=500, an=0., Q_maxiter=2, debuginfo=true)



srand(100)
for i in 1:4*ntrials
    mu[:, i] = rand(ncomponent1) .* (mu_ub .- mu_lb) .+ mu_lb
    sigmas[:, i] = rand(ncomponent1) .* (sigmas_ub .- sigmas_lb) .+ sigmas_lb
    # if ncomponent1 != 2
    #     #fit gmm on gamma_hat with the starting points, to accelerate the latentgmm_ctau
    #     wi[:, i], mu[:, i], sigmas[:, i], tmp = gmm(gamma0, ncomponent1, wi[:, i], mu[:, i], sigmas[:, i], whichtosplit=whichtosplit, tau=tau, mu_lb=mu_lb,mu_ub=mu_ub, maxiter=1, wifixed=true, sn=sn, an=an)
    # end
    wi[:, i], mu[:, i], sigmas[:, i], betas[:, i], ml[i] = LatentGaussianMixtureModel.latentgmm_ctau(X, Y, facility, ncomponent1, betas0, wi[:, i], mu[:, i], sigmas[:, i], whichtosplit, tau, ghx, ghw, mu_lb=mu_lb,mu_ub=mu_ub, maxiteration=8, Mmax=50, M_discard=500, an=0., Q_maxiter=2, debuginfo=false)
end

mlperm = sortperm(ml)
for j in 1:ntrials
    i = mlperm[4*ntrials+1 - j] # start from largest ml 
    wi[:, i], mu[:, i], sigmas[:, i], betas[:, i], ml[i] = LatentGaussianMixtureModel.latentgmm_ctau(X, Y, facility, ncomponent1, betas0, wi[:, i], mu[:, i], sigmas[:, i], whichtosplit, tau, ghx, ghw, mu_lb=mu_lb,mu_ub=mu_ub, maxiteration=50, Mmax=50, M_discard=500, an=0., Q_maxiter=2, debuginfo=true)
end

mlmax, imax = findmax(ml[mlperm[(3*ntrials+1):4*ntrials]])
imax = mlperm[3*ntrials+imax]

re=LatentGaussianMixtureModel.latentgmm(X, Y, facility, ncomponent1, betas[:, imax], wi[:, imax], mu[:, imax], sigmas[:, imax], Mmax=5000, maxiteration=3, initial_iteration=0, an=an)
2(re[5] - ml_C0)
