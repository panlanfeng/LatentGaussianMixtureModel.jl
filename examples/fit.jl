import LatentGaussianMixtureModel
using RCall

b=4
Ctrue=2
include("datagen.jl")

C0 = Ctrue
an1 = 1/n

srand(b);wi_init, mu_init, sigmas_init, betas_init, ml_tmp = LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, 1, [1., 1.], [1.0], [0.], [1.], maxiteration=200, debuginfo=false, Qmaxiteration=6, an=an1)

gamma_init = LatentGaussianMixtureModel.predictgamma(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init);

wi_init, mu_init, sigmas_init, ml_tmp = LatentGaussianMixtureModel.gmm(gamma_init, C0, an=an1)

@time wi, mu, sigmas, betas, ml_C0 = LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, C0, betas_init, wi_init, mu_init, sigmas_init, maxiteration=2000, an=an1, debuginfo=false, sn=std(gamma_init).*ones(C0), tol=.001, Qmaxiteration=6, pl=false, ptau=false)
mhat = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)
println(mhat)
println("ml_C0=",ml_C0)

# trand=LatentGaussianMixtureModel.asymptoticdistribution(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init);

xs = linspace(-8, 7, 400);
dentrue = pdf(m, xs);
denhat = pdf(mhat, xs);

@rput xs dentrue denhat gamma_init gamma_true;

rprint(""" 
plot(xs, dentrue, lwd=3, type="l")
lines(density(gamma_true), lty=2)
rug(gamma_true)
rug(gamma_init, ticksize=-0.03, col="red")
lines(xs, denhat, col="blue")
""")
