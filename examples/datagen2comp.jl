module datagen2comp
export X, Y, facility, beta_true, mu_true, wi_true, sigmas_true, nF, n_ij, J

using Distributions
using StatsBase
using LatentGaussianMixtureModel

nF = 282
n_ij = readcsv(joinpath(Pkg.dir("LatentGaussianMixtureModel"),"examples/count2.csv"), Int64, header=true)[1][:,1]
for i in 1:length(n_ij)
    if n_ij[i] > 20
        n_ij[i] = round(n_ij[i] / 10.0, 0)
    end
end
N = sum(n_ij)

facility = eachrepeat(1:nF, n_ij)

#true values, Y=1 means survive
J=2  #42 #beta dimension


#grand mean = log(1/0.779 - 1)
#Set the overall mean of the mixture to be the grand mean.
srand(100)
X = rand(Normal(), (N, J))
beta_true=ones(J) #rand(Normal(0,1), J)
mu_true = [log(1/0.779 - 1) - 1.0, log(1/0.779 - 1) + 1.0]
wi_true = [.5, .5]
sigmas_true = [1.2, .8]
end
