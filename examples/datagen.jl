# the data generation settings
#Lanfeng Pan
#Oct 29, 2014

using Distributions
using StatsBase


nF = 282
n_ij = readcsv("count2.csv", Int64, header=true)[1][:,1]
N = sum(n_ij)

facility = eachrepeat(1:nF, n_ij)

#true values, Y=1 means survive
J=2  #42 #beta dimension


#grand mean = log(1/0.779 - 1)
#Set the overall mean of the mixture to be the grand mean.
beta_true=rand(Normal(0,1), J)
mu_true = [-4, 0.0, 4] .+ log(1/0.779 - 1)
wi_true = [.25, .5, .25]
sigmas_true = [1.2, 1.0, 1.3]

#########End of Data Generation
