#Read in the file and run latentgmm
#Lanfeng Pan, Oct 21, 2015

#run the following after successfully installed Julia. You only need to do the following once.
# Pkg.init()
# Pkg.clone("where/you/put/thepackage/LatentGaussianMixtureModel")

# To run on the real data, change to the directory where the data is stored
#cd("C:/latentgmmdata/")
#for example go to the simulated data folder
#cd(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "examples"))

#adding all available cpu cores, utilizing the parallel computing
addprocs(2)

#LatentGaussianMixtureModel is our package
import LatentGaussianMixtureModel
import Distributions, StatsBase

@everywhere using LatentGaussianMixtureModel
@everywhere using Distributions, StatsBase

#Read in the patients covariates X
X = readcsv("X.csv");
N, J = size(X)

#read in the groupindex for transplant center and convert to integer vector 
groupindex_raw = readcsv("groupindex.csv");
levelsdictionary = levelsmap(groupindex_raw);
groupindex  = ones(Int64, N);
for i in 1:N
    groupindex[i] = levelsdictionary[groupindex_raw[i]]
end
#after the convert, maximum value in groupindex should be the number of transplant centers
n = maximum(groupindex);

# read in the binary response Y and convert to Bool Vector
#Not sure what Y looks like, so I list several possibilities here.

# *Note*
# Y true means survived, false means dead
Y_raw = readcsv("Y.csv");
Y = Array(Bool, N);
for i in 1:N
    if Y_raw[i] in ["Yes","YES","y", "Y", 1, "1", 1.0, "1.0", "true", "TRUE", "True"]
        Y[i] = true
    else
        Y[i] = false
    end
end

#Test if X, Y, groupindex are in the same length
@assert length(Y) == length(groupindex) == N

X = X .- mean(X, 1);


####-----------------------------
## ** Part One: Hypothesis Test


#The recommended ntrials=25. Setting it to some smaller number can save much time if we just want to test if the code is working.
#If there is error, set debuginfo=true to see more information

#set the number of components We want to test
# Null hypothesis: C = 1
lr1=EMtest(X, Y, groupindex, 1, ntrials=25, debuginfo=false)

#If the reject C=1, further test C=2
lr2=EMtest(X, Y, groupindex, 2, ntrials=25, debuginfo=true)



####-------------------------------------------------
## **Part Two: Model Fitting**



#After decide the number of components, try model fitting
C=2
#initialize
wi_init, mu_init, sigmas_init, betas_init, ml_tmp = latentgmm(X, Y, groupindex, 1, [1., 1.], [1.0], [0.], [1.], maxiteration=100);
gamma_init = predictgamma(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init); 
wi_init, mu_init, sigmas_init, ml_tmp = LatentGaussianMixtureModel.gmm(gamma_init, C)

## Fitting
wi, mu, sigmas, betas, ml_C = latentgmm(X, Y, groupindex, C, betas_init, wi_init, mu_init, sigmas_init, maxiteration=1000, an=1/n, debuginfo=false, tol=.001)

# Print the predicted gamma
gammaprediction = predictgamma(X, Y, groupindex, wi, mu, sigmas, betas);
#writecsv("gammaprediction.csv", gammaprediction)



####-------------------------------------------------
## Part Three: False Dicovery Rate
# 
clFDR, rejectid = LatentGaussianMixtureModel.FDR(X, Y, groupindex, wi, mu, sigmas, betas, [1;], alphalevel=0.05)
println("The rejected transplant centers are:", rejectid)
println("Their probabiity of belong to majority is:", round(clFDR[rejectid], 4))

####--------------------------------------
## How to save the current work

using JLD
@save "saveall.jld"

##To load do
##Warning! Please load all the packages first before laod the jld file.
#Or some of the data may not be able to recover.
using JLD
import LatentGaussianMixtureModel
import Distributions, StatsBase

@everywhere using LatentGaussianMixtureModel
@everywhere using Distributions, StatsBase
@load "saveall.jld"

############----------------------
## Plot a graph

using RCall
mhat = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)
xs = linspace(-8, 7, 400);
denhat = pdf(mhat, xs);

##Send the data to R
@rput gammaprediction xs denhat

# regular R code inside the triple quotes.
rprint("""
#pdf("denhat_mpm.pdf", width=10, heigh=6.18)
plot(xs, denhat, lwd=3, type="l")
lines(density(gammaprediction), lty=2, lwd=2, col="blue")
#dev.off()
""")
