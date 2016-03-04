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
import Distributions, StatsBase, GaussianMixtureTest

#@everywhere using LatentGaussianMixtureModel
@everywhere using Distributions, StatsBase, GaussianMixtureTest

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
lr1=LatentGaussianMixtureModel.EMtest(X, Y, groupindex, 1, ntrials=5, vtau=[.5, .3, .1;], debuginfo=true)

#If the reject C=1, further test C=2
lr2=LatentGaussianMixtureModel.EMtest(X, Y, groupindex, 2, ntrials=5, debuginfo=true)



####-------------------------------------------------
## **Part Two: Model Fitting**



#After decide the number of components, try model fitting
C=2

## Fitting
wi, mu, sigmas, betas, ml_C = LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, C; maxiteration=1000, an=1/n, debuginfo=false, tol=.001, bn=3.0)


###------------
##Only if we need to try multiple initial values
wi0, mu0, sigmas0, betas0, ml_C0 = LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, 1; maxiteration=1000, an=1/n, debuginfo=false, tol=.001, bn=3.0)
gammaprediction = LatentGaussianMixtureModel.predictgamma(X, Y, groupindex, wi0, mu0, sigmas0, betas0);
wi0, mu0, sigmas0, ml_tmp = gmm(gammaprediction, C)
mingamma = minimum(gammaprediction)
maxgamma = maximum(gammaprediction)
wi, mu, sigmas, betas, ml_C = LatentGaussianMixtureModel.latentgmmrepeat(X, Y,
   groupindex, C, betas0, wi0,
   ones(C).*mingamma, ones(C).*maxgamma, 
   0.25 .* sigmas0, 2.*sigmas0,
   ntrials=5, an=1/n, bn=3.0, sn=std(gammaprediction).*ones(C))
###-----------------


# Print the predicted gamma
gammaprediction = LatentGaussianMixtureModel.predictgamma(X, Y, groupindex, wi, mu, sigmas, betas);
#writecsv("gammaprediction.csv", gammaprediction)
LatentGaussianMixtureModel.confidenceinterval(X, Y, groupindex, wi, mu, sigmas, betas, confidencelevel=0.90)


####-------------------------------------------------
## Part Three: False Dicovery Rate
# 
CNull = findmax(wi)[2]
clFDR, rejectid = LatentGaussianMixtureModel.FDR(X, Y, groupindex, wi, mu, sigmas, betas, [CNull;], alphalevel=0.05)
println("The rejected transplant centers are:", rejectid)
println("Their probabiity of belong to majority is:", round(clFDR[rejectid], 4))

####--------------------------------------
## How to save the current work

using JLD
save("saveall.jld", "wi", wi, "mu", mu, "sigmas", sigmas, "betas", betas, "X", X, "Y", Y ,"groupindex", groupindex, "lr1", lr1, "lr2", lr2, "gammaprediction", gammaprediction, "clFDR", clFDR, "rejectid", rejectid)


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
#Pkg.add("KernelEstimator")
using RCall, KernelEstimator
mhat = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)
xs = linspace(minimum(gammaprediction)-0.5, maximum(gammaprediction)+0.5, 400);
denhat = pdf(mhat, xs);
denpredict = kerneldensity(gammaprediction, xeval=xs)
den1 = probs(mhat)[1].*pdf(mhat.components[1], xs);
den2 =  probs(mhat)[2] .* pdf(mhat.components[2], xs);

##Send the data to R
@rput gammaprediction xs denhat den1 den2 denpredict

# regular R code inside the triple quotes.
rprint("""
pdf("denhat_mpm.pdf", width=10, height=6)
plot(xs, denhat, lwd=4, type="l", ylim=c(0, .2+max(c(denpredict, denhat))), xlab=expression(gamma), ylab="")
lines(xs, den1, lwd=2.5, lty=3, col=c(256,256,256,.8))
lines(xs, den2, lwd=2.5, lty=3, col=c(256,256,256,.8))
rug(gammaprediction)
lines(xs, denpredict, lty=4, lwd=3, col="blue")
legend("topright",c("Estimated Density", "Kernel Density of Predictors", "Individual Components"), lty=c(1,4, 3), lwd=c(3, 2, 1.5))
dev.off()
NULL
""")
