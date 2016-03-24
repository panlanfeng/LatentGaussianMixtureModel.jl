#Read in the file and run latentgmm
#Lanfeng Pan, Oct 21, 2015

#change the path to the new folder if there is new version of data
datapath = "C:\\Users\\liyanmin\\Desktop\\LatentGaussianMixtureModel\\LatentGaussianMixtureModel_20160323_2251_v022\\LatentGaussianMixtureModel_20160323_2251_v022\\data"

#adding all available cpu cores, utilizing the parallel computing
addprocs(2)

#These lines loads all the functions
Pkg.update()
import LatentGaussianMixtureModel
println("The loaded package version is", LatentGaussianMixtureModel.PKGVERSION)

#LatentGaussianMixtureModel is our package
#import LatentGaussianMixtureModel
import Distributions, StatsBase, GaussianMixtureTest
@everywhere using Distributions, StatsBase, GaussianMixtureTest

#Read in the patients covariates X

X = readcsv(joinpath(datapath, "X.csv"));
groupindex_raw = readcsv(joinpath(datapath, "groupindex.csv"));
Y_raw = readcsv(joinpath(datapath, "Y.csv"));

N, J = size(X)

#read in the groupindex for transplant center and convert to integer vector 
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
wi, mu, sigmas, betas, ml_C = LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, C; maxiteration=1000, an=1/n, debuginfo=false, tol=.001)


###------------
##Only if we need to try multiple initial values
wi, mu, sigmas, betas, ml_C = LatentGaussianMixtureModel.latentgmmrepeat(X, Y, groupindex, C, ntrials=5)
###-----------------

## If you want to specify your own initial values
#[0.7, .3] is the weight pi
#[-1.0, -0.1] is the means mu
#[0.4, 0.3] is the standard deviation sigma
wi, mu, sigmas, betas, ml_C = LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, 2, ones(J), [0.7, 0.3], [-1.0, -0.1], [0.4, 0.3]; maxiteration=1000, an=1/n, debuginfo=false, tol=.001)



# Print the predicted gamma
gammaprediction = LatentGaussianMixtureModel.predictgamma(X, Y, groupindex, wi, mu, sigmas, betas);
#writecsv("gammaprediction.csv", gammaprediction)
p=LatentGaussianMixtureModel.coefpvalue(X, Y, groupindex, wi, mu, sigmas, betas)


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
save(joinpath(datapath, "saveall.jld"), "wi", wi, "mu", mu, "sigmas", sigmas, "betas", betas, "X", X, "Y", Y ,"groupindex", groupindex, "lr1", lr1, "lr2", lr2, "gammaprediction", gammaprediction, "clFDR", clFDR, "rejectid", rejectid, "PKGVERSION", LatentGaussianMixtureModel.PKGVERSION)


##Warning! Please load all the packages first before laod the jld file.
#Or some of the data may not be able to recover.
using JLD
import Distributions, StatsBase
@everywhere using Distributions, StatsBase
@load joinpath(datapath, "saveall.jld")

############----------------------
## Plot a graph
#Pkg.add("KernelEstimator")
using RCall, KernelEstimator
mhat = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)
xs = linspace(minimum(gammaprediction)-0.5, maximum(gammaprediction)+1., 400);
denhat = pdf(mhat, xs);
denpredict = kerneldensity(gammaprediction, xeval=xs)
den1 = probs(mhat)[1].*pdf(mhat.components[1], xs);
den2 =  probs(mhat)[2] .* pdf(mhat.components[2], xs);

##Send the data to R
@rput gammaprediction xs denhat den1 den2 denpredict

# regular R code inside the triple quotes.
rprint("""
#pdf("denhat_mpm.pdf", width=10, height=6.18)
plot(xs, denhat, lwd=4, type="l", xlab=expression(gamma), ylab="")
lines(xs, den1, lwd=3, lty=2, col="blue")
lines(xs, den2, lwd=3, lty=2, col="red")
#rug(gammaprediction)
#lines(xs, denpredict, lty=4, lwd=3, col="blue")
legend("topright",c("Estimated Density", "Component 1", "Component 2"), lty=c(1,2,2), lwd=c(4, 3, 3), col=c("black", "blue", "red"))
#dev.off()
NULL
""")
