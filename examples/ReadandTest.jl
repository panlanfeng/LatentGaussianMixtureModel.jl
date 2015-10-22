#Read in the file and run latentgmm
#Lanfeng Pan, Oct 21, 2015

#run the following after successfully installed Julia. You only need to do the following once.
# Pkg.init()
# Pkg.clone("where/you/put/thepackage/LatentGaussianMixtureModel")
# Pkg.checkout("FastGaussQuadrature")

# To run on the real data, change to the directory where the data is stored
#cd("C:/latentgmmdata/")
#for example go to the simulated data folder
#cd(joinpath(Pkg.dir("LatentGaussianMixtureModel"), "examples"))

#adding all available cpu cores, utilizing the parallel computing
addprocs(CPU_CORES-1)

#LatentGaussianMixtureModel is our package
import LatentGaussianMixtureModel
import Distributions, StatsBase, Yeppp

@everywhere using LatentGaussianMixtureModel
@everywhere using Distributions, Yeppp, StatsBase

#Read in the patients covariates X
X = readcsv("X.csv")
N, J = size(X)

#read in the groupindex for transplant center and convert to integer vector 
groupindex_raw = readcsv("groupindex.csv")
levelsdictionary = levelsmap(groupindex_raw)
groupindex  = ones(Int64, N)
for i in 1:N
    groupindex[i] = levelsmap(groupindex_raw)[groupindex_raw[i]]
end
#after the convert, maximum value in groupindex should be the number of transplant centers
nF = maximum(groupindex)

# read in the binary response Y and convert to Bool Vector
#Not sure what Y looks like, so I list several possibilities here.
Y_raw = readcsv("Y.csv")
Y = Array(Bool, N)
for i in 1:N
    if Y_raw[i] in ["Yes","YES","y", "Y", 1, "1", 1.0, "1.0", "true", "TRUE", "True"]
        Y[i] = true
    else
        Y[i] = false
    end
end

#Test if X, Y, groupindex are in the same length
@assert length(Y) == length(groupindex) == N

#do a single model fit
C0=2 #number of components to try
#initialize
gamma_init, beta_init, sigmas_tmp = maxposterior(X, Y, groupindex)
wi_init, mu_init, sigmas_init, ml_tmp = gmm(gamma_init, C0, ones(C0)/C0, quantile(gamma_init, linspace(0, 1, C0+2)[2:end-1]), ones(C0), an=1/nF)
#Model fit
#If there is error, set debuginfo=true to see more information
wi, mu, sigmas, betas, ml_C0, gamma_mat = latentgmm(X, Y, groupindex, C0, beta_init, wi_init, mu_init, sigmas_init, Mmax=5000,  maxiteration=100, an=1/nF, sn=std(gamma_init).*ones(C0), debuginfo=true)
#print the output
println("The returned parameters are:")
println(wi, mu, sigmas, betas, ml_C0)

#set the number of components We want to test
C0=1 # Null hypothesis
C1=2 # Alternative hypothesis


#Return  2*loglikelihoodratio and the p value
#The recommended ntrials=25. Setting it to some smaller number can save much time if we just want to test if the code is working.
#If there is error, set debuginfo=true to see more information
lr=loglikelihoodratio(X, Y, groupindex, C1, ntrials=25, debuginfo=false, reportpvalue=true)
println("The test statistica and p value are:")
println(lr)
