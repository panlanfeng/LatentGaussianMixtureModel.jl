addprocs(2)

import LatentGaussianMixtureModel
import Distributions, StatsBase, GaussianMixtureTest
@everywhere using Distributions, StatsBase,GaussianMixtureTest
using JLD
using DataFrames

#@load "saveall.jld"
datapath = Pkg.dir("LatentGaussianMixtureModel", "examples")


X = readcsv(joinpath(datapath, "X.csv"));
groupindex_raw = readcsv(joinpath(datapath, "groupindex.csv"));
Y_raw = readcsv(joinpath(datapath, "Y.csv"));

N, J = size(X)

#read in the groupindex for transplant center and convert to integer vector 
levelsdictionary = levelsmap(groupindex_raw);
groupindex  = ones(UInt32, N);
for i in 1:N
    groupindex[i] = levelsdictionary[groupindex_raw[i]]
end
#groupindex = convert(Vector{UInt32}, groupindex);
#after the convert, maximum value in groupindex should be the number of transplant centers


# read in the binary response Y and convert to Bool Vector
#Not sure what Y looks like, so I list several possibilities here.

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


N, J = size(X)
n = maximum(groupindex)
C=2


m1 = LatentGaussianMixtureModel.LGMModel(X,Y,groupindex, 1, ngh=100);
m2 = LatentGaussianMixtureModel.LGMModel(X,Y,groupindex, 2, ngh=100);

# lr1=LatentGaussianMixtureModel.EMtest(m1, 5, [.5, .3, .1;], debuginfo=true)
#lr2=LatentGaussianMixtureModel.EMtest(m2, 5, debuginfo=true)


## Fitting
#m.p=copy(wi);m.μ=copy(mu); m.σ = copy(sigmas);m.β = copy(betas); m.fit=true;
LatentGaussianMixtureModel.fit!(m2; maxiteration=1000, debuginfo=false, tol=.001)

LatentGaussianMixtureModel.multiplefit!(m2, 5; maxiteration=1000, debuginfo=false, tol=.001)

LatentGaussianMixtureModel.ranef!(m2)

CNull = findmax(m2.p)[2]
LatentGaussianMixtureModel.detect(m2, [CNull;], alphalevel=0.01)




LatentGaussianMixtureModel.ranef!(m2);
gammaprediction = m2.gammaprediction;
using RCall, KernelEstimator
mhat = LatentGaussianMixtureModel.ranefmixture(m2);
xs = linspace(minimum(m2.gammaprediction)-0.2, maximum(m2.gammaprediction)+0.2, 400);
xs2 = linspace(m2.μ[2] - 3*m2.σ[2], m2.μ[2] + 3*m2.σ[2], 400)
denhat = pdf(mhat, xs);
denpredict = kerneldensity(m2.gammaprediction, xeval=xs);
den1 = probs(mhat)[1].*pdf(mhat.components[1], xs);
den2 =  probs(mhat)[2] .* pdf(mhat.components[2], xs2);

##Send the data to R
@rput gammaprediction xs xs2 denhat den1 den2 denpredict;

# regular R code inside the triple quotes.

rprint("""
        library(ggplot2)
        theme_set(theme_bw(base_size=12))
        cols = c( "Component2"="red", "Component1"="blue")
        ltype = c("Component2"=1, "Component1"=2)
        
        p = ggplot()+ xlab(expression(gamma))+ ylab("") +
        geom_line(aes(x=xs, y=den1, color="Component1", linetype="Component1"), size=0.5) +
        geom_line(aes(x=xs2, y=den2, color="Component2", linetype="Component2"), size=.5)+  scale_color_manual(values=cols, name="linetype")+
        scale_linetype_manual(values=ltype, name="linetype") +
         geom_rug(aes(x=gammaprediction), size=0.2) +
         theme(legend.position="top", legend.text=element_text(size=12),
             legend.title=element_blank(),
             legend.key.width=unit(1, "cm"))
        print(p)
        NULL
        """)
    
