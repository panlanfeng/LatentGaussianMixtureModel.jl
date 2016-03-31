# LatentGaussianMixtureModel
#By Lanfeng Pan

The Julia package for Generalized Linear Mixed Model with Normal Mixture random effects. It is named as `LatentGaussianMixtureModel` because we are fitting a Gaussian Mixture Model on the random effect which is latent.

To install this package, please run

~~~julia
Pkg.clone("https://github.com/panlanfeng/LatentGaussianMixtureModel.jl.git")
~~~

Currently this package only support single random effect on intercept with logistic link. The easiest way to use is constructing a `LGMModel` object via the following 

~~~julia
using DataFrames
using LatentGaussianMixtureModel
df = readtable("data.csv")
#fit a two components mixture
m = latentgmm(Y~x1+x2+x3+(1|groupindex), df, 2)

#or 
X = readcsv("X.csv"); 
Y=readcsv("Y.csv");
groupindex = readcsv("groupindex.csv");
m = LGMModel(X, Y, groupindex, 2)
~~~

and then fit the model via the function `fit!`

~~~julia
fit!(m);
m.p, m.μ, m.σ, m.β
~~~

To do the restricted likelihood ratio test on the number of components, use the `EMtest` function, for example

~~~julia
EMtest(m)
~~~
This will print out the test statistic and the p value.

See arguments available for constructing the `LGMModel` by running

~~~julia
?LGMModel
~~~
and see arguments for `fit!` by 

~~~julia
?fit!
~~~

The `LGMModel` object is a subtype of `RegressionModel` and the following methods are available:

 - `nobs` returns the number of random effect levels
 - `model_response` returns the response `Y`
 - `coef` returns the fixed effects `β`
 - `ranef!` return the predict random effects
 - `stderr` gives the standard error of fixed effects
 - `confint` calculates the confidence interval
 - `coeftable` prints the fixed effects and their p values
 - `loglikelihood` calculates the log marginal likelihood
 - `vcov` returns the covariance matrix of fixed effects
 -  `asymptoticdistribution` returns the simulated asymptotic distribution of the restricted likelihood ratio test
 - `predict` computes the probability of `Y` being 1 at given new data.


 
