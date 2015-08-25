# LatentGaussianMixtureModel


The Julia code for Generalized Linear Mixed Model with Normal Mixture random effects. 

Most of the core functions are provided in utility.jl. The function `latentgmm` is the major function estimating the parameters. 

B100tests21.jl generates 100 data sets (with 1 component in the true mixture) and tests the null hypothesis m0=1 on each of the data set. The 100 statistics are saved to show the distribution of the test statistic. This file needs to include datagen1comp.jl. 
