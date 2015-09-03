# LatentGaussianMixtureModel


The Julia package for Generalized Linear Mixed Model with Normal Mixture random effects. 

Most of the core functions are provided in utility.jl. The function `latentgmm` is the major function estimating the parameters. An example call of this function can be 

    latentgmm(X, Y, facility, nF, 3, beta_init, wi_init, mu_init, sigmas_init, Mmax=10000, M_discard=1000, initial_iteration=3, maxiteration=150, tol=0.005, proposingsigma=1.0, ngh=1000)

where `X` and `Y` are the covariates and response. `facility` means denote which facility a patient belongs to, i.e. the random effect levels. `nF` is the number of levels of random effects and 3 is the number of components. `beta_init`, `wi_init`, `mu_init`, `sigmas_init` are the starting values for the fixed effects and parameters of the random effects. `Mmax` means the number of Gibbs samples generated while `M_discard` means how many burn in samples to be dropped. `maxiteration` is the maximum MCEM iterations allowed. `tol` means the stopping criteria. `proposingsigma` is the standard deviation of the proposing distribution in Metropolis Hasting algorithm. `ngh` means the number of points used in gaussian hermite quadrature approximation in calculating the marginal log likelihood.   

See the `examples` folder for more examples. In examples folder,

 - B100tests21.jl generates 100 data sets (with 1 component in the true mixture) and tests m=2 v.s. the null hypothesis m=1 on each of the data set. The 100 statistics are saved to show the distribution of the test statistic. This file needs to include datagen1comp.jl. 
 
 - datagen1comp.jl generates the true random effect with 1 component. 
 - count2.csv stores the number of patients each transplant center has.

To run the simulation,
    
    include("examples/B100tests21.jl")
    pmap(Brun, 1:100)
  
This will generate a csv file contains all the 100 test statistics.
