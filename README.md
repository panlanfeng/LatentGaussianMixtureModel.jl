# LatentGaussianMixtureModel
#By Lanfeng Pan

The Julia package for Generalized Linear Mixed Model with Normal Mixture random effects. 

Most of the core functions are provided in utility.jl. The function `latentgmm` is the major function estimating the parameters. An example call of this function can be 

    latentgmm(X, Y, groupindex, 3, beta_init, wi_init, mu_init, sigmas_init, Mmax=10000, M_discard=1000, initial_iteration=3, maxiteration=150, tol=0.005, proposingsigma=1.0, ngh=1000)

All available parameters are
 
- `X` and `Y` are the covariates and response. 
- `groupindex` means denote which transplant center a patient belongs to, i.e. the random effect levels. 
- ncomponent is the number of components to try
- `beta_init`, `wi_init`, `mu_init`, `sigmas_init` are the starting values for the fixed effects and parameters of the random effects. 
- `Mmax` means the number of Gibbs samples generated while `M_discard` means how many burn in samples to be dropped. 
- `maxiteration` is the maximum MCEM iterations allowed. `tol` means the stopping criteria. `initial_iteration` is the number of initial iterations using smaller `Mmax`. Initial iterations are used to save time in the first few iterations when the parameters are still far from the truth.
-  `proposingsigma` is the standard deviation of the proposing distribution in Metropolis Hasting algorithm. 
- `ngh` means the number of points used in Gaussian-Hermite quadrature approximation in calculating the marginal log likelihood.    
- `sn` is the standard deviation to used in penalty function
- `an` is the penalty factor
- `debuginfo` is the switch to print more information to help debug.
- `restartMCMCsampling` should MCMC sampling use fresh generated random values or use those from last EM iteration.

Another important function is `loglikelihoodratio` which calculates the 2*log likelihood ratio between m=m0+1 and m=m0.

A sample call is 

    loglikelihoodratio(X, Y, groupindex, C1, ntrials=2, debuginfo=false, reportpvalue=true)

All availabe parameters are 

- `X`, `Y`, `groupindex`, `ncomponent1`, `ngh`, `debuginfo`, `restartMCMCsampling` are the same as in `latentgmm`
- `vtau` is the weight proportions to try, default to be 0.5, 0.3 and 0.1.
- `ntrials` is the number of random initial values to try in `latentgmm_ctau`. 25 is the recommended. If is is too small, the test power is smaller than it can be.
- `Mctau` is the number of MCMC samples used in `latentgmm_ctau`
- `reportpvalue` tell it to generate the asymptotic distribution and report the p values. 

See the `examples` folder for examples. In examples folder, the file "ReadandTest.jl" reads the data, convert the data into the right format and run `latentgmm` and `loglikelihoodratio`
