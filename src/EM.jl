function integralweight!(Wim::Matrix{Float64}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,  gammaM::Vector{Float64}, wi::Vector{Float64}, ghw::Vector{Float64}, llN::Vector{Float64}, xb::Vector{Float64},  N::Int, J::Int, n::Int, C::Int, ngh::Int)
    #A_mul_B!(xb, X, betas)

    for jcom in 1:C
        for ix in 1:ngh
            wtmp = log(ghw[ix])+log(wi[jcom])
            ixM = ix+ngh*(jcom-1)
            fill!(llN, gammaM[ixM])
            Yeppp.add!(llN, llN, xb)
            negateiftrue!(llN, Y)
            Yeppp.exp!(llN, llN)
            log1p!(llN)

            for i in 1:n
                Wim[i, ixM] = wtmp
            end
            for i in 1:N
                @inbounds Wim[groupindex[i], ixM] -= llN[i]
            end
        end
    end

    for i in 1:n
        u = maximum(Wim[i, :])
        for jcol in 1:C*ngh
            @inbounds Wim[i, jcol] = Wim[i, jcol] - u
        end
    end
    Yeppp.exp!(Wim, Wim)
    for i in 1:n
        u = sum(Wim[i, :])
        for jcol in 1:C*ngh
            @inbounds Wim[i, jcol] = Wim[i, jcol] / u
        end
    end
end

function updateθ!(wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, gammaM::Vector{Float64}, Wim::Matrix{Float64}, Wm::Matrix{Float64}, sn::Vector{Float64}, an::Real, N::Int, J::Int, n::Int, C::Int, ngh::Int)

    # A_mul_B!(xb, X, betas)
    mean!(Wm, Wim)
    divide!(Wm, Wm, sum(Wm))
    for kcom in 1:C
        ind = (1+ngh*(kcom-1)):ngh*kcom
        wi[kcom] = sum(Wm[ind])
        mu[kcom] = wsum(gammaM[ind], Wm[ind]) / wi[kcom]
        sigmas[kcom] = sqrt((wsum((gammaM[ind] .- mu[kcom]).^2, Wm[ind]) + 2 * an * sn[kcom]^2/n) / (wi[kcom]) + 2 * an/n)
    end

end

function updateβ!(β::Vector{Float64}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,  gammaM::Vector{Float64}, Wim::Matrix{Float64}, lln::Vector{Float64}, llN::Vector{Float64}, llN2::Vector{Float64}, xb::Vector{Float64}, N::Int, J::Int, n::Int, C::Int, ngh::Int, Qmaxiteration::Int)

    opt = Opt(:LD_LBFGS, J)
    maxeval!(opt, Qmaxiteration)
    max_objective!(opt, (beta2, storage)->EM_Q1(beta2, storage, X, Y,  groupindex, gammaM, Wim, lln, llN, llN2, xb, N, J, n, C*ngh))

    #(minf,β,ret)=optimize(opt, β)
    optimize!(opt, β)
end

function EM_Q1(beta2::Array{Float64,1}, storage::Vector, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, gammaM::Vector{Float64}, Wim::Matrix{Float64}, lln::Vector{Float64}, llN::Vector{Float64}, llN2::Vector{Float64}, xb::Vector{Float64}, N::Int, J::Int, n::Int, M::Int)

    if length(storage)>0
        fill!(storage, 0.0)
    end
    A_mul_B!(xb, X, beta2)
    res = 0.0
    for jcol in 1:M

        fill!(llN, gammaM[jcol])
        Yeppp.add!(llN, llN, xb)
        negateiftrue!(llN, Y)
        Yeppp.exp!(llN, llN)
        if length(storage) > 0
            copy!(llN2, llN)
            x1x!(llN2)
            negateiffalse!(llN2, Y)

            for i in 1:N
                groupindexi = groupindex[i]
                for j in 1:J
                    @inbounds storage[j] += llN2[i] * X[i,j] * Wim[groupindexi, jcol]
                end
            end
        end

        log1p!(llN)
        fill!(lln, 0.0)
        for i in 1:N
            @inbounds lln[groupindex[i]] += llN[i]
        end
        res += wsum(lln, Wim[:, jcol])
    end
    -res
end

function latentgmmEM(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent::Int, β_init::Vector{Float64}, wi_init::Vector{Float64}, mu_init::Vector{Float64}, sigmas_init::Vector{Float64};
    maxiteration::Int=100, tol::Real=.005,  ngh::Int=100, ghx::Vector=zeros(ngh), ghw::Vector=zeros(ngh),
     sn::Vector{Float64}=sigmas_init, an::Float64=1.0/maximum(groupindex),
    debuginfo::Bool=false, Qmaxiteration::Int=2, whichtosplit::Int=1, tau::Real=.5, wifixed::Bool=false, dotest::Bool=true, 
     mu_lb::Vector=fill(-Inf, ncomponent), mu_ub::Vector=fill(Inf, ncomponent),
     Wim::Matrix{Float64}=zeros(maximum(groupindex), ncomponent*length(ghx)), Wm::Matrix{Float64}=zeros(1, size(Wim, 2)), lln::Vector{Float64}=zeros(maximum(groupindex)), llN::Vector{Float64}=zeros(length(Y)),
    llN2::Vector{Float64}=zeros(length(Y)), xb::Vector{Float64}=zeros(length(Y)), gammaM::Vector{Float64}=zeros( size(Wim, 2)))

    # initialize theta
    length(wi_init) == length(mu_init) == length(sigmas_init) == ncomponent || error("The length of initial values should be $ncomponent")
    N,J=size(X)
    length(β_init) == J || error("Initial values of fixed efffect coefficients should have same dimension as X")
    length(ghx) == length(ghw) == ngh || error("length of ghx and ghw should be the same as ngh") 
    n = maximum(groupindex)
    M = ncomponent*ngh

    wi = copy(wi_init)
    mu = copy(mu_init)
    sigmas = copy(sigmas_init)
    β = copy(β_init)

    wi_old = ones(wi)./ncomponent
    mu_old = zeros(mu)
    sigmas_old = ones(sigmas)
    beta_old = randn(J)
    if !wifixed || (ghx[1] == 0.0)
        ghx, ghw = hermite(ngh)
    end
    ll0 = -Inf
    for iter_em in 1:maxiteration
        for ix in 1:ngh, jcom in 1:ncomponent
            ixM = ix+ngh*(jcom-1)
            gammaM[ixM] = ghx[ix]*sigmas[jcom]*sqrt(2)+mu[jcom]
        end

        copy!(wi_old, wi)
        copy!(mu_old, mu)
        copy!(sigmas_old, sigmas)

        A_mul_B!(xb, X, β)
        integralweight!(Wim, X, Y, groupindex, gammaM, wi, ghw, llN, xb, N, J, n, ncomponent, ngh)

        if debuginfo
            println("At $(iter_em)th iteration:")
        end
        if !stopRule(β, beta_old, tol=tol/10)
            copy!(beta_old, β)
            updateβ!(β, X, Y, groupindex, gammaM, Wim, lln, llN, llN2, xb, N, J, n, ncomponent, ngh, Qmaxiteration)
            if debuginfo
                println("beta=", β)
            end
        end
        updateθ!(wi, mu, sigmas, X, Y, groupindex, gammaM, Wim, Wm, sn, an, N, J, n, ncomponent, ngh)
        if wifixed
            wi_tmp = wi[whichtosplit]+wi[whichtosplit+1]
            wi[whichtosplit] = wi_tmp*tau
            wi[whichtosplit+1] = wi_tmp*(1-tau)
            Yeppp.max!(mu, mu, mu_lb)
            Yeppp.min!(mu, mu, mu_ub)
        end
        if debuginfo
            println("wi=$wi")
            println("mu=$mu")
            println("sigma=$sigmas")
            println("ll=",marginallikelihood(β, X, Y, groupindex, n, wi, mu, sigmas, ghx, ghw, llN, lln, xb, Wim))
        end
        if dotest
            ll = marginallikelihood(β, X, Y, groupindex, n, wi, mu, sigmas, ghx, ghw, llN, lln, xb, Wim)
            if abs(ll - ll0) < 1e-8
                break
            end 
            ll0 = ll
        end
        if stopRule(vcat(β, wi, mu, sigmas), vcat(beta_old, wi_old, mu_old, sigmas_old), tol=tol) && (iter_em > 3)
            if debuginfo
                println("latentgmmEM converged at $(iter_em)th iteration")
            end
            break
        end
        if (iter_em == maxiteration) && (maxiteration > 15)
            warn("latentgmmEM not converge! $(wifixed)")
        end
    end
    return(wi, mu, sigmas, β, marginallikelihood(β, X, Y, groupindex, n, wi, mu, sigmas, ghx, ghw, llN, lln, xb, Wim)+sum(pn(sigmas, sn, an=an)))
end


"""
The interior step for loglikelihoodratio
"""
function loglikelihoodratioEM_ctau(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent1::Int,  betas0::Vector{Float64}, wi_C1::Vector{Float64},  whichtosplit::Int64, tau::Float64, mu_lb::Vector{Float64}, mu_ub::Vector{Float64}, sigmas_lb::Vector{Float64}, sigmas_ub::Vector{Float64}; ntrials::Int=25, ngh::Int=100, sn::Vector{Float64}=sigmas_ub ./ 2, an=.25, debuginfo::Bool=false, gammaM::Vector = zeros(maximum(groupindex), Mmax), Wim::Matrix = zeros(maximum(groupindex), ngh*ncomponent), llN::Vector=zeros(length(Y)), llN2::Vector = zeros(length(Y)), xb::Vector=zeros(length(Y)))

    nF = maximum(groupindex)
    tau = min(tau, 1-tau)
    ghx, ghw = hermite(ngh)

    wi = repmat(wi_C1, 1, 4*ntrials)
    mu = zeros(ncomponent1, 4*ntrials)
    sigmas = ones(ncomponent1, 4*ntrials)
    betas = repmat(betas0, 1, 4*ntrials)
    ml = -Inf .* ones(4*ntrials)
    for i in 1:4*ntrials
        mu[:, i] = rand(ncomponent1) .* (mu_ub .- mu_lb) .+ mu_lb
        sigmas[:, i] = rand(ncomponent1) .* (sigmas_ub .- sigmas_lb) .+ sigmas_lb

        wi[:, i], mu[:, i], sigmas[:, i], betas[:, i], ml[i] = latentgmmEM(X, Y, groupindex, ncomponent1, betas0, wi[:, i], mu[:, i], sigmas[:, i], whichtosplit=whichtosplit, tau=tau, ghx=ghx, ghw=ghw, mu_lb=mu_lb, mu_ub=mu_ub, maxiteration=10, sn=sn, an=an, gammaM = gammaM, Wim=Wim, llN=llN, llN2=llN2, xb=xb, Qmaxiteration=2, wifixed=true, ngh=ngh)
    end

    mlperm = sortperm(ml)
    for j in 1:ntrials
        i = mlperm[4*ntrials+1 - j] # start from largest ml
        wi[:, i], mu[:, i], sigmas[:, i], betas[:, i], ml[i] = latentgmmEM(X, Y, groupindex, ncomponent1, betas[:, i], wi[:, i], mu[:, i], sigmas[:, i], whichtosplit=whichtosplit, tau=tau, ghx=ghx, ghw=ghw, mu_lb=mu_lb,mu_ub=mu_ub, maxiteration=500, sn=sn, an=an, debuginfo=debuginfo, gammaM = gammaM, Wim=Wim, llN=llN, llN2=llN2, xb=xb, Qmaxiteration=2, wifixed=true, ngh=ngh)
    end

    mlmax, imax = findmax(ml[mlperm[(3*ntrials+1):4*ntrials]])
    imax = mlperm[3*ntrials+imax]

    re=latentgmmEM(X, Y, groupindex, ncomponent1, betas[:, imax], wi[:, imax], mu[:, imax], sigmas[:, imax], maxiteration=3, an=an, sn=sn, debuginfo=debuginfo, ngh=ngh)

    return(re[5])
end

function loglikelihoodratioEM(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent1::Int; vtau::Vector{Float64}=[.5,.3,.1;], ntrials::Int=25, ngh::Int=100, debuginfo::Bool=false, ctauparallel=true)
    C0 = ncomponent1 - 1
    C1 = ncomponent1
    nF = maximum(groupindex)
    M = ngh * ncomponent1
    an1 = 1/nF
    gamma_init, betas_init, sigmas_tmp = maxposterior(X, Y, groupindex)
    if C0 ==1
        wi_init, mu_init, sigmas_init, betas_init, ml_C0 = latentgmmEM(X, Y, groupindex, 1, betas_init, [1.0], [mean(gamma_init)], [std(gamma_init)], maxiteration=1000, an=an1, sn=std(gamma_init).*ones(C0), ngh=ngh)
    else C0 > 1
        wi_init, mu_init, sigmas_init, betas_init, ml_C0 = latentgmmEM(X, Y, groupindex, 1, betas_init, [1.0], [mean(gamma_init)], [std(gamma_init)], maxiteration=100, an=an1, sn=std(gamma_init).*ones(C0), ngh=ngh)
        gamma_init = predictgamma(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init)
        wi_init, mu_init, sigmas_init, ml_tmp = gmm(gamma_init, C0, ones(C0)/C0, quantile(gamma_init, linspace(0, 1, C0+2)[2:end-1]), ones(C0), an=an1)
        wi_init, mu_init, sigmas_init, betas_init, ml_C0 = latentgmmEM(X, Y, groupindex, C0, betas_init, wi_init, mu_init, sigmas_init, maxiteration=500, an=an1, sn=std(gamma_init).*ones(C0), ngh=ngh)
        trand=LatentGaussianMixtureModel.asymptoticdistribution(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init)
    end
    mingamma = minimum(gamma_init)
    maxgamma = maximum(gamma_init)

    or = sortperm(mu_init)
    wi0 = wi_init[or]
    mu0 = mu_init[or]
    sigmas0 = sigmas_init[or]
    betas0 = betas_init
    an = decidepenalty(wi0, mu0, sigmas0, nF)

    N,J=size(X)
    gammaM = zeros(ngh*ncomponent1)
    Wim = zeros(nF, ngh*ncomponent1)
    #Wm = zeros(ngh*ncomponent1)
    llN = zeros(N)
    llN2 = zeros(N)
    xb = zeros(N)
    if ctauparallel
        lr=@parallel (max) for irun in 1:(C0*length(vtau))

            whichtosplit = mod1(irun, C0)
            i = cld(irun, C0)
            ind = [1:whichtosplit, whichtosplit:C0;]
            if C1==2
                mu_lb = mingamma .* ones(2)
                mu_ub = maxgamma .* ones(2)
            elseif C1>2
                mu_lb = [mingamma, (mu0[1:(C0-1)] .+ mu0[2:C0])./2;]
                mu_ub = [(mu0[1:(C0-1)] .+ mu0[2:C0])./2, maxgamma;]
                mu_lb = mu_lb[ind]
                mu_ub = mu_ub[ind]
            end
            sigmas_lb = 0.25 .* sigmas0[ind]
            sigmas_ub = 2 .* sigmas0[ind]

            wi_C1 = wi0[ind]
            wi_C1[whichtosplit] = wi_C1[whichtosplit]*vtau[i]
            wi_C1[whichtosplit+1] = wi_C1[whichtosplit+1]*(1-vtau[i])

            loglikelihoodratioEM_ctau(X, Y, groupindex, ncomponent1, betas0, wi_C1, whichtosplit, vtau[i], mu_lb, mu_ub, sigmas_lb, sigmas_ub, ntrials=ntrials, ngh=ngh, sn=sigmas0[ind], an=an, debuginfo=debuginfo, gammaM = gammaM, Wim=Wim, llN=llN, llN2=llN2, xb=xb)

        end
        if debuginfo
            println(summarystats(trand))
        end
        Tvalue = 2*(lr - ml_C0)
        if C0 == 1
            pvalue = 1 - cdf(Chisq(2), Tvalue)
        else
            pvalue = mean(trand .> 2*(lr - ml_C0))
        end
        return [Tvalue, pvalue;]
    else
        lr = zeros(length(vtau), C0)
        for whichtosplit in 1:C0, i in 1:length(vtau)

             ind = [1:whichtosplit, whichtosplit:C0;]
             if C1==2
                 mu_lb = mingamma .* ones(2)
                 mu_ub = maxgamma .* ones(2)
             elseif C1>2
                 mu_lb = [mingamma, (mu0[1:(C0-1)] .+ mu0[2:C0])./2;]
                 mu_ub = [(mu0[1:(C0-1)] .+ mu0[2:C0])./2, maxgamma;]
                 mu_lb = mu_lb[ind]
                 mu_ub = mu_ub[ind]
             end
             sigmas_lb = 0.25 .* sigmas0[ind]
             sigmas_ub = 2 .* sigmas0[ind]

             wi_C1 = wi0[ind]
             wi_C1[whichtosplit] = wi_C1[whichtosplit]*vtau[i]
             wi_C1[whichtosplit+1] = wi_C1[whichtosplit+1]*(1-vtau[i])

             lr[i, whichtosplit]=loglikelihoodratioEM_ctau(X, Y, groupindex, ncomponent1, betas0, wi_C1, whichtosplit, vtau[i], mu_lb, mu_ub, sigmas_lb, sigmas_ub, ntrials=ntrials, ngh=ngh, sn=sigmas0[ind], an=an, debuginfo=debuginfo, gammaM = gammaM, Wim=Wim, llN=llN, llN2=llN2, xb=xb)
         end
         if debuginfo
             println(summarystats(trand))
         end
         Tvalue = 2*(maximum(lr) - ml_C0)
         if C0 == 1
             pvalue = 1 - cdf(Chisq(2), Tvalue)
         else
             pvalue = mean(trand .> 2*(maximum(lr) - ml_C0))
         end
         return [Tvalue, pvalue;]
    end
end
