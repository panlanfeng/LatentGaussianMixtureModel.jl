function integralweight!(Wim::Matrix{Float64},
    X::Matrix{Float64}, Y::AbstractArray{Bool, 1},
    groupindex::IntegerVector,  gammaM::Vector{Float64},
    wi::Vector{Float64}, ghw::Vector{Float64},
    llN::Vector{Float64}, llN2::Vector{Float64},
    xb::Vector{Float64},  N::Int, J::Int,
    n::Int, C::Int, ngh::Int)
    #A_mul_B!(xb, X, betas)
    ll = 0.
    for jcom in 1:C
        for ix in 1:ngh
            wtmp = log(ghw[ix])+log(wi[jcom])
            ixM = ix+ngh*(jcom-1)
            fill!(llN, gammaM[ixM])
            Yeppp.add!(llN, llN, xb)
            negateiftrue!(llN, Y)
            # Yeppp.exp!(llN, llN)
            # log1p!(llN)
            log1pexp!(llN, llN, llN2, N)

            for i in 1:n
                Wim[i, ixM] = wtmp
            end
            for i in 1:N
                @inbounds Wim[groupindex[i], ixM] -= llN[i]
            end
        end
    end
    for i in 1:n
        ll += logsumexp(Wim[i,:])
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
    return ll - n*logπ/2
end

function updateθ!(wi::Vector{Float64}, mu::Vector{Float64},
    sigmas::Vector{Float64}, X::Matrix{Float64},
    Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,
    gammaM::Vector{Float64}, Wim::Matrix{Float64},
    Wm::Matrix{Float64}, sn::Vector{Float64},
    an::Real, N::Int, J::Int, n::Int, C::Int, ngh::Int)

    # A_mul_B!(xb, X, betas)
    mean!(Wm, Wim)
    divide!(Wm, Wm, sum(Wm))
    for kcom in 1:C
        ind = (1+ngh*(kcom-1)):ngh*kcom
        wi[kcom] = sum(Wm[ind])
        mu[kcom] = wsum(gammaM[ind], Wm[ind]) / wi[kcom]
        sigmas[kcom] = sqrt((wsum((gammaM[ind] .- mu[kcom]).^2, Wm[ind]) + 2 * an * sn[kcom]^2/n) / (wi[kcom] + 2 * an/n))
    end

end
function updateβ!(β::Vector{Float64}, X::Matrix{Float64},
    Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,
    minStepFac::Real, betadevtol::Real,
    XWX::Matrix{Float64}, XWY::Vector{Float64},
    Xscratch::Matrix{Float64},
    gammaM::Vector{Float64}, Wim::Matrix{Float64},
    lln::Vector{Float64}, llN::Vector{Float64},
    llN2::Vector{Float64}, llN3::Vector{Float64}, xb::Vector{Float64},
    N::Int, J::Int, n::Int, C::Int, ngh::Int, Qmaxiteration::Int)
    M = C*ngh
    dev0 = negdeviance(β, X, Y, groupindex,
    gammaM, Wim, lln, llN, llN2, xb, N, J, n, M)

    for iterbeta in 1:Qmaxiteration
        deltabeta!(XWY, XWX, X, Y, groupindex, β, Xscratch, gammaM, Wim,
        llN, llN2, llN3, xb, N, J, n, M)
        if maxabs(XWY ./ (abs(β)+0.001)) < 1e-5
            break
        end
        f = 1.
        dev = negdeviance(β .+ f .* XWY, X, Y, groupindex,
        gammaM, Wim, lln, llN, llN2, xb, N, J, n, M)
        while dev < dev0
            f ./=2
            f > minStepFac || error("step-halving failed at beta = $(β), deltabeta=$(XWY), dev=$(dev), dev0=$dev0, f=$f")
            dev = negdeviance(β .+ f .* XWY, X, Y, groupindex,
            gammaM, Wim, lln, llN, llN2, xb, N, J, n, M)
        end
        for j in 1:J
            β[j] += f * XWY[j]
        end

        if dev - dev0 < betadevtol
            break
        end
        dev0 = dev
    end

end
function deltabeta!(XWY::Vector{Float64},
    XWX::Matrix{Float64}, X::Matrix{Float64},
    Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,
    β::Vector{Float64}, Xscratch::Matrix{Float64},
    gammaM::Vector{Float64}, Wim::Matrix{Float64},
    llN::Vector{Float64},llN2::Vector{Float64},
    llN3::Vector{Float64}, xb::Vector{Float64},
    N::Int, J::Int, n::Int, M::Int)
    A_mul_B!(xb, X, β)
    fill!(XWX, 0.)
    fill!(XWY, 0.)

    for jcol in 1:M
        fill!(llN, gammaM[jcol])
        Yeppp.add!(llN, llN, xb)
        logistic!(llN, llN, N)

        @inbounds for i in 1:N
            llN2[i] = llN[i]*(1.0 -llN[i]) #working weight
            llN[i] = Y[i] ? 1.0 - llN[i] : -llN[i] #working response without denominator
        end
        relocate!(llN3, Wim[:, jcol], groupindex, N)
        scale!(Xscratch, llN3, X)
        Base.BLAS.gemv!('T', 1.0, Xscratch, llN, 1.0, XWY)
        scale!(Xscratch, llN2, Xscratch)
        Base.BLAS.gemm!('T', 'N', 1.0, Xscratch, X, 1.0, XWX)
    end
    A_ldiv_B!(cholfact!(XWX, :U), XWY)
end
function negdeviance(beta2::Vector{Float64}, X::Matrix{Float64},
    Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,
    gammaM::Vector{Float64}, Wim::Matrix{Float64},
    lln::Vector{Float64}, llN::Vector{Float64},
    llN2::Vector{Float64}, xb::Vector{Float64},
    N::Int, J::Int, n::Int, M::Int)
    dev = 0.
    A_mul_B!(xb, X, beta2)
    for jcol in 1:M
        fill!(llN, gammaM[jcol])
        Yeppp.add!(llN, llN, xb)
        negateiftrue!(llN, Y)
        log1pexp!(llN, llN, llN2, N)
        fill!(lln, 0.0)
        for i in 1:N
            @inbounds lln[groupindex[i]] += llN[i]
        end
        dev += wsum(lln, Wim[:, jcol])
    end
    -dev
end

function latentgmm(X::Matrix{Float64},
    Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,
    ncomponent::Int, β_init::Vector{Float64},
    wi_init::Vector{Float64}, mu_init::Vector{Float64},
    sigmas_init::Vector{Float64};
    maxiteration::Int=100, tol::Real=.005,
    ngh::Int=100, ghx::Vector=zeros(ngh), ghw::Vector=zeros(ngh),
    sn::Vector{Float64}=sigmas_init, an::Float64=1.0/maximum(groupindex),
    debuginfo::Bool=false, Qmaxiteration::Int=5,
    whichtosplit::Int=1, tau::Real=.5, taufixed::Bool=false,
    mu_lb::Vector=fill(-Inf, ncomponent),
    mu_ub::Vector=fill(Inf, ncomponent),
    Wim::Matrix{Float64}=zeros(maximum(groupindex), ncomponent*ngh),
    Wm::Matrix{Float64}=zeros(1, ncomponent*ngh),
    lln::Vector{Float64}=zeros(maximum(groupindex)),
    llN::Vector{Float64}=zeros(length(Y)),
    llN2::Vector{Float64}=zeros(length(Y)),
    llN3::Vector{Float64}=zeros(length(Y)),
    Xscratch::Matrix{Float64}=copy(X),
    xb::Vector{Float64}=zeros(length(Y)),
    gammaM::Vector{Float64}=zeros( ncomponent*ngh),
    dotest::Bool=false, epsilon::Real=1e-6,
    updatebeta::Bool=true, pl::Bool=true, ptau::Bool=false)

    # initialize theta
    length(wi_init) == length(mu_init) == length(sigmas_init) == ncomponent || error("The length of initial values should be $ncomponent")
    N,J=size(X)
    length(β_init) == J || error("Initial values of fixed efffect coefficients should have same dimension as X")
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
    if !taufixed || (ghx[1] == 0.0)
        ghx, ghw = gausshermite(ngh)
    end
    XWX = zeros(J, J)
    XWY = zeros(J)

    ll0=-Inf
    ll = 0.
    alreadystable = maxiteration <= 3
    for iter_em in 1:maxiteration
        for ix in 1:ngh, jcom in 1:ncomponent
            ixM = ix+ngh*(jcom-1)
            gammaM[ixM] = ghx[ix]*sigmas[jcom]*sqrt(2)+mu[jcom]
        end

        copy!(wi_old, wi)
        copy!(mu_old, mu)
        copy!(sigmas_old, sigmas)

        A_mul_B!(xb, X, β)
        ll=integralweight!(Wim, X, Y, groupindex, gammaM, wi, ghw, llN, llN2, xb, N, J, n, ncomponent, ngh)
        lldiff = ll - ll0
        ll0 = ll
        if lldiff < 1e-4
            #alreadystable = true
            Qmaxiteration = 2*Qmaxiteration
        end
        if dotest
            if (lldiff < epsilon) && (iter_em > 3)
                break
            end
        end
        if debuginfo
            println("At $(iter_em)th iteration:")
        end
        if updatebeta && (mod1(iter_em, 3) == 1 || alreadystable)
            copy!(beta_old, β)
            updateβ!(β, X, Y, groupindex, .001, .001,
            XWX, XWY, Xscratch, gammaM, Wim, lln, llN, llN2, llN3,
            xb, N, J, n, ncomponent, ngh, Qmaxiteration)
            if debuginfo
                println("beta=", β)
            end
        end
        updateθ!(wi, mu, sigmas, X, Y, groupindex,
        gammaM, Wim, Wm, sn, an, N, J, n, ncomponent, ngh)
        if taufixed
            for kcom in 1:ncomponent
                wi[kcom]=(wi[kcom]*n+1.0/ncomponent)/(n+1)
            end
            wi_tmp = wi[whichtosplit]+wi[whichtosplit+1]
            wi[whichtosplit] = wi_tmp*tau
            wi[whichtosplit+1] = wi_tmp*(1-tau)
            Yeppp.max!(mu, mu, mu_lb)
            Yeppp.min!(mu, mu, mu_ub)
        end
        if any(wi .< 1e-8)
            warn("Some elements of $wi are too small. Consider another starting value or reduce the number of components. Give up.")
            break
        end
        if debuginfo
            println("wi=$wi")
            println("mu=$mu")
            println("sigma=$sigmas")
            println("ll=",ll)
        end

        if !dotest
            if stopRule(vcat(β, wi, mu, sigmas), vcat(beta_old, wi_old, mu_old, sigmas_old), tol=tol)
                if debuginfo
                    println("latentgmm converged at $(iter_em)th iteration")
                end
                break
            end
        end
        if (iter_em == maxiteration) && (maxiteration > 50)
            warn("latentgmm not converge! $(iter_em), $(taufixed),
            $(ll), $(lldiff), $(wi), $(mu), $(sigmas), $(β)")
            println("latentgmm not converge! $(iter_em), $(taufixed),
            $(ll), $(lldiff), $(wi), $(mu), $(sigmas), $(β)")
        end
    end
    if pl
        ll += sum(pn(sigmas, sn, an=an))
    end
    if ptau
        tau2 = wi[whichtosplit] / (wi[whichtosplit]+wi[whichtosplit+1])
        ll += log(1 - abs(1 - 2*tau2))
    end
    return(wi, mu, sigmas, β, ll)
end

function latentgmm(X::Matrix{Float64},
    Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,
    ncomponent::Int; opts...)

    if ncomponent > 1
        wi_init, mu_init, sigmas_init, betas_init, ml_tmp = LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, 1, ones(size(X)[2]), [1.0], [0.], [1.])
        gamma_init = LatentGaussianMixtureModel.predictgamma(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init);
        wi_init, mu_init, sigmas_init, ml_tmp = LatentGaussianMixtureModel.gmm(gamma_init, ncomponent)
    else
        betas_init = ones(size(X)[2])
        wi_init=[1.0;]
        mu_init=[0.0;]
        sigmas_init=[1.0;]
    end
    #debuginfo && println("Initial:", wi_init, mu_init, sigmas_init, betas_init)
    return LatentGaussianMixtureModel.latentgmm(X, Y, groupindex, ncomponent, betas_init, wi_init, mu_init, sigmas_init; opts...)

end

"""
The interior step for `EMtest`
"""
function latentgmmrepeat(X::Matrix{Float64},
    Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, C::Int,
    betas0::Vector{Float64}, wi_C1::Vector{Float64},
    mu_lb::Vector{Float64}, mu_ub::Vector{Float64},
    sigmas_lb::Vector{Float64}, sigmas_ub::Vector{Float64}; 
    taufixed::Bool=false, whichtosplit::Int64=1, tau::Float64=0.5,
    ntrials::Int=25, ngh::Int=100,
    sn::Vector{Float64}=sigmas_ub ./ 2, an=.25, 
    debuginfo::Bool=false,
    gammaM::Vector = zeros(ngh*C),
    Wim::Matrix = zeros(maximum(groupindex), ngh*C),
    llN::Vector=zeros(length(Y)),
    llN2::Vector = zeros(length(Y)),
    llN3::Vector{Float64}=zeros(length(Y)),
    Xscratch::Matrix{Float64}=copy(X),
    xb::Vector=zeros(length(Y)), tol::Real=.005, 
    pl::Bool=false, ptau::Bool=false)

    n = maximum(groupindex)
    tau = min(tau, 1-tau)
    ghx, ghw = gausshermite(ngh)

    wi = repmat(wi_C1, 1, 4*ntrials)
    mu = zeros(C, 4*ntrials)
    sigmas = ones(C, 4*ntrials)
    betas = repmat(betas0, 1, 4*ntrials)
    ml = -Inf .* ones(4*ntrials)
    for i in 1:4*ntrials
        mu[:, i] = rand(C) .* (mu_ub .- mu_lb) .+ mu_lb
        sigmas[:, i] = rand(C) .* (sigmas_ub .- sigmas_lb) .+ sigmas_lb

        wi[:, i], mu[:, i], sigmas[:, i], betas[:, i], ml[i] =
             latentgmm(X, Y, groupindex, C, betas0,
             wi[:, i], mu[:, i], sigmas[:, i],
             whichtosplit=whichtosplit, tau=tau,
             ghx=ghx, ghw=ghw, mu_lb=mu_lb, mu_ub=mu_ub,
             maxiteration=16, sn=sn, an=an,
             gammaM = gammaM, Wim=Wim,
             llN=llN, llN2=llN2, llN3=llN3,
             Xscratch=Xscratch, xb=xb,
             Qmaxiteration=2, taufixed=taufixed, ngh=ngh,
             dotest=false, tol=tol)
    end

    mlperm = sortperm(ml)
    for j in 1:ntrials
        i = mlperm[4*ntrials+1 - j] # start from largest ml
        wi[:, i], mu[:, i], sigmas[:, i], betas[:, i], ml[i] =
            latentgmm(X, Y, groupindex, C, betas[:, i],
            wi[:, i], mu[:, i], sigmas[:, i],
            whichtosplit=whichtosplit, tau=tau, ghx=ghx, ghw=ghw,
            mu_lb=mu_lb,mu_ub=mu_ub, maxiteration=2000,
            sn=sn, an=an, gammaM = gammaM,
            Wim=Wim, llN=llN, llN2=llN2, llN3=llN3,
            Xscratch=Xscratch, xb=xb,
            Qmaxiteration=5, taufixed=taufixed, ngh=ngh,
            dotest=false, tol=tol)
    end

    mlmax, imax = findmax(ml[mlperm[(3*ntrials+1):4*ntrials]])
    imax = mlperm[3*ntrials+imax]

    re=latentgmm(X, Y, groupindex, C,
        betas[:, imax], wi[:, imax], mu[:, imax], sigmas[:, imax],
         maxiteration=2, an=an, sn=sn, debuginfo=false, ngh=ngh,
         tol=0., pl=pl, ptau=ptau, whichtosplit=whichtosplit)
    debuginfo && println("Trial:", re)
    return(re)
end

function EMtest(X::Matrix{Float64},
    Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,
    C0::Int; vtau::Vector{Float64}=[.5;],
    ntrials::Int=25, ngh::Int=100, debuginfo::Bool=false,
    ctauparallel=true, tol::Real=0.001)
    
    C1 = C0 + 1
    n = maximum(groupindex)
    M = ngh * C1
    an1 = 1/n
    N,J=size(X)
    llN = zeros(N)
    llN2 = zeros(N)
    llN3 = zeros(N)
    Xscratch = copy(X)
    xb = zeros(N)
    
    #gamma_init, betas_init, sigmas_tmp = maxposterior(X, Y, groupindex)
    wi_init, mu_init, sigmas_init, betas_init, ml_C0 =
    latentgmm(X, Y, groupindex, 1, ones(J),
    [1.0], [0.], [1.], maxiteration=100, an=an1, sn=ones(C0), tol=.005)
    gamma_init = predictgamma(X, Y, groupindex,
        wi_init, mu_init, sigmas_init, betas_init)
    wi_init, mu_init, sigmas_init, ml_tmp = gmm(gamma_init, C0)
    mingamma = minimum(gamma_init)
    maxgamma = maximum(gamma_init)

    wi_init, mu_init, sigmas_init, betas_init, ml_C0 = latentgmmrepeat(X, Y,
       groupindex, C0, betas_init, wi_init,
       ones(C0).*mingamma, ones(C0).*maxgamma, 
       0.25 .* sigmas_init, 2.*sigmas_init,
       taufixed=false,
       ntrials=ntrials, ngh=ngh, 
       sn=std(gamma_init).*ones(C0), an=an1,
       debuginfo=debuginfo,
       llN=llN, llN2=llN2, xb=xb, tol=tol, 
       pl=false, ptau=false)
    
    # wi_init, mu_init, sigmas_init, betas_init, ml_C0 =
    #     latentgmm(X, Y, groupindex, C0, betas_init, wi_init, mu_init,
    #     sigmas_init, maxiteration=2000, an=an1,
    #     sn=std(gamma_init).*ones(C0), ngh=100, dotest=false, tol=.001,
    #     Qmaxiteration=5, pl=false, ptau=false)

    if C0 > 1
        trand=asymptoticdistribution(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init)
    end

    if debuginfo
        println("ml_C0=", ml_C0)
    end

    or = sortperm(mu_init)
    wi0 = wi_init[or]
    mu0 = mu_init[or]
    sigmas0 = sigmas_init[or]
    betas0 = betas_init
    an = decidepenalty(wi0, mu0, sigmas0, n)

    N,J=size(X)
    gammaM = zeros(ngh*C1)
    Wim = zeros(n, ngh*C1)
    lr = 0.0
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

            ml_tmp=latentgmmrepeat(X, Y,
               groupindex, C1, betas0, wi_C1,
               mu_lb, mu_ub, sigmas_lb, sigmas_ub,
               taufixed=true, whichtosplit=whichtosplit, tau=vtau[i], 
               ntrials=ntrials, ngh=ngh, 
               sn=sigmas0[ind], an=an,
               debuginfo=debuginfo, gammaM = gammaM, Wim=Wim,
               llN=llN, llN2=llN2, xb=xb, tol=tol, 
               pl=false, ptau=false)[5]
            if debuginfo
                println(whichtosplit, " ", vtau[i], "->", ml_tmp)
            end
            ml_tmp
        end
    else
        lrv = zeros(length(vtau), C0)
        kl_tmp=zeros(length(vtau), C0)
        ll_tmp = zeros(length(vtau), C0)
        for whichtosplit in 1:C0, i in 1:length(vtau)

             #whichtosplit = mod1(irun, C0)
             #i = cld(irun, C0)
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

             lrv[i, whichtosplit] = latentgmmrepeat(X, Y,
                groupindex, C1, betas0, wi_C1,
                mu_lb, mu_ub, sigmas_lb, sigmas_ub,
                taufixed=true, whichtosplit=whichtosplit, tau=vtau[i], 
                ntrials=ntrials, ngh=ngh, 
                sn=sigmas0[ind], an=an,
                debuginfo=debuginfo, gammaM = gammaM, Wim=Wim,
                llN=llN, llN2=llN2, xb=xb, tol=tol, 
                pl=false, ptau=false)[5]
            if debuginfo
                println(whichtosplit, " ", vtau[i], "->",
                lrv[i, whichtosplit])
            end
        end
        lr = maximum(lrv)
    end
    if debuginfo
        println("lr=", lr)
    end
    Tvalue = 2*(lr - ml_C0)
    if C0 == 1
        pvalue = 1 - cdf(Chisq(2), Tvalue)
    else
        pvalue = mean(trand .> Tvalue)
    end
    return(Tvalue, pvalue)
end
