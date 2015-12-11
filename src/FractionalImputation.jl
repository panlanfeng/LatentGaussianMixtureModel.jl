function FIintegralweight!(Wim::Matrix{Float64}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,  gammaM::Matrix{Float64}, gammah::Matrix{Float64}, wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, llN::Vector{Float64}, xb::Vector{Float64}, proposingdist::Distribution,  N::Int, J::Int, n::Int, M::Int)
    #A_mul_B!(xb, X, betas)
    m = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)
    copy!(Wim, gammah)
    ll =0.0
    for ixM in 1:M

        relocate!(llN, gammaM[:, ixM], groupindex, N)
        Yeppp.add!(llN, llN, xb)
        negateiftrue!(llN, Y)
        log1pexp!(llN, llN, N)

        for i in 1:n
            Wim[i, ixM] += logpdf(m, gammaM[i,ixM]) #- logpdf(proposingdist, gammaM[i,ixM] - mean(m))
        end
        for i in 1:N
            @inbounds Wim[groupindex[i], ixM] -= llN[i]
        end
    end
    for i in 1:n
        ll += logsumexp(Wim[i,:])
    end
    for i in 1:n
        u = maximum(Wim[i, :])
        for jcol in 1:M
            @inbounds Wim[i, jcol] = Wim[i, jcol] - u
        end
    end
    Yeppp.exp!(Wim, Wim)
    for i in 1:n
        u = sum(Wim[i, :])
        for jcol in 1:M
            @inbounds Wim[i, jcol] = Wim[i, jcol] / u
        end
    end
    return(ll - n*log(M))
end

function completescore!(stheta::Array{Float64, 3}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, β::Vector{Float64}, gammaM::Matrix{Float64}, Wim::Matrix{Float64}, xb::Vector{Float64}, llN::Vector{Float64}, N::Int, J::Int, n::Int, C::Int, M::Int)

    m = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)
    explogf = zeros(C)
    fill!(stheta, 0.0)
    for jcol in 1:M
        relocate!(llN, gammaM[:, jcol], groupindex, N)
        Yeppp.add!(llN, llN, xb)
        negateiftrue!(llN, Y)
        Yeppp.exp!(llN, llN)

        #copy!(llN2, llN)
        x1x!(llN)
        negateiffalse!(llN, Y)

        for i in 1:N
            groupindexi = groupindex[i]
            for j in 1:J
                @inbounds stheta[groupindexi, jcol, j] += llN[i] * X[i,j] #* Wim[groupindexi, jcol]
            end
        end
        # for i in 1:n
        #     for kcom in 1:C
        #         explogf[kcom] = exp(logpdf(m.components[kcom], gammaM[i, jcol]) - logpdf(m, gammaM[i, jcol]))
        #     end
        #
        #     for kcom in 1:C-1
        #         stheta[i, jcol, J+kcom] = explogf[kcom] - explogf[C]
        #     end
        #     for kcom in 1:C
        #         stheta[i, jcol, J+C-1+2*kcom-1] = wi[kcom]*H1(gammaM[i, jcol], mu[kcom], sigmas[kcom]) * explogf[kcom]
        #         stheta[i, jcol, J+C-1+2*kcom] = wi[kcom]*H2(gammaM[i, jcol], mu[kcom], sigmas[kcom]) * explogf[kcom]
        #     end
        # end

    end
end

function calibrate!(Wim::Matrix{Float64}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, β::Vector{Float64}, gammaM::Matrix{Float64}, stheta::Array{Float64, 3}, xb::Vector, llN::Vector, N::Int, J::Int, n::Int, C::Int, M::Int, p::Int;)

    # p = J+3*C-1
    completescore!(stheta, X, Y, groupindex, wi, mu, sigmas, β, gammaM, Wim, xb, llN, N, J, n, C, M)

    sbar = zeros(n, p)
    midmat = zeros(p, p)
    for i in 1:n, jcol in 1:M, j in 1:p
        @inbounds sbar[i, j] += Wim[i, jcol] * stheta[i, jcol, j]
    end
    for i in 1:n, jcol in 1:M
        for j in 1:p
            stheta[i, jcol, j] -= sbar[i, j]
        end
        BLAS.syr!('U', Wim[i, jcol], stheta[i, jcol,:][:], midmat)
    end

    for i in 2:p, j in 1:i-1
        midmat[i, j] = midmat[j, i]
    end
    coef =  inv(midmat) * sum(sbar, 1)[:]

    for i in 1:n, jcol in 1:M
        #Wim[i, jcol] -= Wim[i, jcol] * (dot(coef[:], stheta[i, jcol, :][:]))
         Wim[i, jcol] = log(Wim[i, jcol]) - dot(coef, stheta[i, jcol, :][:])
    end

    for i in 1:n
        u = maximum(Wim[i, :])
        for jcol in 1:M
            @inbounds Wim[i, jcol] = Wim[i, jcol] - u
        end
    end
    Yeppp.exp!(Wim, Wim)
    for i in 1:n
        u = sum(Wim[i, :])
        for jcol in 1:M
            @inbounds Wim[i, jcol] = Wim[i, jcol] / u
        end
    end

end

function FIupdateθ!(wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, gammaM::Matrix{Float64}, Wim::Matrix{Float64}, wipool::Vector, mupool::Vector, sigmaspool::Vector, tmp_p::Vector, tmp_mu::Vector, wi_divide_sigmas::Vector, inv_2sigmas_sq::Vector, sn::Vector, an::Real, tau::Real, wifixed::Bool, tol::Real, mu_lb::Vector, mu_ub::Vector, whichtosplit::Int, N::Int, J::Int, n::Int, C::Int, M::Int, thetamaxiteration::Int)

    for iter in 1:thetamaxiteration
        wi_old=copy(wi)
        mu_old=copy(mu)
        sigmas_old=copy(sigmas)
        fill!(wipool, 0.0)
        fill!(mupool, 0.0)
        fill!(sigmaspool, 0.0)
        for kcom in 1:C
            wi_divide_sigmas[kcom] = wi[kcom]/sigmas[kcom]
            inv_2sigmas_sq[kcom] = 0.5 / sigmas[kcom]^2
        end
        for i in 1:n, jcol in 1:M
            for j in 1:C
                @inbounds tmp_mu[j] = -(mu[j] - gammaM[i, jcol])^2 * inv_2sigmas_sq[j]
            end

            ratiosumexp!(tmp_mu, wi_divide_sigmas, tmp_p, C)
            for k in 1:C
                wipool[k] += Wim[i, jcol] * tmp_p[k]
                mupool[k] += gammaM[i, jcol] * Wim[i, jcol] * tmp_p[k]
                sigmaspool[k] += gammaM[i, jcol]^2 * Wim[i, jcol] * tmp_p[k]
            end
        end
        for kcom in 1:C
            wi[kcom] = wipool[kcom] / sum(wipool)
            mu[kcom] = mupool[kcom] / wipool[kcom]
            sigmas[kcom] = sqrt((sigmaspool[kcom] - wipool[kcom] * mu[kcom] ^ 2 + 2 * an * sn[kcom]^2) / (wipool[kcom] + 2 * an))
        end
        
        if wifixed
            wi_tmp = wi[whichtosplit]+wi[whichtosplit+1]
            wi[whichtosplit] = wi_tmp*tau
            wi[whichtosplit+1] = wi_tmp*(1-tau)
            Yeppp.max!(mu, mu, mu_lb)
            Yeppp.min!(mu, mu, mu_ub)
        end

        if stopRule(vcat(wi, mu, sigmas), vcat(wi_old, mu_old, sigmas_old), tol=tol)
            break
        end
    end
end

function updateβ!(β::Vector{Float64}, X::Matrix{Float64},
    Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,
    minStepFac::Real, betadevtol::Real,
    XWX::Matrix{Float64}, XWY::Vector{Float64},
    Xscratch::Matrix{Float64},
    gammaM::Matrix{Float64}, Wim::Matrix{Float64},
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
    gammaM::Matrix{Float64}, Wim::Matrix{Float64},
    llN::Vector{Float64},llN2::Vector{Float64},
    llN3::Vector{Float64}, xb::Vector{Float64},
    N::Int, J::Int, n::Int, M::Int)
    A_mul_B!(xb, X, β)
    fill!(XWX, 0.)
    fill!(XWY, 0.)

    for jcol in 1:M
        relocate!(llN, gammaM[:, jcol], groupindex, N)
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
    gammaM::Matrix{Float64}, Wim::Matrix{Float64},
    lln::Vector{Float64}, llN::Vector{Float64},
    llN2::Vector{Float64}, xb::Vector{Float64},
    N::Int, J::Int, n::Int, M::Int)
    dev = 0.
    A_mul_B!(xb, X, beta2)
    for jcol in 1:M
        relocate!(llN, gammaM[:, jcol], groupindex, N)
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

function FIupdateβ!(β::Vector{Float64}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,  gammaM::Matrix{Float64}, Wim::Matrix{Float64}, lln::Vector{Float64}, llN::Vector{Float64}, llN2::Vector{Float64}, xb::Vector{Float64}, N::Int, J::Int, n::Int, M::Int, Qmaxiteration::Int)

    opt = Opt(:LD_LBFGS, J)
    maxeval!(opt, Qmaxiteration)
    max_objective!(opt, (beta2, storage)->FI_Q1(beta2, storage, X, Y,  groupindex, gammaM, Wim, lln, llN, llN2, xb, N, J, n, M))

    optimize!(opt, β)
end

function FI_Q1(beta2::Array{Float64,1}, storage::Vector, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, gammaM::Matrix{Float64}, Wim::Matrix{Float64}, lln::Vector{Float64}, llN::Vector{Float64}, llN2::Vector{Float64}, xb::Vector{Float64}, N::Int, J::Int, n::Int, M::Int)

    if length(storage)>0
        fill!(storage, 0.0)
    end
    A_mul_B!(xb, X, beta2)
    res = 0.0
    for jcol in 1:M

        relocate!(llN, gammaM[:, jcol], groupindex, N)
        Yeppp.add!(llN, llN, xb)
        negateiftrue!(llN, Y)
        if length(storage) > 0
            copy!(llN2, llN)
            logistic!(llN2, llN2, N)
            negateiffalse!(llN2, Y)

            for i in 1:N
                groupindexi = groupindex[i]
                for j in 1:J
                    @inbounds storage[j] += llN2[i] * X[i,j] * Wim[groupindexi, jcol]
                end
            end
        end

        log1pexp!(llN, llN, N)
        fill!(lln, 0.0)
        for i in 1:N
            @inbounds lln[groupindex[i]] += llN[i]
        end
        res += wsum(lln, Wim[:, jcol])
    end
    -res
end

function latentgmmFI(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent::Int, β_init::Vector{Float64}, wi_init::Vector{Float64}, mu_init::Vector{Float64}, sigmas_init::Vector{Float64}; 
    M::Int = 2000, proposingdist::Distribution=TDist(3), meaninit::Real = 0.0, sigmasinit::Real = std(proposingdist), 
    maxiteration::Int=100, tol::Real=.005,
     sn::Vector{Float64}=sigmas_init, an::Float64=1.0/maximum(groupindex),
    debuginfo::Bool=false, Qmaxiteration::Int=2, whichtosplit::Int=1, tau::Real=.5, wifixed::Bool=false, dotest::Bool=true, thetamaxiteration::Int=10,
     mu_lb::Vector=fill(-Inf, ncomponent), mu_ub::Vector=fill(Inf, ncomponent),
     Wim::Matrix{Float64}=zeros(maximum(groupindex), M), lln::Vector{Float64}=zeros(maximum(groupindex)), llN::Vector{Float64}=zeros(length(Y)),
    llN2::Vector{Float64}=zeros(length(Y)),
    llN3::Vector{Float64}=zeros(length(Y)),
    Xscratch::Matrix{Float64}=copy(X),
    xb::Vector{Float64}=zeros(length(Y)),
     gammaM::Matrix{Float64}=zeros(maximum(groupindex), M), gammah::Matrix{Float64}=zeros(maximum(groupindex), M), epsilon::Float64=1e-5)

    # initialize theta
    length(wi_init) == length(mu_init) == length(sigmas_init) == ncomponent || error("The length of initial values should be $ncomponent")
    N,J=size(X)
    length(β_init) == J || error("Initial values of fixed efffect coefficients should have same dimension as X")
    n = maximum(groupindex)

    wi = copy(wi_init)
    mu = copy(mu_init)
    sigmas = copy(sigmas_init)
    β = copy(β_init)

    wi_old = ones(wi)./ncomponent
    mu_old = zeros(mu)
    sigmas_old = ones(sigmas)
    beta_old = randn(J)

    wipool = zeros(ncomponent)
    mupool = zeros(ncomponent)
    sigmaspool = zeros(ncomponent)
    tmp_p=ones(ncomponent) / ncomponent
    tmp_mu=zeros(ncomponent)
    wi_divide_sigmas = zeros(wi)
    inv_2sigmas_sq = ones(sigmas) .* 1e20
    #minit = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)
    #meaninit = mean(minit)
    #ratioinit = std(minit)/std(proposingdist)
    ratioinit = sigmasinit / std(proposingdist)
    for i in 1:n, jcol in 1:M
        gammatmp = rand(proposingdist)
        gammaM[i, jcol] = gammatmp * ratioinit + meaninit
        gammah[i, jcol] = log(ratioinit)-logpdf(proposingdist, gammatmp)
    end
    XWX = zeros(J, J)
    XWY = zeros(J)
    ll=0.0
    ll0=-Inf
    for iter_em in 1:maxiteration

        copy!(wi_old, wi)
        copy!(mu_old, mu)
        copy!(sigmas_old, sigmas)

        A_mul_B!(xb, X, β)
        ll=FIintegralweight!(Wim, X, Y, groupindex, gammaM, gammah, wi, mu, sigmas, llN, xb, proposingdist, N, J, n, M)
        if dotest
            lldiff = ll - ll0
            if debuginfo
                println("lldiff=", lldiff)
            end
            if (lldiff < epsilon) && (iter_em > 3)
                break
            end 
            ll0 = ll
        end  
        if debuginfo
            println("At $(iter_em)th iteration:")
        end
        if !stopRule(β, beta_old, tol=tol/10)
            copy!(beta_old, β)
            FIupdateβ!(β, X, Y, groupindex, .001, .001, XWX, XWY, Xscratch, gammaM, Wim, lln, llN, llN2, llN3, xb, N, J, n, M, Qmaxiteration)
            if debuginfo
                println("beta=", β)
            end
        end
        FIupdateθ!(wi, mu, sigmas, X, Y, groupindex, gammaM, Wim, wipool, mupool, sigmaspool, tmp_p, tmp_mu, wi_divide_sigmas, inv_2sigmas_sq, sn, an, tau, wifixed, tol, mu_lb, mu_ub, whichtosplit, N, J, n, ncomponent, M, thetamaxiteration)

        if debuginfo
            println("wi=$wi")
            println("mu=$mu")
            println("sigma=$sigmas")
            #println("ll=",marginallikelihood(β, X, Y, groupindex, n, wi, mu, sigmas, ghx, ghw, llN, lln, xb, sumlogmat))
        end
        if !dotest
            if stopRule(vcat(β, wi, mu, sigmas), vcat(beta_old, wi_old, mu_old, sigmas_old), tol=tol) && (iter_em > 3)
                if debuginfo
                    println("latentgmmFI converged at $(iter_em)th iteration")
                end
                break
            end
        end
        if (iter_em == maxiteration) && (maxiteration > 15)
            warn("latentgmmFI not converge!")
        end
    end
    return(wi, mu, sigmas, β, ll)
end

function loglikelihoodratioFI_ctau(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent1::Int,  betas0::Vector{Float64}, wi_C1::Vector{Float64},  whichtosplit::Int64, tau::Float64, mu_lb::Vector{Float64}, mu_ub::Vector{Float64}, sigmas_lb::Vector{Float64}, sigmas_ub::Vector{Float64}; ntrials::Int=25, M::Int=100, sn::Vector{Float64}=sigmas_ub ./ 2, an=.25, debuginfo::Bool=false, gammaM::Matrix = zeros(maximum(groupindex), M), Wim::Matrix = zeros(maximum(groupindex), M), llN::Vector=zeros(length(Y)), llN2::Vector = zeros(length(Y)), xb::Vector=zeros(length(Y)), proposingdist=modelC0)

    nF = maximum(groupindex)
    tau = min(tau, 1-tau)
    #ghx, ghw = hermite(ngh)

    wi = repmat(wi_C1, 1, 4*ntrials)
    mu = zeros(ncomponent1, 4*ntrials)
    sigmas = ones(ncomponent1, 4*ntrials)
    betas = repmat(betas0, 1, 4*ntrials)
    ml = -Inf .* ones(4*ntrials)
    for i in 1:4*ntrials
        mu[:, i] = rand(ncomponent1) .* (mu_ub .- mu_lb) .+ mu_lb
        sigmas[:, i] = rand(ncomponent1) .* (sigmas_ub .- sigmas_lb) .+ sigmas_lb

        wi[:, i], mu[:, i], sigmas[:, i], betas[:, i], ml[i]= latentgmmFI(X, Y, groupindex, ncomponent1, betas0, wi[:, i], mu[:, i], sigmas[:, i], whichtosplit=whichtosplit, tau=tau, mu_lb=mu_lb, mu_ub=mu_ub, maxiteration=100, sn=sn, an=an, gammaM = gammaM, Wim=Wim, llN=llN, llN2=llN2, xb=xb, Qmaxiteration=2, wifixed=true, M=M, proposingdist=proposingdist, epsilon=0.01)
    end

    mlperm = sortperm(ml)
    for j in 1:ntrials
        i = mlperm[4*ntrials+1 - j] # start from largest ml
        wi[:, i], mu[:, i], sigmas[:, i], betas[:, i], ml[i] = latentgmmFI(X, Y, groupindex, ncomponent1, betas[:, i], wi[:, i], mu[:, i], sigmas[:, i], whichtosplit=whichtosplit, tau=tau, mu_lb=mu_lb,mu_ub=mu_ub, maxiteration=500, sn=sn, an=an, debuginfo=debuginfo, gammaM = gammaM, Wim=Wim, llN=llN, llN2=llN2, xb=xb, Qmaxiteration=2, wifixed=true, M=M, proposingdist=proposingdist)
    end

    mlmax, imax = findmax(ml[mlperm[(3*ntrials+1):4*ntrials]])
    imax = mlperm[3*ntrials+imax]
    modelC1=MixtureModel(map((u, v) -> Normal(u, v), mu[:, imax], sigmas[:, imax]), wi[:, imax])
    wi, mu, sigmas, betas, mlctau=latentgmmFI(X, Y, groupindex, ncomponent1, betas[:, imax], wi[:, imax], mu[:, imax], sigmas[:, imax], maxiteration=3, an=an, sn=sn, debuginfo=debuginfo, M = M, proposingdist=modelC1)
    return mlctau
end

function loglikelihoodratioFI(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent1::Int; vtau::Vector{Float64}=[.5,.3,.1;], ntrials::Int=25, M::Int=100, debuginfo::Bool=false, ctauparallel=true)
    C0 = ncomponent1 - 1
    C1 = ncomponent1
    nF = maximum(groupindex)
    #M = ngh * ncomponent1
    an1 = 1/nF
    wi_init, mu_init, sigmas_init, betas_init, ml_C0= latentgmmFI(X, Y, groupindex, 1, betas_init, [1.0], [mean(gamma_init)], [std(gamma_init)], maxiteration=100, an=an1, sn=std(gamma_init).*ones(C0), M=M, meaninit=mean(gamma_init), sigmasinit=std(gamma_init))
    gamma_init = predictgamma(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init)
    wi_init, mu_init, sigmas_init, ml_tmp = gmm(gamma_init, C0, ones(C0)/C0, quantile(gamma_init, linspace(0, 1, C0+2)[2:end-1]), ones(C0), an=an1)
    wi_init, mu_init, sigmas_init, betas_init, ml_C0= latentgmmFI(X, Y, groupindex, C0, betas_init, wi_init, mu_init, sigmas_init, maxiteration=500, an=an1, sn=std(gamma_init).*ones(C0), M=M, meaninit=mean(gamma_init), sigmasinit=std(gamma_init))
    if C0 > 1
        trand=LatentGaussianMixtureModel.asymptoticdistribution(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init)
    end

    modelC0 = MixtureModel(map((u, v) -> Normal(u, v), mu_init, sigmas_init), wi_init)
    mingamma = minimum(gamma_init)
    maxgamma = maximum(gamma_init)

    or = sortperm(mu_init)
    wi0 = wi_init[or]
    mu0 = mu_init[or]
    sigmas0 = sigmas_init[or]
    betas0 = betas_init
    an = decidepenalty(wi0, mu0, sigmas0, nF)

    N,J=size(X)
    gammaM = zeros(nF, M)
    Wim = zeros(nF, M)
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

            ml_tmp=loglikelihoodratioFI_ctau(X, Y, groupindex, ncomponent1, betas0, wi_C1, whichtosplit, vtau[i], mu_lb, mu_ub, sigmas_lb, sigmas_ub, ntrials=ntrials, sn=sigmas0[ind], an=an, debuginfo=debuginfo, gammaM = gammaM, Wim=Wim, llN=llN, llN2=llN2, xb=xb, proposingdist=modelC0, M=M)
            if debuginfo
                println(whichtosplit, " ", vtau[i], "->", ml_tmp)
            end
            ml_tmp
        end
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

             lr[i, whichtosplit]=loglikelihoodratioFI_ctau(X, Y, groupindex, ncomponent1, betas0, wi_C1, whichtosplit, vtau[i], mu_lb, mu_ub, sigmas_lb, sigmas_ub, ntrials=ntrials, sn=sigmas0[ind], an=an, debuginfo=debuginfo, gammaM = gammaM, Wim=Wim, llN=llN, llN2=llN2, xb=xb, proposingdist=modelC0, M=M)
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
