#The utility functions
#Lanfeng Pan
#Oct 29, 2014

function pn(sigma1::Float64,  sigmahat::Float64; an::Float64=0.25)
    -((sigmahat / sigma1)^2 + 2*log(sigma1 / sigmahat) -1) * an
end
pn(sigma1::Vector{Float64},  sigmahat::Float64; an::Float64 = .25)=Float64[pn(sigma1[i], sigmahat, an=an) for i in 1:length(sigma1)]
pn(sigma1::Vector{Float64},  sigmahat::Vector{Float64}; an::Float64 = .25)=Float64[pn(sigma1[i], sigmahat[i], an=an) for i in 1:length(sigma1)]

function decidepenalty(wi0::Vector, mu0::Vector, sigmas0::Vector, nobs::Int)
    C = length(wi0)
    or = sortperm(mu0)
    wi = wi0[or]
    mu = mu0[or]
    sigmas = sigmas0[or]
    if C==1
        return 0.25
    elseif C == 2
        omega = omega12(wi, mu, sigmas)
        omega = min(max(omega, 1e-16), 1 - 1e-16)
        x = exp(-1.642 -0.434*log(omega/(1-omega)) -101.80/nobs)
        return 1.8*x/(1+x)
    elseif C == 3
        omega = omega123(wi, mu, sigmas)
        omega = min(max(omega, 1e-16), 1 - 1e-16)
        t_omega = (omega[1]*omega[2])/(1-omega[1])/(1-omega[2])
        x =  exp(-1.678 -0.232*log(t_omega) -175.50/nobs)
        return 1.5*x/(1+x)
    else
        return 1.0
    end
end
function omegaji(alpi,mui,sigi,alpj,muj,sigj)
# Computes omega_{j|i} defined in (2.1) of Maitra and Melnykov
    if sigi==sigj
        delta = abs(mui-muj)/sigi
        out = pdf(Normal(), -delta/2 + log(alpj/alpi)/delta)
    else
        ncp = (mui-muj)*sigi/(sigi^2-sigj^2)
        value=sigj^2*(mui-muj)^2/(sigj^2-sigi^2)^2-sigj^2/(sigi^2-sigj^2)*log(alpi^2*sigj^2/alpj^2/sigi^2 )
        sqrtvalue = sqrt(max(value,0.0))

        ind = float(sigi<sigj)
        out = ind + (-1)^ind*(pdf(Normal(), sqrtvalue-ncp)-pdf(Normal(), -sqrtvalue-ncp))
    end
    return(out)
end	# end function omega.ji

function omega12(wi, mu, sigmas)
# Computes omega_{12} for testing H_0:m=2 against H_1:m=3
    alp1 = wi[1]
    alp2 = wi[2]

    mu1 = mu[1]
    mu2 = mu[2]

    sig1 = sigmas[1]
    sig2 = sigmas[2]

    part1 = omegaji(alp1,mu1,sig1,alp2,mu2,sig2)
    part2 = omegaji(alp2,mu2,sig2,alp1,mu1,sig1)

    return((part1+part2)/2)
end	# end function omega.12

function omega123(wi, mu, sigmas)

    alp1 = wi[1]
    alp2 = wi[2]
    alp3 = wi[3]

    mu1 = mu[1]
    mu2 = mu[2]
    mu3 = mu[3]

    sig1 = sigmas[1]
    sig2 = sigmas[2]
    sig3 = sigmas[3]

    part1 = omegaji(alp1,mu1,sig1,alp2,mu2,sig2)
    part2 = omegaji(alp2,mu2,sig2,alp1,mu1,sig1)
    w12 = (part1+part2)/2

    part3 = omegaji(alp2,mu2,sig2,alp3,mu3,sig3)
    part4 = omegaji(alp3,mu3,sig3,alp2,mu2,sig2)
    w23 = (part3+part4)/2

    return([w12,w23])

end	# end function omega.123

function stopRule(pa::Vector, pa_old::Vector; tol=.005)
    maximum(abs(pa .- pa_old)./(abs(pa).+.001)) < tol
end


#Estimate gaussian mixture parameters given the initial value of γ
function gmm(x::RealVector{Float64}, ncomponent::Int, 
    wi_init::Vector{Float64}=ones(ncomponent)/ncomponent,
     mu_init::Vector{Float64}=quantile(x, linspace(0, 1, ncomponent+2)[2:end-1]), 
     sigmas_init::Vector{Float64}=ones(ncomponent).*std(x);
      whichtosplit::Int64=1, tau::Float64=.5,
       mu_lb::Vector{Float64}=-Inf.*ones(wi_init),
        mu_ub::Vector{Float64}=Inf.*ones(wi_init), 
        an::Float64=1/length(x), sn::Vector{Float64}=ones(ncomponent).*std(x),
         maxiteration::Int64=10000, tol::Real=.001, taufixed::Bool=false, pl::Bool=true, ptau::Bool=false)

    if ncomponent == 1
        mu = [mean(x)]
        sigmas = [std(x)]
        ml = loglikelihood(Normal(mean(x), std(x)), x) 
        if pl
            ml += sum(pn(sigmas, sn, an=an)) 
        end
        return([1.0], mu, sigmas, ml)
    end
    n = length(x)
    tau = min(tau, 1-tau)
    wi = copy(wi_init)
    mu = copy(mu_init)
    sigmas = copy(sigmas_init)
    wi_old = copy(wi)
    mu_old = copy(mu)
    sigmas_old=copy(sigmas)

    wi_divide_sigmas = zeros(wi)
    inv_2sigmas_sq = ones(sigmas)
    pwi = ones(n, ncomponent)
    xtmp = copy(x)
    
    for iter_em in 1:maxiteration

        @inbounds for j in 1:length(wi)
            wi_divide_sigmas[j] = wi[j]/sigmas[j]
            inv_2sigmas_sq[j] = 0.5 / sigmas[j]^2
        end
        
        for i in 1:n
            tmp = -Inf
            @inbounds for j in 1:ncomponent
                pwi[i, j] = -(mu[j] - x[i])^2 * inv_2sigmas_sq[j]
                if tmp < pwi[i,j]
                    tmp = pwi[i,j]
                end
            end
            #@inbounds tmp = maximum(pwi[i, :])
            for j in 1:ncomponent
                @inbounds pwi[i, j] -= tmp
            end
        end
        Yeppp.exp!(pwi, pwi)

        @inbounds for i in 1:n
            tmp = 0.0
            for j in 1:ncomponent
                pwi[i, j] *= wi_divide_sigmas[j]
                tmp += pwi[i, j]
            end
            for j in 1:ncomponent
                pwi[i, j] /= tmp
            end
        end

        copy!(wi_old, wi)
        copy!(mu_old, mu)
        copy!(sigmas_old, sigmas)

        for j in 1:ncomponent
            colsum = sum(pwi[:, j])
            if colsum == 0.
                wi[j] = 1/n
                sigmas[j] *=2
                continue
                warn("Zero point component found. Auto increase its variance by a factor 2.")
            end
            wi[j] = colsum / n
            mu[j] = wsum(pwi[:,j], x) / colsum
            
            add!(xtmp, x, -mu[j], n)
            sqr!(xtmp, xtmp, n)
            sigmas[j] = sqrt((wsum(pwi[:,j], xtmp) + 2 * an * sn[j]^2) / (colsum + 2*an))
        end
        if any(isnan(wi))|| any(isnan(mu)) || any(isnan(sigmas))
            println( wi, mu, sigmas)
            error("NaN occur!")
        end
        if taufixed
            wi_tmp = wi[whichtosplit]+wi[whichtosplit+1]
            wi[whichtosplit] = wi_tmp*tau
            wi[whichtosplit+1] = wi_tmp*(1-tau)
            mu = min(max(mu, mu_lb), mu_ub)
        end

        if stopRule(vcat(wi, mu, sigmas), vcat(wi_old, mu_old, sigmas_old), tol=tol)
            break
        end
    end
    m = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)

    ml = loglikelihood(m, x)# + sum(pn(sigmas, sn, an=an)) #+ log(1 - abs(1 - 2*tau))
    if pl
        ml += sum(pn(sigmas, sn, an=an)) 
    end
    if ptau
        tau2 = wi[whichtosplit] / (wi[whichtosplit]+wi[whichtosplit+1])
        ml += log(1 - abs(1 - 2*tau2))
    end
    return (wi, mu, sigmas, ml)
end

function marginallikelihood(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, β::Array{Float64,1}; ngh::Int=100)
    ghx, ghw = gausshermite(ngh)
    N, J = size(X)
    M = ngh * length(wi)
    n = maximum(groupindex)
    marginallikelihood(β, X, Y, groupindex, n, wi, mu, sigmas, ghx, ghw, zeros(N), zeros(n), zeros(N), zeros(n, M))
end

function marginallikelihood(beta_new::Array{Float64,1}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, nF::Int64, wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, ghx::Vector{Float64}, ghw::Vector{Float64}, llvec::Vector, ll_nF::Vector, xb::Vector, sumlogmat::Matrix)

    N,J = size(X)
    M = length(ghx)
    C = length(wi)
    # xb = X*beta_new
    A_mul_B!(xb, X, beta_new)
    ll = 0.0
    #ll_nF=zeros(nF)
    # sumlog_nF = zeros(nF)
    #llvec = zeros(N)
    # sumlogmat = zeros(nF, M*C)
    #fill!(sumlogmat, 0.0)
    for jcom in 1:C
        for ix in 1:M
            # fill!(llvec, ghx[ix]*sigmas[jcom]*sqrt(2)+mu[jcom])
            # add!(llvec, llvec, xb)
            # negateiftrue!(llvec, Y)
            #partially devecterize, speed up by 5%; But exp and log from Yeppp are much faster than devecterized ones.
            xtmp = ghx[ix]*sigmas[jcom]*sqrt(2)+mu[jcom]
            wtmp = log(ghw[ix])+log(wi[jcom])
            for i in 1:N
                @inbounds llvec[i] = ifelse(Y[i], -xtmp - xb[i], xtmp + xb[i])
            end
            # Yeppp.exp!(llvec, llvec)
            # log1p!(llvec)
            log1pexp!(llvec, llvec, N)
            ixM = ix+M*(jcom-1)
            for i in 1:nF
                sumlogmat[i, ixM] = wtmp
            end
            for i in 1:N
                @inbounds sumlogmat[groupindex[i], ixM] -= llvec[i]
            end
        end
    end
    for i in 1:nF
        ll += logsumexp(sumlogmat[i,:])
    end

    ll - nF*log(pi)/2
end

function predictgamma(X::Matrix, Y::Vector{Bool}, groupindex::IntegerVector, wi::Vector, mu::Vector, sigmas::Vector, β::Vector; ngh::Int=100)

    ncomponent = length(wi)
    n = maximum(groupindex)
    M = ngh*ncomponent
    N, J = size(X)

    xb = zeros(N)
    llN = zeros(N)
    llN2 = zeros(N)
    gammaM = zeros(M)
    gammahat = zeros(n)
    
    ghx, ghw = gausshermite(ngh)
    Wim = zeros(n, M)
    for ix in 1:ngh, jcom in 1:ncomponent
        ixM = ix+ngh*(jcom-1)
        gammaM[ixM] = ghx[ix]*sigmas[jcom]*sqrt(2)+mu[jcom]
    end

    A_mul_B!(xb, X, β)
    integralweight!(Wim, X, Y, groupindex, gammaM, wi, ghw, llN, llN2, xb, N, J, n, ncomponent, ngh)
    for i in 1:n
        for j in 1:M
            gammahat[i] += gammaM[j] * Wim[i,j]
        end
    end

    return gammahat
end
function FDR(X::Matrix, Y::Vector{Bool}, groupindex::IntegerVector, wi::Vector, mu::Vector, sigmas::Vector, β::Vector, C0::IntegerVector; ngh::Int=100)

    ncomponent = length(wi)
    n = maximum(groupindex)
    M = ngh*ncomponent
    N, J = size(X)
    if maximum(C0) > ncomponent || minimum(C0) < 1
        error("C0 must be within 1 to number of components")
    end

    xb = zeros(N)
    llN = zeros(N)
    llN2 = zeros(N)
    gammaM = zeros(M)
    piposterior = zeros(n, ncomponent)
    clFDR = zeros(n)
    
    ghx, ghw = gausshermite(ngh)
    Wim = zeros(n, M)
    for ix in 1:ngh, jcom in 1:ncomponent
        ixM = ix+ngh*(jcom-1)
        gammaM[ixM] = ghx[ix]*sigmas[jcom]*sqrt(2)+mu[jcom]
    end

    A_mul_B!(xb, X, β)
    integralweight!(Wim, X, Y, groupindex, gammaM, wi, ghw, llN, llN2, xb, N, J, n, ncomponent, ngh)
    for i in 1:n
        for jcom in 1:C
            for ix in 1:ngh
                ixM = ix+ngh*(jcom-1)
                piposterior[i, jcom] += Wim[i,ixM]
            end
        end
    end
    for i in 1:n
        for jcom in 1:C
            clFDR[i] = sum(piposterior[i, C0])
        end
    end

    return clFDR
end
function asymptoticdistribution(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, betas::Array{Float64,1}; ngh::Int=100, nrep::Int=10000, debuginfo::Bool=false)

    N,J = size(X)
    nF = maximum(groupindex)
    M = ngh
    C = length(wi)
    if C == 1
        return rand(Chisq(2), nrep)
    end
    ghx, ghw = gausshermite(ngh)
    xb = zeros(N)
    A_mul_B!(xb, X, betas)
    llvec = zeros(N)
    llN2 = zeros(N)
    ll_nF = zeros(nF, C)
    sumlogmat = zeros(nF, ngh*C)
    summat_beta = zeros(nF, ngh*C, J)
    S_β = zeros(nF, J)
    S_π = zeros(nF, C-1)
    S_μσ = zeros(nF, 2*C)
    S_λ = zeros(nF, 2*C)
    ml = zeros(nF)
    xtmp = zeros(C*M)
    for jcom in 1:C
        for ix in 1:M
            ixM = ix+M*(jcom-1)
            xtmp[ixM] = ghx[ix]*sigmas[jcom]*sqrt(2)+mu[jcom]
            for i in 1:N
                @inbounds llvec[i] = ifelse(Y[i], -xtmp[ixM] - xb[i], xtmp[ixM] + xb[i])
            end

            # Yeppp.exp!(llvec, llvec)
            copy!(llN2, llvec)
            logistic!(llN2)
            negateiffalse!(llN2, Y)
            for i in 1:N
                for j in 1:J
                    summat_beta[groupindex[i], ixM, j] += llN2[i] * X[i,j]
                end
            end
            log1pexp!(llvec)

            for i in 1:N
                @inbounds sumlogmat[groupindex[i], ixM] -= llvec[i]
            end
            for i in 1:nF
                sumlogmat[i, ixM] +=  log(ghw[ix])
            end
        end
    end
    for i in 1:nF
        u = maximum(sumlogmat[i, :])
        for jcol in 1:C*M
            @inbounds sumlogmat[i, jcol] = sumlogmat[i, jcol] - u
        end
    end
    for i in 1:nF
        for kcom in 1:C
            ll_nF[i, kcom] = sumexp(sumlogmat[i,(1+M*(kcom-1)):M*kcom])
        end
    end
    for i in 1:nF
        for jcom in 1:C
            for ix in 1:M
                sumlogmat[i, ix+M*(jcom-1)] += log(wi[jcom])
            end
        end
        ml[i]=sumexp(sumlogmat[i, :])
    end
    for kcom in 1:(C-1)
        S_π[:, kcom] = (ll_nF[:, kcom] .- ll_nF[:, C]) ./ ml
    end
    for i in 1:nF
        for kcom in 1:C
            ind = (1+M*(kcom-1)):M*kcom
            # w[i, kcom] = ll_nF[i, kcom] * wi[kcom] / ml[i]

            S_μσ[i, 2*kcom-1] = sumexp(sumlogmat[i, ind], H1(xtmp[ind], mu[kcom], sigmas[kcom])) / ml[i]
            S_μσ[i, 2*kcom] = sumexp(sumlogmat[i, ind], H2(xtmp[ind], mu[kcom], sigmas[kcom]))/ml[i]
            S_λ[i, 2*kcom-1] = sumexp(sumlogmat[i, ind], H3(xtmp[ind], mu[kcom], sigmas[kcom]))/ml[i]
            S_λ[i, 2*kcom] = sumexp(sumlogmat[i, ind], H4(xtmp[ind], mu[kcom], sigmas[kcom]))/ml[i]
        end
        for j in 1:J
            S_β[i, j] = sumexp(sumlogmat[i,:], summat_beta[i, :, j])/ml[i]
        end
    end
    S_η = hcat(S_β, S_π, S_μσ)
    debuginfo && println(round(sum(S_η, 1)./sqrt(nF), 6))
    I_η = S_η'*S_η./nF
    I_λη = S_λ'*S_η./nF
    I_λ = S_λ'*S_λ./nF
    I_all = vcat(hcat(I_η, I_λη'), hcat(I_λη, I_λ))
    if 1/cond(I_all) < eps(Float64)
        D, V = eig(I_all)
        debuginfo && println(D)
        tol2 = maximum(abs(D)) * 1e-14
        D[D.<tol2] = tol2
        I_all = V*diagm(D)*V'
    end
    debuginfo && println(round(I_all, 6))
    I_λ_η = I_all[(J+3*C):(J+5*C-1), (J+3*C):(J+5*C-1)] - I_all[(J+3*C):(J+5*C-1), 1:(J+3*C-1)] * inv(I_all[1:(J+3*C-1), 1:(J+3*C-1)]) * I_all[1:(J+3*C-1),(J+3*C):(J+5*C-1) ]
    debuginfo && println(round(I_λ_η, 6))
    D, V = eig(I_λ_η)
    D[D.<0.] = 0.
    I_λ_η2 = V * diagm(sqrt(D)) * V'
    u = randn(nrep, 2*C) * I_λ_η2
    EM = zeros(nrep, C)
    T = zeros(nrep)
    for kcom in 1:C
        EM[:, kcom] = sum(u[:, (2*kcom-1):(2*kcom)] * inv(I_λ_η[(2*kcom-1):(2*kcom), (2*kcom-1):(2*kcom)]) .* u[:, (2*kcom-1):(2*kcom)], 2)
    end
    for i in 1:nrep
        T[i] = maximum(EM[i, :])
    end
    T
    #sum(ll_nF) - nF*log(pi)/2
end

####End of utility functions
