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
function gmm(x::Vector{Float64}, ncomponent::Int, wi_init::Vector{Float64}, mu_init::Vector{Float64}, sigmas_init::Vector{Float64}; whichtosplit::Int64=1, tau::Float64=.5, mu_lb::Vector{Float64}=-Inf.*ones(wi_init), mu_ub::Vector{Float64}=Inf.*ones(wi_init), an::Float64=0.25, sn::Vector{Float64}=ones(wi_init).*std(x), maxiter::Int64=10000, tol=.001, wifixed=false)

    if ncomponent == 1
        mu = [mean(x)]
        sigmas = [std(x)]
        ml = sum(logpdf(Normal(mean(x), std(x)), x)) + sum(pn(sigmas, sn, an=an))
        return([1.0], mu, sigmas, ml)
    end
    nF = length(x)
    #ncomponent = length(wi_init)
    tau = min(tau, 1-tau)
    # sn = var(x)
    wi = copy(wi_init)
    mu = copy(mu_init)
    sigmas = copy(sigmas_init)
    if wifixed
        wi_tmp = wi[whichtosplit]+wi[whichtosplit+1]
        wi[whichtosplit] = wi_tmp*tau
        wi[whichtosplit+1] = wi_tmp*(1-tau)
        mu = min(max(mu, mu_lb), mu_ub)
    end

    pwi = ones(nF, ncomponent) ./ ncomponent
    for iter_em in 1:maxiter
        for i in 1:nF
            # pwi[i, :] = ratiosumexp(-(mu .- x[i]).^2 ./ (2 .* sigmas .^ 2), wi ./ sigmas)

            ratiosumexp!(-(mu .- x[i]).^2 ./ (2 .* sigmas .^ 2), wi ./ sigmas, pwi, i, ncomponent)
        end

        wi_old=copy(wi)
        mu_old=copy(mu)
        sigmas_old=copy(sigmas)

        for j in 1:ncomponent
            colsum = sum(pwi[:, j])
            wi[j] = colsum / nF
            mu[j] = wsum(pwi[:,j] ./ colsum, x)
            sigmas[j] = sqrt((wsum(pwi[:,j], (x .- mu[j]).^2) + 2 * an * sn[j]^2) / (sum(pwi[:,j]) + 2*an))
        end

        if wifixed
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

    ml = sum(logpdf(m, x)) + sum(pn(sigmas, sn, an=an)) #+ log(1 - abs(1 - 2*tau))
    return (wi, mu, sigmas, ml)
end


#for maxposterior
function mpe_goalfun(input::Vector{Float64}, storage::Vector, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, nF::Int, llvec::Vector{Float64},llvecnew::Vector{Float64})

    N,J=size(X)
    mygamma= input[1:nF]
    mybeta = input[(nF+1):(nF+J)]
    mytheta = input[nF+J+1]
    llvec[:] = X*mybeta

    Yeppp.add!(llvec, mygamma[groupindex], llvec)
    # map!(Exp_xy(), llvec, llvec, Y)
    negateiftrue!(llvec, Y)
    Yeppp.exp!(llvec, llvec)

    if length(storage)>0
        fill!(storage, 0.0)
        llvecnew[:] = llvec
        # map!(Xy1x(), llvecnew, llvecnew, Y)
        x1x!(llvecnew)
        negateiffalse!(llvecnew, Y)

        for i in 1:nF
            storage[i] =  - mygamma[i]/mytheta
        end
        for i in 1:N
            storage[groupindex[i]] += llvecnew[i]
        end
        for irow in 1:N
            for j in 1:J
                storage[j+nF] += llvecnew[irow] * X[irow,j]
            end
        end
        storage[nF+J+1] = sumabs2(mygamma) / (mytheta * mytheta) /2 - nF/mytheta/2

        for i in 1:length(input)
            storage[i] = storage[i] / N
        end
    end
    # map!(Log1p(), llvec, llvec)
    log1p!(llvec)

    -mean(llvec) - sumabs2(mygamma)/N/mytheta/2 - log(mytheta) * nF / 2/N
end
function maxposterior(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector)

    N,J = size(X)
    nF = length(unique(groupindex))

    bag = [rand(Normal(), nF), ones(J+1);]
    p=1+J+nF
    opt_init = Opt(:LD_LBFGS, p)
    lower_bounds!(opt_init, [-Inf .* ones(nF+J), 0.0;])
    llvec = zeros(N)
    llvecnew = zeros(N)
    max_objective!(opt_init, (input, storage)->mpe_goalfun(input, storage, X, Y, groupindex, nF, llvec, llvecnew))
    (minf,bag,ret) = optimize(opt_init, bag)
    (bag[1:nF], bag[(1+nF):((nF+J))], bag[nF+J+1])
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
    #ll_nF=zeros(nF)
    # sumlog_nF = zeros(nF)
    #llvec = zeros(N)
    # sumlogmat = zeros(nF, M*C)
    fill!(sumlogmat, 0.0)
    for jcom in 1:C
        for ix in 1:M
            # fill!(llvec, ghx[ix]*sigmas[jcom]*sqrt(2)+mu[jcom])
            # add!(llvec, llvec, xb)
            # negateiftrue!(llvec, Y)
            #partially devecterize, speed up by 5%; But exp and log from Yeppp are much faster than devecterized ones.
            xtmp = ghx[ix]*sigmas[jcom]*sqrt(2)+mu[jcom]
            for i in 1:N
                @inbounds llvec[i] = ifelse(Y[i], -xtmp - xb[i], xtmp + xb[i])
            end
            Yeppp.exp!(llvec, llvec)
            log1p!(llvec)
            ixM = ix+M*(jcom-1)
            for i in 1:N
                @inbounds sumlogmat[groupindex[i], ixM] -= llvec[i]
            end
            for i in 1:nF
                sumlogmat[i, ixM] += log(wi[jcom]) + log(ghw[ix])
            end
        end

    end
    for i in 1:nF
        ll_nF[i] = logsumexp(sumlogmat[i,:])
    end

    sum(ll_nF) - nF*log(pi)/2
end

function asymptoticdistribution(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, betas::Array{Float64,1}; ngh::Int=1000, nrep::Int=10000)

    N,J = size(X)
    nF = maximum(groupindex)
    M = ngh
    C = length(wi)
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

            Yeppp.exp!(llvec, llvec)
            copy!(llN2, llvec)
            x1x!(llN2)
            negateiffalse!(llN2, Y)
            for i in 1:N
                for j in 1:J
                    summat_beta[groupindex[i], ixM, j] += llN2[i] * X[i,j]
                    # ifelse(Y[i], exp(-llvec[i])*X[i, j], -exp(-llvec[i])*X[i, j])
                end
            end
            log1p!(llvec)

            for i in 1:N
                @inbounds sumlogmat[groupindex[i], ixM] -= llvec[i]
            end
            for i in 1:nF
                sumlogmat[i, ixM] +=  log(ghw[ix]) # +log(wi[jcom]) + H1(xtmp, sigmas[jcom])

            end
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
    I_η = S_η'*S_η./nF
    I_λη = S_λ'*S_η./nF
    I_λ = S_λ'*S_λ./nF
    I_all = vcat(hcat(I_η, I_λη'), hcat(I_λη, I_λ))
    D, V = eig(I_all)
    tol2 = abs(D[1]) * 1e-14
    D[abs(D).<tol2] = tol2
    I_all = V*diagm(D)*V'

    I_λ_η = I_all[(J+3*C):(J+5*C-1), (J+3*C):(J+5*C-1)] - I_all[(J+3*C):(J+5*C-1), 1:(J+3*C-1)] * inv(I_all[1:(J+3*C-1), 1:(J+3*C-1)]) * I_all[1:(J+3*C-1),(J+3*C):(J+5*C-1) ]
    D, V = eig(I_λ_η)
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
