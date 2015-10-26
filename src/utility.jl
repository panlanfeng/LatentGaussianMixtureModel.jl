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

#accept prob for γᵢ = ΠΠ(e(ηᵒy)+1)/(e(ηy)+1) #* exp(((γold - mu)²-(γnew-mu)²)/2σ²)exp((γᵒ-γⁿ)²/2gsd²)
function q_gamma(sample_gamma_new::Array{Float64,1}, sample_gamma::Array{Float64,1}, xb::Array{Float64,1}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, mu::Vector{Float64}, sigmas::Vector{Float64}, L::IntegerVector, L_new::IntegerVector, llvec::Vector{Float64}, llvecnew::Vector{Float64},ll_nF::Vector{Float64}, nF::Int, N::Int)

    # llvec[:] = xb .+ sample_gamma[groupindex]
    relocate!(llvec, sample_gamma, groupindex, N)
    Yeppp.add!(llvec, llvec, xb)
    # llvecnew[:] = xb .+ sample_gamma_new[groupindex]
    relocate!(llvecnew, sample_gamma_new, groupindex, N)
    Yeppp.add!(llvecnew, llvecnew, xb)

    # map!(RcpLogistic(), llvec, llvec, Y)
    # map!(RcpLogistic(), llvecnew, llvecnew, Y)
    # rcplogistic!(llvec, Y)
    # rcplogistic!(llvecnew, Y)
    # divide!(llvec, llvecnew, N)
    loglogistic!(llvec, Y, N)
    loglogistic!(llvecnew, Y, N)
    negate!(llvec, llvec, N)
    Yeppp.add!(llvec, llvec, llvecnew)

    # for i in 1:nF
    #     ll_nF[i] = prod(llvec[coll_nF[i]]) * pdf(Normal(mu[L_new[i]], sigmas[L_new[i]]), sample_gamma_new[i])/ pdf(Normal(mu[L[i]], sigmas[L[i]]), sample_gamma[i])
    # end
    for i in 1:nF
        # @inbounds ll_nF[i] = pdf(Normal(mu[L_new[i]], sigmas[L_new[i]]), sample_gamma_new[i])/ pdf(Normal(mu[L[i]], sigmas[L[i]]), sample_gamma[i])
        @inbounds ll_nF[i] = -(sample_gamma_new[i] - mu[L_new[i]])^2/sigmas[L_new[i]]^2*0.5 - log(sigmas[L_new[i]]) + (sample_gamma[i] - mu[L[i]])^2/sigmas[L[i]]^2*0.5 + log(sigmas[L[i]])
    end

    for i in 1:N
        @inbounds ll_nF[groupindex[i]] += llvec[i]
    end
    Yeppp.exp!(ll_nF, ll_nF)
    nothing
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
            sigmas[j] = sqrt((wsum(pwi[:,j], (x .- mu[j]).^2) + 2 * an * sn[j]) / (sum(pwi[:,j]) + 2*an))
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
    exp!(llvec, llvec)

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
            exp!(llvec, llvec)
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
    ll_nF = zeros(nF, C)
    sumlogmat = zeros(nF, ngh*C)
    S_π = zeros(nF, C-1)
    S_μσ = zeros(nF, 2*C)
    S_λ = zeros(nF, 2*C)
    w = zeros(nF, C)
    ml = zeros(nF)
    xtmp = zeros(C*M)
    for jcom in 1:C
        for ix in 1:M
            ixM = ix+M*(jcom-1)
            xtmp[ixM] = ghx[ix]*sigmas[jcom]*sqrt(2)+mu[jcom]
            for i in 1:N
                @inbounds llvec[i] = ifelse(Y[i], -xtmp[ixM] - xb[i], xtmp[ixM] + xb[i])  
            end
            
            exp!(llvec, llvec)
            
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
            w[i, kcom] = ll_nF[i, kcom] * wi[kcom] / ml[i]
            
            S_μσ[i, 2*kcom-1] = sumexp(sumlogmat[i, :], H1(xtmp, mu[kcom], sigmas[kcom]))/ml[i] * w[i, kcom]
            S_μσ[i, 2*kcom] = sumexp(sumlogmat[i, :], H2(xtmp, mu[kcom], sigmas[kcom]))/ml[i] * w[i, kcom]
            S_λ[i, 2*kcom-1] = sumexp(sumlogmat[i, :], H3(xtmp, mu[kcom], sigmas[kcom]))/ml[i] * w[i, kcom]
            S_λ[i, 2*kcom] = sumexp(sumlogmat[i, :], H4(xtmp, mu[kcom], sigmas[kcom]))/ml[i] * w[i, kcom]
            
        end
    end
    S_η = hcat(S_π, S_μσ)
    I_η = S_η'*S_η./nF 
    I_λη = S_λ'*S_η./nF 
    I_λ = S_λ'*S_λ./nF 
    I_all = vcat(hcat(I_η, I_λη'), hcat(I_λη, I_λ))
    D, V = eig(I_all)
    tol2 = abs(D[1]) * 1e-14 
    D[abs(D).<tol2] = tol2
    I_all = V*diagm(D)*V'
    
    I_λ_η = I_all[(3*C):(5*C-1), (3*C):(5*C-1)] - I_all[(3*C):(5*C-1), 1:(3*C-1)] * inv(I_all[1:(3*C-1), 1:(3*C-1)]) * I_all[1:(3*C-1),(3*C):(5*C-1) ]
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
#The first part of Q function for beta, divided by -N
#the goal and gradient function for estimation β
#goal = 1/N ∑∑∑mean [log(1+exp(-η_{im}yᵢ)) for m in 1:M ]
#gradient = 1/N ∑∑∑mean [yᵢx[i,:]/(1+exp(η_{im}yᵢ)) for m in 1:M ]
function Q1(beta_new::Array{Float64,1}, storage::Vector, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, sample_gamma_mat::Matrix{Float64},groupindex::IntegerVector, llvec::Vector{Float64}, llvecnew::Vector{Float64}, xb::Vector)

    N,J = size(X)
    # xb = X*beta_new
    A_mul_B!(xb, X, beta_new)
    ll=0.0
    if length(storage)>0
        fill!(storage, 0.0)
    end
    M = size(sample_gamma_mat,2)
    for jcol in 1:M
        #llvec[:] = xb
        relocate!(llvec, sample_gamma_mat[:,jcol], groupindex, N)
        Yeppp.add!(llvec, llvec, xb)
        #add!(llvec, sample_gamma_mat[groupindex, jcol], llvec)

        # map!(Exp_xy(), llvec, llvec, Y)
        negateiftrue!(llvec, Y)
        exp!(llvec, llvec)
        if length(storage) > 0
            llvecnew[:] = llvec
            # map!(Xy1x(), llvecnew, llvecnew, Y)
            x1x!(llvecnew)
            negateiffalse!(llvecnew, Y)

            for irow in 1:N
                for j in 1:J
                    @inbounds storage[j] += llvecnew[irow] * X[irow,j]
                end
            end
        end
        # map!(Log1p(), llvec, llvec)
        log1p!(llvec)
        ll += sum(llvec)
    end

    if length(storage)>0
        for j in 1:J
            storage[j] = storage[j] / M
        end
    end
    -ll/M
end


function gibbsMH!(X::Matrix, Y::Vector{Bool}, groupindex::IntegerVector, wi::Vector, mu::Vector, sigmas::Vector, β::Vector, ncomponent::Int, nF::Int, N::Int, M::Int, M_discard::Int, proposingsigma::RealVector, wipool::Vector, mupool::Vector, sigmaspool::Vector, L::Vector, L_new::Vector, sample_gamma::Vector, sample_gamma_new::Vector, sample_gamma_mat::Matrix, xb::Vector, llvec::Vector, llvecnew::Vector, ll_nF::Vector, tmp_mu::Vector, wi_divide_sigmas::Vector, inv_2sigmas_sq::Vector, tmp_p::Vector)
    fill!(wipool, 0.0)
    fill!(mupool, 0.0)
    fill!(sigmaspool, 0.0)
    #Gibbs samping for M+M_discard times
    # xb  = X*β
    A_mul_B!(xb, X, β)
    fill!(wi_divide_sigmas, 0.0)
    fill!(inv_2sigmas_sq, 0.0)
    for i in 1:length(wi)
        if sigmas[i] < realmin(Float64)
            wi_divide_sigmas[i] = 0.0
            inv_2sigmas_sq[i] = wi[i]*realmax(Float64)
        else
            wi_divide_sigmas[i] = wi[i]/sigmas[i]
            inv_2sigmas_sq[i] = 0.5 / sigmas[i]^2
        end
    end
    for iter_gibbs in 1:(M+M_discard)
        #update Lᵢ
        for i in 1:nF
            for j in 1:ncomponent
                tmp_mu[j] = -(mu[j] - sample_gamma[i])^2 * inv_2sigmas_sq[j]
            end
            ratiosumexp!(tmp_mu, wi_divide_sigmas, tmp_p, ncomponent)
            L_new[i] = rand(Categorical(tmp_p))
            sample_gamma_new[i] = rand(Normal(sample_gamma[i], proposingsigma[i]))
        end

        #update γᵢ;
        #Calculate the accept probability, stored in ll_nF
        q_gamma(sample_gamma_new, sample_gamma, xb, Y,groupindex, mu, sigmas, L, L_new, llvec, llvecnew, ll_nF, nF, N)
        for i in 1:nF
            if rand() < ll_nF[i]
                sample_gamma[i] = sample_gamma_new[i]
            end
            L[i] = L_new[i]
        end

        #only keep samples after M_discard
        jcol = iter_gibbs - M_discard
        if jcol > 0
            sample_gamma_mat[:, jcol] = sample_gamma
            addcounts!(wipool, L, 1:ncomponent)
            sumby!(mupool, sample_gamma, L, 1:ncomponent)
            sumsqby!(sigmaspool, sample_gamma, L, 1:ncomponent)
        end
    end
    
end
#The main function
#X, Y, groupindex
#nF is the number of groups
#intial values of β, ω, μ and σ must be supplied

function latentgmm(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent::Int, β_init::Vector{Float64}, wi_init::Vector{Float64}, mu_init::Vector{Float64}, sigmas_init::Vector{Float64}; Mmax::Int=10000, M_discard::Int=1000, maxiteration::Int=100, initial_iteration::Int=0, tol::Real=.005, proposingsigma::RealVector=ones(maximum(groupindex)), ngh::Int=1000, sn::Vector{Float64}=sigmas_init, an::Float64=1.0/maximum(groupindex), debuginfo::Bool=false,restartMCMCsampling::Bool=false)

    # initialize theta
    length(wi_init) == length(mu_init) == length(sigmas_init) == ncomponent || error("The length of initial values should be $ncomponent")
    N,J=size(X)
    length(β_init) == J || error("Initial values of fixed efffect coefficients should have same dimension as X")
    nF = maximum(groupindex)
    M = Mmax
    
    wi = copy(wi_init)
    mu = copy(mu_init)
    sigmas = copy(sigmas_init)
    β = copy(β_init)
    
    wi_old = ones(wi)./ncomponent
    mu_old = zeros(mu)
    sigmas_old = ones(sigmas)
    beta_old = randn(J)
    
    ghx, ghw = gausshermite(ngh)
    #Preallocate the storage space, reusable for each iteration
    # L_mat = zeros(Int64, (nF, M))
    L = rand(Categorical(wi), nF)
    L_new = rand(Categorical(wi), nF)
    sample_gamma = zeros(nF)
    sample_gamma_new = zeros(nF)
    sample_gamma_mat = zeros(nF, Mmax)
    sumlogmat = zeros(nF, ngh*ncomponent)
    llvec = zeros(N)
    llvecnew = zeros(N)
    ll_nF = zeros(nF)
    
    wipool = zeros(ncomponent)
    mupool = zeros(ncomponent)
    sigmaspool = zeros(ncomponent)
    tmp_p=ones(ncomponent) / ncomponent
    tmp_mu=zeros(ncomponent)
    wi_divide_sigmas = zeros(wi)
    inv_2sigmas_sq = ones(sigmas) .* 1e20
    xb = zeros(N)
    ################################################################################################################
    #iterattion begins here

    no_iter=1
    M = min(2000, Mmax)
    Q_maxiter = 2
    for iter_em in 1:maxiteration
        if iter_em == (initial_iteration + 1)
            M = Mmax
            Q_maxiter = 10
        end
        if restartMCMCsampling || (iter_em == 1)
            for i in 1:nF
                L[i] = rand(Categorical(wi))
                sample_gamma[i] = rand(Normal(mu[L[i]], sigmas[L[i]])) 
            end
         end
        if any(isnan(sigmas)) || any(isnan(wi)) 
            warn("wi=$wi, sigmas = $sigmas")
            return(wi, mu, sigmas, β, -Inf, [0.0])
        end
         
         gibbsMH!(X, Y, groupindex, wi, mu, sigmas, β, ncomponent, nF, N, M, M_discard, proposingsigma, wipool, mupool, sigmaspool, L, L_new, sample_gamma, sample_gamma_new, sample_gamma_mat, xb, llvec, llvecnew, ll_nF, tmp_mu, wi_divide_sigmas, inv_2sigmas_sq, tmp_p)
    
        for i in 1:nF
            proposingsigma[i] = std(sample_gamma_mat[i,:])+.1
        end
        for j in 1:ncomponent
            if wipool[j] == 0
                wipool[j] = 1
            end
        end
        copy!(wi_old, wi)
        copy!(mu_old, mu)
        copy!(sigmas_old, sigmas)
        #update wi, mu and sigmas
        wipoolsum = sum(wipool)
        for ic in 1:ncomponent
            wi[ic] = wipool[ic] / wipoolsum
            mu[ic] = mupool[ic] / wipool[ic]
            sigmas[ic] = sqrt((sigmaspool[ic] - wipool[ic] * mu[ic] ^2 + 2 * an * sn[ic]) / (wipool[ic] + 2 * an))
        end
        #no longer update beta if it already converged
        if !stopRule(β, beta_old, tol=tol) #(mod(iter_em, 5) == 1 ) & (
            copy!(beta_old, β)
            opt = Opt(:LD_LBFGS, J)
            maxeval!(opt, Q_maxiter)
            max_objective!(opt, (beta_new, storage)->Q1(beta_new, storage, X,Y, sample_gamma_mat[:,1:M], groupindex, llvec, llvecnew, xb))
            (minf,β,ret) = optimize(opt, β)
        end

        if debuginfo
            println(wi, "\t", mu, "\t", sigmas, "\t", marginallikelihood(β, X, Y, groupindex, nF, wi, mu, sigmas, ghx, ghw, llvec, ll_nF, xb, sumlogmat)+sum(pn(sigmas, sn, an=an)))
        end
        if (iter_em == maxiteration) && (maxiteration > 3)
            warn("latentgmm not converge!")
        end
        if stopRule(vcat(β, wi, mu, sigmas), vcat(beta_old, wi_old, mu_old, sigmas_old), tol=tol) && (iter_em > initial_iteration) && (iter_em > 3)
            if debuginfo
                println("latentgmm converged at $(iter_em)th iteration")
            end
            break
        end
    end

     # xb=X*β
     # llmc = Float64[conditionallikelihood(xb, sample_gamma_mat[:,i], Y, groupindex) for i in 1:M]
    #m = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)

    return(wi, mu, sigmas, β, marginallikelihood(β, X, Y, groupindex, nF, wi, mu, sigmas, ghx, ghw, llvec, ll_nF, xb, sumlogmat)+sum(pn(sigmas, sn, an=an)), sample_gamma_mat[:,1:M]) 
end

#For fixed wi
function latentgmm_ctau(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent::Int, β_init::Vector{Float64}, wi_init::Vector{Float64}, mu_init::Vector{Float64}, sigmas_init::Vector{Float64}, whichtosplit::Int64, tau::Float64, ghx::Vector{Float64}, ghw::Vector{Float64}; mu_lb::Vector{Float64}=-Inf.*ones(wi_init), mu_ub::Vector{Float64}=Inf.*ones(wi_init), Mmax::Int=5000, M_discard::Int=1000, maxiteration::Int=100, initial_iteration::Int=0, tol::Real=.005, proposingsigma::RealVector=ones(maximum(groupindex)), sn::Vector{Float64}=sigmas_init, an::Float64=0.25, debuginfo::Bool=false, sample_gamma_mat::Matrix = zeros(maximum(groupindex), Mmax), sumlogmat::Matrix = zeros(maximum(groupindex), length(ghx)*ncomponent), llvec::Vector=zeros(length(Y)), llvecnew::Vector = zeros(length(Y)), ll_nF::Vector = zeros(maximum(groupindex)), xb::Vector=zeros(length(Y)), Q_maxiter::Int = 5, restartMCMCsampling::Bool=false)

    # initialize theta
    N,J=size(X)
    nF = maximum(groupindex)    
    M = Mmax
    tau = min(tau, 1-tau)
    
    wi = copy(wi_init)
    mu = copy(mu_init)
    sigmas = copy(sigmas_init)
    β = copy(β_init)
    
    wi_tmp = wi[whichtosplit]+wi[whichtosplit+1]
    wi[whichtosplit] = wi_tmp*tau
    wi[whichtosplit+1] = wi_tmp*(1-tau)
    mu = min(max(mu, mu_lb), mu_ub)    

    wi_old = ones(wi)./ncomponent
    mu_old = zeros(mu)
    sigmas_old = ones(sigmas)
    beta_old = randn(J)
        
    wi0=copy(wi) # wi0, mu0 is to store the best parameter
    mu0=copy(mu)
    sigmas0=copy(sigmas)
    β0 = copy(β)
    
    #ghx, ghw = gausshermite(ngh)
    #Preallocate the storage space, reusable for each iteration
    # L_mat = zeros(Int64, (nF, M))
    L = rand(Categorical(wi), nF)
    L_new = rand(Categorical(wi), nF)
    sample_gamma = zeros(nF)
    sample_gamma_new = zeros(nF)
    fill!(sample_gamma_mat, 0.0)
    fill!(sumlogmat, 0.0)
    #llvec = zeros(N)
    #llvecnew = zeros(N)
    #ll_nF = zeros(nF)
    
    wipool = zeros(ncomponent)
    mupool = zeros(ncomponent)
    sigmaspool = zeros(ncomponent)
    
    tmp_p=ones(ncomponent) / ncomponent
    tmp_mu = zeros(ncomponent)
    wi_divide_sigmas = zeros(wi)
    inv_2sigmas_sq = ones(sigmas) .* 1e20
    ml0=-Inf

    ################################################################################################################
    #iterattion begins here

    no_iter=1
    #M = min(2000, Mmax)
    #Q_maxiter = 2
    lessthanmax = 0
    for iter_em in 1:maxiteration
        if restartMCMCsampling || (iter_em == 1)
            for i in 1:nF
                L[i] = rand(Categorical(wi))
                sample_gamma[i] = rand(Normal(mu[L[i]], sigmas[L[i]])) 
            end
         end
         if any(isnan(sigmas)) || any(isnan(wi)) 
             warn("wi=$wi, sigmas = $sigmas")
             return(wi, mu, sigmas, β, -Inf)
         end
         gibbsMH!(X, Y, groupindex, wi, mu, sigmas, β, ncomponent, nF, N, M, M_discard, proposingsigma, wipool, mupool, sigmaspool, L, L_new, sample_gamma, sample_gamma_new, sample_gamma_mat, xb, llvec, llvecnew, ll_nF, tmp_mu, wi_divide_sigmas, inv_2sigmas_sq, tmp_p)
         for i in 1:nF
             proposingsigma[i] = std(sample_gamma_mat[i,:]) + .1
         end
         copy!(wi_old, wi)
         copy!(mu_old, mu)
         copy!(sigmas_old, sigmas)
         #update wi, mu and sigmas
         for j in 1:ncomponent
             if wipool[j] <= 1
                 warn("wi contains 0!")
                 return(wi, mu, sigmas, β, -Inf)
             end
         end
         wipoolsum = sum(wipool)
         for ic in 1:ncomponent
             wi[ic] = wipool[ic] / wipoolsum
             mu[ic] = mupool[ic] / wipool[ic]
             sigmas[ic] = sqrt((sigmaspool[ic] - wipool[ic] * mu[ic] ^2 + 2 * an * sn[ic]) / (wipool[ic] + 2 * an))
         end  
         
        wi_tmp = wi[whichtosplit]+wi[whichtosplit+1]
        wi[whichtosplit] = wi_tmp*tau
        wi[whichtosplit+1] = wi_tmp*(1-tau)
        mu = min(max(mu, mu_lb), mu_ub)        
        
        #no longer update beta if it already converged
         if !stopRule(β, beta_old, tol=tol) #(mod(iter_em, 5) == 1 ) 
             copy!(beta_old, β)
             opt = Opt(:LD_LBFGS, J)
             maxeval!(opt, Q_maxiter)
             max_objective!(opt, (beta_new, storage)->Q1(beta_new, storage, X,Y, sample_gamma_mat[:, 1:M], groupindex, llvec, llvecnew, xb))
             (minf,β,ret) = optimize(opt, β)
         end

        if (iter_em == maxiteration) && (maxiteration > 20)
            warn("latentgmm_ctau not yet converge!")
        end
        ml1 = marginallikelihood(β, X, Y, groupindex, nF, wi, mu, sigmas, ghx, ghw, llvec, ll_nF, xb, sumlogmat) + sum(pn(sigmas, sn, an=an))
        if debuginfo
            println(wi, " ", mu, " ", sigmas, " ", sum(pn(sigmas, sn, an=an)), " ", ml1)
        end
        if ml1 > ml0
            ml0 = ml1
            copy!(wi0, wi)
            copy!(mu0, mu)
            copy!(sigmas0, sigmas)
            copy!(β0, β)
            lessthanmax = 0
        else
            lessthanmax += 1
        end
        if lessthanmax > 3
            if debuginfo
                println("latentgmm_ctau stop at $(iter_em)th iteration")
            end
            break
        end
        
    end
    #For fixed wi, no need to output gamma_mat
    return(wi0, mu0, sigmas0, β0, ml0)
end

#Starting from 25 initial values, find the best for fixed wi, used as start of the next 2 more iterations
function loglikelihoodratio_ctau(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent1::Int,  betas0::Vector{Float64}, wi_C1::Vector{Float64},  whichtosplit::Int64, tau::Float64, mu_lb::Vector{Float64}, mu_ub::Vector{Float64}, sigmas_lb::Vector{Float64}, sigmas_ub::Vector{Float64}, gamma0::Vector{Float64}; ntrials::Int=25, ngh::Int=1000, sn::Vector{Float64}=sigmas_ub ./ 2, an=.25, debuginfo::Bool=false, sample_gamma_mat::Matrix = zeros(maximum(groupindex), Mmax), sumlogmat::Matrix = zeros(maximum(groupindex), length(ghx)*ncomponent), llvec::Vector=zeros(length(Y)), llvecnew::Vector = zeros(length(Y)), xb::Vector=zeros(length(Y)), Mctau::Int=1000, restartMCMCsampling::Bool=false)

    nF = maximum(groupindex)
    tau = min(tau, 1-tau)
    ghx, ghw = gausshermite(ngh)
    
    wi = repmat(wi_C1, 1, 4*ntrials)
    mu = zeros(ncomponent1, 4*ntrials)
    sigmas = ones(ncomponent1, 4*ntrials)
    betas = repmat(betas0, 1, 4*ntrials)
    ml = -Inf .* ones(4*ntrials) 
    for i in 1:4*ntrials
        mu[:, i] = rand(ncomponent1) .* (mu_ub .- mu_lb) .+ mu_lb
        sigmas[:, i] = rand(ncomponent1) .* (sigmas_ub .- sigmas_lb) .+ sigmas_lb

        wi[:, i], mu[:, i], sigmas[:, i], betas[:, i], ml[i] = latentgmm_ctau(X, Y, groupindex, ncomponent1, betas0, wi[:, i], mu[:, i], sigmas[:, i], whichtosplit, tau, ghx, ghw, mu_lb=mu_lb,mu_ub=mu_ub, maxiteration=5, Mmax=Mctau, M_discard=500, sn=sn, an=an, sample_gamma_mat = sample_gamma_mat, sumlogmat=sumlogmat, llvec=llvec, llvecnew=llvecnew, xb=xb, Q_maxiter=2, restartMCMCsampling=restartMCMCsampling)
    end
    
    mlperm = sortperm(ml)
    for j in 1:ntrials
        i = mlperm[4*ntrials+1 - j] # start from largest ml 
        wi[:, i], mu[:, i], sigmas[:, i], betas[:, i], ml[i] = latentgmm_ctau(X, Y, groupindex, ncomponent1, betas[:, i], wi[:, i], mu[:, i], sigmas[:, i], whichtosplit, tau, ghx, ghw, mu_lb=mu_lb,mu_ub=mu_ub, maxiteration=200, initial_iteration=0, Mmax=Mctau, M_discard=500, sn=sn, an=an, debuginfo=debuginfo, sample_gamma_mat = sample_gamma_mat, sumlogmat=sumlogmat, llvec=llvec, llvecnew=llvecnew, xb=xb, Q_maxiter=10, restartMCMCsampling=restartMCMCsampling)
    end
    
    mlmax, imax = findmax(ml[mlperm[(3*ntrials+1):4*ntrials]])
    imax = mlperm[3*ntrials+imax]
    
    re=latentgmm(X, Y, groupindex, ncomponent1, betas[:, imax], wi[:, imax], mu[:, imax], sigmas[:, imax], Mmax=5000, maxiteration=3, initial_iteration=0, an=an, sn=sn, debuginfo=debuginfo, restartMCMCsampling=restartMCMCsampling)
    
    return(re[5])
end

function loglikelihoodratio(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent1::Int; vtau::Vector{Float64}=[.5,.3,.1;], ntrials::Int=25, ngh::Int=1000, debuginfo::Bool=false, Mctau::Int=1000, restartMCMCsampling::Bool=false, reportpvalue=false, ctauparallel=true)
    C0 = ncomponent1 - 1
    C1 = ncomponent1 
    nF = maximum(groupindex)
    an1 = 1/nF
    gamma_init, beta_init, sigmas_tmp = maxposterior(X, Y, groupindex)
    wi_init, mu_init, sigmas_init, ml_tmp = gmm(gamma_init, C0, ones(C0)/C0, quantile(gamma_init, linspace(0, 1, C0+2)[2:end-1]), ones(C0), an=an1)

    wi_init, mu_init, sigmas_init, betas_init, ml_C0, gamma_mat = latentgmm(X, Y, groupindex, C0, beta_init, wi_init, mu_init, sigmas_init, Mmax=5000, initial_iteration=10, maxiteration=100, an=an1, sn=std(gamma_init).*ones(C0), restartMCMCsampling=restartMCMCsampling)
    if reportpvalue
        trand=LatentGaussianMixtureModel.asymptoticdistribution(X, Y, groupindex, wi_init, mu_init, sigmas_init, betas_init)
    end
    
    gamma0 = vec(mean(gamma_mat, 2))    
    mingamma = minimum(gamma0)
    maxgamma = maximum(gamma0)
    
    or = sortperm(mu_init)
    wi0 = wi_init[or]
    mu0 = mu_init[or]
    sigmas0 = sigmas_init[or]
    betas0 = betas_init
    an = decidepenalty(wi0, mu0, sigmas0, nF)
    
    N,J=size(X)
    sample_gamma_mat = zeros(nF, Mctau)
    sumlogmat = zeros(nF, ngh*ncomponent1)
    llvec = zeros(N)
    llvecnew = zeros(N)
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

            loglikelihoodratio_ctau(X, Y, groupindex, ncomponent1, betas0, wi_C1, whichtosplit, vtau[i], mu_lb, mu_ub,sigmas_lb, sigmas_ub, gamma0, ntrials=ntrials, ngh=ngh, sn=sigmas0[ind], an=an, debuginfo=debuginfo, sample_gamma_mat = sample_gamma_mat, sumlogmat=sumlogmat, llvec=llvec, llvecnew=llvecnew, Mctau=Mctau, xb=xb, restartMCMCsampling=restartMCMCsampling)

        end

        if reportpvalue
            return 2*(lr - ml_C0), mean(trand .> 2*(lr - ml_C0))
        end
        return 2*(lr - ml_C0)  
    else
        lr = zeros(length(vtau)* C0)
        for irun in 1:(C0*length(vtau))

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

         lr[irun]=loglikelihoodratio_ctau(X, Y, groupindex, ncomponent1, betas0, wi_C1, whichtosplit, vtau[i], mu_lb, mu_ub,sigmas_lb, sigmas_ub, gamma0, ntrials=ntrials, ngh=ngh, sn=sigmas0[ind], an=an, debuginfo=debuginfo, sample_gamma_mat = sample_gamma_mat, sumlogmat=sumlogmat, llvec=llvec, llvecnew=llvecnew, Mctau=Mctau, xb=xb, restartMCMCsampling=restartMCMCsampling)

     end

     if reportpvalue
         return 2*(maximum(lr) - ml_C0), mean(trand .> 2*(maximum(lr) - ml_C0))
     end
     return 2*(maximum(lr) - ml_C0)
        
    end
end

####End of utility functions
