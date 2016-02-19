#The utility functions
#Lanfeng Pan
#Oct 29, 2014

function stopRule(pa::Vector, pa_old::Vector; tol=.005)
    maximum(abs(pa .- pa_old)./(abs(pa).+.001)) < tol
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
function FDR(X::Matrix, Y::Vector{Bool}, groupindex::IntegerVector, wi::Vector, mu::Vector, sigmas::Vector, β::Vector, C0::IntegerVector; ngh::Int=100, alphalevel::Real=.05)

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
        for jcom in 1:ncomponent
            for ix in 1:ngh
                ixM = ix+ngh*(jcom-1)
                piposterior[i, jcom] += Wim[i,ixM]
            end
        end
    end
    for i in 1:n
        for jcom in 1:ncomponent
            clFDR[i] = sum(piposterior[i, C0])
        end
    end
    order = sortperm(clFDR)
    n0 = 0
    for i in 1:n
        if mean(clFDR[order[1:i]]) < alphalevel
            n0 += 1
        end
    end
    return clFDR, order[1:n0]
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
