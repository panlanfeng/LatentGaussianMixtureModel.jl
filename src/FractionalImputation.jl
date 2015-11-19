function FIintegralweight!(Wim::Matrix{Float64}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,  gammaM::Matrix{Float64}, gammah::Matrix{Float64}, wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, llN::Vector{Float64}, xb::Vector{Float64}, proposingdist::Distribution,  N::Int, J::Int, n::Int, M::Int)
    #A_mul_B!(xb, X, betas)
    m = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)
    copy!(Wim, gammah)
    for ixM in 1:M

        relocate!(llN, gammaM[:, ixM], groupindex, N)
        Yeppp.add!(llN, llN, xb)
        negateiftrue!(llN, Y)
        Yeppp.exp!(llN, llN)
        log1p!(llN)

        for i in 1:n
            Wim[i, ixM] += logpdf(m, gammaM[i,ixM]) #- logpdf(proposingdist, gammaM[i,ixM] - mean(m))
        end
        for i in 1:N
            @inbounds Wim[groupindex[i], ixM] -= llN[i]
        end
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

function FIupdateθ!(wi::Vector{Float64}, mu::Vector{Float64}, sigmas::Vector{Float64}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, gammaM::Matrix{Float64}, Wim::Matrix{Float64}, wipool::Vector, mupool::Vector, sigmaspool::Vector, tmp_p::Vector, tmp_mu::Vector, wi_divide_sigmas::Vector, inv_2sigmas_sq::Vector, sn::Vector, an::Real, tau::Real, wifixed::Bool, tol::Real, N::Int, J::Int, n::Int, C::Int, M::Int, thetamaxiteration::Int)

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

function FIupdateβ!(β::Vector{Float64}, X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector,  gammaM::Matrix{Float64}, Wim::Matrix{Float64}, lln::Vector{Float64}, llN::Vector{Float64}, llN2::Vector{Float64}, xb::Vector{Float64}, N::Int, J::Int, n::Int, M::Int, Qmaxiteration::Int)

    opt = Opt(:LD_LBFGS, J)
    maxeval!(opt, Qmaxiteration)
    max_objective!(opt, (beta2, storage)->FI_Q1(beta2, storage, X, Y,  groupindex, gammaM, Wim, lln, llN, llN2, xb, N, J, n, M))

    #(minf,β,ret)=optimize(opt, β)
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

function latentgmmFI(X::Matrix{Float64}, Y::AbstractArray{Bool, 1}, groupindex::IntegerVector, ncomponent::Int, β_init::Vector{Float64}, wi_init::Vector{Float64}, mu_init::Vector{Float64}, sigmas_init::Vector{Float64}; M::Int = 2000, proposingdist::Distribution=TDist(3),
    maxiteration::Int=100, tol::Real=.005,
     sn::Vector{Float64}=sigmas_init, an::Float64=1.0/maximum(groupindex),
    debuginfo::Bool=false, Qmaxiteration::Int=2, whichtosplit::Int=1, tau::Real=.5, wifixed::Bool=false, needcalibration::Bool = (M<100),  thetamaxiteration::Int=10,
     mu_lb::Vector=fill(-Inf, ncomponent), mu_ub::Vector=fill(Inf, ncomponent),
     Wim::Matrix{Float64}=zeros(maximum(groupindex), M), lln::Vector{Float64}=zeros(maximum(groupindex)), llN::Vector{Float64}=zeros(length(Y)),
    llN2::Vector{Float64}=zeros(length(Y)), xb::Vector{Float64}=zeros(length(Y)), gammaM::Matrix{Float64}=zeros(maximum(groupindex), M), gammah::Matrix{Float64}=zeros(maximum(groupindex), M))

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
    p = J #+3*ncomponent-1
    if needcalibration
        stheta::Array{Float64}= zeros(n, M, p)
    end
    wipool = zeros(ncomponent)
    mupool = zeros(ncomponent)
    sigmaspool = zeros(ncomponent)
    tmp_p=ones(ncomponent) / ncomponent
    tmp_mu=zeros(ncomponent)
    wi_divide_sigmas = zeros(wi)
    inv_2sigmas_sq = ones(sigmas) .* 1e20
    minit = MixtureModel(map((u, v) -> Normal(u, v), mu, sigmas), wi)
    meaninit = mean(minit)
    ratioinit = std(minit)/std(proposingdist)
    for i in 1:n, jcol in 1:M
        gammatmp = rand(proposingdist)
        gammaM[i, jcol] = gammatmp * ratioinit + meaninit
        gammah[i, jcol] = log(ratioinit)-logpdf(proposingdist, gammatmp)
    end
    for iter_em in 1:maxiteration

        copy!(wi_old, wi)
        copy!(mu_old, mu)
        copy!(sigmas_old, sigmas)

        A_mul_B!(xb, X, β)
        FIintegralweight!(Wim, X, Y, groupindex, gammaM, gammah, wi, mu, sigmas, llN, xb, proposingdist, N, J, n, M)

        if needcalibration
            calibrate!(Wim, X, Y, groupindex, wi, mu, sigmas, β, gammaM, stheta, xb, llN, N, J, n, ncomponent, M, p)
        end

        if debuginfo
            println("At $(iter_em)th iteration:")
        end
        if !stopRule(β, beta_old, tol=tol/10)
            copy!(beta_old, β)
            FIupdateβ!(β, X, Y, groupindex, gammaM, Wim, lln, llN, llN2, xb, N, J, n, M, Qmaxiteration)
            if debuginfo
                println("beta=", β)
            end
        end
        FIupdateθ!(wi, mu, sigmas, X, Y, groupindex, gammaM, Wim, wipool, mupool, sigmaspool, tmp_p, tmp_mu, wi_divide_sigmas, inv_2sigmas_sq, sn, an, tau, wifixed, tol, N, J, n, ncomponent, M, thetamaxiteration)

        if debuginfo
            println("wi=$wi")
            println("mu=$mu")
            println("sigma=$sigmas")
            #println("ll=",marginallikelihood(β, X, Y, groupindex, n, wi, mu, sigmas, ghx, ghw, llN, lln, xb, sumlogmat))
        end

        if stopRule(vcat(β, wi, mu, sigmas), vcat(beta_old, wi_old, mu_old, sigmas_old), tol=tol) && (iter_em > 3)
            if debuginfo
                println("latentgmmFI converged at $(iter_em)th iteration")
            end
            break
        end
        if (iter_em == maxiteration) && (maxiteration > 15)
            warn("latentgmmFI not converge!")
        end
    end
    return(wi, mu, sigmas, β)
end
