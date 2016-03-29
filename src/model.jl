using StatsBase: StatsBase, StatisticalModel, RegressionModel

type LGMModel <: RegressionModel
    X::Matrix{Float64}
    Y::Vector{Bool}
    groupindex::Vector{Int}
    ncomponent::Int
    p::Vector{Float64}
    μ::Vector{Float64}
    σ::Vector{Float64}
    β::Vector{Float64}
    n::Int
    ngh::Int
    ghx::Vector
    ghw::Vector
    μ_lb::Vector
    μ_ub::Vector
    σ_lb::Vector
    σ_ub::Vector
    Wim::Matrix{Float64}
    Wm::Matrix{Float64}
    lln::Vector{Float64}
    llN::Vector{Float64}
    llN2::Vector{Float64}
    llN3::Vector{Float64}
    Xscratch::Matrix{Float64}
    xb::Vector{Float64}
    gammaM::Vector{Float64}
    XWX::Matrix{Float64}
    XWY::Vector{Float64}
    ll::Float64
    sn::Vector{Float64}
    an::Float64
    taufixed::Bool
    whichtosplit::Int
    tau::Real
    fit::Bool

    function LGMModel(X::Matrix{Float64}, Y::Vector{Bool}, groupindex::Vector{Int}, ncomponent::Int, ngh::Int=100)
        n = maximum(groupindex)
        N, J = size(X)
        Wim::Matrix{Float64}=zeros(n, ncomponent*ngh)
        Wm::Matrix{Float64}=zeros(1, ncomponent*ngh)
        lln::Vector{Float64}=zeros(n)
        llN::Vector{Float64}=zeros(N)
        llN2::Vector{Float64}=zeros(N)
        llN3::Vector{Float64}=zeros(N)
        Xscratch::Matrix{Float64}=copy(X)
        xb::Vector{Float64}=zeros(N)
        gammaM::Vector{Float64}=zeros(ncomponent*ngh)
        XWX = zeros(J, J)
        XWY = zeros(J)
        ghx, ghw = gausshermite(ngh)
        p = ones(ncomponent) ./ ncomponent
        μ = sort!(randn(ncomponent))
        σ = ones(ncomponent)
        β = randn(J)
        μ_lb=fill(-Inf, ncomponent)
        μ_ub=fill(Inf, ncomponent)
        σ_lb = .25 .* σ
        σ_ub = 2.0 .* σ
        ll = 0.0
        sn=ones(ncomponent)
        an=1.0/n
        taufixed=false
        whichtosplit=1
        tau=.5
        
        new(X, Y, groupindex, ncomponent, p, μ, σ, β, n, ngh, ghx, ghw, μ_lb, μ_ub, σ_lb, σ_ub, Wim, Wm, lln, llN, llN2, llN3, Xscratch, xb, gammaM, XWX, XWY, ll, sn, an, taufixed, whichtosplit, tau, false)
    end
end
function initialize!(m::LGMModel)
    if m.ncomponent > 1
        m0 = LGMModel(X, Y, groupindex, 1)
        latentgmm!(m0)
        gammaprediction = predictgamma(m0);
        m.p, m.μ, m.σ, ml_tmp = gmm(gammaprediction, m.ncomponent)
        m.β = m0.β
        # mingamma = minimum(gammaprediction) - 3 * std(gammaprediction)
        # maxgamma = maximum(gammaprediction) + 3 * std(gammaprediction)
        # fill!(m.μ_lb, mingamma)
        # fill!(m.μ_ub, maxgamma)
        # fill!(m.σ_lb, .25 .* m.σ)
        # fill!(m.σ_ub, 2.0 .* m.σ)
    end
end
function StatsBase.fit!(m::LGMModel;
    maxiteration::Int=100, tol::Real=.005,
    debuginfo::Bool=false, Qmaxiteration::Int=5,
    dotest::Bool=false, epsilon::Real=1e-6,
    updatebeta::Bool=true, bn::Real=1e-4,
    pl::Bool=true, ptau::Bool=false)

    m.fit && return m
    N, J = size(m.X)
    n = m.n
    ncomponent = m.ncomponent
    ngh = m.ngh
    M = ncomponent*m.ngh

    p_old = copy(m.p)
    μ_old = copy(m.μ)
    σ_old = copy(m.σ)
    β_old = copy(m.β)

    ll0=-Inf
    alreadystable = false
    if maxiteration <= 3
        alreadystable = true
    end
    for iter_em in 1:maxiteration
        for ix in 1:ngh, jcom in 1:ncomponent
            ixM = ix+ngh*(jcom-1)
            m.gammaM[ixM] = m.ghx[ix]*m.σ[jcom]*sqrt(2)+m.μ[jcom]
        end

        copy!(p_old, m.p)
        copy!(μ_old, m.μ)
        copy!(σ_old, m.σ)

        A_mul_B!(m.xb, m.X, m.β)
        m.ll=integralweight!(m.Wim, m.X, m.Y, m.groupindex, m.gammaM, m.p, m.ghw, m.llN, m.llN2, m.xb, N, J, n, ncomponent, ngh)
        lldiff = m.ll - ll0
        ll0 = m.ll
        if lldiff < 1e-4
            #alreadystable = true
            Qmaxiteration = 2 * Qmaxiteration
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
            copy!(β_old, m.β)
            updateβ!(m.β, m.X, m.Y, m.groupindex, .001, .001,
            m.XWX, m.XWY, m.Xscratch, m.gammaM, m.Wim, m.lln, 
            m.llN, m.llN2, m.llN3,
            m.xb, N, J, n, ncomponent, ngh, Qmaxiteration)
            debuginfo && println("beta=", m.β)
        end
        updateθ!(m.p, m.μ, m.σ, m.X, m.Y, m.groupindex,
        m.gammaM, m.Wim, m.Wm, m.sn, m.an, N, J, n, ncomponent, ngh)

        if bn > 0.0
            for kcom in 1:ncomponent
                m.p[kcom]=(m.p[kcom]*n+bn/ncomponent)/(n+bn)
            end
        end
        if m.taufixed
            Yeppp.max!(m.μ, m.μ, m.μ_lb)
            Yeppp.min!(m.μ, m.μ, m.μ_ub)
            p_tmp = m.p[m.whichtosplit]+m.p[m.whichtosplit+1]
            m.p[m.whichtosplit] = p_tmp*m.tau
            m.p[m.whichtosplit+1] = p_tmp*(1-m.tau)
        end
        if debuginfo
            println("p=$(m.p)")
            println("μ=$(m.μ)")
            println("σ=$(m.σ)")
            println("loglikelihood=$(m.ll)")
        end

        if !dotest
            if stopRule(vcat(m.β, m.p, m.μ, m.σ), vcat(β_old, p_old, μ_old, σ_old), tol=tol)
                if debuginfo
                    println("Converged at $(iter_em)th iteration.")
                end
                break
            end
        end
        if (iter_em == maxiteration) && (maxiteration > 50)
            warn("Fail to converge with $(iter_em) iterations. The taufixed is $(taufixed).")
            println("Current parameters are
            $(m.p), $(m.μ), $(m.σ), $(m.β).")
            println(" Current likelihood is $(m.ll). The likelihood increase from last iteration is $(lldiff).")
        end
    end
    if pl
        m.ll += sum(pn(m.σ, m.sn, an=m.an))
    end
    if ptau
        tau2 = m.p[m.whichtosplit] / (m.p[m.whichtosplit]+m.p[m.whichtosplit+1])
        m.ll += log(1 - abs(1 - 2*tau2))
    end
    m.fit=true
    return(m)
end

function predictgamma(m::LGMModel)
    ncomponent = m.ncomponent
    n = m.n
    ngh = m.ngh
    M = ngh*ncomponent
    N, J = size(m.X)

    gammahat = zeros(n)
    for ix in 1:ngh, jcom in 1:ncomponent
        ixM = ix+ngh*(jcom-1)
        m.gammaM[ixM] = m.ghx[ix]*m.σ[jcom]*sqrt(2)+m.μ[jcom]
    end
    A_mul_B!(m.xb, m.X, m.β)
    integralweight!(m.Wim, m.X, m.Y, m.groupindex, m.gammaM, m.p, m.ghw, m.llN, m.llN2, m.xb, N, J, n, ncomponent, ngh)
    for i in 1:n
        for j in 1:M
            gammahat[i] += m.gammaM[j] * m.Wim[i,j]
        end
    end
    return gammahat
end

nobs(m::LGMModel) = m.n
model_response(m::LGMModel) = m.Y
coef(m::LGMModel) = m.β
ranef(m::LGMModel) = predictgamma(m)
#deviance(m::LGMModel) = -2*loglikelihood(m)
function stderr(m::LGMModel)
    vc = vcov(m)
    J = size(m.X, 2)
    sqrt(diag(vc))[1:J]
end
function confint(m::LGMModel, level::Real)
    hcat(coef(m),coef(m)) + stderr(m)*quantile(Normal(),(1. -level)/2.)*[1. -1.]
end
confint(m::LGMModel) = confint(m, 0.95)
function coeftable(m::LGMModel)
    cc = coef(m)
    se = stderr(m)
    zz = cc ./ se
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["x$i" for i = 1:size(m.X, 2)], 4)
end

function loglikelihood(m::LGMModel)
    N, J = size(m.X)
    A_mul_B!(m.xb, m.X, m.β)
    for ix in 1:m.ngh, jcom in 1:m.ncomponent
        ixM = ix+m.ngh*(jcom-1)
        m.gammaM[ixM] = m.ghx[ix]*m.σ[jcom]*sqrt(2)+m.μ[jcom]
    end

    m.ll=integralweight!(m.Wim, m.X, m.Y, m.groupindex, m.gammaM, m.p, m.ghw, m.llN, m.llN2, m.xb, N, J, m.n, m.ncomponent, m.ngh)
    m.ll
end

function vcov(m::LGMModel; debuginfo::Bool=false)

    N,J = size(m.X)
    n = m.n
    ngh=m.ngh
    C = m.ncomponent

    summat_beta = zeros(n, ngh*C, J)
    llnC = zeros(n, C)
    S_β = zeros(n, J)
    S_π = zeros(n, C-1)
    S_μσ = zeros(n, 2*C)
    A_mul_B!(m.xb, m.X, m.β)
    ml = zeros(n)
    for jcom in 1:C
        for ix in 1:ngh
            ixM = ix+ngh*(jcom-1)
            m.gammaM[ixM] = m.ghx[ix]*m.σ[jcom]*sqrt(2)+m.μ[jcom]
            for i in 1:N
                @inbounds m.llN[i] = ifelse(m.Y[i], -m.gammaM[ixM] - m.xb[i], m.gammaM[ixM] + m.xb[i])
            end
            copy!(m.llN2, m.llN)
            logistic!(m.llN2)
            negateiffalse!(m.llN2, m.Y)
            for i in 1:N
                @inbounds ind = m.groupindex[i]::Int
                for j in 1:J 
                    @inbounds summat_beta[ind, ixM, j] += m.llN2[i] * m.X[i,j]
                end
            end
            log1pexp!(m.llN, m.llN, m.llN3, N)

            for i in 1:N
                @inbounds m.Wim[m.groupindex[i], ixM] -= m.llN[i]
            end
            for i in 1:n
                @inbounds m.Wim[i, ixM] +=  log(m.ghw[ix])
            end
        end
    end
    for i in 1:n
        u = maximum(m.Wim[i, :])
        for jcol in 1:C*ngh
            @inbounds m.Wim[i, jcol] = m.Wim[i, jcol] - u
        end
    end
    for i in 1:n
        for kcom in 1:C
            @inbounds llnC[i, kcom] = sumexp(m.Wim[i,(1+ngh*(kcom-1)):ngh*kcom])
        end
    end
    for i in 1:n
        for jcom in 1:C
            for ix in 1:ngh
                @inbounds m.Wim[i, ix+ngh*(jcom-1)] += log(m.p[jcom])
            end
        end
        ml[i]=sumexp(m.Wim[i, :])
    end
    for kcom in 1:(C-1)
        S_π[:, kcom] = (llnC[:, kcom] .- llnC[:, C]) ./ ml
    end
    for i in 1:n
        for kcom in 1:C
            ind = (1+ngh*(kcom-1)):ngh*kcom
            S_μσ[i, 2*kcom-1] = sumexp(m.Wim[i, ind], H1(m.gammaM[ind], m.μ[kcom], m.σ[kcom])) / ml[i]
            S_μσ[i, 2*kcom] = sumexp(m.Wim[i, ind], H2(m.gammaM[ind], m.μ[kcom], m.σ[kcom]))/ml[i]
        end
        for j in 1:J
            S_β[i, j] = sumexp(m.Wim[i,:], summat_beta[i, :, j])/ml[i]
        end
    end
    S_η = hcat(S_β, S_π, S_μσ)
    debuginfo && println(round(S_η[1:5,:], 5))
    debuginfo && println(round(sum(S_η, 1)./sqrt(n), 6))
    I_η = S_η'*S_η./n
    if 1/cond(I_η) < eps(Float64)
        warn("Information Matrix is singular!")
        D, V = eig(I_η)
        debuginfo && println(D)
        tol2 = maximum(abs(D)) * 1e-14
        D[D.<tol2] = tol2
        I_η = V*diagm(D)*V'
    end
    inv(I_η)[1:J,1:J]./n
end

function predict(m::LGMModel)
    !m.fit && warn("The model is not fitted!")
    N, J = size(m.X)
    A_mul_B!(m.xb, m.X, m.β)
    gammaprediction = ranef(m)
    for i in 1:N
        m.llN[i]=m.xb[i] + gammaprediction[m.groupindex[i]]
    end
    m.llN
end

function imputeY(m::LGMModel)
    
    
end
