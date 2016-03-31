
"""
    LGMModel(X, Y, groupindex, ncomponent)

Constructing an modelling object.
Optional keywords arguments:

 - `X` and `Y` are the covariates and response. 
 - `groupindex` is the random effect groups, should be an interger vector
 - `ncomponent` is the number of components to try
 - `ngh` means the number of points used in Gaussian-Hermite quadrature approximation in calculating the marginal log likelihood.
 - `taufixed` specifies if fixing the ratio of `p[whichtosplit]` and `p[whichtosplit+1])` to be `tau/(1-tau)`
 - `sn` is the standard deviation to used in penalty function 
 - `an` is the penalty factor

The following methods are available:

  - `nobs` returns the number of random effect levels
  - `model_response` returns the response `Y`
  - `coef` returns the fixed effects `β`
  - `ranef!` return the predict random effects
  - `stderr` gives the standard error of fixed effects
  - `confint` calculates the confidence interval
  - `coeftable` prints the fixed effects and their p values
  - `loglikelihood` calculates the log marginal likelihood
  - `vcov` returns the covariance matrix of fixed effects
  -  `asymptoticdistribution` returns the simulated asymptotic distribution of the restricted likelihood ratio test
  - `predict` computes the probability of `Y` being 1 at given new data
  - `FDR` detect the "outstanding" random effects while controlling the False  Discovery Rate.

"""
type LGMModel <: RegressionModel
    X::Matrix{Float64}
    Y::Vector{Bool}
    groupindex::Vector{UInt32}
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
    gammaprediction::Vector{Float64}
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

    function LGMModel(X::Matrix{Float64}, Y::Vector{Bool}, groupindex::Vector{UInt32}, ncomponent::Int;
        ngh::Int=100,
        taufixed::Bool=false, whichtosplit=1, tau=.5,
        sn=ones(ncomponent), an=1.0/maximum(groupindex))
        
        n = maximum(groupindex)
        N, J = size(X)
        Wim=zeros(n, ncomponent*ngh)
        Wm=zeros(1, ncomponent*ngh)
        gammaprediction = zeros(n)
        lln=zeros(n)
        llN=zeros(N)
        llN2=zeros(N)
        llN3=zeros(N)
        Xscratch=copy(X)
        xb::Vector{Float64}=zeros(N)
        gammaM=zeros(ncomponent*ngh)
        XWX = zeros(J, J)
        XWY = zeros(J)
        ghx, ghw = gausshermite(ngh)
        p = ones(ncomponent) ./ ncomponent
        μ = sort!(randn(ncomponent))
        σ = ones(ncomponent)
        β = randn(J)
        ll = -Inf
        μ_lb = μ .- 10
        μ_ub = μ .+ 10
        σ_lb = .25 .* σ
        σ_ub = 2.0 .* σ
        
        new(X, Y, groupindex, ncomponent, p, μ, σ, β, n, ngh, ghx, ghw, μ_lb, μ_ub, σ_lb, σ_ub, Wim, Wm, gammaprediction, lln, llN, llN2, llN3, Xscratch, xb, gammaM, XWX, XWY, ll, sn, an, taufixed, whichtosplit, tau, false)
    end
end
function show(io::IO, obj::LGMModel)
    if !obj.fit
        warn("Model has not been fit")
        return nothing
    end
    
    println(io, "$(typeof(obj)):\n\nFixed Coefficients:\n", coeftable(obj))
    println(io, "Random Effects:\n\n", MixtureModel(map((u, v) -> Normal(u, v), obj.μ, obj.σ), obj.p))
    println(io, "Number of random effect levels:", obj.n)
    println(io, "Log likelihood is: ", obj.ll)
end
function initialize!(m::LGMModel)
    
    m0 = LGMModel(m.X, m.Y, m.groupindex, 1)
    fit!(m0)
    ranef!(m0);
    m.p, m.μ, m.σ, ml_tmp = gmm(m0.gammaprediction, m.ncomponent)
    m.β = m0.β
    or = sortperm(m.μ)
    m.p = m.p[or]
    m.μ = m.μ[or]
    m.σ = m.σ[or]

    mingamma = minimum(m0.gammaprediction) - 3 * std(m0.gammaprediction)
    maxgamma = maximum(m0.gammaprediction) + 3 * std(m0.gammaprediction)
    fill!(m.μ_lb, mingamma)
    fill!(m.μ_ub, maxgamma)
    copy!(m.σ_lb, .25 .* m.σ)
    copy!(m.σ_ub, 2.0 .* m.σ)
    m
end
const valid_opts = [:maxiteration, :tol, :debuginfo, :Qmaxiteration, :dotest, :epsilon, :updatebeta, :bn, :pl, :ptau]
const valid_opt_types = [Int, Real, Bool, Int, Bool, Real, Bool, Real, Bool, Bool]
const deprecated_opts = Dict()
function val_opts(opts)
    d = Dict{Symbol,Union{Bool,NTuple{2,Integer},Char,Integer}}()
    for (opt_name, opt_val) in opts
        !in(opt_name, valid_opts) && throw(ArgumentError("unknown option $opt_name"))
        opt_typ = valid_opt_types[findfirst(valid_opts, opt_name)]
        !isa(opt_val, opt_typ) && throw(ArgumentError("$opt_name should be of type $opt_typ, got $(typeof(opt_val))"))
        d[opt_name] = opt_val
        haskey(deprecated_opts, opt_name) && warn("$opt_name is deprecated, use $(deprecated_opts[opt_name]) instead")
    end
    d
end

"""
    fit!(m)

Fits the model.

 - `maxiteration` is the maximum MCEM iterations allowed. `tol` means the stopping criteria. `initial_iteration` is the number of initial iterations using smaller `Mmax`. Initial iterations are used to save time in the first few iterations when the parameters are still far from the truth
 - `tol` the stoppting criteria
 - `debuginfo` is the switch to print more information to help debug
 - `Qmaxiteration` the number of iterations used in Iteratively Reweighted Least Squares, for updating β
 - `bn` the penalty weight on component prior
 - `pl` and `ptau` should the penalty on σ and `tau` be included in loglikelihood
"""
function StatsBase.fit!(m::LGMModel;
    maxiteration::Int=2000, tol::Real=.001,
    debuginfo::Bool=false, Qmaxiteration::Int=5,
    dotest::Bool=false, epsilon::Real=1e-6,
    updatebeta::Bool=true, bn::Real=1e-4,
    pl::Bool=false, ptau::Bool=false)
    
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
        if any(isnan(m.β)) || any(isnan(m.p))
            error("NaN in paramters!")
        end
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
            warn("Fail to converge with $(iter_em) iterations. The taufixed is $(m.taufixed).")
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

function multipefit!(m::LGMModel, ntrials::Int=25; kwargs...)
    pm = repmat(m.p, 1, ntrials)
    μm = repmat(m.μ, 1, ntrials)
    σm = repmat(m.σ, 1, ntrials)
    βm = repmat(m.β, 1, ntrials)
    ml = -Inf .* ones(ntrials)
    
    for im in 1:ntrials
        m.p = rand(Dirichlet(m.ncomponent, 1.0))
        m.μ = rand(m.ncomponent) .* (m.μ_ub .- m.μ_lb) .+ m.μ_lb
        m.σ = rand(m.ncomponent) .* (m.σ_ub .- m.σ_lb) .+ m.σ_lb
        m.fit=false
        
        fit!(m; kwargs...)
        pm[:, im] = m.p
        μm[:, im] = m.μ
        σm[:, im] = m.σ
        βm[:, im] = m.β
        ml[im] = m.ll
    end
    mlmax, imax = findmax(ml)
    or = sortperm(μm[:, imax])
    m.p = pm[or, imax]
    m.μ = μm[or, imax]
    m.σ = σm[or, imax]
    m.β = βm[:, imax]
    m.ll = mlmax
    return m
end

function EMtest(m::LGMModel, ntrials::Int=25, vtau::Vector{Float64}=[0.5;]; kwargs...)
    C0 = m.ncomponent
    C1 = C0 + 1
    n = m.n
    debuginfo = get(val_opts(kwargs), :debuginfo, false)
    
    initialize!(m)
    multipefit!(m, ntrials; kwargs...)
    C0 > 1 && (trand=asymptoticdistribution(m))
    debuginfo && println("loglikelihood of C0:", m.ll)
    
    ranef!(m)
    mingamma = minimum(m.gammaprediction) - 3 * std(m.gammaprediction)
    maxgamma = maximum(m.gammaprediction) + 3 * std(m.gammaprediction)
    an = decidepenalty(m.p, m.μ, m.σ, n)
    m1 = LGMModel(m.X, m.Y, m.groupindex, C1, taufixed=true, an=an)
    lrv = zeros(length(vtau), C0)
    for whichtosplit in 1:C0, i in 1:length(vtau)
        ind = [1:whichtosplit, whichtosplit:C0;]
        if C1==2
            fill!(m1.μ_lb, mingamma)
            fill!(m1.μ_ub, maxgamma)
        elseif C1>2
            mu_lb = [mingamma, (m.μ[1:(C0-1)] .+ m.μ[2:C0])./2;]
            mu_ub = [(m.μ[1:(C0-1)] .+ m.μ[2:C0])./2, maxgamma;]
            copy!(m1.μ_lb, mu_lb[ind])
            copy!(m1.μ_ub, mu_ub[ind])
        end
        m1.σ_lb = 0.25 .* m.σ[ind]
        m1.σ_ub = 2 .* m.σ[ind]
        m1.whichtosplit = whichtosplit
        m1.tau = vtau[i]
        m1.sn = m.σ[ind]
        m1.β = m.β
        
        multipefit!(m1, ntrials; kwargs...)
        m1.fit=false       
        fit!(m1, maxiteration=2, tol=0.0)
        debuginfo && println(whichtosplit, " ", vtau[i], "->", m1.ll)
        lrv[i, whichtosplit] = m1.ll
    end
    lr = maximum(lrv)
    if debuginfo
        println("lr=", lr)
    end
    Tvalue = 2*(lr - m.ll)
    if C0 == 1
        pvalue = 1 - cdf(Chisq(2), Tvalue)
    else
        pvalue = mean(trand .> Tvalue)
    end
    println("The Test Statistic is ", Tvalue)
    println("The p vlaue of the test is ", pvalue)
    return(Tvalue, pvalue)
end

function ranef!(m::LGMModel)
    ncomponent = m.ncomponent
    n = m.n
    ngh = m.ngh
    M = ngh*ncomponent
    N, J = size(m.X)

    for ix in 1:ngh, jcom in 1:ncomponent
        ixM = ix+ngh*(jcom-1)
        m.gammaM[ixM] = m.ghx[ix]*m.σ[jcom]*sqrt(2)+m.μ[jcom]
    end
    A_mul_B!(m.xb, m.X, m.β)
    integralweight!(m.Wim, m.X, m.Y, m.groupindex, m.gammaM, m.p, m.ghw, m.llN, m.llN2, m.xb, N, J, n, ncomponent, ngh)
    for i in 1:n
        for j in 1:M
            m.gammaprediction[i] += m.gammaM[j] * m.Wim[i,j]
        end
    end
    return m.gammaprediction
end

nobs(m::LGMModel) = m.n
model_response(m::LGMModel) = m.Y
coef(m::LGMModel) = m.β
ranefmixture(m::LGMModel) = MixtureModel(map((u, v) -> Normal(u, v), m.μ, m.σ), m.p)
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

function infomatrix(m::LGMModel; debuginfo::Bool=false, includelambda::Bool=true)

    N,J = size(m.X)
    n = m.n
    ngh=m.ngh
    C = m.ncomponent

    summat_beta = zeros(n, ngh*C, J)
    llnC = zeros(n, C)
    S_β = zeros(n, J)
    S_π = zeros(n, C-1)
    S_μσ = zeros(n, 2*C)
    if includelambda
        S_λ = zeros(n, 2*C)
    end 
    ml = zeros(n)
    A_mul_B!(m.xb, m.X, m.β)
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
                @inbounds ind = m.groupindex[i]::UInt32
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
            if includelambda
                S_λ[i, 2*kcom-1] = sumexp(m.Wim[i, ind], H3(m.gammaM[ind], m.μ[kcom], m.σ[kcom]))/ml[i]
                S_λ[i, 2*kcom] = sumexp(m.Wim[i, ind], H4(m.gammaM[ind], m.μ[kcom], m.σ[kcom]))/ml[i]
            end
        end
        for j in 1:J
            S_β[i, j] = sumexp(m.Wim[i,:], summat_beta[i, :, j])/ml[i]
        end
    end
    S_η = hcat(S_β, S_π, S_μσ)
    debuginfo && println(round(S_η[1:5,:], 5))
    debuginfo && println(round(sum(S_η, 1)./sqrt(n), 6))
    I_η = S_η'*S_η./n
    if includelambda
        I_η = S_η'*S_η./n
        I_λη = S_λ'*S_η./n
        I_λ = S_λ'*S_λ./n
        I_all = vcat(hcat(I_η, I_λη'), hcat(I_λη, I_λ))
    else
        I_all = I_η
    end
    if 1/cond(I_all) < eps(Float64)
        warn("Information Matrix is singular!")
        D, V = eig(I_all)
        debuginfo && println(D)
        tol2 = maximum(abs(D)) * 1e-14
        D[D.<tol2] = tol2
        I_all = V*diagm(D)*V'
    end
    return I_all
    #return inv(I_all)[1:J,1:J]./n
end

function vcov(m::LGMModel; debuginfo::Bool=false)
    J = size(m.X, 2)
    I_all = infomatrix(m, debuginfo=debuginfo)
    return inv(I_all)[1:J, 1:J] ./ m.n
end

function asymptoticdistribution(m::LGMModel; debuginfo::Bool=false, nrep::Int=10000)
    N,J = size(m.X)
    n = m.n
    ngh=m.ngh
    C = m.ncomponent
    if C == 1
        return rand(Chisq(2), nrep)
    end
    
    I_all = infomatrix(m, includelambda=true, debuginfo=debuginfo)
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
end

function predict(m::LGMModel, newX::Matrix{Float64}, newgroup::Vector{UInt32})
    !m.fit && warn("The model is not fitted!")
    N, J = size(m.X)
    N2, J2 = size(newX)
    J != J2 && error("Dimension Dismatch in new data")
    newxb = zeros(N2)
    newgamma = zeros(N2)
    A_mul_B!(newxb, newX, m.β)
    ranef!(m)
    overallmean = sum(m.μ .* m.p)
    for i in eachindex(newgroup)
        if newgroup[i] > m.n
            newgamma[i] = overallmean
        else
            newgamma[i] = m.gammaprediction[newgroup[i]]
        end
    end
    for i in 1:N2
        newxb[i] += newgamma[i]
    end
    logistic!(newxb)
end
""" 
    latentgmm(Y~x1+x2+(1|groupindex), fr, ncomponent)

 - `fr` is a `DataFrame` containing the field `Y`, `x1`, `x2` and `groupindex`
  - `ncomponent` is the number of components in random effects.
"""
function latentgmm(f::Formula, fr::AbstractDataFrame, ncomponent::Int; kwargs...)
    mf = ModelFrame(f,fr)
    X = ModelMatrix(mf).m
    Y = convert(Vector{Bool}, getindex(fr, f.lhs))
    retrms = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    if length(retrms) ≤ 0
        throw(ArgumentError("$f has no random-effects terms"))
    end
    # groupindex =  sort!([remat(e,mf.df) for e in retrms]; by = nlevs, rev = true) #or remat(retrms[1], mf.df).f
    groupindex = convert(Vector{UInt32}, getindex(mf.df, retrms[1].args[3]))
    m = LGMModel(X, Y, groupindex, ncomponent; kwargs...)
    return m
end

"""
    FDR(m, CNull)

 - `CNull` is a vector of numbers, indicating which components should be viewd as "null"; then the elements with smallest density ratio f_{null}/f_{alternative} will be rejected
 - `alphalevel` is the False Discovery Rate
"""
function FDR(m::LGMModel, C0::IntegerVector=[findmax(m.p)[2];]; alphalevel::Real=.05)
    !m.fit && warn("The model is not yet fitted!")
    n = m.n
    X, Y, groupindex, ncomponent = m.X, m.Y, m.groupindex, m.ncomponent
    ghw, ghx = m.ghw, m.ghx
    gammaM = m.gammaM
    Wim = m.Wim
    ngh = m.ngh
    M = m.ngh*ncomponent
    N, J = size(m.X)
    if maximum(C0) > ncomponent || minimum(C0) < 1
        error("C0 must be within 1 to number of components")
    end

    piposterior = zeros(n, ncomponent)
    clFDR = zeros(n)
    
    for ix in 1:ngh, jcom in 1:ncomponent
        ixM = ix+ngh*(jcom-1)
        m.gammaM[ixM] = ghx[ix]*m.σ[jcom]*sqrt(2)+m.μ[jcom]
    end

    A_mul_B!(m.xb, m.X, m.β)
    integralweight!(Wim, X, Y, groupindex, gammaM, m.p, ghw, m.llN, m.llN2, m.xb, N, J, n, ncomponent, ngh)
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
    if n0 == 0 
        println("Nothing is rejected")
    else
        println("The rejected groups are: ", order[1:n0])
        println("with FDR values: ", round(clFDR[order[1:n0]], 4))
    end
    
    return clFDR, order[1:n0]
end
