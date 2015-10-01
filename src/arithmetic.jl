

exp!(x::Vector{Float64}) = Yeppp.exp!(x, x)
log!(x::Vector{Float64}) = Yeppp.log!(x, x)

function sumexp{T<:Real}(x::AbstractArray{T})
    isempty(x) && return -Inf
    u = maximum(x)
    s = 0.
    for i in 1:length(x)
        @inbounds s += exp(x[i] - u)
    end
    s * exp(u)
end
function ratiosumexp!{T<:Real}(x::AbstractArray{T}, coef::AbstractArray{T}, s::AbstractArray{T}, ncomponent::Int)
    #length(x) != length(coef) && error("Length should be the same!")
    #isempty(x) && return -Inf
    u = maximum(x)
    for i in 1:ncomponent
        @inbounds s[i] = coef[i]*exp(x[i] - u)
    end
    divide!(s, s, sum(s), ncomponent)
    #s ./ sum(s) #* exp(u)
    nothing
end

function add!(res::Vector{Float64}, x::Vector{Float64}, y::Float64, n::Int64=length(x))
   for i in 1:n
       @inbounds res[i] = x[i] + y
   end
   nothing
end

add!(res::Vector{Float64}, x::Float64, y::Vector{Float64}, n::Int64=length(y)) = add!(res, y, x, n)

add!(x::Vector{Float64}, y::Float64, n::Int64=length(x))=add!(x, x, y, n)
add!(x::Float64, y::Vector{Float64}, n::Int64=length(y))=add!(y, y, x, n)

plusone!(res::Vector{Float64}, x::Vector{Float64}, n::Int64=length(x)) = add!(res, x, 1.0, n)
plusone!(x::Vector{Float64}, n=length(x)) = plusone!(x, x, n)


function divide!(res::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}, n::Int64=length(x))
    for i in 1:n
        @inbounds res[i] = x[i] / y[i]
    end
    nothing
end

function divide!(res::Vector{Float64}, x::Vector{Float64}, y::Float64, n::Int64=length(x))
    tmp = 1.0/y
    for i in 1:n
        @inbounds res[i] = x[i] * tmp
    end
    nothing
end

function divide!(res::Vector{Float64}, x::Float64, y::Vector{Float64}, n::Int64=length(y))
    for i in 1:n
        @inbounds res[i] = x / y[i]
    end
    nothing
end
divide!(x::Vector{Float64}, y::Vector{Float64}, n::Int64=length(x)) = divide!(x, x, y, n)
divide!(x::Float64, y::Vector{Float64}, n::Int64=length(y)) = divide!(y, x, y, n)
divide!(x::Vector{Float64}, y::Float64, n::Int64=length(x)) = divide!(x, x, y, n)



rcp!(res::Vector{Float64}, x::Vector{Float64}, n::Int64=length(x)) = divide!(res, 1.0, x, n)
rcp!(x::Vector{Float64}, n::Int64=length(x))=rcp!(x, x, n)



function multiply!(res::Vector{Float64}, x::Vector{Float64}, y::Float64, n::Int64=length(x))
    for i in 1:n
        @inbounds res[i] = x[i] * y
    end
    nothing
end

function multiply!(res::Vector{Float64}, x::Float64, y::Vector{Float64}, n::Int64=length(y))
    for i in 1:n
        @inbounds res[i] = x * y[i]
    end
    nothing
end

multiply!(x::Float64, y::Vector{Float64}, n::Int64=length(y)) = multiply!(y, x, y, n)
multiply!(x::Vector{Float64}, y::Float64, n::Int64=length(x)) = multiply!(x, x, y, n)


function negate!(res::Vector{Float64}, x::Vector{Float64}, n::Int64=length(x))
   for i in 1:n
       @inbounds res[i] = -x[i]
   end
   nothing
end
negate!(x::Vector{Float64}, n::Int64=length(x)) = negate!(x, x, n)

function negateiffalse!(x::Vector{Float64}, y::AbstractArray{Bool, 1}, n::Int64=length(x))
    for i in 1:n
        @inbounds x[i] = ifelse(y[i], x[i], -x[i])
    end
    nothing
end

function negateiftrue!(x::Vector{Float64}, y::AbstractArray{Bool, 1}, n::Int64=length(x))
    for i in 1:n
        @inbounds x[i] = ifelse(y[i], -x[i], x[i])
    end
    nothing
end

function log1p!(x::Vector{Float64}, n::Int64=length(x))
    plusone!(x, n)
    log!(x, x)
    nothing
end

function x1x!(x::Vector{Float64}, n::Int64=length(x))
    # n = length(x)
    for i in 1:n
        @inbounds x[i] = x[i] / (1.0 + x[i])
    end
    nothing
end


# -log(1+exp(-xy))
function loglogistic!(x::Vector{Float64}, y::AbstractArray{Bool, 1}, n=length(x))
    negateiftrue!(x, y, n)
    exp!(x, x)
    add!(x, x, 1.0)
    log!(x, x)
    negate!(x, x, n)
    nothing
end

# 1/(1+exp(-x))
function logistic!(x::Vector{Float64})
    n = length(x)
    negate!(x)
    exp!(x, x)
    add!(x, x, 1.0)
    rcp!(x, n)
    nothing
end

# 1+exp(-x*y)
function rcplogistic!(x::Vector{Float64}, y::AbstractArray{Bool, 1})
    n = length(x)
    assert(length(y) == n)
    negateiftrue!(x, y, n)
    exp!(x, x)
    add!(x, x, 1.0)
    nothing
end

function relocate!(res::Vector{Float64}, ga::Vector{Float64}, facility::Vector{Int64}, N::Int)
    for i in 1:N
        @inbounds res[i] = ga[facility[i]]
    end
    nothing
end


function sumby!(r::AbstractArray, y::AbstractArray, x::IntegerArray, levels::IntUnitRange)
	k = length(levels)
	length(r) == k || raise_dimerror()

	m0 = levels[1]
	m1 = levels[end]
	b = m0 - 1

	@inbounds for i in 1 : length(x)
		xi = x[i]
		if m0 <= xi <= m1
			r[xi - b] += y[i]
		end
	end
	#return r
end
function sumsqby!(r::AbstractArray, y::AbstractArray, x::IntegerArray, levels::IntUnitRange)
	k = length(levels)
	length(r) == k || raise_dimerror()

	m0 = levels[1]
	m1 = levels[end]
	b = m0 - 1

	@inbounds for i in 1 : length(x)
		xi = x[i]
		if m0 <= xi <= m1
			r[xi - b] += abs2(y[i])
		end
	end
	#return r
end
