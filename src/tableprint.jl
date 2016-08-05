
function latexprint(io::IO, ct::CoefTable; tableenvir::ASCIIString = "tabular", alignment::ASCIIString="", caption::ASCIIString="", centering::Bool=true)

    cols = ct.cols; rownms = ct.rownms; colnms = ct.colnms;
    nc = length(cols)
    nr = length(cols[1])
    if length(rownms) == 0
        rownms = [lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    rnwidth = max(4,maximum([length(nm) for nm in rownms]) + 1)
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    widths = [length(cn)::Int for cn in colnms]
    str = AbstractString[isa(cols[j][i], AbstractString) ? cols[j][i] :
        sprint(showcompact,cols[j][i]) for i in 1:nr, j in 1:nc]
    for j in 1:nc
        for i in 1:nr
            lij = length(str[i,j])
            if lij > widths[j]
                widths[j] = lij
            end
        end
    end
    widths .+= 1
    @printf(io, "\\begin{%s}{%s}\n", tableenvir, alignment)
    length(caption)>0 && @printf(io, "\n\\caption{%s}", caption)
    (centering && length(caption)>0) && println(io,"\n\\centering")
    @printf(io, "\n%s\n", "\\hline")

    println(io, " " ^ rnwidth * "&" *
            join([lpad(string(colnms[i]), widths[i]) for i = 1:nc], "&") * "\\\\")
    for i = 1:nr
        println(io, rownms[i] * "&" *
                join([lpad(string(str[i, j]), widths[j]) for j = 1:nc], "&") * "\\\\")
    end
    @printf(io, "\n%s", "\\hline")
    @printf(io, "\n\\end{%s}", tableenvir)
end

latexprint(ct; kwargs...) = latexprint(STDIN, ct; kwargs...)

function latexprint(io::IO, m::MixtureModel; rounding::Int = 3, surrounding::Bool=false)
    p = round(probs(m), rounding)
    μ = Float64[round(m.components[i].μ, rounding) for i in eachindex(p)]
    σ = Float64[round(m.components[i].σ, rounding) for i in eachindex(p)]
    surrounding && println(io, "\\[")
    println(io, join( [string(p[i]) * " Normal(" * string(μ[i]) * ", " * string(σ[i]) * "^2)" for i in eachindex(p)], " + "))
    surrounding && println(io, "\\]")
end
