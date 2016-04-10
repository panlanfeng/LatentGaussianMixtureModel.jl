

type ModelTable
    mat::Matrix
    colnms::Vector
    rownms::Vector
    pvalcol::Integer
    integercols::Vector{Int}
    function ModelTable(mat::Matrix,colnms::Vector,rownms::Vector,pvalcol::Int=0, integercols::Vector{Int}=[0;])
        nr,nc = size(mat)
        0 <= pvalcol <= nc || error("pvalcol = $pvalcol should be in 0,...,$nc]")
        all(0 .<= integercols .<= nc) || error("pvalcol = $pvalcol should be in 0,...,$nc]")
        length(colnms) in [0,nc] || error("colnms should have length 0 or $nc")
        length(rownms) in [0,nr] || error("rownms should have length 0 or $nr")
        new(mat,colnms,rownms,pvalcol, integercols)
    end
end

function show(io::IO, ct::ModelTable)
    mat = ct.mat; nr,nc = size(mat); rownms = ct.rownms; colnms = ct.colnms; pvc = ct.pvalcol; ics = ct.integercols;
    if length(rownms) == 0
        rownms = [lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    rnwidth = max(4,maximum([length(nm) for nm in rownms]) + 1)
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    widths = [length(cn)::Int for cn in colnms]
    str = [sprint(showcompact,mat[i,j]) for i in 1:nr, j in 1:nc]
    if pvc != 0                         # format the p-values column
        for i in 1:nr
            str[i,pvc] = StatsBase.format_pvc(mat[i,pvc])
        end
    end
    for ic in ics
        if ic != 0
            for i in 1:nr
                str[i, ic] = @sprintf("%i", mat[i, ic])
            end            
        end
    end
    for j in 1:nc
        for i in 1:nr
            lij = length(str[i,j])
            if lij > widths[j]
                widths[j] = lij
            end
        end
    end
    widths .+= 1
    println(io," " ^ rnwidth *
            join([lpad(string(colnms[i]), widths[i]) for i = 1:nc], ""))
    for i = 1:nr
        print(io, rownms[i])
        for j in 1:nc
            print(io, lpad(str[i,j],widths[j]))
        end
        println(io)
    end
end

function latexprint(io::IO, ct::ModelTable; tableenvir::ASCIIString = "tabular", alignment::ASCIIString="", caption::ASCIIString="", centering::Bool=true)

    mat = ct.mat; nr,nc = size(mat); rownms = ct.rownms; colnms = ct.colnms; pvc = ct.pvalcol; ics = ct.integercols;
    if length(rownms) == 0
        rownms = [lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    rnwidth = max(4,maximum([length(nm) for nm in rownms]) + 1)
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    widths = [length(cn)::Int for cn in colnms]
    str = [sprint(showcompact,mat[i,j]) for i in 1:nr, j in 1:nc]
    if pvc != 0                         # format the p-values column
        for i in 1:nr
            str[i,pvc] = format_pvc(mat[i,pvc])
        end
    end
    for ic in ics
        if ic != 0
            for i in 1:nr
                str[i, ic] = @sprintf("%i", mat[i, ic])
            end            
        end
    end
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

function latexprint(io::IO, m::MixtureModel; rounding::Int = 3)
    p = round(probs(m), rounding)
    μ = Float64[round(m.components[i].μ, rounding) for i in eachindex(p)]
    σ = Float64[round(m.components[i].σ, rounding) for i in eachindex(p)]
    println(io, "\\[")
    println(io, join( [string(p[i]) * " Normal(" * string(μ[i]) * ", " * string(σ[i]) * "^2)" for i in eachindex(p)], " + "))
    println(io, "\\]")
end
