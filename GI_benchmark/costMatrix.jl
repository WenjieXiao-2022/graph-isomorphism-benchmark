function costMatrix(v::AbstractVector, w::AbstractVector)
    n = length(v)
    C = spzeros(Float64, n, n)
    for i in 1:n
        for j in 1:n
            C[i, j] = abs(v[i] - w[j])
        end
    end
    return C
end
