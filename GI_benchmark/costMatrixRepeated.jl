function costMatrixRepeated(E1::AbstractMatrix, E2::AbstractMatrix)
    n = size(E1, 1)
    C = spzeros(Float64, n, n)

    # sort rows of E1 and E2
    E1s = [sort(E1[i, :]) for i in 1:n]
    E2s = [sort(E2[i, :]) for i in 1:n]

    # compare sorted vectors
    for i in 1:n
        for j in 1:n
            C[i, j] = norm(E1s[i] .- E2s[j])
        end
    end

    return C
end
