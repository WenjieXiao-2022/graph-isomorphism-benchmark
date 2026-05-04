using LinearAlgebra
using SparseArrays
using Statistics
using Hungarian

function perturb(A::AbstractMatrix, i::Int, k::Real)
    A = copy(A)
    A[i, i] += k
    return A
end

reorderVec(v::AbstractVector) = -reverse(vec(v))

function makeSignsConsistent(v1::AbstractVector, v2::AbstractVector; eps::Float64=1e-6)
    v1 = vec(v1); v2 = vec(v2)
    v1s = sort(v1)
    v2s = sort(v2)

    e1 = norm(v1s .- v2s)
    e2 = norm(v1s .- reorderVec(v2s))

    if e1 < eps && e2 < eps
        v1 = abs.(v1)
        v2 = abs.(v2)
        e = 0.0
    elseif e1 < eps && e2 > eps
        e = 0.0
    elseif e1 > eps && e2 < eps
        v2 = -v2
        e = 0.0
    else
        e = 1.0
    end
    return v1, v2, e
end

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

"""
    isIsomorphicRepeated(A, B; eps=1e-6, verbose=true)

Spectral-based isomorphism check with handling for repeated eigenvalues.

Returns `(b::Bool, P::Matrix{Int}, e::Float64)` where:
- `b` is `true` if total assignment cost `e < eps`
- `P` is a permutation matrix (n×n)
- `e` is the assignment cost `trace(P' * C)` (computed appropriately)
"""
function isIsomorphicRepeated(
    A::AbstractMatrix,
    B::AbstractMatrix;
    eps = 1e-6,
    verbose = true,
    time_ref = time(),
    time_limit = Inf,
)
    n = size(A, 1)
    @assert size(A) == size(B) "A and B must have the same size"

    # Eigendecomposition (use Symmetric to get real, sorted values/vectors)
    F1 = eigen(Matrix(A))
    F2 = eigen(Matrix(B))
    λ1, V1 = F1.values, F1.vectors   # λ ascending, V columns as eigenvectors
    λ2, V2 = F2.values, F2.vectors
    # λ1, V1 = eigs(A)
    # λ2, V2 = eigs(B)

    # Check eigenvalues match (necessary condition)
    e = norm(λ1 .- λ2)
    if e > eps
        return false, spzeros(Int, n, n), e
    end

    # Group (approximately) equal eigenvalues in λ1
    number = collect(1:n)    # group ids
    distinct = trues(n)       # assume distinct initially
    for i = 2:n
        if abs(λ1[i] - λ1[i-1]) < eps
            distinct[i] = false
            distinct[i-1] = false
            number[i] = number[i-1]   # same group id as previous
        end
    end

    # Build the cost matrix by blocks
    C = zeros(Float64, n, n)
    for k in unique(number)
        elapsed = time() - time_ref
        if elapsed > time_limit
            return nothing, nothing, nothing
        end

        idx = findall(==(k), number)      # indices in this eigengroup
        v1 = V1[:, idx]
        v2 = V2[:, idx]

        if all(distinct[idx]) && length(idx) == 1
            # Distinct eigenvalue: align signs and compare eigenvectors
            # Expect helpers: makeSignsConsistent(::AbstractVector, ::AbstractVector) -> (w1, w2, err)
            w1, w2, _ = makeSignsConsistent(view(v1, :, 1), view(v2, :, 1))
            Ck = costMatrix(w1, w2)
        else
            # Repeated eigenvalue(s): use eigenprojections (basis invariant)
            # Expect helper: costMatrixRepeated(::AbstractMatrix, ::AbstractMatrix) -> n×n matrix
            # ensure that the eigenvectors are orthogonal
            v1, _ = qr(v1)            
            v2, _ = qr(v2)
            # v1 = Matrix(Q1)
            # v2 = Matrix(Q2)
            @assert isapprox(norm(v1' * v1 - I), 0; atol=1e-12)
            @assert isapprox(norm(v2' * v2 - I), 0; atol=1e-12)
            E1 = v1 * v1'
            E2 = v2 * v2'
            Ck = costMatrixRepeated(E1, E2)
        end
        C .+= Ck
    end

    if verbose
        @info "Cost matrix: min=$(minimum(C))  max=$(maximum(C))  mean=$(mean(C))"
    end
    P = Hungarian.munkres(C)
    e = tr(P'*C)
    b = e < eps
    return b, P, e

end

function isIsomorphic(A::AbstractMatrix, B::AbstractMatrix; time_ref=time(), time_limit=Inf, eps=1e-6, verbose=true)
    n = size(A, 2)
    nBacktracking = 0

    # check if initial graphs are potentially isomorphic
    _, _, e0 = isIsomorphicRepeated(A, B; eps=eps, verbose=verbose, time_ref = time_ref, time_limit = time_limit)

    # hit the time limit
    if e0 == nothing
        return nothing, nothing, nBacktracking
    elseif e0 > eps
        # is not isomorphic
        return false, zeros(Int, n, n), nBacktracking
    end

    # perturb A and B and try to find assignment
    A0 = copy(A)
    B0 = copy(B)
    c = ones(Float64, n, n)
    P = zeros(Int, n, n)

    i = 1
    jStart = 1
    while i <= n
        c[i, :] .= 1.0  # reset row
        for j in jStart:n
            An = perturb(A0, i, i)
            Bn = perturb(B0, j, i)

            _, _, e = isIsomorphicRepeated(An, Bn; eps=eps, verbose=verbose, time_ref=time_ref, time_limit=time_limit)

            # hit the time limit
            if e == nothing
                return nothing, nothing, nBacktracking
            end

            c[i, j] = e

            if e < eps
                if verbose
                    println("assign $i -> $j")
                end
                A0 = An
                B0 = Bn
                P[i, j] = 1
                break
            end
        end

        cmin = minimum(c[i, :])
        if cmin > eps
            if i == 1
                return false, zeros(Int, n, n), nBacktracking
            else
                if verbose
                    println("backtracking ...")
                end
                nBacktracking += 1

                A0[i-1, i-1] = 0
                j = findfirst(x -> x == i-1, diag(B0))
                B0[j, j] = 0
                P[i-1, j] = 0

                jStart = j + 1
                i -= 2
            end
        else
            jStart = 1
        end

        i += 1
    end

    b = true
    e = norm(A - P * B * P')
    if e > eps
        error("GI: Wrong permutation matrix!")
    end
    return b, P, nBacktracking
end

