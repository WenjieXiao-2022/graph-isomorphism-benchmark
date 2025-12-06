"""
    isIsomorphicRepeated(A, B; eps=1e-6, verbose=true)

Spectral-based isomorphism check with handling for repeated eigenvalues.

Returns `(b::Bool, P::Matrix{Int}, e::Float64)` where:
- `b` is `true` if total assignment cost `e < eps`
- `P` is a permutation matrix (n×n)
- `e` is the assignment cost `trace(P' * C)` (computed appropriately)
"""

using Arpack

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
