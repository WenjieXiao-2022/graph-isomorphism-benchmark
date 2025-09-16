function isIsomorphic(A::AbstractMatrix, B::AbstractMatrix; eps=1e-6, verbose=true)
    n = size(A, 2)
    nBacktracking = 0

    # check if initial graphs are potentially isomorphic
    _, _, e0 = isIsomorphicRepeated(A, B; eps=eps, verbose=verbose)
    if e0 > eps
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

            _, _, e = isIsomorphicRepeated(An, Bn; eps=eps, verbose=verbose)
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

function perturb(A::AbstractMatrix, i::Int, k::Real)
    A = copy(A)
    A[i, i] += k
    return A
end
