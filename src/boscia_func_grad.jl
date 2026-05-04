function build_function_gradient(A, B, n)
    R = zeros(n, n)

    B2 = Matrix(1.0B^2)
    A2 = Matrix(1.0A^2)
    BX = zeros(n, n)
    function f_acc2(x)
        X = reshape(x, n, n)
        mul!(R, X, A)
        mul!(R, B, X, -1, 1)
        return norm(R)^2
    end

    function f_acc2_check(x)
        X = reshape(x, n, n)
        mul!(R, X, A)
        mul!(R, B, X, -1, 1)
        res = norm(R)^2
        @assert res ≈ norm(X * A - B * X)^2
        return res
    end
    function grad_acc2!(storage, x)
        X = reshape(x, n, n)
        mul!(BX, B, X)
        S = reshape(storage, n, n)
        mul!(S, X, A2, 2, 0)
        mul!(S, BX, A, -4, 1)
        mul!(S, B2, X, 2, 1)
        return nothing
    end
    return f_acc2, grad_acc2!, f_acc2_check
end

function build_exp_function_gradient(A, B, n, tau)
    EA = exp(tau * Matrix(1.0A))
    EB = exp(tau * Matrix(1.0B))
    EA2 = EA^2
    EB2 = EB^2

    R = zeros(n, n)
    EBX = zeros(n, n)

    function f_exp(x)
        X = reshape(x, n, n)
        mul!(R, X, EA)
        mul!(R, EB, X, -1, 1)
        return norm(R)^2
    end

    function grad_exp!(storage, x)
        X = reshape(x, n, n)
        S = reshape(storage, n, n)
        mul!(EBX, EB, X)
        mul!(S, X, EA2, 2, 0)
        mul!(S, EBX, EA, -4, 1)
        mul!(S, EB2, X, 2, 1)
        return nothing
    end

    function f_exp_check(x)
        X = reshape(x, n, n)
        return norm(X * EA - EB * X)^2
    end

    return f_exp, grad_exp!, f_exp_check
end

function build_truncated_exp_function_gradient(A, B, n, tau, K)
    function truncated_matrix_exp(M, tau, K)
        n = size(M, 1)
        T = Matrix{Float64}(I, n, n)
        Mk = Matrix{Float64}(I, n, n)
        coeff = 1.0
        Mf = Matrix(1.0 * M)

        for k in 1:K
            Mk = Mk * Mf
            coeff *= tau / k
            T .+= coeff .* Mk
        end

        return T
    end

    EA = truncated_matrix_exp(A, tau, K)
    EB = truncated_matrix_exp(B, tau, K)

    R = zeros(n, n)
    T1 = zeros(n, n)
    T2 = zeros(n, n)

    function f_exp_trunc(x)
        X = reshape(x, n, n)
        mul!(R, X, EA)
        mul!(R, EB, X, -1, 1)
        return sum(abs2, R)
    end

    function grad_exp_trunc!(storage, x)
        X = reshape(x, n, n)
        S = reshape(storage, n, n)

        mul!(R, X, EA)
        mul!(R, EB, X, -1, 1)
        mul!(T1, R, EA')
        mul!(T2, EB', R)
        @. S = 2.0 * (T1 - T2)
        return nothing
    end

    function f_exp_trunc_check(x)
        X = reshape(x, n, n)
        return norm(X * EA - EB * X)^2
    end

    return f_exp_trunc, grad_exp_trunc!, f_exp_trunc_check
end
