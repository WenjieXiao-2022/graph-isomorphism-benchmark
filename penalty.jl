using LinearAlgebra
using SparseArrays
using JuMP
using SCIP
using FrankWolfe

"""
Frank-Wolfe solver for the graph isomorphism problem. A and B are the
adjacency matrices of the graphs G_A and G_B. The algorithm stops after
nIter iterations if no permutation matrix has been found.

See https://arxiv.org/abs/2311.16912 for details.
"""
function frank_wolfe_graph_isomorphism(
    A,
    B;
    nIter = Inf,
    time_limit = 300,
)
    # problem setup
    n = size(A, 1)
    vec_func(x) = reshape(x, n^2, 1)      # vectorize
    mat_func(x) = reshape(x, n, n)        # matricize
    cost(X) = norm(X * A - B * X, 2)         # cost function (Frobenius norm)
    penalty(X) = -tr(X' * X)               # penalty function

    # construct H matrix
    H = kron(A * A', I(n)) - 2 * kron(A, B) + kron(I(n), B' * B)

    # @show eigmin(Symmetric(Matrix(H)))

    # row and column sum constraints
    C = zeros(2 * n, n^2)
    for i ∈ 1:n
        C[i, (n*(i-1)+1):(n*i)] .= 1
        C[n+i, i:n:(n^2)] .= 1
    end
    d = ones(2 * n)

    # all equality constraints
    A_all = [H; C]
    b_all = [zeros(n^2); d]

    t0 = time()
    # solve original quadratic optimization problem to obtain initial condition
    println("**** Solving original quadratic optimization problem...")
    x0, _ = solve_quadprog_fw_package(
        n,
        A,
        B;
        time_limit = time_limit,
    )

    X0 = mat_func(x0)
    c0 = cost(X0)
    p0 = penalty(X0)

    println("**** Iteration 0: c = $(round(c0, digits=2))  p = $(round(p0, digits=2))")

    if abs(p0 + n) < 1e-4
        println("**** Close to permutation matrix ... Stopping now.")
    end

    # apply Frank-Wolfe algorithm to find permutation matrix
    x = copy(x0)
    X = copy(X0)

    k = 1
    ds = 0:0.01:1
    status = 1
    while (time() - t0) < time_limit && k <= nIter


        time_left = time_limit - (time() - t0)
        # solve linear program: min -x'y subject to constraints
        y = solve_linprog(-x, A_all, b_all, n, time_left)

        if y === nothing
            println("*** The linear problem can not be solved...")
            status = 0
            break
        end

        # line search
        r = zeros(length(ds))
        for (j, d_val) in enumerate(ds)
            x_t = (1 - d_val) * x + d_val * y
            X_t = mat_func(x_t)
            r[j] = penalty(X_t)
        end

        j_min = argmin(r)
        gamma = ds[j_min]

        x = (1 - gamma) * x + gamma * y
        X = mat_func(x)
        c = cost(X)
        p = penalty(X)

        println(
            "**** Iteration $k: c = $(round(c, digits=2))  p = $(round(p, digits=2))  gamma = $(round(gamma, digits=6))",
        )

        if abs(p + n) < 1e-4
            println("**** Close to permutation matrix ... Stopping now.")
            break
        elseif gamma == 0.0
            println("**** Step size is 0.0 ... Stopping now.")
            return nothing, time()-t0
        end

        k += 1
    end

    if (time() - t0) > time_limit || status == 0
        println("**** Problem can not be solved...")
        return nothing, time() - t0
    end
    X = reshape(x, n, n)

    return X, time() - t0
end

"""
Solve linear programming problem: min c'*x subject to A*x = b, 0 <= x <= 1
"""
function solve_linprog(c, A, b, n, time_left)
    model = Model(SCIP.Optimizer)
    set_silent(model)

    if time_left < Inf
        # set a time limit (in seconds)
        MOI.set(model, MOI.TimeLimitSec(), time_left)
    end

    # # Memory limit in MiB (example: 1 GiB)
    # set_optimizer_attribute(model, "limits/memory", 1.0)

    @variable(model, 0 <= x[1:(n^2)] <= 1)
    @objective(model, Min, c' * x)
    @constraint(model, A * x .== b)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return value.(x)
    else
        @show termination_status(model)
        println("Linear programming solver failed")
        return nothing
    end
end
