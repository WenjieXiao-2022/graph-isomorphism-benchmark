using LinearAlgebra
using SparseArrays
using JuMP
using HiGHS  # For linear programming
using Plots  # For plotting
using SCIP

"""
Frank-Wolfe solver for the graph isomorphism problem. A and B are the
adjacency matrices of the graphs G_A and G_B. The algorithm stops after
nIter iterations if no permutation matrix has been found.

See https://arxiv.org/abs/2311.16912 for details.
"""
function frank_wolfe_graph_isomorphism(A, B; nIter=10)
    # problem setup
    n = size(A, 1)
    vec_func(x) = reshape(x, n^2, 1)      # vectorize
    mat_func(x) = reshape(x, n, n)        # matricize
    cost(X) = norm(X*A - B*X, 2)         # cost function (Frobenius norm)
    penalty(X) = -tr(X'*X)               # penalty function
    
    # construct H matrix
    H = kron(A*A', I(n)) - 2*kron(A, B) + kron(I(n), B'*B)

    # row and column sum constraints
    C = zeros(2*n, n^2)
    for i = 1:n
        C[i, n*(i-1)+1:n*i] .= 1
        C[n+i, i:n:n^2] .= 1
    end
    d = ones(2*n)

    # all equality constraints
    A_all = [H; C]
    b_all = [zeros(n^2); d]

    # solve original quadratic optimization problem to obtain initial condition
    x0 = solve_quadprog(H, C, d, n)
    X0 = mat_func(x0)
    c0 = cost(X0)
    p0 = penalty(X0)

    plot_matrix(X0, 0)
    println("**** Iteration 0: c = $(round(c0, digits=2))  p = $(round(p0, digits=2))")

    # apply Frank-Wolfe algorithm to find permutation matrix
    x = copy(x0)
    X = copy(X0)
    
    for k = 1:nIter
        # solve linear program: min -x'y subject to constraints
        y = solve_linprog(-x, A_all, b_all, n)

        # line search
        ds = 0:0.01:1
        r = zeros(length(ds))
        for (j, d_val) in enumerate(ds)
            x_t = (1 - d_val)*x + d_val*y
            X_t = mat_func(x_t)
            r[j] = penalty(X_t)
        end
        
        j_min = argmin(r)
        gamma = ds[j_min]

        x = (1 - gamma)*x + gamma*y
        X = mat_func(x)
        c = cost(X)
        p = penalty(X)

        plot_matrix(X, k)
        println("**** Iteration $k: c = $(round(c, digits=2))  p = $(round(p, digits=2))  gamma = $(round(gamma, digits=6))")

        if abs(p + n) < 1e-4
            println("**** Close to permutation matrix ... Stopping now.")
            break
        end
    end
    
    return X
end

"""
Solve quadratic programming problem: min 0.5*x'*H*x subject to C*x = d, 0 <= x <= 1
"""
function solve_quadprog(H, C, d, n)
    model = Model(SCIP.Optimizer)
    set_silent(model)
    
    @variable(model, 0 <= x[1:n^2] <= 1)
    @objective(model, Min, 0.5 * x' * H * x)
    @constraint(model, C * x .== d)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(x)
    else
        @show termination_status(model)
        error("Quadratic programming solver failed")
    end
end

"""
Solve linear programming problem: min c'*x subject to A*x = b, 0 <= x <= 1
"""
function solve_linprog(c, A, b, n)
    model = Model(SCIP.Optimizer)
    set_silent(model)
    
    @variable(model, 0 <= x[1:n^2] <= 1)
    @objective(model, Min, c' * x)
    @constraint(model, A * x .== b)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(x)
    else
        error("Linear programming solver failed")
    end
end

"""
Plot matrix visualization
"""
function plot_matrix(X, iteration)
    p = heatmap(X, 
                color=:bone, 
                aspect_ratio=:equal, 
                title="Iteration $iteration",
                clims=(0, 1))
    display(p)
end

# Example usage:
# A = [0 1 0; 1 0 1; 0 1 0]  # Simple graph
# B = [0 0 1; 0 0 1; 1 1 0]  # Permuted version
# X = frank_wolfe_graph_isomorphism(A, B, 10) 