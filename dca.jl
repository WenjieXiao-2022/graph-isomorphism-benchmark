using FrankWolfe
using LinearAlgebra
using Hungarian

# DCA (Difference of Convex Algorithm) example with quadratic functions
# We want to minimize φ(x) = f(x) - g(x) over the Birkhoff polytope
# where f and g are convex quadratic functions.

# Problem setup: minimize φ(x) = f(x) - g(x) where:
# f(x) = || X A - B X ||_F^2 (convex quadratic)
# g(x) = || A X - X B ||_F^2  (convex quadratic)

function dca_solver(A, B, n; time_limit = 300.0, use_qua_as = false)

    t0 = time()

    lmo = FrankWolfe.BirkhoffPolytopeLMO()

    λ = 1e-2  # Regularization parameter for g(x)
    
    # f(x) = ||X*A - B*X||^2
    function f(x)
        X = reshape(x, n, n)
        R = X * A - B * X
        return sum(abs2, R)
    end

    function grad_f!(storage, x)
        X = reshape(x, n, n)
        R = X * A - B * X
        G = 2 * (R * A' - B' * R)
        storage .= vec(G)
        return nothing
    end

    # g(x) = λ * ||X||^2  (convex)
    function g(x)
        X = reshape(x, n, n)
        return λ * sum(abs2, X)
    end

    function grad_g!(storage, x)
        X = reshape(x, n, n)
        G = 2 * λ * X
        storage .= vec(G)
        return nothing
    end

    # Objective function for verification
    function phi(x)
        return f(x) - g(x)
    end

    x0, active_set = solve_quadprog_fw_package(
        n,
        A,
        B;
        time_limit = time_limit,
    )

    X0 = reshape(x0, n, n)
    println(
        "||X0*A - B*X0  ||_F^2 at starting point: $(round(sum(abs2, X0 * A - B * X0), digits=8))",
    )

    x_current = x0
    X_final = nothing

    println("Running Enhanced DcAFW example...")
    println("="^60)
    println("Problem: minimize φ(x) = f(x) - g(x) over the Birkhoff polytope")
    println("Dimension: n = $n")
    println("Feasible region: the Birkhoff Polytope")
    println("="^60)

    # For quadratic functions, finding theoretical optimum is more complex
    println("Note: Unlike linear objectives, quadratic difference-of-convex problems")
    println("      may have multiple local minima. DCA finds a stationary point.")
    println()

    time_left = time_limit - (time()-t0)

    if time_left <= 0.0
        return false, time()-t0
    end

    @time x_final,
    primal_final,
    dca_gap_final,
    iterations,
    status,
    traj_data = FrankWolfe.dca_fw(
        f,
        grad_f!,
        g,
        grad_g!,
        lmo,
        x_current,
        max_iteration = 100000, # Outer iterations
        max_inner_iteration = 10000, # Inner iterationsx
        epsilon = 1e-5, # Tolerance for DCA gap
        line_search = FrankWolfe.Secant(),
        verbose = true,
        verbose_inner = false,
        trajectory = true,
        print_iter = 1,
        memory_mode = FrankWolfe.InplaceEmphasis(),
        use_corrective_fw = true,
        use_dca_early_stopping = true,
        boosted = false,
        timeout = time_left,
    )

    X_final = reshape(x_final, n, n)
    residual = X_final * A - B * X_final

    println("\n" * "="^60)
    println("DcAFW Algorithm Results")
    println("="^60)
    println("Convergence:")
    println("  Outer iterations completed: $iterations")
    println("  Final objective value φ(x): $(round(primal_final, digits=8))")
    println("  Verification φ(x) = f(x) - g(x): $(round(phi(x_final), digits=8))")
    println("  Final DCA gap: $(round(dca_gap_final, digits=8))")
    println()
    println("Solution properties:")

    println(" g(x) := ||X||_F^2 at solution: $(round(sum(abs2, reshape(x_final, n, n)), digits=8))")
    println(" f(X) = ||X*A - B*X||_F^2 at solution: $(round(sum(abs2, residual)))")

    if isapprox(X_final * A, B * X_final; atol=1e-6, rtol=1e-6) && sum(abs2, X_final) == n
        println("Found isomorphism!")
        return true, time()-t0
    else
       return false, time()-t0
    end

end
