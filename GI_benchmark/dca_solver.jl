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

    # # precompile
    # @info "Precompiling..."
    # x0, active_set =
    # solve_quadprog_fw_package(n, A, B; use_qua_as = use_qua_as, time_limit = 10)
    # @info "Loaded..."

    t0 = time()

    lmo = FrankWolfe.BirkhoffPolytopeLMO()

    α_max = 1e-2 * (sum(abs2, A) + sum(abs2, B)) / sqrt(n)
    β = 0.0

    # choose strong convexity parameter
    # μ = 1e-4 * (sum(abs2, A) + sum(abs2, B))
    μ = 0.0
    # α_max = 1.0
    # β = 0.0
    # μ = 0.0

    # # Precompute scaling based on x0
    # X0mat = reshape(x0, n, n)
    # R0 = A * X0mat - X0mat * B
    # comm0 = sum(abs2, R0)         # ||A X0 - X0 B||^2
    # xnorm0 = sum(abs2, X0mat)      # ||X0||^2

    # γ = 0.05                  # for example
    # α_max = γ * comm0 / xnorm0
    # β = 0.95
    # μ = 1e-3 * (comm0 / xnorm0)  # or your favorite small strong convexity

    # f(x) = ||X*A - B*X||^2 + μ ||X||^2  (convex, strongly convex)
    function f(x)
        X = reshape(x, n, n)
        R = X * A - B * X
        return sum(abs2, R) + μ * sum(abs2, X)
    end

    function grad_f!(storage, x)
        X = reshape(x, n, n)
        R = X * A - B * X
        # ∇_X ||X*A - B*X||_F^2 = 2(R*A' - B'*R)
        G = 2 * (R * A' - B' * R) + 2 * μ * X
        storage .= vec(G)
        return nothing
    end

    # g(x) = (μ + λ_t) ||X||^2  (convex)
    function g(x)
        X = reshape(x, n, n)
        # λ_t = α_max * (1 - β^(t))
        return α_max * sum(abs2, X)
    end

    function grad_g!(storage, x)
        X = reshape(x, n, n)
        # λ_t = α_max * (1 - β^(t))
        G = 2 * α_max * X
        storage .= vec(G)
        return nothing
    end

    # Objective function for verification
    function phi(x)
        return f(x) - g(x)
    end

    x0, active_set =
        solve_quadprog_fw_package(n, A, B; use_qua_as = use_qua_as, time_limit = time_limit)

    X0 = reshape(x0, n, n)
    println(
        " ||X0*A - B*X0  ||_F^2 at starting point: $(round(sum(abs2, X0 * A - B * X0), digits=8))",
    )

    x_current = x0
    X_final = nothing


    # x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n^2))

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


    # λ_t_final = α_max * (1 - β^iterations)
    # println("λ_t_final = ", λ_t_final)

    # @show length(active_set.atoms)
    # for atom in active_set.atoms
    #     atom_matrix = reshape(atom, n, n)
    #     if atom_matrix' * B * atom_matrix == A
    #         println("Found matching atom: ", atom)
    #     end
    # end

    if isapprox(X_final * A, B * X_final; atol=1e-6, rtol=1e-6) && sum(abs2, X_final) == n
        println("Found isomorphism!")
        return true, time()-t0
    else
       return false, time()-t0
    end

end
