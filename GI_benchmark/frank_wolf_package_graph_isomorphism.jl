using Boscia
using FrankWolfe
using Test
using Random
using SCIP
using HiGHS
using JuMP
using LinearAlgebra
using Distributions
using MathOptInterface: MathOptInterface
const MOI2 = MathOptInterface

# include(
#     "/home/htc/wexiao/project/graph_isomorphism/BenchmarkProblems/julia_benchmarkProblems/dca_solver.jl",
# )

"""
Solve quadratic programming problem: min 0.5*x'*H*x subject to C*x = d, 0 <= x <= 1
"""
function solve_quadprog_fw_package(n, A, B; use_qua_as = false, time_limit = 300)

    # function f(x)
    # 	return 0.5 * x' * H * x
    # end

    # function grad!(storage, x)
    # 	storage .= H * x
    # 	return storage
    # end

    function f(x)
        X = reshape(x, n, n)
        R = X * A - B * X
        return sum(abs2, R)            # == ‖R‖_F^2
    end

    function grad!(storage, x)
        X = reshape(x, n, n)
        grad_matrix = 2 * (X * A - B * X) * A' - 2 * B' * (X * A - B * X)
        storage .= vec(grad_matrix)
    end

    lmo_pre = FrankWolfe.BirkhoffPolytopeLMO()
    lmo = FrankWolfe.BirkhoffPolytopeLMO()


    x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n^2))

    if !use_qua_as
        active_set_pre = FrankWolfe.ActiveSet([(1.0, x0)])
        active_set = FrankWolfe.ActiveSet([(1.0, x0)])

        # # precompile
        # _, _, _, _, _, _, _ =
        # FrankWolfe.blended_pairwise_conditional_gradient(
        #     f,
        #     grad!,
        #     lmo_pre,
        #     active_set_pre,
        #     line_search = FrankWolfe.Secant(),
        #     print_iter = 100,
        #     epsilon = 1e-5,
        #     verbose = true,
        #     max_iteration = Inf,
        #     timeout = 10,
        # )


        x, _, _, _, status, _, active_set =
            FrankWolfe.blended_pairwise_conditional_gradient(
                f,
                grad!,
                lmo,
                active_set,
                line_search = FrankWolfe.Secant(),
                print_iter = 100,
                epsilon = 1e-5,
                verbose = true,
                max_iteration = Inf,
                timeout = time_limit,
            )
        # x, _, _, _, status, _ =
        #     FrankWolfe.decomposition_invariant_conditional_gradient(
        #         f,
        #         grad!,
        #         lmo,
        #         x0,
        #         line_search = FrankWolfe.Secant(),
        #         print_iter = 100,
        #         epsilon = 1e-5,
        #         verbose = true,
        #         max_iteration = Inf,
        #         timeout = time_limit,
        #     )
    else

        active_set_pre = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, x0)], grad!)
        active_set = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, x0)], grad!)

        # # precompile 
        # x, _, _, _, status, _, active_set =
        #     FrankWolfe.blended_pairwise_conditional_gradient(
        #         f,
        #         grad!,
        #         lmo_pre,
        #         active_set_pre,
        #         line_search = FrankWolfe.Secant(),
        #         print_iter = 100,
        #         verbose = true,
        #         epsilon = 1e-5,
        #         max_iteration = Inf,
        #         timeout = 10,
        #     )


        x, _, _, _, status, _, active_set =
            FrankWolfe.blended_pairwise_conditional_gradient(
                f,
                grad!,
                lmo,
                active_set,
                line_search = FrankWolfe.Secant(),
                print_iter = 100,
                verbose = true,
                epsilon = 1e-5,
                max_iteration = Inf,
                timeout = time_limit,
            )
    end

    return x, active_set     
    
end

function fw_package_graph_isomorphism(
    A,
    B;
    variant = "fw",
    time_limit = 300,
    use_qua_as = false,
)
    function f(x)
        return -0.5 * x' * x
    end

    function grad!(storage, x)
        @. storage = -x
    end

    n = size(A, 1)

    # construct H matrix
    H = kron(A' * A, I(n)) - 2 * kron(A, B) + kron(I(n), B' * B)

    # @show eigmin(Symmetric(Matrix(H)))
    # @show issymmetric(H)

    # row and column sum constraints
    C = zeros(2 * n, n^2)
    for i = 1:n
        C[i, (n*(i-1)+1):(n*i)] .= 1
        C[n+i, i:n:(n^2)] .= 1
    end
    d = ones(2 * n)

    function build_lmo(n, C, H, d)
        o = SCIP.Optimizer()
        MOI2.set(o, MOI2.Silent(), true)
        MOI2.empty!(o)
        x = MOI2.add_variables(o, n^2)

        # Equality constraints: Cx = d
        for i = 1:size(C, 1)
            terms = [MOI2.ScalarAffineTerm(C[i, j], x[j]) for j = 1:length(x)]
            func = MOI2.ScalarAffineFunction(terms, 0.0)
            MOI2.add_constraint(o, func, MOI2.EqualTo(d[i]))
        end

        # Equality constraints: Hx = 0
        for i = 1:size(H, 1)
            terms = [MOI2.ScalarAffineTerm(float(H[i, j]), x[j]) for j = 1:length(x)]
            func = MOI2.ScalarAffineFunction(terms, 0.0)
            MOI2.add_constraint(o, func, MOI2.EqualTo(0.0))
        end

        for j = 1:length(x)
            MOI2.add_constraint(o, x[j], MOI2.GreaterThan(0.0))
            MOI2.add_constraint(o, x[j], MOI2.LessThan(1.0))
        end


        return FrankWolfe.MathOptLMO(o)

    end

    status = []
    tot_time = 0

    function make_callback(A, B, status)
        return function callback(state, active_set; kwargs...)
            n = size(A, 1)
            X = reshape(state.x, n, n)
            p = -tr(X' * X)

            if state.gamma == 0.0
                push!(status, "Local Optimum")
            	return false
            end

            # @show B ≈  X * A * X'
            # @show state.x' * H * state.x

            if abs(p + n) < 1e-4
                # @show abs(p+n)
                println("Permutation matrix is found!")
                push!(status, "Optimal")
                return false
            else
                return true
            end
        end
    end

    lmo = build_lmo(n, C, H, d)
    callback = make_callback(A, B, status)
    t0 = time()
    println("**** Solving original quadratic optimization problem...")
    x0, active_set = solve_quadprog_fw_package(
        n,
        A,
        B;
        use_qua_as = use_qua_as,
        time_limit = time_limit,
    )

    if x0 == nothing
        return nothing, "TIMEOUT", time_limit + 1
    end
    X0 = reshape(x0, n, n)

    time_left = time_limit - (time() - t0)
    if variant == "fw"
        x, _, _, _, _, traj_data, _ = FrankWolfe.away_frank_wolfe(
            f,
            grad!,
            lmo,
            x0,
            line_search = FrankWolfe.GeneralizedAgnostic(),
            print_iter = 1000,
            verbose = true,
            trajectory = true,
            callback = callback,
            epsilon = -Inf,
            timeout = time_left,
            max_iteration = Inf,
            lazy = false,
        )
    elseif variant == "dicg"
        x, v, primal, dual_gap, trajectory =
            FrankWolfe.decomposition_invariant_conditional_gradient(
                f,
                grad!,
                lmo,
                x0,
                line_search = FrankWolfe.Secant(),
                print_iter = 100,
                verbose = true,
                trajectory = true,
                callback = callback,
                epsilon = -Inf,
                timeout = time_limit,
            )
    end

    tot_time = traj_data[end][end]
    @show status[end]
    return x, status[end], tot_time
end


# # Example usage:
# A = [0 1 0; 1 0 1; 0 1 0]  # Simple graph
# B = [0 0 1; 0 0 1; 1 1 0]  # Permuted version
# X = fw_graph_isomorphism(A, B)
# @assert A == X' * B * X


