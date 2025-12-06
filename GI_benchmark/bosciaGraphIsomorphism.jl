using Boscia
using SparseArrays
using LinearAlgebra
using Bonobo
using FrankWolfe
using MAT
using Random
using CSV
using DataFrames
using CombinatorialLinearOracles
const CLO = CombinatorialLinearOracles

include(
    "./dca_solver.jl",
)

function random_k_neighbor_matrix(
    tree::Bonobo.BnBTree,
    blmo::Boscia.TimeTrackingLMO,
    x,
    k::Int,
    use_mip = false,
)
    P = tree.incumbent_solution.solution
    n0 = size(P, 1)
    n = Int(sqrt(n0))
    P = reshape(P, n, n)
    new_P = copy(P)

    Ps = []

    for _ = 1:k
        # Pick two distinct rows
        i, j = rand(1:n, 2)
        while i == j
            j = rand(1:n)
        end

        # Find 1s in each row
        col_i = findfirst(x -> x == 1, new_P[i, :])
        col_j = findfirst(x -> x == 1, new_P[j, :])

        # Swap the 1s across columns
        new_P[i, col_i] = 0
        new_P[i, col_j] = 1
        new_P[j, col_j] = 0
        new_P[j, col_i] = 1

        new_p = use_mip ? vec(new_P) : sparsevec(vec(new_P))# Convert to proper SparseVector
        push!(Ps, new_p)
    end

    return Ps, false
end

function get_initial_point(A, B, n; FW = true, time_limit = 300)
    if FW
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

        lmo = FrankWolfe.BirkhoffPolytopeLMO()

        x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n^2))

        active_set = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, x0)], grad!)

        x, _, _, _, _ = FrankWolfe.blended_pairwise_conditional_gradient(
            f,
            grad!,
            lmo,
            x0,
            line_search = FrankWolfe.Secant(),
            print_iter = 100,
            epsilon = 1e-4,
            verbose = true,
            max_iteration = Inf,
            timeout = time_limit,
        )

        # x, _, _, _, _, _ = FrankWolfe.blended_pairwise_conditional_gradient(
        #     f,
        #     grad!,
        #     lmo,
        #     active_set,
        #     line_search = FrankWolfe.Secant(),
        #     print_iter = 100,
        #     verbose = true,
        #     epsilon = 1e-4,
        #     max_iteration = Inf,
        #     timeout = time_limit,
        # )

    else
        x, active_set = dca_solver(A, B, n; time_limit = time_limit)
    end
    return x, active_set
end


function boscia_run(
    A,
    B;
    solver = "boscia_bpcg",
    time_limit = 300,
    verbose = true,
    print_iter = 10,
    fw_iter = 500,
    cut = false,
    fw_verbose = false,
    fw_epsilon = 1e-2,
    lazy = true,
    starting_point_dca = false,
    starting_point_fw = false,
    regularized = false,
    use_depth = false,
    is_graph_matching = false,
    is_modified = false,
    favor_right = true,
    is_frac32 = false,
)
    n = size(A, 1)
    μ = 1e-2

    function f(x)
        X = reshape(x, n, n)
        R = X * A - B * X
        return sum(abs2, R)  # == ‖R‖_F^2 
    end

    function grad!(storage, x)
        X = reshape(x, n, n)
        R = X * A - B * X                     # residual

        grad_matrix = 2 * (R * A' - B' * R)
        storage .= vec(grad_matrix)
    end

    function f_regularized(x)
        X = reshape(x, n, n)
        R = X * A - B * X
        return sum(abs2, R) + μ / 2 * sum(abs2, X)
    end

    function grad_regularized!(storage, x)
        X = reshape(x, n, n)
        R = X * A - B * X                     # residual
        grad_matrix = 2 * (R * A' - B' * R) + μ * X
        storage .= vec(grad_matrix)

    end

    function f_frac32(x)
        X = reshape(x, n, n)
        R = X * A - B * X
        return (sum(abs2, R))^(3 / 4) # == ‖R‖_F^(3/2)
    end

    function grad_frac32!(storage, x)
        X = reshape(x, n, n)
        R = X * A - B * X                     # residual
        # factor = (3/2) * g^(-1/4) = (3/2) * ||R||_F^(-1/2)
        factor = (3 / 2) * sum(abs2, R)^(-1 / 4)
        # Base gradient from ||R||_F^(3/2)
        grad_matrix = factor .* (R * A' .- B' * R)
        storage .= vec(grad_matrix)
    end

    function build_branch_callback()
        return function (tree, node, vidx::Int)
            x = Bonobo.get_relaxed_values(tree, node)
            primal = tree.root.problem.f(x)
            lower_bound = primal - node.dual_gap
            optimal_val = regularized ? μ / 2 * n : 0.0
            if lower_bound > optimal_val + eps()
                println("No need to branch here. Node lower bound already positive.")
            end
            valid_lower = lower_bound > optimal_val + eps()
            return valid_lower, valid_lower
        end
    end

    function build_tree_callback()
        return function (
            tree,
            node;
            worse_than_incumbent = false,
            node_infeasible = false,
            lb_update = false,
        )
            optimal_val = regularized ? μ / 2 * n : 0.0
            if isapprox(tree.incumbent, optimal_val, atol = eps())
                tree.root.problem.solving_stage = Boscia.USER_STOP
                println("Optimal solution found.")
            end
            if Boscia.tree_lb(tree::Bonobo.BnBTree) > optimal_val + eps()
                tree.root.problem.solving_stage = Boscia.USER_STOP
                println("Tree lower bound already positive. No solution possible.")
            end
        end
    end

    blmo = CLO.BirkhoffLMO(n, collect(1:(n^2)))
    blmo_precompile = CLO.BirkhoffLMO(n, collect(1:(n^2)))
    k = Int(round(sqrt(n)))
    swap_heu = Boscia.Heuristic(
        (tree, blmo, x) -> random_k_neighbor_matrix(tree, blmo, x, k, false),
        1.0,
        :swap,
    )

    if use_depth
        favor_children = favor_right ? "right" : "left"
        println("Boscia is using DepthFirstStrategy favoring $(favor_children) children...")
    end

    if starting_point_dca || starting_point_fw
        FW = starting_point_fw ? true : false
        _, active_set_qua_precompile = get_initial_point(A, B, n; FW = FW, time_limit = 10)
        _, active_set_qua = get_initial_point(A, B, n; FW = FW, time_limit = 100)

        if typeof(active_set_qua_precompile).name.wrapper == FrankWolfe.ActiveSet
            active_set_precompile = active_set_qua_precompile
            active_set = active_set_qua
        else
            # Precompile version
            tuple_values_pre = collect(
                zip(active_set_qua_precompile.weights, active_set_qua_precompile.atoms),
            )
            active_set_precompile = FrankWolfe.ActiveSet(tuple_values_pre)

            # Main version
            tuple_values = collect(zip(active_set_qua.weights, active_set_qua.atoms))
            active_set = FrankWolfe.ActiveSet(tuple_values)
        end
    end

    # default is set to BPCG with lazification
    if contains(solver, "dicg")
        variant = Boscia.DecompositionInvariantConditionalGradient()
        lazy = false
    elseif contains(solver, "fw")
        variant = Boscia.StandardFrankWolfe()
        lazy = false
    else
        variant = Boscia.BlendedPairwiseConditionalGradient()
        lazy = true
    end
    # Precompile
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = true
    settings.branch_and_bound[:print_iter] = print_iter

    if !(is_graph_matching && !is_modified)
        @info "Activating callback..."
        settings.branch_and_bound[:bnb_callback] = build_tree_callback()
        settings.branch_and_bound[:branch_callback] = build_branch_callback()
    end
    settings.branch_and_bound[:time_limit] = 10

    use_depth ?
    settings.branch_and_bound[:traverse_strategy] = Boscia.DepthFirstSearch(favor_right) :
    nothing

    settings.heuristic[:custom_heuristics] = [swap_heu]
    settings.frank_wolfe[:variant] = variant
    settings.frank_wolfe[:line_search] = FrankWolfe.Secant()
    settings.frank_wolfe[:lazy] = lazy
    settings.frank_wolfe[:max_fw_iter] = fw_iter
    settings.frank_wolfe[:fw_verbose] = false
    if regularized
        # settings.tightening[:sharpness_constant] = sqrt(2 / μ)
        settings.tightening[:sharpness_constant] = 1
        settings.tightening[:sharpness_exponent] = 1 / 2
        # settings.tightening[:strong_convexity] = μ
    end

    if starting_point_dca || starting_point_fw
        settings.domain[:active_set] = active_set_precompile
    end

    if regularized
        println("Using regularized formulation...")
        f_run = f_regularized
        grad_run! = grad_regularized!
    elseif is_frac32
        println("Using p=1.5 formulation...")
        f_run = f_frac32
        grad_run! = grad_frac32!
    else
        println("Using quadratic formulation...")
        f_run = f
        grad_run! = grad!
    end

    x, _, result = Boscia.solve(f_run, grad_run!, blmo_precompile, settings = settings)

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = verbose
    settings.branch_and_bound[:print_iter] = print_iter

    if !(is_graph_matching && !is_modified)
        @info "Activating callback..."
        settings.branch_and_bound[:bnb_callback] = build_tree_callback()
        settings.branch_and_bound[:branch_callback] = build_branch_callback()
    end

    use_depth ?
    settings.branch_and_bound[:traverse_strategy] = Boscia.DepthFirstSearch(favor_right) :
    nothing

    settings.branch_and_bound[:time_limit] = time_limit
    settings.heuristic[:custom_heuristics] = [swap_heu]
    settings.frank_wolfe[:variant] = variant
    settings.frank_wolfe[:line_search] = FrankWolfe.Secant()
    settings.frank_wolfe[:lazy] = lazy
    settings.frank_wolfe[:max_fw_iter] = fw_iter
    settings.frank_wolfe[:fw_verbose] = false

    if starting_point_dca || starting_point_fw
        settings.domain[:active_set] = active_set
    end

    if regularized
        # settings.tightening[:sharpness_constant] = sqrt(2 / μ)
        settings.tightening[:sharpness_constant] = 1
        settings.tightening[:sharpness_exponent] = 1 / 2
        # settings.tightening[:strong_convexity] = μ
    end

    x, _, result = Boscia.solve(f_run, grad_run!, blmo, settings = settings)

    X = reshape(x, n, n)

    total_time_in_sec = result[:total_time_in_sec]
    status = result[:status_string]
    if occursin("Optimal", status)
        status = "OPTIMAL"
        if !is_graph_matching
            @assert A ≈ X' * B * X
        end
    elseif occursin("Time", status)
        status = "TIME_LIMIT"
    elseif status == "User defined stop"
        if A ≈ X' * B * X
            if !is_graph_matching
                @info "Found Isomorphism"
            end
            status = "OPTIMAL"
        elseif !is_graph_matching
            @show result[:dual_bound]
            @assert result[:dual_bound] > 0.0
            staus = "OPTIMAL"
            @info ("Is not isomorphic...")
        end
    end

    return status, total_time_in_sec, result
end
