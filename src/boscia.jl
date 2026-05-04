include("boscia_preprocessing.jl")
include("boscia_func_grad.jl")

using Boscia
using SparseArrays
using LinearAlgebra
using Bonobo
using FrankWolfe
using Random
using CombinatorialLinearOracles
using Graphs
const CLO = CombinatorialLinearOracles

"""
    random_k_neighbor_matrix(tree, blmo, x, k; use_mip=false)

Generate `k` neighboring permutation candidates from the current incumbent by
swapping the assigned columns of two randomly chosen rows each step.
Returns `(neighbors, false)`, where `neighbors` is a vector of candidate points
in dense (`Vector`) or sparse (`SparseVector`) form depending on `use_mip`.
"""
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

        new_p = use_mip ? vec(new_P) : sparsevec(vec(new_P))  # Convert to proper SparseVector
        push!(Ps, new_p)
    end

    return Ps, false
end

function boscia_run(
    A,
    B;
    solver = "boscia_dicg",
    time_limit = 3600,
    verbose = true,
    print_iter = 100,
    fw_verbose = false,
    fw_epsilon = 1e-2,
    use_depth = false,
    is_graph_matching = false,
    favor_right = nothing,
    iso_generate = true,
    use_OBBT = false,
    use_clique = false,
    use_star = false,
    use_walk_sig = false,
    use_exp_formulation = false,
)
    n = size(A, 1)

    function build_branch_callback()
        return function (tree, node, vidx::Int)
            x = Bonobo.get_relaxed_values(tree, node)
            primal = tree.root.problem.f(x)
            lower_bound = primal - node.dual_gap
            optimal_val = 0.0
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
            optimal_val = 0.0
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

    # default is set to BPCG with lazification
    if contains(solver, "bpcg")
        variant = Boscia.BlendedPairwiseConditionalGradient()
        lazy = true
        fw_iter = 1000
    elseif contains(solver, "fw")
        variant = Boscia.StandardFrankWolfe()
        lazy = false
        fw_iter = 1000
    else
        variant = Boscia.DecompositionInvariantConditionalGradient()
        lazy = false
        fw_iter = 500
    end

    # Precompile
    settings_pre = Boscia.create_default_settings()
    settings_pre.branch_and_bound[:verbose] = true
    settings_pre.branch_and_bound[:print_iter] = print_iter

    if !is_graph_matching
        @info "Activating callback..."
        settings_pre.branch_and_bound[:bnb_callback] = build_tree_callback()
        settings_pre.branch_and_bound[:branch_callback] = build_branch_callback()
    end

    settings_pre.branch_and_bound[:time_limit] = 10

    use_depth ?
    settings_pre.branch_and_bound[:traverse_strategy] =
        Boscia.BiasedDepthFirstSearch(favor_right) : nothing

    settings_pre.heuristic[:custom_heuristics] = [swap_heu]
    settings_pre.frank_wolfe[:variant] = variant
    settings_pre.frank_wolfe[:line_search] = FrankWolfe.Secant()
    settings_pre.frank_wolfe[:lazy] = lazy
    settings_pre.frank_wolfe[:max_fw_iter] = fw_iter
    settings_pre.frank_wolfe[:fw_verbose] = false
    settings_pre.frank_wolfe[:fw_epsilon] = fw_epsilon

    if use_exp_formulation
        tau = choose_tau(A, B)
        tau = 0.1
        f, grad! = build_truncated_exp_function_gradient(A, B, n, tau, 4)
    else
        f, grad! = build_function_gradient(A, B, n)
    end

    _, _, _ = Boscia.solve(f, grad!, blmo_precompile, settings = settings_pre)

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = verbose
    settings.branch_and_bound[:print_iter] = print_iter

    if !is_graph_matching
        @info "Activating iso callback..."
        settings.branch_and_bound[:bnb_callback] = build_tree_callback()
        settings.branch_and_bound[:branch_callback] = build_branch_callback()
    end

    use_depth ?
    settings.branch_and_bound[:traverse_strategy] = Boscia.BiasedDepthFirstSearch(favor_right) :
    nothing

    blmo, preprocessing_results = preprocessing(
        A,
        B,
        n;
        use_clique = use_clique,
        use_star = use_star,
        use_OBBT = use_OBBT,
        use_walk_sig = use_walk_sig,
        iso_generate = iso_generate,
        is_graph_matching = is_graph_matching,
        time_limit = time_limit,
    )

    # Calculate remaining time after preprocessing
    preprocessing_time_elapsed = sum(preprocessing_results[1])
    time_left = max(1, Int(round(time_limit - preprocessing_time_elapsed)))
    settings.branch_and_bound[:time_limit] = time_left
    settings.heuristic[:custom_heuristics] = [swap_heu]
    settings.frank_wolfe[:variant] = variant
    settings.frank_wolfe[:line_search] = FrankWolfe.Secant()
    settings.frank_wolfe[:lazy] = lazy
    settings.frank_wolfe[:max_fw_iter] = fw_iter
    settings.frank_wolfe[:fw_verbose] = fw_verbose
    settings.frank_wolfe[:fw_epsilon] = fw_epsilon

    x, _, result = Boscia.solve(f, grad!, blmo, settings = settings)

    X = reshape(x, n, n)

    total_time_in_sec = result[:total_time_in_sec] + sum(preprocessing_results[1])

    status = result[:status_string]

    if occursin("Optimal", status)
        # Boscia found an optimal solution (isomorphism found)
        status = "OPTIMAL"
        if !is_graph_matching
            @assert A ≈ X' * B * X
        end
    elseif occursin("Time", status)
        # Time limit reached
        status = "TIME_LIMIT"
    elseif status == "User defined stop"
        # Solver stopped early (via callback)
        if A ≈ X' * B * X
            # Found a valid isomorphism (stopped early because solution found)
            if !is_graph_matching
                @info "Found Isomorphism"
            end
            status = "OPTIMAL"
        elseif !is_graph_matching && !iso_generate
            # Proved non-isomorphism: dual_bound > 0 means no solution exists
            # Status is "OPTIMAL" because we optimally determined the answer (no isomorphism exists)
            @show result[:dual_bound]
            @assert result[:dual_bound] > 0.0
            status = "OPTIMAL"
            @info "Is not isomorphic (certified via dual bound)"
        elseif iso_generate && !is_graph_matching
            # iso_generate=true means graphs are isomorphic, so we must find an isomorphism
            @error "iso_generate=true but A ≈ X' * B * X failed. Status: $status, X: $X"
        end
        # Note: If none of the conditions match, status remains "User defined stop"
        # Downstream code will handle this appropriately
    end

    return status, total_time_in_sec, preprocessing_results, result
end
