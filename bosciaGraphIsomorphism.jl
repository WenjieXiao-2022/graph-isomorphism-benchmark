using Boscia
using SparseArrays
using LinearAlgebra
using Bonobo
using FrankWolfe
using Random
using CombinatorialLinearOracles
const CLO = CombinatorialLinearOracles

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
        i, j = rand(1:n, 2)
        while i == j
            j = rand(1:n)
        end
        col_i = findfirst(x -> x == 1, new_P[i, :])
        col_j = findfirst(x -> x == 1, new_P[j, :])
        new_P[i, col_i] = 0
        new_P[i, col_j] = 1
        new_P[j, col_j] = 0
        new_P[j, col_i] = 1
        new_p = use_mip ? vec(new_P) : sparsevec(vec(new_P))
        push!(Ps, new_p)
    end

    return Ps, false
end

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
    function grad_acc2!(storage, x)
        X = reshape(x, n, n)
        mul!(BX, B, X)
        S = reshape(storage, n, n)
        mul!(S, X, A2, 2, 0)
        mul!(S, BX, A, -4, 1)
        mul!(S, B2, X, 2, 1)
        return nothing
    end
    return f_acc2, grad_acc2!
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
    return f_exp, grad_exp!
end

function FW_reduced_cost_preprocess(A, B, n, blmo)
    @info "Running FW reduced-cost preprocess with Feasibility Tracking"
    f, grad! = build_function_gradient(A, B, n)

    # 1. Get a strong Upper Bound (Primal)
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:time_limit] = 50
    blmo_pre_run = CLO.BirkhoffLMO(n, collect(1:n^2))
    _, _, result = Boscia.solve(f, grad!, blmo_pre_run, settings = settings)
    UB = result[:primal_objective]
    
    # 2. Get a strong Lower Bound (Dual)
    blmo_fw_run = CLO.BirkhoffLMO(n)
    x_start = FrankWolfe.compute_extreme_point(blmo_fw_run, ones(n^2))
    fw_res = FrankWolfe.decomposition_invariant_conditional_gradient(
        f, grad!, blmo_fw_run, x_start,
        verbose = false, max_iteration = 2000
    )
    LB = fw_res.primal - fw_res.dual_gap
    @info "Gap: $(UB - LB)"

    g = similar(fw_res.x)
    grad!(g, fw_res.x)

    # 3. Track row/column assignments
    row_fixed = fill(false, n)
    col_fixed = fill(false, n)
    num_fixed_to_zero = 0
    num_fixed_to_one = 0

    # It is best to iterate indices and map them to (row i, col j)
    for k in eachindex(fw_res.x)
        # Convert flat index k to (i, j) - assuming column-major (Julia default)
        i = (k - 1) % n + 1
        j = div(k - 1, n) + 1

        # Skip if already restricted by Boscia
        blmo.upper_bounds[k] == 0.0 && continue

        # Logic for Fixing to Zero (Safe)
        if LB + g[k] >= UB - 1e-8
            blmo.upper_bounds[k] = 0.0
            num_fixed_to_zero += 1
            
        # Logic for Fixing to One (Must check Row/Col conflicts)
        elseif LB - g[k] >= UB - 1e-8
            if !row_fixed[i] && !col_fixed[j]
                blmo.lower_bounds[k] = 1.0
                row_fixed[i] = true
                col_fixed[j] = true
                num_fixed_to_one += 1
                
                # OPTIONAL: If we fix X[i,j] = 1, we can immediately 
                # fix all other entries in row i and col j to 0.
                for row_idx in 1:n
                    idx = (j - 1) * n + row_idx
                    if row_idx != i && blmo.upper_bounds[idx] != 0.0
                        blmo.upper_bounds[idx] = 0.0
                        num_fixed_to_zero += 1
                    end
                end
                for col_idx in 1:n
                    idx = (col_idx - 1) * n + i
                    if col_idx != j && blmo.upper_bounds[idx] != 0.0
                        blmo.upper_bounds[idx] = 0.0
                        num_fixed_to_zero += 1
                    end
                end
            else
                @warn "Conflict detected: logic suggested fixing index $k (row $i, col $j) to 1, but row/col is already occupied."
                @warn "Solution is found."
            end
        end
    end

    is_feasible = Boscia.check_feasibility(blmo) == "Optimal"
    return (is_feasible, blmo, num_fixed_to_zero, num_fixed_to_one)
end

"""
    preprocessing(A, B, n; use_reduced_cost=false, time_limit=3600)

Returns `(blmo, preprocessing_results)` where `preprocessing_results` is
`(times, iters, num_checked, num_fixed_to_zero_tuple, num_fixed_to_one)` for
compatibility with `bench` CSV export (`times` is a 1-tuple of seconds).
"""
function preprocessing(
    A,
    B,
    n;
    use_reduced_cost = false,
    time_limit = 3600,
)
    blmo = CLO.BirkhoffLMO(n, collect(1:(n^2)))
    t_rc = 0.0
    iters_rc = Int[]
    num_checked = 0
    nz_rc = 0
    no_rc = 0

    if use_reduced_cost
        @info "Activating FW-reduced-cost preprocess..."
        t_rc = @elapsed begin
            _, blmo, iters_rc, num_checked, nz_rc, no_rc =
                FW_reduced_cost_preprocess(A, B, n, blmo)
        end
        @info "Reduced-cost preprocess took $(t_rc) s ($(nz_rc) to zero, $(no_rc) to one)"
    end

    num_fixed_to_zero = (nz_rc,)
    times = (t_rc,)
    preprocessing_results = (times, iters_rc, num_checked, num_fixed_to_zero, no_rc)
    return blmo, preprocessing_results
end

function boscia_run(
    A,
    B;
    solver = "boscia_dicg",
    time_limit = 3600,
    verbose = true,
    print_iter = 1,
    fw_verbose = false,
    fw_epsilon = 1e-2,
    use_depth = false,
    favor_right = nothing,
    use_reduced_cost = false,
)
    n = size(A, 1)

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

    settings_pre = Boscia.create_default_settings()
    settings_pre.branch_and_bound[:verbose] = true
    settings_pre.branch_and_bound[:print_iter] = print_iter
    settings_pre.branch_and_bound[:time_limit] = 10
    if use_depth
        settings_pre.branch_and_bound[:traverse_strategy] =
            Boscia.DepthFirstSearch(favor_right)
    end
    settings_pre.heuristic[:custom_heuristics] = [swap_heu]
    settings_pre.frank_wolfe[:variant] = variant
    settings_pre.frank_wolfe[:line_search] = FrankWolfe.Secant()
    settings_pre.frank_wolfe[:lazy] = lazy
    settings_pre.frank_wolfe[:max_fw_iter] = fw_iter
    settings_pre.frank_wolfe[:fw_verbose] = false
    settings_pre.frank_wolfe[:fw_epsilon] = fw_epsilon

    f, grad! = build_function_gradient(A, B, n)
    _, _, _ = Boscia.solve(f, grad!, blmo_precompile, settings = settings_pre)

    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:verbose] = verbose
    settings.branch_and_bound[:print_iter] = print_iter
    if use_depth
        settings.branch_and_bound[:traverse_strategy] =
            Boscia.DepthFirstSearch(favor_right)
    end

    blmo, preprocessing_results = preprocessing(
        A,
        B,
        n;
        use_reduced_cost = use_reduced_cost,
        time_limit = time_limit,
    )

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
        status = "OPTIMAL"
    elseif occursin("Time", status)
        status = "TIME_LIMIT"
    elseif status == "User defined stop"
        if A ≈ X' * B * X
            status = "OPTIMAL"
        end
    end

    return status, total_time_in_sec, preprocessing_results, result
end
