using Boscia
using SparseArrays
using LinearAlgebra
using Bonobo
using FrankWolfe
using Random
using CombinatorialLinearOracles
using Graphs
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

function OBBT_preprocess(A, B, n, blmo)
    int_vars = collect(1:(n^2))

    function build_function_gradient_fixing(A_mat, B_mat, i, j, fix_to_zero = true)
        B2 = Matrix(1.0B_mat^2)
        A2 = Matrix(1.0A_mat^2)

        linear_gradient_term = if fix_to_zero
            n^2
        else
            -n^2
        end

        function f(x)
            X = reshape(x, n, n)
            linear_expr = if fix_to_zero
                X[i, j]
            else
                (1 - X[i, j])
            end
            return norm(X * A_mat - B_mat * X)^2 + n^2 * linear_expr
        end
        function grad!(storage, x)
            X = reshape(x, n, n)
            S = 2 * X * A2 - 4 * B_mat * X * A_mat + 2 * B2 * X
            S[i, j] += linear_gradient_term
            storage .= vec(S)
            return nothing
        end
        return f, grad!
    end

    function build_FW_callback()
        return function callback(state, kwargs...)
            if state.primal - state.dual_gap > 0.0
                # println("Early termination: lower bound is already strictly positive!")
                return false
            end
        end
    end


    iters_OBBT = Int[]

    # number of variables we check
    num_checked = 0

    num_fixed_to_zero = 0
    num_fixed_to_one = 0
    for i = 1:n
        # Skip this row if it's already fixed to one
        if i in blmo.fixed_to_one_rows
            continue
        end
        for j = 1:n
            # Skip this column if it's already fixed to one
            if j in blmo.fixed_to_one_cols
                continue
            end

            num_checked += 1
            linear_idx = (j - 1) * n + i

            if blmo.upper_bounds[linear_idx] == 0.0
                # this entry is set to zero
                continue
            end

            f_ij_zero, grad_ij_zero! = build_function_gradient_fixing(A, B, i, j, true)
            x0_ = FrankWolfe.compute_extreme_point(blmo, ones(n^2))
            fw_res_zero = FrankWolfe.decomposition_invariant_conditional_gradient(
                f_ij_zero,
                grad_ij_zero!,
                blmo,
                x0_,
                verbose = false,
                max_iteration = 100,
                lazy = false,
                trajectory = true,
                callback = build_FW_callback(),
            )

            if fw_res_zero.primal - fw_res_zero.dual_gap > 0
                # Access last iteration and time from trajectory data
                # traj_data contains tuples: (t, primal, dual, dual_gap, time)
                if !isempty(fw_res_zero.traj_data)
                    last_entry = fw_res_zero.traj_data[end]
                    last_iteration = last_entry[1]  # t
                    push!(iters_OBBT, last_iteration)
                end

                blmo.lower_bounds[linear_idx] = 1.0
                push!(blmo.fixed_to_one_cols, j)
                push!(blmo.fixed_to_one_rows, i)
                # Update index_map_rows and index_map_cols
                Boscia.delete_bounds!(blmo, [])
                num_fixed_to_one += 1
                # Jump out of this row since it's now fixed to one
                break

            else
                f_ij_one, grad_ij_one! = build_function_gradient_fixing(A, B, i, j, false)
                x0 = FrankWolfe.compute_extreme_point(blmo, ones(n^2))
                fw_res_one = FrankWolfe.decomposition_invariant_conditional_gradient(
                    f_ij_one,
                    grad_ij_one!,
                    blmo,
                    x0,
                    verbose = false,
                    max_iteration = 100,
                    lazy = false,
                    trajectory = true,
                    callback = build_FW_callback(),
                )
                if fw_res_one.primal - fw_res_one.dual_gap > 0
                    if !isempty(fw_res_one.traj_data)
                        last_entry = fw_res_one.traj_data[end]
                        last_iteration = last_entry[1]  # iteration number
                        push!(iters_OBBT, last_iteration)
                    end
                    blmo.upper_bounds[linear_idx] = 0.0
                    num_fixed_to_zero += 1
                end
            end
        end
    end

    is_feasible = Boscia.check_feasibility(blmo) == "Optimal" ? true : false
    return (is_feasible, blmo, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)
end

function clique_preprocess(A, B, n, blmo)
    # Convert adjacency matrices to Graphs.jl SimpleGraph objects
    # Note: A and B are adjacency matrices, we need to convert them to graphs
    # SimpleGraph can take an adjacency matrix (symmetric, zero diagonal)
    gA = SimpleGraph(Matrix(A))
    gB = SimpleGraph(Matrix(B))

    # Compute triangle counts for each vertex
    # Count the number of triangles each vertex participates in
    function count_triangles(g, v)
        count = 0
        neighbors_v = neighbors(g, v)
        for u in neighbors_v
            for w in neighbors_v
                if u < w && has_edge(g, u, w)
                    count += 1
                end
            end
        end
        return count
    end

    tri_a = [count_triangles(gA, i) for i = 1:n]
    tri_b = [count_triangles(gB, i) for i = 1:n]

    fixed_zero = Set{Tuple{Int,Int}}()
    num_fixed_to_zero = 0

    # Fix variables based on triangle counts
    for i = 1:n
        for j = 1:n
            if tri_a[i] != tri_b[j]
                push!(fixed_zero, (i, j))
            end
        end
    end
    @info "$(length(fixed_zero)) fixed from triangles"

    # Compute maximal cliques
    mclique_a = maximal_cliques(gA)
    mclique_b = maximal_cliques(gB)

    # Check if maximum clique sizes match (quick non-isomorphism check)
    if sort(length.(mclique_a)) != sort(length.(mclique_b))
        @info "Non-isomorphic: maximum clique sizes don't match"
        iters_OBBT = Int[]
        num_checked = 0
        num_fixed_to_one = 0
        return (false, blmo, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)
    end

    # Count clique memberships by size for each vertex
    clique_size_count_a = [Dict{Int,Int}() for _ = 1:nv(gA)]
    clique_size_count_b = [Dict{Int,Int}() for _ = 1:nv(gB)]

    for clique in mclique_a
        clique_size = length(clique)
        for v in clique
            d = clique_size_count_a[v]
            d[clique_size] = get(d, clique_size, 0) + 1
        end
    end

    for clique in mclique_b
        clique_size = length(clique)
        for v in clique
            d = clique_size_count_b[v]
            d[clique_size] = get(d, clique_size, 0) + 1
        end
    end

    # Fix variables based on clique membership counts
    for i = 1:n
        for j = 1:n
            if clique_size_count_a[i] != clique_size_count_b[j]
                push!(fixed_zero, (i, j))
            end
        end
    end
    @info "$(length(fixed_zero)) fixed from maximal clique memberships"

    # Apply OBBT to blmo
    for (i, j) in fixed_zero
        linear_idx = (i - 1) * n + j
        blmo.upper_bounds[linear_idx] = 0.0
        num_fixed_to_zero += 1
    end


    is_feasible = Boscia.check_feasibility(blmo) == "Optimal" ? true : false

    # Return same structure as OBBT_preprocess
    iters_OBBT = Int[]  # No iterations for clique-based fixing
    num_checked = length(fixed_zero)  # Number of variables we checked/fixed
    num_fixed_to_one = 0  # Clique method only fixes to zero

    return (is_feasible, blmo, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)
end

function star_preprocess(A, B, n, blmo)

    # Convert adjacency matrices to Graphs.jl SimpleGraph objects
    gA = SimpleGraph(Matrix(A))
    gB = SimpleGraph(Matrix(B))

    star_size_count_a = [Dict{Int,Int}() for _ = 1:nv(gA)]
    star_size_count_b = [Dict{Int,Int}() for _ = 1:nv(gB)]

    for i = 1:n
        nbrs_a = neighbors(gA, i)
        if !isempty(nbrs_a)
            subg_a, _ = induced_subgraph(gA, nbrs_a)
            cgA = complement(subg_a)
            for clique in maximal_cliques(cgA)
                star_size = length(clique) - 1
                d = star_size_count_a[i]
                d[star_size] = get(d, star_size, 0) + 1
            end
        end

        nbrs_b = neighbors(gB, i)
        if !isempty(nbrs_b)
            subg_b, _ = induced_subgraph(gB, nbrs_b)
            cgB = complement(subg_b)
            for clique in maximal_cliques(cgB)
                star_size = length(clique) - 1
                d = star_size_count_b[i]
                d[star_size] = get(d, star_size, 0) + 1
            end
        end
    end

    # Check if star size counts match (quick non-isomorphism check)
    if sort(length.(star_size_count_a)) != sort(length.(star_size_count_b))
        @info "Non-isomorphic: star size counts don't match"
        iters_OBBT = Int[]
        num_checked = 0
        num_fixed_to_one = 0
        num_fixed_to_zero = 0
        return (false, blmo, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)
    end

    fixed_zero = Set{Tuple{Int,Int}}()
    num_fixed_to_zero = 0
    for i = 1:n
        for j = 1:n
            linear_idx = (i - 1) * n + j
            if blmo.upper_bounds[linear_idx] != 0.0
                if star_size_count_a[i] != star_size_count_b[j]
                    push!(fixed_zero, (i, j))
                    blmo.upper_bounds[linear_idx] = 0.0
                    num_fixed_to_zero += 1
                end
            end
        end
    end

    @info "$(length(fixed_zero)) fixed after star size count"

    is_feasible = Boscia.check_feasibility(blmo) == "Optimal" ? true : false

    # Return same structure as other warm-start functions
    iters_OBBT = Int[]  # No iterations for star2-based fixing
    num_checked = length(fixed_zero)  # Number of variables we checked/fixed
    num_fixed_to_one = 0  # Star2 method only fixes to zero

    return (is_feasible, blmo, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)
end

function walk_signature_preprocess(A, B, n, blmo; K = 10, use_bigint = true)
    # Convert to integer matrices for exact arithmetic
    # For standard unweighted GI, this is the safest route.
    if use_bigint
        Aint = Matrix{BigInt}(A)
        Bint = Matrix{BigInt}(B)
        onevecA = ones(BigInt, n)
        onevecB = ones(BigInt, n)
        sigA = Matrix{BigInt}(undef, n, K)
        sigB = Matrix{BigInt}(undef, n, K)
    else
        # Faster, but less overflow-safe for larger K / larger graphs
        Aint = Matrix{Int}(A)
        Bint = Matrix{Int}(B)
        onevecA = ones(Int, n)
        onevecB = ones(Int, n)
        sigA = Matrix{Int}(undef, n, K)
        sigB = Matrix{Int}(undef, n, K)
    end

    # Compute signatures:
    # column k stores A^k * 1  (and similarly for B), for k = 1,...,K
    vA = copy(onevecA)
    vB = copy(onevecB)

    for k in 1:K
        vA = Aint * vA
        vB = Bint * vB
        sigA[:, k] = vA
        sigB[:, k] = vB
    end

    # Turn each row into a tuple so we can compare signatures exactly
    signaturesA = [Tuple(sigA[i, :]) for i in 1:n]
    signaturesB = [Tuple(sigB[i, :]) for i in 1:n]

    # Quick safe non-isomorphism check:
    # under any isomorphism, the multiset of vertex signatures must match
    if sort(signaturesA) != sort(signaturesB)
        @info "Non-isomorphic: walk-signature multisets do not match"
        iters_OBBT = Int[]
        num_checked = 0
        num_fixed_to_one = 0
        num_fixed_to_zero = 0
        return (false, blmo, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)
    end

    fixed_zero = Set{Tuple{Int,Int}}()
    num_fixed_to_zero = 0
    num_checked = 0

    for i in 1:n
        sig_i = signaturesA[i]
        for j in 1:n
            linear_idx = (i - 1) * n + j
            if blmo.upper_bounds[linear_idx] != 0.0
                num_checked += 1
                if sig_i != signaturesB[j]
                    push!(fixed_zero, (i, j))
                    blmo.upper_bounds[linear_idx] = 0.0
                    num_fixed_to_zero += 1
                end
            end
        end
    end

    @info "$(length(fixed_zero)) variables fixed to zero after walk-signature filtering (K = $K)"

    is_feasible = Boscia.check_feasibility(blmo) == "Optimal" ? true : false

    iters_OBBT = Int[]
    num_fixed_to_one = 0

    return (is_feasible, blmo, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)
end

function preprocessing(
    A,
    B,
    n;
    use_clique = false,
    use_star = false,
    use_OBBT = false,
    use_walk_sig = false,
    iso_generate = true,
    is_graph_matching = false,
    time_limit = 3600,
)

    blmo = CLO.BirkhoffLMO(n, collect(1:(n^2)))
    time_left = time_limit
    t_OBBT = 0.0
    t_clique = 0.0
    t_star = 0.0
    t_walk_sig = 0.0
    num_checked = 0
    iters_OBBT = Int[]
    num_fixed_to_zero_OBBT = 0
    num_fixed_to_zero_clique = 0
    num_fixed_to_zero_star = 0
    num_fixed_to_zero_walk_sig = 0
    num_fixed_to_one = 0
    preprocessing_time() = t_OBBT + t_clique + t_star + t_walk_sig


    if use_clique
        @info "Activating clique-warm-start..."
        t_clique = @elapsed begin
            is_feasible, blmo, _, _, num_fixed_to_zero_clique, _ =
                clique_preprocess(A, B, n, blmo)
        end
        @info "Clique-warm-start took $(t_clique) seconds"

        if !iso_generate && !is_graph_matching && !is_feasible
            @info "Not isomorphic (clique warm-start)"
            return "Optimal",
            preprocessing_time(),
            (
                (t_OBBT, t_clique, t_star),
                iters_OBBT,
                num_checked,
                (num_fixed_to_zero_OBBT, num_fixed_to_zero_clique, num_fixed_to_zero_star),
                num_fixed_to_one,
            ),
            nothing
        end
    end

    if use_star
        @info "Activating star-warm-start..."
        t_star = @elapsed begin
            is_feasible, blmo, _, _, num_fixed_to_zero_star, _ =
                star_preprocess(A, B, n, blmo)
        end
        @info "Star-warm-start took $(t_star) seconds"

        @info "$(num_fixed_to_zero_star) are fixed to zero"


        if !iso_generate && !is_graph_matching && !is_feasible
            @info "Not isomorphic (star warm-start)"
            return "Optimal",
            preprocessing_time(),
            (
                (t_OBBT, t_clique, t_star),
                iters_OBBT,
                num_checked,
                (num_fixed_to_zero_OBBT, num_fixed_to_zero_clique, num_fixed_to_zero_star),
                num_fixed_to_one,
            ),
            nothing
        end
    end

    if use_OBBT
        @info "Activating OBBT-warm-start..."
        t_OBBT = @elapsed begin
            is_feasible,
            blmo,
            iters_OBBT,
            num_checked,
            num_fixed_to_zero_OBBT,
            num_fixed_to_one = OBBT_preprocess(A, B, n, blmo)
        end
        @info "OBBT-warm-start took $(t_OBBT) seconds;"
        @info " $(num_fixed_to_zero_OBBT) are fixed to zero;"
        @info " $(num_fixed_to_one) are fixed to one;"

        if !iso_generate && !is_graph_matching
            @info "Not isomorphic (OBBT-warm-start)"
            return "Optimal",
            preprocessing_time(),
            (
                (t_OBBT, t_clique, t_star),
                iters_OBBT,
                num_checked,
                (num_fixed_to_zero_OBBT, num_fixed_to_zero_clique, num_fixed_to_zero_star),
                num_fixed_to_one,
            ),
            nothing
        end
        time_left = max(1, Int(round(time_limit - t_OBBT)))
    end

    if use_walk_sig
        @info "Activating walk-signature preprocess..."
        t_walk_sig = @elapsed begin
            is_feasible,
            blmo,
            _,
            num_checked_walk,
            num_fixed_to_zero_walk_sig,
            _ = walk_signature_preprocess(A, B, n, blmo)
            num_checked += num_checked_walk
        end
        @info "Walk-signature preprocess took $(t_walk_sig) seconds"
        @info " $(num_fixed_to_zero_walk_sig) are fixed to zero"

        if !iso_generate && !is_graph_matching && !is_feasible
            @info "Not isomorphic (walk-signature preprocess)"
            return "Optimal",
            preprocessing_time(),
            (
                (t_OBBT, t_clique, t_star, t_walk_sig),
                iters_OBBT,
                num_checked,
                (
                    num_fixed_to_zero_OBBT,
                    num_fixed_to_zero_clique,
                    num_fixed_to_zero_star,
                    num_fixed_to_zero_walk_sig,
                ),
                num_fixed_to_one,
            ),
            nothing
        end
    end

    num_fixed_to_zero =
        (
            num_fixed_to_zero_clique,
            num_fixed_to_zero_star,
            num_fixed_to_zero_OBBT,
            num_fixed_to_zero_walk_sig,
        )
    t = (t_clique, t_star, t_OBBT, t_walk_sig)

    preprocessing_results =
        (t, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)

    return blmo, preprocessing_results
end

function build_function_gradient(A, B, n)
    R = zeros(n, n)

    B2 = Matrix(1.0B^2)
    A2 = Matrix(1.0A^2)
    BX = zeros(n, n)
    function f_acc2(x)
        X = reshape(x, n, n)
        # R = X * A - B * X
        mul!(R, X, A)
        mul!(R, B, X, -1, 1)
        return norm(R)^2
    end

    function f_acc2_check(x)
        X = reshape(x, n, n)
        # R = X * A - B * X
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
        # 2A^2 X
        mul!(S, X, A2, 2, 0)
        # -4AXB
        mul!(S, BX, A, -4, 1)
        # +2 X B^2
        mul!(S, B2, X, 2, 1)
        return nothing
    end
    return f_acc2, grad_acc2!, f_acc2_check
end

function build_exp_function_gradient(A, B, n, tau)
    # Precompute Matrix Exponentials
    EA = exp(tau * Matrix(1.0A))
    EB = exp(tau * Matrix(1.0B))

    # Precompute squares for the gradient: 2(X*EA^2 - 2*EB*X*EA + EB^2*X)
    EA2 = EA^2
    EB2 = EB^2

    # Pre-allocate caches
    R = zeros(n, n)
    EBX = zeros(n, n)

    # Objective function using Frobenius norm [cite: 45]
    function f_exp(x)
        X = reshape(x, n, n)
        # R = X * EA - EB * X
        mul!(R, X, EA)
        mul!(R, EB, X, -1, 1)
        return norm(R)^2
    end

    # Gradient computation
    function grad_exp!(storage, x)
        X = reshape(x, n, n)
        S = reshape(storage, n, n)

        # Cache EB * X for the middle term
        mul!(EBX, EB, X)

        # Term 1: 2 * X * EA^2
        mul!(S, X, EA2, 2, 0)

        # Term 2: -4 * (EB * X) * EA
        mul!(S, EBX, EA, -4, 1)

        # Term 3: 2 * EB^2 * X
        mul!(S, EB2, X, 2, 1)

        return nothing
    end

    # Optional check function
    function f_exp_check(x)
        X = reshape(x, n, n)
        res = norm(X * EA - EB * X)^2
        return res
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

        # R = X*EA - EB*X
        mul!(R, X, EA)
        mul!(R, EB, X, -1, 1)

        # T1 = R * EA'
        mul!(T1, R, EA')

        # T2 = EB' * R
        mul!(T2, EB', R)

        # grad = 2 * (R*EA' - EB'*R)
        @. S = 2.0 * (T1 - T2)

        return nothing
    end

    function f_exp_trunc_check(x)
        X = reshape(x, n, n)
        return norm(X * EA - EB * X)^2
    end

    return f_exp_trunc, grad_exp_trunc!, f_exp_trunc_check
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
