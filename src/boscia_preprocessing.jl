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
                return false
            end
        end
    end

    iters_OBBT = Int[]
    num_checked = 0
    num_fixed_to_zero = 0
    num_fixed_to_one = 0
    for i = 1:n
        if i in blmo.fixed_to_one_rows
            continue
        end
        for j = 1:n
            if j in blmo.fixed_to_one_cols
                continue
            end

            num_checked += 1
            linear_idx = (j - 1) * n + i

            if blmo.upper_bounds[linear_idx] == 0.0
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
                if !isempty(fw_res_zero.traj_data)
                    last_entry = fw_res_zero.traj_data[end]
                    last_iteration = last_entry[1]
                    push!(iters_OBBT, last_iteration)
                end

                blmo.lower_bounds[linear_idx] = 1.0
                push!(blmo.fixed_to_one_cols, j)
                push!(blmo.fixed_to_one_rows, i)
                Boscia.delete_bounds!(blmo, [])
                num_fixed_to_one += 1
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
                        last_iteration = last_entry[1]
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
    gA = SimpleGraph(Matrix(A))
    gB = SimpleGraph(Matrix(B))

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

    for i = 1:n
        for j = 1:n
            if tri_a[i] != tri_b[j]
                push!(fixed_zero, (i, j))
            end
        end
    end
    @info "$(length(fixed_zero)) fixed from triangles"

    mclique_a = maximal_cliques(gA)
    mclique_b = maximal_cliques(gB)

    if sort(length.(mclique_a)) != sort(length.(mclique_b))
        @info "Non-isomorphic: maximum clique sizes don't match"
        iters_OBBT = Int[]
        num_checked = 0
        num_fixed_to_one = 0
        return (false, blmo, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)
    end

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

    for i = 1:n
        for j = 1:n
            if clique_size_count_a[i] != clique_size_count_b[j]
                push!(fixed_zero, (i, j))
            end
        end
    end
    @info "$(length(fixed_zero)) fixed from maximal clique memberships"

    for (i, j) in fixed_zero
        linear_idx = (i - 1) * n + j
        blmo.upper_bounds[linear_idx] = 0.0
        num_fixed_to_zero += 1
    end

    is_feasible = Boscia.check_feasibility(blmo) == "Optimal" ? true : false
    iters_OBBT = Int[]
    num_checked = length(fixed_zero)
    num_fixed_to_one = 0

    return (is_feasible, blmo, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)
end

function star_preprocess(A, B, n, blmo)
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
    iters_OBBT = Int[]
    num_checked = length(fixed_zero)
    num_fixed_to_one = 0

    return (is_feasible, blmo, iters_OBBT, num_checked, num_fixed_to_zero, num_fixed_to_one)
end

function walk_signature_preprocess(A, B, n, blmo; K = 10, use_bigint = true)
    if use_bigint
        Aint = Matrix{BigInt}(A)
        Bint = Matrix{BigInt}(B)
        onevecA = ones(BigInt, n)
        onevecB = ones(BigInt, n)
        sigA = Matrix{BigInt}(undef, n, K)
        sigB = Matrix{BigInt}(undef, n, K)
    else
        Aint = Matrix{Int}(A)
        Bint = Matrix{Int}(B)
        onevecA = ones(Int, n)
        onevecB = ones(Int, n)
        sigA = Matrix{Int}(undef, n, K)
        sigB = Matrix{Int}(undef, n, K)
    end

    vA = copy(onevecA)
    vB = copy(onevecB)
    for k in 1:K
        vA = Aint * vA
        vB = Bint * vB
        sigA[:, k] = vA
        sigB[:, k] = vB
    end

    signaturesA = [Tuple(sigA[i, :]) for i in 1:n]
    signaturesB = [Tuple(sigB[i, :]) for i in 1:n]

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
