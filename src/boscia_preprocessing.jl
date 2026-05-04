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

    is_feasible = Boscia.check_feasibility(blmo) == Boscia.OPTIMAL ? true : false
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
        return (false, blmo, num_fixed_to_zero)
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

    is_feasible = Boscia.check_feasibility(blmo) == Boscia.OPTIMAL ? true : false
    return (is_feasible, blmo, num_fixed_to_zero)
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
        num_fixed_to_zero = 0
        return (false, blmo, num_fixed_to_zero)
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
    is_feasible = Boscia.check_feasibility(blmo) == Boscia.OPTIMAL ? true : false
    return (is_feasible, blmo, num_fixed_to_zero)
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
        num_checked = 0
        num_fixed_to_zero = 0
        return (false, blmo, num_checked, num_fixed_to_zero)
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
    is_feasible = Boscia.check_feasibility(blmo) == Boscia.OPTIMAL ? true : false
    return (is_feasible, blmo, num_checked, num_fixed_to_zero)
end

function classical_exp_walk_preprocess(A, B, n, blmo; K = 6)
    Ai = BigInt.(Matrix{Int}(A))
    Bi = BigInt.(Matrix{Int}(B))

    ones_vec = ones(BigInt, n)

    function exact_walk_signatures(M, n, K)
        sigs = [BigInt[] for _ = 1:n]
        v = ones_vec

        for r = 1:K
            v = M * v
            for i = 1:n
                push!(sigs[i], v[i])
            end
        end

        return sigs
    end

    sig_a = exact_walk_signatures(Ai, n, K)
    sig_b = exact_walk_signatures(Bi, n, K)

    if sort(sig_a) != sort(sig_b)
        @info "Non-isomorphic: exact BigInt walk signatures don't match"
        num_fixed_to_zero = 0
        return (false, blmo, num_fixed_to_zero)
    end

    fixed_zero = Set{Tuple{Int,Int}}()
    num_fixed_to_zero = 0

    for i = 1:n
        for j = 1:n
            linear_idx = (i - 1) * n + j

            if blmo.upper_bounds[linear_idx] != 0.0
                if sig_a[i] != sig_b[j]
                    push!(fixed_zero, (i, j))
                    blmo.upper_bounds[linear_idx] = 0.0
                    num_fixed_to_zero += 1
                end
            end
        end
    end

    @info "$(length(fixed_zero)) fixed after exact BigInt exp-walk / walk-count preprocessing"

    is_feasible = Boscia.check_feasibility(blmo) == Boscia.OPTIMAL ? true : false
    return (is_feasible, blmo, num_fixed_to_zero)
end

function quantum_walk_preprocess(A, B, n, blmo;
    times = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0],
    digits = 14,
    match_tol = nothing,
)
    Ad = Matrix{Float64}(A)
    Bd = Matrix{Float64}(B)

    function vertex_quantum_signatures(M, n)
        sigs = [Float64[] for _ = 1:n]

        for t in times
            U = exp(im * t * M)

            for i = 1:n
                # Return amplitude
                push!(sigs[i], round(real(U[i, i]); digits = digits))
                push!(sigs[i], round(imag(U[i, i]); digits = digits))

                # Transition probabilities from i to all vertices.
                # Sorting makes this invariant under relabeling.
                probs = sort(round.(abs2.(U[i, :]); digits = digits))
                append!(sigs[i], probs)
            end
        end

        return sigs
    end

    quantum_sig_a = vertex_quantum_signatures(Ad, n)
    quantum_sig_b = vertex_quantum_signatures(Bd, n)

    # Exact vector equality on rounded Float64 signatures is brittle for matrix
    # exponentials. Use a tolerance-aware compatibility test instead.
    tol = isnothing(match_tol) ? max(1e-12, 10.0^(2 - digits)) : match_tol

    function signatures_close(sig1, sig2)
        length(sig1) == length(sig2) || return false
        return all(isapprox(sig1[k], sig2[k]; atol = tol, rtol = 0.0) for k in eachindex(sig1))
    end

    function has_perfect_signature_matching(sigA, sigB)
        nloc = length(sigA)
        adj = [Int[] for _ = 1:nloc]
        for i in 1:nloc
            for j in 1:nloc
                if signatures_close(sigA[i], sigB[j])
                    push!(adj[i], j)
                end
            end
            if isempty(adj[i])
                return false
            end
        end

        match_to_i = fill(0, nloc)
        seen = fill(false, nloc)

        function augment(i)
            for j in adj[i]
                if seen[j]
                    continue
                end
                seen[j] = true
                if match_to_i[j] == 0 || augment(match_to_i[j])
                    match_to_i[j] = i
                    return true
                end
            end
            return false
        end

        for i in 1:nloc
            fill!(seen, false)
            if !augment(i)
                return false
            end
        end
        return true
    end

    # If the multisets of vertex signatures differ, the graphs are non-isomorphic.
    if !has_perfect_signature_matching(quantum_sig_a, quantum_sig_b)
        @info "Non-isomorphic: quantum-walk signatures don't match"
        num_fixed_to_zero = 0
        return (false, blmo, num_fixed_to_zero)
    end

    fixed_zero = Set{Tuple{Int,Int}}()
    num_fixed_to_zero = 0

    for i = 1:n
        for j = 1:n
            linear_idx = (i - 1) * n + j

            if blmo.upper_bounds[linear_idx] != 0.0
                if !signatures_close(quantum_sig_a[i], quantum_sig_b[j])
                    push!(fixed_zero, (i, j))
                    blmo.upper_bounds[linear_idx] = 0.0
                    num_fixed_to_zero += 1
                end
            end
        end
    end

    @info "$(length(fixed_zero)) fixed after quantum-walk preprocessing"

    is_feasible = Boscia.check_feasibility(blmo) == Boscia.OPTIMAL ? true : false
    return (is_feasible, blmo, num_fixed_to_zero)
end

function k_particle_quantum_walk_preprocess(A, B, n, blmo;
    k = 2,
    times = [0.25, 0.5, 1.0, 2.0],
    digits = 10,
)
    Ad = Matrix{Float64}(A)
    Bd = Matrix{Float64}(B)

    if k < 1
        error("k must be at least 1")
    end

    if k > n
        error("k cannot be larger than n")
    end

    function combinations_k(n, k)
        result = Vector{Vector{Int}}()
        current = Int[]

        function backtrack(start, remaining)
            if remaining == 0
                push!(result, copy(current))
                return
            end

            for v in start:(n - remaining + 1)
                push!(current, v)
                backtrack(v + 1, remaining - 1)
                pop!(current)
            end
        end

        backtrack(1, k)
        return result
    end

    states = combinations_k(n, k)
    num_states = length(states)

    state_index = Dict{Tuple{Vararg{Int}}, Int}()

    for (idx, S) in enumerate(states)
        state_index[Tuple(S)] = idx
    end

    states_containing_vertex = [Int[] for _ = 1:n]

    for (idx, S) in enumerate(states)
        for v in S
            push!(states_containing_vertex[v], idx)
        end
    end

    function build_k_particle_hamiltonian(M)
        H = zeros(Float64, num_states, num_states)

        for (idx, S) in enumerate(states)
            Sset = Set(S)

            for pos in 1:k
                old_v = S[pos]

                for new_v in 1:n
                    if M[old_v, new_v] != 0.0 && !(new_v in Sset)
                        T = copy(S)
                        T[pos] = new_v
                        sort!(T)

                        jdx = state_index[Tuple(T)]
                        H[idx, jdx] = M[old_v, new_v]
                    end
                end
            end
        end

        # Symmetrize to avoid tiny asymmetries from construction/order.
        return 0.5 .* (H .+ H')
    end

    function vertex_k_particle_signatures(M)
        H = build_k_particle_hamiltonian(M)
        sigs = [Float64[] for _ = 1:n]

        for t in times
            U = exp(im * t * H)
            P = abs2.(U)

            for v in 1:n
                start_states = states_containing_vertex[v]

                # Return amplitude aggregated over all k-particle states containing v.
                ret = zero(ComplexF64)
                for s in start_states
                    ret += U[s, s]
                end

                push!(sigs[v], round(real(ret); digits = digits))
                push!(sigs[v], round(imag(ret); digits = digits))

                # Transition probability mass from states containing v
                # to states containing each vertex w.
                vertex_masses = Float64[]

                for w in 1:n
                    target_states = states_containing_vertex[w]

                    mass = 0.0
                    for s in start_states
                        for r in target_states
                            mass += P[s, r]
                        end
                    end

                    push!(vertex_masses, round(mass; digits = digits))
                end

                # Sorting makes the signature independent of vertex labels.
                sort!(vertex_masses)
                append!(sigs[v], vertex_masses)
            end
        end

        return sigs
    end

    quantum_sig_a = vertex_k_particle_signatures(Ad)
    quantum_sig_b = vertex_k_particle_signatures(Bd)

    if sort(quantum_sig_a) != sort(quantum_sig_b)
        @info "Non-isomorphic: k-particle quantum-walk signatures don't match"
        num_fixed_to_zero = 0
        return (false, blmo, num_fixed_to_zero)
    end

    fixed_zero = Set{Tuple{Int,Int}}()
    num_fixed_to_zero = 0

    for i in 1:n
        for j in 1:n
            linear_idx = (i - 1) * n + j

            if blmo.upper_bounds[linear_idx] != 0.0
                if quantum_sig_a[i] != quantum_sig_b[j]
                    push!(fixed_zero, (i, j))
                    blmo.upper_bounds[linear_idx] = 0.0
                    num_fixed_to_zero += 1
                end
            end
        end
    end

    @info "$(length(fixed_zero)) fixed after $k-particle quantum-walk preprocessing"

    is_feasible = Boscia.check_feasibility(blmo) == Boscia.OPTIMAL ? true : false
    return (is_feasible, blmo, num_fixed_to_zero)
end

"""
If the benchmark assumes isomorphic graphs (`iso_generate`) but preprocessing shows
non-isomorphism or an infeasible assignment polytope, throw. Otherwise, for non-iso
benchmarks, return `true` when the caller should set `early_stop`.
"""
function _iso_benchmark_conflict_or_early_stop(
    iso_generate::Bool,
    is_graph_matching::Bool,
    is_feasible::Bool,
    reason::Symbol,
)::Bool
    if is_graph_matching || is_feasible
        return false
    end
    if iso_generate
        error(
            "Preprocessing conflict: iso_generate=true (isomorphic instance) but preprocessing " *
            "indicates non-isomorphism or an infeasible assignment set (stage=$(repr(reason))). " *
            "This should not happen for a correct isomorphic pair; check instance generation and invariants.",
        )
    end
    return true
end

function preprocessing(
    A,
    B,
    n;
    use_clique = false,
    use_star = false,
    use_OBBT = false,
    use_walk_sig = false,
    use_classical_exp_walk = false,
    use_quantum = false,
    use_k_particle_quantum = false,
    k_particle_k = 2,
    iso_generate = true,
    is_graph_matching = false,
    time_limit = 3600,
)
    blmo = CLO.BirkhoffLMO(n, collect(1:(n^2)))

    times = (
        clique = 0.0,
        star = 0.0,
        obbt = 0.0,
        walk_sig = 0.0,
        classical_exp = 0.0,
        quantum = 0.0,
        k_particle = 0.0,
    )
    iters_obbt = Int[]
    checked_total = 0
    fixed_to_zero = (
        clique = 0,
        star = 0,
        obbt = 0,
        walk_sig = 0,
        classical_exp = 0,
        quantum = 0,
        k_particle = 0,
    )
    fixed_to_one = 0
    early_stop = false
    early_reason = nothing

    if use_clique
        @info "Activating clique-warm-start..."
        t = @elapsed begin
            is_feasible, blmo, nfix0 = clique_preprocess(A, B, n, blmo)
            fixed_to_zero = (;
                clique = nfix0,
                star = fixed_to_zero.star,
                obbt = fixed_to_zero.obbt,
                walk_sig = fixed_to_zero.walk_sig,
                classical_exp = fixed_to_zero.classical_exp,
                quantum = fixed_to_zero.quantum,
                k_particle = fixed_to_zero.k_particle,
            )
            if _iso_benchmark_conflict_or_early_stop(
                iso_generate,
                is_graph_matching,
                is_feasible,
                :clique,
            )
                early_stop = true
                early_reason = :clique
            end
        end
        times = (;
            clique = t,
            star = times.star,
            obbt = times.obbt,
            walk_sig = times.walk_sig,
            classical_exp = times.classical_exp,
            quantum = times.quantum,
            k_particle = times.k_particle,
        )
        @info "Clique-warm-start took $(t) seconds"
    end

    if use_star && !early_stop
        @info "Activating star-warm-start..."
        t = @elapsed begin
            is_feasible, blmo, nfix0 = star_preprocess(A, B, n, blmo)
            fixed_to_zero = (;
                clique = fixed_to_zero.clique,
                star = nfix0,
                obbt = fixed_to_zero.obbt,
                walk_sig = fixed_to_zero.walk_sig,
                classical_exp = fixed_to_zero.classical_exp,
                quantum = fixed_to_zero.quantum,
                k_particle = fixed_to_zero.k_particle,
            )
            if _iso_benchmark_conflict_or_early_stop(
                iso_generate,
                is_graph_matching,
                is_feasible,
                :star,
            )
                early_stop = true
                early_reason = :star
            end
        end
        times = (;
            clique = times.clique,
            star = t,
            obbt = times.obbt,
            walk_sig = times.walk_sig,
            classical_exp = times.classical_exp,
            quantum = times.quantum,
            k_particle = times.k_particle,
        )
        @info "Star-warm-start took $(t) seconds"
        @info "$(fixed_to_zero.star) are fixed to zero"
    end

    if use_OBBT && !early_stop
        @info "Activating OBBT-warm-start..."
        t = @elapsed begin
            is_feasible, blmo, iters_obbt, nchecked, nfix0, fixed_to_one =
                OBBT_preprocess(A, B, n, blmo)
            checked_total = nchecked
            fixed_to_zero = (;
                clique = fixed_to_zero.clique,
                star = fixed_to_zero.star,
                obbt = nfix0,
                walk_sig = fixed_to_zero.walk_sig,
                classical_exp = fixed_to_zero.classical_exp,
                quantum = fixed_to_zero.quantum,
                k_particle = fixed_to_zero.k_particle,
            )
            if _iso_benchmark_conflict_or_early_stop(
                iso_generate,
                is_graph_matching,
                is_feasible,
                :obbt,
            )
                early_stop = true
                early_reason = :obbt
            end
        end
        times = (;
            clique = times.clique,
            star = times.star,
            obbt = t,
            walk_sig = times.walk_sig,
            classical_exp = times.classical_exp,
            quantum = times.quantum,
            k_particle = times.k_particle,
        )
        @info "OBBT-warm-start took $(t) seconds;"
        @info " $(fixed_to_zero.obbt) are fixed to zero;"
        @info " $(fixed_to_one) are fixed to one;"
    end

    if use_walk_sig && !early_stop
        @info "Activating walk-signature preprocess..."
        t = @elapsed begin
            is_feasible, blmo, nchecked, nfix0 = walk_signature_preprocess(A, B, n, blmo)
            checked_total += nchecked
            fixed_to_zero = (;
                clique = fixed_to_zero.clique,
                star = fixed_to_zero.star,
                obbt = fixed_to_zero.obbt,
                walk_sig = nfix0,
                classical_exp = fixed_to_zero.classical_exp,
                quantum = fixed_to_zero.quantum,
                k_particle = fixed_to_zero.k_particle,
            )
            if _iso_benchmark_conflict_or_early_stop(
                iso_generate,
                is_graph_matching,
                is_feasible,
                :walk_sig,
            )
                early_stop = true
                early_reason = :walk_sig
            end
        end
        times = (;
            clique = times.clique,
            star = times.star,
            obbt = times.obbt,
            walk_sig = t,
            classical_exp = times.classical_exp,
            quantum = times.quantum,
            k_particle = times.k_particle,
        )
        @info "Walk-signature preprocess took $(t) seconds"
        @info " $(fixed_to_zero.walk_sig) are fixed to zero"
    end

    if use_classical_exp_walk && !early_stop
        @info "Activating classical exp-walk preprocess..."
        t = @elapsed begin
            is_feasible, blmo, nfix0 = classical_exp_walk_preprocess(A, B, n, blmo)
            fixed_to_zero = (;
                clique = fixed_to_zero.clique,
                star = fixed_to_zero.star,
                obbt = fixed_to_zero.obbt,
                walk_sig = fixed_to_zero.walk_sig,
                classical_exp = nfix0,
                quantum = fixed_to_zero.quantum,
                k_particle = fixed_to_zero.k_particle,
            )
            if !iso_generate && !is_graph_matching && !is_feasible
                early_stop = true
                early_reason = :classical_exp
            end
        end
        times = (;
            clique = times.clique,
            star = times.star,
            obbt = times.obbt,
            walk_sig = times.walk_sig,
            classical_exp = t,
            quantum = times.quantum,
            k_particle = times.k_particle,
        )
        @info "Classical exp-walk preprocess took $(t) seconds"
        @info " $(fixed_to_zero.classical_exp) are fixed to zero"
    end

    if use_quantum && !early_stop
        @info "Activating quantum-walk preprocess..."
        t = @elapsed begin
            is_feasible, blmo, nfix0 = quantum_walk_preprocess(A, B, n, blmo)
            fixed_to_zero = (;
                clique = fixed_to_zero.clique,
                star = fixed_to_zero.star,
                obbt = fixed_to_zero.obbt,
                walk_sig = fixed_to_zero.walk_sig,
                classical_exp = fixed_to_zero.classical_exp,
                quantum = nfix0,
                k_particle = fixed_to_zero.k_particle,
            )
            if _iso_benchmark_conflict_or_early_stop(
                iso_generate,
                is_graph_matching,
                is_feasible,
                :quantum,
            )
                early_stop = true
                early_reason = :quantum
            end
        end
        times = (;
            clique = times.clique,
            star = times.star,
            obbt = times.obbt,
            walk_sig = times.walk_sig,
            classical_exp = times.classical_exp,
            quantum = t,
            k_particle = times.k_particle,
        )
        @info "Quantum-walk preprocess took $(t) seconds"
        @info " $(fixed_to_zero.quantum) are fixed to zero"
    end

    if use_k_particle_quantum && !early_stop
        @info "Activating k-particle quantum-walk preprocess (k = $(k_particle_k))..."
        t = @elapsed begin
            is_feasible, blmo, nfix0 = k_particle_quantum_walk_preprocess(
                A,
                B,
                n,
                blmo;
                k = k_particle_k,
            )
            fixed_to_zero = (;
                clique = fixed_to_zero.clique,
                star = fixed_to_zero.star,
                obbt = fixed_to_zero.obbt,
                walk_sig = fixed_to_zero.walk_sig,
                classical_exp = fixed_to_zero.classical_exp,
                quantum = fixed_to_zero.quantum,
                k_particle = nfix0,
            )
            if _iso_benchmark_conflict_or_early_stop(
                iso_generate,
                is_graph_matching,
                is_feasible,
                :k_particle,
            )
                early_stop = true
                early_reason = :k_particle
            end
        end
        times = (;
            clique = times.clique,
            star = times.star,
            obbt = times.obbt,
            walk_sig = times.walk_sig,
            classical_exp = times.classical_exp,
            quantum = times.quantum,
            k_particle = t,
        )
        @info "k-particle quantum-walk preprocess took $(t) seconds"
        @info " $(fixed_to_zero.k_particle) are fixed to zero"
    end

    preprocessing_results = (
        times = times,
        iters_obbt = iters_obbt,
        checked_total = checked_total,
        fixed_to_zero = fixed_to_zero,
        fixed_to_one = fixed_to_one,
        early_stop = early_stop,
        early_reason = early_reason,
    )

    return blmo, preprocessing_results
end
