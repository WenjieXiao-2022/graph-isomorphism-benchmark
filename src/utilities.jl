using LinearAlgebra
using SparseArrays
using Random
using MAT
using FrankWolfe
using Bonobo
using Boscia

# Make simple, undirected, unweighted adjacency (sparse Bool)
function _to_simple_undirected(A::SparseMatrixCSC)
    A = sparse(A)
    A = max.(A, A')                       # symmetrize
    A = A - spdiagm(0 => diag(A))         # remove self-loops
    A = map(!iszero, A)                   # binarize
    return SparseMatrixCSC{Bool,Int}(A)
end

function get_graph_names_paths(root::String)
    names = String[]
    paths = String[]
    for (dir, dirs, files) in walkdir(root)
        for f in files
            if endswith(f, ".mat") || endswith(f, ".dimacs")
                path = joinpath(root, f)
                @assert isfile(path)
                if endswith(f, ".mat")
                    push!(names, replace(f, r"\.mat$" => ""))
                else
                    push!(names, replace(f, r"\.dimacs$" => ""))
                end
                push!(paths, path)
            end
        end

        for subdir in dirs
            path = joinpath(dir, subdir)

            for (_, _, files) in walkdir(path)
                for f in files
                    startswith(f, ".") && continue          # skip hidden files (.directory, etc.)
                    (endswith(f, ".mat") || endswith(f, ".dimacs")) || continue
                    path_tmp = joinpath(path, f)
                    @assert isfile(path_tmp)
                    push!(paths, path_tmp)
                    if endswith(f, ".mat")
                        name = replace(f, r"\.mat$" => "")
                    elseif endswith(f, ".dimacs")
                        name = replace(f, r"\.dimacs$" => "")
                    else
                        name = f
                    end
                    push!(names, name)
                end
            end
        end
    end
    return names, paths
end

function randomPermutation(A::AbstractMatrix{})
    n = size(A, 1)
    p = randperm(n)
    P = sparse(1:n, p, ones(Int, n), n, n)  # sparse permutation matrix
    B = P * A * P'                          # permuted matrix
    return B, P
end

function non_iso_graph(A::AbstractMatrix; edges_flipped::Int = 1)
    B = copy(A)
    n = size(B, 1)

    # All possible undirected edges (i < j)
    all_edges = [(i, j) for i = 1:n-1 for j = i+1:n]
    max_edges = length(all_edges)

    # Clamp requested flips to [0, max_edges]
    k = clamp(edges_flipped, 0, max_edges)

    @info "Requested to flip $edges_flipped edges, actually flipping $k unique edges."

    shuffle!(all_edges)

    for (i, j) in Iterators.take(all_edges, k)
        new_val = 1 - B[i, j]
        B[i, j] = new_val
        B[j, i] = new_val
    end

    return B
end

"""
    load_graph(name::String) -> A, n

Load a benchmark graph adjacency matrix by name from `./test_instances`.

The function searches recursively for either `<name>.mat` or `<name>.dimacs`
and parses based on the discovered extension.
"""
function load_graph(name::String)
    base = joinpath(@__DIR__, "..", "test_instances")
    target_mat = name * ".mat"
    target_dimacs = name * ".dimacs"
    filepath = nothing
    ext = nothing

    for (root, _, files) in walkdir(base)
        if target_mat in files
            filepath = joinpath(root, target_mat)
            ext = ".mat"
            break
        elseif target_dimacs in files
            filepath = joinpath(root, target_dimacs)
            ext = ".dimacs"
            break
        end
    end

    isnothing(filepath) &&
        error("No graph named $name found under $base (expected .mat or .dimacs)")

    if ext == ".mat"
        data = matread(filepath)
        A = sparse(data["A"])
        n = Int(data["n"])
        return SparseMatrixCSC{Int,Int}(A), n
    elseif ext == ".dimacs"
        n = 0
        I = Int[]
        J = Int[]
        open(filepath, "r") do io
            for line in eachline(io)
                s = strip(line)
                isempty(s) && continue
                if startswith(s, "p")
                    n = parse(Int, split(s)[3])
                elseif startswith(s, "e")
                    parts = split(s)
                    length(parts) >= 3 || continue
                    u = parse(Int, parts[2])
                    v = parse(Int, parts[3])
                    u == v && continue
                    # symmetrize
                    push!(I, u)
                    push!(J, v)
                    push!(I, v)
                    push!(J, u)
                end
            end
        end

        if n == 0
            error("Please check again the graph data...")
        end

        n > 0 || error("Could not determine number of nodes in $filepath")

        return sparse(I, J, ones(Int, length(I)), n, n), n
    else
        error("Unsupported graph extension for file: $filepath")
    end
end

"""
Solve quadratic programming problem: min 0.5*x'*H*x subject to C*x = d, 0 <= x <= 1
"""
function solve_quadprog_fw_package(n, A, B; time_limit = 300)
    function f(x)
        X = reshape(x, n, n)
        R = X * A - B * X
        return sum(abs2, R)
    end

    function grad!(storage, x)
        X = reshape(x, n, n)
        grad_matrix = 2 * (X * A - B * X) * A' - 2 * B' * (X * A - B * X)
        storage .= vec(grad_matrix)
    end

    lmo = FrankWolfe.BirkhoffPolytopeLMO()
    x0 = FrankWolfe.compute_extreme_point(lmo, zeros(n^2))


    active_set = FrankWolfe.ActiveSetQuadraticProductCaching([(1.0, x0)], grad!)

    x, _, _, _, _, _, active_set = FrankWolfe.blended_pairwise_conditional_gradient(
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
    return x, active_set
end

function choose_tau(A, B; c = 1.0)
    degA = maximum(vec(sum(A, dims = 2)))
    degB = maximum(vec(sum(B, dims = 2)))
    return c / max(Float64(degA), Float64(degB), 1.0)
end

function build_post_propagate_bounds(
    grad!;
    tol = 1e-8,
    atol = 1e-9,
    propagate_singletons = true,
    early_skip_no_fixing = true,
    verbose = false,
)
    warned_no_incumbent = Ref(false)
    warned_no_feasible_red = Ref(false)

    function idx_from_ij(i, j, n, append_by_column)
        return append_by_column ? ((j - 1) * n + i) : ((i - 1) * n + j)
    end

    function ij_from_idx(idx, n, append_by_column)
        if append_by_column
            j = ceil(Int, idx / n)
            i = idx - (j - 1) * n
        else
            i = ceil(Int, idx / n)
            j = idx - (i - 1) * n
        end
        return i, j
    end

    function current_bound_arrays(node, lmo)
        n = lmo.dim
        N = n^2

        lb = zeros(Float64, N)
        ub = ones(Float64, N)

        for (cidx, var_idx) in enumerate(lmo.int_vars)
            lb[var_idx] = lmo.lower_bounds[cidx]
            ub[var_idx] = lmo.upper_bounds[cidx]
        end

        for (var_idx, val) in node.local_bounds.lower_bounds
            lb[var_idx] = val
        end
        for (var_idx, val) in node.local_bounds.upper_bounds
            ub[var_idx] = val
        end

        return lb, ub
    end

    function recompute_closed_rows_cols(lb, ub, n, append_by_column)
        row_closed = falses(n)
        col_closed = falses(n)

        for idx in 1:(n^2)
            if lb[idx] >= 1.0 - atol && ub[idx] >= 1.0 - atol
                i, j = ij_from_idx(idx, n, append_by_column)
                row_closed[i] = true
                col_closed[j] = true
            end
        end
        return row_closed, col_closed
    end

    function full_bounds_feasible(lb, ub, n, append_by_column)
        for idx in 1:(n^2)
            if ub[idx] < lb[idx] - atol
                return false
            end
        end

        for i in 1:n
            row_lb = 0.0
            row_ub = 0.0
            for j in 1:n
                idx = idx_from_ij(i, j, n, append_by_column)
                row_lb += lb[idx]
                row_ub += ub[idx]
            end
            if row_lb > 1.0 + atol || row_ub < 1.0 - atol
                return false
            end
        end

        for j in 1:n
            col_lb = 0.0
            col_ub = 0.0
            for i in 1:n
                idx = idx_from_ij(i, j, n, append_by_column)
                col_lb += lb[idx]
                col_ub += ub[idx]
            end
            if col_lb > 1.0 + atol || col_ub < 1.0 - atol
                return false
            end
        end

        return true
    end

    function fix_to_one_and_propagate!(
        node,
        lb,
        ub,
        n,
        i,
        j,
        append_by_column,
    )
        idx = idx_from_ij(i, j, n, append_by_column)

        if ub[idx] <= atol
            return false
        end

        for jj in 1:n
            if jj == j
                continue
            end
            idx2 = idx_from_ij(i, jj, n, append_by_column)
            if lb[idx2] >= 1.0 - atol
                return false
            end
        end

        for ii in 1:n
            if ii == i
                continue
            end
            idx2 = idx_from_ij(ii, j, n, append_by_column)
            if lb[idx2] >= 1.0 - atol
                return false
            end
        end

        lb[idx] = 1.0
        ub[idx] = 1.0
        node.local_bounds.lower_bounds[idx] = 1.0
        node.local_bounds.upper_bounds[idx] = 1.0

        for jj in 1:n
            if jj == j
                continue
            end
            idx2 = idx_from_ij(i, jj, n, append_by_column)
            if lb[idx2] >= 1.0 - atol
                return false
            end
            ub[idx2] = 0.0
            node.local_bounds.upper_bounds[idx2] = 0.0
        end

        for ii in 1:n
            if ii == i
                continue
            end
            idx2 = idx_from_ij(ii, j, n, append_by_column)
            if lb[idx2] >= 1.0 - atol
                return false
            end
            ub[idx2] = 0.0
            node.local_bounds.upper_bounds[idx2] = 0.0
        end

        return full_bounds_feasible(lb, ub, n, append_by_column)
    end

    function propagate_singletons!(node, lb, ub, n, append_by_column)
        num_fixed_to_one = 0
        changed = true

        while changed
            changed = false
            row_closed, col_closed =
                recompute_closed_rows_cols(lb, ub, n, append_by_column)

            for i in 1:n
                if row_closed[i]
                    continue
                end

                candidates = Int[]
                for j in 1:n
                    if col_closed[j]
                        continue
                    end
                    idx = idx_from_ij(i, j, n, append_by_column)
                    if ub[idx] > atol
                        push!(candidates, j)
                    end
                end

                if isempty(candidates)
                    return false, num_fixed_to_one
                elseif length(candidates) == 1
                    j = candidates[1]
                    idx = idx_from_ij(i, j, n, append_by_column)
                    if lb[idx] < 1.0 - atol
                        ok = fix_to_one_and_propagate!(
                            node, lb, ub, n, i, j, append_by_column
                        )
                        if !ok
                            return false, num_fixed_to_one
                        end
                        num_fixed_to_one += 1
                        changed = true
                    end
                end
            end

            row_closed, col_closed =
                recompute_closed_rows_cols(lb, ub, n, append_by_column)

            for j in 1:n
                if col_closed[j]
                    continue
                end

                candidates = Int[]
                for i in 1:n
                    if row_closed[i]
                        continue
                    end
                    idx = idx_from_ij(i, j, n, append_by_column)
                    if ub[idx] > atol
                        push!(candidates, i)
                    end
                end

                if isempty(candidates)
                    return false, num_fixed_to_one
                elseif length(candidates) == 1
                    i = candidates[1]
                    idx = idx_from_ij(i, j, n, append_by_column)
                    if lb[idx] < 1.0 - atol
                        ok = fix_to_one_and_propagate!(
                            node, lb, ub, n, i, j, append_by_column
                        )
                        if !ok
                            return false, num_fixed_to_one
                        end
                        num_fixed_to_one += 1
                        changed = true
                    end
                end
            end
        end

        return true, num_fixed_to_one
    end

    function build_reduced_gradient(g, lb, ub, n, append_by_column)
        G = append_by_column ? reshape(g, n, n) : transpose(reshape(g, n, n))

        row_closed, col_closed = recompute_closed_rows_cols(lb, ub, n, append_by_column)
        open_rows = [i for i in 1:n if !row_closed[i]]
        open_cols = [j for j in 1:n if !col_closed[j]]

        nr = length(open_rows)
        nc = length(open_cols)

        if nr == 0 || nc == 0
            return false, nothing, nothing, nothing
        end

        d2 = Matrix{Union{Float64,Missing}}(undef, nr, nc)

        for (ii, i) in enumerate(open_rows)
            for (jj, j) in enumerate(open_cols)
                idx = idx_from_ij(i, j, n, append_by_column)
                if ub[idx] <= atol
                    d2[ii, jj] = missing
                else
                    d2[ii, jj] = G[i, j]
                end
            end
        end

        for ii in 1:nr
            if all(ismissing(d2[ii, jj]) for jj in 1:nc)
                return false, nothing, nothing, nothing
            end
        end
        for jj in 1:nc
            if all(ismissing(d2[ii, jj]) for ii in 1:nr)
                return false, nothing, nothing, nothing
            end
        end

        return true, d2, open_rows, open_cols
    end

    function open_entry_stats(g, x, lb, ub, n, append_by_column)
        gmax = -Inf
        gmin = Inf
        xmax = 0.0
        count_open = 0

        for idx in 1:(n^2)
            if ub[idx] > atol && lb[idx] < 1.0 - atol
                count_open += 1
                gval = g[idx]
                if gval > gmax
                    gmax = gval
                end
                if gval < gmin
                    gmin = gval
                end
                xval = x[idx]
                if xval > xmax
                    xmax = xval
                end
            end
        end

        if count_open == 0
            return false, 0.0, 0.0, 0.0
        end
        return true, gmax, gmin, xmax
    end

    return function post_propagate_bounds(tree, node, x, lower_bound)
        UB = tree.incumbent
        if !isfinite(UB)
            if verbose && !warned_no_incumbent[]
                warned_no_incumbent[] = true
                @info "post_propagate_bounds: skipping while `tree.incumbent` is infinite (no integer UB yet). After heuristics find one, fixings can apply on later nodes."
            end
            return nothing
        end

        blmo = tree.root.problem.tlmo.lmo
        n = blmo.dim
        append_by_column = blmo.append_by_column

        lb, ub = current_bound_arrays(node, blmo)

        g = zeros(Float64, length(x))
        grad!(g, x)

        # Cheap screening before running Hungarian:
        # if the required reduced-cost threshold is larger than a coarse upper
        # bound from gradient spread on open entries, no post-RDC fixing can be
        # triggered at this node.
        if early_skip_no_fixing
            required_rc = UB - tol - lower_bound
            if required_rc > 0
                has_open, gmax, gmin, xmax =
                    open_entry_stats(g, x, lb, ub, n, append_by_column)
                if !has_open
                    return nothing
                end
                rc_upper_bound = max(0.0, gmax - gmin)
                if required_rc > rc_upper_bound + atol
                    if verbose
                        @info "post_propagate_bounds: early skip (required rc=$(required_rc) > coarse upper bound=$(rc_upper_bound); max open x=$(xmax))."
                    end
                    return nothing
                end
            end
        end

        feasible_red, d2, open_rows, open_cols =
            build_reduced_gradient(g, lb, ub, n, append_by_column)

        if !feasible_red
            if verbose && !warned_no_feasible_red[]
                warned_no_feasible_red[] = true
                @info "post_propagate_bounds: reduced submatrix not usable (e.g. all rows/cols closed or zero gradient on open entries); no RDC fixings until the open pattern changes."
            end
            return nothing
        end

        assignment, lap_cost, alpha, beta = Hungarian.hungarian_with_duals(d2)

        nr = length(open_rows)
        nc = length(open_cols)
        rc = fill(Inf, nr, nc)

        for ii in 1:nr
            for jj in 1:nc
                if !ismissing(d2[ii, jj])
                    rc[ii, jj] = d2[ii, jj] - alpha[ii] - beta[jj]
                    if rc[ii, jj] < 0 && abs(rc[ii, jj]) <= 1e-8
                        rc[ii, jj] = 0.0
                    end
                end
            end
        end

        num_fixed_to_zero = 0
        num_fixed_to_one = 0

        for ii in 1:nr
            i = open_rows[ii]
            for jj in 1:nc
                j = open_cols[jj]

                if !isfinite(rc[ii, jj])
                    continue
                end

                idx = idx_from_ij(i, j, n, append_by_column)

                if ub[idx] <= atol || lb[idx] >= 1.0 - atol
                    continue
                end

                if lower_bound + rc[ii, jj] >= UB - tol
                    ub[idx] = 0.0
                    node.local_bounds.upper_bounds[idx] = 0.0
                    num_fixed_to_zero += 1
                end
            end
        end

        if propagate_singletons
            ok, added_fix1 = propagate_singletons!(node, lb, ub, n, append_by_column)
            if ok
                num_fixed_to_one += added_fix1
            end
        end

        node.local_tightenings += num_fixed_to_zero + num_fixed_to_one

        if verbose && (num_fixed_to_zero > 0 || num_fixed_to_one > 0)
            @info "Node $(node.std.id): RDC fixed $(num_fixed_to_zero) to zero, propagated $(num_fixed_to_one) to one."
        end

        return nothing
    end
end

function Bonobo.get_branching_variable(
    tree::Bonobo.BnBTree,
    ::Bonobo.MOST_INFEASIBLE,
    node::Boscia.FrankWolfeNode,
)
    values = Bonobo.get_relaxed_values(tree, node)

    best_idx = -1
    max_distance_to_feasible = 0.0
    atol = tree.options.atol

    local_lbs = node.local_bounds.lower_bounds
    local_ubs = node.local_bounds.upper_bounds

    for i in tree.branching_indices
        lb = get(local_lbs, i, 0.0)
        ub = get(local_ubs, i, 1.0)

        # skip variables already fixed by local bounds
        if ub <= lb + atol
            continue
        end

        value = values[i]

        if !Bonobo.is_approx_feasible(tree, value)
            distance_to_feasible = Bonobo.get_distance_to_feasible(tree, value)
            if distance_to_feasible > max_distance_to_feasible
                best_idx = i
                max_distance_to_feasible = distance_to_feasible
            end
        end
    end

    return best_idx
end