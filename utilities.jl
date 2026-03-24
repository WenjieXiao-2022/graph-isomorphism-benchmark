using LinearAlgebra
using SparseArrays
using Random
using MAT
using FrankWolfe

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
                push!(names, replace(f, r"\.mat$" => ""))
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

function randomPermutation(A::AbstractMatrix)
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
    base = "./test_instances"
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

function generate_wigner_goe(n::Int)
    M = zeros(Float64, n, n)
    for i = 1:n
        for j = i:n
            if i == j
                M[i, j] = randn() * sqrt(2.0 / n)
            else
                val = randn() * sqrt(1.0 / n)
                M[i, j] = val
                M[j, i] = val
            end
        end
    end
    return M
end

function generate_easy_boscia_instance(n::Int; noise_threshold::Float64 = -1.0, epsilon::Float64 = 0.1)
    sigma = n^(noise_threshold - epsilon)
    println("Graph size: $n x $n | Theoretical safe noise (sigma): $sigma")
    A = generate_wigner_goe(n)
    Z = generate_wigner_goe(n)
    p = randperm(n)
    Pi_star = zeros(Int, n, n)
    for i = 1:n
        Pi_star[i, p[i]] = 1
    end
    A_noisy = A + sigma * Z
    B = A_noisy[p, p]
    return A, B, Pi_star, p
end


function OBBT_preprocess(
    A,
    B,
    n,
    blmo;
    time_limit = 3600.0,
    boscia_node_limit = 3,
    boscia_fw_iter = 1000,
    obbt_fw_iter = 50,
    tol = 1e-8,
    verbose = false,
    try_fix_to_one = true,
)

    @info "Running OBBT preprocess for graph matching"

    t_start = time()

    f, grad! = build_function_gradient(A, B, n)

    # ==========================================================
    # 1. Get an incumbent UB from a short Boscia run
    # ==========================================================
    settings = Boscia.create_default_settings()
    settings.branch_and_bound[:node_limit] = boscia_node_limit
    if isfinite(time_limit)
        settings.branch_and_bound[:time_limit] = max(1, floor(Int, time_limit))
    end
    settings.branch_and_bound[:verbose] = verbose
    settings.frank_wolfe[:max_fw_iter] = boscia_fw_iter
    settings.frank_wolfe[:lazy] = false
    settings.frank_wolfe[:variant] = Boscia.DecompositionInvariantConditionalGradient()

    blmo_pre = CLO.BirkhoffLMO(n, collect(1:n^2))
    x_ub, _, result = Boscia.solve(f, grad!, blmo_pre; settings = settings)
    UB = result[:primal_objective]

    @info "OBBT incumbent UB = $UB"

    # ==========================================================
    # 2. Helpers
    # ==========================================================
    idx_from_ij(i, j, n) = (j - 1) * n + i

    function copy_blmo_with_bounds(blmo, n)
        new_blmo = CLO.BirkhoffLMO(n, collect(1:n^2))
        new_blmo.lower_bounds .= copy(blmo.lower_bounds)
        new_blmo.upper_bounds .= copy(blmo.upper_bounds)
        return new_blmo
    end

    # ----------------------------------------------------------
    # Recompute which rows/cols are already closed:
    # row i is closed if some x[i,j] is fixed to 1
    # col j is closed if some x[i,j] is fixed to 1
    # ----------------------------------------------------------
    function recompute_closed_rows_cols(blmo, n)
        row_closed = falses(n)
        col_closed = falses(n)
        for i in 1:n, j in 1:n
            idx = idx_from_ij(i, j, n)
            if blmo.lower_bounds[idx] == 1.0
                row_closed[i] = true
                col_closed[j] = true
            end
        end
        return row_closed, col_closed
    end

    # ----------------------------------------------------------
    # Propagate x[i,j] = 1 in the Birkhoff polytope:
    # - fix x[i,j] = 1
    # - all other vars in row i => 0
    # - all other vars in col j => 0
    # - detect direct contradiction
    # returns true if propagation stays feasible
    # ----------------------------------------------------------
    function fix_to_one_and_propagate!(blmo, n, i, j, row_closed, col_closed)
        idx = idx_from_ij(i, j, n)

        # Contradiction: variable already forced to 0
        if blmo.upper_bounds[idx] == 0.0
            return false
        end

        # Contradiction: another 1 already exists in row i
        for jj in 1:n
            if jj != j
                idx2 = idx_from_ij(i, jj, n)
                if blmo.lower_bounds[idx2] == 1.0
                    return false
                end
            end
        end

        # Contradiction: another 1 already exists in column j
        for ii in 1:n
            if ii != i
                idx2 = idx_from_ij(ii, j, n)
                if blmo.lower_bounds[idx2] == 1.0
                    return false
                end
            end
        end

        # Commit x[i,j] = 1
        blmo.lower_bounds[idx] = 1.0
        blmo.upper_bounds[idx] = 1.0

        # Propagate row i => all other entries 0
        for jj in 1:n
            if jj != j
                idx2 = idx_from_ij(i, jj, n)
                if blmo.lower_bounds[idx2] == 1.0
                    return false
                end
                blmo.upper_bounds[idx2] = 0.0
            end
        end

        # Propagate column j => all other entries 0
        for ii in 1:n
            if ii != i
                idx2 = idx_from_ij(ii, j, n)
                if blmo.lower_bounds[idx2] == 1.0
                    return false
                end
                blmo.upper_bounds[idx2] = 0.0
            end
        end

        row_closed[i] = true
        col_closed[j] = true

        # Quick necessary feasibility checks:
        # every row must still have at least one variable not fixed to 0
        for ii in 1:n
            has_candidate = false
            for jj in 1:n
                idx2 = idx_from_ij(ii, jj, n)
                if blmo.upper_bounds[idx2] > 0.0
                    has_candidate = true
                    break
                end
            end
            if !has_candidate
                return false
            end
        end

        # every column must still have at least one variable not fixed to 0
        for jj in 1:n
            has_candidate = false
            for ii in 1:n
                idx2 = idx_from_ij(ii, jj, n)
                if blmo.upper_bounds[idx2] > 0.0
                    has_candidate = true
                    break
                end
            end
            if !has_candidate
                return false
            end
        end

        return true
    end

    # ----------------------------------------------------------
    # Callback for restricted FW solves:
    # stop once LB >= UB - tol
    # ----------------------------------------------------------
    function build_FW_callback()
        return function callback(state, kwargs...)
            LB = state.primal - state.dual_gap
            return LB < UB - tol
        end
    end

    # ----------------------------------------------------------
    # Solve restricted relaxation and return LB
    # ----------------------------------------------------------
    function restricted_lower_bound(test_blmo)
        x0 = FrankWolfe.compute_extreme_point(test_blmo, ones(n^2))
        fw_res = FrankWolfe.decomposition_invariant_conditional_gradient(
            f,
            grad!,
            test_blmo,
            x0,
            verbose = false,
            max_iteration = obbt_fw_iter,
            lazy = false,
            trajectory = true,
            callback = build_FW_callback(),
        )
        LB = fw_res.primal - fw_res.dual_gap
        return LB, fw_res
    end

    # ==========================================================
    # 3. Initialize row/column closed flags from current blmo
    # ==========================================================
    row_closed, col_closed = recompute_closed_rows_cols(blmo, n)

    # ==========================================================
    # 4. Main OBBT loop
    # ==========================================================
    iters_OBBT = Int[]
    num_checked = 0
    num_fixed_to_zero = 0
    num_fixed_to_one = 0
    optimality_certified = false

    for i in 1:n
        if (time() - t_start) >= time_limit
            @info "OBBT preprocess hit time limit ($(time_limit)s). Stopping early."
            break
        end

        # If row already assigned, skip entire row
        if row_closed[i]
            continue
        end

        for j in 1:n
            if (time() - t_start) >= time_limit
                @info "OBBT preprocess hit time limit ($(time_limit)s). Stopping early."
                break
            end

            # If row or column already assigned, skip
            if row_closed[i] || col_closed[j]
                continue
            end

            idx = idx_from_ij(i, j, n)

            # Skip already fixed vars
            if blmo.upper_bounds[idx] == 0.0 || blmo.lower_bounds[idx] == 1.0
                continue
            end

            num_checked += 1

            # --------------------------------------------------
            # Branch x[i,j] = 1
            # If infeasible or LB >= UB - tol, then branch is closed
            # and x[i,j] = 1 cannot improve incumbent
            # --------------------------------------------------
            branch_one_closed = false
            test_blmo_one = copy_blmo_with_bounds(blmo, n)
            test_row_closed = copy(row_closed)
            test_col_closed = copy(col_closed)

            feasible_one = fix_to_one_and_propagate!(test_blmo_one, n, i, j, test_row_closed, test_col_closed)

            if !feasible_one || Boscia.check_feasibility(test_blmo_one) != Boscia.OPTIMAL
                branch_one_closed = true
            else
                LB_one, fw_res_one = restricted_lower_bound(test_blmo_one)

                if !isempty(fw_res_one.traj_data)
                    push!(iters_OBBT, fw_res_one.traj_data[end][1])
                end

                if LB_one >= UB - tol
                    branch_one_closed = true
                end
            end

            # --------------------------------------------------
            # Branch x[i,j] = 0
            # If infeasible or LB >= UB - tol, then branch is closed
            # and x[i,j] = 0 cannot improve incumbent
            # --------------------------------------------------
            branch_zero_closed = false
            if try_fix_to_one
                test_blmo_zero = copy_blmo_with_bounds(blmo, n)
                test_blmo_zero.upper_bounds[idx] = 0.0

                if Boscia.check_feasibility(test_blmo_zero) != Boscia.OPTIMAL
                    branch_zero_closed = true
                else
                    LB_zero, fw_res_zero = restricted_lower_bound(test_blmo_zero)

                    if !isempty(fw_res_zero.traj_data)
                        push!(iters_OBBT, fw_res_zero.traj_data[end][1])
                    end

                    if LB_zero >= UB - tol
                        branch_zero_closed = true
                    end
                end
            end

            # --------------------------------------------------
            # Early optimality stop:
            # both branches are closed => incumbent is optimal
            # --------------------------------------------------
            if branch_one_closed && branch_zero_closed
                @info "Both branches at (i=$i, j=$j) are closed. Incumbent is optimal within tolerance."
                optimality_certified = true
                break
            end

            # --------------------------------------------------
            # Safe fixing decisions
            # --------------------------------------------------
            # If branch x[i,j]=1 is closed, then x[i,j]=0 is valid
            if branch_one_closed
                blmo.upper_bounds[idx] = 0.0
                num_fixed_to_zero += 1
            end

            # If branch x[i,j]=0 is closed, then x[i,j]=1 is valid
            if branch_zero_closed
                ok = fix_to_one_and_propagate!(blmo, n, i, j, row_closed, col_closed)
                if !ok
                    error("Unexpected contradiction while committing x[$i,$j] = 1 to blmo.")
                end
                num_fixed_to_one += 1
            end
        end

        if optimality_certified
            break
        end
    end

    # ==========================================================
    # 5. Final feasibility check
    # ==========================================================
    is_feasible = Boscia.check_feasibility(blmo) == Boscia.OPTIMAL

    return (
        x_ub,
        UB,
        is_feasible,
        blmo,
        iters_OBBT,
        num_checked,
        num_fixed_to_zero,
        num_fixed_to_one,
        optimality_certified,
        row_closed,
        col_closed,
    )
end