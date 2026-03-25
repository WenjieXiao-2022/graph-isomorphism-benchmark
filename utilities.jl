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
    base = "/home/htc/wexiao/project/graph_isomorphism/graph-isomorphism-benchmark/test_instances"
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