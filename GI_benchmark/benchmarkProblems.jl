using LinearAlgebra
using Statistics
using Random
using Printf
using MAT
using Hungarian
using FrankWolfe
using Boscia
using NautyGraphs
using Graphs

include("costMatrix.jl")
include("costMatrixRepeated.jl")
include("isIsomorphic.jl")
include("isIsomorphicRepeated.jl")
include("makeSignsConsistent_reorderVec.jl")
include("bosciaGraphIsomorphism.jl")
include("./frank_wolf_package_graph_isomorphism.jl")
include(
    "./frank_wolf_package_graph_isomorphism.jl",
)

include(
    "./dca_solver.jl",
)
include(
    "./feasibility_solver.jl",
)

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
            if endswith(f, ".mat")
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

Load a benchmark graph adjacency matrix by name.

Expects `.mat` files created from the exported `.m` graph functions,
each containing variables `A` (adjacency matrix) and `n` (number of nodes).
"""
function load_graph(name::String; format = "mat")
    if format == "mat"
        base = "./export_mat"
        filepath = joinpath(base, name * ".mat")
        isnothing(filepath) && error("No file named $target found under $base")
        # read the data
        data = matread(filepath)
        A = sparse(data["A"])
        n = Int(data["n"])
        return SparseMatrixCSC{Int,Int}(A), n
    elseif format == "dimacs"
        base = "./more_benchmark"
        target = name * ".dimacs"
        filepath = nothing
        for (root, _, files) in walkdir(base)
            if target in files
                filepath = joinpath(root, target)
                break  # stop immediately once found
            end
        end
        isnothing(filepath) && error("No file named $target found under $base")

        #read the data
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

        A = sparse(I, J, ones(Int, length(I)), n, n)
        return A, n
    else
        error("Unsupported format: $format")
    end

    if !isfile(filepath)
        error("Graph file not found: $filepath. Did you run the conversion step?")
    end

    return A, n
end

function bench(
    graph,
    seed,
    path;
    verbose = false,
    solver = "spectral",
    time_limit = Inf,
    write = false,
    fw_iter = 1000,
    format = "dimacs",
    iso_generate = true,
    print_iter = 100,
    starting_point_dca = false,
    starting_point_fw = false,
    use_qua_as = false,
    use_package = false,
)
    Random.seed!(seed)
    println("========================================================================")
    # load graph
    A, n = load_graph(graph; format = format)
    issolved = false
    @assert size(A, 1) == size(A, 2) "Graph $name not square"
    if !issymmetric(A)
        error("Graph $name not undirected (A != A').")
    end
    @printf "\n%s (n = %d): \n" graph n

    if iso_generate
        A1, P1 = randomPermutation(A)
        A2, P2 = randomPermutation(A)
    else
        edges_flipped =
            all(isdigit, split(solver, "_")[end]) ? parse(Int, split(solver, "_")[end]) : 1
        @show edges_flipped

        A1, P1 = randomPermutation(A)
        A2 = non_iso_graph(A; edges_flipped = edges_flipped)
    end

    if solver == "nauty"
        A_nauty = NautyGraph(A1)
        A1_nauty = NautyGraph(A2)
        solving_time = @elapsed begin
            issolved = A_nauty ≃ A1_nauty  # true if isomorphic 
        end
        @printf "\n Nauty result: %s\n Time: %.6f\n" string(issolved) solving_time

    elseif solver == "spectral"
        s1 = Vector{Bool}(undef, 1)   # isIso
        s2 = Vector{Int}(undef, 1)    # nBacktracking
        t = @elapsed begin
            time_ref = time()
            # Your Julia port of isIsomorphic must exist:
            # isIsomorphic(A, B; eps=1e-6, verbose=false) -> (b, P, nBack)
            isIso, _, nBack = isIsomorphic(
                A1,
                A2;
                eps = 1e-6,
                verbose = false,
                time_ref = time_ref,
                time_limit = time_limit,
            )

            if isIso == nothing
                println()
                @info "Instance can not be solved(time limit)..."
                s1[1] = false
                s2[1] = nBack
                isIso = false
            elseif !isIso
                error("Wrong result ...")
            else
                issolved = true
            end
            s1[1] = isIso
            s2[1] = nBack
        end
        solving_time = t
        corr = mean(s1) * 100
        wobt = count(==(0), s2)
        wbt = count(!=(0), s2)
        mbt = wbt > 0 ? sum(s2) / wbt : 0.0
        @printf "\n Correct: %.3f %%\n Without backtracking: %d\n With backtracking: %d (avg: %.3f steps)\n Time: %.6f\n" corr wobt wbt mbt (
            t / 1
        )
    elseif contains(solver, "boscia")
        reg = contains(solver, "reg") ? true : false
        is_frac32 = occursin("frac32", solver) ? true : false
        fw_iter = contains(solver, "dicg") ? 500 : 1000
        favor_right = true
        if occursin("depth", solver)
            use_depth = true
            favor_right = occursin("left", solver) ? false : true
        else
            use_depth = false
        end
        is_modified = false
        if occursin("gm", solver)
            is_graph_matching = true
            is_modified = occursin("modified", solver) ? true : false
        else
            is_graph_matching = false
        end
        is_graph_matching = occursin("gm", solver) ? true : false
        status, solving_time, result = boscia_run(
            A1,
            A2;
            solver = solver,
            time_limit = time_limit,
            verbose = verbose,
            fw_iter = fw_iter,
            print_iter = print_iter,
            starting_point_dca = starting_point_dca,
            starting_point_fw = starting_point_fw,
            regularized = reg,
            use_depth = use_depth,
            is_graph_matching = is_graph_matching,
            favor_right = favor_right,
            is_modified = is_modified,
            is_frac32 = is_frac32,
            
        )
        if status == "OPTIMAL"
            issolved = true
        elseif status == "TIME_LIMIT"
            @info "Instance can not be solved(time limit)..."
        else
            issolved = true
            @info "Not isomorphic"
        end

        rel_dual_gap = result[:rel_dual_gap]
        abs_dual_gap = result[:dual_gap]
        primal_obj = result[:primal_objective]
        dual_bound = result[:dual_bound]
    elseif contains(solver, "penalty")
        use_package = contains(solver, "FW") ? true : false
        use_qua_as = contains(solver, "qua") ? true : false
        X, solving_time = frank_wolfe_graph_isomorphism(
            A1,
            A2;
            time_limit = time_limit,
            use_package = use_package,
            use_qua_as = use_qua_as,
        )

        if solving_time > time_limit
            @info "Instance can not be solved(time limit)..."
        elseif X === nothing || !isapprox(X * A1, A2 * X; rtol = 1e-6, atol = 1e-6)
            @info "Instance can not be solved..."
        else
            issolved = true
            @printf "\n Solving time : %.6f \n" solving_time
        end

    elseif contains(solver, "FW_full")
        use_qua_as = contains(solver, "qua") ? true : false
        x, status, solving_time = fw_package_graph_isomorphism(
            A1,
            A2;
            time_limit = time_limit,
            use_qua_as = use_qua_as,
        )
        if status == "Optimal"
            issolved = true
        elseif status == "TIMEOUT"
            @info "Instance can not be solved(time limit)..."
        elseif status == "Local OPTIMUM"
            @info "Instance can not be solved(local minimal)..."
        else
            @info "Not isomorphic"
        end
    elseif contains(solver, "dca")
        use_qua_as = contains(solver, "qua") ? true : false
        issolved, solving_time =
            dca_solver(A1, A2, n; time_limit = time_limit, use_qua_as = false)
    elseif contains(solver, "mip")
        use_symmetry = occursin("nosym", solver) ? false : true
        formulation = contains(solver, "l1") ? :l1 : :feasibility
        issolved, solving_time = solve_gi_mip(
            A1,
            A2;
            time_limit = time_limit,
            formulation = formulation,
            use_symmetry = use_symmetry,
            iso_generate = iso_generate,
        )
    end

    if write && issolved
        result_path = "/home/htc/wexiao/project/graph_isomorphism/BenchmarkProblems/julia_benchmarkProblems/slurm/result_$(format)/$solver"
        if !ispath(result_path)
            mkpath(result_path)
        end


        filename = joinpath(result_path, "$(graph)_$(seed).csv")

        if contains(solver, "boscia")
            rel_dual_gap = rel_dual_gap
            abs_dual_gap = abs_dual_gap
            primal_obj = primal_obj
            dual_bound = dual_bound
        else
            rel_dual_gap = NaN
            abs_dual_gap = NaN
            primal_obj = NaN
            dual_bound = NaN
        end

        df = DataFrame(
            graph = [graph],
            time = [solving_time],
            rel_dual_gap = [rel_dual_gap],
            abs_dual_gap = [abs_dual_gap],
            primal_obj = [primal_obj],
            dual_bound = [dual_bound],
        )

        if isfile(filename)
            CSV.write(filename, df; append = true)
        else
            CSV.write(filename, df)
        end
    end
end

# run all solvers
# bench_all(nRuns = 1, solver="FW", time_limit=120,)
# bench_all(nRuns = 1, solver="boscia_FW", time_limit=120, verbose=true)
# bench_all(nRuns = 1, solver="boscia_BPCG", time_limit=120, verbose=true)
# bench_all(nRuns = 1, solver="boscia_DICG", time_limit=120, verbose=true)
# bench_all(nRuns = 1, solver="spectral", time_limit=120,)

# bench("exact_025", 1; format = "dimacs", verbose=true, solver="boscia_bpcg", fw_iter=1000)
# bench("exact_025", 1; format = "dimacs", verbose = true, solver = "boscia_bpcg", fw_iter = 1000, iso_generate=false, print_iter=1)
# for seed in (1:100)
#     bench(
#         "cfi-20",
#         2,
#         "/home/htc/wexiao/project/graph_isomorphism/BenchmarkProblems/more_benchmark/cfixl/cfi-20.dimacs";
#         format = "dimacs",
#         verbose = true,
#         solver = "boscia_dicg",
#         iso_generate = true,
#     )
# end

# bench(
#     "latin_7_49",
#     1,
#     "/home/htc/wexiao/project/graph_isomorphism/BenchmarkProblems/export_mat/latin_7_49.mat",
#     format = "mat",
#     verbose = true,
#     solver = "dca_penalty_solver",
#     write = true,
#     starting_point_dca = false,
#     use_qua_as = true,
#     use_package = false,
#     iso_generate = true,
# )











function bench_all(;
    nRuns::Int = 100,
    verbose = false,
    solver = "spectral",
    time_limit = Inf,
)
    # Names (same order as your MATLAB script)
    # names = String[
    #     # strongly regular graphs
    #     "Lattice_4_16",
    #     "Lattice_6_36",
    #     "latin_2_4",
    #     "latin_4_16",
    #     "latin_6_36",
    #     "latin_3_9",
    #     "latin_5_25",
    #     "latin_7_49",
    #     "paley_power_9",
    #     "paley_power_25",
    #     "paley_prime_13",
    #     "paley_prime_29",
    #     "sts_19_57",
    #     "Triangular_7_21",
    #     "Triangular_10_45",
    #     # component-based graphs
    #     "CHH_cc_1_1_22_1",
    #     "CHH_cc_2_1_44_1",
    #     "tnn_1_26_1",
    #     "tnn_2_52_1",
    #     "usr_1_29_1",
    #     # other graph families
    #     "iso_r01N_s20",
    #     "iso_r01N_s40",
    # ]

    root = "/home/htc/wexiao/project/graph_isomorphism/BenchmarkProblems/export_mat_test"
    names, paths = get_graph_names_paths(root)
    # names = ["CHH_cc_2_2_88_1"]
    total_num_solved = 0
    println("========================================================================")
    @printf "\n # of graphs: %d \n" length(names)
    graph_idx = 0
    for name in names
        graph_idx += 1
        # load graph
        A, n = load_graph(name)
        @assert size(A, 1) == size(A, 2) "Graph $name not square"
        if !issymmetric(A)
            error("Graph $name not undirected (A != A').")
        end
        @printf "\n%s (n = %d): \n" name n
        @printf "\n Solved graphs: %d / %d\n" graph_idx length(names)

        if solver == "spectral"
            s1 = Vector{Bool}(undef, nRuns)   # isIso
            s2 = Vector{Int}(undef, nRuns)    # nBacktracking
            t = @elapsed begin
                time_ref = time()
                for iter = 1:nRuns
                    Random.seed!(iter)
                    if iter % 10 == 0
                        print('*')
                        flush(stdout)
                    end
                    # r1 = randperm(n)
                    # r2 = randperm(n)
                    # A1 = A[r1, r1]
                    # A2 = A[r2, r2]
                    A1, P1 = randomPermutation(Matrix(A))
                    A2, P2 = randomPermutation(Matrix(A))

                    # Your Julia port of isIsomorphic must exist:
                    # isIsomorphic(A, B; eps=1e-6, verbose=false) -> (b, P, nBack)
                    isIso, _, nBack = isIsomorphic(
                        A1,
                        A2;
                        eps = 1e-6,
                        verbose = verbose,
                        time_ref = time_ref,
                        time_limit = time_limit,
                    )

                    if isIso == nothing
                        println()
                        @info "Instance can not be solved(time limit)..."
                        s1[iter] = false
                        s2[iter] = nBack
                    elseif !isIso
                        @show iter
                        error("Wrong result ...")
                    else
                        total_num_solved += 1
                    end
                    s1[iter] = isIso
                    s2[iter] = nBack
                end
            end

            corr = mean(s1) * 100
            wobt = count(==(0), s2)
            wbt = count(!=(0), s2)
            mbt = wbt > 0 ? sum(s2) / wbt : 0.0
            @printf "\n Correct: %.3f %%\n Without backtracking: %d\n With backtracking: %d (avg: %.3f steps)\n Time: %.6f\n" corr wobt wbt mbt (
                t / nRuns
            )
        elseif contains(solver, "boscia")

            solving_time = Float64[]
            num_solved_instances = 0
            for iter = 1:nRuns
                Random.seed!(iter)
                A1, P1 = randomPermutation(Matrix(A))
                A2, P2 = randomPermutation(Matrix(A))
                status, insta_solving_time = boscia_run(
                    A1,
                    A2;
                    solver = solver,
                    time_limit = time_limit,
                    verbose = false,
                )
                if status == "OPTIMAL"
                    num_solved_instances += 1
                    total_num_solved += 1
                    push!(solving_time, insta_solving_time)
                elseif status == "TIME_LIMIT"
                    @info "Instance can not be solved(time limit)..."
                else
                    error("Status error!")
                end
            end

            total_time_in_sec = sum(solving_time)
            time_per_insta =
                total_num_solved == 0 ? Inf : total_time_in_sec / num_solved_instances
            @printf "\n Solved instances: %d\n Solving time per instance: %.6f\n Solving time in total: %.6f\n" num_solved_instances time_per_insta total_time_in_sec
        elseif solver == "FW"
            solving_time = Float64[]
            num_solved_instances = 0
            for iter = 1:nRuns
                Random.seed!(iter)
                A1, P1 = randomPermutation(Matrix(A))
                A2, P2 = randomPermutation(Matrix(A))
                X, insta_solving_time =
                    frank_wolfe_graph_isomorphism(A1, A2; time_limit = time_limit)
                if X == nothing
                    @info "Instance can not be solved(local minimal)..."
                elseif insta_solving_time > time_limit
                    @info "Instance can not be solved(time limit)..."
                elseif A1 ≉ X' * A2 * X
                    error("Status error!")
                else
                    push!(solving_time, insta_solving_time)
                    num_solved_instances += 1
                    total_num_solved += 1
                end
            end
            total_time_in_sec = sum(solving_time)
            time_per_insta =
                num_solved_instances == 0 ? Inf : total_time_in_sec / num_solved_instances
            @printf "\n Solved instances: %d\n Solving time per instance: %.6f\n Solving time in total: %.6f\n" num_solved_instances time_per_insta total_time_in_sec
        elseif solver == "mip"
        end
    end
    @printf "\n Solved instances in total: %d\n" total_num_solved
end
