using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using MAT
using CSV
using DataFrames
using Hungarian
using FrankWolfe
using NautyGraphs

include("utilities.jl")
include("spectral.jl")
include("bosciaGraphIsomorphism.jl")
include("penalty.jl")
include("dca.jl")
include("mip.jl")

function bench(
    graph,
    seed;
    solver = "spectral",
    time_limit = Inf,
    write = false,
    format = "dimacs",
    iso_generate = true,
)
    Random.seed!(seed)
    println("========================================================================")
    # load graph
    A, n = load_graph(graph; format = format)
    issolved = false
    @assert size(A, 1) == size(A, 2) "Graph $graph not square"
    if !issymmetric(A)
        error("Graph $graph not undirected (A != A').")
    end
    @printf "\n%s (n = %d): \n" graph n

    # Initialize variables that may be used later
    result = nothing
    rel_dual_gap = NaN
    abs_dual_gap = NaN
    primal_obj = NaN
    dual_bound = NaN

    if iso_generate
        A1, P1 = randomPermutation(A)
        A2, P2 = randomPermutation(A)
    else
        if occursin("cospectral", graph)
            A1, P1 = randomPermutation(A)
            cospectral_graph = "cospectral_$seed"
            B, n = load_graph(cospectral_graph; format = format)
            @info "The non-isomorphic instance pair is $graph and $(cospectral_graph)..."
            A2, P2 = randomPermutation(B)
        else
            edges_flipped =
                all(isdigit, split(solver, "_")[end]) ?
                parse(Int, split(solver, "_")[end]) : 1
            @show edges_flipped

            A1, P1 = randomPermutation(A)
            A2 = non_iso_graph(A; edges_flipped = edges_flipped)
        end
    end

    if solver == "nauty"
        A_nauty = NautyGraph(A1)
        A1_nauty = NautyGraph(A2)
        solving_time = @elapsed begin
            is_iso = A_nauty ≃ A1_nauty  # true if isomorphic 
        end
        @assert is_iso == iso_generate
        @printf "\n Nauty result: %s\n Time: %.6f\n" (is_iso ? "isomorphic" : "non isomorphic") solving_time
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

            if isIso === nothing
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
        @printf "\n Correct: %.3f %%\n Without backtracking: %d\n With backtracking: %d (avg: %.3f steps)\n Time: %.6f\n" corr wobt wbt mbt t
    elseif contains(solver, "boscia")

        solver_parts = split(solver, "_")

        use_star = "star" in solver_parts

        use_clique = "clique" in solver_parts

        use_OBBT = "OBBT" in solver_parts

        favor_right = nothing
        if "DFS" in solver_parts
            use_depth = true
            favor_right = "left" in solver_parts ? false : true
        else
            use_depth = false
        end

        is_graph_matching = "GM" in solver_parts 

        iso_generate ? println("Iso problem...") : println("Non-iso problem...")
        status, solving_time, fixing_res, result = boscia_run(
            A1,
            A2;
            solver = solver,
            time_limit = time_limit,
            use_depth = use_depth,
            is_graph_matching = is_graph_matching,
            favor_right = favor_right,
            iso_generate = iso_generate,
            use_OBBT = use_OBBT,
            use_clique = use_clique,
            use_star = use_star,
        )
        if status == "OPTIMAL"
            issolved = true
        elseif status == "TIME_LIMIT"
            @info "Instance can not be solved(time limit)..."
        else
            issolved = true
            @info "Not isomorphic"
        end
    elseif contains(solver, "penalty")
        X, solving_time = frank_wolfe_graph_isomorphism(
            A1,
            A2;
            time_limit = time_limit,
        )

        if solving_time > time_limit
            @info "Instance can not be solved(time limit)..."
        elseif X === nothing || !isapprox(X * A1, A2 * X; rtol = 1e-6, atol = 1e-6)
            @info "Instance can not be solved..."
        else
            issolved = true
            @printf "\n Solving time : %.6f \n" solving_time
        end
    elseif contains(solver, "dca")
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
        result_path = "./result_$(format)/$solver"
        if !ispath(result_path)
            mkpath(result_path)
        end


        filename = joinpath(result_path, "$(graph)_$(seed).csv")

        if contains(solver, "boscia")
            if result !== nothing
                rel_dual_gap = result[:rel_dual_gap]
                abs_dual_gap = result[:dual_gap]
                primal_obj = result[:primal_objective]
                dual_bound = result[:dual_bound]
            else
                rel_dual_gap = NaN
                abs_dual_gap = NaN
                primal_obj = NaN
                dual_bound = NaN
            end
        end

        if occursin("fixings", solver) ||
           occursin("clique", solver) ||
           occursin("star", solver)

            times_tuple, iters_vec, num_checked_ws, fixed_zero_tuple, num_fixed_to_one =
                fixing_res
            # Total warm-start time (fixings + clique + star)
            fixing_time_val = sum(times_tuple)

            df = DataFrame(
                graph = [graph],
                time = [solving_time],
                fixing_time = [fixing_time_val],
                # number_of_fixings counts only actual fixings iterations
                number_of_fixings = [length(iters_vec)],
                iters_to_fixings = [iters_vec],
                num_checked = [num_checked_ws],
                num_fixed_to_zero = [fixed_zero_tuple],
                num_fixed_to_one = [num_fixed_to_one],
                rel_dual_gap = [rel_dual_gap],
                abs_dual_gap = [abs_dual_gap],
                primal_obj = [primal_obj],
                dual_bound = [dual_bound],
            )
        else
            df = DataFrame(
                graph = [graph],
                time = [solving_time],
                rel_dual_gap = [rel_dual_gap],
                abs_dual_gap = [abs_dual_gap],
                primal_obj = [primal_obj],
                dual_bound = [dual_bound],
            )
        end

        CSV.write(filename, df)
    end
end
