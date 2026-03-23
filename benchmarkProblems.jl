using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using MAT
using CSV
using DataFrames

include("utilities.jl")
include("bosciaGraphIsomorphism.jl")
include("mip.jl")

function bench(
    seed;
    solver = "boscia_DFS",
    time_limit = Inf,
    write = false,
    nauty_graph = "",
    is_GOE_graph = true,
    num_nodes_GOE = 10,
    noise_level = nothing
)
    Random.seed!(seed)
    println("========================================================================")

    issolved = false
    result = nothing
    rel_dual_gap = NaN
    abs_dual_gap = NaN
    primal_obj = NaN
    dual_bound = NaN
    fixing_res = nothing
    P = nothing  # planted permutation (GOE); used only to compare optimal objective, not the iterate

    if is_GOE_graph
        n = num_nodes_GOE

        if noise_level == "small_perm"
            noise_threshold_GOE = -1.0
            GOE_epsilon = 0.1
        elseif noise_level == "intermediate"
            noise_threshold_GOE = -1.0
            GOE_epsilon = -0.1
        elseif noise_level == "well_sep"
            noise_threshold_GOE = -0.5
            GOE_epsilon = -0.1
        elseif noise_level === nothing
            error("Please provide the nosie level.")
        else
            error("$(noise_level) is not supoorted.")
        end

        A1, A2, P, _true_mapping = generate_easy_boscia_instance(
            n;
            noise_threshold = noise_threshold_GOE,
            epsilon = GOE_epsilon,
        )
        @printf "\nGOE synthetic (n = %d): \n" n
    else
        A, n = load_graph(nauty_graph)
        @assert size(A, 1) == size(A, 2) "Graph $(nauty_graph) not square"
        if !issymmetric(A)
            error("Graph $(nauty_graph) not undirected (A != A').")
        end
        @printf "\n%s (n = %d): \n" nauty_graph n
        # Two independent relabelings of the same graph (feasible graph-matching instance)
        A1, _ = randomPermutation(A)
        A2, _ = randomPermutation(A)
    end

    if contains(solver, "boscia")

        solver_parts = split(solver, "_")

        use_reduced_cost = "rdc" in solver_parts

        favor_right = nothing
        if "DFS" in solver_parts
            use_depth = true
            favor_right = "left" in solver_parts ? false : true
        else
            use_depth = false
        end

        println("Graph matching instance...")
        status, solving_time, fixing_res, result = boscia_run(
            A1,
            A2;
            solver = solver,
            time_limit = time_limit,
            use_depth = use_depth,
            favor_right = favor_right,
            use_reduced_cost = use_reduced_cost,
        )
        if status == "OPTIMAL"
            issolved = true
        elseif status == "TIME_LIMIT"
            @info "Instance can not be solved(time limit)..."
        else
            @info "Solver finished with status: $status"
        end
    elseif contains(solver, "mip")
        use_symmetry = occursin("nosym", solver) ? false : true
        formulation = contains(solver, "l1") ? :l1 : :feasibility
        issolved, solving_time = solve_gi_mip(
            A1,
            A2;
            time_limit = time_limit,
            formulation = formulation,
            use_symmetry = use_symmetry,
        )

    else
        error("Unknown solver $(repr(solver)). Use mip or a boscia_* name.")
    end

    if write && issolved
        result_path = "/home/htc/wexiao/project/graph_isomorphism/slurm_script/GM/results/$(noise_level)/$solver"
        if !ispath(result_path)
            mkpath(result_path)
        end


        filename = joinpath(result_path, "$(num_nodes_GOE)_$(seed).csv")

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

        if occursin("rdc", solver)
            times_tuple, iters_vec, num_checked_ws, fixed_zero_tuple, num_fixed_to_one =
                fixing_res
            fixing_time_val = sum(times_tuple)

            df = DataFrame(
                graph = [nauty_graph],
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
                graph = [nauty_graph],
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
