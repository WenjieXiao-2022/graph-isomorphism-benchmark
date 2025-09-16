using LinearAlgebra
using Statistics
using Random
using Printf
using MAT
# using Hungarian

include("costMatrix.jl")
include("costMatrixRepeated.jl")
include("isIsomorphic.jl")
include("isIsomorphicRepeated.jl")
include("makeSignsConsistent_reorderVec.jl")
include("hungarian.jl")

"""
    load_graph(name::String) -> A, n

Load a benchmark graph adjacency matrix by name.

Expects `.mat` files created from the exported `.m` graph functions,
each containing variables `A` (adjacency matrix) and `n` (number of nodes).
"""
function load_graph(name::String)
    base = "/home/htc/wexiao/project/graph_isomorphism/BenchmarkProblems/export_mat"
    filepath = joinpath(base, name * ".mat")
    if !isfile(filepath)
        error("Graph file not found: $filepath. Did you run the conversion step?")
    end

    data = matread(filepath)
    A = data["A"]
    n = Int(data["n"])
    return A, n
end

function bench_all(; nRuns::Int=100, verbose=false)
    # Names (same order as your MATLAB script)
    names = String[
        # strongly regular graphs
        "Lattice_4_16",
        "Lattice_6_36",
        "latin_2_4",
        "latin_4_16",
        "latin_6_36",
        "latin_3_9",
        "latin_5_25",
        "latin_7_49",
        "paley_power_9",
        "paley_power_25",
        "paley_prime_13",
        "paley_prime_29",
        "sts_19_57",
        "Triangular_7_21",
        "Triangular_10_45",
        # component-based graphs
        "CHH_cc_1_1_22_1",
        "CHH_cc_2_1_44_1",
        "tnn_1_26_1",
        "tnn_2_52_1",
        "usr_1_29_1",
        # other graph families
        "iso_r01N_s20",
        "iso_r01N_s40",
    ]

    for name in names
        # load graph
        A, n = load_graph(name)
        @assert size(A,1) == size(A,2) "Graph $name not square"
        if !issymmetric(A)
            error("Graph $name not undirected (A != A').")
        end

        @printf "%s (n = %d): " name n

        s1 = Vector{Bool}(undef, nRuns)   # isIso
        s2 = Vector{Int}(undef, nRuns)    # nBacktracking

        t = @elapsed begin
            for iter in 1:nRuns
                if iter % 10 == 0
                    print('*'); flush(stdout)
                end
                r1 = randperm(n)
                r2 = randperm(n)
                A1 = A[r1, r1]
                A2 = A[r2, r2]

                # Your Julia port of isIsomorphic must exist:
                # isIsomorphic(A, B; eps=1e-6, verbose=false) -> (b, P, nBack)
                isIso, _, nBack = isIsomorphic(A1, A2; eps=1e-6, verbose=verbose)

                if !isIso
                    error("Wrong result ...")
                end
                s1[iter] = isIso
                s2[iter] = nBack
            end
        end

        corr = mean(s1) * 100
        wobt = count(==(0), s2)
        wbt  = count(!=(0), s2)
        mbt  = wbt > 0 ? sum(s2) / wbt : 0.0

        @printf "\n Correct: %.3f %%\n Without backtracking: %d\n With backtracking: %d (avg: %.3f steps)\n Time: %.6f\n" corr wobt wbt mbt (t / nRuns)
    end
end

# run everything
bench_all(nRuns=100, verbose=false)