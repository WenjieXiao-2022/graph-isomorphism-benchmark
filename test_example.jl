using Boscia
using SparseArrays
using LinearAlgebra
using Bonobo
using FrankWolfe
using MAT
using Random

include("boscia_graph_isomorphism.jl")
include("frank_wolfe_graph_isomorphism.jl")

function randomPermutation(A)
    n = size(A, 1)
    p = randperm(n)
    I = Matrix(LinearAlgebra.I, n, n)
    P = I[:, p]
    B = P * A * P'
    return B, P
end

function random_k_neighbor_matrix(tree::Bonobo.BnBTree, blmo::Boscia.TimeTrackingLMO, x, k::Int)
    P = tree.incumbent_solution.solution
    n0 = size(P, 1)
    n = Int(sqrt(n0))
    P = reshape(P, n, n)
    new_P = copy(P)

    Ps = []

    for _ in 1:k
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
        push!(Ps, new_P)
        @show typeof(new_P)
    end

    return Ps, false
end

# Read the Petersen graph matrix from .mat file
data = matread("Petersen.mat") # Petersen.mat, BiggsSmith.mat, Paley29.mat
A = sparse(data["M"])  # The matrix is stored as "M" in the file
n = Int(data["n"])     # The size parameter

println("Matrix size: ", size(A))
println("n parameter: ", n)
println("Matrix sparsity: ", nnz(A), " non-zeros out of ", length(A), " total elements") 

# Create a permuted version of A for the graph isomorphism problem
B, P = randomPermutation(Matrix(A))
B = sparse(B)

x_fw = frank_wolfe_graph_isomorphism(A, B, nIter=1000)
@show x_fw

x = boscia_graph_isomorphism(A, B, print_iter=10, variant=Boscia.DICG())
@show x
