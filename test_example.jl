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

# Read the Petersen graph matrix from .mat file
data = matread("Paley29.mat") # Petersen.mat, BiggsSmith.mat, Paley29.mat
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

x = boscia_graph_isomorphism(A, B, print_iter=10, variant=Boscia.BPCG())
@show x

x = boscia_graph_isomorphism(A, B, print_iter=10, variant=Boscia.BPCG(), mip=true)
@show x
