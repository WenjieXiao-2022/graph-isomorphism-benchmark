using Boscia
using SparseArrays
using LinearAlgebra

# Toy example with the Petersen graph.
# Petersen graph
row_idxs_A = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8]
col_idxs_A = [2, 5, 6, 3, 7, 4, 8, 5, 9, 10, 8, 9, 9, 10, 10]

values_A = fill(1.0, length(row_idxs_A))

A = sparse(row_idxs_A, col_idxs_A, values_A, 10, 10)
A = Symmetric(A)

# strongly regular graph
row_idxs_B = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 8]
col_idxs_B = [4, 6, 7, 3, 4, 9, 7, 8, 10, 7, 9, 10, 8, 9, 10]
values_B = fill(1.0, length(row_idxs_B))
B = sparse(row_idxs_B, col_idxs_B, values_B, 10, 10)
B = Symmetric(B)

# sanity check
for i in 1:10
    @assert length(findall(x -> x == i, row_idxs_A)) + length(findall(x -> x == i, col_idxs_A)) == 3 "Sum entries: $(length(findall(x -> x == i, row_idxs_A))) + $(length(findall(x -> x == i, col_idxs_A))) == 3"
    @assert length(findall(x -> x == i, row_idxs_B)) + length(findall(x -> x == i, col_idxs_B)) == 3 "Sum entries: $(length(findall(x -> x == i, row_idxs_B))) + $(length(findall(x -> x == i, col_idxs_B))) == 3"
end


function f(x)

    return norm(X * A + B * X)
end

function grad!(storage, x)

    storage .= 2 * (X * A + B * X) * A' + 2 * B' * (X * A + B * X)
end

blmo = Boscia.BirkhoffBLMO()

result = Boscia.solve(
    f,
    grad!,
    Boscia.BirkhoffBLMO(),
    fw_variant=Boscia.DICG(),
    verbose=true,
    time_limit=60,
)
