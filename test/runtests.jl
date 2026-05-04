using Test
using GI_benchmark

@testset "GI_benchmark package smoke tests" begin
    A, n = load_graph("exact_001")
    @test size(A, 1) == n
    @test size(A, 2) == n

    B, P = randomPermutation(A)
    @test size(B) == size(A)
    @test size(P) == (n, n)
end
