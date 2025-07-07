using Boscia
using SparseArrays
using LinearAlgebra
using Bonobo
using FrankWolfe
using MAT
using Random

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
    end

    return Ps, false
end

function boscia_graph_isomorphism(A, B; print_iter=10, variant=Boscia.DICG())
    n = size(A, 1)
    function f(x)
        X = reshape(x, n, n)
        return norm(X * A - B * X)
    end
    
    function grad!(storage, x)
        X = reshape(x, n, n)
        grad_matrix = 2 * (X * A - B * X) * A' - 2 * B' * (X * A - B * X)
        storage .= vec(grad_matrix)
    end
    
    function build_branch_callback()
        return function (tree, node, vidx::Int)
            x = Bonobo.get_relaxed_values(tree, node)
            primal = tree.root.problem.f(x)
            lower_bound = primal - node.dual_gap
            if lower_bound > 0.0 + eps()
                println("No need to branch here. Node lower bound already positive.")
            end
            return !(lower_bound > 0.0 + eps())
        end
    end
    
    
    function build_tree_callback()
        return function (tree, node; worse_than_incumbent=false, node_infeasible=false, lb_update=false)
            if isapprox(tree.incumbent, 0.0, atol=eps())
                tree.root.problem.solving_stage = Boscia.USER_STOP
                println("Optimal solution found.")
            end
            if Boscia.tree_lb(tree::Bonobo.BnBTree) > 0.0 + eps()
                tree.root.problem.solving_stage = Boscia.USER_STOP
                println("Tree lower bound already positive. No solution possible.")
            end
        end
    end
    
    sblmo = Boscia.BirkhoffBLMO(n, collect(1:n^2))
    
    lower_bounds = fill(0.0, n^2)
    upper_bounds = fill(1.0, n^2)
    
    k = Int(round(sqrt(n)))
    swap_heu = Boscia.Heuristic((tree, blmo, x) -> random_k_neighbor_matrix(tree, blmo, x, k), 1.0, :swap)
    
    x, _, result = Boscia.solve(
        f,
        grad!,
        sblmo,
        lower_bounds,
        upper_bounds,
        collect(1:n^2),
        n^2;
        variant=variant,
        line_search=FrankWolfe.Secant(),
        verbose=true,
        branch_callback=build_branch_callback(),
        bnb_callback=build_tree_callback(),
        print_iter=print_iter,
    )

    X = reshape(x, n, n)
    return X
end