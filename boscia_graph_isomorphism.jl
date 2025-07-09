using Boscia
using SparseArrays
using LinearAlgebra
using Bonobo
using FrankWolfe
using MAT
using Random
using HiGHS
using MathOptInterface
const MOI = MathOptInterface

function random_k_neighbor_matrix(tree::Bonobo.BnBTree, blmo::Boscia.TimeTrackingLMO, x, k::Int, use_mip=false)
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

        new_p = use_mip ? vec(new_P) : sparsevec(vec(new_P))# Convert to proper SparseVector
        push!(Ps, new_p)
    end

    return Ps, false
end

function build_birkhoff_mip(n, A, B; cut=false)
    o = HiGHS.Optimizer()
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    X = reshape(MOI.add_variables(o, n^2), n, n)

    MOI.add_constraint.(o, X, MOI.ZeroOne())
    
    # Row sum constraints: each row sums to 1
    for i in 1:n
        row_constraint = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, X[i, j]) for j in 1:n],
            0.0
        )
        MOI.add_constraint(o, row_constraint, MOI.EqualTo(1.0))
    end
    
    # Column sum constraints: each column sums to 1
    for j in 1:n
        col_constraint = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, X[i, j]) for i in 1:n],
            0.0
        )
        MOI.add_constraint(o, col_constraint, MOI.EqualTo(1.0))
    end

    if cut 
        # Add gradient = 0 constraints
        # ∇f(X) = 2(XA - BX)A' - 2B'(XA - BX) = 0
        # This gives us: (XA - BX)A' - B'(XA - BX) = 0
        # Expanding: XAA' - BXA' - B'XA + B'BX = 0
        # Rearranging: X(AA' + B'B) - BXA' - B'XA = 0
        
        # Precompute matrices
        AA_t = A * A'
        BB_t = B' * B
        BA_t = B * A'
        B_tA = B' * A
        
        # For each entry (i,j) of the gradient matrix, add constraint = 0
        for i in 1:n
            for j in 1:n
                # Gradient[i,j] = sum_k (X[i,k] * (AA'[k,j] + B'B[k,j]) - B[i,k] * X[k,l] * A'[l,j] - B'[i,k] * X[k,l] * A[l,j])
                grad_terms = MOI.ScalarAffineTerm{Float64}[]
                
                # Terms from X(AA' + B'B)
                for k in 1:n
                    coeff = AA_t[k,j] + BB_t[k,j]
                    if abs(coeff) > 1e-12
                        push!(grad_terms, MOI.ScalarAffineTerm(coeff, X[i,k]))
                    end
                end
                
                # Terms from -BXA'
                for k in 1:n, l in 1:n
                    coeff = -B[i,k] * A[l,j]  # A'[l,j] = A[j,l] but we want A'[l,j]
                    if abs(coeff) > 1e-12
                        push!(grad_terms, MOI.ScalarAffineTerm(coeff, X[k,l]))
                    end
                end
                
                # Terms from -B'XA  
                for k in 1:n, l in 1:n
                    coeff = -B[k,i] * A[l,j]  # B'[i,k] = B[k,i]
                    if abs(coeff) > 1e-12
                        push!(grad_terms, MOI.ScalarAffineTerm(coeff, X[k,l]))
                    end
                end
                
                if !isempty(grad_terms)
                    grad_constraint = MOI.ScalarAffineFunction(grad_terms, 0.0)
                    MOI.add_constraint(o, grad_constraint, MOI.EqualTo(0.0))
                end
            end
        end
    end
    
    return Boscia.MathOptBLMO(o)
end

function boscia_graph_isomorphism(A, B; print_iter=10, variant=Boscia.DICG(), fw_iter=1000, mip=false, cut=false)
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

    blmo = mip ? build_birkhoff_mip(n, A, B; cut=cut) : Boscia.ManagedBoundedLMO(sblmo, lower_bounds, upper_bounds, collect(1:n^2), n^2)
    
    k = Int(round(sqrt(n)))
    swap_heu = Boscia.Heuristic((tree, blmo, x) -> random_k_neighbor_matrix(tree, blmo, x, k, mip), 1.0, :swap)
    
    x, _, result = Boscia.solve(
        f,
        grad!,
        blmo;
        variant=variant,
        line_search=FrankWolfe.Secant(),
        verbose=true,
        branch_callback=build_branch_callback(),
        bnb_callback=build_tree_callback(),
        print_iter=print_iter,
        custom_heuristics=[swap_heu],
        max_fw_iter=fw_iter,
    )

    X = reshape(x, n, n)
    return X
end