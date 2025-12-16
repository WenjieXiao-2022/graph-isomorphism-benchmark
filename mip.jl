using JuMP
using LinearAlgebra
using SCIP


"""
    build_gi_mip_model(A, B; solver=:SCIP, time_limit=60.0, formulation=:feasibility)

Build a MIP model for graph isomorphism.

formulation:
    :feasibility  -> original model: XA == BX, objective = 0
    :l1           -> lienarized model: minimize sum(S_ij) with |XA - BX| <= S_ij
"""
function build_gi_mip_model(
    A::AbstractMatrix,
    B::AbstractMatrix;
    solver::Symbol = :SCIP,
    time_limit::Real = 60.0,
    formulation::Symbol = :feasibility,
    use_symmetry::Bool = true,
)
    nA1, nA2 = size(A)
    nB1, nB2 = size(B)
    @assert nA1 == nA2 == nB1 == nB2 "A and B must be square matrices of the same size"
    n = nA1

    # --- Choose solver ---
    model = if solver == :SCIP
        @info "Using SCIP as MIP solver"
        Model(SCIP.Optimizer)
    elseif solver == :HiGHS
        @info "Using HiGHS as MIP solver"
        Model(HiGHS.Optimizer)
    else
        error("Unknown solver = $solver. Use :SCIP or :HiGHS.")
    end

    use_symmetry ? nothing : JuMP.set_optimizer_attribute(model, "misc/usesymmetry", 0)

    # --- Time limit ---
    if time_limit < Inf
        if solver == :SCIP
            set_optimizer_attribute(model, "limits/time", time_limit)
        elseif solver == :HiGHS
            set_optimizer_attribute(model, "time_limit", time_limit)
        end
    end

    # --- Variables: permutation matrix X ---
    @variable(model, X[1:n, 1:n], Bin)

    # --- Permutation constraints: X ∈ 𝒫 ---
    @constraint(model, [i = 1:n], sum(X[i, j] for j = 1:n) == 1)  # each row sums to 1
    @constraint(model, [j = 1:n], sum(X[i, j] for i = 1:n) == 1)  # each column sums to 1

    if formulation == :feasibility
        @info "feasibility formulation"
        # === ORIGINAL MODEL: feasibility, XA = BX ===
        @constraint(
            model,
            [i = 1:n, j = 1:n],
            sum(X[i, k] * A[k, j] for k = 1:n) == sum(B[i, k] * X[k, j] for k = 1:n)
        )

        @objective(model, Min, 0.0)

    elseif formulation == :l1
        # === L1 MODEL: linearized formulation ===
        # min ∑_ij S_ij
        # s.t.  XA - BX <= S,  BX - XA <= S,  S >= 0
        @info "linearized formulation"
        @variable(model, S[1:n, 1:n] >= 0)

        @constraint(
            model,
            [i = 1:n, j = 1:n],
            sum(X[i, k] * A[k, j] for k = 1:n) - sum(B[i, k] * X[k, j] for k = 1:n) <=
            S[i, j]
        )

        @constraint(
            model,
            [i = 1:n, j = 1:n],
            sum(B[i, k] * X[k, j] for k = 1:n) - sum(X[i, k] * A[k, j] for k = 1:n) <=
            S[i, j]
        )

        @objective(model, Min, sum(S[i, j] for i = 1:n, j = 1:n))
    else
        error("Unknown formulation: $formulation. Use :feasibility or :l1.")
    end

    return model, X
end

function solve_gi_mip(
    A::AbstractMatrix,
    B::AbstractMatrix;
    solver = :SCIP,
    time_limit = 60.0,
    formulation::Symbol = :feasibility,
    use_symmetry::Bool = true,
    iso_generate::Bool = true,
)

    t0 = time()
    model, X = build_gi_mip_model(
        A,
        B;
        solver = solver,
        time_limit = time_limit,
        formulation = formulation,
        use_symmetry = use_symmetry,
    )

    optimize!(model)

    solving_time = time() - t0
    term_status = termination_status(model)

    println("Termination status: ", term_status)

    if term_status == MOI.OPTIMAL || term_status == MOI.FEASIBLE_POINT
        n = size(A, 1)
        Xval = [value(X[i, j]) for i = 1:n, j = 1:n]

        # Round to get a clean permutation matrix (numerical safety)
        Xperm = map(x -> x > 0.5 ? 1 : 0, Xval)

        println("Found feasible permutation matrix X.")
        # Optional: verify AX = XB numerically
        XA = Xperm * A
        BX = B * Xperm
        residual = XA - BX
        resnorm = sum(abs2, residual)

        println("‖XA - BX‖_F^2 = ", resnorm)
        println("‖X‖_F^2 = ", sum(abs2, Xperm))

        if iso_generate && (!isapprox(XA, BX; atol = 1e-6, rtol = 1e-6) || !(sum(abs2, Xperm) == n))
            println("Incorrect permutation matrix is returned")
            return false, solving_time
        end

        return true, solving_time
    elseif term_status == MOI.INFEASIBLE
        println("No feasible points...")
        println("Graphs are non-isomorphic!")
        return !iso_generate, solving_time
    else
        println("No feasible permutation matrix found (graphs likely non-isomorphic).")
        return false, solving_time
    end
end
