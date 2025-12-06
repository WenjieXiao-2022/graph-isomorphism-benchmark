# Hungarian algorithm (MATLAB Alex Melin version → Julia)
# Returns:
#   Matching :: Matrix{Int}  (same size as Perf, 0/1 mask)
#   Cost     :: Float64      (sum of Perf where Matching == 1)
function hungarian(Perf::AbstractMatrix{<:Real})
    m, n = size(Perf)
    Matching = zeros(Int, m, n)

    # --- Condense matrix by removing unconnected rows/cols (all Inf) ---
    num_y = vec(sum(.!isinf.(Perf), dims=1))  # per column count of finite entries
    num_x = vec(sum(.!isinf.(Perf), dims=2))  # per row
    x_con = findall(!=(0), num_x)             # connected row indices
    y_con = findall(!=(0), num_y)             # connected col indices

    P_size = max(length(x_con), length(y_con))
    P_cond = zeros(Float64, P_size, P_size)
    if P_size == 0
        return Matching, 0.0
    end
    # Place the connected submatrix into P_cond
    P_cond[1:length(x_con), 1:length(y_con)] .= Float64.(Perf[x_con, y_con])

    # Ensure a perfect matching exists (pad if necessary)
    Edge = copy(P_cond)
    Edge[P_cond .!= Inf] .= 0.0
    cnum = min_line_cover(Edge)  # deficiency

    # Project additional vertices/edges so a perfect matching exists
    finite_mask = P_cond .!= Inf
    Pmax = maximum(P_cond[finite_mask])  # safe: connected implies at least one finite
    P_size2 = size(P_cond, 1) + cnum
    P_pad = fill(Pmax, P_size2, P_size2)
    P_pad[1:size(P_cond,1), 1:size(P_cond,2)] .= P_cond
    P_cond = P_pad

    # --- Main program: steps 1-6 loop ---
    stepnum = 1
    M = zeros(Int, size(P_cond))  # mask: 1 = starred zero, 2 = primed zero
    r_cov = zeros(Int, size(P_cond, 1))  # covered rows (0/1)
    c_cov = zeros(Int, size(P_cond, 2))  # covered cols (0/1)

    Z_r = 0
    Z_c = 0
    exit_flag = true
    while exit_flag
        if stepnum == 1
            P_cond, stepnum = step1(P_cond)
        elseif stepnum == 2
            r_cov, c_cov, M, stepnum = step2(P_cond)
        elseif stepnum == 3
            c_cov, stepnum = step3(M, size(P_cond,1))
        elseif stepnum == 4
            M, r_cov, c_cov, Z_r, Z_c, stepnum = step4(P_cond, r_cov, c_cov, M)
        elseif stepnum == 5
            M, r_cov, c_cov, stepnum = step5(M, Z_r, Z_c, r_cov, c_cov)
        elseif stepnum == 6
            P_cond, stepnum = step6(P_cond, r_cov, c_cov)
        elseif stepnum == 7
            exit_flag = false
        else
            error("Invalid step number: $stepnum")
        end
    end

    # Uncondense: place M back into the original Matching size
    kx = length(x_con)
    ky = length(y_con)
    if kx > 0 && ky > 0
        Matching[x_con, y_con] .= (M[1:kx, 1:ky] .== 1)
    end

    # Cost in the original Perf
    if any(Matching .== 1)
        Cost = sum(Float64.(Perf)[Matching .== 1])
    else
        Cost = 0.0
    end

    return Matching, Cost
end

# =========================
# Step functions (1–6)
# =========================

# STEP 1: Row reduction: subtract each row's minimum
function step1(P_cond::Matrix{Float64})
    P = copy(P_cond)
    for i in 1:size(P,1)
        rmin = minimum(@view P[i, :])
        P[i, :] .-= rmin
    end
    return P, 2
end

# STEP 2: Star zeros greedily if their row/col has no star yet
function step2(P_cond::Matrix{Float64})
    n = size(P_cond, 1)
    r_cov = zeros(Int, n)
    c_cov = zeros(Int, n)
    M = zeros(Int, n, n)

    @inbounds for i in 1:n
        for j in 1:n
            if P_cond[i, j] == 0.0 && r_cov[i] == 0 && c_cov[j] == 0
                M[i, j] = 1   # star it
                r_cov[i] = 1
                c_cov[j] = 1
            end
        end
    end
    # Re-initialize covers (clear them)
    r_cov .= 0
    c_cov .= 0
    return r_cov, c_cov, M, 3
end

# STEP 3: Cover each column containing a starred zero; if all covered, done
function step3(M::AbstractMatrix{Int}, P_size::Int)
    c_cov = vec(sum(M, dims=1)) .> 0   # Bool vector
    if sum(c_cov) == P_size
        return Int.(c_cov), 7
    else
        return Int.(c_cov), 4
    end
end

# STEP 4: Prime uncovered zero; if no starred zero in its row -> Step 5
#         else cover the row and uncover the col of that starred zero; loop.
function step4(P_cond::Matrix{Float64}, r_cov::Vector{Int}, c_cov::Vector{Int}, M::Matrix{Int})
    n = size(P_cond, 1)
    rr = copy(r_cov)
    cc = copy(c_cov)
    MM = copy(M)

    zflag = true
    Z_r = 0
    Z_c = 0
    while zflag
        # find first uncovered zero, scanning row-major
        row = 0; col = 0
        ii = 1; jj = 1
        exit_search = false
        while !exit_search
            if P_cond[ii, jj] == 0.0 && rr[ii] == 0 && cc[jj] == 0
                row = ii; col = jj; exit_search = true
            end
            jj += 1
            if jj > n
                jj = 1
                ii += 1
            end
            if ii > n
                exit_search = true
            end
        end

        if row == 0
            # no uncovered zeros -> Step 6
            return MM, rr, cc, 0, 0, 6
        else
            # prime it
            MM[row, col] = 2
            # is there a starred zero in this row?
            zcol = findfirst(==(1), @view MM[row, :])
            if zcol !== nothing
                rr[row] = 1
                cc[zcol] = 0
            else
                # go to Step 5 with this primed zero as Z0
                Z_r = row
                Z_c = col
                return MM, rr, cc, Z_r, Z_c, 5
            end
        end
    end
    return MM, rr, cc, Z_r, Z_c, 6  # fallback
end

# STEP 5: Build alternating path of primed/starred zeros, flip stars along the path
function step5(M::Matrix{Int}, Z_r::Int, Z_c::Int, r_cov::Vector{Int}, c_cov::Vector{Int})
    # Z0 = (Z_r, Z_c) is the uncovered primed zero found in Step 4
    zr = Int[]
    zc = Int[]
    push!(zr, Z_r); push!(zc, Z_c)

    zflag = true
    while zflag
        # find starred zero in column of last primed zero
        rindex = findfirst(==(1), @view M[:, zc[end]])
        if rindex !== nothing
            # Z1: starred zero
            push!(zr, rindex)
            push!(zc, zc[end])
            # find primed zero in row of that starred zero
            cindex = findfirst(==(2), @view M[zr[end], :])
            @assert cindex !== nothing "Expected a primed zero in the row of a starred zero."
            push!(zr, zr[end])
            push!(zc, cindex)
        else
            zflag = false
        end
    end

    # Unstar the starred zeros and star the primed zeros along the path
    for t in eachindex(zr)
        i = zr[t]; j = zc[t]
        if M[i, j] == 1
            M[i, j] = 0
        else
            M[i, j] = 1
        end
    end

    # Clear covers
    r_cov .= 0
    c_cov .= 0

    # Remove all primes
    M[M .== 2] .= 0

    return M, r_cov, c_cov, 3
end

# STEP 6: Adjust the matrix: add min uncovered to covered rows, subtract from uncovered cols
function step6(P_cond::Matrix{Float64}, r_cov::Vector{Int}, c_cov::Vector{Int})
    rows_uncovered = findall(==(0), r_cov)
    cols_uncovered = findall(==(0), c_cov)
    @assert !isempty(rows_uncovered) && !isempty(cols_uncovered) "Step 6 needs uncovered rows/cols."

    minval = minimum(P_cond[rows_uncovered, cols_uncovered])

    # add to covered rows
    rows_covered = findall(==(1), r_cov)
    if !isempty(rows_covered)
        P_cond[rows_covered, :] .+= minval
    end
    # subtract from uncovered columns
    if !isempty(cols_uncovered)
        P_cond[:, cols_uncovered] .-= minval
    end

    return P_cond, 4
end

# --- Utility: deficiency via minimum line cover (Steps 2–4) ---
function min_line_cover(Edge::Matrix{Float64})
    r_cov, c_cov, M, stepnum = step2(Edge)
    c_cov, stepnum = step3(M, size(Edge,1))
    M, r_cov, c_cov, Z_r, Z_c, stepnum = step4(Edge, r_cov, c_cov, M)
    # Deficiency
    return size(Edge,1) - sum(r_cov) - sum(c_cov)
end
