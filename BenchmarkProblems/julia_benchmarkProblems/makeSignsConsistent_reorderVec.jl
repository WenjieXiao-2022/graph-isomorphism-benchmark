reorderVec(v::AbstractVector) = -reverse(vec(v))

function makeSignsConsistent(v1::AbstractVector, v2::AbstractVector; eps::Float64=1e-6)
    v1 = vec(v1); v2 = vec(v2)
    v1s = sort(v1)
    v2s = sort(v2)

    e1 = norm(v1s .- v2s)
    e2 = norm(v1s .- reorderVec(v2s))

    if e1 < eps && e2 < eps
        v1 = abs.(v1)
        v2 = abs.(v2)
        e = 0.0
    elseif e1 < eps && e2 > eps
        e = 0.0
    elseif e1 > eps && e2 < eps
        v2 = -v2
        e = 0.0
    else
        e = 1.0
    end
    return v1, v2, e
end
