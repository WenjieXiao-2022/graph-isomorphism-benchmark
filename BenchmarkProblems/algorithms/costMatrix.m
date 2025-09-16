function C = costMatrix(v, w)
    n = length(v);
    C = zeros(n);
    for i = 1:n
        for j = 1:n
            C(i, j) = abs(v(i) - w(j));
        end
    end
end
