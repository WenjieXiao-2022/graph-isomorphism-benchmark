function C = costMatrixRepeated(E1, E2)
    n = length(E1);
    C = zeros(n);
    
    % sort rows of E1 and E2
    for i = 1:n
        E1(i, :) = sort(E1(i, :));
        E2(i, :) = sort(E2(i, :));
    end
    
    % compare sorted vectors
    for i = 1:n
        for j = 1:n
            C(i, j) = norm(E1(i, :) - E2(j, :));
        end
    end
end
