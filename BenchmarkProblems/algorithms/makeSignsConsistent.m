function [v1, v2, e] = makeSignsConsistent(v1, v2)
    eps = 1e-6;
    
    v1s = sort(v1);
    v2s = sort(v2);
    
    e1 = norm(v1s - v2s);
    e2 = norm(v1s - reorderVec(v2s));
    
    if e1 < eps && e2 < eps
        % warning('Unfriendly eigenvectors, absolute values were used!');
        v1 = abs(v1);
        v2 = abs(v2);
        e = 0;
    elseif e1 < eps && e2 > eps
        % signs are consistent ...
        e = 0;
    elseif e1 > eps && e2 < eps
        % change sign of v2
        v2 = -v2;
        e = 0;
    else
        % vectors are different, cannot make signs consistent ...
        e = 1;
    end
end
