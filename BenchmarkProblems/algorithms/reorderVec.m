function w = reorderVec(v)
    % Reorder vector and change signs of the entries.
    n = length(v);
    w = -v(n:-1:1);
end
