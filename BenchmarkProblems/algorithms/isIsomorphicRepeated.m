function [b, P, e] = isIsomorphicRepeated(A, B, eps, verbose)
    % Check whether the two graphs G_A and G_B with adjacency matrices A and B are isomorphic.
    
    if nargin < 4
        verbose = true; % verbose mode
    end
    if nargin < 3
        eps = 1e-6;     % error tolerance
    end
    
    n = size(A, 2); % number of vertices
    
    [V1, D1] = eig(A);
    [V2, D2] = eig(B);
    [lambda1, s1] = sort(diag(D1), 'ascend');
    [lambda2, s2] = sort(diag(D2), 'ascend');
    V1 = V1(:, s1);
    V2 = V2(:, s2);

    %%% check if eigenvalues of A and B are identical
    e = norm(lambda1 - lambda2, 'fro');
    if e > eps
        b = false;
        P = [];
        return;
    end

    %%% check if repeated eigenvalues exist (eigenvalues must be sorted)
    distinct = true(1, n);
    number = 1:n;
    for i = 2:n
        if abs(lambda1(i) - lambda1(i-1)) < eps
            distinct(i)   = false;
            distinct(i-1) = false;
            number(i) = number(i-1);
        end
    end
    
    %%% distinct and repeated eigenvalues
    C = zeros(n); % cost matrix
    ind = unique(number);
    t = 1
    for k = ind
        v1 = V1(:, number == k);
        v2 = V2(:, number == k);
        
        if all(distinct(number == k)) % distinct eigenvalues
            [w1, w2] = makeSignsConsistent(v1, v2);
            if false
                % Print v1, v2, w1, w2 in Julia syntax with full precision
                fprintf("v1_mat = ["); fprintf("%.16f ", v1(1:end-1)); fprintf("%.16f];\n", v1(end));
                fprintf("v2_mat = ["); fprintf("%.16f ", v2(1:end-1)); fprintf("%.16f];\n", v2(end));
                fprintf("w1_mat = ["); fprintf("%.16f ", w1(1:end-1)); fprintf("%.16f];\n", w1(end));
                fprintf("w2_mat = ["); fprintf("%.16f ", w2(1:end-1)); fprintf("%.16f];\n", w2(end));
            end
            Ck = costMatrix(w1, w2);
        else                           % repeated eigenvalues
            E1 = v1*v1'; % eigenpolytopes of A
            E2 = v2*v2'; % eigenpolytopes of B
            Ck = costMatrixRepeated(E1, E2);
            if t == 8
                fprintf("E1_mat = [");
                for i = 1:size(E1,1)
                    fprintf("%.16f ", E1(i,1:end-1));
                    fprintf("%.16f", E1(i,end));
                    if i < size(E1,1)
                        fprintf(";\n     ");  % row separator
                    else
                        fprintf("];\n");      % close matrix
                    end
                end
                fprintf("E2_mat = [");
                for i = 1:size(E2,1)
                    fprintf("%.16f ", E2(i,1:end-1));
                    fprintf("%.16f", E2(i,end));
                    if i < size(E1,1)
                        fprintf(";\n     ");  % row separator
                    else
                        fprintf("];\n");      % close matrix
                    end
                end
                fprintf("\n")
                fprintf("Ck_mat = [");
                for i = 1:size(Ck,1)
                    fprintf("%.16f ", Ck(i,1:end-1));
                    fprintf("%.16f", Ck(i,end));
                    if i < size(Ck,1)
                        fprintf(";\n     ");  % row separator
                    else
                        fprintf("];\n");      % close matrix
                    end
                end
                error("stop here!");
            end
        end
        C = C + Ck;
        if false
            fprintf("C_mat = [");
            for i = 1:size(Ck,1)
                fprintf("%.16f ", C(i,1:end-1));
                fprintf("%.16f", C(i,end));
                if i < size(C,1)
                    fprintf(";\n     ");  % row separator
                else
                    fprintf("];\n");      % close matrix
                end
            end
            fprintf("\n")
            fprintf("Ck_mat = [");
            for i = 1:size(Ck,1)
                fprintf("%.16f ", Ck(i,1:end-1));
                fprintf("%.16f", Ck(i,end));
                if i < size(Ck,1)
                    fprintf(";\n     ");  % row separator
                else
                    fprintf("];\n");      % close matrix
                end
            end
            error("stop here!")
        end

        t = t+1;
    end
    
    if verbose
        figure(1); %plotCostMatrix(C);
        fprintf('Cost matrix: min=%.6g  max=%.6g  mean=%.6g\n', min(C(:)), max(C(:)), mean(C(:)));
    end

    
    P = hungarian(C);
    e = trace(P'*C);
    
    b = e < eps; % graphs isomorphic if e < eps

end
