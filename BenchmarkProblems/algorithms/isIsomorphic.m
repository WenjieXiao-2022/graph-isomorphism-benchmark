function [b, P, nBacktracking] = isIsomorphic(A, B, eps, verbose)
    % Check whether the two graphs G_A and G_B with adjacency matrices A and B are isomorphic.
    
    if nargin < 4
        verbose = true; % verbose mode
    end
    if nargin < 3
        eps = 1e-6;     % error tolerance
    end

    n = size(A, 2);    % number of vertices
    nBacktracking = 0; % number of backtracking steps
    
    %%% check if initial graphs are potentially isomorphic
    [~, ~, e0] = isIsomorphicRepeated(A, B, eps, verbose);
    fprintf('e0 = %i\n', e0)
    if e0 > eps
        b = false;
        P = [];
        return;
    end
    
    %%% perturb A and B and try to find assignment
    A0 = A;
    B0 = B;
    c = ones(n);
    P = zeros(n); % permutation matrix
    
    i = 1;
    jStart = 1;
    while i <= n
        c(i, :) = ones(1, n); % reset entries
        for j = jStart:n
            An = perturb(A0, i, i);
            Bn = perturb(B0, j, i);
            
            %%% check distinct eigenvectors and eigenpolytopes at the same time
            [~, ~, e] = isIsomorphicRepeated(An, Bn, eps, verbose);
            
            %%% assignment cost
            c(i, j) = e;
            
            %%% assign i -> j
            if e < eps
                if verbose
                    fprintf('assign %i -> %i\n', i, j);
                end
                
                A0 = An;
                B0 = Bn;
                P(i, j) = 1;
                
                break;
            end
        end
        
        %%% no assignment exists
        cmin = min(c(i, :));
        if cmin > eps
            if i == 1
                b = false;
                P = [];
                return
            else
                if verbose
                    fprintf('backtracking ...\n');
                end
                nBacktracking = nBacktracking + 1;

                %%% remove previous assignment
                A0(i-1, i-1) = 0;
                j = find(diag(B0) == i-1);
                B0(j, j) = 0;

                %%% update permutation matrix
                P(i-1, j) = 0;

                jStart = j+1; % go to the next possible assignment that has not been tested yet
                i = i-2;
            end
        else
            jStart = 1;
        end
        
        i = i+1;
    end
    
    b = true;
   
    e = norm(A - P*B*P', 'fro');
    if e > eps
        error('GI: Wrong permutation matrix!');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = perturb(A, i, k)
    A(i, i) = A(i, i) + k;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
