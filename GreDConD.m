function [ A, B ] = GreDConD( I, no_of_factors )
% GREDCOND implements GreDConD algorithm for Boolean matrix factorization 

% usage: [A, B] = GreDConD(I);
% returns A \circ B = I (if the no. of factors is not limited)

% if you are using this implementation please cite the following work
% Trnecka M., Trneckova M.: 
% Simple Structure in Boolean Matrix Factorization 
% Submited to CLA 2020


M = logical(I);
[m, n] = size(M);
U = M;
k = 0;

A = logical([]);
B = logical([]);

while any(any(U))
    v = 0;
    d = false(1,n);
    d_old = false(1,n);
    d_mid = false(1,n);
    e = true(m,1); % extent for speed closure
    
    atr = find(sum(U)>0); % only not covered attributes
    
    while 1
        for j=atr
            if ~d(j)
                % computes the value of the cover function for the candidate factor
                % inline function for speed
                % arrow down (speed version)
                a = e & M(:,j);
                % arrow ups
                sum_a = sum(a);
                if sum_a*n > v % check the size of upper bound
                    b = all(M(a,:),1);
                
                    if sum_a*sum(b) > v % check the size of upper bound
                        p = diversity(B,b);
                        cost = p*sum(sum(U(a,b)));
                        
                
                        if cost > v
                            v = cost;
                            d_mid = b;
                            c = a;
                        end
                    end
                end
            end
        end
        
        d = d_mid;
        e = c;
        
        if all(d==d_old)
            break;
        else
            d_old = d;
        end
    end
    
    A = [A, c];
    B = [B; d];
    
    k = k + 1;
    display(k);
    
    % end if the no. of factors is reached
    if nargin==2 && k==no_of_factors
        break;
    end
    
    % delete already covered part
    U(c, d) = 0;
end
end

function p = diversity(B,b)
p = 1;
[m,~]=size(B);
if m>0
    
    x =   sum(B & repmat(b, m, 1), 2) ./ sum(B, 2);
    y =   sum(B & repmat(b, m, 1), 2) ./ sum(repmat(b, m, 1), 2);
    
    p = sum(1-max(x,y))/m;
    if(p==0)
        p=0.001;
    end
end
end