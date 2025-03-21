function H_reconstructed = cvx_implementation(Y, mask)
%Y is here a matrix, not the full tensor

size1 = size(Y,1);
size2 = size(Y,2);


%these define the transform
DFT = dftmtx(size1);
IDFT = DFT / size1;  

%the problem to solve is
%min norm(Y - mask .* IDFT * X *  IDFT  )


lambda = 1;
cvx_begin quiet
   
    variable X_hat(size1, size2); % Define the variable to reconstruct
    minimize(   norm(Y - mask .* IDFT * X_hat * IDFT, 'fro')  + lambda * sum(sum(abs(X_hat)))) % Minimize the L1 norm for sparsity
cvx_end



    H_reconstructed = DFT * X_hat * DFT;


    % somehow the returned H is all zeros... idk??? probably redo the
    % derivation......





end