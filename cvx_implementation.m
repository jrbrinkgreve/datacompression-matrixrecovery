function H_reconstructed = cvx_implementation(Y, mask)
%Y is here a matrix, not the full tensor

size1 = size(Y,1);
size2 = size(Y,2);


%these define the transform
DFT = dftmtx(size1) / sqrt(size1);
IDFT = DFT';



%the problem to solve is
% min norm(Y - mask .* IDFT * X *  IDFT, 'fro')
% s.t. sparse X


lambda = 0.1;
cvx_begin 
    variable X_hat(size1, size2) complex
    variable W1(size1, size2) hermitian
    variable W2(size1, size2) hermitian
    minimize( lambda * (trace(W1) + trace(W2))  + norm(Y - mask.* IDFT * X_hat * IDFT , 'fro')        )
subject to 

[W1, X_hat; X_hat', W2] == hermitian_semidefinite(2*size1);
   
cvx_end

H_reconstructed = IDFT * X_hat * IDFT;


    % somehow the returned H is all zeros... idk??? probably redo the
    % derivation......





end