function H_reconstructed = cvx_implementation(Y, masks)
%Y is here a matrix, not the full tensor

size1 = size(Y,1);
size2 = size(Y,2);

M = sum(masks, 'all'); %num measurements
%these define the transform
DFT = dftmtx(size1) / sqrt(size1);
IDFT = DFT';



%the problem to solve is
% min l2-norm(Y - Ax)
% s.t. sparse X

% extract nonzero elements and store as a vector
y = Y(Y ~= 0);  

% vectorze mask matrix
mask_vec = reshape(masks,1, [])'; %reshape into a col vector
subsample_idx = find(mask_vec == 1);
S = zeros(M, size1*size2);

for i=1:M
    col = subsample_idx(i);
    S(i,col) = 1;
end



U_2d = kron(DFT', DFT);

A = S *U_2d; %weight matrix


cvx_begin 
    variable x_hat(size1 *size2, 1) complex
    minimize( norm(y - A*x_hat, 2 )    )

cvx_end

X_hat = reshape(x_hat, size1, size2);
H_reconstructed = IDFT * X_hat * IDFT;







