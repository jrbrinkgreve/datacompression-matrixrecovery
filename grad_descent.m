%low rank matrix recovery
load("Sparse_Low_Rank_dataset.mat")

M = 100;   %number of measurements
limit = 1;

H = H(:,:,1:limit); %to run only 'limit' number of matrices

size1 = size(H,1);
size2 = size(H,2); %is actually equal to size1
size3 = size(H,3);


%these define the transform
DFT_matrix = dftmtx(size1) / sqrt(size1);  %make it unitary
inv_DFT_matrix = DFT_matrix';

masks = zeros(size(H));   %preallocate
%H_reconstructed = zeros(size(H));

for i = 1:size3
    masks(:,:,i) = mask_generator(H(:,:,i), M);
end

Y = masks .* H;   % 'measured' matrices, this is what we have access to

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



U_2d = kron(DFT_matrix', DFT_matrix);

% masks_mtrx = mask_vec' * mask_vec;
masks_mtrx = S' * S;
inv_DFT_masks = U_2d' * masks_mtrx;


A = S *U_2d; %weight matrix


max_iters = 1000;  % maximum iterations
tol = 1e-6;        % convergence tolerance
alpha = 0.001;

% initialize vectorized x with zeros
x = zeros(size1*size2, 1);



% gradient descent
for iter = 1:max_iters
    grad = 2 * A' * (S * fft2(x) -  y);  % compute gradient
        
    x_new = x - alpha * grad;     % gradient descent update
    x_new = x_new / norm(x_new, 2); %normalize

    % check for convergence
    if norm(x_new - x, 2) < tol
        break;
    end
    x = x_new;  % update x for next iteration
end

disp('Gradient descent completed.');

X = reshape(x, size1, size2); % reshape back to matrix form
H_reconstructed = DFT_matrix' * X * DFT_matrix';