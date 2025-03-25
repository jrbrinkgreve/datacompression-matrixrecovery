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
H_reconstructed = zeros(size(H));

for i = 1:size3
    masks(:,:,i) = mask_generator(H(:,:,i), M);
end

Y = masks .* H;   % 'measured' matrices, this is what we have access to



for i = 1:size3   %do the reconstruction 
H_reconstructed(:,:,i) = cvx_implementation(Y(:,:,i), masks(:,:,i));
i
end


error = abs(H_reconstructed - H);



for i = 1:size3
error_norm(i) = norm(error(:,:,i), 'fro');

end

avg_error_norm = sum(error_norm / size3)

