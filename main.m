%low rank matrix recovery
load("Sparse_Low_Rank_dataset.mat")

M = 100;   %number of measurements
limit = 8;

H = H(:,:,1:limit);

size1 = size(H,1);
size2 = size(H,2);
size3 = size(H,3);


%these define the transform
DFT_matrix = dftmtx(size1);
inv_DFT_matrix = DFT_matrix' / size1;  

masks = zeros(size(H));
H_reconstructed = zeros(size(H));
error = zeros(size(H));

for i = 1:size3
    masks(:,:,i) = mask_generator(H(:,:,i), M);
end

Y = H .* masks;   % 'measured' matrices



parfor i = 1:size3
H_reconstructed(:,:,i) = cvx_implementation(Y(:,:,i), masks(:,:,i));
i
end


error = abs(H_reconstructed - H);



for i = 1:size3
error_norm(i) = norm(error(:,:,i), 'fro');


end

avg_error_norm = sum(error_norm / size3)

