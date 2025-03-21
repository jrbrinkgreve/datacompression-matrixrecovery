%low rank matrix recovery
load("Sparse_Low_Rank_dataset.mat")

size1 = size(H,1);
size2 = size(H,2);
size3 = size(H,3);

H_fft = fft2(H);

M = 50;


%define random subsampling operator as function
H_subsampled = zeros(size(H));
for i = 1:size3
    H_subsampled(:,:,i) = subsample(H(:,:,i),M);

end


%recovery algorithm 1




%recovery algorithm 2











