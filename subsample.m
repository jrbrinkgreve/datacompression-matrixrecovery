function H_subsampled = subsample(H, M)


r = randsample(  (size(H,1)  * size(H,2)),  M);  %select randomly indices to sample
mask = zeros( size(H,1)  , size(H,2)  );
mask(r) = 1;

H_subsampled = H .* mask;    %elementwise product with mask


end
