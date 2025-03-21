function mask = mask_generator(H, M)


r = randsample(  (size(H,1)  * size(H,2)),  M);  %select randomly indices to sample
mask = zeros( size(H,1)  , size(H,2)  );  %no repeated indices
mask(r) = 1;

end
