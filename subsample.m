function y = subsample(H, masks)



size1 = size(H,1);
size2 = size(H,2); %is actually equal to size1
size3 = size(H,3);

[M, ~] = size(masks);


H_vectorized = reshape(H, [size1 * size2, size3]);  %put the 

y = zeros(M, size3);
for i = 1:size3
    y(:,i) = H_vectorized(masks(:,i), i);   %these are our measurements, acquired at indices specified by masks
end
end



