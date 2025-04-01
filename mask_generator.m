function masks = mask_generator(H, M)


size1 = size(H,1);
size2 = size(H,2); %is actually equal to size1
size3 = size(H,3);


masks = zeros(M, size3);   %preallocate
for i = 1:size3
     masks(:,i) = randsample(  size1  * size2,  M);  %select randomly indices to sample
end

end
