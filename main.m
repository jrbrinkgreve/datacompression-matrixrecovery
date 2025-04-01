%low rank matrix recovery
load("Sparse_Low_Rank_dataset.mat")

M = 100;   %number of measurements
limit = 50; %number of matrices
H = H(:,:,1:limit); %to run only 'limit' number of matrices

size1 = size(H,1);
size2 = size(H,2); %is actually equal to size1
size3 = size(H,3); %size3 == limit



masks = mask_generator(H,M);  %we know the masks 
y = subsample(H, masks);   %this is the data we are allowed to work with
A = generate_A(H,masks);  %generates the measurement matrix from mask




for i = 1:size3
%solve the thing




end









