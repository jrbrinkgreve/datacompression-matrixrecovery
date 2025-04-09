function [A, S] = generate_A(H, masks)

size1 = size(H,1);
size2 = size(H,2); %is actually equal to size1
size3 = size(H,3); %size3 == limit

[M, ~] = size(masks);
%these define the transform
DFT_matrix = dftmtx(size1) / sqrt(size1);  %make it unitary OR use fft
inv_DFT_matrix = DFT_matrix';




%S generation
S = zeros(M, size1*size2, size3);
mask_index = 0;

for mask = masks
    mask_index = mask_index + 1;
    for i = 1:length(mask)
        S(i, mask(i), mask_index) = 1;
    end
end


DFT_kronecker = kron(DFT_matrix',DFT_matrix');


A = pagemtimes(S, DFT_kronecker);

end




%A = S * kron(U', U)