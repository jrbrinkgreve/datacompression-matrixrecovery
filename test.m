DFT_matrix = dftmtx(size1) / sqrt(size1);  %make it unitary OR use fft
inv_DFT_matrix = DFT_matrix';



X = fft2(H) / size1;
x = vec(x);



DFT_matrix = dftmtx(size1) / sqrt(size1);  %make it unitary OR use fft
inv_DFT_matrix = DFT_matrix';


norm(DFT_matrix  * H * DFT_matrix - fft2(H) / size1, "fro")


norm(inv_DFT_matrix  * H * inv_DFT_matrix - ifft2(H) * size1, "fro")





kronecker_inv_DFT = kron(DFT_matrix', DFT_matrix');




test1 = kronecker_inv_DFT * x;
test2 = ifft(x);


norm(test1 - test2)