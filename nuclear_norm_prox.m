function [H_reconstructed, obj_vals, time] = nuclear_norm_prox(y, A, S)
tic
%solve min_x 0.5 * ||y - Ax||_2^2 + lambda * ||x||_1 using PGD

[M, len_y] = size(A);  %measurements
size1 = sqrt(len_y); %assume square matrix to recover

%these specific parameters for this approach
max_iter = 500;
lambda = 0.001;  
alpha = 1 / (norm(A, 2)^2);  %constant step size (lipschitz constant)
nu = 0.001;
%predefine often-used matrices
Ah = A'; 
Ahy = Ah * y;
eta_l = alpha * lambda;
eta_nu = alpha * nu;
Gs = S' * S;
g = diag(Gs);

%initialize
x = zeros(len_y, 1); 
obj_vals = zeros(max_iter, 1);


%main loop
for k = 1:max_iter
    %gradient step
    X = reshape(x, size1, size1);
    gamma = g .* vec(ifft2(X));   %perform elementwise multiplication with sampling mask 
    Gamma = reshape(gamma, size1, size1); %put into matrix for fft2
    grad = vec(fft2(Gamma)) - Ahy;     
    z = x - alpha * grad;
    
    %proximal step with l1 norm
    x = max(0, abs(z) - eta_l) .* sign(z);
    
    %objective function value
    %obj_vals(k) = 0.5 * norm(A * x - y, 2)^2 + lambda * norm(x, 1);


end


for k = 1:max_iter
    %gradient step
    X = reshape(x, size1, size1);
    gamma = g .* vec(ifft2(X));   %perform elementwise multiplication with sampling mask 
    Gamma = reshape(gamma, size1, size1); %put into matrix for fft2
    grad = vec(fft2(Gamma)) - Ahy;     
    z = x - alpha * grad;
    
    %proximal step with nuc norm
    x = singular_value_thresholding(z,eta_nu);
    
    %objective function value
    obj_vals(k,1) = norm(y - A*x);
    H_obj = ifft2(X) * size1;
    obj_vals(k,2) = trace(  real((H_obj' * H_obj)^0.5  ));
    obj_vals(k,3) = norm(x, 1);


end



function x_thresh = singular_value_thresholding(z, tau)
%singular Value Thresholding (SVT): applies soft-thresholding to singular values
    Z = reshape(z, size1, size1);
    H = ifft2(Z);
    [U, D, V] = svd(H);
    d = diag(D);
    d_hat = max(d - tau, 0);  % Soft thresholding on singular values
    x_thresh = vec(fft2(U * diag(d_hat) * V'));  % Reconstruct the matrix with thresholded singular values
end










%plot convergence
%{
figure;
plot(1:max_iter, obj_vals, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Objective Function');
title('Proximal Gradient Descent Convergence');
grid on;

%}

%convert back to H
H_reconstructed = ifft2(reshape(x, [size1, size1])) * size1; %ifft2 normalization, so * size1
time = toc;

end