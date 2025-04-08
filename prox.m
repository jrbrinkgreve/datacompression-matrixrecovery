function [H_reconstructed, obj_vals, time] = prox(y, A, S)
tic
%solve min_x 0.5 * ||y - Ax||_2^2 + lambda * ||x||_1 using PGD

[M, len_y] = size(A);  %measurements
size1 = sqrt(len_y); %assume square matrix to recover

%these specific parameters for this approach
max_iter = 500;
lambda = 0.001;  
alpha = 1 / (norm(A, 2)^2);  %constant step size (lipschitz constant)

%predefine often-used matrices and values 
Ah = A';
Ahy = Ah * y;
AhA = Ah * A;
eta = alpha * lambda;

%initialize
x = zeros(len_y, 1); 
obj_vals = zeros(max_iter, 1);


%main loop
for k = 1:max_iter
    %gradient step
    grad = AhA*x - Ahy;
    z = x - alpha * grad;
    
    %proximal step with l1 norm
    x = max(0, abs(z) - eta) .* sign(z);
    
    %objective function value
    obj_vals(k) = 0.5 * norm(A * x - y, 2)^2 + lambda * norm(x, 1);
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

