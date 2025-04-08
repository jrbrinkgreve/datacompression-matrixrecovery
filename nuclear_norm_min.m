function [H_reconstructed, obj_vals, time] = nuclear_norm_min(y, A, S)


tic


[M, len_y] = size(A);  % Measurements
size1 = sqrt(len_y); %assume square matrix to recover
max_iter = 500;
lambda = 0.001;  % Regularization parameter

Gs = S'*S;
Ah = A';
AAh = A*Ah;
max_cycles = 3;

%the algorithm

%init
x = zeros(size1*size1, 1);
%X = zeros(size1);
obj_vals = zeros(max_iter, 2);



for cycles = 1:max_cycles

alpha = 1 / (norm(A, 2)^2);  %step size (lipschitz constant)


%main loop1
for k = 1:max_iter
    %gradient step
    %grad = Ah * (A * x - y);
    X = reshape(x, size1, size1);
    gamma = diag(Gs) .* vec(ifft2(X));
    Gamma = reshape(gamma, size1, size1);
    grad = vec(fft2(Gamma)) - Ah * y;
    z = x - alpha * grad;
    
    %proximal step with l1 norm
    x = max(0, abs(z) - alpha * lambda) .* sign(z);
    
    %objective function value
    %obj_vals(k) = 0.5 * norm(A * x - y, 2)^2 + lambda * norm(x, 1);
end





%now, the prox has finished:
alpha = alpha / 1000;

%try PSGD with nuclear norm minimization:

%main loop2
for k = 1:max_iter
    %gradient step
    X = reshape(x, size1, size1);
    [U, D, V] = svd(X);
    gradX = U*V';
    grad = gradX(:) + 0.5*sign(x);
    z = x - alpha * grad;
    if norm(y-A*z) > eps
        x = z - Ah * ((AAh) \ (A * z - y));  %projection onto affine space
    else
        x = z;
    end
    obj_vals(k) = trace(abs((X'*X)^0.5));
    obj_vals(k,2) = norm(x,1);

end


end
%}

figure;
plot(1:max_iter, obj_vals, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Objective Function');
title('Nuclear norm min convergence');
grid on;
legend('Nuclear norm of X', 'l1 norm of x')
set(gca,'fontsize', 14) 




%convert back to H
H_reconstructed = ifft2(X) * size1; %ifft2 normalization, so * size1
time = toc;
end






%{

for k = 1:max_iter
        % Forward model: compute residual
        Z = ifft2(X);                % Inverse FFT
        r = S * Z(:) - y;            % Residual in measurement space

        % Gradient of smooth term
        grad = fft2(reshape(S' * r, size(X)));  % Backpropagate and take FFT

        % Gradient step + proximal operator (soft thresholding)
        X = singular_value_thresholding(X - alpha * grad, alpha * lambda);
        x = reshape(X, size1*size1, 1);
        obj_vals(k) = norm(x, 1);%0.5 * norm(A * x - y, 2)^2 + lambda * norm(x, 1);
        
end

function X_thresh = singular_value_thresholding(X, tau)
% Singular Value Thresholding (SVT): applies soft-thresholding to singular values
    [U, D, V] = svd(X);
    d = diag(D);
    d_hat = max(d - tau, 0);  % Soft thresholding on singular values
    X_thresh = U * diag(d_hat) * V';  % Reconstruct the matrix with thresholded singular values
end

%}

%{
for k = 1:max_iter
    %gradient step
    
    gamma = diag(Gs) .* vec(ifft2(X));
    Gamma = reshape(gamma, size1, size1);
    grad = fft2(Gamma) - reshape(Ah * y, size1, size1);
    Z = reshape(X - alpha * grad, size1, size1);



    [U, D, V] = svd(Z);
    d = diag(D);
    d_hat = max(0, abs(d) - alpha * lambda) .* sign(d);
    D_hat = diag(d_hat);

    X = U*D_hat*V';

    obj_vals(k) = norm(X(:), 1);%real(trace( (X'*X)^0.5    ));


end

%}

%try singular value thresholding

%{
%main loop
for k = 1:max_iter
    %gradient step
    %grad = Ah * (A * x - y);
    X = reshape(x, size1, size1);
    gamma = diag(Gs) .* vec(ifft2(X));
    Gamma = reshape(gamma, size1, size1);
    grad = vec(fft2(Gamma)) - Ah * y;
    
    z = x - alpha * grad;
    
    %proximal step with nuclear norm
    Z = reshape(z, size1, size1);

    [U, D, V] = svd(Z);
    d = diag(D);
    d_hat = max(0, abs(d) - alpha * lambda) .* sign(d);
    D_hat = diag(d_hat);

    X = U*D_hat*V';
    x = X(:);
    
    %objective function value
    obj_vals(k) = 0.5 * norm(A * x - y, 2)^2 + lambda * norm(x, 1);
end


%}


