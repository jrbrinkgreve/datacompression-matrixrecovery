function [H_reconstructed, obj_vals, time] = nuclear_norm_PSGD(y, A, S)
tic
[M, len_y] = size(A);  %sizes
size1 = sqrt(len_y);
max_iter = 500;
alpha = 0.001;% need a small step size here!
eps_grad = 1e-6;


%initialize
x = zeros(len_y, 1);
obj_vals = zeros(max_iter,1);

%preallocate 
Ah = A';
Ah_AAh_pinv_y = Ah * pinv(A*Ah)*y;
I_Ah_AAh_pinv_A = eye(len_y) - Ah * pinv(A*Ah) * A;


%main loop
for k = 1:max_iter
    % Gradient step
    grad = sign(x);
    grad(abs(x) < eps_grad) = 0;
    z = x - alpha * grad;

    %projection onto affine space
    %x = z - Ah * (AAh_pinv * (A * z - y)); 
    x = I_Ah_AAh_pinv_A*z + Ah_AAh_pinv_y;
   
    
    %objective value (for monitoring)
    %obj_vals(k,1) = norm(x, 1);
    %obj_vals(k,2) = norm(x, 2);

end



for k = 1:max_iter
    X = reshape(x, size1,size1);
    [U, D, V] = svd(X);
    grad = vec(U*V');
    z = x - alpha * grad;

    %projection onto affine space
    %x = z - Ah * (AAh_pinv * (A * z - y)); 
    x = I_Ah_AAh_pinv_A*z + Ah_AAh_pinv_y;
       
    obj_vals(k,1) = norm(y - A*x);
    obj_vals(k,2) = trace(  real((X' * X)^0.5  ));

end





H_reconstructed = ifft2(reshape(x, [size1, size1])) * size1; %ifft2 normalization, so * size1
time = toc;
end






