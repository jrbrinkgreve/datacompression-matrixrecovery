function [H_reconstructed, obj_vals, time] = PSGD(y, A)
tic
[M, len_y] = size(A);  %sizes
size1 = sqrt(len_y);
max_iter = 500;
alpha = 0.001;% need a small step size here!
eps_grad = 1e-3;


%initialize
x = zeros(len_y, 1);
obj_vals = zeros(max_iter,1);
tol = 1e-5;

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
    obj_vals(k,1) = norm(x, 1);
    %obj_vals(k,2) = norm(x, 2);
    
    % check convergence if we have 5 values
    if k >= 10
        avg_prev = mean(obj_vals(k-9:k-1));  
        avg_curr = mean(obj_vals(k-9:k));    
        if abs(avg_curr - avg_prev) < tol
            break;
        end
    end
  

end


H_reconstructed = ifft2(reshape(x, [size1, size1])) * size1; %ifft2 normalization, so * size1
time = toc;
end






