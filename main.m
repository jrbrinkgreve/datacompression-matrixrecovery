%low rank matrix recovery
load("Sparse_Low_Rank_dataset.mat")

close all

M = 150;   %number of measurements
limit = 100; %number of matrices
H = H(:,:,1:limit); %to run only 'limit' number of matrices

size1 = size(H,1);
size2 = size(H,2); %is actually equal to size1
size3 = size(H,3); %size3 == limit



masks = mask_generator(H,M);  %we know the masks 
y = subsample(H, masks);   %this is the data we are allowed to work with
[A, S] = generate_A(H,masks);  %generates the measurement matrix from mask




%preallocate data tensors
H_reconstructed_prox_simple = zeros(size(H));
H_reconstructed_prox_efficient = zeros(size(H));
H_reconstructed_nuc = zeros(size(H));
H_reconstructed_psgd = zeros(size(H));
reconstruction_error_prox_simple = zeros(size3,1);
reconstruction_error_prox_efficient = zeros(size3,1);
reconstruction_error_psgd = zeros(size3,1);
reconstruction_error_nuc = zeros(size3,1);
runtime_psgd = zeros(size3,1);
runtime_prox_simple = zeros(size3,1);
runtime_prox_efficient = zeros(size3,1);
runtime_nuc = zeros(size3,1);


%tracker for solve loop
percent_step = 10;
next_update = percent_step;


%solve the thing!
for i = 1:size3


%note: obj_vals only from last calculation
[H_reconstructed_prox_simple(:,:,i), obj_vals_prox_simple, runtime_prox_simple(i)]       =           prox(y(:,i), A(:,:,i), S(:,:,i));
[H_reconstructed_prox_efficient(:,:,i), obj_vals_prox_efficient, runtime_prox_efficient(i)] = efficient_prox(y(:,i), A(:,:,i), S(:,:,i));
[H_reconstructed_psgd(:,:,i), obj_vals_PSGD, runtime_psgd(i)]                     =           PSGD(y(:,i), A(:,:,i));
%[H_reconstructed_nuc(:,:,i), obj_vals_nuc, runtime_nuc(i] = nuclear_norm_min(y(:,i), A(:,:,i), S(:,:,i));

progress = 100 * i / size3;
if progress >= next_update
    fprintf('Progress: %3.0f%% complete\n', next_update);
    next_update = next_update + percent_step;
end


end








for i = 1:size3

reconstruction_error_prox_simple(i) = norm(H_reconstructed_prox_simple(:,:,i) - H(:,:,i), 'fro');
reconstruction_error_prox_efficient(i) = norm(H_reconstructed_prox_efficient(:,:,i) - H(:,:,i), 'fro');
reconstruction_error_psgd(i) = norm(H_reconstructed_psgd(:,:,i) - H(:,:,i), 'fro');
reconstruction_error_nuc(i) =  norm(H_reconstructed_nuc(:,:,i)  - H(:,:,i), 'fro');



end


avg_error_prox_simple = mean(reconstruction_error_prox_simple);
avg_error_prox_efficient = mean(reconstruction_error_prox_efficient);
avg_error_psgd = mean(reconstruction_error_psgd);
avg_error_nuc = mean(reconstruction_error_nuc);

avg_runtime_prox_simple = mean(runtime_prox_simple);
avg_runtime_prox_efficient = mean(runtime_prox_efficient);
avg_runtime_psgd = mean(runtime_psgd);
avg_runtime_nuc = mean(runtime_nuc);

fprintf('Convergence Summary:\n');
fprintf('---------------------\n');
fprintf('Method                  | Avg. Error     | Avg. Runtime (s)\n');
fprintf('-----------------------------------------------------------\n');
fprintf('Prox (Simple)           | %.4e     | %.4f\n', avg_error_prox_simple, avg_runtime_prox_simple);
fprintf('Prox (Efficient)        | %.4e     | %.4f\n', avg_error_prox_efficient, avg_runtime_prox_efficient);
fprintf('Projected SGD           | %.4e     | %.4f\n', avg_error_psgd, avg_runtime_psgd);
fprintf('Nuclear Norm Minim.     | %.4e     | %.4f\n', avg_error_nuc, avg_runtime_nuc);
fprintf('-----------------------------------------------------------\n');





max_iter = max(size(obj_vals_prox_efficient));

%plot convergences
figure;
plot(1:max_iter, (obj_vals_PSGD), 'LineWidth', 2);
hold on
plot(1:max_iter, 10000*(obj_vals_prox_efficient), 'LineWidth', 2);
xlabel('Iteration'); ylabel('Objective Function');
title('10log of recovery objective functions');
grid on;
legend('PSGD', 'prox')



