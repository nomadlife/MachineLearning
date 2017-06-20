clear ; close all; clc
input_layer_size  = 400; 
num_labels = 10;

load('ex3data1.mat');
m = size(X, 1);

rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

%displayData(sel);

%% ============ Part 2a: Vectorize Logistic Regression ============

% Test case for lrCostFunction
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ============ Part 2b: One-vs-All Training ============

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll_t(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

