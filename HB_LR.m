function [Thetas] = HB_LR(y, X, theta0, lambda, T, m, alpha, beta, b)

% [Thetas] = HB_LR(y, X, theta0, lambda, T, m, alpha, beta, b)
% 
% This function implements the heavy-ball algorithm for empirical 
% risk minimization for the regularized logistic regression model.
% 
% y is the vector of binary responses, X is the matrix of features
% (explanatory variables), theta0 is the initial value of the parameter,
% lambda is the regularizing parameter, T is the number of iterations, 
% m is the subsample size, S1 is the L1-sensitivity of the objective function,
% alpha is the step-size parameter, and finally beta is the momentum parameter,
% and b is the parameter of the Laplace noise on the gradients.
% 
% Thetas are the iterates.
% 
% Sinan Yildirim
% Last update: 03.08.2020

[n, d] = size(X);
Thetas = zeros(d, T);

% current and previous values
theta = theta0;
theta_prev = theta0;
theta_diff = theta - theta_prev;
    
for t = 1:T    
    % subsample data
    if m < n
        idx = randsample(n, m);
        y_sub = y(idx);
        X_sub = X(idx, :);
    else
        y_sub = y;
        X_sub = X;
    end
    
    % noisy gradient
    grad_z = grad_LR(y_sub, X_sub, theta, lambda);
    grad_z_noisy = grad_z + laprnd(d, 1, b);
    
    % update the parameter
    theta_prev = theta;
    theta = theta - alpha*grad_z_noisy + beta*theta_diff;
    theta_diff = theta - theta_prev;
    
    % store the paremter update
    Thetas(:, t) = theta;
end