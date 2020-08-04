function [Thetas] = NAG_LR(y, X, theta0, lambda, T, m, alpha, beta, b, stage_vec)

% [Thetas] = NAG_LR(y, X, theta0, lambda, T, m, alpha, beta, b, stage_vec)
% 
% This function implements the nesterov's accelerated gradient (NAG) algorithm
% and the multi-stage version of NAG (MASG) algorithm for empirical risk 
% minimization for the regularized logistic regression model.
% 
% y is the vector of binary responses, X is the matrix of features
% (explanatory variables), theta0 is the initial value of the parameter,
% lambda is the regularizing parameter, 
% T is the number of iterations, m is the subsample size, S1 is the L1
% sensitivity of the objective function, alpha is the vector of step-sizes, 
% beta is the vector of momentum parameters and stage_vec is the vector of stage
% numbers (needed for MASG)
%
% Thetas contain the iterates.
%
% Sinan Yildirim
% Last update: 03.08.2020

[n, d] = size(X);
Thetas = zeros(d, T);

if nargin < 10
    stage_vec = zeros(1, T);
    if nargin < 9
        b = zeros(1, T);
    end
end

% current and previous values
theta = theta0;
theta_prev = theta0;
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
    
    if t > 2 && stage_vec(t) - stage_vec(t-1) == 1
        theta_prev = theta;
    end
    
    % calculate the intermediate value
    z = (1 + beta(t))*theta - beta(t)*theta_prev;
    % keep the previous value
    theta_prev = theta;
    
    % noisy gradient
    grad_z = grad_LR(y_sub, X_sub, z, lambda);
    
    grad_z_noisy = grad_z + laprnd(d, 1, b(t));
    
    % update the parameter
    theta = z - alpha(t)*grad_z_noisy;
    
    % store the paremter update
    Thetas(:, t) = theta;
end