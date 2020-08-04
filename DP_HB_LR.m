function [Thetas] = DP_HB_LR(y, X, theta0, lambda, eps_DP, T, m, S1, alpha, beta)

% [Thetas] = DP_HB_LR(y, X, theta0, lambda, eps_DP, T, m, S1, alpha, beta)
% 
% This function implements the differentialy private heavy-ball (DP-HB) 
% algorithm for empirical risk minimization for the regularized 
% logistic regression model.
% 
% y is the vector of binary responses, X is the matrix of features
% (explanatory variables), theta0 is the initial value of the parameter,
% lambda is the regularizing parameter, eps_DP is the privacy level, 
% T is the number of iterations, m is the subsample size, S1 is the L1
% sensitivity of the objective function, alpha is the step-size parameter,
% and finally beta is the momentum parameter.
% 
% Thetas are the iterates.
% 
% Sinan Yildirim
% Last update: 03.08.2020

% get the data size
n = size(X, 1);
% Determine the parameter of the laplace noise to be added to the gradients for 
% DP

b = S1/(m*log(1 + (exp(eps_DP/T) - 1)*n/m));

[Thetas] = HB_LR(y, X, theta0, lambda, T, m, alpha, beta, b);