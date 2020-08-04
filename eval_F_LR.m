function F = eval_F_LR(y, X, theta, lambda)

% F = eval_F_LR(y, X, theta, lambda)
% 
% This function evaluates the objective function for the regularized
% logistic regression model.
%
% X is the n x d matrix of explanatory variables
% y is the n x 1 response vector
% theta is the d x 1 parameter of the model
% lambda is the regularizing parameter
%
% Sinan Yildirim
% Last update: 03.08.2020

n = length(y);
F = sum(log(1+exp(-y.*(X*theta))))/n + lambda*norm(theta)^2;