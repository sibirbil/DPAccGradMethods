function F = eval_F_LR(y, X, theta, lambda)

% X is n x d
% y is n x 1
% theta is d x 1

n = length(y);
F = sum(log(1+exp(-y.*(X*theta))))/n + lambda*norm(theta)^2;