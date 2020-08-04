function grad_theta = grad_LR(y, X, theta, lambda)

% y is n x 1,
% X is n x d
% theta is d x 1
n = length(y);

u = X*theta; % n x 1
temp_exp = exp(-y.*u); % n x 1 
grad_theta = -X'*(y.*temp_exp./(1+temp_exp))/n + 2*lambda*theta;