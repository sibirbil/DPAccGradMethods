function [E, V, rho, m] = error_HB_quad(alpha, beta, T, mu, L, P, C_n, C_const, E_0)

% [E, V, rho, m] = error_HB_quad(alpha, beta, T, mu, L, P, C_n, C_const, E_0)
% 
% This function evaluates the error bound for the HB algorithm when the
% objective function is quadratic in the form of.
% theta^T P theta ...
%
% alpha and beta are the stepsize and momentum parameters, 
% T is the number of iterations
% mu and L are the convexity and smoothness parameters
% C_n is a noise level parameter
% C_const is the constant in front of T which makes the linear term (in T) in 
% the error bound 
% E_0 is the initial error.
% 
% Sinan Yildirim
% Last update: 03.08.2020


% Calculate the convergence rate
[rho] = rho_HB_quad(mu, L, alpha, beta);

% eigenvalues of P
eig_vec = eig(P);

% addive error
c_w = (C_n*T).^2;
m = (c_w/2)*2*alpha*(1 + beta)/(1 - beta)...
    *sum(1./(eig_vec.*(2 + 2*beta - alpha*eig_vec)));

% Lyupanov
V = E_0 + c_w.*alpha^2/(1 - rho^2);

% error vector
E = V.*(C_const*T).^2.*rho.^(2*T) + L*m;