function [E, V, rho, m] = error_HB_quad(alpha, beta, T, mu, L, P, C_n, C_const, E_0)

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