function [rho] = rho_HB_quad(mu, L, alpha, beta)

% This function calculate thex rate of the heavy ball algorithm which uses
% alpha as its stepsize and beta as its momentum parameter.
% 
% The rate is calculated for a strongly convex and smooth function with
% convexity parameter mu and smoothness parameter L.
% 
% beta is allowed to be a vector.
% 
% Sinan Yildirim
% Last update: 8 July 2020

determ_mu = sqrt((1 + beta).^2*(1 - alpha*mu)^2 - 4*beta*(1 - alpha*mu));
determ_L = sqrt((1 + beta).^2*(1 - alpha*L)^2 - 4*beta*(1 - alpha*L));

a_mu_minus  = ((1 + beta)*(1 - mu*alpha) - determ_mu)/2;
a_mu_plus = ((1 + beta)*(1 - mu*alpha) + determ_mu)/2;
a_L_minus  = ((1 + beta)*(1 - L*alpha) - determ_L)/2;
a_L_plus = ((1 + beta)*(1 - L*alpha) + determ_L)/2;

rho = max([abs(a_mu_minus), abs(a_mu_plus), abs(a_L_minus), abs(a_L_plus)]);