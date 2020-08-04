function [outputs] = DP_NAG_LR(y, X, theta0, lambda, eps_DP, T, m, S1, c_alpha, mu, L, ...
    alg_type, opt_on_off, alter_T, E0)

% [outputs] = DP_NAG_LR(y, X, theta0, lambda, eps_DP, T, m, S1, c_alpha, mu, ...
%   L, alg_type, opt_on_off, alter_T, eps0_min_c, E0)
% 
% This function implements the differentialy private nesterov's accelerated 
% gradient (DP-NAG) algorithm and differentially private multi-stage version of 
% NAG (DP-MASG) algorithm for empirical risk minimization for the regularized 
% logistic regression model.
% 
% y is the vector of binary responses, X is the matrix of features
% (explanatory variables), theta0 is the initial value of the parameter,
% lambda is the regularizing parameter, eps_DP is the privacy level, 
% T is the number of iterations, m is the subsample size, S1 is the L1
% sensitivity of the objective function, c_alpha is the constant in front of 1/L
% where c_alpha/L is used as step-size parameter, mu and L are the stong convexity
% and smoothness parameters of the function, alg_type is a ternary variable 
% determining the algorithm type:
% - 0 for the regular Stochastic gradient descent
% - 1 for NAG
% - 2 for Multi-stage version of NAG
% 
% opt_on_off is a binary varibale determening whether the DP noise variance 
% distribution as well as the number of iterations is to be adjusted (1) or not (0),
% 
% alter_T is a binary parameter determining whether the number of
% iterations is to be re-adjusted (1) or not (0),
% 
% E_0 is an initial guess on the initial error.
% 
% The struct outputs has
% - Thetas, which contain the iterates,
% - eps_DP_vec, privacy loss vs iteration
% - T_final, the re-adjusted number of iterations
% - b_vec, the noise level vs iteration.
%
% Sinan Yildirim
% Last update: 03.08.2020

% get the sizes    
[n, d] = size(X);

if alg_type == 0 % This is regular SGD
    alpha_NAG = (c_alpha/L)*ones(1, T);
    beta_NAG = zeros(1, T);
    stages = ones(1, T);
elseif alg_type == 1 % This is NAG
    alpha_NAG = (c_alpha/L)*ones(1, T);
    stages = ones(1, T);
    % arrange beta vector according to alpha_vec
    beta_NAG = (1-sqrt(mu*alpha_NAG))./(1+sqrt(mu*alpha_NAG));
elseif alg_type == 2 % This is MASG
    % arrange the stages, their lengths and the stepsizes governing along
    % them
    kappa = L/mu;
    k = 1;
    T_new = min(T, ceil(2*sqrt(kappa)*log(sqrt(kappa))));
    stages = zeros(1, T);
    C = ceil(sqrt(kappa)*log(8));
    stages(1:T_new) = 1;
    alpha_NAG(1:T_new) = c_alpha/L;
    T_prev = T_new;
    while T_prev < T
        k = k + 1;
        stage_length = 2.^k*C;
        T_new = min(T, T_prev + stage_length);
        stages(T_prev+1:T_new) = k;
        alpha_NAG(T_prev+1:T_new) = 2^(-2*k)*c_alpha/L;
        T_prev = T_new;
    end
    % arrange beta vector according to alpha_vec
    beta_NAG = (1-sqrt(mu*alpha_NAG))./(1+sqrt(mu*alpha_NAG));
end

% Distribute the privacy budget to iterations (valid for NAG and MASG)
if opt_on_off == 1 && (alg_type == 1 || alg_type == 2)
    % optimize the distribution of privacy budget to the iterations and
    % alter T (if alter_T = 1)
    [eps_DP_vec, T_final] = opt_noise_dist_NAG(eps_DP, alpha_NAG, stages, mu,...
        L, alter_T, S1, n, d, E0);
else
    T_final = T;
    eps_DP_vec = ones(1, T)*eps_DP/T;
end

% Finally, calculate the noise vector
b_vec = (S1/m)./log((exp(eps_DP_vec) - 1)*n/m + 1);

% Run the NAG algorithm with the specified parameters
Thetas = NAG_LR(y, X, theta0, lambda, T_final, m, alpha_NAG, beta_NAG, b_vec, stages);

% outputs
outputs.Thetas = Thetas;
outputs.eps_DP_vec = eps_DP_vec;
outputs.T_final = T_final;
outputs.b_vec = b_vec;
