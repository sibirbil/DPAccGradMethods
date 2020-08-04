function [eps_DP_vec, T_final] = opt_noise_dist_NAG(eps_DP, alpha_vec, ...
    stage_vec, mu, L, alter_T, S1, n, d, E_0)

% [eps_DP_vec, T_final] = [eps_DP_vec, T_final] = opt_noise_dist_NAG(eps_DP, alpha_vec, ...
%    stage_vec, mu, L, alter_T, S1, n, d, E_0)
%
% eps_DP is the privacy level, alpha_vec is the vector of stepsizes,
% stage_vec is the vector of stage numbers for iterations, mu and L are the
% convexity and smoothness parameters of the regularized logistic regression 
% function, alter_T is a binary variable to determine whether the total
% number of iterations is to be adjusted, S1 is the L1 sensitivity, n is
% the number of rows, d is the dimension, and E_0 is the guess for the
% initial error
% 
% As outputs, eps_DP_vec is the vector of privacy levels vs iterations, and
% T_final is the re-adjusted number of iterations.
% 
% Sinan Yildirim
% Last update: 03.08.2020


T = length(alpha_vec);
temp_vec1 = 1 - sqrt(alpha_vec*mu);
temp_vec2 = (alpha_vec/2).*(1 + alpha_vec*L);
    
if alter_T == 0
    T_final = T;
else % alter T
    score = zeros(1, T);
    for tau = 1:T
        a_vec = zeros(1, tau);
        for t = 1:tau
            a_vec(t) = sum(log(temp_vec1(t+1:tau))) + log(temp_vec2(t))...
                + (stage_vec(tau) - stage_vec(t))*log(2);
        end
        log_sum_a_by_3 = log_sum_exp(a_vec/3);
        a_0 = stage_vec(tau)*log(2) + sum(log(temp_vec1(1:tau)));
        score(tau) = log_sum_exp([3*log_sum_a_by_3 + 2*log(S1/(n*eps_DP)) + log(d)...
            log(E_0) + a_0]);
    end
    [~, T_final] = min(score);
end

% with T determined, calculate the noise levels
a_vec = zeros(1, T_final);
for t = 1:T_final
    a_vec(t) = sum(log(temp_vec1(t+1:T_final))) + log(temp_vec2(t))...
        + (stage_vec(T_final) - stage_vec(t))*log(2);
end
log_sum_a_by_3 = log_sum_exp(a_vec/3);
eps_DP_vec = eps_DP*exp(a_vec/3 - log_sum_a_by_3);