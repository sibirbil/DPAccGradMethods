function [eps_DP_vec, T_final] = opt_noise_dist_HB(eps_DP, alpha_vec, stage_vec, mu, L, eps0_min)

% [eps_DP_vec, T_final] = optimize_noise_dist_NAG(alpha_vec, stage_vec, opt_params)
%
% Sinan Yildirim,
% Last update: 9 July 2020

T = length(alpha_vec);
score = zeros(1, T);
temp_vec1 = 1 - sqrt(alpha_vec*mu);
temp_vec2 = (alpha_vec/2).*(1 + alpha_vec*L);
for T_final_cand = 1:T
    a_vec = zeros(1, T_final_cand);
    for t = 1:T_final_cand
        a_vec(t) = sum(log(temp_vec1(t+1:T_final_cand))) + log(temp_vec2(t))...
            + (stage_vec(T_final_cand) - stage_vec(t))*log(2);
    end
    log_sum_a_by_3 = log_sum_exp(a_vec/3);
    a_0 = stage_vec(T_final_cand)*log(2) + sum(log(temp_vec1(1:T_final_cand)));
    score(T_final_cand) = log_sum_exp([3*log_sum_a_by_3 + 2*log(20/100000)  log(10) + a_0]);
end
[~, T_final] = min(score);


cond = 0;
while cond == 0
    a_vec = zeros(1, T_final);
    for t = 1:T_final
        a_vec(t) = sum(log(temp_vec1(t+1:T_final))) + log(temp_vec2(t))...
            + (stage_vec(T_final) - stage_vec(t))*log(2);
    end
    log_sum_a_by_3 = log_sum_exp(a_vec/3);
    eps_DP_vec = eps_DP*exp(a_vec/3 - log_sum_a_by_3);
    if min(eps_DP_vec) > eps0_min
        cond = 1;
    else
        T_final = T_final - 1;
    end
end

% a_vec = zeros(1, T_final);
% for t = 1:T_final
%     a_vec(t) = sum(log(temp_vec1(t+1:T_final))) + log(temp_vec2(t))...
%         + (stage_vec(T_final) - stage_vec(t))*log(2);
% end
% log_sum_a_by_3 = log_sum_exp(a_vec/3);
% eps_DP_vec = eps_DP*exp(a_vec/3 - log_sum_a_by_3);
    