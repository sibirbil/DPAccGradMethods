% This is the main code that generates the Figure 2 in Section 4 of the 
% paper entitled
%
% "Differentially Private Accelerated Optimization Algorithms"
% 
% authored by
%
% Nurdan Kuru, Ilker Birbil, Mert Gurbuzbalaban, and Sinan Yildirim
% 
% For questions/corrections, please email sinanyildirim@sabanciuniv.edu
% 
% Last update: 3.08.2020

clear all; clc; % close all; fc = 0;
fc = 0;
%% 
kappa = 20;
L = 20;
mu = 1;
p = 1;
n_1 = floor(2*sqrt(kappa)*log(sqrt(kappa)));
a = 1.00;
% c1 = 0.01;
c = 1;
eps_DP = 1;
data_size = 10^5;
S1 = 10/data_size;

% c2 = 1;

%%
K_vec = [2 3 4 5]; L_K = length(K_vec);

T = zeros(1, L_K);

C = ceil(sqrt(kappa)*log(2^(p+2)));
sigma_opt_vec = cell(1, L_K);
log_sigma_opt_vec = cell(1, L_K);
log_opt_error = zeros(1, L_K);

for j = 1:L_K
    K = K_vec(j);
    T_0 = C*(2^(K+1) - 4);
    T(j) = T_0 + n_1;
    n = T(j);
    
    %%% error analysis, optimum sigma using Theorem 2.3 %%%
    alpha_vec = zeros(1, n);
    a_vec = zeros(1, n);
    factor_of_two = zeros(1, n);
    alpha_vec(1:n_1) = c/L;
    
    t_end = n_1;
    for k = 2:K
        factor_of_two(t_end) = log(2);
        t_begin = t_end + 1;
        t_end = t_end + 2^k*C;
        alpha_vec(t_begin:t_end) = c*2^(-2*k)/L;
    end
    temp_vec1 = 1 - sqrt(alpha_vec*mu);
    temp_vec2 = (alpha_vec/2).*(1 + alpha_vec*L);
    for i = 1:n
        a_vec(i) = sum(log(temp_vec1(i+1:n))) + log(temp_vec2(i))...
            + sum(factor_of_two(i+1:n));
    end
    % calculate the optimum b_t's
    log_sum_a_by_3 = log_sum_exp(a_vec/3);
    log_sigma_opt_vec{j} = log_sum_a_by_3 - a_vec/3; % - log(eps_DP/S1);
    sigma_opt_vec{j} = exp(log_sigma_opt_vec{j});
end

%% Plot results:
fc = fc + 1; figure(fc);
% subplot(2, 1, 1);

J = ceil(L_K/4);
for j = 1:L_K
    subplot(J, 4, j);
    plot(1:T(j), sigma_opt_vec{j});
    if ceil(j/4) == J
        xlabel('$t$', 'Interpreter', 'latex');
    end
    if mod(j, 4) == 1
        ylabel('$b_{t}$', 'Interpreter', 'latex');
    end    
    title(sprintf('$K$ = %d, $T$ = %d', K_vec(j), T(j)), 'Interpreter', 'latex');
end
sgtitle('optimum $b_{t}$ ($\times n \epsilon / S_{1}$)', 'Interpreter', 'latex');