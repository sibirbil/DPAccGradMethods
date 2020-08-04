% This is the main code that generates Figure 1 in Section 3.2 of the paper 
% entitled
% "Differentially Private Accelerated Optimization Algorithms"
% 
% authored by
%
% Nurdan Kuru, Ilker Birbil, Mert Gurbuzbalaban, and Sinan Yildirim
% 
% For questions/corrections, please email sinanyildirim@sabanciuniv.edu
% 
% Last update: 3.08.2020

clear; clc; close all; fc = 0;

alpha_vec = [0.1 0.5 1]/L; L_a = length(alpha_vec);
C_n = 1/100; % Make this 1/10000 to generate the second row of Figure 1.

mu = 0.1; L = 1;
P = diag([mu, L]);
alpha_opt = 4/(sqrt(mu) + sqrt(L))^2;
legends_a = cell(1, L_a);
for i = 1:L_a
    legends_a{i} = sprintf('$\\alpha$ = %.2f', alpha_vec(i));
end

beta_opt_vec = (1-sqrt(alpha_vec*mu))./(1+sqrt(alpha_vec*mu));
beta_vec = 0:0.01:0.99; L_b = length(beta_vec);
beta_vec2 = [0 0.3 0.7]; L_b2 = length(beta_vec2);
legends_b = cell(1, L_b2);
for i = 1:L_b2
    legends_b{i} = sprintf('$\\beta$ = %.2f', beta_vec2(i));
end
legends_b{L_b2+1} = '$\beta_{opt}$';
C_const = 1;
E_0 = 10;
T_vec = 1:250;

E = cell(L_a, L_b);
rho = zeros(L_a, L_b);

for i = 1:L_a
    alpha = alpha_vec(i);
    for j = 1:L_b
        beta = beta_vec(j);
        rho(i, j) = rho_HB_quad(mu, L, alpha, beta);
    end
    beta_vec2_temp = [beta_vec2 beta_opt_vec(i)];
    for j = 1:(L_b2+1)
        beta = beta_vec2_temp(j);
        E{i, j} = error_HB_quad(alpha, beta, T_vec, mu, L, P, C_n, C_const, E_0);
    end
end

fc = fc + 1; figure(fc);
J = ceil(L_a/3);
for i = 1:L_a
    subplot(J, 3, i);
    plot(beta_vec, rho(i, :));
    title(sprintf('$\\rho$ vs $\\beta$ at $\\alpha$ = %.2f', alpha_vec(i)), 'Interpreter', 'latex');
    grid on;
    if ceil(i/3) == J
        xlabel('$\beta$', 'Interpreter', 'latex');
    end
    if mod(i, 3) == 1
        ylabel('$\rho$',  'Interpreter', 'latex');
    end
end

color_order = {'r', 'b'};

fc = fc + 1; figure(fc);
J = ceil(L_a/3);
for i = 1:L_a
    beta_vec2_temp = [beta_vec2 beta_opt_vec(i)];
    subplot(J, 3, i);
    plot(T_vec, log(E{i, 1}), 'k');
    for j = 2:(L_b2)
        hold on;
        plot(T_vec, log(E{i, j}), color_order{j-1});  
    end
    hold on;
    plot(T_vec, log(E{i, L_b2+1}), '-.k');
    
    hold off;
    if ceil(i/3) == J
        xlabel('t');
    end
    if mod(i,3) == 1
        ylabel('log-bound');
    end
    set(gca, 'xlim', [0 250], 'ylim', [-10, 10]);
    title(sprintf('$\\alpha$ = %.2f, $c_w$ = %.4f', alpha_vec(i), C_n), 'Interpreter', 'latex');
    
    if i == L_a
        legend(legends_b, 'Interpreter','latex');
    end
    
end
    