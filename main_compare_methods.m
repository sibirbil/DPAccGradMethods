% This is the main code that generates the Figure 3 and 4 in Section 5 of the 
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

%%
clear; clc; close all;

foldertosave = 'outputfiles/';


rng(1);
n = 100000;
d  = 20;
T_vec = [100 200 500 1000]; L_T = length(T_vec);
m_vec = [1000 100000]; L_m = length(m_vec);
eps_DP_vec = 1; L_e = length(eps_DP_vec);
alg_type_mtx = [[0 0];[1 0];[2 0];[3 0];[1 1];[2 1]]; L_a = size(alg_type_mtx, 1);
% alg_type_mtx = [2 0]; L_a = size(alg_type_mtx, 1);
alg_names = {'DP-GD', 'DP-NAG', 'DP-MASG', 'DP-HB', 'DP-NAG-opt', 'DP-MASG-opt'};
c_vec = [0.1 1]; L_c = length(c_vec);
N = 20;
E0 = 10;
alter_T = 1;

%% generate data
X  = rand(n, d);
C = 20;
for i=1:n
    if norm(X(i,:),1)> C
        X(i,:) = C*X(i,:)/norm(X(i,:),1);
    end
end
w_ans = randn(d, 1);
y = sign(X*w_ans);
S1 = 2*C;

%% Logistic Regression
lambda = 0.01;
L = svds(1/n*(X'*X) + 2*lambda*eye(d), 1);
mu = 2*lambda;

%% Finding optimal point
theta0 = 10*ones(d, 1);
T0 = 1000;
al_St = 1/L*ones(1, T0);
be_St = (sqrt(L/mu)-1)/(sqrt(L/mu)+1)*ones(1, T0);
[Thetas_true] = NAG_LR(y, X, theta0, lambda, T0, n, al_St, be_St);
theta_true = Thetas_true(:, end);
F_opt = eval_F_LR(y, X, theta_true, lambda);

disp('finding optimal done');

%% Run algorithms
E = cell(L_T, L_m, L_e, L_a, L_c);

for i1 = 1:L_T
    T = T_vec(i1);
    for i2 = 1:L_m
        m = m_vec(i2);
        for i3 = 1:L_e
            eps_DP = eps_DP_vec(i3);
            for i4 = 1:L_a
                alg_type = alg_type_mtx(i4, 1);
                opt_on_off  = alg_type_mtx(i4, 2);
                for i5 = 1:L_c
                    c_alpha = c_vec(i5);
                    disp([i1, i2, i3, i4, i5]);
                    % stepsize constant
                    err_AG = zeros(N, T);
                    for i6 = 1:N
                        if alg_type == 3
                            alpha_HB = c_alpha/L;
                            beta_HB = (1 - sqrt(alpha_HB*mu))./(1 + sqrt(alpha_HB*mu));
                            Thetas = DP_HB_LR(y, X, theta0, lambda, eps_DP,...
                                T, m, S1, alpha_HB, beta_HB);
                        else
                            outputs = DP_NAG_LR(y, X, theta0, lambda, eps_DP,...
                                T, m, S1, c_alpha, mu, L, alg_type, opt_on_off,...
                                alter_T, E0);
                            Thetas = outputs.Thetas;
                        end
                        T_final = size(Thetas, 2);
                        for t = 1:T_final
                            err_AG(i6, t) = eval_F_LR(y, X, Thetas(:, t), lambda) - F_opt;
                        end
                    end
                    E{i1, i2, i3, i4, i5} = err_AG;
                end
            end
        end
    end
end

% save the output data
save([foldertosave sprintf('Comparison_of_all_algorithms_n_%d_d_%d', n, d)]);

%%
color_vec = {'k', 'b', 'r', 'g'};
fc = 0;
config_mtx = [[1 1 1];[2 1 1];[3 1 1];[4 1 1];[1 2 1];[2 2 1];[3 2 1];[4 2 1]];
L_conf = size(config_mtx, 1);

for i = 1:L_conf
    i1 = config_mtx(i, 1);
    i2 = config_mtx(i, 2);
    i3 = config_mtx(i, 3);
    
    fc = fc + 1; figure(fc);
    legends = cell(1, L_c);
    for i5 = 1:L_c
         legends{i5} = sprintf('$c =$ %.2f', c_vec(i5));
    end
    for i4 = 1:6
        subplot(2, 3, i4);
        for i5 = 1:L_c
            semilogy(1:T_vec(i1), mean(E{i1, i2, i3, i4, i5}), color_vec{i5});
            hold on;
        end
        hold off;
        set(gca, 'xlim', [0 T_vec(i1)]);
        title(alg_names{i4});
        xlabel('t');
        if mod(i4, 3) == 1
            ylabel('$F(x_{t}) - F(x^{\ast})$', 'Interpreter', 'latex');
        end
        if i4 == 6
            legend(legends, 'Interpreter', 'latex');
        end
        set(gca, 'ylim', [10^-4, 100]);
        grid on;
        
    end
    
    sgtitle(sprintf('$n =%d$, $m = %d$, $T = %d$', n, m_vec(i2), T_vec(i1)), ...
        'Interpreter', 'latex');
    
    filenametosave = [foldertosave sprintf('errors_n_%d_m_%d_T_%d_various_c',...
        n, m_vec(i2), T_vec(i1))];
    
    print(gcf, filenametosave, '-depsc');
end


%%
color_vec = {'g', 'k', 'b', 'r'};
config_mtx = [[1 1 1]; [1 1 2]; [1 1 3]; [2 1 1]; [2 1 2]; [2 1 3]];
L_conf = size(config_mtx, 1);

for i = 1:L_conf
    i2 = config_mtx(i, 1);
    i3 = config_mtx(i, 2);
    i5 = config_mtx(i, 3);
    
    fc = fc + 1; figure(fc);
    legends = cell(1, L_T);
    for i1 = 1:L_T
         legends{i1} = sprintf('$T =$ %d', T_vec(i1));
    end
    for i4 = 1:6
        subplot(2, 3, i4);
        for i1 = 1:L_T
            semilogy(1:T_vec(i1), mean(E{i1, i2, i3, i4, i5}), color_vec{i1});
            hold on;
        end
        hold off;
        set(gca, 'xlim', [0 T_vec(i1)]);
        title(alg_names{i4});
        xlabel('t');
        if mod(i4, 3) == 1
            ylabel('$F(x_{t}) - F(x^{\ast})$', 'Interpreter', 'latex');
        end
        if i4 == 6
            legend(legends, 'Interpreter', 'latex');
        end
        set(gca, 'ylim', [10^-3, 100]);
        grid on;
    end
    
    sgtitle(sprintf('$n =%d$, $m = %d$, $c = %.2f$', n, m_vec(i2), c_vec(i5)), ...
        'Interpreter', 'latex');
    
    filenametosave = [foldertosave sprintf('errors_n_%d_m_%d_c_%02d_various_T',...
        n, m_vec(i2), 10*c_vec(i5))];
    
    print(gcf, filenametosave, '-depsc');
end


