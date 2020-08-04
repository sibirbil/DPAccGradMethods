clear
clc
close all
fc = 0;

%% Parameters Related to Data
d  = 2;
T = 1000;
epsilon = 1;
S1 = 0.001;

L = 1;
mu = 0.5; 
eigs = [mu L];
kappa = L/mu;

%% Beta, Stepsize, T Values

% Beta Values
% beta_vec = 0:0.001:1;
% beta_vec = sort(beta_vec);
% L_b = length(beta_vec);
% legendCell2 = cell(1, L_b);
% for i = 1:L_b
%     legendCell2{i} = ['\beta = ' sprintf('%.3f', beta_vec(i))];
% end

% Stepsize Constant
c_vec = [0.001 0.01 0.05 0.1 0.5 1];
L_c = length(c_vec);
legendCell = cell(1, L_c);
for i = 1:L_c
    legendCell{i} = ['c = ' sprintf('%.3f', c_vec(i))];
end

% Number of Iterations
T_vec = [200 500 1000 2000];
L_T = length(T_vec);

color_order = ['r', 'm', 'y', 'b', 'g', 'k', 'c'];

%% Plot 1 - Rho vs Beta
fc = fc + 1; figure(fc);

% Noise Variance
np = S1*T/epsilon;

for j = 1:L_c
    c = c_vec(j);
    alpha = c/L;
    
    % Beta Values
    b_opt = (1-sqrt(alpha*mu))/(1+sqrt(alpha*mu));
    beta_vec = [0:0.001:1 b_opt];
    beta_vec = sort(beta_vec);
    ind_b = find(beta_vec == b_opt);
    L_b = length(beta_vec);
    legendCell2 = cell(1, L_b);
    for i = 1:L_b
        legendCell2{i} = ['\beta = ' sprintf('%.3f', beta_vec(i))];
    end
    
    for k=1:L_b
        beta = beta_vec(k);
        
        for i = 1:length(eigs)
            a_lambda(i,1) = ((1+beta)*(1-alpha*eigs(i))+sqrt((1+beta)^2*(1-alpha*eigs(i))^2-4*beta*(1-alpha*eigs(i))))/2;
            a_lambda(i,2) = ((1+beta)*(1-alpha*eigs(i))-sqrt((1+beta)^2*(1-alpha*eigs(i))^2-4*beta*(1-alpha*eigs(i))))/2;
        end
        rho_vec(k) = max(max(abs(a_lambda)));

    end
    vect_of_bopt = [b_opt max(rho_vec(ind_b))];
    p(2*j-1) = plot(beta_vec, rho_vec, 'color', color_order(j), 'LineWidth', 2);
    hold on
    p(2*j) = scatter(vect_of_bopt(1),vect_of_bopt(2),'fill', color_order(j));
    hold on
    xlabel('\beta')
    ylabel('\rho')
end
hold on
legend([p(1) p(3) p(5) p(7) p(9) p(11)],legendCell);
title(['\epsilon = ' num2str(epsilon) ', T = ' num2str(T)])

%% Plot 2 - m(alpha,beta) vs beta

fc = fc + 1; figure(fc);
for t = 1:L_T
    T = T_vec(t);
    subplot(2, 2, t)
    np = S1*T/epsilon;
    for j = 1:L_c
        c = c_vec(j);
        alpha = c/L;
        for k=1:L_b

            beta = beta_vec(k);
            tr_hb = 0;
            for ind = 1:d
                tr_hb = tr_hb + 2*alpha*(1+beta)/((1-beta)*eigs(ind)*(2+2*beta-alpha*eigs(ind)));
            end
            m_a_b(k) = np^2*tr_hb;

        end
        plot(beta_vec, log(m_a_b), 'color', color_order(j),'LineWidth', 2)
        hold on
        xlabel('\beta')
        ylabel('m(\alpha,\beta)')

    end
    title(['T = ' num2str(T)])
end
legend(legendCell);

%% Plot 3 - Bound vs T

T_vec = 1:1:2000;
L_T = length(T_vec);

fc = fc + 1; figure(fc);

color_vec(:,1:3) = [[1 0 0]; [0 1 0]; [0 0 1]; [0 0 0]; ...
    [1 0 1]; [1 1 0]; [1 rand 0.25]; [rand 0.5 1]; [0.3 rand 1]; rand(L_b,1) rand(L_b,1) rand(L_b,1)];
for j = 1:L_c
    c = c_vec(j);
    alpha = c/L;
    subplot(2, 3, j)
    
    b_opt = (1-sqrt(alpha*mu))/(1+sqrt(alpha*mu));
    beta_vec = [0 0.3 0.7 0.9]; %0:0.1:1;
    beta_vec = [beta_vec b_opt]; %0:0.1:1;
    L_b = length(beta_vec);

    legendCell2 = cell(1, L_b);
    for i = 1:L_b
        if i == L_b
            legendCell2{i} = ['\beta optimum'];
        else
            legendCell2{i} = ['\beta = ' sprintf('%.2f', beta_vec(i))];
        end
    end
    
    for k=1:L_b
        beta = beta_vec(k);
        tr_hb = 0;
        
        for ind = 1:d
            tr_hb = tr_hb + 2*alpha*(1+beta)/((1-beta)*eigs(ind)*(2+2*beta-alpha*eigs(ind)));
        end

        for t=1:L_T
        T = T_vec(t);
        np = S1*T/epsilon;
        m_a_b = np^2*tr_hb;
        Ck = T;

        for i = 1:length(eigs)
            a_lambda(i,1) = ((1+beta)*(1-alpha*eigs(i))+sqrt((1+beta)^2*(1-alpha*eigs(i))^2-4*beta*(1-alpha*eigs(i))))/2;
            a_lambda(i,2) = ((1+beta)*(1-alpha*eigs(i))-sqrt((1+beta)^2*(1-alpha*eigs(i))^2-4*beta*(1-alpha*eigs(i))))/2;
        end
        rho = max(max(abs(a_lambda)));
        V_xi0 = 10 + alpha^2*np^2/(1-rho^2);
        
        rho_vec(j,k,t) = rho;
        V_for_plot(j,k,t) = V_xi0*Ck^2*rho^(2*T);
        bound_err(j,k,t) = L/2*m_a_b + V_xi0*Ck^2*rho^(2*T);
        m_for_plot(j,k,t) = L/2*m_a_b;
        end
        data_for_plot(1,:) = bound_err(j,k,:);
        plot(T_vec, log(data_for_plot), 'color', color_vec(k,:),'LineWidth', 2)
        xlabel('T')
        ylabel('Log Bound')
        title(['c = ' num2str(c)])
        hold on

    end
end
legend(legendCell2)

fc = fc + 1; figure(fc);
for j = 1:L_c
    subplot(2,3,j)
    c = c_vec(j);
    for k = 1:L_b
        data_for_plot2(1,:) = m_for_plot(j,k,:);
        plot(T_vec, log(data_for_plot2), 'color', color_vec(k,:),'LineWidth', 2)
        xlabel('T')
        ylabel('Log Bound Part 1')
        title(['c = ' num2str(c)])
        hold on
    end
end
legend(legendCell2)

fc = fc + 1; figure(fc);
for j = 1:L_c
    c = c_vec(j);
    subplot(2,3,j)
    for k = 1:L_b
        data_for_plot3(1,:) = V_for_plot(j,k,:);
        plot(T_vec, log(data_for_plot3), 'color', color_vec(k,:),'LineWidth', 2)
        xlabel('T')
        ylabel('Log Bound Part 2')
        title(['c = ' num2str(c)])
        hold on
    end
end
legend(legendCell2)


% % Example for some cases
% % Decreasing Example
% vect(1,:) = V_for_plot(1,10,:);
% vect(2,:) = m_for_plot(1,10,:);
% vect(3,:) = bound_err(1,10,:);
% rho_dec = rho_vec(1,10,1);
% 
% figure; subplot(1,3,1), plot(T_vec, log(vect(1,:)), 'r','LineWidth', 2); 
% xlabel('T')
% ylabel('log part 2')
% title('Part 2')
% hold on; subplot(1,3,2), plot(T_vec, log(vect(2,:)), 'g','LineWidth', 2);
% xlabel('T')
% ylabel('log part 1')
% title('Part 1')
% hold on; subplot(1,3,3), plot(T_vec, log(vect(3,:)), 'b','LineWidth', 2);
% xlabel('T')
% ylabel('log bound')
% title('Bound')
% sgtitle(['\beta = 0.9, c = 0.001, rho = ' num2str(rho_dec)])
% 
% % Increasing Example
% 
% vect2(1,:) = V_for_plot(4,10,:);
% vect2(2,:) = m_for_plot(4,10,:);
% vect2(3,:) = bound_err(4,10,:);
% rho_inc = rho_vec(4,10,1);
% 
% figure; subplot(1,3,1), plot(T_vec, log(vect2(1,:)), 'r','LineWidth', 2);
% xlabel('T')
% ylabel('log part 2')
% title('Part 2')
% hold on; subplot(1,3,2), plot(T_vec, log(vect2(2,:)), 'g','LineWidth', 2); 
% xlabel('T')
% ylabel('log part 1')
% title('Part 1')
% hold on; subplot(1,3,3), plot(T_vec, log(vect2(3,:)), 'b','LineWidth', 2);
% xlabel('T')
% ylabel('log bound')
% title('Bound')
% sgtitle(['\beta = 0.9, c = 0.1, rho = ' num2str(rho_inc)])
