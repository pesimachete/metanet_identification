
rng shuffle
%% Simulated anealing calibrazione METANET multiclasse
for k = 1 : K
    
    for i = 2 : N + 1
        
        if Ir(i) == 1
            r_1_att(i,k) = min([ domanda_1(i,k) + l_1(i,k)/T; r_1_cap; r_1_cap*((rho_max(i,1)*lane(i) - rho_1(i,k)*lane(i) - eta*rho_2(i,k)*lane(i))/(rho_max(i,1)*lane(i) - rho_cr*lane(i))) ]);
            r_2_att(i,k) = min([ domanda_2(i,k) + l_2(i,k)/T; r_2_cap; r_2_cap*((rho_max(i,1)*lane(i) - rho_1(i,k)*lane(i) - eta*rho_2(i,k)*lane(i))/(rho_max(i,1)*lane(i) - rho_cr*lane(i))) ]);
        end
        
        q_1(i,k) = rho_1(i,k)*lane(i)*v_1(i,k)*(1 - s_1(i,k));
        q_2(i,k) = rho_2(i,k)*lane(i)*v_2(i,k)*(1 - s_2(i,k));
        
        q_1_tot(i,k + 1) = rho_1(i,k)*lane(i)*v_1(i,k);
        q_2_tot(i,k + 1) = rho_2(i,k)*lane(i)*v_2(i,k);
        
        
        l_1(i,k + 1) = l_1(i,k) + T*(domanda_1(i,k) - r_1_att(i,k));
        l_2(i,k + 1) = l_2(i,k) + T*(domanda_2(i,k) - r_2_att(i,k));
 
        %% Traffic Density in [veh/km/lane]
        rho_1(i,k + 1) = rho_1(i,k) + (T/(Delta(i,1)*lane(i))) * (q_1(i - 1,k) - q_1(i,k)/(1 - s_1(i,k)) + r_1_att(i,k));
        rho_2(i,k + 1) = rho_2(i,k) + (T/(Delta(i,1)*lane(i))) * (q_2(i - 1,k) - q_2(i,k)/(1 - s_2(i,k)) + r_2_att(i,k));
        
        
        %% Fundamental diagram
        
        v_fund_1 = vf_1*exp(( - 1/aexp1)*((rho_1(i,k) + eta*rho_2(i,k))/rho_cr)^aexp1);
        
        v_fund_2 = vf_2*exp(( - 1/aexp2)*((rho_1(i,k) + eta*rho_2(i,k))/rho_cr)^aexp2);
        %
        %                v_fund_1 = vf_1*(1 - ((rho_1(i,k) + eta*rho_2(i,k))/rho_max(i,1))^l1)^m1);
        %
        %                v_fund_2 = vf_2*(1 - ((rho_1(i,k) + eta*rho_2(i,k))/rho_max(i,1))^l2)^m2);
        
        if (rho_1(i,k) + eta*rho_2(i,k))<0.001
            v_fund_1 = vf_1;
        end
        
        %% Speed Dynamics
        
        if i == 2
            
            v_1(i,k + 1) = v_1(i,k) + (T/tau_1)*(v_fund_1 - v_1(i,k))...
                - ((nu_1*T*(rho_1(i + 1,k) + eta*rho_2(i + 1,k) - rho_1(i,k) - eta*rho_2(i,k)))/(Delta(i,1)*tau_1*(rho_1(i,k) + eta*rho_2(i,k) + chi_1)));
            
            v_2(i,k + 1) = v_2(i,k) + (T/tau_2)*(v_fund_2  - v_2(i,k))...
                - ((nu_2*T*(rho_1(i + 1,k) + eta*rho_2(i + 1,k) - rho_1(i,k) - eta*rho_2(i,k)))/(Delta(i,1)*tau_2*(rho_1(i,k) + eta*rho_2(i,k) + chi_2)));
            
            %Merge ramp term
            if Ir(i) == 1
                
                v_1(i,k + 1) = v_1(i,k + 1) - ((delta_on_1*T*v_1(i,k)*(r_1_att(i,k) + eta*r_2_att(i,k)))/(Delta(i,1)*lane(i)*(rho_1(i,k) + eta*rho_2(i,k) + chi_1)));
                
                v_2(i,k + 1) = v_2(i,k + 1) - ((delta_on_2*T*v_2(i,k)*(r_1_att(i,k) + eta*r_2_att(i,k)))/(Delta(i,1)*lane(i)*(rho_1(i,k) + eta*rho_2(i,k) + chi_2)));
                
            end
            %Lane drop term
            
            if lane(i) > lane(i + 1)
                
                v_1(i,k + 1) = v_1(i,k + 1)  - ( phi*T*(lane(i) - lane(i + 1))*v_1(i,k)^2*(rho_1(i,k) + eta*rho_2(i,k)) )/(Delta(i,1)*lane(i)*rho_cr);
                
                v_2(i,k + 1) = v_2(i,k + 1)  - ( phi*T*(lane(i) - lane(i + 1))*v_2(i,k)^2*(rho_1(i,k) + eta*rho_2(i,k)) )/(Delta(i,1)*lane(i)*rho_cr);
                
            end
            
            
        elseif i == N + 1
            
            
            v_1(i,k + 1) = v_1(i,k) + (T/tau_1)*(v_fund_1 - v_1(i,k))...
                + (T/Delta(i,1))*v_1(i,k)*(v_1(i - 1,k) - v_1(i,k));
            
            v_2(i,k + 1) = v_2(i,k) + (T/tau_2)*(v_fund_2  - v_2(i,k))...
                + (T/Delta(i,1))*v_2(i,k)*(v_2(i - 1,k) - v_2(i,k));
            
            %Merge ramp term
            if Ir(i) == 1
                
                v_1(i,k + 1) = v_1(i,k + 1) - ((delta_on_1*T*v_1(i,k)*(r_1_att(i,k) + eta*r_2_att(i,k)))/(Delta(i,1)*lane(i)*(rho_1(i,k) + eta*rho_2(i,k) + chi_1)));
                
                v_2(i,k + 1) = v_2(i,k + 1) - ((delta_on_2*T*v_2(i,k)*(r_1_att(i,k) + eta*r_2_att(i,k)))/(Delta(i,1)*lane(i)*(rho_1(i,k) + eta*rho_2(i,k) + chi_2)));
                
            end
            
            %Lane drop term
            if lane(i) > lane(i + 1)
                
                v_1(i,k + 1) = v_1(i,k + 1)  - ( phi*T*(lane(i) - lane(i + 1))*v_1(i,k)^2*(rho_1(i,k) + eta*rho_2(i,k)) )/(Delta(i,1)*lane(i)*rho_cr);
                
                v_2(i,k + 1) = v_2(i,k + 1)  - ( phi*T*(lane(i) - lane(i + 1))*v_2(i,k)^2*(rho_1(i,k) + eta*rho_2(i,k)) )/(Delta(i,1)*lane(i)*rho_cr);
                
            end
            
            
            if (rho_1(i,k) + eta*rho_2(i,k)) > rho_cr
                v_1(i,k + 1) = v_1(i,k + 1) - nu_1*T*(rho_cr - rho_1(i,k) + eta*rho_2(i,k))/(tau_1*Delta(i,1)*(rho_1(i,k) + eta*rho_2(i,k) + chi_1));
                
                v_2(i,k + 1) = v_2(i,k + 1) - nu_2*T*(rho_cr - rho_1(i,k) + eta*rho_2(i,k))/(tau_2*Delta(i,1)*(rho_1(i,k) + eta*rho_2(i,k) + chi_2));
            end
            
            
        else
            
            v_1(i,k + 1) = v_1(i,k) + (T/tau_1)*(v_fund_1 - v_1(i,k))...
                + (T/Delta(i,1))*v_1(i,k)*(v_1(i - 1,k) - v_1(i,k))...
                - ((nu_1*T*(rho_1(i + 1,k) + eta*rho_2(i + 1,k) - rho_1(i,k) - eta*rho_2(i,k)))/(Delta(i,1)*tau_1*(rho_1(i,k) + eta*rho_2(i,k) + chi_1)));
            
            v_2(i,k + 1) = v_2(i,k) + (T/tau_2)*(v_fund_2  - v_2(i,k))...
                + (T/Delta(i,1))*v_2(i,k)*(v_2(i - 1,k) - v_2(i,k))...
                - ((nu_2*T*(rho_1(i + 1,k) + eta*rho_2(i + 1,k) - rho_1(i,k) - eta*rho_2(i,k)))/(Delta(i,1)*tau_2*(rho_1(i,k) + eta*rho_2(i,k) + chi_2)));
            
            
            %Merge ramp term
            
            if Ir(i) == 1
                
                v_1(i,k + 1) = v_1(i,k + 1) - ((delta_on_1*T*v_1(i,k)*(r_1_att(i,k) + eta*r_2_att(i,k)))/(Delta(i,1)*lane(i)*(rho_1(i,k) + eta*rho_2(i,k) + chi_1)));
                
                v_2(i,k + 1) = v_2(i,k + 1)- ((delta_on_2*T*v_2(i,k)*(r_1_att(i,k) + eta*r_2_att(i,k)))/(Delta(i,1)*lane(i)*(rho_1(i,k) + eta*rho_2(i,k) + chi_2)));
                
            end
            
            %Lane drop term
            
            if lane(i) > lane(i + 1)
                
                v_1(i,k + 1) = v_1(i,k + 1)  - ( phi*T*(lane(i) - lane(i + 1))*v_1(i,k)^2*(rho_1(i,k) + eta*rho_2(i,k)) )/(Delta(i,1)*lane(i)*rho_cr);
                
                
                v_2(i,k + 1) = v_2(i,k + 1)  - ( phi*T*(lane(i) - lane(i + 1))*v_2(i,k)^2*(rho_1(i,k) + eta*rho_2(i,k)) )/(Delta(i,1)*lane(i)*rho_cr);
                
            end
            
            
        end
        
        v_1(i,k + 1) = max(v_min_1,v_1(i,k + 1));
        v_2(i,k + 1) = max(v_min_2,v_2(i,k + 1));
        
        
        
    end
end

Model_speed1 = v_1(Det_position,time_position);
Model_flow1 = q_1_tot(Det_position,time_position);

for k = 1:numel(time_position)
    for i = 1:numel(Det_position)
        Model_density1(i,k) = Model_flow1(i,k)/Model_speed1(i,k);
    end
end

% %
% % Model_speed2 = v_2(Det_position,time_position);
% % Model_flow2 = q_2_tot_tot(Det_position,time_position);
%
J(1,1)=0;

%%per ora è sigle class
for time = 1: ltime_mis
    for sez = 1 : Det
        Error_speed_1(sez,time) = (Real_speed(time,sez) - Model_speed1(sez,time));
        Error_flow_1(sez,time) = (Real_flow(time,sez) - Model_flow1(sez,time));
    end
end
%
% Error_speed_2 = (Real_speed2 - Model_speed2);
% Error_flow_2 = (Real_flow2 - Model_flow2);


RMSE_speed = sqrt(sum(Error_speed_1(:).^2)/numel(Error_speed_1));
RMSE_flow = sqrt(sum(Error_flow_1(:).^2)/numel(Error_flow_1));
%
%
% RMSE_speed2 = sqrt(sum(Error_speed_2(:).^2)/numel(Error_speed_2));
% RMSE_flow2 = sqrt(sum(Error_flow_2(:).^2)/numel(Error_flow_2));

J(1,1) = w_speed*RMSE_speed + w_flow*RMSE_flow; % + w_speed2*RMSE_speed2 + w_flow2*RMSE_flow2

if any(Model_parameter1 < 0)% || Model_parameter2 < 0
    J(1,1) = J(1,1) + 20000;
end

Andamento_J = J(1,1);
%
for k = 1 : K
    
    for i = 2 : N + 1
        
        if Ir(i) == 1
            r_1_att_best(i,k) = min([ domanda_1(i,k) + l_1_best(i,k)/T; r_1_cap; r_1_cap*((rho_max(i,1) - rho_1_best(i,k) - eta*rho_2_best(i,k))/(rho_max(i,1) - rho_cr_best)) ]);
            r_2_att_best(i,k) = min([ domanda_2(i,k) + l_2_best(i,k)/T; r_2_cap; r_2_cap*((rho_max(i,1) - rho_1_best(i,k) - eta*rho_2_best(i,k))/(rho_max(i,1) - rho_cr_best)) ]);
        end
        
        l_1_best(i,k + 1) = l_1_best(i,k) + T*(domanda_1(i,k) - r_1_att_best(i,k));
        l_2_best(i,k + 1) = l_2_best(i,k) + T*(domanda_2(i,k) - r_2_att_best(i,k));
        
        q_1_best(i,k) = rho_1_best(i,k)*lane(i)*v_1_best(i,k)*(1 - s_1(i,k));
        q_2_best(i,k) = rho_2_best(i,k)*lane(i)*v_2_best(i,k)*(1 - s_2(i,k));
        
        q_1_tot_best(i,k + 1) = rho_1_best(i,k)*lane(i)*v_1_best(i,k);
        q_2_tot_best(i,k + 1) = rho_2_best(i,k)*lane(i)*v_2_best(i,k);
        
        %% Traffic Density in [veh/km/lane]
        
        rho_1_best(i,k + 1) = rho_1_best(i,k) + (T/(Delta(i,1)*lane(i))) * (q_1_best(i - 1,k) - q_1_best(i,k)/(1 - s_1(i,k)) + r_1_att_best(i,k));
        rho_2_best(i,k + 1) = rho_2_best(i,k) + (T/(Delta(i,1)*lane(i))) * (q_2_best(i - 1,k) - q_2_best(i,k)/(1 - s_1(i,k)) + r_2_att_best(i,k));
        
        
        %% Fundamental diagram
        
        v_fund_1_best = vf_1_best*exp(( - 1/aexp1_best)*((rho_1_best(i,k) + eta*rho_2_best(i,k))/rho_cr_best)^aexp1_best);
        
        v_fund_2_best = vf_2_best*exp(( - 1/aexp2_best)*((rho_1_best(i,k) + eta*rho_2_best(i,k))/rho_cr_best)^aexp2_best);
        
        %        v_fund_1_best = vf_1_best*(1 - ((rho_1_best(i,k) + eta*rho_2_best(i,k))/rho_max(i,1))^l1_best)^m1_best);
        
        %        v_fund_2_best = vf_2_best*(1 - ((rho_1_best(i,k) + eta*rho_2_best(i,k))/rho_max(i,1))^l2_best)^m2_best);
        
        if (rho_1_best(i,k) + eta*rho_2_best(i,k))<0.001
            v_fund_1_best = vf_1_best;
        end
        
        %% Speed Dynamics
        
        if i == 2
            
            v_1_best(i,k + 1) = v_1_best(i,k) + (T/tau_1_best)*(v_fund_1_best - v_1_best(i,k))...
                - ((nu_1_best*T*(rho_1_best(i + 1,k) + eta*rho_2_best(i + 1,k) - rho_1_best(i,k) - eta*rho_2_best(i,k)))/(Delta(i,1)*tau_1_best*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1)));
            
            v_2_best(i,k + 1) = v_2_best(i,k) + (T/tau_2_best)*(v_fund_2_best  - v_2_best(i,k))...
                - ((nu_2_best*T*(rho_1_best(i + 1,k) + eta*rho_2_best(i + 1,k) - rho_1_best(i,k) - eta*rho_2_best(i,k)))/(Delta(i,1)*tau_2_best*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2)));
            
            % Merge ramp term
            if Ir(i) == 1
                
                v_1_best(i,k + 1) = v_1_best(i,k + 1) - ((delta_on_1_best*T*v_1_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1)));
                
                v_2_best(i,k + 1) = v_2_best(i,k + 1) - ((delta_on_2_best*T*v_2_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2)));
                
            end
            % Lane drop term
            
            if lane(i) > lane(i + 1)
                
                v_1_best(i,k + 1) = v_1_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_1_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                
                v_2_best(i,k + 1) = v_2_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_2_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                
            end
            
            
        elseif i == N + 1
            
            
            v_1_best(i,k + 1) = v_1_best(i,k) + (T/tau_1_best)*(v_fund_1_best - v_1_best(i,k))...
                + (T/Delta(i,1))*v_1_best(i,k)*(v_1_best(i - 1,k) - v_1_best(i,k));
            
            v_2_best(i,k + 1) = v_2_best(i,k) + (T/tau_2_best)*(v_fund_2_best  - v_2_best(i,k))...
                + (T/Delta(i,1))*v_2_best(i,k)*(v_2_best(i - 1,k) - v_2_best(i,k));
            
            % Merge ramp term
            if Ir(i) == 1
                
                v_1_best(i,k + 1) = v_1_best(i,k + 1) - ((delta_on_1_best*T*v_1_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1)));
                
                v_2_best(i,k + 1) = v_2_best(i,k + 1) - ((delta_on_2_best*T*v_2_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2)));
                
            end
            
            if lane(i) > lane(i + 1)
                
                v_1_best(i,k + 1) = v_1_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_1_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                
                v_2_best(i,k + 1) = v_2_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_2_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                
            end
            
            
            if (rho_1_best(i,k) + eta*rho_2_best(i,k)) > rho_cr_best
                v_1_best(i,k + 1) = v_1_best(i,k + 1) - nu_1_best*T*(rho_cr_best - rho_1_best(i,k) + eta*rho_2_best(i,k))/(tau_1_best*Delta(i,1)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1));
                
                v_2_best(i,k + 1) = v_2_best(i,k + 1) - nu_2_best*T*(rho_cr_best - rho_1_best(i,k) + eta*rho_2_best(i,k))/(tau_2_best*Delta(i,1)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2));
            end
            
            
        else
            
            v_1_best(i,k + 1) = v_1_best(i,k) + (T/tau_1_best)*(v_fund_1_best - v_1_best(i,k))...
                + (T/Delta(i,1))*v_1_best(i,k)*(v_1_best(i - 1,k) - v_1_best(i,k))...
                - ((nu_1_best*T*(rho_1_best(i + 1,k) + eta*rho_2_best(i + 1,k) - rho_1_best(i,k) - eta*rho_2_best(i,k)))/(Delta(i,1)*tau_1_best*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1)));
            
            v_2_best(i,k + 1) = v_2_best(i,k) + (T/tau_2_best)*(v_fund_2_best  - v_2_best(i,k))...
                + (T/Delta(i,1))*v_2_best(i,k)*(v_2_best(i - 1,k) - v_2_best(i,k))...
                - ((nu_2_best*T*(rho_1_best(i + 1,k) + eta*rho_2_best(i + 1,k) - rho_1_best(i,k) - eta*rho_2_best(i,k)))/(Delta(i,1)*tau_2_best*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2)));
            
            
            % Merge ramp term
            
            if Ir(i) == 1
                
                v_1_best(i,k + 1) = v_1_best(i,k + 1) - ((delta_on_1_best*T*v_1_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1)));
                
                v_2_best(i,k + 1) = v_2_best(i,k + 1)- ((delta_on_2_best*T*v_2_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2)));
                
            end
            
            % Lane drop term
            
            if lane(i) > lane(i + 1)
                
                v_1_best(i,k + 1) = v_1_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_1_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                
                
                v_2_best(i,k + 1) = v_2_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_2_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                
            end
            
            
        end
        
        v_1_best(i,k + 1) = max(v_min_1,v_1_best(i,k + 1));
        v_2_best(i,k + 1) = max(v_min_2,v_2_best(i,k + 1));
        
        
        
        
    end
end

J_best = 0.0;

Model_speed1_best = v_1_best(Det_position,time_position);
Model_flow1_best = q_1_tot_best(Det_position,time_position);

for k = 1:numel(time_position)
    for i = 1:numel(Det_position)
        Model_density1_best(i,k) = Model_flow1_best(i,k)/Model_speed1_best(i,k);
    end
end

%
% Model_speed2_best = v_2_best(Det_position,time_position);
% Model_flow2_best = q_2_tot_best(Det_position,time_position);


%%per ora è sigle class
for time = 1: ltime_mis
    for sez = 1 : Det
        Error_speed_1_best(sez,time) = (Real_speed(time,sez) - Model_speed1_best(sez,time));
        Error_flow_1_best(sez,time) = (Real_flow(time,sez) - Model_flow1_best(sez,time));
    end
end
%
% Error_speed_2_best = (Real_speed2 - Model_speed2_best);
% Error_flow_2_best = (Real_flow2 - Model_flow2_best);


RMSE_speed_best = sqrt(sum(Error_speed_1_best(:).^2)/numel(Error_speed_1_best));
RMSE_flow_best = sqrt(sum(Error_flow_1_best(:).^2)/numel(Error_flow_1_best));
%
%
% RMSE_speed2_best = sqrt(sum(Error_speed_2_best(:).^2)/numel(Error_speed_2_best));
% RMSE_flow2_best = sqrt(sum(Error_flow_2_best(:).^2)/numel(Error_flow_2_best));

J_best = w_speed*RMSE_speed_best + w_flow*RMSE_flow_best; % + w_speed2*RMSE_speed2 + w_flow2*RMSE_flow2

if any(Model_parameter1_best < 0)% || Model_parameter2 < 0
    J_best = J_best + 20000;
end

j=0;
while(true)
    j=j+1;
    
    
    %% Genero soluzione tentative
    
    for p = 1 : param
        
        Model_parameter1_tentative(p) = Model_parameter1(p) + (ub1(p,1) - lb1(p,1)) * radius * 2.0 * rand  - (ub1(p,1) - lb1(p,1)) * radius;
        
    end
    
    for p = 1 : param - 1
        Model_parameter2_tentative(p) = Model_parameter2(p) + (ub2(p,1) - lb2(p,1)) * radius * 2.0 * rand  - (ub2(p,1) - lb2(p,1)) * radius;
        
    end
    %% Aggiorno le variabili di controllo e verifico sat
    
        for p = 1 : param
    
            if Model_parameter1_tentative(p) < lb1(p)
                Model_parameter1_tentative(p) = lb1(p);
    
            elseif Model_parameter1_tentative(p) > ub1(p)
                Model_parameter1_tentative(p) = ub1(p);
            end
    
        end
    
        for p = 1 : param - 1
    
            if Model_parameter2_tentative(p) < lb2(p)
                Model_parameter2_tentative(p) = lb2(p);
    
            elseif Model_parameter2_tentative(p) > ub2(p)
                Model_parameter2_tentative(p) = ub2(p);
            end
        end
    
    tau_1_tent = Model_parameter1_tentative(1)/3600;
    tau_2_tent = Model_parameter2_tentative(1)/3600;
    
    nu_1_tent = Model_parameter1_tentative(2);
    nu_2_tent = Model_parameter2_tentative(2);
    
    delta_on_1_tent = Model_parameter1_tentative(3);
    delta_on_2_tent = Model_parameter2_tentative(3);
    
    aexp1_tent = Model_parameter1_tentative(4);
    aexp2_tent = Model_parameter2_tentative(4);
    
    phi_tent = Model_parameter1_tentative(5);
    %     l1_tent = Model_parameter1_tentative(4);
    %     l2_tent = Model_parameter2_tentative(4);
    %
    %     m1_tent = Model_parameter1_tentative(5);
    %     m2_tent = Model_parameter2_tentative(5);
    
    vf_1_tent = Model_parameter1_tentative(6);
    vf_2_tent = Model_parameter2_tentative(6);
    
    rho_cr_tent = Model_parameter1_tentative(7);
    
    
    
    %% calcolo lo stato con le nuove variabili di controllo
    for k = 1 : K
        
        for i = 2 : N + 1
            
            
            if Ir(i) == 1
                r_1_att_tentative(i,k) = min([ domanda_1(i,k) + l_1_tentative(i,k)/T; r_1_cap; r_1_cap*((rho_max(i,1) - rho_1_tentative(i,k) - eta*rho_2_tentative(i,k))/(rho_max(i,1) - rho_cr_tent)) ]);
                r_2_att_tentative(i,k) = min([ domanda_2(i,k) + l_2_tentative(i,k)/T; r_2_cap; r_2_cap*((rho_max(i,1) - rho_1_tentative(i,k) - eta*rho_2_tentative(i,k))/(rho_max(i,1) - rho_cr_tent)) ]);
            end
            
            l_1_tentative(i,k + 1) = l_1_tentative(i,k) + T*(domanda_1(i,k) - r_1_att_tentative(i,k));
            l_2_tentative(i,k + 1) = l_2_tentative(i,k) + T*(domanda_2(i,k) - r_2_att_tentative(i,k));
            
            q_1_tentative(i,k) = rho_1_tentative(i,k)*lane(i)*v_1_tentative(i,k)*(1 - s_1(i,k));
            q_2_tentative(i,k) = rho_2_tentative(i,k)*lane(i)*v_2_tentative(i,k)*(1 - s_2(i,k));
            
            q_1_tot_tentative(i,k + 1) = rho_1_tentative(i,k)*lane(i)*v_1_tentative(i,k);
            q_2_tot_tentative(i,k + 1) = rho_2_tentative(i,k)*lane(i)*v_2_tentative(i,k);
            
            %% Traffic Density in [veh/km/lane]
            
            rho_1_tentative(i,k + 1) = rho_1_tentative(i,k) + (T/(Delta(i,1)*lane(i))) * (q_1_tentative(i - 1,k) - q_1_tentative(i,k)/(1 - s_1(i,k)) + r_1_att_tentative(i,k));
            rho_2_tentative(i,k + 1) = rho_2_tentative(i,k) + (T/(Delta(i,1)*lane(i))) * (q_2_tentative(i - 1,k) - q_2_tentative(i,k)/(1 - s_1(i,k)) + r_2_att_tentative(i,k));
            
            %% Fundamental diagram
            
            v_fund_1_tentative = vf_1_tent*exp(( - 1/aexp1_tent)*((rho_1_tentative(i,k) + eta*rho_2_tentative(i,k))/rho_cr_tent)^aexp1_tent);
            
            v_fund_2_tentative = vf_2_tent*exp(( - 1/aexp2_tent)*((rho_1_tentative(i,k) + eta*rho_2_tentative(i,k))/rho_cr_tent)^aexp2_tent);
            
            %        v_fund_1_tentative = vf_1_tent*(1 - ((rho_1_tentative(i,k) + eta*rho_2_tentative(i,k))/rho_max(i,1))^l1_tent)^m1_tent);
            
            %        v_fund_2_tentative = vf_2_tent*(1 - ((rho_1_tentative(i,k) + eta*rho_2_tentative(i,k))/rho_max(i,1))^l2_tent)^m2_tent);
            
            if (rho_1_tentative(i,k) + eta*rho_2_tentative(i,k))<0.001
                v_fund_1_tentative = vf_1_tent;
            end
            
            %% Speed Dynamics
            
            if i == 2
                
                v_1_tentative(i,k + 1) = v_1_tentative(i,k) + (T/tau_1_tent)*(v_fund_1_tentative - v_1_tentative(i,k))...
                    - ((nu_1_tent*T*(rho_1_tentative(i + 1,k) + eta*rho_2_tentative(i + 1,k) - rho_1_tentative(i,k) - eta*rho_2_tentative(i,k)))/(Delta(i,1)*tau_1_tent*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_1)));
                
                v_2_tentative(i,k + 1) = v_2_tentative(i,k) + (T/tau_2_tent)*(v_fund_2_tentative  - v_2_tentative(i,k))...
                    - ((nu_2_tent*T*(rho_1_tentative(i + 1,k) + eta*rho_2_tentative(i + 1,k) - rho_1_tentative(i,k) - eta*rho_2_tentative(i,k)))/(Delta(i,1)*tau_2_tent*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_2)));
                
                % Merge ramp term
                if Ir(i) == 1
                    
                    v_1_tentative(i,k + 1) = v_1_tentative(i,k + 1) - ((delta_on_1_tent*T*v_1_tentative(i,k)*(r_1_att_tentative(i,k) + eta*r_2_att_tentative(i,k)))/(Delta(i,1)*lane(i)*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_1)));
                    
                    v_2_tentative(i,k + 1) = v_2_tentative(i,k + 1) - ((delta_on_2_tent*T*v_2_tentative(i,k)*(r_1_att_tentative(i,k) + eta*r_2_att_tentative(i,k)))/(Delta(i,1)*lane(i)*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_2)));
                    
                end
                % Lane drop term
                
                if lane(i) > lane(i + 1)
                    
                    v_1_tentative(i,k + 1) = v_1_tentative(i,k + 1)  - ( phi_tent*T*(lane(i) - lane(i + 1))*v_1_tentative(i,k)^2*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_tent);
                    
                    v_2_tentative(i,k + 1) = v_2_tentative(i,k + 1)  - ( phi_tent*T*(lane(i) - lane(i + 1))*v_2_tentative(i,k)^2*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_tent);
                    
                end
                
                
            elseif i == N + 1
                
                
                v_1_tentative(i,k + 1) = v_1_tentative(i,k) + (T/tau_1_tent)*(v_fund_1_tentative - v_1_tentative(i,k))...
                    + (T/Delta(i,1))*v_1_tentative(i,k)*(v_1_tentative(i - 1,k) - v_1_tentative(i,k));
                
                v_2_tentative(i,k + 1) = v_2_tentative(i,k) + (T/tau_2_tent)*(v_fund_2_tentative - v_2_tentative(i,k))...
                    + (T/Delta(i,1))*v_2_tentative(i,k)*(v_2_tentative(i - 1,k) - v_2_tentative(i,k));
                
                % Merge ramp term
                if Ir(i) == 1
                    
                    v_1_tentative(i,k + 1) = v_1_tentative(i,k + 1) - ((delta_on_1_tent*T*v_1_tentative(i,k)*(r_1_att_tentative(i,k) + eta*r_2_att_tentative(i,k)))/(Delta(i,1)*lane(i)*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_1)));
                    
                    v_2_tentative(i,k + 1) = v_2_tentative(i,k + 1) - ((delta_on_2_tent*T*v_2_tentative(i,k)*(r_1_att_tentative(i,k) + eta*r_2_att_tentative(i,k)))/(Delta(i,1)*lane(i)*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_2)));
                    
                end
                
                if lane(i) > lane(i + 1)
                    
                    v_1_tentative(i,k + 1) = v_1_tentative(i,k + 1)  - ( phi_tent*T*(lane(i) - lane(i + 1))*v_1_tentative(i,k)^2*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_tent);
                    
                    v_2_tentative(i,k + 1) = v_2_tentative(i,k + 1)  - ( phi_tent*T*(lane(i) - lane(i + 1))*v_2_tentative(i,k)^2*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_tent);
                    
                end
                
                
                if (rho_1_tentative(i,k) + eta*rho_2_tentative(i,k)) > rho_cr_tent
                    v_1_tentative(i,k + 1) = v_1_tentative(i,k + 1) - nu_1_tent*T*(rho_cr_tent - rho_1_tentative(i,k) + eta*rho_2_tentative(i,k))/(tau_1_tent*Delta(i,1)*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_1));
                    
                    v_2_tentative(i,k + 1) = v_2_tentative(i,k + 1) - nu_2_tent*T*(rho_cr_tent - rho_1_tentative(i,k) + eta*rho_2_tentative(i,k))/(tau_2_tent*Delta(i,1)*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_2));
                end
                
                
            else
                
                v_1_tentative(i,k + 1) = v_1_tentative(i,k) + (T/tau_1_tent)*(v_fund_1_tentative - v_1_tentative(i,k))...
                    + (T/Delta(i,1))*v_1_tentative(i,k)*(v_1_tentative(i - 1,k) - v_1_tentative(i,k))...
                    - ((nu_1_tent*T*(rho_1_tentative(i + 1,k) + eta*rho_2_tentative(i + 1,k) - rho_1_tentative(i,k) - eta*rho_2_tentative(i,k)))/(Delta(i,1)*tau_1_tent*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_1)));
                
                v_2_tentative(i,k + 1) = v_2_tentative(i,k) + (T/tau_2_tent)*(v_fund_2_tentative  - v_2_tentative(i,k))...
                    + (T/Delta(i,1))*v_2_tentative(i,k)*(v_2_tentative(i - 1,k) - v_2_tentative(i,k))...
                    - ((nu_2_tent*T*(rho_1_tentative(i + 1,k) + eta*rho_2_tentative(i + 1,k) - rho_1_tentative(i,k) - eta*rho_2_tentative(i,k)))/(Delta(i,1)*tau_2_tent*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_2)));
                
                
                % Merge ramp term
                
                if Ir(i) == 1
                    
                    v_1_tentative(i,k + 1) = v_1_tentative(i,k + 1) - ((delta_on_1_tent*T*v_1_tentative(i,k)*(r_1_att_tentative(i,k) + eta*r_2_att_tentative(i,k)))/(Delta(i,1)*lane(i)*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_1)));
                    
                    v_2_tentative(i,k + 1) = v_2_tentative(i,k + 1)- ((delta_on_2_tent*T*v_2_tentative(i,k)*(r_1_att_tentative(i,k) + eta*r_2_att_tentative(i,k)))/(Delta(i,1)*lane(i)*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k) + chi_2)));
                    
                end
                
                % Lane drop term
                
                if lane(i) > lane(i + 1)
                    
                    v_1_tentative(i,k + 1) = v_1_tentative(i,k + 1)  - ( phi_tent*T*(lane(i) - lane(i + 1))*v_1_tentative(i,k)^2*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_tent);
                    
                    
                    v_2_tentative(i,k + 1) = v_2_tentative(i,k + 1)  - ( phi_tent*T*(lane(i) - lane(i + 1))*v_2_tentative(i,k)^2*(rho_1_tentative(i,k) + eta*rho_2_tentative(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_tent);
                    
                end
                
                
            end
            
            v_1_tentative(i,k + 1) = max(v_min_1,v_1_tentative(i,k + 1));
            v_2_tentative(i,k + 1) = max(v_min_2,v_2_tentative(i,k + 1));
            
        end
    end
    %calcolo il nuovo funzionale di costo
    J(1,2)=0;
    
    Model_speed1_tentative = v_1_tentative(Det_position,time_position);
    Model_flow1_tentative = q_1_tot_tentative(Det_position,time_position);
    
    for k = 1:numel(time_position)
        for i = 1:numel(Det_position)
            Model_density1_tentative(i,k) = Model_flow1_tentative(i,k)/Model_speed1_tentative(i,k);
        end
    end
    %
    % Model_speed2_tentative = v_2_tentative(Det_position,time_position);
    % Model_flow2_tentative = q_2_tot_tentative(Det_position,time_position);
    
    
    %%per ora è sigle class
    for time = 1: ltime_mis
        for sez = 1 : Det
            Error_speed_1_tentative(sez,time) = (Real_speed(time,sez) - Model_speed1_tentative(sez,time));
            Error_flow_1_tentative(sez,time) = (Real_flow(time,sez) - Model_flow1_tentative(sez,time));
        end
    end
    %
    % Error_speed_2_tentative = (Real_speed2 - Model_speed2_tentative);
    % Error_flow_2_tentative = (Real_flow2 - Model_flow2_tentative);
    
    
    RMSE_speed_tentative = sqrt(sum(Error_speed_1_tentative(:).^2)/numel(Error_speed_1_tentative));
    RMSE_flow_tentative = sqrt(sum(Error_flow_1_tentative(:).^2)/numel(Error_flow_1_tentative));
    %
    %
    % RMSE_speed2_tentative = sqrt(sum(Error_speed_2_tentative(:).^2)/numel(Error_speed_2_tentative));
    % RMSE_flow2_tentative = sqrt(sum(Error_flow_2_tentative(:).^2)/numel(Error_flow_2_tentative));
    
    J(1,2) = w_speed*RMSE_speed_tentative + w_flow*RMSE_flow_tentative; % + w_speed2*RMSE_speed2 + w_flow2*RMSE_flow2
    
    if any(Model_parameter1_tentative < 0)% || Model_parameter2 < 0
        J(1,2) = J(1,2) + 20000;
    end
    
    
    Andamento_J = horzcat(Andamento_J, J(1,2));
    
    disp(J(1,2));
    
    C =  - J(1,1) * percPeggiorazioneAccettazione / log(probAccettazione);
    prob = exp((J(1,1) - J(1,2)) / C);
    %
    %     C = abs(J(1,1)/log(probAccettazione))*0.9;
    %     prob = exp( - (J(1,1) - J(1,2)) / C);
    
    if (J(1,2) < J(1,1) || (rand < prob))
        
        
        Model_parameter1(:,:) = Model_parameter1_tentative(:,:);
        Model_parameter2(:,:) = Model_parameter2_tentative(:,:);
        
        J(1,1) = J(1,2);
        

    end
    
    if (J(1,2) < J_best)
        
        
        Model_parameter1_best(:,:) = Model_parameter1_tentative(:,:);
        Model_parameter2_best(:,:) = Model_parameter2_tentative(:,:);
        
        tau_1_best = Model_parameter1(1)/3600;
        tau_2_best = Model_parameter2(1)/3600;
        
        nu_1_best = Model_parameter1(2);
        nu_2_best = Model_parameter2(2);
        
        delta_on_1_best = Model_parameter1(3);
        delta_on_2_best = Model_parameter2(3);
        
        aexp1_best = Model_parameter1(4);
        aexp2_best = Model_parameter2(4);
        
        phi_best = Model_parameter1(5);
        
        %         l1_best = Model_parameter1(4);
        %         l2_best = Model_parameter2(4);
        %
        %         m1_best = Model_parameter1(5);
        %         m2_best = Model_parameter2(5);
        
        vf_1_best = Model_parameter1(6);
        vf_2_best = Model_parameter2(6);
        
        rho_cr_best = Model_parameter1(7);
        
        for k = 1 : K
            
            for i = 2 : N + 1
                
                if Ir(i) == 1
                    r_1_att_best(i,k) = min([ domanda_1(i,k) + l_1_best(i,k)/T; r_1_cap; r_1_cap*((rho_max(i,1) - rho_1_best(i,k) - eta*rho_2_best(i,k))/(rho_max(i,1) - rho_cr_best)) ]);
                    r_2_att_best(i,k) = min([ domanda_2(i,k) + l_2_best(i,k)/T; r_2_cap; r_2_cap*((rho_max(i,1) - rho_1_best(i,k) - eta*rho_2_best(i,k))/(rho_max(i,1) - rho_cr_best)) ]);
                end
                
                l_1_best(i,k + 1) = l_1_best(i,k) + T*(domanda_1(i,k) - r_1_att_best(i,k));
                l_2_best(i,k + 1) = l_2_best(i,k) + T*(domanda_2(i,k) - r_2_att_best(i,k));
                
                q_1_best(i,k) = rho_1_best(i,k)*lane(i)*v_1_best(i,k)*(1 - s_1(i,k));
                q_2_best(i,k) = rho_2_best(i,k)*lane(i)*v_2_best(i,k)*(1 - s_2(i,k));
                
                q_1_tot_best(i,k + 1) = rho_1_best(i,k)*lane(i)*v_1_best(i,k);
                q_2_tot_best(i,k + 1) = rho_2_best(i,k)*lane(i)*v_2_best(i,k);
                
                %% Traffic Density in [veh/km/lane]
                
                rho_1_best(i,k + 1) = rho_1_best(i,k) + (T/(Delta(i,1)*lane(i))) * (q_1_best(i - 1,k) - q_1_best(i,k)/(1 - s_1(i,k)) + r_1_att_best(i,k));
                rho_2_best(i,k + 1) = rho_2_best(i,k) + (T/(Delta(i,1)*lane(i))) * (q_2_best(i - 1,k) - q_2_best(i,k)/(1 - s_1(i,k)) + r_2_att_best(i,k));

                %% Fundamental diagram
                
                v_fund_1_best = vf_1_best*exp(( - 1/aexp1_best)*((rho_1_best(i,k) + eta*rho_2_best(i,k))/rho_cr_best)^aexp1_best);
                
                v_fund_2_best = vf_2_best*exp(( - 1/aexp2_best)*((rho_1_best(i,k) + eta*rho_2_best(i,k))/rho_cr_best)^aexp2_best);
                
                %        v_fund_1_best = vf_1_best*(1 - ((rho_1_best(i,k) + eta*rho_2_best(i,k))/rho_max(i,1))^l1_best)^m1_best);
                
                %        v_fund_2_best = vf_2_best*(1 - ((rho_1_best(i,k) + eta*rho_2_best(i,k))/rho_max(i,1))^l2_best)^m2_best);
                
                if (rho_1_best(i,k) + eta*rho_2_best(i,k))<0.001
                    v_fund_1_best = vf_1_best;
                end
                
                %% Speed Dynamics
                
                if i == 2
                    
                    v_1_best(i,k + 1) = v_1_best(i,k) + (T/tau_1_best)*(v_fund_1_best - v_1_best(i,k))...
                        - ((nu_1_best*T*(rho_1_best(i + 1,k) + eta*rho_2_best(i + 1,k) - rho_1_best(i,k) - eta*rho_2_best(i,k)))/(Delta(i,1)*tau_1_best*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1)));
                    
                    v_2_best(i,k + 1) = v_2_best(i,k) + (T/tau_2_best)*(v_fund_2_best  - v_2_best(i,k))...
                        - ((nu_2_best*T*(rho_1_best(i + 1,k) + eta*rho_2_best(i + 1,k) - rho_1_best(i,k) - eta*rho_2_best(i,k)))/(Delta(i,1)*tau_2_best*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2)));
                    
                    % Merge ramp term
                    if Ir(i) == 1
                        
                        v_1_best(i,k + 1) = v_1_best(i,k + 1) - ((delta_on_1_best*T*v_1_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1)));
                        
                        v_2_best(i,k + 1) = v_2_best(i,k + 1) - ((delta_on_2_best*T*v_2_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2)));
                        
                    end
                    % Lane drop term
                    
                    if lane(i) > lane(i + 1)
                        
                        v_1_best(i,k + 1) = v_1_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_1_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                        
                        v_2_best(i,k + 1) = v_2_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_2_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                        
                    end
                    
                    
                elseif i == N + 1
                    
                    
                    v_1_best(i,k + 1) = v_1_best(i,k) + (T/tau_1_best)*(v_fund_1_best - v_1_best(i,k))...
                        + (T/Delta(i,1))*v_1_best(i,k)*(v_1_best(i - 1,k) - v_1_best(i,k));
                    
                    v_2_best(i,k + 1) = v_2_best(i,k) + (T/tau_2_best)*(v_fund_2_best  - v_2_best(i,k))...
                        + (T/Delta(i,1))*v_2_best(i,k)*(v_2_best(i - 1,k) - v_2_best(i,k));
                    
                    % Merge ramp term
                    if Ir(i) == 1
                        
                        v_1_best(i,k + 1) = v_1_best(i,k + 1) - ((delta_on_1_best*T*v_1_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1)));
                        
                        v_2_best(i,k + 1) = v_2_best(i,k + 1) - ((delta_on_2_best*T*v_2_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2)));
                        
                    end
                    
                    if lane(i) > lane(i + 1)
                        
                        v_1_best(i,k + 1) = v_1_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_1_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                        
                        v_2_best(i,k + 1) = v_2_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_2_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                        
                    end
                    
                    
                    if (rho_1_best(i,k) + eta*rho_2_best(i,k)) > rho_cr_best
                        v_1_best(i,k + 1) = v_1_best(i,k + 1) - nu_1_best*T*(rho_cr_best - rho_1_best(i,k) + eta*rho_2_best(i,k))/(tau_1_best*Delta(i,1)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1));
                        
                        v_2_best(i,k + 1) = v_2_best(i,k + 1) - nu_2_best*T*(rho_cr_best - rho_1_best(i,k) + eta*rho_2_best(i,k))/(tau_2_best*Delta(i,1)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2));
                    end
                    
                    
                else
                    
                    v_1_best(i,k + 1) = v_1_best(i,k) + (T/tau_1_best)*(v_fund_1_best - v_1_best(i,k))...
                        + (T/Delta(i,1))*v_1_best(i,k)*(v_1_best(i - 1,k) - v_1_best(i,k))...
                        - ((nu_1_best*T*(rho_1_best(i + 1,k) + eta*rho_2_best(i + 1,k) - rho_1_best(i,k) - eta*rho_2_best(i,k)))/(Delta(i,1)*tau_1_best*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1)));
                    
                    v_2_best(i,k + 1) = v_2_best(i,k) + (T/tau_2_best)*(v_fund_2_best  - v_2_best(i,k))...
                        + (T/Delta(i,1))*v_2_best(i,k)*(v_2_best(i - 1,k) - v_2_best(i,k))...
                        - ((nu_2_best*T*(rho_1_best(i + 1,k) + eta*rho_2_best(i + 1,k) - rho_1_best(i,k) - eta*rho_2_best(i,k)))/(Delta(i,1)*tau_2_best*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2)));
                    
                    
                    % Merge ramp term
                    
                    if Ir(i) == 1
                        
                        v_1_best(i,k + 1) = v_1_best(i,k + 1) - ((delta_on_1_best*T*v_1_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_1)));
                        
                        v_2_best(i,k + 1) = v_2_best(i,k + 1)- ((delta_on_2_best*T*v_2_best(i,k)*(r_1_att_best(i,k) + eta*r_2_att_best(i,k)))/(Delta(i,1)*lane(i)*(rho_1_best(i,k) + eta*rho_2_best(i,k) + chi_2)));
                        
                    end
                    
                    % Lane drop term
                    
                    if lane(i) > lane(i + 1)
                        
                        v_1_best(i,k + 1) = v_1_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_1_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                        
                        
                        v_2_best(i,k + 1) = v_2_best(i,k + 1)  - ( phi_best*T*(lane(i) - lane(i + 1))*v_2_best(i,k)^2*(rho_1_best(i,k) + eta*rho_2_best(i,k)) )/(Delta(i,1)*lane(i)*rho_cr_best);
                        
                    end
                    
                    
                end
                
                v_1_best(i,k + 1) = max(v_min_1,v_1_best(i,k + 1));
                v_2_best(i,k + 1) = max(v_min_2,v_2_best(i,k + 1));
                
                
            end
        end
        
        J_best = J(1,2);
        
        J(1,1) = J(1,2);
        
        consecutiveIterationsNotImproving = 0;
        probAccettazione = probAccettazione_iniz;
        
    else
        
        consecutiveIterationsNotImproving = consecutiveIterationsNotImproving + 1;
        
        probAccettazione = probAccettazione*descentFactor;
        
    end
    
%     if ( consecutiveIterationsNotImproving > (1/5)*maxIterationsNotImproving )
%         
%         probAccettazione = probAccettazione_iniz;
%         
%     end
    
    %verifico il criterio di arresto
    
    if (j >= maxIterations || consecutiveIterationsNotImproving > maxIterationsNotImproving )
        break
    end
    
    
end

