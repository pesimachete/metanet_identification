clear all

%% TRAFFIC DATA  =============================================
load('Input_data.mat') %dati in ingresso disposti in colonna

%le rampe di ingresso sempre in fondo alla sezione

HR_iniz = 0; % ore inizio simulazione
M_iniz = 0;  % minuti inizio simulazione
S_iniz = 0;  % secondi inizio simulazione

HR_fin = 3;  % ore fine simulazione
M_fin = 59;  % minuti fine simulazione
S_fin = 0;   % secondi fine simulazione

Tm = 60;     % time interval misurazioni sec
Tsim = 5;%10;   % time interval simulazioni sec

Starting_time = HR_iniz*3600 + M_iniz*60 + S_iniz; % in sec
Ending_time = HR_fin*3600 + M_fin*60 + S_fin;  %in sec


lcON_ramp = length(On_ramp_flow(1,:));%find number of columns on-ramp flow
lcOFF_ramp = length(On_ramp_flow(1,:));%find number of columns exit-rate flow
lcBoundary = length(Boundary_mainstream(1,:));%find number of columns boundary condition
time_mis = Starting_time:Tm:Ending_time; %real time vector in sec
ltime_mis = length(time_mis); %length of the real time vector


ts = time_mis(1);
time_sim = ts;

for m = 1: (ltime_mis-1) %calculate simulation time vector
    step = 0;
    
    while (ts + Tsim) < time_mis(m+1)
        step = step + Tsim;
        ts = time_mis(m) + step;
        time_sim = [time_sim ts] ;
    end
    
    time_sim = [time_sim time_mis(m+1)];
    
end


xq = time_sim; %simulation time vector
lxq = length(xq); %length of the simulation time vector


for k = 1:lcON_ramp
    interp_column = interp1(time_mis,On_ramp_flow(:,k),xq);
    Array_on_ramp(:,k) = interp_column;
end


On_ramp_flow_Interp = Array_on_ramp;

for k = 1:lcOFF_ramp
    interp_column = interp1(time_mis,Off_ramp_rate(:,k),xq);
    Array_off_ramp(:,k) = interp_column;
end


Off_ramp_rate_Interp = Array_off_ramp;


for k = 1:lcBoundary
    interp_column = interp1(time_mis,Boundary_mainstream(:,k),xq);
    Array_Boundary(:,k) = interp_column;
end


Boundary_mainstream_Interp = Array_Boundary; %flussi e velocità sez 0 e flussi e velocità sez N+1

%% Calculate time position of estimations for the PI calculation

START_TIME = 1;
END_TIME = lxq; %total simulation steps


extra_points = ceil(Tm/Tsim - 1);
time_position = START_TIME:(extra_points + 1):END_TIME;



%% SIMULATED ANEALING  =============================================

consecutiveIterationsNotImproving = 0;
C = 5000;		%% provo a farla adattiva
probAccettazione_iniz = 0.6;%0.7; %% alla prima iterazione
probAccettazione = 0.6;%0.7;
percPeggiorazioneAccettazione = 0.01;
descentFactor = 0.9995;
maxIterations = 20000;
maxIterationsNotImproving = 2000;
radius = 0.008; %%0.01

param = 7; %numero di parametri da calibrare tau, nu, delta_on, l, m, v_free, rho_cr (aggiungere phi se ho lane closure)

Model_parameter1_tentative = zeros(param,1);
Model_parameter2_tentative = zeros(param - 1,1); %considero unica la rho_cr

Model_parameter1 = zeros(param,1);
Model_parameter2 = zeros(param - 1,1); %considero unica la rho_cr

Model_parameter1_best = zeros(param,1);
Model_parameter2_best = zeros(param - 1,1); %considero unica la rho_cr

% soluzione iniziale

Model_parameter1(1) = 10.2406890760597; %tau
Model_parameter2(1) = 1;

Model_parameter1(2) = 8.96781852567262; %nu
Model_parameter2(2) = 1;

Model_parameter1(3) = 2.92901743328448; %delta_on
Model_parameter2(3) = 1;

Model_parameter1(4) = 1.61817436122415; %aexp
Model_parameter2(4) = 1;

Model_parameter1(5) = 0.000134644888077544; %phi
Model_parameter2(5) = 1;

% Model_parameter1(4) = 3; %l
% Model_parameter2(4) = 1;
% 
% Model_parameter1(5) = 4; %m
% Model_parameter2(5) = 1;

Model_parameter1(6) = 117.008676802763; %vfre
Model_parameter2(6) = 100;

Model_parameter1(7) = 36.6927640563290; %rho_cr
% Model_parameter2(7) = 50; %per lane




Model_parameter1_best(1) = 10.2406890760597; %tau
Model_parameter2_best(1) = 1;

Model_parameter1_best(2) = 8.96781852567262; %nu
Model_parameter2_best(2) = 1;

Model_parameter1_best(3) = 2.92901743328448; %delta_on
Model_parameter2_best(3) = 1;

Model_parameter1_best(4) = 1.61817436122415; %aexp
Model_parameter2_best(4) = 1;

Model_parameter1_best(5) = 0.000134644888077544; %phi
Model_parameter2_best(5) = 1;

% Model_parameter1_best(4) = 3; %l
% Model_parameter2_best(4) = 1;
% 
% Model_parameter1_best(5) = 4; %m
% Model_parameter2_best(5) = 1;

Model_parameter1_best(6) = 117.008676802763; %vfre
Model_parameter2_best(6) = 100;

Model_parameter1_best(7) = 36.6927640563290; %rho_cr
% Model_parameter2_best(7) = 50; %per lane
% 

lb1 = zeros(param,1);
ub1 = zeros(param,1);
lb2 = zeros(param,1);
ub2 = zeros(param,1);

for i=1:param
    
    lb1(i) = 0.1.*Model_parameter1(i);
    ub1(i) = 2.5*Model_parameter1(i);
    
end

for i=1:param -1
    lb2(i) = 0.5.*Model_parameter2(i);
    ub2(i) = 1.5*Model_parameter2(i);
end
J_tentative = 0.0;

w_speed = 1;
w_flow = 0;

%% METANET  =============================================


N = 21;					%% Numero di sezioni
K = lxq;
T = Tsim / 3600;	%% Lunghezza intervallo di discretizzazione in [h]

Real_Mainstream_flow = zeros(N,ltime_mis);
Real_Mainstream_speed = zeros(N,ltime_mis);
Real_Mainstream_density = zeros(N,ltime_mis);

for time = 1 : ltime_mis
    for sez = 1 : N
        Real_Mainstream_flow(sez,time) = Real_flow(time,sez);
        Real_Mainstream_speed(sez,time) = Real_speed(time,sez);
        Real_Mainstream_density(sez,time) = Real_Mainstream_flow(sez,time) / Real_Mainstream_speed(sez,time);
    end
end

Det = 21; %number of detector

Det_position = zeros(Det,1);

for i = 1 : N
    Det_position(i) = i + 1;
end

Ir = zeros(N + 2,1); % posizione rampe ingresso

Ir(1) = 0; %sezione 0
Ir(11) = 1;
Ir(17) = 1;

Or = zeros(N + 2,1); % posizione rampe uscita

Or(9) = 1;
Or(15) = 1;

Delta = zeros(N + 2,1);

% for i = 1: N + 2
%     Delta(i,1) = 500/1000; % Lunghezza di una sezione 500 m (uguale per tutte le sezioni per il momento)
% end

Delta(1,1) = 500/1000; Delta(N + 2,1) = 500/1000; %sezione 0 e N+2
Delta(2,1) = 433/1000; Delta(3,1) = 437/1000; Delta(4,1) = 444/1000; Delta(5,1) = 441/1000; Delta(6,1) = 463/1000; Delta(7,1) = 450/1000; Delta(8,1) = 450/1000;
Delta(9,1) = 448/1000; Delta(10,1) = 306/1000; Delta(11,1) = 340/1000; Delta(12,1) = 350/1000; Delta(13,1) = 424/1000; Delta(14,1) = 566/1000; Delta(15,1) = 706/1000;
Delta(16,1) = 658/1000; Delta(17,1) = 424/1000; Delta(18,1) = 419/1000; Delta(19,1) = 432/1000; Delta(20,1) = 413/1000; Delta(21,1) = 427/1000; Delta(22,1) = 426/1000;

lane = zeros(N+2,1); % numero di lane per sezione

for i = 1: N + 2
    if i <= 9
        lane(i,1) = 3;
    else
        lane(i,1) = 2;
    end
end

eta = 4;       %% Parametro di conversione trucks->cars(L)

chi_1 = 10;  %40
chi_2 = 10;  %40

v_min_1 = 7;
v_min_2 = 7;

tau_1 = Model_parameter1(1)/3600;
tau_2 = Model_parameter2(1)/3600;

nu_1 = Model_parameter1(2);
nu_2 = Model_parameter2(2);

delta_on_1 = Model_parameter1(3);
delta_on_2 = Model_parameter2(3);

aexp1 = Model_parameter1(4);
aexp2 = Model_parameter2(4);

phi = Model_parameter1(5);

% l1 = Model_parameter1(4);
% l2 = Model_parameter2(4);
% 
% m1 = Model_parameter1(5);
% m2 = Model_parameter2(5);

vf_1 = Model_parameter1(6);
vf_2 = Model_parameter2(6);

rho_cr = Model_parameter1(7);

tau_1_best = Model_parameter1_best(1)/3600;
tau_2_best = Model_parameter2_best(1)/3600;

nu_1_best = Model_parameter1_best(2);
nu_2_best = Model_parameter2_best(2);

delta_on_1_best = Model_parameter1_best(3);
delta_on_2_best = Model_parameter2_best(3);

aexp1_best = Model_parameter1_best(4);
aexp2_best = Model_parameter2_best(4);

phi_best = Model_parameter1_best(4);

% l1_best = Model_parameter1_best(4);
% l2_best = Model_parameter2_best(4);
% 
% m1_best = Model_parameter1_best(5);
% m2_best = Model_parameter2_best(5);

vf_1_best = Model_parameter1_best(6);
vf_2_best = Model_parameter2_best(6);

%phi = 0;

rho_cr_best = Model_parameter1_best(7);


tau_1_tent = 0;
tau_2_tent = 0;

nu_1_tent = 0;
nu_2_tent = 0;

delta_on_1_tent = 0;
delta_on_2_tent = 0;

aexp1_tent = 0;
aexp2_tent = 0;

phi_tent = 0;

% l1_tent = 0;
% l2_tent = 0;
% 
% m1_tent = 0;
% m2_tent = 0;

vf_1_tent = 0;
vf_2_tent = 0;

rho_cr_tent = 0;

% rho_cr = Model_parameter2(7);

% phi_1 = 0;
% phi_2 = 0;

rho_max = zeros(N + 2,1);

for i = 1: N + 2
    rho_max(i) = 200; %per lane
end


l_1_max = zeros(N + 2,1);
for i = 1 : N + 2
    if Ir(i) == 1
        l_1_max(i,1) = 1000;
    end
end

l_2_max = zeros(N + 2,1);
for i = 1 : N + 2
    if Ir(i) == 1
        l_2_max(i,1)=1000;
    end
end


r_1_cap = 2500;
r_2_cap = 500;


%% Domanda
domanda_1 = zeros(N + 2, K + 1);
domanda_2 = zeros(N + 2, K + 1);

s_1 = zeros(N + 2,K + 1);
s_2 = zeros(N + 2,K + 1);

for time = 1 : K
    domanda_1(11,time) = On_ramp_flow_Interp(time,1);
    domanda_1(17,time) = On_ramp_flow_Interp(time,2);
    s_1(9,time) = Off_ramp_rate_Interp(time,1);
    s_1(15,time) = Off_ramp_rate_Interp(time,2);
end


%Condizioni iniziali

rho_1_iniz = zeros(N,1);
rho_2_iniz = zeros(N,1);

v_1_iniz = zeros(N,1);
v_2_iniz = zeros(N,1);

q_1_iniz = zeros(N,1);
q_2_iniz = zeros(N,1);

v_1_iniz(:,1) = Initial_mainstream_speed(1,:);
q_1_iniz(:,1) = Initial_mainstream_flow(1,:);

for sez = 1 : N
    rho_1_iniz(sez,1) = q_1_iniz(sez,1)/(v_1_iniz(sez,1)*lane(sez));
end
%% Condizioni al contorno sez 0
rho_1_sez0 = zeros(K,1);
rho_2_sez0 = zeros(K,1);

v_1_sez0 = zeros(K,1);
v_2_sez0 = zeros(K,1);

q_1_sez0 = zeros(K,1);
q_2_sez0 = zeros(K,1);

q_1_sez0(:,1) = Boundary_mainstream_Interp(:,1);
v_1_sez0(:,1) = Boundary_mainstream_Interp(:,2);

for time = 1 : K
    rho_1_sez0(time,1) = q_1_sez0(time,1)/(v_1_sez0(time,1)*lane(1));
end
%% condizioni al contorno sez N+1

rho_1_sezfin = zeros(K,1);
rho_2_sezfin = zeros(K,1);
v_1_sezfin = zeros(K,1);
v_2_sezfin = zeros(K,1);
q_1_sezfin = zeros(K,1);
q_2_sezfin = zeros(K,1);


q_1_sezfin(:,1) = Boundary_mainstream_Interp(:,3);
v_1_sezfin(:,1) = Boundary_mainstream_Interp(:,4);

for time = 1 : K
    rho_1_sezfin(time,1) = q_1_sezfin(time,1)/(v_1_sezfin(time,1)*lane(N+2));
end

%% definisco le variabili con dimensione N + 2 e K + 1
%%(la prima riga corrisponde alla sezione 0, l'ultima riga alla sezione N+1; la prima colonna
% corrisponde all'istante 0).
rho_1 = zeros(N+2,K+1);
rho_2 = zeros(N+2,K+1);

q_1_tot = zeros(N+2,K+1);
q_2_tot = zeros(N+2,K+1);

q_1 = zeros(N+2,K+1);
q_2 = zeros(N+2,K+1);

v_1 = zeros(N+2,K+1);
v_2 = zeros(N+2,K+1);

l_1 = zeros(N+2,K+1);
l_2 = zeros(N+2,K+1);

%% calcolo funzionale di costo
Model_speed1 = zeros(Det,ltime_mis);
Model_flow1 = zeros(Det,ltime_mis);
Model_density1 = zeros(Det,ltime_mis);

Error_speed_1 = zeros(Det,ltime_mis);
Error_flow_1 = zeros(Det,ltime_mis);

Model_speed2 = zeros(Det,ltime_mis);
Model_flow2 = zeros(Det,ltime_mis);
Model_density2 = zeros(Det,ltime_mis);

Error_speed_2 = zeros(Det,ltime_mis);
Error_flow_2 = zeros(Det,ltime_mis);

Model_speed1_best = zeros(Det,ltime_mis);
Model_flow1_best = zeros(Det,ltime_mis);
Model_density1_best = zeros(Det,ltime_mis);

Error_speed_1_best = zeros(Det,ltime_mis);
Error_flow_1_best = zeros(Det,ltime_mis);

Model_speed2_best = zeros(Det,ltime_mis);
Model_flow2_best = zeros(Det,ltime_mis);
Model_density2_best = zeros(Det,ltime_mis);

Error_speed_2_best = zeros(Det,ltime_mis);
Error_flow_2_best = zeros(Det,ltime_mis);

Model_speed1_tentative = zeros(Det,ltime_mis);
Model_flow1_tentative = zeros(Det,ltime_mis);
Model_density1_tentative = zeros(Det,ltime_mis);

Error_speed_1_tentative = zeros(Det,ltime_mis);
Error_flow_1_tentative = zeros(Det,ltime_mis);

Model_speed2_tentative = zeros(Det,ltime_mis);
Model_flow2_tentative = zeros(Det,ltime_mis);
Model_density2_tentative = zeros(Det,ltime_mis);

Error_speed_2_tentative = zeros(Det,ltime_mis);
Error_flow_2_tentative = zeros(Det,ltime_mis);


rho_1_best = zeros(N+2,K+1);
rho_2_best = zeros(N+2,K+1);

v_1_best = zeros(N+2,K+1);
v_2_best = zeros(N+2,K+1);

q_1_tot_best = zeros(N+2,K+1);
q_2_tot_best = zeros(N+2,K+1);

q_1_best = zeros(N+2,K+1);
q_2_best = zeros(N+2,K+1);

l_1_best = zeros(N+2,K+1);
l_2_best = zeros(N+2,K+1);

rho_1_tentative= zeros(N+2,K+1);
rho_2_tentative = zeros(N+2,K+1);

v_1_tentative = zeros(N+2,K+1);
v_2_tentative = zeros(N+2,K+1);

q_1_tot_tentative = zeros(N+2,K+1);
q_2_tot_tentative = zeros(N+2,K+1);

l_1_tentative = zeros(N+2,K+1);
l_2_tentative = zeros(N+2,K+1);

q_1_tentative = zeros(N+2,K+1);
q_2_tentative = zeros(N+2,K+1);

%%condizioni iniziali

rho_1(2:N+1,1) = rho_1_iniz(:,:);
rho_2(2:N+1,1) = rho_2_iniz(:,:);
v_1(2:N+1,1) = v_1_iniz(:,:);
v_2(2:N+1,1) = v_2_iniz(:,:);
q_1_tot(2:N+1,1) = q_1_iniz(:,:);
q_2_tot(2:N+1,1) = q_2_iniz(:,:);
q_1(2:N+1,1) = q_1_iniz(:,:);
q_2(2:N+1,1) = q_2_iniz(:,:);

rho_1_tentative(2:N+1,1) = rho_1_iniz(:,:);
rho_2_tentative(2:N+1,1) = rho_2_iniz(:,:);
v_1_tentative(2:N+1,1) = v_1_iniz(:,:);
v_2_tentative(2:N+1,1) = v_2_iniz(:,:);
q_1_tot_tentative(2:N+1,1) = q_1_iniz(:,:);
q_2_tot_tentative(2:N+1,1) = q_2_iniz(:,:);
q_1_tentative(2:N+1,1) = q_1_iniz(:,:);
q_2_tentative(2:N+1,1) = q_2_iniz(:,:);

rho_1_best(2:N+1,1) = rho_1_iniz(:,:);
rho_2_best(2:N+1,1) = rho_2_iniz(:,:);
v_1_best(2:N+1,1) = v_1_iniz(:,:);
v_2_best(2:N+1,1) = v_2_iniz(:,:);
q_1_tot_best(2:N+1,1) = q_1_iniz(:,:);
q_2_tot_best(2:N+1,1) = q_2_iniz(:,:);
q_1_best(2:N+1,1) = q_1_iniz(:,:);
q_2_best(2:N+1,1) = q_2_iniz(:,:);

% condizioni sulla sezione 0

rho_1(1,1:K) = rho_1_sez0(:,:);
rho_2(1,1:K) = rho_2_sez0(:,:);
v_1(1,1:K) = v_1_sez0(:,:);
v_2(1,1:K) = v_2_sez0(:,:);
q_1_tot(1,1:K) = q_1_sez0(:,:);
q_2_tot(1,1:K) = q_2_sez0(:,:);
q_1(1,1:K) = q_1_sez0(:,:);
q_2(1,1:K) = q_2_sez0(:,:);

rho_1_tentative(1,1:K) = rho_1_sez0(:,:);
rho_2_tentative(1,1:K) = rho_2_sez0(:,:);
v_1_tentative(1,1:K) = v_1_sez0(:,:);
v_2_tentative(1,1:K) = v_2_sez0(:,:);
q_1_tot_tentative(1,1:K) = q_1_sez0(:,:);
q_2_tot_tentative(1,1:K) = q_2_sez0(:,:);
q_1_tentative(1,1:K) = q_1_sez0(:,:);
q_2_tentative(1,1:K) = q_2_sez0(:,:);

rho_1_best(1,1:K) = rho_1_sez0(:,:);
rho_2_best(1,1:K) = rho_2_sez0(:,:);
v_1_best(1,1:K) = v_1_sez0(:,:);
v_2_best(1,1:K) = v_2_sez0(:,:);
q_1_tot_best(1,1:K) = q_1_sez0(:,:);
q_2_tot_best(1,1:K) = q_2_sez0(:,:);
q_1_best(1,1:K) = q_1_sez0(:,:);
q_2_best(1,1:K) = q_2_sez0(:,:);

%condizioni sulla sezione N+1

rho_1(N+2,1:K) = rho_1_sezfin(:,:);
rho_2(N+2,1:K) = rho_2_sezfin(:,:);
v_1(N+2,1:K) = v_1_sezfin(:,:);
v_2(N+2,1:K) = v_2_sezfin(:,:);
q_1_tot(N+2,1:K) = q_1_sezfin(:,:);
q_2_tot(N+2,1:K) = q_2_sezfin(:,:);
q_1(N+2,1:K) = q_1_sezfin(:,:);
q_2(N+2,1:K) = q_2_sezfin(:,:);

rho_1_tentative(N+2,1:K) = rho_1_sezfin(:,:);
rho_2_tentative(N+2,1:K) = rho_2_sezfin(:,:);
v_1_tentative(N+2,1:K) = v_1_sezfin(:,:);
v_2_tentative(N+2,1:K) = v_2_sezfin(:,:);
q_1_tot_tentative(N+2,1:K) = q_1_sezfin(:,:);
q_2_tot_tentative(N+2,1:K) = q_2_sezfin(:,:);
q_1_tentative(N+2,1:K) = q_1_sezfin(:,:);
q_2_tentative(N+2,1:K) = q_2_sezfin(:,:);

rho_1_best(N+2,1:K) = rho_1_sezfin(:,:);
rho_2_best(N+2,1:K) = rho_2_sezfin(:,:);
v_1_best(N+2,1:K) = v_1_sezfin(:,:);
v_2_best(N+2,1:K) = v_2_sezfin(:,:);
q_1_tot_best(N+2,1:K) = q_1_sezfin(:,:);
q_2_tot_best(N+2,1:K) = q_2_sezfin(:,:);
q_1_best(N+2,1:K) = q_1_sezfin(:,:);
q_2_best(N+2,1:K) = q_2_sezfin(:,:);

%%
r_1_att_tentative=zeros(N+1,K+1);
r_2_att_tentative=zeros(N+1,K+1);

r_1_att = zeros(N+1,K+1);
r_2_att = zeros(N+1,K+1);

r_1_att_best=zeros(N+1,K+1);
r_2_att_best=zeros(N+1,K+1);

