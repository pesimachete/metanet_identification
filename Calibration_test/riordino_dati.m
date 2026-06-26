
sez = zeros(1,Det);
sez(1,1) = Det;

for i = 2 : Det
    sez(1,i) = sez(1,i-1) - 1;
end

% rho_sim = zeros(N,K); 
% v_sim = zeros(N,K);
% q_sim = zeros(N,K);

rho_sim = zeros(N,ltime_mis); 
v_sim = zeros(N,ltime_mis);
q_sim = zeros(N,ltime_mis);

rho_real = zeros(Det,ltime_mis); 
v_real = zeros(Det,ltime_mis);
q_real = zeros(Det,ltime_mis);



for k = 1:ltime_mis
 for i = 1:Det
    v_real(i,k) = Real_speed(k,sez(1,i));
  end
end

figure;
pcolor(v_real(:,:));
shading flat
caxis([0,128]);
title 'Real speed'

for k = 1:ltime_mis
    for i = 1:Det
       q_real(i,k) = Real_flow(k,sez(1,i));
    end
end

figure;
pcolor(q_real(:,:));
shading flat
caxis([0,6000]);
title 'Real flow'

for k=1:ltime_mis
    for i=1:Det
       rho_real(i,k)=q_real(i,k)/v_real(i,k);
    end
end

figure;
pcolor(rho_real(:,:));
shading flat
caxis([0,150]);
title 'Real density'

for k=1:ltime_mis
    for i=1:N
       rho_sim(i,k) = Model_density1_best(sez(1,i),k);
    end
end

for k=1:ltime_mis
    for i=1:N
       v_sim(i,k) = Model_speed1_best(sez(1,i),k);
    end
end

for k=1:ltime_mis
    for i=1:N
       q_sim(i,k) = Model_flow1_best(sez(1,i),k);
    end
end

% for k=1:K
%     for i=1:N
%        rho_sim(i,k) = rho_1_best(sez(1,i),k);
%     end
% end
% 
% for k=1:K
%     for i=1:N
%        v_sim(i,k) = v_1_best(sez(1,i),k);
%     end
% end
% 
% for k=1:K
%     for i=1:N
%        q_sim(i,k) = q_1_best(sez(1,i),k);
%     end
% end

figure;
pcolor(v_sim(:,:));
shading flat
caxis([0,128]);
title 'Sim speed'

figure;
pcolor(rho_sim(:,:)*3);
shading flat
caxis([0,150]);
title 'Sim density'

figure;
pcolor(q_sim(:,:));
shading flat
caxis([0,6000]);
title 'Sim flow'



