m_tot=unique(round(logspace(0,3,100)));
L_tot=unique(round(logspace(0,3,100)));
m_indices = 1:length(m_tot);
L_indices = 1:length(L_tot);
m_tick_pos = 1:10:length(m_indices);  % Every 10th position
L_tick_pos = 1:10:length(L_indices);  % Every 10th position
figure;
%%
filename='..//thesis/lin_oscillator/data_lin_oscillator/data_mu0.txt';
[~, ocean, atmosphere] = importfile_atmos_model(filename);
%ocean=resample(ocean,1,10);
%atmosphere=resample(atmosphere,1,10);
TE_O_to_A=zeros(length(m_tot),length(L_tot));
TE_A_to_O=zeros(length(m_tot),length(L_tot));
for m=1:length(m_tot)
    for L=1:length(L_tot)
        d=ocean;t=atmosphere;[x, y, z] = extract_te_variables(d,t, m_tot(m), L_tot(L));TE_O_to_A(m,L) = gccmi_ccc(x, y, z);
        t=ocean;d=atmosphere;[x, y, z] = extract_te_variables(d,t, m_tot(m), L_tot(L));TE_A_to_O(m,L) = gccmi_ccc(x, y, z);
    end
end
%%

subplot(3,2,5);
pcolor(m_indices, L_indices, TE_O_to_A');xlim([1 100]);ylim([1 100]);
set(gca, 'XTick', m_tick_pos, 'XTickLabel', m_tot(m_tick_pos))
set(gca, 'YTick', L_tick_pos, 'YTickLabel', L_tot(L_tick_pos))
shading flat; colorbar;xlabel('history (log scale)');ylabel('lag (log scale)')
xlim([min(m_indices) max(m_indices)]);ylim([min(L_indices) max(L_indices)]);
title('mu = 0, O to A')
subplot(3,2,6);
pcolor(m_indices, L_indices, TE_A_to_O');xlim([1 100]);ylim([1 100]);
set(gca, 'XTick', m_tick_pos, 'XTickLabel', m_tot(m_tick_pos))
set(gca, 'YTick', L_tick_pos, 'YTickLabel', L_tot(L_tick_pos))
shading flat; colorbar;xlabel('history (log scale)');ylabel('lag (log scale)')
shading flat; colorbar;xlabel('history (log scale)');ylabel('lag (log scale)')
xlim([min(m_indices) max(m_indices)]);ylim([min(L_indices) max(L_indices)]);
title('mu = 0, A to O')

%%
filename='..//thesis/lin_oscillator/data_lin_oscillator/data_mu0.001.txt';
[~, ocean, atmosphere] = importfile_atmos_model(filename);
%ocean=resample(ocean,1,10);
%atmosphere=resample(atmosphere,1,10);
TE_O_to_A=zeros(length(m_tot),length(L_tot));
TE_A_to_O=zeros(length(m_tot),length(L_tot));
for m=1:length(m_tot)
    for L=1:length(L_tot)
        d=ocean;t=atmosphere;[x, y, z] = extract_te_variables(d,t, m_tot(m), L_tot(L));TE_O_to_A(m,L) = gccmi_ccc(x, y, z);
        t=ocean;d=atmosphere;[x, y, z] = extract_te_variables(d,t, m_tot(m), L_tot(L));TE_A_to_O(m,L) = gccmi_ccc(x, y, z);
    end
end
%%

subplot(3,2,3);
pcolor(m_indices, L_indices, TE_O_to_A');xlim([1 100]);ylim([1 100]);
set(gca, 'XTick', m_tick_pos, 'XTickLabel', m_tot(m_tick_pos))
set(gca, 'YTick', L_tick_pos, 'YTickLabel', L_tot(L_tick_pos))
shading flat; colorbar;xlabel('history (log scale)');ylabel('lag (log scale)')
xlim([min(m_indices) max(m_indices)]);ylim([min(L_indices) max(L_indices)]);
title('mu = 0.001, O to A')
subplot(3,2,4);
pcolor(m_indices, L_indices, TE_A_to_O');xlim([1 100]);ylim([1 100]);
set(gca, 'XTick', m_tick_pos, 'XTickLabel', m_tot(m_tick_pos))
set(gca, 'YTick', L_tick_pos, 'YTickLabel', L_tot(L_tick_pos))
shading flat; colorbar;xlabel('history (log scale)');ylabel('lag (log scale)')
shading flat; colorbar;xlabel('history (log scale)');ylabel('lag (log scale)')
xlim([min(m_indices) max(m_indices)]);ylim([min(L_indices) max(L_indices)]);
title('mu = 0.001, A to O')

%%

filename='..//thesis/lin_oscillator/data_lin_oscillator/data_mu1.txt';
[~, ocean, atmosphere] = importfile_atmos_model(filename);
%ocean=resample(ocean,1,10);
%atmosphere=resample(atmosphere,1,10);
TE_O_to_A=zeros(length(m_tot),length(L_tot));
TE_A_to_O=zeros(length(m_tot),length(L_tot));
for m=1:length(m_tot)
    for L=1:length(L_tot)
        d=ocean;t=atmosphere;[x, y, z] = extract_te_variables(d,t, m_tot(m), L_tot(L));TE_O_to_A(m,L) = gccmi_ccc(x, y, z);
        t=ocean;d=atmosphere;[x, y, z] = extract_te_variables(d,t, m_tot(m), L_tot(L));TE_A_to_O(m,L) = gccmi_ccc(x, y, z);
    end
end
%%

subplot(3,2,1);
pcolor(m_indices, L_indices, TE_O_to_A');xlim([1 100]);ylim([1 100]);
set(gca, 'XTick', m_tick_pos, 'XTickLabel', m_tot(m_tick_pos))
set(gca, 'YTick', L_tick_pos, 'YTickLabel', L_tot(L_tick_pos))
shading flat; colorbar;xlabel('history (log scale)');ylabel('lag (log scale)')
xlim([min(m_indices) max(m_indices)]);ylim([min(L_indices) max(L_indices)]);
title('mu = 1, O to A')
subplot(3,2,2);
pcolor(m_indices, L_indices, TE_A_to_O');xlim([1 100]);ylim([1 100]);
set(gca, 'XTick', m_tick_pos, 'XTickLabel', m_tot(m_tick_pos))
set(gca, 'YTick', L_tick_pos, 'YTickLabel', L_tot(L_tick_pos))
shading flat; colorbar;xlabel('history (log scale)');ylabel('lag (log scale)')
shading flat; colorbar;xlabel('history (log scale)');ylabel('lag (log scale)')
xlim([min(m_indices) max(m_indices)]);ylim([min(L_indices) max(L_indices)]);
title('mu = 1, A to O')
