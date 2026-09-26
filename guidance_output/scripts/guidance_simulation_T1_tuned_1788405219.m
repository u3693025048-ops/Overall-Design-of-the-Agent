function guidance_simulation_T1_tuned_1788405219()
% 单文件蒙特卡洛仿真 | 修改 RUN_CASE 选择工况
% 'D'设计任务(D1-D3)  'T'目标机动(T1-T4)  'G'交战几何(G1-G5)  'AP'驾驶仪(AP1-AP5)  'R'综合鲁棒(R1-R3)
clc; close all; rng(42);
RUN_CASE='T'; SUB_IDX=1; N_MC=100; KILL_R=10;
% RUN_CASE: 'D'|'T'|'G'|'AP'|'R' 单类别, 'ALL' 全部类别
% SUB_IDX:  0=全部子工况, 1=仅第1子工况, 2=仅第2子工况, ...

%% ── RL_PARAMS_BEGIN (auto-patched by RL optimizer, do not edit manually) ──
rl_w1          = 52.000000;
rl_zeta1       = 0.650000;
rl_tao1        = 0.150000;
rl_w2          = 46.000000;
rl_zeta2       = 0.650000;
rl_tao2        = 0.150000;
rl_w3          = 52.000000;
rl_zeta3       = 0.600000;
rl_N_pn        = 6.000000;
rl_ny_lim      = 20;
rl_gama_max_deg = 45;
%% ── RL_PARAMS_END ──

% 弹体参数: m=231kg  V=4Ma@9km=1200m/s  弹径d=0.203m  弹长L=3.66m
% S=pi/4*0.203^2=0.03236m^2  Jx≈3.0kg·m.^2(空心弹体)  Jy=Jz≈m*L^2/12≈250kg·m.^2
base=struct('x0',0,'y0',9000,'z0',-3000,'V0',1200,...
    'theta0',0,'phiv0',0,'upsilon0',0,'gama0',0,'phi0',0,...
    'Jx',3.0,'Jy',220,'Jz',220,'m',231,'S',0.03236,'L',3.66,'g',9.8,'P',5000,...
    'T_x0',28000,'T_y0',3000,'T_z0',10000,'T_Vx',250,'T_Vy',50,...
    'T_Vz_amp',80,'T_Vz_freq',0.2,'N_pn',rl_N_pn,...
    'CLA_s',1,'CD0_s',1,'mza_s',1,'CZB_s',1,...
    'w1',rl_w1,'zeta1',rl_zeta1,'tao1',rl_tao1,...
    'w2',rl_w2,'zeta2',rl_zeta2,'tao2',rl_tao2,...
    'w3',rl_w3,'zeta3',rl_zeta3,'gama_max_deg',rl_gama_max_deg);

if strcmpi(RUN_CASE,'ALL')
    ALL_SUMM={};
    for rc_={'D','T','G','AP','R'}
        s=run_category(rc_{1},SUB_IDX,N_MC,KILL_R,base);
        ALL_SUMM=[ALL_SUMM, s];
    end
    print_table(ALL_SUMM,N_MC,KILL_R,'D/T/G/AP/R 全工况');
    return
end
s=run_category(RUN_CASE,SUB_IDX,N_MC,KILL_R,base);
print_table(s,N_MC,KILL_R,RUN_CASE);
end % main

function SUMM=run_category(RUN_CASE,SUB_IDX,N_MC,KILL_R,base)
switch upper(RUN_CASE)
  case 'D'  % 三种设计任务场景:不包含目标机动,也不改变驾驶仪常数
    D2R=pi/180;
    SC={struct('name','D1:近距小偏置-标称弹体任务', ...
        'T_x0',12000,'T_y0',8500,'T_z0', 1000,'y0',9000,'z0',-3000, ...
        'T_Vy',0,'T_Vz_amp',0,'T_Vz_freq',0.20, ...
        'CD0_s',1.00,'CLA_s',1.00,'CZB_s',1.00,'mza_s',1.00, ...
        'm',231,'P',5000,'V0',1200, ...
        'theta0',0,'phi0',0,'upsilon0',0,'gama0',0),...
        struct('name','D2:远距大偏置-复杂几何任务', ...
        'T_x0',30000,'T_y0',5000,'T_z0',14000,'y0',9500,'z0',-5000, ...
        'T_Vy',0,'T_Vz_amp',0,'T_Vz_freq',0.20, ...
        'CD0_s',1.00,'CLA_s',1.00,'CZB_s',1.00,'mza_s',1.00, ...
        'm',231,'P',5000,'V0',1200, ...
        'theta0',0,'phi0',0,'upsilon0',0,'gama0',0),...
        struct('name','D3:能量受限-气动姿态强扰动任务', ...
        'T_x0',22000,'T_y0',6500,'T_z0', 9000,'y0',9000,'z0',-3000, ...
        'T_Vy',0,'T_Vz_amp',0,'T_Vz_freq',0.20, ...
        'CD0_s',1.22,'CLA_s',0.70,'CZB_s',1.22,'mza_s',0.65, ...
        'm',275,'P',3600,'V0',980, ...
        'theta0',-18*D2R,'phi0',14*D2R,'upsilon0',-14*D2R,'gama0',-10*D2R)};
    sf={'x0',800;'y0',700;'z0',900; ...
        'T_x0',1800;'T_y0',700;'T_z0',1200; ...
        'CD0_s',0.10;'CLA_s',0.10;'CZB_s',0.10;'mza_s',0.10; ...
        'm',16;'P',350;'V0',60; ...
        'theta0',5*D2R;'phi0',5*D2R;'upsilon0',5*D2R;'gama0',5*D2R};
  case 'T'  % 目标机动
    SC={struct('name','T1:匀速直飞',            'T_Vz_amp',  0,'T_Vy', 0,'T_Vz_freq',0.20),...
        struct('name','T2:低频小幅度机动',       'T_Vz_amp', 25,'T_Vy',12,'T_Vz_freq',0.40),...
        struct('name','T3:中频中等幅度机动',     'T_Vz_amp', 55,'T_Vy',28,'T_Vz_freq',0.75),...
        struct('name','T4:高频较大幅度机动',     'T_Vz_amp', 85,'T_Vy',45,'T_Vz_freq',1.20)};
    sf={'T_Vz_amp',20;'T_Vz_freq',0.20;'T_Vx',20;'T_Vy',15};
  case 'G'  % 交战几何
    SC={struct('name','G1:标称几何≈21km',  'T_x0',15000,'T_y0',4000,'T_z0', 7000,'y0', 9000,'z0',-2000),...
        struct('name','G2:近距小偏置',        'T_x0',10000,'T_y0',3500,'T_z0', 6000,'y0', 5500,'z0', 1500),...
        struct('name','G3:近距中等偏置',      'T_x0', 8500,'T_y0',2000,'T_z0', 6500,'y0', 4000,'z0', 2500),...
        struct('name','G4:远距小偏置',        'T_x0',28000,'T_y0',5000,'T_z0',10000,'y0', 8000,'z0', 5000),...
        struct('name','G5:远距大偏置',        'T_x0',23000,'T_y0',2000,'T_z0',14000,'y0',10000,'z0',-3000)};
    sf={'x0',1200;'y0',1200;'z0',1200;'T_x0',2500;'T_y0',800;'T_z0',1500};
  case 'AP'  % 驾驶仪退化 (标称w1=40 rad/s 匹配4Ma弹体动力学wm≈25rad/s)
    SC={struct('name','AP1:全标称驾驶仪(RL基准)',      'w1',40.0,'zeta1',0.75,'tao1',0.20,'w2',35.0,'zeta2',0.75,'tao2',0.20,'w3',40.0,'zeta3',0.70),...
        struct('name','AP2:通道时间常数单独退化',        'w1',40.0,'zeta1',0.75,'tao1',0.30,'w2',35.0,'zeta2',0.75,'tao2',0.30,'w3',40.0,'zeta3',0.70),...
        struct('name','AP3:驾驶仪轻度退化+tao轻度退化',   'w1',34.0,'zeta1',0.62,'tao1',0.24,'w2',29.8,'zeta2',0.62,'tao2',0.24,'w3',34.0,'zeta3',0.58),...
        struct('name','AP4:驾驶仪中度退化+tao中度退化',   'w1',26.0,'zeta1',0.50,'tao1',0.28,'w2',22.8,'zeta2',0.50,'tao2',0.28,'w3',26.0,'zeta3',0.45),...
        struct('name','AP5:驾驶仪重度退化+tao重度退化',   'w1',19.0,'zeta1',0.40,'tao1',0.32,'w2',16.6,'zeta2',0.40,'tao2',0.32,'w3',19.0,'zeta3',0.35)};
    sf={'w1',5.0;'zeta1',0.08;'w2',4.4;'zeta2',0.08};
  case 'R'  % 综合鲁棒性
    D2R=pi/180;
    SC={struct('name','R1:全标称基准',       'CD0_s',1.00,'CLA_s',1.00,'CZB_s',1.00,'mza_s',1.00,...
                                     'm',231,'P',5000,'V0',1200,'theta0',0,'phi0',0,'upsilon0',0,'gama0',0),...
        struct('name','R2:多参数综合摄动',    'CD0_s',1.18,'CLA_s',0.72,'CZB_s',1.18,'mza_s',0.68,...
                                     'm',270,'P',3800,'V0',1020,'theta0',0,'phi0',0,'upsilon0',0,'gama0',0),...
        struct('name','R3:大初始姿态偏差',    'CD0_s',1.00,'CLA_s',1.00,'CZB_s',1.00,'mza_s',1.00,...
                                     'm',231,'P',5000,'V0',1200,...
                                     'theta0',-20*D2R,'phi0',15*D2R,'upsilon0',-15*D2R,'gama0',-12*D2R)};
    sf={'CD0_s',0.12;'CLA_s',0.12;'CZB_s',0.12;'mza_s',0.12;...
        'm',18;'P',450;'V0',70;...
        'theta0',7*D2R;'phi0',7*D2R;'upsilon0',7*D2R;'gama0',7*D2R};
end

if SUB_IDX>0 && SUB_IDX<=numel(SC); SC=SC(SUB_IDX); end
nsc=numel(SC); RES=cell(nsc,1);
for sc=1:nsc
    fprintf('\n══════ 子工况 %s ══════\n',SC{sc}.name);
    fprintf('  [Run] Hit  Miss(m)  PeakNy(g)  PM(°)   BW(rad/s)  GM(dB)\n');
    res=struct('miss',nan(N_MC,1),'peak_ny',nan(N_MC,1),...
               'pitch_pm',nan(N_MC,1),'pitch_bw',nan(N_MC,1),'pitch_gm',nan(N_MC,1));
    p0=mst(base,SC{sc});
    for k=1:N_MC
        p=anoise(p0,sf);
        try
            r=sim_s(p);
            res.miss(k)=r.miss; res.peak_ny(k)=r.peak_ny;
            res.pitch_pm(k)=r.pitch_pm; res.pitch_bw(k)=r.bw_pitch; res.pitch_gm(k)=r.gm_pitch;
            hit=res.miss(k)<KILL_R;
            fprintf('  [%3d] %-4s %7.2f   %6.2f  %7.1f  %8.2f  %7.2f\n',...
                k,tf2str(hit),res.miss(k),res.peak_ny(k),res.pitch_pm(k),res.pitch_bw(k),res.pitch_gm(k));
        catch ME
            fprintf('  [%3d] ERR: %s\n',k,ME.message);
        end
    end
    RES{sc}=res;
    print_summary(res,KILL_R,SC{sc}.name);
end
plot_results(RES,SC,KILL_R,RUN_CASE);
% 构建返回的汇总数据
SUMM={};
for sc=1:nsc
    mv=RES{sc}.miss(~isnan(RES{sc}.miss));
    SUMM{sc}=struct('name',SC{sc}.name,...
        'hr', sum(mv<KILL_R)/numel(mv)*100,...
        'sep',quantile(mv,0.5),...
        'ny', mean(RES{sc}.peak_ny(~isnan(RES{sc}.peak_ny))),...
        'pm', mean(RES{sc}.pitch_pm(~isnan(RES{sc}.pitch_pm))),...
        'bw', mean(RES{sc}.pitch_bw(~isnan(RES{sc}.pitch_bw))),...
        'gm', mean(RES{sc}.pitch_gm(~isnan(RES{sc}.pitch_gm))));
end
end % run_category

function print_table(SUMM,N_MC,KILL_R,label)
fprintf('\n╔════════════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║  [%s] 汇总总表  N=%d/子工况   KILL_R=%.0fm\n', label, N_MC, KILL_R);
fprintf('╠════════════════════════════════════════════════════════════════════════════════╣\n');
fprintf('  %-34s %7s %7s %8s %8s %8s %8s\n','子工况','命中率%','SEP(m)','PeakNy(g)','PM(°)','BW(r/s)','GM(dB)');
fprintf('  %s\n',repmat('-',1,80));
for i=1:numel(SUMM)
    s=SUMM{i};
    fprintf('  %-34s %6.1f%% %7.2f %8.2f %8.1f %8.2f %8.1f\n',...
        s.name, s.hr, s.sep, s.ny, s.pm, s.bw, s.gm);
end
fprintf('╚════════════════════════════════════════════════════════════════════════════════╝\n');
end

function s=tf2str(b); if b; s='HIT'; else; s='MISS'; end; end

function print_summary(res,kr,nm)
mv=res.miss(~isnan(res.miss));
ny=res.peak_ny(~isnan(res.peak_ny));
pp=res.pitch_pm(~isnan(res.pitch_pm));
bw=res.pitch_bw(~isnan(res.pitch_bw));
gm=res.pitch_gm(~isnan(res.pitch_gm));
sep=quantile(mv,0.5); hr=sum(mv<kr)/numel(mv)*100;
fprintf('──────────────────────────────────\n');
fprintf('【汇总】%s\n',nm);
fprintf('  命中率(miss<%.0fm): %.1f%%  SEP: %.2fm\n',kr,hr,sep);
fprintf('  峰值法向过载: %.2f%.2fg  max=%.2fg\n',mean(ny),std(ny),max(ny));
fprintf('  俯仰PM: %.1f%.1f°  BW: %.2f%.2f rad/s  GM: %.1f%.1f dB\n',...
    mean(pp),std(pp),mean(bw),std(bw),mean(gm),std(gm));
fprintf('──────────────────────────────────\n');
end

function plot_results(RES,SC,kr,rc)
nsc=numel(SC); c=lines(nsc);
figure('Name',sprintf('MC工况%s',rc),'Position',[40 40 1300 800]);
for sc=1:nsc
    mv=RES{sc}.miss(~isnan(RES{sc}.miss));
    subplot(2,3,1); hold on;
    histogram(mv,12,'FaceAlpha',0.5,'DisplayName',SC{sc}.name);
    subplot(2,3,2); hold on;
    mv2=sort(mv); plot(mv2,(1:numel(mv2))'/numel(mv2),'LineWidth',2,'Color',c(sc,:),'DisplayName',SC{sc}.name);
    subplot(2,3,4); hold on;
    ny=RES{sc}.peak_ny(~isnan(RES{sc}.peak_ny));
    histogram(ny,12,'FaceAlpha',0.5,'DisplayName',SC{sc}.name);
    subplot(2,3,5); hold on;
    pp=RES{sc}.pitch_pm(~isnan(RES{sc}.pitch_pm));
    histogram(pp,12,'FaceAlpha',0.5,'DisplayName',SC{sc}.name);
end
subplot(2,3,1); xline(kr,'--r'); xlabel('脱靶量(m)'); title('脱靶量分布'); legend('FontSize',7); grid on;
subplot(2,3,2); xline(kr,'--r'); xlabel('脱靶量(m)'); ylabel('CDF'); title('CDF'); legend('FontSize',7); grid on;
subplot(2,3,3); krs=[5,10,15,20,30,50]; Pk=zeros(numel(krs),nsc);
for sc=1:nsc; mv=RES{sc}.miss(~isnan(RES{sc}.miss));
    for ki=1:numel(krs); Pk(ki,sc)=sum(mv<krs(ki))/numel(mv)*100; end; end
bar(krs,Pk,'grouped'); xlabel('杀伤半径(m)'); ylabel('%'); title('命中概率'); grid on;
subplot(2,3,4); xlabel('峰值法向过载(g)'); title('法向过载分布'); legend('FontSize',7); grid on;
subplot(2,3,5); xlabel('俯仰PM(°)'); title('俯仰相位裕度PM分布'); legend('FontSize',7); grid on;
subplot(2,3,6); hold on; grid on;
for sc=1:nsc
    bw=RES{sc}.pitch_bw(:); gm=RES{sc}.pitch_gm(:); ok=~isnan(bw)&~isnan(gm);
    scatter(bw(ok),gm(ok),10,c(sc,:),'filled','MarkerFaceAlpha',0.5,'DisplayName',SC{sc}.name);
end
xlabel('带宽BW(rad/s)'); ylabel('幅值裕度GM(dB)'); title('BW-GM散点'); legend('FontSize',7);
sgtitle(sprintf('工况%s | N=%d/子工况',rc,numel(RES{1}.miss)),'FontSize',10);
end

% ======== 局部函数 ========
function p=mst(b,s); p=b; fn=fieldnames(s); for i=1:numel(fn); p.(fn{i})=s.(fn{i}); end; end
function p=anoise(p,sf)
for i=1:size(sf,1); f=sf{i,1}; if isfield(p,f); p.(f)=p.(f)+sf{i,2}*randn; end; end
end

function res=sim_s(p)
dt=0.002; tt=70; t=0;
XK=[p.V0,p.upsilon0,p.gama0,p.phi0,p.theta0,p.phiv0,0,0,0,p.x0,p.y0,p.z0];
T=[p.T_x0;p.T_y0;p.T_z0];
gs=cg(p);
cs=struct('Ee_ay',0,'Ee_az',0,'ul',p.upsilon0,'pl',p.phi0,'ddx',0,'ddy',0,'ddz',0);
mn=inf; Xm=XK; Tm=T; tm=0; mp=inf; pny=0; i=0;
while t<tt
    i=i+1; t=t+dt;
    XK=real(rk4f(XK,dt,p,cs.ddz,cs.ddy,cs.ddx));
    TVz=p.T_Vz_amp*cos(p.T_Vz_freq*t);
    T=T+[p.T_Vx;p.T_Vy;TVz]*dt;
    [N1,N2,G1,G2,gc]=gf(T(1),T(2),T(3),XK,p.T_Vx,p.T_Vy,TVz,p.N_pn);
    [~,~,~,cs,ny]=cf(XK,N1,N2,G1,G2,gc,gs,dt,cs,p);
    pny=max(pny,ny);
    mn2=norm(XK(10:12)'-T);
    if mn2<mn; mn=mn2; Xm=XK; Tm=T; tm=t; end
    if mn2<5; break; end
    if i>10&&mn2>mp&&mn2>mn*1.5; break; end
    mp=mn2;
end
res=struct('miss',mn,'t_impact',tm,'peak_ny',pny,'pitch_pm',gs.pm_pitch,'bw_pitch',gs.bw_pitch,'gm_pitch',gs.gm_pitch);
end

function g=cg(p)
C45=180/pi; V=p.V0; y=p.y0; S=p.S; L=p.L;
Jz=p.Jz; Jy=p.Jy; Jx=p.Jx; m=p.m; P=p.P;
ro=1.2495*(1-0.0065*y/288.15)^4.25588; q=0.5*ro*V^2;
CLA=0.6303*C45; CLDZ=0.068651*C45; mza=-0.06982*C45; mzdz=-0.21195*C45;
CZB=-0.31*C45; CZDY=-0.07921*C45; myb=-0.19948*C45; mydy=-0.236*C45;
% 俯仰
aa=-mza*q*S*L/Jz; adz=-mzdz*q*S*L/Jz;
ba=(P+CLA*q*S)/m/V; bdz=CLDZ*q*S/m/V;
wm=sqrt(aa); Tm_=1/wm; zm=(ba)/2*Tm_;
kdu=-(adz*ba-aa*bdz)/aa; A2=-bdz/(adz*ba-aa*bdz);
Tdu=adz/(adz*ba-aa*bdz); KACT=-1; w1=p.w1; z1=p.zeta1; tao=p.tao1;
M0v=w1^2/wm^2/tao; M1v=(tao+2*z1/w1)*M0v-1; M2v=(1/w1^2+2*z1*tao/w1)*M0v-2*zm/wm;
J=KACT*kdu*[Tdu V*A2 0;1 0 Tdu;0 V 1]; x2=J\[M2v;M1v;M0v];
g.Kg=x2(1); g.KA=x2(2); g.WI=x2(3); g.KDC=g.WI/g.KA/V+1;
% 偏航
ab=-myb*q*S*L/Jy; ady=-mydy*q*S*L/Jy;
bb=(P-CZB*q*S)/m/V; bdy=-CZDY*q*S/m/V;
wp=sqrt(ab); Tp_=1/wp; zp=bb/2*Tp_;
kdp=-(ady*bb-ab*bdy)/ab; A2p=-bdy/(ady*bb-ab*bdy);
Tdp=ady/(ady*bb-ab*bdy); w2=p.w2; z2=p.zeta2; tao=p.tao2;
M0p=w2^2/wp^2/tao; M1p=(tao+2*z2/w2)*M0p-1; M2p=(1/w2^2+2*z2*tao/w2)*M0p-2*zp/wp;
Jp=KACT*kdp*[Tdp -V*A2p 0;1 0 Tdp;0 -V 1]; x2p=Jp\[M2p;M1p;M0p];
g.Kgp=x2p(1); g.KAp=x2p(2); g.WIp=x2p(3); g.KDCp=-g.WIp/g.KAp/V+1;
% 滚转
cdx=-(-0.02547*C45)*q*S*L/Jx; w3=p.w3; z3=p.zeta3;
g.Kgama=w3^2/(-KACT*cdx); g.Kwx=(0-2*z3*w3)/(KACT*cdx); g.KACT=KACT;
% ── 鲁棒性指标 ──
try
    M0_ol=KACT*kdu*(g.WI+g.KA*V);
    M1_ol=KACT*kdu*(g.Kg+g.WI*Tdu);
    M2_ol=KACT*kdu*(g.Kg*Tdu+g.KA*V*A2);
    HGp=tf([M2_ol M1_ol M0_ol],[Tm_^2 2*Tm_*zm 1 0]);
    [gm_v,pm_v]=margin(HGp);
    g.pm_pitch=pm_v;
    g.gm_pitch=20*log10(abs(gm_v));
    g.bw_pitch=bandwidth(feedback(HGp,1));
catch; g.pm_pitch=NaN; g.gm_pitch=NaN; g.bw_pitch=NaN; end
end

function c=af(XK,p,ddz,ddy,ddx,bt,al)
C45=180/pi; V=XK(1); wx=XK(7); wy=XK(8); wz=XK(9); L=p.L;
CLA=0.6303*p.CLA_s; CLDZ=0.068651; CD0=0.25023*p.CD0_s;
CZB=-0.31*p.CZB_s; CZDY=-0.07921;
mxa=-0.06982*p.mza_s; mzdz_=-0.21195; mzwz=-64;
mxb=-0.00248; mxdx=-0.02547; mxdy=0.001061; mxwx=-8;
myb=-0.19948; mydy=-0.236; mydx=0.249; mywy=-100;
c=[CD0,...
   CLDZ*ddz*C45+CLA*al*C45,...
   CZB*bt*C45+CZDY*ddy*C45,...
   mxdx*ddx*C45+mxb*bt*C45+mxdy*ddy*C45+mxwx*wx*L/V,...
   myb*bt*C45+mydy*ddy*C45+mydx*ddx*C45+mywy*wy*L/V,...
   mxa*al*C45+mzdz_*ddz*C45+mzwz*wz*L/V];
end

function dXK=df(XK,p,ddz,ddy,ddx)
g0=p.g; m=p.m; S=p.S; L=p.L; P=p.P; Jx=p.Jx; Jy=p.Jy; Jz=p.Jz;
V=XK(1);up=XK(2);ga=XK(3);ph=XK(4);th=XK(5);pv=XK(6);
wx=XK(7);wy=XK(8);wz=XK(9);y=XK(11);
bt=asin(max(-1,min(1,cos(th)*(cos(ga)*sin(ph-pv)+sin(up)*sin(ga)*cos(ph-pv))-sin(th)*cos(up)*sin(ga))));
al=asin(max(-1,min(1,(cos(th)*(sin(up)*cos(ga)*cos(ph-pv)-sin(ga)*sin(ph-pv))-sin(th)*cos(up)*cos(ga))/cos(bt))));
gv=asin(max(-1,min(1,1/cos(th)*(cos(al)*sin(bt)*sin(up)-sin(al)*sin(bt)*cos(ga)*cos(up)+cos(bt)*sin(ga)*cos(up)))));
ro=1.2495*(1-0.0065*y/288.15)^4.25588; q=0.5*ro*V^2;
c=af(XK,p,ddz,ddy,ddx,bt,al);
Fx=c(1)*q*S; Fy=c(2)*q*S; Fz=c(3)*q*S;
Mx=c(4)*q*S*L; My=c(5)*q*S*L; Mz=c(6)*q*S*L;
dv=(P*cos(al)*cos(bt)-Fx-m*g0*sin(th))/m;
dth=(P*(sin(al)*cos(gv)+cos(al)*sin(bt)*sin(gv))+Fy*cos(gv)-Fz*sin(gv)-m*g0*cos(th))/(m*V);
dpv=-(P*(sin(al)*sin(gv)-cos(al)*sin(bt)*cos(gv))+Fy*sin(gv)+Fz*cos(gv))/(m*V*cos(th));
dwx=(Mx-(Jz-Jy)*wy*wz)/Jx; dwy=(My-(Jx-Jz)*wx*wz)/Jy; dwz=(Mz-(Jy-Jx)*wy*wx)/Jz;
dup=wy*sin(ga)+wz*cos(ga);
dph=(wy*cos(ga)-wz*sin(ga))/cos(up);
dga=wx-tan(up)*(wy*cos(ga)-wz*sin(ga));
dXK=[dv,dup,dga,dph,dth,dpv,dwx,dwy,dwz,V*cos(th)*cos(pv),V*sin(th),-V*cos(th)*sin(pv)];
end

function XK1=rk4f(XK,dt,p,ddz,ddy,ddx)
k1=dt*df(XK,p,ddz,ddy,ddx); k2=dt*df(XK+k1/2,p,ddz,ddy,ddx);
k3=dt*df(XK+k2/2,p,ddz,ddy,ddx); k4=dt*df(XK+k3,p,ddz,ddy,ddx);
XK1=XK+(k1+2*k2+2*k3+k4)/6;
end

function [N1,N2,G1,G2,gc]=gf(Tx,Ty,Tz,XK,TVx,TVy,TVz,Np)
g0=9.8; V=XK(1);up=XK(2);ga=XK(3);ph=XK(4);th=XK(5);pv=XK(6);
x=XK(10);y=XK(11);z=XK(12);
Vax=V*cos(th)*cos(pv); Vay=V*sin(th); Vaz=-V*cos(th)*sin(pv);
R=max(norm([x-Tx,y-Ty,z-Tz]),0.1);
Vrx=TVx-Vax; Vry=TVy-Vay; Vrz=TVz-Vaz;
qy=((Tz-z)*Vrx-(Tx-x)*Vrz)/R^2; qz=((Tx-x)*Vry-(Ty-y)*Vrx)/R^2;
Vc=-(Vrx*(Tx-x)+Vry*(Ty-y)+Vrz*(Tz-z))/R;
ayb=cos(up)*cos(ga)*Np*Vc*qz+(sin(up)*sin(ph)*cos(ga)+cos(ph)*sin(ga))*(-Np*Vc*qy);
azb=-cos(up)*sin(ga)*Np*Vc*qz+(cos(ph)*cos(ga)-sin(up)*sin(ph)*sin(ga))*(-Np*Vc*qy);
gc=0; N1=ayb/g0; N2=azb/g0;  % 纯STT控制
G1=cos(up)*cos(ga); G2=-cos(up)*sin(ga);
% ── 过载指令硬限幅 (20g, N1/N2单位为g) ──
N1=max(-20,min(20,N1));
N2=max(-20,min(20,N2));
end

function [ddx,ddy,ddz,cs,ny]=cf(XK,N1,N2,G1,G2,gc,gs,dt,cs,p)
C45=180/pi; g0=p.g; m=p.m; S=p.S; L=p.L; P=p.P;
V=XK(1);up=XK(2);ga=XK(3);ph=XK(4);th=XK(5);pv=XK(6);wx=XK(7);y=XK(11);
bt=asin(max(-1,min(1,cos(th)*(cos(ga)*sin(ph-pv)+sin(up)*sin(ga)*cos(ph-pv))-sin(th)*cos(up)*sin(ga))));
al=asin(max(-1,min(1,(cos(th)*(sin(up)*cos(ga)*cos(ph-pv)-sin(ga)*sin(ph-pv))-sin(th)*cos(up)*cos(ga))/cos(bt))));
ro=1.2495*(1-0.0065*y/288.15)^4.25588; q=0.5*ro*V^2;
c=af(XK,p,cs.ddz,cs.ddy,cs.ddx,bt,al);
Fx=c(1)*q*S; Fy=c(2)*q*S; Fz=c(3)*q*S;
T=[cos(al)*cos(bt) sin(al) -cos(al)*sin(bt);-sin(al)*cos(bt) cos(al) sin(al)*sin(bt);sin(bt) 0 cos(bt)];
Fb=T*[-Fx;Fy;Fz]; ayb=Fb(2)/m; azb=Fb(3)/m;
ny=sqrt(ayb^2+azb^2)/g0;
ayc=N1*g0+G1*g0; eay=gs.KDC*ayc-ayb-(gs.KDC-1)*G1*g0;
ddz=gs.KACT*(-gs.WI*up+(cs.Ee_ay+gs.KA*eay*dt)-gs.Kg/dt*(up-cs.ul));
ddz=max(-15/C45,min(15/C45,ddz));
cs.Ee_ay=cs.Ee_ay+gs.KA*eay*dt; cs.ul=up;
azc=N2*g0+G2*g0; eaz=gs.KDCp*azc-azb-(gs.KDCp-1)*G2*g0;
ddy=gs.KACT*(-gs.WIp*ph+(cs.Ee_az+gs.KAp*eaz*dt)-gs.Kgp/dt*(ph-cs.pl));
ddy=ddy-0.003*wx*ayb; ddy=max(-15/C45,min(15/C45,ddy));
cs.Ee_az=cs.Ee_az+gs.KAp*eaz*dt; cs.pl=ph;
ddx=gs.KACT*(gs.Kgama*(gc-ga)-gs.Kwx*wx);
ddx=max(-15/C45,min(15/C45,ddx));
cs.ddx=ddx; cs.ddy=ddy; cs.ddz=ddz;
end