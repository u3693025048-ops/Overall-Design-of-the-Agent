function plot_experiment_motion_scenarios()
% Plot deterministic motion traces for three design-task scenarios.
% Run:
%   plot_experiment_motion_scenarios
%
% Output figures are saved under:
%   knowledge_base/matlab/guidance/figures/motion_scenarios/
%
% The dynamics, autopilot, and PN guidance law follow monte_carlo_single.m.
% Initial positions are adjusted for an air-to-air engagement visualization:
% the missile starts from a fighter-like launch point, while targets begin
% ahead at comparable altitude with moderate crossrange offsets.
% For clear motion illustrations, align_initial_heading=true points the
% initial missile velocity/attitude toward the initial target line of sight.
% Set it to false to keep the exact zero-angle nominal launch attitude.

clc; close all; rng(42);

out_dir = fullfile(fileparts(mfilename('fullpath')), 'figures', 'motion_scenarios');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

base = default_params();
scenarios = experiment_scenarios();
nsc = numel(scenarios);
traces = cell(nsc, 1);
colors = lines(nsc);
align_initial_heading = true;

fprintf('Running motion traces for %d experiment scenarios...\n', nsc);
for i = 1:nsc
    p = merge_struct(base, scenarios{i}.params);
    if align_initial_heading
        p = align_initial_velocity_to_los(p);
    end
    traces{i} = simulate_trace(p, scenarios{i}.id);
    fprintf('%-14s final_time=%6.2f s  min_range=%8.2f m  peak_load=%6.2f g\n', ...
        scenarios{i}.id, traces{i}.t_end, traces{i}.miss, traces{i}.peak_ny);
end

plot_3d_trajectories(traces, scenarios, colors, out_dir);
plot_range_history(traces, scenarios, colors, out_dir);
plot_planar_views(traces, scenarios, colors, out_dir);
plot_motion_states(traces, scenarios, colors, out_dir);
plot_guidance_loads(traces, scenarios, colors, out_dir);
plot_closest_approach(traces, scenarios, colors, out_dir);

fprintf('Done. Figures saved to:\n  %s\n', out_dir);
end

function base = default_params()
% Parameters follow monte_carlo_single.m nominal values.
base = struct( ...
    'x0',0,'y0',9000,'z0',0,'V0',1200, ...
    'theta0',0,'phiv0',0,'upsilon0',0,'gama0',0,'phi0',0, ...
    'Jx',3.0,'Jy',220,'Jz',220,'m',231,'S',0.03236,'L',3.66,'g',9.8,'P',5000, ...
    'T_x0',16000,'T_y0',8700,'T_z0',2200,'T_Vx',250,'T_Vy',0, ...
    'T_Vz_amp',80,'T_Vz_freq',0.20,'N_pn',4,'ny_lim',20, ...
    'CLA_s',1,'CD0_s',1,'mza_s',1,'CZB_s',1, ...
    'w1',40,'zeta1',0.75,'tao1',0.20, ...
    'w2',35,'zeta2',0.75,'tao2',0.20, ...
    'w3',40,'zeta3',0.70,'gama_max_deg',45);
end

function scenarios = experiment_scenarios()
% Three design-task scenarios coupling target maneuver and autopilot constants.
% D1: low maneuver + nominal autopilot.
% D2: medium maneuver + mild autopilot degradation.
% D3: high maneuver + moderate autopilot degradation.
scenarios = { ...
    struct('id','D1','label','D1 low maneuver, nominal AP', ...
        'params',struct('T_x0',16500,'T_y0',8750,'T_z0',2100, ...
        'T_Vz_amp',25,'T_Vy',8,'T_Vz_freq',0.40, ...
        'w1',40.0,'zeta1',0.75,'tao1',0.20, ...
        'w2',35.0,'zeta2',0.75,'tao2',0.20, ...
        'w3',40.0,'zeta3',0.70)), ...
    struct('id','D2','label','D2 medium maneuver, mild AP degradation', ...
        'params',struct('T_x0',18000,'T_y0',8450,'T_z0',2800, ...
        'T_Vz_amp',55,'T_Vy',16,'T_Vz_freq',0.75, ...
        'w1',34.0,'zeta1',0.62,'tao1',0.24, ...
        'w2',29.8,'zeta2',0.62,'tao2',0.24, ...
        'w3',34.0,'zeta3',0.58)), ...
    struct('id','D3','label','D3 high maneuver, moderate AP degradation', ...
        'params',struct('T_x0',19000,'T_y0',8250,'T_z0',3300, ...
        'T_Vz_amp',85,'T_Vy',24,'T_Vz_freq',1.20, ...
        'w1',26.0,'zeta1',0.50,'tao1',0.28, ...
        'w2',22.8,'zeta2',0.50,'tao2',0.28, ...
        'w3',26.0,'zeta3',0.45)) ...
};
end

function out = merge_struct(a, b)
out = a;
fn = fieldnames(b);
for i = 1:numel(fn)
    out.(fn{i}) = b.(fn{i});
end
end

function p = align_initial_velocity_to_los(p)
% Align velocity and attitude angles with the initial missile-target line of sight.
dx = p.T_x0 - p.x0;
dy = p.T_y0 - p.y0;
dz = p.T_z0 - p.z0;
horiz = max(sqrt(dx^2 + dz^2), 1e-6);
theta_los = atan2(dy, horiz);
phiv_los = atan2(-dz, dx);
p.theta0 = theta_los;
p.phiv0 = phiv_los;
p.upsilon0 = theta_los;
p.phi0 = phiv_los;
p.gama0 = 0;
end

function tr = simulate_trace(p, scenario_id)
dt = 0.002;
t_max = 70.0;
max_steps = ceil(t_max / dt) + 2;

XK = [p.V0,p.upsilon0,p.gama0,p.phi0,p.theta0,p.phiv0,0,0,0,p.x0,p.y0,p.z0];
T = [p.T_x0;p.T_y0;p.T_z0];
gs = cg(p);
cs = struct('Ee_ay',0,'Ee_az',0,'ul',p.upsilon0,'pl',p.phi0,'ddx',0,'ddy',0,'ddz',0);

time = nan(max_steps, 1);
missile = nan(max_steps, 3);
target = nan(max_steps, 3);
speed = nan(max_steps, 1);
range = nan(max_steps, 1);
cmd_n1 = nan(max_steps, 1);
cmd_n2 = nan(max_steps, 1);
cmd_n = nan(max_steps, 1);
load_n = nan(max_steps, 1);
los_rate = nan(max_steps, 1);
closing_speed = nan(max_steps, 1);

t = 0.0;
idx = 0;
min_range = inf;
prev_range = inf;
best_time = 0.0;
peak_ny = 0.0;
best_missile = XK(10:12);
best_target = T(:).';

while t < t_max && idx < max_steps
    idx = idx + 1;
    t = t + dt;

    % Match monte_carlo_single.m: integrate with previous-step controls,
    % then update target motion and compute the next-step guidance command.
    XK = real(rk4f(XK, dt, p, cs.ddz, cs.ddy, cs.ddx));
    TVz = p.T_Vz_amp * cos(p.T_Vz_freq * t);
    T = T + [p.T_Vx;p.T_Vy;TVz] * dt;

    [N1,N2,G1,G2,gc,qy,qz,Vc] = gf(T(1),T(2),T(3),XK,p.T_Vx,p.T_Vy,TVz,p.N_pn,p.ny_lim);
    [~,~,~,cs,ny] = cf(XK,N1,N2,G1,G2,gc,gs,dt,cs,p);
    peak_ny = max(peak_ny, ny);

    pos_m = XK(10:12);
    pos_t = T(:).';
    r_now = norm(pos_m - pos_t);
    if r_now < min_range
        min_range = r_now;
        best_time = t;
        best_missile = pos_m;
        best_target = pos_t;
    end

    time(idx) = t;
    missile(idx,:) = pos_m;
    target(idx,:) = pos_t;
    speed(idx) = XK(1);
    range(idx) = r_now;
    cmd_n1(idx) = N1;
    cmd_n2(idx) = N2;
    cmd_n(idx) = sqrt(N1^2 + N2^2);
    load_n(idx) = ny;
    los_rate(idx) = sqrt(qy^2 + qz^2);
    closing_speed(idx) = Vc;

    if r_now < 5
        break;
    end
    if idx > 10 && r_now > prev_range && r_now > min_range * 1.5
        break;
    end
    prev_range = r_now;
end

valid = find(~isnan(time));
tr = struct();
tr.id = scenario_id;
tr.t = time(valid);
tr.missile = missile(valid,:);
tr.target = target(valid,:);
tr.speed = speed(valid);
tr.range = range(valid);
tr.cmd_n1 = cmd_n1(valid);
tr.cmd_n2 = cmd_n2(valid);
tr.cmd_n = cmd_n(valid);
tr.load_n = load_n(valid);
tr.los_rate = los_rate(valid);
tr.closing_speed = closing_speed(valid);
tr.t_end = tr.t(end);
tr.miss = min_range;
tr.t_miss = best_time;
tr.peak_ny = peak_ny;
tr.closest_missile = best_missile;
tr.closest_target = best_target;
end

function plot_3d_trajectories(traces, scenarios, colors, out_dir)
fig = figure('Name','3D Engagement Trajectories','Color','w','Position',[60 60 1100 780]);
hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    hit = closest_hit_point_km(tr);
    plot3(tr.missile(:,1)/1000, tr.missile(:,3)/1000, tr.missile(:,2)/1000, ...
        '-', 'Color', colors(i,:), 'LineWidth', 2.0, 'DisplayName', [scenarios{i}.id ' missile']);
    plot3(tr.target(:,1)/1000, tr.target(:,3)/1000, tr.target(:,2)/1000, ...
        '--', 'Color', colors(i,:), 'LineWidth', 1.8, 'DisplayName', [scenarios{i}.id ' target']);
    scatter3(tr.missile(1,1)/1000, tr.missile(1,3)/1000, tr.missile(1,2)/1000, ...
        35, colors(i,:), 'filled', 'HandleVisibility','off');
    scatter3(tr.target(1,1)/1000, tr.target(1,3)/1000, tr.target(1,2)/1000, ...
        35, colors(i,:), 'd', 'filled', 'HandleVisibility','off');
    scatter3(hit(1), hit(3), hit(2), 120, colors(i,:), 'p', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth', 0.8, 'HandleVisibility','off');
end
xlabel('Downrange X (km)');
ylabel('Crossrange Z (km)');
zlabel('Altitude Y (km)');
title('3D Missile and Target Trajectories');
legend('Location','bestoutside');
view(42, 22);
axis tight;
save_fig(fig, out_dir, 'figure01_3d_trajectories');
end

function plot_range_history(traces, scenarios, colors, out_dir)
fig = figure('Name','Relative Range History','Color','w','Position',[80 80 1000 620]);
hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    plot(tr.t, tr.range/1000, 'Color', colors(i,:), 'LineWidth', 2.0, ...
        'DisplayName', scenarios{i}.label);
    scatter(tr.t_miss, tr.miss/1000, 90, colors(i,:), 'p', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth', 0.8, 'HandleVisibility','off');
end
add_horizontal_line(gca, 0.010, '--k', 'Kill radius 10 m', true);
xlabel('Time (s)');
ylabel('Relative Range (km)');
title('Closing Range During Guidance');
legend('Location','northeast');
save_fig(fig, out_dir, 'figure02_range_history');
end

function plot_planar_views(traces, scenarios, colors, out_dir)
fig = figure('Name','Planar Motion Views','Color','w','Position',[90 90 1200 520]);

subplot(1,2,1); hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    hit = closest_hit_point_km(tr);
    plot(tr.missile(:,1)/1000, tr.missile(:,3)/1000, '-', ...
        'Color', colors(i,:), 'LineWidth', 2.0, 'DisplayName', [scenarios{i}.id ' missile']);
    plot(tr.target(:,1)/1000, tr.target(:,3)/1000, '--', ...
        'Color', colors(i,:), 'LineWidth', 1.6, 'DisplayName', [scenarios{i}.id ' target']);
    scatter(hit(1), hit(3), 105, colors(i,:), 'p', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth', 0.8, 'HandleVisibility','off');
end
xlabel('Downrange X (km)');
ylabel('Crossrange Z (km)');
title('Top View: Downrange vs Crossrange');
legend('Location','best');

subplot(1,2,2); hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    hit = closest_hit_point_km(tr);
    plot(tr.missile(:,1)/1000, tr.missile(:,2)/1000, '-', ...
        'Color', colors(i,:), 'LineWidth', 2.0, 'DisplayName', [scenarios{i}.id ' missile']);
    plot(tr.target(:,1)/1000, tr.target(:,2)/1000, '--', ...
        'Color', colors(i,:), 'LineWidth', 1.6, 'DisplayName', [scenarios{i}.id ' target']);
    scatter(hit(1), hit(2), 105, colors(i,:), 'p', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth', 0.8, 'HandleVisibility','off');
end
xlabel('Downrange X (km)');
ylabel('Altitude Y (km)');
title('Side View: Downrange vs Altitude');
legend('Location','best');

save_fig(fig, out_dir, 'figure03_planar_views');
end

function plot_motion_states(traces, scenarios, colors, out_dir)
fig = figure('Name','Motion State Histories','Color','w','Position',[100 100 1100 780]);

subplot(3,1,1); hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    plot(tr.t, tr.speed, 'Color', colors(i,:), 'LineWidth', 1.8, 'DisplayName', scenarios{i}.label);
end
xlabel('Time (s)');
ylabel('Missile Speed (m/s)');
title('Missile Speed Profile');
legend('Location','best');

subplot(3,1,2); hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    plot(tr.t, tr.missile(:,2)/1000, '-', 'Color', colors(i,:), 'LineWidth', 1.8, ...
        'DisplayName', [scenarios{i}.id ' missile']);
    plot(tr.t, tr.target(:,2)/1000, '--', 'Color', colors(i,:), 'LineWidth', 1.4, ...
        'DisplayName', [scenarios{i}.id ' target']);
end
xlabel('Time (s)');
ylabel('Altitude (km)');
title('Altitude Profile');
legend('Location','best');

subplot(3,1,3); hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    plot(tr.t, tr.los_rate * 180/pi, 'Color', colors(i,:), 'LineWidth', 1.8, ...
        'DisplayName', scenarios{i}.label);
end
xlabel('Time (s)');
ylabel('LOS Rate (deg/s)');
title('Line-of-Sight Rate Magnitude');
legend('Location','best');

save_fig(fig, out_dir, 'figure04_motion_states');
end

function plot_guidance_loads(traces, scenarios, colors, out_dir)
fig = figure('Name','Guidance Commands and Load','Color','w','Position',[110 110 1100 820]);

subplot(3,1,1); hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    plot(tr.t, tr.cmd_n1, 'Color', colors(i,:), 'LineWidth', 1.6, 'DisplayName', scenarios{i}.label);
end
add_horizontal_line(gca, 20, '--k', 'Command limit', true);
add_horizontal_line(gca, -20, '--k', '', false);
xlabel('Time (s)');
ylabel('N1 Command (g)');
title('Pitch-Plane Load Command');
legend('Location','best');

subplot(3,1,2); hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    plot(tr.t, tr.cmd_n2, 'Color', colors(i,:), 'LineWidth', 1.6, 'DisplayName', scenarios{i}.label);
end
add_horizontal_line(gca, 20, '--k', 'Command limit', true);
add_horizontal_line(gca, -20, '--k', '', false);
xlabel('Time (s)');
ylabel('N2 Command (g)');
title('Yaw-Plane Load Command');
legend('Location','best');

subplot(3,1,3); hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    plot(tr.t, tr.load_n, 'Color', colors(i,:), 'LineWidth', 1.8, 'DisplayName', scenarios{i}.label);
end
add_horizontal_line(gca, 20, '--k', '20 g constraint', true);
xlabel('Time (s)');
ylabel('Achieved Load (g)');
title('Achieved Normal Load Magnitude');
legend('Location','best');

save_fig(fig, out_dir, 'figure05_guidance_loads');
end

function plot_closest_approach(traces, scenarios, colors, out_dir)
fig = figure('Name','Closest Approach Geometry','Color','w','Position',[120 120 1050 720]);
hold on; grid on; box on;
for i = 1:numel(traces)
    tr = traces{i};
    m = tr.closest_missile / 1000;
    tg = tr.closest_target / 1000;
    hit = closest_hit_point_km(tr);
    plot3([m(1) tg(1)], [m(3) tg(3)], [m(2) tg(2)], '-', ...
        'Color', colors(i,:), 'LineWidth', 1.8, 'DisplayName', [scenarios{i}.id ' miss vector']);
    scatter3(m(1), m(3), m(2), 55, colors(i,:), 'o', 'filled', ...
        'HandleVisibility','off');
    scatter3(tg(1), tg(3), tg(2), 70, colors(i,:), 'd', 'filled', ...
        'HandleVisibility','off');
    scatter3(hit(1), hit(3), hit(2), 130, colors(i,:), 'p', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth', 0.8, 'HandleVisibility','off');
    text(tg(1), tg(3), tg(2), sprintf('  %s %.1f m', scenarios{i}.id, tr.miss), ...
        'Color', colors(i,:), 'Interpreter', 'none', 'FontSize', 9, ...
        'HandleVisibility','off');
end
xlabel('Downrange X (km)');
ylabel('Crossrange Z (km)');
zlabel('Altitude Y (km)');
title('Closest Approach Between Missile and Target');
legend('Location','bestoutside');
view(42, 22);
axis tight;
save_fig(fig, out_dir, 'figure06_closest_approach');
end

function hit = closest_hit_point_km(tr)
% Use the midpoint of missile and target positions at closest approach.
hit = 0.5 * (tr.closest_missile + tr.closest_target) / 1000;
end

function save_fig(fig, out_dir, base_name)
png_path = fullfile(out_dir, [base_name '.png']);
fig_path = fullfile(out_dir, [base_name '.fig']);
try
    if exist('exportgraphics', 'file') == 2
        exportgraphics(fig, png_path, 'Resolution', 200);
    else
        saveas(fig, png_path);
    end
catch
    try
        saveas(fig, png_path);
    catch ME
        warning('Could not save PNG %s: %s', png_path, ME.message);
    end
end
try
    saveas(fig, fig_path);
catch
end
end

function add_horizontal_line(ax, y_value, style_spec, label_text, show_in_legend)
axes(ax);
xl = xlim(ax);
h = plot(ax, xl, [y_value y_value], style_spec, 'LineWidth', 1.0);
if show_in_legend && ~isempty(label_text)
    set(h, 'DisplayName', label_text);
else
    set(h, 'HandleVisibility', 'off');
end
if ~isempty(label_text)
    x_text = xl(1) + 0.02 * (xl(2) - xl(1));
    text(ax, x_text, y_value, ['  ' label_text], ...
        'Color', 'k', 'VerticalAlignment', 'bottom', 'Interpreter', 'none', ...
        'HandleVisibility', 'off');
end
end

function g = cg(p)
C45 = 180/pi; V = p.V0; y = p.y0; S = p.S; L = p.L;
Jz = p.Jz; Jy = p.Jy; Jx = p.Jx; m = p.m; P = p.P;
ro = 1.2495 * (1 - 0.0065 * y / 288.15)^4.25588; q = 0.5 * ro * V^2;
CLA = 0.6303 * C45; CLDZ = 0.068651 * C45; mza = -0.06982 * C45; mzdz = -0.21195 * C45;
CZB = -0.31 * C45; CZDY = -0.07921 * C45; myb = -0.19948 * C45; mydy = -0.236 * C45;

aa = -mza * q * S * L / Jz; adz = -mzdz * q * S * L / Jz;
ba = (P + CLA * q * S) / m / V; bdz = CLDZ * q * S / m / V;
wm = sqrt(aa); Tm_ = 1 / wm; zm = ba / 2 * Tm_;
kdu = -(adz * ba - aa * bdz) / aa; A2 = -bdz / (adz * ba - aa * bdz);
Tdu = adz / (adz * ba - aa * bdz); KACT = -1; w1 = p.w1; z1 = p.zeta1; tao = p.tao1;
M0v = w1^2 / wm^2 / tao; M1v = (tao + 2 * z1 / w1) * M0v - 1;
M2v = (1 / w1^2 + 2 * z1 * tao / w1) * M0v - 2 * zm / wm;
J = KACT * kdu * [Tdu V*A2 0; 1 0 Tdu; 0 V 1]; x2 = J \ [M2v; M1v; M0v];
g.Kg = x2(1); g.KA = x2(2); g.WI = x2(3); g.KDC = g.WI / g.KA / V + 1;

ab = -myb * q * S * L / Jy; ady = -mydy * q * S * L / Jy;
bb = (P - CZB * q * S) / m / V; bdy = -CZDY * q * S / m / V;
wp = sqrt(ab); zp = bb / 2 / wp;
kdp = -(ady * bb - ab * bdy) / ab; A2p = -bdy / (ady * bb - ab * bdy);
Tdp = ady / (ady * bb - ab * bdy); w2 = p.w2; z2 = p.zeta2; tao = p.tao2;
M0p = w2^2 / wp^2 / tao; M1p = (tao + 2 * z2 / w2) * M0p - 1;
M2p = (1 / w2^2 + 2 * z2 * tao / w2) * M0p - 2 * zp / wp;
Jp = KACT * kdp * [Tdp -V*A2p 0; 1 0 Tdp; 0 -V 1]; x2p = Jp \ [M2p; M1p; M0p];
g.Kgp = x2p(1); g.KAp = x2p(2); g.WIp = x2p(3); g.KDCp = -g.WIp / g.KAp / V + 1;

cdx = -(-0.02547 * C45) * q * S * L / Jx; w3 = p.w3; z3 = p.zeta3;
g.Kgama = w3^2 / (-KACT * cdx); g.Kwx = (0 - 2 * z3 * w3) / (KACT * cdx); g.KACT = KACT;

try
    M0_ol = KACT * kdu * (g.WI + g.KA * V);
    M1_ol = KACT * kdu * (g.Kg + g.WI * Tdu);
    M2_ol = KACT * kdu * (g.Kg * Tdu + g.KA * V * A2);
    HGp = tf([M2_ol M1_ol M0_ol], [Tm_^2 2*Tm_*zm 1 0]);
    [gm_v, pm_v] = margin(HGp);
    g.pm_pitch = pm_v;
    g.gm_pitch = 20 * log10(abs(gm_v));
    g.bw_pitch = bandwidth(feedback(HGp, 1));
catch
    g.pm_pitch = NaN; g.gm_pitch = NaN; g.bw_pitch = NaN;
end
end

function c = af(XK, p, ddz, ddy, ddx, bt, al)
C45 = 180/pi; V = XK(1); wx = XK(7); wy = XK(8); wz = XK(9); L = p.L;
CLA = 0.6303 * p.CLA_s; CLDZ = 0.068651; CD0 = 0.25023 * p.CD0_s;
CZB = -0.31 * p.CZB_s; CZDY = -0.07921;
mxa = -0.06982 * p.mza_s; mzdz_ = -0.21195; mzwz = -64;
mxb = -0.00248; mxdx = -0.02547; mxdy = 0.001061; mxwx = -8;
myb = -0.19948; mydy = -0.236; mydx = 0.249; mywy = -100;
c = [CD0, ...
   CLDZ * ddz * C45 + CLA * al * C45, ...
   CZB * bt * C45 + CZDY * ddy * C45, ...
   mxdx * ddx * C45 + mxb * bt * C45 + mxdy * ddy * C45 + mxwx * wx * L / V, ...
   myb * bt * C45 + mydy * ddy * C45 + mydx * ddx * C45 + mywy * wy * L / V, ...
   mxa * al * C45 + mzdz_ * ddz * C45 + mzwz * wz * L / V];
end

function dXK = df(XK, p, ddz, ddy, ddx)
g0 = p.g; m = p.m; S = p.S; L = p.L; P = p.P; Jx = p.Jx; Jy = p.Jy; Jz = p.Jz;
V = XK(1); up = XK(2); ga = XK(3); ph = XK(4); th = XK(5); pv = XK(6);
wx = XK(7); wy = XK(8); wz = XK(9); y = XK(11);
bt = asin(max(-1,min(1,cos(th)*(cos(ga)*sin(ph-pv)+sin(up)*sin(ga)*cos(ph-pv))-sin(th)*cos(up)*sin(ga))));
al = asin(max(-1,min(1,(cos(th)*(sin(up)*cos(ga)*cos(ph-pv)-sin(ga)*sin(ph-pv))-sin(th)*cos(up)*cos(ga))/cos(bt))));
gv = asin(max(-1,min(1,1/cos(th)*(cos(al)*sin(bt)*sin(up)-sin(al)*sin(bt)*cos(ga)*cos(up)+cos(bt)*sin(ga)*cos(up)))));
ro = 1.2495 * (1 - 0.0065 * y / 288.15)^4.25588; q = 0.5 * ro * V^2;
c = af(XK, p, ddz, ddy, ddx, bt, al);
Fx = c(1) * q * S; Fy = c(2) * q * S; Fz = c(3) * q * S;
Mx = c(4) * q * S * L; My = c(5) * q * S * L; Mz = c(6) * q * S * L;
dv = (P * cos(al) * cos(bt) - Fx - m * g0 * sin(th)) / m;
dth = (P * (sin(al) * cos(gv) + cos(al) * sin(bt) * sin(gv)) + Fy * cos(gv) - Fz * sin(gv) - m * g0 * cos(th)) / (m * V);
dpv = -(P * (sin(al) * sin(gv) - cos(al) * sin(bt) * cos(gv)) + Fy * sin(gv) + Fz * cos(gv)) / (m * V * cos(th));
dwx = (Mx - (Jz - Jy) * wy * wz) / Jx;
dwy = (My - (Jx - Jz) * wx * wz) / Jy;
dwz = (Mz - (Jy - Jx) * wy * wx) / Jz;
dup = wy * sin(ga) + wz * cos(ga);
dph = (wy * cos(ga) - wz * sin(ga)) / cos(up);
dga = wx - tan(up) * (wy * cos(ga) - wz * sin(ga));
dXK = [dv,dup,dga,dph,dth,dpv,dwx,dwy,dwz,V*cos(th)*cos(pv),V*sin(th),-V*cos(th)*sin(pv)];
end

function XK1 = rk4f(XK, dt, p, ddz, ddy, ddx)
k1 = dt * df(XK, p, ddz, ddy, ddx);
k2 = dt * df(XK + k1/2, p, ddz, ddy, ddx);
k3 = dt * df(XK + k2/2, p, ddz, ddy, ddx);
k4 = dt * df(XK + k3, p, ddz, ddy, ddx);
XK1 = XK + (k1 + 2*k2 + 2*k3 + k4) / 6;
end

function [N1,N2,G1,G2,gc,qy,qz,Vc] = gf(Tx, Ty, Tz, XK, TVx, TVy, TVz, Np, ny_lim)
g0 = 9.8; V = XK(1); up = XK(2); ga = XK(3); ph = XK(4); th = XK(5); pv = XK(6);
x = XK(10); y = XK(11); z = XK(12);
Vax = V * cos(th) * cos(pv); Vay = V * sin(th); Vaz = -V * cos(th) * sin(pv);
R = max(norm([x-Tx,y-Ty,z-Tz]), 0.1);
Vrx = TVx - Vax; Vry = TVy - Vay; Vrz = TVz - Vaz;
qy = ((Tz-z) * Vrx - (Tx-x) * Vrz) / R^2;
qz = ((Tx-x) * Vry - (Ty-y) * Vrx) / R^2;
Vc = -(Vrx * (Tx-x) + Vry * (Ty-y) + Vrz * (Tz-z)) / R;
ayb = cos(up) * cos(ga) * Np * Vc * qz + (sin(up)*sin(ph)*cos(ga)+cos(ph)*sin(ga)) * (-Np * Vc * qy);
azb = -cos(up) * sin(ga) * Np * Vc * qz + (cos(ph)*cos(ga)-sin(up)*sin(ph)*sin(ga)) * (-Np * Vc * qy);
gc = 0; N1 = ayb / g0; N2 = azb / g0;
G1 = cos(up) * cos(ga); G2 = -cos(up) * sin(ga);
N1 = max(-ny_lim, min(ny_lim, N1));
N2 = max(-ny_lim, min(ny_lim, N2));
end

function [ddx,ddy,ddz,cs,ny] = cf(XK, N1, N2, G1, G2, gc, gs, dt, cs, p)
C45 = 180/pi; g0 = p.g; m = p.m; S = p.S; L = p.L; P = p.P;
V = XK(1); up = XK(2); ga = XK(3); ph = XK(4); th = XK(5); pv = XK(6); wx = XK(7); y = XK(11);
bt = asin(max(-1,min(1,cos(th)*(cos(ga)*sin(ph-pv)+sin(up)*sin(ga)*cos(ph-pv))-sin(th)*cos(up)*sin(ga))));
al = asin(max(-1,min(1,(cos(th)*(sin(up)*cos(ga)*cos(ph-pv)-sin(ga)*sin(ph-pv))-sin(th)*cos(up)*cos(ga))/cos(bt))));
ro = 1.2495 * (1 - 0.0065 * y / 288.15)^4.25588; q = 0.5 * ro * V^2;
c = af(XK, p, cs.ddz, cs.ddy, cs.ddx, bt, al);
Fx = c(1) * q * S; Fy = c(2) * q * S; Fz = c(3) * q * S;
T = [cos(al)*cos(bt) sin(al) -cos(al)*sin(bt); -sin(al)*cos(bt) cos(al) sin(al)*sin(bt); sin(bt) 0 cos(bt)];
Fb = T * [-Fx; Fy; Fz]; ayb = Fb(2) / m; azb = Fb(3) / m;
ny = sqrt(ayb^2 + azb^2) / g0;
ayc = N1 * g0 + G1 * g0; eay = gs.KDC * ayc - ayb - (gs.KDC - 1) * G1 * g0;
ddz = gs.KACT * (-gs.WI * up + (cs.Ee_ay + gs.KA * eay * dt) - gs.Kg / dt * (up - cs.ul));
ddz = max(-15/C45, min(15/C45, ddz));
cs.Ee_ay = cs.Ee_ay + gs.KA * eay * dt; cs.ul = up;
azc = N2 * g0 + G2 * g0; eaz = gs.KDCp * azc - azb - (gs.KDCp - 1) * G2 * g0;
ddy = gs.KACT * (-gs.WIp * ph + (cs.Ee_az + gs.KAp * eaz * dt) - gs.Kgp / dt * (ph - cs.pl));
ddy = ddy - 0.003 * wx * ayb; ddy = max(-15/C45, min(15/C45, ddy));
cs.Ee_az = cs.Ee_az + gs.KAp * eaz * dt; cs.pl = ph;
ddx = gs.KACT * (gs.Kgama * (gc - ga) - gs.Kwx * wx);
ddx = max(-15/C45, min(15/C45, ddx));
cs.ddx = ddx; cs.ddy = ddy; cs.ddz = ddz;
end
