# 数据资产 02 — 公式集

> 30 张卡【I - 公式】段中的全部显示公式，按域—卡—序号编号，编号（F-xx）可跨文档引用

> 规模：**197 条**


---


## A-01 《比例导引基本原理与纯/真比例导引（PPN vs TPN）》（域 A 比例导引 PN）


**`F-A-01-01`**


$$
a_m = N \cdot V_c \cdot \dot\lambda
$$


**`F-A-01-02`**


$$
\eta = \theta_m - \lambda
$$


**`F-A-01-03`**


$$
\dot\lambda = \frac{\mathbf{R} \times \mathbf{V}_{rel}}{|\mathbf{R}|^2},\qquad \dot\lambda = \frac{V_t\sin\theta_t - V_m\sin\theta_m}{R}
$$


## A-02 《导航比 N 的影响规律与选取准则》（域 A 比例导引 PN）


**`F-A-02-01`**


$$
\dot V = (2-N)\frac{V_c}{R}\dot\lambda^2 + \frac{\dot\lambda \, a_t}{R}
$$


**`F-A-02-02`**


$$
a_m^* = \frac{3}{t_{go}^2}y + \frac{3}{t_{go}}v_y \;\xrightarrow{\;y\approx0\;}\; a_m^* \approx \frac{3}{t_{go}}v_y = 3V_c\dot\lambda \;\Rightarrow\; N_{opt}=3
$$


**`F-A-02-03`**


$$
N(R) = N_0 + (N_f-N_0)\left(1-\frac{R}{R_0}\right)
$$


**`F-A-02-04`**


$$
N(\dot\lambda) = N_{min} + (N_{max}-N_{min})\tanh\left(k|\dot\lambda|\right),\quad k=10\sim20
$$


**`F-A-02-05`**


$$
N(t_{go}) = N_0 + (N_f-N_0)e^{-t_{go}/\tau}
$$


**`F-A-02-06`**


$$
N(t) = N_0 + (N_f-N_0)\left(\frac{t}{T_{go}}\right)^{k},\; k=0.5\sim1.0
$$


**`F-A-02-07`**


$$
N_{adaptive} = N_{base} + K\frac{|\dot\lambda|}{\dot\lambda_{max}},\; K\text{ 从 }0.5\text{ 起步}
$$


## A-03 《相对运动学与坐标系建模》（域 A 比例导引 PN）


**`F-A-03-01`**


$$
\mathbf{R} = \mathbf{R}_T - \mathbf{R}_M,\quad \mathbf{V}_{rel} = \mathbf{V}_T - \mathbf{V}_M
$$


**`F-A-03-02`**


$$
\frac{d\mathbf{R}}{dt} = \mathbf{V}_{rel},\qquad \frac{d\mathbf{V}_{rel}}{dt} = \mathbf{a}_T - \mathbf{a}_M
$$


**`F-A-03-03`**


$$
\dot R = -V_m\cos\eta_m + V_t\cos\eta_t,\qquad R\dot\lambda = V_m\sin\eta_m - V_t\sin\eta_t
$$


**`F-A-03-04`**


$$
\dot r = V_t\cos(q-\theta_t) - V_m\cos(q-\theta_m),\qquad r\dot q = V_m\sin(q-\theta_m) - V_t\sin(q-\theta_t)
$$


**`F-A-03-05`**


$$
\dot R = -V_c,\qquad \dot\lambda_D = \frac{V_\theta}{R},\qquad \dot\lambda_T = \frac{V_\psi}{R\cos\lambda_D}
$$


**`F-A-03-06`**


$$
C_I^L = \begin{bmatrix}
\cos\lambda_D\cos\lambda_T & \cos\lambda_D\sin\lambda_T & \sin\lambda_D\\
-\sin\lambda_T & \cos\lambda_T & 0\\
-\sin\lambda_D\cos\lambda_T & -\sin\lambda_D\sin\lambda_T & \cos\lambda_D
\end{bmatrix}
$$


## A-04 《脱靶量分析与精度优化技巧》（域 A 比例导引 PN）


**`F-A-04-01`**


$$
\text{ZEM} = \left\|\mathbf{r} - \frac{(\mathbf{r}\cdot\mathbf{v})}{\|\mathbf{v}\|^2}\mathbf{v}\right\|
$$


**`F-A-04-02`**


$$
\text{ZEM}_{PNG} = \frac{V_c t_{go}^2}{2}\dot\lambda,\qquad \text{ZEM}_{APNG} = \frac{V_c t_{go}^2}{2}\dot\lambda + \frac{t_{go}^2}{2}a_{t,\perp}
$$


**`F-A-04-03`**


$$
\dot\lambda_{pred} = \dot\lambda + \ddot\lambda\,\tau \quad\text{(线性外推)}
$$


**`F-A-04-04`**


$$
G_{lead}(s) = \frac{T_1 s + 1}{T_2 s + 1},\; T_1=1.5\tau,\; T_2=0.3\tau \Rightarrow \text{补偿约 }60\%\text{ 相位滞后}
$$


**`F-A-04-05`**


$$
N_{eff} = \frac{N_0}{1 + \tau/T_g}
$$


**`F-A-04-06`**


$$
K = \frac{2V_c N}{R_{min}}\ (\text{留 }10\%\sim20\%\text{ 裕度}),\qquad G(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}
$$


**`F-A-04-07`**


$$
\omega_n = (2\sim3)\times\text{制导回路带宽},\quad \zeta = 0.7\sim1.0,\quad f_s \ge 10 f_c
$$


## A-05 《扩展/增广比例导引（APN/AUGPNG）与机动目标》（域 A 比例导引 PN）


**`F-A-05-01`**


$$
a_{cmd} = N' V_c \dot\lambda + \frac{N'}{2}a_T,\qquad a_{T\perp} = a_T - (a_T\cdot\mathbf{e}_r)\mathbf{e}_r
$$


**`F-A-05-02`**


$$
a_{cmd} = N V_c \dot\lambda + \frac{N}{2}(a_T - a_M)
$$


**`F-A-05-03`**


$$
a_c = N V_c \dot\lambda + f(\lambda, R, V, a_T, \dots)
$$


**`F-A-05-04`**


$$
a_c = N V_c\dot\lambda + K_c\left(\frac{N}{2}\hat a_T\right),\quad K_c\in[0,1]
$$


## A-06 《比例导引与预测制导融合、弹道仿真与工程实现》（域 A 比例导引 PN）


**`F-A-06-01`**


$$
P_{pred} = P_{target}(t_0) + V_{target}\,T_{go} + \tfrac{1}{2}A_{target}\,T_{go}^2
$$


**`F-A-06-02`**


$$
a_{cmd} = w_{pn}\,a_{pn} + w_{pred}\,a_{pred},\qquad w_{pn} + w_{pred} = 1
$$


**`F-A-06-03`**


$$
a_{cmd} = w\,a_{PN} + (1-w)a_{PG},\quad w = \frac{t_{go} - t_{switch\_end}}{t_{switch\_start} - t_{switch\_end}}
$$


**`F-A-06-04`**


$$
\min J = \sum_i (x_{k+i}-x_{ref,k+i})^T Q (x_{k+i}-x_{ref,k+i}) + u_{k+i}^T R\,u_{k+i}
$$


**`F-A-06-05`**


$$
\text{s.t. } x_{k+1}=Ax_k+Bu_k,\quad |u|\le u_{max},\quad |\alpha|\le\alpha_{max},\quad x_N\in X_{terminal}
$$


**`F-A-06-06`**


$$
m\frac{d\mathbf{V}}{dt} = \mathbf{F}_{总},\qquad \mathbf{I}\frac{d\boldsymbol\omega}{dt} + \boldsymbol\omega\times(\mathbf{I}\boldsymbol\omega) = \mathbf{M}_{总}
$$


**`F-A-06-07`**


$$
F_{aero} = \tfrac{1}{2}\rho V^2 S\,C,\qquad M_{aero} = \tfrac{1}{2}\rho V^2 S L\,C_m
$$


## B-01 《最优制导的数学基础 — 变分法、极小值原理与性能指标设计》（域 B 最优制导）


**`F-B-01-01`**


$$
\frac{\partial L}{\partial y} - \frac{d}{dt}\left(\frac{\partial L}{\partial \dot y}\right) = 0
$$


**`F-B-01-02`**


$$
H(x, u, \lambda, t) = L(x, u, t) + \lambda^\top f(x, u, t)
$$


**`F-B-01-03`**


$$
\dot x = \frac{\partial H}{\partial \lambda} = f,\qquad \dot\lambda = -\frac{\partial H}{\partial x},\qquad \lambda(t_f) = \frac{\partial \varphi}{\partial x(t_f)}
$$


**`F-B-01-04`**


$$
u^* = \arg\min_{u\in U} H(x^*, u, \lambda^*, t)
$$


## B-02 《LQR 与 Riccati 方程 — 最优导航比 N=3 的由来与权重矩阵选取》（域 B 最优制导）


**`F-B-02-01`**


$$
\dot x = A(t)x + B(t)u,\qquad J = \tfrac12 x^\top(t_f)S\,x(t_f) + \tfrac12\int_{t_0}^{t_f}\left[x^\top Q x + u^\top R u\right]dt
$$


**`F-B-02-02`**


$$
u^*(t) = -R^{-1}B^\top P(t)\,x(t) = -K(t)x,\qquad K = R^{-1}B^\top P
$$


**`F-B-02-03`**


$$
-\frac{dP}{dt} = A^\top P + P A + Q - P B R^{-1} B^\top P,\qquad P(t_f) = S
$$


**`F-B-02-04`**


$$
A^\top P_\infty + P_\infty A + Q - P_\infty B R^{-1} B^\top P_\infty = 0
$$


**`F-B-02-05`**


$$
x = [y,\; v_y]^\top,\quad \dot y = v_y,\quad \dot v_y = a_T - a_M \;\Rightarrow\; A=\begin{bmatrix}0&1\\0&0\end{bmatrix},\; B=\begin{bmatrix}0\\-1\end{bmatrix}
$$


**`F-B-02-06`**


$$
J = \int (q\,y^2 + r\,u^2)dt + s\,y^2(t_f) \;\Rightarrow\; u^* = -\frac{3q}{r}\,y - \frac{3q}{r}\,t_{go}\,v_y
$$


**`F-B-02-07`**


$$
N_{opt}(t) = 3 + \frac{q_2}{r}\cdot\frac{1}{R^2}\cdot t_{go}^3
$$


**`F-B-02-08`**


$$
\ddot\lambda = -\frac{2\dot R}{R}\dot\lambda - \frac{1}{R}a_M + \frac{1}{R}a_T,\qquad A=\begin{bmatrix}0&1\\0&-\frac{2\dot R}{R}\end{bmatrix},\; B=\begin{bmatrix}0\\-\frac1R\end{bmatrix},\; D=\begin{bmatrix}0\\\frac1R\end{bmatrix}
$$


**`F-B-02-09`**


$$
t_{go} = \frac{R}{V_c} + \frac{0.5\,a_{T,est}\,t_{go}^2}{V_c}\quad(\text{迭代 2\sim3 次}),\qquad t_{go}\in[0.1,\;10]\ \text{s}
$$


**`F-B-02-10`**


$$
Q_{ii} = \frac{1}{(x_{max,i})^2},\qquad R_{jj} = \frac{1}{(u_{max,j})^2}
$$


## B-03 《机动目标拦截 — 目标加速度建模、估计与微分对策制导》（域 B 最优制导）


**`F-B-03-01`**


$$
a_{cmd} = N V_c \dot\lambda + \frac{N}{2}a_{T\perp}
$$


**`F-B-03-02`**


$$
\nu(k) = z(k) - H\hat x(k|k-1),\qquad \varepsilon(k) = \nu^\top(k)S^{-1}(k)\nu(k)
$$


**`F-B-03-03`**


$$
\hat x(k) = \sum_i \mu_i(k)\,\hat x_i(k)\quad(\text{多模型/IMM 加权融合})
$$


**`F-B-03-04`**


$$
\dot x = Ax + Bu + Cv
$$


**`F-B-03-05`**


$$
J = \tfrac12 x^\top(t_f)S\,x(t_f) + \tfrac12\int\left[u^\top R u - \gamma^2 v^\top Q v\right]dt
$$


**`F-B-03-06`**


$$
J(u^*, v) \le J(u^*, v^*) \le J(u, v^*)\quad \forall u, v
$$


**`F-B-03-07`**


$$
-\frac{\partial V}{\partial t} = \min_u\max_v\left\{L + \left(\frac{\partial V}{\partial x}\right)^\top f(x,u,v)\right\}
$$


**`F-B-03-08`**


$$
a_{cmd,i} = N V_{c,i}\dot\lambda_i + \hat a_{T,fusion} + K\left(T_{go,avg} - T_{go,i}\right)
$$


**`F-B-03-09`**


$$
s = (T_{go,i}-T_{go,j}) + k(\lambda_i-\lambda_j),\qquad \dot s = -\varepsilon\,\mathrm{sign}(s) - k s
$$


## B-04 《最优制导律数值求解 — TPBVP、打靶法、伪谱法与非线性近似》（域 B 最优制导）


**`F-B-04-01`**


$$
\dot x = f(x,u,t),\; x(t_0)=x_0 \qquad \dot\lambda = -\frac{\partial H}{\partial x} = -\frac{\partial L}{\partial x} - \left(\frac{\partial f}{\partial x}\right)^\top\lambda,\; \lambda(t_f)=\frac{\partial\varphi}{\partial x(t_f)}
$$


**`F-B-04-02`**


$$
x(t_i^+)=x(t_i^-),\quad \lambda(t_i^+)=\lambda(t_i^-),\qquad s_i = [x(t_i),\;\lambda(t_i)]^\top \Rightarrow \text{求解规模 } N\times 2n
$$


**`F-B-04-03`**


$$
x_{k+1} - x_k - \frac{h}{2}\left(f_k + f_{k+1}\right) = 0
$$


**`F-B-04-04`**


$$
A^\top(x)P + P A(x) - P B(x)R^{-1}B^\top(x)P + Q = 0,\qquad u^* = -R^{-1}B^\top(x)P(x)x
$$


**`F-B-04-05`**


$$
P(x,\theta) = \sum_{i=0}^{\infty} T_i(x)\theta^i
$$


**`F-B-04-06`**


$$
x_{k+1} = x_k - \alpha_k \nabla J(x_k),\qquad \beta_k^{FR} = \frac{\nabla J_{k+1}^\top \nabla J_{k+1}}{\nabla J_k^\top \nabla J_k},\qquad d_{k+1} = -\nabla J_{k+1} + \beta_k d_k
$$


**`F-B-04-07`**


$$
\text{Armijo: } J(x+\alpha d) \le J(x) + c\,\alpha\,\nabla J^\top d,\quad c=0.1,\; \rho=0.5
$$


## B-05 《代码实现与工程化 — 离散化、实时性与在线求解可行性》（域 B 最优制导）


**`F-B-05-01`**


$$
a_{cmd} = N\,V\,\dot\lambda,\qquad a_{cmd} = \mathrm{clip}(a_{cmd}, -a_{max}, +a_{max})
$$


**`F-B-05-02`**


$$
\min J = \int_{t_k}^{t_k+T_p}\left[(y-y_{ref})^\top Q(y-y_{ref}) + u^\top R u\right]dt \quad \text{s.t. } |u|\le u_{max},\; |\alpha|\le\alpha_{max}
$$


**`F-B-05-03`**


$$
\min\ J + \rho\varepsilon^2,\qquad \text{s.t. } |u| \le u_{max} + \varepsilon,\; \varepsilon\ge 0
$$


**`F-B-05-04`**


$$
G_{ap} = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}\ (\omega_n=15\sim30\ \text{rad/s},\ \zeta=0.5\sim0.8),\qquad G_{seeker} = \frac{1}{\tau s + 1}\ (\tau\approx0.02\ \text{s})
$$


## B-06 《多导弹协同最优制导 — 协同架构、攻击时间一致性与任务分配》（域 B 最优制导）


**`F-B-06-01`**


$$
\min J = w_1\|t_{go}-T_d\|^2 + w_2\|\theta-\theta_d\|^2 + w_3\|\mathbf p - \mathbf p_d\|^2 \quad \text{s.t. 动力学、控制约束、通信约束}
$$


**`F-B-06-02`**


$$
\dot x_i = \sum_{j\in N_i} a_{ij}\left(x_j - x_i\right),\qquad x(k+1) = (I - \varepsilon L)x(k),\qquad L = D - A
$$


**`F-B-06-03`**


$$
\dot\tau = -L\tau \;\Rightarrow\; \tau(t) = e^{-Lt}\,\tau(0)
$$


**`F-B-06-04`**


$$
a_{cmd} = \underbrace{N V\dot\lambda}_{\text{PN 项}} + \underbrace{K_t\,(T_{go} - T_{go,des})}_{\text{时间修正}},\qquad K_t = 0.5\sim2.0
$$


**`F-B-06-05`**


$$
a_{cmd} = N V\dot\lambda + K_\theta\,(\theta_d - \theta),\qquad K_\theta = 2\sim6,\quad |\theta_d-\theta|\le 30^\circ
$$


**`F-B-06-06`**


$$
a(t) = N_1 V\dot\lambda + N_2 V\frac{\gamma - \gamma_d}{t_{go}},\qquad N_1 = 3\sim5,\quad N_2 = 1\sim2
$$


**`F-B-06-07`**


$$
a_i = k_1(v_i - v_{des}) + k_2\sum_{j\in N_i}(\tau_j - \tau_i)
$$


**`F-B-06-08`**


$$
\max J = \sum_{i=1}^{M}\sum_{j=1}^{N} x_{ij}\left(w_1 V_j - w_2 T_{ij} - w_3 C_{ij}\right),\quad \sum_j x_{ij}=1,\ \sum_i x_{ij}\ge 1,\ x_{ij}\in\{0,1\}
$$


**`F-B-06-09`**


$$
\min\ \max_i J_i(t),\qquad a_{cmd} = \left(N V\dot\lambda\right)\left(1 + k\frac{J_i - J_{avg}}{J_{avg}}\right),\quad k = 0.3\sim0.5
$$


## C-01 《导引头建模与视场/框架角约束》（域 C 终端制导）


**`F-C-01-01`**


$$
|q - \varepsilon| \le \theta_{FOV},\qquad |{\lambda} - \theta_m| \le \theta_{max}
$$


**`F-C-01-02`**


$$
\varepsilon = \lambda - \theta_m,\qquad \text{丢失条件：}\ |{\lambda} - \theta_m| > \theta_{max}
$$


**`F-C-01-03`**


$$
\cos\varphi = \frac{\mathbf{t}\cdot\mathbf{r}}{\|\mathbf{t}\|\cdot\|\mathbf{r}\|} \ge \cos\theta_{FOV}\ \xrightarrow{\ \|\mathbf{r}\|=1\ }\ \mathbf{t}\cdot\mathbf{r} \ge \|\mathbf{t}\|\cos\theta_{FOV}
$$


## C-02 《视场约束下的制导律设计（BPN / 李雅普诺夫 / MPC / 虚拟目标）》（域 C 终端制导）


**`F-C-02-01`**


$$
a_{cmd} = N\,V\,\dot\lambda + a_{bias},\qquad a_{bias} = K_{bias}\,(\theta_{current} - \theta_{ref})
$$


**`F-C-02-02`**


$$
V = \frac{1}{\lambda_{max}^2 - \lambda^2},\qquad \dot V = \frac{2\lambda}{(\lambda_{max}^2-\lambda^2)^2}\dot\lambda
$$


**`F-C-02-03`**


$$
\dot\lambda = -k\,\lambda\,(\lambda_{max}^2-\lambda^2)^2 \;\Rightarrow\; \dot V = -2k\lambda^2 < 0
$$


**`F-C-02-04`**


$$
J = \sum\left(x^TQx + u^TRu\right) + \rho\,\varepsilon_{slack}^2,\qquad \text{s.t. } |q_{az}|\le 30^\circ+\varepsilon_{slack}
$$


**`F-C-02-05`**


$$
J = \min\Big[r(t_f)^2 + \int_0^{t_f}\big(a_m^2 + w\,g(\varepsilon)\big)dt\Big],\qquad g(\varepsilon) = \frac{1}{(\varepsilon_{max}-|\varepsilon|)^2}
$$


## C-03 《末端攻角/落角约束控制（LQR / 滑模 / 自适应 / 鲁棒）》（域 C 终端制导）


**`F-C-03-01`**


$$
n = \frac{q\,S\,C_L(\alpha)}{m\,g},\qquad T_w = T_\infty\Big(1 + r\,\frac{\gamma-1}{2}Ma^2\Big),\qquad C_{m\delta} \approx \frac{S_\delta L_\delta}{S\,b_A}
$$


**`F-C-03-02`**


$$
J = \tfrac12 x(t_f)^T S\,x(t_f) + \tfrac12\int\big(x^TQx + u^TRu\big)dt,\qquad u^* = -R^{-1}B^T P(t)\,x
$$


**`F-C-03-03`**


$$
-\dot P = A^TP + PA + Q - PBR^{-1}B^TP,\quad P(t_f) = S
$$


**`F-C-03-04`**


$$
e_\alpha = \alpha - \alpha_{max},\qquad s = e_\alpha + \lambda\,\dot e_\alpha,\qquad u = -k_1 s - k_2\,\mathrm{sign}(s)
$$


**`F-C-03-05`**


$$
V = \tfrac12 s^2,\quad \dot V \le -k_1 s^2 \le 0\ \ (\text{条件：}k_2 > |d(t)|_{max})
$$


**`F-C-03-06`**


$$
V(\alpha) = \frac{(\alpha-\alpha_{max})^2}{(\alpha_{max}-\alpha)^2},\qquad s_{new} = s + \frac{\partial V}{\partial \alpha}k_v
$$


**`F-C-03-07`**


$$
\min\gamma\quad\text{s.t.}\quad\begin{bmatrix}A_{cl}^TP + PA_{cl} + Q & P\\ P & -\gamma I\end{bmatrix}<0,\qquad \begin{bmatrix}P & C^T\\ C & \alpha_{max}^2\end{bmatrix}\ge 0
$$


## C-04 《末端能量管理与可用过载预测》（域 C 终端制导）


**`F-C-04-01`**


$$
E_h = h + \frac{V^2}{2g},\qquad e = \tfrac12 V^2 + g\,h\ \ (\mathrm{J/kg}),\qquad \Delta e = e_{actual} - e_{desired}
$$


**`F-C-04-02`**


$$
h_{ref}(R) = h_f + (h_0-h_f)\Big(\frac{R}{R_0}\Big)^{n},\qquad V_{ref}(R) = \sqrt{2g\big(E_{h,ref}(R) - h_{ref}(R)\big)}
$$


**`F-C-04-03`**


$$
\alpha_{cmd} = \alpha_{trim} + K_e\,\Delta e + K_d\,\frac{d(\Delta e)}{dt}
$$


**`F-C-04-04`**


$$
q = \tfrac12\rho V^2 \le q_{max},\qquad \dot Q = k\,\rho^{0.5}\,V^{3.15} \le \dot Q_{max}
$$


**`F-C-04-05`**


$$
\Delta D = D_0\Big[\big(\alpha/\alpha_0\big)^2\cos\gamma + k\big(1-\cos\mu\big)\Big],\quad k=0.3\sim0.5
$$


## C-05 《末端速度矢量控制（最优 / 滑模 / 动态逆）》（域 C 终端制导）


**`F-C-05-01`**


$$
V_{target}=V_{desired}\pm\Delta V,\quad \theta_{target}=\theta_{desired}\pm\Delta\theta,\quad |n|\le n_{max},\quad |\alpha|\le\alpha_{max},\quad |\delta|\le\delta_{max}
$$


**`F-C-05-02`**


$$
a_m = N_1\frac{y}{t_{go}^2} + N_2\frac{\dot y}{t_{go}} + N_3\frac{V_m(\theta_m-\theta_d)}{t_{go}}
$$


**`F-C-05-03`**


$$
J = \tfrac12\Big[y(t_f)^2 + \rho\big(v_y(t_f)-v_d\big)^2\Big] + \tfrac12\int R\,u^2 dt,\qquad u^* = -R^{-1}B^TP(t)x
$$


**`F-C-05-04`**


$$
e_\theta = \theta-\theta_d,\quad e_V = V-V_d,\quad s = c_1 e_\theta + c_2 e_V + \dot e_\theta
$$


**`F-C-05-05`**


$$
u = u_{eq} + u_{sw},\quad u_{sw} = -k\,\mathrm{sign}(s),\quad V=\tfrac12 s^2,\ \dot V = -k|s|\le 0
$$


**`F-C-05-06`**


$$
\dot V = \frac{T\cos\alpha - D}{m} - g\sin\gamma,\qquad \dot\gamma = \frac{L + T\sin\alpha}{mV} - \frac{g\cos\gamma}{V}
$$


**`F-C-05-07`**


$$
T_{des} = \frac{m(\dot V_{des}+g\sin\gamma)+D}{\cos\alpha},\qquad \dot V_{des}=\omega_v(V_{cmd}-V),\quad \dot\gamma_{des}=\omega_\gamma(\gamma_{cmd}-\gamma)
$$


## C-06 《多约束耦合与工程实现（权衡、一体化、验证）》（域 C 终端制导）


**`F-C-06-01`**


$$
J = \sum_i w_i C_i(x),\qquad \Omega = \{x\,|\,C_i(x)\le 0,\ \forall i\}
$$


**`F-C-06-02`**


$$
A=\begin{bmatrix}0&1&0&0&0\\0&-1/\tau_g&0&0&0\\0&0&-a_\alpha&1&a_\delta\\0&0&-a_q&0&0\\0&0&0&0&-1/\tau_a\end{bmatrix},\quad B=[0,\ N/\tau_g,\ 0,\ 0,\ K_p/\tau_a]^T
$$


## D-01 《制导回路分层建模与降阶——从六自由度到等效时间常数》（域 D 制导系统建模仿真）


**`F-D-01-01`**


$$
m\left(\frac{d\mathbf{V}}{dt} + \boldsymbol\omega\times\mathbf{V}\right) = \mathbf{F}_{aero} + \mathbf{F}_{thrust} + m\mathbf{g}
$$


**`F-D-01-02`**


$$
\mathbf{I}\frac{d\boldsymbol\omega}{dt} + \boldsymbol\omega\times(\mathbf{I}\boldsymbol\omega) = \mathbf{M}_{aero} + \mathbf{M}_{thrust}
$$


**`F-D-01-03`**


$$
C_L = C_{L0} + C_{L\alpha}\alpha + C_{L\delta_e}\delta_e + C_{Lq}\frac{qc}{2V},\qquad C_D = C_{D0} + C_{D\alpha}\alpha^2 + C_{D\delta_e}\delta_e^2
$$


**`F-D-01-04`**


$$
C_m = C_{m0} + C_{m\alpha}\alpha + C_{m\delta_e}\delta_e + C_{mq}\frac{qc}{2V}
$$


**`F-D-01-05`**


$$
\Delta\dot\alpha = Z_\alpha\Delta\alpha + \Delta q + Z_\delta\Delta\delta,\quad \Delta\dot q = M_\alpha\Delta\alpha + M_q\Delta q + M_\delta\Delta\delta,\quad \Delta\dot\theta = \Delta q
$$


**`F-D-01-06`**


$$
G(s)=\frac{\Delta\theta(s)}{\Delta\delta(s)}=\frac{M_\delta s + Z_\delta M_\alpha - M_\delta Z_\alpha}{s^3-(Z_\alpha+M_q)s^2+(Z_\alpha M_q-M_\alpha)s}
$$


**`F-D-01-07`**


$$
\text{步长} < \frac{1}{10}\times(\text{系统最小时间常数}),\qquad f_s > 2f_{max}\ (\text{工程取}\ 5\sim10\ \text{倍})
$$


## D-02 《制导回路稳定性分析——根轨迹、频域裕度与失稳机理》（域 D 制导系统建模仿真）


**`F-D-02-01`**


$$
G_m = -20\lg|G(j\omega_p)H(j\omega_p)|\ \ (\angle GH = -180^\circ\ \text{处})
$$


**`F-D-02-02`**


$$
P_m = 180^\circ + \angle G(j\omega_c)H(j\omega_c)\ \ (|GH| = 1\ \text{处})
$$


**`F-D-02-03`**


$$
\zeta = \frac{-\sigma}{\sqrt{\sigma^2+\omega^2}},\quad \omega_n=\sqrt{\sigma^2+\omega^2},\quad M_p=e^{-\pi\zeta/\sqrt{1-\zeta^2}},\quad t_s\approx\frac{4}{\zeta\omega_n}
$$


**`F-D-02-04`**


$$
G_c^{lead}(s)=K_c\frac{1+\alpha Ts}{1+Ts}\ (\alpha>1),\quad \varphi_{max}=\arcsin\frac{\alpha-1}{\alpha+1},\quad \omega_m=\frac{1}{T\sqrt\alpha}
$$


**`F-D-02-05`**


$$
G_c^{lag}(s)=K_c\frac{1+Ts}{1+\beta Ts}\ (\beta>1),\quad \omega_m=\frac{1}{T\sqrt\beta}
$$


**`F-D-02-06`**


$$
z = e^{sT},\quad \text{稳定} \iff |z_i|<1,\quad \text{双线性变换}\ z=\frac{1+w}{1-w}
$$


## D-03 《制导系统参数整定——整定流程、调参顺序与指标权衡》（域 D 制导系统建模仿真）


**`F-D-03-01`**


$$
\zeta\approx0.707\ (\text{对应}\ 5\%\ \text{超调}),\quad \omega_n\ge\frac{4}{\zeta t_s},\quad s=-\zeta\omega_n\pm j\omega_n\sqrt{1-\zeta^2}
$$


**`F-D-03-02`**


$$
\delta = K_p(a_{cmd}-a_{actual}) + K_i\int(a_{cmd}-a_{actual})dt
$$


## D-04 《半实物仿真系统搭建——三层两环架构、实时性、时序同步与可信度》（域 D 制导系统建模仿真）


**`F-D-04-01`**


$$
u_{comp} = u_{cmd} + (u_{cmd}-u_{prev})\frac{T_{delay}}{dt}\quad(\text{一阶外推})
$$


## D-05 《误差建模、蒙特卡洛与精度评估》（域 D 制导系统建模仿真）


**`F-D-05-01`**


$$
\tilde a = a_{true} + b_a + \sigma_a w,\qquad \tilde\omega = \omega_{true} + b_g + \sigma_g w
$$


**`F-D-05-02`**


$$
e_{pos}[k] = e_{pos}[k-1]\,e^{-dt/\tau} + \sigma_{pos}\,w[k]
$$


**`F-D-05-03`**


$$
r=\sqrt{x^2+y^2+z^2}+w_r,\quad \theta=\arctan\frac{y}{x}+w_\theta,\quad \varphi=\arctan\frac{z}{\sqrt{x^2+y^2}}+w_\varphi
$$


**`F-D-05-04`**


$$
N\ge1000\ (\text{高维/强非线性}\ 5000\sim10000),\quad \text{关键看}\ P_{95}\ \text{而非均值}
$$


**`F-D-05-05`**


$$
\text{RMSE}=\sqrt{\text{mean}((y_{act}-y_{ideal})^2)},\qquad \text{CEP}\approx1.1774\sigma
$$


## D-06 《抗干扰设计——干扰机理、多域协同与性能评估》（域 D 制导系统建模仿真）


**`F-D-06-01`**


$$
J_{noise}(t)=A\,n(t)\cos(2\pi f_j t+\varphi)\quad(\text{宽带噪声压制})
$$


**`F-D-06-02`**


$$
J_{spot}(t)=\sqrt{2P_j}\cos\big(2\pi f_s t+\varphi_j(t)\big)\quad(\text{瞄准式，功率利用率是宽带的几十倍})
$$


**`F-D-06-03`**


$$
J_{range}(t)=s(t-\Delta\tau)e^{j2\pi f_d t}\quad(\text{距离欺骗：人为延迟}\ \Delta\tau)
$$


**`F-D-06-04`**


$$
J_{vel}(t)=s(t)e^{j2\pi\Delta f t}\quad(\text{速度欺骗：假多普勒，功率只需真信号的}\ 1/10)
$$


**`F-D-06-05`**


$$
J_{comp}=\alpha J_{noise}+\beta J_{deception},\quad \alpha=0.7\sim0.9,\ \beta=0.1\sim0.3
$$


**`F-D-06-06`**


$$
\min_w\ w^H R w\quad \text{s.t.}\ w^H a(\theta_0)=1\ \Rightarrow\ w_{opt}=\frac{R^{-1}a(\theta_0)}{a(\theta_0)^H R^{-1}a(\theta_0)}
$$


**`F-D-06-07`**


$$
D_{null}=20\lg\frac{|F(\theta_j)|}{|F(\theta_s)|}\quad(\text{好的算法应达} -40\ \text{dB})
$$


**`F-D-06-08`**


$$
\hat d = \frac{u - Js\omega}{P_n(s)}\quad(\text{观测器带宽取伺服带宽的}\ 3\sim5\ \text{倍})
$$


## E-01 《轨迹规划方法全景：图搜索 / 采样 / 参数化优化 / 微分平坦与约束建模》（域 E 飞行器轨迹控制）


**`F-E-01-01`**


$$
f(n) = g(n) + h(n),\quad h(n)=\|n-\text{goal}\|_2,\quad \text{step cost}=\begin{cases}1.0 & \text{直邻}\\ 1.414 & \text{斜邻}\end{cases}
$$


**`F-E-01-02`**


$$
J = \int_0^T \left(\frac{d^4 p}{dt^4}\right)^2 dt = c^{T}Qc,\qquad \min_c\ c^TQc \ \ \text{s.t.}\ A c = b
$$


**`F-E-01-03`**


$$
c_f = -\left(P^{T}QP\right)^{-1}P^{T}Qc_0
$$


**`F-E-01-04`**


$$
N_{i,k}(t)=\frac{t-t_i}{t_{i+k}-t_i}N_{i,k-1}(t)+\frac{t_{i+k+1}-t}{t_{i+k+1}-t_{i+1}}N_{i+1,k-1}(t)
$$


**`F-E-01-05`**


$$
\sigma = [x,\ y,\ z,\ \psi]^T \;\Rightarrow\; v,\ a \;\Rightarrow\; R_d e_3 = \frac{a_{des}+g e_3}{\|a_{des}+g e_3\|} \;\Rightarrow\; \text{推力} f,\ \text{力矩} \tau
$$


**`F-E-01-06`**


$$
\text{凸包 } CH(P)=\{x\mid a_i^Tx\le b_i,\ i=1..m\}
$$


**`F-E-01-07`**


$$
U_{rep}(q)=\begin{cases}\tfrac12 k_{rep}\left(\tfrac1\rho-\tfrac1{\rho_0}\right)^2 & \rho\le\rho_0\\ 0 & \rho>\rho_0\end{cases}
$$


**`F-E-01-08`**


$$
J_{total}=\lambda_1 J_{smooth}+\lambda_2 J_{safe}+\lambda_3 J_{dyn}
$$


**`F-E-01-09`**


$$
J_{smooth}=\int\left(w_1\|\dot p\|^2+w_2\|\ddot p\|^2+w_3\|\dddot p\|^2\right)dt,\quad J_{safe}=\sum_i e^{-\alpha d_i},\quad J_{dyn}=\sum_i\!\left[\max(0,|v_i|-v_{max})^2+\max(0,|a_i|-a_{max})^2\right]
$$


**`F-E-01-10`**


$$
T=\int_0^{s_f}\frac{ds}{v(s)}=\int_0^{s_f}\frac{ds}{\sqrt{b(s)}},\quad \text{s.t. } \frac{db}{ds}\le 2a_{max}\sqrt{b}
$$


**`F-E-01-11`**


$$
\alpha^\* = \max\left\{\frac{\max|v|}{v_{max}},\ \sqrt{\frac{\max|a|}{a_{max}}}\right\}
$$


**`F-E-01-12`**


$$
\text{Score}=w_s S_{smooth}+w_t S_{time}+w_e S_{energy}+w_{saf}S_{safety}-p_{viol}
$$


**`F-E-01-13`**


$$
E_{total}=\int\left(c_1\|v\|^2+c_2\|a\|^2+c_3\left|\dot h\right|\right)dt,\qquad J_{safety}=\int\frac{dt}{d_{obs}^2}
$$


## E-02 《轨迹跟踪控制律：串级PID / LQR / MPC / 反步 / 滑模 / 几何控制 / 前瞻制导的适用边界与整定》（域 E 飞行器轨迹控制）


**`F-E-02-01`**


$$
u=K_pe+K_i\!\int\! e\,dt+K_d\frac{de}{dt},\quad \text{姿态环 200\textasciitilde400 Hz} > \text{速度环 100\textasciitilde200 Hz} > \text{位置环 50\textasciitilde100 Hz}
$$


**`F-E-02-02`**


$$
J=\int_0^\infty (x^TQx+u^TRu)dt,\qquad A^TP+PA-PBR^{-1}B^TP+Q=0,\qquad K=R^{-1}B^TP
$$


**`F-E-02-03`**


$$
\min\sum_{i=0}^{N-1}\Big[(x_{k+i}-x_{ref})^TQ(x_{k+i}-x_{ref})+(u_{k+i}-u_{ref})^TR(u_{k+i}-u_{ref})+\Delta u^T S\Delta u\Big]
$$


**`F-E-02-04`**


$$
\text{s.t. } x_{k+i+1}=Ax_{k+i}+Bu_{k+i}+d_{k+i},\quad u_{min}\le u\le u_{max},\quad x_{min}\le x\le x_{max}
$$


**`F-E-02-05`**


$$
\alpha_1=-c_1z_1+\dot x_{1d},\qquad u=\frac{1}{g_n}\left(-f_n+\dot\alpha_{n-1}-c_nz_n-z_{n-1}\right)
$$


**`F-E-02-06`**


$$
s=\dot e+ce\ \Rightarrow\ \dot e=-ce\ (s=0),\qquad \dot s=-k\,s-\varepsilon\,\mathrm{sign}(s)
$$


**`F-E-02-07`**


$$
e_p=p-p_d,\quad R_e=R_d^TR,\quad e_R=\frac{1}{2}\left(R_e-R_e^T\right)^\vee
$$


**`F-E-02-08`**


$$
\tau=-K_Re_R-K_\omega e_\omega+\omega\times J\omega\ (\text{前馈抵消陀螺力矩}),\qquad f=-\left(a_{des}+g e_3\right)\cdot R e_3
$$


**`F-E-02-09`**


$$
p_{target}=p_d\!\left(t+\frac{L}{V}\right),\qquad a_{des}=k\,\frac{p_{target}-p}{L}
$$


## E-03 《轨迹控制算法工程化：离散化、代码架构、实时性、数值稳健与异常处理》（域 E 飞行器轨迹控制）


**`F-E-03-01`**


$$
f_s \ge 10\, f_{BW}\quad(\text{MPC/滑模/反步/高阶滑模通用})
$$


**`F-E-03-02`**


$$
K(k)=\frac{P(k-1)h(k)}{1+h^T(k)P(k-1)h(k)},\qquad \hat\theta(k)=\hat\theta(k-1)+K(k)\left[z(k)-h^T(k)\hat\theta(k-1)\right]
$$


**`F-E-03-03`**


$$
v = B\,u,\qquad u = B^\dagger v = (B^TWB)^{-1}B^TW v\ (\text{加权伪逆})
$$


**`F-E-03-04`**


$$
\Delta t_{total}=\Delta t_{trans}+\Delta t_{calc}+\Delta t_{DA},\qquad \text{补偿有效条件：}\ \Delta t < \tfrac{1}{10}T_{dom}
$$


**`F-E-03-05`**


$$
\kappa(P)=\frac{\lambda_{max}}{\lambda_{min}},\quad P=L L^T\ (\text{Cholesky}),\quad \text{rank}(A_{con})=\#\text{约束}
$$


## E-04 《鲁棒控制与抗扰设计：不确定性建模、滑模/ESO/DOBC/H∞-μ/LPV 与风场抑制》（域 E 飞行器轨迹控制）


**`F-E-04-01`**


$$
m(\dot V+\omega\times V)=F_g+F_{aero}+F_T,\qquad J\dot\omega+\omega\times(J\omega)=M_{aero}+M_T
$$


**`F-E-04-02`**


$$
C_L=C_{L0}+\Delta C_L,\ \|\Delta C_L\|\le\delta\quad(\text{摄动有界})
$$


**`F-E-04-03`**


$$
s=\dot e+ce,\qquad u=u_{eq}-k\,s-\varepsilon\,\mathrm{sat}(s/\Phi)
$$


**`F-E-04-04`**


$$
u=-k_1|s|^{1/2}\mathrm{sign}(s)+v,\qquad \dot v=-k_2\,\mathrm{sign}(s)
$$


**`F-E-04-05`**


$$
\dot z_1=z_2-\beta_1 e,\quad \dot z_2=z_3+b_0u-\beta_2\,\mathrm{fal}(e,0.5,\delta),\quad \dot z_3=-\beta_3\,\mathrm{fal}(e,0.25,\delta)
$$


**`F-E-04-06`**


$$
\text{带宽法：}\ \beta_1=3\omega_o,\ \beta_2=3\omega_o^2,\ \beta_3=\omega_o^3\quad(\text{二阶形式：}\beta_1=2\omega_o,\beta_2=\omega_o^2)
$$


**`F-E-04-07`**


$$
u_0=k_p(x_1-z_1)+k_d(x_2-z_2),\qquad u=u_0-z_3/b_0
$$


**`F-E-04-08`**


$$
\hat d = Q(s)\left(P^{-1}(s)\,y-u\right),\qquad u_{total}=u_{nominal}-\hat d
$$


**`F-E-04-09`**


$$
\min_K\left\|\begin{bmatrix}W_S S\\ W_T T\end{bmatrix}\right\|_\infty < \gamma,\qquad S=(I+PK)^{-1},\ T=PK(I+PK)^{-1}
$$


**`F-E-04-10`**


$$
G_u(s)=\sigma_u\sqrt{\frac{2L_u}{\pi V}}\frac{1}{1+(L_u/V)s},\qquad G_w(s)=\sigma_w\sqrt{\frac{2L_w}{\pi V}}\frac{1+\sqrt3(L_w/V)s}{\left(1+(L_w/V)s\right)^2}
$$


**`F-E-04-11`**


$$
\text{NDI: } u=g^{-1}(x)\left(\nu-f(x)\right),\qquad \text{INDI: } \Delta u=g^{-1}(x_0)\left(\nu-\dot y_0\right)
$$


## E-05 《气动参数辨识：激励设计、数据预处理、时域/频域方法、可辨识性与结果验证》（域 E 飞行器轨迹控制）


**`F-E-05-01`**


$$
L=\tfrac12\rho V^2S\,C_L,\qquad C_L=C_{L0}+C_{L\alpha}\alpha+C_{L\delta_e}\delta_e+C_{Lq}\frac{q\bar c}{2V}
$$


**`F-E-05-02`**


$$
C_m=C_{m0}+C_{m\alpha}\alpha+C_{m\delta_e}\delta_e+C_{m q}\frac{q\bar c}{2V}
$$


**`F-E-05-03`**


$$
C_l=C_{l0}+C_{l\beta}\beta+C_{lp}\frac{pb}{2V}+C_{lr}\frac{rb}{2V}+C_{l\delta_a}\delta_a+C_{l\delta_r}\delta_r
$$


**`F-E-05-04`**


$$
C_n=C_{n0}+C_{n\beta}\beta+C_{np}\frac{pb}{2V}+C_{nr}\frac{rb}{2V}+C_{n\delta_a}\delta_a+C_{n\delta_r}\delta_r
$$


**`F-E-05-05`**


$$
z=H\theta+v,\qquad \hat\theta_{LS}=(H^TH)^{-1}H^Tz
$$


**`F-E-05-06`**


$$
\hat\theta_{WLS}=(H^TWH)^{-1}H^TWz,\quad W=R^{-1}\ (\text{马尔可夫估计，最小方差线性无偏})
$$


**`F-E-05-07`**


$$
K(k)=\frac{P(k-1)h(k)}{1+h^T(k)P(k-1)h(k)},\quad \hat\theta(k)=\hat\theta(k-1)+K(k)\left[z(k)-h^T(k)\hat\theta(k-1)\right]
$$


**`F-E-05-08`**


$$
\ln L(\theta)=-\frac N2\ln(2\pi R)-\frac1{2R}\sum_{k=1}^N\left(z_k-h(x_k,\theta)\right)^2
$$


**`F-E-05-09`**


$$
\text{Var}(\hat\theta)\ge \mathcal{I}^{-1}(\theta),\qquad \mathcal{I}(\theta)=\frac1R\sum_k\left(\frac{\partial h}{\partial\theta}\right)^2
$$


**`F-E-05-10`**


$$
\text{高斯-牛顿：}\ \theta_{new}=\theta-H^{-1}g,\quad H\approx-\frac1R\sum_k\left(\frac{\partial h}{\partial\theta}\right)^2
$$


**`F-E-05-11`**


$$
X[k]=\sum_n x[n]e^{-j2\pi kn/N},\quad \Delta f=f_s/N
$$


**`F-E-05-12`**


$$
\gamma^2_{xy}(f)=\frac{|G_{xy}(f)|^2}{G_{xx}(f)G_{yy}(f)}\in[0,1],\qquad H_1(f)=\frac{G_{xy}(f)}{G_{xx}(f)}
$$


**`F-E-05-13`**


$$
\text{3211：方波脉冲宽度比 } 3:2:1:1,\qquad \text{扫频：}\ \phi(t)=2\pi f_0\frac{k^t-1}{\ln k}
$$


## E-06 《建模—辨识—控制的闭环迭代：数据融合、模型更新、仿真-试验一致性与蒙特卡洛鲁棒性评估》（域 E 飞行器轨迹控制）


**`F-E-06-01`**


$$
\text{加权平均：}\ C_{L,fused}=w\,C_{L,flight}+(1-w)\,C_{L,WT}
$$


**`F-E-06-02`**


$$
\text{贝叶斯：}\ P(\theta\mid D_{flight})\propto P(D_{flight}\mid\theta)\,P(\theta\mid D_{WT})
$$


**`F-E-06-03`**


$$
\text{深度网络：}\ [C_L,C_D,C_m]=f_{NN}(\alpha,\ \beta,\ Ma,\ p,q,r)
$$


**`F-E-06-04`**


$$
\hat\theta^{(i)}\sim\mathcal{N}(\hat\theta,\ \Sigma_\theta),\quad i=1..N\ \Rightarrow\ \text{输出分布} \to \text{95\% 置信区间}
$$


**`F-E-06-05`**


$$
\text{协方差分析（一阶）：}\ \Sigma_y\approx J_\theta\,\Sigma_\theta\,J_\theta^T
$$


**`F-E-06-06`**


$$
\text{Bootstrap：}\ \text{CI}_{95}=\left[\text{pct}_{2.5}(\bar\theta^\*),\ \text{pct}_{97.5}(\bar\theta^\*)\right]
$$


**`F-E-06-07`**


$$
\text{增益调度：}\ K=K(q,\ Ma,\ \alpha)\ \text{二维查表}
$$


**`F-E-06-08`**


$$
\text{MRAC：}\ y\to y_m,\quad \dot\theta_\Delta=-\Gamma\,e^TPB\,\phi(x)
$$


**`F-E-06-09`**


$$
\text{MPC：}\ \min\sum\|y_{ref}-y_{pred}\|^2_Q+\|u\|^2_R,\quad y_{pred}\ \text{由气动模型积分得到}
$$


**`F-E-06-10`**


$$
C_L(\alpha,\delta_e,q)=C_{L0}+C_{L\alpha}\alpha+C_{L\delta_e}\delta_e+C_{Lq}q,\qquad C_D=C_{D0}+C_{D\alpha}\alpha+C_{D,C_L}C_L^2
$$


**`F-E-06-11`**


$$
u_{k+1}^{ff}=u_k^{ff}+L\,e_k(t),\qquad \dot{\hat\theta}=-\Gamma\frac{\phi\,e}{\alpha+\phi^T\phi}
$$
