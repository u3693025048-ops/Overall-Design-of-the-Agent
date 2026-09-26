#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 book010 导弹制导蒸馏交互导航页 (distill/index.html)

用法:
  /Users/lisen/.workbuddy/binaries/python/envs/default/bin/python3 distill/_gen_index.py

特性:
  - 自动扫描 distill/domain_*.md, 解析 RIA++ 卡片
  - 卡片全文在 Python 端渲染为 HTML 内嵌, 前端零依赖
  - 三视图: 卡片浏览 / 速查表 / 600 章知识地图
"""
import json
import os
import re
import html

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

# ---------------------------------------------------------------- 域定义
DOMAINS = {
    'A': {
        'name': '比例导引 PN',
        'file': 'domain_A_PN.md',
        'color': '#185fa5',
        'courses': ['《比例导引法：从原理到弹道仿真》',
                    '《比例导引与扩展比例导引对比分析》',
                    '《比例导引法脱靶量优化技巧》',
                    '《比例导引与预测制导融合技术》'],
    },
    'B': {
        'name': '最优制导',
        'file': 'domain_B_Optimal.md',
        'color': '#0f6e56',
        'courses': ['《最优制导律设计全流程解析》',
                    '《最优制导在机动目标拦截中的应用》',
                    '《最优制导律数值求解与代码实现》',
                    '《最优制导在多导弹协同中的运用》'],
    },
    'C': {
        'name': '终端制导',
        'file': 'domain_C_Terminal.md',
        'color': '#993c1d',
        'courses': ['《终端制导导引头视场约束处理》',
                    '《终端制导末端攻角约束控制》',
                    '《终端制导末端能量管理实战》',
                    '《终端制导末端速度矢量控制》'],
    },
    'D': {
        'name': '制导系统建模与仿真',
        'file': 'domain_D_SystemModeling.md',
        'color': '#694f8a',
        'courses': ['《导弹制导系统建模与参数整定》',
                    '《导弹制导回路稳定性分析与调试》',
                    '《导弹制导系统半实物仿真搭建实战课程》',
                    '《导弹制导系统抗干扰设计方法》'],
    },
    'E': {
        'name': '飞行器轨迹控制',
        'file': 'domain_E_Trajectory.md',
        'color': '#8a5a1a',
        'courses': ['《飞行器轨迹规划与跟踪控制实战》',
                    '《飞行器轨迹控制算法工程化实战》',
                    '《飞行器轨迹控制鲁棒性提升策略》',
                    '《飞行器轨迹控制中的气动参数辨识》'],
    },
}

# ------------------------------------------------- 速查数据 (摘自各域卡片)
CHEAT = {
    'A': {
        'formulas': [
            ('a_m = N \\cdot V_c \\cdot \\dot{\\lambda}', 'PN 基本律；TPN 用接近速度 V_c，PPN 用弹速 V_m', '卡01'),
            ('\\ddot{\\lambda} = (2-N)\\frac{V_c}{R}\\dot{\\lambda} + \\frac{a_t}{R}',
             'LOS 动力学；<b>N&gt;2 是稳定性红线</b>（N&gt;2 时系数为负→λ̇ 衰减）。'
             '原书记作 −(2−N)，与同段结论矛盾，已校勘', '卡01/02'),
            ('a_{cmd} = N\' V_c \\dot{\\lambda} + \\frac{N\'}{2} a_T',
             'APNG 增广项；N&apos;/2 由 LQR 导出，保证常值机动零脱靶（N&apos;=3 时补偿系数 1.5）', '卡05'),
            ('ZEM = \\frac{V_c \\cdot t_{go}^2}{2}\\dot{\\lambda}',
             '零控脱靶量；目标机动时会严重低估，是 APN 存在的理由', '卡04'),
            ('N_{opt} = 3', 'LQR 二次型指标下的最优导航比（无延迟、无噪声理想条件）', '卡02'),
        ],
        'tips': [
            ('N 调参口诀', '从 N=3 起试；过载大降 2.5、脱靶量大升 4。按舵机延迟修正：0ms→3.5~4.0，100ms→2.8~3.2，200ms→2.5~3.0。', '卡02'),
            ('延迟是头号杀手', '0→50ms，脱靶量 0.5m→12.3m（翻 24 倍）。N=5.5 配 80ms 延迟，脱靶量比 N=4 大 3 倍。', '卡02'),
            ('限幅要"双限"', '幅值限 + 速率限，留 20~30% 过载裕度；λ̇ 绝不能差分，必须滤波。', '卡06'),
            ('APN 的性能上限 = 目标加速度估计精度', '估计误差 &gt;50~60% 时反不如 PN，须加置信度门限与退化机制。', '卡05'),
            ('别只看均值', '某项目仿真均值 3.2m、实弹打 28m；95% 分位数才是命门。', '卡04'),
        ],
    },
    'B': {
        'formulas': [
            ('u^* = -\\frac{3q}{r}y - \\frac{3q}{r}t_{go} v_y',
             '双积分器 LQR 闭式解 ⇒ 退化为 PN，<b>N_opt = 3 由此而来</b>', '卡02'),
            ('-\\dot{P} = A^T P + P A + Q - P B R^{-1} B^T P',
             'Riccati 微分方程；制导是有限时域问题，<b>必须反向积分</b>且依赖 t_go', '卡02'),
            ('J = \\tfrac{1}{2}x^T(t_f)Sx(t_f) + \\tfrac{1}{2}\\int\\![u^T R u - \\gamma^2 v^T Q v]dt',
             'LQDG 微分对策指标；导弹 min、目标 max，γ 为对策参数', '卡03'),
            ('\\dot{x}_i = \\sum_{j\\in N_i} a_{ij}(x_j - x_i)',
             '一致性协议；离散步长需 ε &lt; 1/λ_max(L)，收敛速度由代数连通度决定', '卡06'),
            ('a = N V \\dot{\\lambda} + K_t (T_{go} - T_{go,des})',
             'ITCG 攻击时间约束；IACG 再加 N₂V(γ-γ_d)/t_go 落角项', '卡06'),
        ],
        'tips': [
            ('算力墙', '4 弹协同单步 20ms &gt; 控制周期 10ms，改"离线预计算 + 在线查表修正"；C++ Eigen 固定矩阵版单步仅 0.32μs。', '卡05'),
            ('通信墙', '时延 10→50ms 使一致性收敛从 2s 变 8s；丢包 &gt;40% 协议失效；<b>永远按标称带宽 50% 设计协议</b>。', '卡06'),
            ('数值坑', '伪谱法遇 bang-bang 控制，配点加到 200 仍振荡，换直接配点 50 节点即成；伴随方程禁用欧拉法、必须缩放。', '卡04'),
            ('估计坑', 'APN 估计误差 &gt;30% 反不如 PN，延迟 &gt;0.2s 补偿帮倒忙；<b>模型比滤波器重要（70% 精力建模型）</b>。', '卡03'),
            ('仿真可信度', '目标机动上限要提高 30% 建模（作者按 5G 仿真、实测 7G，三弹全脱靶）；鲁棒性看 σ/μ&lt;0.3 与 P95。', '卡05'),
        ],
    },
    'C': {
        'formulas': [
            ('\\mathbf{t}\\cdot\\mathbf{r} \\ge \\|\\mathbf{t}\\|\\cos\\theta_{FOV}',
             '导引头视场锥体约束；等价框架角判据 |λ-θ_m| &gt; θ_max 即丢失目标', '卡01'),
            ('\\dot{\\varepsilon} = \\frac{V_m\\sin\\varepsilon - V_t\\sin(\\theta_t-\\lambda)}{r} - \\frac{a_m}{V_m}',
             '视场角误差动力学；两项相消即"视场保持"的控制本质', '卡01'),
            ('a_{cmd} = N V \\dot{\\lambda} + K_{bias}(\\theta_{cur} - \\theta_{ref})',
             '偏置比例导引 BPN；用偏置项把 LOS 角拉回视场中心', '卡02'),
            ('a_m = N_1\\frac{y}{t_{go}^2} + N_2\\frac{\\dot y}{t_{go}} + N_3 V_m\\frac{\\theta_m - \\theta_d}{t_{go}}',
             '带落角/速度约束的最优制导律，N₁=N₂=N₃ 时退化为标准形式', '卡03/05'),
            ('s = c_1 e_\\theta + c_2 e_V + \\dot e_\\theta,\\; u = u_{eq} - k\\,\\mathrm{sign}(s)',
             '滑模速度矢量控制；边界层 Φ=0.01~0.05 抑制抖振', '卡05'),
        ],
        'tips': [
            ('视场必须当硬约束', '仿真时视场角缩小 10%~15% 使用，留 3°~5° 遮挡余量；7g 蛇形机动可使脱靶 0.5m→15m。', '卡01'),
            ('BPN 偏置增益有上界', 'K_bias ≤ N/3，且 t_go&lt;1s 时线性衰减到零；实测最大框架角 28.5°→19.2°、视场丢失率 72%→0%。', '卡02'),
            ('李雅普诺夫增益别贪大', 'k=0.5~2 合适，设成 10 会让导弹左右摆动；k₂ 取干扰上界 1.2~1.5 倍。', '卡02'),
            ('约束不是越紧越好', '攻角限在失速点 80%；落角 ±5° 满足率仅 60%，放宽到 ±8° 反升到 98%。', '卡03'),
            ('一体化优于分离设计', '脱靶 2.35→0.87m、舵偏 18.2°→12.5°；±20% 摄动下分离设计失稳、一体化仍稳定。', '卡06'),
        ],
    },
    'D': {
        'formulas': [
            ('G(s)=\\frac{M_\\delta s+Z_\\delta M_\\alpha-M_\\delta Z_\\alpha}{s^3-(Z_\\alpha+M_q)s^2+(Z_\\alpha M_q-M_\\alpha)s}',
             '舵偏→俯仰角三阶传递函数；零极点近似对消后降二阶，ω_n、ζ 即短周期特性', '卡01'),
            ('G_m = -20\\lg|GH|,\\; P_m = 180^\\circ + \\angle GH',
             '稳定裕度；工程区间 <b>G_m=6~12dB、P_m=45°~60°</b>', '卡02'),
            ('u_{comp} = u_{cmd} + (u_{cmd} - u_{prev})\\frac{T_{delay}}{dt}',
             '延迟一阶外推补偿；高动态项目等效延迟 2.3ms→0.4ms', '卡03'),
            ('w_{opt} = \\frac{R^{-1}a}{a^H R^{-1} a}',
             'MVDR 自适应波束形成；对角加载 R+λI 提升稳健性', '卡06'),
            ('\\mathrm{CEP} \\approx 1.1774\\sigma,\\; \\mathrm{RMSE}=\\sqrt{\\mathrm{mean}((y_{act}-y_{ideal})^2)}',
             '精度统计口径；蒙特卡洛 N≥1000 且看 P95 而非均值', '卡05'),
        ],
        'tips': [
            ('带宽分层法则', '内环（自动驾驶仪）带宽须为外环（制导）的 5~10 倍；阻尼回路取弹体带宽 3~5 倍，贪高会灌入高频噪声、舵机抖振。', '卡03'),
            ('积分步长与算法', '步长 &lt; 最小时间常数 1/10，必用 RK4（同弹道同步长 0.01s，欧拉法 10 秒差几百米）。', '卡03'),
            ('干扰影响是非线性的', '干扰功率 +3dB，脱靶量可能 ×10；只盯导引头会翻车（舵机饱和照样打不准）。', '卡06'),
            ('仿真结果要打系数', '仿真结果乘 1.3 才是真实需求；气动偏差可达 20%，过载至少留 30% 裕度。', '卡03/05'),
            ('HIL 时序指标', '总延迟 &lt;1ms、抖动 &lt;50μs；GPS/PTP/硬件触发三级同步，10m 同轴电缆约 50ns。', '卡04'),
            ('外场试验"三三制"', '3 干扰样式 × 3 功率等级 × 3 场景 ≥ 27 次，否则结论不可信。', '卡06'),
        ],
    },
    'E': {
        'formulas': [
            ('J=\\int\\!\\left(\\frac{d^4p}{dt^4}\\right)^2 dt = c^T Q c,\\; c_f = -(P^TQP)^{-1}P^TQc_0',
             'Minimum Snap 闭式解；边界条件线性映射后一步求解', '卡01'),
            ('b(s)=v^2(s),\\; T=\\int\\frac{ds}{\\sqrt{b(s)}}',
             '时间最优问题的弧长换元；缩放律 v→v/α、a→a/α²、j→j/α³', '卡01'),
            ('\\beta_1=3\\omega_o,\\; \\beta_2=3\\omega_o^2,\\; \\beta_3=\\omega_o^3',
             'ADRC/ESO 带宽整定法；u = u₀ - z₃/b₀ 完成扰动补偿', '卡04'),
            ('z = H\\theta + v,\\; \\mathrm{Var}(\\hat\\theta) \\ge \\mathcal{I}^{-1}',
             '气动参数观测方程与 CRLB 下界（信息矩阵 I = 1/R·Σ(∂h/∂θ)²）', '卡05'),
            ('C_{L,fused} = w\\,C_{L,flight} + (1-w)C_{L,WT}',
             '风洞-试飞数据融合；覆盖区内 w=0.7~0.8', '卡06'),
        ],
        'tips': [
            ('采样 vs 最优的取舍', 'RRT 5000 次采样得 32m 路径，RRT* 8000 次优化到 24m（时间 +40%）；A* vs Dijkstra 在 100×100 栅格是 2000 vs 7000 节点。', '卡01'),
            ('ADRC 的 b₀ 必须校准', 'b₀ 差 20%，控制效果天差地别，须用阶跃辨识反推；δ 取 0.01~0.1。', '卡04'),
            ('增益调度的适用边界', '气动摄动 &gt;30% 时线性插值增益调度失效；改用 LPV 后 ±20% 摄动下俯仰误差 &lt;1.5°（传统 &gt;5°）。', '卡04'),
            ('辨识对测量误差极敏感', '侧滑角差 1° → C_nβ/C_nδr 差 20%；换装五孔探针后 C_nδr 误差 6.7%→3%。', '卡05'),
            ('蒙特卡洛样本量', 'N≥5000；15 个样本若用正态假设置信区间会窄一倍，须改用 Bootstrap。', '卡06'),
        ],
    },
}


# ------------------------------------------------- 轻量 Markdown → HTML
def md_to_html(md: str) -> str:
    """把卡片 markdown 正文渲染为 HTML。公式用 LaTeX 原文保留，交给 MathJax。"""
    lines = md.split('\n')
    out = []
    in_code = False
    code_buf = []
    table_buf = []

    def flush_table():
        if not table_buf:
            return
        rows = []
        for i, r in enumerate(table_buf):
            cells = [c.strip() for c in r.strip().strip('|').split('|')]
            if i == 1 and all(set(c) <= set('-: ') and c for c in cells):
                continue  # 分隔行
            tag = 'th' if i == 0 else 'td'
            rows.append('<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>')
        out.append('<table class="ct">' + ''.join(rows) + '</table>')
        table_buf.clear()

    for ln in lines:
        raw = ln.rstrip()
        s = raw.strip()

        # 代码块
        if s.startswith('```'):
            if in_code:
                out.append('<pre><code>' + html.escape('\n'.join(code_buf)) + '</code></pre>')
                code_buf = []
                in_code = False
            else:
                flush_table()
                in_code = True
            continue
        if in_code:
            code_buf.append(raw)
            continue

        # 表格
        if s.startswith('|') and s.endswith('|') and s.count('|') >= 2:
            table_buf.append(s)
            continue
        flush_table()

        if not s:
            continue

        # 块级公式
        if s.startswith('$$'):
            body = s.strip('$').strip()
            out.append('<div class="eq">\\[' + body + '\\]</div>')
            continue

        # 标题
        m = re.match(r'^(#{1,6})\s+(.*)$', s)
        if m:
            lv = len(m.group(1))
            out.append(f'<h{4 + min(lv, 2)}>{inline(m.group(2))}</h{4 + min(lv, 2)}>')
            continue

        # 分隔线
        if re.fullmatch(r'[-*_]{3,}', s):
            out.append('<hr>')
            continue

        # 列表
        m = re.match(r'^([-*+])\s+(.*)$', s)
        if m:
            out.append('<li>' + inline(m.group(2)) + '</li>')
            continue
        m = re.match(r'^(\d+)[.)]\s+(.*)$', s)
        if m:
            out.append('<li class="num">' + inline(m.group(2)) + '</li>')
            continue

        # 引用
        if s.startswith('>'):
            out.append('<blockquote>' + inline(s.lstrip('> ').strip()) + '</blockquote>')
            continue

        out.append('<p>' + inline(s) + '</p>')

    flush_table()
    if in_code and code_buf:
        out.append('<pre><code>' + html.escape('\n'.join(code_buf)) + '</code></pre>')
    return '\n'.join(out)


def inline(t: str) -> str:
    """行内渲染。先 escape，再还原粗体/行内代码/行内公式。"""
    t = html.escape(t, quote=False)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)                  # 行内代码
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)        # 粗体
    t = re.sub(r'(?<!\$)\$([^$]+)\$(?!\$)', r'<span class="il">\(\1\)</span>', t)  # 行内公式
    return t


# ------------------------------------------------- 解析卡片
SEC_PAT = re.compile(r'\*\*【(R - 理论|I - 公式|A\+\+ - 经验|关联)】\*\*')


def parse_cards(path: str):
    txt = open(path, encoding='utf-8').read()
    # 元信息块（第一个 ### 卡 之前的内容）
    head = txt.split('### 卡')[0]
    parts = re.split(r'^### ', txt, flags=re.M)[1:]
    cards = []
    for p in parts:
        title = p.split('\n')[0].strip()
        if not title.startswith('卡'):
            continue
        # 来源行
        src = ''
        m = re.search(r'\*\*来源\*\*[：:]\s*(.*)', p)
        if m:
            src = m.group(1).strip()
        # 分段
        segs = {}
        marks = [(m.start(), m.end(), m.group(1)) for m in SEC_PAT.finditer(p)]
        for i, (st, en, name) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(p)
            segs[name] = p[en:end].strip()
        body_html = md_to_html(p)
        cards.append({
            'title': title,
            'src': src,
            'r': segs.get('R - 理论', ''),
            'i': segs.get('I - 公式', ''),
            'a': segs.get('A++ - 经验', ''),
            'link': segs.get('关联', ''),
            'chars': len(p),
            'html': body_html,
        })
    return head.strip(), cards


# ------------------------------------------------- 主流程
def main():
    # 1. 卡片
    domains = {}
    total_cards = 0
    for k in ['A', 'B', 'C', 'D', 'E']:
        cfg = DOMAINS[k]
        fp = os.path.join(BASE, cfg['file'])
        if not os.path.exists(fp):
            print(f'[skip] 缺少 {cfg["file"]}')
            continue
        head, cards = parse_cards(fp)
        total_cards += len(cards)
        domains[k] = {
            'name': cfg['name'],
            'color': cfg['color'],
            'file': cfg['file'],
            'courses': cfg['courses'],
            'head': head,
            'cards': cards,
            'cheat': CHEAT.get(k, {'formulas': [], 'tips': []}),
        }
        print(f'  域{k} {cfg["name"]}: {len(cards)} 卡')

    # 2. 章节目录
    toc_path = os.path.join(ROOT, 'book010_toc.json')
    chapters = []
    if os.path.exists(toc_path):
        toc = json.load(open(toc_path, encoding='utf-8'))
        c2d = {}
        for k, cfg in DOMAINS.items():
            for c in cfg['courses']:
                c2d[c] = k
        for c, chs in toc.items():
            dom = c2d.get(c, '?')
            for ch in chs:
                chapters.append({
                    'dom': dom,
                    'course': c.strip('《》'),
                    'no': ch['no'],
                    'title': ch['title'],
                    'desc': ch['desc'],
                    'tags': ch['tags'],
                })
        chapters.sort(key=lambda x: (x['dom'], x['course'],
                                     int(x['no']) if x['no'].isdigit() else 0))
    print(f'  章节: {len(chapters)}')

    data = {'domains': domains, 'chapters': chapters, 'totalCards': total_cards}
    json.dump(data, open(os.path.join(BASE, '_index_data.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)

    # 3. HTML
    tpl = open(os.path.join(BASE, '_template.html'), encoding='utf-8').read()
    out = tpl.replace('/*__DATA__*/null', json.dumps(data, ensure_ascii=False))
    op = os.path.join(BASE, 'index.html')
    open(op, 'w', encoding='utf-8').write(out)
    print(f'\n✅ 已生成 {op}')
    print(f'   卡片 {total_cards} 张 / 章节 {len(chapters)} 章 / {len(out)} 字符')


if __name__ == '__main__':
    main()
