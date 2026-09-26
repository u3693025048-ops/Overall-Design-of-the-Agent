# GSMultiAgent 实验设计与复现步骤

本文档用于复现 GSMultiAgent 的外部 Monte Carlo 实验，包括实验设计、数据生成、主实验、消融实验、统计分析、可视化和补充对照。所有命令默认在**仓库根目录**执行。

相关文档：

| 文档 | 内容 |
|------|------|
| 本文 | 实验协议、矩阵、数据生成、第 4.2/4.3 节复现步骤 |
| [single_agent_hermes/README.md](single_agent_hermes/README.md) | 单智能体 Hermes 基线 |
| [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md) | 模型版本、日期、等预算公式 |
| 结果目录 | `results/experiment_plan/`（等预算：`optimizer_equal_mc_75/` 与 `optimizer_equal_mc_75_sah_n10/`） |

旧不等预算基线目录、冒烟结果、重复说明已归档到 `outOfDate/`。

## 1. 实验目标

本实验回答三个问题：

1. GSMultiAgent 在固定 Monte Carlo 随机样本下，是否比传统黑盒搜索和去编排 PPO 更快、更稳定地得到可行制导/驾驶仪参数。
2. 多智能体编排、参数经验库、持久记忆、制导律适配验证、MATLAB 语法检查和 Layer3 优化分别贡献了什么。
3. 最终设计在独立 test Monte Carlo 样本上是否仍满足约束，而不是只在调参用 dev 样本上表现良好。

实验采用 dev/test 分离。每个 master seed 对应两份 `.mat` 文件：dev 数据用于参数搜索、PPO、LLM 反思和经验复用；test 数据只用于最终参数的一次独立评估。

## 2. 实验对象

### 2.1 工况

主实验使用 4 个目标机动场景：

| 场景 | 含义 | 主要变化 |
|------|------|---------|
| `T1` | 匀速/弱机动目标 | 目标纵向速度随机扰动 |
| `T2` | 低频小幅机动 | `omega_z` 小幅随机扰动 |
| `T3` | 中频中等机动 | `Az` 中等幅度随机扰动 |
| `T4` | 高频大幅机动 | `Az` 高强度机动随机扰动 |

补充实验还会生成 `T1_broadside`，用于侧向交会几何鲁棒性检查；该场景不进入默认主实验矩阵，需要按第 14 节单独运行。

### 2.2 可调参数

实验参数空间来自 `multi_agent/rl/matlab_rl_optimizer.py` 的 `ALL_TUNABLE_PARAM_SPECS`，当前为 10 维：

| 参数 | 含义 | 默认值 | 范围 |
|------|------|-------:|------|
| `w1` | 俯仰通道带宽 | 40.0 | [20.0, 60.0] |
| `zeta1` | 俯仰通道阻尼 | 0.75 | [0.35, 1.20] |
| `tao1` | 俯仰通道时间常数 | 0.15 | [0.05, 0.35] |
| `w2` | 偏航通道带宽 | 35.0 | [20.0, 60.0] |
| `zeta2` | 偏航通道阻尼 | 0.75 | [0.35, 1.20] |
| `tao2` | 偏航通道时间常数 | 0.15 | [0.05, 0.35] |
| `w3` | 滚转通道带宽 | 40.0 | [20.0, 60.0] |
| `zeta3` | 滚转通道阻尼 | 0.70 | [0.35, 1.20] |
| `N_pn` | 比例导引系数 | 4.0 | [3.0, 6.0] |
| `ny_lim` | 制导过载限幅，单位 g | 20.0 | [16.0, 20.0] |

所有基线方法和 GSMultiAgent 都使用同一参数边界。候选参数在仿真前会被裁剪到合法范围内。

### 2.3 约束与指标

可行性判断由 `multi_agent.tools.simulation_tool.check_constraints` 统一完成：

| 指标 | 约束 | 说明 |
|------|------|------|
| `hit_rate` | >= 92% | 命中率，命中半径默认 10 m |
| `SEP` | <= 7 m | Spherical Error Probable |
| `peak_nz` | <= 20 g | 法向过载峰值代理指标 |
| `PM` | 45 到 70 deg | 俯仰相位裕度 |
| `BW` | 20 到 85 rad/s | 俯仰带宽 |

CSV 中同时记录 dev 和 test 指标。论文/报告中应优先使用 test 指标，因为 test 数据没有参与搜索和优化。

## 3. 实验矩阵

### 3.1 主实验

默认主实验矩阵为：

```text
4 methods × 4 scenarios × 3 seeds = 48 rows
```

| 维度 | 设置 |
|------|------|
| 方法 | `GSMultiAgent`, `random_search`, `cma_es`, `direct_ppo`, `single_agent_hermes` |
| 场景 | `T1`, `T2`, `T3`, `T4` |
| master seed | `42` 到 `44`（每个 seed 对应一个 dev .mat + 一个 test .mat 文件） |
| 每个 `.mat` 文件 MC 样本数 | 100 |
| 输出 | `results/main_results.csv` |

方法说明：

| 方法 | 设计含义 | 预算/迭代 |
|------|---------|----------|
| `GSMultiAgent` | 完整系统，包含 Layer1 任务解析、Layer2 Hermes 编排、工具调用、经验/记忆复用、Layer3 优化和反思 | 外层上限 `workflow.max_iterations=5`，每层 `max_episodes=15`。**上限** 5×15×3=**225**；表上 1/7/22 是外层反推，常 1 真 seed + bootstrap，与基线 75×3 不可对读 |
| `random_search` | 10 维参数均匀随机搜索 | 与 GSMultiAgent **相同的最大 MC 调用次数**：75 个候选 × 3 个 dev seed = **225** MC calls，然后 test 一次 |
| `cma_es` | 连续黑盒优化基线 | **同一 225 MC 上限**。λ = 10，按候选评估计数而非代数计数；最后一代不足 λ 时只评估剩余名额、不更新协方差 |
| `direct_ppo` | 只使用 PPO 优化器，去掉多智能体编排和记忆复用 | **同一 225 MC 上限**（75 个候选评估） |
| `single_agent_hermes` | 同工具套件（PE 热启动 + PPO + 修正管线），去掉 LLM 级联、反思诊断和 Hermes 记忆注入 | `outer_iterations=5` × `episodes_per_outer=15` = 75 候选，**同一 225 MC 上限** |

所有方法的开发集蒙特卡洛调用**上限**由 `experiments/equal_mc_budget.py` 统一给出（75 候选 × 3 seed = 225），禁止把 CMA-ES 的“一代种群”当成一次迭代。这不等于表上的 MC 整数是同一套计数：

- **搜索弹道**：分层评测时每次 MATLAB **10 条**（`mc_search_n_runs`）；**测试 / 终检 100 条**（`mc_final_n_runs`）。不要把 §4.1 的「100 条」写成搜索宽度。
- **Random / CMA-ES / Direct PPO**：每个候选 3 次真 MATLAB，满额 225。CSV `first_success` = 预算内是否出现可行候选（与最终 `dev_all_pass` 一致）。第一次可行的 **MC 次数**见 `results/experiment_plan/optimizer_equal_mc_75/first_success_mc_calls.csv`（均值 Random 73.9 / CMA 79.2 / PPO 54.0），不要读 `first_success_iter`（CMA 那列是代数）。
- **GSMultiAgent**：表上 1/7/22 是**外层反推**，且常用 `mc_and_bootstrap_fill`（1 真 seed + 2 填充）。`first_success` = **最终开发集可行**（9/12），不是「第一轮外层就成功」。
- **Single-Agent Hermes**：对齐后目录 `results/experiment_plan/optimizer_equal_mc_75_sah_n10/`（搜索 10、test 100、`first_success=final_dev_feasible`）。12 格 **first_success / dev / test 均为 12/12**，开发集 MC **135–225**，Time/call 约 **0.06 min**。旧 100 条目录 `optimizer_equal_mc_75_sah/` 保留备查，**不打进新包、不要覆盖**。Table 6：`results/experiment_plan/table6_equal_mc.csv`。

### 3.2 记忆消融

记忆消融用于验证跨任务经验复用的价值。同一个 seed 内按 `T1 -> T2 -> T3 -> T4` 顺序运行，使系统有机会从前序任务积累经验。

| 组别 | 参数经验库 | 持久记忆/检索 | 验证目标 |
|------|-----------|--------------|---------|
| `E0_full_memory` | 开 | 开 | 完整记忆机制 |
| `E1_no_parameter_experience` | 关 | 开 | 单独移除参数经验库 |
| `E2_no_memory_search` | 开 | 关 | 单独移除持久记忆检索 |
| `E3_no_memory` | 关 | 关 | 完全无跨任务记忆 |

输出文件：`results/ablation_memory.csv`。

### 3.3 优化管线消融

优化管线消融用于拆解数学适配验证、语法检查、RL 优化和多轮迭代的贡献。

| 组别 | 适配验证 | 语法检查 | RL 优化 | 外层轮次 | 验证目标 |
|------|---------|---------|--------|---------|---------|
| `O0_full` | 开 | 开 | 开 | 5 | 完整优化管线 |
| `O1_no_compat` | 关 | 开 | 开 | 5 | 制导律适配验证的贡献 |
| `O2_no_syntax` | 开 | 关 | 开 | 5 | MATLAB 语法检查/修复的贡献 |
| `O3_no_rl` | 开 | 开 | 关 | 5 | PPO 参数优化的贡献 |
| `O4_single_iteration` | 开 | 开 | 开 | 1 | 多轮生成、仿真、反思循环的贡献 |

输出文件：`results/ablation_optimization.csv`。

### 3.4 修正管线消融

修正管线消融聚焦最困难的 `T4 MODIFY_LAW` 场景，检查制导律修改后是否依赖适配验证和语法检查。

| 组别 | 场景 | 适配验证 | 语法检查 | 验证目标 |
|------|------|---------|---------|---------|
| `C0_full_correction` | `T4` | 开 | 开 | 完整修正管线 |
| `C1_no_compat` | `T4` | 关 | 开 | 去掉数学适配验证 |
| `C2_no_syntax` | `T4` | 开 | 关 | 去掉 MATLAB 语法检查 |

输出文件：`results/ablation_correction.csv`。

### 3.5 补充对照

| 方法 | 文件 | 作用 |
|------|------|------|
| DeepSeek-V4 auto feedback | `experiments/fairness_ablation.py` | 模拟自动 LLM 反馈闭环，减少人工延迟差异 |
| Greedy feasibility-first | `experiments/greedy_baseline.py` | 固定约束优先级的贪心可行性修正基线 |

## 4. 输出文件

| 路径 | 内容 |
|------|------|
| `data/mc/` | 外部 Monte Carlo dev/test `.mat` 数据 |
| `data/mc/seed_map.json` | master seed 到 dev/test seed 的映射 |
| `results/main_results.csv` | 主实验结果 |
| `results/ablation_memory.csv` | 记忆消融结果 |
| `results/ablation_optimization.csv` | 优化管线消融结果 |
| `results/ablation_correction.csv` | 修正管线消融结果 |
| `results/tables/` | LaTeX 表格 |
| `results/figures/` | PNG/PDF 图 |
| `guidance_output/hermes_rl_best.json` | GSMultiAgent 最近一次 Layer3 最优快照 |
| `logs/` | CLI、RL episode 和仿真日志 |

批量 runner 采用追加式 CSV 写入。已经成功完成的 `(method, scenario, seed)` 会在续跑时自动跳过；失败行会保留 `error=True` 和 `error_msg`，下次重跑不会被当作已完成。

## 5. 环境准备

推荐先安装项目依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install scipy tqdm numpy pandas matplotlib seaborn cma
```

真实数值实验需要 MATLAB 或 Octave 能从命令行启动。请检查 `config.yaml` 中：

```yaml
simulation:
  engine: matlab
  matlab_path: matlab          # 或本机 MATLAB 可执行文件的绝对路径
  octave_path: octave
```

如果只验证 Python 调度和文件接口，不希望调用 LLM，可设置：

```powershell
$env:GSM_BASELINE_SKIP_LLM='1'
```

## 6. 生成外部 MC 数据

执行默认数据生成：

```powershell
python experiments\mc_data_generator.py --n-runs 100
```

默认输出：

```text
data/mc/
  T1/
    seed_4201_dev.mat
    seed_4202_dev.mat
    seed_4203_dev.mat
    seed_4204_test.mat
    seed_4205_test.mat
    seed_4206_test.mat
    seed_42_dev.mat
    seed_42_test.mat
    ...
  T2/
  T3/
  T4/
  T1_broadside/
  seed_map.json
```

seed 规则：

```text
dev_seeds  = master_seed * 100 + {1, 2, 3}
test_seeds = master_seed * 100 + {4, 5, 6}
```

例如 master seed `42` 会生成 `4201,4202,4203` 三个 dev 文件，以及 `4204,4205,4206` 三个 test 文件。`seed_42_dev.mat` 和 `seed_42_test.mat` 是兼容旧脚本的别名，分别复用第一个 dev/test split。

快速验证：

```powershell
python -c "import scipy.io, os; p='data/mc/T1/seed_42_dev.mat'; assert os.path.exists(p); d=scipy.io.loadmat(p); assert d['missile_init'].shape==(1,100); assert d['target_init'].shape==(1,100); print('MC data OK')"
```

如需快速小样本调试：

```powershell
python experiments\mc_data_generator.py --output-root data/mc_smoke --n-runs 10 --no-broadside
```

## 7. 验证外部 MC 仿真接口

先用 `run_mc_with_external_data` 验证 MATLAB/Octave 能正确读取外部 `.mat` 文件：

```powershell
python -c "from multi_agent.tools.simulation_tool import run_mc_with_external_data; r=run_mc_with_external_data('knowledge_base/matlab/guidance/monte_carlo_single.m','data/mc/T1/seed_42_dev.mat',n_runs=1); print(r)"
```

返回字段应包含：

```text
hit_rate, SEP, peak_nz, PM, BW,
miss_distances, nz_peaks,
feasible, constraint_violations,
sim_time_sec, success, error
```

如果 MATLAB 超时、路径错误或脚本失败，函数会返回 `success=False` 和错误信息，不会让 Python 进程崩溃。

## 8. 基线方法 smoke test

先用 `T1 x seed 42` 和极小预算确认三个基线能跑通。

Random Search：

```powershell
python -c "from experiments.baselines import random_search; r=random_search('T1','data/mc/T1/seed_42_dev.mat','data/mc/T1/seed_42_test.mat',budget=1,seed=42); print(r.keys())"
```

CMA-ES：

```powershell
python -c "from experiments.baselines import cma_es_optimize, DEFAULT_PARAMS; r=cma_es_optimize('T1','data/mc/T1/seed_42_dev.mat','data/mc/T1/seed_42_test.mat',DEFAULT_PARAMS,budget=1,seed=42); print(r.keys())"
```

Direct PPO 默认会尝试调用 LLM 获取初始参数。只做接口 smoke test 时建议跳过：

```powershell
$env:GSM_BASELINE_SKIP_LLM='1'
python -c "from experiments.baselines import direct_ppo_optimize; r=direct_ppo_optimize('T1','data/mc/T1/seed_42_dev.mat','data/mc/T1/seed_42_test.mat',max_iterations=1,seed=42); print(r.keys())"
Remove-Item Env:\GSM_BASELINE_SKIP_LLM
```

正式基线预算（全部方法相同的最大开发集 MC 调用次数）：

```text
max_candidates = 5 × 15 = 75
max_mc_calls_dev = 75 × 3 seeds = 225
Random Search / CMA-ES / Direct PPO / Single-Agent / GSMultiAgent: 同一 *上限*
计数单位不同：基线 = 75×3 真 MATLAB；GSM 表上 1–22 = 外层反推 + 常 bootstrap
搜索 10 条弹道 / 测试 100 条；不要把 CMA 代数当成候选次数
Single-Agent 对齐口径请写入 optimizer_equal_mc_75_sah_n10，勿覆盖旧 100 条目录
```

第 4.3 节正式入口是 `experiments/optimizer_baselines_43.py`（`--iterations` 与 `--max-candidates` 同义，默认 75；`n_runs` 默认 100）：

```powershell
# smoke：单方法、单场景、1 候选
python experiments\optimizer_baselines_43.py --methods random_search --scenarios T1 --master-seeds 42 --max-candidates 1 --n-runs 10 --output-dir results\experiment_plan\optimizer_equal_mc_75_smoke

# 正式等预算（Random / CMA-ES / Direct PPO）
python experiments\optimizer_baselines_43.py `
  --methods random_search cma_es direct_ppo `
  --scenarios T1 T2 T3 T4 `
  --master-seeds 42 43 44 `
  --max-candidates 75 `
  --n-runs 100 `
  --output-dir results\experiment_plan\optimizer_equal_mc_75

# 单智能体对齐口径（搜索 10 条、test 100；新目录，勿覆盖旧 100 条结果）
python experiments\optimizer_baselines_43.py `
  --methods single_agent_hermes `
  --scenarios T1 T2 T3 T4 `
  --master-seeds 42 43 44 `
  --max-candidates 75 `
  --n-runs 100 `
  --output-dir results\experiment_plan\optimizer_equal_mc_75_sah_n10
# 旧 100 条搜索结果保留在 optimizer_equal_mc_75_sah/，不覆盖；表数字用 n10
```

## 9. GSMultiAgent smoke test

单任务 CLI 调用使用 prompt 文件作为实验任务输入：

```powershell
python cli_agent.py --file prompt.txt
```

如果需要切换实验任务，先编辑或替换 `prompt.txt`，再用同一命令运行。外部 MC 批量复现实验仍建议使用下方 `experiments.runner` 入口统一传入 dev/test 数据路径。

批量入口在 `experiments.runner`。先用 dry run 验证 runner、CSV 追加和外部 MC 路径：

```powershell
python -c "from experiments.runner import run_all_experiments; run_all_experiments(['GSMultiAgent'], ['T1'], [42,43], {'dry_run': True, 'n_runs': 1}, output_csv='results/smoke_runner.csv', max_workers=1)"
```

说明：`dry_run=True` 不运行完整多智能体流程，只用默认参数分别评估 dev/test，用于快速检查文件路径、MATLAB 调用和 CSV 写入。

## 10. 运行主实验

确认 smoke test 通过后运行完整主实验：

```powershell
python -c "from experiments.runner import run_all_experiments, MASTER_SEEDS, SCENARIOS; run_all_experiments(['GSMultiAgent','random_search','cma_es','direct_ppo'], SCENARIOS, MASTER_SEEDS, output_csv='results/main_results.csv', max_workers=4)"
```

如果机器内存或 MATLAB license 不适合并行，把 `max_workers` 调小：

```powershell
python -c "from experiments.runner import run_all_experiments, MASTER_SEEDS, SCENARIOS; run_all_experiments(['GSMultiAgent','random_search','cma_es','direct_ppo'], SCENARIOS, MASTER_SEEDS, output_csv='results/main_results.csv', max_workers=1)"
```

只跑子集调试：

```powershell
python -c "from experiments.runner import run_all_experiments; run_all_experiments(['random_search','cma_es'], ['T1'], [42], {'budget': 5}, output_csv='results/debug_baselines.csv', max_workers=1)"
```

论文表 5–7 的等预算对照以 `results/experiment_plan/optimizer_equal_mc_75/`（及 `_sah/`）和 `results/experiment_plan/table5_equal_mc.csv` 为准：Random **8/12**、CMA-ES **12/12**、Direct PPO **11/12**。不要使用已归档的不等预算 `optimizer_baselines_*` 目录。

## 11. 运行消融实验

消融实验通过修改 `config.yaml` 中的 `ablation` 开关来插拔模块，**统一使用以下命令运行**：

```powershell
python cli_agent.py --file prompt.txt
```

每次实验前：按下方对应组的说明修改 `config.yaml` → 保存 → 执行命令。
不同场景对应不同 prompt 文件（`prompt_T1.txt` / `prompt_T2.txt` / `prompt_T3.txt` / `prompt_T4.txt`）；结果保存在 `guidance_output/` 下，批量汇总可另行写入 `results/`。

### 11.0 消融开关速查

`config.yaml` 中 `ablation` 块各字段含义：

| 字段 | 控制模块 | `true` | `false` |
|------|---------|--------|---------|
| `ablation.guidance_compat_verification.enabled` | 制导律数学适配验证 | 验证带宽分离 / 稳定裕度 | 跳过适配验证 |
| `ablation.syntax_check.enabled` | MATLAB 语法检查/修复 | LLM 迭代修正脚本 | 跳过语法检查 |
| `ablation.rl_optimization.enabled` | Layer3 PPO 参数优化 | 开启 RL 优化 | 仅用初始参数单次仿真 |
| `ablation.parameter_experience_reuse.enabled` | 参数经验库 | 复用历史最优参数 | 从零推导参数 |
| `ablation.reflection_agent.enabled` | 反思智能体 | LLM 深度分析 + 早停 | 跳过反思 |
| `ablation.memory_search.enabled` | 持久记忆 + session_search | 开启跨会话记忆注入 | 禁用持久记忆 |
| `workflow.max_iterations` | 外层迭代轮次 | — | — |

### 11.1 记忆消融（Memory Ablation）

验证跨任务经验复用的价值。建议同一 seed 内按 **T1 → T2 → T3 → T4** 顺序运行，使系统有机会从前序任务积累经验。

**E0 — 完整记忆（基准组）**

```yaml
# config.yaml
ablation:
  parameter_experience_reuse: { enabled: true }
  memory_search:               { enabled: true }
workflow:
  max_iterations: 5
```

**E1 — 去参数经验库**

```yaml
ablation:
  parameter_experience_reuse: { enabled: false }
  memory_search:               { enabled: true }
workflow:
  max_iterations: 5
```

**E2 — 去持久记忆检索**

```yaml
ablation:
  parameter_experience_reuse: { enabled: true }
  memory_search:               { enabled: false }
workflow:
  max_iterations: 5
```

**E3 — 完全无跨任务记忆**

```yaml
ablation:
  parameter_experience_reuse: { enabled: false }
  memory_search:               { enabled: false }
workflow:
  max_iterations: 5
```

每组修改 `config.yaml` 后依次运行：

```powershell
python cli_agent.py --file prompt_T1.txt
python cli_agent.py --file prompt_T2.txt
python cli_agent.py --file prompt_T3.txt
python cli_agent.py --file prompt_T4.txt
```

### 11.2 优化管线消融（Optimization Pipeline Ablation）

拆解适配验证、语法检查、RL 优化和多轮迭代各自的贡献。

**O0 — 完整管线（基准组）**

```yaml
ablation:
  guidance_compat_verification: { enabled: true }
  syntax_check:                 { enabled: true }
  rl_optimization:              { enabled: true }
workflow:
  max_iterations: 5
```

**O1 — 去适配验证**

```yaml
ablation:
  guidance_compat_verification: { enabled: false }
  syntax_check:                 { enabled: true }
  rl_optimization:              { enabled: true }
workflow:
  max_iterations: 5
```

**O2 — 去语法检查**

```yaml
ablation:
  guidance_compat_verification: { enabled: true }
  syntax_check:                 { enabled: false }
  rl_optimization:              { enabled: true }
workflow:
  max_iterations: 5
```

**O3 — 去 RL 优化**

```yaml
ablation:
  guidance_compat_verification: { enabled: true }
  syntax_check:                 { enabled: true }
  rl_optimization:              { enabled: false }
workflow:
  max_iterations: 5
```

**O4 — 单轮迭代**

```yaml
ablation:
  guidance_compat_verification: { enabled: true }
  syntax_check:                 { enabled: true }
  rl_optimization:              { enabled: true }
workflow:
  max_iterations: 1
```

每组修改 `config.yaml` 后运行（按需替换场景 prompt）：

```powershell
python cli_agent.py --file prompt_T1.txt
python cli_agent.py --file prompt_T2.txt
python cli_agent.py --file prompt_T3.txt
python cli_agent.py --file prompt_T4.txt
```

### 11.3 修正管线消融（Correction Pipeline Ablation）

仅在 T4 场景（MODIFY_LAW 高难度）下运行。

**C0 — 完整修正管线（基准组）**

```yaml
ablation:
  guidance_compat_verification: { enabled: true }
  syntax_check:                 { enabled: true }
workflow:
  max_iterations: 5
```

**C1 — 去适配验证**

```yaml
ablation:
  guidance_compat_verification: { enabled: false }
  syntax_check:                 { enabled: true }
workflow:
  max_iterations: 5
```

**C2 — 去语法检查**

```yaml
ablation:
  guidance_compat_verification: { enabled: true }
  syntax_check:                 { enabled: false }
workflow:
  max_iterations: 5
```

每组修改 `config.yaml` 后运行：

```powershell
python cli_agent.py --file prompt_T4.txt
```

---

> **运行前检查**：确认 `config.yaml` 中 `llm.api_key` 已填写，`simulation.engine` 和引擎路径正确。如需仅测试 Python 管线（不调 LLM），在运行前设置 `$env:GSM_BASELINE_SKIP_LLM='1'`。

## 12. 统计分析

主实验汇总和 Table 5：

```powershell
python -c "from experiments.analysis import load_results, compute_summary, generate_latex_table_5; df=load_results('results/main_results.csv'); s=compute_summary(df); generate_latex_table_5(s, 'results/tables/table5.tex'); print(s)"
```

记忆消融 Table 6：

```powershell
python -c "from experiments.analysis import load_results, compute_summary, generate_latex_table_6; df=load_results('results/ablation_memory.csv'); s=compute_summary(df); generate_latex_table_6(s, 'results/tables/table6.tex'); print(s)"
```

优化管线消融 Table 7：

```powershell
python -c "from experiments.analysis import load_results, compute_summary, generate_latex_table_7; df=load_results('results/ablation_optimization.csv'); s=compute_summary(df); generate_latex_table_7(s, 'results/tables/table7.tex'); print(s)"
```

配对显著性检验示例：

```powershell
python -c "from experiments.analysis import load_results, pairwise_test; df=load_results('results/main_results.csv'); print(pairwise_test(df, 'GSMultiAgent', 'direct_ppo', 'wall_clock_sec'))"
```

建议至少报告：

| 统计项 | 推荐用途 |
|--------|---------|
| `first_success_rate` + 95% binomial CI | 首次可行：**最终开发集可行**（GSM 与三套优化器）；不要把 9/12 读成「第一轮就成功」 |

基线（Random / CMA-ES / Direct PPO）「第一次可行解的 **MC 调用次数**」**不要**读 CSV 的 `first_success_iter`（CMA 那一列是代数）。算法、手算例子和一键脚本见 `experiments/如何计算基线第一次可行解的MC调用次数.md`。
| test 指标均值和标准差 | 独立泛化性能 |
| `mc_calls` 均值和标准差 | 仿真预算成本 |
| `wall_clock_sec` 均值和标准差 | 端到端时间成本 |
| paired Wilcoxon + bootstrap CI | 同 seed 配对比较 |

## 13. 可视化

生成 Figure 11：

```powershell
python -c "from experiments.analysis import load_results, compute_summary; from experiments.visualization import plot_figure_11; df=load_results('results/main_results.csv'); s=compute_summary(df); plot_figure_11(s, 'results/figures')"
```

生成约束雷达图：

```powershell
python -c "from experiments.analysis import load_results; from experiments.visualization import plot_constraint_radar; df=load_results('results/main_results.csv'); plot_constraint_radar(df, 'T1', 'results/figures/radar_T1.png')"
```

输出示例：

```text
results/figures/figure_11.png
results/figures/figure_11.pdf
results/figures/radar_T1.png
```

## 14. 补充实验

### 14.1 T1_broadside 侧向交会几何实验

`T1_broadside` 的数据由第 6 节的 `mc_data_generator.py` 默认生成，位置为：

```text
data/mc/T1_broadside/
  seed_42_dev.mat
  seed_42_test.mat
  ...
  seed_46_dev.mat
  seed_46_test.mat
```

它不在 `experiments.runner.SCENARIOS = ["T1", "T2", "T3", "T4"]` 默认列表中，因此主实验命令不会自动运行。需要单独把场景名传给 `run_all_experiments`：

```powershell
python -c "from experiments.runner import run_all_experiments; run_all_experiments(['GSMultiAgent','random_search','cma_es','direct_ppo'], ['T1_broadside'], list(range(42,47)), output_csv='results/supp_t1_broadside.csv', max_workers=1)"
```

运行后可单独汇总：

```powershell
python -c "from experiments.analysis import load_results, compute_summary; df=load_results('results/supp_t1_broadside.csv'); print(compute_summary(df))"
```

### 14.2 LLM 反馈与贪心基线

DeepSeek-V4 自动反馈公平性消融：

```powershell
python -c "from experiments.fairness_ablation import deepseek_v4_auto_feedback; r=deepseek_v4_auto_feedback('T1','data/mc/T1/seed_42_dev.mat','data/mc/T1/seed_42_test.mat',42); print(r)"
```

贪心可行性优先基线：

```powershell
python -c "from experiments.greedy_baseline import greedy_feasibility_first; r=greedy_feasibility_first('T1','data/mc/T1/seed_42_dev.mat','data/mc/T1/seed_42_test.mat',42); print(r)"
```

只做快速接口验证时：

```powershell
$env:GSM_BASELINE_SKIP_LLM='1'
$env:GSM_SUPP_MAX_ROUNDS='1'
$env:GSM_SUPP_MAX_PARAMS='1'
python -c "from experiments.greedy_baseline import greedy_feasibility_first; print(greedy_feasibility_first('T1','data/mc/T1/seed_42_dev.mat','data/mc/T1/seed_42_test.mat',42).keys())"
Remove-Item Env:\GSM_BASELINE_SKIP_LLM
Remove-Item Env:\GSM_SUPP_MAX_ROUNDS
Remove-Item Env:\GSM_SUPP_MAX_PARAMS
```

## 15. 推荐完整执行顺序

```text
1. 安装 Python 依赖，并确认 MATLAB/Octave 命令行可用
2. python experiments\mc_data_generator.py --n-runs 100
3. 验证 data/mc/T1/seed_42_dev.mat 结构
4. 调用 run_mc_with_external_data 做 T1 x 1 run smoke test
5. 分别 smoke test random_search, cma_es, direct_ppo
6. 用 dry_run 跑 GSMultiAgent x T1 x 2 seeds
7. 跑完整主实验；第 4.3 节等预算结果写入 results/experiment_plan/optimizer_equal_mc_75/
8. 消融实验：修改 config.yaml ablation 开关 → python cli_agent.py --file prompt.txt
   （三类消融各自按第 11 节的 config 片段切换，无需修改代码或更换入口命令）
9. 用 analysis.py 生成 LaTeX 表和显著性检验
10. 用 visualization.py 生成 Figure 11 和雷达图
11. 视论文需要运行补充实验
```

## 16. 复现检查清单

运行正式实验前建议确认：

- `data/mc/seed_map.json` 存在，且 `n_runs=100`。
- `data/mc/T1/seed_42_dev.mat` 和 `data/mc/T1/seed_42_test.mat` 都能被 `scipy.io.loadmat` 读取。
- `run_mc_with_external_data(..., n_runs=1)` 返回 `success=True`。
- `config.yaml` 中 `simulation.engine` 和 MATLAB/Octave 路径正确。
- `GSM_BASELINE_SKIP_LLM` 没有在正式 LLM 实验中误设为 `1`。
- `results/*.csv` 中没有大量 `error=True` 行。
- 报告指标使用 `*_test` 字段，而不是只报告 `*_dev` 字段。

## 17. 常见问题

如果看到 `No module named scipy`，说明当前 Python 环境依赖不完整，请安装依赖或切换到 Anaconda Python。

如果看到 `No module named cma`，CMA-ES 基线依赖缺失：

```powershell
pip install cma
```

如果看到 MATLAB 启动错误或 300 秒以上超时，先检查 `config.yaml` 的 `simulation.engine`、`matlab_path`、`octave_path`，再单独运行第 7 节接口测试。

如果并行实验不稳定，优先把 `max_workers=1`，确认单进程稳定后再逐步增大。

如果只想检查 Python 管线，不调用 LLM：

```powershell
$env:GSM_BASELINE_SKIP_LLM='1'
```

如果需要恢复 LLM 调用：

```powershell
Remove-Item Env:\GSM_BASELINE_SKIP_LLM
```
