# GSMultiAgent13 运行手册 (RUNBOOK)

本手册供**接收代码后在 VSCode 本地运行**的同事使用。运行完成后请将 `logs/` 下的日志文件发回，便于判断流程阶段与是否达标。仓库根目录的 `outOfDate/` 是过程文件归档，不要从那里启动实验。

---

## 1. 环境要求

| 组件 | 推荐 |
|------|------|
| Python | 3.10 或 3.11（避免 3.13 + matlab_engine 兼容问题） |
| MATLAB 或 GNU Octave | 至少一种可用，且加入 PATH |
| 依赖 | `pip install -r requirements.txt`（若项目有）或按 README 安装 |

### 1.1 配置 API Key

编辑仓库根目录 `config.yaml`：

```yaml
llm:
  api_key: sk-your-key-here
  base_url: https://api.deepseek.com   # 按实际 provider 修改
  model: deepseek-v4-pro
```

仿真引擎（二选一）：

```yaml
simulation:
  engine: auto            # auto = 自动优选 matlab_engine → matlab → octave → python
  matlab_path: matlab     # 本机可填绝对路径，如 D:/MATLAB/R2025a/bin/matlab.exe
  octave_path: octave
```

> `engine: auto` 会**优先使用 in-process 常驻 `matlab.engine` 会话**（冷启动只付一次，
> 之后单次评估 0.2–0.6 s）；未安装 MATLAB Engine API for Python、或 Python ≥ 3.13 时
> **自动回落** `matlab -batch` 子进程，无需手工切换。想强制走子进程：把 `auto` 改回 `matlab`。
>
> 启用 in-process 支持（需本机 MATLAB + license，Python 须 ≤ 3.12）：
> `cd <matlabroot>/extern/engines/python && <venv>/Scripts/python.exe -m pip install .`
> —— 不装也能跑，只是会回落子进程。

---

## 2. 目录与命令

**工作目录必须为仓库根目录**（与 `config.yaml`、`cli_agent.py` 同级）。

### 2.1 创建日志目录

```powershell
mkdir logs -Force
```

### 2.2 冒烟测试（约 15–30 分钟）

先确认环境、Hermes、MATLAB 能跑通。临时修改 `config.yaml`：

```yaml
workflow:
  max_iterations: 1
matlab_rl_optimizer:
  max_episodes: 10
  nmc_per_eval: 5
  constraint_local_search:
    enabled: false
```

运行（PowerShell）：

```powershell
python cli_agent.py --prompt "T4高频大幅度机动：RUN_CASE='T'; SUB_IDX=4; 命中率>=92%; SEP<=7m; PeakNy<=20g; PM在45-70度; BW在20-85rad/s" 2>&1 | Tee-Object -FilePath "logs\smoke_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
```

**冒烟通过标志：**

- 日志末尾无 Python traceback
- 出现 `[Ep 10/10]` 或 Layer3 完成信息
- `status=success` 或 `needs_iteration`（未中断即可）

### 2.3 正式验证（约 1–2 小时）

恢复 `config.yaml` 中推荐配置（本仓库默认）：

- `max_episodes: 15`（与等预算 `5 × 15` 对齐）
- `episodes_per_update: 16`
- `layer3_strategy: expert`
- `constraint_local_search.enabled: true`

更快冒烟可用根目录 `config.smoke.yaml`。

```powershell
python cli_agent.py --file prompt.txt 2>&1 | Tee-Object -FilePath "logs\run_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
```

若任务不是 `prompt.txt` 中的工况，先改该文件或改用 `--prompt "..."`。

### 2.4 仅 RL 优化（跳过 Hermes 主循环）

```powershell
python run_rl_optimize.py --conditions "T:4" --script guidance_output/scripts/你的脚本.m
```

### 2.5 PPO Checkpoint 恢复（中断后继续）

默认每 10 轮自动保存 checkpoint 到 `parameter_experience_base/checkpoints/ppo/`。

**目录结构：**

```
parameter_experience_base/checkpoints/ppo/
├── latest.json                          # 指向最近一次保存
├── ppo_20260608_120000_ep00010.json     # 元数据（episode、best_reward 等）
├── ppo_20260608_120000_ep00010.npz      # Actor/Critic 权重
├── ppo_20260608_120000_ep00020.json
└── ppo_20260608_120000_ep00020.npz
```

`latest.json` 示例：

```json
{
  "run_id": "20260608_120000",
  "stem": "ppo_20260608_120000_ep00020",
  "json_path": "ppo_20260608_120000_ep00020.json",
  "npz_path": "ppo_20260608_120000_ep00020.npz",
  "episode_completed": 20
}
```

**恢复命令（三种路径写法均可）：**

```powershell
# cli_agent — 指向目录（自动读 latest.json）
python cli_agent.py --prompt "T4..." --resume-checkpoint ./parameter_experience_base/checkpoints/ppo

# cli_agent — 指向具体 .json
python cli_agent.py --file prompt.txt --resume-checkpoint ./parameter_experience_base/checkpoints/ppo/ppo_20260608_120000_ep00020.json

# run_rl_optimize — 独立 RL 脚本恢复
python run_rl_optimize.py --conditions "T:4" --script guidance_output/scripts/你的脚本.m --resume-checkpoint ./parameter_experience_base/checkpoints/ppo
```

**离线分析 JSONL 滚动统计**（脚本已归档，需要时从 `outOfDate/scripts_ops/rl_jsonl_stats.py` 取回）：

```powershell
python outOfDate/scripts_ops/rl_jsonl_stats.py logs/rl_episodes_*.jsonl --window 20
```

---

## 3. 发回哪些文件

请打包发送：

| 文件 | 说明 |
|------|------|
| `logs/run_*.txt` 或 `logs/smoke_*.txt` | **必发** — 完整终端输出 |
| `logs/rl_episodes_*.jsonl` | 若有 — 逐 episode 指标 |
| `guidance_output/report_*.md` | 若有 — 最终报告 |
| `config.yaml` | 建议脱敏 api_key 后发送 |

---

## 4. 日志阶段对照表

用下面关键字在日志中搜索，判断运行进度。

### 4.1 启动

| 关键字 | 含义 |
|--------|------|
| `[Step 1]` / `Preflight` | 环境检查 |
| `Hermes initialized` | Layer2 Agent 就绪 |
| `Layer 3 ready` | PPO/Expert 优化器就绪 |
| `PE warm-start` | 参数经验库加载 |

### 4.2 Layer2（脚本生成与仿真）

| 关键字 | 含义 |
|--------|------|
| `mode=MODIFY_LAW` / `TUNE_PARAMS` | Planner 选择的路径 |
| `generate_matlab` | 生成 MATLAB 脚本 |
| `verify_guidance_compat` | 制导律兼容验证 |
| `run_simulation` | Layer2 仿真 |
| `judge_requirements` + `[OK]` / `[NG]` | Layer2 指标判定 |
| `PeakNy(最大值)` | 应使用 MC max 判据（非仅均值） |

### 4.3 Layer3 PPO

| 关键字 | 含义 |
|--------|------|
| `RL networks initialised` + `pitch_std` | PPO 开始 |
| `Baseline reward=` | 基线评估 |
| `[FRE] preset SEP` | FRE 阈值（T4 应为 500→7m） |
| `[Ep N/15]` | 第 N 轮 episode（`max_episodes=15`） |
| `PeakN=XX(max=YY)` | 当前轮均值与 MC 最大值 |
| `reward=-10` 频繁 | 发散探索过多（改后应减少） |
| `best_SEP_ever=` | 历史最优脱靶量 |
| `[FRE] Phase X/Y` | 可行域阶段 |
| `[RL Reflection @ Ep N]` | 反思（每 5 ep 一次） |
| `Multi-criteria requirements met` | **规则早停成功** |
| `task_requirements_met` | **Reflection 早停成功** |
| `Patience early stop` | 停滞早停 |
| `status=interrupted` | 人为 Ctrl+C，非达标 |
| `[RollingStats @ Ep N]` | 滚动窗口均值（reward/SEP/PeakN/hit/diverge%） |
| `[Checkpoint] Saved` | PPO 权重已写入 checkpoint 目录 |
| `[RL] Resuming from checkpoint` | 从 checkpoint 恢复，跳过已完成 episode |

### 4.4 CLS 精修

| 关键字 | 含义 |
|--------|------|
| `[Layer 3/CLS]` | PPO 后邻域搜索 |
| `[CLS] Done` + `PeakN_max=` | CLS 完成 |

### 4.5 结束

| 关键字 | 含义 |
|--------|------|
| `status=done` | 外层迭代达标 |
| `needs_iteration` | 未达标，将进入下一轮 |
| `total_episodes` | 实际 PPO 轮数 |

---

## 5. 论文预算与冒烟

正式对照实验使用 `max_episodes: 15`、外层 5 轮，开发集最多 225 次 MC。冒烟用 `config.smoke.yaml` 或临时把 `max_episodes` 降到 10、`nmc_per_eval` 降到 5。历史 T4 调参笔记已归档到 `outOfDate/docs_old/`。

---

## 6. 常见问题

### Q: `OPENAI_API_KEY` / Chroma 警告

RAG 旧 collection 元数据告警，一般**不影响**主流程；DashScope embedding 成功即可继续。

### Q: MATLAB 超时

检查 `simulation.engine` 与路径；Layer2 探索阶段 nmc 应 ≤30（见 `prompt.txt` 第 5 条）。

### Q: `status=interrupted`

用户中断，不是早停 bug。请跑满或等待 `task_requirements_met`。

### Q: PM/BW 全轮相同 ±0

T 工况下 PM/BW 为设计点常量，**正常现象**；看 Miss / PeakNy(max) 是否变化。

---

## 7. config 快速切换

| 场景 | 修改项 |
|------|--------|
| 更快冒烟 | 使用 `config.smoke.yaml`，或 `max_episodes: 10`, `nmc_per_eval: 5`, CLS `enabled: false` |
| T4 正式 | 使用仓库默认 `config.yaml`（`max_episodes: 15`） |
| 仅 PPO 不调 Expert | `layer3_strategy: ppo` |
| 关闭 CLS | `constraint_local_search.enabled: false` |
| 关闭 checkpoint | `matlab_rl_optimizer.checkpoint.enabled: false` |
| 调整保存频率 | `checkpoint.save_every_episodes: 5`, `keep_last_n: 3` |

---

## 8. 联系开发方时请注明

1. 日志文件名与时间  
2. 使用的 `--prompt` 或 `prompt.txt`  
3. `config.yaml` 中是否改过 `max_episodes` / `layer3_strategy`  
4. MATLAB 还是 Octave、Python 版本  
