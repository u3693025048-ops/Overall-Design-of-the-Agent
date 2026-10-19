# GSMultiAgent —— 制导系统多智能体设计与优化框架

基于 **三层架构** + **子智能体编排** + **PPO-RL 参数优化** + **判断/反思智能体** 的制导律与自动驾驶仪协同设计系统。以 `monte_carlo_single.m` 为仿真模板，支持 T/G/AP/R 四类工况的蒙特卡洛评估。

---

## 三层架构

```
┌────────────────────────────────────────────────────────────────────┐
│  LAYER 1  自然语言任务输入  (cli_agent.py)                          │
│  - 解析工况描述 → T/G/AP/R 类别映射                                │
│  - 解析性能要求 (命中率 / SEP / PeakNy / PM / BW)                  │
│  - 初始化全部组件，驱动 [Layer2→Layer3] 主迭代循环                  │
│  - 人工中断: Ctrl+C → 菜单（修改指令 / 晋升记忆 / 继续 / 退出）     │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│  LAYER 2  Hermes Agent（纯编排器 + 子智能体委托）                   │
│                                                                    │
│  编排方式：                                                         │
│  ① 直接调用工具（直接模式）                                         │
│  ② call_specialist(agent_type, task, context) 委托子智能体           │
│                                                                    │
│  五类专用子智能体（各自工具隔离，防递归）:                           │
│  ┌ rag_agent      → rag_query / rag_retrieve / rag_expand          │
│  ├ matlab_agent   → generate_matlab / syntax_check / compat_verify │
│  ├ sim_agent      → run_simulation / syntax_check_matlab           │
│  ├ analysis_agent → judge_requirements / parameter_experience_*    │
│  └ memory_agent   → agent_memory_remember/recall/list/forget       │
│                                                                    │
│  MODIFY_LAW 推荐委托流程:                                           │
│  rag_agent → matlab_agent → sim_agent → analysis_agent             │
│  （每步 JSON 输出作为下步 context 传入）                            │
│                                                                    │
│  仿真成功后 → judge_requirements → 两条出口:                        │
│  ── satisfied=True  → 跳过 RL，直接进 Layer3 反思 → 经验回写        │
│  ── satisfied=False → 进入 Layer3 RL 参数优化环节                   │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ script_path + initial_metrics
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│  LAYER 3  优化工作流  (optimization_workflow.py, async)             │
│                                                                    │
│  ┌──────────────────┐    每 episode     ┌─────────────────────┐   │
│  │  PPO-RL 优化器    │ ←──────────────→ │  判断智能体          │   │
│  │  10 维参数:       │   新最优时触发    │  ReflectionAgent    │   │
│  │  w1,zeta1,tao1   │                  │  规则检查→早退出     │   │
│  │  w2,zeta2,tao2   │                  │  (规则通过才调LLM)   │   │
│  │  w3,zeta3,N_pn   │                  └─────────────────────┘   │
│  │  ny_lim          │                                             │
│  └────────┬─────────┘                                             │
│           │ 满足 / 达最大轮数                                       │
│           ▼                                                        │
│  反思智能体 (ReflectionAgent) — LLM 深度分析                        │
│     满足 → 经验回写 (params + model) → 仿真报告 (JSON)              │
│     不满足 → 优化建议 → 返回 Layer1 (tune_params / modify_law)      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 文档索引

| 文档 | 用途 |
|------|------|
| [README.md](README.md) | 架构、目录、快速开始（本文） |
| [RUNBOOK.md](RUNBOOK.md) | 本地环境配置与冒烟/正式运行 |
| [experiments/README.md](experiments/README.md) | 实验协议、矩阵、第 4.2/4.3 节复现步骤 |
| [experiments/single_agent_hermes/README.md](experiments/single_agent_hermes/README.md) | 单智能体 Hermes 基线 |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | 模型 ID、实验日期、等 MC 预算 |
| [SUPPLEMENT.md](SUPPLEMENT.md) | 期刊补充材料 ZIP 说明 |
| [paper_revision/处置方式.md](paper_revision/处置方式.md) | 审稿回复与打包口径 |
| [outOfDate/README.md](outOfDate/README.md) | 过程文件 / 旧文档归档（非入口） |

---

## 项目结构

```
GSMultiAgent13-V0.3/
├── cli_agent.py                         # 端到端 CLI 入口（Layer1）
├── config.yaml                          # 运行配置（LLM / RAG / RL / 消融开关）
├── config.smoke.yaml                    # 冒烟配置
├── prompt.txt                           # 任务描述示例
├── requirements.txt
├── run_preflight.py                     # 环境预检
├── run_rl_optimize.py                   # 单独调试 PPO
│
├── experiments/                         # 外部 MC 实验、基线、消融
│   ├── README.md                        # 协议与复现（含第 4.3 节等预算命令）
│   ├── runner.py                        # 主实验/消融批量入口
│   ├── optimizer_baselines_43.py        # Random / CMA-ES / Direct PPO / 单智能体
│   ├── equal_mc_budget.py               # 等预算：75 候选 / 225 MC calls
│   └── mc_data_generator.py             # dev/test 外部 MC 数据生成
│
├── multi_agent/                         # 核心包
│   ├── config_loader.py
│   ├── integration/                     # Hermes、规划、判断、反思、子智能体
│   ├── simulation/                      # Layer3 工作流与仿真引擎
│   ├── rl/                              # PPO 10 维参数优化
│   ├── memory/                          # 参数经验库、Hermes 记忆、RAG
│   └── tools/                           # Layer2 工具（检索/生成/仿真/判断/委托）
│
├── knowledge_base/                      # RAG 源与 monte_carlo_single.m 模板
├── tests/
├── scripts/                             # 打包、拉结果、批量实验（运维一次性脚本已归档）
├── logs/                                # 当前正式运行日志（等预算主日志）
├── results/experiment_plan/             # 主表：消融 CSV + optimizer_equal_mc_75*
├── paper_revision/                      # 回复信、投稿 ZIP（解压快照已归档）
├── parameter_experience_base/           # 运行期经验库（跨任务保留）
├── chroma_db/                           # 向量库缓存（运行生成）
└── outOfDate/                           # 过程文件、旧输出、重复说明（已归档）
```

运行时会在根目录重建 `guidance_output/`。历史输出、冒烟结果、不等预算基线与一次性运维脚本在 `outOfDate/`。

---

## 快速开始

```bash
# 创建并激活虚拟环境 (Windows)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 可选：Layer 2 Hermes 编排（提供 run_agent；无此包时 CLI 仍可导入并完成 Step 1）
pip install git+https://github.com/NousResearch/hermes-agent.git

# 实验运行方式：从 prompt 文件读入任务
python cli_agent.py --file prompt.txt
python cli_agent.py --file prompt_books_rag.txt
```

> 编辑 `config.yaml` 设置 LLM provider/model/api_key 和仿真引擎（octave/matlab）。

---

## 设计模式

`IntelligentTaskPlanner` 由 **Hermes Agent 的语义分析**决定执行模式（非关键词匹配）：

| 模式 | 选择条件 | Layer2 执行方式 |
|------|---------|----------------|
| **REUSE_HISTORY** | 历史参数已足够，无需改算法或重新调参 | 直接用 KB 模板 + 历史最优 `dp_params` 仿真 |
| **TUNE_PARAMS** | 需改善性能，调整驾驶仪/制导律参数 | 生成新脚本 → 校正 → 仿真 |
| **MODIFY_LAW** | 需设计新制导算法（改数学结构） | rag_agent → matlab_agent → sim_agent → analysis_agent |

**TUNE_PARAMS / REUSE_HISTORY 可调参数（RL 优化 10 维）：**

| 类别 | 参数 | 含义 | 默认值 |
|------|------|------|--------|
| 驾驶仪 | `w1`, `zeta1`, `tao1` | 俯仰通道带宽/阻尼/时间常数 | 40, 0.75, 0.15 |
| 驾驶仪 | `w2`, `zeta2`, `tao2` | 偏航通道带宽/阻尼/时间常数 | 35, 0.75, 0.15 |
| 驾驶仪 | `w3`, `zeta3` | 滚转通道带宽/阻尼 | 40, 0.70 |
| 制导律 | `N_pn` | 比例导引比 | 4.0 |
| 制导律 | `ny_lim` | 制导过载限幅（g） | 20.0 |

---

## 任务工况体系（T/G/AP/R）

`monte_carlo_single.m` 定义四类工况，通过 `RUN_CASE` + `SUB_IDX` 选择：

| RUN_CASE | 名称 | 子工况 |
|---------|------|------|
| **T** | 目标机动 | T1:匀速直飞, T2:低频小幅, T3:中频中等, T4:高频大幅 |
| **G** | 交战几何 | G1:标称, G2:近距小偏, G3:近距大偏, G4:远距小偏, G5:远距大偏 |
| **AP** | 驾驶仪退化 | AP1:全标称, AP2:时间常数退化, AP3:轻度, AP4:中度, AP5:重度 |
| **R** | 综合鲁棒 | R1:全标称, R2:多参数摄动, R3:大姿态偏差 |
| **ALL** | 全工况 | 依次运行 T/G/AP/R |

---

## Layer 2 工具清单

> RL 优化完全由 **Layer 3 OptimizationWorkflow** 执行，不在 Hermes 工具集内。

| 工具名 | 所属子智能体 | 功能 |
|--------|------------|------|
| `rag_query` | rag_agent | Agentic 多跳检索（自动分解子查询） |
| `rag_retrieve` | rag_agent | 单次向量检索知识库 |
| `rag_expand` | rag_agent | 按 ID 获取文档全文 |
| `generate_matlab` | matlab_agent | 生成 .m 脚本（monte_carlo_single 模板） |
| `syntax_check_matlab` | matlab_agent / sim_agent | 验证并修复 .m 文件 |
| `verify_guidance_compat` | matlab_agent | 数学适配性证明（带宽分离/Adjoint/PM） |
| `run_simulation` | sim_agent | 运行 monte_carlo_single，输出命中率/SEP/PeakNy/PM/BW |
| `judge_requirements` | analysis_agent | 判断仿真指标是否满足任务要求 |
| `parameter_experience_best` | analysis_agent | 获取历史最优参数 |
| `parameter_experience_search` | analysis_agent | 搜索相似历史参数 |
| `agent_memory_remember` | memory_agent | 持久化写入长期记忆（JSON KV） |
| `agent_memory_recall` | memory_agent | 读取长期记忆条目 |
| `agent_memory_list` | memory_agent | 列出所有记忆键 |
| `agent_memory_forget` | memory_agent | 删除记忆条目 |
| `call_specialist` | Hermes 编排器 | 委托子智能体执行（含 agent_type 路由） |

---

## 子智能体间数据传递

子智能体通过 **`context` 参数**传递上一步的结构化 JSON 输出：

```
Hermes 编排器
│
├─ RAG_RESULT  = call_specialist(agent_type='rag_agent', task='检索APN制导律知识')
│                → 返回 JSON: {law_type, key_params, design_principles, sources}
│
├─ MATLAB_RESULT = call_specialist(agent_type='matlab_agent',
│                    task='生成脚本 mission_conditions="G:3"',
│                    context=RAG_RESULT)          ← RAG结果作为背景
│                  → 返回 JSON: {script_path, compat_overall, compat_score, key_params}
│
├─ SIM_RESULT  = call_specialist(agent_type='sim_agent',
│                    task='运行仿真 script_path=<MATLAB_RESULT中的script_path>')
│                → 返回 JSON: {status, hit_rate, SEP, PeakNy, PM, BW}
│
└─ JUDGE_RESULT = call_specialist(agent_type='analysis_agent',
                     task='判断是否满足需求',
                     context=SIM_RESULT)          ← 仿真指标作为背景
                   → 返回 JSON: {satisfied, layer3_path, metrics_vs_req}
```

每个子智能体强制在响应末尾输出 ` ```json ... ``` ` 块，Hermes 从中提取 `script_path`、`satisfied` 等字段。

---

## PPO 强化学习参数优化

`MatlabRLOptimizer` 对 **monte_carlo_single.m 的 `RL_PARAMS` 块**进行 **10 维** PPO 优化（8 驾驶仪 + N_pn + ny_lim）。

**奖励函数**（`config.yaml` 中 `reward_weights` 可调）：
```
reward = w_hit × hit_rate
       + sep_bonus(SEP)
       + pm_bonus(PM)          # 在目标范围 [pm_min, pm_max] 内加分
       + bw_bonus(BW)
       - peak_ny_penalty(PeakNy)
```

**反思智能体作为早停决策者**：

| 触发 | 退出状态 | 说明 |
|-----|---------|------|
| `ReflectionAgent` 返回 `needs_optimization=False` | `task_requirements_met` | 所有要求满足，立即停止 |
| 跑满 `max_episodes` | `success` | 自然结束，输出最优 |

**典型日志**：
```
[Ep 3/20] reward=2.821  hit=85.0%  SEP=2.94m  PM=38.2°  BW=22.1r/s
[RL Reflection @ Ep 3] needs_optimization=False → 已满足全部要求
✓ RL early-stopped (episode 3): task_requirements_met
```

---

## 实验设计总览

详细协议与命令见 [experiments/README.md](experiments/README.md)。dev/test 分离：每个 master seed 的 dev `.mat` 用于搜索与优化，独立 test `.mat` 只评估最终参数。

**修订稿主实验矩阵**：

| 维度 | 设置 |
|------|------|
| 方法 | `GSMultiAgent`, `random_search`, `cma_es`, `direct_ppo`, `single_agent_hermes` |
| 工况 | `T1`, `T2`, `T3`, `T4` |
| 主种子 | `42`, `43`, `44` |
| 数据 | 每主种子 3 个 dev + 3 个 test `.mat`，各 100 条弹道 |
| 等预算上限 | 75 候选 × 3 个 dev 种子 = **225** 次开发集 MC（`experiments/equal_mc_budget.py`） |
| 约束 | 命中率 ≥92%，SEP ≤7 m，PeakNy ≤20 g，PM ∈ [45, 70]°，BW ∈ [20, 85] rad/s |

CMA-ES 按候选评估次数停，不按代数停。记忆复用与约束满足后的提前停止可以使用更少次数，但不能超过 225。

**结果目录**：

| 内容 | 路径 |
|------|------|
| 等预算 Random / CMA-ES / Direct PPO | `results/experiment_plan/optimizer_equal_mc_75/` |
| 等预算单智能体（对齐口径） | `results/experiment_plan/optimizer_equal_mc_75_sah_n10/` |
| 消融 CSV | `results/experiment_plan/ablation_*.csv` |

**第 4.2 / 4.3 节入口**（命令细节在 `experiments/README.md`）：

```powershell
# 4.2 插拔优化器 / 级联对照
python experiments/section_4_2_runner.py --experiment optimizer --scenarios T1 T2 T3 T4 --seeds 42 43 44
python experiments/section_4_2_runner.py --experiment cascade --scenarios T1 T2 T3 T4 --seeds 42 43 44

# 4.3 等预算基线（默认 75 候选、n_runs=100）
python experiments/optimizer_baselines_43.py --max-candidates 75 --n-runs 100 --output-dir results/experiment_plan/optimizer_equal_mc_75
```

**消融实验分三类，统一通过修改 `config.yaml` 插拔模块，然后执行 `python cli_agent.py --file prompt.txt` 运行**：

| 消融类型 | 组别 | 切换方式 | 验证目标 |
|---------|------|---------|---------|
| 记忆消融 | `E0`～`E3` | `ablation.parameter_experience_reuse` / `ablation.memory_search` | 参数经验库和持久记忆对跨任务复用的贡献 |
| 优化管线消融 | `O0`～`O4` | `ablation.guidance_compat_verification` / `ablation.syntax_check` / `ablation.rl_optimization` / `workflow.max_iterations` | 适配验证、语法检查、RL 和多轮迭代各自的贡献 |
| 修正管线消融 | `C0`～`C2` | `ablation.guidance_compat_verification` / `ablation.syntax_check`（仅 T4） | T4 MODIFY_LAW 场景下制导律修正管线的稳定性 |

详细 config 片段和执行步骤见 [experiments/README.md § 11](experiments/README.md#11-运行消融实验)。

**结果统计指标**：

| 指标 | 含义 |
|------|------|
| `first_success_rate` | dev 阶段首次得到可行解的比例 |
| `feasible_test` | 最终参数在独立 test MC 上是否满足全部约束 |
| `hit_rate_test`, `SEP_test`, `peak_nz_test`, `PM_test`, `BW_test` | 独立 test 集上的主要性能指标 |
| `mc_calls_dev + mc_calls_test` | Monte Carlo 调用成本 |
| `wall_clock_sec`, `inference_sec`, `orchestration_sec`, `sim_sec` | 总耗时、LLM 推理耗时、编排耗时和仿真耗时 |
| `iterations_used`, `design_path` | 收敛轮次和最终设计路径 |

---

## 配置文件关键字段

```yaml
llm:
  provider: custom             # openai | anthropic | openrouter | custom
  base_url: https://api.deepseek.com
  model:    deepseek-v4-pro
  api_key:  <your key>

rag:
  embedding_provider: custom
  embedding_model:    text-embedding-v3
  embedding_dim:      1024

workflow:
  max_iterations:    5         # 外层 Layer2→Layer3 迭代上限
  hermes_execution:  true
  hermes_tool_timeout: 0       # 0=不限时

matlab_rl_optimizer:
  max_episodes:  15            # PPO 最大回合数
  nmc_per_eval:  25            # 每回合蒙特卡洛次数

simulation:
  engine: auto                 # auto 自动优选 matlab_engine → matlab → octave → python
  mc_data_path: ""             # 外部 MC .mat 路径；为空时使用模板内部随机样本
```

---

## 数据与复现

不公开 GitHub。源代码、MC 初值生成器、主实验用 `.mat` 与去密钥配置打成期刊 **Supplementary ZIP**（`python scripts/pack_supplement.py`）。交战条件为按固定种子仿真抽样，不是实装遥测；bootstrap / 事后合成 CSV **不进入主表、不打进 ZIP**。模型版本与实验日期见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)，打包说明见 [SUPPLEMENT.md](SUPPLEMENT.md) 与 [paper_revision/处置方式.md](paper_revision/处置方式.md)。

所有基线与 GSMultiAgent 使用**相同的最大开发集 Monte Carlo 调用次数**：`5 × 15 × 3 = 225`（`experiments/equal_mc_budget.py`）。CMA-ES 按候选评估次数停，不按代数停。

---

## 已知问题

- **Octave/MATLAB 未安装**：RL 自动回退 Python 仿真，不阻塞训练
- **LLM 返回空响应**：TaskPlanner 默认返回 `REUSE_HISTORY`（最安全路径）
- **`agent_memory_*` 报 "Memory not initialized"**：根因是 `HermesAgentMemory.__len__==0` 被当作 falsy，已修复为 `is not None` 判断
- **`roots: NaN`**：RL 极端参数导致驾驶仪设计矩阵 Inf/NaN，模板已加 `safe_coef()` + `try/catch` 防护

---

## 替换仿真模板

替换 `monte_carlo_single.m` 时需同步修改的文件：

| # | 文件 | 检查点 |
|---|------|--------|
| 1 | 新 `.m` 模板 | `RL_PARAMS_BEGIN/END` 块存在，`rl_*` 变量完整 |
| 2 | 新 `.m` 模板 | stdout 含命中率/SEP/PeakNy/PM/BW 字样 |
| 3 | `matlab_rl_optimizer.py` | `AUTOPILOT_PARAM_SPECS`、`RL_PARAM_TO_MATLAB` 与模板一致 |
| 4 | `matlab_rl_optimizer.py` | `CATEGORY_DEFS` 工况编号与模板 SC{} 数组一致 |
| 5 | `conditions.md` | 子工况编号与物理含义同步 |
| 6 | `simulation_tool.py` | KB 模板默认路径指向新文件 |
| 7 | `config.yaml` | 奖励权重和指标阈值匹配新模板量纲 |

---

## 知识库管理 + RAG Web 端

知识库管理 Agent（位于智能层）的混合检索内核与可视化管理办法。对应需求文档 §2.7「知识库管理 + RAG」。

### 设计要点
- **四库模型**：`norm`(规范库) / `model`(模型库) / `template`(模板库) / `experience`(经验库)。
- **混合检索**：向量(Chroma) + BM25 + RRF 融合(k=60)，再按 `level`(总体/几何-气动/内回路/外回路/验证) + `case`(T/G/AP/R/ALL) + `lib` 硬过滤。
- **已知路径直取**：固定文档（如 `next_action.json`、画布模板）用 `doc_id` 直接命中，避免每轮语义重搜。
- **反思产物**：`next_action.json`（机器可校验的下一步动作），由反思 Agent 产出；`propose_next_action()` 可生成草稿。
- **反馈闭环**：`record_feedback(doc_id, success)` 调置信度 ±0.05，低于 0.2 判失效，写入 `kb_bm25_corpus.feedback.jsonl`。

### 目录与文件
| 文件 | 作用 |
|------|------|
| `multi_agent/memory/rag_knowledge_base.py` | 复用的向量库后端 `RAGKnowledgeBase`（已加 `get_all_documents()` 只读全量导出） |
| `multi_agent/memory/knowledge_manager.py` | 新内核：`KBHybrid`(混合检索) + `KnowledgeManager`(四库播种/管理) |
| `web/app.py` | FastAPI 管理端接口 |
| `web/static/index.html` | 暗色单页管理界面（检索/文档/经验卡/新增） |
| `run_web.py` | 启动器（读 `config.yaml` 的 `web:` 段 + `rag.bm25_corpus_path`） |

### 运行
```bash
pip install -r web/requirements.txt
python run_web.py          # 读 config.yaml → web.enabled/host/port
# 浏览器打开 http://localhost:8787
```
- 知识库为空时自动执行「四库播种」（规范 md / 模型 sysml+matlab+books / 模板 / 经验卡）。
- 已索引则仅重建 BM25 互补索引，保证混合检索覆盖。
- 若 `chromadb`/embedding 未就绪，`ensure_ready()` 优雅降级，Web 端仍可启动（检索为空）。

### 接口
`GET /api/stats` · `GET /api/docs`(lib/level/case/q 过滤) · `GET /api/search` · `GET|POST|DELETE /api/doc/{doc_id}` · `POST /api/reindex` · `POST /api/feedback` · `POST /api/next_action` · `GET /api/experience`

---

## License

MIT
