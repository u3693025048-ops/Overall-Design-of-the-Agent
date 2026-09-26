# SingleAgentHermes — 单智能体 LLM 顾问基线

## 定位

| 特性 | GSMultiAgent | **SingleAgentHermes** | direct_ppo |
|------|-------------|----------------------|------------|
| 参数经验库（PE）热启动 | ✓ | **✓** | ✗ |
| PPO 优化器 | ✓ | **✓** | ✓ |
| 修正管线（参数裁剪 + sanitizer）| ✓ | **✓** | ✗ |
| LLM 顾问（每轮外层迭代后反思）| ✓ | **✓** | ✗ |
| 多智能体级联 / 角色分工 | ✓ | **✗** | ✗ |
| 反思诊断（完整 ReflectionAgent）| ✓ | **部分** | ✗ |
| RAG 检索 / HermesAgentMemory | ✓ | **✗** | ✗ |

**作用**：在保留 PE 热启动 + PPO 的基础上，加入轻量 LLM 顾问，隔离「完整多智能体级联」vs「单 Agent + LLM 建议」的贡献差距。

## 架构

### LLM 顾问后端优先级

```
HermesIntegration（run_agent.AIAgent）
    ↓ 不可用时
LangChain ChatOpenAI（与 ReflectionAgent 同配置）
    ↓ 不可用 / 无 api_key 时
静默 no-op（退化为纯 PPO 基线，行为等价于旧版消融基线）
```

后端由 `_HermesLLMAdvisor._init_backend()` 在构造时自动探测，运行日志会报告实际使用的后端：

```
[SAH] LLM advisor active (backend=hermes)
[SAH] LLM advisor active (backend=langchain)
[SAH] LLM advisor disabled or unavailable; running as pure-PPO baseline.
```

### Hermes Agent 接入方式

基于 Hermes Agent（`run_agent.AIAgent`）源码中的实际接口：

**核心接口**（`run_agent.py`）：

```python
from run_agent import AIAgent

agent = AIAgent(
    base_url=cfg.base_url,
    api_key=cfg.api_key,
    provider=cfg.provider,
    model=cfg.model,
    max_iterations=5,                          # 文本生成无需多轮
    enabled_toolsets=["__sah_advisory__"],     # 不存在的 toolset → 0 工具加载
    quiet_mode=True,                           # 抑制控制台输出
    skip_context_files=True,                   # 跳过 SOUL.md / AGENTS.md 注入
)

# run_conversation 是同步方法（非 async）
result = agent.run_conversation(
    user_message=user_msg,
    system_message=SYSTEM_PROMPT,              # 单次调用注入 system prompt
)
raw = result.get("final_response") or ""      # 提取文本响应
```

**工具注册 API**（`tools/registry.py`，本模块不需注册工具，仅供参考）：

```python
from tools.registry import registry

registry.register(
    name="tool_name",
    toolset="my_toolset",
    schema={"name": "...", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: json.dumps({"result": "..."}),
    is_async=False,          # True 时 registry.dispatch() 自动桥接
)
```

`enabled_toolsets=["__sah_advisory__"]` 在 registry 中找不到匹配项，`AIAgent.valid_tool_names` 为空集合，LLM 仅做纯文本生成，无任何工具调用。

### LLM 输入 / 输出契约

**输入**（`context` 字段）：

```
场景: T4
外层迭代: 2/5
当前最优指标: {"hit_rate": 88.0, "SEP": 9.2, "PM": 42.1, "BW": 24.3}
当前最优得分: -0.880001
最近 10 个 episode 历史:
  ep=1  score=-0.810  all_pass=False  backend=torch
  ep=2  score=-0.880  all_pass=False  backend=torch
  ...
```

**输出**（`_LLMAdvice` 数据类）：

| 字段 | 类型 | 含义 |
|------|------|------|
| `continue_optimization` | `bool` | `false` 时触发外层早停 |
| `suggestion` | `str` | 中文简要说明（≤50字） |
| `param_hints` | `dict[str, float]` | 参数偏移建议，`delta_fraction ∈ [-0.15, 0.15]`，应用于下一轮 warm-start |
| `rl_adjustments` | `dict` | RL 超参建议，可含 `lr`、`n_episodes`、`entropy_coef` |

LLM 返回的 JSON 示例：

```json
{
  "continue_optimization": true,
  "suggestion": "命中率距达标尚差 4%，建议增大 w1 带宽并适当降低 N_pn",
  "param_hints": {"w1": 0.08, "N_pn": -0.05},
  "rl_adjustments": {"n_episodes": 20}
}
```

`param_hints` 在写入 PE 前会被安全 clamp 到 `[-0.15, 0.15]`，再经 `_clip()` 限制到各参数合法范围。

## 优化流程

```
for outer in 1..N_OUTER:
    warm_params ← ParameterExperience.retrieve(scenario)  # 或 DEFAULT_PARAMS
    best_params, score ← PPO(warm_params, n_episodes, dev_seeds)
    ParameterExperience.store(scenario, best_params)

    [LLM 顾问调用，仅在非末轮执行]
    advice ← _HermesLLMAdvisor.advise(scenario, outer, best_metrics, history)
    if advice.param_hints:
        nudged = best_params + delta_frac × (hi - lo)    # 参数偏移
        ParameterExperience.store(scenario, nudged)       # 覆写为偏移后的热启动
    if advice.rl_adjustments.n_episodes:
        n_episodes ← advice.rl_adjustments.n_episodes     # 下轮 episode 数调整
    if not advice.continue_optimization:
        break                                              # LLM 建议早停

test_result ← evaluate(best_params, test_seeds, split="test")
```

> LLM 顾问**仅在外层迭代间**调用（每轮 PPO 结束后），不介入 episode 内部。LLM 不可用时（后端均失败）整个顾问块静默跳过，不影响 PPO 运行。

## 预算对齐（与 GSMultiAgent 相同）

| 参数 | 值 | 来源 |
|------|-----|------|
| `n_outer_iterations` | 5 | `workflow.max_iterations` |
| `n_episodes_per_outer` | 15 | `matlab_rl_optimizer.max_episodes` |
| 搜索 `n_runs` | **10** | `simulation.mc_search_n_runs`（`mc_tiered_eval`） |
| 终检 `n_runs` | **100** | `simulation.mc_final_n_runs` / CLI `--n-runs` |
| 3-seed AND | 每次真跑 3 次 MATLAB | `bootstrap_fill=False`（比 GSM 的 1 真+2 抽样更严） |
| `first_success`（Table 5） | 最终 dev 是否可行 | 与 GSM `experiments/runner.py` 同一句：`final_dev_feasible` |
| `first_success_iter` | 第一次出现可行 episode 的外层 | 不作为 Table 5 布尔列 |
| dev seeds | master×100+{1,2,3} | AND-pass，与其他基线一致 |

总 MATLAB 调用上限 = `5 × 15 × 3 seeds = 225`（dev）+ `3`（test）。搜索弹道宽度为 10，不能用旧的 100 条结果事后折算。  
LLM 调用 ≤ `N_OUTER - 1 = 4` 次（末轮不调用）。

## 配置

LLM 后端来自 `config.yaml → llm.*`，无需单独配置：

```yaml
llm:
  provider: custom
  base_url:  https://api.deepseek.com
  model:     deepseek-v4-pro
  api_key:   <your key>
```

禁用 LLM 顾问（退化为纯 PPO 消融基线）：

```python
agent = SingleAgentHermes(..., use_llm=False)
# 或通过 run_single_agent_hermes(..., use_llm=False)
```

也可通过环境变量跳过 LLM（仅影响 runner 层）：

```powershell
$env:GSM_BASELINE_SKIP_LLM='1'
```

## 返回结果

`run()` 返回字典新增以下字段（在原有字段基础上追加）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `llm_advice_log` | `list[dict]` | 每轮 LLM 反思记录（`outer`, `continue`, `suggestion`, `param_hints`, `rl_adjustments`） |
| `llm_backend` | `str` | 实际使用的 LLM 后端（`"hermes"` / `"langchain"` / `"none"`） |
| `iterations_used` | `int` | 实际执行的外层迭代数（LLM 早停时 < `iterations_budget`） |

## 运行

已集成到 `optimizer_baselines_43.py`，直接加 `single_agent_hermes` 到 `--methods`：

```bash
python experiments/optimizer_baselines_43.py \
  --methods single_agent_hermes \
  --scenarios T1 T2 T3 T4 \
  --master-seeds 42 43 44 \
  --max-candidates 75 \
  --n-runs 100 \
  --output-dir results/experiment_plan/optimizer_equal_mc_75_sah
```

也可直接调用：

```python
from experiments.single_agent_hermes.agent import run_single_agent_hermes
from pathlib import Path

result = run_single_agent_hermes(
    evaluator_mc_root=Path("data/mc"),
    matlab_script=Path("knowledge_base/matlab/guidance/monte_carlo_single.m"),
    scenario="T4",
    master_seed=42,
    dev_seeds=[4201, 4202, 4203],
    test_seeds=[4204, 4205, 4206],
    n_outer_iterations=5,
    n_episodes_per_outer=15,
    n_runs=100,
    use_llm=True,   # False → 退化为纯 PPO 消融基线
)
print(result["llm_backend"], result["llm_advice_log"])
```

## 安装依赖

Hermes Agent（主路后端）：

```bash
pip install git+https://github.com/NousResearch/hermes-agent.git
```

LangChain 备路（仅备用，通常已随项目主依赖安装）：

```bash
pip install langchain-openai
```

两者均未安装时，`use_llm=True` 仍可运行，`llm_backend` 报告为 `"none"`，行为等价于旧版消融基线。
