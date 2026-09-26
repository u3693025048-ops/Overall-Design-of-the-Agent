# 回复审稿人 #1（第二轮）

稿件：aerospace-4439606  
感谢审稿人。以下回应第 1、3、4、5 条。源代码与数据随本次修订以补充材料 ZIP 提交，不通过 GitHub 公开。第 2 条不在本回复中作答。

---

## 第 1 条 — 仿真预算对等

所有方法现在按同一套开发集蒙特卡洛（MC）调用**上限**评估。

原协议把 CMA-ES 的*代数*与随机搜索 / Direct PPO 的*候选评估次数*用同一个整数对照，导致 CMA-ES 在每一“轮”实际获得 λ = 10 倍的 100 次弹道批次。统一上限为

    max_candidates = 5 × 15 = 75
    max_mc_calls   = 75 × 3 个开发集种子 = 225

与 `workflow.max_iterations × matlab_rl_optimizer.max_episodes × |S_dev|` 一致。CMA-ES 在候选次数用尽时停止；截断的最后一代仍会评估，但不更新协方差。代码：`experiments/equal_mc_budget.py`。正文第 4.1 节给出协议。

这是对等的*预算上限*，并不要求每次运行都必须用满 225 次，也不表示表上的 MC 整数是同一套计数。搜索每次 MATLAB 为 **10 条弹道**，100 条只用于测试。Random / CMA-ES / Direct PPO 每个候选计 3 次真 MATLAB（满额 75×3=225）。GSMultiAgent 表上的 1–22 是外层反推，且通常只真跑 1 个 seed、另外 2 个 bootstrap。

记忆复用（同一主种子上前序场景的 REUSE_HISTORY）以及约束满足后的提前停止会让记录次数更少。任何方法都不允许超过 225。

Table 5 中 GSMultiAgent、三套优化器与单智能体的「首次成功」= **最终开发集可行**，不是「第一轮外层就成功」。单智能体已按搜索 10 条（test 仍 100）对齐，`first_success` / 开发集 / 测试集均为 **12/12**。

事后从基线 JSON `history` 得到的第一次可行 MC（不是 225 上限）：Random 均值 **73.9**（8/12 找到），CMA-ES **79.2**（12/12），Direct PPO **54.0**（11/12）。文件：`results/experiment_plan/optimizer_equal_mc_75/first_success_mc_calls.csv`。

在该上限下，12 个“场景 × 主种子”格子的开发集/测试集可行率为：

| 方法 | 首次成功 | Dev 可行 | Test 可行 | 记录的开发集 MC |
|------|----------|----------|-----------|-----------------|
| GSMultiAgent | 9/12 | 9/12 | 5/12 | 1–22（外层反推） |
| Random Search | 8/12 | 8/12 | 8/12 | 225（75×3 真 MATLAB） |
| CMA-ES | 12/12 | 12/12 | 12/12 | 225（75×3 真 MATLAB） |
| Direct PPO | 11/12 | 11/12 | 11/12 | 225（75×3 真 MATLAB） |
| Single-Agent | 12/12 | 12/12 | 12/12 | 135–225（搜索 10 条；3 次真 MATLAB） |

旧的不等预算对比（Random / Direct PPO 为 0/12）已从主表和讨论中删除。原始运行见 `results/experiment_plan/optimizer_equal_mc_75/` 与 `optimizer_equal_mc_75_sah_n10/`；汇总表见 `results/experiment_plan/table5_equal_mc.csv`，Table 6 见 `table6_equal_mc.csv`。

---

## 第 3 条 — 摘要（75% vs 25%；“唯一全局帕累托最优”）

摘要不再报告已废止的 75% 对 25% 首次尝试对比，也不再给出缺乏依据的“唯一全局帕累托最优”表述。现按第 4 条，报告修订后的开发集与测试集可行率及其置信区间。讨论部分同样写明：我们不声称唯一全局帕累托最优。

---

## 第 4 条 — 9/12 与 5/12 的置信区间

对 12 个“场景 × 主种子”情形，采用 Clopper–Pearson 精确 95% 区间：

- 开发集可行率 9/12 = 75.0%（95% CI 42.8%–94.5%）
- 测试集可行率 5/12 = 41.7%（95% CI 15.2%–72.3%）

上述区间已写入摘要和第 4.2 节。12 格来自 GSMultiAgent E0（完整记忆）；未收敛格子为 T1/种子 42、T3/种子 44、T4/种子 44。

---

## 第 5 条 — 公开 GitHub、模型版本与实验日期

我们不提供公开的 GitHub 仓库。源代码、蒙特卡洛初值生成器、协议文件、仿真交战 `.mat` 以及等预算运行档案，随修订稿作为**补充材料（ZIP）**提交。清单见下一节。

归档中的交战条件由已发表的麻雀级（AIM-7F）参数与固定随机种子**仿真生成**（`experiments/mc_data_generator.py`），不是实装飞行遥测。事后重抽样 CSV（`*_incl_simulated.csv`）不进入补充材料。GSMultiAgent 主表 AND 使用了运行期 bootstrap 补种（1 个真 MATLAB seed、2 个填充）；独立优化器与单智能体为 3 次真 MATLAB。表上 MC 次数因此不是同一套单位（第 4.1 节）。

请求体中的精确 API 模型标识：

| 角色 | API model ID |
|------|----------------|
| 主实验大模型 | `deepseek-v4-pro` |
| 模型替换 | `qwen3.7-max`、`kimi-k3` |
| 嵌入 | `text-embedding-v3` |

软件：GSMultiAgent 消融为 MATLAB R2025a（i7-13700K 工作站）；等预算基线与单智能体为 MATLAB R2020b（Ubuntu 24.04）。

实验日期（UTC+8）：

- GSMultiAgent 消融、记忆/优化/修正管线、模型替换：**2026 年 8 月 2–12 日**
- 等预算基线（Random Search、CMA-ES、Direct PPO）：**2026 年 8 月 20–22 日**
- 单智能体 Hermes n10 对齐（搜索 10 / 测试 100）：**2026 年 8 月 22–23 日**

上述信息同时写在 ZIP 内 `REPRODUCIBILITY.md`。

---

## 随修订稿提交的附件清单

| 编号 | 文件 | 用途 |
|------|------|------|
| A | 修订稿（`papers.docx`） | 正文，已按第 1、3、4、5 条修改 |
| B | 本点对点回复 | 本函 |
| C | `GSMultiAgent_supplement.zip` | 补充代码、数据与等预算结果（不是公开 GitHub 仓库） |

### 附件 C（`GSMultiAgent_supplement.zip`）内容

压缩约 7.9 MB，解压后 344 个文件。密钥已替换为 `YOUR_API_KEY`。GSMultiAgent 运行使用了 `mc_and_bootstrap_fill`（1 个真 MATLAB seed、另外 2 个填充）；独立优化器与单智能体每个候选使用 3 个真 MATLAB seed。

| ZIP 内路径 | 内容 |
|------------|------|
| `cli_agent.py`、`run_rl_optimize.py`、`config.yaml`、`requirements.txt`、`prompt.txt` | 可运行入口与脱敏配置 |
| `REPRODUCIBILITY.md` | 模型 ID、MATLAB 版本、日期、225 次 MC 上限 |
| `SUPPLEMENT.md`、`提交说明.md`、`打包文件清单.md` | 中英文说明：何可交、何不可交 |
| `multi_agent/` | 框架源码（智能体、PPO、MATLAB 工具、记忆） |
| `experiments/` | 协议、`equal_mc_budget.py`、`mc_data_generator.py`、基线、单智能体 |
| `tests/` | 单元测试（含等预算协议） |
| `knowledge_base/matlab/guidance/monte_carlo_single.m` | 主 MATLAB 蒙特卡洛模板 |
| `data/mc/` | 72 个仿真交战 `.mat`（T1–T4 × 主种子 42/43/44 × 6 个 split）及 `seed_map.json` |
| `results/experiment_plan/optimizer_equal_mc_75/` | 36 份等预算 JSON + `runs.csv`（Random / CMA-ES / Direct PPO） |
| `results/experiment_plan/optimizer_equal_mc_75_sah_n10/` | 12 份等预算单智能体 JSON + `runs.csv`（搜索 10 / 测试 100） |
| `results/experiment_plan/gsm_all_seeds.csv` | GSM 全部消融行一张表（含 E0 的 9/12 与 5/12） |
| `results/experiment_plan/table5_equal_mc.csv` | 主表汇总（8/12、12/12、11/12 等） |
| `results/experiment_plan/ablation_*.csv` | 记忆 / 优化 / 修正 / 顺序消融（仅真实运行） |

**明确不打进 ZIP：** GitHub 地址；API 密钥；`scripts/`（本机运维）；完整过程档案与日志；向量库缓存（`chroma_db/`）；bootstrap 补种；事后 `*_incl_simulated.csv`；单智能体超预算备份。

审稿人可用同一主种子重生 `data/mc/*.mat`：

```
python experiments/mc_data_generator.py --output-root data/mc --n-runs 100 --no-broadside
```

等预算基线复现命令见 `experiments/README.md`。
