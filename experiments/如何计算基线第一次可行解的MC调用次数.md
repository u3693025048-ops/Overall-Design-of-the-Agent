# 如何计算基线「第一次可行解」的 MC 调用次数

给完全不熟悉本仓库的人看。看完应能自己算出一个数字，并知道哪个数字绝对不能用。

本文只讲 **Random Search / CMA-ES / Direct PPO** 三套优化器基线。  
数据在：`results/experiment_plan/optimizer_equal_mc_75/`  
**不用重跑实验**，磁盘上已有逐步记录。

---

## 1. 先记住一句话

> **第一次可行解的 MC 次数 = 按评测顺序，累加到「第一个三种子全过」的那个候选时，一共跑了多少次 MATLAB。**

它 **不是** CSV 里的 `first_success_iter`。  
它 **也不是** CSV 里的 `mc_calls_dev`（那几乎全是 225，搜到上限才停）。

不会算时，直接跑第 7 节的脚本。

---

## 2. 三个容易混的数字，必须分开

打开 `runs.csv` 会看到很多列。和「第一次找到可行解花了多少仿真」有关的，其实只有下面三个，而且含义全不一样：

| 你在表里看到的 | 它实际是什么 | 能不能当「第一次可行的 MC 次数」 |
|----------------|--------------|----------------------------------|
| `mc_calls_dev` | 这一格 **整场搜索** 用掉的开发集 MATLAB 次数，上限 225 | **不能。** 找到可行解后仍继续搜到上限 |
| `first_success` | 这一格搜索过程中 **有没有** 出现过可行候选（True/False） | 只能回答「有没有找到」，不回答「第几次」 |
| `first_success_iter` | 一个 **序号**，三种方法含义还不一样 | **多数情况不能直接当 MC 次数**，尤其是 CMA-ES |

再加一个要自己算出来的：

| 要的数字 | 含义 |
|----------|------|
| `mc_calls_first_success` | 从第 1 个候选评到 **第一个可行候选（含它自己）**，累计 MATLAB 次数 |

---

## 3. 什么叫「1 次 MC 调用」

一次调用 = **对 1 个种子、跑 1 次 MATLAB**（搜索阶段每次 10 条弹道）。

每个候选都要过 **3 个开发集种子**（例如 master seed=42 时是 4201、4202、4203），三个都满足约束才叫可行（AND）。

所以：

```
评完 1 个候选  =  3 次 MC 调用
评完 20 个候选 =  60 次 MC 调用
上限 75 个候选 =  225 次 MC 调用
```

「可行」= 这个候选的 `dev.all_pass == true`  
（三个种子的命中率 / SEP / 过载 / 相位裕度 / 带宽全部过关。）

---

## 4. 为什么 CSV 里的 `first_success_iter` 不能直接用

三种方法写进这一列的东西 **单位不同**：

| 方法 | `first_success_iter` 记的是 | 错把它当 MC 次数会怎样 |
|------|-----------------------------|------------------------|
| Random Search | 第几个 **候选** | 错 3 倍。正确：`候选号 × 3` |
| Direct PPO | 第几个 **episode**（也是 1 候选 = 1 episode） | 错 3 倍。正确：`episode 号 × 3` |
| CMA-ES | 第几 **代**（每代最多 10 个后代） | **完全错。** 第 2 代 ≠ 2 次调用 |

CSV 里 CMA 那一列还有 `iterations_unit = cma_generations`，就是在提醒：这是代数，不是候选序号。

### 反例（请务必看）

`cma_es` / T2 / seed 42：

- CSV：`first_success_iter = 2`
- 若有人说「2 次 MC 就找到了」→ **错**
- 真相：第 **2 代** 的第 10 个后代（全书第 20 个候选）才第一次全过  
  → `20 × 3 = 60` 次 MC 调用

Random 的反例：T1 seed 42 的 `first_success_iter = 36`  
→ 正确 MC 次数是 `36 × 3 = 108`，不是 36。

---

## 5. 唯一可靠的数据：每个格子的 JSON

路径：

```
results/experiment_plan/optimizer_equal_mc_75/runs/
    T1_random_search_seed42.json
    T1_cma_es_seed42.json
    T1_direct_ppo_seed42.json
    … 一共 36 个基线文件（4 场景 × 3 种子 × 3 方法）
```

每个 JSON 里有一个数组 `history`。  
**数组顺序 = 真实评测顺序。** 这是唯一可以累加的顺序。

每一条大致长这样（只看这几个字段就够了）：

```json
{
  "iteration": 2,
  "candidate_index": 10,
  "dev": {
    "all_pass": true,
    "mc_calls": 3
  }
}
```

- `dev.all_pass`：这个候选是否可行  
- `dev.mc_calls`：这个候选花了几次 MATLAB，基线里几乎都是 **3**  
- `iteration`：Random/PPO = 候选号；CMA = **代数**（所以不要用它乘 3）  
- `candidate_index`：这一代里的第几个后代（Random/PPO 总是 1）

### 手算步骤（对任意一个 JSON）

1. 打开该文件，找到 `"history": [`  
2. 准备一个计数器，从 0 开始  
3. 从上往下看每一条：  
   - 计数器 `+=` 这一条的 `dev.mc_calls`（没有就当 3）  
   - 如果 `dev.all_pass` 是 `true`：**停**。此时的计数器就是答案  
4. 如果整份 `history` 都没有 `all_pass=true`：这一格没找到，答案为空（和 CSV 里 `first_success=False` 一致）

CMA T2 seed 42 手算示意：

```
第 1 条  all_pass=false  → 累计 3
第 2 条  all_pass=false  → 累计 6
…
第 19 条 all_pass=false  → 累计 57
第 20 条 all_pass=true   → 累计 60   ← 就是它
```

---

## 6. 捷径（仅 Random / PPO 可用）

这两套每步只评 1 个候选，所以：

```
mc_calls_first_success = first_success_iter × 3
```

空着的 `first_success_iter` = 没找到 = 不要填数字。

**CMA-ES 禁止用这个公式。** 必须走第 5 节或第 7 节。

---

## 7. 一键脚本（推荐）

仓库根目录执行：

```powershell
python scripts/report_first_success_mc_calls.py
```

默认读取 `results/experiment_plan/optimizer_equal_mc_75/runs/`，在终端打印 36 格，并写出：

```
results/experiment_plan/optimizer_equal_mc_75/first_success_mc_calls.csv
```

新 CSV 多一列 `mc_calls_first_success`。没找到的格子该列为空。

---

## 8. 已算好的 36 格（与脚本一致）

`mc_first` 就是第一次可行解的 MC 调用次数。`None` = 搜满 225 仍未找到。

| 方法 | 场景 | seed 42 | seed 43 | seed 44 |
|------|------|--------:|--------:|--------:|
| Random Search | T1 | 108 | — | 27 |
| Random Search | T2 | 108 | — | 27 |
| Random Search | T3 | 108 | — | 27 |
| Random Search | T4 | 108 | — | 78 |
| CMA-ES | T1 | 60 | 120 | 81 |
| CMA-ES | T2 | 60 | 120 | 81 |
| CMA-ES | T3 | 60 | 120 | 81 |
| CMA-ES | T4 | 60 | 63 | 45 |
| Direct PPO | T1 | 51 | 48 | 66 |
| Direct PPO | T2 | 51 | 48 | 66 |
| Direct PPO | T3 | 51 | 48 | 66 |
| Direct PPO | T4 | 51 | 48 | — |

找到的格子再平均（没找到的不进平均）：

| 方法 | 找到 | 各格 `mc_first` | 均值 |
|------|------|-----------------|-----:|
| Random Search | 8/12 | 108, 27, 108, 27, 108, 27, 108, 78 | 73.9 |
| CMA-ES | 12/12 | 60, 120, 81, 60, 120, 81, 60, 120, 81, 60, 63, 45 | 79.2 |
| Direct PPO | 11/12 | 51, 48, 66, 51, 48, 66, 51, 48, 66, 51, 48 | 54.0 |

写论文时请同时写清：**这是「第一次出现可行候选」时已消耗的开发集 MC 次数，不是 225 上限，也不是测试集。**

---

## 9. 不要和 GSM / SAH 的 `first_success` 混用

| | Random / CMA / PPO | GSM | SAH n10（搜索 10） |
|--|--------------------|-----|---------------------|
| Table 的 `first_success` 布尔值 | 搜索中是否出现过可行候选 | **最终开发集是否可行**（9/12） | **最终开发集是否可行**（12/12） |
| 第一次出现的时刻 | JSON `history` 累加 MC | 外层反推，常 bootstrap，与 75×3 不可对读 | `first_success_iter` = 首次可行 episode 所在外层；记录 MC 135–225 |

基线「第一次可行的 MC 次数」**不要**拿去和 GSM 的 `first_success` 百分比或 1–22 次直接比。旧 100 条 SAH 的 0/12 **不要**再和 GSM 的 9/12 比；对齐后的数字在 `optimizer_equal_mc_75_sah_n10/` 与 `table6_equal_mc.csv`。

---

## 10. 常见错误清单

1. 把 `mc_calls_dev=225` 当成第一次成功 → 那是搜完上限。  
2. 把 CMA 的 `first_success_iter=2` 当成 2 次调用 → 那是第 2 代，常见答案是 60。  
3. 把 Random/PPO 的 `first_success_iter` 直接当 MC 次数 → 少乘了 3。  
4. 用 JSON 顶层的 `best_dev`，而不是 `history` 里第一条 `all_pass` → `best_dev` 是搜完后的最优，不一定是第一次可行。  
5. 想把 100 条弹道的旧结果「折成」10 条 → 不能折，搜索宽度不同。本文这批基线搜索已经是 10 条，**不用重跑**。  
6. 把测试集 3 次调用算进去 → 第一次可行只计 **开发集** 搜索。

---

## 11. 和代码的对应关系（给要改程序的人）

`experiments/optimizer_baselines_43.py`：

- Random / PPO：每评完一个候选，若 `dev.all_pass` 且尚未记录，则 `first_success_iter = iteration`（候选号）。  
- CMA：同样条件，但赋的是 `generation`（代数）。  
- 每次 `dev.mc_calls = 3`（三个开发集种子各一次 MATLAB）。  
- `history` 按评测顺序追加，CMA 同代内按 `candidate_index` 排序后写入。

所以事后从 `history` 累加，和当时真实发生的次数一致。
