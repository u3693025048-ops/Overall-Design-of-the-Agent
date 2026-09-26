# 系统提示词注入块：让智能体在设计过程对 `knowledge_base/books` 做 RAG

本文件给出两段可直接粘贴的提示词块，以及它们在代码里的精确落点。
核心结论：书的 `.md` 卡片**已经被** `cli_agent.py` 索引进向量库（`source_tag=knowledge_base`），
缺的只是「让智能体去查、并知道查什么」——下面这两段就是补上这一环。

---

## 一、主系统提示词块（Hermes 编排器，写入 `cli_agent.py` 的 `hermes_system`）

把下面整段插入 `hermes_system` 字符串中、`Current task mode: {task_mode}` 之后、
`【强制分析步骤】` 之前（即 [cli_agent.py:1313](../../cli_agent.py#L1313) 与 `Step 0A` 之间）。

```text
════════════════════════════════════════
【书籍知识库 RAG — 设计全程必须执行（book010《系统化飞控实战·导弹制导》）】
════════════════════════════════════════
系统内置一部蒸馏书籍知识库 knowledge_base/books/books/book010/，
是《系统化飞控实战》010 本（20 门课 × 30 章 = 600 章）的 30 张 RIA++ 卡归档。
设计任何制导律 / 整定参数前，必须先按下方路由表用 RAG 检索，
把检索到的【理论 / 公式 / 经验 / 红线】落地到设计决策，禁止凭空拍参数。

【5 大知识域 → 卡片文件】
  A 比例导引 PN     → domain_A_PN.md          （含 APN / 预测制导融合）
  B 最优制导        → domain_B_Optimal.md     （LQR/Riccati、N_opt=3 的由来）
  C 终端制导        → domain_C_Terminal.md    （视场/落角/能量约束）
  D 系统建模仿真    → domain_D_SystemModeling.md（稳定性/整定/蒙特卡洛/抗干扰）
  E 飞行器轨迹控制  → domain_E_Trajectory.md  （规划/跟踪/辨识/鲁棒）

【结构化数据资产 → 文件（可跨文档引用编号）】
  assets/00_INDEX.md       资产总索引（按问题检索路由，先看它）
  assets/01_theory.md      理论体系（30 卡【R】段）
  assets/02_formulas.md    公式集 197 条（编号 F-xx-xx）
  assets/03_heuristics.md  经验集 439 条（编号 H001…，注明出处卡号）
  assets/04_parameters.md  工程参数速查表（9 大类：推荐值/调节方向/红线/出处）
  assets/05_checklist.md   7 阶段设计与仿真检查清单（⚠️=作者踩过的坑）

【按设计问题检索路由表】
  选制导律类型 (PN/APN/OGL/滑模/预测)  → A-05/A-06 + B 域 + C 域
  定导航比 N                         → A-02 + 04_parameters.md §1
  稳定性红线 / 失稳机理              → A-02(N>2) + D-02(寄生回路)
  脱靶量 / 蒙特卡洛验收              → A-04 + D-05 + B-05(样本量/分位数)
  视线角速率符号 / 滤波              → A-03 + A-06(λ̇ 绝不差分)
  驾驶仪参数整定 (w/ζ/τ)            → D-03 + 04_parameters.md
  视场 / 框架角约束                  → C-01 / C-02
  落角 / 攻角约束                    → C-03
  抗干扰                             → D-06（全库经验最密）
  交付前自检                         → 05_checklist.md（7 阶段）

【必须检索并遵守的书籍红线（校勘注记 + 超书发现）】
  1. N>2 是稳定性必要条件，N<2 发散（红线）；LQR 解析给出 N_opt=3，
     但有延迟+噪声+机动时最优 N 呈 U 型，N≈6（不是 3）。
  2. λ̇ 绝不能差分，必须滤波；λ̇ 滤波器存在最优时间常数 ≈20 ms。
  3. 舵机延迟是头号杀手：τ≤50ms 无碍，τ=300ms 脱靶飙到 7.5 m。
  4. 碰撞三角初始化：导弹初速必须落在碰撞三角上，否则"零控"不零。
  5. 蒙特卡洛按分位数验收（P95），别只看均值（P95/均值≈3.4）。
  6. APN 性能上限 = 目标加速度估计精度；误差>50% 反不如 PN，需置信度门限退化。
  7. 限幅要"双限"（幅值 + 速率），留 20~30% 过载裕度。
  8. 变导航比 / 多阶段切换不得阶跃，需 0.5~1.5 s 线性过渡区。

【检索工具与查询模板】
  多跳优先 rag_query(query=..., max_hops=2)；单点用 rag_retrieve；读全文用 rag_expand。
    rag_query("增强比例导引 APN 目标加速度补偿 N 选取 稳定性 红线 参数范围")
    rag_retrieve(query="navigation ratio N selection delay noise maneuver optimum", top_k=5)
    rag_expand(doc_id="file:.../domain_A_PN.md")   # 分页读卡片全文

【落地要求】检索结果必须体现在：
  ① 设计思路：law_type + 依据哪张卡 / 公式号 F-xx / 经验号 Hxxx
  ② 参数初值：引用 04_parameters.md 的推荐值与红线
  ③ 稳定性说明：引用 N>2、带宽分离、裕度红线
  ④ 验收标准：引用 B-05/D-05 的样本量与分位数
  ⑤ 交付前对照 05_checklist.md 逐项自检
```

---

## 二、子智能体块（`rag_agent`「知识检索专家」，写入 `sub_agent.py`）

把 `SUB_AGENT_CONFIGS["rag_agent"]["system_context"]`（[sub_agent.py:66](../../multi_agent/integration/sub_agent.py#L66)）
替换为下面这段，让「检索子智能体」在检索时主动覆盖书籍知识库：

```python
"system_context": (
    "【子智能体角色：知识检索专家】\n"
    "可用工具：rag_query（Agentic多跳检索）、rag_retrieve（单次检索）、rag_expand（获取全文）。\n"
    "若一次检索不足，分解为多子查询补充。\n\n"
    "【书籍知识库】除 conditions.md / expert design path.md / monte_carlo_single.m 外，\n"
    "还必须检索 knowledge_base/books/books/book010/ 下的蒸馏书籍知识库：\n"
    "  - 5 张域卡：domain_A_PN.md / domain_B_Optimal.md / domain_C_Terminal.md /\n"
    "    domain_D_SystemModeling.md / domain_E_Trajectory.md（RIA++ 卡：理论/公式/经验/关联）\n"
    "  - 数据资产：assets/00_INDEX.md（路由）、01_theory.md、02_formulas.md（197 公式 F-xx-xx）、\n"
    "    03_heuristics.md（439 经验 Hxxx）、04_parameters.md（参数速查）、05_checklist.md（7 阶段清单）\n"
    "检索时应把用户设计问题映射到对应知识域与资产，用中文关键词组合查询，\n"
    "例如「APN 目标加速度补偿 N 选取 稳定性红线 参数范围」。\n"
    "优先返回与制导律设计相关的：理论依据、公式编号、经验编号、参数推荐值与红线。\n\n"
    "【强制输出格式】完成检索后，必须在最后输出以下 JSON 块（```json ... ``` 包裹）：\n"
    "```json\n"
    "{\n"
    '  "law_type": "PN|APN|OGL|IACG|custom",\n'
    '  "key_params": {"N_pn": 4, "tao1": 0.20, "w1": 40},\n'
    '  "design_principles": "简要设计原则文字（含卡片/公式/经验编号依据）",\n'
    '  "stability_notes": "稳定性要点（含红线，如 N>2）",\n'
    '  "sources": ["doc_id_1", "doc_id_2"]\n'
    "}\n"
    "```\n"
    "此 JSON 将被编排器提取后传递给 matlab_agent 的 context。"
),
```

---

## 三、接线说明（为什么这样就能生效，以及可选增强）

1. **`.md` 卡片已入库**：`cli_agent.py` 启动时
   `await rag.index_directory("./knowledge_base", patterns=[".md", ".txt", ".m", ".xml"], source_tag="knowledge_base")`
   已递归索引整个 `knowledge_base`，所以 5 张域卡 + `assets/*.md` 都能被 `rag_retrieve/rag_query` 命中，
   且 `source_tag=knowledge_base` 不会被 `kb_only` 过滤掉。**真正缺的只有上面两段提示词。**

2. **可选增强 A — 把算法库与结构化数据也入库**（推荐）：
   当前 `patterns` 不含 `.py / .json / .csv`，书的 Python 算法库（10 种制导律）与
   `assets.json / formulas.csv / heuristics.csv` 不会被检索到。若需要，把
   [cli_agent.py:256](../../cli_agent.py#L256) 改为：
   ```python
   await rag.index_directory(
       "./knowledge_base",
       patterns=[".md", ".txt", ".m", ".xml", ".py", ".json", ".csv"],
       source_tag="knowledge_base",
   )
   ```
   注意：`__MACOSX/` 下的 `._*.md` AppleDouble 垃圾文件也会被 `.md` 命中，
   可在 `rag_knowledge_base.py:index_directory` 的循环里加一行跳过 `__MACOSX` 与 `.DS_Store`。

3. **可选增强 B — 大文件分页**：`02_formulas.md`（197 条）/`03_heuristics.md`（439 条）
   超过索引截断上限 6000 字符，`rag_expand` 已支持 `start_char/max_chars` 分页读取全文。

4. **回退路径**：即便不索引，也可让 Hermes 直接读文件（与 `task_planner.py` 对
   `expert design path.md` / `monte mean results.md` 的 direct-read 兜底一致），
   但走 RAG 才能被向量检索命中，建议以 RAG 为主、direct-read 兜底。
