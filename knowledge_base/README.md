# 知识库（RAG 源）

Layer 2 检索智能体从本目录读取工程启发式与仿真模板。当前主仿真模板是 `matlab/guidance/monte_carlo_single.m`。

```
knowledge_base/
├── matlab/guidance/
│   └── monte_carlo_single.m     # T/G/AP/R 蒙特卡洛仿真模板
├── conditions.md                # 工况体系
├── expert design path.md        # 专家设计路径
└── monte mean results.md        # 基准仿真结果备忘
```

替换模板时需同步 `matlab_rl_optimizer.py` 的参数映射、`conditions.md` 子工况编号，以及 `config.yaml` 中的指标阈值。详见根目录 [README.md](../README.md)「替换仿真模板」。
