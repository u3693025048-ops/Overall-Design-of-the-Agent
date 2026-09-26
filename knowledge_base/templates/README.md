# 模板库（Template Library / Canvas-IR 模板种子）

本目录是四库中的**模板库**，存放画布生成 Agent 可直接复用/填充的 Canvas-IR 骨架。
画布生成时经 RAG 检索命中此处模板 → 产出 `next_action=复用模板`，再注入参数生成可执行画布。

> 源真相 = Canvas-IR（JSON）；SSIM 工程由导入器从 IR 导出。GUI 未完成时，
> 只要 IR 齐备、导入器可通，即满足「使用了 SSIM」（见架构文档 §6.1.9）。

## 目录约定
```
knowledge_base/templates/
├── README.md
└── verify_chain_template.json   # 验证链骨架（校核：是否满足 R0）
```

## 字段（与架构文档 §3.3 画布最小字段对齐）
- `nodes`    : 模块 / 功能函数节点
- `edges`    : 数据流 / 控制时序
- `bindings` : 参数绑定（含单位、设计层级）
- `exec`     : 拓扑序、并行组、超时、重试
- `artifacts`: Modelica / MATLAB / 气动表 等引用（不把大文件嵌进图里）

## 注意
- 蒸馏书（knowledge_base/books/book010 等）**不**计入本模板库，其卡片须打成带
  `level` 标签的经验卡后，才允许写入「知识缩短周期」（架构文档 §2.7 口径③）。
