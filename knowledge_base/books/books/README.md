# 《系统化飞控实战》系列蒸馏归档

按本→文件夹独立完整。每本包含 RIA++ 卡片、SymPy 符号推导、弹道仿真脚本与图、源书解压与目录。

## 进度

| 编号 | 标题 | 状态 | 归档 | 完成时间 |
|------|------|------|------|----------|
| **010** | 系统化飞控实战（导弹制导） | ✅ **30/30 卡** | [`book010/`](./book010/) | 2026-09-01 |
| 011 | （待启动） | ⏳ | — | — |

## 蒸馏流水线（每本复用）

1. **解压**：`book010/ → source/book010/`，zip 内文件名 GBK → cp437→gbk
2. **目录抽取**：600 章 → `source/book010_toc.json`（章号/标题/摘要/标签）
3. **RIA++ 蒸馏**：5 域 × 6 张卡（4.8 万中文字符 / 11.7 万总字符）→ `domain_*.md`
4. **SymPy 推导**：LQR/Riccati/LOS 动力学 → `verify_lqr_pn.py`（exit 0）
5. **弹道仿真**：末段突然机动最坏工况，8 面板联合验证 → `verify_engagement.py` + `.png`
6. **索引体系**：可搜索的 600 章知识地图 → `index.html`（692 KB）
7. **资产抽取**：卡片【R 理论】【I 公式】【A++ 经验】三段程序化拆出 → `assets/`
8. **算法库**：`algorithms/` 7 模块纯标准库（10 种制导律 + 交战仿真 + 蒙特卡洛），
   63 项自检，`run_all.py` 一键验证 + `--demo` 参数标定
9. **资料库打包**：全部产出内嵌进单文件 HTML → `guidance_library.html`
   （左目录点击即看 / 全文搜索 / 可打开原文件）
10. **生成后自检**：`check_library.py` 静态 + 运行时质检

> 步骤 7~10 是 book010 新增的**资产化层**，与蒸馏（1~6）解耦：卡片改了只需重跑
> 抽取 + 打包，不存在"卡片与资产脱节"。

### 一键流水线（新书首选）

步骤 1、2 原本是当年临时命令跑的，现已固化；`_tools/pipeline.py` 把
**1、7、9、10** 四步串成一条命令：

```bash
cd books/_tools

# 从 zip 开始（解压 → 目录抽取 → 资产抽取 → 资料库打包 → 质检）
python3 pipeline.py --zip "/path/to/011-xxx.zip" --book-dir ../book011 --id book011

# 或书已解压：跳过 Stage 0
python3 pipeline.py --book-dir ../book011 --id book011

# 只跑某几步 / 附带算法库自检
python3 pipeline.py --book-dir ../book011 --id book011 --only 1,2,3
python3 pipeline.py --book-dir ../book011 --id book011 --verify
```

分阶段计时、任一阶段失败立即停并以非零码退出。
`--list` 打印阶段清单。

> **注意**：阶段 3~6（RIA++ 蒸馏 / SymPy 推导 / 弹道仿真 / 知识地图）是
> **内容生产**，不是机械转换，没有脚本化——每本仍需人工（LLM）产出卡片与验证脚本。
> 卡片写完后，上面的流水线接管全部机械环节。

### 分步执行（想细看每步输出时）

```bash
cd books/_tools
python3 unpack_source.py   --zip "/path/to/011-xxx.zip" --book-dir ../book011 --id book011
python3 extract_assets.py  --book-dir ../book011     # 零配置，自动发现 domain_*.md
python3 build_library.py   --book-dir ../book011     # 自动分组打包
python3 check_library.py   ../book011/book011_library.html
```

想控制目录顺序与中文描述时，在书目目录放 `library.json`
（模板见 [`book010/library.json`](./book010/library.json)）。
详见 [`_tools/README.md`](./_tools/README.md)。

## 每本目录布局

```
bookXXX/
├── README.md                    # 主索引（蒸馏规格 / 卡片清单 / 仿真结论）
├── guidance_library.html        # ★ 单文件资料库（28 个文件内嵌，推荐入口）
├── domain_*.md                  # 5 张 RIA++ 卡片（各 4 门课）
├── assets/                      # ★ 数据资产层
│   ├── 00_INDEX.md              #   资产总索引（按问题检索路由）
│   ├── 01_theory.md             #   理论体系（30 卡【R】段）
│   ├── 02_formulas.md           #   公式集（197 条，编号 F-xx-xx）
│   ├── 03_heuristics.md         #   经验集（439 条，编号 Hxxx）
│   ├── 04_parameters.md         #   工程参数速查表（9 大类）
│   ├── 05_checklist.md          #   检查清单（7 阶段）
│   └── data/                    #   assets.json + formulas.csv + heuristics.csv
├── algorithms/                  # ★ Python 算法库（纯标准库，63 项自检）
│   ├── README.md                #   模块清单 + 制导律注册表 + 建模约定
│   ├── gnc_core.py              #   基础工具层（LOS/ZEM/CPA/碰撞三角）
│   ├── guidance_laws.py         #   10 种制导律
│   ├── seeker.py  dynamics.py  engagement.py
│   ├── metrics.py  monte_carlo.py
│   └── run_all.py               #   一键自检 + --demo 演示
├── index.html                   # 交互式知识地图
├── library.json                 # 可选：资料库分组与描述配置（无则自动发现）
├── _extract_assets.py           # 卡片 → 结构化资产（原型，新书用 _tools/）
├── _build_library.py            # 资料库生成器（原型，新书用 _tools/）
├── verify_*.py + .png           # 弹道仿真脚本与图
├── verify_lqr_pn.py             # SymPy 符号推导
├── _gen_index.py + _index_data.json + _template.html
├── source/                      # 源书原始数据
│   ├── bookXXX/                 # 解压后的 20 门课 HTML
│   ├── bookXXX_outline.json
│   └── bookXXX_toc.json
└── prototype/                   # 早期原型（demo 脚本与图）
```

## 方法论沉淀（来自 book010）

- **场景鉴别力**：用末段突然机动最坏工况（t_go ≈ 1 s，目标 9 g，导弹 25 g）确保参数有区分度
- **导引头噪声建模**：必须用「测角 + 一阶滤波微分」（蓝谱）才能复现「N 放大噪声」；平谱速率噪声模型结论相反
- **碰撞三角初始化**：导弹初速必须落在碰撞三角上，否则「零控」不零
- **λ̇ 滤波器最优时间常数 ≈20 ms**，对应典型战术导弹制导滤波器带宽（1~5 Hz）
- **诊断三步**：独立复算 → 参数扫描 → 机理重构（区分代码错 / 模型简化过度 / 场景无鉴别力）

### 资产化层方法论（book010 新增）

- **程序化抽取优于手工整理**：卡片结构统一（【R】【I】【A++】【关联】）时，
  用正则分段抽取 → JSON/CSV/MD 三格式；卡片更新后重跑即可，零维护成本
- **经验条拆分的两类噪声**：段首引导语（"作者踩过的坑："）与 <15 字符碎片，
  需 `merge_lead()` 并入下一条 + `cleanup()` 丢弃，否则出现无编号条目
- **参数资产化**：把散落的"推荐值"句子编成表（推荐值 / 调节方向 / 红线 / 出处卡号），
  比堆砌原文有用得多
- **算法库先立约定再写代码**：符号定义（λ̇ 正负）、初始化（碰撞三角）、
  噪声路径（角度+微分）、离散化（解析解）、验收（分位数）五条写进 README，
  否则模块间会互相打架

### 源书解析（book010 新增）

- **`<title>第N章 X</title>` 是唯一 100% 可靠的锚点**：book010 实测 600/600 命中。
  h2 只有 535/600 是「冒号式摘要」，其余是概览页，需正文首段兜底
- **泛化标题要用 h2 补齐，但别用长度一刀切**：`绪论` / `案例研究` / `附录A`
  信息量不足，可展开成完整 h2；但 `协同制导` / `制导概论` 这类短标题是
  **技术术语**，展开反而丢信息。只认结构性词
- **编号前缀必须吃掉尾随分隔符**：`第3章：X` 剥掉 `第3章` 后残留的冒号会污染
  标题/摘要切分（表现为标题被清空、摘要变成整段）
- **编号单位不止「章」**：`第二十三讲：` 同样常见，正则要覆盖 章/讲/节/篇
- **分隔符不止冒号**：`——` 双破折号同样在用
- **侧边栏会污染正文**：页面有「课程目录」导航区，直接 `text.find("课程目录")`
  定位正文会把侧栏章节列表当成摘要——要先剥离导航容器
- **既有 toc 不适合当金标准**：它是 LLM 分批润色的，风格不统一（全角/半角、
  编号补零与否、附录章号记成 `'?'`）。回归时应做**结构断言**（章节数、编号连续、
  无空值），标题一致率只作信息性报告（book010 为 91.3%，多数差异是我方更完整）

### HTML 资料库打包（book010 新增）

- **file:// 下必须内嵌**：浏览器拦截 fetch/XHR 读本地 JSON，外挂数据文件会全站失效
- **生成器源码收进资料库时的定界符冲突**：源码含 `"""` / `'''` 会提前终止 Python
  三引号字符串 → JS 正则里改用 `[\u0022]{3}` / `[\u0027]{3}`
- **占位符替换必须单次完成**：链式 `str.replace` 会被先插入的内容二次替换，
  改用 `re.sub` 一次性替换
- **`</script>` 转义**：内嵌正文里的 `</` 要写成 `<\/`，否则提前闭合脚本
- **Markdown 渲染器必须先 stash 公式与代码块**：否则 `**`、`*`、`` ` `` 会污染
  LaTeX 内容；`\*`（如 A\* 算法）要单独保护，否则被强调规则误吃导致 `**` 配对整体错位
- **无 Chromium 时如何验证**：用 Node + 极简 DOM stub 执行页面脚本，
  对真实内容跑渲染器（查残留占位符 / undefined / 标签深度错配），能抓到运行时 bug