# `_tools/` — 系列书目通用工具

一套脚本服务**所有书目**（011~020 直接复用，不必复制改路径）。
`book010` 目录里那两个 `_extract_assets.py` / `_build_library.py` 是它的原型，
已冻结归档；新书统一用这里的通用版。

## 工具清单

| 脚本 | 作用 | 依赖 |
|------|------|------|
| `pipeline.py` | **一键流水线**，串起下面四个（新书只跑它） | 纯标准库 |
| `unpack_source.py` | Stage 0：源书 zip 解压（GBK 文件名还原）+ 目录抽取 | 纯标准库 |
| `extract_assets.py` | RIA++ 卡片 → 结构化数据资产（理论/公式/经验） | 纯标准库 |
| `build_library.py` | 书目全部产出 → 单文件 HTML 资料库 | 纯标准库 |
| `check_library.py` | 资料库质检（静态结构 + 调用 Node 跑渲染） | Python + 可选 Node |

## 新书一键流程（推荐）

```bash
cd books/_tools

# 全自动：解压 → 资产 → 资料库 → 质检（--verify 附带算法库自检）
python3 pipeline.py --zip "/path/to/011-xxx.zip" --book-dir ../book011 --id book011

# 或：书已解压好，直接从目录抽取开始
python3 pipeline.py --book-dir ../book011 --id book011
```

分阶段计时、任一阶段失败立即停并以非零码退出。常用开关：

| 开关 | 作用 |
|------|------|
| `--list` | 只打印阶段清单，不执行 |
| `--only 1,2` | 只跑指定阶段；`V` 也可选 |
| `--skip 3` | 跳过指定阶段 |
| `--verify` | 追加算法库自检（耗时长，默认不跑） |
| `--expect-courses 20 --expect-per-course 30` | 传给 Stage 0 的结构自检 |
| `--timeout N` | 单阶段超时秒数（默认 1800） |
| `--html-out NAME` | 指定资料库输出文件名 |
| `--force` | Stage 0 目标已存在时强制覆盖解压 |

> `--timeout` 是子进程的墙钟上限。算法库自检（V）在正常负载下约 37 s，
> 但机器高负载时可能慢上数倍——曾出现过 load 43 时进程被系统杀掉的情况。
> 若 V 阶段被超时打断，先看 `uptime`，别急着改代码。

### 资料库输出文件名

优先级：`--html-out` > `library.json` 的 `"out"` 字段 > `<id>_library.html`。

```json
{ "id": "book010", "out": "guidance_library.html" }
```

留 `out` 这个口子是因为 book010 的既有交付叫 `guidance_library.html`，
已被 README 与索引多处引用，改名会破坏既有链接。新书不配这一项即可，
默认得到 `<id>_library.html`。

## 分步流程（想细看每步输出时）

```bash
cd books/_tools

# 0. 源书 zip → source/ + {id}_outline.json + {id}_toc.json
python3 unpack_source.py --zip "/path/to/011-xxx.zip" --book-dir ../book011 --id book011

# 1. 卡片 → 数据资产（自动发现 domain_*.md）
python3 extract_assets.py --book-dir ../book011

# 2. 全量产出 → 单文件资料库（自动发现分组）
python3 build_library.py --book-dir ../book011

# 3. 生成后自检（强烈建议，能抓到静态看不出的渲染 bug）
python3 check_library.py ../book011/book011_library.html
```

产出：
- `book011/source/` — 解压后的原始 HTML + `{id}_outline.json` + `{id}_toc.json`
- `book011/assets/` — `01_theory.md` / `02_formulas.md` / `03_heuristics.md`
  + `data/{assets.json, formulas.csv, heuristics.csv}`
- `book011/book011_library.html` — 单文件资料库

## 零配置 vs 显式配置

**零配置**（推荐先跑）：两个脚本都会自动发现。

- `extract_assets.py`：扫描 `domain_*.md`，从文件头
  `# 知识域 A：比例导引（PN）— RIA++ 技能卡` 提取域码与域名；
  书名从 `> 蒸馏来源：…` 推断。
- `build_library.py`：按内置规则表分组——

  | 组 | 匹配 |
  |----|------|
  | 入口 | `README.md` |
  | 数据资产（理论 / 公式 / 经验） | `assets/*.md`、`assets/data/*.csv` |
  | Python 算法库 | `algorithms/README.md`、`algorithms/*.py` |
  | RIA++ 卡片原文 | `domain_*.md` |
  | 符号推导与弹道仿真 | `verify_*.py`、`verify_*.png` |
  | 工具脚本 | `_*.py` |

  始终跳过隐藏目录、`source/`、`prototype/`。

**显式配置**：想要**可控顺序 + 每项带中文描述**时，在书目目录放 `library.json`
（参考 `../book010/library.json`）：

```json
{
  "id": "book011",
  "out": "book011_library.html",
  "book": "《系统化飞控实战》011 · XXX",
  "title": "book011 · XXX",
  "html_title": "book011 XXX — 数据资产与算法库",
  "generated_by": "_extract_assets.py",
  "domains": [
    {"code": "A", "file": "domain_A_XXX.md", "name": "XXX"}
  ],
  "groups": [
    {"name": "数据资产", "items": [
      {"path": "assets/00_INDEX.md", "desc": "资产总索引"}
    ]}
  ]
}
```

配了 `groups` 就完全按配置走（不再自动发现）。

## Stage 0：源书解压与目录抽取

`unpack_source.py` 把当年临时执行的命令固化成了脚本：

1. **解压** —— zip 内文件名是 GBK 编码的，zipfile 会按 cp437 解码成乱码，
   需要 `name.encode("cp437").decode("gbk")` 还原。失败时逐条回退原名。
2. **目录抽取** —— 遍历 `source/{id}/《课程名》/NN.html`，每章抽出：
   - `title`：来自 `<title>第N章 X</title>`（book010 实测 600/600 命中）
   - `desc`：来自首个 `<h2>`，剥掉编号前缀后取分隔符后的部分；
     无分隔符则回退到正文首段
   - `tags`：从标题+正文首段里匹配领域词表生成的 2~4 个启发式标签
   - `chars` / `body_head`：字数与正文开头（供 outline 用）

产出 `source/{id}_toc.json`（课程 → 章节列表）与 `source/{id}_outline.json`
（每章的字数/开头摘要，用于快速摸清新书规模）。

**标题补齐**：`<title>` 里的标题有时过于泛化（`绪论` / `案例研究` / `高级话题` /
`附录A`），这时改用完整 h2。判断规则见 `is_terse()`——**只认结构性词**，
`协同制导`、`制导概论` 这类短但有信息量的技术术语不展开（早期版本用
`len<=4` 一刀切，误伤过）。

**自检**：加 `--expect-courses N --expect-per-course M` 做结构性断言
（课程数、每课章节数、编号 1..M 连续、无空标题/空摘要）。
若有既有 `{id}_toc.json`，还会报出标题一致率。

> 注意：既有 toc 的标题是 LLM 分批润色过的（全角括号→半角、重复标题消歧），
> 风格不统一，不适合当金标准。book010 的一致率是 91.3%，差异里的多数是
> 我方更完整（如 `典型案例分析：空空与地空导弹HIL仿真` vs 原文 `典型案例分析`）。
> 因此 `--check` 只做**信息性报告**，结构断言才是硬门槛。

## 关于 `00_INDEX.md`

`extract_assets.py` **不会**生成它——它是手写的导航中枢
（资产全景 / 编号体系 / 卡片速查 / **按问题检索路由** / 卡片↔算法模块映射）。
新书写完卡片后，照 `book010/assets/00_INDEX.md` 的结构仿写一份，
它是整个资产层里最常被打开的文件。

同理，`04_parameters.md`（参数速查表）与 `05_checklist.md`（检查清单）
也是从经验段**人工提炼**的增值资产，不自动生成。

## 质检在查什么

`check_library.py` 的运行时部分（`check_render.js`）用极简 DOM stub 在 Node 里
执行页面脚本，对真实内容跑渲染器。它专门抓**静态校验看不见**的问题：

- stash 占位符没回填（公式/代码块丢失）
- `undefined` / `[object Object]` 泄漏
- **strong 标签栈深度错位** —— LaTeX 的 `\*`（如 A\* 算法）被强调规则误吃后，
  后续 `**` 配对会整体错位。book010 曾一次性错位 428 处，静态校验完全无感
- 标签开闭配平、`\*` 占位符是否还原

没有 Node 时会跳过运行时部分（`--static` 可强制只跑静态）。

## 已知坑（改代码前务必读 `build_library.py` 文件头）

1. JS 正则里不能出现连续三个双引号/单引号——生成器源码会被收进资料库，
   裸三引号会提前终止 Python 三引号字符串。用 `[\u0022]{3}` / `[\u0027]{3}`。
2. 占位符必须单次替换（`re.sub`），链式 `str.replace` 会被先插入的内容二次替换。
3. 正文里的 `</` 要转成 `<\/`，否则提前闭合 `<script>`。
4. `file://` 下浏览器拦截 fetch，所有内容必须内嵌。
5. 自动发现遍历目录时，顶层 `rel_root` 就是 `"."`——
   不能简单用 `startswith(".")` 判断是否隐藏目录，否则顶层文件全被漏掉。

### Stage 0 专属（改 `unpack_source.py` 前必读）

6. **h2 编号前缀必须吃掉尾随分隔符**。h2 形如 `第3章：实时仿真平台选型——…`，
   若只剥掉 `第3章` 而留下冒号，后面的 `partition("：")` 会把整段当摘要、
   标题反而被清空。前缀正则末尾的 `[：:、.]?` 不能省。
7. **编号单位不止「章」**。book010 有 `第二十三讲：`——「讲」没被覆盖时，
   冒号同样会污染标题/摘要切分。现已支持 章/讲/节/篇。
8. **分隔符不止冒号**。`——`（双破折号）同样常见，且它出现在正文首段而非 h2。
9. **正文开头要剥掉 h2 本身**，否则无分隔符章节的摘要会重复标题。
   比对时必须用**去掉编号前缀前**的原始 h2——正文里带 `3. ` 前缀。
10. **侧边栏目录污染**：页面有「课程目录」侧栏，早期版本用
    `text.find("课程目录")` 定位正文，方向反了，摘要取成了侧栏的章节列表。
    正解是先切掉 `<main>` 之外的导航区，再取正文首段。
11. **文件名 GBK**：zip 内中文名走 cp437→gbk；不要直接用 `z.namelist()`。
12. **章号有 `?`**：既有 toc 里附录章被记成 `'?'`（无 `第N章` 标题），
    比对时按数值匹配并容错非数字。
