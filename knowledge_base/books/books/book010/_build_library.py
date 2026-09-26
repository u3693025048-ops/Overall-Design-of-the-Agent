#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_build_library.py — 生成单文件 HTML 资料库

把 book010 的全部产出（数据资产 / Python 算法库 / RIA++ 卡片 / 脚本）
打包进一个 HTML 文件：左侧目录树可点击、右侧即时渲染、支持全文检索，
并提供「打开原文件」链接。

设计约束
--------
* **file:// 直接可用**：所有内容内嵌进 HTML（浏览器在 file 协议下会
  拦截 fetch/XHR 读取本地 JSON，故不能外挂数据文件）。
* 图片用相对路径引用（同源目录，file:// 下可加载）。
* 纯标准库生成，无第三方依赖。

用法：python3 _build_library.py

> 注：这是 book010 的**原始实现，已冻结归档**。
> 后续书目（011~020）请改用系列通用版：`../../_tools/build_library.py`
> （支持自动发现分组 + library.json 配置，CSS/JS 渲染器与本脚本同源）。
"""
import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "guidance_library.html")

# 目录分组（相对 book010 根的路径）
GROUPS = [
    ("入口", [
        ("README.md", "book010 主索引"),
    ]),
    ("数据资产（理论 / 公式 / 经验）", [
        ("assets/00_INDEX.md", "资产总索引"),
        ("assets/01_theory.md", "理论体系（30 卡）"),
        ("assets/02_formulas.md", "公式集（197 条）"),
        ("assets/03_heuristics.md", "经验法则集（439 条）"),
        ("assets/04_parameters.md", "工程参数速查表"),
        ("assets/05_checklist.md", "设计与仿真检查清单"),
        ("assets/data/formulas.csv", "公式 CSV（可检索）"),
        ("assets/data/heuristics.csv", "经验 CSV（可检索）"),
    ]),
    ("Python 算法库", [
        ("algorithms/README.md", "算法库说明"),
        ("algorithms/gnc_core.py", "基础工具层"),
        ("algorithms/dynamics.py", "执行机构与积分"),
        ("algorithms/guidance_laws.py", "10 种制导律"),
        ("algorithms/seeker.py", "导引头与噪声"),
        ("algorithms/engagement.py", "交战仿真引擎"),
        ("algorithms/metrics.py", "指标与评级"),
        ("algorithms/monte_carlo.py", "蒙特卡洛框架"),
        ("algorithms/run_all.py", "一键自检 + 演示"),
    ]),
    ("RIA++ 卡片原文", [
        ("domain_A_PN.md", "域 A 比例导引"),
        ("domain_B_Optimal.md", "域 B 最优制导"),
        ("domain_C_Terminal.md", "域 C 终端制导"),
        ("domain_D_SystemModeling.md", "域 D 系统建模仿真"),
        ("domain_E_Trajectory.md", "域 E 轨迹控制"),
    ]),
    ("符号推导与弹道仿真", [
        ("verify_lqr_pn.py", "SymPy 符号推导"),
        ("verify_engagement.py", "弹道仿真（v3）"),
        ("verify_engagement.png", "8 面板仿真结果图"),
    ]),
    ("资产抽取与站点生成", [
        ("_extract_assets.py", "卡片 → 结构化资产"),
        ("_build_library.py", "本站点生成器"),
    ]),
]

MAX_CSV_ROWS = 300          # CSV 预览行数上限（控制文件体积）
SKIP_EXT = {".json", ".png", ".jpg", ".html"}


def read(p):
    return open(p, encoding="utf-8", errors="replace").read()


def csv_to_md(path, max_rows=MAX_CSV_ROWS):
    """CSV → Markdown 表格（截断到 max_rows 行）。"""
    import csv as _csv
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(_csv.reader(f))
    if not rows:
        return "(空文件)"
    total = len(rows) - 1
    head, body = rows[0], rows[1:max_rows + 1]
    L = ["| " + " | ".join(head) + " |",
         "|" + "|".join(["---"] * len(head)) + "|"]
    for r in body:
        cells = [c.replace("|", "\\|").replace("\n", " ") for c in r]
        cells += [""] * (len(head) - len(cells))
        L.append("| " + " | ".join(cells[:len(head)]) + " |")
    if total > max_rows:
        L.append(f"\n> 仅预览前 {max_rows} 行，完整数据共 {total} 行"
                 f"（见 `{os.path.basename(path)}`）")
    return "\n".join(L)


def build_tree():
    tree, files = [], []
    for gname, items in GROUPS:
        node = {"group": gname, "items": []}
        for rel, desc in items:
            p = os.path.join(HERE, rel)
            if not os.path.exists(p):
                print(f"  ! 缺失：{rel}")
                continue
            ext = os.path.splitext(rel)[1].lower()
            idx = len(files)
            if ext == ".csv":
                content = csv_to_md(p)
                kind = "md"
            elif ext in SKIP_EXT:
                content = ""
                kind = ext.lstrip(".")
            elif ext == ".md":
                content = read(p)
                kind = "md"
            else:
                content = read(p)
                kind = "py"
            size = os.path.getsize(p)
            files.append({"path": rel, "kind": kind, "text": content,
                          "size": size})
            node["items"].append({"i": idx, "name": os.path.basename(rel),
                                  "desc": desc, "size": size})
        if node["items"]:
            tree.append(node)
    return tree, files


CSS = """
:root{
  --bg:#f7f8fa; --panel:#ffffff; --line:#e3e6ea; --text:#1f2328;
  --muted:#6b7280; --accent:#1f6feb; --accent-soft:#e8f0fe;
  --code-bg:#f4f5f7; --tag:#eef2f7;
}
*{box-sizing:border-box}
body{margin:0;font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",
  "Hiragino Sans GB","Microsoft YaHei",sans-serif;color:var(--text);
  background:var(--bg)}
#app{display:flex;height:100vh;overflow:hidden}
/* ---------- 侧栏 ---------- */
#side{width:310px;flex:0 0 310px;background:var(--panel);
  border-right:1px solid var(--line);display:flex;flex-direction:column}
#side header{padding:16px 18px 12px;border-bottom:1px solid var(--line)}
#side h1{margin:0;font-size:16px;font-weight:650}
#side .sub{margin-top:4px;font-size:12px;color:var(--muted)}
#q{width:100%;margin-top:12px;padding:8px 10px;border:1px solid var(--line);
  border-radius:8px;font-size:13px;outline:none}
#q:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
#tree{flex:1;overflow-y:auto;padding:8px 8px 24px}
.grp{margin:10px 4px 4px;font-size:11px;letter-spacing:.5px;color:var(--muted);
  text-transform:uppercase;font-weight:600;padding-left:8px}
.it{display:block;padding:7px 10px;border-radius:7px;cursor:pointer;
  font-size:13.5px;text-decoration:none;color:var(--text);
  border:1px solid transparent}
.it:hover{background:var(--code-bg)}
.it.on{background:var(--accent-soft);border-color:#cfe0fb;color:#0b4fc4;
  font-weight:600}
.it .nm{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.it .ds{display:block;font-size:11.5px;color:var(--muted);margin-top:1px}
.it.hide{display:none}
/* ---------- 主区 ---------- */
#main{flex:1;overflow-y:auto;padding:0 0 80px}
#bar{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.94);
  backdrop-filter:blur(6px);border-bottom:1px solid var(--line);
  padding:10px 28px;display:flex;align-items:center;gap:14px;font-size:13px}
#bar .ttl{font-weight:600}
#bar .meta{color:var(--muted)}
#bar a{margin-left:auto;color:var(--accent);text-decoration:none;
  border:1px solid var(--line);padding:5px 12px;border-radius:7px;
  background:#fff}
#bar a:hover{background:var(--accent-soft)}
#body{max-width:960px;margin:0 auto;padding:26px 28px}
#body h1,#body h2,#body h3,#body h4{line-height:1.35;margin:1.6em 0 .6em}
#body h1{font-size:26px;border-bottom:2px solid var(--line);padding-bottom:10px;
  margin-top:0}
#body h2{font-size:20px;border-left:4px solid var(--accent);
  padding-left:10px}
#body h3{font-size:16.5px;color:#0b4fc4}
#body h4{font-size:15px;color:var(--muted)}
#body p{margin:.7em 0}
#body ul,#body ol{padding-left:24px}
#body li{margin:.3em 0}
#body blockquote{margin:1em 0;padding:8px 14px;background:#fbfcfd;
  border-left:3px solid var(--accent);color:#40464d}
#body blockquote p{margin:.3em 0}
#body table{border-collapse:collapse;width:100%;margin:1em 0;font-size:13.5px;
  display:block;overflow-x:auto}
#body th,#body td{border:1px solid var(--line);padding:7px 10px;
  text-align:left;vertical-align:top}
#body th{background:#f2f4f7;font-weight:600}
#body tr:nth-child(even) td{background:#fafbfc}
#body code{background:var(--code-bg);padding:1px 5px;border-radius:4px;
  font:13px/1.5 "SF Mono",Menlo,Consolas,monospace}
#body pre{background:#1f2430;color:#e6e9ef;padding:14px 16px;
  border-radius:9px;overflow-x:auto;font:12.8px/1.65 "SF Mono",Menlo,
  Consolas,monospace}
#body pre code{background:none;padding:0;color:inherit}
#body hr{border:none;border-top:1px solid var(--line);margin:2em 0}
.tex{background:#fbfcfd;border:1px solid var(--line);border-radius:6px;
  padding:8px 12px;margin:.6em 0;font:14px/1.6 "SF Mono",Menlo,monospace;
  overflow-x:auto;color:#243b53}
.imgbox{text-align:center;margin:1.2em 0}
.imgbox img{max-width:100%;border:1px solid var(--line);border-radius:9px}
/* Python 高亮 */
.k{color:#ff7b72}.s{color:#a5d6ff}.c{color:#8b949e;font-style:italic}
.n{color:#79c0ff}.f{color:#d2a8ff}.d{color:#ffa657}
.empty{color:var(--muted);padding:40px;text-align:center}
"""

JS = r"""
const TREE = __TREE__;
const FILES = __FILES__;
/* 预建小写索引：避免每次按键都对 ~1 MB 正文做 toLowerCase */
const LOWER = FILES.map(f => (f.path + ' ' + (f.text || '')).toLowerCase());
const $ = s => document.querySelector(s);

/* ---------- 工具 ---------- */
function esc(t){return t.replace(/[&<>]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
/* 公式专用转义：只处理尖括号，保留 & 给 LaTeX 对齐符 */
function texEsc(t){return t.replace(/</g,'&lt;').replace(/>/g,'&gt;');}

/* ---------- Python 语法高亮（单遍 token 化，避免嵌套替换） ---------- */
const KW = new Set(('def class return if elif else for while import from as '
 +'in not and or None True False try except finally raise with lambda global '
 +'pass break continue assert yield async await del is None').split(' '));
const BI = new Set(('print len range int float str list dict tuple set math '
 +'sum min max abs open self super isinstance enumerate zip sorted round '
 +'hypot sqrt exp atan2 sin cos tan radians degrees os sys random csv json '
 +'time subprocess').split(' '));
function py(src){
  /* 三引号字符串用 [\u0022]{3} / [\u0027]{3} 表示，避免与宿主语言定界符冲突 */
  const re = /#[^\n]*|[\u0022]{3}[\s\S]*?[\u0022]{3}|[\u0027]{3}[\s\S]*?[\u0027]{3}|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\b\d+\.?\d*(?:[eE][-+]?\d+)?\b|\b[A-Za-z_]\w*\b/g;
  let out='', last=0, m;
  while((m = re.exec(src))){
    out += esc(src.slice(last, m.index));
    const t = m[0];
    if(t[0]==='#') out += '<span class="c">'+esc(t)+'</span>';
    else if(t[0]==='"'||t[0]==="'") out += '<span class="s">'+esc(t)+'</span>';
    else if(/^[0-9]/.test(t)) out += '<span class="n">'+esc(t)+'</span>';
    else if(KW.has(t)) out += '<span class="k">'+esc(t)+'</span>';
    else if(BI.has(t)) out += '<span class="d">'+esc(t)+'</span>';
    else out += esc(t);
    last = m.index + t.length;
  }
  out += esc(src.slice(last));
  return out;
}

/* ---------- Markdown 渲染 ---------- */
function md(src){
  const blocks=[];                       // 占位保存公式/代码
  const stash = s => {blocks.push(s); return '\u0000'+(blocks.length-1)+'\u0000';};
  let t = src;
  // 显示公式 $$...$$（先 stash，避免被后续 Markdown 规则破坏）
  t = t.replace(/\$\$([\s\S]+?)\$\$/g, (m,p)=>
      stash('<div class="tex">$$'+texEsc(p.trim())+'$$</div>'));
  // 围栏代码块
  t = t.replace(/```(\w*)\n([\s\S]*?)```/g, (m,lang,code)=>
      stash('<pre><code>'+(lang==='python'?py(code):esc(code))+'</code></pre>'));
  t = esc(t);
  // 保护 LaTeX 转义 \*（如 A\* 算法 / RRT\*）——否则 `\*` 会被下面的强调规则
  // 当成 emphasis 起始符，吞掉一个 * 后导致后续 ** 配对整体错位。
  t = t.replace(/\\\*/g, '\u0001');
  // 行内公式 $...$
  t = t.replace(/(^|[^\\])\$([^$\n]+?)\$/g,(m,a,p)=>a+'$'+p+'$');
  // 标题
  t = t.replace(/^####\s+(.*)$/gm,'<h4>$1</h4>')
       .replace(/^###\s+(.*)$/gm,'<h3>$1</h3>')
       .replace(/^##\s+(.*)$/gm,'<h2>$1</h2>')
       .replace(/^#\s+(.*)$/gm,'<h1>$1</h1>');
  // 分隔线 / 引用
  t = t.replace(/^---+$/gm,'<hr>');
  t = t.replace(/^&gt;\s?(.*)$/gm,'<blockquote><p>$1</p></blockquote>');
  // 表格
  t = t.replace(/(^\|.*\|$\n?)+/gm, blk=>{
    const rows = blk.trim().split('\n').filter(r=>/^\|/.test(r));
    if(rows.length<2) return blk;
    const cell = r => r.replace(/^\||\|$/g,'').split('|').map(c=>c.trim());
    const hd = cell(rows[0]);
    if(!/^-{2,}$/.test((cell(rows[1])[0]||'').replace(/:/g,'')))
      return blk;
    let h = '<table><thead><tr>'+hd.map(c=>'<th>'+c+'</th>').join('')+
            '</tr></thead><tbody>';
    for(const r of rows.slice(2)){
      const cs = cell(r);
      h += '<tr>'+hd.map((_,i)=>'<td>'+(cs[i]||'')+'</td>').join('')+'</tr>';
    }
    return h+'</tbody></table>';
  });
  // 列表
  t = t.replace(/(?:^[-*]\s+.*$\n?)+/gm, b=>
      '<ul>'+b.trim().split('\n').map(x=>
        '<li>'+x.replace(/^[-*]\s+/,'')+'</li>').join('')+'</ul>');
  t = t.replace(/(?:^\d+[.)]\s+.*$\n?)+/gm, b=>
      '<ol>'+b.trim().split('\n').map(x=>
        '<li>'+x.replace(/^\d+[.)]\s+/,'')+'</li>').join('')+'</ol>');
  // 行内样式
  t = t.replace(/`([^`]+)`/g,'<code>$1</code>')
       .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
       .replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<em>$2</em>');
  // 段落
  t = t.split(/\n{2,}/).map(p=>{
    p = p.trim(); if(!p) return '';
    if(/^<(h\d|ul|ol|table|blockquote|hr|pre|div|p)/.test(p)) return p;
    return '<p>'+p.replace(/\n/g,'<br>')+'</p>';
  }).join('\n');
  return t.replace(/\u0000(\d+)\u0000/g,(m,i)=>blocks[+i])
          .replace(/\u0001/g,'\\*');
}

/* ---------- 渲染条目 ---------- */
function show(i){
  const f = FILES[i];
  document.querySelectorAll('.it').forEach(e=>e.classList.remove('on'));
  const el = document.querySelector('.it[data-i="'+i+'"]');
  if(el){ el.classList.add('on'); el.scrollIntoView({block:'nearest'}); }
  $('#bar .ttl').textContent = f.path;
  $('#bar .meta').textContent = (f.size/1024).toFixed(1)+' KB';
  $('#bar a').href = encodeURI(f.path);
  let h = '';
  if(f.kind==='md')       h = md(f.text);
  else if(f.kind==='py')  h = '<pre><code>'+py(f.text)+'</code></pre>';
  else if(f.kind==='png') h = '<div class="imgbox"><img src="'+
                              encodeURI(f.path)+'" alt="'+f.path+'"></div>';
  else if(f.kind==='.json'||f.kind==='json')
    h = '<div class="empty">JSON 数据文件（'+(f.size/1024).toFixed(1)+
        ' KB）—— 点击右上角「打开原文件」查看完整内容。<br><br>'+
        '结构化资产可通过 <code>assets/data/assets.json</code> 用 Python 加载：'+
        '<pre><code>import json\nd=json.load(open("assets/data/assets.json",'+
        'encoding="utf-8"))\nprint(d["meta"]["stats"])</code></pre></div>';
  else h = '<div class="empty">该文件类型需打开原文件查看。</div>';
  $('#body').innerHTML = h;
  $('#main').scrollTop = 0;
  if(window.MathJax) window.MathJax.typesetPromise &&
    window.MathJax.typesetPromise([$('#body')]);
  location.hash = 'f'+i;
}

/* ---------- 建树 ---------- */
function build(){
  const box = $('#tree'); let first = -1;
  TREE.forEach(g=>{
    const h = document.createElement('div');
    h.className='grp'; h.textContent=g.group; box.appendChild(h);
    g.items.forEach(it=>{
      if(first<0) first=it.i;
      const a = document.createElement('a');
      a.className='it'; a.dataset.i=it.i;
      a.innerHTML='<span class="nm">'+esc(it.name)+'</span>'+
                  '<span class="ds">'+esc(it.desc)+' · '+
                  (it.size/1024).toFixed(1)+' KB</span>';
      a.onclick=()=>show(it.i);
      box.appendChild(a);
    });
  });
  // 搜索（文件名 + 描述 + 正文），150 ms 防抖 + 预建索引
  let timer = null;
  $('#q').oninput = e=>{
    const k = e.target.value.trim().toLowerCase();
    clearTimeout(timer);
    timer = setTimeout(()=>{
      document.querySelectorAll('.it').forEach(a=>{
        if(!k){ a.classList.remove('hide'); return; }
        const i = +a.dataset.i;
        const hit = (LOWER[i] + ' ' + a.textContent.toLowerCase()).includes(k);
        a.classList.toggle('hide', !hit);
      });
      document.querySelectorAll('.grp').forEach(g=>{
      let n=g.nextElementSibling, any=false;
      while(n && n.classList.contains('it')){
        if(!n.classList.contains('hide')) any=true; n=n.nextElementSibling;
      }
      g.style.display = any?'':'none';
      });
    }, 150);
  };
  return first;
}

const start = +(location.hash.replace('#f','')||-1);
const f0 = build();
show(start>=0 && start<FILES.length ? start : f0);

/* ---------- MathJax 离线降级提示 ---------- */
setTimeout(()=>{
  if(!window.MathJax || !window.MathJax.typesetPromise){
    const s=document.createElement('span');
    s.className='meta'; s.style.color='#b45309';
    s.textContent='⚠ 未能加载 MathJax（需联网），公式以 LaTeX 源码原样显示';
    $('#bar').insertBefore(s, $('#bar a'));
  }
}, 5000);
"""


def main():
    tree, files = build_tree()
    total = sum(f["size"] for f in files)

    def jsafe(obj):
        """JSON 序列化并阻断 </script> 提前闭合（JS 中 <\\/ 等价于 </）。"""
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

    js = (JS.replace("__TREE__", jsafe(tree))
            .replace("__FILES__", jsafe(files)))
    doc = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>book010 导弹制导 — 数据资产与算法库</title>
<script>
window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],
  displayMath:[['$$','$$'],['\\\\[','\\\\]']]},
  options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>__CSS__</style></head>
<body><div id="app">
<aside id="side"><header>
  <h1>book010 · 导弹制导</h1>
  <div class="sub">数据资产 + Python 算法库 · 单文件离线资料库</div>
  <input id="q" placeholder="搜索文件名或正文…">
</header><nav id="tree"></nav></aside>
<main id="main"><div id="bar">
  <span class="ttl"></span><span class="meta"></span>
  <a target="_blank" rel="noopener">打开原文件 ↗</a>
</div><div id="body"></div></main>
</div><script>__JS__</script></body></html>"""
    # 单次正则替换：插入的内容不再参与后续匹配。
    # 注意 —— 本站把生成器自身的源码也收进了资料库，源码里含 __CSS__/__JS__
    # 等字面量；若用链式 str.replace，先插入的内容会在后一轮被误替换。
    doc = re.sub(r"__(CSS|JS)__",
                 lambda m: {"CSS": CSS, "JS": js}[m.group(1)], doc)
    open(OUT, "w", encoding="utf-8").write(doc)

    n_files = len(files)
    n_groups = len(tree)
    print("=" * 62)
    print("  HTML 资料库生成完成")
    print("=" * 62)
    for g in tree:
        print(f"  {g['group']}: {len(g['items'])} 项")
    print("-" * 62)
    print(f"  文件 {n_files} 个 / 分组 {n_groups} 个 / "
          f"源文合计 {total/1024:.0f} KB")
    print(f"  输出：{OUT}")
    print(f"  体积：{os.path.getsize(OUT)/1024:.0f} KB")
    print("=" * 62)


if __name__ == "__main__":
    main()
