#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用单文件 HTML 资料库生成器 — 系列书目复用

把一本书的全部产出（数据资产 / Python 算法库 / RIA++ 卡片 / 仿真脚本）
打包进一个 HTML：左侧目录树可点击、右侧即时渲染、全文检索、
并提供「打开原文件」链接。

用法：
  python3 build_library.py --book-dir ../book010
  python3 build_library.py                        # 默认当前目录

分组规则
--------
优先用 <book-dir>/library.json 的 `groups`（显式编排，顺序可控）：
  "groups": [
    {"name": "数据资产", "items": [
      {"path": "assets/00_INDEX.md", "desc": "资产总索引"}
    ]}
  ]

无配置时按 AUTO_GROUPS 规则自动发现（新书零配置即可用）：
  "assets/" → 数据资产    "algorithms/" → Python 算法库
  "domain_*.md" → 卡片原文   "verify_*" → 仿真    "_*.py" → 工具脚本

设计约束
--------
* **file:// 直接可用**：所有内容内嵌进 HTML（浏览器在 file 协议下会拦截
  fetch/XHR 读取本地 JSON，故不能外挂数据文件）。
* 图片用相对路径引用（同源目录，file:// 下可加载）。
* 纯标准库生成，无第三方依赖。

已知坑（改这个文件时务必保留对应修法）
--------------------------------------
1. JS 正则里不能出现连续三个双引号或单引号 —— 生成器源码会被收进资料库，
   裸三引号会提前终止 Python 的三引号字符串。改用 [\\u0022]{3} / [\\u0027]{3}。
2. 占位符必须单次替换（re.sub），链式 str.replace 会被先插入的内容二次替换。
3. 正文里的 `</` 要转成 `<\\/`，否则提前闭合 <script>。
4. Markdown 渲染前必须保护 LaTeX 的 `\\*`（如 A\\* 算法），
   否则被强调规则当 emphasis 起点，导致 `**` 配对整体错位。
"""
import argparse
import csv
import fnmatch
import html
import json
import os
import re
import sys

MAX_CSV_ROWS = 300          # CSV 预览行数上限（控制文件体积）
SKIP_EXT = {".json", ".png", ".jpg", ".jpeg", ".html"}

# 自动发现规则（按序匹配，每个文件归入首个命中的组）
AUTO_GROUPS = [
    ("入口", ["README.md"]),
    ("数据资产（理论 / 公式 / 经验）", ["assets/*.md", "assets/data/*.csv"]),
    ("Python 算法库", ["algorithms/README.md", "algorithms/*.py"]),
    ("RIA++ 卡片原文", ["domain_*.md"]),
    ("符号推导与弹道仿真", ["verify_*.py", "verify_*.png"]),
    ("工具脚本", ["_*.py"]),
]


def read(p):
    return open(p, encoding="utf-8", errors="replace").read()


def csv_to_md(path, max_rows=MAX_CSV_ROWS):
    """CSV → Markdown 表格（截断到 max_rows 行）。"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
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
        L.append("\n> 仅预览前 %d 行，完整数据共 %d 行（见 `%s`）"
                 % (max_rows, total, os.path.basename(path)))
    return "\n".join(L)


def load_config(book_dir, cfg_path=None, cli_id=None):
    """读 library.json。优先级：--id > library.json > 目录名。

    三个脚本必须用同一套 id 优先级，否则流水线里上一阶段写出的文件名
    会和下一阶段要找的对不上（目录名常常不等于书目 id）。
    """
    cfg_path = cfg_path or os.path.join(book_dir, "library.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    if cli_id:
        cfg["id"] = cli_id
    cfg.setdefault("id", os.path.basename(os.path.abspath(book_dir)))
    cfg.setdefault("title", cfg["id"])
    cfg.setdefault("html_title", "%s — 数据资产与算法库" % cfg["id"])
    # 输出文件名：library.json 的 "out" 优先，否则 <id>_library.html。
    # 留这个口子是因为 book010 的既有交付叫 guidance_library.html，
    # 已被 README 与索引多处引用，改名会破坏既有链接。
    cfg.setdefault("out", "%s_library.html" % cfg["id"])
    return cfg


def auto_groups(book_dir):
    """按 AUTO_GROUPS 规则自动发现文件并分组。"""
    collected = {}                       # group -> [(rel, desc), ...]
    used = set()
    for gname, patterns in AUTO_GROUPS:
        hits = []
        for pat in patterns:
            for root, _dirs, files in os.walk(book_dir):
                # 修剪：不进入隐藏目录 / 源码目录 / 原型目录
                _dirs[:] = [d for d in _dirs
                            if not d.startswith(".")
                            and d not in ("source", "prototype")]
                rel_root = os.path.relpath(root, book_dir)
                # 跳过隐藏目录 / 源码目录 / 原型目录。
                # 注意顶层 rel_root 就是 "."，不能简单用 startswith(".")
                # 判断——那会把根目录的所有文件（README.md、domain_*.md）全漏掉。
                if rel_root != ".":
                    parts = rel_root.split(os.sep)
                    if any(p.startswith(".") or p in ("source", "prototype")
                           for p in parts):
                        continue
                for fn in files:
                    rel = os.path.normpath(os.path.join(rel_root, fn)) \
                        if rel_root != "." else fn
                    if rel in used:
                        continue
                    if fnmatch.fnmatch(rel.replace(os.sep, "/"), pat):
                        used.add(rel)
                        hits.append(rel)
        if hits:
            collected[gname] = sorted(set(hits))
    return [(g, [(p, "") for p in ps]) for g, ps in collected.items()]


def resolve_groups(book_dir, cfg):
    """返回 [(组名, [(相对路径, 描述), ...]), ...]"""
    if cfg.get("groups"):
        out = []
        for g in cfg["groups"]:
            items = [(it["path"], it.get("desc", "")) for it in g["items"]]
            out.append((g["name"], items))
        return out
    return auto_groups(book_dir)


def build_tree(book_dir, groups):
    tree, files = [], []
    for gname, items in groups:
        node = {"group": gname, "items": []}
        for rel, desc in items:
            p = os.path.join(book_dir, rel)
            if not os.path.exists(p):
                print("  ! 缺失：%s" % rel)
                continue
            ext = os.path.splitext(rel)[1].lower()
            idx = len(files)
            if ext == ".csv":
                content, kind = csv_to_md(p), "md"
            elif ext in SKIP_EXT:
                content, kind = "", ext.lstrip(".")
            elif ext == ".md":
                content, kind = read(p), "md"
            else:
                content, kind = read(p), "py"
            size = os.path.getsize(p)
            files.append({"path": rel, "kind": kind, "text": content,
                          "size": size})
            node["items"].append({"i": idx, "name": os.path.basename(rel),
                                  "desc": desc, "size": size})
        if node["items"]:
            tree.append(node)
    return tree, files


CSS = r"""
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
    ap = argparse.ArgumentParser(description="书目产出 → 单文件 HTML 资料库")
    ap.add_argument("--book-dir", default=".", help="书目目录（默认当前目录）")
    ap.add_argument("--config", default=None, help="配置文件（默认 <book-dir>/library.json）")
    ap.add_argument("--out", default=None, help="输出 HTML（默认 <book-dir>/<id>_library.html）")
    ap.add_argument("--id", default=None, help="书目 id（默认取目录名）")
    a = ap.parse_args()

    book_dir = os.path.abspath(a.book_dir)
    if not os.path.isdir(book_dir):
        sys.exit("书目目录不存在：%s" % book_dir)
    cfg = load_config(book_dir, a.config, a.id)
    out = os.path.abspath(a.out) if a.out else \
        os.path.join(book_dir, cfg["out"])

    groups = resolve_groups(book_dir, cfg)
    tree, files = build_tree(book_dir, groups)
    if not files:
        sys.exit("没有可打包的文件")

    def jsafe(obj):
        """JSON 序列化并阻断 </script> 提前闭合（JS 中 <\\/ 等价于 </）。"""
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

    js = (JS.replace("__TREE__", jsafe(tree))
            .replace("__FILES__", jsafe(files)))

    doc = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__HTML_TITLE__</title>
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
  <h1>__TITLE__</h1>
  <div class="sub">数据资产 + Python 算法库 · 单文件离线资料库</div>
  <input id="q" placeholder="搜索文件名或正文…">
</header><nav id="tree"></nav></aside>
<main id="main"><div id="bar">
  <span class="ttl"></span><span class="meta"></span>
  <a target="_blank" rel="noopener">打开原文件 ↗</a>
</div><div id="body"></div></main>
</div><script>__JS__</script></body></html>"""

    # 单次正则替换：插入的内容不再参与后续匹配（见文件头「已知坑 2」）
    doc = re.sub(r"__(CSS|JS|HTML_TITLE|TITLE)__", lambda m: {
        "CSS": CSS,
        "JS": js,
        "HTML_TITLE": html.escape(cfg["html_title"]),
        "TITLE": html.escape(cfg["title"]),
    }[m.group(1)], doc)

    open(out, "w", encoding="utf-8").write(doc)

    total = sum(f["size"] for f in files)
    print("=" * 62)
    print("  HTML 资料库生成完成：%s" % cfg["id"])
    print("=" * 62)
    for g in tree:
        print("  %s: %d 项" % (g["group"], len(g["items"])))
    print("-" * 62)
    print("  文件 %d 个 / 分组 %d 个 / 源文合计 %.0f KB"
          % (len(files), len(tree), total / 1024))
    print("  输出：%s" % out)
    print("  体积：%.0f KB" % (os.path.getsize(out) / 1024))
    print("=" * 62)


if __name__ == "__main__":
    main()
