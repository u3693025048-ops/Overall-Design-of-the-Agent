#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 5 个 domain_*.md（30 张 RIA++ 卡）中抽取结构化数据资产。

> 注：这是 book010 的**原始实现，已冻结归档**。
> 后续书目（011~020）请改用系列通用版：`../../_tools/extract_assets.py`
> （支持自动发现 domain_*.md、library.json 配置，产出与本脚本逐字节一致）。

产出：
  assets/data/assets.json      30 卡全量结构化（理论/公式/经验/关联）
  assets/data/formulas.csv     全部 LaTeX 公式（按卡编号，可检索）
  assets/data/heuristics.csv   全部经验条目（按条拆分，带要点标题）
  assets/01_theory.md          理论体系（30 卡理论段汇总）
  assets/02_formulas.md        公式集（公式 + 上下文 + 符号说明）
  assets/03_heuristics.md      经验法则集（按域重排 + 全局编号）

纯标准库实现，Python 3.9+ 可直接运行。
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
DATA = os.path.join(OUT, "data")

DOMAINS = [
    ("A", "domain_A_PN.md", "比例导引 PN"),
    ("B", "domain_B_Optimal.md", "最优制导"),
    ("C", "domain_C_Terminal.md", "终端制导"),
    ("D", "domain_D_SystemModeling.md", "制导系统建模仿真"),
    ("E", "domain_E_Trajectory.md", "飞行器轨迹控制"),
]

CARD_RE = re.compile(r"^### 卡\s*(\d+)[:：]\s*《(.+?)》\s*$", re.M)
SEC_RE = re.compile(r"^\*\*【(.+?)】\*\*\s*$", re.M)
SRC_RE = re.compile(r"^\*\*来源\*\*[：:]\s*(.+?)\s*$", re.M)
DISPLAY_RE = re.compile(r"^\$\$(.+?)\$\$\s*$", re.M | re.S)


def split_sections(body):
    """按 **【XXX】** 标记把卡片正文切成 {标记: 内容}。"""
    marks = list(SEC_RE.finditer(body))
    out = {}
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out[m.group(1)] = body[start:end].strip()
    return out


def parse_cards(path):
    text = open(path, encoding="utf-8").read()
    # 头部元信息（域名、精读章节）
    head = text.split("---", 1)[0]
    chapters = ""
    m = re.search(r"^>\s*精读章节共\s*(\d+)\s*章[：:]\s*(.+?)\s*$", head, re.M)
    if m:
        chapters = m.group(2).strip()

    cards = []
    matches = list(CARD_RE.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        body = body.split("\n---", 1)[0]          # 去掉卡片分隔线之后的内容
        sec = split_sections(body)
        sm = SRC_RE.search(body)
        cards.append({
            "no": int(m.group(1)),
            "title": m.group(2),
            "source": sm.group(1) if sm else "",
            "theory": sec.get("R - 理论", ""),
            "formulas": sec.get("I - 公式", ""),
            "heuristics": sec.get("A++ - 经验", ""),
            "relation": sec.get("关联", ""),
        })
    return {"chapters": chapters, "cards": cards}


def latex_blocks(text):
    """抽取 $$ ... $$ 显示公式块。"""
    return [b.strip() for b in DISPLAY_RE.findall(text) if b.strip()]


def cleanup(items):
    """清理：丢弃段首引导语（无编号且过短）与噪声空条。"""
    return [(n, b) for n, b in items
            if len(b.strip()) >= 15 and not (n is None and len(b.strip()) < 40)]


def split_heuristics(text):
    """把经验段拆成单条：编号列表优先，退化为段落。"""
    items = []
    lines = text.split("\n")
    cur_no, cur = None, []
    num_re = re.compile(r"^\s*(\d+)[\.、)]\s+(.*)$")

    def flush():
        if cur:
            items.append((cur_no, "\n".join(cur).strip()))

    for ln in lines:
        m = num_re.match(ln)
        if m:
            flush()
            cur_no, cur = int(m.group(1)), [m.group(2)]
        elif not ln.strip() or set(ln.strip()) <= {"-", "*"}:
            if cur:
                cur.append("")
        elif cur is not None:
            cur.append(ln)
        elif ln.strip():
            cur_no, cur = None, [ln]
    flush()
    return [(n, c) for n, c in items if c.strip()]


def merge_lead(heur_items):
    """把无编号的段首引导语并入下一条（如「作者踩过的坑：」）。"""
    out = []
    carry = ""
    for n, b in heur_items:
        if n is None:
            carry = (carry + " " + b).strip()
            continue
        out.append((n, (carry + "\n\n" + b).strip() if carry else b))
        carry = ""
    if carry and out:
        out[-1] = (out[-1][0], out[-1][1] + "\n\n" + carry)
    elif carry:
        out.append((0, carry))
    return out


def gist(item):
    """取经验条的要点标题（首个 **加粗** 片段）。"""
    m = re.search(r"\*\*(.+?)\*\*", item)
    if m:
        g = m.group(1)
        return g if len(g) <= 40 else g[:38] + "…"
    return item[:30].replace("\n", " ")


def main():
    os.makedirs(DATA, exist_ok=True)
    all_cards, formulas, heur = [], [], []
    gid = 0

    for code, fn, name in DOMAINS:
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            print(f"  ! 缺少 {fn}")
            continue
        dom = parse_cards(path)
        dom["code"], dom["name"], dom["file"] = code, name, fn
        all_cards.append(dom)

        for c in dom["cards"]:
            cid = f"{code}-{c['no']:02d}"
            fs = latex_blocks(c["formulas"])
            c["id"], c["domain"], c["domain_name"] = cid, code, name
            c["formula_tex"] = fs
            for k, tex in enumerate(fs, 1):
                formulas.append({
                    "card": cid, "card_title": c["title"], "domain": code,
                    "domain_name": name, "idx": k,
                    "fid": f"F-{cid}-{k:02d}", "latex": tex,
                })
            hs = merge_lead(split_heuristics(c["heuristics"]))
            hs = cleanup(hs)
            for n, body in hs:
                gid += 1
                heur.append({
                    "hid": f"H{gid:03d}", "card": cid, "card_title": c["title"],
                    "domain": code, "domain_name": name, "no": n or 0,
                    "gist": gist(body), "body": body,
                })
            c["heur_items"] = [{"no": n, "gist": gist(b), "body": b} for n, b in hs]

    # ---------- JSON ----------
    meta = {
        "book": "《系统化飞控实战》010 · 导弹制导",
        "generated_by": "_extract_assets.py",
        "stats": {
            "domains": len(all_cards),
            "cards": sum(len(d["cards"]) for d in all_cards),
            "formulas": len(formulas),
            "heuristics": len(heur),
            "chapters_read": sum(1 for d in all_cards if d["chapters"]),
        },
    }
    with open(os.path.join(DATA, "assets.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "domains": all_cards}, f,
                  ensure_ascii=False, indent=1)

    # ---------- CSV ----------
    with open(os.path.join(DATA, "formulas.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, ["fid", "domain", "domain_name", "card",
                               "card_title", "idx", "latex"])
        w.writeheader()
        w.writerows(formulas)
    with open(os.path.join(DATA, "heuristics.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, ["hid", "domain", "domain_name", "card",
                               "card_title", "no", "gist", "body"])
        w.writeheader()
        w.writerows(heur)

    # ---------- Markdown ----------
    L = []
    L.append("# 数据资产 01 — 理论体系\n")
    L.append("> 来源：book010《系统化飞控实战》010 · 导弹制导，30 张 RIA++ 卡的"
             "【R - 理论】段全量汇总\n")
    L.append(f"> 规模：**{len(all_cards)} 域 / {meta['stats']['cards']} 卡 / "
             f"{len(formulas)} 条公式 / {len(heur)} 条经验**\n")
    L.append("\n---\n")
    for d in all_cards:
        L.append(f"\n## 域 {d['code']}：{d['name']}\n")
        if d["chapters"]:
            L.append(f"\n> 精读章节：{d['chapters']}\n")
        for c in d["cards"]:
            L.append(f"\n### {c['id']} 《{c['title']}》\n")
            L.append(f"\n> 来源：{c['source']}\n")
            L.append("\n" + c["theory"].strip() + "\n")
    open(os.path.join(OUT, "01_theory.md"), "w", encoding="utf-8").write(
        "\n".join(L))

    L = []
    L.append("# 数据资产 02 — 公式集\n")
    L.append("> 30 张卡【I - 公式】段中的全部显示公式，按域—卡—序号编号，"
             "编号（F-xx）可跨文档引用\n")
    L.append(f"> 规模：**{len(formulas)} 条**\n")
    L.append("\n---\n")
    cur = None
    for r in formulas:
        if r["card"] != cur:
            cur = r["card"]
            L.append(f"\n## {r['card']} 《{r['card_title']}》"
                     f"（域 {r['domain']} {r['domain_name']}）\n")
        L.append(f"\n**`{r['fid']}`**\n")
        L.append("\n$$\n" + r["latex"] + "\n$$\n")
    open(os.path.join(OUT, "02_formulas.md"), "w", encoding="utf-8").write(
        "\n".join(L))

    L = []
    L.append("# 数据资产 03 — 经验法则集\n")
    L.append("> 30 张卡【A++ - 经验】段全部条目，逐条拆分并全局编号（H001…），"
             "编号可跨文档引用\n")
    L.append(f"> 规模：**{len(heur)} 条**\n")
    L.append("\n---\n")
    cur = None
    for h in heur:
        if h["domain"] != cur:
            cur = h["domain"]
            L.append(f"\n## 域 {h['domain']}：{h['domain_name']}\n")
        L.append(f"\n### `{h['hid']}` {h['gist']}\n")
        L.append(f"\n> 出处：{h['card']} 《{h['card_title']}》\n")
        L.append("\n" + h["body"].strip() + "\n")
    open(os.path.join(OUT, "03_heuristics.md"), "w", encoding="utf-8").write(
        "\n".join(L))

    print("=" * 62)
    print("  数据资产抽取完成")
    print("=" * 62)
    for d in all_cards:
        nf = sum(len(c["formula_tex"]) for c in d["cards"])
        nh = sum(len(c["heur_items"]) for c in d["cards"])
        print(f"  域 {d['code']} {d['name']:16s} "
              f"{len(d['cards'])} 卡 / {nf:3d} 公式 / {nh:3d} 经验")
    print("-" * 62)
    print(f"  合计：{meta['stats']['domains']} 域 / "
          f"{meta['stats']['cards']} 卡 / "
          f"{len(formulas)} 公式 / {len(heur)} 经验")
    print("=" * 62)


if __name__ == "__main__":
    main()
