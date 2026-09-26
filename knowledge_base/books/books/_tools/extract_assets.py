#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用数据资产抽取器 — 系列书目复用

从一本书的 `domain_*.md`（RIA++ 技能卡）中抽取结构化数据资产：
  理论（【R】段）/ 公式（【I】段）/ 经验（【A++】段）/ 关联

产出（默认写到 <book-dir>/assets/）：
  data/assets.json      全量结构化（理论/公式/经验/关联）
  data/formulas.csv     全部 LaTeX 公式（编号 F-<卡号>-<序号>，可检索）
  data/heuristics.csv   全部经验条目（逐条拆分 + 全局编号 H001…）
  01_theory.md          理论体系（各卡【R】段汇总）
  02_formulas.md        公式集（公式 + 出处，编号可跨文档引用）
  03_heuristics.md      经验法则集（按域重排 + 全局编号）

用法：
  python3 extract_assets.py --book-dir ../book010
  python3 extract_assets.py                       # 默认当前目录

配置（可选，放 <book-dir>/library.json）：
  {
    "id":   "book010",                              # 书目短名
    "book": "《系统化飞控实战》010 · 导弹制导",        # 全书正式名
    "domains": [                                    # 省略则自动发现 domain_*.md
      {"code": "A", "file": "domain_A_PN.md", "name": "比例导引 PN"}
    ]
  }

纯标准库，Python 3.9+。
"""
import argparse
import csv
import fnmatch
import json
import os
import re
import sys

# ---------------------------------------------------------------- 卡片解析
CARD_RE = re.compile(r"^### 卡\s*(\d+)[:：]\s*《(.+?)》\s*$", re.M)
SEC_RE = re.compile(r"^\*\*【(.+?)】\*\*\s*$", re.M)
SRC_RE = re.compile(r"^\*\*来源\*\*[：:]\s*(.+?)\s*$", re.M)
DISPLAY_RE = re.compile(r"^\$\$(.+?)\$\$\s*$", re.M | re.S)

# 文件头：`# 知识域 A：比例导引（PN）— RIA++ 技能卡`
DOMAIN_HEAD_RE = re.compile(
    r"^#\s*知识域\s*([A-Za-z0-9]+)\s*[：:]\s*(.+?)\s*(?:—|–|-|~)\s*RIA", re.M)
# 文件头：`> 蒸馏来源：《系统化飞控实战》book010 第 010 本 · 导弹制导`
BOOK_SRC_RE = re.compile(r"^>\s*蒸馏来源[：:]\s*(.+?)\s*$", re.M)


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


# ---------------------------------------------------------------- 配置发现
def discover_domains(book_dir):
    """自动发现 domain_*.md，从文件头提取域码与域名。"""
    found = []
    for fn in sorted(os.listdir(book_dir)):
        if not fnmatch.fnmatch(fn, "domain_*.md"):
            continue
        text = open(os.path.join(book_dir, fn), encoding="utf-8").read()
        head = text[:400]
        m = DOMAIN_HEAD_RE.search(head)
        if m:
            code, name = m.group(1), m.group(2).strip()
        else:                                    # 退回文件名：domain_A_PN.md → A / PN
            stem = fn[len("domain_"):-3]
            code = stem.split("_")[0]
            name = stem.split("_", 1)[-1].replace("_", " ") if "_" in stem else stem
        found.append({"code": code, "file": fn, "name": name})
    return found


def load_config(book_dir, cfg_path=None, cli_id=None):
    """读 library.json；缺失字段用自动发现补全。

    cli_id 优先级最高（--id），其次 library.json，最后才是目录名——
    目录名常常不等于书目 id（如解压到 /tmp/bkfull 但 id 是 book010）。
    三个脚本必须用同一套优先级，否则流水线里上一阶段写出的文件名
    会和下一阶段要找的对不上。
    """
    cfg_path = cfg_path or os.path.join(book_dir, "library.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

    if cli_id:
        cfg["id"] = cli_id
    cfg.setdefault("id", os.path.basename(os.path.abspath(book_dir)))
    if not cfg.get("domains"):
        cfg["domains"] = discover_domains(book_dir)
    if not cfg.get("book"):
        cfg["book"] = cfg["id"]
        for d in cfg["domains"]:                 # 从卡片文件头推断书名
            p = os.path.join(book_dir, d["file"])
            if not os.path.exists(p):
                continue
            head = open(p, encoding="utf-8").read()[:600]
            m = BOOK_SRC_RE.search(head)
            if m:
                cfg["book"] = m.group(1).strip()
                break
    cfg.setdefault("generated_by", os.path.basename(__file__))
    return cfg


# ---------------------------------------------------------------- 主流程
def run(book_dir, out_dir, cfg):
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    all_cards, formulas, heur = [], [], []
    gid = 0

    for d in cfg["domains"]:
        path = os.path.join(book_dir, d["file"])
        if not os.path.exists(path):
            print("  ! 缺少 %s" % d["file"])
            continue
        code, name = d["code"], d["name"]
        dom = parse_cards(path)
        dom["code"], dom["name"], dom["file"] = code, name, d["file"]
        all_cards.append(dom)

        for c in dom["cards"]:
            cid = "%s-%02d" % (code, c["no"])
            fs = latex_blocks(c["formulas"])
            c["id"], c["domain"], c["domain_name"] = cid, code, name
            c["formula_tex"] = fs
            for k, tex in enumerate(fs, 1):
                formulas.append({
                    "card": cid, "card_title": c["title"], "domain": code,
                    "domain_name": name, "idx": k,
                    "fid": "F-%s-%02d" % (cid, k), "latex": tex,
                })
            hs = cleanup(merge_lead(split_heuristics(c["heuristics"])))
            for n, body in hs:
                gid += 1
                heur.append({
                    "hid": "H%03d" % gid, "card": cid, "card_title": c["title"],
                    "domain": code, "domain_name": name, "no": n or 0,
                    "gist": gist(body), "body": body,
                })
            c["heur_items"] = [{"no": n, "gist": gist(b), "body": b}
                               for n, b in hs]

    n_cards = sum(len(d["cards"]) for d in all_cards)
    label = "%s%s" % (cfg["id"], cfg["book"])

    # ---------- JSON ----------
    meta = {
        "book": cfg["book"],
        "book_id": cfg["id"],
        "generated_by": cfg["generated_by"],
        "stats": {
            "domains": len(all_cards),
            "cards": n_cards,
            "formulas": len(formulas),
            "heuristics": len(heur),
            "chapters_read": sum(1 for d in all_cards if d["chapters"]),
        },
    }
    with open(os.path.join(data_dir, "assets.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "domains": all_cards}, f,
                  ensure_ascii=False, indent=1)

    # ---------- CSV ----------
    with open(os.path.join(data_dir, "formulas.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, ["fid", "domain", "domain_name", "card",
                               "card_title", "idx", "latex"])
        w.writeheader()
        w.writerows(formulas)
    with open(os.path.join(data_dir, "heuristics.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, ["hid", "domain", "domain_name", "card",
                               "card_title", "no", "gist", "body"])
        w.writeheader()
        w.writerows(heur)

    # ---------- Markdown ----------
    L = ["# 数据资产 01 — 理论体系\n",
         "> 来源：%s，%d 张 RIA++ 卡的【R - 理论】段全量汇总\n" % (label, n_cards),
         "> 规模：**%d 域 / %d 卡 / %d 条公式 / %d 条经验**\n"
         % (len(all_cards), n_cards, len(formulas), len(heur)),
         "\n---\n"]
    for d in all_cards:
        L.append("\n## 域 %s：%s\n" % (d["code"], d["name"]))
        if d["chapters"]:
            L.append("\n> 精读章节：%s\n" % d["chapters"])
        for c in d["cards"]:
            L.append("\n### %s 《%s》\n" % (c["id"], c["title"]))
            L.append("\n> 来源：%s\n" % c["source"])
            L.append("\n" + c["theory"].strip() + "\n")
    open(os.path.join(out_dir, "01_theory.md"), "w",
         encoding="utf-8").write("\n".join(L))

    L = ["# 数据资产 02 — 公式集\n",
         "> %d 张卡【I - 公式】段中的全部显示公式，按域—卡—序号编号，"
         "编号（F-xx）可跨文档引用\n" % n_cards,
         "> 规模：**%d 条**\n" % len(formulas),
         "\n---\n"]
    cur = None
    for r in formulas:
        if r["card"] != cur:
            cur = r["card"]
            L.append("\n## %s 《%s》（域 %s %s）\n"
                     % (r["card"], r["card_title"], r["domain"], r["domain_name"]))
        L.append("\n**`%s`**\n" % r["fid"])
        L.append("\n$$\n" + r["latex"] + "\n$$\n")
    open(os.path.join(out_dir, "02_formulas.md"), "w",
         encoding="utf-8").write("\n".join(L))

    L = ["# 数据资产 03 — 经验法则集\n",
         "> %d 张卡【A++ - 经验】段全部条目，逐条拆分并全局编号（H001…），"
         "编号可跨文档引用\n" % n_cards,
         "> 规模：**%d 条**\n" % len(heur),
         "\n---\n"]
    cur = None
    for h in heur:
        if h["domain"] != cur:
            cur = h["domain"]
            L.append("\n## 域 %s：%s\n" % (h["domain"], h["domain_name"]))
        L.append("\n### `%s` %s\n" % (h["hid"], h["gist"]))
        L.append("\n> 出处：%s 《%s》\n" % (h["card"], h["card_title"]))
        L.append("\n" + h["body"].strip() + "\n")
    open(os.path.join(out_dir, "03_heuristics.md"), "w",
         encoding="utf-8").write("\n".join(L))

    # ---------- 报告 ----------
    print("=" * 62)
    print("  数据资产抽取完成：%s" % cfg["id"])
    print("=" * 62)
    for d in all_cards:
        nf = sum(len(c["formula_tex"]) for c in d["cards"])
        nh = sum(len(c["heur_items"]) for c in d["cards"])
        print("  域 %s %-16s %d 卡 / %3d 公式 / %3d 经验"
              % (d["code"], d["name"], len(d["cards"]), nf, nh))
    print("-" * 62)
    print("  合计：%d 域 / %d 卡 / %d 公式 / %d 经验"
          % (meta["stats"]["domains"], n_cards, len(formulas), len(heur)))
    print("  输出：%s" % out_dir)
    print("=" * 62)
    return meta["stats"]


def main():
    ap = argparse.ArgumentParser(description="RIA++ 卡片 → 结构化数据资产")
    ap.add_argument("--book-dir", default=".", help="书目目录（默认当前目录）")
    ap.add_argument("--config", default=None, help="配置文件（默认 <book-dir>/library.json）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <book-dir>/assets）")
    ap.add_argument("--id", default=None, help="书目 id（默认取目录名）")
    a = ap.parse_args()

    book_dir = os.path.abspath(a.book_dir)
    if not os.path.isdir(book_dir):
        sys.exit("书目目录不存在：%s" % book_dir)
    out_dir = os.path.abspath(a.out) if a.out else os.path.join(book_dir, "assets")
    cfg = load_config(book_dir, a.config, a.id)
    if not cfg["domains"]:
        sys.exit("未发现 domain_*.md，且 library.json 未配置 domains")
    run(book_dir, out_dir, cfg)


if __name__ == "__main__":
    main()
