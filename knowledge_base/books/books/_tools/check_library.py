#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资料库 HTML 质检（静态结构 + 运行时渲染）

静态部分（Python）：占位符注入、script 配平、内嵌 JSON 可解析、内容抽样、
CSV 转表、</script> 转义。
运行时部分（Node，可选）：调用同目录 check_render.js，用 DOM stub 执行页面
脚本，验证渲染器不抛异常、无标签错位（静态校验看不见这类 bug）。

用法：
  python3 check_library.py <library.html>            # 静态 + 运行时（若 node 可用）
  python3 check_library.py <library.html> --static   # 只跑静态

退出码：0 全部通过 / 1 存在问题
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ok = True
fails = []


def chk(cond, msg, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg + ((" | " + extra) if extra else ""))
    if not cond:
        ok = False
        fails.append(msg)


def main():
    ap = argparse.ArgumentParser(description="资料库 HTML 质检")
    ap.add_argument("html", help="待检的 library.html")
    ap.add_argument("--static", action="store_true", help="只跑静态校验")
    a = ap.parse_args()

    p = os.path.abspath(a.html)
    if not os.path.exists(p):
        sys.exit("文件不存在：%s" % p)
    doc = open(p, encoding="utf-8").read()

    print("=" * 64)
    print("  静态质检：" + os.path.basename(p))
    print("=" * 64)
    print("  体积 %.0f KB" % (len(doc.encode("utf-8")) / 1024))

    # 1. 注入锚点（用精确锚点，避免被内嵌源码里的同名字面量误导）
    chk(re.search(r"<style>\s*:root\{", doc) is not None,
        "CSS 已注入（<style> 后紧跟 :root{）")
    chk(re.search(r"</div><script>\s*\nconst TREE", doc) is not None,
        "JS 已注入（</div><script> 后紧跟 const TREE）")
    for anchor in ("<style>__CSS__", "<script>__JS__",
                   "const TREE = __TREE__", "const FILES = __FILES__"):
        c = doc.count(anchor)
        chk(c <= 1, "真实占位符已替换：%s" % anchor,
            "剩余 %d 处（>1 说明未替换）" % c)

    # 2. 结构
    chk(doc.count("</script>") == 3,
        "script 闭合标签数正确（MathJax 配置 + CDN + 主脚本）",
        "close=%d" % doc.count("</script>"))
    chk(doc.rstrip().endswith("</html>"), "文档以 </html> 正常结尾")
    for frag in ('<div id="app">', 'id="tree"', 'id="body"', 'id="q"', 'id="bar"'):
        chk(frag in doc, "骨架节点存在：%s" % frag)

    # 3. 内嵌数据可解析
    files = None
    m = re.search(r"const FILES = (\[.*?\]);\n", doc, re.S)
    if not m:
        chk(False, "定位到 FILES 数组")
    else:
        try:
            files = json.loads(m.group(1).replace("<\\/", "</"))
            chk(True, "FILES 可解析", "%d 个文件" % len(files))
        except Exception as ex:
            chk(False, "FILES 可解析", str(ex)[:100])
    m2 = re.search(r"const TREE = (\[.*?\]);\nconst FILES", doc, re.S)
    if m2:
        try:
            tree = json.loads(m2.group(1).replace("<\\/", "</"))
            chk(True, "TREE 可解析",
                "%d 组 / %d 项" % (len(tree), sum(len(g["items"]) for g in tree)))
        except Exception as ex:
            chk(False, "TREE 可解析", str(ex)[:100])

    # 4. 内容抽样
    if files:
        empty = [f["path"] for f in files
                 if not f["text"] and f["kind"] not in ("png", "json")]
        chk(not empty, "无空内容条目（图片/JSON 除外）", str(empty)[:80])
        kinds = {}
        for f in files:
            kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
        chk(True, "类型分布", str(kinds))
        tot = sum(len(f["text"]) for f in files)
        chk(tot > 0, "内嵌正文合计", "%.0f KB" % (tot / 1024))
        for f in files:
            if f["path"].endswith(".csv"):
                chk(f["text"].startswith("| "),
                    "CSV 已转 Markdown 表格：%s" % f["path"])

    print("=" * 64)
    print("  静态质检：" + ("通过" if ok else "有问题"))
    print("=" * 64)

    # 5. 运行时（Node）
    if a.static:
        print("\n  （--static：跳过运行时质检）")
    else:
        node = shutil.which("node")
        script = os.path.join(HERE, "check_render.js")
        if not node:
            print("\n  ! 未找到 node，跳过运行时质检（建议安装后重跑）")
        elif not os.path.exists(script):
            print("\n  ! 未找到 check_render.js，跳过运行时质检")
        else:
            print()
            r = subprocess.run([node, script, p], capture_output=True, text=True)
            print(r.stdout.rstrip())
            if r.stderr.strip():
                print("  [stderr] " + r.stderr.strip()[:400])
            if r.returncode != 0:
                ok_local = False
                globals()["ok"] = False
                fails.append("运行时质检")

    print()
    print("#" * 64)
    if ok:
        print("  总检通过 ✅  " + os.path.basename(p))
    else:
        print("  存在问题 ❌  %d 项：" % len(fails))
        for f in fails:
            print("    - " + f)
    print("#" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
