#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键流水线 —— 从源书压缩包到可交付的单文件资料库

把 _tools/ 下的四个脚本串起来。每本书只跑这一条命令：

    # 从压缩包开始（全新书目）
    python3 pipeline.py --zip /path/011-xxx.zip --book-dir ../book011 \\
                        --expect-courses 20 --expect-per-course 30

    # 已完成蒸馏，只要资产化（Stage 1~3）
    python3 pipeline.py --book-dir ../book011

    # 只重跑打包 + 质检（改了卡片内容后最常见）
    python3 pipeline.py --book-dir ../book011 --only 2,3

    # 附带跑算法库自检
    python3 pipeline.py --book-dir ../book011 --verify

阶段：
    0  解压 + 目录抽取   unpack_source.py   （需 --zip）
    1  资产抽取         extract_assets.py  （需 domain_*.md 已蒸馏完成）
    2  资料库打包       build_library.py
    3  资料库质检       check_library.py
    V  算法库自检       algorithms/run_all.py（--verify，若存在）

设计约定：
  - 任一阶段失败立即停止，退出码非 0（适合 CI 或批量脚本）
  - 阶段 0 只在给了 --zip 时运行；没有 zip 就从阶段 1 开始
  - 阶段 3 的输入文件由阶段 2 的默认命名规则推算：<book-dir>/<id>_library.html
  - --only / --skip 用逗号分隔的阶段号，如 --only 2,3
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STAGES = [
    (0, "解压 + 目录抽取", "unpack_source.py"),
    (1, "资产抽取", "extract_assets.py"),
    (2, "资料库打包", "build_library.py"),
    (3, "资料库质检", "check_library.py"),
]


def resolve_html_path(book_dir, bid, cli_out=None):
    """推算资料库 HTML 的路径。

    优先级：--html-out > library.json 的 "out" > <id>_library.html。
    library.json 这一档是为了 book010——它的既有交付叫 guidance_library.html，
    已被 README 与索引多处引用，改名会破坏既有链接。
    """
    if cli_out:
        return os.path.abspath(cli_out)
    cfg_path = os.path.join(book_dir, "library.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                out = json.load(f).get("out")
            if out:
                return os.path.join(book_dir, out)
        except (ValueError, OSError) as e:
            print("  ⚠️ library.json 读取失败，按默认命名：%s" % e)
    return os.path.join(book_dir, "%s_library.html" % bid)


def run(script, args, label, timeout=3600):
    """跑一个子脚本，返回 (ok, 耗时秒)。输出原样透传。"""
    cmd = [PY, os.path.join(HERE, script)] + args
    print("\n" + "─" * 66)
    print("▶ %s\n  $ %s %s" % (label, script, " ".join(args)))
    print("─" * 66)
    # 必须 flush：stdout 是管道时父进程块缓冲，不刷会与子进程输出乱序
    sys.stdout.flush()
    t0 = time.time()
    try:
        rc = subprocess.call(cmd, cwd=HERE, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("  ✗ 超时（%d 秒）" % timeout)
        return False, time.time() - t0
    dt = time.time() - t0
    sys.stdout.flush()
    print("─" * 66)
    if rc == 0:
        print("✅ %s 通过（%.1fs）" % (label, dt))
    else:
        print("❌ %s 失败（退出码 %d，%.1fs）" % (label, rc, dt))
    return rc == 0, dt


def main():
    ap = argparse.ArgumentParser(
        description="一键流水线：源书 → 数据资产 → 单文件资料库 → 质检",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # 注意：--book-dir 不能在 add_argument 里设 required，否则 --list 也会被它挡住
    ap.add_argument("--book-dir", help="书目目录，如 ../book011")
    ap.add_argument("--zip", help="源书压缩包（给了才跑阶段 0）")
    ap.add_argument("--id", help="书目 id，默认取目录名")
    ap.add_argument("--only", help="只跑指定阶段，逗号分隔，如 2,3")
    ap.add_argument("--skip", help="跳过指定阶段，逗号分隔，如 0")
    ap.add_argument("--expect-courses", type=int, help="阶段 0 硬断言：课程数")
    ap.add_argument("--expect-per-course", type=int, help="阶段 0 硬断言：每门课章节数")
    ap.add_argument("--verify", action="store_true", help="额外跑 algorithms/run_all.py")
    ap.add_argument("--force", action="store_true", help="阶段 0 已存在时强制覆盖解压")
    ap.add_argument("--html-out", help="资料库输出文件名（默认 <id>_library.html）")
    ap.add_argument("--list", action="store_true", help="列出阶段后退出")
    args = ap.parse_args()

    if args.list:
        print("阶段清单：")
        for no, name, script in STAGES:
            print("  %d  %-16s %s" % (no, name, script))
        print("  V  算法库自检       algorithms/run_all.py（--verify）")
        return 0

    book_dir = os.path.abspath(args.book_dir) if args.book_dir else None
    if book_dir is None:
        sys.exit("需要 --book-dir（或 --list 查看阶段清单）")
    if not os.path.isdir(book_dir):
        sys.exit("书目目录不存在：%s" % book_dir)
    bid = args.id or os.path.basename(book_dir)
    html_path = resolve_html_path(book_dir, bid, args.html_out)

    def parse_sel(s):
        """解析阶段选择。数字或 V（算法库自检），如 "2,3" / "V"。"""
        if not s:
            return None
        out = set()
        for x in s.split(","):
            x = x.strip().upper()
            if not x:
                continue
            out.add(x if x == "V" else int(x))
        return out

    only, skip = parse_sel(args.only), parse_sel(args.skip)

    print("=" * 66)
    print("  一键流水线 ｜ %s ｜ id=%s" % (os.path.basename(book_dir), bid))
    print("=" * 66)

    results = []
    for no, name, script in STAGES:
        if only is not None and no not in only:
            continue
        if skip is not None and no in skip:
            print("\n⏭  跳过阶段 %d：%s" % (no, name))
            continue

        if no == 0:
            if not args.zip:
                print("\n⏭  跳过阶段 0：未给 --zip（如需解压请补上）")
                continue
            a = ["--zip", args.zip, "--book-dir", book_dir, "--id", bid]
            if args.expect_courses:
                a += ["--expect-courses", str(args.expect_courses)]
            if args.expect_per_course:
                a += ["--expect-per-course", str(args.expect_per_course)]
            if args.force:
                a.append("--force")
        elif no == 1:
            # --id 必须透传：子脚本默认从目录名推 id，
            # 目录名 ≠ id 时（如解压到 /tmp/bkfull 但 id 是 book010）
            # 会导致阶段 2 写出的文件名与阶段 3 要找的对不上
            a = ["--book-dir", book_dir, "--id", bid]
        elif no == 2:
            a = ["--book-dir", book_dir, "--id", bid]
            if args.html_out:
                a += ["--out", os.path.abspath(args.html_out)]
        else:  # no == 3
            if not os.path.isfile(html_path):
                print("\n❌ 阶段 3 找不到待检文件：%s" % html_path)
                print("   先跑阶段 2，或用 --id 指定正确的书目 id")
                results.append((no, name, False, 0.0))
                break
            a = [html_path]

        ok, dt = run(script, a, "阶段 %d · %s" % (no, name))
        results.append((no, name, ok, dt))
        if not ok:
            break

    # ---- 可选：算法库自检（V 阶段）
    want_v = args.verify or (only is not None and "V" in only)
    if skip is not None and "V" in skip:
        want_v = False
    if want_v:
        ra = os.path.join(book_dir, "algorithms", "run_all.py")
        if os.path.isfile(ra):
            print("\n" + "─" * 66)
            print("▶ 阶段 V · 算法库自检\n  $ %s" % os.path.relpath(ra, book_dir))
            print("─" * 66)
            sys.stdout.flush()
            t0 = time.time()
            # 用书目目录作 cwd，与 README 里的手工用法保持一致
            # （run_all.py 内部按 __file__ 定位模块，两种 cwd 都能跑，
            #   统一到 book_dir 是为了让手工跑和流水线跑的行为完全一致）
            rc = subprocess.call([PY, ra], cwd=book_dir, timeout=3600)
            dt = time.time() - t0
            sys.stdout.flush()
            print("─" * 66)
            print("%s 阶段 V · 算法库自检（%.1fs）" % ("✅" if rc == 0 else "❌", dt))
            results.append(("V", "算法库自检", rc == 0, dt))
            if rc != 0:
                print("\n❌ 流水线未跑完，停在：算法库自检")
                return 1
        else:
            print("\n⏭  跳过算法库自检：%s 不存在" % os.path.relpath(ra, book_dir))

    # ---- 汇总
    print("\n" + "=" * 66)
    print("  流水线汇总")
    print("=" * 66)
    for no, name, ok, dt in results:
        print("  %-3s %-16s %s  %6.1fs" % (no, name, "✅ 通过" if ok else "❌ 失败", dt))
    total = sum(r[3] for r in results)
    failed = [r for r in results if not r[2]]
    print("  " + "─" * 60)
    print("  合计 %.1fs ｜ %d 项完成 ｜ %d 项失败" % (total, len(results) - len(failed), len(failed)))

    if failed:
        print("\n❌ 流水线未跑完，停在：%s" % failed[0][1])
        return 1

    print("\n✅ 全部通过")
    if any(r[0] == 2 for r in results):
        print("   资料库：%s (%.0f KB)"
              % (html_path, os.path.getsize(html_path) / 1024
                 if os.path.isfile(html_path) else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
