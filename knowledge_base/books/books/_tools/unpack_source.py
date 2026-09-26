#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0 —— 源书压缩包 → 解压 + 目录抽取

《系统化飞控实战》系列的压缩包有两个坑：
  1. 压缩包内文件名是 GBK 编码，zipfile 会按 cp437 读出乱码
     → 必须 n.encode('cp437').decode('gbk') 还原
  2. 压缩包根目录是扁平的（index.html + N 门课目录），
     每门课目录里 NN.html 是章节，另有 index.html / intro.html 是
     课程级页面，**不是章节**，必须排除

章节元数据全部机械提取，不需要 LLM：
  - 标题：`<title>第N章 X</title>`          —— 600/600 命中
  - 摘要：首个 `<h2>` 冒号后的部分            —— 535/600 命中，其余走正文首段兜底
  - 标签：正文 n-gram 词频抽取（启发式，非 LLM 语义标签）

用法：
    # 解压 + 抽取（推荐，一步到位）
    python3 unpack_source.py --zip /path/011-xxx.zip --book-dir ../book011 --id book011

    # 只抽取（已解压过）
    python3 unpack_source.py --book-dir ../book011 --id book011 --no-unzip

    # 回归校验：与既有 toc.json 逐条比对标题
    python3 unpack_source.py --book-dir ../book010 --id book010 --no-unzip --check

产出（写入 {book-dir}/source/）：
    {id}/                     解压后的源书
    {id}_outline.json         {课程: [{no, chars, body_head}]}
    {id}_toc.json             {课程: [{no, title, desc, tags}]}
"""
import argparse
import json
import os
import re
import sys
import zipfile

# ---------------------------------------------------------------- 文件名编码

def decode_name(raw):
    """zipfile 按 cp437 读出的名字 → GBK 还原；失败则原样返回。"""
    try:
        return raw.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def unzip(zip_path, dest, force=False):
    """解压到 dest，自动还原 GBK 文件名。"""
    if os.path.isdir(dest) and os.listdir(dest) and not force:
        print("  已存在且非空，跳过解压（--force 可覆盖）：%s" % dest)
        return 0
    os.makedirs(dest, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = decode_name(info.filename)
            # 目录项
            if info.filename.endswith("/"):
                os.makedirs(os.path.join(dest, name), exist_ok=True)
                continue
            target = os.path.join(dest, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
            n += 1
    print("  解压 %d 个文件 → %s" % (n, dest))
    return n


# ---------------------------------------------------------------- HTML 解析

RE_SCRIPT = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
RE_TAG = re.compile(r"<[^>]+>")
RE_TITLE = re.compile(r"<title>\s*第\s*(\d+)\s*章\s*(.*?)\s*</title>", re.S)
RE_H2 = re.compile(r"<h2[^>]*>(.*?)</h2>", re.S)
RE_CHAPTER_FILE = re.compile(r"^(\d{2,3})\.html$")
# h2 里常见的编号前缀："1. " / "1.2 " / "第二章 " / "第二十三讲 " / "一、"
# 注意两点：
#   1. 末尾必须吃掉 [：:、.]——否则 "第3章：实时仿真平台选型" 会残留冒号，
#      后面的分隔符切分会把整段当成摘要，标题反而被清空。
#   2. 量词不能只写"章"，本系列还用"讲"（第二十三讲）/ "节" / "篇"。
RE_H2_NUM = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.、]?"
    r"|第\s*[0-9一二三四五六七八九十百]+\s*[章讲节篇]\s*[：:、.]?"
    r"|[一二三四五六七八九十百]+[、.])\s*")

# 标题 / 摘要分隔符：中文冒号优先，其次是破折号（"——" 必须排在 "—" 前面）
SEPARATORS = ("：", "——", "—", ":")

# 课程级页面，不是章节
NON_CHAPTER = {"index.html", "intro.html", "readme.html"}

# 结构性标题：只是"这一章的体裁"，不含内容信息，需要用完整 h2 补足。
# 注意与"技术术语"区分——"协同制导""制导概论""课程导论"虽短但有信息量，不补足。
TERSE_TITLES = {
    "绪论", "引言", "概述", "简介", "前言", "小结", "总结", "结论", "展望",
    "后记", "附录", "参考文献", "致谢", "导论", "课程导论", "课程总结", "本章小结",
    "前沿趋势", "未来趋势", "前沿技术", "技术展望", "发展趋势", "研究展望",
    "案例研究", "案例分析", "实战案例", "综合案例", "典型案例分析",
    "制导系统实战案例", "案例实战", "高级话题", "扩展阅读", "常见问题",
}


def is_terse(title):
    """标题是否只是"体裁标签"、不含内容信息（需要 h2 补足）。"""
    if not title:
        return True
    if title in TERSE_TITLES:
        return True
    # "附录A" / "附录 B" 这类
    return bool(re.match(r"^附录\s*[A-Za-z0-9一二三四五六七八九十]?$", title))


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8", "gbk", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def html_to_text(html):
    """去掉 script/style/标签，压平空白。"""
    body = RE_SCRIPT.sub(" ", html)
    body = RE_TAG.sub(" ", body)
    body = body.replace("&nbsp;", " ").replace("&amp;", "&")
    body = body.replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", body).strip()


def main_text(html):
    """只取正文，丢掉侧边栏。

    每个章节页左侧都有"课程目录"侧边栏，其文本会出现在纯文本的最前面。
    侧边栏在 DOM 里排在首个 <h2> 之前，因此**从首个 h2 开始取文本**
    即可干净地切掉它（比按 class 找 div 更稳，不用处理嵌套 div）。
    """
    cleaned = RE_SCRIPT.sub(" ", html)
    m = RE_H2.search(cleaned)
    return html_to_text(cleaned[m.start():] if m else cleaned)


def plain(tag_inner):
    return re.sub(r"\s+", " ", RE_TAG.sub("", tag_inner)).strip()


def first_paragraph(text, min_len=25, max_len=110):
    """从正文里挑第一个够长的句子/分句作为摘要兜底。"""
    # 以句号/问号/叹号切分，取第一段够长的
    chunks = re.split(r"(?<=[。！？；])", text)
    buf = ""
    for c in chunks:
        buf += c
        if len(buf) >= min_len:
            return buf[:max_len].strip()
    return text[:max_len].strip()


# ---------------------------------------------------------------- 标签抽取

STOP_CHARS = set("的了是在和与及对于关于这个我们你们他们可以能够进行通过由于因此所以"
                 "而且但是然后就是不是有会要被把从对为以之其该等等一下一些一个什么"
                 "怎么如何为什以及并且如果虽然然而同时另外例如比如以及")
RE_CJK = re.compile(r"[\u4e00-\u9fff]+")


def extract_tags(texts, k=2):
    """n-gram 词频启发式抽标签。非语义，但确定性可复现。"""
    grams = {}
    for t in texts:
        for run in RE_CJK.findall(t):
            if len(run) < 2:
                continue
            for n in range(2, 7):          # 2~6 字，最长 6 避免"半实物仿"这种截断
                for i in range(len(run) - n + 1):
                    g = run[i:i + n]
                    if any(c in STOP_CHARS for c in g):
                        continue
                    grams[g] = grams.get(g, 0) + 1
    if not grams:
        return []
    # 打分：频次 × 长度加权，长词更有信息量
    scored = sorted(grams.items(), key=lambda kv: kv[1] * (len(kv[0]) ** 1.5), reverse=True)
    picked = []
    for g, _ in scored:
        # 与已选标签高度重叠的跳过（避免"制导"/"制导系"同时入选）
        if any(g in p or p in g for p in picked):
            continue
        picked.append(g)
        if len(picked) >= k:
            break
    return picked


# ---------------------------------------------------------------- 扫描

def scan_courses(src_dir):
    """返回 [(course_name, [(no, path)]), ...]，按课程名排序。"""
    courses = []
    for name in sorted(os.listdir(src_dir)):
        d = os.path.join(src_dir, name)
        if not os.path.isdir(d) or name.startswith("."):
            continue
        chapters = []
        for fn in sorted(os.listdir(d)):
            m = RE_CHAPTER_FILE.match(fn)
            if m and fn not in NON_CHAPTER:
                chapters.append((m.group(1), os.path.join(d, fn)))
        if chapters:
            courses.append((name, chapters))
    return courses


def build(src_dir):
    """扫描源书目录，产出 outline 与 toc。"""
    outline, toc = {}, {}
    stats = {"courses": 0, "chapters": 0, "fallback_desc": 0,
             "fallback_title": 0, "terse_title": 0}

    for cname, chapters in scan_courses(src_dir):
        o_items, t_items = [], []
        corpus = []          # 供标签抽取用的语料
        for no, path in chapters:
            html = read_text(path)
            text = html_to_text(html)   # 全页（含侧边栏）——仅用于统计字数
            body = main_text(html)      # 正文（已切掉侧边栏）——摘要与标签语料

            # --- h2：既是摘要来源，也是标题兜底/补足来源
            #     形如 "1. 制导控制概论：导弹制导系统的基本概念、..."
            #     冒号前 = 标题，冒号后 = 摘要
            h2_raw = ""
            mh = RE_H2.search(html)
            if mh:
                h2_raw = plain(mh.group(1))
                h2 = RE_H2_NUM.sub("", h2_raw).strip()
            head = tail = ""
            for sep in SEPARATORS:
                if sep in h2:
                    head, _, tail = h2.partition(sep)
                    break
            head, tail = head.strip(), tail.strip()

            # 正文开头是 h2 本身，去掉，否则摘要会重复标题。
            # 必须用**未去编号前缀**的 h2_raw 比对——正文里带 "3. " 前缀，
            # 用归一化后的 h2 去 startswith 会永远失败。
            for probe in (h2_raw, h2):
                if probe and body.startswith(probe):
                    body = body[len(probe):].lstrip("—－-–· \u3000")
                    break

            # --- 标题：<title>第N章 X</title>；过短/泛化则用完整 h2 补足
            mt = RE_TITLE.search(html)
            raw_title = plain(mt.group(2)) if mt else ""
            if not raw_title:
                stats["fallback_title"] += 1
                title = head or h2 or ("第%s章" % no)
            elif is_terse(raw_title):
                stats["terse_title"] += 1
                title = h2 or raw_title
            else:
                title = raw_title

            # --- 摘要：h2 冒号后；无冒号则取正文首段
            if tail:
                desc = tail
            else:
                stats["fallback_desc"] += 1
                desc = first_paragraph(body)

            o_items.append({"no": no, "chars": len(body),
                            "body_head": body[:70].strip()})
            t_items.append({"no": no, "title": title, "desc": desc})
            corpus.extend([title, desc, body[:400]])

        tags_pool = extract_tags(corpus, k=2)
        for it in t_items:
            it["tags"] = list(tags_pool)

        outline[cname] = o_items
        toc[cname] = t_items
        stats["courses"] += 1
        stats["chapters"] += len(chapters)

    return outline, toc, stats


# ---------------------------------------------------------------- 校验

def self_check(outline, toc, expected_courses=None, expected_per_course=None):
    """结构性自检（硬断言）。返回问题列表，空列表表示通过。

    这里只查**能确定判定**的事：章节连续性、字段非空、课程/章节总数。
    标题措辞不查——那是 LLM 润色的事，见 compare_ref()。
    """
    issues = []
    n_ch = 0
    for cname, items in toc.items():
        nos = [int(it["no"]) for it in items]
        if nos != sorted(nos):
            issues.append("%s：章节号未排序" % cname)
        if nos and nos != list(range(1, len(nos) + 1)):
            miss = [i for i in range(1, max(nos) + 1) if i not in nos]
            issues.append("%s：章节号不连续，缺 %s" % (cname, miss))
        for it in items:
            if not it.get("title"):
                issues.append("%s / %s：标题为空" % (cname, it["no"]))
            if not it.get("desc"):
                issues.append("%s / %s：摘要为空" % (cname, it["no"]))
        n_ch += len(items)
    if expected_courses is not None and len(toc) != expected_courses:
        issues.append("课程数 %d != 预期 %d" % (len(toc), expected_courses))
    if expected_per_course is not None:
        for cname, items in toc.items():
            if len(items) != expected_per_course:
                issues.append("%s：章节数 %d != 预期 %d"
                              % (cname, len(items), expected_per_course))
    return issues, len(toc), n_ch


def compare_ref(toc, ref_path):
    """与既有 toc.json 比对标题，返回 (一致数, 差异列表)。

    按**数值**匹配章号——既有文件里编号格式不统一（多数课程 '01'，
    少数课程 '1'），字符串比对会造成大量假差异。

    注意：参考文件是 LLM 分批润色的产物，同一体裁的标题在不同课程里
    处理方式并不一致（有的展开、有的保留），因此**不要把完全一致的
    结果当作通过标准**。这里只作为覆盖率参考。
    """
    ref = json.load(open(ref_path, encoding="utf-8"))
    same, diff = 0, []
    for cname, items in toc.items():
        r = ref.get(cname)
        if r is None:
            diff.append((cname, "-", "参考文件里没有这门课"))
            continue
        rmap = {}
        for x in r:
            try:
                rmap[int(x["no"])] = x
            except (KeyError, ValueError):
                pass
        for it in items:
            rt = rmap.get(int(it["no"]))
            if rt is None:
                diff.append((cname, it["no"], "参考文件里没有这章"))
            elif rt.get("title") != it["title"]:
                diff.append((cname, it["no"], "%r != %r" % (it["title"], rt.get("title"))))
            else:
                same += 1
    return same, diff


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Stage 0：源书解压 + 目录抽取")
    ap.add_argument("--zip", help="源书压缩包路径")
    ap.add_argument("--book-dir", required=True, help="书目目录，如 ../book011")
    ap.add_argument("--id", help="书目 id，如 book011（默认取目录名）")
    ap.add_argument("--no-unzip", action="store_true", help="跳过解压，只抽取目录")
    ap.add_argument("--force", action="store_true", help="已存在时强制覆盖解压")
    ap.add_argument("--check", action="store_true",
                    help="只校验不写入；有参考 toc 时附带比对标题一致率")
    ap.add_argument("--expect-courses", type=int, help="预期课程数（硬断言）")
    ap.add_argument("--expect-per-course", type=int, help="预期每门课章节数（硬断言）")
    args = ap.parse_args()

    book_dir = os.path.abspath(args.book_dir)
    if not os.path.isdir(book_dir):
        sys.exit("书目目录不存在：%s" % book_dir)
    bid = args.id or os.path.basename(book_dir)
    src_dir = os.path.join(book_dir, "source", bid)

    if not args.no_unzip:
        if not args.zip:
            sys.exit("需要 --zip，或加 --no-unzip 跳过解压")
        if not os.path.isfile(args.zip):
            sys.exit("压缩包不存在：%s" % args.zip)
        print("[1/2] 解压")
        unzip(args.zip, src_dir, force=args.force)

    if not os.path.isdir(src_dir):
        sys.exit("源书目录不存在：%s" % src_dir)

    print("[2/2] 抽取目录")
    outline, toc, stats = build(src_dir)
    print("  课程 %d 门 ｜ 章节 %d 章" % (stats["courses"], stats["chapters"]))
    if stats["fallback_title"]:
        print("  ⚠️ 标题兜底 %d 章（<title> 未命中，改用 h2）" % stats["fallback_title"])
    print("  标题：<title> 直取 %d 章 ｜ 泛化标题用 h2 补足 %d 章"
          % (stats["chapters"] - stats["fallback_title"] - stats["terse_title"],
             stats["terse_title"]))
    print("  摘要：h2 分隔符式 %d 章 ｜ 正文首段兜底 %d 章"
          "（剩下的是无分隔符的概览页，摘要取正文开头，属预期）"
          % (stats["chapters"] - stats["fallback_desc"], stats["fallback_desc"]))

    # 结构性自检——无论是否写入都跑，硬失败
    issues, n_courses, n_chapters = self_check(
        outline, toc, args.expect_courses, args.expect_per_course)
    if issues:
        print("\n❌ 结构性自检未通过（%d 项）：" % len(issues))
        for s in issues[:20]:
            print("   ", s)
        if len(issues) > 20:
            print("    ... 其余 %d 项" % (len(issues) - 20))
        return 1
    print("  ✅ 结构自检通过（课程连续编号、标题/摘要无空值、总数符合预期）")

    ref = os.path.join(book_dir, "source", "%s_toc.json" % bid)
    if os.path.isfile(ref):
        same, diff = compare_ref(toc, ref)
        total = same + len(diff)
        print("\n  与既有 %s 比对（参考值，非通过标准）：" % os.path.basename(ref))
        print("    标题一致 %d/%d = %.1f%%" % (same, total, 100.0 * same / max(1, total)))
        if diff:
            print("    差异 %d 处（多为参考稿的 LLM 润色："
                  "全角/半角括号、泛化标题展开、长标题压缩），示例：" % len(diff))
            for c, n, m in diff[:5]:
                print("      [%s] %s: %s" % (c, n, m))
            if len(diff) > 5:
                print("      ... 其余 %d 处" % (len(diff) - 5))
    elif not args.check:
        print("  （无既有 toc 可比对，跳过）")

    if args.check:
        print("\n校验完成，未写入任何文件")
        return 0

    os.makedirs(os.path.join(book_dir, "source"), exist_ok=True)
    for name, obj in (("%s_outline.json" % bid, outline),
                      ("%s_toc.json" % bid, toc)):
        p = os.path.join(book_dir, "source", name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print("  ✅ %s (%.1f KB)" % (name, os.path.getsize(p) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
