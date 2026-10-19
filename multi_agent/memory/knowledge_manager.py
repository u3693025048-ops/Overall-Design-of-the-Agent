#!/usr/bin/env python3
"""
知识库管理 + RAG（按《需求架构与工作任务理解》2.0 业务框架设计实现）

设计要点（与文档 §2.7 / §5.5 / §6.2 对齐）：
  - 四库：模型库(model) / 模板库(template) / 经验库(experience) / 规范库(norm)
  - 检索：向量(Chroma) + BM25 + RRF 融合 → 按「设计层级 level + 工况 case + 库 lib」硬过滤
  - 已知路径直达：传入 doc_id 时直接 get_document，不走语义检索（禁止每轮搜固定文件）
  - 回流：成功升 confidence，失败降级/标 failed；失败案例同等归档
  - 默认产物：next_action（可机检下一动作），而非长文

说明：本模块**复用**已有的 `RAGKnowledgeBase`（Chroma 向量后端），在其之上叠加
混合检索与四库编排，不重写底层向量存储，向后兼容 rag_tool / cli_agent。

注：参考架构文档用「特征点 P1~P4」描述工况；本工程实际采用 monte_carlo_single.m 的
RUN_CASE 体系（T / G / AP / R），故 CASES 以 T/G/AP/R/ALL 为准，P1~P4 仅作语义对应。
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .rag_knowledge_base import RAGKnowledgeBase
from .kb_ledger import KBLedger
# P0-3：动作词表的**单一权威**在 action_vocab（零重依赖）。这里只做再导出，
# 不再自行定义枚举 —— 防止"三份定义各自漂移"（见 action_vocab 模块 docstring）。
from .action_vocab import (  # noqa: F401  (再导出：scripts/ 与验收脚本按旧名引用)
    ACTION_CANON,
    ACTION_DISPLAY_ALIASES,
    CONVENTIONS as ACTION_CONVENTIONS,
    KB_FOUR_LIB_ACTIONS,
    KB_FOUR_LIB_CONVENTION,
    POLICY_ACTIONS,
    POLICY_CONVENTION,
    canonicalize as canonicalize_action,
    convention_of as action_convention_of,
    is_valid as is_valid_action,
    resolve_action,
    to_convention as action_to_convention,
)

logger = logging.getLogger(__name__)


# ── 四库 / 层级 / 工况 字典 ────────────────────────────────────────────────
LIBS = ("norm", "model", "template", "experience")          # 规范/模型/模板/经验
LEVELS = ("总体", "几何-气动", "内回路", "外回路", "验证")      # 设计层级（手册第八步）
CASES = ("T", "G", "AP", "R", "ALL")                          # 工况（RUN_CASE 体系）
# 中文展示别名（语料 metadata 的 ``action`` 字段口径）。规范值见 next_action_enum。
NEXT_ACTIONS = tuple(ACTION_CANON)

# 层级别名（A9/A3：口语/论文口径 → 内部层级）。autopilot/inner→内回路；guidance→外回路。
LEVEL_ALIASES = {
    "autopilot": "内回路", "inner": "内回路", "内回路": "内回路",
    "guidance": "外回路", "outer": "外回路", "外回路": "外回路",
    "geo": "几何-气动", "aero": "几何-气动", "几何-气动": "几何-气动",
    "overall": "总体", "总体": "总体",
    "verif": "验证", "验证": "验证",
}
# 四枚举规范动作（建设方案 05 B2/02 口径）：经验卡的 next_action 只取这四值。
# P0-3：值本身来自 action_vocab.KB_FOUR_LIB_ACTIONS（单一权威），此处仅提供旧名别名。
ACTION_ENUMS = KB_FOUR_LIB_ACTIONS

# ── P0-1 门下省：复用时经验卡适用性闸门 ─────────────────────────────────────
# 三档：
#   active                   → 无条件可用（已人审 + 已签字 + 真值已证实）
#   candidate / 其它未审态    → 可用，但标记 review_required，须人审确认适用性
#   rejected / failed        → 硬阻断，不得驱动复用（"失败不封杀检索"≠可驱动动作）
# 冻结金标卡（E001/E002）以 candidate 入库，故不会因闸门而"提前作废"，
# 但任何 rejected/failed 的经验卡从此再也不能静默驱动 next_action。
REUSE_BLOCKED_STATUS = ("rejected",)

# ── P0-2 失败案例归档：failed_under 是「绑工况」的对象，不是字符串 ──────────────
# 规范依据（05 §B4.3 / §3.3 / §B4.4 阶段⑧）：
#   * failed_under 是**对象**，至少 scenario + R0_hash（可选违规指标）；
#   * 新任务工况对不上 → 该失败的惩罚权重为 **0**（"E001 在 P4 失败，不抹掉它
#     在 P1 的经验"）——失败**不得**表达成全局 invalid=true；
#   * 同一张卡可累积多次失败，故本工程接受「对象或对象数组」，内部统一归一为 list。
# 修复前的实现把 failed_under 当字符串且**没有任何写入者**，schema 也就被写成了
# `type: string` —— 三处一起错。这里给出唯一的解析入口。
FAILED_UNDER_REQUIRED = ("scenario", "R0_hash")

# 工况别名：架构文档写 P1~P4，本工程跑次写 RUN_CASE(T/G/AP/R)（见 03_norm/cases-p1-p4.json）。
# 失败绑工况的比对必须两边都能认，否则 "E001 在 P4 失败" 与 scenario='R' 会被判成不匹配。
CASE_ALIASES = {"P1": "T", "P2": "G", "P3": "AP", "P4": "R"}


def normalize_case(value: Any) -> str:
    """工况归一到大写 RUN_CASE 口径；未知值原样大写返回（如 'ALL'）。"""
    s = str(value or "").strip().upper()
    return CASE_ALIASES.get(s, s)



def normalize_failed_under(value: Any) -> List[Dict[str, Any]]:
    """把 failed_under 的任意合法写法归一成 list[dict]（只保留合法条目）。

    接受：``None`` / ``{}`` / 单个对象 / 对象数组 / 上述形态的 JSON 字符串
    （metadata 里为兼容 chromadb 标量约束，存的是 JSON 串）。
    非法条目（缺 scenario 或 R0_hash、非 dict）**丢弃**，绝不臆造。
    """
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: List[Dict[str, Any]] = []
    for ent in value:
        if not isinstance(ent, dict):
            continue
        if not all(str(ent.get(k) or "").strip() for k in FAILED_UNDER_REQUIRED):
            continue
        out.append(dict(ent))
    return out


def failed_under_scenarios(value: Any) -> List[str]:
    """该卡在哪些工况下失败过（归一到大写 RUN_CASE 口径，去重保序）。

    必须归一：规范正文用 P1~P4 讲工况（"E001 在 P4 失败"），而跑次写 RUN_CASE。
    不归一的话 P4 与 R 会被判成两个不相干的工况，失败就白绑了。
    """
    seen: List[str] = []
    for ent in normalize_failed_under(value):
        sc = normalize_case(ent.get("scenario"))
        if sc and sc not in seen:
            seen.append(sc)
    return seen


def dumps_failed_under(value: Any) -> str:
    """metadata 里的存法：JSON 字符串。

    chromadb 只吃标量，而 BM25 语料是 JSONL。两边都存**同一个 JSON 串**，
    才能让 `check_index_metadata_sync.py` 的漂移断言把 failed_under 也纳入比对。
    """
    ents = normalize_failed_under(value)
    return json.dumps(ents, ensure_ascii=False) if ents else ""


# 经验卡 mode → 设计层级（用于把参数经验打到正确层级）
MODE_LEVEL = {
    "MODIFY_LAW": "外回路",
    "MODIFY_AUTOPILOT": "内回路",
    "MODIFY_CONTROLLER": "内回路",
    "MODIFY_GEOMETRY": "几何-气动",
    "TUNE": "内回路",
}


# ── doc_id 归一（05 建设方案 B4.1 第 2125 行约定）──────────────────────────
# 建设方案约定的对外 doc_id 形如：norm:units.json / experience:E001 /
# template:opt-skeleton / model:matlab/pn（B4.1 原文）——"直达失败返回未知 id，
# 不得用最相似的另一篇顶替"。
# 本工程索引主键为 kb:{lib}:{绝对路径}（经验卡为 kb:experience:{card_id}）。
# canonical 是"对外契约 id"，内部键是"索引主键"；本层负责两者互转，
# 使 B13 A1 的字面输入 known_doc_ids=["norm:units.json"] 真正直达命中。
CANONICAL_LIBS = LIBS  # ("norm", "model", "template", "experience")

# canonical 去扩展名比较时认识的扩展名（05 B4.1 的 template:opt-skeleton 即去 .json）
_DOC_EXTS = (".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".py", ".m", ".xml", ".mat")

# canonical 生成时用于剥离出「库内相对路径」的目录标记（按优先级）
_CANON_MARKERS = (
    "/deliverables/03_norm/",
    "/deliverables/05_models/",
    "/deliverables/06_canvas/ir/_templates/",
    "/knowledge_base/",
)


def _stem(name: str) -> str:
    low = name.lower()
    for ext in _DOC_EXTS:
        if low.endswith(ext):
            return name[: len(name) - len(ext)]
    return name


def parse_canonical(doc_id: str):
    """canonical doc_id → (lib, tail)；非 canonical 形式返回 None。

    已带 kb:/file: 前缀的内部键不视作 canonical（避免二次解读）；含 `..` 的
    尾段一律拒绝（canonical 只允许库内相对引用）。
    """
    if not doc_id:
        return None
    s = str(doc_id).strip().replace("\\", "/")
    if s.startswith("kb:") or s.startswith("file:"):
        return None
    if ":" not in s:
        return None
    lib, _, tail = s.partition(":")
    lib = lib.strip().lower()
    tail = tail.strip().lstrip("/")
    if lib not in CANONICAL_LIBS or not tail:
        return None
    if ".." in tail.split("/"):
        return None
    return lib, tail


def to_canonical_id(doc_id: str) -> str:
    """内部 doc_id → canonical doc_id（05 B4.1 口径）。

    kb:experience:E001                            -> experience:E001
    kb:norm:/abs/.../deliverables/03_norm/units.json
                                                  -> norm:units.json
    kb:template:/abs/.../_templates/opt-skeleton.json
                                                  -> template:opt-skeleton（约定去扩展名）
    kb:model:/abs/.../05_models/fluent.json       -> model:fluent.json

    尾段取「库内相对路径」而非裸文件名：model 库内 README.md / _index_data.json
    等重名文件很多，用裸名会产生多个文档共用一个 canonical id 的歧义；
    相对路径天然唯一，也与 B4.1 的 model:matlab/pn 写法一致。
    """
    if not doc_id:
        return ""
    s = str(doc_id).replace("\\", "/")
    if not s.startswith("kb:"):
        return s
    parts = s.split(":", 2)
    if len(parts) < 3:
        return s
    lib, tail = parts[1], parts[2]
    if lib == "experience":
        return f"experience:{tail}"
    low = tail.lower()
    rel = None
    for marker in _CANON_MARKERS:
        idx = low.find(marker)
        if idx >= 0:
            rel = tail[idx + len(marker):]
            break
    if not rel:
        rel = tail.split("/")[-1] if "/" in tail else tail
    if lib == "template" and rel.lower().endswith(".json"):
        rel = rel[:-5]
    return f"{lib}:{rel}"


def _coerce(value, allowed, default=None):
    if value is None:
        return default
    v = str(value).strip()
    if v in allowed:
        return v
    # 大小写不敏感匹配
    for a in allowed:
        if a.lower() == v.lower():
            return a
    return default


def _resolve_level(value):
    """层级别名解析（A9/A3）：优先内部层级，否则查 LEVEL_ALIASES，最后大小写不敏感。"""
    if value is None:
        return None
    v = str(value).strip()
    if v in LEVELS:
        return v
    if v in LEVEL_ALIASES:
        return LEVEL_ALIASES[v]
    for k, val in LEVEL_ALIASES.items():
        if k.lower() == v.lower():
            return val
    return None


# ── BM25 文本分词（中文按字、英文/数字按词） ──────────────────────────────
_CJK_RE = re.compile(r"[一-鿿]")
_ASCII_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """BM25 分词：CJK 单字 + ASCII 词。作为向量的互补信号。"""
    if not text:
        return []
    toks = _ASCII_RE.findall(text)
    toks += _CJK_RE.findall(text)
    return [t.lower() for t in toks if t]


class KBHybrid:
    """在 RAGKnowledgeBase（向量）之上叠加 BM25 + RRF 融合与硬过滤、回流、next_action。"""

    def __init__(
        self,
        rag_kb: RAGKnowledgeBase,
        bm25_corpus_path: str = "./kb_bm25_corpus.jsonl",
        k_rrf: int = 60,
    ):
        self.rag_kb = rag_kb
        self.corpus_path = Path(bm25_corpus_path)
        self.k = k_rrf
        self._corpus: List[Dict[str, Any]] = []   # [{doc_id, text, metadata}]
        self._bm25 = None
        self._feedback_log = Path(bm25_corpus_path).with_suffix(".feedback.jsonl")

    # ── BM25 语料管理 ──────────────────────────────────────────────────────
    def _load_corpus(self) -> None:
        if self._corpus:
            return
        if self.corpus_path.exists():
            try:
                with open(self.corpus_path, "r", encoding="utf-8") as fh:
                    self._corpus = [json.loads(line) for line in fh if line.strip()]
            except Exception as e:  # pragma: no cover
                logger.warning(f"KBHybrid: failed loading bm25 corpus: {e}")
                self._corpus = []
        if not self._corpus and self.rag_kb._collection is not None:
            # 从 Chroma 反向重建（权威来源）
            self._corpus = self._sync_get_all()
        self._rebuild_bm25()

    def _sync_get_all(self) -> List[Dict[str, Any]]:
        """同步拉取 Chroma 全量（KBHybrid 多在同步上下文构建）。"""
        try:
            return self.rag_kb.get_all_documents()
        except Exception:
            return []

    def _rebuild_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("KBHybrid: rank_bm25 not installed; BM25 leg disabled.")
            self._bm25 = None
            return
        if not self._corpus:
            self._bm25 = None
            return
        # 语料来源可能有两种字段名：add_document 写 "text"，Chroma 反向重建
        # (get_all_documents) 给 "content"。统一兜底，避免空 token 触发 idf 除零。
        self._bm25 = BM25Okapi(
            [tokenize(c.get("text") or c.get("content") or "") for c in self._corpus]
        )

    def _save_corpus(self) -> None:
        try:
            self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.corpus_path, "w", encoding="utf-8") as fh:
                for c in self._corpus:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        except Exception as e:  # pragma: no cover
            logger.warning(f"KBHybrid: failed saving bm25 corpus: {e}")

    def rebuild(self, from_chroma: bool = True) -> int:
        """重建 BM25 语料。

        BM25 语料以本地 jsonl 为唯一权威来源，绝不调用 chroma.get()——
        chroma 0.5.x 的 Rust 内核在部分 Windows 上 collection.get() 会原生
        access violation（不可被 try/except 捕获，直接拖死进程）。向量库仅用于
        混合检索的 query/upsert，不参与 BM25 重建。

        播种时本地 jsonl 与 chroma 同步写入，故本地语料始终是最新镜像。
        """
        if not self._corpus and self.corpus_path.exists():
            self._load_corpus()          # 从磁盘恢复语料（纯 BM25 与向量模式通用）
        self._rebuild_bm25()
        self._save_corpus()
        return len(self._corpus)

    def add_document(
        self, doc_id: str, text: str, metadata: Dict[str, Any], defer: bool = False
    ) -> None:
        """增量加入一篇文档（索引后调用，使 BM25 与向量同步）。

        defer=True 时只改内存语料，不重建 BM25 / 不落盘——批量播种时使用，
        由调用方在收尾时统一 rebuild()+_save_corpus()。否则 50+ 篇文档会触发
        50+ 次全量 BM25 重建 + 全量语料重写（O(n²) I/O，实测可拖到 30s 以上）。
        """
        self._corpus = [c for c in self._corpus if c.get("doc_id") != doc_id]
        self._corpus.append({"doc_id": doc_id, "text": text, "metadata": metadata})
        if not defer:
            self._rebuild_bm25()
            self._save_corpus()

    def remove_document(self, doc_id: str) -> None:
        self._corpus = [c for c in self._corpus if c.get("doc_id") != doc_id]
        self._rebuild_bm25()
        self._save_corpus()

    # ── doc_id 归一 ────────────────────────────────────────────────────────
    def resolve_doc_id(self, doc_id: str) -> Optional[str]:
        """把外部 doc_id 解析成语料内真实 doc_id（确定性，绝不做相似度猜测）。

        优先级：
          1) 精确匹配（内部键原样）
          2) canonical（lib:tail，05 B4.1 约定）→ 同库内「完整尾路径」或
             「basename 去扩展名」唯一命中
          3) 路径后缀 / 裸文件名精确 basename
        任一档出现多个并列候选 → 返回 None（A2：未知 id 不得用最相似顶替）。
        """
        if not doc_id:
            return None
        self._load_corpus()

        def norm(s: str) -> str:
            s = str(s or "").replace("\\", "/").lower()
            return s[5:] if s.startswith("file:") else s

        ids = [c.get("doc_id", "") for c in self._corpus]
        needle = norm(doc_id).lstrip("/")
        norm_map = {norm(i): i for i in ids}

        # 1) 精确匹配
        if needle in norm_map:
            return norm_map[needle]

        # 2) canonical：lib:tail
        parsed = parse_canonical(doc_id)
        if parsed:
            lib, tail = parsed
            tail_n = norm(tail).lstrip("/")
            base = tail_n.split("/")[-1]
            prefix = f"kb:{lib}:"
            cands: List[str] = []
            for i in ids:
                ni = norm(i)
                if not ni.startswith(prefix):
                    continue
                # 库内尾部：经验卡为 kb:experience:E001（无路径段），其余为绝对路径
                rest = ni[len(prefix):]
                if rest == tail_n:
                    cands.append(i)
                    continue
                if tail_n and rest.endswith("/" + tail_n):
                    cands.append(i)
                    continue
                b = rest.split("/")[-1]
                if b == base or _stem(b) == _stem(base):
                    cands.append(i)
            cands = list(dict.fromkeys(cands))
            if len(cands) == 1:
                return cands[0]
            return None   # 0 或多候选 → 未知（不猜）

        # 3) 后缀 / 裸文件名
        base = needle.split("/")[-1]
        hits = [i for i in ids if norm(i).split("/")[-1] == base]
        hits = list(dict.fromkeys(hits))
        return hits[0] if len(hits) == 1 else None

    # ── BM25 检索 ──────────────────────────────────────────────────────────
    def _bm25_search(
        self, query: str, top_n: int = 15, allowed_ids: Optional[set] = None
    ) -> List[Dict[str, Any]]:
        """BM25 打分。

        allowed_ids 非 None 时，只在"通过硬过滤"的文档子集内排序取前 top_n，
        否则 lib/level/case 过滤可能把全部命中滤掉、错误地返回 0 条。
        """
        if self._bm25 is None:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        try:
            scores = self._bm25.get_scores(toks)
        except Exception:
            return []
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if i >= len(self._corpus):
                continue
            doc_id = self._corpus[i]["doc_id"]
            if allowed_ids is not None and doc_id not in allowed_ids:
                continue
            out.append({"doc_id": doc_id, "score": float(scores[i])})
            if len(out) >= top_n:
                break
        return out

    # ── 混合检索（RRF 融合 + 硬过滤） ─────────────────────────────────────
    async def hybrid_retrieve(
        self,
        query: str = "",
        top_k: int = 5,
        level: Optional[str] = None,
        case: Optional[str] = None,
        lib: Optional[str] = None,
        doc_id: Optional[str] = None,
        kind: Optional[str] = None,
        discipline: Optional[str] = None,
        fetch_k_mult: int = 3,
        kb_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """混合检索主入口。

        - doc_id 给定 → 直达（已知路径，不走语义检索）
        - 否则：向量 top-N + BM25 top-N → RRF 融合 → 硬过滤（level/case/lib）
        - 返回每条含 fused_score / vector_score / bm25_score / metadata
        """
        level = _resolve_level(level)
        case = _coerce(case, CASES)
        lib = _coerce(lib, LIBS)
        kind = str(kind).strip() if kind else None
        discipline = str(discipline).strip() if discipline else None
        # A10：规范库默认只进 doc_id 直达索引；语义检索（非 doc_id 路径）不返回规范文档，
        # 除非调用方显式 lib=="norm"。
        exclude_norm = (doc_id is None and lib != "norm")

        self._load_corpus()

        # 已知路径直达：优先查本地语料（同步、无向量依赖），必要时再回退向量库。
        if doc_id:
            # doc_id 归一：canonical（如 norm:units.json，05 B4.1 约定）与路径后缀
            # 一律解析成语料内真实键；解析不到时原样兜底（既不命中即返回空，A2）。
            rid = self.resolve_doc_id(doc_id)
            key = rid or doc_id
            doc = None
            for c in self._corpus:
                if c.get("doc_id") == key:
                    doc = {
                        "doc_id": key,
                        "content": c.get("text") or c.get("content") or "",
                        "metadata": c.get("metadata", {}) or {},
                    }
                    break
            if doc is None and self.rag_kb._collection is not None:
                try:
                    doc = await self.rag_kb.get_document(key)
                except Exception:
                    doc = None
            if doc:
                doc = dict(doc)
                doc["canonical_id"] = to_canonical_id(doc.get("doc_id", ""))
                doc["direct"] = True
                doc["fused_score"] = 1.0
                doc["vector_score"] = None
                doc["bm25_score"] = None
                return [doc]
            return []

        fetch_k = max(top_k * fetch_k_mult, top_k)
        # 硬过滤下推到候选阶段：先算出合规 doc_id 集合，再让 BM25 只在该子集内排序。
        allowed_ids = None
        if lib or level or case or kind or discipline:
            allowed_ids = set()
            for c in self._corpus:
                m = c.get("metadata", {}) or {}
                if lib and m.get("lib") != lib:
                    continue
                if level and m.get("level") != level:
                    continue
                if case and m.get("case") != case:
                    continue
                if kind and m.get("kind") != kind:
                    continue
                if discipline and m.get("discipline") != discipline:
                    continue
                if exclude_norm and m.get("lib") == "norm":
                    continue
                allowed_ids.add(c.get("doc_id"))
            if allowed_ids:
                # 候选池至少覆盖全部合规文档，避免"有匹配却返回 0 条"
                fetch_k = max(fetch_k, min(len(allowed_ids), 200))
        vec = await self.rag_kb.retrieve(query, top_k=fetch_k, kb_only=kb_only)
        bm25 = self._bm25_search(query, top_n=fetch_k, allowed_ids=allowed_ids)

        # RRF 融合
        fused: Dict[str, float] = {}
        vec_by_id: Dict[str, Dict[str, Any]] = {}
        for rank, r in enumerate(vec):
            did = r.get("doc_id")
            fused[did] = fused.get(did, 0.0) + 1.0 / (self.k + rank + 1)
            vec_by_id[did] = r
        bm25_by_id: Dict[str, float] = {}
        for rank, r in enumerate(bm25):
            did = r.get("doc_id")
            fused[did] = fused.get(did, 0.0) + 1.0 / (self.k + rank + 1)
            bm25_by_id[did] = r.get("score", 0.0)

        # 组装结果（优先用向量侧 content/metadata；BM25 独有则补取）
        results: List[Dict[str, Any]] = []
        for did, fscore in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
            if did in vec_by_id:
                item = dict(vec_by_id[did])
            else:
                # BM25 独有命中：直接从本地语料取（不可同步调 async 的 get_document）
                d = next((c for c in self._corpus if c.get("doc_id") == did), None)
                if d is None:
                    continue
                item = {
                    "doc_id": did,
                    "content": d.get("text") or d.get("content") or "",
                    "metadata": d.get("metadata", {}) or {},
                }
            meta = item.get("metadata", {}) or {}
            # 硬过滤：level/case/lib/kind/discipline 任一带过滤条件时，缺失或不匹配即剔除
            if lib and meta.get("lib") != lib:
                continue
            if level and meta.get("level") != level:
                continue
            if case and meta.get("case") != case:
                continue
            if kind and meta.get("kind") != kind:
                continue
            if discipline and meta.get("discipline") != discipline:
                continue
            if exclude_norm and meta.get("lib") == "norm":
                continue
            if kb_only and meta.get("source_tag") in ("pe_result", "intermediate_output"):
                continue
            item["direct"] = False
            item["fused_score"] = round(fscore, 6)
            item["vector_score"] = round(vec_by_id.get(did, {}).get("score", 0.0), 4) if did in vec_by_id else None
            item["bm25_score"] = round(bm25_by_id.get(did, 0.0), 4)
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    # ── 回流（feedback） ────────────────────────────────────────────────────
    # 分层约定（P0-2）：本层只改**索引侧**（BM25 语料 + 向量 metadata）；
    # 经验卡的**源真相是磁盘上的卡文件**，回写文件与台账留痕属 KnowledgeManager
    # （cards_dir / card_file / next_card_id / _write_failed_under_to_card /
    #   archive_failure），因为 `_base_dir` 与 `ledger` 都是管理器的状态。
    async def record_feedback(
        self, doc_id: str, success: bool,
        scenario: Optional[str] = None, r0_hash: Optional[str] = None,
        run_id: Optional[str] = None, violations: Optional[List[str]] = None,
        note: str = "", recorded_by: str = "",
    ) -> Dict[str, Any]:
        """回流：成功升 confidence；失败降级并把失败**绑工况归档**（P0-2 / R7）。

        失败不再只是"悄悄降个置信度"——按 05 §B4.3 / §B4.4 阶段⑧，失败必须写
        ``failed_under = {scenario, R0_hash, ...}``（失败绑工况，惩罚权重按工况计）：
        同一 scenario+R0_hash 重复回流 → 覆盖该条（幂等），不无限堆积。
        没给工况就**不伪造**：只做旧的弱降级，并回报 ``failed_under_written=False``。

        与 P0-1 同纪律：**不写任何真值行**。失败的真值来自 ``run_id`` 指向的真实跑次，
        本方法只登记"卡在某工况失效"这一事实。
        """
        meta = None
        for c in self._corpus:
            if c.get("doc_id") == doc_id:
                meta = dict(c.get("metadata", {}) or {})
                break
        if meta is None:
            return {"status": "error", "message": f"doc_id '{doc_id}' not found"}

        conf = float(meta.get("confidence", 1.0) or 1.0)
        reflow_written = False
        if success:
            conf = min(1.0, conf + 0.05)
            meta["failed"] = False
            meta["outcome"] = "success"
        else:
            conf = max(0.0, conf - 0.10)
            if conf < 0.2:
                meta["failed"] = True
            if scenario and r0_hash:
                entries = normalize_failed_under(meta.get("failed_under"))
                key = (normalize_case(scenario), str(r0_hash))
                entries = [e for e in entries
                           if (normalize_case(e.get("scenario")),
                               str(e.get("R0_hash"))) != key]
                entry: Dict[str, Any] = {
                    "scenario": normalize_case(scenario),
                    "R0_hash": str(r0_hash),
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                }
                if run_id:
                    entry["run_id"] = str(run_id)
                if violations:
                    entry["violations"] = [str(v) for v in violations]
                if note:
                    entry["note"] = str(note)
                if recorded_by:
                    entry["recorded_by"] = str(recorded_by)
                entries.append(entry)
                meta["failed_under"] = dumps_failed_under(entries)
                meta["failed_scenarios"] = ",".join(failed_under_scenarios(entries))
                meta["failed"] = True
                meta["outcome"] = "failure"
                reflow_written = True
        meta["confidence"] = round(conf, 3)

        await self._sync_update(doc_id, meta)
        for c in self._corpus:
            if c.get("doc_id") == doc_id:
                c["metadata"] = meta
        self._save_corpus()
        # 反馈日志（审计/论文可复现）
        try:
            with open(self._feedback_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(
                    {"doc_id": doc_id, "success": success, "confidence": meta["confidence"],
                     "scenario": normalize_case(scenario) if scenario else None,
                     "R0_hash": r0_hash or None,
                     "failed_scenarios": meta.get("failed_scenarios") or ""},
                    ensure_ascii=False,
                ) + "\n")
        except Exception:
            pass
        return {"status": "success", "doc_id": doc_id, "confidence": meta["confidence"],
                "failed": meta.get("failed", False),
                "failed_under_written": reflow_written and not success,
                "failed_scenarios": meta.get("failed_scenarios") or ""}

    async def _sync_update(self, doc_id: str, meta: Dict[str, Any]) -> None:
        if self.rag_kb._collection is None:
            return                       # 无向量模式：元信息只落在 BM25 语料
        try:
            await self.rag_kb.update_document(doc_id, metadata=meta)
        except Exception as e:
            logger.warning(f"KBHybrid: update_document failed for {doc_id}: {e}")

    # ── 门下省：复用时适用性闸门（P0-1 / P0-2） ──────────────────────────────
    @staticmethod
    def vet_for_reuse(meta: Dict[str, Any], scenario: Optional[str] = None) -> Dict[str, Any]:
        """判定一条检索命中能否驱动 next_action 复用（门下省事前审核）。

        返回 ``{allowed, review_required, status, provenance_attested,
        failed_scenarios, matched_failure, reason}``。

        - 非经验库命中（规范/模板/模型）→ 直接放行（它们不经人审晋升闸门）；
        - 经验卡 rejected           → ``allowed=False``（人审驳回是**全局**否决）；
        - 经验卡 failed 但**无**工况绑定 → ``allowed=False``（无法界定适用范围，
          等同全局失败；兼容 P0-2 之前的 legacy ``failed`` 标记）；
        - 经验卡有 failed_under    → **只在该工况失效**（P0-2）：
              · 给了工况且命中失败工况（含 'ALL'）→ ``allowed=False``
              · 给了工况但没命中             → ``allowed=True`` + ``review_required``
                （规范 §3.3「工况对不上则惩罚权重 0」= A6 失败不封杀）
              · 没给工况                     → ``allowed=True`` + ``review_required``
                （不能因为"不知道工况"就把卡一票否决，那是 invalid=true 的老毛病）
        - 经验卡 candidate / 未审   → ``allowed=True`` 但 ``review_required=True``；
        - 经验卡 active（且无命中失败工况） → 放行。

        「失败不封杀」讨论的是**检索**：failed 卡仍可被检索到（A6），
        但它不得再驱动复用——两者不矛盾。
        """
        if not isinstance(meta, dict):
            return {"allowed": False, "review_required": True, "status": None,
                    "provenance_attested": False, "failed_scenarios": [],
                    "matched_failure": None,
                    "reason": "命中无 metadata，无法审查"}
        base = {"status": meta.get("status"),
                "provenance_attested": bool(meta.get("provenance_attested")),
                "failed_scenarios": failed_under_scenarios(meta.get("failed_under")),
                "matched_failure": None}
        if meta.get("lib") != "experience":
            return {**base, "allowed": True, "review_required": False,
                    "reason": f"lib={meta.get('lib')} 非经验库，无须人审闸门"}

        status = str(meta.get("status") or "candidate").strip().lower()
        attest = base["provenance_attested"]
        entries = normalize_failed_under(meta.get("failed_under"))
        fails = base["failed_scenarios"]

        if status in REUSE_BLOCKED_STATUS:
            return {**base, "allowed": False, "review_required": True,
                    "reason": (f"经验卡 status={status}：已被人审驳回，"
                               "不得驱动复用（检索仍可用）")}

        if bool(meta.get("failed")) and not entries:
            # legacy/P0-1 口径：failed 无工况绑定，无法界定适用范围 → 按全局失败处理
            return {**base, "allowed": False, "review_required": True,
                    "reason": ("经验卡 failed=True 且未绑定工况（无 failed_under）："
                               "无法界定失效范围，不得驱动复用（检索仍可用）")}

        want = normalize_case(scenario)
        if entries:
            if want and (want in fails or "ALL" in fails):
                hit = next(e for e in entries
                           if normalize_case(e.get("scenario")) == want) \
                    if want in fails else entries[0]
                return {**base, "allowed": False, "review_required": True,
                        "matched_failure": hit,
                        "reason": (f"经验卡在工况 {want} 有失败记录"
                                   f"（R0={hit.get('R0_hash')}）：该处置在此工况已失效，"
                                   "不得复用，应改走升级人工/改结构")}
            if want:
                return {**base, "allowed": True, "review_required": True,
                        "reason": (f"经验卡在 {fails} 工况失败过，与本工况 {want} 对不上"
                                   " → 惩罚权重 0（A6 失败不封杀）；复用时仍请人核适用性")}
            return {**base, "allowed": True, "review_required": True,
                    "reason": (f"经验卡在 {fails} 工况有失败记录，本次未给工况、"
                               "无法判定是否命中 → 可参考，复用时须人审确认工况")}

        if status != "active":
            return {**base, "allowed": True, "review_required": True,
                    "reason": (f"经验卡 status={status}（未过门下省）：可参考，"
                               "但复用时须人审确认适用性")}
        return {**base, "allowed": True, "review_required": False,
                "reason": "经验卡已 active（人审通过 + 签字 + 真值已证实），"
                          "且本工况无失败记录"}


    # ── 可机检下一动作 ──────────────────────────────────────────────────────
    async def propose_next_action(
        self,
        query: str,
        level: Optional[str] = None,
        case: Optional[str] = None,
        lib: Optional[str] = None,
        kind: Optional[str] = None,
        discipline: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """默认产物是可机检的 next_action，而非长文（§2.7 口径①）。

        P0-1：出口前过一遍门下省 ``vet_for_reuse``——被驳回/标记失败的经验卡
        不得驱动复用（降级升级人工）；未审经验卡照常输出动作但带 ``review_gate``。
        P0-2：闸门传入本次 ``case``（工况），使 ``failed_under`` 的工况绑定真正生效
        ——同一张卡在工况 T 失效，不影响它在工况 R 继续驱动动作。
        """
        res = await self.hybrid_retrieve(
            query, top_k=top_k, level=level, case=case, lib=lib,
            kind=kind, discipline=discipline,
        )
        if not res:
            return {"action": "升级人工", "action_enum": "escalate_human",
                    "action_convention": KB_FOUR_LIB_CONVENTION,
                    "doc_id": None, "confidence": None,
                    "reason": "no retrieved context; escalate to human",
                    "review_gate": {"allowed": True, "review_required": True,
                                    "status": None, "provenance_attested": False,
                                    "failed_scenarios": [], "matched_failure": None,
                                    "reason": "袋空，直接升级人工"}}
        top = res[0]
        meta = top.get("metadata", {}) or {}
        if meta.get("action") in NEXT_ACTIONS:
            action = meta["action"]
        elif meta.get("lib") == "template":
            action = "复用模板"
        elif meta.get("lib") == "norm":
            action = "只调本层"
        elif meta.get("lib") == "experience":
            action = "只调本层" if (meta.get("confidence", 1.0) or 1.0) >= 0.6 else "升级人工"
        else:
            action = "复用模板"
        # A5 动作枚举：对外统一输出规范四枚举（reuse_template/tune_layer/change_structure/escalate_human）
        # P0-3：规范值优先取卡上的 ``next_action_enum``（唯一权威），再退回由展示别名归一 ——
        # 不再"同一件事两种叫法各自猜"。
        action_enum = canonicalize_action(meta.get("next_action_enum"))
        if action_enum is None:
            action_enum = ACTION_CANON.get(action, "escalate_human")
        # P0-1 门下省事前审核：驳回/失败的经验卡硬阻断 → 降级升级人工
        gate = self.vet_for_reuse(meta, scenario=case)
        if not gate["allowed"]:
            action = "升级人工"
            action_enum = "escalate_human"
        return {
            "action": action,
            "action_enum": action_enum,
            "action_convention": KB_FOUR_LIB_CONVENTION,
            "doc_id": top.get("doc_id"),
            "confidence": meta.get("confidence"),
            "reason": f"top hit lib={meta.get('lib')} level={meta.get('level')} case={meta.get('case')}",
            "review_gate": gate,
        }


class KnowledgeManager:
    """知识库管理 Agent：四库编排 + 检索/入库/审核/回流/next_action。"""

    def __init__(
        self,
        rag_kb: RAGKnowledgeBase,
        bm25_corpus_path: str = "./kb_bm25_corpus.jsonl",
        vector_enabled: bool = True,
        ledger_enabled: bool = True,
    ):
        self.rag_kb = rag_kb
        # vector_enabled=False → 纯 BM25 模式：全程不 import / 不调用 chromadb。
        # 目的：chromadb 的原生依赖在部分 Windows 机器上会让整个进程崩溃，
        # 而管理端本身不依赖向量检索，故提供一条完全绕开 chromadb 的可用路径。
        self.vector_enabled = vector_enabled
        self.hybrid = KBHybrid(rag_kb, bm25_corpus_path=bm25_corpus_path)
        # 本轮播种中"向量入库失败"的 doc_id：用于决定是否写入播种清单
        # （失败不记账 → 下次增量播种能自动补灌，避免静默缺口）。
        self._seed_vec_failures: List[str] = []
        # P0-2：经验卡的**源真相是磁盘上的卡文件**（播种从那里读）。失败回流要回写文件，
        # 否则下一次全量重灌会把回流记录抹掉。此目录由 seed_four_libraries(base_dir)
        # 注入 —— 未播种时为 None，回流只改 metadata、不碰磁盘（验收脚本因此天然隔离）。
        self._base_dir: Optional[Path] = None
        # P0.5 台账：记录"发生过什么"(runs) 与经验卡晋升闸门(card_promotions)。
        # 纯标准库 sqlite3，不依赖 chromadb；失败也不影响检索主路径。
        # P2-7：`ledger_enabled=False`（ablation.kb_ledger=false）⇒ ledger=None，
        # 全部 `if self.ledger is not None` 守卫自然跳过 ⇒ runs/card_promotions 不记账，
        # 等价 P0.5 台账接入前。**默认 True**（所有既有调用方行为不变）。
        self._ledger_enabled = bool(ledger_enabled)
        if not self._ledger_enabled:
            self.ledger = None
            logger.info("KnowledgeManager: 台账已按消融开关 kb_ledger=false 关闭（不记账）。")
        else:
            try:
                self.ledger = KBLedger()
            except Exception as e:  # pragma: no cover
                logger.warning("KnowledgeManager: 台账初始化失败（忽略，不影响检索）：%s", e)
                self.ledger = None

    async def ensure_ready(self) -> bool:
        """就绪检查。

        纯 BM25 模式（vector_enabled=False）下**完全不触碰 chromadb**：
        直接从本地 BM25 语料恢复，永远返回 True，保证管理端开箱可用。
        """
        if not self.vector_enabled:
            n = self.hybrid.rebuild(from_chroma=False)
            logger.info("纯 BM25 模式就绪（语料 %s 篇，未加载向量库）。", n)
            return True

        if self.rag_kb._collection is None:
            try:
                ok = await asyncio.wait_for(self.rag_kb.initialize(), timeout=60.0)
            except asyncio.TimeoutError:
                logger.error(
                    "知识库初始化超时（embedding/Chroma 连接超时）。降级为纯 BM25。"
                )
                ok = False
            if not ok:
                # 向量不可用（chromadb 未装/导入失败/embedding 不可达）→ 降级纯 BM25，
                # 从本地 jsonl 恢复语料，保证管理端仍可检索，而非返回空。
                n = self.hybrid.rebuild(from_chroma=False)
                logger.warning("向量检索不可用，已降级为纯 BM25（语料 %s 篇）。", n)
                return True
        self.hybrid.rebuild(from_chroma=False)
        return True

    # ── 四库播种 ────────────────────────────────────────────────────────────
    def _manifest_path(self) -> Path:
        return self.hybrid.corpus_path.with_name("kb_seed_manifest.json")

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        """读取"文件 → (mtime,size)"清单，用于增量播种时跳过未变更文件。"""
        p = self._manifest_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"KnowledgeManager: manifest 读取失败，将全量重扫：{e}")
        # 首次（无清单）：用现有语料 + 文件 stat 预置清单（只 stat 不读文件，很快），
        # 这样升级后第一次点"四库重索引"就能直接跳过未变更文件。
        seeded: Dict[str, Dict[str, Any]] = {}
        for c in self.hybrid._corpus:
            did = c.get("doc_id") or ""
            if not did.startswith("kb:"):
                continue
            parts = did.split(":", 2)
            if len(parts) < 3:
                continue
            try:
                st = Path(parts[2]).stat()
            except Exception:
                continue
            seeded[did] = {"mtime": st.st_mtime, "size": st.st_size}
        return seeded

    def _save_manifest(self, manifest: Dict[str, Dict[str, Any]]) -> None:
        try:
            self._manifest_path().write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"KnowledgeManager: manifest 写入失败（忽略）：{e}")

    def _needs_ingest(
        self, path: Path, doc_id: str, manifest: Dict[str, Any],
        present: Optional[bool] = None,
    ) -> bool:
        """文件与上次播种时完全一致且目标已在语料中 → 无需重新读取入库。"""
        if present is None:
            present = any(c.get("doc_id") == doc_id for c in self.hybrid._corpus)
        if not present:
            return True
        rec = manifest.get(doc_id)
        if not rec:
            return True
        try:
            st = path.stat()
        except Exception:
            return True
        return rec.get("mtime") != st.st_mtime or rec.get("size") != st.st_size

    def _mark_manifest(self, path: Path, doc_id: str, manifest: Dict[str, Any]) -> None:
        try:
            st = path.stat()
            manifest[doc_id] = {"mtime": st.st_mtime, "size": st.st_size}
        except Exception:
            pass

    async def seed_four_libraries(self, base_dir: str = ".") -> Dict[str, int]:
        """把现有仓库内容按四库分类入库（幂等：同 doc_id 覆盖，未变更文件跳过）。

        批量播种期间 defer=True，最后统一重建 BM25 落盘一次，避免 O(n²) I/O。
        """
        base = Path(base_dir)
        self._base_dir = base          # P0-2：失败回流回写卡文件时的根
        counts = {"norm": 0, "model": 0, "template": 0, "experience": 0, "skipped": 0}
        seeded_doc_ids: List[str] = []   # P0.5：收集本次入库的 doc_id，写入台账 runs
        self._seed_vec_failures = []     # 本轮向量入库失败清单（决定是否记账）
        manifest = self._load_manifest()

        # 1) 规范库：knowledge_base 下的 .md 说明文件
        for md in (base / "knowledge_base").glob("*.md"):
            if md.name.lower() in ("readme.md",):
                continue
            n = await self._ingest_file(md, lib="norm", level="总体", case="ALL",
                                        source_tag="knowledge_base", manifest=manifest, acc=seeded_doc_ids)
            if n:
                counts["norm"] += n
            else:
                counts["skipped"] += 1

        # 2) 模型库：sysml / matlab / books（蒸馏书标 distilled=True，不自动计入"知识缩短周期"）
        for pat in ("knowledge_base/sysml/*", "knowledge_base/matlab/**/*"):
            for f in sorted(base.glob(pat)):
                if f.is_file() and f.suffix.lower() in (".xml", ".m", ".md", ".txt"):
                    n = await self._ingest_file(f, lib="model", level="总体", case="ALL",
                                                source_tag="knowledge_base", manifest=manifest, acc=seeded_doc_ids)
                    if n:
                        counts["model"] += n
                    else:
                        counts["skipped"] += 1
        books_root = base / "knowledge_base" / "books" / "books"
        if books_root.exists():
            for f in books_root.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".md", ".py", ".m", ".csv", ".json"):
                    if "__MACOSX" in f.parts:
                        continue
                    n = await self._ingest_file(f, lib="model", level="总体", case="ALL",
                                                source_tag="knowledge_base", distilled=True,
                                                manifest=manifest, acc=seeded_doc_ids)
                    if n:
                        counts["model"] += n
                    else:
                        counts["skipped"] += 1

        # 3) 模板库：knowledge_base/templates（Canvas-IR 模板种子）
        tpl_dir = base / "knowledge_base" / "templates"
        if tpl_dir.exists():
            for f in sorted(tpl_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".json", ".md", ".txt"):
                    if f.name.lower() == "readme.md":
                        continue
                    n = await self._ingest_file(f, lib="template", level="验证", case="ALL",
                                                source_tag="knowledge_base", manifest=manifest, acc=seeded_doc_ids)
                    if n:
                        counts["template"] += n
                    else:
                        counts["skipped"] += 1

        # 4) 经验库：parameter_experience_memory.json → 经验卡
        pe_json = base / "parameter_experience_memory.json"
        if pe_json.exists():
            pe_id = f"kb:pe:{pe_json.as_posix()}"
            has_exp = any(
                str(c.get("doc_id", "")).startswith("kb:experience:")
                for c in self.hybrid._corpus
            )
            if not self._needs_ingest(pe_json, pe_id, manifest, present=has_exp):
                counts["skipped"] += 1
            else:
                counts["experience"] = await self._seed_experience(pe_json)
                # 有经验卡向量入库失败 → 不记账，下次增量播种仍能补灌。
                if not self._seed_vec_failures:
                    self._mark_manifest(pe_json, pe_id, manifest)

        # 4.5) 冻结经验卡：deliverables/experience_cards（B11 最低种子 E001/E002）
        n_exp = await self._seed_experience_cards(base, manifest, acc=seeded_doc_ids)
        counts["experience"] += n_exp

        # 5) 规范直达表：deliverables/03_norm（建设方案 B11 要求进入 doc_id 直达索引）
        norm_dir = base / "deliverables" / "03_norm"
        if norm_dir.exists():
            for f in sorted(norm_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".json", ".yaml", ".yml", ".md", ".txt"):
                    n = await self._ingest_file(
                        f, lib="norm", level="总体", case="ALL",
                        source_tag="norm_table", manifest=manifest, acc=seeded_doc_ids,
                    )
                    counts["norm"] += n
                    if not n:
                        counts["skipped"] += 1

        # 6) 学科参数卡：deliverables/05_models（P0.4 预留扫描，目录不存在时跳过）
        models_dir = base / "deliverables" / "05_models"
        if models_dir.exists():
            for f in sorted(models_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".json", ".yaml", ".yml", ".py", ".md"):
                    n = await self._ingest_file(
                        f, lib="model", level="总体", case="ALL",
                        source_tag="model_card", manifest=manifest, acc=seeded_doc_ids,
                    )
                    counts["model"] += n
                    if not n:
                        counts["skipped"] += 1

        # 7) 画布骨架模板：deliverables/06_canvas（P0.3 预留扫描，目录不存在时跳过）
        canvas_dir = base / "deliverables" / "06_canvas"
        if canvas_dir.exists():
            for f in sorted(canvas_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".json", ".md", ".txt"):
                    n = await self._ingest_file(
                        f, lib="template", level="验证", case="ALL",
                        source_tag="canvas_template", manifest=manifest, acc=seeded_doc_ids,
                    )
                    counts["template"] += n
                    if not n:
                        counts["skipped"] += 1

        self._save_manifest(manifest)
        # 收尾统一重建 BM25 + 落盘（覆盖本次批量变更）
        self.hybrid.rebuild(from_chroma=True)
        # P0.5 台账：本次播种作为一次 run 落盘，证明它真实发生过（论文可复现审计）。
        if self.ledger is not None:
            seed_run_id = f"seed-{int(time.time()*1000)}"
            self.ledger.log_run(
                seed_run_id, prompt="seed_four_libraries", source="seed",
                doc_ids=seeded_doc_ids, status="done",
            )
        logger.info(
            "四库播种完成：norm=%s model=%s template=%s experience=%s（跳过/空文件 %s）",
            counts["norm"], counts["model"], counts["template"],
            counts["experience"], counts["skipped"],
        )
        return counts

    async def _ingest_file(
        self, path: Path, lib: str, level: str, case: str,
        source_tag: str = "knowledge_base", distilled: bool = False,
        manifest: Optional[Dict[str, Any]] = None,
        acc: Optional[List[str]] = None,
    ) -> int:
        # doc_id 一律用绝对规范化路径，避免"同文件因 cwd/base_dir 不同生成两套
        # doc_id"导致语料翻倍（生产 run_web 用 os.getcwd()，与首次种子绝对路径不一致）。
        doc_id = f"kb:{lib}:{path.resolve().as_posix()}"
        # 增量跳过：文件未变更且已在语料中 → 不读取、不重建
        if manifest is not None and not self._needs_ingest(path, doc_id, manifest):
            return 0
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return 0
        if not text.strip():
            return 0
        meta = self._build_meta(lib, level, case, source_tag,
                                source_of_truth=str(path.resolve()),
                                distilled=distilled, doc_id=doc_id)
        # P0.7 A7/A12：模板/模型卡从 JSON 文件内容抽取 kind/discipline 进入 metadata，
        # 使 hybrid_retrieve 的 kind=/discipline= 硬过滤可定位（如优化链骨架、Fluent 卡）。
        if lib in ("template", "model") and path.suffix.lower() == ".json":
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    if obj.get("kind"):
                        meta["kind"] = str(obj["kind"])
                    if obj.get("discipline"):
                        meta["discipline"] = str(obj["discipline"])
            except Exception:
                pass
        vec_ok = await self._index_vector(doc_id, text, meta)
        # 无论向量是否成功，都写入 BM25 语料（这是纯 BM25 模式的唯一索引，
        # 也保证 chromadb 坏掉时管理端依然能检索到入库内容）。
        self.hybrid.add_document(doc_id, text, meta, defer=True)
        # 播种清单只在"向量确实写进去了"或"纯 BM25 模式（本就不写向量）"时记账。
        # 否则向量失败的文件会被记为已入库，下次增量播种直接跳过 → 永久缺口。
        if manifest is not None and (vec_ok or not self.vector_enabled):
            self._mark_manifest(path, doc_id, manifest)
        elif manifest is not None:
            logger.warning(
                f"KnowledgeManager: {doc_id} 向量入库失败，不写入播种清单"
                f"（下次启动自动补灌）；BM25 索引已写入。"
            )
        if acc is not None:
            acc.append(doc_id)
        return 1

    async def _seed_experience(self, pe_json: Path, defer: bool = True) -> int:
        """把 parameter_experience_memory.json 转成经验卡。

        defer=True：只写内存语料，不逐张重建 BM25（由 seed_four_libraries 收尾统一落盘）。
        """
        try:
            data = json.loads(pe_json.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"KnowledgeManager: cannot read PE json: {e}")
            return 0
        entries = []
        for key in ("short_term", "long_term", "episodic"):
            v = data.get(key)
            if isinstance(v, list):
                entries.extend(v)
        cnt = 0
        for e in entries:
            tc = (e.get("task_context") or {})
            mode = tc.get("mode") or "TUNE"
            run_case = tc.get("RUN_CASE") or tc.get("category") or "ALL"
            level = MODE_LEVEL.get(mode, "总体")
            case = run_case if run_case in CASES else "ALL"
            fitness = float(e.get("fitness", 0.0) or 0.0)
            action = ("复用模板" if fitness >= 0.9 else
                      "只调本层" if fitness >= 0.6 else "升级人工")
            mid = e.get("memory_id") or f"{hash(json.dumps(e, sort_keys=True))}"
            doc_id = f"kb:experience:{mid}"
            params = e.get("parameters", {})
            objs = e.get("objectives", {})
            content = (
                f"[经验卡] mode={mode} case={case} level={level}\n"
                f"任务: {tc.get('task','')} ({tc.get('prompt','')[:200]})\n"
                f"参数: {json.dumps(params, ensure_ascii=False)}\n"
                f"目标: {json.dumps(objs, ensure_ascii=False)}\n"
                f"适应度 fitness={fitness:.3f} → 建议动作={action}"
            )
            meta = self._build_meta(
                "experience", level, case, "experience",
                source_of_truth=str(pe_json.resolve()),
                machine_checkable=True, confidence=min(1.0, max(0.0, fitness)),
                action=action, symptom=mode, doc_id=doc_id,
            )
            ok = await self._index_vector(doc_id, content, meta)
            self.hybrid.add_document(doc_id, content, meta, defer=defer)
            if not ok and self.vector_enabled:
                self._seed_vec_failures.append(doc_id)
            cnt += 1
        if self._seed_vec_failures:
            logger.warning(
                f"KnowledgeManager: 经验向量入库失败 {len(self._seed_vec_failures)} 条"
                f"（BM25 已写入，播种清单不记账以便下轮补灌）："
                f"{self._seed_vec_failures[:5]}"
            )
        return cnt

    async def _seed_experience_cards(
        self, base: Path, manifest: Dict[str, Any], acc: Optional[List[str]] = None
    ) -> int:
        """把 deliverables/experience_cards/*.json 冻结经验卡入库（B11 最低种子 E001/E002）。

        与 _seed_experience（来自 parameter_experience_memory.json 自动派生）不同：这里是
        显式、带层级金标、可机检的经验卡；doc_id = kb:experience:<card.id>（稳定、可直达）。
        卡内字段（status/source_run_ids/root_cause/failed_under/next_action）写入 metadata
        （kb_doc schema 的 metadata 允许 additionalProperties）。

        P0-1（2026-09-30）两条硬规则：
          1. **声明 ≠ 真值**：卡内 ``source_run_ids`` 只是声明，播种不得代写进 runs
             （修复前此处直接写入，使 gate_check 对同源卡恒真 = 自证循环）。
             声明改为落 ``card_promotions`` 的 draft 行留痕；真值只能由真实来源经
             ``KBLedger.attest_run()`` 登记。
          2. **卡内 status 不作数**：播种一律以 candidate 入库（卡内声明 active 的
             E001/E002 曾直接入库，绕过人审闸门）。已有人审结论（台账有 accepted/active
             /rejected 记录）才沿用，避免全量重灌把已审卡打回未审。
        """
        cards_dir = base / "deliverables" / "experience_cards"
        if not cards_dir.exists():
            return 0
        cnt = 0
        for f in sorted(cards_dir.glob("*.json")):
            try:
                card = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"KnowledgeManager: 经验卡 {f.name} 解析失败：{e}")
                continue
            cid = card.get("id")
            if not cid:
                continue
            doc_id = f"kb:experience:{cid}"
            declared_runs = list(card.get("source_run_ids", []) or [])
            declared_status = str(card.get("status", "") or "")
            # P0-1①：声明留痕（draft），绝不代写 runs 真值。幂等，重复播种无副作用。
            if self.ledger is not None:
                try:
                    self.ledger.log_claim(
                        card_id=cid,
                        source_run_ids=declared_runs,
                        proposed_layer=str(card.get("layer", "") or ""),
                        proposed_action=str(card.get("next_action", "") or ""),
                        declared_status=declared_status,
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning(f"KnowledgeManager: 经验卡声明留痕失败（忽略）：{e}")
            if not self._needs_ingest(f, doc_id, manifest):
                cnt += 1
                continue
            # 卡 → metadata/content 的换算收敛到 `_card_meta_and_content`，
            # 与单卡入库入口 `ingest_card()` **同一份口径**（避免两处各写一遍）。
            meta, content, eff_status = self._card_meta_and_content(
                card, doc_id, source_of_truth=str(f.resolve()))
            if eff_status != "active":
                logger.info(
                    "KnowledgeManager: 经验卡 %s 以 %s 入库（卡内声明 status=%r 不作数，"
                    "须走 review→activate 人审闸门）", cid, eff_status, declared_status,
                )
            vec_ok = await self._index_vector(doc_id, content, meta)
            self.hybrid.add_document(doc_id, content, meta, defer=True)
            # 同上：仅在向量入库成功（或纯 BM25 模式）时记账，避免"失败却被标记完成"。
            if vec_ok or not self.vector_enabled:
                self._mark_manifest(f, doc_id, manifest)
            else:
                logger.warning(
                    f"KnowledgeManager: {doc_id} 向量入库失败，不写入播种清单"
                    f"（下次启动自动补灌）；BM25 索引已写入。"
                )
            if acc is not None:
                acc.append(doc_id)
            cnt += 1
        return cnt

    def _build_meta(
        self, lib: str, level: str, case: str, source_tag: str,
        source_of_truth: str, distilled: bool = False,
        machine_checkable: bool = False, confidence: float = 1.0,
        action: Optional[str] = None, symptom: Optional[str] = None,
        version: str = "v1", doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        tags = [lib, level, case]
        if distilled:
            tags.append("distilled")
        meta: Dict[str, Any] = {
            "lib": lib,
            "level": level,
            "case": case,
            "source_tag": source_tag,
            "type": lib,
            "version": version,
            "source_of_truth": source_of_truth,
            "confidence": round(float(confidence), 3),
            "machine_checkable": bool(machine_checkable),
            "distilled": bool(distilled),
            "reviewed": False,
            "failed": False,
            "status": "candidate",          # P0.6 人审状态机初态：待人审
            "tags": ",".join(tags),
            **({"action": action} if action else {}),
            **({"symptom": symptom} if symptom else {}),
        }
        if doc_id:
            # 05 B4.1 doc_id 约定的"对外契约 id"，与内部主键 kb:{lib}:{path} 并存
            meta["canonical_id"] = to_canonical_id(doc_id)
        return meta

    def _card_meta_and_content(
        self, card: Dict[str, Any], doc_id: str, source_of_truth: str = "",
    ) -> "tuple[Dict[str, Any], str, str]":
        """经验卡 → ``(metadata, content, 生效状态)`` 的**唯一**换算口。

        口径三条（P0-1 / P0-2，播种与单卡入库**共用**，不得各写一份）：
          1. **入库状态不以卡内声明为准**，只认台账里的人审结论（无结论 ⇒ candidate）；
          2. **卡内 status / source_run_ids 原样留档**，便于审计"卡说它是什么"与
             "台账认它是什么"的差异；声明本身不构成真值；
          3. ``failed_under`` 统一存 JSON 串（chroma 只吃标量；BM25 两侧同存法，
             否则 ``check_index_metadata_sync`` 的漂移断言形同虚设）。
        """
        cid = str(card.get("id") or "").strip()
        level = _coerce(card.get("layer"), LEVELS, "总体")
        case = _coerce(card.get("case"), CASES, "ALL")
        action_zh = {
            "reuse_template": "复用模板",
            "tune_layer": "只调本层",
            "change_structure": "改结构",
            "escalate_human": "升级人工",
        }.get(card.get("next_action"), "只调本层")
        content = card.get("content") or json.dumps(card, ensure_ascii=False)
        # P0-1②：入库状态不以卡内声明为准，只认台账里的人审结论。
        eff_status = self._review_state_from_ledger(cid) or "candidate"
        # P0-2：失败卡/成功卡同目录同编号，用 outcome 与 failed_under 区分
        fu_entries = normalize_failed_under(card.get("failed_under"))
        outcome = str(card.get("outcome") or "").strip().lower()
        if outcome not in ("success", "failure"):
            # 未标 outcome 的存量卡：有失败回流记录即失败卡，否则按成功卡
            outcome = "failure" if fu_entries else "success"
        meta = self._build_meta(
            "experience", level, case, "experience_card",
            source_of_truth=source_of_truth,
            machine_checkable=True,
            confidence=float(card.get("confidence", 1.0)),
            action=action_zh, symptom=card.get("symptom", ""),
            doc_id=doc_id,
        )
        meta.update({
            "card_id": cid,
            "status": eff_status,
            # 卡内声明原样留档，便于审计"卡说它是什么"与"台账认它是什么"的差异
            "declared_status": str(card.get("status", "") or ""),
            "source_run_ids": list(card.get("source_run_ids", []) or []),
            "root_cause": card.get("root_cause", ""),
            # P0-2：failed_under 统一存 JSON 串（chroma 只吃标量；BM25 两侧同存法，
            # 否则 check_index_metadata_sync 的漂移断言形同虚设）
            "failed_under": dumps_failed_under(fu_entries),
            "outcome": outcome,
            "failed_scenarios": ",".join(failed_under_scenarios(fu_entries)),
            "paired_with": str(card.get("paired_with") or ""),
            "next_action_enum": card.get("next_action", "tune_layer"),
            # 真值是否已证实（activate 通过闸门时置 True）
            "provenance_attested": eff_status == "active",
        })
        # failed 标记沿用既有语义（record_feedback 弱降级也用它）：有失败回流记录即置位。
        # 注意它**不再**等于"全局封杀"——vet_for_reuse 只按 failed_under 的工况判失效范围。
        if fu_entries or outcome == "failure":
            meta["failed"] = True
        return meta, content, eff_status

    async def ingest_card(
        self, card: Dict[str, Any], *, claim: bool = True, source_of_truth: str = "",
    ) -> Dict[str, Any]:
        """把**一张**经验卡入库（状态一律 candidate，待人审）。

        这是"跑完 → 待审 → 入库"回流链的入库口，与批量播种 ``_seed_experience_cards``
        共用 :meth:`_card_meta_and_content`，因此两条路径的元数据口径**逐字相同**。

        纪律（与 P0-1 一致，**不得放宽**）：
          * 卡内 ``status`` 不作数 —— 只认台账人审结论；无结论 ⇒ candidate；
          * ``source_run_ids`` 只是**声明**，经 ``log_claim`` 落 ``draft`` 行留痕，
            **绝不代写 runs 真值**（真值只能由 ``KBLedger.attest_run()`` 登记）；
          * 本方法**不会**把卡置为 active —— 晋升必须走 ``review()`` + ``activate()``。
        """
        cid = str(card.get("id") or "").strip()
        if not cid:
            return {"status": "error", "message": "经验卡缺少 id，拒绝入库"}
        doc_id = f"kb:experience:{cid}"
        declared_runs = list(card.get("source_run_ids", []) or [])
        declared_status = str(card.get("status", "") or "")

        if claim and self.ledger is not None:
            try:
                self.ledger.log_claim(
                    card_id=cid, source_run_ids=declared_runs,
                    proposed_layer=str(card.get("layer", "") or ""),
                    proposed_action=str(card.get("next_action", "") or ""),
                    declared_status=declared_status,
                    note="reflow: 回流卡内来源声明（未经证实），待人审 + 真值登记",
                )
            except Exception as e:                            # pragma: no cover
                logger.warning("KnowledgeManager: 回流卡声明留痕失败（忽略）：%s", e)

        meta, content, eff_status = self._card_meta_and_content(
            card, doc_id, source_of_truth=source_of_truth)
        vec_ok = await self._index_vector(doc_id, content, meta)
        # defer=False：单卡入库即重建 BM25 并**落盘语料**，使同进程 / 重启后都能检索到
        self.hybrid.add_document(doc_id, content, meta)
        if eff_status != "active":
            logger.info(
                "KnowledgeManager: 回流卡 %s 以 %s 入库（卡内声明 status=%r 不作数，"
                "须走 review→activate 人审闸门）", cid, eff_status, declared_status,
            )
        return {"status": "success", "doc_id": doc_id, "card_id": cid,
                "status_now": eff_status, "declared_status": declared_status,
                "source_run_ids": declared_runs, "vector": bool(vec_ok),
                "claim_logged": bool(claim and self.ledger is not None),
                "gate": ("待人审：review(accept) → activate(签字 + 真值已证实)"
                         if eff_status != "active" else "已 active")}

    async def _index_vector(self, doc_id: str, content: str, meta: Dict[str, Any]) -> bool:
        """向量入库（最佳努力）。

        纯 BM25 模式或 chromadb 不可用时静默跳过，绝不因此中断入库流程——
        BM25 语料才是管理端可用的保底索引。
        """
        if not self.vector_enabled:
            return False
        try:
            return bool(await self.rag_kb.index_documents([{
                "id": doc_id, "content": content, "metadata": meta,
            }]))
        except Exception as e:
            logger.warning(f"KnowledgeManager: 向量入库失败（忽略，BM25 仍生效）：{e}")
            return False

    # ── 对外能力（对齐方案 C1~C4） ─────────────────────────────────────────
    async def query(self, query: str, level=None, case=None, lib=None, top_k=5) -> List[Dict[str, Any]]:
        return await self.hybrid.hybrid_retrieve(query, top_k=top_k, level=level, case=case, lib=lib)

    async def get_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """取单篇文档全文。优先本地 BM25 语料（无向量模式也可用），回落向量库。

        doc_id 先经 :meth:`KBHybrid.resolve_doc_id` 归一（canonical / 路径后缀 /
        裸文件名 → 语料内真实键），因此 05 B4.1 约定的 ``norm:units.json`` 与
        展开接口用的 ``kb:norm:<abs path>`` 都能命中；解析不到再回落向量库
        （A2：未知 id 返回 None，绝不用最相似的另一篇顶替）。
        """
        if not doc_id:
            return None
        rid = self.hybrid.resolve_doc_id(doc_id)
        if rid:
            for c in self.hybrid._corpus:
                if c.get("doc_id") == rid:
                    return {
                        "doc_id": rid,
                        "canonical_id": to_canonical_id(rid),
                        "content": c.get("text", ""),
                        "metadata": c.get("metadata", {}) or {},
                    }
        # 回落向量库（已就绪时）
        if self.vector_enabled:
            try:
                return await self.rag_kb.get_document(doc_id)
            except Exception:
                return None
        return None

    async def next_action(self, query: str, level=None, case=None, lib=None,
                           kind=None, discipline=None) -> Dict[str, Any]:
        return await self.hybrid.propose_next_action(
            query, level=level, case=case, lib=lib, kind=kind, discipline=discipline)

    async def feedback(
        self, doc_id: str, success: bool,
        scenario: Optional[str] = None, r0_hash: Optional[str] = None,
        run_id: Optional[str] = None, violations: Optional[List[str]] = None,
        note: str = "", recorded_by: str = "",
    ) -> Dict[str, Any]:
        """回流入口（P0-2 扩展）：索引侧改动由 KBHybrid 做，**卡文件回写由本层做**。

        Web `/api/feedback` 与 CLI 都走这里；带 scenario + r0_hash 的失败会同时
        写进卡文件（卡的源真相），否则下一次全量重灌会把失败回流记录抹掉。
        """
        res = await self.hybrid.record_feedback(
            doc_id, success, scenario=scenario, r0_hash=r0_hash, run_id=run_id,
            violations=violations, note=note, recorded_by=recorded_by)
        if res.get("status") == "success" and res.get("failed_under_written"):
            res["card_file"] = self._write_failed_under_to_card(doc_id)
        return res

    # ── P0-2 失败案例归档（05 §B4.4 阶段⑧ / §B11 / R7）──────────────────────
    def set_base_dir(self, base_dir) -> None:
        """指定工程根（经验卡目录 = ``{base}/deliverables/experience_cards``）。

        播种时会自动设置；独立工具（如 scripts/archive_failure.py）只想定位卡目录、
        不想顺带重灌索引时走这里。
        """
        self._base_dir = Path(base_dir)

    def cards_dir(self) -> Optional[Path]:
        """经验卡目录（播种根下的 deliverables/experience_cards）。未播种 → None。"""
        return None if self._base_dir is None \
            else self._base_dir / "deliverables" / "experience_cards"

    def card_file(self, card_id: str) -> Optional[Path]:
        """按裸卡号（E003 / kb:experience:E003 / experience:E003）定位卡文件。"""
        d = self.cards_dir()
        if d is None:
            return None
        cid = str(card_id or "").strip()
        for prefix in ("kb:experience:", "experience:"):
            if cid.startswith(prefix):
                cid = cid[len(prefix):]
        f = d / f"{cid}.json"
        return f if f.is_file() else None

    def next_card_id(self) -> str:
        """分配下一个 E### 空位（只看文件名，不猜内容）。"""
        d = self.cards_dir()
        used = set()
        if d is not None and d.is_dir():
            for f in d.glob("E*.json"):
                m = re.fullmatch(r"E(\d{3})", f.stem)
                if m:
                    used.add(int(m.group(1)))
        n = 1
        while n in used:
            n += 1
        return f"E{n:03d}"

    def _write_failed_under_to_card(self, doc_id: str) -> Optional[str]:
        """把索引侧的 failed_under 回写到卡文件（卡的源真相）。

        未播种（``_base_dir is None``）→ 返回 None，只改索引。验收脚本因此天然隔离：
        临时语料 + 未播种 = 绝不碰生产卡文件。
        """
        meta = self._get_doc_meta(doc_id)
        if not meta:
            return None
        path = self.card_file(meta.get("card_id") or doc_id)
        if path is None:
            return None
        entries = normalize_failed_under(meta.get("failed_under"))
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
            card["failed_under"] = entries          # 规范口径：对象（或对象数组）
            card["outcome"] = "failure"
            card["confidence"] = meta.get("confidence", card.get("confidence", 1.0))
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
            return str(path)
        except Exception as e:  # pragma: no cover
            logger.warning("KnowledgeManager: 回写失败回流到卡文件失败（%s）：%s", path, e)
            return None

    async def archive_failure(
        self, symptom: str, layer: str, scenario: str, r0_hash: str,
        approach: str, root_cause: str = "", run_id: str = "",
        violations: Optional[List[str]] = None, next_action: str = "change_structure",
        paired_with: str = "", source_run_ids: Optional[List[str]] = None,
        confidence: float = 0.5, recorded_by: str = "", card_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """另提一张**失败卡**（candidate），沉淀"此工况勿用该处置"。

        规范要点（05 §B4.4 阶段⑧ / §B11 / R7）：
          * 失败**不另开失败库**——失败卡与成功卡同目录、同 E### 编号，
            靠 outcome / failed_under 区分（05 §5「失败回流改经验数据上的 failed_under」）；
          * 若要把"此工况勿用该处置"沉淀成一张卡 → 必须先提 **candidate，仍须人审**；
          * 失败必须绑工况（scenario + R0_hash），**禁止**用全局 invalid=true 表达失败；
          * 与成功卡一成一败成对（``paired_with``，工程扩展字段，便于机检）。

        与 P0-1 同纪律：``source_run_ids`` 只是**声明**（落 draft 留痕），
        绝不代写 runs 真值。
        卡文件写盘后**并不自动入库**——须经播种（seed_four_libraries / Web
        四库重索引）读入，与其它卡同一路径；状态停在 candidate 待人审。
        """
        d = self.cards_dir()
        if d is None:
            return {"status": "error",
                    "message": "未播种过（_base_dir 为空），无法定位经验卡目录；"
                               "请先 seed_four_libraries(base_dir=...)"}
        layer = _coerce(layer, LEVELS, "总体")
        if next_action not in ACTION_ENUMS:
            return {"status": "error",
                    "message": f"next_action 必须是四枚举之一，收到 {next_action!r}"}
        sc = normalize_case(scenario)
        if not sc or not str(r0_hash or "").strip():
            return {"status": "error",
                    "message": "失败必须绑工况：scenario 与 R0_hash 均必填（05 §B4.3）"}

        cid = card_id or self.next_card_id()
        path = d / f"{cid}.json"
        if path.exists() and not card_id:
            return {"status": "error", "message": f"卡号 {cid} 已存在（并发写入？）"}

        declared = list(source_run_ids or ([run_id] if run_id else []))
        entry: Dict[str, Any] = {"scenario": sc, "R0_hash": str(r0_hash),
                                 "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                              time.localtime())}
        if run_id:
            entry["run_id"] = str(run_id)
        if violations:
            entry["violations"] = [str(v) for v in violations]
        if recorded_by:
            entry["recorded_by"] = str(recorded_by)

        action_zh = {"reuse_template": "复用模板", "tune_layer": "只调本层",
                     "change_structure": "改结构", "escalate_human": "升级人工"}[next_action]
        card: Dict[str, Any] = {
            "id": cid,
            "layer": layer,
            "symptom": symptom,
            "next_action": next_action,
            "status": "candidate",              # 规范：新失败卡必须提 candidate 且经人审
            "outcome": "failure",
            "source_run_ids": declared,
            "root_cause": root_cause,
            "failed_under": entry,              # 对象（单条）；回流累积时变对象数组
            "confidence": float(confidence),
            "content": (f"失败经验 {cid}（{layer}层）：症状「{symptom}」，"
                        f"在工况 {sc}（R0_hash={r0_hash}）下处置「{approach}」失效。"
                        f"根因：{root_cause or '（待补）'}。"
                        f"该失败只绑工况 {sc}，不得据此全局否定同族经验；"
                        f"应改走 {action_zh}({next_action})。"),
        }
        if paired_with:
            card["paired_with"] = str(paired_with)

        d.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

        # 声明留痕（draft），与 _seed_experience_cards 同口径——绝不代写 runs 真值
        if self.ledger is not None:
            try:
                self.ledger.log_claim(
                    card_id=cid, source_run_ids=declared, proposed_layer=layer,
                    proposed_action=next_action, declared_status="candidate",
                    note=f"failure_reflow: scenario={sc} R0_hash={r0_hash}",
                )
            except Exception as e:  # pragma: no cover
                logger.warning("KnowledgeManager: 失败卡声明留痕失败（忽略）：%s", e)

        logger.info("已归档失败卡 %s（工况 %s，待播种入库 + 人审）：%s", cid, sc, path)
        return {"status": "success", "card_id": cid, "doc_id": f"kb:experience:{cid}",
                "card_file": str(path), "scenario": sc, "R0_hash": str(r0_hash),
                "next_action": next_action,
                "note": "卡文件已写盘；入库须跑 seed_four_libraries（或 Web 四库重索引），"
                        "状态停在 candidate 待人审"}

    # ── 人审状态机（P0.6 / P0-1） ─────────────────────────────────────────────
    # 状态：draft(播种只留声明) → candidate(已入库,待人审) → accepted(已过审,待签字)
    #        → active(已签字+真值已证实)；candidate → rejected(驳回,留痕)
    # 规则：activate 必须有签字(signer 非空)，且 source_run_ids 全部「已证实」
    #       （"无真值/无签字不得 active"）。
    def _review_state_from_ledger(self, card_id: str) -> Optional[str]:
        """取该卡的既有**人审结论**（accepted/active/rejected）；无则 None。

        播种时用它保留人审结果：全量重灌不得把已审卡打回 candidate。
        只认 review()/activate() 落下的行（draft 声明行不计入），故不是绕过闸门的后门。
        """
        if self.ledger is None:
            return None
        try:
            return self.ledger.latest_review_status(card_id)
        except Exception as e:  # pragma: no cover
            logger.warning(f"KnowledgeManager: 读取人审结论失败（按未审处理）：{e}")
            return None

    def _get_doc_meta(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """取文档 metadata（以内存语料为准）。

        修复（2026-09-30）：原实现先试 `self.rag_kb.get_document(doc_id)`，
        但那是 async 方法，这里没 await —— 拿到的是 coroutine 对象，对它调
        `.get()` 抛 AttributeError 被 `except Exception: pass` 吞掉。结果是
        （a）该分支永远走不到，纯属死代码；（b）每次人审都刷一条
        "coroutine 'RAGKnowledgeBase.get_document' was never awaited" RuntimeWarning。

        内存语料 `hybrid._corpus` 本身就是「向量库 + 本地语料」合并后的权威视图，
        且 `_set_doc_meta()` 双写它，故直接以它为准即可，不必再进 async 边界。
        """
        for c in self.hybrid._corpus:
            if c.get("doc_id") == doc_id:
                return dict(c.get("metadata", {}) or {})
        return None

    async def _set_doc_meta(self, doc_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """更新文档 metadata：内存语料 + 落盘 + 向量库，并**如实回报**同步结果。

        修复（2026-09-30）：原实现用 `run_coroutine_threadsafe` 把向量回写**甩给**
        一个可能没在跑的 loop，既不 await 也不看返回值——失败就永久静默。实测后果
        是 BM25 语料写成了 active、Chroma 里还停在 candidate，两个索引对同一张卡
        的审查状态各说各话，而接口照样返回 success。

        现在改为 await，并把结果作为 ``vector_synced`` 回传；
        向量库没同步上时接口会带 ``vector_synced=False`` + ``vector_error``，
        让调用方/界面看得见"索引不同步"，而不是假装成功。
        """
        for c in self.hybrid._corpus:
            if c.get("doc_id") == doc_id:
                c["metadata"] = meta
                break
        self.hybrid._save_corpus()

        result: Dict[str, Any] = {"vector_synced": None, "vector_error": ""}
        if not self.vector_enabled:
            result["vector_synced"] = False
            result["vector_error"] = "纯 BM25 模式，未写向量库"
            return result
        if getattr(self, "rag_kb", None) is None or self.rag_kb._collection is None:
            result["vector_synced"] = False
            result["vector_error"] = "向量库未初始化（_collection is None）"
            logger.warning("KnowledgeManager: %s 向量库未初始化，代码=%s 仅落 BM25 语料",
                           doc_id, result["vector_error"])
            return result
        try:
            ok = await self.rag_kb.update_document(doc_id, metadata=meta)
            result["vector_synced"] = bool(ok)
            if not ok:
                result["vector_error"] = "update_document 返回 False（详见向量库日志）"
                logger.error("KnowledgeManager: %s 回写向量库失败：%s",
                             doc_id, result["vector_error"])
        except Exception as e:
            result["vector_synced"] = False
            result["vector_error"] = str(e)
            logger.error("KnowledgeManager: %s 回写向量库异常：%s", doc_id, e)
        return result

    async def review(
        self, doc_id: str, ok: Optional[bool] = None,
        decision: Optional[str] = None, reviewer: str = "",
    ) -> Dict[str, Any]:
        """人审：candidate → accepted(通过) / rejected(驳回)。

        decision 优先；兼容旧签名 review(doc_id, ok) —— ok=True 视为 accept，False 视为 reject。
        """
        if decision is None:
            decision = "accept" if (ok is None or ok) else "reject"
        decision = str(decision).lower()
        if decision not in ("accept", "reject"):
            return {"status": "error", "message": "decision 必须是 accept/reject"}

        meta = self._get_doc_meta(doc_id)
        if meta is None:
            return {"status": "error", "message": f"doc_id '{doc_id}' not found"}

        from_status = meta.get("status", "candidate")
        if decision == "accept":
            to_status = "accepted"
        else:
            to_status = "rejected"

        if from_status == to_status:
            # 已处于目标状态：不动索引，但保持返回体字段一致（vector_synced=None 表"未尝试写"）
            return {"status": "success", "doc_id": doc_id, "status_now": to_status,
                    "unchanged": True, "vector_synced": None, "vector_error": ""}

        meta["status"] = to_status
        meta["reviewed"] = True
        meta["reviewed_by"] = reviewer
        meta["reviewed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        if decision == "reject":
            meta["failed"] = True
        sync = await self._set_doc_meta(doc_id, meta)

        if self.ledger is not None:
            self.ledger.log_promotion(
                card_id=doc_id, from_status=from_status, to_status=to_status,
                gate_passed=True, reviewed_by=reviewer,
                note=f"review:{decision}",
            )
        logger.info("人审 %s：%s → %s (reviewer=%s)", doc_id, from_status, to_status, reviewer)
        out = {"status": "success", "doc_id": doc_id, "status_now": to_status,
               "from": from_status, "reviewed_by": reviewer}
        out.update(sync)
        if sync.get("vector_synced") is False:
            out["warning"] = ("已更新 BM25 语料，但向量库未同步（%s）——"
                              "两索引对该卡状态不一致，请重跑同步" % sync.get("vector_error"))
        return out

    async def activate(self, doc_id: str, signer: str = "") -> Dict[str, Any]:
        """签字生效：accepted → active。

        闸门（基于台账 source_run_ids）：
          1) 必须已处于 accepted（即已过人审）；
          2) 必须有签字（signer 非空）；
          3) doc 的 source_run_ids 必须全部已在台账 runs（无真值不得 active）。
        任一不满足均拒绝并留痕，绝不把卡置为 active。
        """
        meta = self._get_doc_meta(doc_id)
        if meta is None:
            return {"status": "error", "message": f"doc_id '{doc_id}' not found"}

        from_status = meta.get("status", "candidate")
        if from_status != "accepted":
            return {"status": "error",
                    "message": f"需先 review(accept) 才能签字生效；当前状态={from_status}",
                    "status_now": from_status}
        if not signer:
            # 无签字不得 active
            if self.ledger is not None:
                self.ledger.log_promotion(
                    card_id=doc_id, from_status=from_status, to_status="active",
                    gate_passed=False, missing_runs=[], signer="",
                    note="rejected: 无签字",
                )
            return {"status": "error", "message": "无签字不得 active（signer 必填）",
                    "status_now": from_status}

        # 真值闸门：source_run_ids 是否全部在台账 runs
        #
        # ⚠️ fail-closed 前置：**空的来源声明不可当作"没有要查的"而放行**。
        # `gate_check([])` 恒真 ⇒ 任何卡只要过审 + 有签字就能 active，
        # "无真值不得 active"（P0-1）会被静默打穿。
        # （2026-10-16 由阶段 3 端到端探针在"跑次无 run_id 的卡"上复现后补。）
        srcs = list(meta.get("source_run_ids") or [])
        if not srcs:
            if self.ledger is not None:
                self.ledger.log_promotion(
                    card_id=doc_id, from_status=from_status, to_status="active",
                    gate_passed=False, missing_runs=[], signer=signer,
                    note="rejected: 来源声明为空(source_run_ids=[])，无真值可证",
                )
            return {
                "status": "error",
                "message": ("闸门未过：该卡没有任何 source_run_ids 声明，无真值可证 "
                            "⇒ 不得 active（须先补来源并经 attest_run 登记真值）"),
                "status_now": from_status, "missing_runs": [],
                "reason": "empty_source_run_ids",
            }
        gate = {"passed": True, "missing": []}
        if self.ledger is not None:
            gate = self.ledger.gate_check(srcs)
        if not gate["passed"]:
            # 闸门未过也留痕，但绝不置 active
            self.ledger.log_promotion(
                card_id=doc_id, from_status=from_status, to_status="active",
                gate_passed=False, missing_runs=gate["missing"], signer=signer,
                note="rejected: 闸门未过(source_run_ids 未证实)",
            )
            hint = "；其中 %s 仅为卡内声明（未经真实来源证实）" % gate.get("self_claimed") \
                if gate.get("self_claimed") else ""
            return {
                "status": "error",
                "message": ("闸门未过：source_run_ids 未在台账中「已证实」"
                            f"，不得 active{hint}"),
                "status_now": from_status, "missing_runs": gate["missing"],
                "self_claimed": gate.get("self_claimed", []),
            }

        meta["status"] = "active"
        meta["signer"] = signer
        meta["activated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        # P0-1：签字生效 = 来源真值已证实，供复用时门下省闸门（vet_for_reuse）读取
        meta["provenance_attested"] = True
        sync = await self._set_doc_meta(doc_id, meta)

        if self.ledger is not None:
            self.ledger.log_promotion(
                card_id=doc_id, from_status=from_status, to_status="active",
                gate_passed=True, signer=signer,
                note="activate: 闸门通过+签字",
            )
        logger.info("签字生效 %s：accepted → active (signer=%s)", doc_id, signer)
        out = {"status": "success", "doc_id": doc_id, "status_now": "active",
               "signer": signer, "gate": gate}
        out.update(sync)
        if sync.get("vector_synced") is False:
            out["warning"] = ("已更新 BM25 语料，但向量库未同步（%s）——"
                              "两索引对该卡状态不一致，请重跑同步" % sync.get("vector_error"))
        return out

    async def ingest_doc(
        self, content: str, lib: str, level: str, case: str,
        source_of_truth: str = "", doc_id: Optional[str] = None,
        distilled: bool = False, machine_checkable: bool = False,
        confidence: float = 1.0, action: Optional[str] = None,
        symptom: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Web/工具入口：新增或覆盖一篇文档（自动归类到四库）。"""
        import hashlib
        lib = _coerce(lib, LIBS, "norm")
        level = _coerce(level, LEVELS, "总体")
        case = _coerce(case, CASES, "ALL")
        if not doc_id:
            doc_id = f"kb:{lib}:{hashlib.md5(content.encode('utf-8')).hexdigest()[:12]}"
        meta = self._build_meta(
            lib, level, case, "knowledge_base", source_of_truth or "",
            distilled=distilled, machine_checkable=machine_checkable,
            confidence=confidence, action=action, symptom=symptom, doc_id=doc_id,
        )
        ok = await self._index_vector(doc_id, content, meta)
        self.hybrid.add_document(doc_id, content, meta)
        return {"status": "success", "doc_id": doc_id, "vector": ok}

    async def delete_doc(self, doc_id: str) -> Dict[str, Any]:
        vec_ok = False
        if self.vector_enabled:
            try:
                vec_ok = bool(await self.rag_kb.delete_document(doc_id))
            except Exception as e:
                logger.warning(f"KnowledgeManager: 向量删除失败（忽略）：{e}")
        self.hybrid.remove_document(doc_id)
        return {"status": "success", "doc_id": doc_id, "vector": vec_ok}

    async def list_docs(
        self, lib: Optional[str] = None, level: Optional[str] = None,
        case: Optional[str] = None, q: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """浏览/筛选四库文档（来自 BM25 语料元信息，轻量）。"""
        lib = _coerce(lib, LIBS)
        level = _coerce(level, LEVELS)
        case = _coerce(case, CASES)
        out = []
        for c in self.hybrid._corpus:
            m = c.get("metadata", {}) or {}
            if lib and m.get("lib") != lib:
                continue
            if level and m.get("level") != level:
                continue
            if case and m.get("case") != case:
                continue
            if q:
                tags = m.get("tags") or ""
                if isinstance(tags, (list, tuple)):
                    tags = " ".join(str(t) for t in tags)
                hay = ((c.get("text") or "") + " " + str(tags)).lower()
                if q.lower() not in hay:
                    continue
            out.append({
                "doc_id": c.get("doc_id"),
                "lib": m.get("lib"), "level": m.get("level"),
                "case": m.get("case"), "confidence": m.get("confidence"),
                "reviewed": m.get("reviewed"), "failed": m.get("failed"),
                "distilled": m.get("distilled"),
                # P0-1：人审闸门状态必须在浏览视图里可见，否则管理端看不出"未审"
                "canonical_id": m.get("canonical_id", to_canonical_id(c.get("doc_id", ""))),
                "status": m.get("status"),
                "declared_status": m.get("declared_status"),
                "provenance_attested": m.get("provenance_attested"),
                "card_id": m.get("card_id"),
                "next_action_enum": m.get("next_action_enum"),
                "source_run_ids": m.get("source_run_ids"),
                # P0-2：失败回流视图——一成一败、绑哪个工况、配对的对手卡是谁
                "outcome": m.get("outcome"),
                "failed_scenarios": m.get("failed_scenarios"),
                "failed_under": normalize_failed_under(m.get("failed_under")),
                "paired_with": m.get("paired_with"),
                "snippet": (c.get("text", "") or "")[:160],
            })
        return out

    async def stats(self) -> Dict[str, Any]:
        base = await self.rag_kb.get_statistics()
        # vector_doc_count 取 chroma 侧真实数量（get_statistics 来自集合 count）；
        # 纯 BM25 模式下集合为 None，get_statistics 返回 {"total_documents": 0} → 0。
        vector_doc_count = base.get("total_documents", 0) or 0
        by_lib, by_level, by_case = {}, {}, {}
        for c in self.hybrid._corpus:
            m = c.get("metadata", {}) or {}
            by_lib[m.get("lib", "unknown")] = by_lib.get(m.get("lib", "unknown"), 0) + 1
            by_level[m.get("level", "unknown")] = by_level.get(m.get("level", "unknown"), 0) + 1
            by_case[m.get("case", "unknown")] = by_case.get(m.get("case", "unknown"), 0) + 1
        base.update({
            "by_lib": by_lib, "by_level": by_level, "by_case": by_case,
            # 无向量模式下 BM25 语料即为权威总数，避免前端显示 0 篇
            "total_documents": len(self.hybrid._corpus) or vector_doc_count,
            "vector_doc_count": vector_doc_count,
            "vector_enabled": bool(self.vector_enabled),
        })
        return base
