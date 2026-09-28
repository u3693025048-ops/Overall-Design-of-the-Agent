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
from pathlib import Path
from typing import Any, Dict, List, Optional

from .rag_knowledge_base import RAGKnowledgeBase

logger = logging.getLogger(__name__)


# ── 四库 / 层级 / 工况 字典 ────────────────────────────────────────────────
LIBS = ("norm", "model", "template", "experience")          # 规范/模型/模板/经验
LEVELS = ("总体", "几何-气动", "内回路", "外回路", "验证")      # 设计层级（手册第八步）
CASES = ("T", "G", "AP", "R", "ALL")                          # 工况（RUN_CASE 体系）
NEXT_ACTIONS = ("复用画布", "复用模板", "只调本层", "改结构", "升级人工")

# 经验卡 mode → 设计层级（用于把参数经验打到正确层级）
MODE_LEVEL = {
    "MODIFY_LAW": "外回路",
    "MODIFY_AUTOPILOT": "内回路",
    "MODIFY_CONTROLLER": "内回路",
    "MODIFY_GEOMETRY": "几何-气动",
    "TUNE": "内回路",
}


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
        fetch_k_mult: int = 3,
        kb_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """混合检索主入口。

        - doc_id 给定 → 直达（已知路径，不走语义检索）
        - 否则：向量 top-N + BM25 top-N → RRF 融合 → 硬过滤（level/case/lib）
        - 返回每条含 fused_score / vector_score / bm25_score / metadata
        """
        level = _coerce(level, LEVELS)
        case = _coerce(case, CASES)
        lib = _coerce(lib, LIBS)

        self._load_corpus()

        # 已知路径直达：优先查本地语料（同步、无向量依赖），必要时再回退向量库。
        if doc_id:
            doc = None
            for c in self._corpus:
                if c.get("doc_id") == doc_id:
                    doc = {
                        "doc_id": doc_id,
                        "content": c.get("text") or c.get("content") or "",
                        "metadata": c.get("metadata", {}) or {},
                    }
                    break
            if doc is None and self.rag_kb._collection is not None:
                try:
                    doc = await self.rag_kb.get_document(doc_id)
                except Exception:
                    doc = None
            if doc:
                doc = dict(doc)
                doc["direct"] = True
                doc["fused_score"] = 1.0
                doc["vector_score"] = None
                doc["bm25_score"] = None
                return [doc]
            return []

        fetch_k = max(top_k * fetch_k_mult, top_k)
        # 硬过滤下推到候选阶段：先算出合规 doc_id 集合，再让 BM25 只在该子集内排序。
        allowed_ids = None
        if lib or level or case:
            allowed_ids = set()
            for c in self._corpus:
                m = c.get("metadata", {}) or {}
                if lib and m.get("lib") != lib:
                    continue
                if level and m.get("level") != level:
                    continue
                if case and m.get("case") != case:
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
            # 硬过滤：level/case/lib 任一带过滤条件时，缺失或不匹配即剔除
            if lib and meta.get("lib") != lib:
                continue
            if level and meta.get("level") != level:
                continue
            if case and meta.get("case") != case:
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
    async def record_feedback(self, doc_id: str, success: bool) -> Dict[str, Any]:
        """成功升 confidence；失败降级，低于阈值标 failed。失败案例同等归档。"""
        doc = None
        if self.rag_kb._collection is not None:
            try:
                doc = await self.rag_kb.get_document(doc_id)
            except Exception:
                doc = None
        if doc:
            meta: Optional[Dict[str, Any]] = dict(doc.get("metadata", {}))
        else:
            # 无向量模式：从本地 BM25 语料取元信息，保证回流依然可用
            meta = None
            for c in self._corpus:
                if c.get("doc_id") == doc_id:
                    meta = dict(c.get("metadata", {}) or {})
                    break
        if meta is None:
            return {"status": "error", "message": f"doc_id '{doc_id}' not found"}
        conf = float(meta.get("confidence", 1.0) or 1.0)
        if success:
            conf = min(1.0, conf + 0.05)
            meta["failed"] = False
        else:
            conf = max(0.0, conf - 0.10)
            if conf < 0.2:
                meta["failed"] = True
        meta["confidence"] = round(conf, 3)
        await self._sync_update(doc_id, meta)
        # 同步 BM25 语料元数据
        for c in self._corpus:
            if c.get("doc_id") == doc_id:
                c["metadata"] = meta
        self._save_corpus()
        # 反馈日志（审计/论文可复现）
        try:
            with open(self._feedback_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(
                    {"doc_id": doc_id, "success": success, "confidence": meta["confidence"]},
                    ensure_ascii=False,
                ) + "\n")
        except Exception:
            pass
        return {"status": "success", "doc_id": doc_id, "confidence": meta["confidence"],
                "failed": meta.get("failed", False)}

    async def _sync_update(self, doc_id: str, meta: Dict[str, Any]) -> None:
        if self.rag_kb._collection is None:
            return                       # 无向量模式：元信息只落在 BM25 语料
        try:
            await self.rag_kb.update_document(doc_id, metadata=meta)
        except Exception as e:
            logger.warning(f"KBHybrid: update_document failed for {doc_id}: {e}")

    # ── 可机检下一动作 ──────────────────────────────────────────────────────
    async def propose_next_action(
        self,
        query: str,
        level: Optional[str] = None,
        case: Optional[str] = None,
        lib: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """默认产物是可机检的 next_action，而非长文（§2.7 口径①）。"""
        res = await self.hybrid_retrieve(query, top_k=top_k, level=level, case=case, lib=lib)
        if not res:
            return {"action": "升级人工", "doc_id": None,
                    "reason": "no retrieved context; escalate to human"}
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
        return {
            "action": action,
            "doc_id": top.get("doc_id"),
            "confidence": meta.get("confidence"),
            "reason": f"top hit lib={meta.get('lib')} level={meta.get('level')} case={meta.get('case')}",
        }


class KnowledgeManager:
    """知识库管理 Agent：四库编排 + 检索/入库/审核/回流/next_action。"""

    def __init__(
        self,
        rag_kb: RAGKnowledgeBase,
        bm25_corpus_path: str = "./kb_bm25_corpus.jsonl",
        vector_enabled: bool = True,
    ):
        self.rag_kb = rag_kb
        # vector_enabled=False → 纯 BM25 模式：全程不 import / 不调用 chromadb。
        # 目的：chromadb 的原生依赖在部分 Windows 机器上会让整个进程崩溃，
        # 而管理端本身不依赖向量检索，故提供一条完全绕开 chromadb 的可用路径。
        self.vector_enabled = vector_enabled
        self.hybrid = KBHybrid(rag_kb, bm25_corpus_path=bm25_corpus_path)

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
        counts = {"norm": 0, "model": 0, "template": 0, "experience": 0, "skipped": 0}
        manifest = self._load_manifest()

        # 1) 规范库：knowledge_base 下的 .md 说明文件
        for md in (base / "knowledge_base").glob("*.md"):
            if md.name.lower() in ("readme.md",):
                continue
            n = await self._ingest_file(md, lib="norm", level="总体", case="ALL",
                                        source_tag="knowledge_base", manifest=manifest)
            if n:
                counts["norm"] += n
            else:
                counts["skipped"] += 1

        # 2) 模型库：sysml / matlab / books（蒸馏书标 distilled=True，不自动计入"知识缩短周期"）
        for pat in ("knowledge_base/sysml/*", "knowledge_base/matlab/**/*"):
            for f in sorted(base.glob(pat)):
                if f.is_file() and f.suffix.lower() in (".xml", ".m", ".md", ".txt"):
                    n = await self._ingest_file(f, lib="model", level="总体", case="ALL",
                                                source_tag="knowledge_base", manifest=manifest)
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
                                                manifest=manifest)
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
                                                source_tag="knowledge_base", manifest=manifest)
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
                self._mark_manifest(pe_json, pe_id, manifest)

        self._save_manifest(manifest)
        # 收尾统一重建 BM25 + 落盘（覆盖本次批量变更）
        self.hybrid.rebuild(from_chroma=True)
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
    ) -> int:
        doc_id = f"kb:{lib}:{path.as_posix()}"
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
                                distilled=distilled)
        await self._index_vector(doc_id, text, meta)
        # 无论向量是否成功，都写入 BM25 语料（这是纯 BM25 模式的唯一索引，
        # 也保证 chromadb 坏掉时管理端依然能检索到入库内容）。
        self.hybrid.add_document(doc_id, text, meta, defer=True)
        if manifest is not None:
            self._mark_manifest(path, doc_id, manifest)
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
                action=action, symptom=mode,
            )
            await self._index_vector(doc_id, content, meta)
            self.hybrid.add_document(doc_id, content, meta, defer=defer)
            cnt += 1
        return cnt

    def _build_meta(
        self, lib: str, level: str, case: str, source_tag: str,
        source_of_truth: str, distilled: bool = False,
        machine_checkable: bool = False, confidence: float = 1.0,
        action: Optional[str] = None, symptom: Optional[str] = None,
        version: str = "v1",
    ) -> Dict[str, Any]:
        tags = [lib, level, case]
        if distilled:
            tags.append("distilled")
        return {
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
            "tags": ",".join(tags),
            **({"action": action} if action else {}),
            **({"symptom": symptom} if symptom else {}),
        }

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

        doc_id 支持精确匹配、路径后缀匹配、裸文件名匹配——覆盖 LLM 给出的
        相对路径（file:knowledge_base/xxx.md）与向量库存储的绝对路径不一致的情况。
        """
        def _norm(s: str) -> str:
            return s.replace("\\", "/").replace("\\\\", "/").lower()

        needle = _norm(doc_id)
        if needle.startswith("file:"):
            needle = needle[5:]

        # 1) 精确匹配（本地语料）
        for c in self.hybrid._corpus:
            if _norm(c.get("doc_id", "")) == needle:
                return {"doc_id": c["doc_id"], "content": c.get("text", ""), "metadata": c.get("metadata", {}) or {}}

        # 2) 后缀/裸文件名匹配（本地语料）
        needle_name = needle.split("/")[-1]
        for c in self.hybrid._corpus:
            cid = _norm(c.get("doc_id", ""))
            if cid.endswith(needle) or cid.endswith(needle_name):
                return {"doc_id": c["doc_id"], "content": c.get("text", ""), "metadata": c.get("metadata", {}) or {}}

        # 3) 回落向量库（已就绪时）
        if self.vector_enabled:
            try:
                return await self.rag_kb.get_document(doc_id)
            except Exception:
                return None
        return None

    async def next_action(self, query: str, level=None, case=None, lib=None) -> Dict[str, Any]:
        return await self.hybrid.propose_next_action(query, level=level, case=case, lib=lib)

    async def feedback(self, doc_id: str, success: bool) -> Dict[str, Any]:
        return await self.hybrid.record_feedback(doc_id, success)

    async def review(self, doc_id: str, ok: bool) -> Dict[str, Any]:
        # 元信息优先取向量库；无向量模式则从本地 BM25 语料取，保证审核可用。
        meta: Optional[Dict[str, Any]] = None
        if self.vector_enabled:
            try:
                doc = await self.rag_kb.get_document(doc_id)
                if doc:
                    meta = dict(doc.get("metadata", {}) or {})
            except Exception:
                meta = None
        if meta is None:
            for c in self.hybrid._corpus:
                if c.get("doc_id") == doc_id:
                    meta = dict(c.get("metadata", {}) or {})
                    break
        if meta is None:
            return {"status": "error", "message": f"doc_id '{doc_id}' not found"}

        meta["reviewed"] = bool(ok)
        meta["failed"] = False if ok else meta.get("failed", False)

        # 写回 BM25 语料（始终生效）
        for c in self.hybrid._corpus:
            if c.get("doc_id") == doc_id:
                c["metadata"] = meta
                break
        self.hybrid._save_corpus()

        if self.vector_enabled:
            try:
                await self.rag_kb.update_document(doc_id, metadata=meta)
            except Exception as e:
                logger.warning(f"KnowledgeManager: 审核回写向量库失败（忽略）：{e}")
        return {"status": "success", "doc_id": doc_id, "reviewed": bool(ok)}

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
            confidence=confidence, action=action, symptom=symptom,
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
