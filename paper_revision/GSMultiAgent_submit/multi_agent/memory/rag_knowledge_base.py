#!/usr/bin/env python3
"""
RAG Knowledge Base
基于 ChromaDB 的向量知识库，支持可配置的 Embedding 模型
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""

    provider: str = "local"  # "local", "openai", "anthropic"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 本地模型
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimension: int = 384


@dataclass
class RAGConfig:
    """RAG 知识库配置"""

    persist_directory: str = "./chroma_db"
    collection_name: str = "knowledge_base"
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)


# Max characters to send per document to the embedding API.
# DashScope text-embedding-v3 limit: 8192 tokens ≈ 24 000 chars for code.
# We use a conservative 6000 to stay safely under and handle tokeniser variance.
_MAX_EMBED_CHARS = 6000


class RAGKnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库

    支持多种 Embedding 模型：
    - local: 使用 sentence-transformers 本地模型
    - openai: 使用 OpenAI Embedding API
    - anthropic: 使用 Anthropic Embedding API
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        from ..config_loader import get_config

        app_cfg = get_config()
        rag_cfg = app_cfg.rag
        llm_cfg = app_cfg.llm

        if config is None:
            config = RAGConfig(
                persist_directory=rag_cfg.persist_dir,
                collection_name=rag_cfg.collection,
            )

        if embedding_config is None:
            # 优先从 RAG 配置中取，如果 provider 没有指定 api_key/base_url，则 fallback 到全局大模型的 API_KEY/BASE_URL
            embedding_config = EmbeddingConfig(
                provider=rag_cfg.embedding_provider,
                model=rag_cfg.embedding_model,
                api_key=rag_cfg.embedding_api_key or llm_cfg.api_key, 
                base_url=rag_cfg.embedding_base_url or llm_cfg.base_url,
                dimension=rag_cfg.embedding_dim,
            )

        self.config = config
        self.embedding_config = embedding_config
        self._client = None
        self._collection = None
        self._embedding_function = None

    # ------------------------------------------------------------------
    # Async helper: run any synchronous callable in a thread-pool executor
    # so it never blocks the asyncio event loop.  A hard timeout (default
    # 30 s) prevents hangs when the embedding / Chroma API is unreachable.
    # ------------------------------------------------------------------
    @staticmethod
    async def _blocking(
        fn: Callable,
        *args,
        timeout: float = 30.0,
        **kwargs,
    ):
        loop = asyncio.get_event_loop()
        coro = loop.run_in_executor(None, lambda: fn(*args, **kwargs))
        return await asyncio.wait_for(coro, timeout=timeout)

    async def initialize(self) -> bool:
        """初始化 ChromaDB 和 Embedding 模型"""
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=self.config.persist_directory, settings=Settings(anonymized_telemetry=False)
            )

            self._embedding_function = self._create_embedding_function()

            # Validate before committing: a bad model name / expired key causes
            # collection.upsert() to fail silently, leaving 0 indexed documents.
            if self._embedding_function is not None:
                try:
                    await self._blocking(self._embedding_function, ["ok"], timeout=15.0)
                except Exception as emb_exc:
                    logger.error(
                        f"Embedding API test failed "
                        f"({self.embedding_config.provider}/{self.embedding_config.model}): "
                        f"{emb_exc}\n"
                        f"  ► Fix: check rag.embedding_model / rag.embedding_api_key "
                        f"/ rag.embedding_base_url in config.yaml\n"
                        f"  ► DashScope users: change embedding_model to text-embedding-v3\n"
                        f"  ► Falling back to local sentence-transformers."
                    )
                    self._embedding_function = self._create_local_embedding_function()

            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=self._embedding_function,
                metadata={"description": "RAG Knowledge Base"},
            )

            logger.info(
                f"Initialized RAG with {self.embedding_config.provider}/{self.embedding_config.model}"
            )
            return True

        except ImportError:
            logger.error("ChromaDB not installed. Run: pip install chromadb sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            return False

    def _create_embedding_function(self):
        """创建 Embedding 函数"""
        if self.embedding_config.provider == "local":
            return self._create_local_embedding_function()
        elif self.embedding_config.provider == "openai":
            return self._create_openai_embedding_function()
        elif self.embedding_config.provider == "anthropic":
            return self._create_anthropic_embedding_function()
        elif self.embedding_config.provider == "custom":
            return self._create_custom_embedding_function()
        else:
            return self._create_local_embedding_function()

    def _create_local_embedding_function(self):
        """创建本地 Embedding 函数"""
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            return SentenceTransformerEmbeddingFunction(model_name=self.embedding_config.model)
        except ImportError:
            logger.warning("sentence-transformers not installed, using default")
            return None

    def _create_openai_embedding_function(self):
        """创建 OpenAI Embedding 函数"""
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

            return OpenAIEmbeddingFunction(
                api_key=self.embedding_config.api_key or os.getenv("OPENAI_API_KEY"),
                model_name=self.embedding_config.model or "text-embedding-3-small",
            )
        except ImportError:
            logger.error("OpenAI embedding not available")
            return None

    def _create_custom_embedding_function(self):
        """创建自定义 OpenAI-Like Embedding 函数"""
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

            base_url = self.embedding_config.base_url or os.getenv("EMBEDDING_BASE_URL")
            if not base_url:
                logger.warning("Custom embedding provider requires EMBEDDING_BASE_URL, falling back to local")
                return self._create_local_embedding_function()

            # For chromadb OpenAIEmbeddingFunction, we might need to set openai.api_base directly 
            # or pass api_base instead of base_url depending on chromadb version.
            # Usually, chromadb uses `api_base` or `api_base` parameter in OpenAIEmbeddingFunction.
            return OpenAIEmbeddingFunction(
                api_key=self.embedding_config.api_key,
                model_name=self.embedding_config.model or "text-embedding-3-small",
                api_base=base_url,
            )
        except ImportError:
            logger.error("OpenAI embedding function not available")
            return None

    def _create_anthropic_embedding_function(self):
        """创建 Anthropic Embedding 函数"""
        logger.warning("Anthropic does not provide standalone embedding API, falling back to local")
        return self._create_local_embedding_function()

    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None,
    ) -> int:
        """索引文档到 ChromaDB"""
        if self._collection is None:
            await self.initialize()

        if self._collection is None:
            return 0

        try:
            ids = []
            contents = []
            metas = []

            for i, doc_data in enumerate(documents):
                content = doc_data.get("content", "")
                # Guard: truncate at index time in case callers don't go through
                # index_directory() (e.g. direct index_documents() calls).
                if len(content) > _MAX_EMBED_CHARS:
                    content = content[:_MAX_EMBED_CHARS]
                doc_metadata = {**(metadata or {}), **doc_data.get("metadata", {})}
                doc_id = doc_data.get("id") or f"doc_{i}_{hash(content) % 100000}"
                ids.append(doc_id)
                contents.append(content)
                metas.append(doc_metadata)

            # DashScope embedding API: at most 10 items per call.
            # upsert() calls the embedding API synchronously — use _blocking().
            _UPSERT_CHUNK = 10
            ok = 0
            for _start in range(0, len(ids), _UPSERT_CHUNK):
                _sl = slice(_start, _start + _UPSERT_CHUNK)
                _u_ids = ids[_sl]
                _u_docs = contents[_sl]
                _u_metas = metas[_sl]
                try:
                    await self._blocking(
                        self._collection.upsert,
                        ids=_u_ids, documents=_u_docs, metadatas=_u_metas,
                        timeout=30.0,
                    )
                    ok += len(_u_ids)
                except Exception as chunk_exc:
                    logger.warning(
                        f"Chunk upsert failed (docs {_start}-{_start + _UPSERT_CHUNK}): "
                        f"{chunk_exc}; retrying one by one."
                    )
                    for _doc_id, _content, _meta in zip(_u_ids, _u_docs, _u_metas):
                        try:
                            await self._blocking(
                                self._collection.upsert,
                                ids=[_doc_id], documents=[_content], metadatas=[_meta],
                                timeout=30.0,
                            )
                            ok += 1
                        except Exception as single_exc:
                            logger.warning(f"  Skipping doc '{_doc_id}': {single_exc}")
            logger.info(f"Indexed {ok}/{len(documents)} documents")
            return ok

        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return 0

    async def index_directory(
        self,
        directory: str,
        patterns: Optional[List[str]] = None,
        source_tag: Optional[str] = None,
    ) -> int:
        """
        Index all matching files from a directory into the knowledge base.

        Parameters
        ----------
        directory : path to scan (recursively)
        patterns  : list of file suffixes to include, e.g. [".json", ".md", ".txt"]
                    Defaults to [".json", ".md", ".txt", ".m"]
        """
        import json as _json
        if patterns is None:
            patterns = [".json", ".md", ".txt", ".m"]

        dir_path = Path(directory)
        if not dir_path.exists():
            logger.info(f"index_directory: {directory} does not exist, skipping.")
            return 0

        documents = []
        for file_path in sorted(dir_path.rglob("*")):
            if not file_path.is_file() or file_path.suffix not in patterns:
                continue
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                if not content.strip():
                    continue
                # Truncate oversized documents so they fit within the embedding
                # API's token limit (DashScope v3: 8192 tok ≈ 24K chars).
                if len(content) > _MAX_EMBED_CHARS:
                    logger.debug(
                        f"index_directory: truncating '{file_path.name}' "
                        f"({len(content)} → {_MAX_EMBED_CHARS} chars) for embedding."
                    )
                    content = content[:_MAX_EMBED_CHARS]
                # Pretty-print JSON for better embedding quality
                if file_path.suffix == ".json":
                    try:
                        content = _json.dumps(_json.loads(content), ensure_ascii=False, indent=2)
                    except _json.JSONDecodeError:
                        pass
                # ── Auto-tagging by path and filename ────────────────────────
                # Priority (first match wins):
                #   1. Path contains parameter_experience_base or guidance_output
                #      → "pe_result"   (excluded by kb_only filter)
                #   2. Path contains matlab_scripts
                #      → "intermediate_output" (excluded by kb_only filter)
                #   3. Filename matches known PE JSON patterns
                #      → "pe_result"
                #   4. Caller-supplied source_tag, or default "knowledge_base"
                _path_parts = {p.lower() for p in file_path.parts}
                _name_lower = file_path.name.lower()

                _is_pe_path = bool(
                    _path_parts & {"parameter_experience_base", "guidance_output"}
                )
                _is_matlab_scripts = "matlab_scripts" in _path_parts
                _is_pe_name = (
                    file_path.suffix == ".json"
                    and any(
                        tok in _name_lower
                        for tok in ("pe_", "param_exp", "experience", "rl_result",
                                    "best_params", "pe_record")
                    )
                )

                if _is_pe_path or _is_pe_name:
                    _effective_tag = "pe_result"
                    logger.warning(
                        f"index_directory: '{file_path}' → tagged 'pe_result' "
                        f"(path={_is_pe_path}, name={_is_pe_name}); excluded from KB."
                    )
                elif _is_matlab_scripts:
                    _effective_tag = "intermediate_output"
                    logger.warning(
                        f"index_directory: '{file_path}' → tagged 'intermediate_output' "
                        f"(matlab_scripts path); excluded from KB."
                    )
                else:
                    _effective_tag = source_tag or "knowledge_base"
                documents.append({
                    "id": f"file:{file_path}",
                    "content": content,
                    "metadata": {
                        "source":     str(file_path),
                        "source_tag": _effective_tag,
                        "filename":   file_path.name,
                        "type":       file_path.suffix.lstrip("."),
                        "dir":        str(file_path.parent),
                    },
                })
            except Exception as exc:
                logger.warning(f"index_directory: could not read {file_path}: {exc}")

        if not documents:
            return 0

        count = await self.index_documents(documents)
        logger.info(f"index_directory: indexed {count} docs from '{directory}'")
        return count

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None,
        kb_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """检索相关文档

        Parameters
        ----------
        kb_only : bool, default True
            When True, apply a strict Python-side post-filter that keeps only
            documents whose ``metadata.source_tag == 'knowledge_base'``.
            Documents without a ``source_tag`` key (e.g. PE JSON files indexed
            by older runs before the source_tag fix) are excluded.
            Pass ``kb_only=False`` only when you intentionally need all docs.
        """
        if self._collection is None:
            await self.initialize()

        if self._collection is None:
            return []

        try:
            where_filter = filters if filters else None

            # count() and query() call the embedding API synchronously.
            # Use _blocking() so the asyncio event loop is never blocked.
            try:
                _col_size = await self._blocking(
                    self._collection.count, timeout=10.0
                )
            except Exception:
                _col_size = top_k
            fetch_k = min((top_k * 3) if kb_only else top_k, max(1, _col_size))

            results = await self._blocking(
                self._collection.query,
                query_texts=[query],
                n_results=fetch_k,
                where=where_filter,
                timeout=30.0,
            )

            retrieved = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distance = results["distances"][0][i] if results.get("distances") else 0.0

                    retrieved.append(
                        {
                            "doc_id": results["ids"][0][i],
                            "content": doc,
                            "score": 1.0 - distance if distance else 0.0,
                            "metadata": metadata,
                        }
                    )

            # ── Knowledge-base-only post-filter ──────────────────────────────
            # Lenient check: keep docs that are explicitly tagged as
            # "knowledge_base" OR that have no source_tag at all (backward
            # compatibility — KB docs indexed before the tag fix).
            # Only exclude docs explicitly tagged as something else (e.g. a
            # future "pe_result" tag added in error).
            if kb_only:
                _NON_KB_TAGS = {"pe_result", "intermediate_output"}
                retrieved = [
                    r for r in retrieved
                    if r.get("metadata", {}).get("source_tag") not in _NON_KB_TAGS
                ]

            return retrieved[:top_k]

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []

    async def similarity_search(
        self,
        query: str,
        threshold: float = 0.7,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """相似性搜索"""
        results = await self.retrieve(query, top_k=top_k * 2)
        filtered = [r for r in results if r["score"] >= threshold]
        return filtered[:top_k]

    def purge_stale_pe_docs(self) -> int:
        """
        Remove stale PE / intermediate-output documents from the ChromaDB
        collection.  These documents may have been indexed by older code
        versions that did not set ``source_tag`` correctly, causing them to
        appear in ``rag_retrieve`` results even when ``parameter_experience_reuse``
        and ``memory_search`` ablation flags are disabled.

        Targets (any match → delete):
        1. Documents explicitly tagged ``pe_result`` or ``intermediate_output``
           (already flagged by auto-tagging; still need removal from DB).
        2. Documents whose ``source`` path contains ``parameter_experience_base``
           (stale KB-tagged PE docs from before the source_tag fix).
        3. Documents whose ``source`` path contains ``matlab_scripts``
           (stale intermediate MATLAB temp files).
        4. Documents whose filename matches known PE JSON patterns and suffix is
           ``.json`` (belt-and-suspenders guard).

        Returns the number of documents deleted.
        """
        if self._collection is None:
            return 0

        _NON_KB_TAGS   = {"pe_result", "intermediate_output"}
        _PE_PATH_TOKENS = {"parameter_experience_base", "matlab_scripts"}
        _PE_NAME_TOKENS = ("pe_", "param_exp", "experience", "rl_result",
                           "best_params", "pe_record")

        to_delete: list = []
        try:
            total = self._collection.count()
            if total == 0:
                return 0

            batch, offset = 500, 0
            while offset < total:
                res = self._collection.get(
                    limit=batch,
                    offset=offset,
                    include=["metadatas"],
                )
                ids   = res.get("ids") or []
                metas = res.get("metadatas") or []
                for doc_id, meta in zip(ids, metas):
                    meta  = meta or {}
                    tag   = meta.get("source_tag", "")
                    source = meta.get("source", "")
                    fname  = meta.get("filename", "").lower()

                    if tag in _NON_KB_TAGS:
                        to_delete.append(doc_id)
                        continue

                    path_parts = set()
                    if source:
                        try:
                            path_parts = {p.lower() for p in Path(source).parts}
                        except Exception:
                            pass
                    if path_parts & _PE_PATH_TOKENS:
                        to_delete.append(doc_id)
                        continue

                    if fname.endswith(".json") and any(tok in fname for tok in _PE_NAME_TOKENS):
                        to_delete.append(doc_id)

                offset += batch

            if to_delete:
                self._collection.delete(ids=to_delete)
                logger.info(
                    f"purge_stale_pe_docs: removed {len(to_delete)} "
                    f"non-KB doc(s) from collection."
                )
        except Exception as exc:
            logger.error(f"purge_stale_pe_docs failed: {exc}")

        return len(to_delete)

    def audit_tags(self) -> Dict[str, Any]:
        """Scan the entire collection and return a tag distribution summary.

        Returns a dict with:
            total        : total document count
            by_tag       : {tag_value: count}
            non_kb_files : list of (filename, source_tag) for non-KB documents
        """
        if self._collection is None:
            return {"total": 0, "by_tag": {}, "non_kb_files": []}

        _NON_KB_TAGS = {"pe_result", "intermediate_output"}
        summary: Dict[str, Any] = {"total": 0, "by_tag": {}, "non_kb_files": []}

        try:
            total = self._collection.count()
            summary["total"] = total
            if total == 0:
                return summary

            # Fetch all metadata in batches (Chroma has no native group-by)
            batch = 500
            offset = 0
            while offset < total:
                res = self._collection.get(
                    limit=batch,
                    offset=offset,
                    include=["metadatas"],
                )
                for meta in (res.get("metadatas") or []):
                    tag = (meta or {}).get("source_tag") or "knowledge_base"
                    summary["by_tag"][tag] = summary["by_tag"].get(tag, 0) + 1
                    if tag in _NON_KB_TAGS:
                        fname = (meta or {}).get("filename", "?")
                        summary["non_kb_files"].append((fname, tag))
                offset += batch

        except Exception as exc:
            logger.error(f"audit_tags failed: {exc}")

        return summary

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取指定文档.

        Falls back to suffix/filename matching when the exact doc_id is not
        found.  This handles the common case where the LLM provides a
        relative path (e.g. "file:knowledge_base\\\\monte mean results.md")
        while ChromaDB stores absolute paths
        (e.g. "file:C:\\Users\\...\\knowledge_base\\monte mean results.md").
        """
        if self._collection is None:
            return None

        try:
            result = self._collection.get(ids=[doc_id])
            if result and result.get("documents"):
                return {
                    "doc_id": doc_id,
                    "content": result["documents"][0],
                    "metadata": result["metadatas"][0] if result.get("metadatas") else {},
                }
        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return None

        # ── Fuzzy fallback: match by path suffix or bare filename ─────────────
        # Normalise separators so "knowledge_base\\\\monte mean results.md"
        # and "knowledge_base/monte mean results.md" both work.
        try:
            _needle = doc_id.replace("\\\\", "/").replace("\\", "/").lower()
            # Strip the "file:" scheme prefix for suffix comparison
            if _needle.startswith("file:"):
                _needle = _needle[5:]
            # Also accept bare filename without any path prefix
            _needle_name = _needle.split("/")[-1]

            all_ids_result = self._collection.get(include=[])
            all_ids = all_ids_result.get("ids", []) if all_ids_result else []
            matched_id = None
            for cid in all_ids:
                _cid_norm = cid.replace("\\", "/").lower()
                if _cid_norm.endswith(_needle) or _cid_norm.endswith(_needle_name):
                    matched_id = cid
                    break

            if matched_id:
                logger.info(
                    f"get_document: fuzzy match '{doc_id}' → '{matched_id}'"
                )
                result2 = self._collection.get(ids=[matched_id])
                if result2 and result2.get("documents"):
                    return {
                        "doc_id": matched_id,
                        "content": result2["documents"][0],
                        "metadata": result2["metadatas"][0] if result2.get("metadatas") else {},
                    }
        except Exception as e2:
            logger.error(f"get_document fuzzy fallback failed: {e2}")

        return None

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        if self._collection is None:
            return False

        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    async def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新文档"""
        if self._collection is None:
            return False

        try:
            update_data = {"id": doc_id}
            if content is not None:
                update_data["documents"] = [content]
            if metadata is not None:
                update_data["metadatas"] = [metadata]

            self._collection.update(**update_data)
            return True
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self._collection is None:
            return {"total_documents": 0}

        try:
            count = self._collection.count()
            return {
                "total_documents": count,
                "collection_name": self.config.collection_name,
                "persist_directory": self.config.persist_directory,
                "embedding_provider": self.embedding_config.provider,
                "embedding_model": self.embedding_config.model,
                "embedding_dimension": self.embedding_config.dimension,
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}

    async def clear(self) -> bool:
        """清空知识库"""
        if self._collection is None:
            return False

        try:
            self._client.delete_collection(self.config.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=self._embedding_function,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self._client:
            self._client = None
            self._collection = None
