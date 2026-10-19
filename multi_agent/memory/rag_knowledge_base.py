#!/usr/bin/env python3
"""
RAG Knowledge Base
基于 ChromaDB 的向量知识库，支持可配置的 Embedding 模型
"""

import asyncio
import json
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


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """把 metadata 值域压成 chromadb 接受的标量，作为**入库边界的最后一道防线**。

    背景：chromadb 0.4.x 的 validate_metadata 只放行 str/int/float/bool，任何
    list / dict / None 会让**整批** upsert 失败（`Expected metadata value to be
    a str, int, float or bool, got [...] which is a <class 'list'>`），退化到逐篇
    重试后这些文档被静默丢弃——正是 kb:experience:E001/E002 丢向量的原因。

    只在这里（写入 Chroma 前）做转换，不动调用方的原始 dict：业务侧
    （如 ledger.gate_check / 人审闸门）仍拿到结构化的 list / dict。

    list/tuple/set -> 逗号连接的字符串
    dict           -> JSON 字符串（保持可读、可全文匹配）
    None           -> 丢弃该键（0.4.x 不接受 None）
    numpy 标量     -> 原生 Python 标量（0.4.x 不认识 np.float64 等）
    """
    out: Dict[str, Any] = {}
    for key, val in (meta or {}).items():
        key = str(key)
        # numpy / pandas 标量 -> 原生类型（np.float64 不是 Python float 的子类问题）
        item = getattr(val, "item", None)
        if callable(item) and not isinstance(val, (str, bytes)):
            try:
                val = val.item()
            except Exception:
                pass
        if val is None:
            continue
        # bool 必须排在 int 之前：isinstance(True, int) 为 True
        if isinstance(val, bool) or isinstance(val, (str, int, float)):
            out[key] = val
        elif isinstance(val, (list, tuple, set, frozenset)):
            out[key] = ",".join(str(x) for x in val)
        elif isinstance(val, dict):
            try:
                out[key] = json.dumps(val, ensure_ascii=False)
            except Exception:
                out[key] = str(val)
        else:
            out[key] = str(val)
    return out


class OpenAICompatibleEmbeddingFunction:
    """自实现的 OpenAI 兼容 Embedding 函数（标准库 urllib，零 openai 依赖）。

    chromadb 调用约定：传入 ``List[str]``，返回 ``List[List[float]]``。
    适配 DashScope compatible-mode
    （``https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings``）等任意
    OpenAI 格式 ``/embeddings`` 端点，与服务器装了 openai>=1.0 还是 <1.0 完全无关。
    """

    def __init__(self, api_key: str, model: str, api_base: str, dimensions: int = None):
        self.api_key = api_key
        self.model = model
        self.dimensions = int(dimensions) if dimensions else None
        base = (api_base or "").rstrip("/")
        self.url = f"{base}/embeddings" if base else "https://api.openai.com/v1/embeddings"

    def __call__(self, input):
        # 注意：chromadb 0.4.x 用 *参数名* 校验 EmbeddingFunction 接口，
        # 第一个业务参数必须叫 `input`（见 chromadb 的 EmbeddingFunction 协议），
        # 命名为 texts 会触发 "Expected ... signature ... ['self','input']" 报错。
        if isinstance(input, str):
            input = [input]
        texts = list(input)
        if not texts:
            return []
        out: List[List[float]] = []
        # 分批（DashScope /embeddings 建议单次 ≤ 若干条；保守 10 条/批）
        for i in range(0, len(texts), 10):
            out.extend(self._embed_batch(texts[i:i + 10]))
        return out

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        import json
        import urllib.error
        import urllib.request

        body = {"model": self.model, "input": batch}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:800]
            except Exception:
                pass
            raise RuntimeError(
                f"Embedding API HTTP {e.code} {getattr(e, 'reason', '')}: {detail}"
            ) from e
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Embedding API 请求失败: {exc}") from exc
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Embedding API 返回非 JSON: {raw[:200]}") from exc
        items = payload.get("data") or []
        if not items:
            msg = payload.get("message") or payload.get("error") or "empty data"
            raise RuntimeError(f"Embedding API 返回为空: {msg}")
        # 按 index 排序，保证与输入顺序一致（部分服务商不保序）
        items = sorted(items, key=lambda d: d.get("index", 0))
        return [list(d["embedding"]) for d in items]


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
        # 一次性初始化失败标记：initialize() 一旦失败（如 NumPy 版本不兼容、
        # embedding 不可达），本进程内不再对每篇文档重复尝试，避免日志被刷爆。
        self._init_failed = False

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
        """初始化 ChromaDB 和 Embedding 模型。

        关键：Chroma 客户端/集合创建是同步阻塞调用，必须放进线程池并加超时，
        否则一旦卡住会占住事件循环，导致 ensure_ready 的 wait_for 超时失效、
        Web 启动被拖死。每步打印日志，便于定位卡点。
        """
        try:
            import sys
            if "onnxruntime" not in sys.modules:
                try:
                    from unittest import mock
                    # 占位：本项目用 DashScope API 做 embedding，从不使用 chromadb
                    # 的默认 ONNX MiniLM embedding；但 chromadb 在 import 时会 eager
                    # 加载 onnxruntime 原生扩展，在部分 Windows 上会触发 access
                    # violation 直接拖死进程。用 MagicMock 替换，使其 import 不再加载
                    # 原生 .pyd（我们不使用它，无副作用）。对 chromadb 0.4.x / 0.5.x 均有效。
                    sys.modules["onnxruntime"] = mock.MagicMock()
                except Exception:
                    pass

            # NumPy 2.0 彻底移除了若干旧类型别名（np.float_ / np.complex_ /
            # np.object_ / np.unicode_ / np.long_ 等）。chromadb 0.4.x 及其依赖
            #（pypika-fork）仍会引用 np.float_，在 NumPy 2.0 下抛
            # "np.float_ was removed in the NumPy 2.0 release. Use np.float64
            # instead."，导致向量初始化失败。这里在 numpy 模块对象上全局恢复这些
            # 别名，使第三方包在 NumPy 2.0 下仍可用——无需在服务器上把 numpy 降级
            # 到 <2（降级可能破坏其它依赖 NumPy 2.0 的库）。别名目标均为官方推荐替身。
            try:
                import numpy as _np
                _np_legacy_aliases = {
                    "float_": _np.float64,
                    "complex_": _np.complex128,
                    "object_": object,
                    "unicode_": _np.str_,
                    "long_": _np.int64,
                    "int_": _np.int64,
                    "bool_": _np.bool_,
                    "str_": _np.str_,
                }
                for _name, _target in _np_legacy_aliases.items():
                    if not hasattr(_np, _name):
                        setattr(_np, _name, _target)
            except Exception:
                pass

            import chromadb
            from chromadb.config import Settings

            logger.info("[kb] 创建 Chroma 客户端（persist_directory=%s）…", self.config.persist_directory)
            self._client = await self._blocking(
                lambda: chromadb.PersistentClient(
                    path=self.config.persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                ),
                timeout=30.0,
            )
            logger.info("[kb] Chroma 客户端就绪")

            self._embedding_function = self._create_embedding_function()
            logger.info("[kb] embedding 函数已创建（provider=%s）", self.embedding_config.provider)

            # Validate before committing: a bad model name / expired key causes
            # collection.upsert() to fail silently, leaving 0 indexed documents.
            if self._embedding_function is not None:
                logger.info("[kb] 探测 Embedding API（15s 超时）…")
                try:
                    await self._blocking(self._embedding_function, ["ok"], timeout=15.0)
                except Exception as emb_exc:
                    logger.error(
                        f"Embedding API 探测失败 "
                        f"({self.embedding_config.provider}/{self.embedding_config.model}): "
                        f"{emb_exc}\n"
                        f"  ► 知识库向量检索将不可用，但 Web 管理端仍会启动。\n"
                        f"  ► 请检查 config.yaml 的 rag.embedding_api_key / rag.embedding_base_url 是否可达。\n"
                        f"  ► 该知识库向量由 DashScope(text-embedding-v3) 生成，不可用本地模型兜底"
                        f"（维度不兼容，会导致静默 0 结果），故不再下载本地大模型。"
                    )
                    # 关键修复：不要回退下载本地 sentence-transformers 模型。
                    # 否则在 huggingface.co 不可达时会长时间卡死，且本地模型维度与
                    # 已建好的 chroma_db 向量不兼容。直接判定向量检索不可用。
                    self._embedding_function = None
                    self._collection = None
                    self._init_failed = True
                    return False

            if self._embedding_function is None:
                # embedding 不可用：不创建 collection，检索方法会安全返回空。
                # 服务端仍可启动，便于在 embedding 配置就绪后重启。
                logger.warning(
                    "Embedding 不可用：向量检索已关闭。Web 管理端正常启动，"
                    "但搜索/索引将返回空，直到 embedding 配置可用。"
                )
                self._init_failed = True
                return False

            logger.info("[kb] 创建/获取 Chroma 集合（%s）…", self.config.collection_name)
            self._collection = await self._blocking(
                lambda: self._client.get_or_create_collection(
                    name=self.config.collection_name,
                    embedding_function=self._embedding_function,
                    metadata={"description": "RAG Knowledge Base"},
                ),
                timeout=30.0,
            )
            logger.info("[kb] Chroma 集合就绪")

            logger.info(
                f"Initialized RAG with {self.embedding_config.provider}/{self.embedding_config.model}"
            )
            self._init_failed = False
            return True

        except ImportError:
            logger.error("ChromaDB not installed. Run: pip install chromadb sentence-transformers")
            self._init_failed = True
            return False
        except asyncio.TimeoutError:
            logger.error(
                "initialize 超时（Chroma 客户端/集合创建 或 Embedding 探测 卡住）。"
                "Web 端仍启动，检索为空。"
            )
            self._init_failed = True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            self._init_failed = True
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
        except ImportError:
            logger.warning("sentence-transformers not installed, using default")
            return None
        model = self.embedding_config.model or ""
        # 回退场景：provider 为 custom/openai 时 model 可能是 API 模型名
        # （如 text-embedding-v3 / text-embedding-3-small），不能直接当 HF 模型下载，
        # 否则会去 HuggingFace 拉一个不存在/拉不到的模型并长时间卡死。
        # 用中文友好的多语 ST 模型兜底，确保本地回退可正常下载与推理。
        if "/" not in model or model.startswith("text-embedding"):
            model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        return SentenceTransformerEmbeddingFunction(model_name=model)

    def _create_openai_compatible_embedding_function(self):
        """用标准库 urllib 直连 OpenAI 兼容 /embeddings 端点（如 DashScope）。

        为什么不复用 chromadb 自带的 OpenAIEmbeddingFunction：
        chromadb 0.4.x 的 OpenAIEmbeddingFunction 内部调用 `openai.Embedding.create`，
        那是 openai <1.0 的旧接口。服务器若装了 openai>=1.0，会直接抛
        "You tried to access openai.Embedding, but this is no longer supported in
        openai>=1.0.0"。本项目用 urllib 自己打 /embeddings，与 openai 版本彻底解耦，
        且零额外依赖（DashScope 的 compatible-mode 即标准 OpenAI 格式）。
        """
        api_key = self.embedding_config.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("EMBEDDING_API_KEY")
        model = self.embedding_config.model or "text-embedding-3-small"
        base_url = (
            self.embedding_config.base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )
        if not api_key:
            logger.error("Embedding 需要 api_key（config.yaml rag.embedding_api_key 或环境变量），无法创建。")
            return None
        if not base_url:
            # OpenAI 官方兜底端点
            base_url = "https://api.openai.com/v1"
            logger.warning("未配置 embedding base_url，回退 OpenAI 官方端点 %s", base_url)
        return OpenAICompatibleEmbeddingFunction(
            api_key=api_key,
            model=model,
            api_base=base_url,
            dimensions=self.embedding_config.dimension or None,
        )

    def _create_openai_embedding_function(self):
        """OpenAI Embedding（走 OpenAI 兼容接口，不依赖 openai 包版本）。"""
        return self._create_openai_compatible_embedding_function()

    def _create_custom_embedding_function(self):
        """自定义 OpenAI-Like Embedding（DashScope / vLLM / Ollama 等）。"""
        base_url = self.embedding_config.base_url or os.getenv("EMBEDDING_BASE_URL")
        if not base_url:
            logger.warning("Custom embedding provider 缺少 base_url，回退本地模型。")
            return self._create_local_embedding_function()
        return self._create_openai_compatible_embedding_function()

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
        if self._collection is None and not self._init_failed:
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
                # 边界标量化：任何非标量字段在此拍平，避免整批 upsert 被拒后
                # 该文档被静默丢弃（只影响 Chroma，不改动调用方的原始 meta）。
                doc_metadata = _sanitize_metadata(
                    {**(metadata or {}), **doc_data.get("metadata", {})}
                )
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
            # 初始化已在 ensure_ready 阶段尝试过；此处直接返回空，
            # 避免每次检索都重跑 initialize()（含 15s embedding 探测）。
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
        """更新文档

        注意（2026-09-30 修复）：metadata 必须过 `_sanitize_metadata()` 才能交给
        chromadb。原实现把调用方的 dict 原样塞进 `collection.update()`，于是
        含 list 的键（如人审闸门要用的 `source_run_ids`）会让 update 抛
        `Expected metadata value to be a str, int, float or bool, got [...]`——
        人审/签发只改了 BM25 语料、向量库没改，两个索引对同一张卡的
        `status` 各说各话。入库路径（add_documents）早就过了消毒，update 漏了。
        """
        if self._collection is None:
            return False

        try:
            update_data = {"ids": [doc_id]}
            if content is not None:
                update_data["documents"] = [content]
            if metadata is not None:
                update_data["metadatas"] = [_sanitize_metadata(metadata)]

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

    def get_all_documents(
        self,
        include: Optional[List[str]] = None,
        batch: int = 500,
    ) -> List[Dict[str, Any]]:
        """返回集合内全部文档（id + content + metadata）。

        供 BM25 索引重建、标签审计、Web 端全量浏览使用。
        集合未初始化或为空时返回空列表。
        """
        if self._collection is None:
            return []

        _include = include or ["documents", "metadatas"]
        out: List[Dict[str, Any]] = []
        try:
            offset = 0
            while True:
                res = self._collection.get(
                    limit=batch, offset=offset, include=_include
                )
                ids = res.get("ids") or []
                if not ids:
                    break
                docs = res.get("documents") or []
                metas = res.get("metadatas") or []
                for i, doc_id in enumerate(ids):
                    out.append({
                        "doc_id": doc_id,
                        "content": docs[i] if docs else "",
                        "metadata": metas[i] if metas else {},
                    })
                offset += batch
                if len(ids) < batch:
                    break
        except Exception as e:
            logger.error(f"get_all_documents failed: {e}")
        return out

    def close(self):
        """关闭连接"""
        if self._client:
            self._client = None
            self._collection = None
