#!/usr/bin/env python3
"""
知识库管理 Web 端启动器

用法：
    python run_web.py
默认读取 config.yaml 的 web: 段（host/port/enabled）与 rag.bm25_corpus_path。
知识库为空时自动执行四库播种。
"""
import asyncio
import faulthandler
import os
import sys
import threading
import time
from pathlib import Path

import yaml

# 关键：让「原生库崩溃」（如 chromadb/onnxruntime 的 access violation）也能打印堆栈。
# 否则 Windows 上进程会毫无输出地直接消失，表现为「日志停在某行、端口拒绝连接」。
faulthandler.enable()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from multi_agent.memory.rag_knowledge_base import RAGKnowledgeBase  # noqa: E402
from multi_agent.memory.knowledge_manager import KnowledgeManager    # noqa: E402
from web.app import APP, set_manager                                  # noqa: E402
import uvicorn                                                        # noqa: E402


def _heartbeat() -> None:
    """每 60s 打一行心跳：用于判断「进程是否还活着」。

    若日志里心跳还在，说明服务正常，访问不了就是网络/端口问题；
    若心跳停了，说明进程已崩溃（配合 faulthandler 的堆栈定位原因）。
    """
    while True:
        time.sleep(60)
        print(f"[web] 服务运行中（pid={os.getpid()}，端口已监听）…", flush=True)


def _background_init(mgr: "KnowledgeManager", root: str) -> None:
    """在后台线程初始化知识库，绝不阻塞 Web 服务启动。

    早期版本把初始化放在 asyncio 事件循环里，一旦 chromadb import /
    PersistentClient 同步阻塞，wait_for 超时无法触发，导致 Web 永远卡在启动前。
    现在服务器先秒起，这里后台慢慢初始化；初始化的 [kb] 日志可直接定位 chromadb
    到底卡在哪一步。
    """
    try:
        mode = "纯 BM25" if not getattr(mgr, "vector_enabled", True) else "向量+BM25"
        print(f"[bg] 开始后台初始化（模式：{mode}）…", flush=True)
        ok = asyncio.run(asyncio.wait_for(mgr.ensure_ready(), timeout=120.0))
        if ok:
            st = asyncio.run(mgr.stats())
            if (st.get("total_documents", 0) or 0) == 0:
                print("[bg] 知识库为空，执行四库播种…", flush=True)
                asyncio.run(mgr.seed_four_libraries(str(root)))
            else:
                # 已索引则仅重建 BM25，保证混合检索覆盖
                print("[bg] 已索引，重建 BM25 互补索引…", flush=True)
                mgr.hybrid.rebuild(from_chroma=True)
            print("[bg] 知识库初始化完成，检索已可用。", flush=True)
        else:
            print("[bg] 知识库初始化未完成（embedding/Chroma 未就绪）。"
                  "检索返回空；配置好 rag.embedding / chromadb 后重启即可。", flush=True)
    except asyncio.TimeoutError:
        print("[bg] 后台初始化 120s 超时（chromadb 或 embedding 连接过慢）。"
              "Web 端仍可用，检索暂时为空。", flush=True)
    except Exception as exc:  # pragma: no cover
        import traceback
        print("[bg] 后台初始化异常（Web 端仍可用）：", flush=True)
        traceback.print_exc()


def main() -> None:
    print("[web] 启动中…（读取 config.yaml / web 段）", flush=True)
    cfg = {}
    try:
        with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as e:
        print(f"[web] 读取 config.yaml 失败：{e}", flush=True)

    web = cfg.get("web", {}) or {}
    host = web.get("host", "0.0.0.0")
    port = int(web.get("port", 8787))
    enabled = web.get("enabled", True)
    if not enabled:
        print("[web] web.enabled=false，未启动。", flush=True)
        return

    rag_cfg = cfg.get("rag", {}) or {}
    bm25_path = rag_cfg.get("bm25_corpus_path", "./kb_bm25_corpus.jsonl")

    # web.vector_search:
    #   false（默认）→ 纯 BM25 模式，全程不 import chromadb。适用于 chromadb
    #                  在该机器上不可用/会让进程崩溃的场景；管理端与检索照常可用。
    #   true         → 启用向量混合检索（需 chromadb + embedding 真正可用）。
    vector_enabled = bool(web.get("vector_search", False))

    print("[web] 构建向量库后端 RAGKnowledgeBase…", flush=True)
    rag = RAGKnowledgeBase()
    mgr = KnowledgeManager(rag, bm25_corpus_path=bm25_path, vector_enabled=vector_enabled)
    if not vector_enabled:
        print("[web] 运行模式：纯 BM25（未启用向量检索，不加载 chromadb）", flush=True)
    # 先挂上 manager：即使初始化未完成，Web 端也能响应（检索返回空，而非 500）。
    set_manager(mgr)

    # 关键设计：服务器**立即**启动，知识库初始化放到后台线程。
    # 这样无论 chromadb / embedding 卡多久，UI 都打得开、服务都不挂。
    print(f"[web] 知识库管理端已启动 → http://{host}:{port}", flush=True)
    print("[web] 知识库正在后台初始化，详见下方 [bg]/[kb] 日志…", flush=True)
    threading.Thread(target=_heartbeat, daemon=True).start()
    threading.Thread(target=_background_init, args=(mgr, str(ROOT)), daemon=True).start()

    uvicorn.run(APP, host=host, port=port)


if __name__ == "__main__":
    main()
