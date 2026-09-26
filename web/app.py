#!/usr/bin/env python3
"""
知识库管理 Web 端（FastAPI）

提供：仪表盘统计、混合检索（level/case/lib 硬过滤）、文档增删改查、
四库一键重索引、经验卡浏览、回流反馈、next_action 推断。

启动：python run_web.py   （见仓库根 run_web.py）
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── 全局：由 run_web.py 注入 ────────────────────────────────────────────────
APP = FastAPI(title="GSMultiAgent 知识库管理", version="1.0.0")
_MANAGER = None          # KnowledgeManager 实例
_WEB_DIR = Path(__file__).resolve().parent / "static"


# ── 请求体 ───────────────────────────────────────────────────────────────────
class DocIn(BaseModel):
    content: str
    lib: str = "norm"
    level: str = "总体"
    case: str = "ALL"
    source_of_truth: str = ""
    doc_id: Optional[str] = None
    distilled: bool = False
    machine_checkable: bool = False
    confidence: float = 1.0
    action: Optional[str] = None
    symptom: Optional[str] = None


class FeedbackIn(BaseModel):
    doc_id: str
    success: bool = True


class NextActionIn(BaseModel):
    query: str
    level: Optional[str] = None
    case: Optional[str] = None
    lib: Optional[str] = None


def set_manager(mgr) -> None:
    global _MANAGER
    _MANAGER = mgr


def _mgr():
    if _MANAGER is None:
        raise HTTPException(status_code=503, detail="知识库未初始化")
    return _MANAGER


# ── 静态页 ───────────────────────────────────────────────────────────────────
@APP.get("/", include_in_schema=False)
async def index():
    html = _WEB_DIR / "index.html"
    if html.exists():
        # 禁缓存：前端改动后刷新即生效，避免浏览器拿到旧的 index.html
        return FileResponse(
            str(html),
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )
    return JSONResponse({"detail": "index.html not found in web/static/"})


# ── API ──────────────────────────────────────────────────────────────────────
@APP.get("/api/stats")
async def api_stats():
    return await _mgr().stats()


@APP.get("/api/docs")
async def api_list_docs(
    lib: Optional[str] = None, level: Optional[str] = None,
    case: Optional[str] = None, q: Optional[str] = None,
):
    return await _mgr().list_docs(lib=lib, level=level, case=case, q=q)


@APP.get("/api/search")
async def api_search(
    q: str = Query(..., min_length=1),
    level: Optional[str] = None, case: Optional[str] = None,
    lib: Optional[str] = None, top_k: int = 10,
):
    res = await _mgr().query(q, level=level, case=case, lib=lib, top_k=top_k)
    return {"query": q, "results": res}


@APP.get("/api/doc/{doc_id:path}")
async def api_get_doc(doc_id: str):
    # 优先本地 BM25 语料（无向量模式也能查看全文），回落向量库
    doc = await _mgr().get_doc(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="doc not found")
    return doc


@APP.post("/api/doc")
async def api_add_doc(body: DocIn):
    return await _mgr().ingest_doc(
        content=body.content, lib=body.lib, level=body.level, case=body.case,
        source_of_truth=body.source_of_truth, doc_id=body.doc_id,
        distilled=body.distilled, machine_checkable=body.machine_checkable,
        confidence=body.confidence, action=body.action, symptom=body.symptom,
    )


@APP.delete("/api/doc/{doc_id:path}")
async def api_del_doc(doc_id: str):
    return await _mgr().delete_doc(doc_id)


@APP.post("/api/reindex")
async def api_reindex():
    counts = await _mgr().seed_four_libraries(os.getcwd())
    return {
        "status": "success",
        "counts": counts,
        "message": "四库重索引完成：规范 {norm} · 模型 {model} · 模板 {template} · 经验 {experience}（跳过未变更 {skipped}）".format(
            norm=counts.get("norm", 0), model=counts.get("model", 0),
            template=counts.get("template", 0), experience=counts.get("experience", 0),
            skipped=counts.get("skipped", 0),
        ),
    }


@APP.post("/api/feedback")
async def api_feedback(body: FeedbackIn):
    return await _mgr().feedback(body.doc_id, body.success)


@APP.post("/api/next_action")
async def api_next_action(body: NextActionIn):
    return await _mgr().next_action(body.query, level=body.level, case=body.case, lib=body.lib)


@APP.get("/api/experience")
async def api_experience():
    docs = await _mgr().list_docs(lib="experience")
    return {"count": len(docs), "cards": docs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(APP, host="0.0.0.0", port=8787)
