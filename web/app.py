#!/usr/bin/env python3
"""
知识库管理 Web 端（FastAPI）

提供：仪表盘统计、混合检索（level/case/lib 硬过滤）、文档增删改查、
四库一键重索引、经验卡浏览、回流反馈、next_action 推断。

启动：python run_web.py   （见仓库根 run_web.py）
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .auth import current_user, register_auth
from multi_agent.memory import doc_extract as docx_extract
from multi_agent.memory.retrieval_eval import run_retrieval_eval
from multi_agent.memory.schema_gate import run_gate

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
    """回流。P0-2：失败的**必须绑工况**（scenario + r0_hash）才能归档进 failed_under。"""
    doc_id: str
    success: bool = True
    scenario: Optional[str] = None
    r0_hash: Optional[str] = None
    run_id: Optional[str] = None
    violations: Optional[List[str]] = None
    note: str = ""


class FailureIn(BaseModel):
    """另提失败卡（P0-2，05 §B4.4 阶段⑧）：沉淀"此工况勿用该处置"，须再过人审。

    署名（recorded_by）只取登录会话，与 /api/review、/api/activate 同口径。
    """
    symptom: str
    approach: str                      # 失效的处置（人读）
    layer: str = "总体"
    scenario: str = ""                 # RUN_CASE（T/G/AP/R），P1~P4 亦可
    r0_hash: str = ""
    run_id: str = ""
    root_cause: str = ""
    violations: Optional[List[str]] = None
    next_action: str = "change_structure"
    paired_with: str = ""
    confidence: float = 0.5


class NextActionIn(BaseModel):
    query: str
    level: Optional[str] = None
    case: Optional[str] = None
    lib: Optional[str] = None


class ReviewIn(BaseModel):
    """人审裁决（P0-1）：candidate → accepted / rejected。

    ``reviewer`` 仅作**兼容字段**保留：服务端只认登录会话身份（P0-1d），
    若请求体自称的身份与会话不一致 → 403（防冒签），一致或留空则忽略。
    """
    doc_id: str
    decision: str = "accept"          # accept | reject
    reviewer: str = ""


class ActivateIn(BaseModel):
    """签字生效（P0-1）：accepted → active，须签字 + 真值已证实。

    ``signer`` 同上：签字人一律取自登录会话，不得由请求体指定他人。
    """
    doc_id: str
    signer: str = ""


class ConfirmIn(BaseModel):
    """高风险/不可逆操作确认（P1-3）：批准 / 驳回某条待确认请求。

    ``operator`` 仅作**兼容字段**保留：裁决署名一律取登录会话（P0-1d 同口径），
    请求体自称他人 → 403。
    """
    confirmation_id: str
    decision: str = "approve"          # approve | reject
    operator: str = ""
    note: str = ""


def set_manager(mgr) -> None:
    global _MANAGER
    _MANAGER = mgr


def _mgr():
    if _MANAGER is None:
        raise HTTPException(status_code=503, detail="知识库未初始化")
    return _MANAGER


# ── 署名身份：一律以登录会话为准（P0-1d 防冒签）────────────────────────────
# 缺陷背景（2026-09-30）：`/api/review`、`/api/activate` 原先直接采信请求体里的
# `reviewer` / `signer`。中间件虽然要求"已登录"，但登录成 alice 的人可以填
#   {"signer": "admin"}
# 于是台账上留下的是"admin 签的字"——**人审闸门形同虚设**（谁登录谁就能冒充任何人
# 署名）。R8 要求"人审无签字不得作为 active 知识"，签字必须真的对得上人。
#
# 现在的口径：
#   1. 身份只来自会话 cookie（`current_user`），请求体永远不能指定；
#   2. 请求体若自称了一个**不等于**会话用户的名字 → 403 明确拒绝（而不是静默改写，
#      否则调用方以为冒签成功了）；
#   3. 留痕（runs / card_promotions 的 reviewed_by / signer）写会话用户名，
#      响应体同时回传 acting_user / acting_role 供界面与审计核对。
def _acting_user(request: Request) -> Dict[str, Any]:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="未登录或会话已过期")
    return user


def _bind_identity(claimed: str, user: Dict[str, Any], what: str) -> str:
    """把请求体自称的身份与登录会话比对，返回权威署名。不一致即 403。"""
    who = str(user.get("username") or "").strip()
    claimed = str(claimed or "").strip()
    if claimed and claimed != who:
        raise HTTPException(
            status_code=403,
            detail=(f"{what}必须以登录身份署名：当前登录 {who!r}，"
                    f"请求体自称 {claimed!r}；不得代他人{what}。"),
        )
    return who


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


# ── 上传解析（txt/pdf/docx…） ────────────────────────────────────────────────
# ⚠️ 这两个路由**必须注册在** ``/api/doc/{doc_id:path}`` 之前：Starlette 按注册
#    顺序匹配，``{doc_id:path}`` 会把 ``/api/doc/upload/formats`` 整个吞掉。
@APP.get("/api/doc/upload/formats")
async def api_upload_formats():
    """上传前的能力探测：哪些解析器可用、支持哪些扩展名、上限多少。

    前端拿它决定 accept 属性与灰显提示（例如未装 pypdf 时禁用 PDF 入口）。
    """
    return {
        "parsers": docx_extract.available_parsers(),
        "exts": docx_extract.supported_exts(),
        "max_bytes": docx_extract.DEFAULT_MAX_BYTES,
        "max_chars": docx_extract.DEFAULT_MAX_CHARS,
    }


@APP.post("/api/doc/upload")
async def api_upload_doc(
    file: UploadFile = File(...),
    lib: str = Form("norm"),
    level: str = Form("总体"),
    case: str = Form("ALL"),
    source_of_truth: str = Form(""),
    apply: bool = Form(False),
    truncate: bool = Form(False),
    max_chars: int = Form(0),
    distilled: bool = Form(False),
    machine_checkable: bool = Form(False),
):
    """上传文件 → 解析为纯文本。

    两段式：**解析**与**入库**是两个动作，默认只做前者。

    * ``apply=false``（默认）：只解析、**不落盘**，把全文回给前端预览/人工编辑，
      再由人工点「入库」走 ``POST /api/doc``。
    * ``apply=true``：解析后直接 ``ingest_doc`` 入库（等价于把解析结果填进
      ``/api/doc`` 的 content）。

    解析失败一律 400 + ``{code, message, detail}``（``code`` 机器可判：
    ``unsupported_type`` / ``too_large`` / ``too_long`` / ``parse_failed`` /
    ``missing_dependency`` / ``empty_text`` / ``empty_file`` / ``decode_failed``），
    **绝不静默返回空文本**。
    """
    raw = await file.read()
    try:
        res = docx_extract.extract_text(
            raw, file.filename,
            max_chars=(max_chars or None),
            truncate=bool(truncate),
        )
    except docx_extract.ExtractError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    out = res.to_dict()
    out["apply"] = bool(apply)
    if apply:
        out["ingest"] = await _mgr().ingest_doc(
            content=res.text, lib=lib, level=level, case=case,
            source_of_truth=source_of_truth or res.filename,
            distilled=distilled, machine_checkable=machine_checkable,
        )
    return out


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
async def api_feedback(body: FeedbackIn, request: Request):
    """回流。P0-2：失败带工况时同时写 failed_under（索引 + 卡文件）。

    ``recorded_by`` 取登录会话（防冒名登记回流）。
    """
    user = _acting_user(request)
    return await _mgr().feedback(
        body.doc_id, body.success,
        scenario=body.scenario, r0_hash=body.r0_hash, run_id=body.run_id,
        violations=body.violations, note=body.note,
        recorded_by=str(user.get("username") or ""),
    )


@APP.post("/api/failure")
async def api_archive_failure(body: FailureIn, request: Request):
    """另提失败卡（candidate）：写盘 → 须播种入库 → 须人审（P0-2，R7）。

    ``recorded_by`` 只取登录会话；失败必须绑工况（scenario + r0_hash）。
    """
    user = _acting_user(request)
    return await _mgr().archive_failure(
        symptom=body.symptom, layer=body.layer, scenario=body.scenario,
        r0_hash=body.r0_hash, approach=body.approach, root_cause=body.root_cause,
        run_id=body.run_id, violations=body.violations,
        next_action=body.next_action, paired_with=body.paired_with,
        confidence=body.confidence,
        recorded_by=str(user.get("username") or ""),
    )


@APP.get("/api/failures")
async def api_failures(paired_only: bool = False):
    """失败案例一览（P0-2）：outcome=failure 的卡 + 各自绑的失败工况 + 配对成功卡。

    一成一败：``paired_with`` 指向的成功卡若确实存在，回 ``paired_ok=True``。
    """
    cards = await _mgr().list_docs(lib="experience")
    by_card = {c.get("card_id"): c for c in cards if c.get("card_id")}
    out = []
    for c in cards:
        if (c.get("outcome") or "success") != "failure":
            continue
        pw = c.get("paired_with") or ""
        peer = by_card.get(pw) if pw else None
        out.append({
            "card_id": c.get("card_id"),
            "doc_id": c.get("doc_id"),
            "status": c.get("status"),
            "symptom": (c.get("snippet") or "")[:80],
            "failed_under": c.get("failed_under") or [],
            "failed_scenarios": c.get("failed_scenarios"),
            "paired_with": pw,
            "paired_ok": bool(peer),
            "paired_status": (peer or {}).get("status"),
        })
    if paired_only:
        out = [x for x in out if x["paired_ok"]]
    return {"count": len(out), "failures": out}


@APP.post("/api/next_action")
async def api_next_action(body: NextActionIn):
    return await _mgr().next_action(body.query, level=body.level, case=body.case, lib=body.lib)


@APP.get("/api/experience")
async def api_experience():
    docs = await _mgr().list_docs(lib="experience")
    return {"count": len(docs), "cards": docs}


# ── 人审闸门（P0-1）：从"写了没接线"变成可操作端点 ────────────────────────────
@APP.get("/api/promotions")
async def api_promotions(card_id: Optional[str] = None):
    """晋升/声明留痕：card_promotions 全量（含 draft 声明行）或按卡过滤。"""
    led = _mgr().ledger
    if led is None:
        raise HTTPException(status_code=503, detail="台账不可用")
    rows = led.list_promotions(card_id=card_id)
    return {"count": len(rows), "promotions": rows}


@APP.get("/api/review_queue")
async def api_review_queue():
    """人审待办：有 draft 声明但尚无人审结论的经验卡（门下省队列）。"""
    led = _mgr().ledger
    if led is None:
        raise HTTPException(status_code=503, detail="台账不可用")
    queue = led.review_queue()
    cards = await _mgr().list_docs(lib="experience")   # 已带 card_id/status 等字段
    by_card = {c.get("card_id"): c for c in cards if c.get("card_id")}
    pending = []
    for cid in queue:
        c = by_card.get(cid) or {}
        pending.append({
            "card_id": cid,
            "doc_id": c.get("doc_id") or f"kb:experience:{cid}",
            "status": c.get("status"),
            "declared_status": c.get("declared_status"),
            "provenance_attested": c.get("provenance_attested"),
            "next_action_enum": c.get("next_action_enum"),
            "source_run_ids": c.get("source_run_ids"),
        })
    return {"count": len(pending), "pending": pending}


@APP.post("/api/review")
async def api_review(body: ReviewIn, request: Request):
    """人审：candidate → accepted / rejected（写 card_promotions）。

    署名（reviewer）**只取登录会话**；请求体自称他人 → 403。
    """
    user = _acting_user(request)
    reviewer = _bind_identity(body.reviewer, user, "审核")
    res = await _mgr().review(
        body.doc_id, decision=body.decision, reviewer=reviewer)
    res["acting_user"] = reviewer
    res["acting_role"] = user.get("role")
    return res


@APP.post("/api/activate")
async def api_activate(body: ActivateIn, request: Request):
    """签字生效：accepted → active；闸门要求签字非空且 source_run_ids 已证实。

    签字人（signer）**只取登录会话**；请求体自称他人 → 403。
    """
    user = _acting_user(request)
    signer = _bind_identity(body.signer, user, "签字")
    res = await _mgr().activate(body.doc_id, signer=signer)
    res["acting_user"] = signer
    res["acting_role"] = user.get("role")
    return res


# ── 高风险/不可逆操作确认（P1-3）────────────────────────────────────────────
def _confirm_gate():
    """载入确认闸门并绑定与 cli 侧共享的落盘路径（跨进程确认）。

    仅当**尚未**绑定路径时才回落到默认路径 —— 不覆盖显式配置（如测试用的
    临时台账），避免把两个进程指向不同文件。
    """
    from multi_agent.security import confirm_gate as cg
    if not cg.store_path():
        cg.set_store_path(cg.default_store_path())
    return cg


@APP.get("/api/confirmations")
async def api_confirmations(status: str = ""):
    """高风险操作确认队列（默认 pending；status=all 看全部）。"""
    cg = _confirm_gate()
    cg.refresh()
    rows = cg.pending() if status in ("", "pending") else cg.all_requests()
    snap = cg.audit_snapshot()
    return {
        "count": len(rows),
        "mode": snap["mode"],
        "scope": snap["scope"],
        "standing": snap["standing"],
        "pending": rows,
    }


@APP.post("/api/confirm")
async def api_confirm(body: ConfirmIn, request: Request):
    """批准 / 驳回一条待确认请求（P1-3）。

    裁决署名（operator）**只取登录会话**；请求体自称他人 → 403（防冒签）。
    批准后，调用方以**同参数重试**即放行（确认号一次性消耗）。
    """
    user = _acting_user(request)
    operator = _bind_identity(body.operator, user, "确认")
    cg = _confirm_gate()
    cg.refresh()
    action = str(body.decision or "approve").strip().lower()
    if action in ("approve", "accept", "ok", "yes"):
        ok = cg.approve(body.confirmation_id, by=operator, note=body.note)
        decision = "approved"
    elif action in ("reject", "deny", "no"):
        ok = cg.reject(body.confirmation_id, by=operator, note=body.note)
        decision = "rejected"
    else:
        raise HTTPException(status_code=400, detail=f"未知裁决 decision={body.decision!r}")
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=f"确认号 {body.confirmation_id!r} 不存在或状态不可裁决",
        )
    req = cg.get(body.confirmation_id)
    return {
        "status": "ok",
        "decision": decision,
        "confirmation": req.to_dict() if req else None,
        "acting_user": operator,
        "acting_role": user.get("role"),
    }


# ── P2-2 吏部：角色注册 + 用户画像 ────────────────────────────────────────────
class ProfileIn(BaseModel):
    """本人画像更新（P2-2）。``roles`` **刻意不开放** —— 角色分配属管理员动作，
    请求体不得自封角色（与 P0-1d 的"不得代他人署名"同口径）。"""

    display_name: Optional[str] = None
    department: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


def _profile_store():
    """惰性绑定画像存储（与 cli 进程约定同一路径 <repo>/logs/user_profiles.json）。"""
    from multi_agent.security import user_profile as up
    path = ""
    try:
        from multi_agent.config_loader import get_config
        path = str(getattr(get_config().identity, "profiles_path", "") or "")
    except Exception:  # pragma: no cover — 配置不可用则用默认路径
        path = ""
    return up.configure_store(path or None)


@APP.get("/api/roles")
async def api_roles():
    """角色注册表（P2-2 吏部）：执行角色 / 用户角色的能力画像。只读。

    中间件已要求登录；这里不再额外校验（角色表本身不含敏感数据）。
    """
    from multi_agent.security import role_registry as rr
    reg = rr.get_registry()
    rep = rr.validate_registry(reg)
    return {
        "policy_version": reg.to_dict()["policy_version"],
        "n_roles": reg.to_dict()["n_roles"],
        "agent_roles": reg.agent_roles(),
        "user_roles": reg.user_roles(),
        "roles": reg.to_dict()["roles"],
        "consistent": rep["ok"],
        "problems": rep["problems"],
    }


@APP.get("/api/profile")
async def api_profile(request: Request):
    """当前登录用户的画像 + 兑现后的有效权限（P2-2）。

    - 已登记画像 → 画像角色 ∪ web 登录角色桥接；
    - 未登记画像 → ``registered=false``，但 web 登录角色仍经桥接表给出兜底
      （admin→operator / viewer→auditor），因此不会出现"能登录却零权限"；
    - 完全未知的用户名 → **最小权限**（fail-closed）。
    """
    user = _acting_user(request)
    from multi_agent.security import user_profile as up
    st = _profile_store()
    name = str(user.get("username") or "")
    ident = up.resolve_identity(name, store=st, web_role=user.get("role", ""))
    prof = st.get(name)
    return {
        "identity": ident.to_dict(),
        "profile": prof.to_dict() if prof else None,
        "web_role": user.get("role"),
    }


@APP.put("/api/profile")
async def api_update_profile(body: ProfileIn, request: Request):
    """本人更新画像（展示名 / 岗位 / 偏好 / 备注）。**不得自封角色**。"""
    user = _acting_user(request)
    from multi_agent.security import user_profile as up
    st = _profile_store()
    name = str(user.get("username") or "")
    cur = st.ensure(name)
    merged = up.UserProfile(
        username=cur.username,
        display_name=(cur.display_name if body.display_name is None else str(body.display_name)),
        department=(cur.department if body.department is None else str(body.department)),
        roles=list(cur.roles),                      # 角色只能由管理员分配，此处原样保留
        preferences=dict(cur.preferences),
        notes=(cur.notes if body.notes is None else str(body.notes)),
    )
    if body.preferences:
        merged.preferences.update(dict(body.preferences))
    try:
        prof = st.upsert(merged)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ident = up.resolve_identity(name, store=st, web_role=user.get("role", ""))
    return {"status": "ok", "profile": prof.to_dict(), "identity": ident.to_dict()}


# ── 配置可视化（只读展示 + 白名单字段写回）──────────────────────────────────
# 三条纪律见 `web/config_api.py` 模块头：
#   1. 只有白名单字段可写；ablation / identity / metrics **只读**（改它会打穿机检或安全层）；
#   2. 写回是**定点行替换**，不 dump YAML ⇒ config.yaml 里的运维注释原样保留；
#   3. 写后复校：目标值必须真的变了，且**除该路径外全文无差异**（防改到同名 key）。
#
# 权限：配置能改外部工具路径、LLM 端点与预算上限 ⇒ **仅 admin**。
# viewer 一律 403（不是"前端隐藏按钮"，是服务端拒绝）。
class ConfigIn(BaseModel):
    """配置写回。``updates`` 形如 ``{"simulation.matlab_path": "…"}``。"""

    updates: Dict[str, Any] = {}
    confirm: bool = False          # 预留：未来高风险项（如关掉向量检索）要求两次确认


def _config_admin(request: Request) -> Dict[str, Any]:
    user = _acting_user(request)
    if str(user.get("role") or "") != "admin":
        raise HTTPException(
            status_code=403,
            detail=(f"配置写回仅限 admin：当前登录 {user.get('username')!r}"
                    f"（role={user.get('role')!r}）。请联系管理员。"),
        )
    return user


@APP.get("/api/config")
async def api_config_get(request: Request):
    """当前配置：可编辑字段（分组，脱敏）+ 只读块 + 未纳管块。仅 admin。"""
    _config_admin(request)
    from . import config_api
    return await asyncio.to_thread(config_api.read_config)


@APP.post("/api/config")
async def api_config_set(body: ConfigIn, request: Request):
    """写回配置：校验 → 备份 → 定点替换 → 复校 → 热重载。

    拒绝一律 400 + ``{code, path, error}``（``not_whitelisted`` / ``bad_type`` /
    ``bad_enum`` / ``out_of_range`` / ``path_not_found`` / ``verify_failed`` /
    ``collateral_change``），**绝不部分写入** —— 复校不过就一行都不落盘。
    """
    user = _config_admin(request)
    from . import config_api
    try:
        res = await asyncio.to_thread(config_api.apply_updates, body.updates)
    except config_api.ConfigRefused as exc:
        raise HTTPException(status_code=400, detail=exc.to_dict())
    res["acting_user"] = user.get("username")
    res["acting_role"] = user.get("role")
    return res


@APP.get("/api/config/meta")
async def api_config_meta(request: Request):
    """配置文件的落点与校验和（审计用：确认某次改动改的是哪个文件）。"""
    _config_admin(request)
    from . import config_api
    return {
        "file": str(config_api.CONFIG_PATH),
        "sha256": config_api.sha256_of(),
        "n_fields": len(config_api.FIELDS),
        "groups": list(config_api.GROUP_ORDER),
    }


@APP.get("/api/eval")
async def api_eval(top_k: int = 10, sample: int = 0, lib: str = ""):
    """Stage D：检索质量自评（MRR/Recall/HitRate + trace 抽样）。"""
    sample = sample if sample and sample > 0 else None
    lib = lib or None
    res = await run_retrieval_eval(_mgr(), top_k=top_k, sample=sample, lib_filter=lib)
    return res


@APP.get("/api/validate")
async def api_validate(artifact: str = "all"):
    """Stage E：JSON Schema 机检关门。返回各产物 pass/fail 报告。

    artifact: all | next_action | kb_corpus | kb_seed_manifest | retrieval_eval
    """
    allowed = {"all", "next_action", "kb_corpus", "kb_seed_manifest", "retrieval_eval"}
    if artifact not in allowed:
        artifact = "all"
    report = run_gate(artifacts=artifact)
    return report


# ── 需求执行（阶段 1 计划 → 阶段 2 逐任务真调工具 + 两道闸门）────────────────
# 编排层在 `multi_agent/integration/requirement_runner.py`（决策 1=(c)：桥在上、
# 论文链在下），执行时每步都过 `tool_guard`（计划级权限）+ `confirm_gate`
#（高风险确认），再调 `multi_agent/tools/stage2_tools.py` 里登记的工具。
#
# 四条不变量（与既有闸门同口径，**不得放宽**）：
#   1. 署名一律取登录会话：`approver`/`operator` 只是兼容字段，自称他人 → 403。
#   2. **未获人工批准不得执行** —— 执行前状态必须是 `approved`（或已批准、
#      正等确认的 `awaiting_confirmation`），且批准人的权限快照必须齐备
#      （由 `plan_review.MODE_PERMISSIONS` 派生，不手写映射）。
#   3. 高风险动作（`simulate_modelica`）**每次调用**生成确认号，须人工裁决后
#      以同参数重试才放行（`/api/requirement/confirm`）。
#   4. 所有跑次落 `logs/requirement_runner/<task_id>/`；**绝不写** `deliverables/07_runs/`。
#   5. **人审不可绕过**（阶段 3）：跑完自动生成的回流卡只落 `candidate`，
#      `active` 只能经 `/api/requirement/reflow/review` + `/reflow/activate`
#      （署名取会话 + 闸门要求来源真值已登记）。不存在"自动入库即 active"的通路。
#   6. **检索留痕**：`plan["retrieval"]` 逐项带 `source`
#      （`kb:…` / `pe:…` / `llm` / `default` / `text`）⇒ 可审计"这一轮靠什么"。

_REQ_TASKS: Dict[str, Any] = {}          # 在跑的后台任务引用（防 GC）


def _rr():
    from multi_agent.integration import requirement_runner
    return requirement_runner


class RequirementIn(BaseModel):
    """提交一段需求做分析（**只分析，不执行**）。"""

    text: str
    top_k: int = 5
    use_llm: bool = True
    lib: Optional[str] = None


class RequirementApproveIn(BaseModel):
    """人审裁决。``approver`` 仅作兼容字段：署名一律取登录会话。"""

    task_id: str
    decision: str = "approve"           # approve | reject
    note: str = ""
    approver: str = ""


class RequirementRunIn(BaseModel):
    task_id: str


def _req_plan_or_400(task_id: str) -> Dict[str, Any]:
    rr = _rr()
    try:
        return rr.load_plan(task_id)
    except rr.RunnerRefused as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                                 # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取计划失败：{exc}")


@APP.post("/api/requirement/analyze")
async def api_req_analyze(body: RequirementIn, request: Request):
    """一段需求 → KB 命中 + LLM 计划参数 + 固定 5 步计划 + 机器预审结论。

    ``kb`` 未初始化时仍可用：检索侧如实返回 ``available=false``，计划照常生成。
    """
    user = _acting_user(request)
    rr = _rr()
    if not (body.text or "").strip():
        raise HTTPException(status_code=400, detail="需求文本为空")
    try:
        plan = await rr.analyze(
            body.text, kb=_MANAGER, username=str(user.get("username") or ""),
            web_role=str(user.get("role") or ""), top_k=int(body.top_k),
            use_llm=bool(body.use_llm), lib=body.lib,
        )
    except Exception as exc:                                 # noqa: BLE001
        logger.exception("需求分析失败")
        raise HTTPException(status_code=500, detail=f"需求分析失败：{exc}")
    plan["acting_user"] = user.get("username")
    plan["acting_role"] = user.get("role")
    return plan


@APP.post("/api/requirement/approve")
async def api_req_approve(body: RequirementApproveIn, request: Request):
    """人工批准 / 驳回（驳回带反馈则重拟一次，上限 1）。

    ``approver`` 必须与登录会话一致（防冒签）；**批准人自身须具备该计划所需的
    计划级权限**，否则 fail-closed 拒绝。
    """
    user = _acting_user(request)
    approver = _bind_identity(body.approver, user, "批准")
    rr = _rr()
    _req_plan_or_400(body.task_id)
    res = await rr.approve(
        body.task_id, decision=body.decision, note=body.note, operator=approver,
        web_role=str(user.get("role") or ""), kb=_MANAGER,
    )
    res["acting_user"] = approver
    res["acting_role"] = user.get("role")
    if not res.get("ok"):
        res["status_code"] = 409 if res.get("code") == "insufficient_permission" else 400
    return res


async def _req_run_bg(task_id: str) -> None:
    rr = _rr()
    try:
        await rr.run(task_id)
    except Exception:                                        # noqa: BLE001
        logger.exception("需求执行后台任务异常：%s", task_id)


@APP.post("/api/requirement/run")
async def api_req_run(body: RequirementRunIn, request: Request):
    """**执行**已批准的计划（后台线程跑，立即返回；用 /status 轮询）。

    未批准 / 权限快照不全 / 已有任务在跑 ⇒ 结构化拒绝，不抛。
    """
    user = _acting_user(request)
    rr = _rr()
    plan = _req_plan_or_400(body.task_id)
    # 与 `run_sync` 同口径：approved 放行；awaiting_confirmation 也放行
    #（已批准、正等高风险确认 —— 重入时该步会重新走一遍闸门，未确认仍停在原处）。
    if plan.get("status") not in (rr.STATUS_APPROVED, rr.STATUS_AWAITING_CONFIRM):
        return {"ok": False, "code": "not_approved", "status": plan.get("status"),
                "error": f"未获人工批准（status={plan.get('status')!r}），拒绝执行。"}
    running = _REQ_TASKS.get(body.task_id)
    if running is not None and not running.done():
        return {"ok": False, "code": "already_running", "status": "running"}
    task = asyncio.create_task(_req_run_bg(body.task_id))
    _REQ_TASKS[body.task_id] = task
    task.add_done_callback(lambda _t: _REQ_TASKS.pop(body.task_id, None))
    return {"ok": True, "status": "started", "task_id": body.task_id,
            "acting_user": user.get("username"), "acting_role": user.get("role"),
            "note": "已在后台执行；轮询 GET /api/requirement/status?task_id=… 。"}


class RequirementConfirmIn(BaseModel):
    """高风险确认裁决（``simulate_modelica`` 拉起外部进程前的人工二次确认）。

    ``operator`` 仅作兼容字段：署名一律取登录会话（``_bind_identity``）。
    """

    task_id: str
    confirmation_id: str = ""           # 留空 ⇒ 取任务当前待确认号
    decision: str = "approve"           # approve | reject
    note: str = ""
    operator: str = ""


@APP.get("/api/requirement/status")
async def api_req_status(task_id: str):
    """任务状态 + **按每步真实执行状态**换算的进度 + 5 步状态（含待确认项）。"""
    rr = _rr()
    try:
        return rr.status(task_id)
    except rr.RunnerRefused as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@APP.get("/api/requirement/confirmations")
async def api_req_confirmations(task_id: str):
    """本任务当前等待的高风险确认（计划视图 + 确认闸门权威状态 + 闸门模式）。"""
    rr = _rr()
    _req_plan_or_400(task_id)
    try:
        return await asyncio.to_thread(rr.pending_confirmations, task_id)
    except rr.RunnerRefused as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@APP.post("/api/requirement/confirm")
async def api_req_confirm(body: RequirementConfirmIn, request: Request):
    """**批准 / 驳回**待确认项；批准后自动在后台续跑（回到 ``run`` 的同一通路）。

    三条不变量：
      1. 署名取登录会话（``_bind_identity``），自称他人 403；
      2. 确认人不得缺该计划所需的计划级权限（``missing_plan_permissions`` 兜底）；
      3. 未处于 ``awaiting_confirmation`` ⇒ 结构化拒绝，不抛。
    """
    user = _acting_user(request)
    operator = _bind_identity(body.operator, user, "确认")
    rr = _rr()
    plan = _req_plan_or_400(body.task_id)
    if plan.get("status") != rr.STATUS_AWAITING_CONFIRM:
        return {"ok": False, "code": "no_pending_confirmation",
                "status": plan.get("status"),
                "error": (f"当前状态 {plan.get('status')!r} 无待确认项"
                          "（仅 awaiting_confirmation 可裁决）。")}
    res = await asyncio.to_thread(
        rr.resolve_confirmation, body.task_id,
        confirmation_id=body.confirmation_id, decision=body.decision,
        note=body.note, operator=operator, web_role=str(user.get("role") or ""),
    )
    res["acting_user"] = operator
    res["acting_role"] = user.get("role")
    if not res.get("ok"):
        res["status_code"] = 409 if res.get("code") == "insufficient_permission" else 400
        return res

    if str(res.get("decision")) == "approved":
        running = _REQ_TASKS.get(body.task_id)
        if running is not None and not running.done():
            res["resume"] = {"ok": False, "code": "already_running"}
        else:
            task = asyncio.create_task(_req_run_bg(body.task_id))
            _REQ_TASKS[body.task_id] = task
            task.add_done_callback(lambda _t: _REQ_TASKS.pop(body.task_id, None))
            res["resume"] = {"ok": True, "status": "started"}
            res["note"] = ("已确认并在后台续跑；轮询 GET /api/requirement/status"
                           "?task_id=… 。")
    return res


@APP.get("/api/requirement/plan")
async def api_req_plan(task_id: str):
    """计划全文（含 KB 命中、抽取结果、LLM 元信息、机器预审结论）。"""
    return _req_plan_or_400(task_id)


@APP.get("/api/requirement/plans")
async def api_req_plans(limit: int = 50):
    """历史任务列表（新→旧）。"""
    return {"plans": _rr().list_plans(limit=int(limit))}


@APP.get("/api/requirement/env")
async def api_req_env():
    """四个外部工具的本机可用性（与 CLI tools-status 同源）。探测同步且较慢，放线程里。"""
    return await asyncio.to_thread(_rr().probe_environment)


@APP.get("/api/requirement/artifact")
async def api_req_artifact(task_id: str, name: str):
    """下载本轮产物（canvas_ir.json / requirement_extraction.json / <id>.ssim）。

    **只暴露任务目录内的单个文件**：``name`` 含路径分隔或 ``..`` 一律 400，
    解析后若逃出任务目录同样拒绝（下载口是外部输入）。
    """
    rr = _rr()
    try:
        p = rr.resolve_artifact(task_id, name)
    except rr.RunnerRefused as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return FileResponse(str(p), filename=p.name)


# ── 阶段 3 回流闭环：跑完沉淀的经验卡「待审 → 人审 → 签字生效」─────────────
# 回流卡与话料落在**运行期沉淀区**（`logs/requirement_runner/experience_reflow/`，
# 可用 `REQUIREMENT_REFLOW_ROOT` 覆盖），台账也自带一本（`ledger.reflow.sqlite`）
# —— 绝不写四库生产台账 / 语料 / 卡目录。
#
# 三条不变量（与四库人审同口径，**不得放宽**）：
#   1. 署名只取登录会话（`reviewer` / `signer` 自称他人 → 403）；
#   2. `log_claim()` 只落 draft 声明；真值只能经 `attest_run()` 登记；
#   3. `activate` 闸门要求 `source_run_ids` **全部已证实** ⇒ 伪造来源必被拦。
class ReflowReviewIn(BaseModel):
    """回流卡人审裁决。``reviewer`` 仅作兼容字段：署名一律取登录会话。"""

    card_id: str
    decision: str = "accept"            # accept | reject
    note: str = ""
    reviewer: str = ""


class ReflowActivateIn(BaseModel):
    """回流卡签字生效。``signer`` 仅作兼容字段：署名一律取登录会话。"""

    card_id: str
    signer: str = ""


@APP.get("/api/requirement/reflow/queue")
async def api_req_reflow_queue():
    """回流待审队列（**台账为权威**）：candidate / accepted / active / rejected 快照。"""
    rr = _rr()
    return await asyncio.to_thread(rr.reflow_queue)


@APP.post("/api/requirement/reflow/review")
async def api_req_reflow_review(body: ReflowReviewIn, request: Request):
    """人审回流卡：candidate → accepted / rejected（写回流台账的 card_promotions）。"""
    user = _acting_user(request)
    reviewer = _bind_identity(body.reviewer, user, "审核")
    rr = _rr()
    res = await rr.reflow_review(body.card_id, decision=body.decision,
                                 reviewer=reviewer, note=body.note)
    res["acting_user"] = reviewer
    res["acting_role"] = user.get("role")
    return res


@APP.post("/api/requirement/reflow/activate")
async def api_req_reflow_activate(body: ReflowActivateIn, request: Request):
    """签字生效：accepted → active；闸门要求签字非空且来源真值全部已证实。

    未证实来源（如伪造的 run_id）⇒ 结构化拒绝并回 ``missing_runs``，**不抛**。
    """
    user = _acting_user(request)
    signer = _bind_identity(body.signer, user, "签字")
    rr = _rr()
    res = await rr.reflow_activate(body.card_id, signer=signer)
    res["acting_user"] = signer
    res["acting_role"] = user.get("role")
    return res



# ── 认证 / 用户管理（登录、会话、用户 CRUD、/api/* 保护） ───────────────────
register_auth(APP)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(APP, host="0.0.0.0", port=6006)
