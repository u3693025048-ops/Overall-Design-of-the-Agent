"""Pipeline warmup for next-scenario overlap (MATLAB / MC data / RAG).

This does NOT start formal T2/T3/T4 optimization and must not write PE.
It only:
  * ensures the MATLAB job pool is up
  * page-cache prefaults external MC ``.mat`` files
  * optionally runs a tiny no-op eval on a warm worker
  * read-only KB / RAG touch for the upcoming scenario
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline_warmup")
_LOCK = threading.Lock()
_PENDING: Dict[str, Future] = {}
_DONE: Dict[str, Dict[str, Any]] = {}


def warmup_enabled(default: bool = True) -> bool:
    env = os.environ.get("GSM_PIPELINE_WARMUP", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    try:
        from multi_agent.config_loader import get_config

        return bool(getattr(get_config().simulation, "pipeline_warmup", default))
    except Exception:
        return bool(default)


def _mc_root() -> Path:
    try:
        from multi_agent.config_loader import get_config

        return Path(getattr(get_config().simulation, "mc_root", "data/mc") or "data/mc")
    except Exception:
        return ROOT / "data" / "mc"


def _dev_seeds(master_seed: int) -> List[int]:
    master = int(master_seed)
    return [master * 100 + offset for offset in (1, 2, 3)]


def _prefault_file(path: Path, chunk: int = 8 * 1024 * 1024) -> int:
    if not path.is_file():
        return 0
    total = 0
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            total += len(buf)
    return total


def _warmup_matlab_pool() -> Dict[str, Any]:
    out: Dict[str, Any] = {"pool": False, "noop": False}
    try:
        from multi_agent.simulation.matlab_job_pool import get_job_pool

        pool = get_job_pool()
        if pool is None:
            return out
        out["pool"] = bool(pool.started)
        # Tiny no-op keeps JIT / path warm without touching PE or MC protocol.
        try:
            pool.run_eval("disp('pipeline_warmup');", timeout_sec=120.0)
            out["noop"] = True
        except Exception as exc:
            out["noop_error"] = str(exc)[:200]
    except Exception as exc:
        out["pool_error"] = str(exc)[:200]
    return out


def _warmup_mc_files(scenario: str, master_seed: int, mc_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(mc_root) if mc_root is not None else _mc_root()
    bytes_read = 0
    found = 0
    missing: List[str] = []
    for seed in _dev_seeds(master_seed):
        for split in ("dev", "test"):
            path = root / scenario / f"seed_{seed}_{split}.mat"
            if path.is_file():
                found += 1
                bytes_read += _prefault_file(path)
            else:
                missing.append(str(path))
    # Also touch the KB Monte Carlo template (script generation path).
    template = ROOT / "knowledge_base" / "matlab" / "guidance" / "monte_carlo_single.m"
    if template.is_file():
        bytes_read += _prefault_file(template)
        found += 1
    return {
        "files_found": found,
        "bytes_read": bytes_read,
        "missing": missing[:6],
    }


def _warmup_kb_rag(scenario: str) -> Dict[str, Any]:
    """Read-only KB touch + best-effort RAG retrieve (no PE writes)."""
    out: Dict[str, Any] = {"kb_files": 0, "rag": False}
    kb_paths = [
        ROOT / "knowledge_base" / "conditions.md",
        ROOT / "knowledge_base" / "expert" / "design_path.md",
        ROOT / "knowledge_base" / "matlab" / "guidance" / "monte_carlo_single.m",
    ]
    # Soft-match scenario docs if present.
    for pattern in (f"*{scenario}*", f"*{scenario.lower()}*"):
        kb_paths.extend(ROOT.glob(f"knowledge_base/**/{pattern}"))
    seen = set()
    for path in kb_paths:
        try:
            rp = path.resolve()
        except Exception:
            continue
        if rp in seen or not path.is_file():
            continue
        seen.add(rp)
        try:
            _prefault_file(path)
            out["kb_files"] += 1
        except Exception:
            pass

    query = (
        f"{scenario} guidance autopilot parameters TUNE_PARAMS REUSE_HISTORY "
        f"hit_rate SEP PeakNy PM BW monte_carlo"
    )
    try:
        import asyncio

        from multi_agent.memory.rag_knowledge_base import RAGKnowledgeBase

        async def _run() -> int:
            rag = RAGKnowledgeBase()
            await rag.initialize()
            hits = await rag.retrieve(query, top_k=3)
            return len(hits or [])

        # Avoid nested-loop issues: only when no running loop.
        try:
            asyncio.get_running_loop()
            out["rag"] = False
            out["rag_skip"] = "running_loop"
        except RuntimeError:
            n = asyncio.run(_run())
            out["rag"] = True
            out["rag_hits"] = int(n)
    except Exception as exc:
        out["rag_error"] = str(exc)[:200]
    return out


def warmup_scenario(
    scenario: str,
    *,
    master_seed: int,
    mc_root: Optional[str] = None,
    include_matlab: bool = True,
    include_rag: bool = True,
) -> Dict[str, Any]:
    """Synchronously warm resources for one upcoming scenario (read-only)."""
    if not warmup_enabled():
        return {"enabled": False, "scenario": scenario}
    started = time.perf_counter()
    report: Dict[str, Any] = {
        "enabled": True,
        "scenario": str(scenario),
        "master_seed": int(master_seed),
    }
    root = Path(mc_root) if mc_root else _mc_root()
    report["mc_files"] = _warmup_mc_files(str(scenario), int(master_seed), root)
    if include_matlab:
        report["matlab"] = _warmup_matlab_pool()
    if include_rag:
        report["kb_rag"] = _warmup_kb_rag(str(scenario))
    report["elapsed_sec"] = round(time.perf_counter() - started, 3)
    logger.info(
        "[pipeline_warmup] scenario=%s seed=%s elapsed=%.2fs files=%s pool=%s",
        scenario,
        master_seed,
        report["elapsed_sec"],
        report.get("mc_files", {}).get("files_found"),
        report.get("matlab", {}).get("pool"),
    )
    return report


def _job_key(scenario: str, master_seed: int) -> str:
    return f"{scenario}::{int(master_seed)}"


def start_warmup_async(
    scenario: str,
    *,
    master_seed: int,
    mc_root: Optional[str] = None,
) -> Optional[Future]:
    """Fire-and-forget warmup for the next scenario."""
    if not warmup_enabled():
        return None
    key = _job_key(scenario, master_seed)
    with _LOCK:
        existing = _PENDING.get(key)
        if existing is not None and not existing.done():
            return existing

        def _run() -> Dict[str, Any]:
            report = warmup_scenario(
                scenario, master_seed=master_seed, mc_root=mc_root
            )
            with _LOCK:
                _DONE[key] = report
                _PENDING.pop(key, None)
            return report

        fut = _EXECUTOR.submit(_run)
        _PENDING[key] = fut
        return fut


def wait_warmup(
    scenario: str,
    *,
    master_seed: int,
    timeout_sec: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """Wait for a previously started warmup (best-effort)."""
    key = _job_key(scenario, master_seed)
    with _LOCK:
        if key in _DONE:
            return dict(_DONE[key])
        fut = _PENDING.get(key)
    if fut is None:
        return None
    try:
        return fut.result(timeout=timeout_sec)
    except Exception as exc:
        logger.warning("[pipeline_warmup] wait failed for %s: %s", key, exc)
        return None


def prefetch_scenario_sequence(
    scenarios: Sequence[str],
    *,
    master_seed: int,
    mc_root: Optional[str] = None,
) -> None:
    """Warm the first scenario now; schedule the second in background."""
    scenarios = [str(s) for s in scenarios if s]
    if not scenarios or not warmup_enabled():
        return
    warmup_scenario(scenarios[0], master_seed=master_seed, mc_root=mc_root)
    if len(scenarios) > 1:
        start_warmup_async(scenarios[1], master_seed=master_seed, mc_root=mc_root)
