"""Cache for external-MC AND evaluations (identical params → reuse)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_MEMORY: Dict[str, Dict[str, Any]] = {}
_CACHE_DIR = Path(__file__).resolve().parents[2] / "guidance_output" / "mc_eval_cache"


def cache_enabled(default: bool = True) -> bool:
    env = os.environ.get("GSM_MC_EVAL_CACHE", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    try:
        from multi_agent.config_loader import get_config

        return bool(getattr(get_config().simulation, "mc_eval_cache", default))
    except Exception:
        return bool(default)


def make_cache_key(
    *,
    matlab_script_path: str,
    mc_root: str,
    scenario: str,
    seeds: list[int],
    split: str,
    n_runs: int,
    kill_radius: float,
    dp_params: Optional[Mapping[str, float]],
    mode: str = "and_full",
) -> str:
    payload = {
        "script": str(Path(matlab_script_path).resolve()),
        "mc_root": str(Path(mc_root).resolve()),
        "scenario": scenario,
        "seeds": [int(s) for s in seeds],
        "split": split,
        "n_runs": int(n_runs),
        "kill_radius": float(kill_radius),
        "dp_params": {k: float(v) for k, v in sorted((dp_params or {}).items())},
        "mode": str(mode or "and_full"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not cache_enabled():
        return None
    with _LOCK:
        hit = _MEMORY.get(key)
        if hit is not None:
            return dict(hit)
    path = _CACHE_DIR / f"{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        with _LOCK:
            _MEMORY[key] = dict(data)
        logger.info("[mc_eval_cache] disk hit %s", key[:10])
        return dict(data)
    except Exception:
        return None


def cache_put(key: str, value: Mapping[str, Any]) -> None:
    if not cache_enabled():
        return
    # Do not cache hard failures (engine missing etc.) — transient.
    if not value.get("success", False) and value.get("error"):
        err = str(value.get("error") or "")
        if "not found" in err.lower() or "timeout" in err.lower():
            return
    data = dict(value)
    data["from_cache"] = False
    with _LOCK:
        _MEMORY[key] = dict(data)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=0),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("[mc_eval_cache] disk write failed: %s", exc)
