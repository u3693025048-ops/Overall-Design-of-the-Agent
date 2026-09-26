"""Memory / process-pressure guards for MATLAB-heavy GSM runs."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_OOM_PATTERNS = (
    re.compile(r"out of memory", re.I),
    re.compile(r"insufficient memory", re.I),
    re.compile(r"cannot allocate", re.I),
    re.compile(r"memory allocation failed", re.I),
    re.compile(r"R6025", re.I),
    re.compile(r"pure virtual function call", re.I),
    re.compile(r"内存不足"),
    re.compile(r"内存溢出"),
)


def mc_use_parfor(default: bool = False) -> bool:
    """Whether external-MC scripts may use MATLAB parfor (memory-heavy)."""
    env = os.environ.get("GSM_MC_USE_PARFOR", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    try:
        from multi_agent.config_loader import get_config

        return bool(getattr(get_config().simulation, "mc_use_parfor", default))
    except Exception:
        return bool(default)


def matlab_max_comp_threads(default: int = 2) -> int:
    env = os.environ.get("GSM_MATLAB_MAX_COMP_THREADS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        from multi_agent.config_loader import get_config

        return max(
            1, int(getattr(get_config().simulation, "matlab_max_comp_threads", default) or default)
        )
    except Exception:
        return max(1, int(default))


def matlab_eval_preamble(*, exit_at_end: bool = False) -> str:
    """Memory-safe MATLAB -batch preamble: no figures, limited threads, no parpool."""
    n = matlab_max_comp_threads()
    parts = [
        "set(0,'DefaultFigureVisible','off')",
        f"try, maxNumCompThreads({n}); end",
        "try, delete(gcp('nocreate')); end",
    ]
    body = "; ".join(parts) + "; "
    return body


def looks_like_oom(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _OOM_PATTERNS)


def system_memory_snapshot() -> Dict[str, Any]:
    """Return available/total memory in GB when possible."""
    out: Dict[str, Any] = {
        "ok": False,
        "total_gb": None,
        "available_gb": None,
        "matlab_procs": 0,
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        out["total_gb"] = round(vm.total / (1024**3), 2)
        out["available_gb"] = round(vm.available / (1024**3), 2)
        out["ok"] = True
        n_mat = 0
        for proc in psutil.process_iter(attrs=["name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if "matlab" in name:
                    n_mat += 1
            except Exception:
                continue
        out["matlab_procs"] = n_mat
        return out
    except Exception:
        pass

    # Windows fallback without psutil
    if os.name == "nt":
        try:
            import subprocess

            ps = (
                "$os=Get-CimInstance Win32_OperatingSystem; "
                "[PSCustomObject]@{Total=$os.TotalVisibleMemorySize; "
                "Free=$os.FreePhysicalMemory} | ConvertTo-Json"
            )
            raw = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps],
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            import json

            data = json.loads(raw)
            # Values are KB
            total_kb = float(data.get("Total") or 0)
            free_kb = float(data.get("Free") or 0)
            out["total_gb"] = round(total_kb / (1024**2), 2)
            out["available_gb"] = round(free_kb / (1024**2), 2)
            out["ok"] = total_kb > 0
        except Exception as exc:
            logger.debug("memory snapshot failed: %s", exc)
        try:
            import subprocess

            raw = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-Process MATLAB*,matlab* -ErrorAction SilentlyContinue | Measure-Object).Count"],
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            out["matlab_procs"] = int(str(raw).strip() or 0)
        except Exception:
            pass
    return out


def memory_pressure(
    *,
    min_free_gb: float = 2.0,
    max_matlab_procs: int = 12,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Return (is_pressure, reason, snapshot)."""
    snap = system_memory_snapshot()
    avail = snap.get("available_gb")
    n_mat = int(snap.get("matlab_procs") or 0)
    if max_matlab_procs > 0 and n_mat > int(max_matlab_procs):
        return True, f"matlab_procs={n_mat}>{max_matlab_procs}", snap
    if avail is not None and float(avail) < float(min_free_gb):
        return True, f"available_gb={avail}<{min_free_gb}", snap
    return False, "", snap
