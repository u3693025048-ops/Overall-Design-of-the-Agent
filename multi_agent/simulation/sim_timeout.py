"""Central MATLAB subprocess timeout helpers (config + optional Nmc scaling)."""

from __future__ import annotations

from typing import Optional


def _sim_cfg():
    try:
        from multi_agent.config_loader import get_config

        return get_config().simulation
    except Exception:
        return None


def get_matlab_timeout_sec(nmc: Optional[int] = None) -> int:
    """Return timeout for a full MC script run.

    Scales with *nmc* but never below *matlab_timeout_floor_sec* and never above
    *matlab_timeout_sec*.  Env ``MATLAB_TIMEOUT_SEC`` still applies as the
    default ceiling via ``guidance_simulator.DEFAULT_SUBPROCESS_TIMEOUT_SEC``.
    """
    from multi_agent.simulation.guidance_simulator import DEFAULT_SUBPROCESS_TIMEOUT_SEC

    cfg = _sim_cfg()
    ceiling = int(getattr(cfg, "matlab_timeout_sec", 0) or 0) if cfg else 0
    if ceiling <= 0:
        ceiling = int(DEFAULT_SUBPROCESS_TIMEOUT_SEC)
    floor = int(getattr(cfg, "matlab_timeout_floor_sec", 360) or 360) if cfg else 360
    per_mc = int(getattr(cfg, "matlab_timeout_per_mc_sec", 22) or 22) if cfg else 22

    if nmc is None or int(nmc) <= 0:
        return max(floor, ceiling)

    scaled = floor + int(nmc) * per_mc
    return int(min(ceiling, max(floor, scaled)))


def get_physics_bounds_timeout_sec() -> float:
    cfg = _sim_cfg()
    if cfg is not None:
        t = float(getattr(cfg, "physics_bounds_timeout_sec", 0) or 0)
        if t > 0:
            return t
    return 120.0
