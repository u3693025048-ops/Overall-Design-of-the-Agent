"""Engine resolver for the simulation backend.

Single source of truth for selecting and locating the simulation engine
(MATLAB subprocess, MATLAB Engine API for Python, GNU Octave subprocess,
or pure-Python fallback).

All call sites (``cli_agent.py``, ``GuidanceSimulator``, ``RunSimulationTool``,
``MatlabRLOptimizer``) go through :func:`resolve_engine_config` so that
engine selection, env-var overrides and executable discovery behave
identically everywhere.

Resolution rules
================

1.  Environment variables override config values:

    *   ``SIMULATION_ENGINE``   →  ``engine`` argument
        (one of ``"auto"``, ``"matlab_engine"``, ``"matlab"``, ``"octave"``,
        ``"python"``)
    *   ``MATLAB_PATH``         →  ``matlab_path`` argument
    *   ``OCTAVE_PATH``         →  ``octave_path`` argument

2.  If the resolved ``engine`` is ``"auto"``, prefer:

        matlab_engine  →  matlab  →  octave  →  python

    where each option is only considered if its prerequisite
    (importable Python package or executable on PATH) is satisfied.

3.  When an explicit engine is requested but its prerequisite is missing,
    a warning is logged and the call still returns the requested engine
    so the caller can decide whether to fall back or surface the error.
    (We *do not* silently substitute, because RL/CI runs need to fail
    loudly when the expected engine is unavailable.)
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Public engine identifiers, in preference order for "auto" resolution.
ENGINE_MATLAB_ENGINE = "matlab_engine"
ENGINE_MATLAB = "matlab"
ENGINE_OCTAVE = "octave"
ENGINE_PYTHON = "python"
ENGINE_AUTO = "auto"

_VALID_ENGINES = {
    ENGINE_AUTO,
    ENGINE_MATLAB_ENGINE,
    ENGINE_MATLAB,
    ENGINE_OCTAVE,
    ENGINE_PYTHON,
}


@dataclass(frozen=True)
class EngineConfig:
    """Resolved simulation-engine configuration.

    Attributes
    ----------
    engine
        One of ``"matlab_engine"``, ``"matlab"``, ``"octave"``, ``"python"``.
        ``"auto"`` is *not* a valid value here — :func:`resolve_engine_config`
        always reduces it to a concrete choice.
    octave_path
        Resolved path to the Octave executable (absolute when
        ``shutil.which`` succeeded, else the original config string).
    matlab_path
        Resolved path to the MATLAB executable (absolute when
        ``shutil.which`` succeeded, else the original config string).
    matlab_engine_available
        True if ``import matlab.engine`` succeeded.  Useful for callers
        that want to know whether the in-process backend is reachable
        even when the resolved engine is something else.
    matlab_executable_available
        True if a MATLAB binary was found on PATH or at *matlab_path*.
    octave_executable_available
        True if an Octave binary was found on PATH or at *octave_path*.
    """

    engine: str
    octave_path: str
    matlab_path: str
    matlab_engine_available: bool = False
    matlab_executable_available: bool = False
    octave_executable_available: bool = False


# ---------------------------------------------------------------------------
# Detection helpers (cheap; do *not* spawn a subprocess)
# ---------------------------------------------------------------------------


def expand_path(path: str) -> str:
    """Expand ``~`` and ``${VAR}`` / ``%VAR%`` in a user-supplied path.

    Used for every path coming from ``config.yaml`` or env vars, so users
    can write portable entries like::

        simulation:
          matlab_path: "${MATLAB_ROOT}/bin/matlab.exe"
          octave_path: "~/octave/bin/octave"

    Returns the original string unchanged if it is empty.
    """
    if not path:
        return path
    return os.path.expandvars(os.path.expanduser(path))


def _which(cmd: str) -> Optional[str]:
    """Like :func:`shutil.which` but accepts an already-absolute path.

    Performs ``~`` and env-var expansion first so manual config entries
    using shell-style placeholders resolve correctly.  Returns ``None``
    if *cmd* is empty or cannot be resolved.
    """
    if not cmd:
        return None
    cmd = expand_path(cmd)
    # Already an absolute, existing path?  Use it as-is.
    if os.path.isabs(cmd) and os.path.exists(cmd):
        return cmd
    return shutil.which(cmd)


def _matlab_engine_importable() -> bool:
    """Return True if ``import matlab.engine`` would succeed.

    We use :mod:`importlib.util.find_spec` so we don't actually pay the
    import cost (the MATLAB Engine for Python is slow to import on cold
    start because it loads the MATLAB runtime DLLs).
    """
    try:
        spec = importlib.util.find_spec("matlab.engine")
    except (ImportError, ValueError):
        return False
    return spec is not None


def detect_available_engines(
    octave_path: str = "octave",
    matlab_path: str = "matlab",
) -> dict:
    """Probe the host for each engine prerequisite.

    Returns a dict with boolean flags keyed by engine identifier
    (``matlab_engine``, ``matlab``, ``octave``).  ``python`` is omitted
    because it is always available.
    """
    return {
        ENGINE_MATLAB_ENGINE: _matlab_engine_importable(),
        ENGINE_MATLAB: _which(matlab_path) is not None,
        ENGINE_OCTAVE: _which(octave_path) is not None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_engine_config(
    engine: str = ENGINE_AUTO,
    octave_path: str = "octave",
    matlab_path: str = "matlab",
) -> EngineConfig:
    """Resolve a final :class:`EngineConfig`, honouring env vars and auto-detect.

    Parameters
    ----------
    engine
        Caller's preferred engine.  ``"auto"`` triggers preference-based
        selection (matlab_engine > matlab > octave > python).
    octave_path
        Caller's configured Octave executable name or absolute path.
    matlab_path
        Caller's configured MATLAB executable name or absolute path.

    The function is *idempotent* and *side-effect free* apart from
    informational logging.  It does not start any engine — only probes
    for availability.
    """
    # ── Env-var overrides ──────────────────────────────────────────────
    env_engine = os.environ.get("SIMULATION_ENGINE", "").strip().lower()
    if env_engine:
        engine = env_engine

    octave_path = os.environ.get("OCTAVE_PATH", octave_path) or "octave"
    matlab_path = os.environ.get("MATLAB_PATH", matlab_path) or "matlab"

    # Expand ``~`` and ``${VAR}`` / ``%VAR%`` in user-supplied paths so
    # config.yaml entries like "${MATLAB_ROOT}/bin/matlab.exe" or
    # "~/octave/bin/octave" resolve correctly across machines.
    octave_path = expand_path(octave_path)
    matlab_path = expand_path(matlab_path)

    # ── Early validation: explicit absolute path that does not exist ──
    # When the user manually sets a path in config.yaml or via env var,
    # warn loudly *before* the first subprocess call so misconfiguration
    # is easy to spot in startup logs.  We only warn for absolute paths
    # because bare basenames like "matlab" rely on PATH lookup.
    if os.path.isabs(matlab_path) and not os.path.exists(matlab_path):
        logger.warning(
            f"Configured matlab_path does not exist: {matlab_path!r}.  "
            f"Check simulation.matlab_path in config.yaml or the "
            f"MATLAB_PATH env var."
        )
    if os.path.isabs(octave_path) and not os.path.exists(octave_path):
        logger.warning(
            f"Configured octave_path does not exist: {octave_path!r}.  "
            f"Check simulation.octave_path in config.yaml or the "
            f"OCTAVE_PATH env var."
        )

    if engine not in _VALID_ENGINES:
        logger.warning(
            f"Unknown engine {engine!r}; valid options are "
            f"{sorted(_VALID_ENGINES)!r}.  Falling back to 'auto'."
        )
        engine = ENGINE_AUTO

    # ── Probe availability ─────────────────────────────────────────────
    avail = detect_available_engines(octave_path, matlab_path)
    matlab_engine_avail = avail[ENGINE_MATLAB_ENGINE]
    matlab_exec_avail = avail[ENGINE_MATLAB]
    octave_exec_avail = avail[ENGINE_OCTAVE]

    # ── Resolve "auto" to a concrete engine ───────────────────────────
    if engine == ENGINE_AUTO:
        if matlab_engine_avail:
            chosen = ENGINE_MATLAB_ENGINE
        elif matlab_exec_avail:
            chosen = ENGINE_MATLAB
        elif octave_exec_avail:
            chosen = ENGINE_OCTAVE
        else:
            chosen = ENGINE_PYTHON
            logger.warning(
                "auto-detect: no MATLAB / matlab.engine / Octave found.  "
                "Falling back to the pure-Python simulator (lower fidelity).  "
                "Install Octave or MATLAB and/or set OCTAVE_PATH/MATLAB_PATH."
            )
    else:
        chosen = engine
        # Loud warning if explicit engine is unavailable — but don't
        # silently substitute.  Caller decides the fallback policy.
        if chosen == ENGINE_MATLAB_ENGINE and not matlab_engine_avail:
            logger.warning(
                "engine='matlab_engine' but the MATLAB Engine API for Python "
                "is not importable.  Install with:\n"
                "    cd <matlabroot>/extern/engines/python && python -m pip install ."
            )
        elif chosen == ENGINE_MATLAB and not matlab_exec_avail:
            logger.warning(
                f"engine='matlab' but no MATLAB executable was found at "
                f"{matlab_path!r}.  Set MATLAB_PATH env var or "
                f"simulation.matlab_path in config.yaml."
            )
        elif chosen == ENGINE_OCTAVE and not octave_exec_avail:
            logger.warning(
                f"engine='octave' but no Octave executable was found at "
                f"{octave_path!r}.  Set OCTAVE_PATH env var or "
                f"simulation.octave_path in config.yaml."
            )

    # Resolve to absolute paths when possible.  This makes subprocess
    # invocations on Windows more reliable (PATH may not propagate).
    octave_resolved = _which(octave_path) or octave_path
    matlab_resolved = _which(matlab_path) or matlab_path

    cfg = EngineConfig(
        engine=chosen,
        octave_path=octave_resolved,
        matlab_path=matlab_resolved,
        matlab_engine_available=matlab_engine_avail,
        matlab_executable_available=matlab_exec_avail,
        octave_executable_available=octave_exec_avail,
    )
    logger.info(
        f"Simulation engine resolved: {cfg.engine!r}  "
        f"(matlab.engine={'yes' if matlab_engine_avail else 'no'}, "
        f"matlab.exe={'yes' if matlab_exec_avail else 'no'}, "
        f"octave={'yes' if octave_exec_avail else 'no'})"
    )
    return cfg
