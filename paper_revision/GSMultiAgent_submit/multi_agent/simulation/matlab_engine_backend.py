"""In-process MATLAB Engine API backend.

Wraps the *MATLAB Engine API for Python* (``matlab.engine``) so that
simulation scripts can be executed without paying the per-call MATLAB
subprocess startup cost (5–15 s on Windows).  This is a substantial
speed-up for RL training (one MATLAB launch per episode → one MATLAB
launch *for the whole run*).

The backend is **optional** — install with::

    cd <matlabroot>/extern/engines/python && python -m pip install .

Where ``<matlabroot>`` is the directory printed by ``matlabroot`` inside
MATLAB itself (e.g. ``C:/Program Files/MATLAB/R2024a``).  If the package
is not importable, :class:`MatlabEngineBackend.start` returns ``False``
and the caller is expected to fall back to the subprocess path.

Concurrency
-----------

A single MATLAB engine session is **not thread-safe** — concurrent calls
on the same engine handle deadlock the MATLAB worker.  The backend
therefore holds a re-entrant lock around every call.  For RL training
(parallel envs) callers may need one backend instance per worker thread.

Lifecycle
---------

The class is designed for explicit ``start() / stop()`` lifecycle.  It
is NOT a singleton: the owner (typically :class:`SimulationExecutor`)
keeps a reference and calls :meth:`stop` on cleanup.  Re-entry is safe.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


# Module-level singleton flag — we lazy-import matlab.engine on first use
# to avoid loading the (slow) MATLAB runtime DLLs on processes that don't
# need them.
_MATLAB_ENGINE_MODULE = None


def _import_matlab_engine():
    """Lazy-import ``matlab.engine``.  Returns the module or ``None``."""
    global _MATLAB_ENGINE_MODULE
    if _MATLAB_ENGINE_MODULE is not None:
        return _MATLAB_ENGINE_MODULE
    try:
        import matlab.engine as _me  # noqa: WPS433 — runtime import is intentional
    except Exception as exc:  # ImportError | OSError (missing MATLAB DLLs)
        logger.info(
            "MATLAB Engine API for Python is not importable (%s); "
            "in-process backend will be unavailable.",
            exc,
        )
        return None
    _MATLAB_ENGINE_MODULE = _me
    return _me


class MatlabEngineBackend:
    """Thin wrapper around an in-process ``matlab.engine`` session.

    Typical usage from inside :class:`SimulationExecutor`::

        backend = MatlabEngineBackend()
        if backend.start():
            stdout, stderr, ok = backend.run_script(path, call_name="foo")
        else:
            # fall back to subprocess
            ...

    All public methods are safe to call before :meth:`start` — they will
    return error tuples / ``False`` rather than raising.
    """

    def __init__(self) -> None:
        self._engine: Any = None
        self._lock = threading.RLock()
        self._started = False
        # True once start() succeeds at least once; distinguishes a transient
        # crash (restartable) from a permanent failure (MATLAB not installed).
        self._ever_started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._started and self._engine is not None

    def start(self, options: Tuple[str, ...] = ("-nodisplay", "-nodesktop", "-nosplash")) -> bool:
        """Start a shared MATLAB engine session.

        Returns ``True`` on success, ``False`` if the package is not
        importable or the engine fails to launch (e.g. no licence).
        Idempotent: re-calling on a started backend is a no-op.
        """
        with self._lock:
            if self._started and self._engine is not None:
                return True

            me = _import_matlab_engine()
            if me is None:
                return False

            # MATLAB Engine API for Python officially supports 3.9-3.12.
            # On Python 3.13+ the C extension may crash or hang silently.
            # Log a warning but still attempt — some patch levels work.
            import sys
            if sys.version_info >= (3, 13):
                logger.warning(
                    "Python %d.%d detected — MATLAB Engine API officially "
                    "supports 3.9-3.12.  In-process engine may be unstable; "
                    "subprocess fallback ('matlab -batch') is recommended.",
                    sys.version_info.major, sys.version_info.minor,
                )

            # ── Attempt 1: connect to an already-running shared MATLAB session.
            # Start MATLAB manually and run  matlab.engine.shareEngine  (or
            # matlab.engine.shareEngine('mySession')) in the MATLAB Command
            # Window, then Python connects in <1 s instead of 30–90 s.
            # If no shared session exists, fall through to start_matlab().
            try:
                shared = me.find_matlab()
                if shared:
                    self._engine = me.connect_matlab(shared[0])
                    logger.info(
                        "MATLAB engine connected to existing shared session '%s' "
                        "(fast path — no cold-start delay).",
                        shared[0],
                    )
            except Exception as _conn_exc:
                logger.debug("connect_matlab() failed (%s); will start a new session.", _conn_exc)
                self._engine = None

            if self._engine is None:
                try:
                    # ``start_matlab`` accepts the same flags as ``matlab -<flag>``.
                    # On Windows ``-nojvm`` would block ``addpath`` so we don't
                    # request it; ``-nodisplay`` already suppresses figures.
                    logger.info("Starting new MATLAB engine session (this takes 30–90 s on Windows)…")
                    self._engine = me.start_matlab(" ".join(options))
                except Exception as exc:
                    logger.warning(
                        "Failed to start MATLAB engine session (%s); "
                        "in-process backend disabled.",
                        exc,
                    )
                    self._engine = None
                    self._started = False
                    return False

            # Suppress figures globally for the lifetime of the session.
            # Mirrors what the subprocess path does via OCTAVE_BATCH_PREFIX
            # / set(0,'DefaultFigureVisible','off') in the eval string.
            try:
                self._engine.eval(
                    "set(0,'DefaultFigureVisible','off'); "
                    "warning('off','all'); "
                    "format compact;",
                    nargout=0,
                )
            except Exception as exc:  # pragma: no cover  (best-effort)
                logger.debug("Could not apply figure suppression: %s", exc)

            self._started = True
            self._ever_started = True
            logger.info("MATLAB engine session started (in-process).")
            return True

    def stop(self) -> None:
        """Tear down the MATLAB engine session.  Safe to call multiple times."""
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.quit()
                except Exception as exc:  # pragma: no cover
                    logger.debug("Engine quit raised %s; ignoring.", exc)
            self._engine = None
            self._started = False

    def __enter__(self) -> "MatlabEngineBackend":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def restart(
        self,
        options: tuple = ("-nodisplay", "-nodesktop", "-nosplash"),
    ) -> bool:
        """Stop the current session (if any) and start a fresh one.

        Returns ``True`` on success.  Useful for crash-recovery: call this
        when :pyattr:`started` is ``False`` after a previously-successful
        session (i.e. :pyattr:`_ever_started` is ``True``).
        """
        logger.info("Restarting MATLAB engine session…")
        self.stop()
        return self.start(options)

    # ------------------------------------------------------------------
    # Script execution
    # ------------------------------------------------------------------

    def run_script(
        self,
        script_path: str,
        call_name: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> Tuple[str, str, bool]:
        """Execute a MATLAB script or function file in the running engine.

        Parameters
        ----------
        script_path
            Absolute path to the ``.m`` file.  Backslashes are handled.
        call_name
            If the file is a function file, pass the function name to
            invoke.  When ``None`` the file is executed via ``run()``
            (script-style).
        timeout_sec
            Per-call timeout in seconds.  ``None`` disables it.  Note
            that MATLAB's engine API does not expose a true cancellation
            primitive; we use ``background=True`` + ``future.result(...)``
            which raises ``matlab.engine.TimeoutError`` on overshoot but
            leaves the underlying work running until it finishes.

        Returns
        -------
        ``(stdout, stderr, ok)`` triple.  ``ok`` is False on engine
        errors / timeouts; in that case ``stderr`` carries the message.
        """
        if not self.started:
            return "", "engine not started", False

        if not os.path.exists(script_path):
            return "", f"script not found: {script_path}", False

        abs_path = os.path.abspath(script_path).replace("\\", "/")
        script_dir = os.path.dirname(abs_path)

        # Buffer stdout/stderr from the MATLAB workspace into Python
        # strings so the caller gets the same shape of output as the
        # subprocess path.
        out_buf = io.StringIO()
        err_buf = io.StringIO()

        with self._lock:
            try:
                # Make sure the script's directory is on the MATLAB path
                # — needed for both script-files and function-files when
                # they reference helpers in the same folder.
                self._engine.cd(script_dir, nargout=0)
                self._engine.addpath(script_dir, nargout=0)

                if call_name:
                    # For dynamically created temp files (e.g. _rltmp_*),
                    # MATLAB's function resolver cache is stale. We MUST
                    # cd + clear functions + rehash + call in ONE eval
                    # statement to guarantee the function is discoverable
                    # in the same workspace.  'clear functions' invalidates
                    # ALL JIT-compiled function caches (not just path cache)
                    # which is essential when running many unique temp
                    # function files in rapid succession (RL loop).
                    eval_cmd = (
                        f"cd('{script_dir}'); "
                        f"addpath('{script_dir}'); "
                        f"clear functions; "
                        f"rehash; "
                        f"{call_name}();"
                    )
                    if timeout_sec is not None and timeout_sec > 0:
                        future = self._engine.eval(
                            eval_cmd,
                            background=True,
                            nargout=0,
                            stdout=out_buf,
                            stderr=err_buf,
                        )
                        future.result(timeout=timeout_sec)
                    else:
                        self._engine.eval(
                            eval_cmd,
                            nargout=0,
                            stdout=out_buf,
                            stderr=err_buf,
                        )
                else:
                    # Script-style: run('/abs/path.m')
                    if timeout_sec is not None and timeout_sec > 0:
                        future = self._engine.run(
                            abs_path,
                            background=True,
                            nargout=0,
                            stdout=out_buf,
                            stderr=err_buf,
                        )
                        future.result(timeout=timeout_sec)
                    else:
                        self._engine.run(
                            abs_path,
                            nargout=0,
                            stdout=out_buf,
                            stderr=err_buf,
                        )
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                err_str = err_buf.getvalue()
                if err_str:
                    msg = f"{msg}\n--- captured stderr ---\n{err_str}"
                # Detect MATLAB feature-server / engine crashes and invalidate
                # the session so callers fall back to subprocess on next call.
                exc_str = str(exc)
                if (
                    "TS_PROCESS_CRASHED" in exc_str
                    or "feature at URL" in exc_str
                    or "crashed" in exc_str.lower()
                    or "MatlabEngine" in type(exc).__name__
                    or "engine is not running" in exc_str.lower()
                ):
                    logger.warning(
                        "MATLAB engine session crashed (%s: %s); marking as "
                        "stopped — next access via matlab_engine_backend "
                        "property will attempt an auto-restart.",
                        type(exc).__name__, exc_str[:200],
                    )
                    self._started = False
                    self._engine = None
                return out_buf.getvalue(), msg, False

        return out_buf.getvalue(), err_buf.getvalue(), True


# ---------------------------------------------------------------------------
# Module-level convenience: a process-wide shared backend.
#
# Many call sites (RL optimizer, run_simulation tool) want to share the
# *same* MATLAB session across calls to amortise the start-up cost.  We
# provide a lazy singleton that the SimulationExecutor caches in its
# constructor, but also expose it here for callers that don't have a
# convenient executor handle.
# ---------------------------------------------------------------------------

_SHARED_BACKEND: Optional[MatlabEngineBackend] = None
_SHARED_BACKEND_LOCK = threading.Lock()


def get_shared_backend() -> Optional[MatlabEngineBackend]:
    """Return a process-wide :class:`MatlabEngineBackend`, starting it on demand.

    Returns ``None`` if MATLAB Engine API for Python is not importable
    or the engine fails to start.  The result is cached for the
    lifetime of the process.
    """
    global _SHARED_BACKEND
    with _SHARED_BACKEND_LOCK:
        if _SHARED_BACKEND is not None and _SHARED_BACKEND.started:
            return _SHARED_BACKEND
        backend = MatlabEngineBackend()
        if backend.start():
            _SHARED_BACKEND = backend
            return backend
        return None


def stop_shared_backend() -> None:
    """Tear down the process-wide backend if it was started.

    Intended for atexit hooks and unit tests.
    """
    global _SHARED_BACKEND
    with _SHARED_BACKEND_LOCK:
        if _SHARED_BACKEND is not None:
            _SHARED_BACKEND.stop()
            _SHARED_BACKEND = None
