#!/usr/bin/env python3
"""
Run-session logging: one log file per invocation under ./logs/logMMDDHHMM.txt

Captures:
  - all logging.Logger output (root + child loggers)
  - stdout / stderr (print, tracebacks, subprocess messages)
"""

from __future__ import annotations

import atexit
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TextIO

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_active_session: Optional["RunLogSession"] = None


def get_run_log_path() -> Optional[Path]:
    """Return the active run log file path, or None if logging not initialized."""
    if _active_session is None:
        return None
    return _active_session.log_path


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _allocate_log_path(log_dir: Path, when: Optional[datetime] = None) -> Path:
    """Return logs/logMMDDHHMM.txt, with _01/_02 suffix if the name already exists."""
    when = when or datetime.now()
    stem = f"log{when.strftime('%m%d%H%M')}"
    candidate = log_dir / f"{stem}.txt"
    if not candidate.exists():
        return candidate
    for idx in range(1, 100):
        alt = log_dir / f"{stem}_{idx:02d}.txt"
        if not alt.exists():
            return alt
    return log_dir / f"{stem}_{datetime.now().strftime('%S')}.txt"


class _TeeStream:
    """Duplicate writes to the original stream and the run log file."""

    def __init__(self, original: TextIO, log_file: TextIO):
        self._original = original
        self._log_file = log_file
        self.encoding = getattr(original, "encoding", "utf-8")

    def write(self, data: str) -> int:
        if not data:
            return 0
        try:
            self._original.write(data)
        except UnicodeEncodeError:
            enc = getattr(self._original, "encoding", None) or "utf-8"
            self._original.write(data.encode(enc, errors="replace").decode(enc))
        try:
            self._log_file.write(data)
            self._log_file.flush()
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        try:
            self._log_file.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return self._original.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        return self._original.fileno()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


class RunLogSession:
    """Context manager that tees stdout/stderr and configures root logging."""

    def __init__(
        self,
        log_dir: str = "logs",
        level: int = logging.INFO,
        project_root: Optional[Path] = None,
        argv: Optional[List[str]] = None,
    ):
        self.log_dir_name = log_dir
        self.level = level
        self.project_root = project_root or _default_project_root()
        self.argv = argv if argv is not None else sys.argv[:]
        self.log_path: Optional[Path] = None
        self._log_file: Optional[TextIO] = None
        self._orig_stdout: Optional[TextIO] = None
        self._orig_stderr: Optional[TextIO] = None
        self._stream_handler: Optional[logging.StreamHandler] = None
        self._started_at: Optional[datetime] = None
        self._exit_code: int = 0

    def __enter__(self) -> "RunLogSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._exit_code = 1
            if self._log_file is not None:
                self._log_file.write(
                    f"\n[EXCEPTION] {exc_type.__name__}: {exc}\n"
                )
                traceback.print_exception(exc_type, exc, tb, file=self._log_file)
                self._log_file.flush()
        self.stop(exit_code=self._exit_code if exc_type is not None else 0)
        return False

    def start(self) -> Path:
        global _active_session
        if _active_session is not None:
            return _active_session.log_path  # type: ignore[return-value]

        self._started_at = datetime.now()
        log_dir = self.project_root / self.log_dir_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = _allocate_log_path(log_dir, self._started_at)

        self._log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self._write_header()

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, self._log_file)  # type: ignore[assignment]
        sys.stderr = _TeeStream(self._orig_stderr, self._log_file)  # type: ignore[assignment]

        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(self.level)

        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FMT)

        # stderr is teed to the log file; attach handler after Tee is installed.
        self._stream_handler = logging.StreamHandler(sys.stderr)
        self._stream_handler.setFormatter(formatter)
        root.addHandler(self._stream_handler)

        _active_session = self
        atexit.register(self._atexit_finalize)
        logging.getLogger(__name__).info("Run log file: %s", self.log_path)
        return self.log_path

    def stop(self, exit_code: int = 0) -> None:
        self._exit_code = exit_code
        self._finalize()

    def _write_header(self) -> None:
        if self._log_file is None:
            return
        started = (self._started_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        cmd = " ".join(self.argv)
        self._log_file.write("=" * 80 + "\n")
        self._log_file.write("GSMultiAgent Run Log\n")
        self._log_file.write(f"Started     : {started}\n")
        self._log_file.write(f"Command     : {cmd}\n")
        self._log_file.write(f"Working dir : {Path.cwd()}\n")
        self._log_file.write(f"Log file    : {self.log_path}\n")
        self._log_file.write("=" * 80 + "\n\n")
        self._log_file.flush()

    def _write_footer(self, exit_code: int) -> None:
        if self._log_file is None or self._log_file.closed:
            return
        finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_file.write("\n" + "=" * 80 + "\n")
        self._log_file.write(f"Finished    : {finished}\n")
        self._log_file.write(f"Exit code   : {exit_code}\n")
        self._log_file.write("=" * 80 + "\n")
        self._log_file.flush()

    def _finalize(self) -> None:
        global _active_session
        if _active_session is not self:
            return

        self._write_footer(self._exit_code)

        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr

        root = logging.getLogger()
        if self._stream_handler is not None:
            root.removeHandler(self._stream_handler)
            self._stream_handler.close()

        if self._log_file is not None and not self._log_file.closed:
            self._log_file.close()

        _active_session = None
        try:
            atexit.unregister(self._atexit_finalize)
        except Exception:
            pass

    def _atexit_finalize(self) -> None:
        if _active_session is self:
            self._finalize()


def init_run_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    argv: Optional[List[str]] = None,
    project_root: Optional[Path] = None,
) -> Path:
    """
    Initialize run logging for the current process.

    Returns the path to logs/logMMDDHHMM.txt (or suffixed variant if colliding).
    """
    session = RunLogSession(
        log_dir=log_dir,
        level=level,
        argv=argv,
        project_root=project_root,
    )
    return session.start()


def finalize_run_log(exit_code: int = 0) -> None:
    """Explicitly close the active run log session (optional; atexit also handles this)."""
    global _active_session
    if _active_session is not None:
        _active_session.stop(exit_code=exit_code)
