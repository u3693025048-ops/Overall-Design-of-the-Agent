"""Run artifact & memory file manager.

Manages the lifecycle of experience / Hermes memory and run artifacts across
sequential experiment groups (e.g. T1→T2→T3→T4 = one group):

* At the **start of each group**: clear experience base + Hermes memory
  (optionally backup first).
* After **each task** (or on demand): archive ``guidance_output/scripts`` and
  reports into a dedicated save folder under ``DATA/run_archives/``.

Typical layout after one group::

    DATA/run_archives/group_20260802_154200/
      manifest.json
      T1/scripts/  T1/reports/
      T2/scripts/  T2/reports/
      ...
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

# ── Default paths (aligned with config.yaml / run_cold_start.py) ─────────────
_DEFAULT_PE_JSON = ROOT / "parameter_experience_memory.json"
_DEFAULT_HERMES_JSON = ROOT / "guidance_output" / "hermes_agent_memory.json"
_DEFAULT_HERMES_RL_BEST = ROOT / "guidance_output" / "hermes_rl_best.json"
_DEFAULT_PE_BASE = ROOT / "parameter_experience_base"
SCRIPTS_DIR = ROOT / "guidance_output" / "scripts"
REPORTS_JSON_DIR = ROOT / "guidance_output" / "reports"
GUIDANCE_OUTPUT = ROOT / "guidance_output"
BACKUPS_DIR = ROOT / "backups"
DEFAULT_ARCHIVE_ROOT = ROOT / "DATA" / "run_archives"


def pe_json_path() -> Path:
    return Path(os.environ.get("GSM_PE_JSON", str(_DEFAULT_PE_JSON)))


def pe_base_path() -> Path:
    return Path(os.environ.get("GSM_PE_BASE", str(_DEFAULT_PE_BASE)))


def hermes_json_path() -> Path:
    return Path(os.environ.get("GSM_HERMES_JSON", str(_DEFAULT_HERMES_JSON)))


def hermes_rl_best_path() -> Path:
    return Path(os.environ.get("GSM_HERMES_RL_BEST", str(_DEFAULT_HERMES_RL_BEST)))


# Back-compat aliases (resolved at import; prefer pe_*_path() for isolation).
PE_JSON = _DEFAULT_PE_JSON
HERMES_JSON = _DEFAULT_HERMES_JSON
HERMES_RL_BEST = _DEFAULT_HERMES_RL_BEST
PE_BASE = _DEFAULT_PE_BASE

PE_SUBDIRS = (
    "params/short_term",
    "params/long_term",
    "models/short_term",
    "models/long_term",
)

REPORT_MD_GLOB = "report_*.md"
REPORT_JSON_GLOB = "report_*.json"
SCRIPT_GLOBS = ("*.m", "*_interface_check.txt")

PathLike = Union[str, Path]


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _ensure_pe_skeleton(base: Path = PE_BASE) -> None:
    for sub in PE_SUBDIRS:
        (base / sub).mkdir(parents=True, exist_ok=True)


@dataclass
class ClearResult:
    """Result of clearing experience + memory stores."""

    backup_dir: Optional[str] = None
    cleared: Dict[str, bool] = field(default_factory=dict)
    stamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArchiveResult:
    """Result of archiving scripts / reports for one task or batch."""

    archive_dir: str
    task_label: str
    moved_scripts: List[str] = field(default_factory=list)
    moved_reports: List[str] = field(default_factory=list)
    copied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clear_experience_and_memory(
    *,
    backup: bool = True,
    backup_root: PathLike = BACKUPS_DIR,
    clear_checkpoints: bool = True,
    clear_hermes_snapshots: bool = True,
) -> ClearResult:
    """Backup (optional) then wipe PE JSON, PE base dir, and Hermes memory.

    Recreates empty ``parameter_experience_base/params|models/{short,long}_term``
    so subsequent runs can write without missing-directory errors.
    """
    stamp = _stamp()
    result = ClearResult(stamp=stamp, cleared={})
    backup_root = Path(backup_root)
    backup_dir: Optional[Path] = None

    if backup:
        # Include pid so parallel master-seed processes don't collide on same stamp.
        backup_dir = backup_root / f"mem_backup_{stamp}_{os.getpid()}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        result.backup_dir = _rel(backup_dir)

    def _clear_file(path: Path, key: str) -> None:
        if path.is_file():
            if backup_dir is not None:
                shutil.copy2(path, backup_dir / path.name)
            path.unlink()
            result.cleared[key] = True
            logger.info("[file_mgr] cleared file %s", path)
        else:
            result.cleared[key] = False

    _clear_file(pe_json_path(), "parameter_experience_memory.json")
    _clear_file(hermes_json_path(), "hermes_agent_memory.json")
    if clear_hermes_snapshots:
        _clear_file(hermes_rl_best_path(), "hermes_rl_best.json")

    pe_base = pe_base_path()
    if pe_base.is_dir():
        if backup_dir is not None:
            shutil.copytree(
                pe_base,
                backup_dir / "parameter_experience_base",
                dirs_exist_ok=True,
            )
        if clear_checkpoints:
            shutil.rmtree(pe_base)
            result.cleared["parameter_experience_base"] = True
        else:
            # Keep checkpoints/; wipe params + models only
            for sub in ("params", "models"):
                target = pe_base / sub
                if target.is_dir():
                    shutil.rmtree(target)
            result.cleared["parameter_experience_base"] = True
        logger.info("[file_mgr] cleared experience base %s", pe_base)
    else:
        result.cleared["parameter_experience_base"] = False

    _ensure_pe_skeleton(pe_base)
    return result


def list_script_files(scripts_dir: PathLike = SCRIPTS_DIR) -> List[Path]:
    """List archive-worthy files under guidance_output/scripts."""
    root = Path(scripts_dir)
    if not root.is_dir():
        return []
    files: List[Path] = []
    for pattern in SCRIPT_GLOBS:
        files.extend(p for p in root.glob(pattern) if p.is_file())
    # de-dupe while preserving order
    seen = set()
    out: List[Path] = []
    for p in sorted(files, key=lambda x: x.name):
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def list_report_files(
    *,
    guidance_dir: PathLike = GUIDANCE_OUTPUT,
    reports_json_dir: PathLike = REPORTS_JSON_DIR,
) -> List[Path]:
    """List MD reports in guidance_output/ and JSON reports in reports/."""
    files: List[Path] = []
    gdir = Path(guidance_dir)
    if gdir.is_dir():
        files.extend(p for p in gdir.glob(REPORT_MD_GLOB) if p.is_file())
        latest = gdir / "report_latest.md"
        if latest.is_file():
            files.append(latest)
    jdir = Path(reports_json_dir)
    if jdir.is_dir():
        files.extend(p for p in jdir.glob(REPORT_JSON_GLOB) if p.is_file())
    seen = set()
    out: List[Path] = []
    for p in sorted(files, key=lambda x: (x.parent.name, x.name)):
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _move_or_copy(src: Path, dest: Path, *, copy: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dest)
    else:
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))


def archive_scripts_and_reports(
    archive_dir: PathLike,
    task_label: str = "task",
    *,
    scripts_dir: PathLike = SCRIPTS_DIR,
    guidance_dir: PathLike = GUIDANCE_OUTPUT,
    reports_json_dir: PathLike = REPORTS_JSON_DIR,
    copy: bool = False,
    only_newer_than: Optional[datetime] = None,
) -> ArchiveResult:
    """Archive scripts + reports into ``archive_dir/{task_label}/{scripts,reports}``.

    Parameters
    ----------
    archive_dir:
        Group (or batch) archive root, e.g. ``DATA/run_archives/group_...``.
    task_label:
        Subfolder name, typically ``T1`` / ``T2`` / …
    copy:
        If True, copy instead of move (working tree kept). Default moves so
        the next task starts with a clean scripts/reports area.
    only_newer_than:
        If set, only archive files whose mtime is strictly after this time
        (useful to skip leftovers from earlier runs).
    """
    group = Path(archive_dir)
    scripts_dest = group / task_label / "scripts"
    reports_dest = group / task_label / "reports"
    scripts_dest.mkdir(parents=True, exist_ok=True)
    reports_dest.mkdir(parents=True, exist_ok=True)

    result = ArchiveResult(
        archive_dir=_rel(group),
        task_label=task_label,
        copied=copy,
    )

    cutoff = only_newer_than.timestamp() if only_newer_than is not None else None

    for src in list_script_files(scripts_dir):
        if cutoff is not None and src.stat().st_mtime <= cutoff:
            continue
        dest = scripts_dest / src.name
        _move_or_copy(src, dest, copy=copy)
        result.moved_scripts.append(_rel(dest))

    for src in list_report_files(
        guidance_dir=guidance_dir, reports_json_dir=reports_json_dir
    ):
        if src.name == "report_latest.md" and not copy:
            # Keep a fresh working copy name out of archives as duplicate of stamped MD
            # when moving: still archive content under task folder, remove working latest.
            pass
        if cutoff is not None and src.stat().st_mtime <= cutoff:
            continue
        dest = reports_dest / src.name
        _move_or_copy(src, dest, copy=copy)
        result.moved_reports.append(_rel(dest))

    logger.info(
        "[file_mgr] archived task=%s scripts=%d reports=%d → %s",
        task_label,
        len(result.moved_scripts),
        len(result.moved_reports),
        result.archive_dir,
    )
    return result


def new_group_archive_dir(
    archive_root: PathLike = DEFAULT_ARCHIVE_ROOT,
    group_index: Optional[int] = None,
    stamp: Optional[str] = None,
) -> Path:
    """Create and return a new group archive directory."""
    root = Path(archive_root)
    root.mkdir(parents=True, exist_ok=True)
    ts = stamp or _stamp()
    if group_index is None:
        name = f"group_{ts}"
    else:
        name = f"group_{int(group_index):03d}_{ts}"
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_group_manifest(group_dir: PathLike, payload: Dict[str, Any]) -> Path:
    """Write / update ``manifest.json`` inside a group archive folder."""
    path = Path(group_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.update(payload)
    existing["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


class RunGroupSession:
    """Orchestrate clear-at-group-start + archive-after-each-task.

    Example
    -------
    ::

        session = RunGroupSession(group_index=1)
        session.begin_group()          # clear PE + Hermes memory
        for task in ("T1", "T2", "T3", "T4"):
            session.mark_task_start(task)
            # ... run cli_agent for task ...
            session.archive_after_task(task)
        session.end_group(status="completed")
    """

    DEFAULT_ORDER: Sequence[str] = ("T1", "T2", "T3", "T4")

    def __init__(
        self,
        *,
        group_index: Optional[int] = None,
        archive_root: PathLike = DEFAULT_ARCHIVE_ROOT,
        backup_on_clear: bool = True,
        clear_checkpoints: bool = True,
        archive_copy: bool = False,
    ) -> None:
        self.group_index = group_index
        self.archive_root = Path(archive_root)
        self.backup_on_clear = backup_on_clear
        self.clear_checkpoints = clear_checkpoints
        self.archive_copy = archive_copy
        self.group_dir: Optional[Path] = None
        self.clear_result: Optional[ClearResult] = None
        self.archives: List[ArchiveResult] = []
        self._task_started_at: Dict[str, datetime] = {}
        self._started_at: Optional[datetime] = None

    def begin_group(self) -> ClearResult:
        """Clear experience + memory, then open a fresh archive folder."""
        self._started_at = datetime.now()
        self.clear_result = clear_experience_and_memory(
            backup=self.backup_on_clear,
            clear_checkpoints=self.clear_checkpoints,
        )
        self.group_dir = new_group_archive_dir(
            self.archive_root,
            group_index=self.group_index,
            stamp=self.clear_result.stamp,
        )
        write_group_manifest(
            self.group_dir,
            {
                "group_index": self.group_index,
                "started_at": self._started_at.isoformat(timespec="seconds"),
                "status": "running",
                "order": list(self.DEFAULT_ORDER),
                "clear": self.clear_result.to_dict(),
                "tasks": {},
            },
        )
        logger.info(
            "[file_mgr] group start index=%s archive=%s",
            self.group_index,
            _rel(self.group_dir),
        )
        return self.clear_result

    def mark_task_start(self, task_label: str) -> datetime:
        """Record wall-clock start so archive can filter by mtime."""
        started = datetime.now()
        self._task_started_at[task_label] = started
        return started

    def archive_after_task(self, task_label: str) -> ArchiveResult:
        """Move/copy new scripts + reports into this group's archive."""
        if self.group_dir is None:
            raise RuntimeError("Call begin_group() before archive_after_task()")
        started = self._task_started_at.get(task_label)
        # Subtract 1s slack so files written in the same second are included
        cutoff = None
        if started is not None:
            cutoff = datetime.fromtimestamp(started.timestamp() - 1.0)

        result = archive_scripts_and_reports(
            self.group_dir,
            task_label=task_label,
            copy=self.archive_copy,
            only_newer_than=cutoff,
        )
        self.archives.append(result)
        write_group_manifest(
            self.group_dir,
            {
                "tasks": {
                    **{
                        a.task_label: {
                            "scripts": a.moved_scripts,
                            "reports": a.moved_reports,
                        }
                        for a in self.archives
                    }
                }
            },
        )
        return result

    def end_group(self, status: str = "completed", extra: Optional[Dict[str, Any]] = None) -> Path:
        """Finalize group manifest."""
        if self.group_dir is None:
            raise RuntimeError("Call begin_group() before end_group()")
        payload: Dict[str, Any] = {
            "status": status,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "archives": [a.to_dict() for a in self.archives],
        }
        if extra:
            payload.update(extra)
        path = write_group_manifest(self.group_dir, payload)
        logger.info("[file_mgr] group end status=%s → %s", status, _rel(self.group_dir))
        return path


def build_t_prompt(sub_idx: int, template_path: PathLike = ROOT / "prompt.txt") -> str:
    """Build a T{sub_idx} prompt from ``prompt.txt`` by rewriting SUB_IDX."""
    import re

    text = Path(template_path).read_text(encoding="utf-8")
    text, n = re.subn(
        r"(RUN_CASE\s*=\s*['\"]T['\"]\s*;\s*SUB_IDX\s*=\s*)\d+",
        rf"\g<1>{int(sub_idx)}",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    if n == 0:
        text, n = re.subn(
            r"(SUB_IDX\s*=\s*)\d+",
            rf"\g<1>{int(sub_idx)}",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    if n == 0:
        raise ValueError(
            f"Could not patch SUB_IDX in template {template_path}; "
            "expected RUN_CASE='T'; SUB_IDX=N;"
        )
    return text


def begin_experiment_group(
    tag: str,
    *,
    archive_root: PathLike = DEFAULT_ARCHIVE_ROOT,
    backup: bool = True,
    clear_checkpoints: bool = True,
) -> tuple[ClearResult, Path]:
    """Clear PE + Hermes memory and open ``archive_root/experiments/{tag}_{ts}/``."""
    clear_info = clear_experience_and_memory(
        backup=backup,
        clear_checkpoints=clear_checkpoints,
    )
    root = Path(archive_root) / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    group_dir = root / f"{safe}_{clear_info.stamp}"
    group_dir.mkdir(parents=True, exist_ok=True)
    write_group_manifest(
        group_dir,
        {
            "tag": tag,
            "status": "running",
            "clear": clear_info.to_dict(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    logger.info("[file_mgr] experiment group %s → %s", tag, _rel(group_dir))
    return clear_info, group_dir


__all__ = [
    "ArchiveResult",
    "ClearResult",
    "RunGroupSession",
    "archive_scripts_and_reports",
    "begin_experiment_group",
    "build_t_prompt",
    "clear_experience_and_memory",
    "list_report_files",
    "list_script_files",
    "new_group_archive_dir",
    "write_group_manifest",
    "DEFAULT_ARCHIVE_ROOT",
    "PE_BASE",
    "PE_JSON",
    "HERMES_JSON",
    "SCRIPTS_DIR",
]
