"""
MATLAB script seed selection for generate_matlab.

Balances safety (KB template / RL validation) with iterative MODIFY/TUNE
(gate-passed or Layer3 scripts as the next seed).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from multi_agent.logging.log_verbosity import vlog

logger = logging.getLogger(__name__)


@dataclass
class SeedChoice:
    content: str
    source: str
    path: str = ""


def is_kb_seed_source(source: str) -> bool:
    """True when seed came from knowledge-base monte_carlo template."""
    s = (source or "").lower()
    return (
        s.startswith("local:")
        or s.startswith("kb:")
        or "monte_carlo_single" in s
        or s == "not_found"
    )


def read_script_file(path: str) -> Optional[Tuple[str, str]]:
    """Return (content, source_label) or None."""
    if not path or not os.path.isfile(path):
        return None
    try:
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.debug("[SeedPolicy] read failed %s: %s", path, exc)
        return None
    if not content.strip():
        return None
    return content, f"file:{Path(path).name}"


def script_content_rl_ready(content: str) -> Tuple[bool, str]:
    from multi_agent.rl.matlab_rl_optimizer import validate_rl_script_content

    return validate_rl_script_content(content or "")


def script_path_rl_ready(path: str) -> Tuple[bool, str]:
    loaded = read_script_file(path)
    if not loaded:
        return False, "文件不存在或为空"
    ok, reason = script_content_rl_ready(loaded[0])
    return ok, reason


def pick_cli_seed_path(
    *,
    task_mode: str,
    gate_passed_script: str = "",
    layer3_script: str = "",
    hermes_script: str = "",
    seed_policy: str = "iterative",
) -> str:
    """
    Pick an on-disk script path for the next generate_matlab call.

    Returns empty string when the tool should fall back to PE / scripts scan / KB.
    """
    if seed_policy == "safe_kb":
        return ""

    mode = (task_mode or "").upper()
    if mode not in ("MODIFY_LAW", "TUNE_PARAMS"):
        return ""

    for label, path in (
        ("gate_passed", gate_passed_script),
        ("layer3", layer3_script),
        ("hermes", hermes_script),
    ):
        if not path:
            continue
        ok, reason = script_path_rl_ready(path)
        if ok:
            logger.info("[SeedPolicy] cli seed from %s → %s", label, Path(path).name)
            return path
        logger.debug("[SeedPolicy] skip %s %s: %s", label, path, reason)
    return ""


async def resolve_matlab_seed(
    *,
    mode: str,
    task_description: str,
    force_kb_template: bool,
    prefer_kb_template: bool,
    seed_policy: str,
    explicit_seed_path: str,
    gate_passed_path: str,
    layer3_script_path: str,
    load_pe_model: Callable,
    load_scripts_folder_model: Callable,
    load_kb_template: Callable,
) -> SeedChoice:
    """
    Resolve template content for generate_matlab.

    Priority (seed_policy=iterative, not force_kb):
      1. explicit_seed_path (cli set_iterative_seed)
      2. gate_passed_path
      3. layer3_script_path
      4. PE best model
      5. scripts folder (if not prefer_kb)
      6. KB template

    seed_policy=safe_kb mirrors legacy: PE → prefer_kb KB → scripts → KB.
    """
    if force_kb_template:
        content, source = await load_kb_template(task_description)
        if not content:
            return SeedChoice("", "not_found")
        logger.info("[SeedPolicy] force_kb_template → %s", source)
        return SeedChoice(content, source)

    if seed_policy == "iterative":
        for path in (explicit_seed_path, gate_passed_path, layer3_script_path):
            loaded = read_script_file(path)
            if not loaded:
                continue
            content, src = loaded
            ok, reason = script_content_rl_ready(content)
            if ok:
                logger.info(
                    "[SeedPolicy] iterative seed %s (src=%s)", Path(path).name, src
                )
                return SeedChoice(content, src, path=path)
            logger.warning(
                "[SeedPolicy] reject seed %s: %s", Path(path).name, reason
            )

    pe_result = await load_pe_model(task_description)
    if pe_result:
        content, source = pe_result
        ok, reason = script_content_rl_ready(content)
        if ok:
            logger.info("[SeedPolicy] PE seed → %s", source)
            return SeedChoice(content, source)
        logger.warning("[SeedPolicy] PE model not RL-ready: %s", reason)

    if prefer_kb_template or seed_policy == "safe_kb":
        content, source = await load_kb_template(task_description)
        if content:
            logger.info("[SeedPolicy] KB template → %s", source)
            return SeedChoice(content, source)

    scripts_result = load_scripts_folder_model(task_description)
    if scripts_result:
        content, source = scripts_result
        ok, reason = script_content_rl_ready(content)
        if ok:
            logger.info("[SeedPolicy] scripts folder → %s", source)
            return SeedChoice(content, source)
        logger.warning("[SeedPolicy] scripts scan reject %s: %s", source, reason)

    content, source = await load_kb_template(task_description)
    if content:
        logger.info("[SeedPolicy] KB fallback → %s", source)
        return SeedChoice(content, source)

    return SeedChoice("", "not_found")


def should_apply_deterministic_apn_on_seed(
    *,
    mission_conditions: Optional[str],
    task_prompt: Optional[str],
    seed_source: str,
    mode: str = "MODIFY_LAW",
) -> bool:
    """
    When deterministic_apn_kb_only is true (default), only stamp built-in APN
    on KB seeds — iterative MODIFY/TUNE keeps the gate-passed gf() structure.
    """
    from multi_agent.integration.t4_low_risk import use_deterministic_apn

    if not use_deterministic_apn(mission_conditions, task_prompt):
        return False
    try:
        from multi_agent.config_loader import get_config

        kb_only = bool(getattr(get_config().t4_low_risk, "deterministic_apn_kb_only", True))
    except Exception:
        kb_only = True
    if kb_only and not is_kb_seed_source(seed_source):
        logger.info(
            "[SeedPolicy] skip deterministic APN on seed %s (mode=%s, kb_only=True)",
            seed_source,
            mode,
        )
        return False
    return True


__all__ = [
    "SeedChoice",
    "is_kb_seed_source",
    "pick_cli_seed_path",
    "read_script_file",
    "resolve_matlab_seed",
    "script_content_rl_ready",
    "script_path_rl_ready",
    "should_apply_deterministic_apn_on_seed",
]
