"""
Central log verbosity for CLI / RL / Hermes.

Levels (config ``logging.verbosity`` or CLI ``--verbose`` / ``--quiet``):
  quiet   — milestones + warnings/errors only
  normal  — default; compact progress (default)
  verbose — full diagnostic (legacy behaviour)
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

Verbosity = Literal["quiet", "normal", "verbose"]

_LEVELS = {"quiet": 0, "normal": 1, "verbose": 2}
_current: Verbosity = "normal"
_rl_episode_interval: int = 10


def set_rl_episode_interval(n: int) -> None:
    global _rl_episode_interval
    _rl_episode_interval = max(1, int(n))

# Third-party loggers demoted when not verbose
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "run_agent",
    "agent.conversation_loop",
    "agent.tool_executor",
    "agent.context_compressor",
    "agent.auxiliary_client",
    "langchain_openai",
    "chromadb",
    "multi_agent.memory.rag_knowledge_base",
)

_PROJECT_LOGGERS = (
    "multi_agent.tools.simulation_tool",
    "multi_agent.integration",
    "multi_agent.simulation.optimization_workflow",
    "multi_agent.rl.matlab_rl_optimizer",
    "multi_agent.rl.constraint_local_search",
    "multi_agent.rl.rolling_stats",
    "multi_agent.integration.script_seed_policy",
)


def set_verbosity(level: str) -> None:
    global _current
    key = (level or "normal").strip().lower()
    if key not in _LEVELS:
        key = "normal"
    _current = key  # type: ignore[assignment]


def get_verbosity() -> Verbosity:
    return _current


def is_quiet() -> bool:
    return _current == "quiet"


def is_verbose() -> bool:
    return _current == "verbose"


def is_normal_or_quiet() -> bool:
    return not is_verbose()


def apply_third_party_log_levels() -> None:
    """Call once at CLI startup after set_verbosity."""
    if is_verbose():
        return
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    if is_quiet():
        logging.getLogger("multi_agent.tools").setLevel(logging.WARNING)
        logging.getLogger("multi_agent.integration").setLevel(logging.WARNING)


def configure_logging() -> None:
    """Apply root + project logger levels from current verbosity."""
    apply_third_party_log_levels()
    root = logging.getLogger()
    if is_quiet():
        root.setLevel(logging.WARNING)
    elif is_verbose():
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(logging.INFO)
    if is_verbose():
        return
    proj_level = logging.WARNING if is_quiet() else logging.INFO
    for name in _PROJECT_LOGGERS:
        logging.getLogger(name).setLevel(proj_level)


def init_verbosity_from_config(cfg: Any, args: Optional[Any] = None) -> Verbosity:
    """Read config + CLI flags, apply logging levels."""
    level = "normal"
    if args is not None:
        if getattr(args, "verbose", False):
            level = "verbose"
        elif getattr(args, "quiet", False):
            level = "quiet"
    if level == "normal":
        log_cfg = getattr(cfg, "logging", None)
        level = str(getattr(log_cfg, "verbosity", "normal") or "normal").strip().lower()
        if level not in _LEVELS:
            level = "normal"
    set_verbosity(level)
    log_cfg = getattr(cfg, "logging", None)
    set_rl_episode_interval(int(getattr(log_cfg, "rl_episode_interval", 10)))
    configure_logging()
    return get_verbosity()


def hermes_verbose_thinking() -> bool:
    return is_verbose()


def cli_debug_enabled() -> bool:
    return is_verbose()


def rl_episode_log_interval() -> int:
    if is_verbose():
        return 1
    if is_quiet():
        return max(25, _rl_episode_interval * 2)
    return _rl_episode_interval


def should_log_rl_episode(ep: int, max_ep: int, *, new_best: bool = False) -> bool:
    if is_verbose() or new_best:
        return True
    if ep == 0 or ep + 1 >= max_ep:
        return True
    interval = rl_episode_log_interval()
    return (ep + 1) % interval == 0


def should_log_expert_round(round_i: int, max_rounds: int, *, improved: bool = False) -> bool:
    if is_verbose() or improved:
        return True
    return round_i >= max_rounds


def should_log_cls_step(step: int, max_steps: int, *, improved: bool = False) -> bool:
    if is_verbose() or improved:
        return True
    if step >= max_steps:
        return True
    return step % max(10, rl_episode_log_interval()) == 0


def task_summary_line(prompt: str, max_len: int = 160) -> str:
    one_line = " ".join((prompt or "").split())
    return truncate_text(one_line, max_len)


def mprint(*args, **kwargs) -> None:
    print(*args, **kwargs)


def cprint(*args, **kwargs) -> None:
    if not is_quiet():
        print(*args, **kwargs)


def dprint(*args, **kwargs) -> None:
    if is_verbose():
        print(*args, **kwargs)


def vlog(logger: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Verbose-only info."""
    if is_verbose():
        logger.info(msg, *args, **kwargs)
    else:
        logger.debug(msg, *args, **kwargs)


def truncate_text(text: str, max_len: int = 240) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
