#!/usr/bin/env python3
"""
hermes_agent_memory.py
Two-tier persistent memory for the Hermes Agent in GSMultiAgent13.

Pattern follows hermes-agent-demo/memory.py:
  • Long-term  : key-value store persisted to JSON on disk between runs
  • Short-term : in-session conversation history (list of messages)

The long-term store is exposed to the LLM via four tool objects defined in
multi_agent/tools/memory_tools.py and registered with the Hermes registry in
hermes_integration.py.  On every run a compact summary of stored memories is
injected into the Hermes system prompt so the agent "knows" what it remembered
without having to call agent_memory_recall first.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class HermesAgentMemory:
    """Persistent key-value memory + in-session conversation history.

    Parameters
    ----------
    memory_file : str | Path
        Path to the JSON file used for persistence.  Parent directories are
        created automatically on first write.  Defaults to
        ``./guidance_output/hermes_agent_memory.json``.
    """

    DEFAULT_PATH = "./guidance_output/hermes_agent_memory.json"

    def __init__(self, memory_file: str = None) -> None:
        path = memory_file or self.DEFAULT_PATH
        self._path: Path = Path(path).expanduser().resolve()
        self._store: Dict[str, Any] = self._load()
        self.conversation: List[dict] = []
        logger.info(
            f"[HermesAgentMemory] Loaded {len(self._store)} entr"
            f"{'y' if len(self._store) == 1 else 'ies'} from {self._path}"
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    f"[HermesAgentMemory] Could not load {self._path}: {exc}. "
                    "Starting with empty store."
                )
        return {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Long-term memory operations ──────────────────────────────────────────

    def remember(self, key: str, value: str) -> str:
        """Store *value* under *key* and persist to disk."""
        self._store[key] = {
            "value": value,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.save()
        return f"Remembered '{key}' = {value!r}"

    def recall(self, key: str) -> str:
        """Retrieve the value stored under *key*."""
        if key not in self._store:
            return f"No memory found for key '{key}'."
        entry = self._store[key]
        return f"[{entry['saved_at']}] {key} = {entry['value']!r}"

    def forget(self, key: str) -> str:
        """Delete the memory stored under *key*."""
        if key in self._store:
            del self._store[key]
            self.save()
            return f"Forgot '{key}'."
        return f"Key '{key}' was not in memory."

    def list_memories(self) -> str:
        """Return all stored keys and their saved timestamps."""
        if not self._store:
            return "No memories stored yet."
        lines = [
            f"  • {k}  (saved {v['saved_at']})"
            for k, v in self._store.items()
        ]
        return "Stored memories:\n" + "\n".join(lines)

    # ── Short-term conversation history ──────────────────────────────────────

    def add_message(self, role: str, content: str, **kwargs) -> None:
        msg: dict = {"role": role, "content": content}
        msg.update(kwargs)
        self.conversation.append(msg)

    def get_messages(self) -> List[dict]:
        return self.conversation.copy()

    def clear_conversation(self) -> None:
        self.conversation.clear()

    def clear_all(self) -> None:
        """Wipe long-term store + conversation and persist empty JSON."""
        self._store.clear()
        self.conversation.clear()
        self.save()
        logger.info("[HermesAgentMemory] Cleared all memories at %s", self._path)

    # ── System-prompt context injection ──────────────────────────────────────

    def get_context_block(self) -> str:
        """Return a compact block for injection into the Hermes system prompt.

        When the store is empty, returns an empty string (no noise added to
        the prompt).  Otherwise returns a fenced section listing every key,
        its value, and the timestamp it was saved, so the LLM is immediately
        aware of previously remembered facts without having to call
        agent_memory_list first.
        """
        if not self._store:
            return ""
        lines = []
        for k, v in self._store.items():
            lines.append(f"  [{v['saved_at']}] {k}: {v['value']}")
        body = "\n".join(lines)
        return (
            "════════════════════════════════════════\n"
            "【持久记忆 — 来自上次会话的已保存信息】\n"
            "════════════════════════════════════════\n"
            f"{body}\n"
            "（如需更新或删除，调用 agent_memory_remember / agent_memory_forget）\n\n"
        )

    # ── Utility ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        n = len([m for m in self.conversation if m["role"] == "user"])
        keys = len(self._store)
        return (
            f"{n} user turn(s) this session | "
            f"{keys} long-term memor{'y' if keys == 1 else 'ies'}"
        )

    def __len__(self) -> int:
        return len(self._store)
