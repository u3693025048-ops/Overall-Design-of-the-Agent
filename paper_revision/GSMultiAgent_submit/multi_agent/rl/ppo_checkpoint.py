"""Save / restore PPO (Actor-Critic) checkpoints for resume after interrupt."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from multi_agent.rl.matlab_rl_optimizer import ActorNet, CriticNet, MatlabRLOptimizer

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


def _numpy_net_arrays(net: Any) -> Dict[str, np.ndarray]:
    out = {
        "W1": net.W1.copy(),
        "b1": net.b1.copy(),
        "W2": net.W2.copy(),
        "b2": net.b2.copy(),
    }
    if hasattr(net, "log_std"):
        out["log_std"] = net.log_std.copy()
    return out


def _load_numpy_net(net: Any, arrays: Dict[str, np.ndarray]) -> None:
    net.W1 = np.array(arrays["W1"], dtype=np.float32)
    net.b1 = np.array(arrays["b1"], dtype=np.float32)
    net.W2 = np.array(arrays["W2"], dtype=np.float32)
    net.b2 = np.array(arrays["b2"], dtype=np.float32)
    if hasattr(net, "log_std") and "log_std" in arrays:
        net.log_std = np.array(arrays["log_std"], dtype=np.float32)


def checkpoint_stem(run_id: str, episode: int) -> str:
    return f"ppo_{run_id}_ep{episode:05d}"


class PPOCheckpointManager:
    """Writes ``<stem>.npz`` (weights) + ``<stem>.json`` (metadata) + ``latest.json``."""

    def __init__(
        self,
        checkpoint_dir: str = "./parameter_experience_base/checkpoints/ppo",
        save_every_episodes: int = 10,
        keep_last_n: int = 5,
        enabled: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_every_episodes = max(1, int(save_every_episodes))
        self.keep_last_n = max(1, int(keep_last_n))
        self.enabled = bool(enabled)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def set_run_id(self, run_id: str) -> None:
        self.run_id = run_id

    def _stem_path(self, stem: str) -> str:
        return os.path.join(self.checkpoint_dir, stem)

    def save(
        self,
        optimizer: "MatlabRLOptimizer",
        *,
        episode: int,
        max_episodes: int,
        script_path: Optional[str] = None,
        nmc: int = 10,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.enabled or optimizer._actor is None or optimizer._critic is None:
            return None

        stem = checkpoint_stem(self.run_id, episode)
        base = self._stem_path(stem)
        npz_path = base + ".npz"
        json_path = base + ".json"

        arrays: Dict[str, np.ndarray] = {}
        for prefix, net in (("actor", optimizer._actor), ("critic", optimizer._critic)):
            for k, v in _numpy_net_arrays(net).items():
                arrays[f"{prefix}_{k}"] = v

        np.savez_compressed(npz_path, **arrays)

        meta: Dict[str, Any] = {
            "version": CHECKPOINT_VERSION,
            "run_id": self.run_id,
            "episode_completed": int(episode),
            "max_episodes": int(max_episodes),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "npz_path": os.path.basename(npz_path),
            "script_path": script_path or "",
            "nmc": int(nmc),
            "best_reward": float(optimizer._best_reward),
            "best_params": dict(optimizer._best_params),
            "best_metrics": dict(optimizer._best_metrics),
            "best_constrained_params": dict(optimizer._best_constrained_params),
            "best_constrained_metrics": dict(optimizer._best_constrained_metrics),
            "best_constrained_fitness": float(optimizer._best_constrained_fitness),
            "best_miss_ever": float(optimizer._best_miss_ever),
            "best_peak_n_ever": float(optimizer._best_peak_n_ever),
            "last_constrained_improve_ep": int(optimizer._last_constrained_improve_ep),
            "current_params": dict(optimizer._current_params),
            "last_metrics": dict(optimizer._last_metrics),
            "base_auto": dict(optimizer._base_auto),
            "action_keys": list(optimizer._action_keys),
            "state_dim": int(optimizer._state_dim),
            "action_dim": int(optimizer._action_dim),
        }
        if extra:
            meta["extra"] = extra

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        latest = {
            "run_id": self.run_id,
            "stem": stem,
            "json_path": os.path.basename(json_path),
            "npz_path": os.path.basename(npz_path),
            "episode_completed": int(episode),
            "updated_at": meta["saved_at"],
        }
        latest_path = os.path.join(self.checkpoint_dir, "latest.json")
        with open(latest_path, "w", encoding="utf-8") as fh:
            json.dump(latest, fh, ensure_ascii=False, indent=2)

        self._prune_old_checkpoints(stem)
        logger.info(
            f"[Checkpoint] Saved ep {episode} → {json_path} "
            f"(best_SEP_ever={meta['best_miss_ever']:.2f}m, "
            f"best_reward={meta['best_reward']:.3f})"
        )
        return json_path

    def _prune_old_checkpoints(self, current_stem: str) -> None:
        if self.keep_last_n <= 0:
            return
        stems: List[str] = []
        for name in os.listdir(self.checkpoint_dir):
            if name.startswith(f"ppo_{self.run_id}_ep") and name.endswith(".json"):
                stems.append(name[:-5])
        stems.sort()
        to_remove = stems[: max(0, len(stems) - self.keep_last_n)]
        for stem in to_remove:
            if stem == current_stem:
                continue
            for ext in (".json", ".npz"):
                p = self._stem_path(stem) + ext
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                        logger.debug(f"[Checkpoint] pruned {p}")
                    except OSError:
                        pass

    @staticmethod
    def _resolve_from_latest_json(latest_path: str) -> Optional[str]:
        if not os.path.isfile(latest_path):
            return None
        base_dir = os.path.dirname(latest_path)
        with open(latest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        stem = data.get("stem", "")
        cand = os.path.join(base_dir, data.get("json_path", stem + ".json"))
        if os.path.isfile(cand):
            return cand
        return None

    @staticmethod
    def resolve_checkpoint_path(path: Optional[str]) -> Optional[str]:
        """Accept meta .json path, stem, latest.json, or checkpoint directory."""
        if not path:
            return None
        path = os.path.abspath(path)
        if os.path.isdir(path):
            return PPOCheckpointManager._resolve_from_latest_json(
                os.path.join(path, "latest.json")
            )
        if os.path.basename(path) == "latest.json":
            return PPOCheckpointManager._resolve_from_latest_json(path)
        if path.endswith(".json") and os.path.isfile(path):
            return path
        if os.path.isfile(path + ".json"):
            return path + ".json"
        return None

    def load(
        self,
        optimizer: "MatlabRLOptimizer",
        checkpoint_json: str,
    ) -> Dict[str, Any]:
        json_path = self.resolve_checkpoint_path(checkpoint_json)
        if not json_path or not os.path.isfile(json_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_json}")

        with open(json_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        if meta.get("version") != CHECKPOINT_VERSION:
            logger.warning(
                f"[Checkpoint] version mismatch: file={meta.get('version')} "
                f"expected={CHECKPOINT_VERSION}"
            )

        npz_path = os.path.join(
            os.path.dirname(json_path),
            meta.get("npz_path", os.path.basename(json_path).replace(".json", ".npz")),
        )
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Checkpoint weights missing: {npz_path}")

        arrays = np.load(npz_path)
        if optimizer._actor is None or optimizer._critic is None:
            optimizer._init_networks()

        actor_arrays = {
            k.replace("actor_", ""): arrays[f"actor_{k}"]
            for k in ("W1", "b1", "W2", "b2", "log_std")
            if f"actor_{k}" in arrays
        }
        critic_arrays = {
            k.replace("critic_", ""): arrays[f"critic_{k}"]
            for k in ("W1", "b1", "W2", "b2")
            if f"critic_{k}" in arrays
        }
        _load_numpy_net(optimizer._actor, actor_arrays)
        _load_numpy_net(optimizer._critic, critic_arrays)

        optimizer._best_reward = float(meta.get("best_reward", -1e9))
        optimizer._best_params = dict(meta.get("best_params", {}))
        optimizer._best_metrics = dict(meta.get("best_metrics", {}))
        optimizer._best_constrained_params = dict(meta.get("best_constrained_params", {}))
        optimizer._best_constrained_metrics = dict(meta.get("best_constrained_metrics", {}))
        optimizer._best_constrained_fitness = float(meta.get("best_constrained_fitness", -1.0))
        optimizer._best_miss_ever = float(meta.get("best_miss_ever", float("inf")))
        optimizer._best_peak_n_ever = float(meta.get("best_peak_n_ever", float("inf")))
        optimizer._last_constrained_improve_ep = int(meta.get("last_constrained_improve_ep", 0))
        optimizer._current_params = dict(meta.get("current_params", optimizer._base_auto))
        optimizer._last_metrics = dict(meta.get("last_metrics", optimizer._last_metrics))
        for k, v in meta.get("base_auto", {}).items():
            if k in optimizer._base_auto:
                optimizer._base_auto[k] = float(v)

        self.run_id = str(meta.get("run_id", self.run_id))
        meta["_json_path"] = json_path
        meta["_npz_path"] = npz_path
        logger.info(
            f"[Checkpoint] Restored from {json_path} | "
            f"resume at ep {meta.get('episode_completed', 0) + 1} | "
            f"run_id={self.run_id} | best_SEP_ever={optimizer._best_miss_ever:.2f}m"
        )
        return meta
