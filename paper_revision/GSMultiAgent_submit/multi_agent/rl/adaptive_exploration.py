#!/usr/bin/env python3
"""
自适应多区域优化（AMRO）模块

处理参数空间中的跳变区域（不连续性），通过以下机制提升 RL 优化效率：
1. 跳变边界检测：监测奖励函数的梯度突变
2. 禁区标记：记录低效参数组合，避免重复探索
3. 大步长跳跃：在跳变区域执行大幅度参数调整
4. 自适应探索：根据区域特性动态调整 action 的缩放因子
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegionInfo:
    """参数空间区域信息"""
    region_id: int
    center: np.ndarray
    radius: float
    reward_mean: float
    reward_std: float
    is_continuous: bool
    episode_count: int = 0
    last_episode: int = 0
    gradient_magnitude: float = 0.0


@dataclass
class JumpZone:
    """跳变区域（不连续性区域）"""
    param_name: str
    threshold_value: float
    reward_drop: float
    direction: str
    episode_detected: int
    severity: float


@dataclass
class AdaptiveExplorationState:
    """Runtime state for episode-wise exploration scaling."""
    steepness_ema: float = 0.0
    neighborhood_steepness: float = 0.0
    explore_mode: str = "aggressive"
    log_std_scale: float = 1.0


class AdaptiveExplorationController:
    """
    Reward-driven adaptive entropy / log_std scaling for contextual-bandit PPO.

    Shrinks exploration when:
      - per-episode reward gradient is steep (|Δreward| / mean|Δparam|)
      - neighborhood probes report high steepness
      - consecutive low rewards (≈ hard truncation at -10)
    """

    def __init__(
        self,
        steepness_threshold: float = 2.0,
        log_std_min_scale: float = 0.35,
        log_std_max_scale: float = 1.0,
        ema_alpha: float = 0.3,
        low_reward_threshold: float = -9.5,
    ) -> None:
        self.steepness_threshold = float(steepness_threshold)
        self.log_std_min_scale = float(log_std_min_scale)
        self.log_std_max_scale = float(log_std_max_scale)
        self.ema_alpha = float(ema_alpha)
        self.low_reward_threshold = float(low_reward_threshold)
        self.state = AdaptiveExplorationState(log_std_scale=self.log_std_max_scale)
        self._prev_params: Dict[str, float] = {}
        self._prev_reward: Optional[float] = None

    @staticmethod
    def _param_delta(
        params_prev: Dict[str, float],
        params_curr: Dict[str, float],
    ) -> float:
        keys = set(params_prev) | set(params_curr)
        if not keys:
            return 0.0
        total = 0.0
        for k in keys:
            try:
                total += abs(float(params_curr.get(k, 0.0)) - float(params_prev.get(k, 0.0)))
            except (TypeError, ValueError):
                continue
        return total / max(len(keys), 1)

    @staticmethod
    def _episode_steepness(
        params_prev: Dict[str, float],
        params_curr: Dict[str, float],
        reward_prev: float,
        reward_curr: float,
    ) -> float:
        delta_r = abs(float(reward_curr) - float(reward_prev))
        delta_p = AdaptiveExplorationController._param_delta(params_prev, params_curr)
        if delta_p < 1e-6:
            return 0.0
        return delta_r / delta_p

    def _apply_mode_from_steepness(self) -> None:
        steep = max(self.state.steepness_ema, self.state.neighborhood_steepness)
        if steep >= self.steepness_threshold:
            self.state.explore_mode = "fine"
            self.state.log_std_scale = self.log_std_min_scale
        else:
            self.state.explore_mode = "aggressive"
            self.state.log_std_scale = self.log_std_max_scale

    def update_episode(
        self,
        params_prev: Dict[str, float],
        params_curr: Dict[str, float],
        reward_prev: float,
        reward_curr: float,
        *,
        new_best: bool = False,
    ) -> None:
        steep = self._episode_steepness(params_prev, params_curr, reward_prev, reward_curr)
        if params_prev or params_curr:
            self.state.steepness_ema = (
                self.ema_alpha * steep
                + (1.0 - self.ema_alpha) * self.state.steepness_ema
            )
        if new_best and steep >= self.steepness_threshold * 0.5:
            self.state.steepness_ema = max(self.state.steepness_ema, steep)
        self._apply_mode_from_steepness()
        self._prev_params = dict(params_curr)
        self._prev_reward = float(reward_curr)

    def ingest_neighborhood_steepness(self, steepness: float, blend: float = 1.0) -> None:
        b = float(np.clip(blend, 0.0, 1.0))
        self.state.neighborhood_steepness = (
            b * float(steepness) + (1.0 - b) * self.state.neighborhood_steepness
        )
        self._apply_mode_from_steepness()

    def on_low_reward(self) -> None:
        """Called when reward hits hard-truncation floor — shrink exploration."""
        self.state.explore_mode = "fine"
        self.state.log_std_scale = max(
            self.log_std_min_scale,
            self.state.log_std_scale * 0.65,
        )
        logger.debug(
            "[AdaptiveExplore] low reward (≤%.1f) → log_std_scale=%.3f mode=%s",
            self.low_reward_threshold,
            self.state.log_std_scale,
            self.state.explore_mode,
        )

    def explore_extra(self) -> Dict[str, float]:
        return {
            "log_std_scale": float(self.state.log_std_scale),
            "explore_mode": self.state.explore_mode,
            "steepness_ema": float(self.state.steepness_ema),
            "neighborhood_steepness": float(self.state.neighborhood_steepness),
        }


class AdaptiveExplorer:
    """
    自适应探索管理器
    
    核心功能：
    - 监测参数空间的不连续性
    - 标记禁区（低效区域）
    - 根据区域特性调整探索策略
    - 提供自适应的 action 缩放
    """

    def __init__(
        self,
        param_names: List[str],
        param_specs: Dict[str, Dict[str, float]],
        history_window: int = 10,
        gradient_threshold: float = 0.5,
        forbidden_zone_radius: float = 0.15,
    ):
        self.param_names = param_names
        self.param_specs = param_specs
        self.history_window = history_window
        self.gradient_threshold = gradient_threshold
        self.forbidden_zone_radius = forbidden_zone_radius

        self.episode_history: List[Dict] = []
        self.jump_zones: List[JumpZone] = []
        self.forbidden_zones: List[RegionInfo] = []
        self.regions: Dict[int, RegionInfo] = {}
        self.region_counter = 0

        self.total_episodes = 0
        self.wasted_episodes = 0
        self.jump_detections = 0

    def record_episode(
        self,
        episode: int,
        params: Dict[str, float],
        reward: float,
        state: np.ndarray,
    ) -> None:
        """记录一个 episode 的结果"""
        self.total_episodes = episode
        self.episode_history.append({
            "episode": episode,
            "params": params.copy(),
            "reward": reward,
            "state": state.copy(),
        })

        if len(self.episode_history) > self.history_window * 2:
            self.episode_history = self.episode_history[-self.history_window * 2:]

        self._detect_jumps(episode, params, reward)
        self._update_regions(params, reward)

    def _detect_jumps(
        self, episode: int, params: Dict[str, float], reward: float
    ) -> None:
        """检测参数空间中的跳变（不连续性）"""
        if len(self.episode_history) < 2:
            return

        prev = self.episode_history[-2]
        reward_drop = prev["reward"] - reward

        if reward_drop < self.gradient_threshold:
            return

        for pname in self.param_names:
            prev_val = prev["params"].get(pname, 0.0)
            curr_val = params.get(pname, 0.0)
            if abs(curr_val - prev_val) < 1e-6:
                continue

            jump = JumpZone(
                param_name=pname,
                threshold_value=curr_val,
                reward_drop=reward_drop,
                direction="up" if curr_val > prev_val else "down",
                episode_detected=episode,
                severity=min(1.0, reward_drop / 5.0),
            )
            self.jump_zones.append(jump)
            self.jump_detections += 1
            logger.debug(
                f"[AdaptiveExplorer] Jump detected at {pname}={curr_val:.3f}, "
                f"reward drop={reward_drop:.2f}"
            )
            break

    def _update_regions(self, params: Dict[str, float], reward: float) -> None:
        state_vec = self._params_to_state_vec(params)
        best_region = None
        best_dist = float("inf")

        for region in self.regions.values():
            dist = np.linalg.norm(state_vec - region.center)
            if dist < region.radius and dist < best_dist:
                best_dist = dist
                best_region = region

        if best_region is not None:
            n = best_region.episode_count + 1
            best_region.reward_mean = (
                best_region.reward_mean * best_region.episode_count + reward
            ) / n
            best_region.episode_count = n
            best_region.last_episode = self.total_episodes
        else:
            self.region_counter += 1
            self.regions[self.region_counter] = RegionInfo(
                region_id=self.region_counter,
                center=state_vec,
                radius=self.forbidden_zone_radius,
                reward_mean=reward,
                reward_std=0.0,
                is_continuous=True,
                episode_count=1,
                last_episode=self.total_episodes,
            )

    def _params_to_state_vec(self, params: Dict[str, float]) -> np.ndarray:
        vec = []
        for pname in self.param_names:
            spec = self.param_specs.get(pname, {"min": 0, "max": 1, "nominal": 0.5})
            mid = (spec["min"] + spec["max"]) / 2.0
            half = (spec["max"] - spec["min"]) / 2.0
            val = params.get(pname, spec.get("nominal", mid))
            vec.append((val - mid) / max(half, 1e-6))
        return np.array(vec, dtype=np.float32)

    def is_in_forbidden_zone(self, params: Dict[str, float]) -> bool:
        state_vec = self._params_to_state_vec(params)
        for region in self.forbidden_zones:
            if np.linalg.norm(state_vec - region.center) < region.radius:
                return True
        return False

    def get_adaptive_action_scale(
        self, params: Dict[str, float], base_scale: float = 1.0
    ) -> float:
        if self.is_in_forbidden_zone(params):
            return 0.0

        state_vec = self._params_to_state_vec(params)

        min_dist_to_jump = float("inf")
        for jump in self.jump_zones[-5:]:
            pname = jump.param_name
            if pname in self.param_names:
                idx = self.param_names.index(pname)
                dist = abs(state_vec[idx])
                min_dist_to_jump = min(min_dist_to_jump, dist)

        if min_dist_to_jump < 0.3:
            scale = 1.0 + (0.3 - min_dist_to_jump) / 0.3 * 1.0
            return base_scale * scale

        return base_scale

    def get_exploration_suggestion(self) -> Optional[str]:
        if not self.jump_zones:
            return None
        latest = self.jump_zones[-1]
        return (
            f"Detected jump at {latest.param_name}="
            f"{latest.threshold_value:.3f} (reward drop {latest.reward_drop:.2f})"
        )


class AdaptiveActionModifier:
    """Modify Actor action based on parameter-space characteristics."""

    def __init__(self, explorer: AdaptiveExplorer):
        self.explorer = explorer

    def modify_action(
        self,
        action: np.ndarray,
        params: Dict[str, float],
        param_names: List[str],
    ) -> np.ndarray:
        modified = action.copy()

        for i, pname in enumerate(param_names):
            spec = self.explorer.param_specs[pname]
            mid = (spec["min"] + spec["max"]) / 2.0
            half = (spec["max"] - spec["min"]) / 2.0
            new_val = np.clip(mid + float(action[i]) * half, spec["min"], spec["max"])

            test_params = params.copy()
            test_params[pname] = new_val

            if self.explorer.is_in_forbidden_zone(test_params):
                modified[i] = -action[i]

        scale = self.explorer.get_adaptive_action_scale(params)
        if scale < 1.0:
            modified *= scale
        elif scale > 1.0:
            modified *= min(scale, 2.0)

        return modified
