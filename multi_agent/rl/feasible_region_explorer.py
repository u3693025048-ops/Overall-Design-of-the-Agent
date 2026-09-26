#!/usr/bin/env python3
"""
Feasible Region Explorer (FRE)

逐步探索可行域的算法。针对T4工况可行域狭窄、容易跳变的问题。

核心思想：
1. 先用宽松的约束探索，逐步紧化约束
2. 在找到可行解后，逐步改进
3. 避免在约束边界处跳变
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from multi_agent.metric_contract import HIT_RATE_MIN, SEP_MAX

logger = logging.getLogger(__name__)


class FeasibleRegionExplorer:
    """逐步探索可行域的探索器"""
    
    def __init__(
        self,
        initial_sep_threshold: float = 100.0,  # 初始SEP宽松约束（m）
        initial_hit_rate_threshold: float = 50.0,  # 初始命中率宽松约束（%）
        target_sep_threshold: float = 7.0,  # 目标SEP约束（m）
        target_hit_rate_threshold: float = 92.0,  # 目标命中率约束（%）
        phase_episodes: int = 20,  # 每个阶段的episode数
        num_phases: int = 3,  # 约束紧化的阶段数
    ):
        """
        Args:
            initial_sep_threshold: 初始SEP约束（宽松）
            initial_hit_rate_threshold: 初始命中率约束（宽松）
            target_sep_threshold: 目标SEP约束（严格）
            target_hit_rate_threshold: 目标命中率约束（严格）
            phase_episodes: 每个阶段的episode数
            num_phases: 约束紧化的阶段数
        """
        if target_sep_threshold == 7.0:
            target_sep_threshold = SEP_MAX
        if target_hit_rate_threshold == 92.0:
            target_hit_rate_threshold = HIT_RATE_MIN
        self.initial_sep_threshold = initial_sep_threshold
        self.initial_hit_rate_threshold = initial_hit_rate_threshold
        self.target_sep_threshold = target_sep_threshold
        self.target_hit_rate_threshold = target_hit_rate_threshold
        self.phase_episodes = phase_episodes
        self.num_phases = num_phases
        
        # 当前阶段
        self.current_phase = 0
        self.episodes_in_phase = 0
        
        # 约束轨迹（从宽松到严格）
        self.sep_thresholds = self._generate_thresholds(
            initial_sep_threshold, target_sep_threshold, num_phases
        )
        self.hit_rate_thresholds = self._generate_thresholds(
            initial_hit_rate_threshold, target_hit_rate_threshold, num_phases
        )
        
        # 统计信息
        self.phase_best_rewards: List[float] = []
        self.phase_feasible_count: List[int] = []
        self.phase_transition_episodes: List[int] = []
        
        logger.info(
            f"[FRE] Initialized with {num_phases} phases | "
            f"SEP: {initial_sep_threshold:.1f}m → {target_sep_threshold:.1f}m | "
            f"HitRate: {initial_hit_rate_threshold:.1f}% → {target_hit_rate_threshold:.1f}%"
        )
        logger.info(
            f"[FRE] SEP thresholds: {[f'{t:.1f}' for t in self.sep_thresholds]}"
        )
        logger.info(
            f"[FRE] HitRate thresholds: {[f'{t:.1f}' for t in self.hit_rate_thresholds]}"
        )
    
    @staticmethod
    def _generate_thresholds(
        initial: float, target: float, num_phases: int
    ) -> List[float]:
        """生成从初始值到目标值的线性插值阈值"""
        return [
            initial + (target - initial) * i / (num_phases - 1)
            for i in range(num_phases)
        ]
    
    def get_current_thresholds(self) -> Tuple[float, float]:
        """获取当前阶段的约束阈值"""
        return (
            self.sep_thresholds[self.current_phase],
            self.hit_rate_thresholds[self.current_phase],
        )
    
    def is_feasible(self, metrics: Dict[str, float]) -> bool:
        """检查是否满足当前阶段的可行性约束"""
        sep_threshold, hit_rate_threshold = self.get_current_thresholds()
        
        sep = metrics.get("SEP", metrics.get("miss_distance", float("inf")))
        hit_rate = metrics.get("hit_rate", 0.0)
        
        try:
            sep = float(sep)
        except (TypeError, ValueError):
            return False
        
        if math.isnan(sep) or math.isinf(sep):
            return False
        
        return sep <= sep_threshold and hit_rate >= hit_rate_threshold
    
    def update_episode(self, metrics: Dict[str, float], reward: float) -> None:
        """更新episode计数和统计信息"""
        self.episodes_in_phase += 1
        
        if self.is_feasible(metrics):
            if not self.phase_feasible_count:
                self.phase_feasible_count = [0] * (self.current_phase + 1)
            while len(self.phase_feasible_count) <= self.current_phase:
                self.phase_feasible_count.append(0)
            self.phase_feasible_count[self.current_phase] += 1
    
    def should_advance_phase(self) -> bool:
        """判断是否应该进入下一个阶段"""
        if self.current_phase >= self.num_phases - 1:
            return False
        
        # 条件1：当前阶段已运行足够的episode
        if self.episodes_in_phase < self.phase_episodes:
            return False
        
        # 条件2：当前阶段找到至少1个可行解
        if not self.phase_feasible_count or self.phase_feasible_count[self.current_phase] < 1:
            return False
        
        return True
    
    def advance_phase(self) -> None:
        """进入下一个阶段"""
        if self.current_phase < self.num_phases - 1:
            sep_threshold, hit_rate_threshold = self.get_current_thresholds()
            logger.info(
                f"[FRE] Phase {self.current_phase} completed "
                f"({self.episodes_in_phase} episodes, "
                f"{self.phase_feasible_count[self.current_phase] if self.phase_feasible_count else 0} feasible). "
                f"Advancing to phase {self.current_phase + 1}..."
            )
            
            self.current_phase += 1
            self.episodes_in_phase = 0
            self.phase_transition_episodes.append(
                sum(self.phase_feasible_count) if self.phase_feasible_count else 0
            )
            
            new_sep_threshold, new_hit_rate_threshold = self.get_current_thresholds()
            logger.info(
                f"[FRE] Phase {self.current_phase} constraints: "
                f"SEP <= {new_sep_threshold:.1f}m, HitRate >= {new_hit_rate_threshold:.1f}%"
            )
    
    def get_exploration_bonus(self, metrics: Dict[str, float]) -> float:
        """
        获取探索奖励加成。
        在当前阶段的可行域内，给予额外奖励以鼓励探索。
        """
        if not self.is_feasible(metrics):
            return 0.0
        
        # 在可行域内，根据离约束边界的距离给予奖励
        sep_threshold, hit_rate_threshold = self.get_current_thresholds()
        
        sep = metrics.get("SEP", metrics.get("miss_distance", float("inf")))
        hit_rate = metrics.get("hit_rate", 0.0)
        
        try:
            sep = float(sep)
        except (TypeError, ValueError):
            return 0.0
        
        if math.isnan(sep) or math.isinf(sep):
            return 0.0
        
        # SEP离约束边界越远，奖励越大
        sep_margin = (sep_threshold - sep) / sep_threshold
        sep_bonus = max(0.0, sep_margin * 0.5)
        
        # 命中率离约束边界越远，奖励越大
        hit_rate_margin = (hit_rate - hit_rate_threshold) / max(hit_rate_threshold, 1.0)
        hit_rate_bonus = max(0.0, hit_rate_margin * 0.3)
        
        return sep_bonus + hit_rate_bonus
    
    def get_phase_progress(self) -> Dict[str, Any]:
        """获取当前阶段的进度信息"""
        sep_threshold, hit_rate_threshold = self.get_current_thresholds()
        feasible_count = (
            self.phase_feasible_count[self.current_phase]
            if self.phase_feasible_count and self.current_phase < len(self.phase_feasible_count)
            else 0
        )
        
        return {
            "current_phase": self.current_phase,
            "total_phases": self.num_phases,
            "episodes_in_phase": self.episodes_in_phase,
            "phase_episodes": self.phase_episodes,
            "sep_threshold": sep_threshold,
            "hit_rate_threshold": hit_rate_threshold,
            "feasible_count": feasible_count,
            "progress_percent": (self.episodes_in_phase / self.phase_episodes) * 100,
        }
    
    def log_phase_status(self) -> None:
        """记录当前阶段的状态"""
        progress = self.get_phase_progress()
        logger.info(
            f"[FRE] Phase {progress['current_phase']}/{progress['total_phases']-1} | "
            f"Episodes: {progress['episodes_in_phase']}/{progress['phase_episodes']} | "
            f"Feasible: {progress['feasible_count']} | "
            f"SEP <= {progress['sep_threshold']:.1f}m, HitRate >= {progress['hit_rate_threshold']:.1f}%"
        )


class FeasibleRegionRewardModifier:
    """修改奖励函数以支持逐步探索"""
    
    def __init__(self, explorer: FeasibleRegionExplorer):
        self.explorer = explorer
    
    def modify_reward(
        self,
        base_reward: float,
        metrics: Dict[str, float],
    ) -> float:
        """
        修改奖励函数，加入可行域探索的奖励加成
        
        Args:
            base_reward: 基础奖励（来自标准奖励函数）
            metrics: 仿真指标
        
        Returns:
            修改后的奖励
        """
        # 获取探索奖励加成
        exploration_bonus = self.explorer.get_exploration_bonus(metrics)
        
        # 修改后的奖励 = 基础奖励 + 探索加成
        modified_reward = base_reward + exploration_bonus
        
        return modified_reward
