#!/usr/bin/env python3
"""
Feasible Region Explorer v2 (FRE v2) - 增强版

核心改进：
1. 多目标约束：明确定义每个阶段的硬约束和软约束
2. 异常处理与回滚：检测极端异常值，自动回滚到上一阶段
3. 自适应超参数：根据阶段进度动态调整学习率和探索率
4. 智能阶段管理：提前终止、失败回滚、最大轮次保护
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhaseConstraints:
    """单个阶段的约束定义"""
    phase_id: int
    sep_max: float              # SEP硬约束上限
    hit_rate_min: float         # 命中率硬约束下限
    peak_ny_max: float          # PeakNy硬约束上限
    pm_min: float               # PM硬约束下限
    pm_max: float               # PM硬约束上限
    bw_min: float               # BW硬约束下限
    bw_max: float               # BW硬约束上限
    
    # 异常检测阈值（硬约束的倍数）
    anomaly_threshold: float = 2.0  # 超过硬约束2倍以上为异常
    
    def is_feasible(self, metrics: Dict[str, float]) -> bool:
        """检查是否满足所有硬约束"""
        from multi_agent.rl.metric_utils import get_peak_ny

        sep = metrics.get("SEP", metrics.get("miss_distance", float("inf")))
        hit_rate = metrics.get("hit_rate", 0.0)
        peak_ny = get_peak_ny(metrics)
        pm = metrics.get("pitch_PM", 0.0)
        bw = metrics.get("pitch_BW", 0.0)
        
        try:
            sep = float(sep)
        except (TypeError, ValueError):
            return False
        
        if math.isnan(sep) or math.isinf(sep):
            return False
        
        return (
            sep <= self.sep_max and
            hit_rate >= self.hit_rate_min and
            peak_ny <= self.peak_ny_max and
            self.pm_min <= pm <= self.pm_max and
            self.bw_min <= bw <= self.bw_max
        )
    
    def is_anomalous(self, metrics: Dict[str, float]) -> bool:
        """检测极端异常值（超过硬约束的倍数）"""
        sep = metrics.get("SEP", metrics.get("miss_distance", 0.0))
        from multi_agent.rl.metric_utils import get_peak_ny
        peak_ny = get_peak_ny(metrics)
        pm = metrics.get("pitch_PM", 0.0)
        bw = metrics.get("pitch_BW", 0.0)
        
        try:
            sep = float(sep)
            peak_ny = float(peak_ny)
            pm = float(pm)
            bw = float(bw)
        except (TypeError, ValueError):
            return False
        
        # 检查是否超过硬约束的倍数
        if sep > self.sep_max * self.anomaly_threshold:
            logger.warning(f"[FRE] 异常检测：SEP={sep:.2f}m 超过阈值 {self.sep_max * self.anomaly_threshold:.2f}m")
            return True
        
        if peak_ny > self.peak_ny_max * self.anomaly_threshold:
            logger.warning(f"[FRE] 异常检测：PeakNy={peak_ny:.2f}g 超过阈值 {self.peak_ny_max * self.anomaly_threshold:.2f}g")
            return True
        
        if pm < self.pm_min * 0.5 or pm > self.pm_max * 2.0:
            logger.warning(f"[FRE] 异常检测：PM={pm:.2f}° 超过范围 [{self.pm_min * 0.5:.2f}, {self.pm_max * 2.0:.2f}]")
            return True
        
        if bw > self.bw_max * self.anomaly_threshold:
            logger.warning(f"[FRE] 异常检测：BW={bw:.2f} rad/s 超过阈值 {self.bw_max * self.anomaly_threshold:.2f}")
            return True
        
        return False


@dataclass
class HyperParameters:
    """阶段超参数"""
    phase_id: int
    lr_actor: float = 1e-4      # Actor学习率
    lr_critic: float = 5e-4     # Critic学习率
    entropy_coef: float = 0.05  # 熵系数
    exploration_std: float = 0.37  # 探索标准差
    
    def decay(self, progress: float) -> 'HyperParameters':
        """根据阶段进度衰减超参数"""
        # progress: 0.0 (开始) → 1.0 (结束)
        decay_factor = 1.0 - progress * 0.5  # 最多衰减50%
        
        return HyperParameters(
            phase_id=self.phase_id,
            lr_actor=self.lr_actor * decay_factor,
            lr_critic=self.lr_critic * decay_factor,
            entropy_coef=self.entropy_coef * decay_factor,
            exploration_std=self.exploration_std * decay_factor,
        )


class FeasibleRegionExplorerV2:
    """增强版可行域探索器"""
    
    def __init__(
        self,
        phase_constraints: List[PhaseConstraints],
        phase_hyperparams: Optional[List[HyperParameters]] = None,
        early_stop_window: int = 5,           # 提前终止窗口（连续5个episodes）
        failure_threshold: int = 10,          # 失败回滚阈值（连续10个episodes）
        max_phase_episodes: int = 30,         # 每个阶段最大轮次
    ):
        """
        Args:
            phase_constraints: 各阶段的约束定义
            phase_hyperparams: 各阶段的超参数（可选，自动生成）
            early_stop_window: 提前终止的连续episodes数
            failure_threshold: 失败回滚的连续episodes数
            max_phase_episodes: 每个阶段的最大轮次
        """
        self.phase_constraints = phase_constraints
        self.num_phases = len(phase_constraints)
        self.current_phase = 0
        
        # 初始化超参数
        if phase_hyperparams is None:
            self.phase_hyperparams = [
                HyperParameters(phase_id=i) for i in range(self.num_phases)
            ]
        else:
            self.phase_hyperparams = phase_hyperparams
        
        # 阶段管理参数
        self.early_stop_window = early_stop_window
        self.failure_threshold = failure_threshold
        self.max_phase_episodes = max_phase_episodes
        
        # 统计信息
        self.episodes_in_phase = 0
        self.phase_best_reward = -float("inf")
        self.phase_best_metrics: Dict[str, float] = {}
        self.phase_best_params: Dict[str, float] = {}
        
        # 历史记录
        self.episode_metrics_history: List[Dict[str, float]] = []
        self.phase_transition_log: List[Dict[str, Any]] = []
        self.rollback_log: List[Dict[str, Any]] = []
        
        # 失败计数
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        logger.info(
            f"[FRE v2] 初始化 | 阶段数={self.num_phases} | "
            f"提前终止窗口={early_stop_window} | "
            f"失败回滚阈值={failure_threshold} | "
            f"最大轮次={max_phase_episodes}"
        )
        
        for i, constraint in enumerate(phase_constraints):
            logger.info(
                f"[FRE v2] Phase {i}: "
                f"SEP<={constraint.sep_max:.1f}m, "
                f"HR>={constraint.hit_rate_min:.1f}%, "
                f"PeakNy<={constraint.peak_ny_max:.1f}g, "
                f"PM∈[{constraint.pm_min:.1f}, {constraint.pm_max:.1f}]°, "
                f"BW∈[{constraint.bw_min:.1f}, {constraint.bw_max:.1f}]"
            )
    
    def get_current_constraints(self) -> PhaseConstraints:
        """获取当前阶段的约束"""
        return self.phase_constraints[self.current_phase]
    
    def get_current_hyperparams(self) -> HyperParameters:
        """获取当前阶段的超参数（考虑进度衰减）"""
        base_params = self.phase_hyperparams[self.current_phase]
        progress = self.episodes_in_phase / self.max_phase_episodes
        return base_params.decay(progress)
    
    def update_episode(
        self,
        metrics: Dict[str, float],
        reward: float,
        params: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        更新episode信息，返回阶段管理决策
        
        Returns:
            {
                'action': 'continue' | 'advance' | 'rollback',
                'reason': str,
                'should_stop': bool,
            }
        """
        self.episodes_in_phase += 1
        self.episode_metrics_history.append(metrics)
        
        constraints = self.get_current_constraints()
        
        # 检查异常
        if constraints.is_anomalous(metrics):
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            logger.warning(
                f"[FRE v2] Phase {self.current_phase} Ep {self.episodes_in_phase}: "
                f"检测到异常值 | 连续失败 {self.consecutive_failures}/{self.failure_threshold}"
            )
            
            # 失败回滚条件
            if self.consecutive_failures >= self.failure_threshold:
                return self._rollback("连续失败超过阈值")
            
            return {'action': 'continue', 'reason': '异常值，继续尝试', 'should_stop': False}
        
        # 检查可行性
        is_feasible = constraints.is_feasible(metrics)
        
        if is_feasible:
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            
            # 更新阶段最优
            if reward > self.phase_best_reward:
                self.phase_best_reward = reward
                self.phase_best_metrics = metrics.copy()
                self.phase_best_params = params.copy()
            
            logger.info(
                f"[FRE v2] Phase {self.current_phase} Ep {self.episodes_in_phase}: "
                f"可行解 ✅ | 连续成功 {self.consecutive_successes}/{self.early_stop_window}"
            )
            
            # 提前终止条件
            if self.consecutive_successes >= self.early_stop_window:
                return self._advance("连续成功，提前进入下一阶段")
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            logger.info(
                f"[FRE v2] Phase {self.current_phase} Ep {self.episodes_in_phase}: "
                f"不可行 ❌ | 连续失败 {self.consecutive_failures}/{self.failure_threshold}"
            )
            
            # 失败回滚条件
            if self.consecutive_failures >= self.failure_threshold:
                return self._rollback("连续失败超过阈值")
        
        # 最大轮次保护
        if self.episodes_in_phase >= self.max_phase_episodes:
            if self.consecutive_successes >= self.early_stop_window:
                return self._advance("达到最大轮次，进入下一阶段")
            else:
                return self._rollback("达到最大轮次但未充分成功")
        
        return {'action': 'continue', 'reason': '继续当前阶段', 'should_stop': False}
    
    def _advance(self, reason: str) -> Dict[str, Any]:
        """进入下一阶段"""
        if self.current_phase < self.num_phases - 1:
            self.phase_transition_log.append({
                'from_phase': self.current_phase,
                'to_phase': self.current_phase + 1,
                'episodes': self.episodes_in_phase,
                'reason': reason,
                'best_reward': self.phase_best_reward,
            })
            
            logger.info(
                f"[FRE v2] Phase {self.current_phase} → {self.current_phase + 1} | "
                f"Episodes: {self.episodes_in_phase} | Reason: {reason}"
            )
            
            self.current_phase += 1
            self._reset_phase_state()
            
            return {
                'action': 'advance',
                'reason': reason,
                'should_stop': False,
            }
        else:
            logger.info(f"[FRE v2] 所有阶段完成！")
            return {
                'action': 'continue',
                'reason': '已完成所有阶段',
                'should_stop': True,
            }
    
    def _rollback(self, reason: str) -> Dict[str, Any]:
        """回滚到上一阶段"""
        if self.current_phase > 0:
            self.rollback_log.append({
                'from_phase': self.current_phase,
                'to_phase': self.current_phase - 1,
                'episodes': self.episodes_in_phase,
                'reason': reason,
            })
            
            logger.warning(
                f"[FRE v2] Phase {self.current_phase} 回滚到 {self.current_phase - 1} | "
                f"Episodes: {self.episodes_in_phase} | Reason: {reason}"
            )
            
            self.current_phase -= 1
            self._reset_phase_state()
            
            return {
                'action': 'rollback',
                'reason': reason,
                'should_stop': False,
            }
        else:
            logger.error(f"[FRE v2] 无法回滚：已在第一阶段")
            return {
                'action': 'continue',
                'reason': '无法回滚',
                'should_stop': False,
            }
    
    def _reset_phase_state(self):
        """重置阶段状态"""
        self.episodes_in_phase = 0
        self.phase_best_reward = -float("inf")
        self.phase_best_metrics = {}
        self.phase_best_params = {}
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.episode_metrics_history = []
    
    def get_phase_progress(self) -> Dict[str, Any]:
        """获取当前阶段的进度"""
        constraints = self.get_current_constraints()
        hyperparams = self.get_current_hyperparams()
        
        return {
            'current_phase': self.current_phase,
            'total_phases': self.num_phases,
            'episodes_in_phase': self.episodes_in_phase,
            'max_phase_episodes': self.max_phase_episodes,
            'progress_percent': (self.episodes_in_phase / self.max_phase_episodes) * 100,
            'consecutive_successes': self.consecutive_successes,
            'consecutive_failures': self.consecutive_failures,
            'phase_best_reward': self.phase_best_reward,
            'constraints': {
                'sep_max': constraints.sep_max,
                'hit_rate_min': constraints.hit_rate_min,
                'peak_ny_max': constraints.peak_ny_max,
                'pm_range': [constraints.pm_min, constraints.pm_max],
                'bw_range': [constraints.bw_min, constraints.bw_max],
            },
            'hyperparams': {
                'lr_actor': hyperparams.lr_actor,
                'lr_critic': hyperparams.lr_critic,
                'entropy_coef': hyperparams.entropy_coef,
                'exploration_std': hyperparams.exploration_std,
            },
        }
    
    def log_phase_status(self):
        """记录当前阶段状态"""
        progress = self.get_phase_progress()
        logger.info(
            f"[FRE v2] Phase {progress['current_phase']}/{progress['total_phases']-1} | "
            f"Episodes: {progress['episodes_in_phase']}/{progress['max_phase_episodes']} | "
            f"Success: {progress['consecutive_successes']}/{self.early_stop_window} | "
            f"Failures: {progress['consecutive_failures']}/{self.failure_threshold} | "
            f"Progress: {progress['progress_percent']:.1f}%"
        )


class FeasibleRegionRewardModifierV2:
    """增强版奖励修改器"""
    
    def __init__(self, explorer: FeasibleRegionExplorerV2):
        self.explorer = explorer
    
    def modify_reward(
        self,
        base_reward: float,
        metrics: Dict[str, float],
    ) -> float:
        """
        修改奖励，加入多目标约束的奖励加成
        
        Args:
            base_reward: 基础奖励
            metrics: 仿真指标
        
        Returns:
            修改后的奖励
        """
        constraints = self.explorer.get_current_constraints()
        
        # 基础奖励
        reward = base_reward
        
        # 多目标约束奖励加成
        from multi_agent.rl.metric_utils import get_peak_ny

        sep = metrics.get("SEP", metrics.get("miss_distance", float("inf")))
        hit_rate = metrics.get("hit_rate", 0.0)
        peak_ny = get_peak_ny(metrics)
        pm = metrics.get("pitch_PM", 0.0)
        bw = metrics.get("pitch_BW", 0.0)
        
        try:
            sep = float(sep)
        except (TypeError, ValueError):
            sep = float("inf")
        
        # SEP奖励
        if sep <= constraints.sep_max:
            sep_bonus = 1.0 * (1.0 - sep / constraints.sep_max)
            reward += sep_bonus
        
        # 命中率奖励
        if hit_rate >= constraints.hit_rate_min:
            hr_bonus = 0.5 * ((hit_rate - constraints.hit_rate_min) / (100.0 - constraints.hit_rate_min))
            reward += hr_bonus
        
        # PeakNy奖励
        if peak_ny <= constraints.peak_ny_max:
            pny_bonus = 0.5 * (1.0 - peak_ny / constraints.peak_ny_max)
            reward += pny_bonus
        
        # PM奖励
        pm_mid = (constraints.pm_min + constraints.pm_max) / 2.0
        pm_range = (constraints.pm_max - constraints.pm_min) / 2.0
        if constraints.pm_min <= pm <= constraints.pm_max:
            pm_bonus = 0.3 * (1.0 - abs(pm - pm_mid) / pm_range)
            reward += pm_bonus
        
        # BW奖励
        bw_mid = (constraints.bw_min + constraints.bw_max) / 2.0
        bw_range = (constraints.bw_max - constraints.bw_min) / 2.0
        if constraints.bw_min <= bw <= constraints.bw_max:
            bw_bonus = 0.3 * (1.0 - abs(bw - bw_mid) / bw_range)
            reward += bw_bonus
        
        return reward
