#!/usr/bin/env python3
"""Equal Monte Carlo budget shared by GSMultiAgent and all baselines.

Reviewer #1 (round 2) required every method to be evaluated with the same
maximum number of Monte Carlo calls.  The previous protocol counted CMA-ES
*generations* against the same integer used as Random Search / Direct PPO
*candidate evaluations*, so CMA-ES received λ× more 100-run batches.

This module is the single source of truth:

    max_candidates = workflow.max_iterations × matlab_rl_optimizer.max_episodes
                   = 5 × 15 = 75
    max_mc_calls   = max_candidates × n_dev_seeds
                   = 75 × 3 = 225

One candidate evaluation always costs ``n_dev_seeds`` MC calls (AND-logic over
the three development seeds). With ``mc_tiered_eval``, search uses 10
trajectories per call and the held-out test uses 100. CMA-ES may complete
fewer than λ offspring in the final generation; leftover slots are evaluated
but do not update the covariance (truncated generation).
"""

from __future__ import annotations

from typing import Tuple

# Mirrors config.yaml: workflow.max_iterations and matlab_rl_optimizer.max_episodes.
GSM_OUTER_ITERATIONS = 5
GSM_MAX_EPISODES = 15
N_DEV_SEEDS = 3

DEFAULT_MAX_CANDIDATES = GSM_OUTER_ITERATIONS * GSM_MAX_EPISODES  # 75
DEFAULT_MAX_MC_CALLS = DEFAULT_MAX_CANDIDATES * N_DEV_SEEDS  # 225


def max_mc_calls(n_dev_seeds: int = N_DEV_SEEDS, max_candidates: int = DEFAULT_MAX_CANDIDATES) -> int:
    """Return the shared development MC-call ceiling."""
    return int(max_candidates) * max(1, int(n_dev_seeds))


def remaining_candidates(mc_calls_used: int, max_mc: int, n_dev_seeds: int) -> int:
    """How many further candidate evaluations fit in the remaining MC budget."""
    leftover = max(0, int(max_mc) - int(mc_calls_used))
    n_seeds = max(1, int(n_dev_seeds))
    return leftover // n_seeds


def split_agent_budget(
    max_candidates: int,
    n_outer: int = GSM_OUTER_ITERATIONS,
) -> Tuple[int, int]:
    """Split a candidate ceiling into (outer iterations, episodes per outer).

    Agent methods (GSMultiAgent, Single-Agent Hermes) have an outer loop.
    The product is at most ``max_candidates``, matching the algorithmic
    baselines' candidate count.
    """
    outer = max(1, int(n_outer))
    episodes = max(1, int(max_candidates) // outer)
    return outer, episodes
