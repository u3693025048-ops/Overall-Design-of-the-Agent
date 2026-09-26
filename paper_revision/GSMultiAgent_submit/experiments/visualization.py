#!/usr/bin/env python3
"""Visualization helpers for GSMultiAgent experiment results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_agent.metric_contract import BW_MAX, BW_MIN, PEAK_NY_MAX, PM_MAX, PM_MIN, SEP_MAX
import seaborn as sns

from experiments.analysis import parse_convergence_log

logger = logging.getLogger(__name__)


def _finite(series: pd.Series) -> pd.Series:
    """Replace infinities with NaN for plotting."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def plot_figure_11(summary_df: pd.DataFrame, output_dir):
    """Plot the four-panel Figure 11 summary.

    Args:
        summary_df: DataFrame returned by ``compute_summary``.
        output_dir: Directory for ``figure_11.png`` and ``figure_11.pdf``.

    Returns:
        Path to the PNG file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = summary_df.copy()
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    sns.barplot(data=df, x="method", y="wall_clock_sec_mean", hue="scenario", ax=ax)
    ax.set_title("(a) Total design time")
    ax.set_xlabel("")
    ax.set_ylabel("Seconds")

    ax = axes[0, 1]
    sns.barplot(data=df, x="method", y="iterations_used_mean", hue="scenario", ax=ax)
    ax.set_title("(b) Iteration rounds")
    ax.set_xlabel("")
    ax.set_ylabel("Iterations")

    ax = axes[1, 0]
    sns.scatterplot(
        data=df,
        x="mc_calls_mean",
        y="iterations_used_mean",
        hue="method",
        style="scenario",
        s=70,
        ax=ax,
    )
    ax.set_title("(c) Global Pareto frontier")
    ax.set_xlabel("MC calls")
    ax.set_ylabel("Iterations")

    ax = axes[1, 1]
    sns.barplot(data=df, x="method", y="first_success_rate", hue="scenario", ax=ax)
    ax.set_title("(d) First-attempt success rate")
    ax.set_xlabel("")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=25)
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("")
    fig.tight_layout()
    png = out_dir / "figure_11.png"
    pdf = out_dir / "figure_11.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote Figure 11: %s", png)
    return png


def plot_convergence_curves(results_by_method, scenario, output_path):
    """Plot CMA-ES / PPO convergence curves.

    Args:
        results_by_method: Mapping from method to logs, or DataFrame with
            ``method``, ``scenario``, and ``convergence_log`` columns.
        scenario: Scenario filter.
        output_path: Destination image path.

    Returns:
        Output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7, 4))
    if isinstance(results_by_method, pd.DataFrame):
        data = results_by_method[results_by_method["scenario"] == scenario]
        iterator = []
        for method, group in data.groupby("method"):
            logs = []
            for value in group.get("convergence_log", []):
                logs.extend(parse_convergence_log(value))
            iterator.append((method, logs))
    else:
        iterator = list(results_by_method.items())
    for method, log in iterator:
        if isinstance(log, pd.DataFrame):
            points = list(zip(log["eval_num"], log["fitness"]))
        else:
            points = parse_convergence_log(str(log)) if isinstance(log, str) else list(log or [])
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=str(method))
    ax.set_title(f"Convergence - {scenario}")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Fitness / reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _radar_values(metrics: Mapping[str, Any]) -> list[float]:
    """Normalize five constraint dimensions to 0..1 for radar plotting."""
    hit = min(max(float(metrics.get("hit_rate_test", metrics.get("hit_rate", 0.0))) / 100.0, 0.0), 1.0)
    sep = 1.0 / (1.0 + max(float(metrics.get("SEP_test", metrics.get("SEP", 1e9))), 0.0) / SEP_MAX)
    peak = 1.0 / (
        1.0
        + max(float(metrics.get("peak_nz_test", metrics.get("peak_nz", 1e9))), 0.0)
        / PEAK_NY_MAX
    )
    pm = float(metrics.get("PM_test", metrics.get("PM", 0.0)))
    pm_mid = 0.5 * (PM_MIN + PM_MAX)
    pm_half = max(0.5 * (PM_MAX - PM_MIN), 1.0)
    pm_score = 1.0 if PM_MIN <= pm <= PM_MAX else max(0.0, 1.0 - abs(pm - pm_mid) / pm_half)
    bw = float(metrics.get("BW_test", metrics.get("BW", 0.0)))
    bw_mid = 0.5 * (BW_MIN + BW_MAX)
    bw_half = max(0.5 * (BW_MAX - BW_MIN), 1.0)
    bw_score = 1.0 if BW_MIN <= bw <= BW_MAX else max(0.0, 1.0 - abs(bw - bw_mid) / bw_half)
    return [hit, sep, peak, pm_score, bw_score]


def plot_constraint_radar(results_by_method, scenario, output_path):
    """Plot a five-dimensional constraint radar chart.

    Args:
        results_by_method: Mapping from method to metric dictionaries, or a
            DataFrame containing test metrics.
        scenario: Scenario filter.
        output_path: Destination image path.

    Returns:
        Output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Hit", "SEP", "PeakNz", "PM", "BW"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
    if isinstance(results_by_method, pd.DataFrame):
        iterator = []
        sub = results_by_method[results_by_method["scenario"] == scenario]
        for method, group in sub.groupby("method"):
            finite = group.replace([np.inf, -np.inf], np.nan)
            metrics = {
                "hit_rate_test": finite["hit_rate_test"].mean(),
                "SEP_test": finite["SEP_test"].mean(),
                "peak_nz_test": finite["peak_nz_test"].mean(),
                "PM_test": finite["PM_test"].mean(),
                "BW_test": finite["BW_test"].mean(),
            }
            iterator.append((method, metrics))
    else:
        iterator = list(results_by_method.items())
    for method, metrics in iterator:
        values = _radar_values(metrics)
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=str(method))
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Constraint radar - {scenario}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = [
    "plot_figure_11",
    "plot_convergence_curves",
    "plot_constraint_radar",
]
