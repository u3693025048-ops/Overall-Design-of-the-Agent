#!/usr/bin/env python3
"""Statistical analysis utilities for GSMultiAgent experiments."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

logger = logging.getLogger(__name__)


def load_results(csv_path: str) -> pd.DataFrame:
    """Load an experiment CSV and normalize common column types.

    Args:
        csv_path: Path to a runner-produced CSV.

    Returns:
        Cleaned ``DataFrame`` with numeric metrics and boolean flags.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(path)
    for col in ("error", "first_success", "feasible_dev", "feasible_test"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    numeric_cols = [
        "seed",
        "mc_calls_dev",
        "mc_calls_test",
        "wall_clock_sec",
        "inference_sec",
        "orchestration_sec",
        "sim_sec",
        "iterations_used",
        "hit_rate_dev",
        "SEP_dev",
        "peak_nz_dev",
        "PM_dev",
        "BW_dev",
        "hit_rate_test",
        "SEP_test",
        "peak_nz_test",
        "PM_test",
        "BW_test",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "mc_calls" not in df.columns and {"mc_calls_dev", "mc_calls_test"}.issubset(df.columns):
        df["mc_calls"] = df["mc_calls_dev"].fillna(0) + df["mc_calls_test"].fillna(0)
    return df


def _mean_std(group: pd.DataFrame, col: str) -> tuple[float, float]:
    """Return finite mean/std for a group column."""
    vals = pd.to_numeric(group[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(vals.mean()), float(vals.std(ddof=1))


def _binomial_ci(k: int, n: int) -> tuple[float, float]:
    """Return exact binomial confidence interval for a count."""
    if n <= 0:
        return float("nan"), float("nan")
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95)
    return float(ci.low), float(ci.high)


def _design_path_mode(paths: Iterable[object]) -> str:
    """Return a compact mode/count string for design paths."""
    series = pd.Series([str(p) for p in paths if str(p) and str(p) != "nan"])
    if series.empty:
        return "N/A"
    counts = series.value_counts()
    return ", ".join(f"{idx} ({int(count)}/{int(counts.sum())})" for idx, count in counts.items())


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by method and scenario.

    Args:
        df: Loaded experiment dataframe.

    Returns:
        Summary dataframe with means, standard deviations, binomial confidence
        intervals, and design-path modes.
    """
    if "mc_calls" not in df.columns and {"mc_calls_dev", "mc_calls_test"}.issubset(df.columns):
        df = df.copy()
        df["mc_calls"] = df["mc_calls_dev"].fillna(0) + df["mc_calls_test"].fillna(0)
    rows = []
    for (method, scenario), group in df.groupby(["method", "scenario"], dropna=False):
        n = int(len(group))
        k = int(group.get("first_success", pd.Series(dtype=bool)).sum())
        ci_low, ci_high = _binomial_ci(k, n)
        row = {
            "method": method,
            "scenario": scenario,
            "n": n,
            "first_success_k": k,
            "first_success_n": n,
            "first_success_rate": k / n if n else float("nan"),
            "first_success_ci_lower": ci_low,
            "first_success_ci_upper": ci_high,
            "design_path_mode": _design_path_mode(group.get("design_path", [])),
        }
        for col in (
            "mc_calls",
            "hit_rate_test",
            "SEP_test",
            "peak_nz_test",
            "PM_test",
            "BW_test",
            "wall_clock_sec",
            "iterations_used",
        ):
            if col in group.columns:
                mean, std = _mean_std(group, col)
                row[f"{col}_mean"] = mean
                row[f"{col}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(paired_diffs, n_boot: int = 10000, alpha: float = 0.05):
    """Bootstrap confidence interval for paired differences.

    Args:
        paired_diffs: Paired metric differences.
        n_boot: Number of bootstrap samples.
        alpha: Tail probability.

    Returns:
        Tuple ``(lower, upper)``.
    """
    diffs = np.asarray(paired_diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        sample = rng.choice(diffs, size=diffs.size, replace=True)
        means[idx] = float(np.mean(sample))
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def cohens_d(paired_diffs):
    """Cohen's d for paired differences.

    Args:
        paired_diffs: Paired metric differences.

    Returns:
        Paired Cohen's d.
    """
    diffs = np.asarray(paired_diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    std = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
    if std <= 1e-12:
        return 0.0
    return float(np.mean(diffs) / std)


def pairwise_test(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str,
    scenarios: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Run paired Wilcoxon signed-rank tests by scenario.

    Args:
        df: Loaded experiment dataframe.
        method_a: First method name.
        method_b: Second method name.
        metric: Metric column to compare.
        scenarios: Optional scenario filter.

    Returns:
        DataFrame with scenario-level effect sizes and p-values.
    """
    rows = []
    scenario_values = list(scenarios) if scenarios else sorted(df["scenario"].dropna().unique())
    for scenario in scenario_values:
        sub = df[df["scenario"] == scenario]
        a = sub[sub["method"] == method_a][["seed", metric]].rename(columns={metric: "a"})
        b = sub[sub["method"] == method_b][["seed", metric]].rename(columns={metric: "b"})
        paired = a.merge(b, on="seed", how="inner").replace([np.inf, -np.inf], np.nan).dropna()
        if paired.empty:
            continue
        diffs = paired["a"].to_numpy(dtype=float) - paired["b"].to_numpy(dtype=float)
        if len(diffs) < 2 or np.allclose(diffs, 0.0):
            p_value = 1.0
        else:
            try:
                p_value = float(wilcoxon(paired["a"], paired["b"]).pvalue)
            except ValueError:
                p_value = 1.0
        ci_lower, ci_upper = bootstrap_ci(diffs)
        rows.append(
            {
                "scenario": scenario,
                "mean_a": float(paired["a"].mean()),
                "mean_b": float(paired["b"].mean()),
                "mean_diff": float(np.mean(diffs)),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "cohens_d": cohens_d(diffs),
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def _fmt_mean_std(row: pd.Series, stem: str, digits: int = 2) -> str:
    """Format mean ± std from summary row."""
    mean = row.get(f"{stem}_mean", np.nan)
    std = row.get(f"{stem}_std", np.nan)
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _write_latex(summary_df: pd.DataFrame, output_path: str | Path, caption: str) -> None:
    """Write a compact LaTeX table."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in summary_df.iterrows():
        rate = row.get("first_success_rate", np.nan)
        ci_l = row.get("first_success_ci_lower", np.nan)
        ci_u = row.get("first_success_ci_upper", np.nan)
        first = "--" if pd.isna(rate) else f"{rate:.2f} [{ci_l:.2f},{ci_u:.2f}]"
        rows.append(
            {
                "Method": row["method"],
                "Scenario": row["scenario"],
                "MC Calls": _fmt_mean_std(row, "mc_calls", 1),
                "First Success": first,
                "Hit Rate": _fmt_mean_std(row, "hit_rate_test", 1),
                "SEP": _fmt_mean_std(row, "SEP_test", 2),
                "PeakNz": _fmt_mean_std(row, "peak_nz_test", 2),
                "PM": _fmt_mean_std(row, "PM_test", 1),
                "BW": _fmt_mean_std(row, "BW_test", 1),
                "Time": _fmt_mean_std(row, "wall_clock_sec", 1),
                "Iter": _fmt_mean_std(row, "iterations_used", 1),
                "Path": row.get("design_path_mode", "N/A"),
            }
        )
    table_df = pd.DataFrame(rows)
    latex = table_df.to_latex(index=False, escape=False, caption=caption)
    path.write_text(latex, encoding="utf-8")
    logger.info("Wrote LaTeX table: %s", path)


def generate_latex_table_5(summary_df: pd.DataFrame, output_path):
    """Generate the main experiment LaTeX table."""
    _write_latex(summary_df, output_path, "Main experiment results.")


def generate_latex_table_6(summary_df: pd.DataFrame, output_path):
    """Generate the memory ablation LaTeX table."""
    _write_latex(summary_df, output_path, "Memory ablation results.")


def generate_latex_table_7(summary_df: pd.DataFrame, output_path):
    """Generate the optimization ablation LaTeX table."""
    _write_latex(summary_df, output_path, "Optimization ablation results.")


def parse_convergence_log(value: object) -> list[tuple[float, float]]:
    """Parse a convergence-log cell from CSV or pass through list values."""
    if isinstance(value, list):
        return [(float(a), float(b)) for a, b in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    out = []
    for item in parsed or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((float(item[0]), float(item[1])))
    return out


__all__ = [
    "load_results",
    "compute_summary",
    "pairwise_test",
    "bootstrap_ci",
    "cohens_d",
    "generate_latex_table_5",
    "generate_latex_table_6",
    "generate_latex_table_7",
    "parse_convergence_log",
]
