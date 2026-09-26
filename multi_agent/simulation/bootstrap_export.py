"""Export real-MC seed results for remote Ubuntu bootstrap / contract fill.

Local MATLAB still runs the first AND seed (and may fill locally for live
control flow). In parallel we dump export JSON under
``guidance_output/bootstrap_export/outbox`` so Ubuntu can re-sample the
remaining seeds and run contract checks offline. After the experiment,
``scripts/bootstrap_remote_pipeline.py`` pulls Ubuntu artifacts back into
``results/experiment_plan/bootstrap_remote/``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_ROOT = ROOT / "guidance_output" / "bootstrap_export"
DEFAULT_LOCAL_IMPORT = ROOT / "results" / "experiment_plan" / "bootstrap_remote"


def bootstrap_export_enabled(default: bool = True) -> bool:
    env = os.environ.get("GSM_BOOTSTRAP_EXPORT", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    try:
        from multi_agent.config_loader import get_config

        return bool(getattr(get_config().simulation, "bootstrap_export", default))
    except Exception:
        return bool(default)


def export_root() -> Path:
    raw = os.environ.get("GSM_BOOTSTRAP_EXPORT_DIR", "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else ROOT / p
    try:
        from multi_agent.config_loader import get_config

        cfg = str(
            getattr(get_config().simulation, "bootstrap_export_dir", "") or ""
        ).strip()
        if cfg:
            p = Path(cfg)
            return p if p.is_absolute() else ROOT / p
    except Exception:
        pass
    return DEFAULT_EXPORT_ROOT


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    try:
        return float(obj)
    except Exception:
        return str(obj)


def write_bootstrap_export(
    *,
    scenario: str,
    split: str,
    source_seed: int,
    target_seeds: Sequence[int],
    n_runs: int,
    kill_radius: float,
    base_result: Dict[str, Any],
    matlab_script_path: str = "",
    dp_params: Optional[Dict[str, float]] = None,
    local_aggregate: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Write one export JSON for remote bootstrap processing."""
    if not bootstrap_export_enabled(default=True):
        return None
    if not target_seeds:
        return None
    root = export_root()
    outbox = root / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    export_id = (
        f"{scenario}_{split}_{int(source_seed)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    payload = {
        "schema": "gsm_bootstrap_export_v1",
        "export_id": export_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario": str(scenario),
        "split": str(split),
        "source_seed": int(source_seed),
        "target_seeds": [int(s) for s in target_seeds],
        "n_runs": int(n_runs),
        "kill_radius": float(kill_radius),
        "matlab_script_path": str(matlab_script_path or ""),
        "dp_params": _jsonable(dp_params or {}),
        "base_result": _jsonable(base_result),
        "local_aggregate": _jsonable(local_aggregate or {}),
    }
    path = outbox / f"{export_id}.export.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    logger.info("[bootstrap_export] wrote %s", path)
    return path


def aggregate_seed_results(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """AND-style aggregate matching run_mc_and_across_seeds pooling."""
    hit_vals: List[float] = []
    sep_vals: List[float] = []
    peak_vals: List[float] = []
    pm_vals: List[float] = []
    bw_vals: List[float] = []
    all_pass = True
    sim_ok = True
    for item in seed_results:
        ran = bool(item.get("success", False))
        feasible = bool(item.get("feasible", False))
        sim_ok = sim_ok and ran
        all_pass = all_pass and ran and feasible
        try:
            hit_vals.append(float(item.get("hit_rate", 0.0)))
        except (TypeError, ValueError):
            hit_vals.append(0.0)
        try:
            sep_vals.append(float(item.get("SEP", 1e9)))
        except (TypeError, ValueError):
            sep_vals.append(1e9)
        try:
            peak_vals.append(
                float(item.get("peak_nz", item.get("peak_ny", item.get("peak_n", 1e9))))
            )
        except (TypeError, ValueError):
            peak_vals.append(1e9)
        try:
            pm_vals.append(float(item.get("PM", item.get("pitch_PM", 0.0))))
        except (TypeError, ValueError):
            pm_vals.append(0.0)
        try:
            bw_vals.append(float(item.get("BW", item.get("pitch_BW", 0.0))))
        except (TypeError, ValueError):
            bw_vals.append(0.0)
    return {
        "hit_rate": min(hit_vals) if hit_vals else 0.0,
        "SEP": max(sep_vals) if sep_vals else 1e9,
        "peak_nz": max(peak_vals) if peak_vals else 1e9,
        "peak_ny": max(peak_vals) if peak_vals else 1e9,
        "PM": min(pm_vals) if pm_vals else 0.0,
        "BW": min(bw_vals) if bw_vals else 0.0,
        "feasible": bool(all_pass),
        "success": bool(sim_ok),
        "and_logic": True,
        "constraint_violations": {} if all_pass else {"and_logic": 1.0},
    }


def process_export_file(path: Path, *, out_dir: Path) -> Path:
    """Bootstrap + contract-fill one export JSON → filled artifact."""
    from multi_agent.simulation.mc_seed_bootstrap import bootstrap_seed_result
    from multi_agent.metric_contract import constraint_violations

    data = json.loads(path.read_text(encoding="utf-8"))
    source_seed = int(data["source_seed"])
    split = str(data.get("split") or "dev")
    kill_radius = float(data.get("kill_radius") or 10.0)
    n_runs = int(data.get("n_runs") or 100)
    base = dict(data.get("base_result") or {})
    targets = [int(s) for s in (data.get("target_seeds") or [])]

    seed_results: List[Dict[str, Any]] = [dict(base, synthetic=False, mc_seed=source_seed)]
    for target in targets:
        seed_results.append(
            bootstrap_seed_result(
                base,
                target_seed=target,
                source_seed=source_seed,
                split=split,
                kill_radius=kill_radius,
                n_runs=n_runs,
            )
        )

    # Per-seed contract re-check (explicit, for remote audit)
    contracts = []
    for item in seed_results:
        try:
            ok, viol = constraint_violations(
                float(item.get("hit_rate", 0.0)),
                float(item.get("SEP", 1e9)),
                float(item.get("peak_nz", item.get("peak_ny", 1e9))),
                float(item.get("PM", 0.0)),
                float(item.get("BW", 0.0)),
            )
        except Exception as exc:
            ok, viol = False, {"contract_error": 1.0, "detail": str(exc)}
        contracts.append(
            {
                "mc_seed": item.get("mc_seed"),
                "synthetic": bool(item.get("synthetic")),
                "feasible": bool(ok),
                "violations": viol,
            }
        )
        item["feasible"] = bool(ok)
        item["constraint_violations"] = viol

    agg = aggregate_seed_results(seed_results)
    filled = {
        "schema": "gsm_bootstrap_filled_v1",
        "export_id": data.get("export_id"),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "source_export": path.name,
        "scenario": data.get("scenario"),
        "split": split,
        "source_seed": source_seed,
        "target_seeds": targets,
        "seed_results": seed_results,
        "contracts": contracts,
        "aggregate": agg,
        "dp_params": data.get("dp_params") or {},
        "local_aggregate": data.get("local_aggregate") or {},
        "host": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{data.get('export_id', path.stem)}.filled.json"
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(filled, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)

    # Compact CSV row for spreadsheet merge
    csv_path = out_dir / "filled_summary.csv"
    row = {
        "export_id": filled.get("export_id"),
        "scenario": filled.get("scenario"),
        "split": split,
        "source_seed": source_seed,
        "feasible": agg.get("feasible"),
        "hit_rate": agg.get("hit_rate"),
        "SEP": agg.get("SEP"),
        "peak_nz": agg.get("peak_nz"),
        "PM": agg.get("PM"),
        "BW": agg.get("BW"),
        "n_synthetic": sum(1 for r in seed_results if r.get("synthetic")),
        "processed_at": filled.get("processed_at"),
    }
    import csv

    write_header = not csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return out_path


def process_export_directory(
    inbox: Path,
    out_dir: Path,
    *,
    move_done: bool = True,
) -> Dict[str, Any]:
    """Process all ``*.export.json`` under inbox."""
    inbox = Path(inbox)
    out_dir = Path(out_dir)
    done_dir = inbox.parent / "done"
    if move_done:
        done_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(inbox.glob("*.export.json"))
    written: List[str] = []
    errors: List[str] = []
    t0 = time.perf_counter()
    for path in files:
        try:
            out = process_export_file(path, out_dir=out_dir)
            written.append(str(out))
            if move_done:
                dest = done_dir / path.name
                path.replace(dest)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            logger.exception("[bootstrap_export] failed %s", path)
    return {
        "processed": len(written),
        "errors": errors,
        "outputs": written,
        "wall_sec": round(time.perf_counter() - t0, 3),
    }
