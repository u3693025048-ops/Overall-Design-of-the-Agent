#!/usr/bin/env python3
"""Generate paper-aligned Monte Carlo initial-condition datasets."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np
from scipy.io import savemat
from tqdm import tqdm

logger = logging.getLogger(__name__)

MISSILE_PARAMS: Dict[str, float] = {
    "mass": 231.0,
    "length": 3.66,
    "diameter": 0.203,
    "S_ref": 0.03236,
    "I_xx": 3.0,
    "I_yy": 220.0,
    "I_zz": 220.0,
    "thrust": 5000.0,
    "V_nominal": 1200.0,
}

INITIAL_CONDITIONS: Dict[str, Dict[str, float]] = {
    "missile": {
        "x": 0.0,
        "alt": 9000.0,
        "z": -3000.0,
        "gamma": 0.0,
        "psi": 0.0,
        "theta": 0.0,
        "phi": 0.0,
        "psi_body": 0.0,
    },
    "target": {
        "x": 28000.0,
        "alt": 3000.0,
        "z": 10000.0,
        "Vx": 250.0,
        "Vy": 50.0,
        "Az": 80.0,
        "omega_z": 0.20,
    },
}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "T1": {"Az": 0.0, "Vy": 0.0, "omega_z": 0.20, "dispersion": {"Vx": 10.0}},
    "T2": {"Az": 25.0, "Vy": 12.0, "omega_z": 0.40, "dispersion": {"omega_z": 0.05}},
    "T3": {"Az": 55.0, "Vy": 28.0, "omega_z": 0.75, "dispersion": {"Az": 5.0}},
    "T4": {"Az": 85.0, "Vy": 45.0, "omega_z": 1.20, "dispersion": {"Az": 5.0}},
    "T1_broadside": {
        "Az": 0.0,
        "Vy": 0.0,
        "omega_z": 0.20,
        "dispersion": {"Vx": 10.0},
        "overrides": {"target": {"z": 15000.0}},
    },
}

MASTER_SEEDS = list(range(42, 52))
BROADSIDE_MASTER_SEEDS = list(range(42, 47))
DEV_OFFSETS = (1, 2, 3)
TEST_OFFSETS = (4, 5, 6)
SIM_CONFIG: Dict[str, Any] = {"dt": 0.001, "t_max": 70.0, "integrator": "RK4"}

_MISSILE_DTYPE = np.dtype(
    [
        ("x", "O"),
        ("alt", "O"),
        ("z", "O"),
        ("V", "O"),
        ("gamma", "O"),
        ("psi", "O"),
        ("theta", "O"),
        ("phi", "O"),
        ("psi_body", "O"),
    ]
)
_TARGET_DTYPE = np.dtype(
    [
        ("x", "O"),
        ("alt", "O"),
        ("z", "O"),
        ("Vx", "O"),
        ("Vy", "O"),
        ("Az_realized", "O"),
        ("omega_z_realized", "O"),
    ]
)
_SIM_DTYPE = np.dtype([("dt", "O"), ("t_max", "O"), ("integrator", "O")])


def _deep_update(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> None:
    """Recursively merge *overrides* into *base* in place."""
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _scenario_initial_conditions(scenario_cfg: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """Return base initial conditions after scenario constants and overrides."""
    conditions = deepcopy(INITIAL_CONDITIONS)
    target = conditions["target"]
    target["Az"] = float(scenario_cfg["Az"])
    target["Vy"] = float(scenario_cfg["Vy"])
    target["omega_z"] = float(scenario_cfg["omega_z"])
    overrides = scenario_cfg.get("overrides") or {}
    if overrides:
        _deep_update(conditions, overrides)
    return conditions


def _realized_target(
    base_target: Mapping[str, float],
    scenario_cfg: Mapping[str, Any],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Sample one target realization using scenario 1-sigma dispersions."""
    target = {
        "x": float(base_target["x"]),
        "alt": float(base_target["alt"]),
        "z": float(base_target["z"]),
        "Vx": float(base_target["Vx"]),
        "Vy": float(scenario_cfg["Vy"]),
        "Az_realized": float(scenario_cfg["Az"]),
        "omega_z_realized": float(scenario_cfg["omega_z"]),
    }
    dispersion = scenario_cfg.get("dispersion") or {}
    for key, sigma in dispersion.items():
        sample = rng.normal(0.0, float(sigma))
        if key == "Az":
            target["Az_realized"] = float(scenario_cfg["Az"]) + float(sample)
        elif key == "omega_z":
            target["omega_z_realized"] = float(scenario_cfg["omega_z"]) + float(sample)
        elif key == "Vx":
            target["Vx"] = float(base_target["Vx"]) + float(sample)
        elif key == "Vy":
            target["Vy"] = float(scenario_cfg["Vy"]) + float(sample)
        else:
            logger.warning("Ignoring unsupported dispersion key: %s", key)
    return target


def _build_struct_arrays(
    scenario: str,
    split_seed: int,
    n_runs: int,
) -> Dict[str, np.ndarray]:
    """Build MATLAB-compatible struct arrays for one scenario and split seed.

    Args:
        scenario: Scenario key in :data:`SCENARIOS`.
        split_seed: Deterministic dev/test seed used by ``default_rng``.
        n_runs: Number of Monte Carlo runs in this file.

    Returns:
        Dictionary ready for :func:`scipy.io.savemat`.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    scenario_cfg = SCENARIOS[scenario]
    conditions = _scenario_initial_conditions(scenario_cfg)
    missile = conditions["missile"]
    target = conditions["target"]
    rng = np.random.default_rng(split_seed)

    missile_init = np.empty((1, n_runs), dtype=_MISSILE_DTYPE)
    target_init = np.empty((1, n_runs), dtype=_TARGET_DTYPE)
    sim_config = np.empty((1, n_runs), dtype=_SIM_DTYPE)

    for idx in range(n_runs):
        missile_init[0, idx]["x"] = float(missile["x"])
        missile_init[0, idx]["alt"] = float(missile["alt"])
        missile_init[0, idx]["z"] = float(missile["z"])
        missile_init[0, idx]["V"] = float(MISSILE_PARAMS["V_nominal"])
        missile_init[0, idx]["gamma"] = float(missile["gamma"])
        missile_init[0, idx]["psi"] = float(missile["psi"])
        missile_init[0, idx]["theta"] = float(missile["theta"])
        missile_init[0, idx]["phi"] = float(missile["phi"])
        missile_init[0, idx]["psi_body"] = float(missile.get("psi_body", 0.0))

        realized = _realized_target(target, scenario_cfg, rng)
        for field, value in realized.items():
            target_init[0, idx][field] = float(value)

        sim_config[0, idx]["dt"] = float(SIM_CONFIG["dt"])
        sim_config[0, idx]["t_max"] = float(SIM_CONFIG["t_max"])
        sim_config[0, idx]["integrator"] = SIM_CONFIG["integrator"]

    missile_params = np.array(
        [(float(MISSILE_PARAMS["mass"]), float(MISSILE_PARAMS["length"]),
          float(MISSILE_PARAMS["diameter"]), float(MISSILE_PARAMS["S_ref"]),
          float(MISSILE_PARAMS["I_xx"]), float(MISSILE_PARAMS["I_yy"]),
          float(MISSILE_PARAMS["I_zz"]), float(MISSILE_PARAMS["thrust"]),
          float(MISSILE_PARAMS["V_nominal"]))],
        dtype=[
            ("mass", "O"),
            ("length", "O"),
            ("diameter", "O"),
            ("S_ref", "O"),
            ("I_xx", "O"),
            ("I_yy", "O"),
            ("I_zz", "O"),
            ("thrust", "O"),
            ("V_nominal", "O"),
        ],
    )

    return {
        "missile_init": missile_init,
        "target_init": target_init,
        "sim_config": sim_config,
        "missile_params": missile_params,
        "scenario": scenario,
        "split_seed": int(split_seed),
        "n_runs": int(n_runs),
    }


def generate_mc_file(
    scenario: str,
    output_path: str | Path,
    split_seed: int,
    n_runs: int = 100,
) -> Path:
    """Generate one MATLAB ``.mat`` MC dataset.

    Args:
        scenario: Scenario key, e.g. ``"T1"`` or ``"T1_broadside"``.
        output_path: Destination ``.mat`` path.
        split_seed: Seed used for all randomized target dispersions.
        n_runs: Number of MC runs to store.

    Returns:
        The written output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_struct_arrays(scenario, split_seed, n_runs)
    savemat(str(path), payload, do_compression=True)
    logger.info("Wrote MC data: %s", path)
    return path


def generate_all_mc_data(
    output_root: str | Path = "data/mc",
    n_runs: int = 100,
    master_seeds: Optional[Iterable[int]] = None,
    include_broadside: bool = True,
) -> Dict[str, Any]:
    """Generate all configured dev/test MC datasets.

    Args:
        output_root: Root directory for ``data/mc`` style outputs.
        n_runs: Number of MC runs per generated ``.mat`` file.
        master_seeds: Master seeds for T1-T4. Defaults to 42 through 51.
        include_broadside: Whether to generate ``T1_broadside`` seeds 42 through 46.

    Returns:
        Seed map dictionary also written to ``seed_map.json``.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    seeds = list(master_seeds or MASTER_SEEDS)

    scenario_seeds: Dict[str, list[int]] = {
        name: seeds for name in ("T1", "T2", "T3", "T4")
    }
    if include_broadside:
        scenario_seeds["T1_broadside"] = BROADSIDE_MASTER_SEEDS

    seed_map: Dict[str, Any] = {
        "master_seeds": seeds,
        "broadside_master_seeds": BROADSIDE_MASTER_SEEDS if include_broadside else [],
        "n_runs": int(n_runs),
        "seed_rule": {
            "dev_seeds": "master_seed * 100 + {1, 2, 3}",
            "test_seeds": "master_seed * 100 + {4, 5, 6}",
            "legacy_aliases": "seed_<master>_dev.mat and seed_<master>_test.mat duplicate the first dev/test split",
        },
        "scenarios": {},
    }

    total_files = sum(
        len(s) * (len(DEV_OFFSETS) + len(TEST_OFFSETS) + 2)
        for s in scenario_seeds.values()
    )
    progress = tqdm(total=total_files, desc="Generating MC data", unit="file")
    for scenario, scenario_master_seeds in scenario_seeds.items():
        seed_map["scenarios"][scenario] = {}
        for master_seed in scenario_master_seeds:
            scenario_dir = root / scenario
            dev_entries = []
            test_entries = []

            for offset in DEV_OFFSETS:
                dev_seed = master_seed * 100 + offset
                dev_path = scenario_dir / f"seed_{dev_seed}_dev.mat"
                generate_mc_file(scenario, dev_path, dev_seed, n_runs=n_runs)
                progress.update(1)
                dev_entries.append({
                    "seed": dev_seed,
                    "path": str(dev_path.as_posix()),
                })

            for offset in TEST_OFFSETS:
                test_seed = master_seed * 100 + offset
                test_path = scenario_dir / f"seed_{test_seed}_test.mat"
                generate_mc_file(scenario, test_path, test_seed, n_runs=n_runs)
                progress.update(1)
                test_entries.append({
                    "seed": test_seed,
                    "path": str(test_path.as_posix()),
                })

            legacy_dev_path = scenario_dir / f"seed_{master_seed}_dev.mat"
            legacy_test_path = scenario_dir / f"seed_{master_seed}_test.mat"
            generate_mc_file(scenario, legacy_dev_path, dev_entries[0]["seed"], n_runs=n_runs)
            progress.update(1)
            generate_mc_file(scenario, legacy_test_path, test_entries[0]["seed"], n_runs=n_runs)
            progress.update(1)

            seed_map["scenarios"][scenario][str(master_seed)] = {
                "dev": dev_entries,
                "test": test_entries,
                "legacy_dev_path": str(legacy_dev_path.as_posix()),
                "legacy_test_path": str(legacy_test_path.as_posix()),
            }
    progress.close()

    seed_map_path = root / "seed_map.json"
    seed_map_path.write_text(json.dumps(seed_map, indent=2), encoding="utf-8")
    logger.info("Wrote seed map: %s", seed_map_path)
    return seed_map


def main() -> None:
    """Run the MC data generator from the command line."""
    parser = argparse.ArgumentParser(description="Generate GSMultiAgent MC datasets.")
    parser.add_argument("--output-root", default="data/mc", help="Output root directory.")
    parser.add_argument("--n-runs", type=int, default=100, help="MC runs per .mat file.")
    parser.add_argument(
        "--no-broadside",
        action="store_true",
        help="Skip T1_broadside supplemental geometry datasets.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    generate_all_mc_data(
        output_root=args.output_root,
        n_runs=args.n_runs,
        include_broadside=not args.no_broadside,
    )


if __name__ == "__main__":
    main()
