#!/usr/bin/env python3
"""Build a journal supplementary ZIP: no GitHub, no secrets, no synthetic CSVs."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMIT_DIR = ROOT / "paper_revision" / "GSMultiAgent_submit"
OUT = ROOT / "paper_revision" / "GSMultiAgent_supplement.zip"
STAGE = SUBMIT_DIR
OLD_STAGE = ROOT / "paper_revision" / "_supplement_stage"

INCLUDE_TOP = (
    "cli_agent.py",
    "run_rl_optimize.py",
    "requirements.txt",
    "prompt.txt",
    "REPRODUCIBILITY.md",
    "SUPPLEMENT.md",
    "experiments",
    "multi_agent",
    "knowledge_base",
    "tests",
)

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "chroma_db",
    "DATA",
    "logs",
    "guidance_output",
    "parameter_experience_base",
    "outOfDate",
    "output-0811",
    "output-0812",
    "output-0813",
    "新建文件夹",
    "results",
    "collected_bundle_20260808_103453",
    "collected_bundle_20260809_220450",
    "collected_bundle_20260810_075014",
    "simulated",
}

EXCLUDE_FILE_PATTERNS = (
    re.compile(r"incl_simulated", re.I),
    re.compile(r"all_simulated", re.I),
    re.compile(r"\.gsm_ssh_pass$"),
    re.compile(r"\.env$"),
    re.compile(r"credentials", re.I),
)

KEY_LINE = re.compile(
    r"^(\s*(?:api_key|embedding_api_key)\s*:\s*).+$",
    re.I,
)
KEY_COMMENT = re.compile(r"sk-[A-Za-z0-9._\-]+")
# Avoid matching English "task-specific" / "task-requirement".
LEAKED_KEY = re.compile(r"(?<![A-Za-z])sk-(?!dummy\b)[A-Za-z0-9._\-]{8,}")

MC_SCENARIOS = ("T1", "T2", "T3", "T4")
MC_MASTERS = (42, 43, 44)
MC_OFFSETS = (1, 2, 3, 4, 5, 6)


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_DIR_NAMES for part in rel.parts):
        return True
    name = path.name
    if name.endswith((".pyc", ".pyo")):
        return True
    return any(pat.search(name) or pat.search(str(rel).replace("\\", "/")) for pat in EXCLUDE_FILE_PATTERNS)


def sanitize_config_text(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        m = KEY_LINE.match(line.rstrip("\n"))
        if m:
            nl = "\n" if line.endswith("\n") else ""
            out.append(f"{m.group(1)}YOUR_API_KEY{nl}")
            continue
        out.append(KEY_COMMENT.sub("YOUR_API_KEY", line))
    return "".join(out)


def copy_tree(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.name == "config.yaml":
            dst.write_text(sanitize_config_text(src.read_text(encoding="utf-8")), encoding="utf-8")
            return
        shutil.copy2(src, dst)
        return
    for child in src.rglob("*"):
        if not child.is_file() or should_skip(child):
            continue
        rel = child.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if child.name == "config.yaml":
            target.write_text(sanitize_config_text(child.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(child, target)


def _load_mc_generator():
    path = ROOT / "experiments" / "mc_data_generator.py"
    spec = importlib.util.spec_from_file_location("mc_data_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def copy_or_generate_mc(stage: Path) -> int:
    """Copy existing T1-T4 / 42-44 .mat files, or generate them if absent."""
    src_roots = (ROOT / "data" / "mc", ROOT / "outOfDate" / "DATA" / "mc")
    dst_root = stage / "data" / "mc"
    dst_root.mkdir(parents=True, exist_ok=True)
    generator = None
    n = 0
    seed_map = {"n_runs": 100, "scenarios": {}}
    for scenario in MC_SCENARIOS:
        seed_map["scenarios"][scenario] = {}
        for master in MC_MASTERS:
            dev_entries = []
            test_entries = []
            for offset in MC_OFFSETS:
                split = "dev" if offset <= 3 else "test"
                file_seed = master * 100 + offset
                name = f"seed_{file_seed}_{split}.mat"
                src = next((r / scenario / name for r in src_roots if (r / scenario / name).is_file()), None)
                dst = dst_root / scenario / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src is not None:
                    shutil.copy2(src, dst)
                else:
                    if generator is None:
                        generator = _load_mc_generator()
                    generator.generate_mc_file(scenario, dst, file_seed, n_runs=100)
                n += 1
                entry = {"seed": file_seed, "path": dst.relative_to(stage).as_posix()}
                if split == "dev":
                    dev_entries.append(entry)
                else:
                    test_entries.append(entry)
            seed_map["scenarios"][scenario][str(master)] = {
                "dev": dev_entries,
                "test": test_entries,
            }
    (dst_root / "seed_map.json").write_text(
        json.dumps(seed_map, indent=2), encoding="utf-8"
    )
    return n


def _is_simulated_row(row: dict) -> bool:
    return str(row.get("_is_simulated", "")).lower() in {"1", "true", "yes"}


def _write_filtered_csv(src: Path, dst: Path) -> None:
    with src.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys()) if rows else []
    if not fieldnames:
        return
    real = [r for r in rows if not _is_simulated_row(r)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(real)


def copy_real_result_csvs(stage: Path) -> None:
    """Copy main-table CSVs from experiment_plan root, dropping simulated rows."""
    plan = ROOT / "results" / "experiment_plan"
    if not plan.is_dir():
        return
    dest = stage / "results" / "experiment_plan"
    dest.mkdir(parents=True, exist_ok=True)

    src_plain = plan / "gsm_all_seeds.csv"
    src_comb = plan / "gsm_all_seeds_incl_simulated.csv"
    src = src_plain if src_plain.is_file() else src_comb
    if src.is_file():
        _write_filtered_csv(src, dest / "gsm_all_seeds.csv")

    for csv_path in plan.glob("*.csv"):
        name = csv_path.name.lower()
        if "incl_simulated" in name or "all_simulated" in name:
            continue
        if csv_path.name == "gsm_all_seeds.csv":
            continue
        # Per-seed / machine shards still have pre-repair iteration labels.
        # Canonical GSM rows are gsm_all_seeds.csv only.
        if csv_path.name.startswith("ablation_"):
            continue
        _write_filtered_csv(csv_path, dest / csv_path.name)


EQUAL_MC_DIRS = (
    ROOT / "results" / "experiment_plan" / "optimizer_equal_mc_75",
    ROOT / "results" / "experiment_plan" / "optimizer_equal_mc_75_sah_n10",
)
SKIP_EQUAL_MC_PARTS = {"_overbudget_backup_20260821_172528"}


def copy_equal_mc_dirs(stage: Path) -> int:
    """Copy equal-budget run JSON/CSV used for Table 5 (cap 225)."""
    n = 0
    for src in EQUAL_MC_DIRS:
        if not src.is_dir():
            continue
        dest = stage / src.relative_to(ROOT)
        for child in src.rglob("*"):
            if not child.is_file():
                continue
            rel = child.relative_to(src)
            if any(part in SKIP_EQUAL_MC_PARTS or part.startswith("_overbudget") for part in rel.parts):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
            n += 1
    plan = ROOT / "results" / "experiment_plan"
    extra_names = ("table5_equal_mc.csv", "table6_equal_mc.csv", "first_success_mc_calls.csv")
    dest_plan = stage / "results" / "experiment_plan"
    dest_plan.mkdir(parents=True, exist_ok=True)
    for name in extra_names:
        src = plan / name
        if src.is_file():
            shutil.copy2(src, dest_plan / name)
            n += 1
    for src in plan.glob("*.xlsx"):
        if "before_equal_mc" in src.name:
            continue
        shutil.copy2(src, dest_plan / src.name)
        n += 1
    tables_dir = ROOT / "results" / "tables"
    for name in ("table5_equal_mc.csv", "table5_equal_mc.tex", "table6_equal_mc.csv", "table6_equal_mc.tex", "table7_sah_n10.csv"):
        src = tables_dir / name
        if src.is_file():
            out = stage / "results" / "tables" / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            n += 1
    return n


def zip_stage(stage: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in stage.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(stage).as_posix())


def assert_no_secrets(stage: Path) -> None:
    leaks = []
    for path in stage.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".mat", ".png", ".jpg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if LEAKED_KEY.search(text):
            leaks.append(str(path.relative_to(stage)))
    if leaks:
        raise SystemExit("secret-like tokens remain in: " + ", ".join(leaks[:20]))


SUBMIT_NOTE = """# 本文件夹用途

这是可随修订稿上传的投稿包（已脱敏）。不要把整个工作区仓库直接打包。

期刊系统若只收 ZIP，使用同目录上一级的 `GSMultiAgent_supplement.zip`（由本文件夹压缩而成）。

## 抽样生成代码不要删

`experiments/mc_data_generator.py` 必须保留。交战条件 `.mat` 来自已发表的麻雀级参数和固定种子，不是实装遥测；把生成器交给审稿人，才能按同样种子重生数据。删掉生成器并不能让试验数据变得可以全部提交，反而无法说明数据从何而来。

## 能提交的数据

- 协议内 MC 初值：`data/mc/`（T1–T4，主种子 42/43/44 的 dev/test `.mat`）
- GSM 全部真实运行一行一张表：`results/experiment_plan/gsm_all_seeds.csv`
- 主表对应的真实运行汇总 CSV：`results/experiment_plan/`（已去掉 `_is_simulated=True` 的行与 ablation 分片）
- 等预算基线与对齐口径单智能体：`results/experiment_plan/optimizer_equal_mc_75/` 与 `optimizer_equal_mc_75_sah_n10/`（搜索 10 条；不含旧 100 条目录与超预算备份）
- 去密钥配置：`config.yaml`（`mc_and_bootstrap_fill: false`）

## 不能、也不该整包提交的数据

- 完整试验过程：`DATA/run_archives/`、`guidance_output/`、`logs/`、`chroma_db/`（体积大，且日志里有密钥）
- 运行期 bootstrap 补种、事后合成 CSV（`*_incl_simulated.csv`、`simulated/`）
- 本机运维脚本 `scripts/`（含密钥，未纳入本文件夹）

主文表格只用真实 MATLAB 评估行。完整弹道中间件无法也不需要全部上传；审稿人可用本包代码与 `.mat` 按协议复现。
"""


def main() -> None:
    if OLD_STAGE.exists():
        shutil.rmtree(OLD_STAGE)
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for name in INCLUDE_TOP:
        src = ROOT / name
        if src.exists():
            copy_tree(src, STAGE / name)

    cfg = ROOT / "config.yaml"
    if cfg.is_file():
        text = sanitize_config_text(cfg.read_text(encoding="utf-8"))
        text = text.replace("mc_and_bootstrap_fill: true", "mc_and_bootstrap_fill: false")
        (STAGE / "config.yaml").write_text(text, encoding="utf-8")

    (STAGE / "提交说明.md").write_text(SUBMIT_NOTE, encoding="utf-8")
    pack_list = ROOT / "paper_revision" / "打包文件清单.md"
    if pack_list.is_file():
        shutil.copy2(pack_list, STAGE / "打包文件清单.md")

    n_mat = copy_or_generate_mc(STAGE)
    copy_real_result_csvs(STAGE)
    n_eq = copy_equal_mc_dirs(STAGE)
    assert_no_secrets(STAGE)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    zip_stage(STAGE, OUT)
    n_files = sum(1 for p in STAGE.rglob("*") if p.is_file())
    print(f"folder: {STAGE}")
    print(f"mc files: {n_mat}")
    print(f"equal-mc files: {n_eq}")
    print(f"staged files: {n_files}")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
